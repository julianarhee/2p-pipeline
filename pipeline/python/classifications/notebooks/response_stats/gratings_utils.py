
import os
import glob
import json
import copy
import traceback
import dill as pkl
import numpy as np
import pandas as pd
import scipy.stats as spstats

import py3utils as p3

def get_fit_desc(response_type='dff', responsive_test=None, 
                 responsive_thr=10, n_stds=2.5,
                 n_bootstrap_iters=1000, n_resamples=20):
    '''
    Set standardized naming scheme for ori_fit_desc
    '''
    if responsive_test is None:
        fit_desc = 'fit-%s_all-cells_boot-%i-resample-%i' \
                        % (response_type, n_bootstrap_iters, n_resamples) #, responsive_test, responsive_thr)
    elif responsive_test == 'nstds':
        fit_desc = 'fit-%s_responsive-%s-%.2f-thr%.2f_boot-%i-resample-%i' \
                        % (response_type, responsive_test, n_stds, responsive_thr, n_bootstrap_iters, n_resamples)
    else:
        fit_desc = 'fit-%s_responsive-%s-thr%.2f_boot-%i-resample-%i' \
                        % (response_type, responsive_test, responsive_thr, n_bootstrap_iters, n_resamples)
    return fit_desc

def get_ori_dir(datakey, traceid='traces001', fit_desc=None, 
                rootdir='/n/coxfs01/2p-data'):
    ori_dir=None
    session, animalid, fovnum = p3.split_datakey_str(datakey)
    ori_dir = glob.glob(os.path.join(rootdir, animalid, session, 'FOV%i_*' % fovnum,
                         'combined_gratings_static', 'traces', '%s*' % traceid,
                         'tuning', fit_desc))
    assert len(ori_dir)==1, \
            "[%s]: Ambiguous dir for fit <%s>:\n%s" % (datakey, fit_desc, str(ori_dir))

    return ori_dir[0]

def load_tuning_results(datakey, run_name='gratings', traceid='traces001',
                        fit_desc=None, rootdir='/n/coxfs01/2p-data', verbose=True):

    bootresults=None; fitparams=None;
    try: 
        ori_dir = get_ori_dir(datakey, traceid=traceid, fit_desc=fit_desc)
        results_fpath = os.path.join(ori_dir, 'fitresults.pkl')
        params_fpath = os.path.join(ori_dir, 'fitparams.json')
        # open 
        with open(results_fpath, 'rb') as f:
            bootresults = pkl.load(f, encoding='latin1')
        with open(params_fpath, 'r') as f:
            fitparams = json.load(f, encoding='latin1')
    except Exception as e:
        print("[ERROR]: NO FITS FOUND: %s" % ori_dir)
        traceback.print_exc() 
                        
    return bootresults, fitparams


# Fitting
def interp_values(response_vector, n_intervals=3, as_series=False, wrap_value=None, wrap=True):
    resps_interp = []
    rvectors = copy.copy(response_vector)
    if wrap_value is None and wrap is True:
        wrap_value = response_vector[0]
    if wrap:
        rvectors = np.append(response_vector, wrap_value)

    for orix, rvec in enumerate(rvectors[0:-1]):
        if rvec == rvectors[-2]:
            resps_interp.extend(np.linspace(rvec, rvectors[orix+1], endpoint=True, num=n_intervals+1))
        else:
            resps_interp.extend(np.linspace(rvec, rvectors[orix+1], endpoint=False, num=n_intervals))    
    
    if as_series:
        return pd.Series(resps_interp, name=response_vector.name)
    else:
        return resps_interp

def angdir180(x):
    '''wraps anguar diff values to interval 0, 180'''
    return min(np.abs([x, x-360, x+360]))

def double_gaussian( x, c1, c2, mu, sigma, C ):
    #(c1, c2, mu, sigma) = params
    x1vals = np.array([angdir180(xi - mu) for xi in x])
    x2vals = np.array([angdir180(xi - mu - 180 ) for xi in x])
    res =   C + c1 * np.exp( -(x1vals**2.0) / (2.0 * sigma**2.0) ) + c2 * np.exp( -(x2vals**2.0) / (2.0 * sigma**2.0) )

#    res =   C + c1 * np.exp( - ((x - mu) % 360.)**2.0 / (2.0 * sigma**2.0) ) \
#            + c2 * np.exp( - ((x + 180 - mu) % 360.)**2.0 / (2.0 * sigma**2.0) )

#        res =   c1 * np.exp( - (x - mu)**2.0 / (2.0 * sigma**2.0) ) \
#                #+ c2 * np.exp( - (x - mu2)**2.0 / (2.0 * sigma2**2.0) )
    return res


# Evaluation
def average_metrics_across_iters(fitdf):
    ''' 
    Average bootstrapped params to get an "average" set of params.
    '''
    means = {}
    roi = int(fitdf['cell'].unique()[0])
    #print("COLS:", fitdf.columns)
    for param in fitdf.columns:
        if 'stim' in param:
            meanval = fitdf[param].values[0]
        elif 'theta' in param:
            # meanval = np.rad2deg(spstats.circmean(np.deg2rad(fitdf[param] % 360.)))
            # Use Median, since could have double-peaks
            #meanval = fitdf[param].median() 
            cnts, bns = np.histogram(fitdf[param] % 360., 
                            bins=np.linspace(0, 360., 50))
            meanval = float(bns[np.where(cnts==max(cnts))[0][0]])
        else:
            meanval = fitdf[param].mean()
        means[param] = meanval
    
    return pd.DataFrame(means, index=[roi])


def coeff_determination(origr, fitr):
    residuals = origr - fitr
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((origr - np.mean(origr))**2)
    r2 = 1 - (ss_res / ss_tot)
    
    return r2, residuals
       
def evaluate_fits(bootr, interp=False):
    '''
    Create an averaged param set (from bootstrapped iters). 
    Fit tuning curve, calculate R2 and goodness-of-fit.
    '''
    # Average fit parameters aross boot iters
    params = [c for c in bootr['results'].columns if 'stim' not in c]
    avg_metrics = average_metrics_across_iters(bootr['results'][params])

    # Get mean response (deal with offset)    
    orig_ = bootr['data']['responses'].mean(axis=0)
    #orig_data = np.abs(orig_ - np.mean(orig_)) 
    orig_data = (orig_ - orig_.min()) #- (orig_ - orig_.mean()).min()

    # Get combined r2 between original and avg-fit
    if interp:
        origr = interp_values(orig_data)
        thetas = bootr['fits']['xv']
    else:
        origr = orig_data #bootr['data']['responses'].mean(axis=0).values
        thetas = bootr['data']['tested_values']

    # Fit to average, evaluate fit   
    cpopt = tuple(avg_metrics[['response_pref', 'response_null', 'theta_pref', 'sigma', 'response_offset']].values[0])
    fitr = double_gaussian( thetas, *cpopt)
    r2_comb, _ = coeff_determination(origr, fitr) 
    # Get Goodness-of-fit
    iqr = spstats.iqr(bootr['results']['r2'])
    gfit = np.mean(bootr['results']['r2']) * (1-iqr) * np.sqrt(r2_comb)
    
    return r2_comb, gfit, fitr
    


def get_good_fits(bootresults, fitparams, gof_thr=0.66, verbose=True):
   
    best_rfits=None; all_rfits=None; 
    niters = fitparams['n_bootstrap_iters']
    interp = fitparams['n_intervals_interp']>1
    passrois = sorted([k for k, v in bootresults.items() if any(v.values())])
           
    all_dfs = []; best_dfs=[];
    goodrois = []
    for roi in passrois: 
        fitresults = []
        stimkeys = []
        for stimparam, bootr in bootresults[roi].items():
            if bootr['fits'] is None:
                continue
            # Evaluate current fits from bootstrapped results
            r2comb, gof, fitr = evaluate_fits(bootr, interp=interp)
            if np.isnan(gof) or (gof_thr is not None and (gof < gof_thr)):
                continue            
            rfdf = bootr['results'] # All 1000 iterations
            rfdf['r2comb'] = [r2comb for _ in range(niters)] # add combined R2 val
            rfdf['gof'] = [gof for _ in range(niters)] # add GoF metric            
            rfdf['sf'] = float(stimparam[0])
            rfdf['size'] = float(stimparam[1])
            rfdf['speed'] = float(stimparam[2])

            tmpd = average_metrics_across_iters(rfdf) # Average current roi, current condition results
            stimkey = 'sf-%.2f-sz-%i-sp-%i' % stimparam 
            fitresults.append(tmpd)
            stimkeys.append(stimkey)
 
        if len(fitresults) > 0:
            roif = pd.concat(fitresults, axis=0).reset_index(drop=True)
            roif.index = stimkeys
            gof =  roif.mean()['gof']  
            if (gof_thr is not None) and (gof >= gof_thr): #0.66:
                goodrois.append(roi) # This is just for repoorting 
            # Select the "best" condition, so each cell has 1 fit
            if gof_thr is not None:
                best_cond_df = pd.DataFrame(roif.sort_values(by='r2comb').iloc[-1]).T
            else:
                best_cond_df = pd.DataFrame(roif.sort_values(by='gof').iloc[-1]).T
            best_dfs.append(best_cond_df)
            # But also save all results
            all_dfs.append(roif)

    # Save fit info for each stimconfig   
    if len(best_dfs) > 0: 
        best_rfits = pd.concat(best_dfs, axis=0)
        # rename indices to rois
        new_ixs = [int(i) for i in best_rfits['cell'].values]
        best_rfits.index = new_ixs
        # and all configs
        all_rfits = pd.concat(all_dfs, axis=0) 
        if verbose: 
            if gof_thr is not None: 
                print("... %i (of %i) fitable cells pass GoF thr %.2f" % (len(goodrois), len(passrois), gof_thr))
            else:
                print("... %i (of %i) fitable cells (no GoF thr)" % (best_rfits.shape[0], len(passrois)))

    return best_rfits, all_rfits #rmetrics_by_cfg

def aggregate_ori_fits(CELLS, traceid='traces001', fit_desc=None,
                       response_type='dff', responsive_test='nstds', responsive_thr=10.,
                       n_bootstrap_iters=1000, n_resamples=20, verbose=False,
                       return_missing=False, rootdir='/n/coxfs01/2p-data'):
    '''
    assigned_cells:  dataframe w/ assigned cells of dsets that have gratings
    '''
    if fit_desc is None:
        fit_desc = get_fit_desc(response_type=response_type, 
                            responsive_test=responsive_test, 
                            n_stds=n_stds, responsive_thr=responsive_thr, 
                            n_bootstrap_iters=n_bootstrap_iters, 
                            n_resamples=n_resamples)

    gdata=None
    no_fits=[]; g_=[];
    i = 0
    for (va, dk), g in CELLS.groupby(['visual_area', 'datakey']):
        try:
            # Get src dir
            ori_dir = get_ori_dir(dk, traceid=traceid, fit_desc=fit_desc)
            #print(ori_dir)
            # Load tuning results
            fitresults, fitparams = load_tuning_results(dk, 
                                            fit_desc=fit_desc, traceid=traceid)
            # Get OSI results for assigned cells
            rois_ = g['cell'].unique()
            boot_ = dict((k, v) for k, v in fitresults.items() if k in rois_)
        except Exception as e:
            print(e)
            print('ERROR: %s' % dk)
            continue

        # Aggregate fits
        best_fits, curr_fits = get_good_fits(boot_, fitparams, 
                                                 gof_thr=None, verbose=verbose)
        if best_fits is None:
            no_fits.append('%s_%s' % (va, dk))
            continue
        curr_fits['visual_area'] = va
        curr_fits['datakey'] = dk
        g_.append(curr_fits)
    gdata = pd.concat(g_, axis=0).reset_index(drop=True)
    if verbose:
        print("Datasets with NO fits found:")
        for s in no_fits:
            print(s)
    if return_missing:
        return gdata, no_fits
    else:
        return gdata

