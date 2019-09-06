#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 13:36:00 2019

@author: julianarhee
"""

#!/usr/bin/env python2
"""
Created on Fri Aug 2 16:20:01 2019

@author: julianarhee
"""

import datetime
import os
import cv2
import glob
import h5py
import sys
import optparse
import copy
import json
import traceback
import time
import pylab as pl
from collections import Counter
import seaborn as sns
import cPickle as pkl
import numpy as np
import pylab as pl
import pandas as pd
import seaborn as sns
import tifffile as tf


import multiprocessing as mp
import itertools

from pipeline.python.classifications import osi_dsi as osi
from pipeline.python.classifications import test_responsivity as resp
#from pipeline.python.classifications import experiment_classes as util
from pipeline.python.utils import natural_keys, label_figure#, load_data
from pipeline.python.traces.trial_alignment import aggregate_experiment_runs 

#from pipeline.python.retinotopy import fit_2d_rfs as rf

from pipeline.python.utils import uint16_to_RGB
from skimage import exposure
from matplotlib import patches

from scipy import stats as spstats
from scipy.interpolate import interp1d
import scipy.optimize as spopt


#%%
# #############################################################################
# Metric calculations:
# #############################################################################

def group_configs(group, response_type):
    config = group['config'].unique()[0]
    group.index = np.arange(0, group.shape[0])

    return pd.DataFrame(data={'%s' % config: group[response_type]})

def get_ASI(response_vector, thetas):
    if np.max(thetas) > np.pi:
        thetas = [np.deg2rad(th) for th in thetas]
        
    asi = np.abs( np.sum( [theta_resp * np.exp((2j*2*np.pi*theta_val) / (2*np.pi))\
                           for theta_resp, theta_val in zip(response_vector, thetas)] ) ) / np.sum(np.abs(response_vector))
    return asi

def get_DSI(response_vector, thetas):
    if np.max(thetas) > np.pi:
        thetas = [np.deg2rad(th) for th in thetas]
    dsi = np.abs( np.sum( [theta_resp * np.exp((1j*2*np.pi*theta_val) / (2*np.pi))\
                           for theta_resp, theta_val in zip(response_vector, thetas)] ) ) / np.sum(np.abs(response_vector))
    return dsi


def get_circular_variance(response_vector, thetas):
    if np.max(thetas) > np.pi:
        thetas = [np.deg2rad(th) for th in thetas]
        
    circvar_asi = 1 - (np.abs( np.sum( [theta_resp * np.exp( (1j*theta_val) ) \
                                        for theta_resp, theta_val in zip(response_vector, thetas)] ) ) / np.sum(np.abs(response_vector)) )
    
    circvar_dsi = 1 - (np.abs( np.sum( [theta_resp * np.exp( (1j*theta_val*2) ) \
                           for theta_resp, theta_val in zip(response_vector, thetas)] ) ) / np.sum(np.abs(response_vector)) )
    
    return circvar_asi, circvar_dsi

#%%

# #############################################################################
# Data loading, saving, formatting
# #############################################################################


def get_gratings_run(animalid, session, fov, traceid='traces001', rootdir='/n/coxfs01/2p-data'):
    
    fovdir = os.path.join(rootdir, animalid, session, fov)
    
    found_dirs = glob.glob(os.path.join(fovdir, '*gratings*', 'traces', '%s*' % traceid, 'data_arrays', 'datasets.npz'))
    if int(session) < 20190511:
        return None
    
    try:
        if any('combined' in d for d in found_dirs):
            extracted_dir = [d for d in found_dirs if 'combined' in d][0]
        else:
            assert len(found_dirs) == 1, "ERROR: [%s|%s|%s|%s] More than 1 gratings experiments found, with no combined!" % (animalid, session, fov, traceid)
            extracted_dir = found_dirs[0]
    except Exception as e:
        print e
        return None
    
    run_name = os.path.split(extracted_dir.split('/traces')[0])[-1]
        
    return run_name


def convert_uint16(tracemat):
    offset = 32768
    arr = np.zeros(tracemat.shape, dtype='uint16')
    arr[:] = tracemat + offset
    return arr

def load_data(data_fpath, add_offset=True, make_equal=False):
    from pipeline.python.classifications import experiment_classes as util
    soma_fpath = data_fpath.replace('datasets', 'np_subtracted')
    print soma_fpath
    dset = np.load(soma_fpath)
    
    xdata_df = pd.DataFrame(dset['data'][:]) # neuropil-subtracted & detrended
    F0 = pd.DataFrame(dset['f0'][:]).mean().mean() # detrended offset
    #neuropil_df = pd.concat(dfs['neuropil-detrended'], axis=0).reset_index(drop=True) #drop=True)
    #neuropil_F0 = pd.concat(dfs['np_subtracted-F0'], axis=0).reset_index(drop=True).mean() #drop=True)

    if add_offset:
        #% Add baseline offset back into raw traces:
        neuropil_fpath = soma_fpath.replace('np_subtracted', 'neuropil')
        npdata = np.load(neuropil_fpath)
        neuropil_df = pd.DataFrame(npdata['data'][:]) #+ pd.DataFrame(npdata['f0'][:])
        print("adding NP offset...")
        raw_traces = xdata_df + neuropil_df.mean(axis=0) + F0 #neuropil_F0 + F0
    else:
        raw_traces = xdata_df + F0

    labels = pd.DataFrame(data=dset['labels_data'], columns=dset['labels_columns'])
    
    if make_equal:
        raw_traces, labels = util.check_counts_per_condition(raw_traces, labels)

    sdf = pd.DataFrame(dset['sconfigs'][()]).T
    
    gdf = resp.group_roidata_stimresponse(raw_traces.values, labels, return_grouped=True) # Each group is roi's trials x metrics
    
    #% # Convert raw + offset traces to df/F traces
    #min_mov = raw_traces.min().min()
    #if min_mov < 0:
    #    raw_traces = raw_traces - min_mov

    stim_on_frame = labels['stim_on_frame'].unique()[0]
    tmp_df = []
    for k, g in labels.groupby(['trial']):
        tmat = raw_traces.loc[g.index]
        bas_mean = np.nanmean(tmat[0:stim_on_frame], axis=0)
        tmat_df = (tmat - bas_mean) / bas_mean
        tmp_df.append(tmat_df)
    df_traces = pd.concat(tmp_df, axis=0)
    del tmp_df

    return df_traces, labels, gdf, sdf


def load_gratings_data(data_fpath, add_offset=True, make_equal=False):
    from pipeline.python.classifications import experiment_classes as util

    dset = np.load(data_fpath)
    
    if add_offset:
        #% Add baseline offset back into raw traces:
        F0 = np.nanmean(dset['corrected'][:] / dset['dff'][:] )
        print("offset: %.2f" % F0)
        raw_traces = pd.DataFrame(dset['corrected']) + F0
    else:
        raw_traces = pd.DataFrame(dset['corrected']) 
        minval = raw_traces.min().min()
        if minval < 0:
            raw_traces = raw_traces - minval

    labels = pd.DataFrame(data=dset['labels_data'], columns=dset['labels_columns'])
    
    if make_equal:
        raw_traces, labels = util.check_counts_per_condition(raw_traces, labels)

    sdf = pd.DataFrame(dset['sconfigs'][()]).T
    
    gdf = resp.group_roidata_stimresponse(raw_traces.values, labels, return_grouped=True) # Each group is roi's trials x metrics
    
    #% # Convert raw + offset traces to df/F traces
    stim_on_frame = labels['stim_on_frame'].unique()[0]
    tmp_df = []
    for k, g in labels.groupby(['trial']):
        tmat = raw_traces.loc[g.index]
        bas_mean = np.nanmean(tmat[0:stim_on_frame], axis=0)
        tmat_df = (tmat - bas_mean) / bas_mean
        tmp_df.append(tmat_df)
    df_traces = pd.concat(tmp_df, axis=0)
    del tmp_df

    return df_traces, labels, gdf, sdf


def get_stimulus_configs(animalid, session, fov, run_name, rootdir='/n/coxfs01/2p-data'):
    # Get stimulus configs
    stiminfo_fpath = os.path.join(rootdir, animalid, session, fov, run_name, 'paradigm', 'stimulus_configs.json')
    with open(stiminfo_fpath, 'r') as f:
        stiminfo = json.load(f)
    sdf = pd.DataFrame(stiminfo).T
    
    return sdf
    



def load_tuning_results(animalid='', session='', fov='', run_name='', traceid='traces001',
                        fit_desc='', traceid_dir=None, rootdir='/n/coxfs01/2p-data'):

    bootresults=None; fitparams=None;
    if traceid_dir is None:
        osidir = glob.glob(os.path.join(rootdir, animalid, session, fov, run_name,
                                        'traces', '%s*' % traceid, '*tuning*', fit_desc))[0]
    else:
        osidir = glob.glob(os.path.join(traceid_dir, '*tuning*', fit_desc))[0]

    results_fpath = os.path.join(osidir, 'fitresults.pkl')
    params_fpath = os.path.join(osidir, 'fitparams.json')
    
    if os.path.exists(results_fpath):
        print("Loading existing fits.")
        with open(results_fpath, 'rb') as f:
            bootresults = pkl.load(f)
        with open(params_fpath, 'r') as f:
            fitparams = json.load(f)
    else:
        print "NO FITS FOUND: %s" % fit_desc
        
    return bootresults, fitparams

def save_tuning_results(bootresults, fitparams):
    osidir = fitparams['directory']
    results_fpath = os.path.join(osidir, 'fitresults.pkl')
    params_fpath = os.path.join(osidir, 'fitparams.json')
    #bootdata_fpath = os.path.join(osidir, 'tuning_bootstrap_data.pkl')
    
    with open(results_fpath, 'wb') as f:
        pkl.dump(bootresults, f, protocol=pkl.HIGHEST_PROTOCOL)

    
    with open(results_fpath, 'wb') as f:
        pkl.dump(bootresults, f, protocol=pkl.HIGHEST_PROTOCOL)
    
    # Save params:
    with open(params_fpath, 'w') as f:
        json.dump(fitparams, f, indent=4, sort_keys=True)

    print("Saved!")
    return osidir


def get_fit_desc(response_type='dff', responsive_test=None, responsive_thr=0.05, n_stds=2.5,
                 n_bootstrap_iters=1000, n_resamples=20):
    if responsive_test is None:
        fit_desc = 'fit-%s_all-cells_boot-%i-resample-%i' % (response_type, n_bootstrap_iters, n_resamples) #, responsive_test, responsive_thr)
    elif responsive_test == 'nstds':
        fit_desc = 'fit-%s_responsive-%s-%.2f-thr%.2f_boot-%i-resample-%i' % (response_type, responsive_test, n_stds, responsive_thr, n_bootstrap_iters, n_resamples)
    else:
        fit_desc = 'fit-%s_responsive-%s-thr%.2f_boot-%i-resample-%i' % (response_type, responsive_test, responsive_thr, n_bootstrap_iters, n_resamples)

    return fit_desc


def create_osi_dir(animalid, session, fov, run_name='gratings', 
                   traceid='traces001', response_type='dff', n_stds=2.5,
                   responsive_test=None, responsive_thr=0.05,
                   n_bootstrap_iters=1000, n_resamples=20,
                   rootdir='/n/coxfs01/2p-data', traceid_dir=None):
        

    # Get RF dir for current fit type
    fit_desc = get_fit_desc(response_type=response_type, responsive_test=responsive_test, n_stds=n_stds,
                            responsive_thr=responsive_thr, n_bootstrap_iters=n_bootstrap_iters,
                            n_resamples=n_resamples)

    if traceid_dir is None:
        fov_dir = os.path.join(rootdir, animalid, session, fov)
        traceid_dirs = glob.glob(os.path.join(fov_dir, run_name, 'traces', '%s*' % traceid))
        if len(traceid_dirs) > 1:
            print "More than 1 trace ID found:"
            for ti, traceid_dir in enumerate(traceid_dirs):
                print ti, traceid_dir
            sel = input("Select IDX of traceid to use: ")
            traceid_dir = traceid_dirs[int(sel)]
        else:
            traceid_dir = traceid_dirs[0]
        #traceid = os.path.split(traceid_dir)[-1]
    
    osidir = os.path.join(traceid_dir, 'tuning', fit_desc)
    if not os.path.exists(osidir):
        os.makedirs(osidir)

    return osidir, fit_desc


#%%

# #############################################################################
# Fitting functions:
# #############################################################################

def get_init_params(response_vector):
    theta_pref = response_vector.idxmax()
    theta_null = (theta_pref + 180) % 360.
    r_pref = response_vector.loc[theta_pref]
    r_null = response_vector.loc[theta_null]
    sigma = np.mean(np.diff([response_vector.index.tolist()]))
    non_prefs = [t for t in response_vector.index.tolist() if t not in [theta_pref, theta_null]]
    r_offset = np.mean([response_vector.loc[t] for t in non_prefs])
    return r_pref, r_null, theta_pref, sigma, r_offset


def angdir180(x):
    '''wraps anguar diff values to interval 0, 180'''
    return min(np.abs([x, x-360, x+360]))

def double_gaussian( x, c1, c2, mu, sigma, C ):
    #(c1, c2, mu, sigma) = params
    x1vals = np.array([angdir180(xi - mu) for xi in x])
    x2vals = np.array([angdir180(xi - mu - 180 ) for xi in x])
    res =   C + c1 * np.exp( - x1vals**2.0 / (2.0 * sigma**2.0) )             + c2 * np.exp( - x2vals**2.0 / (2.0 * sigma**2.0) )

#    res =   C + c1 * np.exp( - ((x - mu) % 360.)**2.0 / (2.0 * sigma**2.0) ) \
#            + c2 * np.exp( - ((x + 180 - mu) % 360.)**2.0 / (2.0 * sigma**2.0) )

#        res =   c1 * np.exp( - (x - mu)**2.0 / (2.0 * sigma**2.0) ) \
#                #+ c2 * np.exp( - (x - mu2)**2.0 / (2.0 * sigma2**2.0) )
    return res

def evaluate_fit_params(x, y, popt):
    fitr = double_gaussian( x, *popt)
        
    # Get residual sum of squares 
    residuals = y - fitr
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - (ss_res / ss_tot)
    
    return fitr
    
def fit_osi_params(x, y, init_params=[0, 0, 0, 0, 0], bounds=[np.inf, np.inf, np.inf, np.inf, np.inf]):

    roi_fit=None; fitr=None;
    try:
        popt, pcov = spopt.curve_fit(double_gaussian, x, y, p0=init_params, maxfev=1000, bounds=bounds)
        fitr = double_gaussian( x, *popt)
        assert pcov.max() != np.inf
        success = True
    except Exception as e:
        success = False
        pcov =None
        popt = None
        
#    # Get residual sum of squares 
#    residuals = y - fitr
#    ss_res = np.sum(residuals**2)
#    ss_tot = np.sum((y - np.mean(y))**2)
#    r2 = 1 - (ss_res / ss_tot)
        
    roi_fit = {'pcov': pcov,
               'popt': popt,
               #'fit_y': fitr,
               #'r2': r2,
                 #'x': x,
                 #'y': y,
               #'init': init_params,
               'success': success}

        
    return roi_fit, fitr
#
#def fit_direction_selectivity(x, y, init_params=[0, 0, 0, 0, 0], bounds=[np.inf, np.inf, np.inf, np.inf, np.inf]):
#    roi_fit = None
#    
#    popt, pcov = spopt.curve_fit(double_gaussian, x, y, p0=init_params, maxfev=1000, bounds=bounds)
#    fitr = double_gaussian( x, *popt)
#        
#    # Get residual sum of squares 
#    residuals = y - fitr
#    ss_res = np.sum(residuals**2)
#    ss_tot = np.sum((y - np.mean(y))**2)
#    r2 = 1 - (ss_res / ss_tot)
#    
#    if pcov.max() == np.inf: # or r2 == 1: #round(r2, 3) < 0.15 or 
#        success = False
#    else:
#        success = True
#        
#    if success:
#        roi_fit = {'pcov': pcov,
#                   'popt': popt,
#                   'fit_y': fitr,
#                   'r2': r2,
#                     #'x': x,
#                     #'y': y,
#                   #'init': init_params,
#                   'success': success}
#    return roi_fit


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


def fit_ori_tuning(responsedf, n_intervals_interp=3):
    '''
    responsedf = Series
        index : tested_oris
        values : mean value at tested ori
    '''
    
    response_pref=None; response_null=None; theta_pref=None; 
    sigma=None; response_offset=None;
    asi_t=None; dsi_t=None;

    tested_oris = responsedf.index.tolist()
    oris_interp = interp_values(tested_oris, n_intervals=n_intervals_interp, wrap_value=360)
    resps_interp = interp_values(responsedf, n_intervals=n_intervals_interp, wrap_value=responsedf[0])

    # initial params
    init_params = get_init_params(responsedf)
    r_pref, r_null, theta_pref, sigma, r_offset = init_params
    init_bounds = ([0, 0, -np.inf, sigma/2., -r_pref], [3*r_pref, 3*r_pref, np.inf, np.inf, r_pref])
    
    rfit, fitv = fit_osi_params(oris_interp, resps_interp, init_params, bounds=init_bounds)
    
    response_pref=None; response_null=None; theta_pref=None; 

    if rfit['success']:
         asi_t = get_ASI(fitv[0:], oris_interp[0:])
         dsi_t = get_DSI(fitv[0:], oris_interp[0:])
         #circvar_asi_t, circvar_dsi_t = get_circular_variance(rfit['fit_y'][0:], oris_interp[0:])
         response_pref, response_null, theta_pref, sigma, response_offset = rfit['popt']
         #r2 = rfit['r2']
    else:
        #print('no fits')
        response_pref=None; response_null=None; theta_pref=None; 
        sigma=None; response_offset=None;
        asi_t=None; dsi_t=None;
         
    fitres = pd.Series({'response_pref': response_pref,
                        'response_null': response_null,
                        'theta_pref': theta_pref,
                        'sigma': sigma,
                        'response_offset': response_offset,
                        'asi': asi_t,
                        'dsi': dsi_t},  name=responsedf.name)
#                        'r2': r2}, name=responsedf.name)
    
    return fitres

def fit_from_params( fitres, tested_oris, n_intervals_interp=3):
    fitparams = ['response_pref', 'response_null', 'theta_pref', 'sigma', 'response_offset']
    popt = tuple(fitres.loc[fitparams].values)
    oris_interp = interp_values(tested_oris, n_intervals=n_intervals_interp, wrap_value=360)
    try:
        fitr = double_gaussian( oris_interp, *popt)
    except Exception as e:
        fitr = [None for _ in range(len(oris_interp))]
        
    return pd.Series(fitr, name=fitres.name)

def get_r2(fitr, y):
        
    # Get residual sum of squares 
    residuals = y - fitr
    ss_res = np.sum(residuals**2, axis=0)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - (ss_res / ss_tot)
    
    return r2




#%%
import math
from functools import partial
from contextlib import contextmanager

class NoDaemonProcess(mp.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
import multiprocessing.pool
class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess


@contextmanager
def poolcontext(*args, **kwargs):
    pool = mp.Pool(*args, **kwargs)
    yield pool
    pool.terminate()

def pool_bootstrap(rdf_list, sdf, allconfigs=False,
                   statdf=None, response_type='dff',
                    n_bootstrap_iters=1000, n_resamples=20, 
                    n_intervals_interp=3, min_cfgs_above=2, min_nframes_above=10,
                    n_processes=1):

    if allconfigs:
        #with poolcontext(processes=n_processes) as pool:
        pool = MyPool(n_processes)
    
        results = pool.map(partial(boot_roi_responses_allconfigs, 
                                   sdf=sdf,
                                   statdf=statdf,
                                   response_type=response_type, 
                                   n_bootstrap_iters=n_bootstrap_iters,
                                   n_resamples=n_resamples,
                                   n_intervals_interp=n_intervals_interp,
                                   min_cfgs_above=min_cfgs_above,
                                   min_nframes_above=min_nframes_above), rdf_list)
    else:
        pool = mp.Pool(processes=n_processes)
        #results = pool.map(boot_star, itertools.izip(rdf_list, itertools.repeat(sdf)))
        results = pool.map(partial(boot_roi_responses_bestconfig, 
                                                   sdf=sdf,
                                                   statdf=statdf,
                                                   response_type=response_type, 
                                                   n_bootstrap_iters=n_bootstrap_iters,
                                                   n_resamples=n_resamples,
                                                   n_intervals_interp=n_intervals_interp,
                                                   min_cfgs_above=min_cfgs_above,
                                                   min_nframes_above=min_nframes_above), rdf_list)

    pool.close()
    pool.join()
    
    bootresults = {k: v for d in results for k, v in d.items()}

    return bootresults
    
#%

def boot_roi_responses_allconfigs(roi_df, sdf, statdf=None, response_type='dff',
                            n_bootstrap_iters=1000, n_resamples=20, 
                            n_intervals_interp=3, min_cfgs_above=2, min_nframes_above=10,
                            n_processes=1):
    '''
    Inputs
        roi_df (pd.DataFrame) 
            Response metrics for all trials for 1 cell.
        sdf (pd.DataFrame) 
            Stimulus config key / info
        statdf (pd.DataFrame) 
            Dataframe from n_stds responsive test that gives N frames above baseline for each stimulus config (avg over trials)
        response_type (str)
            Response metric to use for calculating tuning.
        min_cfgs_above (int)
            Number of stimulus configs that should pass the "responsive" threshold for cell to count as responsive
        min_nframes_above (int)
            Min num frames (from statdf) that counts as a cell to be reposnsive for a given stimulus config
            
    Returns
        List of dicts from mp for each roi's results
        {configkey:   'data':
                            {'responses': dataframe, rows=trials and cols=configs,
                             'tested_values': tested values we are interpreting over and fitting}
                      'stimulus_configs': list of stimulus configs corresponding to current set
                      'fits': 
                             {'xv': interpreted x-values,
                              'yv': dataframe of each boot iter of response values
                              'fitv': dataframe of each boot iter's fit responses}
                      'results': dataframe of all fit params
                  
        Non-responsive cells are {roi: None}
        Responsive cells without tuning fits will return with 'fits' and 'results' as None
        
    '''

    roi = roi_df.index[0]    
    filter_configs = statdf is not None

    constant_params = ['aspect', 'luminance', 'position', 'stimtype', 'direction', 'xpos', 'ypos']
    params = [c for c in sdf.columns if c not in constant_params]
    stimdf = sdf[params]
    tested_oris = sdf['ori'].unique()

    # Get all config sets: Each set is a set of the 8 tested oris at a specific combination of non-ori params
    configsets = dict((tuple(round(ki, 1) for ki in k), sorted(cfgs.index.tolist(), key=lambda x: stimdf['ori'][x]) )\
                     for k, cfgs in stimdf.groupby(['sf', 'size', 'speed']) )

    def bootstrap_fitter(curr_configset, roi_df, statdf, out_q):
        roi = roi_df.index[0]
        
        oridata = {}
        for ckey, currcfgs in curr_configset.items():
            if filter_configs:
                # Idenfify cells that are responsive to given stimulus condition before trying to fit...
                responsive = len(np.where(statdf[roi].loc[currcfgs] >= min_nframes_above)[0]) >= min_cfgs_above
                if not responsive:
                    oridata[ckey] = None
                    continue
       
            # Get all trials of current set of cfgs:
            rdf = roi_df[roi_df['config'].isin(currcfgs)][['config', 'trial', response_type]]
            responses_df = pd.concat([pd.Series(g[response_type], name=c).reset_index(drop=True)\
                                      for c, g in rdf.groupby(['config'])], axis=1)
            datadict = {'responses': responses_df, 
                        'tested_values': tested_oris}
            
            # Bootstrap distN of responses (rand w replacement):
            bootdf = pd.concat([responses_df.sample(n_resamples, replace=True).mean(axis=0) for ni in range(n_bootstrap_iters)], axis=1)
            bootdf.index = [sdf['ori'][c] for c in bootdf.index]
        
            # Find init params for tuning fits and set fit constraints:
            fitp = bootdf.apply(fit_ori_tuning, args=[n_intervals_interp], axis=0) # Get fit params
            if fitp.dropna().shape[0] == 0:
                fitdict = None
                fitp = None
            else:
                # Get fit and data values
                fitv = fitp.apply(fit_from_params, args=[tested_oris], axis=0)        # Get fit response
                yvs = bootdf.apply(interp_values, args=[n_intervals_interp, True], axis=0, reduce=True) # Get original (interp) responses
                xvs = interp_values(tested_oris, n_intervals=n_intervals_interp, wrap_value=360)
                fitdict = {'xv': xvs, 
                           'yv': yvs, 
                           'fitv': fitv,
                           'n_intervals_interp': n_intervals_interp}
                
                # Create dataframe of all fit params
                fitp = fitp.T
                fitp['r2'] = get_r2(fitv, yvs) # Calculate coeff of deterim
                fitp['cell'] = [roi for _ in range(n_bootstrap_iters)]
            
            oridata[ckey] = {'results': fitp, 
                             'fits': fitdict,
                             'data': datadict,
                             'stimulus_configs': currcfgs}
            
        # Get rid of cells that didn't pass response threshold
        for ckey, orid in oridata.items():
            if orid is None:
                oridata.pop(ckey)
                    
        out_q.put(oridata)

    start_t = time.time()
    # Each process gets "chunksize' filenames and a queue to put his out-dict into:
    config_list = configsets.keys()
    out_q = mp.Queue()
    chunksize = int(math.ceil(len(config_list) / float(n_processes)))
    procs = []
    for i in range(n_processes):
        currcfgs = config_list[chunksize*i:chunksize*(i+1)]
        curr_configset = dict((ckey, configsets[ckey]) for ckey in currcfgs)
        p = mp.Process(target=bootstrap_fitter,
                       args=(curr_configset, roi_df, statdf, out_q))
        
        procs.append(p)
        p.start()
        
    # Collect all results into single results dict. We should know how many dicts to expect:
    results = {}
    for i in range(n_processes):
        results.update(out_q.get())

    # Wait for all worker processes to finish
    for p in procs:
        print "Finished:", p
        p.join()
        
    end_t = time.time() - start_t
    print("--> Elapsed time: {0:.2f}sec".format(end_t))
      
    return {roi: results}

#%
    
def boot_roi_responses_bestconfig(roi_df, sdf, statdf=None, response_type='dff',
                            n_bootstrap_iters=1000, n_resamples=20, 
                            n_intervals_interp=3, min_cfgs_above=2, min_nframes_above=10):
    '''
    Find stimulus config (any) that has max response, fit tuning to that config.
    '''
    filter_configs = statdf is not None

    
    #bootstrapfits = []
    roi = roi_df.index[0]
    
    constant_params = ['aspect', 'luminance', 'position', 'stimtype']
    params = [c for c in sdf.columns if c not in constant_params]
    stimdf = sdf[params]
    tested_oris = sdf['ori'].unique()

    # Find best config:
    best_cfg = roi_df.groupby(['config']).mean()[response_type].idxmax()
    best_cfg_params = stimdf.loc[best_cfg][[p for p in params if p!='ori']]
    currcfgs = sorted([c for c in stimdf.index.tolist() \
                        if all(stimdf.loc[c][[p for p in params if p!='ori']] == best_cfg_params)],\
                        key = lambda x: stimdf['ori'][x])
    
    sz = np.mean([sdf['size'][c] for c in currcfgs])[0]
    sf = np.mean([sdf['sf'][c] for c in currcfgs])[0]
    speed = np.mean([sdf['speed'][c] for c in currcfgs])[0]
    ckey = (round(sf, 1), round(sz, 1), round(speed, 1))
    
    oridata = {}
    if filter_configs:
        responsive = len(np.where(statdf[roi].loc[currcfgs] >= min_nframes_above)[0]) >= min_cfgs_above
        if not responsive:
            oridata[ckey] = None
            return {roi: oridata}

    # Get all trials of current set of cfgs:
    #trialdf = roi_df[roi_df['config'].isin(curr_cfgs)]
    #rdf = trialdf[['config', 'trial', response_type]]
    rdf = roi_df[roi_df['config'].isin(currcfgs)][['config', 'trial', response_type]]
    #grouplist = [group_configs(group, response_type) for config, group in rdf.groupby(['config'])] 
    #responses_df = pd.concat(grouplist, axis=1)
    responses_df = pd.concat([pd.Series(g[response_type], name=c) for c, g, in rdf.groupby(['config'])], axis=1)

    # Get all trials of current set of cfgs:
    rdf = roi_df[roi_df['config'].isin(currcfgs)][['config', 'trial', response_type]]
    responses_df = pd.concat([pd.Series(g[response_type], name=c).reset_index(drop=True)\
                              for c, g, in rdf.groupby(['config'])], axis=1)
    datadict = {'responses': responses_df, 
                'tested_values': tested_oris}
            
    # Bootstrap distN of responses (rand w replacement):
    bootdf = pd.concat([responses_df.sample(n_resamples, replace=True).mean(axis=0) for ni in range(n_bootstrap_iters)], axis=1)
    bootdf.index = [sdf['ori'][c] for c in bootdf.index]

    # Find init params for tuning fits and set fit constraints:
    fitp = bootdf.apply(fit_ori_tuning, args=[n_intervals_interp], axis=0) # Get fit params
    if fitp.dropna().shape[0] == 0:
        fitdict = None
        fitp = None
    else:
        # Get fit and data values
        fitv = fitp.apply(fit_from_params, args=[tested_oris], axis=0)        # Get fit response
        yvs = bootdf.apply(interp_values, args=[n_intervals_interp, True], axis=0, reduce=True) # Get original (interp) responses
        xvs = interp_values(tested_oris, n_intervals=n_intervals_interp, wrap_value=360)
        fitdict = {'xv': xvs, 
                   'yv': yvs, 
                   'fitv': fitv,
                   'n_intervals_interp': n_intervals_interp}
        
        # Create dataframe of all fit params
        fitp = fitp.T
        fitp['r2'] = get_r2(fitv, yvs) # Calculate coeff of deterim
        fitp['cell'] = [roi for _ in range(n_bootstrap_iters)]
    
    oridata[ckey] = {'results': fitp, 
                     'fits': fitdict,
                     'data': datadict,
                     'stimulus_configs': currcfgs}
            
    return {roi: oridata}


#def boot_star(a_b):
#    """Convert `f([1,2])` to `f(1,2)` call."""
#    return boot_roi_responses_bestconfig(*a_b)
#
#def pool_bootstrap_bestconfig(rdf_list, sdf, statdf=None, response_type='dff',
#                    n_bootstrap_iters=1000, n_resamples=20, 
#                    n_intervals_interp=3, min_cfgs_above=2, min_nframes_above=10,
#                    n_processes=1):
#    
#    pool = mp.Pool(processes=n_processes)
#    #results = pool.map(boot_star, itertools.izip(rdf_list, itertools.repeat(sdf)))
#    results = pool.map(partial(boot_roi_responses_bestconfig, 
#                                               sdf=sdf,
#                                               statdf=statdf,
#                                               response_type=response_type, 
#                                               n_bootstrap_iters=n_bootstrap_iters,
#                                               n_resamples=n_resamples,
#                                               n_intervals_interp=n_intervals_interp,
#                                               min_cfgs_above=min_cfgs_above,
#                                               min_nframes_above=min_nframes_above), rdf_list)
#    pool.close()
#    pool.join()
#    
#    bootresults = {k: v for d in results for k, v in d.items()}
#
#    return bootresults


#%%



def do_bootstrap(gdf, sdf, roi_list=None, allconfigs=True,
                 statdf=None, response_type='dff',
                n_bootstrap_iters=1000, n_resamples=20,
                n_intervals_interp=3, n_processes=1,
                min_cfgs_above=2, min_nframes_above=10):

    if roi_list is None:
        roi_list = np.arange(0, len(gdf.groups))
    
    #statdf = roistats['nframes_above']
    start_t = time.time()
    rdf_list = [gdf.get_group(roi) for roi in roi_list]
    results = pool_bootstrap(rdf_list, sdf, allconfigs=allconfigs,
                                 statdf=statdf, response_type=response_type,
                                        n_bootstrap_iters=n_bootstrap_iters, n_resamples=n_resamples,
                                        n_intervals_interp=n_intervals_interp, min_cfgs_above=min_cfgs_above,
                                        min_nframes_above=min_nframes_above, n_processes=n_processes)

    end_t = time.time() - start_t
    print "Multiple processes: {0:.2f}sec".format(end_t)
    #bootresults = {k: v for d in results for k, v in d.items()}
    #bootresults = dict((k, v) for d in results for k, v in d.items() if d is not None)

    return results

#%%

def plot_tuning_bootresults(roi, bootr, df_traces, labels, sdf, trace_type='dff'):
    curr_cfgs = bootr['stimulus_configs']
    responses_df = bootr['data']['responses']
    #roi = bootr['results']['cell'].unique()[0]
    
    fit_success = bootr['fits'] is not None
    
    fig = pl.figure(figsize=(12,8))
    fig.patch.set_alpha(1)
    nr=2; nc=8;
    #s_row=0
    
    # Plot original data - PSTH
    fig, ax = plot_psth_roi(roi, df_traces, labels, curr_cfgs, sdf, 
                            trace_type='dff',
                            fig=fig, nr=nr, nc=nc, s_row=0)
    ymin = np.min([0, ax.get_ylim()[0]])
    ax.set_ylim([ymin, ax.get_ylim()[1]])
    
    # Plot original data - tuning curves
    curr_oris = np.array([sdf['ori'][c] for c in curr_cfgs])  
    sz = np.mean([sdf['size'][c] for c in curr_cfgs])
    sf = np.mean([sdf['sf'][c] for c in curr_cfgs])
    sp = np.mean([sdf['speed'][c] for c in curr_cfgs])
    
    curr_resps = responses_df.mean()
    curr_sems = responses_df.sem()
    fig, ax1 = plot_tuning_curve_roi(curr_oris, curr_resps, curr_sems=curr_sems, 
                                     response_type=trace_type,
                                     fig=fig, nr=nr, nc=nc, s_row=1, colspan=5,
                                     marker='o', markersize=5, lw=0)

    fig, ax2 = plot_tuning_polar_roi(curr_oris, curr_resps, curr_sems=curr_sems, 
                                     response_type=trace_type,
                                     fig=fig, nr=nr, nc=nc, s_row=1, s_col=6, colspan=2)

    if fit_success:
        # Plot bootstrap fits
        oris_interp = bootr['fits']['xv']
        #resps_interp = bootr['fits']['yv'].mean(axis=1)
        #resps_interp_sem = spstats.sem(bootr['fits']['yv'].values, axis=1) # get sem across itters
        resps_fit = bootr['fits']['fitv'].mean(axis=1)
        resps_fit_sem = spstats.sem(bootr['fits']['fitv'].values, axis=1) # get sem across itters
        n_intervals_interp = bootr['fits']['n_intervals_interp']
        
    
        fig, ax1 = plot_tuning_curve_roi(oris_interp[0:-n_intervals_interp], 
                                         resps_fit[0:-n_intervals_interp], 
                                         curr_sems=resps_fit_sem[0:-n_intervals_interp], 
                                         response_type=trace_type,color='cornflowerblue',
                                         markersize=0, lw=1, marker=None,
                                         fig=fig, ax=ax1, nr=nr, nc=nc, s_row=1, colspan=5)
    
        fig, ax2 = plot_tuning_polar_roi(oris_interp, 
                                         resps_fit, 
                                         curr_sems=resps_fit_sem, 
                                         response_type=trace_type, color='cornflowerblue',
                                         fig=fig, ax=ax2, nr=nr, nc=nc, s_row=1, s_col=6, colspan=2)
            
        r2_avg = bootr['results']['r2'].mean()
        ax1.plot(0, 0, alpha=0, label='r2=%.2f' % r2_avg)
        ax1.legend() #loc='upper left')
        #ax1.text(0, ax1.get_ylim()[-1]*0.75, 'r2=%.2f' % r2_avg, fontsize=6)
    else:
        ax1.plot(0, 0, alpha=0, label='no fit')
        ax1.legend(loc='upper left')
        #ax1.text(0, ax1.get_ylim()[-1]*0.75, 'no fit', fontsize=10)

    ymin = np.min([0, ax1.get_ylim()[0]])
    ax1.set_ylim([ymin,  ax1.get_ylim()[1]])
    ax1.set_yticks([ymin, ax1.get_ylim()[1]])
    ax1.set_yticklabels([round(ymin, 2), round( ax1.get_ylim()[1], 2)])
    sns.despine(trim=True, offset=4, ax=ax1)
    ax1.set_xticks(curr_oris)
    ax1.set_xticklabels(curr_oris)
    

    stimkey = 'sf-%i-sz-%.2f-speed-%if' % (sf, sz, sp)
    fig.suptitle('roi %i (sf %.1f, sz %i, speed %i)' % (roi, sf, sz, sp), fontsize=8)

    return fig, stimkey


    #%%
    
#
#def do_bootstrap_fits(gdf, sdf, roi_list=None, 
#                        response_type='dff',
#                        n_bootstrap_iters=1000, n_resamples=20,
#                        n_intervals_interp=3, n_processes=1):
#
#    if roi_list is None:
#        roi_list = np.arange(0, len(gdf.groups))
#        
#    start_t = time.time()
#    results = pool_bootstrap([gdf.get_group(roi) for roi in roi_list], sdf, n_processes=n_processes)
#    end_t = time.time() - start_t
#    print "Multiple processes: {0:.2f}sec".format(end_t)
#
#    tested_values = sdf['ori'].unique()
#    print len(results)
#    doables = [r for r in results if len(r['fitdata']['results_by_iter']) > 0]
#    print "Doable fits: %i" % len(doables)
#    #interp_values = results[0]['fitdata']['results_by_iter'][0]['x']
#    interp_values = doables[0]['fitdata']['results_by_iter'][0]['x']
#
#    fitdf = pd.concat([res['fitdf'] for res in results])
#
#    fitparams = {'n_bootstrap_iters': n_bootstrap_iters,
#                 'n_resamples': n_resamples,
#                 'n_intervals_interp': n_intervals_interp,
#                 'tested_values': list(tested_values),
#                 'interp_values': interp_values,
#                 'roi_list': [r for r in roi_list]}    
#        
#    fitdata = {'results_by_iter': dict((res['fitdf']['cell'].unique()[0], res['fitdata']['results_by_iter'])\
#                                       for res in results if len(res['fitdf']['cell'].unique())>0),
#               'original_data': dict((res['fitdf']['cell'].unique()[0], res['fitdata']['original_data'])\
#                                     for res in results if len(res['fitdf']['cell'].unique()) > 0)}
#    
#    #-------------------------
#
#    return fitdf, fitparams, fitdata


    
#%%


def get_tuning(animalid, session, fov, run_name, return_iters=False,
               traceid='traces001', roi_list=None, statdf=None,
               response_type='dff', allconfigs=True,
               n_bootstrap_iters=1000, n_resamples=20, n_intervals_interp=3,
               make_plots=True, responsive_test='ROC', responsive_thr=0.05, n_stds=2.5,
               create_new=False, rootdir='/n/coxfs01/2p-data', n_processes=1,
               min_cfgs_above=2, min_nframes_above=10):
    
    from pipeline.python.classifications import experiment_classes as util
    
    roi_list=None; statdf=None;
    # Select only responsive cells:
    if responsive_test is not None:
        roistats, roi_list, nrois_total = util.get_roi_stats(animalid, session, fov, 
                                                             exp_name=run_name, 
                                                             traceid=traceid, 
                                                             response_type=response_type, 
                                                             responsive_test=responsive_test, 
                                                             responsive_thr=responsive_thr, n_stds=n_stds,
                                                             rootdir=rootdir)
        if responsive_test == 'nstds':
            statdf = roistats['nframes_above']
        else:
            print("%s -- not implemented for gratings...")
            statdf = None
        
    # Get tuning dirs
    osidir, fit_desc = create_osi_dir(animalid, session, fov, run_name, traceid=traceid,
                                      response_type=response_type, responsive_test=responsive_test, n_stds=n_stds,
                                      responsive_thr=responsive_thr, n_bootstrap_iters=n_bootstrap_iters,
                                      n_resamples=n_resamples, rootdir=rootdir)
    
    if not os.path.exists(osidir):
        os.makedirs(osidir)
    
    traceid_dir =  osidir.split('/tuning/')[0] #glob.glob(os.path.join(rootdir, animalid, session, fov, run_name, 'traces', '%s*' % traceid))[0]    
    
    data_fpath = os.path.join(traceid_dir, 'data_arrays', 'np_subtracted.npz')
    do_fits = False

    if not os.path.exists(data_fpath):
        # Realign traces
        print("*****corrected offset unfound, running now*****")
        print("%s | %s | %s | %s | %s" % (animalid, session, fov, run_name, traceid))

        aggregate_experiment_runs(animalid, session, fov, 'gratings', traceid=traceid)
        print("*****corrected offsets!*****")
        do_fits=True                        

    if create_new is False:
        try:
            bootresults, fitparams = load_tuning_results(traceid_dir=traceid_dir,
                                                        fit_desc=fit_desc)
            assert bootresults is not None, "Unable to load tuning: %s" % fit_desc
        except Exception as e:
            traceback.print_exc()
            do_fits = True
    else:
        do_fits=True
    
    data_identifier = '%s\n%s' % ('|'.join([animalid, session, fov, run_name, traceid]), fit_desc)

    # Do fits
    if do_fits:
        print("Loading data and doing fits")
        df_traces, labels, gdf, sdf = load_data(data_fpath, add_offset=True, make_equal=False)
        
        if roi_list is None:
            roi_list = np.arange(0, len(gdf.groups))
    
        #% # Fit all rois in list
        print "... Fitting %i rois:" % len(roi_list), roi_list

        # Save fitso_bootstrap_fits
        bootresults = do_bootstrap(gdf, sdf, allconfigs=allconfigs,
                                   roi_list=roi_list, statdf=statdf,
                                    response_type=response_type,
                                    n_bootstrap_iters=n_bootstrap_iters, n_resamples=n_resamples,
                                    n_intervals_interp=n_intervals_interp, n_processes=n_processes,
                                    min_cfgs_above=min_cfgs_above, min_nframes_above=min_nframes_above)
        
        passrois = sorted([k for k, v in bootresults.items() if any(v.values())])
        print("%i cells fit at least 1 tuning curve." % len(passrois))
        
        non_ori_configs = get_non_ori_params(sdf)
        fitparams = {'directory': osidir,
                          'response_type': response_type,
                          'responsive_test': responsive_test,
                          'responsive_thr': responsive_thr if responsive_test is not None else None,
                          'n_stds': n_stds if responsive_test=='nstds' else None,
                          'n_bootstrap_iters': n_bootstrap_iters,
                          'n_resamples': n_resamples,
                          'n_intervals_interp': n_intervals_interp,
                          'min_cfgs_above': min_cfgs_above if statdf is not None else None,
                          'min_nframes_above': min_nframes_above if statdf is not None else None,
                          'non_ori_configs': non_ori_configs}
        save_tuning_results(bootresults, fitparams)
        
        if make_plots:
            if not os.path.exists(os.path.join(osidir, 'roi-fits')):
                os.makedirs(os.path.join(osidir, 'roi-fits'))
            print("Saving roi tuning fits to: %s" % os.path.join(osidir, 'roi-fits'))
            for roi in passrois:
                #stimparams = [cfg for cfg, res in bootresults[roi].items() if res is not None]
                for stimpara, bootr in bootresults[roi].items():
                    #  ['fits', 'stimulus_configs', 'data', 'results']
                    fig, stimkey = plot_tuning_bootresults(roi, bootr, df_traces, labels, sdf, trace_type='dff')
                    label_figure(fig, data_identifier)
                    pl.savefig(os.path.join(osidir, 'roi-fits', 'roi%05d__%s.png' % (int(roi+1), stimkey)))
                    pl.close()

    return bootresults, fitparams


def evaluate_tuning(animalid, session, fov, run_name, traceid='traces001', fit_desc='', gof_thr=0.66,
                   create_new=False, rootdir='/n/coxfs01/2p-data', plot_metrics=True):

    osidir = glob.glob(os.path.join(rootdir, animalid, session, fov, run_name, 
                          'traces', '%s*' % traceid, 'tuning', fit_desc))[0]
    
    assert os.path.exists(osidir), "Directory not found: %s - %s\--in dir: %s" % (traceid, fit_desc, os.path.join(rootdir, animalid, session, fov, run_name))
    response_type = fit_desc.split('-')[1].split('_')[0]
    
    data_identifier = '%s\n%s' % ('|'.join([animalid, session, fov, run_name]), fit_desc)
    bootresults, fitparams = load_tuning_results(animalid=animalid, session=session, fov=fov, 
                                                 run_name=run_name, fit_desc=fit_desc, rootdir=rootdir)

    # Evaluate metric fits
    if not os.path.exists(os.path.join(osidir, 'evaluation', 'gof-rois')):
        os.makedirs(os.path.join(osidir, 'evaluation', 'gof-rois'))

    
    rmetrics, rmetrics_by_cfg = get_good_fits(bootresults, fitparams, gof_thr=gof_thr)
    if rmetrics is None:
        print("Nothing to do here, all rois suck!")
        return None, None

    goodrois = rmetrics.index.tolist()
    print("%i cells have good fits (thr >= %.2f)" % (len(goodrois), gof_thr))

    #plot_metrics = create_new
    if len(goodrois)>0 and (plot_metrics or create_new):
        print("... plotting comparison metrics across bootstrap iters ...")
        bootd = aggregate_all_iters(bootresults, fitparams, gof_thr=gof_thr)
        plot_bootstrapped_params(bootd, fitparams, fit_metric='gof', fit_thr=gof_thr, data_identifier=data_identifier)
        plot_top_asi_and_dsi(bootd, fitparams, fit_metric='gof', fit_thr=gof_thr, topn=10, data_identifier=data_identifier)
    
        # Visualize all metrics
        metrics_to_plot = ['asi', 'dsi', 'response_pref', 'theta_pref', 'r2comb', 'gof']
        g = sns.pairplot(rmetrics[metrics_to_plot], height=2, aspect=1)
        label_figure(g.fig, data_identifier)
        pl.subplots_adjust(top=0.9, right=0.9)
        pl.savefig(os.path.join(osidir, 'evaluation', 'metrics_avg-iters_gof-thr-%.2f_%i-rois.png' % (gof_thr, len(goodrois))))
        pl.close()
        
        fig = roi_polar_plot_by_config(bootresults, fitparams)
        label_figure(fig, data_identifier)
        pl.savefig(os.path.join(osidir, 'evaluation', 'polar-plots_gof-thr-%.2f_%irois.png' % (gof_thr, len(goodrois))))
        pl.close()
        print("*** done! ***")
   
    if len(goodrois)>0 and plot_metrics:
        # Visualize fit metrics for each roi's stimulus config
        for roi, g in rmetrics_by_cfg.groupby(['cell']):
            for skey in g.index.tolist():
                stimparam = tuple(float(i) for i in skey.split('-')[1::2])
                bootr = bootresults[roi][stimparam]
                fig = evaluate_fit_roi(roi, bootr, fitparams, response_type=response_type)
                label_figure(fig, data_identifier)
                fig.suptitle('roi %i (%s)' % (int(roi+1), skey))
                pl.savefig(os.path.join(osidir, 'evaluation', 'gof-rois', 'roi%05d__%s.png' % (int(roi+1), skey)))
                pl.close()
        
    return rmetrics, rmetrics_by_cfg    
    

#%%
# #############################################################################
# EVALUATION:
# #############################################################################

def aggregate_all_iters(bootresults, fitparams, gof_thr=0.66):
    
    niters = fitparams['n_bootstrap_iters']
    interp = fitparams['n_intervals_interp'] > 1
    
    passrois = sorted([k for k, v in bootresults.items() if any(v.values())])
    print("%i cells fit at least 1 tuning curve." % len(passrois))

    bootdata = []
    for roi in passrois:
        for stimparam, bootr in bootresults[roi].items():
            if bootr['fits'] is None:
                #print("%s: no fit" % str(stimparam))
                continue
            r2comb, gof, fitr = evaluate_fits(bootr, interp=interp)
            if np.isnan(gof): # or gof < gof_thr:
                #print("%s: bad fit" % str(stimparam))
                continue
            
            rfdf = bootr['results']
            rfdf['r2comb'] = [r2comb for _ in range(niters)]
            rfdf['gof'] = [gof for _ in range(niters)]
            rfdf['stimulus'] = str(stimparam)
            #tmpd = pd.DataFrame(rfdf.mean(axis=0)).T #, index=[roi])
            #stimkey = 'sf-%.2f-sz-%i-sp-%i' % stimparam
            #tmpd['stimconfig'] = str(stimparam)
                
            bootdata.append(rfdf)
            
    bootd = pd.concat(bootdata, axis=0)
    
    return bootd
            
def evaluate_fits(bootr, interp=False):
    # Average fit parameters aross boot iters
    params = [c for c in bootr['results'].columns if 'stim' not in c]
    avg_metrics = average_metrics_across_iters(bootr['results'][params])
    
    # Get combined r2 between original and avg-fit
    if interp:
        origr = interp_values(bootr['data']['responses'].mean(axis=0))
        thetas = bootr['fits']['xv']
    else:
        origr = bootr['data']['responses'].mean(axis=0).values
        thetas = bootr['data']['tested_values']
    #thetas = bootr['fits']['xv'][0:-1]
    #origr = interp_values(origr, n_intervals=3, wrap_value=origr[0])[0:-1]
    
    cpopt = tuple(avg_metrics[['response_pref', 'response_null', 'theta_pref', 'sigma', 'response_offset']].values[0])
    fitr = double_gaussian( thetas, *cpopt)
    r2_comb, _ = coeff_determination(origr, fitr)
    
    # Get Goodness-of-fit
    iqr = spstats.iqr(bootr['results']['r2'])
    gfit = np.mean(bootr['results']['r2']) * (1-iqr) * np.sqrt(r2_comb)
    
    return r2_comb, gfit, fitr
    

def get_good_fits(bootresults, fitparams, gof_thr=0.66):
   
    rmetrics=None; rmetrics_by_cfg=None; 
    niters = fitparams['n_bootstrap_iters']
    interp = fitparams['n_intervals_interp']>1

    passrois = sorted([k for k, v in bootresults.items() if any(v.values())])
    print("%i cells fit at least 1 tuning curve." % len(passrois))
            
    metrics_by_config = []
    roidfs=[]
    goodrois = []
    for roi in passrois:
        
        fitresults = []
        stimkeys = []
        for stimparam, bootr in bootresults[roi].items():
            if bootr['fits'] is None:
                #print("%s: no fit" % str(stimparam))
                continue
            r2comb, gof, fitr = evaluate_fits(bootr, interp=interp)
            if np.isnan(gof) or gof < gof_thr:
                #print("%s: bad fit" % str(stimparam))
                continue
            
            rfdf = bootr['results']
            rfdf['r2comb'] = [r2comb for _ in range(niters)]
            rfdf['gof'] = [gof for _ in range(niters)]
            
            tmpd = average_metrics_across_iters(rfdf) #pd.DataFrame(rfdf.mean(axis=0)).T #, index=[roi])
            stimkey = 'sf-%.2f-sz-%i-sp-%i' % stimparam
            #tmpd['stimconfig'] = str(stimparam)
                
            fitresults.append(tmpd)
            stimkeys.append(stimkey)
        
        if len(fitresults) > 0:
            roif = pd.concat(fitresults, axis=0).reset_index(drop=True)
            roif.index = stimkeys
            gof =  roif.mean()['gof']
            
            if gof >= 0.66:
                goodrois.append(roi)
                
            roidfs.append(average_metrics_across_iters(roif))
            
            #roidfs.append(pd.Series(roif.mean(axis=0), name=roi))
            metrics_by_config.append(roif)
   
    if len(roidfs) > 0: 
        rmetrics = pd.concat(roidfs, axis=0)
        rmetrics_by_cfg = pd.concat(metrics_by_config, axis=0)
    
    return rmetrics, rmetrics_by_cfg


def evaluate_fit_roi(roi, bootr, fitparams, response_type='dff'):
    #%  Look at residuals        
    n_intervals_interp = fitparams['n_intervals_interp']
    
    residuals = bootr['fits']['yv'].subtract(bootr['fits']['fitv'])
    mean_residuals = residuals.mean(axis=0)
    
    fig, axes = pl.subplots(2,3, figsize=(10,6))
    xv = bootr['fits']['xv'][0:-1]
    for fiter in bootr['fits']['yv'].columns:
        yv = bootr['fits']['yv'][fiter][0:-1]
        fitv = bootr['fits']['fitv'][fiter][0:-1]
        axes[0,0].plot(xv, fitv, lw=0.5)
        axes[0,1].scatter(yv, residuals[fiter][0:-1], alpha=0.5)
        
    # ax0: adjust ticks/labels
    ax = axes[0,0]
    ax.set_xticks(xv[0::n_intervals_interp])
    ax.set_xticklabels([int(x) for x in xv[0::n_intervals_interp]])
    ax.set_xlabel('thetas')
    ax.set_ylabel('fit')
    ymin = np.min([0, ax.get_ylim()[0]])
    ax.set_ylim([ymin, ax.get_ylim()[1]])
    ax.tick_params(labelsize=8)
    ax.set_ylabel(response_type)
    sns.despine(ax=ax, trim=True, offset=2)
    # ax1: adjust ticks/labels
    ax = axes[0,1]
    ax.axhline(y=0, linestyle=':', color='k')
    ax.set_ylabel('residuals')
    ax.set_xlabel('fitted value')
    ax.tick_params(labelsize=8)
    
    ax = axes[0,2]
    ax.hist(mean_residuals, bins=20, color='k', alpha=0.5)
    ax.set_xlabel('mean residuals')
    ax.set_ylabel('counts of iters')
    ax.set_xticks([ax.get_xlim()[0], ax.get_xlim()[-1]])
    ax.tick_params(labelsize=8)
    sns.despine(ax=ax, trim=True, offset=2)

    # Compare the original direction tuning curve with the fitted curve derived 
    # using the average of each fitting parameter (across 100 iterations)

    # Get residual sum of squares and compare ORIG and AVG FIT:
    thetas = xv[0::n_intervals_interp] #[0:-1]
    origr = bootr['data']['responses'].mean(axis=0).values
    
    r2comb, gof, fitr = evaluate_fits(bootr, interp=False)        
    ax = axes[1,0]
    ax.plot(thetas, origr, 'k', label='orig')
    ax.plot(thetas, fitr, 'r:', label='avg-fit')
    ax.set_title('r2-comb: %.2f' % r2comb)
    ax.legend()
    ax.set_xticks(xv[0::n_intervals_interp])
    ax.set_xticklabels([int(x) for x in xv[0::n_intervals_interp]], fontsize=8)
    ax.tick_params(labelsize=8)
    ax.set_xlabel('thetas')
    ax.set_ylabel(response_type)
    sns.despine(ax=ax, trim=True, offset=2)
    
    
    # Compare distN of preferred orientation across all iters
    fparams = bootr['results'] #get_params_all_iters(fitdata['results_by_iter'][roi])
    ax = axes[1,1]
    ax.hist(fparams['theta_pref'], alpha=0.5, bins=20, color='k', )
    #ax.set_xlim([fparams['theta_pref'].min(), fparams['theta_pref'].max()])
    ax.set_xticks(np.linspace(int(np.floor(fparams['theta_pref'].min())), int(np.ceil(fparams['theta_pref'].max())), num=5))
    sns.despine(ax=ax, trim=True, offset=2)
    ax.set_xlabel('preferred theta')
    ax.set_ylabel('counts of iters')
    ax.tick_params(labelsize=8)
    
    # Look at calculated ASI/DSIs across iters:
    ax = axes[1,2]
    ax.scatter(bootr['results']['asi'], bootr['results']['dsi'], c='k', marker='+', alpha=0.5)
    ax.set_xlim([0, 1]); ax.set_xticks([0, 0.5, 1]); ax.set_xlabel('ASI');
    ax.set_ylim([0, 1]); ax.set_yticks([0, 0.5, 1]); ax.set_ylabel('DSI');
    ax.set_aspect('equal')
    
    pl.subplots_adjust(hspace=.5, wspace=.5)
    
    fig.suptitle('roi %i' % int(roi+1))

    return fig

def average_metrics_across_iters(fitdf):
    means = {}
    roi = int(fitdf['cell'].unique()[0])
    #print("COLS:", fitdf.columns)
    for param in fitdf.columns:
        if 'stim' in param:
            meanval = fitdf[param].values[0]
        elif 'theta' in param:
            meanval = np.rad2deg(spstats.circmean(np.deg2rad(fitdf[param] % 360.)))
        else:
            meanval = fitdf[param].mean()
        means[param] = meanval
    return pd.DataFrame(means, index=[roi])



## Functions from when we did bestconfigs (instead of allconfigs)
#def get_combined_r2(fitdata):
#    '''
#    Control step 2: combined coefficient of determination
#    compare original direction tuning curve w/ fitted curve derived from 
#    average of each fitting param (across all bootstrap iters).
#    '''
#    r2c = {}
#    for roi in fitdata['results_by_iter'].keys():
#        origr, fitr = get_average_fit(roi, fitdata)
#        r2_comb, resid = coeff_determination(origr, fitr)
#        r2c[roi] = r2_comb
#    return r2c
#

#def get_average_fit(roi, fitdata):
#    
#    n_intervals_interp = fitdata['results_by_iter'][roi][0]['n_intervals_interp']
#    popt = get_average_params_over_iters(fitdata['results_by_iter'][roi])
#    thetas = fitdata['results_by_iter'][roi][0]['x'][0::n_intervals_interp][0:-1]
#    responses = fitdata['original_data'][roi]['responses']
#    origr = responses.mean().values
#    fitr = double_gaussian( thetas, *popt)
#    
#    return origr, fitr

def coeff_determination(origr, fitr):
    residuals = origr - fitr
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((origr - np.mean(origr))**2)
    r2 = 1 - (ss_res / ss_tot)
    
    return r2, residuals


#def get_gof(fitdf_roi):
#    '''
#    Use metric from Liang et al (2018). 
#    Note: Will return NaN of fit is crappy
#    
#    IQR:  interquartile range (difference bw 75th and 25th percentile of r2 across iterations)
#    r2_comb:  combined coeff of determination.
#    
#    '''
#    r2_comb = fitdf_roi['r2comb'].unique()[0]
#    iqr = spstats.iqr(fitdf_roi['r2'])
#    gfit = np.mean(fitdf_roi['r2']) * (1-iqr) * np.sqrt(r2_comb)
#    return gfit


#
#def check_fit_quality(fitdf, fitdata):
#    # Get residual sum of squares and compare ORIG and AVG FIT:
#    r2_comb_values = get_combined_r2(fitdata)
#    fitdf['r2comb'] = [r2_comb_values[r] for r in fitdf['cell'].values]
#    
#    # Calculate goodness of fit heuristic (quality + reliability of fit):
#    gof = fitdf.groupby(['cell']).apply(get_gof).dropna()
#    
#    if isinstance(fitdf['cell'].unique()[0], float):
#        fitdf['cell'] = fitdf['cell'].astype(int)
#        
#    #passrois = threshold_fitness_quality(gof, goodness_thr=goodness_thr, plot_hist=False)
#    #failrois = [r for r in fitdf['cell'].unique() if r not in passrois]
#    #gof_dict = dict((roi, fitval) for roi, fitval in zip(passrois, gof.values))
#    #gof_dict.update(dict((roi, None) for roi in failrois))    
#    
#    failrois = [r for r in fitdf['cell'].unique() if r not in gof.index.tolist()]
#    gof_dict = dict((roi, fitval) for roi, fitval in zip(gof.index.tolist(), gof.values))
#    gof_dict.update(dict((roi, None) for roi in failrois))
#    
#    fitdf['gof'] = [gof_dict[r] for r in fitdf['cell'].values]
#    
#    return fitdf
#
#def get_params_all_iters(fiters):
#    fparams = pd.DataFrame({'r_pref':  [fiter['popt'][0] for fiter in fiters],
#                        'r_null': [fiter['popt'][1] for fiter in fiters],
#                        'theta_pref': [fiter['popt'][2] for fiter in fiters],
#                        'sigma': [fiter['popt'][3] for fiter in fiters],
#                        'r_offset': [fiter['popt'][4] for fiter in fiters]
#                        })
#    fparams['theta_pref'] = fparams['theta_pref'] % 360.
#    
#    return fparams
#
#def get_average_params_over_iters(fiters):
#    '''
#    Take all bootstrap iterations and get average value for each parameter.
#    r_pref, r_null, theta_pref, sigma, r_offset = popt
#    '''
#    fparams = get_params_all_iters(fiters)
#    
#    r_pref = fparams.mean()['r_pref']
#    r_null = fparams.mean()['r_null']
#
#    theta_r = np.array([np.deg2rad(t) for t in [fparams['theta_pref'].values % 360.]])
#    theta_r.min()
#    theta_r.max()
#    np.rad2deg(spstats.circmean(theta_r))
#    theta_pref = np.rad2deg(spstats.circmean(theta_r)) #fparams.mean()['theta_pref']
#    sigma = fparams.mean()['sigma']
#    
#    r_offset = fparams.mean()['r_offset']
#    
#    popt = (r_pref, r_null, theta_pref, sigma, r_offset)
#    
#    return popt

#%
#
#def get_reliable_fits(fitdf, goodness_thr=0.66):
#    '''
#    Drops cells with goodness_thr below threshold
#    '''
#    #x = fitdf.groupby(['cell']).apply(lambda x: any(np.isnan(x['gof'])))==False
#    #goodfits = x[x.values==True].index.tolist()
#    goodfits = fitdf[fitdf['gof'] > goodness_thr]['cell'].unique()
#    
#    return goodfits
#    
#def threshold_fitness_quality(gof, goodness_thr=0.66, plot_hist=True, ax=None):    
#    if plot_hist:
#        if ax is None:
#            fig, ax = pl.subplots() #.figure()
#        ax.hist(gof, alpha=0.5)
#        ax.axvline(x=goodness_thr, linestyle=':', color='k')
#        ax.set_xlim([0, 1])
#        sns.despine(ax=ax, trim=True, offset=2)
#        ax.set_xlabel('G-fit values')
#        ax.set_ylabel('counts')
#
#        
#    good_fits = [int(r) for r in gof.index.tolist() if gof[r] > goodness_thr]
#    
#    if not plot_hist:
#        return good_fits
#    else:
#        return good_fits, ax
##    
#def evaluate_bootstrapped_tuning(fitdf, fitparams, fitdata, goodness_thr=0.66,
#                                 response_type='dff', create_new=False,
#                                 data_identifier='METADATA'):
#
#    # Create output dir for evaluation
#    roi_fitdir = fitparams['directory']
#    evaldir = os.path.join(roi_fitdir, 'evaluation')
#    if not os.path.exists(evaldir):
#        os.makedirs(evaldir)
#    
#    roi_evaldir = os.path.join(evaldir, 'fit_rois')
#    if not os.path.exists(roi_evaldir):
#        os.makedirs(roi_evaldir)
#    
#        
#    # Get residual sum of squares and compare ORIG and AVG FIT:
#    do_evaluation = False
#    evaluate_fpath = os.path.join(roi_fitdir, 'tuning_bootstrap_evaluation.pkl')    
#    if os.path.exists(evaluate_fpath) and create_new is False:
#        try:
#            print("Loading existing evaluation results")
#            with open(evaluate_fpath, 'rb') as f:
#                evaluation_results = pkl.load(f)
#            fitdf = evaluation_results['fits']
#            assert 'gof' in fitdf.columns, "-- doing evaluation --"
#        except Exception as e:
#            traceback.print_exc()
#            do_evaluation = True
#    else:
#        do_evaluation = True
#        
#    if do_evaluation:
#        print("Evaluating bootstrap tuning results")
#        #% # Plot visualizations for each cell's bootstrap iters for fit
#        rois_fit = fitdf['cell'].unique()
#        #n_rois_fit = len(rois_fit)
#        for roi in rois_fit:
#            fig = evaluate_fit_roi(roi, fitdf, fitparams, fitdata, response_type=response_type)
#            label_figure(fig, data_identifier)
#            pl.savefig(os.path.join(roi_evaldir, 'roi%05d.png' % int(roi+1)))
#            pl.close()
#        
#        fitdf = check_fit_quality(fitdf, fitdata)
#        evaluation_results = {'fits': fitdf}
#        with open(evaluate_fpath, 'wb') as f:
#            pkl.dump(evaluation_results, f, protocol=pkl.HIGHEST_PROTOCOL)
#        
#    goodfits = get_reliable_fits(fitdf, goodness_thr=goodness_thr)
#
#    n_rois_pass = len(goodfits)
#    n_rois_fit = len(fitdf['cell'].unique())
#    print("%i out of %i fit cells pass goodness-thr %.2f" % (n_rois_pass, n_rois_fit, goodness_thr))
#
#    if do_evaluation:
#        print("plotting some metrics...")
#        # Plot pairwise comparisons of metrics:
#        fig = compare_all_metrics_for_good_fits(fitdf, good_fits=goodfits)
#        label_figure(fig, data_identifier)
#        pl.savefig(os.path.join(evaldir, 'pairplot-all-metrics_goodness-thr%.2f.png' % goodness_thr))
#        pl.close()
#        
#        #% Plot tuning curves for all cells with good fits:
#        fig = plot_tuning_for_good_fits(fitdata, fitparams, good_fits=goodfits, plot_polar=True)
#        label_figure(fig, data_identifier)
#        fig.suptitle("Cells with GoF > %.2f" % goodness_thr)
#        figname = 'polar-tuning_goodness-of-fit%.2f_%irois' % (goodness_thr, len(goodfits))
#        pl.savefig(os.path.join(evaldir, '%s.png' % figname))
#        pl.close()
#    
#    return fitdf, goodfits



#%%


#def get_tuning_for_fov(animalid, session, fov, traceid='traces001', response_type='dff', 
#                          n_bootstrap_iters=1000, n_resamples=20, n_intervals_interp=3,
#                          responsive_test='ROC', responsive_thr=0.05, n_stds=2.5,
#                          make_plots=True, plot_metrics=True, return_iters=True,
#                          create_new=False, n_processes=1, rootdir='/n/coxfs01/2p-data',
#                          min_cfgs_above=2, min_nframes_above=10):
#    
#    fitdf = None
#    fitparams = None
#    fitdata = None
#    
#    run_name = get_gratings_run(animalid, session, fov, traceid=traceid, rootdir=rootdir)    
#    assert run_name is not None, "ERROR: [%s|%s|%s|%s] Unable to find gratings run..." % (animalid, session, fov, traceid)
#
#    # Select only responsive cells:
#    if responsive_test is not None:
##        roi_list, nrois_total = util.get_responsive_cells(animalid, session, fov, run=run_name, traceid=traceid, 
##                                        responsive_test=responsive_test, responsive_thr=responsive_thr, n_stds=n_stds,
##                                        rootdir=rootdir)
#        
#        roistats, roi_list, nrois_total = util.get_roi_stats(animalid, session, fov, 
#                                                             exp_name=run_name, 
#                                                             traceid=traceid, 
#                                                           response_type=response_type, 
#                                                           responsive_test=responsive_test, 
#                                                           responsive_thr=responsive_thr, n_stds=n_stds,
#                                                           rootdir=rootdir)
#        statdf = roistats['nframes_above']
#    else:
#        roi_list = None
#        statdf = None
#    
#    #% GET FITS:
#    fitdf, fitparams, fitdata = get_tuning(animalid, session, fov, run_name, return_iters=return_iters,
#                                           traceid=traceid, roi_list=roi_list, statdf=statdf, response_type=response_type,
#                                           n_bootstrap_iters=n_bootstrap_iters, n_resamples=n_resamples, 
#                                           n_intervals_interp=n_intervals_interp, make_plots=make_plots,
#                                           responsive_test=responsive_test, responsive_thr=responsive_thr, n_stds=n_stds,
#                                           create_new=create_new, rootdir=rootdir, n_processes=n_processes,
#                                           min_cfgs_above=min_cfgs_above, min_nframes_above=min_nframes_above)
#
#    print("... plotting comparison metrics ...")
#    roi_fitdir = fitparams['directory']
#    run_name = os.path.split(roi_fitdir.split('/traces')[0])[-1]
#    data_identifier = '|'.join([animalid, session, fov, run_name, traceid])
#
#    if plot_metrics:
#        plot_selectivity_metrics(fitdf, fitparams, fit_thr=0.9, data_identifier=data_identifier)
#        plot_top_asi_and_dsi(fitdf, fitparams, fit_thr=0.9, topn=10, data_identifier=data_identifier)
#    
#    print("*** done! ***")
#    
#    return fitdf, fitparams, fitdata


# In[21]:

# #############################################################################
# Plotting functions:
# #############################################################################

def cleanup_axes(axes_list, which_axis='y'):    
    for ax in axes_list: 
        if which_axis=='y':
            # get the yticklabels from the axis and set visibility to False
            for label in ax.get_yticklabels():
                label.set_visible(False)
            ax.yaxis.offsetText.set_visible(False)
        elif which_axis=='x':
            # get the xticklabels from the axis and set visibility to False
            for label in ax.get_xticklabels():
                label.set_visible(False)
            ax.xaxis.offsetText.set_visible(False)


def plot_psth_roi(roi, raw_traces, labels, curr_cfgs, sdf,  trace_type='dff', fig=None, nr=1, nc=1, s_row=0, colspan=1):
    if fig is None:
        fig = pl.figure()

    pl.figure(fig.number)
        
    # ---------------------------------------------------------------------
    #% plot raw traces:
    mean_traces, std_traces, tpoints = osi.get_mean_and_std_traces(roi, raw_traces, labels, curr_cfgs, sdf)
    
    stim_on_frame = labels['stim_on_frame'].unique()[0]
    nframes_on = labels['nframes_on'].unique()[0]
    
    ymin = (mean_traces - std_traces ).min()
    ymax = (mean_traces + std_traces ).max()
    for icfg in range(len(curr_cfgs)):
        ax = pl.subplot2grid((nr, nc), (s_row, icfg), colspan=colspan)
        ax.plot(tpoints, mean_traces[icfg, :], color='k')
        ax.set_xticks([tpoints[stim_on_frame], round(tpoints[stim_on_frame+nframes_on], 1)])
        ax.set_xticklabels(['', round(tpoints[stim_on_frame+nframes_on], 1)])
        ax.set_ylim([ymin, ymax])
        if icfg > 0:
            ax.set_yticks([]); ax.set_yticklabels([]);
            ax.set_xticks([]); ax.set_xticklabels([]);
            sns.despine(ax=ax, offset=4, trim=True, left=True, bottom=True)
        else:
            ax.set_ylabel(trace_type); ax.set_xlabel('time (s)');
            sns.despine(ax=ax, offset=4, trim=True)
        sem_plus = np.array(mean_traces[icfg,:]) + np.array(std_traces[icfg,:])
        sem_minus = np.array(mean_traces[icfg,:]) - np.array(std_traces[icfg,:])
        ax.fill_between(tpoints, sem_plus, y2=sem_minus, alpha=0.5, color='k')

    return fig, ax


def plot_tuning_curve_roi(curr_oris, curr_resps, curr_sems=None, response_type='dff',
                          fig=None, ax=None, nr=1, nc=1, colspan=1, s_row=0, s_col=0, color='k',
                         marker='o', lw=1, markersize=5):
    if fig is None:
        fig = pl.figure()
    
    pl.figure(fig.number)
        
    # Plot tuning curve:
    if ax is None:
        ax = pl.subplot2grid((nr, nc), (s_row, s_col), colspan=colspan)
    ax.plot(curr_oris, curr_resps, color=color, marker=marker, markersize=markersize, lw=lw)
    if curr_sems is not None:
        ax.errorbar(curr_oris, curr_resps, yerr=curr_sems, fmt='none', ecolor=color)
    ax.set_xticks(curr_oris)
    ax.set_xticklabels(curr_oris)
    ax.set_ylabel(response_type)
    #ax.set_title('(sz %i, sf %.2f)' % (best_cfg_params['size'], best_cfg_params['sf']), fontsize=8)
    #sns.despine(trim=True, offset=4, ax=ax)
    
    return fig, ax

def plot_tuning_polar_roi(curr_oris, curr_resps, curr_sems=None, response_type='dff',
                          fig=None, ax=None, nr=1, nc=1, colspan=1, s_row=0, s_col=0, 
                          color='k', linestyle='-', label=None, alpha=1.0):
    if fig is None:
        fig = pl.figure()
    
    pl.figure(fig.number)
    
    # Plot polar graph:
    if ax is None:
        ax = pl.subplot2grid((nr,nc), (s_row, s_col), colspan=colspan, polar=True)
    thetas = np.array([np.deg2rad(c) for c in curr_oris])
    radii = curr_resps.copy()
    thetas = np.append(thetas, np.deg2rad(curr_oris[0]))  # append first value so plot line connects back to start
    radii = np.append(radii, curr_resps[0]) # append first value so plot line connects back to start
    ax.plot(thetas, radii, '-', color=color, label=label, linestyle=linestyle, alpha=alpha)
    ax.set_theta_zero_location("N")
    ax.set_yticks([curr_resps.min(), curr_resps.max()])
    ax.set_yticklabels(['', round(curr_resps.max(), 1)])

    
    return fig, ax

#
#
#def plot_roi_tuning_raw_and_fit(roi, responses_df, curr_cfgs,
#                                raw_traces, labels, sdf, fit_results,
#                               trace_type='dff'):
#
#    fig = pl.figure(figsize=(12,8))
#    fig.patch.set_alpha(1)
#    nr=2; nc=8;
#    s_row=0
#    fig, ax = plot_psth_roi(roi, raw_traces, labels, curr_cfgs, sdf, 
#                            trace_type=trace_type,
#                            fig=fig, nr=nr, nc=nc, s_row=0)
#    ymin = np.min([0, ax.get_ylim()[0]])
#    ax.set_ylim([ymin, ax.get_ylim()[1]])
#
#
#    curr_oris = np.array([sdf['ori'][c] for c in curr_cfgs])  
#    curr_resps = responses_df.mean()
#    curr_sems = responses_df.sem()
#    fig, ax1 = plot_tuning_curve_roi(curr_oris, curr_resps, curr_sems=curr_sems, 
#                                     response_type=trace_type,
#                                     fig=fig, nr=nr, nc=nc, s_row=1, colspan=5,
#                                     marker='o', markersize=5, lw=0)
#
#
#    fig, ax2 = plot_tuning_polar_roi(curr_oris, curr_resps, curr_sems=curr_sems, 
#                                     response_type=trace_type,
#                                     fig=fig, nr=nr, nc=nc, s_row=1, s_col=6, colspan=2)
#
#
#    if fit_results is not None:
#        oris_interp = np.array([rfit['x'] for rfit in fit_results]).mean(axis=0)
#        resps_interp = np.array([rfit['y'] for rfit in fit_results]).mean(axis=0)
#        resps_interp_sem = spstats.sem(np.array([rfit['y'] for rfit in fit_results]), axis=0)
#        resps_fit = np.array([rfit['fit_y'] for rfit in fit_results]).mean(axis=0)
#        n_intervals_interp = rfit['n_intervals_interp']
#
#        fig, ax1 = plot_tuning_curve_roi(oris_interp[0:-n_intervals_interp], 
#                                         resps_fit[0:-n_intervals_interp], 
#                                         curr_sems=resps_interp_sem[0:-n_intervals_interp], 
#                                         response_type=trace_type,color='cornflowerblue',
#                                         markersize=0, lw=1, marker=None,
#                                         fig=fig, ax=ax1, nr=nr, nc=nc, s_row=1, colspan=5)
#
#        fig, ax2 = plot_tuning_polar_roi(oris_interp, 
#                                         resps_fit, 
#                                         curr_sems=resps_interp_sem, 
#                                         response_type=trace_type, color='cornflowerblue',
#                                         fig=fig, ax=ax2, nr=nr, nc=nc, s_row=1, s_col=6, colspan=2)
#        
#    ymin = np.min([0, ax1.get_ylim()[0]])
#    ax1.set_ylim([ymin,  ax1.get_ylim()[1]])
#    
#    ax1.set_yticks([ymin, ax1.get_ylim()[1]])
#    ax1.set_yticklabels([round(ymin, 2), round( ax1.get_ylim()[1], 2)])
#    sns.despine(trim=True, offset=4, ax=ax1)
#
#    
#    if any([rfit['success'] for rfit in fit_results]):
#        r2_avg = np.mean([rfit['r2'] for rfit in fit_results])
#        ax1.text(0, ax1.get_ylim()[-1]*0.75, 'r2=%.2f' % r2_avg, fontsize=6)
#    else:
#        ax1.text(0, ax.get_ylim()[-1]*0.75, 'no fit', fontsize=6)
#    
#    return fig, ax, ax1, ax2
#
#
#
#def plot_roi_tuning(roi, fitdata, sdf, df_traces, labels, trace_type='dff'):
#    #print("... plotting")
#    
#    #print("... Saving ROI fit plots to:\n%s" % roi_fitdir_figures)
#
#    responses_df = fitdata['original_data'][roi]['responses']
#    curr_cfgs = fitdata['original_data'][roi]['stimulus_configs']
#    best_cfg = best_cfg = fitdata['original_data'][roi]['responses'].mean().argmax()
#    
#    fig, ax, ax1, ax2 = plot_roi_tuning_raw_and_fit(roi, responses_df, curr_cfgs,
#                                                    df_traces, labels, sdf, fitdata['results_by_iter'][roi], trace_type=trace_type)
#    curr_oris = sorted(sdf['ori'].unique())
#    ax1.set_xticks(curr_oris)
#    ax1.set_xticklabels(curr_oris)
#    ax1.set_title('(sz %i, sf %.2f)' % (sdf['size'][best_cfg], sdf['sf'][best_cfg]), fontsize=8)
#
#    fig.suptitle('roi %i' % int(roi+1))
#    
#    return fig

# Summary plotting:

def compare_selectivity_all_fits(fitdf, fit_metric='gof', fit_thr=0.66):
    
    strong_fits = [r for r, v in fitdf.groupby(['cell']) if v.mean()[fit_metric] >= fit_thr] # check if average gof good
    print("%i out of %i cells with strong fits (%.2f)" % (len(strong_fits), len(fitdf['cell'].unique()), fit_thr))
    if len(strong_fits)==0:
        return pl.figure(), strong_fits
 
   
    df = fitdf[fitdf['cell'].isin(strong_fits)]
    df['cell'] = df['cell'].astype(str)
    
    g = sns.PairGrid(df, hue='cell', vars=['asi', 'dsi', 'r2comb'])
    g.fig.patch.set_alpha(1)
    
   
    g = g.map_offdiag(pl.scatter, marker='o',  alpha=0.5, s=1)
    
    
    g = g.map_diag(pl.hist, normed=True) #histtype="step",  
    
    #g.set(ylim=(0, 1))
    #g.set(xlim=(0, 1))
    #g.set(xticks=(0, 1))
    #g.set(yticks=(0, 1))
    #g.set(aspect='equal')
    
    #sns.distplot, kde=False, hist=True, rug=True,\
                   #hist_kws={"histtype": "step", "linewidth": 2, "alpha": 1.0})
    
    if df.shape[0] < 10:
        g = g.add_legend(bbox_to_anchor=(1.01,.5))
    
    pl.subplots_adjust(left=0.1, right=0.85)

    #cleanup_axes(g.axes[:, 1:].flat, which_axis='y')
    #cleanup_axes( g.axes[:-1, :].flat, which_axis='x')
    
    
    return g.fig, strong_fits


def sort_by_selectivity(fitdf, fit_metric='gof', fit_thr=0.66, topn=10):
    strong_fits = [r for r, v in fitdf.groupby(['cell']) if v.mean()[fit_metric] >= fit_thr]
    print("%i out of %i cells with strong fits (%.2f)" % (len(strong_fits), len(fitdf['cell'].unique()), fit_thr))
    if len(strong_fits)==0:
        return None, [], []
    
    df = fitdf[fitdf['cell'].isin(strong_fits)]
        
    #df.loc[:, 'cell'] = np.array([int(c) for c in df['cell'].values])
    
    top_asi = df.groupby(['cell']).mean().sort_values(['asi'], ascending=False)
    top_dsi = df.groupby(['cell']).mean().sort_values(['dsi'], ascending=False)
    #top_r2 = df.groupby(['cell']).mean().sort_values(['r2'], ascending=False)
    
    top_asi_cells = top_asi.index.tolist()[0:topn]
    top_dsi_cells = top_dsi.index.tolist()[0:topn]
    #top_r2_cells = top_r2.index.tolist()[0:topn]

    top10_asi = [roi for rank, roi in enumerate(top_asi.index.tolist()) if rank < topn]
    top10_dsi = [roi for rank, roi in enumerate(top_dsi.index.tolist()) if rank < topn]
    
    df.loc[:, 'top_asi'] = np.array([ roi if roi in top10_asi else -10 for roi in df['cell']])
    df.loc[:, 'top_dsi'] = np.array([ roi if roi in top10_dsi else -10 for roi in df['cell']])

    
    #% # Convert to str for plotting:
        
    df.loc[:, 'top_asi'] = [str(s) for s in df['top_asi'].values]
    df.loc[:, 'top_dsi'] = [str(s) for s in df['top_dsi'].values]

    
    return df, top_asi_cells, top_dsi_cells


def compare_topn_selective(df, color_by='ASI', palette='cubehelix'):
    
    hue = 'top_asi' if color_by in ['ASI', 'asi'] else 'top_dsi'
#    if color_by == 'ASI':
#        hue = 'top_asi'
#        palette = asi_colordict
#    elif color_by == 'DSI':
#        hue = 'top_dsi'
#        palette = dsi_colordict
#    

    g = sns.PairGrid(df, hue=hue, vars=['asi', 'dsi'], palette=palette, size=5)#,
                    #hue_kws={"alpha": alphadict.values()}) # 'cubehelix_r') #'cubehelix') #'')
    
    g.fig.patch.set_alpha(1)
    g = g.map_offdiag(pl.scatter, marker='o', s=5, alpha=0.7) #, color=[asi_colordict[r] for r in ddf[hue]]) #=alphadict.values()[::-1]) #,  alpha=0.5, s=5, )
    
    
    g = g.map_diag(pl.hist, normed=True, alpha=0.5) #histtype="step",  
    
    
    g.set(ylim=(0, 1))
    g.set(xlim=(0, 1))
    
    g = g.add_legend(bbox_to_anchor=(1.0,0.2))
    for li, lh in enumerate(g._legend.legendHandles): 
        if not all([round(l, 1)==0.5 for l in lh.get_facecolor()[0][0:3]]):
            
            lh.set_alpha(1)
            lh._sizes = [20] 
        
    pl.subplots_adjust(left=0.15, right=0.8, top=0.9, bottom=0.1, wspace=0.2, hspace=0.2)
    
    #g.set(xlim=(0,1), ylim=(0,1))
    #g.set(xticks=[0, 1])
    #g.set(yticks=[0, 1])
    #sns.despine(trim=True)
            
    cleanup_axes(g.axes[:, 1:].flat, which_axis='y')
    cleanup_axes( g.axes[:-1, :].flat, which_axis='x')
    
    
    return g.fig


    
def plot_bootstrapped_params(fitdf, fitparams, fit_metric='gof', fit_thr=0.66, data_identifier='METADATA'):
    
    roi_fitdir = fitparams['directory']
    
    #% PLOT -- plot ALL fits:
    if 'ASI_cv' in fitdf.columns.tolist():
        fitdf['ASI_cv'] = [1-f for f in fitdf['ASI_cv'].values] # Revert to make 1 = very tuned, 0 = not tuned
    if 'DSI_cv' in fitdf.columns.tolist():
        fitdf['DSI_cv'] = [1-f for f in fitdf['DSI_cv'].values] # Revert to make 1 = very tuned, 0 = not tuned
    
    fig, strong_fits = compare_selectivity_all_fits(fitdf, fit_metric=fit_metric, fit_thr=fit_thr)
    label_figure(fig, data_identifier)
    
    #nrois_fit = len(fitdf['cell'].unique())
    #nrois_thr = len(strong_fits)
    n_bootstrap_iters = fitparams['n_bootstrap_iters']
    n_intervals_interp = fitparams['n_intervals_interp']
    
    figname = 'compare-bootstrapped-params_tuning-fit-thr%.2f_bootstrap-%iiters-interp%i' % (fit_thr, n_bootstrap_iters, n_intervals_interp)
    
    pl.savefig(os.path.join(roi_fitdir, '%s.png' % figname))
    pl.close()
    print("Saved:\n%s" % os.path.join(roi_fitdir, '%s.png' % figname))
    
    
def plot_top_asi_and_dsi(fitdf, fitparams, fit_metric='gof', fit_thr=0.66, topn=10, data_identifier='METADATA'):
    #%
    # ##### Compare metrics
    
    # Sort cells by ASi and DSi    
    df, top_asi_cells, top_dsi_cells = sort_by_selectivity(fitdf, fit_metric=fit_metric, fit_thr=fit_thr, topn=topn)
    if df is None:
        return
 
    #% Set color palettes:
    palette = sns.color_palette('cubehelix', len(top_asi_cells))
    main_alpha = 0.8
    sub_alpha = 0.01
    asi_colordict = dict(( str(roi), palette[i]) for i, roi in enumerate(top_asi_cells))
    for k, v in asi_colordict.items():
        asi_colordict[k] = (v[0], v[1], v[2], main_alpha)
        
    dsi_colordict = dict(( str(roi), palette[i]) for i, roi in enumerate(top_dsi_cells))
    for k, v in dsi_colordict.items():
        dsi_colordict[k] = (v[0], v[1], v[2], main_alpha)
          
    asi_colordict.update({ str(-10): (0.8, 0.8, 0.8, sub_alpha)})
    dsi_colordict.update({ str(-10): (0.8, 0.8, 0.8, sub_alpha)})
        
    
    #% PLOT by ASI:
    roi_fitdir = fitparams['directory']
    n_bootstrap_iters = fitparams['n_bootstrap_iters']
    nrois_fit = len(fitdf['cell'].unique())
    nrois_pass = len(df['cell'].unique())
    
    color_by = 'asi'
    palette = asi_colordict if color_by=='asi' else dsi_colordict
            
    fig = compare_topn_selective(df, color_by=color_by, palette=palette)
    label_figure(fig, data_identifier)
    
    figname = 'sort-by-%s_top%i_tuning-fit-thr%.2f_bootstrap-%iiters_%iof%i' % (color_by, topn, fit_thr, n_bootstrap_iters, nrois_pass, nrois_fit)

    pl.savefig(os.path.join(roi_fitdir, '%s.png' % figname))
    pl.close()
    
    #% Color by DSI:
    color_by = 'dsi'
    palette = asi_colordict if color_by=='asi' else dsi_colordict

    fig = compare_topn_selective(df, color_by=color_by, palette=palette)
    label_figure(fig, data_identifier)
    figname = 'sort-by-%s_top%i_tuning-fit-thr%.2f_bootstrap-%iiters_%iof%i' % (color_by, topn, fit_thr, n_bootstrap_iters, nrois_pass, nrois_fit)

    pl.savefig(os.path.join(roi_fitdir, '%s.png' % figname))
    pl.close()
    
    



#%%

#% Evaluation -- plotting

def compare_all_metrics_for_good_fits(fitdf, good_fits=None):
    
    if good_fits is not None:
        df = fitdf[fitdf['cell'].isin(good_fits)]
    else:
        df = fitdf.copy()
    df['cell'] = df['cell'].astype(str)
    
    g = sns.PairGrid(df, hue='cell', vars=[c for c in fitdf.columns.tolist() if c != 'cell'], palette='cubehelix')
    g.fig.patch.set_alpha(1)
    g = g.map_offdiag(pl.scatter, marker='o', s=5, alpha=0.7)
    g = g.map_diag(pl.hist, normed=True, alpha=0.5) #histtype="step",  
    
    #g.set(ylim=(0, 1))
    #g.set(xlim=(0, 1))
    #g.set(aspect='equal')
    
    return g.fig
    

#%%


def get_non_ori_params(sdf):

    cfgs = list(itertools.product(*[sdf['sf'].unique(), sdf['size'].unique(), sdf['speed'].unique()]))

    return cfgs

def roi_polar_plot_by_config(bootresults, fitparams, gof_thr=0.66, plot_polar=True):
    rmetrics, rmetrics_by_cfg = get_good_fits(bootresults, fitparams, gof_thr=gof_thr)
    if rmetrics is None:
        print("No good rois!")
        return None

    goodrois = rmetrics.index.tolist()
    
    n_intervals_interp = fitparams['n_intervals_interp']

    n_rois_pass = len(goodrois)
    nr = int(np.ceil(np.sqrt(n_rois_pass))) + 1 # add extra row for legend
    nc = int(np.ceil(float(n_rois_pass) / nr))
    
    cfgs = [tuple(c) for c in fitparams['non_ori_configs']]
    colors = sns.color_palette(palette='cubehelix', n_colors=len(cfgs))
    
    
    fig, axes = pl.subplots(nr, nc, figsize=(nr*2,nc*2), subplot_kw=dict(polar=True))
    
    for ax, (roi, g) in zip(axes.flat, rmetrics_by_cfg.groupby(['cell'])):
        allgofs = []
        for skey in g.index.tolist():
            stimparam = tuple(float(i) for i in skey.split('-')[1::2])
            si = cfgs.index(stimparam)
            bootr = bootresults[roi][stimparam]
            
            if plot_polar:
                thetas_interp = bootr['fits']['xv']
                thetas = bootr['fits']['xv'][0::n_intervals_interp]
            else:
                thetas_interp = bootr['fits']['xv'][0:-1]
                thetas = bootr['fits']['xv'][0::n_intervals_interp][0:-1]
                
            # Get combined tuning across iters for current stim config
            #params = [c for c in bootr['results'].columns if 'stim' not in c]
            #avg_metrics = average_metrics_across_iters(bootr['results'][params])
            r2comb, gof, fitr = evaluate_fits(bootr, interp=True)
            origr = bootr['data']['responses'].mean(axis=0).values
    
            if plot_polar:
                origr = np.append(origr, origr[0]) # wrap back around
                plot_tuning_polar_roi(thetas, origr, curr_sems=None, response_type='dff',
                                          fig=fig, ax=ax, color=colors[si], linestyle='--')
        
                plot_tuning_polar_roi(thetas_interp, fitr, curr_sems=None, response_type='dff',
                                          fig=fig, ax=ax, color=colors[si], linestyle='-', alpha=0.8,
                                          label='gof %.2f\ndff %.2f' % (gof, origr.max()) )
            allgofs.append(gof)
            
        ax.set_title('%i (GoF: %.2f)' % (int(roi), np.mean(allgofs)), fontsize=6, y=1)
        ax.legend(bbox_to_anchor=(0, 1), loc='upper right', ncol=1, fontsize=6)
        ax.yaxis.grid(False)
        ax.yaxis.set_ticklabels([])
        #ax.tick_params(pad=1)
        
        ax.xaxis.grid(True)
        ax.xaxis.set_ticklabels([])
        #thetaticks = np.arange(0,360,45)
        #ax.tick_params(pad=0.2)
        
    for ax in axes.flat[len(goodrois):]:
        ax.axis('off')
    pl.subplots_adjust(hspace=0.3, wspace=0.8)
    
    
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color=colors[i], lw=4) for i, k in enumerate(cfgs)]
    custom_labels = ['sf-%.2f, sz-%i, speed-%i' % k for i, k in enumerate(cfgs)]
    ax = axes.flat[-1]
    ax.legend(custom_lines, custom_labels, ncol=2, fontsize=8)

    return fig

#
#def plot_tuning_for_good_fits(fitdata, fitparams, good_fits=[], plot_polar=True):
#    n_intervals_interp = fitparams['n_intervals_interp']
#    if plot_polar:
#        thetas_interp = fitparams['interp_values']
#        thetas = thetas_interp[0::n_intervals_interp]
#    else:
#        thetas_interp = fitparams['interp_values'][0:-1]
#        thetas = thetas_interp[0::n_intervals_interp][0:-1]
#        
#    n_rois_pass = len(good_fits)
#    nr = int(np.ceil(np.sqrt(n_rois_pass)))
#    nc = int(np.ceil(float(n_rois_pass) / nr))
#
#
#    fig, axes = pl.subplots(nr, nc, figsize=(nr*2,nc*2), subplot_kw=dict(polar=True))
#    
#    for ax, roi in zip(axes.flat, good_fits):
#        
#        responses = fitdata['original_data'][roi]['responses']
#        origr = responses.mean().values
#    
#        popt = get_average_params_over_iters(fitdata['results_by_iter'][roi])
#        
#        fitr = double_gaussian(thetas_interp, *popt)
#        
#        if plot_polar:
#            origr = np.append(origr, origr[0]) # wrap back around
#            plot_tuning_polar_roi(thetas, origr, curr_sems=None, response_type='dff',
#                                      fig=fig, ax=ax, color='k')
#    
#            plot_tuning_polar_roi(thetas_interp, fitr, curr_sems=None, response_type='dff',
#                                      fig=fig, ax=ax, color='cornflowerblue')
#            
#        ax.set_title('%i' % int(roi), fontsize=6, y=1)
#        
#        ax.yaxis.grid(False)
#        ax.yaxis.set_ticklabels([])
#        
#        ax.xaxis.grid(True)
#        ax.xaxis.set_ticklabels([])
#        #thetaticks = np.arange(0,360,45)
#        #ax.tick_params(pad=0.2)
#    
#    for ax in axes.flat[len(good_fits):]:
#        ax.axis('off')
#        
#    pl.subplots_adjust(hspace=0.3, wspace=0.3)
#    
#    return fig
#

#%%

#
#def plot_tuning_for_good_fits(fitdata, fitparams, good_fits=[], plot_polar=True):
#    n_intervals_interp = fitparams['n_intervals_interp']
#    if plot_polar:
#        thetas_interp = fitparams['interp_values']
#        thetas = thetas_interp[0::n_intervals_interp]
#    else:
#        thetas_interp = fitparams['interp_values'][0:-1]
#        thetas = thetas_interp[0::n_intervals_interp][0:-1]
#        
#    n_rois_pass = len(good_fits)
#    nr = int(np.ceil(np.sqrt(n_rois_pass)))
#    nc = int(np.ceil(float(n_rois_pass) / nr))
#        
#    fig, axes = pl.subplots(nr, nc, figsize=(nr*2,nc*2), subplot_kw=dict(polar=True))
#    
#    for ax, roi in zip(axes.flat, good_fits):
#        
#        responses = fitdata['original_data'][roi]['responses']
#        origr = responses.mean().values
#    
#        popt = get_average_params_over_iters(fitdata['results_by_iter'][roi])
#        
#        fitr = double_gaussian(thetas_interp, *popt)
#        
#        if plot_polar:
#            origr = np.append(origr, origr[0]) # wrap back around
#            plot_tuning_polar_roi(thetas, origr, curr_sems=None, response_type='dff',
#                                      fig=fig, ax=ax, color='k')
#    
#            plot_tuning_polar_roi(thetas_interp, fitr, curr_sems=None, response_type='dff',
#                                      fig=fig, ax=ax, color='cornflowerblue')
#            
#        ax.set_title('%i' % int(roi), fontsize=6, y=1)
#        
#        ax.yaxis.grid(False)
#        ax.yaxis.set_ticklabels([])
#        
#        ax.xaxis.grid(True)
#        ax.xaxis.set_ticklabels([])
#        #thetaticks = np.arange(0,360,45)
#        #ax.tick_params(pad=0.2)
#    
#    for ax in axes.flat[len(good_fits):]:
#        ax.axis('off')
#        
#    pl.subplots_adjust(hspace=0.3, wspace=0.3)
#    
#    return fig
#
#
##%


#%%




def extract_options(options):
    parser = optparse.OptionParser()

    # PATH opts:
    parser.add_option('-D', '--root', action='store', dest='rootdir', default='/n/coxfs01/2p-data', 
                      help='root project dir containing all animalids [default: /n/coxfs01/2pdata]')
    parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', 
                      help='Animal ID')
    parser.add_option('-S', '--session', action='store', dest='session', default='', 
                      help='Session (format: YYYYMMDD)')

    # Set specific session/run for current animal:
    parser.add_option('-A', '--fov', action='store', dest='fov', default='FOV1_zoom2p0x', 
                      help="fov name (default: FOV1_zoom2p0x)")
    
    parser.add_option('-t', '--traceid', action='store', dest='traceid', default='traces001', 
                      help="traceid (default: traces001)")
    
    # Responsivity params:
     # Responsivity params:
    choices_resptest = ('ROC','nstds', None)
    default_resptest = None
    
    parser.add_option('-R', '--response-test', type='choice', choices=choices_resptest,
                      dest='responsive_test', default=default_resptest, 
                      help="Stat to get. Valid choices are %s. Default: %s" % (choices_resptest, str(default_resptest)))
    parser.add_option('-f', '--responsive-thr', action='store', dest='responsive_thr', default=0.05, 
                      help="responsive test threshold (default: p<0.05 for responsive_test=ROC)")
    parser.add_option('-s', '--n-stds', action='store', dest='n_stds', default=2.5, 
                      help="[n_stds only] n stds above/below baseline to count frames, if test=nstds (default: 2.5)") 
    parser.add_option('-m', '--min-frames', action='store', dest='min_nframes_above', default=10, 
                      help="[n_stds only] Min N frames above baseline std (responsive_thr), if responsive_test=nstds (default: 10)")   
    parser.add_option('-c', '--min-configs', action='store', dest='min_cfgs_above', default=2, 
                      help="[n_stds only] Min N configs in which min-n-frames threshold is met, if responsive_test=nstds (default: 2)")   

    # Tuning params:
    parser.add_option('-b', '--iter', action='store', dest='n_bootstrap_iters', default=1000, 
                      help="N bootstrap iterations (default: 1000)")
    parser.add_option('-k', '--samples', action='store', dest='n_resamples', default=20, 
                      help="N trials to sample w/ replacement (default: 20)")
    parser.add_option('-p', '--interp', action='store', dest='n_intervals_interp', default=3, 
                      help="N intervals to interp between tested angles (default: 3)")
    
    parser.add_option('-d', '--response-type', action='store', dest='response_type', default='dff', 
                      help="Trial response measure to use for fits (default: dff)")
    parser.add_option('-n', '--n-processes', action='store', dest='n_processes', default=1, 
                      help="N processes (default: 1)")

    parser.add_option('-G', '--goodness-thr', action='store', dest='goodness_thr', default=0.66, 
                      help="Goodness-of-fit threshold (default: 0.66)")


    parser.add_option('--new', action='store_true', dest='create_new', default=False, 
                      help="Create all session objects from scratch")

    (options, args) = parser.parse_args(options)
#    
#    if len(options.visual_areas) == 0:
#        options.visual_areas = ['V1', 'Lm', 'Li']

    return options

#%%

#rootdir = '/n/coxfs01/2p-data'
#animalid = 'JC084' 
#session = '20190522' #'20190319'
#fov = 'FOV1_zoom2p0x' 
#run = 'combined_gratings_static'
#traceid = 'traces001' #'traces002'
##trace_type = 'corrected'
#
#response_type = 'dff'
##metric_type = 'dff'
#make_plots = True
#n_bootstrap_iters = 100
#n_intervals_interp = 3
#
#responsive_test = 'ROC'
#responsive_thr = 0.05
#


def bootstrap_tuning_curves_and_evaluate(animalid, session, fov, traceid='traces001', response_type='dff',
                                         responsive_test='ROC', responsive_thr=0.05, n_stds=2.5,
                                         n_bootstrap_iters=1000, n_resamples=20,
                                         n_intervals_interp=3, goodness_thr = 0.66,
                                         n_processes=1, create_new=False, rootdir='/n/coxfs01/2p-data',
                                         min_cfgs_above=2, min_nframes_above=10, make_plots=True):

    from pipeline.python.classifications import experiment_classes as util

    
    run_name = get_gratings_run(animalid, session, fov, traceid=traceid, rootdir=rootdir)    
    assert run_name is not None, "ERROR: [%s|%s|%s|%s] Unable to find gratings run..." % (animalid, session, fov, traceid)

    bootresults, fitparams = get_tuning(animalid, session, fov, run_name,
                                         traceid=traceid, response_type=response_type, 
                                         n_bootstrap_iters=int(n_bootstrap_iters), 
                                         n_resamples = int(n_resamples),
                                         n_intervals_interp=int(n_intervals_interp),
                                         responsive_test=responsive_test, responsive_thr=responsive_thr, n_stds=n_stds,
                                         create_new=create_new, n_processes=n_processes, rootdir=rootdir,
                                         min_cfgs_above=min_cfgs_above, min_nframes_above=min_nframes_above, make_plots=make_plots)

    fit_desc = os.path.split(fitparams['directory'])[-1]
    print("----- COMPLETED 1/2: bootstrap tuning! ------")

    rmetrics, rmetrics_by_cfg = evaluate_tuning(animalid, session, fov, run_name, 
                                                traceid=traceid, fit_desc=fit_desc, gof_thr=goodness_thr,
                                                create_new=create_new, rootdir=rootdir, plot_metrics=make_plots)
   
    if rmetrics is None:
        n_goodcells = 0
    else:
        n_goodcells = len(rmetrics.index.tolist())
 
    print("----- COMPLETED 2/2: evaluation (%i good cells)! -----" % n_goodcells)
    
    return bootresults, fitparams, rmetrics, rmetrics_by_cfg

#%%

def main(options):
    opts = extract_options(options)
    rootdir = opts.rootdir
    animalid = opts.animalid
    session = opts.session
    fov = opts.fov
    traceid = opts.traceid
    response_type = opts.response_type
    n_bootstrap_iters = int(opts.n_bootstrap_iters)
    n_resamples = int(opts.n_resamples)
    n_intervals_interp = int(opts.n_intervals_interp)
    responsive_test = opts.responsive_test
    responsive_thr = float(opts.responsive_thr)
    n_stds = float(opts.n_stds)
    n_processes = int(opts.n_processes)
    min_nframes_above = int(opts.min_nframes_above)
    min_cfgs_above = int(opts.min_cfgs_above)
    create_new = opts.create_new
    goodness_thr = float(opts.goodness_thr)
    
    bootresults, fitparams, rmetrics, rmetrics_by_cfg = bootstrap_tuning_curves_and_evaluate(
                                             animalid, session, fov, 
                                             traceid=traceid, response_type=response_type, 
                                             n_bootstrap_iters=n_bootstrap_iters, 
                                             n_resamples=n_resamples,
                                             n_intervals_interp=n_intervals_interp,
                                             responsive_test=responsive_test, responsive_thr=responsive_thr, n_stds=n_stds,
                                             create_new=create_new, n_processes=n_processes, rootdir=rootdir,
                                             min_nframes_above=min_nframes_above, min_cfgs_above=min_cfgs_above,
                                             goodness_thr=goodness_thr)
    print("***** DONE *****")
    
if __name__ == '__main__':
    main(sys.argv[1:])
    


#%%

options = ['-i', 'JC084', '-S', '20190522', '-t', 'traces001', '-R', 'nstds', '-f', 10, '-s', 2.5]


# In[429]: ALL CELLS:



#g = sns.PairGrid(fitdf, hue='cell', vars=['ASI', 'DSI', 'r2'])
#
#g = g.map_diag(sns.distplot, kde=False, hist=True, rug=True,
#               hist_kws={"histtype": "step", "linewidth": 2, "alpha": 1.0})
#g = g.map_offdiag(pl.scatter, marker='+')
#
#g.set(xlim=(0,1), ylim=(0,1))
#g.set(xticks=[0, 1])
#g.set(yticks=[0, 1])
#sns.despine(trim=True)
#
#cleanup_axes(g.axes[:, 1:].flat, which_axis='y')
#cleanup_axes( g.axes[:-1, :].flat, which_axis='x')
#

