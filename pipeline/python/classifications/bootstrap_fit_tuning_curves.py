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

import pylab as pl
from collections import Counter
import seaborn as sns
import cPickle as pkl
import numpy as np
import pylab as pl
import pandas as pd
import seaborn as sns
import tifffile as tf


from pipeline.python.classifications import osi_dsi as osi
from pipeline.python.classifications import test_responsivity as resp
from pipeline.python.classifications import utils as util
from pipeline.python.utils import natural_keys, label_figure

#from pipeline.python.retinotopy import fit_2d_rfs as rf

from pipeline.python.utils import uint16_to_RGB
from skimage import exposure
from matplotlib import patches

from scipy import stats as spstats
from scipy.interpolate import interp1d
import scipy.optimize as spopt

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


def fit_direction_selectivity(x, y, init_params=[0, 0, 0, 0, 0], bounds=[np.inf, np.inf, np.inf, np.inf, np.inf]):
    roi_fit = None
    
    popt, pcov = spopt.curve_fit(double_gaussian, x, y, p0=init_params, maxfev=1000, bounds=bounds)
    fitr = double_gaussian( x, *popt)
        
    # Get residual sum of squares 
    residuals = y - fitr
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - (ss_res / ss_tot)
    
    if pcov.max() == np.inf: # or r2 == 1: #round(r2, 3) < 0.15 or 
        success = False
    else:
        success = True
        
    if success:
        roi_fit = {'pcov': pcov,
                     'popt': popt,
                     'fit_y': fitr,
                     'r2': r2,
                     #'x': x,
                     #'y': y,
                     'init': init_params,
                     'success': success}
    return roi_fit


def interp_values(response_vector, n_intervals=3, wrap_value=None):
    resps_interp = []
    rvectors = copy.copy(response_vector)
    if wrap_value is not None:
        rvectors = np.append(response_vector, wrap_value)
    for orix, rvec in enumerate(rvectors[0:-1]):
        if rvec == rvectors[-2]:
            resps_interp.extend(np.linspace(rvec, rvectors[orix+1], endpoint=True, num=n_intervals+1))
        else:
            resps_interp.extend(np.linspace(rvec, rvectors[orix+1], endpoint=False, num=n_intervals))          
    return resps_interp




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

#%



def load_tuning_results(traceid_dir, fit_desc='', return_iters=True):

    fitdf=None; fitparams=None; fitdata=None;
    roi_fitdir = os.path.join(traceid_dir, 'tuning', fit_desc)

    tuning_fit_results_path = os.path.join(roi_fitdir, 'tuning_bootstrap_results.pkl')
    params_info_path = os.path.join(roi_fitdir, 'tuning_bootstrap_params.json')
    fit_data_path = os.path.join(roi_fitdir, 'tuning_bootstrap_data.pkl')
    
    if os.path.exists(tuning_fit_results_path):
        print("Loading existing fits.")
        with open(tuning_fit_results_path, 'rb') as f:
            fitdf = pkl.load(f)
        with open(params_info_path, 'r') as f:
            fitparams = json.load(f)
    else:
        print "NO FITS FOUND: %s" % fit_desc
        
    if return_iters:
        if os.path.exists(fit_data_path):
            with open(fit_data_path, 'r') as f:
                fitdata = pkl.load(f)
        return fitdf, fitparams, fitdata
    else:
        return fitdf, fitparams

def save_tuning_results(fitdf, fitparams, fitdata):
    roi_fitdir = fitparams['directory']
    tuning_fit_results_path = os.path.join(roi_fitdir, 'tuning_bootstrap_results.pkl')
    params_info_path = os.path.join(roi_fitdir, 'tuning_bootstrap_params.json')
    fit_data_path = os.path.join(roi_fitdir, 'tuning_bootstrap_data.pkl')
    
    with open(tuning_fit_results_path, 'wb') as f:
        pkl.dump(fitdf, f, protocol=pkl.HIGHEST_PROTOCOL)

    
    with open(fit_data_path, 'wb') as f:
        pkl.dump(fitdata, f, protocol=pkl.HIGHEST_PROTOCOL)
    
    # Save params:
    with open(params_info_path, 'w') as f:
        json.dump(fitparams, f, indent=4, sort_keys=True)

    print("Saved!")
    return roi_fitdir


def get_fit_desc(response_type='dff', responsive_test=None, responsive_thr=0.05):
    if responsive_test is None:
        fit_desc = 'bootstrap-fit-%s_all-cells' % (response_type) #, responsive_test, responsive_thr)
    else:
        fit_desc = 'bootstrap-fit-%s_responsive-%s-thr%.2f' % (response_type, responsive_test, responsive_thr)

    return fit_desc

#%%
import multiprocessing as mp
import itertools

def bootstrap_roi_responses(roi_df, sdf, response_type='dff',
                            n_bootstrap_iters=1000, n_resamples=60, 
                            n_intervals_interp=3):
    bootstrapfits = []
    roi = roi_df.index[0]
    
    constant_params = ['aspect', 'luminance', 'position', 'stimtype']
    params = [c for c in sdf.columns if c not in constant_params]
    stimdf = sdf[params]
    tested_oris = sdf['ori'].unique()

    # Find best config:
    best_cfg = roi_df.groupby(['config']).mean()[response_type].idxmax()
    best_cfg_params = stimdf.loc[best_cfg][[p for p in params if p!='ori']]
    curr_cfgs = sorted([c for c in stimdf.index.tolist() \
                        if all(stimdf.loc[c][[p for p in params if p!='ori']] == best_cfg_params)],\
                        key = lambda x: stimdf['ori'][x])


    # Get all trials of current set of cfgs:
    trialdf = roi_df[roi_df['config'].isin(curr_cfgs)]
    rdf = trialdf[['config', 'trial', response_type]]
    grouplist = [group_configs(group, response_type) for config, group in rdf.groupby(['config'])]
    responses_df = pd.concat(grouplist, axis=1)

    # Bootstrap distN of responses (rand w replacement):
    bootdf = [responses_df.sample(n_resamples, replace=True).mean(axis=0) for ni in range(n_bootstrap_iters)]
    bootstrapped_df = pd.concat(bootdf, axis=1)
    bootstrapped_df.index = [sdf['ori'][c] for c in bootstrapped_df.index]

    # Find init params for tuning fits and set fit constraints:
    init_params = get_init_params(bootstrapped_df[0])
    r_pref, r_null, theta_pref, sigma, r_offset = init_params
    init_bounds = ([0, 0, -np.inf, sigma/2., -r_pref], [3*r_pref, 3*r_pref, np.inf, np.inf, r_pref])

    # Interpolate values for finer steps:
    asi=[];dsi=[];r2=[]; preferred_theta=[];
    #circvar_asi=[]; circvar_dsi=[];
    for niter in bootstrapped_df.columns:
        oris_interp = interp_values(tested_oris, n_intervals=n_intervals_interp, wrap_value=360)
        resps_interp = interp_values(bootstrapped_df[niter], n_intervals=n_intervals_interp, wrap_value=bootstrapped_df[niter][0])


        init_params = get_init_params(bootstrapped_df[niter])
        r_pref, r_null, theta_pref, sigma, r_offset = init_params
        init_bounds = ([0, 0, -np.inf, sigma/2., -r_pref], [3*r_pref, 3*r_pref, np.inf, np.inf, r_pref])

        success = True
        try:
            rfit = fit_direction_selectivity(oris_interp, resps_interp, init_params, bounds=init_bounds)
            asi_t = get_ASI(rfit['fit_y'][0:], oris_interp[0:])
            dsi_t = get_DSI(rfit['fit_y'][0:], oris_interp[0:])
            #circvar_asi_t, circvar_dsi_t = get_circular_variance(rfit['fit_y'][0:], oris_interp[0:])

            asi.append(asi_t)
            dsi.append(dsi_t)
            preferred_theta.append(rfit['popt'][2])
            #circvar_asi.append(circvar_asi_t)
            #circvar_dsi.append(circvar_dsi_t)
            r2.append(rfit['r2'])

            rfit['x'] = oris_interp
            rfit['y'] = resps_interp
            rfit['n_intervals_interp'] = n_intervals_interp

            bootstrapfits.append(rfit)

        except Exception as e:
            #print(e)
            success = False
            #print("... skipping %i" % roi)
            break
        
    roi_fitdf = pd.DataFrame({'ASI': asi,
                              'DSI': dsi,
                              'r2': r2,
                              #'ASI_cv': circvar_asi,
                              #'DSI_cv': circvar_dsi,
                              'preferred_theta': preferred_theta,
                              'cell': [roi for _ in np.arange(0, len(asi))]})

    
    origdata = {'responses': responses_df, 
                 'stimulus_configs': curr_cfgs,
                 'init_params': init_params}


    return {'fitdata': {'results_by_iter': bootstrapfits, 
                        'original_data': origdata},
            'fitdf': roi_fitdf}

def boot_star(a_b):
    """Convert `f([1,2])` to `f(1,2)` call."""
    return bootstrap_roi_responses(*a_b)

def pool_bootstrap(rdf_list, sdf, n_processes=1):
    pool = mp.Pool(processes=n_processes)
    results = pool.map(boot_star, itertools.izip(rdf_list, itertools.repeat(sdf)))
    return results

#%
    
def do_bootstrap_fits(gdf, sdf, roi_list=None, 
                        response_type='dff',
                        n_bootstrap_iters=100, n_resamples=60,
                        n_intervals_interp=3, n_processes=1):

    if roi_list is None:
        roi_list = np.arange(0, len(gdf.groups))

    results = pool_bootstrap([gdf.get_group(roi) for roi in roi_list], sdf, n_processes=n_processes)
    tested_values = sdf['ori'].unique()
    interp_values = results[0]['fitdata']['results_by_iter'][0]['x']
    

    fitdf = pd.concat([res['fitdf'] for res in results])


    fitparams = {'n_bootstrap_iters': n_bootstrap_iters,
                'n_intervals_interp': n_intervals_interp,
                'tested_values': list(tested_values),
                'interp_values': interp_values,
                'roi_list': [r for r in roi_list]}    
        
    fitdata = {'results_by_iter': dict((res['fitdf']['cell'].unique()[0], res['fitdata']['results_by_iter'])\
                                       for res in results if len(res['fitdf']['cell'].unique())>0),
               'original_data': dict((res['fitdf']['cell'].unique()[0], res['fitdata']['original_data'])\
                                     for res in results if len(res['fitdf']['cell'].unique()) > 0)}
    
    #-------------------------

    return fitdf, fitparams, fitdata


def get_tuning(animalid, session, fov, run_name, return_iters=False,
               traceid='traces001', roi_list=None, response_type='dff',
               n_bootstrap_iters=100, n_resamples=60, n_intervals_interp=3,
               make_plots=True, responsive_test='ROC', responsive_thr=0.05,
               create_new=False, rootdir='/n/coxfs01/2p-data', n_processes=1):

    # Do fits:
    traceid_dir =  glob.glob(os.path.join(rootdir, animalid, session, fov, run_name, 'traces', '%s*' % traceid))[0]
    data_fpath = glob.glob(os.path.join(traceid_dir, 'data_arrays', 'datasets.npz'))[0]

    fit_desc = get_fit_desc(response_type=response_type, responsive_test=responsive_test, responsive_thr=responsive_thr)
    data_identifier = '|'.join([animalid, session, fov, run_name, traceid, fit_desc])

    roi_fitdir = os.path.join(traceid_dir, 'tuning', fit_desc)
    if not os.path.exists(roi_fitdir):
        os.makedirs(roi_fitdir)
        
    if create_new is False:
        fitdf, fitparams, fitdata = load_tuning_results(traceid_dir, fit_desc=fit_desc, return_iters=return_iters)
        do_fits = fitdf is not None
    else:
        do_fits=True
    
        
    if do_fits:
        print("Loading data and doing fits")
        
        df_traces, labels, gdf, sdf = load_gratings_data(data_fpath)
        
        if roi_list is None:
            roi_list = np.arange(0, len(gdf.groups))
    
        #% # Fit all rois in list
        print "... Fitting %i rois:" % len(roi_list), roi_list

        # Save fitso_bootstrap_fits
        fitdf, fitparams, fitdata = do_bootstrap_fits(gdf, sdf, roi_list=roi_list, 
                                                     n_bootstrap_iters=n_bootstrap_iters, 
                                                     n_resamples=n_resamples,
                                                     n_intervals_interp=n_intervals_interp, n_processes=n_processes)
        
        fitparams.update({'directory': roi_fitdir,
                          'response_type': response_type})
                        
        save_tuning_results(fitdf, fitparams, fitdata)
            
        if make_plots:
            if not os.path.exists(os.path.join(roi_fitdir, 'roi_fits')):
                os.makedirs(os.path.join(roi_fitdir, 'roi_fits'))
            print("Saving roi tuning fits to: %s" % os.path.join(roi_fitdir, 'roi_fits'))
            for roi in fitdf['cell'].unique():
                fig = plot_roi_tuning(roi, fitdata, sdf, df_traces, labels, roi_fitdir=roi_fitdir,
                                      trace_type=response_type)
                label_figure(fig, data_identifier)
                pl.savefig(os.path.join(roi_fitdir, 'roi_fits', 'roi%05d.png' % int(roi+1)))
                pl.close()
        
    return fitdf, fitparams, fitdata




def get_tuning_for_fov(animalid, session, fov, traceid='traces001', response_type='dff', 
                          n_bootstrap_iters=100, n_resamples=60, n_intervals_interp=3,
                          responsive_test='ROC', responsive_thr=0.05, 
                          make_plots=True, plot_metrics=True, return_iters=True,
                          create_new=False, n_processes=1, rootdir='/n/coxfs01/2p-data'):
    
    fitdf = None
    fitparams = None
    fitdata = None
    
    run_name = get_gratings_run(animalid, session, fov, traceid=traceid, rootdir=rootdir)    
    assert run_name is not None, "ERROR: [%s|%s|%s|%s] Unable to find gratings run..." % (animalid, session, fov, traceid)

    # Select only responsive cells:
    roi_list = util.get_responsive_cells(animalid, session, fov, run=run_name, traceid=traceid, 
                                    responsive_test=responsive_test, responsive_thr=responsive_thr,
                                    rootdir=rootdir)
    
    #% GET FITS:
    fitdf, fitparams, fitdata = get_tuning(animalid, session, fov, run_name, return_iters=return_iters,
                                           traceid=traceid, roi_list=roi_list, response_type=response_type,
                                           n_bootstrap_iters=n_bootstrap_iters, n_resamples=n_resamples, 
                                           n_intervals_interp=n_intervals_interp, make_plots=make_plots,
                                           responsive_test=responsive_test, responsive_thr=responsive_thr,
                                           create_new=create_new, rootdir=rootdir, n_processes=n_processes)

    print("... plotting comparison metrics ...")
    roi_fitdir = fitparams['directory']
    run_name = os.path.split(roi_fitdir.split('/traces')[0])[-1]
    data_identifier = '|'.join([animalid, session, fov, run_name, traceid])

    if plot_metrics:
        plot_selectivity_metrics(fitdf, fitparams, fit_thr=0.9, data_identifier=data_identifier)
        plot_top_asi_and_dsi(fitdf, fitparams, fit_thr=0.9, topn=10, data_identifier=data_identifier)
    
    print("*** done! ***")
    
    return fitdf, fitparams, fitdata


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
                          fig=None, ax=None, nr=1, nc=1, colspan=1, s_row=0, s_col=0, color='k'):
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
    ax.plot(thetas, radii, '-', color=color)
    ax.set_theta_zero_location("N")
    ax.set_yticks([curr_resps.min(), curr_resps.max()])
    ax.set_yticklabels(['', round(curr_resps.max(), 1)])

    
    return fig, ax



def plot_roi_tuning_raw_and_fit(roi, responses_df, curr_cfgs,
                                raw_traces, labels, sdf, fit_results,
                               trace_type='dff'):

    fig = pl.figure(figsize=(12,8))
    fig.patch.set_alpha(1)
    nr=2; nc=8;
    s_row=0
    fig, ax = plot_psth_roi(roi, raw_traces, labels, curr_cfgs, sdf, 
                            trace_type=trace_type,
                            fig=fig, nr=nr, nc=nc, s_row=0)
    ymin = np.min([0, ax.get_ylim()[0]])
    ax.set_ylim([ymin, ax.get_ylim()[1]])


    curr_oris = np.array([sdf['ori'][c] for c in curr_cfgs])  
    curr_resps = responses_df.mean()
    curr_sems = responses_df.sem()
    fig, ax1 = plot_tuning_curve_roi(curr_oris, curr_resps, curr_sems=curr_sems, 
                                     response_type=trace_type,
                                     fig=fig, nr=nr, nc=nc, s_row=1, colspan=5,
                                     marker='o', markersize=5, lw=0)


    fig, ax2 = plot_tuning_polar_roi(curr_oris, curr_resps, curr_sems=curr_sems, 
                                     response_type=trace_type,
                                     fig=fig, nr=nr, nc=nc, s_row=1, s_col=6, colspan=2)


    if fit_results is not None:
        oris_interp = np.array([rfit['x'] for rfit in fit_results]).mean(axis=0)
        resps_interp = np.array([rfit['y'] for rfit in fit_results]).mean(axis=0)
        resps_interp_sem = spstats.sem(np.array([rfit['y'] for rfit in fit_results]), axis=0)
        resps_fit = np.array([rfit['fit_y'] for rfit in fit_results]).mean(axis=0)
        n_intervals_interp = rfit['n_intervals_interp']

        fig, ax1 = plot_tuning_curve_roi(oris_interp[0:-n_intervals_interp], 
                                         resps_fit[0:-n_intervals_interp], 
                                         curr_sems=resps_interp_sem[0:-n_intervals_interp], 
                                         response_type=trace_type,color='cornflowerblue',
                                         markersize=0, lw=1, marker=None,
                                         fig=fig, ax=ax1, nr=nr, nc=nc, s_row=1, colspan=5)

        fig, ax2 = plot_tuning_polar_roi(oris_interp, 
                                         resps_fit, 
                                         curr_sems=resps_interp_sem, 
                                         response_type=trace_type, color='cornflowerblue',
                                         fig=fig, ax=ax2, nr=nr, nc=nc, s_row=1, s_col=6, colspan=2)
        
    ymin = np.min([0, ax1.get_ylim()[0]])
    ax1.set_ylim([ymin,  ax1.get_ylim()[1]])
    
    ax1.set_yticks([ymin, ax1.get_ylim()[1]])
    ax1.set_yticklabels([round(ymin, 2), round( ax1.get_ylim()[1], 2)])
    sns.despine(trim=True, offset=4, ax=ax1)

    
    if any([rfit['success'] for rfit in fit_results]):
        r2_avg = np.mean([rfit['r2'] for rfit in fit_results])
        ax1.text(0, ax1.get_ylim()[-1]*0.75, 'r2=%.2f' % r2_avg, fontsize=6)
    else:
        ax1.text(0, ax.get_ylim()[-1]*0.75, 'no fit', fontsize=6)
    
    return fig, ax, ax1, ax2



def plot_roi_tuning(roi, fitdata, sdf, df_traces, labels, trace_type='dff',
                    roi_fitdir='/tmp'):
    #print("... plotting")
    
    #% Set output dir for roi plots:
    roi_fitdir_figures = os.path.join(roi_fitdir, 'roi_fits')
    if not os.path.exists(roi_fitdir_figures):
        os.makedirs(roi_fitdir_figures)
    #print("... Saving ROI fit plots to:\n%s" % roi_fitdir_figures)

    responses_df = fitdata['original_data'][roi]['responses']
    curr_cfgs = fitdata['original_data'][roi]['stimulus_configs']
    best_cfg = best_cfg = fitdata['original_data'][roi]['responses'].mean().argmax()
    
    fig, ax, ax1, ax2 = plot_roi_tuning_raw_and_fit(roi, responses_df, curr_cfgs,
                                                    df_traces, labels, sdf, fitdata['results_by_iter'][roi], trace_type=trace_type)
    curr_oris = sorted(sdf['ori'].unique())
    ax1.set_xticks(curr_oris)
    ax1.set_xticklabels(curr_oris)
    ax1.set_title('(sz %i, sf %.2f)' % (sdf['size'][best_cfg], sdf['sf'][best_cfg]), fontsize=8)

    fig.suptitle('roi %i' % int(roi+1))
    
    return fig

# Summary plotting:

def compare_selectivity_all_fits(fitdf, fit_thr=0.9):
    
    strong_fits = [r for r, v in fitdf.groupby(['cell']) if v.mean()['r2'] >= fit_thr]
    print("%i out of %i cells with strong fits (%.2f)" % (len(strong_fits), len(fitdf['cell'].unique()), fit_thr))
    
    df = fitdf[fitdf['cell'].isin(strong_fits)]
    df['cell'] = df['cell'].astype(str)
    
    g = sns.PairGrid(df, hue='cell', vars=['ASI', 'DSI', 'r2'])
    g.fig.patch.set_alpha(1)
    
    
    g = g.map_offdiag(pl.scatter, marker='o',  alpha=0.5, s=1)
    
    
    g = g.map_diag(pl.hist, normed=True) #histtype="step",  
    
    g.set(ylim=(0, 1))
    g.set(xlim=(0, 1))
    #g.set(xticks=(0, 1))
    #g.set(yticks=(0, 1))
    g.set(aspect='equal')
    
    #sns.distplot, kde=False, hist=True, rug=True,\
                   #hist_kws={"histtype": "step", "linewidth": 2, "alpha": 1.0})
    
    if df.shape[0] < 10:
        g = g.add_legend(bbox_to_anchor=(1.01,.5))
    
    pl.subplots_adjust(left=0.1, right=0.85)

    #cleanup_axes(g.axes[:, 1:].flat, which_axis='y')
    #cleanup_axes( g.axes[:-1, :].flat, which_axis='x')
    
    
    return g.fig, strong_fits


def sort_by_selectivity(fitdf, fit_thr=0.9, topn=10):
    strong_fits = [r for r, v in fitdf.groupby(['cell']) if v.mean()['r2'] >= fit_thr]
    print("%i out of %i cells with strong fits (%.2f)" % (len(strong_fits), len(fitdf['cell'].unique()), fit_thr))
    
    df = fitdf[fitdf['cell'].isin(strong_fits)]
        
    #df.loc[:, 'cell'] = np.array([int(c) for c in df['cell'].values])
    
    top_asi = df.groupby(['cell']).mean().sort_values(['ASI'], ascending=False)
    top_dsi = df.groupby(['cell']).mean().sort_values(['DSI'], ascending=False)
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
    
    hue = 'top_asi' if color_by=='ASI' else 'top_dsi'
#    if color_by == 'ASI':
#        hue = 'top_asi'
#        palette = asi_colordict
#    elif color_by == 'DSI':
#        hue = 'top_dsi'
#        palette = dsi_colordict
#    

    g = sns.PairGrid(df, hue=hue, vars=['ASI', 'DSI'], palette=palette, size=5)#,
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


    
def plot_selectivity_metrics(fitdf, fitparams, fit_thr=0.9, data_identifier='METADATA'):
    
    roi_fitdir = fitparams['directory']
    
    #% PLOT -- plot ALL fits:
    if 'ASI_cv' in fitdf.columns.tolist():
        fitdf['ASI_cv'] = [1-f for f in fitdf['ASI_cv'].values] # Revert to make 1 = very tuned, 0 = not tuned
    if 'DSI_cv' in fitdf.columns.tolist():
        fitdf['DSI_cv'] = [1-f for f in fitdf['DSI_cv'].values] # Revert to make 1 = very tuned, 0 = not tuned
    
    fig, strong_fits = compare_selectivity_all_fits(fitdf, fit_thr=fit_thr)
    label_figure(fig, data_identifier)
    
    #nrois_fit = len(fitdf['cell'].unique())
    #nrois_thr = len(strong_fits)
    n_bootstrap_iters = fitparams['n_bootstrap_iters']
    n_intervals_interp = fitparams['n_intervals_interp']
    
    figname = 'compare-metrics_tuning-fit-thr%.2f_bootstrap-%iiters-interp%i' % (fit_thr, n_bootstrap_iters, n_intervals_interp)
    
    pl.savefig(os.path.join(roi_fitdir, '%s.png' % figname))
    pl.close()
    print("Saved:\n%s" % os.path.join(roi_fitdir, '%s.png' % figname))
    
    
def plot_top_asi_and_dsi(fitdf, fitparams, fit_thr=0.9, topn=10, data_identifier='METADATA'):
    #%
    # ##### Compare metrics
    
    # Sort cells by ASi and DSi    
    df, top_asi_cells, top_dsi_cells = sort_by_selectivity(fitdf, fit_thr=fit_thr, topn=topn)
    
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
          
    asi_colordict.update({ str(-10.0): (0.8, 0.8, 0.8, sub_alpha)})
    dsi_colordict.update({ str(-10.0): (0.8, 0.8, 0.8, sub_alpha)})
        
    
    #% PLOT by ASI:
    roi_fitdir = fitparams['directory']
    n_bootstrap_iters = fitparams['n_bootstrap_iters']
    nrois_fit = len(fitdf['cell'].unique())
    nrois_pass = len(df['cell'].unique())
    
    color_by = 'ASI'
    
    if color_by == 'ASI':
        palette = asi_colordict
    elif color_by == 'DSI':
        palette = dsi_colordict
            
    fig = compare_topn_selective(df, color_by=color_by, palette=palette)
    label_figure(fig, data_identifier)
    
    figname = 'sort-by-%s_top%i_tuning-fit-thr%.2f_bootstrap-%iiters_%iof%i' % (color_by, topn, fit_thr, n_bootstrap_iters, nrois_pass, nrois_fit)

    pl.savefig(os.path.join(roi_fitdir, '%s.png' % figname))
    pl.close()
    
    #% Color by DSI:
    color_by = 'DSI'
    if color_by == 'ASI':
        palette = asi_colordict
    elif color_by == 'DSI':
        palette = dsi_colordict
            
    fig = compare_topn_selective(df, color_by=color_by, palette=palette)
    label_figure(fig, data_identifier)
    figname = 'sort-by-%s_top%i_tuning-fit-thr%.2f_bootstrap-%iiters_%iof%i' % (color_by, topn, fit_thr, n_bootstrap_iters, nrois_pass, nrois_fit)

    pl.savefig(os.path.join(roi_fitdir, '%s.png' % figname))
    pl.close()
    
    



#%%

# #############################################################################
# Generic data loading:
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


def load_gratings_data(data_fpath):
    dset = np.load(data_fpath)
    
    #% Add baseline offset back into raw traces:
    F0 = np.nanmean(dset['corrected'][:] / dset['dff'][:] )
    print("offset: %.2f" % F0)
    raw_traces = pd.DataFrame(dset['corrected']) + F0
    
    labels = pd.DataFrame(data=dset['labels_data'], columns=dset['labels_columns'])
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
    

#%%


# #############################################################################
# EVALUATION:
# #############################################################################

def get_params_all_iters(fiters):
    fparams = pd.DataFrame({'r_pref':  [fiter['popt'][0] for fiter in fiters],
                        'r_null': [fiter['popt'][1] for fiter in fiters],
                        'theta_pref': [fiter['popt'][2] for fiter in fiters],
                        'sigma': [fiter['popt'][3] for fiter in fiters],
                        'r_offset': [fiter['popt'][4] for fiter in fiters]
                        })
    
    return fparams

def get_average_params_over_iters(fiters):
    '''
    Take all bootstrap iterations and get average value for each parameter.
    r_pref, r_null, theta_pref, sigma, r_offset = popt
    '''
    fparams = get_params_all_iters(fiters)
    
    r_pref = fparams.mean()['r_pref']
    r_null = fparams.mean()['r_null']

    theta_r = np.array([np.deg2rad(t) for t in fparams['theta_pref'].values])
    theta_r.min()
    theta_r.max()
    np.rad2deg(spstats.circmean(theta_r))
    theta_pref = np.rad2deg(spstats.circmean(theta_r)) #fparams.mean()['theta_pref']
    sigma = fparams.mean()['sigma']
    
    r_offset = fparams.mean()['r_offset']
    
    popt = (r_pref, r_null, theta_pref, sigma, r_offset)
    
    return popt


#%%  Look at residuals

def evaluate_fit_roi(roi, fitdf, fitparams, fitdata, response_type='dff'):
        
    n_intervals_interp = fitparams['n_intervals_interp']
    
    fig, axes = pl.subplots(2,3, figsize=(10,6))
    mean_residuals = []
    for fiter in fitdata['results_by_iter'][roi]:
        xs = fiter['x'][0:-1]
        fity = fiter['y'][0:-1]
        origy = fiter['fit_y'][0:-1]
        residuals = origy - fity
        
        axes[0,0].plot(xs, fity, lw=0.5)
        axes[0,1].scatter(origy, residuals, alpha=0.5)
        
        mean_residuals.append(np.mean(residuals))
    
    ax = axes[0,0]
    ax.set_xticks(xs[0::n_intervals_interp])
    ax.set_xticklabels([int(x) for x in xs[0::n_intervals_interp]])
    ax.set_xlabel('thetas')
    ax.set_ylabel('fit')
    ymin = np.min([0, ax.get_ylim()[0]])
    ax.set_ylim([ymin, ax.get_ylim()[1]])
    ax.tick_params(labelsize=8)
    ax.set_ylabel(response_type)
    sns.despine(ax=ax, trim=True, offset=2)
    
    
    ax = axes[0,1]
    ax.axhline(y=0, linestyle=':', color='k')
    ax.set_ylabel('residuals')
    ax.set_xlabel('fitted value')
    ax.tick_params(labelsize=8)
    
    ax = axes[0,2]
    ax.hist(mean_residuals, bins=20, color='k', alpha=0.5)
    ax.set_xlabel('mean residuals')
    ax.set_ylabel('counts of iters')
    ax.tick_params(labelsize=8)
    sns.despine(ax=ax, trim=True, offset=2)

    # Compare the original direction tuning curve with the fitted curve derived 
    # using the average of each fitting parameter (across 100 iterations)
    
    popt = get_average_params_over_iters(fitdata['results_by_iter'][roi])
    thetas = fitdata['results_by_iter'][roi][0]['x'][0::n_intervals_interp][0:-1]
    responses = fitdata['original_data'][roi]['responses']
    origr = responses.mean().values
    fitr = double_gaussian( thetas, *popt)
    
    # Get residual sum of squares and compare ORIG and AVG FIT:
    r2_comb, residuals = coeff_determination(origr, fitr)
        
    ax = axes[1,0]
    ax.plot(thetas, origr, 'k', label='orig')
    ax.plot(thetas, fitr, 'r:', label='avg-fit')
    ax.set_title('r2-comb: %.2f' % r2_comb)
    ax.legend()
    ax.set_xticks(xs[0::n_intervals_interp])
    ax.set_xticklabels([int(x) for x in xs[0::n_intervals_interp]], fontsize=8)
    ax.tick_params(labelsize=8)
    ax.set_xlabel('thetas')
    ax.set_ylabel(response_type)
    sns.despine(ax=ax, trim=True, offset=2)
    
    
    # Compare distN of preferred orientation across all iters
    fparams = get_params_all_iters(fitdata['results_by_iter'][roi])
    ax = axes[1,1]
    ax.hist(fparams['theta_pref'], alpha=0.5, bins=20, color='k', )
    #ax.set_xlim([fparams['theta_pref'].min(), fparams['theta_pref'].max()])
    ax.set_xticks([int(np.floor(fparams['theta_pref'].min())), int(np.ceil(fparams['theta_pref'].max()))])
    sns.despine(ax=ax, trim=True, offset=2)
    ax.set_xlabel('preferred theta')
    ax.set_ylabel('counts of iters')
    ax.tick_params(labelsize=8)
    
    # Look at calculated ASI/DSIs across iters:
    ax = axes[1,2]
    ax.scatter(fitdf[fitdf['cell']==roi]['ASI'], fitdf[fitdf['cell']==roi]['DSI'], c='k', marker='+', alpha=0.5)
    ax.set_xlim([0, 1]); ax.set_xticks([0, 0.5, 1]); ax.set_xlabel('ASI');
    ax.set_ylim([0, 1]); ax.set_yticks([0, 0.5, 1]); ax.set_ylabel('DSI');
    ax.set_aspect('equal')
    
    pl.subplots_adjust(hspace=.5, wspace=.5)
    
    fig.suptitle('roi %i' % int(roi+1))

    return fig



#%

def get_average_fit(roi, fitdata):
    
    n_intervals_interp = fitdata['results_by_iter'][roi][0]['n_intervals_interp']
    popt = get_average_params_over_iters(fitdata['results_by_iter'][roi])
    thetas = fitdata['results_by_iter'][roi][0]['x'][0::n_intervals_interp][0:-1]
    responses = fitdata['original_data'][roi]['responses']
    origr = responses.mean().values
    fitr = double_gaussian( thetas, *popt)
    
    return origr, fitr

def get_gfit(fitdf_roi):
    '''
    Use metric from Liang et al (2018). 
    Note: Will return NaN of fit is crappy
    
    IQR:  interquartile range (difference bw 75th and 25th percentile of r2 across iterations)
    r2_comb:  combined coeff of determination.
    
    '''
    r2_comb = fitdf_roi['r2comb'].unique()[0]
    iqr = spstats.iqr(fitdf_roi['r2'])
    gfit = np.mean(fitdf_roi['r2']) * (1-iqr) * np.sqrt(r2_comb)
    return gfit


def coeff_determination(origr, fitr):
    residuals = origr - fitr
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((origr - np.mean(origr))**2)
    r2 = 1 - (ss_res / ss_tot)
    
    return r2, residuals

def get_combined_r2(fitdata):
    '''
    Control step 2: combined coefficient of determination
    compare original direction tuning curve w/ fitted curve derived from 
    average of each fitting param (across all bootstrap iters).
    '''
    r2c = {}
    for roi in fitdata['results_by_iter'].keys():
        origr, fitr = get_average_fit(roi, fitdata)
        r2_comb, resid = coeff_determination(origr, fitr)
        r2c[roi] = r2_comb
    return r2c


def check_fit_quality(fitdf, fitdata):
    # Get residual sum of squares and compare ORIG and AVG FIT:
    r2_comb_values = get_combined_r2(fitdata)
    fitdf['r2comb'] = [r2_comb_values[r] for r in fitdf['cell'].values]
    
    # Calculate goodness of fit heuristic (quality + reliability of fit):
    gof = fitdf.groupby(['cell']).apply(get_gfit).dropna()
    
    if isinstance(fitdf['cell'].unique()[0], float):
        fitdf['cell'] = fitdf['cell'].astype(int)
        
    #passrois = threshold_fitness_quality(gof, goodness_thr=goodness_thr, plot_hist=False)
    #failrois = [r for r in fitdf['cell'].unique() if r not in passrois]
    #gof_dict = dict((roi, fitval) for roi, fitval in zip(passrois, gof.values))
    #gof_dict.update(dict((roi, None) for roi in failrois))    
    
    failrois = [r for r in fitdf['cell'].unique() if r not in gof.index.tolist()]
    gof_dict = dict((roi, fitval) for roi, fitval in zip(gof.index.tolist(), gof.values))
    gof_dict.update(dict((roi, None) for roi in failrois))
    
    fitdf['gof'] = [gof_dict[r] for r in fitdf['cell'].values]
    
    return fitdf

def get_reliable_fits(fitdf, goodness_thr=0.66):
    '''
    Drops cells with goodness_thr below threshold
    '''
    #x = fitdf.groupby(['cell']).apply(lambda x: any(np.isnan(x['gof'])))==False
    #goodfits = x[x.values==True].index.tolist()
    goodfits = fitdf[fitdf['gof'] > goodness_thr]['cell'].unique()
    
    return goodfits
    
def threshold_fitness_quality(gof, goodness_thr=0.66, plot_hist=True, ax=None):    
    if plot_hist:
        if ax is None:
            fig, ax = pl.subplots() #.figure()
        ax.hist(gof, alpha=0.5)
        ax.axvline(x=goodness_thr, linestyle=':', color='k')
        ax.set_xlim([0, 1])
        sns.despine(ax=ax, trim=True, offset=2)
        ax.set_xlabel('G-fit values')
        ax.set_ylabel('counts')

        
    good_fits = [int(r) for r in gof.index.tolist() if gof[r] > goodness_thr]
    
    if not plot_hist:
        return good_fits
    else:
        return good_fits, ax

#% Evaluation -- plotting

def compare_all_metrics_for_good_fits(fitdf, good_fits=[]):
    
    df = fitdf[fitdf['cell'].isin(good_fits)]
    df['cell'] = df['cell'].astype(str)
    
    g = sns.PairGrid(df, hue='cell', vars=[c for c in fitdf.columns.tolist() if c != 'cell'], palette='cubehelix')
    g.fig.patch.set_alpha(1)
    g = g.map_offdiag(pl.scatter, marker='o', s=5, alpha=0.7)
    g = g.map_diag(pl.hist, normed=True, alpha=0.5) #histtype="step",  
    
    #g.set(ylim=(0, 1))
    #g.set(xlim=(0, 1))
    #g.set(aspect='equal')
    
    return g.fig
    

def plot_tuning_for_good_fits(fitdata, fitparams, good_fits=[], plot_polar=True):
    n_intervals_interp = fitparams['n_intervals_interp']
    if plot_polar:
        thetas_interp = fitparams['interp_values']
        thetas = thetas_interp[0::n_intervals_interp]
    else:
        thetas_interp = fitparams['interp_values'][0:-1]
        thetas = thetas_interp[0::n_intervals_interp][0:-1]
        
    n_rois_pass = len(good_fits)
    nr = int(np.ceil(np.sqrt(n_rois_pass)))
    nc = int(np.ceil(float(n_rois_pass) / nr))
        
    fig, axes = pl.subplots(nr, nc, figsize=(nr*2,nc*2), subplot_kw=dict(polar=True))
    
    for ax, roi in zip(axes.flat, good_fits):
        
        responses = fitdata['original_data'][roi]['responses']
        origr = responses.mean().values
    
        popt = get_average_params_over_iters(fitdata['results_by_iter'][roi])
        
        fitr = double_gaussian(thetas_interp, *popt)
        
        if plot_polar:
            origr = np.append(origr, origr[0]) # wrap back around
            plot_tuning_polar_roi(thetas, origr, curr_sems=None, response_type='dff',
                                      fig=fig, ax=ax, color='k')
    
            plot_tuning_polar_roi(thetas_interp, fitr, curr_sems=None, response_type='dff',
                                      fig=fig, ax=ax, color='cornflowerblue')
            
        ax.set_title('%i' % int(roi), fontsize=6, y=1)
        
        ax.yaxis.grid(False)
        ax.yaxis.set_ticklabels([])
        
        ax.xaxis.grid(True)
        ax.xaxis.set_ticklabels([])
        #thetaticks = np.arange(0,360,45)
        #ax.tick_params(pad=0.2)
    
    for ax in axes.flat[len(good_fits):]:
        ax.axis('off')
        
    pl.subplots_adjust(hspace=0.3, wspace=0.3)
    
    return fig


#%
def evaluate_bootstrapped_tuning(fitdf, fitparams, fitdata, goodness_thr=0.66,
                                 response_type='dff', create_new=False,
                                 data_identifier='METADATA'):

    # Create output dir for evaluation
    roi_fitdir = fitparams['directory']
    evaldir = os.path.join(roi_fitdir, 'evaluation')
    if not os.path.exists(evaldir):
        os.makedirs(evaldir)
    
    roi_evaldir = os.path.join(evaldir, 'fit_rois')
    if not os.path.exists(roi_evaldir):
        os.makedirs(roi_evaldir)
    
        
    # Get residual sum of squares and compare ORIG and AVG FIT:
    do_evaluation = False
    evaluate_fpath = os.path.join(roi_fitdir, 'tuning_bootstrap_evaluation.pkl')    
    if os.path.exists(evaluate_fpath) and create_new is False:
        try:
            print("Loading existing evaluation results")
            with open(evaluate_fpath, 'rb') as f:
                evaluation_results = pkl.load(f)
            fitdf = evaluation_results['fits']
            assert 'gof' in fitdf.columns, "-- doing evaluation --"
        except Exception as e:
            traceback.print_exc()
            do_evaluation = True
    else:
        do_evaluation = True
        
    if do_evaluation:
        print("Evaluating bootstrap tuning results")
        #% # Plot visualizations for each cell's bootstrap iters for fit
        rois_fit = fitdf['cell'].unique()
        #n_rois_fit = len(rois_fit)
        for roi in rois_fit:
            fig = evaluate_fit_roi(roi, fitdf, fitparams, fitdata, response_type=response_type)
            label_figure(fig, data_identifier)
            pl.savefig(os.path.join(roi_evaldir, 'roi%05d.png' % int(roi+1)))
            pl.close()
        
        fitdf = check_fit_quality(fitdf, fitdata)
        evaluation_results = {'fits': fitdf}
        with open(evaluate_fpath, 'wb') as f:
            pkl.dump(evaluation_results, f, protocol=pkl.HIGHEST_PROTOCOL)
        
    goodfits = get_reliable_fits(fitdf, goodness_thr=goodness_thr)

    n_rois_pass = len(goodfits)
    n_rois_fit = len(fitdf['cell'].unique())
    print("%i out of %i fit cells pass goodness-thr %.2f" % (n_rois_pass, n_rois_fit, goodness_thr))

    if do_evaluation:
        print("plotting some metrics...")
        # Plot pairwise comparisons of metrics:
        fig = compare_all_metrics_for_good_fits(fitdf, good_fits=goodfits)
        label_figure(fig, data_identifier)
        pl.savefig(os.path.join(evaldir, 'pairplot-all-metrics_goodness-thr%.2f.png' % goodness_thr))
        pl.close()
        
        #% Plot tuning curves for all cells with good fits:
        fig = plot_tuning_for_good_fits(fitdata, fitparams, good_fits=goodfits, plot_polar=True)
        label_figure(fig, data_identifier)
        fig.suptitle("Cells with GoF > %.2f" % goodness_thr)
        
        figname = 'polar-tuning_goodness-of-fit%.2f_%irois' % (goodness_thr, len(goodfits))
        pl.savefig(os.path.join(evaldir, '%s.png' % figname))
        pl.close()
    
    return fitdf, goodfits



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
    parser.add_option('-R', '--responsive-test', action='store', dest='responsive_test', default='ROC', 
                      help="responsive test (default: ROC)")
    parser.add_option('-f', '--responsive-thr', action='store', dest='responsive_thr', default=0.05, 
                      help="responsive test threshold (default: p<0.05 for responsive_test=ROC)")
    
    # Tuning params:
    parser.add_option('-b', '--iter', action='store', dest='n_bootstrap_iters', default=100, 
                      help="N bootstrap iterations (default: 100)")
    parser.add_option('-s', '--samples', action='store', dest='n_resamples', default=60, 
                      help="N trials to sample w/ replacement (default: 60)")
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
                                         responsive_test='ROC', responsive_thr=0.05,
                                         n_bootstrap_iters=100, n_resamples=60,
                                         n_intervals_interp=3, goodness_thr = 0.66,
                                         n_processes=1, create_new=False, rootdir='/n/coxfs01/2p-data'):
    
    fitdf, fitparams, fitdata = get_tuning_for_fov(animalid, session, fov, 
                                             traceid=traceid, response_type=response_type, 
                                             n_bootstrap_iters=int(n_bootstrap_iters), 
                                             n_resamples = int(n_resamples),
                                             n_intervals_interp=int(n_intervals_interp),
                                             responsive_test=responsive_test, responsive_thr=responsive_thr,
                                             create_new=create_new, n_processes=n_processes, rootdir=rootdir)
    
    fit_desc = os.path.split(fitparams['directory'])[-1]
    run_name = os.path.split(fitparams['directory'].split('/traces')[0])[-1]
    data_identifier = '|'.join([animalid, session, fov, run_name, traceid, fit_desc])
    
    print("----- COMPLETED 1/2: bootstrap tuning! ------")

    fitdf, good_fits = evaluate_bootstrapped_tuning(fitdf, fitparams, fitdata, goodness_thr=goodness_thr,
                                             response_type=response_type, data_identifier=data_identifier)

    print("----- COMPLETED 2/2: evaluation (%i good cells)! -----" % len(good_fits))
    
    return fitdf, fitparams, good_fits

#%%

def main(options):
    opts = extract_options(options)
    
    fitdf, fitparams, good_fits = bootstrap_tuning_curves_and_evaluate(
                                             opts.animalid, opts.session, opts.fov, 
                                             traceid=opts.traceid, response_type=opts.response_type, 
                                             n_bootstrap_iters=int(opts.n_bootstrap_iters), 
                                             n_resamples=int(opts.n_resamples),
                                             n_intervals_interp=int(opts.n_intervals_interp),
                                             responsive_test=opts.responsive_test, responsive_thr=float(opts.responsive_thr),
                                             create_new=opts.create_new, n_processes=int(opts.n_processes), rootdir=opts.rootdir)
    print("***** DONE *****")
    
if __name__ == '__main__':
    main(sys.argv[1:])
    

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

