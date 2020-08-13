#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 11:10:13 2019

@author: julianarhee
"""
#%%
import matplotlib as mpl
mpl.use('agg')
import os
import glob
import json
#import h5py
import copy
#import cv2
#import imutils
import sys
import optparse
import shutil
import traceback
import time
import inspect

import numpy as np
import pandas as pd
import pylab as pl
import seaborn as sns
import cPickle as pkl
#import matplotlib.gridspec as gridspec

#from pipeline.python.classifications import experiment_classes as util
#from pipeline.python.classifications import test_responsivity as resp
#from pipeline.python.classifications import responsivity_stats as respstats
from pipeline.python.utils import label_figure, natural_keys, convert_range
from pipeline.python.retinotopy import convert_coords as cc
from pipeline.python.retinotopy import fit_2d_rfs as fitrf
#from matplotlib.patches import Ellipse, Rectangle

from shapely.geometry.point import Point
from shapely import affinity
import multiprocessing as mp

#%%

# ############################################
# Functions for processing visual field coverage
# ############################################

#def create_ellipse(center, lengths, angle=0):
#    """
#    create a shapely ellipse. adapted from
#    https://gis.stackexchange.com/a/243462
#    """
#    circ = Point(center).buffer(1)
#    ell = affinity.scale(circ, int(lengths[0]), int(lengths[1]))
#    ellr = affinity.rotate(ell, angle)
#    return ellr

#def intersection(lst1, lst2): 
#    return list(set(lst1) & set(lst2)) 

#%%

#row_vals = bootparams['row_vals']
#col_vals = bootparams['col_vals']


    # try:
    #     pool = mp.Pool(*args, **kwargs)
    #     yield pool
    #     pool.terminate()
    #     pool.join()
    # except KeyboardInterrupt:
    #     pool.terminate()
    #     sys.exit(1)
        
    
def group_configs(group, response_type):
    '''
    Takes each trial's reponse for specified config, and puts into dataframe
    '''
    config = group['config'].unique()[0]
    group.index = np.arange(0, group.shape[0])

    return pd.DataFrame(data={'%s' % config: group[response_type]})
 
def bootstrap_rf_params(rdf, response_type='dff',
                        row_vals=[], col_vals=[], sigma_scale=2.35,
                        n_resamples=10, n_bootstrap_iters=1000):
    #print(rdf.index.tolist()[0])
     
    paramsdf = None
    try:
        if not terminating.is_set():
            time.sleep(1)
            
        xres = np.unique(np.diff(row_vals))[0]
        yres = np.unique(np.diff(col_vals))[0]
        min_sigma=2.5; max_sigma=50;

        grouplist = [group_configs(group, response_type) for config, group in rdf.groupby(['config'])]
        responses_df = pd.concat(grouplist, axis=1) # indices = trial reps, columns = conditions

        # Get mean response across re-sampled trials for each condition (i.e., each position)
        # Do this n-bootstrap-iters times
        bootdf = pd.concat([responses_df.sample(n_resamples, replace=True).mean(axis=0) 
                                for ni in range(n_bootstrap_iters)], axis=1)
        
        # Fit receptive field for each set of bootstrapped samples 
        bparams = []; #x0=[]; y0=[];
        for ii in bootdf.columns:
            response_vector = bootdf[ii].values
            rfmap = fitrf.get_rf_map(response_vector, len(col_vals), len(row_vals))
            fitr, fit_y = fitrf.do_2d_fit(rfmap, nx=len(col_vals), ny=len(row_vals))
            if fitr['success']:
                amp_f, x0_f, y0_f, sigx_f, sigy_f, theta_f, offset_f = fitr['popt']
                if any(s < min_sigma for s in [abs(sigx_f)*xres*sigma_scale, abs(sigy_f)*yres*sigma_scale])\
                    or any(s > max_sigma for s in [abs(sigx_f)*xres*sigma_scale, abs(sigy_f)*yres*sigma_scale]):
                    fitr['success'] = False
                
            # if the fit for current bootstrap sample is good, add it to dataframe of bootstrapped rf params
            if fitr['success']:
                #amp_f, x0_f, y0_f, sigx_f, sigy_f, theta_f, offset_f = fitr['popt']
                curr_fit_results = list(fitr['popt'])
                curr_fit_results.append(fitr['r2'])
                bparams.append(tuple(curr_fit_results)) #(fitr['popt'])
        #%    
        if len(bparams)==0:
            return None

        bparams = np.array(bparams)   
        paramsdf = pd.DataFrame(data=bparams, columns=['amp', 'x0', 'y0', 'sigma_x', 'sigma_y', 'theta', 'offset', 'r2'])
        paramsdf['cell'] = [rdf.index[0] for _ in range(bparams.shape[0])]
   
    except KeyboardInterrupt:
        print("----exiting----")
        terminating.set()
        print("---set terminating---")

    return paramsdf

#%%
# --------------------------------------------------------
# Bootstrap (and corresponding pool/mp functions)
# --------------------------------------------------------

from functools import partial
from contextlib import contextmanager

@contextmanager
def poolcontext(*args, **kwargs):
    pool = mp.Pool(*args, **kwargs)
    yield pool
    pool.terminate()
    pool.join()
  
def initializer(terminating_):
    # This places terminating in the global namespace of the worker subprocesses.
    # This allows the worker function to access `terminating` even though it is
    # not passed as an argument to the function.
    global terminating
    terminating = terminating_

def pool_bootstrap(rdf_list, params, n_processes=1):   
    #try:
    results = []# None
    terminating = mp.Event()
    
#    with poolcontext(initializer=initializer, 
#                      initargs=(terminating, ),
#                      processes=n_processes) as pool:
#        
    pool = mp.Pool(initializer=initializer, initargs=(terminating, ), processes=n_processes)
    try:
        results = pool.map_async(partial(bootstrap_rf_params, 
                            response_type=params['response_type'],
                            row_vals=params['row_vals'], 
                            col_vals=params['col_vals'],
                            n_resamples=params['n_resamples'], 
                            n_bootstrap_iters=params['n_bootstrap_iters']), 
                        rdf_list).get(99999999)
        #pool.close()
    except KeyboardInterrupt:
        print("**interupt")
        pool.terminate()
        print("***Terminating!")
    finally:
        pool.close()
        pool.join()

    # except KeyboardInterrupt:
    #     pool.terminate()
    #     pool.join()
    #     sys.exit(1)
        
    return results
    
#def merge_names(a, b, c=1, d=2):
#    return '{} & {} - ({}, {})'.format(a, b, c, d)

#names = ['Brown', 'Wilson', 'Bartlett', 'Rivera', 'Molloy', 'Opie']
#with poolcontext(processes=2) as pool:
#    results = pool.map(partial(merge_names, b='Sons', c=100), names)
#print(results)
#

def run_bootstrap_evaluation(estats, fit_params, 
                            n_bootstrap_iters=1000, n_resamples=10,
                            ci=0.95, n_processes=1):
    eval_results = {}
    scale_sigma = fit_params['scale_sigma']
    sigma_scale = fit_params['sigma_scale'] if scale_sigma else 1.0
    response_type = fit_params['response_type']
             
    print("... doing bootstrap analysis for param fits.")
    roi_list = estats.rois  # Get list of all cells that pass fit-thr
    rdf_list = [estats.gdf.get_group(roi)[['config', 'trial', response_type]] for roi in roi_list]
    bootparams = copy.copy(fit_params)
    bootparams.update({'n_bootstrap_iters': n_bootstrap_iters, 
                       'n_resamples': n_resamples,
                       'ci': ci})   
    start_t = time.time()
    bootstrap_results = pool_bootstrap(rdf_list, bootparams, n_processes=n_processes)
    #except KeyboardInterrupt:
        #pool.terminate()
    end_t = time.time() - start_t
    print "Multiple processes: {0:.2f}sec".format(end_t)
    print "--- %i results" % len(bootstrap_results)

    if len(bootstrap_results)==0:
        return eval_results #None

    # Create dataframe of bootstrapped data
    bootdata = pd.concat(bootstrap_results)
    
    xx, yy, sigx, sigy = fitrf.convert_fit_to_coords(bootdata, 
                                                     fit_params['row_vals'], 
                                                     fit_params['col_vals'])
    bootdata['x0'] = xx
    bootdata['y0'] = yy
    bootdata['sigma_x'] = sigx * sigma_scale
    bootdata['sigma_y'] = sigy * sigma_scale
    
    # Calculate confidence intervals
    bootcis = get_cis_for_params(bootdata, ci=ci)

    # Plot bootstrapped distn of x0 and y0 parameters for each roi (w/ CIs)
    counts = bootdata.groupby(['cell']).count()['x0']
    unreliable = counts[counts < n_bootstrap_iters*0.5].index.tolist()
    print("%i cells seem to have <50%% iters with fits" % len(unreliable))
    
    eval_results = {'data': bootdata, 'params': bootparams, 'cis': bootcis, 'unreliable': unreliable}
    
    return eval_results

#%%
# ############################################################################
# Functions for receptive field fitting and evaluation
# ############################################################################
def get_empirical_ci(stat, ci=0.95):
    p = ((1.0-ci)/2.0) * 100
    lower = np.percentile(stat, p) #max(0.0, np.percentile(stat, p))
    p = (ci+((1.0-ci)/2.0)) * 100
    upper = np.percentile(stat, p) # min(1.0, np.percentile(x0, p))
    #print('%.1f confidence interval %.2f and %.2f' % (alpha*100, lower, upper))
    return lower, upper

def plot_bootstrapped_position_estimates(x0, y0, true_x, true_y, ci=0.95):
    lower_x0, upper_x0 = get_empirical_ci(x0, ci=ci)
    lower_y0, upper_y0 = get_empirical_ci(y0, ci=ci)

    fig, axes = pl.subplots(1, 2, figsize=(5,3))
    ax=axes[0]
    ax.hist(x0, color='k', alpha=0.5)
    ax.axvline(x=lower_x0, color='k', linestyle=':')
    ax.axvline(x=upper_x0, color='k', linestyle=':')
    ax.axvline(x=true_x, color='r', linestyle='-')
    ax.set_title('x0 (n=%i)' % len(x0))
    
    ax=axes[1]
    ax.hist(y0, color='k', alpha=0.5)
    ax.axvline(x=lower_y0, color='k', linestyle=':')
    ax.axvline(x=upper_y0, color='k', linestyle=':')
    ax.axvline(x=true_y, color='r', linestyle='-')
    ax.set_title('y0 (n=%i)' % len(y0))
    pl.subplots_adjust(wspace=0.5, top=0.8)
    
    return fig

def plot_bootstrapped_distribution(boot_values, true_x, ci=0.95, ax=None, param_name=''):
    lower_x0, upper_x0 = get_empirical_ci(boot_values, ci=ci)

    if ax is None:
        fig, ax = pl.subplots()
    ax.hist(boot_values, color='k', alpha=0.5)
    ax.axvline(x=lower_x0, color='k', linestyle=':')
    ax.axvline(x=upper_x0, color='k', linestyle=':')
    ax.axvline(x=true_x, color='r', linestyle='-')
    ax.set_title('%s (n=%i)' % (param_name, len(boot_values)))
   
    return ax

def plot_roi_evaluation(rid, meas_df, _fitr, _bootdata, ci=0.95, 
                             scale_sigma=True, sigma_scale=2.35):
    
    fig, axn = pl.subplots(2,3, figsize=(10,6))
    ax = axn.flat[0]
    ax = fitrf.plot_rf_map(_fitr['data'], cmap='inferno', ax=ax)
    ax = fitrf.plot_rf_ellipse(_fitr['fit_r'], ax=ax, scale_sigma=scale_sigma)
    params = ['sigma_x', 'sigma_y', 'theta', 'x0', 'y0']
    ai=0
    for param in params:
        ai += 1
        try:
            ax = axn.flat[ai]
            ax = plot_bootstrapped_distribution(_bootdata[param], meas_df[param][rid], 
                                                    ci=ci, ax=ax, param_name=param)
            pl.subplots_adjust(wspace=0.7, hspace=0.5, top=0.8)
            fig.suptitle('rid %i' % rid)
        except Exception as e:
            print("!! eval error (plot_boot_distn): rid %i, param %s" % (rid, param))
            #traceback.print_exc()
            
    return fig


def get_reliable_fits(pass_cis, pass_criterion='all'):
    if pass_criterion=='all':
        keep_rids = [i for i in pass_cis.index.tolist() if all(pass_cis.loc[i])]
        pass_df = pass_cis.loc[keep_rids]
    else:   
        keep_rids = [i for i in pass_cis.index.tolist() if any(pass_cis.loc[i])]
        pass_df = pass_cis.loc[keep_rids]
    
    reliable_rois = sorted(pass_df.index.tolist())

    return reliable_rois

 
def check_reliable_fits(meas_df, boot_cis): 
    # Test which params lie within 95% CI
    params = [p for p in meas_df.columns.tolist() if p!='r2']
    pass_cis = pd.concat([pd.DataFrame(
            [boot_cis['%s_lower' % p][ri] <= meas_df[p][ri] <= boot_cis['%s_upper' % p][ri] \
            for p in params], columns=[ri], index=params) \
                for ri in meas_df.index.tolist()], axis=1).T
       
    return pass_cis


def get_cis_for_params(bdata, ci=0.95):
    roi_list = [roi for roi, bdf in bdata.groupby(['cell'])]
    param_names = [p for p in bdata.columns if p != 'cell']
    CI = {}
    for p in param_names:
        CI[p] = dict((roi, get_empirical_ci(bdf[p].values, ci=ci)) for roi, bdf in bdata.groupby(['cell']))
    
    cis = {}
    for p in param_names:
        cvals = np.array([get_empirical_ci(bdf[p].values, ci=ci) for roi, bdf in bdata.groupby(['cell'])])
        cis['%s_lower' % p] = cvals[:, 0]
        cis['%s_upper' % p] = cvals[:, 1]
    cis = pd.DataFrame(cis, index=[roi_list])
    
    return cis
    
def visualize_bootstrapped_params(bdata, sorted_rois=[], sorted_values=[], nplot=20, rank_type='R2'):
    if sorted_rois is None:
        sorted_rois = bdata['cell'].unique()[0:nplot]
        rank_type = 'no rank'
    
    nplot = 20
    dflist = []
    for roi, d in bdata.groupby(['cell']): #.items():
        if roi not in sorted_rois[0:nplot]:
            continue
        tmpd = d.copy()
        tmpd['cell'] = [roi for _ in range(len(tmpd))]
        tmpd['rank'] = [sorted_values[roi] for _ in range(len(tmpd))]
        dflist.append(tmpd)
    df = pd.concat(dflist, axis=0)
    df['theta'] = [np.rad2deg(theta) % 360. for theta in df['theta'].values]
        
    fig, axes = pl.subplots(2,3, figsize=(15, 5))
    sns.boxplot(x='rank', y='amp', data=df, ax=axes[0,0])
    sns.boxplot(x='rank', y='x0', data=df, ax=axes[0,1])
    sns.boxplot(x='rank', y='y0', data=df, ax=axes[0,2])
    sns.boxplot(x='rank', y='theta', data=df, ax=axes[1,0])
    sns.boxplot(x='rank', y='sigma_x', data=df, ax=axes[1,1])
    sns.boxplot(x='rank', y='sigma_y', data=df, ax=axes[1,2])
    for ax in axes.flat:
        ax.set_xticks([]) 
        ax.set_xlabel('')
        sns.despine(ax=ax, trim=True, offset=2)
    fig.suptitle('bootstrapped param distNs (top 20 cells by %s)' % rank_type)
    
    return fig


       
#%%
from sklearn.linear_model import LinearRegression
import scipy.stats as spstats
import sklearn.metrics as skmetrics #import mean_squared_error

def regplot(x, y, data=None, x_estimator=None, x_bins=None, x_ci="ci",
            scatter=True, fit_reg=True, ci=95, n_boot=1000, units=None,
            order=1, logistic=False, lowess=False, robust=False,
            logx=False, x_partial=None, y_partial=None,
            truncate=False, dropna=True, x_jitter=None, y_jitter=None,
            label=None, color=None, marker="o",
            scatter_kws=None, line_kws=None, ax=None):
    '''
    Adjust regplot from Seaborn to return data (to access CIs)
    '''
    plotter = sns.regression._RegressionPlotter(x, y, data, x_estimator, x_bins, x_ci,
                                 scatter, fit_reg, ci, n_boot, units,
                                 order, logistic, lowess, robust, logx,
                                 x_partial, y_partial, truncate, dropna,
                                 x_jitter, y_jitter, color, label)
    if ax is None:
        ax = pl.gca()
        
    scatter_kws = {} if scatter_kws is None else copy.copy(scatter_kws)
    scatter_kws["marker"] = marker
    line_kws = {} if line_kws is None else copy.copy(line_kws)
    plotter.plot(ax, scatter_kws, line_kws)
    return ax, plotter

def do_regr_on_fov_cis(bootdata, bootcis, posdf, cond='azimuth',
                   roi_list=[], xaxis_lim=None, ci=.95, 
                   deviant_color='dodgerblue', marker='o', marker_size=20,
                   plot_boot_med=False, fill_marker=True):

    filter_weird=False; plot_all_cis=False;
   
    '''
    Plot "scatter":
    
    1. Mark all ROIs with fit (R2>0.5)
    2. Linear regression + CI (based off of seaborn's function)
    3. Mark cells with reliable fits (R2>0.5 + measured value w/in 95% CI)
    4. Mark reliable cells w/ CI outside of linear fit (1).
    
    ''' 
    axname = 'xpos' if cond=='azimuth' else 'ypos'
    parname = 'x0' if cond=='azimuth' else 'y0'


    fig, ax = pl.subplots(figsize=(10,8)); ax.set_title(cond);
    ax.set_ylabel('RF position (rel. deg.)')
    ax.set_xlabel('FOV position (um)')
    if xaxis_lim is not None:
        ax.set_xlim([0, xaxis_lim])
    else:
        ax.set_xlim([0, 1200])
       
    # 1. Identify which cells fail bootstrap fits - do not include in fit.
    fail_rois = [r for r in posdf.index.tolist() if r not in roi_list]
    fail_df = posdf.loc[fail_rois].copy()
    sns.regplot('%s_fov' % axname, '%s_rf' % axname, data=fail_df, ax=ax, 
                label='unreliable (R2>0.5)', color='gray', marker='x', fit_reg=False,
                scatter_kws=dict(s=marker_size, alpha=0.5))

    # 2a. Linear regression, include cells with reliable fits (R2 + 95% CI) 
    scatter_kws = dict(s=marker_size, alpha=1.0, facecolors='k')
    if not fill_marker:
        scatter_kws.update({'facecolors':'none', 'edgecolors':'k'})
    ax, plotter = regplot('%s_fov' % axname, '%s_rf' % axname,  ax=ax,
                          data=posdf.loc[roi_list], ci=ci*100, color='k', marker=marker, 
                          scatter_kws=scatter_kws, label='reliable (%i%% CI)' % int(ci*100)) #reliable') #measured' )

    # 2b. Get CIs from linear fit (fit w/ reliable rois only)
    grid, yhat, err_bands = plotter.fit_regression(grid=plotter.x)
    e1 = err_bands[0, :] # err_bands[0, np.argsort(xvals)] <- sort by xpos to plot
    e2 = err_bands[1, :] #err_bands[1, np.argsort(xvals)]
    regr_cis = np.array([(ex, ey) for ex, ey in zip(e1, e2)])

    
    # Get mean and upper/lower CI bounds of bootstrapped distn for each cell
    boot_rois = [k for k, g in bootdata.groupby(['cell'])] 
    roi_ixs = [boot_rois.index(ri) for ri in roi_list]

    #boot_meds = np.array([g[parname].mean() for k, g in bootdata.groupby(['cell'])])
    #x0_lower = bootcis['%s_lower' % parname][boot_rois] #[roi_list]
    #x0_upper = bootcis['%s_upper' % parname][boot_rois] #[roi_list]
  
    fov_pos = posdf['%s_fov' % axname][roi_list].values
    rf_pos = posdf['%s_rf' % axname][roi_list].values 
    fitv, regr = fit_linear_regr(fov_pos, rf_pos, return_regr=True)
    #ax.plot(fov_pos, fitv, 'r:')
    eq_str = 'y=%.2fx + %.2f' % (regr.coef_[0], regr.intercept_[0])
    ax.set_title(eq_str, loc='left', fontsize=12)
     
#%
    boot_meds = np.array([g[parname].mean() for k, g in bootdata[bootdata['cell'].isin(roi_list)].groupby(['cell'])])
    bootc = [(lo, up) for lo, up in zip(bootcis['%s_lower' % parname][roi_list].values, 
                                         bootcis['%s_upper' % parname][roi_list].values)]
   
    # Get YERR for plotting, (2, N), where 1st row=lower errors, 2nd row=upper errors
    boot_errs = np.array(zip(boot_meds - bootcis['%s_lower' % parname].loc[roi_list].values, 
                            bootcis['%s_upper' % parname].loc[roi_list].values - boot_meds)).T

    if plot_all_cis:
        # Plot bootstrap results for all RELIABLE cells 
        ax.scatter(fov_pos, boot_meds, c='k', marker='_', alpha=1.0, 
                   label='bootstrapped (%i%% CI)' % int(ci*100) )
        ax.errorbar(fov_pos, boot_meds, yerr=boot_errs, 
                    fmt='none', color='k', alpha=0.7, lw=1)
    sns.despine(offset=4, trim=True, ax=ax)

    # Check that values make sense and mark deviants
    vals = [(ri, roi, posdf['%s_fov' % axname][roi], posdf['%s_rf' % axname][roi]) \
            for ri, (roi, (bootL, bootU), (regL, regU), measured)
                in enumerate(zip(roi_list, bootc, regr_cis, rf_pos)) \
                if (bootL <= measured <= bootU) and ( (regL > bootU) or (regU < bootL) )]
     
    deviants = [v[1] for v in vals]
    xv = np.array([v[2] for v in vals])
    yv = np.array([v[3] for v in vals])
    dev_ixs = np.array([v[0] for v in vals])
    # Color/mark reliable fits that are also deviants
    if len(dev_ixs) > 0:
        yerrs = boot_errs[:, dev_ixs]
        ax.scatter(xv, yv, label='scattered', marker=marker,
                   s=marker_size, facecolors=deviant_color if fill_marker else 'none', 
                   edgecolors=deviant_color, alpha=1.0)
        if plot_boot_med:
            ax.scatter(xv, boot_meds[dev_ixs], c=deviant_color, marker='_', alpha=1.0) 
        ax.errorbar(xv, boot_meds[dev_ixs], yerr=yerrs, 
                        fmt='none', color=deviant_color, alpha=0.7, lw=1)
    ax.legend()

    bad_fits = [roi for rix, (roi, (lo, up), med) \
                in enumerate(zip(roi_list, bootc, rf_pos)) if not (lo <= med <= up) ]

    print("[%s] N deviants: %i (of %i reliable fits) | %i bad fits" % (cond, len(deviants), len(roi_list), len(bad_fits)))
 
    return fig, regr_cis, deviants, bad_fits

#%%


#%%
def plot_regr_and_cis(eval_results, posdf, cond='azimuth', ci=.95, xaxis_lim=1200, ax=None):
    
    bootdata = eval_results['data']
    bootcis = eval_results['cis']
    roi_list = [k for k, g in bootdata.groupby(['cell'])]    
    
    if ax is None:
        fig, ax = pl.subplots(figsize=(20,10))
    
    ax.set_title(cond)
    ax.set_ylabel('RF position (rel. deg.)')
    ax.set_xlabel('FOV position (um)')
    ax.set_xlim([0, xaxis_lim])
    
    axname = 'xpos' if cond=='azimuth' else 'ypos'
    parname = 'x0' if cond=='azimuth' else 'y0'
    
    g = sns.regplot('%s_fov' % axname, '%s_rf' % axname, data=posdf.loc[roi_list], ci=ci*100, color='k', marker='o',
                scatter_kws=dict(s=50, alpha=0.5), ax=ax, label='measured (regr: %i%% CI)' % int(ci*100) )


    xvals = posdf['%s_fov' % axname][roi_list].values
    yvals = posdf['%s_rf' % axname][roi_list].values
    
    # Get rois sorted by position:
    x0_meds = np.array([g[parname].mean() for k, g in bootdata.groupby(['cell'])])
    x0_lower = bootcis['%s_lower' % parname][roi_list]
    x0_upper = bootcis['%s_upper' % parname][roi_list]

    ci_intervals = bootcis['x0_upper'] - bootcis['x0_lower']
    weird = [i for i in ci_intervals.index.tolist() if ci_intervals[i] > 10]
    #weird = [i for ii, i in enumerate(bootcis.index.tolist()) if ((bootcis['%s_upper' % parname][i]) - (bootcis['%s_lower' % parname][i])) > 40]
    rlist = [i for i in roi_list if i not in weird]
    roi_ixs = np.array([roi_list.index(i) for i in rlist])
    roi_list = np.array([i for i in roi_list if i not in weird])
   
    if len(roi_ixs)==0:
        return ax

    # Plot bootstrap results
    xvals = posdf['%s_fov' % axname][roi_list].values
    yvals = posdf['%s_rf' % axname][roi_list].values
    
    ax.scatter(xvals, x0_meds[roi_ixs], c='k', marker='_', label='bootstrapped (%i%% CI)' % int(ci*100) )
    ax.errorbar(xvals, x0_meds[roi_ixs], yerr=np.array(zip(x0_meds[roi_ixs]-x0_lower.iloc[roi_ixs], x0_upper.iloc[roi_ixs]-x0_meds[roi_ixs])).T, 
            fmt='none', color='k', alpha=0.5)
    
#
#    ax.scatter(xvals, x0_meds, c='k', marker='_', label='bootstrapped (%i%% CI)' % int(ci*100) )
#    ax.errorbar(xvals, x0_meds, yerr=np.array(zip(x0_meds-x0_lower, x0_upper-x0_meds)).T, 
#            fmt='none', color='k', alpha=0.5)
    ax.set_xticks(np.arange(0, xaxis_lim, 100))
    #sns.despine(offset=1, trim=True, ax=ax)
    
    ax.legend()
            
    return ax

def fit_linear_regr(xvals, yvals, return_regr=False):
    regr = LinearRegression()
    if len(xvals.shape) == 1:
        xvals = np.array(xvals).reshape(-1, 1)
        yvals = np.array(yvals).reshape(-1, 1)
    else:
        xvals = np.array(xvals)
        yvals = np.array(yvals)
    if any(np.isnan(xvals)) or any(np.isnan(yvals)):
        print("NAN")
        #print(np.where(np.isnan(xvals)))
        #print(np.where(np.isnan(yvals)))
    regr.fit(xvals, yvals)
    fitv = regr.predict(xvals)
    if return_regr:
        return fitv.reshape(-1), regr
    else:
        return fitv.reshape(-1)


def plot_linear_regr(xv, yv, ax=None, 
                     marker='o', marker_size=30, alpha=1.0, marker_color='k',
                     linestyle='_', linecolor='r'):
    try:
        fitv, regr = fit_linear_regr(xv, yv, return_regr=True)
    except Exception as e:
        traceback.print_exc()
        print("... no lin fit")
        return None

    if ax is none:
        fig, ax = pl.subplots()
        
    rmse = np.sqrt(skmetrics.mean_squared_error(yv, fitv))
    r2 = float(skmetrics.r2_score(yv, fitv))
    print("[%s] Mean squared error: %.2f" % (cond, rmse))
    print('[%s] Variance score: %.2f' % (cond, r2))
    
    ax.scatter(xv, yv, c=marker_color, marker=marker, s=marker_size, alpha=alpha)
    ax.plot(xv, fitv, linestyle, color=linecolor)
    ax.set_xlim([0, 1200])
    #ax.set_ylim()    
    eq_str = 'y=%.2fx + %.2f' % (regr.coef_[0], regr.intercept_[0])
    ax.set_title(eq_str, loc='left', fontsize=12)
 
    r, p = spstats.pearsonr(xv, yv) #.abs())
    corr_str = 'pearson=%.2f (p=%.2f)' % (r, p)
    ax.plot(ax.get_xlim()[0], ax.get_ylim()[0], alpha=0, label=corr_str)
    ax.legend(loc='upper right', fontsize=8)

    return regr


def plot_linear_regr_by_condition(posdf, meas_df,):
    
    #posdf = posdf.loc[meas_df.index]
    fig, axes = pl.subplots(2, 3, figsize=(10, 6))
    for ri, cond in enumerate(['azimuth', 'elevation']):
        axname = 'xpos' if cond=='azimuth' else 'ypos'
            
        yv = posdf['%s_rf' % axname].values
        xv = posdf['%s_fov' % axname].values    
        try:
            fitv, regr = fit_linear_regr(xv, yv, return_regr=True)
        except Exception as e:
            traceback.print_exc()
            print("Error fitting cond %s" % cond)
            continue

        rmse = np.sqrt(skmetrics.mean_squared_error(yv, fitv))
        r2 = float(skmetrics.r2_score(yv, fitv))
        print("[%s] Mean squared error: %.2f | Variance score: %.2f" % (cond, rmse, r2))
        
        ax=axes[ri, 0]
        ax.set_title(cond, fontsize=12, loc='left')
        ax.scatter(xv, yv, c='k', alpha=0.5)
        ax.set_ylabel('RF position (rel. deg.)')
        ax.set_xlabel('FOV position (um)')
        #ax.set_xlim([0, ylim])
        #sns.despine(offset=1, trim=True, ax=ax)
        ax.plot(xv, fitv, 'r')
        ax.set_xlim([0, 1200])
        #ax.set_ylim()    
        r, p = spstats.pearsonr(posdf['%s_fov' % axname], posdf['%s_rf' % axname]) #.abs())
        corr_str = 'pearson=%.2f (p=%.2f)' % (r, p)
        ax.plot(ax.get_xlim()[0], ax.get_ylim()[0], alpha=0, label=corr_str)
        ax.legend(loc='upper right', fontsize=8)
    
        ax = axes[ri, 1]
        #ax.set_title(cond)
        #fig, ax = pl.subplots()
        residuals = fitv - yv
        ax.hist(residuals, histtype='step', color='k')
        #sns.despine(offset=1, trim=True, ax=ax)
        ax.set_xlabel('residuals')
        ax.set_ylabel('counts')
        maxval = max (abs(residuals))
        ax.set_xlim([-maxval, maxval])
         
        ax = axes[ri, 2]
        #fig, ax = pl.subplots()
        r2_vals = meas_df['r2']
        ax.scatter(r2_vals, abs(residuals), c='k', alpha=0.5)
        ax.set_xlabel('r2')
        ax.set_ylabel('abs(residuals)')
        
        testregr = LinearRegression()
        testregr.fit(r2_vals.reshape(-1, 1), residuals.reshape(-1, 1)) #, yv)
        r2_dist_corr = testregr.predict(r2_vals.reshape(-1, 1))
        ax.plot(r2_vals, r2_dist_corr, 'r')
        #sns.despine(offset=1, trim=True, ax=ax)
        r, p = spstats.pearsonr(r2_vals.values, np.abs(residuals))
        corr_str2 = 'pearson=%.2f (p=%.2f)' % (r, p)
        ax.plot(ax.get_xlim()[0], ax.get_xlim()[-1], alpha=0, label=corr_str2)
        ax.legend(loc='upper right', fontsize=8)
    
    pl.subplots_adjust(hspace=0.5, wspace=0.5)    
    return fig

#
#%%

def compare_regr_to_boot_params(eval_results, posdf, xlim=None, ylim=None, 
                                pass_criterion='all',
                                deviant_color='dodgerblue', marker='o',
                                marker_size=20, fill_marker=True,
                                outdir='/tmp', data_id='DATAID',
                                filter_weird=False, plot_all_cis=False):

    '''
    deviants:  cells w/ good RF fits (boostrapped, measured lies within some CI), but
               even CI lies outside of estimated regression CI
    bad_fits:  cells w/ measured RF locations that do not fall within the CI from bootstrapping
    
    To get all "pass" rois, include all returned ROIs with fits that are NOT in bad_fits.
    '''
    bootdata = eval_results['data']
    bootcis = eval_results['cis']
    fit_rois = [int(k) for k, g in bootdata.groupby(['cell'])]    
    pass_rois = eval_results['pass_cis'].index.tolist()
    reliable_rois = get_reliable_fits(eval_results['pass_cis'], pass_criterion=pass_criterion)
    #print('%i reliable of %i fit (thr>.5)' % (len(reliable_rois), len(pass_rois)))
    
    #% # Plot bootstrapped param CIs + regression CI
    xaxis_lim = max([xlim, ylim])
    reg_results = {}
    filter_str = '_filter-weird' if filter_weird else ''

    for cond in ['azimuth', 'elevation']:
        fig, regci, deviants, bad_fits = do_regr_on_fov_cis(bootdata, bootcis, posdf, cond=cond,
                                                        roi_list=reliable_rois,
                                                        deviant_color=deviant_color,
                                                        fill_marker=fill_marker,
                                                        marker=marker, marker_size=marker_size,
                                                        xaxis_lim=xlim) #xaxis_lim)
                                                        #plot_all_cis=plot_all_cis, filter_weird=filter_weird, )

        pass_rois = [i for i in fit_rois if i not in bad_fits]
        reg_results[cond] = {'cis': [tuple(ci) for ci in regci], 
                            'deviants': deviants, 
                            'bad_fits': bad_fits, 
                            'pass_rois': pass_rois}
 
        label_figure(fig, data_id)
        pl.savefig(os.path.join(outdir, 'VF2RF_regr_deviants_%s%s.svg' % (cond, filter_str)))
        pl.close()

    reg_results['reliable_rois'] = reliable_rois
    
    with open(os.path.join(outdir, 'regr_results_deviants_bycond.json'), 'w') as f:
        json.dump(reg_results, f, indent=4)    
    print("--- saved roi info after evaluation.")
  
    return reg_results

#%%
def identify_deviants(reg_results, eval_results, posdf, roi_list=[],
                      ci=0.95, offset_labels=False, rfdir='/tmp'):
    
    bootcis = eval_results['cis']

    # Which cells' CI contain the regression line, and which don't?
    print("Checking for deviants.")

    label_deviants = True
    deviants = {}
    conditions = ['azimuth', 'elevation'] #reg_results.keys() 
    
    # "bad_rois":cells whose mean value does not lie within the 95% CI for BOTH conditions
    bad_rois = np.intersect1d(reg_results['azimuth']['bad_fits'], 
                              reg_results['elevation']['bad_fits'])

    roi_list = reliable_rois
    if len(roi_list)==0:
        roi_list = bootcis.index.tolist()

    within = dict((k, []) for k in conditions)
    for cond in conditions:
        print("%s:  checking for deviants..." % cond)
        param = 'x0' if cond=='azimuth' else 'y0'
        axname = 'xpos' if cond=='azimuth' else 'ypos'
        
        regcis = reg_results[cond]['cis']
        paramcis = [(c1, c2) for c1, c2 in 
                        zip(bootcis['%s_lower' % param][roi_list], bootcis['%s_upper' % param][roi_list])]
     
        measured_vals = posdf['%s_rf' % axname].loc[roi_list].values
        in_ci=[]; nicht_gut=[];
        for roi, meas, (pci_lo, pci_up), (rci_lo, rci_up) in zip(roi_list, measured_vals, paramcis, regcis):
            if roi in bad_rois or not (pci_lo <= meas <= pci_up):
                nicht_gut.append(roi)
                continue
            if (rci_lo <= pci_lo <= rci_up) or (rci_lo <= pci_up <= rci_up):
                in_ci.append(roi)
            elif (pci_lo <= rci_lo <= pci_up) or (pci_lo <= rci_up <= pci_up):
                in_ci.append(roi)
        within[cond] = list(set(in_ci))
        print("... %i out of %i cells' bootstrapped param distNs lie within %i%% CI of regression fit" 
                % (len(within[cond]), len(roi_list), int(ci*100)))
        #trudeviants = [r for r in roi_list if r not in within and r not in reg_results[cond]['outliers']]
        trudeviants = [r for r in roi_list if r not in within[cond] 
                        and r not in reg_results[cond]['bad_fits']]
        print("... There are %i true deviants!" % len(trudeviants))
        print("same?", trudeviants==reg_results[cond]['deviants'])       
       
        fig, ax = pl.subplots(figsize=(20,10))
        ax = plot_regr_and_cis(eval_results, posdf, cond=cond, ax=ax)
        if len(trudeviants) > 0:
            deviant_fpos = posdf['%s_fov' % axname][trudeviants]
            deviant_rpos = posdf['%s_rf' % axname][trudeviants]
            ax.scatter(deviant_fpos, deviant_rpos, marker='*', c='dodgerblue', s=35, 
                       alpha=0.95, linewidth=2)

            avg_interval = np.diff(ax.get_yticks()).mean()
            if label_deviants:
                deviant_ixs = [roi_list.index(d) for d in trudeviants]
                for ix, roi in zip(deviant_ixs, trudeviants):
                    if offset_labels:
                        if deviant_rpos[roi]  > max(regcis[ix]):
                            ax.annotate(roi, (deviant_fpos[roi], deviant_rpos[roi]+avg_interval/2.), fontsize=6)
                        else:
                            ax.annotate(roi, (deviant_fpos[roi], deviant_rpos[roi]-avg_interval/2.), fontsize=6)        
                    else:
                        ax.annotate(roi, (deviant_fpos[roi], deviant_rpos[roi]), fontsize=10)
            #ax.set_ylim([-10, 40])
            #ax.set_xlim([0, 800])            
        sns.despine( trim=True, ax=ax)
        
        pl.savefig(os.path.join(rfdir, 'evaluation', 'regr-with-deviants_%s-test.svg' % cond))
        pl.close()

        deviants[cond] = trudeviants

    #%
    deviant_both = np.intersect1d(deviants['azimuth'], deviants['elevation'])
    #print deviant_both
    within_both = np.intersect1d(within['azimuth'], within['elevation'])
   
    deviants['bad'] = bad_rois
    deviants['pass'] = within_both
    deviants['deviant'] = deviant_both
 
    return deviants
   
    #%%
    

#%%
def evaluate_rfs(estats, fit_params, 
                 n_bootstrap_iters=1000, n_resamples=10,
                 ci=0.95, n_processes=1, 
                 create_new=False, rootdir='/n/coxfs01/2p-data'):
    '''
    Evaluate receptive field fits for cells with R2 > fit_thr.

    Returns:
        eval_results = {'data': bootdata, 'params': bootparams, 'cis': bootcis}
        
        bootdata : dataframe containing results of param fits for bootstrap iterations
        bootparams: params used to do bootstrapping
        cis: confidence intervals of fit params

        If no fits, returns {}
    '''
    eval_results = None
    eval_params = None
    # Create output dir for bootstrap results
    rfdir = fit_params['rfdir']
    evaldir = os.path.join(rfdir, 'evaluation')
    rf_eval_fpath = os.path.join(evaldir, 'evaluation_results.pkl')
           
    #if create_new:
    print("... do bootstrap analysis")
    eval_results = run_bootstrap_evaluation(estats, fit_params, 
                                         n_bootstrap_iters=n_bootstrap_iters, 
                                         n_resamples=n_resamples,
                                         ci=ci, n_processes=n_processes)
    
    # Update params if re-did evaluation
    eval_params_fpath = os.path.join(evaldir, 'evaluation_params.json')
    optsdict = locals()
    keys = [k for k in inspect.getargspec(evaluate_rfs).args if k not in ['estats', 'fit_params']]
    opts_update = dict((k, v) for k, v in optsdict.items() if k in keys)
    #if os.path.exists(eval_params_fpath):
    print(keys)
    try:
        eval_params = load_params(eval_params_fpath)
        for k, v in opts_update.items():
            eval_params.update({k: v})
    except Exception as e:
        traceback.print_exc()
        eval_params = opts_update.copy()
    save_params(eval_params_fpath, eval_params)
    print("... updated eval params")

    #%% Identify reliable fits 
    if eval_results is not None:
        meas_df = estats.fits.copy()
        pass_cis = check_reliable_fits(meas_df, eval_results['cis']) 
        eval_results.update({'pass_cis': pass_cis})
        
        # Save results
        with open(rf_eval_fpath, 'wb') as f:
            pkl.dump(eval_results, f, protocol=pkl.HIGHEST_PROTOCOL)

    return eval_results, eval_params
       
#%%
def identify_reliable_fits(eval_results, fit_results, fit_params, pass_criterion='all',
                           plot_boot_distns=True, plot_rois=[],
                           plot_format='svg', 
                           outdir='/tmp/roi_bootdistns', data_id='DATAID'):

    meas_df = fitrf.rfits_to_df(fit_results,
                            row_vals=fit_params['row_vals'], col_vals=fit_params['col_vals'],
                            scale_sigma=fit_params['scale_sigma'], sigma_scale=fit_params['sigma_scale'])
    meas_df = meas_df[meas_df['r2']>fit_params['fit_thr']]
 
    reliable_rois = get_reliable_fits(eval_results['pass_cis'], 
                                      pass_criterion=pass_criterion)
    if len(plot_rois)==0:
        plot_rois = reliable_rois
    
    pass_rois = meas_df.index.tolist()
    # Plot distribution of params w/ 95% CI
    if plot_boot_distns:
        print("... plotting boot distn") #.\n(to: %s" % outdir)
        for r in glob.glob(os.path.join(outdir, 'roi*')):
            os.remove(r)
        plot_eval_summary(meas_df, fit_results, eval_results, reliable_rois=pass_rois, #reliable_rois,
                          sigma_scale=fit_params['sigma_scale'], scale_sigma=fit_params['scale_sigma'],
                          outdir=outdir, plot_format=plot_format, 
                          data_id=data_id)
    print("%i out of %i cells w. R2>0.5 are reliable (95%% CI)" 
            % (len(reliable_rois), len(pass_rois)))
    return reliable_rois    


def plot_eval_summary(meas_df, fit_results, eval_results, reliable_rois=[],
                        sigma_scale=2.35, scale_sigma=True, 
                        outdir='/tmp/rf_fit_evaluation', plot_format='svg',
                        data_id='DATA ID'):
    '''
    For all fit ROIs, plot summary of results (fit + evaluation).
    Expect that meas_df has R2>fit_thr, since those are the ones that get bootstrap evaluation 
    '''
    bootdata = eval_results['data']
    roi_list = meas_df.index.tolist() #sorted(bootdata['cell'].unique())
    
    for ri, rid in enumerate(sorted(roi_list)):
        if ri % 20 == 0:
            print("... plotting eval summary (%i of %i)" % (int(ri+1), len(roi_list))) 
        _fitr = fit_results[rid]
        _bootdata = bootdata[bootdata['cell']==rid]
        fig = plot_roi_evaluation(rid, meas_df, _fitr, _bootdata, 
                                        scale_sigma=scale_sigma, sigma_scale=sigma_scale)
        if rid in reliable_rois:
            fig.suptitle('rid %i**' % rid)
        label_figure(fig, data_id)
        pl.savefig(os.path.join(outdir, 'roi%05d.%s' % (int(rid+1), plot_format)))
        pl.close()
    return


#%%
def load_params(params_fpath):
    with open(params_fpath, 'r') as f:
        options_dict = json.load(f)
    return options_dict

def save_params(params_fpath, opts):
    if isinstance(opts, dict):
        options_dict = opts.copy()
    else:
        options_dict = vars(opts)
    with open(params_fpath, 'w') as f:
        json.dump(options_dict, f, indent=4, sort_keys=True) 
    return
  
#%%
def load_eval_results(animalid, session, fov, experiment='rfs',
                        traceid='traces001', response_type='dff', 
                        fit_desc=None,
                        rootdir='/n/coxfs01/2p-data'):
    eval_results = None
    eval_params = None
                
    if fit_desc is None:
        fit_desc = 'fit-2dgaus_%s-no-cutoff' % response_type

    if 'combined' in experiment:
        rfname = experiment.split('_')[1]
    else:
        rfname = experiment
    try: 
        #print(rfname, glob.glob(os.path.join(rootdir, animalid, session, fov)))
        #for i in os.listdir(glob.glob(os.path.join(rootdir, animalid, session, fov))[0]):
        #    print("--", i)

        rfdir = glob.glob(os.path.join(rootdir, animalid, session, fov, '*%s_*' % rfname,
                        'traces', '%s*' % traceid, 'receptive_fields', 
                        '%s*' % fit_desc))[0]
        evaldir = os.path.join(rfdir, 'evaluation')
        assert os.path.exists(evaldir), "No evaluation exists. Aborting"
    except IndexError as e:
        traceback.print_exc()
    except AssertionError as e:
        traceback.print_exc()

    # Load results
    rf_eval_fpath = os.path.join(evaldir, 'evaluation_results.pkl')
    assert os.path.exists(rf_eval_fpath), "No evaluatoin file: %s" % rf_eval_fpath
    with open(rf_eval_fpath, 'rb') as f:
        eval_results = pkl.load(f)
   
    #  Load params 
    eval_params_fpath = os.path.join(evaldir, 'evaluation_params.json')
    with open(eval_params_fpath, 'r') as f:
        eval_params = json.load(f)
        
    return eval_results, eval_params

#%%
def load_matching_fit_results(animalid, session, fov, traceid='traces001',
                              experiment='rfs', response_type='dff',
                              nframes_post=0, 
                              sigma_scale=2.35, scale_sigma=True):
    fit_results=None
    fit_params=None
    try:
        fit_results, fit_params = fitrf.load_fit_results(animalid, session,
                                                            fov, traceid=traceid,
                                                            experiment=experiment,
                                                            response_type=response_type)
        assert fit_params['nframes_post_onset'] == nframes_post, \
            "Incorrect nframes_post (found %i, requested %i" % (fit_params['nframes_post_onset'], nframes_post)
        assert fit_params['response_type'] == response_type, \
            "Incorrect response type (found %i, requested %i" %(fit_params['repsonse_type'], response_type)
        if sigma_scale != fit_params['sigma_scale'] or scale_sigma != fit_params['scale_sigma']:
            print("... updating scale_sigma: %s" % str(fit_params['sigma_scale']))
            scale_sigma=fit_params['scale_sigma']               
            print("... updating sigma_scale to %.2f (from %.2f)" % (fit_params['sigma_scale'], sigma_scale))
            sigma_scale=fit_params['sigma_scale']
            do_fits=True
    except Exception as e:
        print(e)
        traceback.print_exc()
        print("[err]: unable to load fit results, re-fitting.")
    
    return fit_results, fit_params

#%%
def do_rf_fits_and_evaluation(animalid, session, fov, rfname=None, traceid='traces001', 
                              response_type='dff', n_processes=1,
                              fit_thr=0.5, n_resamples=10, n_bootstrap_iters=1000, 
                              post_stimulus_sec=0., sigma_scale=2.35, scale_sigma=True,
                              ci=0.95, pass_criterion='all',
                              plot_boot_distns=True, plot_pretty_rfs=True,  
                              deviant_color='dodgerblue', filter_weird=False, plot_all_cis=False,
                              do_fits=False, do_evaluation=False, reload_data=False,
                              create_stats=False,
                              rootdir='/n/coxfs01/2p-data', opts=None):

    from pipeline.python.classifications import experiment_classes as util
#    rfname= 'rfs' 
#    do_fits =False
#    do_evaluation = True
#    reload_data=False
#   
    #%% Get session info 
    S = util.Session(animalid, session, fov)
    experiment_list = S.get_experiment_list(traceid=traceid)
    assert rfname in experiment_list, "[%s] NO receptive field experiments found! (%s)" % (str(rfname), str(experiment_list))
    #assert 'rfs' in experiment_list or 'rfs10' in experiment_list, "NO receptive field experiments found!"
    #if rfname is None or 'rfs' not in rfname:
    #    rfname = 'rfs10' if 'rfs10' in experiment_list else 'rfs'      
   
    # Create experiment object 
    exp = util.ReceptiveFields(rfname, animalid, session, fov, 
                               traceid=traceid, trace_type='corrected')
    # Check if should do fitting 
    nframes_post = int(round(post_stimulus_sec*44.65))
    if not do_fits:
        try:
            fit_results, fit_params = load_matching_fit_results(animalid, session, fov,
                                                experiment=rfname, traceid=traceid, 
                                                response_type=response_type, nframes_post=nframes_post,
                                                sigma_scale=sigma_scale, scale_sigma=scale_sigma)
            assert fit_results is not None
            
        except Exception as e:
            print(e)
            traceback.print_exc()
            print("[err]: unable to load fit results, re-fitting.")
            do_fits = True
           
    #%% Get RF fit stat
    estats = exp.get_stats(response_type=response_type, fit_thr=fit_thr, 
                            scale_sigma=scale_sigma, sigma_scale=sigma_scale,
                            nframes_post=nframes_post,
                            do_fits=do_fits, plot_pretty_rfs=plot_pretty_rfs,
                            return_all_rois=False,
                            reload_data=reload_data,
                            create_new=any([create_new, do_fits, do_evaluation])) #create_stats)
    assert estats is not None, "Failed to get exp.get_stats(). ABORTING."
 
    rfdir = estats.fitinfo['rfdir']
    fit_desc = estats.fitinfo['fit_desc'] 
    fit_params = estats.fitinfo
    data_id = '|'.join([exp.animalid, exp.session, exp.fov, \
                            exp.traceid, exp.rois, exp.trace_type, fit_desc])

    if 'rfs10' in exp.name:
        #print("Found: %i" % estats.fitinfo['column_spacing'])
        assert fit_params['column_spacing']==10, "WRONG SPACING (%s), is %.2f." % (exp.name, fit_params['column_spacing'])
        
    # Set directories
    evaldir = os.path.join(estats.fitinfo['rfdir'], 'evaluation')
    roidir = os.path.join(evaldir, 'rois_bootstrap-%i-iters_%i-resample' % (n_bootstrap_iters, n_resamples))
    if not os.path.exists(roidir):
        os.makedirs(roidir) 
    if os.path.exists(os.path.join(evaldir, 'rois')):
        shutil.rmtree(os.path.join(evaldir, 'rois'))

    #%% Do bootstrap analysis    
    print("-evaluating (%s)-" % str(do_evaluation))
    if create_new is False: #nd create_new is False:
        try:
            print("... loading eval results")
            eval_results, eval_params = load_eval_results(animalid, session, fov, 
                                                         experiment=rfname,
                                                         fit_desc=os.path.split(rfdir)[-1],
                                                         response_type=response_type) 
            print("N eval:", len(eval_results['pass_cis'].index.tolist()))
            assert 'data' in eval_results.keys(), "... old datafile, redoing boot analysis"
            assert 'pass_cis' in eval_results.keys(), "... no criteria passed, redoing"
        except Exception as e:
            traceback.print_exc()
            do_evaluation=True
 
    if do_evaluation: 
        print("... doing rf evaluation")
        eval_results, eval_params = evaluate_rfs(estats, fit_params, 
                                    n_bootstrap_iters=n_bootstrap_iters, 
                                    n_resamples=n_resamples,
                                    ci=ci, n_processes=n_processes, 
                                    create_new=do_evaluation)

    # Update params
    if len(eval_results.keys())==0:# is None: # or 'data' not in eval_results:
        return {} #None

    ##------------------------------------------------
    #%% Identify reliable fits
    #if not do_fits: # need to load fit results
    fit_results, fit_params = exp.get_rf_fits(response_type=response_type, fit_thr=fit_thr, 
                                                    make_pretty_plots=False, reload_data=False,create_new=False)

    reliable_rois = identify_reliable_fits(eval_results, fit_results, fit_params,
                                           pass_criterion=pass_criterion, 
                                           plot_boot_distns=plot_boot_distns, #do_evaluation,
                                           plot_format='svg', outdir=roidir, data_id=data_id)
    
    eval_results.update({'reliable_rois': reliable_rois})

    fovcoords = exp.get_roi_coordinates()
    marker_size=30; fill_marker=True; marker='o';
    reg_results, posdf = regr_rf_fov(fovcoords, fit_results, fit_params, eval_results, data_id=data_id,
                                     pass_criterion=pass_criterion, marker=marker, marker_size=marker_size, 
                                     fill_marker=fill_marker, deviant_color=deviant_color)
    
    return eval_results, eval_params


def regr_rf_fov(fovcoords, fit_results, fit_params, eval_results, 
                pass_criterion='all', data_id='ID', 
                deviant_color='magenta', marker='o', 
                marker_size=20, fill_marker=True):
    reliable_rois = get_reliable_fits(eval_results['pass_cis'], pass_criterion=pass_criterion)

    #%% Get measured fits
    meas_df = fitrf.rfits_to_df(fit_results,
                            row_vals=fit_params['row_vals'], col_vals=fit_params['col_vals'],
                            scale_sigma=fit_params['scale_sigma'], sigma_scale=fit_params['sigma_scale'])
    meas_df = meas_df[meas_df['r2']>fit_params['fit_thr']]
 
    #%% Fit linear regression for brain coords vs VF coords 
    posdf = pd.concat([meas_df[['x0', 'y0']].copy(), 
                       fovcoords['roi_positions'].copy()], axis=1) 
    posdf = posdf.rename(columns={'x0': 'xpos_rf', 'y0': 'ypos_rf',
                          'ml_pos': 'xpos_fov', 'ap_pos': 'ypos_fov'})

    evaldir = os.path.join(fit_params['rfdir'], 'evaluation')
    fig = plot_linear_regr_by_condition( posdf.loc[reliable_rois], meas_df.loc[reliable_rois])
    pl.subplots_adjust(top=0.9, bottom=0.1, hspace=0.5)
    label_figure(fig, data_id)
    pl.savefig(os.path.join(evaldir, 'RFpos_v_FOVpos_split_axes_reliable_cells_.svg'))   
    pl.close()
   
    #%% Compare regression fit to bootstrapped params 
    reg_results = compare_regr_to_boot_params(eval_results, posdf, 
                                            outdir=evaldir, data_id=data_id, 
                                            pass_criterion=pass_criterion,
                                            deviant_color=deviant_color, marker=marker,
                                            marker_size=marker_size, fill_marker=fill_marker)
                                            #filter_weird=filter_weird, plot_all_cis=plot_all_cis 
    
    #%% Identify "deviants" based on spatial coordinates
#    deviants = identify_deviants(reg_results, eval_results, estats.fovinfo['positions'],
#                                 offset_labels=False, 
#                                 ci=ci, rfdir=rfdir)    
##    with open(os.path.join(rfdir, 'evaluation', 'deviants_bothconds.json'), 'w') as f:
#        json.dump(deviants, f, indent=4)
    print('%i reliable of %i fit (thr>.5)' % (len(reg_results['reliable_rois']), len(meas_df)))

    return reg_results, posdf #deviants


 
   
#%%
def extract_options(options):
    
    parser = optparse.OptionParser()

    parser.add_option('-D', '--root', action='store', dest='rootdir', default='/n/coxfs01/2p-data',\
                      help='data root dir (root project dir containing all animalids) [default: /n/coxfs01/2pdata]')
    parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')

    # Set specific session/run for current animal:
    parser.add_option('-S', '--session', action='store', dest='session', default='FOV1_zoom2p0x', \
                      help='session dir (format: YYYMMDD_ANIMALID')
    parser.add_option('-A', '--fov', action='store', dest='fov', default='FOV1_zoom2p0x', \
                      help="acquisition folder (ex: 'FOV2_zoom3x') [default: FOV1]")
    parser.add_option('-R', '--rfname', action='store', dest='rfname', default=None, \
                      help="name of rfs to process (default uses rfs10, if exists, else rfs)")
    parser.add_option('--fit', action='store_true', dest='do_fits', default=False, \
                      help="flag to do RF fitting anew")
    parser.add_option('--eval', action='store_true', dest='do_evaluation', default=False, \
                      help="flag to do RF evaluation anew")
    parser.add_option('--load', action='store_true', dest='reload_data', default=False, \
                      help="flag to reload data arrays and save (e.g., dff.pkl)")



    parser.add_option('-t', '--traceid', action='store', dest='traceid', default='traces001', \
                      help="name of traces ID [default: traces001]")
    parser.add_option('-d', '--data-type', action='store', dest='trace_type', default='corrected', \
                      help="Trace type to use for analysis [default: corrected]")

    parser.add_option('-M', '--resp', action='store', dest='response_type', default='dff', \
                      help="Response metric to use for creating RF maps (default: dff)")    
    parser.add_option('-f', '--fit-thr', action='store', dest='fit_thr', default=0.5, \
                      help="Threshold for RF fits (default: 0.5)")

    parser.add_option('-b', '--n-boot', action='store', dest='n_bootstrap_iters', default=1000, \
                      help="N bootstrap iterations for evaluating RF param fits (default: 1000)")
    parser.add_option('-s', '--n-resamples', action='store', dest='n_resamples', default=10, \
                      help="N trials to sample with replacement (default: 10)")
    parser.add_option('-n', '--n-processes', action='store', dest='n_processes', default=1, \
                      help="N processes (default: 1)")
    
    parser.add_option('-C', '--ci', action='store', dest='ci', default=0.95, \
                      help="CI percentile(default: 0.95)")

    parser.add_option('--no-boot-plot', action='store_false', dest='plot_boot_distns', default=True, \
                      help="flag to not plot bootstrapped distNs of x0, y0 for each roi")

#    parser.add_option('--pixels', action='store_false', dest='transform_fov', default=True, \
#                      help="flag to not convert fov space into microns (keep as pixels)")
#
    parser.add_option('--remove-weird', action='store_true', dest='filter_weird', default=False, \
                      help="flag to remove really funky fits")
    parser.add_option('--all-cis', action='store_true', dest='plot_all_cis', default=False, \
                      help="flag to plot CIs for all cells (not just deviants)")
    parser.add_option('-c', '--color', action='store', dest='deviant_color', default='dodgerblue', \
            help="color to plot deviants to stand out (default: dodgerblue)")

    parser.add_option('--pass', action='store', dest='pass_criterion', default='all', \
                      help="criterion for ROI passing fit(default: 'all' - all params pass 95% CI)")
 

    parser.add_option('--sigma', action='store', dest='sigma_scale', default=2.35, \
                      help="sigma scale factor for FWHM (default: 2.35)")
    parser.add_option('--no-scale', action='store_false', dest='scale_sigma', default=True, \
                      help="set to scale sigma to be true sigma, rather than FWHM")
    parser.add_option('-p', '--post', action='store', dest='post_stimulus_sec', default=0.0, 
                      help="N sec to include in stimulus-response calculation for maps (default:0.0)")

    parser.add_option('--test', action='store_true', dest='test_run', default=False, 
                      help="Flag to just wait 2 sec, for test")


    (options, args) = parser.parse_args(options)

    return options

#%%
rootdir = '/n/coxfs01/2p-data'
animalid = 'JC084' #JC076'
session = '20190522' #'20190501'
fov = 'FOV1_zoom2p0x'
create_new = False

# Data
traceid = 'traces001'
trace_type = 'corrected'
response_type = 'dff'
fit_thr = 0.5
#transform_fov = True

# Bootstrap params
n_bootstrap_iters=1000
n_resamples = 10
plot_boot_distns = True
ci = 0.95
n_processes=1   
sigma_scale = 2.35
scale_sigma = True
post_stimulus_sec=0.5

do_fits=False
do_evaluation=False
reload_data=False

options = ['-i', animalid, '-S', session, '-A', fov, '-t', traceid,
           '-R', 'rfs', '-M', response_type, '-p', 0.5 ]
#%%

def main(options):
    opts = extract_options(options)
    animalid = opts.animalid
    session = opts.session
    fov = opts.fov
    traceid = opts.traceid
    response_type = opts.response_type
    rootdir = opts.rootdir

    do_fits = opts.do_fits
    do_evaluation = opts.do_evaluation
    reload_data = opts.reload_data
   
    n_resamples = opts.n_resamples
    n_bootstrap_iters = opts.n_bootstrap_iters
    n_processes = int(opts.n_processes)
    pass_criterion = opts.pass_criterion
    
    ci = opts.ci
    plot_boot_distns = opts.plot_boot_distns

    rfname = opts.rfname
    filter_weird = opts.filter_weird
    plot_all_cis = opts.plot_all_cis
    deviant_color = opts.deviant_color

    sigma_scale = float(opts.sigma_scale)
    scale_sigma = opts.scale_sigma
    post_stimulus_sec = float(opts.post_stimulus_sec)
    fit_thr = float(opts.fit_thr)
   
    print("STATS?", any([do_fits, do_evaluation, reload_data]))
    if opts.test_run:
        print(">>> testing <<<")
        assert opts.test_run is False, "FAKE ERROR, test."

    else: 
        eval_results, eval_params = do_rf_fits_and_evaluation(animalid, session, fov, rfname=rfname,
                              traceid=traceid, response_type=response_type, fit_thr=fit_thr,
                              n_bootstrap_iters=n_bootstrap_iters, n_resamples=n_resamples, ci=ci,
                              #transform_fov=transform_fov, 
                              plot_boot_distns=plot_boot_distns, 
                              post_stimulus_sec=post_stimulus_sec, 
                              n_processes=n_processes, filter_weird=filter_weird, plot_all_cis=plot_all_cis,
                              deviant_color=deviant_color, 
                              scale_sigma=scale_sigma, sigma_scale=sigma_scale,
                              pass_criterion=pass_criterion,
                              do_fits=do_fits, do_evaluation=do_evaluation, reload_data=reload_data,
                              create_stats=any([do_fits, do_evaluation, reload_data]),
                              rootdir=rootdir, opts=opts)
        
    print("***DONE!***")

if __name__ == '__main__':
    from pipeline.python.classifications import experiment_classes as util

    main(sys.argv[1:])
           
    #%%
    
    #options = ['-i', 'JC084', '-S', '20190525', '-A', 'FOV1_zoom2p0x', '-R', 'rfs']
    


# %%
