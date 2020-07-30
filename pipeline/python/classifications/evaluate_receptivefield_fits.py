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

import numpy as np
import pandas as pd
import pylab as pl
import seaborn as sns
import cPickle as pkl
#import matplotlib.gridspec as gridspec

#from pipeline.python.classifications import experiment_classes as util
#from pipeline.python.classifications import test_responsivity as resp
from pipeline.python.classifications import responsivity_stats as respstats
from pipeline.python.utils import label_figure, natural_keys, convert_range
from pipeline.python.retinotopy import convert_coords as coor
from pipeline.python.retinotopy import fit_2d_rfs as fitrf
#from matplotlib.patches import Ellipse, Rectangle

from shapely.geometry.point import Point
from shapely import affinity
import multiprocessing as mp

#%%

# ############################################
# Functions for processing visual field coverage
# ############################################

def create_ellipse(center, lengths, angle=0):
    """
    create a shapely ellipse. adapted from
    https://gis.stackexchange.com/a/243462
    """
    circ = Point(center).buffer(1)
    ell = affinity.scale(circ, int(lengths[0]), int(lengths[1]))
    ellr = affinity.rotate(ell, angle)
    return ellr

def intersection(lst1, lst2): 
    return list(set(lst1) & set(lst2)) 

#%%
def get_session_object(animalid, session, fov, traceid='traces001', trace_type='corrected',
                       create_new=True, rootdir='/n/coxfs01/2p-data'):
        
    from pipeline.python.classifications import experiment_classes as util

    # # Creat session object
    summarydir = os.path.join(rootdir, animalid, session, fov, 'summaries')
    session_outfile = os.path.join(summarydir, 'sessiondata.pkl')
    if os.path.exists(session_outfile) and create_new is False:
        print("... loading session object")
        with open(session_outfile, 'rb') as f:
            S = pkl.load(f)
    else:
        print("... creating new session object")
        S = util.Session(animalid, session, fov, rootdir=rootdir)
    
        # Save session data object
        if not os.path.exists(summarydir):
            os.makedirs(summarydir)
            
        with open(session_outfile, 'wb') as f:
            pkl.dump(S, f, protocol=pkl.HIGHEST_PROTOCOL)
            print("... new session object to: %s" % session_outfile)
    print("... got session object w/ experiments:", S.experiments)

    try:
        print("Found %i experiments in current session:" % len(S.experiment_list), S.experiment_list)
        assert 'rfs' in S.experiment_list or 'rfs10' in S.experiment_list, "ERROR:  No receptive field mapping found for current dataset: [%s|%s|%s]" % (S.animalid, S.session, S.fov)
    except Exception as e:
        print e
        return None

    return S


   


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

    paramsdf = None
    try:
        if not terminating.is_set():
            time.sleep(1)
            
        xres = np.unique(np.diff(row_vals))[0]
        yres = np.unique(np.diff(col_vals))[0]
        min_sigma=5; max_sigma=50;

        grouplist = [group_configs(group, response_type) for config, group in rdf.groupby(['config'])]
        responses_df = pd.concat(grouplist, axis=1) # indices = trial reps, columns = conditions

        # Get mean response across re-sampled trials for each condition (i.e., each position)
        # Do this n-bootstrap-iters times
        bootdf = pd.concat([responses_df.sample(n_resamples, replace=True).mean(axis=0) for ni in range(n_bootstrap_iters)], axis=1)
        
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

from functools import partial
from contextlib import contextmanager

@contextmanager
def poolcontext(*args, **kwargs):
    pool = mp.Pool(*args, **kwargs)
    yield pool
    pool.terminate()

  
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
    
    # with poolcontext(initializer=initializer, 
    #                  initargs=(terminating, ),
    #                  processes=n_processes) as pool:
        
    pool = mp.Pool(initializer=initializer, initargs=(terminating, ), processes=n_processes)
    try:
        results = pool.map(partial(bootstrap_rf_params, 
                            response_type=params['response_type'],
                            row_vals=params['row_vals'], 
                            col_vals=params['col_vals'],
                            n_resamples=params['n_resamples'], 
                            n_bootstrap_iters=params['n_bootstrap_iters']), 
                        rdf_list)
        pool.close()
    except KeyboardInterrupt:
        print("**interupt")
        pool.terminate()
        print("***Termiating!")
    finally:
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

def bootstrap_param_fits(estats, response_type='dff',
                            n_bootstrap_iters=1000, n_resamples=10,
                            ci=0.95, n_processes=1, 
                            scale_sigma=True, sigma_scale=2.35):
    bootresults = {}
    sigma_scale = sigma_scale if scale_sigma else 1.0
         
    print("... doing bootstrap analysis for param fits.")
    roi_list = estats.rois  # Get list of all cells that pass fit-thr
    rdf_list = [estats.gdf.get_group(roi)[['config', 'trial', response_type]] for roi in roi_list]
    bootparams = {'row_vals': estats.fitinfo['row_vals'], 'col_vals': estats.fitinfo['col_vals'],
               'n_bootstrap_iters': n_bootstrap_iters, 'n_resamples': n_resamples,
               'ci': ci,
               'response_type': response_type}
    
    start_t = time.time()
    bootstrap_results = pool_bootstrap(rdf_list, bootparams, n_processes=n_processes)
    #except KeyboardInterrupt:
        #pool.terminate()
    end_t = time.time() - start_t
    print "Multiple processes: {0:.2f}sec".format(end_t)
    print "--- %i results" % len(bootstrap_results)

    if len(bootstrap_results)==0:
        return bootresults #None

    # Create dataframe of bootstrapped data
    bootdata = pd.concat(bootstrap_results)
    
    xx, yy, sigx, sigy = fitrf.convert_fit_to_coords(bootdata, 
                                                     estats.fitinfo['row_vals'], 
                                                     estats.fitinfo['col_vals'])
    bootdata['x0'] = xx
    bootdata['y0'] = yy
    bootdata['sigma_x'] = sigx * sigma_scale
    bootdata['sigma_y'] = sigy * sigma_scale
    
    # Calculate confidence intervals
    bootcis = get_cis_for_params(bootdata, ci=ci)

    # Plot bootstrapped distn of x0 and y0 parameters for each roi (w/ CIs)
    counts = bootdata.groupby(['cell']).count()['x0']
    unreliable = counts[counts < n_bootstrap_iters*0.5].index.tolist()
    print("%i cells seem to have unreliable estimates." % len(unreliable))
    
    bootresults = {'data': bootdata, 'params': bootparams, 'cis': bootcis}
    
    return bootresults

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

def plot_all_param_estimates(rid, meas_df, _fitr, _bootdata, ci=0.95, scale_sigma=True, sigma_scale=2.35):
    fig, axn = pl.subplots(2,3, figsize=(10,6))
    ax = axn.flat[0]
    ax = fitrf.plot_rf_map(_fitr['data'], cmap='inferno', ax=ax)
    ax = fitrf.plot_rf_ellipse(_fitr['fit_r'], ax=ax, scale_sigma=scale_sigma)
    params = ['sigma_x', 'sigma_y', 'theta', 'x0', 'y0']
    ai=1
    for param in params:
        ax = axn.flat[ai]
        ax = plot_bootstrapped_distribution(_bootdata[param], meas_df[param][rid], 
                                            ci=ci, ax=ax, param_name=param)
        ai += 1
    pl.subplots_adjust(wspace=0.7, hspace=0.5, top=0.8)
    fig.suptitle('rid %i' % rid)

    return fig

def get_reliable_fits(fit_results, bootresults, fit_thr=0.5, 
                      scale_sigma=True, ci=0.95, plot_boot_distn=True,
                      pass_all_params=True):

    bootdata = bootresults['data']
    cis = bootresults['cis']
    sigma_scale = 2.35 if scale_sigma else 1.0

    meas_df = fitrf.rfits_to_df(fit_results['fit_results'],
                                row_vals=fit_results['row_vals'],
                                col_vals=fit_results['col_vals'],
                                scale_size=scale_sigma, sigma_scale=sigma_scale)
    meas_df = meas_df[meas_df['r2']>fit_thr]
    pass_rois = meas_df.index.tolist()

    params = [p for p in meas_df.columns.tolist() if p!='r2']
    pass_cis = pd.concat([pd.DataFrame(
            [cis['%s_lower' % p][ri]<=meas_df[p][ri]<=cis['%s_upper' % p][ri] \
            for p in params], columns=[ri], index=params) for ri in meas_df.index.tolist()], axis=1).T
     
    if pass_all_params:
        keep_rids = [i for i in pass_cis.index.tolist() if all(pass_cis.loc[i])]
        pass_df = pass_cis.loc[keep_rids]
    else:   
        keep_rids = [i for i in pass_cis.index.tolist() if any(pass_cis.loc[i])]
        pass_df = pass_cis.loc[keep_rids]
        
    return pass_df


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

def do_regr_on_fov(bootdata, bootcis, posdf, cond='azimuth', ci=.95, xaxis_lim=None, 
                    filter_weird=False, plot_all_cis=False, deviant_color='dodgerblue'):
   
    '''
    Plot "scatter":
    
    1. Mark all ROIs with no fit (R2<=0.5)
    2. Linear regression + CI (based off of seaborn's function)
    3. Mark cells with reliable fits (R2>0.5, measured value w/in 95% CI)
    4. Mark reliable cells w/ CI outside of linear fit (1).
    
    ''' 
    fig, ax = pl.subplots(figsize=(10,8))
    ax.set_title(cond)
    ax.set_ylabel('RF position (rel. deg.)')
    ax.set_xlabel('FOV position (um)')
    if xaxis_lim is not None:
        ax.set_xlim([0, xaxis_lim])
    
    axname = 'xpos' if cond=='azimuth' else 'ypos'
    parname = 'x0' if cond=='azimuth' else 'y0'
    
    # Get lis of cells that pass bootstrap analysis
    roi_list = [k for k, g in bootdata.groupby(['cell'])]  
    
    # 1. Identify which cells fail bootstrap fits - do not include in fit.
    fail_rois = [r for r in posdf.index.tolist() if r not in roi_list]
    fail_df = posdf.loc[fail_rois]
    sns.regplot('%s_fov' % axname, '%s_rf' % axname, data=fail_df, ax=ax, 
                label='no_fit', color='gray', marker='x', fit_reg=False,
                scatter_kws=dict(s=15, alpha=0.5))

    # 2a. Linear regression, include cells with reliable fits (R2 + 95% CI) 
    ax, plotter = regplot('%s_fov' % axname, '%s_rf' % axname, 
                          data=posdf.loc[roi_list], ci=ci*100, 
                          color='k', marker='x',
                          scatter_kws=dict(s=15, alpha=1.0), ax=ax, 
                          label='reliable') #measured' )

    # 2b. Get CIs from linear fit (fit w/ reliable rois only)
    grid, yhat, err_bands = plotter.fit_regression(grid=plotter.x)
    e1 = err_bands[0, :] # err_bands[0, np.argsort(xvals)] <- sort by xpos to plot
    e2 = err_bands[1, :] #err_bands[1, np.argsort(xvals)]
    regr_cis = np.array([(ex, ey) for ex, ey in zip(e1, e2)])
    
    # Get mean and upper/lower CI bounds for each cell
    x0_meds = np.array([g[parname].mean() for k, g in bootdata.groupby(['cell'])])
    x0_lower = bootcis['%s_lower' % parname][roi_list]
    x0_upper = bootcis['%s_upper' % parname][roi_list]
   
    if filter_weird:
        # Filter cells with big CIs for plotting
        ci_intervals = bootcis['%s_upper' % parname] - bootcis['%s_lower' % parname]
        weird = [i for i in ci_intervals.index.tolist() if ci_intervals[i] > 10]
        print('weird: %i rois' % len(weird))
        rlist = [i for i in roi_list if i not in weird]
    else:
        rlist = copy.copy(roi_list)
    roi_ixs = np.array([roi_list.index(i) for i in rlist])

    if len(roi_ixs)==0:
        print("no good cells, returning")
        deviants = []
        bad_fits = []
        return fig, regr_cis, deviants, bad_fits
    
    xvals = posdf['%s_fov' % axname][rlist].values
    yvals = posdf['%s_rf' % axname][rlist].values 
    roi_lower = x0_lower.iloc[roi_ixs]
    roi_upper = x0_upper.iloc[roi_ixs]
    yerr_all = np.array(zip(x0_meds[roi_ixs]-x0_lower.iloc[roi_ixs], 
                            x0_upper.iloc[roi_ixs]-x0_meds[roi_ixs])).T
    if plot_all_cis:
        # Plot bootstrap results for each cell 
        ax.scatter(xvals, x0_meds[roi_ixs], c='k', marker='_', alpha=1.0, 
                   label='bootstrapped (%i%% CI)' % int(ci*100) )
        ax.errorbar(xvals, x0_meds[roi_ixs], yerr=yerr_all, 
                    fmt='none', color='k', alpha=0.7, lw=1)
    sns.despine(offset=4, trim=True, ax=ax)

    # Check that values make sense and mark deviants
    deviants = []
    bad_fits = []
    vals = [(rix, roi, posdf['%s_fov' % axname][roi], posdf['%s_rf' % axname][roi]) \
            for rix, roi, lo, up, (regL, regU), med 
            in zip(roi_ixs, rlist, roi_lower, roi_upper, regr_cis[roi_ixs], yvals)\
            if (lo <= med <= up) and ( ((regL > lo and regL > up) or (regU < lo and regU < up)) ) ]
    deviants = [v[1] for v in vals]
    xv = np.array([v[2] for v in vals])
    yv = np.array([v[3] for v in vals])
    dev_ixs = np.array([v[0] for v in vals])
    print("N deviants: %i" % len(dev_ixs))
    print("N attempts: %i" % len(x0_meds))

    # Color/mark reliable fits that are also deviants
    if len(dev_ixs) > 0:
        yerrs = np.array(zip(x0_meds[dev_ixs]-x0_lower.iloc[dev_ixs], 
                             x0_upper.iloc[dev_ixs]-x0_meds[dev_ixs])).T
        ax.scatter(xv, yv, c=deviant_color, marker='o', alpha=1.0, 
                    label='sig. scattered (%i%% CI)' % int(ci*100) )
        ax.scatter(xv, x0_meds[dev_ixs], c=deviant_color, marker='_', alpha=1.0) 
        ax.errorbar(xv, x0_meds[dev_ixs], yerr=yerrs, 
                        fmt='none', color=deviant_color, alpha=0.7, lw=1)
    ax.legend()

    bad_fits = [roi for rix, roi, lo, up, med \
                in zip(roi_ixs, rlist, roi_lower, roi_upper, yvals)\
                if not (lo <= med <= up) ]
    print("---> %s: deviants" % cond, deviants)
  
    return fig, regr_cis, deviants, bad_fits

def plot_regr_and_cis(bootresults, posdf, cond='azimuth', ci=.95, xaxis_lim=1200, ax=None):
    
    bootdata = bootresults['data']
    bootcis = bootresults['cis']
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
        print(np.where(np.isnan(xvals)))
        print(np.where(np.isnan(yvals)))
    regr.fit(xvals, yvals)
    fitv = regr.predict(xvals)
    if return_regr:
        return fitv.reshape(-1), regr
    else:
        return fitv.reshape(-1)


def plot_linear_regr_by_condition(posdf, rfdf):
    
    posdf = posdf.loc[rfdf.index]
    fig, axes = pl.subplots(2, 3, figsize=(10, 6))
    for ri, cond in enumerate(['azimuth', 'elevation']):
        axname = 'xpos' if cond=='azimuth' else 'ypos'
            
        yv = posdf['%s_rf' % axname].values
        xv = posdf['%s_fov' % axname].values    
        try:
            fitv, regr = fit_linear_regr(xv, yv, return_regr=True)
        except Exception as e:
            traceback.print_exc()
            print("Error fitting: rid %i" % ri)
            continue

        rmse = np.sqrt(skmetrics.mean_squared_error(yv, fitv))
        r2 = float(skmetrics.r2_score(yv, fitv))
        print("[%s] Mean squared error: %.2f" % (cond, rmse))
        print('[%s] Variance score: %.2f' % (cond, r2))
        
        ax=axes[ri, 0]
        ax.scatter(xv, yv, c='k', alpha=0.5)
        ax.set_ylabel('RF position (rel. deg.)')
        ax.set_xlabel('FOV position (um)')
        #ax.set_xlim([0, ylim])
        sns.despine(offset=1, trim=True, ax=ax)
        ax.plot(xv, fitv, 'r')
        
        r, p = spstats.pearsonr(posdf['xpos_fov'], posdf['xpos_rf'].abs())
        corr_str = 'pearson=%.2f (p=%.2f)' % (r, p)
        ax.plot(ax.get_xlim()[0], ax.get_ylim()[0], alpha=0, label=corr_str)
        ax.legend(loc='upper right', fontsize=8)
    
        ax = axes[ri, 1]
        ax.set_title(cond)
        residuals = fitv - yv
        ax.hist(residuals, histtype='step', color='k')
        sns.despine(offset=1, trim=True, ax=ax)
        ax.set_xlabel('residuals')
        ax.set_ylabel('counts')
        
        ax = axes[ri, 2]
        r2_vals = rfdf['r2']
        ax.scatter(r2_vals, abs(residuals), c='k', alpha=0.5)
        ax.set_xlabel('r2')
        ax.set_ylabel('abs(residuals)')
        
        testregr = LinearRegression()
        testregr.fit(r2_vals.reshape(-1, 1), residuals.reshape(-1, 1)) #, yv)
        r2_dist_corr = testregr.predict(r2_vals.reshape(-1, 1))
        ax.plot(r2_vals, r2_dist_corr, 'r')
        sns.despine(offset=1, trim=True, ax=ax)
        r, p = spstats.pearsonr(r2_vals.values, np.abs(residuals))
        corr_str2 = 'pearson=%.2f (p=%.2f)' % (r, p)
        ax.plot(ax.get_xlim()[0], ax.get_xlim()[-1], alpha=0, label=corr_str2)
        ax.legend(loc='upper right', fontsize=8)
        
    return fig

#
#%%

def compare_regr_to_boot_params(bootresults, fovinfo, outdir='/tmp', 
                                filter_weird=False, plot_all_cis=False, deviant_color='dodgerblue', 
                                data_id='METADATA'):

    '''
    deviants:  cells w/ good RF fits (boostrapped, measured lies within some CI), but
               even CI lies outside of estimated regression CI
    bad_fits:  cells w/ measured RF locations that do not fall within the CI from bootstrapping
    
    To get all "pass" rois, include all returned ROIs with fits that are NOT in bad_fits.
    '''
    bootdata = bootresults['data']
    bootcis = bootresults['cis']
    fit_rois = [int(k) for k, g in bootdata.groupby(['cell'])]    
 
    posdf = fovinfo['positions']
    xlim, ylim = fovinfo['xlim'], fovinfo['ylim']

    #% # Plot bootstrapped param CIs + regression CI
    xaxis_lim = max([xlim, ylim])
    regresults = {}
    evalresults = {}
    filter_str = '_filter-weird' if filter_weird else ''

    for cond in ['azimuth', 'elevation']:
        fig, regci, deviants, bad_fits = do_regr_on_fov(bootdata, bootcis, posdf, cond=cond, 
                                                        plot_all_cis=plot_all_cis, deviant_color=deviant_color,
                                                        filter_weird=filter_weird, xaxis_lim=xaxis_lim)
        
        #regresults[cond] = {'cis': regci, 'outliers': outliers}
        pass_rois = [i for i in fit_rois if i not in bad_fits]
        evalresults[cond] = {'deviants': deviants, 'bad_fits': bad_fits, 'pass_rois': pass_rois}
        regresults[cond] = {'cis': regci, 'deviants': deviants, 'bad_fits': bad_fits, 'pass_rois': pass_rois}
 
        label_figure(fig, data_id)
        pl.savefig(os.path.join(outdir, 'fit-regr_bootstrap-params_%s%s.svg' % (cond, filter_str)))
        pl.close()

    with open(os.path.join(outdir, 'good_bad_weird_rois_bycond.json'), 'w') as f:
        json.dump(evalresults, f, indent=4)    
    print("--- saved roi info after evaluation.")
  
    return regresults


def identify_deviants(regresults, bootresults, posdf, ci=0.95, rfdir='/tmp'):
    bootcis = bootresults['cis']

    # Which cells' CI contain the regression line, and which don't?
    print("Checking for deviants.")

    label_deviants = True
    deviants = {}
    conditions = regresults.keys() 
    bad_rois = intersection(regresults['azimuth']['bad_fits'], regresults['elevation']['bad_fits']) # these are cells whose mean value does not lie within the 95% CI 

    within = dict((k, []) for k in conditions)
    for cond in conditions:
        print("%s:  checking for deviants..." % cond)
        param = 'x0' if cond=='azimuth' else 'y0'
        axname = 'xpos' if cond=='azimuth' else 'ypos'
        
        regcis = regresults[cond]['cis']
        paramcis = [(c1, c2) for c1, c2 in zip(bootcis['%s_lower' % param], bootcis['%s_upper' % param])]
        roi_list = bootcis.index.tolist()
        
        #within=[]
        for roi, pci, rci in zip(roi_list, paramcis, regcis):
            if roi in bad_rois:
                continue
            if (rci[0] <= pci[0] <= rci[1]) or (rci[0] <= pci[1] <= rci[1]):
                within[cond].append(roi)
            elif (pci[0] <= rci[0] <= pci[1]) or (pci[0] <= rci[1] <= pci[1]):
                within[cond].append(roi)
        print("... %i out of %i cells' bootstrapped param distNs lie within %i%% CI of regression fit" % (len(within[cond]), len(roi_list), int(ci*100)))
        #trudeviants = [r for r in roi_list if r not in within and r not in regresults[cond]['outliers']]
        trudeviants = [r for r in roi_list if r not in within[cond] and r not in regresults[cond]['bad_fits']]

        print("... There are %i true deviants!" % len(trudeviants), trudeviants)
        
        fig, ax = pl.subplots(figsize=(20,10))
        ax = plot_regr_and_cis(bootresults, posdf, cond=cond, ax=ax)
        if len(trudeviants) > 0:
            deviant_fpos = posdf['%s_fov' % axname][trudeviants]
            deviant_rpos = posdf['%s_rf' % axname][trudeviants]
            ax.scatter(deviant_fpos, deviant_rpos, marker='*', c='dodgerblue', s=30, alpha=0.8)

            avg_interval = np.diff(ax.get_yticks()).mean()
            if label_deviants:
                deviant_ixs = [roi_list.index(d) for d in trudeviants]
                for ix, roi in zip(deviant_ixs, trudeviants):
                    if deviant_rpos[roi]  > max(regcis[ix]):
                        #print roi
                        ax.annotate(roi, (deviant_fpos[roi], deviant_rpos[roi]+avg_interval/2.), fontsize=6)
                    else:
                        ax.annotate(roi, (deviant_fpos[roi], deviant_rpos[roi]-avg_interval/2.), fontsize=6)        
            #ax.set_ylim([-10, 40])
            #ax.set_xlim([0, 800])
            
        sns.despine( trim=True, ax=ax)
      
        pl.savefig(os.path.join(rfdir, 'evaluation', 'regr-with-deviants_%s-test.svg' % cond))
        pl.close()

        deviants[cond] = trudeviants

    #%
    deviant_both = sorted(intersection(deviants['azimuth'], deviants['elevation']))
    #print deviant_both
    within_both = sorted(intersection(within['azimuth'], within['elevation']))
   
    deviants['bad'] = bad_rois
    deviants['pass'] = within_both
    deviants['deviant'] = deviant_both
 
    return deviants
   
    #%%
    

#%%
def evaluate_rfs(estats, rfdir='/tmp', response_type='dff', n_bootstrap_iters=1000, n_resamples=10,
                 ci=0.95, n_processes=1, sigma_scale=2.35, scale_sigma=True,
                 create_new=False, rootdir='/n/coxfs01/2p-data'):

    '''
    Evaluate receptive field fits for cells with R2 > fit_thr.

    Returns:
        bootresults = {'data': bootdata, 'params': bootparams, 'cis': bootcis}
        
        bootdata : dataframe containing results of param fits for bootstrap iterations
        bootparams: params used to do bootstrapping
        cis: confidence intervals of fit params

        If no fits, returns {}

    '''
    bootresults = None
    # Create output dir for bootstrap results
    evaldir = os.path.join(rfdir, 'evaluation')
    if not os.path.exists(os.path.join(evaldir)):
        os.makedirs(evaldir)
    rf_eval_fpath = os.path.join(evaldir, 'evaluation_results.pkl')
    
    do_evaluation = True
    if os.path.exists(rf_eval_fpath) and create_new is False:
        print("... loading existing evaluation results.")
        try:
            with open(rf_eval_fpath, 'rb') as f:
                bootresults = pkl.load(f)
            assert 'data' in bootresults.keys(), "... old datafile, redoing boot analysis"
            do_bootstrap = False
        except Exception as e:
            print("... ERROR loading evaluation results. Doing it now.")
       
    if do_bootstrap:
        bootresults = bootstrap_param_fits(estats, response_type=response_type,
                                             n_bootstrap_iters=n_bootstrap_iters, 
                                             n_resamples=n_resamples,
                                             ci=ci, n_processes=n_processes, 
                                             sigma_scale=sigma_scale,
                                             scale_sigma=scale_sigma)

        # Save results
        #if bootresults is not None:
        with open(rf_eval_fpath, 'wb') as f:
            pkl.dump(bootresults, f, protocol=pkl.HIGHEST_PROTOCOL)

        # Update params
        print("... updated eval params")
        eval_params_fpath = os.path.join(evaldir, 'evaluation_params.json')
        opts_dict = load_params(eval_params_fpath)
        opts_update = dict((k, eval(k)) for k in inspect.getargspec(evaluate_rfs).args)
        for k, v in opts_update:
            opts_dict.update({k: v})
        save_params(eval_params_fpath, opts_dict)

    return bootresults
        
def plot_boot_summary(rfdf, fit_results, bootresults, reliable_rois=[],
                        sigma_scale=2.35, scale_sigma=True, 
                        outdir='/tmp/rf_fit_evaluation', plot_format='svg',
                        data_id='DATA ID'):
    '''
    For all fit ROIs, plot summary of results (fit + evaluation).
    '''
    bootdata = bootresults['data']
    roi_list = rfdf.index.tolist() #sorted(bootdata['cell'].unique())
    for rid in roi_list:
        _fitr = fit_results['fit_results'][rid]
        _bootdata = bootdata[bootdata['cell']==rid]
        fig = plot_all_param_estimates(rid, rfdf, _fitr, _bootdata, 
                                        scale_sigma=scale_sigma, sigma_scale=sigma_scale)
        if rid in reliable_rois:
            fig.suptitle('rid %i**' % rid)
        label_figure(fig, data_id)
        pl.savefig(os.path.join(outdir, 'roi%05d.%s' % (int(rid+1), plot_format)))
        pl.close()
    return


def do_rf_fits_and_evaluation(animalid, session, fov, rfname=None, traceid='traces001', 
                              response_type='dff',
                              fit_thr=0.5, n_resamples=10, n_bootstrap_iters=1000, ci=0.95,
                              post_stimulus_sec=0., sigma_scale=2.35, scale_sigma=True,
                              #transform_fov=True, 
                              plot_boot_distns=True, calculate_metrics=False,
                              n_processes=1, filter_weird=False, plot_all_cis=False, 
                              deviant_color='dodgerblue', pass_all_params=True,
                              do_fits=False, do_evaluation=False, create_new=False,
                              rootdir='/n/coxfs01/2p-data', opts=None):

    from pipeline.python.classifications import experiment_classes as util
    #reload(util)
    #rfname= 'rfs' 
    #do_fits =False
     
    #%% Create session and experiment objects
    S = util.Session(animalid, session, fov)
    experiment_list = S.get_experiment_list(traceid=traceid)
    assert 'rfs' in experiment_list or 'rfs10' in experiment_list, "NO receptive field experiments found!"
    if rfname is None:
        rfname = 'rfs10' if 'rfs10' in experiment_list else 'rfs'      
    exp = util.ReceptiveFields(rfname, animalid, session, fov, 
                               traceid=traceid, trace_type='corrected')

    # Get RF dir for current fit type         
    rfdir, fit_desc = fitrf.create_rf_dir(exp.animalid, exp.session, exp.fov, 
                                          exp.name, traceid=exp.traceid,
                                          response_type=response_type, fit_thr=fit_thr)
    data_id = '|'.join([exp.animalid, exp.session, exp.fov, \
                            exp.traceid, exp.rois, exp.trace_type, fit_desc])
    #view_str = '_transformed' if transform_fov else '' 

    reload(fitrf)
    # Get RF params
    nframes_post = int(round(post_stimulus_sec*44.65))
    do_fits=False
    if not do_fits:
        try:
            fit_results, fit_params = fitrf.load_rf_fit_results(animalid, session,
                                                                fov, traceid=traceid,
                                                                experiment=rfname,
                                                                response_type=response_type)
            assert fit_params['nframes_post_onset'] == nframes_post, \
                "Incorrect nframes_post (found %i, requested %i" % (nframes_post, fit_params['nframes_post_onset'])
       
        except AssertionError as e:
            print(e)
            print("Redoing original fit")       
            do_fits = True
        except Exception as e:
            traceback.print_exc()
            print("[err]: unable to load fit results, re-fitting.")
            do_fits = True
            
    #%% Get RF fit stats
    estats = exp.get_stats(response_type=response_type, 
                            fit_thr=fit_thr, 
                            scale_sigma=scale_sigma,
                            nframes_post=nframes_post,
                            #create_new=calculate_metrics, #
                            do_fits=do_fits, 
                            pretty_plots=do_fits,
                            return_all_rois=False,
                            reload_data=reload_data) #True) 
   
    
    # Set directories
    evaldir = os.path.join(rfdir, 'evaluation')
    roidir = os.path.join(evaldir, 'rois_bootstrap-%i-iters_%i-resample' % (n_bootstrap_iters, n_resamples))
    if not os.path.exists(roidir):
        os.makedirs(roidir) 
    if os.path.exists(os.path.join(evaldir, 'rois')):
        shutil.rmtree(os.path.join(evaldir, 'rois'))
 
    #%% Do bootstrap analysis    
    bootresults = evaluate_rfs(estats, rfdir=rfdir, 
                                n_bootstrap_iters=n_bootstrap_iters, 
                                n_resamples=n_resamples,
                                ci=ci, n_processes=n_processes, 
                                sigma_scale=sigma_scale, scale_sigma=scale_sigma,
                                create_new=do_evaluation)

    #rf_eval_fpath = os.path.join(evaldir, 'evaluation_results.pkl')
    eval_params_fpath = os.path.join(rfdir, 'evaluation', 'evaluation_params.json')
    opts_dict = dict((k, eval(k)) for k in inspect.getargspec(do_rf_fits_and_evaluation).args)
    save_params(eval_params_fpath, opts_dict)
#

    if len(bootresults.keys())==0:# is None: # or 'data' not in bootresults:
        return {} #None

    bootdata = bootresults['data']
    cis = bootresults['cis']
    sigma_scale = sigma_scale if scale_sigma else 1.0
    rfdf = estats.fits.copy() # N cells fit w.o evaluation
    rfdf = rfdf[rfdf['r2']>fit_thr]
    pass_rois = rfdf.index.tolist()
    
    try:
        fit_results_dpath = os.path.join(rfdir, 'fit_results.pkl')
        with open(fit_results_dpath, 'rb') as f:
            fit_results = pkl.load(f)
    except Exception as e:
        traceback.print_exc()
        print("--- unable to load fit results from path:\n  %s" % fit_results_dpath)


    #%% Identify reliable fits 
    pass_cis = get_reliable_fits(fit_results, bootresults, pass_all_params=pass_all_params)
    reliable_rois = sorted(pass_cis.index.tolist())
     
    # Plot distribution of params w/ 95% CI
    if plot_boot_distns:
        print("... plotting boot distn.\n(to: %s" % roidir)
        for r in glob.glob(os.path.join(roidir, 'roi*')):
            os.remove(r)
        plot_boot_summary(rfdf, fit_results, bootresults, reliable_rois=reliable_rois,
                          sigma_scale=sigma_scale, scale_sigma=scale_sigma,
                          outdir=roidir, plot_format='svg', 
                          data_id=data_id)
    

    #% Fit linear regression for brain coords vs VF coords 
    fig = plot_linear_regr_by_condition( estats.fovinfo['positions'].loc[reliable_rois], 
                                     estats.fits.loc[reliable_rois] )
    pl.subplots_adjust(top=0.9, bottom=0.1, hspace=0.5)
    label_figure(fig, data_id)
    pl.savefig(os.path.join(evaldir, 'RF_fits-by-az-el.svg'))   
    pl.close()
    
    # Compare regression fit to bootstrapped params
    regresults = compare_regr_to_boot_params(bootresults, estats.fovinfo, 
                                            outdir=evaldir,
                                            data_id=data_id, 
                                            filter_weird=filter_weird, 
                                            plot_all_cis=plot_all_cis, 
                                            deviant_color=deviant_color)
    
    #%% Identify "deviants" based on spatial coordinates
    deviants = identify_deviants(regresults, bootresults, estats.fovinfo['positions'], 
                                 ci=ci, rfdir=rfdir)    
#    with open(os.path.join(rfdir, 'evaluation', 'deviants_bothconds.json'), 'w') as f:
#        json.dump(deviants, f, indent=4)


    #rf_eval_fpath = os.path.join(evaldir, 'evaluation_results.pkl')
    eval_params_fpath = os.path.join(rfdir, 'evaluation', 'evaluation_params.json')
    opts_dict = dict((k, eval(k)) for k in inspect.getargspec(do_rf_fits_and_evaluation).args)
    save_params(eval_params_fpath, opts_dict)
#
    return regresults #deviants

def load_params(params_fpath):
    with open(params_fpath, 'r') as f:
        options_dict = json.load(f)
    return options_dict

def save_params(params_fpath, opts):

    if isinstance(opts, dict):
        options_dict = vars(opts)
    else:
        options_dict = opts.copy()
    with open(params_fpath, 'w') as f:
        json.dump(options_dict, f, indent=4, sort_keys=True)
 
   
   
#%%
def extract_options(options):
    
    parser = optparse.OptionParser()

    parser.add_option('-D', '--root', action='store', dest='rootdir', default='/n/coxfs01/2p-data',\
                      help='data root dir (root project dir containing all animalids) [default: /n/coxfs01/2pdata]')
    parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')

    # Set specific session/run for current animal:
    parser.add_option('-S', '--session', action='store', dest='session', default='', \
                      help='session dir (format: YYYMMDD_ANIMALID')
    parser.add_option('-A', '--fov', action='store', dest='fov', default='FOV1_zoom2p0x', \
                      help="acquisition folder (ex: 'FOV2_zoom3x') [default: FOV1]")
    parser.add_option('-R', '--rfname', action='store', dest='rfname', default=None, \
                      help="name of rfs to process (default uses rfs10, if exists, else rfs)")
    parser.add_option('--fit', action='store_true', dest='do_fits', default=False, \
                      help="flag to do RF fitting anew")
    parser.add_option('--eval', action='store_true', dest='do_evaluation', default=False, \
                      help="flag to do RF evaluation anew")


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



    parser.add_option('--sigma', action='store', dest='sigma_scale', default=2.35, \
                      help="sigma scale factor for FWHM (default: 2.35)")
    parser.add_option('--no-scale', action='store_false', dest='scale_sigma', default=True, \
                      help="set to scale sigma to be true sigma, rather than FWHM")
    parser.add_option('-p', '--post', action='store', dest='post_stimulus_sec', default=0.0, 
                      help="N sec to include in stimulus-response calculation for maps (default:0.0)")


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


options = ['-i', animalid]
#%%

def main(options):
    opts = extract_options(options)
    animalid = opts.animalid
    session = opts.session
    fov = opts.fov
    traceid = opts.traceid
    response_type = opts.response_type
    rootdir = opts.rootdir
    
    n_resamples = opts.n_resamples
    n_bootstrap_iters = opts.n_bootstrap_iters
    # create_new = opts.create_new
    n_processes = int(opts.n_processes)
    do_fits = opts.do_fits
    do_evaluation = opts.do_evaluation
    pass_all_params = opts.pass_all_params
    
    ci = opts.ci
    #transform_fov = opts.transform_fov
    plot_boot_distns = opts.plot_boot_distns
    sigma_scale = float(opts.sigma_scale)
    rfname = opts.rfname
    filter_weird = opts.filter_weird
    plot_all_cis = opts.plot_all_cis
    deviant_color = opts.deviant_color

    scale_sigma = opts.scale_sigma
    post_stimulus_sec = float(opts.post_stimulus_sec)
    fit_thr = float(opts.fit_thr)
     
    from pipeline.python.classifications import experiment_classes as util


    deviants = do_rf_fits_and_evaluation(animalid, session, fov, rfname=rfname,
                              traceid=traceid, response_type=response_type, fit_thr=fit_thr,
                              n_bootstrap_iters=n_bootstrap_iters, n_resamples=n_resamples, ci=ci,
                              #transform_fov=transform_fov, 
                              plot_boot_distns=plot_boot_distns, 
                              post_stimulus_sec=post_stimulus_sec, 
                              n_processes=n_processes, filter_weird=filter_weird, plot_all_cis=plot_all_cis,
                              deviant_color=deviant_color, 
                              scale_sigma=scale_sigma, sigma_scale=sigma_scale,
                              pass_all_params=pass_all_params,
                              do_fits=do_fits, do_evaluation=do_evaluation,rootdir=rootdir,
                              opts=opts)
        
    print("***DONE!***")

if __name__ == '__main__':
    from pipeline.python.classifications import experiment_classes as util

    main(sys.argv[1:])
           
    #%%
    
    #options = ['-i', 'JC084', '-S', '20190525', '-A', 'FOV1_zoom2p0x', '-R', 'rfs']
    
