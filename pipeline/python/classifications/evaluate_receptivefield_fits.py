#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 11:10:13 2019

@author: julianarhee
"""
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
#from matplotlib.patches import Polygon

#from matplotlib.ticker import FormatStrFormatter
#from matplotlib.ticker import MaxNLocator

#from shapely.geometry import box


#import matplotlib_venn as mpvenn
#import itertools
import time
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


def group_configs(group, response_type):
    '''
    Takes each trial's reponse for specified config, and puts into dataframe
    '''
    config = group['config'].unique()[0]
    group.index = np.arange(0, group.shape[0])

    return pd.DataFrame(data={'%s' % config: group[response_type]})
    

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

#%%


from functools import partial
from contextlib import contextmanager

@contextmanager
def poolcontext(*args, **kwargs):
    pool = mp.Pool(*args, **kwargs)
    yield pool
    pool.terminate()


def bootstrap_rf_params(rdf, response_type='dff',
                        row_vals=[], col_vals=[], sigma_scale=2.35,
                        n_resamples=10, n_bootstrap_iters=1000):

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
            amp_f, x0_f, y0_f, sigx_f, sigy_f, theta_f, offset_f = fitr['popt']
            bparams.append(fitr['popt'])

    #%    
    if len(bparams)==0:
        return None

    bparams = np.array(bparams)   
    paramsdf = pd.DataFrame(data=bparams, columns=['amp', 'x0', 'y0', 'sigma_x', 'sigma_y', 'theta', 'offset'])
    paramsdf['cell'] = [rdf.index[0] for _ in range(bparams.shape[0])]
    
    return paramsdf


def pool_bootstrap(rdf_list, params, n_processes=1):
    
    with poolcontext(processes=n_processes) as pool:
        results = pool.map(partial(bootstrap_rf_params, 
                                   response_type=params['response_type'],
                                   row_vals=params['row_vals'], 
                                   col_vals=params['col_vals'],
                                   n_resamples=params['n_resamples'], 
                                   n_bootstrap_iters=params['n_bootstrap_iters']), rdf_list)
    return results
    
#def merge_names(a, b, c=1, d=2):
#    return '{} & {} - ({}, {})'.format(a, b, c, d)

#names = ['Brown', 'Wilson', 'Bartlett', 'Rivera', 'Molloy', 'Opie']
#with poolcontext(processes=2) as pool:
#    results = pool.map(partial(merge_names, b='Sons', c=100), names)
#print(results)
#

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
    


def bootstrap_param_fits(estats, response_type='dff',
                            n_bootstrap_iters=1000, n_resamples=10,
                            ci=0.95, n_processes=1, sigma_scale=2.35):
    
    print("... doing bootstrap analysis for param fits.")
    roi_list = estats.rois  # Get list of all cells that pass fit-thr
    rdf_list = [estats.gdf.get_group(roi)[['config', 'trial', response_type]] for roi in roi_list]
    bootparams = {'row_vals': estats.fitinfo['row_vals'], 'col_vals': estats.fitinfo['col_vals'],
               'n_bootstrap_iters': n_bootstrap_iters, 'n_resamples': n_resamples,
               'ci': ci,
               'response_type': response_type}
    
    start_t = time.time()
    bootstrap_results = pool_bootstrap(rdf_list, bootparams, n_processes=n_processes)
    end_t = time.time() - start_t
    print "Multiple processes: {0:.2f}sec".format(end_t)
    print "--- %i results" % len(bootstrap_results)

    if len(bootstrap_results)==0:
        return None

    # Create dataframe of bootstrapped data
    bootdata = pd.concat(bootstrap_results)
    
    xx, yy, sigx, sigy = fitrf.convert_fit_to_coords(bootdata, estats.fitinfo['row_vals'], estats.fitinfo['col_vals'])
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
    
    # Identify which cells fail bootstrap fits
    fail_rois = [r for r in posdf.index.tolist() if r not in roi_list]
    fadedf = posdf.loc[fail_rois]
    sns.regplot('%s_fov' % axname, '%s_rf' % axname, data=fadedf, color='gray', marker='x', fit_reg=False,
                scatter_kws=dict(s=15, alpha=0.5), ax=ax, label='no fit')

    # Plot cells that fit with bootstrap and fit linear model with them 
    ax, plotter = regplot('%s_fov' % axname, '%s_rf' % axname, data=posdf.loc[roi_list], ci=ci*100, 
                          color='k', marker='x',
                          scatter_kws=dict(s=15, alpha=1.0), ax=ax, 
                          label='measured' )

    # Get CIs from regression fit to "good data"
    grid, yhat, err_bands = plotter.fit_regression(grid=plotter.x)
    e1 = err_bands[0, :] # err_bands[0, np.argsort(xvals)] <- sort by xpos to plot
    e2 = err_bands[1, :] #err_bands[1, np.argsort(xvals)]
    regr_cis = np.array([(ex, ey) for ex, ey in zip(e1, e2)])
    
    # Get rois sorted by position:
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
    yerr_all = np.array(zip(x0_meds[roi_ixs]-x0_lower.iloc[roi_ixs], x0_upper.iloc[roi_ixs]-x0_meds[roi_ixs])).T
    if plot_all_cis:
        # Plot bootstrap results for each cell 
        ax.scatter(xvals, x0_meds[roi_ixs], c='k', marker='_', alpha=1.0, 
                   label='bootstrapped (%i%% CI)' % int(ci*100) )
        ax.errorbar(xvals, x0_meds[roi_ixs], yerr=yerr, 
                    fmt='none', color='k', alpha=0.7, lw=1)
        
    #if xaxis_lim is not None:
    #    ax.set_xticks(np.arange(0, xaxis_lim, 100))
        
    #ax.set_ylim([-10, 40])
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

    if len(dev_ixs) > 0:
        yerrs = np.array(zip(x0_meds[dev_ixs]-x0_lower.iloc[dev_ixs], x0_upper.iloc[dev_ixs]-x0_meds[dev_ixs])).T
        
        ax.scatter(xv, yv, c=deviant_color, marker='o', alpha=1.0, 
                    label='sig. scattered (%i%% CI)' % int(ci*100) )
        ax.scatter(xv, x0_meds[dev_ixs], c=deviant_color, marker='_', alpha=1.0) 
        ax.errorbar(xv, x0_meds[dev_ixs], yerr=yerrs, 
                        fmt='none', color=deviant_color, alpha=0.7, lw=1)

    ax.legend()

    bad_fits = [roi for rix, roi, lo, up, med \
                in zip(roi_ixs, rlist, roi_lower, roi_upper, yvals)\
                if not (lo <= med <= up) ]

#    for roi, lo, up, (regL, regU), med in zip(roi_list, roi_lower, roi_upper, regr_cis[roi_ixs], yvals):
#        if (lo <= med <= up): # Is measured value within lower/upper CIs?
#            if ((regL > lo and regL > up) or (regU < lo and regU < up)): 
#                # CI bounds are outside of regression CI -- these are deviants
#                xv = posdf['%s_fov' % axname][roi]
#                yv = posdf['%s_rf' % axname][roi]
#                ax.plot(xv, yv, marker='o', markersize=5, color='dodgerblue', alpha=1.0)
#                #ax.plot(xv, yv, marker='x', markersize=5, color='magenta', alpha=1.0)
#                deviants.append(roi)            
#        else:
#            # Measured not within CIs
#            bad_fits.append(roi)
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



#%%

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


def fit_linear_regr(xvals, yvals, return_regr=False):
    regr = LinearRegression()
    if len(xvals.shape) == 1:
        xvals = np.array(xvals).reshape(-1, 1)
        yvals = np.array(yvals).reshape(-1, 1)
    else:
        xvals = np.array(xvals)
        yvals = np.array(yvals)
    regr.fit(xvals, yvals)
    fitv = regr.predict(xvals)
    if return_regr:
        return fitv.reshape(-1), regr
    else:
        return fitv.reshape(-1)


def compare_fits_by_condition(posdf, rfdf):
    
    posdf = posdf.loc[rfdf.index]
    fig, axes = pl.subplots(2, 3, figsize=(10, 6))
    
    for ri, cond in enumerate(['azimuth', 'elevation']):
        axname = 'xpos' if cond=='azimuth' else 'ypos'
            
        yv = posdf['%s_rf' % axname].values
        xv = posdf['%s_fov' % axname].values    
        fitv, regr = fit_linear_regr(xv, yv, return_regr=True)
    
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



def spatially_sort_compare_position(fovinfo, transform=True):
    from pipeline.python.classifications import experiment_classes as util

    zimg = fovinfo['zimg']
    roi_contours = fovinfo['roi_contours']
    posdf = fovinfo['positions']
    npix_y, npix_x = zimg.shape
    
    # # Sort ROIs by x,y position:
    sorted_roi_indices_xaxis, sorted_roi_contours_xaxis = coor.spatially_sort_contours(roi_contours, sort_by='x', dims=(npix_x, npix_y))
    sorted_roi_indices_yaxis, sorted_roi_contours_yaxis = coor.spatially_sort_contours(roi_contours, sort_by='y', dims=(npix_x, npix_y))
    
    _, sorted_roi_centroids_xaxis = coor.spatially_sort_contours(roi_contours, sort_by='x', dims=(npix_x, npix_y), get_centroids=True)
    _, sorted_roi_centroids_yaxis = coor.spatially_sort_contours(roi_contours, sort_by='y', dims=(npix_x, npix_y), get_centroids=True)

    # x-axis in FOV = posterior to anterior, from left to right (0-->512)
    # y-axis in FOV = lateral to medial, from top to bottom (0-->512)
    
    # Color by RF position:
    rf_rois = posdf.index.tolist()
    #% #### Plot

    convert_um = True

    fig, axes = pl.subplots(2,2)
    fig.patch.set_alpha(1)
    ### Plot ALL sorted rois:
    ax = axes[0,1] 
    util.plot_roi_contours(zimg, sorted_roi_indices_xaxis, sorted_roi_contours_xaxis, 
                           label_rois=rf_rois, label=False, single_color=False, overlay=True,
                           clip_limit=0.02, draw_box=False, thickness=2, 
                           ax=ax, transform=transform)
    ax.axis('off')
                        
    ax = axes[0,0]
    util.plot_roi_contours(zimg, sorted_roi_indices_yaxis, sorted_roi_contours_yaxis,
                           label_rois=rf_rois, label=False, single_color=False, overlay=True,
                           clip_limit=0.02, draw_box=False, thickness=2, 
                           ax=ax, transform=transform)
    ax.axis('off')
                    
    ### Plot corresponding RF centroids:
    colors = ['k' for roi in rf_rois]
    units = 'um' if convert_um else 'pixels'
    # Get values for azimuth:    
    ax = axes[1,0]
    ax.scatter(posdf['xpos_fov'], posdf['xpos_rf'], c=colors, alpha=0.5)
    ax.set_title('Azimuth')
    ax.set_ylabel('RF position (rel. deg.)')
    ax.set_xlabel('FOV position (%s)' % units)
    #ax.set_xlim([0, ylim])
    sns.despine(offset=4, trim=True, ax=ax)
    # Get values for elevation:
    ax = axes[1,1]
    ax.scatter(posdf['ypos_fov'], posdf['ypos_rf'], c=colors, alpha=0.5)
    ax.set_title('Elevation')
    ax.set_ylabel('RF position (rel. deg.)')
    ax.set_xlabel('FOV position (%s)' % units)
    #ax.set_xlim([0, xlim])
    ax.axis('on')
    sns.despine(offset=4, trim=True, ax=ax)
    
    pl.subplots_adjust(wspace=0.5)
    
    return fig
    
#
def extract_options(options):
    
    parser = optparse.OptionParser()

    parser.add_option('-D', '--root', action='store', dest='rootdir', default='/n/coxfs01/2p-data',\
                      help='data root dir (root project dir containing all animalids) [default: /n/coxfs01/2pdata]')
    parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')

    # Set specific session/run for current animal:
    parser.add_option('-S', '--session', action='store', dest='session', default='', \
                      help='session dir (format: YYYMMDD_ANIMALID')
    parser.add_option('-A', '--fov', action='store', dest='fov', default='FOV1_zoom2p0x', \
                      help="acquisition folder (ex: 'FOV1_zoom3x') [default: FOV1]")
    parser.add_option('-R', '--rfname', action='store', dest='rfname', default=None, \
                      help="name of rfs to process (default uses rfs10, if exists, else rfs)")
    parser.add_option('--new', action='store_true', dest='create_new', default=False, \
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

    parser.add_option('--pixels', action='store_false', dest='transform_fov', default=True, \
                      help="flag to not convert fov space into microns (keep as pixels)")

    parser.add_option('--remove-weird', action='store_true', dest='filter_weird', default=False, \
                      help="flag to remove really funky fits")
    parser.add_option('--all-cis', action='store_true', dest='plot_all_cis', default=False, \
                      help="flag to plot CIs for all cells (not just deviants)")
    parser.add_option('-c', '--color', action='store', dest='deviant_color', default='dodgerblue', \
            help="color to plot deviants to stand out (default: dodgerblue)")



    parser.add_option('--sigma', action='store', dest='sigma_scale', default=2.35, \
                      help="sigma scale factor for FWHM (default: 2.35)")


    (options, args) = parser.parse_args(options)

    return options


#rootdir = '/n/coxfs01/2p-data'
#animalid = 'JC084' #JC076'
#session = '20190525' #'20190501'
#fov = 'FOV1_zoom2p0x'
#create_new = False

# Data type
#traceid = 'traces001'
#trace_type = 'corrected'
#response_type = 'dff'
#fit_thr = 0.5

# Bootstrap params
#n_bootstrap_iters=1000
#n_resamples = 10
#plot_boot_distns = True
#ci = 0.95
#n_processes=2
#sigma_scale = 2.35
#transform_fov = True



    #%%
def evaluate_rfs(estats, rfdir='/tmp', response_type='dff', n_bootstrap_iters=1000, n_resamples=10,
                 ci=0.95, n_processes=1, sigma_scale=2.35, 
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
    bootstrapdir = os.path.join(rfdir, 'evaluation')
    if not os.path.exists(os.path.join(bootstrapdir)):
        os.makedirs(bootstrapdir)
    bootstrap_fpath = os.path.join(bootstrapdir, 'evaluation_results.pkl')
    
    do_bootstrap = False
    if os.path.exists(bootstrap_fpath) and create_new is False:
        print("... loading existing evaluation results.")
        try:
            with open(bootstrap_fpath, 'rb') as f:
                bootresults = pkl.load(f)
            assert 'data' in bootresults.keys(), "... old datafile, redoing boot analysis"
        except Exception as e:
            do_bootstrap = True
    else:
        do_bootstrap = True
        
    if do_bootstrap:
        bootresults = bootstrap_param_fits(estats, response_type=response_type,
                                             n_bootstrap_iters=n_bootstrap_iters, n_resamples=n_resamples,
                                             ci=ci, n_processes=n_processes, sigma_scale=sigma_scale)
    
        # Save results
        if bootresults is not None:
            with open(bootstrap_fpath, 'wb') as f:
                pkl.dump(bootresults, f, protocol=pkl.HIGHEST_PROTOCOL)

    return bootresults
            
         


#%%

def compare_regr_to_boot_params(bootresults, fovinfo, evaldir='/tmp', 
                                filter_weird=False, plot_all_cis=False, deviant_color='dodgerblue', 
                                statsdir='/tmp', data_identifier='METADATA'):

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
 
        label_figure(fig, data_identifier)
        pl.savefig(os.path.join(evaldir, 'fit-regr_bootstrap-params_%s%s.svg' % (cond, filter_str)))
        pl.savefig(os.path.join(statsdir, 'RFs__fit-regr-bootstrap-params_%s%s.svg' % (cond, filter_str) )) 
        #pl.savefig(os.path.join(statsdir, 'receptive_fields', 'fit-regr_bootstrap-params_%s.svg' % cond))
        pl.close()

    with open(os.path.join(evaldir, 'good_bad_weird_rois_bycond.json'), 'w') as f:
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
                        print roi
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
    


def do_rf_fits_and_evaluation(animalid, session, fov, rfname=None, traceid='traces001', response_type='dff',
                              fit_thr=0.5, n_resamples=10, n_bootstrap_iters=1000, ci=0.95,
                              transform_fov=True, plot_boot_distns=True, sigma_scale=2.35,
                              n_processes=1, filter_weird=False, plot_all_cis=False, deviant_color='dodgerblue',
                              create_new=False, rootdir='/n/coxfs01/2p-data'):

    from pipeline.python.classifications import experiment_classes as util

    #%% Create session and experiment objects
    S = util.Session(animalid, session, fov)
    experiment_list = S.get_experiment_list(traceid=traceid)
    assert 'rfs' in experiment_list or 'rfs10' in experiment_list, "NO receptive field experiments found!"
    if rfname is None:
        rfname = 'rfs10' if 'rfs10' in experiment_list else 'rfs'      
    exp = util.ReceptiveFields(rfname, S.animalid, S.session, S.fov, 
                               traceid=traceid, trace_type='corrected')

    # Create output dir in "summmaries" folder
    statsdir, stats_desc = util.create_stats_dir(exp.animalid, exp.session, exp.fov,
                                                  traceid=exp.traceid, trace_type=exp.trace_type,
                                                  response_type=response_type, 
                                                  responsive_test=None, responsive_thr=0)
    
    if not os.path.exists(os.path.join(statsdir, 'receptive_fields')):
        os.makedirs(os.path.join(statsdir, 'receptive_fields'))
    print("Saving stats output to: %s" % statsdir)    

    #%% Get RF fit stats
    estats = exp.get_stats(response_type=response_type, fit_thr=fit_thr) 

    # Get RF dir for current fit type         
    rfdir, fit_desc = fitrf.create_rf_dir(exp.animalid, exp.session, exp.fov, exp.name, traceid=exp.traceid,
                                    response_type=response_type, fit_thr=fit_thr)
    data_identifier = '|'.join([exp.animalid, exp.session, exp.fov, \
                            exp.traceid, exp.rois, exp.trace_type, fit_desc])
    view_str = '_transformed' if transform_fov else '' 

    #% Plot spatially ordered rois
    if len(glob.glob(os.path.join(statsdir, 'receptive_fields', 'spatially_sorted*.svg'))) == 0: 
        print("Getting FOV info for rois.")
        fig = spatially_sort_compare_position(estats.fovinfo, transform=transform_fov)
        label_figure(fig, data_identifier)
        pl.savefig(os.path.join(statsdir, 'receptive_fields', \
                'spatially_sorted_rois_%s%s.svg' % (rfname, view_str)))
        pl.close()


    #%% Do bootstrap analysis
    bootstrapdir = os.path.join(rfdir, 'evaluation')
    roidir = os.path.join(bootstrapdir, 'rois_bootstrap-%i-iters_%i-resample' % (n_bootstrap_iters, n_resamples))
    if not os.path.exists(roidir):
        os.makedirs(roidir) 
    if os.path.exists(os.path.join(bootstrapdir, 'rois')):
        shutil.rmtree(os.path.join(bootstrapdir, 'rois'))
        
    bootresults = evaluate_rfs(estats, rfdir=rfdir, n_bootstrap_iters=n_bootstrap_iters, n_resamples=n_resamples,
                               ci=ci, n_processes=n_processes, sigma_scale=sigma_scale, create_new=create_new)
  
    if bootresults is None: # or 'data' not in bootresults:
        return {} #None

    #%% Identify "deviants" based on spatial coordinates
    deviants = {}

    # Plot distribution of params w/ 95% CI
    rfdf = estats.fits # N cells fit w.o evaluation
    roi_list = rfdf.index.tolist()

    if plot_boot_distns:
        for roi, paramsdf in bootresults['data'].groupby(['cell']):
            true_x = rfdf['x0'][roi]
            true_y = rfdf['y0'][roi]
            fig = plot_bootstrapped_position_estimates(paramsdf['x0'], paramsdf['y0'], true_x, true_y, ci=ci)
            fig.suptitle('roi %i' % int(roi+1))        
            pl.savefig(os.path.join(roidir, 'roi%05d' % (int(roi+1))))
            pl.close()

    #% Fit linear regression for brain coords vs VF coords
    print("Doing linear regression of RF position on FOV position.")
    fig = compare_fits_by_condition( estats.fovinfo['positions'].loc[roi_list], estats.fits )
    pl.subplots_adjust(top=0.9, bottom=0.1, hspace=0.5)
    label_figure(fig, data_identifier)
    pl.savefig(os.path.join(statsdir, 'receptive_fields', 'RF_fits-by-az-el.svg'))
    pl.close()
    
    # Compare regression fit to bootstrapped params
    regresults = compare_regr_to_boot_params(bootresults, estats.fovinfo, evaldir=bootstrapdir,
                                    statsdir=statsdir, data_identifier=data_identifier, 
                                    filter_weird=filter_weird, plot_all_cis=plot_all_cis, 
                                    deviant_color=deviant_color)
    
    # Identify deviants
    deviants = identify_deviants(regresults, bootresults, estats.fovinfo['positions'], ci=ci, rfdir=rfdir)
    
#    with open(os.path.join(rfdir, 'evaluation', 'deviants_bothconds.json'), 'w') as f:
#        json.dump(deviants, f, indent=4)
#
    return regresults #deviants


#%%

def main(options):
    opts = extract_options(options)
    animalid = opts.animalid
    session = opts.session
    fov = opts.fov
    traceid = opts.traceid
    response_type = opts.response_type
    fit_thr = opts.fit_thr
    rootdir = opts.rootdir
    
    n_resamples = opts.n_resamples
    n_bootstrap_iters = opts.n_bootstrap_iters
    create_new = opts.create_new
    n_processes = int(opts.n_processes)
    
    ci = opts.ci
    transform_fov = opts.transform_fov
    plot_boot_distns = opts.plot_boot_distns
    sigma_scale = opts.sigma_scale
    rfname = opts.rfname
    filter_weird = opts.filter_weird
    plot_all_cis = opts.plot_all_cis
    deviant_color = opts.deviant_color

    from pipeline.python.classifications import experiment_classes as util

    deviants = do_rf_fits_and_evaluation(animalid, session, fov, rfname=rfname,
                              traceid=traceid, response_type=response_type, fit_thr=fit_thr,
                              n_bootstrap_iters=n_bootstrap_iters, n_resamples=n_resamples, ci=ci,
                              transform_fov=transform_fov, plot_boot_distns=plot_boot_distns, sigma_scale=sigma_scale,
                              n_processes=n_processes, filter_weird=filter_weird, plot_all_cis=plot_all_cis,
                              deviant_color=deviant_color,
                              create_new=create_new, rootdir=rootdir)
        
    print("***DONE!***")

if __name__ == '__main__':
    from pipeline.python.classifications import experiment_classes as util

    main(sys.argv[1:])
           
    #%%
    
    options = ['-i', 'JC084', '-S', '20190525', '-A', 'FOV1_zoom2p0x', '-R', 'rfs']
    
