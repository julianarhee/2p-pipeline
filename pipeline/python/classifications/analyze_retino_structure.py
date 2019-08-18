#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 11:10:13 2019

@author: julianarhee
"""
import os
import glob
import json
import h5py
import copy
import cv2
import imutils

import numpy as np
import pandas as pd
import pylab as pl
import seaborn as sns
import cPickle as pkl
import matplotlib.gridspec as gridspec

from pipeline.python.classifications import experiment_classes as util
from pipeline.python.classifications import test_responsivity as resp
from pipeline.python.classifications import responsivity_stats as rstats
from pipeline.python.utils import label_figure, natural_keys, convert_range
from pipeline.python.retinotopy import convert_coords as coor
from pipeline.python.retinotopy import fit_2d_rfs as fitrf
from matplotlib.patches import Ellipse, Rectangle

from shapely.geometry.point import Point
from shapely import affinity
from matplotlib.patches import Polygon

from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import MaxNLocator

from shapely.geometry import box


import matplotlib_venn as mpvenn
import itertools
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

def get_empirical_ci(stat, alpha=0.95):
    p = ((1.0-alpha)/2.0) * 100
    lower = np.percentile(stat, p) #max(0.0, np.percentile(stat, p))
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = np.percentile(stat, p) # min(1.0, np.percentile(x0, p))
    #print('%.1f confidence interval %.2f and %.2f' % (alpha*100, lower, upper))
    return lower, upper

def plot_bootstrapped_position_estimates(x0, y0, true_x, true_y, alpha=0.9):
    lower_x0, upper_x0 = get_empirical_ci(x0, alpha=alpha)
    lower_y0, upper_y0 = get_empirical_ci(y0, alpha=alpha)

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


def bootstrap_rf_params(rdf, rows=[], cols=[], sigma_scale=2.35,
                            n_resamples=10, n_bootstrap_iters=1000):

    xres = np.unique(np.diff(row_vals))[0]
    yres = np.unique(np.diff(col_vals))[0]
    min_sigma=5; max_sigma=50;

    grouplist = [group_configs(group, response_type) for config, group in rdf.groupby(['config'])]
    responses_df = pd.concat(grouplist, axis=1) # indices = trial reps, columns = conditions

    # Get mean response across re-sampled trials for each condition, do this n-bootstrap-iters times
    bootdf = pd.concat([responses_df.sample(n_resamples, replace=True).mean(axis=0) for ni in range(n_bootstrap_iters)], axis=1)
    
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
                
        if fitr['success']:
            amp_f, x0_f, y0_f, sigx_f, sigy_f, theta_f, offset_f = fitr['popt']
            bparams.append(fitr['popt'])

    #%    
    bparams = np.array(bparams)  
    paramsdf = pd.DataFrame(data=bparams, columns=['amp', 'x0', 'y0', 'sigma_x', 'sigma_y', 'theta', 'offset'])
    paramsdf['cell'] = [rdf.index[0] for _ in range(bparams.shape[0])]
    
    return paramsdf

#def pool_bootstrap(rdf_list, n_processes=1):
#    pool = mp.Pool(processes=n_processes)
#    results = pool.map(bootstrap_roi_responses, rdf_list)
#    return results

def pool_bootstrap(rdf_list, params, n_processes=1):
    
    with poolcontext(processes=n_processes) as pool:
        results = pool.map(partial(bootstrap_rf_params, 
                                   rows=params['row_vals'], 
                                   cols=params['col_vals'],
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

def get_cis_for_params(bdata, alpha=0.95):
    roi_list = [roi for roi, bdf in bdata.groupby(['cell'])]
    param_names = [p for p in bdata.columns if p != 'cell']
    CI = {}
    for p in param_names:
        CI[p] = dict((roi, get_empirical_ci(bdf[p].values, alpha=alpha)) for roi, bdf in bdata.groupby(['cell']))
    
    cis = {}
    for p in param_names:
        cvals = np.array([get_empirical_ci(bdf[p].values, alpha=alpha) for roi, bdf in bdata.groupby(['cell'])])
        cis['%s_lower' % p] = cvals[:, 0]
        cis['%s_upper' % p] = cvals[:, 1]
    cis = pd.DataFrame(cis, index=[roi_list])
    
    return cis
    

def get_bootstrapped_params(estats, rfdir='/tmp', n_bootstrap_iters=1000, n_resamples=10,
                            plot_distns=True, alpha=0.95, n_processes=1, sigma_scale=2.35):
    
    # Create output dir for bootstrap results
    bootstrapdir = os.path.join(rfdir, 'evaluation')
    if not os.path.exists(os.path.join(bootstrapdir, 'rois')):
        os.makedirs(os.path.join(bootstrapdir, 'rois'))
    
    bootstrap_fpath = os.path.join(bootstrapdir, 'evaluation_results.pkl')
    
    do_bootstrap = False
    if os.path.exists(bootstrap_fpath):
        try:
            with open(bootstrap_fpath, 'rb') as f:
                res = pkl.load(f)
            bdata = res['bootstrap_data']
            bparams = res['bootstrap_params']
            cis = res['confidence_intervals']
        except Exception as e:
            do_bootstrap = True
    else:
        do_bootstrap = True
    
    if do_bootstrap:
        roi_list = estats.rois  # Get list of all cells that pass fit-thr
        rdf_list = [estats.gdf.get_group(roi)[['config', 'trial', response_type]] for roi in roi_list]
        bparams = {'row_vals': row_vals, 'col_vals': col_vals,
                   'n_bootstrap_iters': n_bootstrap_iters, 'n_resamples': n_resamples,
                   'alpha': alpha}
        
        start_t = time.time()
        bootstrap_results = pool_bootstrap(rdf_list, bparams, n_processes=n_processes)
        end_t = time.time() - start_t
        print "Multiple processes: {0:.2f}sec".format(end_t)
        
        bdata = pd.concat(bootstrap_results)
        
        xx, yy, sigx, sigy = fitrf.convert_fit_to_coords(bdata, row_vals, col_vals)
        bdata['x0'] = xx
        bdata['y0'] = yy
        bdata['sigma_x'] = sigx * sigma_scale
        bdata['sigma_y'] = sigy * sigma_scale
        
        # Calculate confidence intervals
        cis = get_cis_for_params(bdata, alpha=alpha)

        # Save results
        res = {'bootstrap_data': bdata, 'bootstrap_params': bparams, 'confidence_intervals': cis}
        with open(bootstrap_fpath, 'wb') as f:
            pkl.dump(res, f, protocol=pkl.HIGHEST_PROTOCOL)

        # Plot bootstrapped distn of x0 and y0 parameters for each roi (w/ CIs)
        counts = bdata.groupby(['cell']).count()['x0']
        unreliable = counts[counts < n_bootstrap_iters*0.5].index.tolist()
        print("%i cells seem to have unreliable estimates." % len(unreliable))
        
        # Plot distribution of params w/ 95% CI
        if plot_distns:
            for roi, paramsdf in bdata.groupby(['cell']):
                
                true_x = estats.fits['x0'][roi]
                true_y = estats.fits['y0'][roi]
                fig = plot_bootstrapped_position_estimates(paramsdf['x0'], paramsdf['y0'], true_x, true_y, alpha=alpha)
                fig.suptitle('roi %i' % int(roi+1))
                
                pl.savefig(os.path.join(bootstrapdir, 'rois', 'roi%05d_%i-bootstrap-iters_%i-resample' % (int(roi+1), n_bootstrap_iters, n_resamples)))
                pl.close()

    return bdata, bparams, cis

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



def plot_regr_and_bootstrapped_params(bdata, cis, posdf, cond='azimuth', alpha=.95, xaxis_lim=1200):
    
    fig, ax = pl.subplots(figsize=(15,10))
    
    ax.set_title(cond)
    ax.set_ylabel('RF position (rel. deg.)')
    ax.set_xlabel('FOV position (um)')
    ax.set_xlim([0, xaxis_lim])
    
    axname = 'xpos' if cond=='azimuth' else 'ypos'
    parname = 'x0' if cond=='azimuth' else 'y0'
    
#    g = sns.regplot('%s_fov' % axname, '%s_rf' % axname, data=posdf, ci=alpha*100, color='r', marker='o',
#                scatter_kws=dict(s=5, alpha=0.5), ax=ax, label='measured (regr: %i%% CI)' % int(alpha*100) )

    sax, plotter = regplot('%s_fov' % axname, '%s_rf' % axname, data=posdf, ci=alpha*100, color='r', marker='o',
                scatter_kws=dict(s=5, alpha=0.5), ax=ax, label='measured (regr: %i%% CI)' % int(alpha*100) )
    
    grid, yhat, err_bands = plotter.fit_regression(grid=plotter.x)

    roi_list = [k for k, g in bdata.groupby(['cell'])]    
    xvals = posdf['%s_fov' % axname][roi_list].values
    yvals = posdf['%s_rf' % axname][roi_list].values
    
    # GET CI limits
    e1 = err_bands[0, :] # err_bands[0, np.argsort(xvals)] <- sort by xpos to plot
    e2 = err_bands[1, :] #err_bands[1, np.argsort(xvals)]
    regr_cis = [(ex, ey) for ex, ey in zip(e1, e2)]
    
    # Get rois sorted by position:
    x0_meds = [g[parname].mean() for k, g in bdata.groupby(['cell'])]
    x0_lower = cis['%s_lower' % parname][roi_list]
    x0_upper = cis['%s_upper' % parname][roi_list]
    
    ax.scatter(xvals, x0_meds, c='k', marker='_', label='bootstrapped (%i%% CI)' % int(alpha*100) )
    ax.errorbar(xvals, x0_meds, yerr=np.array(zip(x0_meds-x0_lower, x0_upper-x0_meds)).T, 
            fmt='none', color='k', alpha=0.5)
    ax.set_xticks(np.arange(0, xaxis_lim, 100))
    sns.despine(offset=1, trim=True, ax=ax)
    
    ax.legend()

    # Check that values make sense
    deviants = []
    for roi,lo,up,med in zip(roi_list, x0_lower, x0_upper, yvals):
        if lo <= med <= up:
            continue
        else:
            deviants.append(roi)
            
    return fig, regr_cis, deviants


def plot_regr_and_cis(bdata, cis, posdf, cond='azimuth', alpha=.95, xaxis_lim=1200, ax=None):
    
    if ax is None:
        fig, ax = pl.subplots(figsize=(15,10))
    
    ax.set_title(cond)
    ax.set_ylabel('RF position (rel. deg.)')
    ax.set_xlabel('FOV position (um)')
    ax.set_xlim([0, xaxis_lim])
    
    axname = 'xpos' if cond=='azimuth' else 'ypos'
    parname = 'x0' if cond=='azimuth' else 'y0'
    
    g = sns.regplot('%s_fov' % axname, '%s_rf' % axname, data=posdf, ci=alpha*100, color='r', marker='o',
                scatter_kws=dict(s=5, alpha=0.5), ax=ax, label='measured (regr: %i%% CI)' % int(alpha*100) )


    roi_list = [k for k, g in bdata.groupby(['cell'])]    
    xvals = posdf['%s_fov' % axname][roi_list].values
    yvals = posdf['%s_rf' % axname][roi_list].values
    
    # Get rois sorted by position:
    x0_meds = [g[parname].mean() for k, g in bdata.groupby(['cell'])]
    x0_lower = cis['%s_lower' % parname][roi_list]
    x0_upper = cis['%s_upper' % parname][roi_list]
    
    ax.scatter(xvals, x0_meds, c='k', marker='_', label='bootstrapped (%i%% CI)' % int(alpha*100) )
    ax.errorbar(xvals, x0_meds, yerr=np.array(zip(x0_meds-x0_lower, x0_upper-x0_meds)).T, 
            fmt='none', color='k', alpha=0.5)
    ax.set_xticks(np.arange(0, xaxis_lim, 100))
    sns.despine(offset=1, trim=True, ax=ax)
    
    ax.legend()
            
    return ax



#%%

def visualize_bootstrapped_params(bdata, sorted_rois=[], nplot=20, rank_type='R2'):
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
        tmpd['rank'] = [sorted_r2[roi] for _ in range(len(tmpd))]
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


def compare_fits_by_condition(posdf):
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
        r2_vals = estats.fits['r2']
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
        ax.legend(loc='uppder right', fontsize=8)
        
    return fig



def spatially_sort_compare_position(zimg, roi_contours, posdf, transform=True):
    
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
                           label_rois=rf_rois, label=False, single_color=False, overlay=False,
                           clip_limit=0.02, draw_box=False, thickness=2, 
                           ax=ax, transform=transform)
    ax.axis('off')
                        
    ax = axes[0,0]
    util.plot_roi_contours(zimg, sorted_roi_indices_yaxis, sorted_roi_contours_yaxis, label_rois=rf_rois,
                           clip_limit=0.008, label=False, draw_box=False, 
                           thickness=2, roi_color=(255, 255, 255), single_color=False, ax=ax, transform=transform)
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
    ax.set_xlim([0, ylim])
    sns.despine(offset=4, trim=True, ax=ax)
    # Get values for elevation:
    ax = axes[1,1]
    ax.scatter(posdf['ypos_fov'], posdf['ypos_rf'], c=colors, alpha=0.5)
    ax.set_title('Elevation')
    ax.set_ylabel('RF position (rel. deg.)')
    ax.set_xlabel('FOV position (%s)' % units)
    ax.set_xlim([0, xlim])
    ax.axis('on')
    sns.despine(offset=4, trim=True, ax=ax)
    
    return fig
    
    

# In[2]:


rootdir = '/n/coxfs01/2p-data'
animalid = 'JC084' #JC076'
session = '20190522' #'20190501'
fov = 'FOV1_zoom2p0x'
create_new = False

# Data type
traceid = 'traces001'
trace_type = 'corrected'
response_type = 'dff'
fit_thr = 0.5

# Bootstrap params
n_bootstrap_iters=1000
n_resamples = 10
plot_distns = True
alpha = 0.95
n_processes=2
sigma_scale = 2.35

#%%
# Create session and experiment objects
S = util.Session(animalid, session, fov)
experiment_list = S.get_experiment_list(traceid=traceid)
assert 'rfs' in experiment_list or 'rfs10' in experiment_list, "NO receptive field experiments found!"
rfname = 'rfs10' if 'rfs10' in experiment_list else 'rfs'
exp = util.ReceptiveFields(rfname, S.animalid, S.session, S.fov, traceid=traceid)
exp.print_info()

# Get output dir for stats
estats = exp.get_stats(response_type=response_type, fit_thr=fit_thr)
statsdir, stats_desc = rstats.create_stats_dir(exp.animalid, exp.session, exp.fov,
                                              traceid=exp.traceid, trace_type=exp.trace_type,
                                              response_type=response_type)
print("Saving stats output to: %s" % statsdir)

# Get RF dir for current fit type
fit_desc = fitrf.get_fit_desc(response_type=response_type)
rfdir = glob.glob(os.path.join(rootdir, exp.animalid, exp.session, exp.fov, 
                               exp.name, 'traces/%s*' % exp.traceid,
                               'receptive_fields', fit_desc))[0]

data_identifier = '|'.join([exp.animalid, exp.session, exp.fov, exp.traceid, exp.rois, exp.trace_type, response_type])
        
# Get RF protocol info
row_vals, col_vals = estats.fitinfo['row_vals'], estats.fitinfo['col_vals']

# Get roi list
roi_list = estats.rois  # Get list of all cells that pass fit-thr

#%% Bootstrap RF parameter estimates
bdata, bparams, cis = get_bootstrapped_params(estats, rfdir=rfdir, n_bootstrap_iters=n_bootstrap_iters, 
                                         n_resamples=n_resamples, plot_distns=plot_distns, 
                                         n_processes=n_processes, alpha=alpha)


#%% Plot estimated x0, y0 as a function of r2 rank (plot top 30 neurons)

rank_type = 'r2'
sorted_r2 = estats.fits['r2'].argsort()[::-1]
sorted_rois = np.array(roi_list)[sorted_r2.values]
for roi in sorted_rois:
    print roi, estats.fits['r2'][roi]

nplot = 20
fig = visualize_bootstrapped_params(bdata, sorted_rois=sorted_rois, nplot=nplot, rank_type='r2')

label_figure(fig, data_identifier)
figname = 'RF_bootstrapped_distns_by_param_top%i_by_R2' % nplot
pl.savefig(os.path.join(statsdir, 'figures', '%s.png' % figname))

#%% Load session's rois:
    
#S.load_data(rfname, traceid=traceid) # Load data to get traceid and roiid
masks, zimg = S.load_masks(rois=exp.rois)
npix_y, npix_x = zimg.shape

# Create contours from maskL
roi_contours = coor.contours_from_masks(masks)

# Convert to brain coords
fov_x, rf_x, xlim, fov_y, rf_y, ylim = coor.get_roi_position_um(estats.fits, roi_contours, 
                                                                 rf_exp_name=rfname,
                                                                 convert_um=True,
                                                                 npix_y=npix_y,
                                                                 npix_x=npix_x)

posdf = pd.DataFrame({'xpos_fov': fov_y,
                       'xpos_rf': rf_x,
                       'ypos_fov': fov_x,
                       'ypos_rf': rf_y}, index=roi_list)
            
            
#%%
plot_spatial = False
transform_fov=True

if plot_spatial:
    
    fig = spatially_sort_compare_position(zimg, roi_contours, posdf, transform=transform_fov)
    #%
    label_figure(fig, data_identifier)
    if transform_fov:
        pl.savefig(os.path.join(statsdir, 'figures', 'transformed_spatially_sorted_rois_%s.png' % (rfname)))
    else:
        pl.savefig(os.path.join(statsdir, 'figures', 'spatially_sorted_rois_%s.png' % (rfname)))


#%% Fit linear regression for brain coords vs VF coords

fig = compare_fits_by_condition(posdf)
pl.subplots_adjust(top=0.9, bottom=0.1, hspace=0.5)
label_figure(fig, data_identifier)
pl.savefig(os.path.join(statsdir, 'RF_fits-by-az-el.png'))


#%
#fig, axes = pl.subplots(1,2)
#
#sns.regplot('xpos_fov', 'xpos_rf', data=posdf, ci=alpha, color='k', marker='o',
#            scatter_kws=dict(s=5, alpha=0.5), ax=axes[0])
#sns.regplot('ypos_fov', 'ypos_rf', data=posdf, ci=alpha, color='k', marker='o',
#            scatter_kws=dict(s=5, alpha=0.5), ax=axes[1])


#%% # Plot bootstrapped param CIs + regression CI

xaxis_lim = max([xlim, ylim])
reg_info={}
for cond in ['azimuth', 'elevation']:
    fig, regci, deviants = plot_regr_and_bootstrapped_params(bdata, cis, posdf, cond=cond, xaxis_lim=xaxis_lim)
    
    reg_info[cond] = {'cis': regci, 'deviants': deviants}
    
    label_figure(fig, data_identifier)
    pl.savefig(os.path.join(statsdir, 'figures', 'fit-regr_bootstrap-params_%s.png' % cond))
    

#%% # Which cells' CI contain the regression line, and which don't?

deviants = {}
conditions = ['azimuth', 'elevation']
for cond in conditions:
    param = 'x0' if cond=='azimuth' else 'y0'
    axname = 'xpos' if cond=='azimuth' else 'ypos'
    
    regcis = reg_info[cond]['cis']
    paramcis = [(c1, c2) for c1, c2 in zip(cis['%s_lower' % param], cis['%s_upper' % param])]
    roi_list = cis.index.tolist()
    
    within=[]
    for roi, pci, rci in zip(roi_list, paramcis, regcis):
        if (rci[0] <= pci[0] <= rci[1]) or (rci[0] <= pci[1] <= rci[1]):
            within.append(roi)
        elif (pci[0] <= rci[0] <= pci[1]) or (pci[0] <= rci[1] <= pci[1]):
            within.append(roi)
    print("%i out of %i cells' bootstrapped param distNs lie within %i%% CI of regression fit" % (len(within), len(roi_list), int(alpha*100)))
    trudeviants = [r for r in roi_list if r not in within and r not in reg_info[cond]['deviants']]
    print("There are %i true deviants!" % len(trudeviants))
    
    
    fig, ax = pl.subplots()
    ax = plot_regr_and_cis(bdata, cis, posdf, cond=cond, ax=ax)
    
    deviant_fpos = posdf['%s_fov' % axname][trudeviants]
    deviant_rpos = posdf['%s_rf' % axname][trudeviants]
    ax.scatter(deviant_fpos, deviant_rpos, c='cornflowerblue')

    deviants[cond] = trudeviants

#%%

deviant_both = sorted(intersection(deviants['azimuth'], deviants['elevation']))
print deviant_both

