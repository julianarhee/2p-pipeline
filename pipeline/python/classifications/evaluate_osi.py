#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 18:22:01 2019

@author: julianarhee
"""

import os
import glob
import json
import sys

import numpy as np
import seaborn as sns
import pylab as pl
import pandas as pd

from scipy import stats as spstats

from pipeline.python.utils import natural_keys, label_figure
from pipeline.python.classifications import utils as util

#%%


#theta_r = np.array([np.deg2rad(t) for t in fparams['theta_pref'].values])
#theta_r.min()
#theta_r.max()
#np.rad2deg(spstats.circmean(theta_r))

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
    fitr = bs.double_gaussian( thetas, *popt)
    
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



#%%

def get_average_fit(roi, fitdata):
    
    n_intervals_interp = fitdata['results_by_iter'][roi][0]['n_intervals_interp']
    popt = get_average_params_over_iters(fitdata['results_by_iter'][roi])
    thetas = fitdata['results_by_iter'][roi][0]['x'][0::n_intervals_interp][0:-1]
    responses = fitdata['original_data'][roi]['responses']
    origr = responses.mean().values
    fitr = bs.double_gaussian( thetas, *popt)
    
    return origr, fitr

def get_gfit(fitdf_roi):
    '''
    Use metric from Liang et al (2018). 
    Note: Will return NaN of fit is crappy
    
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
    
    gfits = fitdf.groupby(['cell']).apply(get_gfit).dropna()
    
    return fitdf, gfits


    
def threshold_fitness_quality(gfits, goodness_thr=0.66, plot_hist=True, ax=None):    
    if plot_hist:
        if ax is None:
            fig, ax = pl.subplots() #.figure()
        ax.hist(gfits, alpha=0.5)
        ax.axvline(x=goodness_thr, linestyle=':', color='k')
        ax.set_xlim([0, 1])
        sns.despine(ax=ax, trim=True, offset=2)
        ax.set_xlabel('G-fit values')
        ax.set_ylabel('counts')

        
    good_fits = [int(r) for r in gfits.index.tolist() if gfits[r] > goodness_thr]
    
    if not plot_hist:
        return good_fits
    else:
        return good_fits, ax


def get_stimulus_configs(animalid, session, fov, run_name, rootdir='/n/coxfs01/2p-data'):
    # Get stimulus configs
    stiminfo_fpath = os.path.join(rootdir, animalid, session, fov, run_name, 'paradigm', 'stimulus_configs.json')
    with open(stiminfo_fpath, 'r') as f:
        stiminfo = json.load(f)
    sdf = pd.DataFrame(stiminfo).T
    
    return sdf
    
    
def compare_metrics_for_good_fits(fitdf, good_fits=[]):
    
    df = fitdf[fitdf['cell'].isin(good_fits)]
    df['cell'] = df['cell'].astype(str)
    
    g = sns.PairGrid(df, hue='cell', vars=[c for c in fitdf.columns.tolist() if c != 'cell'], palette='cubehelix')
    g.fig.patch.set_alpha(1)
    g = g.map_offdiag(pl.scatter, marker='o', s=5, alpha=0.7)
    g = g.map_diag(pl.hist, normed=True, alpha=0.5) #histtype="step",  
    
    g.set(ylim=(0, 1))
    g.set(xlim=(0, 1))
    g.set(aspect='equal')
    
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
        
        fitr = bs.double_gaussian(thetas_interp, *popt)
        
        if plot_polar:
            origr = np.append(origr, origr[0]) # wrap back around
            bs.plot_tuning_polar_roi(thetas, origr, curr_sems=None, response_type='dff',
                                      fig=fig, ax=ax, color='k')
    
            bs.plot_tuning_polar_roi(thetas_interp, fitr, curr_sems=None, response_type='dff',
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


#%%
def evaluate_bootstrapped_tuning(fitdf, fitparams, fitdata, goodness_thr=0.66,
                                 response_type='dff', data_identifier='METADATA'):

    # Create output dir for evaluation
    roi_fitdir = fitparams['directory']
    evaldir = os.path.join(roi_fitdir, 'evaluation')
    if not os.path.exists(evaldir):
        os.makedirs(evaldir)

    #% # Plot visualizations for each cell's bootstrap iters for fit
    rois_fit = fitdf['cell'].unique()
    #n_rois_fit = len(rois_fit)
    
    roi_evaldir = os.path.join(evaldir, 'fit_rois')
    if not os.path.exists(roi_evaldir):
        os.makedirs(roi_evaldir)
    
    for roi in rois_fit:
        fig = evaluate_fit_roi(roi, fitdf, fitparams, fitdata, response_type=response_type)
        label_figure(fig, data_identifier)
        pl.savefig(os.path.join(roi_evaldir, 'roi%05d.png' % int(roi+1)))
        pl.close()
        
    # Get residual sum of squares and compare ORIG and AVG FIT:
    fitdf, gfits = check_fit_quality(fitdf, fitdata)
    good_fits = threshold_fitness_quality(gfits, goodness_thr=goodness_thr, plot_hist=False)

    n_rois_pass = len(good_fits)
    n_rois_fit = len(fitdf['cell'].unique())
    print("%i out of %i fit cells pass goodness-thr %.2f" % (n_rois_pass, n_rois_fit, goodness_thr))

    # Plot pairwise comparisons of metrics:
    fig = compare_metrics_for_good_fits(gfits, good_fits=good_fits)
    label_figure(fig, data_identifier)
    pl.savefig(os.path.join(evaldir, 'pairplot-all-metrics_goodness-thr%.2f.png' % goodness_thr))
    pl.close()
    
    #% Plot tuning curves for all cells with good fits:
    fig = plot_tuning_for_good_fits(fitdata, fitparams, good_fits=good_fits, plot_polar=True)
    label_figure(fig, data_identifier)
    fig.suptitle("Cells with GoF > %.2f" % goodness_thr)
    
    figname = 'polar-tuning_goodness-of-fit%.2f_%irois' % (goodness_thr, len(good_fits))
    pl.savefig(os.path.join(evaldir, '%s.png' % figname))
    pl.close()
    
    return good_fits

#%%

#rootdir = '/n/coxfs01/2p-data'
#animalid = 'JC084' 
#session = '20190522' #'20190319'
#fov = 'FOV1_zoom2p0x' 
#run = 'combined_gratings_static'
#traceid = 'traces001' #'traces002'
#
#response_type = 'dff'
#make_plots = True
#n_bootstrap_iters = 100
#n_intervals_interp = 3
#
#responsive_test = 'ROC'
#responsive_thr = 0.05
#
#create_new = False
#goodness_thr = 0.66
#

#%%

def main(options):
    from pipeline.python.classifications import bootstrap_fit_tuning_curves as bs 

    opts = bs.extract_options(options)
        
    # Get gratings dataset and responsive cells
    run_name = bs.get_gratings_run(opts.animalid, opts.session, opts.fov, traceid=opts.traceid, rootdir=opts.rootdir) 
    roi_list = util.get_responsive_cells(opts.animalid, opts.session, opts.fov, run=run_name, traceid=opts.traceid,
                                         responsive_test=opts.responsive_test, responsive_thr=opts.responsive_thr,
                                         rootdir=opts.rootdir)
    
    #% Select tuning-fit to evaluate
    roi_fitdir = glob.glob(os.path.join(opts.rootdir, opts.animalid, opts.session, opts.fov, run_name,
                                        'traces', '%s*' % opts.traceid, 'tuning', 
                                        'bootstrap*%s-thr%.2f' % (opts.responsive_test, opts.responsive_thr)))[0]
    responsive_desc = os.path.split(roi_fitdir)[-1]
    data_identifier = '|'.join([opts.animalid, opts.session, opts.fov, opts.traceid, responsive_desc])
    
    #% Load fit results, examine each cell's iterations:  
    fitdf, fitparams, fitdata = bs.get_tuning(opts.animalid, opts.session, opts.fov, run_name, 
                                              roi_list=roi_list, create_new=opts.create_new,
                                              n_bootstrap_iters=int(opts.n_bootstrap_iters), 
                                              n_intervals_interp=int(opts.n_intervals_interp),
                                              response_type=opts.response_type, 
                                              make_plots=opts.make_plots, 
                                              roi_fitdir=roi_fitdir, return_iters=True)

    fitdf['cell'] = [int(i) for i in fitdf['cell'].values]
    
    good_fits = evaluate_bootstrapped_tuning(fitdf, fitparams, fitdata, goodness_thr=float(opts.goodness_thr),
                                             data_identifier=data_identifier, response_type=opts.response_type)
    
    
    print("DONE! Evaulated fits from: %s" % responsive_desc)
    print("--- Found %i cells with good fits (thr=%.2f)" % (len(good_fits), opts.goodness_thr))
    
if __name__ == '__main__':
    main(sys.argv[1:])
    