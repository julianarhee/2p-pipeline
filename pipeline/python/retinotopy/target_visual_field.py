#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 14:50:56 2019

@author: julianarhee
"""

import os
import glob
import json
import h5py
import optparse
import sys
import traceback

import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import pylab as pl
import seaborn as sns
import numpy as np
import scipy as sp
import statsmodels as sm
import cPickle as pkl
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pipeline.python.utils import natural_keys, label_figure
from pipeline.python.retinotopy import visualize_rois as visroi
from pipeline.python.retinotopy import utils as rutils
from scipy.signal import argrelextrema


#%%
def convert_values(oldval, newmin=None, newmax=None, oldmax=None, oldmin=None):
    oldrange = (oldmax - oldmin)
    newrange = (newmax - newmin)
    newval = (((oldval - oldmin) * newrange) / oldrange) + newmin
    return newval

def trials_to_dataframes(processed_fpaths, conditions_fpath):
    
    # Get condition / trial info:
    with open(conditions_fpath, 'r') as f:
        conds = json.load(f)
    cond_list = list(set([cond_dict['stimuli']['stimulus'] for trial_num, cond_dict in conds.items()]))
    trials_by_cond = dict((cond, [int(k) for k, v in conds.items() if v['stimuli']['stimulus']==cond]) for cond in cond_list)

    excluded_tifs = [] 
    for cond, tif_list in trials_by_cond.items():
        for tifnum in tif_list:
            processed_tif = [f for f in processed_fpaths if 'File%03d' % tifnum in f]
            if len(processed_tif) == 0:
                print "No analysis found for file: %s" % tifnum
                excluded_tifs.append(tifnum)
        trials_by_cond[cond] = [t for t in tif_list if t not in excluded_tifs]
    print "TRIALS BY COND:"
    print trials_by_cond 
    trial_list = [int(t) for t in conds.keys() if int(t) not in excluded_tifs]
    print "Trials:", trial_list

    fits = []
    phases = []
    mags = []
    for trial_num, trial_fpath in zip(sorted(trial_list), sorted(processed_fpaths, key=natural_keys)):
        
        print("%i: %s" % (trial_num, os.path.split(trial_fpath)[-1]))
        df = h5py.File(trial_fpath, 'r')
        fits.append(pd.Series(data=df['var_exp_array'][:], name=trial_num))
        phases.append(pd.Series(data=df['phase_array'][:], name=trial_num))
        mags.append(pd.Series(data=df['mag_ratio_array'][:], name=trial_num))
        df.close()
        
    fit = pd.concat(fits, axis=1)
    magratio = pd.concat(mags, axis=1)
    phase = pd.concat(phases, axis=1)
    
    return fit, magratio, phase, trials_by_cond


def plot_signal_fits_by_roi(fit, magratio, threshold=0.2, 
                            data_identifier='', fov='FOV', retinoid='retinoID', output_dir='/tmp'):
    
    fig = pl.figure(figsize=(20,10))
    nr = 2
    nc = 3
    
    #%
    
    rnums = np.arange(1, fit.shape[0]+1)
    mean_fits = fit.mean(axis=1)
    mean_fits_std = fit.std(axis=1)
    
    max_fits = fit.max(axis=1)
    
    #%
    ax1 = pl.subplot2grid((nr,nc), (0, 0), colspan=3)
    ax1.plot(rnums, mean_fits, '.')
    ax1.set_xticks(rnums)
    if len(rnums) > 50:
        rnum_labels = [r if (r==1 or r%10==0) else '' for r in rnums]
    else:
        rnum_labels = rnums.copy()
    ax1.set_xticklabels(rnum_labels)
    ax1.errorbar(x=rnums, y=mean_fits, yerr=mean_fits_std, color='k', fmt='.', elinewidth=1, label='var exp')
    ax1.plot(rnums, max_fits, '.', color='b', label='max fit')
    
    ax1.set_ylabel('mean fit (+/- std)')
    ax1.set_xlabel('roi')
    ax1.axhline(y=threshold, color='r', linewidth=0.5, label='threshold')
    
    ax1.legend()
    
    #%
    mean_magratios = magratio.mean(axis=1)
    max_magratios = magratio.max(axis=1)
    
    ax2 = pl.subplot2grid((nr, nc), (1, 0), colspan=1)
    ax2.scatter(mean_fits, mean_magratios)
    ax2.set_xlabel('mean fit')
    ax2.set_ylabel('mean mag ratio')
    #ax2.set_ylim([0, mean_magratios.max()])
    
    ax3 = pl.subplot2grid((nr, nc), (1, 1), colspan=1)
    ax3.scatter(max_fits, mean_magratios)
    ax3.set_xlabel('max fit')
    #ax3.set_ylim([0, mean_magratios.max()])
    
    ax4 = pl.subplot2grid((nr, nc), (1, 2), colspan=1)
    ax4.scatter(max_fits, max_magratios)
    ax4.set_xlabel('max fit')
    ax4.set_ylabel('max mag ratio')
    
    #% Save figure
    
    label_figure(fig, data_identifier)
    figname = 'var_exp_by_roi_%s_%s.png' % (fov, retinoid)
    
    pl.savefig(os.path.join(output_dir, figname))

    return fig


def label_rois(ax, xlocs, ylocs, roi_list, supp_roi_list=[]):
    for ri, rindex in enumerate(roi_list):
        if rindex in supp_roi_list:
            ax.text(xlocs[ri], ylocs[ri], '%s' % str(rindex+1), color='r') #fontweight='bold', alpha=0.5, style='italic')
        else:
            ax.text(xlocs[ri], ylocs[ri], '%s' % str(rindex+1)) #, alpha=0.5)
            
    return
    
def visualize_fits_by_condition(fit, magratio, corrected_phase, trials_by_cond, screen, #use_circ=False,
                                labeled_rois=[], use_linear=True, fit_thresh_az=0.2, fit_thresh_el=0.2,
                                data_identifier='', fov='FOV', retinoid='retinoID', output_dir='/tmp'):
    
    fig = pl.figure(figsize=(20,20))
    nr = 3
    nc = 3
    
    mean_fits = fit.mean(axis=1)
    mean_magratios = magratio.mean(axis=1)
    
    
    mean_fits_az = fit[trials_by_cond['right']].mean(axis=1)
    mean_fits_el = fit[trials_by_cond['top']].mean(axis=1)
    
    std_fits_az = fit[trials_by_cond['right']].std(axis=1)
    std_fits_el = fit[trials_by_cond['top']].std(axis=1)
    
    #%
    ax1 = pl.subplot2grid((nr, nc), (0, 0), colspan=2, rowspan=2)
    ax1.scatter(mean_fits_az, mean_fits_el, alpha=0.5, s=100)
    ax1.set_xlabel('mean fits (az)')
    ax1.set_ylabel('mean fits (el)')
    ax1.errorbar(mean_fits_az, mean_fits_el, yerr=std_fits_el, xerr=std_fits_az, fmt='.', alpha=0.5)
    
    #% Label "good" rois:
    
    #fit_thresh_el = 0.20
    #fit_thresh_az = 0.20
    good_fits = [ri for ri in mean_fits_az.index.tolist() if mean_fits_el.loc[ri] >= fit_thresh_el\
                                                             and mean_fits_az.loc[ri] >= fit_thresh_az]
    good_fits.extend(labeled_rois)
    
    print good_fits
    
    label_rois(ax1, [mean_fits_az.iloc[ri] for ri in good_fits],
                    [mean_fits_el.iloc[ri] for ri in good_fits], 
                    good_fits, supp_roi_list=list(labeled_rois))
    
    ax1.text(mean_fits_az.max(), mean_fits_el.max(), 'az thr: %.2f\nel thr: %.2f' % (fit_thresh_az, fit_thresh_el), fontsize=16)
    
    #%
    phase_std_az = corrected_phase[trials_by_cond['right']].std(axis=1)
    fit_std_az = fit[trials_by_cond['right']].std(axis=1)
    
    phase_std_el = corrected_phase[trials_by_cond['top']].std(axis=1)
    fit_std_el = fit[trials_by_cond['top']].std(axis=1)
    
    ax4 = pl.subplot2grid((nr, nc), (0, 2), colspan=1)
    ax4.scatter(fit_std_az, phase_std_az, alpha=0.5, s=50)
    ax4.set_ylabel('std PHASE')
    ax4.set_xlabel('std FIT')
    ax4.set_title('azimuth')
    
    
    ax5 = pl.subplot2grid((nr, nc), (1, 2), colspan=1)
    ax5.scatter(fit_std_el, phase_std_el, alpha=0.5, s=50)
    ax5.set_ylabel('std PHASE')
    ax5.set_xlabel('std FIT')
    ax5.set_title('elevation')

    
    label_rois(ax4, [fit_std_az.iloc[ri] for ri in good_fits],
                    [phase_std_az.iloc[ri] for ri in good_fits], 
                    good_fits, supp_roi_list=list(labeled_rois))

    label_rois(ax5, [fit_std_el.iloc[ri] for ri in good_fits],
                    [phase_std_el.iloc[ri] for ri in good_fits], 
                    good_fits, supp_roi_list=list(labeled_rois))
    
    #%
    #use_linear = True
    
#    if use_circ:
    mean_phase_az = sp.stats.circmean(corrected_phase[trials_by_cond['right']], axis=1)
    mean_phase_el = sp.stats.circmean(corrected_phase[trials_by_cond['top']], axis=1)
#    else:
#        mean_phase_az = corrected_phase[trials_by_cond['right']].mean(axis=1)
#        mean_phase_el = corrected_phase[trials_by_cond['top']].mean(axis=1)

    #print "MEAN PHASE", mean_phase_el.head()

    
    
    # Convert phase to linear coords:
    
    az_extent = screen['bb_right'] - screen['bb_left']
    el_extent = screen['bb_upper'] - screen['bb_lower']
    
    print("AZ extent: %.2f" % az_extent)
    print("EL extend: %.2f" % el_extent)
    
    ax2 = pl.subplot2grid((nr, nc), (2, 0), colspan=1)
    ax3 = pl.subplot2grid((nr, nc), (2, 1), colspan=1)
    
    if use_linear:
        screen_left = -1*screen['azimuth']/2.
        screen_right = screen['azimuth']/2. #screen['azimuth']/2.
        screen_lower = -1*screen['elevation']/2.
        screen_upper = screen['elevation']/2. #screen['elevation']/2.
        
        linX = convert_values(mean_phase_az, newmin=screen_right, newmax=screen_left,
                             oldmax=2*np.pi, oldmin=0)  # If cond is 'right':  positive values = 0, negative values = 2pi
        linY = convert_values(mean_phase_el, newmin=screen_upper, newmax=screen_lower, #screen_upper,
                             oldmax=2*np.pi, oldmin=0)  # If cond is 'right':  positive values = 0, negative values = 2pi
        
        # Draw azimuth value as a function of mean fit (color code by standard cmap, too)
        ax2.scatter(mean_fits, linX, c=linX, cmap='nipy_spectral_r', vmin=screen_left, vmax=screen_right)
        ax2.set_ylim([screen_left, screen_right])
        ax2.invert_yaxis()
        ax2.set_xlabel('mean fit')
        ax2.set_ylabel('mean xpos')     
        
        # Draw BB from epi:
        ax2.axhline(y=screen['bb_right'], color='k', linestyle='--', linewidth=1)
        ax2.axhline(y=screen['bb_left'], color='k', linestyle='--', linewidth=1)
        
        # Draw elevation value as a function of mean fit (color code by standard cmap, too)
        ax3.scatter(mean_fits, linY, c=linY, cmap='nipy_spectral_r', vmin=screen_lower, vmax=screen_upper)
        ax3.set_ylim([screen_lower, screen_upper])
        ax3.invert_yaxis()
        ax3.set_xlabel('mean fit')
        ax3.set_ylabel('mean ypos')  
        
        # Draw BB from epi:
        ax3.axhline(y=screen['bb_upper'], color='k', linestyle='--', linewidth=1)
        ax3.axhline(y=screen['bb_lower'], color='k', linestyle='--', linewidth=1)
        
        # Label good cells:
        label_rois(ax2, [mean_fits.iloc[ri] for ri in good_fits],
                        [linX[ri] for ri in good_fits], #[linX.iloc[ri] for ri in good_fits], 
                        good_fits, supp_roi_list=list(labeled_rois))
    
        label_rois(ax3, [mean_fits.iloc[ri] for ri in good_fits],
                        [linY[ri] for ri in good_fits], #[linY.iloc[ri] for ri in good_fits], 
                        good_fits, supp_roi_list=list(labeled_rois))
        
    #%
    
    else:
        #%
        ax2.scatter(mean_fits, mean_phase_az, c=mean_phase_az, cmap='nipy_spectral', vmin=0, vmax=np.pi*2)
        ax2.set_ylim([0, np.pi*2])
        ax2.set_xlabel('mean fit')
        ax2.set_ylabel('average phase (AZ)')
        
        ax3.scatter(mean_fits, mean_phase_el, c=mean_phase_el, cmap='nipy_spectral', vmin=0, vmax=np.pi*2)
        ax3.set_ylim([0, np.pi*2])
        ax3.set_xlabel('mean fit')
        ax3.set_ylabel('average phase (EL)')
        
    #%
    
    ax0 = pl.subplot2grid((nr, nc), (2, 2), colspan=1)
    ax0.scatter(mean_fits, mean_magratios)
    ax0.set_xlabel('mean fit')
    ax0.set_ylabel('mean mag ratio')
    
    label_rois(ax0, [mean_fits.iloc[ri] for ri in good_fits],
                    [mean_magratios.iloc[ri] for ri in good_fits], 
                    good_fits, supp_roi_list=list(labeled_rois))
        
        
    
    
    label_figure(fig, data_identifier)
    if use_linear:
        phase_space = 'linspace'
    else:
        phase_space = 'phase'
    figname = 'compare_fit_and_pos_by_condition_%s_%s_%s.png' % (fov, retinoid, phase_space)
    
    pl.savefig(os.path.join(output_dir, figname))
    
    return fig, good_fits

#%%

def get_center_of_mass(fit, linX, linY, screen, marker_scale=200):
    
    mean_fits = fit.mean(axis=1)
    
    # Convert phase to linear coords:
    screen_left = -1*screen['azimuth']/2.
    screen_right = screen['azimuth']/2. #screen['azimuth']/2.
    screen_lower = -1*screen['elevation']/2.
    screen_upper = screen['elevation']/2. #screen['elevation']/2.

    fig = pl.figure(figsize=(10,6))
    ax = pl.subplot2grid((1, 2), (0, 0), colspan=2, fig=fig)
    
    # Draw azimuth value as a function of mean fit (color code by standard cmap, too)
    ax.scatter(linX, linY, s=mean_fits*marker_scale, alpha=0.5) # cmap='nipy_spectral', vmin=screen_left, vmax=screen_right)
    ax.set_xlim([screen_left, screen_right])
    ax.set_ylim([screen_lower, screen_upper])
    ax.set_xlabel('xpos (deg)')
    ax.set_ylabel('ypos (deg)')     
    
    
    
    # Draw BB from epi:
    ax.axvline(x=screen['bb_right'], color='k', linestyle='--', linewidth=1)
    ax.axvline(x=screen['bb_left'], color='k', linestyle='--', linewidth=1)
    ax.axhline(y=screen['bb_upper'], color='k', linestyle='--', linewidth=1)
    ax.axhline(y=screen['bb_lower'], color='k', linestyle='--', linewidth=1)

    
    cgx = np.sum(linX * mean_fits) / np.sum(mean_fits)
    cgy = np.sum(linY * mean_fits) / np.sum(mean_fits)
    
#    cgx = np.sum(linX[mean_fits>=0.3] * mean_fits[mean_fits>=0.3]) / np.sum(mean_fits[mean_fits>=0.3])
#    cgy = np.sum(linY[mean_fits>=0.3] * mean_fits[mean_fits>=0.3]) / np.sum(mean_fits[mean_fits>=0.3])
    print('Center of mass on x: %f' % cgx)
    print('Center of mass on y: %f' % cgy)
    ax.scatter(cgx, cgy, color='k', marker='+', s=1e4);
    ax.text(cgx+1, cgy+1, 'x: %.2f\ny: %.2f' % (cgx, cgy))

    return fig


def find_local_min_max(xvals, yvals):
    
    lmins = argrelextrema(yvals, np.less)[0]
    lmaxs = argrelextrema(yvals, np.greater)[0]
    print "local minima:", lmins
    print "local maxima:", lmaxs
        
    lmax_value = np.max([yvals[mx] for mx in lmaxs])
    lmax = [mx for mx in lmaxs if yvals[mx] == lmax_value][0]
    lmaxs = [mx for mx in lmaxs if yvals[mx] >= lmax_value*0.5]
    
    
        
    if len(lmins) == 1:
        lmin = lmins[0]
        if xvals[lmins] < xvals[lmax]: 
            # min is to the left (less than), so look for max on right
            lmin2 = lmax + np.where(np.abs(yvals[lmax:]-yvals[lmin]) == np.min( np.abs(yvals[lmax:]-yvals[lmin]) ))[0]            
        else:
            lmin2 = lmax - np.where(np.abs(yvals[:lmax]-yvals[lmin]) == np.min( np.abs(yvals[:lmax]-yvals[lmin]) ))[0] 
    elif len(lmins) == 0: # no local maxima, just take values at half-max:
        halfmax = lmax_value/2. 
        lmin = lmax - np.where( np.abs(yvals[:lmax] - halfmax) == np.min( np.abs(yvals[:lmax] - halfmax) ) )[0] 
        lmin2 = lmax + np.where( np.abs(yvals[lmax:] - halfmax) == np.min( np.abs(yvals[lmax:] - halfmax) ) )[0] 

    else:
        lmin = sorted(lmins)[0]
        lmin2 = sorted(lmins)[-1]
        
    return lmax, lmin, lmin2, lmaxs, lmins
    
 
def plot_kde_min_max(xvals, yvals, maxval=0, minval1=0, minval2=0, title='', ax=None):
    
    if ax is None:
        fig, ax = pl.subplots()
        
    ax.plot(xvals, yvals, 'k')
    ax.plot(xvals[maxval], yvals[maxval], 'r*') # plot local max
    ax.plot(xvals[minval1], yvals[minval1], 'b*') # plot local minima
    ax.plot(xvals[minval2], yvals[minval2], 'b*') # plot local minima
    ax.set_title('%s' % title)
    ax.set_ylabel('weighted kde')
    
    return



def plot_kde_maxima(kde_results, magratio, linX, linY, screen, use_peak=True, \
                    draw_bb=True, marker_scale=200, exclude_bad=False, min_thr=0.01):
    
    mean_magratios = magratio.mean(axis=1)
    
    # Convert phase to linear coords:
    screen_left = -1*screen['azimuth']/2.
    screen_right = screen['azimuth']/2. #screen['azimuth']/2.
    screen_lower = -1*screen['elevation']/2.
    screen_upper = screen['elevation']/2. #screen['elevation']/2.

    fig = pl.figure(figsize=(10,6))
    ax = pl.subplot2grid((1, 2), (0, 0), colspan=2, fig=fig)
    
    if exclude_bad:
        bad_cells = mean_magratios[mean_magratios < min_thr].index.tolist()
        kept_cells = [i for i in mean_magratios.index.tolist() if i not in bad_cells]
        linX = linX[kept_cells]
        linY = linY[kept_cells]
        mean_magratios = mean_magratios.iloc[kept_cells]
    
    # Draw azimuth value as a function of mean fit (color code by standard cmap, too)
    im = ax.scatter(linX, linY, s=mean_magratios*marker_scale, c=mean_magratios, cmap='inferno', alpha=0.5) # cmap='nipy_spectral', vmin=screen_left, vmax=screen_right)
    ax.set_xlim([screen_left, screen_right])
    ax.set_ylim([screen_lower, screen_upper])
    ax.set_xlabel('xpos (deg)')
    ax.set_ylabel('ypos (deg)')     

    # Add color bar:
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05) 
    alpha_min = mean_magratios.min()
    alpha_max = mean_magratios.max() 
    magnorm = mpl.colors.Normalize(vmin=alpha_min, vmax=alpha_max)
    magcmap=mpl.cm.inferno
    pl.colorbar(im, cax=cax, cmap=magcmap, norm=magnorm)
    cax.yaxis.set_ticks_position('right')


    
    if draw_bb:
        ax.axvline(x=kde_results['az_bounds'][0], color='k', linestyle='--', linewidth=0.5)
        ax.axvline(x=kde_results['az_bounds'][1], color='k', linestyle='--', linewidth=0.5)
        ax.axhline(y=kde_results['el_bounds'][0], color='k', linestyle='--', linewidth=0.5)
        ax.axhline(y=kde_results['el_bounds'][1], color='k', linestyle='--', linewidth=0.5)

    if use_peak:
        cgx = kde_results['az_max']
        cgy = kde_results['el_max']
        centroid_type = 'peak'
    else:
        cgx = kde_results['center_x'] #np.sum(linX * mean_fits) / np.sum(mean_fits)
        cgy = kde_results['center_y'] #np.sum(linY * mean_fits) / np.sum(mean_fits)
        centroid_type = 'center'
        
    print('%s x: %f' % (centroid_type, cgx))
    print('%s y: %f' % (centroid_type, cgy))
    ax.scatter(cgx, cgy, color='k', marker='+', s=1e4);
    ax.text(cgx+3, cgy+3, '%s x, y:\n(%.2f, %.2f)' % (centroid_type, cgx, cgy), color='k', fontweight='bold')

    # Also plot alternative maxima if they exist:
    for az in kde_results['az_maxima']:
        for el in kde_results['el_maxima']:
            if az == kde_results['az_max'] and el == kde_results['el_max']:
                continue
            ax.scatter(az, el, color='b', marker='+', s=1e3);
            ax.text(az+3, el+3, 'pk x, y:\n(%.2f, %.2f)' % (az, el), color='b', fontweight='bold')


    return fig



#%%


def get_absolute_centers(phase, magratio, trials_by_cond, stim_positions, absolute=True, \
                         mag_thr=0.02, delay_thr=np.pi/2, equal_travel_lengths=True):
    
    # Convert phase to linear coords:
    # screen_left = -1*screen['azimuth']/2.
    # screen_right = screen['azimuth']/2. #screen['azimuth']/2.
    # if equal_travel_lengths:
    #     screen_lower = screen_left
    #     screen_upper = screen_right
    # else:
    #     screen_lower = -1*screen['elevation']/2.
    #     screen_upper = screen['elevation']/2. #screen['elevation']/2.
    
    screen_left = stim_positions['left'].iloc[0,:].mean()
    #screen_right = stim_positions['right'].iloc[0,:].mean()
    screen_right = stim_positions['left'].iloc[-1,:].mean()
    #screen_upper = stim_positions['top'].iloc[0,:].mean()
    screen_upper = stim_positions['bottom'].iloc[-1,:].mean()
    screen_lower = stim_positions['bottom'].iloc[0,:].mean()


    # Find strongly responding cells to calcualte delay
    mean_mags = magratio.mean(axis=1)
    strong_cells = mean_mags[mean_mags >= mag_thr].index.tolist()
    print("ROIs with best mag-ratio (n=%i, thr=%.2f):" % (len(strong_cells), mag_thr), strong_cells)
        
#        mean_fits = fit.mean(axis=1)
#        best_fits = mean_fits[mean_fits >= fit_thr].index.tolist()
#        print("ROIs with best fit (n=%i, thr=%.2f):" % (len(best_fits), fit_thr), best_fits)

    if absolute:
        use_relative = False
        mean_phase_left = sp.stats.circmean(phase[trials_by_cond['left']], axis=1, low=-np.pi, high=np.pi)
        mean_phase_right = sp.stats.circmean(phase[trials_by_cond['right']], axis=1, low=-np.pi, high=np.pi)
        mean_phase_bottom = sp.stats.circmean(phase[trials_by_cond['bottom']], axis=1, low=-np.pi, high=np.pi)
        mean_phase_top = sp.stats.circmean(phase[trials_by_cond['top']], axis=1, low=-np.pi, high=np.pi)

        # Calculate delay map from non-corrected 
        delay_az = (mean_phase_left[strong_cells] + mean_phase_right[strong_cells]) / 2.
        avg_delay_az = delay_az.mean()
        std_delay_az = delay_az.std()
        print "[AZ] Average delay (std): %.2f (%.2f)" % (avg_delay_az, std_delay_az)
        
        delay_el = (mean_phase_bottom[strong_cells] + mean_phase_top[strong_cells]) / 2.
        avg_delay_el = delay_el.mean()
        std_delay_el = delay_el.std()
        print "[EL] Average delay (std): %.2f (%.2f)" % (avg_delay_el, std_delay_el)
    
        #delay_thr = np.pi/2 #1.0
        if std_delay_az > delay_thr or std_delay_el > delay_thr:
            print "*** WARNING:  Bad delay in AZ (%.2f) or EL (%.2f) condition" % (avg_delay_az, avg_delay_el)
            use_relative = True
    else:
        use_relative = True
    
    if not use_relative:
        # Calculate absoluate maps if delay is reasonable:
        mean_phase_az = (mean_phase_left - mean_phase_right) / 2.0
        mean_phase_el = (mean_phase_bottom - mean_phase_top) / 2.0
        
        # Convert to linear coords:
        # When converted to absolute (left-right), neg screen values = neg pi, pos screen values = pos pi
        linX = convert_values(mean_phase_az, 
                              newmin=screen_left, newmax=screen_right,
                              oldmin=-np.pi, oldmax=np.pi) 
        linY = convert_values(mean_phase_el, 
                              newmin=screen_lower, newmax=screen_upper, 
                              oldmin=-np.pi, oldmax=np.pi)  
        
    else:
        corrected_phase_right = correct_phase_wrap(phase[trials_by_cond['right']])
        corrected_phase_top = correct_phase_wrap(phase[trials_by_cond['top']])
                                             
        mean_phase_az = sp.stats.circmean(corrected_phase_right, axis=1, low=0, high=2*np.pi)
        mean_phase_el = sp.stats.circmean(corrected_phase_top, axis=1, low=0, high=2*np.pi)

        # Convert to linear coords:
        # If cond is 'right':  positive screen values = 0, negative screen values = 2pi
        # If cond is 'top':  positive values = 0, negative values = 2pi
        linX = convert_values(mean_phase_az, 
                              newmin=screen_right, newmax=screen_left,
                              oldmin=0, oldmax=2*np.pi)  
        linY = convert_values(mean_phase_el, 
                              newmin=screen_upper, newmax=screen_lower, #screen_upper,
                              oldmin=0, oldmax=2*np.pi)  
        
    absolute_coords = {'mean_phase_az': mean_phase_az,
                       'mean_phase_el': mean_phase_el,
                       'linX': linX,
                       'linY': linY,
                       'delay_az': delay_az,
                       'delay_el': delay_el,
                       'delay_thr': delay_thr,
                       'strong_cells': strong_cells,
                       'mag_thr': mag_thr,
                       'used_relative': use_relative,
                       'screen_bb': [screen_lower, screen_left, screen_upper, screen_right]}
                       
    return absolute_coords, strong_cells
    
    
def correct_phase_wrap(phase):
        
    corrected_phase = phase.copy()
    
    corrected_phase[phase<0] =- phase[phase<0]
    corrected_phase[phase>0] = (2*np.pi) - phase[phase>0]
    
    return corrected_phase


#%%
#def get_interp_positions(condname, mwinfo, stiminfo, trials_by_cond):
#    mw_fps = 1./np.diff(np.array(mwinfo[str(trials_by_cond[condname][0])]['stiminfo']['tstamps'])/1E6).mean()
#    si_fps = stiminfo[condname]['frame_rate']
#    print "[%s]: Downsampling MW positions (sampled at %.2fHz) to SI frame rate (%.2fHz)" % (condname, mw_fps, si_fps)
#
#    si_cyc_ixs = stiminfo[condname]['cycle_start_ixs']
#    si_tstamps = runinfo['frame_tstamps_sec']
#
#
#    #fig, axes = pl.subplots(1, len(trials_by_cond[condname]))
#
#    stim_pos_list = []
#    stim_tstamp_list = []
#
#    for ti, trial in enumerate(trials_by_cond[condname]):
#        #ax = axes[ti]
#
#        pos_list = []
#        tstamp_list = []
#        mw_cyc_ixs = mwinfo[str(trial)]['stiminfo']['start_indices']
#        for cix in np.arange(0, len(mw_cyc_ixs)):
#            if cix==len(mw_cyc_ixs)-1:
#                mw_ts = [t/1E6 for t in mwinfo[str(trial)]['stiminfo']['tstamps'][mw_cyc_ixs[cix]:]]
#                xs = mwinfo[str(trial)]['stiminfo']['values'][mw_cyc_ixs[cix]:]
#                si_ts = si_tstamps[si_cyc_ixs[cix]:]
#            else:
#                mw_ts = np.array([t/1E6 for t in mwinfo[str(trial)]['stiminfo']['tstamps'][mw_cyc_ixs[cix]:mw_cyc_ixs[cix+1]]])
#                xs = np.array(mwinfo[str(trial)]['stiminfo']['values'][mw_cyc_ixs[cix]:mw_cyc_ixs[cix+1]])
#                si_ts = si_tstamps[si_cyc_ixs[cix]:si_cyc_ixs[cix+1]]
#
#            recentered_mw_ts = [t-mw_ts[0] for t in mw_ts]
#            recentered_si_ts = [t-si_ts[0] for t in si_ts]
#
#            # Since MW tstamps are linear, SI tstamps linear, interpolate position values down to SI's lower framerate:
#            interpos = sp.interpolate.interp1d(recentered_mw_ts, xs, fill_value='extrapolate')
#            resampled_xs = interpos(recentered_si_ts)
#
#            pos_list.append(pd.Series(resampled_xs, name=trial))
#            tstamp_list.append(pd.Series(recentered_si_ts, name=trial))
#
#            #ax.plot(recentered_mw_ts, xs, 'ro', alpha=0.5, markersize=2)
#            #ax.plot(recentered_si_ts, resampled_xs, 'bx', alpha=0.5, markersize=2)
#
#        pos_vals = pd.concat(pos_list, axis=0).reset_index(drop=True) 
#        tstamp_vals = pd.concat(tstamp_list, axis=0).reset_index(drop=True)
#
#        stim_pos_list.append(pos_vals)
#        stim_tstamp_list.append(tstamp_vals)
#
#    stim_positions = pd.concat(stim_pos_list, axis=1)
#    stim_tstamps = pd.concat(stim_tstamp_list, axis=1)
#
#
#    return stim_positions, stim_tstamps
#
#def get_retino_stimulus_info(mwinfo, runinfo):
#    
#    stiminfo = dict((cond, dict()) for cond in conditions)
#    for curr_cond in conditions:
#        # get some info from paradigm and run file
#        stimfreq = np.unique([v['stimuli']['scale'] for k,v in mwinfo.items() if v['stimuli']['stimulus']==curr_cond])[0]
#        stimperiod = 1./stimfreq # sec per cycle
#        
#        n_frames = runinfo['nvolumes']
#        fr = runinfo['frame_rate']
#        
#        n_cycles = int(round((n_frames/fr) / stimperiod))
#        print n_cycles
#
#        n_frames_per_cycle = int(np.floor(stimperiod * fr))
#        cycle_starts = np.round(np.arange(0, n_frames_per_cycle * n_cycles, n_frames_per_cycle)).astype('int')
#
#        stiminfo[curr_cond] = {'stimfreq': stimfreq,
#                               'frame_rate': fr,
#                               'n_reps': len(trials_by_cond[curr_cond]),
#                               'nframes': n_frames,
#                               'n_cycles': n_cycles,
#                               'n_frames_per_cycle': n_frames_per_cycle,
#                               'cycle_start_ixs': cycle_starts
#                              }
#
#    return stiminfo
#

#%%

#rootdir = '/n/coxfs01/2p-data'
#animalid = 'JC022'
#session = '20181005'
#fov = 'FOV1_zoom1p7x'
#run = 'retino_run1'
#retinoid = 'analysis002'

def extract_options(options):
    
    parser = optparse.OptionParser()

    parser.add_option('-D', '--root', action='store', dest='rootdir', default='/n/coxfs01/2p-data', help='data root dir (root project dir containing all animalids) [default: /nas/volume1/2photon/data, /n/coxfs01/2pdata if --slurm]')
    parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')

    # Set specific session/run for current animal:
    parser.add_option('-S', '--session', action='store', dest='session', default='', \
                      help='session dir (format: YYYMMDD_ANIMALID')
    parser.add_option('-A', '--acq', action='store', dest='acquisition', default='FOV1', \
                      help="acquisition folder (ex: 'FOV1_zoom3x') [default: FOV1]")
    parser.add_option('-R', '--run', action='store', dest='run', default='retino_run1', \
                      help="name of run dir containing tiffs to be processed (ex: gratings_phasemod_run1)")
    parser.add_option('-r', '--retinoid', action='store', dest='retinoid', default='analysis001', \
                      help="name of retino ID (roi analysis) [default: analysis001]")
    
    parser.add_option('--angular', action='store_false', dest='use_linear', default=True, \
                      help="Plot az/el coordinates in angular spce [default: plots linear coords]")
    parser.add_option('-e', '--thr-el', action='store', dest='fit_thresh_el', default=0.2, \
                      help="fit threshold for elevation [default: 0.2]")
    parser.add_option('-a', '--thr-az', action='store', dest='fit_thresh_az', default=0.2, \
                      help="fit threshold for azimuth [default: 0.2]")
    parser.add_option('-t', '--thr', action='store', dest='threshold', default=0.2, \
                      help="fit threshold for all conds [default: 0.2]")
#    parser.add_option('--peak', action='store_true', dest='use_peak', default=False, \
#                      help='Flag to use PEAK instead of centroid of found CoMs.')
#    parser.add_option('--circ', action='store_true', dest='use_circ', default=False, \
#                      help='Flag to average by circular.')
#    
    (options, args) = parser.parse_args(options)

    return options

#%%

#options = ['-i', 'JC047', '-S', '20190215', '-A', 'FOV1']
#options = ['-i', 'JC070', '-S', '20190314', '-A', 'FOV1', '-t', 'analysis002']
#options = ['-i', 'JC070', '-S', '20190314', '-A', 'FOV1', '-R', 'retino_run1', '-t', 'analysis003']

#options = ['-i', 'JC070', '-S', '20190315', '-A', 'FOV1', '-R', 'retino_run2', '-t', 'analysis002']
#options = ['-i', 'JC070', '-S', '20190315', '-A', 'FOV2', '-R', 'retino_run1', '-t', 'analysis002']

options = ['-i', 'JC076', '-S', '20190408', '-A', 'FOV1', '-R', 'retino_run1', '-r', 'analysis002']


#%%

def main(options):
#%%
    opts = extract_options(options)
    
    rootdir = opts.rootdir
    animalid = opts.animalid
    session = opts.session
    fov = opts.acquisition
    run = opts.run
    retinoid = opts.retinoid
    use_linear = opts.use_linear
    fit_thresh_az = float(opts.fit_thresh_az)
    fit_thresh_el = float(opts.fit_thresh_el) #0.2
    #use_circ = opts.use_circ
    
    
    #%%
    
    run_dir = glob.glob(os.path.join(rootdir, animalid, session, '%s*' % fov, run))[0]
    fov = os.path.split(os.path.split(run_dir)[0])[-1]
    print("FOV: %s, run: %s" % (fov, run))
    retinoids_fpath = glob.glob(os.path.join(run_dir, 'retino_analysis', 'analysisids_*.json'))[0]
    with open(retinoids_fpath, 'r') as f:
        rids = json.load(f)
    roi_analyses = [r for r, rinfo in rids.items() if rinfo['PARAMS']['roi_type'] != 'pixels']
    if retinoid not in roi_analyses:
        retinoid = sorted(roi_analyses, key=natural_keys)[-1] # use most recent roi analysis
        print("Fixed retino id to most recent: %s" % retinoid)
    retinoid_info = rids[retinoid]
    
    data_identifier = '|'.join([animalid, session, fov, run, retinoid])
    
    print("*** Dataset: %s ***" % data_identifier)
    
    #%%
    
    # Get processed retino data:
    processed_dir = glob.glob(os.path.join(run_dir, 'retino_analysis', '%s*' % retinoid))[0]
    processed_fpaths = glob.glob(os.path.join(processed_dir, 'files', '*.h5'))
    print("Found %i processed retino runs." % len(processed_fpaths))
    
    # Get condition info for trials:
    conditions_fpath = glob.glob(os.path.join(run_dir, 'paradigm', 'files', '*.json'))[0]
    with open(conditions_fpath, 'r') as f:
        mwinfo = json.load(f)

    
    # Get run info:
    runinfo_fpath = glob.glob(os.path.join(run_dir, '*.json'))[0]
    with open(runinfo_fpath, 'r') as f:
        runinfo = json.load(f)

    # Get stimulus info:
    stiminfo, trials_by_cond = rutils.get_retino_stimulus_info(mwinfo, runinfo)
   # stiminfo['trials_by_cond'] = trials_by_cond
    print "---------------------------------"
    print "Trials by condN:",trials_by_cond
    
    #%%
    
    # Comine all trial data into data frames:
    fit, magratio, phase, trials_by_cond = trials_to_dataframes(processed_fpaths, conditions_fpath)
    print fit.head()
    print trials_by_cond

    # Correct phase to wrap around:
    corrected_phase = correct_phase_wrap(phase)
    
    # Get screen info:
    screen = visroi.get_screen_info(animalid, session, rootdir=rootdir)
    
    # Convert phase to linear coords:
    screen_left = -1*screen['azimuth']/2.
    screen_right = screen['azimuth']/2. #screen['azimuth']/2.
    screen_lower = -1*screen['elevation']/2.
    screen_upper = screen['elevation']/2. #screen['elevation']/2.
    
    #%%
    # Create output dir:

    if len(trials_by_cond.keys()) == 4:
        absolute = True
        loctype = 'absolute'
    else:
        absolute = False
        loctype = 'relative'
        
    output_dir = os.path.join(processed_dir, 'visualization', 'target_vf_%s' % loctype)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    #%%
    
    # Identify "good" cells by mag-ratio or FIT from regression (do_retinotopy_analysis)
    # -------------------------------------------------------------------------
    threshold = opts.threshold
    fig = plot_signal_fits_by_roi(fit, magratio, threshold=threshold, data_identifier=data_identifier,
                                  fov=fov, retinoid=retinoid, output_dir=output_dir)

    
#%%

    # -------------------------------------------------------------------------
    # Get AVERAGE of phase values across condition reps.
    # Convert to linear coords.
    # -------------------------------------------------------------------------
    
    # Get MW info:
    stim_positions = dict()
    stim_tstamps = dict()
    for cond in trials_by_cond.keys():
        stim_positions[cond], stim_tstamps[cond] = rutils.get_interp_positions(cond, mwinfo, stiminfo, trials_by_cond)

    #----
    mag_thr = magratio.max(axis=1).max() * 0.25 #0.02
    #mag_thr = magratio.max(axis=1).max() * 0.25 #5 #0.02

    #fit_thr = 0.20
    absolute = True
    absolute_coords, strong_cells = get_absolute_centers(phase, magratio, trials_by_cond, stim_positions, \
                                                                    absolute=absolute, mag_thr=mag_thr)

    # Screen dims are not necessarily left/right and bottom/top starting pos for cycle (off-scren start)
    [bottom_pos, left_pos, top_pos, right_pos] = absolute_coords['screen_bb']

    mean_phase_az = absolute_coords['mean_phase_az']
    mean_phase_el = absolute_coords['mean_phase_el']
    linX = absolute_coords['linX']
    linY = absolute_coords['linY']
    use_relative = absolute_coords['used_relative']
    
    # Plot ROI centers using phase values as sanity check against conversion to linear coords:
    fig, ax = pl.subplots(figsize=(10,5)) # pl.figure()
    ax.scatter(mean_phase_az, mean_phase_el)
    if use_relative:
        ax.invert_xaxis()
        ax.invert_yaxis()
    fig.savefig(os.path.join(output_dir, 'roi_centers_phase_space_sanitycheck_%s.png' % loctype))
        
    # Plot CoM:
    # -------------------------------------------------------------------------                          
    fig = get_center_of_mass(fit, linX, linY, screen, marker_scale=200)
        
    label_figure(fig, data_identifier)
    pl.savefig(os.path.join(output_dir, 'mean_xy_CoM_%s.png' % loctype))
    
    
    #
    #%% 
    
    # Add "good" ROIs from event-based runs (e.g., gratings if diagnostic)
    # -------------------------------------------------------------------------
    select_rois = None #'gratings_run1' #'gratings_run1' #'gratings_run1' #combined_gratings_static' # 'combined_gratings_static' # None #'gratings_run1'
    labeled_rois = []
    # Load "selective cells" and label:
    if select_rois is not None:
        fov_dir = os.path.split(run_dir)[0]
        # Get traceids of selected run extracted with same ROIs as retino:
        select_rois_dir = glob.glob(os.path.join(fov_dir, select_rois))[0]
        if 'combined' in select_rois:
            # Load one of the combined runs:
            combo_base = select_rois.split('_')[1]
            traceids_fpath = glob.glob(os.path.join(fov_dir, '%s_run*' % combo_base, 'traces', 'traceids*.json'))[0]
        else:
            traceids_fpath = glob.glob(os.path.join(select_rois_dir, 'traces', 'traceids*.json'))[0]
        with open(traceids_fpath, 'r') as f:
            tids = json.load(f)
        matching_traceids = [t for t, tinfo in tids.items() if tinfo['PARAMS']['roi_id'] == retinoid_info['PARAMS']['roi_id']]
        assert len(matching_traceids) > 0, "ERROR: No traceids for run -- %s -- with retino roi id (%s)" % (select_rois, retinoid_info['PARAMS']['roi_id'])
        traceid = matching_traceids[0]
        
        roistats_fpath = glob.glob(os.path.join(select_rois_dir, 'traces', '%s*' % traceid, 'sorted_rois', 'roistats*.npz'))[0]
        roistats = np.load(roistats_fpath)
        labeled_rois = roistats['sorted_selective']
        if len(labeled_rois) > 20:
            labeled_rois = labeled_rois[0:20]
        
    
    #%%
    # -----------------------------------------------------------------------------
    # Visualize FITS by condition:
    # -----------------------------------------------------------------------------
    #use_linear = True
    
    # fig, good_fits = visualize_fits_by_condition(fit, magratio, corrected_phase, trials_by_cond, screen, 
    #                                              labeled_rois=[], use_linear=use_linear, #use_circ=use_circ,
    #                                              fit_thresh_az=fit_thresh_az, fit_thresh_el=fit_thresh_el,
    #                                              data_identifier=data_identifier, fov=fov, retinoid=retinoid,
    #                                              output_dir=output_dir)
    
    #%%

    # Smooth ROI centroid on screen and visualize as heatmap to find "hot spots"
    # -------------------------------------------------------------------------
    fig = pl.figure(figsize=(10,6))
    ax = pl.subplot2grid((1, 2), (0, 0), colspan=2, fig=fig)
    #screen_divs_az = int(round(screen['azimuth']))
    #screen_divs_el = int(round(screen['elevation']))

    screen_divs_az = int(round(right_pos - left_pos))
    screen_divs_el = int(round(top_pos - bottom_pos))
    #heatmap, xedges, yedges = np.histogram2d(linX.values, linY.values, bins=(screen_divs_az, screen_divs_el))
    heatmap, xedges, yedges = np.histogram2d(linX, linY, bins=(screen_divs_az, screen_divs_el))
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    ax.imshow(heatmap.T, extent=extent, origin='lower')
    ax.set_ylim([screen_lower, screen_upper])
    ax.set_xlim([screen_left, screen_right])

    # Apply gaussian filter
    sigma_val = 5
    sigma = [sigma_val, sigma_val]
    smoothed = sp.ndimage.filters.gaussian_filter(heatmap, sigma, mode='constant')
    ax.imshow(smoothed.T, extent=extent, origin='lower')
    #ax.invert_yaxis()

    label_figure(fig, data_identifier)
    fig.savefig(os.path.join(output_dir, 'smoothed_heatmap_rois_on_screen_%s.png' % loctype))
    

#    # Draw azimuth value as a function of mean fit (color code by standard cmap, too)
#    ax.scatter(linX, linY, s=mean_fits*marker_scale, alpha=0.5) # cmap='nipy_spectral', vmin=screen_left, vmax=screen_right)
#    ax.set_xlim([screen_left, screen_right])
#    ax.set_ylim([screen_lower, screen_upper])
#    ax.set_xlabel('xpos (deg)')
#    ax.set_ylabel('ypos (deg)') 
    
    #%%
#    from sklearn.neighbors import KernelDensity
#    from sklearn.grid_search import GridSearchCV
#    
    # 1.  Find best BW, KDE with sklearn:
#    vals = np.linspace(screen_left, screen_right, len(mean_fits))
#    grid = GridSearchCV(KernelDensity(),
#                        {'bandwidth': np.linspace(0.1, 1.0, 30)},
#                        cv=5) # 20-fold cross-validation
#    grid.fit(linX.reshape(-1, 1))
#    print grid.best_params_
#
#    kde = grid.best_estimator_
#    pdf = np.exp(kde.score_samples(vals.reshape(-1, 1)))
#    
#    fig, ax = pl.subplots()
#    ax.plot(vals, pdf, linewidth=3, alpha=0.5, label='bw=%.2f' % kde.bandwidth)
#    ax.hist(linX.values, 30, fc='gray', histtype='stepfilled', alpha=0.3, normed=True)
#    ax.legend(loc='upper left')
    # ^ bleh looks funky
    
    mean_mags = magratio.mean(axis=1)

    
    # Plot KDE:
    j = sns.jointplot(linX, linY, kind='kde', xlim=(screen_left, screen_right), ylim=(screen_lower, screen_upper))
    elev_x, elev_y = j.ax_marg_y.lines[0].get_data()
    azim_x, azim_y = j.ax_marg_x.lines[0].get_data()

    smstats_kde_az = sp.stats.gaussian_kde(linX) #, weights=mean_fits)
    #az_vals = np.linspace(screen_left, screen_right, len(mean_mags))
    az_vals = np.linspace(left_pos, right_pos, len(mean_mags))
    
    smstats_kde_el = sp.stats.gaussian_kde(linY)
    #el_vals = np.linspace(screen_lower, screen_upper, len(mean_mags))
    el_vals = np.linspace(bottom_pos, top_pos, len(mean_mags))
    

    smstats_az = smstats_kde_az(az_vals)
    smstats_el = smstats_kde_el(el_vals)
    #wa = kdea(vals)
#    fig, ax = pl.subplots() #pl.figure()
#    ax.plot(vals, wa)
#    ax.plot(azim_x, azim_y)


    # 2. Use weights with KDEUnivariate (no FFT):
    #weighted_kde_az = sm.nonparametric.kde.KDEUnivariate(linX.values)
    weighted_kde_az = sm.nonparametric.kde.KDEUnivariate(linX)
    weighted_kde_az.fit(weights=mean_mags.values, fft=False)
    #weighted_kde_el = sm.nonparametric.kde.KDEUnivariate(linY.values)
    weighted_kde_el = sm.nonparametric.kde.KDEUnivariate(linY)
    weighted_kde_el.fit(weights=mean_mags.values, fft=False)
    
    fig, axes = pl.subplots(1,2, figsize=(10,5))

    axes[0].set_title('azimuth')    
    axes[0].plot(weighted_kde_az.support, weighted_kde_az.density, label='KDEuniv')
    axes[0].plot(azim_x, azim_y, label='sns-marginal (unweighted)')
    axes[0].plot(az_vals, smstats_az, label='gauss-kde (unweighted)')
    
    axes[1].set_title('elevation')    
    axes[1].plot(weighted_kde_el.support, weighted_kde_el.density, label='KDEuniv')
    axes[1].plot(elev_y, elev_x, label='sns-marginal (unweighted)')
    axes[1].plot(el_vals, smstats_el, label='gauss-kde (unweighted)')
    axes[1].legend(fontsize=8)
    
    pl.savefig(os.path.join(output_dir, 'compare_kde_weighted_%s.png' % loctype))
        

    # Plot weighted KDE to marginals on joint plot:
    j.ax_marg_y.plot(weighted_kde_el.density, weighted_kde_el.support, color='orange', label='weighted')
    j.ax_marg_x.plot(weighted_kde_az.support, weighted_kde_az.density, color='orange', label='weighted')
    j.ax_marg_x.set_ylim([0, max([j.ax_marg_x.get_ylim()[-1], weighted_kde_az.density.max()]) + 0.005])
    j.ax_marg_y.set_xlim([0, max([j.ax_marg_y.get_xlim()[-1], weighted_kde_el.density.max()]) + 0.005])
    j.ax_marg_x.legend(fontsize=8)
    
    j.savefig(os.path.join(output_dir, 'weighted_marginals_%s.png' % loctype))
    
    


#%%
    kde_az =  weighted_kde_az.density.copy()
    vals_az = weighted_kde_az.support.copy()
    
    kde_el = weighted_kde_el.density.copy()
    vals_el = weighted_kde_el.support.copy()
    
    az_max, az_min1, az_min2, az_maxima, az_minima = find_local_min_max(vals_az, kde_az)
    el_max, el_min1, el_min2, el_maxima, el_minima = find_local_min_max(vals_el, kde_el)

    

    fig, axes = pl.subplots(1,2, figsize=(10,5)) #pl.figure();
    plot_kde_min_max(vals_az, kde_az, maxval=az_max, minval1=az_min1, minval2=az_min2, title='azimuth', ax=axes[0])
    plot_kde_min_max(vals_el, kde_el, maxval=el_max, minval1=el_min1, minval2=el_min2, title='elevation', ax=axes[1])
    
    label_figure(fig, data_identifier)
    fig.savefig(os.path.join(output_dir, 'weighted_kde_min_max_%s.png' % loctype))
    
    az_bounds = sorted([float(vals_az[az_min1]), float(vals_az[az_min2])])
    el_bounds = sorted([float(vals_el[el_min1]), float(vals_el[el_min2])])
    # Make sure bounds are within screen:
    if az_bounds[0] < screen_left:
        az_bounds[0] = screen_left
    if az_bounds[1] > screen_right:
        az_bounds[1] = screen_right
    if el_bounds[0] < screen_lower:
        el_bounds[0] = screen_lower
    if el_bounds[1] > screen_upper:
        el_bounds[1] = screen_upper
        
    kde_results = {'az_max': vals_az[az_max],
                   'el_max': vals_el[el_max],
                   'az_maxima': [vals_az[azm] for azm in az_maxima],
                   'el_maxima': [vals_el[elm] for elm in el_maxima],
                   'az_bounds': az_bounds,
                   'el_bounds': el_bounds,
                   'center_x': az_bounds[1] - (az_bounds[1]-az_bounds[0]) / 2.,
                   'center_y': el_bounds[1] - (el_bounds[1]-el_bounds[0]) / 2. }


    print("AZIMUTH bounds: %s" % str(kde_results['az_bounds']))
    print("ELEV bounds: %s" % str(kde_results['el_bounds']))
    print("CENTER: %.2f, %.2f" % (kde_results['center_x'], kde_results['center_y']))
    
    
    
    min_thr = 0.01
    
    marker_scale = 100./round(magratio.mean().mean(), 3)
    fig = plot_kde_maxima(kde_results, magratio, linX, linY, screen, \
                          use_peak=True, marker_scale=marker_scale, exclude_bad=True, min_thr=min_thr)
    print("LINX:", linX.shape)
    for ri in strong_cells:
        fig.axes[0].text(linX[ri], linY[ri], '%s' % (ri+1))
    label_figure(fig, data_identifier)
    pl.savefig(os.path.join(output_dir, 'centroid_peak_rois_by_pos_%s.png' % (loctype)))


#    fig = plot_kde_maxima(kde_results, magratio, linX, linY, screen, use_peak=False, marker_scale=marker_scale)
#    print("LINX:", linX.shape)
#    for ri in strong_cells:
#        fig.axes[0].text(linX[ri], linY[ri], '%s' % (ri+1))
#    label_figure(fig, data_identifier)
#    pl.savefig(os.path.join(output_dir, 'centroid_kdecenter_rois_by_pos_%s.png' % (loctype)))
    
    with open(os.path.join(output_dir, 'fit_centroid_results_%s.json' % loctype), 'w') as f:
        json.dump(kde_results, f, sort_keys=True, indent=4)


    #%%


if __name__ == '__main__':
    main(sys.argv[1:])


    

#%%


