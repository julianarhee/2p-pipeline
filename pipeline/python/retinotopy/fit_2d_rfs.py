#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 15:25:28 2019

@author: julianarhee
"""


import os
import glob
import json
import copy
import optparse
import sys
import traceback
import matplotlib as mpl
mpl.use('agg')
import pylab as pl
import seaborn as sns
import cPickle as pkl
import numpy as np
import scipy as sp
from scipy import stats
import pandas as pd
import matplotlib as mpl
import matplotlib.cm as cm
    
import scipy.optimize as opt
from matplotlib.patches import Ellipse, Rectangle

from mpl_toolkits.axes_grid1 import AxesGrid
from pipeline.python.utils import natural_keys, label_figure, load_data
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import argrelextrema
from scipy.interpolate import splrep, sproot, splev, interp1d

from pipeline.python.retinotopy import utils as rutils
#from pipeline.python.traces import utils as tutils
from pipeline.python.classifications import test_responsivity as resp
from pipeline.python.rois.utils import load_roi_masks, get_roiid_from_traceid
from pipeline.python.retinotopy import convert_coords as coor
from matplotlib.pyplot import cm
import statsmodels as sm
from pipeline.python.retinotopy import target_visual_field as targ
from pipeline.python.traces.trial_alignment import aggregate_experiment_runs 



def rfits_to_df(rffits, row_vals=[], col_vals=[], roi_list=None,
                scale_size=False, sigma_scale=2.35):
    '''
    Takes each roi's RF fit results, converts to screen units, and return as dataframe.
    Scale to make size FWFM if scale_size is True.
    '''
    if roi_list is None:
        roi_list = sorted(rffits.keys())
       
    scale_ = sigma_scale if scale_size else 1.0

    rf_fits_df = pd.DataFrame({'x0': [rffits[r]['x0'] for r in roi_list],
                               'y0': [rffits[r]['y0'] for r in roi_list],
                               'sigma_x': [rffits[r]['sigma_x'] for r in roi_list],
                               'sigma_y': [rffits[r]['sigma_y'] for r in roi_list],
                               'theta': [rffits[r]['theta'] for r in roi_list],
                               'r2': [rffits[r]['r2'] for r in roi_list]},
                              index=roi_list)

    x0, y0, sigma_x, sigma_y = convert_fit_to_coords(rf_fits_df, row_vals, col_vals)
    rf_fits_df['x0'] = x0
    rf_fits_df['y0'] = y0
    rf_fits_df['sigma_x'] = sigma_x * scale_
    rf_fits_df['sigma_y'] = sigma_y * scale_ 
    
    return rf_fits_df



def convert_values(oldval, newmin=None, newmax=None, oldmax=None, oldmin=None):
    oldrange = (oldmax - oldmin)
    newrange = (newmax - newmin)
    newval = (((oldval - oldmin) * newrange) / oldrange) + newmin
    return newval

def convert_fit_to_coords(fitdf, row_vals, col_vals, rid=None):
    
    if rid is not None:
        xx = convert_values(fitdf['x0'][rid], 
                            newmin=min(col_vals), newmax=max(col_vals), 
                            oldmax=len(col_vals)-1, oldmin=0)
        
        sigma_x = convert_values(abs(fitdf['sigma_x'][rid]), 
                            newmin=0, newmax=max(col_vals)-min(col_vals), 
                            oldmax=len(col_vals)-1, oldmin=0)
        
        yy = convert_values(fitdf['y0'][rid], 
                            newmin=min(row_vals), newmax=max(row_vals), 
                            oldmax=len(row_vals)-1, oldmin=0)
        
        sigma_y = convert_values(abs(fitdf['sigma_y'][rid]), 
                            newmin=0, newmax=max(row_vals)-min(row_vals), 
                            oldmax=len(row_vals)-1, oldmin=0)
    else:
        xx = convert_values(fitdf['x0'], 
                            newmin=min(col_vals), newmax=max(col_vals), 
                            oldmax=len(col_vals)-1, oldmin=0)
        
        sigma_x = convert_values(abs(fitdf['sigma_x']), 
                            newmin=0, newmax=max(col_vals)-min(col_vals), 
                            oldmax=len(col_vals)-1, oldmin=0)
        
        yy = convert_values(fitdf['y0'], 
                            newmin=min(row_vals), newmax=max(row_vals), 
                            oldmax=len(row_vals)-1, oldmin=0)
        
        sigma_y = convert_values(abs(fitdf['sigma_y']), 
                            newmin=0, newmax=max(row_vals)-min(row_vals), 
                            oldmax=len(row_vals)-1, oldmin=0)
    
    return xx, yy, sigma_x, sigma_y


#%%

# -----------------------------------------------------------------------------
# Data formatting functions:
# -----------------------------------------------------------------------------

def get_trials_by_cond(labels):
    # Get single value for each trial and sort by config:
    trials_by_cond = dict()
    for k, g in labels.groupby(['config']):
        trials_by_cond[k] = sorted([int(tr[5:])-1 for tr in g['trial'].unique()])
    return trials_by_cond


def group_trial_values_by_cond(zscores, trials_by_cond):
    resp_by_cond = dict()
    for cfg, trial_ixs in trials_by_cond.items():
        resp_by_cond[cfg] = zscores.iloc[trial_ixs]  # For each config, array of size ntrials x nrois

    trialvalues_by_cond = pd.DataFrame([resp_by_cond[cfg].mean(axis=0) \
                                            for cfg in sorted(resp_by_cond.keys(), key=natural_keys)]) # nconfigs x nrois
         
    return trialvalues_by_cond


def process_traces(raw_traces, labels, response_type='zscore', nframes_post_onset=None):        
    # Get stim onset frame: 
    stim_on_frame = labels['stim_on_frame'].unique()
    assert len(stim_on_frame) == 1, "---[stim_on_frame]: More than 1 stim onset found: %s" % str(stim_on_frame)
    stim_on_frame = stim_on_frame[0]
    
    # Get n frames stimulus on:
    nframes_on = labels['nframes_on'].unique()
    assert len(nframes_on) == 1, "---[nframes_on]: More than 1 stim dur found: %s" % str(nframes_on)
    nframes_on = nframes_on[0]
        
    if nframes_post_onset is None:
        nframes_post_onset = nframes_on*2
        
    zscored_traces_list = []
    zscores_list = []
    #snrs_list = []
    for trial, tmat in labels.groupby(['trial']):

        # Get traces using current trial's indices: divide by std of baseline
        curr_traces = raw_traces.iloc[tmat.index] 
        bas_std = curr_traces.iloc[0:stim_on_frame].std(axis=0)
        bas_mean = curr_traces.iloc[0:stim_on_frame].mean(axis=0)
        if response_type == 'zscore':
            curr_zscored_traces = pd.DataFrame(curr_traces).subtract(bas_mean).divide(bas_std, axis='columns')
        else:
            curr_zscored_traces = pd.DataFrame(curr_traces).subtract(bas_mean).divide(bas_mean, axis='columns')
        zscored_traces_list.append(curr_zscored_traces)
        
        # Also get zscore (single value) for each trial:
        stim_mean = curr_traces.iloc[stim_on_frame:stim_on_frame+nframes_post_onset].mean(axis=0)
        if response_type == 'zscore':
            zscores_list.append((stim_mean-bas_mean)/bas_std)
        elif response_type == 'snr':
            zscores_list.append(stim_mean/bas_mean)
        elif response_type == 'meanstim':
            zscores_list.append(stim_mean)
        elif response_type == 'dff':
            zscores_list.append((stim_mean-bas_mean) / bas_mean)
        
        #zscores_list.append(curr_zscored_traces.iloc[stim_on_frame:stim_on_frame+nframes_post_onset].mean(axis=0)) # Get average zscore value for current trial
        
    zscored_traces = pd.concat(zscored_traces_list, axis=0)
    zscores =  pd.concat(zscores_list, axis=1).T # cols=rois, rows = trials
    
    return zscored_traces, zscores


#%%

#%


#def order_configs_by_grid(sdf, labels, rows='ypos', cols='xpos'):
#    
#    assert rows in sdf.columns, "Specified ROWS <%s> not found." % rows
#    assert cols in sdf.columns, "Specified COLS <%s> not found." % cols
#    
#    row_vals = sorted(sdf[rows].unique())
#    col_vals = sorted(sdf[cols].unique())
#    
#    config_trial_ixs = dict()
#    cix = 0
#    for si, row_val in enumerate(sorted(row_vals)):
#        for mi, col_val in enumerate(col_vals):
#            config_trial_ixs[cix] = {}
#            cfg = sdf[(sdf[rows]==row_val) & (sdf[cols]==col_val)].index.tolist()[0]
#            trial_ixs = sorted( list(set( [int(tr[5:])-1 for tr in labels[labels['config']==cfg]['trial']] )) )
#            config_trial_ixs[cix]['config'] = cfg
#            config_trial_ixs[cix]['trial_ixs'] = trial_ixs
#            cix += 1
#
#    return row_vals, col_vals, config_trial_ixs
#


#%%

# -----------------------------------------------------------------------------
# ROI filtering functions:
# -----------------------------------------------------------------------------


def get_rois_by_visual_area(fov_dir, visual_area=''):
    
    included_rois = []
    segmentations = glob.glob(os.path.join(fov_dir, 'visual_areas', '*.pkl'))
    assert len(segmentations) > 0, "Specified to segment, but no segmentation file found in acq. dir!"
    if len(segmentations) == 1:
        segmentation_fpath = segmentations[0]
    else:
        for si, spath in enumerate(sorted(segmentations, key=natural_keys)):
            print si, spath
        sel = input("Select IDX of seg file to use: ")
        segmentation_fpath = sorted(segmentations, key=natural_keys)[sel]
    with open(segmentation_fpath, 'rb') as f:
        seg = pkl.load(f)
            
    included_rois = seg.regions[visual_area]['included_rois']
    print "Found %i rois in visual area %s" % (len(included_rois), visual_area)

    return included_rois



def get_responsive_rois(traceid_dir, included_rois=[]):
    # Set dirs:
    try:
        sorting_subdir = 'response_stats'
        sorted_dir = sorted(glob.glob(os.path.join(traceid_dir, '%s*' % sorting_subdir)))[-1]
    except Exception as e:
        sorting_subdir = 'sorted_rois'
        sorted_dir = sorted(glob.glob(os.path.join(traceid_dir, '%s*' % sorting_subdir)))[-1]
    sort_name = os.path.split(sorted_dir)[-1]
    print "Selected stats results: %s" % sort_name
    
    
    # Load roi stats:    
    rstats_fpath = glob.glob(os.path.join(sorted_dir, 'roistats_results.npz'))[0]
    rstats = np.load(rstats_fpath)
    
    #%
    if len(included_rois) > 0:
        all_rois = np.array(copy.copy(included_rois))
    else:
        all_rois = np.arange(0, rstats['nrois_total'])
    
    visual_rois = np.array([r for r in rstats['sorted_visual'] if r in all_rois])
    selective_rois = np.array([r for r in rstats['sorted_selective'] if r in all_rois])
    
    print "Found %i cells that pass responsivity test (%s, p<%.2f)." % (len(visual_rois), rstats['responsivity_test'], rstats['visual_pval'])
    print "Found %i cells that pass responsivity test (%s, p<%.2f)." % (len(selective_rois), rstats['selectivity_test'], rstats['selective_pval'])
    
    #del rstats
    
    return visual_rois, selective_rois, rstats_fpath




def sort_rois_by_max_response(avg_zscores_by_cond, visual_rois):
    '''
    Expects a dataframe with shape (nconds, nrois), i.e., a single value
    assigned to each ROI for each condition for sorting.
    '''
    visual_max_avg_zscore = np.array([avg_zscores_by_cond[rid].max() for rid in visual_rois])
    visual_sort_by_max_zscore = np.argsort(visual_max_avg_zscore)[::-1]
    sorted_visual = visual_rois[visual_sort_by_max_zscore]
    
    return sorted_visual


#%%

# -----------------------------------------------------------------------------
# Fitting functions
# -----------------------------------------------------------------------------

def twoD_Gaussian((x, y), amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    # b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    b = (np.sin(2*theta))/(2*sigma_x**2) - (np.sin(2*theta))/(2*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
#    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
#                            + c*((y-yo)**2)))
    g = offset + amplitude*np.exp( -a*((x-xo)**2) - b*(x-xo)*(y-yo) - c*((y-yo)**2) )
    return g.ravel()


def twoD_gauss((x, y), b, x0, y0, sigma_x, sigma_y, theta, a):

    RF = a + b * np.exp( -( ((x-x0)*np.cos(theta) + (y-y0)*np.sin(theta)) / (np.sqrt(2)*sigma_x) )**2 - ( ( -(x-x0)*np.sin(theta) + (y-y0)*np.cos(theta) ) / (np.sqrt(2)*sigma_y) )**2 )
    
    return RF.ravel()

#%

#import scipy as sp
#
#def gaussian(height, center_x, center_y, width_x, width_y, rotation):
#    """Returns a gaussian function with the given parameters"""
#    width_x = float(width_x)
#    width_y = float(width_y)
#
#    rotation = np.deg2rad(rotation)
#    center_x = center_x * np.cos(rotation) - center_y * np.sin(rotation)
#    center_y = center_x * np.sin(rotation) + center_y * np.cos(rotation)
#
#    def rotgauss(x,y):
#        xp = x * np.cos(rotation) - y * np.sin(rotation)
#        yp = x * np.sin(rotation) + y * np.cos(rotation)
#        g = height*np.exp(
#            -(((center_x-xp)/width_x)**2+
#              ((center_y-yp)/width_y)**2)/2.)
#        return g
#    return rotgauss
#
#def moments(data):
#    """Returns (height, x, y, width_x, width_y)
#    the gaussian parameters of a 2D distribution by calculating its
#    moments """
#    total = data.sum()
#    X, Y = np.indices(data.shape)
#    
#    x = (X*data).sum()/total
#    y = (Y*data).sum()/total
#    col = data[:, int(y)]
#    width_x = np.sqrt(abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
#    row = data[int(x), :]
#    width_y = np.sqrt(abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
#        
#        
##    y = (X*data).sum()/total
##    x = (Y*data).sum()/total
##    
##    row = data[int(y), :] # Get all 'x' values where y==center  #col = data[:, int(y)]
##    width_x = np.sqrt(abs((np.arange(row.size)-y)**2*row).sum()/row.sum())
##    
##    col = data[:, int(x)]
##    width_y = np.sqrt(abs((np.arange(col.size)-x)**2*col).sum()/col.sum())
##    
#    
#    
#    height = data.max()
#    #return height, x, y, width_x, width_y, 0.0
#    return height, y, x, width_y, width_x, 0.0
#
#
#def fitgaussian(data):
#    """Returns (height, x, y, width_x, width_y)
#    the gaussian parameters of a 2D distribution found by a fit"""
#    params = moments(data)
#    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) - data)
#    p, pcov, infod, msg, success = sp.optimize.leastsq(errorfunction, params, full_output=True)
#    return p, pcov
        
#
#%

class MultiplePeaks(Exception): pass
class NoPeaksFound(Exception): pass

def fwhm(x, y, k=3):
    """
    Determine full-with-half-maximum of a peaked set of points, x and y.

    Assumes that there is only one peak present in the datasset.  The function
    uses a spline interpolation of order k.
    """

    half_max = max(y)/2.0
    s = splrep(x, y - half_max, k=k)
    roots = sproot(s)

    if len(roots) > 2:
        return None
#        raise MultiplePeaks("The dataset appears to have multiple peaks, and "
#                "thus the FWHM can't be determined.")
    elif len(roots) < 2:
        return None
#        raise NoPeaksFound("No proper peaks were found in the data set; likely "
#                "the dataset is flat (e.g. all zeros).")
    else:
        return abs(roots[1] - roots[0])


def raw_fwhm(arr):
    
    interpf = interp1d(np.linspace(0, len(arr)-1, num=len(arr)), arr, kind='linear')
    xnew = np.linspace(0, len(arr)-1, num=len(arr)*3)
    ynew = interpf(xnew)
    
    
    hm = ((ynew.max() - ynew.min()) / 2 ) + ynew.min()
    pk = ynew.argmax()
    if pk == 0:
        r2 = pk + np.abs(ynew[pk:] - hm).argmin()
        return abs(xnew[r2]*2)
    else:
        r1 = np.abs(ynew[0:pk] - hm).argmin()
        r2 = pk + np.abs(ynew[pk:] - hm).argmin()
        
        return abs(xnew[r2]-xnew[r1])
    


def get_rf_map(response_vector, ncols, nrows):
    
    coordmap_r = np.reshape(response_vector, (ncols, nrows)).T
    
    return coordmap_r

def plot_roi_RF(response_vector, ncols, nrows, ax=None, trim=True,
                hard_cutoff=True,
                set_to_min=True, map_thr=2.0, perc_min=0.1):
    
    if ax is None:
        fig, ax = pl.subplots()
        
    coordmap_r = np.reshape(response_vector, (ncols, nrows)).T
     
    rfmap = coordmap_r.copy()
    if trim:
        if hard_cutoff:
            rfmap[coordmap_r < map_thr] = coordmap_r.min()*perc_min if set_to_min else 0
        else:
            rfmap[coordmap_r <= (coordmap_r.max()*map_thr)] = coordmap_r.min()*perc_min if set_to_min else 0
        
    im = ax.imshow(rfmap, cmap='inferno')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    ax.figure.colorbar(im, cax=cax, orientation='vertical')
    
    return ax, rfmap


    
def do_2d_fit(rfmap, nx=None, ny=None, verbose=False):

    # Set params for fit:
    xi = np.arange(0, nx)
    yi = np.arange(0, ny)
    popt=None; pcov=None; fitr=None; r2=None; success=None;
    xx, yy = np.meshgrid(xi, yi)
    initial_guess = None
    try:
            
        #plot_xx = xx.copy()
        #plot_yy = yy.copy()
        amplitude = rfmap.max()

        y0, x0 = np.where(rfmap == rfmap.max())
        #print "x0, y0: (%i, %i)" % (int(x0), int(y0))
    
        try:
            sigma_x = fwhm(xi, rfmap.sum(axis=0))
            assert sigma_x is not None
        except Exception as e:
            #print e
            sigma_x = raw_fwhm(rfmap.sum(axis=0)) 
        try:
            sigma_y = fwhm(yi, rfmap.sum(axis=1))
            assert sigma_y is not None
        except Exception as e:
            #print e
            sigma_y = raw_fwhm(rfmap.sum(axis=1))
        #print "sig-X, sig-Y:", sigma_x, sigma_y
        theta = 0
        offset=0
        initial_guess = (amplitude, int(x0), int(y0), sigma_x, sigma_y, theta, offset)
        popt, pcov = opt.curve_fit(twoD_gauss, (xx, yy), rfmap.ravel(), p0=initial_guess, maxfev=2000)
        fitr = twoD_gauss((xx, yy), *popt)
            
        # Get residual sum of squares 
        residuals = rfmap.ravel() - fitr
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((rfmap.ravel() - np.mean(rfmap.ravel()))**2)
        r2 = 1 - (ss_res / ss_tot)
        if len(np.where(fitr > fitr.min())[0]) < 2 or pcov.max() == np.inf or r2 == 1: #round(r2, 3) < 0.15 or 
            success = False
        else:
            success = True
    except Exception as e:
        if verbose:
            print e
    
    return {'popt': popt, 'pcov': pcov, 'init': initial_guess, 'r2': r2, 'success': success}, fitr

#%
# -----------------------------------------------------------------------------
# PLOTTING FUNCTIONS:
# -----------------------------------------------------------------------------

def plot_and_fit_roi_RF(response_vector, row_vals, col_vals, 
                        min_sigma=5, max_sigma=50, sigma_scale=2.35,
                        trim=False, hard_cutoff=False, map_thr=None, set_to_min=False, perc_min=None):
#
#    if not trim:
#        set_to_min = False
#        hard_cutoff = False
#        set_to_min_str = ''    
#        cutoff_type = 'no_trim'
#        map_thr=''
#    else:
#        cutoff_type = 'hard_thr' if hard_cutoff else 'perc_min'
#        map_thr = 1.5 if (trim and hard_cutoff) else perc_min
#        
#    set_to_min_str = 'set_min' if set_to_min else ''
#
#    
    results = {}
    fig, axes = pl.subplots(1,2, figsize=(8, 4)) # pl.figure()
    ax = axes[0]
    #ax, rfmap = plot_roi_RF(avg_zscores_by_cond[rid], ncols=len(col_vals), nrows=len(row_vals), ax=ax)
    ax, rfmap = plot_roi_RF(response_vector, ax=ax,
                            ncols=len(col_vals), nrows=len(row_vals), trim=trim,
                            perc_min=perc_min,
                            hard_cutoff=False, map_thr=map_thr, set_to_min=set_to_min)

    ax2 = axes[1]

    # Do fit 
    # ---------------------------------------------------------------------
    denoised=False
    if hard_cutoff and (rfmap.max() < map_thr):
        fitr = {'success': False}
    else:
        fitr, fit_y = do_2d_fit(rfmap, nx=len(col_vals), ny=len(row_vals))
        if rfmap.max() > 3.0 and fit_y is None:
            try:
                rfmap[rfmap<rfmap.max()*0.2] = rfmap.min()
                fitr, fit_y = do_2d_fit(rfmap, nx=len(col_vals), ny=len(row_vals))
                assert fitr is not None, "--- no fit, trying with denoised..."
                denoised=True
            except Exception as e:
                print e
                pass
            
    xres = np.mean(np.diff(sorted(row_vals)))
    yres = np.mean(np.diff(sorted(col_vals)))
    min_sigma = xres
    
    if fitr['success']:
        amp_f, x0_f, y0_f, sigx_f, sigy_f, theta_f, offset_f = fitr['popt']
        if any(s < min_sigma for s in [abs(sigx_f)*xres*sigma_scale, abs(sigy_f)*yres*sigma_scale])\
            or any(s > max_sigma for s in [abs(sigx_f)*xres*sigma_scale, abs(sigy_f)*yres*sigma_scale]):
            fitr['success'] = False

    if fitr['success']:    
        # Draw ellipse: #ax.contour(plot_xx, plot_yy, fitr.reshape(rfmap.shape), 3, colors='b')
        amp_f, x0_f, y0_f, sigx_f, sigy_f, theta_f, offset_f = fitr['popt']
        ell = Ellipse((x0_f, y0_f), abs(sigx_f)*sigma_scale, abs(sigy_f)*sigma_scale, 
                      angle=np.rad2deg(theta_f), alpha=0.5, edgecolor='w') #theta_f)
        ax.add_patch(ell)
        ax.text(0, -1, 'r2=%.2f, theta=%.2f' % (fitr['r2'], theta_f), color='k')
        
        # Visualize fit results:
        im2 = ax2.imshow(fitr['pcov'])
        ax2.set_yticks(np.arange(0, 7))
        ax2.set_yticklabels(['amp', 'x0', 'y0', 'sigx', 'sigy', 'theta', 'offset'], rotation=0)
        
        divider = make_axes_locatable(ax2)
        cax2 = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im2, cax=cax2, orientation='vertical')
        
        # Adjust subplot:
        bbox1 = ax.get_position()
        subplot_ht = bbox1.height
        bbox2 = ax2.get_position()
        ax2.set_position([bbox2.x0, bbox1.y0, subplot_ht, subplot_ht])

    else:
        ax.text(0, -1, 'no fit')
        ax2.axis('off')
        
    pl.subplots_adjust(wspace=0.3, left=0.1, right=0.9)

    xi = np.arange(0, len(col_vals))
    yi = np.arange(0, len(row_vals))
    xx, yy = np.meshgrid(xi, yi)


    if fitr['success']:
        results = {'amplitude': amp_f,
                   'x0': x0_f,
                   'y0': y0_f,
                   'sigma_x': sigx_f,
                   'sigma_y': sigy_f,
                   'theta': theta_f,
                   'offset': offset_f,
                   'r2': fitr['r2'],
                   'fit_y': fit_y,
                   'fit_r': fitr,
                   'data': rfmap,
                   'denoised': denoised}
        
    
    return results, fig
    

def plot_kde_maxima(kde_results, weights, linX, linY, screen, use_peak=True, \
                    draw_bb=True, marker_scale=200, exclude_bad=False, min_thr=0.01):
        
    # Convert phase to linear coords:
    screen_left = -1*screen['azimuth']/2.
    screen_right = screen['azimuth']/2. #screen['azimuth']/2.
    screen_lower = -1*screen['elevation']/2.
    screen_upper = screen['elevation']/2. #screen['elevation']/2.

    fig = pl.figure(figsize=(10,6))
    ax = pl.subplot2grid((1, 2), (0, 0), colspan=2, fig=fig)
    
    if exclude_bad:
        bad_cells = np.array([i for i, w in enumerate(weights) if w < min_thr]) #weights[weights < min_thr].index.tolist()
        kept_cells = np.array([i for i in np.arange(len(weights)) if i not in bad_cells])
        linX = linX[kept_cells]
        linY = linY[kept_cells]
        mean_magratios = weights[kept_cells]
    else:
        kept_cells = np.arange(len(weights))
        mean_magratios = weights.copy()
    
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


    return fig, kept_cells

#%%

def plot_best_rfs(fit_roi_list, avg_resp_by_cond, fitdf, 
                  row_vals=[], col_vals=[], response_type='response',
                  sigma_scale=2.35, plot_ellipse=True, single_colorbar=True):
    #plot_ellipse = True
    #single_colorbar = True
    
    cbar_pad = 0.05 if not single_colorbar else 0.5
    
    cmap = 'magma' if plot_ellipse else 'inferno' # inferno
    cbar_mode = 'single' if single_colorbar else  'each'
    
    vmin = round(max([avg_resp_by_cond.min().min(), 0]), 1)
    vmax = round(min([5, avg_resp_by_cond.max().max()]), 1)
    
    nr = 6# 6 #6
    nc=10 #10 #10
    fig = pl.figure(figsize=(nc*2,nr+2))
    grid = AxesGrid(fig, 111,
                nrows_ncols=(nr, nc),
                axes_pad=0.5,
                cbar_mode=cbar_mode,
                cbar_location='right',
                cbar_pad=cbar_pad, cbar_size="3%")
    
    for aix, rid in enumerate(fit_roi_list[0:nr*nc]):
        ax = grid.axes_all[aix]
        ax.clear()
        coordmap = np.reshape(avg_resp_by_cond[rid], (len(col_vals), len(row_vals))).T
        
        im = ax.imshow(coordmap, cmap=cmap, vmin=vmin, vmax=vmax) #, vmin=vmin, vmax=vmax)
        #ax.contour(results['fit_params']['xx'], results['fit_params']['yy'], fitdf['fit'][rid].reshape(coordmap.shape), 1, colors='w')
        ax.set_title('roi %i (r2=%.2f)' % (int(rid+1), fitdf['r2'][rid]), fontsize=8)
        
        if plot_ellipse:
            # = Ellipse((x0_f, y0_f), abs(sigx_f)*sig_scale, abs(sigy_f)*sig_scale, angle=np.rad2deg(theta_f)) #theta_f)
    
            ell = Ellipse((fitdf['x0'][rid], fitdf['y0'][rid]), 
                          abs(fitdf['sigma_x'][rid])*sigma_scale, abs(fitdf['sigma_y'][rid])*sigma_scale, 
                          angle=np.rad2deg(fitdf['theta'][rid]))
            ell.set_alpha(0.5)
            ell.set_edgecolor('w')
            ell.set_facecolor('none')
            ax.add_patch(ell)
            
        if not single_colorbar:
            cbar = ax.cax.colorbar(im)
            cbar = grid.cbar_axes[aix].colorbar(im)
            cbar_yticks = [vmin, vmax] #[coordmap.min(), coordmap.max()]
            cbar.cbar_axis.axes.set_yticks(cbar_yticks)
            cbar.cbar_axis.axes.set_yticklabels([ cy for cy in cbar_yticks], fontsize=8)
        
        ax.set_ylim([0, len(row_vals)]) # This inverts y-axis so values go from positive to negative
        ax.set_xlim([0, len(col_vals)])
        #ax.invert_yaxis()
    
    if single_colorbar:
        cbar = ax.cax.colorbar(im)
        cbar = grid.cbar_axes[0].colorbar(im)
        cbar.ax.set_title(response_type)
    
    #%
    for a in np.arange(0, nr*nc):
        grid.axes_all[a].set_axis_off() 
    
    if not single_colorbar and len(fit_roi_list) < (nr*nc):
        for nix in np.arange(len(fit_roi_list), nr*nc):
            grid.cbar_axes[nix].remove()
    
    pl.subplots_adjust(left=0.05, right=0.95, wspace=0.3, hspace=0.3)
    
    return fig

def overlay_traces_on_rfmap(rid, avg_resp_by_cond, zscored_traces, labels, sdf, 
                            nframes_per_trial=89, vmin=None, vmax=None, scaley=None,
                            response_type='response', nframes_plot=45, yunit_sec=1.0, lw=1, 
                            cmap='bone', linecolor='darkslateblue', start_frame=0, row_vals=[], col_vals=[]):
    nr = len(row_vals)
    nc = len(col_vals)
    #nframes_on = labels['nframes_on'].unique()[0]
    #stim_on_frame = labels['stim_on_frame'].unique()[0]
    
    fig, axes = pl.subplots(nr, nc, figsize=(12, 6))

    coordmap = np.flipud(np.reshape(avg_resp_by_cond[rid], (len(col_vals), len(row_vals))).T) # Fipud to match screen
    vmin = coordmap.min() if vmin is None else vmin
    vmax = coordmap.max() if vmax is None else vmax
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    
    ymin=0; ymax=0;
    for ri, rval in enumerate(sorted(row_vals)): # Draw plots in reverse to match screen
        for ci, cval in enumerate(sorted(col_vals)[::-1]):
            #print rval, cval
            ax = axes[nr-ri-1, nc-ci-1]
            ax.clear()
            pcolor = cmapper.to_rgba(coordmap[nr-ri-1, nc-ci-1])
            ax.patch.set_color(pcolor)
            
            cfg = sdf[((sdf['xpos']==cval) & (sdf['ypos']==rval))].index[0] # Get current cfg
            fr_ixs = np.array(labels[labels['config']==cfg].index.tolist()) # Get corresponding trial indices
            ntrials_cond = len(labels[labels['config']==cfg]['trial'].unique())
            tmat = np.reshape(zscored_traces[rid][fr_ixs].values, (ntrials_cond, nframes_per_trial)) # reshape to get ntrials x nframes_in_trial
            avg_trace = np.nanmean(tmat, axis=0)[start_frame:start_frame+nframes_plot] # Get average across trials
            sem_trace = stats.sem(tmat, axis=0, nan_policy='omit')[start_frame:start_frame+nframes_plot] # Get sem across trials
            tsecs = np.nanmean(np.vstack(labels[labels['config']==cfg].groupby(['trial'])['tsec'].apply(np.array)), axis=0)
            if len(avg_trace) < nframes_plot: # Pad with nans
                avg_trace = np.pad(avg_trace, (0, nframes_plot-len(avg_trace)), mode='constant', constant_values=[np.nan])
                sem_trace = np.pad(sem_trace, (0, nframes_plot-len(sem_trace)), mode='constant', constant_values=[np.nan])
            if len(tsecs) < nframes_plot:
                tsecs = np.pad(tsecs, (0, nframes_plot-len(tsecs)), mode='constant', constant_values=[np.nan])
            elif len(tsecs) > nframes_plot:
                tsecs = tsecs[0:nframes_plot]
            tsecs = np.array(tsecs).astype('float')
            ax.plot(tsecs, avg_trace, linecolor, lw=lw)
            y1 = np.array(avg_trace + sem_trace)
            y2 = np.array(avg_trace - sem_trace)
            
            ax.fill_between(tsecs, y1, y2=y2, color=linecolor, alpha=0.4)
            ax.tick_params(axis='both', which='both', length=0, labelsize=0)
            ax.set_xticks([])
            ax.set_yticks([])
            ymin = min([ymin, np.nanmin(avg_trace)-np.nanmin(sem_trace)])
            ymax = max([ymax, np.nanmax(avg_trace) + np.nanmax(sem_trace)])

    ymin = round(ymin, 2) #if scaley is None else scaley[0] #round(min([ymin, 0]), 1)
    ymax = round(ymax, 2) #if scaley is None else scaley[1]
    for ax in axes.flat:
        ax.set_ylim([ymin, ymax])
        #for pos in ['top', 'bottom', 'right', 'left']:
        #    ax.spines[pos].set_edgecolor('w')

    pl.subplots_adjust(left=0.05, right=0.8, wspace=0, hspace=0)

    # Add colorbar for RF map
    cmapper._A = []
    cbar_ax = fig.add_axes([0.82, 0.15, 0.015, 0.5])
    cbar = fig.colorbar(cmapper, cax=cbar_ax)
    cbar.set_label('%s' % response_type)

    # Add legend for traces
    rect = [0.84, 0.78, .1, .1]
    leg = fig.add_subplot(111, position=rect, aspect='auto')
    leg.clear()
    leg.plot(tsecs, avg_trace, alpha=1)
     
    leg.set_ylim([ymin, ymax])
    if response_type == 'zscore':
        trace_scale = min([ymax, 2.0])
    else:
        trace_scale = min([ymax, 0.5])
    yscale = trace_scale #min([ymax, trace_scale])
    leg.set_yticks([ymin, yscale-np.abs(ymin)])
    yunits = 'std' if response_type=='zscore' else response_type
    leg.set_ylabel('%.1f %s' % (yscale, yunits))
    
    xmin = round(np.nanmin(tsecs), 1)
    xmax = round(np.nanmax(tsecs), 1)
    leg.set_xlim([xmin, xmax])
    leg.set_xticks([0, yunit_sec])
    leg.set_xticklabels([])
    leg.tick_params(axis='both', which='both', length=0, labelsize=0, pad=0.01)
    sns.despine(ax=leg, trim=True)
    leg.set_xlabel('%.1f s' % yunit_sec, horizontalalignment='left', x=0)

    return fig



def plot_rfs_to_screen(fitdf, sdf, screen, sigma_scale=2.35, fit_roi_list=[]):
    
    '''
    fitdf:  dataframe w/ converted fit params 
    '''
    row_vals = sorted(sdf[rows].unique())
    col_vals = sorted(sdf[cols].unique())

    #x_res = np.unique(np.diff(sdf['xpos'].unique()))[0]
    #y_res = np.unique(np.diff(sdf['ypos'].unique()))[0]
    
    #majors = np.array([abs(fitdf['sigma_x'][rid])*sigma_scale*x_res for rid in fit_roi_list])
    #minors = np.array([abs(fitdf['sigma_y'][rid])*sigma_scale*y_res for rid in fit_roi_list])
    
    majors = np.array([np.abs(fitdf['sigma_x'][rid]*sigma_scale) for rid in fit_roi_list])
    minors = np.array([np.abs(fitdf['sigma_y'][rid]*sigma_scale) for rid in fit_roi_list])

    print "Avg sigma-x, -y: %.2f" % majors.mean()
    print "Avg sigma-y: %.2f" % minors.mean()
    print "Average RF size: %.2f" % np.mean([majors.mean(), minors.mean()])
    
    avg_rfs = (majors + minors) / 2.
    
    screen_left = -1*screen['azimuth']/2.
    screen_right = screen['azimuth']/2.
    screen_top = screen['elevation']/2.
    screen_bottom = -1*screen['elevation']/2.
    
    
    fig, ax = pl.subplots(figsize=(12, 6))
    screen_rect = Rectangle(( min(col_vals), min(row_vals)), max(col_vals)-min(col_vals), 
                            max(row_vals)-min(row_vals), facecolor='none', edgecolor='k', lw=0.5)
    ax.add_patch(screen_rect)
    
    rcolors=iter(cm.rainbow(np.linspace(0,1,len(fit_roi_list))))
    for rid in fit_roi_list:
        rcolor = next(rcolors)
        #ax.plot(fitdf['x0'][rid], fitdf['y0'][rid], marker='*', color=rcolor)        
        #xx, yy, sigma_x, sigma_y = convert_fit_to_coords(fitdf, row_vals, col_vals, rid=rid)        
        ell = Ellipse((fitdf['x0'][rid], fitdf['y0'][rid]), 
                      abs(fitdf['sigma_x'][rid])*sigma_scale, abs(fitdf['sigma_y'][rid])*sigma_scale, 
                      angle=np.rad2deg(fitdf['theta'][rid]))

        ell.set_alpha(0.5)
        ell.set_edgecolor(rcolor)
        ell.set_facecolor('none')
        ax.add_patch(ell)
    #ax.invert_yaxis()
    
    ax.set_ylim([screen_bottom, screen_top])
    ax.set_xlim([screen_left, screen_right])
    
    summary_str = "Avg sigma-x, -y: (%.2f, %.2f)\nAvg RF size: %.2f (min: %.2f, max: %.2f)" % (np.mean(majors), np.mean(minors), np.mean([np.mean(majors), np.mean(minors)]), avg_rfs.min(), avg_rfs.max())
    pl.text(ax.get_xlim()[0]-12, ax.get_ylim()[0]-8, summary_str, ha='left', rotation=0, wrap=True)
    
    return fig
    



def plot_rfs_to_screen_pretty(fitdf, sdf, screen, sigma_scale=2.35, fit_roi_list=[]):
    
    '''
    fitdf:  dataframe w/ converted fit params 
    '''
    row_vals = sorted(sdf[rows].unique())
    col_vals = sorted(sdf[cols].unique())

    majors = np.array([np.abs(fitdf['sigma_x'][rid]*sigma_scale) for rid in fit_roi_list])
    minors = np.array([np.abs(fitdf['sigma_y'][rid]*sigma_scale) for rid in fit_roi_list])

    print "Avg sigma-x, -y: %.2f" % majors.mean()
    print "Avg sigma-y: %.2f" % minors.mean()
    print "Average RF size: %.2f" % np.mean([majors.mean(), minors.mean()])
    
    avg_rfs = (majors + minors) / 2.
    
    screen_left = -1*screen['azimuth']/2.
    screen_right = screen['azimuth']/2.
    screen_top = screen['elevation']/2.
    screen_bottom = -1*screen['elevation']/2.
    
    
    fig, ax = pl.subplots(figsize=(12, 6))

    
    rcolors=iter(cm.bone(np.linspace(0,1,len(fit_roi_list)+5)))
    for rid in fit_roi_list:
        rcolor = next(rcolors)
        #ax.plot(fitdf['x0'][rid], fitdf['y0'][rid], marker='*', color=rcolor)        
        #xx, yy, sigma_x, sigma_y = convert_fit_to_coords(fitdf, row_vals, col_vals, rid=rid)        
        ell = Ellipse((fitdf['x0'][rid], fitdf['y0'][rid]), 
                      abs(fitdf['sigma_x'][rid])*sigma_scale, abs(fitdf['sigma_y'][rid])*sigma_scale, 
                      angle=np.rad2deg(fitdf['theta'][rid]))

        ell.set_alpha(0.9)
        ell.set_edgecolor(rcolor)
        ell.set_facecolor('none')
        ax.add_patch(ell)
    #ax.invert_yaxis()
    
    ax.set_ylim([screen_bottom, screen_top])
    ax.set_xlim([screen_left, screen_right])
    ax.patch.set_color('gray')
    ax.patch.set_alpha(0.5)
    
    fig.patch.set_visible(False) #(False) #('off')
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
        
        
    #summary_str = "Avg sigma-x, -y: (%.2f, %.2f)\nAvg RF size: %.2f (min: %.2f, max: %.2f)" % (np.mean(majors), np.mean(minors), np.mean([np.mean(majors), np.mean(minors)]), avg_rfs.min(), avg_rfs.max())
    #pl.text(ax.get_xlim()[0]-12, ax.get_ylim()[0]-8, summary_str, ha='left', rotation=0, wrap=True)
    
    return fig
    


#%%

# #############################################################################
# FOV Targeting
# #############################################################################

def get_weighted_rf_density(xvals, yvals, weights, screen, weight_type='kde'):
    wtcoords = {}
    
    screen_left = -1*screen['azimuth']/2.
    screen_right = screen['azimuth']/2.
    screen_top = screen['elevation']/2.
    screen_bottom = -1*screen['elevation']/2.

    # Plot KDE:
    j = sns.jointplot(xvals, yvals, kind='kde', xlim=(screen_left, screen_right), ylim=(screen_bottom, screen_top))
    
    if weight_type == 'sns':
        # Seaborn's marginal distNs (unweighted)
        elev_x, elev_y = j.ax_marg_y.lines[0].get_data()
        azim_x, azim_y = j.ax_marg_x.lines[0].get_data()
        pl.close()
        
    elif weight_type == 'gaus':
        # smstats Gaussian KDE (unweighted):
        smstats_kde_az = sp.stats.gaussian_kde(xvals) #, weights=mean_fits)
        azim_x = np.linspace(screen_left, screen_right, len(xvals))
        azim_y = smstats_kde_az(azim_x)
        smstats_kde_el = sp.stats.gaussian_kde(yvals)
        elev_x = np.linspace(screen_bottom, screen_top, len(yvals))    
        elev_y = smstats_kde_el(elev_x)
    elif weight_type == 'kde':
        # 2. Use weights with KDEUnivariate (no FFT):
        weighted_kde_az = sm.nonparametric.kde.KDEUnivariate(xvals)
        weighted_kde_az.fit(weights=weights, fft=False)
        weighted_kde_el = sm.nonparametric.kde.KDEUnivariate(yvals)
        weighted_kde_el.fit(weights=weights, fft=False)
        
        elev_x = weighted_kde_el.support
        elev_y = weighted_kde_el.density
        azim_x = weighted_kde_az.support
        azim_y = weighted_kde_az.density
        
    wtcoords['elev_x'] = elev_x
    wtcoords['elev_y'] = elev_y
    wtcoords['azim_x'] = azim_x
    wtcoords['azim_y'] = azim_y

    return wtcoords, j


def target_screen_coords(wtcoords, screen, data_identifier='METADATA', target_fov_dir='/tmp', fit_thr=0.5):
    
    screen_left = -1*screen['azimuth']/2.
    screen_right = screen['azimuth']/2.
    screen_top = screen['elevation']/2.
    screen_bottom = -1*screen['elevation']/2.
    
    # Get targeting info from weights:
    az_max, az_min1, az_min2, az_maxima, az_minima = targ.find_local_min_max(wtcoords['azim_x'], wtcoords['azim_y'])
    el_max, el_min1, el_min2, el_maxima, el_minima = targ.find_local_min_max(wtcoords['elev_x'], wtcoords['elev_y'] )
    
    fig, axes = pl.subplots(1,2, figsize=(10,5)) #pl.figure();
    targ.plot_kde_min_max(wtcoords['azim_x'], wtcoords['azim_y'], maxval=az_max, minval1=az_min1, minval2=az_min2, title='azimuth', ax=axes[0])
    targ.plot_kde_min_max(wtcoords['elev_x'], wtcoords['elev_y'], maxval=el_max, minval1=el_min1, minval2=el_min2, title='elevation', ax=axes[1])
    
    label_figure(fig, data_identifier)
    fig.savefig(os.path.join(target_fov_dir, 'weighted_kde_min_max_fit_thr_%.2f.png' % (fit_thr)))
    pl.close()
    
    az_bounds = sorted([float(wtcoords['azim_x'][az_min1]), float(wtcoords['azim_x'][az_min2])])
    el_bounds = sorted([float(wtcoords['elev_x'][el_min1]), float(wtcoords['elev_x'][el_min2])])
    # Make sure bounds are within screen:
    if az_bounds[0] < screen_left:
        az_bounds[0] = screen_left
    if az_bounds[1] > screen_right:
        az_bounds[1] = screen_right
    if el_bounds[0] < screen_bottom:
        el_bounds[0] = screen_bottom
    if el_bounds[1] > screen_top:
        el_bounds[1] = screen_top
        
    kde_results = {'az_max': wtcoords['azim_x'][az_max],
                   'el_max': wtcoords['elev_x'][el_max],
                   'az_maxima': [wtcoords['azim_x'][azm] for azm in az_maxima],
                   'el_maxima': [wtcoords['elev_x'][elm] for elm in el_maxima],
                   'az_bounds': az_bounds,
                   'el_bounds': el_bounds,
                   'center_x': az_bounds[1] - (az_bounds[1]-az_bounds[0]) / 2.,
                   'center_y': el_bounds[1] - (el_bounds[1]-el_bounds[0]) / 2. }

    return kde_results


def compare_kde_weights(xvals, yvals, weights, screen):
    
    fig, axes = pl.subplots(1,2, figsize=(10,5))
    
    axes[0].set_title('azimuth')    
    wtcoords, j = get_weighted_rf_density(xvals, yvals, weights, screen, weight_type='kde')
    axes[0].plot(wtcoords['azim_x'], wtcoords['azim_y'], label='KDEuniv')
    axes[1].plot(wtcoords['azim_y'], wtcoords['azim_y'], label='KDEuniv')
    pl.close(j.fig)
    
    wtcoords, j = get_weighted_rf_density(xvals, yvals, weights, screen, weight_type='sns')
    axes[0].plot(wtcoords['azim_x'], wtcoords['azim_y'], label='sns-marginal (unweighted)')
    axes[1].plot(wtcoords['azim_y'], wtcoords['azim_y'], label='sns-marginal (unweighted)')
    pl.close(j.fig)
    
    wtcoords, j = get_weighted_rf_density(xvals, yvals, weights, screen, weight_type='gaus')
    axes[0].plot(wtcoords['azim_x'], wtcoords['azim_y'], label='smstats-gaus (unweighted)')
    axes[1].plot(wtcoords['azim_y'], wtcoords['azim_y'], label='smstats-gaus (unweighted)')
    pl.close(j.fig)

    axes[1].set_title('elevation')    
    axes[1].legend(fontsize=8)

    return fig

    
def plot_centroids(kde_results, weights, screen, xvals=[], yvals=[], fit_roi_list=[],
                   use_peak=True, exclude_bad=False, min_thr=0.01, marker_scale=100):
    
    #marker_scale = 100./round(magratio.mean().mean(), 3)
    fig, strong_cells = plot_kde_maxima(kde_results, weights, xvals, yvals, screen, \
                          use_peak=True, exclude_bad=False, min_thr=min_thr, marker_scale=100)
    
    for ri in strong_cells:
        fig.axes[0].text(xvals[ri], yvals[ri], '%s' % (fit_roi_list[ri]+1))
        
    return fig


def target_fov(avg_resp_by_cond, fitdf, screen, fit_roi_list=[], row_vals=[], col_vals=[], 
               compare_kdes=False, weight_type='kde', fit_thr=0.5, target_fov_dir='/tmp',
               data_identifier='METADATA'):
    
    compare_kdes = False

    max_zscores = avg_resp_by_cond.max(axis=0)
    #xx, yy, sigma_x, sigma_y = convert_fit_to_coords(fitdf, row_vals, col_vals)
    xx = fitdf['x0']
    yy = fitdf['y0']
    
    xvals = np.array([xx[rid] for rid in fit_roi_list])
    yvals = np.array([yy[rid] for rid in fit_roi_list])
    weights = np.array([max_zscores[rid] for rid in fit_roi_list])

    wtcoords, j = get_weighted_rf_density(xvals, yvals, weights, screen, weight_type='kde')

    # Plot weighted KDE to marginals on joint plot:
    j.ax_marg_y.plot(wtcoords['elev_y'], wtcoords['elev_x'], color='orange', label='weighted')
    j.ax_marg_x.plot(wtcoords['azim_x'], wtcoords['azim_y'], color='orange', label='weighted')
    j.ax_marg_x.set_ylim([0, max([j.ax_marg_x.get_ylim()[-1], wtcoords['azim_y'].max()]) + 0.005])
    j.ax_marg_y.set_xlim([0, max([j.ax_marg_y.get_xlim()[-1], wtcoords['elev_y'].max()]) + 0.005])
    j.ax_marg_x.legend(fontsize=8)
    
    j.savefig(os.path.join(target_fov_dir, 'weighted_marginals_fit_thr_%.2f.png' % (fit_thr) ))
    pl.close()
    
    if compare_kdes:
        
        fig = compare_kde_weights(xvals, yvals, weights, screen)
        label_figure(fig, data_identifier)
        pl.savefig(os.path.join(target_fov_dir, 'compare_kde_weighted_fit_thr_%.2f.png' % (fit_thr) ))
        pl.close(fig)      

    #%

    kde_results = target_screen_coords(wtcoords, screen, data_identifier=data_identifier, target_fov_dir=target_fov_dir)
    with open(os.path.join(target_fov_dir, 'RESULTS_target_fov_fit_thr_%.2f.json' % (fit_thr)), 'w') as f:
        json.dump(kde_results, f, sort_keys=True, indent=4)

    
    print("AZIMUTH bounds: %s" % str(kde_results['az_bounds']))
    print("ELEV bounds: %s" % str(kde_results['el_bounds']))
    print("CENTER: %.2f, %.2f" % (kde_results['center_x'], kde_results['center_y']))

    
    #%
    fig = plot_centroids(kde_results, weights, screen, xvals=xvals, yvals=yvals,
                         fit_roi_list=fit_roi_list, use_peak=True, exclude_bad=False, min_thr=0.01, marker_scale=100)
    
    label_figure(fig, data_identifier)
    
    pl.savefig(os.path.join(target_fov_dir, 'centroid_peak_rois_by_pos_fit_thr_%.2f.png' % (fit_thr)))
    pl.close()
    
    return kde_results

    #%%

def fit_rfs(avg_resp_by_cond, row_vals=[], col_vals=[], response_type='dff', roi_list=None,
            rf_results_fpath='/tmp/fit_results.pkl', data_identifier='METADATA', 
            response_thr=None, create_new=False):

    trim=False; hard_cutoff=False; map_thr=''; set_to_min=False; 

    '''
    hard_cutoff:
        (bool) Use hard cut-off for zscores (set to False to use some % of max value)
    set_to_min: 
        (bool) Threshold x,y condition grid and set non-passing conditions to min value or 0 
    '''

    #%
    rfdir = os.path.split(rf_results_fpath)[0]
    
    #if do_fits:
    # Create subdir for saving each roi's fit
    if not os.path.exists(os.path.join(rfdir, 'roi_fits')):
        os.makedirs(os.path.join(rfdir, 'roi_fits'))


    #sigma_scale = 2.35   # Value to scale sigma in order to get FW (instead of FWHM)
    if roi_list is None:
        if response_thr == None:
            roi_list = avg_resp_by_cond.columns.tolist()
            response_thr = 0
        else:    
            roi_list = [r for r in avg_resp_by_cond.columns.tolist() if avg_resp_by_cond[r].max() >= response_thr]
            print("%i out of %i cells meet min req. of %.2f" % (len(roi_list), avg_resp_by_cond.shape[1], response_thr))
         
    RF = {}
    results = {}
    for rid in roi_list:
        #print rid
        roi_fit_results, fig = plot_and_fit_roi_RF(avg_resp_by_cond[rid], row_vals, col_vals) 
#                                                       trim=trim,
#                                                       hard_cutoff=hard_cutoff, map_thr=map_thr, 
#                                                       set_to_min=set_to_min, perc_min=perc_min)
        fig.suptitle('roi %i' % int(rid+1))
        
        label_figure(fig, data_identifier)            
        figname = '%s_%s_RF_roi%05d' % (trace_type, response_type, int(rid+1))
        pl.savefig(os.path.join(rfdir, 'roi_fits', '%s.png' % figname))
        pl.close()
        
        if roi_fit_results != {}:
            RF[rid] = roi_fit_results
        #%


    xi = np.arange(0, len(col_vals))
    yi = np.arange(0, len(row_vals))
    xx, yy = np.meshgrid(xi, yi)

    if len(RF.keys())>0:
        results = {'fit_results': RF, #RF
                   'fit_params': {'rfmap_thr': map_thr,
                                  'cut_off': hard_cutoff,
                                  'set_to_min': set_to_min,
                                  'trim': trim,
                                  'xx': xx,
                                  'yy': yy,
                                  'metric': response_type},
                   'row_vals': row_vals,
                   'col_vals': col_vals}
                   
    with open(rf_results_fpath, 'wb') as f:
        pkl.dump(results, f, protocol=pkl.HIGHEST_PROTOCOL)

    return results

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
                      help="acquisition folder (ex: 'FOV1_zoom3x') [default: FOV1]")
    parser.add_option('-R', '--run', action='store', dest='run', default='retino_run1', \
                      help="name of run dir containing tiffs to be processed (ex: gratings_phasemod_run1)")
    parser.add_option('-t', '--traceid', action='store', dest='traceid', default='traces001', \
                      help="name of traces ID [default: traces001]")
    parser.add_option('-d', '--data-type', action='store', dest='trace_type', default='corrected', \
                      help="Trace type to use for analysis [default: corrected]")
    
    parser.add_option('--segment', action='store_true', dest='segment', default=False, \
                      help="Set flag to use segmentation of FOV for select visual area")
    parser.add_option('-V', '--area', action='store', dest='visual_area', default='', \
                      help="Name of visual area, if --segment is flagged")
    parser.add_option('--select-rois', action='store_true', dest='select_rois', default=False, \
                      help="Set flag to select only visual or selective rois")
    
    parser.add_option('-T', '--thr', action='store', dest='response_thr', default=None, \
                      help="Min snr or zscore value for cells to fit (default: None (fits all))")
    parser.add_option('-M', '--resp', action='store', dest='response_type', default='dff', \
                      help="Response metric to use for creating RF maps (default: dff)")
    parser.add_option('--plot', action='store_true', dest='plot_rois', default=False, \
                      help="Flag to make pretty plots for roi fits")
    parser.add_option('--new', action='store_true', dest='create_new', default=False, \
                      help="Flag to refit all rois")


    (options, args) = parser.parse_args(options)

    return options



#%%

def get_fit_desc(response_type='dff'):
    fit_desc = 'fit-2dgaus_%s-no-cutoff' % response_type
    return fit_desc


def create_rf_dir(animalid, session, fov, run_name, 
               traceid='traces001', response_type='dff', fit_thr=0.5,
               rootdir='/n/coxfs01/2p-data'):
    

    # Get RF dir for current fit type
    fit_desc = get_fit_desc(response_type=response_type)

    fov_dir = os.path.join(rootdir, animalid, session, fov)
    traceid_dirs = [t for t in glob.glob(os.path.join(fov_dir, '*%s*' % run_name, 'traces', '%s*' % traceid)) if 'combined' in t]
    if len(traceid_dirs) > 1:
        print "More than 1 trace ID found:"
        for ti, traceid_dir in enumerate(traceid_dirs):
            print ti, traceid_dir
        sel = input("Select IDX of traceid to use: ")
        traceid_dir = traceid_dirs[int(sel)]
    else:
        traceid_dir = traceid_dirs[0]
    #traceid = os.path.split(traceid_dir)[-1]
         
    rfdir = os.path.join(traceid_dir, 'receptive_fields', fit_desc)
    if not os.path.exists(rfdir):
        os.makedirs(rfdir)

    return rfdir, fit_desc

#%%




def get_rf_to_fov_info(masks, rfdf, zimg, rfdir='/tmp', rfname='rfs',
                       plot_spatial=False, transform_fov=True, create_new=False):    
    '''
    Get FOV info relating cortical position to RF position of all cells.
    Info is saved in: rfdir/fov_info.pkl
    
    Returns:
        fovinfo (dict)
            'positions': 
                dataframe of azimuth (xpos) and elevation (ypos) for cell's 
                cortical position and cell's RF position (i.e., 'posdf')
            'zimg': 
                (array) z-projected image 
            'roi_contours': 
                (list) roi contours created from classifications.convert_coords.contours_from_masks()
            'xlim' and 'ylim': 
                (float) FOV limits (in pixels or um) for azimuth and elevation axes
    '''
    
    get_fovinfo = create_new
    fovinfo_fpath = os.path.join(rfdir, 'fov_info.pkl')
    if os.path.exists(fovinfo_fpath) and create_new is False:
        print("... loading fov info")
        try:
            with open(fovinfo_fpath, 'rb') as f:
                fovinfo = pkl.load(f)
            assert 'zimg' in fovinfo.keys(), "Redoing rf-to-fov conversion"
        except Exception as e:
            get_fovinfo = True
    else:
        get_fovinfo = True
            
    if get_fovinfo:
        print("... getting fov info")
        # Get masks
        npix_y, npix_x, nrois_total = masks.shape
        
        # Create contours from maskL
        roi_contours = coor.contours_from_masks(masks)
        
        # Convert to brain coords
        rfdf['cell'] = rfdf.index.tolist()
        fov_x, rf_x, xlim, fov_y, rf_y, ylim = coor.get_roi_position_um(rfdf, roi_contours, 
                                                                         rf_exp_name=rfname,
                                                                         convert_um=True,
                                                                         npix_y=npix_y,
                                                                         npix_x=npix_x)
        posdf = pd.DataFrame({'xpos_fov': fov_y,
                               'xpos_rf': rf_x,
                               'ypos_fov': fov_x,
                               'ypos_rf': rf_y}, index=rfdf.index)
    
        # Save fov info
        fovinfo = {'zimg': zimg,
                    'roi_contours': roi_contours,
                    'positions': posdf,
                    'xlim': xlim,
                    'ylim': ylim}

        with open(fovinfo_fpath, 'wb') as f:
            pkl.dump(fovinfo, f, protocol=pkl.HIGHEST_PROTOCOL)

    return fovinfo   

   #%%     
def fit_2d_receptive_fields(animalid, session, fov, run, traceid, create_new=False,
                            trace_type='corrected', response_type='dff', make_pretty_plots=False,
                            visual_area='', select_rois=False, segment=False,
                            fit_thr=0.5, rootdir='/n/coxfs01/2p-data'):

    rows = 'ypos'
    cols = 'xpos'

    
    # Set output dirs
    # -----------------------------------------------------------------------------
    # rfdir = os.path.join(traceid_dir, 'figures', 'receptive_fields')
    # rf_param_str = 'fit-2dgaus_%s-no-cutoff' % (response_type) #, response_thr) #, cutoff_type, set_to_min_str)
    rfdir, fit_desc = create_rf_dir(animalid, session, fov, run, traceid=traceid,
                                    response_type=response_type, fit_thr=fit_thr)

    traceid_dir = rfdir.split('/receptive_fields/')[0]
    #print(traceid_dir)
    data_fpath = os.path.join(traceid_dir, 'data_arrays', 'np_subtracted.npz')

    if not os.path.exists(data_fpath):
        # Realign traces
        print("*****corrected offset unfound, running now*****")
        print("%s | %s | %s | %s | %s" % (animalid, session, fov, run, traceid))
        if 'combined' in run:
            run_name = run.split('_')[1]
        else:
            run_name = run
        aggregate_experiment_runs(animalid, session, fov, run_name, traceid=traceid)
        print("*****corrected offsets!*****")
                        
        
    dset = np.load(data_fpath)
    #dset.keys()
    
    
    if segment:
        rfdir = os.path.join(rfdir, visual_area)
    if not os.path.exists(rfdir):
        os.makedirs(rfdir)
    #print "Saving output to:", rfdir
    
    #%
    # Create subdir for saving figs/results based on fit params
    # -----------------------------------------------------------------------------
    data_identifier = '|'.join([animalid, session, fov, run, traceid, visual_area, fit_desc])
    
    # Create results outfile, or load existing:
    do_fits = False
    rf_results_fpath = os.path.join(rfdir, 'fit_results.pkl') #results_outfile = 'RESULTS_%s.pkl' % fit_desc
    if os.path.exists(rf_results_fpath) or create_new is False:
        try:
            print "... checking for existing results"
            with open(rf_results_fpath, 'rb') as f:
                results = pkl.load(f) 
            print "... loaded RF fit results"
        except Exception as e:
            traceback.print_exc()
            do_fits = True
    else:
        do_fits = True
        
    #%%
    
    if do_fits or make_pretty_plots:
        
        raw_traces, labels, gdf, sdf = load_data(data_fpath, add_offset=True, make_equal=False)

        print("--- [%s|%s|%s|%s]: fitting receptive fields." % (animalid, session, fov, run))        #%%
        # Load parsed data:
        #F0 = np.nanmean(dset['corrected'][:] / dset['dff'][:] )
        #print("offset: %.2f" % F0)
        #raw_traces = pd.DataFrame(dset['corrected']) + F0 #pd.DataFrame(dset[trace_type]) #, index=zscored_traces.index)
        #labels = pd.DataFrame(data=dset['labels_data'], columns=dset['labels_columns'])
    
        # Format condition info:
        #sdf = pd.DataFrame(dset['sconfigs'][()]).T
        if 'image' in sdf['stimtype']:
            aspect_ratio = sdf['aspect'].unique()[0]
            sdf['size'] = [round(sz/aspect_ratio, 1) for sz in sdf['size']]
        row_vals = sorted(sdf[rows].unique())
        col_vals = sorted(sdf[cols].unique())

        fr = 44.65 #dset['run_info'][()]['framerate']
        nframes_per_trial = int(dset['run_info'][()]['nframes_per_trial'][0])
        nframes_on = labels['nframes_on'].unique()[0]
        stim_on_frame = labels['stim_on_frame'].unique()[0]
        
        
        #%
        #response_type = 'dff' #'None'
        #response_thr = None
        
        # zscore the traces:
        nframes_post_onset = nframes_on + int(round(1.*fr))
        trials_by_cond = get_trials_by_cond(labels)
        zscored_traces, zscores = process_traces(raw_traces, labels, response_type=response_type,
                                                nframes_post_onset=nframes_post_onset)
        avg_resp_by_cond = group_trial_values_by_cond(zscores, trials_by_cond)

    
        #%
        if do_fits:
            results = fit_rfs(avg_resp_by_cond, response_type=response_type, row_vals=row_vals, col_vals=col_vals,
                              rf_results_fpath=rf_results_fpath, data_identifier=data_identifier, create_new=create_new)
        
        #%%
        row_vals = results['row_vals']
        col_vals = results['col_vals']
        fitdf = rfits_to_df(results['fit_results'], row_vals=row_vals, col_vals=col_vals) #, roi_list=None)
        fit_thr = 0.5
        fit_roi_list = fitdf[fitdf['r2'] > fit_thr].sort_values('r2', axis=0, ascending=False).index.tolist()
        print "%i out of %i fit rois with r2 > %.2f" % (len(fit_roi_list), fitdf.shape[0], fit_thr)
    #    
        # Plot all RF maps for fit cells (limit = 60 to plot)
        
        #rf_results_dir = os.path.split(rf_results_fpath)[0]
        #rf_param_str = os.path.split(rf_results_dir)[-1]
        try:
            fitdf_pos = pd.DataFrame(results['fit_results']).T
            fig = plot_best_rfs(fit_roi_list, avg_resp_by_cond, fitdf_pos, 
                                row_vals=row_vals, col_vals=col_vals, response_type=response_type,
                                plot_ellipse=True, single_colorbar=True)
            
            label_figure(fig, data_identifier)
            figname = 'top%i_fit_thr_%.2f_%s_ellipse' % (len(fit_roi_list), fit_thr, fit_desc)
            pl.savefig(os.path.join(rfdir, '%s.png' % figname))
            print figname
            pl.close()
        except Exception as e:
            traceback.print_exc()
            print("Error plotting all RF maps that pass thr.")
        
        #%%
        #make_pretty_plots= False
        if make_pretty_plots:
            zscored_traces_plot, zscores_plot = process_traces(raw_traces, labels, response_type='zscore',
                                                nframes_post_onset=nframes_post_onset)
            avg_resp_by_cond_plot = group_trial_values_by_cond(zscores, trials_by_cond)


            #%
            # Overlay RF map and mean traces:
            # -----------------------------------------------------------------------------
            linecolor = 'darkslateblue' #'purple'
            cmap = 'bone'
            nframes_plot = stim_on_frame + nframes_on + int(np.floor(fr*1.5))
            start_frame=0
            yunit_sec = round(nframes_on/fr, 1)
                
            best_rois_figdir = os.path.join(rfdir, 'best_rfs')
            # Plot overlay?
            if not os.path.exists(os.path.join(best_rois_figdir, 'svg')):
                os.makedirs(os.path.join(best_rois_figdir, 'svg'))
              
            for rid in fit_roi_list:
                fig = overlay_traces_on_rfmap(rid, avg_resp_by_cond_plot, zscored_traces_plot, labels, sdf,
                                              #vmin=-0.05, vmax=0.5, scaley=[-0.05, 0.6], lw=0.5,
                                              nframes_per_trial=nframes_per_trial, response_type='zscore', #response_type,
                                              nframes_plot=nframes_plot, start_frame=start_frame, yunit_sec=yunit_sec,
                                              row_vals=row_vals, col_vals=col_vals, linecolor=linecolor, cmap=cmap)
                
                label_figure(fig, data_identifier)
                fig.suptitle('roi %i' % int(rid+1))
            
                figname = 'roi%05d-overlay_zscore' % (int(rid+1)) #, fit_thr, response_type)
                pl.savefig(os.path.join(best_rois_figdir, 'svg', '%s.svg' % figname))
                #pl.savefig(os.path.join(best_rois_figdir, '%s.png' % figname))

                pl.close()
                
        #%%
        #fitdf = rfits_to_df(results['fit_results'], row_vals=results['row_vals'], col_vals=results['col_vals']) #, roi_list=None)

        try:
            screen = rutils.get_screen_info(animalid, session, rootdir=rootdir)
        
            fig = plot_rfs_to_screen(fitdf, sdf, screen, fit_roi_list=fit_roi_list)
            label_figure(fig, data_identifier)
            #%
            figname = 'overlaid_RFs_top%i_fit_thr_%.2f_%s' % (len(fit_roi_list), fit_thr, fit_desc)
            pl.savefig(os.path.join(rfdir, '%s.png' % figname))
            print figname
            pl.close()
        except Exception as e:
            traceback.print_exc()
            print("Error plotting RFs to screen coords.")
            
        try:
            screen = rutils.get_screen_info(animalid, session, rootdir=rootdir)
        
            fig = plot_rfs_to_screen_pretty(fitdf, sdf, screen, fit_roi_list=fit_roi_list)
            label_figure(fig, data_identifier)
            #%
            figname = 'overlaid_RFs_pretty' 
            pl.savefig(os.path.join(rfdir, '%s.svg' % figname))
            print figname
            pl.close()
        except Exception as e:
            traceback.print_exc()
            print("Error plotting RFs to screen coords.")
            
            #%%
        try:
            # Identify VF area to target:
            target_fov_dir = os.path.join(rfdir, 'target_fov')
            if not os.path.exists(target_fov_dir):
                os.makedirs(target_fov_dir)
                
            kde_results = target_fov(avg_resp_by_cond, fitdf, screen, fit_roi_list=fit_roi_list, 
                                     row_vals=row_vals, col_vals=col_vals, compare_kdes=False,
                                     weight_type='kde', target_fov_dir=target_fov_dir, 
                                     data_identifier=data_identifier)
        except Exception as e:
            traceback.print_exc()
            print("Error finding target coords for FOV.")
            

    fitdf = rfits_to_df(results['fit_results'], row_vals=results['row_vals'], col_vals=results['col_vals']) #, roi_list=None)
    fit_thr = 0.5
    #fit_roi_list = fitdf[fitdf['r2'] >= fit_thr].sort_values('r2', axis=0, ascending=False).index.tolist()
    #print "%i out of %i fit rois with r2 > %.2f" % (len(fit_roi_list), fitdf.shape[0], fit_thr)
#    
    if int(session) < 20190511:
        rois = get_roiid_from_traceid(animalid, session, fov, run_type='gratings', traceid=traceid)
    else:
        rois = get_roiid_from_traceid(animalid, session, fov, run_type='rfs', traceid=traceid)
    masks, zimg = load_roi_masks(animalid, session, fov, rois=rois, rootdir=rootdir)
    fovinfo = get_rf_to_fov_info(masks, fitdf, zimg, rfdir=rfdir, create_new=create_new)
  
    return results, fovinfo

#%%


rootdir = '/n/coxfs01/2p-data'
animalid = 'JC111' #'JC097' #'JC084' #'JC059'
session = '20191003' #'20190623' #'20190522' #'20190227'
fov = 'FOV1_zoom2p0x' #'FOV4_zoom4p0x'
run = 'combined_rfs10_static'
traceid = 'traces001' #'traces001'
segment = False
#visual_area = 'V1'
trace_type = 'corrected'
select_rois = False
rows = 'ypos'
cols = 'xpos'

options = ['-i', animalid, '-S', session, '-A', fov, '-R', run, '-t', traceid]
#if segment:
#    options.extend(['--segment', '-V', visual_area])


#%%%
def main(options):

    optsE = extract_options(options)
    
    rootdir = optsE.rootdir
    animalid = optsE.animalid
    session = optsE.session
    fov = optsE.fov
    run = optsE.run
    traceid = optsE.traceid
    trace_type = optsE.trace_type
    
    segment = optsE.segment
    visual_area = optsE.visual_area
    select_rois = optsE.select_rois
    
    response_type = optsE.response_type
    response_thr = optsE.response_thr
    plot_rois = optsE.plot_rois
    create_new= optsE.create_new

    fit_thr = 0.50 
    rfresults, fovinfo = fit_2d_receptive_fields(animalid, session, fov, run, traceid, 
                                trace_type=trace_type, visual_area=visual_area, select_rois=select_rois,
                                #response_type=response_type, response_thr=response_thr, 
                                make_pretty_plots=plot_rois, create_new=create_new)

    print("--- fit %i rois (R2 thr > %.2f) ---" % (len(rfresults['fit_results'].keys()), fit_thr))

    #%

    print "((( RFs done! )))))"
        
        
if __name__ == '__main__':
    main(sys.argv[1:])


    
