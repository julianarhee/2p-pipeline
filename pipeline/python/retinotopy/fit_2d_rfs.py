#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 15:25:28 2019

@author: julianarhee
"""

#%%
import os
import glob
import json
import copy
import optparse
import sys
import traceback
import cv2
from scipy import interpolate
import math

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
import matplotlib.gridspec as gridspec
    
import scipy.optimize as opt
from matplotlib.patches import Ellipse, Rectangle

from mpl_toolkits.axes_grid1 import AxesGrid
from pipeline.python.utils import natural_keys, convert_range, label_figure, load_dataset, load_run_info, get_screen_dims, warp_spherical, get_spherical_coords, get_lin_coords
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


#%% Data formating for working with fit params (custom functions)
def rfits_to_df(fitr, row_vals=[], col_vals=[], roi_list=None, fit_params={},
                scale_sigma=True, sigma_scale=2.35, convert_coords=True, spherical=False):
    '''
    Takes each roi's RF fit results, converts to screen units, and return as dataframe.
    Scale to make size FWFM if scale_sigma is True.
    '''
    if roi_list is None:
        roi_list = sorted(fitr.keys())
       
    sigma_scale = sigma_scale if scale_sigma else 1.0

    fitdf = pd.DataFrame({'x0': [fitr[r]['x0'] for r in roi_list],
                          'y0': [fitr[r]['y0'] for r in roi_list],
                          'sigma_x': [fitr[r]['sigma_x'] for r in roi_list],
                          'sigma_y': [fitr[r]['sigma_y'] for r in roi_list],
                          'theta': [fitr[r]['theta'] % (2*np.pi) for r in roi_list],
                          'r2': [fitr[r]['r2'] for r in roi_list]},
                              index=roi_list)

    if convert_coords:
        if spherical:
            fitdf = convert_fit_to_coords_spherical(fitdf, fit_params, spherical=spherical)
        else:
            x0, y0, sigma_x, sigma_y = convert_fit_to_coords(fitdf, row_vals, col_vals)
            fitdf['x0'] = x0
            fitdf['y0'] = y0
            fitdf['sigma_x'] = sigma_x * sigma_scale
            fitdf['sigma_y'] = sigma_y * sigma_scale

    return fitdf

def apply_scaling_to_df(row, grid_points=None, new_values=None):
    #r2 = row['r2']
    #theta = row['theta']
    #offset = row['offset']
    x0, y0, sx, sy = get_scaled_sigmas(grid_points, new_values,
                                             row['x0'], row['y0'], 
                                             row['sigma_x'], row['sigma_y'], row['theta'],
                                             convert=True)
    return x0, y0, sx, sy #sx, sy, x0, y0


def convert_fit_to_coords_spherical(fitdf, fit_params, sigma_scale=2.35, spherical=True):
    grid_points, cart_values, sphr_values = coordinates_for_transformation(fit_params)
    
    if spherical:
        converted = fitdf.apply(apply_scaling_to_df, args=(grid_points, sphr_values), axis=1)
    else:
        converted = fitdf.apply(apply_scaling_to_df, args=(grid_points, cart_values), axis=1)
    newdf = pd.DataFrame([[x0, y0, sx*sigma_scale, sy*sigma_scale] 
                          for x0, y0, sx, sy in converted.values], 
                             index=converted.index, 
                             columns=['x0', 'y0', 'sigma_x', 'sigma_y'])
    fitdf[['sigma_x', 'sigma_y', 'x0', 'y0']] = newdf[['sigma_x', 'sigma_y', 'x0', 'y0']]

    return fitdf


def convert_fit_to_coords(fitdf, row_vals, col_vals, rid=None):
    
    if rid is not None:
        xx = convert_range(fitdf['x0'][rid], 
                            newmin=min(col_vals), newmax=max(col_vals), 
                            oldmax=len(col_vals)-1, oldmin=0)
        
        sigma_x = convert_range(abs(fitdf['sigma_x'][rid]), 
                            newmin=0, newmax=max(col_vals)-min(col_vals), 
                            oldmax=len(col_vals)-1, oldmin=0)
        
        yy = convert_range(fitdf['y0'][rid], 
                            newmin=min(row_vals), newmax=max(row_vals), 
                            oldmax=len(row_vals)-1, oldmin=0)
        
        sigma_y = convert_range(abs(fitdf['sigma_y'][rid]), 
                            newmin=0, newmax=max(row_vals)-min(row_vals), 
                            oldmax=len(row_vals)-1, oldmin=0)
    else:
        xx = convert_range(fitdf['x0'], 
                            newmin=min(col_vals), newmax=max(col_vals), 
                            oldmax=len(col_vals)-1, oldmin=0)
        
        sigma_x = convert_range(abs(fitdf['sigma_x']), 
                            newmin=0, newmax=max(col_vals)-min(col_vals), 
                            oldmax=len(col_vals)-1, oldmin=0)
        
        yy = convert_range(fitdf['y0'], 
                            newmin=min(row_vals), newmax=max(row_vals), 
                            oldmax=len(row_vals)-1, oldmin=0)
        
        sigma_y = convert_range(abs(fitdf['sigma_y']), 
                            newmin=0, newmax=max(row_vals)-min(row_vals), 
                            oldmax=len(row_vals)-1, oldmin=0)
    
    return xx, yy, sigma_x, sigma_y

def ang(lineA, lineB):
    # Get nicer vector form
    vA = [(lineA[0][0]-lineA[1][0]), (lineA[0][1]-lineA[1][1])]
    vB = [(lineB[0][0]-lineB[1][0]), (lineB[0][1]-lineB[1][1])]
    # Get dot prod
    dot_prod = np.dot(vA, vB)
    # Get magnitudes
    magA = np.dot(vA, vA)**0.5
    magB = np.dot(vB, vB)**0.5
    # Get cosine value
    cos_ = dot_prod/magA/magB
    # Get angle in radians and then convert to degrees
    angle = math.acos(dot_prod/magB/magA)
    # Basically doing angle <- angle mod 360
    ang_deg = math.degrees(angle)%360.

    if ang_deg-180>=0:
        # As in if statement
        return round(360 - ang_deg, 2)
    else: 
        return round(ang_deg, 2)


def get_endpoints_from_sigma(x0, y0, sx, sy, th, scale_sigma=False, sigma_scale=2.35):
    
    sx = sx*sigma_scale if scale_sigma else sx
    sy = sy*sigma_scale if scale_sigma else sy
    
    sx_x1, sx_y1 = (x0-(sx/2.)*np.cos(th), y0-(sx/2.)*np.sin(th)) # Get min half
    sx_x2, sx_y2 = (x0+(sx/2.)*np.cos(th), y0+(sx/2.)*np.sin(th)) # Get other half

    th_orth = th + (np.pi/2.)
    sy_x1, sy_y1 = (x0-(sy/2.)*np.cos(th_orth), y0-(sy/2.)*np.sin(th_orth))
    sy_x2, sy_y2 = (x0+(sy/2.)*np.cos(th_orth), y0+(sy/2.)*np.sin(th_orth))

    lA = (sy_x1, sy_y1), (sy_x2, sy_y2)
    lB = (sx_x1, sx_y1), (sx_x2, sx_y2)
    ang_deg = ang(lA, lB)
    assert ang_deg==90.0, "bad angle calculation (%.1f)..." % ang_deg

    return (sx_x1, sx_y1), (sx_x2, sx_y2), (sy_x1, sy_y1), (sy_x2, sy_y2)


def coordinates_for_transformation(fit_params):
    ds_factor = fit_params['downsample_factor']
    col_vals = fit_params['col_vals']
    row_vals = fit_params['row_vals']
    nx = len(col_vals)
    ny = len(row_vals)

    # Downsample screen resolution
    resolution_ds = [int(i/ds_factor) for i in fit_params['screen']['resolution'][::-1]]

    # Get linear coordinates in degrees (downsampled)
    lin_x, lin_y = get_lin_coords(resolution=resolution_ds, cm_to_deg=True) 
    print("Screen res (ds=%ix): [%i, %i]" % (ds_factor, resolution_ds[0], resolution_ds[1]))

    # Get Spherical coordinate mapping
    cart_x, cart_y, sphr_x, sphr_y = get_spherical_coords(cart_pointsX=lin_x, 
                                                            cart_pointsY=lin_y,
                                                            cm_to_degrees=False) # already in deg

    screen_bounds_pix = get_screen_lim_pixels(lin_x, lin_y, 
                                            row_vals=row_vals, col_vals=col_vals)
    (pix_bottom_edge, pix_left_edge, pix_top_edge, pix_right_edge) = screen_bounds_pix
 
    # Trim and downsample coordinate space to match corrected map
    cart_x_ds  = cv2.resize(cart_x[pix_top_edge:pix_bottom_edge, pix_left_edge:pix_right_edge], 
                            (nx, ny))
    cart_y_ds  = cv2.resize(cart_y[pix_top_edge:pix_bottom_edge, pix_left_edge:pix_right_edge], 
                            (nx, ny))

    sphr_x_ds  = cv2.resize(sphr_x[pix_top_edge:pix_bottom_edge, pix_left_edge:pix_right_edge], 
                            (nx,ny))
    sphr_y_ds  = cv2.resize(sphr_y[pix_top_edge:pix_bottom_edge, pix_left_edge:pix_right_edge], 
                            (nx, ny))

    grid_x, grid_y = np.meshgrid(range(nx),range(ny)[::-1])
    grid_points = np.array( (grid_x.flatten(), grid_y.flatten()) ).T
    cart_values = np.array( (cart_x_ds.flatten(), cart_y_ds.flatten()) ).T
    sphr_values = np.array( (np.rad2deg(sphr_x_ds).flatten(), np.rad2deg(sphr_y_ds).flatten()) ).T
    return grid_points, cart_values, sphr_values

def get_scaled_sigmas(grid_points, new_values, x0, y0, sx, sy, th, convert=True):

    x0_scaled, y0_scaled = interpolate.griddata(grid_points, new_values, (x0, y0))
    x0_scaled, y0_scaled = interpolate.griddata(grid_points, new_values, (x0, y0))

    # Get flanking points spanned by sx, sy
    sx_linel, sx_line2, sy_line1, sy_line2 = get_endpoints_from_sigma(x0, y0, sx, sy, th, 
                                                                        scale_sigma=False)

    # Get distances
    if convert:
        # Convert coordinates of array to new coordinate system
        sx_x1_sc, sx_y1_sc = interpolate.griddata(grid_points, new_values, sx_linel) 
        sx_x2_sc, sx_y2_sc = interpolate.griddata(grid_points, new_values, sx_line2)
        sx_scaled = math.hypot(sx_x2_sc - sx_x1_sc, sx_y2_sc - sx_y1_sc)
    else:
        #sx_scaled = math.hypot(sx_x2 - sx_x1, sx_y2 - sx_y1)
        sx_scaled = math.hypot(sx_line2[0] - sx_linel[0], sx_line2[1] - sx_linel[1])

    if convert:
        sy_x1_sc, sy_y1_sc = interpolate.griddata(grid_points, new_values, sy_line1)
        sy_x2_sc, sy_y2_sc = interpolate.griddata(grid_points, new_values, sy_line2)
        sy_scaled = math.hypot(sy_x2_sc - sy_x1_sc, sy_y2_sc - sy_y1_sc)
    else:
        #sy_scaled = math.hypot(sy_x2 - sy_x1, sy_y2 - sy_y1)
        sy_scaled = math.hypot(sy_line2[0] - sy_line1[0], sy_line2[1] - sy_line1[1])
    
    return x0_scaled, y0_scaled, abs(sx_scaled), abs(sy_scaled)



def get_screen_lim_pixels(lin_coord_x, lin_coord_y, row_vals=None, col_vals=None):
    
    #pix_per_deg=16.050716 pix_per_deg = screeninfo['pix_per_deg']
    stim_size = float(np.unique(np.diff(row_vals)))

    right_lim = max(col_vals) + (stim_size/2.)
    left_lim = min(col_vals) - (stim_size/2.)
    top_lim = max(row_vals) + (stim_size/2.)
    bottom_lim = min(row_vals) - (stim_size/2.)

    # Get actual stimulated locs in pixels
    i_x, i_y = np.where( np.abs(lin_coord_x-right_lim) == np.abs(lin_coord_x-right_lim).min() )
    pix_right_edge = int(np.unique(i_y))

    i_x, i_y = np.where( np.abs(lin_coord_x-left_lim) == np.abs(lin_coord_x-left_lim).min() )
    pix_left_edge = int(np.unique(i_y))
    #print("AZ bounds (pixels): ", pix_right_edge, pix_left_edge)

    i_x, i_y = np.where( np.abs(lin_coord_y-top_lim) == np.abs(lin_coord_y-top_lim).min() )
    pix_top_edge = int(np.unique(i_x))

    i_x, i_y = np.where( np.abs(lin_coord_y-bottom_lim) == np.abs(lin_coord_y-bottom_lim).min() )
    pix_bottom_edge = int(np.unique(i_x))
    #print("EL bounds (pixels): ", pix_top_edge, pix_bottom_edge)

    # Check expected tile size
    #ncols = len(col_vals); nrows = len(row_vals);
    #expected_sz_x = (pix_right_edge-pix_left_edge+1) * (1./pix_per_deg) / ncols
    #expected_sz_y = (pix_bottom_edge-pix_top_edge+1) * (1./pix_per_deg) / nrows
    #print("tile sz-x, -y should be ~(%.2f, %.2f) deg" % (expected_sz_x, expected_sz_y))
    
    return (pix_bottom_edge, pix_left_edge, pix_top_edge,  pix_right_edge)


def resample_map(rfmap, lin_coord_x, lin_coord_y, row_vals=None, col_vals=None,
                 resolution=(1080,1920)):

    screen_bounds_pix=get_screen_lim_pixels(lin_coord_x, lin_coord_y, 
                                            row_vals=row_vals, col_vals=col_vals)
    (pix_bottom_edge, pix_left_edge, pix_top_edge, pix_right_edge) = screen_bounds_pix

    stim_height = pix_bottom_edge-pix_top_edge+1
    stim_width = pix_right_edge-pix_left_edge+1
    stim_resolution = [stim_height, stim_width]

    # Upsample rfmap to match resolution of stimulus-occupied space
    rfmap_r = cv2.resize(rfmap.astype(float), (stim_resolution[1], stim_resolution[0]), 
                          interpolation=cv2.INTER_NEAREST)

    rfmap_to_screen = np.ones((resolution[0], resolution[1]))*np.nan
    rfmap_to_screen[pix_top_edge:pix_bottom_edge+1, pix_left_edge:pix_right_edge+1] = rfmap_r

    return rfmap_to_screen


def warp_spherical_fromarr(rfmap_values, cart_x=None, cart_y=None, sphr_th=None, sphr_ph=None, 
                            row_vals=None, col_vals=None, resolution=(1080, 1920),
                            normalize_range=True):  
    nx = len(col_vals)
    ny = len(row_vals)
    rfmap = rfmap_values.reshape(ny, nx) #rfmap_values.reshape(nx, ny).T

    # Upsample to screen resolution (pixels)
    rfmap_orig = resample_map(rfmap, cart_x, cart_y, #lin_coord_x, lin_coord_y, 
                            row_vals=row_vals, col_vals=col_vals,
                            resolution=resolution) 
    # Warp upsampled
    rfmap_warp = warp_spherical(rfmap_orig, sphr_th, sphr_ph, cart_x, cart_y,
                                normalize_range=normalize_range, method='linear')

    # Crop
    screen_bounds_pix = get_screen_lim_pixels(cart_x, cart_y, 
                                              row_vals=row_vals, col_vals=col_vals)
    (pix_bottom_edge, pix_left_edge, pix_top_edge, pix_right_edge) = screen_bounds_pix
    rfmap_trim = rfmap_warp[pix_top_edge:pix_bottom_edge, pix_left_edge:pix_right_edge]

    # Resize back to known grid
    rfmap_resize = cv2.resize(rfmap_trim, (nx, ny))
 
    return rfmap_resize.flatten()

def reshape_array_for_nynx(rfmap_values, nx, ny):
    rfmap_orig = rfmap_values.reshape(nx, ny).T
    return rfmap_orig.ravel()



def sphr_correct_maps(avg_resp_by_cond, fit_params=None, multiproc=True):

    if multiproc:
        avg_resp_by_cond = avg_resp_by_cond.T

    ds_factor = fit_params['downsample_factor']
    col_vals = fit_params['col_vals']
    row_vals = fit_params['row_vals']

    # Downsample screen resolution
    resolution_ds = [int(i/ds_factor) for i in fit_params['screen']['resolution'][::-1]]

    # Get linear coordinates in degrees (downsampled)
    lin_x, lin_y = get_lin_coords(resolution=resolution_ds, cm_to_deg=True) 
    print("Screen res (ds=%ix): [%i, %i]" % (ds_factor, resolution_ds[0], resolution_ds[1]))

    # Get Spherical coordinate mapping
    cart_x, cart_y, sphr_th, sphr_ph = get_spherical_coords(cart_pointsX=lin_x, 
                                                            cart_pointsY=lin_y,
                                                            cm_to_degrees=False) # already in deg

    args=(cart_x, cart_y, sphr_th, sphr_ph, row_vals, col_vals, resolution_ds,)
    avg_t = avg_resp_by_cond.apply(warp_spherical_fromarr, axis=0, args=args)

    return avg_t.reset_index(drop=True)

from functools import partial
import multiprocessing as mp

def sphr_correct_maps_mp(avg_resp_by_cond, fit_params, n_processes=2, test_subset=False):
    
    if test_subset:
        roi_list=[92, 249, 91, 162, 61, 202, 32, 339]
        df_ = avg_resp_by_cond[roi_list]
    else:
        df_ = avg_resp_by_cond.copy()
    print("Parallel", df_.shape)

    df = parallelize_dataframe(df_.T, sphr_correct_maps, fit_params, n_processes=n_processes)

    return df

def initializer(terminating_):
    # This places terminating in the global namespace of the worker subprocesses.
    # This allows the worker function to access `terminating` even though it is
    # not passed as an argument to the function.
    global terminating
    terminating = terminating_

def parallelize_dataframe(df, func, fit_params, n_processes=4):
    #cart_x=None, cart_y=None, sphr_th=None, sphr_ph=None,
    #                      row_vals=None, col_vals=None, resolution=None, n_processes=4):
    results = []
    terminating = mp.Event()
    
    df_split = np.array_split(df, n_processes)
    pool = mp.Pool(processes=n_processes, initializer=initializer, initargs=(terminating,))
    try:
        results = pool.map(partial(func, fit_params=fit_params), df_split)
        print("done!")
    except KeyboardInterrupt:
        pool.terminate()
        print("terminating")
    finally:
        pool.close()
        pool.join()
  
    print(results[0].shape)
    df = pd.concat(results, axis=1)
    print(df.shape)
    return df #results

#%%
# -----------------------------------------------------------------------------
# Data grouping by condition 
# -----------------------------------------------------------------------------
def get_trials_by_cond(labels):
    # Get single value for each trial and sort by config:
    trials_by_cond = dict()
    for k, g in labels.groupby(['config']):
        trials_by_cond[k] = sorted([int(tr[5:])-1 for tr in g['trial'].unique()])

    return trials_by_cond

def group_trial_values_by_cond(zscores, trials_by_cond, do_spherical_correction=False, nx=21, ny=11):
    resp_by_cond = dict()
    for cfg, trial_ixs in trials_by_cond.items():
        resp_by_cond[cfg] = zscores.iloc[trial_ixs]  # For each config, array of size ntrials x nrois

    trialvalues_by_cond = pd.DataFrame([resp_by_cond[cfg].mean(axis=0) \
                                            for cfg in sorted(resp_by_cond.keys(), key=natural_keys)]) # nconfigs x nrois
    #if do_spherical_correction:
    avg_t = trialvalues_by_cond.apply(reshape_array_for_nynx, args=(nx, ny))
    trialvalues_by_cond = avg_t.copy()
            
    return trialvalues_by_cond


def process_traces(raw_traces, labels, response_type='zscore', nframes_post_onset=None):        
    print("--- processed traces: %s" % response_type)
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
        stim_mean = curr_traces.iloc[stim_on_frame:(stim_on_frame+nframes_on+nframes_post_onset)].mean(axis=0)
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

    res = a + b * np.exp( -( ((x-x0)*np.cos(theta) + (y-y0)*np.sin(theta)) / (np.sqrt(2)*sigma_x) )**2 - ( ( -(x-x0)*np.sin(theta) + (y-y0)*np.cos(theta) ) / (np.sqrt(2)*sigma_y) )**2 )
    
    return res.ravel()

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
    #print(roots)
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
    
    hm = ((ynew.max() - ynew.min()) / 2 ) + ynew.min() # half-max
    pk = ynew.argmax() # find peak

    if pk == 0:
        r2 = pk + np.abs(ynew[pk:] - hm).argmin()
        return abs(xnew[r2]*2)
    else:
        r1 = np.abs(ynew[0:pk] - hm).argmin() # find index of local min on left
        r2 = pk + np.abs(ynew[pk:] - hm).argmin() # find index of local min on right
        
        return abs(xnew[r2]-xnew[r1]) # return full width
    


def get_rf_map(response_vector, ncols, nrows, do_spherical_correction=False):
    #if do_spherical_correction:
    coordmap_r = np.reshape(response_vector, (nrows, ncols))
    #else:
    #    coordmap_r = np.reshape(response_vector, (ncols, nrows)).T
    
    return coordmap_r

def plot_rf_map(rfmap, cmap='inferno', ax=None):
    if ax is None:
        fig, ax = pl.subplots()
    im = ax.imshow(rfmap, cmap='inferno')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    ax.figure.colorbar(im, cax=cax, orientation='vertical')
   
    return ax

def plot_rf_ellipse(fitr, ax=None, sigma_scale=2.35, scale_sigma=True):
      
    sigma_scale = sigma_scale if scale_sigma else 1.0
    
    # Draw ellipse: #ax.contour(plot_xx, plot_yy, fitr.reshape(rfmap.shape), 3, colors='b')
    amp_f, x0_f, y0_f, sigx_f, sigy_f, theta_f, offset_f = fitr['popt']
    
    if ax is None:
        fig, ax = pl.subplots()
        ax.set_ylim([y0_f-sigy_f*2., y0_f+sigy_f*2.])
        ax.set_xlim([x0_f-sigx_f*2., x0_f+sigx_f*2.])
 
    ell = Ellipse((x0_f, y0_f), abs(sigx_f)*sigma_scale, abs(sigy_f)*sigma_scale, 
                    angle=np.rad2deg(theta_f), alpha=0.5, edgecolor='w') #theta_f)
    ax.add_patch(ell)
    ax.text(0, -1, 'r2=%.2f, theta=%.2f' % (fitr['r2'], theta_f), color='k')

    return ax


def plot_roi_RF(response_vector, ncols, nrows, ax=None, trim=False, cmap='inferno',
                hard_cutoff=False, set_to_min=True, map_thr=2.0, perc_min=0.1):
    
    if ax is None:
        fig, ax = pl.subplots()
        
    #if do_spherical_correction:
    coordmap_r = np.reshape(response_vector, (nrows, ncols))
    #else:
    #    coordmap_r = np.reshape(response_vector, (ncols, nrows)).T
     
    rfmap = coordmap_r.copy()
#    if trim:
#        if hard_cutoff:
#            rfmap[coordmap_r < map_thr] = coordmap_r.min()*perc_min if set_to_min else 0
#        else:
#            rfmap[coordmap_r <= (coordmap_r.max()*map_thr)] = coordmap_r.min()*perc_min if set_to_min else 0
#        
    ax = plot_rf_map(coordmap_r, cmap=cmap, ax=ax)
    
    return ax, rfmap
#%%

def do_2d_fit(rfmap, nx=None, ny=None, verbose=False):

    #TODO:  Instead of finding critical pts w/ squared RF map, do:
    #    mean subtraction, followed by finding max delta from the  ____
    #nx=len(col_vals); ny=len(row_vals);
    # Set params for fit:
    xi = np.arange(0, nx)
    yi = np.arange(0, ny)
    popt=None; pcov=None; fitr=None; r2=None; success=None;
    xx, yy = np.meshgrid(xi, yi)
    initial_guess = None
    try:
        #amplitude = (rfmap**2).max()
        #y0, x0 = np.where(rfmap == rfmap.max())
        #y0, x0 = np.where(rfmap**2. == (rfmap**2.).max())
        #print "x0, y0: (%i, %i)" % (int(x0), int(y0))    

        rfmap_sub = np.abs(rfmap - np.nanmean(rfmap))
        y0, x0 = np.where(rfmap_sub == np.nanmax(rfmap_sub))
        amplitude = rfmap[y0, x0][0]
        #print "x0, y0: (%i, %i) | %.2f" % (int(x0), int(y0), amplitude)    
        try:
            #sigma_x = fwhm(xi, (rfmap**2).sum(axis=0))
            #sigma_x = fwhm(xi, abs(rfmap.sum(axis=0) - rfmap.sum(axis=0).mean()) )
            sigma_x = fwhm(xi, np.nansum(rfmap_sub, axis=0) )
            assert sigma_x is not None
        except AssertionError:
            #sigma_x = raw_fwhm(rfmap.sum(axis=0)) 
            sigma_x = raw_fwhm( np.nansum(rfmap_sub, axis=0) ) 
        try:
            sigma_y = fwhm(yi, np.nansum(rfmap_sub, axis=1))
            assert sigma_y is not None
        except AssertionError: #Exception as e:
            sigma_y = raw_fwhm(np.nansum(rfmap_sub, axis=1))
        #print "sig-X, sig-Y:", sigma_x, sigma_y
        theta = 0
        offset=0
        initial_guess = (amplitude, int(x0), int(y0), sigma_x, sigma_y, theta, offset)
        valid_ixs = ~np.isnan(rfmap)
        popt, pcov = opt.curve_fit(twoD_gauss, (xx[valid_ixs], yy[valid_ixs]), rfmap[valid_ixs].ravel(), 
                                   p0=initial_guess, maxfev=2000)
        fitr = twoD_gauss((xx, yy), *popt)

        # Get residual sum of squares 
        residuals = rfmap.ravel() - fitr
        ss_res = np.nansum(residuals**2)
        ss_tot = np.nansum((rfmap.ravel() - np.nanmean(rfmap.ravel()))**2)
        r2 = 1 - (ss_res / ss_tot)
        #print(r2)
        if len(np.where(fitr > fitr.min())[0]) < 2 or pcov.max() == np.inf or r2 == 1: 
            success = False
        else:
            success = True
            # modulo theta
            #mod_theta = popt[5] % np.pi
            #popt[5] = mod_theta
            
    except Exception as e:
        if verbose:
            print e
    
    return {'popt': popt, 'pcov': pcov, 'init': initial_guess, 'r2': r2, 'success': success}, fitr

#%%
# -----------------------------------------------------------------------------
# PLOTTING FUNCTIONS:
# -----------------------------------------------------------------------------


def plot_and_fit_roi_RF(response_vector, row_vals, col_vals, 
                        min_sigma=2.5, max_sigma=50, sigma_scale=2.35, scale_sigma=True,
                        trim=False, hard_cutoff=False, map_thr=None, set_to_min=False, perc_min=None):
    '''
    Fits RF for single ROI. 
    Note: This does not filter by R2, includes all fit-able.
    
    Returns a dict with fit info if doesn't error out.
    
    Sigma must be [2.5, 50]...
    '''
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
    sigma_scale = sigma_scale if scale_sigma else 1.0
    results = {}
    fig, axes = pl.subplots(1,2, figsize=(8, 4)) # pl.figure()
    ax = axes[0]
    #ax, rfmap = plot_roi_RF(avg_zscores_by_cond[rid], ncols=len(col_vals), nrows=len(row_vals), ax=ax)
    ax, rfmap = plot_roi_RF(response_vector, ax=ax, 
                            ncols=len(col_vals), nrows=len(row_vals))
#                            , trim=trim,
#                            perc_min=perc_min,
#                            hard_cutoff=False, map_thr=map_thr, set_to_min=set_to_min)
#
    ax2 = axes[1]
    # Do fit 
    # ---------------------------------------------------------------------
    denoised=False
    #if hard_cutoff and (rfmap.max() < map_thr):
    #    fitr = {'success': False}
    #else:
    fitr, fit_y = do_2d_fit(rfmap, nx=len(col_vals), ny=len(row_vals))
#        if rfmap.max() > 3.0 and fit_y is None:
#            try:
#                rfmap[rfmap<rfmap.max()*0.2] = rfmap.min()
#                fitr, fit_y = do_2d_fit(rfmap, nx=len(col_vals), ny=len(row_vals))
#                assert fitr is not None, "--- no fit, trying with denoised..."
#                denoised=True
#            except Exception as e:
#                print e
#                pass
#            
    xres = np.mean(np.diff(sorted(row_vals)))
    yres = np.mean(np.diff(sorted(col_vals)))
    min_sigma = xres/2.0
    
    if fitr['success']:
        amp_f, x0_f, y0_f, sigx_f, sigy_f, theta_f, offset_f = fitr['popt']
        if any(s < min_sigma for s in [abs(sigx_f)*xres*sigma_scale, abs(sigy_f)*yres*sigma_scale])\
            or any(s > max_sigma for s in [abs(sigx_f)*xres*sigma_scale, abs(sigy_f)*yres*sigma_scale]):
            fitr['success'] = False

    if fitr['success']:    
        # Draw ellipse: #ax.contour(plot_xx, plot_yy, fitr.reshape(rfmap.shape), 3, colors='b')
#        amp_f, x0_f, y0_f, sigx_f, sigy_f, theta_f, offset_f = fitr['popt']
#        ell = Ellipse((x0_f, y0_f), abs(sigx_f)*sigma_scale, abs(sigy_f)*sigma_scale, 
#                      angle=np.rad2deg(theta_f), alpha=0.5, edgecolor='w') #theta_f)
#        ax.add_patch(ell)
#        ax.text(0, -1, 'r2=%.2f, theta=%.2f' % (fitr['r2'], theta_f), color='k')
        ax = plot_rf_ellipse(fitr, ax, scale_sigma=scale_sigma)
                
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
    


#%%
def plot_kde_maxima(kde_results, weights, linX, linY, screen, use_peak=True, \
                    draw_bb=True, marker_scale=200, exclude_bad=False, min_thr=0.01):
        
    # Convert phase to linear coords:
    screen_left = -1*screen['azimuth_deg']/2.
    screen_right = screen['azimuth_deg']/2. #screen['azimuth']/2.
    screen_lower = -1*screen['altitude_deg']/2.
    screen_upper = screen['altitude_deg']/2. #screen['elevation']/2.

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
        
    print('%s x, y: (%f, %f)' % (centroid_type, cgx, cgy))
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

#%% Plotting p

def plot_best_rfs(fit_roi_list, avg_resp_by_cond, fitdf, fit_params,
                    plot_ellipse=True, single_colorbar=True, nr=6, nc=10):
    #plot_ellipse = True
    #single_colorbar = True
    
    row_vals = fit_params['row_vals']
    col_vals = fit_params['col_vals']
    response_type = fit_params['response_type']
    sigma_scale = fit_params['sigma_scale'] if fit_params['scale_sigma'] else 1.0
     
    cbar_pad = 0.05 if not single_colorbar else 0.5
    
    cmap = 'magma' if plot_ellipse else 'inferno' # inferno
    cbar_mode = 'single' if single_colorbar else  'each'
    
    vmin = round(max([avg_resp_by_cond.min().min(), 0]), 1)
    vmax = round(min([.5, avg_resp_by_cond.max().max()]), 1)
   
    nx = len(col_vals)
    ny = len(row_vals)
 
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
        coordmap = avg_resp_by_cond[rid].reshape(ny, nx) 
        #coordmap = np.reshape(avg_resp_by_cond[rid], (len(col_vals), len(row_vals))).T
        
        im = ax.imshow(coordmap, cmap=cmap, vmin=vmin, vmax=vmax) #, vmin=vmin, vmax=vmax)
        ax.set_title('roi %i (r2=%.2f)' % (int(rid+1), fitdf['r2'][rid]), fontsize=8)
        
        if plot_ellipse:    
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


#%%
def get_trial_averaged_timecourse(rid, traces, labels, run_info, config='config001', 
                                  nframes_plot=None, start_frame=0):
    
    fr_ixs = np.array(labels[labels['config']==config].index.tolist()) # Get corresponding trial indices
    ntrials_cond = len(labels[labels['config']==config]['trial'].unique())
    nframes_trial = run_info['nframes_per_trial'][0]
    ntrials_cond = run_info['ntrials_by_cond'][config] 
    if nframes_plot is None:
        nframes_plot = nframes_trial # plot whole trial
    tmat = np.reshape(traces[rid][fr_ixs].values, (ntrials_cond, nframes_trial)) # reshape to get ntrials x nframes_in_trial
    avg_trace = np.nanmean(tmat, axis=0)[start_frame:start_frame+nframes_plot] # Get average across trials
    sem_trace = stats.sem(tmat, axis=0, nan_policy='omit')[start_frame:start_frame+nframes_plot] # Get sem across trials
    tsecs = np.nanmean(np.vstack(labels[labels['config']==config].groupby(['trial'])['tsec'].apply(np.array)), axis=0)
    if len(avg_trace) < nframes_plot: # Pad with nans
        avg_trace = np.pad(avg_trace, (0, nframes_plot-len(avg_trace)), mode='constant', constant_values=[np.nan])
        sem_trace = np.pad(sem_trace, (0, nframes_plot-len(sem_trace)), mode='constant', constant_values=[np.nan])
    if len(tsecs) < nframes_plot:
        tsecs = np.pad(tsecs, (0, nframes_plot-len(tsecs)), mode='constant', constant_values=[np.nan])
    elif len(tsecs) > nframes_plot:
        tsecs = tsecs[0:nframes_plot]
    tsecs = np.array(tsecs).astype('float')
    
    return tsecs, avg_trace, sem_trace

#%%
def overlay_traces_on_rfmap(rid, avg_resp_by_cond, zscored_traces, labels, sdf, run_info,
                            fitdf=None, response_type='response', start_frame=0, nframes_plot=45, 
                            yunit_sec=1.0, scaley=None, vmin=None, vmax=None, 
                            lw=1, legend_lw=2., cmap='bone', linecolor='darkslateblue', 
                            row_vals=[], col_vals=[], 
                            plot_ellipse=True, scale_sigma=True, sigma_scale=2.35, 
                            ellipse_ec='w', ellipse_fc='none', ellipse_lw=1, ellipse_alpha=1.0): 
                            #screen_ylim=[-33.6615, 33.6615], screen_xlim=[-59.7782, 59.7782]):
   
    nr = len(row_vals)
    nc = len(col_vals)
       
    rfmap = np.flipud(avg_resp_by_cond[rid].reshape(len(row_vals), len(col_vals))) # Fipud to match screen
    #    rfmap = np.flipud(np.reshape(avg_resp_by_cond[rid], (len(col_vals), len(row_vals))).T) # Fipud to match screen
    vmin = np.nanmin(rfmap) if vmin is None else vmin
    vmax = np.nanmax(rfmap) if vmax is None else vmax
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmapper = cm.ScalarMappable(norm=norm, cmap=cmap)
   
    fig = pl.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(nr, nc)
    all_axes = [] 
    ymin=0; ymax=0; k=0;
    for ri, rval in enumerate(sorted(row_vals)): # Draw plots in reverse to match screen
        for ci, cval in enumerate(sorted(col_vals)):
            map_ix_row = nr-ri-1 #row_vals.index(rval)
            map_ix_col = col_vals.index(cval)
            ax = fig.add_subplot(gs[map_ix_row, map_ix_col]) #nr-ri-1, nc-ci-1]) #gs[k])
            ax.clear()
            #pcolor = cmapper.to_rgba(coordmap[nr-ri-1, nc-ci-1])
            pcolor = cmapper.to_rgba(rfmap[nr-ri-1, ci])
            ax.patch.set_color(pcolor)
            k+=1
           
            cfg = sdf[((sdf['xpos']==cval) & (sdf['ypos']==rval))].index[0] # Get current cfg
            tsecs, avg_trace, sem_trace = get_trial_averaged_timecourse(rid, 
                                                        zscored_traces, labels, 
                                                        run_info, config=cfg, 
                                                        nframes_plot=nframes_plot, start_frame=start_frame)
            ax.plot(tsecs, avg_trace, linecolor, lw=lw)
       
            # Fill sem trace 
            y1 = np.array(avg_trace + sem_trace)
            y2 = np.array(avg_trace - sem_trace)
            ax.fill_between(tsecs, y1, y2=y2, color=linecolor, alpha=0.4)
            
            # Hide tick stuff
            ymin = min([ymin, np.nanmin(avg_trace)-np.nanmin(sem_trace)])
            ymax = max([ymax, np.nanmax(avg_trace) + np.nanmax(sem_trace)])
            all_axes.append(ax)
    
    #ymin = round(ymin, 2) #if scaley is None else scaley[0] #round(min([ymin, 0]), 1)
    #ymax = round(ymax, 2) #if scaley is None else scaley[1]
    if scaley is not None:
        subplot_ylims = (round(ymin, 2), round(scaley, 2)) 
    else:
        subplot_ylims = (round(ymin, 2), round(ymax, 2))
    subplot_xlims = ax.get_xlim() #(round(np.nanmin(tsecs), 1), round(np.nanmax(tsecs), 1))    
    #print("setting ymin/ymax: %.2f, %.2f" % (subplot_ylims[0], subplot_ylims[1]))
    #print("setting xmin/xmax: %.2f, %.2f" % (subplot_xlims[0], subplot_xlims[1]))
    ax.set_ylim(subplot_ylims)
    ax.set_xlim(subplot_xlims)
    subplot_pos = ax.get_position()
    for ax in all_axes: #axes.flat:
        ax.set_ylim(subplot_ylims)
        ax.set_xlim(subplot_xlims)
        ax.tick_params(axis='both', which='both', length=0, labelsize=0)
        ax.set_xticks([])
        ax.set_yticks([])
        curr_pos = ax.get_position()
        new_pos = [curr_pos.x0, curr_pos.y0, subplot_pos.width, subplot_pos.height] #.1, .1]
        ax.set_position(new_pos)

    #subplot_xlims = ax.get_xlim()
    #subplot_ylims = ax.get_ylim()
    #subplot_pos = ax.get_position()
   
     
    # Reduce spacing between subplots
    pl.subplots_adjust(left=0.05, right=0.8, wspace=0, hspace=0)

    if plot_ellipse:
        pos_ylim = (min(row_vals), max(row_vals))
        pos_xlim = (min(col_vals), max(col_vals))
        screen_xlim_centered = get_centered_screen_points(pos_xlim, nc)
        screen_ylim_centered = get_centered_screen_points(pos_ylim, nr)
                  
        outergs = gridspec.GridSpec(1,1)
        outerax = fig.add_subplot(outergs[0])
        outerax.tick_params(axis='both', which='both', bottom=0, left=0, 
                        labelbottom=0, labelleft=0)
        outerax.set_facecolor('crimson')
        outerax.patch.set_alpha(0.1)
        outerax = fig.add_subplot(outergs[0])
        outerax.set_ylim(screen_ylim_centered) #[screen_bottom, screen_top])
        outerax.set_xlim(screen_xlim_centered) #[screen_left, screen_right])
 
        ell = Ellipse((fitdf['x0'][rid], fitdf['y0'][rid]), 
                        abs(fitdf['sigma_x'][rid])*sigma_scale, 
                        abs(fitdf['sigma_y'][rid])*sigma_scale, 
                        angle=np.rad2deg(fitdf['theta'][rid]))
        ell.set_alpha(ellipse_alpha)
        ell.set_edgecolor(ellipse_ec)
        ell.set_facecolor(ellipse_fc)
        ell.set_linewidth(ellipse_lw)
        outerax.add_patch(ell) 

    # Add colorbar for RF map
    cmapper._A = []
    cbar_ax = fig.add_axes([0.82, 0.15, 0.015, 0.3])
    cbar = fig.colorbar(cmapper, cax=cbar_ax)
    cbar.set_label('%s' % response_type)
    cbar.set_ticks([vmin, vmax])
    cbar.set_ticklabels([round(vmin, 2), round(vmax, 2)])
    cbar.ax.tick_params(axis='both', which='both', length=0, pad=1)

    # Add legend for traces
    legend_pos = [0.84, 0.78, subplot_pos.width, subplot_pos.height] #.1, .1]
    leg = fig.add_subplot(111, position=legend_pos) #, aspect='auto')
    leg.clear()
    leg.plot(tsecs, avg_trace, 'w', alpha=1)
    leg.set_ylim(subplot_ylims)
    leg.set_xlim(subplot_xlims)
    yscale = min([ymax, 2.0]) if response_type=='zscore' else min([ymax, 0.5])
    #yscale = trace_scale #min([ymax, trace_scale])
    ymin, ymax = subplot_ylims
    leg.set_yticks([ymin, ymin+yscale])
    #print(ymin, yscale, yscale+ymin)
    #leg.set_yticks([0, 0.1])
    yunits = 'std' if response_type=='zscore' else response_type
    leg.set_ylabel('%.1f %s' % (yscale, yunits))
    
    xmin, xmax = subplot_xlims 
    leg.set_xticks([xmin, xmin+yunit_sec])
    leg.set_xticklabels([])
    sns.despine(ax=leg, trim=True, offset=0)

    leg.tick_params(axis='both', which='both', length=0, labelsize=0, pad=0.01)
    for axlabel in ['left', 'bottom']:
        leg.spines[axlabel].set_linewidth(legend_lw)
    leg.set_xlabel('%.1f s' % yunit_sec, horizontalalignment='left', x=0)

    return fig

#%%
        

 
#%%
def get_centered_screen_points(screen_xlim, nc):
    col_pts = np.linspace(screen_xlim[0], screen_xlim[1], nc+1) # n points for NC columns
    pt_spacing = np.mean(np.diff(col_pts)) 
    # Add half point spacing on either side of boundary points to center the column points
    xlim_min = col_pts.min() - (pt_spacing/2.) 
    xlim_max = col_pts.max() + (pt_spacing/2.)
    return (xlim_min, xlim_max)



#%%

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

    print "Avg sigma-x, -y: %.2f, %.2f (avg sz, %.2f)" % (majors.mean(), minors.mean(), 
                                                                np.mean([majors.mean(), minors.mean()]))
    
    avg_rfs = (majors + minors) / 2.
    
    screen_left = -1*screen['azimuth_deg']/2.
    screen_right = screen['azimuth_deg']/2.
    screen_top = screen['altitude_deg']/2.
    screen_bottom = -1*screen['altitude_deg']/2.
    
    
    fig, ax = pl.subplots(figsize=(12, 6))
    screen_rect = Rectangle(( min(col_vals), min(row_vals)), max(col_vals)-min(col_vals), 
                            max(row_vals)-min(row_vals), facecolor='none', edgecolor='k', lw=0.5)
    ax.add_patch(screen_rect)
    
    rcolors=iter(cm.rainbow(np.linspace(0,1,len(fit_roi_list))))
    for rid in fit_roi_list:
        rcolor = next(rcolors)
        #ax.plot(fitdf['x0'][rid], fitdf['y0'][rid], marker='*', color=rcolor)        
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

    #print "Avg sigma-x, -y: %.2f" % majors.mean()
    #print "Avg sigma-y: %.2f" % minors.mean()
    #print "Average RF size: %.2f" % np.mean([majors.mean(), minors.mean()])
    
    avg_rfs = (majors + minors) / 2.
    
    screen_left = -1*screen['azimuth_deg']/2.
    screen_right = screen['azimuth_deg']/2.
    screen_top = screen['altitude_deg']/2.
    screen_bottom = -1*screen['altitude_deg']/2.
    
    
    fig, ax = pl.subplots(figsize=(12, 6))

    
    rcolors=iter(cm.bone(np.linspace(0,1,len(fit_roi_list)+5)))
    for rid in fit_roi_list:
        rcolor = next(rcolors)
        #ax.plot(fitdf['x0'][rid], fitdf['y0'][rid], marker='*', color=rcolor)        
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
    
    screen_left = -1*screen['azimuth_deg']/2.
    screen_right = screen['azimuth_deg']/2.
    screen_top = screen['altitude_deg']/2.
    screen_bottom = -1*screen['altitude_deg']/2.

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
    
    screen_left = -1*screen['azimuth_deg']/2.
    screen_right = screen['azimuth_deg']/2.
    screen_top = screen['altitude_deg']/2.
    screen_bottom = -1*screen['altitude_deg']/2.
    
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

    
    print("AZIMUTH bounds: [%.2f, %.2f]" % (kde_results['az_bounds'][0], kde_results['az_bounds'][1]))
    print("ELEV bounds: [%.2f, %.2f]" % (kde_results['el_bounds'][0], kde_results['el_bounds'][1]))
    print("CENTER: %.2f, %.2f" % (kde_results['center_x'], kde_results['center_y']))

    #%
    fig = plot_centroids(kde_results, weights, screen, xvals=xvals, yvals=yvals,
                         fit_roi_list=fit_roi_list, use_peak=True, exclude_bad=False, min_thr=0.01, marker_scale=100)
    
    label_figure(fig, data_identifier)
    
    pl.savefig(os.path.join(target_fov_dir, 'centroid_peak_rois_by_pos_fit_thr_%.2f.png' % (fit_thr)))
    pl.close()
    
    return kde_results

    #%%

def load_fit_results(animalid, session, fov, experiment='rfs',
                        traceid='traces001', response_type='dff', 
                        fit_desc=None, do_spherical_correction=False, 
                        rootdir='/n/coxfs01/2p-data'):
 
    fit_results = None
    fit_params = None
    try: 
        if fit_desc is None:
            assert response_type is not None, "No response_type or fit_desc provided"
            fit_desc = get_fit_desc(response_type=response_type, do_spherical_correction=do_spherical_correction) 
            #'fit-2dgaus_%s-no-cutoff' % response_type
    
        rfname = 'gratings' if int(session) < 20190511 else experiment
        rfname = rfname.split('_')[1] if 'combined' in rfname else rfname

        #print("... >>> (fitrfs) Loading results: %s (%s)" % (rfname, fit_desc))
        rfdir = glob.glob(os.path.join(rootdir, animalid, session, fov, 
                        '*%s_*' % rfname, #experiment
                        'traces', '%s*' % traceid, 'receptive_fields', 
                        '%s' % fit_desc))[0]
    except AssertionError as e:
        traceback.print_exc()
       
    # Load results
    rf_results_fpath = os.path.join(rfdir, 'fit_results.pkl')
    with open(rf_results_fpath, 'rb') as f:
        fit_results = pkl.load(f)
   
    # Load params 
    rf_params_fpath = os.path.join(rfdir, 'fit_params.json')
    with open(rf_params_fpath, 'r') as f:
        fit_params = json.load(f)
        
    return fit_results, fit_params
  

def fit_rfs(avg_resp_by_cond, fit_params={}, #row_vals=[], col_vals=[], fitparams=None,
            response_type='dff', roi_list=None, #scale_sigma=True,
            #rf_results_fpath='/tmp/fit_results.pkl', 
            do_spherical_correction=False,
            data_identifier='METADATA'):
            #response_thr=None):

    '''
    Main fitting function.    
    Saves 2 output files for fitting: 
        fit_results.pkl 
        fit_params.json
    '''
    #trim=False; hard_cutoff=False; map_thr=''; set_to_min=False; 
     # '''
    # hard_cutoff:
    #     (bool) Use hard cut-off for zscores (set to False to use some % of max value)
    # set_to_min: 
    #     (bool) Threshold x,y condition grid and set non-passing conditions to min value or 0 
    # '''
    print("@@@ doing rf fits @@@")
    scale_sigma = fit_params['scale_sigma']
    sigma_scale = fit_params['sigma_scale']
    row_vals = fit_params['row_vals']
    col_vals = fit_params['col_vals']

    rfdir = fit_params['rfdir'] #os.path.split(rf_results_fpath)[0]    
    rf_results_fpath = os.path.join(rfdir, 'fit_results.pkl')
    rf_params_fpath = os.path.join(rfdir, 'fit_params.json')

    # Save params
    with open(rf_params_fpath, 'w') as f:
        json.dump(fit_params, f, indent=4, sort_keys=True)
    
    # Create subdir for saving each roi's fit
    if not os.path.exists(os.path.join(rfdir, 'roi_fits')):
        os.makedirs(os.path.join(rfdir, 'roi_fits'))

    roi_list = avg_resp_by_cond.columns.tolist()

    bad_rois = [r for r in roi_list if avg_resp_by_cond.max()[r] > 1.0]
    print("... %i bad rois (skipping: %s)" % (len(bad_rois), str(bad_rois)))
    if len(bad_rois) > 0:
        badr_fpath = os.path.join(rfdir.split('/receptive_fields/')[0], 'funky.json')
        with open(badr_fpath, 'w') as f:
            json.dump(bad_rois, f)
     
    fit_results = {}
    for rid in roi_list:
        #print rid
        if rid in bad_rois:
            continue

        roi_fit_results, fig = plot_and_fit_roi_RF(avg_resp_by_cond[rid], 
                                                    row_vals, col_vals,
                                                    scale_sigma=scale_sigma, 
                                                    sigma_scale=sigma_scale) 
        fig.suptitle('roi %i' % int(rid+1))
        label_figure(fig, data_identifier)            
        figname = '%s_%s_RF_roi%05d' % (trace_type, response_type, int(rid+1))
        pl.savefig(os.path.join(rfdir, 'roi_fits', '%s.png' % figname))
        pl.close()
        
        if roi_fit_results != {}:
            fit_results[rid] = roi_fit_results
        #%
    xi = np.arange(0, len(col_vals))
    yi = np.arange(0, len(row_vals))
    xx, yy = np.meshgrid(xi, yi)
        
    with open(rf_results_fpath, 'wb') as f:
        pkl.dump(fit_results, f, protocol=pkl.HIGHEST_PROTOCOL)

    return fit_results, fit_params

#%%

def get_fit_desc(response_type='dff', do_spherical_correction=False):
    if do_spherical_correction:
        fit_desc = 'fit-2dgaus_%s_sphr' % response_type
    else:
        fit_desc = 'fit-2dgaus_%s-no-cutoff' % response_type        
    return fit_desc


def create_rf_dir(animalid, session, fov, run_name, 
               traceid='traces001', response_type='dff', do_spherical_correction=False, fit_thr=0.5,
               rootdir='/n/coxfs01/2p-data'):
    # Get RF dir for current fit type
    fit_desc = get_fit_desc(response_type=response_type, do_spherical_correction=do_spherical_correction)
    fov_dir = os.path.join(rootdir, animalid, session, fov)
    #print("... >>  (fitrfs) creating RF dir:", run_name)

    if 'combined' in run_name:
        traceid_dirs = [t for t in glob.glob(os.path.join(fov_dir, run_name, 'traces', '%s*' % traceid))]
    else: 
        traceid_dirs = [t for t in glob.glob(os.path.join(fov_dir, 'combined_%s_*' % run_name, 'traces', '%s*' % traceid))]
    if len(traceid_dirs) > 1:
        print "[creating RF dir, %s] More than 1 trace ID found:" % run_name
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


def get_rf_to_fov_info(masks, rfdf, zimg, rfname='rfs', #rfdir='/tmp', rfname='rfs',
                       plot_spatial=False, transform_fov=True, create_new=False):    
    '''
    Get FOV info relating cortical position to RF position of all cells.
    Info is saved in: rfdir/fov_info.pkl
    
    Returns:
        fovinfo (dict)
            'roi_positions': 
                dataframe of azimuth (xpos) and elevation (ypos) for cell's 
                cortical position and cell's RF position (i.e., 'posdf')
            'zimg': 
                (array) z-projected image 
            'roi_contours': 
                (list) roi contours, retinotopy.convert_coords.contours_from_masks()
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
        fov_x, fov_y, xlim, ylim = coor.get_roi_position_in_fov(
                                                rfdf, roi_contours, 
                                                rf_exp_name=rfname,
                                                convert_um=True,
                                                roi_list=sorted(rfdf['cell'].values),
                                                npix_y=npix_y,
                                                npix_x=npix_x)
        posdf = pd.DataFrame({'xpos_fov': fov_y,  # ML axis correpsonds to fov y-axis
                               'xpos_rf': rf_x,   # xpos VF should go with ML axis on brain
                               'ypos_fov': fov_x, # AP axis on brain corresponds to fov x-axis
                               'ypos_rf': rf_y    # ypos in VF goes with AP axis on brain.., ish
                               }, index=rfdf.index)
    
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
def get_fit_params(animalid, session, fov, run='combined_rfs_static', traceid='traces001', 
                   response_type='dff', fit_thr=0.5, do_spherical_correction=False, ds_factor=3.,
                   post_stimulus_sec=0.5, sigma_scale=2.35, scale_sigma=True,
                   rootdir='/n/coxfs01/2p-data'):
    
    screen = get_screen_dims()
    run_info, sdf = load_run_info(animalid, session, fov, run,
                                   traceid=traceid, rootdir=rootdir)
    
    row_vals = sorted(sdf['ypos'].unique())
    col_vals = sorted(sdf['xpos'].unique())

    fr = run_info['framerate'] 
    nframes_post_onset = int(round(post_stimulus_sec * fr))
   
     
    rfdir, fit_desc = create_rf_dir(animalid, session, fov, 
                                    run, traceid=traceid,
                                    response_type=response_type, 
                                    do_spherical_correction=do_spherical_correction, 
                                    fit_thr=fit_thr)


    fit_params = {
            'response_type': response_type,
            'frame_rate': fr,
            'nframes_per_trial': run_info['nframes_per_trial'][0],
            'stim_on_frame': run_info['stim_on_frame'],
            'nframes_on': int(run_info['nframes_on'][0]),
            'post_stimulus_sec': post_stimulus_sec,
            'nframes_post_onset': nframes_post_onset,
            'row_spacing': np.mean(np.diff(row_vals)),
            'column_spacing': np.mean(np.diff(col_vals)),
            'fit_thr': fit_thr,
            'sigma_scale': float(sigma_scale),
            'scale_sigma': scale_sigma,
            'screen': screen,
            'row_vals': row_vals,
            'col_vals': col_vals,
            'rfdir': rfdir,
            'fit_desc': fit_desc,
            'do_spherical_correction': do_spherical_correction,
            'downsample_factor': ds_factor
            } 
   
    with open(os.path.join(rfdir, 'fit_params.json'), 'w') as f:
        json.dump(fit_params, f, indent=4, sort_keys=True)
    
    return fit_params

def load_rfmap_array(rfdir, do_spherical_correction=True):  
    rfarray_dpath = os.path.join(rfdir, 'rfmap_array.pkl')    
    avg_resp_by_cond=None
    if os.path.exists(rfarray_dpath):
        print("-- loading: %s" % rfarray_dpath)
        with open(rfarray_dpath, 'rb') as f:
            avg_resp_by_cond = pkl.load(f)
    return avg_resp_by_cond

def save_rfmap_array(avg_resp_by_cond, rfdir):  
    rfarray_dpath = os.path.join(rfdir, 'rfmap_array.pkl')    
    with open(rfarray_dpath, 'wb') as f:
        pkl.dump(avg_resp_by_cond, f, protocol=pkl.HIGHEST_PROTOCOL)
    return

#%%     
def fit_2d_receptive_fields(animalid, session, fov, run, traceid, 
                            reload_data=False, create_new=False,
                            trace_type='corrected', response_type='dff', 
                            do_spherical_correction=False, ds_factor=3, 
                            post_stimulus_sec=0.5, scaley=None,
                            make_pretty_plots=False, nrois_plot=10,
                            plot_response_type='dff', plot_format='svg',
                            #visual_area='', select_rois=False, segment=False,
                            ellipse_ec='w', ellipse_fc='none', ellipse_lw=2, 
                            plot_ellipse=True, scale_sigma=True, sigma_scale=2.35,
                            linecolor='darkslateblue', cmap='bone', legend_lw=2, 
                            fit_thr=0.5, rootdir='/n/coxfs01/2p-data', n_processes=1, test_subset=False):

    rows = 'ypos'; cols = 'xpos';

    # Set output dirs
    # -----------------------------------------------------------------------------
    # rf_param_str = 'fit-2dgaus_%s-no-cutoff' % (response_type) 
    run_name = run.split('_')[1] if 'combined' in run else run
    rfdir, fit_desc = create_rf_dir(animalid, session, fov, 
                                    'combined_%s_static' % run_name, traceid=traceid,
                                    response_type=response_type, 
                                    do_spherical_correction=do_spherical_correction, fit_thr=fit_thr)
    # Get data source
    traceid_dir = rfdir.split('/receptive_fields/')[0]
    data_fpath = os.path.join(traceid_dir, 'data_arrays', 'np_subtracted.npz')
    data_id = '|'.join([animalid, session, fov, run, traceid, fit_desc])
    if not os.path.exists(data_fpath):
        # Realign traces
        print("*****corrected offset unfound, running now*****")
        print("%s | %s | %s | %s | %s" % (animalid, session, fov, run, traceid))
        aggregate_experiment_runs(animalid, session, fov, run_name, traceid=traceid)
        print("*****corrected offsets!*****")
  
    # Create results outfile, or load existing:
    if create_new is False:
        try:
            print "... checking for existing fit results"
            fit_results, fit_params = load_fit_results(animalid, session, fov,
                                        experiment=run_name, traceid=traceid,
                                        response_type=response_type, 
                                        do_spherical_correction=do_spherical_correction)
            print "... loaded RF fit results"
            assert fit_results is not None and fit_params is not None, "EMPTY fit_results"
        except Exception as e:
            traceback.print_exc()
            create_new = True
    print("... do fits?", create_new) 
    avg_resp_by_cond=None

    if create_new or reload_data: #do_fits:
        # Load processed traces 
        raw_traces, labels, sdf, run_info = load_dataset(data_fpath, 
                                                    trace_type=trace_type,
                                                    add_offset=True, make_equal=False,
                                                    create_new=reload_data)        
        #print("--- [%s|%s|%s|%s]: loaded traces (%s, for %s)." % (animalid, session, fov, run, trace_type, response_type))  
        # Get screen dims and fit params
        fit_params = get_fit_params(animalid, session, fov, run=run, traceid=traceid, 
                                    response_type=response_type, 
                                    do_spherical_correction=do_spherical_correction, ds_factor=ds_factor,
                                    fit_thr=fit_thr,
                                    post_stimulus_sec=post_stimulus_sec, 
                                    sigma_scale=sigma_scale, scale_sigma=scale_sigma)

        print('-----------------------------')
        print(fit_params['rfdir'])
        print('-----------------------------')

        # Z-score or dff the traces:
        trials_by_cond = get_trials_by_cond(labels)
        zscored_traces, zscores = process_traces(raw_traces, labels, 
                                                response_type=fit_params['response_type'],
                                                nframes_post_onset=fit_params['nframes_post_onset']) 
        nx = len(fit_params['col_vals'])
        ny = len(fit_params['row_vals'])
       
        # -------------------------------------------------------
        if create_new is False:
            avg_resp_by_cond = load_rfmap_array(fit_params['rfdir'], 
                                            do_spherical_correction=do_spherical_correction)
        if avg_resp_by_cond is None:
            print("Error loading array, extracting now")

            print("...getting avg by cond")
            avg_resp_by_cond = group_trial_values_by_cond(zscores, trials_by_cond, nx=nx, ny=ny,
                                                        do_spherical_correction=do_spherical_correction)
            if do_spherical_correction:
                print("...doin spherical warps")
                if n_processes>1:
                    avg_resp_by_cond = sphr_correct_maps_mp(avg_resp_by_cond, fit_params, 
                                                                n_processes=n_processes, test_subset=test_subset)
                else:
                    avg_resp_by_cond = sphr_correct_maps(avg_resp_by_cond, fit_params, 
                                                                multiproc=False)
            print("...saved array")
            save_rfmap_array(avg_resp_by_cond, fit_params['rfdir'])
         
        # Do fits 
        print("...now, fitting")
        fit_results, fit_params = fit_rfs(avg_resp_by_cond, response_type=response_type, 
                                          do_spherical_correction=do_spherical_correction, 
                                            fit_params=fit_params, data_identifier=data_id)            
    try:
        # Convert to dataframe
        if avg_resp_by_cond is None:
            avg_resp_by_cond = load_rfmap_array(fit_params['rfdir'], 
                                        do_spherical_correction=do_spherical_correction)

        fitdf_pos = rfits_to_df(fit_results, scale_sigma=False, convert_coords=False,
                            fit_params=fit_params, spherical=do_spherical_correction, 
                            row_vals=fit_params['row_vals'], col_vals=fit_params['col_vals'])
        fit_roi_list = fitdf_pos[fitdf_pos['r2'] > fit_thr].sort_values('r2', axis=0, ascending=False).index.tolist()
        print("... %i out of %i fit rois with r2 > %.2f" % 
                    (len(fit_roi_list), fitdf_pos.shape[0], fit_thr))

        # Plot all RF maps for fit cells (limit = 60 to plot)
        fig = plot_best_rfs(fit_roi_list, avg_resp_by_cond, fitdf_pos, fit_params,
                            single_colorbar=True, plot_ellipse=True, nr=6, nc=10)
        label_figure(fig, data_id)
        figname = 'top%i_fit_thr_%.2f_%s_ellipse_sc_2' % (len(fit_roi_list), fit_thr, fit_desc)
        pl.savefig(os.path.join(rfdir, '%s.png' % figname))
        print figname
        pl.close()
    except Exception as e:
        traceback.print_exc()
        print("Error plotting all RF maps that pass thr.")
    
    #%%
    #make_pretty_plots= False
    if make_pretty_plots:
        print("... plottin' pretty (%s)" % plot_response_type)
        if response_type != plot_response_type or create_new is False:
            raw_traces, labels, sdf, run_info = load_dataset(data_fpath, 
                                            trace_type=trace_type,
                                            add_offset=True, make_equal=False)
            # Z-score or dff the traces:
            trials_by_cond = get_trials_by_cond(labels)
            zscored_traces, zscores = process_traces(raw_traces, labels, 
                                        response_type=plot_response_type,
                                        nframes_post_onset=fit_params['nframes_post_onset'])
            nx = len(fit_params['col_vals'])
            ny = len(fit_params['row_vals'])
            # -------------------------------------------------------
            avg_resp_by_cond = load_rfmap_array(fit_params['rfdir'], do_spherical_correction=do_spherical_correction)
            if avg_resp_by_cond is None:
                print("Error loading array, ABORTING.")
                return None   
    #%
        # Overlay RF map and mean traces:
        # -----------------------------------------------------------------------------
#        linecolor = 'darkslateblue' #'purple'; cmap = 'bone'; ellipse_fc = 'none'; ellipse_ec = 'w'; ellipse_lw=2.;

        nframes_plot = fit_params['stim_on_frame'] + fit_params['nframes_on'] + fit_params['nframes_post_onset'] 
        start_frame = fit_params['stim_on_frame'] #stim_on_frame #plot_start_frame #stim_on_frame #0
        yunit_sec = round(fit_params['nframes_on']/fit_params['frame_rate'], 1)
            
        best_rois_figdir = os.path.join(rfdir, 'best_rfs', '%s_%s' % (plot_response_type, plot_format))
        if not os.path.isdir(best_rois_figdir):
            os.makedirs(best_rois_figdir)
    
        #if not do_fits: # need to load results
        fitdf = rfits_to_df(fit_results, scale_sigma=False, fit_params=fit_params, 
                            row_vals=fit_params['row_vals'], col_vals=fit_params['col_vals'],
                            convert_coords=True, spherical=do_spherical_correction)
        fit_roi_list = fitdf[fitdf['r2'] > fit_thr].sort_values('r2', axis=0, ascending=False).index.tolist()
        print("%i out of %i fit rois with r2 > %.2f" % 
                    (len(fit_roi_list), fitdf.shape[0], fit_thr))

        plot_ellipse=True
        for ri, rid in enumerate(fit_roi_list[0:nrois_plot]):
            if ri % 20 == 0:
                print("    %i of %i pretty plots (total was %i)" % (int(ri+1),nrois_plot,len(fit_roi_list)))
            fig = overlay_traces_on_rfmap(rid, avg_resp_by_cond, zscored_traces, 
                                        labels, sdf, run_info,
                                        response_type=plot_response_type, 
                                        nframes_plot=nframes_plot, 
                                        start_frame=start_frame, yunit_sec=yunit_sec, scaley=scaley,
                                        row_vals=fit_params['row_vals'], col_vals=fit_params['col_vals'],
                                        linecolor=linecolor, cmap=cmap,
                                        fitdf=fitdf, plot_ellipse=plot_ellipse,
                                        ellipse_fc=ellipse_fc, ellipse_ec=ellipse_ec,
                                        ellipse_lw=ellipse_lw, legend_lw=legend_lw)
            label_figure(fig, data_id)
            fig.suptitle('roi %i' % int(rid+1))
            figname = 'roi%05d-overlay_%s' % (int(rid+1), plot_response_type) 
            pl.savefig(os.path.join(best_rois_figdir, '%s.%s' % (figname, plot_format)), 
                        bboxx_inches='tight')
            pl.close()
            
            
    #if not do_fits: # need to load results
    fitdf = rfits_to_df(fit_results, scale_sigma=False, fit_params=fit_params,
                        row_vals=fit_params['row_vals'], col_vals=fit_params['col_vals'],
                        convert_coords=True, spherical=do_spherical_correction)

    #%%
    if create_new or make_pretty_plots:
        try:
            # Identify VF area to target:
            target_fov_dir = os.path.join(rfdir, 'target_fov')
            if not os.path.exists(target_fov_dir):
                os.makedirs(target_fov_dir)
                
            kde_results = target_fov(avg_resp_by_cond, fitdf, fit_params['screen'], 
                                        fit_roi_list=fit_roi_list, 
                                        row_vals=fit_params['row_vals'], col_vals=fit_params['col_vals'], 
                                        compare_kdes=False,
                                        weight_type='kde', target_fov_dir=target_fov_dir, 
                                        data_identifier=data_id)
        except Exception as e:
            traceback.print_exc()
            print("Error finding target coords for FOV.")
   
            # plot all RFs to screen  
        try:
            fig = plot_rfs_to_screen(fitdf, sdf, fit_params['screen'], fit_roi_list=fit_roi_list)
            label_figure(fig, data_id)
            #%
            figname = 'overlaid_RFs_top%i_fit_thr_%.2f_%s' % (len(fit_roi_list), fit_thr, fit_desc)
            pl.savefig(os.path.join(rfdir, '%s_sc.png' % figname))
            print figname
            pl.close()
        except Exception as e:
            traceback.print_exc()
            print("Error plotting RFs to screen coords.")
            
        try:
            # plot all RFs to screen, but make it look nicer
            fig = plot_rfs_to_screen_pretty(fitdf, sdf, fit_params['screen'], fit_roi_list=fit_roi_list)
            label_figure(fig, data_id)
            #%
            figname = 'overlaid_RFs_pretty' 
            pl.savefig(os.path.join(rfdir, '%s_sc.svg' % figname))
            print figname
            pl.close()
        except Exception as e:
            traceback.print_exc()
            print("Error plotting RFs to screen coords.")
        #%%       
    print("DONE! :)")
 
    return fit_results, fit_params #fovinfo

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
    
#    parser.add_option('--segment', action='store_true', dest='segment', default=False, \
#                      help="Set flag to use segmentation of FOV for select visual area")
#    parser.add_option('-V', '--area', action='store', dest='visual_area', default='', \
#                      help="Name of visual area, if --segment is flagged")
#    parser.add_option('--select-rois', action='store_true', dest='select_rois', default=False, \
#                      help="Set flag to select only visual or selective rois")
    
#    parser.add_option('-T', '--thr', action='store', dest='response_thr', default=None, \
#                      help="Min snr or zscore value for cells to fit (default: None (fits all))")
    parser.add_option('-M', '--resp', action='store', dest='response_type', default='dff', \
                      help="Response metric to use for creating RF maps (default: dff)")
#    
    parser.add_option('--new', action='store_true', dest='create_new', default=False, \
                      help="Flag to refit all rois")

    # pretty plotting options
    parser.add_option('--pretty', action='store_true', dest='make_pretty_plots', default=False, \
                      help="Flag to make pretty plots for roi fits")
    parser.add_option('-f', '--ellipse-fc', action='store', dest='ellipse_fc', 
                      default='none', help="[prettyplots] Ellipse face color (default:none)")
    parser.add_option('-e', '--ellipse-ec', action='store', dest='ellipse_ec', 
                      default='w', help="[prettyplots] Ellipse edge color (default:w)")
    parser.add_option('-l', '--ellipse-lw', action='store', dest='ellipse_lw', 
                      default=2, help="[prettyplots] Ellipse linewidth (default:2)")
    parser.add_option('--no-ellipse', action='store_false', dest='plot_ellipse', 
                      default=True, help="[prettyplots] Flag to NOT plot fit RF as ellipse")
    parser.add_option('-L', '--linecolor', action='store', dest='linecolor', 
                      default='darkslateblue', help="[prettyplots] Color for traces (default:darkslateblue)")
    parser.add_option('-c', '--cmap', action='store', dest='cmap', 
                      default='bone', help="[prettyplots] Cmap for RF maps (default:bone)")
    parser.add_option('-W', '--legend-lw', action='store', dest='legend_lw', 
                      default=2.0, help="[prettyplots] Lw for df/f legend (default:2)")
    parser.add_option('--fmt', action='store', dest='plot_format', 
                      default='svg', help="[prettyplots] Plot format (default:svg)")
    parser.add_option('-y', '--scaley', action='store', dest='scaley', default=None, 
                        help="[prettyplots] Set to float to set scale y across all plots (default: max of current trace)")
    parser.add_option('--nrois',  action='store', dest='nrois_plot', default=10, 
                      help="[prettyplots] N rois plot")



    # RF fitting options
    parser.add_option('--no-scale', action='store_false', dest='scale_sigma', 
                      default=False, help="flag to NOT scale sigma (use true sigma)")
    parser.add_option('--sigma', action='store', dest='sigma_scale', 
                      default=2.35, help="Sigma size to scale (FWHM, 2.35)")
    parser.add_option('-F', '--fit-thr', action='store', dest='fit_thr', default=0.5, 
                      help="Fit threshold (default:0.5)")
    parser.add_option('-p', '--post', action='store', dest='post_stimulus_sec', default=0.0, 
                      help="N sec to include in stimulus-response calculation for maps (default:0.0)")

    parser.add_option('--load', action='store_true', dest='reload_data', default=False, 
                      help="flag to reload/reprocess data arrays")
    parser.add_option('-n', '--nproc', action='store', dest='n_processes', default=1, 
                      help="N processes")

    parser.add_option('--sphere', action='store_true', 
                        dest='do_spherical_correction', default=False, help="N processes")
    (options, args) = parser.parse_args(options)

    return options

#%%

rootdir = '/n/coxfs01/2p-data'
animalid = 'JC085' #'JC097' #'JC084' #'JC059'
session = '20190622' #'20190623' #'20190522' #'20190227'
fov = 'FOV1_zoom2p0x' #'FOV4_zoom4p0x'
run = 'rfs' #'combined_s_static'
traceid = 'traces001' #'traces001'
#segment = False
#visual_area = 'V1'
trace_type = 'corrected'
response_type = 'dff'
#select_rois = False
rows = 'ypos'
cols = 'xpos'
fit_thr = 0.5
post_stimulus_sec = 0.5

options = ['-i', animalid, '-S', session, '-A', fov, '-R', run, '-t', traceid,
           '--pretty', '--new', '-p', post_stimulus_sec]

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
    
    #segment = optsE.segment
    #visual_area = optsE.visual_area
    #select_rois = optsE.select_rois
    
    response_type = optsE.response_type
    do_spherical_correction = optsE.do_spherical_correction
    
    #response_thr = optsE.response_thr
    create_new= optsE.create_new

    fit_thr = float(optsE.fit_thr) 
    post_stimulus_sec = float(optsE.post_stimulus_sec)
    reload_data = optsE.reload_data

    scaley = float(optsE.scaley) if optsE.scaley is not None else optsE.scaley
    
    make_pretty_plots = optsE.make_pretty_plots
    plot_format = optsE.plot_format

    n_processes = int(optsE.n_processes)
    test_subset=False

    fit_results, fit_params = fit_2d_receptive_fields(animalid, session, fov, 
                                run, traceid, 
                                trace_type=trace_type, 
                                post_stimulus_sec=post_stimulus_sec,
                                scaley=scaley,
                                fit_thr=fit_thr,
                                reload_data=reload_data,
                                #visual_area=visual_area, select_rois=select_rois,
                                response_type=response_type, #response_thr=response_thr, 
                                do_spherical_correction=do_spherical_correction,
                                create_new=create_new,
                                make_pretty_plots=make_pretty_plots, 
                                nrois_plot=int(optsE.nrois_plot),
                                ellipse_ec=optsE.ellipse_ec, 
                                ellipse_fc=optsE.ellipse_fc, 
                                ellipse_lw=optsE.ellipse_lw, 
                                plot_ellipse=optsE.plot_ellipse,
                                linecolor=optsE.linecolor, cmap=optsE.cmap, 
                                legend_lw=optsE.legend_lw, 
                                plot_format=plot_format, n_processes=n_processes, 
                                test_subset=test_subset)
    
    print("--- fit %i rois total ---" % (len(fit_results.keys())))

    #%

    print "((( RFs done! )))))"
        
if __name__ == '__main__':
    main(sys.argv[1:])


    


# %%
