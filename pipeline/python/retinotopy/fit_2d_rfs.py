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

import pylab as pl
import seaborn as sns
import cPickle as pkl
import numpy as np
import scipy as sp
from scipy import stats
import pandas as pd

import scipy.optimize as opt
from matplotlib.patches import Ellipse, Rectangle

from mpl_toolkits.axes_grid1 import AxesGrid
from pipeline.python.utils import natural_keys, label_figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import argrelextrema
from scipy.interpolate import splrep, sproot, splev, interp1d

from pipeline.python.retinotopy import utils as rutils

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


def zscore_traces(raw_traces, labels, nframes_post_onset=None):
    
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
    for trial, tmat in labels.groupby(['trial']):

        # Get traces using current trial's indices: divide by std of baseline
        curr_traces = raw_traces.iloc[tmat.index] 
        bas_std = curr_traces.iloc[0:stim_on_frame].std(axis=0)
        curr_zscored_traces = pd.DataFrame(curr_traces).divide(bas_std, axis='columns')
        zscored_traces_list.append(curr_zscored_traces)
        
        # Also get zscore (single value) for each trial:
        zscores_list.append(curr_zscored_traces.iloc[stim_on_frame:stim_on_frame+nframes_post_onset].mean(axis=0)) # Get average zscore value for current trial
        
    zscored_traces = pd.concat(zscored_traces_list, axis=0)
    zscores =  pd.concat(zscores_list, axis=1).T # cols=rois, rows = trials

    return zscored_traces, zscores

def group_zscores_by_cond(zscores, trials_by_cond):
    zscores_by_cond = dict()
    for cfg, trial_ixs in trials_by_cond.items():
        zscores_by_cond[cfg] = zscores.iloc[trial_ixs]  # For each config, array of size ntrials x nrois
    return zscores_by_cond

#%


def order_configs_by_grid(sdf, labels, rows='ypos', cols='xpos'):
    
    assert rows in sdf.columns, "Specified ROWS <%s> not found." % rows
    assert cols in sdf.columns, "Specified COLS <%s> not found." % cols
    
    row_vals = sorted(sdf[rows].unique())
    col_vals = sorted(sdf[cols].unique())
    
    config_trial_ixs = dict()
    cix = 0
    for si, row_val in enumerate(sorted(row_vals)):
        for mi, col_val in enumerate(col_vals):
            config_trial_ixs[cix] = {}
            cfg = sdf[(sdf[rows]==row_val) & (sdf[cols]==col_val)].index.tolist()[0]
            trial_ixs = sorted( list(set( [int(tr[5:])-1 for tr in labels[labels['config']==cfg]['trial']] )) )
            config_trial_ixs[cix]['config'] = cfg
            config_trial_ixs[cix]['trial_ixs'] = trial_ixs
            cix += 1

    return row_vals, col_vals, config_trial_ixs



#%%

# -----------------------------------------------------------------------------
# ROI filtering functions:
# -----------------------------------------------------------------------------


def get_rois_by_visual_area(fov_dir):
    
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
    g = offset + amplitude*np.exp( -a*((x-xo)**2) - b*(x-xo)*(y-yo) - c*((y-y0)**2) )
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
        raise MultiplePeaks("The dataset appears to have multiple peaks, and "
                "thus the FWHM can't be determined.")
    elif len(roots) < 2:
        raise NoPeaksFound("No proper peaks were found in the data set; likely "
                "the dataset is flat (e.g. all zeros).")
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
    
#%%

def extract_options(options):
    
    parser = optparse.OptionParser()

    parser.add_option('-D', '--root', action='store', dest='rootdir', default='/n/coxfs01/2p-data',\
                      help='data root dir (root project dir containing all animalids) [default: /n/coxfs01/2pdata]')
    parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')

    # Set specific session/run for current animal:
    parser.add_option('-S', '--session', action='store', dest='session', default='', \
                      help='session dir (format: YYYMMDD_ANIMALID')
    parser.add_option('-A', '--acq', action='store', dest='acquisition', default='FOV1', \
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
    (options, args) = parser.parse_args(options)

    return options


#%%

rootdir = '/n/coxfs01/2p-data'
animalid = 'JC080' #'JC059'
session = '20190430' #'20190227'
fov = 'FOV1_zoom2p0x' #'FOV4_zoom4p0x'
run = 'combined_gratings_static'
traceid = 'traces001' #'traces001'
segment = False
visual_area = ''
trace_type = 'corrected'


options = ['-i', animalid, '-S', session, '-A', fov, '-R', run, '-t', traceid]
if segment:
    options.extend(['--segment', '-V', visual_area])
    



fov_dir = os.path.join(rootdir, animalid, session, fov)
traceid_dirs = glob.glob(os.path.join(fov_dir, run, 'traces', '%s*' % traceid))
if len(traceid_dirs) > 1:
    print "More than 1 trace ID found:"
    for ti, traceid_dir in enumerate(traceid_dirs):
        print ti, traceid_dir
    sel = input("Select IDX of traceid to use: ")
    traceid_dir = traceid_dirs[int(sel)]
else:
    traceid_dir = traceid_dirs[0]
traceid = os.path.split(traceid_dir)[-1]
    
    
data_fpath = glob.glob(os.path.join(traceid_dir, 'data_arrays', '*.npz'))[0]
dset = np.load(data_fpath)
dset.keys()

data_identifier = '|'.join([animalid, session, fov, run, traceid, visual_area])
print data_identifier


#%%

# Select ROIs:
if segment:
    included_rois = get_rois_by_visual_area(fov_dir, segment=segment)
else:
    included_rois = []
    
# Set output dir:
output_dir = os.path.join(traceid_dir, 'figures', 'receptive_fields')
if segment:
    output_dir = os.path.join(output_dir, visual_area)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
print "Saving output to:", output_dir


#%%#%%

# Load parsed data:
raw_traces = pd.DataFrame(dset[trace_type]) #, index=zscored_traces.index)

labels = pd.DataFrame(data=dset['labels_data'], columns=dset['labels_columns'])
trials_by_cond = get_trials_by_cond(labels)

#%
select_rois = False

if select_rois:
    visual_rois, selective_rois, rstats_fpath = get_responsive_rois(traceid_dir, included_rois=included_rois)
else:
    visual_rois = np.arange(raw_traces.shape[-1])
    selective_rois = np.arange(raw_traces.shape[-1])

sdf = pd.DataFrame(dset['sconfigs'][()]).T

# Format condition info:
if 'image' in sdf['stimtype']:
    aspect_ratio = sdf['aspect'].unique()[0]
    sdf['size'] = [round(sz/aspect_ratio, 1) for sz in sdf['size']]


#%%
# zscore the traces:
# -----------------------------------------------------------------------------
zscored_traces, zscores = zscore_traces(raw_traces, labels)
zscores_by_cond = group_zscores_by_cond(zscores, trials_by_cond)

# Sort ROIs by zscore by cond
# -----------------------------------------------------------------------------
avg_zscores_by_cond = pd.DataFrame([zscores_by_cond[cfg].mean(axis=0) \
                                    for cfg in sorted(zscores_by_cond.keys(), key=natural_keys)]) # nconfigs x nrois

    
#%%
# Sort mean (or max) zscore across trials for each config, and find "best config"

sorted_visual = sort_rois_by_max_response(avg_zscores_by_cond, visual_rois)
sorted_selective = sort_rois_by_max_response(avg_zscores_by_cond, selective_rois)

#%%%
rows = 'ypos'
cols = 'xpos'

row_vals, col_vals, config_trial_ixs = order_configs_by_grid(sdf, labels, rows=rows, cols=cols)


#%%



def plot_roi_RF(response_vector, ncols, nrows, ax=None, hard_cutoff=True,
                set_to_min=True, map_thr=1.5):
    
    if ax is None:
        fig, ax = pl.subplots()
        
    coordmap_r = np.reshape(response_vector, (len(col_vals), len(row_vals))).T
     
    rfmap = coordmap_r.copy()
    if hard_cutoff:
        rfmap[coordmap_r < map_thr] = coordmap_r.min()*0.1 if set_to_min else 0
    else:
        rfmap[coordmap_r < coordmap_r.max()*map_thr] =  coordmap_r.min()*0.3 if set_to_min else 0
    im = ax.imshow(rfmap, cmap='inferno')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    ax.figure.colorbar(im, cax=cax, orientation='vertical')
    
    return ax, rfmap


    
def do_2d_fit(rfmap, nx=None, ny=None):

    # Set params for fit:
    xi = np.arange(0, nx)
    yi = np.arange(0, ny)
    
    popt=None; pcov=None; fitr=None; r2=None; success=None;
    try:
        xx, yy = np.meshgrid(xi, yi)      
        #plot_xx = xx.copy()
        #plot_yy = yy.copy()
        amplitude = rfmap.max()
        y0, x0 = np.where(rfmap == rfmap.max())
        print "x0, y0: (%i, %i)" % (int(x0), int(y0))
    
        try:
            sigma_x = fwhm(xi, rfmap.sum(axis=0))
        except Exception as e:
            print e
            sigma_x = raw_fwhm(rfmap.sum(axis=0)) 
        try:
            sigma_y = fwhm(yi, rfmap.sum(axis=1))
        except Exception as e:
            print e
            sigma_y = raw_fwhm(rfmap.sum(axis=1))
        print "sig-X, sig-Y:", sigma_x, sigma_y
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
        if round(r2, 3) < 0.15 or len(np.where(fitr > fitr.min())[0]) < 4:
            success = False
        else:
            success = True
    except Exception as e:
        print e
    
    return {'popt': popt, 'pcov': pcov, 'init': initial_guess, 'r2': r2, 'success': success}, fitr, xx, yy

#%%
def plot_and_fit_roi_RF(response_vector, row_vals, col_vals, 
                        hard_cutoff=True, map_thr=1.0, set_to_min=True):
    
    results = {}
    fig, axes = pl.subplots(1,2, figsize=(8, 4)) # pl.figure()
    ax = axes[0]
    #ax, rfmap = plot_roi_RF(avg_zscores_by_cond[rid], ncols=len(col_vals), nrows=len(row_vals), ax=ax)
    ax, rfmap = plot_roi_RF(response_vector, ax=ax,
                            ncols=len(col_vals), nrows=len(row_vals),
                            hard_cutoff=hard_cutoff, map_thr=map_thr, set_to_min=set_to_min)

    ax2 = axes[1]

    # Do fit 
    # ---------------------------------------------------------------------
    fitr, fit_y, xx, yy = do_2d_fit(rfmap, nx=len(col_vals), ny=len(row_vals))

    
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
                   'xx': xx,
                   'yy': yy}
        
    
    return results, fig
    
    #%%

#map_thr = 0.6
hard_cutoff = True   # Use hard cut-off for zscores (set to False to use some % of max value)
plot_zscored = True  # Use zscore as metric (alt: snr, df/f?)
set_to_min = True    # Threshold x,y condition grid and set non-passing conditions to min value or 0.
#use_moments = False
sigma_scale = 2.35   # Value to scale sigma in order to get FW (instead of FWHM)

# Create subdir for saving figs/results based on fit params:
# -----------------------------------------------------------------------------
set_to_min_str = 'min' if set_to_min else 'zeros'

if hard_cutoff:
    map_thr = 2.0
    cutoff_type = 'hard_thr'
else:
    map_thr = 0.6
    cutoff_type = 'perc_max'
    
trace_type = 'zscored_RF' if plot_zscored else 'raw_RF'

#metric_name = 'zscore' if plot_zscored else 'snr'
#value_name = 'zscore' if plot_zscored else 'intensity'

        
# Create output dir to save ROI plots in:
roi_rf_dir = os.path.join(output_dir, 'rfs_by_roi_2dgaus_%s_%.2f_set_%s' % (cutoff_type, map_thr, set_to_min_str))
if not os.path.exists(roi_rf_dir):
    os.makedirs(roi_rf_dir)
print "Saving figures to:", roi_rf_dir


#%%


results_outfile = 'roi_fit_results_2dgaus_%s_%.2f_set_%s.pkl' % (cutoff_type, map_thr, set_to_min_str)
print results_outfile

do_fits = False
if os.path.exists(os.path.join(output_dir, results_outfile)):
    print "Loading existing results..."
    with open(os.path.join(output_dir, results_outfile), 'rb') as f:
        results = pkl.load(f)
        
else:
    do_fits = True
    
#%%
    
#rid = 106 ##89 #106 #36 # 89
if do_fits:
    #%
    RF = {}
    for rid in selective_rois:
        #%
        print rid

        roi_fit_results, fig = plot_and_fit_roi_RF(avg_zscores_by_cond[rid], row_vals, col_vals)
        fig.suptitle('roi %i' % int(rid+1))
        
        label_figure(fig, data_identifier)            
        figname = '%s_roi%05d' % (trace_type, int(rid+1))
        pl.savefig(os.path.join(roi_rf_dir, '%s.png' % figname))
        pl.close()
        
        if roi_fit_results != {}:
            RF[rid] = roi_fit_results
        
        #%
    results = {'fits': RF,
               'fit_params': {'rfmap_thr': map_thr,
                              'cut_off': cutoff_type,
                              'set_to_min': set_to_min_str,
                              'xx': RF[RF.keys()[0]]['xx'],
                              'yy': RF[RF.keys()[0]]['yy'],
                              'zscored': plot_zscored},
               'row_vals': row_vals,
               'col_vals': col_vals}
                   
    with open(os.path.join(output_dir, results_outfile), 'wb') as f:
        pkl.dump(results, f, protocol=pkl.HIGHEST_PROTOCOL)
        


#%%

if select_rois:
    # See if there are any cells missed by visual/selectivity tests:
    missed_rois = [r for r in np.arange(0, avg_zscores_by_cond.shape[-1]) if r not in selective_rois]
    print "%i missed rois." % len(missed_rois)
    
    missed_roi_dir = os.path.join(roi_rf_dir, 'missed_rois')
    if not os.path.exists(missed_roi_dir): os.makedirs(missed_roi_dir)
    
    missed_RFs = {}
    
    for rid in missed_rois:
    
        roi_fit_results, fig = plot_and_fit_roi_RF(avg_zscores_by_cond[rid], row_vals, col_vals)
        
        if roi_fit_results!={} and roi_fit_results['fit_r']['success']:
            missed_RFs[rid] = roi_fit_results
            print rid
            fig.suptitle('roi %i' % int(rid+1))        
            label_figure(fig, data_identifier)
            figname = '%s_roi%05d' % (trace_type, int(rid+1))
            pl.savefig(os.path.join(missed_roi_dir, '%s.png' % figname))
        pl.close()
    
        #%
    missed_results = {'fits': missed_RFs,
                   'fit_params': {'rfmap_thr': map_thr,
                                  'cut_off': cutoff_type,
                                  'set_to_min': set_to_min_str,
                                  'xx': missed_RFs[missed_RFs.keys()[0]]['xx'],
                                  'yy': missed_RFs[missed_RFs.keys()[0]]['yy'],
                                  'zscored': plot_zscored},
                   'row_vals': row_vals,
                   'col_vals': col_vals}
    
    
    missed_results_outfile = 'missed_%s' % results_outfile
    with open(os.path.join(missed_roi_dir, missed_results_outfile), 'wb') as f:
        pkl.dump(missed_results, f, protocol=pkl.HIGHEST_PROTOCOL)


#%%

fit_thr = 0.51
fitdf = pd.DataFrame(results['fits']).T
fitted_rois = fitdf[fitdf['r2'] > fit_thr].sort_values('r2', axis=0, ascending=False).index.tolist()
print "%i out of %i fit rois with r2 > %.2f" % (len(fitted_rois), fitdf.shape[0], fit_thr)

if select_rois:
    fitdf2 = pd.DataFrame(missed_results['fits']).T
    fitted_rois2 = fitdf2[fitdf2['r2'] > fit_thr].sort_values('r2', axis=0, ascending=False).index.tolist()
    print "%i out of %i non-selective rois with r2 > %.2f" % (len(fitted_rois2), len(missed_rois), fit_thr)


#%%
plot_ellipse = True
single_colorbar = False

cmap = 'magma' if plot_ellipse else 'inferno' # inferno
cbar_mode = 'single' if single_colorbar else  'each'

vmin = max([avg_zscores_by_cond.min().min(), 0])
vmax = min([5, avg_zscores_by_cond.max().max()])


nr=6# 6 #6
nc=10 #10 #10
fig = pl.figure(figsize=(nc*2,nr+2))
grid = AxesGrid(fig, 111,
            nrows_ncols=(nr, nc),
            axes_pad=0.5,
            cbar_mode=cbar_mode,
            cbar_location='right',
            cbar_pad=0.05, cbar_size="3%")


plot_missed = False
if select_rois:
    if plot_missed:
        rfdf = fitdf2.copy()
        plot_str = 'missed_RFs'
        fit_roi_list = copy.copy(fitted_rois2)
    else:
        rfdf = fitdf.copy()
        plot_str = 'selective_RFs'
        fit_roi_list = copy.copy(fitted_rois)
else:
    rfdf = fitdf.copy()
    plot_str = 'allfitRFs'
    fit_roi_list = copy.copy(fitted_rois)
    

for aix, rid in enumerate(fit_roi_list[0:nr*nc]):
    ax = grid.axes_all[aix]
    ax.clear()
    coordmap = np.reshape(avg_zscores_by_cond[rid], (len(col_vals), len(row_vals))).T
    
    #rfmap = fitdf['data'][rid]
    im = ax.imshow(coordmap, cmap=cmap) #, vmin=vmin, vmax=vmax)
    #ax.contour(results['fit_params']['xx'], results['fit_params']['yy'], fitdf['fit'][rid].reshape(coordmap.shape), 1, colors='w')
    ax.set_title('roi %i (r2=%.2f)' % (int(rid+1), rfdf['r2'][rid]), fontsize=8)
    
    if plot_ellipse:
        # = Ellipse((x0_f, y0_f), abs(sigx_f)*sig_scale, abs(sigy_f)*sig_scale, angle=np.rad2deg(theta_f)) #theta_f)

        ell = Ellipse((rfdf['x0'][rid], rfdf['y0'][rid]), abs(rfdf['sigma_x'][rid])*sigma_scale, abs(rfdf['sigma_y'][rid])*sigma_scale, angle=np.rad2deg(rfdf['theta'][rid]))
        ell.set_alpha(0.5)
        ell.set_edgecolor('w')
        ell.set_facecolor('cornflowerblue')
        ax.add_patch(ell)
        
    if not single_colorbar:
        cbar = ax.cax.colorbar(im)
        cbar = grid.cbar_axes[aix].colorbar(im)
        cbar_yticks = [coordmap.min(), coordmap.max()]
        cbar.cbar_axis.axes.set_yticks(cbar_yticks)
        cbar.cbar_axis.axes.set_yticklabels([int(round(cy)) for cy in cbar_yticks], fontsize=8)
    
    ax.set_ylim([0, len(row_vals)])
    ax.set_xlim([0, len(col_vals)])
    
    ax.invert_yaxis()

if single_colorbar:
    cbar = ax.cax.colorbar(im)
    cbar = grid.cbar_axes[0].colorbar(im)

#%
for a in np.arange(0, nr*nc):
    grid.axes_all[a].set_axis_off() 

if not single_colorbar and len(fit_roi_list) < (nr*nc):
    for nix in np.arange(len(fit_roi_list), nr*nc):
        grid.cbar_axes[nix].remove()

pl.subplots_adjust(left=0.05, right=0.95, wspace=0.3, hspace=0.3)

label_figure(fig, data_identifier)

#%
figname = '%s_fits_2dgaus_FIT%s_%s_%.2f_set_%s_top%i_fit_thr_%.2f' % (plot_str, trace_type,  cutoff_type, map_thr, set_to_min_str, len(fit_roi_list), fit_thr)
if plot_ellipse:
    figname = '%s_ellipse' % figname
pl.savefig(os.path.join(output_dir, '%s.png' % figname))
print figname



#%%

screen = rutils.get_screen_info(animalid, session, rootdir=rootdir)
screen_left = -1*screen['azimuth']/2.
screen_right = screen['azimuth']/2.
screen_top = screen['elevation']/2.
screen_bottom = -1*screen['elevation']/2.


#%%

from matplotlib.pyplot import cm

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


plot_missed = False

if select_rois:
    if plot_missed:
        rfdf = fitdf2.copy()
        plot_str = 'missed_RFs'
        fit_roi_list = copy.copy(fitted_rois2)
    else:
        rfdf = fitdf.copy()
        plot_str = 'selective_RFs'
        fit_roi_list = copy.copy(fitted_rois)
else:
    rfdf = fitdf.copy()
    plot_str = 'allfitRFs'
    fit_roi_list = copy.copy(fitted_rois)
    
fig, ax = pl.subplots(figsize=(12, 6))

screen_rect = Rectangle(( min(col_vals), min(row_vals)), max(col_vals)-min(col_vals), 
                        max(row_vals)-min(row_vals), facecolor='none', edgecolor='k')
ax.add_patch(screen_rect)

rcolors=iter(cm.rainbow(np.linspace(0,1,len(fit_roi_list))))
for rid in fit_roi_list:
    rcolor = next(rcolors)
    #ax.plot(fitdf['x0'][rid], fitdf['y0'][rid], marker='*', color=rcolor)
    
    xx, yy, sigma_x, sigma_y = convert_fit_to_coords(rfdf, row_vals, col_vals, rid=rid)
        
    ell = Ellipse((xx, yy), abs(sigma_x)*sigma_scale, abs(sigma_y)*sigma_scale, angle=np.rad2deg(rfdf['theta'][rid]))
    ell.set_alpha(0.5)
    ell.set_edgecolor(rcolor)
    ell.set_facecolor('none')
    ax.add_patch(ell)
#ax.invert_yaxis()

ax.set_ylim([screen_bottom, screen_top])
ax.set_xlim([screen_left, screen_right])

#%
figname = '%s_overlaid_RFs_%s_FIT2dgaus_%s_%.2f_set_%s_top%i_fit_thr_%.2f' % (plot_str, trace_type,  cutoff_type, map_thr, set_to_min_str, len(fitted_rois), fit_thr)
pl.savefig(os.path.join(output_dir, '%s.png' % figname))
print figname

#%%

fig, ax = pl.subplots(figsize=(12, 6))
ax.set_ylim([screen_bottom, screen_top])
ax.set_xlim([screen_left, screen_right])

#%
screen_rect = Rectangle(( min(col_vals), min(row_vals)), max(col_vals)-min(col_vals), 
                        max(row_vals)-min(row_vals), facecolor='none', edgecolor='k')
ax.add_patch(screen_rect)

max_zscores = avg_zscores_by_cond.max(axis=0)

xx, yy, sigma_x, sigma_y = convert_fit_to_coords(rfdf, row_vals, col_vals)
    
xvals = np.array([xx[rid] for rid in fit_roi_list])
yvals = np.array([yy[rid] for rid in fit_roi_list])
zs = np.array([max_zscores[rid] for rid in fit_roi_list])

ax.scatter(xvals, yvals, c=zs, marker='o', alpha=0.5, s=zs*100, cmap='inferno', vmin=0, vmax=6)




#%%
import statsmodels as sm
import matplotlib as mpl


# Plot KDE:
j = sns.jointplot(xvals, yvals, kind='kde', xlim=(screen_left, screen_right), ylim=(screen_bottom, screen_top))
elev_x, elev_y = j.ax_marg_y.lines[0].get_data()
azim_x, azim_y = j.ax_marg_x.lines[0].get_data()

smstats_kde_az = sp.stats.gaussian_kde(xvals) #, weights=mean_fits)
#az_vals = np.linspace(screen_left, screen_right, len(mean_mags))
az_vals = np.linspace(screen_left, screen_right, len(xvals))

smstats_kde_el = sp.stats.gaussian_kde(yvals)
#el_vals = np.linspace(screen_lower, screen_upper, len(mean_mags))
el_vals = np.linspace(screen_bottom, screen_top, len(yvals))


smstats_az = smstats_kde_az(az_vals)
smstats_el = smstats_kde_el(el_vals)
    #wa = kdea(vals)
#    fig, ax = pl.subplots() #pl.figure()
#    ax.plot(vals, wa)
#    ax.plot(azim_x, azim_y)


# 2. Use weights with KDEUnivariate (no FFT):
#weighted_kde_az = sm.nonparametric.kde.KDEUnivariate(linX.values)
weighted_kde_az = sm.nonparametric.kde.KDEUnivariate(xvals)
weighted_kde_az.fit(weights=zs, fft=False)
#weighted_kde_el = sm.nonparametric.kde.KDEUnivariate(linY.values)
weighted_kde_el = sm.nonparametric.kde.KDEUnivariate(yvals)
weighted_kde_el.fit(weights=zs, fft=False)

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

pl.savefig(os.path.join(output_dir, '%s_compare_kde_weighted_fit_thr_%.2f.png' % (plot_str, fit_thr) ))
        

# Plot weighted KDE to marginals on joint plot:
j.ax_marg_y.plot(weighted_kde_el.density, weighted_kde_el.support, color='orange', label='weighted')
j.ax_marg_x.plot(weighted_kde_az.support, weighted_kde_az.density, color='orange', label='weighted')
j.ax_marg_x.set_ylim([0, max([j.ax_marg_x.get_ylim()[-1], weighted_kde_az.density.max()]) + 0.005])
j.ax_marg_y.set_xlim([0, max([j.ax_marg_y.get_xlim()[-1], weighted_kde_el.density.max()]) + 0.005])
j.ax_marg_x.legend(fontsize=8)

j.savefig(os.path.join(output_dir, '%s_weighted_marginals_fit_thr_%.2f.png' % (plot_str, fit_thr) ))



from pipeline.python.retinotopy import target_visual_field as targ

#%%
kde_az =  weighted_kde_az.density.copy()
vals_az = weighted_kde_az.support.copy()

kde_el = weighted_kde_el.density.copy()
vals_el = weighted_kde_el.support.copy()

az_max, az_min1, az_min2, az_maxima, az_minima = targ.find_local_min_max(vals_az, kde_az)
el_max, el_min1, el_min2, el_maxima, el_minima = targ.find_local_min_max(vals_el, kde_el)



fig, axes = pl.subplots(1,2, figsize=(10,5)) #pl.figure();
targ.plot_kde_min_max(vals_az, kde_az, maxval=az_max, minval1=az_min1, minval2=az_min2, title='azimuth', ax=axes[0])
targ.plot_kde_min_max(vals_el, kde_el, maxval=el_max, minval1=el_min1, minval2=el_min2, title='elevation', ax=axes[1])

label_figure(fig, data_identifier)
fig.savefig(os.path.join(output_dir, '%s_weighted_kde_min_max_fit_thr_%.2f.png' % (plot_str, fit_thr)))
    
az_bounds = sorted([float(vals_az[az_min1]), float(vals_az[az_min2])])
el_bounds = sorted([float(vals_el[el_min1]), float(vals_el[el_min2])])
# Make sure bounds are within screen:
if az_bounds[0] < screen_left:
    az_bounds[0] = screen_left
if az_bounds[1] > screen_right:
    az_bounds[1] = screen_right
if el_bounds[0] < screen_bottom:
    el_bounds[0] = screen_bottom
if el_bounds[1] > screen_top:
    el_bounds[1] = screen_top
    
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

#%%


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


zs = np.array([max_zscores[rid] for rid in fit_roi_list])


min_thr = 0.01
#marker_scale = 100./round(magratio.mean().mean(), 3)
fig, strong_cells = plot_kde_maxima(kde_results, zs, xvals, yvals, screen, \
                      use_peak=True, exclude_bad=False, min_thr=min_thr, marker_scale=100)


print("LINX:", xvals.shape)


for ri in strong_cells:
    fig.axes[0].text(xvals[ri], yvals[ri], '%s' % (fit_roi_list[ri]+1))
label_figure(fig, data_identifier)
pl.savefig(os.path.join(output_dir, '%s_centroid_peak_rois_by_pos_fit_thr_%.2f.png' % (plot_str, fit_thr)))


#    fig = plot_kde_maxima(kde_results, magratio, linX, linY, screen, use_peak=False, marker_scale=marker_scale)
#    print("LINX:", linX.shape)
#    for ri in strong_cells:
#        fig.axes[0].text(linX[ri], linY[ri], '%s' % (ri+1))
#    label_figure(fig, data_identifier)
#    pl.savefig(os.path.join(output_dir, 'centroid_kdecenter_rois_by_pos_%s.png' % (loctype)))
    
with open(os.path.join(output_dir, '%s_fit_centroid_results_fit_thr_%.2f.json' % (plot_str, fit_thr)), 'w') as f:
    json.dump(kde_results, f, sort_keys=True, indent=4)



    