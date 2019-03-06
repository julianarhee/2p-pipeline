#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 20:22:42 2019

@author: julianarhee
"""


import os
import glob
import json
import h5py
import optparse
import sys

import pandas as pd
import pylab as pl
import seaborn as sns
import numpy as np

from pipeline.python.utils import natural_keys, label_figure

from mpl_toolkits.axes_grid1 import make_axes_locatable
from pipeline.python.retinotopy import target_visual_field as targ


#%%
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
    parser.add_option('-t', '--retinoid', action='store', dest='retinoid', default='analysis001', \
                      help="name of retino ID (roi analysis) [default: analysis001]")
    
    parser.add_option('--angular', action='store_false', dest='use_linear', default=True, \
                      help="Plot az/el coordinates in angular spce [default: plots linear coords]")
    parser.add_option('-e', '--thr-el', action='store', dest='fit_thresh_el', default=0.2, \
                      help="fit threshold for elevation [default: 0.2]")
    parser.add_option('-a', '--thr-az', action='store', dest='fit_thresh_az', default=0.2, \
                      help="fit threshold for azimuth [default: 0.2]")

    parser.add_option('--pixels', action='store_true', dest='use_pixels', default=False, \
                      help="Do pixel-based analysis")
    
    (options, args) = parser.parse_args(options)

    return options

#%%

def correct_phase_wrap(phase):
        
    corrected_phase = phase.copy()
    
    corrected_phase[phase<0] =- phase[phase<0]
    corrected_phase[phase>0] = (2*np.pi) - phase[phase>0]
    
    return corrected_phase

def reverse_phase_wrap(corrected_phase):
    phase = corrected_phase.copy()
    
    phase[corrected_phase>np.pi] = (corrected_phase[corrected_phase > np.pi] + (2*np.pi)) * -1
    phase[corrected_phase<np.pi] = -corrected_phase[corrected_phase < np.pi]
    
    return phase

def load_masks(processed_fpath):
    pfile = h5py.File(processed_fpath)
    masks = pfile['masks'][:]
    pfile.close()
    return masks

#%%

#options = ['-i', 'JC047', '-S', '20190215', '-A', 'FOV1', '--pixels']
options = ['-i', 'JC059', '-S', '20190227', '-A', 'FOV1', '--pixels']


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
    use_pixels = opts.use_pixels
    if use_pixels:
        print("-- using pixel-based analysis --")
    else:
        print("-- using ROI-based analysis --")
        
    
    #%%
    
    run_dir = glob.glob(os.path.join(rootdir, animalid, session, '%s*' % fov, run))[0]
    fov = os.path.split(os.path.split(run_dir)[0])[-1]
    print("FOV: %s, run: %s" % (fov, run))
    retinoids_fpath = glob.glob(os.path.join(run_dir, 'retino_analysis', 'analysisids_*.json'))[0]
    with open(retinoids_fpath, 'r') as f:
        rids = json.load(f)
    if use_pixels:
        roi_analyses = [r for r, rinfo in rids.items() if rinfo['PARAMS']['roi_type'] == 'pixels']
    else:
        roi_analyses = [r for r, rinfo in rids.items() if rinfo['PARAMS']['roi_type'] != 'pixels']
    if retinoid not in roi_analyses:
        retinoid = sorted(roi_analyses, key=natural_keys)[-1] # use most recent roi analysis
        print("Fixed retino id to most recent: %s" % retinoid)
    
    data_identifier = '|'.join([animalid, session, fov, run, retinoid])
    
    print("*** Dataset: %s ***" % data_identifier)
    
    #%%
    
    processed_dir = glob.glob(os.path.join(run_dir, 'retino_analysis', '%s*' % retinoid))[0]
    
    processed_fpaths = glob.glob(os.path.join(processed_dir, 'files', '*.h5'))
    print("Found %i processed retino runs." % len(processed_fpaths))
    
    conditions_fpath = glob.glob(os.path.join(run_dir, 'paradigm', 'files', '*.json'))[0]


    #%%
    
    # Comine all trial data into data frames:
    fit, magratio, phase, trials_by_cond = targ.trials_to_dataframes(processed_fpaths, conditions_fpath)
    print fit.head()
    print trials_by_cond


    
#%%

    # Compare L/R and U/D to check for delay in GCaMP signal:
    az = ['right', 'left']
    el = ['top', 'bottom']
        
    phase_right = correct_phase_wrap(phase[trials_by_cond['right']])
    phase_left = correct_phase_wrap(phase[trials_by_cond['left']])
    phase_top = correct_phase_wrap(phase[trials_by_cond['top']])
    phase_bottom = correct_phase_wrap(phase[trials_by_cond['bottom']])
    
    avg_right = phase_right.mean(axis=1)
    avg_left = phase_left.mean(axis=1)
    avg_top = phase_top.mean(axis=1)
    avg_bottom = phase_bottom.mean(axis=1)
    
    delay_az = (avg_right + avg_left ) / 2.
    delay_el = (avg_top + avg_bottom ) / 2.

    overall_mag_ratios = magratio.mean(axis=1)
    print(overall_mag_ratios.shape)
    
    cmap = 'nipy_spectral'
    vmin = 0
    vmax = 2*np.pi
    d1 = int(np.sqrt(overall_mag_ratios.shape[0]))
    d2 = int(np.sqrt(overall_mag_ratios.shape[0]))
    
    
    #%%%
    for direction in [az, el]:
        assert len(direction) == 2, "Not enough directions found: %s" % str(direction)
                
        nreps = len(phase_right.columns.tolist())
        fig, axes = pl.subplots(2, nreps+1,figsize=(5*(nreps+1), 10))

        for cond_ix, cond in enumerate(direction):
            if cond == 'right':
                nreps = len(phase_right.columns.tolist())
                trial_phase_vals = phase_right.copy()
                avg_phase_vals = avg_right.copy()
            elif cond == 'left':
                nreps = len(phase_left.columns.tolist())
                trial_phase_vals = phase_left.copy()
                avg_phase_vals = avg_left.copy()
            elif cond == 'top':
                nreps = len(phase_top.columns.tolist())
                trial_phase_vals = phase_top.copy()
                avg_phase_vals = avg_top.copy()
            else:
                nreps = len(phase_bottom.columns.tolist())
                trial_phase_vals = phase_bottom.copy()
                avg_phase_vals = avg_bottom.copy()
                    
            
            for trialnum in np.arange(0, nreps):
                axes[cond_ix, trialnum].imshow(trial_phase_vals[trials_by_cond[cond][trialnum]].reshape((d1, d2)), cmap=cmap, vmin=vmin, vmax=vmax)
                axes[cond_ix, trialnum].set_title('%s, rep %i' % (cond, trialnum))
                axes[cond_ix, trialnum].imshow(trial_phase_vals[trials_by_cond[cond][trialnum]].reshape((d1, d2)), cmap=cmap, vmin=vmin, vmax=vmax)
                axes[cond_ix, trialnum].set_title('%s, rep %i' % (cond, trialnum))
                axes[cond_ix, trialnum].axis('off')

                axes[cond_ix, nreps].imshow(avg_phase_vals.reshape((d1, d2)), cmap=cmap, vmin=vmin, vmax=vmax)
                axes[cond_ix, nreps].set_title('%s, average' % cond)
                axes[cond_ix, nreps].axis('off')
            
    #%%
    
    phasemap_right = np.reshape(avg_right, (d1, d2))
    print(phasemap_right).shape
    phasemap_left = np.reshape(avg_left, (d1, d2))
    
    az_delay = (phasemap_left + phasemap_right) / 2.
    
    absolute_right = (phasemap_right - phasemap_left) / 2.
    
    absolute_left = (phasemap_left - phasemap_right) / 2.
    left_subtract_delay = (phasemap_left - az_delay)
    right_subtract_delay = (az_delay - phasemap_right)
    
    fig, ax = pl.subplots(3,3)
    im0 = ax[0, 0].imshow(phasemap_right, cmap=cmap, vmin=vmin, vmax=vmax)
    ax[0, 0].set_title('right'); ax[0, 0].axis('off');
    im1 = ax[0, 1].imshow(phasemap_left, cmap=cmap, vmin=vmin, vmax=vmax)
    ax[0, 1].set_title('left'); ax[0, 1].axis('off')
    ax[0, 2].imshow(az_delay, cmap=cmap, vmin=vmin, vmax=vmax)
    ax[0, 2].set_title('delay'); ax[0, 2].axis('off')
    
    ax[1, 0].imshow(absolute_right, cmap=cmap, vmin=vmin, vmax=vmax)
    ax[1, 1].imshow(absolute_left, cmap=cmap, vmin=vmin, vmax=vmax)

    ax[2, 0].imshow(right_subtract_delay, cmap=cmap, vmin=vmin, vmax=vmax)
    ax[2, 1].imshow(left_subtract_delay, cmap=cmap, vmin=vmin, vmax=vmax)

    pl.subplots_adjust(bottom=0.05, top=0.85)
    # Now adding the colorbar
    cbaxes = fig.add_axes([0.1, 0.95, 0.5, 0.05]) 
    cb = pl.colorbar(im1, cax = cbaxes, orientation='horizontal')  
    cb.ax.invert_xaxis() 
    
    #%%
    threshold = True
    threshold_value = 0.03
        
    # load masks
    if not use_pixels:
        masks = load_masks(processed_fpaths[0])
        print(masks.shape)
        masks[masks>0] = 1
        nrois = masks.shape[0]
    
        if threshold:
            roi_list = [ri for ri in overall_mag_ratios.index.tolist() if overall_mag_ratios.iloc[ri]>0.05]
        else:
            roi_list = np.arange(0, nrois)
    else:
        if threshold:
            overall_mag_ratios = overall_mag_ratios[overall_mag_ratios >= threshold_value]
            
            
    #%%
    masks_right = np.array([masks[ri,:,:] * avg_right.values[ri] for ri in roi_list])
    masks_left = np.array([masks[ri,:,:] * avg_left.values[ri] for ri in roi_list])
    masks_bottom = np.array([masks[ri,:,:] * avg_bottom.values[ri] for ri in roi_list])
    masks_top = np.array([masks[ri,:,:] * avg_top.values[ri] for ri in roi_list])

    masks_delay_az = np.array([masks[ri,:,:] * delay_az.values[ri] for ri in roi_list])
    masks_delay_el = np.array([masks[ri,:,:] * delay_el.values[ri] for ri in roi_list])

    
    fig, axes = pl.subplots(2,3)
    axes[0,0].imshow(masks_right.sum(axis=0), cmap='nipy_spectral', vmin=0, vmax=2*np.pi)
    axes[0,1].imshow(masks_left.sum(axis=0), cmap='nipy_spectral', vmin=0, vmax=2*np.pi)
    axes[0,2].imshow(masks_delay_az.sum(axis=0), cmap='nipy_spectral', vmin=0, vmax=2*np.pi)

    axes[1,0].imshow(masks_top.sum(axis=0), cmap='nipy_spectral', vmin=0, vmax=2*np.pi)
    axes[1,1].imshow(masks_bottom.sum(axis=0), cmap='nipy_spectral', vmin=0, vmax=2*np.pi)
    axes[1,2].imshow(masks_delay_el.sum(axis=0), cmap='nipy_spectral', vmin=0, vmax=2*np.pi)

    #%%
    
    fig, axes = pl.subplots(1,2) #.figure()
    axes[0].scatter(avg_left.values, avg_right.values)
    axes[0].plot([0, avg_left.max()], [avg_right.max(), 0], 'r--')
    axes[0].set_xlabel('left')
    axes[0].set_ylabel('right')
    axes[0].set_title('phase')
    for ri in roi_list:
        axes[0].text(avg_left.values[ri], avg_right.values[ri], '%s' % ri)
    axes[0].invert_yaxis()

    axes[1].scatter(avg_bottom.values, avg_top.values)
    axes[1].plot([0, avg_bottom.max()], [avg_top.max(), 0], 'r--')
    axes[1].set_xlabel('bottom')
    axes[1].set_ylabel('top')
    axes[1].set_title('phase')
    for ri in roi_list:
        axes[1].text(avg_bottom.values[ri], avg_top.values[ri], '%s' % ri)
    axes[1].invert_yaxis()
    
    
    for ri in roi_list:
        print ri, overall_mag_ratios.iloc[ri]
