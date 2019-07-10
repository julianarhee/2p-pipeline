#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 16:37:13 2019

@author: julianarhee
"""

#import matplotlib as mpl
#mpl.use('Agg')
import os
import h5py
import json
import traceback
#import re
#import sys
#import datetime
#import optparse
import pprint
import copy

#import cPickle as pkl
#import tifffile as tf
import pylab as pl
import numpy as np
#from scipy import ndimage
#import cv2
import glob
#from scipy.optimize import curve_fit
#import seaborn as sns
#from pipeline.python.retinotopy import visualize_rois as vis
#from matplotlib.patches import Ellipse
#from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy as sp
import pandas as pd


from pipeline.python.utils import natural_keys, label_figure, replace_root
#from matplotlib.patches import Ellipse, Rectangle

pp = pprint.PrettyPrinter(indent=4)
#from scipy.signal import argrelextrema


#%%

# -----------------------------------------------------------------------------
# General utitilies often used for retino-related analysis
# -----------------------------------------------------------------------------

def convert_values(oldval, newmin=None, newmax=None, oldmax=None, oldmin=None):
    oldrange = (oldmax - oldmin)
    newrange = (newmax - newmin)
    newval = (((oldval - oldmin) * newrange) / oldrange) + newmin
    return newval


def correct_phase_wrap(phase):
        
    corrected_phase = phase.copy()
    
    corrected_phase[phase<0] =- phase[phase<0]
    corrected_phase[phase>0] = (2*np.pi) - phase[phase>0]
    
    return corrected_phase

def select_rois(mean_magratios, mag_thr=0.02, mag_thr_stat='max'):

    if mag_thr_stat == 'mean':
        best_mags = mean_magratios.loc[mean_magratios.mean(axis=1)>mag_thr].index.tolist()
    elif mag_thr_stat == 'max':
        best_mags = mean_magratios.loc[mean_magratios.max(axis=1)>mag_thr].index.tolist()
    else:
        mag_thr_stat == 'allconds'
        best_mags = [rid for rid in mean_magratios.index.tolist() if all(mean_magratios.iloc[rid] > mag_thr)]
    top_rois = copy.copy(np.array(best_mags))

    return top_rois


#%%

# -----------------------------------------------------------------------------
# Data formatting
# -----------------------------------------------------------------------------

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
                print("No analysis found for file: %s" % tifnum)
                excluded_tifs.append(tifnum)
        trials_by_cond[cond] = [t for t in tif_list if t not in excluded_tifs]
    print("TRIALS BY COND:")
    print(trials_by_cond) 
    trial_list = [int(t) for t in conds.keys() if int(t) not in excluded_tifs]
    print("Trials:", trial_list)

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


def load_retino_analysis_info(animalid, session, fov, run, retinoid, use_pixels=False, rootdir='/n/coxfs01/2p-data'):
    
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
        
    return retinoid, rids[retinoid]

#%%
# Convert degs to centimeters:
def get_linear_coords(width, height, resolution, leftedge=None, rightedge=None, bottomedge=None, topedge=None):
    #width = 103 # in cm
    #height = 58 # in cm
    #resolution = [1920, 1080]

    if leftedge is None:
        leftedge = -1*width/2.
    if rightedge is None:
        rightedge = width/2.
    if bottomedge is None:
        bottomedge = -1*height/2.
    if topedge is None:
        topedge = height/2.

    print("center 2 Top/Anterior:", topedge, rightedge)


    mapx = np.linspace(leftedge, rightedge, resolution[0] * ((rightedge-leftedge)/float(width)))
    mapy = np.linspace(bottomedge, topedge, resolution[1] * ((topedge-bottomedge)/float(height)))

    lin_coord_x, lin_coord_y = np.meshgrid(mapx, mapy, sparse=False)

    return lin_coord_x, lin_coord_y


def get_retino_info(animalid, session, fov=None, interactive=True, rootdir='/n/coxfs01/2p-data',
                    azimuth='right', elevation='top',
                    leftedge=None, rightedge=None, bottomedge=None, topedge=None):

    screen_info = get_screen_info(animalid, session, fov=fov, interactive=interactive,
                                  rootdir=rootdir)

    lin_coord_x, lin_coord_y = get_linear_coords(screen_info['azimuth'], 
                                                 screen_info['elevation'], 
                                                 screen_info['resolution'], 
                                                 leftedge=leftedge, rightedge=rightedge, 
                                                 bottomedge=bottomedge, topedge=topedge)
    
    linminW = lin_coord_x.min(); linmaxW = lin_coord_x.max()
    linminH = lin_coord_y.min(); linmaxH = lin_coord_y.max()

        
        
    retino_info = {}
    retino_info['width'] = screen_info['azimuth']
    retino_info['height'] = screen_info['elevation']
    retino_info['resolution'] = screen_info['resolution']
    #aspect_ratio = float(height)/float(width)
    retino_info['aspect'] = retino_info['height'] / retino_info['width']#aspect_ratio
    retino_info['azimuth'] = azimuth
    retino_info['elevation'] = elevation
    retino_info['linminW'] = linminW
    retino_info['linmaxW'] = linmaxW
    retino_info['linminH'] = linminH
    retino_info['linmaxH'] = linmaxH
    retino_info['bounding_box'] = [leftedge, bottomedge, rightedge, topedge]

    return retino_info

def format_retino_traces(data_fpath, info=None, trace_type='corrected'):
    f = h5py.File(data_fpath, 'r')
    
    labels_list = []
    nrois, nframes_per = f[f.keys()[0]][trace_type].shape
    dtype = f[f.keys()[0]][trace_type].dtype
    nfiles = len(f.keys())
    tmat = np.empty((nframes_per*nfiles, nrois), dtype=dtype)
    s_ix = 0
    for tix, tfile in enumerate(sorted(f.keys(), key=natural_keys)):
        tmat[s_ix:s_ix+nframes_per, :] = f[tfile][trace_type][:].T
        
        if info is not None:
            trials_by_cond = info['trials']
            # Condition info:
            curr_cond = [cond for cond, trial_list in trials_by_cond.items() if int(tix+1) in trial_list][0]
            curr_trial = 'trial%05d' % int(tix+1)
            frame_tsecs = info['frame_tstamps_sec']
            if info['nchannels'] == 2:
                frame_tsecs = frame_tsecs[0::2]
                
            curr_labels = pd.DataFrame({'tsec': frame_tsecs,
                                        'config': [curr_cond for _ in range(len(frame_tsecs))],
                                        'trial': [curr_trial for _ in range(len(frame_tsecs))],
                                        'frame': [i for i in range(len(frame_tsecs))]
                                        })
            labels_list.append(curr_labels)
            
        s_ix = s_ix + nframes_per
        
    traces = pd.DataFrame(tmat)
    labels = pd.concat(labels_list, axis=0).reset_index(drop=True)
    
    return traces, labels
    
    
    
def get_protocol_info(animalid, session, fov, run='retino_run1', rootdir='/n/coxfs01/2p-data'):
    
    run_dir = os.path.join(rootdir, animalid, session, fov, run)
    mw_fpath = glob.glob(os.path.join(run_dir, 'paradigm', 'files', 'parsed*.json'))[0]
    with open(mw_fpath, 'r') as f:
        mwinfo = json.load(f)
    
    si_fpath = glob.glob(os.path.join(run_dir, '*.json'))[0]
    with open(si_fpath, 'r') as f:
        scaninfo = json.load(f)
    
    conditions = list(set([cdict['stimuli']['stimulus'] for trial_num, cdict in mwinfo.items()]))
    trials_by_cond = dict((cond, [int(k) for k, v in mwinfo.items() if v['stimuli']['stimulus']==cond]) \
                           for cond in conditions)
    print(trials_by_cond)
    n_frames = scaninfo['nvolumes']
    fr = scaninfo['frame_rate']
        

    stiminfo = dict((cond, dict()) for cond in conditions)
    curr_cond = conditions[0]
    # get some info from paradigm and run file
    stimfreq = np.unique([v['stimuli']['scale'] for k,v in mwinfo.items() if v['stimuli']['stimulus']==curr_cond])[0]
    stimperiod = 1./stimfreq # sec per cycle
    
    n_cycles = int(round((n_frames/fr) / stimperiod))
    n_frames_per_cycle = int(np.floor(stimperiod * fr))
    cycle_starts = np.round(np.arange(0, n_frames_per_cycle * n_cycles, n_frames_per_cycle)).astype('int')


    stiminfo = {'stimfreq': stimfreq,
               'frame_rate': fr,
               'n_reps': len(trials_by_cond[curr_cond]),
               'nframes': n_frames,
               'n_cycles': n_cycles,
               'n_frames_per_cycle': n_frames_per_cycle,
               'cycle_start_ixs': cycle_starts,
                }
    
    scaninfo.update({'stimulus': stiminfo})
    scaninfo.update({'trials': trials_by_cond})


    return scaninfo


def get_retino_stimulus_info(mwinfo, runinfo):
    conditions = list(set([cdict['stimuli']['stimulus'] for trial_num, cdict in mwinfo.items()]))
    trials_by_cond = dict((cond, [int(k) for k, v in mwinfo.items() if v['stimuli']['stimulus']==cond]) \
                           for cond in conditions)
    
    stiminfo = dict((cond, dict()) for cond in conditions)
    for curr_cond in conditions:
        # get some info from paradigm and run file
        stimfreq = np.unique([v['stimuli']['scale'] for k,v in mwinfo.items() if v['stimuli']['stimulus']==curr_cond])[0]
        stimperiod = 1./stimfreq # sec per cycle
        
        n_frames = runinfo['nvolumes']
        fr = runinfo['frame_rate']
        
        n_cycles = int(round((n_frames/fr) / stimperiod))
        #print n_cycles

        n_frames_per_cycle = int(np.floor(stimperiod * fr))
        cycle_starts = np.round(np.arange(0, n_frames_per_cycle * n_cycles, n_frames_per_cycle)).astype('int')

        stiminfo[curr_cond] = {'stimfreq': stimfreq,
                               'frame_rate': fr,
                               'n_reps': len(trials_by_cond[curr_cond]),
                               'nframes': n_frames,
                               'n_cycles': n_cycles,
                               'n_frames_per_cycle': n_frames_per_cycle,
                               'cycle_start_ixs': cycle_starts,
                               #'trials_by_cond': trials_by_cond,
                               'frame_tstamps': runinfo['frame_tstamps_sec']
                              }

    return stiminfo, trials_by_cond


# Interpolate bar position for found SI frame using upsampled MW tstamps and positions:
def get_interp_positions(condname, mwinfo, stiminfo, trials_by_cond):
    mw_fps = 1./np.diff(np.array(mwinfo[str(trials_by_cond[condname][0])]['stiminfo']['tstamps'])/1E6).mean()
    si_fps = stiminfo[condname]['frame_rate']
    print("[%s]: Downsampling MW positions (sampled at %.2fHz) to SI frame rate (%.2fHz)" % (condname, mw_fps, si_fps))

    si_cyc_ixs = stiminfo[condname]['cycle_start_ixs']
    assert len(np.unique(np.diff(si_cyc_ixs))) == 1, "Uneven cycle durs found!\n--> %s" % str(np.diff(si_cyc_ixs))
    fr_per_cyc = int(np.unique(np.diff(si_cyc_ixs))[0])
    
    si_tstamps = stiminfo[condname]['frame_tstamps']


    #fig, axes = pl.subplots(1, len(trials_by_cond[condname]))

    stim_pos_list = []
    stim_tstamp_list = []

    for ti, trial in enumerate(trials_by_cond[condname]):
        #ax = axes[ti]

        pos_list = []
        tstamp_list = []
        mw_cyc_ixs = mwinfo[str(trial)]['stiminfo']['start_indices']
        for cix in np.arange(0, len(mw_cyc_ixs)):
            if cix==len(mw_cyc_ixs)-1:
                mw_ts = [t/1E6 for t in mwinfo[str(trial)]['stiminfo']['tstamps'][mw_cyc_ixs[cix]:]]
                xs = mwinfo[str(trial)]['stiminfo']['values'][mw_cyc_ixs[cix]:]
                si_ts = si_tstamps[si_cyc_ixs[cix]:si_cyc_ixs[cix]+fr_per_cyc]
            else:
                mw_ts = np.array([t/1E6 for t in mwinfo[str(trial)]['stiminfo']['tstamps'][mw_cyc_ixs[cix]:mw_cyc_ixs[cix+1]]])
                xs = np.array(mwinfo[str(trial)]['stiminfo']['values'][mw_cyc_ixs[cix]:mw_cyc_ixs[cix+1]])
                si_ts = si_tstamps[si_cyc_ixs[cix]:si_cyc_ixs[cix+1]]

            recentered_mw_ts = [t-mw_ts[0] for t in mw_ts]
            recentered_si_ts = [t-si_ts[0] for t in si_ts]

            # Since MW tstamps are linear, SI tstamps linear, interpolate position values down to SI's lower framerate:
            interpos = sp.interpolate.interp1d(recentered_mw_ts, xs, fill_value='extrapolate')
            resampled_xs = interpos(recentered_si_ts)

            pos_list.append(pd.Series(resampled_xs, name=trial))
            tstamp_list.append(pd.Series(recentered_si_ts, name=trial))

            #ax.plot(recentered_mw_ts, xs, 'ro', alpha=0.5, markersize=2)
            #ax.plot(recentered_si_ts, resampled_xs, 'bx', alpha=0.5, markersize=2)

        pos_vals = pd.concat(pos_list, axis=0).reset_index(drop=True) 
        tstamp_vals = pd.concat(tstamp_list, axis=0).reset_index(drop=True)

        stim_pos_list.append(pos_vals)
        stim_tstamp_list.append(tstamp_vals)

    stim_positions = pd.concat(stim_pos_list, axis=1)
    stim_tstamps = pd.concat(stim_tstamp_list, axis=1)


    return stim_positions, stim_tstamps


#%%
def get_screen_info(animalid, session, fov=None, interactive=True, rootdir='/n/coxfs01/2p-data'):
        
    screen = {}
    
    try:
        # Get bounding box values from epi:
        epi_session_paths = sorted(glob.glob(os.path.join(rootdir, animalid, 'epi_maps', '20*')), key=natural_keys)
        epi_sessions = sorted([os.path.split(s)[-1].split('_')[0] for s in epi_session_paths], key=natural_keys)
        print("Found epi sessions: %s" % str(epi_sessions))
        if len(epi_sessions) > 0:
            epi_sesh = [datestr for datestr in sorted(epi_sessions, key=natural_keys) if int(datestr) <= int(session)][-1] # Use most recent session
            print("Most recent: %s" % str(epi_sesh))

            epi_fpaths = glob.glob(os.path.join(rootdir, animalid, 'epi_maps', '*%s*' % epi_sesh, 'screen_boundaries*.json'))
            if len(epi_fpaths) == 0:
                epi_fpaths = glob.glob(os.path.join(rootdir, animalid, 'epi_maps', '*%s*' % epi_sesh, '*', 'screen_boundaries*.json'))

        else:
            print("No EPI maps found for session: %s * (trying to use tmp session boundaries file)")
            epi_fpaths = glob.glob(os.path.join(rootdir, animalid, 'epi_maps', 'screen_boundaries*.json'))
        
        assert len(epi_fpaths) > 0, "No epi screen info found!"
        
        # Each epi run should have only 2 .json files (1 for each condition):
        if len(epi_fpaths) > 2:
            print("-- found %i screen boundaries files: --" % len(epi_fpaths))
            repeat_epi_sessions = sorted(list(set( [os.path.split(s)[0] for s in epi_fpaths] )), key=natural_keys)
            for ei, repeat_epi in enumerate(sorted(repeat_epi_sessions, key=natural_keys)):
                print(ei, repeat_epi)
            if interactive:
                selected_epi = input("Select IDX of epi run to use: ")
            else:
                assert fov is not None, "ERROR: not interactive, but no FOV specified and multiple epis for session %s" % session
                
                selected_fovs = [fi for fi, epi_session_name in enumerate(repeat_epi_sessions) if fov in epi_session_name]
                print("Found FOVs: %s" % str(selected_fovs))

                if len(selected_fovs) == 0:
                    selected_epi = sorted(selected_fovs, key=natural_keys)[-1]
                else:
                    selected_epi = selected_fovs[0]
                
            epi_fpaths = [s for s in epi_fpaths if repeat_epi_sessions[selected_epi] in s]
        
        print("-- getting screen info from:", epi_fpaths)
        
        for epath in epi_fpaths:
            with open(epath, 'r') as f:
                epi = json.load(f)
            
            screen['azimuth'] = epi['screen_params']['screen_size_x_degrees']
            screen['elevation'] = epi['screen_params']['screen_size_t_degrees']
            screen['resolution'] = [epi['screen_params']['screen_size_x_pixels'], epi['screen_params']['screen_size_y_pixels']]

            if 'screen_boundaries' in epi.keys():
                if 'boundary_left_degrees' in epi['screen_boundaries'].keys():
                    screen['bb_left'] = epi['screen_boundaries']['boundary_left_degrees']
                    screen['bb_right'] = epi['screen_boundaries']['boundary_right_degrees']
                elif 'boundary_down_degrees' in epi['screen_boundaries'].keys():
                    screen['bb_lower'] = epi['screen_boundaries']['boundary_down_degrees']
                    screen['bb_upper'] = epi['screen_boundaries']['boundary_up_degrees']
            
            else:
                screen['bb_lower'] = -1*screen['elevation']/2.0
                screen['bb_upper'] = screen['elevation']/2.0
                screen['bb_left']  = -1*screen['azimuth']/2.0
                screen['bb_right'] = screen['azimuth']/2.0


        print("*********************************")
        pp.pprint(screen)
        print("*********************************")
      
    except Exception as e:
        traceback.print_exc()
        
    return screen

#%%

# -----------------------------------------------------------------------------
# plotting
# -----------------------------------------------------------------------------


def plot_roi_traces_by_cond(roi_traces, stiminfo):
        
    fig, axes = pl.subplots(2,2, sharex=True, sharey=True, figsize=(10,6)) #, figsize=(8,))
    
    for aix, cond in enumerate(['left', 'right', 'bottom', 'top']):
        ax = axes.flat[aix]
        for repnum in np.arange(0, roi_traces[cond].shape[0]):
            ax.plot(roi_traces[cond][repnum, :], 'k', alpha=0.5, lw=0.5)
        ax.plot(roi_traces[cond].mean(axis=0), 'k', linewidth=1)
        ax.set_title(cond)
        for cyc in stiminfo[cond]['cycle_start_ixs']:
            ax.axvline(x=cyc, color='k', linestyle='--', linewidth=0.5)
            
    return fig


