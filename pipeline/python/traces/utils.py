#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 20:23:06 2018

@author: juliana
"""
#import matplotlib
#matplotlib.use('Agg')
import os
import re
import sys
import json
import optparse
import operator
import h5py
import pprint
import itertools
import time
import shutil
import datetime
import traceback
import glob

from scipy import stats
import pandas as pd
import seaborn as sns
import pylab as pl
import numpy as np
#from pipeline.python.utils import natural_keys, hash_file_read_only, print_elapsed_time, hash_file
pp = pprint.PrettyPrinter(indent=4)
#from pipeline.python.classifications import test_responsivity as resp

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]


def get_metric_set(traceid_dir, filter_pupil=True, pupil_radius_min=None, pupil_radius_max=None,
                       pupil_dist_thr=None, auto=False):

    metrics_dir = os.path.join(traceid_dir, 'metrics')
    metric_list = [m for m in os.listdir(metrics_dir) if os.path.isdir(os.path.join(metrics_dir, m))]
    if filter_pupil is True:
        #metric_desc_base = 'pupil_size%i-dist%i-blinks%i_' % (pupil_size_thr, pupil_dist_thr, pupil_max_nblinks)
        metric_desc_base = 'pupil_rmin%.2f-rmax%.2f-dist%.2f_' % (pupil_radius_min, pupil_radius_max, pupil_dist_thr)

    else:
        metric_desc_base = 'unfiltered_'

    # First check for requested metrics set:
    metric_matches = [m for m in metric_list if metric_desc_base in m]
    try:
        assert len(metric_matches) == 1, "Unable to find unique metrics set with base: %s" % metric_desc_base
        selected_metric = metric_matches[0]
        if auto is True:
            return selected_metric
        else:
            while True:
                # Get confirmation first:
                confirm = raw_input('Selected metric: %s\nUse?  Press <Y> to confirm, <n> to reset: ' % selected_metric)
                if confirm == 'Y':
                    break
                elif confirm == 'n':
                    selected_metric = user_select_metric(metrics_dir)
                    break

    except Exception as e:
        if auto is True:
            # Sort by modified date, and select most-recent:
            metric_list.sort(key=lambda s: os.path.getmtime(os.path.join(metrics_dir, s)), reverse=True)
            selected_metric = metric_list[0]
            print("Using most recent metric set: %s" % selected_metric)
            return selected_metric
        else:
            selected_metric = user_select_metric(metrics_dir)

    return selected_metric



def user_select_metric(metrics_dir):
    # Load particular metrics set:
    metric_list = [m for m in os.listdir(metrics_dir) if os.path.isdir(os.path.join(metrics_dir, m))]

    while True:
        print("Found %i metric sets:" % len(metric_list))
        for mi, metric in enumerate(metric_list):
            print(mi, metric)
        user_choice = input('Select IDX of metric set to view: ')
        selected_metric = metric_list[int(user_choice)]
        print("Viewing metric: %s" % selected_metric)
        with open(os.path.join(metrics_dir, selected_metric, 'pupil_params.json'), 'r') as f:
            pupil_params = json.load(f)
        pp.pprint(pupil_params)
        confirm = raw_input("\nUse this metric? Select <Y> to use, or enter to try again: ")
        if confirm == 'Y':
            break

    return selected_metric


#%
def load_TID(run_dir, trace_id):
    run = os.path.split(run_dir)[-1]
    trace_basedir = os.path.join(run_dir, 'traces')
    try:
        tracedict_path = os.path.join(trace_basedir, 'traceids_%s.json' % run)
        with open(tracedict_path, 'r') as tr:
            tracedict = json.load(tr)
        TID = tracedict[trace_id]
        print("USING TRACE ID: %s" % TID['trace_id'])
        pp.pprint(TID)
    except Exception as e:
        print("Unable to load TRACE params info: %s:" % trace_id)
        print("Aborting with error:")
        print(e)

    return TID
#
#
#def get_frame_info(run_dir):
#    si_info = {}
#
#    run = os.path.split(run_dir)[-1]
#    runinfo_path = os.path.join(run_dir, '%s.json' % run)
#    with open(runinfo_path, 'r') as fr:
#        runinfo = json.load(fr)
#    nfiles = runinfo['ntiffs']
#    file_names = sorted(['File%03d' % int(f+1) for f in range(nfiles)], key=natural_keys)
#
#    # Get frame_idxs -- these are FRAME indices in the current .tif file, i.e.,
#    # removed flyback frames and discard frames at the top and bottom of the
#    # volume should not be included in the indices...
#    frame_idxs = runinfo['frame_idxs']
#    if len(frame_idxs) > 0:
#        print "Found %i frames from flyback correction." % len(frame_idxs)
#    else:
#        frame_idxs = np.arange(0, runinfo['nvolumes'] * len(runinfo['slices']))
#
#    ntiffs = runinfo['ntiffs']
#    file_names = sorted(['File%03d' % int(f+1) for f in range(ntiffs)], key=natural_keys)
#    volumerate = runinfo['volume_rate']
#    framerate = runinfo['frame_rate']
#    nvolumes = runinfo['nvolumes']
#    nslices = int(len(runinfo['slices']))
#    nchannels = runinfo['nchannels']
#
#
#    nslices_full = int(round(runinfo['frame_rate']/runinfo['volume_rate']))
#    nframes_per_file = nslices_full * nvolumes
#
#    # =============================================================================
#    # Get VOLUME indices to assign frame numbers to volumes:
#    # =============================================================================
#    vol_idxs_file = np.empty((nvolumes*nslices_full,))
#    vcounter = 0
#    for v in range(nvolumes):
#        vol_idxs_file[vcounter:vcounter+nslices_full] = np.ones((nslices_full, )) * v
#        vcounter += nslices_full
#    vol_idxs_file = [int(v) for v in vol_idxs_file]
#
#
#    vol_idxs = []
#    vol_idxs.extend(np.array(vol_idxs_file) + nvolumes*tiffnum for tiffnum in range(nfiles))
#    vol_idxs = np.array(sorted(np.concatenate(vol_idxs).ravel()))
#
#    si_info['nslices_full'] = nslices_full
#    si_info['nframes_per_file'] = nframes_per_file
#    si_info['vol_idxs'] = vol_idxs
#    si_info['volumerate'] = volumerate
#    si_info['framerate'] = framerate
#    si_info['nslices'] = nslices
#    si_info['nchannels'] = nchannels
#    si_info['ntiffs'] = ntiffs
#    si_info['frames_tsec'] = runinfo['frame_tstamps_sec']
#    si_info['nvolumes'] = nvolumes
#
#    return si_info


def get_frame_info(run_dir):
    si_info = {}

    run = os.path.split(run_dir)[-1]
    runinfo_path = os.path.join(run_dir, '%s.json' % run)
    with open(runinfo_path, 'r') as fr:
        runinfo = json.load(fr)
    nfiles = runinfo['ntiffs']
    file_names = sorted(['File%03d' % int(f+1) for f in range(nfiles)], key=natural_keys)

    # Get frame_idxs -- these are FRAME indices in the current .tif file, i.e.,
    # removed flyback frames and discard frames at the top and bottom of the
    # volume should not be included in the indices...
    frame_idxs = runinfo['frame_idxs']
    if len(frame_idxs) > 0:
        print("Found %i frames from flyback correction." % len(frame_idxs))
    else:
        frame_idxs = np.arange(0, runinfo['nvolumes'] * len(runinfo['slices']))

    ntiffs = runinfo['ntiffs']
    file_names = sorted(['File%03d' % int(f+1) for f in range(ntiffs)], key=natural_keys)
    volumerate = runinfo['volume_rate']
    framerate = runinfo['frame_rate']
    nvolumes = runinfo['nvolumes']
    nslices = int(len(runinfo['slices']))
    nchannels = runinfo['nchannels']
    nslices_full = int(round(runinfo['frame_rate']/runinfo['volume_rate']))
    nframes_per_file = nslices_full * nvolumes

    # =============================================================================
    # Get VOLUME indices to assign frame numbers to volumes:
    # =============================================================================
    vol_idxs_file = np.empty((nvolumes*nslices_full,))
    vcounter = 0
    for v in range(nvolumes):
        vol_idxs_file[vcounter:vcounter+nslices_full] = np.ones((nslices_full, )) * v
        vcounter += nslices_full
    vol_idxs_file = [int(v) for v in vol_idxs_file]
    vol_idxs = []
    vol_idxs.extend(np.array(vol_idxs_file) + nvolumes*tiffnum for tiffnum in range(nfiles))
    vol_idxs = np.array(sorted(np.concatenate(vol_idxs).ravel()))

    si_info['nslices_full'] = nslices_full
    si_info['nframes_per_file'] = nframes_per_file
    si_info['vol_idxs'] = vol_idxs
    si_info['vol_idxs_file'] = np.array(vol_idxs_file)

    si_info['volumerate'] = volumerate
    si_info['framerate'] = framerate
    si_info['nslices'] = nslices
    si_info['nchannels'] = nchannels
    si_info['ntiffs'] = ntiffs
    si_info['nvolumes'] = nvolumes
    all_frames_tsecs = runinfo['frame_tstamps_sec']
    if nchannels==2:
        all_frames_tsecs = np.array(all_frames_tsecs[0::2])
    si_info['frames_tsec'] = all_frames_tsecs #runinfo['frame_tstamps_sec']

    return si_info




def frames_to_trials(curr_rundir, trials_in_block, file_ix, frame_shift=0,
                     frame_ixs=None, verbose=False):

    '''
    Adapted for CaImAn from pipeline.python.paradigm.trial_alignment.frames_to_trials()
    This is the frame-trial alignment function used by aggregate_experiment_runs()
    
    '''
    parsed_frames_fpath = glob.glob(os.path.join(curr_rundir, 'paradigm', 'parsed_frames_*.hdf5'))[0]
    
    si = get_frame_info(curr_rundir)
    all_frames_tsecs = np.array(si['frames_tsec']) # Corrected frame tstamps (single-channel)
    if verbose:
        print("N tsecs:", len(all_frames_tsecs))
    vol_ixs = si['vol_idxs'] # Volume indices to assign frame numbers to volumes (across all tifs)
    vol_ixs_tif = si['vol_idxs_file']

    try:
        parsed_frames = h5py.File(parsed_frames_fpath, 'r')

        # Get trial list across all tif files
        trial_list = sorted(parsed_frames.keys(), key=natural_keys)
        if file_ix==0:
            print("... getting ixs for %i of %i total trials across all .tif files." % (len(trials_in_block), len(trial_list)))

        # Check if frame indices are indexed relative to full run (all .tif files) or within-tif (i.e., a "block")
        block_indexed = all([all(parsed_frames[t]['frames_in_run'][:] == parsed_frames[t]['frames_in_file'][:])\
                             for t in trial_list]) is False

        # Calculate trial epochs in frames (assumes all trials have same structure)
        min_frame_interval = 1
        nframes_pre = int(round(parsed_frames['trial00001']['frames_in_run'].attrs['baseline_dur_sec'] * si['volumerate']))
        nframes_post = int(round(parsed_frames['trial00001']['frames_in_run'].attrs['iti_dur_sec'] * si['volumerate']))
        nframes_on = int(round(parsed_frames['trial00001']['frames_in_run'].attrs['stim_dur_sec'] * si['volumerate']))
        nframes_per_trial = nframes_pre + nframes_on + nframes_post 
    
        # Get all frame indices for trial epochs (if there are overlapping frame indices, there will be repeats)    
        all_frames_in_trials = np.hstack([np.array(parsed_frames[t]['frames_in_file']) \
                                   for t in trials_in_block])
        if verbose:
            print("*first frame: %i" % all_frames_in_trials[0])

        if verbose:
            print("... N frames to align:", len(all_frames_in_trials))
            print("... N unique frames:", len(np.unique(all_frames_in_trials)))
        stim_onset_idxs = np.array([parsed_frames[t]['frames_in_file'].attrs['stim_on_idx'] \
                                    for t in trials_in_block])
    
        # Adjust frame indices to match within-file, rather than across-files, indices
        if block_indexed is False:
            all_frames_in_trials = all_frames_in_trials - len(all_frames_tsecs)*file_ix - frame_shift  
            if all_frames_in_trials[-1] >= len(all_frames_tsecs):
                print("... File: %i (has %i frames)" % (file_ix, len(all_frames_tsecs)))
                print("... asking for %i extra frames..." % (all_frames_in_trials[-1] - len(all_frames_tsecs)))
            stim_onset_idxs = stim_onset_idxs - len(all_frames_tsecs)*file_ix - frame_shift 
        if verbose:
            print("... Last frame to align: %i (N frames total, %i)" % (all_frames_in_trials[-1], len(all_frames_tsecs)))
    
        stim_onset_idxs_adjusted = vol_ixs_tif[stim_onset_idxs]
        stim_onset_idxs = copy.copy(stim_onset_idxs_adjusted)
        varying_stim_dur=False
        trial_frames_to_vols = dict((t, []) for t in trials_in_block)
        for t in trials_in_block: 
            frames_to_vols = parsed_frames[t]['frames_in_file'][:] 
            frames_to_vols = frames_to_vols - len(all_frames_tsecs)*file_ix - frame_shift  
            actual_frames_in_trial = [i for i in frames_to_vols if i < len(vol_ixs_tif)]
            #trial_vol_ixs = np.empty(frames_to_vols.shape, dtype=int)
            trial_vol_ixs = np.ones(frames_to_vols.shape, dtype=int)*-100 #np.nan
            trial_vol_ixs[0:len(actual_frames_in_trial)] = vol_ixs_tif[actual_frames_in_trial]
            if varying_stim_dur is False:
                trial_vol_ixs = trial_vol_ixs[0:nframes_per_trial]
            trial_frames_to_vols[t] = np.array(trial_vol_ixs)

    except Exception as e:
        traceback.print_exc()
    finally:
        parsed_frames.close()
            
    #%
    # Convert frame-reference to volume-reference. Only select first frame for each volume.
    # Only relevant for multi-plane. Don't take unique values, since stim period of trial N 
    # can be ITI of trial N-1
    actual_frames = np.array([i for i in all_frames_in_trials if i < len(vol_ixs_tif)])
    frames_in_trials = vol_ixs_tif[actual_frames]
    
    # Turn frame_tsecs into RELATIVE tstamps (to stim onset):
    # ------------------------------------------------
    first_plane_tstamps = all_frames_tsecs[np.array(frame_ixs)]
    trial_tstamps = first_plane_tstamps[frames_in_trials[0:len(actual_frames)]] #all_frames_tsecs[frames_in_trials] #indices]  

    # Check whether we are asking for more frames than there are unique, and pad array if so
    if len(trial_tstamps) < len(all_frames_in_trials): #len(frames_in_trials):
        print("... padding trial tstamps array... (should be %i)" % len(all_frames_in_trials))
        trial_tstamps = np.pad(trial_tstamps, (0, len(all_frames_in_trials)-len(trial_tstamps)),\
                               mode='constant', constant_values=-100) #np.nan)
        frames_in_trials = np.pad(frames_in_trials, (0, len(all_frames_in_trials)-len(frames_in_trials)),\
                                  mode='constant', constant_values=-100) #np.nan)

    # All trials have the same structure:
    reformat_tstamps = False
    if verbose:
        print("N frames per trial:", nframes_per_trial)
        print("N tstamps:", len(trial_tstamps))
        print("N trials in block:", len(trials_in_block))
    try: 
        tsec_mat = np.reshape(trial_tstamps, (len(trials_in_block), nframes_per_trial))
        # Subtract stim_on tstamp from each frame of each trial to get relative tstamp:
        tsec_mat -= np.tile(all_frames_tsecs[stim_onset_idxs].T, (tsec_mat.shape[1], 1)).T
        
    except Exception as e: #ValueError:
        traceback.print_exc()

    # Double check trial alignment
    x, y = np.where(tsec_mat==0)
    assert len(list(set(y)))==1, "Incorrect stim onset alignment: %s" % str(list(set(y)))
    relative_tsecs = np.reshape(tsec_mat, (len(trials_in_block)*nframes_per_trial, ))

    # Convert frames_in_file to volume idxs:
    trial_frames_to_vols = pd.DataFrame(trial_frames_to_vols)
    trial_frames_to_vols = trial_frames_to_vols.replace(-100, np.nan)

    return trial_frames_to_vols, relative_tsecs


def get_roiid_from_traceid(animalid, session, fov, run_type=None, traceid='traces001', rootdir='/n/coxfs01/2p-data'):
    
    if run_type is not None:
        if int(session) < 20190511 and run_type == 'gratings':
            a_traceid_dict = glob.glob(os.path.join(rootdir, animalid, session, fov, '*run*', 'traces', 'traceids*.json'))[0]
        else:
            a_traceid_dict = glob.glob(os.path.join(rootdir, animalid, session, fov, '*%s*' % run_type, 'traces', 'traceids*.json'))[0]
    else:
        a_traceid_dict = glob.glob(os.path.join(rootdir, animalid, session, fov, '*run*', 'traces', 'traceids*.json'))[0]
    with open(a_traceid_dict, 'r') as f:
        tracedict = json.load(f)
    
    tid = tracedict[traceid]
    roiid = tid['PARAMS']['roi_id']
    
    return roiid
