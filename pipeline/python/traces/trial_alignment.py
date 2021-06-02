#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 11:01:55 2019

@author: julianarhee
"""
#%%
import h5py
import glob
import os
import json
import copy
import traceback
import re
import optparse
import sys

import pandas as pd
import numpy as np
from pipeline.python.utils import natural_keys, get_frame_info, isnumber

from pipeline.python.paradigm import utils as putils

#%%


def get_run_summary(xdata_df, labels_df, stimconfigs, si, verbose=False):
    
    run_info = {}
    transform_dict, object_transformations = putils.get_transforms(stimconfigs)
    trans_types = object_transformations.keys()

    conditions = sorted(list(set(labels_df['config'])), key=natural_keys)
 
    # Get trun info:
    roi_list = sorted(list(set([r for r in xdata_df.columns.tolist() if isnumber(r)]))) #not r=='index'])))
    ntrials_total = len(sorted(list(set(labels_df['trial'])), key=natural_keys))
    trial_counts = labels_df.groupby(['config'])['trial'].apply(set)
    ntrials_by_cond = dict((k, len(trial_counts[i])) for i,k in enumerate(trial_counts.index.tolist()))
    nframes_per_trial = list(set(labels_df.groupby(['trial'])['stim_on_frame'].count())) #[0]
    nframes_on = list(set(labels_df['stim_dur']))
    nframes_on = [int(round(si['framerate'])) * n for n in nframes_on]

    try:
        ons = [int(np.where(np.array(t)==0)[0]) for t in labels_df.groupby('trial')['tsec'].apply(np.array)]
        assert len(list(set(ons))) == 1
        stim_on_frame = list(set(ons))[0]
    except Exception as e: 
        all_ons = [np.where(np.array(t)==0)[0] for t in labels_df.groupby('trial')['tsec'].apply(np.array)]
        all_ons = np.concatenate(all_ons).ravel()
        print("N stim onsets:", len(all_ons))
        unique_ons = np.unique(all_ons)
        print("**** WARNING: multiple stim onset idxs found - %s" % str(list(set(unique_ons))))
        stim_on_frame = int(round( np.mean(unique_ons) ))
        print("--- assigning stim on frame: %i" % stim_on_frame)  
        
    #ons = [int(np.where(t==0)[0]) for t in labels_df.groupby('trial')['tsec'].apply(np.array)]
    #assert len(list(set(ons)))==1, "More than one unique stim ON idx found!"
    #stim_on_frame = list(set(ons))[0]

    if verbose:
        print "-------------------------------------------"
        print "Run summary:"
        print "-------------------------------------------"
        print "N rois:", len(roi_list)
        print "N trials:", ntrials_total
        print "N frames per trial:", nframes_per_trial
        print "N trials per stimulus:", ntrials_by_cond
        print "-------------------------------------------"

    run_info['roi_list'] = roi_list
    run_info['ntrials_total'] = ntrials_total
    run_info['nframes_per_trial'] = nframes_per_trial
    run_info['ntrials_by_cond'] = ntrials_by_cond
    run_info['condition_list'] = conditions
    run_info['stim_on_frame'] = stim_on_frame
    run_info['nframes_on'] = nframes_on
    #run_info['trace_type'] = trace_type
    run_info['transforms'] = object_transformations
    #run_info['datakey'] = datakey
    run_info['trans_types'] = trans_types
    run_info['framerate'] = si['framerate']
    run_info['nfiles'] = len(labels_df['file_ix'].unique())

    return run_info


#%%

def frames_to_trials(parsed_frames_fpath, trials_in_block, file_ix, si, frame_shift=0,
                     frame_ixs=None):

    all_frames_tsecs = np.array(si['frames_tsec'])
    nslices_full = len(all_frames_tsecs) / si['nvolumes']
    if si['nchannels']==2:
        all_frames_tsecs = np.array(all_frames_tsecs[0::2])
    print "N tsecs:", len(all_frames_tsecs)

    # Get volume indices to assign frame numbers to volumes:
    vol_ixs_tif = np.empty((si['nvolumes']*nslices_full,))
    vcounter = 0
    for v in range(si['nvolumes']):
        vol_ixs_tif[vcounter:vcounter+nslices_full] = np.ones((nslices_full, )) * v
        vcounter += nslices_full
    vol_ixs_tif = np.array([int(v) for v in vol_ixs_tif])
    vol_ixs = []
    vol_ixs.extend(np.array(vol_ixs_tif) + si['nvolumes']*tiffnum for tiffnum in range(si['ntiffs']))
    vol_ixs = np.array(sorted(np.concatenate(vol_ixs).ravel()))
    
    try:
        parsed_frames = h5py.File(parsed_frames_fpath, 'r') 
        trial_list = sorted(parsed_frames.keys(), key=natural_keys)
        print "There are %i total trials across all .tif files." % len(trial_list)
        
        # Check if frame indices are indexed relative to full run (all .tif files)
        # or relative to within-tif frames (i.e., a "block")
        block_indexed = True
        if all([all(parsed_frames[t]['frames_in_run'][:] == parsed_frames[t]['frames_in_file'][:]) for t in trial_list]):
            block_indexed = False
            print "Frame indices are NOT block indexed"
            
        # Assumes all trials have same structure
        min_frame_interval = 1 #list(set(np.diff(frames_to_select['Slice01'].values)))  # 1 if not slices
        nframes_pre = int(round(parsed_frames['trial00001']['frames_in_run'].attrs['baseline_dur_sec'] * si['volumerate']))
        nframes_post = int(round(parsed_frames['trial00001']['frames_in_run'].attrs['iti_dur_sec'] * si['volumerate']))
        nframes_on = int(round(parsed_frames['trial00001']['frames_in_run'].attrs['stim_dur_sec'] * si['volumerate']))
        nframes_per_trial = nframes_pre + nframes_on + nframes_post 
    
    
        # Get ALL frames corresponding to trial epochs:
        # -----------------------------------------------------
        # Get all frame indices for trial epochs (if there are overlapping frame indices, there will be repeats)    
        all_frames_in_trials = np.hstack([np.array(parsed_frames[t]['frames_in_file']) \
                                   for t in trials_in_block])
        print "... N frames to align:", len(all_frames_in_trials)
        print "... N unique frames:", len(np.unique(all_frames_in_trials))
        stim_onset_idxs = np.array([parsed_frames[t]['frames_in_file'].attrs['stim_on_idx'] \
                                    for t in trials_in_block])
    
        # Since we are cycling thru FILES readjust frame indices to match within-file, rather than across-files.
        # block_frame_offset set in extract_paradigm_events (MW parsing) - supposedly set this up to deal with skipped tifs?
        # -------------------------------------------------------
        if block_indexed is False:
            all_frames_in_trials = all_frames_in_trials - len(all_frames_tsecs)*file_ix - frame_shift  
            if all_frames_in_trials[-1] >= len(all_frames_tsecs):
                print '... File: %i (has %i frames)' % (file_ix, len(all_frames_tsecs))
                print "... asking for %i extra frames..." % (all_frames_in_trials[-1] - len(all_frames_tsecs))
            stim_onset_idxs = stim_onset_idxs - len(all_frames_tsecs)*file_ix - frame_shift 
        print "... Last frame to align: %i (N frames total, %i)" % (all_frames_in_trials[-1], len(all_frames_tsecs))
    
        stim_onset_idxs_adjusted = vol_ixs_tif[stim_onset_idxs]
        stim_onset_idxs = copy.copy(stim_onset_idxs_adjusted)
        varying_stim_dur=False
        trial_frames_to_vols = dict((t, []) for t in trials_in_block)
        for t in trials_in_block: 
            frames_to_vols = parsed_frames[t]['frames_in_file'][:] 
            frames_to_vols = frames_to_vols - len(all_frames_tsecs)*file_ix - frame_shift  
            actual_frames_in_trial = [i for i in frames_to_vols if i < len(vol_ixs_tif)]
            trial_vol_ixs = np.empty(frames_to_vols.shape, dtype=int)
            trial_vol_ixs[0:len(actual_frames_in_trial)] = vol_ixs_tif[actual_frames_in_trial]
            if varying_stim_dur is False:
                trial_vol_ixs = trial_vol_ixs[0:nframes_per_trial]
            trial_frames_to_vols[t] = np.array(trial_vol_ixs)


    except Exception as e:
        traceback.print_exc()
    finally:
        parsed_frames.close()
            
    #%
    # Convert frame- to volume-reference, select 1st frame for each volume. (Only relevant for multi-plane)
    # -------------------------------------------------------
    # Don't take unique values, since stim period of trial N can be ITI of trial N-1
    actual_frames = [i for i in all_frames_in_trials if i < len(vol_ixs_tif)]
    frames_in_trials = vol_ixs_tif[actual_frames]
    
    # Turn frame_tsecs into RELATIVE tstamps (to stim onset):
    # ------------------------------------------------
    first_plane_tstamps = all_frames_tsecs[np.array(frame_ixs)]
    print "... N tstamps:", len(first_plane_tstamps)
    trial_tstamps = first_plane_tstamps[frames_in_trials[0:len(actual_frames)]] #all_frames_tsecs[frames_in_trials] #indices]  
    # Check whether we are asking for more frames than there are unique, and pad array if so
    if len(trial_tstamps) < len(all_frames_in_trials): #len(frames_in_trials):
        print "... padding trial tstamps array... (should be %i)" % len(all_frames_in_trials)
        trial_tstamps = np.pad(trial_tstamps, (0, len(all_frames_in_trials)-len(trial_tstamps)), mode='constant', constant_values=np.nan)
        frames_in_trials = np.pad(frames_in_trials, (0, len(all_frames_in_trials)-len(frames_in_trials)), mode='constant', constant_values=np.nan)

    # All trials have the same structure:
    reformat_tstamps = False
    print "N frames per trial:", nframes_per_trial
    print "N tstamps:", len(trial_tstamps)
    print "N trials in block:", len(trials_in_block)
    try: 
        tsec_mat = np.reshape(trial_tstamps, (len(trials_in_block), nframes_per_trial))
        # Subtract stim_on tstamp from each frame of each trial to get relative tstamp:
        tsec_mat -= np.tile(all_frames_tsecs[stim_onset_idxs].T, (tsec_mat.shape[1], 1)).T
        
    except Exception as e: #ValueError:
        traceback.print_exc()
#        reformat_tstamps = True

    x, y = np.where(tsec_mat==0)
    assert len(list(set(y)))==1, "Incorrect stim onset alignment: %s" % str(list(set(y)))
       
    relative_tsecs = np.reshape(tsec_mat, (len(trials_in_block)*nframes_per_trial, ))

    # Convert frames_in_file to volume idxs:
#    trial_frames_to_vols = pd.DataFrame(data=np.reshape(all_frames_in_trials, (len(trials_in_block), nframes_per_trial)),
#                                        index=trials_in_block)
    trial_frames_to_vols = pd.DataFrame(trial_frames_to_vols)

    return trial_frames_to_vols, relative_tsecs

#%%


rootdir = '/n/coxfs01/2p-data'
animalid = 'JC084'
session = '20190522'
fov = 'FOV1_zoom2p0x'

experiment = 'rfs'
traceid = 'traces001'





def extract_options(options):
    parser = optparse.OptionParser()

    # PATH opts:
    parser.add_option('-D', '--root', action='store', dest='rootdir', default='/n/coxfs01/2p-data', 
                      help='root project dir containing all animalids [default: /n/coxfs01/2pdata]')
    parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', 
                      help='Animal ID')
    parser.add_option('-S', '--session', action='store', dest='session', default='', 
                      help='Session (format: YYYYMMDD)')
    # Set specific session/run for current animal:
    parser.add_option('-A', '--fov', action='store', dest='fov', default='FOV1_zoom2p0x', 
                      help="fov name (default: FOV1_zoom2p0x)")
    parser.add_option('-E', '--experiment', action='store', dest='experiment', default='', 
                      help="experiment name (e.g,. gratings, rfs, rfs10, or blobs)") #: FOV1_zoom2p0x)")
    
    parser.add_option('-t', '--traceid', action='store', dest='traceid', default='traces001', 
                      help="traceid (default: traces001)")

    (options, args) = parser.parse_args(options)

    return options

#%%

def aggregate_experiment_runs(animalid, session, fov, experiment, traceid='traces001'):
#%%    
    fovdir = os.path.join(rootdir, animalid, session, fov)
    if int(session) < 20190511 and experiment=='rfs':
        print("This is actually a RFs, but was previously called 'gratings'")
        experiment = 'gratings'

    rawfns = sorted(glob.glob(os.path.join(fovdir, '*%s_*' % experiment, 'traces', '%s*' % traceid, 'files', '*.hdf5')), key=natural_keys)
    print("[%s]: Found %i raw file arrays." % (experiment, len(rawfns)))
    
    #%
    runpaths = sorted(glob.glob(os.path.join(rootdir, animalid, session, fov, '*%s_*' % experiment,
                              'traces', '%s*' % traceid, 'files')), key=natural_keys)
    assert len(runpaths) > 0, "No extracted traces for run %s (%s)" % (experiment, traceid)
    
    # Get .tif file list with corresponding aux file (i.e., run) index:
    rawfns = [(run_ix, file_ix, fn) for run_ix, rpath in enumerate(sorted(runpaths, key=natural_keys))\
              for file_ix, fn in enumerate(sorted(glob.glob(os.path.join(rpath, '*.hdf5')), key=natural_keys))]
    
    
    # Check if this run has any excluded tifs
    rundirs = sorted([d for d in glob.glob(os.path.join(rootdir, animalid, session, fov, '%s_*' % experiment))\
              if 'combined' not in d and os.path.isdir(d)], key=natural_keys)

#%%
    # #########################################################################
    #% Cycle through all tifs, detrend, then get aligned frames
    # #########################################################################
    dfs = {}
    frame_times=[]; trial_ids=[]; config_ids=[]; sdf_list=[]; run_ids=[]; file_ids=[];
    frame_indices = []
    for total_ix, (run_ix, file_ix, fpath) in enumerate(rawfns):
        print("**** File %i of %i *****" % (int(total_ix+1), len(rawfns)))
        try:
            rfile = h5py.File(fpath, 'r')
            fdata = rfile['Slice01']
            trace_types = list(fdata['traces'].keys())
            for trace_type in trace_types:
                if not any([trace_type in k for k in dfs.keys()]):
                    dfs['%s-detrended' % trace_type] = []
                    dfs['%s-F0' % trace_type] = []
            frames_to_select = pd.DataFrame(fdata['frames_indices'][:])        
            #%
            rundir = rundirs[run_ix]
            tid_fpath = glob.glob(os.path.join(rundir, 'traces', '*.json'))[0]
            with open(tid_fpath, 'r') as f:
                tids = json.load(f)
            excluded_tifs = tids[traceid]['PARAMS']['excluded_tiffs']
            print "*** Excluding:", excluded_tifs
            currfile = str(re.search(r"File\d{3}", fpath).group())
            if currfile in excluded_tifs:
                print "... skipping..."
                continue
            
            basedir = os.path.split(os.path.split(fpath)[0])[0]
            
            # Set output dir
            data_array_dir = os.path.join(basedir, 'data_arrays')
            if not os.path.exists(data_array_dir):
                os.makedirs(data_array_dir)
                    
            #% # Get SCAN IMAGE info for run:
            run_name = os.path.split(rundir)[-1]
            si = get_frame_info(rundir)
            
            #% # Load MW info to get stimulus details:
            mw_fpath = glob.glob(os.path.join(rundir, 'paradigm', 'trials_*.json'))[0] # 
            with open(mw_fpath,'r') as m:
                mwinfo = json.load(m)
            pre_iti_sec = round(mwinfo[mwinfo.keys()[0]]['iti_dur_ms']/1E3) 
            nframes_iti_full = int(round(pre_iti_sec * si['volumerate']))
            
            with open(os.path.join(rundir, 'paradigm', 'stimulus_configs.json'), 'r') as s:
                stimconfigs = json.load(s)
            if 'frequency' in stimconfigs[stimconfigs.keys()[0]].keys():
                stimtype = 'gratings'
            elif 'fps' in stimconfigs[stimconfigs.keys()[0]].keys():
                stimtype = 'movie'
            else:
                stimtype = 'image'
       
            # Get all trials contained in current .tif file:
            tmp_trials_in_block = sorted([t for t, mdict in mwinfo.items() if mdict['block_idx']==file_ix], key=natural_keys)
            # 20181016 BUG: ignore trials that are BLANKS:
            trials_in_block = sorted([t for t in tmp_trials_in_block if mwinfo[t]['stimuli']['type'] != 'blank'], key=natural_keys)
        
            frame_shift = 0 if 'block_frame_offset' not in mwinfo[trials_in_block[0]].keys() else mwinfo[trials_in_block[0]]['block_frame_offset']
            parsed_frames_fpath = glob.glob(os.path.join(rundir, 'paradigm', 'parsed_frames_*.hdf5'))[0] #' in pfn][0]
            frame_ixs = np.array(frames_to_select[0].values)
            
            # Assign frames to trials 
            trial_frames_to_vols, relative_tsecs = frames_to_trials(parsed_frames_fpath, trials_in_block, file_ix,
                                                                    si, frame_shift=frame_shift, frame_ixs=frame_ixs)
        
        #%
            # Get stimulus info for each trial:        
            # -----------------------------------------------------
            excluded_params = [k for k in mwinfo[trials_in_block[0]]['stimuli'].keys() if k not in stimconfigs['config001'].keys()]
            print("Excluding:", excluded_params)
            #if 'filename' in stimconfigs['config001'].keys() and stimtype=='image':
            #    excluded_params.append('filename')
            curr_trial_stimconfigs = dict((trial, dict((k,v) for k,v in mwinfo[trial]['stimuli'].items() \
                                           if k not in excluded_params)) for trial in trials_in_block)
            for k, v in curr_trial_stimconfigs.items():
                if v['scale'][0] is not None:
                    curr_trial_stimconfigs[k]['scale'] = [round(v['scale'][0], 1), round(v['scale'][1], 1)]
            for k, v in stimconfigs.items():
                if v['scale'][0] is not None:
                    stimconfigs[k]['scale'] = [round(v['scale'][0], 1), round(v['scale'][1], 1)]
            if stimtype=='image' and 'filepath' in mwinfo['trial00001']['stimuli'].keys():
                for t, v in curr_trial_stimconfigs.items():
                    if 'filename' not in v.keys():
                        curr_trial_stimconfigs[t].update({'filename': os.path.split(mwinfo[t]['stimuli']['filepath'])[-1]})
            
            varying_stim_dur = False
            # Add stim_dur if included in stim params:
            if 'stim_dur' in stimconfigs[stimconfigs.keys()[0]].keys():
                varying_stim_dur = True
                for ti, trial in enumerate(sorted(trials_in_block, key=natural_keys)):
                    curr_trial_stimconfigs[trial]['stim_dur'] = round(mwinfo[trial]['stim_dur_ms']/1E3, 1)
        
            trial_configs=[]
            for trial, sparams in curr_trial_stimconfigs.items():
                #print sparams.keys()
                #print "-------"
                config_name = [k for k, v in stimconfigs.items() if v==sparams]
                #print v.keys()
                assert len(config_name) == 1, "Bad configs - %s" % trial
                #config_name = config_name[0]
                sparams['position'] = tuple(sparams['position'])
                sparams['scale'] = sparams['scale'][0] if not isinstance(sparams['scale'], int) else sparams['scale']
                sparams['config'] = config_name[0]
                trial_configs.append(pd.Series(sparams, name=trial))
                
            trial_configs = pd.concat(trial_configs, axis=1).T
            trial_configs = trial_configs.sort_index() #(by=)
            
            # Get corresponding stimulus/trial labels for each frame in each trial:
            # --------------------------------------------------------------        
            tlength = trial_frames_to_vols.shape[0]
            config_labels = np.hstack([np.tile(trial_configs.T[trial]['config'], (tlength, ))\
                                       for trial in trials_in_block])
            
            trial_labels = np.hstack([np.tile(trial, (tlength,)) \
                                      for trial in trials_in_block])
        
            
            # Get relevant timecourse points
            frames_in_trials = trial_frames_to_vols.T.values.ravel()
            
            #%
            window_size_sec = 30.
            framerate = si['framerate']
            quantile= 0.10
            windowsize = window_size_sec*framerate
                
            for trace_type in trace_types:
                print("... processing trace type: %s" % trace_type)
                # Load raw traces and detrend within .tif file
                df = pd.DataFrame(fdata['traces'][trace_type][:])

                # If trace_type is np_subtracted, need to add original offset back in, first.
                # np_subtracted traces are created in traces/get_traces.py, without offset added.
                #if trace_type == 'np_subtracted':
                #    print "Adding offset for np_subtracted traces"
                #    orig_offset = pd.DataFrame(fdata['traces']['raw'][:]).mean.mean()
                #    df = df + orig_offset

                # Remove rolling baseline, return detrended traces with offset added back in? 
                detrended_df, F0_df = putils.get_rolling_baseline(df, windowsize, quantile=quantile)
                print "Showing initial drift correction (quantile: %.2f)" % quantile
                print "Min value for all ROIs:", np.min(np.min(detrended_df, axis=0))
                currdf = detrended_df.loc[frames_in_trials]
                currdf['ix'] = [total_ix for _ in range(currdf.shape[0])]
                dfs['%s-detrended' % trace_type].append(currdf)
                
                currf0 = F0_df.loc[frames_in_trials]
                currf0['ix'] = [total_ix for _ in range(currdf.shape[0])]
                dfs['%s-F0' % trace_type].append(currf0)
            
            frame_indices.append(frames_in_trials) # added 2019-05-21
        
            frame_times.append(relative_tsecs)
            trial_ids.append(trial_labels)
            config_ids.append(config_labels)
            run_ids.append([run_ix for _ in range(len(trial_labels))])
            file_ids.append([file_ix for _ in range(len(trial_labels))])
            
            sdf = pd.DataFrame(putils.format_stimconfigs(stimconfigs)).T
            sdf_list.append(sdf)
        except Exception as e:
            traceback.print_exc()
            print(e)
        finally:
            rfile.close()

#%%
    # #########################################################################
    #% Concatenate all runs into 1 giant dataframe
    # #########################################################################
    trial_list = sorted(mwinfo.keys(), key=natural_keys)
    
    # Make combined stimconfigs
    sdfcombined = pd.concat(sdf_list, axis=0)
    if 'position' in sdfcombined.columns:
        sdfcombined['position'] = [tuple(s) for s in sdfcombined['position'].values]
    sdf = sdfcombined.drop_duplicates()
    param_names = sorted(sdf.columns.tolist())
    sdf = sdf.sort_values(by=sorted(param_names))
    sdf.index = ['config%03d' % int(ci+1) for ci in range(sdf.shape[0])]
    
    # Rename each run's configs according to combined sconfigs
    new_config_ids=[]
    for orig_cfgs, orig_sdf in zip(config_ids, sdf_list):
        if 'position' in orig_sdf.columns:
            orig_sdf['position'] = [tuple(s) for s in orig_sdf['position'].values]
        cfg_cipher= {}
        for old_cfg_name in orig_cfgs:
            new_cfg_name = sdf[sdf.eq(orig_sdf.loc[old_cfg_name], axis=1).all(axis=1)].index[0]
            cfg_cipher[old_cfg_name] = new_cfg_name
        new_config_ids.append([cfg_cipher[c] for c in orig_cfgs])
    configs = np.hstack(new_config_ids)
            
    # Reindex trial numbers in order
    trials = np.hstack(trial_ids)  # Need to reindex trials
    run_ids = np.hstack(run_ids)
    last_trial_num = 0
    for run_id in sorted(np.unique(run_ids)):
        next_run_ixs = np.where(run_ids==run_id)[0]
        old_trial_names = trials[next_run_ixs]
        new_trial_names = ['trial%05d' % int(int(ti[-5:])+last_trial_num) for ti in old_trial_names]
        trials[next_run_ixs] = new_trial_names
        last_trial_num = int(sorted(trials[next_run_ixs], key=natural_keys)[-1][-5:])
        
    # Check for stim durations
    if 'stim_dur' in stimconfigs[stimconfigs.keys()[0]].keys():
        stim_durs = np.array([stimconfigs[c]['stim_dur'] for c in configs])
    else:
        stim_durs = list(set([round(mwinfo[t]['stim_dur_ms']/1e3, 1) for t in trial_list]))
    nframes_on = np.array([int(round(dur*si['volumerate'])) for dur in stim_durs])
    print "Nframes on:", nframes_on
    print "stim_durs (sec):", stim_durs
    
    # Also collate relevant frame info (i.e., labels):
    tstamps = np.hstack(frame_times)
    f_indices = np.hstack(frame_indices) 
  
#%% 
    #HERE.
    # Get concatenated df for indexing meta info
    roi_list = np.array([r for r in dfs[dfs.keys()[0]][0].columns.tolist() if isnumber(r)])
    xdata_df = pd.concat([d[roi_list] for d in dfs[dfs.keys()[0]]], axis=0).reset_index(drop=True) #drop=True)
    print "XDATA concatenated: %s" % str(xdata_df.shape)
    
     # Turn paradigm info into dataframe: 
    labels_df = pd.DataFrame({'tsec': tstamps, 
                              'frame': f_indices,
                              'config': configs,
                              'trial': trials,
                              'stim_dur': stim_durs #np.tile(stim_dur, trials.shape)
                              }, index=xdata_df.index)
    try:
        ons = [int(np.where(np.array(g['tsec'])==0)[0]) for t, g in labels_df.groupby('trial')]
        assert len(list(set(ons))) == 1
        stim_on_frame = list(set(ons))[0]
    except Exception as e: 
        all_ons = [np.where(np.array(t)==0)[0] for t in labels_df.groupby('trial')['tsec'].apply(np.array)]
        all_ons = np.concatenate(all_ons).ravel()
        unique_ons = np.unique(all_ons)
        print("**** WARNING: multiple stim onset idxs found - %s" % str(list(set(unique_ons))))
        stim_on_frame = int(round( np.mean(unique_ons) ))
        print("--- assigning stim on frame: %i" % stim_on_frame)
     
    labels_df['stim_on_frame'] = np.tile(stim_on_frame, (len(tstamps),))
    labels_df['nframes_on'] = np.tile(int(nframes_on), (len(tstamps),))
    labels_df['run_ix'] = run_ids
    labels_df['file_ix'] = np.hstack(file_ids)
    print("*** LABELS:", labels_df.shape)
    
    sconfigs = sdf.T.to_dict()
    
    run_info = get_run_summary(xdata_df, labels_df, stimconfigs, si)
    
          
    # #########################################################################
    #% Combine all data trace types and save
    # #########################################################################
    # Get combo dir
    existing_combined = glob.glob(os.path.join(fovdir, 'combined_%s_static' % experiment, 
                                               'traces', '%s*' % traceid))
    if len(existing_combined) > 0:
        combined_dir = os.path.join(existing_combined[0], 'data_arrays')
    else:
        combined_traceids = '_'.join([os.path.split(f)[-1] \
                                  for f in [glob.glob(os.path.join(rundir, 'traces', '%s*' % traceid))[0] \
                                            for rundir in rundirs]])
    
        combined_dir = os.path.join(fovdir, 'combined_%s_static' % experiment, 
                                    'traces', combined_traceids, 'data_arrays')
    if not os.path.exists(combined_dir):
        os.makedirs(combined_dir)
        
    
    labels_fpath = os.path.join(combined_dir, 'labels.npz')
    print "Saving labels data...", labels_fpath
    np.savez(labels_fpath, 
             sconfigs = sconfigs,
             labels_data=labels_df,
             labels_columns=labels_df.columns.tolist(),
             run_info=run_info)
    
    # Save all the dtypes
    for trace_type in trace_types:
        print trace_type
        xdata_df = pd.concat(dfs['%s-detrended' % trace_type], axis=0).reset_index() 
        f0_df = pd.concat(dfs['%s-F0' % trace_type], axis=0).reset_index() 
        roidata = [c for c in xdata_df.columns if isnumber(c)] #c != 'ix']
        
        data_fpath = os.path.join(combined_dir, '%s.npz' % trace_type)
        print "Saving labels data...", data_fpath
        np.savez(data_fpath, 
                 data=xdata_df[roidata].values,
                 f0=f0_df[roidata].values,
                 file_ixs=xdata_df['ix'].values,
                 sconfigs=sconfigs,
                 labels_data=labels_df,
                 labels_columns=labels_df.columns.tolist(),
                 run_info=run_info)
        
        del f0_df
        del xdata_df


#%%

def main(options):
    opts = extract_options(options)
    animalid = opts.animalid
    session = opts.session
    fov = opts.fov
    experiment = opts.experiment
    traceid = opts.traceid
    
    aggregate_experiment_runs(animalid, session, fov, experiment, traceid=traceid)


    
if __name__ == '__main__':
    main(sys.argv[1:])
    


#%%
#
#xdata_df = pd.concat(dfs['np_subtracted-detrended'], axis=0).reset_index(drop=True) #drop=True)
#F0 = pd.concat(dfs['np_subtracted-F0'], axis=0).reset_index(drop=True).mean().mean() #drop=True)
#neuropil_df = pd.concat(dfs['neuropil-detrended'], axis=0).reset_index(drop=True) #drop=True)
#neuropil_F0 = pd.concat(dfs['np_subtracted-F0'], axis=0).reset_index(drop=True).mean() #drop=True)
#    
#xdata_df = xdata_df + neuropil_df.mean(axis=0) + F0 #neuropil_F0 + F0
#
#roi=30 #11
#
#pl.figure()
#tmat=[]
#currcfgs = sdf[(sdf['size']==20) & (sdf['speed']==10) & (sdf['sf']==0.1)].index.tolist()
#for k, g in labels_df[labels_df['config'].isin(currcfgs)].groupby(['config']):
#    if k == 'config053':
#        for t, gg in g.groupby(['trial']):
#            #pl.plot(gg['tsec'].values, xdata_df[roi][gg.index])
#            pl.plot(gg['tsec'].values, df_traces[roi][gg.index], lw=0.5, color='k', alpha=0.5)
#    
#
#
##% # Convert raw + offset traces to df/F traces
#stim_on_frame = labels_df['stim_on_frame'].unique()[0]
#tmp_df = []
#for k, g in labels_df.groupby(['trial']):
#    tmat = xdata_df.loc[g.index]
#    bas_mean = np.nanmean(tmat[0:stim_on_frame], axis=0)
#    tmat_df = (tmat - bas_mean) / bas_mean
#    tmp_df.append(tmat_df)
#df_traces = pd.concat(tmp_df, axis=0)
#del tmp_df
#    
#
#sdf_c = sdf.copy()
#rdata = labels_df.copy()
#combo_param_name = '_'.join(['size', 'speed'])
#sdf_c[combo_param_name] = ['_'.join([str(c) for c in list(combo[0])]) for combo in list(zip(sdf_c[['size', 'speed']].values))]
#
#rdata['response'] = df_traces[roi] #xdata_df[roi]
#for p in ['ori', 'size_speed', 'sf']:
#    rdata[p] = [sdf_c[p][c] for c in rdata['config'].values]
#
#
#p = sns.FacetGrid(rdata, col='ori', row='size_speed', hue='sf', sharex=True, sharey=True)
#p.map(pl.plot, "tsec", 'response', lw=0.5, alpha=0.5)
#
#for trial, gg in g.groupby(['trial']):
#    pl.plot(mean_tsec, gg['response'])
#    
#mean_tsec = g.groupby(['trial'])['tsec'].apply(np.array).mean(axis=0)
#pl.figure()
#pl.plot(mean_tsec, g.groupby(['trial'])[ylabel].apply(np.array).mean(axis=0), 'k')

