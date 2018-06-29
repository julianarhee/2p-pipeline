#!/usr/bin/env python2
'''
This script assigns frame indices to trial epochs for each trial in the specified run.
The baseline period (period before stimulus onset) is specified by input opt ITI_PRE ('-b' or '--baseline').

Inputs required:

    a.  Paradigm info:

    <RUN_DIR>/paradigm/trials_<TRIALINFO_HASH>.json
        -- All trials in run with relevant stimulus info and times.
        -- The specific .tif file in which a given trial occurs in the run is stored in trialdict[trialname]['block_idx']
        -- (see paradigm/extract_stimulus_events.py, Step 1 for more info)

    b.  Timecourse info:

    <TRACEID_DIR>/roi_timecourses_YYYYMMDD_HH_MM_SS_<FILEHASH>.hdf5
        -- Output file from traces/get_traces.py (all traces are combined across tiff files)
        -- We want to use this file since we are interested in the behavior at the level of the ROI
        -- (see traces/get_traces.py for more info)


Outputs:

    a.  Stim config info:

    <RUN_DIR>/paradigm/stimulus_configs.json
    -- Configuration for each unique stimulus (stim id, position, size, etc.)
    -- Each config is given an indexed name, for example:

         'config027': {'filename':  name of image file
                       'filepath':  path to image file on stimulus-presentation computer
                       'ntrials' :  number of trials found for this stim config in this run
                       'position':  [xpos, ypos]
                       'rotation':  (float)
                       'scale'   : [sizex, sizey]
                       }

    b.  Parsed frames aligned to stim-onset, with specified baseline period:

    <RUN_DIR>/paradigm/parsed_frames_<filehash>.hdf5
    -- Assigns all frame indices to trial epochs for all trials in run.
    -- Contains a dataset for each trial in the run (ntrials-per-file * nfiles-in-run)
    -- Frame indices are with respect to the entire file, so volume-parsing should be done if nslices > 1
    -- File hierarchy is:

    <TRACEID_DIR>/roi_trials_YYYYMMDD_HH_mm_SS.hdf5

        [stimconfig]/
            attrs: direction, frequency, phase, position_x, position_y, rotation, scale, size_x, size_y, etc.

            [roi_name]/
                attrs: id_in_set, id_in_src, idx_in_slice, slice

                [trial_name]/
                    attrs: volume_stim_on, frame_idxs
                    'raw' -- dataset
                    'denoise_nmf' -- dataset
                    ...etc.

    c.  Dataframe combining trace info, aux info, and eye-tracker data:

    <TRACEID_DIR>/ROIDATA_<CORR_DATESTR>.hdf5
    -- CORR_DATESTR = date-str that corresponds to 'roi_trials_....hdf5' file from (b)
    -- each 'dataset' is an ROI:
        '/roiXXXXX' = dataframe with all trial info, trace info, and eye-tracker info for given ROI



'''
import matplotlib
matplotlib.use('Agg')
import os
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
import copy
from itertools import permutations
from scipy import stats
import pandas as pd
import seaborn as sns
import pylab as pl
import numpy as np
from stat import S_IREAD, S_IRGRP, S_IROTH

from pipeline.python.utils import natural_keys, hash_file_read_only, print_elapsed_time, hash_file
from pipeline.python.traces.utils import get_frame_info, load_TID, get_metric_set
pp = pprint.PrettyPrinter(indent=4)

flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
sns.set_palette(flatui)

#%%
class StimInfo:
    def __init__(self):
        self.stiminfo = dict()

        self.trials = []
        self.frames = []
        self.frames_sec = []
        self.stim_on_idx = []
	self.stim_dur = None #[]
	self.iti_dur = None #[]
	self.volumerate = None

def serialize_json(instance=None, path=None):
    dt = {}
    dt.update(vars(instance))
    return dt

#%%
def load_parsed_trials(parsed_trials_path):
    with open(parsed_trials_path, 'r') as f:
        trialdict = json.load(f)
    return trialdict

#%%
def get_alignment_specs(paradigm_dir, si_info, iti_pre=1.0, iti_post=None, same_order=False):
    trial_epoch_info = {}
    run = os.path.split(os.path.split(paradigm_dir)[0])[-1] #options.run

#    if custom_mw is True:
#        trial_fn = None
#        stimtype = None
#        # Get trial info if custom (non-MW) stim presentation protocols:
#        # -------------------------------------------------------------------------
#        try:
#            stim_on_sec = float(options.stim_on_sec) #2. # 0.5
#            first_stimulus_volume_num = int(options.first_stim_volume_num) #50
#            vols_per_trial = float(options.vols_per_trial) # 15
#            iti_full = (vols_per_trial/si_info['volumerate']) - stim_on_sec
#            iti_post = iti_full - iti_pre
#            print "==============================================================="
#            print "Using CUSTOM trial-info (not MW)."
#            print "==============================================================="
#            print "First stim on:", first_stimulus_volume_num
#            print "Volumes per trial:", vols_per_trial
#            print "ITI POST (s):", iti_post
#            print "ITT full (s):", iti_full
#            print "TRIAL dur (s):", stim_on_sec + iti_full
#            print "Vols per trial (calc):", (stim_on_sec + iti_pre + iti_post) * si_info['volumerate']
#            print "==============================================================="
#
#            # Get stim-order files:
#            stimorder_fns = sorted([f for f in os.listdir(paradigm_dir) if 'stimorder' in f and f.endswith('txt')], key=natural_keys)
#            print "Found %i stim-order files, and %i TIFFs." % (len(stimorder_fns), si_info['ntiffs'])
#            if len(stimorder_fns) < si_info['ntiffs']:
#                if same_order: # Same stimulus order for each file (complete set)
#                    stimorder_fns = np.tile(stimorder_fns, [si_info['ntiffs'],])
#
#        except Exception as e:
#            print "---------------------------------------------------------------"
#            print "Using CUSTOM trial-info. Use -h to check required options:"
#            print "---------------------------------------------------------------"
#            print "- volume num of 1st stimulus in tif"
#            print "- N vols per trial"
#            print "- duration (s) of stimulus-on period in trial"
#            print " - stimorder.txt file with each line containing IDX of stim id."
#            print " - whether the same order of stimuli are presented across all files."
#            print "Aborting with error:"
#            print "---------------------------------------------------------------"
#            print e
#            print "---------------------------------------------------------------"
#
#    else:
    stimorder_fns = None
    first_stimulus_volume_num = None
    vols_per_trial = None

    ### Get PARADIGM INFO if using standard MW:
    # -------------------------------------------------------------------------
    try:
        trial_fn = [t for t in os.listdir(paradigm_dir) if 'trials_' in t and t.endswith('json')]
        assert len(trial_fn)==1, "Unable to find unique trials .json in %s" % paradigm_dir
        trial_fn = trial_fn[0]
        #print paradigm_dir
        parsed_trials_path = os.path.join(paradigm_dir, trial_fn)
        trialdict = load_parsed_trials(parsed_trials_path)
        trial_list = sorted(trialdict.keys(), key=natural_keys)
        stimtype = trialdict[trial_list[0]]['stimuli']['type']


        # Get presentation info (should be constant across trials and files):
        trial_list = sorted(trialdict.keys(), key=natural_keys)
        stim_durs = [round(np.floor(trialdict[t]['stim_dur_ms']/1E3), 1) for t in trial_list]
        #assert len(list(set(stim_durs))) == 1, "More than 1 stim_dur found..."
        if len(list(set(stim_durs))) > 1:
            print "more than 1 stim_dur found..."
            stim_on_sec = dict((t, round(trialdict[t]['stim_dur_ms']/1E3, 1)) for t in trial_list)
        else:
            stim_on_sec = stim_durs[0]
       
        iti_durs = [round(np.floor(trialdict[t]['iti_dur_ms']/1E3), 0) for t in trial_list]
        print list(set(iti_durs))
        if len(list(set(iti_durs))) > 1:
            iti_jitter = 1.0 # TMP TMP 
            replace_max = max(len(list(set(iti_durs)))) - iti_jitter
            iti_durs_tmp = list(set(iti_durs))
            max_ix = iti_durs_tmp.index(max(iti_durs))
            iti_durs_tmp[max_ix] = replace_max
            iti_durs_unique = list(set(iti_durs_tmp))
        else:
            iti_durs_unique = list(set(iti_durs))
        assert len(iti_durs_unique) == 1, "More than 1 iti_dur found..."
        iti_full = iti_durs_unique[0]
        if iti_post is None:
            iti_post = iti_full - iti_pre
        print "ITI POST:", iti_post

        # Check whether acquisition method is one-to-one (1 aux file per SI tif) or single-to-many:
        if trialdict[trial_list[0]]['ntiffs_per_auxfile'] == 1:
            one_to_one =  True
        else:
            one_to_one = False

    except Exception as e:
        print "Could not find unique trial-file for current run %s..." % run
        print "Aborting with error:"
        print "---------------------------------------------------------------"
        traceback.print_exc()
        print "---------------------------------------------------------------"

    try:
        nframes_iti_pre = iti_pre * si_info['framerate'] #framerate
        nframes_iti_post = iti_post*si_info['framerate'] #framerate # int(round(iti_post * volumerate))
        nframes_iti_full = iti_full * si_info['framerate'] #framerate #int(round(iti_full * volumerate))
        if isinstance(stim_on_sec, dict):
            nframes_on = dict((t, stim_on_sec[t] * si_info['framerate']) for t in sorted(stim_on_sec.keys(), key=natural_keys)) #framerate #int(round(stim_on_sec * volumerate))
            nframes_post_onset = dict((t, (stim_on_sec[t] + iti_post) * si_info['framerate']) for t in sorted(stim_on_sec.keys(), key=natural_keys))
            vols_per_trial = dict((t, (iti_pre + stim_on_sec[t] + iti_post) * si_info['volumerate']) for t in sorted(stim_on_sec.keys(), key=natural_keys))
        else:
            nframes_on = stim_on_sec * si_info['framerate'] #framerate #int(round(stim_on_sec * volumerate))
            nframes_post_onset = (stim_on_sec + iti_post) * si_info['framerate'] #framerat
            vols_per_trial = (iti_pre + stim_on_sec + iti_post) * si_info['volumerate']
    except Exception as e:
        print "Problem calcuating nframes for trial epochs..."
        traceback.print_exc()

    trial_epoch_info['stim_on_sec'] = stim_on_sec
    trial_epoch_info['iti_full'] = iti_full
    trial_epoch_info['iti_post'] = iti_post
    trial_epoch_info['iti_pre'] = iti_pre
    trial_epoch_info['one_to_one'] = one_to_one
    trial_epoch_info['nframes_on'] = nframes_on
    trial_epoch_info['nframes_iti_pre'] = nframes_iti_pre
    trial_epoch_info['nframes_iti_post'] = nframes_iti_post
    trial_epoch_info['nframes_iti_full'] = nframes_iti_full
    trial_epoch_info['nframes_post_onset'] = nframes_post_onset
    trial_epoch_info['parsed_trials_source'] = parsed_trials_path
    trial_epoch_info['stimorder_source'] = stimorder_fns
    trial_epoch_info['framerate'] = si_info['framerate']
    trial_epoch_info['volumerate'] = si_info['volumerate']

    trial_epoch_info['first_stimulus_volume_num'] = first_stimulus_volume_num
    trial_epoch_info['vols_per_trial'] = vols_per_trial
    trial_epoch_info['stimtype'] = stimtype
    trial_epoch_info['custom_mw'] = False

    return trial_epoch_info


#%%

def assign_frames_to_trials(si_info, trial_info, paradigm_dir, create_new=False):

    run = os.path.split(os.path.split(paradigm_dir)[0])[-1]
    # First check if parsed frame file already exists:
    existing_parsed_frames_fns = sorted([t for t in os.listdir(paradigm_dir) if 'parsed_frames_' in t and t.endswith('hdf5')], key=natural_keys)
    existing_parsed_frames_fns.sort(key=lambda x: os.stat(os.path.join(paradigm_dir, x)).st_mtime) # Sort by date modified
    if len(existing_parsed_frames_fns) > 0 and create_new is False:
        parsed_frames_filepath = os.path.join(paradigm_dir, existing_parsed_frames_fns[-1]) # Get most recently modified file
        print "---> Got existing parsed-frames file:", parsed_frames_filepath
        return parsed_frames_filepath


    print "---> Creating NEW parsed-frames file..."
    parsed_frames_filepath = os.path.join(paradigm_dir, 'parsed_frames.hdf5')

    # 1. Create HDF5 file to store ALL trials in run with stimulus info and frame info:
    parsed_frames = h5py.File(parsed_frames_filepath, 'w')
    parsed_frames.attrs['framerate'] = si_info['framerate'] #framerate
    parsed_frames.attrs['volumerate'] = si_info['volumerate'] #volumerate
    parsed_frames.attrs['baseline_dur'] = trial_info['iti_pre'] #iti_pre
    #run_grp.attrs['creation_time'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if trial_info['custom_mw'] is False:
        trialdict = load_parsed_trials(trial_info['parsed_trials_source'])
        # Get presentation info (should be constant across trials and files):
        trial_list = sorted(trialdict.keys(), key=natural_keys)

    #%
    # 1. Get stimulus preseentation order for each TIFF found:
    try:
        trial_counter = 0
        for tiffnum in range(si_info['ntiffs']): #ntiffs):
            trial_in_file = 0
            currfile= "File%03d" % int(tiffnum+1)
            print currfile, trial_in_file

            if trial_info['custom_mw'] is True:
                stimorder_fns = trial_info['simorder_source']
                with open(os.path.join(paradigm_dir, stimorder_fns[tiffnum])) as f:
                    stimorder_data = f.readlines()
                stimorder = [l.strip() for l in stimorder_data]
            else:
                stimorder = [trialdict[t]['stimuli']['stimulus'] for t in trial_list\
                                 if trialdict[t]['block_idx'] == tiffnum]
                trials_in_run = sorted([t for t in trial_list if trialdict[t]['block_idx'] == tiffnum], key=natural_keys)
            for trialidx,trialstim in enumerate(sorted(stimorder, key=natural_keys)):
                trial_counter += 1
                trial_in_file += 1
                currtrial_in_file = 'trial%03d' % int(trial_in_file)

                if trial_info['custom_mw'] is True:
                    if trialidx==0:
                        first_frame_on = si_info['first_stimulus_volume_num'] #first_stimulus_volume_num
                    else:
                        first_frame_on += si_info['vols_per_trial'] #vols_per_trial
                    currtrial_in_run = 'trial%05d' % int(trial_counter)
                else:
                    currtrial_in_run = trials_in_run[trialidx]
                    #first_frame_on = int(round(trialdict[currfile][currtrial]['stim_on_idx']/nslices))
                    no_frame_match = False
                    first_frame_on = int(trialdict[currtrial_in_run]['frame_stim_on'])
        #            try:
        #                first_frame_on = frame_idxs.index(first_frame_on)
        #            except Exception as e:
        #                print "------------------------------------------------------------------"
        #                print "Found first frame on from serialdata file NOT found in frame_idxs."
        #                print "Trying 1 frame before / after..."
        #                try:
        #                    if first_frame_on+1 in frame_idxs:
        #                        first_frame_on = frame_idxs.index(first_frame_on+1)
        #                    else:
        #                        # Try first_frame_on-1 in frame_idxs:
        #                        first_frame_on = frame_idxs.index(first_frame_on-1)
        #                except Exception as e:
        #                    print "------------------------------------------------------------------"
        #                    print "NO match found for FIRST frame ON:", first_frame_on
        #                    print "File: %s, Trial %s, Stim: %s." % (currfile, currtrial_in_run, trialstim)
        #                    print e
        #                    print "------------------------------------------------------------------"
        #                    no_frame_match = True
        #                if no_frame_match is True:
        #                    print "Aborting."
        #                    print "------------------------------------------------------------------"

                preframes = list(np.arange(int(first_frame_on - trial_info['nframes_iti_pre']), first_frame_on, 1))
                if isinstance(trial_info['nframes_post_onset'], dict):
                    postframes = list(np.arange(int(first_frame_on + 1), int(round(first_frame_on + trial_info['nframes_post_onset'][currtrial_in_run]))))
                else:
                    postframes = list(np.arange(int(first_frame_on + 1), int(round(first_frame_on + trial_info['nframes_post_onset']))))
                # Check to make sure that rounding errors do not cause frame idxs to go beyond the number of frames in a file:
                if postframes[-1] > len(si_info['vol_idxs']):
                    extraframes = [p for p in postframes if p > len(si_info['vol_idxs'])-1]
                    postframes = [p for p in postframes if p <= len(si_info['vol_idxs'])-1]
                    print "%s:  %i extra frames calculated. Cropping extra post-stim-onset indices." % (currtrial_in_run, len(extraframes))

                framenums = [preframes, [first_frame_on], postframes]
                framenums = reduce(operator.add, framenums)
                #print "POST FRAMES:", len(framenums)
                diffs = np.diff(framenums)
                consec = [i for i in np.diff(diffs) if not i==0]
                assert len(consec)==0, "Bad frame parsing in %s, %s, frames: %s " % (currtrial_in_run, trialstim, str(framenums))

                # Create dataset for current trial with frame indices:
                fridxs_in_file = parsed_frames.create_dataset('/'.join((currtrial_in_run, 'frames_in_file')), np.array(framenums).shape, np.array(framenums).dtype)
                fridxs_in_file[...] = np.array(framenums)
                fridxs_in_file.attrs['trial'] = currtrial_in_file
                fridxs_in_file.attrs['aux_file_idx'] = tiffnum
                fridxs_in_file.attrs['stim_on_idx'] = first_frame_on

                if trial_info['one_to_one'] is True:
                    framenums_in_run = np.array(framenums) + (si_info['nframes_per_file']*tiffnum)
                    abs_stim_on_idx = first_frame_on + (si_info['nframes_per_file']*tiffnum)
                else:
                    framenums_in_run = np.array(framenums)
                    abs_stim_on_idx = first_frame_on

                fridxs = parsed_frames.create_dataset('/'.join((currtrial_in_run, 'frames_in_run')), np.array(framenums_in_run).shape, np.array(framenums_in_run).dtype)
                fridxs[...] = np.array(framenums_in_run)
                fridxs.attrs['trial'] = currtrial_in_run
                fridxs.attrs['aux_file_idx'] = tiffnum
                fridxs.attrs['stim_on_idx'] = abs_stim_on_idx
                if isinstance(trial_info['stim_on_sec'], dict):
                    fridxs.attrs['stim_dur_sec'] = trial_info['stim_on_sec'][currtrial_in_run]
                else: 
                    fridxs.attrs['stim_dur_sec'] = trial_info['stim_on_sec']
                fridxs.attrs['iti_dur_sec'] = trial_info['iti_full']
                fridxs.attrs['baseline_dur_sec'] = trial_info['iti_pre']
    except Exception as e:
        print "Error parsing frames into trials: current file - %s" % currfile
        print "%s in tiff file %s (%i trial out of total in run)." % (currtrial_in_file, currfile, trial_counter)
        traceback.print_exc()
        print "-------------------------------------------------------------------"
    finally:
        parsed_frames.close()

    # Get unique hash for current PARSED FRAMES file:
    parsed_frames_hash = hash_file(parsed_frames_filepath, hashtype='sha1')

    # Check existing files:
    outdir = os.path.split(parsed_frames_filepath)[0]
    existing_files = [f for f in os.listdir(outdir) if 'parsed_frames_' in f and f.endswith('hdf5') and parsed_frames_hash not in f]
    if len(existing_files) > 0:
        old = os.path.join(os.path.split(outdir)[0], 'paradigm', 'old')
        if not os.path.exists(old):
            os.makedirs(old)

        for f in existing_files:
            shutil.move(os.path.join(outdir, f), os.path.join(old, f))

    if parsed_frames_hash not in parsed_frames_filepath:
        parsed_frames_filepath = hash_file_read_only(parsed_frames_filepath)

    print "Finished assigning frame idxs across all tiffs to trials in run %s." % run
    print "Saved parsed frame info to file:", parsed_frames_filepath

    print "-----------------------------------------------------------------------"

    return parsed_frames_filepath


#%%
def get_stimulus_configs(trial_info):
    paradigm_dir = os.path.split(trial_info['parsed_trials_source'])[0]
    trialdict = load_parsed_trials(trial_info['parsed_trials_source'])

    # Get presentation info (should be constant across trials and files):
    trial_list = sorted(trialdict.keys(), key=natural_keys)

    # Get all varying stimulus parameters:
    stimtype = trialdict[trial_list[0]]['stimuli']['type']
    if 'grating' in stimtype:
        exclude_params = ['stimulus', 'type', 'rotation_range']
        stimparams = [k for k in trialdict[trial_list[0]]['stimuli'].keys() if k not in exclude_params]
    else:
        stimparams = [k for k in trialdict[trial_list[0]]['stimuli'].keys() if not (k=='stimulus' or k=='type' or k=='filehash')]

    # Determine whether there are varying stimulus durations to be included as cond:
#    unique_stim_durs = list(set([round(trialdict[t]['stim_dur_ms']/1E3) for t in trial_list]))
#    if len(unique_stim_durs) > 1:
#        stimparams.append('stim_dur')
    stimparams = sorted(stimparams, key=natural_keys)
    
    # Get all unique stimulus configurations (ID, filehash, position, size, rotation):
    allparams = []
    for param in stimparams:
#        if param == 'stim_dur':
#            # Stim DUR:
#            currvals = [round(trialdict[trial]['stim_dur_ms']/1E3) for trial in trial_list]
        if isinstance(trialdict[trial_list[0]]['stimuli'][param], list):
            currvals = [tuple(trialdict[trial]['stimuli'][param]) for trial in trial_list]
        else:
            currvals = [trialdict[trial]['stimuli'][param] for trial in trial_list]
        allparams.append([i for i in list(set(currvals))])

    # Get all combinations of stimulus params:
    transform_combos = list(itertools.product(*allparams))
    ncombinations = len(transform_combos)

    configs = dict()
    for configidx in range(ncombinations):
        configname = 'config%03d' % int(configidx+1)
        configs[configname] = dict()
        for pidx, param in enumerate(sorted(stimparams, key=natural_keys)):
            if isinstance(transform_combos[configidx][pidx], tuple):
                configs[configname][param] = [transform_combos[configidx][pidx][0], transform_combos[configidx][pidx][1]]
            else:
                configs[configname][param] = transform_combos[configidx][pidx]

                
    if stimtype=='image':
        stimids = sorted(list(set([os.path.split(trialdict[t]['stimuli']['filepath'])[1] for t in trial_list])), key=natural_keys)
    elif 'movie' in stimtype:
        stimids = sorted(list(set([os.path.split(os.path.split(trialdict[t]['stimuli']['filepath'])[0])[1] for t in trial_list])), key=natural_keys)
    
    if stimtype == 'image' or 'movie' in stimtype:
        filepaths = list(set([trialdict[trial]['stimuli']['filepath'] for trial in trial_list]))
        filehashes = list(set([trialdict[trial]['stimuli']['filehash'] for trial in trial_list]))

        assert len(filepaths) == len(stimids), "More than 1 file path per stim ID found!"
        assert len(filehashes) == len(stimids), "More than 1 file hash per stim ID found!"

        stimhash_combos = list(set([(trialdict[trial]['stimuli']['stimulus'], trialdict[trial]['stimuli']['filehash']) for trial in trial_list]))
        assert len(stimhash_combos) == len(stimids), "Bad stim ID - stim file hash combo..."
        stimhash = dict((stimid, hashval) for stimid, hashval in zip([v[0] for v in stimhash_combos], [v[1] for v in stimhash_combos]))

    print "---> Found %i unique stimulus configs." % len(configs.keys())

    if stimtype == 'image':
        for config in configs.keys():
            configs[config]['filename'] = os.path.split(configs[config]['filepath'])[1]
    elif 'movie' in stimtype:
        for config in configs.keys():
            configs[config]['filename'] = os.path.split(os.path.split(configs[config]['filepath'])[0])[1]
            
    # Sort config dict by value:
#    sorted_configs = dict(sorted(configs.items(), key=operator.itemgetter(1)))
#    sorted_confignames = sorted_configs.keys()
#    for cidx, configname in sorted(configs.keys(), key=natural_keys):
#        configs[configname] = sorted_configs[sorted_configs.keys()[cidx]

    # SAVE CONFIG info:
    config_filename = 'stimulus_configs.json'
    with open(os.path.join(paradigm_dir, config_filename), 'w') as f:
        json.dump(configs, f, sort_keys=True, indent=4)

    return configs, stimtype


#%%
def load_timecourses(traceid_dir):
    roi_tcourse_filepath=None; curr_slices=None; roi_list=None
    print "-----------------------------------------------------------------------"
    trace_id = os.path.split(traceid_dir)[-1]
    run = os.path.split(os.path.split(traceid_dir)[0])[-1]
    print "Loading time courses for run %s, from trace set: %s" % (run, trace_id)
    try:
        # Since roi_timecourses are datestamped and hashed, sort all found files and use the most recent file:
        tcourse_fn = sorted([t for t in os.listdir(traceid_dir) if t.endswith('hdf5') and 'roi_timecourses' in t], key=natural_keys)[-1]
        roi_tcourse_filepath = os.path.join(traceid_dir, tcourse_fn)
        roi_timecourses = h5py.File(roi_tcourse_filepath, 'r')
        roi_list = sorted(roi_timecourses.keys(), key=natural_keys)
        print "Loaded time-courses file: %s" % tcourse_fn
        print "Found %i ROIs total." % len(roi_list)
    except Exception as e:
        print "***ERROR:  Unable to load time courses for current trace set: %s" % trace_id
        print "File not found in dir: %s" % traceid_dir
        traceback.print_exc()
        print "Aborting with error:"

    try:
        curr_slices = sorted(list(set([roi_timecourses[roi].attrs['slice'] for roi in roi_list])), key=natural_keys)
        print "ROIs are distributed across %i slices:" % len(curr_slices)
        print(curr_slices)
    except Exception as e:
        print "-------------------------------------------------------------------"
        print "***ERROR:  Unable to load slices..."
        traceback.print_exc()
        print "-------------------------------------------------------------------"
    finally:
        roi_timecourses.close()

    print "-----------------------------------------------------------------------"

    return roi_tcourse_filepath, curr_slices, roi_list

#%%
def group_rois_by_trial_type(traceid_dir, parsed_frames_filepath, trial_info, si_info, excluded_tiffs=[], create_new=True):

    paradigm_dir = os.path.split(parsed_frames_filepath)[0]
    trace_hash = os.path.split(traceid_dir)[-1].split('_')[-1]
    run = os.path.split(traceid_dir.split('/traces')[0])[-1]
    excluded_tiff_idxs = [int(tf[4:])-1 for tf in excluded_tiffs]

    # Load trial info:
    trialdict = load_parsed_trials(trial_info['parsed_trials_source'])

    # Get presentation info (should be constant across trials and files):
    #trial_list = sorted(trialdict.keys(), key=natural_keys)
    #print "PARSING %i trials" % len(trial_list)
    vol_idxs = si_info['vol_idxs']

    # Load ROI timecourses file -- this should exist in TRACEID DIR:
    # file format: roi_timecourses_YYYYMMDD_hh_mm_ss_<HASH>,hdf5
    print "---> Loading ROI timecourses by trial and stim config..."
    roi_tcourse_filepath, curr_slices, roi_list = load_timecourses(traceid_dir)
    roi_timecourses = h5py.File(roi_tcourse_filepath, 'r')  # Load ROI traces
    parsed_frames = h5py.File(parsed_frames_filepath, 'r')  # Load PARSED FRAMES output file:


    # Get trial list:
    trial_list = sorted(parsed_frames.keys(), key=natural_keys)
    print "PARSING %i trials" % len(trial_list)

    # Check for stimulus configs:
    configs, stimtype = get_stimulus_configs(trial_info)

    # Create OUTFILE to save each ROI's time course for each trial, sorted by stimulus config
    t_roitrials = time.time()
    # First check if ROI_TRIALS exist -- extraction takes awhile...
    existing_roi_trial_fns = sorted([t for t in os.listdir(traceid_dir) if 'roi_trials_' in t and t.endswith('hdf5')], key=natural_keys)
    if len(existing_roi_trial_fns) > 0 and create_new is False:
        roi_trials_by_stim_path = os.path.join(traceid_dir, existing_roi_trial_fns[-1])
        print "TID %s -- Loaded ROI TRIALS for run %s." % (trace_hash, run)
        print "File path is: %s" % roi_trials_by_stim_path

        # Check file to make sure it is complete:
        roi_trials = h5py.File(roi_trials_by_stim_path, 'r')
        if not len(roi_trials.keys()) == len(configs.keys()):
            print "Incomplete stim-config list found in loaded roi-trials file."
            print "Found %i out of %i stim configs." % (len(roi_trials.keys()), len(configs.keys()))
            print "Creating new...!"
            roi_trials.close()
        else:
            roi_trials.close()
            return roi_trials_by_stim_path

    # This executes only of legit roi_trials file not found--------------------

    # First move old files, if exist:
    if len(existing_roi_trial_fns) > 0:
        olddir = os.path.join(traceid_dir, 'old')
        if not os.path.exists(olddir):
            os.makedirs(olddir)
        for ex in existing_roi_trial_fns:
            os.rename(os.path.join(traceid_dir, ex), os.path.join(olddir, ex))
        print "Moving old roi_trials_ files..."

    tstamp = datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")
    roi_trials_by_stim_path = os.path.join(traceid_dir, 'roi_trials_%s.hdf5' % tstamp)
    roi_trials = h5py.File(roi_trials_by_stim_path, 'w')
    roi = None; trial = None; configname = None
    try:
        print "TID %s -- Creating ROI-TRIALS file, tstamp: %s" % (trace_hash, tstamp)
        for configname in sorted(configs.keys(), key=natural_keys):
            currconfig = configs[configname]
            configparams = [k for k in currconfig.keys() if not k=='filename']
            curr_trials = [t for t in trial_list
                           if all(trialdict[t]['stimuli'][param] == currconfig[param]
                           and trialdict[t]['block_idx'] not in excluded_tiff_idxs for param in configparams)]
            print "Found %i trials for current stim config." % len(curr_trials)
            configs[configname]['ntrials'] = len(curr_trials)

            if configname not in roi_trials.keys():
                config_grp = roi_trials.create_group(configname)
                nonlist_attrs = [int, float, str, unicode]
                for attr_key in configs[configname].keys():
                    if type(configs[configname][attr_key]) in nonlist_attrs:
                        config_grp.attrs[attr_key] = configs[configname][attr_key]
                    else: # isinstance(configs[configname][attr_key], list):
                        attr_key_x = '%s_x' % attr_key
                        attr_key_y = '%s_y' % attr_key
                        config_grp.attrs[attr_key_x] = configs[configname][attr_key][0]
                        config_grp.attrs[attr_key_y] = configs[configname][attr_key][1]
            else:
                config_grp = roi_trials[configname]

            for tidx, trial in enumerate(sorted(curr_trials, key=natural_keys)):
                currtrial = trial # 'trial%03d' % int(tidx + 1)
                curr_trial_volume_idxs = [vol_idxs[int(i)] for i in parsed_frames[trial]['frames_in_run']]

                stim_on_frame_idx = parsed_frames[trial]['frames_in_run'].attrs['stim_on_idx']
                stim_on_volume_idx = vol_idxs[stim_on_frame_idx]
                trial_idxs = sorted(list(set(curr_trial_volume_idxs)))

                for roi in roi_list:
                    if roi not in config_grp.keys():
                        roi_grp = config_grp.create_group(roi)
                    else:
                        roi_grp = config_grp[roi]

                    if currtrial not in roi_grp.keys():
                        trial_grp = roi_grp.create_group(currtrial)
                        trial_grp.attrs['volume_stim_on'] = stim_on_volume_idx
                        trial_grp.attrs['frame_idxs'] = trial_idxs
                        trial_grp.attrs['aux_file_idx'] = parsed_frames[trial]['frames_in_run'].attrs['aux_file_idx']
                    else:
                        trial_grp = roi_grp[currtrial]

                    timecourse_opts = roi_timecourses[roi]['timecourse'].keys()
                    for tc in timecourse_opts:
                        tcourse = roi_timecourses[roi]['timecourse'][tc][trial_idxs]
                        tset = trial_grp.create_dataset(tc, tcourse.shape, tcourse.dtype)
                        tset[...] = tcourse

                    config_grp[roi].attrs['id_in_set'] = roi_timecourses[roi].attrs['id_in_set']
                    #config_grp[roi].attrs['id_in_src'] = roi_timecourses[roi].attrs['id_in_src']
                    config_grp[roi].attrs['idx_in_slice'] = roi_timecourses[roi].attrs['idx_in_slice']
                    config_grp[roi].attrs['slice'] = roi_timecourses[roi].attrs['slice']

    except Exception as e:
        print "--- ERROR grouping ROI time courses by stimulus config. ---"
        print configname, trial, roi
        traceback.print_exc()
        print "-------------------------------------------------------------"
    finally:
        roi_trials.close()

    # Get unique hash for current PARSED FRAMES file:
    roi_trials_hash = hash_file(roi_trials_by_stim_path, hashtype='sha1')

    # Check existing files, move old ones, and set read-only with file hash on new HDF5:
    outdir = os.path.split(roi_trials_by_stim_path)[0]
    existing_files = [f for f in os.listdir(outdir) if 'roi_trials_' in f
                          and f.endswith('hdf5') and roi_trials_hash not in f and tstamp not in f]
    if len(existing_files) > 0:
        old = os.path.join(outdir, 'old')
        if not os.path.exists(old):
            os.makedirs(old)
        for f in existing_files:
            shutil.move(os.path.join(outdir, f), os.path.join(old, f))
    if roi_trials_hash not in roi_trials_by_stim_path:
        roi_trials_by_stim_path = hash_file_read_only(roi_trials_by_stim_path)

    print "TID %s -- Finished extracting time course for run %s by roi." % (trace_hash, run)
    print_elapsed_time(t_roitrials)
    print "Saved ROI TIME COURSE file to:", roi_trials_by_stim_path

    print "-----------------------------------------------------------------------"

    # Update CONFIG info:
    config_filename = 'stimulus_configs.json'
    with open(os.path.join(paradigm_dir, config_filename), 'w') as f:
        json.dump(configs, f, sort_keys=True, indent=4)

    return roi_trials_by_stim_path


#%%
def traces_to_trials(trial_info, si_info, configs, roi_trials_by_stim_path, trace_type='raw', eye_info=None):
    print "-------------------------------------------------------------------"
    print "Aligning TRACES into parsed trials by stimulus type."

    roi=None; configname=None; trial=None
    DATA = dict()

    if 'grating' in trial_info['stimtype']:
        stimtype = 'grating'
    elif 'movie' in trial_info['stimtype']:
        stimtype = 'movie'
    else:
        stimtype = 'image'

    if stimtype == 'movie':
        last_trials_in_block = []
        #last_trials_in_block = ['trial%05d' % i for i in np.arange(16, 416, 16)]
    else:
        last_trials_in_block= []

    # Load ROI list and traces:
    roi_trials = h5py.File(roi_trials_by_stim_path, 'r')
    roi_list = sorted(roi_trials[roi_trials.keys()[0]].keys(), key=natural_keys)

    # Get info for TRIAL EPOCH for alignment:
    volumerate = trial_info['volumerate'] #parsed_frames.attrs['volumerate']
    framerate = trial_info['framerate']
    iti_pre = trial_info['iti_pre']
    iti_post = trial_info['iti_post']
    nframes_on = trial_info['nframes_on'] #parsed_frames['trial00001']['frames_in_run'].attrs['stim_dur_sec'] * volumerate
    stim_dur = trial_info['stim_on_sec'] #trialdict['trial00001']['stim_dur_ms']/1E3
    iti_dur = trial_info['iti_full'] #trialdict['trial00001']['iti_dur_ms']/1E3
    tpoints = [int(i) for i in np.arange(-1*iti_pre, stim_dur+iti_dur)]

    config_list = sorted(roi_trials.keys(), key=natural_keys)
    try:
        for roi in roi_list:
            roi_dfs = []
            bad_trials = []
            for configname in sorted(config_list, key=natural_keys): #sorted(ROIs.keys(), key=natural_key):

                curr_slice = roi_trials[configname][roi].attrs['slice']
                roi_in_slice = roi_trials[configname][roi].attrs['idx_in_slice']
                stim_trials = sorted([t for t in roi_trials[configname][roi].keys()], key=natural_keys)
                nvols = max([roi_trials[configname][roi][t][trace_type].shape[0] for t in stim_trials])
                ntrials = len(stim_trials)

                # initialize TRIALMAT: each row is a trial, each column is a frame of that trial
                trialmat = np.ones((ntrials, nvols)) * np.nan
                dfmat = []
                tsecmat = np.ones((ntrials, nvols)) * np.nan

                # Identify the first frame (across all trials) that the stimulus comes on --
                # This frame is the one we will align all other trials to.
                first_on = int(min([[i for i in roi_trials[configname][roi][t].attrs['frame_idxs']].index(roi_trials[configname][roi][t].attrs['volume_stim_on']) for t in stim_trials]))

                #tsecs = (np.arange(0, nvols) - first_on ) / volumerate  # Using volumerate, since assuming we look at 1 roi on 1 slice
                sidx = int(curr_slice[5:]) - 1 # Get slice index
                nslices_full = si_info['nslices_full']
                all_frame_idxs = np.array(si_info['frames_tsec'])[sidx::nslices_full]

                # If stimulus is an object, we should parse the image name into
                # object ID + transform type and level:
                if stimtype != 'gratings':
                    # Figure out Morph IDX for 1st and last anchor image:
                    if os.path.splitext(configs[configs.keys()[0]]['filename'])[-1] == '.png':
                        fns = [configs[c]['filename'] for c in configs.keys() if 'morph' in configs[c]['filename']]
                        mlevels = sorted(list(set([int(fn.split('_')[0][5:]) for fn in fns])))
                    elif 'fps' in configs[configs.keys()[0]].keys():
                        fns = [configs[c]['filename'] for c in configs.keys() if 'Blob_M' in configs[c]['filename']]
                        mlevels = sorted(list(set([int(fn.split('_')[1][1:]) for fn in fns])))   
                    #print "FN parsed:", fns[0].split('_')
                    if mlevels[-1] > 22:
                        anchor2 = 106
                    else:
                        anchor2 = 22
                    assert all([anchor2 > m for m in mlevels]), "Possibly incorrect morphlevel assignment (%i). Found morphs %s." % (anchor2, str(mlevels))

            
                if stimtype == 'image':
                    imname = os.path.splitext(configs[configname]['filename'])[0]
                    if ('CamRot' in imname) and not('morph' in imname):
                        objectid = imname.split('_CamRot_')[0]
                        yrot = int(imname.split('_CamRot_y')[-1])
                        if 'N1' in imname:
                            morphlevel = 0
                        elif 'N2' in imname:
                            morphlevel = anchor2
                            
                    elif '_yRot' in imname:
                        # Real-world objects:  format is 'IDENTIFIER_xRot0_yRot0_xRot0'
                        objectid = imname.split('_')[0]
                        yrot = int(imname.split('_')[3][4:])
                        morphlevel = 0
                        
                    elif 'morph' in imname:
                        if '_y' not in imname and '_yrot' not in imname:
                            objectid = imname #'morph' #imname
                            yrot = 0
                            morphlevel = int(imname.split('morph')[-1])
                        else:
                            #objectid = imname #'morph' #imname.split('_y')[0]
                            if 'CamRot' in imname:
                                objectid = imname.split('_CamRot_')[0] #'morph' #imname.split('_y')[0]
                                yrot = int(imname.split('_CamRot_y')[-1])
                                morphlevel = int(imname.split('_CamRot_y')[0].split('morph')[-1])
                            else:
                                objectid = imname.split('_y')[0]
                                yrot = int(imname.split('_y')[-1])
                                morphlevel = int(imname.split('_y')[0].split('morph')[-1])
                            
                elif 'movie' in stimtype:
                    imname = os.path.splitext(configs[configname]['filename'])[0]
                    objectid = imname.split('_movie')[0] #'_'.join(imname.split('_')[0:-1])
                    if 'reverse' in imname:
                        yrot = -1
                    else:
                        yrot = 1
                    if imname.split('_')[1] == 'D1':
                        morphlevel = 0
                    elif imname.split('_')[1] == 'D2':
                        morphlevel = anchor2
                    elif imname.split('_')[1][0] == 'M':
                        # Blob_M11_Rot_y_etc.
                        morphlevel = int(imname.split('_')[1][1:])
                    elif imname.split('_')[1] == 'morph':
                        # This is a full morph movie:
                        morphlevel = -1
                        

                for tidx, trial in enumerate(sorted(stim_trials, key=natural_keys)): #[15:25]):
                    if trial in last_trials_in_block:
                        continue
                    
                    frame_idxs = roi_trials[configname][roi][trial].attrs['frame_idxs']
                    adj_frame_idxs = frame_idxs - roi_trials[configname][roi][trial].attrs['aux_file_idx'] * len(all_frame_idxs)

                    # Check if last calculated frame is actually not included (can happend at end of .tif file):
                    if adj_frame_idxs[-1] > len(all_frame_idxs):
                        last_idx = [i for i in adj_frame_idxs].index(len(all_frame_idxs))
                        trailing_frames = np.arange(last_idx, len(adj_frame_idxs))
                        print "**WARNING: %i extra frames calculated for %s" % (len(trailing_frames), trial)
                        adj_frame_idxs = adj_frame_idxs[0:last_idx]

                    # Get stim_dur and iti_dur:
                    if isinstance(trial_info['stim_on_sec'], dict):
                        stim_dur = trial_info['stim_on_sec'][trial]
                        nframes_on = trial_info['nframes_on'][trial] 

                    tsecs = all_frame_idxs[adj_frame_idxs] - all_frame_idxs[adj_frame_idxs][first_on]
                    if not (round(tsecs[0]) == -1*iti_pre and round(tsecs[-1]) == (stim_dur+iti_post)):
                        print "Bad trial indices found!", roi, configname, trial
                        print "Aux file idx:", roi_trials[configname][roi][trial].attrs['aux_file_idx']
                        print "tsecs:", tsecs
                        bad_trials.append((configname, trial, roi_trials[configname][roi][trial].attrs['aux_file_idx']))
                        continue
                        

                    # Get raw (or other specified) timecourse for current trial:
                    trial_timecourse = roi_trials[configname][roi][trial][trace_type]

                    # Identify the frame index within the current trial that the stimulus comes on:
                    curr_on = int([i for i in roi_trials[configname][roi][trial].attrs['frame_idxs']].index(int(roi_trials[configname][roi][trial].attrs['volume_stim_on'])))

                    # Align current trial frames to the "stim onset" point:
                    trialmat[tidx, first_on:first_on+len(trial_timecourse[curr_on:])] = trial_timecourse[curr_on:]
                    tsecmat[tidx, first_on:first_on+len(tsecs[curr_on:])] = tsecs[curr_on:]
                    if first_on < curr_on:
                        trialmat[tidx, 0:first_on] = trial_timecourse[1:curr_on]
                        tsecmat[tidx, 0:first_on] = tsecs[1:curr_on]
                    else:
                        trialmat[tidx, 0:first_on] = trial_timecourse[0:curr_on]
                        tsecmat[tidx, 0:first_on] = tsecs[0:curr_on]

                    # Identify the baseline period in order to calculate DF/F.
                    # NOTE:  if baseline is nans or 0s, this is likely a "bad" ROI
                    # that is actually on the edge of some motion-corrected artefact.
                    # These traces should just be set to NaN:
                    baseline = np.nanmean(trialmat[tidx, 0:first_on]) #[0:on_idx])
                    if baseline == 0 or baseline == np.nan:
                        df = np.ones(trialmat[tidx,:].shape) * np.nan
                    else:
                        df = (trialmat[tidx,:] - baseline) / baseline

                    # check for FUNKY due to NP subtraction:
#                    if max(df) > 10 and not trace_type == 'raw':
#                        df = np.ones(trialmat[tidx,:].shape) * np.nan

                    dfmat.append(df)
                    nframes = len(df)

                    # Create DATAFRAME for current ROI ------------------------

                    # First, get all the info standard to all experiments:
                    df_main = pd.DataFrame({ 'trial': np.tile(trial, (nframes,)),
                                             'config': np.tile(configname, (nframes,)),
                                             'xpos': np.tile(configs[configname]['position'][0], (nframes,)),
                                             'ypos': np.tile(configs[configname]['position'][1], (nframes,)),
                                             'size': np.tile(configs[configname]['scale'][0], (nframes,)),
                                             'tsec': tsecmat[tidx, :], #tsecs,
                                             'raw': trialmat[tidx, :],
                                             'df': df,
                                             'first_on': np.tile(first_on, (nframes,)),
                                             'nframes_on': np.tile(nframes_on, (nframes,)),
                                             'nsecs_on': np.tile(nframes_on/framerate, (nframes,)),
                                             'slice': np.tile(curr_slice, (nframes,)),
                                             'roi_in_slice': np.tile(roi_in_slice, (nframes,)) })

                    # Second, append all stimulus-specific info:
                    if stimtype == 'grating':
                        df_stim = pd.DataFrame({ 'ori': np.tile(configs[configname]['rotation'], (nframes,)),
                                                 'sf': np.tile(configs[configname]['frequency'], (nframes,)) })
                    else:
                        df_stim = pd.DataFrame({ 'object': np.tile(objectid, (nframes,)),
                                                 'yrot': np.tile(yrot, (nframes,)),
                                                 'morphlevel': np.tile(morphlevel, (nframes,)) })
                    df_main = pd.concat([df_main, df_stim], axis=1)

                    # Third, append eye-tracker info, if it exists:
                    if eye_info is not None:
                        df_eye = pd.DataFrame({ 'pupil_size_baseline': np.tile(eye_info[trial]['pupil_size_baseline'], (nframes,)),
                                                'pupil_size_stimulus': np.tile(eye_info[trial]['pupil_size_stim'], (nframes,)),
                                                'pupil_dist_baseline': np.tile(eye_info[trial]['pupil_dist_baseline'], (nframes,)),
                                                'pupil_dist_stimulus': np.tile(eye_info[trial]['pupil_dist_stim'], (nframes,)),
                                                'pupil_nblinks_baseline': np.tile(eye_info[trial]['blink_event_count_baseline'], (nframes,)),
                                                'pupil_nblinks_stim': np.tile(eye_info[trial]['blink_event_count_stim'], (nframes,)) })
                        df_main = pd.concat([df_main, df_eye], axis=1)

                    # Concatenate all info for this current trial:
                    roi_dfs.append(df_main)

            # Finally, concatenate all trials across all configs for current ROI dataframe:
            ROI = pd.concat(roi_dfs, axis=0)

            DATA[roi] = ROI

    except Exception as e:

        print "--- Error configuring ROIDATA dataframe ---------------------------------"
        print roi, configname, trial
        traceback.print_exc()

    return DATA

#%%
def load_eye_data(run_dir):
    eye_file_dir = os.path.join(run_dir,'eyetracker','files')
    run = os.path.split(run_dir)[-1]
    acquisition_dir = os.path.split(run_dir)[0]
    session_dir =os.path.split(acquisition_dir)[0]
    session = os.path.split(session_dir)[-1]
    animalid = os.path.split(os.path.split(session_dir)[0])[-1]
    eye_info_fn = 'parsed_eye_%s_%s_%s.json'%(session, animalid, run)
    eye_info_filepath = os.path.join(eye_file_dir, eye_info_fn)
    try:
        with open(eye_info_filepath, 'r') as f:
            eye_info = json.load(f)
    except Exception as e:
        print "--------------------------------------------------------------"
        print "***WARNING:  NO eye-tracker info found!"
        print "Setting EYE-INFO to None."
        traceback.print_exc()
        eye_info = None
        print "--------------------------------------------------------------"

    return eye_info

#%%
def set_pupil_params(radius_min=30, radius_max=60, dist_thr=8, create_empty=False):
    pupil_params = dict()
    if create_empty is True:
        pupil_params['radius_min'] = None
        pupil_params['radius_max'] = None
        pupil_params['dist_thr'] = None
        pupil_params['max_nblinks']= None
    else:
        pupil_params['radius_min'] = radius_min
        pupil_params['radius_max'] = radius_max
        pupil_params['dist_thr'] = dist_thr
        pupil_params['max_nblinks']= 0

    # Generate hash ID for current pupil params set
    pupil_params_hash = hash(json.dumps(pupil_params, sort_keys=True, ensure_ascii=True)) % ((sys.maxsize + 1) * 2) #[0:6]
    pupil_params['hash'] = pupil_params_hash

    return pupil_params

#%%
def calculate_metrics(DATA, filter_pupil=False, pupil_params=None):
    '''
    This function creates a new dict of DATAFRAMES (keys are rois).
    First, apply filtering using pupil info, if relevant.
    Then, for each stimulus configuration (unique combinations of pos, size, yrot, morph),
    calculate:
        - mean df/f value during stimulus-ON period of each trial
        - std of df/f during baseline period of each trial
        - zscore value for each trial
        - include whether trial is pass/fail based on pupil criteria
    '''

    print "--->  Calculating ROI metrics......"

    METRICS = {}
    PASSTRIALS = {}

    if filter_pupil is True:
        if pupil_params is None:
            pupil_params = set_pupil_params(create_empty=False)
        pupil_radius_min = pupil_params['radius_min']
        pupil_radius_max = pupil_params['radius_max']
        pupil_dist_thr = pupil_params['dist_thr']
        pupil_max_nblinks = pupil_params['max_nblinks']

    if isinstance(DATA, pd.DataFrame):
        is_dataframe = True
        roi_list = sorted(list(set(DATA['roi'])), key=natural_keys)
        config_list = sorted(list(set(DATA[DATA['roi']==roi_list[0]]['config'])), key=natural_keys)
        # Get nframes for trial epochs:
        first_on = int(list(set(DATA[DATA['roi']==roi_list[0]]['first_on']))[0])
        nframes_on = int(round(list(set(DATA[DATA['roi']==roi_list[0]]['nframes_on']))[0]))
        # Check if eye info exists in DataFrame:
        if 'pupil_size_stimulus' in DATA[DATA['roi']==roi_list[0]].keys():
            eye_info_exists = True
        else:
            eye_info_exists = False
    else:
        is_dataframe = False
        roi_list = sorted(DATA.keys(), key=natural_keys)
        config_list = sorted(list(set(DATA[roi_list[0]]['config'])), key=natural_keys)
        # Get nframes for trial epochs:
        first_on = int(list(set(DATA[roi_list[0]]['first_on']))[0])
        nframes_on = int(round(list(set(DATA[roi_list[0]]['nframes_on']))[0]))
        # Check if eye info exists in DataFrame:
        if 'pupil_size_stimulus' in DATA[roi_list[0]].keys():
            eye_info_exists = True
        else:
            eye_info_exists = False

    print "--> Found %i rois." % len(roi_list)
    print "--> Found %i stim configs." % len(config_list)
#
#    # Get nframes for trial epochs:
#    first_on = list(set(DATA[roi_list[0]]['first_on']))[0]
#    nframes_on = int(round(list(set(DATA[roi_list[0]]['nframes_on']))[0]))

#    if 'pupil_size_stimulus' in DATA[roi_list[0]].keys():
#        eye_info_exists = True
#    else:
#        eye_info_exists = False

    for ri,roi in enumerate(roi_list):
        if ri % 10 == 0:
            print "      ... processing %i (of %i)" % (ri, len(roi_list))
        metrics_df = []
        for config in config_list:
            if is_dataframe:
                DF = DATA[((DATA['roi']==roi) & (DATA['config']==config))]
            else:
                DF = DATA[roi][DATA[roi]['config'] == config]

            trial_list = sorted(list(set(DF['trial'])), key=natural_keys)

            if filter_pupil is True:
                filtered_DF = DF.query('pupil_size_stimulus > @pupil_radius_min \
                                       & pupil_size_baseline > @pupil_radius_min \
                                       & pupil_size_stimulus < @pupil_radius_max \
                                       & pupil_size_baseline < @pupil_radius_max \
                                       & pupil_dist_stimulus < @pupil_dist_thr \
                                       & pupil_dist_baseline < @pupil_dist_thr \
                                       & pupil_nblinks_stim <= @pupil_max_nblinks \
                                       & pupil_nblinks_baseline >= @pupil_max_nblinks')
                #print "FILTERED:", len(filtered_DF)
                if len(filtered_DF)==0:
                    print "NONE found for config: %s" % config
                    print "...skipping"
                    continue
                pass_trials = sorted(list(set(filtered_DF['trial'])), key=natural_keys)
                fail_trials = sorted([t for t in trial_list if t not in pass_trials], key=natural_keys)
            else:
                filtered_DF = DF.copy()
                pass_trials = sorted(trial_list, key=natural_keys)
                fail_trials = []

            # Turn DF values into matrix with rows=trial, cols=df value for each frame:
            trials = np.vstack((filtered_DF.groupby(['trial'])['df'].apply(np.array)).as_matrix())
            #print trials.shape
            #print 'FIRST ON IDX:', first_on
            std_baseline_values = np.nanstd(trials[:, 0:first_on], axis=1)
            mean_baseline_values = np.nanmean(trials[:, 0:first_on], axis=1)
            mean_stim_on_values = np.nanmean(trials[:, first_on:first_on+nframes_on], axis=1)
            zscore_values = [meanval/stdval for (meanval, stdval) in zip(mean_stim_on_values, std_baseline_values)]

            idx = 0
            for tidx, trial in enumerate(trial_list): #enumerate(pass_trials):
                #print trial
                if trial in fail_trials:
                    pass_flag = False
                    zscore_val = np.nan
                    mean_stim_val = np.nan
                    mean_baseline_val = np.nan
                else:
                    pass_trial_idx = [t for t in pass_trials].index(trial)
                    pass_flag = True
                    zscore_val = zscore_values[pass_trial_idx]
                    mean_stim_val = mean_stim_on_values[pass_trial_idx]
                    mean_baseline_val = mean_baseline_values[pass_trial_idx]

                # Add trial stats and stim-info to dataframe:
                df_main = pd.DataFrame({'trial': trial,
                                        'config': config,
                                        'zscore': zscore_val,
                                        'mean_stim_on': mean_stim_val,
                                        'mean_baseline': mean_baseline_val,
                                        'pass': pass_flag}, index=[idx])
                # Also add eye-tracker info, if exists:
                if eye_info_exists:
                    eye_df = pd.DataFrame({'pupil_size_baseline': list(set(DF[DF['trial']==trial]['pupil_size_baseline']))[0],
                                           'pupil_size_stimulus': list(set(DF[DF['trial']==trial]['pupil_size_stimulus']))[0],
                                           'pupil_dist_stimulus': list(set(DF[DF['trial']==trial]['pupil_dist_stimulus']))[0],
                                           'pupil_dist_baseline': list(set(DF[DF['trial']==trial]['pupil_dist_baseline']))[0],
                                           'pupil_nblinks_stim': list(set(DF[DF['trial']==trial]['pupil_nblinks_stim']))[0],
                                           'pupil_nblinks_baseline': list(set(DF[DF['trial']==trial]['pupil_nblinks_baseline']))[0]}, index=[idx])
                    df_main = pd.concat([df_main, eye_df], axis=1)

                # Append all relevant trial info to current ROI's metric DF:
                metrics_df.append(df_main)
                idx += 1

            # Save list of trials that pass for current stim-config:
            PASSTRIALS[config] = pass_trials

        # Save all trial info for all stim-configs to current ROI's metric DF:
        ROI = pd.concat(metrics_df, axis=0)

        METRICS[roi] = ROI

    return METRICS, PASSTRIALS

#%%

def get_roi_metrics(roidata_filepath, configs, traceid_dir, trace_type='raw', filter_pupil=False, pupil_params=None, create_new=False):
    '''
    Calculates zscore for each trial for each stimulus configuration for each ROI trace.
    Applies filter to trials based on pupil info, if relevant.

    Main calculation/sorting is done in calculate_metrics().

    OUTPUTS:
    1.  <TRACEID_DIR>/metrics/roi_metrics_<FILTERHASH>_<DATESTR>.hdf5

        File hierarchy:
            /roiXXX -- group
                /dataframe
                    - trial
                    - config
                    - mean_stim_on
                    - zscore
                    - pass

    2. <TRACEID_DIR>/metrics/pass_trials.json

        For each stimulus configuration, lists which trials pass based on specified pupil filtering params, if relevant.

    Returns:

        metrics_filepath -- full path to saved HDF5 file (Output 1)

    '''

    DATA = pd.HDFStore(roidata_filepath, 'r')

    # Use default pupil params if none provided:
    if filter_pupil is True:
        if pupil_params is None:
            print "No pupil params provided. Using default..."
            pupil_params = set_pupil_params(create_empty=False)
        #metric_desc = 'pupil_size%i-dist%i-blinks%i_%s' % (pupil_params['size_thr'], pupil_params['dist_thr'], pupil_params['max_nblinks'], pupil_params['hash'])
        metric_desc = 'pupil_rmin%.2f-rmax%.2f-dist%.2f_%s' % (pupil_params['radius_min'], pupil_params['radius_max'], pupil_params['dist_thr'], pupil_params['hash'])
    elif filter_pupil is False:
        pupil_params = set_pupil_params(create_empty=True)
        metric_desc = 'unfiltered_%d' % pupil_params['hash']

    # Create descriptive directory name for current metrics set:
    metrics_basedir = os.path.join(traceid_dir, 'metrics')
    if not os.path.exists(metrics_basedir):
        os.makedirs(metrics_basedir)
    metrics_dir = os.path.join(traceid_dir, 'metrics', metric_desc)
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)

    # Save JSON for quick-viewing of params (might be unnecessary since now params are in directory name)
    pupil_params_filepath = os.path.join(metrics_dir, 'pupil_params.json')
    if not os.path.exists(pupil_params_filepath):
        with open(pupil_params_filepath, 'w') as f:
            json.dump(pupil_params, f)

    # Check for existing metrics files (hdf5) in current metrics dir:
    metric_hash = pupil_params['hash']
    existing_metrics_files = [f for f in os.listdir(metrics_dir) if 'roi_metrics_%s_%s' % (metric_hash, trace_type) in f and f.endswith('hdf5')]
    if create_new is False:
        try:
            assert len(existing_metrics_files) == 1, "Unably to find unique metrics file for hash %s" % metric_hash
            metrics_filepath = os.path.join(metrics_dir, existing_metrics_files[0])
            METRICS = pd.HDFStore(metrics_filepath, 'r')
            with open(os.path.join(metrics_dir, 'pass_trials.json'), 'r') as f:
                PASSTRIALS = json.load(f)
            return metrics_filepath
        except Exception as e:
            create_new = True
            print "*** ERROR:  Unable to get METRICS and PASS-TRIALS info."
            print "Creating new!"
            traceback.print_exc()
        finally:
            if 'METRICS' in locals():
                METRICS.close()

     # Only do this if no proper metrics filepath found:-----------------------
    # Move or get rid of existing files:
    if len(existing_metrics_files) > 0:
        if not os.path.exists(os.path.join(metrics_dir, 'old')):
            os.makedirs(os.path.join(metrics_dir, 'old'))
        for ex in existing_metrics_files:
            shutil.move(os.path.join(metrics_dir, ex), os.path.join(metrics_dir, 'old', ex))
        print "Moving old METRICS files..."

    # Create new METRICS dataframe file using current date-time:
    datestr = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    metrics_filepath = os.path.join(metrics_dir, 'roi_metrics_%s_%s_%s.hdf5' % (metric_hash, trace_type, datestr))
    METRICS, PASSTRIALS = calculate_metrics(DATA, filter_pupil=filter_pupil, pupil_params=pupil_params)
    datastore = pd.HDFStore(metrics_filepath, 'w')
    for roi in METRICS.keys():
        datastore[str(roi)] = METRICS[roi]
    datastore.close()
    os.chmod(metrics_filepath, S_IREAD|S_IRGRP|S_IROTH)
    with open(os.path.join(metrics_dir, 'pass_trials.json'), 'w') as f:
        json.dump(PASSTRIALS, f, indent=4, sort_keys=True)

    # REPORT to CL:
    print "*************************************"
    print "Got ROI METRICS."
    if filter_pupil is True:
        print "-------------------------------------"
        print "Used pupil data to filter ROI traces:"
        pp.pprint(pupil_params)
        print "....................................."
        print "Trials that passed threshold params:"
        print "....................................."
        for config in sorted(PASSTRIALS.keys(), key=natural_keys):
            print "    %s -- %i trials." % (config, len(PASSTRIALS[config]))
    else:
        print "-------------------------------------"
        print "No filtering done on ROI traces."
        print "Calculated metrics for ALL found trials."
    print "*************************************"

    return metrics_filepath

#%%
def get_roi_summary_stats(metrics_filepath, configs, trace_type='raw', create_new=False):
    '''
    Collate stimulus and metrics for each ROI's traces (i.e., trials).
    Main function is collate_roi_stats().

    OUTPUTS:

        <TRACEID_DIR>/metrics/roi_stats_<FILTERHASH>_<DATESTR>.hdf5
        File hierarchy:

            /df -- single group
                /dataframe
                    - roi
                    - config
                    - trial
                    - xpois
                    - ypos
                    - size
                    - yrot/morph (or ori/sf)
                    - zscore
                    - stimdf

    Returns:

        roistats_filepath -- path to above output

    '''
    # Load METRICS:
    METRICS = pd.HDFStore(metrics_filepath, 'r')

    # First, check if corresponding ROISTATS file exists:
    curr_metrics_dir = os.path.split(metrics_filepath)[0]
    metrics_hash = os.path.splitext(os.path.split(metrics_filepath)[-1])[0].split('roi_metrics_')[-1].split('_')[0]
    existing_files = [f for f in os.listdir(curr_metrics_dir) if 'roi_stats_%s_%s' % (metrics_hash, trace_type) in f]
    if create_new is False:
        print "---> Looking for existing ROI STATS..."
        try:
            assert len(existing_files) == 1, "Unable to find unique ROI-STATS file..."
            roistats_filepath = os.path.join(curr_metrics_dir, existing_files[0])
            ROISTATS = pd.HDFStore(roistats_filepath, 'r')
            if len(ROISTATS.keys()) == 0:
                create_new = True
            ROISTATS.close()
            METRICS.close()
            return roistats_filepath
        except Exception as e:
            create_new = True
            print "*** Creating new ROI-STATS file."
        finally:
            if 'ROISTATS' in locals():
                ROISTATS.close()

    try:
        # Move or get rid of existing files:
        if len(existing_files) > 0:
            if not os.path.exists(os.path.join(curr_metrics_dir, 'old')):
                os.makedirs(curr_metrics_dir, 'old')
            for ex in existing_files:
                shutil.move(os.path.join(curr_metrics_dir, ex), os.path.join(curr_metrics_dir, 'old', ex))
            print "Moving old STATS files..."

        print "--->  Collating ROI metrics......"
        datestr = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        roistats_filepath = os.path.join(curr_metrics_dir, 'roi_stats_%s_%s_%s.hdf5' % (metrics_hash, trace_type, datestr))
        ROISTATS = collate_roi_stats(METRICS, configs)
        datastore = pd.HDFStore(roistats_filepath, 'w')
        df = {'df': ROISTATS}
        for d in df:
            datastore[d] = df[d]
        datastore.close()
        os.chmod(roistats_filepath, S_IREAD|S_IRGRP|S_IROTH)
    except Exception as e:
        print "***ERROR in ROI-STATS!"
        traceback.print_exc()
    finally:
        METRICS.close()
        if 'datastore' in locals():
            datastore.close()

    return roistats_filepath

#%%
def format_stimconfigs(configs):
    
    stimconfigs = copy.deepcopy(configs)
    
    if 'frequency' in configs[configs.keys()[0]].keys():
        stimtype = 'grating'
    elif 'fps' in configs[configs.keys()[0]].keys():
        stimtype = 'movie'
    else:
        stimtype = 'image'
        
    # Split position into x,y:
    for config in stimconfigs.keys():
        stimconfigs[config]['xpos'] = configs[config]['position'][0]
        stimconfigs[config]['ypos'] = configs[config]['position'][1]
        stimconfigs[config]['size'] = configs[config]['scale'][0]
        stimconfigs[config].pop('position', None)
        stimconfigs[config].pop('scale', None)
        
        # stimulus-type specific variables:
        if stimtype == 'grating':
            stimconfigs[config]['sf'] = configs[config]['frequency']
            stimconfigs[config]['ori'] = configs[config]['rotation']
            stimconfigs[config].pop('frequency', None)
            stimconfigs[config].pop('rotation', None)
        else:
            transform_variables = ['object', 'xpos', 'ypos', 'size', 'yrot', 'morphlevel', 'stimtype']
            
            # FIgure out which is the LAST morph, i.e., last anchor:
            if os.path.splitext(configs[configs.keys()[0]]['filename'])[-1] == '.png':
                fns = [configs[c]['filename'] for c in configs.keys() if 'morph' in configs[c]['filename']]
                mlevels = sorted(list(set([int(fn.split('_')[0][5:]) for fn in fns])))
            elif 'fps' in configs[configs.keys()[0]].keys():
                fns = [configs[c]['filename'] for c in configs.keys() if 'Blob_M' in configs[c]['filename']]
                mlevels = sorted(list(set([int(fn.split('_')[1][1:]) for fn in fns])))   
            #print "FN parsed:", fns[0].split('_')
            if mlevels[-1] > 22:
                anchor2 = 106
            else:
                anchor2 = 22
            assert all([anchor2 > m for m in mlevels]), "Possibly incorrect morphlevel assignment (%i). Found morphs %s." % (anchor2, str(mlevels))

            
            if stimtype == 'image':
                imname = os.path.splitext(configs[config]['filename'])[0]
                if ('CamRot' in imname):
                    objectid = imname.split('_CamRot_')[0]
                    yrot = int(imname.split('_CamRot_y')[-1])
                    if 'N1' in imname or 'D1' in imname:
                        morphlevel = 0
                    elif 'N2' in imname or 'D2' in imname:
                        morphlevel = anchor2
                    elif 'morph' in imname:
                        morphlevel = int(imname.split('_CamRot_y')[0].split('morph')[-1])   
                elif '_zRot' in imname:
                    # Real-world objects:  format is 'IDENTIFIER_xRot0_yRot0_zRot0'
                    objectid = imname.split('_')[0]
                    yrot = int(imname.split('_')[3][4:])
                    morphlevel = 0
                elif 'morph' in imname: 
                    # These are morphs w/ old naming convention, 'CamRot' not in filename)
                    if '_y' not in imname and '_yrot' not in imname:
                        objectid = imname #'morph' #imname
                        yrot = 0
                        morphlevel = int(imname.split('morph')[-1])
                    else:
                        objectid = imname.split('_y')[0]
                        yrot = int(imname.split('_y')[-1])
                        morphlevel = int(imname.split('_y')[0].split('morph')[-1])
            elif stimtype == 'movie':
                imname = os.path.splitext(configs[config]['filename'])[0]
                objectid = imname.split('_movie')[0] #'_'.join(imname.split('_')[0:-1])
                if 'reverse' in imname:
                    yrot = -1
                else:
                    yrot = 1
                if imname.split('_')[1] == 'D1':
                    morphlevel = 0
                elif imname.split('_')[1] == 'D2':
                    morphlevel = anchor2
                elif imname.split('_')[1][0] == 'M':
                    # Blob_M11_Rot_y_etc.
                    morphlevel = int(imname.split('_')[1][1:])
                elif imname.split('_')[1] == 'morph':
                    # This is a full morph movie:
                    morphlevel = -1
                    
            stimconfigs[config]['object'] = objectid
            stimconfigs[config]['yrot'] = yrot
            stimconfigs[config]['morphlevel'] = morphlevel
            stimconfigs[config]['stimtype'] = stimtype
        
            for skey in stimconfigs[config].keys():
                if skey not in transform_variables:
                    stimconfigs[config].pop(skey, None)

    
    return stimconfigs

    
#%%
def collate_roi_stats(METRICS, configs):
    '''
    This is really just a formatting function for easier plotting.
    Takes information from METRICS hdf5 file (output of get_roi_metrics()) and
    creates a dataframe that combines:
        - stimulus-config info (xpos, ypos, size, object ID, and ori/sf for gratings or yrot/morph for objects)
        - roi and trial names
        - zscore calculations
    '''

    if 'frequency' in configs[configs.keys()[0]].keys():
        stimtype = 'grating'
    elif 'fps' in configs[configs.keys()[0]].keys():
        stimtype = 'movie'
    else:
        stimtype = 'image'
        
    sconfigs = format_stimconfigs(configs)
    

    # Sort metrics by stimulus-params and calculate sumamry stats:
    roistats_df =[]
    for ri,roi in enumerate(sorted(METRICS.keys(), key=natural_keys)):
        rdata = METRICS[roi]
        
        # filter funky trials:
        rdata = rdata[rdata['mean_stim_on']<20].reset_index()
        if ri % 10 == 0:
            print "... collating %i of %i rois" % (ri, len(METRICS.keys()))
            
        grouped = rdata.groupby(['config'])
        ntrials_per_cond = [len(g['trial']) for k,g in grouped]
        max_ntrials = max(ntrials_per_cond)
    
        # First, get all general info:
        df_main = pd.concat([pd.DataFrame({
                                'roi': np.tile(str(roi), (max_ntrials,)),
                                'config': np.tile(k, (max_ntrials,)),
                                'trial': np.pad(g['trial'].values, (0, max_ntrials - len(g['trial'])), mode='constant', constant_values=(np.nan, np.nan)),
                                'zscore': np.pad(g['zscore'].values, (0, max_ntrials - len(g['trial'])), mode='constant', constant_values=(np.nan, np.nan)),
                                'stim_df': np.pad(g['mean_stim_on'].values, (0, max_ntrials - len(g['trial'])), mode='constant', constant_values=(np.nan, np.nan)),
                                'baseline_df': np.pad(g['mean_baseline'].values, (0, max_ntrials - len(g['trial'])), mode='constant', constant_values=(np.nan, np.nan))
                                }) for k,g in grouped], axis=0)
        # Add eye info:
        df_eye = pd.DataFrame()
        if 'pupil_size_stimulus' in rdata.keys():
            df_eye = pd.concat([pd.DataFrame({
                                'pupil_size_baseline': np.pad(g['pupil_size_baseline'].values, (0, max_ntrials - len(g['trial'])), mode='constant', constant_values=(np.nan, np.nan)),
                                'pupil_size_stimulus': np.pad(g['pupil_size_stimulus'].values, (0, max_ntrials - len(g['trial'])), mode='constant', constant_values=(np.nan, np.nan)),
                                'pupil_dist_stimulus': np.pad(g['pupil_dist_stimulus'].values, (0, max_ntrials - len(g['trial'])), mode='constant', constant_values=(np.nan, np.nan)),
                                'pupil_dist_baseline': np.pad(g['pupil_dist_baseline'].values, (0, max_ntrials - len(g['trial'])), mode='constant', constant_values=(np.nan, np.nan)),
                                'pupil_nblinks_stim': np.pad(g['pupil_nblinks_stim'].values, (0, max_ntrials - len(g['trial'])), mode='constant', constant_values=(np.nan, np.nan)),
                                'pupil_nblinks_baseline': np.pad(g['pupil_nblinks_baseline'].values, (0, max_ntrials - len(g['trial'])), mode='constant', constant_values=(np.nan, np.nan))
                                }) for k,g in grouped], axis=0)
        df_main = pd.concat([df_main, df_eye], axis=1)
        
        # Add stimulus-specific info:
        stimlist = []
        for k,g in grouped:
            tmpdf = dict((k, np.tile(v, max_ntrials)) for k,v in sconfigs[k].iteritems())
            stimlist.append(pd.DataFrame(tmpdf))  
        df_stim = pd.concat(stimlist, axis=0)
    
        # Put all DFs together:
        df_main = pd.concat([df_main, df_stim], axis=1)

        # Append current config info for all trials ROI DF:
        roistats_df.append(df_main)

    # Concatenate all trials across all configs for all ROIs into main dataframe:
    ROISTATS = pd.concat(roistats_df, axis=0)

    return ROISTATS

#%%
def align_roi_traces(trace_type, TID, si_info, traceid_dir, run_dir, iti_pre=1.0, iti_post=None, create_new=False,
                     filter_pupil=False, pupil_radius_min=None, pupil_radius_max=None, pupil_dist_thr=None):
    # Get paradigm/AUX info:
    # =============================================================================
    paradigm_dir = os.path.join(run_dir, 'paradigm')
    trial_info = get_alignment_specs(paradigm_dir, si_info, iti_pre=iti_pre, iti_post=iti_post)

    print "-------------------------------------------------------------------"
    print "Getting frame indices for trial epochs..."
    parsed_frames_filepath = assign_frames_to_trials(si_info, trial_info, paradigm_dir, create_new=create_new)


    # Get all unique stimulus configurations:
    # =========================================================================
    print "-----------------------------------------------------------------------"
    print "Getting stimulus configs..."
    configs, stimtype = get_stimulus_configs(trial_info)

    #%
    # Group ROI time-courses for each trial by stimulus config:
    # =========================================================================
    excluded_tiffs = TID['PARAMS']['excluded_tiffs']
    print "-------------------------------------------------------------------"
    print "Getting ROI timecourses by trial and stim config"
    print "Current trace ID - %s - excludes %i tiffs." % (TID['trace_id'], len(excluded_tiffs))
    roi_trials_by_stim_path = group_rois_by_trial_type(traceid_dir, parsed_frames_filepath, trial_info, si_info, excluded_tiffs=excluded_tiffs, create_new=create_new)
    #%

    # FILTER DATA with PUPIL PARAM thresholds, if relevant:
    # =========================================================================
    pupil_params = None
    #eye_info = None
    print "-------------------------------------------------------------------"
    print "Loading EYE INFO."
    eye_info = load_eye_data(run_dir)
    if filter_pupil is True:
        if eye_info is None:
            filter_pupil = False
            pupil_params = None
        else:
            pupil_params = set_pupil_params(radius_min=pupil_radius_min,
                                            radius_max=pupil_radius_max,
                                            dist_thr=pupil_dist_thr,
                                            create_empty=False)
            print "Pupil params requested:"
            pp.pprint(pupil_params)

    # GET ALL DATA into a dataframe:
    # =========================================================================
    print "-------------------------------------------------------------------"
    print "Getting ROIDATA into dataframe."
    roitrials_datestr = os.path.splitext(os.path.split(roi_trials_by_stim_path)[-1])[0]
    roitrials_hash = roitrials_datestr.split('_')[-1]
    roidata_filepath = os.path.join(traceid_dir, 'ROIDATA_%s_%s.hdf5' % (roitrials_hash, trace_type))
    if create_new is False:
        try:
            print "--> Trying to load existing file..."
            print "--> Loading ROIDATA file: %s" % roidata_filepath
            DATA = pd.HDFStore(roidata_filepath, 'r')
            assert len(DATA) > 0, "Empty ROIDATA file!"
        except Exception as e:
            create_new = True
            if 'DATA' in locals():
                DATA.close()

    if create_new is True:
        # First move old files:
        existing_df_files = [f for f in os.listdir(traceid_dir) if 'ROIDATA_' in f and trace_type in f and f.endswith('hdf5')]
        old_dir = os.path.join(traceid_dir, 'old')
        if not os.path.exists(old_dir):
            os.makedirs(old_dir)
        for e in existing_df_files:
            shutil.move(os.path.join(traceid_dir, e), os.path.join(old_dir, e))
        print "Moving old ROIDATA files..."

        # Align extracted traces into trials, and create dataframe:
        DATA = traces_to_trials(trial_info, si_info, configs, roi_trials_by_stim_path, trace_type=trace_type, eye_info=eye_info)

        # Save dataframe with same datestr as roi_trials.hdf5 file in traceid dir:
        datastore = pd.HDFStore(roidata_filepath, 'w')
        for roi in DATA.keys():
            datastore[roi] = DATA[roi]
        datastore.close()
        os.chmod(roidata_filepath, S_IREAD|S_IRGRP|S_IROTH)

    print "Got ROIDATA -- trials parsed by stim-config for each ROI."
    print "Saved ROIDATA hdf5 to: %s" % roidata_filepath

    # Save trial-alignment info with hash:
    alignment_info_filepath = os.path.join(traceid_dir, 'event_alignment_%s.json' % roitrials_hash)
    # Move old jsons if exist:
    existing_alignment_info = [e for e in os.listdir(traceid_dir) if 'event_alignment' in e and e.endswith('json') and not roitrials_hash in e]
    if len(existing_alignment_info) > 0:
        print "Moving old trial-alignment info files."
        if not os.path.exists(os.path.join(traceid_dir, 'old')):
            os.makedirs(os.path.join(traceid_dir, 'old'))
        for e in existing_alignment_info:
            shutil.move(os.path.join(traceid_dir, e), os.path.join(traceid_dir, 'old', e))
    with open(alignment_info_filepath, 'w') as f:
        json.dump(trial_info, f, sort_keys=True, indent=4)


    # Calculate metrics based on trial-filtering (eyetracker info):
    print "-------------------------------------------------------------------"
    print "Getting ROI METRICS."
    metrics_filepath = get_roi_metrics(roidata_filepath, configs, traceid_dir, trace_type=trace_type, filter_pupil=filter_pupil, pupil_params=pupil_params, create_new=create_new)

    # Cacluate some metrics, and plot tuning curves:
    print "-------------------------------------------------------------------"
    print "Collating ROI metrics into dataframe."
    roistats_filepath = get_roi_summary_stats(metrics_filepath, configs, trace_type=trace_type, create_new=create_new)

    return roidata_filepath, roistats_filepath


#%%

def extract_options(options):

    choices_tracetype = ('raw', 'raw_fissa', 'denoised_nmf', 'np_corrected_fissa', 'neuropil_fissa', 'np_subtracted', 'neuropil')
    default_tracetype = 'raw'

    parser = optparse.OptionParser()

    parser.add_option('-D', '--root', action='store', dest='rootdir', default='/nas/volume1/2photon/data', help='data root dir (root project dir containing all animalids) [default: /nas/volume1/2photon/data, /n/coxfs01/2pdata if --slurm]')
    parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')

    # Set specific session/run for current animal:
    parser.add_option('-S', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID')
    parser.add_option('-A', '--acq', action='store', dest='acquisition', default='FOV1', help="acquisition folder (ex: 'FOV1_zoom3x') [default: FOV1]")
    parser.add_option('-R', '--run', action='store', dest='run', default='', help="name of run dir containing tiffs to be processed (ex: gratings_phasemod_run1)")
    parser.add_option('--slurm', action='store_true', dest='slurm', default=False, help="set if running as SLURM job on Odyssey")
    parser.add_option('-t', '--trace-id', action='store', dest='trace_id', default='', help="Trace ID for current trace set (created with set_trace_params.py, e.g., traces001, traces020, etc.)")

    parser.add_option('-b', '--baseline', action="store",
                      dest="iti_pre", default=1.0, help="Time (s) pre-stimulus to use for baseline. [default: 1.0]")

    parser.add_option('-T', '--trace-type', type='choice', choices=choices_tracetype, action='store', dest='trace_type', default=default_tracetype, help="Type of timecourse to plot PSTHs. Valid choices: %s [default: %s]" % (choices_tracetype, default_tracetype))

    parser.add_option('--new', action="store_true",
                      dest="create_new", default=False, help="Set flag to create new output files (/paradigm/parsed_frames.hdf5, roi_trials.hdf5")

    # Only need to set these if using custom-paradigm file:
    parser.add_option('--custom', action="store_true",
                      dest="custom_mw", default=False, help="Not using MW (custom params must be specified)")
    parser.add_option('--order', action="store_true",
                      dest="same_order", default=False, help="Set if same stimulus order across all files (1 stimorder.txt)")
    parser.add_option('-O', '--stimon', action="store",
                      dest="stim_on_sec", default=0, help="Time (s) stimulus ON.")
    parser.add_option('-v', '--vol', action="store",
                      dest="vols_per_trial", default=0, help="Num volumes per trial. Specifiy if custom_mw=True")
    parser.add_option('-V', '--first', action="store",
                      dest="first_stim_volume_num", default=0, help="First volume stimulus occurs (py-indexed). Specifiy if custom_mw=True")

    # What metrics to calculate:
#    parser.add_option('--metric', action="store",
#                      dest="roi_metric", default="zscore", help="ROI metric to use for tuning curves [default: 'zscore']")

    # Pupil filtering info:
    parser.add_option('--no-pupil', action="store_false",
                      dest="filter_pupil", default=True, help="Set flag NOT to filter PSTH traces by pupil threshold params")
    parser.add_option('-s', '--radius-min', action="store",
                      dest="pupil_radius_min", default=25, help="Cut-off for smnallest pupil radius, if --pupil set [default: 25]")
    parser.add_option('-B', '--radius-max', action="store",
                      dest="pupil_radius_max", default=65, help="Cut-off for biggest pupil radius, if --pupil set [default: 65]")
    parser.add_option('-d', '--dist', action="store",
                      dest="pupil_dist_thr", default=5, help="Cut-off for pupil distance from start, if --pupil set [default: 5]")
#    parser.add_option('-x', '--blinks', action="store",
#                      dest="pupil_max_nblinks", default=1, help="Cut-off for N blinks allowed in trial, if --pupil set [default: 1 (i.e., 0 blinks allowed)]")


    (options, args) = parser.parse_args(options)

    return options





#%%
#
#options = ['-D', '/mnt/odyssey', '-i', 'CE077', '-S', '20180425', '-A', 'FOV2_zoom1x',
#           '-R', 'gratings_run1',
#           '-T', 'np_subtracted', '-t', 'traces001',
#           '--no-pupil', '--new',
#           '-b', 2.0]
#
#options = ['-D', '/mnt/odyssey', '-i', 'CE077', '-S', '20180516', '-A', 'FOV1_zoom1x',
#           '-R', 'blobs_movies_run2',
#           '-T', 'np_subtracted', '-t', 'traces001',
#           '--no-pupil', '--new',
#           '-b', 2.0]
#
options = ['-D', '/mnt/odyssey', '-i', 'CE077', '-S', '20180518',
        '-A', 'FOV1_zoom1x', '-R', 'blobs_dynamic_run3', '-t', 'traces001',
        '--no-pupil',
        '-T', 'np_subtracted',
        '-b', 2.0]


 #%%
# Set USER INPUT options:

def create_roi_dataframes(options):
    
    options = extract_options(options)

    rootdir = options.rootdir
    slurm = options.slurm
    if slurm is True and 'coxfs01' not in rootdir:
        rootdir = '/n/coxfs01/2p-data'
    animalid = options.animalid
    session = options.session
    acquisition = options.acquisition
    run = options.run
    trace_id = options.trace_id

    iti_pre = float(options.iti_pre)

    #custom_mw = options.custom_mw

    trace_type = options.trace_type
    create_new = options.create_new

    filter_pupil = options.filter_pupil
    pupil_radius_max = float(options.pupil_radius_max)
    pupil_radius_min = float(options.pupil_radius_min)
    pupil_dist_thr = float(options.pupil_dist_thr)

    #%
    # =============================================================================
    # Get meta/SI info for RUN:
    # =============================================================================
    run_dir = os.path.join(rootdir, animalid, session, acquisition, run)
    #si_info = {'framerate': 44.67, 'volumerate': 44.67 }
    si_info = get_frame_info(run_dir)

    # Load TRACE ID info:
    # =========================================================================
    TID = load_TID(run_dir, trace_id)
    traceid_dir = TID['DST']
    if rootdir not in traceid_dir:
        orig_root = traceid_dir.split('/%s/%s' % (animalid, session))[0]
        traceid_dir = traceid_dir.replace(orig_root, rootdir)
        print "Replacing orig root with dir:", traceid_dir
        #trace_hash = TID['trace_hash']

    # Assign frame indices for specified trial epochs:
    # =========================================================================
    roidata_filepath, roistats_filepath = align_roi_traces(trace_type, TID, si_info, traceid_dir, run_dir,
                                                           iti_pre=iti_pre,
                                                           create_new=create_new,
                                                           filter_pupil=filter_pupil,
                                                           pupil_radius_min=pupil_radius_min,
                                                           pupil_radius_max=pupil_radius_max,
                                                           pupil_dist_thr=pupil_dist_thr)

    return roidata_filepath, roistats_filepath

#%%

def main(options):

    roidata_filepath, roistats_filepath = create_roi_dataframes(options)

    print "*******************************************************************"
    print "Done aligning all acquisition events!"
    print "-------------------------------------------------------------------"
    print "DATA frame saved to: %s" % roidata_filepath
    print "ROI STATS info saved to: %s" % roistats_filepath
    print "*******************************************************************"


#%%

if __name__ == '__main__':
    main(sys.argv[1:])


#%%
# =============================================================================
# Plot included/excluded trials with eyetracker info
# =============================================================================


        #%%
    # =============================================================================
    # Make and save scatterplots of trial response vs. eye features:
    # =============================================================================
#
#        sns.set_style("darkgrid", {"axes.facecolor": ".9"})
#        roi_trials = h5py.File(roi_trials_by_stim_path, 'r')   # Load ROI TRIALS file
#        parsed_frames = h5py.File(parsed_frames_filepath, 'r')  # Load PARSED FRAMES output file:
#
#        roi_scatter_stim_dir = os.path.join(traceid_dir, 'figures', 'eye_scatterplots','stimulation')
#        if not os.path.exists(roi_scatter_stim_dir):
#            os.makedirs(roi_scatter_stim_dir)
#        print "Saving scatter plots to: %s" % roi_scatter_stim_dir
#
#
#        for roi in roi_list:
#            #%
#            print roi
#            if nrows==1:
#                figwidth_multiplier = ncols*1
#            else:
#                figwidth_multiplier = 3
#
#
#            for img in subplot_stimlist.keys():
#                curr_subplots = subplot_stimlist[img]
#                if stimid_only is True:
#                    figname = 'all_objects_default_pos_size'
#                else:
#                    figname = '%s_pos%i_size%i' % (os.path.splitext(img)[0], len(position_vals), len(size_vals))
#
#                fig, axs = pl.subplots(
#                    nrows=nrows,
#                    ncols=ncols,
#                    sharex=True,
#                    sharey=True,
#                    figsize=(figure_height*figwidth_multiplier,figure_height))
#
#                row=0
#                col=0
#                plotidx = 0
#
#                nframes_on = parsed_frames['trial00001']['frames_in_run'].attrs['stim_dur_sec'] * volumerate
#                stim_dur = trialdict['trial00001']['stim_dur_ms']/1E3
#                iti_dur = trialdict['trial00001']['iti_dur_ms']/1E3
#                tpoints = [int(i) for i in np.arange(-1*iti_pre, stim_dur+iti_dur)]
#
#                #roi = 'roi00003'
#                for configname in curr_subplots:
#                    df_values = []
#                    pupil_rad = []
#                    curr_slice = roi_trials[configname][roi].attrs['slice']
#                    roi_in_slice = roi_trials[configname][roi].attrs['idx_in_slice']
#
#                    if col==(ncols) and nrows>1:
#                        row += 1
#                        col = 0
#                    if len(axs.shape)>1:
#                        ax_curr = axs[row, col] #, col]
#                    else:
#                        ax_curr = axs[col]
#
#                    stim_trials = sorted([t for t in roi_trials[configname][roi].keys()], key=natural_keys)
#                    nvols = max([roi_trials[configname][roi][t][trace_type].shape[0] for t in stim_trials])
#                    ntrials = len(stim_trials)
#                    trialmat = np.ones((ntrials, nvols)) * np.nan
#                    dfmat = []
#
#                    first_on = int(min([[i for i in roi_trials[configname][roi][t].attrs['frame_idxs']].index(roi_trials[configname][roi][t].attrs['volume_stim_on']) for t in stim_trials]))
#                    tsecs = (np.arange(0, nvols) - first_on ) / volumerate
#
#                    if 'grating' in stimtype:
#                        stimname = 'Ori %.0f, SF: %.2f' % (configs[configname]['rotation'], configs[configname]['frequency'])
#                    else:
#                        stimname = '%s- pos (%.1f, %.1f) - siz %.1f' % (os.path.splitext(configs[configname]['filename'])[0], configs[configname]['position'][0], configs[configname]['position'][1], configs[configname]['scale'][0])
#
#                    ax_curr.annotate(stimname,xy=(0.1,1), xycoords='axes fraction', horizontalalignment='middle', verticalalignment='top', weight='bold')
#
#                    for tidx, trial in enumerate(sorted(stim_trials, key=natural_keys)):
#                        trial_timecourse = roi_trials[configname][roi][trial][trace_type]
#                        curr_on = int([i for i in roi_trials[configname][roi][trial].attrs['frame_idxs']].index(int(roi_trials[configname][roi][trial].attrs['volume_stim_on'])))
#                        trialmat[tidx, first_on:first_on+len(trial_timecourse[curr_on:])] = trial_timecourse[curr_on:]
#                        if first_on < curr_on:
#                            trialmat[tidx, 0:first_on] = trial_timecourse[1:curr_on]
#                        else:
#                            trialmat[tidx, 0:first_on] = trial_timecourse[0:curr_on]
#                        #trialmat[tidx, first_on-curr_on:first_on] = trace[0:curr_on]
#
#                        baseline = np.nanmean(trialmat[tidx, 0:first_on]) #[0:on_idx])
#                        if baseline == 0 or baseline == np.nan:
#                            df = np.ones(trialmat[tidx,:].shape) * np.nan
#                        else:
#                            df = (trialmat[tidx,:] - baseline) / baseline
#
#                        df_values.append(np.nanmean(df[first_on:first_on+int(nframes_on)]))
#                        pupil_rad.append(float(eye_info[trial]['pupil_size_stim']))
#
#                    ax_curr.plot(pupil_rad,df_values,'ob')
#                    col = col + 1
#                    plotidx += 1
#
#            fig_fn = '%s_%s_%s_%s.png' % (roi, curr_slice, roi_in_slice, trace_type)
#            figure_file = os.path.join(roi_scatter_stim_dir,fig_fn)
#
#            pl.savefig(figure_file, bbox_inches='tight')
#            pl.close()
#
#
#
#        roi_scatter_base_dir = os.path.join(traceid_dir, 'figures', 'eye_scatterplots','baseline')
#        if not os.path.exists(roi_scatter_base_dir):
#            os.makedirs(roi_scatter_base_dir)
#        print "Saving scatter plots to: %s" % roi_scatter_base_dir
#
#        for roi in roi_list:
#            #%
#            print roi
#            if nrows==1:
#                figwidth_multiplier = ncols*1
#            else:
#                figwidth_multiplier = 1
#
#
#            for img in subplot_stimlist.keys():
#                curr_subplots = subplot_stimlist[img]
#                if stimid_only is True:
#                    figname = 'all_objects_default_pos_size'
#                else:
#                    figname = '%s_pos%i_size%i' % (os.path.splitext(img)[0], len(position_vals), len(size_vals))
#
#                fig, axs = pl.subplots(
#                    nrows=nrows,
#                    ncols=ncols,
#                    sharex=True,
#                    sharey=True,
#                    figsize=(figure_height*2,figure_height))
#
#                row=0
#                col=0
#                plotidx = 0
#
#                nframes_on = parsed_frames['trial00001']['frames_in_run'].attrs['stim_dur_sec'] * volumerate
#                stim_dur = trialdict['trial00001']['stim_dur_ms']/1E3
#                iti_dur = trialdict['trial00001']['iti_dur_ms']/1E3
#                tpoints = [int(i) for i in np.arange(-1*iti_pre, stim_dur+iti_dur)]
#
#                #roi = 'roi00003'
#                for configname in curr_subplots:
#                    df_values = []
#                    pupil_rad = []
#                    curr_slice = roi_trials[configname][roi].attrs['slice']
#                    roi_in_slice = roi_trials[configname][roi].attrs['idx_in_slice']
#
#                    if col==(ncols) and nrows>1:
#                        row += 1
#                        col = 0
#                    if len(axs.shape)>1:
#                        ax_curr = axs[row, col] #, col]
#                    else:
#                        ax_curr = axs[col]
#
#                    stim_trials = sorted([t for t in roi_trials[configname][roi].keys()], key=natural_keys)
#                    nvols = max([roi_trials[configname][roi][t][trace_type].shape[0] for t in stim_trials])
#                    ntrials = len(stim_trials)
#                    trialmat = np.ones((ntrials, nvols)) * np.nan
#                    dfmat = []
#
#                    first_on = int(min([[i for i in roi_trials[configname][roi][t].attrs['frame_idxs']].index(roi_trials[configname][roi][t].attrs['volume_stim_on']) for t in stim_trials]))
#                    tsecs = (np.arange(0, nvols) - first_on ) / volumerate
#
#                    if 'grating' in stimtype:
#                        stimname = 'Ori %.0f, SF: %.2f' % (configs[configname]['rotation'], configs[configname]['frequency'])
#                    else:
#                        stimname = '%s- pos (%.1f, %.1f) - siz %.1f' % (os.path.splitext(configs[configname]['filename'])[0], configs[configname]['position'][0], configs[configname]['position'][1], configs[configname]['scale'][0])
#
#                    ax_curr.annotate(stimname,xy=(0.1,1), xycoords='axes fraction', horizontalalignment='middle', verticalalignment='top', weight='bold')
#
#                    for tidx, trial in enumerate(sorted(stim_trials, key=natural_keys)):
#                        trial_timecourse = roi_trials[configname][roi][trial][trace_type]
#                        curr_on = int([i for i in roi_trials[configname][roi][trial].attrs['frame_idxs']].index(int(roi_trials[configname][roi][trial].attrs['volume_stim_on'])))
#                        trialmat[tidx, first_on:first_on+len(trial_timecourse[curr_on:])] = trial_timecourse[curr_on:]
#                        if first_on < curr_on:
#                            trialmat[tidx, 0:first_on] = trial_timecourse[1:curr_on]
#                        else:
#                            trialmat[tidx, 0:first_on] = trial_timecourse[0:curr_on]
#                        #trialmat[tidx, first_on-curr_on:first_on] = trace[0:curr_on]
#
#                        baseline = np.nanmean(trialmat[tidx, 0:first_on]) #[0:on_idx])
#                        if baseline == 0 or baseline == np.nan:
#                            df = np.ones(trialmat[tidx,:].shape) * np.nan
#                        else:
#                            df = (trialmat[tidx,:] - baseline) / baseline
#
#                        df_values.append(np.nanmean(df[first_on:first_on+int(nframes_on)]))
#                        pupil_rad.append(float(eye_info[trial]['pupil_size_baseline']))
#
#                    ax_curr.plot(pupil_rad,df_values,'ob')
#                    col = col + 1
#                    plotidx += 1
#
#            fig_fn = '%s_%s_%s_%s.png' % (roi, curr_slice, roi_in_slice, trace_type)
#            figure_file = os.path.join(roi_scatter_base_dir,fig_fn)
#
#            pl.savefig(figure_file, bbox_inches='tight')
#            pl.close()
#        roi_trials.close()
#        parsed_frames.close()



