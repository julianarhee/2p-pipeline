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

    c.  PSTHs, if relevant for each ROI:

    <TRACEID_DIR>/figures/psths/<TRACE_TYPE>/roiXXXXX_SliceXX_IDX.png
    -- trace_type = 'raw' or 'denoised_nmf' for now
    -- IDX = roi idx in current roi set


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

from scipy import stats
import pandas as pd
import seaborn as sns
import pylab as pl
import numpy as np
from pipeline.python.utils import natural_keys, hash_file_read_only, print_elapsed_time, hash_file
from pipeline.python.traces.utils import get_frame_info
pp = pprint.PrettyPrinter(indent=4)


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
def get_alignment_specs(paradigm_dir, si_info, custom_mw=False, options=None):
    trial_epoch_info = {}
    iti_pre = options.iti_pre
    same_order = options.same_order
    run = options.run

    if custom_mw is True:
        trial_fn = None
        stimtype = None
        # Get trial info if custom (non-MW) stim presentation protocols:
        # -------------------------------------------------------------------------
        try:
            stim_on_sec = float(options.stim_on_sec) #2. # 0.5
            first_stimulus_volume_num = int(options.first_stim_volume_num) #50
            vols_per_trial = float(options.vols_per_trial) # 15
            iti_full = (vols_per_trial/si_info['volumerate']) - stim_on_sec
            iti_post = iti_full - iti_pre
            print "==============================================================="
            print "Using CUSTOM trial-info (not MW)."
            print "==============================================================="
            print "First stim on:", first_stimulus_volume_num
            print "Volumes per trial:", vols_per_trial
            print "ITI POST (s):", iti_post
            print "ITT full (s):", iti_full
            print "TRIAL dur (s):", stim_on_sec + iti_full
            print "Vols per trial (calc):", (stim_on_sec + iti_pre + iti_post) * si_info['volumerate']
            print "==============================================================="

            # Get stim-order files:
            stimorder_fns = sorted([f for f in os.listdir(paradigm_dir) if 'stimorder' in f and f.endswith('txt')], key=natural_keys)
            print "Found %i stim-order files, and %i TIFFs." % (len(stimorder_fns), si_info['ntiffs'])
            if len(stimorder_fns) < si_info['ntiffs']:
                if same_order: # Same stimulus order for each file (complete set)
                    stimorder_fns = np.tile(stimorder_fns, [si_info['ntiffs'],])

        except Exception as e:
            print "---------------------------------------------------------------"
            print "Using CUSTOM trial-info. Use -h to check required options:"
            print "---------------------------------------------------------------"
            print "- volume num of 1st stimulus in tif"
            print "- N vols per trial"
            print "- duration (s) of stimulus-on period in trial"
            print " - stimorder.txt file with each line containing IDX of stim id."
            print " - whether the same order of stimuli are presented across all files."
            print "Aborting with error:"
            print "---------------------------------------------------------------"
            print e
            print "---------------------------------------------------------------"

    else:
        stimorder_fns = None
        first_stimulus_volume_num = None
        vols_per_trial = None

        ### Get PARADIGM INFO if using standard MW:
        # -------------------------------------------------------------------------
        try:
            trial_fn = [t for t in os.listdir(paradigm_dir) if 'trials_' in t and t.endswith('json')][0]
            parsed_trials_path = os.path.join(paradigm_dir, trial_fn)
            trialdict = load_parsed_trials(parsed_trials_path)
            trial_list = sorted(trialdict.keys(), key=natural_keys)
            stimtype = trialdict[trial_list[0]]['stimuli']['type']


            # Get presentation info (should be constant across trials and files):
            trial_list = sorted(trialdict.keys(), key=natural_keys)
            stim_durs = [round(trialdict[t]['stim_dur_ms']/1E3, 1)for t in trial_list]
            assert len(list(set(stim_durs))) == 1, "More than 1 stim_dur found..."
            stim_on_sec = stim_durs[0]
            iti_durs = [round(trialdict[t]['iti_dur_ms']/1E3, 1) for t in trial_list]
            assert len(list(set(iti_durs))) == 1, "More than 1 iti_dur found..."
            iti_full = iti_durs[0]
            iti_post = iti_full - iti_pre

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
        nframes_on = stim_on_sec * si_info['framerate'] #framerate #int(round(stim_on_sec * volumerate))
        nframes_iti_pre = iti_pre * si_info['framerate'] #framerate
        nframes_iti_post = iti_post*si_info['framerate'] #framerate # int(round(iti_post * volumerate))
        nframes_iti_full = iti_full * si_info['framerate'] #framerate #int(round(iti_full * volumerate))
        nframes_post_onset = (stim_on_sec + iti_post) * si_info['framerate'] #framerate
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
    trial_epoch_info['custom_mw'] = custom_mw

    return trial_epoch_info

#%%
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
        print "Found %i frames from flyback correction." % len(frame_idxs)
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
    si_info['volumerate'] = volumerate
    si_info['framerate'] = framerate
    si_info['nslices'] = nslices
    si_info['nchannels'] = nchannels
    si_info['ntiffs'] = ntiffs


    return si_info

#%%

def load_TID(run_dir, trace_id):
    run = os.path.split(run_dir)[-1]
    trace_basedir = os.path.join(run_dir, 'traces')
    try:
        tracedict_path = os.path.join(trace_basedir, 'traceids_%s.json' % run)
        with open(tracedict_path, 'r') as tr:
            tracedict = json.load(tr)
        TID = tracedict[trace_id]
        print "USING TRACE ID: %s" % TID['trace_id']
        pp.pprint(TID)
    except Exception as e:
        print "Unable to load TRACE params info: %s:" % trace_id
        print "Aborting with error:"
        print e

    return TID

#%%

def assign_frames_to_trials(si_info, trial_info, paradigm_dir, create_new=True):
    print "-----------------------------------------------------------------------"
    print "Getting frame indices for trial epochs..."

    run = os.path.split(os.path.split(paradigm_dir)[0])[-1]
    # First check if parsed frame file already exists:
    existing_parsed_frames_fns = sorted([t for t in os.listdir(paradigm_dir) if 'parsed_frames_' in t and t.endswith('hdf5')], key=natural_keys)
    existing_parsed_frames_fns.sort(key=lambda x: os.stat(os.path.join(paradigm_dir, x)).st_mtime) # Sort by date modified
    if len(existing_parsed_frames_fns) > 0 and create_new is False:
        parsed_frames_filepath = os.path.join(paradigm_dir, existing_parsed_frames_fns[-1]) # Get most recently modified file
        print "Got existing parsed-frames file:", parsed_frames_filepath
    else:
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
                #frames_tsecs = np.arange(0, nvolumes)*(1/volumerate)

                if trial_info['custom_mw'] is True:
                    stimorder_fns = trial_info['simorder_source']
                    with open(os.path.join(paradigm_dir, stimorder_fns[tiffnum])) as f:
                        stimorder_data = f.readlines()
                    stimorder = [l.strip() for l in stimorder_data]
                else:
                    stimorder = [trialdict[t]['stimuli']['stimulus'] for t in trial_list\
                                     if trialdict[t]['block_idx'] == tiffnum]
                    trials_in_run = sorted([t for t in trial_list if trialdict[t]['block_idx'] == tiffnum], key=natural_keys)

                #unique_stims = sorted(set(stimorder), key=natural_keys)

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
                    #print "nframes", len(framenums)
                    #print "on idx", framenums.index(first_frame_on)

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

        #parsed_frames_filepath = hash_file_read_only(parsed_frames_filepath)

        # Get unique hash for current PARSED FRAMES file:
        parsed_frames_hash = hash_file(parsed_frames_filepath, hashtype='sha1') #hashlib.sha1(json.dumps(RUN, indent=4, sort_keys=True)).hexdigest()[0:6]

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

#% Rename FRAME file with hash:
#frame_file_hashid = hash_file(parsed_frames_filepath)
#framestruct_path = '%s_%s.hdf5' % (os.path.splitext(parsed_frames_filepath)[0], frame_file_hashid)
#os.rename(parsed_frames_filepath, framestruct_path)

#print "Finished assigning frame idxs across all tiffs to trials in run %s." % run
#print "Saved frame file to:", framestruct_path

#%%
def get_stimulus_configs(trial_info):
    paradigm_dir = os.path.split(trial_info['parsed_trials_source'])[0]
    trialdict = load_parsed_trials(trial_info['parsed_trials_source'])
    # Get presentation info (should be constant across trials and files):
    trial_list = sorted(trialdict.keys(), key=natural_keys)

    print "-----------------------------------------------------------------------"
    print "Getting stimulus configs..."

    #stimids = sorted(list(set([trialdict[t]['stimuli']['stimulus'] for t in trial_list])), key=natural_keys)
    stimtype = trialdict[trial_list[0]]['stimuli']['type']
    if 'grating' in stimtype:
        # Likely varying gabor types...
        stimparams = [k for k in trialdict[trial_list[0]]['stimuli'].keys() if not (k=='stimulus' or k=='type')]
    else:
        stimparams = [k for k in trialdict[trial_list[0]]['stimuli'].keys() if not (k=='stimulus' or k=='type' or k=='filehash')]

    stimparams = sorted(stimparams, key=natural_keys)
    # Get all unique stimulus configurations (ID, filehash, position, size, rotation):
    allparams = []
    for param in stimparams:
        if isinstance(trialdict[trial_list[0]]['stimuli'][param], list):
            currvals = [tuple(trialdict[trial]['stimuli'][param]) for trial in trial_list]
        else:
            currvals = [trialdict[trial]['stimuli'][param] for trial in trial_list]
    #        if param == 'filepath':
    #            currvals = [os.path.split(f)[1] for f in currvals]

        #allparams.append(list(set(currvals)))
        allparams.append([i for i in list(set(currvals))])

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
        #stimids = sorted(list(set([trialdict[t]['stimuli']['stimulus'] for t in trial_list])), key=natural_keys)
        stimids = sorted(list(set([os.path.split(trialdict[t]['stimuli']['filepath'])[1] for t in trial_list])), key=natural_keys)
        filepaths = list(set([trialdict[trial]['stimuli']['filepath'] for trial in trial_list]))
        filehashes = list(set([trialdict[trial]['stimuli']['filehash'] for trial in trial_list]))

        assert len(filepaths) == len(stimids), "More than 1 file path per stim ID found!"
        assert len(filehashes) == len(stimids), "More than 1 file hash per stim ID found!"

        stimhash_combos = list(set([(trialdict[trial]['stimuli']['stimulus'], trialdict[trial]['stimuli']['filehash']) for trial in trial_list]))
        assert len(stimhash_combos) == len(stimids), "Bad stim ID - stim file hash combo..."
        stimhash = dict((stimid, hashval) for stimid, hashval in zip([v[0] for v in stimhash_combos], [v[1] for v in stimhash_combos]))

    print "Found %i unique stimulus configs." % len(configs.keys())

    # SAVE CONFIG info:
    config_filename = 'stimulus_configs.json'
    with open(os.path.join(paradigm_dir, config_filename), 'w') as f:
        json.dump(configs, f, sort_keys=True, indent=4)

    return configs, stimtype


#%%
def load_timecourses(traceid_dir):
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
        #roi_list = sorted([r for r in roi_timecourses.keys() if 'roi' in r], key=natural_keys)
        #bg_list = sorted([b for b in roi_timecourses.keys() if 'bg' in r], key=natural_keys)
        print "Loaded time-courses file: %s" % tcourse_fn
        print "Found %i ROIs total." % len(roi_list)
    except Exception as e:
        print "-------------------------------------------------------------------"
        print "Unable to load time courses for current trace set: %s" % trace_id
        print "File not found in dir: %s" % traceid_dir
        traceback.print_exc()
        print "Aborting with error:"
        print "-------------------------------------------------------------------"

    try:
        curr_slices = sorted(list(set([roi_timecourses[roi].attrs['slice'] for roi in roi_list])), key=natural_keys)
        print "ROIs are distributed across %i slices:" % len(curr_slices)
        print(curr_slices)
    except Exception as e:
        print "-------------------------------------------------------------------"
        print "Unable to load slices..."
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
    #trace_id = os.path.split(traceid_dir)[-1].split('_')[0]
    run = os.path.split(traceid_dir.split('/traces')[0])[-1]
    excluded_tiff_idxs = [int(tf[4:])-1 for tf in excluded_tiffs]

    trialdict = load_parsed_trials(trial_info['parsed_trials_source'])
    # Get presentation info (should be constant across trials and files):
    trial_list = sorted(trialdict.keys(), key=natural_keys)
    vol_idxs = si_info['vol_idxs']

    print "-----------------------------------------------------------------------"
    print "Getting ROI timecourses by trial and stim config"
    roi_tcourse_filepath, curr_slices, roi_list = load_timecourses(traceid_dir)
    roi_timecourses = h5py.File(roi_tcourse_filepath, 'r')  # Load ROI traces
    parsed_frames = h5py.File(parsed_frames_filepath, 'r')  # Load PARSED FRAMES output file:
    #sliceids = dict((curr_slices[s], s) for s in range(len(curr_slices)))

    # Check for stimulus configs:
    configs, stimtype = get_stimulus_configs(trial_info)

    # Create OUTFILE to save each ROI's time course for each trial, sorted by stimulus config
    # First check if ROI_TRIALS exist -- extraction takes awhile, and if just replotting, no need
    t_roitrials = time.time()
    existing_roi_trial_fns = sorted([t for t in os.listdir(traceid_dir) if 'roi_trials' in t and t.endswith('hdf5')], key=natural_keys)
    #print len(existing_roi_trial_fns)
    if len(existing_roi_trial_fns) > 0 and create_new is False:
        roi_trials_by_stim_path = os.path.join(traceid_dir, existing_roi_trial_fns[-1])
        print "TID %s -- Loaded ROI timecourses for run %s." % (trace_hash, run)
        print "ROI trial file path is: %s" % roi_trials_by_stim_path

        # CHeck file to make sure it is complete:
        roi_trials = h5py.File(roi_trials_by_stim_path, 'r')
        if not len(roi_trials.keys()) == len(configs.keys()):
            print "Incomplete stim-config list found in loaded roi-trials file."
            print "Found %i out of %i stim configs." % (len(roi_trials.keys()), len(configs.keys()))
            print "Creating new...!"
            create_new = True
    else:
        create_new = True

    #%
    if create_new is True:
        tstamp = datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")
        roi_trials_by_stim_path = os.path.join(traceid_dir, 'roi_trials_%s.hdf5' % tstamp)
        roi_trials = h5py.File(roi_trials_by_stim_path, 'w')

        try:
            print "TID %s -- Creating NEW ROI timecourses file, tstamp: %s" % (trace_hash, tstamp)
            for configname in sorted(configs.keys(), key=natural_keys):
                print "Getting all time-courses associated with STIM: %s" % configname
                currconfig = configs[configname]
                #pp.pprint(currconfig)
                curr_trials = [trial for trial in trial_list
                               if all(trialdict[trial]['stimuli'][param] == currconfig[param]
                               and trialdict[trial]['block_idx'] not in excluded_tiff_idxs for param in currconfig.keys())]

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

                tracemat = []
                for tidx, trial in enumerate(sorted(curr_trials, key=natural_keys)):
                    currtrial = trial # 'trial%03d' % int(tidx + 1)


                    curr_trial_volume_idxs = [vol_idxs[int(i)] for i in parsed_frames[trial]['frames_in_run']]
            #            slicenum = sliceids[tracestruct['roi00003'].attrs['slice']]
            #            slice_idxs = vol_idxs[3::nslices_full]
            #            tcourse_indices = [s for s in slice_idxs if s in curr_trial_volume_idxs]
            #            tcourse_indices = sorted(list(set(np.array(tcourse_indices))))
            #            if len(tcourse_indices)==30:
            #                print curr_stim, trial
                    stim_on_frame_idx = parsed_frames[trial]['frames_in_run'].attrs['stim_on_idx']
                    stim_on_volume_idx = vol_idxs[stim_on_frame_idx]
                    trial_idxs = sorted(list(set(curr_trial_volume_idxs)))
                    #print trial_idxs.index(stim_on_volume_idx)

                    for roi in roi_list:
                        #print roi
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

        #                tcourse_raw = roi_timecourses[roi]['timecourse']['raw'][trial_idxs]
        #
        #                tset = config_grp.create_dataset('/'.join((roi, currtrial)), tcourse_raw.shape, tcourse_raw.dtype)
        #                tset[...] = tcourse_raw
        #                tset.attrs['frame_on'] = stim_on_volume_idx #framestruct[trial]['frames_in_run'].attrs['stim_on_idx']
        #                tset.attrs['frame_idxs'] = trial_idxs

                        config_grp[roi].attrs['id_in_set'] = roi_timecourses[roi].attrs['id_in_set']
                        config_grp[roi].attrs['id_in_src'] = roi_timecourses[roi].attrs['id_in_src']
                        config_grp[roi].attrs['idx_in_slice'] = roi_timecourses[roi].attrs['idx_in_slice']
                        config_grp[roi].attrs['slice'] = roi_timecourses[roi].attrs['slice']

                        #tset.attrs['slice'] = roi_timecourses[roi].attrs['slice']
                        #tset.attrs['roi_slice_id'] = roi_timecourses[roi].attrs['id_in_slice']
                        #tset.attrs['stim_dur_sec'] = framestruct[trial]['frames_in_run'].attrs['stim_dur_sec']
        except Exception as e:
            print "--- ERROR in parsing ROI time courses by stimulus config. ---"
            print configname, trial, roi
            traceback.print_exc()
            print "-------------------------------------------------------------"
        finally:
            roi_trials.close()


        # Get unique hash for current PARSED FRAMES file:
        roi_trials_hash = hash_file(roi_trials_by_stim_path, hashtype='sha1') #hashlib.sha1(json.dumps(RUN, indent=4, sort_keys=True)).hexdigest()[0:6]

        # Check existing files:
        outdir = os.path.split(roi_trials_by_stim_path)[0]
        existing_files = [f for f in os.listdir(outdir) if 'roi_trials_' in f and f.endswith('hdf5') and roi_trials_hash not in f and tstamp not in f]
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

# Change config filepath for plotting:

def set_subplot_order(configs, stimtype, universal_scale=False):  # in percentage of the figure height
    plot_info = {}
    stiminfo = {}

    if 'grating' in stimtype:
        sfs = list(set([configs[c]['frequency'] for c in configs.keys()]))
        oris = list(set([configs[c]['rotation'] for c in configs.keys()]))

        noris = len(oris)
        nsfs = len(sfs)

        nrows = min([noris, nsfs])
        ncols = (nsfs * noris) / nrows

        tmp_list = []
        for sf in sorted(sfs):
            for oi in sorted(oris):
                match = [k for k in configs.keys() if configs[k]['rotation']==oi and configs[k]['frequency']==sf][0]
                tmp_list.append(match)
        stimid_only = True
        subplot_stimlist = dict()
        subplot_stimlist['defaultgratings'] = tmp_list
        #print subplot_stimlist

        stiminfo['sfs'] = sfs
        stiminfo['oris'] = oris

    else:
        configparams = configs[configs.keys()[0]].keys()
        for config in configs.keys():
            configs[config]['filename'] = os.path.split(configs[config]['filepath'])[1]

        # Create figure(s) based on stim configs:
        position_vals = list(set([tuple(configs[k]['position']) for k in configs.keys()]))
        size_vals = list(set([configs[k]['scale'][0] for k in configs.keys()]))
        img_vals = list(set([configs[k]['filename'] for k in configs.keys()]))
        if len(position_vals) > 1 or len(size_vals) > 1:
            stimid_only = False
        else:
            stimid_only = True

        if stimid_only is True:
            nfigures = 1
            nrows = int(np.ceil(np.sqrt(len(configs.keys()))))
            ncols = len(configs.keys()) / nrows
            img = img_vals[0]
            subplot_stimlist = dict()
            subplot_stimlist[img] = sorted(configs.keys(), key=lambda x: configs[x]['filename'])
        else:
            nfigures = len(img_vals)
            nrows = len(size_vals)
            ncols = len(position_vals)
            subplot_stimlist = dict()
            for img in img_vals:
                curr_img_configs = [c for c in configs.keys() if configs[c]['filename'] == img]
                subplot_stimlist[img] = sorted(curr_img_configs, key=lambda x: (configs[x].get('position'), configs[x].get('size')))

        stiminfo['position_vals'] = position_vals
        stiminfo['size_vals'] = size_vals
        stiminfo['img_vals'] = img_vals

    plot_info['stimuli'] = stiminfo
    plot_info['stimid_only'] = stimid_only
    plot_info['subplot_stimlist'] = subplot_stimlist
    plot_info['nrows'] = nrows
    plot_info['ncols'] = ncols

    figure_height, top_margin, bottom_margin =  set_figure_params(nrows, ncols)
    plot_info['figure_height'] = figure_height
    plot_info['top_margin'] = top_margin
    plot_info['bottom_margin'] = bottom_margin
    plot_info['universal_scale'] = universal_scale
#

    return plot_info #subplot_stimlist, nrows, ncols

#%%

def set_figure_params(nrows, ncols, fontsize_pt=20, dpi=72.27, spacer=20, top_margin=0.01, bottom_margin=0.05):

    # comput the matrix height in points and inches
    matrix_height_pt = fontsize_pt * nrows * spacer
    matrix_height_in = matrix_height_pt / dpi

    # compute the required figure height  # in percentage of the figure height
    figure_height = matrix_height_in / (1 - top_margin - bottom_margin)

    return figure_height, top_margin, bottom_margin


#%%


def traces_to_trials(trial_info, configs, roi_trials_by_stim_path, trace_type='raw', eye_info=None):
    roi=None; configname=None; trial=None
    #ROIS = dict()
    DATA = dict()

    # Load ROI list and traces:
    roi_trials = h5py.File(roi_trials_by_stim_path, 'r')   # Load ROI TRIALS file
    roi_list = sorted(roi_trials[roi_trials.keys()[0]].keys(), key=natural_keys)

    # Get info for TRIAL EPOCH for alignment:
    volumerate = trial_info['volumerate'] #parsed_frames.attrs['volumerate']
    iti_pre = trial_info['iti_pre']
    nframes_on = trial_info['nframes_on'] #parsed_frames['trial00001']['frames_in_run'].attrs['stim_dur_sec'] * volumerate
    stim_dur = trial_info['stim_on_sec'] #trialdict['trial00001']['stim_dur_ms']/1E3
    iti_dur = trial_info['iti_full'] #trialdict['trial00001']['iti_dur_ms']/1E3
    tpoints = [int(i) for i in np.arange(-1*iti_pre, stim_dur+iti_dur)]

    # Initialize ROI trial dict:
#    for configname in list(set(roi_trials.keys())):
#        ROIS[configname] = dict((roi, dict()) for roi in roi_list)

    try:
        for roi in roi_list:
            roi_dfs = []
            #%
            print roi
            for configname in sorted(roi_trials.keys(), key=natural_keys): #sorted(ROIs.keys(), key=natural_key):

                curr_slice = roi_trials[configname][roi].attrs['slice']
                roi_in_slice = roi_trials[configname][roi].attrs['idx_in_slice']

                stim_trials = sorted([t for t in roi_trials[configname][roi].keys()], key=natural_keys)
                nvols = max([roi_trials[configname][roi][t][trace_type].shape[0] for t in stim_trials])
                ntrials = len(stim_trials)

                # initialize TRIALMAT: each row is a trial, each column is a frame of that trial
                trialmat = np.ones((ntrials, nvols)) * np.nan
                dfmat = []

                first_on = int(min([[i for i in roi_trials[configname][roi][t].attrs['frame_idxs']].index(roi_trials[configname][roi][t].attrs['volume_stim_on']) for t in stim_trials]))
                tsecs = (np.arange(0, nvols) - first_on ) / volumerate

#                if 'grating' in trial_info['stimtype']:
#                    stimname = 'Ori %.0f, SF: %.2f' % (configs[configname]['rotation'], configs[configname]['frequency'])
#                else:
#                    stimname = '%s- pos (%.1f, %.1f) - siz %.1f' % (os.path.splitext(configs[configname]['filename'])[0], configs[configname]['position'][0], configs[configname]['position'][1], configs[configname]['scale'][0])

                for tidx, trial in enumerate(sorted(stim_trials, key=natural_keys)):
#                    if trial not in ROIS[configname][roi].keys():
#                        ROIS[configname][roi][trial] = dict()

                    trial_timecourse = roi_trials[configname][roi][trial][trace_type]
                    curr_on = int([i for i in roi_trials[configname][roi][trial].attrs['frame_idxs']].index(int(roi_trials[configname][roi][trial].attrs['volume_stim_on'])))

                    trialmat[tidx, first_on:first_on+len(trial_timecourse[curr_on:])] = trial_timecourse[curr_on:] #trialmat[tidx, first_on-curr_on:first_on] = trace[0:curr_on]
                    if first_on < curr_on:
                        trialmat[tidx, 0:first_on] = trial_timecourse[1:curr_on]
                    else:
                        trialmat[tidx, 0:first_on] = trial_timecourse[0:curr_on]

                    baseline = np.nanmean(trialmat[tidx, 0:first_on]) #[0:on_idx])
                    if baseline == 0 or baseline == np.nan:
                        df = np.ones(trialmat[tidx,:].shape) * np.nan
                    else:
                        df = (trialmat[tidx,:] - baseline) / baseline
                    dfmat.append(df)

                    if eye_info is not None:
                        eye_radius_baseline = eye_info[trial]['pupil_size_baseline']
                        eye_radius_stimulus = eye_info[trial]['pupil_size_stim']
                        eye_dist_baseline = eye_info[trial]['pupil_dist_baseline']
                        eye_dist_stimulus = eye_info[trial]['pupil_dist_stim']
                    else:
                        eye_radius_baseline = None
                        eye_radius_stimulus = None
                        eye_dist_baseline = None
                        eye_dist_stimulus = None


                    curr_slice = roi_trials[configname][roi].attrs['slice']
                    roi_in_slice = roi_trials[configname][roi].attrs['idx_in_slice']

                    #ax_curr.plot(tsecs, df, trial_color, alpha=0.2, linewidth=0.5)
                    #ax_curr.plot([tsecs[first_on], tsecs[first_on]+nframes_on/volumerate], [0, 0], stim_dur_color, linewidth=1, alpha=0.1)

                    #dfmat.append(df)
                    nframes = len(df)
                    if 'grating' in 'grating' in trial_info['stimtype']:
                        roi_dfs.append(pd.DataFrame({'trial': np.tile(trial, (nframes,)),
                                                 'config': np.tile(configname, (nframes,)),
                                                 'ori': np.tile(configs[configname]['rotation'], (nframes,)),
                                                 'sf': np.tile(configs[configname]['frequency'], (nframes,)),
                                                 'xpos': np.tile(configs[configname]['position'][0], (nframes,)),
                                                 'ypos': np.tile(configs[configname]['position'][1], (nframes,)),
                                                 'size': np.tile(configs[configname]['scale'][0], (nframes,)),
                                                 'tsec': tsecs,
                                                 'raw': trialmat[tidx,:],
                                                 'df': df,
                                                 'first_on': np.tile(first_on, (nframes,)),
                                                 'nframes_on': np.tile(nframes_on, (nframes,)),
                                                 'nsecs_on': np.tile(nframes_on/volumerate, (nframes,)),
                                                 'slice': np.tile(curr_slice, (nframes,)),
                                                 'roi_in_slice': np.tile(roi_in_slice, (nframes,)),
                                                 'pupil_size_baseline': np.tile(eye_info[trial]['pupil_size_baseline'], (nframes,)),
                                                 'pupil_size_stimulus': np.tile(eye_info[trial]['pupil_size_stim'], (nframes,)),
                                                 'pupil_dist_baseline': np.tile(eye_info[trial]['pupil_dist_baseline'], (nframes,)),
                                                 'pupil_dist_stimulus': np.tile(eye_info[trial]['pupil_dist_stim'], (nframes,)),
                                                 'nblinks_baseline': np.tile(eye_info[trial]['blink_event_count_baseline'], (nframes,)),
                                                 'nblinks_stim': np.tile(eye_info[trial]['blink_event_count_stim'], (nframes,))
                                                 }))
                    else:
                        roi_dfs.append(pd.DataFrame({'trial': np.tile(trial, (nframes,)),
                                                 'config': np.tile(configname, (nframes,)),
                                                 'img': os.path.splitext(os.path.split(configs[configname]['filepath'])[1])[0],
                                                 'xpos': np.tile(configs[configname]['position'][0], (nframes,)),
                                                 'ypos': np.tile(configs[configname]['position'][1], (nframes,)),
                                                 'size': np.tile(configs[configname]['scale'][0], (nframes,)),
                                                 'tsec': tsecs,
                                                 'raw': trialmat[tidx,:],
                                                 'df': df,
                                                 'first_on': np.tile(first_on, (nframes,)),
                                                 'nframes_on': np.tile(nframes_on, (nframes,)),
                                                 'nsecs_on': np.tile(nframes_on/volumerate, (nframes,)),
                                                 'slice': np.tile(curr_slice, (nframes,)),
                                                 'roi_in_slice': np.tile(roi_in_slice, (nframes,)),
                                                 'pupil_size_baseline': np.tile(eye_info[trial]['pupil_size_baseline'], (nframes,)),
                                                 'pupil_size_stimulus': np.tile(eye_info[trial]['pupil_size_stim'], (nframes,)),
                                                 'pupil_dist_baseline': np.tile(eye_info[trial]['pupil_dist_baseline'], (nframes,)),
                                                 'pupil_dist_stimulus': np.tile(eye_info[trial]['pupil_dist_stim'], (nframes,)),
                                                 'pupil_nblinks_baseline': np.tile(eye_info[trial]['blink_event_count_baseline'], (nframes,)),
                                                 'pupil_nblinks_stim': np.tile(eye_info[trial]['blink_event_count_stim'], (nframes,))
                                                 }))

#                                                 index=[tidx]
#                                                 ))

            ROI = pd.concat(roi_dfs, axis=0)

            DATA[roi] = ROI

    except Exception as e:

        print "--- Error plotting PSTH ---------------------------------"
        print roi, configname, trial
        traceback.print_exc()
    #    print "---------------------------------------------------------"

    return DATA

#%%



#%%
def plot_psths(DATA, plot_info, trial_info, configs, roi_psth_dir='/tmp', trace_type='raw', filter_pupil=True, pupil_params=None):

    if pupil_params is None:
        pupil_params = set_pupil_params()
    pupil_max_nblinks = pupil_params['max_nblinks']
    pupil_size_thr = pupil_params['size_thr']
    pupil_dist_thr = pupil_params['dist_thr']

    roi_psth_dir_all = os.path.join(roi_psth_dir, 'all')
    if not os.path.exists(roi_psth_dir_all):
        os.makedirs(roi_psth_dir_all)
    if filter_pupil is True:
        pupil_thresh_str = 'size%.2f-dist%.2f-blinks%i' % (pupil_size_thr, pupil_dist_thr, int(pupil_max_nblinks))
        roi_psth_dir_include = os.path.join(roi_psth_dir, 'include', pupil_thresh_str)
        if not os.path.exists(roi_psth_dir_include):
            os.makedirs(roi_psth_dir_include)
        roi_psth_dir_exclude = os.path.join(roi_psth_dir, 'exclude', pupil_thresh_str)
        if not os.path.exists(roi_psth_dir_exclude):
            os.makedirs(roi_psth_dir_exclude)

    volumerate = trial_info['volumerate'] #parsed_frames.attrs['volumerate']
    iti_pre = trial_info['iti_pre']
    nframes_on = trial_info['nframes_on'] #parsed_frames['trial00001']['frames_in_run'].attrs['stim_dur_sec'] * volumerate
    stim_dur = trial_info['stim_on_sec'] #trialdict['trial00001']['stim_dur_ms']/1E3
    iti_dur = trial_info['iti_full'] #trialdict['trial00001']['iti_dur_ms']/1E3
    tpoints = [int(i) for i in np.arange(-1*iti_pre, stim_dur+iti_dur)]

    roi_list = sorted(DATA.keys(), key=natural_keys)
    subplot_stimlist = plot_info['subplot_stimlist']
    roi =None; configname=None; trial=None
    try:
        for roi in roi_list:

            #%
            print roi
            DF = DATA[roi]
            if plot_info['nrows']==1:
                figwidth_multiplier = plot_info['ncols']*1
            else:
                figwidth_multiplier = 1

            for img in subplot_stimlist.keys():
                curr_subplots = subplot_stimlist[img]

                if plot_info['stimid_only'] is True:
                    figname = 'all_objects_default_pos_size'
                else:
                    figname = '%s_pos%i_size%i' % (os.path.splitext(img)[0], len(plot_info['stimuli']['position_vals']), len(plot_info['size_vals']))

                fig, axs = pl.subplots(
            	    nrows=plot_info['nrows'],
            	    ncols=plot_info['ncols'],
            	    sharex=True,
            	    sharey=True,
            	    figsize=(plot_info['figure_height']*figwidth_multiplier,plot_info['figure_height']),
            	    gridspec_kw=dict(top=1-plot_info['top_margin'], bottom=plot_info['bottom_margin'], wspace=0.05, hspace=0.05))

                if filter_pupil is True:
                    fig2, axs2 = pl.subplots(
                	    nrows=plot_info['nrows'],
                	    ncols=plot_info['ncols'],
                	    sharex=True,
                	    sharey=True,
                	    figsize=(plot_info['figure_height']*figwidth_multiplier,plot_info['figure_height']),
                	    gridspec_kw=dict(top=1-plot_info['top_margin'], bottom=plot_info['bottom_margin'], wspace=0.05, hspace=0.05))
                    fig3, axs3 = pl.subplots(
                	    nrows=plot_info['nrows'],
                	    ncols=plot_info['ncols'],
                	    sharex=True,
                	    sharey=True,
                	    figsize=(plot_info['figure_height']*figwidth_multiplier,plot_info['figure_height']),
                	    gridspec_kw=dict(top=1-plot_info['top_margin'], bottom=plot_info['bottom_margin'], wspace=0.05, hspace=0.05))


                row=0
                col=0
                plotidx = 0

                #roi = 'roi00003'
                for configname in curr_subplots:

                    stim_trials = sorted(list(set(DF[DF['config'] == configname]['trial'])), key=natural_keys)
                    curr_DF = DF[DF['trial'].isin(stim_trials)]
                    if filter_pupil is True:
                        filtered_DF = curr_DF[((curr_DF['pupil_size_stimulus'] > pupil_size_thr)
                                                & (curr_DF['pupil_size_baseline'] > pupil_size_thr)
                                                & (curr_DF['pupil_dist_baseline'] < pupil_dist_thr)
                                                & (curr_DF['pupil_dist_stimulus'] < pupil_dist_thr)
                                                & (curr_DF['pupil_nblinks_stim'] < pupil_max_nblinks)
                                                & (curr_DF['pupil_nblinks_baseline'] < pupil_max_nblinks)
                                                )]
                        pass_trials = sorted(list(set(filtered_DF['trial'])), key=natural_keys)
                        fail_trials = [t for t in stim_trials if t not in pass_trials]
                    else:
                        pass_trials = stim_trials.copy()
                        fail_trials = []

                    curr_slice = list(set(DF['slice']))[0] #roi_trials[configname][roi].attrs['slice']
                    roi_in_slice = list(set(DF['roi_in_slice']))[0] #roi_trials[configname][roi].attrs['idx_in_slice']

                    if col==(plot_info['ncols']) and plot_info['nrows']>1:
                        row += 1
                        col = 0
                    if len(axs.shape)>1:
                        ax_curr = axs[row, col] #, col]
                        if filter_pupil is True:
                            ax_curr2 = axs2[row, col]
                            ax_curr3 = axs3[row, col]
                    else:
                        ax_curr = axs[col]
                        if filter_pupil is True:
                            ax_curr2 = axs2[col]
                            ax_curr3 = axs3[col]

                    dfmat = []
                    if filter_pupil is True:
                        dfmat_fail = []
                        dfmat_pass = []

                    if 'grating' in trial_info['stimtype']:
                        stimname = 'Ori %.0f, SF: %.2f' % (configs[configname]['rotation'], configs[configname]['frequency'])
                    else:
                        stimname = '%s- pos (%.1f, %.1f) - siz %.1f' % (os.path.splitext(configs[configname]['filename'])[0], configs[configname]['position'][0], configs[configname]['position'][1], configs[configname]['scale'][0])

                    ax_curr.annotate(stimname,xy=(0.1,1), xycoords='axes fraction', horizontalalignment='middle', verticalalignment='top', weight='bold')
                    if filter_pupil is True:
                        ax_curr2.annotate(stimname,xy=(0.1,1), xycoords='axes fraction', horizontalalignment='middle', verticalalignment='top', weight='bold')
                        ax_curr3.annotate(stimname,xy=(0.1,1), xycoords='axes fraction', horizontalalignment='middle', verticalalignment='top', weight='bold')


                    for tidx, trial in enumerate(sorted(stim_trials, key=natural_keys)):

                        first_on = list(set(DF[DF['trial'] == trial]['first_on']))[0]
                        tsecs = DF[DF['trial'] == trial]['tsec']

                        df = DF[DF['trial'] == trial]['df'] #roi_trials[configname][roi][trial][trace_type]

                        #if plot_all is True:
                        trial_color = 'k'
                        stim_dur_color = 'r'
                        dfmat.append(df)

                        ax_curr.plot(tsecs, df, trial_color, alpha=0.2, linewidth=0.5)
                        ax_curr.plot([tsecs[first_on], tsecs[first_on]+nframes_on/volumerate], [0, 0], stim_dur_color, linewidth=1, alpha=0.1)

                        if filter_pupil is True:
                            if trial in pass_trials:
                                dfmat_pass.append(df)
                                ax_curr2.plot(tsecs, df, 'b', alpha=0.2, linewidth=0.5)
                                ax_curr2.plot([tsecs[first_on], tsecs[first_on]+nframes_on/volumerate], [0, 0], stim_dur_color, linewidth=1, alpha=0.1)
                            else:
                                dfmat_fail.append(df)
                                ax_curr3.plot(tsecs, df, 'r', alpha=0.2, linewidth=0.5)
                                ax_curr3.plot([tsecs[first_on], tsecs[first_on]+nframes_on/volumerate], [0, 0], stim_dur_color, linewidth=1, alpha=0.1)

                    # Plot MEAN DF:
                    ax_curr.plot(tsecs, np.nanmean(dfmat, axis=0), 'k', alpha=1, linewidth=1)
                    if plot_info['universal_scale'] is True:
                        ax_curr.set_ylim([plot_info['ylim_min'], plot_info['ylim_max']])
                    ax_curr.set(xticks=tpoints)
                    ax_curr.tick_params(axis='x', which='both',length=0)

                    if filter_pupil is True:
                        # Plot PASS trials:
                        if len(dfmat_pass) > 0:
                            ax_curr2.plot(tsecs, np.nanmean(dfmat_pass, axis=0), 'b', alpha=1, linewidth=1)
                        if plot_info['universal_scale'] is True:
                            ax_curr2.set_ylim([plot_info['ylim_min'], plot_info['ylim_max']])
                        ax_curr2.set(xticks=tpoints)
                        ax_curr2.tick_params(axis='x', which='both',length=0)
                        # Plot FAIL trials:
                        if len(dfmat_fail) > 0:
                            ax_curr3.plot(tsecs, np.nanmean(dfmat_fail, axis=0), 'r', alpha=1, linewidth=1)
                        if plot_info['universal_scale'] is True:
                            ax_curr3.set_ylim([plot_info['ylim_min'], plot_info['ylim_max']])
                        ax_curr3.set(xticks=tpoints)
                        ax_curr3.tick_params(axis='x', which='both',length=0)

                    col = col + 1
                    plotidx += 1

            sns.despine(offset=2, trim=True)
            #%
            fig.suptitle(roi)
            psth_fig_fn = '%s_%s_%s_%s_%s_ALL.png' % (roi, curr_slice, roi_in_slice, trace_type, figname)
            fig.savefig(os.path.join(roi_psth_dir_all, psth_fig_fn))
            pl.close(fig)

            if filter_pupil is True:
                fig2.suptitle(roi)
                psth_fig_fn = '%s_%s_%s_%s_%s_PUPIL_%s_pass.png' % (roi, curr_slice, roi_in_slice, trace_type, figname, pupil_thresh_str)
                fig2.savefig(os.path.join(roi_psth_dir_include, psth_fig_fn))
                pl.close(fig2)
                fig3.suptitle(roi)
                psth_fig_fn = '%s_%s_%s_%s_%s_PUPIL_%s_fail.png' % (roi, curr_slice, roi_in_slice, trace_type, figname, pupil_thresh_str)
                fig3.savefig(os.path.join(roi_psth_dir_exclude, psth_fig_fn))
                pl.close(fig3)


    except Exception as e:

        print "--- Error plotting PSTH ---------------------------------"
        print roi, configname, trial
        traceback.print_exc()
    #    print "---------------------------------------------------------"
#        finally:
#            roi_trials.close()
    #parsed_frames.close()

    print "PSTHs saved to: %s" % roi_psth_dir


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

def set_pupil_params(size_thr=30, dist_thr=8, max_nblinks=2):
    pupil_params = dict()
    pupil_params['size_thr'] = size_thr
    pupil_params['dist_thr'] = dist_thr
    pupil_params['max_nblinks']= max_nblinks

    # Generate hash ID for current pupil params set
    pupil_params_hash = hash(json.dumps(pupil_params, sort_keys=True, ensure_ascii=True)) % ((sys.maxsize + 1) * 2) #[0:6]
    pupil_params['hash'] = pupil_params_hash

    return pupil_params

#%%
def calculate_metrics(DATA, filter_pupil=False, pupil_params=None):
    METRICS = {}
    PASSTRIALS = {}

    if filter_pupil is True and pupil_params is None:
        pupil_params = set_pupil_params()
    pupil_size_thr = pupil_params['size_thr']
    pupil_dist_thr = pupil_params['dist_thr']
    pupil_max_nblinks = pupil_params['max_nblinks']

    roi_list = sorted(DATA.keys(), key=natural_keys)
    config_list = sorted(list(set(DATA[roi_list[0]]['config'])), key=natural_keys)

    #for config in config_list:
    for roi in roi_list:
        DF = DATA[roi]
        metrics_df = []
        for config in config_list:
            trial_list = sorted(list(set(DF[DF['config']==config]['trial'])), key=natural_keys)

            if filter_pupil is True:
                curr_DF = DF[DF['trial'].isin(trial_list)]
                if filter_pupil is True:
                    filtered_DF = curr_DF[((curr_DF['pupil_size_stimulus'] > pupil_size_thr)
                                            & (curr_DF['pupil_size_baseline'] > pupil_size_thr)
                                            & (curr_DF['pupil_dist_baseline'] < pupil_dist_thr)
                                            & (curr_DF['pupil_dist_stimulus'] < pupil_dist_thr)
                                            & (curr_DF['pupil_nblinks_stim'] < pupil_max_nblinks)
                                            & (curr_DF['pupil_nblinks_baseline'] < pupil_max_nblinks)
                                            )]

                    pass_trials = sorted(list(set(filtered_DF['trial'])), key=natural_keys)
                    fail_trials = [t for t in trial_list if t not in pass_trials]
                else:
                    pass_trials = trial_list.copy()
                    fail_trials = []

            for trial in trial_list:
                df = DF[DF['trial'] == trial]['df']
                nframes = len(df)

                if filter_pupil is True and trial in fail_trials:
                    df = np.ones(df.shape) * np.nan
                    zscore_val = np.nan
                    mean_stim_on = np.nan
                    passes = False

                else:
                    first_on = list(set(DATA[roi][DATA[roi]['trial'] == trial]['first_on']))[0]
                    nframes_on = int(round(list(set(DATA[roi][DATA[roi]['trial'] == trial]['nframes_on']))[0]))
                    baseline = df[0:first_on]
                    stimulus = df[first_on:first_on+nframes_on]
                    mean_stim_on = np.nanmean(stimulus)
                    std_baseline = np.nanstd(baseline)
                    zscore_val = mean_stim_on / std_baseline
                    passes = True

                metrics_df.append(pd.DataFrame({'trial': np.tile(trial, (nframes,)),
                                                'config': np.tile(config, (nframes,)),
                                                'zscores': np.tile(zscore_val, (nframes,)),
                                                'mean_stim_on': np.tile(mean_stim_on, (nframes,)),
                                                'df': df,
                                                'pass': passes
                                                }))
            PASSTRIALS[config] = pass_trials

        ROI = pd.concat(metrics_df, axis=0)

        METRICS[roi] = ROI

    return METRICS, PASSTRIALS

#%%

def get_roi_metrics(DATA, configs, traceid_dir, filter_pupil=False, pupil_params=None):

    # Use default pupil params if none provided:
    if filter_pupil is True and pupil_params is None:
        pupil_params = set_pupil_params()

    # Check to see whether current pupil param set already exists:
    metrics_basedir = os.path.join(traceid_dir, 'metrics')
    if not os.path.exists(metrics_basedir):
        os.makedirs(metrics_basedir)
    pupil_metrics_dir = os.path.join(metrics_basedir, 'pupil_%d' % pupil_params['hash'])
    if not os.path.exists(pupil_metrics_dir):
        os.makedirs(pupil_metrics_dir)
    pupil_params_filepath = os.path.join(pupil_metrics_dir, 'pupil_params.json')
    if not os.path.exists(pupil_params_filepath):
        with open(pupil_params_filepath, 'w') as f:
            json.dump(pupil_params, f)

    existing_metrics_files = [f for f in os.listdir(pupil_metrics_dir) if 'roi_metrics_%s' % pupil_params['hash'] in f and f.endswith('hdf5')]
    if not len(existing_metrics_files) == 1:
        datestr = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        metrics_filepath = os.path.join(pupil_metrics_dir, 'roi_metrics_%s_%s.hdf5' % (pupil_params['hash'], datestr))
        METRICS, PASSTRIALS =  calculate_metrics(DATA, filter_pupil=filter_pupil, pupil_params=pupil_params)
        datastore = pd.HDFStore(metrics_filepath)
        for roi in METRICS.keys():
            datastore[str(roi)] = METRICS[roi]
        with open(os.path.join(pupil_metrics_dir, 'pass_trials.json'), 'w') as f:
            json.dump(PASSTRIALS, f, indent=4, sort_keys=True)
    else:
        metrics_filepath = os.path.join(pupil_metrics_dir, existing_metrics_files[0])
        METRICS = pd.HDFStore(metrics_filepath)
        with open(os.path.join(pupil_metrics_dir, 'pass_trials.json'), 'r') as f:
            PASSTRIALS = json.load(f)

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
def get_roi_summary_stats(metrics_filepath, configs):

    # Load METRICS:
    METRICS = pd.HDFStore(metrics_filepath)

    # First, check if corresponding ROISTATS file exists:
    curr_metrics_dir = os.path.split(metrics_filepath)[0]
    existing_files = [f for f in os.listdir(curr_metrics_dir) if 'roi_stats_' in f]
    if not len(existing_files) == 1:
        datestr = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        metrics_hash = os.path.splitext(os.path.split(metrics_filepath)[-1])[0].split('roi_metrics_')[-1].split('_')[0]
        roistats_filepath = os.path.join(curr_metrics_dir, 'roi_stats_%s_%s.hdf5' % (metrics_hash, datestr))
        ROISTATS = collate_roi_stats(METRICS, configs)
        datastore = pd.HDFStore(roistats_filepath)
        df = {'df': ROISTATS}
        for d in df:
            datastore[d] = df[d]
    else:
        roistats_filepath = os.path.join(curr_metrics_dir, existing_files[0])
        ROISTATS = pd.HDFStore(roistats_filepath)
        #ROISTATS = datastore['ROISTATS']

    return roistats_filepath

#%%
def collate_roi_stats(METRICS, configs):

    if 'frequency' in configs[configs.keys()[0]].keys():
        stimtype = 'grating'
    else:
        stimtype = 'image'

    # Sort metrics by stimulus-params and calculate sumamry stats:
    roistats_df =[]
    for roi in sorted(METRICS.keys(), key=natural_keys):
        print roi
        for config in sorted(configs.keys(), key = natural_keys):
            #print config
            DF = METRICS[roi][METRICS[roi]['config'] == config]

            all_trials = sorted(list(set(DF['trial'])), key=natural_keys)
            #pass_trials = sorted(list(set(DF[DF['pass'] == True]['trial'])), key=natural_keys)

            zscore_trial = [DF[DF['trial'] == trial]['zscores'][0] for trial in sorted(all_trials, key=natural_keys)]
            stimdf_trial = [DF[DF['trial'] == trial]['mean_stim_on'][0] for trial in sorted(all_trials, key=natural_keys)]

            xpos_trial = np.tile(configs[config]['position'][0], (len(zscore_trial),))
            ypos_trial = np.tile(configs[config]['position'][1], (len(zscore_trial),))
            size_trial = np.tile(configs[config]['scale'][0], (len(zscore_trial),))

            if stimtype == 'grating':
                sf_trial = np.tile(configs[config]['frequency'], (len(zscore_trial),))
                ori_trial = np.tile(configs[config]['rotation'], (len(zscore_trial),))

                # Create DF entry:
                roistats_df.append(pd.DataFrame({
                                'roi': np.tile(str(roi), (len(zscore_trial),)),
                                'config': np.tile(config, (len(zscore_trial),)),
                                'trial': sorted(all_trials, key=natural_keys),
                                'zscore': zscore_trial,
                                'stimdf': stimdf_trial,
                                'xpos': xpos_trial,
                                'ypos': ypos_trial,
                                'size': size_trial,
                                'sf': sf_trial,
                                'ori': ori_trial
                                }))

            else:
                imname = os.path.splitext(configs[config]['filename'])[0]
                if 'CamRot' in imname:
                    objectid = imname.split('_CamRot_')[0]
                    yrot = int(imname.split('_CamRot_y')[-1])
                    if 'N1' in imname:
                        morph = 1
                    elif 'N2' in imname:
                        morph = 20
                elif 'morph' in imname:
                    if 'yrot' not in imname:
                        objectid = 'morph' #imname
                        yrot = 0
                        morph = int(imname.split('morph')[-1])
                    else:
                        objectid = 'morph' #imname.split('_y')[0]
                        yrot = int(imname.split('_y')[-1])
                        morph = int(imname.split('_y')[0].split('morph'))


                img_trial = np.tile(imname, (len(zscore_trial),))
                # Create DF entry:
                roistats_df.append(pd.DataFrame({
                                'roi': np.tile(str(roi), (len(zscore_trial),)),
                                'config': np.tile(config, (len(zscore_trial),)),
                                'trial': sorted(all_trials, key=natural_keys),
                                'zscore': zscore_trial,
                                'stimdf': stimdf_trial,
                                'xpos': xpos_trial,
                                'ypos': ypos_trial,
                                'size': size_trial,
                                'img': img_trial,
                                'object': np.tile(objectid, (len(zscore_trial),)),
                                'yrot': np.tile(yrot, (len(zscore_trial),)),
                                'morph': np.tile(morph, (len(zscore_trial),))
                                }))

    ROISTATS = pd.concat(roistats_df, axis=0)

    return ROISTATS

#%%

def plot_transform_tuning(roi, curr_df, trans_type, object_type='object', metric_type='zscore', output_dir='/tmp', include_trials=True):
    if object_type == 'object':
        curr_objects = list(set(curr_df[object_type]))
        hue = object_type
        object_sorter = object_type
    else:
        curr_objects = ['morph']
        hue = None
        object_sorter = 'morph'

    fig, ax = pl.subplots()
    # Plot mean value across all trials of stim-config:
    sns.pointplot(x=trans_type, y=metric_type, hue=hue, data=curr_df.sort_values([object_sorter]),
                      ci=None, legend=False, join=False, markers='_', scale=2, ax=ax)

    # Add dots for individual trial z-score values:
    sns.stripplot(x=trans_type, y=metric_type, hue=hue, data=curr_df.sort_values([object_sorter]),
                      edgecolor="w", split=True, size=2, ax=ax)

    # Add SEM cuz it looks good:
    transforms = sorted(list(set(curr_df[trans_type])))
    for oi, obj in enumerate(sorted(curr_objects, key=natural_keys)):
        if hue=='object' and 'morph' in obj:
            continue
        if object_sorter == 'object':
            curr_zmeans = [np.nanmean(curr_df[((curr_df[object_sorter] == obj) & (curr_df[trans_type] == trans))][metric_type]) for trans in sorted(transforms)]
            curr_zmeans_yerr = [stats.sem(curr_df[((curr_df[object_sorter] == obj) & (curr_df[trans_type] == trans))][metric_type], nan_policy='omit') for trans in sorted(transforms)]
        else:
            curr_zmeans = [np.nanmean(curr_df[curr_df[trans_type] == trans][metric_type]) for trans in sorted(transforms)]
            curr_zmeans_yerr = [stats.sem(curr_df[curr_df[trans_type] == trans][metric_type], nan_policy='omit') for trans in sorted(transforms)]
        ax.errorbar(np.arange(0, len(curr_zmeans)), curr_zmeans, yerr=curr_zmeans_yerr, capsize=5, elinewidth=0)

    # Format, title, and save:
    pl.title(roi)
    sns.despine(offset=0, trim=True, bottom=True)
    if include_trials is True:
        figname = '%s_tuning_%s_trials_%s.png' % (trans_type, metric_type, roi)
    else:
        figname = '%s_tuning_%s_%s.png' % (trans_type, metric_type, roi)
    pl.savefig(os.path.join(output_dir, figname))
    pl.close()

#%%
def plot_tuning_curves(roistats_filepath, configs, output_dir, metric_type='zscore', include_trials=True):

    STATS = pd.HDFStore(roistats_filepath)['/df']


    if 'frequency' in configs[configs.keys()[0]].keys():
        stimtype = 'grating'
    else:
        stimtype = 'image'

    roi_list = sorted(list(set(STATS['roi'])), key=natural_keys)
#    config_list = sorted(list(set(STATS['config'])), key=natural_keys)
#
#    Htranslations = list(set([configs[c]['position'][0] for c in configs.keys()]))
#    Vtranslations = list(set([configs[c]['position'][1] for c in configs.keys()]))
#    sizes = list(set(([configs[c]['scale'][0] for c in configs.keys()])))
#
#    ncols = len(positions)
#    nrows = len(sizes)


    for roi in sorted(roi_list, key=natural_keys):

        if stimtype == 'image':
            objectid_list = sorted(list(set(STATS['object'])))

            # ----- First, plot TUNING curves for YROT (color by object ID) --------
            curr_objects = [i for i in objectid_list if 'morph' not in i]
            curr_df = STATS[((STATS['roi'] == roi) & (STATS['object'].isin(curr_objects)))]

            plot_transform_tuning(roi, curr_df, "yrot", "object", metric_type, output_dir=output_dir, include_trials=include_trials)

            # ----- Now, plot TUNING curves for MORPHS (color by morph level) --------
            curr_df = STATS[((STATS['roi'] == roi) & (STATS['yrot'] == 0))]
            plot_transform_tuning(roi, curr_df, "morph", object_type='morph', metric_type='zscore', output_dir=output_dir, include_trials=include_trials)



#%%

def extract_options(options):

    choices_tracetype = ('raw', 'denoised_nmf')
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

    parser.add_option('--eye', action="store_true",
                      dest="eyetracker", default=False, help="trial inclusion with eyetrackr info")

    parser.add_option('--custom', action="store_true",
                      dest="custom_mw", default=False, help="Not using MW (custom params must be specified)")

    # Only need to set these if using custom-paradigm file:
    parser.add_option('--order', action="store_true",
                      dest="same_order", default=False, help="Set if same stimulus order across all files (1 stimorder.txt)")
    parser.add_option('-O', '--stimon', action="store",
                      dest="stim_on_sec", default=0, help="Time (s) stimulus ON.")
    parser.add_option('-v', '--vol', action="store",
                      dest="vols_per_trial", default=0, help="Num volumes per trial. Specifiy if custom_mw=True")
    parser.add_option('-V', '--first', action="store",
                      dest="first_stim_volume_num", default=0, help="First volume stimulus occurs (py-indexed). Specifiy if custom_mw=True")

    parser.add_option('-y', '--ylim_min', action="store",
                      dest="ylim_min", default=-1.0, help="min lim for Y axis, df/f plots [default: -1.0]")
    parser.add_option('-Y', '--ylim_max', action="store",
                      dest="ylim_max", default=1.0, help="max lim for Y axis, df/f plots [default: 1.0]")

    parser.add_option('-T', '--trace-type', type='choice', choices=choices_tracetype, action='store', dest='trace_type', default=default_tracetype, help="Type of timecourse to plot PSTHs. Valid choices: %s [default: %s]" % (choices_tracetype, default_tracetype))

    parser.add_option('--new', action="store_true",
                      dest="create_new", default=False, help="Set flag to create new output files (/paradigm/parsed_frames.hdf5, roi_trials.hdf5")
    parser.add_option('--scale', action="store_true",
                      dest="universal_scale", default=False, help="Set flag to plot all PSTH plots with same y-axis scale")

#    parser.add_option('--omit-err', action="store_false",
#                      dest="use_errorbar", default=True, help="Set flag to plot PSTHs without error bars (default: std)")

    parser.add_option('--omit-trials', action="store_false",
                      dest="include_trials", default=True, help="Set flag to plot PSTHS without individual trial values")

    parser.add_option('--pupil', action="store_true",
                      dest="filter_pupil", default=False, help="Set flag to filter PSTH traces by pupil threshold params")
    parser.add_option('--size', action="store",
                      dest="pupil_size_thr", default=30, help="Cut-off for pupil radius, if --pupil set [default: 30]")
    parser.add_option('--dist', action="store",
                      dest="pupil_dist_thr", default=5, help="Cut-off for pupil distance from start, if --pupil set [default: 5]")
    parser.add_option('--blinks', action="store",
                      dest="pupil_max_nblinks", default=1, help="Cut-off for N blinks allowed in trial, if --pupil set [default: 1 (i.e., 0 blinks allowed)]")

    parser.add_option('--metric', action="store",
                      dest="roi_metric", default="zscore", help="ROI metric to use for tuning curves [default: 'zscore']")


    (options, args) = parser.parse_args(options)

    return options


#%%
# Set USER INPUT options:

def plot_traceid_psths(options):
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
    eyetracker = options.eyetracker

    custom_mw = options.custom_mw
    same_order = options.same_order #False #True

    ylim_min = float(options.ylim_min) #-1.0
    ylim_max = float(options.ylim_max) #3.0


    trace_type = options.trace_type
    print "Plotting PSTH for %s timecourses." % trace_type

    create_new = options.create_new
    universal_scale = options.universal_scale

    #use_errorbar = options.use_errorbar
    include_trials = options.include_trials
    filter_pupil = options.filter_pupil
    pupil_size_thr = float(options.pupil_size_thr)
    pupil_dist_thr = float(options.pupil_dist_thr)
    pupil_max_nblinks = float(options.pupil_max_nblinks)

    roi_metric = options.roi_metric
    #%
    # =============================================================================
    # Get meta/SI info for RUN:
    # =============================================================================
    run_dir = os.path.join(rootdir, animalid, session, acquisition, run)
    run_dir = os.path.join(rootdir, animalid, session, acquisition, run)
    si_info = get_frame_info(run_dir)

    # Get paradigm/AUX info:
    # =============================================================================
    paradigm_dir = os.path.join(run_dir, 'paradigm')
    trial_info = get_alignment_specs(paradigm_dir, si_info, custom_mw=custom_mw, options=options)

    # Load TRACE ID info:
    # =========================================================================
    TID = load_TID(run_dir, trace_id)
    traceid_dir = TID['DST']
    if rootdir not in traceid_dir:
        orig_root = traceid_dir.split('/%s/%s' % (animalid, session))[0]
        traceid_dir = traceid_dir.replace(orig_root, rootdir)
        print "Replacing orig root with dir:", traceid_dir
        trace_hash = TID['trace_hash']

    # Assign frame indices for specified trial epochs:
    # =========================================================================
    parsed_frames_filepath = assign_frames_to_trials(si_info, trial_info, paradigm_dir, create_new=create_new)

    # Get all unique stimulus configurations:
    # =========================================================================
    configs, stimtype = get_stimulus_configs(trial_info)

    #%
    # Group ROI time-courses for each trial by stimulus config:
    # =========================================================================
    excluded_tiffs = TID['PARAMS']['excluded_tiffs']
    print "Current trace ID - %s - excludes %i tiffs." % (trace_id, len(excluded_tiffs))
    roi_trials_by_stim_path = group_rois_by_trial_type(traceid_dir, parsed_frames_filepath, trial_info, si_info, excluded_tiffs=excluded_tiffs, create_new=create_new)
    #%
    # GET eye-info:?
    eye_info = load_eye_data(run_dir)
    if eye_info is None:
        filter_pupil = False

    # GET ALL DATA into a dataframe:
    # =============================================================================
    corresponding_datestr = os.path.splitext(os.path.split(roi_trials_by_stim_path)[-1])[0]
    corresponding_datestr = corresponding_datestr.split('roi_trials_')[-1]
    roidata_filepath = os.path.join(traceid_dir, 'ROIDATA_%s.hdf5' % corresponding_datestr)
    if os.path.exists(roidata_filepath) and create_new is False:
        DATA = pd.HDFStore(roidata_filepath)
    else:
        # First remove old files:
        existing_df_files = [f for f in os.listdir(traceid_dir) if 'ROIDATA_' in f and f.endswith('hdf5')]
        old_dir = os.path.join(traceid_dir, 'old')
        if not os.path.exists(old_dir):
            os.makedirs(old_dir)
        for e in existing_df_files:
            shutil.move(os.path.join(traceid_dir, e), os.path.join(old_dir, e))
        print "Moved old files..."
        DATA = traces_to_trials(trial_info, configs, roi_trials_by_stim_path, trace_type='raw', eye_info=eye_info)
        # Save dataframe with same datestr as roi_trials.hdf5 file in traceid dir:
        datastore = pd.HDFStore(roidata_filepath)
        for roi in DATA.keys():
            datastore[str(roi)] = DATA[roi]


    # FILTER DATA with PUPIL PARAM thresholds, if relevant:
    # =============================================================================
    pupil_params = set_pupil_params(size_thr=pupil_size_thr, dist_thr=pupil_dist_thr, max_nblinks=pupil_max_nblinks)

    # =============================================================================
    # Set plotting params for trial average plots for each ROI:
    # =============================================================================
    plot_info = set_subplot_order(configs, stimtype, universal_scale=universal_scale)

    #% For each ROI, plot PSTH for all stim configs:
    roi_psth_dir = os.path.join(traceid_dir, 'figures', 'psths', trace_type)
    if not os.path.exists(roi_psth_dir):
        os.makedirs(roi_psth_dir)
    print "Saving PSTH plots to: %s" % roi_psth_dir

    plot_psths(DATA, plot_info, trial_info, configs, roi_psth_dir=roi_psth_dir, trace_type='raw',
                   filter_pupil=filter_pupil, pupil_params=pupil_params)



    # Cacluate some metrics, and plot tuning curves:
    metrics_filepath = get_roi_metrics(DATA, configs, traceid_dir, filter_pupil=filter_pupil, pupil_params=pupil_params)
    roistats_filepath = get_roi_summary_stats(metrics_filepath, configs)


    # PLOT TUNING CURVES
    tuning_figdir_base = os.path.join(traceid_dir, 'figures', 'tuning', trace_type)

    # First, plot with ALL trials included:
    tuning_figdir = os.path.join(tuning_figdir_base, roi_metric)
    if filter_pupil is True:
        curr_tuning_figdir = os.path.join(tuning_figdir, 'size%.2f-dist%.2f-blinks%i' % (pupil_size_thr, pupil_dist_thr, pupil_max_nblinks))
    else:
        curr_tuning_figdir = os.path.join(tuning_figdir, 'all')
    if not os.path.exists(curr_tuning_figdir):
        os.makedirs(curr_tuning_figdir)

    plot_tuning_curves(roistats_filepath, configs, curr_tuning_figdir, metric_type=roi_metric, include_trials=include_trials)


    return roidata_filepath, roi_psth_dir

#%%

def main(options):

    roidata_filepath, roi_psth_dir = plot_traceid_psths(options)

    print "DONE PLOTTING PSTHS!"
    print "All ROI plots saved to: %s" % roi_psth_dir
    print "DATA frame saved to: %s" % roidata_filepath


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



