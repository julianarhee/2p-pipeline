#!/usr/bin/env python2
'''
This script assigns frame indices to trial epochs for each trial in the specified run.
The baseline period (period before stimulus onset) is specified by input opt ITI_PRE ('-b' or '--baseline').

It requries a .json file that contains trial details across the entire run:
    - Created by extract_stimulus_events.py, saved to:  <run_dir>/paradigm/trials_<trialdict_hash>.json
    - The specific .tif file in which a given trial occurs in the run is stored in trialdict[trialname]['aux_file_idx']

It outputs a .hdf5 file that contains a dataset for each trial in the run (ntrials-per-file * nfiles-in-run)
with the frame indices of the associate .tif file
    - Output file is saved to:  <run_dir>/paradigm/frames_by_file_<framedict_hash>.hdf5
    - Frame indices are with respect to the entire file, so volume-parsing should be done (see: files_to_trials.py) if nslices > 1.

This output info for the run can then be used as indices into extracted traces.
'''

import os
import json
import re
import optparse
import operator
import h5py
import pprint
import itertools
import seaborn as sns
import pylab as pl
import numpy as np
from pipeline.python.utils import hash_file
pp = pprint.PrettyPrinter(indent=4)

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]

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


# source = '/nas/volume1/2photon/projects'
# experiment = 'scenes'
# session = '20171003_JW016'
# acquisition = 'FOV1'
# functional_dir = 'functional'
# mw = False

# stim_on_sec = float(options.stim_on_sec) #2. # 0.5
## iti_pre = float(options.iti_pre)
# same_order = False #True

#%%

parser = optparse.OptionParser()

parser.add_option('-R', '--root', action='store', dest='rootdir', default='/nas/volume1/2photon/data', help='data root dir (root project dir containing all animalids) [default: /nas/volume1/2photon/data, /n/coxfs01/2pdata if --slurm]')
parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')

# Set specific session/run for current animal:
parser.add_option('-S', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID')
parser.add_option('-A', '--acq', action='store', dest='acquisition', default='FOV1', help="acquisition folder (ex: 'FOV1_zoom3x') [default: FOV1]")
parser.add_option('-r', '--run', action='store', dest='run', default='', help="name of run dir containing tiffs to be processed (ex: gratings_phasemod_run1)")
parser.add_option('--slurm', action='store_true', dest='slurm', default=False, help="set if running as SLURM job on Odyssey")

parser.add_option('-b', '--baseline', action="store",
                  dest="iti_pre", default=1.0, help="Time (s) pre-stimulus to use for baseline. [default: 1.0]")


parser.add_option('--custom', action="store_true",
                  dest="custom_mw", default=False, help="Not using MW (custom params must be specified)")

# Only need to set these if using custom-paradigm file:
parser.add_option('--order', action="store_true",
                  dest="same_order", default=False, help="Set if same stimulus order across all files (1 stimorder.txt)")
parser.add_option('-O', '--stimon', action="store",
                  dest="stim_on_sec", default=0, help="Time (s) stimulus ON.")
parser.add_option('-t', '--vol', action="store",
                  dest="vols_per_trial", default=0, help="Num volumes per trial. Specifiy if custom_mw=True")
parser.add_option('-v', '--first', action="store",
                  dest="first_stim_volume_num", default=0, help="First volume stimulus occurs (py-indexed). Specifiy if custom_mw=True")


(options, args) = parser.parse_args()


# Set USER INPUT options:
rootdir = options.rootdir
animalid = options.animalid
session = options.session
acquisition = options.acquisition
run = options.run

trace_id = options.trace_id

iti_pre = float(options.iti_pre)

custom_mw = options.custom_mw
same_order = options.same_order #False #True


#%%

run_dir = os.path.join(rootdir, animalid, session, acquisition, run)
paradigm_dir = os.path.join(run_dir, 'paradigm')

outfile = os.path.join(paradigm_dir, 'frames_by_file_.hdf5')

# -------------------------------------------------------------------------
# Load reference info:
# -------------------------------------------------------------------------
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

#%%
# =========================================================================
# This section takes care of custom (non-MW) stim presentation protocols:
# =========================================================================
if custom_mw is True:
    try:
        stim_on_sec = float(options.stim_on_sec) #2. # 0.5
        first_stimulus_volume_num = int(options.first_stim_volume_num) #50
        vols_per_trial = float(options.vols_per_trial) # 15
        iti_full = (vols_per_trial/volumerate) - stim_on_sec
        iti_post = iti_full - iti_pre
        print "==============================================================="
        print "Using CUSTOM trial-info (not MW)."
        print "==============================================================="
        print "First stim on:", first_stimulus_volume_num
        print "Volumes per trial:", vols_per_trial
        print "ITI POST (s):", iti_post
        print "ITT full (s):", iti_full
        print "TRIAL dur (s):", stim_on_sec + iti_full
        print "Vols per trial (calc):", (stim_on_sec + iti_pre + iti_post) * volumerate
        print "==============================================================="

        ### Get stim-order files:
        stimorder_fns = sorted([f for f in os.listdir(paradigm_dir) if 'stimorder' in f and f.endswith('txt')], key=natural_keys)
        print "Found %i stim-order files, and %i TIFFs." % (len(stimorder_fns), nfiles)
        if len(stimorder_fns) < ntiffs:
            if same_order:
                # Same stimulus order for each file (complete set)
                stimorder_fns = np.tile(stimorder_fns, [ntiffs,])

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
    ### Get PARADIGM INFO if using standard MW:
    try:
        trial_fn = [t for t in os.listdir(paradigm_dir) if 'trials_' in t and t.endswith('json')][0]

        with open(os.path.join(paradigm_dir, trial_fn), 'r') as f:
            trialdict = json.load(f)

        # Get presentation info (should be constant across trials and files):
        trial_list = sorted(trialdict.keys(), key=natural_keys)
        stim_durs = [round(trialdict[t]['stim_dur_ms']/1E3, 1)for t in trial_list]
        assert len(list(set(stim_durs))) == 1, "More than 1 stim_dur found..."
        stim_on_sec = stim_durs[0]
        iti_durs = [round(trialdict[t]['iti_dur_ms']/1E3, 1) for t in trial_list]
        assert len(list(set(iti_durs))) == 1, "More than 1 iti_dur found..."
        iti_full = iti_durs[0]
        iti_post = iti_full - iti_pre

    except Exception as e:
        print "Could not find unique trial-file for current run %s..." % run
        print "Aborting with error:"
        print "---------------------------------------------------------------"
        print e
        print "---------------------------------------------------------------"


try:
    nframes_on = stim_on_sec * framerate #int(round(stim_on_sec * volumerate))
    nframes_iti_pre = iti_pre * framerate
    nframes_iti_post = iti_post*framerate # int(round(iti_post * volumerate))
    nframes_iti_full = iti_full * framerate #int(round(iti_full * volumerate))
    nframes_post_onset = (stim_on_sec + iti_post) * framerate
except Exception as e:
    print "Problem calcuating nframes for trial epochs..."
    print e

#%%

# =================================================================================
# Get frame indices for specified trial epochs:
# =================================================================================

# 1. Create HDF5 file to store ALL trials in run with stimulus info and frame info:
run_grp = h5py.File(outfile, 'w')
run_grp.attrs['framerate'] = framerate
run_grp.attrs['volumerate'] = volumerate
run_grp.attrs['baseline_dur'] = iti_pre

nslices_full = int(round(runinfo['frame_rate']/runinfo['volume_rate']))
nframes_per_file = nslices_full * nvolumes

#%
# 1. Get stimulus preseentation order for each TIFF found:

trial_counter = 0
for tiffnum in range(ntiffs):
    trial_in_file = 0
    currfile= "File%03d" % int(tiffnum+1)
    #frames_tsecs = np.arange(0, nvolumes)*(1/volumerate)

    if custom_mw is True:
        stim_fn = stimorder_fns[tiffnum]
        with open(os.path.join(paradigm_dir, stimorder_fns[tiffnum])) as f:
            stimorder_data = f.readlines()
        stimorder = [l.strip() for l in stimorder]
    else:
        stimorder = [trialdict[t]['stimuli']['stimulus'] for t in trial_list\
                         if trialdict[t]['aux_file_idx'] == tiffnum]
        trials_in_run = sorted([t for t in trial_list if trialdict[t]['aux_file_idx'] == tiffnum], key=natural_keys)

    unique_stims = sorted(set(stimorder), key=natural_keys)

    for trialidx,trialstim in enumerate(sorted(stimorder, key=natural_keys)):
        trial_counter += 1
        trial_in_file += 1
        currtrial_in_file = 'trial%03d' % int(trial_in_file)

        if custom_mw is True:
            if trialidx==0:
                first_frame_on = first_stimulus_volume_num
            else:
                first_frame_on += vols_per_trial
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

        preframes = list(np.arange(int(first_frame_on - nframes_iti_pre), first_frame_on, 1))
        postframes = list(np.arange(int(first_frame_on + 1), int(round(first_frame_on + nframes_post_onset))))

        framenums = [preframes, [first_frame_on], postframes]
        framenums = reduce(operator.add, framenums)
        #print "POST FRAMES:", len(framenums)
        diffs = np.diff(framenums)
        consec = [i for i in np.diff(diffs) if not i==0]
        assert len(consec)==0, "Bad frame parsing in %s, %s, frames: %s " % (currtrial_in_run, trialstim, str(framenums))

        # Create dataset for current trial with frame indices:
        #print "nframes", len(framenums)
        #print "on idx", framenums.index(first_frame_on)

        fridxs_in_file = run_grp.create_dataset('/'.join((currtrial_in_run, 'frames_in_file')), np.array(framenums).shape, np.array(framenums).dtype)
        fridxs_in_file[...] = np.array(framenums)
        fridxs_in_file.attrs['trial'] = currtrial_in_file
        fridxs_in_file.attrs['aux_file_idx'] = tiffnum
        fridxs_in_file.attrs['stim_on_idx'] = first_frame_on


        framenums_in_run = np.array(framenums) + (nframes_per_file*tiffnum)
        fridxs = run_grp.create_dataset('/'.join((currtrial_in_run, 'frames_in_run')), np.array(framenums_in_run).shape, np.array(framenums_in_run).dtype)
        fridxs[...] = np.array(framenums_in_run)
        fridxs.attrs['trial'] = currtrial_in_run
        fridxs.attrs['aux_file_idx'] = tiffnum
        fridxs.attrs['stim_on_idx'] = first_frame_on + (nframes_per_file*tiffnum)
        fridxs.attrs['stim_dur_sec'] = stim_on_sec
        fridxs.attrs['iti_dur_sec'] = iti_full
        fridxs.attrs['baseline_dur_sec'] = iti_pre

run_grp.close()

# Rename FRAME file with hash:
frame_file_hashid = hash_file(outfile)
file_frames_outfilename = os.path.splitext(outfile)[0] + frame_file_hashid + os.path.splitext(outfile)[1]
os.rename(outfile, file_frames_outfilename)

print "Finished assigning frame idxs across all tiffs to trials in run %s." % run
print "Saved frame file to:", file_frames_outfilename


#%%

# =================================================================================
# Get VOLUME indices to assign frame numbers to volumes:
# =================================================================================
vol_idxs_file = np.empty((nvolumes*nslices_full,))
vcounter = 0
for v in range(nvolumes):
    vol_idxs_file[vcounter:vcounter+nslices_full] = np.ones((nslices_full, )) * v
    vcounter += nslices_full
vol_idxs_file = [int(v) for v in vol_idxs_file]


vol_idxs = []
vol_idxs.extend(np.array(vol_idxs_file) + nvolumes*tiffnum for tiffnum in range(nfiles))
vol_idxs = np.array(sorted(np.concatenate(vol_idxs).ravel()))


#%%
# =================================================================================
# Load TRACE ID:
# =================================================================================

run_dir = os.path.join(rootdir, animalid, session, acquisition, run)
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

#%%
# =================================================================================
# Load timecourses for specified trace set:
# =================================================================================

traceid_dir = TID['DST']
print "================================================="
print "Loading time courses from trace set: %s"%  trace_id
print "================================================="

try:
    tcourse_fn = [t for t in os.listdir(traceid_dir) if t.endswith('hdf5') and 'roi_' in t][0]
    tracestruct = h5py.File(os.path.join(traceid_dir, tcourse_fn), 'r')
    roi_list = sorted(tracestruct.keys(), key=natural_keys)
    print "Loaded time-courses for run %s. Found %i ROIs total." % (run, len(roi_list))
except Exception as e:
    print "-------------------------------------------------------------------"
    print "Unable to load time courses for current trace set: %s" % trace_id
    print "File not found in dir: %s" % traceid_dir
    print "Aborting with error:"
    print e
    print "-------------------------------------------------------------------"

try:
    curr_slices = sorted(list(set([tracestruct[roi].attrs['slice'] for roi in roi_list])), key=natural_keys)
    print "ROIs are distributed across %i slices:" % len(curr_slices)
    print(curr_slices)
except Exception as e:
    print "-------------------------------------------------------------------"
    print "Unable to load slices..."
    print e
    print "-------------------------------------------------------------------"


#%%

framestruct = h5py.File(file_frames_outfilename, 'r')

#%%

trial_list = sorted(trialdict.keys(), key=natural_keys)
#stimids = sorted(list(set([trialdict[t]['stimuli']['stimulus'] for t in trial_list])), key=natural_keys)
stimtype = trialdict[trial_list[0]]['stimuli']['type']
if 'grating' in stimtype:
    # Likely varying gabor types...
    stimparams = [k for k in trialdict[trial_list[0]]['stimuli'].keys() if not (k=='stimulus' or k=='type')]
else:
    stimparams = [k for k in trialdict[trial_list[0]]['stimuli'].keys() if not (k=='type' or k=='filehash' or k=='filepath')]

stimparams = sorted(stimparams, key=natural_keys)
# Get all unique stimulus configurations (ID, filehash, position, size, rotation):
allparams = []
for param in stimparams:
    if isinstance(trialdict[trial_list[0]]['stimuli'][param], list):
        currvals = [tuple(trialdict[trial]['stimuli'][param]) for trial in trial_list]
    else:
        currvals = [trialdict[trial]['stimuli'][param] for trial in trial_list]
    #allparams.append(list(set(currvals)))
    allparams.append([i for i in list(set(currvals))])


transform_combos = list(itertools.product(*allparams))
ncombinations = len(transform_combos)


configs = dict()
for config in range(ncombinations):
    configname = 'config%03d' % int(config)
    configs[configname] = dict()
    for pidx, param in enumerate(sorted(stimparams, key=natural_keys)):
        if isinstance(transform_combos[config][pidx], tuple):
            configs[configname][param] = [transform_combos[config][pidx][0], transform_combos[config][pidx][1]]
        else:
            configs[configname][param] = transform_combos[config][pidx]


if stimtype=='image':
    stimids = sorted(list(set([trialdict[t]['stimuli']['stimulus'] for t in trial_list])), key=natural_keys)
    filepaths = list(set([trialdict[trial]['stimuli']['filepath'] for trial in trial_list]))
    filehashes = list(set([trialdict[trial]['stimuli']['filehash'] for trial in trial_list]))

    assert len(filepaths) == len(stimids), "More than 1 file path per stim ID found!"
    assert len(filehashes) == len(stimids), "More than 1 file hash per stim ID found!"

    stimhash_combos = list(set([(trialdict[trial]['stimuli']['stimulus'], trialdict[trial]['stimuli']['filehash']) for trial in trial_list]))
    assert len(stimhash_combos) == len(stimids), "Bad stim ID - stim file hash combo..."
    stimhash = dict((stimid, hashval) for stimid, hashval in zip([v[0] for v in stimhash_combos], [v[1] for v in stimhash_combos]))


#%%
sliceids = dict((curr_slices[s], s) for s in range(len(curr_slices)))


roi_stimdict_path = os.path.join(traceid_dir, 'rois_by_stim.hdf5')
stiminfo = h5py.File(roi_stimdict_path, 'w')


#for roi in roi_list[0:5]:
#    print roi

for config_idx in range(ncombinations):
#        curr_pos = transform_combos[config_idx][0]
#        curr_size = transform_combos[config_idx][1]
#        curr_rot = transform_combos[config_idx][2]
#        curr_stim = [k for k in stimhash.keys() if stimhash[k] == transform_combos[config_idx][3]][0]
#        configname = 'stimconfig%i' % config_idx # int(curr_stim)
#
#        curr_trials = [trial for trial in trial_list
#                       if trialdict[trial]['stimuli']['stimulus'] == curr_stim
#                        and trialdict[trial]['stimuli']['position'] == [curr_pos[0], curr_pos[1]]
#                        and trialdict[trial]['stimuli']['rotation'] == curr_rot
#                        and trialdict[trial]['stimuli']['scale'] == [curr_size[0], curr_size[1]]]


    configname = 'config%03d' % int(config_idx)
    currconfig = configs[configname]
    curr_trials = [trial for trial in trial_list
                   if all(trialdict[trial]['stimuli'][param] == currconfig[param] for param in currconfig.keys())]

    if configname not in stiminfo.keys():
        config_grp = stiminfo.create_group(configname)
    else:
        config_grp = stiminfo[configname]

    tracemat = []
    for tidx, trial in enumerate(sorted(curr_trials, key=natural_keys)):
#            if trial=='trial00160':
#                continue
        curr_trial_volume_idxs = [vol_idxs[int(i)] for i in framestruct[trial]['frames_in_run']]
#            slicenum = sliceids[tracestruct['roi00003'].attrs['slice']]
#            slice_idxs = vol_idxs[3::nslices_full]
#            tcourse_indices = [s for s in slice_idxs if s in curr_trial_volume_idxs]
#            tcourse_indices = sorted(list(set(np.array(tcourse_indices))))
#            if len(tcourse_indices)==30:
#                print curr_stim, trial
        stim_on_frame_idx = framestruct[trial]['frames_in_run'].attrs['stim_on_idx']
        stim_on_volume_idx = vol_idxs[stim_on_frame_idx]
        trial_idxs = sorted(list(set(curr_trial_volume_idxs)))
        #print trial_idxs.index(stim_on_volume_idx)

        for roi in roi_list:
            #print roi
            tcourse = tracestruct[roi]['timecourse'][trial_idxs]

            #tracemat.append(tracestruct['roi00003']['timecourse'][tcourse_indices])
            #print trial, np.array(tracemat).dtype

            currtrial = 'trial%03d' % int(tidx + 1)

            tset = config_grp.create_dataset('/'.join((roi, currtrial)), tcourse.shape, tcourse.dtype)
            tset[...] = tcourse
            tset.attrs['frame_on'] = stim_on_volume_idx #framestruct[trial]['frames_in_run'].attrs['stim_on_idx']
            tset.attrs['frame_idxs'] = trial_idxs

            #tset.attrs['stim_dur_sec'] = framestruct[trial]['frames_in_run'].attrs['stim_dur_sec']

stiminfo.close()

#%%
stiminfo = h5py.File(roi_stimdict_path, 'r')

#%%

roi_output_figdir = os.path.join(os.path.split(roi_stimdict_path)[0], 'figures', 'psths')
if not os.path.exists(roi_output_figdir):
    os.makedirs(roi_output_figdir)

for roi in roi_list:
    print roi
    fig = pl.figure(figsize=(15,10))


    nframes_on = framestruct['trial00001']['frames_in_run'].attrs['stim_dur_sec'] * volumerate
    pidx = 1
    #roi = 'roi00003'
    for config in stiminfo.keys():

        stim_trials = sorted([t for t in stiminfo[config][roi].keys()], key=natural_keys)
        nvols = max([stiminfo[config][roi][t].shape[0] for t in stim_trials])
        ntrials = len(stim_trials)
        trialmat = np.ones((ntrials, nvols)) * np.nan
        dfmat = []

        first_on = int(min([[i for i in stiminfo[config][roi][t].attrs['frame_idxs']].index(stiminfo[config][roi][t].attrs['frame_on']) for t in stim_trials]))

        ax = fig.add_subplot(4,5,pidx)
        if 'grating' in stimtype:
            stimname = 'Ori %.2d, SF: %.2d' % (configs[config]['rotation'], configs[config]['frequency'])
        else:
            stimname = configs[config]['stimulus']

        ax.annotate(stimname,xy=(0.1,1), xycoords='axes fraction', horizontalalignment='middle', verticalalignment='top', weight='bold')

        for tidx, trial in enumerate(sorted(stim_trials, key=natural_keys)):
            trace = stiminfo[config][roi][trial]
            curr_on = int([i for i in trace.attrs['frame_idxs']].index(int(trace.attrs['frame_on'])))
            trialmat[tidx, first_on:first_on+len(trace[curr_on:])] = trace[curr_on:]
            if first_on < curr_on:
                trialmat[tidx, 0:first_on] = trace[1:curr_on]
            else:
                trialmat[tidx, 0:first_on] = trace[0:curr_on]

            #trialmat[tidx, first_on-curr_on:first_on] = trace[0:curr_on]

            baseline = np.nanmean(trialmat[tidx, 0:first_on]) #[0:on_idx])
            df = (trialmat[tidx,:] - baseline) / baseline

            ax.plot(df, 'k', alpha=0.2, linewidth=0.5)
            ax.plot([first_on, first_on+nframes_on], [0, 0], 'r', linewidth=1, alpha=0.1)

            dfmat.append(df)

        ax.plot(np.nanmean(dfmat, axis=0), 'k', alpha=1, linewidth=1)


        #stim_frames = [traces.attrs['frame_on']
        #pl.plot()
        pidx += 1

    sns.despine(offset=2, trim=True)

    pl.savefig(os.path.join(roi_output_figdir, '%s.png' % roi))
    pl.close()



#%%
#
#for trial in trial_list:
#    stimdurs.appends(trialdict[trial]['frame_stim_off'] - trialdict[trial]['frame_stim_on'])
#
#    durs = [(t, trialdict[trial]['frame_stim_off'] - trialdict[trial]['frame_stim_on']) for t, trial in enumerate(trial_list)]
#    short = [d[0] for d in durs if d[1]==45]
#    short_trials = ['trial%05d' % int(i+1) for i in short]
#
#
#    tracemat.shape
#    pl.figure()
#    for t in range(len(curr_trials)):
#        pl.plot(tracemat[t,:], 'k', alpha=0.5)
#    pl.plot(np.mean(tracemat, axis=0), alpha=1.0)
#
#
#
#        ntransforms = len(stimconfigs[stim].keys())
#        transform_combos = list(itertools.product(list(set(positions)), list(set(sizes)), list(set(rotations)), filepaths, filehashes))
#        ncombinations = len(transform_combos)
#
#        stimconfigs[stim]['position'] = list(set(positions))
#        stimconfigs[stim]['rotation'] = list(set(rotations))
#        stimconfigs[stim]['size'] = list(set(sizes))
#
#
#
#        for config_idx in range(ncombinations):
#
#            curr_pos = transform_combos[config_idx][0]
#            curr_size = transform_combos[config_idx][1]
#            curr_rot = transform_combos[config_idx][2]
#
#            curr_trials = [trial for trial in trial_list\
#                            if (trialdict[trial]['stimuli']['position'] == curr_pos).all()\
#                            and trialdict[trial]['stimuli']['rotation'] == curr_rot\
#                            and (trialdict[trial]['stimuli']['scale'] == curr_size).all()])]
#
#
#    stimulus[trial for trial in trial_list if trialdict[t]['stimuli']['stimulus'] == stim]
#
#[trial for trial in trialdict.keys() if ]


#%%

#            stimdict[stim][currfile].stiminfo = trialdict[currfile][currtrial]['stiminfo']

#            stimdict[stim][currfile].trials.append(trialnum)
#            stimdict[stim][currfile].frames.append(framenums)
#            #stimdict[stim][currfile].frames_sec.append(frametimes)
#            stimdict[stim][currfile].stim_on_idx.append(first_frame_on)
#            stimdict[stim][currfile].stim_dur = round(stim_on_sec) #.append(stim_on_sec)
#            stimdict[stim][currfile].iti_dur = round(iti_full) #.append(iti_full)
#            stimdict[stim][currfile].baseline_dur = round(iti_pre) #.append(iti_full)
#
#            stimdict[stim][currfile].volumerate = volumerate

        #print [len(stimdict[stim][currfile].frames[i]) for i in range(len(stimdict[stim][currfile].frames))]

#    # Save to PKL:
#    curr_stimdict_pkl = 'stimdict.pkl' #% currfile # % currslice
#    print curr_stimdict_pkl
#    with open(os.path.join(paradigm_outdir, curr_stimdict_pkl), 'wb') as f:
#        pkl.dump(stimdict, f, protocol=pkl.HIGHEST_PROTOCOL) #, f, indent=4)
#
#    # Save to JSON:
#    for fi in range(nfiles):
#        currfile = "File%03d" % int(fi+1)
#        for stim in stimdict.keys():
#            stimdict[stim][currfile] = serialize_json(stimdict[stim][currfile])
#
#    curr_stimdict_json = 'stimdict.json' #% currfile # % currslice
#    print curr_stimdict_json
#    with open(os.path.join(paradigm_outdir, curr_stimdict_json), 'w') as f:
#        dump(stimdict, f, indent=4)
#

