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
import seaborn as sns
import pylab as pl
import numpy as np
from pipeline.python.utils import natural_keys, hash_file_read_only, print_elapsed_time, hash_file
pp = pprint.PrettyPrinter(indent=4)

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

(options, args) = parser.parse_args()


# Set USER INPUT options:
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

#%%
#
#rootdir = '/mnt/odyssey' #'/nas/volume1/2photon/data'
#animalid = 'CE074'
#session = '20180215'
#acquisition = 'FOV1_zoom1x_V1'
#run = 'blobs'
#trace_id = 'traces001'
#custom_mw = False
#same_order = False
#
#ylim_min = -1.0
#ylim_max = 2.0
#iti_pre = 1.0
#trace_type = 'raw'
#create_new = True
#
#%%
# =============================================================================
# Get meta info for RUN:
# =============================================================================

run_dir = os.path.join(rootdir, animalid, session, acquisition, run)

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
# =============================================================================
# Load TRACE ID:
# =============================================================================

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

traceid_dir = TID['DST']
if rootdir not in traceid_dir:
    orig_root = traceid_dir.split('/%s/%s' % (animalid, session))[0]
    traceid_dir = traceid_dir.replace(orig_root, rootdir)
    print "Replacing orig root with dir:", traceid_dir
trace_hash = TID['trace_hash']

excluded_tiffs = TID['PARAMS']['excluded_tiffs']
print "Current trace ID - %s - excludes %i tiffs." % (trace_id, len(excluded_tiffs))
excluded_tiff_idxs = [int(tf[4:])-1 for tf in excluded_tiffs]

#%%
# =============================================================================
# Get paradigm info:
# =============================================================================
paradigm_dir = os.path.join(run_dir, 'paradigm')

if custom_mw is True:
    # Get trial info if custom (non-MW) stim presentation protocols:
    # -------------------------------------------------------------------------
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

        # Get stim-order files:
        stimorder_fns = sorted([f for f in os.listdir(paradigm_dir) if 'stimorder' in f and f.endswith('txt')], key=natural_keys)
        print "Found %i stim-order files, and %i TIFFs." % (len(stimorder_fns), nfiles)
        if len(stimorder_fns) < ntiffs:
            if same_order: # Same stimulus order for each file (complete set)
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
    # -------------------------------------------------------------------------
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

        # Check whether acquisition method is one-to-one (1 aux file per SI tif) or single-to-many:
        if trialdict[trial_list[0]]['ntiffs_per_auxfile'] == 1:
            one_to_one =  True
        else:
            one_to_one = False

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

nslices_full = int(round(runinfo['frame_rate']/runinfo['volume_rate']))
nframes_per_file = nslices_full * nvolumes

#%%
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


#%%
# =============================================================================
# Assign frame indices for specified trial epochs:
# =============================================================================
print "-----------------------------------------------------------------------"
print "Getting frame indices for trial epochs..."

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
    parsed_frames.attrs['framerate'] = framerate
    parsed_frames.attrs['volumerate'] = volumerate
    parsed_frames.attrs['baseline_dur'] = iti_pre
    #run_grp.attrs['creation_time'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    #%
    # 1. Get stimulus preseentation order for each TIFF found:
    try:
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
                                 if trialdict[t]['block_idx'] == tiffnum]
                trials_in_run = sorted([t for t in trial_list if trialdict[t]['block_idx'] == tiffnum], key=natural_keys)

            #unique_stims = sorted(set(stimorder), key=natural_keys)

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
                # Check to make sure that rounding errors do not cause frame idxs to go beyond the number of frames in a file:
                if postframes[-1] > len(vol_idxs):
                    extraframes = [p for p in postframes if p > len(vol_idxs)-1]
                    postframes = [p for p in postframes if p <= len(vol_idxs)-1]
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

                if one_to_one is True:
                    framenums_in_run = np.array(framenums) + (nframes_per_file*tiffnum)
                    abs_stim_on_idx = first_frame_on + (nframes_per_file*tiffnum)
                else:
                    framenums_in_run = np.array(framenums)
                    abs_stim_on_idx = first_frame_on

                fridxs = parsed_frames.create_dataset('/'.join((currtrial_in_run, 'frames_in_run')), np.array(framenums_in_run).shape, np.array(framenums_in_run).dtype)
                fridxs[...] = np.array(framenums_in_run)
                fridxs.attrs['trial'] = currtrial_in_run
                fridxs.attrs['aux_file_idx'] = tiffnum
                fridxs.attrs['stim_on_idx'] = abs_stim_on_idx
                fridxs.attrs['stim_dur_sec'] = stim_on_sec
                fridxs.attrs['iti_dur_sec'] = iti_full
                fridxs.attrs['baseline_dur_sec'] = iti_pre
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

#% Rename FRAME file with hash:
#frame_file_hashid = hash_file(parsed_frames_filepath)
#framestruct_path = '%s_%s.hdf5' % (os.path.splitext(parsed_frames_filepath)[0], frame_file_hashid)
#os.rename(parsed_frames_filepath, framestruct_path)

#print "Finished assigning frame idxs across all tiffs to trials in run %s." % run
#print "Saved frame file to:", framestruct_path

#%%
# =============================================================================
# Get all unique stimulus configurations:
# =============================================================================

print "-----------------------------------------------------------------------"
print "Getting stimulus configs..."

trial_list = sorted(trialdict.keys(), key=natural_keys)
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

#%%
# =============================================================================
# Load timecourses for specified trace set:
# =============================================================================
print "-----------------------------------------------------------------------"
print "Loading time courses for run %s, from trace set: %s" % (run, trace_id)

try:
    # Since roi_timecourses are datestamped and hashed, sort all found files and use the most recent file:
    tcourse_fn = sorted([t for t in os.listdir(traceid_dir) if t.endswith('hdf5') and 'roi_timecourses' in t], key=natural_keys)[-1]
    roi_timecourses = h5py.File(os.path.join(traceid_dir, tcourse_fn), 'r')
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

print "-----------------------------------------------------------------------"

#%%
# =============================================================================
# Group ROI time-courses for each trial by stimulus config:
# =============================================================================
print "-----------------------------------------------------------------------"
print "Getting ROI timecourses by trial and stim config"

parsed_frames = h5py.File(parsed_frames_filepath, 'r')  # Load PARSED FRAMES output file:

sliceids = dict((curr_slices[s], s) for s in range(len(curr_slices)))

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

#%%
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

#%%
# Save config info to file:
config_filename = 'stimulus_configs.json'
with open(os.path.join(paradigm_dir, config_filename), 'w') as f:
    json.dump(configs, f, sort_keys=True, indent=4)


#%%
# =============================================================================
# Set plotting params for trial average plots for each ROI:
# =============================================================================

# Change config filepath for plotting:
if 'grating' not in stimtype:
    configparams = configs[configs.keys()[0]].keys()
    for config in configs.keys():
        configs[config]['filename'] = os.path.split(configs[config]['filepath'])[1]

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

else:
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

# get the tick label font size
fontsize_pt = 20 #float(plt.rcParams['ytick.labelsize'])
dpi = 72.27
spacer = 20

# comput the matrix height in points and inches
matrix_height_pt = fontsize_pt * nrows * spacer
matrix_height_in = matrix_height_pt / dpi

# compute the required figure height
top_margin = 0.01  # in percentage of the figure height
bottom_margin = 0.05 # in percentage of the figure height
figure_height = matrix_height_in / (1 - top_margin - bottom_margin)


#%% For each ROI, plot PSTH for all stim configs:

roi_psth_dir = os.path.join(traceid_dir, 'figures', 'psths', trace_type)
if not os.path.exists(roi_psth_dir):
    os.makedirs(roi_psth_dir)
print "Saving PSTH plots to: %s" % roi_psth_dir

roi_trials = h5py.File(roi_trials_by_stim_path, 'r')   # Load ROI TRIALS file
parsed_frames = h5py.File(parsed_frames_filepath, 'r')  # Load PARSED FRAMES output file:

try:
    for roi in roi_list:
        #%
        print roi
        if nrows==1:
            figwidth_multiplier = ncols*1
        else:
            figwidth_multiplier = 1


        for img in subplot_stimlist.keys():
            curr_subplots = subplot_stimlist[img]
            if stimid_only is True:
                figname = 'all_objects_default_pos_size'
            else:
                figname = '%s_pos%i_size%i' % (os.path.splitext(img)[0], len(position_vals), len(size_vals))

            fig, axs = pl.subplots(
        	    nrows=nrows,
        	    ncols=ncols,
        	    sharex=True,
        	    sharey=True,
        	    figsize=(figure_height*figwidth_multiplier,figure_height),
        	    gridspec_kw=dict(top=1-top_margin, bottom=bottom_margin, wspace=0.05, hspace=0.05))

            row=0
            col=0
            plotidx = 0

            nframes_on = parsed_frames['trial00001']['frames_in_run'].attrs['stim_dur_sec'] * volumerate
            stim_dur = trialdict['trial00001']['stim_dur_ms']/1E3
            iti_dur = trialdict['trial00001']['iti_dur_ms']/1E3
            tpoints = [int(i) for i in np.arange(-1*iti_pre, stim_dur+iti_dur)]

            #roi = 'roi00003'
            for configname in curr_subplots:
                curr_slice = roi_trials[configname][roi].attrs['slice']
                roi_in_slice = roi_trials[configname][roi].attrs['idx_in_slice']

                if col==(ncols) and nrows>1:
                    row += 1
                    col = 0
                if len(axs.shape)>1:
                    ax_curr = axs[row, col] #, col]
                else:
                    ax_curr = axs[col]

                stim_trials = sorted([t for t in roi_trials[configname][roi].keys()], key=natural_keys)
                nvols = max([roi_trials[configname][roi][t][trace_type].shape[0] for t in stim_trials])
                ntrials = len(stim_trials)
                trialmat = np.ones((ntrials, nvols)) * np.nan
                dfmat = []

                first_on = int(min([[i for i in roi_trials[configname][roi][t].attrs['frame_idxs']].index(roi_trials[configname][roi][t].attrs['volume_stim_on']) for t in stim_trials]))
                tsecs = (np.arange(0, nvols) - first_on ) / volumerate

                if 'grating' in stimtype:
                    stimname = 'Ori %.0f, SF: %.2f' % (configs[configname]['rotation'], configs[configname]['frequency'])
                else:
                    stimname = '%s- pos (%.1f, %.1f) - siz %.1f' % (os.path.splitext(configs[configname]['filename'])[0], configs[configname]['position'][0], configs[configname]['position'][1], configs[configname]['scale'][0])

                ax_curr.annotate(stimname,xy=(0.1,1), xycoords='axes fraction', horizontalalignment='middle', verticalalignment='top', weight='bold')

                for tidx, trial in enumerate(sorted(stim_trials, key=natural_keys)):
                    trial_timecourse = roi_trials[configname][roi][trial][trace_type]
                    curr_on = int([i for i in roi_trials[configname][roi][trial].attrs['frame_idxs']].index(int(roi_trials[configname][roi][trial].attrs['volume_stim_on'])))
                    trialmat[tidx, first_on:first_on+len(trial_timecourse[curr_on:])] = trial_timecourse[curr_on:]
                    if first_on < curr_on:
                        trialmat[tidx, 0:first_on] = trial_timecourse[1:curr_on]
                    else:
                        trialmat[tidx, 0:first_on] = trial_timecourse[0:curr_on]
                    #trialmat[tidx, first_on-curr_on:first_on] = trace[0:curr_on]

                    baseline = np.nanmean(trialmat[tidx, 0:first_on]) #[0:on_idx])
                    if baseline == 0 or baseline == np.nan:
                        df = np.ones(trialmat[tidx,:].shape) * np.nan
                    else:
                        df = (trialmat[tidx,:] - baseline) / baseline

                    ax_curr.plot(tsecs, df, 'k', alpha=0.2, linewidth=0.5)
                    ax_curr.plot([tsecs[first_on], tsecs[first_on]+nframes_on/volumerate], [0, 0], 'r', linewidth=1, alpha=0.1)

                    dfmat.append(df)

                ax_curr.plot(tsecs, np.nanmean(dfmat, axis=0), 'k', alpha=1, linewidth=1)
                if universal_scale is True:
                    ax_curr.set_ylim([ylim_min, ylim_max])
#                else:
#                    ax_curr.set_ylim([ylim_min, 
                ax_curr.set(xticks=tpoints)
                ax_curr.tick_params(axis='x', which='both',length=0)

                col = col + 1
                plotidx += 1

            sns.despine(offset=2, trim=True)
            pl.title(roi)
            #%
            psth_fig_fn = '%s_%s_%s_%s_%s.png' % (roi, curr_slice, roi_in_slice, trace_type, figname)
            pl.savefig(os.path.join(roi_psth_dir, psth_fig_fn))
            pl.close()
            print psth_fig_fn
except Exception as e:
    print "--- Error plotting PSTH ---------------------------------"
    print roi, configname, trial
    traceback.print_exc()
#    print "---------------------------------------------------------"
finally:
    roi_trials.close()
    parsed_frames.close()

# =============================================================================
# Plot included/excluded trials with eyetracker info
# =============================================================================

if eyetracker:
    # Get eye featue info:
    eye_file_dir = os.path.join(run_dir,'eyetracker','files')
    eye_info_fn = 'parsed_eye_%s_%s_%s.json'%(session,animalid,run)

    print 'Getting eye feature info from: %s'%(os.path.join(eye_file_dir, eye_info_fn))
    with open(os.path.join(eye_file_dir, eye_info_fn), 'r') as f:
        eye_info = json.load(f)

    roi_psth_include_dir = os.path.join(traceid_dir, 'figures', 'psths',trace_type, 'eye_include')
    if not os.path.exists(roi_psth_include_dir):
        os.makedirs(roi_psth_include_dir)
    print "Saving included PSTH plots to: %s" % roi_psth_include_dir

    roi_psth_exclude_dir = os.path.join(traceid_dir, 'figures', 'psths',trace_type, 'eye_exclude')
    if not os.path.exists(roi_psth_exclude_dir):
        os.makedirs(roi_psth_exclude_dir)
    print "Saving excluded PSTH plots to: %s" % roi_psth_exclude_dir


    roi_trials = h5py.File(roi_trials_by_stim_path, 'r')   # Load ROI TRIALS file
    parsed_frames = h5py.File(parsed_frames_filepath, 'r')  # Load PARSED FRAMES output file:

    try:
        for roi in roi_list:
            #%
            print roi
            if nrows==1:
                figwidth_multiplier = ncols*1
            else:
                figwidth_multiplier = 1


            for img in subplot_stimlist.keys():
                curr_subplots = subplot_stimlist[img]
                if stimid_only is True:
                    figname = 'all_objects_default_pos_size'
                else:
                    figname = '%s_pos%i_size%i' % (os.path.splitext(img)[0], len(position_vals), len(size_vals))

                fig, axs = pl.subplots(
                    nrows=nrows,
                    ncols=ncols,
                    sharex=True,
                    sharey=True,
                    figsize=(figure_height*figwidth_multiplier,figure_height),
                    gridspec_kw=dict(top=1-top_margin, bottom=bottom_margin, wspace=0.05, hspace=0.05))

                row=0
                col=0
                plotidx = 0

                nframes_on = parsed_frames['trial00001']['frames_in_run'].attrs['stim_dur_sec'] * volumerate
                stim_dur = trialdict['trial00001']['stim_dur_ms']/1E3
                iti_dur = trialdict['trial00001']['iti_dur_ms']/1E3
                tpoints = [int(i) for i in np.arange(-1*iti_pre, stim_dur+iti_dur)]

                #roi = 'roi00003'
                for configname in curr_subplots:
                    curr_slice = roi_trials[configname][roi].attrs['slice']
                    roi_in_slice = roi_trials[configname][roi].attrs['idx_in_slice']

                    if col==(ncols) and nrows>1:
                        row += 1
                        col = 0
                    if len(axs.shape)>1:
                        ax_curr = axs[row, col] #, col]
                    else:
                        ax_curr = axs[col]

                    stim_trials = sorted([t for t in roi_trials[configname][roi].keys()], key=natural_keys)
                    nvols = max([roi_trials[configname][roi][t][trace_type].shape[0] for t in stim_trials])
                    ntrials = len(stim_trials)
                    trialmat = np.ones((ntrials, nvols)) * np.nan
                    dfmat = []

                    first_on = int(min([[i for i in roi_trials[configname][roi][t].attrs['frame_idxs']].index(roi_trials[configname][roi][t].attrs['volume_stim_on']) for t in stim_trials]))
                    tsecs = (np.arange(0, nvols) - first_on ) / volumerate

                    if 'grating' in stimtype:
                        stimname = 'Ori %.0f, SF: %.2f' % (configs[configname]['rotation'], configs[configname]['frequency'])
                    else:
                        stimname = '%s- pos (%.1f, %.1f) - siz %.1f' % (os.path.splitext(configs[configname]['filename'])[0], configs[configname]['position'][0], configs[configname]['position'][1], configs[configname]['scale'][0])

                    ax_curr.annotate(stimname,xy=(0.1,1), xycoords='axes fraction', horizontalalignment='middle', verticalalignment='top', weight='bold')

                    for tidx, trial in enumerate(sorted(stim_trials, key=natural_keys)):
                        trial_timecourse = roi_trials[configname][roi][trial][trace_type]
                        curr_on = int([i for i in roi_trials[configname][roi][trial].attrs['frame_idxs']].index(int(roi_trials[configname][roi][trial].attrs['volume_stim_on'])))
                        if eye_info[trial]['include_trial']:
                            trialmat[tidx, first_on:first_on+len(trial_timecourse[curr_on:])] = trial_timecourse[curr_on:]
                            if first_on < curr_on:
                                trialmat[tidx, 0:first_on] = trial_timecourse[1:curr_on]
                            else:
                                trialmat[tidx, 0:first_on] = trial_timecourse[0:curr_on]
                            #trialmat[tidx, first_on-curr_on:first_on] = trace[0:curr_on]

                            baseline = np.nanmean(trialmat[tidx, 0:first_on]) #[0:on_idx])
                            if baseline == 0 or baseline == np.nan:
                                df = np.ones(trialmat[tidx,:].shape) * np.nan
                            else:
                                df = (trialmat[tidx,:] - baseline) / baseline

                            ax_curr.plot(tsecs, df, 'b', alpha=0.2, linewidth=0.5)
                            ax_curr.plot([tsecs[first_on], tsecs[first_on]+nframes_on/volumerate], [0, 0], 'r', linewidth=1, alpha=0.1)

                            dfmat.append(df)

                    ax_curr.plot(tsecs, np.nanmean(dfmat, axis=0), 'b', alpha=1, linewidth=1)
                    if universal_scale is True:
                        ax_curr.set_ylim([ylim_min, ylim_max])
    #                else:
    #                    ax_curr.set_ylim([ylim_min, 
                    ax_curr.set(xticks=tpoints)
                    ax_curr.tick_params(axis='x', which='both',length=0)

                    col = col + 1
                    plotidx += 1

                sns.despine(offset=2, trim=True)
                pl.title(roi)
                #%
                psth_fig_fn = '%s_%s_%s_%s_%s.png' % (roi, curr_slice, roi_in_slice, trace_type, figname)
                pl.savefig(os.path.join(roi_psth_include_dir, psth_fig_fn))
                pl.close()
                print psth_fig_fn
        for roi in roi_list:
            #%
            print roi
            if nrows==1:
                figwidth_multiplier = ncols*1
            else:
                figwidth_multiplier = 1


            for img in subplot_stimlist.keys():
                curr_subplots = subplot_stimlist[img]
                if stimid_only is True:
                    figname = 'all_objects_default_pos_size'
                else:
                    figname = '%s_pos%i_size%i' % (os.path.splitext(img)[0], len(position_vals), len(size_vals))

                fig, axs = pl.subplots(
                    nrows=nrows,
                    ncols=ncols,
                    sharex=True,
                    sharey=True,
                    figsize=(figure_height*figwidth_multiplier,figure_height),
                    gridspec_kw=dict(top=1-top_margin, bottom=bottom_margin, wspace=0.05, hspace=0.05))

                row=0
                col=0
                plotidx = 0

                nframes_on = parsed_frames['trial00001']['frames_in_run'].attrs['stim_dur_sec'] * volumerate
                stim_dur = trialdict['trial00001']['stim_dur_ms']/1E3
                iti_dur = trialdict['trial00001']['iti_dur_ms']/1E3
                tpoints = [int(i) for i in np.arange(-1*iti_pre, stim_dur+iti_dur)]

                #roi = 'roi00003'
                for configname in curr_subplots:
                    curr_slice = roi_trials[configname][roi].attrs['slice']
                    roi_in_slice = roi_trials[configname][roi].attrs['idx_in_slice']

                    if col==(ncols) and nrows>1:
                        row += 1
                        col = 0
                    if len(axs.shape)>1:
                        ax_curr = axs[row, col] #, col]
                    else:
                        ax_curr = axs[col]

                    stim_trials = sorted([t for t in roi_trials[configname][roi].keys()], key=natural_keys)
                    nvols = max([roi_trials[configname][roi][t][trace_type].shape[0] for t in stim_trials])
                    ntrials = len(stim_trials)
                    trialmat = np.ones((ntrials, nvols)) * np.nan
                    dfmat = []

                    first_on = int(min([[i for i in roi_trials[configname][roi][t].attrs['frame_idxs']].index(roi_trials[configname][roi][t].attrs['volume_stim_on']) for t in stim_trials]))
                    tsecs = (np.arange(0, nvols) - first_on ) / volumerate

                    if 'grating' in stimtype:
                        stimname = 'Ori %.0f, SF: %.2f' % (configs[configname]['rotation'], configs[configname]['frequency'])
                    else:
                        stimname = '%s- pos (%.1f, %.1f) - siz %.1f' % (os.path.splitext(configs[configname]['filename'])[0], configs[configname]['position'][0], configs[configname]['position'][1], configs[configname]['scale'][0])

                    ax_curr.annotate(stimname,xy=(0.1,1), xycoords='axes fraction', horizontalalignment='middle', verticalalignment='top', weight='bold')

                    for tidx, trial in enumerate(sorted(stim_trials, key=natural_keys)):
                        trial_timecourse = roi_trials[configname][roi][trial][trace_type]
                        curr_on = int([i for i in roi_trials[configname][roi][trial].attrs['frame_idxs']].index(int(roi_trials[configname][roi][trial].attrs['volume_stim_on'])))
                        if not eye_info[trial]['include_trial']:
                            trialmat[tidx, first_on:first_on+len(trial_timecourse[curr_on:])] = trial_timecourse[curr_on:]
                            if first_on < curr_on:
                                trialmat[tidx, 0:first_on] = trial_timecourse[1:curr_on]
                            else:
                                trialmat[tidx, 0:first_on] = trial_timecourse[0:curr_on]
                            #trialmat[tidx, first_on-curr_on:first_on] = trace[0:curr_on]

                            baseline = np.nanmean(trialmat[tidx, 0:first_on]) #[0:on_idx])
                            if baseline == 0 or baseline == np.nan:
                                df = np.ones(trialmat[tidx,:].shape) * np.nan
                            else:
                                df = (trialmat[tidx,:] - baseline) / baseline

                            ax_curr.plot(tsecs, df, 'r', alpha=0.2, linewidth=0.5)
                            ax_curr.plot([tsecs[first_on], tsecs[first_on]+nframes_on/volumerate], [0, 0], 'k', linewidth=1, alpha=0.1)

                            dfmat.append(df)

                    ax_curr.plot(tsecs, np.nanmean(dfmat, axis=0), 'r', alpha=1, linewidth=1)
                    if universal_scale is True:
                        ax_curr.set_ylim([ylim_min, ylim_max])
    #                else:
    #                    ax_curr.set_ylim([ylim_min, 
                    ax_curr.set(xticks=tpoints)
                    ax_curr.tick_params(axis='x', which='both',length=0)

                    col = col + 1
                    plotidx += 1

                sns.despine(offset=2, trim=True)
                pl.title(roi)
                #%
                psth_fig_fn = '%s_%s_%s_%s_%s.png' % (roi, curr_slice, roi_in_slice, trace_type, figname)
                pl.savefig(os.path.join(roi_psth_exclude_dir, psth_fig_fn))
                pl.close()
                print psth_fig_fn
    except Exception as e:
        print "--- Error plotting PSTH ---------------------------------"
        print roi, configname, trial
        traceback.print_exc()
    #    print "---------------------------------------------------------"
    finally:
        roi_trials.close()
        parsed_frames.close()

    #%%
# =============================================================================
# Make and save scatterplots of trial response vs. eye features:
# =============================================================================

    sns.set_style("darkgrid", {"axes.facecolor": ".9"})
    roi_trials = h5py.File(roi_trials_by_stim_path, 'r')   # Load ROI TRIALS file
    parsed_frames = h5py.File(parsed_frames_filepath, 'r')  # Load PARSED FRAMES output file:

    roi_scatter_stim_dir = os.path.join(traceid_dir, 'figures', 'eye_scatterplots','stimulation')
    if not os.path.exists(roi_scatter_stim_dir):
        os.makedirs(roi_scatter_stim_dir)
    print "Saving scatter plots to: %s" % roi_scatter_stim_dir


    for roi in roi_list:
        #%
        print roi
        if nrows==1:
            figwidth_multiplier = ncols*1
        else:
            figwidth_multiplier = 3


        for img in subplot_stimlist.keys():
            curr_subplots = subplot_stimlist[img]
            if stimid_only is True:
                figname = 'all_objects_default_pos_size'
            else:
                figname = '%s_pos%i_size%i' % (os.path.splitext(img)[0], len(position_vals), len(size_vals))

            fig, axs = pl.subplots(
                nrows=nrows,
                ncols=ncols,
                sharex=True,
                sharey=True,
                figsize=(figure_height*figwidth_multiplier,figure_height))

            row=0
            col=0
            plotidx = 0

            nframes_on = parsed_frames['trial00001']['frames_in_run'].attrs['stim_dur_sec'] * volumerate
            stim_dur = trialdict['trial00001']['stim_dur_ms']/1E3
            iti_dur = trialdict['trial00001']['iti_dur_ms']/1E3
            tpoints = [int(i) for i in np.arange(-1*iti_pre, stim_dur+iti_dur)]

            #roi = 'roi00003'
            for configname in curr_subplots:
                df_values = []
                pupil_rad = []
                curr_slice = roi_trials[configname][roi].attrs['slice']
                roi_in_slice = roi_trials[configname][roi].attrs['idx_in_slice']

                if col==(ncols) and nrows>1:
                    row += 1
                    col = 0
                if len(axs.shape)>1:
                    ax_curr = axs[row, col] #, col]
                else:
                    ax_curr = axs[col]

                stim_trials = sorted([t for t in roi_trials[configname][roi].keys()], key=natural_keys)
                nvols = max([roi_trials[configname][roi][t][trace_type].shape[0] for t in stim_trials])
                ntrials = len(stim_trials)
                trialmat = np.ones((ntrials, nvols)) * np.nan
                dfmat = []

                first_on = int(min([[i for i in roi_trials[configname][roi][t].attrs['frame_idxs']].index(roi_trials[configname][roi][t].attrs['volume_stim_on']) for t in stim_trials]))
                tsecs = (np.arange(0, nvols) - first_on ) / volumerate

                if 'grating' in stimtype:
                    stimname = 'Ori %.0f, SF: %.2f' % (configs[configname]['rotation'], configs[configname]['frequency'])
                else:
                    stimname = '%s- pos (%.1f, %.1f) - siz %.1f' % (os.path.splitext(configs[configname]['filename'])[0], configs[configname]['position'][0], configs[configname]['position'][1], configs[configname]['scale'][0])

                ax_curr.annotate(stimname,xy=(0.1,1), xycoords='axes fraction', horizontalalignment='middle', verticalalignment='top', weight='bold')

                for tidx, trial in enumerate(sorted(stim_trials, key=natural_keys)):
                    trial_timecourse = roi_trials[configname][roi][trial][trace_type]
                    curr_on = int([i for i in roi_trials[configname][roi][trial].attrs['frame_idxs']].index(int(roi_trials[configname][roi][trial].attrs['volume_stim_on'])))
                    trialmat[tidx, first_on:first_on+len(trial_timecourse[curr_on:])] = trial_timecourse[curr_on:]
                    if first_on < curr_on:
                        trialmat[tidx, 0:first_on] = trial_timecourse[1:curr_on]
                    else:
                        trialmat[tidx, 0:first_on] = trial_timecourse[0:curr_on]
                    #trialmat[tidx, first_on-curr_on:first_on] = trace[0:curr_on]

                    baseline = np.nanmean(trialmat[tidx, 0:first_on]) #[0:on_idx])
                    if baseline == 0 or baseline == np.nan:
                        df = np.ones(trialmat[tidx,:].shape) * np.nan
                    else:
                        df = (trialmat[tidx,:] - baseline) / baseline

                    df_values.append(np.nanmean(df[first_on:first_on+int(nframes_on)]))
                    pupil_rad.append(float(eye_info[trial]['pupil_size_stim']))

                ax_curr.plot(pupil_rad,df_values,'ob')
                col = col + 1
                plotidx += 1

        fig_fn = '%s_%s_%s_%s.png' % (roi, curr_slice, roi_in_slice, trace_type)
        figure_file = os.path.join(roi_scatter_stim_dir,fig_fn)

        pl.savefig(figure_file, bbox_inches='tight')
        pl.close()



    roi_scatter_base_dir = os.path.join(traceid_dir, 'figures', 'eye_scatterplots','baseline')
    if not os.path.exists(roi_scatter_base_dir):
        os.makedirs(roi_scatter_base_dir)
    print "Saving scatter plots to: %s" % roi_scatter_base_dir

    for roi in roi_list:
        #%
        print roi
        if nrows==1:
            figwidth_multiplier = ncols*1
        else:
            figwidth_multiplier = 1


        for img in subplot_stimlist.keys():
            curr_subplots = subplot_stimlist[img]
            if stimid_only is True:
                figname = 'all_objects_default_pos_size'
            else:
                figname = '%s_pos%i_size%i' % (os.path.splitext(img)[0], len(position_vals), len(size_vals))

            fig, axs = pl.subplots(
                nrows=nrows,
                ncols=ncols,
                sharex=True,
                sharey=True,
                figsize=(figure_height*2,figure_height))

            row=0
            col=0
            plotidx = 0

            nframes_on = parsed_frames['trial00001']['frames_in_run'].attrs['stim_dur_sec'] * volumerate
            stim_dur = trialdict['trial00001']['stim_dur_ms']/1E3
            iti_dur = trialdict['trial00001']['iti_dur_ms']/1E3
            tpoints = [int(i) for i in np.arange(-1*iti_pre, stim_dur+iti_dur)]

            #roi = 'roi00003'
            for configname in curr_subplots:
                df_values = []
                pupil_rad = []
                curr_slice = roi_trials[configname][roi].attrs['slice']
                roi_in_slice = roi_trials[configname][roi].attrs['idx_in_slice']

                if col==(ncols) and nrows>1:
                    row += 1
                    col = 0
                if len(axs.shape)>1:
                    ax_curr = axs[row, col] #, col]
                else:
                    ax_curr = axs[col]

                stim_trials = sorted([t for t in roi_trials[configname][roi].keys()], key=natural_keys)
                nvols = max([roi_trials[configname][roi][t][trace_type].shape[0] for t in stim_trials])
                ntrials = len(stim_trials)
                trialmat = np.ones((ntrials, nvols)) * np.nan
                dfmat = []

                first_on = int(min([[i for i in roi_trials[configname][roi][t].attrs['frame_idxs']].index(roi_trials[configname][roi][t].attrs['volume_stim_on']) for t in stim_trials]))
                tsecs = (np.arange(0, nvols) - first_on ) / volumerate

                if 'grating' in stimtype:
                    stimname = 'Ori %.0f, SF: %.2f' % (configs[configname]['rotation'], configs[configname]['frequency'])
                else:
                    stimname = '%s- pos (%.1f, %.1f) - siz %.1f' % (os.path.splitext(configs[configname]['filename'])[0], configs[configname]['position'][0], configs[configname]['position'][1], configs[configname]['scale'][0])

                ax_curr.annotate(stimname,xy=(0.1,1), xycoords='axes fraction', horizontalalignment='middle', verticalalignment='top', weight='bold')

                for tidx, trial in enumerate(sorted(stim_trials, key=natural_keys)):
                    trial_timecourse = roi_trials[configname][roi][trial][trace_type]
                    curr_on = int([i for i in roi_trials[configname][roi][trial].attrs['frame_idxs']].index(int(roi_trials[configname][roi][trial].attrs['volume_stim_on'])))
                    trialmat[tidx, first_on:first_on+len(trial_timecourse[curr_on:])] = trial_timecourse[curr_on:]
                    if first_on < curr_on:
                        trialmat[tidx, 0:first_on] = trial_timecourse[1:curr_on]
                    else:
                        trialmat[tidx, 0:first_on] = trial_timecourse[0:curr_on]
                    #trialmat[tidx, first_on-curr_on:first_on] = trace[0:curr_on]

                    baseline = np.nanmean(trialmat[tidx, 0:first_on]) #[0:on_idx])
                    if baseline == 0 or baseline == np.nan:
                        df = np.ones(trialmat[tidx,:].shape) * np.nan
                    else:
                        df = (trialmat[tidx,:] - baseline) / baseline

                    df_values.append(np.nanmean(df[first_on:first_on+int(nframes_on)]))
                    pupil_rad.append(float(eye_info[trial]['pupil_size_baseline']))

                ax_curr.plot(pupil_rad,df_values,'ob')
                col = col + 1
                plotidx += 1

        fig_fn = '%s_%s_%s_%s.png' % (roi, curr_slice, roi_in_slice, trace_type)
        figure_file = os.path.join(roi_scatter_base_dir,fig_fn)

        pl.savefig(figure_file, bbox_inches='tight')
        pl.close()
    roi_trials.close()
    parsed_frames.close()

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

