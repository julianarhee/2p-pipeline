###!/usr/bin/env python2
import os
import json
import re
import optparse
import operator
import h5py
import numpy as np
from pipeline.python.utils import hash_file


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


parser.add_option('--custom', action="store_true",
                  dest="custom_mw", default=False, help="Not using MW (custom params must be specified)")
parser.add_option('--order', action="store_true",
                  dest="same_order", default=False, help="Set if same stimulus order across all files (1 stimorder.txt)")

parser.add_option('-p', '--pre', action="store",
                  dest="iti_pre", default=1.0, help="Time (s) pre-stimulus to use for baseline. [default: 1.0]")

# parser.add_option('-i', '--iti', action="store",
#                   dest="iti_full", default='', help="Time (s) between stimuli (inter-trial interval).")


# Only need to set these if using custom-paradigm file:
parser.add_option('-O', '--stimon', action="store",
                  dest="stim_on_sec", default=0, help="Time (s) stimulus ON.")

parser.add_option('-t', '--vol', action="store",
                  dest="vols_per_trial", default=0, help="Num volumes per trial. Specifiy if custom_mw=True")
parser.add_option('-v', '--first', action="store",
                  dest="first_stim_volume_num", default=0, help="First volume stimulus occurs (py-indexed). Specifiy if custom_mw=True")


parser.add_option('--flyback', action="store_true",
                  dest="flyback_corrected", default=False, help="Set if corrected extra flyback frames (in process_raw.py->correct_flyback.py")


(options, args) = parser.parse_args()

flyback_corrected = options.flyback_corrected

(options, args) = parser.parse_args()


# Set USER INPUT options:
rootdir = options.rootdir
animalid = options.animalid
session = options.session
acquisition = options.acquisition
run = options.run

iti_pre = float(options.iti_pre)

custom_mw = options.custom_mw
same_order = options.same_order #False #True




#%%

run_dir = os.path.join(rootdir, animalid, session, acquisition, run)
paradigm_dir = os.path.join(run_dir, 'paradigm')

outfile = os.path.join(paradigm_dir, 'frames_.hdf5')

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
    nframes_on = stim_on_sec * volumerate #int(round(stim_on_sec * volumerate))
    nframes_iti_pre = iti_pre * volumerate
    nframes_iti_post = iti_post*volumerate # int(round(iti_post * volumerate))
    nframes_iti_full = iti_full * volumerate #int(round(iti_full * volumerate))
    nframes_post_onset = (stim_on_sec + iti_post) * volumerate
except Exception as e:
    print "Problem calcuating nframes for trial epochs..."
    print e

#%%

# =================================================================================
# Create stimulusdict:
# =================================================================================

# stimdict[stim][currfile].trials
# stimdict[stim][currfile].frames
# stimdict[stim][currfile].frames_sec
# stimdict[stim][currfile].stim_on_idx

# 1. Create HDF5 file to store ALL trials in run with stimulus info and frame info:
run_grp = h5py.File(outfile, 'w')

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
            try:
                first_frame_on = frame_idxs.index(first_frame_on)
            except Exception as e:
                print "------------------------------------------------------------------"
                print "Found first frame on from serialdata file NOT found in frame_idxs."
                print "Trying 1 frame before / after..."
                try:
                    if first_frame_on+1 in frame_idxs:
                        first_frame_on = frame_idxs.index(first_frame_on+1)
                    else:
                        # Try first_frame_on-1 in frame_idxs:
                        first_frame_on = frame_idxs.index(first_frame_on-1)
                except Exception as e:
                    print "------------------------------------------------------------------"
                    print "NO match found for FIRST frame ON:", first_frame_on
                    print "File: %s, Trial %s, Stim: %s." % (currfile, currtrial_in_run, trialstim)
                    print e
                    print "------------------------------------------------------------------"
                    no_frame_match = True
                if no_frame_match is True:
                    print "Aborting."
                    print "------------------------------------------------------------------"

        preframes = list(np.arange(int(first_frame_on - nframes_iti_pre), first_frame_on, 1))
        postframes = list(np.arange(int(first_frame_on + 1), int(round(first_frame_on + nframes_post_onset))))

        framenums = [preframes, [first_frame_on], postframes]
        framenums = reduce(operator.add, framenums)
        #print "POST FRAMES:", len(framenums)
        diffs = np.diff(framenums)
        consec = [i for i in np.diff(diffs) if not i==0]
        assert len(consec)==0, "Bad frame parsing in %s, %s, frames: %s " % (currtrial_in_run, trialstim, str(framenums))

        # Create dataset for current trial with frame indices:
        fridxs = run_grp.create_dataset(currtrial_in_run, np.array(framenums).shape, np.array(framenums).dtype)
        fridxs[...] = np.array(framenums)
        fridxs.attrs['trial_in_run'] = currtrial_in_run
        fridxs.attrs['trial_in_file'] = currtrial_in_file
        fridxs.attrs['aux_file_idx'] = tiffnum
        fridxs.attrs['stim_on_idxs'] = first_frame_on
        fridxs.attrs['stim_dur_sec'] = stim_on_sec
        fridxs.attrs['iti_dur_sec'] = iti_full
        fridxs.attrs['baseline_dur_sec'] = iti_pre

run_grp.close()

# Rename FRAME file with hash:
frame_file_hashid = hash_file(outfile)
new_filename = os.path.splitext(outfile)[0] + frame_file_hashid + os.path.splitext(outfile)[1]
os.rename(outfile, new_filename)

print "Finished assigning frame idxs across all tiffs to trials in run %s." % run
print "Saved frame file to:", new_filename

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

