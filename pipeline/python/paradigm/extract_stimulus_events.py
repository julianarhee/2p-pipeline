#!/usr/bin/env python2
import os
import json
import re
import hashlib
import optparse

import numpy as np
import pandas as pd
import cPickle as pkl
from collections import Counter
from pipeline.python.paradigm import process_mw_files as mw
from pipeline.python.utils import file_hash

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]

#%%

def extract_frames_to_trials(serialfn_path, mwtrial_path, framerate, verbose=False):

    trialevents = None

    ### LOAD MW DATA.
    with open(mwtrial_path, 'r') as f:
        mwtrials = json.load(f)

    ### LOAD SERIAL DATA.
    serialdata = pd.read_csv(serialfn_path, sep='\t')
    if verbose is True:
        print serialdata.columns

    ### Extract events from serialdata:
    frame_triggers = serialdata[' frame_trigger']
    bitcodes = serialdata[' pixel_clock']

    ### Find frame ON triggers (from NIDAQ-SI):
    frame_on_idxs = [idx+1 for idx,diff in enumerate(np.diff(frame_triggers)) if diff==1]
    frame_on_idxs.append(0)
    frame_on_idxs = sorted(frame_on_idxs)
    print "Found %i frame-triggers:" % len(frame_on_idxs)

    ### Get arduino-processed bitcodes for each frame:
    frame_bitcodes = dict()
    for idx,frameidx in enumerate(frame_on_idxs):
        #framenum = 'frame'+str(idx)
        if idx==len(frame_on_idxs)-1:
            bcodes = bitcodes[frameidx:]
        else:
            bcodes = bitcodes[frameidx:frame_on_idxs[idx+1]]
        frame_bitcodes[idx] = bcodes


    ### Find first frame of MW experiment start:
    modes_by_frame = dict()
    for frame in frame_bitcodes.keys():
        bitcode_counts = Counter(frame_bitcodes[frame])
        modes_by_frame[frame] = bitcode_counts.most_common(1)[0][0]

    # Take the 2nd frame that has the first-stim value (in case bitcode of Image on Trial1 is 0):
    trialnames = sorted(mwtrials.keys(), key=natural_keys)
    if 'grating' in mwtrials[trialnames[0]]['stimuli']['type']:
        first_stim_frame = [k for k in sorted(modes_by_frame.keys()) if modes_by_frame[k]>0][0]
    else:
        first_stim_frame = [k for k in sorted(modes_by_frame.keys()) if modes_by_frame[k]>0][1] #[0]


    ### Get all bitcodes and corresonding frame-numbers for each trial:
    trialevents = dict()
    allframes = sorted(frame_bitcodes.keys()) #, key=natural_keys)
    curr_frames = sorted(allframes[first_stim_frame+1:]) #, key=natural_keys)
    first_frame = first_stim_frame


    for tidx, trial in enumerate(sorted(mwtrials.keys(), key=natural_keys)):

        # Create hash of current MWTRIAL dict:
        mwtrial_hash = hashlib.sha1(json.dumps(mwtrials[trial], sort_keys=True)).hexdigest()


        #print trial
        trialevents[mwtrial_hash] = dict()
        #trialevents[trial]['mwtrial_hash'] = mwtrial_hash
        #trialevents[trial]['stiminfo'] = mwtrials[trial]['stimuli']
        trialevents[mwtrial_hash]['stim_dur_ms'] = mwtrials[trial]['stim_off_times'] - mwtrials[trial]['stim_on_times']
        #trialevents[trial]['iti_dur_ms'] = mwtrials[trial]['iti_duration']

        if int(tidx+1)>1:
        	    # Skip a good number of frames from the last "found" index of previous trial.
        	    # Since ITI is long (relative to framerate), this is safe to do. Avoids possibility that
        	    # first bitcode of trial N happened to be last stimulus bitcode of trial N-1
        	    nframes_to_skip = int(((mwtrials[trial]['iti_duration']/1E3) * framerate) - 5)
        	    #print 'skipping iti...', nframes_to_skip
        	    curr_frames = allframes[first_frame+nframes_to_skip:]

        first_found_frame = []
        minframes = 5
        for bitcode in mwtrials[trial]['all_bitcodes']:
            looking = True
            while looking is True:
                for frame in sorted(curr_frames):
                    tmp_frames = [i for i in frame_bitcodes[frame] if i==bitcode]
                    consecutives = [i for i in np.diff(tmp_frames) if i==0]

                    if frame>1:
                        tmp_frames_pre = [i for i in frame_bitcodes[int(frame)-1] if i==bitcode]
                        consecutives_pre = [i for i in np.diff(tmp_frames_pre) if i==0]

                    if len(mwtrials[trial]['all_bitcodes'])<3:
                    #Single-image (static images) will only have a single bitcode, plus ITI bitcode,
                        # Don't look before/after found-frame idx.
                        if len(consecutives)>=minframes:
                            first_frame = frame
                            looking = False

                    else:
                        if frame>1 and len(consecutives_pre)>=minframes:
                            if len(consecutives_pre) > len(consecutives):
                                first_frame = int(frame) - 1
                            elif len(consecutives)>=minframes:
                                first_frame = int(frame)
                            #print "found2...", bitcode, first_frame #len(curr_frames)
                            looking = False

                        elif len(consecutives)>=minframes:
                            first_frame = frame
                            #print "found...", bitcode, first_frame #len(curr_frames)
                            looking = False

                    if looking is False:
                        break

            first_found_frame.append((bitcode, first_frame)) #first_frame))
            curr_frames = allframes[first_frame+1:] #curr_frames[idx:] #curr_frames[first_frame:]

        #if (first_found_frame[-1][1] - first_found_frame[0][1])/framerate > 2.5:
        #print "Trial %i dur (s):" % int(trial)
        print (first_found_frame[-1][1] - first_found_frame[0][1])/framerate, '[%s]' % trial

        trialevents[mwtrial_hash]['stim_on_idx'] = first_found_frame[0][1]
        trialevents[mwtrial_hash]['stim_off_idx'] = first_found_frame[-1][1]
        trialevents[mwtrial_hash]['mw_trial'] = mwtrials[trial]

    return trialevents
#%%

parser = optparse.OptionParser()

parser.add_option('-R', '--root', action='store', dest='rootdir', default='/nas/volume1/2photon/data', help='data root dir (root project dir containing all animalids) [default: /nas/volume1/2photon/data, /n/coxfs01/2pdata if --slurm]')
parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')

# Set specific session/run for current animal:
parser.add_option('-S', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID')
parser.add_option('-A', '--acq', action='store', dest='acquisition', default='FOV1', help="acquisition folder (ex: 'FOV1_zoom3x') [default: FOV1]")
parser.add_option('-r', '--run', action='store', dest='run', default='', help="name of run dir containing tiffs to be processed (ex: gratings_phasemod_run1)")
parser.add_option('--slurm', action='store_true', dest='slurm', default=False, help="set if running as SLURM job on Odyssey")


parser.add_option('--retinobar', action="store_true",
                  dest="retionbar", default=False, help="Set flag if stimulus is moving-bar for retinotopy.")
parser.add_option('--phasemod', action="store_true",
                  dest="phasemod", default=False, help="Set flag if using dynamic, phase-modulated gratings.")
parser.add_option('-t', '--triggervar', action="store",
                  dest="frametrigger_varname", default='frame_trigger', help="Temp way of dealing with multiple trigger variable names [default: frame_trigger]")


(options, args) = parser.parse_args()

# Set USER INPUT options:
rootdir = options.rootdir
animalid = options.animalid
session = options.session
acquisition = options.acquisition
run = options.run
slurm = options.slurm

if slurm is True and 'coxfs01' not in rootdir:
    rootdir = '/n/coxfs01/julianarhee/testdata'

# MW specific options:
retinobar = options.retinobar
phasemod = options.phasemod
trigger_varname = options.frametrigger_varname

stimorder_files = True

#%%
# ================================================================================
# MW trial extraction:
# ================================================================================
mwopts = ['-R', rootdir, '-i', animalid, '-S', session, '-A', acquisition, '-r', run, '-t', trigger_varname]
if slurm is True:
    mwopts.extend(['--slurm'])
if retinobar is True:
    mwopts.extend(['--retinobar'])
if phasemod is True:
    mwopts.extend(['--phasemod'])

paradigm_outdir = mw.parse_mw_trials(mwopts)

#%%
if stimorder_files is True:
    mw.create_stimorder_files(paradigm_outdir)

#%%
# ================================================================================
# Load reference info:
# ================================================================================
run_dir = os.path.join(rootdir, animalid, session, acquisition, run)
runinfo_path = os.path.join(run_dir, '%s.json' % run)

with open(runinfo_path, 'r') as fr:
    runinfo = json.load(fr)
nfiles = runinfo['ntiffs']
file_names = sorted(['File%03d' % int(f+1) for f in range(nfiles)], key=natural_keys)

#%%

# Set outpath to save trial info file for whole run:
outdir = os.path.join(run_dir, 'paradigm')

#%%
# ================================================================================
# Get SERIAL data:
# ================================================================================
paradigm_rawdir = os.path.join(run_dir, runinfo['rawtiff_dir'], 'paradigm_files')
serialdata_fns = sorted([s for s in os.listdir(paradigm_rawdir) if s.endswith('txt') if 'serial' in s], key=natural_keys)
print "Found %02d serial-data files, and %i TIFFs." % (len(serialdata_fns), nfiles)

# Load MW info:
mwtrial_fns = sorted([j for j in os.listdir(paradigm_outdir) if j.endswith('json') and 'trials_' in j], key=natural_keys)
print "Found %02d MW files, and %02d ARD files." % (len(mwtrial_fns), len(serialdata_fns))


#%%
RUN = dict()
trialnum = 0
for fid,serialfn in enumerate(sorted(serialdata_fns, key=natural_keys)):

    framerate = float(runinfo['frame_rate'])

    currfile = "File%03d" % int(fid+1)

    print "================================="
    print "Processing files:"
    print "MW: ", mwtrial_fns[fid]
    print "ARD: ", serialdata_fns[fid]
    print "---------------------------------"

    mwtrial_path = os.path.join(paradigm_outdir, mwtrial_fns[fid])
    serialfn_path = os.path.join(paradigm_rawdir, serialfn)


    trialevents = extract_frames_to_trials(serialfn_path, mwtrial_path, framerate, verbose=False)
    sorted_trials_in_run = sorted(trialevents.keys(), key=lambda x: trialevents[x]['stim_on_idx'])
    sorted_stim_frames = [(trialevents[t]['stim_on_idx'], trialevents[t]['stim_off_idx']) for t in sorted_trials_in_run]

    for trialhash in sorted_trials_in_run:
        trialnum += 1
        trialname = 'trial%05d' % int(trialnum)

        RUN[trialname] = dict()
        RUN[trialname]['trial_hash'] = trialhash
        RUN[trialname]['aux_file_idx'] = fid
        RUN[trialname]['behavior_data_path'] = mwtrial_path
        RUN[trialname]['serial_data_path'] = serialfn_path

        RUN[trialname]['start_time_ms'] = trialevents[trialhash]['mw_trial']['end_time_ms']
        RUN[trialname]['end_time_ms'] = trialevents[trialhash]['mw_trial']['start_time_ms']
        RUN[trialname]['stim_dur_ms'] = trialevents[trialhash]['mw_trial']['stim_off_times']\
                                                - trialevents[trialhash]['mw_trial']['stim_on_times']
        RUN[trialname]['iti_dur_ms'] = trialevents[trialhash]['mw_trial']['iti_duration']
        RUN[trialname]['stimuli'] = trialevents[trialhash]['mw_trial']['stimuli']

        RUN[trialname]['frame_stim_on'] = trialevents[trialhash]['stim_on_idx']
        RUN[trialname]['frame_stim_off'] = trialevents[trialhash]['stim_off_idx']
        RUN[trialname]['trial_in_run'] = trialnum


run_trial_hash = hashlib.sha1(json.dumps(RUN, indent=4, sort_keys=True)).hexdigest()[0:6]
with open(os.path.join(outdir, 'trials_%s.json' % run_trial_hash), 'w') as f:
    json.dump(RUN, f, sort_keys=True, indent=4)



