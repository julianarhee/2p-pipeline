#!/usr/bin/env python2
import os
import json
import re
import scipy.io as spio
import numpy as np
from json_tricks.np import dump, dumps, load, loads
import pandas as pd
import pymworks
import cPickle as pkl
from collections import Counter

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]

import optparse

parser = optparse.OptionParser()

parser.add_option('-R', '--root', action='store', dest='rootdir', default='/nas/volume1/2photon/data', help='data root dir (root project dir containing all animalids) [default: /nas/volume1/2photon/data, /n/coxfs01/2pdata if --slurm]')
parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')

# Set specific session/run for current animal:
parser.add_option('-S', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID')
parser.add_option('-A', '--acq', action='store', dest='acquisition', default='FOV1', help="acquisition folder (ex: 'FOV1_zoom3x') [default: FOV1]")
parser.add_option('-r', '--run', action='store', dest='run', default='', help="name of run dir containing tiffs to be processed (ex: gratings_phasemod_run1)")

parser.add_option('--slurm', action='store_true', dest='slurm', default=False, help="set if running as SLURM job on Odyssey")


parser.add_option('--stim', action="store",
                  dest="stimtype", default="grating", help="stimulus type [options: grating, image, bar].")
parser.add_option('--phasemod', action="store_true",
                  dest="phasemod", default=False, help="include if stimulus mod (phase-modulation).")
parser.add_option('-t', '--triggervar', action="store",
                  dest="frametrigger_varname", default='frame_trigger', help="Temp way of dealing with multiple trigger variable names [default: frame_trigger]")


(options, args) = parser.parse_args()

trigger_varname = options.frametrigger_varname

# Set USER INPUT options:
rootdir = options.rootdir
animalid = options.animalid
session = options.session
acquisition = options.acquisition
run = options.run

#%%

run_dir = os.path.join(rootdir, animalid, session, acquisition, run)

# # ================================================================================
# # frame info:
# # ================================================================================
# first_frame_on = 50
# stim_on_sec = 0.5
# iti = 1.
# vols_per_trial = 15
# same_order = True
# # =================================================================================

# Load reference info:
runinfo_path = os.path.join(run_dir, '%s.json' % run)

with open(runinfo_path, 'r') as fr:
    runinfo = json.load(fr)
nfiles = runinfo['ntiffs']
file_names = sorted(['File%03d' % int(f+1) for f in range(nfiles)], key=natural_keys)



# Get PARADIGM INFO:
paradigm_rawdir = os.path.join(run_dir, runinfo['rawtiff_dir'], 'paradigm_files')
paradigm_outdir = os.path.join(run_dir, 'paradigm')

# Get SERIAL data:
serialdata_fns = sorted([s for s in os.listdir(paradigm_rawdir) if s.endswith('txt') if 'serial' in s], key=natural_keys)
print "Found %02d serial-data files, and %i TIFFs." % (len(serialdata_fns), nfiles)

# Load MW info:
trialinfo_jsons = sorted([j for j in os.listdir(paradigm_outdir) if j.endswith('json') and 'trial_info' in j], key=natural_keys)
print "Found %02d MW files, and %02d ARD files." % (len(trialinfo_jsons), len(serialdata_fns))

trialdict_by_file = dict()
for fid,fn in enumerate(sorted(serialdata_fns, key=natural_keys)):

    framerate = float(runinfo['frame_rate'])

    currfile = "File%03d" % int(fid+1)

    print "================================="
    print "Processing files:"
    print "MW: ", trialinfo_jsons[fid]
    print "ARD: ", serialdata_fns[fid]
    print "---------------------------------"

    ### LOAD MW DATA.
    with open(os.path.join(paradigm_outdir, trialinfo_jsons[fid]), 'r') as f:
        trials = json.load(f)

    ### LOAD SERIAL DATA.
    ardata = pd.read_csv(os.path.join(paradigm_rawdir, serialdata_fns[fid]), sep='\t')
    print ardata.columns

    frame_triggers = ardata[' frame_trigger']
    arduino_time = ardata[' relative_arduino_time']
    bitcodes = ardata[' pixel_clock']

    frame_on_idxs = [idx+1 for idx,diff in enumerate(np.diff(frame_triggers)) if diff==1]
    #print len(frame_on_idxs)
    frame_on_idxs.append(0)
    frame_on_idxs = sorted(frame_on_idxs)
    print "Found %i frame-triggers:" % len(frame_on_idxs)


    ### Get bitcodes for each frame:
    frame_bitcodes = dict()
    for idx,frameidx in enumerate(frame_on_idxs):
        #framenum = 'frame'+str(idx)
        if idx==len(frame_on_idxs)-1:
            bcodes = bitcodes[frameidx:]
        else:
            bcodes = bitcodes[frameidx:frame_on_idxs[idx+1]]
        frame_bitcodes[idx] = bcodes

    ### Get first frame of trial start:
    modes_by_frame = dict()
    for frame in frame_bitcodes.keys():
        bitcode_counts = Counter(frame_bitcodes[frame])
        modes_by_frame[frame] = bitcode_counts.most_common(1)[0][0]

    # Take the 2nd frame that has the first-stim value (in case bitcode of Image on Trial1 is 0):
    if 'grating' in trials['1']['stimuli']['type']:
        first_stim_frame = [k for k in sorted(modes_by_frame.keys()) if modes_by_frame[k]>0][0]
    else:
        first_stim_frame = [k for k in sorted(modes_by_frame.keys()) if modes_by_frame[k]>0][1] #[0]

    #%%Get all bitcodes and corresonding frame-numbers for each trial:
    trialdict = dict()
    allframes = sorted(frame_bitcodes.keys()) #, key=natural_keys)
    curr_frames = sorted(allframes[first_stim_frame+1:]) #, key=natural_keys)
    first_frame = first_stim_frame

    for trial in sorted(trials.keys(), key=natural_keys): #sorted(trials.keys, key=natural_keys):
        #print trial
        trialdict[trial] = dict()
        trialdict[trial]['name'] = trials[trial]['stimuli']['stimulus']
        trialdict[trial]['stim_dur_ms'] = trials[trial]['stim_off_times'] - trials[trial]['stim_on_times']
        trialdict[trial]['iti_dur_ms'] = trials[trial]['iti_duration']

        if int(trial)>1:
        	    # Skip a good number of frames from the last "found" index of previous trial.
        	    # Since ITI is long (relative to framerate), this is safe to do. Avoids possibility that
        	    # first bitcode of trial N happened to be last stimulus bitcode of trial N-1
        	    nframes_to_skip = int(((trials[trial]['iti_duration']/1E3) * framerate) - 5)
        	    print 'skipping iti...', nframes_to_skip
        	    curr_frames = allframes[first_frame+nframes_to_skip:]

        first_found_frame = []
        minframes = 5
        for bitcode in trials[trial]['all_bitcodes']:
            looking = True
            while looking is True:
                for frame in sorted(curr_frames):
                    tmp_frames = [i for i in frame_bitcodes[frame] if i==bitcode]
                    consecutives = [i for i in np.diff(tmp_frames) if i==0]

                    if frame>1:
                        tmp_frames_pre = [i for i in frame_bitcodes[int(frame)-1] if i==bitcode]
                        consecutives_pre = [i for i in np.diff(tmp_frames_pre) if i==0]

                    if len(trials[trial]['all_bitcodes'])<3:
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
        print (first_found_frame[-1][1] - first_found_frame[0][1])/framerate, '[Trial %i]' % int(trial)

        trialdict[trial]['stim_on_idx'] = first_found_frame[0][1]
        trialdict[trial]['stim_off_idx'] = first_found_frame[-1][1]


    #%% Get stimulus list:
    if fid==0:
        stimlist = set([trialdict[trial]['name'] for trial in trialdict.keys()])
        stimlist = sorted(stimlist, key=natural_keys)
        print stimlist

    for trial in sorted(trialdict.keys(), key=natural_keys): #sorted(trials.keys, key=natural_keys):
        stimid = stimlist.index(trialdict[trial]['name'])
        trialdict[trial]['stimid'] = stimid #+ 1

    stimorder = [trialdict[trial]['stimid'] for trial in sorted(trialdict.keys(), key=natural_keys)]
    with open(os.path.join(paradigm_outdir, 'stimorder_%s.txt' % currfile),'w') as f:
        f.write('\n'.join([str(n) for n in stimorder])+'\n')

    trialdict_by_file[currfile] = trialdict

with open(os.path.join(paradigm_outdir, 'parsed_trials.pkl'), 'wb') as f:
    pkl.dump(trialdict_by_file, f, protocol=pkl.HIGHEST_PROTOCOL)
print "PARSED TRIALS saved to:", os.path.join(paradigm_outdir, 'parsed_trials.pkl')
print trialdict_by_file.keys()
# pkl.dump(stimdict, f, protocol=pkl.HIGHEST_PROTOCOL) #, f, indent=4)





