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
parser.add_option('-S', '--source', action='store', dest='source', default='/nas/volume1/2photon/projects', help='source dir (root project dir containing all expts) [default: /nas/volume1/2photon/projects]')
parser.add_option('-E', '--experiment', action='store', dest='experiment', default='', help='experiment type (parent of session dir)') 
parser.add_option('-s', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID') 
parser.add_option('-A', '--acq', action='store', dest='acquisition', default='', help="acquisition folder (ex: 'FOV1_zoom3x')")
parser.add_option('-f', '--functional', action='store', dest='functional_dir', default='functional', help="folder containing functional TIFFs. [default: 'functional']")

(options, args) = parser.parse_args() 

# source = '/nas/volume1/2photon/projects'
# experiment = 'gratings_phaseMod'
# session = '20171009_CE059'
# acquisition = 'FOV1_zoom3x'
# functional_dir = 'functional'

source = options.source #'/nas/volume1/2photon/projects'
experiment = options.experiment #'scenes' #'gratings_phaseMod' #'retino_bar' #'gratings_phaseMod'
session = options.session #'20171003_JW016' #'20170927_CE059' #'20170902_CE054' #'20170825_CE055'
acquisition = options.acquisition #'FOV1' #'FOV1_zoom3x' #'FOV1_zoom3x_run2' #'FOV1_planar'
functional_dir = options.functional_dir #'functional' #'functional_subset'

acquisition_dir = os.path.join(source, experiment, session, acquisition)
figdir = os.path.join(acquisition_dir, 'example_figures')

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
ref_json = 'reference_%s.json' % functional_dir 
with open(os.path.join(acquisition_dir, ref_json), 'r') as fr:
    ref = json.load(fr)

# Load SI meta data:
si_basepath = ref['raw_simeta_path'][0:-4]
simeta_json_path = '%s.json' % si_basepath
with open(simeta_json_path, 'r') as fs:
    simeta = json.load(fs)

file_names = sorted([k for k in simeta.keys() if 'File' in k], key=natural_keys)
nfiles = len(file_names)


# Get PARADIGM INFO:
path_to_functional = os.path.join(acquisition_dir, functional_dir)
paradigm_dir = 'paradigm_files'
path_to_paradigm_files = os.path.join(path_to_functional, paradigm_dir)

# Get SERIAL data:
serialdata_fns = os.listdir(path_to_paradigm_files)
serialdata_fns = sorted([f for f in serialdata_fns if 'serial_data' in f])
print "Found %02d serial-data files, and %03d TIFFs." % (len(serialdata_fns), nfiles)

# Load MW info:
pydict_jsons = os.listdir(path_to_paradigm_files)
pydict_jsons = sorted([p for p in pydict_jsons if p.endswith('.json') and 'trial_info' in p], key=natural_keys)
print "Found %02d MW files, and %02d ARD files." % (len(pydict_jsons), len(serialdata_fns))

trialdict_by_file = dict()
for fid,fn in enumerate(sorted(serialdata_fns, key=natural_keys)):
    framerate = float(simeta['File001']['SI']['hRoiManager']['scanFrameRate'])

    currfile = "File%03d" % int(fid+1)
    
    print "================================="
    print "Processing files:"
    print "MW: ", pydict_jsons[fid]
    print "ARD: ", serialdata_fns[fid]
    print "================================="
    
    ### LOAD MW DATA.
    with open(os.path.join(path_to_paradigm_files, pydict_jsons[fid]), 'r') as f:
        trials = json.load(f)
        
    ### LOAD SERIAL DATA.
    ardata = pd.read_csv(os.path.join(path_to_paradigm_files, serialdata_fns[fid]), sep='\t')
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

    first_stim_frame = [k for k in sorted(modes_by_frame.keys()) if modes_by_frame[k]>0][0]

    ### Get all bitcodes and corresonding frame-numbers for each trial:
    trialdict = dict()
    allframes = sorted(frame_bitcodes.keys()) #, key=natural_keys)
    curr_frames = sorted(allframes[first_stim_frame+1:]) #, key=natural_keys)

    for trial in sorted(trials.keys(), key=natural_keys): #sorted(trials.keys, key=natural_keys):
        #print trial
        trialdict[trial] = dict()
        trialdict[trial]['name'] = trials[trial]['stimuli']['stimulus']
        trialdict[trial]['duration'] = trials[trial]['stim_off_times'] - trials[trial]['stim_on_times']

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
                
        if (first_found_frame[-1][1] - first_found_frame[0][1])/framerate > 2.5:
            print "Trial %i dur (s):" % int(trial)
            print (first_found_frame[-1][1] - first_found_frame[0][1])/framerate

        trialdict[trial]['stim_on_idx'] = first_found_frame[0][1]
        trialdict[trial]['stim_off_idx'] = first_found_frame[-1][1]
    

    ### Get stimulus list:
    if fid==0:
        stimlist = set([trialdict[trial]['name'] for trial in trialdict.keys()])
        stimlist = sorted(stimlist, key=natural_keys)
        print stimlist
    
    for trial in sorted(trialdict.keys(), key=natural_keys): #sorted(trials.keys, key=natural_keys):
        stimid = stimlist.index(trialdict[trial]['name'])
        trialdict[trial]['stimid'] = stimid + 1
    
    stimorder = [trialdict[trial]['stimid'] for trial in sorted(trialdict.keys(), key=natural_keys)]
    with open(os.path.join(path_to_paradigm_files, 'stimorder_%s.txt' % currfile),'w') as f:
        f.write('\n'.join([str(n) for n in stimorder])+'\n')
    
    trialdict_by_file[currfile] = trialdict
    
with open(os.path.join(path_to_paradigm_files, 'parsed_trials.pkl'), 'wb') as f:
    pkl.dump(trialdict_by_file, f, protocol=pkl.HIGHEST_PROTOCOL)
# pkl.dump(stimdict, f, protocol=pkl.HIGHEST_PROTOCOL) #, f, indent=4)


             

    
