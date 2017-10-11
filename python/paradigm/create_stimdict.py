#!/usr/bin/env python2
import os
import json
import re
import scipy.io as spio
import numpy as np
from json_tricks.np import dump, dumps, load, loads
from mat2py import loadmat
import cPickle as pkl

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]
 
class StimInfo:
    def __init__(self):
        self.stimid = ''
        self.trials = []
        self.frames = []
        self.frames_sec = []
        self.stim_on_idx = []

source = '/nas/volume1/2photon/projects'
experiment = 'scenes'
session = '20171003_JW016'
acquisition = 'FOV1'
functional_dir = 'functional'

acquisition_dir = os.path.join(source, experiment, session, acquisition)
figdir = os.path.join(acquisition_dir, 'example_figures')

# ================================================================================
# frame info:
# ================================================================================
first_frame_on = 50
stim_on_sec = 0.5
iti = 1.
vols_per_trial = 15
same_order = True
# =================================================================================

# Load reference info:
ref_json = 'reference_%s.json' % functional_dir 
with open(os.path.join(acquisition_dir, ref_json), 'r') as fr:
    ref = json.load(fr)

# =====================================================
# Set ROI method and Trace method:
# =====================================================
curr_roi_method = ref['roi_id'] #'blobs_DoG'
#curr_trace_method = ref['trace_id'] #'blobs_DoG'
trace_dir = os.path.join(ref['trace_dir'], curr_roi_method)
#trace_dir = ref['trace_dir']
# =====================================================

# Create parsed-trials dir with default format:
parsed_traces_dir = os.path.join(trace_dir, 'Parsed')
if not os.path.exists(parsed_traces_dir):
    os.mkdir(parsed_traces_dir)

# Get PARADIGM INFO:
path_to_functional = os.path.join(acquisition_dir, functional_dir)
paradigm_dir = 'paradigm_files'
path_to_paradigm_files = os.path.join(path_to_functional, paradigm_dir)


# Load SI meta data:
si_basepath = ref['raw_simeta_path'][0:-4]
simeta_json_path = '%s.json' % si_basepath
with open(simeta_json_path, 'r') as fs:
    simeta = json.load(fs)

file_names = sorted([k for k in simeta.keys() if 'File' in k], key=natural_keys)
nfiles = len(file_names)

# Get stim-order files:
stimorder_fns = os.listdir(path_to_paradigm_files)
stimorder_fns = sorted([f for f in stimorder_fns if 'stim_order' in f])
print "Found %i stim-order files, and %i TIFFs." % (len(stimorder_fns), nfiles)
if len(stimorder_fns) < nfiles:
    if same_order:
        # Same stimulus order for each file (complete set)
        stimorder_fns = np.tile(stimorder_fns, [nfiles,])


# Create stimulus-dict:
#stimdict_basename = 'stimdict'

def serialize_json(instance=None, path=None):
    dt = {}
    dt.update(vars(instance))
    return dt

stimdict = dict()
for fi in range(nfiles):
    currfile= "File%03d" % int(fi+1)
       
    nframes = int(simeta[currfile]['SI']['hFastZ']['numVolumes'])
    framerate = float(simeta[currfile]['SI']['hRoiManager']['scanFrameRate'])
    volumerate = float(simeta[currfile]['SI']['hRoiManager']['scanVolumeRate'])
    frames_tsecs = np.arange(0, nframes)*(1/volumerate)

    nframes_on = stim_on_sec * volumerate
    nframes_off = vols_per_trial - nframes_on
    frames_iti = round(iti * volumerate) 

    # Load stim-order:
    stim_fn = stimorder_fns[fi] #'stim_order.txt'
    with open(os.path.join(path_to_paradigm_files, stim_fn)) as f:
        stimorder = f.readlines()
    curr_stimorder = [l.strip() for l in stimorder]
    unique_stims = sorted(set(curr_stimorder), key=natural_keys)
    first_frame_on = 50 
    for trialnum,stim in enumerate(curr_stimorder):
        #print "Stim on frame:", first_frame_on
        if not stim in stimdict.keys():
            stimdict[stim] = dict()
        if not currfile in stimdict[stim].keys():
            stimdict[stim][currfile] = StimInfo()  #StimInfo()
        framenums = list(np.arange(int(first_frame_on-frames_iti), int(first_frame_on+(vols_per_trial))))
        frametimes = [frames_tsecs[f] for f in framenums]
        stimdict[stim][currfile].trials.append(trialnum)      
        stimdict[stim][currfile].frames.append(framenums)
        stimdict[stim][currfile].frames_sec.append(frametimes)
        stimdict[stim][currfile].stim_on_idx.append(framenums.index(first_frame_on))
        first_frame_on = first_frame_on + vols_per_trial

# Save to PKL:
curr_stimdict_pkl = 'stimdict.pkl' #% currfile # % currslice
print curr_stimdict_pkl
with open(os.path.join(path_to_paradigm_files, curr_stimdict_pkl), 'w') as f:
    pkl.dump(stimdict, f, protocol=pkl.HIGHEST_PROTOCOL) #, f, indent=4)

# Save to JSON:
for fi in range(nfiles):
    currfile = "File%03d" % int(fi+1)
    for stim in stimdict.keys():
        stimdict[stim][currfile] = serialize_json(stimdict[stim][currfile]) 

curr_stimdict_json = 'stimdict.json' #% currfile # % currslice
print curr_stimdict_json
with open(os.path.join(path_to_paradigm_files, curr_stimdict_json), 'w') as f:
    dump(stimdict, f, indent=4)


