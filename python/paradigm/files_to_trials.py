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
    def _init_(self, stimid=None, trials=None, frames=None, frames_sec=None, stim_on_idx=None):
        self.stimid = stimid #''
        self.trials = trials # []
        self.frames = frames # []
        self.frames_sec = frames_sec # []
        self.stim_on_idx = stim_on_idx #[]


class StimInfo:
    def _init_(self):
        self.stimid = ''
        self.trials = []
        self.frames = []
        self.frames_sec = []
        self.stim_on_idx = []

def serialize_json(instance=None, path=None):
    dt = {}
    dt.update(vars(instance))

#     with open(path, "w") as file:
#         json.dump(dt, file)
# 

# def deserialize_json(cls=None, data=None):
#     print cls
#     instance = object.__new__(cls)
# 
#     for key, value in data.items():
#         setattr(instance, key, value)
# 
#     return instance
# 
# 
# def deserialize_json(cls=None, path=None):
# 
#     def read_json(_path):
#         with open(_path, "r") as file:
#             return json.load(file)
# 
#     data = read_json(path)
# 
#     instance = object.__new__(cls)
# 
#     for key, value in data.items():
#         setattr(instance, key, value)
# 
#     return instance
 
source = '/nas/volume1/2photon/projects'
experiment = 'scenes'
session = '20171003_JW016'
acquisition = 'FOV1'
functional_dir = 'functional'

acquisition_dir = os.path.join(source, experiment, session, acquisition)
figdir = os.path.join(acquisition_dir, 'example_figures')


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


# Get masks for each slice: 
roi_methods_dir = os.path.join(acquisition_dir, 'ROIs')
roiparams = loadmat(os.path.join(roi_methods_dir, curr_roi_method, 'roiparams.mat'))
maskpaths = roiparams['roiparams']['maskpaths']
if not isinstance(maskpaths, list):
    maskpaths = [maskpaths]

masks = dict(("Slice%02d" % int(slice_idx+1), dict()) for slice_idx in range(len(maskpaths)))
for slice_idx,maskpath in enumerate(sorted(maskpaths, key=natural_keys)):
    slice_name = "Slice%02d" % int(slice_idx+1)
    print "Loading masks: %s..." % slice_name 
    currmasks = loadmat(maskpath); currmasks = currmasks['masks']
    masks[slice_name]['nrois'] =  currmasks.shape[2]
    masks[slice_name]['masks'] = currmasks


# Get PARADIGM INFO:
path_to_functional = os.path.join(acquisition_dir, functional_dir)
paradigm_dir = 'paradigm_files'
path_to_paradigm_files = os.path.join(path_to_functional, paradigm_dir)

# Load stimulus dict:
stimdict_fn = 'stimdict.pkl'
with open(os.path.join(path_to_paradigm_files, stimdict_fn), 'r') as f:
     stimdict = pkl.load(f) #json.load(f)
 

# Split all traces by stimulus-ID:
# ----------------------------------------------------------------------------
stim_ntrials = dict()
for stim in stimdict.keys():
    stim_ntrials[stim] = 0
    for fi in stimdict[stim].keys():
        stim_ntrials[stim] += len(stimdict[stim][fi].trials)

# Load trace structs:
curr_tracestruct_fns = os.listdir(trace_dir)
trace_fns_by_slice = sorted([t for t in curr_tracestruct_fns if 'traces_Slice' in t], key=natural_keys)
#trace_fns_by_slice = sorted(ref['trace_structs'], key=natural_keys)

#traces_by_stim = dict((stim, dict()) for stim in stimdict.keys())
#frames_stim_on = dict((stim, dict()) for stim in stimdict.keys())
stimtraces_all_slices = dict()

for slice_idx,trace_fn in enumerate(sorted(trace_fns_by_slice, key=natural_keys)):

    currslice = "Slice%02d" % int(slice_idx+1)
    stimtraces = dict((stim, dict()) for stim in stimdict.keys())

    tracestruct = loadmat(os.path.join(trace_dir, trace_fn))
    
    # To look at all traces for ROI 3 for stimulus 1:
    # traces_by_stim['1']['Slice01'][:,roi,:]
    for stim in stimdict.keys():
        #stimtraces[stim] = dict()
        repidx = 0
        curr_traces_allrois = []
        stim_on_frames = []
        for fi,currfile in enumerate(sorted(file_names, key=natural_keys)):
            curr_ntrials = len(stimdict[stim][currfile].frames)
            currtraces = tracestruct['file'][fi].tracematDC
            for currtrial_idx in range(curr_ntrials):
                currtrial_frames = stimdict[stim][currfile].frames[currtrial_idx]
   
                # .T to make rows = rois, cols = frames 
                nframes = currtraces.shape[0]
                nrois = currtraces.shape[1] 
                curr_traces_allrois.append(currtraces[currtrial_frames, :])
                repidx += 1
                
                curr_frame_onset = stimdict[stim][currfile].stim_on_idx[currtrial_idx]
                stim_on_frames.append([curr_frame_onset, curr_frame_onset + stim_on_sec*volumerate])

        stimtraces[stim]['traces'] = np.asarray(curr_traces_allrois)
	stimtraces[stim]['frames_stim_on'] = stim_on_frames 
        stimtraces[stim]['ntrials'] = stim_ntrials[stim]
        stimtraces[stim]['nrois'] = nrois

    curr_stimtraces_json = 'stimtraces_%s.json' % currslice
    print curr_stimtraces_json
    with open(os.path.join(parsed_traces_dir, curr_stimtraces_json), 'w') as f:
        dump(stimtraces, f, indent=4)

    stimtraces_all_slices[currslice] = stimtraces

        #traces_by_stim[stim][currslice] = np.asarray(curr_traces_allrois)
        #frames_stim_on[stim][currslice] = stim_on_frames

# 
# def default(obj):
#     if isinstance(obj, np.ndarray):
#         return obj.tolist()
#     raise TypeError('Not serializable')
# 
# stimtraces_json = 'stimtraces.json'
# with open(os.path.join(path_to_paradigm_files, stimtraces_json), 'w') as f:
#     dump(stimtraces, f, indent=4)
#  
	
# nframes = traces.df_f.T.shape[1]
# nrois = traces.df_f.T.shape[0]
# print "N files:", nfiles
# print "N frames:", nframes
# print "N rois:", nrois
# 
#    

#raw = traces_by_stim[stim][:, roi, :]
#avg = np.mean(raw, axis=0)
for slice_idx,currslice in enumerate(sorted(stimtraces_all_slices.keys(), key=natural_keys)):
    traces_by_stim = stimtraces_all_slices[currslice] #[currstim]['traces']
    nrois = traces_by_stim['1']['nrois']
    traces_by_roi = dict((str(roi), dict()) for roi in range(nrois))

    for stim in sorted(traces_by_stim.keys(), key=natural_keys):
        for roi in range(nrois):
            roiname = str(roi)
            raw = traces_by_stim[stim]['traces'][:,:,roi]
            ntrials = raw.shape[0]
            nframes_in_trial = raw.shape[1]
            curr_dfs = np.empty((ntrials, nframes_in_trial))
            for trial in range(ntrials):
		frame_on = traces_by_stim[stim]['frames_stim_on'][trial][0]
                baseline = np.mean(raw[trial, 0:frame_on])
                df = (raw[trial, :] - baseline) / baseline
		curr_dfs[trial, :] = df
              
            traces_by_roi[roiname][stim] = curr_dfs

    curr_roitraces_json = 'roitraces_%s.json' % currslice
    print curr_roitraces_json
    with open(os.path.join(parsed_traces_dir, curr_roitraces_json), 'w') as f:
        dump(traces_by_roi, f, indent=4)

