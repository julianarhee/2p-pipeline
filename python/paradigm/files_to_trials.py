#!/usr/bin/env python2
import os
import json
import re
import scipy.io as spio
import numpy as np
from json_tricks.np import dump, dumps, load, loads
from mat2py import loadmat
import cPickle as pkl
import scipy.io

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]
# 
# class StimInfo:
#     def _init_(self, stimid=None, trials=None, frames=None, frames_sec=None, stim_on_idx=None):
#         self.stimid = stimid #''
#         self.trials = trials # []
#         self.frames = frames # []
#         self.frames_sec = frames_sec # []
#         self.stim_on_idx = stim_on_idx #[]
# 

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
 
import optparse

parser = optparse.OptionParser()
parser.add_option('-S', '--source', action='store', dest='source', default='/nas/volume1/2photon/projects', help='source dir (root project dir containing all expts) [default: /nas/volume1/2photon/projects]')
parser.add_option('-E', '--experiment', action='store', dest='experiment', default='', help='experiment type (parent of session dir)') 
parser.add_option('-s', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID') 
parser.add_option('-A', '--acq', action='store', dest='acquisition', default='', help="acquisition folder (ex: 'FOV1_zoom3x')")
parser.add_option('-f', '--functional', action='store', dest='functional_dir', default='functional', help="folder containing functional TIFFs. [default: 'functional']")

parser.add_option('-r', '--roi', action="store",
                  dest="roi_method", default='blobs_DoG', help="roi method [default: 'blobsDoG]")

parser.add_option('-O', '--stimon', action="store",
                  dest="stim_on_sec", default='', help="Time (s) stimulus ON.")


(options, args) = parser.parse_args() 

source = options.source #'/nas/volume1/2photon/projects'
experiment = options.experiment #'scenes' #'gratings_phaseMod' #'retino_bar' #'gratings_phaseMod'
session = options.session #'20171003_JW016' #'20170927_CE059' #'20170902_CE054' #'20170825_CE055'
acquisition = options.acquisition #'FOV1' #'FOV1_zoom3x' #'FOV1_zoom3x_run2' #'FOV1_planar'
functional_dir = options.functional_dir #'functional' #'functional_subset'

roi_method = options.roi_method

stim_on_sec = float(options.stim_on_sec) #2. # 0.5

# source = '/nas/volume1/2photon/projects'
# # experiment = 'scenes'
# # session = '20171003_JW016'
# # acquisition = 'FOV1'
# # functional_dir = 'functional'

# experiment = 'gratings_phaseMod'
# session = '20171009_CE059'
# acquisition = 'FOV1_zoom3x'
# functional_dir = 'functional'
 
# ================================================================================
# frame info:
# ================================================================================
#first_frame_on = 50
#stim_on_sec = 2. #0.5
#iti = 1.
#vols_per_trial = 15
#same_order = True
# =================================================================================


acquisition_dir = os.path.join(source, experiment, session, acquisition)
figdir = os.path.join(acquisition_dir, 'example_figures')


# Load reference info:
ref_json = 'reference_%s.json' % functional_dir 
with open(os.path.join(acquisition_dir, ref_json), 'r') as fr:
    ref = json.load(fr)

# =====================================================
# Set ROI method and Trace method:
# =====================================================
#curr_roi_method = 'blobs_DoG' #ref['roi_id'] #'blobs_DoG'
#curr_trace_method = ref['trace_id'] #'blobs_DoG'
trace_dir = os.path.join(ref['trace_dir'], roi_method)
#trace_dir = ref['trace_dir']
# =====================================================

# Create parsed-trials dir with default format:
parsed_traces_dir = os.path.join(trace_dir, 'Parsed')
if not os.path.exists(parsed_traces_dir):
    os.mkdir(parsed_traces_dir)


# Load SI meta data:
si_basepath = ref['raw_simeta_path'][0:-4]
simeta_json_path = '%s.json' % si_basepath
with open(simeta_json_path, 'r') as fs:
    simeta = json.load(fs)

file_names = sorted([k for k in simeta.keys() if 'File' in k], key=natural_keys)
nfiles = len(file_names)


# Get masks for each slice: 
roi_methods_dir = os.path.join(acquisition_dir, 'ROIs')
roiparams = loadmat(os.path.join(roi_methods_dir, roi_method, 'roiparams.mat'))
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
print "STIMDICT: ", sorted(stimdict.keys(), key=natural_keys)


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
    for stim in sorted(stimdict.keys(), key=natural_keys):
        #stimtraces[stim] = dict()
        repidx = 0
        curr_traces_allrois = []
        curr_frames_allrois = []
        stim_on_frames = []
        for fi,currfile in enumerate(sorted(file_names, key=natural_keys)):
#            nframes = int(simeta[currfile]['SI']['hFastZ']['numVolumes'])
#            framerate = float(simeta[currfile]['SI']['hRoiManager']['scanFrameRate'])
            volumerate = float(simeta[currfile]['SI']['hRoiManager']['scanVolumeRate'])
#            frames_tsecs = np.arange(0, nframes)*(1/volumerate)

#            nframes_on = stim_on_sec * volumerate
#            nframes_off = vols_per_trial - nframes_on
#            frames_iti = round(iti * volumerate) 

            curr_ntrials = len(stimdict[stim][currfile].frames)
            currtraces = tracestruct['file'][fi].tracematDC
            for currtrial_idx in range(curr_ntrials):
                currtrial_frames = stimdict[stim][currfile].frames[currtrial_idx]
                #print len(currtrial_frames)
   
                # .T to make rows = rois, cols = frames 
                nframes = currtraces.shape[0]
                nrois = currtraces.shape[1] 
                #print currtraces[currtrial_frames, :].shape
                curr_traces_allrois.append(currtraces[currtrial_frames, :])
                curr_frames_allrois.append(currtrial_frames)
                repidx += 1
                
                curr_frame_onset = stimdict[stim][currfile].stim_on_idx[currtrial_idx]
                stim_on_frames.append([curr_frame_onset, curr_frame_onset + (stim_on_sec*volumerate)-1])
                
        check_stimname = list(set(stimdict[stim][currfile].stimid))
        if len(check_stimname)>1:
            print "******************************"
            print "Bad Stim to Trial parsing!."
            print "------------------------------"
            print check_stimname
            print "STIM:", stim, "File:", currfile
            print "------------------------------"
            print "Check extract_acquisition_events.py and create_stimdict.py"
            print "******************************"
        else:
            stimname = check_stimname[0]
        
        stimtraces[stim]['name'] = stimname
        stimtraces[stim]['traces'] = np.asarray(curr_traces_allrois)
    	stimtraces[stim]['frames_stim_on'] = stim_on_frames 
        stimtraces[stim]['frames'] = np.asarray(curr_frames_allrois)
        stimtraces[stim]['ntrials'] = stim_ntrials[stim]
        stimtraces[stim]['nrois'] = nrois


    curr_stimtraces_json = 'stimtraces_%s.json' % currslice
    print curr_stimtraces_json
    with open(os.path.join(parsed_traces_dir, curr_stimtraces_json), 'w') as f:
        dump(stimtraces, f, indent=4)


    curr_stimtraces_pkl = 'stimtraces_%s.pkl' % currslice
    print curr_stimtraces_pkl
    with open(os.path.join(parsed_traces_dir, curr_stimtraces_pkl), 'wb') as f:
        pkl.dump(stimtraces, f, protocol=pkl.HIGHEST_PROTOCOL)


    # save as .mat:
    curr_stimtraces_mat = 'stimtraces_%s.mat' % currslice
    # scipy.io.savemat(os.path.join(source_dir, condition, tif_fn), mdict=pydict)
    stimtraces_mat = dict()
    for stim in sorted(stimdict.keys(), key=natural_keys):
        currstim = "stim%02d" % int(stim)
        print currstim
        stimtraces_mat[currstim] = stimtraces[stim]

    scipy.io.savemat(os.path.join(parsed_traces_dir, curr_stimtraces_mat), mdict=stimtraces_mat)
    print os.path.join(parsed_traces_dir, curr_stimtraces_mat)
  
     
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
# for slice_idx,currslice in enumerate(sorted(stimtraces_all_slices.keys(), key=natural_keys)):
#     traces_by_stim = stimtraces_all_slices[currslice] #[currstim]['traces']
#     nrois = traces_by_stim['1']['nrois']
#     traces_by_roi = dict((str(roi), dict()) for roi in range(nrois))
# 
#     for stim in sorted(traces_by_stim.keys(), key=natural_keys):
#         print "STIM:", stim
#         for roi in range(nrois):
#             print "ROI:", roi
#             roiname = str(roi)
#             print traces_by_stim[stim]['traces'].shape
#             raw = traces_by_stim[stim]['traces'][:,:,roi]
#             ntrials = raw.shape[0]
#             nframes_in_trial = raw.shape[1]
#             curr_dfs = np.empty((ntrials, nframes_in_trial))
#             for trial in range(ntrials):
# 		frame_on = traces_by_stim[stim]['frames_stim_on'][trial][0]
#                 baseline = np.mean(raw[trial, 0:frame_on])
#                 df = (raw[trial, :] - baseline) / baseline
# 		curr_dfs[trial, :] = df
#               
#             traces_by_roi[roiname][stim] = curr_dfs
# 
#     curr_roitraces_json = 'roitraces_%s.json' % currslice
#     print curr_roitraces_json
#     with open(os.path.join(parsed_traces_dir, curr_roitraces_json), 'w') as f:
#         dump(traces_by_roi, f, indent=4)
# 
