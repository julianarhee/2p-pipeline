
import os
import json
import re
import scipy.io as spio
import numpy as np
from json_tricks.np import dump, dumps, load, loads

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)

    return _check_keys(data)


def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])

    return dict


def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        elif isinstance(elem,np.ndarray):
            dict[strg] = _tolist(elem)
        else:
            dict[strg] = elem

    return dict


def _tolist(ndarray):
    '''
    A recursive function which constructs lists from cellarrays 
    (which are loaded as numpy ndarrays), recursing into the elements
    if they contain matobjects.
    '''
    elem_list = []
    for sub_elem in ndarray:
        if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
            elem_list.append(_todict(sub_elem))
        elif isinstance(sub_elem,np.ndarray):
            elem_list.append(_tolist(sub_elem))
        else:
            elem_list.append(sub_elem)

    return elem_list


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]


class StimInfo:
    def _init_(self):
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

trial_dir = os.path.join(acquisition_dir, 'Trials')
if not os.path.exists(trial_dir):
    os.mkdir(trial_dir)


# Load reference info:
ref_json = 'reference_%s.json' % functional_dir 
with open(os.path.join(acquisition_dir, ref_json), 'r') as fr:
    ref = json.load(fr)

curr_roi_method = ref['roi_id'] #'blobs_DoG'
curr_trace_method = ref['trace_id'] #'blobs_DoG'

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

stiminfo_basename = 'stiminfo'

# ================================================================================
# frame info:
# ================================================================================
first_frame_on = 50
stim_on_sec = 0.5
iti = 1.
vols_per_trial = 15
# =================================================================================

# Load SI meta data:
si_basepath = ref['raw_simeta_path'][0:-4]
simeta_json_path = '%s.json' % si_basepath
with open(simeta_json_path, 'r') as fs:
    simeta = json.load(fs)

file_names = sorted([k for k in simeta.keys() if 'File' in k], key=natural_keys)
nfiles = len(file_names)

# Create stimulus-dict:
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
    stim_fn = 'stim_order.txt'
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
            stimdict[stim][currfile] = StimInfo()
            stimdict[stim][currfile].trials = []
            stimdict[stim][currfile].frames = []
            stimdict[stim][currfile].frames_sec = []
            stimdict[stim][currfile].stim_on_idx = []

        framenums = list(np.arange(int(first_frame_on-frames_iti), int(first_frame_on+(vols_per_trial))))
        frametimes = [frames_tsecs[f] for f in framenums]
        stimdict[stim][currfile].trials.append(trialnum)      
        stimdict[stim][currfile].frames.append(framenums)
        stimdict[stim][currfile].frames_sec.append(frametimes)
        stimdict[stim][currfile].stim_on_idx.append(framenums.index(first_frame_on))
        first_frame_on = first_frame_on + vols_per_trial


# Split all traces by stimulus-ID:
# ----------------------------------------------------------------------------
stim_ntrials = dict()
for stim in stimdict.keys():
    stim_ntrials[stim] = 0
    for fi in stimdict[stim].keys():
        stim_ntrials[stim] += len(stimdict[stim][fi].trials)

# Load trace structs:
trace_fns_by_slice = sorted(ref['trace_structs'], key=natural_keys)
#traces_by_stim = dict((stim, dict()) for stim in stimdict.keys())
#frames_stim_on = dict((stim, dict()) for stim in stimdict.keys())
stimtraces_all_slices = dict()

for slice_idx,trace_fn in enumerate(sorted(trace_fns_by_slice, key=natural_keys)):

    currslice = "Slice%02d" % int(slice_idx+1)
    stimtraces = dict((stim, dict()) for stim in stimdict.keys())

    tracestruct = loadmat(os.path.join(ref['trace_dir'], trace_fn))
    
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
    with open(os.path.join(trial_dir, curr_stimtraces_json), 'w') as f:
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
    with open(os.path.join(trial_dir, curr_roitraces_json), 'w') as f:
        dump(traces_by_roi, f, indent=4)

