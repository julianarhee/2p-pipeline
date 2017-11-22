
# coding: utf-8

# In[79]:


import os
import json
import re
import scipy.io as spio
import numpy as np
from bokeh.plotting import figure
import tifffile as tf
import seaborn as sns
# %matplotlib notebook
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import skimage.color
from json_tricks.np import dump, dumps, load, loads
from mat2py import loadmat
from skimage import color
import cPickle as pkl
from  matplotlib.ticker import FuncFormatter
from matplotlib.ticker import MaxNLocator

def get_axis_limits(ax, scale=(0.9, 0.9)):
    return ax.get_xlim()[1]*scale[0], ax.get_ylim()[1]*scale[1]


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]
import os
import json
import re
import scipy.io as spio
import numpy as np
from bokeh.plotting import figure
import tifffile as tf
import seaborn as sns
# %matplotlib notebook
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import skimage.color
from json_tricks.np import dump, dumps, load, loads
from mat2py import loadmat
from skimage import color
import cPickle as pkl
from  matplotlib.ticker import FuncFormatter
from matplotlib.ticker import MaxNLocator

def get_axis_limits(ax, scale=(0.9, 0.9)):
    return ax.get_xlim()[1]*scale[0], ax.get_ylim()[1]*scale[1]


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]

import optparse

parser = optparse.OptionParser()
parser.add_option('-S', '--source', action='store', dest='source', default='/nas/volume1/2photon/projects', help='source dir (root project dir containing all expts) [default: /nas/volume1/2photon/projects]')
parser.add_option('-E', '--experiment', action='store', dest='experiment', default='', help='experiment type (parent of session dir)') 
parser.add_option('-s', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID') 
parser.add_option('-A', '--acq', action='store', dest='acquisition', default='FOV1', help="acquisition folder [default: FOV1]")
parser.add_option('-f', '--functional', action='store', dest='functional', default='functional', help="folder containing functional TIFFs. [default: 'functional']")

# Analysis-specific params:
parser.add_option('-I', '--id', action="store",
                  dest="analysis_id", default='', help="analysis_id (includes mcparams, roiparams, and ch). see <acquisition_dir>/analysis_record.json for help.")

parser.add_option('-z', '--slice', action="store",
                  dest="sliceidx", default=1, help="Slice number look at (1-index) [default: 1]")


parser.add_option('--processed', action="store_true",
                  dest="dont_use_raw", default=False, help="Flag to use processed traces instead of raw.")

parser.add_option('--df', action="store_true",
                  dest="use_df", default=False, help="Flag to use NMF-extracted df traces.")



(options, args) = parser.parse_args() 


# In[96]:


analysis_id = options.analysis_id #'analysis02'

source = options.source #'/nas/volume1/2photon/projects'
experiment = options.experiment #'gratings_phaseMod' #'gratings_phaseMod' #'retino_bar' #'gratings_phaseMod'
session = options.session #'20171025_CE062' #'20170927_CE059' #'20170902_CE054' #'20170825_CE055'
acquisition = options.acquisition #'FOV1' #'FOV1_zoom3x' #'FOV1_zoom3x_run2' #'FOV1_planar'
functional_dir = options.functional #'functional' #'functional_subset'

#iti_pre = float(options.iti_pre) # 1.0 #float(options.iti_pre)

curr_slice_idx = int(options.sliceidx) #int(1) #int(options.sliceidx)



# In[97]:


# ---------------------------------------------------------------------------------
# PLOTTING parameters:
# ---------------------------------------------------------------------------------
dont_use_raw = options.dont_use_raw #True #options.dont_use_raw
use_df = options.use_df


# ---------------------------------------------------------------------
# Get relevant stucts:
# ---------------------------------------------------------------------
acquisition_dir = os.path.join(source, experiment, session, acquisition)


# Load reference info:
ref_json = 'reference_%s.json' % functional_dir 
with open(os.path.join(acquisition_dir, ref_json), 'r') as fr:
    ref = json.load(fr)
    
# Get ROIPARAMS:
roi_dir = os.path.join(ref['roi_dir'], ref['roi_id'][analysis_id]) #, 'ROIs')
roiparams = loadmat(os.path.join(roi_dir, 'roiparams.mat'))
if 'roiparams' in roiparams.keys():
    roiparams = roiparams['roiparams']
maskpaths = roiparams['maskpaths']
print maskpaths
if not isinstance(maskpaths, list) and len(maskpaths)==1:
    maskpaths = [maskpaths] #[str(i) for i in maskpaths]
if isinstance(maskpaths, unicode):
    maskpaths = [maskpaths]

nrois = roiparams['nrois']
if isinstance(nrois, float) or isinstance(nrois, int):
    nrois = int(nrois)
else:
    nrois = int(roiparams['nrois'][curr_slice_idx])

# Check slices to see if maskpaths exist for all slices, or just a subset:
if 'sourceslices' in roiparams.keys():
    slices = roiparams['sourceslices']
else:
    slices = np.arange(1, len(maskpaths)+1) #range(len(maskpaths))
print "Found masks for slices:", slices
if isinstance(slices, int):
    slices = [slices]
if not isinstance(slices, list): # i.e., only 1 slice
    slices = [int(i) for i in slices]
print "SLICES:", slices


# Get selected slice for current slice:
if isinstance(ref['slices'], int):
    all_slices = ['Slice%02d' % int(i+1) for i in range(ref['slices'])]
else:
    all_slices = ['Slice%02d' % int(i+1) for i in range(len(ref['slices']))]

curr_slice_idx = [i for i in sorted(slices)].index(curr_slice_idx)

slice_names = sorted(['Slice%02d' % int(i) for i in sorted(slices)], key=natural_keys)
print "SLICE NAMES:", slice_names
curr_slice_name = "Slice%02d" % slices[curr_slice_idx]


# In[101]:

if use_df is True:
    roi_trace_type = 'df'
else:
    if dont_use_raw is True:
        roi_trace_type = 'processed' #figdir = os.path.join(figbase, 'rois_processed')
    else:
        roi_trace_type = 'raw'

# In[102]:



# Get PARADIGM INFO:
path_to_functional = os.path.join(acquisition_dir, functional_dir)
paradigm_dir = 'paradigm_files'
path_to_paradigm_files = os.path.join(path_to_functional, paradigm_dir)
#path_to_trace_structs = os.path.join(acquisition_dir, 'Traces', roi_method, 'Parsed')
path_to_trace_structs = os.path.join(acquisition_dir, 'Traces', ref['trace_id'][analysis_id], 'Parsed')
   
roi_struct_dir = os.path.join(path_to_trace_structs, 'rois')
if not os.path.exists(roi_struct_dir):
    os.mkdir(roi_struct_dir)
 
# Load stim trace structs:
print "Loading parsed traces..."
signal_channel = ref['signal_channel'][analysis_id] #int(options.selected_channel)

currchannel = "Channel%02d" % int(signal_channel)
currslice = "Slice%02d" % slices[curr_slice_idx] # curr_slice_idx
stimtrace_fns = os.listdir(path_to_trace_structs)
stimtrace_fn = "%s_stimtraces_%s_%s.pkl" % (analysis_id, currslice, currchannel)

if not stimtrace_fn in stimtrace_fns:
    print "No stimtraces found for %s: %s, %s. Did you run files_to_trials.py?" % (analysis_id, currslice, currchannel)
else: 
    print "Selected Channel, Slice:", currchannel, currslice
    with open(os.path.join(path_to_trace_structs, stimtrace_fn), 'rb') as f:
	stimtraces = pkl.load(f)

stimlist = sorted(stimtraces.keys(), key=natural_keys)
nstimuli = len(stimlist)

print stimtraces[stimlist[0]].keys()

# In[103]:


# Get stim names:
stiminfo = dict()
if experiment=='gratings_phaseMod' or experiment=='gratings_static':
    print "STIM | ori - sf"
    for stim in stimlist: #sorted(stimtraces.keys(), key=natural_keys):
	
        ori = stimtraces[stim]['name'].split('-')[2]
        sf = stimtraces[stim]['name'].split('-')[4]
        stiminfo[stim] = (int(ori), float(sf))
        print stim, ori, sf

    oris = sorted(list(set([stiminfo[stim][0] for stim in stimlist]))) #, key=natural_keys)
    sfs = sorted(list(set([stiminfo[stim][1] for stim in stimlist]))) #, key=natural_keys)
    noris = len(oris)
    nsfs = len(sfs)
else:
    for stim in sorted(stimlist, key=natural_keys):
        stiminfo[stim] = int(stim)


# ### EXTRACT DF info:

# In[104]:

rois_to_plot = np.arange(0, nrois)  #int(nrois/2)

calcs = dict((roi, dict((stim, dict()) for stim in stimlist)) for roi in rois_to_plot)
dfstruct = dict((roi, dict((stim, dict()) for stim in stimlist)) for roi in rois_to_plot)

for roi in rois_to_plot:
    for stimnum,stim in enumerate(stimlist):

        if use_df is True:
            currtraces = stimtraces[stim]['df']
            #print currtraces.shape
        else:
            if dont_use_raw is True:
                currtraces = stimtraces[stim]['traces']
            else:
                currtraces = stimtraces[stim]['raw_traces']

	#print currtraces.shape
	if len(currtraces.shape)==1:
	    print "Incorrect number of frames provided across trials... Check files_to_trials.py"
        ntrialstmp = len(currtraces)
        nframestmp = [currtraces[i].shape[0] for i in range(len(currtraces))]
        diffs = np.diff(nframestmp)
        if sum(diffs)>0:
            print "Incorrect frame nums per trial:", stimnum, stim
            print nframestmp
	    	
        else:
            nframestmp = nframestmp[0]
 
#         raw = np.empty((ntrialstmp, nframestmp))
#         for trialnum in range(ntrialstmp):
# 	    print currtraces[trialnum].shape
#             raw[trialnum, :] = currtraces[trialnum][0:nframestmp, roi].T

	raw = currtraces[:,:,roi]
        #print raw.shape 
	ntrials = raw.shape[0]
        nframes_in_trial = raw.shape[1]

        xvals = np.empty((ntrials, nframes_in_trial))
        curr_dfs = np.empty((ntrials, nframes_in_trial))

        calcs[roi][stim] = dict()
        calcs[roi][stim]['zscores'] = []
        calcs[roi][stim]['mean_stim_on'] = []
        for trial in range(ntrials):
	    frame_on = stimtraces[stim]['frames_stim_on'][trial][0]
	    frame_on_idx = [i for i in stimtraces[stim]['frames'][trial]].index(frame_on)

            xvals[trial, :] = (stimtraces[stim]['frames'][trial] - frame_on) #+ stimnum*spacing
            
            if use_df is True:
                curr_dfs[trial,:] = raw[trial,:]
            else: 
                baseline = np.mean(raw[trial, 0:frame_on_idx])
                if baseline==0: # or (abs(baseline)>max(raw[trial,:])):
                    print stim, trial, baseline
                    df = np.ones((1, nframes_in_trial))*np.nan
                else:
                    df = (raw[trial,:] - baseline) / baseline
                        #print stim, trial
                curr_dfs[trial,:] = df

            #stim_dur = stimtraces[stim]['frames_stim_on'] 
	    #stimtraces[stim]['frames_stim_on'][trial][1]-stimtraces[stim]['frames_stim_on'][trial][0]
	    volumerate = float(stimtraces[stim]['volumerate'])
	    nframes_on = int(round(stimtraces[stim]['stim_dur'] * volumerate))
            #print stimtraces[stim]['stim_dur'], nframes_on+frame_on_idx 
            baseline_frames = curr_dfs[trial, 0:frame_on_idx]
            stim_frames = curr_dfs[trial, frame_on_idx:frame_on_idx+nframes_on]
	    #print stim, len(stim_frames)
            std_baseline = np.nanstd(baseline_frames)
	    #print np.mean(stim_frames) 
            mean_stim_on = np.nanmean(stim_frames)
            zval_trace = mean_stim_on / std_baseline

            calcs[roi][stim]['zscores'].append(zval_trace)
            calcs[roi][stim]['mean_stim_on'].append(mean_stim_on)
            calcs[roi][stim]['name'] = stimtraces[stim]['name']

            dfstruct[roi][stim]['name'] = stimtraces[stim]['name']
            dfstruct[roi][stim]['tsec'] = xvals[trial,:]/volumerate
	    dfstruct[roi][stim]['raw'] = raw
            dfstruct[roi][stim]['df'] = curr_dfs
            dfstruct[roi][stim]['frame_on'] = (frame_on_idx, frame_on)
            dfstruct[roi][stim]['baseline_vals'] = baseline_frames
            dfstruct[roi][stim]['stim_on_vals'] = stim_frames
            dfstruct[roi][stim]['stim_on_nframes'] = nframes_on 
	    dfstruct[roi][stim]['stim_dur'] = stimtraces[stim]['stim_dur']
	    dfstruct[roi][stim]['iti_dur'] = stimtraces[stim]['iti_dur']
            #print stimtraces[stim]['filesource']
            dfstruct[roi][stim]['files'] = stimtraces[stim]['filesource'] #[trial]
            #print dfstruct[roi][stim]['files']
	    #print dfstruct[roi][stim]['tsec']

dfstruct_fn = '%s_roi_dfstructs_%s.pkl' % (currslice, roi_trace_type)
with open(os.path.join(roi_struct_dir, dfstruct_fn), 'wb') as f:
    pkl.dump(dfstruct, f, protocol=pkl.HIGHEST_PROTOCOL)


