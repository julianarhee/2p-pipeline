#!/usr/bin/env python2
'''
'''

# In[79]:


import os
import json
import re
import h5py
import optparse
import sys
import itertools
import scipy.io as spio
import numpy as np
import tifffile as tf
import seaborn as sns
import cPickle as pkl

# %matplotlib notebook
from bokeh.plotting import figure
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import skimage.color
from json_tricks.np import dump, dumps, load, loads
from  matplotlib.ticker import FuncFormatter
from matplotlib.ticker import MaxNLocator

def get_axis_limits(ax, scale=(0.9, 0.9)):
    return ax.get_xlim()[1]*scale[0], ax.get_ylim()[1]*scale[1]


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]

#%%%
parser = optparse.OptionParser()

parser.add_option('-R', '--root', action='store', dest='rootdir', default='/nas/volume1/2photon/data', help='data root dir (root project dir containing all animalids) [default: /nas/volume1/2photon/data, /n/coxfs01/2pdata if --slurm]')
parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')

# Set specific session/run for current animal:
parser.add_option('-S', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID')
parser.add_option('-A', '--acq', action='store', dest='acquisition', default='FOV1', help="acquisition folder (ex: 'FOV1_zoom3x') [default: FOV1]")
parser.add_option('-r', '--run', action='store', dest='run', default='', help="name of run dir containing tiffs to be processed (ex: gratings_phasemod_run1)")

parser.add_option('--slurm', action='store_true', dest='slurm', default=False, help="set if running as SLURM job on Odyssey")

parser.add_option('-t', '--trace-id', action="store",
                  dest="trace_id", default='', help="Name of trace extraction set to use.")

(options, args) = parser.parse_args()



# Set USER INPUT options:
rootdir = options.rootdir
animalid = options.animalid
session = options.session
acquisition = options.acquisition
run = options.run
slurm = options.slurm
trace_id = options.trace_id

trace_type = 'parsed'

#%%
session_dir = os.path.join(rootdir, animalid, session)
run_dir = os.path.join(rootdir, animalid, session, acquisition, run)

# Load reference info:
runinfo_path = os.path.join(run_dir, '%s.json' % run)

with open(runinfo_path, 'r') as fr:
    runinfo = json.load(fr)
nfiles = runinfo['ntiffs']
file_names = sorted(['File%03d' % int(f+1) for f in range(nfiles)], key=natural_keys)
nvolumes = runinfo['nvolumes']
frame_idxs = runinfo['frame_idxs']
if len(frame_idxs) == 0:
    # No flyback
    frame_idxs = np.arange(0, nvolumes)

# LOAD TID:
trace_dir_base = os.path.join(session_dir, 'Traces')
tracedict_path = os.path.join(trace_dir_base, 'tid_%s.json' % session)
with open(tracedict_path, 'r') as tr:
    tracedict = json.load(tr)
TID = tracedict[trace_id]


# Get TRACE DIR, create output dir:
traceid_dir = os.path.join(trace_dir_base, '%s_%s' % (TID['trace_id'], TID['trace_hash']))
trace_dir = os.path.join(traceid_dir, trace_type)

print "Loading traces from:", trace_dir

# Create parsed-trials dir with default format:
parsed_fn = [p for p in os.listdir(trace_dir) if p.endswith('hdf5') and TID['trace_hash'] in p and 'rois' not in p]
print "Found %i trace-files corresponding to %i tiffs." % (len(parsed_fn), nfiles)
parsed_path = os.path.join(trace_dir, parsed_fn[0])

infile = h5py.File(parsed_path, 'r')


roistruct_path = os.path.join(trace_dir, 'rois_%s_%s.hdf5' % (TID['trace_id'], TID['trace_hash']))
outfile = h5py.File(roistruct_path, 'w')

# In[97]:


# ---------------------------------------------------------------------------------
# PLOTTING parameters:
# ---------------------------------------------------------------------------------



# ---------------------------------------------------------------------
# Get relevant stucts:
# ---------------------------------------------------------------------
slice_names = sorted([str(k) for k in infile.keys()], key=natural_keys)
print "Extracting traces from ROIs across %i slices...." % len(slice_names)

# In[102]:

nrois = dict()
for currslice in slice_names:
    if currslice not in outfile.keys():
        slice_grp = outfile.create_group(currslice)
    else:
        slice_grp = outfile[currslice]

    stim_names = sorted([str(k) for k in infile[currslice].keys()], key=natural_keys)

    nrois[currslice] = infile[currslice][stim_names[0]]['trial00001'].shape[1]

    # Get unique configurations of stim ID, position, rotation, and size:
    stimconfigs = dict()
    stimfile = dict()
    stimlist = sorted(infile[currslice].keys(), key=natural_keys)
    for stim_name in sorted(infile[currslice].keys(), key=natural_keys):
        stimconfigs[stim_name] = {'position': [],
                                  'size': [],
                                  'rotation': []
                                  }
        positions = [tuple(infile[currslice][stim_name][trialnum].attrs['position']) for trialnum in infile[currslice][stim_name].keys()]
        rotations = [infile[currslice][stim_name][trialnum].attrs['rotation'] for trialnum in infile[currslice][stim_name].keys()]
        sizes = [tuple(infile[currslice][stim_name][trialnum].attrs['scale']) for trialnum in infile[currslice][stim_name].keys()]

        ntransforms = len(stimconfigs[stim_name].keys())
        transform_combos = list(itertools.product(list(set(positions)), list(set(sizes)), list(set(rotations))))
        ncombinations = len(transform_combos)

        stimconfigs[stim_name]['position'] = list(set(positions))
        stimconfigs[stim_name]['rotation'] = list(set(rotations))
        stimconfigs[stim_name]['size'] = list(set(sizes))

        stimfile[stim_name] = dict()
        filepaths = list(set([infile[currslice][stim_name][trialnum].attrs['filepath'] for trialnum in infile[currslice][stim_name].keys()]))
        filehashes = list(set([infile[currslice][stim_name][trialnum].attrs['filehash'] for trialnum in infile[currslice][stim_name].keys()]))

        if len(filepaths) > 1 or len(filehashes) > 1:
            print "*********WARNING*****************************"
            print "STIM %s: multiple file paths or hashes found!" % stim_name
            print "*********WARNING*****************************"
            stimfile[stim_name]['filepath'] = filepaths
            stimfile[stim_name]['filehashes'] = filehashes
        else:
            stimfile[stim_name]['filepath'] = filepaths[0]
            stimfile[stim_name]['filehash'] = filehashes[0]

    for roi in range(nrois[currslice]):
        roi_name = 'roi%05d' % int(roi)
        if roi_name not in slice_grp.keys():
            roi_grp = outfile[currslice].create_group(roi_name)
        else:
            roi_grp = outfile[currslice][roi_name]

        for config_idx in range(ncombinations):

            curr_pos = transform_combos[config_idx][0]
            curr_size = transform_combos[config_idx][1]
            curr_rot = transform_combos[config_idx][2]

            curr_trials = dict((stim_name, [trialname for trialname in infile[currslice][stim_name].keys()\
                                            if (infile[currslice][stim_name][trialname].attrs['position'] == curr_pos).all()\
                                            and infile[currslice][stim_name][trialname].attrs['rotation'] == curr_rot\
                                            and (infile[currslice][stim_name][trialname].attrs['scale'] == curr_size).all()])\
                                for stim_name in stimlist)


            for stim_name in sorted(curr_trials.keys(), key=natural_keys):
                print "roi", roi, stim_name
                trialmat = np.array([infile[currslice][stim_name][trialnum][:, roi] for trialnum in curr_trials[stim_name]])
                print trialmat.dtype

                dset = roi_grp.create_dataset(stim_name, trialmat.shape, dtype=trialmat.dtype)
                dset[...] = trialmat
                dset.attrs['position'] = curr_pos
                dset.attrs['size'] = curr_size
                dset.attrs['rotation'] = curr_rot
                dset.attrs['filepath'] = stimfile[stim_name]['filepath']
                dset.attrs['filehash'] = stimfile[stim_name]['filehash']

#%%


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


