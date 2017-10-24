
# coding: utf-8

# In[1]:


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

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]


# In[2]:


import optparse

parser = optparse.OptionParser()
parser.add_option('-S', '--source', action='store', dest='source', default='/nas/volume1/2photon/projects', help='source dir (root project dir containing all expts) [default: /nas/volume1/2photon/projects]')
parser.add_option('-E', '--experiment', action='store', dest='experiment', default='', help='experiment type (parent of session dir)') 
parser.add_option('-s', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID') 
parser.add_option('-A', '--acq', action='store', dest='acquisition', default='', help="acquisition folder (ex: 'FOV1_zoom3x')")
parser.add_option('-f', '--functional', action='store', dest='functional_dir', default='functional', help="folder containing functional TIFFs. [default: 'functional']")

# parser.add_option('-r', '--roi', action="store",
#                   dest="roi_method", default='blobs_DoG', help="roi method [default: 'blobsDoG]")
# 
parser.add_option('-O', '--stimon', action="store",
                  dest="stim_on_sec", default='', help="Time (s) stimulus ON.")
parser.add_option('-i', '--iti', action="store",
                  dest="iti_pre", default=1., help="Time (s) before stim onset to use as basline [default=1].")

parser.add_option('-z', '--slice', action="store",
                  dest="sliceidx", default=0, help="Slice index to look at (0-index) [default: 0]")

parser.add_option('--custom', action="store_true",
                  dest="custom_mw", default=False, help="Not using MW (custom params must be specified)")

parser.add_option('-g', '--gap', action="store",
                  dest="gap", default=400, help="num frames to separate subplots [default: 400]")

parser.add_option('-I', '--id', action="store",
                  dest="analysis_id", default='', help="analysis_id (includes mcparams, roiparams, and ch). see <acquisition_dir>/analysis_record.json for help.")

parser.add_option('-R', '--rois', action="store",
                  dest="rois_to_plot", default='', help="index of ROIs to plot")

parser.add_option('--interval', action="store",
                  dest="roi_interval", default=10, help="Plot every Nth interval [default: 10]")


parser.add_option('-Y', '--ymax', action="store",
                  dest="ymax", default=3, help="Limit for max y-axis across subplots [default: 3]")
parser.add_option('-y', '--ymin', action="store",
                  dest="ymin", default=-1, help="Limit for min y-axis across subplots [default: -1]")
parser.add_option('-o', '--offset', action="store",
                  dest="stimbar_offset", default=-0.5, help="Y-value at which to draw stim-on bar [default: -0.5]")


parser.add_option('--alpha', action="store",
                  dest="avg_alpha", default=1, help="Alpha val for average trace [default: 1]")
parser.add_option('--width', action="store",
                  dest="avg_width", default=1.2, help="Line width for average trace [default: 1.2]")
parser.add_option('--talpha', action="store",
                  dest="trial_alpha", default=0.3, help="Alpha val for indivdual trials [default: 0.3]")
parser.add_option('--twidth', action="store",
                  dest="trial_width", default=0.1, help="Line width for individual trials [default: 0.1]")

parser.add_option('--no-color', action="store_true",
                  dest="no_color", default=False, help="Don't plot by color (just grayscale)")

parser.add_option('--stim-color', action="store_false",
                  dest="color_by_roi", default=True, help="Color by STIM instead of ROI (default: color by ROI id).")

parser.add_option('--processed', action="store_true",
                  dest="dont_use_raw", default=False, help="Flag to use processed traces instead of raw.")


(options, args) = parser.parse_args() 

analysis_id = options.analysis_id

source = options.source #'/nas/volume1/2photon/projects'
experiment = options.experiment #'scenes' #'gratings_phaseMod' #'retino_bar' #'gratings_phaseMod'
session = options.session #'20171003_JW016' #'20170927_CE059' #'20170902_CE054' #'20170825_CE055'
acquisition = options.acquisition #'FOV1' #'FOV1_zoom3x' #'FOV1_zoom3x_run2' #'FOV1_planar'
functional_dir = options.functional_dir #'functional' #'functional_subset'

stim_on_sec = float(options.stim_on_sec) #2. # 0.5
iti_pre = float(options.iti_pre)

custom_mw = options.custom_mw
spacing = int(options.gap)
curr_slice_idx = int(options.sliceidx)

# ---------------------------------------------------------------------------------
# PLOTTING parameters:
# ---------------------------------------------------------------------------------
dont_use_raw = options.dont_use_raw
# mw = False
# spacing = 25 #400
roi_interval = int(options.roi_interval)

avg_alpha = float(options.avg_alpha) # 1
avg_width = float(options.avg_width) #1.2
trial_alpha = float(options.trial_alpha) #0.8 #0.5 #0.5 #0.7
trial_width = float(options.trial_width) #0.2 #0.3

stim_offset = float(options.stimbar_offset) #-.75 #2.0
ylim_min = float(options.ymin) #-3
ylim_max = float(options.ymax) #50 #3 #100 #5.0 # 3.0

backgroundoffset =  0.3 #0.8

no_color = options.no_color
color_by_roi = options.color_by_roi
#color_by_roi = True

cmaptype = 'rainbow'

# ---------------------------------------------------------------------
# Get relevant stucts:
# ---------------------------------------------------------------------
acquisition_dir = os.path.join(source, experiment, session, acquisition)

# Create ROI dir in figures:
figbase = os.path.join(acquisition_dir, 'figures', analysis_id) #'example_figures'
if not os.path.exists(figbase):
    os.makedirs(figbase)

if dont_use_raw is True:
    figdir = os.path.join(figbase, 'rois_processed')
else:
    figdir = os.path.join(figbase, 'rois')
if not os.path.exists(figdir):
    os.mkdir(figdir)
print "Saving ROI subplots to dir:", figdir
 

# Load reference info:
ref_json = 'reference_%s.json' % functional_dir 
with open(os.path.join(acquisition_dir, ref_json), 'r') as fr:
    ref = json.load(fr)
    
# Load SI meta data:
si_basepath = ref['raw_simeta_path'][0:-4]
simeta_json_path = '%s.json' % si_basepath
with open(simeta_json_path, 'r') as fs:
    simeta = json.load(fs)

# Get stim params:
if custom_mw is False:
    currfile='File001'
    # stim_on_sec = 2.
    # iti = 1. #4.
    nframes = int(simeta[currfile]['SI']['hFastZ']['numVolumes'])
    framerate = float(simeta[currfile]['SI']['hRoiManager']['scanFrameRate'])
    volumerate = float(simeta[currfile]['SI']['hRoiManager']['scanVolumeRate'])
    frames_tsecs = np.arange(0, nframes)*(1/volumerate)

    nframes_on = stim_on_sec * volumerate
    #nframes_off = vols_per_trial - nframes_on
    nframes_iti_pre = round(iti_pre * volumerate) 
    print nframes_on
    print nframes_iti_pre


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
print type(maskpaths)

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


# Load masks:
masks = dict(("Slice%02d" % int(slice_idx), dict()) for slice_idx in slices)
for sidx,maskpath in zip(sorted(slices), sorted(maskpaths, key=natural_keys)):
    slice_name = "Slice%02d" % int(sidx) #+1)
    print "Loading masks: %s..." % slice_name 
    tmpmasks = loadmat(maskpath); tmpmasks = tmpmasks['masks']
    masks[slice_name]['nrois'] =  tmpmasks.shape[2]
    masks[slice_name]['masks'] = tmpmasks


# Get masks for current slice:
slice_names = sorted(masks.keys(), key=natural_keys)
print "SLICE NAMES:", slice_names
curr_slice_name = "Slice%02d" % slices[curr_slice_idx]
currmasks = masks[curr_slice_name]['masks']
print currmasks.shape

nrois = masks[curr_slice_name]['nrois']
print "NROIS:", nrois
#rois_to_plot = np.arange(0, nrois)


roi_interval = 1
tmprois = options.rois_to_plot
if len(tmprois)==0:
    rois_to_plot = np.arange(0, nrois, roi_interval) #int(nrois/2)
    sort_name = 'all' #% roi_interval
else:
    rois_to_plot = options.rois_to_plot.split(',')
    rois_to_plot = [int(r) for r in rois_to_plot]
    roi_string = "".join(['r%i' % int(r) for r in rois_to_plot])
    print roi_string
    sort_name = 'selected_%s' % roi_string

print "ROIS TO PLOT:", rois_to_plot


# Create ROI dir in figures:
figbase = os.path.join(acquisition_dir, 'figures', analysis_id) #'example_figures'
if not os.path.exists(figbase):
    os.makedirs(figbase)

if len(tmprois)==0:
    figdir = os.path.join(figbase, 'rois')
else:
    figdir = os.path.join(figbase, 'roi_subsets', sort_name, 'roi_traces')
if not os.path.exists(figdir):
    os.makedirs(figdir)
print "Saving ROI subplots to dir:", figdir
 

# Get PARADIGM INFO:
path_to_functional = os.path.join(acquisition_dir, functional_dir)
paradigm_dir = 'paradigm_files'
path_to_paradigm_files = os.path.join(path_to_functional, paradigm_dir)
#path_to_trace_structs = os.path.join(acquisition_dir, 'Traces', roi_method, 'Parsed')
path_to_trace_structs = os.path.join(acquisition_dir, 'Traces', ref['trace_id'][analysis_id], 'Parsed')
    

# Load stim trace structs:
print "Loading parsed traces..."
signal_channel = ref['signal_channel'][analysis_id] #int(options.selected_channel)

currchannel = "Channel%02d" % int(signal_channel)
currslice = "Slice%02d" % slices[curr_slice_idx] # curr_slice_idx
stimtrace_fns = os.listdir(path_to_trace_structs)
stimtrace_fn = "stimtraces_%s_%s.pkl" % (currslice, currchannel)
if not stimtrace_fn in stimtrace_fns:
    print "No stimtraces found for %s: %s, %s. Did you run files_to_trials.py?" % (analysis_id, currslice, currchannel)
else: 
    print "Selected Channel, Slice:", currchannel, currslice
    with open(os.path.join(path_to_trace_structs, stimtrace_fn), 'rb') as f:
	stimtraces = pkl.load(f)
# stimtrace_fns = sorted([f for f in stimtrace_fns if 'stimtraces' in f and currchannel in f and f.endswith('.pkl')], key=natural_keys)
# if len(stimtrace_fns)==0:
#     print "No stim traces found for Channel %i" % int(selected_channel)
# 
# stimtrace_fn = [f for f in stimtrace_fns if currchannel in f and currslice in f][0]
stimlist = sorted(stimtraces.keys(), key=natural_keys)
nstimuli = len(stimlist)

### Load trial info dict:
# with open(os.path.join(path_to_paradigm_files, 'parsed_trials.pkl'), 'rb') as f:
#     trialdict = pkl.load(f)
# trialdict.keys()
# 

# In[11]:

# ---------------------------------------------------------------------------
# PLOTTING:
# ----------------------------------------------------------------------------
if no_color is True:
    colorvals = np.zeros((nrois, 3));
    #colorvals[:,1] = 1
else:
    colormap = plt.get_cmap(cmaptype)
    if color_by_roi:
	colorvals = colormap(np.linspace(0, 1, nrois)) #get_spaced_colors(nrois)
    else:
	colorvals = colormap(np.linspace(0, 1, nstimuli)) #get_spaced_colors(nstimuli)


# Get stim names:
stiminfo = dict()
if experiment=='gratings_phaseMod':
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

# In[15]:


#stiminfo


# In[80]:


#print rois_to_plot


even_idxs = np.arange(noris, noris*2)
print even_idxs
for ridx, roi in enumerate(rois_to_plot):

    fig = plt.figure(figsize=(noris, noris))
    gs = gridspec.GridSpec(nsfs, noris)
    gs.update(wspace=.01, hspace=.01)
    
    rowidx= 0
    plotidx = 0
    for stimidx, stim in enumerate(stimlist):
    #for sf in range(nsfs):
        ori = stiminfo[stim][0]
        sf = stiminfo[stim][1]
       
        if np.mod(stimidx,2)==0:
            plt.subplot(gs[stimidx/2])
            bottomrow = False
        else: 
            plt.subplot(gs[even_idxs[rowidx]])
            bottomrow = True
            rowidx += 1
	
	# Get traces for curent stimulus from 'stimtraces' struct:
        if dont_use_raw is True:
            currtraces = stimtraces[stim]['traces']
        else:
            currtraces = stimtraces[stim]['raw_traces']

	ntrialstmp = len(currtraces)
	nframestmp = [currtraces[i].shape[0] for i in range(len(currtraces))]
	#print "N frames per trial:", nframestmp
	nframestmp = nframestmp[0]

	# Get RAW traces for each trial:
	raw = np.empty((ntrialstmp, nframestmp))
	for trialnum in range(ntrialstmp):
	    raw[trialnum, :] = currtraces[trialnum][0:nframestmp, roi].T

        xvals = np.arange(0, raw.shape[1])
	
	ntrials = raw.shape[0]
	nframes_in_trial = raw.shape[1]
	#print "ntrials: %i, nframes in trial: %i" % (ntrials, nframes_in_trial)

	# Calculate df/f:
	curr_dfs = np.empty((ntrials, nframes_in_trial))
	for trial in range(ntrials):
	    if custom_mw is True:
		#print stimtraces[stim]['frames_stim_on'][trial]
		frame_on = stimtraces[stim]['frames_stim_on'][trial][0]
		frame_off = stimtraces[stim]['frames_stim_on'][trial][-1]
	     
		frame_on_idx = [i for i in stimtraces[stim]['frames'][trial]].index(frame_on)
		# print frame_off - frame_on
		#frame_on = int(frames_iti)+1 #stimtraces[stim]['frames_stim_on'][trial][0]
	    else:
		# frame_on = int(nframes_iti_pre)+1 #stimtraces[stim]['frames_stim_on'][trial][0]
		frame_on = stimtraces[stim]['frames_stim_on'][trial][0]
		frame_on_idx = [i for i in stimtraces[stim]['frames'][trial]].index(frame_on)

	    baseline = np.mean(raw[trial, 0:frame_on])
	    #print "baseline:", baseline
	    df = (raw[trial,:] - baseline) / baseline
	    #print stim, trial
	    curr_dfs[trial,:] = df

	    if color_by_roi:
		plt.plot(xvals, df, color=colorvals[roi], alpha=trial_alpha, linewidth=trial_width)
	    else:
		plt.plot(xvals, df, color=colorvals[stimnum], alpha=trial_alpha, linewidth=trial_width)

	# Plot average:
	avg = np.mean(curr_dfs, axis=0) 
	if color_by_roi:
	    plt.plot(xvals, avg, color=colorvals[roi], alpha=avg_alpha, linewidth=avg_width)
	    #plt.plot(xvals, avg, color='k', alpha=.5, linewidth=1.2)                
	else:
	    plt.plot(xvals, avg, color=colorvals[stimnum], alpha=avg_alpha, linewidth=avg_width)

	# Plot stimulus ON period:
	if custom_mw is True:
	    #frame_off_idx = 
	    stim_frames = xvals[0] + [frame_on_idx, frame_on_idx + (frame_off - frame_on)] 	
	    #stim_frames = xvals[0] + stimtraces[stim]['frames_stim_on'][trial] #frames_stim_on[stim][trial]
	    # start_fr = int(frames_iti) + 1
	    # stim_frames = xvals[0] + [start_fr, start_fr+nframes_on]
	else:
	    #stim_frames = xvals[0] + stimtraces[stim]['frames_stim_on'][trial] #frames_stim_on[stim][trial]
	    # start_fr = int(nframes_iti_pre) + 1
	    # stim_frames = xvals[0] + [start_fr, start_fr+nframes_on-1]
	    #on_fr_idx = stimtraces[stim]['frames'][trial].index(stimtraces[stim]['frames_stim_on'][trial][0])
	    on_fr_idx =  int(nframes_iti_pre)
	    nframes_on = (stimtraces[stim]['frames_stim_on'][trial][1] - stimtraces[stim]['frames_stim_on'][trial][0] + 1)
	    off_fr_idx = on_fr_idx + nframes_on - 1
	    stim_frames = xvals[0] + [on_fr_idx, off_fr_idx]

	if no_color is True:
            plt.plot(stim_frames, np.ones((2,))*stim_offset, color='r')
	else:
	    plt.plot(stim_frames, np.ones((2,))*stim_offset, color='k')


        plt.ylim([ylim_min, ylim_max])
       

        # Format subplots:
        if plotidx==0:
            plt.axis('off')
            ax = plt.gca()
            ax.text(-.75,0.5, sfs[0], size=12, ha="center", rotation=90, transform=ax.transAxes)
        elif plotidx==nsfs-1:
            plt.yticks([0, 1])
            sns.despine(bottom=True, trim=True, offset=5)
            ax = plt.gca()
            ax.text(-.75,0.5, sfs[1], size=12, ha="center", rotation=90, transform=ax.transAxes)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_xaxis().set_ticks([])
        else:
            plt.axis('off')
        
        if bottomrow is True:
            ax = plt.gca()
            ax.text(.5,-0.2, oris[rowidx-1], size=12, ha="center", rotation=0, transform=ax.transAxes)
           
        ptitle = "%2.0f., %0.2f" % (float(stiminfo[stim][0]), float(stiminfo[stim][1]))
        #plt.title(ptitle)
        plotidx += 1

    plt.subplots_adjust(top=1)
    plt.suptitle('ROI: %i' % int(roi))
    
    figname = '%s_stimgrid_roi%i.png' % (currslice, int(roi))
    plt.savefig(os.path.join(figdir, figname), bbox_inches='tight', pad=0)

   # plt.show()

    plt.close(fig)

# In[ ]:




