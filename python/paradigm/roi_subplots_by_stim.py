
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

parser.add_option('-R', '--rois', action="store",
                  dest="rois_to_plot", default='', help="index of ROIs to plot")

parser.add_option('-i', '--iti', action="store",
                  dest="iti_pre", default=1., help="Time (s) before stim onset to use as basline [default=1].")

parser.add_option('-z', '--slice', action="store",
                  dest="sliceidx", default=1, help="Slice number look at (1-index) [default: 1]")

# PLOTTING PARAMS:
parser.add_option('-r', '--nrows', action="store",
                  dest="nrows", default=5, help="num rows in subplots [default: 5]")


parser.add_option('-g', '--gap', action="store",
                  dest="gap", default=400, help="num frames to separate subplots [default: 400]")

# parser.add_option('--interval', action="store",
#                   dest="roi_interval", default=10, help="Plot every Nth interval [default: 10]")

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
parser.add_option('--file-color', action="store_true",
                  dest="color_by_file", default=False, help="Color by FILE SOURCE.")


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

iti_pre = float(options.iti_pre) # 1.0 #float(options.iti_pre)

spacing = int(options.gap) # int(400) #int(options.gap)
curr_slice_idx = int(options.sliceidx) #int(1) #int(options.sliceidx)



# In[97]:


# ---------------------------------------------------------------------------------
# PLOTTING parameters:
# ---------------------------------------------------------------------------------
dont_use_raw = options.dont_use_raw #True #options.dont_use_raw
use_df = options.use_df

#roi_interval = int(options.roi_interval)

avg_alpha = float(options.avg_alpha) # 1
avg_width = float(options.avg_width) #1.2
trial_alpha = float(options.trial_alpha) #0.8 #0.5 #0.5 #0.7
trial_width = float(options.trial_width) #0.2 #0.3

stim_offset = float(options.stimbar_offset) #-.75 #2.0
ylim_min = float(options.ymin) #-3
ylim_max = float(options.ymax) #50 #3 #100 #5.0 # 3.0

no_color = options.no_color
color_by_roi = options.color_by_roi
color_by_file = options.color_by_file
cmaptype = 'rainbow'


# In[98]:



# ---------------------------------------------------------------------
# Get relevant stucts:
# ---------------------------------------------------------------------
acquisition_dir = os.path.join(source, experiment, session, acquisition)

# Create ROI dir in figures:
figbase = os.path.join(acquisition_dir, 'figures', analysis_id) #'example_figures'
if not os.path.exists(figbase):
    os.makedirs(figbase)

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


# In[99]:


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


# In[100]:




roi_interval = 1
tmprois = '' #options.rois_to_plot
if len(tmprois)==0:
    rois_to_plot = np.arange(0, nrois, roi_interval) #int(nrois/2)
    sort_name = '%s_all' % curr_slice_name #% roi_interval
    no_color = True # by default, if plotting all ROIs, don't color by ROI    
else:
    rois_to_plot = tmprois.split(',')
    rois_to_plot = [int(r) for r in rois_to_plot]
    roi_string = "".join(['r%i' % int(r) for r in rois_to_plot])
    print roi_string
    sort_name = '%s_selected_%s' % (curr_slice_name, roi_string)

print "ROIS TO PLOT:", rois_to_plot


# In[101]:

if use_df is True:
    roi_trace_type = 'df'
else:
    if dont_use_raw is True:
        roi_trace_type = 'processed' #figdir = os.path.join(figbase, 'rois_processed')
    else:
        roi_trace_type = 'raw'

if len(tmprois)==0:
    figdir = os.path.join(figbase, 'rois', 'all', roi_trace_type) #'rois')
else:
    figdir = os.path.join(figbase, 'rois', 'selected',  sort_name, roi_trace_type)
if not os.path.exists(figdir):
    os.makedirs(figdir)
print "Saving ROI subplots to dir:", figdir


# In[102]:



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
stimtrace_fn = "%s_stimtraces_%s_%s.pkl" % (analysis_id, currslice, currchannel)
#stimtrace_fn = "stimtraces_%s_%s.pkl" % (currslice, currchannel)
if not stimtrace_fn in stimtrace_fns:
    print "No stimtraces found for %s: %s, %s. Did you run files_to_trials.py?" % (analysis_id, currslice, currchannel)
else: 
    print "Selected Channel, Slice:", currchannel, currslice
    with open(os.path.join(path_to_trace_structs, stimtrace_fn), 'rb') as f:
	stimtraces = pkl.load(f)


stimlist = sorted(stimtraces.keys(), key=natural_keys)
nstimuli = len(stimlist)

roi_struct_dir = os.path.join(path_to_trace_structs, 'rois')

# LOAD DF STRUCT for current slice:
dfstruct_fn = '%s_roi_dfstructs_%s.pkl' % (currslice, roi_trace_type)
with open(os.path.join(roi_struct_dir, dfstruct_fn), 'rb') as f:
    dfstruct = pkl.load(f)
print dfstruct_fn


# In[103]:

# Order stimlist by subplot idx:

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
    subplot_stimlist = []
    for sf in sorted(sfs):
	for oi in sorted(oris):
	    match = [k for k in stiminfo.keys() if stiminfo[k][0]==oi and stiminfo[k][1]==sf][0]
	    subplot_stimlist.append(match)
    print subplot_stimlist 
            
    
else:
    for stim in sorted(stimlist, key=natural_keys):
        stiminfo[stim] = int(stim)

    subplot_stimlist = sorted(stimlist, key=natural_keys)
# In[105]:


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


if color_by_file is True:
    colormap = plt.get_cmap(cmaptype)

    nfiles = ref['ntiffs']
    file_names = ['File%03d' % int(i+1) for i in range(nfiles)]
    filecolors = colormap(np.linspace(0,1, nfiles))

# In[106]:


# Set up subplots:
if experiment=='gratings_phaseMod' or experiment=='gratings_static':
    nrows = nsfs
    ncols = noris
else:
    nrows = int(options.nrows) #5 #int(np.ceil(np.sqrt(len(stimlist))))
    ncols = int(len(stimlist))/nrows #int(np.ceil(len(stimlist)/float(nrows)))

print nrows, ncols


# In[107]:


stimnames = [str(int(i)) for i in stimlist]
stimnums = [int(i) for i in stimlist]

# get the tick label font size
fontsize_pt = 20 #float(plt.rcParams['ytick.labelsize'])
dpi = 72.27
spacer = 20

# comput the matrix height in points and inches
matrix_height_pt = fontsize_pt * nrows * spacer
matrix_height_in = matrix_height_pt / dpi

# compute the required figure height 
top_margin = 0.01  # in percentage of the figure height
bottom_margin = 0.05 # in percentage of the figure height
figure_height = matrix_height_in / (1 - top_margin - bottom_margin)


# In[109]:


for roi in rois_to_plot:
#for roi in [0]:
    # build the figure instance with the desired height
    if nrows==1:
        figwidth_multiplier = ncols*1
    else:
	figwidth_multiplier = 1
    fig, axs = plt.subplots(
	    nrows=nrows, 
	    ncols=ncols, 
	    sharex=True,
	    sharey=True,
	    figsize=(figure_height*figwidth_multiplier,figure_height), 
	    gridspec_kw=dict(top=1-top_margin, bottom=bottom_margin, wspace=0.05, hspace=0.05))

    row=0
    col=0
    #print nrows, ncols
    plotidx = 0
    for stim in subplot_stimlist: #sorted(stimlist, key=natural_keys): #ROIs:

	if col==(ncols) and nrows>1:
	    row += 1
	    col = 0

	if len(axs.shape)>1:
	    ax_curr = axs[row, col] #, col]
	else:
	    ax_curr = axs[col]
	
	
	tmp_curr_dfs = np.copy(dfstruct[roi][stim]['df'])
	tmp_ntrials = tmp_curr_dfs.shape[0]
	ntrialframes = tmp_curr_dfs.shape[1]
        bad_trials = [tr for tr in range(tmp_ntrials) if any(dv > 3 for dv in dfstruct[roi][stim]['df'][tr,:])]
        good_trials = [tr for tr in range(tmp_ntrials) if tr not in bad_trials]
        curr_dfs = np.empty((len(good_trials), ntrialframes))
        for ti,tr in enumerate(sorted(good_trials)):
            curr_dfs[ti,:] = tmp_curr_dfs[tr,:]

	ntrials = curr_dfs.shape[0]
	xvals = np.arange(0, ntrialframes)
	tsecs = [i for i in dfstruct[roi][stim]['tsec']]
	stim_dur = dfstruct[roi][stim]['stim_dur']
 	iti_dur = dfstruct[roi][stim]['iti_dur']	
	tpoints = [int(i) for i in np.arange(-1*iti_pre, stim_dur+iti_dur)]

	for trial in range(ntrials):
            #print dfstruct[roi][stim]['files'][trial]
            if color_by_file:
                which_file = dfstruct[roi][stim]['files'][trial]
                file_idx = file_names.index(which_file)

	    if color_by_roi:
                if color_by_file:
                    ax_curr.plot(tsecs, curr_dfs[trial,:], color=filecolors[file_idx], alpha=1, linewidth=trial_width*2, label=which_file)
                       
   
                else:
		    ax_curr.plot(tsecs, curr_dfs[trial,:], color=colorvals[roi], alpha=trial_alpha, linewidth=trial_width)
            
	    else:
		ax_curr.plot(tsecs, curr_dfs[trial,:], color=colorvals[stimnum], alpha=trial_alpha, linewidth=trial_width)

	# Plot average:
	avg = np.nanmean(curr_dfs, axis=0) 
	if color_by_roi:
	    ax_curr.plot(tsecs, avg, color=colorvals[roi], alpha=avg_alpha, linewidth=avg_width)
	else:
	    ax_curr.plot(tsecs, avg, color=colorvals[stimnum], alpha=avg_alpha, linewidth=avg_width)

	# Plot stimulus ON period:
        frame_on_idx = dfstruct[roi][stim]['frame_on'][0]
        frame_on_tsec = tsecs[frame_on_idx]
        #print frame_on_idx
        #stim_frames = [frame_on_tsec, frame_on_tsec+dfstruct[roi][stim]['stim_dur']]
        stim_times = [frame_on_tsec, frame_on_tsec+dfstruct[roi][stim]['stim_dur']]

	if no_color is True:
            ax_curr.plot(stim_times, np.ones((2,))*stim_offset, color='r')
	else:
	    ax_curr.plot(stim_times, np.ones((2,))*stim_offset, color='k')


	# Deal with axes:  
	ax_curr.set_ylim([ylim_min, ylim_max])

	if col==0:
            if row>0:
	        ax_curr.set_xlabel('time (s)')
	        ax_curr.tick_params(axis='x', which='both',length=0)
            else:
                ax_curr.set_xlabel('')
                ax_curr.tick_params(axis='x', which='both',length=0)
 
	    ax_curr.yaxis.set_major_locator(MaxNLocator(5, integer=True))
	    # ax_curr.xaxis.set_major_locator(MaxNLocator(noris, integer=True))
	    sns.despine(bottom=True, right=True, offset=5, trim=True, ax=ax_curr)
            ax_curr.set(xticks=tpoints)
	else:
	    ax_curr.axis('off')

        ax_curr.set_title(stiminfo[stim])
	#ax_curr.annotate(dfstruct[roi][stim]['name'], xy=get_axis_limits(ax_curr))
	#ax_curr.legend().set_visible(False)

	col = col + 1
	    
	#ptitle = "%2.0f., %0.2f" % (float(stiminfo[stim][0]), float(stiminfo[stim][1]))
	
	plotidx += 1

    nleftover = (nrows*ncols)-plotidx
    for p in range(nleftover):
	ax_curr = axs[row, col+p]
	ax_curr.axis('off')


    #plt.subplots_adjust(top=1)
    plt.suptitle('ROI: %i' % int(roi))
    if color_by_file is True:
        ax_curr.legend().set_visible(True) #, label=file_names) 

    figname = '%s_stimgrid_roi%i_%s.png' % (currslice, int(roi), roi_trace_type)
    plt.savefig(os.path.join(figdir, figname), bbox_inches='tight', pad=0)

    #plt.show()

    plt.close(fig)


# In[ ]:





# In[ ]:




