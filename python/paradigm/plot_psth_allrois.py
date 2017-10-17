
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

# def get_spaced_colors(n):
#     max_value = 16581375 #255**3
#     interval = int(max_value / n)
#     colors = [hex(I)[2:].zfill(6) for I in range(1, max_value, interval)]

#     return [(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors]


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
parser.add_option('-i', '--iti', action="store",
                  dest="iti_pre", default=1., help="Time (s) before stim onset to use as basline [default=1].")
parser.add_option('-z', '--slice', action="store",
                  dest="sliceidx", default=0, help="Slice index to look at (0-index) [default: 0]")

parser.add_option('--custom', action="store_true",
                  dest="custom_mw", default=False, help="Not using MW (custom params must be specified)")
parser.add_option('-g', '--gap', action="store",
                  dest="gap", default=400, help="num frames to separate subplots [default: 400]")
parser.add_option('-c', '--channel', action="store",
                  dest="selected_channel", default=1, help="Channel idx of signal channel. [default: 1]")

(options, args) = parser.parse_args() 


source = options.source #'/nas/volume1/2photon/projects'
experiment = options.experiment #'scenes' #'gratings_phaseMod' #'retino_bar' #'gratings_phaseMod'
session = options.session #'20171003_JW016' #'20170927_CE059' #'20170902_CE054' #'20170825_CE055'
acquisition = options.acquisition #'FOV1' #'FOV1_zoom3x' #'FOV1_zoom3x_run2' #'FOV1_planar'
functional_dir = options.functional_dir #'functional' #'functional_subset'

roi_method = options.roi_method

stim_on_sec = float(options.stim_on_sec) #2. # 0.5
iti_pre = float(options.iti_pre)

custom_mw = options.custom_mw
spacing = int(options.gap)
curr_slice_idx = int(options.sliceidx)
selected_channel = int(options.selected_channel)

# source = '/nas/volume1/2photon/projects'
# experiment = 'gratings_phaseMod'
# session = '20171009_CE059'
# acquisition = 'FOV1_zoom3x'
# functional_dir = 'functional'

# curr_file_idx = 6
# curr_slice_idx = 0 #1 #20
# stim_on_sec = 2.
# iti = 1. #4.


# source = '/nas/volume1/2photon/projects'
# experiment = 'scenes'
# session = '20171003_JW016'
# acquisition = 'FOV1'
# functional_dir = 'functional'

# ---------------------------------------------------------------------------------
# PLOTTING parameters:
# ---------------------------------------------------------------------------------
# mw = False
# spacing = 25 #400
roi_interval = 1

trial_alpha = 0.2 #0.5 #0.7
trial_width = 0.1 #0.3

stim_offset = -.75 #2.0
ylim_min = -3
ylim_max = 100 #5.0 # 3.0

backgroundoffset =  0.3 #0.8


# curr_slice_idx = 20
rois_to_plot = []

# curr_file_idx = 2
# curr_slice_idx = 0 #20
# rois_to_plot = np.array([1, 20, 23, 27, 35, 42, 45, 50, 57, 66, 76, 92, 110]) - 1

# curr_slice_idx = 19 #20
# rois_to_plot = np.array([27, 58, 134]) #27    58   134


#curr_roi_method = 'blobs_DoG'
plot_traces = True #False
#rois_to_plot = np.array([1, 20, 23, 27, 35, 42, 45, 50, 57, 66, 76, 92, 110]) - 1



color_by_roi = True
cmaptype = 'rainbow'

# ---------------------------------------------------------------------------------


acquisition_dir = os.path.join(source, experiment, session, acquisition)
figdir = os.path.join(acquisition_dir, 'example_figures')

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


# Create tmp fig dir:
figdir = os.path.join(acquisition_dir, 'example_figures')
if not os.path.exists(figdir):
    os.mkdir(figdir)

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

slice_names = sorted(masks.keys(), key=natural_keys)
print "SLICE NAMES:", slice_names
curr_slice_name = slice_names[curr_slice_idx]

nrois = masks[curr_slice_name]['nrois']
print "NROIS:", nrois

if len(rois_to_plot)==0:
    rois_to_plot = np.arange(0, nrois, roi_interval) #int(nrois/2)
    sort_name = '_every%i' % roi_interval
else:
    sort_name = '_sorted'
    

# Get FILE ("tiff") list:
average_source = 'Averaged_Slices_Corrected'
signal_channel = int(options.selected_channel)
average_slice_dir = os.path.join(acquisition_dir, functional_dir, 'DATA', average_source, "Channel{:02d}".format(signal_channel))
file_names = [f for f in os.listdir(average_slice_dir) if '_vis' not in f]
print "File names:", file_names
nfiles = len(file_names)

# Get AVERAGE slices (for current file):
curr_file_idx = 1
curr_file_name = file_names[curr_file_idx]
#curr_file_name = file_names[ref['refidx']]
curr_slice_dir = os.path.join(average_slice_dir, curr_file_name)
slice_fns = sorted([f for f in os.listdir(curr_slice_dir) if f.endswith('.tif')], key=natural_keys)

# Get average slice image for current-file, current-slice:
curr_slice_fn = slice_fns[curr_slice_idx]
avg_tiff_path = os.path.join(curr_slice_dir, curr_slice_fn)
with tf.TiffFile(avg_tiff_path) as tif:
    avgimg = tif.asarray()


# Get PARADIGM INFO:
path_to_functional = os.path.join(acquisition_dir, functional_dir)
paradigm_dir = 'paradigm_files'
path_to_paradigm_files = os.path.join(path_to_functional, paradigm_dir)
path_to_trace_structs = os.path.join(acquisition_dir, 'Traces', roi_method, 'Parsed')


# Load stim trace structs:
print "Loading parsed traces..."
currchannel = "Channel%02d" % int(selected_channel)
stimtrace_fns = os.listdir(path_to_trace_structs)
stimtrace_fns = sorted([f for f in stimtrace_fns if 'stimtraces' in f and currchannel in f and f.endswith('.pkl')], key=natural_keys)
stimtrace_fn = stimtrace_fns[curr_slice_idx]
with open(os.path.join(path_to_trace_structs, stimtrace_fn), 'rb') as f:
    stimtraces = pkl.load(f)


stimlist = sorted(stimtraces.keys(), key=natural_keys)
nstimuli = len(stimlist)

    
# ---------------------------------------------------------------------------
# PLOTTING:
# ----------------------------------------------------------------------------

colormap = plt.get_cmap(cmaptype)

if color_by_roi:
    colorvals = colormap(np.linspace(0, 1, nrois)) #get_spaced_colors(nrois)
else:
    colorvals = colormap(np.linspace(0, 1, nstimuli)) #get_spaced_colors(nstimuli)

colorvals255 = [c[0:-1]*255 for c in colorvals]
#colorvals = np.true_divide(colorvals255, 255.)
#print len(colorvals255)


if plot_traces:
    fig = plt.figure(figsize=(nstimuli,int(len(rois_to_plot))))
    gs = gridspec.GridSpec(len(rois_to_plot), 1) #, height_ratios=[1,1,1,1]) 
    gs.update(wspace=0.01, hspace=0.01)
    
    for ridx,roi in enumerate(rois_to_plot): #np.arange(0, nrois, 2): # range(plot_rois): # nrois
        #rowindex = roi + roi*nstimuli
        print "plotting ROI:", roi
        plt.subplot(gs[ridx])
        #plt.axis('off')
        ax = plt.gca()
        print colorvals[roi] 
        currcolor = colorvals[roi]
        for stimnum,stim in enumerate(stimlist):
            #dfs = traces[curr_roi][stim] #[:, roi, :]
            #raw = stimtraces[stim]['traces'][:, :, roi]
            ntrialstmp = len(stimtraces[stim]['traces'])
            nframestmp = min([stimtraces[stim]['traces'][i].shape[0] for i in range(len(stimtraces[stim]['traces']))])
            raw = np.empty((ntrialstmp, nframestmp))
            for trialnum in range(ntrialstmp):
                raw[trialnum, :] = stimtraces[stim]['traces'][trialnum][0:nframestmp, roi].T
                #print raw.shape
            #avg = np.mean(raw, axis=0)
            xvals = np.arange(0, raw.shape[1]) + stimnum*spacing
            #xvals = np.tile(np.arange(0, raw.shape[1]), (raw.shape[0], 1))
            ntrials = raw.shape[0]
            nframes_in_trial = raw.shape[1]
            #print "ntrials: %i, nframes in trial: %i" % (ntrials, nframes_in_trial)
            
            curr_dfs = np.empty((ntrials, nframes_in_trial))
            for trial in range(ntrials):
                if custom_mw is True:
                    frame_on = stimtraces[stim]['frames_stim_on'][trial][0]
                    #frame_on = int(frames_iti)+1 #stimtraces[stim]['frames_stim_on'][trial][0]
                else:
                    frame_on = int(nframes_iti_pre)+1 #stimtraces[stim]['frames_stim_on'][trial][0]

                baseline = np.mean(raw[trial, 0:frame_on])
                #print "baseline:", baseline
		df = (raw[trial,:] - baseline) / baseline
                if roi==0:
		    print baseline
                #print stim, trial
		curr_dfs[trial,:] = df
                if color_by_roi:
                    plt.plot(xvals, df, color=currcolor, alpha=trial_alpha, linewidth=trial_width)
                else:
                    plt.plot(xvals, df, color=colorvals[stimnum], alpha=trial_alpha, linewidth=trial_width)

            # Plot average:
            avg = np.mean(curr_dfs, axis=0) 
            if color_by_roi:
                plt.plot(xvals, avg, color=currcolor, alpha=1, linewidth=1.2)
            else:
                plt.plot(xvals, avg, color=colorvals[stimnum], alpha=1, linewidth=1.2)

            if custom_mw is True:
                stim_frames = xvals[0] + stimtraces[stim]['frames_stim_on'][trial] #frames_stim_on[stim][trial]
                # start_fr = int(frames_iti) + 1
                # stim_frames = xvals[0] + [start_fr, start_fr+nframes_on]
            else:
                #stim_frames = xvals[0] + stimtraces[stim]['frames_stim_on'][trial] #frames_stim_on[stim][trial]
                # start_fr = int(nframes_iti_pre) + 1
                # stim_frames = xvals[0] + [start_fr, start_fr+nframes_on-1]
                on_fr_idx = stimtraces[stim]['frames'][trial].index(stimtraces[stim]['frames_stim_on'][trial][0])
                nframes_on = (stimtraces[stim]['frames_stim_on'][trial][1] - stimtraces[stim]['frames_stim_on'][trial][0] + 1)
                off_fr_idx = on_fr_idx + nframes_on - 1
                stim_frames = xvals[0] + [on_fr_idx, off_fr_idx]
                
            plt.plot(stim_frames, np.ones((2,))*stim_offset, color='k')

        if ridx<len(rois_to_plot)-1:
            #sns.despine(bottom=True)
            plt.axis('off')
            plt.ylabel(str(roi))
        else:
            ax.axes.get_yaxis().set_visible(False)
            ax.axes.get_xaxis().set_ticks([])
            plt.yticks([0, 1])

        plt.ylim([ylim_min, ylim_max])

    #fig.tight_layout()
    sns.despine(bottom=True, offset=.5, trim=True)

    figname = 'traces_by_stim_per_roi_slice%i%s.png' % (curr_slice_idx, sort_name)
    plt.savefig(os.path.join(figdir, figname), bbox_inches='tight', pad=0)

    plt.show()

# PLOT ROIs:
img = np.copy(avgimg)

plt.figure()
# plt.imshow(img)
#img = exposure.rescale_intensity(avgimg, in_range=(avgimg.min(), avgimg.max()))
img = np.copy(avgimg)
factor = 1
imgnorm = np.true_divide((img - img.min()), factor*(img.max()-img.min()))
imgnorm[imgnorm<backgroundoffset] += 0.15

alpha = 0.9 #0.8 #1 #0.8 #0.5 #0.99 #0.8
nr,nc = imgnorm.shape
color_mask = np.zeros((nr, nc, 3))
#color_mask = np.dstack((imgnorm, imgnorm, imgnorm)) 
for roi in rois_to_plot:
    color_mask[currmasks[:,:,roi]==1] = colorvals[roi][0:3]

# Construct RGB version of grey-level image
img_color = np.dstack((imgnorm, imgnorm, imgnorm))

# Convert the input image and color mask to Hue Saturation Value (HSV)
# colorspace
img_hsv = color.rgb2hsv(img_color)
color_mask_hsv = color.rgb2hsv(color_mask)

# Replace the hue and saturation of the original image
# with that of the color mask
img_hsv[..., 0] = color_mask_hsv[..., 0]
img_hsv[..., 1] = color_mask_hsv[..., 1] #* alpha
#img_hsv[..., 2] = 0.5


img_masked = color.hsv2rgb(img_hsv)

plt.figure()
plt.imshow(img_masked, cmap=cmaptype)
plt.axis('off')

figname = 'rois_average_slice%i%s.png' % (curr_slice_idx, sort_name)
plt.savefig(os.path.join(figdir, figname), bbox_inches='tight')



