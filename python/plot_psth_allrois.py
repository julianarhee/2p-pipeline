
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

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]


source = '/nas/volume1/2photon/projects'
experiment = 'scenes'
session = '20171003_JW016'
acquisition = 'FOV1'
functional_dir = 'functional'

curr_file_idx = 2
curr_slice_idx = 20
curr_roi_method = 'blobs_DoG'
plot_traces = True #False

acquisition_dir = os.path.join(source, experiment, session, acquisition)
figdir = os.path.join(acquisition_dir, 'example_figures')

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

slice_names = sorted(masks.keys(), key=natural_keys)
curr_slice_name = slice_names[curr_slice_idx]


# Get FILE list:
average_source = 'Averaged_Slices_Corrected'
signal_channel = 1
average_slice_dir = os.path.join(acquisition_dir, functional_dir, 'DATA', average_source, "Channel{:02d}".format(signal_channel))
file_names = [f for f in os.listdir(average_slice_dir) if '_vis' not in f]
print "File names:", file_names
nfiles = len(file_names)

# Get AVERAGE slices for current file:
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
path_to_trace_structs = os.path.join(acquisition_dir, 'Trials')


# Load stim trace structs:
print "Loading parsed traces..."
stimtrace_fns = os.listdir(path_to_trace_structs)
stimtrace_fns = sorted([f for f in stimtrace_fns if 'stimtraces' in f], key=natural_keys)
stimtrace_fn = stimtrace_fns[curr_slice_idx]
with open(os.path.join(path_to_trace_structs, stimtrace_fn), 'r') as f:
    stimtraces = load(f)
 
# stimtraces[stim]['traces'] = np.asarray(curr_traces_allrois)
# stimtraces[stim]['frames_stim_on'] = stim_on_frames 
# stimtraces[stim]['ntrials'] = stim_ntrials[stim]
# stimtraces[stim]['nrois'] = nrois

stimlist = sorted(stimtraces.keys(), key=natural_keys)
nstimuli = len(stimlist)
nrois = stimtraces[stimlist[0]]['nrois']


# Load roi traces:
# roitrace_fns = os.listdir(path_to_trace_structs)
# roitrace_fns = sorted([f for f in roitrace_fns if 'roitraces' in f], key=natural_keys)
# roitrace_fn = roitrace_fns[curr_slice_idx]
# with open(os.path.join(path_to_trace_structs, roitrace_fn), 'r') as f:
#     traces = load(f)
#  
# roilist = sorted(traces.keys(), key=natural_keys)
# nrois = len(roilist)
# stimlist = sorted(traces['1'].keys(), key=natural_keys)
# nstimuli = len(stimlist)
# 
# ---------------------------------------------------------------------------
# PLOTTING:
# ----------------------------------------------------------------------------

def get_spaced_colors(n):
    max_value = 16581375 #255**3
    interval = int(max_value / n)
    colors = [hex(I)[2:].zfill(6) for I in range(1, max_value, interval)]

    return [(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors]

color_by_roi = True

spacing = 25
cmaptype = 'rainbow'
colormap = plt.get_cmap(cmaptype)

if color_by_roi:
    colorvals = colormap(np.linspace(0, 1, nrois)) #get_spaced_colors(nrois)
else:
    colorvals = colormap(np.linspace(0, 1, nstimuli)) #get_spaced_colors(nstimuli)

colorvals255 = [c[0:-1]*255 for c in colorvals]
#colorvals = np.true_divide(colorvals255, 255.)
#print len(colorvals255)
roi_interval = 10
plot_rois = np.arange(0, nrois, roi_interval) #int(nrois/2)


if plot_traces:
    fig = plt.figure(figsize=(nstimuli,int(len(plot_rois)/2)))
    gs = gridspec.GridSpec(len(plot_rois), 1) #, height_ratios=[1,1,1,1]) 
    gs.update(wspace=0.01, hspace=0.01)
    for ridx,roi in enumerate(plot_rois): #np.arange(0, nrois, 2): # range(plot_rois): # nrois
	#rowindex = roi + roi*nstimuli
	print "plotting ROI:", roi
	plt.subplot(gs[ridx])
	#plt.axis('off')
	#ax = plt.gca()
	print colorvals[roi] 
	currcolor = colorvals[roi]
	for stimnum,stim in enumerate(stimlist):
	    #dfs = traces[curr_roi][stim] #[:, roi, :]
	    raw = stimtraces[stim]['traces'][:, :, roi]
	    #avg = np.mean(raw, axis=0)
	    xvals = np.arange(0, raw.shape[1]) + stimnum*spacing
	    #xvals = np.tile(np.arange(0, raw.shape[1]), (raw.shape[0], 1))
	    ntrials = raw.shape[0]
	    nframes_in_trial = raw.shape[1]
	    curr_dfs = np.empty((ntrials, nframes_in_trial))
	    for trial in range(ntrials):
		frame_on = stimtraces[stim]['frames_stim_on'][trial][0]
		baseline = np.mean(raw[trial, 0:frame_on])
		df = (raw[trial,:] - baseline) / baseline
		curr_dfs[trial,:] = df
		if color_by_roi:
		    plt.plot(xvals, df, color=currcolor, alpha=1, linewidth=0.2)
		else:
		    plt.plot(xvals, df, color=colorvals[stimnum], alpha=1, linewidth=0.2)
		
		stim_frames = xvals[0] + stimtraces[stim]['frames_stim_on'][trial] #frames_stim_on[stim][trial]
		
		plt.plot(stim_frames, np.ones((2,))*-0.5, color='k')
			
	    # Plot average:
	    avg = np.mean(curr_dfs, axis=0) 
	    if color_by_roi:
		plt.plot(xvals, avg, color=currcolor, alpha=1, linewidth=1.2)
	    else:
		plt.plot(xvals, avg, color=colorvals[stimnum], alpha=1, linewidth=1.2)
	    
	if ridx<len(plot_rois)-1:
	    #sns.despine(bottom=True)
	    plt.axis('off')
	    plt.ylabel(str(roi))
	else:
	    #plt.axes.get_yaxis().set_visible(True)
	    #ax.axes.get_xaxis().set_ticks([])
	    plt.yticks([0, 1])

	plt.ylim([-1, 1])

    #fig.tight_layout()
    sns.despine(bottom=True, offset=.5, trim=True)

    figname = 'all_rois_traces_by_stim.png'
    plt.savefig(os.path.join(figdir, figname), bbox_inches='tight', pad=0)
    #plt.show()



# PLOT ROIs:
img = np.copy(avgimg)

plt.figure()
# plt.imshow(img)
#img = exposure.rescale_intensity(avgimg, in_range=(avgimg.min(), avgimg.max()))
img = np.copy(avgimg)
factor = 1
imgnorm = np.true_divide((img - img.min()), factor*(img.max()-img.min()))
#imgnorm += (1./factor) #0.25

imgnorm[imgnorm<0.3] += 0.15


# img_uint = img_as_ubyte(imgnorm)
# img_float = img_as_float(img_uint)

#label_masks = np.zeros((curr_masks.shape[0], curr_masks.shape[1]))
#print label_masks.shape
# 
#imgnorm = np.random.rand(nr, nc)
alpha = 0.9 #0.8 #1 #0.8 #0.5 #0.99 #0.8
nr,nc = imgnorm.shape
color_mask = np.zeros((nr, nc, 3))
#color_mask = np.dstack((imgnorm, imgnorm, imgnorm)) 
for roi in plot_rois:
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

figname = 'all_rois_average_slice.png'
plt.savefig(os.path.join(figdir, figname), bbox_inches='tight')
# 

# 
# alphaval = 0.8 #0.5 #0.99 #0.8
# nr,nc = imgnorm.shape
# color_mask = np.ones((nr, nc, 3))*np.nan
# for roi in plot_rois:
#     color_mask[curr_masks[:,:,roi]==1] = colorvals[roi][0:3]
# 
# img_color = np.dstack((imgnorm, imgnorm, imgnorm))
# #img_color = color.gray2rgb(imgnorm)
# img_hsv = color.rgb2hsv(img_color)
# color_mask_hsv = color.rgb2hsv(color_mask)
# 
# img_hsv[...,0] = color_mask_hsv[..., 0]
# img_hsv[..., 1] = color_mask_hsv[..., 1] * alphaval
# 
# img_masked = color.hsv2rgb(img_hsv)
# 
# img_color = np.dstack( 
# plt.figure()
# plt.imshow(img_masked)
# plt.axis('off')
# 
# roi_idx = 1
# for roi in plot_rois:
#     #label_masks[curr_masks[:,:,roi]==1] = int(roi) #int(roi_idx)
#     rgb_masks[curr_masks[:,:,roi]==1, 
#     roi_idx += 1
# 
# plt.axis('off')
# 
# plt.figure()
# # plt.imshow(img)
# imgnorm = np.true_divide((img - img.min()), (img.max()-img.min()))
# #plt.imshow(imgnorm, cmap='gray'); plt.colorbar()
# #print colorvals255
# plt.imshow(skimage.color.label2rgb(label_masks, image=imgnorm, alpha=0.5, colors=colorvals255[roi], bg_label=0)) #, cmap=cmaptype)
# plt.axis('off')
# 
figname = 'all_rois_average_slice.png'
plt.savefig(os.path.join(figdir, figname), bbox_inches='tight')
#plt.show()


