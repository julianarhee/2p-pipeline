#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 19:30:49 2018

@author: juliana
"""

import os
import glob
import h5py
import cv2
import json
import scipy.ndimage
import scipy.signal
import datetime

import numpy as np
import cPickle as pkl
import pandas as pd
import pylab as pl
import seaborn as sns
import scipy.optimize as opt
import tifffile as tf

from skimage import exposure

from mpl_toolkits.axes_grid1 import make_axes_locatable
from pipeline.python.utils import replace_root, label_figure

from PIL import Image
from scipy.fftpack import fft2, ifft2

from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops, find_contours
from skimage.morphology import closing, square
from skimage.color import label2rgb
import matplotlib.patches as mpatches

import skimage.measure

from matplotlib import cm


import imutils

def get_roi_contours(roi_masks, roi_axis=0):
    
    cnts = []
    nrois = roi_masks.shape[roi_axis]
    for ridx in range(nrois):
        if roi_axis == 0:
            im = np.copy(roi_masks[ridx, :, :])
        else:
            im = np.copy(roi_masks[:, :, ridx])
        im[im>0] = 1
        im[im==0] = np.nan #1
        im = im.astype('uint8')
        edged = cv2.Canny(im, 0, 0.9)
        tmp_cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        tmp_cnts = tmp_cnts[0] if imutils.is_cv2() else tmp_cnts[1]
        cnts.append((ridx, tmp_cnts[0]))
    print "Created %i contours for rois." % len(cnts)

    return cnts

def uint16_to_RGB(img):
    im = img.astype(np.float64)/img.max()
    im = 255 * im
    im = im.astype(np.uint8)
    rgb = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    return rgb

def plot_roi_contours(zproj, cnts, clip_limit=0.01, ax=None, 
                          roi_highlight = [],
                          label_all=False, roi_color_default=(127, 127, 127),
                          label_highlight=False, roi_color_highlight=(0, 255, 0),
                          thickness=1, fontsize=12):


    if ax is None:
        fig, ax = pl.subplots(1, figsize=(10,10))


    #clip_limit=0.02
    # Create ZPROJ img to draw on:
    refRGB = uint16_to_RGB(zproj)        
#    p2, p98 = np.percentile(refRGB, (1, 99))
#    img_rescale = exposure.rescale_intensity(refRGB, in_range=(p2, p98))
    im_adapthist = exposure.equalize_adapthist(refRGB, clip_limit=clip_limit)
    im_adapthist *= 256
    im_adapthist= im_adapthist.astype('uint8')
    ax.imshow(im_adapthist) #pl.figure(); pl.imshow(refRGB) # cmap='gray')

    orig = im_adapthist.copy()

    # loop over the contours individually
    for rid, cnt in enumerate(cnts):
        contour = np.squeeze(cnt[-1])

        # draw the contours on the image
        orig = refRGB.copy()

        if len(roi_highlight) > 0 and rid in roi_highlight:
            col255 = tuple([cval/255. for cval in roi_color_highlight])
            if label_highlight:
                ax.text(contour[-1, 0], contour[-1, 1], str(rid+1), color='gray', fontsize=fontsize)
        else:
            col255 = tuple([cval/255. for cval in roi_color_default])
            if label_all:
                ax.text(contour[-1, 0], contour[-1, 1], str(rid+1), color='gray', fontsize=fontsize)
            
        #cv2.drawContours(orig, cnt, -1, col255, thickness)
        ax.plot(contour[:, 0], contour[:, 1], color=col255)


        ax.imshow(orig)

    #pl.axis('off')
    
    
    

#%%
def fftconvolve2d(x, y):
    # This assumes y is "smaller" than x.
    f2 = ifft2(fft2(x, shape=x.shape) * fft2(y, shape=x.shape)).real
    f2 = np.roll(f2, (-((y.shape[0] - 1)//2), -((y.shape[1] - 1)//2)), axis=(0, 1))
    return f2



def convert_range(img, min_new=0.0, max_new=255.0):
    img_new = (img - img.min()) * ((max_new - min_new) / (img.max() - img.min())) + min_new
    return img_new

def smooth_array(inputArray,fwhm):
	szList=np.array([None,None,None,11,None,21,None,27,None,31,None,37,None,43,None,49,None,53,None,59,None,55,None,69,None,79,None,89,None,99])
	sigmaList=np.array([None,None,None,.9,None,1.7,None,2.6,None,3.4,None,4.3,None,5.1,None,6.4,None,6.8,None,7.6,None,8.5,None,9.4,None,10.3,None,11.2,None,12])
	sigma=sigmaList[fwhm]
	sz=szList[fwhm]

	outputArray=cv2.GaussianBlur(inputArray, (sz,sz), sigma, sigma)
	return outputArray

def get_fov_mask(img, max_val, min_val=0):
    
    msk = np.copy(img)
    msk[img > max_val] = np.nan
    msk[img < min_val] = np.nan
    
    return msk


def smooth_fov(img, kernel_type='median', kernel_size=5):

    omax = img.max()
    omin = img.min()
    if img.dtype != 'uint8':
        tmpimg = convert_range(img)
        tmpimg = np.array(Image.fromarray(tmpimg).convert("L"))
    
    if kernel_type == 'median':
        img_smoothed = cv2.medianBlur(tmpimg, kernel_size)
    elif kernel_type == 'gaussian':
        img_smoothed = smooth_array(tmpimg, kernel_size)
    elif kernel_type == 'uniform':
        kernel = np.ones((kernel_size,kernel_size), np.float32)/(kernel_size*kernel_size)
        img_smoothed = cv2.filter2D(tmpimg,-1,kernel)
    
    # Convert range and type back:
    final_img = convert_range(img_smoothed, min_new=omin, max_new=omax)
    
    return final_img
        
def morph_fov(img, kernel_size=5, n_iterations=2):
        
    # noise removal
    kernel = np.ones((kernel_size, kernel_size),np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=n_iterations)
    # sure background area
    dilated = cv2.dilate(opening,kernel,iterations=n_iterations)
    
    return dilated
    

#%%
    

def get_retinorun_info(acquisition_dir, retino_run='retino_run1'):
    print 'Getting paradigm file info'
    paradigm_fpath = glob.glob(os.path.join(acquisition_dir, '%s*' % retino_run, 'paradigm', 'files', '*.json'))[0]
    with open(paradigm_fpath, 'r') as r: mwinfo = json.load(r)
    # pp.pprint(mwinfo)
    
    rep_list = [(k, v['stimuli']['stimulus']) for k,v in mwinfo.items()]
    unique_conditions = np.unique([rep[1] for rep in rep_list])
    conditions = dict((cond, [int(run) for run,config in rep_list if config==cond]) for cond in unique_conditions)
    print conditions
    
    return conditions
    
#%%
rootdir = '/n/coxfs01/2p-data'

# Combine different conditions of the SAME acquisition:
animalid = 'JC015'
session = '20180925'
acquisition = 'FOV1_zoom2p0x'
use_azimuth = True
use_single_ref = False


# Get data sources:
# -----------------------------------------------------------------------------
acquisition_dir = os.path.join(rootdir, animalid, session, acquisition)

retino_id_fpath = glob.glob(os.path.join(acquisition_dir, 'retino*', 'retino_analysis', 'analysisids_*.json'))[0]

with open(retino_id_fpath, 'r') as f:
    retinodict = json.load(f)
pixel_id = [k for k, analysis_info in retinodict.items() if analysis_info['PARAMS']['roi_type'] == 'pixels']
assert len(pixel_id) > 0, "NO pixel analysis found in retino run: %s" % retino_id_fpath.split('/retino_analysis')[0]
pixel_id = pixel_id[0]
retinoID = retinodict[pixel_id]

retino_run = os.path.split(retinoID['DST'].split('/retino_analysis')[0])[-1]
retino_id = '%s_%s' % (retino_run, pixel_id)
print " --- using %s" % retino_id

# Create output dir:
# -----------------------------------------------------------------------------
area_dir = os.path.join(acquisition_dir, 'visual_areas')
if not os.path.exists(os.path.join(area_dir, 'figures')): os.makedirs(os.path.join(area_dir, 'figures'))
print "*** Saving parsing output to:", area_dir

# Get run / frame info:
runinfo_fpath = glob.glob(os.path.join(acquisition_dir, retino_run, 'retino*.json'))[0]
with open(runinfo_fpath, 'r') as f: runinfo = json.load(f)
d1 = runinfo['lines_per_frame']
d2 = runinfo['pixels_per_line']
print " --- original frame size: (%i, %i)" % (d1, d2)

retino_dir = glob.glob(os.path.join(acquisition_dir, 'retino*', 'retino_analysis', '%s*' % pixel_id))[0]
retino_files = glob.glob(os.path.join(retino_dir, 'files', '*.h5'))
print "Found %i analysis files." % len(retino_files)

conditions = get_retinorun_info(acquisition_dir, retino_run=retino_run)
if use_azimuth:
    cond_runs = [c-1 for c in conditions['right']]
else:
    cond_runs = [c-1 for c in conditions['top']]
    
scale_factor = int(retinoID['PARAMS']['downsample_factor'])

 
#%%
cmap = cm.Spectral_r

# Load data:
# -----------------------------------------------------------------------------
if use_single_ref:
    curr_fpath = retino_files[0]
    ret = h5py.File(curr_fpath, 'r')
    
    tmp_phasemap = ret['phase_array'][:].reshape((d1/scale_factor, d2/scale_factor))
    phasemap=np.copy(tmp_phasemap)	
    phasemap[tmp_phasemap<0]=-tmp_phasemap[tmp_phasemap<0]
    phasemap[tmp_phasemap>0]=(2*np.pi)-tmp_phasemap[tmp_phasemap>0]

else:
    
    P = []; #cond_runs = np.array([0, 4, 5])
    for curr_fpath in [fi for i, fi in enumerate(retino_files) if i in cond_runs]:
        ret = h5py.File(curr_fpath, 'r')
        tmp_phasemap = ret['phase_array'][:].reshape((d1/scale_factor, d2/scale_factor))
        phasemap=np.copy(tmp_phasemap)	
        phasemap[tmp_phasemap<0]=-tmp_phasemap[tmp_phasemap<0]
        phasemap[tmp_phasemap>0]=(2*np.pi)-tmp_phasemap[tmp_phasemap>0]
        P.append(phasemap)
    phasemap = np.mean(np.dstack(P), axis=-1)
    #pl.imshow(P, cmap=cmap)



# 1.  Rescale image to original size of image:    
# --------------------------------------------------
scaled_map = scipy.ndimage.zoom(phasemap, 2, order=0)
phasemap = np.copy(scaled_map)

#
## 2. Smooth the image:
#kernel_size=21
#
#smooth_gaus = smooth_fov(phasemap, kernel_type='gaussian', kernel_size=kernel_size)
#smooth_med = smooth_fov(phasemap, kernel_type='median', kernel_size=kernel_size)
#smooth_uni = smooth_fov(phasemap, kernel_type='uniform', kernel_size=kernel_size)
#
#
## 3.  Open/dilate image to get rid of small corners:
## --------------------------------------------------
#morph_kernel = 5
#pmap_orig = morph_fov(phasemap, kernel_size=morph_kernel)
#pmap_gaus = morph_fov(smooth_gaus, kernel_size=morph_kernel)
#pmap_med = morph_fov(smooth_med, kernel_size=morph_kernel)
#pmap_uni = morph_fov(smooth_uni, kernel_size=morph_kernel)
#
#
## Plot processing output:
## --------------------------------------------------
#pl.ion()
#f2, ax2 = pl.subplots(2,3, figsize=(15,5)) #pl.figure();
#ax2.flat[0].imshow(smooth_gaus, cmap=cmap); ax2.flat[0].set_title('gaussian')
#ax2.flat[1].imshow(smooth_med, cmap=cmap); ax2.flat[1].set_title('median')
#ax2.flat[2].imshow(smooth_uni, cmap=cmap); ax2.flat[2].set_title('uniform')
#
#ax2.flat[3].imshow(pmap_gaus, cmap=cmap); #ax2[0].set_title('gaussian')
#ax2.flat[4].imshow(pmap_med, cmap=cmap); #ax2[1].set_title('median')
#ax2.flat[5].imshow(pmap_uni, cmap=cmap); #ax2[2].set_title('uniform')

#kernel_size=10
#kernel = np.ones((kernel_size,kernel_size), np.float32)/(kernel_size*kernel_size)
#
#pmap = np.copy(phasemap)
#f2 = ifft2(fft2(pmap, shape=pmap.shape) * fft2(kernel, shape=pmap.shape)).real
#print f2.shape
#pl.figure(); pl.subplot(1,2,1); pl.imshow(f2, cmap=cmap); pl.colorbar()
#
#f3 = cv2.filter2D(pmap,-1,kernel)
#pl.subplot(1,2,2); pl.imshow(f3, cmap=cmap)




#%%
# apply threshold
#thresh = threshold_otsu(pmap_med)
#bw = closing(pmap_med > thresh, square(20))

pl.ion()
fig, axes = pl.subplots(2,4, figsize=(20,15))


# 2.  Smooth the image:
# --------------------------------------------------
kernel_size=21
smooth_gaus = smooth_fov(phasemap, kernel_type='gaussian', kernel_size=kernel_size)
smooth_med = smooth_fov(phasemap, kernel_type='median', kernel_size=kernel_size)
smooth_uni = smooth_fov(phasemap, kernel_type='uniform', kernel_size=kernel_size)

axes.flat[0].imshow(phasemap, cmap=cmap); axes.flat[0].set_title('original')
axes.flat[1].imshow(smooth_gaus, cmap=cmap); axes.flat[1].set_title('gaussian (%i)' % (kernel_size))
axes.flat[2].imshow(smooth_med, cmap=cmap); axes.flat[2].set_title('median (%i)' % (kernel_size))
im = axes.flat[3].imshow(smooth_uni, cmap=cmap); axes.flat[3].set_title('uniform (%i)' % (kernel_size))

divider = make_axes_locatable(axes.flat[3])
cax = divider.append_axes("right", size="5%", pad=0.05)
pl.colorbar(im, cax=cax)

# 3.  Open/dilate image to get rid of small corners:
# --------------------------------------------------
morph_kernel = 5
morph_iterations = 2
pmap_orig = morph_fov(phasemap, kernel_size=morph_kernel, n_iterations=morph_iterations)
pmap_gaus = morph_fov(smooth_gaus, kernel_size=morph_kernel, n_iterations=morph_iterations)
pmap_med = morph_fov(smooth_med, kernel_size=morph_kernel, n_iterations=morph_iterations)
pmap_uni = morph_fov(smooth_uni, kernel_size=morph_kernel, n_iterations=morph_iterations)

axes.flat[4].imshow(pmap_orig, cmap=cmap); #ax2[0].set_title('gaussian')
axes.flat[5].imshow(pmap_gaus, cmap=cmap); #ax2[0].set_title('gaussian')
axes.flat[6].imshow(pmap_med, cmap=cmap); #ax2[1].set_title('median')
axes.flat[7].imshow(pmap_uni, cmap=cmap); #ax2[2].set_title('uniform')


pl.draw()
#pl.show(block=False)
pl.pause(1.0)

filter_choices = ['gaussian', 'median', 'uniform']
for fi, filter_choice in enumerate(filter_choices):
    print fi, filter_choice
filter_ix = input("Select IDX of filter method to use: ")
kernel_type = filter_choices[filter_ix]
pl.savefig(os.path.join(area_dir, 'figures', 'image_processing_options.png'))

pl.close()

#%%
# 4.  MASKING:
# --------------------------------------------------
split_intervals = np.linspace(0, np.pi*2, 16) # ~ 5 degree steps

sns.palplot(sns.color_palette("Spectral_r", len(split_intervals)))
ax_legend = pl.gca()
for xv,mapval in enumerate(split_intervals):
    ax_legend.text(xv-0.4, 0, '%i: %.2f' % (xv, mapval))

#%
fig, ax = pl.subplots()
im = ax.imshow(pmap_gaus, cmap=cmap)
pl.draw()
pl.pause(1.0)

selected_map_thr = input("Select IDX of map cut off value to use: ")
map_thr = split_intervals[int(selected_map_thr)]


while True:
    mask_ = create_morphological_mask(pmap_gaus, map_thr)
    #ax.imshow(mask_, cmap=cmap)
    im.set_data(mask_, )
    #pl.draw()
    pl.pause(2.0)
    selected_thr_confirm = raw_input("Keep threshold used? Enter <Y> to accept, enter thr IDX to redraw: ")
    if selected_thr_confirm == 'Y':
        pl.close(fig)
        break
    else:
       map_thr = split_intervals[int(selected_thr_confirm)]

pl.close(ax_legend.get_figure())
        
mask_template = np.copy(mask_) + 1

#%%
# Create mask:

#mask_orig = get_fov_mask(pmap_orig, map_thr)
#mask_gaus = get_fov_mask(pmap_gaus, map_thr)
#mask_med = get_fov_mask(pmap_med, map_thr)
#mask_uni = get_fov_mask(pmap_uni, map_thr)

def create_morphological_mask(pmap, map_thr):
    tmp_bw = closing(pmap > map_thr, square(50))
    mask = scipy.ndimage.binary_opening(tmp_bw, structure=np.ones((30,30))).astype(np.int)
    
    return mask

#tmp_bw = closing(pmap_gaus > map_thr, square(50))
#mask_gaus = scipy.ndimage.binary_opening(tmp_bw, structure=np.ones((30,30))).astype(np.int)
#
#tmp_bw = closing(pmap_med > map_thr, square(50))
#mask_med = scipy.ndimage.binary_opening(tmp_bw, structure=np.ones((30,30))).astype(np.int)
#
#tmp_bw = closing(pmap_uni > map_thr, square(50))
#mask_uni = scipy.ndimage.binary_opening(tmp_bw, structure=np.ones((30,30))).astype(np.int)
#
#tmp_bw = closing(pmap_orig > map_thr, square(50))
#mask_orig = scipy.ndimage.binary_opening(tmp_bw, structure=np.ones((30,30))).astype(np.int)

#
#axes.flat[8].imshow(mask_orig, cmap=cmap); #ax2[0].set_title('gaussian')
#axes.flat[9].imshow(mask_gaus, cmap=cmap); #ax2[0].set_title('gaussian')
#axes.flat[10].imshow(mask_med, cmap=cmap); #ax2[1].set_title('median')
#axes.flat[11].imshow(mask_uni, cmap=cmap); #ax2[2].set_title('uniform')


# 
#if kernel_type == 'median':
#    mask_template = mask_med+1
#elif kernel_type == 'gaussian':
#    mask_template = mask_gaus+1
#elif kernel_type == 'uniform':
#    mask_template = mask_uni+1
#else:
#    mask_template = mask_orig+1
    
#%%

# -----------------------------------------------------------------------------
# SEGMENT image:
# -----------------------------------------------------------------------------

# Get contours:
#contours = find_contours(mask_med, 0.5, fully_connected='high')
#
#fig, ax = pl.subplots()
#ax.imshow(pmap_med, alpha=0.5, cmap=cmap)
#for n, contour in enumerate(contours):
#    ax.plot(contour[:, 1], contour[:, 0], linewidth=2)


# Label image regions:

im_labeled, n_labels = skimage.measure.label(
                            mask_template, background=0, return_num=True)
print "N labels:", n_labels

image_label_overlay = label2rgb(im_labeled, image=mask_template) #pmap_med)

fig, ax = pl.subplots(figsize=(10, 6))
ax.imshow(image_label_overlay)

areas = regionprops(im_labeled)
print [a.area for a in areas]
print [a.label for a in areas]
print [a.centroid for a in areas]

colors = ['r', 'g', 'c', 'm']

#testmask = np.copy(im_labeled.astype('float'))
#testmask[im_labeled!=3] = np.nan
#pl.figure(); pl.imshow(testmask)

for ri, region in enumerate(areas): #regionprops(label_image):
    # take regions with large enough areas
    #if region.area >= 200: #and region.euler_number > 0:
        # draw rectangle around segmented coins
    ax.text(region.centroid[1], region.centroid[0], '%i' % region.label, fontsize=24, color='w')


pl.draw()
#pl.show(block=False)
pl.pause(1.0)

region_labels = [region.label for region in areas]
for rlabel in region_labels:
    print rlabel
region_id = input("Select ID of region to keep: ")

datestr = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
pl.savefig(os.path.join(area_dir, 'figures', 'visual_areas_%s.png' % datestr))

pl.close()


#%%  Draw selected visual area:

area_mask = np.copy(im_labeled.astype('float'))
area_mask[im_labeled != region_id] = 0
area_mask[im_labeled == region_id] = 1

# Get ROI analysis info:
# -----------------------------------------------------------------------------
retino_ids = [k for k, analysis_info in retinodict.items() if analysis_info['PARAMS']['roi_type'] != 'pixels']
assert len(retino_ids) > 0, "NO roi analyses found in retino run: %s" % retino_run
for rid, retino_id in enumerate(retino_ids):
    print rid, retino_id
roi_selector = input("Select IDX of ROI-based analysis ID to use: ")
rois_id = retino_ids[roi_selector]
retinoID_rois = retinodict[rois_id]

if rootdir not in retinoID_rois['DST']:
    retinoID_rois['DST'] = replace_root(retinoID_rois['DST'], rootdir, animalid, session)
    
analysis_output_files = glob.glob(os.path.join(retinoID_rois['DST'], 'files', '*.h5'))


# Get Mean image path to visualize:
# -----------------------------------------------------------------------------
if rootdir not in retinoID_rois['PARAMS']['tiff_source']:
    retinoID_rois['PARAMS']['tiff_source'] = replace_root( retinoID_rois['PARAMS']['tiff_source'], rootdir, animalid, session )

mean_img_paths = glob.glob(os.path.join('%s_mean_deinterleaved' % retinoID_rois['PARAMS']['tiff_source'], 'Channel%02d' % retinoID_rois['PARAMS']['signal_channel'], 'File*', '*.tif'))
fovs = []
for fpath in mean_img_paths:
    img = tf.imread(fpath)
    fovs.append(img)
FOV = np.mean(np.dstack(fovs), axis=-1)



# Read in first file to get masks and dims:
# -----------------------------------------------------------------------------
ret = h5py.File(analysis_output_files[0], 'r')
roi_masks = ret['masks'][:]
print roi_masks.shape
scale_factor = int(retinoID_rois['PARAMS']['downsample_factor'])
scaled_masks = [scipy.ndimage.zoom(roi_masks[r, :, :], scale_factor, order=0) for r in range(roi_masks.shape[0])]
roi_masks = np.dstack(scaled_masks) # roi ix is now -1

all_rois = np.sum(roi_masks, axis=-1)



# Get roi contours:
# -----------------------------------------------------------------------------
roi_contours = get_roi_contours(roi_masks, roi_axis=-1)



# Get average retinotopy ROI phase map:
# -----------------------------------------------------------------------------
roi_phases = []
for df in analysis_output_files:
    ret = h5py.File(df, 'r')
    tmp_roi_phases = ret['phase_array'][:]

    phases_shifted=np.copy(tmp_roi_phases)	
    phases_shifted[tmp_roi_phases<0]=-tmp_roi_phases[tmp_roi_phases<0]
    phases_shifted[tmp_roi_phases>0]=(2*np.pi)-tmp_roi_phases[tmp_roi_phases>0]
    
    roi_phases.append(phases_shifted)

    
        
roi_phases = np.mean(np.array(roi_phases), axis=0)
nrois = roi_masks.shape[-1]
print "Getting phases for %i rois" % nrois
roi_phase_masks = np.array([roi_masks[:, :, ridx] * roi_phases[ridx] for ridx in range(nrois)])
print roi_phase_masks.shape

all_phases = np.sum(roi_phase_masks, axis=0)

#%%
# PLOT:

fig, axes = pl.subplots(2,2, figsize=(15,10))
axes.flat[0].imshow(area_mask);

plot_roi_contours(FOV, roi_contours, ax=axes.flat[1], thickness=0.1, label_all=False, clip_limit=0.02)

axes.flat[2].imshow(FOV, cmap='gray')
all_phases_mask = np.copy(all_phases)
all_phases_mask[all_rois==0] = np.nan
axes.flat[2].imshow(all_phases_mask, cmap=cmap, vmin=0, vmax=np.pi*2)


# Mask ROIs with area mask:
area_mask_copy = np.copy(area_mask)
area_mask_copy[area_mask==0] = np.nan

included_rois = [ri for ri in range(nrois) if ((roi_masks[:, :, ri] + area_mask) > 1).any()]
plot_roi_contours(FOV, roi_contours, ax=axes.flat[3], thickness=0.1, 
                      roi_highlight = included_rois,
                      roi_color_default=(255,255,255),
                      roi_color_highlight=(255,0,0),
                      label_highlight=True,
                      fontsize=8)

axes.flat[3].imshow(area_mask_copy, alpha=0.1, cmap='Blues')

for ax in axes.flat:
    ax.axis('off')

pl.tight_layout()

pl.draw()
#pl.show(block=False)
pl.pause(3.0)

region_name = raw_input("Enter name of visual area: ")
selected_area = areas[int(filter_ix-1)]
axes.flat[0].text(selected_area.centroid[1], selected_area.centroid[0], '%s' % region_name, fontsize=24, color='w')

pl.savefig(os.path.join(area_dir, 'figures', 'segmented_%s_%s.png' % (region_name, datestr)))

pl.close()

#%%

# Save segmentation results:

AREAS = {'regions': areas,
         'selected_area': selected_area,
         'selected_id': region_id,
         'selected_name': region_name,
         'processing': {'kernel_type': kernel_type,
                        'kernel_size': kernel_size,
                        'morph_kernel': morph_kernel,
                        'morph_iterations': morph_iterations,
                        'mask_template': mask_template},
         'included_rois': included_rois,
         'source': {'retinoID_rois': rois_id,
                    'retinoID_pixels': pixel_id,
                    'run': retino_run,
                    'animalid': animalid,
                    'session': session,
                    'acquisition': acquisition},
         'conditions': conditions,
         'phase_maps': {'pixels': phasemap,
                        'rois': roi_phase_masks},
         'fov_image': FOV
         }

results_fpath = os.path.join(area_dir, 'segmentation_results_%s.pkl' % datestr)    
with open(results_fpath, 'wb') as f:
    pkl.dump(AREAS, f, protocol=pkl.HIGHEST_PROTOCOL)
    
    
    
    
