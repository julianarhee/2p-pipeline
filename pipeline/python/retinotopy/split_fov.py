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
from skimage.morphology import closing, square, disk
from skimage.color import label2rgb
import matplotlib.patches as mpatches

import skimage.measure

from matplotlib import cm


import imutils


from pipeline.python.retinotopy import segmentation as seg
from pipeline.python.rois.utils import get_roi_contours, uint16_to_RGB, plot_roi_contours

#%%

def load_retino_id(run_dir, retino_id):
    rdict_path = glob.glob(os.path.join(run_dir, 'retino_analysis', 'analysisids*.json'))[0]
    with open(rdict_path, 'r') as f: rdict = json.load(f)
    retinoID = rdict[retino_id]
    return retinoID
#
#
#def get_roi_contours(roi_masks, roi_axis=0):
#    
#    cnts = []
#    nrois = roi_masks.shape[roi_axis]
#    for ridx in range(nrois):
#        if roi_axis == 0:
#            im = np.copy(roi_masks[ridx, :, :])
#        else:
#            im = np.copy(roi_masks[:, :, ridx])
#        im[im>0] = 1
#        im[im==0] = np.nan #1
#        im = im.astype('uint8')
#        edged = cv2.Canny(im, 0, 0.9)
#        tmp_cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#        tmp_cnts = tmp_cnts[0] if imutils.is_cv2() else tmp_cnts[1]
#        cnts.append((ridx, tmp_cnts[0]))
#    print "Created %i contours for rois." % len(cnts)
#
#    return cnts
#
#def uint16_to_RGB(img):
#    im = img.astype(np.float64)/img.max()
#    im = 255 * im
#    im = im.astype(np.uint8)
#    rgb = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
#    return rgb
#
#def plot_roi_contours(zproj, cnts, clip_limit=0.01, ax=None, 
#                          roi_highlight = [],
#                          label_all=False, roi_color_default=(127, 127, 127),
#                          label_highlight=False, roi_color_highlight=(0, 255, 0),
#                          thickness=1, fontsize=12):
#
#
#    if ax is None:
#        fig, ax = pl.subplots(1, figsize=(10,10))
#
#
#    #clip_limit=0.02
#    # Create ZPROJ img to draw on:
#    refRGB = uint16_to_RGB(zproj)        
##    p2, p98 = np.percentile(refRGB, (1, 99))
##    img_rescale = exposure.rescale_intensity(refRGB, in_range=(p2, p98))
#    im_adapthist = exposure.equalize_adapthist(refRGB, clip_limit=clip_limit)
#    im_adapthist *= 256
#    im_adapthist= im_adapthist.astype('uint8')
#    ax.imshow(im_adapthist) #pl.figure(); pl.imshow(refRGB) # cmap='gray')
#
#    orig = im_adapthist.copy()
#
#    # loop over the contours individually
#    for rid, cnt in enumerate(cnts):
#        contour = np.squeeze(cnt[-1])
#
#        # draw the contours on the image
#        orig = refRGB.copy()
#
#        if len(roi_highlight) > 0 and rid in roi_highlight:
#            col255 = tuple([cval/255. for cval in roi_color_highlight])
#            if label_highlight:
#                ax.text(contour[-1, 0], contour[-1, 1], str(rid+1), color='gray', fontsize=fontsize)
#        else:
#            col255 = tuple([cval/255. for cval in roi_color_default])
#            if label_all:
#                ax.text(contour[-1, 0], contour[-1, 1], str(rid+1), color='gray', fontsize=fontsize)
#            
#        #cv2.drawContours(orig, cnt, -1, col255, thickness)
#        ax.plot(contour[:, 0], contour[:, 1], color=col255)
#
#
#        ax.imshow(orig)


    
#%%
rootdir = '/n/coxfs01/2p-data'

# Combine different conditions of the SAME acquisition:
#animalid = 'JC015'
#session = '20180919'
#acquisition = 'FOV1_zoom2p0x'
#retino_run = 'retino_run1'

animalid = 'JC015'
session = '20180925'
acquisition = 'FOV1_zoom2p0x'
retino_run = 'retino_run1'

#use_azimuth = True
#use_single_ref = True
#retino_file_ix = 0

cmap = cm.Spectral_r

datestr = datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")

visualareas_fpath = os.path.join(rootdir, animalid, session, acquisition, 'visual_areas', 'visual_areas_%s.pkl' % datestr)

if os.path.exists(visualareas_fpath):
    with open(visualareas_fpath, 'rb') as f:
        fov = pkl.load(f)

#%%
else:
    
    #%%CREATE NEW:
    
    datestr = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    
    
    fov = seg.Segmentations(animalid, session, acquisition, retino_run)
    pix_phasemap = fov.get_phase_data(analysis_type='pixels', use_azimuth=True, use_single_ref=True, retino_file_ix=1)
    roi_phasemap = fov.get_phase_data(analysis_type='rois', use_azimuth=True, use_single_ref=True, retino_file_ix=1)
    
    
    roi_phase_mask = np.copy(roi_phasemap)
    roi_phase_mask[roi_phasemap==0] = np.nan
    #pl.figure()
    #pl.imshow(FOV, cmap='gray')
    #pl.imshow(roi_phase_mask, cmap=cmap, vmin=0, vmax=np.pi*2)
    
    # 1.  Processing retinotopy map image:
    # -----------------------------------------------------------------------------
    kernel_size=21
    morph_kernel=5
    morph_iterations=2
    
    processing_params = {'kernel_size': kernel_size,
                         'morph_kernel': morph_kernel,
                         'morph_iterations': morph_iterations}
    
    kernel_type, processing_params = fov.test_filter_types(pix_phasemap, processing_params=None)
    
    
    mask_template, map_thr, split_intervals = fov.segment_fov(pix_phasemap, nsplits=20,
                                                              kernel_type=kernel_type, 
                                                              processing_params=processing_params)
    




#%%

FOV = fov.get_fov_image(fov.source.run)

retinoID = load_retino_id(os.path.join(fov.source.rootdir, fov.source.animalid, fov.source.session, fov.source.acquisition, fov.source.run), fov.source.retinoID_rois)
roi_masks = fov.get_roi_masks(retinoID)
nrois = roi_masks.shape[-1]
roi_contours = get_roi_contours(roi_masks, roi_axis=-1)

#%%
# 2.  Select regions from segmentation:
# -----------------------------------------------------------------------------
labeled_image, region_id, region_mask = fov.select_visual_area(mask_template, datestr=datestr)
regions = regionprops(labeled_image)

fig, axes = pl.subplots(2,2, figsize=(15,10))
axes.flat[0].imshow(region_mask);

plot_roi_contours(FOV, roi_contours, ax=axes.flat[1], thickness=0.1, label_all=False, clip_limit=0.02)

axes.flat[2].imshow(FOV, cmap='gray')
axes.flat[2].imshow(roi_phase_mask, cmap=cmap, vmin=0, vmax=np.pi*2)


# Mask ROIs with area mask:
region_mask_copy = np.copy(region_mask)
region_mask_copy[region_mask==0] = np.nan

included_rois = [ri for ri in range(nrois) if ((roi_masks[:, :, ri] + region_mask_copy) > 1).any()]
plot_roi_contours(FOV, roi_contours, ax=axes.flat[3], thickness=0.1, 
                      roi_highlight = included_rois,
                      roi_color_default=(255,255,255),
                      roi_color_highlight=(255,0,0),
                      label_highlight=True,
                      fontsize=8)

axes.flat[3].imshow(region_mask_copy, alpha=0.1, cmap='Blues')

for ax in axes.flat:
    ax.axis('off')

pl.tight_layout()

pl.draw()
#pl.show(block=False)
pl.pause(3.0)

region_name = raw_input("Enter name of visual area: ")
selected_area = regions[int(region_id-1)]
axes.flat[0].text(selected_area.centroid[1], selected_area.centroid[0], '%s' % region_name, fontsize=24, color='w')

pl.savefig(os.path.join(fov.output_dir, 'figures', 'segmented_%s_%s.png' % (region_name, datestr)))

pl.close()

#%%

fov.save_visual_area(selected_area, region_name, region_id, region_mask, datestr)

#%%

with open(visualareas_fpath, 'wb') as f:
    pkl.dump(fov, f, protocol=pkl.HIGHEST_PROTOCOL)
             
             

