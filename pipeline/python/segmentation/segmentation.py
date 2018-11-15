#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 19:06:04 2018

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
    elif kernel_type == 'orig':
        img_smoothed = np.copy(img)
    
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
    

def circular_kernel(radius):
    kernel = np.zeros((2*radius+1, 2*radius+1))
    y,x = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = x**2 + y**2 <= radius**2
    kernel[mask] = 1
    return kernel

def create_morphological_mask(pmap, map_thr):
    tmp_bw = closing(pmap > map_thr, disk(20))
    mask = scipy.ndimage.binary_opening(tmp_bw, structure=circular_kernel(3)) #structure=np.ones((5,5))).astype(np.int)
    
    return mask

#%%
    


class struct():
    pass 

class Segmentations():
    def __init__(self, animalid, session, acquisition, run, rootdir='/n/coxfs01/2p-data'):
        self.source =  struct()
        self.source.rootdir = rootdir
        self.source.animalid = animalid
        self.source.session = session
        self.source.acquisition = acquisition
        self.source.run = run
        self.source.retinoID_pixels = None
        self.source.retinoID_rois = None
        self.regions = {}
        self.processing_results = {}
                
        acquisition_dir = os.path.join(self.source.rootdir, self.source.animalid, self.source.session, self.source.acquisition)
        area_dir = os.path.join(acquisition_dir, 'visual_areas')
        if not os.path.exists(os.path.join(area_dir, 'figures')): os.makedirs(os.path.join(area_dir, 'figures'))
        print "*** Saving parsing output to:", area_dir
        self.output_dir = area_dir
    
        # Get run / frame info:
        runinfo_fpath = glob.glob(os.path.join(acquisition_dir, 'retino*', 'retino*.json'))[0]
        with open(runinfo_fpath, 'r') as f: runinfo = json.load(f)
        d1 = runinfo['lines_per_frame']
        d2 = runinfo['pixels_per_line']
        print " --- original frame size: (%i, %i)" % (d1, d2)
        self.source.dims = (d1, d2)

    def get_analysis_params(self, analysis_type='pixels'):
        acquisition_dir = os.path.join(self.source.rootdir, self.source.animalid, self.source.session, self.source.acquisition)
        
        retino_id_fpath = glob.glob(os.path.join(acquisition_dir, 'retino*', 'retino_analysis', 'analysisids_*.json'))[0]
        retino_run = os.path.split(retino_id_fpath.split('/retino_analysis')[0])[-1]
        
        with open(retino_id_fpath, 'r') as f:
            retinodict = json.load(f)
                
        if analysis_type == 'pixels':
            pixel_ids = [k for k, analysis_info in retinodict.items() if analysis_info['PARAMS']['roi_type'] == 'pixels']
            assert len(pixel_ids) > 0, "NO pixel analysis found in retino run: %s" % retino_id_fpath.split('/retino_analysis')[0]
            if len(pixel_ids) > 1:
                print "More than 1 pixel analysis found:"
                for pi, pixid in enumerate(pixel_ids):
                    print pi, pixid
                pi = input("Select IDX of pixel-based retino analysis to use: ")
                retino_analysis_id = pixel_ids[pi]
            else:
                retino_analysis_id = pixel_ids[0]
        else:
            roi_ids = [k for k, analysis_info in retinodict.items() if analysis_info['PARAMS']['roi_type'] != 'pixels']
            assert len(roi_ids) > 0, "NO roi analyses found in retino run: %s" % retino_run
            if len(roi_ids) > 1:
                print "More than 1 pixel analysis found:"
                for ri, retino_id in enumerate(roi_ids):
                    print ri, retino_id
                ri = input("Select IDX of ROI-based analysis ID to use: ")
                retino_analysis_id = roi_ids[ri]
            else:
                retino_analysis_id = roi_ids[0]

        retinoID = retinodict[retino_analysis_id]
        
        retino_id = '%s_%s' % (retino_run, retino_analysis_id)
        print " --- using %s" % retino_id
        
        if analysis_type == 'pixels':
            self.source.retinoID_pixels = retino_analysis_id
        else:
            self.source.retinoID_rois = retino_analysis_id
        
        return retinoID
    

    def get_phase_data(self, analysis_type='pixels', use_azimuth=True, use_single_ref=False, retino_file_ix=0):
        acquisition_dir = os.path.join(self.source.rootdir, self.source.animalid, self.source.session, self.source.acquisition)
        retinoID = self.get_analysis_params(analysis_type=analysis_type)
        retino_run = os.path.split(retinoID['DST'].split('/retino_analysis')[0])[-1]

        conditions = get_retinorun_info(acquisition_dir, retino_run=retino_run)
        self.conditions = conditions
        
        if analysis_type == 'pixels':
            retino_analysis_id = self.source.retinoID_pixels
        else:
            retino_analysis_id = self.source.retinoID_rois
        retino_dir = glob.glob(os.path.join(acquisition_dir, 'retino*', 'retino_analysis', '%s*' % retino_analysis_id))[0]
        retino_files = glob.glob(os.path.join(retino_dir, 'files', '*.h5'))
        print "Found %i analysis files." % len(retino_files)

        if use_azimuth:
            cond_runs = [c-1 for c in conditions['right']]
        else:
            cond_runs = [c-1 for c in conditions['top']]
        scale_factor = int(retinoID['PARAMS']['downsample_factor'])

        # Load data:
        # -----------------------------------------------------------------------------
        if use_single_ref:
            curr_fpath = retino_files[retino_file_ix]
            ret = h5py.File(curr_fpath, 'r')
            if analysis_type == 'pixels':
                tmp_phasemap = ret['phase_array'][:].reshape((self.source.dims[0]/scale_factor, self.source.dims[1]/scale_factor))
            else:
                tmp_phasemap = ret['phase_array'][:] # Each value corresponds to an ROI
            phasemap=np.copy(tmp_phasemap)	
            phasemap[tmp_phasemap<0]=-tmp_phasemap[tmp_phasemap<0]
            phasemap[tmp_phasemap>0]=(2*np.pi)-tmp_phasemap[tmp_phasemap>0]
        
        else:
            
            P = []; #cond_runs = np.array([0, 4, 5])
            for curr_fpath in [fi for i, fi in enumerate(retino_files) if i in cond_runs]:
                ret = h5py.File(curr_fpath, 'r')
                if analysis_type == 'pixels':
                    tmp_phasemap = ret['phase_array'][:].reshape((self.source.dims[0]/scale_factor, self.source.dims[1]/scale_factor))
                else:
                    tmp_phasemap = ret['phase_array'][:] # Each value corresponds to an ROI
                phasemap=np.copy(tmp_phasemap)	
                phasemap[tmp_phasemap<0]=-tmp_phasemap[tmp_phasemap<0]
                phasemap[tmp_phasemap>0]=(2*np.pi)-tmp_phasemap[tmp_phasemap>0]
                P.append(phasemap)
            
            phasemap = np.mean(np.array(P), axis=0)

        # 1.  Rescale image to original size of image:    
        # --------------------------------------------------
        if analysis_type == 'pixels':
            scaled_map = scipy.ndimage.zoom(phasemap, 2, order=0)
            phasemap = np.copy(scaled_map)
        else:
            # Need to reconstruct "map" using masks + phase values:
            roi_masks = self.get_roi_masks(retinoID)
            roi_values = np.copy(phasemap)
            phasemap = self.roi_values_to_map(roi_values, roi_masks)
        
        return phasemap
    
    def get_fov_image(self, run, process_id='processed001', process_type='mcorrected', signal_channel=1):

        # Get Mean image path to visualize:
        # -----------------------------------------------------------------------------
        mean_img_paths = glob.glob(os.path.join(self.source.rootdir, self.source.animalid, 
                                                self.source.session, self.source.acquisition, 
                                                run, 'processed', '%s*' % process_id, '%s*' % process_type, 
                                                'Channel%02d' % signal_channel, 'File*', '*.tif'))
    
        fovs = []
        for fpath in mean_img_paths:
            img = tf.imread(fpath)
            fovs.append(img)
        FOV = np.mean(np.dstack(fovs), axis=-1)
        
        return FOV

    def get_roi_masks(self, retinoID):
            
        if self.source.rootdir not in retinoID['DST']:
            retinoID['DST'] = replace_root(retinoID['DST'], self.source.rootdir, self.source.animalid, self.source.session)
            
        analysis_output_files = glob.glob(os.path.join(retinoID['DST'], 'files', '*.h5'))

        # Read in first file to get masks and dims:
        # -----------------------------------------------------------------------------
        ret = h5py.File(analysis_output_files[0], 'r')
        roi_masks = ret['masks'][:]
        print roi_masks.shape
        scale_factor = int(retinoID['PARAMS']['downsample_factor'])
        scaled_masks = [scipy.ndimage.zoom(roi_masks[r, :, :], scale_factor, order=0) for r in range(roi_masks.shape[0])]
        roi_masks = np.dstack(scaled_masks) # roi ix is now -1
        
        return roi_masks
    
    def roi_values_to_map(self, roi_values, roi_masks):
    
        nrois = roi_masks.shape[-1]
        print "Applying ROI values to masks for %i rois" % nrois
        roi_phase_masks = np.array([roi_masks[:, :, ridx] * roi_values[ridx] for ridx in range(nrois)])
        
        phasemap = np.sum(roi_phase_masks, axis=0)

        return phasemap
    
    def select_visual_area(self, mask_template, datestr=''):
        
        labeled_image, n_labels = skimage.measure.label(
                                    mask_template, background=0, return_num=True)
        print "N labels:", n_labels
        
        image_label_overlay = label2rgb(labeled_image, image=mask_template) #pmap_med)
        
        fig, ax = pl.subplots(figsize=(10, 6))
        ax.imshow(image_label_overlay)
        
        areas = regionprops(labeled_image)
        for ri, region in enumerate(areas): #regionprops(label_image):
            ax.text(region.centroid[1], region.centroid[0], '%i' % region.label, fontsize=24, color='w')

        pl.show(block=False)
        pl.pause(1.0)
        
        region_labels = [region.label for region in areas]
        for rlabel in region_labels:
            print rlabel
        region_id = input("Select ID of region to keep: ")
        
        pl.savefig(os.path.join(self.output_dir, 'figures', 'visual_areas_%s.png' % datestr))
        
        pl.close()
        
        region_mask = np.copy(labeled_image.astype('float'))
        region_mask[labeled_image != region_id] = 0
        region_mask[labeled_image == region_id] = 1
        
        return labeled_image, region_id, region_mask
        

        
    def save_visual_area(self, selected_region, region_name, region_id, region_mask, datestr):
    
        region_props = {'selected_region': selected_region,
                        'region_name': region_name,
                        'region_id': region_id,
                        'region_mask': region_mask,
                        'datestr': datestr}
        
        if any([region_name in key for key in self.regions.keys()]):
            overwrite = raw_input("Visual area -- %s -- already segmented. Overwrite, append, or escape? <Y>/<A>/<ENTER>?" % region_name)
            if overwrite == 'Y':
                self.regions[region_name] = region_props
            elif overwrite == 'A':
                existing_keys = [k for k in self.regions.keys() if region_name in k]
                new_key = '%s_%i' % (region_name, len(existing_keys)+1)
                self.regions[new_key] = region_props
        else:
            self.regions[region_name] = region_props

    
    def segment_fov(self, phasemap, nsplits=20, kernel_type='gaussian', processing_params=None, cmap=cm.Spectral_r):
        processed_map = self.process_phasemap(phasemap, kernel_type=kernel_type, processing_params=processing_params)

        # 4.  MASKING:
        # --------------------------------------------------
        split_intervals = np.linspace(0, np.pi*2, nsplits) # ~ 5 degree steps
        
        sns.palplot(sns.color_palette("Spectral_r", len(split_intervals)))
        ax_legend = pl.gca()
        for xv,mapval in enumerate(split_intervals):
            ax_legend.text(xv-0.4, 0, '%i: %.2f' % (xv, mapval))
        
        #%
        fig, ax = pl.subplots()
        im = ax.imshow(processed_map, cmap=cmap)
        pl.show(block=False)
        pl.pause(1.0)
        
        selected_map_thr = input("Select IDX of map cut off value to use: ")
        map_thr = split_intervals[int(selected_map_thr)]
        
        while True:
            mask_ = create_morphological_mask(processed_map, map_thr)
            #ax.imshow(mask_, cmap=cmap)
            im.set_data(mask_)
            pl.show(block=False)
            pl.pause(1.0)
            selected_thr_confirm = raw_input("Keep threshold used? Enter <Y> to accept, enter thr IDX to redraw: ")
            if selected_thr_confirm == 'Y':
                pl.close(fig)
                break
            else:
               map_thr = split_intervals[int(selected_thr_confirm)]
        
        pl.close(ax_legend.get_figure())
                
        mask_template = np.copy(mask_) + 1
        
        # Update processing info:
        self.processing_results['mask_template'] = mask_template
        self.processing_results['map_threshold_max'] = map_thr
        self.processing_results['split_intervals'] = split_intervals
        
        return mask_template, map_thr, split_intervals



    def process_phasemap(self, phasemap, kernel_type='gaussian', processing_params=None):
        if processing_params is None:
            processing_params = self.default_processing_params()
            
        smoothed_image = smooth_fov(phasemap, kernel_type=kernel_type, 
                                              kernel_size=processing_params['kernel_size'])
        
        processed_map = morph_fov(smoothed_image, kernel_size=processing_params['morph_kernel'],
                                                  n_iterations=processing_params['morph_iterations'])
        
        self.processing_results['kernel_type'] = kernel_type,
        self.processing_results['kernel_size'] = processing_params['kernel_size']
        self.processing_results['morph_kernel'] = processing_params['morph_kernel']
        self.processing_results['morph_iterations'] = processing_params['morph_iterations']
        self.processing_results['processed_map']= processed_map
        
        return processed_map
    
    
    def default_processing_params(self):
        processing_params = {'kernel_size': 21,
                             'morph_kernel': 5,
                             'morph_iterations': 2}
        
        return processing_params

    def test_filter_types(self, phasemap, processing_params=None):
        
        if processing_params is None:
            processing_params = self.default_processing_params()
            
        fig = self.plot_filter_types(phasemap, processing_params=processing_params)
        filter_choices = ['orig', 'gaussian', 'median', 'uniform']
        
        for fi, filter_choice in enumerate(filter_choices):
            print fi, filter_choice
        filter_ix = input("Select IDX of filter method to use: ")
        kernel_type = filter_choices[filter_ix]
        pl.savefig(os.path.join(self.output_dir, 'figures', 'image_processing_options.png'))
        
        pl.close(fig)
    
        return kernel_type, processing_params
    
    def plot_filter_types(self, phasemap, processing_params=None, cmap=cm.Spectral_r):

        if processing_params is None:
            processing_params = self.default_processing_params()
        
        pl.ion()
        fig, axes = pl.subplots(2,4, figsize=(20,15))

        # 2.  Smooth the image:
        # --------------------------------------------------
        kernel_size = processing_params['kernel_size']
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
        morph_kernel = processing_params['morph_kernel']
        morph_iterations = processing_params['morph_iterations']
        pmap_orig = morph_fov(phasemap, kernel_size=morph_kernel, n_iterations=morph_iterations)
        pmap_gaus = morph_fov(smooth_gaus, kernel_size=morph_kernel, n_iterations=morph_iterations)
        pmap_med = morph_fov(smooth_med, kernel_size=morph_kernel, n_iterations=morph_iterations)
        pmap_uni = morph_fov(smooth_uni, kernel_size=morph_kernel, n_iterations=morph_iterations)
        
        axes.flat[4].imshow(pmap_orig, cmap=cmap); #ax2[0].set_title('gaussian')
        axes.flat[5].imshow(pmap_gaus, cmap=cmap); #ax2[0].set_title('gaussian')
        axes.flat[6].imshow(pmap_med, cmap=cmap); #ax2[1].set_title('median')
        axes.flat[7].imshow(pmap_uni, cmap=cmap); #ax2[2].set_title('uniform')
        
        
        pl.show(block=False)
        #pl.show(block=False)
        pl.pause(1.0)
        
        #filter_choices = ['gaussian', 'median', 'uniform']
        return fig
    
