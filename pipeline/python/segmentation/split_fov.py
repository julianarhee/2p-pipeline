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
import skimage.measure
import imutils
import sys
import optparse
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
from matplotlib import cm
from pipeline.python.segmentation import segmentation as seg
from pipeline.python.rois.utils import get_roi_contours, uint16_to_RGB, plot_roi_contours
from pipeline.python.utils import label_figure, natural_keys
#%%

    
#%%
#rootdir = '/n/coxfs01/2p-data'

# Combine different conditions of the SAME acquisition:
#animalid = 'JC015'
#session = '20180919'
#acquisition = 'FOV1_zoom2p0x'
#retino_run = 'retino_run1'

#animalid = 'JC015'
#session = '20180925'
#acquisition = 'FOV1_zoom2p0x'
#retino_run = 'retino_run1'

#use_azimuth = True
#use_single_ref = True
#retino_file_ix = 0

#cmap = cm.Spectral_r

def extract_options(options):

    def comma_sep_list(option, opt, value, parser):
        setattr(parser.values, option.dest, value.split(','))

    parser = optparse.OptionParser()

    parser.add_option('-D', '--root', action='store', dest='rootdir',
                          default='/n/coxfs01/2p-data',
                          help='data root dir (dir w/ all animalids) [default: /nas/volume1/2photon/data, /n/coxfs01/2pdata if --slurm]')
    parser.add_option('--slurm', action='store_true', dest='slurm', default=False, help="set if running as SLURM job on Odyssey")

    parser.add_option('-i', '--animalid', action='store', dest='animalid',
                          default='', help='Animal ID')

    # Set specific session/run for current animal:
    parser.add_option('-S', '--session', action='store', dest='session',
                          default='', help='session dir (format: YYYMMDD_ANIMALID')
    parser.add_option('-A', '--acq', action='store', dest='acquisition',
                          default='FOV1', help="acquisition folder (ex: 'FOV1_zoom3x') [default: FOV1]")

    # Processing info:
    kernel_choices = ('original', 'gaussian', 'median', 'uniform')
    kernel_default = None#'median'
    parser.add_option('-K', '--kernel-type', type='choice', choices=kernel_choices, action='store', dest='kernel_type', default=kernel_default, help="Kernel type to smooth retino pixel image. Valid choices: %s [default: %s]" % (kernel_choices, kernel_default))
    parser.add_option('-k', '--kernel-size', action='store', dest='kernel_size', default=None, help="kernel size for smoothing retino phase map (default: 21). Must be an odd number.")
    parser.add_option('-m', '--morph-size', action='store', dest='morph_kernel', default=None, help="kernel size for morphological opening/dilation when creating processed image for segmentation (default: 5).")
    parser.add_option('-n', '--morph-iter', action='store', dest='morph_iterations', default=None, help="number of iterations for morphological opening/dilation (default: 2).") 

    parser.add_option('--elevation', action='store_false', dest='use_azimuth', default=True, help="set to use ELEVATION map instead of azimuth")
    parser.add_option('--single', action='store_true', dest='use_single_ref', default=False, help="set to use single reference instead of average across condition reps")
    parser.add_option('-f', '--ref', action='store', dest='retino_file_ix', default=None, help="File number to use if single reference (default: 1)")
    parser.add_option('-V', '--area', action='store', dest='visual_area', default=None, help='Name of visual area to segment and label (if interactive, can enter after segmentation)')

    parser.add_option('--append', action='store_true', dest='append', default=False, help="set to append another FOV to existing segmentation object")
    parser.add_option('--test', action='store_true', dest='test', default=False, help="set to test processing params (interactive mode)")
    parser.add_option('--new', action='store_true', dest='create_new', default=False, help="set to create new segmentation object")

    parser.add_option('-p', '--nproc', action='store', dest='nprocesses', default=4, help="N processes if running in par (default=4)")

    (options, args) = parser.parse_args(options)
    if options.slurm is True and '/n/coxfs01' not in options.rootdir:
        options.rootdir = '/n/coxfs01/2p-data'
    return options


def create_segmentation_object(animalid, session, acquisition, rootdir='/n/coxfs01/2p-data',
                                visual_area=None, cmap=cm.Spectral_r,
                                use_azimuth=None, use_single_ref=None, retino_file_ix=None):
    # Initialize segmentation class instance:
    fov = seg.Segmentations(animalid, session, acquisition, rootdir=rootdir,
                            use_azimuth=use_azimuth, use_single_ref=use_single_ref, retino_file_ix=retino_file_ix)
    fov.get_analyzed_source_data()

    print "Single ref:", fov.use_single_ref
    print "Azimuth:", fov.use_azimuth

    # Get averaged phasemap for each condition:
    az_phasemap = fov.get_phase_map(analysis_type='pixels', use_azimuth=True, use_single_ref=False)
    el_phasemap = fov.get_phase_map(analysis_type='pixels', use_azimuth=False, use_single_ref=False)
    fig, ax = pl.subplots(1,2,figsize=(10,6))
    ax[0].imshow(az_phasemap, cmap=cmap); ax[0].set_title('avg azimuth'); ax[0].axis('off')
    ax[1].imshow(el_phasemap, cmap=cmap); ax[1].set_title('avg elevation'); ax[1].axis('off')
    
    if fov.use_azimuth is None:
        pl.show(block=False)
        pl.pause(1.0)

        which_cond = input('Select 1 to use azimuth, 0 to use elevation: ')
        if which_cond == 1:
            fov.use_azimuth = True
        else:
            fov.use_azimuth = False

    if fov.use_single_ref is None:
        which_ref = input("Select 1 to use SINGLE condition, or 0 to use AVERAGED conditions: ")
        if which_ref == 1:
            fov.use_single_ref = True
            if fov.retino_file_ix is None:
                fov.retino_file_ix = input('Enter file num (1-indexed) to use as single reference: ')
        else:
            fov.use_single_ref = False
 
    label_figure(fig, fov.data_identifier)
    pl.savefig(os.path.join(fov.output_dir, 'figures', 'averaged_conditions.png'))
    pl.close()

    # Save params:
    params = {'use_azimuth': fov.use_azimuth,
              'use_single_ref': fov.use_single_ref,
              'retino_file_ix': retino_file_ix if use_single_ref else -1,
              'roi_analysis': fov.source.retinoID_rois,
              'pixel_analysis': fov.source.retinoID_pixels}
    with open(os.path.join(fov.output_dir, 'retino_params.json'), 'w') as f:
        json.dump(params, f, indent=4)

    if fov.use_azimuth:
        if fov.use_single_ref:
            fov.phasemap = fov.get_phase_map(analysis_type='pixels', 
                                             use_azimuth=fov.use_azimuth,
                                             use_single_ref=fov.use_single_ref,
                                             retino_file_ix=fov.retino_file_ix)
        else:
            fov.phasemap = az_phasemap
    else:
        if use_single_ref:
            fov.phasemap = fov.get_phase_map(analysis_type='pixels', 
                                             use_azimuth=fov.use_azimuth,
                                             use_single_ref=fov.use_single_ref,
                                             retino_file_ix=fov.retino_file_ix)
        else:
            fov.phasemap = el_phasemap

    segmentation_fpath = fov.save_me()
    print "Created and saved Segmentation() object to:\n%s" % segmentation_fpath

    return fov

def process_phasemap(fov, kernel_type=None, kernel_size=21, morph_kernel=5, morph_iterations=2, cmap=cm.Spectral_r):


        # 1.  Processing retinotopy map image:
    # -----------------------------------------------------------------------------
    kernel_size = 21 if kernel_size is None else kernel_size
    morph_kernel = 5 if morph_kernel is None else kernel_size
    morph_iterations = 2 if morph_iterations is None else kernel_size
    preprocessing_params = fov.default_preprocessing_params(kernel_size=kernel_size, 
                                                            morph_kernel=morph_kernel,
                                                            morph_iterations=morph_iterations)
    if kernel_type is None: 
        kernel_type, preprocessing_params = fov.test_filter_types(fov.phasemap, preprocessing_params=preprocessing_params)

    mask_template, map_thr, split_intervals = fov.segment_fov(fov.phasemap, nsplits=40,
                                                              kernel_type=kernel_type, 
                                                              preprocessing_params=preprocessing_params, 
                                                              cmap=cmap) 

    return mask_template

def save_visual_area(fov, cmap='Spectral_r', visual_area=None):

    mask_template = fov.preprocessing['mask_template']

    #%%
    # 2.  Select regions from segmentation:
    # -----------------------------------------------------------------------------
    labeled_image, region_id, region_mask = fov.select_visual_area(mask_template)
    regions = regionprops(labeled_image)

    # 3.  Plot original phase map, ROI phase map, and included ROIs on fov image:
    # -----------------------------------------------------------------------------
    fov_img = fov.get_fov_image(fov.source.retino_run)
    retino_rundir = os.path.join(fov.source.rootdir, fov.source.animalid, fov.source.session, \
                                fov.source.acquisition, fov.source.retino_run)
    retino_params = seg.load_retino_params(retino_rundir, fov.source.retinoID_rois) 
    roi_masks = fov.get_roi_masks(retino_params)
    nrois = roi_masks.shape[-1]
    roi_contours = get_roi_contours(roi_masks, roi_axis=-1)

    # Get roi phase map:
    roi_phasemap = fov.get_phase_map(analysis_type='rois', use_azimuth=True, use_single_ref=True, retino_file_ix=1)
    roi_phase_mask = np.copy(roi_phasemap)
    roi_phase_mask[roi_phasemap==0] = np.nan
  
    # Load saved pixel phase map:
    pixel_phasemap = fov.phasemap

    # PLOT:
    fig, axes = pl.subplots(2,2, figsize=(15,10))
    axes.flat[0].imshow(region_mask)
    plot_roi_contours(fov_img, roi_contours, ax=axes.flat[1], thickness=0.1, label_all=False, clip_limit=0.02)

    axes.flat[2].imshow(fov_img, cmap='gray')
    axes.flat[2].imshow(roi_phase_mask, cmap=cmap, vmin=0, vmax=np.pi*2)

    # Mask ROIs with area mask:
    region_mask_copy = np.copy(region_mask)
    region_mask_copy[region_mask==0] = np.nan

    included_rois = [ri for ri in range(nrois) if ((roi_masks[:, :, ri] + region_mask_copy) > 1).any()]
    plot_roi_contours(fov_img, roi_contours, ax=axes.flat[3], thickness=0.1, 
                          roi_highlight = included_rois,
                          roi_color_default=(127,127,127),
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

    if visual_area is None:
        region_name = raw_input("Enter name of visual area: ")
    else:
        region_name = visual_area 
    selected_area = regions[int(region_id-1)]
    axes.flat[0].text(selected_area.centroid[1], selected_area.centroid[0], '%s' % region_name, fontsize=24, color='w')

    pl.savefig(os.path.join(fov.output_dir, 'figures', 'segmented_%s_%s.png' % (region_name, fov.datestr)))

    pl.close()

    #%%

    fov.save_visual_area(selected_area, region_name, region_id, region_mask, included_rois)

    #%%
    segmentation_fpath = fov.save_me()
   
    return segmentation_fpath


def load_segmentation_object(animalid, session, acquisition, rootdir='/n/coxfs01/2p-data'):
    existing_seg_fpaths = sorted(glob.glob(os.path.join(rootdir, animalid, session, \
                                   acquisition, 'visual_areas', 'segmentation_*.pkl')), key=natural_keys)[::-1]
    print "Found %i existing seg objects, listing by most recent:"
    for fi, fpath in enumerate(existing_seg_fpaths):
        print fi, fpath
    sel = input("Select IDX of seg oject to load: ")
    seg_fpath = existing_seg_fpaths[sel]
    
    with open(seg_fpath, 'rb') as f:
        fov = pkl.load(f)
    return fov
 
def do_fov_segmentation(animalid, session, acquisition, rootdir='/n/coxfs01/2p-data', append=False,
                        visual_area=None, cmap='Spectral_r',
                        use_azimuth=True, use_single_ref=False, retino_file_ix=1,
                        kernel_type=None, kernel_size=None, morph_kernel=None, morph_iterations=None):

    if kernel_size is not None:
        kernel_size = int(optsE.kernel_size)
    if morph_kernel is not None:
        morph_kernel = int(optsE.morph_kernel)
    if morph_iterations is not None:
        morph_iterations = int(optsE.morph_iterations)

    if append:
        fov = load_segmentation_object(animalid, session, acquisition, rootdir=rootdir)

    else:
        fov = create_segmentation_object(animalid, session, acquisition, rootdir=rootdir,
                                visual_area=visual_area, cmap=cmap,
                                use_azimuth=use_azimuth, use_single_ref=use_single_ref, 
                                retino_file_ix=retino_file_ix)

        mask_template = process_phasemap(fov, visual_area=visual_area, kernel_type=kernel_type, kernel_size=kernel_size, 
                                morph_kernel=morph_kernel, morph_iterations=morph_iterations, cmap=cmap)

    segmentation_fpath = save_visual_area(fov, visual_area=visual_area, cmap=cmap)


def main(options):
    optsE = extract_options(options)
    cmap = 'Spectral_r'

    append = optsE.append
    if optsE.create_new:
        append = False
    
    if optsE.test:
        use_single_ref = None #True
    else:
        use_single_ref = optsE.use_single_ref
 
    if optsE.test:
        use_azimuth = None #True #optsE.use_azimuth 
    else:
        use_azimuth = optsE.use_azimuth
 
    retino_file_ix = optsE.retino_file_ix
    if optsE.retino_file_ix is not None:
        retino_file_ix = int(optsE.retino_file_ix)
    
    kernel_type = optsE.kernel_type
    kernel_size = optsE.kernel_size
    morph_kernel = optsE.morph_kernel
    morph_iterations = optsE.morph_iterations
    do_fov_segmentation(optsE.animalid, optsE.session, optsE.acquisition, rootdir=optsE.rootdir, append=append,
                        visual_area=optsE.visual_area, cmap=cmap, use_azimuth=use_azimuth,
                        use_single_ref=use_single_ref, retino_file_ix=retino_file_ix,
                        kernel_type=kernel_type, kernel_size=kernel_size,
                        morph_kernel=morph_kernel, morph_iterations=morph_iterations)


if __name__ == '__main__':
    main(sys.argv[1:])

            

