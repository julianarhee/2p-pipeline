#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 01 16:24:13 2020

@author: julianarhee
"""

import re
import matplotlib as mpl

import glob
import os
import shutil
import traceback
import json
import cv2
import h5py

import math
import skimage
import time


import numpy as np
import pylab as pl
import seaborn as sns
import pandas as pd
import tifffile as tf
import cPickle as pkl
import matplotlib.colors as mcolors
import sklearn.metrics as skmetrics 
import seaborn as sns

from mpl_toolkits.axes_grid1 import make_axes_locatable
from pipeline.python import utils as putils
from pipeline.python.retinotopy import utils as ret_utils
from pipeline.python.rois import utils as roi_utils
from pipeline.python.coregistration import align_fov as coreg
from pipeline.python.classifications import gradient_estimation as grd
from pipeline.python.classifications import aggregate_data_stats as aggr

from scipy import misc,interpolate,stats,signal
from matplotlib.colors import LinearSegmentedColormap


from skimage.color import label2rgb
#from skimage.measure import label, regionprops, find_contours
import skimage.measure as skmeasure
from skimage.measure import block_reduce



# --------------------------------------------------
# Plotting 
# --------------------------------------------------
def overlay_contours(labeled_image, ax=None, lc='w', lw=2):
    if ax is None:
        fig, ax = pl.subplots()
    
    label_ids = [l for l in np.unique(labeled_image) if l!=0]
    print(label_ids)
    for label in label_ids: # range(1, labeled_image.max()):
        #contour = skmeasure.find_contours(labeled_image == label, 0.5)[-1]
        #print(label, len(contour))
        #ax.plot(contour[:, 1], contour[:, 0], lc, lw=lw)
        contours = skmeasure.find_contours(labeled_image, level=label)
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], lc, lw=lw)

    return ax


def plot_gradients_in_area(labeled_image, img_az, img_el, grad_az, grad_el, 
                           cmap_phase='nipy_Spectral', contour_color='r',
                           spacing=200, scale=None, width=0.01, headwidth=5):
    '''
    Retinomaps overlaid w/ gradient field, plus average gradient dir.
    '''
    fig, axn = pl.subplots(2,2, figsize=(6,6))

    # Maps ------------
    ax=axn[0, 0]
    im = ax.imshow(img_az,cmap=cmap_phase) #, vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, shrink=0.7)
    ax.set_title('azimuth')
    ax = overlay_contours(labeled_image, ax=ax, lw=2, lc=contour_color)

    ax=axn[1, 0]
    im = ax.imshow(img_el,  cmap=cmap_phase, vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, shrink=0.7)
    ax.set_title('elevation')
    ax = overlay_contours(labeled_image, ax=ax, lw=2, lc=contour_color)

    # Gradients ------------   
    ax=axn[0,0]
    #ax.imshow(thr_img_az, cmap=cmap_phase, vmin=vmin, vmax=vmax)
    grd.plot_gradients(grad_az, ax=ax, draw_interval=spacing, scale=scale, width=width,
                  headwidth=headwidth)
    ax=axn[1, 0]
    #ax.imshow(thr_img_el, cmap=cmap_phase, vmin=vmin, vmax=vmax)
    grd.plot_gradients(grad_el, ax=ax, draw_interval=spacing, scale=scale, width=width,
                  headwidth=headwidth)

    # Unit vectors ------------
    # Get average unit vector
    avg_dir_el = np.rad2deg(grad_el['mean_direction'])
    #print('[EL] avg dir: %.2f deg' % avg_dir_el)
    vhat_el = grad_el['vhat']
    avg_dir_az = np.rad2deg(grad_az['mean_direction'])
    #print('[AZ] avg dir: %.2f deg' % avg_dir_az)
    vhat_az = grad_az['vhat']

    ax= axn[0,1]
    ax.grid(True)
    vh = grad_az['vhat'].copy()
    edir_str = "u=(%.2f, %.2f), %.2f deg" % (vhat_az[0], vhat_az[1], avg_dir_az)
    ax.set_title('azimuth\n%s' % edir_str)
    ax.quiver(0,0, vhat_az[0], vhat_az[1],  scale=1, scale_units='xy', 
              units='xy', angles='xy', width=.05, pivot='tail')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_aspect('equal')
    ax.invert_yaxis()

    ax = axn[1,1]
    ax.grid(True)
    edir_str = "u=(%.2f, %.2f), %.2f deg" % (vhat_el[0], vhat_el[1], avg_dir_el)
    ax.set_title('elevation\n%s' % edir_str)
    ax.quiver(0,0, vhat_el[0], vhat_el[1],  scale=1, scale_units='xy', 
              units='xy', angles='xy', width=.05, pivot='tail')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_aspect('equal')
    ax.invert_yaxis()
    
    pl.subplots_adjust(wspace=0.5, hspace=0.5)

    return fig


def plot_segmentation_steps(img_az, img_el, surface=None, O=None, S_thr=None, sign_map_thr=None, 
                           cmap='viridis', labeled_image=None, region_props=None):

    fig, axf = pl.subplots(2, 3, figsize=(8,8))
    axn = axf.flat

    ax=axn[0]; #ax.set_title(proc_info_str, loc='left', fontsize=12)
    im0 = ax.imshow(surface, cmap='gray'); ax.axis('off');

    ax=axn[1]
    im0 = ax.imshow(img_az, cmap=cmap); ax.axis('off');
    putils.colorbar(im0, label='az')

    ax=axn[2]
    im0 = ax.imshow(img_el, cmap=cmap); ax.axis('off');
    putils.colorbar(im0, label='el')

    ax=axn[3]; ax.set_title('Sign Map, O');
    im0 = ax.imshow(O, cmap='jet'); ax.axis('off');

    ax=axn[4]; ax.set_title('Visual Field Patches\n(std_thr=%.2f)' % sign_map_thr)
    im = ax.imshow(S_thr, cmap='jet'); ax.axis('off');

    cbar_ax = fig.add_axes([0.35, 0.1, 0.3, 0.02])
    cbar_ticks = np.linspace(-1, 1, 5)
    fig.colorbar(im, cax=cbar_ax, orientation='horizontal', ticks=cbar_ticks)
    cbar_ax.tick_params(size=0)

    ax = axn[5]
    ax.imshow(labeled_image)
    for ri, region in enumerate(region_props): 
        ax.text(region.centroid[1], region.centroid[0], 
                '%i' % region.label, fontsize=24, color='k')
    for index in np.arange(0, len(region_props)):
        label = region_props[index].label
        contour = skmeasure.find_contours(labeled_image == label, 0.5)[0]
        ax.plot(contour[:, 1], contour[:, 0], 'w')
    ax.set_title('Labeled (%i patches)' % len(region_labels))
    ax.axis('off')

    return fig


# with rois---
def plot_segmented_rois(seg_results, roi_assignments, roi_masks_labeled, 
                            cmap='viridis', surface=None, ax=None, random_labels=False):

    labeled_image = seg_results['labeled_image']
    region_props = seg_results['region_props']
    label_keys = seg_results['label_keys']
    segmented_areas = seg_results['areas']
        
    d1, d2, nrois = roi_masks.shape
    all_ids = [i for i in np.unique(labeled_image) if i>0]
    
    # Plot rois on visual areas
    roi_int_img = np.zeros((d1, d2))
    for ri in np.arange(0, nrois):
        curr_msk = roi_masks_labeled[:, :, ri].copy()
        if random_labels:
            roi_int_img[curr_msk>0] = (ri+1) #curr_msk.max()
        else:
            found_id = curr_msk.max()
            if found_id in all_ids:
                roi_int_img[curr_msk>0] = found_id
    # plot
    if ax is None:
        fig, ax = pl.subplots(figsize=(3,4))

    plot_roi_area_mask_overlay(labeled_image, region_props, roi_int_img=roi_int_img, 
                               surface=surface, cmap=cmap, ax=ax)
    ax = overlay_contours(labeled_image, ax=ax, lw=2, lc='w')
    plot_keys = [(k, v['id']) for k, v in segmented_areas.items()]
    add_id_legend(plot_keys, ax, cmap=cmap, bbox_to_anchor=(1.1, 1))

    return ax

def plot_roi_area_mask_overlay(labeled_image, region_props, roi_masks=None, roi_int_img=None, 
                               surface=None, ax=None, cmap='hsv'):
    '''
    One of roi_masks or roi_ints must NOT be None.
    
    roi_masks: 3d array of masks (d1, d2, nrois)
    roi_ints: same shape as roi_masks, but 1's are assigne integer values for plotting (None, will be sequential)

    labeled_image: image with assigned segmentations
    region_props: oputput of skimage.measure.region_props() on gradient/signed maps
    '''
    if ax is None:
        f, ax = pl.subplots() 
    
    if roi_masks is not None:
        d1, d2, nrois = roi_masks.shape
    else:
        d1, d2 = roi_int_img.shape
        
    if surface is None:
        surface = np.zeros((d1, d2))
        
    if roi_int_img is None:
        print("Rando assigning ints to rois")
        roi_int_img = roi_utils.assign_int_to_masks(roi_masks)
    
    vmin=labeled_image.min()
    vmax=labeled_image.max()
    print(vmin, vmax)
    
    roi_utils.plot_roi_overlay(roi_int_img, surface, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax)
    ax = overlay_contours(labeled_image, ax=ax, lw=2, lc='w')

    return ax


def add_id_legend(label_keys, ax, cmap='viridis', bbox_to_anchor=(1,1)):
    label_ids = [v[1] for v in label_keys]
    vmin, vmax = 0, max(label_ids)
    
    cmap = mpl.cm.get_cmap(cmap)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    id_colors = [cmap(norm(i)) for (n, i) in label_keys]
    id_names = [n for (n, i) in label_keys]
    lhandles = putils.custom_legend_markers(colors=id_colors, labels=id_names)
    ax.legend(handles=lhandles, bbox_to_anchor=bbox_to_anchor)

# --------------------------------------------------
# Segmentation functions
# --------------------------------------------------
def segment_areas(img_az, img_el, sign_map_thr=0.5, min_region_area=500, surface=None):
    # Calculate gradients
    # ---------------------------------------------------------
    h_map = img_el.copy()
    v_map = img_az.copy()
    [h_gy, h_gx] = np.array(grd.gradient_phase(h_map))
    [v_gy, v_gx] = np.array(grd.gradient_phase(v_map))

    h_gdir = np.arctan2(h_gy, h_gx) # gradient direction
    v_gdir = np.arctan2(v_gy, v_gx)

    # Create sign map
    # ---------------------------------------------------------
    gdiff = v_gdir-h_gdir
    gdiff = (gdiff + math.pi) % (2*math.pi) - math.pi

    #O=-1*np.sin(gdiff)
    O=np.sin(gdiff) # LEFT goes w/ BOTTOM.  RIGHT goes w/ TOP.
    S=np.sign(O) # Discretize into patches

    # Calculate STD, and threshold to separate areas (simple morph. step)
    # ---------------------------------------------------------
    O_sigma = np.nanstd(O)
    S_thr = np.zeros(np.shape(O))
    S_thr[O>(O_sigma*sign_map_thr)] = 1
    S_thr[O<(-1*O_sigma*sign_map_thr)] = -1
    
    return O, S_thr

def segment_and_label(S_thr):

    # Create segmented + labeled map
    # ---------------------------------------------------------
    filled_smap = grd.fill_nans(S_thr)
    labeled_image_tmp, n_labels = skmeasure.label(
                                 filled_smap, background=0, return_num=True)

    image_label_overlay = label2rgb(labeled_image_tmp) #, image=segmented_img) 
    print(labeled_image_tmp.shape, image_label_overlay.shape)
    rprops_ = skmeasure.regionprops(labeled_image_tmp, filled_smap)
    region_props = [r for r in rprops_ if r.area > min_region_area]
    
    # Relabel image
    labeled_image = np.zeros(labeled_image_tmp.shape)
    for ri, rprop in enumerate(region_props):
        new_label = int(ri+1)
        labeled_image[labeled_image_tmp==rprop.label] = new_label
        rprop.label = new_label
        region_props[ri] = rprop
        
    return region_props, labeled_image 


# --------------------------------------------------
# ROI ASSIGNMENT
# --------------------------------------------------

def get_transformed_rois(animalid, session, fov, retinorun='retino_run1', 
                        roi_id=None, traceid='traces001'):
    if roi_id is None:
        roi_id = roi_utils.get_roiid_from_traceid(animalid, session, fov, traceid=traceid)
    roi_masks, zprog_img = roi_utils.load_roi_masks(animalid, session, fov, rois=roi_id)
    print("Loaded rois: %s" % roi_id)
    d1, d2, nrois = roi_masks.shape

    pixel_size = putils.get_pixel_size()    
    # pixel_size = (pixel_size[0]*ds_factor, pixel_size[1]*ds_factor)
    roi_masks_tr = np.dstack([coreg.transform_2p_fov(roi_masks[:, :, i].astype(float), pixel_size) \
                   for i in np.arange(0, nrois)]) # transform/orient
    roi_masks = roi_masks_tr.astype(bool).astype(int)
    
    return roi_masks


def label_roi_masks(seg_results, roi_masks):
    d1, d2, nrois = roi_masks.shape
    print("Roi masks:", d1, d2, nrois)
    roi_list = [roi_masks[:, :, r] for r in np.arange(0, nrois)] # list makes calc faster
    
    #print(seg_mask.shape)
    
    label_keys = seg_results['label_keys']
    roi_assignments={}
    for area_name, seg in seg_results['areas'].items():
#         if putils.isnumeric(area_name):
#             continue
        seg_mask = cv2.resize(seg['mask'], (d2, d1))
        id_mask = seg['id'] * seg_mask
        incl_rois = np.array([int(i) for i, rmask in enumerate(roi_list) if (id_mask*rmask).max()==seg['id']])
        roi_assignments[area_name]=list(incl_rois)

        # assign label to masks
        if len(incl_rois)>0:
            roi_masks[:,:,incl_rois] *= seg['id']
    
    return roi_assignments, roi_masks


def assign_rois_to_visual_area(animalid, session, fov, retinorun='retino_run1',roi_id=None, 
                                traceid='traces001', return_labeled_masks=False, verbose=False):

    # Load ROIs
    roi_masks = get_transformed_rois(animalid, session, fov, retinorun=retinorun, 
                                     traceid=traceid)
    d1, d2, nrois = roi_masks.shape

    # Load segmentation results
    seg_results = roi_utils.load_segmentation_results(animalid, session, fov, retinorun=retinorun)

    # Assign each ROI to visual areas
    roi_assignments, roi_masks_labeled = label_roi_masks(seg_results, roi_masks.astype(int))

    if verbose:
        for v, r in roi_assignments.items():
            print("%s: %i cells" % (v, len(r)))

    if return_labeled_masks:
        return roi_assignments, roi_masks_labeled
    else:
        return roi_assignments


# --------------------------------------------------
# Data loading
# --------------------------------------------------
def load_segmentation_results(animalid, session, fov, retinorun='retino_run1', rootdir='/n/coxfs01/2p-data'):
    results_fpath = os.path.join(rootdir, animalid, session, fov, retinorun, 
                              'retino_analysis', 'segmentation', 'results.pkl')
    
    assert os.path.exists(results_fpath), "Segmentation not found: %s" % results_fpath
    with open(results_fpath, 'r') as f:
        seg_areas = pkl.load(f)
        
    return seg_areas



def load_roi_assignments(animalid, session, fov, retinorun=None, rootdir='/n/coxfs01/2p-data'):
    
    if retinorun is None:
        retinorun='retino_run*'
    results_fpath = glob.glob(os.path.join(rootdir, animalid, session, fov, retinorun, 
                              'retino_analysis', 'segmentation', 'roi_assignments.json'))
    datakey = '%s_%s_%s' % (session, animalid, fov)
    assert len(results_fpath)>0, "No roi assignments found: %s" % datakey
    
    #assert os.path.exists(results_fpath[0]), "Assignment results not found: %s" % results_fpath
    with open(results_fpath[0], 'r') as f:
        roi_assignments = json.load(f)
    #roi_assignments = r_['roi_assignments']
    #roi_masks_labeled = r_['roi_masks_labeled']
    
    return roi_assignments #, roi_masks_labeled


# Relevant calcs
def get_gradients_in_area(curr_segmented_mask, img_az, img_el):
    thr_img_az = img_az.copy()
    thr_img_az[curr_segmented_mask==0] = np.nan
    grad_az = grd.calculate_gradients(thr_img_az)

    thr_img_el = img_el.copy()
    thr_img_el[curr_segmented_mask==0] = np.nan
    grad_el = grd.calculate_gradients(thr_img_el)

    return grad_az, grad_el


def get_cells_by_area(sdata, visual_areas=['V1', 'Lm', 'Li'],
                  excluded_datasets=['20190602_JC080_fov1', '20190605_JC090_fov1', 
                                     '20191003_JC111_fov1', '20191104_JC117_fov1']):
    dsets = sdata[~sdata['datakey'].isin(excluded_datasets)]
    
    missing_cells=[]
    d_ = []
    for (animalid, session, fovnum, datakey), g in dsets.groupby(['animalid', 'session', 'fovnum', 'datakey']):
        fov = g['fov'].iloc[0]
        try:
            roi_assignments = load_roi_assignments(animalid, session, fov, retinorun=None)
        except AssertionError:
            missing_cells.append(datakey)
            continue
        for v, rlist in roi_assignments.items():
            if v in visual_areas:
                tmpd = pd.DataFrame({'cell': rlist})
                metainfo = {'visual_area': v, 'datakey': datakey, 'fovnum': fovnum, 
                            'animalid': animalid, 'session': session}
                tmpd = putils.add_meta_to_df(tmpd, metainfo)
                d_.append(tmpd)

    rois = pd.concat(d_, axis=0).reset_index(drop=True)
    print("Need to segment %i datasets" % len(missing_cells))
    
    return rois


# ------------------------------------------------------------------------------------------

#datakey = '20190522_JC089_fov1'
#session, animalid, fovn = datakey.split('_')
#fovnum = int(fovn[3:])
#
## animalid = 'JC097'
## session = '20190613'
## fovnum = 1
#
#fov = 'FOV%i_zoom2p0x' % fovnum
#traceid = 'traces001'
#
#datakey='%s_%s_fov%i' % (session, animalid, fovnum)
#
## Get retino runs
#found_retinodirs = glob.glob(os.path.join(rootdir, animalid, session, fov, 'retino*'))
#found_retinoruns = [os.path.split(d)[-1] for d in found_retinodirs]
#print("Found %i runs" % len(found_retinoruns))
#
## Set current animal's retino output dir
#run_ix = 0
#retinorun = found_retinoruns[run_ix]
#curr_dst_dir = os.path.join(found_retinodirs[run_ix], 'retino_analysis', 'segmentation')
#print(curr_dst_dir)
#
#data_id = '_'.join([animalid, session, fov, retinorun, traceid])
#
## Map smoothing
## ----------------------------------------------------------
#delay_map_thr = 0.8
#pix_mag_thr = 0.002
#smooth_fwhm = 5
#smooth_spline=2
#cmap_name = 'nic_Edge'
## ----------------------------------------------------------
## Segmenting params
#sign_map_thr = 0.2
#min_region_area = 500
#
#
#
## Smooth retino maps
#az_fill, el_fill, params, RETID = grd.pixel_gradients(animalid, session, fov, traceid=traceid, 
#                                                    retinorun=retinorun, 
#                                                    mag_thr=pix_mag_thr, 
#                                                    delay_map_thr=delay_map_thr, 
#                                                    dst_dir=curr_dst_dir, 
#                                                    cmap=cmap_name, 
#                                                    smooth_fwhm=smooth_fwhm, 
#                                                    smooth_spline=smooth_spline,
#                                                    full_cmap_range=False) 
#
#
## Get surface image
#surface_img = ret_utils.load_2p_surface(animalid, session, fov, ch_num=1, retinorun=retinorun)
#pixel_size = putils.get_pixel_size()
#surface_2p = coreg.transform_2p_fov(surface_img, pixel_size, normalize=False)
#surface_2p = putils.adjust_image_contrast(surface_2p, clip_limit=5.0, tile_size=5)
#
## Convert to screen units
#vmin, vmax = (-np.pi, np.pi)
#img_az = putils.convert_range(az_fill, oldmin=vmin, oldmax=vmax, newmin=screen_min, newmax=screen_max)
#img_el = putils.convert_range(el_fill, oldmin=vmin, oldmax=vmax, newmin=screen_min, newmax=screen_max)
#vmin, vmax = (screen_min, screen_max)   
#
#
## Segment areas
## -------------------------------------------------------------------
#O, S_thr = segment_areas(img_az, img_el, sign_map_thr=sign_map_thr, 
#                         min_region_area=min_region_area, surface=surface_2p)
## Label image
#region_props, labeled_image  = segment_and_label(S_thr)
#region_labels = [region.label for region in region_props]
#print('Found %i regions: %s' % (len(region_labels), str(region_labels)))
#
## Save
#orig_d1, orig_d2 = surface_2p.shape
#labeled_image_2p = cv2.resize(labeled_image.astype(np.uint8), (orig_d2, orig_d1)) #surface_2p.shape)
#results = {'labeled_image_ds': labeled_image, 
#           'labeled_image': labeled_image_2p,
#           'region_props': region_props}
#
## Plot segmentation results
#proc_info_str = 'pixthr=%.3f (delay thr=%.2f), smooth=%i' % (pix_mag_thr, delay_map_thr, smooth_fwhm)
#fig = plot_segmentation_steps(img_az, img_el, surface=surface_2p, O=O, S_thr=S_thr, 
#                                sign_map_thr=sign_map_thr, cmap=cmap_phase, 
#                                labeled_image=labeled_image, region_props=region_props)
#
#putils.label_figure(fig, '%s | %s' % (data_id, proc_info_str))
#pl.subplots_adjust(hspace=0.5, bottom=0.2)    
#pl.savefig(os.path.join(curr_dst_dir, 'segemented_areas.png'))
#pl.show()
#
## Label
#region_dict = {}
#while True:
#    for ri, region in enumerate(region_props):
#        user_sel = raw_input('[%i] Enter name: ' % region.label)
#        if len(user_sel)>0 and putils.isnumber(user_sel):
#            region_dict.update({str(user_sel): int(region.label)})
#    for rname, rlabel in region_dict.items():
#        print('  %s: %i' %(rname, rlabel))
#    user_approve = raw_input("Press <ENTER> to keep, <r> to redo: ")
#    if user_approve != 'r':
#        break
#
## Assign labels
#seg_areas = {}
#label_keys=[]
#for ri, region in enumerate(region_props):
#    region_id = region.label
#    if region.label in region_dict.keys():
#        region_name = region_dict[region.label]
#        label_keys.append((region_name, region_id))
#    else:
#        region_name = region.label
#
#    # save mask
#    region_mask = np.copy(labeled_image.astype('float'))
#    region_mask[labeled_image != region_id] = 0
#    region_mask[labeled_image == region_id] = 1
#    seg_areas[region_name] = {'id': region_id, 'mask': region_mask}
#   
#
## Plot and confirm
## double check labeling/naming of segmented areas
#area_ids = [k[1] for k in label_keys]
#labeled_image_incl = np.ones(labeled_image.shape)*np.nan #labeled_image.copy()
#for idx in area_ids:
#    labeled_image_incl[labeled_image==idx] = idx
#
#fig, ax = pl.subplots()
#ax.imshow(labeled_image_incl, cmap='jet')    
#for region in region_props:
#    if region.label in area_ids:
#        region_name = str([k[0] for k in label_keys if k[1]==region.label][0])
#        ax.text(region.centroid[1], region.centroid[0], 
#                        '%s (%i)' % (region_name, region.label), fontsize=24, color='k')
#    # plot
#    contour = skmeasure.find_contours(labeled_image == region.label, 0.5)[0]
#    ax.plot(contour[:, 1], contour[:, 0], 'w', lw=5)
#ax.set_title('Labeled (%i patches)' % len(area_ids))
#ax.axis('off')
#putils.label_figure(fig, '%s | %s' % (data_id, proc_info_str))
#pl.savefig(os.path.join(curr_dst_dir, 'labeled_areas.png'))
#
#
## Save
## Load data metainfo
#print("Current run: %s" % retinorun)
#retinoid, RETID = ret_utils.load_retino_analysis_info(animalid, session, fov, retinorun, traceid, use_pixels=True)
#data_id = '_'.join([animalid, session, fov, retinorun, retinoid])
#print("DATA ID: %s" % data_id)
#
## Get ROIID and projection image
#ds_factor = int(RETID['PARAMS']['downsample_factor'])
#print('Data were downsampled by %i.' % ds_factor)
#
#
#segparams_fpath = os.path.join(curr_dst_dir, 'params.json')
#segresults_fpath = os.path.join(curr_dst_dir, 'results.pkl')
#
#seg_params = {'pixel_mag_thr': pix_mag_thr,
#              'downsample_factor': ds_factor,
#              'delay_map_thr': delay_map_thr,
#              'smooth_fwhm': smooth_fwhm,
#              'smooth_spline': smooth_spline,
#              'sign_map_thr': sign_map_thr,
#              'min_region_area': min_region_area,
#              'retino_id': retinoid, 
#              'retino_run': retinorun}
#
#results.update({'areas': seg_areas})
#results.update({'label_keys': label_keys})
#
#with open(segparams_fpath, 'w') as f:
#    json.dump(seg_params, f, indent=4, sort_keys=True)
#    
#with open(segresults_fpath, 'wb') as f:
#    pkl.dump(results, f, protocol=pkl.HIGHEST_PROTOCOL)
