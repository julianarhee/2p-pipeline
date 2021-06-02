#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 01 16:24:13 2020

@author: julianarhee
"""

import re
import matplotlib as mpl
mpl.use('agg')

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
#from pipeline.python.classifications import gradient_estimation as grd
from pipeline.python.classifications import aggregate_data_stats as aggr

from scipy import misc,interpolate,stats,signal
from matplotlib.colors import LinearSegmentedColormap


from skimage.color import label2rgb
#from skimage.measure import label, regionprops, find_contours
import skimage.measure as skmeasure
from skimage.measure import block_reduce
#pl.switch_backend('TkAgg')

# --------------------------------------------------------------------
# plotting
# -------------------------------------------------------------------- 
def plot_gradients_in_area(labeled_image, img_az, img_el, grad_az, grad_el, 
                           cmap_phase='nipy_Spectral', contour_lc='r', contour_lw=1,
                           spacing=200, scale=None, width=0.01, headwidth=5, vmin=-59, vmax=59):
    '''
    Retinomaps overlaid w/ gradient field, plus average gradient dir.
    '''
    from pipeline.python.classifications import gradient_estimation as grd

    fig, axn = pl.subplots(2,2, figsize=(6,6))

    # Maps ------------
    ax=axn[0, 0]
    im = ax.imshow(img_az,cmap=cmap_phase) #, vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, shrink=0.7)
    ax.set_title('azimuth')
    ax = overlay_all_contours(labeled_image, ax=ax, lw=contour_lw, lc=contour_lc)

    ax=axn[1, 0]
    im = ax.imshow(img_el,  cmap=cmap_phase, vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, shrink=0.7)
    ax.set_title('elevation')
    ax = overlay_all_contours(labeled_image, ax=ax, lw=contour_lw, lc=contour_lc)

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


def plot_segmentation_steps(img_az, img_el, surface=None, O=None, S_thr=None, params=None,
                            cmap='viridis', labeled_image=None, region_props=None, 
                            label_color='w', lw=1):
    
    sign_map_thr = 0 if params is None else params['sign_map_thr']

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
                '%i' % region.label, fontsize=24, color=label_color)
#     for index in np.arange(0, len(region_props)):
#         label = region_props[index].label
#         contour = skmeasure.find_contours(labeled_image == label, 0.5)[0]
#         ax.plot(contour[:, 1], contour[:, 0], label_color)
    contours = skmeasure.find_contours(labeled_image, level=0)
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], label_color, lw=lw)
        
    ax.set_title('Labeled (%i patches)' % len(region_props))
    ax.axis('off')

    return fig

def overlay_contours(labeled_image, ax=None, lc='w', lw=2):
    if ax is None:
        fig, ax = pl.subplots()
    
    label_ids = [l for l in np.unique(labeled_image) if l!=0]
    #print(label_ids)
    for label in label_ids: # range(1, labeled_image.max()):
        #contour = skmeasure.find_contours(labeled_image == label, 0.5)[-1]
        #print(label, len(contour))
        #ax.plot(contour[:, 1], contour[:, 0], lc, lw=lw)
        contours = skmeasure.find_contours(labeled_image, level=label)
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], lc, lw=lw)

    return ax

def overlay_all_contours(labeled_image, ax=None, lc='w', lw=2):
    if ax is None:
        fig, ax = pl.subplots()
    
    label_ids = [l for l in np.unique(labeled_image) if l!=0]
    #print(label_ids)
    #for label in label_ids: # range(1, labeled_image.max()):
        #contour = skmeasure.find_contours(labeled_image == label, 0.5)[-1]
        #print(label, len(contour))
        #ax.plot(contour[:, 1], contour[:, 0], lc, lw=lw)
    contours = skmeasure.find_contours(labeled_image, level=0)
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], lc, lw=lw)

    return ax



# --------------------------------------------------------------------
# segmentation
# -------------------------------------------------------------------- 
def segment_areas(img_az, img_el, sign_map_thr=0.5):
    from pipeline.python.classifications import gradient_estimation as grd

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

def segment_and_label(S_thr, min_region_area=500):
    from pipeline.python.classifications import gradient_estimation as grd

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

# --------------------------------------------------------------------
# data loading/saving
# -------------------------------------------------------------------- \
def load_segmentation_results(animalid, session, fov, retinorun='retino_run1', 
                                rootdir='/n/coxfs01/2p-data'):
    
    retino_seg_dir = os.path.join(rootdir, animalid, session, fov, retinorun, 
                              'retino_analysis', 'segmentation')
    
    results_fpath = os.path.join(retino_seg_dir, 'results.pkl')
    assert os.path.exists(results_fpath), "Segmentation not found: %s" % results_fpath
    with open(results_fpath, 'r') as f:
        results = pkl.load(f)

    params_fpath = os.path.join(retino_seg_dir, 'params.json')
    assert os.path.exists(params_fpath), "Segmentation params not found: %s" % params_fpath
    with open(params_fpath, 'r') as f:
        params = json.load(f)
    
    return results, params

def load_roi_assignments(animalid, session, fov, retinorun='retino_run1', 
                            rootdir='/n/coxfs01/2p-data'):
    
    results_fpath = os.path.join(rootdir, animalid, session, fov, retinorun, 
                              'retino_analysis', 'segmentation', 'roi_assignments.json')
    
    assert os.path.exists(results_fpath), "Assignment results not found: %s" % results_fpath
    with open(results_fpath, 'r') as f:
        roi_assignments = json.load(f)
   
    return roi_assignments #, roi_masks_labeled

def get_cells_by_area(sdata, excluded_datasets=[], return_missing=False, verbose=False,
                    rootdir='/n/coxfs01/2p-data'):
    '''
    Use retionrun to ID area boundaries. If more than 1 retino, combine.
    '''

    excluded_datasets = ['20190602_JC080_fov1', '20190605_JC090_fov1',
                         '20191003_JC111_fov1', 
                         '20191104_JC117_fov1', '20191104_JC117_fov2', #'20191105_JC117_fov1',
                         '20191108_JC113_fov1', '20191004_JC110_fov3',
                         '20191008_JC091_fov'] 
    missing_segmentation=[]
    d_ = []
    for (animalid, session, fov, datakey), g in sdata.groupby(['animalid', 'session', 'fov', 'datakey']):
        if datakey in excluded_datasets:
            continue
        retinoruns = [os.path.split(r)[-1] for r in glob.glob(os.path.join(rootdir, animalid, session, fov, 'retino*'))]
        roi_assignments=dict()
        for retinorun in retinoruns:
            try:
                rois_ = load_roi_assignments(animalid, session, fov, retinorun=retinorun)
                for varea, rlist in rois_.items():
                    if varea not in roi_assignments.keys():
                        roi_assignments[varea] = []
                    roi_assignments[varea].extend(rlist)
            except Exception as e:
                if verbose:
                    print("... skipping %s (%s)" % (datakey, retinorun))
                missing_segmentation.append((datakey, retinorun))
                continue
 
        for varea, rlist in roi_assignments.items():
            if putils.isnumber(varea):
                continue
             
            tmpd = pd.DataFrame({'cell': list(set(rlist))})
            metainfo = {'visual_area': varea, 'animalid': animalid, 'session': session,
                        'fov': fov, 'fovnum': g['fovnum'].values[0], 'datakey': g['datakey'].values[0]}
            tmpd = putils.add_meta_to_df(tmpd, metainfo)
            d_.append(tmpd)

    cells = pd.concat(d_, axis=0).reset_index(drop=True)
    cells = cells[~cells['datakey'].isin(excluded_datasets)]
    
    #print("Missing %i datasets for segmentation:" % len(missing_segmentation)) 
    if verbose: 
        print("Segmentation, missing:")
        for r in missing_segmentation:
            print(r)
    else:
        print("Segmentation: missing %i dsets" % len(missing_segmentation))
    if return_missing:
        return cells, missing_segmentation
    else:
        return cells

def calculate_gradients(curr_segmented_mask, img_az, img_el):
    from pipeline.python.classifications import gradient_estimation as grd

    thr_img_az = img_az.copy()
    thr_img_az[curr_segmented_mask==0] = np.nan
    grad_az = grd.calculate_gradients(thr_img_az)

    thr_img_el = img_el.copy()
    thr_img_el[curr_segmented_mask==0] = np.nan
    grad_el = grd.calculate_gradients(thr_img_el)

    return grad_az, grad_el

# #### Area assignment functions
def get_transformed_rois(animalid, session, fov, retinorun='retino_run1', 
                        roi_id=None, traceid='traces001'):
    if roi_id is None:
        roi_id = roi_utils.get_roiid_from_traceid(animalid, session, fov, traceid=traceid)
    roi_masks, zprog_img = roi_utils.load_roi_masks(animalid, session, fov, rois=roi_id)
    print("Loaded rois: %s" % roi_id)
    d1, d2, nrois = roi_masks.shape

    pixel_size = putils.get_pixel_size()    
    # pixel_size = (pixel_size[0]*ds_factor, pixel_size[1]*ds_factor)
    roi_masks_tr = np.dstack([coreg.transform_2p_fov(roi_masks[:, :, i].astype(float), pixel_size)                    for i in np.arange(0, nrois)]) # transform/orient
    roi_masks = roi_masks_tr.astype(bool).astype(int)
    
    return roi_masks


def label_roi_masks(seg_results, roi_masks):
    d1, d2, nrois = roi_masks.shape
    print("Roi masks:", d1, d2, nrois)
    roi_assignments={}
    for area_name, seg in seg_results['areas'].items():
        #seg_mask = cv2.resize(seg['mask'], (d2, d1))
        id_mask = seg['id'] * seg['mask'].astype(int)

        multi = roi_masks*id_mask[:,:,np.newaxis]
        curr_rois = np.where(multi.max(axis=0).max(axis=0)==seg['id'])[0]
        print(area_name, len(curr_rois))
        roi_assignments[area_name] = list(curr_rois)

    return roi_assignments #, roi_masks


def assign_rois_to_visual_area(animalid, session, fov, retinorun='retino_run1',roi_id=None, 
                                traceid='traces001', verbose=False):

    # Load ROIs
    roi_masks = get_transformed_rois(animalid, session, fov, retinorun=retinorun, 
                                     traceid=traceid)
    #d1, d2, nrois = roi_masks.shape

    # Load segmentation results
    seg_results, seg_params = load_segmentation_results(animalid, session, fov, 
                                                            retinorun=retinorun)

    # Assign each ROI to visual areas
    roi_assignments = label_roi_masks(seg_results, roi_masks)

    if verbose:
        for v, r in roi_assignments.items():
            print("%s: %i cells" % (v, len(r)))

    return roi_assignments


def plot_labeled_rois(labeled_image, roi_assignments, roi_masks, cmap='colorblind', 
                        surface=None, ax=None, contour_lw=1, contour_lc='w'):
    
    d1, d2, nr = roi_masks.shape
    
    defined_names = [k for k in roi_assignments.keys() if not(putils.isnumber(k))]

    color_list = sns.color_palette('colorblind', n_colors=len(defined_names))
    color_dict = dict((k, v) for k, v in zip(defined_names, color_list))

    # Plot rois on visual areas
    roi_int_img = np.zeros((d1, d2,4), dtype=float) #*np.nan

    for area_name, roi_list in roi_assignments.items():
        rc = color_dict[area_name] if area_name in defined_names else (0.5, 0.5, 0.5, 1)
        for ri in roi_list:
            curr_msk = roi_masks[:, :, ri].copy() #* color_list[0]
            roi_int_img[curr_msk>0, :] = [rc[0], rc[1], rc[2], 1] 

    if ax is None:
        fig, ax = pl.subplots(figsize=(3,4))
    
    if surface is not None:
        ax.imshow(surface, cmap='gray')
    ax.imshow(roi_int_img)
    ax = overlay_all_contours(labeled_image, ax=ax, lw=contour_lw, lc=contour_lc)

    lhandles = putils.custom_legend_markers(colors=color_list, labels=defined_names)
    bbox_to_anchor=(1.6, 1)
    ax.legend(handles=lhandles, bbox_to_anchor=bbox_to_anchor)
    
    return ax


def plot_labeled_areas(filt_azim_r, filt_elev_r, surface_2p, label_keys,
                        labeled_image_2p, labeled_image_incl, region_props, 
                        cmap_phase='nipy_spectral', pos_multiplier=(1,1)):

    fig, axn = pl.subplots(1,3, figsize=(9,3))
    ax=axn[0]
    ax.imshow(surface_2p, cmap='gray')
    ax.set_title('Azimuth')
    im0 = ax.imshow(filt_azim_r, cmap=cmap_phase)
    putils.colorbar(im0)
    ax = overlay_all_contours(labeled_image_2p, ax=ax, lw=2, lc='k')
    putils.turn_off_axis_ticks(ax, despine=False)

    ax=axn[1]
    ax.imshow(surface_2p, cmap='gray')
    im1=ax.imshow(filt_elev_r, cmap=cmap_phase)
    putils.colorbar(im1)
    ax.set_title('Elevation')
    ax = overlay_all_contours(labeled_image_2p, ax=ax, lw=2, lc='k')
    putils.turn_off_axis_ticks(ax, despine=False)

    ax=axn[2]
    ax.imshow(surface_2p, cmap='gray')
    labeled_image_incl_2p = cv2.resize(labeled_image_incl, (surface_2p.shape[1], surface_2p.shape[0]))
    ax.imshow(labeled_image_incl_2p, cmap='jet', alpha=0.5)
    ax = overlay_all_contours(labeled_image_2p, ax=ax, lw=2, lc='k')

    area_ids = [k[1] for k in label_keys]
    for region in region_props:
        if region.label in area_ids:
            region_name = str([k[0] for k in label_keys if k[1]==region.label][0])
            ax.text(region.centroid[1]*pos_multiplier[0], 
                    region.centroid[0]*pos_multiplier[1], 
                    '%s (%i)' % (region_name, region.label), fontsize=18, color='r')
            print(region_name, region.area)
    ax.set_title('Labeled (%i patches)' % len(area_ids))
    putils.turn_off_axis_ticks(ax, despine=False)

    return fig


# processing
def load_processed_maps(animalid, session, fov, retinorun='retino_run1', 
                        rootdir='/n/coxfs01/2p-data'):

    results_dir = glob.glob(os.path.join(rootdir, animalid, session, fov, retinorun, 'retino_analysis',
                            'segmentation'))
    assert len(results_dir)==1, "No segmentation results, %s" % retinorun
     
    processedmaps_fpath = os.path.join(results_dir[0], 'processed_maps.npz')
    pmaps = np.load(processedmaps_fpath)
    
    processingparams_fpath = os.path.join(results_dir[0], 'processing_params.json')
    with open(processingparams_fpath, 'r') as f:
        pparams = json.load(f)

    return pmaps, pparams

def get_processed_maps(animalid, session, fov, retinorun='retino_run1', 
                        analysis_id=None, create_new=False, pix_mag_thr=0.002, delay_map_thr=1, 
                        rootdir='/n/coxfs01/2p-data'):

    if not create_new:
        try:
            pmaps, pparams = load_processed_maps(animalid, session, fov, retinorun, rootdir=rootdir)
        except Exception as e:
            print(e)
            print(" -- procssing maps now...")
            create_new=True

    if create_new:
        # Load data metainfo
        print("Current run: %s" % retinorun)
        retinoid, RETID = ret_utils.load_retino_analysis_info(animalid, session, fov, retinorun, 
                                                              use_pixels=True)
        data_id = '_'.join([animalid, session, fov, retinorun, retinoid])
        print("DATA ID: %s" % data_id)
        curr_dst_dir = os.path.join(rootdir, animalid, session, fov, retinorun, 
                                        'retino_analysis', 'segmentation')
        # Load MW info and SI info
        mwinfo = ret_utils.load_mw_info(animalid, session, fov, retinorun)
        scaninfo = ret_utils.get_protocol_info(animalid, session, fov, run=retinorun) 
        trials_by_cond = scaninfo['trials']
     
        # Get run results
        magratio, phase, trials_by_cond = ret_utils.fft_results_by_trial(RETID)

        d2 = scaninfo['pixels_per_line']
        d1 = scaninfo['lines_per_frame']
        print("Original dims: [%i, %i]" % (d1, d2))
        ds_factor = int(RETID['PARAMS']['downsample_factor'])
        print('Data were downsampled by %i.' % ds_factor)

        # Get pixel size
        pixel_size = putils.get_pixel_size()
        pixel_size_ds = (pixel_size[0]*ds_factor, pixel_size[1]*ds_factor)

        # #### Get maps
        abs_vmin, abs_vmax = (-np.pi, np.pi)
        absolute_az, absolute_el, delay_az, delay_el = ret_utils.absolute_maps_from_conds(
                                                                magratio, phase, trials_by_cond,
                                                                mag_thr=pix_mag_thr, dims=(d1, d2),
                                                                plot_conditions=False, 
                                                                ds_factor=ds_factor)

        fig = ret_utils.plot_phase_and_delay_maps(absolute_az, absolute_el, 
                                                  delay_az, delay_el,
                                                  cmap='nipy_spectral', 
                                                   vmin=abs_vmin, vmax=abs_vmax)

        # #### Filter where delay map is not uniform (Az v El)
        fig, filt_az, filt_el = ret_utils.filter_by_delay_map(absolute_az, absolute_el, 
                                                            delay_az, delay_el, 
                                                            delay_map_thr=delay_map_thr, plot=True)
        filt_azim_r = coreg.transform_2p_fov(filt_az, pixel_size_ds, normalize=False)
        filt_elev_r = coreg.transform_2p_fov(filt_el, pixel_size_ds, normalize=False)
        putils.label_figure(fig, data_id)
        pl.savefig(os.path.join(curr_dst_dir, 'delay_map_filters.png'))

        # Save processing results + params
        processedmaps_fpath = os.path.join(curr_dst_dir, 'processed_maps.npz')
        np.savez(processedmaps_fpath, 
                 absolute_az=absolute_az, absolute_el=absolute_el,
                 filtered_az=filt_az, filtered_el=filt_el,
                 filtered_az_scaled=filt_azim_r, filtered_el_scaled=filt_elev_r)

        pmaps = np.load(processedmaps_fpath)

        processedparams_fpath = os.path.join(curr_dst_dir, 'processing_params.json')
        pparams = {'pixel_mag_thr': pix_mag_thr,
                    'ds_factor': ds_factor,
                    'delay_map_thr': delay_map_thr,
                    'dims': (d1, d2),
                    'pixel_size': pixel_size,
                    'retino_id': retinoid, 
                    'retino_run': retinorun}

        with open(processedparams_fpath, 'w') as f:
            json.dump(pparams, f, indent=4)

    return pmaps, pparams



def smooth_maps(start_az, start_el, smooth_fwhm=12, smooth_spline=2, fill_nans=True,
                smooth_spline_x=None, smooth_spline_y=None, target_sigma_um=25, 
                start_with_transformed=True, use_phase_smooth=False, ds_factor=2):
    from pipeline.python.classifications import gradient_estimation as grd

    pixel_size = putils.get_pixel_size()
    pixel_size_ds = (pixel_size[0]*ds_factor, pixel_size[1]*ds_factor)

    smooth_spline_x = smooth_spline if smooth_spline_x is None else smooth_spline_x
    smooth_spline_y = smooth_spline if smooth_spline_y is None else smooth_spline_y

    um_per_pix = np.mean(pixel_size) if start_with_transformed else np.mean(pixel_size_ds)
    smooth_fwhm = int(round(target_sigma_um/um_per_pix))  # int(25*pix_per_deg) #11
    sz=smooth_fwhm*2 #smooth_fwhm
    print("Target: %i (fwhm=%i, k=(%i, %i))" % (target_sigma_um, smooth_fwhm, smooth_spline_x, smooth_spline_y)) #, sz)
    smooth_type = 'phasenan'if use_phase_smooth else 'gaussian'

    print("start", np.nanmin(start_az), np.nanmax(start_az))
    if use_phase_smooth:
        azim_smoothed = ret_utils.smooth_phase_nans(start_az, smooth_fwhm, sz)
        elev_smoothed = ret_utils.smooth_phase_nans(start_el, smooth_fwhm, sz)
    else:
        azim_smoothed = ret_utils.smooth_neuropil(start_az, smooth_fwhm=smooth_fwhm)
        elev_smoothed = ret_utils.smooth_neuropil(start_el, smooth_fwhm=smooth_fwhm)
    print("smoothed", np.nanmin(azim_smoothed), np.nanmax(azim_smoothed))

    if fill_nans:
        azim_fillnan=None
        elev_fillnan=None
        try:
            azim_fillnan = grd.fill_and_smooth_nans_missing(azim_smoothed, 
                                                            kx=smooth_spline_x, ky=smooth_spline_x)
        except Exception as e: # sometimes if too filled, fails w ValueError
            print("[AZ] Bad NaN fill. Try a smaller target_smooth_um value")
             #azim_fillnan = grd.fill_and_smooth_nans(azim_smoothed, 
             #                                       kx=smooth_spline_x, ky=smooth_spline_x)
        try:
             elev_fillnan = grd.fill_and_smooth_nans_missing(elev_smoothed, 
                                                            kx=smooth_spline_y, ky=smooth_spline_y)
        except Exception as e:
            print("[EL] Bad NaN fill. Try a smaller target_smooth_um value")
            #elev_fillnan = elev_smoothed.copy()
            #elev_fillnan = grd.fill_and_smooth_nans(elev_smoothed, 
            #                                        kx=smooth_spline_y, ky=smooth_spline_y)
      
#        if len(np.where(np.isnan(azim_smoothed))[0])>2500:
#            azim_fillnan = grd.fill_and_smooth_nans_missing(azim_smoothed, 
#                                                            kx=smooth_spline_x, ky=smooth_spline_x)
#        else:
#            #azim_fillnan = azim_smoothed.copy()
#            azim_fillnan = grd.fill_and_smooth_nans(azim_smoothed, 
#                                                    kx=smooth_spline_x, ky=smooth_spline_x)
#        if len(np.where(np.isnan(elev_smoothed))[0])>2500:
#            elev_fillnan = grd.fill_and_smooth_nans_missing(elev_smoothed, 
#                                                            kx=smooth_spline_y, ky=smooth_spline_y)
#        else:
#            #elev_fillnan = elev_smoothed.copy()
#            elev_fillnan = grd.fill_and_smooth_nans(elev_smoothed, 
#                                                    kx=smooth_spline_y, ky=smooth_spline_y)
        print("fillnan", np.nanmin(azim_fillnan), np.nanmax(azim_fillnan))
    else:
        
        azim_fillnan = ret_utils.smooth_neuropil(azim_smoothed, smooth_fwhm=smooth_fwhm)
        elev_fillnan = ret_utils.smooth_neuropil(elev_smoothed, smooth_fwhm=smooth_fwhm)
    # Transform FOV to match widefield
    az_fill = coreg.transform_2p_fov(azim_fillnan, pixel_size, normalize=False) if not start_with_transformed else azim_fillnan
    el_fill = coreg.transform_2p_fov(elev_fillnan, pixel_size, normalize=False) if not start_with_transformed else elev_fillnan
    print("fillnan", np.nanmin(az_fill), np.nanmax(az_fill))

    azim_ = {'smoothed': azim_smoothed, 'nan_filled': azim_fillnan, 'final': az_fill}
    elev_ = {'smoothed': elev_smoothed, 'nan_filled': elev_fillnan, 'final': el_fill}

    return azim_, elev_ 


def smooth_processed_maps(animalid, session, fov, retinorun='retino_run1', 
                            target_sigma_um=25., start_with_transformed=True,
                            smooth_spline=2, use_phase_smooth=False, 
                            smooth_spline_x=None, smooth_spline_y=None,fill_nans=True,
                            reprocess=False, cmap_phase='nic_Edge', 
                            pix_mag_thr=0.002, delay_map_thr=1.0, 
                            rootdir='/n/coxfs01/2p-data'):
    from pipeline.python.classifications import gradient_estimation as grd

    if cmap_phase=='nic_Edge':
        _, cmap_phase = ret_utils.get_retino_legends(cmap_name=cmap_phase, zero_center=True, 
                                                   return_cmap=True)

    pmaps, pparams = get_processed_maps(animalid, session, fov, retinorun=retinorun,
                                        pix_mag_thr=pix_mag_thr, delay_map_thr=delay_map_thr,
                                        create_new=reprocess, rootdir=rootdir)
    curr_dst_dir = os.path.join(rootdir, animalid, session, fov, retinorun, 
                                'retino_analysis', 'segmentation')
    data_id = '%s_%s_fov%i' % (session, animalid, int(fov.split('_')[0][3:]))

    filt_az = pmaps['filtered_az']
    filt_el = pmaps['filtered_el']
    filt_azim_r = pmaps['filtered_az_scaled']
    filt_elev_r = pmaps['filtered_el_scaled']
    ds_factor = pparams['ds_factor']
    delay_map_thr = pparams['delay_map_thr']
    pix_mag_thr = pparams['pixel_mag_thr']
    smooth_type = 'phasenan'if use_phase_smooth else 'gaussian'

    # #### Smooth
    start_az = filt_azim_r.copy() if start_with_transformed else filt_az.copy()
    start_el = filt_elev_r.copy() if start_with_transformed else filt_el.copy()

    # Get smooth sparams
    pixel_size = putils.get_pixel_size()
    pixel_size_ds = (pixel_size[0]*ds_factor, pixel_size[1]*ds_factor)
    um_per_pix = np.mean(pixel_size) if start_with_transformed else np.mean(pixel_size_ds)
    smooth_fwhm = int(round(target_sigma_um/um_per_pix))  # int(25*pix_per_deg) #11

    if isinstance(smooth_spline, tuple):
        smooth_spline_x, smooth_spline_y = smooth_spline
    else:
        smooth_spline_x = smooth_spline_x if smooth_spline_x is not None else smooth_spline
        smooth_spline_y = smooth_spline_y if smooth_spline_y is not None else smooth_spline

    sm_azim, sm_elev = smooth_maps(start_az, start_el, smooth_fwhm=smooth_fwhm, fill_nans=fill_nans,
                                    smooth_spline_x=smooth_spline_x, smooth_spline_y=smooth_spline_y, 
                                    target_sigma_um=target_sigma_um, 
                                    start_with_transformed=start_with_transformed, 
                                    use_phase_smooth=use_phase_smooth, ds_factor=ds_factor)

    vmin, vmax = (-np.pi, np.pi)
    fig = grd.plot_retinomap_processing_pixels(
                                start_az, sm_azim['smoothed'], sm_azim['nan_filled'], sm_azim['final'], 
                                start_el, sm_elev['smoothed'], sm_elev['nan_filled'], sm_elev['final'],
                                               cmap_phase=cmap_phase, vmin=vmin, vmax=vmax, \
                                               smooth_fwhm=smooth_fwhm, 
                                               smooth_spline=smooth_spline,
                                               delay_map_thr=delay_map_thr, 
                                               full_cmap_range=False, show_cbar=True)
    putils.label_figure(fig, data_id)        
    figname = 'pixelmaps_smooth-%i_magthr-%.3f_delaymapthr-%.2f' % (smooth_fwhm, pix_mag_thr, delay_map_thr)
    pl.savefig(os.path.join(curr_dst_dir, '%s.png' % figname))
    print(curr_dst_dir, figname)

    # Save 
    smoothedmaps_fpath = os.path.join(curr_dst_dir, 'smoothed_maps.npz')
    np.savez(smoothedmaps_fpath,
             start_az=start_az, start_el=start_el,
             azimuth=sm_azim['final'], elevation=sm_elev['final'])
    smoothparams = {'smooth_fwhm': smooth_fwhm, 
                    'smooth_spline': (smooth_spline_x, smooth_spline_y),
                    'target_sigma_um': target_sigma_um, 
                    'start_woth_transformed': start_with_transformed,
                    'use_phase_smooth': use_phase_smooth,
                    'smooth_type': smooth_type, 'fill_nans': fill_nans}
    pparams.update(smoothparams)

    return sm_azim['final'], sm_elev['final'], pparams


def load_final_maps(animalid, session, fov, retinorun='retino_run1', return_screen=True,
                    rootdir='/n/coxfs01/2p-data'):

    curr_dst_dir = os.path.join(rootdir, animalid, session, fov, retinorun, 
                                'retino_analysis', 'segmentation')

    results_fpath = os.path.join(curr_dst_dir, 'smoothed_maps.npz')
    assert os.path.exists(results_fpath), "No smoothed maps found."

    results = np.load(results_fpath)
    
    az_fill = results['azimuth']
    el_fill = results['elevation']
    
    params_fpath = os.path.join(curr_dst_dir, 'processing_params.json')
    with open(params_fpath, 'r') as f:
        pparams = json.load(f)

    if return_screen:
        screen = putils.get_screen_dims()
        screen_max = screen['azimuth_deg']/2.
        screen_min = -screen_max
        vmin, vmax = (-np.pi, np.pi)
        img_az = putils.convert_range(az_fill, oldmin=vmin, oldmax=vmax, 
                                newmin=screen_min, newmax=screen_max)
        img_el = putils.convert_range(el_fill, oldmin=vmin, oldmax=vmax,
                                newmin=screen_min, newmax=screen_max) 
        return img_az, img_el, pparams
    else:
        return az_fill, el_fill, pparams


# Segmentation
def do_morphological_steps(S, close_k=31, open_k=131, dilate_k=31):

    # Morphological closing
    kernel =  np.ones((close_k, close_k))
    closing_s1 = cv2.morphologyEx(S, cv2.MORPH_CLOSE, kernel, iterations=1)
    # Morphological opening
    ernel = np.ones((open_k, open_k))
    opening_s1 = cv2.morphologyEx(closing_s1, cv2.MORPH_OPEN, kernel, iterations=1)
    # Morphological dilation
    kernel = np.ones((dilate_k, dilate_k))
    dilation = cv2.dilate(opening_s1, kernel, iterations=1)
    # dilation = cv2.morphologyEx(opening_1, cv2.MORPH_CLOSE, kernel, iterations=niter)

    return S, closing_s1, opening_s1, dilation

def plot_morphological_steps(S, closing_s1, opening_s1, dilation,
                            close_k=None, open_k=None, dilate_k=None):
    # Plot steps
    f, axf = pl.subplots(1,4) #pl.figure()
    axn = axf.flat
    ax=axn[0]
    ax.set_title("sign map")
    ax.imshow(S,cmap='jet')

    ax=axn[1]
    im=ax.imshow(closing_s1, cmap='jet')
    ax.set_title('closing (%i)' % close_k)

    ax=axn[2]
    im=ax.imshow(opening_s1, cmap='jet')
    ax.set_title('opening (%i)' % open_k)

    ax=axn[3]
    im=ax.imshow(dilation, cmap='jet')
    putils.colorbar(im)
    ax.set_title('dilation (%i)' % dilate_k)

    return f



# --------------------------------------------------------------------

if __name__ == '__main__':

    cmap_name = 'nic_Edge'
    rootdir = '/n/coxfs01/2p-data'
    aggr_dir = '/n/coxfs01/julianarhee/aggregate-visual-areas'
    #traceid='traces001'

    animalid = 'JC084'
    session = '20190522'
    fov = 'FOV1_zoom2p0x'
    retinorun='retino_run1'


    # In[89]:

    # Map filtering params
    pix_mag_thr=0.002
    delay_map_thr=1

    # Smoothing params
    # Marhsel et al 2014 (25um sigma)
    # For some reson fill_and_smooth_nans_missing() fails if too covered
    # smooth spline:  make 2 if >1 area
    smooth_spline=1 
    target_sigma_um=25 # 
    start_with_transformed=True
    use_phase_smooth=False

    # Segmenting params
    sign_map_thr = 0.1
    min_region_area = 2000

    absolute_S_thr=False
    close_k = 91
    open_k = 151
    dilate_k =31 


    reprocess=True
    resmooth=True
    interactive=True

    # -------------------------------------------------------------------

    #### Some aggregate plotting stuff
    visual_areas, area_colors = putils.set_threecolor_palette()
    dpi = putils.set_plot_params(lw_axes=2)

    screen, cmap_phase = ret_utils.get_retino_legends(cmap_name=cmap_name, zero_center=True, 
                                                       return_cmap=True)

    #### Set output dirs
    aggr_retino_dir = os.path.join(aggr_dir, 'retinotopy') #, 'figures', 'caiman-examples')
    if not os.path.exists(aggr_retino_dir):
        os.makedirs(aggr_retino_dir)

    ##### Metadata
    #sdata = aggr.get_aggregate_info(traceid=traceid) #, fov_type=fov_type, state=state)
    #retinodata = sdata[sdata['experiment']=='retino'].copy()
    #retinodata.groupby(['visual_area']).count()

    # Get retino runs
    fovnum = int(fov.split('_')[0][3:])
    datakey='%s_%s_fov%i' % (session, animalid, fovnum)


    found_retinoruns = [os.path.split(d)[-1] for d in 
                        glob.glob(os.path.join(rootdir, animalid, session, fov, 'retino*'))]
    print("Found %i runs" % len(found_retinoruns))
    retinorun = found_retinoruns[0] if retinorun is None else retinorun

    # Set current animal's retino output dir
    run_dir = os.path.join(rootdir, animalid, session, fov, retinorun)
    curr_dst_dir = os.path.join(run_dir, 'retino_analysis', 'segmentation')
    if not os.path.exists(curr_dst_dir):
        os.makedirs(curr_dst_dir)
        print("Saving output to:\n %s" % curr_dst_dir)

    # Params for processing/smoothing
    processedparams_fpath = os.path.join(curr_dst_dir, 'processing_params.json')

    if reprocess or resmooth:
        # #### Processing maps
        az_fill, el_fill, pparams = smooth_processed_maps(animalid, session, fov, retinorun=retinorun, 
                                                    target_sigma_um=target_sigma_um, 
                                                    start_with_transformed=start_with_transformed,
                                                    smooth_spline=smooth_spline, 
                                                    use_phase_smooth=use_phase_smooth, 
                                                    reprocess=reprocess)
        with open(processedparams_fpath, 'w') as f:
            json.dump(pparams, f, indent=4)

    else:
        az_fill, el_fill, pparams = load_final_maps(animalid, session, fov, retinorun=retinorun, 
                                                        rootdir=rootdir)


    # screen info
    screen = putils.get_screen_dims()
    screen_max = screen['azimuth_deg']/2.
    screen_min = -screen_max

    #### Convert to screen units
    vmin, vmax = (-np.pi, np.pi)
    img_az = putils.convert_range(az_fill, oldmin=vmin, oldmax=vmax, newmin=screen_min, newmax=screen_max)
    img_el = putils.convert_range(el_fill, oldmin=vmin, oldmax=vmax, newmin=screen_min, newmax=screen_max)
    vmin, vmax = (screen_min, screen_max)   

    #### Get surface img
    surface_img = ret_utils.load_2p_surface(animalid, session, fov, ch_num=1, retinorun=retinorun)
    pixel_size = putils.get_pixel_size()
    surface_2p = coreg.transform_2p_fov(surface_img, pixel_size, normalize=False)
    surface_2p = putils.adjust_image_contrast(surface_2p, clip_limit=10.0, tile_size=5)
    #fig, ax = pl.subplots(figsize=(4,3))
    #ax.imshow(surface_2p, cmap='gray')

    ##### Segement areas
    if interactive:
        while True:
            abs_or_no = raw_input("Use abs(sign map), enter <y> or <n>: ")
            absolute_S_thr = abs_or_no=='y'
            sign_map_thr = float(input("Enter sign map thr: "))
            print("Current kernel sizes for segmentation: (%i, %i, %i)" % (close_k, open_k, dilate_k))
            user_input = raw_input("Press <ENTER> to go ahead, or enter <close,open,dilate> kernel vals: ")
            if len(user_input)>0:
                close_k, open_k, dilate_k = [int(v) for v in user_input.split(',')]
                print("Updating kernel sizes: (%i, %i, %i)" % (close_k, open_k, dilate_k))
           
            # Create sign map 
            O, S_thr = segment_areas(img_az, img_el, sign_map_thr=sign_map_thr)
            S = abs(S_thr) if absolute_S_thr else S_thr.copy()
            S[np.isnan(O)]=0
            # Morphological steps
            S, closing_s1, opening_s1, dilation = do_morphological_steps(S,
                                    close_k=close_k, open_k=open_k, dilate_k=dilate_k)
            sfig = plot_morphological_steps(S, closing_s1, opening_s1, dilation, 
                                            close_k=close_k, open_k=open_k, dilate_k=dilate_k) 
            pl.show(block=False)
            pl.pause(1.0)
     
            user_confirm = raw_input("Is this good? Press <ENTER> to accept, 'R' to redo: ")
            if user_confirm != 'R':
                sfig.close()
                break
            elif user_confirm=='R':
                sfig.close()
    else:
        # Create sign map 
        O, S_thr = segment_areas(img_az, img_el, sign_map_thr=sign_map_thr)
        S = abs(S_thr) if absolute_S_thr else S_thr.copy()
        S[np.isnan(O)]=0
        # Morphological steps
        S, closing_s1, opening_s1, dilation = do_morphological_steps(S,
                                                                     close_k=close_k, open_k=open_k, dilate_k=dilate_k)
        sfig = plot_morphological_steps(S, closing_s1, opening_s1, dilation, 
                                        close_k=close_k, open_k=open_k, dilate_k=dilate_k) 
            
    # #### Update segmentation params
    seg_params = pparams.copy()
    seg_params.update({'morphological_kernels': (close_k, open_k, dilate_k),
                       'absolute_S_thr': absolute_S_thr,
                       'sign_map_thr': sign_map_thr,
                       'min_region_area': min_region_area})

    #### Label image
    region_props, labeled_image  = segment_and_label(dilation, min_region_area=min_region_area)
    region_labels = [region.label for region in region_props]
    print('Found %i regions: %s' % (len(region_labels), str(region_labels)))

    ##### Save
    orig_d1, orig_d2 = surface_2p.shape
    labeled_image_2p = cv2.resize(labeled_image.astype(np.uint8), (orig_d2, orig_d1))
    results = {'labeled_image_ds': labeled_image, 
               'labeled_image': labeled_image_2p,
               'region_props': region_props}

    ##### Plot segmentation results
    proc_info_str = 'pixthr=%.3f (delay thr=%.2f), smooth=%i (spline=%i, %s)' % (pix_mag_thr, delay_map_thr, smooth_fwhm, smooth_spline, smooth_type)
    fig = plot_segmentation_steps(img_az, img_el, surface=surface_2p, O=O, S_thr=S_thr, 
                                    params=seg_params, cmap=cmap_phase, 
                                    labeled_image=labeled_image, region_props=region_props, 
                                    label_color='w')
    putils.label_figure(fig, '%s | %s' % (data_id, proc_info_str))
    pl.subplots_adjust(hspace=0.5, bottom=0.2) 
    pl.savefig(os.path.join(curr_dst_dir, 'segemented_areas.png'))
    pl.show()

    # # Select areas and label
    while True:
        region_dict={}
        for ri, region in enumerate(region_props):
            user_label = raw_input("%i: " % region.label)
            if len(user_label)>0:
                region_dict.update({int(region.label): user_label})
        print("Labeled %i areas: " % len(region_props))
        for ri, rl in region_dict.items():
            print("Region %i: %s" % (ri, rl))
        user_confirm = raw_input("Is dis gut? <ENTER> to continue, 'R' to redo: ")
        if user_confirm != 'R':
            break

    #region_dict={1: 'Lm'} #,5:'Lm'} #, 2: 'AL', 4: 'Li'} #V1'}
    seg_areas = {}
    for ri, region in enumerate(region_props):
        region_id = region.label
        if region.label in region_dict.keys():
            region_name = region_dict[region.label]
        else:
            region_name = region.label
        # save mask
        region_mask = np.copy(labeled_image.astype('float'))
        region_mask[labeled_image != region_id] = 0
        region_mask[labeled_image == region_id] = 1
        if region_name in seg_areas.keys():
            region_mask = seg_areas[region_name]['mask'] + region_mask # Update region mask
            region_id = seg_areas[region_name]['id']
            labeled_image[region_mask==1] = seg_areas[region_name]['id'] # Update labeled image
        seg_areas[region_name] = {'id': region_id, 'mask': region_mask}


    # double check labeling/naming of segmented areas
    label_keys = [(k, v['id']) for k, v in seg_areas.items() if not(putils.isnumber(k))]
    results.update({'areas': seg_areas})
    results.update({'label_keys': label_keys})

    # Plot results with segmented areas
    pos_multiplier = (1,1) if start_with_transformed else (pixel_size[0], pixel_size[1]) 
    area_ids = [k[1] for k in label_keys]
    labeled_image_incl = np.ones(labeled_image.shape)*np.nan #labeled_image.copy()
    for idx in area_ids:
        labeled_image_incl[labeled_image==idx] = idx

    fig = plot_labeled_areas(filt_azim_r, filt_elev_r, surface_2p, label_keys,
                            labeled_image_2p, labeled_image_incl,
                            region_props, surface_2p=surface_2p, cmap_phase=cmap_phase,
                            pos_multiplier=pos_multiplier)
    pl.subplots_adjust(wspace=0.3, top=0.8)
    putils.label_figure(fig, '%s | %s' % (data_id, proc_info_str))
    pl.savefig(os.path.join(curr_dst_dir, 'labeled_areas.png'))


    # ## Save results
    segparams_fpath = os.path.join(curr_dst_dir, 'params.json')
    with open(segparams_fpath, 'w') as f:
        json.dump(seg_params, f, indent=4, sort_keys=True) 
    segresults_fpath = os.path.join(curr_dst_dir, 'results.pkl')
    with open(segresults_fpath, 'wb') as f:
        pkl.dump(results, f, protocol=pkl.HIGHEST_PROTOCOL)

    print("Completed segmentation!")
    

    # DO STUFF.
    # =====================================================================================
    # ## Calculate gradient for segmented areas
    seg_results, seg_params = load_segmentation_results(animalid, session, fov, 
                                                        retinorun=retinorun)
    segmented_areas = seg_results['areas']
    region_props = seg_results['region_props']


    contour_lc='r'
    contour_lw=1
    spacing =100
    scale = 0.001 #0.0001
    width = 0.01 #1 #0.01
    headwidth=5

    for vi, (curr_visual_area, area_results) in enumerate(segmented_areas.items()):
        print(vi, curr_visual_area)
        if putils.isnumber(curr_visual_area):
            continue
        curr_segmented_mask = area_results['mask']
        grad_az, grad_el = calculate_gradients(curr_segmented_mask, img_az, img_el)
        
        # Plot results ------------
        curr_labeled_image = np.zeros(labeled_image.shape)
        curr_labeled_image[labeled_image==area_results['id']] = 1
        fig = plot_gradients_in_area(curr_labeled_image, img_az, img_el, grad_az, grad_el, 
                                    cmap_phase=cmap_phase,
                                    contour_lc=contour_lc, contour_lw=contour_lw, 
                                    spacing=spacing, scale=scale, width=width, 
                                    headwidth=headwidth, vmin=vmin, vmax=vmax)
        pl.subplots_adjust(wspace=0.5, hspace=0.5, top=0.8)
        putils.label_figure(fig, data_id)
        fig.suptitle(curr_visual_area)

        figname = 'gradients_%s' % curr_visual_area
        pl.savefig(os.path.join(curr_dst_dir, '%s.png' % figname))
        print(curr_dst_dir, figname)


    # ## Assign cells to visual area(s)

    #### Get roi masks 
    traceid= 'traces001'
    roi_id = None
    if roi_id is None:
        roi_id = roi_utils.get_roiid_from_traceid(animalid, session, fov, traceid=traceid)
    r_masks, zprog_img = roi_utils.load_roi_masks(animalid, session, fov, rois=roi_id)
    print("Loaded rois: %s" % roi_id)
    d1, d2, nrois = r_masks.shape

    ##### Reshape and transform to match 'natural view'
    pixel_size = putils.get_pixel_size() #* ds_factor
    roi_masks_tr = np.dstack([coreg.transform_2p_fov(r_masks[:, :, i].astype(float), pixel_size)                for i in np.arange(0, nrois)]) # transform/orient
    roi_masks = roi_masks_tr.astype(bool).astype(int)
    print(roi_masks.shape)


    # ## Load segmentation results
    seg_results = load_segmentation_results(animalid, session, fov, retinorun=retinorun)
    seg_areas = seg_results['areas']


    # ## Assign roi IDs to visual area
    id_cmap='colorblind'
    contour_lw=1
    contour_lc='w'
    verbose=False
    create_new = False
    plot_rois = True

    if not create_new:
        try:
            print("Loading roi assignments")
            roi_assignments = load_roi_assignments(animalid, session, fov, retinorun=retinorun)
        except AssertionError:
            create_new=True

    if create_new:
        print("Assigning rois to visual areas...")
        # Assign each ROI to visual areas
        roi_assignments = label_roi_masks(seg_results, roi_masks)

        # Save assignments
        assignments_fpath = os.path.join(curr_dst_dir, 'roi_assignments.json')
        with open(assignments_fpath, 'w') as f:
            json.dump(roi_assignments, f, indent=4)
        create_new=False

    if plot_rois:
        print("plotting assigned rois")
        labeled_image = seg_results['labeled_image']
        f, ax = pl.subplots(figsize=(2,3), dpi=dpi)
        plot_labeled_rois(labeled_image, roi_assignments, roi_masks, cmap=id_cmap, surface=surface_2p, ax=ax,
                         contour_lw=contour_lw, contour_lc=contour_lc)
        
        putils.label_figure(fig, data_id)
        ax.set_title('%s\n%s' % (datakey, retinorun))
        pl.savefig(os.path.join(curr_dst_dir, 'assigned_rois.svg'))



