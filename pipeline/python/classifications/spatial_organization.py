#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 11:00:17 2020

@author: julianarhee
"""
#%%
import matplotlib as mpl
mpl.use('agg')
import os
import glob
import json
#import h5py
import copy
#import cv2
#import imutils
import sys
import optparse
import shutil
import traceback
import time
import imutils

import seaborn as sns
import numpy as np
import pylab as pl

from pipeline.python.rois.utils import load_roi_masks, get_roiid_from_traceid, plot_roi_centroids
from pipeline.python.retinotopy import convert_coords as coor
from pipeline.python.classifications import experiment_classes as util

from skimage import exposure

#%%
def spatially_sort_compare_position(fovcoords, posdf, transform=True):
    
    '''
    x-axis in FOV = posterior to anterior, from left to right (0-->512)
    y-axis in FOV = lateral to medial, from top to bottom (0-->512)
    '''
    
    zimg = fovcoords['zimg']
    roi_contours = fovcoords['roi_contours']
    #posdf = fovcoords['positions']
    npix_y, npix_x = zimg.shape
    
    # # Sort ROIs by x,y position:
    sorted_indices_x, sorted_contours_x = coor.spatially_sort_contours(roi_contours, sort_by='x', 
                                                                       dims=(npix_x, npix_y))
    sorted_indices_y, sorted_contours_y = coor.spatially_sort_contours(roi_contours, sort_by='y', 
                                                                       dims=(npix_x, npix_y))
    
    _, sorted_centroids_x = coor.spatially_sort_contours(roi_contours, sort_by='x', 
                                                         dims=(npix_x, npix_y), return_centroids=True)
    _, sorted_centroids_y = coor.spatially_sort_contours(roi_contours, sort_by='y', 
                                                         dims=(npix_x, npix_y), return_centroids=True)

    # Color by RF position:
    rf_rois = posdf.index.tolist()
    
    ### Plot ALL sorted rois
    convert_um = True
    clip_limit=0.05
    fig, axes = pl.subplots(2,2, figsize=(12,12))
    fig.patch.set_alpha(1)
    
    # "xaxis" corrresponds to the AP axis after transformed
    ax = axes[0,1] 
    util.plot_roi_contours(zimg, sorted_indices_x, sorted_contours_x, 
                           label_rois=rf_rois, label=False, single_color=False, overlay=True,
                           clip_limit=clip_limit, draw_box=False, thickness=3, 
                           ax=ax, transform=transform, font_scale=0.9)
    #ax.axis('off')
#    for ri, cnt in zip(sorted_roi_indices_xaxis, sorted_roi_contours_xaxis):
#        rid = roi_contours[ri][0]+1
#        ax.text(cnt[0][0][0], cnt[0][0][1], rid, color='w')
#        
#                        
    ax = axes[0,0]
    util.plot_roi_contours(zimg, sorted_indices_y, sorted_contours_y,
                           label_rois=rf_rois, label=False, single_color=False, overlay=True,
                           clip_limit=clip_limit, draw_box=False, thickness=3, 
                           ax=ax, transform=transform)
    #ax.axis('off')
                    
    ### Plot corresponding RF centroids:
    colors = ['k' for roi in rf_rois]
    units = 'um' if convert_um else 'pixels'
    
    # Get values for azimuth:    
    ax = axes[1,0]
    ax.scatter(posdf['ml_pos'], posdf['x0'], c=colors, alpha=0.5)
    ax.set_title('Azimuth')
    ax.set_ylabel('RF position (rel. deg.)')
    ax.set_xlabel('FOV position (%s)' % units)
    ax.set_xlim([0, fovcoords['ml_lim']])
    sns.despine(offset=4, ax=ax) #trim=True, ax=ax)
    
    # Get values for elevation:
    ax = axes[1,1]
    ax.scatter(posdf['ap_pos'], posdf['y0'], c=colors, alpha=0.5)
    ax.set_title('Elevation')
    ax.set_ylabel('RF position (rel. deg.)')
    ax.set_xlabel('FOV position (%s)' % units)
    ax.set_xlim([0, fovcoords['ap_lim']])
    ax.axis('on')
    sns.despine(offset=4, ax=ax) #trim=True, ax=ax)
    
    pl.subplots_adjust(wspace=0.5)
    
    return fig


#%%

#%%ki#%%s

def adjust_grayscale_image(zimg, clip_limit=0.01):
    im_adapthist = exposure.equalize_adapthist(zimg, clip_limit=clip_limit)
    im_adapthist *= 256
    im_adapthist= im_adapthist.astype('uint8')
    #ax.imshow(im_adapthist) #pl.figure(); pl.imshow(refRGB) # cmap='gray')
    orig = im_adapthist.copy()

    return orig

def transform_rotate_fov_image(img):
    # Rotate 90 left (CCW) and flip L/R
    img_t = imutils.rotate(img, 90)
    img_t = np.fliplr(img_t)
  
    return img_t

def transform_rotate_coordinates(positions, ap_lim=512, ml_lim=512):
    # Rotate 90 degrees (ie., rotate counter clockwise about origin: (-y, x))
    # For image, where 0 is at top (y-axis points downward), 
    # then rotate CLOCKWISE, i.e., (y, -x)
    # Flip L/R, too.
    
    # (y, -x):  (pos[1], 512-pos[0]) --> 512 to make it non-neg, and align to image
    # flip l/r: (512-pos[1], ...) --> flips x-axis l/r 
    positions_t = [(ml_lim-pos[1], ap_lim-pos[0]) for pos in positions]
    
    return positions_t
        

def plot_roi_centers(zimg, sorted_rids, positions, clip_limit=0.01, 
                        sorted_colors=None, overlay=True, label_rois=False, roi_list=[],
                        markersize=20,  roi_color='r',
                        single_color=False, ax=None, fontsize=12, fontcolor='k',
                        cmap='Spectral', transform=False):
    '''
    sorted_rids : list of roi indices
    positions : lits of positions that correspond to original roi indices 

    '''
    if label_rois and len(roi_list)==0:
        roi_list = sorted_rids
        
    if sorted_colors is None:
        sorted_colors = sns.color_palette(cmap, len(sorted_rids)) 

    img = adjust_grayscale_image(zimg, clip_limit=clip_limit) if overlay else np.zeros(zimg.shape).astype('uint8')
    
    if ax is None:
        fig, ax = pl.subplots()       
    ax.imshow(img, cmap='gray')

    #for rank, rid in enumerate(sorted_rids):
    for rank, (rid, pos) in enumerate(zip(sorted_rids, positions)):
        #pos = positions[rid]
        ax.plot(pos[0], pos[1], marker='o', markersize=markersize, 
                color=sorted_colors[rank]) #rid])
        
        if label_rois and rid in roi_list:
            #print(rank, rid, pos)
            ax.text(pos[0], pos[1], str(rid+1), fontsize=fontsize, color=fontcolor)
                
    return ax

#fig, ax = pl.subplots()
#ax = plot_roi_centers(zimg_p, sorted_indices_y, orig_cc_y, ax=ax,
#            markersize=markersize,
#            cmap=cmap, label_rois=rf_rois, label=True, 
#            single_color=False, overlay=True,
#            clip_limit=clip_limit, fontsize=12, fontcolor='w') #, transform=True)
#


#sorted_indices_x, sorted_centroids_x, ax=None,
#%%
def scatter_sorted_fovpos_rfpos(sorted_indices_y, posdf, roi_list=[],
                                xax_key='ml_pos', yax_key='x0', 
                                cmap='Spectral', label_rois=False, ax=None):
 
    if len(roi_list) == 0:
        roi_list = sorted(sorted_indices_y)
         
    # Get list of colors 
    sorted_colors = sns.color_palette(cmap, len(sorted_indices_y)) 
     
    # Get selected cells in the correct (sorted) order) 
    selected_ris_y = [ri for ri in sorted_indices_y if ri in roi_list]
    
    # Get corresponding xaxis values, yaxis values
    selected_ml = [posdf[xax_key][ri] for ri in selected_ris_y]
    selected_x0 = [posdf[yax_key][ri] for ri in selected_ris_y]
    
    # Get corresponding colors
    selected_colors = [sorted_colors[ii] for ii, ri in enumerate(sorted_indices_y)\
                        if ri in selected_ris_y]

    if ax is None:
        fig, ax = pl.subplots()    
    # Plot
    ax.scatter(selected_ml, selected_x0, c=selected_colors, alpha=0.9)
    ax.set_xlabel(xax_key)
    ax.set_ylabel(yax_key)
     
    if label_rois:# label
        for xv, yv, ri in zip(selected_ml, selected_x0, selected_ris_y):
            if ri in roi_list:
                ax.text(xv, yv, int(ri+1))
    
    return ax


def plot_sorted_roi_position(fovcoords, posdf, transform=True, label_rois=False):
    
    '''
    x-axis in FOV = posterior to anterior, from left to right (0-->512)
    y-axis in FOV = lateral to medial, from top to bottom (0-->512)
    '''
    zimg = fovcoords['zimg'].copy()
    roi_contours = copy.copy(fovcoords['roi_contours'])
    #posdf = fovcoords['positions']
    npix_y, npix_x = zimg.shape
    
    # # Sort ROIs by x,y position:   
    sorted_indices_x, sorted_centroids_x = coor.spatially_sort_contours(roi_contours, sort_by='x', 
                                                         dims=(npix_x, npix_y), return_centroids=True)
    sorted_indices_y, sorted_centroids_y = coor.spatially_sort_contours(roi_contours, sort_by='y', 
                                                         dims=(npix_x, npix_y), return_centroids=True)

    # Color by RF position:
    rf_rois = posdf.index.tolist()

    zimg_p = ((zimg.astype(np.float64)/zimg.max())*255).astype(np.uint8)
    zimg_p.min(), zimg_p.max(), zimg_p.dtype
   
    ### Plot ALL sorted rois
    convert_um = True
    clip_limit=0.05
    cmap='Spectral'
    markersize=2 
    #colors = ['k' for roi in rf_rois]
    units = 'um' if convert_um else 'pixels'
 
    #orig_cc_x = copy.copy(sorted_centroids_x)
    #orig_cc_y = copy.copy(sorted_centroids_y)
    
    if transform:
        zimg_p = transform_rotate_fov_image(zimg_p)
        sc_x = copy.copy(sorted_centroids_x)
        sc_y = copy.copy(sorted_centroids_y)
        zimg_d1, zimg_d2 = zimg_p.shape
        sorted_centroids_x = transform_rotate_coordinates(sc_x, ap_lim=zimg_d1, ml_lim=zimg_d2)
        sorted_centroids_y = transform_rotate_coordinates(sc_y, ap_lim=zimg_d1, ml_lim=zimg_d2)

    fig, axes = pl.subplots(2,2, figsize=(12,12))
    fig.patch.set_alpha(1)
   
    # "y" corresponds to ML axis after transformation
    # i.e., sorted along y-axis goes w/ desired sorting along ML in fov view
    ax = axes[0,0]
    ax = plot_roi_centers(zimg_p, sorted_indices_y, sorted_centroids_y, ax=ax,
                markersize=markersize,
                cmap=cmap, roi_list=rf_rois, label_rois=label_rois, 
                single_color=False, overlay=True,
                clip_limit=clip_limit, fontsize=12, fontcolor='w') #, transform=True)

    # Plot FOV position along AZ as a function of RF azimuth position    
    ax = axes[1,0]
    ax = scatter_sorted_fovpos_rfpos(sorted_indices_y, posdf, roi_list=rf_rois,
                                     xax_key='ml_pos', yax_key='x0', 
                                     cmap=cmap, label_rois=label_rois, ax=ax)
    #ax.scatter(posdf['ml_pos'], posdf['x0'], c=colors, alpha=0.5)
    ax.set_title('Azimuth')
    ax.set_ylabel('RF position (rel. deg.)')
    ax.set_xlabel('FOV position (%s)' % units)
    ax.set_xlim([0, fovcoords['ml_lim']])
    sns.despine(offset=4, ax=ax) #trim=True, ax=ax)
 
    # "xaxis" corrresponds to the AP axis after transformed
    ax = axes[0,1]
    plot_roi_centers(zimg_p, sorted_indices_x, sorted_centroids_x, ax=ax, #sorted_centroids_x, ax=ax,
                    markersize=markersize,
                    cmap=cmap, roi_list=rf_rois, label_rois=label_rois, 
                    single_color=False, overlay=True,
                    clip_limit=clip_limit, fontsize=12) #, transform=True)
                   
    ### Plot corresponding RF centroids:
    ax = axes[1,1]
    ax = scatter_sorted_fovpos_rfpos(sorted_indices_x, posdf, roi_list=rf_rois,
                                    xax_key='ap_pos', yax_key='y0', 
                                    cmap=cmap, label_rois=label_rois, ax=ax)
    ax.set_title('Elevation')
    ax.set_ylabel('RF position (rel. deg.)')
    ax.set_xlabel('FOV position (%s)' % units)
    ax.set_xlim([0, fovcoords['ap_lim']])
    ax.axis('on')
    sns.despine(offset=4, ax=ax) #trim=True, ax=ax)
    
    pl.subplots_adjust(wspace=0.5)
    
    return fig


plot_sorted_roi_position(fovcoords, posdf, transform=True, label_rois=True)





#%%

rootdir = '/n/coxfs01/2p-data'
animalid = 'JC084'
session = '20190522'
fov = 'FOV1_zoom2p0x'
traceid = 'traces001'
rfname = 'rfs'
response_type = 'dff'

scale_sigma=True
sigma_scale = 2.35
fit_thr = 0.5

from pipeline.python.retinotopy import convert_coords as cc
from pipeline.python.retinotopy import fit_2d_rfs as fitrf
from pipeline.python.classifications import evaluate_receptivefield_fits as evalrf
from pipeline.python.classifications import experiment_classes as util

from pipeline.python.utils import label_figure

#%%

#    # Create output dir in "summmaries" folder
#    statsdir, stats_desc = util.create_stats_dir(animalid, session, fov,
#                                                  traceid=traceid, trace_type=trace_type,
#                                                  response_type=response_type, 
#                                                  responsive_test=None, responsive_thr=0)
#    if not os.path.exists(os.path.join(statsdir, 'receptive_fields')):
#        os.makedirs(os.path.join(statsdir, 'receptive_fields'))
#    print("Saving stats output to: %s" % statsdir)    
    reload(util)
  
    exp = util.ReceptiveFields(rfname, animalid, session, fov, 
                               traceid=traceid, trace_type='corrected')    
    # Load ROI masks
    roiid = exp.get_roi_id(traceid=traceid)    
    masks, zimg = exp.load_masks(rois=roiid)

    # Get ROI coordinates
    print("roiid: %s | traceid: %s" % (exp.rois, exp.traceid))
    fovcoords = exp.get_roi_coordinates(convert_um=True, create_new=True)

    # Load fit RFs
    rfdir, fit_desc = fitrf.create_rf_dir(animalid, session, fov, 'combined_%s_static' % rfname,
                                         traceid=traceid, response_type=response_type, fit_thr=fit_thr)
    data_id = '|'.join([animalid, session, fov, traceid, fit_desc])
    print(data_id)
    
    fit_results, fit_params = fitrf.load_fit_results(animalid, session, fov, 
                                        experiment=rfname, traceid=traceid,
                                        response_type=response_type)
    nrois_total = fovcoords['positions'].shape[0]
    
    # Convert fit reusults to dataframe
    rfdf = fitrf.rfits_to_df(fit_results['fit_results'], 
                        row_vals=fit_params['row_vals'], col_vals=fit_params['col_vals'],
                        scale_sigma=scale_sigma, sigma_scale=sigma_scale) 
    fitdf = rfdf[rfdf['r2']>fit_thr]
    rf_rois = fitdf.index.tolist()
 
    # Update position df to only include fit rois to comrae
    print("Comparing ctx position vs. retino position for %i of %i total rois" % (len(rf_rois), nrois_total))
    posdf = fitdf[['x0', 'y0']].copy()
    posdf['ap_pos'] = fovcoords['positions']['ap_pos'][rf_rois].copy()
    posdf['ml_pos'] = fovcoords['positions']['ml_pos'][rf_rois].copy()
   
    posdf.head()

#    # Create output dir in "summmaries" folder
#    statsdir, stats_desc = util.create_stats_dir(animalid, session, fov,
#                                                  traceid=traceid, trace_type=trace_type,
#                                                  response_type=response_type, 
#                                                  responsive_test=None, responsive_thr=0)
#    if not os.path.exists(os.path.join(statsdir, 'receptive_fields')):
#        os.makedirs(os.path.join(statsdir, 'receptive_fields'))
#    print("Saving stats output to: %s" % statsdir)    

    transform_fov = True
    #% Plot spatially ordered rois
    if len(glob.glob(os.path.join(statsdir, 'receptive_fields', 'spatially_sorted*.svg'))) == 0: 
        print("Getting FOV info for rois.")
        fig = plot_sorted_roi_position(fovcoords, posdf, transform=True, label_rois=True)
        label_figure(fig, data_id)
        figname = 'spatially_sorted_rois_RFpos_VFpos_%s_labeled' % rfname 
        pl.savefig(os.path.join(rfdir, '%s.svg' % figname))
        pl.close()

        fig = plot_sorted_roi_position(fovcoords, posdf, transform=True, label_rois=False)
        label_figure(fig, data_id)
        figname = 'spatially_sorted_rois_RFpos_VFpos_%s' % rfname 
        pl.savefig(os.path.join(rfdir, '%s.svg' % figname))
  
r        #fig = spatially_sort_compare_position(fovcoords, posdf, transform=transform_fov)
t