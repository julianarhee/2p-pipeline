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
import pandas as pd

from skimage import exposure

from pipeline.python.rois.utils import load_roi_masks, get_roiid_from_traceid 
from pipeline.python.retinotopy import convert_coords as cc
from pipeline.python.classifications import experiment_classes as util
from pipeline.python.utils import adjust_image_contrast, label_figure, natural_keys, get_pixel_size

from pipeline.python.retinotopy import fit_2d_rfs as fitrf
from pipeline.python.classifications import evaluate_receptivefield_fits as evalrf

from pipeline.python.coregistration import align_fov as coreg

%matplotlib inline

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
    sorted_indices_x, sorted_contours_x = cc.spatially_sort_contours(roi_contours, sort_by='x', 
                                                                       dims=(npix_x, npix_y))
    sorted_indices_y, sorted_contours_y = cc.spatially_sort_contours(roi_contours, sort_by='y', 
                                                                       dims=(npix_x, npix_y))
    
    _, sorted_centroids_x = cc.spatially_sort_contours(roi_contours, sort_by='x', 
                                                         dims=(npix_x, npix_y), return_centroids=True)
    _, sorted_centroids_y = cc.spatially_sort_contours(roi_contours, sort_by='y', 
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

def sort_rois_2D(traceid_dir):

    run_dir = traceid_dir.split('/traces')[0]
    acquisition_dir = os.path.split(run_dir)[0]; acquisition = os.path.split(acquisition_dir)[1]
    session_dir = os.path.split(acquisition_dir)[0]; session = os.path.split(session_dir)[1]
    animalid = os.path.split(os.path.split(session_dir)[0])[1]
    rootdir = session_dir.split('/%s' % animalid)[0]

    # Load formatted mask file:
    mask_fpath = os.path.join(traceid_dir, 'MASKS.hdf5')
    maskfile =h5py.File(mask_fpath, 'r')

    # Get REFERENCE file (file from which masks were made):
    mask_src = maskfile.attrs['source_file']
    if rootdir not in mask_src:
        mask_src = replace_root(mask_src, rootdir, animalid, session)
    tmp_msrc = h5py.File(mask_src, 'r')
    ref_file = tmp_msrc.keys()[0]
    tmp_msrc.close()

    # Load masks and reshape to 2D:
    if ref_file not in maskfile.keys():
        ref_file = maskfile.keys()[0]
    masks = np.array(maskfile[ref_file]['Slice01']['maskarray'])
    dims = maskfile[ref_file]['Slice01']['zproj'].shape
    masks_r = np.reshape(masks, (dims[0], dims[1], masks.shape[-1]))
    print "Masks: (%i, %i), % rois." % (masks_r.shape[0], masks_r.shape[1], masks_r.shape[-1])

    # Load zprojection image:
    zproj = np.array(maskfile[ref_file]['Slice01']['zproj'])


    # Cycle through all ROIs and get their edges
    # (note:  tried doing this on sum of all ROIs, but fails if ROIs are overlapping at all)
    cnts = []
    for ridx in range(masks_r.shape[-1]):
        im = masks_r[:,:,ridx]
        im[im>0] = 1
        im[im==0] = np.nan #1
        im = im.astype('uint8')
        edged = cv2.Canny(im, 0, 0.9)
        tmp_cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        tmp_cnts = tmp_cnts[0] if imutils.is_cv2() else tmp_cnts[1]
        cnts.append((ridx, tmp_cnts[0]))
    print "Created %i contours for rois." % len(cnts)

    # Sort ROIs b y x,y position:
    sorted_cnts =  sorted(cnts, key=lambda ctr: (cv2.boundingRect(ctr[1])[1] + cv2.boundingRect(ctr[1])[0]) * zproj.shape[1] )
    cnts = [c[1] for c in sorted_cnts]
    sorted_rids = [c[0] for c in sorted_cnts]

    return sorted_rids, cnts, zproj



#%%ki#%%s

def plot_roi_positions(sorted_rids, sorted_positions, img=None, ax=None,
                        sorted_colors=None, single_color=False, cmap='Spectral',                        
                        label_rois=False, roi_list=[], fontsize=12, fontcolor='k',
                        markersize=20,  roi_color='r'):
    '''
    sorted_rids : list of roi indices (ordered by spatial position, if desired)
    positions : list of positions, order should correspond to sorted_rids order
    roi_list : list of ROIs to label, if label_rois=True
    '''
    if label_rois and len(roi_list)==0:
        roi_list = sorted_rids
        
    if sorted_colors is None:
        sorted_colors = sns.color_palette(cmap, len(sorted_rids)) 
   
    if ax is None:
        fig, ax = pl.subplots()       
    if img is None:
        ax.invert_axis() 
    else: 
        ax.imshow(img, cmap='gray')

    #for rank, rid in enumerate(sorted_rids):
    for rank, (rid, pos) in enumerate(zip(sorted_rids, sorted_positions)):
        ax.plot(pos[0], pos[1], marker='o', markersize=markersize, 
                color=sorted_colors[rank]) #rid])        
        if label_rois and rid in roi_list:
            ax.text(pos[0], pos[1], str(rid+1), fontsize=fontsize, color=fontcolor)
                
    return ax

#fig, ax = pl.subplots()
#ax = plot_roi_postitions(zimg_p, sorted_indices_y, orig_cc_y, ax=ax,
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


#%%

def plot_sorted_roi_position(fovcoords, posdf, label_rois=False,
                             transform_fov=True, convert_um=True):
    
    '''
    x-axis in FOV = posterior to anterior, from left to right (0-->512)
    y-axis in FOV = lateral to medial, from top to bottom (0-->512)
    '''
    #convert_um = True
    cmap='Spectral'
    markersize=2 
    #colors = ['k' for roi in rf_rois]
    units = 'um' if convert_um else 'pixels'


    #reload(cc)
    
    zimg = fovcoords['zimg'].copy()
    npix_y, npix_x = zimg.shape
   
    # # Sort ROIs by x,y position: 
    roi_positions = fovcoords['roi_positions'].copy()
    if transform_fov:
        #if convert_um:
        azi_key = 'ml_pos' #if transform_fov else 'fov_xpos'
        alt_key = 'ap_pos' #if transform_fov else 'fov_ypos'
        ml_lim = fovcoords['ml_lim'] if convert_um else npix_x #if transform_fov else fovcoords['ap_lim']
        ap_lim = fovcoords['ap_lim'] if convert_um else npix_y #if transform_fov else fovcoords['ml_lim']
        if not convert_um:
            # Trasnform and rotate pixel coords:
            print("transforming/rotating coors, no um conversion")
            roi_positions = cc.transform_fov_posdf(roi_positions, 
                                                   fov_keys=('fov_xpos_pix', 'fov_ypos_pix'),
                                                   ml_lim=npix_x, ap_lim=npix_y)
            # Update posdf
            posdf['ml_pos'] = [roi_positions['ml_pos'][ri] for ri in posdf.index.tolist()]
            posdf['ap_pos'] = [roi_positions['ap_pos'][ri] for ri in posdf.index.tolist()]

    else:
        azi_key = 'fov_xpos' if convert_um else 'fov_xpos_pix' 
        alt_key = 'fov_ypos' if convert_um else 'fov_ypos_pix' 
        ml_lim = fovcoords['ap_lim'] if convert_um else npix_x
        ap_lim = fovcoords['ml_lim'] if convert_um else npix_y
        
    roi_coords = [(xv, yv) for xv, yv in roi_positions[[azi_key, alt_key]].values]  
    sorted_indices_x = np.argsort(roi_positions[azi_key]).values[::-1]
    sorted_centroids_x = [roi_coords[ri] for ri in sorted_indices_x] #sorted(roi_coords, key=lambda x: x[0])
    sorted_indices_y = np.argsort(roi_positions[alt_key]).values[::-1]
    sorted_centroids_y = [roi_coords[ri] for ri in sorted_indices_y] #sorted(roi_coords, key=lambda x: x[1])

    # Color by RF position:
    rf_rois = posdf.index.tolist()
    
    zimg_p = adjust_image_contrast(zimg, clip_limit=5, tile_size=(10,10))
   
    pixel_size = get_pixel_size() if convert_um else (1, 1)
    if transform_fov:
        zimg_p = coreg.transform_2p_fov(zimg_p, pixel_size, zoom_factor=1., normalize=True) 
    else: 
        zimg_p = coreg.scale_2p_fov(zimg_p, pixel_size=(pixel_size[1], pixel_size[0]))
            
    ### Plot ALL sorted rois
    fig, axes = pl.subplots(2,2, figsize=(12,12))
    fig.patch.set_alpha(1)
   
    # Azimuth dimension
    ax = axes[0,0]
    #fig, ax = pl.subplots()
    ax = plot_roi_positions(sorted_indices_x, sorted_centroids_x, 
                            img=zimg_p, ax=ax,
                            markersize=markersize,
                            cmap=cmap, roi_list=rf_rois, label_rois=label_rois, 
                            single_color=False, fontsize=12, fontcolor='gray') #, transform=True)
               
    # Plot FOV position along AZ as a function of RF azimuth position    
    ax = axes[1,0]
    axis_matched_key = azi_key #if transform_fov else alt_key # Need to switch x, y if not transformed fov (fov rotated 90d)
    axis_matched_ixs = sorted_indices_x #if transform_fov else sorted_indices_y
    ax = scatter_sorted_fovpos_rfpos(axis_matched_ixs, posdf, roi_list=rf_rois,
                                     xax_key=axis_matched_key, 
                                     yax_key='x0' if transform_fov else 'y0', 
                                     cmap=cmap, label_rois=label_rois, ax=ax)
    ax.set_xlim([0, ml_lim])
    ax.set_title('Azimuth')
    ax.set_ylabel('RF position (rel. deg.)')
    ax.set_xlabel('FOV position (%s)' % units)
    sns.despine(offset=4, ax=ax) #trim=True, ax=ax)
 
    # Altitude dimension
    ax = axes[0,1]
    ax = plot_roi_positions(sorted_indices_y, sorted_centroids_y, 
                            img=zimg_p, ax=ax,
                            markersize=markersize,
                            cmap=cmap, roi_list=rf_rois, label_rois=label_rois, 
                            single_color=False, fontsize=12, fontcolor='gray') #, transform=True)                  
    ### Plot corresponding RF centroids:
    ax = axes[1,1]
    axis_matched_key = alt_key #if transform_fov else azi_key # Need to switch x, y if not transformed fov (fov rotated 90d)
    axis_matched_ixs = sorted_indices_y #if transform_fov else sorted_indices_x
    ax = scatter_sorted_fovpos_rfpos(axis_matched_ixs, posdf, roi_list=rf_rois,
                                     xax_key=axis_matched_key, 
                                     yax_key='y0' if transform_fov else 'x0', 
                                     cmap=cmap, label_rois=label_rois, ax=ax)
    ax.set_xlim([0, ap_lim]) 
    ax.set_title('Elevation')
    ax.set_ylabel('RF position (rel. deg.)')
    ax.set_xlabel('FOV position (%s)' % units)
    ax.axis('on')
    sns.despine(offset=4, ax=ax) #trim=True, ax=ax)
    
    pl.subplots_adjust(wspace=0.5)
    
    return fig


#plot_sorted_roi_position(fovcoords, posdf, transform=True, label_rois=True)



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

#%%


def spatially_sort_RF_versus_VF(animalid, session, fov, traceid='traces001', 
                                rfname='rfs', response_type='dff', 
                                convert_um=True, transform_fov=True, plot=True,
                                create_new=True, rootdir='/n/coxfs01/2p-data'):
    exp = util.ReceptiveFields(rfname, animalid, session, fov, 
                               traceid=traceid, trace_type='corrected')    
    # Load ROI masks
    roiid = exp.get_roi_id(traceid=traceid)    

    # Get ROI coordinates
    #print("roiid: %s | traceid: %s" % (exp.rois, exp.traceid))
    fovcoords = exp.get_roi_coordinates(create_new=create_new)
    
    # Load fit RFs
    rfdir, fit_desc = fitrf.create_rf_dir(animalid, session, fov, 'combined_%s_static' % rfname,
                                         traceid=traceid, response_type=response_type, fit_thr=fit_thr)
    data_id = '|'.join([animalid, session, fov, traceid, fit_desc])
    print(data_id)    
    fit_results, fit_params = fitrf.load_fit_results(animalid, session, fov, 
                                        experiment=rfname, traceid=traceid,
                                        response_type=response_type)
    nrois_total = fovcoords['roi_positions'].shape[0]
    
    # Convert fit reusults to dataframe
    rfdf = fitrf.rfits_to_df(fit_results['fit_results'], 
                        row_vals=fit_params['row_vals'], col_vals=fit_params['col_vals'],
                        scale_sigma=scale_sigma, sigma_scale=sigma_scale) 
    fitdf = rfdf[rfdf['r2']>fit_thr]
    rf_rois = fitdf.index.tolist()
 
    # Update position df to only include fit rois to comrae
    print("Comparing ctx position vs. retino position for %i of %i total rois" % (len(rf_rois), nrois_total))
    rcoords = fovcoords['roi_positions'].loc[rf_rois]          
    posdf = pd.concat([fitdf[['x0', 'y0']].copy(), rcoords], axis=1) 
    posdf.head()

#    # Create output dir in "summmaries" folder
#    statsdir, stats_desc = util.create_stats_dir(animalid, session, fov,
#                                                  traceid=traceid, trace_type=trace_type,
#                                                  response_type=response_type, 
#                                                  responsive_test=None, responsive_thr=0)
#    if not os.path.exists(os.path.join(statsdir, 'receptive_fields')):
#        os.makedirs(os.path.join(statsdir, 'receptive_fields'))
#    print("Saving stats output to: %s" % statsdir)    

    #transform_fov = True
   
    plot_str = '_transformed' if transform_fov else ''
    plot_str = '%s_microns' % plot_str if convert_um else plot_str
     
    #% Plot spatially ordered rois
    if plot: 
        print("... plotting: %s" % plot_str)

        fig = plot_sorted_roi_position(fovcoords, posdf, label_rois=True,
                                transform_fov=transform_fov, convert_um=convert_um)
        label_figure(fig, data_id)
        figname = 'spatially_sorted_rois_RFpos_VFpos_%s__%s_labeled' % (rfname, plot_str)
        pl.savefig(os.path.join(rfdir, '%s.svg' % figname))
        pl.close()

        fig = plot_sorted_roi_position(fovcoords, posdf, label_rois=False,
                                transform_fov=transform_fov, convert_um=convert_um)
        label_figure(fig, data_id)
        figname = 'spatially_sorted_rois_RFpos_VFpos_%s__%s' % (rfname, plot_str)
        pl.savefig(os.path.join(rfdir, '%s.svg' % figname))





# %%
%