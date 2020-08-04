#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 14:39:38 2019

@author: julianarhee
"""


import os
import glob
import json
import h5py
import copy
import cv2
import imutils

import numpy as np
import pandas as pd
import pylab as pl
import seaborn as sns
import cPickle as pkl
import matplotlib.gridspec as gridspec

from pipeline.python.classifications import test_responsivity as resp
from pipeline.python.utils import label_figure, natural_keys, convert_range, get_pixel_size

from matplotlib.patches import Ellipse, Rectangle

from shapely.geometry.point import Point
from shapely import affinity
from matplotlib.patches import Polygon

from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import MaxNLocator

from shapely.geometry import box


import matplotlib_venn as mpvenn
import itertools
import time
import multiprocessing as mp


# Functions for transforming fov image/coordinates
def transform_rotate_fov_image(img):
    # Rotate 90 left (CCW) and flip L/R
    img_t = imutils.rotate(img, 90)
    img_t = np.fliplr(img_t)
  
    return img_t

def transform_rotate_coordinates(positions, ap_lim=1177., ml_lim=972.):
    # Rotate 90 degrees (ie., rotate counter clockwise about origin: (-y, x))
    # For image, where 0 is at top (y-axis points downward), 
    # then rotate CLOCKWISE, i.e., (y, -x)
    # Flip L/R, too.
    
    # (y, -x):  (pos[1], 512-pos[0]) --> 512 to make it non-neg, and align to image
    # flip l/r: (512-pos[1], ...) --> flips x-axis l/r 
    positions_t = [(ml_lim-pos[1], ap_lim-pos[0]) for pos in positions]
    
    return positions_t
        

# ############################################
# Functions for processing ROIs (masks)
# ############################################

def transform_fov_posdf(posdf, fov_keys=('fov_xpos', 'fov_ypos'), 
                         ml_lim=972, ap_lim=1177.):
    posdf_transf = posdf.copy()

    fov_xkey, fov_ykey = fov_keys
    fov_xpos = posdf_transf[fov_xkey].values
    fov_ypos = posdf_transf[fov_ykey].values

    o_coords = [(xv, yv) for xv, yv in zip(fov_xpos, fov_ypos)]
    t_coords = transform_rotate_coordinates(o_coords, ap_lim=ap_lim, ml_lim=ml_lim)
    posdf['ml_pos'] = [t[0] for t in t_coords]
    posdf['ap_pos'] = [t[1] for t in t_coords]
    
    return posdf


def calculate_roi_coords(masks, zimg, roi_list=None, convert_um=True): 
    '''
    Get FOV info relating cortical position to RF position of all cells.
    Info should be saved in: rfdir/fov_info.pkl
    
    Returns:
        fovinfo (dict)
            'positions': 
                dataframe of azimuth (xpos) and elevation (ypos) for cell's 
                cortical position and cell's RF position (i.e., 'posdf')
            'zimg': 
                (array) z-projected image 
            'roi_contours': 
                (list) roi contours, from classifications.convert_coords.contours_from_masks()
            'xlim' and 'ylim': 
                (float) FOV limits (in pixels or um) for azimuth and elevation axes
    '''

        
    print("... getting fov info")
    # Get masks
    npix_y, npix_x, nrois_total = masks.shape
    
    if roi_list is None:
        roi_list = range(nrois_total)
    
    # Create contours from maskL
    roi_contours = contours_from_masks(masks)
    
    # Convert to brain coords (scale to microns)
    fov_pos_x, fov_pos_y, xlim, ylim, centroids = get_roi_position_in_fov(roi_contours, 
                                                               roi_list=roi_list,
                                                                 convert_um=convert_um,
                                                                 npix_y=npix_y,
                                                                 npix_x=npix_x)
 
    #posdf = pd.DataFrame({'ml_pos': fov_pos_y, 'ap_pos': fov_pos_x, #fov_x,
    posdf = pd.DataFrame({'fov_xpos': fov_pos_x, # corresponds to AP axis ('ap_pos')
                          'fov_ypos': fov_pos_y, # corresponds to ML axis ('ml_pos')
                          'fov_xpos_pix': [c[0] for c in centroids],
                          'fov_ypos_pix': [c[1] for c in centroids]
                          }, index=roi_list)

    posdf = transform_fov_posdf(posdf, ml_lim=ylim, ap_lim=xlim)

    # Save fov info
    pixel_size = get_pixel_size()
    fovinfo = {'zimg': zimg,
                'convert_um': convert_um,
                'pixel_size': pixel_size,
                'roi_contours': roi_contours,
                'roi_positions': posdf,
                'ap_lim': xlim, # natural orientation AP (since 2p fov is rotated 90d)
                'ml_lim': ylim} # natural orientation ML

    return fovinfo   



def get_roi_position_in_fov(tmp_roi_contours, roi_list=None, 
                            convert_um=True, npix_y=512, npix_x=512):
                            #xaxis_conversion=2.3, yaxis_conversion=1.9):
    
    '''
    From 20190605 PSF measurement:
        xaxis_conversion = 2.312
        yaxis_conversion = 1.904
    '''
    print("not sorting")

    if not convert_um:
        xaxis_conversion = 1.
        yaxis_conversion = 1.

    else:
        (xaxis_conversion, yaxis_conversion) = get_pixel_size()

    # Get ROI centroids:
    print(tmp_roi_contours[0])
    centroids = [get_contour_center(cnt[1]) for cnt in tmp_roi_contours]

    # Convert pixels to um:
    xlinspace = np.linspace(0, npix_x*xaxis_conversion, num=npix_x)
    ylinspace = np.linspace(0, npix_y*yaxis_conversion, num=npix_y)

    xlim=xlinspace.max() if convert_um else npix_x
    ylim = ylinspace.max() if convert_um else npix_y

    if roi_list is None:
        roi_list = [cnt[1] for cnt in tmp_roi_contours] #range(len(tmp_roi_contours)) #sorted_roi_indices_xaxis))
        #print(roi_list[0:10])

    fov_pos_x = [xlinspace[pos[0]] for pos in centroids]
    fov_pos_y = [ylinspace[pos[1]] for pos in centroids]
    
#    # ---- Spatially sorted ROIs vs. RF position -----------------------
#    # Sort contours along X- or Y-axis of FOV
#    sorted_roi_indices_xaxis, sorted_roi_centroids_xaxis = spatially_sort_contours(tmp_roi_contours, dims=(npix_x, npix_y), sort_by='x', return_centroids=True)
#    sorted_roi_indices_yaxis, sorted_roi_centroids_yaxis = spatially_sort_contours(tmp_roi_contours, dims=(npix_x, npix_y), sort_by='y', return_centroids=True)
#

#    # Sorted along fov's X-axis
#
#    # Get sorted rank for indexing
#    spatial_rank_x = [sorted_roi_indices_xaxis.index(roi) \
#                        for roi in roi_list] 
#    # Get corresponding spatial position in FOV
#    pixel_order_x = [sorted_roi_centroids_xaxis[s] \
#                        for s in spatial_rank_x]    
#    pixel_order_xvals = [p[0] for p in pixel_order_x]
#    fov_pos_x = [xlinspace[p] for p in pixel_order_xvals] \
#                        if convert_um else pixel_order_xvals
#
#    # Get sorted along fov's Y-axis 
#
#    # Get sorted rank for indexing
#    spatial_rank_y = [sorted_roi_indices_yaxis.index(roi) \
#                        for roi in roi_list] 
#    # Get corresponding spatial position in FOV
#    pixel_order_y = [sorted_roi_centroids_yaxis[s] \
#                        for s in spatial_rank_y] 
#    pixel_order_yvals = [p[1] for p in pixel_order_y]
#    fov_pos_y = [ylinspace[p] for p in pixel_order_yvals] \
#                        if convert_um else pixel_order_yvals
#
    return fov_pos_x, fov_pos_y, xlim, ylim, centroids


def get_roi_position_um(rffits, tmp_roi_contours, rf_exp_name='rfs', 
                        convert_um=True, npix_y=512, npix_x=512,
                        xaxis_conversion=2.3, yaxis_conversion=1.9):
    
    '''
    Same as get_roi_position_fov, but includes RF pos

    From 20190605 PSF measurement:
        xaxis_conversion = 2.312
        yaxis_conversion = 1.904
    '''
      
   # ---- Spatially sorted ROIs vs. RF position -----------------------
    rf_rois = rffits['cell'].values #.index.tolist() 

    fov_pos_x, fov_pos_y, xlim, ylim, centroids =  get_roi_position_in_fov(
                                                tmp_roi_contours, 
                                                roi_list=rf_rois, 
                                                convert_um=convert_um, 
                                                npix_y=npix_y, npix_x=npix_x,
                                                xaxis_conversion=xaxis_conversion, 
                                                yaxis_conversion=yaxis_conversion)
    
    # Get cell's VF coords
    rf_xpos = rffits['x0'][rf_rois] 
    rf_ypos = rffits['y0'][rf_rois]

    return fov_pos_x, rf_xpos, xlim, fov_pos_y, rf_ypos, ylim


def contours_from_masks(masks):
    # Cycle through all ROIs and get their edges
    # (note:  tried doing this on sum of all ROIs, but fails if ROIs are overlapping at all)
    tmp_roi_contours = []
    for ridx in range(masks.shape[-1]):
        im = masks[:,:,ridx]
        im[im>0] = 1
        im[im==0] = np.nan #1
        im = im.astype('uint8')
        edged = cv2.Canny(im, 0, 0.9)
        tmp_cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        tmp_cnts = tmp_cnts[0] if imutils.is_cv2() else tmp_cnts[1]
        tmp_roi_contours.append((ridx, tmp_cnts[0]))
    print "Created %i contours for rois." % len(tmp_roi_contours)
    
    return tmp_roi_contours


def spatially_sort_contours(tmp_roi_contours, dims=(512, 512), 
                            sort_by='xy', return_centroids=False):

    if sort_by == 'xy':
        sorted_rois =  sorted(tmp_roi_contours, key=lambda ctr: (cv2.boundingRect(ctr[1])[1] + cv2.boundingRect(ctr[1])[0]) * dims[1])  
    elif sort_by == 'x':
        sorted_rois = sorted(tmp_roi_contours, key=lambda ctr: cv2.boundingRect(ctr[1])[0])
    elif sort_by == 'y':
        sorted_rois = sorted(tmp_roi_contours, key=lambda ctr: cv2.boundingRect(ctr[1])[1])
    else:
        print("Unknown sort-by: %s" % sort_by)
        return None
    
    sorted_ixs = [c[0] for c in sorted_rois]
    
    if return_centroids:
        sorted_contours = [get_contour_center(cnt[1]) for cnt in sorted_rois]
    else:
        sorted_contours = [cnt[1] for cnt in sorted_rois]
        
    return sorted_ixs, sorted_contours

def get_contour_center(cnt):

    M = cv2.moments(cnt)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return (cX, cY)

