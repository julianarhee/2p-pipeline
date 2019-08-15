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

from pipeline.python.classifications import experiment_classes as util
from pipeline.python.classifications import test_responsivity as resp
from pipeline.python.classifications import run_experiment_stats as rstats
from pipeline.python.utils import label_figure, natural_keys, convert_range

from pipeline.python.retinotopy import fit_2d_rfs as fitrf
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

# ############################################
# Functions for processing ROIs (masks)
# ############################################

def get_roi_position_um(rffits, tmp_roi_contours, rf_exp_name='rfs', convert_um=True, npix_y=512, npix_x=512,
                        xaxis_conversion=2.312, yaxis_conversion=1.904):
    
    '''
    From 20190605 PSF measurement:
        xaxis_conversion = 2.312
        yaxis_conversion = 1.904
    '''
    
    # Sort ROIs b y x,y position:
    sorted_roi_indices_xaxis, sorted_roi_contours_xaxis = spatially_sort_contours(tmp_roi_contours, dims=(npix_x, npix_y), sort_by='x')
    sorted_roi_indices_yaxis, sorted_roi_contours_yaxis = spatially_sort_contours(tmp_roi_contours, dims=(npix_x, npix_y), sort_by='y')

    _, sorted_roi_centroids_xaxis = spatially_sort_contours(tmp_roi_contours, dims=(npix_x, npix_y), sort_by='x', get_centroids=True)
    _, sorted_roi_centroids_yaxis = spatially_sort_contours(tmp_roi_contours, dims=(npix_x, npix_y), sort_by='y', get_centroids=True)

    # Convert pixels to um:
    xlinspace = np.linspace(0, npix_x*xaxis_conversion, num=npix_x)
    ylinspace = np.linspace(0, npix_y*yaxis_conversion, num=npix_y)
    
    # ---- Spatially sorted ROIs vs. RF position -----------------------
    rf_rois = rffits.index.tolist() 
    #colors = ['k' for _ in range(len(rf_rois))]
    # Get values for azimuth:
    spatial_rank_x = [sorted_roi_indices_xaxis.index(roi) for roi in rf_rois] # Get sorted rank for indexing
    pixel_order_x = [sorted_roi_centroids_xaxis[s] for s in spatial_rank_x]    # Get corresponding spatial position in FOV
    pixel_order_xvals = [p[0] for p in pixel_order_x]
    if convert_um:
        fov_pos_x = [xlinspace[p] for p in pixel_order_xvals]
        xlim=xlinspace.max()
    else:
        fov_pos_x = pixel_order_xvals
        xlim = npix_x
    rf_xpos = rffits['x0'][rf_rois] #[gdfs[rf_exp_name].fits.loc[roi]['x0'] for roi in rf_rois]

    # Get values for elevation
    spatial_rank_y = [sorted_roi_indices_yaxis.index(roi) for roi in rf_rois] # Get sorted rank for indexing
    pixel_order_y = [sorted_roi_centroids_yaxis[s] for s in spatial_rank_y]    # Get corresponding spatial position in FOV
    pixel_order_yvals = [p[1] for p in pixel_order_y]
    if convert_um:
        fov_pos_y = [ylinspace[p] for p in pixel_order_yvals]
        ylim = ylinspace.max()
    else:
        fov_pos_y = pixel_order_yvals
        ylim = npix_y
    rf_ypos = rffits['y0'][rf_rois] #[gdfs[rf_exp_name].fits.loc[roi]['y0'] for roi in rf_rois]

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


def spatially_sort_contours(tmp_roi_contours, dims=(512, 512), sort_by='xy', get_centroids=False):

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
    
    if get_centroids:
        sorted_contours = [get_contour_center(cnt[1]) for cnt in sorted_rois]
    else:
        sorted_contours = [cnt[1] for cnt in sorted_rois]
        
    return sorted_ixs, sorted_contours

def get_contour_center(cnt):

    M = cv2.moments(cnt)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return (cX, cY)


#%%

# ############################################
# Functions for processing visual field coverage
# ############################################

def create_ellipse(center, lengths, angle=0):
    """
    create a shapely ellipse. adapted from
    https://gis.stackexchange.com/a/243462
    """
    circ = Point(center).buffer(1)
    ell = affinity.scale(circ, int(lengths[0]), int(lengths[1]))
    ellr = affinity.rotate(ell, angle)
    return ellr
