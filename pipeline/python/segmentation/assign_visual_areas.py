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
    print('[EL] avg dir: %.2f deg' % avg_dir_el)
    vhat_el = grad_el['vhat']
    avg_dir_az = np.rad2deg(grad_az['mean_direction'])
    print('[AZ] avg dir: %.2f deg' % avg_dir_az)
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

def overlay_contours(labeled_image, ax=None, lc='w', lw=2):
    if ax is None:
        fig, ax = pl.subplots()
    
    label_ids = [l for l in np.unique(labeled_image) if l!=0]
    for label in label_ids: # range(1, labeled_image.max()):
        #label = props[index].label
        contour = skmeasure.find_contours(labeled_image == label, 0.5)[0]
        ax.plot(contour[:, 1], contour[:, 0], lc, lw=lw)
    return ax

def plot_segmentation_steps(img_az, img_el, surface=None, O=None, S_thr=None, 
                            sign_map_thr=None, cmap='viridis', 
                            labeled_image=None, region_props=None):

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

def load_segmentation_results(animalid, session, fov, retinorun='retino_run1', rootdir='/n/coxfs01/2p-data'):
    results_fpath = os.path.join(rootdir, animalid, session, fov, retinorun, 
                              'retino_analysis', 'segmentation', 'results.pkl')
    
    assert os.path.exists(results_fpath), "Segmentation not found: %s" % results_fpath
    with open(results_fpath, 'r') as f:
        seg_areas = pkl.load(f)
        
    return seg_areas

def get_gradients_in_area(curr_segmented_mask, img_az, img_el):
    thr_img_az = img_az.copy()
    thr_img_az[curr_segmented_mask==0] = np.nan
    grad_az = grd.calculate_gradients(thr_img_az)

    thr_img_el = img_el.copy()
    thr_img_el[curr_segmented_mask==0] = np.nan
    grad_el = grd.calculate_gradients(thr_img_el)

    return grad_az, grad_el


