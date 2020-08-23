#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 16:43:28 2020

@author: julianarhee
"""

# In[5]:
import matplotlib
matplotlib.use('agg')
import glob
import os
import shutil
import traceback
import json
import cv2
import h5py
import imutils
import sys
import optparse

import numpy as np
import pylab as pl
import seaborn as sns
import pandas as pd
import scipy.stats as spstats
import tifffile as tf
import matplotlib.colors as mcolors
import cPickle as pkl

from mpl_toolkits.axes_grid1 import make_axes_locatable
from pipeline.python.utils import convert_range

from pipeline.python.utils import natural_keys, label_figure
from pipeline.python import utils as putils
from pipeline.python.retinotopy import utils as ret_utils
from pipeline.python.rois import utils as roi_utils
from pipeline.python.paradigm import utils as par_utils
from pipeline.python.coregistration import align_fov as coreg
from pipeline.python.classifications import evaluate_receptivefield_fits as evalrf

from scipy import misc,interpolate,stats,signal,ndimage
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage.morphology import binary_dilation
from scipy.interpolate import SmoothBivariateSpline

from sklearn.linear_model import LinearRegression, Ridge, Lasso
#from sklearn.model_selection import train_test_split
import scipy.stats as spstats
import sklearn.metrics as skmetrics #import mean_squared_error


#%%
# Functions for dilating and smoothing masks
# ---------------------------------------------------------------
from scipy.interpolate import SmoothBivariateSpline

def fill_and_smooth_nans(img):

    y, x = np.meshgrid(np.arange(0, img.shape[1]), np.arange(0, img.shape[0]))
    x = x.astype(float)
    y = y.astype(float)
    z = img.copy()
    
    xx = x.copy()
    yy = y.copy()
    xx[np.isnan(z)] = np.nan
    yy[np.isnan(z)] = np.nan

    xx=xx.ravel()
    xx=(xx[~np.isnan(xx)])
    yy=yy.ravel()
    yy=(yy[~np.isnan(yy)])
    zz=z.ravel()
    zz=(zz[~np.isnan(zz)])

#     xnew = np.arange(xx.min(), xx.max()+1) #np.arange(9,11.5, 0.01)
#     ynew = np.arange(yy.min(), yy.max()+1) #np.arange(10.5,15, 0.01)

#     f = SmoothBivariateSpline(xx,yy,zz,kx=1,ky=1)
#     znew=np.transpose(f(xnew, ynew)).T

    xnew = np.arange(x.ravel().min(), x.ravel().max()+1) #np.arange(9,11.5, 0.01)
    ynew = np.arange(y.ravel().min(), y.ravel().max()+1) #np.arange(10.5,15, 0.01)
    
    #print(xnew.min(), xnew.max())
    
    f = SmoothBivariateSpline(xx,yy,zz,kx=1,ky=1)
    znew=f(xnew, ynew) #).T
    
    znew[np.isnan(z)] = np.nan
    
    #print(z.shape, znew.shape)
    return znew #.T #a

def fill_and_smooth_nans_missing(img):
    '''
    Smooths image and fills over NaNs. Useful for dealing with holes from neuropil masks
    '''
    y, x = np.meshgrid(np.arange(0, img.shape[1]), np.arange(0, img.shape[0]))
    x = x.astype(float)
    y = y.astype(float)
    z = img.copy()

    x[np.isnan(z)] = np.nan
    y[np.isnan(z)] = np.nan

    x=x.ravel()
    x=(x[~np.isnan(x)])
    y=y.ravel()
    y=(y[~np.isnan(y)])
    z=z.ravel()
    z=(z[~np.isnan(z)])

    xnew = np.arange(x.min(), x.max()+1) #np.arange(9,11.5, 0.01)
    ynew = np.arange(y.min(), y.max()+1) #np.arange(10.5,15, 0.01)

    f = SmoothBivariateSpline(x,y,z,kx=1,ky=1)
    znew=np.transpose(f(xnew, ynew))

    return znew.T #a

def dilate_mask_centers(maskcenters, kernel_size=9):
    '''Calculate center of soma, then dilate to create masks for smoothed  neuropil
    '''
    kernel_radius = (kernel_size - 1) // 2
    x, y = np.ogrid[-kernel_radius:kernel_radius+1, -kernel_radius:kernel_radius+1]
    dist = (x**2 + y**2)**0.5 # shape (kernel_size, kernel_size)

    # let's create three kernels for the sake of example
    radii = np.array([kernel_size/3., kernel_size/2.5, kernel_size/2.])[...,None,None] # shape (num_radii, 1, 1)
    # using ... allows compatibility with arbitrarily-shaped radius arrays

    kernel = (1 - (dist - radii).clip(0,1)).sum(axis=0)# shape (num_radii, kernel_size, kernel_size)

    dilated_masks = np.zeros(maskcenters.shape, dtype=maskcenters.dtype)
    for roi in range(maskcenters.shape[0]):
        img = maskcenters[roi, :, :].copy()
        x, y = np.where(img>0)
        centroid = (sum(x) / len(x), sum(y) / len(x))
        #print(centroid)
        np_tmp = np.zeros(img.shape, dtype=bool)
        np_tmp[centroid] = True
        dilation = binary_dilation(np_tmp, structure=kernel )
        dilated_masks[roi, : :] = dilation
    return dilated_masks

def mask_rois(masks, value_array, mask_thr=0.1, return_array=False):
    '''
    Assign a value to each mask (value_array should match N masks).
    value_array:  indices should be RIDs, these ixs are used to index into masks.
    '''
    nrois, d1, d2 = masks.shape
    dims = (d1, d2)

    if return_array:
        value_mask = np.ones(masks.shape)*np.nan #-100
        for rid in value_array.index.tolist():
            value_mask[rid, masks[rid,:,:]>=mask_thr] = value_array[rid]
    else:
        value_mask =  np.ones(dims)*-100
        for rid in value_array.index.tolist():
            value_mask[masks[rid,:,:]>=mask_thr] = value_array[rid]

    return value_mask

def mask_with_overlaps_averaged(dilated_masks, value_array, mask_thr=0.1,
                                is_circular=False, vmin=-np.pi, vmax=np.pi):
    '''
    Assign value to masks, but average any masks where they are overlapping.
    value_array:  indices should be RIDs, these ixs are used to index into masks.
    '''
    nrois_total, d1, d2 = dilated_masks.shape
    #print("dilated: %i" % nrois_total) 
    #print("N rois: %i" % len(value_array))

    # Get non-averaged array
    tmpmask = mask_rois(dilated_masks, value_array, mask_thr=mask_thr, return_array=False)
    
    # Get full array to average across overlapping pixels
    tmpmask_full = mask_rois(dilated_masks, value_array, mask_thr=mask_thr, return_array=True)
    tmpmask_r = np.reshape(tmpmask_full, (nrois_total, d1*d2))
    
    # Replace overlapping pixels with average value
    avg_mask = tmpmask.copy().ravel()
    multi_ixs = [i for i in range(tmpmask_r.shape[-1]) if len(np.where(tmpmask_r[:, i])[0]) > 1]
    for ix in multi_ixs:
        #avg_azim[ix] = spstats.circmean([v for v in azim_phase2[:, ix] if not np.isnan(v)], low=vmin, high=vmax)
        avg_mask[ix] = np.nanmean([v for v in tmpmask_r[:, ix] if not np.isnan(v)])#, low=vmin, high=vmax)

    avg_mask = np.reshape(avg_mask, (d1, d2))

    return avg_mask

def get_phase_masks(masks, phases, average_overlap=True, roi_list=None, 
                    use_cont=True, mask_thr=0.01, is_circular=True):
    '''
    Create masks with assigned phase values (converted as specfied).
    To create continueous maps, use_cont=True.
    '''
    # Convert phase to continuous:
    phases_cont = -1 * phases
    phases_cont = phases_cont % (2*np.pi)
    
    # Only include specified rois:
    if roi_list is None:
        roi_list = phases.index.tolist()
        
    # Get absolute maps:
    if use_cont:
        elev = (phases_cont['bottom'] - phases_cont['top']) / 2.
        azim = (phases_cont['left'] - phases_cont['right']) / 2.
        vmin, vmax = (-np.pi, np.pi)
    else:
        # Get absolute maps:
        elev = (phases['bottom'] - phases['top']) / 2.
        azim = (phases['left'] - phases['right']) / 2.
        
        # Convert to continueous:
        elev_c = -1 * elev
        elev_c = elev_c % (2*np.pi)
        azim_c = -1 * azim
        azim_c = azim_c % (2*np.pi)

        azim = copy.copy(azim_c)
        elev = copy.copy(elev_c)

        vmin, vmax = (0, 2*np.pi)
        
    if average_overlap:
        azim_phase = mask_with_overlaps_averaged(masks, azim[roi_list], mask_thr=mask_thr,
                                                is_circular=is_circular, vmin=vmin, vmax=vmax)
        elev_phase = mask_with_overlaps_averaged(masks, elev[roi_list], mask_thr=mask_thr,
                                                is_circular=is_circular, vmin=vmin, vmax=vmax)
    else:
        azim_phase = mask_rois(masks, azim[roi_list], mask_thr=mask_thr)
        elev_phase = mask_rois(masks, elev[roi_list], mask_thr=mask_thr)   
    
    return azim_phase, elev_phase

# Plotting functions 
def plot_retinomap_processing(azim_phase_soma, azim_phase_np, azim_smoothed, az_fill,
                             elev_phase_soma, elev_phase_np, elev_smoothed, el_fill, 
                             cmap='nipy_spectral', vmin=None, vmax=None, 
                             spatial_smooth_fwhm=7):
    '''
    Plot all steps, from soma and NP masks, to dilating/smoothing/etc.
    '''
    # Create masks for plotting
    # Mask images for plotting
    azim_phase_mask_np = np.ma.masked_where(azim_phase_np==-100, azim_phase_np)
    elev_phase_mask_np = np.ma.masked_where(elev_phase_np==-100, elev_phase_np)

    azim_phase_mask_soma = np.ma.masked_where(azim_phase_soma==-100, azim_phase_soma)
    elev_phase_mask_soma = np.ma.masked_where(elev_phase_soma==-100, elev_phase_soma)

    fig, axn = pl.subplots(2,4, figsize=(10,6))

    ax = axn[0,0]
    find_minmax=False
    if vmin is None:
        find_minmax=True
        vmin, vmax = (azim_phase_mask_soma.max(), azim_phase_mask_soma.min())
    ax.imshow(azim_phase_mask_soma, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title('soma')
    ax.set_ylabel('Azimuth')

    ax = axn[0, 1]
    ax.imshow(azim_phase_mask_np, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title('neuropil, center-dilated')

    ax = axn[0, 2]
    ax.imshow(azim_smoothed, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title('spatial smooth (%i)' % spatial_smooth_fwhm)

    ax = axn[0, 3]
    ax.imshow(az_fill, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title('filled NaNs')

    ax = axn[1, 0]
    if find_minmax:
        vmin, vmax = (elev_phase_mask_soma.max(), elev_phase_mask_soma.min())
    ax.imshow(elev_phase_mask_soma, cmap=cmap, vmin=vmin, vmax=vmax)
    #ax.set_title('soma')
    ax.set_ylabel('Altitude')

    ax = axn[1, 1]
    ax.imshow(elev_phase_mask_np, cmap=cmap, vmin=vmin, vmax=vmax)

    ax = axn[1, 2]
    ax.imshow(elev_smoothed, cmap=cmap, vmin=vmin, vmax=vmax)

    ax = axn[1, 3]
    ax.imshow(el_fill, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title('filled NaNs')

    pl.subplots_adjust(wspace=0.3, hspace=0.3)

    return fig

# Gradient functions
def calculate_gradients(img):
    '''
    Calculate 2d gradient, plus mean direction, etc. Return as dict.
    '''
    # Get gradient
    gdy, gdx = np.gradient(img)
    
    # 3) Calculate the magnitude
    gradmag = np.sqrt(gdx**2 + gdy**2)

    # 3) Take the absolute value of the x and y gradients
    abs_gdx = np.absolute(gdx)
    abs_gdy = np.absolute(gdy)

    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    abs_gd = np.arctan2(gdy, gdx) # np.arctan2(abs_gdy, abs_gdx) # [-pi, pi]

    # Get mean direction
    #mean_dir = np.rad2deg(np.arctan2(gdy.mean(), gdx.mean())) 
#    mean_dir = np.rad2deg(spstats.circmean([np.arctan2(gy, gx) 
#                         for gy, gx in zip(gdy.ravel(), gdx.ravel())],
#                         low=-np.pi, high=np.pi)) # TODO why this diff
    mean_dir = np.rad2deg(spstats.circmean([np.arctan2(gy, gx) 
                         for gy, gx in zip(gdy.ravel(), gdx.ravel()) \
                                 if ((not np.isnan(gy)) and (not np.isnan(gx)))],
                         low=-np.pi, high=np.pi))


    # Get unit vector
    avg_gradient = spstats.circmean(abs_gd[~np.isnan(abs_gd)], low=-np.pi, high=np.pi) 
    dirvec = (np.cos(avg_gradient), np.sin(avg_gradient))
    vhat = dirvec / np.linalg.norm(dirvec)

    grad_ = {'image': img,
             'magnitude': gradmag,
             'gradient_x': gdx,
             'gradient_y': gdy,
             'direction': abs_gd,
             'mean_deg': mean_dir, # DEG
             'mean_direction': avg_gradient, # RADIANS
             'vhat': vhat}
    
    return grad_


def plot_gradients(grad_, ax=None, draw_interval=3, 
                   scale=1, width=0.005, toy=False, headwidth=5):
    '''
    Simple sub function to plot a given gradient, using info provided in dict
    grad_: (dict)
        Output of calculate_gradients()
    scale: # of data units per arrow length unit (smaller=longer arrow)
    weight = width of plot
    
    Note: Arrows should point TOWARD larger numbers
    angles='xy' (i.e., arrows point from (x,y) to (x+u, y+v))
    '''
    if ax is None:
        fig, ax = pl.subplots()
        
    gradimg = grad_['image']
    mean_dir = grad_['mean_deg']
    gdx = grad_['gradient_x']
    gdy = grad_['gradient_y']
    
    # Set limits and number of points in grid
    y, x = np.mgrid[0:gradimg.shape[0], 0:gradimg.shape[1]]

    # Every 3rd point in each direction.
    skip = (slice(None, None, draw_interval), slice(None, None, draw_interval))
    
    # plot
    ax.quiver(x[skip], y[skip], gdx[skip], gdy[skip], color='k',
              scale=scale, width=width,
              scale_units='xy', angles='xy', pivot='mid', units='width',
              headwidth=headwidth)

    gdir_ = grad_['direction'].copy()
    gmean = spstats.circmean(gdir_[~np.isnan(gdir_)], low=-np.pi, high=np.pi)
    avg_dir_grad = np.rad2deg(gmean) #np.rad2deg(grad_['direction'].mean())
    ax.set(aspect=1, title="Mean: %.2f\n(dir: %.2f)" % (mean_dir, avg_dir_grad))

    return ax


def plot_retinomap_gradients(grad_az, grad_el, img_az=None, img_el=None,
                             spacing=200, scale=None, width=0.01, headwidth=5, cmap=None):
    '''
    Create nice retinomap + overlaid vector field showing gradients
    '''
    if img_az is None:
        img_az = grad_az['image']
    if img_el is None:
        img_el = grad_el['image']
    if cmap is None:
        cmap='nipy_spectral'
        
    fig, axn = pl.subplots(1, 2, figsize=(8,6))
    ax = axn[0]
    im = ax.imshow(img_az, cmap=cmap)#, vmin=vmin, vmax=vmax)
    plot_gradients(grad_az, ax=ax, draw_interval=spacing, scale=scale, width=width,
                headwidth=headwidth)
    fig.colorbar(im, ax=ax, shrink=0.3, label='Azimuth')

    ax = axn[1]
    im = ax.imshow(img_el,cmap=cmap) #, vmin=vmin, vmax=vmax)
    plot_gradients(grad_el, ax=ax, draw_interval=spacing, scale=scale, width=width,
                headwidth=headwidth)
    fig.colorbar(im, ax=ax, shrink=0.3, label='Altitude')
    pl.subplots_adjust(wspace=0.5, hspace=0.3)

    return fig

def plot_unit_vectors(grad_az, grad_el):
    '''
    Sanity check to make sure normalized mean gradient vector is correct
    '''
    #%% ## Compute unit vector and project
    avg_dir_el = np.rad2deg(grad_el['mean_direction'])
    print('[EL]avg dir: %.2f deg' % avg_dir_el)
    vhat_el = grad_el['vhat']

    avg_dir_az = np.rad2deg(grad_az['mean_direction'])
    print('[AZ]avg dir: %.2f deg' % avg_dir_az)
    vhat_az = grad_az['vhat']
    print(vhat_az, vhat_el)

    fig, axn = pl.subplots(1, 2, figsize=(6,3))
    ax = axn[0]
    ax.grid(True)
    ax.set_title('azimuth')
    vh = grad_az['vhat'].copy()
    az_dir = np.rad2deg(np.arctan2(vh[1], vh[0])) #+ 360.) % 360
    vhat_az = (np.cos(np.deg2rad(az_dir)), np.sin(np.deg2rad(az_dir)))

    ax.text(-.9, -.8, "u=(%.2f, %.2f), %.2f deg" % (vhat_az[0], vhat_az[1], az_dir))
    ax.quiver(0,0, vhat_az[0], vhat_az[1],  scale=1, scale_units='xy', 
            units='xy', angles='xy', width=.05, pivot='tail')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_aspect('equal')
    ax.invert_yaxis()

    ax = axn[1]
    ax.grid(True)
    ax.set_title('elevation')
    el_dir = np.rad2deg(np.arctan2(vhat_el[1], vhat_el[0]))
    ax.text(-0.9, -0.8, "u=(%.2f, %.2f), %.2f deg" % (vhat_el[0], vhat_el[1], el_dir))
    ax.quiver(0,0, vhat_el[0], vhat_el[1],  scale=1, scale_units='xy', 
            units='xy', angles='xy', width=.05, pivot='tail')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_aspect('equal')
    ax.invert_yaxis()

    return fig

# Projection related functions
def get_projection_points(grad_az, grad_el):
    '''
    Use gradient info and FOV info of image (pixel locs) to  project pixel locations
    onto the direction of the normalized mean gradient vector.
    '''
    gimg_az = grad_az['image'].copy()
    gimg_el = grad_el['image'].copy()
    d1, d2 = grad_az['image'].shape

    vhat_az = grad_az['vhat'].copy() #[0], -0.04) #abs(grad_az['vhat'][1]))
    vhat_el = grad_el['vhat'].copy() #[0], -0.04) #abs(grad_az['vhat'][1]))

    proj_az = np.array([np.dot(np.array((xv, yv)), vhat_az) for yv in np.arange(0, d1) for xv in np.arange(0, d2)])
    ret_az = np.array([gimg_az[xv, yv] for xv in np.arange(0, d1) for yv in np.arange(0, d2)] )

    proj_el = np.array([np.dot(np.array((xv, yv)), vhat_el) for yv in np.arange(0, d1) for xv in np.arange(0, d2)])
    ret_el = np.array([gimg_el[xv, yv] for xv in np.arange(0, d1) for yv in np.arange(0, d2)] )

    pix = np.array([xv for yv in np.arange(0, d1) for xv in np.arange(0, d2) ])
    #coords = np.array([np.array((xv, yv)) for yv in np.arange(0, d1) for xv in np.arange(0, d2)])
    
    projections = {'proj_az': proj_az,
                   'proj_el': proj_el,
                   'retino_az': ret_az,
                   'retino_el': ret_el,
                   'pixel_ixs': pix}
    
    return projections 

def test_plot_projections(projections, ncyc=10, startcyc=50, imshape=(512,512)):
    '''
    Plot a subset of "rows" of the image and make sure offsets match.
    '''
    proj_az = projections['proj_az']
    proj_el = projections['proj_el']
    pix = projections['pixel_ixs']
    ret_az = projections['retino_az']
    ret_el = projections['retino_el']

    d1, d2 = imshape
    #ncyc=10
    #startcyc=50
    endcyc=startcyc+ncyc
    npts = d2*endcyc
    spts = d2*startcyc

    fig, axn = pl.subplots(2,2, figsize=(8,8)) #.figure()
    fig.suptitle("%i thru %i (of %i) cycles" % (startcyc, endcyc, d1))

    ax = axn[0,0]
    ax.plot(pix[spts:npts], 'k',  marker='.', lw=.5, markersize=0.5, alpha=0.5)
    ax.plot(proj_az[spts:npts], 'r',  marker='.', lw=0.5, markersize=0.5)
    ax.plot(ret_az[spts:npts], 'b', marker='.', lw=0, markersize=0.5)

    ax.set_ylabel('value along d2')
    ax = axn[0,1]
    offsets = [0 for i in np.arange(spts, npts)]
    ax.scatter(proj_az[spts:npts], ret_az[spts:npts]+offsets, marker='.', s=2, c='b')
    ax.set_xlabel('projected')
    ax.set_ylabel('retino')

    ax = axn[1,0]
    ax.plot(pix[spts:npts], 'k',  marker='.', lw=.5, markersize=0.5, alpha=0.5)
    ax.plot(proj_el[spts:npts], 'r',  marker='.', lw=0.5, markersize=0.5)
    ax.plot(ret_el[spts:npts], 'b', marker='.', lw=0, markersize=0.5)

    ax.set_ylabel('value along d2')
    ax = axn[1,1]
    offsets = [0 for i in np.arange(spts, npts)]
    ax.scatter(proj_el[spts:npts], ret_el[spts:npts]+offsets, marker='.', s=2, c='b')
    ax.set_xlabel('projected')
    ax.set_ylabel('retino')

    return fig

def plot_projected_vs_retino_positions(projections, fit_results, 
                                       spacing=200, regr_color='magenta'):
    '''
    Visualize linear fit beween projected pixel positions and retinotopic position.
    '''
    fitv_az = fit_results['fitv_az']
    fitv_el = fit_results['fitv_el']
    
    proj_az = projections['proj_az']
    proj_el = projections['proj_el']
    ret_az = projections['retino_az']
    ret_el = projections['retino_el']

    regr_az = fit_results['regr_az']
    regr_el = fit_results['regr_el']
    
    regr_model = fit_results['model']

    fig, axn = pl.subplots(1,2, figsize=(12, 5))
    ax=axn[0]
    ax.scatter(proj_az[0::spacing], ret_az[0::spacing], marker='.', lw=0, color='k', s=1)
    r2_v = skmetrics.r2_score(ret_az[~np.isnan(ret_az)], fitv_az)
    lfit_str = '(%s) y=%.2fx+%.2f, R2=%.2f' % (regr_model, float(regr_az.coef_), float(regr_az.intercept_), r2_v) 
    ax.plot(proj_az[~np.isnan(ret_az)], fitv_az, c=regr_color, label=lfit_str)
    ax.set_title('Azimuth', loc='left') #linfit_str, color=regr_color)
    ax.set_ylabel('azimuth (deg)', fontsize=12)
    ax.set_xlabel('projected pos (um)', fontsize=12)
    ax.legend()

    ax=axn[1]
    #sns.regplot(proj_el, ret_el, ax=ax, scatter=False, color='k')
    ax.scatter(proj_el[0::spacing], ret_el[0::spacing], marker='.', lw=0, color='k', s=1)
    r2_v = skmetrics.r2_score(ret_el[~np.isnan(ret_el)], fitv_el)
    lfit_str = '(%s) y=%.2fx+%.2f, R2=%.2f' % (regr_model, float(regr_el.coef_), float(regr_el.intercept_), r2_v) 
    ax.plot(proj_el[~np.isnan(ret_el)], fitv_el, c=regr_color, label=lfit_str)
    ax.set_title('Elevation', loc='left') #linfit_str, color=regr_color)
    ax.set_ylabel('altitude (deg)', fontsize=12)
    ax.set_xlabel('projected pos (um)', fontsize=12)
    ax.legend()

    return fig






#%% Variables
rootdir = '/n/coxfs01/2p-data'
aggr_dir = '/n/coxfs01/julianarhee/aggregate-visual-areas'
traceid = 'traces001'

animalid = 'JC084' #'JC085'
session = '20190522' #'20190626'
fov = 'FOV1_zoom2p0x'
retinorun = 'retino_run1'

mag_thr=0.01 #if trace_type == 'neuropil' else 0.02
pass_criterion='all' #all_conds_pass = True
plot_examples = True

# plotting
cmap_name = 'nic_Edge'
zero_center = True
spatial_smooth_fwhm = 7 #21

desired_radius_um = 10.0 #20.0
regr_plot_spacing=200
regr_line_color='magenta'



def extract_options(options):
    
    parser = optparse.OptionParser()

    parser.add_option('-D', '--root', action='store', dest='rootdir', 
                      default='/n/coxfs01/2p-data',\
                      help='data root dir [default: /n/coxfs01/2pdata]')
    parser.add_option('-i', '--animalid', action='store', dest='animalid', 
                        default='', help='Animal ID')

    # Set specific session/run for current animal:
    parser.add_option('-S', '--session', action='store', dest='session', default='', \
                      help='session dir (format: YYYMMDD_ANIMALID')
    parser.add_option('-A', '--fov', action='store', dest='fov', default='FOV1_zoom2p0x', \
                      help="acquisition folder (ex: 'FOV1_zoom3x') [default: FOV1_zoom2p0x]")
    parser.add_option('-R', '--run', action='store', dest='run', default='retino_run1', \
                      help="name of run (default: retino_run1")
    parser.add_option('-t', '--traceid', action='store', dest='traceid', default='traces001', \
                      help="name of traces ID [default: traces001]")
       
    parser.add_option('--new', action='store_true', dest='create_new', default=False, \
                      help="Flag to refit all rois")

    # data filtering 
    parser.add_option('--thr', action='store', dest='mag_thr', 
            default=0.01, help="magnitude-ratio thr (default: 0.01)")
    parser.add_option('-p', '--crit', action='store', dest='pass_criterion', 
            default='all', 
            help="Criterion for passing cells as responsive (default: 'all', can by 'any' or None)")
    parser.add_option('--plot-examples', action='store_true', dest='plot_examples', 
            default=False, help="Flag to plot top 3 examples cell traces")

    # plotting
    parser.add_option('--cmap', action='store_true', dest='cmap', 
            default='nic_Edge', help="Colormap (default: nic_Edge)")
    parser.add_option('--plot-spacing', action='store', dest='regr_plot_spacing', 
            default=200, help="Plot every N points for regression (default: 200)")
    parser.add_option('-c', '--plot-color', action='store', dest='regr_line_color', 
            default='magenta', help="Plot color for regression line (default: magenta)")

    
    parser.add_option('-s', '--spatial', action='store', dest='spatial_smooth_fwhm', 
            default=7.0, help="FWHM for spatial smoothing (default: 7)")
    parser.add_option('-d', '--dilate', action='store', dest='dilate_um', 
            default=10.0, help="Desired radius for dilation (default: 10.0 um)")
    parser.add_option('-M', '--model', action='store', dest='regr_model', 
            default='ridge', help="Desired radius for dilation (default: ridge)")

    
    (options, args) = parser.parse_args(options)

    return options



def main(options):

    opts = extract_options(options)
    rootdir=opts.rootdir
    animalid=opts.animalid
    session=opts.session
    fov=opts.fov
    retinorun=opts.run
    traceid=opts.traceid
    mag_thr=float(opts.mag_thr)
    pass_criterion=opts.pass_criterion    

    plot_examples=opts.plot_examples
    cmap_name=opts.cmap

    spatial_smooth_fwhm=opts.spatial_smooth_fwhm
    desired_radius_um=float(opts.dilate_um)
    regr_plot_spacing=int(opts.regr_plot_spacing)
    regr_line_color=opts.regr_line_color 
    zero_center=True
   
    regr_model = opts.regr_model

    #%% Load data metainfo
    run_dir = os.path.join(rootdir, animalid, session, fov, retinorun)
    RETID = ret_utils.load_retinoanalysis(run_dir, traceid)
    analysis_dir = RETID['DST']
    retinoid = RETID['analysis_id']
    print("--- Loaded: %s, %s (%s))" % (retinorun, retinoid, run_dir))
    data_id = '_'.join([animalid, session, fov, retinorun, retinoid])
    print("--- Data ID: %s" % data_id)

    # Load MW info and SI info
    mwinfo = ret_utils.load_mw_info(animalid, session, fov, retinorun)
    scaninfo = ret_utils.get_protocol_info(animalid, session, fov, run=retinorun)
    trials_by_cond = scaninfo['trials']

    # Set current animal's retino output dir
    curr_dst_dir = os.path.join(analysis_dir, 'retino-structure')
    if not os.path.exists(curr_dst_dir):
        os.makedirs(curr_dst_dir)
    print("--- Saving output to:\n %s" % curr_dst_dir)

    # Move old stuff
    old_dir = os.path.join(curr_dst_dir, 'tests')
    if not os.path.exists(old_dir):
        os.makedirs(old_dir)
    oldimgs = [i for i in os.listdir(curr_dst_dir) if i.endswith('.svg') or i.endswith('.png')]
    for i in oldimgs:
        shutil.move(os.path.join(curr_dst_dir, i), os.path.join(old_dir, i))

    # Load colormap
    screen, cmap_phase = ret_utils.get_retino_legends(cmap_name=cmap_name, 
                                                      zero_center=zero_center,
                                                      return_cmap=True, dst_dir=curr_dst_dir)  


    #%% Process traces
    # Load raw and process traces -- returns average trace for condition
    np_traces = ret_utils.load_traces(animalid, session, fov, run=retinorun,
                                      analysisid=retinoid, trace_type='neuropil')
    soma_traces = ret_utils.load_traces(animalid, session, fov, run=retinorun,
                                      analysisid=retinoid, trace_type='raw')

    #%% Do FFT
    # Get params
    n_frames = scaninfo['stimulus']['n_frames']
    frame_rate = scaninfo['stimulus']['frame_rate']
    stim_freq_idx = scaninfo['stimulus']['stim_freq_idx']
    freqs = np.fft.fftfreq(n_frames, float(1./frame_rate)) # Label frequency bins
    sorted_freq_idxs = np.argsort(freqs)

    # Do FFT
    fft_soma = dict((cond, ret_utils.do_fft_analysis(tdf, sorted_freq_idxs, stim_freq_idx)) 
                        for cond, tdf in soma_traces.items())
    fft_np = dict((cond, ret_utils.do_fft_analysis(tdf, sorted_freq_idxs, stim_freq_idx)) 
                        for cond, tdf in np_traces.items())

    # Get magratios -- each column is a condition
    magratios_soma = pd.DataFrame(dict((cond, k[0]) for cond, k in fft_soma.items()))
    magratios_np = pd.DataFrame(dict((cond, k[0]) for cond, k in fft_np.items()))

    # Get phases
    phases_soma = pd.DataFrame(dict((cond, k[1]) for cond, k in fft_soma.items()))
    phases_np = pd.DataFrame(dict((cond, k[1]) for cond, k in fft_np.items()))

    # Get average across conditions
    mean_magratio_values_soma = magratios_soma.mean(axis=1).values 
    mean_magratio_values_np = magratios_np.mean(axis=1).values

    # Sort ROIs by their average mag ratios
    sorted_rois_soma = np.argsort(mean_magratio_values_soma)[::-1]

    # Filter out bad cells
    conds = [c for c in magratios_soma.columns if c!='blank']
    if pass_criterion=='all': 
        roi_list = [i for i in magratios_soma.index if all(magratios_soma[conds].loc[i] > mag_thr)]
    elif pass_criterion=='any': 
        roi_list = [i for i in magratios_soma.index if any(magratios_soma[conds].loc[i] > mag_thr)]
    else:
        roi_list = magratios_soma.index.tolist()
    print("... %i out of %i cells pass mag-ratio thr (thr>%.2f)" 
                % (len(roi_list), len(mean_magratio_values_soma), mag_thr))
    sorted_by_mag = [r for r in sorted_rois_soma if r in roi_list]

    # Look at example cell
    if plot_examples:
        ret_utils.plot_some_example_traces(soma_traces, np_traces, 
                                            plot_rois=sorted_rois_soma[0:3],
                                            dst_dir=curr_dst_dir, data_id=data_id)
      
  
    #%% Get mask info
    masks_soma, masks_np, zimg = ret_utils.load_soma_and_np_masks(RETID)
    roiid = RETID['PARAMS']['roi_id']
    ds_factor = int(RETID['PARAMS']['downsample_factor'])
    nrois_total, d1, d2 = masks_soma.shape
    print("... got masks: %s (downsample=%.2f)" % (str(masks_soma.shape), ds_factor))

    # In[61]:
    fig, ax = pl.subplots()
    roi_utils.plot_neuropil_masks(masks_soma, masks_np, zimg, ax=ax)
    ax.set_title('soma + neuropil masks')
    label_figure(fig, data_id)
    pl.savefig(os.path.join(curr_dst_dir, 'soma-v-neuropil-masks.png'))
    pl.close()

    #%% # Dilate soma masks
    # From Liang et al., 2018, Cell.
    # 1. Assign center of neuropil ring w/ preferred retino location.
    # 2. From Neurpil ring center, dilate by a disk of 10um radius (20um diam)
    # 3. Average overlapping disks
    # 4. Spatially smooth w/ isotropic 2D Gaus filter (std=2um) for final pixe-wise estimates 

    # measured pixel size: (2.3, 1.9)
    # want to dilate by ~9.52380952381

    pixel_size = putils.get_pixel_size()
    um_per_pixel = np.mean(pixel_size) / ds_factor # divide by DS (pixels are half if ds=2)
    pixels2dilate = desired_radius_um/um_per_pixel

    # Set kernel params
    kernel_size = np.ceil(pixels2dilate+0) #2) #21
    kernel_radius = (kernel_size - 1) // 2
    x, y = np.ogrid[-kernel_radius:kernel_radius+1, -kernel_radius:kernel_radius+1]
    dist = (x**2 + y**2)**0.5 # shape (kernel_size, kernel_size)

    # Create three kernels for the sake of example
    radii = np.array([kernel_size/3., kernel_size/2.5, kernel_size/2.])[...,None,None] # shape (num_radii, 1, 1)
    # shape (num_radii, kernel_size, kernel_size)
    kernel = (1 - (dist - radii).clip(0,1)).sum(axis=0)
    kernel_diam_um = (kernel.shape[0]*um_per_pixel)
    print("... dilation diameter (kernel, %s): %.2fum" % (str(kernel.shape), kernel_diam_um))

    # Dilate all masks with kernel
    dilated_masks = dilate_mask_centers(masks_soma.astype(float), kernel_size=kernel_size)

    #%% Assign phase to neuropil
    use_cont = True
    average_overlap = True
    filter_by_mag = True
    mask_thr=0.01

    # Set bounds for averaging
    vmin = -np.pi if use_cont else 0
    vmax = np.pi if use_cont else 2*np.pi

    # Assign phase value to dilated masks
    azim_phase_np, elev_phase_np = get_phase_masks(dilated_masks, phases_np, 
                                                   average_overlap=average_overlap, 
                                                   roi_list=roi_list, #None, 
                                                   use_cont=use_cont, mask_thr=mask_thr)

    azim_phase_soma, elev_phase_soma = get_phase_masks(masks_soma, phases_soma, 
                                                    average_overlap=average_overlap, 
                                                    roi_list=roi_list, 
                                                    use_cont=use_cont, mask_thr=mask_thr)

    #%% Resize image 
    pixel_size = putils.get_pixel_size()
    pixel_size = (pixel_size[0]*ds_factor, pixel_size[1]*ds_factor)
    zimg_r = coreg.transform_2p_fov(zimg, pixel_size)
    print("... pixel size: %s (ds_factor=%.2f)" % (str(pixel_size), ds_factor))

    #%% Spatial smooth neuropil dilated masks 
    azim_smoothed = ret_utils.smooth_neuropil(azim_phase_np, smooth_fwhm=spatial_smooth_fwhm)
    elev_smoothed = ret_utils.smooth_neuropil(elev_phase_np, smooth_fwhm=spatial_smooth_fwhm)

    if 'zoom1p0x' in fov:
        print("... resizing")
        print("ABORT.")
        #azim_smoothed = cv2.resize(azim_smoothed, (new_d1, new_d2))
        #elev_smoothed = cv2.resize(elev_smoothed, (new_d1, new_d2))    
    azim_smoothed = fill_and_smooth_nans(azim_smoothed) # This should NOT resize img
    elev_smoothed = fill_and_smooth_nans(elev_smoothed)

    #%% Transform FOV to match widefield
    azim_r = coreg.transform_2p_fov(azim_smoothed, pixel_size, normalize=False)
    elev_r = coreg.transform_2p_fov(elev_smoothed, pixel_size, normalize=False)
    #print(azim_r[~np.isnan(azim_r)].min(), azim_r[~np.isnan(azim_r)].max())

    az_fill = azim_r.copy()
    el_fill = elev_r.copy()
    #print(az_fill.shape)

    # In[43]:
    fig = plot_retinomap_processing(azim_phase_soma, azim_phase_np, azim_smoothed, az_fill,
                                   elev_phase_soma, elev_phase_np, elev_smoothed, el_fill, 
                                   cmap=cmap_phase, vmin=vmin, vmax=vmax, 
                                   spatial_smooth_fwhm=spatial_smooth_fwhm)
    putils.label_figure(fig, data_id)
    figname = 'soma_neuropil_dilate-%i_smooth-%i_magthr-%.2f-%s' % (kernel_size, spatial_smooth_fwhm, mag_thr, pass_criterion)
    pl.savefig(os.path.join(curr_dst_dir, '%s.png' % figname))

    #%% ## Calculate gradient on retino map
    # Convert to degrees
    plot_degrees = True
    screen = putils.get_screen_dims()
    screen_max = screen['azimuth_deg']/2.
    screen_min = -screen_max

    img_az = convert_range(az_fill, oldmin=vmin, oldmax=vmax, 
                            newmin=screen_min, newmax=screen_max) if plot_degrees else az_fill.copy()
    img_el = convert_range(el_fill, oldmin=vmin, oldmax=vmax,
                            newmin=screen_min, newmax=screen_max) if plot_degrees else el_fill.copy()
    grad_az = calculate_gradients(img_az)
    grad_el = calculate_gradients(img_el)
    vmin, vmax = (screen_min, screen_max) if plot_degrees else (-np.pi, np.pi)
    #print(vmin, vmax)

    #%% Plot gradients 
    spacing = 200
    scale = None #0.0001
    width = 0.01 #0.01
    headwidth=5

    plot_str = 'degrees' if plot_degrees else ''
    fig = plot_retinomap_gradients(grad_az, grad_el, cmap=cmap_phase)
    putils.label_figure(fig, data_id)
    figname = 'gradients_dilate-%i_smooth-%i_%s_circ_magthr-%.2f-%s' % (kernel_size, spatial_smooth_fwhm, plot_str, mag_thr, pass_criterion)
    pl.savefig(os.path.join(curr_dst_dir, '%s.svg' % figname))
    print('-- [f] %s' % figname)

    fig = plot_unit_vectors(grad_az, grad_el)
    label_figure(fig, data_id)
    pl.subplots_adjust(left=0.1, wspace=0.5)
    figname = 'unitvec_dilate-%i_smooth-%i_%s_circ_magthr-%.2f-%s' % (kernel_size, spatial_smooth_fwhm, plot_str, mag_thr, pass_criterion)
    pl.savefig(os.path.join(curr_dst_dir, '%s.svg' % figname))
    print('-- [f] %s' % figname)

    # Save gradients
    gradients = {'az': grad_az, 'el': grad_el}
    grad_fpath = os.path.join(curr_dst_dir, 'gradients.pkl')
    with open(grad_fpath, 'wb') as f:
        pkl.dump(gradients, f, protocol=pkl.HIGHEST_PROTOCOL)

    #%% ## Calculate gradients and projec to get mean
    projections = get_projection_points(grad_az, grad_el)

    d1, d2 = grad_az['image'].shape
    fig = test_plot_projections(projections, ncyc=10, startcyc=50, imshape=(d1,d2))
    label_figure(fig, data_id)
    pl.subplots_adjust(left=0.1, wspace=0.5)
    figname = 'test_projections__dilate-center-%i_spatial-smooth-%i_%s_circ_magthr-%.2f-%s' % (kernel_size, spatial_smooth_fwhm, plot_str, mag_thr, pass_criterion)
    pl.savefig(os.path.join(curr_dst_dir, '%s.svg' % figname))

    #%% ## Fit linear  
    proj_fit_results = {}
    d_list = []
    di = 0
    for i, cond in enumerate(['az', 'el']):
        proj_v = projections['proj_%s' % cond].copy()
        ret_v = projections['retino_%s' % cond].copy()
        #xv = xv[~np.isnan(yv)]
        #yv = yv[~np.isnan(yv)]
        fitv, regr = evalrf.fit_linear_regr(proj_v[~np.isnan(ret_v)], 
                                            ret_v[~np.isnan(ret_v)],
                                            return_regr=True, model=regr_model)
     
        rmse = np.sqrt(skmetrics.mean_squared_error(ret_v[~np.isnan(ret_v)], fitv))
        r2 = skmetrics.r2_score(ret_v[~np.isnan(ret_v)], fitv)
        pearson_r, pearson_p = spstats.pearsonr(proj_v[~np.isnan(ret_v)], ret_v[~np.isnan(ret_v)]) 
        slope = float(regr.coef_)
        intercept = float(regr.intercept_)

        proj_fit_results.update({'fitv_%s' % cond: fitv, 
                                 'regr_%s' % cond: regr})

        d_ = pd.DataFrame({'cond': cond, 
                          'R2': r2,
                          'RMSE': rmse,
                          'pearson_p': pearson_p,
                          'pearson_r': pearson_r,
                          'coefficient': slope, # float(regr.coef_), 
                          'intercept': intercept, #float(regr.intercept_)
                          }, index=[di])
        print("~~~regr results: y = %.2f + %.2f (R2=%.2f)" % (slope, intercept, r2))

        d_list.append(d_)
        di += 1

    regr_df = pd.concat(d_list, axis=0)
    proj_fit_results.update({'projections': projections, 
                             'model': regr_model,
                             'regr_df': regr_df})

    proj_fpath = os.path.join(curr_dst_dir, 'projection_results.pkl')
    with open(proj_fpath, 'wb') as f:
        pkl.dump(proj_fit_results, f, protocol=pkl.HIGHEST_PROTOCOL)
    print(proj_fpath)
        
    #%% Plot linear fit
    fig = plot_projected_vs_retino_positions(projections, proj_fit_results,
                                             spacing=regr_plot_spacing, 
                                             regr_color=regr_line_color)

    label_figure(fig, data_id)
    pl.subplots_adjust(left=0.1, wspace=0.5)
    figname = 'Proj_versus_Retinopos__dilate-%i_smooth-%i_%s_circ_magthr-%.2f-%s' % (kernel_size, spatial_smooth_fwhm, plot_str, mag_thr, pass_criterion)
    pl.savefig(os.path.join(curr_dst_dir, '%s.svg' % figname))


if __name__ == '__main__':
    main(sys.argv[1:])



