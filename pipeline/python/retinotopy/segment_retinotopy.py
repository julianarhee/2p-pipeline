#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 12:06:11 2019

@author: julianarhee
"""


import os
import glob
import json
import cv2

import cPickle as pkl
import numpy as np
import pylab as pl
import tifffile as tf

from scipy import stats
from skimage.measure import block_reduce


from mpl_toolkits.axes_grid1 import make_axes_locatable

from pipeline.python.retinotopy import utils as rutils
from pipeline.python.utils import natural_keys, label_figure, get_screen_dims


#%%


def filter_map_by_magratio(magratio, phase, dims=(512, 512), ds_factor=2, cond='right', mag_thr=None, mag_perc=0.05):
    if mag_thr is None:
        mag_thr = magratio.max().max()*mag_perc
        
    currmags = magratio[trials_by_cond[cond]]
    currmags[currmags<mag_thr] = np.nan
    currmags_mean = np.nanmean(currmags, axis=1)
    #d1 = int(np.sqrt(currmags_mean.shape[0]))
    d1 = dims[0] / ds_factor
    d2 = dims[1] / ds_factor
    currmags_map = np.reshape(currmags_mean, (d1, d2))
    

    currphase = phase[trials_by_cond[cond]]
    currphase_mean = stats.circmean(currphase, low=-np.pi, high=np.pi, axis=1)
    currphase_mean_c = rutils.correct_phase_wrap(currphase_mean)
    
    currphase_mean_c[np.isnan(currmags_mean)] = np.nan
    currphase_map_c = np.reshape(currphase_mean_c, (d1, d2))
    
    return currmags_map, currphase_map_c, mag_thr


def plot_filtered_maps(cond, currmags_map, currphase_map_c, mag_thr):
    fig, axes = pl.subplots(1, 2) #pl.figure()
    im = axes[0].imshow(currmags_map)
    divider = make_axes_locatable(axes[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    
    im2 = axes[1].imshow(currphase_map_c, cmap='nipy_spectral', vmin=0, vmax=2*np.pi)
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im2, cax=cax, orientation='vertical')
    
    pl.subplots_adjust(wspace=0.5)
    fig.suptitle('%s (mag_thr: %.4f)' % (cond, mag_thr))

    return fig


#%%

rootdir = '/n/coxfs01/2p-data'
animalid = 'JC090' # 'JC076' #'JC091' #'JC059'
session = '20190604' #'20190420' #20190623' #'20190227'
fov = 'FOV3_zoom2p0x' #'FOV1_zoom2p0x' #'FOV4_zoom4p0x'
run = 'retino_run1'
traceid = 'analysis001' #'traces001'
visual_area = ''

if 'retino' in run:
    traces_subdir = 'retino_analysis'
    is_retino = True
else:
    traces_subdir = 'traces'
    is_retino = False
    
run_dir = os.path.join(rootdir, animalid, session, fov, run)
#traceid_dir = glob.glob(os.path.join(fov_dir, run, traces_subdir, '%s*' % traceid))[0]


retinoid, RID = rutils.load_retino_analysis_info(animalid, session, fov, run, traceid, use_pixels=True, rootdir=rootdir)

data_identifier = '|'.join([animalid, session, fov, run, retinoid, visual_area])
print("*** Dataset: %s ***" % data_identifier)



#%%
    
# Get processed retino data:
processed_dir = glob.glob(os.path.join(run_dir, 'retino_analysis', '%s*' % retinoid))[0]
processed_fpaths = glob.glob(os.path.join(processed_dir, 'files', '*.h5'))
print("Found %i processed retino runs." % len(processed_fpaths))


# Get condition info for trials:
conditions_fpath = glob.glob(os.path.join(run_dir, 'paradigm', 'files', 'parsed_trials*.json'))[0]
with open(conditions_fpath, 'r') as f:
    mwinfo = json.load(f)

# Get run info:
runinfo_fpath = glob.glob(os.path.join(run_dir, '*.json'))[0]
with open(runinfo_fpath, 'r') as f:
    runinfo = json.load(f)
print "---------------------------------"
#print "Trials by condN:", trials_by_cond

# Get stimulus info:
stiminfo, trials_by_cond = rutils.get_retino_stimulus_info(mwinfo, runinfo)
#stiminfo['trials_by_cond'] = trials_by_cond
print "Trials by condN:", trials_by_cond


#%%


# adjust elevation limit to show only monitor extent
screen = get_screen_dims() 
screen_left = -1*screen['azimuth_deg']/2.
screen_right = screen['azimuth_deg']/2.
screen_top = screen['altitude_deg']/2.
screen_bottom = -1*screen['altitude_deg']/2.
    
elev_cutoff = screen_top / screen_right




#%%


fit, magratio, phase, trials_by_cond = rutils.trials_to_dataframes(processed_fpaths, conditions_fpath)


#%%

mag_thr = 0.0025
d2 = runinfo['pixels_per_line']
d1 = runinfo['lines_per_frame']

magmaps = {}
phasemaps = {}
magthrs = {}
for cond in trials_by_cond.keys():    
    magmaps[cond], phasemaps[cond], magthrs[cond] = filter_map_by_magratio(magratio, phase, cond=cond, mag_thr=mag_thr, dims=(d1, d2))
    fig = plot_filtered_maps(cond, magmaps[cond], phasemaps[cond], magthrs[cond])


absolute_az = (phasemaps['left'] - phasemaps['right']) / 2.
delay_az = (phasemaps['left'] + phasemaps['right']) / 2.


absolute_el = (phasemaps['bottom'] - phasemaps['top']) / 2.
delay_el = (phasemaps['bottom'] + phasemaps['top']) / 2.

#%%
fig, axes = pl.subplots(2,2)
im1 = axes[0,0].imshow(absolute_az, cmap='nipy_spectral_r', vmin=-np.pi, vmax=np.pi)
im2 = axes[0,1].imshow(absolute_el, cmap='nipy_spectral_r', vmin=-np.pi, vmax=np.pi)
axes[1,0].imshow(delay_az, cmap='nipy_spectral', vmin=-np.pi, vmax=np.pi)
axes[1,1].imshow(delay_el, cmap='nipy_spectral', vmin=-np.pi, vmax=np.pi)

cbar1_orientation='horizontal'
cbar1_axes = [0.35, 0.9, 0.1, 0.05]
cbar2_orientation='vertical'
cbar2_axes = [0.75, 0.9, 0.1, 0.05]

cbaxes = fig.add_axes(cbar1_axes) 
cb = pl.colorbar(im1, cax = cbaxes, orientation=cbar1_orientation)  
cb.ax.axis('off')
cb.outline.set_visible(False)

cbaxes = fig.add_axes(cbar2_axes) 
cb = pl.colorbar(im2, cax = cbaxes, orientation=cbar2_orientation)
#cb.ax.set_ylim([cb.norm(-np.pi*top_cutoff), cb.norm(np.pi*top_cutoff)])
cb.ax.axhline(y=cb.norm(-np.pi*elev_cutoff), color='w', lw=1)
cb.ax.axhline(y=cb.norm(np.pi*elev_cutoff), color='w', lw=1)
cb.ax.axis('off')
cb.outline.set_visible(False)

pl.subplots_adjust(top=0.8)

for ax in axes.flat:
    ax.axis('off')
    
label_figure(fig, data_identifier)
figname = 'absolute_and_delay_maps_magthr_%.3f' % mag_thr
pl.savefig(os.path.join(processed_dir, 'figures', '%s.png' % figname))




#%%
# Load surface image to plot overlay:
#surface_fpath = glob.glob(os.path.join(rootdir, animalid, 'macro_maps', '*', '*urf*'))[0]
#surface_img = cv2.imread(surface_fpath, -1)
#print(surface_img.shape)

overlay_surface = False
if overlay_surface:
    ch_num = 2
    fov_imgs = glob.glob(os.path.join(rootdir, animalid, session, fov, 'anatomical', 'processed',\
                                   'processed*', 'mcorrected_*mean_deinterleaved', 'Channel%02d' % ch_num, 'File*', '*.tif'))
else:
    ch_num = 1
    fov_imgs = glob.glob(os.path.join(run_dir, 'processed', 'processed*', 'mcorrected_*mean_deinterleaved',\
                                      'Channel%02d' % ch_num, 'File*', '*.tif'))
    
    
imlist = []
for anat in fov_imgs:
    im = tf.imread(anat)
    imlist.append(im)
surface_img = np.array(imlist).mean(axis=0)


pl.figure()
pl.imshow(surface_img, cmap='gray')

if surface_img.shape[0] != absolute_az.shape[0]:
    reduce_factor = surface_img.shape[0] / absolute_az.shape[0]
    surface_img = block_reduce(surface_img, (2,2), func=np.mean)
    
#%%

import matplotlib as mpl
import matplotlib.cm as cmx


vmin = -np.pi
vmax = np.pi

fig, axes = pl.subplots(1,2)
ax = axes[0]
ax.imshow(surface_img, cmap='gray', origin='upper')


ax = axes[0]
im1 = ax.imshow(absolute_az, cmap='nipy_spectral', vmin=-np.pi, vmax=np.pi, alpha=0.7, origin='upper')

ax = axes[1]
ax.imshow(surface_img, cmap='gray')
im2 = ax.imshow(absolute_el, cmap='nipy_spectral', vmin=-np.pi, vmax=np.pi, alpha=0.7, origin='upper')

cbar1_orientation='horizontal'
cbar1_axes = [0.37, 0.65, 0.1, 0.07]
cbar2_orientation='vertical'
cbar2_axes = [0.8, 0.65, 0.1, 0.07]


cnorm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
scalarmap = cmx.ScalarMappable(norm=cnorm, cmap='nipy_spectral')
print("scaled cmap lim:", scalarmap.get_clim())
bounds = np.linspace(vmin, vmax)
scalarmap.set_array(bounds)

cbar1_ax = fig.add_axes(cbar1_axes)
cbar1 = fig.colorbar(im1, cax=cbar1_ax, orientation=cbar1_orientation)
cbar1.ax.axis('off')

cbar2_ax = fig.add_axes(cbar2_axes)
cbar2 = fig.colorbar(im2, cax=cbar2_ax, orientation=cbar2_orientation)
cbar2.ax.axhline(y=cbar2.norm(-np.pi*elev_cutoff), color='w', lw=1)
cbar2.ax.axhline(y=cbar2.norm(np.pi*elev_cutoff), color='w', lw=1)
cbar2.ax.axis('off')

cbar1.outline.set_visible(False)
cbar2.outline.set_visible(False)

#pl.subplots_adjust(top=0.8)

for ax in axes.flat:
    ax.axis('off')

label_figure(fig, data_identifier)
figname = 'absolute_maps_magthr_%.3f' % mag_thr
pl.savefig(os.path.join(processed_dir, 'figures', '%s.png' % figname))



