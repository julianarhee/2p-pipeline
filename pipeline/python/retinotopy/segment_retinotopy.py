#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 12:06:11 2019

@author: julianarhee
"""


import os
import glob
import json

import cPickle as pkl
import numpy as np
import scipy as sp
import pylab as pl

from mpl_toolkits.axes_grid1 import make_axes_locatable

from pipeline.python.retinotopy import utils as rutils


#%%


def filter_map_by_magratio(magratio, phase, cond='right', mag_thr=None, mag_perc=0.05):
    if mag_thr is None:
        mag_thr = magratio.max().max()*mag_perc
        
    currmags = magratio[trials_by_cond[cond]]
    currmags[currmags<mag_thr] = np.nan
    currmags_mean = np.nanmean(currmags, axis=1)
    d1 = int(np.sqrt(currmags_mean.shape[0]))
    currmags_map = np.reshape(currmags_mean, (d1, d1))
    

    currphase = phase[trials_by_cond[cond]]
    currphase_mean = sp.stats.circmean(currphase, low=-np.pi, high=np.pi, axis=1)
    currphase_mean_c = rutils.correct_phase_wrap(currphase_mean)
    
    currphase_mean_c[np.isnan(currmags_mean)] = np.nan
    currphase_map_c = np.reshape(currphase_mean_c, (d1, d1))
    
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
animalid = 'JC076' #'JC059'
session = '20190410' #'20190227'
fov = 'FOV1_zoom2p0x' #'FOV4_zoom4p0x'
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

fit, magratio, phase, trials_by_cond = rutils.trials_to_dataframes(processed_fpaths, conditions_fpath)


#%%

mag_thr = 0.002

magmaps = {}
phasemaps = {}
magthrs = {}
for cond in trials_by_cond.keys():    
    magmaps[cond], phasemaps[cond], magthrs[cond] = filter_map_by_magratio(magratio, phase, cond=cond, mag_thr=mag_thr)
    fig = plot_filtered_maps(cond, magmaps[cond], phasemaps[cond], magthrs[cond])


absolute_az = (phasemaps['left'] - phasemaps['right']) / 2.
delay_az = (phasemaps['left'] + phasemaps['right']) / 2.


absolute_el = (phasemaps['bottom'] - phasemaps['top']) / 2.
delay_el = (phasemaps['bottom'] + phasemaps['top']) / 2.


#%%
fig, axes = pl.subplots(2,2)
im1 = axes[0,0].imshow(absolute_az, cmap='nipy_spectral', vmin=-np.pi, vmax=np.pi)
im2 = axes[0,1].imshow(absolute_el, cmap='nipy_spectral', vmin=-np.pi, vmax=np.pi)
axes[1,0].imshow(delay_az, cmap='nipy_spectral', vmin=-np.pi, vmax=np.pi)
axes[1,1].imshow(delay_el, cmap='nipy_spectral', vmin=-np.pi, vmax=np.pi)

cbar1_orientation='horizontal'
cbar1_axes = [0.35, 0.9, 0.1, 0.05]
cbar2_orientation='vertical'
cbar2_axes = [0.75, 0.9, 0.1, 0.05]

cbaxes = fig.add_axes(cbar1_axes) 
cb = pl.colorbar(im1, cax = cbaxes, orientation=cbar1_orientation)  
cbaxes = fig.add_axes(cbar2_axes) 
cb = pl.colorbar(im2, cax = cbaxes, orientation=cbar2_orientation)  














