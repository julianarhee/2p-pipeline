#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 18:36:46 2019

@author: julianarhee
"""


# coding: utf-8

# In[107]:


import os
import glob
import json
import h5py
import optparse
import sys
import traceback
import copy

import pandas as pd
import pylab as pl
import seaborn as sns
import numpy as np
import scipy as sp
import statsmodels as sm
import cPickle as pkl
import tifffile as tf

import numpy as np
from scipy.optimize import leastsq
import pylab as plt


from pipeline.python.utils import natural_keys, label_figure
from pipeline.python.retinotopy import target_visual_field as vf
from pipeline.python.retinotopy import utils as util
from pipeline.python.retinotopy import do_retinotopy_analysis as ra
from pipeline.python.retinotopy import estimate_RF_size as est
# In[3]:


#get_ipython().magic(u'matplotlib notebook')


# In[4]:


def extract_options(options):
    
    parser = optparse.OptionParser()

    parser.add_option('-D', '--root', action='store', dest='rootdir', default='/n/coxfs01/2p-data', help='data root dir (root project dir containing all animalids) [default: /nas/volume1/2photon/data, /n/coxfs01/2pdata if --slurm]')
    parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')

    # Set specific session/run for current animal:
    parser.add_option('-S', '--session', action='store', dest='session', default='',                       help='session dir (format: YYYMMDD_ANIMALID')
    parser.add_option('-A', '--acq', action='store', dest='acquisition', default='FOV1',                       help="acquisition folder (ex: 'FOV1_zoom3x') [default: FOV1]")
    parser.add_option('-R', '--run', action='store', dest='run', default='retino_run1',                       help="name of run dir containing tiffs to be processed (ex: gratings_phasemod_run1)")
    parser.add_option('-r', '--retinoid', action='store', dest='retinoid', default='analysis001',                       help="name of retino ID (roi analysis) [default: analysis001]")
    
    parser.add_option('--angular', action='store_false', dest='use_linear', default=True,                       help="Plot az/el coordinates in angular spce [default: plots linear coords]")
#     parser.add_option('-e', '--thr-el', action='store', dest='fit_thresh_el', default=0.2, \
#                       help="fit threshold for elevation [default: 0.2]")
#     parser.add_option('-a', '--thr-az', action='store', dest='fit_thresh_az', default=0.2, \
#                       help="fit threshold for azimuth [default: 0.2]")
    
    (options, args) = parser.parse_args(options)

    return options


# # Select data source and params

# In[5]:


#options = ['-i', 'JC047', '-S', '20190215', '-A', 'FOV1']
#options = ['-i', 'JC070', '-S', '20190314', '-A', 'FOV1', '-R', 'retino_run1', '-t', 'analysis003']
#options = ['-i', 'JC070', '-S', '20190314', '-A', 'FOV1', '-R', 'retino_run1', '-t', 'analysis003']
#options = ['-i', 'JC070', '-S', '20190315', '-A', 'FOV2', '-R', 'retino_run1', '-t', 'analysis002']
#options = ['-i', 'JC070', '-S', '20190316', '-A', 'FOV1', '-R', 'retino_run1', '-t', 'analysis002']

#options = ['-i', 'JC067', '-S', '20190320', '-A', 'FOV1', '-R', 'retino_run2', '-r', 'analysis002']
#options = ['-i', 'JC076', '-S', '20190406', '-A', 'FOV1', '-R', 'retino_run1', '-r', 'analysis002']
#options = ['-i', 'JC076', '-S', '20190420', '-A', 'FOV1', '-R', 'retino_run1', '-r', 'analysis002']
#options = ['-i', 'JC090', '-S', '20190604', '-A', 'FOV3', '-R', 'retino_run1', '-r', 'analysis002']
#options = ['-i', 'JC085', '-S', '20190622', '-A', 'FOV1', '-R', 'retino_run1', '-r', 'analysis002']
#options = ['-i', 'JC084', '-S', '20190522', '-A', 'FOV1', '-R', 'retino_run2', '-r', 'analysis002']

#options = ['-i', 'JC091', '-S', '20190623', '-A', 'FOV1', '-R', 'retino_run1', '-r', 'analysis002']
options = ['-i', 'JC085', '-S', '20190626', '-A', 'FOV1', '-R', 'retino_run1', '-r', 'analysis002']


opts = extract_options(options)

rootdir = opts.rootdir
animalid = opts.animalid
session = opts.session
fov = opts.acquisition
run = opts.run
retinoid = opts.retinoid
use_linear = opts.use_linear
#fit_thresh_az = float(opts.fit_thresh_az)
#fit_thresh_el = float(opts.fit_thresh_el) #0.2


# # Select analyzed dataset
#%%
use_pixels = False

run_dir = glob.glob(os.path.join(rootdir, animalid, session, '%s*' % fov, run))[0]
fov = os.path.split(os.path.split(run_dir)[0])[-1]
print("FOV: %s, run: %s" % (fov, run))
retinoids_fpath = glob.glob(os.path.join(run_dir, 'retino_analysis', 'analysisids_*.json'))[0]
with open(retinoids_fpath, 'r') as f:
    rids = json.load(f)
if use_pixels:
    roi_analyses = [r for r, rinfo in rids.items() if rinfo['PARAMS']['roi_type'] == 'pixels']
else:
    roi_analyses = [r for r, rinfo in rids.items() if rinfo['PARAMS']['roi_type'] != 'pixels']
if retinoid not in roi_analyses:
    retinoid = sorted(roi_analyses, key=natural_keys)[-1] # use most recent roi analysis
    print("Fixed retino id to most recent: %s" % retinoid)

data_identifier = '|'.join([animalid, session, fov, run, retinoid])

print("*** Dataset: %s ***" % data_identifier)

#%%
# Get processed retino data:
processed_dir = glob.glob(os.path.join(run_dir, 'retino_analysis', '%s*' % retinoid))[0]
processed_fpaths = glob.glob(os.path.join(processed_dir, 'files', '*.h5'))
print("Found %i processed retino runs." % len(processed_fpaths))

# Get condition info for trials:
conditions_fpath = glob.glob(os.path.join(run_dir, 'paradigm', 'files', 'parsed_trials*.json'))[0]


# # Set output dir

# In[23]:


output_dir = os.path.join(processed_dir, 'visualization', 'absolute_maps')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
print("Saving fit results to: %s" % output_dir)


# # Get screen info

# In[8]:


# Get screen info:
screen = util.get_screen_info(animalid, session, rootdir=rootdir)


# Convert phase to linear coords:
screen_left = -1*screen['azimuth']/2.
screen_right = screen['azimuth']/2. #screen['azimuth']/2.
screen_lower = -1*screen['elevation']/2.
screen_upper = screen['elevation']/2. #screen['elevation']/2.


# # Load data

# ### 1.  Load zprojection image for visualization

# In[10]:


mean_projection_fpaths = glob.glob(os.path.join('%s_mean_deinterleaved' % rids[retinoid]['SRC'],
                                                'visible', '*.tif'))
print "Found %i mean-projection imgs." % len(mean_projection_fpaths)
imgs = []
for fp in mean_projection_fpaths:
    im = tf.imread(fp)
    
    if 'zoom1p0x' in fov:
        import cv2
        im = cv2.resize(im, (512, 512))
    else:
        if rids[retinoid]['PARAMS']['downsample_factor'] is not None:
            ds = int(rids[retinoid]['PARAMS']['downsample_factor'])
            im = ra.block_mean(im, ds)
    tmp_im = np.uint8(np.true_divide(im, np.max(im))*255)
    imgs.append(tmp_im)
    
fov_img = np.array(imgs).mean(axis=0)

pl.figure()
pl.imshow(fov_img, 'gray')
pl.axis('off')


# ### 2.  Load averaged traces (timecourses)

# In[11]:


# This file gets created with visualization/get_session_summary() when estimate_RF_size.py is used

avg_traces_fpath = glob.glob(os.path.join(processed_dir, 'traces', '*.pkl'))
if len(avg_traces_fpath) == 0:
    print "Getting averaged ROI traces by cond and estimating RFs..."
    # Need to run estimate_RF_size.py:
    est.estimate_RFs_and_plot(['-i', animalid,'-S', session, '-A', fov, '-R', run, '-r', retinoid])
    avg_traces_fpath = glob.glob(os.path.join(processed_dir, 'traces', '*.pkl'))[0]
else:
    avg_traces_fpath = avg_traces_fpath[0]
    
print("Loading pre-averaged traces from: %s" % avg_traces_fpath)

with open(avg_traces_fpath, 'rb') as f:
    traces = pkl.load(f)
print("averaged_traces.pkl contains:", traces.keys())

# trials_by_cond = traces['conditions']
# print("Conditions (by rep):", trials_by_cond)

print "Cond dict:", traces['traces']['right']


# ### 3.  Load and format FFT analysis

# In[89]:


# Comine all trial data into data frames:
fit, magratio, phase, trials_by_cond = util.trials_to_dataframes(processed_fpaths, conditions_fpath)
#print fit.head()
print trials_by_cond
conditions = trials_by_cond.keys()

# Correct phase to wrap around:
corrected_phase = util.correct_phase_wrap(phase)


# ### 4.  Get stimulus parameters

# In[15]:


# Get cycle starts:
cond = 'right'
stimfreq = traces['traces'][cond]['info']['stimfreq']
stimperiod = 1./stimfreq # sec per cycle
fr = traces['traces'][cond]['info']['frame_rate']
nframes = traces['traces'][cond]['traces'].shape[-1]

ncycles = int(round((nframes/fr) / stimperiod))
print stimperiod

nframes_per_cycle = int(np.floor(stimperiod * fr))
cycle_starts = np.round(np.arange(0, stimperiod*fr*ncycles, nframes_per_cycle)).astype('int')
print("Cycle starts (%i cycles):" % (ncycles), cycle_starts)


# # Visualize "strong" cells

# In[72]:


use_magratio = True

fit_thr = 0.2
#mag_thr = 0.02
mag_thr = 0.01 #magratio.max(axis=1).max() * 0.25 #0.02

if use_magratio:
    roi_metric = 'magratio'
    means_by_metric = magratio.mean(axis=1)
    metric_thr = mag_thr
else:
    roi_metric = 'sinufit'
    means_by_metric = fit.mean(axis=1)
    metric_thr = fit_thr
    
strong_cells = means_by_metric[means_by_metric >= metric_thr].index.tolist()

print('ROIs with best %s (n=%i, thr=%.3f):' % (roi_metric, len(strong_cells), metric_thr), strong_cells)
print "Means by metric:"
print means_by_metric.head(), means_by_metric.shape


# ### 1.  Plot timecourses by condition for each cell

# In[80]:


rid = strong_cells[0]
fig, axes = pl.subplots(1, 2, figsize=(12,3))

ax = axes[0]
ax.plot(traces['traces']['right']['traces'][rid, :], 'orange', lw=0.5, label='right')
ax.plot(traces['traces']['left']['traces'][rid, :], 'b', lw=0.5, label='left')
for cyc in cycle_starts:
    ax.axvline(x=cyc, color='k', lw=0.5, linestyle=':')
ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.5))
sns.despine(ax=ax, trim=True, offset=2)

ax = axes[1]
ax.plot(traces['traces']['top']['traces'][rid, :], 'orange', lw=0.5, label='top')
ax.plot(traces['traces']['bottom']['traces'][rid, :], 'b', lw=0.5, label='bottom')
for cyc in cycle_starts:
    ax.axvline(x=cyc, color='k', lw=0.5, linestyle=':')
ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.5))
sns.despine(ax=ax, trim=True, offset=2)

fig.suptitle('roi %i (%s %.2f)' % (int(rid+1), roi_metric, means_by_metric[rid]), fontsize=10)

fig.subplots_adjust(right=0.8, top=0.7, bottom=0.3, left=0.05, wspace=0.2)

label_figure(fig, data_identifier)


# In[103]:



if not os.path.exists(os.path.join(output_dir, 'roi_traces')):
    os.makedirs(os.path.join(output_dir, 'roi_traces'))

for ri,rid in enumerate(strong_cells):
    fig, axes = pl.subplots(1, 2, figsize=(12,3))

    ax = axes[0]
    ax.plot(traces['traces']['right']['traces'][rid, :], 'orange', lw=0.5, label='right')
    ax.plot(traces['traces']['left']['traces'][rid, :], 'b', lw=0.5, label='left')
    for cyc in cycle_starts:
        ax.axvline(x=cyc, color='k', lw=0.5, linestyle=':')
    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.5))
    sns.despine(ax=ax, trim=True, offset=2)

    ax = axes[1]
    ax.plot(traces['traces']['top']['traces'][rid, :], 'orange', lw=0.5, label='top')
    ax.plot(traces['traces']['bottom']['traces'][rid, :], 'b', lw=0.5, label='bottom')
    for cyc in cycle_starts:
        ax.axvline(x=cyc, color='k', lw=0.5, linestyle=':')
    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.5))
    sns.despine(ax=ax, trim=True, offset=2)

    fig.suptitle('roi %i (%s %.2f)' % (int(rid+1), roi_metric, means_by_metric[rid]), fontsize=10)
    fig.subplots_adjust(right=0.8, top=0.7, bottom=0.3, left=0.05, wspace=0.2)

    label_figure(fig, data_identifier)
    pl.savefig(os.path.join(output_dir, 'roi_traces', 'average_traces_by_cond_roi%05d.png' % int(rid+1)))
    pl.close()
    

# # Get absolute maps

# #### Get average phase of each condition (for all cells)

# In[194]:


# Use uncorrected/wrapped phase values:

lowval = -np.pi
highval = np.pi

mean_phases = pd.concat((pd.DataFrame(sp.stats.circmean(phase[trials_by_cond[cond]], axis=1, low=lowval, high=highval),
                                    columns=[cond], index=phase.index) for cond in conditions), axis=1)

mean_phases.head()                 


# #### Load masks for ROIs

# In[195]:


masks = traces['masks']
nrois, d1, d2 = masks.shape



# First look at left v. right:
#
#is_azimuth = True
threshold_by_metric = True
#
#if is_azimuth:
#    c1 = 'left' #'right'
#    c2 = 'right' #'left'
#else:
#    c1 = 'bottom' #'right'
#    c2 = 'top' #'left'


for (c1, c2) in [('left', 'right'), ('bottom', 'top')]:
    
    print c1, c2
    
    # #### Apply phase values to masks
    tmp_cmask1 = np.ones((d1, d2))*100
    tmp_cmask2 = np.ones((d1, d2)) *100
    
    if threshold_by_metric:
        roi_list = copy.copy(strong_cells)
    else:
        roi_list = np.arange(0, nrois)
        
    for ri in roi_list: #np.arange(0, nrois):
        maskpix = np.where(np.squeeze(masks[ri,:,:])) 
            
        tmp_cmask1[maskpix] = mean_phases[c1][ri] 
        tmp_cmask2[maskpix] = mean_phases[c2][ri]
    
        tmp_cmask1 = np.ma.array(tmp_cmask1) # create masking array
        cmask1 = np.ma.masked_where(tmp_cmask1 == 100 , tmp_cmask1) # mask non-roi pixels
    
        tmp_cmask2 = np.ma.array(tmp_cmask2) # create masking array
        cmask2 = np.ma.masked_where(tmp_cmask2 == 100 , tmp_cmask2) # mask non-roi pixels
    
    
    # # Plot absolute maps
    
    
    fig, axes = pl.subplots(3,3, figsize=(8,15)) #pl.figure()
    pl.subplots_adjust(hspace=0.2, wspace=0.15, top=0.8, left=0.01, right=0.99, bottom=0.01)
    
    if c1 == 'right' or c1 == 'left':
        cbar_orientation='horizontal'
        cbar_axes1 = [0.1, 0.8, 0.2, 0.05]
        cbar_axes2 = [0.4, 0.8, 0.2, 0.05]
        cbar_axes1a = [0.1, 0.52, 0.2, 0.05]
        cbar_axes2a = [0.4, 0.52, 0.2, 0.05]
    else:
        cbar_orientation='vertical'
        cbar_axes1 = [0.1, 0.9, 0.2, 0.05]
        cbar_axes2 = [0.4, 0.9, 0.2, 0.05]
        
    # 1. Condition1 map
    ax = axes[0, 0]
    ax.imshow(fov_img, 'gray')
    cmask1_wrap = util.correct_phase_wrap(cmask1)
    im = ax.imshow(cmask1_wrap.T, cmap='nipy_spectral', vmin=0, vmax=np.pi*2)
    ax.set_title(c1); ax.axis('off')
    cbaxes = fig.add_axes(cbar_axes1) 
    cb = pl.colorbar(im, cax = cbaxes, orientation=cbar_orientation)  
    # if is_azimuth:
    #     cb.ax.invert_xaxis()
    # else:
    #     cb.ax.invert_yaxis() 
    
    
    # 2.  Condition2 map
    ax = axes[0, 1]
    ax.imshow(fov_img, 'gray')
    cmask2_wrap = util.correct_phase_wrap(cmask2)
    im = ax.imshow(cmask2_wrap.T, cmap='nipy_spectral', vmin=0, vmax=np.pi*2)
    ax.set_title(c2); ax.axis('off')
    cbaxes = fig.add_axes(cbar_axes2) 
    cb = pl.colorbar(im, cax = cbaxes, orientation=cbar_orientation)  
    if c1 == 'right' or c1 == 'left':
        cb.ax.invert_xaxis()
    else:
        cb.ax.invert_yaxis() 
        
    # --------------------------------------------------
    # 3.  Delay map:
    ax = axes[0, 2]
    ax.imshow(fov_img, 'gray')
    delay_map = (cmask1 + cmask2) / 2.
    #delay_map = vf.correct_phase_wrap(delay_map) # do dis
    delay = delay_map.mean()
    im = ax.imshow(delay_map.T, cmap='nipy_spectral', vmin=0, vmax=np.pi*2)
    ax.set_title('delay'); ax.axis('off')
    ax.annotate('avg delay: %.2f' % delay_map.mean(), (0, 0.5))
    
    # --------------------------------------------------
    
    # 4. Abs map - cond1 
    ax = axes[1, 0]
    ax.imshow(fov_img, 'gray')
    abs_map1 = (cmask1_wrap - cmask2_wrap) / 2.
    #abs_map1 = vf.correct_phase_wrap(abs_map1) <-- dont do this if starting w/ corrected
    im = ax.imshow(abs_map1.T, cmap='nipy_spectral', vmin=-np.pi, vmax=np.pi) #vmin=0, vmax=np.pi*2)
    ax.set_title('abs'); ax.axis('off')
    cbaxes = fig.add_axes(cbar_axes1a) 
    cb = pl.colorbar(im, cax = cbaxes, orientation=cbar_orientation)  
    # if is_azimuth:
    #     cb.ax.invert_xaxis()
    # else:
    #     cb.ax.invert_yaxis() 
        
        
    # 5. Abs map - cond2
    ax = axes[1, 1]
    ax.imshow(fov_img, 'gray')
    abs_map2 = (cmask2_wrap - cmask1_wrap) / 2.
    #abs_map2 = vf.correct_phase_wrap(abs_map2) <-- dont do this if starting w/ corrected
    im = ax.imshow(abs_map2.T, cmap='nipy_spectral_r', vmin=-np.pi, vmax=np.pi) # <-- use reverse cmap to match abs condN
    ax.set_title('abs - rev cmap'); ax.axis('off')
    cbaxes = fig.add_axes(cbar_axes2a) 
    cb = pl.colorbar(im, cax = cbaxes, orientation=cbar_orientation)  
    if c1 == 'right' or c1 == 'left':
        cb.ax.invert_xaxis()
    else:
        cb.ax.invert_yaxis() 
        
    # --------------------------------------------------
    
    # 6. Cond1, shifted by delay
    ax = axes[2, 0]
    ax.imshow(fov_img, 'gray')
    #shift_map1 = cmask1_wrap - delay_map # <-- for shift, need phase calculated by non-wrapped, subtract from wrapped
    shift_map1 = cmask1_wrap - np.abs(delay) #delay_map
    #shift_map1 = vf.correct_phase_wrap(shift_map1)
    im = ax.imshow(shift_map1.T, cmap='nipy_spectral', vmin=0, vmax=np.pi*2)
    ax.set_title('%s - delay' % c1); ax.axis('off')
    
    # 7. Cond2, shifted by delay
    ax = axes[2, 1]
    ax.imshow(fov_img, 'gray')
    shift_map2 = cmask2_wrap - np.abs(delay) # delay_map
    #shift_map2 = cmask2 - delay_map
    #shift_map2 = vf.correct_phase_wrap(shift_map2)
    im = ax.imshow(shift_map2.T, cmap='nipy_spectral_r', vmin=0, vmax=np.pi*2)
    ax.set_title('%s - delay (+wrap)' % c2); ax.axis('off')
    
    
    axes[1,2].axis('off')
    axes[2,2].axis('off')
    
    if c1 == 'right' or c1 =='left':
        fig.suptitle('azimuth')
    else:
        fig.suptitle('elevation')
    
    label_figure(fig, data_identifier)
    
    if threshold_by_metric: 
        figname = 'absolute_phase_by_roi_%s_%s_%s_thr%.2f.png' % (c1, c2, roi_metric, metric_thr)
    else:
        figname = 'absolute_phase_by_roi_%s_%s_all.png' % (c1, c2)
    pl.savefig(os.path.join(output_dir, figname))
    
    
    print "Mean delay (std): %.2f (%.2f)" % (delay_map.mean(), delay_map.std())
    print "%s (mean): %.2f" % (c1, cmask1.mean()) # cmask2.max()
    print "%s (mean): %.2f" % (c2, cmask2.mean()) # cmask2.max()
    
    
    print "%s - %s: max/min = %.2f, %.2f" % (c1, c2, abs_map1.max(),abs_map1.min())
    print "%s - %s: max/min = %.2f, %.2f" % (c2, c1, abs_map2.max(),abs_map2.min())
    
    # Save absolute map to var:
    if c1 == 'left' or c1 == 'right':
        absolute_az = abs_map1.copy()
    else:
        absolute_el = abs_map1.copy()
        

    #pl.close()
    
    
# # Plot a nicer plot with just absolute maps and legend

# In[191]:


fig, axes = pl.subplots(1,2, figsize=(8,6))


cbar_axes1 = [0.22, 0.85, 0.15, 0.1]
cbar_axes2 = [0.65, 0.85, 0.15, 0.1]

    
ax = axes[0]
ax.imshow(fov_img, 'gray')
im = ax.imshow(absolute_az.T, cmap='nipy_spectral_r', vmin=-np.pi, vmax=np.pi) #vmin=0, vmax=np.pi*2)
ax.set_title('azimuth'); ax.axis('off')
cbaxes = fig.add_axes(cbar_axes1) 
cb = pl.colorbar(im, cax = cbaxes, orientation='horizontal')  

# 5. Abs map - cond2
ax = axes[1]
ax.imshow(fov_img, 'gray')
im = ax.imshow(absolute_el.T, cmap='nipy_spectral_r', vmin=-np.pi, vmax=np.pi) # <-- use reverse cmap to match abs condN
ax.set_title('elevation'); ax.axis('off')
cbaxes = fig.add_axes(cbar_axes2) 
cb = pl.colorbar(im, cax = cbaxes, orientation='vertical')  

pl.subplots_adjust(wspace=0.2, hspace=0.1)

label_figure(fig, data_identifier)

pl.savefig(os.path.join(output_dir, 'absolute_maps_%s_thr%.2f.png' % (roi_metric, metric_thr)))


# In[ ]:





# In[ ]:




