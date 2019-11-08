#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 15:49:33 2019

@author: julianarhee
"""

import os
import glob
import json
import copy
import optparse

import pylab as pl
import seaborn as sns
import cPickle as pkl
import numpy as np
import scipy as sp
from scipy import stats
import pandas as pd

import scipy.optimize as opt
from matplotlib.patches import Ellipse, Rectangle

from mpl_toolkits.axes_grid1 import AxesGrid
from pipeline.python.utils import natural_keys, label_figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import argrelextrema
from scipy.interpolate import splrep, sproot, splev, interp1d

from pipeline.python.retinotopy import utils as rutils
from pipeline.python.retinotopy import fit_2d_rfs as rf


#%%

#%%

def extract_options(options):
    
    parser = optparse.OptionParser()

    parser.add_option('-D', '--root', action='store', dest='rootdir', default='/n/coxfs01/2p-data',\
                      help='data root dir (root project dir containing all animalids) [default: /n/coxfs01/2pdata]')
    parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')

    # Set specific session/run for current animal:
    parser.add_option('-S', '--session', action='store', dest='session', default='', \
                      help='session dir (format: YYYMMDD_ANIMALID')
    parser.add_option('-A', '--acq', action='store', dest='acquisition', default='FOV1', \
                      help="acquisition folder (ex: 'FOV1_zoom3x') [default: FOV1]")
    parser.add_option('-R', '--run', action='store', dest='run', default='retino_run1', \
                      help="name of run dir containing tiffs to be processed (ex: gratings_phasemod_run1)")
    parser.add_option('-t', '--traceid', action='store', dest='traceid', default='traces001', \
                      help="name of traces ID [default: traces001]")
    parser.add_option('-d', '--data-type', action='store', dest='trace_type', default='corrected', \
                      help="Trace type to use for analysis [default: corrected]")
    parser.add_option('--segment', action='store_true', dest='segment', default=False, \
                      help="Set flag to use segmentation of FOV for select visual area")
    parser.add_option('-V', '--area', action='store', dest='visual_area', default='', \
                      help="Name of visual area, if --segment is flagged")
    (options, args) = parser.parse_args(options)

    return options


#%%

rootdir = '/n/coxfs01/2p-data'


# Anesthetized:
#V1_datasets = {'JC076': ['20190408_FOV1'], #'20190408_FOV1', '20190419_FOV1'],
#               'JC083': ['20190505_FOV1'],
#               'JC084': ['20190518_FOV1', '20190518_FOV2']
#               }
#
#LM_datasets = {'JC076': ['20190424_FOV1', '20190427_FOV1'],
#               'JC078': ['20190424_FOV1', '20190427_FOV1'],
#               'JC080': ['20190430_FOV1'],
#               'JC083': ['20190505_FOV2']
#               }
#
#LI_datasets = {'JC076': ['20190410_FOV1', '20190503_FOV1'],
#               'JC086': ['20190515_FOV1'], #, '20190515_FOV2'],
#               'JC089': ['20190520_FOV1']
#               }

# Awake:
V1_datasets = {'JC076': ['20190420_FOV1', '20190501_FOV1'],
               'JC083': ['20190507_FOV1', '20190510_FOV1', '20190511_FOV1'],
               'JC084': ['20190522_FOV1'] #, '20190510_FOV1']
               }

LM_datasets = {'JC078': ['20190426_FOV1', '20190430_FOV1', '20190504_FOV1',\
                         '20190509_FOV1', '20190513_FOV1'],
               'JC080': ['20190506_FOV1', '20190602_FOV2', '20190603_FOV1'],
               'JC083': ['20190508_FOV1', '20190512_FOV1', '20190517_FOV1'],
               'JC084': ['20190525_FOV1']
               }

LI_datasets = {'JC076': ['20190502_FOV1'],
               'JC090': ['20190605_FOV1'],
               'JC091': ['20190602_FOV1', '20190606_FOV1']
               }

#%%


def get_traceid(animalid, session, fov, traceid='traces001', rootdir='/n/coxfs01/2p-data'):
        
    fov_dir = glob.glob(os.path.join(rootdir, animalid, session, '%s*' % fov))[0]
    if int(session) < 20190511:
        traceid_dirs = glob.glob(os.path.join(fov_dir, 'combined_gratings*', 'traces', '%s*' % traceid))
    else:
        traceid_dirs = glob.glob(os.path.join(fov_dir, 'combined_rfs*', 'traces', '%s*' % traceid))
    if len(traceid_dirs) > 1:
        print "More than 1 trace ID found:"
        for ti, traceid_dir in enumerate(traceid_dirs):
            print ti, traceid_dir
        sel = input("Select IDX of traceid to use: ")
        traceid_dir = traceid_dirs[int(sel)]
    else:
        print(traceid_dirs)
        traceid_dir = traceid_dirs[0]
        
    return traceid_dir

def get_rf_results(traceid_dir, rf_param_str='', trace_type='corrected',
                   visual_area='', select_rois=False):
    
    rf_dir = os.path.join(traceid_dir, 'figures', 'receptive_fields', rf_param_str)
    #results_outfile = 'roi_fit_results_2dgaus_%s_%.2f_set_%s.pkl' % (cutoff_type, map_thr, set_to_min_str)
    results_outfile = 'RESULTS_%s.pkl' % rf_param_str
    print("Loading... %s" % traceid_dir.split('/traces/')[0])
    if not os.path.exists(os.path.join(rf_dir, results_outfile)):
        rf_params = rf_param_str.split('responsemin')[-1]
        thr_info = rf_params.split('_')[1]
        if 'snr' in thr_info: 
            metric_type = 'snr'
            response_thr = float(thr_info.split('snr')[1])
        else:
            metric_type = 'zscore'
            response_thr = float(thr_info.split('zscore')[1])
        
        traceid = os.path.split(traceid_dir)[-1].split('_')[0]
        run_dir = traceid_dir.split('/traces/')[0]
        run = os.path.split(run_dir)[-1]
        fov_dir = os.path.split(run_dir)[0]
        fov = os.path.split(fov_dir)[-1]
        session_dir = os.path.split(fov_dir)[0]
        session = os.path.split(session_dir)[-1]
        animalid = os.path.split(os.path.split(session_dir)[0])[-1]
        results = rf.fit_2d_receptive_fields(animalid, session, fov, run, traceid, 
                                             trace_type=trace_type, visual_area=visual_area, select_rois=select_rois,
                                             metric_type=metric_type, response_thr=response_thr)


    #assert os.path.exists(os.path.join(rf_dir, results_outfile)), "No RF fits with specified params found! -- %s" % results_outfile
    print "Loading existing results..."
    with open(os.path.join(rf_dir, results_outfile), 'rb') as f:
        results = pkl.load(f)
    
    return results


def get_rfs_by_visual_area(V1_datasets, rf_param_str='', fit_thr=0.7):
    V1 = {}
    for animalid, dataids in V1_datasets.items():
        print "Loading %i datasets for: %s" % (len(dataids), animalid)
        traceid_dirs = [get_traceid(animalid, dataid.split('_')[0], dataid.split('_')[1]) for dataid in dataids]
        
        tmp_fits = []
        for traceid_dir in traceid_dirs:
            rf_results = get_rf_results(traceid_dirs[0], rf_param_str=rf_param_str) 
            fitdf = pd.DataFrame(rf_results['fits']).T
            
            tmp_fits.append(fitdf)
            
        V1[animalid] = pd.concat(tmp_fits, axis=0).reset_index()
        
        fitted_rois = V1[animalid][V1[animalid]['r2'] >= fit_thr].sort_values('r2', axis=0, ascending=False).index.tolist()
        print "%i out of %i fit rois with r2 > %.2f" % (len(fitted_rois), V1[animalid].shape[0], fit_thr)

    # TODO:  fix this so that RF values are scaled to each stimulus set (RFs can be 10deg or 5deg resolution)
    data_fpath = glob.glob(os.path.join(traceid_dir, 'data_arrays', '*.npz'))[0]
    dset = np.load(data_fpath)
    sdf = pd.DataFrame(dset['sconfigs'][()]).T

    return V1, sdf

#%%

from matplotlib.pyplot import cm

def convert_values(oldval, newmin=None, newmax=None, oldmax=None, oldmin=None):
    oldrange = (oldmax - oldmin)
    newrange = (newmax - newmin)
    newval = (((oldval - oldmin) * newrange) / oldrange) + newmin
    return newval

def convert_fit_to_coords(fitdf, row_vals, col_vals, rid=None):
    
    if rid is not None:
        xx = convert_values(fitdf['x0'][rid], 
                            newmin=min(col_vals), newmax=max(col_vals), 
                            oldmax=len(col_vals)-1, oldmin=0)
        
        sigma_x = convert_values(abs(fitdf['sigma_x'][rid]), 
                            newmin=0, newmax=max(col_vals)-min(col_vals), 
                            oldmax=len(col_vals)-1, oldmin=0)
        
        yy = convert_values(fitdf['y0'][rid], 
                            newmin=min(row_vals), newmax=max(row_vals), 
                            oldmax=len(row_vals)-1, oldmin=0)
        
        sigma_y = convert_values(abs(fitdf['sigma_y'][rid]), 
                            newmin=0, newmax=max(row_vals)-min(row_vals), 
                            oldmax=len(row_vals)-1, oldmin=0)
    else:
        xx = convert_values(fitdf['x0'], 
                            newmin=min(col_vals), newmax=max(col_vals), 
                            oldmax=len(col_vals)-1, oldmin=0)
        
        sigma_x = convert_values(abs(fitdf['sigma_x']), 
                            newmin=0, newmax=max(col_vals)-min(col_vals), 
                            oldmax=len(col_vals)-1, oldmin=0)
        
        yy = convert_values(fitdf['y0'], 
                            newmin=min(row_vals), newmax=max(row_vals), 
                            oldmax=len(row_vals)-1, oldmin=0)
        
        sigma_y = convert_values(abs(fitdf['sigma_y']), 
                            newmin=0, newmax=max(row_vals)-min(row_vals), 
                            oldmax=len(row_vals)-1, oldmin=0)
    
    return xx, yy, sigma_x, sigma_y


def get_rf_stats(rfdf, row_vals, col_vals, max_size=80, fit_thr=0.7, sigma_scale=2.35, tile_res=5):
    
    #fit_roi_list = rfdf[((rfdf['r2'] >= fit_thr) & (rfdf[['sigma_x', 'sigma_y']].mean(axis=1)*tile_res*sigma_scale <= max_size))].sort_values('r2', axis=0, ascending=False).index.tolist()
    fit_roi_list = rfdf[rfdf['r2']>=fit_thr].sort_values('r2', axis=0, ascending=False).index.tolist()
    rstats = pd.DataFrame([convert_fit_to_coords(rfdf, row_vals, col_vals, rid=rid) for rid in fit_roi_list],
                           columns=['x0', 'y0', 'width', 'height'])
    
    
#    majors = [abs(rfdf['sigma_x'][rid])*sigma_scale*x_res for rid in fit_roi_list]
#    minors = [abs(rfdf['sigma_y'][rid])*sigma_scale*y_res for rid in fit_roi_list]
#    center_xs = [convert_values(fitdf['x0'][rid], 
#                            newmin=min(col_vals), newmax=max(col_vals), 
#                            oldmax=len(col_vals)-1, oldmin=0) for rid in fit_roi_list]
#    center_ys = [convert_values(fitdf['y0'], 
#                            newmin=min(row_vals), newmax=max(row_vals), 
#                            oldmax=len(row_vals)-1, oldmin=0) for rid in fit_roi_list]

    return rstats

#%%
metric_type = 'snr'
response_thr = 1.5
trim = False

perc_min = 0.5
hard_cutoff = True   # Use hard cut-off for zscores (set to False to use some % of max value)
set_to_min = True    # Threshold x,y condition grid and set non-passing conditions to min value or 0.
set_to_min_str = 'set_min' if set_to_min else 'set_zeros'

if not trim:
    set_to_min = False
    hard_cutoff = False
    set_to_min_str = ''    
    cutoff_type = 'no_trim'
    map_thr=''
else:
    cutoff_type = 'hard_thr' if hard_cutoff else 'perc_min'
    map_thr = 1.5 if (trim and hard_cutoff) else perc_min


# Create subdir for saving figs/results based on fit params:
# -----------------------------------------------------------------------------
#rf_param_str = 'rfs_2dgaus_responsemin_%s%.2f_%s_%s' % (metric_type, response_thr, cutoff_type, set_to_min_str)
rf_param_str = 'rfs_2dgaus_responsemin_%s%.2f_%s_%s' % (metric_type, response_thr, cutoff_type, set_to_min_str)

print rf_param_str

fit_thr = 0.5
V1, sdf = get_rfs_by_visual_area(V1_datasets, fit_thr=fit_thr, rf_param_str=rf_param_str)
LM, _ = get_rfs_by_visual_area(LM_datasets, fit_thr=fit_thr, rf_param_str=rf_param_str)
LI, _ = get_rfs_by_visual_area(LI_datasets, fit_thr=fit_thr, rf_param_str=rf_param_str)

#%%

fit_thr = 0.5
sigma_scale = 2.35   # Value to scale sigma in order to get FW (instead of FWHM)
rows = 'ypos'
cols = 'xpos'
max_size = 100
plot_kde = False
plot_rug = False
norm_hist = True
if plot_kde:
    hist_alpha = 0.5
else:
    hist_alpha = 1.0

#row_vals = sorted(sdf[rows].unique())
#col_vals = sorted(sdf[cols].unique())
    
#x_res = np.unique(np.diff(col_vals))[0]
#y_res = np.unique(np.diff(row_vals))[0]

rfs_list = []
fig = pl.figure()
area_colors = ['cornflowerblue', 'green', 'magenta']
area_names = ['V1', 'LM', 'LI']
area_nrats = [len(V1_datasets.keys()), len(LM_datasets.keys()), len(LI_datasets.keys())]

for vi, (curr_color, visual_area_name, nrats, Vx) in enumerate(zip(area_colors, area_names, area_nrats, [V1, LM, LI])):
    curr_rfs = {}
    for animalid, rfdf in Vx.items():
        nr, nc = rfdf['xx'][0].shape
        if nc == 21:
            row_vals = np.arange(-25, 30, step=5)
            col_vals = np.arange(-50, 55, step=5)
        else:
            row_vals = np.arange(-25, 35, step=10)
            col_vals = np.arange(-50, 60, step=10)
            
        rstats = get_rf_stats(rfdf, row_vals, col_vals, fit_thr=fit_thr, max_size=max_size)
        curr_rfs[animalid] = rstats

        curr_avg_rfs = rstats[['width', 'height']].mean(axis=1)*sigma_scale
        vis_area = [visual_area_name for _ in range(len(curr_avg_rfs))]
        vis_ix = [vi for _ in range(len(curr_avg_rfs))]
        animal_name = [animalid for _ in range(len(curr_avg_rfs))]
        rfs_list.append(pd.DataFrame({'rf': curr_avg_rfs,
                                      'animalid': animalid,
                                      'visual_area': vis_area,
                                      'visual_area_ix': vis_ix
                                      }))
        
        #sns.distplot(rstats[['width', 'height']].mean(axis=1)*sigma_scale, color=curr_color, label=visual_area_name)
    vstats = pd.concat([v for k, v in curr_rfs.items()], axis=0)
    sns.distplot(vstats[['width', 'height']].mean(axis=1)*sigma_scale, norm_hist=norm_hist,
                 color=curr_color, label='%s (n=%i (%i))' % (visual_area_name, nrats, vstats.shape[0]),
                 kde=plot_kde, rug=plot_rug, hist=True,
                 rug_kws={"color": curr_color},
                 #norm_hist=True)
                 hist_kws={"histtype": "step", "linewidth": 2, "color": curr_color, "alpha": hist_alpha})
    
    

pl.legend()
pl.xlabel('RF size (deg)')
if plot_kde:
    pl.ylabel('kde')
else:
    if norm_hist:
        pl.ylabel('fraction')
    else:
        pl.ylabel('counts')
sns.despine(offset=4, trim=True)

#%%

RFs = pd.concat(rfs_list)

sns.violinplot(x='visual_area', y='rf', data=RFs,
               scale='count', inner="stick", palette='muted')

#%%

import itertools
palette = itertools.cycle(sns.color_palette())

fig, ax = pl.subplots()
RFdf_list = []
for animalid, rfdf in V1.items():
    nr, nc = rfdf['xx'][0].shape
    if nc == 21:
        row_vals = np.arange(-25, 30, step=5)
        col_vals = np.arange(-50, 55, step=5)
    else:
        row_vals = np.arange(-25, 35, step=10)
        col_vals = np.arange(-50, 60, step=10)
        
    rstats = get_rf_stats(rfdf, row_vals, col_vals, fit_thr=fit_thr, max_size=max_size)
    avg_rf_sizes = rstats[['width', 'height']].mean(axis=1)*sigma_scale
    sns.swarmplot(avg_rf_sizes, ax=ax, color=next(palette))
    

