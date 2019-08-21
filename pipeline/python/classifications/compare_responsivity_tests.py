#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 15:45:27 2019

@author: julianarhee
"""


import os
import glob
import json
import h5py
import optparse
import sys
import math
import cPickle as pkl
import json

import matplotlib as mpl
mpl.use('agg')

import numpy as np
import pandas as pd
import pylab as pl
import seaborn as sns
import multiprocessing as mp

from sklearn.utils import shuffle

from pipeline.python.classifications import experiment_classes as util
from pipeline.python.classifications import test_responsivity as resp
from pipeline.python.utils import label_figure, natural_keys

from pipeline.python.classifications import responsivity_stats as respstats



def extract_options(options):

    parser = optparse.OptionParser()

    # PATH opts:
    parser.add_option('-D', '--root', action='store', dest='rootdir', default='/n/coxfs01/2p-data', help='data root dir (root project dir containing all animalids) [default: /nas/volume1/2photon/data, /n/coxfs01/2pdata if --slurm]')
    parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')
    parser.add_option('-S', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID')
    parser.add_option('-A', '--acq', action='store', dest='fov', default='FOV1_zoom2p0x', help="acquisition folder (ex: 'FOV1_zoom2p0x') [default: FOV1_zoom2p0x]")
    parser.add_option('-E', '--exp', action='store', dest='experiment_type', default='', help="Name of experiment (stimulus type), e.g., rfs")
    parser.add_option('--slurm', action='store_true', dest='slurm', default=False, help="set if running as SLURM job on Odyssey")
    parser.add_option('-t', '--trace-id', action='store', dest='traceid', default='', help="Trace ID for current trace set (created with set_trace_params.py, e.g., traces001, traces020, etc.)")

    parser.add_option('-n', '--nproc', action="store",
                      dest="n_processes", default=2, help="N processes [default: 1]")
    parser.add_option('-d', '--trace-type', action="store",
                      dest="trace_type", default='corrected', help="Trace type to use for calculating stats [default: corrected]")

    parser.add_option('-m', '--minframes', action="store",
                      dest="minframes", default=10, help="Min n frames above baseline std (default: 10)") # for bootstrap [default: 1000]")
    parser.add_option('-b', '--stds', action="store",
                      dest="n_stds", default=2.5, help="N stds above/below baseline (default: 2.5)") # for bootstrap [default: 1000]")
    
    parser.add_option('--plot', action='store_true', dest='plot_rois', default=False, help="set to plot results of each roi's analysis")

    (options, args) = parser.parse_args(options)

    return options
#%%

options = ['-i', 'JC084', '-S', '20190522', '-A', 'FOV1_zoom2p0x', '-t', 'traces001', 
           '-E', 'gratings', '-n', 1, '-d', 'dff']

opts = extract_options(options)


#%%

animalid = opts.animalid
session = opts.session


fov = opts.fov
traceid = opts.traceid
trace_type = opts.trace_type
min_nframes = opts.minframes
n_stds = opts.n_stds



experiment_type = opts.experiment_type



#%% Get aggregated session data
from pipeline.python.classifications import get_dataset_stats as gd
options = ['-t', 'traces001']
optsE = gd.extract_options(options)
aggregate_dir = optsE.aggregate_dir
print aggregate_dir



nstd_desc = util.get_stats_desc(traceid=optsE.traceid, trace_type=optsE.trace_type, response_type=optsE.response_type,
                                responsive_test='nstds', responsive_thr=min_nframes, n_stds=n_stds)
roc_desc = util.get_stats_desc(traceid=optsE.traceid, trace_type=optsE.trace_type, response_type=optsE.response_type,
                                responsive_test='roc', responsive_thr=0.05)

data_identifier = '|'.join([optsE.fov_type, optsE.traceid, '\nstd-%s' % nstd_desc, '\nroc-%s' % roc_desc])
print data_identifier

#%%

sdata_fpath = os.path.join(aggregate_dir, 'dataset_info.pkl')
if os.path.exists(sdata_fpath):
    with open(sdata_fpath, 'rb') as f:
        sdata = pkl.load(f)
else:
    sdata = gd.aggregate_session_info(traceid=optsE.traceid, trace_type=optsE.trace_type, 
                                       state=optsE.state, fov_type=optsE.fov_type, 
                                       visual_areas=optsE.visual_areas,
                                       blacklist=optsE.blacklist, 
                                       rootdir=optsE.rootdir)
    with open(sdata_fpath, 'wb') as f:
        pkl.dump(sdata, f, protocol=pkl.HIGHEST_PROTOCOL)
    
#%%
#visual_area = 'V1'
#sinfo = sdata[sdata['visual_area']==visual_area]


stds={}; rocs={}; ncells={};
for (visual_area, experiment_type, animalid, session, fov), g in sdata.groupby(['visual_area', 'experiment', 'animalid', 'session', 'fov']):
    skey = '-'.join([visual_area, animalid, session, fov])
    if skey not in stds.keys():
        stds[skey] = {}
        rocs[skey] = {}
        ncells[skey] = {}
    #%
    if experiment_type == 'gratings':
        # Initialize experiment object
        exp = util.Gratings(animalid, session, fov, traceid=traceid)
        #exp.print_info()
    elif experiment_type == 'blobs':
        exp = util.Objects(animalid, session, fov, traceid=traceid)
    else:
        continue
    
    nstd_cells, nrois_total1 = exp.get_responsive_cells(responsive_test='nstds', n_stds=n_stds, responsive_thr=min_nframes)
    roc_cells, nrois_total2 = exp.get_responsive_cells(responsive_test='ROC', responsive_thr=0.05)
    
    assert nrois_total1 == nrois_total2, "Um wrong n cells..."
    #ncells_total = len(exp.data.info['roi_list'])
    
    stds[skey][experiment_type] = nstd_cells
    rocs[skey][experiment_type] = roc_cells
    ncells[skey][experiment_type] = nrois_total1
        
        
    #print("minframes: %i out of %i cells pass" % (len(nstd_cells), ncells_total))
    #print("ROC: %i out of %i cells pass" % (len(roc_cells), ncells_total))
#%%
counts=[]
for skey in ncells.keys():
    for exp in ncells[skey].keys():
        visual_area, animalid, session, fov = skey.split('-')
        cc = pd.Series({'visual_area': visual_area,
                        'animalid': animalid,
                        'session': session,
                        'fov': fov,
                         'exp': exp,
                         'roc': len(rocs[skey][exp]),
                         'std': len(stds[skey][exp]),
                         'total': ncells[skey][exp]})
        counts.append(cc)

counts = pd.DataFrame(counts)
              
#%%
visual_areas = ['V1', 'Lm', 'Li']
colors = ['k', 'royalblue', 'darkorange'] #sns.color_palette(palette='colorblind') #, n_colors=3)
area_colors = {'V1': colors[0], 'Lm': colors[1], 'Li': colors[2]}

#%%
from sklearn.linear_model import LinearRegression


minv = min([counts['roc'].min(), counts['std'].min()])
maxv = max([counts['roc'].max(), counts['std'].max()])

fig, axes = pl.subplots(2,3, figsize=(15,12), sharex=True, sharey=True)
ai = 0
for (experiment, visual_area), g in counts.groupby(['exp', 'visual_area']):
    ax = axes.flat[ai]
    sns.regplot('roc', 'std', data=g, ax=ax, color=area_colors[visual_area])
    ax.set_aspect('equal')
    ax.set_xlim([minv, maxv])
    ax.set_ylim([minv, maxv])
    
    regr = LinearRegression()  # create object for the class
    xv = g['roc'].values.reshape(-1, 1)
    yv = g['std'].values.reshape(-1, 1)
    regr.fit(xv, yv)  # perform linear regression
    fitv = regr.predict(xv)  # make predictions
    ax.plot(minv, minv, alpha=0, label='y=%.2fx + %.2f' % (float(regr.coef_), float(regr.intercept_)))
    ax.legend(loc='lower right')
    ax.set_title('%s: %s' % (visual_area, experiment))
    
    ax.set_xlabel('N cells w/ roc')
    ax.set_ylabel('N cells w/ std')
    ax.set_aspect('equal')
        
    ai += 1

pl.subplots_adjust(top=0.9, hspace=0.5)

label_figure(fig, data_identifier)


pl.savefig(os.path.join(aggregate_dir, 'responsivity', 'compare_ROC-vs-NSTDS-by-experiment.png'))

#%%

experiment_type = 'gratings'
fig, ax =pl.subplots()
for skey in stds.keys():
    if experiment_type not in stds[skey].keys():
        continue
    df = pd.DataFrame({'std': stds[skey][experiment_type],
                       'roc': rocs[skey][experiment_type]})
    sns.regplot('std', 'roc', data=df, ax=ax)


#%%

