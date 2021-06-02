#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 17:07:49 2018

@author: juliana
"""

import os
import glob

import numpy as np
import pandas as pd

from pipeline.python.utils import natural_keys, label_figure

# -----------------------------------------------
# Data source:
# -----------------------------------------------
rootdir = '/n/coxfs01/2p-data'
animalid = 'JC026'
session = '20181209'
acquisition = 'FOV1_zoom2p0x'
run = 'gratings_run1'

# -----------------------------------------------
# Compare traces extracted by manual vs. cnmf:
# -----------------------------------------------
manual_id = 'traces001'
cnmf_id = 'cnmf001'
roi_id = 'rois001'


# Get paths:
# -----------------------------------------------
fov_dir = os.path.join(rootdir, animalid, session, acquisition)
manual_dir = glob.glob(os.path.join(fov_dir, run, 'traces', '%s*' % manual_id))[0]
nmf_dir = glob.glob(os.path.join(fov_dir, run, 'traces', '%s*' % cnmf_id))[0]

manual_fpath = glob.glob(os.path.join(manual_dir, 'data_arrays', 'datasets.npz'))[0]
nmf_fpath = glob.glob(os.path.join(nmf_dir, 'data_arrays', 'datasets.npz'))[0]

man = np.load(manual_fpath)
nmf = np.load(nmf_fpath)

# Extract traces:
inputdata = 'corrected'
man_data = man[inputdata]
nmf_data = nmf[inputdata]


run_info = man['run_info'][()]
nframes_per_trial = run_info['nframes_per_trial']
ntrials_by_cond = run_info['ntrials_by_cond']
ntrials_total = sum([val for k,val in ntrials_by_cond.iteritems()])

nrois = man_data.shape[-1]
labels_df = pd.DataFrame(data=man['labels_data'], columns=man['labels_columns'])
config_list = sorted(list(set(labels_df['config'])), key=natural_keys)

def zscored_tracemat(tracemat, stim_on, nframes_on):
    basemat = np.std(tracemat[:, 0:stim_on], axis=1)
    stimmat = np.mean(tracemat[:, stim_on:stim_on+nframes_on], axis=1)
    zscores_by_trial = stimmat / basemat
    zscores_mean = np.mean(zscores_by_trial)
    zscores_std = np.std(zscores_by_trial)

    return zscores_mean, zscores_std

def raw_to_dff(tracemat, stim_on, nframes_on):
    baseline_mat = tracemat[:, 0:stim_on]
    basemat = np.mean(baseline_mat, axis=1)
    dff = ((tracemat.T - basemat) / basemat).T
    dff = (tracemat.T - basemat).T
    return dff


ridx = 138


M_means = []; M_stds = [];
N_means = []; N_stds = [];
for ridx in range(nrois):
    
    man_zscores={}
    nmf_zscores={}

    man_rdata = labels_df.copy()
    man_rdata['data'] = man_data[:, ridx]
    
    nmf_rdata = labels_df.copy()
    nmf_rdata['data'] = nmf_data[:, ridx]
    
    
    for curr_config in config_list:
        man_df = man_rdata[man_rdata['config']==curr_config]
        man_traces = np.vstack(man_df.groupby('trial')['data'].apply(np.array))
        #man_traces -= np.mean(man_traces[:, 0:stim_on])
        
        
        nmf_df = nmf_rdata[nmf_rdata['config']==curr_config]
        nmf_traces = np.vstack(nmf_df.groupby('trial')['data'].apply(np.array))
        #nmf_traces -= np.mean(nmf_traces[:, 0:stim_on])
        
        assert len(list(set(man_df['nframes_on']))) == 1, "More than 1 stimdur parsed for current config..."
        stim_on = list(set(labels_df[labels_df['config']==curr_config]['stim_on_frame']))[0]
        nframes_on = list(set(labels_df[labels_df['config']==curr_config]['nframes_on']))[0]
    
        man_traces = raw_to_dff(man_traces, stim_on, nframes_on)
        nmf_traces = raw_to_dff(nmf_traces, stim_on, nframes_on)


        man_zmean, man_zstd = zscored_tracemat(man_traces, stim_on, nframes_on)
        nmf_zmean, nmf_zstd = zscored_tracemat(nmf_traces, stim_on, nframes_on)
    
        man_zscores[curr_config] = (man_zmean, man_zstd)
        nmf_zscores[curr_config] = (nmf_zmean, nmf_zstd)
        
    manual_z = np.nanmean([vals[0] for key, vals in man_zscores.items()])
    nmf_z  =  np.nanmean([vals[0] for key, vals in nmf_zscores.items()])
    
    manual_stds = np.nanstd([vals[0] for key, vals in man_zscores.items()])
    nmf_stds = np.nanstd([vals[0] for key, vals in nmf_zscores.items()])
    
    M_means.append(manual_z)
    N_means.append(nmf_z)

    M_stds.append(manual_stds)
    N_stds.append(nmf_stds)



pl.figure()
pl.plot(M_means, N_means, '.')
pl.errorbar(M_means, N_means, yerr=N_stds, xerr=M_stds, elinewidth=0.5, fmt='none')