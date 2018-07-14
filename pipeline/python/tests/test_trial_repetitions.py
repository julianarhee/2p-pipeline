#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 20:36:25 2018

@author: juliana
"""
import matplotlib as mpl
mpl.use('agg')
import os
import sys
import optparse
import random
import seaborn as sns
import numpy as np
import pandas as pd
import pylab as pl
from scipy import stats
from pipeline.python.paradigm import utils as util

from pipeline.python.paradigm import align_acquisition_events as acq
from pipeline.python.traces.utils import get_frame_info

#rootdir = '/mnt/odyssey'
#animalid = 'CE077'
#session = '20180521'
#acquisition = 'FOV2_zoom1x'
#run = 'gratings_run1'
#traceid = 'traces001'
#run_dir = os.path.join(rootdir, animalid, session, acquisition, run)
#iti_pre = 1.0


options = ['-D', '/mnt/odyssey', '-i', 'CE077', '-S', '20180523', '-A', 'FOV1_zoom1x',
           '-T', 'np_subtracted',
           '-R', 'blobs_run2', '-t', 'traces001', '-d', 'dff']

#options = ['-D', '/mnt/odyssey', '-i', 'CE077', '-S', '20180602', '-A', 'FOV1_zoom1x',
#           '-T', 'np_subtracted',
#           '-R', 'blobs_dynamic_run7', '-t', 'traces001', '-d', 'dff']

#options = ['-D', '/mnt/odyssey', '-i', 'CE077', '-S', '20180612', '-A', 'FOV1_zoom1x',
#           '-T', 'np_subtracted',
#           '-R', 'blobs_run1', '-t', 'traces001', '-d', 'dff', 
#           '-r', 'yrot', '-c', 'xpos', '-H', 'morphlevel']

#options = ['-D', '/mnt/odyssey', '-i', 'CE077', '-S', '20180626', '-A', 'FOV1_zoom1x',
#           '-T', 'np_subtracted',
#           '-R', 'gratings_rotating_drifting', '-t', 'traces001', '-d', 'dff']


def extract_options(options):

    parser = optparse.OptionParser()

    parser.add_option('-D', '--root', action='store', dest='rootdir',
                          default='/nas/volume1/2photon/data',
                          help='data root dir (dir w/ all animalids) [default: /nas/volume1/2photon/data, /n/coxfs01/2pdata if --slurm]')
    parser.add_option('--slurm', action='store_true', dest='slurm', default=False, help="set if running as SLURM job on Odyssey")

    parser.add_option('-i', '--animalid', action='store', dest='animalid',
                          default='', help='Animal ID')

    # Set specific session/run for current animal:
    parser.add_option('-S', '--session', action='store', dest='session',
                          default='', help='session dir (format: YYYMMDD_ANIMALID')
    parser.add_option('-A', '--acq', action='store', dest='acquisition',
                          default='FOV1', help="acquisition folder (ex: 'FOV1_zoom3x') [default: FOV1]")
    parser.add_option('-T', '--trace-type', action='store', dest='trace_type',
                          default='raw', help="trace type [default: 'raw']")
    parser.add_option('-R', '--run', dest='run_list', default=[], nargs=1,
                          action='append',
                          help="run ID in order of runs")
    parser.add_option('-t', '--traceid', dest='traceid_list', default=[], nargs=1,
                          action='append',
                          help="trace ID in order of runs")
    
    # Set specific session/run for current animal:
    parser.add_option('-d', '--datatype', action='store', dest='datatype',
                          default='corrected', help='Traces to plot (must be in dataset.npz [default: corrected]')
    parser.add_option('--offset', action='store_true', dest='correct_offset',
                          default=False, help='Set to correct df/f offset after drift correction')           
    parser.add_option('-f', '--filetype', action='store', dest='filetype',
                          default='png', help='File type for images [default: png]')
    parser.add_option('--scale', action='store_true', dest='scale_y',
                          default=False, help='Set to scale y-axis across roi images')
    parser.add_option('-y', '--ymax', action='store', dest='dfmax',
                          default=None, help='Set value for y-axis scaling (if not provided, and --scale, uses max across rois)')
    parser.add_option('--shade', action='store_false', dest='plot_trials',
                          default=True, help='Set to plot mean and sem as shaded (default plots individual trials)')
    parser.add_option('-r', '--rows', action='store', dest='rows',
                          default=None, help='Transform to plot along ROWS (only relevant if >2 trans_types) - default uses objects or morphlevel')
    parser.add_option('-c', '--columns', action='store', dest='columns',
                          default=None, help='Transform to plot along COLUMNS')
    parser.add_option('-H', '--hue', action='store', dest='subplot_hue',
                          default=None, help='Transform to plot by HUE within each subplot')
    
    (options, args) = parser.parse_args(options)
    if options.slurm:
        options.rootdir = '/n/coxfs01/2p-data'
    
    return options


#%% Extract options and get path to data sources:
    

optsE = extract_options(options)

#traceid_dir = util.get_traceid_dir(options)
run = optsE.run_list[0]
traceid = optsE.traceid_list[0]
acquisition_dir = os.path.join(optsE.rootdir, optsE.animalid, optsE.session, optsE.acquisition)

traceid_dir = util.get_traceid_from_acquisition(acquisition_dir, run, traceid)
data_fpath = os.path.join(traceid_dir, 'data_arrays', 'datasets.npz')
print "Loaded data from: %s" % traceid_dir
#    dataset = np.load(data_fpath)
#    print dataset.keys()
#        
#%% Load data:

dataset = np.load(data_fpath)
print dataset.keys()
    

#%% 

from scipy.stats.stats import pearsonr   
from scipy import stats

info = dataset['run_info'][()]
nframes_per_trial = info['nframes_per_trial']
ntrials_total = info['ntrials_total']
print "N frames per trial:", nframes_per_trial
labels = dataset['ylabels']
zscores = dataset['zscore']
nrois = zscores.shape[-1]

ntrials_per_cond = list(set(info['ntrials_by_cond'].values()))[0]


# Make sure all trials have the same structure (only for runs where stim_dur does not vary)
assert len(labels) == nframes_per_trial * ntrials_total, "N datapoints (%i) does not match total." % len(labels)

# Get condition labels for each trial:
config_labels = np.reshape(labels, (ntrials_total, nframes_per_trial), order='C')[:, 0] # only need first column of each trial
assert len(config_labels) == zscores.shape[0], "N labels (%i) does not match N zscore datapoints (%i)" % (len(config_labels), zscores.shape[0])

conditions = list(set(config_labels))

nreps_to_test = [int(n) for n in np.arange(2, int(np.floor(ntrials_per_cond/2.))+1)]

corrs = {}
for nreps in nreps_to_test:
    # Randomly draw N trial indices:
    trial_ixs = random.sample(range(0, ntrials_per_cond), int(nreps*2))
    
    pcorrs = []
    for cond in conditions:
        ixs = np.where(config_labels==cond)[0][trial_ixs]
        half_a = np.mean(zscores[ixs[0:nreps], :], axis=0)
        half_b = np.mean(zscores[ixs[nreps:], :], axis=0)
        
        pcorrs.append(pearsonr(half_a, half_b)[0])
        
    corrs[nreps] = pcorrs
    


# Plot correlation as a function of nreps:
pl.figure()
mean_corrs = [np.mean(corrs[nreps]) for nreps in nreps_to_test]
sem_corrs = [stats.sem(corrs[nreps]) for nreps in nreps_to_test]
pl.plot(nreps_to_test, mean_corrs)
pl.errorbar(nreps_to_test, mean_corrs, yerr=sem_corrs)
pl.xticks(nreps_to_test)
pl.xlabel('N reps in half')
pl.ylabel('pearsons corr')
sns.despine(offset=4, trim=True)



 
        