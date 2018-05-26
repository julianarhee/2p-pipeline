#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May  9 16:10:21 2018

@author: juliana
"""

import h5py
import os
import json
import cv2
import time
import math
import random
import itertools
import optparse
import pandas as pd
import numpy as np
import pylab as pl
import matplotlib as mpl
import seaborn as sns
import pyvttbl as pt
import multiprocessing as mp
import tifffile as tf
from collections import namedtuple
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import scikit_posthocs as sp
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison

from pipeline.python.utils import natural_keys, replace_root, print_elapsed_time
import pipeline.python.traces.combine_runs as cb
import pipeline.python.paradigm.align_acquisition_events as acq
import pipeline.python.visualization.plot_psths_from_dataframe as vis
from pipeline.python.traces.utils import load_TID

from mpl_toolkits.axes_grid1 import make_axes_locatable


# Get mean trace for each condition:
def get_mean_cond_traces(roi, Xdata, ylabels):
    conditions = sorted(list(set(ylabels)))
    ridx = int(roi[3:]) - 1
    mean_cond_traces = []
    mean_tsecs = []
    for cond in conditions:
        ixs = np.where(ylabels==cond)
        curr_trace = np.squeeze(Xdata[ixs, ridx])
        curr_tracemat = np.reshape(curr_trace, (ntrials_per_stim, nframes_per_trial))
        curr_tsecs = np.reshape(np.squeeze(tsecs[ixs,ridx]), (ntrials_per_stim, nframes_per_trial))

        #print curr_tracemat.shape
        mean_ctrace = np.mean(curr_tracemat, axis=0)
        mean_cond_traces.append(mean_ctrace)
        mean_tsecs = np.mean(curr_tsecs, axis=0)

    mean_cond_traces = np.array(mean_cond_traces)
    mean_tsecs = np.array(mean_tsecs)
    #print mean_cond_traces.shape
    return mean_cond_traces, mean_tsecs



#%%

from statsmodels.nonparametric.smoothers_lowess import lowess

def smooth_traces(trace, frac=0.002):
    '''
    lowess algo (from docs):

    Suppose the input data has N points. The algorithm works by estimating the
    smooth y_i by taking the frac*N closest points to (x_i,y_i) based on their
    x values and estimating y_i using a weighted linear regression. The weight
    for (x_j,y_j) is tricube function applied to abs(x_i-x_j).
    '''
    xvals = np.arange(len(trace))
    print len(xvals)
    filtered = lowess(trace, xvals, is_sorted=True, frac=frac, it=0, missing='drop', return_sorted=False)
    print filtered.shape
    if len(filtered.shape)==1:
        return filtered
    else:
        return filtered[:, 1]


#%%
def load_roi_dataframe(roidata_filepath):

    fn_parts = os.path.split(roidata_filepath)[-1].split('_')
    roidata_hash = fn_parts[1]
    trace_type = os.path.splitext(fn_parts[-1])[0]

    df_list = []

    #DATA = pd.read_hdf(combined_roidata_fpath, key=datakey, mode='r')

    df = pd.HDFStore(roidata_filepath, 'r')
    datakeys = df.keys()
    if 'roi' in datakeys[0]:
        for roi in datakeys:
            if '/' in roi:
                roiname = roi[1:]
            else:
                roiname = roi
            dfr = df[roi]
            dfr['roi'] = pd.Series(np.tile(roiname, (len(dfr .index),)), index=dfr.index)
            df_list.append(dfr)
        DATA = pd.concat(df_list, axis=0, ignore_index=True)
        datakey = '%s_%s' % (trace_type, roidata_hash)
    else:
        print "Found %i datakeys" % len(datakeys)
        datakey = datakeys[0]
        #df.close()
        #del df
        DATA = pd.read_hdf(roidata_filepath, key=datakey, mode='r')
        #DATA = df[datakey]
        df.close()
        del df

    return DATA, datakey

#%%

def extract_options(options):

    parser = optparse.OptionParser()

    parser.add_option('-D', '--root', action='store', dest='rootdir',
                          default='/nas/volume1/2photon/data',
                          help='data root dir (dir w/ all animalids) [default: /nas/volume1/2photon/data, /n/coxfs01/2pdata if --slurm]')
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
    parser.add_option('-n', '--nruns', action='store', dest='nruns', default=1, help="Number of consecutive runs if combined")
    parser.add_option('--slurm', action='store_true', dest='slurm', default=False, help="set if running as SLURM job on Odyssey")
    parser.add_option('--par', action='store_true', dest='multiproc', default=False, help="set if want to run MP on roi stats, when possible")
    parser.add_option('--nproc', action='store', dest='nprocesses', default=4, help="N processes if running in par (default=4)")

    parser.add_option('--combo', action='store_true', dest='combined', default=False, help="Set if using combined runs with same default name (blobs_run1, blobs_run2, etc.)")


    # Pupil filtering info:
    parser.add_option('--no-pupil', action="store_false",
                      dest="filter_pupil", default=True, help="Set flag NOT to filter PSTH traces by pupil threshold params")
    parser.add_option('-s', '--radius-min', action="store",
                      dest="pupil_radius_min", default=25, help="Cut-off for smnallest pupil radius, if --pupil set [default: 25]")
    parser.add_option('-B', '--radius-max', action="store",
                      dest="pupil_radius_max", default=65, help="Cut-off for biggest pupil radius, if --pupil set [default: 65]")
    parser.add_option('-d', '--dist', action="store",
                      dest="pupil_dist_thr", default=5, help="Cut-off for pupil distance from start, if --pupil set [default: 5]")

    (options, args) = parser.parse_args(options)

    return options

#%%

optset1 = ['-D', '/mnt/odyssey', '-i', 'CE084', '-S', '20180507', '-A', 'FOV1_zoom1x',
           '-T', 'np_subtracted', '--no-pupil',
           '-R', 'gratings_run2', '-t', 'traces001',
           '-n', '1']


optset2 = ['-D', '/mnt/odyssey', '-i', 'CE084', '-S', '20180507', '-A', 'FOV2_zoom1x',
           '-T', 'np_subtracted', '--no-pupil',
           '-R', 'gratings_run1', '-t', 'traces001',
           '-n', '1']


optset3 = ['-D', '/mnt/odyssey', '-i', 'CE084', '-S', '20180507', '-A', 'FOV3_zoom1x',
           '-T', 'np_subtracted', '--no-pupil',
           '-R', 'gratings_run1', '-t', 'traces001',
           '-n', '1']

options_list = [optset1, optset2, optset3]

averages_df = []
normed_df = []

#%%

for options in options_list:

    options = extract_options(options)

    rootdir = options.rootdir
    animalid = options.animalid
    session = options.session
    acquisition = options.acquisition
    slurm = options.slurm
    if slurm is True:
        rootdir = '/n/coxfs01/2p-data'

    trace_type = options.trace_type

    run_list = options.run_list
    traceid_list = options.traceid_list

    filter_pupil = options.filter_pupil
    pupil_radius_max = float(options.pupil_radius_max)
    pupil_radius_min = float(options.pupil_radius_min)
    pupil_dist_thr = float(options.pupil_dist_thr)
    pupil_max_nblinks = 0

    multiproc = options.multiproc
    nprocesses = int(options.nprocesses)
    combined = options.combined
    nruns = int(options.nruns)

    #%
    # =============================================================================
    # Create output dir for combined acquisitions:
    # =============================================================================
    session_dir = os.path.join(rootdir, animalid, session)
    output_basedir = os.path.join(session_dir, 'joint_datasets')
    if not os.path.exists(output_basedir):
        os.makedirs(output_basedir)
    print output_basedir


    acquisition_dir = os.path.join(rootdir, animalid, session, acquisition)
    if combined is False:
        runfolder = run_list[0]
        traceid = traceid_list[0]
        with open(os.path.join(acquisition_dir, runfolder, 'traces', 'traceids_%s.json' % runfolder), 'r') as f:
            tdict = json.load(f)
        tracefolder = '%s_%s' % (traceid, tdict[traceid]['trace_hash'])
        traceid_dir = os.path.join(rootdir, animalid, session, acquisition, runfolder, 'traces', tracefolder)
    else:
        assert len(run_list) == nruns, "Incorrect runs or number of runs (%i) specified!\n%s" % (nruns, str(run_list))
        if len(run_list) > 2:
            runfolder = '_'.join([run_list[0], 'thru', run_list[-1]])
        else:
            runfolder = '_'.join(run_list)
        if len(traceid_list)==1:
            if len(run_list) > 2:
                traceid = traceid_list[0]
            else:
                traceid = '_'.join([traceid_list[0] for i in range(nruns)])
        traceid_dir = os.path.join(rootdir, animalid, session, acquisition, runfolder, traceid)


    print(traceid_dir)
    assert os.path.exists(traceid_dir), "Specified traceid-dir does not exist!"

    #%%

    #% # Load ROIDATA file:
    print "Loading ROIDATA file..."

    roidf_fn = [i for i in os.listdir(traceid_dir) if i.endswith('hdf5') and 'ROIDATA' in i and trace_type in i][0]
    roidata_filepath = os.path.join(traceid_dir, roidf_fn) #'ROIDATA_098054_626d01_raw.hdf5')
    DATA, datakey = load_roi_dataframe(roidata_filepath)

    transform_dict, object_transformations = vis.get_object_transforms(DATA)
    trans_types = object_transformations.keys()

    #%% Set filter params:

    if filter_pupil is True:
        pupil_params = acq.set_pupil_params(radius_min=pupil_radius_min,
                                            radius_max=pupil_radius_max,
                                            dist_thr=pupil_dist_thr,
                                            create_empty=False)
    elif filter_pupil is False:
        pupil_params = acq.set_pupil_params(create_empty=True)


#    #%% Calculate metrics & get stats ---------------------------------------------
#
#    print "Getting ROI STATS..."
#    STATS, stats_filepath = cb.get_combined_stats(DATA, datakey, traceid_dir, trace_type=trace_type, filter_pupil=filter_pupil, pupil_params=pupil_params)


    #%% Get stimulus config info:assign_roi_selectivity
    # =============================================================================

    rundir = os.path.join(rootdir, animalid, session, acquisition, runfolder)

    if combined is True:
        stimconfigs_fpath = os.path.join(traceid_dir, 'stimulus_configs.json')
    else:
        stimconfigs_fpath = os.path.join(rundir, 'paradigm', 'stimulus_configs.json')

    with open(stimconfigs_fpath, 'r') as f:
        stimconfigs = json.load(f)

    print "Loaded %i stimulus configurations." % len(stimconfigs.keys())

    #%

    configs = sorted([k for k in stimconfigs.keys()], key=lambda x: stimconfigs[x]['rotation'])
    orientations = [stimconfigs[c]['rotation'] for c in configs]
    nconds = len(orientations)

    #%% Format ROIDATA into dataframes:
    # =============================================================================
    #
    #stats = STATS[['roi', 'config', 'trial', 'baseline_df', 'stim_df', 'zscore']] #STATS['zscore']
    #
    #std_baseline = stats['stim_df'].values / stats['zscore'].values
    #zscored_resp = (stats['stim_df'].values - stats['baseline_df'].values ) /std_baseline
    #
    #zscore_vals = stats['zscore'].values

    assert len(list(set(DATA['first_on'])))==1, "More than 1 frame idx found for stimulus ON"
    assert len(list(set(DATA['nframes_on'])))==1, "More than 1 value found for nframes on."

    stim_on_frame = int(list(set(DATA['first_on']))[0])
    nframes_on = int(round(list(set(DATA['nframes_on']))[0]))

    # Turn DF values into matrix with rows=trial, cols=df value for each frame:
    roi_list = sorted(list(set(DATA['roi'])), key=natural_keys)
    nrois = len(roi_list)


    sDATA = DATA[['roi', 'config', 'trial', 'raw', 'df', 'tsec']].reset_index()
    sDATA.loc[:, 'config'] = [stimconfigs[c]['rotation'] for c in sDATA.loc[:,'config'].values]
    sDATA = sDATA.sort_values(by=['config', 'trial'], inplace=False)
    sDATA.head()

    nframes_per_trial = len(sDATA[sDATA['trial']=='trial00001']['tsec']) / nrois
    ntrials_per_stim = len(list(set(sDATA[sDATA['config']==0]['trial']))) # Assumes all stim have same # trials!

    #%% FORMAT DATA:
    # -----------------------------------------------------------------------------

    trace = 'df'

    trial_labels = [t[2] for t in sDATA.groupby(['roi', 'config', 'trial'])[trace].apply(np.array).keys().tolist()]
    config_labels = [t[1] for t in sDATA.groupby(['roi', 'config', 'trial'])[trace].apply(np.array).keys().tolist()]

    # Get labels:
    ntrials_total = len(list(set(trial_labels)))
    trial_labels = np.reshape(trial_labels, (nrois, ntrials_total))[0,:]    # DON'T SORT trials, since these are ordered by stimulus angle
    config_labels = np.reshape(config_labels, (nrois, ntrials_total))[0,:]  # These are already sorted, by increasing angle

    roi_labels = [t[0] for t in sDATA.groupby(['roi', 'config', 'trial'])[trace].apply(np.array).keys().tolist()]

    # Get list of features and class labels:
    roi_list = sorted(list(set(roi_labels)), key=natural_keys)  # Sort by name, since dataframe grouped by ROI first
    #trial_list = list(set(trial_labels))                        # DON'T SORT trials, since these are ordered by stimulus angle
    ntrials_total = len(list(set(trial_labels)))


    print "-----------------------------------------"
    print "Run summary:"
    print "-----------------------------------------"
    print "N rois:", len(roi_list)
    print "N conds:", nconds
    print "N trials:", ntrials_total
    print "N frames per trial:", nframes_per_trial
    print "N trials per stimulus:", ntrials_per_stim
    print "-----------------------------------------"

    #%% Load time-course data:
    #% TIMECOURSE --format input data

    sort_by_config = False

    Xdata = np.array(sDATA.sort_values(['trial', 'tsec']).groupby(['roi'])['df'].apply(np.array).tolist()).T         # rows = frames, cols = rois
    ylabels = np.array(sDATA.sort_values(['trial', 'tsec']).groupby(['roi'])['config'].apply(np.array).tolist()).T[:,0] # only need one col, since trial id same for all rois
    groups = np.array(sDATA.sort_values(['trial']).groupby(['roi'])['trial'].apply(np.array).tolist()).T[:,0]
    tsecs = np.array(sDATA.sort_values(['trial', 'tsec']).groupby(['roi'])['tsec'].apply(np.array).tolist()).T


#    sample_list = []
#    class_list = []
#    tpoints_list = []
#    if sort_by_config:
#        trial_sorter = np.copy(trial_labels)
#    else:
#        trial_sorter = sorted(trial_labels, key=natural_keys)
#
#    for trial in trial_sorter:
#        img = np.vstack([vals for vals in sDATA[sDATA['trial']==trial].groupby(['roi'])['df'].apply(np.array).values])
#        tsec = np.vstack([vals for vals in sDATA[sDATA['trial']==trial].groupby(['roi'])['tsec'].apply(np.array).values])
#        curr_config = sDATA[sDATA['trial']==trial]['config'].values[0]  #[0]
#        sample_list.append(img)
#        class_list.append(curr_config)
#        tpoints_list.append(tsec)
#
#    ylabels = np.hstack([np.tile(c, (nframes_per_trial, )) for c in class_list])
#    Xdata = np.vstack([s.T for s in sample_list])
#    groups = np.hstack([np.tile(c, (nframes_per_trial, )) for c in range(len(trial_labels))]) #y])
#    tsecs = np.vstack([s.T for s in tpoints_list])

    #
    print "X:", Xdata.shape
    print "y (labels):", ylabels.shape
    print "N groupings of trials:", len(list(set(groups)))

    print "N samples: %i, N features: %i" % (Xdata.shape[0], Xdata.shape[1])

    #%% smooth that shit:


    # TO TEST fraction:
    # Test smoothing on a few ROIs:
    ridx = 101
    trace_test = Xdata[:, ridx:ridx+1]
    print trace_test.shape


#    trace_test_filt = np.apply_along_axis(smooth_traces, 0, trace_test, frac=0.0003)
#    print trace_test_filt.shape

    # Plot the same trial on top:
    ixs = np.where(ylabels==225)[0]

    #frac_range = np.linspace(0.0001, 0.005, num=8)
    frac_range = np.linspace(0.0005, 0.0015, num=8)
    fig, axes = pl.subplots(2,4) #pl.figure()
    #ax = axes.flat()
    for i, ax, in enumerate(axes.flat):

        trace_test_filt = np.apply_along_axis(smooth_traces, 0, trace_test, frac=frac_range[i])
        print trace_test_filt.shape

        tmat = []
        for tidx in range(ntrials_per_stim):
            fstart = tidx*nframes_per_trial
            #pl.plot(xrange(nframes_per_trial), trace_test[ixs[fstart]:ixs[fstart]+nframes_per_trial, 0])
            tr = trace_test_filt[ixs[fstart]:ixs[fstart]+nframes_per_trial, 0]
            ax.plot(xrange(nframes_per_trial), trace_test_filt[ixs[fstart]:ixs[fstart]+nframes_per_trial, 0], 'k', linewidth=0.5)
            tmat.append(tr)
        ax.plot(xrange(nframes_per_trial), np.mean(np.array(tmat), axis=0), 'r', linewidth=1)
        ax.set_title(frac_range[i])
#
#
#    frac = 0.001 # For a single trial, fraction is a larger number (since frac = fraction of size input data to use)
#
#    # Test different values for fraction:
#    filtered = lowess(trace, xvals, is_sorted=True, frac=frac, it=0)
#    pl.figure();
#    pl.plot(xvals, trace, 'r')
#    pl.plot(filtered[:,0], filtered[:,1], 'k')


    #%% SMOOTH data, since frame rate is very high (44.75 Hz)
    print "SMOOTHING...."

    # CHeck for NANs:

    X = np.apply_along_axis(smooth_traces, 0, Xdata, frac=0.001)
    y = ylabels.copy()
    print X.shape


    #%%

    smooth = True

    for roi in roi_list:
        if smooth:
            mean_cond_traces, mean_tsecs = get_mean_cond_traces(roi, X, y)
        else:
            mean_cond_traces, mean_tsecs = get_mean_cond_traces(roi, Xdata, ylabels)

        xcond_mean = np.mean(mean_cond_traces, axis=0)
        normed = mean_cond_traces - xcond_mean

        averages_df.append(pd.DataFrame(data=np.reshape(mean_cond_traces, (nconds*nframes_per_trial,)),
                                        columns = [roi],
                                        index=np.array(range(nconds*nframes_per_trial))
                                        ))

        normed_df.append(pd.DataFrame(data=np.reshape(normed, (nconds*nframes_per_trial,)),
                                        columns = [roi],
                                         index=np.array(range(nconds*nframes_per_trial))
                                        ))

#%% Concat into datagrame

import scipy.io
a = averages_df[0:-5]
print len(a)

avgDF = pd.concat(a, axis=1)
avgDF.head()

normDF = pd.concat(normed_df, axis=1)
normDF.head()


#%%
norm_labels = np.hstack([np.tile(c, (nframes_per_trial,)) for c in orientations])

D = {}
for cond in orientations:
    ixs = np.where(norm_labels==cond)[0]
    curr_cond_data = avgDF.values[ixs, :]
    D['angle_%s' % str(cond)] = curr_cond_data
D['time_ms'] = mean_tsecs * 1000

mfile_path = os.path.join(output_basedir, 'jpca_avg_trials.mat')
print mfile_path

scipy.io.savemat(mfile_path, D)