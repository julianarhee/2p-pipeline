#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 12:31:35 2018

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
import copy
import scipy.io
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
from statsmodels.nonparametric.smoothers_lowess import lowess
from skimage import exposure

from pipeline.python.utils import natural_keys, replace_root, print_elapsed_time
import pipeline.python.traces.combine_runs as cb
import pipeline.python.paradigm.align_acquisition_events as acq
import pipeline.python.visualization.plot_psths_from_dataframe as vis
from pipeline.python.traces.utils import load_TID

from mpl_toolkits.axes_grid1 import make_axes_locatable
from pipeline.python.traces.utils import get_frame_info

#%% Load Datasets:

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

def get_traceid_dir(options):
    traceid_dir = None

    optsE = extract_options(options)

    rootdir = optsE.rootdir
    animalid = optsE.animalid
    session = optsE.session
    acquisition = optsE.acquisition
    slurm = optsE.slurm
    if slurm is True:
        rootdir = '/n/coxfs01/2p-data'

    trace_type = optsE.trace_type

    run_list = optsE.run_list
    traceid_list = optsE.traceid_list
    combined = optsE.combined
    nruns = int(optsE.nruns)

    # Get paths to data source:
    acquisition_dir = os.path.join(rootdir, animalid, session, acquisition)
    
    if combined is False:
        print "Getting path info for single run dataset..."
        runfolder = run_list[0]
        traceid = traceid_list[0]
        with open(os.path.join(acquisition_dir, runfolder, 'traces', 'traceids_%s.json' % runfolder), 'r') as f:
            tdict = json.load(f)
        tracefolder = '%s_%s' % (traceid, tdict[traceid]['trace_hash'])
        traceid_dir = os.path.join(rootdir, animalid, session, acquisition, runfolder, 'traces', tracefolder)
    else:
        print "Getting path info for combined run dataset..."
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
    
    return traceid_dir


#%
def get_run_details(options, verbose=True):
    run_info = {}

    optsE = extract_options(options)
#
#    rootdir = optsE.rootdir
#    animalid = optsE.animalid
#    session = optsE.session
#    acquisition = optsE.acquisition
#    slurm = optsE.slurm
#    if slurm is True:
#        rootdir = '/n/coxfs01/2p-data'

    trace_type = optsE.trace_type

    #run_list = optsE.run_list
    #traceid_list = optsE.traceid_list

    filter_pupil = optsE.filter_pupil
    pupil_radius_max = float(optsE.pupil_radius_max)
    pupil_radius_min = float(optsE.pupil_radius_min)
    pupil_dist_thr = float(optsE.pupil_dist_thr)
    pupil_max_nblinks = 0

    #multiproc = options.multiproc
    #nprocesses = int(options.nprocesses)
    combined = optsE.combined
    nruns = int(optsE.nruns)

    # Get paths to data source:
    #acquisition_dir = os.path.join(rootdir, animalid, session, acquisition)
    traceid_dir = get_traceid_dir(options)
    
    run_dir = traceid_dir.split('/traces')[0]
    si_info = get_frame_info(run_dir)
    

    #% # Load ROIDATA file:
    print "Loading ROIDATA file..."
    roidf_fn = [i for i in os.listdir(traceid_dir) if i.endswith('hdf5') and 'ROIDATA' in i and trace_type in i][0]
    roidata_filepath = os.path.join(traceid_dir, roidf_fn) #'ROIDATA_098054_626d01_raw.hdf5')
    DATA, datakey = load_roi_dataframe(roidata_filepath)

    transform_dict, object_transformations = vis.get_object_transforms(DATA)
    trans_types = object_transformations.keys()

    #% Set filter params:
    if filter_pupil is True:
        pupil_params = acq.set_pupil_params(radius_min=pupil_radius_min,
                                            radius_max=pupil_radius_max,
                                            dist_thr=pupil_dist_thr,
                                            create_empty=False)
    elif filter_pupil is False:
        pupil_params = acq.set_pupil_params(create_empty=True)


    ##%% Calculate metrics & get stats ---------------------------------------------
    #
    #print "Getting ROI STATS..."
    #STATS, stats_filepath = cb.get_combined_stats(DATA, datakey, traceid_dir, trace_type=trace_type, filter_pupil=filter_pupil, pupil_params=pupil_params)


    #% Get stimulus config info:assign_roi_selectivity
    # =============================================================================
    rundir = traceid_dir.split('/traces')[0] #os.path.join(rootdir, animalid, session, acquisition, runfolder)
    if combined is True:
        stimconfigs_fpath = os.path.join(traceid_dir, 'stimulus_configs.json')
    else:
        stimconfigs_fpath = os.path.join(rundir, 'paradigm', 'stimulus_configs.json')
    with open(stimconfigs_fpath, 'r') as f:
        stimconfigs = json.load(f)
    print "Loaded %i stimulus configurations." % len(stimconfigs.keys())

    #%
    if 'gratings' in traceid_dir:
        stimtype = 'gratings'
        configs = sorted([k for k in stimconfigs.keys()], key=lambda x: stimconfigs[x]['rotation'])
        conditions = [stimconfigs[c]['rotation'] for c in configs]
    else:
        stimtype = 'image'
        conditions = stimconfigs.keys()
    #nconds = len(orientations)

    #%
    # =============================================================================
    # Extract data subset:
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
    if stimtype == 'gratings':
        sDATA.loc[:, 'config'] = [stimconfigs[c]['rotation'] for c in sDATA.loc[:,'config'].values]
    sDATA = sDATA.sort_values(by=['config', 'trial'], inplace=False)
    #sDATA.head()

    nframes_per_trial = len(sDATA[sDATA['trial']=='trial00001']['tsec']) / nrois
    config_list = list(set(sDATA['config']))
    #ntrials_per_stim = len(list(set(sDATA[sDATA['config']==config_list[0]]['trial']))) # Assumes all stim have same # trials!
    ntrials_per_stim = [len(list(set(sDATA[sDATA['config']==c]['trial']))) for c in config_list] # Assumes all stim have same # trials!
    ntrials_total = len(list(set(sDATA['trial'])))

    if verbose:
        print "-------------------------------------------"
        print "Run summary:"
        print "-------------------------------------------"
        print "N rois:", len(roi_list)
        print "N trials:", ntrials_total
        print "N frames per trial:", nframes_per_trial
        print "N trials per stimulus:", ntrials_per_stim
        print "-------------------------------------------"

    run_info['roi_list'] = roi_list
    run_info['ntrials_total'] = ntrials_total
    run_info['nframes_per_trial'] = nframes_per_trial
    run_info['ntrials_per_cond'] = ntrials_per_stim
    run_info['condition_list'] = conditions
    run_info['stim_on_frame'] = stim_on_frame
    run_info['nframes_on'] = nframes_on
    run_info['traceid_dir'] = traceid_dir
    run_info['trace_type'] = trace_type
    run_info['transforms'] = object_transformations
    run_info['datakey'] = datakey
    run_info['trans_types'] = trans_types
    run_info['framerate'] = si_info['framerate']


    return sDATA, run_info, stimconfigs


#%% Format data:
    

def format_stimconfigs(configs):
    
    stimconfigs = copy.deepcopy(configs)
    
    if 'frequency' in configs[configs.keys()[0]].keys():
        stimtype = 'gratings'
    elif 'fps' in configs[configs.keys()[0]].keys():
        stimtype = 'movie'
    else:
        stimtype = 'image'
    
    print "STIM TYPE:", stimtype
        
    # Split position into x,y:
    for config in stimconfigs.keys():
        stimconfigs[config]['xpos'] = configs[config]['position'][0]
        stimconfigs[config]['ypos'] = configs[config]['position'][1]
        stimconfigs[config]['size'] = configs[config]['scale'][0]
        stimconfigs[config].pop('position', None)
        stimconfigs[config].pop('scale', None)
        stimconfigs[config]['stimtype'] = stimtype
        
        # stimulus-type specific variables:
        if stimtype == 'gratings':
            stimconfigs[config]['sf'] = configs[config]['frequency']
            stimconfigs[config]['ori'] = configs[config]['rotation']
            stimconfigs[config].pop('frequency', None)
            stimconfigs[config].pop('rotation', None)
        else:
            transform_variables = ['object', 'xpos', 'ypos', 'size', 'yrot', 'morphlevel', 'stimtype']
            if stimtype == 'image':
                imname = os.path.splitext(configs[config]['filename'])[0]
                if ('CamRot' in imname):
                    objectid = imname.split('_CamRot_')[0]
                    yrot = int(imname.split('_CamRot_y')[-1])
                    if 'N1' in imname or 'D1' in imname:
                        morphlevel = 0
                    elif 'N2' in imname or 'D2' in imname:
                        morphlevel = 22
                    elif 'morph' in imname:
                        morphlevel = int(imname.split('_CamRot_y')[0].split('morph')[-1])   
                elif '_zRot' in imname:
                    # Real-world objects:  format is 'IDENTIFIER_xRot0_yRot0_zRot0'
                    objectid = imname.split('_')[0]
                    yrot = int(imname.split('_')[3][4:])
                    morphlevel = 0
                elif 'morph' in imname: 
                    # These are morphs w/ old naming convention, 'CamRot' not in filename)
                    if '_y' not in imname and '_yrot' not in imname:
                        objectid = imname #'morph' #imname
                        yrot = 0
                        morphlevel = int(imname.split('morph')[-1])
                    else:
                        objectid = imname.split('_y')[0]
                        yrot = int(imname.split('_y')[-1])
                        morphlevel = int(imname.split('_y')[0].split('morph')[-1])
            elif stimtype == 'movie':
                imname = os.path.splitext(configs[config]['filename'])[0]
                objectid = imname.split('_movie')[0] #'_'.join(imname.split('_')[0:-1])
                if 'reverse' in imname:
                    yrot = -1
                else:
                    yrot = 1
                if imname.split('_')[1] == 'D1':
                    morphlevel = 0
                elif imname.split('_')[1] == 'D2':
                    morphlevel = 22
                elif imname.split('_')[1][0] == 'M':
                    # Blob_M11_Rot_y_etc.
                    morphlevel = int(imname.split('_')[1][1:])
                elif imname.split('_')[1] == 'morph':
                    # This is a full morph movie:
                    morphlevel = -1
                    
            stimconfigs[config]['object'] = objectid
            stimconfigs[config]['yrot'] = yrot
            stimconfigs[config]['morphlevel'] = morphlevel
            stimconfigs[config]['stimtype'] = stimtype
        
            for skey in stimconfigs[config].keys():
                if skey not in transform_variables:
                    stimconfigs[config].pop(skey, None)

    return stimconfigs

#%%

# Use cnvlib.smoothing functions to deal get mirrored edges on rolling quantile:
    
def check_inputs(x, width, as_series=True):
    """Transform width into a half-window size.

    `width` is either a fraction of the length of `x` or an integer size of the
    whole window. The output half-window size is truncated to the length of `x`
    if needed.
    """
    x = np.asfarray(x)
    wing = _width2wing(width, x)
    signal = _pad_array(x, wing)
    if as_series:
        signal = pd.Series(signal)
    return x, wing, signal


def _width2wing(width, x, min_wing=3):
    """Convert a fractional or absolute width to integer half-width ("wing").
    """
    if 0 < width < 1:
        wing = int(math.ceil(len(x) * width * 0.5))
    elif width >= 2 and int(width) == width:
        wing = int(width // 2)
    else:
        raise ValueError("width must be either a fraction between 0 and 1 "
                         "or an integer greater than 1 (got %s)" % width)
    wing = max(wing, min_wing)
    wing = min(wing, len(x) - 1)
    assert wing >= 1, "Wing must be at least 1 (got %s)" % wing
    return wing


def _pad_array(x, wing):
    """Pad the edges of the input array with mirror copies."""
    return np.concatenate((x[wing-1::-1],
                           x,
                           x[:-wing-1:-1]))

def rolling_quantile(x, width, quantile):
    """Rolling quantile (0--1) with mirrored edges."""
    x, wing, signal = check_inputs(x, width)
    rolled = signal.rolling(2 * wing + 1, 2, center=True).quantile(quantile)
    return np.asfarray(rolled[wing:-wing])

#%%
def format_framesXrois(sDATA, roi_list, nframes_on, framerate, trace='raw', verbose=True, missing='drop'):

    # Format data: rows = frames, cols = rois
    raw_xdata = np.array(sDATA.sort_values(['trial', 'tsec']).groupby(['roi'])[trace].apply(np.array).tolist()).T
    
    roi_list = sorted(roi_list, key=natural_keys)
    Xdf = pd.DataFrame(raw_xdata, columns=roi_list)
    #decay_constant = 71./1000 # in sec -- this is what Romano et al. bioRxiv 2017 do for Fsmooth (decay_constant of indicator * 40)
    # vs. Dombeck et al. Neuron 2007 methods (15 sec +/- tpoint 8th percentile)
    
    window_size_sec = (nframes_on/framerate) * 4 # decay_constant * 40
    decay_frames = window_size_sec * framerate # decay_constant in frames
    window_size = int(round(decay_frames))
    quantile = 0.08
    
    Fsmooth = Xdf.apply(rolling_quantile, args=(window_size, quantile))
    Xdata_tmp = (Xdf - Fsmooth)
    Xdata = np.array(Xdata_tmp)
    
    
    # Get rid of "bad rois" that have np.nan on some of the trials:
    # NOTE:  This is not exactly the best way, but if the df/f trace is wild, np.nan is set for df value on that trial
    # Since this is done in traces/get_traces.py, for now, just deal with this by ignoring ROI
    bad_roi = None
    if missing == 'drop':
        ix, iv = np.where(np.isnan(Xdata))
        bad_roi = list(set(iv))
        if len(bad_roi) == 0:
            bad_roi = None

    if bad_roi is not None:
        Xdata = np.delete(Xdata, bad_roi, 1)
        roi_list = [r for ri,r in enumerate(roi_list) if ri not in bad_roi]

    tsecs = np.array(sDATA.sort_values(['trial', 'tsec']).groupby(['roi'])['tsec'].apply(np.array).tolist()).T
    if bad_roi is not None:
        tsecs = np.delete(tsecs, bad_roi, 1)

    # Get labels: # only need one col, since trial id same for all rois
    ylabels = np.array(sDATA.sort_values(['trial', 'tsec']).groupby(['roi'])['config'].apply(np.array).tolist()).T[:,0]
    groups = np.array(sDATA.sort_values(['trial']).groupby(['roi'])['trial'].apply(np.array).tolist()).T[:,0]

    if verbose:
        print "-------------------------------------------"
        print "Formatting summary:"
        print "-------------------------------------------"
        print "X:", Xdata.shape
        print "y (labels):", ylabels.shape
        print "N groupings of trials:", len(list(set(groups)))
        print "N samples: %i, N features: %i" % (Xdata.shape[0], Xdata.shape[1])
        print "-------------------------------------------"

    return Xdata, ylabels, groups, tsecs, roi_list

#%%
def format_roisXvalue(sDATA, run_info, value_type='meanstimdf', trace='raw'):

    # Make sure that we only get ROIs in provided list (we are dropping ROIs w/ np.nan dfs on any trials...)
    #sDATA = sDATA[sDATA['roi'].isin(roi_list)]
    stim_on_frame = run_info['stim_on_frame']
    nframes_on = run_info['nframes_on']
    ntrials_total = run_info['ntrials_total']
    nrois = len(run_info['roi_list'])

    trace = 'raw'
    rawtraces = np.vstack((sDATA.groupby(['roi', 'config', 'trial'])[trace].apply(np.array)).as_matrix())

    std_baseline_values = np.nanstd(rawtraces[:, 0:stim_on_frame], axis=1)
    mean_baseline_values = np.nanmean(rawtraces[:, 0:stim_on_frame], axis=1)
    mean_stim_on_values = np.nanmean(rawtraces[:, stim_on_frame:stim_on_frame+nframes_on], axis=1)

    #zscore_values_raw = np.array([meanval/stdval for (meanval, stdval) in zip(mean_stim_on_values, std_baseline_values)])
    if value_type == 'zscore':
        values_df = (mean_stim_on_values - mean_baseline_values ) / std_baseline_values
    else:
        values_df = mean_stim_on_values #- mean_baseline_values ) / std_baseline_values
    
    rois_by_value = np.reshape(values_df, (nrois, ntrials_total))
        
#    if bad_roi is not None:
#        rois_by_zscore = np.delete(rois_by_zscore, bad_roi, 0)

    return rois_by_value

#%% Preprocess data:
#%
def smooth_traces(trace, frac=0.002, missing='none'):
    '''
    lowess algo (from docs):

    Suppose the input data has N points. The algorithm works by estimating the
    smooth y_i by taking the frac*N closest points to (x_i,y_i) based on their
    x values and estimating y_i using a weighted linear regression. The weight
    for (x_j,y_j) is tricube function applied to abs(x_i-x_j).

    Set 'missing' to 'drop' to ignore NaNs. Set 'return_sorted' to False to
    return array of the same sequence as input (doesn't omit NaNs)
    '''
    xvals = np.arange(len(trace))
    filtered = lowess(trace, xvals, is_sorted=True, frac=frac, it=0, missing=missing, return_sorted=False)
    if len(filtered.shape) > 1:
        return filtered[:, 1]
    else:
        return filtered
   
def test_smoothing_fractions(ridx, Xdata, ylabels, missing='drop',
                             nframes_per_trial=358, ntrials_per_cond=10,
                             condlabel=0, fmin=0.0005, fmax=0.05):

    trace_test = Xdata[:, ridx:ridx+1]
    #print trace_test.shape

#    trace_test_filt = np.apply_along_axis(smooth_traces, 0, trace_test, frac=0.0003)
#    print trace_test_filt.shape

    # Plot the same trial on top:
    ixs = np.where(ylabels==condlabel)[0]
    assert len(ixs) > 0, "No frames found for condition with label: %s" % str(condlabel)

    #frac_range = np.linspace(0.0001, 0.005, num=8)
    frac_range = np.linspace(fmin, fmax, num=8)
    fig, axes = pl.subplots(2,4, figsize=(12,8)) #pl.figure()
    #ax = axes.flat()
    for i, ax, in enumerate(axes.flat):
        trace_test_filt = np.apply_along_axis(smooth_traces, 0, trace_test, frac=frac_range[i], missing=missing)
        #print trace_test_filt.shape
        tmat = []
        for tidx in range(ntrials_per_cond):
            fstart = tidx*nframes_per_trial
            #pl.plot(xrange(nframes_per_trial), trace_test[ixs[fstart]:ixs[fstart]+nframes_per_trial, 0])
            tr = trace_test_filt[ixs[fstart]:ixs[fstart]+nframes_per_trial, 0]
            ax.plot(xrange(nframes_per_trial), trace_test_filt[ixs[fstart]:ixs[fstart]+nframes_per_trial, 0], 'k', linewidth=0.5)
            tmat.append(tr)
        ax.plot(xrange(nframes_per_trial), np.nanmean(np.array(tmat), axis=0), 'r', linewidth=1)
        ax.set_title('%.04f' % frac_range[i])
#
        
#% Get mean trace for each condition:
def get_mean_cond_traces(ridx, Xdata, ylabels, tsecs, nframes_per_trial):
    '''For each ROI, get average trace for each condition.
    '''
    if isinstance(ylabels[0], str):
        conditions = sorted(list(set(ylabels)), key=natural_keys)
    else:
        conditions = sorted(list(set(ylabels)))

    mean_cond_traces = []
    mean_cond_tsecs = []
    for cond in conditions:
        ixs = np.where(ylabels==cond)                                          # Get sample indices for current condition
        curr_trace = np.squeeze(Xdata[ixs, ridx])                              # Grab subset of sample data 
        ntrials_in_cond = curr_trace.shape[0]/nframes_per_trial                # Identify the number of trials for current condition
        
        # Reshape both traces and corresponding time stamps:  
        # Shape (ntrials, nframes) to get average:
        curr_tracemat = np.reshape(curr_trace, (ntrials_in_cond, nframes_per_trial))
        curr_tsecs = np.reshape(np.squeeze(tsecs[ixs,ridx]), (ntrials_in_cond, nframes_per_trial))

        mean_ctrace = np.mean(curr_tracemat, axis=0)
        mean_cond_traces.append(mean_ctrace)
        mean_tsecs = np.mean(curr_tsecs, axis=0)
        mean_cond_tsecs.append(mean_tsecs)

    mean_cond_traces = np.array(mean_cond_traces)
    mean_cond_tsecs = np.array(mean_tsecs)
    #print mean_cond_traces.shape
    return mean_cond_traces, mean_cond_tsecs


def get_xcond_dfs(roi_list, X, y, tsecs, run_info):
    nconds = len(run_info['condition_list'])
    averages_list = []
    normed_list = []
    for ridx, roi in enumerate(sorted(roi_list, key=natural_keys)):
        mean_cond_traces, mean_tsecs = get_mean_cond_traces(ridx, X, y, tsecs, run_info['nframes_per_trial']) #get_mean_cond_traces(ridx, X, y)
        xcond_mean = np.mean(mean_cond_traces, axis=0)
        normed = mean_cond_traces - xcond_mean

        averages_list.append(pd.DataFrame(data=np.reshape(mean_cond_traces, (nconds*run_info['nframes_per_trial'],)),
                                        columns = [roi],
                                        index=np.array(range(nconds*run_info['nframes_per_trial']))
                                        ))

        normed_list.append(pd.DataFrame(data=np.reshape(normed, (nconds*run_info['nframes_per_trial'],)),
                                        columns = [roi],
                                         index=np.array(range(nconds*run_info['nframes_per_trial']))
                                        ))
    return averages_list, normed_list


#%% Visualization:

# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import imutils

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def uint16_to_RGB(img):
    im = img.astype(np.float64)/img.max()
    im = 255 * im
    im = im.astype(np.uint8)
    rgb = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    return rgb

def sort_rois_2D(traceid_dir):

    run_dir = traceid_dir.split('/traces')[0]
    acquisition_dir = os.path.split(run_dir)[0]; acquisition = os.path.split(acquisition_dir)[1]
    session_dir = os.path.split(acquisition_dir)[0]; session = os.path.split(session_dir)[1]
    animalid = os.path.split(os.path.split(session_dir)[0])[1]
    rootdir = session_dir.split('/%s' % animalid)[0]

    # Load formatted mask file:
    mask_fpath = os.path.join(traceid_dir, 'MASKS.hdf5')
    maskfile =h5py.File(mask_fpath, 'r')

    # Get REFERENCE file (file from which masks were made):
    mask_src = maskfile.attrs['source_file']
    if rootdir not in mask_src:
        mask_src = replace_root(mask_src, rootdir, animalid, session)
    tmp_msrc = h5py.File(mask_src, 'r')
    ref_file = tmp_msrc.keys()[0]
    tmp_msrc.close()

    # Load masks and reshape to 2D:
    if ref_file not in maskfile.keys():
        ref_file = maskfile.keys()[0]
    masks = np.array(maskfile[ref_file]['Slice01']['maskarray'])
    dims = maskfile[ref_file]['Slice01']['zproj'].shape
    masks_r = np.reshape(masks, (dims[0], dims[1], masks.shape[-1]))
    print "Masks: (%i, %i), % rois." % (masks_r.shape[0], masks_r.shape[1], masks_r.shape[-1])

    # Load zprojection image:
    zproj = np.array(maskfile[ref_file]['Slice01']['zproj'])


    # Cycle through all ROIs and get their edges
    # (note:  tried doing this on sum of all ROIs, but fails if ROIs are overlapping at all)
    cnts = []
    for ridx in range(masks_r.shape[-1]):
        im = masks_r[:,:,ridx]
        im[im>0] = 1
        im[im==0] = np.nan #1
        im = im.astype('uint8')
        edged = cv2.Canny(im, 0, 0.9)
        tmp_cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        tmp_cnts = tmp_cnts[0] if imutils.is_cv2() else tmp_cnts[1]
        cnts.append((ridx, tmp_cnts[0]))
    print "Created %i contours for rois." % len(cnts)

    # Sort ROIs b y x,y position:
    sorted_cnts =  sorted(cnts, key=lambda ctr: (cv2.boundingRect(ctr[1])[1] + cv2.boundingRect(ctr[1])[0]) * zproj.shape[1] )
    cnts = [c[1] for c in sorted_cnts]
    sorted_rids = [c[0] for c in sorted_cnts]

    return sorted_rids, cnts, zproj

#
def plot_roi_contours(zproj, sorted_rids, cnts):

    # Create ZPROJ img to draw on:
    refRGB = uint16_to_RGB(zproj)

    # Use some color map to indicate distance from upper-left corner:
    sorted_colors = sns.color_palette("Spectral", len(sorted_rids)) #masks.shape[-1])

    fig, ax = pl.subplots(1)
#    p2, p98 = np.percentile(refRGB, (1, 99))
#    img_rescale = exposure.rescale_intensity(refRGB, in_range=(p2, p98))
    im_adapthist = exposure.equalize_adapthist(refRGB, clip_limit=0.03)
    im_adapthist *= 256
    im_adapthist= im_adapthist.astype('uint8')
    ax.imshow(im_adapthist) #pl.figure(); pl.imshow(refRGB) # cmap='gray')

    refObj = None
    orig = im_adapthist.copy()
    distances = []
    # loop over the contours individually
    for cidx, (rid, cnt) in enumerate(zip(sorted_rids, cnts)):

        # compute the rotated bounding box of the contour
        box = cv2.minAreaRect(cnt)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")

    	# order the points in the contour such that they appear
    	# in top-left, top-right, bottom-right, and bottom-left order
        box = perspective.order_points(box)

    	# compute the center of the bounding box
        cX = np.average(box[:, 0])
        cY = np.average(box[:, 1])

        # if this is the first contour we are examining (i.e.,
        # the left-most contour), we presume this is the
        # reference object
        if refObj is None:
            # unpack the ordered bounding box, then compute the
            # midpoint between the top-left and top-right points,
            # followed by the midpoint between the top-right and
            # bottom-right
            (tl, tr, br, bl) = box
            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)

            # compute the Euclidean distance between the midpoints,
            # then construct the reference object
            D = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
            refObj = (cidx, (cX, cY)) #(box, (cX, cY), D) # / args["width"])
            continue

        # draw the contours on the image
        #orig = refRGB.copy()
        col255 = tuple([cval*255 for cval in sorted_colors[cidx]])
        cv2.drawContours(orig, [box.astype("int")], -1, col255, 2)
        cv2.putText(orig, str(rid+1), cv2.boundingRect(cnt)[:2], cv2.FONT_HERSHEY_COMPLEX, .5, [0])
        ax.imshow(orig)

        # stack the reference coordinates and the object coordinates
        # to include the object center
        refCoords = refObj[1] #np.vstack([refObj[0], refObj[1]])
        objCoords = (cX, cY) #np.vstack([box, (cX, cY)])

        D = dist.euclidean((cX, cY), (refCoords[0], refCoords[1])) #/ refObj[2]
        distances.append(D)

    pl.axis('off')
    
    
#
def psth_from_full_trace(roi, tracevec, mean_tsecs, nr, nc,
                                  color_codes=None, orientations=None,
                                  stim_on_frame=None, nframes_on=None,
                                  plot_legend=True, plot_average=True, as_percent=False,
                                  roi_psth_dir='/tmp', save_and_close=True):

    '''Pasre a full time-series (of a given run) and plot as stimulus-aligned
    PSTH for a given ROI.
    '''

    pl.figure()
    traces = np.reshape(tracevec, (nr, nc))

    if as_percent:
        multiplier = 100
        units_str = ' (%)'
    else:
        multiplier = 1
        units_str = ''

    if color_codes is None:
        color_codes = sns.color_palette("Greys_r", nr*2)
        color_codes = color_codes[0::2]
    if orientations is None:
        orientations = np.arange(0, nr)

    for c in range(traces.shape[0]):
        pl.plot(mean_tsecs, traces[c,:] * multiplier, c=color_codes[c], linewidth=2, label=orientations[c])

    if plot_average:
        pl.plot(mean_tsecs, np.mean(traces, axis=0)*multiplier, c='r', linewidth=2.0)
    sns.despine(offset=4, trim=True)

    if stim_on_frame is not None and nframes_on is not None:
        stimbar_loc = traces.min() - (0.1*traces.min()) #8.0

        stimon_frames = mean_tsecs[stim_on_frame:stim_on_frame + nframes_on]
        pl.plot(stimon_frames, stimbar_loc*np.ones(stimon_frames.shape), 'g')

    pl.xlabel('tsec')
    pl.ylabel('mean df/f%s' % units_str)
    pl.title(roi)

    if plot_legend:
        pl.legend(orientations)

    if save_and_close:
        pl.savefig(os.path.join(roi_psth_dir, '%s_psth_mean.png' % roi))
        pl.close()
