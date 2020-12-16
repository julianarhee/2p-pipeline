#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 21:09:57 2019

@author: julianarhee
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 12:31:35 2018

@author: juliana
"""
import warnings
warnings.filterwarnings("ignore")
import matplotlib as mpl
mpl.use('agg')
import h5py
import os
import json
import cv2
import time
import math
import random
import itertools
import copy
import traceback
import shutil

import scipy.io
import optparse
import cPickle as pkl
import pandas as pd
import numpy as np
import pylab as pl
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
from collections import Counter

from pipeline.python.traces.trial_alignment import aggregate_experiment_runs 
from pipeline.python.utils import natural_keys, replace_root, print_elapsed_time

import pipeline.python.traces.combine_runs as cb
import pipeline.python.paradigm.align_acquisition_events as acq
import pipeline.python.visualization.plot_psths_from_dataframe as vis
from pipeline.python.traces.utils import load_TID
from pipeline.python.rois.utils import load_roi_masks, get_roiid_from_traceid, load_roi_coords
from mpl_toolkits.axes_grid1 import make_axes_locatable
#from pipeline.python.traces.utils import get_frame_info

from pipeline.python.retinotopy import utils as retinotools
from pipeline.python.retinotopy import fit_2d_rfs as fitrf
#from pipeline.python.classifications.analyze_retino_structure import evaluate_rfs as func_evaluate_rfs
from pipeline.python.classifications import test_responsivity as resp
#from pipeline.python.classifications import responsivity_stats as respstats

#from pipeline.python.classifications import bootstrap_fit_tuning_curves as osi
from pipeline.python.classifications import bootstrap_osi as osi
from pipeline.python.utils import label_figure, natural_keys#, get_frame_info, check_counts_per_condition
from pipeline.python import utils as util
from pipeline.python.classifications.bootstrap_roc import bootstrap_roc_func

import glob
import os
import json
import re
import pandas as pd

#%%

# #############################################################################
# General functions
# #############################################################################



def get_stats_desc(traceid='traces001', trace_type='corrected', response_type='dff',
                   responsive_test=None, responsive_thr=0.05, n_stds=2.35):

    if responsive_test is None:
        dtype_str = '-'.join([traceid, trace_type, response_type, 'all'])
    elif responsive_test == 'nstd':
        dtype_str = '-'.join([traceid, trace_type, response_type, responsive_test, '%.2f' % n_stds, 'thr-%.2f' % responsive_thr])
    else:
        dtype_str = '-'.join([traceid, trace_type, response_type, responsive_test, 'thr-%.2f' % responsive_thr])
    
    stats_desc = 'stats-%s' % dtype_str
    return stats_desc

def create_stats_dir(animalid, session, fov, traceid='traces001', 
                     trace_type='corrected', response_type='dff', 
                     responsive_test=None, responsive_thr=0.05, n_stds=2.5,
                     rootdir='/n/coxfs01/2p-data'):

    # Create output dirs:    
    output_dir = os.path.join(rootdir, animalid, session, fov, 'summaries')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    stats_desc = get_stats_desc(traceid=traceid, trace_type=trace_type,
                                response_type=response_type, 
                                responsive_test=responsive_test, responsive_thr=responsive_thr,
                                n_stds=n_stds)
    
    statsdir = os.path.join(output_dir, stats_desc)
    if not os.path.exists(statsdir):
        os.makedirs(statsdir)

#    statsfigdir = os.path.join(statsdir, 'figures')
#    if not os.path.exists(statsfigdir):
#        os.makedirs(statsfigdir)
        
    return statsdir, stats_desc


#%%

class Struct():
    pass

def get_roi_id(animalid, session, fov, traceid, run_name='', rootdir='/n/coxfs01/2p-data'):
    extraction_type = re.sub('[0-9]+', '', traceid) if 'traces' in traceid else 'retino_analysis'
    #extraction_num = int(re.findall(r'\d+', traceid)[0])
    
    if 'retino' in run_name and extraction_type=='traces': #using traceid in reference to other run types
        traceid_info_fpath = glob.glob(os.path.join(rootdir, animalid, session, fov, '*', \
                                         'traces', 'traceids_*.json'))[0] # % traceid, ))
    else:
        traceid_info_fpath = glob.glob(os.path.join(rootdir, animalid, session, fov, '*%s*' % run_name, \
                                             '%s' % extraction_type, '*.json'))[0] # % traceid, ))
    with open(traceid_info_fpath, 'r') as f:
        traceids = json.load(f)
        
    roi_id = traceids[traceid]['PARAMS']['roi_id']
    
    if 'retino' in run_name: #extraction_type == 'retino_analysis':
        extraction_type = 'retino_analysis'
        try:
            retinoid_info_fpath = glob.glob(os.path.join(rootdir, animalid, session, fov, '*%s*' % run_name, \
                                                 '%s' % extraction_type, '*.json'))[0] # % traceid, ))
            with open(retinoid_info_fpath, 'r') as f:
                retino_ids = json.load(f)
            found_ids = [t for t, tinfo in retino_ids.items() if 'roi_id' in tinfo['PARAMS'].keys()\
                         and tinfo['PARAMS']['roi_id'] == roi_id]
            if len(found_ids) > 1:
                for fi, fid in enumerate(found_ids):
                    print fi, fid
                sel = input("More than 1 retino analysis using [%s]. Select IDX to use: " % roi_id)
                traceid = found_ids[int(sel)]
            else:
                traceid = found_ids[0]
        except Exception as e:
            print("--- %s: not processed" % run_name)
            return None, None
        
    return roi_id, traceid

def load_anatomical(animalid, session, fov, channel_num=2, verbose=False, 
                    rootdir='/n/coxfs01/2p-data'):
    anatomical = None
    fov_dir = os.path.join(rootdir, animalid, session, fov)
    anatomical_dirs = glob.glob(os.path.join(fov_dir, 'anatomical'))
    try:
        if verbose:
            assert len(anatomical_dirs) > 0, "---> (warning): no anatomicals for (%s|%s|%s)" % (animalid, session, fov)
        anatomical_dir = anatomical_dirs[0]
        if verbose:
            print("... found %i anatomical runs." % len(anatomical_dirs))
        anatomical_imgs = glob.glob(os.path.join(anatomical_dir, 'processed',
                                                 'processed*', 'mcorrected_*_mean_deinterleaved',
                                                 'Channel%02d' % channel_num, 'File*', '*.tif'))
        if verbose:
            assert len(anatomical_imgs) > 0, "... No processed anatomicals found!"
        images=[]
        for fpath in anatomical_imgs:
            im = tf.imread(fpath)
            images.append(im)
        anatomical = np.array(images).sum(axis=0)
    except Exception as e:
        print("... no anat")
        
    return anatomical
        
# #############################################################################
# Data processing and formatting
# #############################################################################

def process_traces(raw_traces, labels, response_type='dff', nframes_post_onset=None):

#    stim_on_frame = labels['stim_on_frame'].unique()[0]
#    tmp_df = []
#    for k, g in labels.groupby(['trial']):
#        tmat = raw_traces.loc[g.index]
#        bas_mean = np.nanmean(tmat[0:stim_on_frame], axis=0)
#        tmat_df = (tmat - bas_mean) / bas_mean
#        tmp_df.append(tmat_df)
#    df_traces = pd.concat(tmp_df, axis=0)
#    

    stim_on_frame = labels['stim_on_frame'].unique()[0]
    tmp_df = []
    for k, g in labels.groupby(['trial']):
        tmat = raw_traces.loc[g.index]
        bas_mean = np.nanmean(tmat[0:stim_on_frame], axis=0)
        if response_type == 'dff':
            tmat_df = (tmat - bas_mean) / bas_mean
        elif response_type == 'zscore':
            bas_std = np.nanstd(tmat[0:stim_on_frame], axis=0)
            tmat_df = (tmat - bas_mean) / bas_std
        tmp_df.append(tmat_df)
    processed_traces = pd.concat(tmp_df, axis=0)
    
    return processed_traces

def get_trial_metrics(raw_traces, labels, response_type='mean', nframes_post_onset=None):
    '''
    Parses raw trace dataframes into single metric assign to each trial.
    
    raw_traces:
        (dataframe) indices = frames, columns = rois. Assumes "raw" (i.e., only corrected) for
        calculating anything besides "dff" (Provide 'corrected' to use zscore or other).
        
    nframes_post_onset:
        (int) N frames to include as "trial" response (default uses nframes_on*2, if None provided)

    '''
    # Get stim onset frame: 
    stim_on_frame = labels['stim_on_frame'].unique()
    assert len(stim_on_frame) == 1, "---[stim_on_frame]: More than 1 stim onset found: %s" % str(stim_on_frame)
    stim_on_frame = stim_on_frame[0]
    
    # Get n frames stimulus on:
    nframes_on = labels['nframes_on'].unique()
    assert len(nframes_on) == 1, "---[nframes_on]: More than 1 stim dur found: %s" % str(nframes_on)
    nframes_on = nframes_on[0]
        
    if nframes_post_onset is None:
        nframes_post_onset = nframes_on #*2
        
    stats_list = []
    for trial, tmat in labels.groupby(['trial']):

        # Get traces using current trial's indices: divide by std of baseline
        curr_traces = raw_traces.iloc[tmat.index] 
        bas_std = curr_traces.iloc[0:stim_on_frame].std(axis=0)
        bas_mean = curr_traces.iloc[0:stim_on_frame].mean(axis=0)
        
        # Also get zscore (single value) for each trial:
        stim_mean = curr_traces.iloc[stim_on_frame:stim_on_frame+nframes_post_onset].mean(axis=0)
        if response_type == 'zscore':
            #stats_list.append((stim_mean-bas_mean)/bas_std)
            stats_list.append((stim_mean)/bas_std)
        elif response_type == 'snr':
            stats_list.append(stim_mean/bas_mean)
        elif response_type == 'mean':
            stats_list.append(stim_mean)
        elif response_type == 'dff':
            #stats_list.append((stim_mean-bas_mean) / bas_mean)
            stats_list.append((stim_mean) / bas_mean)
                
    trialstats =  pd.concat(stats_list, axis=1).T # cols=rois, rows = trials
    
    return trialstats

def get_trials_by_cond(labels):
    # Get single value for each trial and sort by config:
    trials_by_cond = dict()
    for k, g in labels.groupby(['config']):
        trials_by_cond[k] = sorted([int(tr[5:])-1 for tr in g['trial'].unique()])
    return trials_by_cond

def group_trial_values_by_cond(trialstats, labels):
    trials_by_cond = get_trials_by_cond(labels)

    resp_by_cond = dict()
    for cfg, trial_ixs in trials_by_cond.items():
        resp_by_cond[cfg] = trialstats.iloc[trial_ixs]  # For each config, array of size ntrials x nrois

    trialstats_by_cond = pd.DataFrame([resp_by_cond[cfg].mean(axis=0) \
                                            for cfg in sorted(resp_by_cond.keys(), key=util.natural_keys)]) # nconfigs x nrois
         
    return trialstats_by_cond



#%%
# #############################################################################
# RETINOBAR FUNCTIONS
# #############################################################################

def get_retino_analysis(animalid, session, fov, run='retino_run1', rois=None, rootdir='/n/coxfs01/2p-data'):
    
    run_dir = os.path.join(rootdir, animalid, session, fov, run)
    analysis_info_fpath = glob.glob(os.path.join(run_dir, 'retino_analysis', 'analysisids*.json'))[0]
    with open(analysis_info_fpath, 'r') as f:
        ainfo = json.load(f)
        
    # Find analysis id using roi type specified:
    if rois == 'pixels':
        found_ids = sorted([a for a, info in ainfo.items() if info['PARAMS']['roi_type']==rois], key=util.natural_keys)
    else:
        found_ids = sorted([a for a, info in ainfo.items() if 'roi_id' in info['PARAMS'].keys()\
                            and info['PARAMS']['roi_id'] == rois], key=util.natural_keys)
    assert len(found_ids) > 0, "No analysis ids found of type: %s (run dir:\n%s)" % (rois, run_dir)
    if len(found_ids) > 1:
        for fi, fr in enumerate(found_ids):
            print fi, fr
        sel = input("Select ID of analysis to use: ")
        analysis_id = found_ids[int(sel)]
    else:
        analysis_id = found_ids[0]
        
    data_fpath = glob.glob(os.path.join(run_dir, 'retino_analysis', '%s*' % analysis_id, 'traces', 'extracted_traces.h5'))[0]
    
    return data_fpath

def get_retino_stats(expdata, responsive_thr=0.01):
    magratios, phases, traces, trials_by_cond = do_retino_analysis_on_raw(expdata)
    roi_list = [r for r in magratios.index.tolist() if any(magratios.loc[r] > responsive_thr)]
    nrois_total = len(magratios.index.tolist())
    rstats = {'magratios': magratios, 'phases': phases, 'traces': traces}
    
    return rstats, roi_list, nrois_total, trials_by_cond
        

def do_retino_analysis_on_raw(expdata):
    n_frames = expdata.info['stimulus']['nframes']
    n_files = expdata.info['ntiffs']
    fr = expdata.info['stimulus']['frame_rate']
    stimfreq = expdata.info['stimulus']['stimfreq']

    # label frequency bins
    freqs = np.fft.fftfreq(n_frames, float(1/fr))
    sorted_freq_ixs = np.argsort(freqs)
    freqs=freqs[sorted_freq_ixs]
    #print(freqs)

    # exclude DC offset from data
    freqs=freqs[int(np.round(n_frames/2.))+1:]

    # Identify freq idx:
    stim_freq_ix=np.argmin(np.absolute(freqs-stimfreq))#find out index of stimulation freq
    top_freq_ix=np.where(freqs>1)[0][0]#find out index of 1Hz, to cut-off zoomed out plot
    print("Target freq: %.3f Hz" % (freqs[stim_freq_ix]))
    

    trials_by_cond = expdata.info['trials']
    trial_nums = np.array([v for k,v in trials_by_cond.items()])
    trial_nums = sorted(trial_nums.flatten())

    nframes_total, nrois = expdata.traces.shape
    magratios=[]
    phases=[]
    conds=[]
    traces={}
    for curr_cond in trials_by_cond.keys():
        avg_traces = []
        for rid in expdata.traces.columns:
            tracemat = pd.DataFrame(np.reshape(expdata.traces[rid], (n_frames, n_files), order='F'),\
                                    columns=trial_nums)
            avg = tracemat[trials_by_cond[curr_cond]].mean(axis=1)
            avg_traces.append(avg)
        avg_traces = pd.DataFrame(np.array(avg_traces).T, columns=expdata.traces.columns)
        traces[curr_cond] = avg_traces

        magratio_array, phase_array = do_fft_analysis(avg_traces, sorted_freq_ixs, stim_freq_ix, n_frames)

        magratios.append(magratio_array)
        phases.append(phase_array)
        conds.append(curr_cond)
        
    magratios = pd.DataFrame(np.array(magratios).T, columns=conds)
    phases = pd.DataFrame(np.array(phases).T, columns=conds)
    
    return magratios, phases, traces, trials_by_cond


def do_fft_analysis(avg_traces, sorted_freq_ixs, stim_freq_ix, n_frames):
    fft_results = np.fft.fft(avg_traces, axis=0) #avg_traces.apply(np.fft.fft, axis=1)

    # get phase and magnitude
    mag_data = abs(fft_results)
    phase_data = np.angle(fft_results)

    # sort mag and phase by freq idx:
    mag_data = mag_data[sorted_freq_ixs]
    phase_data = phase_data[sorted_freq_ixs]

    # exclude DC offset from data
    mag_data = mag_data[int(np.round(n_frames/2.))+1:, :]
    phase_data = phase_data[int(np.round(n_frames/2.))+1:, :]

    #unpack values from frequency analysis
    mag_array = mag_data[stim_freq_ix, :]
    phase_array = phase_data[stim_freq_ix, :]

    #get magnitude ratio
    tmp = np.copy(mag_data)
    #tmp = np.delete(tmp,freq_idx,0)
    nontarget_mag_array=np.sum(tmp,0)
    magratio_array=mag_array/nontarget_mag_array

    return magratio_array, phase_array


# #############################################################################
# RECEPTIVE FIELD EXPERIMENT functions
# #############################################################################

def get_receptive_field_fits(animalid, session, fov, response_type='dff', 
                             reload_data=False, do_spherical_correction=False,
                             run='combined_rfs*_static', traceid='traces001', 
                             pretty_plots=False, create_new=False,
                             rootdir='/n/coxfs01/2p-data'):
    #assert 'rfs' in S.experiments['rfs'].name, "This is not a RF experiment object! %s" % exp.name
    rfits = None
#    fov_dir = os.path.join(rootdir, animalid, session, fov)
    rfdir, fit_desc = fitrf.create_rf_dir(animalid, session, fov, run, traceid=traceid, 
                                          do_spherical_correction=do_spherical_correction,
                                          response_type=response_type, rootdir=rootdir)
    rfs_fpath = os.path.join(rfdir, 'fit_results.pkl')
    fov_fpath = os.path.join(rfdir, 'fov_info.pkl')
    
    do_fits = create_new #False
    
    if create_new is False:
        try:
            print("... loading RF fits (response-type: %s)" % response_type)
            with open(rfs_fpath, 'rb') as f:
                rfits = pkl.load(f)
            with open(fov_fpath, 'rb') as f:
                fovinfo = pkl.load(f)
        except Exception as e:
            do_fits = True
    
    if do_fits:
        print("... specified RF fit method not found, running now: %s" % fit_desc)
        fit_results, fit_params = fitrf.fit_2d_receptive_fields(animalid, session, fov, run, traceid,
                                                        reload_data=reload_data,
                                                        make_pretty_plots=pretty_plots,
                                                        create_new=create_new,
                                                        trace_type='corrected', 
                                                        response_type=response_type,
                                                        do_spherical_correction=do_spherical_correction,
                                                        rootdir=rootdir)
        #    except Exception as e:
#        print("*** NO receptive field fits found: %s ***" % '|'.join([animalid, session, fov, run, traceid]))
#        traceback.print_exc()
        
    return fit_results, fit_params #rfits, fovinfo

#

# #############################################################################
# Generic "event"-based experiments
# #############################################################################
def get_responsive_cells(animalid, session, fov, run=None, traceid='traces001',
                         response_type='dff',create_new=False, n_processes=1,
                         responsive_test='ROC', responsive_thr=0.05, n_stds=2.5,
                         rootdir='/n/coxfs01/2p-data'):
        
    roi_list=None; nrois_total=None;
    rname = run if 'combined' in run else 'combined_%s_' % run

    traceid_dir =  glob.glob(os.path.join(rootdir, animalid, session, 
                                    fov, '%s*' % rname, 'traces', '%s*' % traceid))[0]        
    stat_dir = os.path.join(traceid_dir, 'summary_stats', responsive_test)
    if not os.path.exists(stat_dir):
        os.makedirs(stat_dir) 
    #assert os.path.exists(stats_dir), "Stats dir does not exist: %s" % stats_dir
    #results_str = '' % responsive_thr if responsive_test=='nstds' else ''
    # move old dir
    if create_new and (('gratings' in run) or ('blobs' in run)):
        print("@@@ running anew, might take awhile (%s|%s|%s) @@@" % (animalid, session, fov))
        try:
            if responsive_test=='ROC':
                print("DOING BOOT - run: %s" % run) 
                bootstrap_roc_func(animalid, session, fov, traceid, run, 
                            trace_type='corrected', rootdir=rootdir,
                            n_processes=n_processes, plot_rois=True, n_iters=1000)
            elif responsive_test=='nstds':
                fdf = calculate_nframes_above_nstds(animalid, session, fov, 
                            run=run, traceid=traceid, n_stds=n_stds, 
                            #response_type=response_type, 
                            n_processes=n_processes, rootdir=rootdir, 
                            create_new=True)
            print('@@@@@@ finished responsivity test (%s|%s|%s) @@@@@@' % (animalid, session, fov))

        except Exception as e:
            traceback.print_exc()
            print("JK ERROR")
            return None, None 

    #if len(stats_fpath)==0:
    #    print("-- using old stats")
    #    stats_dir = os.path.join(traceid_dir, 'summary_stats', '_%s' % responsive_test)
    #    stats_fpath = glob.glob(os.path.join(stats_dir, '*results*.pkl'))

    if responsive_test=='nstds':
        stats_fpath = glob.glob(os.path.join(stat_dir, 
                            '%s-%.2f_result*.pkl' % (responsive_test, n_stds)))
    else:
        stats_fpath = glob.glob(os.path.join(stat_dir, 'roc_result*.pkl'))



    try:
        #stats_fpath = glob.glob(os.path.join(stats_dir, '*results*.pkl'))
        #assert len(stats_fpath) == 1, "Stats results paths: %s" % str(stats_fpath)
        with open(stats_fpath[0], 'rb') as f:
            print("... loading stats")
            rstats = pkl.load(f)
        # print("...loaded")        
        if responsive_test == 'ROC':
            roi_list = [r for r, res in rstats.items() if res['pval'] < responsive_thr]
            nrois_total = len(rstats.keys())
        elif responsive_test == 'nstds':
            assert n_stds == rstats['nstds'], "... incorrect nstds, need to recalculate"
            #print rstats
            roi_list = [r for r in rstats['nframes_above'].columns \
                            if any(rstats['nframes_above'][r] > responsive_thr)]
            nrois_total = rstats['nframes_above'].shape[-1]
    except Exception as e:
        print(e)
        traceback.print_exc()

    print("... %i of %i cells responsive" % (len(roi_list), nrois_total))
 
    return roi_list, nrois_total
   
def calculate_nframes_above_nstds(animalid, session, fov, run=None, traceid='traces001',
                         #response_type='dff', 
                        n_stds=2.5, create_new=False,
                         n_processes=1, rootdir='/n/coxfs01/2p-data'):

    if 'combined' in run:
        rname = run
    else:
        rname = 'combined_%s' % run

    traceid_dir =  glob.glob(os.path.join(rootdir, animalid, session, 
                                    fov, '%s*' % rname, 'traces', '%s*' % traceid))[0]        
    stat_dir = os.path.join(traceid_dir, 'summary_stats', 'nstds')
    if not os.path.exists(stat_dir):
        os.makedirs(stat_dir) 
    results_fpath = os.path.join(stat_dir, 'nstds-%.2f_results.pkl' % n_stds)
    
    calculate_frames = False
    if  os.path.exists(results_fpath) and create_new is False:
        try:
            with open(results_fpath, 'rb') as f:
                results = pkl.load(f)
            assert results['nstds'] == n_stds, "... different nstds requested. Re-calculating"
            framesdf = results['nframes_above']            
        except Exception as e:
            calculate_frames = True
    else:
        calculate_frames = True
    
    if calculate_frames:
        print("... Testing responsive (n_stds=%.2f)" % n_stds)
        # Load data
        soma_fpath = glob.glob(os.path.join(traceid_dir, 
                                    'data_arrays', 'np_subtracted.npz'))[0]
        traces, labels, sdf, run_info = util.load_dataset(soma_fpath, 
                                            trace_type='corrected', #response_type, 
                                            add_offset=True, 
                                            make_equal=False) #make_equal)
        #self.load(trace_type=trace_type, add_offset=add_offset)
        ncells_total = traces.shape[-1]
        
        # Calculate N frames 
        print("... Traces: %s, Labels: %s" % (str(traces.shape), str(labels.shape)))
        framesdf = pd.concat([resp.find_n_responsive_frames(traces[roi], labels, 
                                n_stds=n_stds) for roi in range(ncells_total)], axis=1)
        results = {'nframes_above': framesdf,
                   'nstds': n_stds}
        # Save    
        with open(results_fpath, 'wb') as f:
            pkl.dump(results, f, protocol=pkl.HIGHEST_PROTOCOL)
        print("... Saved: %s" % os.path.split(results_fpath)[-1])
 
    return framesdf

 
def get_roi_stats(animalid, session, fov, exp_name=None, traceid='traces001', 
                  trace_type='corrected', response_type='dff',
                  responsive_test='ROC', responsive_thr=0.05, n_stds=2.5,
                  rootdir='/n/coxfs01/2p-data', create_new=False, n_processes=1):
    rstats = None
    roi_list = None
    nrois_total = None

    # Load list of "visually responsive" cells    
    print("... loading ROI stats: %s" % responsive_test)
    curr_traceid_dir = glob.glob(os.path.join(rootdir, animalid, session, fov, \
                                              exp_name, 'traces', '%s*' % traceid))[0]
    try:
        curr_stats_dir = os.path.join(curr_traceid_dir, 'summary_stats', responsive_test)
        #results_str = '_thr-%.2f' % responsive_thr if responsive_test=='nstds' else ''
        stats_fpath = glob.glob(os.path.join(curr_stats_dir, '*results*.pkl'))
        #stats_fpath = glob.glob(os.path.join(curr_stats_dir, '*results_*thr-%.2f%*.pkl' % responsive_thr))
        with open(stats_fpath[0], 'rb') as f:
            rstats = pkl.load(f)
        roi_list, nrois_total = get_responsive_cells(animalid, session, fov, 
                                                run=exp_name, traceid=traceid,
                                                responsive_test=responsive_test, 
                                                responsive_thr=responsive_thr,
                                                n_stds=n_stds, 
                                                response_type=response_type, 
                                                rootdir=rootdir, 
                                                create_new=create_new, 
                                                n_processes=n_processes)
        #nrois_total = len(rstats.keys())
    except Exception as e:
        print e
        print("-- Unable to load stats: %s [%s]" % (responsive_test, exp_name))
        
    return rstats, roi_list, nrois_total
#
#def check_counts_per_condition(raw_traces, labels):
#    # Check trial counts / condn:
#    #print("Checking counts / condition...")
#    min_n = labels.groupby(['config'])['trial'].unique().apply(len).min()
#    conds_to_downsample = np.where( labels.groupby(['config'])['trial'].unique().apply(len) != min_n)[0]
#    if len(conds_to_downsample) > 0:
#        print("... adjusting for equal reps / condn...")
#        d_cfgs = [sorted(labels.groupby(['config']).groups.keys())[i]\
#                  for i in conds_to_downsample]
#        trials_kept = []
#        for cfg in labels['config'].unique():
#            c_trialnames = labels[labels['config']==cfg]['trial'].unique()
#            if cfg in d_cfgs:
#                #ntrials_remove = len(c_trialnames) - min_n
#                #print("... removing %i trials" % ntrials_remove)
#    
#                # In-place shuffle
#                random.shuffle(c_trialnames)
#    
#                # Take the first 2 elements of the now randomized array
#                trials_kept.extend(c_trialnames[0:min_n])
#            else:
#                trials_kept.extend(c_trialnames)
#    
#        ixs_kept = labels[labels['trial'].isin(trials_kept)].index.tolist()
#        
#        tmp_traces = raw_traces.loc[ixs_kept].reset_index(drop=True)
#        tmp_labels = labels[labels['trial'].isin(trials_kept)].reset_index(drop=True)
#        return tmp_traces, tmp_labels
#
#    else:
#        return raw_traces, labels
        
#%%

class Session():
    def __init__(self, animalid, session, fov, visual_area=None, state=None, 
                    get_anatomical=True, verbose=False, rootdir='/n/coxfs01/2p-data'):
        self.animalid = animalid
        self.session = session
        self.fov = fov
        
        if visual_area is None or state is None:
            with open(os.path.join(rootdir, animalid, 'sessionmeta.json'), 'r') as f:
                sessionmeta = json.load(f)
            skey = [k for k in sessionmeta.keys() if k.split('_')[0] == session \
                and k.split('_')[1] in fov][0]
            visual_area = sessionmeta[skey]['visual_area']
            state = sessionmeta[skey]['state']
            
        self.visual_area = visual_area
        self.state = state
        
        if verbose:
            print("Creating session object [%s|%s|%s]" % (animalid, session, fov))
        
        self.anatomical=None
        if get_anatomical:
            self.anatomical = load_anatomical(animalid, session, fov, rootdir=rootdir)

        self.rois = None
        self.traceid = None
        self.trace_type = None
        self.experiments = {}
        self.experiment_list = self.get_experiment_list(rootdir=rootdir)
        
        self.screen = retinotools.get_retino_info(animalid, session, 
                                                  fov=fov, rootdir=rootdir)
        #print("checking res...")
        #self.screen['resolution'] = [1920, 1080]
        #self.screen['linmaxH'] = 33.6615
        #self.screen['linminH'] = -33.6615
        #self.screen['linmaxW'] = 59.7782
        #self.screen['linminW'] = -59.7782

    def get_stimulus_coordinates(self, experiments=['blobs', 'gratings'],
                                    update_self=False):

        # Get stimulus positions - blobs and gratings only
        xpositions=[]; ypositions=[];
        for ex in experiments: #['blobs', 'gratings']:
            if ex not in self.experiment_list: #.keys():
                print("[%s|%s] not found: %s" % (self.animalid, self.session, ex))
                continue
            if ex not in self.experiments.keys():
                traceid = 'traces001' if self.traceid is None else self.traceid
                if 'grating' in ex:
                    exp = Gratings(self.animalid, self.session, 
                                    self.fov, traceid)
                elif 'blob' in ex:
                    # TODO:  ficx this, not implemented
                    exp = Objects(self.animalid, self.session, 
                                    self.fov, traceid)
                sdf = exp.get_stimuli()
            else:
                expdata = self.experiments[ex]
                sdf = expdata.data.sdf.copy()
            if ex == 'gratings' and len(sdf['size'].unique())>1: 
                # deal with FF stimuli
                sdf = sdf[sdf['size']<200]
                sdf.pop('luminance')
            #print(sdf.head(), sdf['xpos'].unique(), sdf['ypos'].unique())      

            curr_xpos = sdf['xpos'].unique()
            assert len(curr_xpos)==1, "[%s] more than 1 xpos found! %s" % (ex, str(curr_xpos))
            curr_ypos = sdf['ypos'].unique()
            assert len(curr_ypos)==1, "[%s] more than 1 ypos found! %s" % (ex, str(curr_ypos))
            xpositions.append(float(curr_xpos))
            ypositions.append(float(curr_ypos))
        
        if len(xpositions) > 0 and len(ypositions) > 0:
            xpos = list(set(xpositions))
            assert len(xpos)==1, "blobs and gratings have different XPOS: %s" % str(xpos)
            ypos = list(set(ypositions))
            assert len(ypos)==1, "blobs and gratings have different YPOS: %s" % str(ypos)
            xpos = float(xpos[0])
            ypos = float(ypos[0])
            #print("Stimuli presented at coords: (%i, %i)" % (xpos, ypos))
            
            return xpos, ypos
        else:
            return None, None
    

    def get_stimulus_sizes(self, size_tested = ['gratings', 'blobs']):        
        tested_exps = [e for e in self.experiment_list if e in size_tested]    
        stimsizes = {}
        for exp in tested_exps:
            if exp not in self.experiments.keys():
                E = Experiment(exp, self.animalid, self.session, self.fov)
                sdf = E.get_stimuli()
            else:
                sdf = self.experiments[exp].data.sdf
            if 'blobs' in exp:
                stimsizes[exp] = np.array([int(round(i, 1)) for i in sdf.dropna()['size'].unique()])
            else:
                stimsizes[exp] = sdf['size'].unique()
            #stimsizes[exp] = self.experiments[exp].data.sdf.dropna()['size'].unique()
        return stimsizes
 
    
    def save_session(self, rootdir='/n/coxfs01/2p-data'):
        outdir = os.path.join(rootdir, self.animalid, self.session, self.fov, 'summaries')
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        with open(os.path.join(outdir, 'sessiondata.pkl'), 'wb') as f:
            pkl.dump(self, f, protocol=pkl.HIGHEST_PROTOCOL)
            
    def load_masks(self, rois='', rootdir='/n/coxfs01/2p-data'):
        if rois == '':
            rois = self.rois
        print("... loading masks (roiid: %s)" % rois)
        masks, zimg = load_roi_masks(self.animalid, self.session, self.fov, 
                                        rois=rois, rootdir=rootdir)
        return masks, zimg
    
    
    def get_all_responsive_cells(self, traceid='traces001', response_type='dff',
                                 responsive_test='ROC', responsive_thr=0.05, n_stds=2.5, fit_thr=0.5,
                                 n_processes=1):
        '''Get all cells that were "responsive" to something
        '''
        all_rois = []
        experiment_list = self.get_experiment_list(traceid=traceid)
        for e in experiment_list:
            if 'rfs' in e:
                exp = ReceptiveFields(e, self.animalid, self.session, 
                                      self.fov, traceid=traceid)
                roi_list, _ = exp.get_responsive_cells(response_type=response_type,
                                                       responsive_test=responsive_test, 
                                                       fit_thr=fit_thr, create_new=create_new,
                                                       n_processes=n_processes)
            elif 'gratings' in e:
                exp = Gratings(self.animalid, self.session, self.fov, traceid=traceid)
                roi_list, _ = exp.get_responsive_cells(response_type=response_type,
                                                       responsive_test=responsive_test, 
                                                       responsive_thr=responsive_thr, 
                                                       create_new=create_new, n_processes=n_processes)
            elif 'blobs' in e:
                #print(traceid)
                exp = Objects(self.animalid, self.session, self.fov, traceid=traceid)
                roi_list, _ = exp.get_responsive_cells(response_type=response_type,
                                                       responsive_test=responsive_test, 
                                                       responsive_thr=responsive_thr,
                                                       create_new=create_new, n_processes=n_processes)
            else:
                print("--> [%s] not implemented" % e)
                continue
            
            exp.data = None
            self.experiments.update({e: exp})
            all_rois.extend(roi_list)
        
        unique_cells = list(set(all_rois))
        return np.array(unique_cells)
    
                
    def load_data(self, experiment=None, traceid='traces001', trace_type='corrected',\
                  make_equal=False, rootdir='/n/coxfs01/2p-data', update_self=True, add_offset=True):
        
        '''Set experiment = None to load all data'''
        
        #print("Session Object- loading data")
        if update_self:
            self.traceid = traceid
            self.trace_type = trace_type
        if experiment is not None:
            print("... Loading data (%s - %s - %s)" % (experiment, traceid, trace_type))
        else:
            print("... Loading all experiments (%s - %s)" % (traceid, trace_type))

        expdict = self.get_experiment_data(experiment=experiment,
                                           traceid=traceid,
                                           trace_type=trace_type,
                                           rootdir=rootdir, 
                                           make_equal=make_equal)
        
        if update_self and expdict is not None:
            self.experiments.update(expdict)
        return expdict
    

    def get_experiment_list(self, traceid='traces001', trace_type='corrected',\
                            rootdir='/n/coxfs01/2p-data'):

        fov_dir = os.path.join(rootdir, self.animalid, self.session, self.fov)
        run_list = sorted(glob.glob(os.path.join(fov_dir, '*_run[0-9]')), 
                            key=util.natural_keys)

        experiment_list = list(set([os.path.split(f)[-1].split('_run')[0] \
                                    for f in run_list]))
        
        if 'gratings' in experiment_list and (20190405 <= int(self.session) < 20190511):
            # Old experiment, where "gratings" were actually RFs
            experiment_list = [e for e in experiment_list if e != 'gratings']
            experiment_list.append('rfs') # These are always 5 degree res        
        
        
        return experiment_list
    
    def get_experiment_data(self, experiment=None, traceid='traces001', trace_type='corrected',\
                            add_offset=True, rootdir='/n/coxfs01/2p-data', make_equal=False):
        experiment_dict = {}

        all_experiments = self.get_experiment_list(traceid=traceid, trace_type=trace_type, rootdir=rootdir)
         
        if experiment is None: # Get ALL experiments
            experiment_types = all_experiments
        else:
            if not isinstance(experiment, list):
                experiment_types = [experiment]
            else:
                experiment_types = experiment
        print("... getting experiment data:", experiment_types)

        for experiment_type in experiment_types:     
            print("... ... loading: %s" % experiment_type)
            if int(self.session) < 20190511 and experiment_type == 'rfs':
                experiment_type = 'gratings' # Temporarily revert back to old-name since get_experiment_list() changed
                self.rois, tmp_tid = get_roi_id(self.animalid, self.session, self.fov, traceid, run_name=experiment_type, rootdir=rootdir)
                experiment_type = 'rfs' # change back
            else:
                self.rois, tmp_tid = get_roi_id(self.animalid, self.session, self.fov, traceid, run_name=experiment_type, rootdir=rootdir)
            print("... ... got rois")
            try:        
                exp = None
                if tmp_tid != self.traceid and 'retino' not in experiment_type:
                    self.traceid = tmp_tid
                    print("... ... (renamed traceid)")
                # exp = Experiment(experiment_type, self.animalid, self.session, self.fov, self.traceid, rootdir=rootdir)
                if 'rf' in experiment_type:
                    exp = ReceptiveFields(experiment_type, self.animalid, self.session, self.fov, self.traceid, rootdir=rootdir)
                elif 'grating' in experiment_type:
                    exp = Gratings(self.animalid, self.session, self.fov, self.traceid, rootdir=rootdir)
                elif 'blob' in experiment_type:
                    # TODO:  ficx this, not implemented
                    exp = Objects(self.animalid, self.session, self.fov, self.traceid, rootdir=rootdir)
                else:
                    print("Not implemented")
                    #exp = Experiment(self.animalid, self.session, self.fov, self.traceid, rootdir=rootdir)
                    #assert exp.roi_id is not None, "NOT PROCESSED"
                    
                if exp is None or exp.source is None:
                    continue
                exp.load(trace_type=trace_type, update_self=True, make_equal=make_equal, add_offset=add_offset)
                print("... ... loaded traces")
                if 'gratings' in experiment_type and int(self.session) < 20190511:
                    experiment_type = 'rfs'
            except Exception as e:
                traceback.print_exc()
                print("--- %s skipping ---" % experiment_type)
                exp = None
                #experiment_dict = None
            
            experiment_dict[experiment_type] = exp
               
        return experiment_dict
    
    
    def get_grouped_stats(self, experiment_type, response_type='dff',
                          responsive_thr=0.01, responsive_test=None, n_stds=2.5,
                          update=True, get_grouped=True, make_equal=False,
                          nframes_post=0., create_new=False,
                          pretty_plots=False, n_processes=1, add_offset=True, 
                          traceid='traces001', trace_type='corrected',
                          rootdir='/n/coxfs01/2p-data'):
        print("Session Object- Getting grouped roi stats: exp is", experiment_type)

        #assert exp in [v for k, v in self.experiments.items()], "*ERROR* - specified experiment (%s) not found in Session object." % exp.name
        expdict=None; estats_dict=None; nostats=[];
        
        experiment_names = [k for k, v in self.experiments.items()]
        # see if already loaded data:
        if experiment_type is not None:
            found_exp_names = [k for k in experiment_names if experiment_type == k]
        else:
            found_exp_names = experiment_names
        
        print("... found loaded experiments: ", found_exp_names)
        if isinstance(found_exp_names, list) and len(found_exp_names) > 0:
            print("... loading found experiments")
            if len(found_exp_names) > 1:
                for fi, fname in enumerate(found_exp_names):
                    print fi, fname
                sel = raw_input("Select IDX of exp to use: ")
                if sel == '':
                    expdict = dict((exp_name, self.experiments[exp_name]) for exp_name in found_exp_names)
                else:
                    exp_name = found_exp_names[int(sel)]
                    expdict = {exp_name: self.experiments[exp_name]}
            elif len(found_exp_names) == 1:
                exp_name = found_exp_names[0]
                expdict = {exp_name: self.experiments[exp_name]}
        else:
            # Load just this experiment type:
            print("... no experiment data saved, loading now...")
            expdict = self.load_data(experiment=experiment_type, traceid=traceid,
                                     trace_type=trace_type, rootdir=rootdir, 
                                     update_self=update, make_equal=make_equal, add_offset=add_offset)
            #exp = None if expdict is None else expdict[expdict.keys()[0]] 
        
        if expdict is not None:
            estats_dict = {}
            for exp_name, exp in expdict.items():
                if exp is None or 'retino' in exp_name:
                    continue
                print("... %s: calculating stats" % exp_name)
                #print("... [%s] Loading roi stats and cell list..." % exp.name)
                tmp_estats_dict = exp.get_stats(response_type=response_type, 
                                                pretty_plots=pretty_plots,
                                                responsive_test=responsive_test, 
                                                responsive_thr=responsive_thr, 
                                                n_stds=n_stds,
                                                get_grouped=get_grouped, 
                                                make_equal=make_equal,
                                                nframes_post=nframes_post,
                                                create_new=create_new,
                                                n_processes=n_processes)
                if tmp_estats_dict is not None:
                    estats_dict.update({exp_name: tmp_estats_dict})
                else:
                    nostats.append(','.join([self.animalid, self.session, self.fov, self.traceid, exp_name]))
                
        return estats_dict, nostats

    def get_roi_coordinates(rois='rois001', convert_um=True, create_new=False,
                            traceid=None, rootdir='/n/coxfs01/2p-data'):

        fovinfo = load_roi_coords(self.animalid, self.session, self.fov, 
                            roiid=rois, convert_um=convert_um, 
                            create_new=create_new,
                            traceid=self.traceid, rootdir=rootdir)

        return fovinfo

        
    #%%
    
class Experiment(object):
    def __init__(self, experiment_type, animalid, session, fov, \
                 traceid='traces001', rootdir='/n/coxfs01/2p-data', verbose=False):
        self.name = experiment_type
        self.animalid = animalid
        self.session = session
        self.fov = fov
        if int(session) < 20190511 and experiment_type=='gratings':
            experiment_type = 'rfs'
        self.experiment_type = experiment_type  
        if verbose:
            print("Creating %s object [%s|%s|%s|%s]" % (experiment_type, animalid, session, fov, traceid))

        self.get_roi_id(traceid=traceid, rootdir=rootdir)            
        paths = self.get_data_paths(rootdir=rootdir)
        assert paths is not None, "[ERROR] no paths found!"
        self.source = paths
        self.trace_type = None #trace_type
        self.data = None #Struct() #self.load()
        
        #print("SRC:", self.source)
        self.experiment_type = experiment_type
        if isinstance(self.source, list) and 'datasets' in os.path.split(self.source[0])[-1]:
            self.load()
        elif 'datasets' in os.path.split(self.source)[-1]:
            self.load()

       
        if 'gratings' in self.name and int(self.session) < 20190511:
            self.experiment_type = 'rfs'
        else:
            self.experiment_type = self.name
        
   
    def get_roi_masks(self, rois='', rootdir='/n/coxfs01/2p-data'):
        if rois == '':
            rois = self.rois
        masks, zimg = load_roi_masks(self.animalid, self.session, self.fov, rois=rois, rootdir=rootdir)
        return masks, zimg
 
    def get_roi_coordinates(self, rois='', convert_um=True, create_new=False,
                            traceid=None, rootdir='/n/coxfs01/2p-data'):
        if rois == '':
            rois = self.rois
        else:
            if rois != self.rois:
                print("updating roi id from %s to %s" % (rois, self.rois))
                self.rois = rois
        fovinfo = load_roi_coords(self.animalid, self.session, self.fov, 
                            roiid=rois, convert_um=convert_um,  
                            create_new=create_new,
                            traceid=self.traceid, rootdir=rootdir)

        return fovinfo

    def load_masks(self, rois='', rootdir='/n/coxfs01/2p-data'):
        if rois == '':
            rois = self.rois
        print("... loading masks (roiid: %s)" % rois)
        masks, zimg = load_roi_masks(self.animalid, self.session, self.fov, 
                                        rois=rois, rootdir=rootdir)
        return masks, zimg
 

    def print_info(self):
        print("************* Experiment Object info *************")
        print("Name: %s" % self.name)
        print("Experiment type: %s" % self.experiment_type)
        print("Animalid: %s" % self.animalid)
        print("Session: %s" % self.session)
        print("FOV: %s" % self.fov)
        print("roi-id, trace-id: %s, %s" % (self.rois, self.traceid))
        print("Data source:", self.source)
        if self.trace_type is not None:
            print("Loaded trace-type: %s" % self.trace_type)
        else:
            print("No data loaded yet.")
        print("**************************************************")
   

    def get_roi_id(self, traceid='traces001', rootdir='/n/coxfs01/2p-data'):

        extraction_type = re.sub('[0-9]+', '', traceid) \
                            if 'traces' in traceid else 'retino_analysis'
        
        if 'retino' in self.name and extraction_type=='traces':
            # using traceid in reference to other run types
            traceid_info_fpath = glob.glob(os.path.join(rootdir, self.animalid, 
                                              self.session, self.fov, '*', \
                                              'traces', 'traceids_*.json'))[0] 
        else:
            if int(self.session) < 20190511 and self.experiment_type=='rfs':
                exp_name = 'gratings'
            else:
                exp_name = self.experiment_type
            traceid_info_fpath = glob.glob(os.path.join(rootdir, self.animalid, 
                                            self.session, self.fov, 
                                            '*%s*' % exp_name, \
                                            'traces', '*.json'))[0] # % traceid, ))
        with open(traceid_info_fpath, 'r') as f:
            traceids = json.load(f)
            
        roi_id = traceids[traceid]['PARAMS']['roi_id']
        #print("GET ROI SET: %s" % roi_id)
        
        if 'retino' in self.name: #extraction_type == 'retino_analysis':
            extraction_type = 'retino_analysis'
            try:
                retinoid_info_fpath = glob.glob(os.path.join(rootdir, 
                                                self.animalid, self.session, self.fov, 
                                                '*%s*' % 'retino', 
                                                '%s' % extraction_type, '*.json'))[0] 

                with open(retinoid_info_fpath, 'r') as f:
                    retino_ids = json.load(f)
                found_ids = [t for t, tinfo in retino_ids.items() 
                                if 'roi_id' in tinfo['PARAMS'].keys()
                                and tinfo['PARAMS']['roi_id'] == roi_id]
                if len(found_ids) > 1:
                    for fi, fid in enumerate(found_ids):
                        print fi, fid
                    sel = input(">1 retino analysis [%s]. Select IDX to use: " % roi_id)
                    traceid = found_ids[int(sel)]
                else:
                    traceid = found_ids[0]
            except Exception as e:
                print(e)
                print("-- can't create EXP for retino, not processed: %s|%s|%s|%s" 
                        % (self.animalid, self.session, self.fov, self.name))
                roi_id = None
            
        self.rois = roi_id
        self.traceid = traceid
        
        return roi_id#, traceid

    def get_stimuli(self, rootdir='/n/coxfs01/2p-data', verbose=False):
        if verbose:
            print("... getting stimulus info for: %s" % self.name)
        dset_path = glob.glob(os.path.join(rootdir, self.animalid, self.session,
                                           self.fov, self.name, 'traces/traces*', 
                                           'data_arrays', 'labels.npz'))[0]
        dset = np.load(dset_path)
        sdf = pd.DataFrame(dset['sconfigs'][()]).T
        if 'blobs' in self.name:
            sdf = util.reformat_morph_values(sdf)
     
        return sdf
    
    def load(self, trace_type='corrected', update_self=True, 
                make_equal=False, add_offset=True, 
                rootdir='/n/coxfs01/2p-data', create_new=False):
        '''
        Populates trace_type and data
        '''
        
        self.trace_type=trace_type
        self.data = Struct()
        
        if not(isinstance(self.source, list)):
            assert os.path.exists(self.source), "Path does not exist! -- %s" % self.source
            #print("... exp.load()") 
            try:
                if self.source.endswith('npz'):
                    basename = os.path.splitext(os.path.split(self.source)[-1])[0]
                    if 'np_subtraced' != basename:
                        soma_fpath = self.source.replace(basename, 'np_subtracted')
                        raw_fpath = self.source.replace(basename, 'raw')
                    #print(soma_fpath)
                    if create_new is True or not os.path.exists(soma_fpath) or not os.path.exists(raw_fpath):
                        # Realign traces
                        print("*****corrected offset unfound, running now*****")
                        print("%s | %s | %s | %s | %s" % (self.animalid, 
                            self.session, self.fov, self.experiment_type, self.traceid))

                        aggregate_experiment_runs(self.animalid, self.session, 
                                                    self.fov, self.experiment_type, 
                                                    traceid=self.traceid)
                        print("*****corrected offsets!*****")

                    traces, labels, sdf, run_info = util.load_dataset(soma_fpath, 
                                                            trace_type=trace_type, 
                                                            add_offset=add_offset, 
                                                            make_equal=make_equal)
                   
                    # Stimulus / condition info
                    self.data.labels = labels 
                    self.data.sdf = sdf
                    self.data.info = run_info 
                    
                elif self.source.endswith('h5'):
                    #dfile = h5py.File(self.source, 'r')
                    # TODO: formatt retino data in sensible way with rutils
                    self.data.info = retinotools.get_protocol_info(self.animalid, 
                                            self.session, self.fov, run=self.name,
                                            rootdir=rootdir)
                    traces, self.data.labels = retinotools.format_retino_traces(
                                                    self.source, info=self.data.info)       
                # Update self:"
                if update_self:
                    #print("... updating self")
                    self.data.traces = traces
                else:
                    #print(".... returning")
                    return traces, self.data.labels
                
            except Exception as e:
                traceback.print_exc()
                print("ERROR LOADING DATA")
                self.data = None
    
        else:
            print("*** NOT IMPLEMENTED ***\n--%s--" % self.source)
            return None, None
        #return data
    
            
    def get_data_paths(self, rootdir='/n/coxfs01/2p-data'):
        # print("... getting data paths - name: %s" % self.name)
        fov_dir = os.path.join(rootdir, self.animalid, self.session, self.fov)
        if 'retino' in self.name:
            print("...retino traceid:", self.traceid)
            # Check that this is ROI:
            roi_info_fp = glob.glob(os.path.join(fov_dir, 'retino*', 
                                        'retino_analysis', 'analysis*.json'))[0]
            with open(roi_info_fp, 'r') as f:
                rids = json.load(f)
            traceid_name = self.traceid if 'traces' not in self.traceid else self.traceid.replace('traces', 'analysis')
            if rids[traceid_name]['PARAMS']['roi_type'] != 'manual2D_circle':
                traceid_name = [r for r, k in rids.items() 
                                if k['PARAMS']['roi_type']=='manual2D_circle'][0]
            self.traceid = traceid_name
            print("...updated analysis id:", traceid_name)

            all_runs = glob.glob(os.path.join(fov_dir, '*%s*' % 'retino', 
                                        'retino_analysis', 'analysis*', 'traces', '*.h5'))
            trace_extraction = 'retino_analysis'

        else:
            try:
                all_runs = glob.glob(os.path.join(fov_dir, '*%s_*' % self.name, 
                                        'traces', 'traces*', 'data_arrays', 
                                        'np_subtracted*.npz'))
                assert len(all_runs) > 0, "np_subtracted not found!"
            except Exception as e:
                all_runs = glob.glob(os.path.join(fov_dir, '*%s_*' % self.name, 
                                        'traces', 'traces*', 'data_arrays', 
                                        'datasets.npz'))
                
            trace_extraction = 'traces'

        #print("FOUND RUNS:", all_runs)    
        if len(all_runs) == 0:
            print("[ERROR]: No extracted traces: %s" % self.name) 
            return None
        
        all_runs = [s.split('/%s' % trace_extraction)[0] for s in all_runs]
        combined_runs = np.unique([r for r in all_runs if 'combined' in r])
        #print(combined_runs)
        single_runs = []
        for crun in combined_runs:
            stim_type = re.search('combined_(.+?)_static', os.path.split(crun)[-1]).group(1)
            #print stim_type
            single_runs.extend(glob.glob(os.path.join(fov_dir, '%s_run*' % stim_type)))
        run_list = list(set([r for r in all_runs if r not in single_runs and 'compare' not in r]))
        #print run_list
      
        data_fpaths = []
        for run_dir in run_list:
            run_name = os.path.split(run_dir)[-1]
            #print("... ... %s" % run_name)
            try:
                #print("... run: %s" % run_name)
                if 'retino' in run_name:
                    # Select analysis ID that corresponds to current ROI set:
                    extraction_name = 'retino_analysis'
                    # returns extracted raw_traces (.h5)
                    fpath = get_retino_analysis(self.animalid, self.session, self.fov,\
                                                run=run_name, rois=self.rois, 
                                                rootdir=rootdir)
                    
                else:
                    extraction_name = 'traces'
                    try:
                        fpath = glob.glob(os.path.join(run_dir, 'traces', 
                                                    '%s*' % self.traceid, \
                                                   'data_arrays', 'np_subtracted.npz'))[0] 
                    except Exception as e:
                        fpath = glob.glob(os.path.join(run_dir, 'traces', 
                                                    '%s*' % self.traceid, \
                                                   'data_arrays', 'datasets.npz'))[0] #
                data_fpaths.append(fpath)
            except IndexError:
                print("... no data arrays found: %s" % run_name)

            data_fpaths = list(set(data_fpaths))
            if len(data_fpaths) > 1:
                data_fpath = [f for f in data_fpaths if self.name in f][0]
            else:
                data_fpath = data_fpaths[0]

            if not isinstance(data_fpath, list):
                corresp_run_name = os.path.split(data_fpath.split('/%s/' % extraction_name)[0])[-1]
                if self.name != corresp_run_name:
                    #print("... renaming experiment to run name: %s" % corresp_run_name)
                    self.name = corresp_run_name

        return data_fpath

    def process_traces(self, response_type='dff', nframes_post_onset=None):
        print("... getting traces: %s" % response_type)
        traces = process_traces(self.data.traces, self.data.labels, 
                                response_type=response_type, 
                                nframes_post_onset=nframes_post_onset)
        
        return traces
    
    def get_trial_metrics(self, response_type='dff', nframes_post_onset=None, 
                            add_offset=True):

        if self.data is None or self.trace_type != 'corrected':
            traces, labels = self.load(trace_type='corrected', 
                                        update_self=False, add_offset=add_offset)
        else:
            traces = self.data.traces
            labels = self.data.labels
            
        trialmetrics = get_trial_metrics(traces, labels, response_type=response_type, 
                                            nframes_post_onset=nframes_post_onset)
        
        return trialmetrics
                    

    def get_responsive_cells(self, response_type='dff', responsive_test='ROC', 
                                responsive_thr=0.05, n_stds=2.5, 
                                create_new=False, n_processes=1):
        print("... (%s|%s) getting responsive cells (test: %s, thr: %.2f')" % (self.animalid, self.session, responsive_test, responsive_thr))
        assert ('blobs' in self.name) or ('gratings' in self.name and ((int(self.session) >= 20190511) or (int(self.session)<=20190405))), "Incorrect call for event data analysis (expecting gratings or blobs)."
        try:
            roi_list, nrois_total = get_responsive_cells(self.animalid, 
                                            self.session, 
                                            self.fov, run=self.name, 
                                            traceid=self.traceid, 
                                            responsive_test=responsive_test, 
                                            responsive_thr=responsive_thr, 
                                            n_stds=n_stds,
                                            create_new=create_new, 
                                            n_processes=n_processes)
            assert roi_list is not None, "--- no stats on initial pass"
        except Exception as e:
            print(e)
            return None, None
#            if responsive_test == 'nstds':
#                print("... trying calculating nframes above/below nstd")
#                framesdf = self.calculate_nframes_above_nstds(n_stds=n_stds, 
#                                                        trace_type=response_type) #'dff')
#                roi_list = [roi for roi in framesdf.columns \
#                                if any(framesdf[roi] > responsive_thr)]
#                nrois_total = framesdf.shape[-1]
#            else:
#                return None, None
            
        return roi_list, nrois_total

    def get_stats(self, trace_type='corrected', response_type='dff', 
                    responsive_test=None, responsive_thr=0.05, n_stds=2.5, 
                    n_processes=1, add_offset=True, create_new=False,
                    make_equal=False, get_grouped=True,
                    nframes_post=0, **kwargs): 
        print("... [%s] Loading roi stats and cell list..." % self.name)
        dircontents = [i for i in dir(self.data) if not i.startswith('__')]
        if self.data is None or len(dircontents)==0 \
                or ('traces' in dir(self.data) and self.data.traces is None) \
                or self.trace_type != 'corrected':
            self.load(trace_type='corrected', make_equal=make_equal, 
                        add_offset=add_offset, update_self=True)
        
        if responsive_test is not None:    
            print("filtering responsive cells: %s" % responsive_test)
            roi_list, nrois_total = self.get_responsive_cells(
                                                response_type=response_type, 
                                                responsive_test=responsive_test, 
                                                responsive_thr=responsive_thr,
                                                n_stds=n_stds,
                                                create_new=create_new, 
                                                n_processes=n_processes)
            
            if roi_list is None:
                print("--- NO STATS (%s)" % responsive_test)
                return None
        else:
            print("... no responsivity test specified. grabbing all.")
            roi_list = range(self.data.traces.shape[-1])
            #nrois_total = len(roi_list)
            
        if 'combined' in self.name:
            experiment_id = self.name.split('_')[1]
        else:
            experiment_id = self.name.split('_')[0]
            
        estats = Struct()
        estats.experiment_id = experiment_id
        estats.gdf = None
        estats.sdf = None
    
        estats.rois = roi_list
        estats.nrois = self.data.traces.shape[-1]
        
        estats.gdf = resp.group_roidata_stimresponse(self.data.traces[roi_list], 
                                                    self.data.labels, 
                                                    roi_list=roi_list,
                                                    return_grouped=get_grouped,
                                                    nframes_post=nframes_post)
        estats.sdf = self.data.sdf

        return estats #{experiment_id: estats}
    

    def calculate_nframes_above_nstds(self, n_stds=2.5, trace_type='dff', add_offset=True,
                        rootdir='/n/coxfs01/2p-data'):
        traces_basedir = self.source.split('/data_arrays/')[0]
        output_dir = os.path.join(traces_basedir, 'summary_stats', 'nstds')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
 
        framesdf = calculate_nframes_above_nstds(self.animalid, self.session, self.fov, 
                                                run=self.name, traceid=self.traceid,
                                            n_stds=2.5, create_new=True,
                                            n_processes=1, rootdir=rootdir)
               
        return framesdf
    
    
#%%
               
class Gratings(Experiment):
    def __init__(self, animalid, session, fov, traceid='traces001', rootdir='/n/coxfs01/2p-data'):
        super(Gratings, self).__init__('gratings', animalid, session, fov, traceid=traceid, rootdir=rootdir)
        self.experiment_type = 'gratings'
        
    def get_tuning(self, response_type='dff', responsive_test='nstds', 
                   responsive_thr=10, n_stds=2.5,
                   n_bootstrap_iters=1000, n_resamples=20, n_intervals_interp=3, 
                   min_cfgs_above=2, nframes_post=0, 
                   rootdir='/n/coxfs01/2p-data', create_new=False, n_processes=1, make_plots=False):
        '''
        This method is effecively the same as osi.get_tuning(), but without 
        having to do redundant data loading.
        '''
        osidir, fit_desc = osi.create_osi_dir(self.animalid, self.session, self.fov, self.name, 
                                          traceid=self.traceid, response_type=response_type, 
                                          responsive_test=responsive_test, responsive_thr=responsive_thr, n_stds=n_stds,
                                          n_bootstrap_iters=n_bootstrap_iters, n_resamples=n_resamples, 
                                          rootdir=rootdir)
        #print("... getting OSI results: %s" % fit_desc)
        
        # tmp:
        if not os.path.exists(os.path.join(os.path.split(self.source)[0], 'np_subtracted.npz')):
            print("--- no corrected data found, re-doing fits.")
            create_new = True
    
        if create_new is False:
            bootresults, fitparams = osi.load_tuning_results(self.animalid, self.session, self.fov,
                                                                self.name, traceid=self.traceid, 
                                                                fit_desc=fit_desc,
                                                                rootdir=rootdir)
            do_fits = bootresults is None
        else:
            do_fits=True
    
        if do_fits:
            print("---- doing fits ----")
#            fitdf, fitparams, fitdata = self.fit_tuning(n_processes=n_processes, response_type=response_type,
#                                                        responsive_test=responsive_test, responsive_thr=responsive_thr, n_stds=n_stds,
#                                                        n_resamples=n_resamples, n_bootstrap_iters=n_bootstrap_iters,
#                                                        n_intervals_interp=n_intervals_interp)        
#            fitparams.update({'directory': osidir,
#                              'response_type': response_type,
#                              'responsive_test': responsive_test,
#                              'responsive_thr': responsive_thr if responsive_test is not None else None,
#                              'n_stds': n_stds if responsive_test=='nstds' else None})
#            osi.save_tuning_results(fitdf, fitparams, fitdata)
            bootresults, fitparams = self.fit_tuning(n_processes=n_processes, 
                                                    response_type=response_type,
                                                    responsive_test=responsive_test, 
                                                    responsive_thr=responsive_thr, 
                                                    n_stds=n_stds,
                                                    n_resamples=n_resamples, 
                                                    n_bootstrap_iters=n_bootstrap_iters,
                                                    n_intervals_interp=n_intervals_interp,
                                                    min_cfgs_above=min_cfgs_above,
                                                    nframes_post=nframes_post)                
        
            fitparams['directory'] = osidir
            if create_new:
                print("Moving old files...")
                old_dir = os.path.join(osidir, 'old-fits')
                if not os.path.exists(old_dir):
                    os.makedirs(old_dir)
                old_roi_dir = os.path.join(osidir, 'roi_fits')
                if os.path.exists(old_roi_dir):
                    shutil.move(old_roi_dir, old_dir) 
                
                if os.path.exists(os.path.join(osidir, 'evaluation')):
                    shutil.move(os.path.join(osidir, 'evaluation'), old_dir) 
                for f in glob.glob(os.path.join(osidir, 'tuning_bootstrap*.pkl')):
                    shutil.move(f, os.path.join(old_dir, os.path.split(f)[-1]))
                
            osi.save_tuning_results(bootresults, fitparams)
            if make_plots:
                self.plot_roi_tuning_and_fit(bootresults, fitparams)

        return bootresults, fitparams

                    
    def plot_roi_tuning_and_fit(self, bootresults, fitparams):
        print("---- plotting tuning curves and fits for each roi ----")
        osidir = fitparams['directory']
        if not os.path.exists(os.path.join(osidir, 'roi-fits')):
            os.makedirs(os.path.join(osidir, 'roi-fits'))
        print("Saving roi tuning fits to: %s" % os.path.join(osidir, 'roi-fits'))

        if fitparams['response_type'] == 'dff' and self.trace_type == 'corrected': #'dff':
            raw_traces = self.process_traces(response_type=fitparams['response_type'])
        else:
            raw_traces = self.traces
        
        passrois = sorted([k for k, v in bootresults.items() if any(v.values())])
        print("%i cells fit at least 1 tuning curve." % len(passrois))
        
        sdf = self.get_stimuli()
        fit_desc = os.path.split(osidir)[-1]
        data_identifier = '|'.join([self.animalid, self.session, self.fov, self.name, self.traceid, fit_desc])
        for roi in passrois:
            #stimparams = [cfg for cfg, res in bootresults[roi].items() if res is not None]
            for stimpara, bootr in bootresults[roi].items():
                #  ['fits', 'stimulus_configs', 'data', 'results']
                fig, stimkey = osi.plot_tuning_bootresults(roi, bootr, raw_traces, self.data.labels, sdf, trace_type=fitparams['response_type'])
                util.label_figure(fig, data_identifier)
                pl.savefig(os.path.join(osidir, 'roi-fits', 'roi%05d__%s.png' % (int(roi+1), stimkey)))
                pl.close()
                    
   
        
    def fit_tuning(self, response_type='dff',
                   n_bootstrap_iters=100, n_resamples=60, n_intervals_interp=3,
                   responsive_test=None, responsive_thr=0.05, n_stds=2.5, create_new=False,
                   add_offset=True, nframes_post=0, 
                   rootdir='/n/coxfs01/2p-data', n_processes=8, min_cfgs_above=2):
       
        print("... FIT: getting stats") 
        rstats, roi_list, nrois_total = get_roi_stats(self.animalid, self.session, 
                                                    self.fov, traceid=self.traceid,
                                                    exp_name=self.name, 
                                                    responsive_test=responsive_test,
                                                    responsive_thr=responsive_thr, 
                                                    n_stds=n_stds, 
                                                    create_new=create_new, 
                                                    n_processes=n_processes)
        if rstats is None:
            return None, None

        if responsive_test == 'nstds':
            statdf = rstats['nframes_above']
        else:
            statdf = None
            
        estats = self.get_stats(responsive_test=responsive_test, 
                                responsive_thr=responsive_thr, n_stds=n_stds,
                                add_offset=add_offset, nframes_post=nframes_post, 
                                create_new=create_new, n_processes=n_processes)
       
        print("... FIT: doing bootstrap") 
#        fitdf, fitparams, fitdata = osi.do_bootstrap_fits(estats.gdf, estats.sdf, roi_list=estats.rois, 
#                                                     n_bootstrap_iters=n_bootstrap_iters, 
#                                                     n_resamples=n_resamples,
#                                                     n_intervals_interp=n_intervals_interp, n_processes=n_processes)
        
        # Save fitso_bootstrap_fits
        bootresults = osi.do_bootstrap(estats.gdf, estats.sdf, allconfigs=True,
                                   roi_list=estats.rois, statdf=statdf,
                                    response_type=response_type,
                                    n_bootstrap_iters=n_bootstrap_iters, n_resamples=n_resamples,
                                    n_intervals_interp=n_intervals_interp, n_processes=n_processes,
                                    min_cfgs_above=min_cfgs_above, min_nframes_above=responsive_thr)

        non_ori_configs = osi.get_non_ori_params(estats.sdf)
        fitparams = {#'directory': osidir,
                          'response_type': response_type,
                          'responsive_test': responsive_test,
                          'responsive_thr': responsive_thr if responsive_test is not None else None,
                          'n_stds': n_stds if responsive_test=='nstds' else None,
                          'n_bootstrap_iters': n_bootstrap_iters,
                          'n_resamples': n_resamples,
                          'n_intervals_interp': n_intervals_interp,
                          'min_cfgs_above': min_cfgs_above if statdf is not None else None,
                          'min_nframes_above': responsive_thr if statdf is not None else None,
                          'non_ori_configs': non_ori_configs}

        print("... FIT: done!")  
        return bootresults, fitparams
    

    def evaluate_fits(self, bootresults, fitparams, goodness_thr=0.66, 
                      make_plots=True, create_new=False, 
                      n_processes=1, rootdir='/n/coxfs01/2p-data'):
        rmetrics = None; goodrois = None;

        fit_desc = os.path.split(fitparams['directory'])[-1]
        
        #data_identifier = '|'.join([self.animalid, self.session, self.fov, self.traceid, fit_desc])
        rmetrics, rmetrics_by_cfg = osi.evaluate_tuning(self.animalid, self.session, self.fov, self.name, 
                                                    traceid=self.traceid, fit_desc=fit_desc, gof_thr=goodness_thr,
                                                    rootdir=rootdir, plot_metrics=make_plots)

#        fitdf, goodfits = osi.evaluate_bootstrapped_tuning(fitdf, fitparams, fitdata, goodness_thr=goodness_thr,
#                                                     response_type=fitparams['response_type'], data_identifier=data_identifier)
        if rmetrics is not None: 
            goodrois = rmetrics.index.tolist()
      
        return rmetrics, goodrois
    

    def get_nonori_params(self, paramnames=['size', 'speed', 'sf'], response_type='dff',
                          responsive_test='nstds', responsive_thr=10, n_stds=2.5, 
                          add_offset=True, nframes_post=0, create_new=False, n_processes=1):
        '''Returns average response to each NON-ori config (averaged over oris)
        '''
        estats = self.get_stats(responsive_test=responsive_test, 
                                responsive_thr=responsive_thr, 
                                n_stds=n_stds, add_offset=add_offset,
                                nframes_post=nframes_post, 
                                create_new=create_new, n_proceses=n_processes) 
        sdf = self.get_stimuli()
                
        n_params = len(paramnames)        
        tmplist = []
        for roi in estats.rois:
            # Get mean response across trials for each config
            responsevec = estats.gdf.get_group(roi).groupby(['config']).mean()[response_type]
            # Get all combinations of non-orientation configs
            paramconfigs = list(itertools.product(*[sdf[p].unique() for p in paramnames]))
            
            allresponses = []
            for paramvals in paramconfigs:
                if n_params == 3:
                    currcfgs = sdf[( (sdf[paramnames[0]]==paramvals[0]) \
                                      & (sdf[paramnames[1]]==paramvals[1]) \
                                      & (sdf[paramnames[2]]==paramvals[2]) )].index.tolist()
                elif n_params == 2:
                    currcfgs = sdf[( (sdf[paramnames[0]]==paramvals[0]) \
                                      & (sdf[paramnames[1]]==paramvals[1]) )].index.tolist()
                    
                if len(currcfgs)==0:
                    continue
                meanr = responsevec.loc[currcfgs].mean()
                currvs = dict((pname, pval) for pname, pval in zip(paramnames, paramvals))
                currvs['response'] = meanr
                allresponses.append(pd.Series(currvs))
            respdf = pd.concat(allresponses, axis=1).T
            respdf['cell'] = [roi for _ in range(respdf.shape[0])]
            tmplist.append(respdf)
        nonori_responses = pd.concat(tmplist, axis=0)
        return nonori_responses


#%%
            
class Objects(Experiment):
    def __init__(self, animalid, session, fov, traceid='traces001', rootdir='/n/coxfs01/2p-data', verbose=False):
        super(Objects, self).__init__('blobs', animalid, session, fov, traceid=traceid, rootdir=rootdir)
        self.experiment_type = 'blobs'
        

#%%


class ReceptiveFields(Experiment):
    def __init__(self, experiment_name, animalid, session, fov, traceid='traces001', trace_type='corrected', rootdir='/n/coxfs01/2p-data'):
        if int(session) < 20190511:
            super(ReceptiveFields, self).__init__('gratings', animalid, session, fov, traceid=traceid, rootdir=rootdir)
        else:
            super(ReceptiveFields, self).__init__(experiment_name, animalid, session, fov, traceid=traceid, rootdir=rootdir)
        self.experiment_type = 'rfs'
        self.traceid = traceid
        self.trace_type = trace_type
    
    def get_responsive_cells(self, response_type='dff', fit_thr=0.5, 
                                reload_data=False, create_new=False, n_processes=1, **kwargs):
        fit_results, fit_params = self.get_rf_fits(response_type=response_type, 
                                        fit_thr=fit_thr, reload_data=reload_data,
                                        n_processes=n_processes,
                                        create_new=create_new, make_pretty_plots=False)

        roi_list = [r for r, res in fit_results.items() \
                    if res['r2'] > fit_thr]
        nrois_total = len(fit_results.keys())
        print("... Getting responsive cells from fit_results (%i of %i fit with R2 > %.2f)" 
                % (len(roi_list), nrois_total, fit_thr))

        return roi_list, nrois_total
        
    def get_rf_fits(self, response_type='dff', fit_thr=0.5, make_pretty_plots=False,
                    scale_sigma=True, sigma_scale=2.35, reload_data=False,
                    create_new=False, rootdir='/n/coxfs01/2p-data', n_processes=1,
                    do_spherical_correction=False):
        '''
        Loads or does RF 2d-gaussian fit.
        reload_data = set True to reprocess data arrays

        Returns:
        
        fit_results (dict) - keys are rids, vals are fit results               
        fit_params (dict) - all meta info
        '''

        fit_results=None
        fit_params=None
        do_fits = create_new #False

        if create_new is False:
            try:
                print("... (stats) loading receptive field fits")
                fit_results, fit_params = fitrf.load_fit_results(self.animalid, 
                                                    self.session, self.fov,
                                                    experiment=self.name, 
                                                    traceid=self.traceid,
                                                    response_type=response_type,
                                                    do_spherical_correction=do_spherical_correction)
                print("... loaded fits (%s, %s)" % (self.name, fit_params['fit_desc']))
            except Exception as e:
                print(".... unable to load RF fit results. re-fitting...")
                do_fits = True
                traceback.print_exc()

        if do_fits:
            #make_pretty_plots=True
            try:
                print("... (stats) fitting receptive fields")
                fit_results, fit_params = fitrf.fit_2d_receptive_fields(self.animalid, 
                                                      self.session, 
                                                      self.fov, 
                                                      self.name, self.traceid, 
                                                      make_pretty_plots=make_pretty_plots,
                                                      create_new=True, #create_new,
                                                      reload_data=reload_data,
                                                      scale_sigma=scale_sigma,
                                                      sigma_scale=sigma_scale,
                                                      trace_type='corrected', 
                                                      response_type=response_type,
                                                      rootdir=rootdir, n_processes=n_processes,
                                                      do_spherical_correction=do_spherical_correction)
            except Exception as e:
                print("*** [ERROR]: UNABLE TO GET RF FITS ***")
                traceback.print_exc()

        if fit_results is not None:
            roi_list = [r for r, res in fit_results.items() \
                        if res['r2'] > fit_thr]
            nrois_total = len(fit_results.keys())
            print("... fit results (%i of %i attempted fits with R2 > %.2f)" 
                        % (len(roi_list), nrois_total, fit_thr))

        return fit_results, fit_params #roi_list, nrois_total

    def get_stats(self, response_type='dff', fit_thr=0.5, plot_pretty_rfs=False, 
                    scale_sigma=True, sigma_scale=2.35, nframes_post=0, 
                    do_fits=False, return_all_rois=True,
                    create_new=False, # create stats.pkl anew
                    add_offset=True,
                    reload_data=False, 
                    n_processes=1,
                    do_spherical_correction=False,
                    rootdir='/n/coxfs01/2p-data', **kwargs):
        '''
        return_all_rois will return all rois in grouped gdf.
        do_fits will redo 2d RF fits.

        '''
        make_equal = kwargs.get('make_equal', False)
        get_grouped = kwargs.get('get_grouped', True)
        print("... (%s) getting stats, reload_data=%s" % (self.name, str(reload_data)))

        # Get RF fit info
        rfdir, fit_desc= fitrf.create_rf_dir(self.animalid, self.session, 
                                            self.fov, self.name, 
                                            traceid=self.traceid, 
                                            response_type=response_type, 
                                            fit_thr=fit_thr, 
                                            do_spherical_correction=do_spherical_correction,
                                            rootdir=rootdir)
 
        estats = Struct()
        stats_fpath = os.path.join(rfdir, 'stats.pkl') 

        #check existing 
        if create_new is False:
            try:
                print("... loading existing stats")
                with open(stats_fpath, 'rb') as f:
                    estats = pkl.load(f)
                    assert 'rfits' not in dir(estats), "... recalculating stats"
                    assert 'fovinfo' not in dir(estats), "... recalculating stats"
            except Exception as e:
                reload_data = True
                create_new = True
  
        # Load fits
        fit_results, fit_params = self.get_rf_fits(response_type=response_type, 
                                                    fit_thr=fit_thr,
                                                    make_pretty_plots=plot_pretty_rfs,
                                                    do_spherical_correction=do_spherical_correction,
                                                    create_new=do_fits,
                                                    reload_data=reload_data, 
                                                    n_processes=n_processes)
         
        if fit_results is None:
            #print("--- NO STATS (%s)" % response_type)
            #create_new=True 
            return None
 
        roi_list = [r for r, res in fit_results.items() \
                    if res['r2'] > fit_thr]
        nrois_total = len(fit_results.keys())

        estats.rois = roi_list # This is list of cells that had R2 > 0.5
        estats.nrois = nrois_total # N rois that were fit attempted
        estats.fits = fitrf.rfits_to_df(fit_results, 
                                        fit_params=fit_params,
                                        row_vals=fit_params['row_vals'],
                                        col_vals=fit_params['col_vals'],
                                        roi_list=sorted(roi_list), 
                                        scale_sigma=scale_sigma, 
                                        sigma_scale=sigma_scale,
                                        spherical=do_spherical_correction)
        estats.fitinfo = fit_params
     
        if create_new:         
            estats.experiment_id = 'rfs'
            estats.gdf = None
            estats.sdf = None
            
            if self.data is None or self.trace_type != 'corrected':
                self.load(trace_type='corrected', 
                          make_equal=make_equal, 
                          add_offset=add_offset)
            all_rois = [r for r in self.data.traces.columns.tolist() 
                        if util.isnumber(r)]
            selected_rois = all_rois if return_all_rois else roi_list
            print("... selecting %i rois (return_all=%s)" 
                    % (len(selected_rois), str(return_all_rois)))
            estats.gdf = resp.group_roidata_stimresponse(self.data.traces[selected_rois], 
                                                         self.data.labels, 
                                                         roi_list=selected_rois,
                                                         return_grouped=get_grouped,
                                                         nframes_post=nframes_post)
            #estats.rfits = rfits 
            # print("... got rfits") 
            #rfits.pop('fit_results')                   
            #estats.fitinfo = rfits
            estats.sdf = self.data.sdf
            # also save FOV info
            #estats.fovcoords = self.get_fov_info(estats.fits, rfdir=rfdir)
    
            #print("... saving stats (%s)" % stats_fpath)
            with open(stats_fpath, 'wb') as f:
                pkl.dump(estats, f, protocol=pkl.HIGHEST_PROTOCOL)
        
        print("... got stats (n fit: %i, n attempted: %i" % (len(estats.rois), estats.nrois))
        return estats #{experiment_id: estats}


#    def get_fov_and_rf_coords(self, convert_um=True, transform_fov=True, 
#                                create_new=False):
#        fovcoords = self.get_roi_coordinates(create_new=create_new) 
#         
#        rfname = 'rfs10' if 'rfs10' in self.name else 'rfs'
#        fovinfo = fitrf.get_rf_to_fov_info(masks, rfdf, zimg, rfdir=rfdir, rfname=rfname, transform_fov=True)
#    
#        return fovinfo
#    
    
    def evaluate_fits(self, response_type='dff', fit_thr=0.5, n_processes=1,
                      n_bootstrap_iters=1000, n_resamples=10, ci=0.95, 
                      plot_boot_distns=True, nframes_post=0,
                      transform_fov=True, sigma_scale=2.35, scale_sigma=True,
                      do_fits=False, reload_data=False, 
                      do_spherical_correction=False,
                      create_new=False, rootdir='/n/coxfs01/2p-data'):
        
        from pipeline.python.classifications import evaluate_receptivefield_fits as evalrfs

        estats = self.get_stats(response_type=response_type, 
                                fit_thr=fit_thr,
                                nframes_post=nframes_post,
                                sigma_scale=sigma_scale, 
                                scale_sigma=scale_sigma,
                                plot_pretty_rfs=False,
                                do_fits=do_fits,
                                reload_data=reload_data,
                                create_new=reload_data,
                                n_processes=n_processes,
                                do_spherical_correction=do_spherical_correction) 

        data_id = '|'.join([self.animalid, self.session, self.fov, self.traceid, 
                            self.rois, self.trace_type, estats.fitinfo['fit_desc']])
      
        # get or do RF evaluation                
        eval_results, eval_params = evalrfs.evaluate_rfs(estats, 
                                            estats.fit_params,
                                            n_bootstrap_iters=n_bootstrap_iters, 
                                            n_resamples=n_resamples,
                                            ci=ci, n_processes=n_processes, 
                                            create_new=create_new) 

        #%% Identify reliable fits 
        reliable_rois = identify_reliable_fits(eval_results, fit_results, fit_params,
                                           pass_criterion=pass_criterion, 
                                           plot_boot_distns=plot_boot_distns, 
                                           plot_format=plot_format, outdir=roidir, 
                                           data_id=data_id)
        eval_results.update({'reliable_rois': reliable_rois}) 
        
        return eval_results, eval_params #deviants
    

    def regr_rf_fov(self, pass_criterion='all', sigma_scale=2.35, scale_sigma=True,
                    marker='o', marker_size=30, fill_marker=True, deviant_color='magenta'):
        # Get reliable
        fit_results, fit_params = evalrfs.load_matching_fit_results(self.animalid, 
                                                self.session, self.fov,
                                                experiment=self.experiment_type, 
                                                traceid=self.traceid, 
                                                response_type=self.response_type,
                                                sigma_scale=sigma_scale, 
                                                scale_sigma=scale_sigma)

        eval_results, eval_params = evalrfs.load_eval_results(self.animalid,
                                                self.session, self.fov,
                                                fit_desc=fit_params['fit_desc'],
                                                response_type=self.response_type)

        reliable_rois = identify_reliable_fits(eval_results, fit_results, fit_params,
                                           pass_criterion=pass_criterion, 
                                           plot_boot_distns=False) 

        evaldir = os.path.join(fit_params['rfdir'],'evaluation')
        data_id = '|'.join([self.animalid, self.session, self.fov, self.traceid, 
                            self.rois, self.trace_type, fit_params['fit_desc']])
 
        #%% Get measured fits
        meas_df = fitrf.rfits_to_df(fit_results, row_vals=fit_params['row_vals'], 
                                    col_vals=fit_params['col_vals'],
                                    scale_sigma=fit_params['scale_sigma'], 
                                    sigma_scale=fit_params['sigma_scale'])
        meas_df = meas_df[meas_df['r2']>fit_params['fit_thr']]
     

        #%% Fit linear regression for brain coords vs VF coords 
        fovcoords = self.get_roi_coordinates()
        posdf = pd.concat([meas_df[['x0', 'y0']].copy(), 
                           fovcoords['roi_positions'].copy()], axis=1) 
        posdf = posdf.rename(columns={'x0': 'xpos_rf', 'y0': 'ypos_rf',
                                      'ml_pos': 'xpos_fov', 'ap_pos': 'ypos_fov'})
            
        fig = evalrfs.plot_linear_regr_by_condition(posdf.loc[reliable_rois], 
                                                    meas_df.loc[reliable_rois])
        pl.subplots_adjust(top=0.9, bottom=0.1, hspace=0.5)
        label_figure(fig, data_id)
        pl.savefig(os.path.join(evaldir, 'RFpos_v_FOVpos_split_axes_reliable_cells.svg'))   
        pl.close()

        reg_results = evalrfs.compare_regr_to_boot_params(eval_results, posdf, 
                                            outdir=evaldir, data_id=data_id, 
                                            deviant_color=deviant_color, marker=marker,
                                            marker_size=marker_size, 
                                            fill_marker=fill_marker)

        return posdf, reg_results

    
class Retinobar(Experiment):
    def __init__(self, animalid, session, fov, traceid='analysis002', rootdir='/n/coxfs01/2p-data'):
        super(Retinobar, self).__init__('retinobar', animalid, session, fov, traceid=traceid, rootdir=rootdir)
        self.experiment_type = 'retinobar'
        self.load(update_self=True)
        print("--> loaded RetinoBar data:", self.data.traces.shape)
    
    def get_responsive_cells(self, responsive_thr=0.01):
        assert 'retino' in self.name, "Incorrect call for retinotopy moving-bar analysis."
        rstats, roi_list, nrois_total, trials_by_cond = get_retino_stats(self.data, responsive_thr=responsive_thr)

        return rstats, roi_list, nrois_total, trials_by_cond
    

    def get_stats(self, responsive_test='ROC', responsive_thr=0.01, 
                  make_equal=False):
                  #receptive_field_fit='zscore0.00_no_trim', 
                  #get_grouped=True, nframes_post=0):
        print("... [%s] Loading roi stats and cell list..." % self.name)

        rstats, roi_list, nrois_total, trials_by_cond = self.get_responsive_cells(
                                                                 responsive_test=responsive_test,
                                                                 responsive_thr=responsive_thr,
                                                                 receptive_field_fit=receptive_field_fit)
        
        if rstats is None and roi_list is None:
            print("--- NO STATS (%s)" % responsive_test)
            return None
    
    
        if 'gratings' in self.name and int(self.session) < 20190511:
            print("... renaming experiment to 'rfs' (old, pre-20190511)")
            experiment_id = 'rfs'
        else:
            if 'combined' in self.name:
                experiment_id = self.name.split('_')[1]
            else:
                experiment_id = self.name.split('_')[0]
            
        estats = Struct()
        estats.experiment_id = experiment_id
        estats.rois = roi_list
        estats.nrois = nrois_total
        estats.gdf = None
        estats.sdf = None
        
        if 'retino' not in experiment_id:
            self.load(trace_type='dff', make_equal=make_equal)
            estats.gdf = resp.group_roidata_stimresponse(self.data.traces[roi_list], 
                                                         self.data.labels, 
                                                         roi_list=roi_list,
                                                         return_grouped=get_grouped,
                                                         nframes_post=nframes_post)
              
            estats.finfo = rstats
            estats.sdf = self.data.sdf
        else:
            estats.gdf = rstats #rstats['magratios'].max(axis=1)
            assert trials_by_cond is not None, "Retino trial data failed to return"
            estats.sdf = trials_by_cond
            
        return {experiment_id: estats}
        
    
    
#%%
def format_roisXvalue(Xdata, run_info, fsmooth=None, sorted_ixs=None, value_type='meanstim', trace='raw'):

    #if isinstance(Xdata, pd.DataFrame):
    Xdata = np.array(Xdata)
        
    # Make sure that we only get ROIs in provided list (we are dropping ROIs w/ np.nan dfs on any trials...)
    #sDATA = sDATA[sDATA['roi'].isin(roi_list)]
    stim_on_frame = run_info['stim_on_frame']
    nframes_on = int(round(run_info['nframes_on']))
    ntrials_total = run_info['ntrials_total']
    nframes_per_trial = run_info['nframes_per_trial']
    nrois = Xdata.shape[-1] #len(run_info['roi_list'])
    
    if sorted_ixs is None:
        print "Trials are sorted by time of occurrence, not stimulus type."
        sorted_ixs = xrange(ntrials_total) # Just sort in trial order

    #trace = 'raw'
    traces = np.reshape(Xdata, (ntrials_total, nframes_per_trial, nrois), order='C')
    traces = traces[sorted_ixs,:,:]
    #rawtraces = np.vstack((sDATA.groupby(['roi', 'config', 'trial'])[trace].apply(np.array)).as_matrix())

    
#    if value_type == 'meanstimdff' and fsmooth is not None:
#        dftraces = np.array(Xdata/fsmooth)
#        dftraces = np.reshape(dftraces, (ntrials_total, nframes_per_trial, nrois), order='C')
#        dftraces = dftraces[sorted_ixs,:,:]
#        mean_stim_dff_values = np.nanmean(dftraces[:, stim_on_frame:stim_on_frame+nframes_on], axis=1)

    std_baseline_values = np.nanstd(traces[:, 0:stim_on_frame], axis=1)
    mean_baseline_values = np.nanmean(traces[:, 0:stim_on_frame], axis=1)
    mean_stim_on_values = np.nanmean(traces[:, stim_on_frame:stim_on_frame+nframes_on], axis=1)
    

    #zscore_values_raw = np.array([meanval/stdval for (meanval, stdval) in zip(mean_stim_on_values, std_baseline_values)])
    if value_type == 'zscore':
        values_df = (mean_stim_on_values - mean_baseline_values ) / std_baseline_values
    elif value_type == 'meanstim':
        values_df = mean_stim_on_values #- mean_baseline_values ) / std_baseline_values
#    elif value_type == 'meanstimdff':
#        values_df = mean_stim_dff_values #- mean_baseline_values ) / std_baseline_values
        
    #rois_by_value = np.reshape(values_df, (nrois, ntrials_total))
        
#    if bad_roi is not None:
#        rois_by_zscore = np.delete(rois_by_zscore, bad_roi, 0)

    return values_df #rois_by_value


#%%
        
#% Get mean trace for each condition:
def get_mean_cond_traces(ridx, Xdata, ylabels, tsecs, nframes_per_trial):
    '''For each ROI, get average trace for each condition.
    '''
    if isinstance(ylabels[0], str):
        conditions = sorted(list(set(ylabels)), key=util.natural_keys)
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


#%%

def get_xcond_dfs(roi_list, X, y, tsecs, run_info):
    nconds = len(run_info['condition_list'])
    averages_list = []
    normed_list = []
    for ridx, roi in enumerate(sorted(roi_list, key=util.natural_keys)):
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

#
def plot_roi_contours(zproj, sorted_rids, cnts, clip_limit=0.008, sorted_colors=None,
                      overlay=True, label=True, label_rois=[], thickness=2, 
                      draw_box=False, roi_color=(0, 255, 0), transform=False,
                      single_color=False, ax=None, font_scale=0.5):

    # Create ZPROJ img to draw on:
    refRGB = util.uint16_to_RGB(zproj)

    # Use some color map to indicate distance from upper-left corner:
    if sorted_colors is None:
        sorted_colors = sns.color_palette("Spectral", len(sorted_rids)) #masks.shape[-1])
    if ax is None:
        fig, ax = pl.subplots(1, figsize=(10,10))
    
    if overlay:
        #p2, p98 = np.percentile(refRGB, (1, 99))
        #img_rescale = exposure.rescale_intensity(refRGB, in_range=(p2, p98))
        im_adapthist = exposure.equalize_adapthist(refRGB, clip_limit=clip_limit)
        im_adapthist *= 256
        im_adapthist= im_adapthist.astype('uint8')
        ax.imshow(im_adapthist) #pl.figure(); pl.imshow(refRGB) # cmap='gray')
        orig = im_adapthist.copy()
    else:
        orig = np.zeros(refRGB.shape).astype('uint8')
        ax.imshow(orig)
        
    refObj = None
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
            (tlblX, tlblY) = util.midpoint(tl, bl)
            (trbrX, trbrY) = util.midpoint(tr, br)

            # compute the Euclidean distance between the midpoints,
            # then construct the reference object
            D = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
            refObj = (cidx, (cX, cY)) #(box, (cX, cY), D) # / args["width"])
            continue

        # draw the contours on the image
        #orig = refRGB.copy()
        if len(label_rois) > 1 and rid not in label_rois:
            col255 = 0
        else:
            if single_color:
                col255 = roi_color
            else:
                col255 = tuple([cval*255 for cval in sorted_colors[cidx]])
                
        if draw_box:
            cv2.drawContours(orig, [box.astype("int")], -1, col255, thickness)
        else:
            cv2.drawContours(orig, cnt, -1, col255, thickness)
            
        if label:
            cv2.putText(orig, str(rid+1), cv2.boundingRect(cnt)[:2], cv2.FONT_HERSHEY_COMPLEX, font_scale, [0])
        
        if transform:
            img = imutils.rotate(orig, 90)  
            #imageROI = orig.copy()
            #img = imutils.rotate_bound(imageROI, -90)
            
            ax.imshow(img)
            ax.invert_xaxis()
        else:
            ax.imshow(orig)
        
        
        # stack the reference coordinates and the object coordinates
        # to include the object center
        refCoords = refObj[1] #np.vstack([refObj[0], refObj[1]])
        objCoords = (cX, cY) #np.vstack([box, (cX, cY)])

        D = dist.euclidean((cX, cY), (refCoords[0], refCoords[1])) #/ refObj[2]
        distances.append(D)

    pl.axis('off')
    
    return ax
    
    
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


