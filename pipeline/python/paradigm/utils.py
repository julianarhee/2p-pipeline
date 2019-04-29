#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 12:31:35 2018

@author: juliana
"""
import re
import glob
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
import pprint
pp = pprint.PrettyPrinter(indent=4)
import cPickle as pkl
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
from collections import Counter
##
from pipeline.python.utils import natural_keys, replace_root, print_elapsed_time
#
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pipeline.python.traces.utils import get_frame_info
#
#%%
def combine_run_info(D, identical_fields=[], combined_fields=[]):
    
    run_info = {}
    
    info_keys = D[D.keys()[0]]['run_info'].keys()
    
    for info_key in info_keys:
        print info_key
        run_vals = [D[curr_run]['run_info'][info_key] for curr_run in D.keys()]
        if info_key in identical_fields:
            if isinstance(run_vals[0], list):
                for curr_run in D.keys():
                    if info_key in ['roi_list', 'condition_list']:
                        continue
                    print curr_run, info_key, D[curr_run]['run_info'][info_key]
                if info_key in ['frame_rate', 'nframes_on']:
                    assert all([int(round(run_vals[0][0])) == int(round(D[curr_run]['run_info'][info_key][0])) for curr_run in D.keys()]), "%s: All vals not equal!" % info_key
                else: 
                    assert all([run_vals[0] == D[curr_run]['run_info'][info_key] for curr_run in D.keys()]), "%s: All vals not equal!" % info_key
                run_info[info_key] = run_vals[0]
            elif isinstance(run_vals[0], (int, long, float)):
                uvals = np.unique([int(round(v)) for v in list(set(run_vals))] )
                assert len( uvals ) == 1, "** %s: runs do not match!" % info_key
                run_info[info_key] = uvals[0]
        elif info_key in combined_fields:

            if isinstance(run_vals[0], dict):
                uvals = dict((k,v) for rd in run_vals for k,v in rd.items())
            else:
                if isinstance(run_vals[0], list):
                    uvals = np.unique(list(itertools.chain.from_iterable(run_vals))) 
                else:
                    uvals = list(set(run_vals))
                if isinstance(uvals[0], (str, unicode)):
                    run_info[info_key] = list(uvals)
                elif isinstance(uvals[0], (int, float, long)):
                    run_info[info_key] = sum(uvals)
        else:
            print "%s: Not sure what to do with this..." % info_key
            run_info[info_key] = None
            
    return run_info

def combine_static_runs(check_blobs_dir, combined_name='combined', create_new=False, make_equal=True):
    print "Make Equal?", make_equal
    
    # First check if specified combo run exists:
    traceid_string = '_'.join([blobdir.split('/traces/')[-1] for blobdir in sorted(check_blobs_dir)])
    acquisition_dir = os.path.split(check_blobs_dir[0].split('/traces/')[0])[0]
    combined_traceid_dir = os.path.join(acquisition_dir, combined_name, 'traces', traceid_string)
    combined_darray_dir = os.path.join(combined_traceid_dir, 'data_arrays')
    if not os.path.exists(combined_darray_dir): os.makedirs(combined_darray_dir)
    combo_dpath = os.path.join(combined_darray_dir, 'datasets.npz')    
    
    if create_new is False:
        try:
            assert os.path.exists(combo_dpath), "Combined dset %s does not exist!" % combined_name
        except Exception as e:
            print "Creating new combined dataset: %s" % combined_name
            print "Combining data from dirs:\n", check_blobs_dir
            create_new = True
        
    if create_new:
        
        D = dict()
        for blobdir in check_blobs_dir:
            curr_run = os.path.split(blobdir.split('/traces')[0])[-1]
            print "Getting data array for run: %s" % curr_run
            print blobdir
            darray_fpath = glob.glob(os.path.join(blobdir, 'data_arrays', 'datasets.npz'))[0]
            curr_dset = np.load(darray_fpath)
            
            D[curr_run] = {'data':  curr_dset['corrected'],
                           'dff': curr_dset['dff'],
                            'meanstim': curr_dset['meanstim'],
                            'zscore': curr_dset['zscore'],
                           'labels_df':  pd.DataFrame(data=curr_dset['labels_data'], columns=curr_dset['labels_columns']),
                           'sconfigs':  curr_dset['sconfigs'][()],
                           'run_info': curr_dset['run_info'][()]
                           }
            
        unique_sconfigs = list(np.unique(np.array(list(itertools.chain.from_iterable([D[curr_run]['sconfigs'].values() for curr_run in D.keys()])))))
        sconfigs = dict(('config%03d' % int(cix+1), cfg) for cix, cfg in enumerate(unique_sconfigs))
        
        new_paradigm_dir = os.path.join(acquisition_dir, combined_name, 'paradigm')
        if not os.path.exists(new_paradigm_dir): os.makedirs(new_paradigm_dir);
        with open(os.path.join(new_paradigm_dir, 'stimulus_configs.json'), 'w') as f:
            json.dump(sconfigs, f, indent=4, sort_keys=True)
        
        # Remap config names for each run:
        last_trial_prev_run = 0
        prev_run = None
        for ridx, curr_run in enumerate(sorted(D.keys(), key=natural_keys)):
            print curr_run
            # Get the correspondence between current run's original keys, and the new keys from the combined stim list
            remapping = dict((oldkey, newkey) for oldkey, oldval in D[curr_run]['sconfigs'].items() for newkey, newval in sconfigs.items() 
                                                if newval==oldval)
            # Create dict of DF indices <--> new config key to replace each index once
            ixs_to_replace = dict((ix, remapping[oldval]) for ix, oldval in 
                                      zip(D[curr_run]['labels_df']['config'].index.tolist(), D[curr_run]['labels_df']['config'].values))
            # Replace old config with new config at the correct index
            D[curr_run]['labels_df']['config'].put(ixs_to_replace.keys(), ixs_to_replace.values())
            
            # Also replace trial names so that they have unique values between the two runs:
            if prev_run is not None:        
                last_trial_prev_run = int(sorted(D[prev_run]['labels_df']['trial'].unique(), key=natural_keys)[-1][5:]) #len(D[curr_run]['labels_df']['trial'].unique())
    
                trials_to_replace = dict((ix, 'trial%05d' % int(int(oldval[5:]) + last_trial_prev_run)) for ix, oldval in 
                                     zip(D[curr_run]['labels_df']['trial'].index.tolist(), D[curr_run]['labels_df']['trial'].values))
                D[curr_run]['labels_df']['trial'].put(trials_to_replace.keys(), trials_to_replace.values())
                
            prev_run = curr_run
            
    
        # Combine runs in order of their alphanumeric name:
        tmp_data = np.vstack([D[curr_run]['data'] for curr_run in sorted(D.keys(), key=natural_keys)]) 
        tmp_data_dff = np.vstack([D[curr_run]['dff'] for curr_run in sorted(D.keys(), key=natural_keys)]) 

        tmp_data_meanstim = np.vstack([D[curr_run]['meanstim'] for curr_run in sorted(D.keys(), key=natural_keys)])
        tmp_data_zscore = np.vstack([D[curr_run]['zscore'] for curr_run in sorted(D.keys(), key=natural_keys)])
        tmp_labels_df = pd.concat([D[curr_run]['labels_df'] for curr_run in sorted(D.keys(), key=natural_keys)], axis=0).reset_index(drop=True)
        
        # Get run_info dict:
        identical_fields = ['trace_type', 'roi_list', 'nframes_on', 'framerate', 'stim_on_frame', 'nframes_per_trial']
        combined_fields = ['traceid_dir', 'trans_types', 'transforms', 'nfiles', 'ntrials_total']
        
        rinfo = combine_run_info(D, identical_fields=identical_fields, combined_fields=combined_fields)
        
        replace_fields = ['condition_list', 'ntrials_by_cond']
        replace_keys = [k for k,v in rinfo.items() if v is None]
        assert replace_fields == replace_keys, "Replace fields (%s) and None keys (%s) do not match!" % (str(replace_fields), str(replace_keys))
        rinfo['condition_list'] = sorted(tmp_labels_df['config'].unique())
        rinfo['ntrials_by_cond'] = dict((cf, len(tmp_labels_df[tmp_labels_df['config']==cf]['trial'].unique())) for cf in rinfo['condition_list'])
        rinfo['ntrials_total'] =  sum([val for fi,val in rinfo['ntrials_by_cond'].items()])
        
        # CHeck N trials per condition:
        ntrials_by_cond = list(set([v for k,v in rinfo['ntrials_by_cond'].items()]))
        if make_equal is True and len(ntrials_by_cond) > 1:
            print "Uneven numbers of trials per cond. Making equal."
            configs_with_more = [k for k,v in rinfo['ntrials_by_cond'].items() if v>min(ntrials_by_cond)]
            ntrials_target = min(ntrials_by_cond)
            remove_ixs = []; removed_trials = [];
            for cf in configs_with_more:
                curr_trials = tmp_labels_df[tmp_labels_df['config']==cf]['trial'].unique()
                remove_rand_trial_ixs = random.sample(range(0, len(curr_trials)), len(curr_trials) - ntrials_target)
                remove_selected_trials = curr_trials[remove_rand_trial_ixs] 
                print "Removing %i trials (out of %i total)" % (len(remove_selected_trials), len(curr_trials))
                tmp_remove_ixs = tmp_labels_df[tmp_labels_df['trial'].isin(remove_selected_trials)].index.tolist()
                remove_ixs.extend(tmp_remove_ixs)
                removed_trials.extend(remove_selected_trials)
            
            all_ixs = np.arange(0, tmp_labels_df.shape[0])
            all_trials = sorted(tmp_labels_df['trial'].unique(), key=natural_keys)
            kept_frame_ixs = np.delete(all_ixs, remove_ixs)
            kept_trial_indices = [ti for ti,trial in enumerate(sorted(all_trials, key=natural_keys)) if trial not in removed_trials]
            
            labels_df = tmp_labels_df.iloc[kept_frame_ixs, :].reset_index(drop=True)
            data = tmp_data[kept_frame_ixs, :]
            data_dff = tmp_data_dff[kept_frame_ixs, :]

            data_meanstim = tmp_data_meanstim[kept_trial_indices, :]
            data_zscore = tmp_data_zscore[kept_trial_indices, :]
            
            rinfo['ntrials_by_cond'] = dict((cf, len(labels_df[labels_df['config']==cf]['trial'].unique())) for cf in rinfo['condition_list'])
            pp.pprint(rinfo['ntrials_by_cond'])
            rinfo['ntrials_total'] =  sum([val for fi,val in rinfo['ntrials_by_cond'].items()])

        else:
            labels_df = tmp_labels_df
            data = tmp_data
            data_dff = tmp_data_dff
            data_meanstim = tmp_data_meanstim
            data_zscore = tmp_data_zscore
            
        ylabels = labels_df['config'].values
        
        # Make sure runinfo traceid_dir now points to the COMBINED dir:
        rinfo['traceid_dir'] = combined_traceid_dir
       
        # Save it:
        np.savez(combo_dpath,
                 corrected=data,
                 dff=data_dff, 
                 meanstim=data_meanstim,
                 zscore=data_zscore,
                 ylabels=ylabels,
                 labels_data=labels_df,
                 labels_columns=labels_df.columns.tolist(),
                 run_info = rinfo,
                 sconfigs=sconfigs
                 )
    
    return combo_dpath

def get_equal_reps(dataset, data_fpath):
    eq_data_fpath = '%s_eq.npz' % os.path.splitext(data_fpath)[0]

    try:
        dataset = np.load(eq_data_fpath)
        return dataset, eq_data_fpath
    except Exception as e:
        print "Unable to load equalized data file -- creating new."
        
    tmp_data = dataset['corrected']
    tmp_data_meanstim = dataset['meanstim']
    tmp_data_zscore = dataset['zscore']
    tmp_labels_df = pd.DataFrame(data=dataset['labels_data'], columns=dataset['labels_columns'])
    rinfo = dataset['run_info'] if isinstance(dataset['run_info'], dict) else dataset['run_info'][()]
    sconfigs = dataset['sconfigs'] if isinstance(dataset['sconfigs'], dict) else dataset['sconfigs'][()]

    
    # CHeck N trials per condition:
    ntrials_by_cond = list(set([v for k,v in rinfo['ntrials_by_cond'].items()]))
    if len(ntrials_by_cond) > 1:
        print "Uneven numbers of trials per cond. Making equal."
        configs_with_more = [k for k,v in rinfo['ntrials_by_cond'].items() if v>min(ntrials_by_cond)]
        ntrials_target = min(ntrials_by_cond)
        remove_ixs = []; removed_trials = [];
        for cf in configs_with_more:
            curr_trials = tmp_labels_df[tmp_labels_df['config']==cf]['trial'].unique()
            remove_rand_trial_ixs = random.sample(range(0, len(curr_trials)), len(curr_trials) - ntrials_target)
            remove_selected_trials = curr_trials[remove_rand_trial_ixs] 
            print "Removing %i trials (out of %i total)" % (len(remove_selected_trials), len(curr_trials))
            tmp_remove_ixs = tmp_labels_df[tmp_labels_df['trial'].isin(remove_selected_trials)].index.tolist()
            remove_ixs.extend(tmp_remove_ixs)
            removed_trials.extend(remove_selected_trials)
        
        all_ixs = np.arange(0, tmp_labels_df.shape[0])
        all_trials = sorted(tmp_labels_df['trial'].unique(), key=natural_keys)
        kept_frame_ixs = np.delete(all_ixs, remove_ixs)
        kept_trial_indices = [ti for ti,trial in enumerate(sorted(all_trials, key=natural_keys)) if trial not in removed_trials]
        
        labels_df = tmp_labels_df.iloc[kept_frame_ixs, :].reset_index(drop=True)
        data = tmp_data[kept_frame_ixs, :]
        data_meanstim = tmp_data_meanstim[kept_trial_indices, :]
        data_zscore = tmp_data_zscore[kept_trial_indices, :]
        ylabels = labels_df['config'].values
        
        rinfo['ntrials_by_cond'] = dict((cf, len(labels_df[labels_df['config']==cf]['trial'].unique())) for cf in rinfo['condition_list'])
        pp.pprint(rinfo['ntrials_by_cond'])
        rinfo['ntrials_total'] =  sum([val for fi,val in rinfo['ntrials_by_cond'].items()])

#    dset = dict(dataset)
#    dset['corrected'] = data
#    dset['meanstim'] = data_meanstim
#    dset['zscore'] = data_zscore
#    dset['ylabels'] = ylabels
#    dset['run_info'][()] = rinfo
#    dset['labels_data'] = labels_df
#    dset['labels_columns'] = labels_df.columns.tolist()
#    
#        # Save it:
    np.savez(eq_data_fpath,
                 corrected=data,
                 meanstim=data_meanstim,
                 zscore=data_zscore,
                 ylabels=ylabels,
                 labels_data=labels_df,
                 labels_columns=labels_df.columns.tolist(),
                 run_info = rinfo,
                 sconfigs=sconfigs
                 )
    
    print "Saved adjusted dataset to:", eq_data_fpath
    
    dataset = np.load(eq_data_fpath)
            
    return dataset, eq_data_fpath

#%%
def df_to_sarray(df):
    """
    Convert a pandas DataFrame object to a numpy structured array.
    Also, for every column of a str type, convert it into 
    a 'bytes' str literal of length = max(len(col)).

    :param df: the data frame to convert
    :return: a numpy structured array representation of df
    """

    def make_col_type(col_type, col):
        try:
            if 'numpy.object_' in str(col_type.type):
                maxlens = col.dropna().str.len()
                if maxlens.any():
                    maxlen = maxlens.max().astype(int) 
                    col_type = ('S%s' % maxlen, 1)
                else:
                    col_type = 'f2'
            return col.name, col_type
        except:
            print(col.name, col_type, col_type.type, type(col))
            raise

    v = df.values            
    types = df.dtypes
    numpy_struct_types = [make_col_type(types[col], df.loc[:, col]) for col in df.columns]
    dtype = np.dtype(numpy_struct_types)
    z = np.zeros(v.shape[0], dtype)
    for (i, k) in enumerate(z.dtype.names):
        # This is in case you have problems with the encoding, remove the if branch if not
        try:
            if dtype[i].str.startswith('|S'):
                z[k] = df[k].str.encode('latin').astype('S')
            else:
                z[k] = v[:, i]
        except:
            print(k, v[:, i])
            raise

    return z, dtype

#%% Formatting functions for raw traces:

def run_info_from_dfs(rundir, raw_df, labels_df, traceid_dir='', trace_type='np_subtracted', ntiffs=None, verbose=True):
    
    conditions = sorted(list(set(labels_df['config'])), key=natural_keys)
    
    # Get SI info:
    si_info = get_frame_info(rundir)
    if ntiffs is None:
        ntiffs = si_info['ntiffs']
        
    
    # Get stimulus info:
    stimconfigs_fpath = os.path.join(rundir, 'paradigm', 'stimulus_configs.json')
    with open(stimconfigs_fpath, 'r') as f:
        stimconfigs = json.load(f)
    print "Loaded %i stimulus configurations." % len(stimconfigs.keys())

    transform_dict, object_transformations = get_transforms(stimconfigs)
    trans_types = object_transformations.keys()

    # Get trun info:
    roi_list = sorted(list(set([r for r in raw_df.columns if not r=='index'])), key=natural_keys)
    ntrials_total = len(sorted(list(set(labels_df['trial'])), key=natural_keys))
    trial_counts = labels_df.groupby(['config'])['trial'].apply(set)
    ntrials_by_cond = dict((k, len(trial_counts[i])) for i,k in enumerate(trial_counts.index.tolist()))
    #assert len(list(set(labels_df.groupby(['trial'])['tsec'].count()))) == 1, "Multiple counts found for ntframes_per_trial."
    nframes_per_trial = list(set(labels_df.groupby(['trial'])['tsec'].count())) #[0]
    if len(list(set(labels_df.groupby(['trial'])['tsec'].count()))) > 1:
        print "Multiple nframes per trial found:", nframes_per_trial
        
    nframes_on = list(set(labels_df['stim_dur']))
    if len(nframes_on) > 1:
        print "More than 1 stim dur found:", nframes_on
    nframes_on = [nf * si_info['framerate'] for nf in nframes_on]
    
    #assert len(nframes_on) == 1, "More than 1 unique stim duration found in Sdf..."
    #nframes_on = nframes_on[0] * si_info['framerate']

    # Get stim onset index for all trials:
    #tmat = np.reshape(labels_df['tsec'].values, (ntrials_total,nframes_per_trial))    
    ons = []
    all_trials = sorted(list(set(labels_df['trial'])), key=natural_keys)
    for trial in all_trials: #range(tmat.shape[0]):
        curr_tsecs = labels_df[labels_df['trial']==trial]['tsec'].values
        on_idx = [t for t in curr_tsecs].index(0)
        ons.append(on_idx)
    assert len(list(set(ons)))==1, "More than one unique stim ON idx found!"
    stim_on_frame = list(set(ons))[0]

    if verbose:
        print "-------------------------------------------"
        print "Run summary:"
        print "-------------------------------------------"
        print "N rois:", len(roi_list)
        print "N trials:", ntrials_total
        print "N frames per trial:", nframes_per_trial
        print "N trials per stimulus:"
        pp.pprint(ntrials_by_cond)
        print "-------------------------------------------"

    run_info = {'roi_list': roi_list,
                'ntrials_total': ntrials_total,
                'nframes_per_trial': nframes_per_trial,
                'ntrials_by_cond': ntrials_by_cond,
                'condition_list': conditions,
                'stim_on_frame': stim_on_frame,
                'nframes_on': nframes_on,
                'traceid_dir': traceid_dir, 
                'transforms': object_transformations,
                'trans_types': trans_types,
                'framerate': si_info['framerate'],
                'nfiles': ntiffs} #len([i for i in os.listdir(os.path.join(traceid_dir, 'files')) if i.endswith('hdf5')])

    return run_info

#%
def load_raw_run(traceid_dir, trace_type='np_subtracted', combined=False, create_new=False, fmt='hdf5', verbose=True):
    
    run_info = {}
            
    # Get paths to data source:
    run_dir = traceid_dir.split('/traces')[0]
    si_info = get_frame_info(run_dir)
    
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

    transform_dict, object_transformations = get_transforms(stimconfigs)
    trans_types = object_transformations.keys()

    labels_df, raw_df = get_raw_run(traceid_dir, create_new=create_new, fmt=fmt)
    conditions = sorted(list(set(labels_df['config'])), key=natural_keys)
    print "Got labels:"
    print labels_df.head()
 
    # Get trun info:
    roi_list = sorted(list(set([r for r in raw_df.columns if not r=='index'])), key=natural_keys)
    ntrials_total = len(sorted(list(set(labels_df['trial'])), key=natural_keys))
    trial_counts = labels_df.groupby(['config'])['trial'].apply(set)
    ntrials_by_cond = dict((k, len(trial_counts[i])) for i,k in enumerate(trial_counts.index.tolist()))
    #assert len(list(set(labels_df.groupby(['trial'])['tsec'].count()))) == 1, "Multiple counts found for ntframes_per_trial."
    nframes_per_trial = list(set(labels_df.groupby(['trial'])['stim_on_frame'].count())) #[0]
    nframes_on = list(set(labels_df['stim_dur']))
    #assert len(nframes_on) == 1, "More than 1 unique stim duration found in Sdf..."
    #nframes_on = nframes_on[0] * si_info['framerate']
    nframes_on = [int(round(si_info['framerate'])) * n for n in nframes_on]
    
    # Get stim onset index for all trials:
#    tmat = np.reshape(labels_df['tsec'].values, (ntrials_total, nframes_per_trial))    
#    ons = []
#    for ts in range(tmat.shape[0]):
#        on_idx = [t for t in tmat[ts,:]].index(0)
#        ons.append(on_idx)
#    assert len(list(set(ons)))==1, "More than one unique stim ON idx found!"
#    stim_on_frame = list(set(ons))[0]
    ons = [int(np.where(t==0)[0]) for t in labels_df.groupby('trial')['tsec'].apply(np.array)]
    assert len(list(set(ons)))==1, "More than one unique stim ON idx found!"
    stim_on_frame = list(set(ons))[0]

    if verbose:
        print "-------------------------------------------"
        print "Run summary:"
        print "-------------------------------------------"
        print "N rois:", len(roi_list)
        print "N trials:", ntrials_total
        print "N frames per trial:", nframes_per_trial
        print "N trials per stimulus:", ntrials_by_cond
        print "-------------------------------------------"

    run_info['roi_list'] = roi_list
    run_info['ntrials_total'] = ntrials_total
    run_info['nframes_per_trial'] = nframes_per_trial
    run_info['ntrials_by_cond'] = ntrials_by_cond
    run_info['condition_list'] = conditions
    run_info['stim_on_frame'] = stim_on_frame
    run_info['nframes_on'] = nframes_on
    run_info['traceid_dir'] = traceid_dir
    run_info['trace_type'] = trace_type
    run_info['transforms'] = object_transformations
    #run_info['datakey'] = datakey
    run_info['trans_types'] = trans_types
    run_info['framerate'] = si_info['framerate']
    run_info['nfiles'] = len([i for i in os.listdir(os.path.join(traceid_dir, 'files')) if i.endswith('hdf5')])

    return run_info, stimconfigs, labels_df, raw_df


#%%
def get_raw_run(traceid_dir, create_new=False, fmt='hdf5'):
    
    xdata_df=None; labels_df=None;
    # Set up paths to look for saved dataframes:
    data_array_dir = os.path.join(traceid_dir, 'data_arrays')
    if not os.path.exists(data_array_dir):
        os.makedirs(data_array_dir)
        
    # Standard filepaths for data arrays:
    xdata_fpath = os.path.join(data_array_dir, 'roiXtrials_raw.%s' % fmt) # processed OR raw
    
    # First, check that there data arrays have been extract from hdf5 output (traces/get_traces.py):
    n_orig_tiffs = len([r for r in os.listdir(os.path.join(traceid_dir, 'files')) if r.endswith('hdf5')])

    trace_arrays_dir = os.path.join(traceid_dir, 'files', 'trace_arrays_raw')
    if not os.path.exists(trace_arrays_dir):
        os.makedirs(trace_arrays_dir)
        
    n_src_dataframes = len([r for r in os.listdir(trace_arrays_dir) if 'File' in r and r.endswith(fmt)])
    if not n_orig_tiffs == n_src_dataframes or create_new is True:
        print "Extract RAW trace arrays from .tif files."
        raw_hdf_to_dataframe(traceid_dir, fmt=fmt)
        
    if not create_new:
        try:
            if fmt == 'pkl':
                with open(xdata_fpath, 'rb') as f:
                    dfile = pkl.load(f)
                xdata_df = dfile['xdata']
                labels_df = dfile['labels']
                print "Loaded XDATA."
            
            elif fmt == 'hdf5':
                hfile = h5py.File(xdata_fpath, 'r')
                xdata_df = pd.DataFrame(hfile['xdata'][:])
                labels_df = pd.DataFrame(hfile['labels'][:])
                print "Loaded XDATA."
                
            return labels_df, xdata_df
        
        except Exception as e:
            print "Unable to load raw data path: %s" % xdata_df
            print "Re-collating raw data..."

    # Create the .pkl file for rawXtrials:
    labels_df, xdata_df, _ = collate_trials(trace_arrays_dir, dff=False, smoothed=False, fmt=fmt)
    
    return labels_df, xdata_df
        

def raw_hdf_to_dataframe(traceid_dir, roi_list=[], fmt='pkl'):
#def get_raw_runs(acquisition_dir, run, traceid, roi_list=[], fmt='pkl'):
    
    #traceid_dir = get_traceid_from_acquisition(acquisition_dir, run, traceid)
    filetraces_dir = os.path.join(traceid_dir, 'files')
    raw_trace_arrays_dir = os.path.join(filetraces_dir, 'trace_arrays_raw')
    if not os.path.exists(raw_trace_arrays_dir):
        os.makedirs(raw_trace_arrays_dir)
    
    trace_fns = glob.glob(os.path.join(filetraces_dir, '*.hdf5')) # [f for f in os.listdir(filetraces_dir) if f.endswith('hdf5')]
    
    print "Getting frame x roi arrays for %i tifs." % len(trace_fns)
    print "... ... Ouput format: %s." % fmt
    print "... ... Saving to: %s" % raw_trace_arrays_dir
    
    for trace_fn_path in trace_fns:
        print "... ... ... %s" % trace_fn_path
        tfile = h5py.File(trace_fn_path, 'r')
        slice_list = sorted([str(s) for s in tfile.keys()], key=natural_keys)
        rois_by_slice = dict((curr_slice, []) for curr_slice in slice_list)
        curr_file = str(re.search(r"File\d{3}", trace_fn_path).group())


        df_list = []; frames_df_list=[];
        for curr_slice in slice_list:
            print "... ... getting %s, %s ROIs." % (curr_file, curr_slice)
            # Get frames x roi arrays (frames in single-plane):
            tracemat = np.array(tfile[curr_slice]['traces']['np_subtracted'])  # (nvolumes x nrois)
            #if len(roi_list)==0:
            if 'roi_indices' not in tfile[curr_slice].attrs.keys():
                roi_list = sorted(['roi%05d' % int(r+1) for r in range(tracemat.shape[1])], key=natural_keys)
                print roi_list[0:5]
            else:
                roi_list = sorted(['roi%05d' % int(r+1) for r in tfile[curr_slice].attrs['roi_indices']], key=natural_keys)
            curr_df = pd.DataFrame(data=tracemat, columns=roi_list, index=range(tracemat.shape[0]))
            df_list.append(curr_df)
            
            # Get frame indices:
            frame_indices = np.array(tfile[curr_slice]['frames_indices'])
            curr_frames_df = pd.DataFrame(data=frame_indices, columns=[curr_slice], index=range(tracemat.shape[0]))
            rois_by_slice.update({curr_slice: roi_list})
            frames_df_list.append(curr_frames_df)


        df = pd.concat(df_list, axis=1) # concat columns
        frames_df = pd.concat(frames_df_list, axis=1) # concat columns

        #out_fname = '%s.%s' % (str(os.path.splitext(trace_fn_path)[0]), fmt)
        out_fpath = os.path.join(raw_trace_arrays_dir, os.path.split(trace_fn_path)[-1])
        print "OUT: %s" % out_fpath
        if fmt=='hdf5':
            outfile = h5py.File(out_fpath, 'a')
            sa, saType = df_to_sarray(df)
            if 'raw' in outfile.keys():
                del f['raw']
            outfile.create_dataset('raw', data=sa, dtype=saType)

            sa, saType = df_to_sarray(frames_df)
            if 'frames_indices' in outfile.keys():
                del f['frames_indices'] 
            outfile.create_dataset('frames_indices', data=sa, dtype=saType)

            if 'rois_by_slice' in outfile.keys():
                del f['rois_by_slice']
            for curr_slice, curr_rois in rois_by_slice.items():
                outfile.create_dataset('rois_by_slice/%s' % curr_slice, data=curr_rois)

            outfile.close()
            del outfile

        elif fmt=='pkl':        
            with open(os.path.join(raw_trace_arrays_dir, out_fname), 'wb') as o:
                pkl.dump(df, o, protocol=pkl.HIGHEST_PROTOCOL)
        print "Saved %s" % out_fpath
    
    return raw_trace_arrays_dir


def make_raw_trace_arrays(options=[], acquisition_dir='', run='', traceid='', fmt='pkl'):
    '''
    Converts hdf5 files created from initial trace extraction step (traces/get_traces.py)
    into standard frames X roi dataframes. Uses get_raw_runs(), which assumes
    the raw traces are neuropil-subtracted.
    
    Either provide options list to extract info, 
    OR provide acquisition_dir, run id, trace id.
    '''
    
    if len(options) > 0:
        optsE = extract_options(options)
        run = optsE.run_list[0]
        traceid = optsE.traceid_list[0]
        acquisition_dir = os.path.join(optsE.rootdir, optsE.animalid, optsE.session, optsE.acquisition)
    
    traceid_dir = get_traceid_from_acquisition(acquisition_dir, run, traceid)
    raw_trace_arrays_dir = raw_hdf_to_dataframe(traceid_dir, fmt=fmt)
    
    return raw_trace_arrays_dir

#%%
def test_smooth_frac(trace_df, smoothed_df, nframes_to_show, dff=True, test_roi='roi00001'):
    
    # Check smoothing
    pl.plot(trace_df[test_roi][0:nframes_to_show], label='orig')
    pl.plot(smoothed_df[test_roi][0:nframes_to_show], label='smoothed')
    pl.legend()
    pl.draw()
    pl.pause(0.001)


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
   
    
def smooth_trace_arrays(traceid_dir, trace_type='processed', dff=False, frac=0.01, fmt='hdf5', user_test=False, test_roi='roi00001'):
    '''
    Smooth traces for each ROI by .tif file (continuous time points). These
    trace "chunks" can later be parsed and combined into a dataframe for trial
    analyses, but smoothing should be done within continuous periods of shutter
    on/off (i.e., a literal 'acquisition' in SI).
    '''
    
    # Create output dir for smoothing examples:
    smooth_fig_dir = os.path.join(traceid_dir, 'files', 'processing_examples', 'smoothing')
    if not os.path.exists(smooth_fig_dir):
        os.makedirs(smooth_fig_dir)

    # Get source data:
    trace_arrays_src = os.path.join(traceid_dir, 'files', 'trace_arrays_%s' % trace_type)
        
    # Set output:
    if dff:
        smoothed_trace_arrays_dir = os.path.join(traceid_dir, 'files', 'trace_arrays_%s_dff_smoothed%s' % (trace_type, str(frac).split('.')[-1]))
    else:
        smoothed_trace_arrays_dir = os.path.join(traceid_dir, 'files', 'trace_arrays_%s_smoothed%s' % (trace_type, str(frac).split('.')[-1]))
        
    print "Saving smoothed array files to:", smoothed_trace_arrays_dir
    if not os.path.exists(smoothed_trace_arrays_dir):
        os.makedirs(smoothed_trace_arrays_dir)
        
    # Load each file and smooth:
    trace_fns = [f for f in os.listdir(trace_arrays_src) if 'File' in f and f.endswith(fmt)]
    
    for dfn in trace_fns:
        if fmt == 'pkl':            
            with open(os.path.join(trace_arrays_src, dfn),'rb') as f:
                tfile = pkl.load(f)
            if trace_type == 'processed':
                trace_df = tfile['corrected']
                if dff:
                    trace_df /= tfile['F0']
            elif trace_type == 'raw':
                trace_df = tfile
                
        elif fmt == 'hdf5':
            f = h5py.File(os.path.join(trace_arrays_src, dfn), 'r')
            if trace_type == 'processed':
                trace_df = pd.DataFrame(f['corrected'][:])
                if dff:
                    trace_df /= pd.DataFrame(f['F0'][:])
            elif trace_type == 'raw':
                trace_df = pd.DataFrame(f['raw'][:])
            f.close()

        nframes, nrois = trace_df.shape
        
        print "... ... smoothing"
        if user_test:
            while True:
                smoothed_df = trace_df.apply(smooth_traces, frac=frac, missing='drop')
                print "Selected frac %0.2f to use for smoothing trace..." % (frac)
                nframes_to_show = int(round(1500.))
                pl.figure(figsize=(15,6))
                pl.ion()
                pl.show()
                test_smooth_frac(trace_df, smoothed_df, nframes_to_show, dff=dff, test_roi=test_roi)
                pl.pause(0.001)
                frac_sel = raw_input('Press <Y> if smoothing is good. Otherwise, select FRAC to use as smoothing window: ')
                
                if frac_sel=='Y':
                    pl.savefig(os.path.join(smooth_fig_dir, '%s_smoothing_%s.png' % (test_roi, str(frac))))
                    pl.close()
                    user_test=False
                    break
                else:
                    frac = float(frac_sel)
                    pl.close()
        else:
            smoothed_df = trace_df.apply(smooth_traces, frac=frac, missing='drop')
            nframes_to_show = int(round(1500.))
            pl.figure(figsize=(15,6))
            test_smooth_frac(trace_df, smoothed_df, nframes_to_show, dff=dff, test_roi=test_roi)
            pl.savefig(os.path.join(smooth_fig_dir, '%s_smoothing_%s_%s.png' % (test_roi, str(frac), os.path.splitext(dfn)[0])))
            pl.close()
            print "saved smoothing example, %s" % test_roi
        
        out_fname = dfn.replace(trace_type, 'smoothed%s' % str(frac).split('.')[-1])
        if fmt == 'pkl':
            with open(os.path.join(smoothed_trace_arrays_dir, out_fname), 'wb') as o:
                pkl.dump(smoothed_df, o, protocol=pkl.HIGHEST_PROTOCOL)
        elif fmt == 'hdf5':
            f = h5py.File(os.path.join(smoothed_trace_arrays_dir, out_fname), 'a')
            for k in f.keys():
                del f[k]
            sa, sa_type = df_to_sarray(smoothed_df)
            f.create_dataset('smoothed', data=sa, dtype=sa_type)
            f.close()
            
        print "Saved %s" % out_fname
    
    return smoothed_trace_arrays_dir        

#%
    
def get_smoothed_run(traceid_dir, trace_type='processed', dff=False, frac=0.01, create_new=False, fmt='hdf5', user_test=False, test_roi='roi00001'):
    xdata_df=None;
    # Set up paths to look for saved dataframes:
    data_array_dir = os.path.join(traceid_dir, 'data_arrays')
    if not os.path.exists(data_array_dir):
        os.makedirs(data_array_dir)

    # Data source files for collated dataframe:
    if dff:
        trace_array_type = os.path.join(traceid_dir, 'files', 'trace_arrays_%s_dff_smoothed%s' % (trace_type, str(frac).split('.')[-1]))
        xdata_fpath = os.path.join(data_array_dir, 'roiXtrials_%s.%s' % (trace_array_type, fmt)) # processed OR raw
        trace_arrays_dir = os.path.join(traceid_dir, 'files', trace_array_type)
    else:
        trace_array_type = os.path.join(traceid_dir, 'files', 'trace_arrays_%s_smoothed%s' % (trace_type, str(frac).split('.')[-1]))
        xdata_fpath = os.path.join(data_array_dir, 'roiXtrials_%s.%s' % (trace_array_type, fmt)) # processed OR raw
        trace_arrays_dir = os.path.join(traceid_dir, 'files', trace_array_type)

    if not os.path.exists(trace_arrays_dir):
        os.makedirs(trace_arrays_dir)
        
    # First, check that there are data arrays have been extract from hdf5 output (traces/get_traces.py):
    n_orig_tiffs = len([r for r in os.listdir(os.path.join(traceid_dir, 'files')) if r.endswith('hdf5')])
    n_src_dataframes = len([r for r in os.listdir(trace_arrays_dir) if 'File' in r and r.endswith(fmt)])
    if not n_orig_tiffs == n_src_dataframes or create_new is True:
        print "SMOOTHING trace arrays from .tif files (smoothing frac: %.2f)" % frac
        smooth_trace_arrays(traceid_dir, trace_type='processed', frac=frac,  dff=dff, fmt=fmt, user_test=user_test, test_roi=test_roi)
        
    if not create_new:
        try:
            if fmt == 'pkl':
                with open(xdata_fpath, 'rb') as f:
                    dfile = pkl.load(f)
                xdata_df = dfile['xdata']
                print "Loaded XDATA."
            
            elif fmt == 'hdf5':
                f = h5py.File(xdata_fpath, 'r')
                xdata_df = pd.DataFrame(f['xdata'][:])
                print "Loaded XDATA."
                f.close()
                
            return xdata_df
        
        except Exception as e:
            print "Unable to load raw data path: %s"
            print "Re-collating raw data..."

    # Create the .pkl file for rawXtrials:
    _, xdata_df, _ = collate_trials(trace_arrays_dir, dff=dff, smoothed=True, fmt=fmt)
    
    return xdata_df

    

#%%
def get_processed_run(traceid_dir, quantile=0.20, window_size_sec=None, create_new=False, fmt='hdf5', user_test=False, nonnegative=False):
    
    xdata_df=None; labels_df=None; F0_df=None;
    # Set up paths to look for saved dataframes:
    data_array_dir = os.path.join(traceid_dir, 'data_arrays')
    if not os.path.exists(data_array_dir):
        os.makedirs(data_array_dir)
    xdata_fpath = os.path.join(data_array_dir, 'roiXtrials_processed.%s' % fmt) # processed OR raw
    
    # Data source files for collated dataframe:
    trace_arrays_dir = os.path.join(traceid_dir, 'files', 'trace_arrays_processed')
    if not os.path.exists(trace_arrays_dir):
        os.makedirs(trace_arrays_dir)
        
    # First, check that there are data arrays have been extract from hdf5 output (traces/get_traces.py):
    n_orig_tiffs = len([r for r in os.listdir(os.path.join(traceid_dir, 'files')) if r.endswith('hdf5')])
    n_src_dataframes = len([r for r in os.listdir(trace_arrays_dir) if 'File' in r and r.endswith(fmt)])
    if not n_orig_tiffs == n_src_dataframes or create_new is True:
        print "Extracting PROCESSED trace arrays from .tif files."
        process_trace_arrays(traceid_dir, window_size_sec=window_size_sec, quantile=quantile, fmt=fmt, user_test=user_test, nonnegative=nonnegative)
        
    if not create_new:
        try:
            if fmt == 'pkl':
                with open(xdata_fpath, 'rb') as f:
                    dfile = pkl.load(f)
                xdata_df = dfile['xdata']
                labels_df = dfile['labels']
                F0_df = dfile['F0']
                print "Loaded XDATA."
            
            elif fmt == 'hdf5':
                hfile = h5py.File(xdata_fpath, 'r')
                xdata_df = pd.DataFrame(hfile['xdata'][:])
                labels_df = pd.DataFrame(hfile['labels'][:])
                F0_df = pd.DataFrame(hfile['F0'][:])
                print "Loaded XDATA."
                
            return labels_df, xdata_df, F0_df
        
        except Exception as e:
            print "Unable to load raw data path: %s"
            print "Re-collating raw data..."

    # Create the .pkl file for rawXtrials:
    labels_df, xdata_df, F0_df = collate_trials(trace_arrays_dir, dff=False, smoothed=False, fmt=fmt)
    
    return labels_df, xdata_df, F0_df


def test_drift_correction(raw_df, F0_df, corrected_df, nframes_to_show, test_roi='roi00001'):
    
    #test_roi = 'roi%05d' % int(roi_id)
    # Check data:
#    nframes_per_trial = run_info['nframes_per_trial']
#    ntrials_per_file = run_info['ntrials_total'] / run_info['nfiles']
    #pl.figure(figsize=(15,6))
    pl.plot(raw_df[test_roi][0:nframes_to_show], label='raw')
    pl.plot(F0_df[test_roi][0:nframes_to_show], label='drift')
    pl.plot(corrected_df[test_roi][0:nframes_to_show], label='corrected')
    pl.legend()
    pl.draw()
    pl.pause(0.001)

def process_trace_arrays(traceid_dir, window_size_sec=None, quantile=0.08, create_new=False, 
                             fmt='pkl', user_test=False, test_roi='roi00001', nonnegative=False):
    '''
    Calculate F0 for each ROI by .tif file (continuous time points). These
    trace "chunks" can later be parsed and combined into a dataframe for trial
    analyses, but smoothing should be done within continuous periods of shutter
    on/off (i.e., a literal 'acquisition' in SI).
    
    If window_size_sec unspecified, then default is to use 3 trials for window size.
    '''
    
    # Create fig dir to save example traces of drift-correction:
    drift_fig_dir = os.path.join(traceid_dir, 'files', 'processing_examples', 'drift_correction')
    if not os.path.exists(drift_fig_dir):
        os.makedirs(drift_fig_dir)
        
    # Get frame rate from SI info:
    run_dir = traceid_dir.split('/traces')[0]
    run = os.path.split(run_dir)[-1]
    with open(os.path.join(run_dir, '%s.json' % run), 'r') as f:
        scan_info = json.load(f)
    framerate = scan_info['frame_rate']
    print "Got framerate from RUN INFO:", framerate
        
#    if window_size_sec is None:
#        paradigm_dir = os.path.join(traceid_dir.split('/traces')[0], 'paradigm')
#        parsed_frames_fpath = [os.path.join(paradigm_dir, pfn) for pfn in os.listdir(paradigm_dir) if 'parsed_frames_' in pfn][0]
#        parsed_frames = h5py.File(parsed_frames_fpath, 'r')
#    
#        trial_list = sorted(parsed_frames.keys(), key=natural_keys)
#        print "There are %i total trials across all .tif files." % len(trial_list)
#        stimdurs = list(set([parsed_frames[t]['frames_in_run'].attrs['stim_dur_sec'] for t in trial_list]))
#        itidurs = list(set([parsed_frames[t]['frames_in_run'].attrs['iti_dur_sec'] for t in trial_list]))
#        assert len(stimdurs)==1, "More than 1 unique value for stim dur found in parsed_frames_ file!"
#        assert len(itidurs)==1, "More than 1 unique value for iti dur found in parsed_frames_ file!"
#        trial_dur_sec = stimdurs[0] + itidurs[0]
#        window_size_sec = trial_dur_sec * 3
#        #nframes_on = round(int(stimdurs[0] * framerate))
#        #nframes_iti = round(int(itidurs[0] * framerate))
#        #nframes_trial = nframes_on + nframes_iti
#        parsed_frames.close()
#        print "Using default window size of 3 trials (total is %i sec)" % trial_dur_sec
#        
    # Create output dir to save processed trace_arrays:
    processed_trace_arrays_dir = os.path.join(traceid_dir, 'files', 'trace_arrays_processed')
    if not os.path.exists(processed_trace_arrays_dir) or create_new:
        os.makedirs(processed_trace_arrays_dir)
        
        
    # First, make sure the raw trace arrays have been extracted:
    n_orig_tiffs = len([r for r in os.listdir(os.path.join(traceid_dir, 'files')) if r.endswith('hdf5')])
    raw_trace_arrays_dir = os.path.join(traceid_dir, 'files', 'trace_arrays_raw')
    try:
        rtrace_fns = [f for f in os.listdir(raw_trace_arrays_dir) if 'rawtraces' in f and f.endswith(fmt)]
        assert len(rtrace_fns)==n_orig_tiffs, "Unable to find correct set of raw trace array files (format %s)" % fmt
    except Exception as e:
        print "Creating raw data arrays from extracted .hdf5 files."
        raw_hdf_to_dataframe(traceid_dir, fmt=fmt)
        rtrace_fns = [f for f in os.listdir(raw_trace_arrays_dir) if 'rawtraces' in f and f.endswith(fmt)]
        

    # Load raw trace arrays from which to calculate drift:    
    for dfn in rtrace_fns:
        if fmt == 'pkl':            
            with open(os.path.join(raw_trace_arrays_dir, dfn),'rb') as f:
                raw_df = pkl.load(f)
        elif fmt == 'hdf5':
            rawfile = h5py.File(os.path.join(raw_trace_arrays_dir, dfn), 'r')
            raw_df = pd.DataFrame(rawfile['raw'][:])
                
        dfs = {}
        # Make data non-negative:
        raw_xdata = np.array(raw_df)
        if raw_xdata.min() < 0 and nonnegative is True:
            print "Making data non-negative"
            raw_xdata = raw_xdata - raw_xdata.min()
            raw_df = pd.DataFrame(raw_xdata, columns=raw_df.columns.tolist(), index=raw_df.index)
        
        # Get BASELINE from rolling window:
        print "... and extracting %i percentile as F0" % (quantile*100)

        if user_test:
            while True:
                print "Selected %i sec to use for calculating rolling %.3f percentile..." % (window_size_sec, quantile)
                corrected_df, F0_df = get_rolling_baseline(raw_df, window_size_sec*framerate, quantile=quantile)
                nframes_to_show = int(round(window_size_sec*framerate*10))
                pl.figure()
                pl.ion()
                pl.show()
                test_drift_correction(raw_df, F0_df, corrected_df, nframes_to_show, test_roi=test_roi)
                pl.show()
                pl.pause(0.001)
                print "Showing initial drift correction (quantile: %.2f)" % quantile
                print "Mean baseline for all ROIs:", np.mean(np.mean(F0_df, axis=0))
                window_sel = raw_input('Press <Y> if F0 calculation is good. Otherwise, select NSEC to use as window: ')
                quantile_sel = raw_input('Press <Y> if F0 drift is good. Otherwise, select QUANTILE to use as baseline (default: 0.08): ')
                
                if window_sel=='Y' and quantile_sel=='Y':
                    pl.savefig(os.path.join(drift_fig_dir, '%s_drift_correction.png' % test_roi))
                    pl.close()
                    user_test=False
                    break
                else:
                    window_size_sec = float(window_sel)
                    quantile = float(quantile_sel)
                    pl.close()
        else:
            corrected_df, F0_df = get_rolling_baseline(raw_df, window_size_sec*framerate, quantile=quantile)
            print "Showing initial drift correction (quantile: %.2f)" % quantile
            print "Min value for all ROIs:", np.min(np.min(corrected_df, axis=0))

            nframes_to_show = int(round(window_size_sec*framerate*10))
            pl.figure(figsize=(20,6))
            test_drift_correction(raw_df, F0_df, corrected_df, nframes_to_show, test_roi=test_roi)
            pl.savefig(os.path.join(drift_fig_dir, '%s_drift_correction_%s.png' % (test_roi, os.path.splitext(dfn)[0])))
            pl.close()
            print "saved drift-correction example, %s" % test_roi
        

                
#        processing_info = {'window_size': window_size_sec*framerate,
#                           'window_size_sec': window_size_sec,
#                           'framerate': framerate,
#                           'quantile': quantile,
#                           'nonnegative': nonnegative,
#                           }
#        
        # Save traces:
        out_fname = dfn.replace('rawtraces', 'processed')
        if fmt=='hdf5':
            f = h5py.File(os.path.join(processed_trace_arrays_dir, out_fname), 'a')
            for k in f.keys():
                del f[k]
            sa, saType = df_to_sarray(corrected_df)
            f.create_dataset('corrected', data=sa, dtype=saType)
            sa, saType = df_to_sarray(F0_df)
            f.create_dataset('F0', data=sa, dtype=saType)
           
            f.create_dataset('frames_indices', data=rawfile['frames_indices'])
            for slicekey in rawfile['rois_by_slice'].keys():
                f.create_dataset('rois_by_slice/%s' % slicekey, data=rawfile['rois_by_slice'][slicekey])

 
            f.close()
            rawfile.close()

        elif fmt=='pkl':   
            dfs = {'corrected': corrected_df, 'F0': F0_df}
            with open(os.path.join(raw_trace_arrays_dir, out_fname), 'wb') as o:
                pkl.dump(dfs, o, protocol=pkl.HIGHEST_PROTOCOL)
                
#        # Also save output info:
#        with open(os.path.join(processed_trace_arrays_dir, 'processing_info.json'), 'w') as f:
#            json.dump(processing_info, f, indent=4)
#        
    return processed_trace_arrays_dir        
    

    
def get_rolling_baseline(Xdf, window_size, quantile=0.08):
        
    #window_size_sec = (nframes_trial/framerate) * 2 # decay_constant * 40
    #decay_frames = window_size_sec * framerate # decay_constant in frames
    #window_size = int(round(decay_frames))
    #quantile = 0.08
    window_size = int(round(window_size))
    Fsmooth = Xdf.apply(rolling_quantile, args=(window_size, quantile))
    Xdata = (Xdf - Fsmooth)
    #Xdata = np.array(Xdata_tmp)
    
    return Xdata, Fsmooth


# Use cnvlib.smoothing functions to deal get mirrored edges on rolling quantile:
def rolling_quantile(x, width, quantile):
    """Rolling quantile (0--1) with mirrored edges."""
    x, wing, signal = check_inputs(x, width)
    rolled = signal.rolling(2 * wing + 1, 2, center=True).quantile(quantile)
    return np.asfarray(rolled[wing:-wing])


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



#%%
    
def collate_trials(trace_arrays_dir, dff=False, smoothed=False, fmt='hdf5', nonnegative=False):
    
    xdata_df=None; F0_df=None; labels_df=None
    
    # Load TID params to see if any excluded tifs:
    traceid_dir = trace_arrays_dir.split('/files')[0] #os.path.split(trace_arrays_dir)[0]
    traceid = os.path.split(traceid_dir)[-1].split('_')[0]
    trace_basedir = os.path.split(traceid_dir)[0] 
    print "Loading %s from dir: %s" % (traceid, trace_basedir)
    tid_fpath = glob.glob(os.path.join(trace_basedir, '*.json'))[0]
    with open(tid_fpath, 'r') as f:
        tids = json.load(f)
    excluded_tifs = tids[traceid]['PARAMS']['excluded_tiffs']
    print "*** Excluding:", excluded_tifs

    # Set up output dir for collated trials (combine all trials from each .tif file)
    trace_arrays_type = os.path.split(trace_arrays_dir)[-1]    
    roixtrial_fn = 'roiXtrials_%s.%s' % (trace_arrays_type, fmt) # Procssed or SMoothed -- output file
    traceid_dir = trace_arrays_dir.split('/files')[0]
    data_array_dir = os.path.join(traceid_dir, 'data_arrays')
    if not os.path.exists(data_array_dir):
        os.makedirs(data_array_dir)
        
    # Get SCAN IMAGE info for run:
    run_dir = traceid_dir.split('/traces')[0]
    run = os.path.split(run_dir)[-1]
    with open(os.path.join(run_dir, '%s.json' % run), 'r') as fr:
        scan_info = json.load(fr)
    all_frames_tsecs = np.array(scan_info['frame_tstamps_sec'])
    nslices_full = len(all_frames_tsecs) / scan_info['nvolumes']
    nslices = len(scan_info['slices'])
    if scan_info['nchannels']==2:
        all_frames_tsecs = np.array(all_frames_tsecs[0::2])

#    if nslices_full > nslices:
#        # There are discard frames per volume to discount
#        subset_frame_tsecs = []
#        for slicenum in range(nslices):
#            subset_frame_tsecs.extend(frame_tsecs[slicenum::nslices_full])
#        frame_tsecs = np.array(sorted(subset_frame_tsecs))
    print "N tsecs:", len(all_frames_tsecs)
    framerate = scan_info['frame_rate']
    volumerate = scan_info['volume_rate']
    nvolumes = scan_info['nvolumes']
    nfiles = scan_info['ntiffs']


    # Get alignment specs:


    # Get volume indices to assign frame numbers to volumes:
    vol_ixs_tif = np.empty((nvolumes*nslices_full,))
    vcounter = 0
    for v in range(nvolumes):
        vol_ixs_tif[vcounter:vcounter+nslices_full] = np.ones((nslices_full, )) * v
        vcounter += nslices_full
    vol_ixs_tif = np.array([int(v) for v in vol_ixs_tif])
    vol_ixs = []
    vol_ixs.extend(np.array(vol_ixs_tif) + nvolumes*tiffnum for tiffnum in range(nfiles))
    vol_ixs = np.array(sorted(np.concatenate(vol_ixs).ravel()))


    # Load MW info to get stimulus details:
    paradigm_dir = os.path.join(run_dir, 'paradigm')
    mw_fpath = [os.path.join(paradigm_dir, m) for m in os.listdir(paradigm_dir) if 'trials_' in m and m.endswith('json')][0]
    with open(mw_fpath,'r') as m:
        mwinfo = json.load(m)
    pre_iti_sec = round(mwinfo[mwinfo.keys()[0]]['iti_dur_ms']/1E3) 
    nframes_iti_full = int(round(pre_iti_sec * volumerate))

    with open(os.path.join(paradigm_dir, 'stimulus_configs.json'), 'r') as s:
        stimconfigs = json.load(s)
    if 'frequency' in stimconfigs[stimconfigs.keys()[0]].keys():
        stimtype = 'gratings'
    elif 'fps' in stimconfigs[stimconfigs.keys()[0]].keys():
        stimtype = 'movie'
    else:
        stimtype = 'image'
        
    # Fix MW trial info because the way protocol created is wonky:
#    unique_stim_durs = list(set([round(mwinfo[t]['stim_dur_ms']/1E3) for t in mwinfo.keys()]))
#    if stimtype == 'gratings' and len(unique_stim_durs) > 1:
#        mwinfo = adjust_mw_stim_values(mwinfo)
#        
    # For movies, remove last trial:
    skip_last_trial = False
    acqdir = os.path.split(run_dir)[0]
    session = os.path.split(os.path.split(acqdir)[0])[-1]
    if (stimtype == 'movie' and int(session) < 20180525): #or int(session)==20190424:
        print "Skipping last trial..."
        skip_last_trial = True

    for conf, params in stimconfigs.items():
        if 'filename' in params.keys():
            params.pop('filename')
        stimconfigs[conf] = params
        
    
    # Load frames--trial info:
    # -----------------------------------------------------

    parsed_frames_fpath = [os.path.join(paradigm_dir, pfn) for pfn in os.listdir(paradigm_dir) if 'parsed_frames_' in pfn][0]
    parsed_frames = h5py.File(parsed_frames_fpath, 'r')
    
    trial_list = sorted(parsed_frames.keys(), key=natural_keys)
    print "There are %i total trials across all .tif files." % len(trial_list)

   
    print "Collating trials across all files from %s" % trace_arrays_type
    trace_arrays_dir = os.path.join(traceid_dir, 'files', trace_arrays_type)
    
    # Check if frame indices are indexed relative to full run (all .tif files)
    # or relative to within-tif frames (i.e., a "block")
    block_indexed = True
    if all([all(parsed_frames[t]['frames_in_run'][:] == parsed_frames[t]['frames_in_file'][:]) for t in trial_list]):
        block_indexed = False
        print "Frame indices are NOT block indexed"
        
    trace_fns = sorted([f for f in os.listdir(trace_arrays_dir) if 'File' in f and f.endswith(fmt)], key=natural_keys)
    print "Found %i files to collate." % len(trace_fns)
    
    frame_df_list = []
    drift_df_list = []
    frame_times = []
    trial_ids = []
    config_ids = []
    #relative_frame_indices = []
    for fidx, dfn in enumerate(trace_fns):
        curr_file = str(re.search(r"File\d{3}", dfn).group())
        if curr_file in excluded_tifs:
            print "... skipping..."
            continue

        file_F0_df = None; file_df = None;
        if fmt == 'pkl':
            print "Collating... %s" % dfn
            with open(os.path.join(trace_arrays_dir, dfn), 'rb') as fi:
                f = pkl.load(fi)
            if 'corrected' in f.keys():
                file_df = f['corrected']; file_F0_df = f['F0']; trace_type = 'corrected';
            else:
                file_df = f; file_F0_df = None; trace_type = f.keys()[0];
        elif fmt == 'hdf5':
            f = h5py.File(os.path.join(trace_arrays_dir, dfn), 'a')
            if 'corrected' in f.keys():
                file_df = pd.DataFrame(f['corrected'][:])
                file_F0_df = pd.DataFrame(f['F0'][:])
                trace_type = 'corrected'
            else:
                trace_type = 'raw' #.keys()[0]
                file_df = pd.DataFrame(f[trace_type][:])
                file_F0_df = None
            # Get frame indices corresponding to slices:
            frames_to_select = pd.DataFrame(f['frames_indices'][:])
            rois_by_slice = dict((curr_slice, list(roi_list[:])) for curr_slice, roi_list in f['rois_by_slice'].items())


        if trace_type=='raw' and nonnegative and np.array(file_df).min() < 0:
            print "making nonnegative..."
            file_df -= np.array(file_df).min()

        # Get all trials contained in current .tif file:
        tmp_trials_in_block = sorted([t for t in trial_list \
                                  if parsed_frames[t]['frames_in_file'].attrs['aux_file_idx'] == fidx], key=natural_keys)
        # 20181016 BUG: ignore trials that are BLANKS:
        trials_in_block = sorted([t for t in tmp_trials_in_block if mwinfo[t]['stimuli']['type'] != 'blank'], key=natural_keys)
   
        # Only grab subset of frame_tsecs that correpsond to data frames:
        frame_tstamps = dict()
        for slicekey, roilist in rois_by_slice.items():
            frame_tstamps[slicekey] = all_frames_tsecs[frames_to_select[slicekey].values]
 
             
        if len(trials_in_block) == 0:
            continue

        # Get ALL frames corresponding to trial epochs:
        # -----------------------------------------------------
        #print "--> first trials:", trials_in_block[0:5]        
        all_frames_in_trials = np.hstack([np.array(parsed_frames[t]['frames_in_file']) \
                                   for t in trials_in_block])
        print "N frames in curr trials:", len(all_frames_in_trials)
        print "Last frame idx:", all_frames_in_trials[-1]

        stim_onset_idxs = np.array([parsed_frames[t]['frames_in_file'].attrs['stim_on_idx'] \
                                    for t in trials_in_block])
        #print "diff bw trials:", np.diff(stim_onset_idxs) 
        #stim_offset_idxs = np.array([mwinfo[t]['frame_stim_off'] for t in trials_in_block])

 
        # Since we are cycling thru FILES readjust frame indices to match within-file, rather than across-files.
        # -------------------------------------------------------
        # block_frame_offset set in extract_paradigm_events (MW parsing) 
        # supposedly set this up to deal with skipped tifs?
        if 'block_frame_offset' not in mwinfo[trials_in_block[0]].keys():
            frame_shift = 0
        else:
            frame_shift = mwinfo[trials_in_block[0]]['block_frame_offset']
        if block_indexed is False:
#            # Frame shift by "first" frame idx of current tif:
#            nframes_pre = int(round(parsed_frames['trial00001']['frames_in_run'].attrs['baseline_dur_sec'] * volumerate))
#            iti_remainder = nframes_iti_full - nframes_pre
#            alignment_start_offset = all_frames_in_trials.min()
#            all_frames_in_trials = all_frames_in_trials - alignment_start_offset + iti_remainder 
#            stim_onset_idxs = stim_onset_idxs - alignment_start_offset + iti_remainder            
#            print "-- shifted all_frames_in_trials, last fr index:", all_frames_in_trials[-1]

            all_frames_in_trials = all_frames_in_trials - len(all_frames_tsecs)*fidx - frame_shift  

            if all_frames_in_trials[-1] >= len(all_frames_tsecs):
                print 'File: %i' % fidx, len(all_frames_tsecs)
                print "*** %i extra frames..." % (all_frames_in_trials[-1] - len(all_frames_tsecs))
                #print all_frames_in_trials[-10:]
                #last_ix = np.where(all_frames_in_trials==len(all_frames_tsecs)-1)[0][0]
                #all_frames_in_trials = all_frames_in_trials[0:last_ix] 
            stim_onset_idxs = stim_onset_idxs - len(all_frames_tsecs)*fidx - frame_shift 
            #stim_offset_idxs = stim_offset_idxs - len(all_frames_tsecs)*fidx - frame_shift 
        
        # Need to deal with skipped trials WITHIN tifs?:
        # print "File ix: %i - all_frames_in_trials:" % fidx, all_frames_in_trials[0:10]   
        

 
        # Since multiple slices might be acquired, convert frame-reference to volume-reference. Only select first frame for each volume.
        # -------------------------------------------------------
        first_plane_frames = frames_to_select['Slice01'].values.tolist()
         
        #kept_frames_in_trials = [frame_ix for frame_ix in all_frames_in_trials if frame_ix in first_plane_frames]
        #frames_in_trials = [first_plane_frames.index(fr) for fr in kept_frames_in_trials]   
        #frames_in_trials = sorted(list(set(vol_ixs_tif[all_frames_in_trials])))

        # Don't take unique values, since stim period of trial N can be ITI of trial N-1
        frames_in_trials = np.empty(all_frames_in_trials.shape, dtype=int)
        actual_frames = [i for i in all_frames_in_trials if i < len(vol_ixs_tif)]
        frames_in_trials[0:len(actual_frames)] = vol_ixs_tif[actual_frames] #vol_ixs_tif[all_frames_in_trials]

        stim_onset_idxs_adjusted = vol_ixs_tif[stim_onset_idxs]
        stim_onset_idxs = copy.copy(stim_onset_idxs_adjusted)


        print "%s - N trials in block: %i (%i frames)" % (curr_file, len(trials_in_block), len(frames_in_trials))
        #print "Found %i chunks where frame interval > 1" % len(np.where(np.diff(frames_in_trials) > 1)[0]) 

        # Get Stimulus info for each trial:        
        # -----------------------------------------------------
        #excluded_params = ['filehash', 'stimulus', 'type', 'rotation_range', 'phase'] # Only include param keys saved in stimconfigs
        excluded_params = [k for k in mwinfo[trials_in_block[0]]['stimuli'].keys() if k not in stimconfigs['config001'].keys()]

        curr_trial_stimconfigs = [dict((k,v) for k,v in mwinfo[t]['stimuli'].iteritems() \
                                       if k not in excluded_params) for t in trials_in_block]
        varying_stim_dur = False
        # Add stim_dur if included in stim params:
        if 'stim_dur' in stimconfigs[stimconfigs.keys()[0]].keys():
            varying_stim_dur = True
            for ti, trial in enumerate(sorted(trials_in_block, key=natural_keys)):
                #print curr_trial_stimconfigs[ti]
                curr_trial_stimconfigs[ti]['stim_dur'] = round(mwinfo[trial]['stim_dur_ms']/1E3, 1)
        

        # #### Parse full file into "trials" w/ frame indices:
        # ------------------------------------------------
       
        # Turn frame_tsecs into RELATIVE tstamps (to stim onset):
        first_plane_tstamps = all_frames_tsecs[first_plane_frames]
        print "--> N tstamps:", len(first_plane_tstamps)
        trial_tstamps = first_plane_tstamps[frames_in_trials[0:len(actual_frames)]] #all_frames_tsecs[frames_in_trials] #indices]  
        if len(trial_tstamps) < len(frames_in_trials):
            print "padding trial tstamps array..."
            trial_tstamps = np.pad(trial_tstamps, (0, len(frames_in_trials)-len(trial_tstamps)), mode='constant', constant_values=np.nan)
        print "trial tstamps: size", len(trial_tstamps)

        min_frame_interval = 1 #list(set(np.diff(frames_to_select['Slice01'].values)))  # 1 if not slices
        #assert len(min_frame_interval) == 1, "More than 1 min frame interval found... %s" % str(min_frame_interval)
        #min_frame_interval = min_frame_interval[0]
        if varying_stim_dur:
           
            trial_end_idxs = np.where(np.diff(frames_in_trials) > min_frame_interval)[0]
            trial_end_idxs = np.append(trial_end_idxs, len(trial_tstamps)-1)
            trial_start_idxs = [0]
            trial_start_idxs.extend([i+min_frame_interval for i in trial_end_idxs[0:-1]])
            
            
            relative_tsecs = []; #curr_trial_start_ix = 0;
            for ti, (trial_start_ix, trial_end_ix) in enumerate(zip(sorted(trial_start_idxs), sorted(trial_end_idxs))):
                
                stim_on_ix = stim_onset_idxs[ti]
                curr_tstamps = trial_tstamps[trial_start_ix:trial_end_ix+1]
                
                zeroed_tstamps = curr_tstamps - all_frames_tsecs[stim_on_ix]
                print ti, zeroed_tstamps[0]
                relative_tsecs.append(zeroed_tstamps)
                #prev_trial_end = trial_end_ix
                
            relative_tsecs = np.hstack(relative_tsecs)

        else:
            nframes_pre = int(round(parsed_frames['trial00001']['frames_in_run'].attrs['baseline_dur_sec'] * volumerate))
            nframes_post = int(round(parsed_frames['trial00001']['frames_in_run'].attrs['iti_dur_sec'] * volumerate))
            nframes_on = int(round(parsed_frames['trial00001']['frames_in_run'].attrs['stim_dur_sec'] * volumerate))
            nframes_per_trial = nframes_pre + nframes_on + nframes_post 
            # All trials have the same structure:
            reformat_tstamps = False
            #nframes_per_trial = len(frames_in_trials) / len(trials_in_block)
            print "nframes per trial:", nframes_per_trial
            print "N tstamps:", len(trial_tstamps)
            print "N trials in block:", len(trials_in_block)
            try: 
                tsec_mat = np.reshape(trial_tstamps, (len(trials_in_block), nframes_per_trial))
                # Subtract stim_on tstamp from each frame of each trial to get relative tstamp:
                tsec_mat -= np.tile(all_frames_tsecs[stim_onset_idxs].T, (tsec_mat.shape[1], 1)).T
               #print tsec_mat
                
            except Exception as e: #ValueError:
                print e
                #print tsec_mat
                reformat_tstamps = True
           
            if reformat_tstamps:
                print "REFORMATTING tstamps..."
                adjusted_frames_in_trials = []
                tsec_mat = np.empty((len(trials_in_block), nframes_per_trial))
                nframes_pre = int(round(parsed_frames['trial00001']['frames_in_run'].attrs['baseline_dur_sec'] * volumerate))
                nframes_post = int(round(parsed_frames['trial00001']['frames_in_run'].attrs['iti_dur_sec'] * volumerate))
                nframes_on = int(round(parsed_frames['trial00001']['frames_in_run'].attrs['stim_dur_sec'] * volumerate))
                for si, stimon in enumerate(stim_onset_idxs):
                    tsec_mat[si, :] = first_plane_tstamps[stimon-nframes_pre:stimon+(nframes_on+nframes_post)] - first_plane_tstamps[stimon]
                    adjusted_frames_in_trials.extend(np.arange(stimon-nframes_pre, stimon + (nframes_on+nframes_post)).tolist())

                frames_in_trials = copy.copy(adjusted_frames_in_trials)
                print "*warning* - adjusted trial frame indices (%i frs total)" % len(frames_in_trials)


           
            relative_tsecs = np.reshape(tsec_mat, (len(trials_in_block)*nframes_per_trial, ))

        # Get corresponding stimulus/trial labels for each frame in each trial:
        # --------------------------------------------------------------
        curr_config_ids = [k for trial_configs in curr_trial_stimconfigs \
                        for k,v in stimconfigs.iteritems() if v==trial_configs]
        # Convert frames_in_file to volume idxs:
        trial_frames_to_vols = dict((t, []) for t in trials_in_block)
        for t in trials_in_block: 
            frames_to_vols = parsed_frames[t]['frames_in_file'][:] 
            frames_to_vols = frames_to_vols - len(all_frames_tsecs)*fidx - frame_shift  
            #trial_vol_ixs = sorted(list(set(vol_ixs_tif[frames_to_vols])))
            
            actual_frames_in_trial = [i for i in frames_to_vols if i < len(vol_ixs_tif)]
            trial_vol_ixs = np.empty(frames_to_vols.shape, dtype=int)
            trial_vol_ixs[0:len(actual_frames_in_trial)] = vol_ixs_tif[actual_frames_in_trial]

#            frames_to_vols = parsed_frames[t]['frames_in_file'][:] - alignment_start_offset + iti_remainder
#            trial_vol_ixs = np.empty(frames_to_vols.shape, dtype=int)
#            actual_frames_in_trial = [i for i in frames_to_vols if i < len(vol_ixs_tif)]
#            trial_vol_ixs[0:len(actual_frames)] = vol_ixs_tif[actual_frames_in_trial]  
            if varying_stim_dur is False:
                trial_vol_ixs = trial_vol_ixs[0:nframes_per_trial]
            trial_frames_to_vols[t] = np.array(trial_vol_ixs)

        print "n trials in block:", len(trials_in_block)
        print "n configs in block:", len(curr_config_ids)
 
        config_labels = np.hstack([np.tile(conf, \
                            trial_frames_to_vols[t].shape) \
                            for conf, t in \
                            zip(curr_config_ids, trials_in_block)])
        trial_labels = np.hstack([np.tile(\
                            parsed_frames[t]['frames_in_run'].attrs['trial'], \
                            trial_frames_to_vols[t].shape) \
                            for t in trials_in_block])
    
                    
        currtrials_df = file_df.loc[frames_in_trials, :]  # DF (nframes_per_trial*ntrials_in_tiff X nrois)
        if file_F0_df is not None:
            currbaseline_df = file_F0_df.loc[frames_in_trials, :]
 
        # Add current block of trial info:
        frame_df_list.append(currtrials_df)
        if file_F0_df is not None:
            drift_df_list.append(currbaseline_df)


       
        frame_times.append(relative_tsecs)
        trial_ids.append(trial_labels)
        config_ids.append(config_labels)


    xdata_df = pd.concat(frame_df_list, axis=0).reset_index(drop=True)
    print "XDATA concatenated: %s" % str(xdata_df.shape)
    #print xdata_df.tail()

    if len(drift_df_list)>0:
        F0_df = pd.concat(drift_df_list, axis=0).reset_index(drop=True)
        
    # Also collate relevant frame info (i.e., labels):
    tstamps = np.hstack(frame_times)
    print tstamps.shape
    trials = np.hstack(trial_ids)
    configs = np.hstack(config_ids)
    if 'stim_dur' in stimconfigs[stimconfigs.keys()[0]].keys():
        stim_durs = np.array([stimconfigs[c]['stim_dur'] for c in configs])
    else:
        stim_durs = list(set([round(mwinfo[t]['stim_dur_ms']/1e3, 1) for t in trial_list]))
    nframes_on = np.array([int(round(dur*volumerate)) for dur in stim_durs])
    print "Nframes on:", nframes_on
    print "stim_durs (sec):", stim_durs
 

    if skip_last_trial:
        # Look in data_arrays dir to see if trials_to_dump were selected from
        # a previous run:
        dump_info_fpath = os.path.join(data_array_dir, 'dump_info.pkl')
        if os.path.exists(dump_info_fpath):
            with open(dump_info_fpath, 'rb') as d:
                dumpinfo = pkl.load(d)
        else:
            dumpinfo = {}
            # Need to remove trials for certain conditions since we're missing traces...
            counts = Counter(configs)
            dumpinfo['orig_counts'] = counts
            min_frames = min([v for k,v in counts.items()])
            conds_with_less = [k for k,v in counts.items() if v==min_frames]
            conds_to_lop = [k for k in counts.keys() if k not in conds_with_less]
            dumpinfo['conds_to_lop'] = conds_to_lop
            kept_indices = []
            dumpedtrials = {}
            for cond in counts.keys():
                currcond_idxs = np.where(configs==cond)[0]  # Original indices into the full array for current condition
                if cond in conds_to_lop:
                    curr_trials = trials[currcond_idxs]         # Trials 
                    trial_labels = np.array(sorted(list(set(curr_trials)), key=natural_keys)) # Get trial IDs in order
                    currcond_nframes = counts[cond]
                    nframes_over = currcond_nframes - min_frames
                    ntrials_to_remove = nframes_over/nframes_per_trial
                    ntrials_orig = currcond_nframes/nframes_per_trial
                    randomly_removed_trials = random.sample(range(0, ntrials_orig), ntrials_to_remove)
                    trials_to_dump = trial_labels[randomly_removed_trials]
                    cond_indices_to_keep = [i for i,trial in enumerate(curr_trials) if trial not in trials_to_dump]
                    kept_indices.append(currcond_idxs[cond_indices_to_keep])
                    dumpedtrials[cond] = trials_to_dump
                else:
                    kept_indices.append(currcond_idxs)
            keep = np.array(sorted(list(itertools.chain(*kept_indices))))
            dumpinfo['keep'] = keep
            dumpinfo['dumpedtrials'] = dumpedtrials
            
            with open(dump_info_fpath, 'wb') as d:
                pkl.dump(dumpinfo, d, protocol=pkl.HIGHEST_PROTOCOL)

        if len(dumpinfo['keep']) > 0:
            keep = np.array(dumpinfo['keep'])
            xdata_df = xdata_df.loc[keep, :].reset_index(drop=True)
            tstamps = tstamps[keep]
            trials = trials[keep]
            configs = configs[keep]
            stim_durs = stim_durs[keep]
            if F0_df is not None:
                F0_df = F0_df.loc[keep, :].reset_index(drop=True)
    

    print configs.shape
    print trials.shape
    print len(stim_durs) 
    # Turn paradigm info into dataframe:
    labels_df = pd.DataFrame({'tsec': tstamps, 
                              'config': configs,
                              'trial': trials,
                              'stim_dur': stim_durs #np.tile(stim_dur, trials.shape)
                              }, index=xdata_df.index)
    #print np.where([t for t in labels_df.groupby('trial')['tsec'].apply(np.array)][0] == 0)[0]
    #print [t for t in labels_df.groupby('trial')['tsec'].apply(np.array)]
    for t in sorted(list(set(labels_df['trial'])), key=natural_keys):
        onset = np.where(labels_df[labels_df['trial']==t]['tsec'].values==0)[0]
        if len(onset) == 0:
            break
            #print t
            #print labels_df['tsec']

    ons = [int(np.where(np.array(t)==0)[0]) for t in labels_df.groupby('trial')['tsec'].apply(np.array)]
    assert len(list(set(ons))) == 1, "Stim onset index has multiple values..."
    stim_on_frame = list(set(ons))[0]
    stim_ons_df = pd.DataFrame({'stim_on_frame': np.tile(stim_on_frame, (len(tstamps),)),
                                'nframes_on': nframes_on,
                                }, index=labels_df.index)
    labels_df = pd.concat([stim_ons_df, labels_df], axis=1)
    


    if fmt == 'pkl':
        dfile = {'xdata': xdata_df, 'labels': labels_df}
        if trace_type=='processed':
            dfile['F0'] = F0_df
        with open(os.path.join(data_array_dir, roixtrial_fn), 'wb') as f:
            pkl.dump(dfile, f, protocol=pkl.HIGHEST_PROTOCOL)
            
    elif fmt == 'hdf5':
        f = h5py.File(os.path.join(data_array_dir, roixtrial_fn),'a')
        for k in f.keys():
            del f[k]
        sa, saType = df_to_sarray(xdata_df)
        f.create_dataset('xdata', data=sa, dtype=saType)
        sa_labels, saType_labels = df_to_sarray(labels_df)
        f.create_dataset('labels', data=sa_labels, dtype=saType_labels)
        if trace_type == 'processed':
            sa, saType = df_to_sarray(F0_df)
            f.create_dataset('F0', data=sa, dtype=saType)
        f.close()
    
    return labels_df, xdata_df, F0_df


#    if skip_last_trial:
#        # Look in data_arrays dir to see if trials_to_dump were selected from
#        # a previous run:
#        dump_info_fpath = os.path.join(data_array_dir, 'dump_info.pkl')
#        if os.path.exists(dump_info_fpath):
#            with open(dump_info_fpath, 'rb') as d:
#                dumpinfo = pkl.load(d)
#        else:
#            dumpinfo = {}
#            # Need to remove trials for certain conditions since we're missing traces...
#            counts = Counter(configs)
#            dumpinfo['orig_counts'] = counts
#            min_frames = min([v for k,v in counts.items()])
#            conds_with_less = [k for k,v in counts.items() if v==min_frames]
#            conds_to_lop = [k for k in counts.keys() if k not in conds_with_less]
#            dumpinfo['conds_to_lop'] = conds_to_lop
#            kept_indices = []
#            dumpedtrials = {}
#            for cond in counts.keys():
#                currcond_idxs = np.where(configs==cond)[0]  # Original indices into the full array for current condition
#                if cond in conds_to_lop:
#                    curr_trials = trials[currcond_idxs]         # Trials 
#                    trial_labels = np.array(sorted(list(set(curr_trials)), key=natural_keys)) # Get trial IDs in order
#                    currcond_nframes = counts[cond]
#                    nframes_over = currcond_nframes - min_frames
#                    ntrials_to_remove = nframes_over/nframes_per_trial
#                    ntrials_orig = currcond_nframes/nframes_per_trial
#                    randomly_removed_trials = random.sample(range(0, ntrials_orig), ntrials_to_remove)
#                    trials_to_dump = trial_labels[randomly_removed_trials]
#                    cond_indices_to_keep = [i for i,trial in enumerate(curr_trials) if trial not in trials_to_dump]
#                    kept_indices.append(currcond_idxs[cond_indices_to_keep])
#                    dumpedtrials[cond] = trials_to_dump
#                else:
#                    kept_indices.append(currcond_idxs)
#            keep = np.array(sorted(list(itertools.chain(*kept_indices))))
#            dumpinfo['keep'] = keep
#            dumpinfo['dumpedtrials'] = dumpedtrials
#            
#            with open(dump_info_fpath, 'wb') as d:
#                pkl.dump(dumpinfo, d, protocol=pkl.HIGHEST_PROTOCOL)
#
#        if len(dumpinfo['keep']) > 0:
#            keep = np.array(dumpinfo['keep'])
#            xdata_df = xdata_df.loc[keep, :].reset_index(drop=True)
#            tstamps = tstamps[keep]
#            trials = trials[keep]
#            configs = configs[keep]
#            if F0_df is not None:
#                F0_df = F0_df.loc[keep, :].reset_index(drop=True)
#        
        
      #%%      


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
        
        

def test_file_smooth(traceid_dir, use_raw=False, ridx=0, fmin=0.001, fmax=0.02, save_and_close=True, output_dir='/tmp', quantile=0.08):
    '''
    Same as smooth_trace_arrays() but only does 1 file with specified 
    smoothing fraction. Plots figure with user-provided example ROI.
    '''
    
    tracefile_dir = os.path.join(traceid_dir, 'files')
    if use_raw:
        trace_arrays_dir = os.path.join(tracefile_dir, 'raw_trace_arrays')
    else:
        trace_arrays_dir = os.path.join(tracefile_dir, 'processed_trace_arrays')
    
    if os.path.exists(trace_arrays_dir):
        trace_fns = [f for f in os.listdir(trace_arrays_dir) if 'File' in f]
        
    if (not os.path.exists(trace_arrays_dir)) or (len(trace_fns) == 0):
        traceid = os.path.split(traceid_dir)[-1].split('_')[0]
        run_dir = os.path.split(os.path.split(traceid_dir)[0])[0]
        run = os.path.split(run_dir)[-1]
        acquisition_dir = traceid_dir.split('/%s' % run)[0]
        if use_raw:
            print "Creating raw trace arrays for: %s" % acquisition_dir
            print "Run: %s, Trace ID: %s" % (run, traceid)
            raw_hdf_to_dataframe(traceid_dir)
        else:
            print "Creating F0-subtracted arrays for: %s" % acquisition_dir
            print "Run: %s, Trace ID: %s" % (run, traceid)
            #process_trace_arrays(traceid_dir, quantile=quantile)
            process_trace_arrays(traceid_dir, window_size_sec=window_size_sec, quantile=quantile, fmt=fmt, user_test=user_test, nonnegative=nonnegative)

        trace_fns = [f for f in os.listdir(trace_arrays_dir) if 'File' in f]

    fmt = os.path.splitext(trace_fns[0])[-1]
    
    dfn = trace_fns[0]
    if fmt == '.pkl':            
        with open(os.path.join(trace_arrays_dir, dfn),'rb') as f:
            tfile = pkl.load(f)
            if 'F0' in tfile.keys():
                F0 = tfile['F0']
                tfile = tfile['processed']
            
    roi_id = 'roi%05d' % int(ridx+1)
    roi_trace = tfile[roi_id].values
    
    frac_range = np.linspace(fmin, fmax, num=8)
    fig, axes = pl.subplots(2,4, figsize=(30,6)) #pl.figure()

    for i, ax, in enumerate(axes.flat):
        
        filtered = np.apply_along_axis(smooth_traces, 0, roi_trace, frac=frac_range[i], missing='drop')
        
        ax.plot(xrange(len(filtered)), roi_trace, 'k', linewidth=0.5)
        ax.plot(xrange(len(filtered)), filtered, 'r', linewidth=1, alpha=0.8)
        ax.set_title('%.04f' % frac_range[i])

    pl.suptitle('%s_%s' % (roi_id, dfn))
    figstring = '%s_%s_smoothed_fmin%s_fmax%s' % (roi_id, dfn, str(fmin)[2:], str(fmax)[2:])
    pl.show()
    
    if save_and_close:
        pl.savefig(os.path.join(output_dir, '%s.png' % figstring))
        pl.close()
    
    return figstring


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


def get_traceid_from_acquisition(acquisition_dir, run, traceid):
    # Get paths to data source:
    print "Getting path info for single run dataset..."
    if 'cnmf' in traceid:
        if 'comb' in run:
            print "Combo run!"
            cnmf_dirs = glob.glob(os.path.join(acquisition_dir, run, 'traces', 'cnmf_*'))
        else:
            cnmf_dirs = glob.glob(os.path.join(acquisition_dir, run, 'traces', 'cnmf', 'cnmf_*'))
        cnmf_dir = [c for c in cnmf_dirs if os.path.split(c)[-1] == traceid]
        if len(cnmf_dir) == 0:
            print "Specified dir %s not found." % traceid
            cnmf_dirs = sorted(cnmf_dirs)
            for ci, cdir in enumerate(cnmf_dirs):
                print ci, os.path.split(cdir)[-1]
            selected_idx = input("Select IDX of dir to use: ")
            traceid_dir = cnmf_dirs[int(selected_idx)]
        else:
            assert len(cnmf_dir) == 1, "More than 1 unique cnmf datestr found...\n %s" % str(cnmf_dirs)
            traceid_dir = cnmf_dir[0]
    else:
        with open(os.path.join(acquisition_dir, run, 'traces', 'traceids_%s.json' % run), 'r') as f:
            tdict = json.load(f)
        if '_' in traceid and 'comb' not in run:
            traceid = traceid.split('_')[0]
        tracefolder = '%s_%s' % (traceid, tdict[traceid]['trace_hash'])
        traceid_dir = os.path.join(acquisition_dir, run, 'traces', tracefolder)
        
    return traceid_dir


#%%
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

#%%
def get_transforms(stimconfigs):
    
    if 'frequency' in stimconfigs[stimconfigs.keys()[0]] or 'ori' in stimconfigs[stimconfigs.keys()[0]]:
        stimtype = 'grating'
#    elif 'fps' in stimconfigs[stimconfigs.keys()[0]]:
#        stimtype = 'movie'
    else:
        stimtype = 'image'

    if 'position' in stimconfigs[stimconfigs.keys()[0]].keys() and ('xpos' not in stimconfigs[stimconfigs.keys()[0]].keys() or 'ypos' not in stimconfigs[stimconfigs.keys()[0]].keys()):
        # Need to reformat scnofigs:
        sconfigs = format_stimconfigs(stimconfigs)
    else:
        sconfigs = stimconfigs.copy()
    
    transform_dict = {'xpos': [x for x in list(set([sconfigs[c]['xpos'] for c in sconfigs.keys()])) if x is not None],
                       'ypos': [x for x in list(set([sconfigs[c]['ypos'] for c in sconfigs.keys()])) if x is not None],
                       'size': [x for x in list(set(([sconfigs[c]['size'] for c in sconfigs.keys()]))) if x is not None]
                       }
    
    if stimtype == 'image':
        transform_dict['yrot'] = list(set([sconfigs[c]['yrot'] for c in sconfigs.keys()]))
        transform_dict['morphlevel'] = list(set([sconfigs[c]['morphlevel'] for c in sconfigs.keys()]))
    else:
        transform_dict['ori'] = sorted(list(set([sconfigs[c]['ori'] for c in sconfigs.keys()])))
        transform_dict['sf'] = sorted(list(set([sconfigs[c]['sf'] for c in sconfigs.keys()])))
        if 'stim_dur' in sconfigs[sconfigs.keys()[0]].keys():
            transform_dict['direction'] = sorted(list(set([sconfigs[c]['direction'] for c in sconfigs.keys()])))
            transform_dict['duration'] = sorted(list(set([sconfigs[c]['stim_dur'] for c in sconfigs.keys()])))
        
    trans_types = [t for t in transform_dict.keys() if len(transform_dict[t]) > 1]

    object_transformations = {}
    for trans in trans_types:
        if stimtype == 'image':
            curr_objects = []
            for transval in transform_dict[trans]:
                curr_configs = [c for c,v in sconfigs.iteritems() if v[trans] == transval]
                tmp_obj = [list(set([sconfigs[c]['object'] for c in curr_configs])) for t in transform_dict[trans]]
                tmp_obj = list(itertools.chain(*tmp_obj))
                curr_objects.append(tmp_obj)
                
            if len(list(itertools.chain(*curr_objects))) == len(transform_dict[trans]):
                # There should be a one-to-one correspondence between object id and the transformation (i.e., morphs)
                included_objects = list(itertools.chain(*curr_objects))
#            elif trans == 'morphlevel':
#                included_objects = list(set(list(itertools.chain(*curr_objects))))
            else:
                included_objects = list(set(curr_objects[0]).intersection(*curr_objects[1:]))
        else:
            included_objects = transform_dict[trans]
            print included_objects
        object_transformations[trans] = included_objects

    return transform_dict, object_transformations



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
    print "configs:", configs
 
    # Split position into x,y:
    for config in stimconfigs.keys():
        stimconfigs[config]['xpos'] = None if configs[config]['position'][0] is None else round(configs[config]['position'][0], 2)
        stimconfigs[config]['ypos'] = None if configs[config]['position'][1] is None else round(configs[config]['position'][1], 2)
     
        stimconfigs[config]['size'] = None if configs[config]['scale'][0] is None else configs[config]['scale'][0]
        if 'aspect' in configs[config].keys() and stimconfigs[config]['size'] is not None:
            stimconfigs[config]['size'] = stimconfigs[config]['size'] / configs[config]['aspect']
        #stimconfigs[config].pop('position', None)
        stimconfigs[config].pop('scale', None)
        stimconfigs[config]['stimtype'] = stimtype
        if 'color' in configs[config].keys():
            stimconfigs[config]['luminance'] = round(configs[config]['color'], 3) if isinstance(configs[config]['color'], float) else None
        else:
            stimconfigs[config]['luminance'] = None
        
        # stimulus-type specific variables:
        if stimtype == 'gratings':
            stimconfigs[config]['sf'] = configs[config]['frequency']
            stimconfigs[config]['ori'] = configs[config]['rotation']
            stimconfigs[config].pop('frequency', None)
            stimconfigs[config].pop('rotation', None)
        else:
            transform_variables = ['object', 'xpos', 'ypos', 'size', 'yrot', 'morphlevel', 'stimtype', 'color']
            
            # Figure out Morph IDX for 1st and last anchor image:
            image_names = list(set([configs[c]['filename'] for c in configs.keys()]))
            if any([os.path.splitext(configs[k]['filename'])[-1]=='.png' for k in configs.keys()]):
            #if os.path.splitext(configs[configs.keys()[0]]['filename'])[-1] == '.png':
                if len(image_names) >= 2:
                    if any(['Blob_N1' in i for i in image_names]) and any(['Blob_N2' in i for i in image_names]):
                        mlevels = []
                        anchor1 = 0; anchor2 = 106
                else:
                    fns = [configs[c]['filename'] for c in configs.keys() if 'morph' in configs[c]['filename']]
                    mlevels = sorted(list(set([int(fn.split('_')[0][5:]) for fn in fns])))
            elif 'fps' in configs[configs.keys()[0]].keys():
                fns = [configs[c]['filename'] for c in configs.keys() if 'Blob_M' in configs[c]['filename']]
                mlevels = sorted(list(set([int(fn.split('_')[1][1:]) for fn in fns])))   
            
            if len(mlevels) > 0:
                if mlevels[-1] > 22:
                    anchor2 = 106
                else:
                    anchor2 = 22
            assert all([anchor2 > m for m in mlevels]), "Possibly incorrect morphlevel assignment (%i). Found morphs %s." % (anchor2, str(mlevels))

            if stimtype == 'image':
                imname = os.path.splitext(configs[config]['filename'])[0]
                if ('CamRot' in imname):
                    objectid = imname.split('_CamRot_')[0]
                    yrot = int(imname.split('_CamRot_y')[-1])
                    if 'N1' in imname or 'D1' in imname:
                        morphlevel = 0
                    elif 'N2' in imname or 'D2' in imname:
                        morphlevel = anchor2
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
                elif configs[config]['filename']=='' and configs[config]['stimulus']=='control':
                    objectid = 'control'
                    yrot = 0
                    morphlevel = -1

                    
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
                    morphlevel = anchor2
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


def save_stimulus_configs(trialdict, paradigm_dir):
#    paradigm_dir = os.path.split(trial_info['parsed_trials_source'])[0]
#    print paradigm_dir
#    trialdict = load_parsed_trials(trial_info['parsed_trials_source'])

    # Get presentation info (should be constant across trials and files):
    tmp_trial_list = sorted(trialdict.keys(), key=natural_keys)
    
    # 201810116 BUG -- ignore trials (and stimconfigs) that are "blanks"
    ignore_trials = [t for t in tmp_trial_list if trialdict[t]['stimuli']['type'] == 'blank']
    trial_list = sorted([t for t in tmp_trial_list if t not in ignore_trials], key=natural_keys)
    
    # Get all varying stimulus parameters:
    stimtype = list(set([trialdict[trial]['stimuli']['type'] for trial in trial_list]))
    #prin stimtype
    
    if any(['grating' in s for s in stimtype]):
        excluded_params = ['stimulus', 'type', 'rotation_range', 'phase']
    else: 
        excluded_params = ['filehash']
    config_list = [] 
    for trial in trial_list:
        curr_cfg = trialdict[trial]['stimuli'].copy()
        for vexcl in excluded_params:
            if vexcl in curr_cfg.keys():
                curr_cfg.pop(vexcl) 
        if 'filepath' in curr_cfg.keys():
            curr_cfg.update({'filename': os.path.split(curr_cfg['filepath'])[-1]})
            curr_cfg.pop('filepath')
        if curr_cfg not in config_list:
            config_list.append(curr_cfg)
 
    cfgs = pd.DataFrame(config_list) 
    configs = dict()
    for ci in cfgs.index.tolist():
        configs['config%03d' % int(ci+1)] = dict(cfgs.iloc[ci])

    print "---> Found %i unique stimulus configs." % len(configs.keys())
      
    # SAVE CONFIG info:
    config_fpath = os.path.join(paradigm_dir, 'stimulus_configs.json')
    with open(config_fpath, 'w') as f:
        json.dump(configs, f, sort_keys=True, indent=4)
    print "===== saved formated stimulus_configs to json ====="
    print config_fpath
    print "---------------------------------------------------"
    
    return configs, stimtype



#%%
#def save_stimulus_configs(trialdict, paradigm_dir):
#    # Get presentation info (should be constant across trials and files):
#    tmp_trial_list = sorted(trialdict.keys(), key=natural_keys)
#    
#    # 201810116 BUG -- ignore trials (and stimconfigs) that are "blanks"
#    ignore_trials = [t for t in tmp_trial_list if trialdict[t]['stimuli']['type'] == 'blank']
#    trial_list = sorted([t for t in tmp_trial_list if t not in ignore_trials], key=natural_keys)
#    
#    # Get all varying stimulus parameters:
#    stimtype = list(set([trialdict[trial]['stimuli']['type'] for trial in trial_list]))
#    #prin stimtype
#
#    if 'grating' in stimtype:
#        exclude_params = ['stimulus', 'type', 'rotation_range', 'phase']
#        stimparams = [k for k in trialdict[trial_list[0]]['stimuli'].keys() if k not in exclude_params]
#    else:
#        exclude_params = ['filehash', 'type']
#        stimparams = [k for k in trialdict[trial_list[0]]['stimuli'].keys() if k not in exclude_params]
#
#    # Determine whether there are varying stimulus durations to be included as cond:
##    unique_stim_durs = list(set([round(trialdict[t]['stim_dur_ms']/1E3) for t in trial_list]))
##    if len(unique_stim_durs) > 1:
##        stimparams.append('stim_dur')
#    stimparams = sorted(stimparams, key=natural_keys)
#    
#    # Get all unique stimulus configurations (ID, filehash, position, size, rotation):
#    allparams = []
#    for param in stimparams:
##        if param == 'stim_dur':
##            # Stim DUR:
##            currvals = [round(trialdict[trial]['stim_dur_ms']/1E3) for trial in trial_list]
#        if isinstance(trialdict[trial_list[0]]['stimuli'][param], list):
#            currvals = [tuple(trialdict[trial]['stimuli'][param]) for trial in trial_list]
#        else:
#            currvals = [trialdict[trial]['stimuli'][param] for trial in trial_list]
#        allparams.append([i for i in list(set(currvals))])
#
#    # Get all combinations of stimulus params:
#    transform_combos = list(itertools.product(*allparams))
#    ncombinations = len(transform_combos)
#
#    configs = dict()
#    for configidx in range(ncombinations):
#        configname = 'config%03d' % int(configidx+1)
#        configs[configname] = dict()
#        for pidx, param in enumerate(sorted(stimparams, key=natural_keys)):
#            if isinstance(transform_combos[configidx][pidx], tuple):
#                configs[configname][param] = [transform_combos[configidx][pidx][0], transform_combos[configidx][pidx][1]]
#            else:
#                configs[configname][param] = transform_combos[configidx][pidx]
#
#    # Cycle thru gratings (variable stim dur) to ignore configs that don't exist:
#    if 'grating' in stimtype and 'stim_dur' in stimparams:
#        subparams_dicts = [dict((k,v) for k,v in trialdict[t]['stimuli'].items() if k not in exclude_params) for t in trialdict.keys()]
#        keep_configs=[]; remove_configs=[];
#        for config in configs.keys():
#            if configs[config] not in subparams_dicts:
#                remove_configs.append(config)
#            else:
#                keep_configs.append(config)
#        tmp_configs = {}; 
#        for configidx, config in enumerate(keep_configs):
#            configname = 'config%03d' % int(configidx+1)
#            tmp_configs[configname] = configs[config]
#        configs = tmp_configs
#            
#                
#        
#    if stimtype=='image':
#        stimids = sorted(list(set([os.path.split(trialdict[t]['stimuli']['filepath'])[1] for t in trial_list])), key=natural_keys)
#    elif 'movie' in stimtype:
#        stimids = sorted(list(set([os.path.split(os.path.split(trialdict[t]['stimuli']['filepath'])[0])[1] for t in trial_list])), key=natural_keys)
#    
#    if stimtype == 'image' or 'movie' in stimtype:
#        filepaths = list(set([trialdict[trial]['stimuli']['filepath'] for trial in trial_list]))
#        filehashes = list(set([trialdict[trial]['stimuli']['filehash'] for trial in trial_list]))
#
#        assert len(filepaths) == len(stimids), "More than 1 file path per stim ID found!"
#        assert len(filehashes) == len(stimids), "More than 1 file hash per stim ID found!"
#
#        stimhash_combos = list(set([(trialdict[trial]['stimuli']['stimulus'], trialdict[trial]['stimuli']['filehash']) for trial in trial_list]))
#        assert len(stimhash_combos) == len(stimids), "Bad stim ID - stim file hash combo..."
#        stimhash = dict((stimid, hashval) for stimid, hashval in zip([v[0] for v in stimhash_combos], [v[1] for v in stimhash_combos]))
#
#    print "---> Found %i unique stimulus configs." % len(configs.keys())
#
#    if stimtype == 'image':
#        for config in configs.keys():
#            configs[config]['filename'] = os.path.split(configs[config]['filepath'])[1]
#    elif 'movie' in stimtype:
#        for config in configs.keys():
#            configs[config]['filename'] = os.path.split(os.path.split(configs[config]['filepath'])[0])[1]
#            
#    # Sort config dict by value:
##    sorted_configs = dict(sorted(configs.items(), key=operator.itemgetter(1)))
##    sorted_confignames = sorted_configs.keys()
##    for cidx, configname in sorted(configs.keys(), key=natural_keys):
##        configs[configname] = sorted_configs[sorted_configs.keys()[cidx]
#
#    # SAVE CONFIG info:
#    config_fpath = os.path.join(paradigm_dir, 'stimulus_configs.json')
#    with open(config_fpath, 'w') as f:
#        json.dump(configs, f, sort_keys=True, indent=4)
#    print "===== saved formated stimulus_configs to json ====="
#    print config_fpath
#    print "---------------------------------------------------"
#    
#    return configs, stimtype
#


#%%
def format_roisXvalue(Xdata, run_info, labels_df=None, fsmooth=None, sorted_ixs=None, value_type='meanstim', trace='raw'):
    '''
    Expects output to be ntrials x nrois.
    '''
    #if isinstance(Xdata, pd.DataFrame):
    Xdata = np.array(Xdata)
        
    # Make sure that we only get ROIs in provided list (we are dropping ROIs w/ np.nan dfs on any trials...)
    #sDATA = sDATA[sDATA['roi'].isin(roi_list)]
    stim_on_frame = run_info['stim_on_frame']
    if isinstance(run_info['nframes_on'], list):
        nframes_on = [int(round(nf)) for nf in run_info['nframes_on']]
        nframes_per_trial = run_info['nframes_per_trial']   
        multiple_durs = True
    else:
        nframes_on = int(round(run_info['nframes_on']))
        nframes_per_trial = run_info['nframes_per_trial']
        multiple_durs = False

    ntrials_total = run_info['ntrials_total']
    nrois = Xdata.shape[-1] #len(run_info['roi_list'])
    
    if sorted_ixs is None:
        print "Trials are sorted by time of occurrence, not stimulus type."
        sorted_ixs = xrange(ntrials_total) # Just sort in trial order

    # Get baseline and stimulus indices for each trial:
    if multiple_durs:
        assert labels_df is not None, "LABELS_DF must be provided if multiple stim durs..."
        tgroups = labels_df.groupby('trial')
        
        stim_durs = list(set(labels_df['stim_dur']))
        nframes_dict = dict((k,v) for k,v in zip(sorted(stim_durs), sorted(nframes_on)))
        std_baseline_values=[]; mean_baseline_values=[]; mean_stimulus_values=[];
        for k,g in tgroups:
            curr_baseline_stds = np.nanstd(Xdata[g['tsec'][0:stim_on_frame].index.tolist(), :], axis=0)
            curr_baseline_means = np.nanmean(Xdata[g['tsec'][0:stim_on_frame].index.tolist(), :], axis=0)
            curr_dur = list(set(g['stim_dur']))[0]
            curr_stimulus_means = np.nanmean(Xdata[g['tsec'][stim_on_frame:stim_on_frame+nframes_dict[curr_dur]].index.tolist(), :], axis=0)
            
            std_baseline_values.append(curr_baseline_stds)
            mean_baseline_values.append(curr_baseline_means)
            mean_stimulus_values.append(curr_stimulus_means)
        
        mean_stim_on_values = np.vstack(mean_stimulus_values)
        mean_baseline_values = np.vstack(mean_baseline_values)
        std_baseline_values = np.vstack(std_baseline_values)
            
    else:
        traces = np.reshape(Xdata, (ntrials_total, nframes_per_trial, nrois), order='C')
        traces = traces[sorted_ixs,:,:]    
    
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

#%% Preprocess data:

        
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
def plot_roi_contours(zproj, sorted_rids, cnts, clip_limit=0.008, label=True, 
                          draw_box=False, thickness=1, roi_color=(0, 255, 0), single_color=False):

    # Create ZPROJ img to draw on:
    refRGB = uint16_to_RGB(zproj)

    # Use some color map to indicate distance from upper-left corner:
    sorted_colors = sns.color_palette("Spectral", len(sorted_rids)) #masks.shape[-1])

    fig, ax = pl.subplots(1, figsize=(10,10))
#    p2, p98 = np.percentile(refRGB, (1, 99))
#    img_rescale = exposure.rescale_intensity(refRGB, in_range=(p2, p98))
    im_adapthist = exposure.equalize_adapthist(refRGB, clip_limit=clip_limit)
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
        if single_color:
            col255 = roi_color
        else:
            col255 = tuple([cval*255 for cval in sorted_colors[cidx]])
        if draw_box:
            cv2.drawContours(orig, [box.astype("int")], -1, col255, thickness)
        else:
            cv2.drawContours(orig, cnt, -1, col255, thickness)
        if label:
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
