#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 13:59:19 2018

@author: juliana
"""


import h5py
import copy
import hashlib
import os
import json
import cv2
import time
import math
import sys
import random
import itertools
import scipy.io
import optparse
import sys
import glob
import itertools

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
import scikit_posthocs as sp
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison
from ast import literal_eval

from pipeline.python.utils import natural_keys, replace_root, print_elapsed_time, label_figure
import pipeline.python.traces.combine_runs as cb
import pipeline.python.paradigm.align_acquisition_events as acq
import pipeline.python.visualization.plot_psths_from_dataframe as vis
from pipeline.python.traces.utils import load_TID
from pipeline.python.paradigm import utils as fmt
from pipeline.python.classifications import utils as util
from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections import Counter

from sklearn.manifold import MDS
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import cross_validation

from sklearn.datasets import make_multilabel_classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelBinarizer
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA

from sklearn.feature_selection import RFE
from sklearn.svm import SVR #LinearSVR
from sklearn.externals import joblib

from sklearn.linear_model import SGDClassifier, LogisticRegression, LogisticRegressionCV
from sklearn.calibration import CalibratedClassifierCV



from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold, cross_val_score
from sklearn import svm
from sklearn import cross_validation
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
#from joblib import Parallel, delayed
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import permutation_test_score


from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import LeavePGroupsOut, LeaveOneGroupOut, LeaveOneOut
from sklearn import metrics


#%%
# =============================================================================
# Setting classifier params
# =============================================================================
#%
def get_classifier_params(**cparams):
    '''
    Specified ways to group and label classifier data.
    
    Params:
    =======
    'data_type': (str)
        - 'frames' :  Classify frames (or binned frames)
        - 'stat'   :  Classify some statistic taken across frames/trials (e.g., mean, zscore, etc.)
        
    'inputdata': (str)
        - 'meanstim'    :  Use mean of the stimulus period (raw/corrected values)
        - 'meanstimdff' :  Use df/f of stimulus period
    
    'inputdata_type':  (str)
        - only relevant for cnmf datasets
        - if we use a stat-based dataset, need to specify which raw input type to use (e.g., spikes, corrected, dff, etc.)
        
    'class_name':  (str) 
        - stimulus parameter to decode (e.g., 'morphlevel', 'yrot', 'ori', etc.)
    
    'aggregate_type':  (str)
        - method of grouping (averaging) trials together
        - options:
            'all' = label each trial 
            'single' :  Only include trials of a specific transform type and value,
                        Use this if want to decode 'morphlevel' on trials in which 'xpos'=5.0, for ex.
    
    'subset': (str) or None
        - Different from aggregate_type because no averaging or grouping is done across classes.
        - This is used for taking only a subset of classes *within* the class_name specified.
        - options:
            'two_class' :  This is specific to morphlevel, i.e., only take anchor1 and anchor2 to do a binary classification.
            'no_morphing' :  This is specific to rotating movies, i.e., don't take the funky A-B/B-A morphs (just single object movies)

    'subset_nsamples': int or None
        - number of samples to combine together if averaging subset halves (aggregate_type='half')
        - use this param to draw specified number of repetitions to average together (default is half)

    'const_trans':  (str) 
        - Transform type to keep constant (i.e., only label instances in which const_trans = trans_value)
    
    'trans_value':  (int, float)
        - Value to set 'const_trans' to.
    '''
    
    clfparams = dict((k,v) for k,v in cparams.items())
    
#     Make sure aggregate type matches:
#    if clfparams['const_trans'] is not '' and clfparams['trans_value'] is not '':
#        optsE.aggregate_type = 'single' # TODO:  fix this so that tuples of const-trans are allowed?
    #clfparams['binsize'] = clfparams['binsize'] if clfparams['data_type'] =='frames' else ''
    
    
    return clfparams


def get_default_gratings_params():
    
    clfparams = get_classifier_params(data_type='stat', 
                                      inputdata='meanstim', 
                                      inputdata_type='',
                                      roi_selector='visual', 
                                      class_name='ori', 
                                      aggregate_type='all',
                                      subset=None, subset_nsamples=None,
                                      const_trans='', trans_value='',
                                      cv_method='kfold', 
                                      cv_nfolds=5,
                                      cv_ngroups=1,
                                      classifier='LinearSVC', 
                                      binsize=None,
                                      get_null=False,
                                      C=1e9)
    return clfparams
    

def get_best_C(svc, X, y, output_dir='/tmp', classifier_str=''):
    # Look at cross-validation scores as a function of parameter C
    C_s = np.logspace(-10, 10, 50)
    scores = list()
    scores_std = list()
    for C in C_s:
        svc.C = C
        this_scores = cross_val_score(svc, X, y, n_jobs=1)
        scores.append(np.mean(this_scores))
        scores_std.append(np.std(this_scores))

    # Do the plotting
    pl.figure(figsize=(6, 6))
    pl.semilogx(C_s, scores)
    pl.semilogx(C_s, np.array(scores) + np.array(scores_std), 'b--')
    pl.semilogx(C_s, np.array(scores) - np.array(scores_std), 'b--')
    locs, labels = pl.yticks()
    pl.yticks(locs, list(map(lambda x: "%g" % x, locs)))
    pl.ylabel('CV score')
    pl.xlabel('Parameter C')
    pl.ylim(0, 1.1)

    best_idx_C = scores.index(np.max(scores))
    best_C = C_s[best_idx_C]
    pl.title('best C: %0.4f' % best_C)

    figname = 'crossval_scores_by_C_%s.png' % classifier_str
    pl.savefig(os.path.join(output_dir, figname))
    pl.close()

    return best_C


#%%
    

def get_roi_list(run_info, roi_selector='visual', metric='meanstimdf'):
        
    trans_types = run_info['transforms'].keys()
    
    # Load sorted ROI info:
    # -----------------------------------------------------------------------------
#    datakey = run_info['datakey']
#    if '/' in datakey:
#        datakey = datakey[1:]
#    sort_dir = os.path.join(traceid_dir, 'sorted_%s' % datakey)

    traceid_dir = run_info['traceid_dir']
    sorted_dirs = [s for s in os.listdir(traceid_dir) if 'sorted_' in s and os.path.isdir(os.path.join(traceid_dir, s))]
    sort_dir = os.path.join(traceid_dir, sorted_dirs[0])
    
    if roi_selector == 'visual':
        
        # Get list of visually repsonsive ROIs:
        responsive_anova_fpath = os.path.join(sort_dir, 'visual_rois_anova_results.json')
        assert os.path.exists(responsive_anova_fpath), "No results found for VISUAL rois: %s" % sort_dir
        
        print "Loading existing split ANOVA results:\n", responsive_anova_fpath
        with open(responsive_anova_fpath, 'r') as f:
            responsive_anova = json.load(f)
        
        # Sort ROIs:
        responsive_rois = [r for r in responsive_anova.keys() if responsive_anova[r]['p'] < 0.05]
        sorted_visual = sorted(responsive_rois, key=lambda x: responsive_anova[x]['F'])[::-1]
        print "Loaded %i visual out of %i neurons (split-plot ANOVA (p<0.05)." % (len(sorted_visual), len(run_info['roi_list']))
        
        visual_rids = [int(r[3:])-1 for r in sorted_visual]
    
    elif roi_selector == 'selectiveanova':
    
        selective_anova_fpath = os.path.join(sort_dir, 'selective_rois_anova_results_%s.json' % metric)
        assert os.path.exists(selective_anova_fpath), "No results found for SELECTIVE rois (anova): %s" % sort_dir
        
        if os.path.exists(selective_anova_fpath):
            print "Loading existing %i-way ANOVA results: %s" % (len(trans_types), selective_anova_fpath)
            with open(selective_anova_fpath, 'r') as f:
                selective_anova = json.load(f)
            if '(' in selective_anova[selective_anova.keys()[0]].keys()[0]:
                for r in selective_anova.keys():
                    selective_anova[r] = {literal_eval(k):v for k, v in selective_anova[r].items()}
    
        if len(trans_types) > 1:    
            selective_rois_anova = [k for k in selective_anova.keys()
                                    if any([selective_anova[k][t]['p'] < 0.05
                                    for t in selective_anova[k].keys()]) ]
            # Sort by F-value, from biggest --> smallest:
            selective_rois_anova = sorted(selective_rois_anova, key=lambda x: max([selective_anova[x][f]['F'] for f in selective_anova[x].keys()]))[::-1]
            
        else:
            selective_rois_anova = [k for k in selective_anova.keys()
                                    if selective_anova[k]['p'] < 0.05]
            
        visual_rids = [int(r[3:])-1 for r in selective_rois_anova]
        
    return visual_rids


#%%
#    
#def group_by_class(clfparams, cX, cy, sconfigs):
#    
#    cy = np.array([sconfigs[cv][clfparams['class_name']] if cv != 'bas' else 'bas' for cv in cy])
#    class_labels = sorted(np.unique(cy))
#    
#    return cX, cy, class_labels
#
#def group_by_class_subset(clfparams, cX, cy, sconfigs):
#    '''
#    Only grab samples belonging to subset of class types.
#    Expects a list of included class types,
#        e.g., can provide anchors if class_name is 'morphlevel', or can provide xpos values [-20, 20], etc.)
#    
#    For indexing purposes, cy should still be of form 'config001', 'config002', etc.
#    
#    Label assignment happens here.
#    '''
#    
#    sconfigs_df = pd.DataFrame(sconfigs).T
#    configs_included = sconfigs_df[sconfigs_df[class_name].isin(clfparams['class_subset'])].index.tolist()
#    if clfparams['get_null']:
#        configs_included.append('bas')
#        
#    kept_ixs = np.array([cix for cix, cname in enumerate(cy) if cname in configs_included])
#    cX_tmp = cX[kept_ixs, :] 
#    cy_tmp = cy[kept_ixs]
#    
#    cy = np.array([sconfigs[cname][clfparams['class_name']] if cname != 'bas' else 'bas' for cname in cy_tmp])
#    class_labels = sorted(np.unique(cy))
#    
#    return cX_tmp, cy, class_labels
#
#
#def group_by_transform_subset(clfparams, cX, cy, sconfigs):
#    # Select only those samples w/ values equal to specificed transform value.
#    # 'const_trans' :  transform type desired
#    # 'trans_value' :  value of const_trans to use.
#    
#    sconfigs_df = pd.DataFrame(sconfigs).T
#
#    # Check that provided trans_value is valid:
#    const_trans = clfparams['const_trans'] #if isinstance(clfparams['const_trans'], list) else [clfparams['const_trans']]
#    trans_value = clfparams['trans_value'] #if isinstance(clfparams['trans_value'], list) else [clfparams['trans_value']]
#    if trans_value == '': 
#        # Get all values for specified const_trans:
#        trans_value = []
#        for trans in const_trans:
#            trans_value.append(sorted(sconfigs_df[trans].unique().tolist()))
#    
#    
#    # Check that all provided trans-values are valid and get config IDs to include:
#    const_trans_dict = dict((k, v) for k,v in zip([t for t in const_trans], [v for v in trans_value]))
#    
#    
#    configs_included = []; configs_pile = sconfigs.keys()
#    for transix, (trans_name, trans_value) in enumerate(const_trans_dict.items()):
#        if len(trans_value) > 1:
#            assert all([trans_v in sconfigs_df[trans_name].unique() for trans_v in trans_value]), "Specified trans_name, trans_value not found: %s" % str((trans_name, trans_val))
#        else:
#            assert trans_value in sconfigs_df[trans_name].unique(), "Specified trans_name, trans_value not found: %s" % str((trans_name, trans_value))
#        
#        if transix == 0:
#            first_culling = sconfigs_df[sconfigs_df[trans_name].isin(trans_value)].index.tolist()
#            configs_tmp = [c for c in configs_pile if c in first_culling]
#        else:
#            configs_pile = copy.copy(configs_tmp)
#            first_culling = sconfigs_df[sconfigs_df[trans_name].isin(trans_value)].index.tolist()
#            configs_tmp = [c for c in configs_pile if c in first_culling]
#    configs_included = configs_tmp
#
#    if clfparams['get_null']:
#        configs_included.append('bas')
#        
#    kept_ixs = np.array([cix for cix, cname in enumerate(cy) if cname in configs_included])
#    cX_tmp = cX[kept_ixs, :] 
#    cy_tmp = cy[kept_ixs]
#    
#    cy = np.array([sconfigs[cname][clfparams['class_name']] if cname != 'bas' else 'bas' for cname in cy_tmp])
#    class_labels = sorted(np.unique(cy))
#    
#    return cX_tmp, cy, class_labels
#        
    

#%%
# =============================================================================
# Data loading functions:
# =============================================================================
    
#def get_datapaths_from_opts(optsE):
#    
#    acquisition_dir = os.path.join(optsE.rootdir, optsE.animalid, optsE.session, optsE.acquisition)
#    run_list = optsE.run_list
#    traceid_list = optsE.traceid_list
#    if not len(run_list) == len(traceid_list):
#        assert len(traceid_list) == 1 and len(run_list) > 1, \
#                "[E]: I don't know what to do with this combination of RUNS\n%s\nand TRACEIDS\n\%s." % (str(run_list), str(traceid_list))
#        traceid_list = [traceid_list[0] for _ in range(len(run_list))]
#        print "Assuming same traceid for all specified runs: %s" % str(run_list)
#    
#        
#    data_paths = {}
#    for dix, (run, traceid) in enumerate(zip(run_list, traceid_list)):
#        #%
#        print "**********************************"
#        print "Processing %i of %i runs." % (int(dix+1), len(run_list))
#        print "**********************************"
#    
#        #%
#        traceid_dir = fmt.get_traceid_from_acquisition(acquisition_dir, run, traceid)
#        
#        # Data array dir:
#        data_basedir = os.path.join(traceid_dir, 'data_arrays')
#        data_fpath = os.path.join(data_basedir, 'datasets.npz')
#        assert os.path.exists(data_fpath), "[E]: Data array not found! Did you run tifs_to_data_arrays.py?"
#    
#        # Create output base dir for classifier results:
#        clf_basedir = os.path.join(traceid_dir, 'classifiers')
#        if not os.path.exists(os.path.join(clf_basedir, 'figures')):
#            os.makedirs(os.path.join(clf_basedir, 'figures'))
#    
#
#        data_paths[dix] = data_fpath
#    
#    return data_paths
#
##%
#
#def combine_datasets(data_paths, combo_name='combo'):
#    excluded_keys = ['traceid_dir', 'ntrials_by_cond', 'nfiles', 'ntrials_total']
#    
#    # Load the first dataset's run_info, and use as reference for all other dsets:
#    d1 = np.load(data_paths[0])
#    s1 = d1['sconfigs'][()]
#
#    ref_info = d1['run_info'][()]
#    info_keys = ref_info.keys()
#    non_array_keys = ['frac', 'tsecs', 'quantile', 'sconfigs', 'run_info', 'labels_columns']
#
#    summed_info = dict((k, [ref_info[k]]) for k in excluded_keys)
#    concat_arrays = dict((k, d1[k]) for k in d1.keys() if k not in non_array_keys)
#    
#    # Initialize trial counter to update trial number in labels:
#    trial_counter = ref_info['ntrials_total']
#    trial_labels_col = [i for i in d1['labels_data'][0, :]].index('trial00001') # should be col 4
#    
#    for di,dpath in data_paths.items():
#        if di==0:
#            continue
#        d = np.load(dpath)
#        currblock_ntrials = d['run_info'][()]['ntrials_total']
#        print "Run %i of %i:  Adding %i trials to combined dset" % (di, len(data_paths), currblock_ntrials)
#        
#        # Make sure we're combining the same trial types together:
#        assert all([ref_info[k]==d['run_info'][()][k] for k in info_keys if k not in excluded_keys]), "Trying to combine unequal runs!"
#        
#        # Make sure stimulus configs are the exact same:
#        assert d['sconfigs'][()]==s1, "Stim configs are different!"
#        
#        # Make sure preprocessing parameters are the same:
#        if 'frac' in d.keys():
#            assert d1['frac'] == d['frac'], "Smoothing fractions differ: ref %d, %s %d" % (d1['frac'], dpath, d['frac'])
#        if 'quantile' in d.keys():
#            assert d1['quantile'] == d['quantile'], "Quantile values differ: ref %d, %s %d" % (d1['quantile'], dpath, d['quantile'])
#        
#        # Append data arrays for combo:
#        for array_key in [k for k in d1.keys() if k not in non_array_keys]:
#            #print "Array for comb: %s" % array_key
#            darray = d[array_key]
#            if array_key == 'labels_data':
#                # Need to replace trial labels to be relative to first trial of first run
#                tlabels = d[array_key][:, trial_labels_col]
#                darray[:, trial_labels_col] = np.array(['trial%05d' % (int(t[5:])+trial_counter) for t in tlabels], dtype=d[array_key].dtype)
#
#            if len(concat_arrays[array_key].shape) == 2:
#                tmp = np.vstack((concat_arrays[array_key], darray))
#            else:
#                tmp = np.hstack((concat_arrays[array_key], darray))
#            concat_arrays[array_key] = tmp
#            
#        # Append run info for combo:
#        for exk in excluded_keys:
#            summed_info[exk].append(d['run_info'][()][exk])
#            
#        trial_counter += currblock_ntrials  # Increment last trial num
#    
#    # Combined info that represents combo:
#    for k,v in summed_info.items():
#        if isinstance(v[0], dict):
#            tmp_entry = dict((kk, sum([v[i][kk] for i in range(len(v))])) for kk in v[0].keys())
#            summed_info[k] = tmp_entry
#        elif isinstance(v[0], (float, int)):
#            summed_info[k] = sum(v)
#    combined_run_info = dict((k, summed_info[k]) if k in excluded_keys else (k, v) for k,v in ref_info.items())
#
#    # Combine data arrays:
#        
#    dataset = dict((k, v) if k in non_array_keys else (k, concat_arrays[k]) for k,v in d1.items())
#    dataset['run_info'] = combined_run_info
#    orig_srcs = dataset['run_info']['traceid_dir']
#    dataset['source_paths'] = orig_srcs
#    
#    # Save to new trace dir:
#    if 'traces00' in data_paths.values()[0]:
#        tid_str = '_'.join([tdir.split('/traces')[-1].split('/')[0] for tdir in data_paths.values()])
#    else:
#        tid_str = '_'.join([tdir.split('/cnmf/')[-1].split('/')[0] for tdir in data_paths.values()])
#    acquisition_dir = os.path.split(data_paths[0].split('/traces')[0])[0]
#    combined_darray_dir = os.path.join(acquisition_dir, combo_name, 'traces', tid_str, 'data_arrays')
#    if not os.path.exists(combined_darray_dir): os.makedirs(combined_darray_dir)
#    data_fpath = os.path.join(combined_darray_dir, 'datasets.npz')
#    dataset['run_info']['traceid_dir'] = data_fpath.split('/data_arrays')[0]
#    
#    print "Saving combined dataset to:\n%s" % data_fpath
#    np.savez(data_fpath, dataset)
#    
#    
#    
#    return data_fpath



#%%

# =============================================================================
# Generic, non-classifier plotting functions (plots for training data source)
# =============================================================================

def correlation_matrix(clfparams, class_labels, cX, cy, data_identifier='', output_dir='/tmp'):
    
    # Correlation between stim classes across ROIs -- compute as RDM:
    # -----------------------------------------------------------------------------
    zdf_cmap = 'PRGn' #pl.cm.hot
    
    df_list = []
    for label in class_labels:
        tidxs = np.where(cy==label)             # Get indices of trials (or frames) with current label
        curr_samples = np.squeeze(cX[tidxs,:])  # nsamples_current_label x nrois
        cname = str(label)
        currstim = np.reshape(curr_samples, (curr_samples.shape[0]*curr_samples.shape[1],), order='F') # Reshape into 1D vector for current label
        df_list.append(pd.DataFrame({cname: currstim}))
    df = pd.concat(df_list, axis=1)
    df.head()
    
    corrs = df.corr(method='pearson')
    fig = pl.figure()
    sns.heatmap(1-corrs, cmap=zdf_cmap, vmax=2.0)
    #sns.heatmap(1-corrs, cmap=zdf_cmap, vmin=0.7, vmax=1.3)
    figbase = 'RDM_%srois_%s_classes_%i%s_%s' % (clfparams['roi_selector'], clfparams['data_type'], len(class_labels), 
                                                     clfparams['class_name'], clfparams['aggregate_type'])
        
    if clfparams['aggregate_type'] == 'single':
        pl.title('RDM (%s) - %s: %s (%s %i)' % (clfparams['data_type'], 
                                                 clfparams['class_name'], clfparams['aggregate_type'],
                                                 clfparams['const_trans'], clfparams['trans_value']))
        figbase = '%s_%s%i' % (figbase, clfparams['const_trans'], clfparams['trans_value'])
    else:
        pl.title('RDM (%s) - %s: %s' % (clfparams['data_type'], clfparams['class_name'], clfparams['aggregate_type']))

    label_figure(fig, data_identifier)
    
    pl.savefig(os.path.join(output_dir, '%s.png' % figbase))
    pl.close()
    
    #pl.figure()
    #sns.heatmap(corrs, cmap='PRGn', vmin=-1, vmax=1)
    
    # Look at RDM within trial:
    # -----------------------------------------------------------------------------
    nframes_per_trial = list(set([v for k,v in Counter(cy).items()]))[0] #run_info['nframes_per_trial']
    df_list = []
    cnames = []
    for label in class_labels:
        tidxs = np.where(cy==label)[0]
        currtrials = np.squeeze(cX[tidxs,:])
        if clfparams['data_type'] == 'xcondsub':
            ax_labels = np.hstack([np.tile(i, (nframes_per_trial,)) for i in range(len(tidxs)/nframes_per_trial)])
            new_ax_labels = ['%s_%s' % (str(label), ti) for ti in ax_labels]
            cnames.extend(new_ax_labels)
            cell_unit = 'frames'
        else:
            #new_ax_labels = ['%s_%s' % (str(label), ti) for ti in range(len(tidxs))]
            new_ax_labels = ['%s' % (str(label)) for ti in range(len(tidxs))]
            cnames.extend(new_ax_labels)
            cell_unit = 'trials'
        df_list.append(pd.DataFrame(data=currtrials.T, columns=new_ax_labels))
    
    df_trials = pd.concat(df_list, axis=1)
    
    corrs = df_trials.corr(method='pearson')
    fig = pl.figure()
    ax = sns.heatmap(1-corrs, vmax=2, cmap=zdf_cmap) #, vmax=1)
    
    indices = { value : [ i for i, v in enumerate(cnames) if v == value ] for value in list(set(cnames)) }
    xtick_indices = [(k, indices[k][-1]) for k in list(set(cnames))]
    ax.set(xticks = [])
    ax.set(xticks = [x[1] for x in xtick_indices])
    ax.set(xticklabels = [x[0] for x in xtick_indices])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=8)
    ax.set(yticks = [])
    ax.set(yticks = [x[1] for x in xtick_indices])
    ax.set(yticklabels = [x[0] for x in xtick_indices])
    ax.set_yticklabels(ax.get_yticklabels(), rotation=45, fontsize=8)
    ax.set(xlabel=cell_unit)
    
    pl.title('RDM (%s, %s)' % (clfparams['data_type'], cell_unit))
    
    figname = '%s_%s' % (figbase, cell_unit)
    label_figure(fig, data_identifier)
    pl.savefig(os.path.join(output_dir, figname))
    pl.close()


def print_top10(vectorizer, clf, class_labels):
    """Prints features with the highest coefficient values, per class"""
    feature_names = vectorizer.get_feature_names()
    for i, class_label in enumerate(class_labels):
        top10 = np.argsort(clf.coef_[i])[-10:]
        print("%s: %s" % (class_label,
              " ".join(feature_names[j] for j in top10)))

def plot_coefficients(classifier, feature_names, class_idx=0, top_features=20):
    if len(classifier.classes_) >= 3:
        coef = classifier.coef_[class_idx, :].ravel()
    else:
        coef = classifier.coef_.ravel()
    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    # create plot
    sns.set_style('white')
    fig = pl.figure(figsize=(15, 5))
    colors = ['r' if c < 0 else 'b' for c in coef[top_coefficients]]
    sns.barplot(np.arange(2 * top_features), coef[top_coefficients], palette=colors) #colors)
    sns.despine(offset=4, trim=True)

    feature_names = np.array(feature_names)
    pl.xticks(np.arange(0, 1 + 2 * top_features), feature_names[top_coefficients], rotation=45, ha='right')
    pl.title('Sorted weights (top %i), class %i' % (top_features, class_idx))

    return fig

def plot_weight_matrix(svc, absolute_value=True):
        
    if absolute_value:
        weight_values_ref = np.abs(svc.coef_[0,:])
        cmap='magma'
    else:
        weight_values_ref = svc.coef_[0]
        cmap='RdBu'
        
    sorted_weights = np.argsort(weight_values_ref)[::-1]
    
    sorted_weight_matrix = np.empty(svc.coef_.shape)
    if len(svc.classes_) < 3:
        if absolute_value:
            sorted_weight_matrix[0,:] = np.abs(svc.coef_[0,sorted_weights])
        else:
            sorted_weight_matrix[0,:] = svc.coef_[0,sorted_weights]
    else:
        for ci,cv in enumerate(svc.classes_):
            if absolute_value:
                sorted_weight_matrix[ci,:] = np.abs(svc.coef_[ci,sorted_weights])
            else:
                sorted_weight_matrix[ci,:] = svc.coef_[ci,sorted_weights]
        
    fig, ax = pl.subplots(figsize=(50,5))
    cbar_ax = fig.add_axes([.905, .3, .005, .5])
    g = sns.heatmap(sorted_weight_matrix, cmap=cmap, ax=ax, cbar_ax = cbar_ax, cbar=True)
    for li,label in enumerate(g.get_xticklabels()):
        if li % 20 == 0:
            continue
        label.set_visible(False)
    g.set_xticklabels(g.get_xticklabels(), rotation=45, fontsize=8)
    box = g.get_position()
    g.set_position([box.x0, box.y0 + box.height * 0.2,
                     box.width, box.height * 0.8])
    g.set_xlabel('features')
    g.set_ylabel('class')
    if absolute_value:
        g.set_title('abs(coef), sorted by class 0')
    else:
        g.set_title('coefs, sorted by class 0')

    return fig
#%
    
#def spatially_sort_rois(traceid_dirs):
#    
#    # =========================================================================
#    # Load ROI masks to sort ROIs by spatial distance
#    # =========================================================================
#        
#    for traceid_dir in traceid_dirs: #opts in options_list:
#        #optsE = extract_options(opts)
#        run_dir = traceid_dir.split('/traces/')[0]
#        acquisition_dir = os.path.split(run_dir)[0]
#        acquisition = os.path.split(acquisition_dir)[-1]
#        
#        #traceid_dir = dataset['run_info'][()]['traceid_dir']
#        sorted_rids, cnts, zproj = util.sort_rois_2D(traceid_dir)
#        util.plot_roi_contours(zproj, sorted_rids, cnts, clip_limit=0.005, label=False)
#        
#        figname = 'spatially_sorted_rois_%s.png' % acquisition
#        pl.savefig(os.path.join(acquisition_dir, figname))
#        pl.close()
#        
#    return sorted_rids

##%
#def plot_grand_mean_traces(dset_list, response_type='dff', label_list=[], color_list=['b','m','g'], output_dir='/tmp', save_and_close=True):
#        
#    mval = []
#    fig, ax = pl.subplots(1) #pl.figure()
#    for di,dtrace in enumerate(dset_list):
#        pl.plot(dtrace['mean'], color=color_list[di], label=label_list[di])
#        pl.fill_between(xrange(len(dtrace['mean'])), dtrace['mean']-dtrace['sem'], dtrace['mean']+dtrace['sem'], alpha=0.5, color=color_list[di])
#        pl.plot(dtrace['stimframes'], np.ones(dtrace['stimframes'].shape) * -0.005*(di+1), color=color_list[di])
#        mval.append(dtrace['mean'].max())
#
#    ax.set(xticks = [])
#    ax.set(xticklabels = [])
#    ax.set(yticks = [0,max(mval)])
#    ax.set(ylabel=response_type)
#    sns.despine(offset=4, trim=True, bottom=True)
#    pl.legend()
#    
#    figname_base = 'avgtrace_%s' % (response_type)
#    run_str = '_'.join(['%s_%s' % (dtrace['run'], dtrace['traceid']) for dtrace in dset_list])
#    figname = '%s_%s.pdf' % (figname_base, run_str)
#
#    if save_and_close:
#        pl.savefig(os.path.join(output_dir, figname))
#        pl.close()
#        
#    return figname
#
#def get_grand_mean_trace(d1, response_type='dff'):
#    d = {}
#    assert response_type in d1.keys(), "Specified response type (%s) not found. Choose from: %s" % (response_type, str(d1.keys()))
#    d1_meanstims = d1[response_type]
#    d1_run = os.path.split(d1['run_info'][()]['traceid_dir'].split('/traces')[0])[-1]
#    print d1_run, d1_meanstims.shape
#    
#    # Get run info:
#    nrois = d1_meanstims.shape[-1]
#    assert len(d1['run_info'][()]['nframes_per_trial']) == 1, "More than 1 val for nframes_per_trial! -- %s" % str(d1['run_info'][()]['nframes_per_trial'])
#    assert len(d1['run_info'][()]['nframes_on']) == 1, "More than 1 val for nframes_on! -- %s" % str(d1['run_info'][()]['nframes_on'])
#
#    nframes_per_trial = d1['run_info'][()]['nframes_per_trial'][0]
#    nframes_on = d1['run_info'][()]['nframes_on'][0]
#    d1_nframes = nframes_per_trial
#    d1_tmat = np.reshape(d1_meanstims, (d1_meanstims.shape[0]/d1_nframes, d1_nframes, nrois))
#    meantrace_rois1 = np.mean(d1_tmat, axis=0)
#    mean_baselines = np.mean(meantrace_rois1[0:d1['run_info'][()]['stim_on_frame'], :], axis=0)
#    
#    meantrace_rois1 -= mean_baselines
#    
#    meantrace1 = np.mean(meantrace_rois1, axis=1)
#    semtrace1 = stats.sem(meantrace_rois1, axis=1)
#    d1_stim_frames = np.array([d1['run_info'][()]['stim_on_frame'], int(round(d1['run_info'][()]['stim_on_frame'] + nframes_on))])
#    
#    d['run'] = d1_run
#    d['mean'] = meantrace1
#    d['sem'] = semtrace1
#    d['stimframes'] = d1_stim_frames
#    d['traceid'] = d1['run_info'][()]['traceid_dir'].split('/traces/')[-1].split('/')[-1]
#    return d
#
#
#def compare_grand_mean_traces(data_paths):
#    assert len(data_paths.keys()) > 1, "Only 1 data path specified, nothing to compare..."
#    
#    dset_list = []
#    response_type = 'dff'
#
#    for di, dpath in data_paths.items():
#        d1 = np.load(dpath)
#        # Load data:
#        dtrace = get_grand_mean_trace(d1, response_type=response_type)
#        dset_list.append(dtrace)
#    
#    # PLOT:
#    a_run_dir = data_paths[0].split('/traces')[0]
#    acquisition_dir = os.path.split(a_run_dir)[0]
#    figname = plot_grand_mean_traces(dset_list, response_type=response_type,
#                                         label_list=['%s_%s' % (dtrace['run'], dtrace['traceid']) for dtrace in dset_list], 
#                                         output_dir=acquisition_dir, 
#                                         save_and_close=False)
#    pl.savefig(os.path.join(acquisition_dir, figname))
#    pl.close()
#


#%%

# =============================================================================
# FRAMES data-type classifications:
# =============================================================================
#cX = C.cX; cy = C.cy; class_name = C.clfparams['class_name']; 
#sconfigs = C.data.sconfigs; run_info = C.data.run_info;
#binsize = C.clfparams['binsize']

def split_trial_epochs(cX, cy, class_name, sconfigs, run_info, binsize=10, relabel=False):
    # Reshape (nframes_per_trial*ntrials_total X nrois) into ntrials_total x nframes_per_trial x nrois)
    nframes_per_trial = run_info['nframes_per_trial']
    if isinstance(nframes_per_trial, list):
        assert len(nframes_per_trial) == 1, "More than 1 unique value found for n frames per trial..."
        nframes_per_trial = nframes_per_trial[0]
    ntrials_total = run_info['ntrials_total']
    nrois = cX.shape[-1] # len(run_info['roi_list'])
    print "Reshaping input data into trials (%i) x frames (%i) x rois (%i)" % (ntrials_total, nframes_per_trial, nrois) 
    cX_r = np.reshape(cX, (ntrials_total, nframes_per_trial, nrois))
    cy_r = np.reshape(C.cy, (ntrials_total, nframes_per_trial))
    print cy_r.shape
    
    # Bin frames into smaller chunks so we don't have to train 300 classifiers...
    # nframes_on = int(round(run_info['nframes_on']))
    # stim_on = int(run_info['stim_on_frame'])
    bin_idxs =  np.arange(0, nframes_per_trial, binsize)
    bin_idxs = np.append(bin_idxs, nframes_per_trial+1) # Add the last frame 
 
    # Make binned frame data into shape: ntrial_epochs x ntrials_total x nrois.
    cX_tmp = np.array([np.mean(cX_r[:, bstart:bin_idxs[bi+1], :], axis=1) for bi, bstart in enumerate(bin_idxs[0:-1])])
    
    # There will be cX_tmp.shape[0] classifiers, each with ntrials_total x nrois dataset:
    epochs = dict((epoch, cX_tmp[epoch, :, :]) for epoch in range(cX_tmp.shape[0])) # Each dataset is ntrials x nrois
    if isinstance(cy_r[0, 0], (str, unicode)) and 'config' in cy_r[0, 0]:
        trial_labels = [sconfigs[cv][class_name] for cv in cy_r[:,0]]
    else:
        trial_labels = cy_r[:, 0]
    if relabel:
        labels = dict((epoch, trial_labels) for epoch in range(cX_tmp.shape[0]))           # Each sample in dataset (ntrials) is labeled by stim type
    else:
        labels = dict((epoch, cy_r[:,0]) for epoch in range(cX_tmp.shape[0]))           # Each sample in dataset (ntrials) is labeled by stim type

    return epochs, labels, bin_idxs[0:-1]
    

def get_trial_bins(run_info, binsize=10):
    bins = []
    # Figure out bin size if using FRAMES:
    nframes_per_trial = run_info['nframes_per_trial']
    if isinstance(nframes_per_trial, list):
        assert len(nframes_per_trial) == 1, "More than 1 value found for nframes_per_trial: %s" % str(nframes_per_trial)
        nframes_per_trial = nframes_per_trial[0]
        
    stim_on_frame = run_info['stim_on_frame']
    try:
        bins = np.arange(0, nframes_per_trial, binsize)
        assert stim_on_frame in bins, "Stim on frame %i not in bins: %s" % (stim_on_frame, str(bins))
    except Exception as e:
        binsize += 5
        bins = np.arange(0, nframes_per_trial, binsize)
        assert stim_on_frame in bins, "Stim on frame %i not in bins: %s" % (stim_on_frame, str(bins))
        
    return bins, binsize


def format_epoch_dataset(clfparams, cX, cy, run_info, sconfigs):

    class_name = clfparams['class_name']
    binsize = clfparams['binsize']; const_trans=clfparams['const_trans']
    bins, binsize = get_trial_bins(run_info, binsize=binsize) #run_info['nframes_per_trial'], run_info['stim_on_frame'], binsize=binsize)
    print "Dividing trial into %i epochs (binsize=%i)." % (len(bins), binsize)
    
    # Get values for specified const-trans value:
    const_trans_values = []
    if not const_trans=='':
        const_trans_values = sorted(list(set([sconfigs[c][const_trans] for c in sconfigs.keys()])))
        print "Selected subsets of transform: %s. Found values %s" % (const_trans, str(const_trans_values))
        
    epochs, labels, bins = split_trial_epochs(cX, cy, class_name, sconfigs, run_info, binsize=binsize, relabel=False)
    
    if len(const_trans_values) > 0:
        decode_dict = dict((trans_value, {}) for trans_value in const_trans_values)
        for trans_value in const_trans_values:
            subepochs = epochs.copy() #copy.copy(epochs)
            sublabels = labels.copy()
            kept_configs = [c for c in sconfigs.keys() if sconfigs[c][const_trans]==trans_value]
            config_ixs = [ci for ci,cv in enumerate(labels[0]) if cv in kept_configs]
            # Take subset of trials:
            for c in epochs.keys():
                subepochs[c] = subepochs[c][config_ixs, :]
                sublabels[c] = sublabels[c][config_ixs]
                new_labels = np.array([sconfigs[cv][class_name] for cv in sublabels[c]])
                sublabels[c] = new_labels
                
            decode_dict[trans_value]['epochs'] = subepochs
            decode_dict[trans_value]['labels'] = sublabels
    else:
        decode_dict = {'all': {}}
        if isinstance(labels[0][0], (str, unicode)) and 'config' in labels[0,0]:
            new_labels = np.array([sconfigs[cv][class_name] for cv in labels[0]]) # Each classifier has the same labels, since we are comparing epochs
            labels = dict((k, new_labels) for k in labels.keys())

        decode_dict['all']['epochs'] = epochs
        decode_dict['all']['labels'] = labels
        
    class_labels = sorted(list(set(decode_dict[decode_dict.keys()[0]]['labels'][0])))
    print "Labels:", class_labels

    return epochs, decode_dict, bins, binsize, class_labels


def decode_trial_epochs(class_labels, clfparams, bins, decode_dict, run_info, 
                            data_identifier='', niterations=10, scoring='accuracy',
                            output_dir='/tmp'):
    '''
    Get formatted data and bins/binsize from format_epoch_dataset(), and train
    classifier for each epoch.
    
    Plot classifier scoring method (scoring='accuracy' default) for each epoch.
    '''
    
    for view_key in decode_dict.keys():
        
        epochs = decode_dict[view_key]['epochs']
        labels = decode_dict[view_key]['labels']
        
        # Create pipeline:
        # ------------------
#        if epochs[0].shape[0] > epochs[0].shape[1]: # nsamples > nfeatures
#            dual = False
#        else:
#            dual = True
#        #pipe_svc = Pipeline([('scl', StandardScaler()),
#        #                     ('clf', LinearSVC(random_state=0, dual=dual, multi_class='ovr', C=big_C))]) #))])
#        
        for runiter in np.arange(0,niterations): #range(5):
            # evaluate each model in turn
            print "... %i of %i iterations" % (runiter, niterations)
            results = []
            names = []
            for idx, curr_epoch in enumerate(sorted(epochs.keys())):
                model = LinearSVC(random_state=0, dual=clfparams['dual'], multi_class='ovr', C=clfparams['C_val'])
                kfold = StratifiedKFold(n_splits=clfparams['cv_nfolds'], shuffle=True)
                #x_std = StandardScaler().fit_transform(epochs[curr_epoch])
                x_std = epochs[curr_epoch] # Already do standard scaling before
                cv_results = cross_val_score(model, x_std, labels[curr_epoch], cv=kfold, scoring=scoring)
                results.append(cv_results)
                names.append(curr_epoch)
                msg = "%s: %f (%f)" % (curr_epoch, cv_results.mean(), cv_results.std())
                #print(msg)
    
            fig = plot_decoding_performance_trial_epochs(results, bins, names, 
                                                       scoring=scoring, 
                                                       stim_on_frame=run_info['stim_on_frame'],
                                                       nframes_on=run_info['nframes_on'],
                                                       nclasses=len(class_labels),
                                                       binsize=clfparams['binsize'])
    
            fig.suptitle('Estimator Comparison: decode %s (%s: %s)' % (clfparams['class_name'], clfparams['const_trans'], clfparams['trans_value']))
            figname = 'trial_epoch_classifiers_binsize%i_%s_label_%s_iter%i.png' % (clfparams['binsize'], scoring, clfparams['class_name'], runiter)
            label_figure(fig, data_identifier)
            
            pl.savefig(os.path.join(output_dir, 'figures', figname))
            pl.close()
    
    #return model


def plot_decoding_performance_trial_epochs(results, bins, names, scoring='accuracy',    
                                           stim_on_frame=None, nframes_on=None,
                                           nclasses=None, binsize=10):
    # boxplot algorithm comparison
    fig = pl.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    ax = sns.boxplot(data=results, showmeans=True, linewidth=0.5)
    # Add transparency to colors
    for patch in ax.artists:
     r, g, b, a = patch.get_facecolor()
     patch.set_facecolor((r, g, b, .3))
    sns.swarmplot(data=results, ax=ax) #, color="grey")
    
    #sns.violinplot(data=results)
    ax.set_xticklabels(names)
    pl.ylabel(scoring)

    pl.ylim([0, 1])
    sns.despine(offset=4, trim=True)
    if stim_on_frame is not None and nframes_on is not None:
        if isinstance(stim_on_frame, list):
            assert len(stim_on_frame)==1, "More than 1 val found for stim_on frame: %s" % str(stim_on_frame)
            stim_on_frame = stim_on_frame[0]
        if isinstance(nframes_on, list):
            assert len(nframes_on)==1, "More than 1 val found for nframes_on: %s" % str(nframes_on)
            nframes_on = nframes_on[0]
        # Identify the time bin(s) in which the stimulus is on:
        #stim_on_frame = run_info['stim_on_frame']
        #nframes_on = int(round(run_info['nframes_on']))
        last_on_frame = stim_on_frame + nframes_on
        
        # Color the bins in which the stimulus is one:
        stimulus_bins = np.array([bins[i] for i in range(len(bins)) if stim_on_frame <= bins[i] <= last_on_frame])
        xlabel_colors = ['r' if v in stimulus_bins else 'k' for v in sorted(bins) ]
        #ax.plot(stimulus_bins, np.ones(stimulus_bins.shape)*-0.05, 'r')
        [t.set_color(i) for (i,t) in zip(xlabel_colors,ax.xaxis.get_ticklabels())]
    
    if nclasses is not None:
        chance_level = float(1./nclasses)
        ax.axhline(y=chance_level, linestyle='--', linewidth=1.0, color='k')
    
    pl.xlabel('trial epoch (bin size %i)' % binsize)
    #pl.show()
    
    return fig

#%%
            
# =============================================================================
# Cross-Validation Functions
# =============================================================================

def cv_permutation_test(svc, cX_std, cy, clfparams, scoring='accuracy', 
                        n_permutations=500, n_jobs=4, data_identifier=''):
    
    kfold = StratifiedKFold(n_splits=clfparams['cv_nfolds'], shuffle=True)
    
    cv_results = cross_val_score(svc, cX_std, cy, cv=kfold, scoring=scoring)
    print "CV RESULTS: %f (%f)" % (cv_results.mean(), cv_results.std())
    
    score, permutation_scores, pvalue = permutation_test_score(
        svc, cX_std, cy, scoring=scoring, cv=kfold, 
        n_permutations=n_permutations, n_jobs=n_jobs)
    
    print("Classification score %s (pvalue : %s)" % (score, pvalue))
    
    # -----------------------------------------------------------------------------
    # How significant is our classification score(s)?
    # Calculate p-value as percentage of runs for which obtained score is greater 
    # than the initial classification score (i.e., repeat classification after
    # randomizing and permuting labels).
    # -----------------------------------------------------------------------------
    fig = pl.figure()
    n_classes = np.unique([cy]).size
    
    # View histogram of permutation scores
    pl.hist(permutation_scores, 20, label='Permutation scores',
             edgecolor='black')
    ylim = pl.ylim()
    pl.plot(2 * [score], ylim, '--g', linewidth=3,
             label='Classification Score'
             ' (pvalue %s)' % round(pvalue, 4))
    pl.plot(2 * [1. / n_classes], ylim, '--k', linewidth=3, label='Luck')
    
    pl.ylim(ylim)
    pl.legend()
    pl.xlabel('Score - %s' % scoring)
    #pl.show()
    
    if svc.C == 1: Cstring = 'C1';
    elif svc.C == 1E9: Cstring = 'bigC';
    else: Cstring = 'C%i' % svc.C
        
    figname = 'cv_permutation_test_%s.png' % Cstring
    label_figure(fig, data_identifier)
    
    pl.savefig(os.path.join(clfparams['classifier_dir'], 'figures', figname))
    print "Saved CV results to: %s" % os.path.join(clfparams['classifier_dir'], 'figures', figname)
    pl.close()
    
    
def get_cv_folds(svc, clfparams, cX_std, cy, output_dir='/tmp'):
    
    cv_method = clfparams['cv_method']
    cv_nfolds = clfparams['cv_nfolds']
    cv_ngroups = clfparams['cv_ngroups']
    
    training_data = cX_std.copy()
    
    n_samples = cX_std.shape[0]
    print "N samples for CV:", n_samples
    classes = sorted(np.unique(cy))

    predicted = []
    true = []
    # Cross-validate for t-series samples:
    if cv_method=='splithalf':
    
        # Fit with subset of data:
        svc.fit(training_data[:n_samples // 2], cy[:n_samples // 2])
    
        # Now predict the value of the class label on the second half:
        y_test = cy[n_samples // 2:]
        y_pred = svc.predict(training_data[n_samples // 2:])
        predicted.append(y_pred) #=y_test])
        true.append(y_test)
        
    elif cv_method=='kfold':
        loo = cross_validation.StratifiedKFold(cy, n_folds=cv_nfolds, shuffle=True)

        for train, test in loo: #, groups=groups):
            print train, test
            X_train, X_test = training_data[train], training_data[test]
            y_train, y_test = cy[train], cy[test]
            y_pred = svc.fit(X_train, y_train).predict(X_test)
            predicted.append(y_pred) #=y_test])
            true.append(y_test)
    
    elif cv_method in ['LOGO', 'LOO', 'LPGO']:
        nframes_per_trial_tmp = list(set(Counter(cy)))
        assert len(nframes_per_trial_tmp)==1, "More than 1 value found for N frames per trial..."
        nframes_per_trial = nframes_per_trial_tmp[0]
        if cv_method=='LOGO':
            loo = LeaveOneGroupOut()
            if clfparams['data_type'] == 'xcondsub':
                ngroups = len(cy) / nframes_per_trial
                groups = np.hstack(np.tile(f, (nframes_per_trial,)) for f in range(ngroups))
        elif cv_method=='LOO':
            loo = LeaveOneOut()
            groups=None
        elif cv_method=='LPGO':
            loo = LeavePGroupsOut(5)
    
    
        for train, test in loo.split(training_data, cy, groups=groups):
            print train, test
            X_train, X_test = training_data[train], training_data[test]
            y_train, y_test = cy[train], cy[test]
    
            y_pred = svc.fit(X_train, y_train).predict(X_test)
    
            predicted.append(y_pred) #=y_test])
            true.append(y_test)
    
        
        if groups is not None:
            # Find "best fold"?
            avg_scores = []
            for y_pred, y_test in zip(predicted, true):
                pred_score = len([i for i in y_pred==y_test if i]) / float(len(y_pred))
                avg_scores.append(pred_score)
            best_fold = avg_scores.index(np.max(avg_scores))
            folds = [i for i in enumerate(loo.split(training_data, cy, groups=groups))]
            train = folds[best_fold][1][0]
            test = folds[best_fold][1][1]
            X_train, X_test = training_data[train], training_data[test]
            y_train, y_test = cy[train], cy[test]
            y_pred = predicted[best_fold]

#        else:
#            y_pred = np.array([i[0] for i in predicted])
#            y_test = np.array([i[0] for i in true])
    
        #avg_score = np.array([int(p==t) for p,t in zip(predicted, true)]).mean()

    # Save CV info:
    # -----------------------------------------------------------------------------
    
    f = open(os.path.join(output_dir, 'results', 'CV_report.txt'), 'w')
    for y_true, y_pred in zip(true, predicted):
        f.write(metrics.classification_report(y_true, y_pred, target_names=[str(c) for c in classes]))
    f.close()
    
    cv_results = {'predicted': [list(p) for p in predicted], #.tolist(), #list(y_pred),
                  'true': [list(p) for i in true], # list(y_test),
                  'classifier': clfparams['classifier'],
                  'cv_method': clfparams['cv_method'],
                  'ngroups': clfparams['cv_ngroups'],
                  'nfolds': clfparams['cv_nfolds'],
                  'classes': classes
                  }
    with open(os.path.join(output_dir, 'results', 'CV_results.json'), 'w') as f:
        json.dump(cv_results, f, sort_keys=True, indent=4)
    
    print "Saved CV results: %s" % output_dir
    
    return predicted, true, classes


def get_confusion_matrix(predicted, true, classes, average_iters=True):
    # Compute confusion matrix:
    # -----------------------------------------------------------------------------
    #if clfparams['classifier'] == 'LinearSVC':
        #average_iters = True
        
#        if (clfparams['cv_method'] == 'LOO' or clfparams['cv_method'] == 'splithalf') and (clfparams['data_type'] != 'xcondsub'):
#            # These have single valued folds (I think...):
#            y_test = np.array([int(i) for i in true])
#            y_pred = np.array([int(i) for i in predicted])
#            cmatrix_tframes = confusion_matrix(y_test, y_pred, labels=classes)
#            conf_mat_str = 'trials'
#        else:
    if average_iters:
        cmatrix_tframes = confusion_matrix(true[0], predicted[0], labels=classes)
        for iter_idx in range(len(predicted))[1:]:
            print "adding iter %i" % iter_idx
            cmatrix_tframes += confusion_matrix(true[iter_idx], predicted[iter_idx], labels=classes)
        conf_mat_str = 'AVG'
        #cmatrix_tframes /= float(len(pred_results))
    else:
        avg_scores = []
        for y_pred, y_test in zip(predicted, true):
            pred_score = len([i for i in y_pred==y_test if i]) / float(len(y_pred))
            avg_scores.append(pred_score)
        best_fold = avg_scores.index(np.max(avg_scores))

        cmatrix_tframes = confusion_matrix(true[best_fold], predicted[best_fold], labels=classes)
        conf_mat_str = 'best'
                
    return cmatrix_tframes, conf_mat_str


def plot_confusion_matrix_subplots(predicted, true, classes, cv_method='kfold', data_identifier='', output_dir='/tmp'): #calculate_confusion_matrix(predicted, true, clfparams, data_identifier=''):

    cmatrix_tframes, conf_mat_str = get_confusion_matrix(predicted, true, classes)

    #% Plot confusion matrix:
    # -----------------------------------------------------------------------------
    sns.set_style('white')
    fig = pl.figure(figsize=(10,4))
    ax1 = fig.add_subplot(1,2,1)
    plot_confusion_matrix(cmatrix_tframes, classes=classes, ax=ax1, normalize=False,
                      title='Confusion matrix (%s, %s)' % (conf_mat_str, cv_method))
    
    ax2 = fig.add_subplot(1,2,2)
    plot_confusion_matrix(cmatrix_tframes, classes=classes, ax=ax2, normalize=True,
                          title='Normalized')
    
    #%
    if '/classifiers' in output_dir:
        classif_identifier = os.path.split(output_dir.split('/classifiers')[-1])[-1]
        figname = '%s__confusion_%s_iters.png' % (classif_identifier, conf_mat_str)
    else:
        figname = 'confusion_matrix_%s_iters.png' % conf_mat_str
    label_figure(fig, data_identifier)
    pl.savefig(os.path.join(output_dir, 'figures', figname))
    pl.close()

def plot_normed_confusion_matrix(predicted, true, classes, cv_method='kfold', ax=None):
    
    # Compute confusion matrix:
    # -----------------------------------------------------------------------------
    cmatrix_tframes, conf_mat_str = get_confusion_matrix(predicted, true, classes)
        
    #% Plot confusion matrix:
    # -----------------------------------------------------------------------------
    #sns.set_style('white')
    if ax is None:
        fig, ax = pl.subplots(figsize=(10,4))

    plot_confusion_matrix(cmatrix_tframes, classes=classes, ax=ax, normalize=True,
                          title='Normalized confusion (%s, %s)' % (conf_mat_str, cv_method))
    
    return
    
#%
def plot_confusion_matrix(cm, classes,
                          ax=None,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=pl.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    #fig = pl.figure(figsize=(4,4))
    if ax is None:
        fig = pl.figure(figsize=(4,4))
        ax = fig.add_subplot(111)

    ax.set_title(title, fontsize=10)

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45, fontsize=10)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=10)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    #pl.tight_layout()
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    pl.colorbar(im, cax=cax)


#%%
    

def iterate_RFE(clfparams, cX_std, cy, scoring='accuracy', 
                    output_dir='/tmp', datasubset='full'):
    
    nrois_total = cX_std.shape[-1]
    
    if datasubset == 'half':
        # SPlit into halves:
        train_ixs=[]; test_ixs=[]
        for c in np.unique(cy):
            curr_ixs = np.where(cy==c)[0]
            train_ixs.append(curr_ixs[0::2])
            test_ixs.append(curr_ixs[1::2])
            
        train_ixs = np.hstack(train_ixs)
        test_ixs = np.hstack(test_ixs)
        
        train_data = cX_std[train_ixs,:]
        train_labels = cy[train_ixs]
        test_data = cX_std[test_ixs, :]
        test_labels = cy[test_ixs]
    
    elif datasubset == 'full':
        train_data = cX_std.copy()
        train_labels = cy.copy()
        test_data = cX_std.copy()
        test_labels = cy.copy()
    
    
    results_topN = []
    orig_rids = xrange(0, nrois_total)
    print len(orig_rids)
    
    svc = LinearSVC(random_state=0, dual=clfparams['dual'], multi_class='ovr', C=clfparams['C_val'])
    kfold = StratifiedKFold(n_splits=5, shuffle=True)
    
    # First, get accuracy with all features:
    rfe = RFE(svc, n_features_to_select=nrois_total)
    rfe.fit(train_data, train_labels)
    cv_results = cross_val_score(rfe, test_data, test_labels, cv=kfold, scoring=scoring)
    results_topN.append(cv_results)
    
    kept_rids_by_iter = []
    nfeatures_to_keep = np.arange(1, nrois_total, 1)
    for nkeep in nfeatures_to_keep[::-1]:
        rfe = RFE(svc, n_features_to_select=nkeep)
        #rfe.fit(cX_std, cy)
        new_y = rfe.fit_transform(train_data, train_labels)
    
        removed_rids = np.where(rfe.ranking_!=1)[0]
        kept_rids = np.array([i for i in orig_rids if i not in removed_rids])
        print "Keeping %i.... (%i)" % (nkeep, len(kept_rids))
        kept_rids_by_iter.append(kept_rids)
        cv_results = cross_val_score(rfe, test_data[:, kept_rids], test_labels, cv=kfold, scoring=scoring)
        results_topN.append(cv_results)

    # Save info:
    rfe_cv_info = {'kfold': kfold,
              'rfe': rfe,
              'results': results_topN,
              'kept_rids_by_iter': kept_rids_by_iter}
    
    with open(os.path.join(output_dir, 'results', 'RFE_cv_output_%s_%s.pkl' %  (scoring, datasubset)), 'wb') as f:
        pkl.dump(rfe_cv_info, f, protocol=pkl.HIGHEST_PROTOCOL)
                
    return results_topN


        
def plot_RFE_results(results_topN, nclasses, scoring='accuracy', 
                         output_dir='/tmp', datasubset='full', data_identifier=''):
    
    #nclasses = len(clfparams['class_labels'])

    # Plot number of features VS. cross-validation scores
    fig, ax = pl.subplots(1) #figure()
    pl.xlabel("Number of features selected")
    pl.ylabel("Cross validation score (nb of correct classifications)")
    #pl.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    means_by_feat = np.mean(results_topN, axis=1)[::-1]
    sems_by_feat = stats.sem(results_topN, axis=1)[::-1]
    
    ax.plot(xrange(len(results_topN)), means_by_feat, linewidth=1, color='k')
    ax.fill_between(xrange(len(results_topN)), means_by_feat-sems_by_feat, means_by_feat+sems_by_feat, color='k', alpha=0.2)
    
    pl.title("Ranked features (recursive elim.)")
    ax.axhline(y=1./nclasses, linestyle='--', linewidth=1, color='k')
    # draw dotted line at chance level:
    
    #print("Optimal number of features : %d" % rfecv.n_features_)
    label_figure(fig, data_identifier)
    
    pl.savefig(os.path.join(output_dir, 'figures', 'RFE_fittransform_%s_%s.png' % (scoring, datasubset)))
    pl.close()
    
    

    #%%

def get_stat_samples(Xdata, labels_df, clfparams, multiple_durs=True):
    print "Trials are sorted by time of occurrence, not stimulus type."

    ntrials_total = len(labels_df['trial'].unique())
    ntrials_per_cond = [len(t) for t in labels_df.groupby('config')['trial'].unique()]
    assert len(np.unique(ntrials_per_cond)) == 1, "Uneven reps per condition! %s" % str(ntrials_per_cond)
    ntrials = np.unique(ntrials_per_cond)[0]
    
    # Get baseline and stimulus indices for each trial:
    sample_labels = []
    stim_on_frame = labels_df['stim_on_frame'].unique()
    assert len(stim_on_frame) == 1, "More than 1 stim on frame found! %s" % str(stim_on_frame)
    stim_on_frame = stim_on_frame[0]
    if multiple_durs:
        tgroups = labels_df.groupby('trial')            
        #stim_durs = sorted(labels_df['stim_dur'].unique()) # longer stim durs will match with longer nframes on
        #nframes_on = sorted(labels_df['nframes_on'].unique())
        std_baseline_values=[]; mean_baseline_values=[]; mean_stimulus_values=[];
        for k,g in tgroups:
            curr_nframes_on = g['nframes_on'].unique()[0]

            curr_baseline_stds = np.nanstd(Xdata[g['tsec'][0:stim_on_frame].index.tolist(), :], axis=0)
            curr_baseline_means = np.nanmean(Xdata[g['tsec'][0:stim_on_frame].index.tolist(), :], axis=0)

            curr_stimulus_means = np.nanmean(Xdata[g['tsec'][stim_on_frame:stim_on_frame+curr_nframes_on].index.tolist(), :], axis=0)
            
            std_baseline_values.append(curr_baseline_stds)
            mean_baseline_values.append(curr_baseline_means)
            mean_stimulus_values.append(curr_stimulus_means)
            
            curr_config = g['config'].unique()[0]
            sample_labels.append(curr_config)
        
        mean_stim_on_values = np.vstack(mean_stimulus_values)
        mean_baseline_values = np.vstack(mean_baseline_values)
        std_baseline_values = np.vstack(std_baseline_values)
    else:
        nrois = Xdata.shape[-1]
        nframes_per_trial = Xdata.shape[0] / ntrials_total
        nframes_on = labels_df['nframes_on'].unique()[0]
        
        traces = np.reshape(Xdata, (ntrials_total, nframes_per_trial, nrois), order='C')
        std_baseline_values = np.nanstd(traces[:, 0:stim_on_frame], axis=1)
        mean_baseline_values = np.nanmean(traces[:, 0:stim_on_frame], axis=1)
        mean_stim_on_values = np.nanmean(traces[:, stim_on_frame:stim_on_frame+nframes_on], axis=1)
            
    if clfparams['stat_type'] == 'zscore':
        sample_array = (mean_stim_on_values - mean_baseline_values ) / std_baseline_values
    elif clfparams['stat_type'] == 'meanstimdff':
        sample_array = (mean_stim_on_values - mean_baseline_values ) / mean_baseline_values
    else:
        sample_array = mean_stim_on_values
    
    if clfparams['get_null']:
        random_draw = True
        print "Stim values:", sample_array.shape
        if random_draw:
            selected_trial_ixs = random.sample(range(0, mean_baseline_values.shape[0]), ntrials)
        bas = mean_baseline_values[selected_trial_ixs, :]
    
        sample_array = np.append(sample_array, bas, axis=0)
        print "Added null cases:", sample_array.shape
        sample_labels.extend(['bas' for _ in range(bas.shape[0])])
    
    return sample_array, np.array(sample_labels)

#%%
    
def get_frame_samples(Xdata, labels_df, clfparams):
    print "Trials are sorted by time of occurrence, not stimulus type."

    ntrials_total = len(labels_df['trial'].unique())
    ntrials_per_cond = [len(t) for t in labels_df.groupby('config')['trial'].unique()]
    assert len(np.unique(ntrials_per_cond)) == 1, "Uneven reps per condition! %s" % str(ntrials_per_cond)
    ntrials = np.unique(ntrials_per_cond)[0]
    
    
    # Get baseline and stimulus indices for each trial:
    sample_labels = []; sample_array = []; sample_bas = []
    stim_on_frame = labels_df['stim_on_frame'].unique()
    assert len(stim_on_frame) == 1, "More than 1 stim on frame found! %s" % str(stim_on_frame)
    stim_on_frame = stim_on_frame[0]
    
    tgroups = labels_df.groupby('trial')            
    #stim_durs = sorted(labels_df['stim_dur'].unique()) # longer stim durs will match with longer nframes on
    #nframes_on = sorted(labels_df['nframes_on'].unique())

    for k,g in tgroups:
        if clfparams['stat_type'] == 'trial':
            sample_ixs = g.index.tolist()
            curr_values = Xdata[sample_ixs,:]
        else:
            curr_nframes_on = g['nframes_on'].unique()[0]
            if clfparams['stat_type'] == 'stimulus':
                sample_ixs = g['tsec'][stim_on_frame:stim_on_frame+curr_nframes_on].index.tolist()
                curr_values = Xdata[sample_ixs, :]
            elif clfparams['stat_type'] == 'post':
                sample_ixs = g['tsec'][stim_on_frame+curr_nframes_on:].index.tolist()
                curr_values = Xdata[sample_ixs, :]
                
            if clfparams['get_null']:
                sample_ixs = g['tsec'][0:stim_on_frame].index.tolist()
                curr_bas = Xdata[sample_ixs, :]
                sample_bas.append(curr_bas)
                
        sample_labels.extend(g.loc[sample_ixs, 'config'].values.tolist())
        sample_array.append(curr_values)
    
    sample_array = np.vstack(sample_array)

    if clfparams['get_null']: # Can only do this if STAT_TYPE is 'stimulus' or 'post'
        random_draw = True
        print "Stim values:", sample_array.shape
        if random_draw:
            # Randomly select NTRIALS to use for baseline samples:
            selected_trials = random.sample(range(0, ntrials_total), ntrials)
        
        selected_bas_vals = np.vstack(sample_bas[selected_trials])
        
        sample_array = np.append(sample_array, selected_bas_vals, axis=0)
        print "Added null cases:", sample_array.shape
        sample_labels.extend(['bas' for _ in range(len(selected_bas_vals))])
    
    return sample_array, np.array(sample_labels)
     
    
#%%
#def extract_options(options):
#
#    def comma_sep_list(option, opt, value, parser):
#        setattr(parser.values, option.dest, value.split(','))
#
#
#    parser = optparse.OptionParser()
#
#    parser.add_option('-D', '--root', action='store', dest='rootdir',
#                          default='/nas/volume1/2photon/data',
#                          help='data root dir (dir w/ all animalids) [default: /nas/volume1/2photon/data, /n/coxfs01/2pdata if --slurm]')
#    parser.add_option('-i', '--animalid', action='store', dest='animalid',
#                          default='', help='Animal ID')
#
#    # Set specific session/run for current animal:
#    parser.add_option('-S', '--session', action='store', dest='session',
#                          default='', help='session dir (format: YYYMMDD_ANIMALID')
#    parser.add_option('-A', '--acq', action='store', dest='acquisition',
#                          default='FOV1', help="acquisition folder (ex: 'FOV1_zoom3x') [default: FOV1]")
##    parser.add_option('-T', '--trace-type', action='store', dest='trace_type',
##                          default='raw', help="trace type [default: 'raw']")
##    parser.add_option('-R', '--run', dest='run_list', default=[], nargs=1,
##                          action='append',
##                          help="run ID in order of runs")
##    parser.add_option('-t', '--traceid', dest='traceid_list', default=[], nargs=1,
##                          action='append',
##                          help="trace ID in order of runs")
##    parser.add_option('-n', '--nruns', action='store', dest='nruns', default=1, help="Number of consecutive runs if combined")
#    parser.add_option('-R', '--run', action='store', dest='run',
#                          default='', help="RUN name (e.g., gratings_run1)")
#    parser.add_option('-t', '--traceid', action='store', dest='traceid',
#                          default='', help="traceid name (e.g., traces001)")
#    
#    parser.add_option('--slurm', action='store_true', dest='slurm', default=False, help="set if running as SLURM job on Odyssey")
#    parser.add_option('--par', action='store_true', dest='multiproc', default=False, help="set if want to run MP on roi stats, when possible")
#    parser.add_option('--nproc', action='store', dest='nprocesses', default=4, help="N processes if running in par (default=4)")
#
#    # Classifier info:
#    parser.add_option('-r', '--rois', action='store', dest='roi_selector', default='all', help="(options: all, visual)")
#    parser.add_option('-d', '--dtype', action='store', dest='data_type', default='stat', help="(options: frames, stat)")
#    stat_choices = {'stat': ['meanstim', 'meanstimdff', 'zscore'],
#                    'frames': ['trial', 'stimulus', 'post']}
#    parser.add_option('-s', '--stype', action='store', dest='stat_type', default='meanstim', 
#                      help="If dtype is STAT, options: %s. If dtype is FRAMES, options: %s" % (str(stat_choices['stat']), str(stat_choices['frames'])))
#
#    parser.add_option('-p', '--indata_type', action='store', dest='inputdata_type', default='corrected', help="data processing type (dff, corrected, raw, etc.)")
#    parser.add_option('--null', action='store_true', dest='get_null', default=False, help='Include a null class in addition to stim conditions')
#    parser.add_option('-N', '--name', action='store', dest='class_name', default='', help='Name of transform to classify (e.g., ori, xpos, morphlevel, etc.)')
#    
##    choices_agg = ('all', 'single', 'averagereps', 'collapse')
##    default_agg = 'all'
##    parser.add_option('-z', '--agg', dest='aggregate_type', type="choice", choices=choices_agg, default=default_agg, 
##                      help='Aggregate method. Valid choices: %s. Default %s' % (choices_agg, default_agg)) 
##    
##    choices_subset = (None, 'two_class', 'no_morphing')
##    default_subset = None
##    parser.add_option('--subset', dest='subset', type='choice', choices=choices_subset, default=default_subset,
##                      help='Set if only want to consider subset of data. Valid choices: %s. Default %s' % (choices_subset, default_subset)) 
##    parser.add_option('--subset-samples', action='store', dest='subset_nsamples', default=None, help='N samples to draw if aggregate_type=half, but want to use N other than half')
#    
#    parser.add_option('--subset', action='store', dest='class_subset', default='', help='Subset of class_name types to learn')
#    parser.add_option('-c', '--const', dest='const_trans', default='', type='string', action='callback', 
#                          callback=comma_sep_list, help="Transform name to hold constant if classifying a different transform")
#    parser.add_option('-v', '--tval', dest='trans_value', default='', type='string', action='callback', 
#                          callback=comma_sep_list, help="Value to set const_trans to")
#
#    #parser.add_option('-c', '--const', action='store', dest='const_trans', default='', help='Transform name to hold constant if classifying a different transform')
#    #parser.add_option('-v', '--tval', action='store', dest='trans_value', default='', help='Value to set const_trans to')
#    
#    parser.add_option('-L', '--clf', action='store', dest='classifier', default='LinearSVC', help='Classifier type (default: LinearSVC)')
#    parser.add_option('-k', '--cv', action='store', dest='cv_method', default='kfold', help='Method of cross-validation (default: kfold)')
#    parser.add_option('-f', '--folds', action='store', dest='cv_nfolds', default=5, help='N folds for CV (default: 5)')
#    parser.add_option('-C', '--cval', action='store', dest='C_val', default=1e9, help='Value for C param if using SVC (default: 1e9)')
#    parser.add_option('-g', '--groups', action='store', dest='cv_ngroups', default=1, help='N groups for CV, relevant only for data_type=frames (default: 1)')
#    parser.add_option('-b', '--bin', action='store', dest='binsize', default=10, help='Bin size, relevant only for data_type=frames (default: 10)')
##    parser.add_option('--combine', action='store_true', dest='combine_data', default=False, help='Flag to combine multiple runs of the same thing')
##    parser.add_option('--combo', action='store', dest='combo_name', default='combo', help='Name of new, combined dataset (default: combined)')
#
#    (options, args) = parser.parse_args(options)
#    
#    assert options.stat_type in stat_choices[options.data_type], "Invalid STAT selected for data_type %s. Run -h for options." % options.data_type
#
#    return options

                    
            
#%%
class TransformClassifier():
    
    def __init__(self, animalid, session, acquisition, run, traceid, rootdir='/n/coxfs01/2p-data',
                         roi_selector='visual', data_type='stat', stat_type='meanstim',
                         inputdata_type='corrected', 
                         get_null=False, class_name='', class_subset='',
                         const_trans='', trans_value='',
                         cv_method='kfold', cv_nfolds=5, cv_ngroups=1, C_val=1e9, binsize=10):
        
        self.rootdir = rootdir
        self.animalid = animalid
        self.session = session
        self.acquisition = acquisition
        self.run = run
        if 'cnmf' in traceid:
            tracedir_type = 'cnmf'
        else:
            tracedir_type = 'traces'
        self.traceid_dir = glob.glob(os.path.join(rootdir, animalid, 
                                              session, acquisition, 
                                              run, tracedir_type, 
                                              '%s*' % traceid))[0]
        self.traceid = os.path.split(self.traceid_dir)[-1]        
        self.data_fpath = self.get_data_fpath()
        self.classifiers = []

        train_params = {'roi_selector': roi_selector,
                        'data_type': data_type,
                        'stat_type': stat_type,
                        'inputdata_type': inputdata_type,
                        'get_null': get_null,
                        'class_name': class_name,
                        'class_subset': class_subset,
                        'const_trans': const_trans,
                        'trans_value': trans_value,
                        'binsize': binsize,
                        'cv_method': cv_method,
                        'cv_nfolds': cv_nfolds,
                        'cv_ngroups': cv_ngroups,
                        'C_val': C_val}
        self.set_params(train_params) # Set up classifier parameters
        
                
    def get_data_fpath(self):

        # Data array dir:
        data_basedir = os.path.join(self.traceid_dir, 'data_arrays')
        data_fpath = os.path.join(data_basedir, 'datasets.npz')
        assert os.path.exists(data_fpath), "[E]: Data array not found! Did you run tifs_to_data_arrays.py?"
    
        # Create output base dir for classifier results:
        clf_basedir = os.path.join(self.traceid_dir, 'classifiers')
        if not os.path.exists(os.path.join(clf_basedir, 'figures')):
            os.makedirs(os.path.join(clf_basedir, 'figures'))
            
        return data_fpath
            

    def load_dataset(self):
        # Store DATASET:            
        dt = np.load(self.data_fpath)
        if 'arr_0' in dt.keys():
            self.dataset = dt['arr_0'][()]
        else:
            self.dataset = dt           
        
        # Store run info:
        if isinstance(self.dataset['run_info'], dict):
            self.run_info = self.dataset['run_info']
        else:
            self.run_info = self.dataset['run_info'][()]

        # Store stim configs:
        if isinstance(self.dataset['sconfigs'], dict):    
            self.sconfigs = self.dataset['sconfigs']
        else:
            self.sconfigs = self.dataset['sconfigs'][()]

        self.data_identifier = '_'.join((self.animalid, self.session, self.acquisition, self.run, self.traceid))
        
        self.sample_data, self.sample_labels = self.get_formatted_data()
        

    def get_formatted_data(self): #get_training_data(self):
        '''
        Returns input data formatted as:
            ntrials x nrois (data_type=='stat')
            nframes x nrois (data_type = 'frames')
        Filters nrois by roi_selector.
        '''
        # Get data array:
        assert self.params['inputdata_type'] in self.dataset.keys(), "Specified dtype %s not found. Select from %s." % (self.params['data_type'], str(self.dataset.keys()))
        Xdata = np.array(self.dataset[self.params['inputdata_type']])
        
        # Get subset of ROIs, if roi_selector is not 'all':
        self.rois = self.load_roi_list(roi_selector=self.params['roi_selector'])
        if self.rois is not None:
            print "Selecting %i out of %i ROIs (selector: %s)" % (len(self.rois), Xdata.shape[-1], self.params['roi_selector'])
            Xdata = Xdata[:, self.rois]
        
        # Determine whether all trials have the same structure or not:
        multiple_durs = isinstance(self.run_info['nframes_on'], list)

        # Make sure all conds have same N trials:
        ntrials_by_cond = self.run_info['ntrials_by_cond']
        ntrials_tmp = list(set([v for k, v in ntrials_by_cond.items()]))
        assert len(ntrials_tmp)==1, "Unequal reps per condition!"
        labels_df = pd.DataFrame(data=self.dataset['labels_data'], columns=self.dataset['labels_columns'])
        
        if self.params['data_type'] == 'stat':
            cX, cy = get_stat_samples(Xdata, labels_df, self.params, multiple_durs=multiple_durs)
        else:
            cX, cy = get_frame_samples(Xdata, labels_df, self.params)
            
        print "Ungrouped dataset cX:", cX.shape
        print "Ungrouped dataset labels cy:", cy.shape
        
        return cX, cy
            
    def load_roi_list(self, roi_selector='visual'):
        
        if roi_selector == 'all':
            roi_list = None
        else:
            roistats_results_fpath = os.path.join(self.traceid_dir, 'sorted_rois', 'roistats_results.npz')
            roistats = np.load(roistats_results_fpath)
            
            roi_subset_type = 'sorted_%s' % roi_selector
            roi_list = roistats[roi_subset_type]
        
        return roi_list
    
    def set_params(self, train_params):
        # Check arg validity:
        valid_choices = {
                        'data_type': ['frames', 'stat'],
                        'stat_type': ['meanstim', 'zscore', 'meanstimdiff'],
                        'cv_method': ['kfold', 'splithalf', 'LOGO', 'LOO', 'LPGO']
                        }
        for opt, choices in valid_choices.items():
            assert train_params[opt] in choices, "Specified %s --%s-- NOT valid. Select from %s" % (opt, train_params[opt], str(choices))                
        
        self.params = get_classifier_params(
                                    classifier = 'LinearSVC', 
                                    cv_method = train_params['cv_method'], 
                                    cv_nfolds = int(train_params['cv_nfolds']),
                                    cv_ngroups = train_params['cv_ngroups'],
                                    C_val = train_params['C_val'],
                                    roi_selector = train_params['roi_selector'],         # Can be 'all', or 'visual' or 'selective' (must have sorted ROIs)
                                    data_type = train_params['data_type'],               # Can be 'stat' or 'frames' 
                                    stat_type = train_params['stat_type'],               # Can be 'meanstim', 'zscore', 'meanstimdff'
                                    inputdata_type = train_params['inputdata_type'],     # is '', unless CNMF traces, in which case 'corrected', 'spikes', etc.
                                    get_null = train_params['get_null'],                 # Whether null-class should be trained
                                    class_name = train_params['class_name'],             # Should be a transform (ori, morphlevel, xpos, ypos, size)
                                    #aggregate_type = optsE.aggregate_type,     # Should be 'all', 'single', 'half' -- TOD:  add multiepl trans-constants
                                    const_trans = train_params['const_trans'],           # '' Transform type to hold at a constant value (must be different than class_name)
                                    trans_value = train_params['trans_value'],           # '' Transform value to hold const_trans at
                                    class_subset = [float(c) if c.isdigit() else c for c in train_params['class_subset']],         # LIST of subset of class_name types to include
                                    #subset_nsamples = optsE.subset_nsamples,   #**# None; TODO:  fix this and 'subset' options -- these make no sense
                                    binsize = train_params['binsize'] if train_params['data_type'] =='frames' else '')
    
    def create_classifier_dirs(self):
        # What are we classifying:
        if len(self.params['class_subset']) > 0:
            class_list = self.params['class_subset']
        else:
            class_list = pd.DataFrame(self.sconfigs).T[self.params['class_name']].unique().tolist()
        nclasses = len(class_list)
        classes_desc = '%i%s' % (nclasses, self.params['class_name'])
        if self.params['get_null']:
            classes_desc = '%s_plusnull' % classes_desc
    
        # Is there a subgroup of class_name that we want to train the classifier on:
        if self.params['const_trans'] is not '':
            const_trans_dict = self.get_constant_transforms()
            transforms_desc = '_'.join('%i%s' % (len(v), k) for k,v in const_trans_dict.items())
        else:
            const_trans_dict = None
            transforms_desc = 'alltransforms'
        
        # What is the input data type:
        data_desc = '%s_%s_%s' % (self.params['data_type'], self.params['inputdata_type'], self.params['stat_type'])

        classif_identifier = '{clf}_{cd}_{td}_{rs}_{dd}'.format(clf=self.params['classifier'],
                                                           cd=classes_desc,
                                                           td = transforms_desc,
                                                           rs='%srois' % self.params['roi_selector'],
                                                           dd = data_desc)
        
        # Set output dirs:
        self.classifier_dir = os.path.join(self.traceid_dir, 'classifiers', classif_identifier)
        self.const_trans_dict = const_trans_dict
        print "Creating CLF base dir:", self.classifier_dir
        print "Training classifiers on the following constant transform values:", self.const_trans_dict
            
    # CLASSIFIER CREATION:
    
    def get_constant_transforms(self):
        # Select only those samples w/ values equal to specificed transform value.
        # 'const_trans' :  transform type desired
        # 'trans_value' :  value of const_trans to use.
        
        sconfigs_df = pd.DataFrame(self.sconfigs).T
        const_trans = [trans.strip() for trans in self.params['const_trans']]
        trans_value = self.params['trans_value']
        if trans_value == '': 
            # Get all values for specified const_trans:
            trans_value = []
            for trans in const_trans:
                trans_value.append(sorted(sconfigs_df[trans].unique().tolist()))
        
        # Check that all provided trans-values are valid and get config IDs to include:
        const_trans_dict = dict((k, v) for k,v in zip([t for t in const_trans], [v for v in trans_value]))
        
        return const_trans_dict
    
    
    def initialize_classifiers(self):

        if self.const_trans_dict is not None:
            keys, values = zip(*self.const_trans_dict.items())
            transforms = [dict(zip(keys, v)) for v in itertools.product(*values)]
        else:
            transforms = []
            
        # If we are testing subset of the data (const_trans and trans_val are non-empty),
        # create a classifier + output subdirs for each subset:
        if len(transforms) > 0:
            for transform in transforms:
                curr_clfparams = self.params.copy()
                curr_clfparams['const_trans'] = transform.keys()
                curr_clfparams['trans_value'] = sorted(transform.values(), key=lambda x: transform.keys())
                self.classifiers.append(LinearSVM(curr_clfparams, 
                                                  self.sample_data, 
                                                  self.sample_labels, 
                                                  self.sconfigs, 
                                                  self.classifier_dir,
                                                  self.run_info,
                                                  data_identifier=self.data_identifier))
        else:
            self.classifiers.append(LinearSVM(self.params, 
                                              self.sample_data,
                                              self.sample_labels,
                                              self.sconfigs,
                                              self.classifier_dir,
                                              self.run_info,
                                              data_identifier=self.data_identifier))
            
    def label_classifier_data(self):
        for ci, clf in enumerate(self.classifiers):
            clf.label_training_data()
            clf.create_classifier()
            print "Created %i of %i classifiers: %s" % (ci+1, len(self.classifiers), clf.classifier_dir)
            
#    
#    def train_classifier(self, clf):
#        print "Training classifier.\n--- output saved to: %s" % clf.classifier_dir
#        if self.params['data_type'] == 'frames':
#            self.train_on_trial_epochs(clf)
#        else:
#            self.train_on_trials(clf)
#            
#    def train_on_trial_epochs(self, clf):
#        
#        epochs, decode_dict, bins, clf.clfparams['binsize'], class_labels = \
#                                        format_epoch_dataset(clf.clfparams, clf.cX, clf.cy, self.run_info, self.sconfigs)
#                                        
#        decode_trial_epochs(clf.class_labels, clf.clfparams, bins, decode_dict, 
#                                self.run_info, data_identifier=self.data_identifier, 
#                                niterations=10, scoring='accuracy', output_dir=clf.classifier_dir)
#        
#    def train_on_trials(self, clf):
#        
#        print "... running permutation test for CV accuracy."
#        clf.cv_kfold_permutation(data_identifier=self.data_identifier,
#                                  scoring='accuracy', 
#                                  permutation_test=True, 
#                                  n_permutations=500)
#        
#        print "... plotting confusion matrix."
#        clf.confusion_matrix(data_identifier=self.data_identifier)
#        
#        print "... doing RFE."
#        clf.do_RFE(data_identifier=self.data_identifier, scoring='accuracy')
    
    def save_me(self):
        with open(os.path.join(self.classifier_dir, 'TransformClassifier.pkl'), 'wb') as f:
            pkl.dump(self, f, protocol=pkl.HIGHEST_PROTOCOL)
    
#%%
class LinearSVM():
    
    def __init__(self, clfparams, sample_data, sample_labels, sconfigs, classifier_dir,
                         run_info, data_identifier=''):
        self.clfparams = clfparams
        self.cX = sample_data
        self.cy = sample_labels
        self.sconfigs = sconfigs
        self.classifier_dir = classifier_dir
        self.run_info = run_info
        self.data_identifier = data_identifier
    
        if self.clfparams['const_trans'] is not '':
            # Create SUBDIR for specific const-trans and trans-val pair:
            const_trans_dict = dict((k, v) for k,v in zip([t for t in self.clfparams['const_trans']], [v for v in self.clfparams['trans_value']]))
            transforms_desc = '_'.join('%s_n%.1f' % (k, abs(v)) if v < 0 else '%s_%.1f' % (k, v) for k,v in const_trans_dict.items())
            self.classifier_dir = os.path.join(self.classifier_dir, transforms_desc)
            
        # Set output dirs:
        if not os.path.exists(os.path.join(self.classifier_dir, 'figures')):
            os.makedirs(os.path.join(self.classifier_dir, 'figures'))
        if not os.path.exists(os.path.join(self.classifier_dir, 'results')):
            os.makedirs(os.path.join(self.classifier_dir, 'results'))
            
        self.results = {}
            
    def label_training_data(self):
        
        if self.clfparams['const_trans'] != '' and self.clfparams['class_subset'] == '':
            # Only group and take subset of data for specified const-trans/trans-value pair
            cX, cy, class_labels = self.group_by_transform_subset()
        
        elif self.clfparams['class_subset'] != '' and self.clfparams['const_trans'] == '':
            # Only group and take subset of data for subset of class-to-be-trained (across all transforms)
            cX, cy, class_labels = self.group_by_class_subset()
        
        elif self.clfparams['class_subset'] != '' and self.clfparams['const_trans'] != '':
            # Only group and take subset of data for class subset within a sub-subset of specified cons-trans/trans-value pair
            cX, cy, class_labels = self.group_by_class_and_transform_subset()
        
        else:
            cX, cy, class_labels = self.group_by_class()
            
        self.cX = StandardScaler().fit_transform(cX)
        self.cy = cy
        self.class_labels = class_labels

        # Add finalized info to clfparams:        
        self.clfparams['dual'] = cX.shape[0] > cX.shape[1]
        
    
    def group_by_class(self):
        
        cX = self.cX
        cy = np.array([self.sconfigs[cv][self.clfparams['class_name']] if cv != 'bas' else 'bas' for cv in self.cy])
        class_labels = sorted(np.unique(cy))
        
        return cX, cy, class_labels
    
    def group_by_class_subset(self):
        '''
        Only grab samples belonging to subset of class types.
        Expects a list of included class types,
            e.g., can provide anchors if class_name is 'morphlevel', or can provide xpos values [-20, 20], etc.)
        
        For indexing purposes, cy should still be of form 'config001', 'config002', etc.
        
        Label assignment happens here.
        '''
        
        sconfigs_df = pd.DataFrame(self.sconfigs).T
        configs_included = sconfigs_df[sconfigs_df[self.clfparams['class_name']].isin(self.clfparams['class_subset'])].index.tolist()
        if self.clfparams['get_null']:
            configs_included.append('bas')
            
        kept_ixs = np.array([cix for cix, cname in enumerate(self.cy) if cname in configs_included])
        cX = self.cX[kept_ixs, :] 
        cy_tmp = self.cy[kept_ixs]
        
        cy = np.array([self.sconfigs[cname][self.clfparams['class_name']] if cname != 'bas' else 'bas' for cname in cy_tmp])
        class_labels = sorted(np.unique(cy))
        
        return cX, cy, class_labels


    def group_by_transform_subset(self):
        const_trans_dict = dict((k, v) if isinstance(v, list) else (k, [v]) for k,v in zip([t for t in self.clfparams['const_trans']], [v for v in self.clfparams['trans_value']]))
        sconfigs_df = pd.DataFrame(self.sconfigs).T
        configs_included = []; configs_pile = self.sconfigs.keys()
        for transix, (trans_name, trans_value) in enumerate(const_trans_dict.items()):
            assert trans_value in sconfigs_df[trans_name].unique(), "Specified trans_name, trans_value not found: %s" % str((trans_name, trans_value))
            
            if transix == 0:
                first_culling = sconfigs_df[sconfigs_df[trans_name].isin(trans_value)].index.tolist()
                configs_tmp = [c for c in configs_pile if c in first_culling]
            else:
                configs_pile = copy.copy(configs_tmp)
                first_culling = sconfigs_df[sconfigs_df[trans_name].isin(trans_value)].index.tolist()
                configs_tmp = [c for c in configs_pile if c in first_culling]
        configs_included = configs_tmp
    
        if self.clfparams['get_null']:
            configs_included.append('bas')
            
        kept_ixs = np.array([cix for cix, cname in enumerate(self.cy) if cname in configs_included])
        cX = self.cX[kept_ixs, :] 
        cy_tmp = self.cy[kept_ixs]
        
        cy = np.array([self.sconfigs[cname][self.clfparams['class_name']] if cname != 'bas' else 'bas' for cname in cy_tmp])
        class_labels = sorted(np.unique(cy))
        
        return cX, cy, class_labels


    def group_by_class_and_transform_subset(self):
        const_trans_dict = dict((k, v) if isinstance(v, list) else (k, [v]) for k,v in zip([t for t in self.clfparams['const_trans']], [v for v in self.clfparams['trans_value']]))
        sconfigs_df = pd.DataFrame(self.sconfigs).T
        configs_subset = sconfigs_df[sconfigs_df[self.clfparams['class_name']].isin(self.clfparams['class_subset'])].index.tolist()
        sconfigs_df = sconfigs_df[sconfigs_df.index.isin(configs_subset)]
        
        configs_included = []; configs_pile = self.sconfigs.keys()
        for transix, (trans_name, trans_value) in enumerate(const_trans_dict.items()):
            assert trans_value in sconfigs_df[trans_name].unique(), "Specified trans_name, trans_value not found: %s" % str((trans_name, trans_value))
            
            if transix == 0:
                first_culling = sconfigs_df[sconfigs_df[trans_name].isin(trans_value)].index.tolist()
                configs_tmp = [c for c in configs_pile if c in first_culling]
            else:
                configs_pile = copy.copy(configs_tmp)
                first_culling = sconfigs_df[sconfigs_df[trans_name].isin(trans_value)].index.tolist()
                configs_tmp = [c for c in configs_pile if c in first_culling]
        configs_included = configs_tmp
    
        if self.clfparams['get_null']:
            configs_included.append('bas')
            
        kept_ixs = np.array([cix for cix, cname in enumerate(self.cy) if cname in configs_included])
        cX = self.cX[kept_ixs, :] 
        cy_tmp = self.cy[kept_ixs]
        
        cy = np.array([self.sconfigs[cname][self.clfparams['class_name']] if cname != 'bas' else 'bas' for cname in cy_tmp])
        class_labels = sorted(np.unique(cy))
        
        return cX, cy, class_labels
    
    
    def create_classifier(self):
        
#        nclasses = len([l for l in self.class_labels if l != 'bas'])
#        classes_desc = '%i%s' % (nclasses, self.clfparams['class_name'])
#        if self.clfparams['get_null']:
#            classes_desc = '%s_plusnull' % classes_desc

            
        # Save clfparams and check file hash:
        clfparams_hash = hashlib.md5(json.dumps(self.clfparams, ensure_ascii=True, indent=4, sort_keys=True)).hexdigest()
        clfparams_fpath = os.path.join(self.classifier_dir, 'clfparams_%s.json' % clfparams_hash[0:6])
        with open(clfparams_fpath, 'w') as f: 
            json.dump(self.clfparams, f, indent=4, sort_keys=True)
        
        self.hash = clfparams_hash
        
        self.clf = LinearSVC(random_state=0, dual=self.clfparams['dual'], multi_class='ovr', C=self.clfparams['C_val'])

#            
#    def correlation_matrix(self):
#        '''
#        Correlation matrix only works with data_type == 'stat'.
#        '''
#        # Create output dir for population-level figures:
#        output_dir = os.path.join(self.data.traceid_dir, 'figures', 'population')
#        if not os.path.exists(output_dir): os.makedirs(output_dir)
#            
#        correlation_matrix(self.clfparams, self.class_labels, self.cX, self.cy, 
#                               data_identifier=self.data.data_identifier, output_dir=output_dir)
        

    def train_classifier(self):
        print "Training classifier.\n--- output saved to: %s" % self.classifier_dir
        if self.clfparams['data_type'] == 'frames':
            self.train_on_trial_epochs()
        else:
            self.train_on_trials()
            
    def train_on_trial_epochs(self):
        
        epochs, decode_dict, bins, self.clfparams['binsize'], class_labels = \
                                        format_epoch_dataset(self.clfparams, self.cX, self.cy, self.run_info, self.sconfigs)
                                        
        decode_trial_epochs(self.class_labels, self.clfparams, bins, decode_dict, 
                                self.run_info, data_identifier=self.data_identifier, 
                                niterations=10, scoring='accuracy', output_dir=self.classifier_dir)
        
    def train_on_trials(self):
        
        print "... running permutation test for CV accuracy."
        self.cv_kfold_permutation(data_identifier=self.data_identifier,
                                  scoring='accuracy', 
                                  permutation_test=True, 
                                  n_permutations=500)
        
        print "... plotting confusion matrix."
        self.confusion_matrix(data_identifier=self.data_identifier)
        
        print "... doing RFE."
        self.do_RFE(data_identifier=self.data_identifier, scoring='accuracy')
        
        
#    def train_on_trial_epochs(self, data_identifier=''):
#        
#        epochs, decode_dict, bins, self.clfparams['binsize'], class_labels = \
#                                        format_epoch_dataset(self.clfparams, self.cX, self.cy, self.data.run_info, self.data.sconfigs)
#        decode_trial_epochs(self.class_labels, self.clfparams, bins, decode_dict, 
#                                self.data.run_info, data_identifier=data_identifier, 
#                                niterations=10, scoring='accuracy', output_dir=self.classifier_dir)
        
    def cv_kfold_permutation(self, scoring='accuracy', permutation_test=True, n_jobs=4, n_permutations=500, data_identifier=''):
        # -----------------------------------------------------------------------------
        # Do cross-validation
        # -----------------------------------------------------------------------------
        kfold = StratifiedKFold(n_splits=self.clfparams['cv_nfolds'], shuffle=True)
    
        cv_results = cross_val_score(self.clf, self.cX, self.cy, cv=kfold, scoring=scoring)
        print "CV RESULTS: %f (%f)" % (cv_results.mean(), cv_results.std())
        self.results['cv_results'] = cv_results
        
        if permutation_test:
            # -----------------------------------------------------------------------------
            # How significant is our classification score(s)?
            # Calculate p-value as percentage of runs for which obtained score is greater 
            # than the initial classification score (i.e., repeat classification after
            # randomizing and permuting labels).
            # -----------------------------------------------------------------------------
            score, permutation_scores, pvalue = permutation_test_score(
                                                    self.clf, self.cX, self.cy, 
                                                    scoring=scoring, 
                                                    cv=kfold, 
                                                    n_permutations=n_permutations, 
                                                    n_jobs=n_jobs)
            
            print("Classification score %s (pvalue : %s)" % (score, pvalue))

            # View histogram of permutation scores            
            fig = pl.figure()
            n_classes = np.unique([self.cy]).size
            
            pl.hist(permutation_scores, 20, label='Permutation scores', edgecolor='black')
            ylim = pl.ylim()
            pl.plot(2 * [score], ylim, '--g', linewidth=3, label='Classification Score %.4f (p=%.4f)' % (score, pvalue))
            pl.plot(2 * [1. / n_classes], ylim, '--k', linewidth=3, label='Chance')
            pl.ylim(ylim)
            pl.legend()
            pl.xlabel('Score - %s' % scoring)
            
            if self.clf.C == 1: Cstring = 'C1';
            elif self.clf.C == 1E9: Cstring = 'bigC';
            else: Cstring = 'C%i' % self.clf.C
                
            label_figure(fig, data_identifier)
            figname = 'cv_permutation_test_%s_%s_%i.png' % (Cstring, self.clfparams['cv_method'], self.clfparams['cv_nfolds'])            
            pl.savefig(os.path.join(self.classifier_dir, 'figures', figname))
            pl.close()
            
            
    def confusion_matrix(self, data_identifier=''):
                
        predicted, true, classes = get_cv_folds(self.clf, self.clfparams, self.cX, self.cy, output_dir=self.classifier_dir)
        plot_confusion_matrix_subplots(predicted, true, classes, cv_method=self.clfparams['cv_method'], 
                                       data_identifier=data_identifier, output_dir=self.classifier_dir)
        self.results['test'] = {'predicted': predicted,
                                'true': true,
                                'classes': classes}

    def do_RFE(self, scoring='accuracy', data_identifier=''):
        results_topN = iterate_RFE(self.clfparams, self.cX, self.cy, scoring=scoring, 
                                       output_dir=self.classifier_dir)
        plot_RFE_results(results_topN, len(self.class_labels), scoring=scoring, 
                                     data_identifier=data_identifier, 
                                     output_dir=self.classifier_dir)
    
#%%
    
#
#    
#
#    if data_type == 'frames':
#        cX_std = StandardScaler().fit_transform(cX)
#        epochs, decode_dict, bins, clfparams['binsize'] = format_epoch_dataset(clfparams, cX, cy, run_info, sconfigs)
#        class_labels = sorted(list(set([sconfigs[c][clfparams['class_name']] for c in sconfigs.keys()])))
#
#        svc = decode_trial_epochs(clfparams, bins, decode_dict, run_info, data_identifier=data_identifier, 
#                            niterations=10, scoring='accuracy')
#    
#    
#    #%
#    # -----------------------------------------------------------------------------
#    # Save classifier and formatted data:
#    # -----------------------------------------------------------------------------
#        
#    clf_fpath = os.path.join(classifier_dir, '%s_datasets.npz' % classif_identifier)
#    np.savez(clf_fpath, cX=cX, cX_std=cX_std, cy=cy,
#                 data_type=data_type,
#                 inputdata=inputdata,
#                 inputdata_type=inputdata_type,
#                 data_fpath=data_fpath,
#                 sconfigs=sconfigs, run_info=run_info)
#
#    joblib.dump(svc, os.path.join(classifier_dir, '%s.pkl' % classif_identifier), compress=9)
#     
#    svc_params = svc.get_params().copy()
#    if 'base_estimator' in svc_params.keys():
#        svc_params['base_estimator'] = str(svc_params['base_estimator'] )
#    #clf_params['cv'] = str(clf_params['cv'])
#    svc_params['identifier'] = classif_identifier
#    svc_params_hash = hash(json.dumps(svc_params, sort_keys=True, ensure_ascii=True)) % ((sys.maxsize + 1) * 2) #[0:6]
#    
#    with open(os.path.join(classifier_dir, 'params_%s.json' % svc_params_hash), 'w') as f:
#        json.dump(svc_params, f, indent=4, sort_keys=True, ensure_ascii=True)
#    
#    clfparams['classifier_info'] = svc_params
#    
#    with open(os.path.join(classifier_dir, 'classifier_params.json'), 'w') as f:
#        json.dump(clfparams, f, indent=4, sort_keys=True, ensure_ascii=True)
#        
#        
#    #%%
#    if clfparams['data_type'] == 'stat':
#  
#        # Visualize feature weights:
#        # =============================================================================
#            
#        # svc.coef_ :  array shape [n_classes, n_features] (if n_classes=2, shape [n_features])
#        # svc.coef_ :  array shape [n_classes, n_features] (if n_classes=2, shape [n_features])
#    
#        # Sort the weights by their strength, take out bottom N rois, iterate.
#        fig = plot_weight_matrix(svc, absolute_value=True)
#        label_figure(fig, data_identifier)
#        pl.savefig(os.path.join(classifier_dir, 'figures', 'sorted_weights_abs.png'))
#        pl.close()
#        
#        fig = plot_weight_matrix(svc, absolute_value=False)
#        label_figure(fig, data_identifier)
#        pl.savefig(os.path.join(classifier_dir, 'figures', 'sorted_weights_raw.png'))
#        pl.close()
#        
#        nrois = len(run_info['roi_list'])
#        for class_idx in range(len(svc.classes_)):
#            fig = plot_coefficients(svc, xrange(nrois), class_idx=class_idx, top_features=20)
#            label_figure(fig, data_identifier)
#            pl.savefig(os.path.join(classifier_dir, 'figures', 'sorted_feature_weights_%s.png' % class_idx))
#            pl.close()
        
        #%

    #%%
        
#    # #############################################################################
#    # Visualize the data
#    # #############################################################################
#    
#        
#    # Look at a few neurons to see what the data looks like for binary classif:
#    ddf = pd.DataFrame(data=cX_std,
#                          columns=range(nrois))
#    labels_df = pd.DataFrame(data=cy,
#                             columns=['class'])
#    
#    df = pd.concat([ddf, labels_df], axis=1).reset_index(drop=True)
#    
#    _ = sns.pairplot(df[:50], vars=[8, 11, 12, 14, 19], hue="class", size=1.5)
#    
#    
#    
#    # How correlated are the features (neurons) to class labels?  
#    # -----------------------------------------------------------------------------
#    
#    corr = ddf.corr()
#    
#    # Generate a mask for the upper triangle
#    mask = np.zeros_like(corr, dtype=np.bool)
#    mask[np.triu_indices_from(mask)] = True
#    
#    # Set up the matplotlib figure
#    f, ax = pl.subplots(figsize=(11, 9))
#    
#    # Generate a custom diverging colormap
#    cmap = sns.diverging_palette(220, 10, as_cmap=True)
#    
#    # Draw the heatmap with the mask and correct aspect ratio
#    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
#                square=True, linewidths=.5, cbar_kws={"shrink": .5})
#    
#    
#      
#    #%%
#    
#    # #############################################################################
#    # Learning curves
#    # #############################################################################
#    
#    
#    # Modified from http://scikit-learn.org/stable/auto_examples/plot_learning_curve.html
#    from sklearn.learning_curve import learning_curve
#    def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
#                            train_sizes=np.linspace(.1, 1.0, 5)):
#        """
#        Generate a simple plot of the test and traning learning curve.
#    
#        Parameters
#        ----------
#        estimator : object type that implements the "fit" and "predict" methods
#            An object of that type which is cloned for each validation.
#    
#        title : string
#            Title for the chart.
#    
#        X : array-like, shape (n_samples, n_features)
#            Training vector, where n_samples is the number of samples and
#            n_features is the number of features.
#    
#        y : array-like, shape (n_samples) or (n_samples, n_features), optional
#            Target relative to X for classification or regression;
#            None for unsupervised learning.
#    
#        ylim : tuple, shape (ymin, ymax), optional
#            Defines minimum and maximum yvalues plotted.
#    
#        cv : integer, cross-validation generator, optional
#            If an integer is passed, it is the number of folds (defaults to 3).
#            Specific cross-validation objects can be passed, see
#            sklearn.cross_validation module for the list of possible objects
#        """
#        
#        pl.figure()
#        train_sizes, train_scores, test_scores = learning_curve(
#            estimator, X, y, cv=5, n_jobs=1, train_sizes=train_sizes)
#        train_scores_mean = np.mean(train_scores, axis=1)
#        train_scores_std = np.std(train_scores, axis=1)
#        test_scores_mean = np.mean(test_scores, axis=1)
#        test_scores_std = np.std(test_scores, axis=1)
#    
#        pl.fill_between(train_sizes, train_scores_mean - train_scores_std,
#                         train_scores_mean + train_scores_std, alpha=0.1,
#                         color="r")
#        pl.fill_between(train_sizes, test_scores_mean - test_scores_std,
#                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
#        pl.plot(train_sizes, train_scores_mean, 'o-', color="r", alpha=0.8,
#                 label="Training score")
#        pl.plot(train_sizes, test_scores_mean, 'o-', color="g", alpha=0.8,
#                 label="Cross-validation score")
#    
#        pl.xlabel("Training examples")
#        pl.ylabel("Score")
#        pl.legend(loc="best")
#        pl.grid("on") 
#        if ylim:
#            pl.ylim(ylim)
#        pl.title(title)
#        
#    # 
#    rand_order = random.sample(xrange(len(cy)), len(cy))
#    
#    
#    # 1.  Look at Training score vs. CV score.
#    # -----------------------------------------------------------------------------
#    # Training score is always at max, regardless of N training examples 
#    # --> OVERFITTING.
#    # CV score increases over time?  
#    # Gap between CV and training scores -- big gap ~ high variance!
#    # -----------------------------------------------------------------------------
#    plot_learning_curve(LinearSVC(C=.10), "LinearSVC(C=10.0)",
#                        cX_std, cy, ylim=(0.0, 1.01),
#                        cv = 5)
#    
#    
#    # 2.  Reduce the complexity -- fewer features.
#    # -----------------------------------------------------------------------------
#    # SelectKBest(f_classif, k=2) will select the k=2 best features according to their Anova F-value
#     
#    from sklearn.pipeline import Pipeline
#    from sklearn.feature_selection import SelectKBest, f_classif
#    
#    nfeatures = 10
#    C_val = 1E9
#    plot_learning_curve(Pipeline([("fs", SelectKBest(f_classif, k=nfeatures)),
#                                   ("svc", svc)]),
#                        "SelectKBest(f_classif, k=%i) + LinearSVC(C=%.2f)" % (nfeatures, C_val),
#                        cX_std, cy, ylim=(0.0, 1.0))
#    
#        
#    # 3.  Use better regularization term. 
#    # -----------------------------------------------------------------------------
#    # Increase classifier regularization (smaller C).
#    # Or, select C automatically w/ grid-search.
#    
#    from sklearn.grid_search import GridSearchCV
#    est = GridSearchCV(LinearSVC(), 
#                       param_grid={"C": [0.001, 0.01, 0.1, 1.0, 10.0]})
#    plot_learning_curve(est, "LinearSVC(C=AUTO)", 
#                        cX_std, cy, ylim=(0.0, 1.0))
#    print "Chosen parameter on 100 datapoints: %s" % est.fit(cX_std[:100], cy[:100]).best_params_
#    
#    
#    # 4.  Try L1 penalty for a sparser solution.
#    # -----------------------------------------------------------------------------
#    # With L1 penalty, implicit feature selection. Can look at learned coefficients
#    # to see how many are 0 (ignored feature), and which have the stongest weights.
#    
#    plot_learning_curve(LinearSVC(C=0.1, penalty='l1', dual=False), 
#                        "LinearSVC(C=0.1, penalty='l1')", 
#                        cX_std, cy, ylim=(0.0, 1.0))
#    
#    est = LinearSVC(C=0.1, penalty='l1', dual=False)
#    est.fit(cX_std[:100], cy[:100])  # fit on 100 datapoints
#    print "Coefficients learned: %s" % est.coef_
#    print "Non-zero coefficients: %s" % np.nonzero(est.coef_)[1]
#    
#    strongest_weights = np.where(np.abs(est.coef_) == np.abs(est.coef_).max())
#    print "Best features: %s" % str(strongest_weights)
#    
#    pl.figure()
#    sns.heatmap(est.coef_)

#%%
##%%
## Plot decision function with all data:
#
#svc.classes_
#decisionfunc = svc.decision_function(cX_std) #print cdata.shape
#
#pl.figure()
#g = sns.heatmap(decisionfunc, cmap='PRGn')
#
#indices = { value : [ i for i, v in enumerate(cnames) if v == value ] for value in list(set(cnames)) }
#ytick_indices = [(k, indices[k][-1]) for k in list(set(cnames))]
#g.set(yticks = [])
#g.set(yticks = [x[1] for x in ytick_indices])
#g.set(yticklabels = [x[0] for x in ytick_indices])
#g.set_yticklabels(g.get_yticklabels(), rotation=45, fontsize=8)
#g.set(xlabel = 'class')
#g.set(ylabel = 'trials by label')
#
#pl.title('%s (decision functions)' % data_type)
#
#figname = '%s__decisionfunction.png' % classif_identifier
#
#pl.savefig(os.path.join(classifier_dir, figname))
#
#
##%%
#    
## Project zscores onto normals:
#svc.classes_
#cdata = np.array([svc.coef_[c].dot(cX_std.T) + svc.intercept_[c] for c in range(len(svc.classes_))])
#print cdata.shape
#
#pl.figure()
#g = sns.heatmap(cdata, cmap='PRGn')
#
#indices = { value : [ i for i, v in enumerate(cnames) if v == value ] for value in list(set(cnames)) }
#xtick_indices = [(k, indices[k][-1]) for k in list(set(cnames))]
#g.set(xticks = [])
#g.set(xticks = [x[1] for x in xtick_indices])
#g.set(xticklabels = [x[0] for x in xtick_indices])
#g.set_xticklabels(g.get_xticklabels(), rotation=45, fontsize=8)
#g.set(xlabel = cell_unit)
#g.set(ylabel = 'class')
#
#pl.title('%s data proj onto normals' % data_type)
#
#figname = '%s__proj_normals.png' % classif_identifier
#
#pl.savefig(os.path.join(classifier_dir, figname))

#options = ['-D', '/mnt/odyssey', '-i', 'JC015', '-S', '20180915', '-A', 'FOV1_zoom2p7x',
#           '-R', 'combined_gratings_static', '-t', 'traces001',
#           '-r', 'visual', '-d', 'stat', '-s', 'meanstim',
#           '-p', 'corrected', '-N', 'ori',
#           '-c', 'xpos'
#           ]
#
#
#def train_test_linearSVC(options):
#
#    optsE = extract_options(options)
#        
#    C = LinearSVM(optsE)
#    C.get_training_data()
#    C.label_training_data()
#    C.initialize_classifier()
#    
#    if C.clfparams['data_type'] == 'frames':
#        C.train_on_trial_epochs()
#    else:
#        C.cv_kfold_permutation(scoring='accuracy', permutation_test=True, n_permutations=500)
#        C.confusion_matrix()
#        C.do_RFE(scoring='accuracy')
#    
#    
##%%
#
#def main(options):
#    train_test_linearSVC(options)
#
#
#
#if __name__ == '__main__':
#    main(sys.argv[1:])
