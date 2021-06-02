#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on  Apr 24 19:56:32 2020

@author: julianarhee
"""
import matplotlib as mpl
mpl.use('agg')
import os
import json
import glob
import copy
import copy
import itertools
import datetime
import time
import math
import pprint 
pp = pprint.PrettyPrinter(indent=4)
import traceback

import numpy as np
import pylab as pl
import seaborn as sns
import pandas as pd
import statsmodels as sm
import cPickle as pkl
import multiprocessing as mp
from functools import partial

from scipy import stats as spstats

from pipeline.python.classifications import experiment_classes as util
from pipeline.python.classifications import aggregate_data_stats as aggr
from pipeline.python.classifications import rf_utils as rfutils
from pipeline.python import utils as putils
from pipeline.python.eyetracker import dlc_utils as dlcutils

from pipeline.python.retinotopy import fit_2d_rfs as fitrf
from pipeline.python.rois.utils import load_roi_coords


from matplotlib.lines import Line2D
import matplotlib.patches as patches

import scipy.stats as spstats
import sklearn.metrics as skmetrics
import random

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC
from sklearn import preprocessing
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn import svm


def shuffle_pupil_labels(pupil_low, pupil_high):
    '''
    Shuffle pupil labels (assumes only 'low' and 'high'), but preserves # trials per config
    '''
    lo_=[]
    hi_=[]
    for cfg, low_g in pupil_low.groupby(['config']):
        high_g = pupil_high[pupil_high.config==cfg].copy()
        n_low = low_g.shape[0]
        n_high = high_g.shape[0]
        p_all = pd.concat([low_g, high_g], axis=0)
        p_shuffled = p_all.sample(frac=1).reset_index(drop=True)
        lo_shuff = p_shuffled.iloc[0:n_low]
        hi_shuff = p_shuffled.iloc[n_low:]
        lo_.append(lo_shuff)
        hi_.append(hi_shuff)
    pupil_low_shuffled = pd.concat(lo_, axis=0).reset_index()
    pupil_high_shuffled = pd.concat(hi_, axis=0).reset_index()
    
    return pupil_low_shuffled, pupil_high_shuffled


def balance_pupil_split(pupildf, feature_name='pupil_fraction', n_cuts=3,
                        match_cond=True, match_cond_name='size', equalize_after_split=True, 
                        equalize_by='config', common_labels=None, verbose=False, shuffle_labels=False):
    '''
    Split pupildf (trials) by "low" and "high" states, with diff options for balancing samples.
    match_cond (bool): 
        Set to split pupil distribution within condition (for ex., match pupil for each SIZE separately).
    match_cond_name (str):
        The condition to split by, if match_cond==True *(Should be a column in pupildf)
    equalize_after_split (bool):
        Set to equalize across stimulus configs AFTER doing pupil split.
    equalize_by (str):
        Condition to match across, e.g., config label, or, morphlevel, etc. *(Should be column in pupildf)
    '''
    from pipeline.python.eyetracker import dlc_utils as dlcutils
    if match_cond:
        assert match_cond_name in pupildf.columns, "Requested split within <%s>, but not found." % match_cond_name
        low_=[]; high_=[];
        for sz, sd in pupildf.groupby([match_cond_name]):
            p_low, p_high = dlcutils.split_pupil_range(sd, feature_name=feature_name, n_cuts=n_cuts)
            # if shuffle, shuffle labels, then equalize conds
            if shuffle_labels:
                n_low = p_low.shape[0]
                n_high = p_high.shape[0]
                p_all = pd.concat([p_low, p_high], axis=0)
                p_all_shuffled = p_all.sample(frac=1).reset_index(drop=True)
                p_low = p_all_shuffled.sample(n=n_low)
                unused_ixs = [i for i in p_all_shuffled['trial'].values if i not in p_low['trial'].values]
                p_high = p_all_shuffled[p_all_shuffled['trial'].isin(unused_ixs)] #p_all_shuffled.sample(n=n_high)
            if equalize_after_split:
                p_low_eq, p_high_eq = balance_samples_by_config(p_low, p_high, 
                                            config_label=equalize_by, common_labels=common_labels)
                if verbose:
                    print("... sz %i, pre/post balance. Low: %i/%i | High: %i/%i" \
                      % (sz, p_low.shape[0], p_low_eq.shape[0], p_high.shape[0], p_high_eq.shape[0]))
                low_.append(p_low_eq)
                high_.append(p_high_eq)
            else:
                low_.append(p_low)
                high_.append(p_high)
        high = pd.concat(high_)
        low = pd.concat(low_)
    else:
        low, high = dlcutils.split_pupil_range(pupildf, feature_name=pupil_feature, n_cuts=n_cuts)

    return low, high

def balance_samples_by_config(df1, df2, config_label='config', common_labels=None):
    '''
    Given 2 dataframes with config labels on each trial, match reps per config.
    config_label: if 'config', matches across all configs found (config001, config002, etc.)
    For ex., can also use "morphlevel" to just match counts of morphlevel (ignoring unique size, etc.)
    '''
    assert config_label in df1.columns, "Label <%s> not in df1 columns" % config_label
    assert config_label in df2.columns, "Label <%s> not in df2 columns" % config_label

    if common_labels is None:
        common_labels = np.intersect1d(df1[config_label].unique(), df2[config_label].unique())
        config_subset = False
    else:
        config_subset = True

    # Get equal counts for specified labels 
    df1_eq = aggr.equal_counts_df(df1[df1[config_label].isin(common_labels)], equalize_by=config_label)
    df2_eq = aggr.equal_counts_df(df2[df2[config_label].isin(common_labels)], equalize_by=config_label) 
    # Check that each partition has the same # of labels (i.e., if missing 1 label, will ahve diff #s)
    #df1_labels = df1_eq[df1_eq[config_label].isin(common_labels)][config_label].unique()
    #df2_labels = df2_eq[df2_eq[config_label].isin(common_labels)][config_label].unique()

 
    # Get min N reps per condition (based on N reps per train condition)
    min_reps_per = min([df1_eq[config_label].value_counts()[0], df2_eq[config_label].value_counts()[0]])
    
    # Draw min_reps_per samples without replacement
    if config_subset:
        # we set sample #s by train configs, and include the other trials for test
        df1_resampled = pd.concat([g.sample(n=min_reps_per, replace=False) if g.shape[0]>min_reps_per \
                                    else g for c, g in df1.groupby([config_label])])
        df2_resampled = pd.concat([g.sample(n=min_reps_per, replace=False) if g.shape[0]>min_reps_per \
                                    else g for c, g in df2.groupby([config_label])])

    else:
        df2_resampled = pd.concat([g.sample(n=min_reps_per, replace=False) \
                                    for c, g in df2_eq.groupby([config_label])])
        df1_resampled = pd.concat([g.sample(n=min_reps_per, replace=False) \
                                    for c, g in df1_eq.groupby([config_label])])

    return df1_resampled, df2_resampled


def get_percentile_shuffled(iterdf, metric='heldout_test_score'):
    # Use bootstrap distn to calculate percentiles
    s_=[]
    if 'cell' in iterdf.columns:
        if 'train_transform' in iterdf.columns:
            percentiles = group_iters_by_cell_and_transform(iterdf, metric=metric)
        else:
            percentiles = group_iters_by_cell(iterdf, metric=metric)
    else:
        if 'train_transform' in iterdf.columns:
            percentiles = group_iters_by_fov_and_transform(iterdf, metric=metric)
        else:
            percentiles = group_iters_by_fov(iterdf, metric=metric)
   
    return percentiles

def group_iters_by_cell(iterdf, metric='heldout_test_score'):
    # p_thr=0.05
    excl_cols = ['fit_time', 'iteration', 'n_cells',  'randi', 'score_time']
    incl_cols = [c for c in iterdf.columns if c not in excl_cols]

    s_=[]
    for (visual_area, datakey, rid), d_ in iterdf.groupby(['visual_area', 'datakey', 'cell']):

        mean_score = d_[d_['condition']=='data'][metric].mean()
        percentile = np.mean(mean_score < d_[d_['condition']=='shuffled'][metric])
        n_iterations = d_[d_['condition']=='data'].shape[0]
        rdict = dict(d_[d_['condition']=='data'][incl_cols].mean())
        rdict.update({'visual_area': visual_area, 'datakey': datakey, 'cell': rid, 
                       'mean_score': mean_score, 'percentile': percentile, 'n_iterations': n_iterations})
        s = pd.Series(rdict)
        s_.append(s) 
    percentiles = pd.concat(s_, axis=1).T.reset_index(drop=True)

    return percentiles 

def group_iters_by_cell_and_transform(iterdf, metric='heldout_test_score'):
    # p_thr=0.05
    excl_cols = ['fit_time', 'iteration', 'n_cells',  'randi', 'score_time']
    incl_cols = [c for c in iterdf.columns if c not in excl_cols]

    s_=[]
    for (visual_area, datakey, rid, tr), d_ in iterdf.groupby(['visual_area', 'datakey', 'cell', 'train_transform']):

        mean_score = d_[d_['condition']=='data'][metric].mean()
        percentile = np.mean(mean_score < d_[d_['condition']=='shuffled'][metric])
        n_iterations = d_[d_['condition']=='data'].shape[0]
        rdict = dict(d_[d_['condition']=='data'][incl_cols].mean())
        rdict.update({'visual_area': visual_area, 'datakey': datakey, 'cell': rid, 'train_transform': tr,
                       'mean_score': mean_score, 'percentile': percentile, 'n_iterations': n_iterations})
        s = pd.Series(rdict)
        s_.append(s) 
    percentiles = pd.concat(s_, axis=1).T.reset_index(drop=True)

    return percentiles 


def group_iters_by_fov_and_transform(iterdf, metric='heldout_test_score'):
    # p_thr=0.05
    excl_cols = ['fit_time', 'iteration', 'n_cells',  'randi', 'score_time']
    incl_cols = [c for c in iterdf.columns if c not in excl_cols]

    s_=[]
    for (visual_area, datakey, sz), d_ in iterdf.groupby(['visual_area', 'datakey', 'train_transform']):

        mean_score = d_[d_['condition']=='data'][metric].mean()
        percentile = np.mean(mean_score < d_[d_['condition']=='shuffled'][metric])
        n_iterations = d_[d_['condition']=='data'].shape[0]
        rdict = dict(d_[d_['condition']=='data'][incl_cols].mean())
        rdict.update({'visual_area': visual_area, 'datakey': datakey, 'train_transform': sz,
                       'mean_score': mean_score, 'percentile': percentile, 'n_iterations': n_iterations})
        s = pd.Series(rdict)
        s_.append(s) 
    percentiles = pd.concat(s_, axis=1).T.reset_index(drop=True)

    return percentiles 

def group_iters_by_fov(iterdf, metric='heldout_test_score'):
    # p_thr=0.05
    excl_cols = ['fit_time', 'iteration', 'n_cells',  'randi', 'score_time']
    incl_cols = [c for c in iterdf.columns if c not in excl_cols]

    s_=[]
    for (visual_area, datakey), d_ in iterdf.groupby(['visual_area', 'datakey']):

        mean_score = d_[d_['condition']=='data'][metric].mean()
        percentile = np.mean(mean_score < d_[d_['condition']=='shuffled'][metric])
        n_iterations = d_[d_['condition']=='data'].shape[0]
        rdict = dict(d_[d_['condition']=='data'][incl_cols].mean())
        rdict.update({'visual_area': visual_area, 'datakey': datakey,
                       'mean_score': mean_score, 'percentile': percentile, 'n_iterations': n_iterations})
        s = pd.Series(rdict)
        s_.append(s) 
    percentiles = pd.concat(s_, axis=1).T.reset_index(drop=True)

    return percentiles 


def get_training_results(iterdf, test_type=None, train_classes=[0, 106], drop_arousal=True):
    if 'arousal' not in iterdf.columns:
        drop_arousal=False
    if 'train_transform' in iterdf.columns:
        print("transforms are split")
        if 'morph' in test_type:
            traindf = iterdf[(iterdf.train_transform==iterdf.test_transform) 
                        & (iterdf['morphlevel']==train_classes[0])]\
                        .drop(['morphlevel', 'p_chooseB'], axis=1).drop_duplicates()
#            traindf = iterdf[(iterdf.train_transform==iterdf.test_transform) 
#                        & (iterdf['morphlevel'].isin(train_classes))]\
#                        .drop(['morphlevel', 'p_chooseB'], axis=1).drop_duplicates()
# 
        else:
            traindf = iterdf[(iterdf.train_transform==iterdf.test_transform) 
                        & (iterdf.novel==False)].drop_duplicates()
    else:
        print("reg")
        traindf = iterdf.copy()

    if drop_arousal:
        traindf = traindf[(traindf.arousal=='all')].drop_duplicates()
        #traindf.groupby( 
    return traindf


# ======================================================================
# Load/Aggregate data functions 
# ======================================================================
# load_aggregate_rfs : now in rf_utils.py
# get_rf_positions: in rf_utils.py
def initializer(terminating_):
    # This places terminating in the global namespace of the worker subprocesses.
    # This allows the worker function to access `terminating` even though it is
    # not passed as an argument to the function.
    global terminating
    terminating = terminating_

def pool_bootstrap(neuraldf, sdf, n_iterations=50, n_processes=1, 
                   C_value=None, cv_nfolds=5, test_split=0.2, 
                   test_type=None, n_train_configs=4, verbose=False, within_fov=True,
                   class_a=0, class_b=106, do_shuffle=True, balance_configs=True):   
    '''
    This function replaces fit_svm_mp() -- includes opts for generalization test.
    Only tested for within-fov analyses (by_fov).

    test_type (str, None)
        None  : Classify A/B only 
                single=True to train/test on each size
        morph : Train on anchors, test on intermediate morphs
                single=True to train/test on each size
        size  : Train on specific size(s), test on un-trained sizes
                single=True to train/test on each size
    '''
    iter_df = None

    C=C_value
    vb = verbose 
    results = []
    terminating = mp.Event()

    pool = mp.Pool(initializer=initializer, initargs=(terminating, ), processes=n_processes)  
    try:
        ntrials, sample_size = neuraldf.shape
        print("[%s]... n: %i (%i procs)" % (test_type, int(sample_size-1), n_processes))
        if test_type=='morph':
            func = partial(train_test_morph, curr_data=neuraldf, sdf=sdf, 
                                verbose=verbose, C_value=C_value, test_split=test_split, cv_nfolds=cv_nfolds,
                                class_a=class_a, class_b=class_b, balance_configs=balance_configs) 
#                               MEANS=MEANS, sdf=sdf, sample_size=sample_size)
        elif test_type=='morph_single':
            func = partial(train_test_morph_single, curr_data=neuraldf, sdf=sdf, 
                                verbose=verbose, C_value=C_value, test_split=test_split, cv_nfolds=cv_nfolds,
                                class_a=class_a, class_b=class_b, balance_configs=balance_configs) 
#
#            if single: # train on 1 size, test on other sizes
#                func = partial(dutils.do_fit_train_single_test_morph, global_rois=global_rois, 
#                               MEANS=MEANS, sdf=sdf, sample_size=sample_size)
#            else: # combine data across sizes
#                func = partial(dutils.do_fit_train_test_morph, global_rois=global_rois, 
#                               MEANS=MEANS, sdf=sdf, sample_size=sample_size)             
        elif test_type=='size_single':
            func = partial(train_test_size_single, curr_data=neuraldf, sdf=sdf,
                                verbose=verbose, C_value=C_value, balance_configs=balance_configs) 
#                func = partial(dutils.do_fit_train_test_single, global_rois=global_rois, 
#                               MEANS=MEANS, sdf=sdf, sample_size=sample_size)
        elif test_type=='size_subset':
            func = partial(train_test_size_subset, curr_data=neuraldf, sdf=sdf,
                                verbose=verbose, C_value=C_value, n_train_configs=n_train_configs,
                                balance_configs=balance_configs) 
#                func = partial(dutils.cycle_train_sets, global_rois=global_rois, 
#                               MEANS=MEANS, sdf=sdf, sample_size=sample_size, 
#                                n_train_configs=n_train_configs)
        else:
            func = partial(do_fit_within_fov, curr_data=neuraldf, sdf=sdf, verbose=verbose,
                                C_value=C_value, balance_configs=balance_configs)
        output = pool.map_async(func, range(n_iterations)).get() #999999)
        #results = [pool.apply_async(do_fit_within_fov, args=(i, neuraldf, sdf, vb, C)) \
        #            for i in range(n_iterations)]
        #output= [p.get(99999999) for p in results]
        iter_df = pd.concat(output, axis=0)
 
    except KeyboardInterrupt:
        terminating.set()
        print("**interupt")
        pool.terminate()
        print("***Terminating!")
    except Exception as e:
        traceback.print_exc()
        pool.terminate() 
    finally:
        pool.close()
        pool.join()

    return iter_df #output #results
#

def iterate_by_ncells(n_cells, NEURALDATA, CELLS, sdf, n_iterations=100, n_processes=1, 
                        C_value=None, cv_nfolds=5, test_split=0.2, 
                        test_type=None, n_train_configs=4, verbose=False, within_fov=False,
                        class_name='morphlevel', class_a=0, class_b=106, do_shuffle=True, balance_configs=True,
                        feature_name='pupil_fraction', n_cuts=3, equalize_by='config', match_all_configs=True,
                        with_replacement=False):   
    #from pipeline.python.eyetracker import dlc_utils as dlcutils
    iterdf = None 

    # Select how to filter trial-matching
    train_labels = sdf[sdf[class_name].isin([class_a, class_b])][equalize_by].unique()
    common_labels = None if match_all_configs else train_labels

    #### Define MP worker
    results = []
    terminating = mp.Event() 
    def worker_by_ncells(out_q, n_iters, n_cells, NEURALDATA, CELLS, sdf, common_labels, test_type,
                    C_value=None, verbose=False, class_a=0, class_b=106, cv_nfolds=5, test_split=0.2):
        r_ = []        
        i_=[]
        for ni in n_iters:
            randi = random.randint(1, 10000)
            #### Get new sample set
            print("... sampling data, n=%i cells" % n_cells)
            neuraldf = sample_neuraldata(n_cells, CELLS, NEURALDATA, with_replacement=with_replacement,
                                                    train_configs=common_labels, randi=randi)
            neuraldf = aggr.zscore_neuraldf(neuraldf)
            n_cells = int(neuraldf.shape[1]-1) 
            #print("... doing decode BY_NCELLS | n=%i cells" % (n_cells))

            # Decoding -----------------------------------------------------
            # Fit.
            start_t = time.time()
            i_df = select_test(ni, neuraldf, sdf, 
                            C_value=C_value, class_a=class_a, class_b=class_b, 
                            cv_nfolds=cv_nfolds, test_split=test_split, 
                            verbose=verbose, do_shuffle=True, balance_configs=True,
                            test_type=test_type, n_train_configs=n_train_configs)  
            if i_df is None:
                out_q.put(None)
                raise ValueError("No results for current iter")
            end_t = time.time() - start_t
            print("--> Elapsed time: {0:.2f}sec".format(end_t))
            i_df['randi'] = randi
            i_.append(i_df)
        curr_iterdf = pd.concat(i_, axis=0)
        out_q.put(curr_iterdf) 
    try:        
        # Each process gets "chunksize' filenames and a queue to put his out-dict into:
        iter_list = np.arange(0, n_iterations) #gdf.groups.keys()
        out_q = mp.Queue()
        chunksize = int(math.ceil(len(iter_list) / float(n_processes)))
        procs = []
        for i in range(n_processes):
            p = mp.Process(target=worker_by_ncells,
                           args=(out_q, iter_list[chunksize * i:chunksize * (i + 1)],
                                    n_cells, NEURALDATA, CELLS, sdf, common_labels, test_type))
            procs.append(p)
            p.start() # start asynchronously
        # Collect all results into 1 results dict. We should know how many dicts to expect:
        results = []
        for i in range(n_processes):
            results.append(out_q.get(99999))
        # Wait for all worker processes to finish
        for p in procs:
            p.join() # will block until finished
    except KeyboardInterrupt:
        terminating.set()
        print("***Terminating!")
    except Exception as e:
        traceback.print_exc()
    finally:
        for p in procs:
            p.join()

    if len(results)>0:
        iterdf = pd.concat(results, axis=0)

    return iterdf


def select_test_return_clf(ni, neuraldf, sdf, C_value=None, cv_nfolds=5, test_split=0.2, 
                   test_type=None, n_train_configs=4, verbose=False, within_fov=True,
                   class_a=0, class_b=106, do_shuffle=True, balance_configs=True):   
    curr_iter = None
    try:
        if test_type=='size_subset':
            curr_iter = train_test_size_subset(ni, curr_data=neuraldf, sdf=sdf, 
                                C_value=C_value, class_a=class_a, class_b=class_b, 
                                cv_nfolds=cv_nfolds, test_split=test_split, 
                                verbose=verbose, do_shuffle=do_shuffle, 
                                balance_configs=balance_configs,
                                n_train_configs=n_train_configs)
        elif test_type=='size_single':
            curr_iter = train_test_size_single(ni, curr_data=neuraldf, sdf=sdf, 
                                C_value=C_value, class_a=class_a, class_b=class_b, 
                                cv_nfolds=cv_nfolds, test_split=test_split, 
                                verbose=verbose, do_shuffle=do_shuffle, 
                                balance_configs=balance_configs) 
        elif test_type=='morph_single':
            curr_iter = train_test_morph_single(ni, curr_data=neuraldf, sdf=sdf, 
                                    C_value=C_value, class_a=class_a, class_b=class_b, 
                                    cv_nfolds=cv_nfolds, test_split=test_split, 
                                    verbose=verbose, do_shuffle=do_shuffle, 
                                    balance_configs=balance_configs) 
        elif test_type=='morph':
            curr_iter = train_test_morph(ni, curr_data=neuraldf, sdf=sdf, 
                                    C_value=C_value, class_a=class_a, class_b=class_b, 
                                    cv_nfolds=cv_nfolds, test_split=test_split, 
                                    verbose=verbose, do_shuffle=do_shuffle, 
                                    balance_configs=balance_configs)  
        else: 
            curr_iter, curr_clf = do_fit_within_fov(ni, curr_data=neuraldf, sdf=sdf, 
                                C_value=C_value, class_a=class_a, class_b=class_b, 
                                cv_nfolds=cv_nfolds, test_split=test_split, 
                                verbose=verbose, do_shuffle=do_shuffle, 
                                balance_configs=balance_configs, return_clf=True)
        curr_iter['iteration'] = ni 
    except Exception as e:
        return None, None

    return curr_iter, curr_clf



def select_test(ni, neuraldf, sdf, C_value=None, cv_nfolds=5, test_split=0.2, 
                   test_type=None, n_train_configs=4, verbose=False, within_fov=True,
                   class_a=0, class_b=106, do_shuffle=True, balance_configs=True):   
    curr_iter = None
    try:
        if test_type=='size_subset':
            curr_iter = train_test_size_subset(ni, curr_data=neuraldf, sdf=sdf, 
                                            C_value=C_value, class_a=class_a, class_b=class_b, 
                                            cv_nfolds=cv_nfolds, test_split=test_split, 
                                            verbose=verbose, do_shuffle=do_shuffle, balance_configs=balance_configs,
                                            n_train_configs=n_train_configs)
        elif test_type=='size_single':
            curr_iter = train_test_size_single(ni, curr_data=neuraldf, sdf=sdf, 
                                            C_value=C_value, class_a=class_a, class_b=class_b, 
                                            cv_nfolds=cv_nfolds, test_split=test_split, 
                                            verbose=verbose, do_shuffle=do_shuffle, balance_configs=balance_configs) 
        elif test_type=='morph_single':
            curr_iter = train_test_morph_single(ni, curr_data=neuraldf, sdf=sdf, 
                                            C_value=C_value, class_a=class_a, class_b=class_b, 
                                            cv_nfolds=cv_nfolds, test_split=test_split, 
                                            verbose=verbose, do_shuffle=do_shuffle, balance_configs=balance_configs) 
        elif test_type=='morph':
            curr_iter = train_test_morph(ni, curr_data=neuraldf, sdf=sdf, 
                                            C_value=C_value, class_a=class_a, class_b=class_b, 
                                            cv_nfolds=cv_nfolds, test_split=test_split, 
                                            verbose=verbose, do_shuffle=do_shuffle, balance_configs=balance_configs)  
        else: 
            curr_iter = do_fit_within_fov(ni, curr_data=neuraldf, sdf=sdf, 
                                            C_value=C_value, class_a=class_a, class_b=class_b, 
                                            cv_nfolds=cv_nfolds, test_split=test_split, 
                                            verbose=verbose, do_shuffle=do_shuffle, balance_configs=balance_configs)
        curr_iter['iteration'] = ni 
    except Exception as e:
        return None

    return curr_iter


def split_and_match_arousal_trials(neuraldf, pupildf, sdf, feature_name='pupil_fraction', n_cuts=3, 
                                equalize_by='config', common_labels=None, verbose=False):
    '''
    Returns trial nums for matched neural and pupil data, split by arousal type
    common_labels: list of Configs to match between neural and pupil data
    '''
    from pipeline.python.eyetracker import dlc_utils as dlcutils

    low_ixs=None; high_ixs=None; 
    ndf, pdf = dlcutils.match_neural_and_pupil_trials(neuraldf, pupildf, equalize_conditions=False)
    p_low, p_high = balance_pupil_split(pdf, feature_name=feature_name, n_cuts=n_cuts, 
                                    match_cond=True, match_cond_name='size', equalize_after_split=True, 
                                    equalize_by='config', common_labels=common_labels, verbose=verbose,
                                    shuffle_labels=False)

    # Make sure training set conditions are balanced 
    train_sizes_low = p_low[p_low[equalize_by].isin(common_labels)].shape[0] \
                                if common_labels is not None else p_low.shape[0]
    train_sizes_high = p_high[p_high[equalize_by].isin(common_labels)].shape[0] \
                                if common_labels is not None else p_high.shape[0]
    assert train_sizes_low==train_sizes_high, \
            "Unequal pupil training trials: low=%i, high=%i" % (train_sizes_low, train_sizes_high) 
    low_ixs = p_low['trial'].unique()
    high_ixs = p_high['trial'].unique()

    return p_low, p_high

def shuffle_arousal_trials(p_low, p_high, equalize_by='config', common_labels=None):
    '''
    Returns trial indices for low and high arousal trials, shuffled labels
    '''
    low_shuffle_ixs=None
    high_shuffle_ixs=None 
    p_low_shuff, p_high_shuff = shuffle_pupil_labels(p_low, p_high) 
    train_sizes_low_shuffled = p_low_shuff[p_low_shuff[equalize_by].isin(common_labels)].shape[0] \
                                if common_labels is not None else p_low_shuff.shape[0]
    train_sizes_high_shuffled = p_high_shuff[p_high_shuff[equalize_by].isin(common_labels)].shape[0] \
                                if common_labels is not None else p_high_shuff.shape[0]
    assert train_sizes_low_shuffled==train_sizes_high_shuffled, \
            "[SHUFFLED] Unequal pupil training trials: low=%i, high=%i" \
            % (train_sizes_low_shuffled, train_sizes_high_shuffled) 

    low_shuffle_ixs = p_low_shuff['trial'].unique()
    high_shuffle_ixs = p_high_shuff['trial'].unique()

    return low_shuffle_ixs, high_shuffle_ixs

class WorkerStop(Exception):
    pass

def pupil_worker(out_q, n_iters, neuraldf, pupildf, sdf, equalize_by, common_labels, 
                test_type, cv_nfolds=5, test_split=0.2, C_value=None, verbose=True, return_clf=True,
                class_a=0, class_b=106, do_shuffle=True, feature_name='pupil_fraction', n_cuts=3,
                n_train_configs=4):

    curr_iterdf=None; inputdf=None; coefdf=None;
    curr_rois = np.array([c for c in neuraldf.columns if putils.isnumber(c)])
    print("ROIS:", curr_rois.shape)
    r_ = []        
    i_=[]
    c_=[]
    for ni in n_iters:
        print("... iter %i" % ni)
        #randi = random.randint(1, 10000)
        p_low, p_high = split_and_match_arousal_trials(neuraldf, pupildf, sdf, 
                                        feature_name=feature_name, n_cuts=n_cuts, 
                                        equalize_by=equalize_by, common_labels=common_labels, 
                                        verbose=verbose)
        # get shuffled
        low_shuff_ixs, high_shuff_ixs = shuffle_arousal_trials(p_low, p_high, 
                                            equalize_by=equalize_by,
                                            common_labels=common_labels)
        # trial indices of low/high pupil 
        low_trial_ixs = p_low['trial'].unique()
        high_trial_ixs = p_high['trial'].unique()
        #all_trial_ixs = pdf_matched['trial'].unique()

        # Decodinng -----------------------------------------------------
        start_t = time.time()
        arousal_conds = ['low', 'high', 'low_shuffle', 'high_shuffle']
        arousal_trial_ixs = [low_trial_ixs, high_trial_ixs, low_shuff_ixs, high_shuff_ixs]
        iter_list=[]
        trial_list=[]
        coef_list=[]
        for arousal_cond, curr_trial_ixs in zip(arousal_conds, arousal_trial_ixs):
            #print(arousal_cond)
            # Get neuraldf for current trials
            curr_data = neuraldf.loc[curr_trial_ixs].copy()
            print(curr_data.shape)
            print(curr_data.head())
            # Fit.
            start_t = time.time()
            if return_clf:
                cond_df, cond_clf = select_test_return_clf(ni, curr_data, sdf, 
                                C_value=C_value, class_a=class_a, class_b=class_b, 
                                cv_nfolds=cv_nfolds, test_split=test_split, 
                                verbose=verbose, do_shuffle=do_shuffle, balance_configs=True,
                                test_type=test_type, n_train_configs=n_train_configs)  
            else:
                cond_clf=None
                cond_df = select_test(ni, curr_data, sdf, 
                                C_value=C_value, class_a=class_a, class_b=class_b, 
                                cv_nfolds=cv_nfolds, test_split=test_split, 
                                verbose=verbose, do_shuffle=do_shuffle, balance_configs=True,
                                test_type=test_type, n_train_configs=n_train_configs)  
            if cond_df is None:
                out_q.put((None, None, None))
                raise WorkerStop("No results for current iter")
                #break
            end_t = time.time() - start_t
            print("--> Elapsed time: {0:.2f}sec".format(end_t))
            # clf results
            cond_df['n_trials'] = len(curr_trial_ixs)
            cond_df['arousal'] = 'high' if 'high' in arousal_cond else 'low'
            cond_df['true_labels'] = 'shuffle' not in arousal_cond
            iter_list.append(cond_df)

            # input trials
            trials_ = pd.DataFrame({'trial': curr_trial_ixs}) #, index=[ni])
            trials_['iteration'] = ni
            trials_['arousal'] = 'high' if 'high' in arousal_cond else 'low'
            trials_['true_labels'] = 'shuffle' not in arousal_cond 
            trial_list.append(trials_)

            if return_clf is True and cond_clf is not None:
                # weights
                coefs_= pd.DataFrame(cond_clf.coef_[0], columns=['coef'])
                #print(coefs_)
                coefs_['iteration'] = ni
                coefs_['arousal'] = 'high' if 'high' in arousal_cond else 'low'
                coefs_['true_labels'] = 'shuffle' not in arousal_cond 
                coefs_['cell'] = curr_rois
                coef_list.append(coefs_)

        curr_iter = pd.concat(iter_list, axis=0) 
        curr_trials = pd.concat(trial_list, axis=0)
        if return_clf:
            curr_coefs = pd.concat(coef_list, axis=0)
        r_.append(curr_iter)
        i_.append(curr_trials)
        c_.append(curr_coefs)
 
    curr_iterdf = pd.concat(r_, axis=0)
    inputdf = pd.concat(i_, axis=0)

    if return_clf:
        coefdf = pd.concat(c_, axis=0)

    out_q.put((curr_iterdf, inputdf, coefdf))
    
def iterate_split_pupil(neuraldf, pupildf, sdf, n_iterations=100, n_processes=1, 
                    C_value=None, cv_nfolds=5, test_split=0.2, 
                    test_type=None, n_train_configs=4, verbose=False, within_fov=True,
                    class_name='morphlevel', class_a=0, class_b=106, 
                    do_shuffle=True, balance_configs=True,
                    feature_name='pupil_fraction', n_cuts=3, equalize_by='config', 
                    match_all_configs=True, return_clf=True):   
    '''
    Run classifier analysis for all arousal conditions, resample and repeat for n_iterations.
    match_all_configs: False to only match trial #s for TRAIN conditions
    Returns:
        inptudf (input data selected for each iteration): TODO, just use random_state
        iterdf (results from bootstrapped decoding analysis)
    '''
    #from pipeline.python.eyetracker import dlc_utils as dlcutils
    iterdf = None 
    inputsdf = None
    coefsdf = None

    # Select how to filter trial-matching
    if 'config' not in sdf.columns:
        sdf['config'] = sdf.index.tolist()
    train_labels = sdf[sdf[class_name].isin([class_a, class_b])][equalize_by].unique()
    common_labels = None if match_all_configs else train_labels 

    #### Define MP worker
    results = []
    terminating = mp.Event() 
    try:        
        # Each process gets "chunksize' filenames and a queue to put his out-dict into:
        iter_list = np.arange(0, n_iterations) #gdf.groups.keys()
        out_q = mp.Queue()
        #data_q = mp.Queue()
        chunksize = int(math.ceil(len(iter_list) / float(n_processes)))
        procs = []
        for i in range(n_processes):
            p = mp.Process(target=pupil_worker,
                           args=(out_q, iter_list[chunksize * i:chunksize * (i + 1)],
                                neuraldf, pupildf, sdf, equalize_by, common_labels, test_type,
                                cv_nfolds, test_split, C_value, verbose, return_clf))
            procs.append(p)
            p.start() # start asynchronously
        # Collect all results into 1 results dict. We should know how many dicts to expect:
        results = []
        inputs = []
        coefs=[]
        for i in range(n_processes):
            res = out_q.get(99999)
            results.append(res[0])
            inputs.append(res[1])
            if return_clf:
                coefs.append(res[2])
        # Wait for all worker processes to finish
        for p in procs:
            p.join() # will block until finished
    except WorkerStop:
        print("No result (terminating)")
        terminating.set()
    except KeyboardInterrupt:
        terminating.set()
        print("***Terminating!")
    except Exception as e:
        traceback.print_exc()
        print("----- here.")
    finally:
        for p in procs:
            p.join()
            print '%s.exitcode = %s' % (p.name, p.exitcode)

    res_ = [i for i in results if i is not None]
    in_ = [i for i in inputs if i is not None]
    coef_ = [i for i in coefs if i is not None]
    if len(res_)>0:
        iterdf = pd.concat(res_, axis=0)
        inputsdf = pd.concat(in_, axis=0)
        if return_clf:
            coefsdf = pd.concat(coef_, axis=0)
    if return_clf:
        return iterdf, inputsdf, coefsdf
    else:
        return iterdf, inputsdf #results

def fit_svm_mp(neuraldf, sdf, n_iterations=50, n_processes=1, 
                   C_value=None, cv_nfolds=5, test_split=0.2, 
                   test_type=None, n_train_configs=4, verbose=False, within_fov=True,
                   class_a=0, class_b=106, do_shuffle=True, balance_configs=True):   
    iter_df = None
    #neuraldf = aggr.zscore_neuraldf(neuraldf)
    #### Define MP worker
    results = []
    terminating = mp.Event() 
    def worker(out_q, n_iters, neuraldf, sdf, test_type, C_value, verbose, 
                class_a, class_b, cv_nfold, test_split, do_shuffle, balance_configs, n_train_configs):
        i_ = []        
        for ni in n_iters:
            n_cells = int(neuraldf.shape[1]-1) 
            # Decoding -----------------------------------------------------
            # Fit.
            start_t = time.time()
            i_df = select_test(ni, neuraldf, sdf, test_type=test_type,
                                    C_value=C_value, class_a=class_a, class_b=class_b, 
                                    cv_nfolds=cv_nfolds, test_split=test_split, 
                                    verbose=verbose, do_shuffle=do_shuffle, balance_configs=balance_configs,
                                    n_train_configs=n_train_configs)  
            if i_df is None:
                out_q.put(None)
                raise ValueError("No results for current iter")
            end_t = time.time() - start_t
            print("--> Elapsed time: {0:.2f}sec".format(end_t))
            #i_df['n_trials'] = neuraldf.shape[0]
            i_.append(i_df)
        curr_iterdf = pd.concat(i_, axis=0)
        out_q.put(curr_iterdf)  
    try:        
        # Each process gets "chunksize' filenames and a queue to put his out-dict into:
        iter_list = np.arange(0, n_iterations) #gdf.groups.keys()
        out_q = mp.Queue()
        chunksize = int(math.ceil(len(iter_list) / float(n_processes)))
        procs = []
        for i in range(n_processes):
            p = mp.Process(target=worker, args=(out_q, iter_list[chunksize * i:chunksize * (i + 1)],
                                neuraldf, sdf, test_type, C_value, verbose, class_a, class_b,
                                cv_nfolds, test_split, do_shuffle, balance_configs, n_train_configs))
            procs.append(p)
            p.start() # start asynchronously
        # Collect all results into 1 results dict. We should know how many dicts to expect:
        results = []
        for i in range(n_processes):
            results.append(out_q.get(99999))
        # Wait for all worker processes to finish
        for p in procs:
            p.join() # will block until finished
    except KeyboardInterrupt:
        terminating.set()
        print("***Terminating!")
    except Exception as e:
        traceback.print_exc()
        terminating.set()
    finally:
        for p in procs:
            p.join()

    if len(results)>0:
        iterdf = pd.concat(results, axis=0)

    return iterdf


def fit_svm_mp0(neuraldf, sdf, n_iterations=50, n_processes=1, 
                   C_value=None, cv_nfolds=5, test_split=0.2, 
                   test_type=None, n_train_configs=4, verbose=False, within_fov=True,
                   class_a=0, class_b=106, do_shuffle=True, balance_configs=True):   
    iter_df = None

    #### Define MP worker
    results = []
    terminating = mp.Event() 
    def worker(n_iters, neuraldf, sdf, C_value, verbose, class_a, class_b, do_shuffle, balance_configs, out_q):
        r_ = []        
        for ni in n_iters:

            curr_iter = do_fit_within_fov(ni, curr_data=neuraldf, sdf=sdf, 
                                        C_value=C_value, class_a=class_a, class_b=class_b,  
                                        verbose=verbose, do_shuffle=do_shuffle, balance_configs=balance_configs)
            r_.append(curr_iter)
        curr_iterdf = pd.concat(r_, axis=0)
        out_q.put(curr_iterdf) 

    def worker_size_subset(n_iters, neuraldf, sdf, C_value, verbose, class_a, class_b, 
                                do_shuffle, balance_configs, n_train_configs, out_q):
        r_ = []        
        curr_iterdf=None
        for ni in n_iters: 
            curr_iter = train_test_size_subset(ni, curr_data=neuraldf, sdf=sdf, 
                                        C_value=C_value, class_a=class_a, class_b=class_b, 
                                        verbose=verbose, do_shuffle=do_shuffle, balance_configs=balance_configs,
                                        n_train_configs=n_train_configs)
            r_.append(curr_iter)
        if len(r_)>0:
            curr_iterdf = pd.concat(r_, axis=0)
        out_q.put(curr_iterdf)

    def worker_size_single(n_iters, neuraldf, sdf, C_value, verbose, class_a, class_b, 
                                do_shuffle, balance_configs, out_q):
        curr_iterdf=None
        r_ = []        
        for ni in n_iters: 
            curr_iter = train_test_size_single(ni, curr_data=neuraldf, sdf=sdf, 
                                        C_value=C_value, class_a=class_a, class_b=class_b, 
                                        verbose=verbose, do_shuffle=do_shuffle, balance_configs=balance_configs) 
            r_.append(curr_iter)
        if len(r_)>0:
            curr_iterdf = pd.concat(r_, axis=0)
        out_q.put(curr_iterdf)

    def worker_morph_single(n_iters, neuraldf, sdf, C_value, verbose, class_a, class_b, 
                                do_shuffle, balance_configs, out_q):
        curr_iterdf=None
        r_ = []        
        for ni in n_iters: 
            curr_iter = train_test_morph_single(ni, curr_data=neuraldf, sdf=sdf, 
                                        C_value=C_value, class_a=class_a, class_b=class_b, 
                                        verbose=verbose, do_shuffle=do_shuffle, balance_configs=balance_configs) 
            r_.append(curr_iter)
        if len(r_)>0:
            curr_iterdf = pd.concat(r_, axis=0)
        out_q.put(curr_iterdf)

    def worker_morph(n_iters, neuraldf, sdf, C_value, verbose, class_a, class_b, 
                                do_shuffle, balance_configs, out_q):
        curr_iterdf=None
        r_ = []        
        for ni in n_iters: 
            curr_iter = train_test_morph(ni, curr_data=neuraldf, sdf=sdf, 
                                        C_value=C_value, class_a=class_a, class_b=class_b, 
                                        verbose=verbose, do_shuffle=do_shuffle, balance_configs=balance_configs) 
            r_.append(curr_iter)
        if len(r_)>0:
            curr_iterdf = pd.concat(r_, axis=0)
        out_q.put(curr_iterdf)

    try:        
        # Each process gets "chunksize' filenames and a queue to put his out-dict into:
        iter_list = np.arange(0, n_iterations) #gdf.groups.keys()
        out_q = mp.Queue()
        chunksize = int(math.ceil(len(iter_list) / float(n_processes)))
        procs = []
        for i in range(n_processes):
            if test_type=='size_subset':
                p = mp.Process(target=worker_size_subset,
                           args=(iter_list[chunksize * i:chunksize * (i + 1)],
                                    neuraldf, sdf, C_value, verbose, class_a, class_b,
                                    do_shuffle, balance_configs, n_train_configs, out_q))
            elif test_type=='size_single':
                p = mp.Process(target=worker_size_single,
                           args=(iter_list[chunksize * i:chunksize * (i + 1)],
                                    neuraldf, sdf, C_value, verbose, class_a, class_b,
                                    do_shuffle, balance_configs, out_q))
 
            elif test_type=='morph_single':
                p = mp.Process(target=worker_morph_single,
                           args=(iter_list[chunksize * i:chunksize * (i + 1)],
                                    neuraldf, sdf, C_value, verbose, class_a, class_b,
                                    do_shuffle, balance_configs, out_q)) 
            elif test_type=='morph':
                p = mp.Process(target=worker_morph,
                           args=(iter_list[chunksize * i:chunksize * (i + 1)],
                                    neuraldf, sdf, C_value, verbose, class_a, class_b,
                                    do_shuffle, balance_configs, out_q)) 
 
            else: 
                p = mp.Process(target=worker,
                           args=(iter_list[chunksize * i:chunksize * (i + 1)],
                                    neuraldf, sdf, C_value, verbose, class_a, class_b,
                                    do_shuffle, balance_configs, out_q))
            print(p)
            procs.append(p)
            p.start() # start asynchronously

        # Collect all results into 1 results dict. We should know how many dicts to expect:
        results = []
        for i in range(n_processes):
            results.append(out_q.get(99999))
        # Wait for all worker processes to finish
        for p in procs:
            p.join() # will block until finished
    except KeyboardInterrupt:
        terminating.set()
        print("***Terminating!")
    except Exception as e:
        traceback.print_exc()
    finally:
        for p in procs:
            p.join()

    if len(results)>0:
        iter_df = pd.concat(results, axis=0)

    return iter_df #results


#def by_ncells_fit_svm_mp(ncells, celldf, NEURALDATA, sdf, 
#                           n_iterations=50, n_processes=1, 
#                           C_value=None, cv_nfolds=5, test_split=0.2, 
#                           test=None, single=False, n_train_configs=4, verbose=False,
#                           class_a=0, class_b=106):   
#    iter_df = None
#    #### Define multiprocessing worker
#    results = []
#    terminating = mp.Event()    
#    def worker(n_iters, ncells, celldf, sdf, NEURALDATA, C_value, class_a, class_b, out_q):
#        r_ = []        
#        for ni in n_iters:
#            curr_iter = do_fit_sample_cells(ni, sample_ize=sample_size, global_rois=celldf, sdf=sdf,
#                                        MEANS=NEURALDATA, do_shuffle=True, 
#                                        C_value=C_value, class_a=class_a, class_b=class_b)
#            r_.append(curr_iter)
#        curr_iterdf = pd.concat(r_, axis=0)
#        out_q.put(curr_iterdf)
#        
#    try:        
#        # Each process gets "chunksize' filenames and a queue to put his out-dict into:
#        iter_list = np.arange(0, n_iterations) #gdf.groups.keys()
#        out_q = mp.Queue()
#        chunksize = int(math.ceil(len(iter_list) / float(n_processes)))
#        procs = []
#        for i in range(n_processes):
#            p = mp.Process(target=worker,
#                           args=(iter_list[chunksize * i:chunksize * (i + 1)],
#                                 ncells, celldf, sdf, NEURALDATA, C_value,
#                                 class_a, class_b, out_q))
#            procs.append(p)
#            p.start()
#
#        # Collect all results into single results dict. We should know how many dicts to expect:
#        results = []
#        for i in range(n_processes):
#            results.append(out_q.get(99999))
#        # Wait for all worker processes to finish
#        for p in procs:
#            p.join()
#    except KeyboardInterrupt:
#        terminating.set()
#        print("***Terminating!")
#    except Exception as e:
#        traceback.print_exc()
#    finally:
#        for p in procs:
#            p.join()
#
#    if len(results)>0:
#        iter_df = pd.concat(results, axis=0)
# 
#    return iter_df #results

#
# ======================================================================
# Calculation functions 
# ======================================================================
def computeMI(x, y):
    sum_mi = 0.0
    x_value_list = np.unique(x)
    y_value_list = np.unique(y)
    Px = np.array([ len(x[x==xval])/float(len(x)) for xval in x_value_list ]) #P(x)
    Py = np.array([ len(y[y==yval])/float(len(y)) for yval in y_value_list ]) #P(y)
    for i in xrange(len(x_value_list)):
        if Px[i] ==0.:
            continue
        sy = y[x == x_value_list[i]]
        if len(sy)== 0:
            continue
        pxy = np.array([len(sy[sy==yval])/float(len(y))  for yval in y_value_list]) #p(x,y)
        t = pxy[Py>0.]/Py[Py>0.] /Px[i] # log(P(x,y)/( P(x)*P(y))
        sum_mi += sum(pxy[t>0]*np.log2( t[t>0]) ) # sum ( P(x,y)* log(P(x,y)/( P(x)*P(y)) )
    return sum_mi


def mean_confidence_interval(data, ci=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), spstats.sem(a)
    h = se * spstats.t.ppf((1 + ci) / 2., n-1)
    return m, h #m-h, m+h

def calculate_ci(scores, ci=95):
    med = np.median(scores)    
    # calculate 95% confidence intervals (100 - alpha)
    alpha = 100. - ci #5.0
    # retrieve observation at lower percentile
    lower_p = alpha / 2.0
    lower = max(0.0, np.percentile(scores, lower_p))

    # retrieve observation at upper percentile
    upper_p = (100 - alpha) + (alpha / 2.0)
    upper = min(1.0, np.percentile(scores, upper_p))
    #print(med, lower, upper)
    return med, lower, upper


# ======================================================================
# Split/pool functions 
# ======================================================================

# global_rois() now in aggr
# filter_rois() now in aggr
# get_pooled_cells() now in aggr

def get_ntrials_per_config(neuraldf, n_trials=10):
    t_list=[]
    for cfg, trialmat in neuraldf.groupby(['config']):
        trial_ixs = trialmat.index.tolist() # trial numbers
        np.random.shuffle(trial_ixs) # shuffle the trials randomly
        curr_trials = trialmat.loc[trial_ixs[0:n_trials]].copy()
        t_list.append(curr_trials)
    resampled_df = pd.concat(t_list, axis=0)

    return resampled_df

def get_trials_for_N_cells_df(curr_ncells, gdf, NEURALDATA, with_replacement=False, train_configs=None, randi=None): 
    # Get current global RIDs
    ncells_t = gdf.shape[0]                      
    roi_ids = np.array(gdf['roi'].values.copy()) 

    # Random sample w/ replacement
#    if with_replacement:
#        rand_ixs = np.array([random.randint(0, ncells_t-1) for _ in np.arange(0, curr_ncells)])
#        curr_roi_list = roi_ids[rand_ixs]
#        curr_roidf = gdf[gdf['roi'].isin(curr_roi_list)].copy()
#    else:
    curr_roidf = gdf.sample(n=curr_ncells, replace=with_replacement, random_state=randi)
    curr_roi_list = curr_roidf['roi'].values

    # Make sure equal num trials per condition for all dsets
    curr_dkeys = curr_roidf['datakey'].unique()
    currd = NEURALDATA[NEURALDATA['datakey'].isin(curr_dkeys)].copy()
    currd['cell'] = currd['cell'].astype(float)

    # Make sure equal num trials per condition for all dsets
    if train_configs is not None:
        min_ntrials_by_config = currd[currd['config'].isin(train_configs)][['datakey', 'config', 'trial']].drop_duplicates().groupby(['datakey'])['config'].value_counts().min()
        #print("MIn N trials in configs: %i" % min_ntrials_by_config)
    else: 
        min_ntrials_by_config = currd[['datakey', 'config', 'trial']].drop_duplicates().groupby(['datakey'])['config'].value_counts().min()
    print("Min samples per config: %i" % min_ntrials_by_config)

    d_list=[]
    for datakey, dkey_rois in curr_roidf.groupby(['datakey']):
        assert datakey in currd['datakey'].unique(), "ERROR: %s not found" % datakey
        # Get current trials, make equal to min_ntrials_by_config
        tmpd = pd.concat([trialmat.sample(n=min_ntrials_by_config, replace=False, random_state=None) 
                         for (rid, cfg), trialmat in currd[currd['datakey']==datakey].groupby(['cell', 'config'])], axis=0)
        tmpd['cell'] = tmpd['cell'].astype(float)

        # For each RID sample belonging to current dataset, get RID order
        #if with_replacement:
        sampled_cells = pd.concat([dkey_rois[dkey_rois['roi']==globalid][['roi', 'dset_roi']] 
                                   for globalid in curr_roi_list])
#        else:
#            sampled_cells = dkey_rois[dkey_rois['roi'].isin(curr_roidf['roi'])][['roi', 'dset_roi']]
#            curr_roi_list = curr_roidf['roi'].values
#
        sampled_dset_rois = sampled_cells['dset_roi'].values
        sampled_global_rois = sampled_cells['roi'].values
        cell_lut = dict((k, v) for k, v in zip(sampled_dset_rois, sampled_global_rois))

        # Get response + config, replace dset roi  name with global roi name
        slist = [tmpd[tmpd['cell']==rid][['config', 'response']].rename(columns={'response': cell_lut[rid]})
                 .sort_values(by='config').reset_index(drop=True) for rid in sampled_dset_rois]
        curr_roidata = pd.concat(slist, axis=1)
        curr_roidata = curr_roidata.loc[:,~curr_roidata.T.duplicated(keep='first')]
        d_list.append(curr_roidata)
    curr_neuraldf = pd.concat(d_list, axis=1)[curr_roi_list]

    cfg_df = pd.concat(d_list, axis=1)['config']

    assert cfg_df.shape[0]==curr_neuraldf.shape[0], "Bad trials"
    if len(cfg_df.shape) > 1:
        print("Requested configs: %s" % 'all' if train_configs is None else str(train_configs)) 
        cfg_df = cfg_df.loc[:,~cfg_df.T.duplicated(keep='first')]
        assert cfg_df.shape[1]==1, "Bad configs: %s" % str(curr_roidf['datakey'].unique()) #cfg_df

    df = pd.concat([curr_neuraldf, cfg_df], axis=1)
    
    return df

# ======================================================================
# Fitting functions 
# ======================================================================
def tune_C(sample_data, target_labels, scoring_metric='accuracy', 
                        cv_nfolds=3, test_split=0.2, verbose=False, n_processes=1):
    
    train_data = sample_data.copy()
    train_labels = target_labels
 
    #### DATA - Fit classifier
    scaler = StandardScaler()
    scaler.fit(train_data)
    train_data = scaler.transform(train_data)
    #test_data = scaler.transform(test_data)

    # Set the parameters by cross-validation
    tuned_parameters = [{'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}]

    results ={} 
    if verbose:
        print("# Tuning hyper-parameters for %s" % scoring_metric)
    #print()
    clf = GridSearchCV(svm.SVC(kernel='linear'), tuned_parameters, 
                            scoring=scoring_metric, cv=cv_nfolds, n_jobs=1) #n_processes)  
    clf.fit(train_data, train_labels)
    if verbose:
        print("Best parameters set found on development set:")
        print(clf.best_params_)
    if verbose:
        print("Grid scores on development set (scoring=%s):" % scoring_metric)
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))

#    y_true, y_pred = test_labels, clf.predict(test_data)
#    if verbose:
#        print("Detailed classification report:")
#        print("The model is trained on the full development set.")
#        print("The scores are computed on the full evaluation set.")
#        print(classification_report(y_true, y_pred))
#
#    test_score = clf.score(test_data, test_labels)
#    if verbose:
#        print("Held out test score: %.2f" % test_score)
#    results.update({'%s' % scorer: {'C': clf.best_params_['C'], 'test_score': test_score}})
#    
    return clf #results #clf.best_params_

#def fit_svm_shuffle(zdata, targets, test_split=0.2, cv_nfolds=5, verbose=False, C_value=None, randi=10):
#
#    cv = C_value is None
#
#    if verbose:
#        print("Labels=%s" % (str(targets['label'].unique())))
#
#    #### For each transformation, split trials into 80% and 20%
#    train_data, test_data, train_labels, test_labels = train_test_split(zdata, 
#                                                        targets['label'].values, 
#                                                        test_size=test_split, 
#                                                        stratify=targets['label'], #targets['group'],
#                                                        shuffle=True, random_state=randi)
#    #print("first few:", test_labels[0:10])
#    #### Cross validate (tune C w/ train data)
#    if cv:
#        cv_grid = tune_C(train_data, train_labels, scoring_metric='accuracy', 
#                            cv_nfolds=cv_nfolds, 
#                           test_split=test_split, verbose=verbose) #, n_processes=n_processes)
#
#        C_value = cv_grid.best_params_['C'] #cv_results['accuracy']['C']
#    else:
#        assert C_value is not None, "Provide value for hyperparam C..."
#    #trained_svc = cv_grid.best_estimator_
#
#    #### Fit SVM
#    scaler = StandardScaler().fit(train_data)
#    train_data = scaler.transform(train_data)
#
#    svc = svm.SVC(kernel='linear', C=C_value, random_state=randi) #, random_state=10)
#    #print("... cv")
#    scores = cross_validate(svc, train_data, train_labels, cv=cv_nfolds,
#                            scoring=('accuracy'),
#                            #scoring=('precision_macro', 'recall_macro', 'accuracy'),
#                            return_train_score=True)
#    iterdict = dict((s, values.mean()) for s, values in scores.items())
#    if verbose:
#        print('... train (C=%.2f): %.2f, test: %.2f' % (C_value, iterdict['train_score'], iterdict['test_score']))
#    trained_svc = svc.fit(train_data, train_labels)
#       
#    #### DATA - Test with held-out data
#    test_data = scaler.transform(test_data)
#    test_score = trained_svc.score(test_data, test_labels)
#
#    #### DATA - Calculate MI
#    predicted_labels = trained_svc.predict(test_data)
#    if verbose:    
#        print("Detailed classification report:")
#        print("The model is trained on the full development set.")
#        print("The scores are computed on the full evaluation set.")
#        print(classification_report(test_labels, predicted_labels))
#
#    #predicted_labels = trained_svc.predict(test_data)
#    mi = skmetrics.mutual_info_score(test_labels, predicted_labels)
#    ami = skmetrics.adjusted_mutual_info_score(test_labels, predicted_labels)
#    log2_mi = computeMI(test_labels, predicted_labels)
#    iterdict.update({'heldout_test_score': test_score, 
#                     'heldout_MI': mi, 'heldout_aMI': ami, 'heldout_log2MI': log2_mi,
#                     'C': C_value, 'randi': randi})
#
#    # ------------------------------------------------------------------
#    # Shuffle LABELS to calculate chance level
#    train_labels_chance = train_labels.copy()
#    np.random.shuffle(train_labels_chance)
#    test_labels_chance = test_labels.copy()
#    np.random.shuffle(test_labels_chance)
#
#    #### CHANCE - Fit classifier
#    chance_svc = svm.SVC(kernel='linear', C=C_value, random_state=randi)
#    scores_chance = cross_validate(chance_svc, train_data, train_labels_chance, cv=cv_nfolds,
#                            scoring=('accuracy'),
#                            #scoring=('precision_macro', 'recall_macro', 'accuracy'),
#                            return_train_score=True)
#    iterdict_chance = dict((s, values.mean()) for s, values in scores_chance.items())
#
#    # CHANCE - Test with held-out data
#    trained_svc_chance = chance_svc.fit(train_data, train_labels_chance)
#    test_score_chance = trained_svc_chance.score(test_data, test_labels_chance)  
#
#    # Chance - Calculate MI
#    predicted_labels = trained_svc_chance.predict(test_data)
#    mi = skmetrics.mutual_info_score(test_labels, predicted_labels)
#    ami = skmetrics.adjusted_mutual_info_score(test_labels, predicted_labels)
#    log2_mi = computeMI(test_labels, predicted_labels)
#
#    iterdict_chance.update({'heldout_test_score': test_score_chance, 
#                            'heldout_MI': mi, 'heldout_aMI': ami, 
#                            'heldout_log2MI': log2_mi, 'C': C_value, 'randi': randi})
#
#    return iterdict, iterdict_chance
#

def fit_svm(zdata, targets, test_split=0.2, cv_nfolds=5,  n_processes=1,
                C_value=None, verbose=False, return_clf=False, return_predictions=False,
                randi=10):

    cv = C_value is None

    #### For each transformation, split trials into 80% and 20%
    train_data, test_data, train_labels, test_labels = train_test_split(
                                                        zdata, targets['label'].values, 
                                                        test_size=test_split, 
                                                        stratify=targets['label'], #targets['group'], 
                                                        shuffle=True, random_state=randi)
    print("Unique train: %s (%i)" % (str(np.unique(train_labels)), len(train_labels)))
#    if (len(train_labels)/2.) < cv_nfolds:
#        cv_nfolds = 2 #int(len(train_labels)/2.)
#        print("Not enough train trials... trying w/ CV n=%i folds" % cv_nfolds)
#        return None
#
    #print("first few:", test_labels[0:10])
    #### Cross validate (tune C w/ train data)
    if cv:
        cv_grid = tune_C(train_data, train_labels, scoring_metric='accuracy', 
                        cv_nfolds=cv_nfolds, #cv_nfolds, 
                        test_split=test_split, verbose=verbose) #, n_processes=n_processes)

        C_value = cv_grid.best_params_['C'] #cv_results['accuracy']['C']
    else:
        assert C_value is not None, "Provide value for hyperparam C..."

    #trained_svc = cv_grid.best_estimator_

    #### Fit SVM
    scaler = StandardScaler().fit(train_data)
    train_data = scaler.transform(train_data)

    trained_svc = svm.SVC(kernel='linear', C=C_value, random_state=10)
    #print("... cv")
    scores = cross_validate(trained_svc, train_data, train_labels, cv=cv_nfolds,
                            scoring=('accuracy'),
                            #scoring=('precision_macro', 'recall_macro', 'accuracy'),
                            return_train_score=True)
    iterdict = dict((s, values.mean()) for s, values in scores.items())
    if verbose:
        print('... train (C=%.2f): %.2f, test: %.2f' % (C_value, iterdict['train_score'], iterdict['test_score']))
    trained_svc = svm.SVC(kernel='linear', C=C_value, random_state=randi).fit(train_data, train_labels)
       
    #### DATA - Test with held-out data
    test_data = scaler.transform(test_data)
    test_score = trained_svc.score(test_data, test_labels)

    #### DATA - Calculate MI
    predicted_labels = trained_svc.predict(test_data)
    if verbose:    
        print("Detailed classification report:")
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print(classification_report(test_labels, predicted_labels))

    #predicted_labels = trained_svc.predict(test_data)
    mi_dict = get_mutual_info_metrics(test_labels, predicted_labels)
    iterdict.update(mi_dict) 
    iterdict.update({'heldout_test_score': test_score, 'C': C_value, 'randi': randi})

    if return_clf:
        if return_predictions:
            return iterdict, trained_svc, scaler, (predicted_labels, test_labels)
        else:
            return iterdict, trained_svc, scaler
    else:
        return iterdict

def get_mutual_info_metrics(curr_test_labels, predicted_labels):
    mi = skmetrics.mutual_info_score(curr_test_labels, predicted_labels)
    ami = skmetrics.adjusted_mutual_info_score(curr_test_labels, predicted_labels)
    log2_mi = computeMI(curr_test_labels, predicted_labels)

    mi_dict = {'heldout_MI': mi, 'heldout_aMI': ami, 'heldout_log2MI': log2_mi}

    return mi_dict

# --------------------------------------------------------------------------------
# Wrappers for fitting functions - specifies what type of analysis to do
# --------------------------------------------------------------------------------
def do_fit_within_fov(iter_num, curr_data=None, sdf=None, verbose=False,
                    C_value=None, test_split=0.2, cv_nfolds=5, class_a=0, class_b=106,
                    do_shuffle=True, balance_configs=True, return_clf=False):

    #[gdf, MEANS, sdf, sample_size, cv] * n_times)
    '''
    Does SVC fit for cells within FOV (no global rois). Assumes 'config' column in curr_data.
    Does n_iterations, return mean/sem/std over iterations as dict of results.
    Classes (class_a, class_b) should be the labels of the target (i.e., value of morph level).
   
    do_shuffle (bool):  Runs fit_svm() twice, once reg and once with labels shuffled. 
    '''   
    i_list=[]
    #### Select train/test configs for clf A vs B
    train_configs = sdf[sdf['morphlevel'].isin([class_a, class_b])].index.tolist() 

    #### Get trial data for selected cells and config types
    curr_roi_list = [int(c) for c in curr_data.columns if c not in ['config', 'trial']]
    sample_data = curr_data[curr_data['config'].isin(train_configs)]
    if balance_configs:
        #### Make sure training data has equal nums of each config
        sample_data = aggr.equal_counts_df(sample_data)

    curr_rois = [r for r in sample_data.columns if putils.isnumber(r)]
    zdata = sample_data[curr_rois] #.drop('config', 1)

    #### Get labels
    targets = pd.DataFrame(sample_data['config'].copy(), columns=['config'])
    targets['label'] = [sdf['morphlevel'][cfg] for cfg in targets['config'].values]
    targets['group'] = [sdf['size'][cfg] for cfg in targets['config'].values]

    if verbose:
        print("Labels: %s\nGroups: %s" % (str(targets['label'].unique()), str(targets['group'].unique())))

    #### Fit
    cv = C_value is None
    randi = random.randint(1, 10000) 
    if return_clf:
        curr_iter, curr_clf, _ = fit_svm(zdata, targets, C_value=C_value, verbose=verbose,
                                    test_split=test_split, cv_nfolds=cv_nfolds, randi=randi,
                                    return_clf=True)
    else:
        curr_iter = fit_svm(zdata, targets, C_value=C_value, verbose=verbose,
                            test_split=test_split, cv_nfolds=cv_nfolds, randi=randi)
    tmpdf = pd.DataFrame(curr_iter, index=[iter_num])
    tmpdf['condition'] = 'data'
    i_list.append(tmpdf)

    #### Shuffle labels
    if do_shuffle:
        tmpdf_shuffled = fit_shuffled(zdata, targets, C_value=C_value, verbose=verbose,
                                test_split=test_split, cv_nfolds=cv_nfolds, randi=randi)
        tmpdf_shuffled.index = [iter_num]   
        i_list.append(tmpdf_shuffled)
 
    # Combine TRUE/SHUFF, add Meta info
    iter_df = pd.concat(i_list, axis=0) 
    iter_df['n_cells'] = zdata.shape[1]
    iter_df['n_trials'] = zdata.shape[0]
    for label, g in targets.groupby(['label']):
        iter_df['n_samples_%i' % label] = len(g['label'])
    iter_df['iteration'] = iter_num

    if return_clf:
        return iter_df, curr_clf
    else:
        return iter_df 

def sample_neuraldata(sample_size, global_rois, MEANS, with_replacement=False, train_configs=None, randi=None):
#    if isinstance(MEANS, dict):
#        if isinstance(MEANS[MEANS.keys()[0]], dict): # df_is_split
#            curr_data = get_trials_for_N_cells_split(sample_size, global_rois, 
#                                        MEANS, train_configs=train_configs)
#        else:
#            curr_data = get_trials_for_N_cells(sample_size, global_rois, 
#                                        MEANS, train_configs=train_configs)
#    else:
#   
    if isinstance(MEANS, dict):
        MEANS = aggr.neuraldf_dict_to_dataframe(MEANS)

    curr_data = get_trials_for_N_cells_df(sample_size, global_rois, MEANS, 
                        train_configs=train_configs, with_replacement=with_replacement, randi=randi)

    return curr_data

#def do_fit_within_fov(iter_num, curr_data=None, sdf=None, verbose=False,
#                    C_value=None, test_split=0.2, cv_nfolds=5, class_a=0, class_b=106,
#                    do_shuffle=True, balance_configs=True):
#

#def do_fit_sample_cells(iter_num, sample_size=1, global_rois=None, MEANS=None, sdf=None, 
#           C_value=None, test_split=0.2, cv_nfolds=5, class_a=0, class_b=106, 
#           do_shuffle=True, verbose=False, balance_configs=True, train_configs=None,
#            with_replacement=False):
#    #[gdf, MEANS, sdf, sample_size, cv] * n_times)
#    '''
#    Resample w/ replacement from pooled cells (across datasets). Assumes 'sdf' is same for all datasets.
#    Do n_iterations, return mean/sem/std over iterations as dict of results.
#    Classes (class_a, class_b) should be the actual labels of the target (i.e., value of morph level)
#
#    do_shuffle (bool):   Calls 'fit_svm' twice, once on true data, once on shuffled labels
#   
#    sample_size (int):  sample size 
#    '''   
#    i_list=[]
#    #### Get new sample set
#    try:
#        curr_data = sample_neuraldata(sample_size, global_rois, MEANS, train_configs=train_configs,
#                                    with_replacement=with_replacement)
#    except Exception as e:
#        traceback.print_exc()
#        return None
#
#    #### Select train/test configs for clf A vs B
#    train_configs = sdf[sdf['morphlevel'].isin([class_a, class_b])].index.tolist() 
#
#    #### Get trial data for selected cells and config types
#    curr_roi_list = [int(c) for c in curr_data.columns if c not in ['config', 'trial']]
#    sample_data = curr_data[curr_data['config'].isin(train_configs)]
#    if balance_configs:
#        #### Make sure equal counts per config
#        sample_data = aggr.equal_counts_df(sample_data)
#
#    zdata = sample_data.drop('config', 1) #sample_data[curr_roi_list].copy()
#    #zdata = (data - data.mean()) / data.std()
#
#    #### Get labels
#    targets = pd.DataFrame(sample_data['config'].copy(), columns=['config'])
#    targets['label'] = [sdf['morphlevel'][cfg] for cfg in targets['config'].values]
#    targets['group'] = [sdf['size'][cfg] for cfg in targets['config'].values]
#
#    #### Fit
#    randi = random.randint(1, 10000)
#    curr_iter = fit_svm(zdata, targets, C_value=C_value, test_split=test_split, 
#                            cv_nfolds=cv_nfolds, randi=randi)
#    curr_iter['condition'] = 'data'
#    tmp_df = pd.DataFrame(curr_iter, index=[iter_num]) 
#    i_list.append(tmp_df)
#
#    #### Shuffle labels
#    if do_shuffle:
#        tmpdf_shuffled = fit_shuffled(zdata, targets, C_value=C_value, verbose=verbose,
#                                test_split=test_split, cv_nfolds=cv_nfolds, randi=randi)
#        tmpdf_shuffled.index = [iter_num]   
#        i_list.append(tmpdf_shuffled)
# 
#    iter_df = pd.concat(i_list, axis=0) 
# 
#    iter_df['n_cells'] = zdata.shape[1]
#    iter_df['n_trials'] = zdata.shape[0]
#    iter_df['iteration'] = iter_num 
#    
#    return iter_df 
#

def fit_shuffled(zdata, targets, C_value=None, test_split=0.2, cv_nfolds=5, randi=0, verbose=False,
                    class_types=[0, 106], class_name='morph', do_pchoose=False, return_svc=False, i=0):
    '''
    Shuffle target labels, do fit_svm()
    '''
    labels_shuffled = targets['label'].copy().values 
    np.random.shuffle(labels_shuffled)
    targets['label'] = labels_shuffled

    tmp_=[]
    if do_pchoose:
        class_a, class_b = class_types
        iter_shuffled, trained_svc, trained_scaler, (predicted_labels, true_labels) = fit_svm(
                                                            zdata, targets, 
                                                            return_clf=True, return_predictions=True,
                                                            test_split=test_split, cv_nfolds=cv_nfolds, 
                                                            C_value=C_value, randi=randi) 
        # Calculate P(choose B)
        #pchoose_dict = get_pchoose(predicted_labels, true_labels, class_a=class_a, class_b=class_b)
        for anchor in class_types: #[class_a, class_b]:
            a_ixs = [i for i, v in enumerate(true_labels) if v==anchor] 
            p_chooseB = sum([1 if p==class_b else 0 for p in predicted_labels[a_ixs]])/float(len(a_ixs))
            iter_shuffled.update({'p_chooseB': p_chooseB, '%s' % class_name: anchor, 'n_samples': len(a_ixs)})
            tmpdf_shuffled = pd.DataFrame(iter_shuffled, index=[i])
            i+=1
            tmp_.append(tmpdf_shuffled)            
        iterdf = pd.concat(tmp_, axis=0).reset_index(drop=True)
    else:
        iterdict = fit_svm(zdata, targets, C_value=C_value, verbose=verbose,
                                test_split=test_split, cv_nfolds=cv_nfolds, randi=randi)
        iterdf = pd.DataFrame(iterdict, index=[i])

    iterdf['condition'] = 'shuffled'
    #print("shuffled")

    if return_svc:
        return iterdf, trained_svc, trained_scaler
    else:
        return iterdf

 
# ------
def train_test_size_single(iter_num, curr_data=None, sdf=None, verbose=False,
                    C_value=None, test_split=0.2, cv_nfolds=5, class_a=0, class_b=106,
                    do_shuffle=True, balance_configs=True):

    #[gdf, MEANS, sdf, sample_size, cv] * n_times)
    '''
    Does SVC fit for cells within FOV (no global rois). Assumes 'config' column in curr_data.
    Does n_iterations, return mean/sem/std over iterations as dict of results.
    Classes (class_a, class_b) should be the labels of the target (i.e., value of morph level).
   
    do_shuffle (bool):  Runs fit_svm() twice, once reg and once with labels shuffled. 
    '''   
    #### Select train/test configs for clf A vs B
    class_types = [class_a, class_b]
    restrict_transform = True
    class_name='morphlevel'
    constant_transform = 'size'
    sizes = sorted(sdf[constant_transform].unique()) 

    i_list=[]
    i=0
    # Go thru all training sizes, then test on non-trained sizes
    for train_transform in sizes:
        # Get train configs
        train_configs = sdf[((sdf[class_name].isin(class_types))\
                                & (sdf[constant_transform]==train_transform))].index.tolist()

        #### TRAIN SET: Get trial data for selected cells and config types
        curr_roi_list = [int(c) for c in curr_data.columns if c not in ['config', 'trial']]
        trainset = curr_data[curr_data['config'].isin(train_configs)].copy()
        if balance_configs:
            #### Make sure train set has equal counts per config
            trainset = aggr.equal_counts_df(trainset)

        train_data = trainset.drop('config', 1)#zdata = (data - data.mean()) / data.std()

        #### TRAIN SET: Get labels
        targets = pd.DataFrame(trainset['config'].copy(), columns=['config'])
        targets['label'] = [sdf['morphlevel'][cfg] for cfg in targets['config'].values]
        targets['group'] = [sdf['size'][cfg] for cfg in targets['config'].values]
        if verbose:
            print("Labels: %s\nGroups: %s" % (str(targets['label'].unique()), str(targets['group'].unique())))

        try:
            #### Train SVM
            randi = random.randint(1, 10000)
            iterdict, trained_svc, trained_scaler = fit_svm(train_data, targets, return_clf=True,
                                                    test_split=test_split, cv_nfolds=cv_nfolds, 
                                                    C_value=C_value, randi=randi)
            iterdict.update({'train_transform': train_transform, 'test_transform': train_transform,
                            'condition': 'data', 'n_trials': len(targets), 'novel': False})
            tmpdf = pd.DataFrame(iterdict, index=[i])
            for label, g in targets.groupby(['label']):
                tmpdf['n_samples_%i' % label] = len(g['label'])
             
            i_list.append(tmpdf) #_shuffled = pd.DataFrame(curr_iter_shuffled, index=[i])
            i+=1
            train_columns = tmpdf.columns.tolist()
        except Exception as e:
            print(e)
            return None

        #### Shuffle labels
        if do_shuffle:
            tmpdf_shuffled = fit_shuffled(train_data, targets, C_value=C_value, verbose=verbose,
                                    test_split=test_split, cv_nfolds=cv_nfolds, randi=randi, i=i)
            tmpdf_shuffled['train_transform'] = train_transform
            tmpdf_shuffled['test_transform'] = train_transform
            tmpdf_shuffled['n_trials'] = len(targets)
            tmpdf_shuffled['novel'] = False
            for label, g in targets.groupby(['label']):
                tmpdf_shuffled['n_samples_%i' % label] = len(g['label']) 
            i_list.append(tmpdf_shuffled) #iter_df) 
            i+=1

        #### Select generalization-test set
        test_configs = sdf[((sdf[class_name].isin(class_types))\
                                & (sdf[constant_transform]!=train_transform))].index.tolist()
        testset = curr_data[curr_data['config'].isin(test_configs)]
        test_data = testset.drop('config', 1) #zdata = (data - data.mean()) / data.std()

        test_targets = pd.DataFrame(testset['config'].copy(), columns=['config'])
        test_targets['label'] = [sdf['morphlevel'][cfg] for cfg in test_targets['config'].values]
        test_targets['group'] = [sdf['size'][cfg] for cfg in test_targets['config'].values]

           
        #### Test SVM
        fit_C_value = iterdict['C']
        for test_transform, curr_test_group in test_targets.groupby(['group']):
            iterdict = dict((k, None) for k in train_columns) 
            curr_test_labels = curr_test_group['label'].values
            curr_test_data = test_data.loc[curr_test_group.index].copy()
            curr_test_data = trained_scaler.transform(curr_test_data)
            curr_test_score = trained_svc.score(curr_test_data, curr_test_labels)

            #### Calculate additional metrics (MI)
            predicted_labels = trained_svc.predict(curr_test_data)
            mi_dict = get_mutual_info_metrics(curr_test_labels, predicted_labels)
            iterdict.update(mi_dict) 
            is_novel = train_transform!=test_transform
            iterdict.update({'heldout_test_score': curr_test_score, 'C': fit_C_value, 'randi': randi,
                             'train_transform': train_transform, 'test_transform': test_transform,
                             'n_trials': len(predicted_labels), 'novel': is_novel, 'condition': 'data'}) 
            testdf = pd.DataFrame(iterdict, index=[i])
            for label, g in test_targets.groupby(['label']):
                testdf['n_samples_%i' % label] = len(g['label']) 
            i += 1
            #### Shuffle labels  - no shuffle, already testd above
            i_list.append(testdf) 

    iterdf = pd.concat(i_list, axis=0).reset_index(drop=True)
    iterdf['iteration'] = [iter_num for _ in np.arange(0, len(iterdf))]
    iterdf['n_cells'] = curr_data.shape[1]-1
 
    return iterdf


def train_test_size_subset(iter_num, curr_data=None, sdf=None, verbose=False,
                    C_value=None, test_split=0.2, cv_nfolds=5, class_a=0, class_b=106,
                    do_shuffle=True, n_train_configs=4, balance_configs=True):

    #[gdf, MEANS, sdf, sample_size, cv] * n_times)
    '''
    Does SVC fit for cells within FOV (no global rois). Assumes 'config' column in curr_data.
    Does n_iterations, return mean/sem/std over iterations as dict of results.
    Classes (class_a, class_b) should be the labels of the target (i.e., value of morph level).
   
    do_shuffle (bool):  Runs fit_svm() twice, once reg and once with labels shuffled. 
    '''   
    #### Select train/test configs for clf A vs B
    class_types = [class_a, class_b]
    class_name='morphlevel'
    constant_transform = 'size'
    sizes = sorted(sdf[constant_transform].unique()) 

    #### Get all combinations of n_train_configs    
    combo_train_sizes = list(itertools.combinations(sizes, n_train_configs))

    # Go thru all training sizes, then test on non-trained sizes
    i_list=[]
    i=0
    for train_sizes in combo_train_sizes: #training_sets:

        # Get train configs
        train_configs = sdf[(sdf[class_name].isin(class_types))\
                                & (sdf[constant_transform].isin(train_sizes))].index.tolist()

        #### TRAIN SET: Get trial data for selected cells and config types
        curr_roi_list = [int(c) for c in curr_data.columns if c not in ['config', 'trial']]
        trainset = curr_data[curr_data['config'].isin(train_configs)].copy()
        if balance_configs:
            #### Make sure train set has equal counts per config
            trainset = aggr.equal_counts_df(trainset)

        train_data = trainset.drop('config', 1)#zdata = (data - data.mean()) / data.std()

        #### TRAIN SET: Get labels
        targets = pd.DataFrame(trainset['config'].copy(), columns=['config'])
        targets['label'] = [sdf['morphlevel'][cfg] for cfg in targets['config'].values]
        targets['group'] = [sdf['size'][cfg] for cfg in targets['config'].values]
        if verbose:
            print("Labels: %s\nGroups: %s" % (str(targets['label'].unique()), str(targets['group'].unique())))
        train_transform = '_'.join([str(int(s)) for s in train_sizes])

        #### Train SVM
        randi = random.randint(1, 10000)
        iterdict, trained_svc, trained_scaler = fit_svm(train_data, targets, return_clf=True,
                                                test_split=test_split, cv_nfolds=cv_nfolds, 
                                                C_value=C_value, randi=randi)
        iterdict.update({'train_transform': train_transform, 'test_transform': train_transform, 
                         'condition': 'data', 'n_trials': len(targets), 'novel': False})
        tmpdf = pd.DataFrame(iterdict, index=[i])
        for label, g in targets.groupby(['label']):
            tmpdf['n_samples_%i' % label] = len(g['label']) 
        i+=1
        i_list.append(tmpdf)
        train_columns = tmpdf.columns.tolist()
    
        #### Shuffle labels
        if do_shuffle:
            tmpdf_shuffled = fit_shuffled(train_data, targets, C_value=C_value, verbose=verbose,
                                    test_split=test_split, cv_nfolds=cv_nfolds, randi=randi, i=i)
            tmpdf_shuffled['train_transform'] = train_transform
            tmpdf_shuffled['test_transform'] = train_transform
            tmpdf_shuffled['n_trials'] = len(targets)
            tmpdf_shuffled['novel'] = False
            for label, g in targets.groupby(['label']):
                tmpdf_shuffled['n_samples_%i' % label] = len(g['label'])  
            # combine
            i_list.append(tmpdf_shuffled)
            i+=1

        #### Select generalization-test set
        test_sizes = [t for t in sizes if t not in train_sizes]
        test_configs = sdf[((sdf[class_name].isin(class_types))\
                                & (sdf[constant_transform].isin(test_sizes)))].index.tolist()
        test_subset = curr_data[curr_data['config'].isin(test_configs)]
        test_data = test_subset.drop('config', 1) #zdata = (data - data.mean()) / data.std()

        test_targets = pd.DataFrame(test_subset['config'].copy(), columns=['config'])
        test_targets['label'] = [sdf['morphlevel'][cfg] for cfg in test_targets['config'].values]
        test_targets['group'] = [sdf['size'][cfg] for cfg in test_targets['config'].values]
 
        #### Test SVM
        fit_C_value = iterdict['C']
        for test_transform, curr_test_group in test_targets.groupby(['group']):
            iterdict = dict((k, None) for k in train_columns) 
            curr_test_labels = curr_test_group['label'].values
            curr_test_data = test_data.loc[curr_test_group.index].copy()
            curr_test_data = trained_scaler.transform(curr_test_data)
            curr_test_score = trained_svc.score(curr_test_data, curr_test_labels)

            #### Calculate additional metrics (MI)
            predicted_labels = trained_svc.predict(curr_test_data)
            mi_dict = get_mutual_info_metrics(curr_test_labels, predicted_labels)
            iterdict.update(mi_dict) 
            is_novel = train_transform!=test_transform
            iterdict.update({'heldout_test_score': curr_test_score, 'C': fit_C_value, 'randi': randi,
                             'train_transform': train_transform, 'test_transform': test_transform, 
                             'novel': is_novel,
                             'condition': 'data', 'n_trials': len(predicted_labels)}) 
            testdf = pd.DataFrame(iterdict, index=[i])
            for label, g in test_targets.groupby(['label']):
                testdf['n_samples_%i' % label] = len(g['label']) 
            i += 1
            #### Shuffle labels  - no shuffle, already testd above
            i_list.append(testdf) 
    
    #print([i for i in i_list if not isinstance(i, pd.DataFrame)])
    iterdf = pd.concat(i_list, axis=0).reset_index(drop=True)
    iterdf['iteration'] = [iter_num for _ in np.arange(0, len(iterdf))]
    iterdf['n_cells'] = curr_data.shape[1]-1

    return iterdf


# -------------------------
#def do_fit_train_test_single(iter_num, sample_size=None, global_rois=None, MEANS=None, sdf=None,
#                            cv=True, C_value=None, test_size=0.2, cv_nfolds=5, class_a=0, class_b=106,
#                            balance_configs=True):
#    '''
#    Train/test PER SIZE.
#
#    Resample w/ replacement from pooled cells (across datasets). 
#    Assumes 'sdf' is same for all datasets.
#
#    Return fit results for 1 iteration.
#    Classes (class_a, class_b) should be labels of the target (i.e., value of morph level)
#    '''
#    #### Get new sample set
#    try:
#        curr_data = sample_neuraldata(sample_size, global_rois, MEANS)
#    except Exception as e:
#        traceback.print_exc()
#        return None
#
#    #### Select train/test configs for clf A vs B
#    class_types = [class_a, class_b]
#    restrict_transform = True
#    class_name='morphlevel'
#    constant_transform = 'size'
#    sizes = sorted(sdf[constant_transform].unique())
#    
#    i_list=[]
#    i=0
#    # Go thru all training sizes, then test on non-trained sizes
#    for train_transform in sizes:
#        # Get train configs
#        train_configs = sdf[((sdf[class_name].isin(class_types))\
#                                & (sdf[constant_transform]==train_transform))].index.tolist()
#
#        #### TRAIN SET: Get trial data for selected cells and config types
#        curr_roi_list = [int(c) for c in curr_data.columns if c not in ['config', 'trial']]
#        trainset = curr_data[curr_data['config'].isin(train_configs)].copy()
#        if balance_configs:
#            #### Make sure train set has equal counts per config
#            trainset = aggr.equal_counts_df(trainset)
#
#        train_data = trainset.drop('config', 1)#zdata = (data - data.mean()) / data.std()
#
#        #### TRAIN SET: Get labels
#        targets = pd.DataFrame(trainset['config'].copy(), columns=['config'])
#        targets['label'] = [sdf['morphlevel'][cfg] for cfg in targets['config'].values]
#        targets['group'] = [sdf['size'][cfg] for cfg in targets['config'].values]
#
#        # Select generalization-test set
#        # untrained_class_types = [c for c in stimdf[class_name].unique() if c not in class_types]
#        test_configs = sdf[((sdf[class_name].isin(class_types))\
#                                & (sdf[constant_transform]!=train_transform))].index.tolist()
#        testset = curr_data[curr_data['config'].isin(test_configs)]
#        test_data = testset.drop('config', 1) #zdata = (data - data.mean()) / data.std()
#
#        test_targets = pd.DataFrame(testset['config'].copy(), columns=['config'])
#        test_targets['label'] = [sdf['morphlevel'][cfg] for cfg in test_targets['config'].values]
#        test_targets['group'] = [sdf['size'][cfg] for cfg in test_targets['config'].values]
#
#        #### Train SVM
#        randi = random.randint(1, 10000)
#        iterdict, trained_svc, trained_scaler = fit_svm(train_data, targets, return_clf=True,
#                                                test_split=test_size, cv_nfolds=cv_nfolds, 
#                                                C_value=C_value, randi=randi)
#        iterdict.update({'train_transform': train_transform, 'test_transform': train_transform})
#        i_list.append(pd.DataFrame(iterdict, index=[i]))
#        i+=1
#        
#        #### Test SVM
#        for test_transform, curr_test_group in test_targets.groupby(['group']):
#            curr_test_labels = curr_test_group['label'].values
#            curr_test_data = test_data.loc[curr_test_group.index].copy()
#            curr_test_data = trained_scaler.transform(curr_test_data)
#            #test_labels = test_targets['label'].values
#            curr_test_score = trained_svc.score(curr_test_data, curr_test_labels)
#            #print(test_transform, curr_test_score)
#
#            #### Calculate additional metrics (MI)
#            predicted_labels = trained_svc.predict(curr_test_data)
#            mi = skmetrics.mutual_info_score(curr_test_labels, predicted_labels)
#            ami = skmetrics.adjusted_mutual_info_score(curr_test_labels, predicted_labels)
#            log2_mi = computeMI(curr_test_labels, predicted_labels)
#            
#            iterdict.update({'heldout_test_score': curr_test_score, 
#                             'heldout_MI': mi, 'heldout_aMI': ami, 'heldout_log2MI': log2_mi,
#                             'train_transform': train_transform,
#                             'test_transform': test_transform}) 
#            i_list.append(pd.DataFrame(iterdict, index=[i]))
#            i+=1 
#    iterdf = pd.concat(i_list, axis=0).reset_index(drop=True)
#    iterdf['iteration'] = [iter_num for _ in np.arange(0, len(iterdf))]
#    
#    return iterdf
#

def cycle_train_sets(iter_num, global_rois=None, MEANS=None, sdf=None, sample_size=None, n_train_configs=4,
                      cv=True, C_value=None, test_split=0.2, cv_nfolds=5, class_a=0, class_b=106):
    
    constant_transform = 'size'
    sizes = sorted(sdf[constant_transform].unique())
    
    training_sets = list(itertools.combinations(sizes, n_train_configs))

    i_list=[]
    for train_set in training_sets:
        test_set = [t for t in sizes if t not in train_set]
        tmpdf = do_fit_train_test_subset(iter_num, global_rois=global_rois, MEANS=MEANS, sdf=sdf, 
                                        sample_size=sample_size,
                                        train_sizes=train_set, test_sizes=test_set,
                                        cv=cv, C_value=C_value, test_split=test_split, 
                                        cv_nfolds=cv_nfolds, class_a=class_a, class_b=class_b)

        i_list.append(tmpdf)
    iterdf = pd.concat(i_list, axis=0).reset_index(drop=True)
    iterdf['iteration'] = [iter_num for _ in np.arange(0, len(iterdf))]
    
    return iterdf

#-----
def get_pchoose(predicted_labels, true_labels, class_a=0, class_b=106):
    '''
    For given predicted and true labels, calculate p_chooseB and p_correct. Return dict.
    '''
    iterdict={}
    for anchor in [class_a, class_b]:
        a_ixs = np.array([i for i, v in enumerate(true_labels) if v==anchor]) # trials where A shown
        if len(a_ixs)==0:
            p_chooseB=0
            p_correct=0
        else:
            p_chooseB = sum([1 if p==class_b else 0 \
                                for p in predicted_labels[a_ixs]])/float(len(a_ixs))
            p_correct = sum([1 if p==anchor else 0 \
                                for p in predicted_labels[a_ixs]])/float(len(a_ixs))
        iterdict.update({'p_chooseB': p_chooseB, 'morph_level': anchor, 'p_correct': p_correct})

    return iterdict

def train_test_morph(iter_num, curr_data=None, sdf=None, verbose=False,
                    C_value=None, test_split=0.2, cv_nfolds=5, class_a=0, class_b=106, midp=53,
                    do_shuffle=True, balance_configs=True, 
                    fit_psycho=True, P_model='weibull', par0=np.array([0.5, 0.5, 0.1]), nfits=20):

    #[gdf, MEANS, sdf, sample_size, cv] * n_times)
    '''
    Test generalization to morph stimuli (combine diff sizes)
    train_transform: 0_106 (class types)
    test_transform:  intermediate morphs

    Note: 
    If do_pchoose=True for fit_shuffled(), returns df updated with ['p_chooseB', 'morphlevel', n_samples']. 
    '''   
    #### Select train/test configs for clf A vs B
    class_types = [class_a, class_b]
    class_name='morphlevel'
    constant_transform = 'size'
    sizes = sorted(sdf[constant_transform].unique()) 

    #### Get train configs -- ANCHORS (A/B)
    train_configs = sdf[sdf[class_name].isin(class_types)].index.tolist()

    # Go thru all training sizes, then test on non-trained sizes
    i_list=[]
    i=0

    #### TRAIN SET: Get trial data for selected cells and config types
    curr_roi_list = [int(c) for c in curr_data.columns if c not in ['config', 'trial']]
    trainset = curr_data[curr_data['config'].isin(train_configs)].copy()
    if balance_configs:
        #### Make sure train set has equal counts per config
        trainset = aggr.equal_counts_df(trainset)

    train_data = trainset.drop('config', 1)#zdata = (data - data.mean()) / data.std()
    train_transform = '_'.join([str(c) for c in class_types]) #'anchor'

    #### TRAIN SET: Get labels
    targets = pd.DataFrame(trainset['config'].copy(), columns=['config'])
    targets['label'] = [sdf['morphlevel'][cfg] for cfg in targets['config'].values]
    targets['group'] = [sdf['size'][cfg] for cfg in targets['config'].values]
    if verbose:
        print("Labels: %s\nGroups: %s" % (str(targets['label'].unique()), str(targets['group'].unique())))


    #### Train SVM ----------------------------------------------------------------------
    randi = random.randint(1, 10000)
    iterdict, trained_svc, trained_scaler, (predicted_labels, true_labels) = fit_svm(
                                                        train_data, targets, 
                                                        return_clf=True, return_predictions=True,
                                                        test_split=test_split, cv_nfolds=cv_nfolds, 
                                                        C_value=C_value, randi=randi) 
    iterdict.update({'train_transform': train_transform, 'test_transform': train_transform, 
                     'condition': 'data', 'novel': False, 'n_trials': len(targets)})
    # Calculate P(choose B)
    #pchoose_dict = get_pchoose(predicted_labels, true_labels, class_a=class_a, class_b=class_b)
    train_samples={}
    for t in targets['label'].unique():
        curr_ncounts = len(np.where(targets['label'].values==t)[0])
        train_samples.update({'n_samples_%i' % t: curr_ncounts})

    for anchor in class_types:
        a_ixs = [i for i, v in enumerate(true_labels) if v==anchor] 
        p_chooseB = sum([1 if p==class_b else 0 for p in predicted_labels[a_ixs]])/float(len(a_ixs))
        iterdict.update({'p_chooseB': p_chooseB, 'morphlevel': anchor, 'n_samples': len(a_ixs)})
        tmpdf = pd.DataFrame(iterdict, index=[i])
        #for label, g in targets.groupby(['label']):
        for t, v in train_samples.items():
            tmpdf[t] = v
        i_list.append(tmpdf)
        i += 1
    train_columns = tmpdf.columns.tolist()
 
    #### Shuffle labels
    if do_shuffle:
        tmpdf_shuffled, shuffled_svc, shuffled_scaler = fit_shuffled(train_data, targets, C_value=C_value, 
                                                    verbose=verbose, test_split=test_split, 
                                                    cv_nfolds=cv_nfolds, randi=randi, do_pchoose=True, 
                                                    class_types=[class_a, class_b], class_name=class_name,
                                                    return_svc=True, i=i)
        tmpdf_shuffled['train_transform'] = train_transform
        tmpdf_shuffled['test_transform'] = train_transform
        tmpdf_shuffled['n_trials'] = len(targets)
        tmpdf_shuffled['novel'] = False
        for t, v in train_samples.items():
            tmpdf_shuffled[t] = v
        i_list.append(tmpdf_shuffled) 
        i+=1
    #### TEST SET --------------------------------------------------------------------
    novel_class_types = [c for c in sdf[class_name].unique() if c not in class_types]
    test_configs = sdf[sdf[class_name].isin(novel_class_types)].index.tolist()
    testset = curr_data[curr_data['config'].isin(test_configs)]
    test_data = testset.drop('config', 1) #zdata = (data - data.mean()) / data.std()

    # Get labels.
    test_targets = pd.DataFrame(testset['config'].copy(), columns=['config'])
    test_targets['label'] = [sdf['morphlevel'][cfg] for cfg in test_targets['config'].values]
    test_targets['group'] = [sdf['size'][cfg] for cfg in test_targets['config'].values]
       
    #### Test SVM
    fit_C_value = iterdict['C']
    if do_shuffle:
        fit_C_shuffled = float(tmpdf_shuffled['C'].unique())

    for test_transform, curr_test_group in test_targets.groupby(['label']): #(['group']):
        # print(test_transform, curr_test_group.shape)
        iterdict = dict((k, None) for k in train_columns) 
        if do_shuffle:
            shuffdict = dict((k, None) for k in train_columns)
        curr_test_labels = curr_test_group['label'].values
        curr_test_data = test_data.loc[curr_test_group.index].copy()
        curr_test_data = trained_scaler.transform(curr_test_data)
       
        # Calculate "score" 
        if test_transform in [-1, midp]:         # Ignore midp trials
            split_trials = np.array([int(i) for i, v in enumerate(curr_test_labels) ])
            # rando assign values
            split_labels = [class_a if i<0.5 else class_b for i in np.random.rand(len(split_trials),)]
        else:
            split_trials = np.array([int(i) for i, v in enumerate(curr_test_labels) if v!=midp])
            split_labels = [class_a if lvl < midp else class_b for lvl in curr_test_labels[split_trials]] 
        split_data = curr_test_data[split_trials, :]
        curr_test_score = trained_svc.score(split_data, split_labels) #(curr_test_data, curr_test_labels)
        iterdict.update({'heldout_test_score': curr_test_score, 'C': fit_C_value, 'randi': randi,
                         'train_transform': train_transform, 'test_transform': test_transform, 
                         'novel': True, 'n_trials': len(curr_test_labels), 'condition': 'data'}) 

        # predict p_chooseB
        predicted_labels = trained_svc.predict(curr_test_data)
        predicted_labels = np.array(predicted_labels)
        p_chooseB = sum([1 if p==class_b else 0 for p in predicted_labels])/float(len(predicted_labels)) 
        iterdict.update({'p_chooseB': p_chooseB, 'morphlevel': test_transform, 
                         'n_samples': len(curr_test_labels)}) 
        testdf = pd.DataFrame(iterdict, index=[i])
        for t, v in train_samples.items():
            testdf[t] = v
        i += 1
        i_list.append(testdf) 

        # shuffled:  Calculate score
        if do_shuffle:
            curr_score_shuff = shuffled_svc.score(split_data, split_labels) 
            shuffdict = iterdict.copy()
            # shuffled:  predict p_chooseB
            predicted_labels = shuffled_svc.predict(curr_test_data)
            predicted_labels = np.array(predicted_labels)
            p_chooseB = sum([1 if p==class_b else 0 for p in predicted_labels])/float(len(predicted_labels))
            shuffdict.update({'heldout_test_score': curr_score_shuff, 'p_chooseB': p_chooseB,
                              'condition': 'shuffled', 'C': fit_C_shuffled}) 
            testdf_shuff = pd.DataFrame(shuffdict, index=[i])
            for t, v in train_samples.items():
                testdf_shuff[t] = v
            i += 1
            i_list.append(testdf_shuff) 

    # Aggregate 
    iterdf = pd.concat(i_list, axis=0).reset_index(drop=True)
    iterdf['iteration'] = [iter_num for _ in np.arange(0, len(iterdf))]
    iterdf['n_cells'] = curr_data.shape[1]-1

    # fit curve?
    if fit_psycho:
        iterdf['threshold']=None
        iterdf['slope'] = None
        iterdf['lapse'] = None
        iterdf['likelihood'] = None
        for dcond, mdf in iterdf.groupby(['condition']):
            data = mdf[mdf.morphlevel!=-1].sort_values(by=['morphlevel'])\
                            [['morphlevel', 'n_samples', 'p_chooseB']].values.T
            max_v = max([class_a, class_b])
            data[0,:] /= float(max_v)
            try:
                par, L = mle_weibull(data, P_model=P_model, parstart=par0, nfits=nfits) 
                iterdf['threshold'].loc[mdf.index] = float(par[0])
                iterdf['slope'].loc[mdf.index] = float(par[1])
                iterdf['lapse'].loc[mdf.index] = float(par[2])
                iterdf['likelihood'].loc[mdf.index] = float(L)
            except Exception as e:
                traceback.print_exc()
                continue
           
    return iterdf


def train_test_morph_single(iter_num, curr_data=None, sdf=None, verbose=False,
                    C_value=None, test_split=0.2, cv_nfolds=5, class_a=0, class_b=106, midp=53,
                    do_shuffle=True, balance_configs=True,
                    fit_psycho=True, P_model='weibull', par0=np.array([0.5, 0.5, 0.1]), nfits=20):

    '''
    Resample w/ replacement from pooled cells (across datasets). Assumes 'sdf' is same for all datasets.
    Return fit results for 1 iteration.
    Classes (class_a, class_b) should be the actual labels of the target (i.e., value of morph level)
    '''
    #### Select train/test configs for clf A vs B
    class_types = [class_a, class_b]
    restrict_transform = True
    class_name='morphlevel'
    constant_transform = 'size'
    sizes = sorted(sdf[constant_transform].unique())
    
    i_list=[]
    i=0
    # Go thru all training sizes, then test on non-trained sizes
    for train_transform in sizes:

        # Get train configs -- ANCHORS (A/B)
        train_configs = sdf[(sdf[class_name].isin(class_types))
                           & (sdf[constant_transform]==train_transform)].index.tolist()

        #### TRAIN SET --------------------------------------------------------------------
        # Get trial data for selected cells and config types
        curr_roi_list = [int(c) for c in curr_data.columns if c not in ['config', 'trial']]
        trainset = curr_data[curr_data['config'].isin(train_configs)].copy()
        if balance_configs:
            #### Make sure train set has equal counts per config
            trainset = aggr.equal_counts_df(trainset)

        train_data = trainset.drop('config', 1)#zdata = (data - data.mean()) / data.std()

        # Get labels
        targets = pd.DataFrame(trainset['config'].copy(), columns=['config'])
        targets['label'] = [sdf['morphlevel'][cfg] for cfg in targets['config'].values]
        targets['group'] = [sdf['size'][cfg] for cfg in targets['config'].values]

        #### Train SVM ----------------------------------------------------------------------
        randi = random.randint(1, 10000)
        iterdict, trained_svc, trained_scaler, (predicted_labels, true_labels) = fit_svm(
                                                            train_data, targets, 
                                                            return_clf=True, return_predictions=True,
                                                            test_split=test_split, cv_nfolds=cv_nfolds, 
                                                            C_value=C_value, randi=randi) 
        iterdict.update({'train_transform': train_transform, 'test_transform': train_transform, 
                         'condition': 'data', 'n_trials': len(true_labels), 'novel': False})

        train_samples={}
        for t in targets['label'].unique():
            curr_ncounts = len(np.where(targets['label'].values==t)[0])
            train_samples.update({'n_samples_%i' % t: curr_ncounts})

        for anchor in class_types:
            a_ixs = [i for i, v in enumerate(true_labels) if v==anchor] 
            p_chooseB = sum([1 if p==class_b else 0 for p in predicted_labels[a_ixs]])/float(len(a_ixs))
            iterdict.update({'p_chooseB': p_chooseB, 
                            '%s' % class_name: anchor, #'%s' % constant_transform: train_transform, 
                            'n_samples': len(a_ixs)})
            tmpdf = pd.DataFrame(iterdict, index=[i])
            for t, v in train_samples.items():
                tmpdf[t] = v
            i += 1
            i_list.append(tmpdf)
        train_columns = tmpdf.columns.tolist()
     
        #### Shuffle labels
        if do_shuffle:
            tmpdf_shuffled, shuffled_svc, shuffled_scaler = fit_shuffled(train_data, targets, C_value=C_value, 
                                                    verbose=verbose, test_split=test_split, 
                                                    cv_nfolds=cv_nfolds, randi=randi, do_pchoose=True, 
                                                    class_types=[class_a, class_b], class_name=class_name,
                                                    return_svc=True, i=i) 
            tmpdf_shuffled['train_transform'] = train_transform
            tmpdf_shuffled['test_transform'] = train_transform
            tmpdf_shuffled['n_trials'] = len(targets) 
            tmpdf_shuffled['novel'] = False
            for t, v in train_samples.items():
                tmpdf_shuffled[t] = v
            i_list.append(tmpdf_shuffled) 
            i+=1 

        #### TEST SET --------------------------------------------------------------------
        # Get data, specify configs
        novel_class_types = [c for c in sdf[class_name].unique() if c not in class_types]
        test_configs = sdf[(sdf[class_name].isin(novel_class_types))
                          & (sdf[constant_transform]==train_transform)].index.tolist()
        
        testset = curr_data[curr_data['config'].isin(test_configs)]
        test_data = testset.drop('config', 1) #zdata = (data - data.mean()) / data.std()

        # Get labels.
        test_targets = pd.DataFrame(testset['config'].copy(), columns=['config'])
        test_targets['label'] = [sdf['morphlevel'][cfg] for cfg in test_targets['config'].values]
        test_targets['group'] = [sdf['size'][cfg] for cfg in test_targets['config'].values]

        fit_C_value = iterdict['C']
        if do_shuffle:
            fit_C_shuffled = float(tmpdf_shuffled['C'].unique())
        #### Test SVM
        for curr_morph_test, curr_test_group in test_targets.groupby(['label']):
            iterdict = dict((k, None) for k in train_columns) 
            if do_shuffle:
                shuffdict = dict((k, None) for k in train_columns)
            curr_test_labels = curr_test_group['label'].values
            curr_test_data = test_data.loc[curr_test_group.index].copy()
            curr_test_data = trained_scaler.transform(curr_test_data)

            # Calculate scores
            if curr_morph_test in [-1, midp]:        # Ignore midp trials
                split_trials = np.array([int(i) for i, v in enumerate(curr_test_labels) ])
                # rando assign values
                split_labels = [class_a if i<0.5 else class_b for i in np.random.rand(len(split_trials),)]
            else:
                split_trials = np.array([int(i) for i, v in enumerate(curr_test_labels) if v!=midp])
                split_labels = [class_a if lvl < midp else class_b for lvl in curr_test_labels[split_trials]] 
            split_data = curr_test_data[split_trials, :]
            curr_test_score = trained_svc.score(split_data, split_labels) #(curr_test_data, curr_test_labels)
            iterdict.update({'heldout_test_score': curr_test_score, 'C': fit_C_value, 'randi': randi,
                             'train_transform': train_transform, 'test_transform': train_transform, 
                             'novel': True, 'n_trials': len(curr_test_labels), 'condition': 'data'}) 

            #### Calculate p choose B on trials where morph X shown (test_transform)
            predicted_labels = trained_svc.predict(curr_test_data)
            p_chooseB = sum([1 if p==class_b else 0 \
                                for p in predicted_labels])/float(len(predicted_labels))
            iterdict.update({'p_chooseB': p_chooseB, '%s' % class_name: curr_morph_test, #test_transform,
                            'n_samples': len(predicted_labels)})
            testdf = pd.DataFrame(iterdict, index=[i])
            # add N training samples            
            for t, v in train_samples.items():
                testdf[t] = v
            i_list.append(testdf) #pd.DataFrame(iterdict, index=[i]))
            i+=1 

            # shuffled:  Calculate score
            if do_shuffle:
                curr_score_shuff = shuffled_svc.score(split_data, split_labels) 
                shuffdict = iterdict.copy()
                # shuffled:  predict p_chooseB
                predicted_labels = shuffled_svc.predict(curr_test_data)
                predicted_labels = np.array(predicted_labels)
                p_chooseB = sum([1 if p==class_b else 0 for p in predicted_labels])/float(len(predicted_labels))
                shuffdict.update({'heldout_test_score': curr_score_shuff, 'p_chooseB': p_chooseB,
                                  'condition': 'shuffled', 'C': fit_C_shuffled}) 
                testdf_shuff = pd.DataFrame(shuffdict, index=[i])
                for t, v in train_samples.items():
                    testdf_shuff[t] = v
                i_list.append(testdf_shuff) 
                i+=1

    # Aggregate            
    iterdf = pd.concat(i_list, axis=0).reset_index(drop=True)
    iterdf['iteration'] = [iter_num for _ in np.arange(0, len(iterdf))]
    iterdf['n_cells'] = curr_data.shape[1]-1
   
     # fit curve?
    if fit_psycho:
        iterdf['threshold']=None
        iterdf['slope'] = None
        iterdf['lapse'] = None
        iterdf['likelihood'] = None
        for (dcond, train_transform), mdf in iterdf.groupby(['condition', 'train_transform']):
            data = mdf[mdf.morphlevel!=-1].sort_values(by=['morphlevel'])\
                            [['morphlevel', 'n_samples', 'p_chooseB']].values.T
            max_v = max([class_a, class_b])
            data[0,:] /= float(max_v)
            try:
                par, L = mle_weibull(data, P_model=P_model, parstart=par0, nfits=nfits) 
                iterdf['threshold'].loc[mdf.index] = float(par[0])
                iterdf['slope'].loc[mdf.index] = float(par[1])
                iterdf['lapse'].loc[mdf.index] = float(par[2])
                iterdf['likelihood'].loc[mdf.index] = float(L)
            except Exception as e:
                traceback.print_exc()
                continue
           
    return iterdf

# ------


# ======================================================================
# Performance plotting 
# ======================================================================
def plot_individual_shuffle_distn(dk, va, traindf, metric='heldout_test_score', axn=None):

    d_ = traindf[(traindf.visual_area==va) & (traindf.datakey==dk) ].copy()


    if 'train_transform' in d_.columns and len(d_['train_transform'].unique())>1:
        if axn is None:
            fig, axn = pl.subplots(1, 5, figsize=(10,4), sharex=True, sharey=True)

        for ai, (transf, g) in enumerate(d_.groupby(['train_transform'])):
            ax = axn[ai]
            mean_score = g[g['condition']=='data'][metric].mean()
            percentile = np.mean(mean_score < g[g['condition']=='shuffled'][metric])
            ax.set_title('%s: \navg.%.2f (p=%.2f)' % (str(transf), mean_score, percentile), loc='left', fontsize=8)
            sns.distplot(g[g['condition']=='data'][metric], color='m', ax=ax)
            sns.distplot(g[g['condition']=='shuffled'][metric], color='k', ax=ax)
            ax.set_xlabel('')
        fig.text(0.5, 0.05, metric, ha='center', fontsize=12)
        fig.text(0.05, 0.95, '%s (%s)' % (dk, va), fontsize=16)
        return fig
    else:       
        is_fig=False
        if axn is None:
            fig, axn = pl.subplots(1, 1, figsize=(5,4), sharex=True, sharey=True)
            is_fig=True
        if isinstance(axn, list):
            ax=axn[0]
        else:
            ax=axn
        mean_score = d_[d_['condition']=='data'][metric].mean()
        percentile = np.mean(mean_score < d_[d_['condition']=='shuffled'][metric])
        ax.set_title('avg.%.2f (p=%.2f)' % ( mean_score, percentile), loc='left', fontsize=8)
        sns.distplot(d_[d_['condition']=='data'][metric], color='m', ax=ax)
        sns.distplot(d_[d_['condition']=='shuffled'][metric], color='k', ax=ax)
        ax.set_xlabel('')
        if is_fig:
            fig.text(0.5, 0.05, metric, ha='center', fontsize=12)
            fig.text(0.05, 0.95, '%s (%s)' % (dk, va), fontsize=16)
            pl.subplots_adjust(bottom=0.2, left=0.1, right=0.95, wspace=0.5, top=0.8)
        else:
            ax.set_title(dk, loc='left')
        return ax
    

def plot_score_by_ncells(pooled, metric='heldout_test_score', area_colors=None,
        lw=2, ls='-', capsize=3, ax=None, dpi=150):

    if area_colors is None:
        visual_area, area_colors = putils.set_threecolor_palette()
        dpi = putils.set_plot_params()
       
    if ax is None:
        fig, ax = pl.subplots(figsize=(5,4), sharex=True, sharey=True, dpi=dpi)

    for ai, (visual_area, g) in enumerate(pooled.groupby(['visual_area'])):
        mean_scores = g.sort_values(by='n_units')[metric]
        std_scores = g.sort_values(by='n_units')['%s_sem' % metric]
        n_units_per = g.groupby(['n_units'])[metric].mean().index.tolist()
        ax.plot(n_units_per, mean_scores, color=area_colors[visual_area], 
                alpha=1, lw=lw,
                label='%s' % (visual_area))
        ax.errorbar(n_units_per, mean_scores, yerr=std_scores, color=area_colors[visual_area], 
                    capthick=lw, capsize=capsize, label=None, alpha=1, lw=lw, linestyle=ls)
    ax.legend(bbox_to_anchor=(1., 1))
    ax.set_xlabel("N units")
    ax.set_ylabel(metric)

    return ax

def default_classifier_by_ncells(pooled, plot_str='traintestAB', dst_dir='/tmp', 
                                data_id='DATAID', area_colors=None, date_str='YYYYMMDD', 
                                dpi=150, lw=2, capsize=2, metric='heldout_test_score', xlim=100):
    # Plot
    for zoom in [True, False]:
        fig, ax = pl.subplots(figsize=(5,4), sharex=True, sharey=True, dpi=dpi)
        ax = plot_score_by_ncells(pooled, metric=metric, area_colors=area_colors, 
                                lw=lw, capsize=capsize, ax=ax)
        ax.set_title(plot_str) #overlap_thr)
        if metric=='heldout_test_score':
            ax.set_ylim([0.4, 1.0])
        ax.set_ylabel(metric)

        zoom_str=''
        if zoom:
            ax.set_xlim([0, xlim])
            zoom_str = 'zoom'

        sns.despine(trim=True, offset=4)
        pl.subplots_adjust(right=0.75, left=0.2, wspace=0.5, bottom=0.2, top=0.8)

        putils.label_figure(fig, data_id)

        figname = '%s_decode_%s%s' % (plot_str, metric, zoom_str)
        pl.savefig(os.path.join(dst_dir, '%s_%s.svg' % (figname, date_str)))
        print(dst_dir, figname)
    return


def plot_morph_curves(results, sdf, col_name='test_transform', plot_ci=False, ci=95, 
                        plot_luminance=True, lw=2, capsize=2, markersize=5,
                        curr_color='k', ax=None, dpi=150, alpha=1, label=None):
    
    if ax is None:
        fig, ax = pl.subplots(dpi=dpi, figsize=(5,4))

    morphlevels = sorted([s for s in sdf['morphlevel'].unique() if s!=-1])
    xvs = np.arange(1, len(morphlevels)+1) #if plot_luminance else np.arange(0, len(morphlevels))
    
    for visual_area, df_ in results.groupby(['visual_area']):
        # Set color
        if plot_luminance:
            # plot luminance control
            control_val=-1
            if plot_ci:
                ctl, ctl_lo, ctl_hi = calculate_ci(df_[df_[col_name]==control_val]['p_chooseB'].values, ci=ci)
                yerr = [abs(np.array([ctl-ctl_lo])), abs(np.array([ctl_hi-ctl]))]
            else:
                ctl = df_[df_[col_name]==control_val]['p_chooseB'].mean()
                yerr = df_[df_[col_name]==control_val]['p_chooseB'].sem()

            ax.errorbar(0, ctl, yerr=yerr, color=curr_color,
                           marker='o', markersize=markersize, capsize=capsize, alpha=alpha)
            
        # plot morph curves
        if plot_ci:
            ci_vals = dict((val, calculate_ci(g['p_chooseB'].values, ci=ci)) \
                             for val, g in df_[df_[col_name].isin(morphlevels)].groupby([col_name]))
            mean_vals = np.array([ci_vals[k][0] for k in morphlevels])
            lowers = np.array([ci_vals[k][1] for k in morphlevels])
            uppers =  np.array([ci_vals[k][2] for k in morphlevels])
            yerr = [np.array([mean_vals - lowers]), np.array([mean_vals-uppers])]
        else:
            mean_vals = df_[df_[col_name].isin(morphlevels)].groupby([col_name]).mean()['p_chooseB']
            yerr = df_[df_[col_name].isin(morphlevels)].groupby([col_name]).sem()['p_chooseB']

        ax.plot(xvs, mean_vals, color=curr_color, lw=lw, alpha=alpha, label=label)
        ax.errorbar(xvs, mean_vals, yerr=yerr, color=curr_color,
                          capsize=capsize, alpha=alpha, label=None)
        ax.set_ylim([0, 1])

    xticks = np.arange(0, len(morphlevels)+1) if plot_luminance else xvs
    xlabels = sdf['morphlevel'].unique() if plot_luminance \
                    else sdf[sdf['morphlevel']!=-1]['morphlevel'].unique()
    ax.set_xticks(xticks)
    ax.set_xticklabels( [int(m) for m in sorted(xlabels)] )
    ax.set_ylabel('p(choose B)')
    ax.set_xlabel('Morph level')
    
    return ax


def default_morphcurves_split_size(results, sdf, area_colors=None, dst_dir='/tmp', data_id='DATAID',
                                    lw=2, capsize=2, plot_legend=False, hue_size=False,
                                  train_str='train-anchors-test-intermed'):
    #if area_colors is None:
    visual_areas, area_colors = putils.set_threecolor_palette()
    cpalettes = {'V1': 'cubehelix', 'Lm': 'colorblind', 'Li': 'hsv'}
    
    fig, axn = pl.subplots(1, 3, figsize=(12,4), sharex=True, sharey=True, dpi=150)
    alphas = np.linspace(0.1, 1, 5)
    ci = 95
    shade=False
    plot_ci=False
    plot_luminance= True
    use_alpha = hue_size==False
    
    plot_str = 'wLum' if plot_luminance else ''
    plot_str = '%s_ci%i' % (plot_str, ci) if plot_ci else plot_str
    
    for visual_area, vdf in results.groupby(['visual_area']):
        if hue_size:
            color_palette=cpalettes[visual_area]
            size_colors = sns.color_palette(color_palette, n_colors=5)

        ai = visual_areas.index(visual_area)
        ax = axn[ai]
        for si, (sz, df_) in enumerate(vdf.groupby(['size'])):
            alpha_val = alphas[si] if use_alpha else 1
            curr_color = size_colors[si] if hue_size else area_colors[visual_area]
            ax = plot_morph_curves(df_, sdf, col_name='morphlevel', 
                                   plot_luminance=plot_luminance, plot_ci=plot_ci, capsize=capsize,
                                   lw=lw, curr_color=curr_color, ax=ax, alpha=alpha_val, label=sz)
        ax.axhline(y=0.5, linestyle=':', color='k', lw=1)

        if plot_legend:
            ax.legend(bbox_to_anchor=(1, 1.1))  
        else:
            if ai==2: 
                ax.legend(bbox_to_anchor=(1, 1.1))
                
    pl.subplots_adjust(left=0.1, right=0.9, bottom=0.2, top=0.8)
    sns.despine(trim=True, offset=4)
    pl.suptitle("Train on anchors, test on intermediates", fontsize=8)
    
    putils.label_figure(fig, data_id)

    figname = '%s_morphcurves_split-size__%s' % (train_str, plot_str)
    pl.savefig(os.path.join(dst_dir, '%s.svg' % figname))
    print(dst_dir, figname)
    
    return
 
def default_morphcurves_avg_size(results, sdf, area_colors=None, dst_dir='/tmp', data_id='DATAID',
                                    lw=2, train_str='train-anchors-test-intermed', capsize=2):
    if area_colors is None:
        visual_areas, area_colors = putils.set_threecolor_palette()

    ci=95
    markersize=5
    plot_luminance=True
    plot_ci=False
    #shade=False
    plot_str = 'wLum' if plot_luminance else ''
    plot_str = '%s_ci%i' % (plot_str, ci) if plot_ci else plot_str

    fig, ax = pl.subplots(dpi=150, figsize=(5,4))
    ax = plot_morph_curves(results, sdf, col_name='morphlevel', ci=ci, plot_luminance=plot_luminance, 
                          lw=lw, capsize=capsize, markersize=markersize, plot_ci=plot_ci,
                           area_colors=area_colors, ax=ax, dpi=150)
    pl.subplots_adjust(left=0.2, bottom=0.2, right=0.8, top=0.8)

    sns.despine(trim=True, offset=4)
    pl.suptitle("Train on anchors, test on intermediates\n(n=%i iters, overlap=%.2f) - avg across size" % (n_iterations, overlap_thr), fontsize=8)

    putils.label_figure(fig, data_id)

    figname = '%s_morphcurves_avg-size__%s' % (train_str, plot_str)
    pl.savefig(os.path.join(dst_dir, '%s.svg' % figname))
    print(dst_dir, figname)

    return


# Train/Test on SIZE SUBSETS
# ----------------------------------------------------------------------------
def plot_scores_by_test_set(results, sdf, metric='heldout_test_score',  
                            area_colors=None, ax=None, plot_sem=True):
    if area_colors is None:
        visual_areas, area_colors = putils.set_threecolor_palette()

    if ax is None:
        fig, ax = pl.subplots(dpi=dpi, figsize=(5,4), sharex=True, sharey=True)
    sizes = [str(s) for s in sdf['size'].unique()]
    markersize=5
    for visual_area, vdf in results.groupby(['visual_area']):

        mean_vals = vdf[vdf['test_transform'].isin(sizes)].groupby(['test_transform']).mean()[metric]
        if plot_sem:
            sem_vals = vdf[vdf['test_transform'].isin(sizes)].groupby(['test_transform']).sem()[metric]
        else:
            sem_vals = vdf[vdf['test_transform'].isin(sizes)].groupby(['test_transform']).std()[metric]

        ax.plot(np.arange(0, len(sizes)), mean_vals, color=area_colors[visual_area],
                   marker='o', markersize=markersize, label=visual_area)
        ax.errorbar(np.arange(0, len(sizes)), mean_vals, yerr=sem_vals, color=area_colors[visual_area],
                   marker='o', markersize=markersize, label=None)

        #ax.set_title(train_transform)
        if metric=='heldout_test_score':
            ax.axhline(y=0.5, color='k', linestyle=':')
            ax.set_ylim([0.4, 1])
        ax.set_xticks(np.arange(0, len(sizes)))
        ax.set_xticklabels(sizes)

    ax.set_xlabel('Test Size', fontsize=16)
    ax.set_ylabel(metric, fontsize=16)
    pl.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.3, wspace=0.3)
    ax.legend(bbox_to_anchor=(1, 1.1))
    return ax

def default_train_test_subset(results, sdf, metric='heldout_test_score', area_colors=None,
                                plot_title='Train on subset, test on remainder',
                                plot_str='traintest-size-subset', dst_dir='/tmp', data_id='DATAID'):

    #if area_colors is None:
    visual_areas, area_colors = putils.set_threecolor_palette()

    # First plot score for each heldout test size
    fig, ax = pl.subplots(dpi=150, figsize=(4,4), sharex=True, sharey=True)
    pl.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.3, wspace=0.3)
    ax.set_title(plot_title, fontsize=8)
    plot_scores_by_test_set(results, sdf, metric=metric, ax=ax)
    sns.despine(trim=True, offset=4)
    putils.label_figure(fig, data_id)

    figname = '%s_generalize_size' % (plot_str)
    pl.savefig(os.path.join(dst_dir, '%s.svg' % figname))
    print(dst_dir, figname)
   
    # Plot "trained" v "novel" for all training subsets
    fig, axn = pl.subplots(1, 3, sharex=True, sharey=True, figsize=(8,3), dpi=150)
    for ax, (visual_area, vdf) in zip(axn.flat[::-1], results.groupby(['visual_area'])):
        means = vdf.groupby(['train_transform', 'test_transform']).mean().reset_index()
        test_on_trained = [float(g[g['test_transform']==train][metric]) \
                            for train, g in means.groupby(['train_transform'])]
        test_on_novel = [float(g[g['test_transform']!=train][metric]) \
                            for train, g in means.groupby(['train_transform'])]
        train_labels = [train for train, g in means.groupby(['train_transform'])]
        
        for train_label, trained, novel in zip(train_labels, test_on_trained, test_on_novel):
            ax.plot([0, 1], [trained, novel], label=train_label)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['trained', 'novel'])
        ax.set_title(visual_area, loc='left')
        #ax.set_ylim([0.4, 1])
    axn[-1].legend(bbox_to_anchor=(1., 1.))
    axn[0].set_ylabel(metric)

    pl.suptitle('Train/test scores')
    pl.subplots_adjust(left=0.1, right=0.7, wspace=0.5, bottom=0.2, top=0.8)
    sns.despine(trim=True, offset=4)
    putils.label_figure(fig, data_id)

    figname = '%s_generalize_size__avg-novel-v-trained' % (plot_str)
    pl.savefig(os.path.join(dst_dir, '%s.svg' % figname))
    print(dst_dir, figname)

    # Then, plot average differences
    means = results.groupby(['visual_area', 'train_transform', 'test_transform']).mean().reset_index()
    trained = pd.concat([g[g['test_transform']==train][['visual_area', metric]]
                    .rename(columns={metric: 'trained'}) \
                    for (visual_area, train), g in means.groupby(['visual_area', 'train_transform'])])
    novel = pd.concat([g[g['test_transform']!=train][['visual_area', metric]]
                    .rename(columns={metric: 'novel'}) \
                    for (visual_area, train), g in means.groupby(['visual_area', 'train_transform'])])
    diff_df = pd.merge(trained, novel)
    diff_df['difference'] = diff_df['novel'].values - diff_df['trained'].values


    fig, ax = pl.subplots(1, sharex=True, sharey=True, figsize=(5,4), dpi=150)
    # sns.stripplot(x='visual_area', y='difference', hue='visual_area', data=diff_df, ax=ax, 
    #               palette=area_colors, order=visual_areas, dodge=True)
    sns.pointplot(x='visual_area', y='difference', hue='visual_area', data=diff_df, ax=ax, 
                  palette=area_colors, order=visual_areas, zorder=0)
    ax.legend_.remove()
    ax.set_ylabel(metric)

    pl.suptitle('Train/test scores')
    pl.subplots_adjust(left=0.2, right=0.7, wspace=0.5, bottom=0.2, top=0.8)
    sns.despine(trim=True, offset=4)

    putils.label_figure(fig, data_id)

    figname = '%s_generalize_size__avg-novel-v-trained-difference' % (plot_str)
    pl.savefig(os.path.join(dst_dir, '%s.svg' % figname))
    print(dst_dir, figname)

    return


#  MISC fitting

import functools
import numpy as np
import scipy.optimize
from scipy.special import erf

def neg_likelihood(pars, data, P_model='weibull', 
                   parmin=np.array([.005, 0., 0.]), parmax=np.array([.5, 10., .25])):
    """
    From: https://github.com/cortex-lab/psychofit/blob/master/psychofit.py
    Compare with scipy.optimize
    
    Negative likelihood of a psychometric function.
    Args:
        pars: Model parameters [threshold, slope, gamma], or if
              using the 'erf_psycho_2gammas' model append a second gamma value.
        data: 3 x n matrix where first row corresponds to stim levels,
              the second to number of trials for each stim level (int),
              the third to proportion correct / proportion rightward (float between 0 and 1)
        P_model: The psychometric function. Possibilities include 'weibull'
                 (DEFAULT), 'weibull50', 'erf_psycho' and 'erf_psycho_2gammas'
        parmin: Minimum bound for parameters.  If None, some reasonable defaults are used
        parmax: Maximum bound for parameters.  If None, some reasonable defaults are used
    Returns:
        l: The likelihood of the parameters.  The equation is:
            - sum(nn.*(pp.*log10(P_model)+(1-pp).*log10(1-P_model)))
            See the the appendix of Watson, A.B. (1979). Probability
            summation over time. Vision Res 19, 515-522.
    """
    # Validate input
    if isinstance(data, (list, tuple)):
        data = np.array(data)
    elif not isinstance(data, np.ndarray):
        raise TypeError('data must be a list or numpy array')

    if data.shape[0] == 3:
        xx = data[0, :]
        nn = data[1, :]
        pp = data[2, :]
    else:
        raise ValueError('data must be m by 3 matrix')

    # here is where you effectively put the constraints.
    if (any(pars < parmin)) or (any(pars > parmax)):
        l = 10000000
        return l

    dispatcher = {
        'weibull': weibull,
        'weibull50': weibull50,
        'weibull_wh': weibull_wh
#         'erf_psycho': erf_psycho,
#         'erf_psycho_2gammas': erf_psycho_2gammas
    }
    try:
        probs = dispatcher[P_model](pars, xx)
    except KeyError:
        raise ValueError('invalid model, options are "weibull", ' +
                         '"weibull50", "erf_psycho" and "erf_psycho_2gammas"')

    assert (max(probs) <= 1) or (min(probs) >= 0), 'At least one of the probabilities is not ' \
                                                   'between 0 and 1'

    probs[probs == 0] = np.finfo(float).eps
    probs[probs == 1] = 1 - np.finfo(float).eps

    l = - sum(nn * (pp * np.log(probs) + (1 - pp) * np.log(1 - probs)))
    
    return l

def weibull(pars, xx):
    """
    From: https://github.com/cortex-lab/psychofit/blob/master/psychofit.py

    Weibull function from 0 to 1, with lapse rate.
    Args:
        pars: Model parameters [alpha, beta, gamma].
        xx: vector of stim levels.
    Returns:
        A vector of length xx
    Raises:
        ValueError: pars must be of length 3
        TypeError: pars must be list-like or numpy array
    """
    # Validate input
    if not isinstance(pars, (list, tuple, np.ndarray)):
        raise TypeError('pars must be list-like or numpy array')

    if len(pars) != 3:
        raise ValueError('pars must be of length 3')

    alpha, beta, gamma = pars
    
    return (1 - gamma) - (1 - 2*gamma) * np.exp(-((xx / alpha)**beta))

def weibull50(pars, xx):
    """
    From: https://github.com/cortex-lab/psychofit/blob/master/psychofit.py

    Weibull function from 0.5 to 1, with lapse rate.
    Args:
        pars: Model parameters [alpha, beta, gamma].
        xx: vector of stim levels.
    Returns:
        A vector of length xx
    Raises:
        ValueError: pars must be of length 3
        TypeError: pars must be list-like or numpy array
    Information:
        2000-04 MC wrote it
        2018-08 MW ported to Python
    """
    # Validate input
    if not isinstance(pars, (list, tuple, np.ndarray)):
        raise TypeError('pars must be list-like or numpy array')

    if len(pars) != 3:
        raise ValueError('pars must be of length 3')

    alpha, beta, gamma = pars
    return (1 - gamma) - (.5 - gamma) * np.exp(-((xx / alpha) ** beta))


def mle_weibull(data, P_model='weibull', parstart=None, parmin=None, parmax=None, nfits=5):
    '''
    From: https://github.com/cortex-lab/psychofit/blob/master/psychofit.py
    Compare with scipy.optimize
    '''
    # Input validation
    if isinstance(data, (list, tuple)):
        data = np.array(data)
    elif not isinstance(data, np.ndarray):
        raise TypeError('data must be a list or numpy array')

    if data.shape[0] != 3:
        raise ValueError('data must be m by 3 matrix')

    if parstart is None:
        parstart = np.array([np.mean(data[0, :]), 0.5, 0.5])
    if parmin is None:
        parmin = np.array([np.min(data[0, :]), 0.0, 0.0])
    if parmax is None:
        parmax = np.array([np.max(data[0, :]), 10., .5])
        
    ii = np.isfinite(data[2, :])

    likelihoods = np.zeros(nfits,)
    pars = np.empty((nfits, parstart.size))
    
    f = functools.partial(neg_likelihood, data=data[:, ii],
                          P_model=P_model, parmin=parmin, parmax=parmax)
    for ifit in range(nfits):
        pars[ifit, :] = scipy.optimize.fmin(f, parstart, disp=False)
        parstart = parmin + np.random.rand(parmin.size) * (parmax-parmin)
        likelihoods[ifit] = -neg_likelihood(pars[ifit, :], data[:, ii], P_model, parmin, parmax)

    # the values to be output
    L = likelihoods.max()
    iBestFit = likelihoods.argmax()
    
    return pars[iBestFit, :], L


def weibull_wh(pars, xx):
    """
    Weibull function from 0.5 to 1, with lapse rate.
    Britten et al., 1993; Wichmann & Hill.
    alpha: threshold
    beta:  slope
    lm:  lambda, ceiling
    gamma: floor (min=0.5)

    Args:
        pars: Model parameters [alpha, beta, lambda, gamma].
        xx: vector of stim levels.
    Returns:
        A vector of length xx
    Raises:
        ValueError: pars must be of length 3
        TypeError: pars must be list-like or numpy array
    Information:
    """
    # Validate input
    if not isinstance(pars, (list, tuple, np.ndarray)):
        raise TypeError('pars must be list-like or numpy array')

    if len(pars) != 3:
        raise ValueError('pars must be of length 3')

    alpha, beta, lm = pars
    #alpha, beta, lm, gamma = pars
    #return gamma + (lm-gamma)*(1-np.exp(-1*((xx/alpha)**beta)))
    #return gamma + (lm-gamma)*(1-np.exp(-1*((xx/alpha)**beta)))
    return lm - (lm-0.5)*np.exp(-((xx/alpha)**beta))



