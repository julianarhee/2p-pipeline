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
from scipy import stats as spstats

from pipeline.python.classifications import experiment_classes as util
from pipeline.python.classifications import aggregate_data_stats as aggr
from pipeline.python.classifications import rf_utils as rfutils
from pipeline.python import utils as putils

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


def get_percentile_shuffled(iterdf, metric='heldout_test_score'):
    # p_thr=0.05
    excl_cols = ['fit_time', 'iteration', 'n_cells',  'randi', 'score_time']
    incl_cols = [c for c in iterdf.columns if c not in excl_cols]

    # Use bootstrap distn to calculate percentiles
    s_=[]
    if 'cell' in iterdf.columns:
        for (visual_area, datakey, rid), d_ in iterdf.groupby(['visual_area', 'datakey', 'cell']):

            mean_score = d_[d_['condition']=='data'][metric].mean()
            percentile = np.mean(mean_score < d_[d_['condition']=='shuffled'][metric])
            n_iterations = d_[d_['condition']=='data'].shape[0]
            rdict = dict(d_[d_['condition']=='data'][incl_cols].mean())
            rdict.update({'visual_area': visual_area, 'datakey': datakey, 'cell': rid, 
                           'mean_score': mean_score, 'percentile': percentile, 'n_iterations': n_iterations})
            s = pd.Series(rdict)
            s_.append(s)
    else:
         for (visual_area, datakey), d_ in iterdf.groupby(['visual_area', 'datakey']):

            mean_score = d_[d_['condition']=='data'][metric].mean()
            percentile = np.mean(mean_score < d_[d_['condition']=='shuffled'][metric])
            n_iterations = d_[d_['condition']=='data'].shape[0]
            rdict = dict(d_[d_['condition']=='data'][incl_cols].mean())
            rdict.update({'visual_area': visual_area, 'datakey': datakey, 
                           'mean_score': mean_score, 'percentile': percentile, 'n_iterations': n_iterations})
            s = pd.Series(rdict)
            s_.append(s)
            
    scores_by_cell = pd.concat(s_, axis=1).T.reset_index(drop=True)
    
    return scores_by_cell


# ======================================================================
# Load/Aggregate data functions 
# ======================================================================
# load_aggregate_rfs : now in rf_utils.py
# get_rf_positions: in rf_utils.py
def pool_bootstrap(neuraldf, sdf, n_iterations=50, n_processes=1, C_value=None,
                   test=None, single=False, n_train_configs=4, verbose=False):   
    '''
    test (string, None)
        None  : Classify A/B only 
                single=True to train/test on each size
        morph : Train on anchors, test on intermediate morphs
                single=True to train/test on each size
        size  : Train on specific size(s), test on un-trained sizes
                single=True to train/test on each size
    '''
    C=C_value
    vb = verbose 
    results = []
    terminating = mp.Event()

    pool = mp.Pool(initializer=initializer, initargs=(terminating, ), processes=n_processes)  
    try:
        ntrials, sample_ncells = neuraldf.shape
        print("... n: %i (%i procs)" % (int(sample_ncells-1), n_processes))
#        if test=='morph':
#            if single: # train on 1 size, test on other sizes
#                func = partial(dutils.do_fit_train_single_test_morph, global_rois=global_rois, 
#                               MEANS=MEANS, sdf=sdf, sample_ncells=sample_ncells)
#            else: # combine data across sizes
#                func = partial(dutils.do_fit_train_test_morph, global_rois=global_rois, 
#                               MEANS=MEANS, sdf=sdf, sample_ncells=sample_ncells)             
#        elif test=='size':
#            if single:
#                func = partial(dutils.do_fit_train_test_single, global_rois=global_rois, 
#                               MEANS=MEANS, sdf=sdf, sample_ncells=sample_ncells)
#            else:
#                func = partial(dutils.cycle_train_sets, global_rois=global_rois, 
#                               MEANS=MEANS, sdf=sdf, sample_ncells=sample_ncells, 
#                                n_train_configs=n_train_configs)
#        else:
#        func = partial(dutils.do_fit_within_fov, curr_data=neuraldf, sdf=sdf, verbose=verbose,
#                                                    C_value=C_value)
#        results = pool.map_async(func, range(n_iterations)).get(99999999)
        results = [pool.apply_async(dutils.do_fit_within_fov, args=(i, neuraldf, sdf, vb, C)) \
                    for i in range(n_iterations)]
        output= [p.get(99999999) for p in results]
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

    return output #results


#def fit_svm_mp_shuffle(neuraldf, sdf, n_iterations=50, n_processes=1, 
#                   C_value=None, cv_nfolds=5, test_split=0.2, 
#                   test=None, single=False, n_train_configs=4, verbose=False,
#                   class_a=0, class_b=106):   
#    #### Select train/test configs for clf A vs B
#    train_configs = sdf[sdf['morphlevel'].isin([class_a, class_b])].index.tolist() 
#
#    #### Define MP worker
#    results = []
#    terminating = mp.Event() 
#    def worker(n_iters, neuraldf, sdf, C_value, verbose, class_a, class_b, out_q, shu_q):
#        r_ = []; s_=[];
#        for ni in n_iters:
#            curr_iter, curr_shuf = do_fit_within_fov(ni, curr_data=neuraldf, sdf=sdf, 
#                                        C_value=C_value, class_a=class_a, class_b=class_b,                                            verbose=verbose, return_shuffle=True)
#            r_.append(curr_iter)
#            s_.append(curr_shuf)
#        curr_iterdf = pd.concat(r_, axis=0)
#        curr_shufdf = pd.concat(s_, axis=0)
#        out_q.put(curr_iterdf) 
#        shu_q.put(curr_shufdf)
#
#    try:        
#        # Each process gets "chunksize' filenames and a queue to put his out-dict into:
#        iter_list = np.arange(0, n_iterations) #gdf.groups.keys()
#        out_q = mp.Queue()
#        shu_q = mp.Queue()
#        chunksize = int(math.ceil(len(iter_list) / float(n_processes)))
#        procs = []
#        for i in range(n_processes):
#            p = mp.Process(target=worker,
#                           args=(iter_list[chunksize * i:chunksize * (i + 1)],
#                                          neuraldf, sdf, C_value, verbose, class_a, class_b,
#                                          out_q, shu_q))
#            procs.append(p)
#            p.start()
#
#        # Collect all results into 1 results dict. We should know how many dicts to expect:
#        results = []
#        results_shuffled = []
#        for i in range(n_processes):
#            results.append(out_q.get())
#            results_shuffled.append(shu_q.get())
#        # Wait for all worker processes to finish
#        for p in procs:
#            #print "Finished:", p
#            p.join()
#    except KeyboardInterrupt:
#        terminating.set()
#        print("***Terminating!")
#    except Exception as e:
#        traceback.print_exc()
#    finally:
#        for p in procs:
#            p.join    
#    return results, results_shuffled
#
def fit_svm_mp(neuraldf, sdf, n_iterations=50, n_processes=1, 
                   C_value=None, cv_nfolds=5, test_split=0.2, 
                   test=None, single=False, n_train_configs=4, verbose=False,
                   class_a=0, class_b=106, do_shuffle=True):   
    #### Select train/test configs for clf A vs B
    train_configs = sdf[sdf['morphlevel'].isin([class_a, class_b])].index.tolist() 
    
    #### Define MP worker
    results = []
    terminating = mp.Event() 
    def worker(n_iters, neuraldf, sdf, C_value, verbose, class_a, class_b, do_shuffle, out_q):
        r_ = []        
        for ni in n_iters:
            curr_iter = do_fit_within_fov(ni, curr_data=neuraldf, sdf=sdf, 
                                        C_value=C_value, class_a=class_a, class_b=class_b, 
                                        verbose=verbose, do_shuffle=do_shuffle)
            r_.append(curr_iter)
        curr_iterdf = pd.concat(r_, axis=0)
        out_q.put(curr_iterdf) 

    try:        
        # Each process gets "chunksize' filenames and a queue to put his out-dict into:
        iter_list = np.arange(0, n_iterations) #gdf.groups.keys()
        out_q = mp.Queue()
        chunksize = int(math.ceil(len(iter_list) / float(n_processes)))
        procs = []
        for i in range(n_processes):
            p = mp.Process(target=worker,
                           args=(iter_list[chunksize * i:chunksize * (i + 1)],
                                          neuraldf, sdf, C_value, verbose, class_a, class_b,
                                          do_shuffle, out_q))
            procs.append(p)
            p.start() # start asynchronously

        # Collect all results into 1 results dict. We should know how many dicts to expect:
        results = []
        for i in range(n_processes):
            results.append(out_q.get())
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
    return results


def by_ncells_fit_svm_mp(ncells, celldf, NEURALDATA, sdf, 
                           n_iterations=50, n_processes=1, 
                           C_value=None, cv_nfolds=5, test_split=0.2, 
                           test=None, single=False, n_train_configs=4, verbose=False,
                           class_a=0, class_b=106):   
    #### Select train/test configs for clf A vs B
    train_configs = sdf[sdf['morphlevel'].isin([class_a, class_b])].index.tolist() 

    #### Define multiprocessing worker
    results = []
    terminating = mp.Event()    
    def worker(n_iters, ncells, celldf, sdf, NEURALDATA, C_value, class_a, class_b, out_q):
        r_ = []        
        for ni in n_iters:
            curr_iter = do_fit(ni, sample_ncells=ncells, global_rois=celldf, sdf=sdf,
                                        MEANS=NEURALDATA, df_is_split=True, do_shuffle=True, 
                                        C_value=C_value, class_a=class_a, class_b=class_b)
            r_.append(curr_iter)
        curr_iterdf = pd.concat(r_, axis=0)
        out_q.put(curr_iterdf)
        
    try:        
        # Each process gets "chunksize' filenames and a queue to put his out-dict into:
        iter_list = np.arange(0, n_iterations) #gdf.groups.keys()
        out_q = mp.Queue()
        chunksize = int(math.ceil(len(iter_list) / float(n_processes)))
        procs = []
        for i in range(n_processes):
            p = mp.Process(target=worker,
                           args=(iter_list[chunksize * i:chunksize * (i + 1)],
                                 ncells, celldf, sdf, NEURALDATA, C_value,
                                 class_a, class_b, out_q))
            procs.append(p)
            p.start()

        # Collect all results into single results dict. We should know how many dicts to expect:
        results = []
        for i in range(n_processes):
            results.append(out_q.get(99999))
        # Wait for all worker processes to finish
        for p in procs:
            p.join()
    except KeyboardInterrupt:
        terminating.set()
        print("***Terminating!")
    except Exception as e:
        traceback.print_exc()
    finally:
        for p in procs:
            p.join()

    return results

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


def get_trials_for_N_cells(curr_ncells, gdf, MEANS):
    '''
    Randomly select N cells from global roi list (gdf), get cell's responses to all trials.
    
    gdf = dataframe (subset of global_rois dataframe), contains 
    - all rois for a given visual area
    - corresponding within-datakey roi IDs
    '''
    # Get current global RIDs
    ncells_t = gdf.shape[0]                      
    roi_ids = np.array(gdf['roi'].values.copy()) 

    # Random sample w/ replacement
    rand_ixs = np.array([random.randint(0, ncells_t-1) for _ in np.arange(0, curr_ncells)])
    curr_roi_list = roi_ids[rand_ixs]
    curr_roidf = gdf[gdf['roi'].isin(curr_roi_list)].copy()

    # Make sure equal num trials per condition for all dsets
    min_ntrials_by_config = min([MEANS[k]['config'].value_counts().min() for k in curr_roidf['datakey'].unique()])

    # Get data samples for these cells
    d_list=[]; c_list=[];
    for datakey, dkey_rois in curr_roidf.groupby(['datakey']):
        # Get subset of trials per cond to match min N trials
        tmpd_list=[]
        for cfg, trialmat in MEANS[datakey].groupby(['config']):
            # Get indices of trials in current dataset
            trial_ixs = trialmat.index.tolist() 
            # Shuffle them to get random order
            np.random.shuffle(trial_ixs)                
            # Select min_ntrials randomly
            curr_cfg_trials = trialmat.loc[trial_ixs[0:min_ntrials_by_config]].copy() 
            # Add current trials of current config to list
            tmpd_list.append(curr_cfg_trials)        
        tmpd = pd.concat(tmpd_list, axis=0) 

        # For each RID sample belonging to current dataset, get RID order
        sampled_cells = pd.concat([dkey_rois[dkey_rois['roi']==globalid][['roi', 'dset_roi']]                                          for globalid in curr_roi_list])
        sampled_dset_rois = sampled_cells['dset_roi'].values
        sampled_global_rois = sampled_cells['roi'].values

        # Get trial responses (some columns are repeats)
        curr_roidata = tmpd[sampled_dset_rois].copy().reset_index(drop=True)
        assert len(sampled_global_rois)==curr_roidata.shape[1], "Incorrect column grabbing" 
        curr_roidata.columns = sampled_global_rois # Rename ROI columns to global-rois
        config_list = tmpd['config'].reset_index(drop=True)  # Get configs on selected trials
        d_list.append(curr_roidata)
        c_list.append(config_list)
    curr_neuraldf = pd.concat(d_list, axis=1).reset_index(drop=True)

    cfg_df = pd.concat(c_list, axis=1)
    cfg_df = cfg_df.T.drop_duplicates().T
    assert cfg_df.shape[0]==curr_neuraldf.shape[0], "Bad trials"

    assert curr_configdf.shape[1]==1, "Bad configs: %s" % str(curr_roidf['datakey'].unique())
    df = pd.concat([curr_neuraldf, cfg_df], axis=1)

    return df


def get_trials_for_N_cells_split(curr_ncells, gdf, NEURALDATA):
    '''
    Randomly select N cells from global roi list (gdf), get cell's responses to all trials.
    NEURALDATA (dict)
        keys : visual areas
        vals : same as MEANS() -- i.e., dict, k=datakey, v=neuraldf
 
    gdf = dataframe (subset of global_rois dataframe), contains 
    - all rois for a given visual area
    - corresponding within-datakey roi IDs
    '''
    # Get current global RIDs
    ncells_t = gdf.shape[0]
    roi_ids = np.array(gdf['roi'].values.copy())
    len(roi_ids)

    # Random sample w/ replacement
    rand_ixs = np.array([random.randint(0, ncells_t-1) for _ in np.arange(0, curr_ncells)])
    curr_roi_list = roi_ids[rand_ixs] # can have repeats 
    curr_roidf = gdf[gdf['roi'].isin(curr_roi_list)].copy() # N rows=N unique cells
    # len(curr_roidf['roi'].unique()), len(curr_roi_list), len(np.unique(curr_roi_list))

    # Make sure equal num trials per condition for all dsets
    min_ntrials_per_cfg = min([NEURALDATA[visual_area][datakey]['config'].value_counts().min() \
                            for (visual_area, datakey), g in \
                            curr_roidf.groupby(['visual_area', 'datakey'])])

    # Get data samples for these cells
    d_list=[]; c_list=[];
    for (varea, dkey), dkey_rois in curr_roidf.groupby(['visual_area', 'datakey']):
            
        neuraldf = NEURALDATA[varea][dkey].copy()
        resampled_df = get_ntrials_per_config(neuraldf, n_trials=min_ntrials_per_cfg)
        #resampled_df.sort_index(inplace=True) # resort by trial index
        
        # For each RID sample belonging to current dataset, get RID order
        sampled_cells = pd.concat([dkey_rois[dkey_rois['roi']==globalid][['roi', 'dset_roi']] 
                                   for globalid in curr_roi_list])
        # sampled_cells['roi'].value_counts()
        sampled_dset_rois = sampled_cells['dset_roi'].values
        sampled_global_rois = sampled_cells['roi'].values

        # Get trial responses (some columns are repeats)
        curr_roidata = resampled_df[sampled_dset_rois].copy().reset_index(drop=True)
        assert len(sampled_global_rois)==curr_roidata.shape[1], "Incorrect column grabbing"
        curr_roidata.columns = sampled_global_rois # Rename ROI columns to global-rois
        config_list = resampled_df['config'].reset_index(drop=True)  # Get configs on selected trials
        d_list.append(curr_roidata)
        c_list.append(config_list)

    curr_neuraldf = pd.concat(d_list, axis=1).reset_index(drop=True)
    curr_configdf = pd.concat(c_list, axis=1).T.drop_duplicates().T

    assert curr_configdf.shape[0]==curr_neuraldf.shape[0], "Bad trials"
    assert cfg_df.shape[1]==1, "Bad configs: %s" % str(curr_roidf['datakey'].unique()) #cfg_df
    df = pd.concat([curr_neuraldf, curr_configdf], axis=1)

    return df

def get_trials_for_N_cells_df(curr_ncells, gdf, NEURALDATA): 
    # Get current global RIDs
    ncells_t = gdf.shape[0]                      
    roi_ids = np.array(gdf['roi'].values.copy()) 

    # Random sample w/ replacement
    rand_ixs = np.array([random.randint(0, ncells_t-1) for _ in np.arange(0, curr_ncells)])
    curr_roi_list = roi_ids[rand_ixs]
    curr_roidf = gdf[gdf['roi'].isin(curr_roi_list)].copy()

    # Make sure equal num trials per condition for all dsets
    curr_dkeys = curr_roidf['datakey'].unique()
    currd = NEURALDATA[NEURALDATA['datakey'].isin(curr_dkeys)].copy()
    min_ntrials_by_config = currd[['datakey', 'config', 'trial']].drop_duplicates().groupby(['datakey'])['config'].value_counts().min()
    #print(min_ntrials_by_config)

    d_list=[]
    for datakey, dkey_rois in curr_roidf.groupby(['datakey']):
        assert datakey in currd['datakey'].unique(), "ERROR: %s not found" % datakey
        # Get current trials, make equal to min_ntrials_by_config
        tmpd = pd.concat([trialmat.sample(n=min_ntrials_by_config) 
                         for (rid, cfg), trialmat in currd[currd['datakey']==datakey].groupby(['cell', 'config'])], axis=0)
        tmpd['cell'] = tmpd['cell'].astype(float)

        # For each RID sample belonging to current dataset, get RID order
        sampled_cells = pd.concat([dkey_rois[dkey_rois['roi']==globalid][['roi', 'dset_roi']] 
                                   for globalid in curr_roi_list])
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
        cfg_df = cfg_df.loc[:,~cfg_df.T.duplicated(keep='first')]
        assert cfg_df.shape[1]==1, "Bad configs: %s" % str(curr_roidf['datakey'].unique()) #cfg_df

    df = pd.concat([curr_neuraldf, cfg_df], axis=1)
    
    return df

# ======================================================================
# Fitting functions 
# ======================================================================
def tune_C(sample_data, target_labels, scoring_metric='accuracy', 
                        cv_nfolds=5, test_split=0.2, verbose=False, n_processes=1):
    
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
                        scoring=scoring_metric, cv=cv_nfolds, n_jobs=-1) #n_processes)
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

def fit_svm_shuffle(zdata, targets, test_split=0.2, cv_nfolds=5, verbose=False, C_value=None, randi=10):

    cv = C_value is None

    if verbose:
        print("Labels=%s" % (str(targets['label'].unique())))

    #### For each transformation, split trials into 80% and 20%
    train_data, test_data, train_labels, test_labels = train_test_split(zdata, 
                                                        targets['label'].values, 
                                                        test_size=test_split, 
                                                        stratify=targets['label'], #targets['group'],
                                                        shuffle=True, random_state=randi)
    #print("first few:", test_labels[0:10])
    #### Cross validate (tune C w/ train data)
    if cv:
        cv_grid = tune_C(train_data, train_labels, scoring_metric='accuracy', 
                            cv_nfolds=3, #cv_nfolds, 
                           test_split=test_split, verbose=verbose) #, n_processes=n_processes)

        C_value = cv_grid.best_params_['C'] #cv_results['accuracy']['C']
    else:
        assert C_value is not None, "Provide value for hyperparam C..."
    #trained_svc = cv_grid.best_estimator_

    #### Fit SVM
    scaler = StandardScaler().fit(train_data)
    train_data = scaler.transform(train_data)

    svc = svm.SVC(kernel='linear', C=C_value, random_state=randi) #, random_state=10)
    #print("... cv")
    scores = cross_validate(svc, train_data, train_labels, cv=cv_nfolds,
                            scoring=('accuracy'),
                            #scoring=('precision_macro', 'recall_macro', 'accuracy'),
                            return_train_score=True)
    iterdict = dict((s, values.mean()) for s, values in scores.items())
    if verbose:
        print('... train (C=%.2f): %.2f, test: %.2f' % (C_value, iterdict['train_score'], iterdict['test_score']))
    trained_svc = svc.fit(train_data, train_labels)
       
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
    mi = skmetrics.mutual_info_score(test_labels, predicted_labels)
    ami = skmetrics.adjusted_mutual_info_score(test_labels, predicted_labels)
    log2_mi = computeMI(test_labels, predicted_labels)
    iterdict.update({'heldout_test_score': test_score, 
                     'heldout_MI': mi, 'heldout_aMI': ami, 'heldout_log2MI': log2_mi,
                     'C': C_value, 'randi': randi})

    # ------------------------------------------------------------------
    # Shuffle LABELS to calculate chance level
    train_labels_chance = train_labels.copy()
    np.random.shuffle(train_labels_chance)
    test_labels_chance = test_labels.copy()
    np.random.shuffle(test_labels_chance)

    #### CHANCE - Fit classifier
    chance_svc = svm.SVC(kernel='linear', C=C_value, random_state=randi)
    scores_chance = cross_validate(chance_svc, train_data, train_labels_chance, cv=cv_nfolds,
                            scoring=('accuracy'),
                            #scoring=('precision_macro', 'recall_macro', 'accuracy'),
                            return_train_score=True)
    iterdict_chance = dict((s, values.mean()) for s, values in scores_chance.items())

    # CHANCE - Test with held-out data
    trained_svc_chance = chance_svc.fit(train_data, train_labels_chance)
    test_score_chance = trained_svc_chance.score(test_data, test_labels_chance)  

    # Chance - Calculate MI
    predicted_labels = trained_svc_chance.predict(test_data)
    mi = skmetrics.mutual_info_score(test_labels, predicted_labels)
    ami = skmetrics.adjusted_mutual_info_score(test_labels, predicted_labels)
    log2_mi = computeMI(test_labels, predicted_labels)

    iterdict_chance.update({'heldout_test_score': test_score_chance, 
                            'heldout_MI': mi, 'heldout_aMI': ami, 
                            'heldout_log2MI': log2_mi, 'C': C_value, 'randi': randi})

    return iterdict, iterdict_chance


def fit_svm_no_C(train_data, test_data,train_labels, test_labels, C_value=None, cv_nfolds=5):
    #### Fit SVM
    scaler = StandardScaler().fit(train_data)
    train_data = scaler.transform(train_data)

    trained_svc = svm.SVC(kernel='linear', C=C_value, random_state=10)
    print("... cv")
    scores = cross_validate(trained_svc, train_data, train_labels, cv=cv_nfolds,
                            scoring=('accuracy'),
                            #scoring=('precision_macro', 'recall_macro', 'accuracy'),
                            return_train_score=True)
    iterdict = dict((s, values.mean()) for s, values in scores.items())
    print('train (C=%.2f):' % C_value, iterdict)
    print("... fit")
    trained_svc = svm.SVC(kernel='linear', C=C_value, random_state=10).fit(train_data, train_labels)
       
    print("... test") 
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
    mi = skmetrics.mutual_info_score(test_labels, predicted_labels)
    ami = skmetrics.adjusted_mutual_info_score(test_labels, predicted_labels)
    log2_mi = computeMI(test_labels, predicted_labels)
    iterdict.update({'heldout_test_score': test_score, 
                     'heldout_MI': mi, 'heldout_aMI': ami, 'heldout_log2MI': log2_mi,
                     'C': C_value})
    if return_clf:
        if return_predictions:
            return iterdict, trained_svc, scaler, (predicted_labels, test_labels)
        else:
            return iterdict, trained_svc, scaler
    else:
        return iterdict


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
    #print("first few:", test_labels[0:10])
    #### Cross validate (tune C w/ train data)
    if cv:
        cv_grid = tune_C(train_data, train_labels, scoring_metric='accuracy', 
                            cv_nfolds=3, #cv_nfolds, 
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
    trained_svc = svm.SVC(kernel='linear', C=C_value, random_state=10).fit(train_data, train_labels)
       
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
    mi = skmetrics.mutual_info_score(test_labels, predicted_labels)
    ami = skmetrics.adjusted_mutual_info_score(test_labels, predicted_labels)
    log2_mi = computeMI(test_labels, predicted_labels)
    iterdict.update({'heldout_test_score': test_score, 
                     'heldout_MI': mi, 'heldout_aMI': ami, 'heldout_log2MI': log2_mi,
                     'C': C_value})
    if return_clf:
        if return_predictions:
            return iterdict, trained_svc, scaler, (predicted_labels, test_labels)
        else:
            return iterdict, trained_svc, scaler
    else:
        return iterdict


# --------------------------------------------------------------------------------
# Wrappers for fitting functions - specifies what type of analysis to do
# --------------------------------------------------------------------------------
def do_fit_within_fov(iter_num, curr_data=None, sdf=None, verbose=False,
                    C_value=None, test_split=0.2, cv_nfolds=5, class_a=0, class_b=106,
                    do_shuffle=True):

    #[gdf, MEANS, sdf, sample_ncells, cv] * n_times)
    '''
    Does SVC fit for cells within FOV (no global rois). Assumes 'config' column in curr_data.
    Does n_iterations, return mean/sem/std over iterations as dict of results.
    Classes (class_a, class_b) should be the labels of the target (i.e., value of morph level).
   
    do_shuffle (bool):  Runs fit_svm() twice, once reg and once with labels shuffled. 
    '''   
    #### Select train/test configs for clf A vs B
    train_configs = sdf[sdf['morphlevel'].isin([class_a, class_b])].index.tolist() 

    #### Get trial data for selected cells and config types
    curr_roi_list = [int(c) for c in curr_data.columns if c != 'config']
    sample_data = curr_data[curr_data['config'].isin(train_configs)]
    zdata = sample_data.drop('config', 1)

    #### Get labels
    targets = pd.DataFrame(sample_data['config'].copy(), columns=['config'])
    targets['label'] = [sdf['morphlevel'][cfg] for cfg in targets['config'].values]
    targets['group'] = [sdf['size'][cfg] for cfg in targets['config'].values]

    if verbose:
        print("Labels: %s\nGroups: %s" % (str(targets['label'].unique()), str(targets['group'].unique())))

    #### Fit
    cv = C_value is None
    randi = random.randint(1, 10000)
    
#    if do_shuffle:
#        curr_iter, curr_shuff = fit_svm_shuffle(zdata, targets, C_value=C_value, 
#                                verbose=verbose,
#                                test_split=test_split, cv_nfolds=cv_nfolds, randi=randi)
#        tmp_iter_df = pd.DataFrame(curr_iter, index=[iter_num])
#        shuff_df = pd.DataFrame(curr_shuff, index=[iter_num])
#        tmp_iter_df['condition'] = 'data'
#        shuff_df['condition'] = 'shuffled'
#        iter_df = pd.concat([tmp_iter_df, shuff_df], axis=0)
#    #else:
    curr_iter = fit_svm(zdata, targets, C_value=C_value, verbose=verbose,
                            test_split=test_split, cv_nfolds=cv_nfolds, randi=randi)
    tmp_df = pd.DataFrame(curr_iter, index=[iter_num])

    if do_shuffle:
        labels_shuffled = targets['label'].copy().values 
        np.random.shuffle(labels_shuffled)
        targets['label'] = labels_shuffled
        curr_iter_shuffled = fit_svm(zdata, targets, C_value=C_value, verbose=verbose,
                                test_split=test_split, cv_nfolds=cv_nfolds, randi=randi)
        iter_df_shuffled = pd.DataFrame(curr_iter_shuffled, index=[iter_num])
    
        # combine
        tmp_df['condition'] = 'data'
        iter_df_shuffled['condition'] = 'shuffled'
        iter_df = pd.concat([tmp_df, iter_df_shuffled], axis=0)
    else:
        iter_df = tmp_df.copy()
  
    # Meta info
    iter_df['n_cells'] = zdata.shape[1]
    iter_df['n_trials'] = zdata.shape[0]

    return iter_df 


def do_fit(iter_num, global_rois=None, MEANS=None, sdf=None, sample_ncells=None, 
           C_value=None, test_split=0.2, cv_nfolds=5, class_a=0, class_b=106, 
           do_shuffle=True, df_is_split=True, verbose=False):
    #[gdf, MEANS, sdf, sample_ncells, cv] * n_times)
    '''
    Resample w/ replacement from pooled cells (across datasets). Assumes 'sdf' is same for all datasets.
    Do n_iterations, return mean/sem/std over iterations as dict of results.
    Classes (class_a, class_b) should be the actual labels of the target (i.e., value of morph level)

    df_is_split (bool):  Whether MEANS dict is actually MEANS[visual_area][datakey]
    do_shuffle (bool):   Calls 'fit_svm' twice, once on true data, once on shuffled labels
    
    '''   
    #### Get new sample set
    try:
        if isinstance(MEANS, dict):
            if df_is_split:
                curr_data = get_trials_for_N_cells_split(sample_ncells, global_rois, MEANS)
            else:
                curr_data = get_trials_for_N_cells(sample_ncells, global_rois, MEANS)
        else:
            curr_data = get_trials_for_N_cells_df(sample_ncells, global_rois, MEANS)
    except Exception as e:
        traceback.print_exc()
        return None

    #### Select train/test configs for clf A vs B
    train_configs = sdf[sdf['morphlevel'].isin([class_a, class_b])].index.tolist() 

    #### Get trial data for selected cells and config types
    curr_roi_list = [int(c) for c in curr_data.columns if c != 'config']
    sample_data = curr_data[curr_data['config'].isin(train_configs)]
    zdata = sample_data.drop('config', 1) #sample_data[curr_roi_list].copy()
    #zdata = (data - data.mean()) / data.std()

    #### Get labels
    targets = pd.DataFrame(sample_data['config'].copy(), columns=['config'])
    targets['label'] = [sdf['morphlevel'][cfg] for cfg in targets['config'].values]
    targets['group'] = [sdf['size'][cfg] for cfg in targets['config'].values]

    #### Fit
    randi = random.randint(1, 10000)
    curr_iter = fit_svm(zdata, targets, C_value=C_value, test_split=test_split, 
                            cv_nfolds=cv_nfolds, randi=randi)

    if do_shuffle:
        #print("... shuffling")
        labels_shuffled = targets['label'].copy().values 
        np.random.shuffle(labels_shuffled)
        targets['label'] = labels_shuffled
        curr_iter_shuffled = fit_svm(zdata, targets, C_value=C_value, verbose=verbose,
                                test_split=test_split, cv_nfolds=cv_nfolds, randi=randi)
        iter_df_shuffled = pd.DataFrame(curr_iter_shuffled, index=[iter_num])
    
        # combine
        tmp_df = pd.DataFrame(curr_iter, index=[iter_num]) 
        tmp_df['condition'] = 'data'
        iter_df_shuffled['condition'] = 'shuffled'
        iter_df = pd.concat([tmp_df, iter_df_shuffled], axis=0)
 
    else:
        iter_df = pd.DataFrame(curr_iter, index=[iter_num])

    return iter_df 


def do_fit_train_test_single(iter_num, global_rois=None, MEANS=None, sdf=None, sample_ncells=None,
                            cv=True, C_value=None, test_size=0.2, cv_nfolds=5, class_a=0, class_b=106):
    '''
    Train/test PER SIZE.

    Resample w/ replacement from pooled cells (across datasets). 
    Assumes 'sdf' is same for all datasets.

    Return fit results for 1 iteration.
    Classes (class_a, class_b) should be labels of the target (i.e., value of morph level)
    '''
    # Get new sample set
    curr_data = get_trials_for_N_cells(sample_ncells, global_rois, MEANS)

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
        curr_roi_list = [int(c) for c in curr_data.columns if c != 'config']
        trainset = curr_data[curr_data['config'].isin(train_configs)].copy()
        train_data = trainset.drop('config', 1)#zdata = (data - data.mean()) / data.std()

        #### TRAIN SET: Get labels
        targets = pd.DataFrame(trainset['config'].copy(), columns=['config'])
        targets['label'] = [sdf['morphlevel'][cfg] for cfg in targets['config'].values]
        targets['group'] = [sdf['size'][cfg] for cfg in targets['config'].values]

        # Select generalization-test set
        # untrained_class_types = [c for c in stimdf[class_name].unique() if c not in class_types]
        test_configs = sdf[((sdf[class_name].isin(class_types))\
                                & (sdf[constant_transform]!=train_transform))].index.tolist()
        testset = curr_data[curr_data['config'].isin(test_configs)]
        test_data = testset.drop('config', 1) #zdata = (data - data.mean()) / data.std()

        test_targets = pd.DataFrame(testset['config'].copy(), columns=['config'])
        test_targets['label'] = [sdf['morphlevel'][cfg] for cfg in test_targets['config'].values]
        test_targets['group'] = [sdf['size'][cfg] for cfg in test_targets['config'].values]

        #### Train SVM
        randi = random.randint(1, 10000)
        iterdict, trained_svc, trained_scaler = fit_svm(train_data, targets, return_clf=True,
                                                test_split=test_size, cv_nfolds=cv_nfolds, 
                                                C_value=C_value, randi=randi)
        iterdict.update({'train_transform': train_transform, 'test_transform': train_transform})
        i_list.append(pd.DataFrame(iterdict, index=[i]))
        i+=1
        
        #### Test SVM
        for test_transform, curr_test_group in test_targets.groupby(['group']):
            curr_test_labels = curr_test_group['label'].values
            curr_test_data = test_data.loc[curr_test_group.index].copy()
            curr_test_data = trained_scaler.transform(curr_test_data)
            #test_labels = test_targets['label'].values
            curr_test_score = trained_svc.score(curr_test_data, curr_test_labels)
            #print(test_transform, curr_test_score)

            #### Calculate additional metrics (MI)
            predicted_labels = trained_svc.predict(curr_test_data)
            mi = skmetrics.mutual_info_score(curr_test_labels, predicted_labels)
            ami = skmetrics.adjusted_mutual_info_score(curr_test_labels, predicted_labels)
            log2_mi = computeMI(curr_test_labels, predicted_labels)
            
            iterdict.update({'heldout_test_score': curr_test_score, 
                             'heldout_MI': mi, 'heldout_aMI': ami, 'heldout_log2MI': log2_mi,
                             'train_transform': train_transform,
                             'test_transform': test_transform}) 
            i_list.append(pd.DataFrame(iterdict, index=[i]))
            i+=1 
    iterdf = pd.concat(i_list, axis=0).reset_index(drop=True)
    iterdf['iteration'] = [iter_num for _ in np.arange(0, len(iterdf))]
    
    return iterdf


def do_fit_train_test_subset(iter_num, global_rois=None, MEANS=None, sdf=None, sample_ncells=None,
                             train_sizes=[10, 30, 50], test_sizes=[20, 40],
                             cv=True, C_value=None, test_split=0.2, cv_nfolds=5, class_a=0, class_b=106):
    '''
    Resample w/ replacement from pooled cells (across datasets). Assumes 'sdf' is same for all datasets.
    Return fit results for 1 iteration.
    Classes (class_a, class_b) should be the actual labels of the target (i.e., value of morph level)
    '''
    
    # Get new sample set
    curr_data = get_trials_for_N_cells(sample_ncells, global_rois, MEANS)

    #### Select train/test configs for clf A vs B
    class_types = [class_a, class_b]
    restrict_transform = True
    class_name = 'morphlevel'
    constant_transform = 'size'
    sizes = sorted(sdf[constant_transform].unique())
    
    i_list=[]
    i=0
    # Go thru all training sizes, then test on non-trained sizes
    #for train_transform in sizes:
    
    # Get train configs
    train_configs = sdf[((sdf[class_name].isin(class_types))\
                            & (sdf[constant_transform].isin(train_sizes)))].index.tolist()

    #### TRAIN SET: Get trial data for selected cells and config types
    curr_roi_list = np.array([int(c) for c in curr_data.columns if c != 'config'])
    train_subset = curr_data[curr_data['config'].isin(train_configs)].copy()
    train_data = train_subset.drop('config', 1)#zdata = (data - data.mean()) / data.std()

    #### TRAIN SET: Get labels
    targets = pd.DataFrame(train_subset['config'].copy(), columns=['config'])
    targets['label'] = [sdf['morphlevel'][cfg] for cfg in targets['config'].values]
    targets['group'] = [sdf['size'][cfg] for cfg in targets['config'].values]


    # Select generalization-test set
    # untrained_class_types = [c for c in stimdf[class_name].unique() if c not in class_types]
    test_configs = sdf[((sdf[class_name].isin(class_types))\
                            & (sdf[constant_transform].isin(test_sizes)))].index.tolist()
    test_subset = curr_data[curr_data['config'].isin(test_configs)]
    test_data = test_subset.drop('config', 1) #zdata = (data - data.mean()) / data.std()

    test_targets = pd.DataFrame(test_subset['config'].copy(), columns=['config'])
    test_targets['label'] = [sdf['morphlevel'][cfg] for cfg in test_targets['config'].values]
    test_targets['group'] = [sdf['size'][cfg] for cfg in test_targets['config'].values]

    #### Train SVM 
    randi = random.randint(1, 10000)
    iterdict, trained_svc, trained_scaler = fit_svm(train_data, targets, return_clf=True,
                                                    test_split=test_split, cv_nfolds=cv_nfolds, 
                                                     C_value=C_value, randi=randi)
    iterdict.update({'train_transform': '_'.join([str(s) for s in train_sizes]),
                     'test_transform': '_'.join([str(s) for s in train_sizes])})

    i_list.append(pd.DataFrame(iterdict, index=[i]))
    i+=1

    #### Test SVM
    #for test_transform, curr_test_group in test_targets.groupby(['group']):
    test_labels = test_targets['label'].values
    test_data = trained_scaler.transform(test_data)
    curr_test_score = trained_svc.score(test_data, test_labels)

    #### Calculate additional metrics (MI)
    predicted_labels = trained_svc.predict(test_data)
    mi = skmetrics.mutual_info_score(test_labels, predicted_labels)
    ami = skmetrics.adjusted_mutual_info_score(test_labels, predicted_labels)
    log2_mi = computeMI(test_labels, predicted_labels)

    iterdict.update({'heldout_test_score': curr_test_score, 
                     'heldout_MI': mi, 'heldout_aMI': ami, 'heldout_log2MI': log2_mi,
                     'train_transform': '_'.join([str(s) for s in train_sizes]),
                     'test_transform': '_'.join([str(s) for s in test_sizes])})

    i_list.append(pd.DataFrame(iterdict, index=[i]))
    i+=1
    
    iterdf = pd.concat(i_list, axis=0).reset_index(drop=True)
    iterdf['iteration'] = [iter_num for _ in np.arange(0, len(iterdf))]
    
    return iterdf


def cycle_train_sets(iter_num, global_rois=None, MEANS=None, sdf=None, sample_ncells=None, n_train_configs=4,
                      cv=True, C_value=None, test_split=0.2, cv_nfolds=5, class_a=0, class_b=106):
    
    constant_transform = 'size'
    sizes = sorted(sdf[constant_transform].unique())
    
    training_sets = list(itertools.combinations(sizes, n_train_configs))

    i_list=[]
    for train_set in training_sets:
        test_set = [t for t in sizes if t not in train_set]
        tmpdf = do_fit_train_test_subset(iter_num, global_rois=global_rois, MEANS=MEANS, sdf=sdf, 
                                        sample_ncells=sample_ncells,
                                        train_sizes=train_set, test_sizes=test_set,
                                        cv=cv, C_value=C_value, test_split=test_split, 
                                        cv_nfolds=cv_nfolds, class_a=class_a, class_b=class_b)

        i_list.append(tmpdf)
    iterdf = pd.concat(i_list, axis=0).reset_index(drop=True)
    iterdf['iteration'] = [iter_num for _ in np.arange(0, len(iterdf))]
    
    return iterdf


def do_fit_train_test_morph(iter_num, global_rois=None, MEANS=None, sdf=None, sample_ncells=None,
                               cv=True, C_value=None, test_size=0.2, cv_nfolds=5, class_a=0, class_b=106):
    '''
    Resample w/ replacement from pooled cells (across datasets). Assumes 'sdf' is same for all datasets.
    Return fit results for 1 iteration.
    Classes (class_a, class_b) should be the actual labels of the target (i.e., value of morph level)
    '''
    # Get new sample set
    curr_data = get_trials_for_N_cells(sample_ncells, global_rois, MEANS)

    #### Select train/test configs for clf A vs B
    class_types = [class_a, class_b]
    restrict_transform = True
    class_name='morphlevel'
    constant_transform = 'size'
    sizes = sorted(sdf[constant_transform].unique())
    
    i_list=[]
    i=0
    # Go thru all training sizes, then test on non-trained sizes
    #for train_transform in sizes:
    
    # Get train configs -- ANCHORS (A/B)
    train_configs = sdf[sdf[class_name].isin(class_types)].index.tolist()

    #### TRAIN SET --------------------------------------------------------------------
    # Get trial data for selected cells and config types
    curr_roi_list = [int(c) for c in curr_data.columns if c != 'config']
    trainset = curr_data[curr_data['config'].isin(train_configs)].copy()
    train_data = trainset.drop('config', 1)#zdata = (data - data.mean()) / data.std()

    # Get labels
    targets = pd.DataFrame(trainset['config'].copy(), columns=['config'])
    targets['label'] = [sdf['morphlevel'][cfg] for cfg in targets['config'].values]
    targets['group'] = [sdf['size'][cfg] for cfg in targets['config'].values]

    #### TEST SET --------------------------------------------------------------------
    # Get data, specify configs
    novel_class_types = [c for c in sdf[class_name].unique() if c not in class_types]
    test_configs = sdf[sdf[class_name].isin(novel_class_types)].index.tolist()
    testset = curr_data[curr_data['config'].isin(test_configs)]
    test_data = testset.drop('config', 1) #zdata = (data - data.mean()) / data.std()

    # Get labels.
    test_targets = pd.DataFrame(testset['config'].copy(), columns=['config'])
    test_targets['label'] = [sdf['morphlevel'][cfg] for cfg in test_targets['config'].values]
    test_targets['group'] = [sdf['size'][cfg] for cfg in test_targets['config'].values]

    #### Train SVM ----------------------------------------------------------------------
    randi = random.randint(1, 10000)
    iterdict, trained_svc, trained_scaler, (predicted_labels, true_labels) = fit_svm(
                                                        train_data, targets, 
                                                        return_clf=True, return_predictions=True,
                                                        test_split=test_size, cv_nfolds=cv_nfolds, 
                                                        C_value=C_value, randi=randi)
    for anchor in [class_a, class_b]:
        a_ixs = np.array([i for i, v in enumerate(true_labels) if v==anchor])
        p_chooseB = sum([1 if p==class_b else 0 \
                            for p in predicted_labels[a_ixs]])/float(len(predicted_labels[a_ixs]))
        iterdict.update({'p_chooseB': p_chooseB, 'test_transform': anchor})
        i_list.append(pd.DataFrame(iterdict, index=[i]))
        i+=1

    #### Test SVM
    for test_transform, curr_test_group in test_targets.groupby(['label']):
        curr_test_data = test_data.loc[curr_test_group.index].copy()
        curr_test_data = trained_scaler.transform(curr_test_data)

        #### Calculate p choose B on trials where morph X shown (test_transform)
        predicted_labels = trained_svc.predict(curr_test_data)
        p_chooseB = sum([1 if p==class_b else 0 for p in predicted_labels])/float(len(predicted_labels))
 
        iterdict.update({'p_chooseB': p_chooseB, 'test_transform': test_transform}) 
        i_list.append(pd.DataFrame(iterdict, index=[i]))
        i+=1 
    iterdf = pd.concat(i_list, axis=0).reset_index(drop=True)
    iterdf['iteration'] = [iter_num for _ in np.arange(0, len(iterdf))]
    
    return iterdf



def do_fit_train_single_test_morph(iter_num, global_rois=None, MEANS=None, sdf=None, sample_ncells=None,
                               cv=True, C_value=None, test_size=0.2, cv_nfolds=5, class_a=0, class_b=106):
    '''
    Resample w/ replacement from pooled cells (across datasets). Assumes 'sdf' is same for all datasets.
    Return fit results for 1 iteration.
    Classes (class_a, class_b) should be the actual labels of the target (i.e., value of morph level)
    '''
    # Get new sample set
    curr_data = get_trials_for_N_cells(sample_ncells, global_rois, MEANS)

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
        curr_roi_list = [int(c) for c in curr_data.columns if c != 'config']
        trainset = curr_data[curr_data['config'].isin(train_configs)].copy()
        train_data = trainset.drop('config', 1)#zdata = (data - data.mean()) / data.std()

        # Get labels
        targets = pd.DataFrame(trainset['config'].copy(), columns=['config'])
        targets['label'] = [sdf['morphlevel'][cfg] for cfg in targets['config'].values]
        targets['group'] = [sdf['size'][cfg] for cfg in targets['config'].values]

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

        #### Train SVM ----------------------------------------------------------------------
        randi = random.randint(1, 10000)
        iterdict, trained_svc, trained_scaler, (predicted_labels, true_labels) = fit_svm(
                                                            train_data, targets, 
                                                            return_clf=True, return_predictions=True,
                                                            test_split=test_size, cv_nfolds=cv_nfolds, 
                                                            C_value=C_value, randi=randi)
        for anchor in [class_a, class_b]:
            a_ixs = np.array([i for i, v in enumerate(true_labels) if v==anchor])
            p_chooseB = sum([1 if p==class_b else 0 \
                                for p in predicted_labels[a_ixs]])/float(len(predicted_labels[a_ixs]))
            iterdict.update({'p_chooseB': p_chooseB, 
                             '%s' % class_name: anchor, 
                             '%s' % constant_transform: train_transform})
            i_list.append(pd.DataFrame(iterdict, index=[i]))
            i+=1

        #### Test SVM
        for test_transform, curr_test_group in test_targets.groupby(['label']):
            curr_test_data = test_data.loc[curr_test_group.index].copy()
            curr_test_data = trained_scaler.transform(curr_test_data)

            #### Calculate p choose B on trials where morph X shown (test_transform)
            predicted_labels = trained_svc.predict(curr_test_data)
            p_chooseB = sum([1 if p==class_b else 0 \
                                for p in predicted_labels])/float(len(predicted_labels))

            iterdict.update({'p_chooseB': p_chooseB, 
                             '%s' % class_name: test_transform,
                             '%s' % constant_transform: train_transform})
            
            i_list.append(pd.DataFrame(iterdict, index=[i]))
            i+=1 
            
    iterdf = pd.concat(i_list, axis=0).reset_index(drop=True)
    iterdf['iteration'] = [iter_num for _ in np.arange(0, len(iterdf))]
    
    return iterdf



# ======================================================================
# Performance plotting 
# ======================================================================

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


# 
