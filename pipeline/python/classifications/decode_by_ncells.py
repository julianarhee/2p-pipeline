#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on  Apr 24 17:16:28 2020

@author: julianarhee
"""
import matplotlib as mpl
mpl.use('agg')

import sys
import optparse
import os
import json
import glob
import copy
import copy
import itertools
import datetime
import time
import pprint 
pp = pprint.PrettyPrinter(indent=4)
import traceback
import math

import numpy as np
import pylab as pl
import seaborn as sns
import pandas as pd
import statsmodels as sm
import cPickle as pkl

from scipy import stats as spstats

from pipeline.python.classifications import experiment_classes as util
from pipeline.python.classifications import aggregate_data_stats as aggr
from pipeline.python.classifications import rf_utils as rfutils
from pipeline.python import utils as putils
from pipeline.python.retinotopy import segment_retinotopy as seg
#from pipeline.python.classifications import decode_utils as dc
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


from pipeline.python.classifications import decode_utils as decutils

import multiprocessing as mp
from functools import partial
from contextlib import contextmanager


import multiprocessing as mp
from functools import partial

def initializer(terminating_):
    # This places terminating in the global namespace of the worker subprocesses.
    # This allows the worker function to access `terminating` even though it is
    # not passed as an argument to the function.
    global terminating
    terminating = terminating_


def decode_from_fov(datakey, visual_area, neuraldf, sdf=None, #min_ncells=5,
                    C_value=None, experiment='blobs',
                    n_iterations=50, n_processes=2, results_id='fov_results',
                    class_a=0, class_b=0, do_shuffle=True,
                    rootdir='/n/coxfs01/2p-data', create_new=False, verbose=False,
                    test_type=None, n_train_configs=4): 
    '''
    Fit FOV n_iterations times (multiproc). Save all iterations in dataframe.
    '''
    # tmp save
    session, animalid, fovnum = putils.split_datakey_str(datakey)
    traceid_dir = glob.glob(os.path.join(rootdir, animalid, session, 'FOV%i_zoom2p0x' % fovnum,
                            'combined_%s_static' % experiment, 'traces', '%s*' % traceid))[0]
    curr_dst_dir = os.path.join(traceid_dir, 'decoding')
    if not os.path.exists(curr_dst_dir):
        os.makedirs(curr_dst_dir)
        print("... saving tmp results to:\n  %s" % curr_dst_dir)

    results_outfile = os.path.join(curr_dst_dir, '%s.pkl' % results_id)

    #### Get neural means
    print("... Starting decoding analysis")
    # zscore full
    neuraldf = aggr.zscore_neuraldf(neuraldf)
    n_cells = int(neuraldf.shape[1]-1) 
    print("... BY_FOV | [%s] %s, n=%i cells" % (visual_area, datakey, n_cells))

    # ------ STIMULUS INFO -----------------------------------------
    session, animalid, fov_ = datakey.split('_')
    fovnum = int(fov_[3:])
    if sdf is None:
        obj = util.Objects(animalid, session, 'FOV%i_zoom2p0x' %  fovnum, traceid=traceid)
        sdf = obj.get_stimuli()

    # Decodinng -----------------------------------------------------
    start_t = time.time()
    iter_results = decutils.pool_bootstrap(neuraldf, sdf, C_value=C_value, 
                                n_iterations=n_iterations, n_processes=n_processes, verbose=verbose,
                                class_a=class_a, class_b=class_b, do_shuffle=do_shuffle,
                                test_type=test_type, n_train_configs=n_train_configs) 
    end_t = time.time() - start_t
    print("--> Elapsed time: {0:.2f}sec".format(end_t))
    # DATA - get mean across items
    # iter_results = pd.concat(i_chunks, axis=0)
    iter_results['n_cells'] = n_cells 
    iter_results['visual_area'] = visual_area
    iter_results['datakey'] = datakey

    # Save all iterations
    #iter_results = pd.concat(iter_list, axis=0) 
    #metainfo = {'visual_area': visual_area, 'datakey': datakey,'n_cells': n_cells}
    #iter_results = putils.add_meta_to_df(iter_results, metainfo)
    with open(results_outfile, 'wb') as f:
        pkl.dump(iter_results, f, protocol=pkl.HIGHEST_PROTOCOL)

    print(iter_results.groupby(['condition']).mean())   
    print("@@@@@@@@@ done. %s|%s  @@@@@@@@@@" % (visual_area, datakey))
    print(results_outfile) 
    return


def decode_split_pupil(datakey, visual_area, cells, MEANS, pupildata,
                    results_id='splitpupil_results',
                    n_cuts=3, feature_name='pupil_fraction',
                    experiment='blobs', min_ncells=5, C_value=None, 
                    n_iterations=50, n_processes=2, 
                    class_a=0, class_b=106, do_shuffle=True,
                    rootdir='/n/coxfs01/2p-data', create_new=False, verbose=False):
    '''
    Decode within FOV, split trials into high/low arousal states. 
    Repeat n_iterations (mulitproc)
    '''
    session, animalid, fov_ = datakey.split('_')
    fovnum = int(fov_[3:])
    fov = 'FOV%i_zoom2p0x' % fovnum
    traceid_dir = glob.glob(os.path.join(rootdir, animalid, session, fov, 
                            'combined_%s*' % experiment, 'traces', '%s*' % traceid))[0]
    #### Set output dir for current fov
    curr_dst_dir = os.path.join(traceid_dir, 'decoding')
    if not os.path.exists(curr_dst_dir):
        os.makedirs(curr_dst_dir)
        print("... saving tmp results to:\n  %s" % curr_dst_dir)

    #### Create or load results
    results_outfile = os.path.join(curr_dst_dir, '%s.pkl' % results_id)
    if create_new is False: 
        try:
            with open(results_outfile, 'rb') as f:
                iter_results = pkl.load(f)
            if do_shuffle:
                assert 'condition' in iter_results.columns, "No shuffle, re-running now."
        except Exception as e:
            create_new=True 

    if create_new:    
        #### Get neural means
        print("... Stating decoding analysis")
        neuraldf = aggr.get_neuraldf_for_cells_in_area(cells, MEANS, 
                                                       datakey=datakey, visual_area=visual_area)
        if int(neuraldf.shape[1]-1)<min_ncells:
            return None
        n_cells = int(neuraldf.shape[1]-1) 
        print("... [%s] %s, n=%i cells" % (visual_area, datakey, n_cells))

        #### Get pupil means\
        pupildf = pupildata[datakey].copy()
        
        #### Match trial numbers
        neuraldf, pupildf = dlcutils.match_trials_df(neuraldf, pupildf, equalize_conditions=True)
       
        # zscore full
        #neuraldf = aggr.zscore_neuraldf(neuraldf)
 
        # ------ Split trials by quantiles ---------------------------------
        pupil_low, pupil_high = dlcutils.split_pupil_range(pupildf, 
                                                    feature_name=feature_name, n_cuts=n_cuts)
        #assert pupil_low.shape==pupil_high.shape, \
            # "Unequal pupil trials: %s, %s" % (str(pupil_low.shape), str(pupil_high.shape))
        print("SPLIT PUPIL: %s (low), %s (high)" % (str(pupil_low.shape), str(pupil_high.shape)))

        # trial indices of low/high pupil 
        low_trial_ixs = pupil_low['trial'].unique()
        high_trial_ixs = pupil_high['trial'].unique()
        all_trial_ixs = pupildf['trial'].unique()

        # ------ STIMULUS INFO -----------------------------------------
        obj = util.Objects(animalid, session, 'FOV%i_zoom2p0x' %  fovnum, traceid=traceid)
        sdf = obj.get_stimuli()

        # Decoding -----------------------------------------------------
        arousal_conds = ['all', 'low', 'high']
        arousal_trial_ixs = [all_trial_ixs, low_trial_ixs, high_trial_ixs]
        iter_list=[]
        for arousal_cond, curr_trial_ixs in zip(arousal_conds, arousal_trial_ixs):
            print(arousal_cond)
            # Get neuraldf for current trials
            curr_data = neuraldf.loc[curr_trial_ixs].copy()
            # Fit.
            start_t = time.time()
            arousal_df = decutils.fit_svm_mp(curr_data, sdf, C_value=C_value, 
                                    n_iterations=n_iterations, 
                                    n_processes=n_processes, verbose=verbose,
                                    class_a=class_a, class_b=class_b, do_shuffle=do_shuffle) 
            end_t = time.time() - start_t
            print("--> Elapsed time: {0:.2f}sec".format(end_t))

            arousal_df['n_trials'] = len(curr_trial_ixs)
            arousal_df['arousal'] = arousal_cond
            # Aggregate 
            iter_list.append(arousal_df)
       
        print("%i items in split-pupil list" % len(iter_list))
        # DATA - get mean across items
        iter_results = pd.concat(iter_list, axis=0)
        iter_results['visual_area'] = visual_area
        iter_results['datakey'] = datakey
        iter_results['n_cells'] = n_cells
        with open(results_outfile, 'wb') as f:
            pkl.dump(iter_results, f, protocol=pkl.HIGHEST_PROTOCOL)
        data_inputfile = os.path.join(curr_dst_dir, 'inputdata_%s.pkl' % results_id)
        inputdata = {'neuraldf': neuraldf, 'pupildf': pupildf, 'sdf': sdf,
                    'low_ixs': low_trial_ixs, 'high_ixs': high_trial_ixs}
        with open(data_inputfile, 'wb') as f:
            pkl.dump(inputdata, f, protocol=pkl.HIGHEST_PROTOCOL)

    #print(iter_results.columns)
    print(iter_results.groupby(['condition', 'arousal']).mean())
    print("@@@@@@@@@ done. %s|%s (%s)  @@@@@@@@@@" % (visual_area, datakey, results_id))

    return 


def decode_by_ncells(ncells, celldf, sdf, NEURALDATA, 
                    C_value=None, results_id='by_fov', 
                    n_iterations=50, n_processes=2, 
                    class_a=0, class_b=106,
                    dst_dir='/n/coxfs01/2p-data', create_new=False, verbose=False):

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
        print("... saving tmp results to:\n  %s" % dst_dir)

    results_outfile = os.path.join(dst_dir, '%s_%i.pkl' % (results_id, ncells))
    if create_new is False: 
        try:
            print("... loading: %s" % results_outfile)
            with open(results_outfile, 'rb') as f:
                iter_results = pkl.load(f)
        except Exception as e:
            print("... no results. creating new.")
            create_new=True 

    if create_new:    
        # Decoding -----------------------------------------------------
        start_t = time.time()
        iter_results = decutils.by_ncells_fit_svm_mp(ncells, celldf, NEURALDATA, sdf, 
                        n_iterations=n_iterations, n_processes=n_processes, 
                        C_value=C_value, cv_nfolds=cv_nfolds, test_split=test_split, 
                        class_a=class_a, class_b=class_b)
        end_t = time.time() - start_t
        print("--> Elapsed time: {0:.2f}sec".format(end_t))
        # DATA - get mean across items
        # iter_results = pd.concat(i_chunks, axis=0)
        iter_results['n_cells'] = ncells 
   
    # Save 
    with open(results_outfile, 'wb') as f:
        pkl.dump(iter_results, f, protocol=pkl.HIGHEST_PROTOCOL)

    print(iter_results.groupby(['condition']).mean())

    print("@@@@@@@@@ done. %s, n=%i @@@@@@@@@@" % (results_id, ncells))

    return 


def get_traceid_dir_from_datakey(datakey, traceid='traces001', rootdir='/n/coxfs01/2p-data'):
    session, animalid, fov_ = datakey.split('_')
    fovnum = int(fov_[3:])
    fov = 'FOV%i_zoom2p0x' % fovnum
    traceid_dir = glob.glob(os.path.join(rootdir, animalid, session, 
                            fov, 'combined_%s*' % experiment, 'traces', '%s*' % traceid))[0]
    return traceid_dir

def single_cell_dst_dir(traceid_dir, results_id):
    analysis_flag, rparams, tepoch, C_str = results_id.split('__')
    response_filter, rf_filter = rparams.split('_')
    response_type, response_test = response_filter.split('-')

    curr_dst_dir = os.path.join(traceid_dir, 'decoding', 'single_cells', '%s_%s' % (response_type, tepoch))
    if not os.path.exists(curr_dst_dir):
        os.makedirs(curr_dst_dir)
 
    varea = analysis_flag.split('_')[-1]
    # results_outfile = os.path.join(curr_dst_dir,'%s_%s__%03d.pkl' % (varea, C_str, int(rid+1)))
    save_prefix = '%s_%s' % (varea, C_str)

    return curr_dst_dir, save_prefix
 
     
def decode_from_cell(datakey, rid, neuraldf, sdf, results_outfile='/tmp/roi.pkl', 
                    do_shuffle=True, C_value=None, experiment='blobs',
                    n_iterations=50, n_processes=2, results_id='single_cell',
                    class_a=0, class_b=0, visual_area=None, verbose=False):
    # tmp save
   #results_id='%s_%s__%s-%s_%s__%s__%s' \
    #                % (prefix, visual_area, response_type, responsive_test, overlap_str, trial_epoch, C_str)

 
    #print("SINGLE_CELLS | (rid=%i, %s)" % (rid, results_id))

#    traceid_dir = get_traceid_dir_from_datakey(datakey, traceid=traceid) 
#    curr_dst_dir, save_prefix = single_cell_dst_dir(traceid_dir, results_id)    
#    results_outfile = os.path.join(curr_dst_dir, '%s__%03d.pkl' % (save_prefix, int(rid+1)))
#
#    if not os.path.exists(results_outfile):
#        create_new=True
#
#    if create_new and os.path.exists(results_outfile):
#        fname = os.path.split(results_outfile)[-1]
#        print("... deleting old files: %s" % fname)
#        os.remove(results_outfile)
#   
    #if create_new:    
    print("... starting analysis ")
    # zscore full
    #neuraldf = aggr.zscore_neuraldf(neuraldf)
    # Decodinng -----------------------------------------------------
    start_t = time.time()
    iter_results = decutils.fit_svm_mp(neuraldf, sdf, C_value=C_value, 
                                n_iterations=n_iterations, 
                                n_processes=n_processes, verbose=verbose,
                                class_a=class_a, class_b=class_b, do_shuffle=do_shuffle)
    end_t = time.time() - start_t
    print("--> Elapsed time: {0:.2f}sec".format(end_t))
 
    iter_results['cell'] = rid
    iter_results['datakey'] = datakey

    with open(results_outfile, 'wb') as f:
        pkl.dump(iter_results, f, protocol=pkl.HIGHEST_PROTOCOL)

    print("... saved: %s" % os.path.split(results_outfile)[-1])
    #print(iter_results.mean())

    print("Done!")

    return 


# --------------------------------------------------------------------
# Aggregating, Loading, saving, ec.
# -------------------------------------------------------------------
#def set_results_prefix(analysis_type='by_fov'):
#    prefix=None
#    if analysis_type=='by_fov':
#        prefix = 'fov_results'
#    elif analysis_type=='split_pupil':
#        prefix = 'splitpupil_results'
#    else:
#        prefix = analysis_type
#
#    return prefix
# 
def create_results_id(prefix='fov_results', visual_area='varea', C_value=None, 
                        trial_epoch='stimulus', has_retino=False, threshold_dff=False,
                        response_type='dff', responsive_test='resp', overlap_thr=None,
                        test_type=None): 

    C_str = 'tuneC' if C_value is None else 'C%.2f' % C_value
    if threshold_dff:
        overlap_str = 'threshdff'
    else:
        if has_retino:      
            overlap_str = 'retino'
        else:
            overlap_str = 'noRF' if overlap_thr is None else 'overlap%.1f' % overlap_thr
        #results_id='%s_%s_%s__%s-%s_%s__%s' % (prefix, visual_area, C_str, response_type, responsive_test, overlap_str, trial_epoch)
    results_id='%s_%s__%s-%s_%s__%s__%s' \
                    % (prefix, visual_area, response_type, responsive_test, overlap_str, trial_epoch, C_str)
   
    if test_type is not None:
        results_id = '%s__%s' % (results_id, test_type)
 
    return results_id

def create_results_id_aggr(prefix='fov_results', C_value=None, 
                        response_type='dff', trial_epoch='stimulus',
                        responsive_test='resp', overlap_thr=None, has_retino=False, threshold_dff=False,
                        test_type=None): 

    tmp_id = create_results_id(prefix=prefix, visual_area='NA',
                        C_value=C_value, response_type=response_type, trial_epoch=trial_epoch, 
                        responsive_test=responsive_test, overlap_thr=overlap_thr, has_retino=has_retino,
                        threshold_dff=threshold_dff, 
                        test_type=test_type) 
    param_str = tmp_id.split('_NA_')[-1]
    aggr_results_id = '%s__%s' % (prefix, param_str) 
    print("AGGREGATE: %s" % aggr_results_id)
    return aggr_results_id



def check_old_naming(animalid, session, fov, experiment='blobs', traceid='traces001',
                decode_type='single_cells', sub_dir='dff-nstds_stimulus', C_str='tuneC',
                rootdir='/n/coxfs01/2p-data'):
    
    res_files = glob.glob(os.path.join(rootdir, animalid, session, fov, 
                            'combined_%s_static' % experiment, 'traces', '%s*' % traceid, 
                            'decoding', decode_type, sub_dir, '*%s*.pkl' % C_str))
    for r in res_files:
        curr_dir, fname = os.path.split(r)
        if fname.startswith('single_cells_'):
            new_name = fname.split('single_cells_')[-1]
            os.rename(r, os.path.join(curr_dir, new_name))
    return

def load_fov_results(animalid, session, fov, traceid='traces001', 
                     visual_area=None, C_value=None, response_type='dff', 
                    responsive_test='nstds', trial_epoch='stimulus', 
                    overlap_thr=None, has_retino=False, threshold_dff=False,
                    test_type=None): 
    analysis_type='by_fov'
    # Get result ID
    results_id = create_results_id(prefix=analysis_type, visual_area=visual_area, 
                        C_value=C_value, response_type=response_type, responsive_test=responsive_test,
                        trial_epoch=trial_epoch, 
                        overlap_thr=overlap_thr, has_retino=has_retino, threshold_dff=threshold_dff,
                        test_type=test_type) 
    # Load FOV results
    iterdf = load_decode_within_fov(animalid, session, fov, traceid=traceid, results_id=results_id)
    
    return iterdf, results_id

def load_decode_within_fov(animalid, session, fov, results_id='fov_results',
                            traceid='traces001', 
                            rootdir='/n/coxfs01/2p-data', verbose=False):
    iter_df=None

    traceid_dir = glob.glob(os.path.join(rootdir, animalid, session, fov, 'combined_blobs*', 
                            'traces', '%s*' % traceid))[0]
    curr_dst_dir = os.path.join(traceid_dir, 'decoding')
    if not os.path.exists(curr_dst_dir):
        os.makedirs(curr_dst_dir)
        print("... saving tmp results to:\n  %s" % curr_dst_dir)

    if not os.path.exists(os.path.join(curr_dst_dir, '%s.pkl' % results_id)):
        if len(results_id.split('__'))==5:
            varea, rparams, tepoch, cstr, testinfo = results_id.split('__')
        else:
            varea, rparams, tepoch, cstr = results_id.split('__')
        if 'noRF' in rparams:
            rparams = rparams.replace('noRF', 'no-rfs')
        else:
            rparams = rparams.replace('overlap', 'overlap-')
        old_id = '%s_%s__%s__%s' % (varea, cstr, rparams, tepoch)
        old_outfile = os.path.join(curr_dst_dir, '%s.pkl' % old_id)
        if os.path.exists(old_outfile):
            print("... renaming (%s-->%s" % (old_id, results_id))
            os.rename(old_outfile, os.path.join(curr_dst_dir, '%s.pkl' % results_id))
    results_outfile = os.path.join(curr_dst_dir, '%s.pkl' % results_id)
       
    if verbose:
        print('%s|%s|%s -- %s' % (animalid, session, fov, results_id))

    try:
        with open(results_outfile, 'rb') as f:
            iter_df = pkl.load(f)
        if 'iteration' not in iter_df.columns:
            iter_df['iteration'] = iter_df.index.tolist()
        iter_df = iter_df.sort_values(by='iteration').reset_index(drop=True)
 
    except Exception as e:
        #print("Unable to find file: %s" % results_outfile)
        pass

    return iter_df

def aggregate_decode_within_fov(dsets, results_prefix='fov_results', 
                 C_value=None, response_type='dff', trial_epoch='stimulus',
                responsive_test='nstds', responsive_thr=10., overlap_thr=None, 
                has_retino=False, threshold_dff=False, 
                test_type=None,
                rootdir='/n/coxfs01/2p-data', verbose=False):
    no_results=[]
    found_results=[]
    #i=0
    popdf = []
    for (visual_area, datakey), g in dsets.groupby(['visual_area', 'datakey']): 
        results_id=create_results_id(prefix=results_prefix, 
                                visual_area=visual_area, C_value=C_value, has_retino=has_retino,
                                response_type=response_type, responsive_test=responsive_test,
                                overlap_thr=overlap_thr, trial_epoch=trial_epoch, threshold_dff=threshold_dff,
                                test_type=test_type)
        # Load dataset results
        session, animalid, fov_ = datakey.split('_')
        fovnum = int(fov_[3:])
        fov = 'FOV%i_zoom2p0x' % fovnum
        iter_df = load_decode_within_fov(animalid, session, fov, results_id=results_id,
                                                traceid=traceid, rootdir=rootdir, verbose=verbose)
       
        if iter_df is None:
            no_results.append((visual_area, datakey))
            continue
        else:
            found_results.append((visual_area, datakey)) 
#        # Pool mean
#        if 'fov' in results_prefix:
#            iterd = dict(iter_results.mean())
#            iterd.update( dict(('%s_std' % k, v) \
#                    for k, v in zip(iter_results.std().index, iter_results.std().values)) )
#            iterd.update( dict(('%s_sem' % k, v) \
#                    for k, v in zip(iter_results.sem().index, iter_results.sem().values)) )
#            iter_df = pd.DataFrame(iterd, index=[i])
#        else:
#            iter_df = iter_results.groupby(['arousal']).mean().reset_index()
#  
        metainfo = {'visual_area': visual_area, 'datakey': datakey} 
        iter_df = putils.add_meta_to_df(iter_df, metainfo)
        popdf.append(iter_df)
        #i += 1

    if len(no_results)>0:
        print("No results for %i dsets:" % len(no_results))
        if verbose:
            for d in no_results:
                print(d)
    else:
        print("Found results for %i dsets." % len(found_results))


    if len(popdf)==0:
        return None
    pooled = pd.concat(popdf, axis=0)

    return pooled


def do_decode_within_fov(analysis_type='by_fov', experiment='blobs', 
                        responsive_test='nstds', responsive_thr=10.,
                        response_type='dff', trial_epoch='stimulus', 
                        min_ncells=5, n_iterations=100, C_value=None,
                        match_distns=False, overlap_thr=0.5, has_retino=False, threshold_dff=False,
                        test_type=None, 
                        test_split=0.2, cv_nfolds=5, class_a=0, class_b=106, 
                        traceid='traces001', fov_type='zoom2p0x', state='awake',
                        aggregate_dir='/n/coxfs01/julianarhee/aggregate-visual-areas',
                        rootdir='/n/coxfs01/2p-data', verbose=False):

    #### Output dir
    stats_dir = os.path.join(aggregate_dir, 'data-stats')
    dst_dir = os.path.join(aggregate_dir, 'decoding', analysis_type)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    results_prefix = analysis_type #set_results_prefix(analysis_type=analysis_type)
    #aggr_results_id='%s__%s_%s-%s_%s' % (analysis_type, C_str, response_type, responsive_test, C_str)

    aggr_results_id = create_results_id_aggr(prefix=results_prefix,
                                 C_value=C_value, response_type=response_type, 
                                 trial_epoch=trial_epoch,  
                                 responsive_test=responsive_test, overlap_thr=overlap_thr, 
                                has_retino=has_retino, threshold_dff=threshold_dff,
                                test_type=test_type)
    # Get all data sets
    if (has_retino is False) and (overlap_thr is not None):
        edata = aggr.get_blobs_and_rf_meta(experiment=experiment, traceid=traceid, 
                                        stim_filterby=None)
    else:
        sdata = aggr.get_aggregate_info(traceid=traceid)
        edata = sdata[sdata['experiment']=='blobs'].copy()

    pooled = aggregate_decode_within_fov(edata, C_value=C_value, results_prefix=results_prefix,
                                response_type=response_type, responsive_test=responsive_test, 
                                trial_epoch=trial_epoch,
                                responsive_thr=responsive_thr, overlap_thr=overlap_thr, 
                                has_retino=has_retino, threshold_dff=threshold_dff,
                                rootdir=rootdir, verbose=verbose) 
    # Save data
    print("SAVING.....")  #datestr = datetime.datetime.now().strftime("%Y%m%d")

    # Save classifier results
    results_outfile = os.path.join(dst_dir, '%s.pkl' % (aggr_results_id))
    results = {'results': pooled}
    with open(results_outfile, 'wb') as f:
        pkl.dump(pooled, f, protocol=pkl.HIGHEST_PROTOCOL) 
    print("-- results: %s" % results_outfile)

    # Save params
    params_outfile = os.path.join(dst_dir, '%s_params.json' % (aggr_results_id))
    params = {'test_split': test_split, 
              'cv_nfolds': cv_nfolds, 
              'C_value': C_value,
              'n_iterations': n_iterations, 
              'overlap_thr': overlap_thr,
              'match_distns': match_distns,
              'class_a': class_a, 'class_b': class_b, 
              'response_type': response_type, 
              'responsive_test': responsive_test,
              'responsive_thr': responsive_thr, 
              'trial_epoch': trial_epoch}
    with open(params_outfile, 'w') as f:
        json.dump(params, f,  indent=4, sort_keys=True)
    print("-- params: %s" % params_outfile)

    print("DONE!")

    return pooled


def get_cells_and_data(all_cells, MEANS, traceid='traces001', response_type='dff', stack_neuraldf=True,
                       overlap_thr=None, has_retino=False, threshold_snr=False, snr_thr=10, max_snr_thr=None,
                      remove_too_few=False, min_ncells=5, match_distns=False, threshold_dff=False):
    
    has_rfs = (overlap_thr is not None) and (has_retino is False) and (threshold_dff is False)
    #### Get metadata for experiment type
    dsets, keys_by_area = aggr.experiment_datakeys(experiment=experiment, experiment_only=False,
                                    has_gratings=False, stim_filterby=None, has_rfs=has_rfs)
    
    #### Load RFs
    NEURALDATA=None; RFDATA=None;
    if has_rfs:
        print("~~~~~~~~~~~~~~~~Loading RFs~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        reliable_only=True
        rf_fit_desc = fitrf.get_fit_desc(response_type=response_type)
        reliable_str = 'reliable' if reliable_only else ''
        # Get position info for RFs
        rfdf = aggr.load_rfdf_and_pos(dsets, rf_filter_by=None, assign_cells=True,
                                    reliable_only=True, traceid=traceid)
        #rfdf_avg = aggr.get_rfdata(all_cells, rfdf, average_repeats=True)

        # RF dataframes
        NEURALDATA, RFDATA, assigned_cells = aggr.get_neuraldata_and_rfdata(all_cells, rfdf, MEANS,
                                                stack=stack_neuraldf)
    elif has_retino:
        print("~~~~~~~~~~~~~~~~Loading Retinotopy Bar ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        retino_mag_thr=0.01
        retino_pass_criterion='all'
        retino_cells = aggr.aggregate_responsive_retino(all_cells, traceid=traceid,
                                        mag_thr=retino_mag_thr, pass_criterion=retino_pass_criterion,
                                        verbose=False, create_new=True)
        NEURALDATA = aggr.get_neuraldata(retino_cells, MEANS, stack=stack_neuraldf) #True)
        assigned_cells = aggr.cells_in_experiment_df(retino_cells, NEURALDATA)
    else:
        print("~~~~~~~~~~~~~~~~No Receptive Fields~~~~~~~~~~~~~~~~~~~~~~~~~")
        NEURALDATA = aggr.get_neuraldata(all_cells, MEANS, stack=stack_neuraldf)
        assigned_cells = aggr.cells_in_experiment_df(all_cells, NEURALDATA)

    if match_distns:
        print("~~~~~~~~~~~~~~~~Matching max %s distNs~~~~~~~~~~~~~~~~~~~~~" % response_type)
        NEURALDATA, assigned_cells = aggr.match_neuraldata_distn(NEURALDATA, src='Li')
        if has_rfs:
            RFDATA = aggr.select_dataframe_subset(assigned_cells, RFDATA)

    if NEURALDATA is None: # or RFDATA is None:
        print("There is no data. Aborting.")
        return None, None

    if has_rfs:
        print("~~~~~~~~~~~~~~~~Calculating overlaps (thr=%.2f)~~~~~~~~~~~~~" % overlap_thr)
        # Calculate overlap with stimulus
        stim_overlaps = rfutils.calculate_overlaps(RFDATA, experiment=experiment)
        # Filter cells
        globalcells, cell_counts = aggr.get_pooled_cells(stim_overlaps, assigned_cells=assigned_cells,
                                            remove_too_few=remove_too_few, 
                                            overlap_thr=0 if overlap_thr is None else overlap_thr,
                                            min_ncells=min_ncells)
        #print(globalcells.head())
    else:
        globalcells, cell_counts = aggr.global_cells(assigned_cells,
                                            remove_too_few=remove_too_few,
                                            min_ncells=min_ncells, return_counts=True)
    if globalcells is None:
        print("NO CELLS. Exiting.")
        return None

    print("@@@@@@@@ cell counts @@@@@@@@@@@")
    print(cell_counts)

    # TMP TMP
    if threshold_snr:
        #snr_thr=10.0
        #max_snr_thr=None #15.0 #None #15.0 #None
        #match_str='snrlim_' if max_snr_thr is not None else 'snr_'
        print("~~~~~~~~~~~~~~~~SNR (thr=%.2f)~~~~~~~~~~~~~" % snr_thr)
        mean_snr = aggr.get_mean_snr(experiment=experiment, traceid=traceid, responsive_test=responsive_test,
                                        responsive_thr=responsive_thr, trial_epoch=trial_epoch)
        CELLS = aggr.threshold_cells_by_snr(mean_snr, globalcells, snr_thr=snr_thr, max_snr_thr=max_snr_thr)
        if has_rfs:
            NEURALDATA, RFDATA, CELLS = aggr.get_neuraldata_and_rfdata(CELLS, rfdf, MEANS,
                                                stack=stack_neuraldf)
        else:
            NEURALDATA = aggr.get_neuraldata(CELLS, MEANS, stack=stack_neuraldf, verbose=False)
    else:
        #### Get final cells dataframe
        CELLS = globalcells.copy()
        CELLS['cell'] = globalcells['dset_roi']

    print("------------------------------------")
    #print("Final cell counts:")
    #CELLS[['visual_area', 'datakey', 'cell']].drop_duplicates().groupby(['visual_area']).count()

    return NEURALDATA, CELLS.reset_index(drop=True)


def filter_cells_by_dff(all_cells, MEANS, traceid='traces001', response_type='dff', 
                       minv=0., maxv=1.0, aggregate_dir='/n/coxfs01/julianarhee/aggregate-visual-areas'):
   
    cells_fn = os.path.join(aggregate_dir, 'decoding', 'thr_dff_cells_m%.2f_M%.2f.pkl' % (minv, maxv))
    print(cells_fn)

    if os.path.exists(cells_fn):
        with open(cells_fn, 'rb') as f:
            thr_cells = pkl.load(f)

    else:
        ndata_df, cells_df = get_cells_and_data(all_cells, MEANS, traceid=traceid, response_type=response_type, 
                                    stack_neuraldf=True, overlap_thr=None, has_retino=False, threshold_snr=False) 
                                    
        meandf = ndata_df.groupby(['visual_area', 'datakey', 'cell', 'config']).mean().reset_index()
        means_c = meandf.groupby(['visual_area', 'datakey', 'cell']).mean().reset_index()

        thr_resp = means_c[(means_c['response']<=maxv) & (means_c['response']>=minv)].reset_index(drop=True)

        thr_cells = pd.concat([cells_df[(cells_df['visual_area']==v) & (cells_df['datakey']==d)
                      & (cells_df['cell'].isin(g['cell'].unique()))] \
                    for (v, d), g in thr_resp.groupby(['visual_area', 'datakey'])]).reset_index(drop=True)

        with open(cells_fn, 'wb') as f:
            pkl.dump(thr_cells, f, protocol=pkl.HIGHEST_PROTOCOL)
        
    return thr_cells


#%%

traceid = 'traces001'
fov_type = 'zoom2p0x'
state = 'awake'
aggregate_dir = '/n/coxfs01/julianarhee/aggregate-visual-areas'

response_type = 'dff'
responsive_test = 'nstds' # 'nstds' #'ROC' #None
responsive_thr = 10

# CV stuff
experiment = 'blobs'
m0=0
m100=106
n_iterations=100 
n_processes = 2
#print(m0, m100, '%i iters' % n_iterations)

test_split=0.2
cv_nfolds=5
C_value=None
cv=True

min_ncells = 20
overlap_thr=0.
filter_fovs = True
remove_too_few = False


def extract_options(options):
    
    parser = optparse.OptionParser()

    parser.add_option('-D', '--root', action='store', dest='rootdir', 
                      default='/n/coxfs01/2p-data',\
                      help='data root dir [default: /n/coxfs01/2pdata]')

    # Set specific session/run for current animal:
    parser.add_option('-E', '--experiment', action='store', dest='experiment', 
                        default='blobs', help="experiment type [default: blobs]")
    parser.add_option('-t', '--traceid', action='store', dest='traceid', default='traces001', \
                      help="name of traces ID [default: traces001]")
    parser.add_option('-d', '--response-type', action='store', dest='response_type', 
                        default='dff', help="response type [default: dff]")


      
    # data filtering 
    choices_e = ('stimulus', 'firsthalf', 'plushalf', 'baseline')
    default_e = 'stimulus'
    parser.add_option('--epoch', action='store', dest='trial_epoch', 
            default=default_e, type='choice', choices=choices_e,
            help="Trial epoch for input data, choices: %s. (default: %s" % (choices_e, default_e))

    choices_c = ('all', 'ROC', 'nstds', None, 'None')
    default_c = 'nstds'
    parser.add_option('-R', '--responsive_test', action='store', dest='responsive_test', 
            default=default_c, type='choice', choices=choices_c,
            help="Responsive test, choices: %s. (default: %s" % (choices_c, default_c))
    parser.add_option('-r', '--responsive-thr', action='store', dest='responsive_thr', 
                        default=10, help="response type [default: 10, nstds]")
 
    # classifier
    parser.add_option('-a', action='store', dest='class_a', 
            default=0, help="m0 (default: 0 morph)")
    parser.add_option('-b', action='store', dest='class_b', 
            default=106, help="m100 (default: 106 morph)")
    parser.add_option('-n', action='store', dest='n_processes', 
            default=1, help="N processes (default: 1)")
    parser.add_option('-N', action='store', dest='n_iterations', 
            default=100, help="N iterations (default: 100)")

    parser.add_option('-o', action='store', dest='overlap_thr', 
            default=0.5, help="% overlap between RF and stimulus (default: 0.5)")
    parser.add_option('--verbose', action='store_true', dest='verbose', 
            default=False, help="verbose printage")
    parser.add_option('--new', action='store_true', dest='create_new', 
            default=False, help="re-do decode")

    parser.add_option('-C','--cvalue', action='store', dest='C_value', 
            default=1.0, help="tune for C (default: 1)")


    choices_a = ('by_fov', 'split_pupil', 'by_ncells', 'single_cells')
    default_a = 'by_ncells'
    parser.add_option('-X','--analysis', action='store', dest='analysis_type', 
            default=default_a, type='choice', choices=choices_a,
            help="Analysis type, choices: %s. (default: %s)" % (choices_a, default_a))

    parser.add_option('-V','--visual-area', action='store', dest='visual_area', 
            default=None, help="(set for by_ncells) Must be None to run all serially")
    parser.add_option('-S','--ncells', action='store', dest='ncells', 
            default=None, help="Must be None to run all serially")
    parser.add_option('-k','--datakey', action='store', dest='datakey', 
            default=None, help="(set for single_cells) Must be None to run all serially")

    parser.add_option('--match-distns', action='store_true', dest='match_distns', 
            default=False, help="(set for by_ncells) Match distns of neuraldf to Li")

    parser.add_option('--shuffle', action='store_true', dest='do_shuffle', 
            default=False, help="included shuffled results")
    parser.add_option('--snr', action='store_true', dest='threshold_snr', 
            default=False, help="use min. snr to filter out cells")

    parser.add_option('--snr-min', action='store', dest='snr_thr', 
            default=10.0, help="Min cut-off if --snr (default: 10.)")
    parser.add_option('--snr-max', action='store', dest='max_snr_thr', 
            default=None, help="Max cut-off if --snr (default: None)")

    parser.add_option('--retino', action='store_true', dest='has_retino', 
            default=False, help="Use retino for filtering)")

    parser.add_option('--dff', action='store_true', dest='threshold_dff', 
            default=False, help="Threshold dff values")

#    parser.add_option('--single', action='store_true', dest='train_test_single', 
#            default=False, help="Set to split train/test by single transform")
#
    choices_t = (None, 'None', 'size_single', 'size_subset', 'morph', 'morph_single')
    default_t = None  
    parser.add_option('-T', '--test', action='store', dest='test_type', 
            default=default_t, type='choice', choices=choices_t,
            help="Test type, choices: %s. (default: %s)" % (choices_t, default_t))
    parser.add_option('--ntrain', action='store', dest='n_train_configs', 
            default=4, help="N training sizes to use (default: 4, test 1)")
 



    (options, args) = parser.parse_args(options)

    return options


def main(options):
    opts = extract_options(options)
    fov_type = 'zoom2p0x'
    state = 'awake'
    aggregate_dir = '/n/coxfs01/julianarhee/aggregate-visual-areas'
    rootdir = opts.rootdir
    create_new = opts.create_new
    verbose=opts.verbose

    # Pick dataset ------------------------------------ 
    traceid = opts.traceid #'traces001'

    response_type = opts.response_type #'dff'
    responsive_test = opts.responsive_test #'nstds' # 'nstds' #'ROC' #None
    if responsive_test=='None':
        responsive_test=None
    responsive_thr = float(opts.responsive_thr) if responsive_test is not None else 0.05 #10
    #responsive_thr = float(5) if responsive_test is not None else 0.05 #10
    if responsive_test == 'ROC':
        responsive_thr = 0.05

    # Classifier info ---------------------------------
    experiment = opts.experiment #'blobs'
    class_a=int(opts.class_a) #0
    class_b=int(opts.class_b) #106
    n_iterations=int(opts.n_iterations) #100 
    n_processes=int(opts.n_processes) #2

    analysis_type=opts.analysis_type
    do_shuffle=opts.do_shuffle

    # CV ----------------------------------------------
    test_split=0.2
    cv_nfolds=5
 
    C_value = None if opts.C_value in ['None', None] else float(opts.C_value)
    do_cv = C_value in ['None', None]
    print("Do CV -%s- (C=%s)" % (str(do_cv), str(C_value)))

    # Dataset filtering --------------------------------
    filter_fovs = True
    remove_too_few = False #True
    min_ncells = 5 #10 if remove_too_few else 0
    overlap_thr = None if opts.overlap_thr in ['None', None] else float(opts.overlap_thr)
    has_retino = opts.has_retino

    has_rfs = (overlap_thr is not None) and (has_retino is False)

    threshold_dff = opts.threshold_dff

    if threshold_dff:
        overlap_str = 'threshdff'
    else:
        if has_retino:
            overlap_str = 'retino'
            has_rfs = False
        else:
            overlap_str = 'noRF' if overlap_thr is None else 'overlap%.1f' % overlap_thr
       
    stim_filterby = None # 'first'
    has_gratings = experiment!='blobs'

    match_distns = opts.match_distns
    if analysis_type in ['single_cells', 'by_fov']:
        match_distns = False
    match_str = 'matchdistns_' if match_distns else ''
    print("INFO: %s|overlap=%s|match-distns? %s" % (analysis_type, overlap_str, str(match_distns)))
    threshold_snr = opts.threshold_snr
    snr_thr = float(opts.snr_thr)
    max_snr_thr = None if opts.max_snr_thr in ['None', None] else float(opts.max_snr_thr)
    if threshold_snr:
        match_str='snrlim_' if max_snr_thr is not None else 'snr_'

    print("INFO: %s|overlap=%s|match-distns? %s (%s)" % (analysis_type, overlap_str, str(match_distns), match_str))

    trial_epoch = opts.trial_epoch #'plushalf' # 'stimulus'

    # Generalization Test ------------------------------
    test_type = None if opts.test_type in ['None', None] else opts.test_type
    n_train_configs = int(opts.n_train_configs) 
    #train_test_single = opts.train_test_single

    # Pupil -------------------------------------------
    pupil_feature='pupil_fraction'
    pupil_alignment='trial'
    pupil_epoch='pre'
    pupil_snapshot=391800
    redo_pupil=False
    pupil_framerate=20.
    # -------------------------------------------------
    # Alignment 
    iti_pre=1.
    iti_post=1.
    stim_dur=1.
    # -------------------------------------------------

    # RF stuff 
    rf_filter_by=None
    reliable_only = True
    rf_fit_thr = 0.5

    # Retino stuf
    #has_retino=True
    retino_mag_thr = 0.01
    retino_pass_criterion='all'

    # Create data ID for labeling figures with data-types
    #### Responsive params
    n_stds = None if responsive_test=='ROC' else 2.5 #None
    response_str = '%s_%s-thr-%.2f' % (response_type, responsive_test, responsive_thr) 
    g_str = 'hasgratings' if has_gratings else 'blobsonly'
    filter_str = 'filter_%s_%s' % (stim_filterby, g_str)
    data_id = '|'.join([traceid, filter_str, response_str])
    print(data_id)


    #### Get metadata for experiment type
    sdata = aggr.get_aggregate_info(traceid=traceid, fov_type=fov_type, state=state)
    has_filter = 'retino' if has_retino else 'rfs'
    dsets, keys_by_area = aggr.experiment_datakeys(experiment=experiment, experiment_only=False,
                                      has_gratings=has_gratings, stim_filterby=stim_filterby, has_rfs=has_rfs)
     
    #### Check stimulus configs
    stim_datakeys = dsets['datakey'].unique()

    #### Source data
    visual_areas = ['V1', 'Lm', 'Li', 'Ll']
    curr_visual_area = None if opts.visual_area in ['None', None] else opts.visual_area
    curr_datakey = None if opts.datakey in ['None', None] else opts.datakey    

    
    diff_sdfs = ['20190327_JC073_fov1', '20190314_JC070_fov1'] # 20190426_JC078 (LM, backlight)
    if curr_datakey in diff_sdfs:
        images_only=False #True
    else:
        images_only = analysis_type=='by_ncells'
    # Notes:
    # images_only=True if by_ncells, since need to concatenate trials 
    # TODO:  Fix so that we can train on anchors only and/or subset of configs
    _, all_cells, MEANS, SDF = aggr.get_source_data(experiment, 
                        equalize_now=True, zscore_now=True,
                        response_type=response_type, responsive_test=responsive_test, 
                        responsive_thr=responsive_thr, trial_epoch=trial_epoch, #use_all=False,
                        visual_area=None, # if match_distns else curr_visual_area,
                        datakey=None, # if match_distns else curr_datakey,
                        check_configs=True, return_configs=True, return_missing=False,
                        images_only=images_only) #analysis_type=='by_ncells')
    all_cells = all_cells[all_cells['visual_area'].isin(visual_areas)].copy() #, 'Ll'])]
    stack_neuraldf = analysis_type in ['by_ncells'] #match_distns==True

    # THRESHOLD
    min_dff=0
    max_dff=1.0
    if threshold_dff:
        print("TMP: loading thresholded cells")
        all_cells = filter_cells_by_dff(all_cells, MEANS, traceid=traceid, response_type=response_type,
                       minv=min_dff, maxv=max_dff)

    NEURALDATA, CELLS = get_cells_and_data(all_cells, MEANS, traceid=traceid, response_type=response_type, 
                            stack_neuraldf=stack_neuraldf, overlap_thr=overlap_thr, has_retino=has_retino, 
                            threshold_snr=threshold_snr, snr_thr=snr_thr, max_snr_thr=max_snr_thr,
                            remove_too_few=remove_too_few, min_ncells=min_ncells, match_distns=match_distns)
    visual_areas=['V1', 'Lm', 'Li']
    CELLS = CELLS[CELLS['visual_area'].isin(visual_areas)]

    if NEURALDATA is None:
        print("***NO DATA. ABORTING***")
        return None

    if analysis_type=='by_ncells' and responsive_test=='ROC':
        incl_ddict = {'Li': ['20190315_JC070_fov1', '20190322_JC073_fov1',
                            '20190602_JC091_fov1', '20190609_JC099_fov1', '20190614_JC091_fov1',
                            '20191018_JC113_fov1', '20191111_JC120_fov1'],
                     'Lm': ['20190306_JC061_fov3', '20190322_JC073_fov1', '20190504_JC078_fov1', '20190512_JC083_fov1', '20190517_JC083_fov1', '20190618_JC097_fov1'], 
                     'V1': ['20190501_JC076_fov1', '20190504_JC078_fov1', '20190508_JC083_fov1', '20190512_JC083_fov1', '20191006_JC110_fov1']}

        tmpc = pd.concat([g for (v, d), g in CELLS.groupby(['visual_area', 'datakey']) if d in incl_ddict[v]])
        CELLS=tmpc.copy()
    print("------------------------------------")
    print("Final cell counts:")
    print(CELLS[['visual_area', 'datakey', 'cell']].drop_duplicates().groupby(['visual_area']).count())


    #### Get pupil responses
    if 'pupil' in analysis_type:
        print("~~~~~~~~~~~~~~~~Loading pupil dataframes (%s)~~~~~~~~~~~~~" % pupil_feature)
        pupildata = dlcutils.get_aggregate_pupildfs(experiment=experiment, 
                                    alignment_type=pupil_alignment, 
                                    feature_name=pupil_feature, trial_epoch=pupil_epoch,
                                    iti_pre=iti_pre, iti_post=iti_post, stim_dur=stim_dur,
                                    in_rate=pupil_framerate, out_rate=pupil_framerate,
                                    snapshot=pupil_snapshot, create_new=redo_pupil)

    #### Setup output dirs  
    results_prefix = analysis_type #set_results_prefix(analysis_type=analysis_type)
    if threshold_dff:
        overlap_str = 'threshdff'
    else:
        if has_retino:
            overlap_str = 'retino'
        else:
            overlap_str = 'noRF' if overlap_thr is None else 'overlap%.1f' % overlap_thr
        #data_info='%s%s_%s_%s_iter-%i' \
        #    % (match_str, response_type, responsive_test, overlap_str, n_iterations)

    print('... Classify %i v %i (C=%s)' % (m0, m100, str(C_value)))
    print('... N=%i iters (%i proc), %s' % (n_iterations, n_processes, overlap_str))
    print("... Shuffle? %s" % str(do_shuffle))

    if n_processes > n_iterations:
        n_processes = n_iterations

    # ---------------------------------------------------------------------   
    if (curr_visual_area is None) and (curr_datakey is not None):
        # Get the visual area (first one) for datakey
        assert curr_datakey in CELLS['datakey'].unique(), "No dkey: %s" % curr_datakey
        curr_visual_area = CELLS[CELLS['datakey']==curr_datakey]['visual_area'].unique()[0]

    if curr_datakey is not None:
        sdf = SDF[curr_datakey].copy()
        print(sdf.head())
    else:
        images_only=analysis_type=='by_ncells'
        sdf_master = aggr.get_master_sdf(images_only=images_only)
        sdf = sdf_master.copy() #SDF[SDF.keys()[-1]].copy()

    # ============================================================ 
    # PER FOV analysis - for each fov, do something. 
    # ============================================================ 
    # datakey is not None and visual_area is not None
    if (curr_visual_area is not None) and (curr_datakey is not None):
        gdf = CELLS[(CELLS['visual_area']==curr_visual_area) 
                        & (CELLS['datakey']==curr_datakey)]
        ncells_total = gdf.shape[0] 
        if gdf.shape[0]==0:
            print("... skipping %s (%s) - no cells." % (curr_datakey, curr_visual_area))
            return None
        results_id = create_results_id(prefix=results_prefix, 
                            visual_area=curr_visual_area, C_value=C_value, has_retino=has_retino,
                            response_type=response_type, responsive_test=responsive_test,
                            overlap_thr=overlap_thr, trial_epoch=trial_epoch, threshold_dff=threshold_dff,
                            test_type=test_type) 
        print("~~~~~~~~~~~~~~~~ RESULTS ID ~~~~~~~~~~~~~~~~~~~~~")
        print(results_id)
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        if analysis_type=='by_fov':
            # -----------------------------------------------------------------------
            # BY_FOV - for each fov, do_decode
            # -----------------------------------------------------------------------
            neuraldf = aggr.get_neuraldf_for_cells_in_area(CELLS, MEANS, 
                                                       datakey=curr_datakey, visual_area=curr_visual_area)
            if int(neuraldf.shape[1]-1) > 0:# min_ncells: 
                decode_from_fov(curr_datakey, curr_visual_area, neuraldf, sdf, C_value=C_value,
                        n_iterations=n_iterations, n_processes=n_processes, 
                        results_id=results_id,
                        class_a=class_a, class_b=class_b, do_shuffle=do_shuffle,
                        rootdir=rootdir, create_new=create_new, verbose=verbose,
                        test_type=test_type, n_train_configs=n_train_configs) 
            print("--- done by_fov ---")

        elif analysis_type=='split_pupil': 
            # -----------------------------------------------------------------------
            # SPLIT_PUPIL - for each fov, split trials into arousal states, decode.
            # -----------------------------------------------------------------------
            if curr_datakey not in pupildata.keys():
                print("ERR: PUPILDATA, %s key not found in dict." % curr_datakey)
                return None

            decode_split_pupil(curr_datakey, curr_visual_area, CELLS, MEANS, pupildata,
                            n_cuts=3, feature_name=pupil_feature,
                            min_ncells=min_ncells, C_value=C_value,
                            n_iterations=n_iterations, n_processes=n_processes, 
                            results_id=results_id,
                            class_a=class_a, class_b=class_b, do_shuffle=do_shuffle,
                            rootdir=rootdir, create_new=create_new, verbose=verbose)

        elif analysis_type=='single_cells':
            # -----------------------------------------------------------------------
            # SINGLE_CELLS - for each fov, for each cell, do decode
            # -----------------------------------------------------------------------
            traceid_dir = get_traceid_dir_from_datakey(curr_datakey, traceid=traceid) 
            curr_dst_dir, save_prefix = single_cell_dst_dir(traceid_dir, results_id)    
            # old_files = glob.glob(os.path.join(curr_dst_dir, '%s_*.pkl' % save_prefix))

            all_rois = gdf['dset_roi'].unique()
            all_fnames = ['%s__%03d.pkl' % (save_prefix, int(rid+1)) for rid in all_rois]
            old_files = glob.glob(os.path.join(curr_dst_dir, '%s__*.pkl' % save_prefix)) # fn) for fn in all_fnames]
            #old_files = [os.path.join(curr_dst_dir, fn) for fn in all_fnames] 
            if create_new: # remove all old files
                print("... deleting %i old files" % len(old_files))
                for f in old_files:
                    os.remove(f)
            else:
                todo_fnames = [fname for fname in all_fnames if \
                                    os.path.join(curr_dst_dir, fname) not in old_files]
                print("SINGLE_CELLS: Need to run %i out of %i cells." % (len(todo_fnames), len(all_fnames)))
                #todo_rois = [int(os.path.splitext(fn)[0].split('__')[-1]) for fn in todo_fnames]

            nt = len(todo_fnames)
            for ri, fname in enumerate(todo_fnames): #enumerate(todo_rois): #enumerate(gdf['dset_roi'].values):
                rid = int(os.path.splitext(fname)[0].split('__')[-1]) - 1
                if ri % 10 == 0:
                    print("%i of %i cells (rid %i, %s|%s)" % (int(ri+1), nt, rid, curr_datakey, curr_visual_area))
                results_outfile = os.path.join(curr_dst_dir, fname)
                neuraldf = NEURALDATA[curr_visual_area][curr_datakey][[rid, 'config']].copy()
                #print(sorted(neuraldf['config'].unique()))
                decode_from_cell(curr_datakey, rid, neuraldf, sdf, experiment=experiment,
                                C_value=C_value, results_outfile=results_outfile,
                                n_iterations=n_iterations, n_processes=n_processes, 
                                results_id=results_id, visual_area=curr_visual_area,
                                class_a=class_a, class_b=class_b, verbose=verbose, do_shuffle=do_shuffle)          
            print("Finished %s (%s). ID=%s" % (curr_datakey, curr_visual_area, results_id))

    elif analysis_type=='by_ncells':
        data_info='%s%s-%s_%s_iter%i' % (match_str, response_type, responsive_test, overlap_str, n_iterations)

        # Create aggregate output dir
        dst_dir = os.path.join(aggregate_dir, 'decoding', analysis_type, data_info)
        try:
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)
                print("...making dir")
        except OSError:
            print("dir exists")
        print("DST: %s" % dst_dir)
        # ============================================================ 
        # by NCELLS - pools across datasets
        # ============================================================ 
        curr_visual_area = opts.visual_area
        curr_ncells = None if opts.ncells in ['None', None] else int(opts.ncells) 

        # Save inputs
        inputs_file = os.path.join(dst_dir, 'cells_%s_%s.pkl' % (analysis_type, str(curr_visual_area)))
        with open(inputs_file, 'wb') as f: pkl.dump(CELLS, f, protocol=pkl.HIGHEST_PROTOCOL) 

        if curr_visual_area is not None:
            gdf = CELLS[CELLS['visual_area']==curr_visual_area]
            results_id = create_results_id(prefix=results_prefix, 
                            visual_area=curr_visual_area, C_value=C_value, has_retino=has_retino, 
                            response_type=response_type, responsive_test=responsive_test,
                            overlap_thr=overlap_thr, trial_epoch=trial_epoch, threshold_dff=threshold_dff,
                            test_type=test_type)

            if curr_ncells is not None:
                # ----------------------------------------------
                # Do decode w CURR_NCELLS, CURR_VISUAL_AREA
                # ----------------------------------------------
                print("**** %s (n=%i cells)****" % (curr_visual_area, curr_ncells))
                decode_by_ncells(curr_ncells, gdf, sdf, NEURALDATA, 
                                C_value=C_value,
                                n_iterations=n_iterations, n_processes=n_processes, 
                                results_id=results_id,
                                class_a=class_a, class_b=class_b,
                                dst_dir=dst_dir, create_new=create_new, verbose=verbose)
                print("***** finished %s, ncells=%i *******" % (curr_visual_area, curr_ncells))
            else:
                # ----------------------------------------------
                # Loop thru NCELLS
                # ----------------------------------------------
                min_cells_total = min(cell_counts.values())
                reasonable_range = [2**i for i in np.arange(0, 10)]
                incl_range = [i for i in reasonable_range if i<min_cells_total]
                incl_range.append(min_cells_total)
                NCELLS = incl_range
                    
                for curr_ncells in NCELLS:
                    print("**** %s (n=%i cells)****" % (curr_visual_area, curr_ncells))

                    decode_by_ncells(curr_ncells, gdf, sdf, NEURALDATA, 
                                    C_value=C_value,
                                    n_iterations=n_iterations, n_processes=n_processes, 
                                    results_id=results_id,
                                    class_a=class_a, class_b=class_b,
                                    dst_dir=dst_dir, create_new=create_new, 
                                    verbose=verbose)
                print("********* finished %s, (ncells looped: %s) **********" % (curr_visual_area, str(NCELLS)))
        else:
            # ----------------------------------------------
            # Loop thru all visual areas, all NCELLS
            # ----------------------------------------------
            #### Get NCELLS
            min_cells_total = min(cell_counts.values())
            reasonable_range = [2**i for i in np.arange(0, 10)]
            incl_range = [i for i in reasonable_range if i<min_cells_total]
            incl_range.append(min_cells_total)
            NCELLS = incl_range

            try:
                for curr_visual_area, gdf in CELLS.groupby(['visual_area']):    
                    results_id = create_results_id(prefix=results_prefix,
                                        visual_area=curr_visual_area, 
                                        C_value=C_value, 
                                        response_type=response_type, 
                                        responsive_test=responsive_test, 
                                        overlap_thr=overlap_thr, has_retino=has_retino,
                                        trial_epoch=trial_epoch, threshold_dff=threshold_dff,
                                        test_type=test_type) 
                        
                    for curr_ncells in NCELLS:
                        print("**** %s (n=%i cells)****" % (curr_visual_area, curr_ncells))

                        decode_by_ncells(curr_ncells, gdf, sdf, NEURALDATA, 
                                        C_value=C_value,
                                        n_iterations=n_iterations, n_processes=n_processes, 
                                        results_id=results_id,
                                        class_a=class_a, class_b=class_b,
                                        dst_dir=dst_dir, create_new=create_new, 
                                        verbose=verbose)
                    print("********* finished **********")
            except Exception as e:
                traceback.print_exc()
         
    return 


if __name__ == '__main__':
    main(sys.argv[1:])
