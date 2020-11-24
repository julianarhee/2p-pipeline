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
from pipeline.python.classifications import decode_utils as dc
from pipeline.python.classifications import dlc_utils as dlcutils

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


from pipeline.python.classifications import decode_utils as dutils

import multiprocessing as mp
from functools import partial
from contextlib import contextmanager


# --------------------------------------------------------------------------
# within FOV
# -------------------------------------------------------------------------

def decode_from_fov(datakey, visual_area, cells, MEANS, min_ncells=5,
                    C_value=None, experiment='blobs',
                    n_iterations=50, n_processes=2, results_id='fov_results',
                    class_a=0, class_b=0,
                    rootdir='/n/coxfs01/2p-data', create_new=False, verbose=False):
    '''
    Fit FOV n_iterations times (multiproc). Save all iterations in dataframe.
    '''
    # tmp save
    session, animalid, fov_ = datakey.split('_')
    fovnum = int(fov_[3:])
    fov = 'FOV%i_zoom2p0x' % fovnum
    traceid_dir = glob.glob(os.path.join(rootdir, animalid, session, 
                            fov, 'combined_%s_static' % experiment, 'traces', '%s*' % traceid))[0]
    curr_dst_dir = os.path.join(traceid_dir, 'decoding')
    if not os.path.exists(curr_dst_dir):
        os.makedirs(curr_dst_dir)
        print("... saving tmp results to:\n  %s" % curr_dst_dir)

    results_outfile = os.path.join(curr_dst_dir, '%s.pkl' % results_id)
    if create_new is False: 
        try:
            with open(results_outfile, 'rb') as f:
                iter_results = pkl.load(f)
        except Exception as e:
            create_new=True 

    if create_new:    
        #### Get neural means
        print("... Stating decoding analysis")
        neuraldf = aggr.get_neuraldf_for_cells_in_area(cells, MEANS, 
                                                       datakey=datakey, visual_area=visual_area)
        if int(neuraldf.shape[1]-1)<min_ncells:
            return None

        # zscore full
        neuraldf = aggr.zscore_neuraldf(neuraldf)
        n_cells = int(neuraldf.shape[1]-1) 
        print("... [%s] %s, n=%i cells" % (visual_area, datakey, n_cells))

        # ------ STIMULUS INFO -----------------------------------------
        session, animalid, fov_ = datakey.split('_')
        fovnum = int(fov_[3:])
        obj = util.Objects(animalid, session, 'FOV%i_zoom2p0x' %  fovnum, traceid=traceid)
        sdf = obj.get_stimuli()

        # Decodinng -----------------------------------------------------
        iter_list = fit_svm_mp(neuraldf, sdf, C_value=C_value, 
                                    n_iterations=n_iterations, 
                                    n_processes=n_processes, verbose=verbose,
                                    class_a=class_a, class_b=class_b)
        print("%i items in mp list" % len(iter_list))
        # Save all iterations
        iter_results = pd.concat(iter_list, axis=0) 
        metainfo = {'visual_area': visual_area, 'datakey': datakey,'n_cells': n_cells}
        iter_results = putils.add_meta_to_df(iter_results, metainfo)

        with open(results_outfile, 'wb') as f:
            pkl.dump(iter_results, f, protocol=pkl.HIGHEST_PROTOCOL)

    # Pool mean
    print("... finished all iters: %s" % str(iter_results.shape))
    iterd = dict(iter_results.mean())
    iterd.update( dict(('%s_std' % k, v) \
            for k, v in zip(iter_results.std().index, iter_results.std().values)) )
    iterd.update( dict(('%s_sem' % k, v) \
            for k, v in zip(iter_results.sem().index, iter_results.sem().values)) )
    iterd.update({'n_units': n_cells, 
                  'visual_area': visual_area, 'datakey': datakey})
    print("::FINAL::")
    pp.pprint(iterd)
    print("@@@@@@@@@ done. %s|%s (by_fov)  @@@@@@@@@@" % (visual_area, datakey))

    return iterd


def decode_split_pupil(datakey, visual_area, cells, MEANS, pupildata,
                    results_id='splitpupil_results',
                    n_cuts=3, feature_name='pupil_fraction',
                    experiment='blobs', min_ncells=5, C_value=None, 
                    n_iterations=50, n_processes=2, 
                    class_a=0, class_b=106,
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
        neuraldf = aggr.zscore_neuraldf(neuraldf)
 
        # ------ Split trials by quantiles ---------------------------------
        pupil_low, pupil_high = dlcutils.split_pupil_range(pupildf, 
                                                    feature_name=feature_name, n_cuts=n_cuts)
        assert pupil_low.shape==pupil_high.shape, "Unequal pupil trials: %s, %s" % (str(pupil_low.shape), str(pupil_high.shape))
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
            # Get neuraldf for current trials
            curr_data = neuraldf.loc[curr_trial_ixs].copy()
            # Fit.
            a_list = fit_svm_mp(curr_data, sdf, C_value=C_value, 
                                    n_iterations=n_iterations, 
                                    n_processes=n_processes, verbose=verbose,
                                    class_a=class_a, class_b=class_b) 
            print("%i items in mp list" % len(a_list))
            # Aggregate 
            arousal_df = pd.concat(a_list, axis=0)
            metainfo = {'visual_area': visual_area, 'datakey': datakey,
                        'arousal': arousal_cond, 'n_cells': n_cells}
            arousal_df = putils.add_meta_to_df(arousal_df, metainfo)
            iter_list.append(arousal_df)
       
        print("%i items in split-pupil list" % len(iter_list))
        # DATA - get mean across items
        iter_results = pd.concat(iter_list, axis=0)
        with open(results_outfile, 'wb') as f:
            pkl.dump(iter_results, f, protocol=pkl.HIGHEST_PROTOCOL)

    # Pool mean
    #print(iter_results)
    print("... finished all iters: %s" % str(iter_results.shape))
    iterd = dict(iter_results.groupby(['arousal']).mean())
#    iterd.update( dict(('%s_std' % k, v) \
#            for k, v in zip(iter_results.std().index, iter_results.std().values)) )
#    iterd.update( dict(('%s_sem' % k, v) \
#            for k, v in zip(iter_results.sem().index, iter_results.sem().values)) )
#    iterd.update({'n_units': n_cells, 
#                  'visual_area': visual_area, 'datakey': datakey})
    print("::FINAL::")
    pp.pprint(iterd)
    print("@@@@@@@@@ done. %s|%s (split_pupil)  @@@@@@@@@@" % (visual_area, datakey))

    return iterd

# --------------------------------------------------------------------
# Loading, saving, ec.
# -------------------------------------------------------------------
def set_results_prefix(analysis_type='by_fov'):
    prefix=None
    if analysis_type=='by_fov':
        prefix='fov_results'
    elif analysis_type=='split_pupil':
        prefix='splitpupil_results'
    else:
        print("UNKNOWN: %s" % analysis_type)

    return prefix
 

def create_results_id(prefix='fov_results', visual_area='varea', C_value=None, 
                        response_type='dff', responsive_test='resp'):
    C_str = 'tuneC' if C_value is None else 'C-%.2f' % C_value
    results_id='%s_%s_%s__%s-%s' % (prefix, visual_area, C_str, response_type, responsive_test)
    return results_id

def load_decode_within_fov(animalid, session, fov, results_id='fov_results',
                            traceid='traces001', 
                            rootdir='/n/coxfs01/2p-data'):
    iter_results=None

    traceid_dir = glob.glob(os.path.join(rootdir, animalid, session, fov, 'combined_blobs*', 
                            'traces', '%s*' % traceid))[0]
    curr_dst_dir = os.path.join(traceid_dir, 'decoding')
    if not os.path.exists(curr_dst_dir):
        os.makedirs(curr_dst_dir)
        print("... saving tmp results to:\n  %s" % curr_dst_dir)

    results_outfile = os.path.join(curr_dst_dir, '%s.pkl' % results_id)
    try:
        with open(results_outfile, 'rb') as f:
            iter_results = pkl.load(f)
    except Exception as e:
        #print("Unable to find file: %s" % results_outfile)
        pass

    return iter_results


def aggregate_decode_within_fov(dsets, results_prefix='fov_results', 
                 C_value=None, response_type='dff', 
                responsive_test='nstds', responsive_thr=10.,
                rootdir='/n/coxfs01/2p-data'):

    no_results=[]
    i=0
    popdf = []
    for (visual_area, datakey), g in dsets.groupby(['visual_area', 'datakey']): 
        #print("[%s]: %s" % (visual_area, datakey))
    
        results_id=create_results_id(prefix=results_prefix, 
                                    visual_area=visual_area, C_value=C_value, 
                                    response_type=response_type, responsive_test=responsive_test)

        # Load dataset results
        session, animalid, fov_ = datakey.split('_')
        fovnum = int(fov_[3:])
        fov = 'FOV%i_zoom2p0x' % fovnum
        iter_results = load_decode_within_fov(animalid, session, fov, results_id=results_id,
                                                traceid=traceid, rootdir=rootdir)
       
        if iter_results is None:
            no_results.append((visual_area, datakey))
            continue
 
        # Pool mean
        #print("... all iters: %s" % str(iter_results.shape))
        if 'fov' in results_prefix:
            iterd = dict(iter_results.mean())
            iterd.update( dict(('%s_std' % k, v) \
                    for k, v in zip(iter_results.std().index, iter_results.std().values)) )
            iterd.update( dict(('%s_sem' % k, v) \
                    for k, v in zip(iter_results.sem().index, iter_results.sem().values)) )
            iter_df = pd.DataFrame(iterd, index=[i])
        else:
            iter_df = iter_results.groupby(['arousal']).mean().reset_index()
     
        metainfo = {'visual_area': visual_area, 'datakey': datakey} 
        iter_df = putils.add_meta_to_df(iter_df, metainfo)
    
        popdf.append(iter_df)
        i += 1
    pooled = pd.concat(popdf, axis=0)

    if len(no_results)>0:
        print("No results for %i dsets:" % len(no_results))
        for d in no_results:
            print(d)

    return pooled


def do_decode_within_fov(analysis_type='by_fov', experiment='blobs', 
                        responsive_test='nstds', responsive_thr=10.,
                        response_type='dff', trial_epoch='stimulus', 
                        min_ncells=5, n_iterations=100, C_value=None,
                        test_split=0.2, cv_nfolds=5, class_a=0, class_b=106, 
                        traceid='traces001', fov_type='zoom2p0x', state='awake',
                        aggregate_dir='/n/coxfs01/julianarhee/aggregate-visual-areas',
                        rootdir='/n/coxfs01/2p-data'):

    #### Output dir
    stats_dir = os.path.join(aggregate_dir, 'data-stats')
    dst_dir = os.path.join(aggregate_dir, 'decoding', 'by_fov')
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    C_str = 'tuneC' if C_value is None else 'C-%.2f' % C_value

    results_prefix = set_results_prefix(analysis_type=analysis_type)
    aggr_results_id='%s__%s_%s-%s' % (analysis_type, C_str, response_type, responsive_test)


    #dst_dir = os.path.join(aggregate_dir, 'decoding')
    MEANS = aggr.load_aggregate_data(experiment, 
                responsive_test=responsive_test, responsive_thr=responsive_thr, 
                response_type=response_type, epoch=trial_epoch)
    #MEANS = aggr.equal_counts_per_condition(MEANS)

    # Get all data sets
    sdata = aggr.get_aggregate_info(traceid=traceid, fov_type=fov_type, state=state)
    edata = sdata[sdata['experiment']==experiment]
    rois = seg.get_cells_by_area(edata)
    cells = aggr.get_active_cells_in_current_datasets(rois, MEANS, verbose=False)
    seg_dkeys = cells['datakey'].unique()
    dsets = sdata[(sdata['datakey'].isin(seg_dkeys)) & (sdata['experiment']==experiment)]
 
    pooled = aggregate_decode_within_fov(dsets, C_value=C_value, results_prefix=results_prefix,
                                response_type=response_type, responsive_test=responsive_test, 
                                responsive_thr=responsive_thr,
                                rootdir=rootdir) 
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
    params = {'test_split': test_split, 'cv_nfolds': cv_nfolds, 'C_value': C_value,
              'n_iterations': n_iterations, #'overlap_thr': overlap_thr,
              'class_a': class_a, 'class_b': class_b, 
              'response_type': response_type, 'responsive_test': responsive_test,
              'responsive_thr': responsive_thr, 'trial_epoch': trial_epoch}
    with open(params_outfile, 'w') as f:
        json.dump(params, f,  indent=4, sort_keys=True)
    print("-- params: %s" % params_outfile)
       
    # Plot
    #plot_str = '%s' % (train_str)
    #dutils.default_classifier_by_ncells(pooled, plot_str=plot_str, dst_dir=dst_dir, 
    #                    data_id=data_id, area_colors=area_colors, datestr=datestr)
    print("DONE!")

    return pooled



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

    parser.add_option('-i', '--animalid', action='store', dest='animalid', 
                        default='', help="animalid")
    parser.add_option('-A', '--fov', action='store', dest='fov', 
                        default='FOV1_zoom2p0x', help="fov (default: FOV1_zoom2p0x)")
    parser.add_option('-S', '--session', action='store', dest='session', 
                        default='', help="session (YYYYMMDD)")
 
       
    # data filtering 
    choices_c = ('all', 'roc', 'nstds')
    default_c = 'nstds'
    parser.add_option('-R', '--responsive_test', action='store', dest='responsive_test', 
            default=default_c, type='choice', choices=choices_c,
            help="Responsive test, choices: %s. (default: %s" % (choices_c, default_c))
    parser.add_option('-r', '--responsive-thr', action='store', dest='responsive_thr', 
                        default=10, help="response type [default: 10, nstds]")
 
    # plotting
    parser.add_option('-a', action='store', dest='class_a', 
            default=0, help="m0 (default: 0 morph)")
    parser.add_option('-b', action='store', dest='class_b', 
            default=106, help="m100 (default: 106 morph)")
    parser.add_option('-n', action='store', dest='n_processes', 
            default=1, help="N processes (default: 1)")
    parser.add_option('-N', action='store', dest='n_iterations', 
            default=100, help="N iterations (default: 100)")

    parser.add_option('-o', action='store', dest='overlap_thr', 
            default=0.8, help="% overlap between RF and stimulus (default: 0.8)")
    parser.add_option('-V', action='store_true', dest='verbose', 
            default=False, help="verbose printage")
    parser.add_option('--new', action='store_true', dest='create_new', 
            default=False, help="re-do decode")

    parser.add_option('--cv', action='store_true', dest='do_cv', 
            default=False, help="tune for C")
    parser.add_option('-C','--cvalue', action='store', dest='C_value', 
            default=1.0, help="tune for C (default: 1)")


    choices_a = ('by_fov', 'split_pupil', '')
    default_a = 'by_fov'
    parser.add_option('-X','--analysis', action='store', dest='analysis_type', 
            default=default_a, type='choice', choices=choices_a,
            help="Analysis type, choices: %s. (default: %s" % (choices_a, default_a))

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
    animalid = opts.animalid
    session = opts.session
    fov = opts.fov
    traceid = opts.traceid #'traces001'

    response_type = opts.response_type #'dff'
    responsive_test = opts.responsive_test #'nstds' # 'nstds' #'ROC' #None
    responsive_thr = float(opts.responsive_thr) #10

    # Classifier info ---------------------------------
    experiment = opts.experiment #'blobs'
    class_a=int(opts.class_a) #0
    class_b=int(opts.class_b) #106
    n_iterations=int(opts.n_iterations) #100 
    n_processes=int(opts.n_processes) #2
    trial_epoch = 'stimulus'

    analysis_type=opts.analysis_type

    # CV ----------------------------------------------
    test_split=0.2
    cv_nfolds=5
    C_value=opts.C_value
    do_cv = opts.do_cv
    C_value = opts.C_value
 
    do_cv = C_value in ['None', None]
    C_value = None if do_cv else float(opts.C_value)
    print("Do CV -%s- (C=%s)" % (str(do_cv), str(C_value)))

    # Dataset filtering --------------------------------
    filter_fovs = True
    remove_too_few = False
    min_ncells = 20 if remove_too_few else 0
    overlap_thr = float(opts.overlap_thr)
    stim_filterby = None # 'first'
    has_gratings = experiment!='blobs'
 
    # Pupil -------------------------------------------
    pupil_feature='pupil_fraction'
    pupil_epoch='pre'
    pupil_snapshot=391800
    redo_pupil=False
    pupil_framerate=20.
    # -------------------------------------------------

    iti_pre=1.
    iti_post=1.
    stim_dur=1.

    # Set colors
    visual_areas, area_colors = putils.set_threecolor_palette()
    dpi = putils.set_plot_params()

    #### Responsive params
    n_stds = None if responsive_test=='ROC' else 2.5 #None
    response_str = '%s_%s-thr-%.2f' % (response_type, responsive_test, responsive_thr) 

    #### Source data
    edata, cells, MEANS = aggr.get_source_data(experiment, 
                    equalize_now='pupil' not in analysis_type,
                    responsive_test=responsive_test, responsive_thr=responsive_thr, 
                    response_type=response_type, trial_epoch=trial_epoch) 

    #### Get pupil responses
    if 'pupil' in analysis_type:
        pupildata = dlcutils.get_aggregate_pupildfs(experiment=experiment, 
                                    feature_name=pupil_feature, trial_epoch=pupil_epoch,
                                    iti_pre=iti_pre, iti_post=iti_post, stim_dur=stim_dur,
                                    in_rate=pupil_framerate, out_rate=pupil_framerate,
                                    snapshot=pupil_snapshot, create_new=redo_pupil)

    fovnum = int(fov.split('_')[0][3:])
    datakey = '%s_%s_fov%i' % (session, animalid, fovnum)
    assert datakey in cells['datakey'].unique(), "Dataset %s not segmented. Aborting." % datakey
    curr_visual_areas = cells[cells['datakey']==datakey]['visual_area'].unique()
    print("[%s] %i visul areas in current fov." % (datakey, len(visual_areas)))

    #### Setup output stuff    
    results_prefix = set_results_prefix(analysis_type=analysis_type)

    #### Do eeeeeet
    try:
        for visual_area in curr_visual_areas:    
            results_id=create_results_id(prefix=results_prefix, 
                            visual_area=visual_area, C_value=C_value, 
                            response_type=response_type, responsive_test=responsive_test)

            if analysis_type=='by_fov':
                decode_from_fov(datakey, visual_area, cells, MEANS, 
                                min_ncells=5, C_value=C_value,
                                n_iterations=n_iterations, n_processes=n_processes, 
                                results_id=results_id,
                                class_a=class_a, class_b=class_b,
                                rootdir=rootdir, create_new=create_new, verbose=verbose)

            elif analysis_type=='split_pupil': 
                decode_split_pupil(datakey, visual_area, cells, MEANS, pupildata,
                                n_cuts=3, feature_name='pupil_fraction',
                                min_ncells=5, C_value=C_value,
                                n_iterations=n_iterations, n_processes=n_processes, 
                                results_id=results_id,
                                class_a=class_a, class_b=class_b,
                                rootdir=rootdir, create_new=create_new, verbose=verbose)

            print("... finished %s (%s). ID=%s" % (datakey, visual_area, results_id))
    except Exception as e:
        traceback.print_exc()
 
    return 


if __name__ == '__main__':
    main(sys.argv[1:])
