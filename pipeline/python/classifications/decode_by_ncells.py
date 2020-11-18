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
            with open(results_outfile, 'rb') as f:
                iter_results = pkl.load(f)
        except Exception as e:
            create_new=True 


    if create_new:    
        # Decoding -----------------------------------------------------
        i_chunks = decutils.by_ncells_fit_svm_mp(ncells, celldf, NEURALDATA, sdf, 
                        n_iterations=n_iterations, n_processes=n_processes, 
                        C_value=C_value, cv_nfolds=cv_nfolds, test_split=test_split, 
                        class_a=class_a, class_b=class_b)
        print("%i items in mp list" % len(i_chunks))
        # DATA - get mean across items
        iter_results = pd.concat(i_chunks, axis=0)
        iter_results['n_cells'] = [ncells for _ in np.arange(0, iter_results.shape[0])]
        #iter_list.append(iter_df)
   
    print("Got curr results: %s" % str(iter_results.shape))

    # DATA - get mean across items
    #iter_results = pd.concat(iter_list, axis=0)
    with open(results_outfile, 'wb') as f:
        pkl.dump(iter_results, f, protocol=pkl.HIGHEST_PROTOCOL)

    # Pool mean
    #print(iter_results)
    print("... finished all iters: %s" % str(iter_results.shape))
    iterd = dict(iter_results.mean())
    iterd.update( dict(('%s_std' % k, v) \
            for k, v in zip(iter_results.std().index, iter_results.std().values)) )
    iterd.update( dict(('%s_sem' % k, v) \
            for k, v in zip(iter_results.sem().index, iter_results.sem().values)) )
    print("::FINAL::")
    pp.pprint(iterd)
    print("@@@@@@@@@ done. %s, n=%i @@@@@@@@@@" % (results_id, ncells))
    return iterd

def decode_from_cell(datakey, rid, neuraldf, sdf, return_shuffle=False, 
                    C_value=None, experiment='blobs',
                    n_iterations=50, n_processes=2, results_id='single_cell',
                    class_a=0, class_b=0, 
                    rootdir='/n/coxfs01/2p-data', create_new=False, verbose=False):
    # tmp save
    session, animalid, fov_ = datakey.split('_')
    fovnum = int(fov_[3:])
    fov = 'FOV%i_zoom2p0x' % fovnum
    traceid_dir = glob.glob(os.path.join(rootdir, animalid, session, 
                            fov, 'combined_%s*' % experiment, 'traces', '%s*' % traceid))[0]
    analysis_flag, subdir = results_id.split('__')
    curr_dst_dir = os.path.join(traceid_dir, 'decoding', 'single_cells', subdir)
    if not os.path.exists(curr_dst_dir):
        os.makedirs(curr_dst_dir)
    C_str = analysis_flag.split('_')[-1]
    bd_files = glob.glob(os.path.join(curr_dst_dir, '*%s*_%i.pkl' % (C_str, int(rid+1))))
    print("... deleting %i old files" % len(bd_files))
    for f in bd_files:
        os.remove(f)

    #print("***** Saving tmp results to:\n  %s" % curr_dst_dir)
    results_outfile = os.path.join(curr_dst_dir,'%s_%03d.pkl' % (analysis_flag, int(rid+1)))
    if create_new is False: 
        try:
            with open(results_outfile, 'rb') as f:
                iter_results = pkl.load(f)
        except Exception as e:
            create_new=True 
    
    metainfo = {'cell': rid, 'datakey': datakey}
    if create_new:    
        print("... Starting decoding analysis (rid=%i, %s)" % (rid, analysis_flag))
        # zscore full
        #neuraldf = aggr.zscore_neuraldf(neuraldf)

        # Decodinng -----------------------------------------------------
        if return_shuffle:
            iter_list, shuf_list = decutils.fit_svm_mp_shuffle(neuraldf, sdf, 
                                        C_value=C_value, 
                                        n_iterations=n_iterations, 
                                        n_processes=n_processes, verbose=verbose,
                                        class_a=class_a, class_b=class_b)
            print("%i items in mp list" % len(iter_list))
            # DATA - get mean across items
            iter_results = pd.concat(iter_list, axis=0) 
            shuf_results = pd.concat(shuf_list, axis=0)
            iter_results = putils.add_meta_to_df(iter_results, metainfo)
            shuf_results = putils.add_meta_to_df(shuf_results, metainfo)            
            all_results = {'results': iter_results, 'shuffled': shuf_results}
            with open(results_outfile, 'wb') as f:
                pkl.dump(all_results, f, protocol=pkl.HIGHEST_PROTOCOL)
            
        else: 
            iter_list = decutils.fit_svm_mp(neuraldf, sdf, C_value=C_value, 
                                        n_iterations=n_iterations, 
                                        n_processes=n_processes, verbose=verbose,
                                        class_a=class_a, class_b=class_b)
            print("%i items in mp list" % len(iter_list))
            # DATA - get mean across items
            iter_results = pd.concat(iter_list, axis=0) 
            iter_results = putils.add_meta_to_df(iter_results, metainfo)
            with open(results_outfile, 'wb') as f:
                pkl.dump(iter_results, f, protocol=pkl.HIGHEST_PROTOCOL)

    # Pool mean
    print("... saved: %s" % os.path.split(results_outfile)[-1])

    print("... finished all iters: %s" % str(iter_results.shape))
    iterd = dict(iter_results.mean())
    iterd.update( dict(('%s_std' % k, v) \
            for k, v in zip(iter_results.std().index, iter_results.std().values)) )
    iterd.update( dict(('%s_sem' % k, v) \
            for k, v in zip(iter_results.sem().index, iter_results.sem().values)) )
    iterd.update({'cell': rid, 'datakey': datakey})
    #print("::FINAL::")
    #pp.pprint(iterd)
    print("@@@@@@@@@ done. %s, rid=%i @@@@@@@@@@" % (datakey, rid))

    return iterd


# --------------------------------------------------------------------
# Loading, saving, ec.
# -------------------------------------------------------------------
def set_results_prefix(analysis_type='by_fov'):
    prefix=None
    if analysis_type=='by_fov':
        prefix = 'fov_results'
    elif analysis_type=='split_pupil':
        prefix = 'splitpupil_results'
    else:
        prefix = analysis_type

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
       
    # data filtering 
    choices_c = ('all', 'roc', 'nstds', None, 'None')
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
    parser.add_option('-v', action='store_true', dest='verbose', 
            default=False, help="verbose printage")
    parser.add_option('--new', action='store_true', dest='create_new', 
            default=False, help="re-do decode")

    parser.add_option('--cv', action='store_true', dest='do_cv', 
            default=False, help="tune for C")
    parser.add_option('-C','--cvalue', action='store', dest='C_value', 
            default=1.0, help="tune for C (default: 1)")


    choices_a = ('by_fov', 'split_pupil', 'by_ncells', 'single_cells')
    default_a = 'by_ncells'
    parser.add_option('-X','--analysis', action='store', dest='analysis_type', 
            default=default_a, type='choice', choices=choices_a,
            help="Analysis type, choices: %s. (default: %s" % (choices_a, default_a))


    parser.add_option('-V','--visual-area', action='store', dest='visual_area', 
            default=None, help="(set for by_ncells) Must be None to run all serially")
    parser.add_option('-S','--ncells', action='store', dest='ncells', 
            default=None, help="Must be None to run all serially")
    parser.add_option('-k','--datakey', action='store', dest='datakey', 
            default=None, help="(set for single_cells) Must be None to run all serially")

    parser.add_option('--match-distns', action='store_true', dest='match_distns', 
            default=False, help="(set for by_ncells) Match distns of neuraldf to Li")




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
    remove_too_few = True
    min_ncells = 10 if remove_too_few else 0
    overlap_thr = float(opts.overlap_thr)
    stim_filterby = None # 'first'
    has_gratings = experiment!='blobs'

    match_distns = opts.match_distns
    if analysis_type=='single_cells':
        match_distns = False

    # Pupil -------------------------------------------
    pupil_feature='pupil_fraction'
    pupil_epoch='pre'
    pupil_snapshot=391800
    redo_pupil=False
    pupil_framerate=20.
    # -------------------------------------------------

    # Alignment 
    iti_pre=1.
    iti_post=1.
    stim_dur=1.

    # RF stuff 
    rf_filter_by=None
    reliable_only = True
    rf_fit_thr = 0.05

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
    edata, expmeta = aggr.experiment_datakeys(sdata, experiment=experiment,
                                has_gratings=has_gratings, stim_filterby=stim_filterby)
        
    # Get blob metadata only - and only if have RFs
    dsets = pd.concat([g for k, g in edata.groupby(['animalid', 'session', 'fov']) if 
                (experiment in g['experiment'].values 
                and ('rfs' in g['experiment'].values or 'rfs10' in g['experiment'].values)) ])
    dsets[['visual_area', 'datakey']].drop_duplicates().groupby(['visual_area']).count()

    #### Check stimulus configs
    stim_datakeys = dsets['datakey'].unique()
    SDF = aggr.check_sdfs(stim_datakeys, traceid=traceid)

    #### Source data
    _, cells, MEANS = aggr.get_source_data(experiment, 
                        equalize_now=True, zscore_now=True,
                        response_type=response_type, responsive_test=responsive_test, 
                        responsive_thr=responsive_thr, trial_epoch=trial_epoch, use_all=False) 

    #### Load RFs
    rf_fit_desc = fitrf.get_fit_desc(response_type=response_type)
    reliable_str = 'reliable' if reliable_only else ''
    rf_str = 'match%s_%s' % (experiment, reliable_str)
    # Get position info for RFs 
    rfdf = aggr.load_rfdf_and_pos(dsets, rf_filter_by=None, 
                                    reliable_only=True, traceid=traceid)

    #### Final datasets
    #NEURALDATA, RFDATA = aggr.get_neuraldata_and_rfdata(cells, rfdf, MEANS)
    #match_distns = analysis_type=='by_ncells'
    stack_neuraldf = match_distns==True
    NEURALDATA, RFDATA = aggr.get_neuraldata_and_rfdata(cells, rfdf, MEANS, stack=stack_neuraldf)
    if match_distns:
        print("~~~~~~~~~~~~~~~~Matching max %s distNs~~~~~~~~~~~~~~~~~~~~~~~" % response_type)
        NEURALDATA, RFDATA = aggr.match_distns_neuraldata_and_rfdata(NEURALDATA, RFDATA)
    dist_str = 'matchdist_' if match_distns else ''

    #### Get pupil responses
    if 'pupil' in analysis_type:
        pupildata = dlcutils.get_aggregate_pupildfs(experiment=experiment, 
                                    feature_name=pupil_feature, trial_epoch=pupil_epoch,
                                    iti_pre=iti_pre, iti_post=iti_post, stim_dur=stim_dur,
                                    in_rate=pupil_framerate, out_rate=pupil_framerate,
                                    snapshot=pupil_snapshot, create_new=redo_pupil)
    #### Setup output dirs  
    results_prefix = set_results_prefix(analysis_type=analysis_type)
    data_info='%s%s_%s_overlap-%.1f_iter-%i' % (distn_str, response_type, responsive_test, overlap_thr, n_iterations)
    dst_dir = os.path.join(aggregate_dir, 'decoding', analysis_type, data_info)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
        print("...making dir")
    print("DST: %s" % dst_dir)

    #### Calculate overlap with stimulus
    print("Calculating overlaps (thr=%.2f)" % overlap_thr)
    stim_overlaps = rfutils.calculate_overlaps(RFDATA, experiment=experiment)

    #### Filter cells
    globalcells, cell_counts = decutils.get_pooled_cells(stim_overlaps,remove_too_few=remove_too_few, 
                                                        overlap_thr=overlap_thr, min_ncells=min_ncells)
    print("@@@@@@@@ cell counts @@@@@@@@@@@")
    print(cell_counts)
    sdf = SDF[SDF.keys()[-1]].copy()

    print('Classify %i v %i (C=%s)' % (m0, m100, str(C_value)))
    print('N=%i iters (%i proc), overlap=%.2f, C=%s' % (n_iterations, n_processes, overlap_thr))


    # ---------------------------------------------------------------------
    curr_visual_area = None if opts.visual_area in ['None', None] else opts.visual_area
    curr_datakey = None if opts.datakey in ['None', None] else opts.datakey
    
    if (curr_visual_area is None) and (curr_datakey is not None):
        # Get the visual area (first one) for datakey
        assert curr_datakey in globalcells['datakey'].unique(), "No dkey: %s" % curr_datakey
        curr_visual_area = globalcells[globalcells['datakey']==curr_datakey]['visual_area'].unique()[0]
    # ---------------------------------------------------------------------


    if analysis_type=='single_cells':
        # ============================================================ 
        # SINGLE_CELLS - for each fov, for each cell, do decode
        # ============================================================ 
        # datakey is not None and visual_area is not None
        gdf = globalcells[(globalcells['visual_area']==curr_visual_area) 
                        & (globalcells['datakey']==curr_datakey)]
        ncells_total = gdf.shape[0] 
        if gdf.shape[0]==0:
            print("... skipping %s (%s) - no cells." % (curr_datakey, curr_visual_area))
            return None
        results_id = create_results_id(prefix=results_prefix, 
                                visual_area=curr_visual_area, C_value=C_value, 
                                response_type=response_type, responsive_test=responsive_test)

        for ri, rid in enumerate(gdf['dset_roi'].values):
            if ri % 10 == 0:
                print("... %i of %i cells (%s|%s), rid=%i." % (int(ri+1), ncells_total, curr_datakey, curr_visual_area, rid))

            neuraldf = NEURALDATA[curr_visual_area][curr_datakey][[rid, 'config']].copy()
            decode_from_cell(curr_datakey, rid, neuraldf, sdf, experiment=experiment,
                            C_value=C_value, return_shuffle=False, 
                            n_iterations=n_iterations, n_processes=n_processes, 
                            results_id=results_id,
                            class_a=class_a, class_b=class_b,
                            create_new=create_new, verbose=verbose)          
        print("Finished %s (%s). ID=%s" % (curr_datakey, curr_visual_area, results_id))

    elif analysis_type=='by_ncells':
        # ============================================================ 
        # by NCELLS - pools across datasets
        # ============================================================ 
        curr_visual_area = opts.visual_area
        curr_ncells = None if opts.ncells in ['None', None] else int(opts.ncells) 

        if curr_visual_area is not None:
            gdf = globalcells[globalcells['visual_area']==curr_visual_area]
            results_id = create_results_id(prefix=results_prefix, 
                            visual_area=curr_visual_area, C_value=C_value, 
                            response_type=response_type, responsive_test=responsive_test)

            if curr_ncells is not None:
                # Do decode w CURR_NCELLS, CURR_VISUAL_AREA
                # ----------------------------------------------
                decode_by_ncells(curr_ncells, gdf, sdf, NEURALDATA, 
                                C_value=C_value,
                                n_iterations=n_iterations, n_processes=n_processes, 
                                results_id=results_id,
                                class_a=class_a, class_b=class_b,
                                dst_dir=dst_dir, create_new=create_new, verbose=verbose)
                print("***** finished %s, ncells=%i *******" % (curr_visual_area, curr_ncells))
            else:
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
            # Loop thru all visual areas, all NCELLS
            # ----------------------------------------------
            #### Get NCELLS
            min_cells_total = min(cell_counts.values())
            reasonable_range = [2**i for i in np.arange(0, 10)]
            incl_range = [i for i in reasonable_range if i<min_cells_total]
            incl_range.append(min_cells_total)
            NCELLS = incl_range

            try:
                for curr_visual_area, gdf in globalcells.groupby(['visual_area']):    
                    results_id = create_results_id(prefix=results_prefix, 
                                        visual_area=curr_visual_area, 
                                        C_value=C_value, 
                                        response_type=response_type, 
                                        responsive_test=responsive_test)
                        
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

        #            if analysis_type=='by_fov':
        #                decode_from_fov(datakey, visual_area, cells, MEANS, 
        #                                min_ncells=5, C_value=C_value,
        #                                n_iterations=n_iterations, n_processes=n_processes, 
        #                                results_id=results_id,
        #                                class_a=class_a, class_b=class_b,
        #                                rootdir=rootdir, create_new=create_new, verbose=verbose)
        #
        #            elif analysis_type=='split_pupil': 
        #                decode_split_pupil(datakey, visual_area, cells, MEANS, pupildata,
        #                                n_cuts=3, feature_name='pupil_fraction',
        #                                min_ncells=5, C_value=C_value,
        #                                n_iterations=n_iterations, n_processes=n_processes, 
        #                                results_id=results_id,
        #                                class_a=class_a, class_b=class_b,
        #                                rootdir=rootdir, create_new=create_new, verbose=verbose)
        #
            except Exception as e:
                traceback.print_exc()
         
    return 


if __name__ == '__main__':
    main(sys.argv[1:])
