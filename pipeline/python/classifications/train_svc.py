#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 16:32:57 2018

@author: juliana
"""


import sys
import optparse
import math
import os
import time
import copy
import glob
import itertools
import json
import h5py
import pandas as pd
import numpy as np
import multiprocessing as mp
import cPickle as pkl
import pylab as pl
import tifffile as tf

from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections import Counter
from scipy import stats

from pipeline.python.classifications import linearSVC_class as lsvc
from pipeline.python.utils import print_elapsed_time, natural_keys, label_figure, replace_root
from pipeline.python.rois import utils as util  #get_roi_contours, plot_roi_contours


from sklearn.feature_selection import RFE

from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold, cross_val_score
from sklearn import svm
from sklearn import cross_validation
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import permutation_test_score


#%%

def extract_options(options):

    def comma_sep_list(option, opt, value, parser):
        setattr(parser.values, option.dest, value.split(','))


    parser = optparse.OptionParser()

    parser.add_option('-D', '--root', action='store', dest='rootdir',
                          default='/nas/volume1/2photon/data',
                          help='data root dir (dir w/ all animalids) [default: /nas/volume1/2photon/data, /n/coxfs01/2pdata if --slurm]')
    parser.add_option('-i', '--animalid', action='store', dest='animalid',
                          default='', help='Animal ID')

    # Set specific session/run for current animal:
#    parser.add_option('-S', '--session', action='store', dest='session',
#                          default='', help='session dir (format: YYYMMDD_ANIMALID')
#    parser.add_option('-A', '--acq', action='store', dest='acquisition',
#                          default='FOV1', help="acquisition folder (ex: 'FOV1_zoom3x') [default: FOV1]")
#    parser.add_option('-R', '--run', action='store', dest='run',
#                          default='', help="RUN name (e.g., gratings_run1)")
#    parser.add_option('-t', '--traceid', action='store', dest='traceid',
#                          default='', help="traceid name (e.g., traces001)")

    # Run specific info:
    parser.add_option('-S', '--session', dest='session_list', default=[], type='string', action='callback', callback=comma_sep_list, help="SESSIONS for corresponding runs [default: []]")

    parser.add_option('-A', '--fov', dest='fov_list', default=[], type='string', action='callback', callback=comma_sep_list, help="FOVs for corresponding runs [default: []]")

    parser.add_option('-t', '--traceid', dest='traceid_list', default=[], type='string', action='callback', callback=comma_sep_list, help="TRACEIDs for corresponding runs [default: []]")

    parser.add_option('-R', '--run', dest='run_list', default=[], type='string', action='callback', callback=comma_sep_list, help='list of run IDs [default: []')
    

    parser.add_option('--slurm', action='store_true', dest='slurm', default=False, help="set if running as SLURM job on Odyssey")
    parser.add_option('--par', action='store_true', dest='multiproc', default=False, help="set if want to run MP on roi stats, when possible")
    parser.add_option('--nproc', action='store', dest='nprocesses', default=4, help="N processes if running in par (default=4)")

    # Classifier info:
    parser.add_option('-r', '--rois', action='store', dest='roi_selector', default='all', help="(options: all, visual)")
    parser.add_option('-d', '--dtype', action='store', dest='data_type', default='stat', help="(options: frames, stat)")
    stat_choices = {'stat': ['meanstim', 'meanstimdff', 'zscore'],
                    'frames': ['trial', 'stimulus', 'post']}
    parser.add_option('-s', '--stype', action='store', dest='stat_type', default='meanstim', 
                      help="If dtype is STAT, options: %s. If dtype is FRAMES, options: %s" % (str(stat_choices['stat']), str(stat_choices['frames'])))

    parser.add_option('-p', '--indata_type', action='store', dest='inputdata_type', default='corrected', help="data processing type (dff, corrected, raw, etc.)")
    parser.add_option('--null', action='store_true', dest='get_null', default=False, help='Include a null class in addition to stim conditions')
    parser.add_option('-N', '--name', action='store', dest='class_name', default='', help='Name of transform to classify (e.g., ori, xpos, morphlevel, etc.)')
    
    parser.add_option('--subset', dest='class_subset', default='', type='string', action='callback', 
                          callback=comma_sep_list, help='Subset of class_name types to learn')
    parser.add_option('-c', '--const', dest='const_trans', default='', type='string', action='callback', 
                          callback=comma_sep_list, help="Transform name to hold constant if classifying a different transform")
    parser.add_option('-v', '--tval', dest='trans_value', default='', type='string', action='callback', 
                          callback=comma_sep_list, help="Value to set const_trans to")
    parser.add_option('-T', '--testvals', dest='test_values', default=[], action='append', 
                          help="Values to hold as test set (in order listed in const_trans assignment")
    parser.add_option('--indie', action='store_true', dest='indie', default=False, help="set if each transform-value pair should be trained/tested independently")
    parser.add_option('-V', '--area', action='store', dest='visual_area', default='', help='Name of visual area (e.g., LI, LL, etc.)')


    parser.add_option('-L', '--clf', action='store', dest='classifier', default='LinearSVC', help='Classifier type (default: LinearSVC)')
    parser.add_option('-k', '--cv', action='store', dest='cv_method', default='kfold', help='Method of cross-validation (default: kfold)')
    parser.add_option('-f', '--folds', action='store', dest='cv_nfolds', default=5, help='N folds for CV (default: 5)')
    parser.add_option('-C', '--cval', action='store', dest='C_val', default=1e9, help='Value for C param if using SVC (default: 1e9)')
    parser.add_option('-g', '--groups', action='store', dest='cv_ngroups', default=1, help='N groups for CV, relevant only for data_type=frames (default: 1)')
    parser.add_option('-b', '--bin', action='store', dest='binsize', default=10, help='Bin size, relevant only for data_type=frames (default: 10)')

    parser.add_option('--Cval', action='store', dest='setC', default=1e9, help='C value or type to use for final trained classifier')
    parser.add_option('--nfeatures', action='store', dest='nfeatures_select', default='', help='Whether to use RFE to find best N rois (default: best, can be int)')
    parser.add_option('--partial', action='store_false', dest='full_train', default=True, help='Set flag to leave aside test_size of the training dataset for the test plots')
    parser.add_option('--test-size', action='store', dest='test_size', default=0.33, help='Fraction of train data to set aside for plotting test resutls (default: 0.33)')

    
    
    (options, args) = parser.parse_args(options)
    
    assert options.stat_type in stat_choices[options.data_type], "Invalid STAT selected for data_type %s. Run -h for options." % options.data_type

    return options

#%%

#options = ['-D', '/mnt/odyssey', '-i', 'JC015', '-S', '20180915', '-A', 'FOV1_zoom2p7x',
#           '-R', 'combined_gratings_static', '-t', 'traces001',
#           '-r', 'visual', '-d', 'stat', '-s', 'meanstim',
#           '-p', 'corrected', '-N', 'ori',
#           '-c', 'xpos',
#           '--nproc=4'
#           ]


#def get_transform_classifiers(optsE):
def get_transform_classifiers(animalid, session, acquisition, run, traceid, rootdir='/n/coxfs01/2p-data',
                         roi_selector='visual', data_type='stat', stat_type='meanstim',
                         inputdata_type='corrected', 
                         get_null=False, class_name='', class_subset='',
                         const_trans='', trans_value='', test_set=[], indie=False,
                         cv_method='kfold', cv_nfolds=5, cv_ngroups=1, C_val=1e9, binsize=10,
                         nprocesses=2):

    nprocs = int(nprocesses)
    
    C = lsvc.TransformClassifier(animalid, session, acquisition, run, traceid, rootdir=rootdir,
                         roi_selector=roi_selector, data_type=data_type, stat_type=stat_type,
                         inputdata_type=inputdata_type, 
                         get_null=get_null, class_name=class_name, class_subset=class_subset,
                         const_trans=const_trans, trans_value=trans_value, test_set=test_set, indie=indie,
                         cv_method=cv_method, cv_nfolds=cv_nfolds, cv_ngroups=cv_ngroups, C_val=C_val, binsize=binsize)
    
#    C = lsvc.TransformClassifier(optsE.animalid, optsE.session, optsE.acquisition, optsE.run, optsE.traceid,
#                                rootdir=optsE.rootdir, roi_selector=optsE.roi_selector, data_type=optsE.data_type,
#                                stat_type=optsE.stat_type, inputdata_type=optsE.inputdata_type,
#                                get_null=optsE.get_null, class_name=optsE.class_name, class_subset=optsE.class_subset,
#                                const_trans=optsE.const_trans, trans_value=optsE.trans_value, 
#                                cv_method=optsE.cv_method, cv_nfolds=optsE.cv_nfolds, cv_ngroups=optsE.cv_ngroups, binsize=optsE.binsize)
#    
    C.load_dataset()
    
    C.create_classifier_dirs()
    C.initialize_classifiers()
    
    C.label_classifier_data()
    
    t_train_mp = time.time()
#    def trainer(clf_list, out_q):
#        
#        outdict = {}
#        for clf in clf_list:
#            clf.train_classifier()
#            outdict[os.path.split(clf.classifier_dir)[-1]] = 'done!'
#            print "Finished training", clf
#        out_q.put(outdict)
#        
#    # Each process gets "chunksize' filenames and a queue to put his out-dict into:
#    clf_list = C.classifiers
#    out_q = mp.Queue()
#    chunksize = int(math.ceil(len(C.classifiers) / float(nprocs)))
#    procs = []
#    for i in range(nprocs):
#        p = mp.Process(target=trainer, 
#                       args=(clf_list[chunksize * i:chunksize * (i + 1)],
#                             out_q))
#        procs.append(p)
#        p.start()
#
#    # Collect all results into single results dict. We should know how many dicts to expect:
#    resultdict = {}
#    for i in range(nprocs):
#        resultdict.update(out_q.get())
#
#    # Wait for all worker processes to finish
#    for p in procs:
#        print "Finished:", p
#        p.join()
#        
#    print_elapsed_time(t_train_mp)
#    print resultdict

    for clf in C.classifiers:
        clf.train_classifier()
        with open(os.path.join(clf.classifier_dir, 'classifier_%s.pkl' % clf.hash), 'wb') as f:
            pkl.dump(clf, f, protocol=pkl.HIGHEST_PROTOCOL)
        print "Saved object to:", os.path.join(C.classifier_dir, os.path.join(clf.classifier_dir, 'classifier_%s.pkl' % clf.hash)) #'TransformClassifier.pkl')

    # Replace 'dataset':
    C.dataset = copy.copy(C.data_fpath)
    with open(os.path.join(C.classifier_dir, 'TransformClassifier.pkl'), 'wb') as f:
        pkl.dump(C, f, protocol=pkl.HIGHEST_PROTOCOL)
        
    print "*****************************"
    print "Finished training all %i classifiers." % len(C.classifiers)
    print "*****************************"
   
    # Reload dataset to return:
    tmp_dset = np.load(C.data_fpath)
    C.dataset = tmp_dset
    
    return C

#%%

def get_traceids_from_lists(animalid, session_list, fov_list, run_list, traceid_list, rootdir='/n/coxfs01/2p-data'):
    traceid_dirs = {}
    assert len(session_list)==len(fov_list)==len(run_list)==len(traceid_list), "Unmatched sessions, fovs, runs, and traceids provided."
    for sesh, fov, run, traceid in zip(session_list, fov_list, run_list, traceid_list):
        traceid_dir = get_traceid_dir(animalid, sesh, fov, run, traceid, rootdir=rootdir)
        fov_key = '%s_%s' % (sesh, fov)
        traceid_dirs[fov_key] = traceid_dir
        
    return traceid_dirs
            
def get_traceid_dir(animalid, session, acquisition, run, traceid, rootdir='/n/coxfs01/2p-data'):
    found_traceid_dirs = sorted(glob.glob(os.path.join(rootdir, animalid, session, acquisition, run, 'traces', '%s*' % traceid)), key=natural_keys)
    #print rootdir, animalid, session, acquisition, run, traceid
    assert len(found_traceid_dirs) > 0, "No traceids found."
    print rootdir, animalid, session, acquisition, run, traceid

    if len(found_traceid_dirs) > 1:
        for ti, tdir in enumerate(found_traceid_dirs):
            print ti, os.path.split(tdir)[-1]
        sel = input("Select IDX of traceid to use: ")
        traceid_dir = found_traceid_dirs[int(sel)]
    else:
        traceid_dir = found_traceid_dirs[0]
    
    return traceid_dir

def load_classifier_object(traceid_dir, rootdir='/n/coxfs01/2p-data', animalid='', session=''):
    C=None
    try:
        found_transform_classifiers = sorted(glob.glob(os.path.join(traceid_dir, 'classifiers', 'LinearSVC*', 'TransformClassifier.pkl')), key=natural_keys)
        assert len(found_transform_classifiers) > 0, "No classifiers found."
    except AssertionError:
        return C
    
    for ti, tclassifier in enumerate(found_transform_classifiers):
        print ti, os.path.split(os.path.split(tclassifier)[0])[-1]
    sel = raw_input("Select IDX of TransformClassifier object to load: ")
    if len(sel) > 0:
        tclf_fpath = found_transform_classifiers[int(sel)]
        if rootdir not in tclf_fpath:
            tclf_fpath = replace_root(tclf_fpath, rootdir, animalid, session)
        with open(tclf_fpath, 'rb') as f:
            C = pkl.load(f)
        if isinstance(C.dataset, str):
            if rootdir not in C.dataset:
                C.dataset = replace_root(C.dataset, rootdir, C.animalid, C.session)
            loaded_dset = np.load(C.dataset)
            C.dataset = loaded_dset
        
    return C

def get_clf_RFE(clf):
    rfe_results = {}
    clf_key = os.path.split(clf.classifier_dir)[-1]
    rresults = load_RFE_results(clf)
    rfe_scores = np.array([np.mean(cv_scores) for ri, cv_scores in enumerate(rresults['results'])])
    best_iter = rfe_scores.argmax()
    kept_rids = rresults['kept_rids_by_iter'][best_iter-1]
    bestN = len(kept_rids)
    print clf_key, "Best score: %.2f" % rfe_scores.max(), "N rois: %i" % bestN
    print " --->", kept_rids
    rfe_results['best'] = {'iter_ix': best_iter,
                           'kept_rids': kept_rids,
                           'max_score': rfe_scores.max()}
    return rfe_results
    


def get_RFE_results(C):
    '''For each classifier trained in TransformClassifier object (C), get RFE resuults
    '''
    rfe_results={}
    for clf in C.classifiers:
        clf_key = os.path.split(clf.classifier_dir)[-1]
        rfe_results[clf_key] = get_clf_RFE #load_RFE_results(clf)
    
#    for clf_key, res in rfe_results.items():
#        rfe_scores = np.array([np.mean(cv_scores) for ri, cv_scores in enumerate(res['results'])])
#        best_iter = rfe_scores.argmax()
#        kept_rids = res['kept_rids_by_iter'][best_iter-1]
#        bestN = len(kept_rids)
#        print clf_key, "Best score: %.2f" % rfe_scores.max(), "N rois: %i" % bestN
#        print " --->", kept_rids
#        rfe_results[clf_key]['best'] = {'iter_ix': best_iter,
#                                        'kept_rids': kept_rids,
#                                        'max_score': rfe_scores.max()}
#    
    return rfe_results


def load_RFE_results(clf):
    
    found_rfe_results = sorted(glob.glob(os.path.join(clf.classifier_dir, 'results', 'RFE*.pkl')), key=natural_keys)
    assert len(found_rfe_results) > 0, "No RFE results found in clf dir: %s" % clf.classifier_dir
    if len(found_rfe_results) > 1:
        for ri, rfe in enumerate(found_rfe_results):
            print ri, rfe
        sel = input("Select IDX of RFE results to use: ")
        rfe_fpath = found_rfe_results[sel]
    else:
        rfe_fpath = found_rfe_results[0]
    
    with open(rfe_fpath, 'rb') as f:
        rfe = pkl.load(f)

    return rfe

#%%

def train_and_validate_best_clf(clf, setC='big', nfeatures_select='all', full_train=False, test_size=0.33,
                                secondary_output_dir=None, data_identifier='dataID'):

    # =============================================================================
    # Train the classifier:
    # =============================================================================
    #setC = optsE.setC #'best' # choices: 'best' (test diff values of C on log scale), 'big' (1E9), 'small' (w.0)
    #nfeatures_select = int(optsE.nfeatures_select) if optsE.nfeatures_select.isdigit() else optsE.nfeatures_select # 'best' # choices: 'best' (uses RFE to find cells with max. clf accuracy), int (uses RFE to find top N cells)
    #full_train = optsE.full_train #True
    #test_size = float(optsE.test_size) #0.33
    
    if full_train:
        train_set = 'fulltrain'
    else:
        train_set = 'testsize%.2f' % test_size
        
    if nfeatures_select == 'best':
        rfe_results = get_clf_RFE(clf) #TODO:  This actually uses RFE results assuming default clf params (i.e., doesn't take into account params of specified clf)
    else:
        rfe_results = {}
    svc, cX, cy, kept_rids, clf_output_dir = find_best_classifier(clf.cX, clf.cy, clf.clf.get_params(), 
                                                                      setC=setC, 
                                                                      nfeatures_select=nfeatures_select,
                                                                      rfe_results=rfe_results,
                                                                      full_train=full_train, 
                                                                      test_size=test_size, 
                                                                      classifier_base_dir=clf.classifier_dir,
                                                                      secondary_output_dir=secondary_output_dir,
                                                                      data_identifier=data_identifier)
    
    svc, traindata, testdata, test = train_and_validate(svc, cX, cy, clf, clf_output_dir, full_train=full_train, test_size=test_size,
                                                                      secondary_output_dir=secondary_output_dir,
                                                                      data_identifier=data_identifier)
    
    return svc, traindata, testdata, test, kept_rids

def find_best_classifier(cX, cy, start_params, setC='best', nfeatures_select='best', 
                             full_train=True, test_size=0.33, classifier_base_dir='/tmp',
                             rfe_results={}, secondary_output_dir=None, data_identifier='dataID'):
    svc = LinearSVC()
    svc.set_params(C=start_params['C'],
                   loss=start_params['loss'],
                   class_weight=start_params['class_weight'],
                   dual=start_params['dual'],
                   fit_intercept=start_params['fit_intercept'],
                   intercept_scaling=start_params['intercept_scaling'],
                   max_iter=start_params['max_iter'],
                   multi_class=start_params['multi_class'],
                   penalty=start_params['penalty'],
                   random_state=start_params['random_state'],
                   tol=start_params['tol'],
                   verbose=0)
    
    tmp_cX = np.copy(cX)
    cy = np.copy(cy)
       
    # Select features (i.e., cells):
    # ----------------
    if isinstance(nfeatures_select, int):
    
        rfe = RFE(svc, n_features_to_select=nfeatures_select)
        rfe.fit(tmp_cX, cy)
        
        removed_rids = np.where(rfe.ranking_!=1)[0]
        kept_rids = np.array([i for i in np.arange(0, tmp_cX.shape[-1]) if i not in removed_rids])
        feature_str = 'fit%i_RFE' % nfeatures_select
        cX = tmp_cX[:, kept_rids]
    
    elif nfeatures_select == 'best':
        kept_rids = rfe_results['best']['kept_rids']
        feature_str = 'bestRFE'
        cX = tmp_cX[:, kept_rids]
    
    else:
        kept_rids = range(tmp_cX.shape[-1]) #C.rois
        cX = np.copy(tmp_cX)
        feature_str = 'allcells'
    
    # Set C:
    # ----------------
    if setC == 'best':
        C_val = lsvc.get_best_C(svc, cX, cy)
        pl.savefig(os.path.join(classifier_base_dir, 'figures', 'C_values.png'))
        if secondary_output_dir is not None:
            pl.savefig(os.path.join(secondary_output_dir, 'C_values_%s.png' % data_identifier))
        pl.close()
    elif setC == 'big':
        C_val = 1E9 #clf.clfparams['C_val']
    elif setC == 'small':
        C_val = 1.0
    else:
        C_val = svc.C
    C_str = '%sC%.2f' % (setC, C_val)
    
    # Create output dir:
    # ----------------
    clf_subdir = '%s_%s' % (feature_str, C_str)
    clf_output_dir = os.path.join(classifier_base_dir, clf_subdir)
    if not os.path.exists(os.path.join(clf_output_dir, 'figures')): os.makedirs(os.path.join(clf_output_dir, 'figures'))
    print "Saving current CLF results to:", clf_output_dir
    
    # Set classifier parameters:
    # ----------------
    dual = cX.shape[0] > cX.shape[1]
    svc.C = C_val
    svc.dual = dual
    print svc
    
    return svc, cX, cy, kept_rids, clf_output_dir


def train_and_validate(svc, cX, cy, clf, clf_output_dir, full_train=False, test_size=0.33,
                           secondary_output_dir=None, data_identifier='dataID'):
    #nfolds = clf.clfparams['cv_nfolds']
    #kfold = StratifiedKFold(n_splits=nfolds, shuffle=True)
    
    
    # Do cross-validation and plot results:
    # -------------------------------------------------------------------------
    
    # Do CV in limited dataset:
    fig = lsvc.hist_cv_permutations(svc, cX, cy, clf.clfparams)
    label_figure(fig, clf.data_identifier)
    pl.savefig(os.path.join(clf_output_dir, 'figures', 'cv_permutation.png'))
    if secondary_output_dir is not None:
        pl.savefig(os.path.join(secondary_output_dir, 'cv_permutation_%s.png' % data_identifier))
    pl.close()
    
    #% Confusion matrix:
    predicted, true, classes, config_names = lsvc.get_cv_folds(svc, clf.clfparams, cX, cy, cy_labels=clf.cy_labels, output_dir=None)
    cmatrix, cstr = lsvc.get_confusion_matrix(predicted, true, classes, average_iters=True)
    lsvc.plot_confusion_matrix(cmatrix, classes, ax=None, normalize=True, title='%s conf matrix (n=%i)' % (cstr, clf.clfparams['cv_nfolds']), cmap=pl.cm.Blues)
    pl.savefig(os.path.join(clf_output_dir, 'figures', 'cv_confusion.png'))
    if secondary_output_dir is not None:
        pl.savefig(os.path.join(secondary_output_dir, 'cv_confusion_%s.png' % data_identifier))
    pl.close()
    
    # 
    # =============================================================================
    # Train the classifier with specified params:
    # =============================================================================
    X_test = None; test_true=None; test_predicted=None;
    traintest = {}
    if full_train:
        svc.fit(cX, cy)
        # TODO: fix this, can't just use all cv test results -- do it confusion-matrix style and get totals for each tested class
        for classix, classname in enumerate(classes):
            ncorrect = cmatrix[classix,classix]
            ntotal = np.sum(cmatrix[classix, :])
            traintest[classname] = float(ncorrect) / float(ntotal)

        X_train = cX.copy()
        train_true = cy.copy()
        X_test = None
        test_true = None
    else:
        X_train, X_test, train_true, test_true = train_test_split(cX, cy, test_size=test_size, random_state=0, shuffle=True)
        
        svc.fit(X_train, train_true)
        test_predicted = svc.predict(X_test)
        #test_score = svc.score(X_test, test_true)
        for classix, classname in enumerate(sorted(np.unique(test_true))):
            sample_ixs = np.array([vi for vi,val in enumerate(test_true) if val == classname])
            curr_true = test_true[sample_ixs]
            curr_pred = test_predicted[sample_ixs]
            ncorrect = np.sum([1 if predval==trueval else 0 for predval,trueval in zip(curr_pred, curr_true)]) #np.sum([1 if p == classname else 0 for p in test_predicted])
            ntotal = len(sample_ixs)
            print  float(ncorrect) / float(ntotal)
            
            traintest[classname] = float(ncorrect) / float(ntotal)

            
    # Also save CV results by config, if relevant:
    trained_configs = list(set([cfg for sublist in config_names for cfg in sublist]))
    traintest_by_config = dict((cfg, {'ncorrect': [], 'ntotal': []}) for cfg in trained_configs)
    for cfg in trained_configs:
        for fold in range(len(predicted)):
            tested_ixs = np.where(config_names[fold]==cfg)[0]
            if len(tested_ixs)> 0:
                curr_true = true[fold][tested_ixs]
                curr_pred = predicted[fold][tested_ixs]
                curr_ncorrect = sum([1 if tru==guess else 0 for tru, guess in zip(curr_true, curr_pred)])
                curr_ntotal = len(tested_ixs)
            else:
                curr_ncorrect = 0
                curr_ntotal = 0
            traintest_by_config[cfg]['ncorrect'].append(curr_ncorrect)
            traintest_by_config[cfg]['ntotal'].append(curr_ntotal)
        traintest_by_config[cfg]['percent_correct'] = np.nanmean([float(corr)/float(tot) if tot > 0 else np.nan for corr, tot in zip(traintest_by_config[cfg]['ncorrect'], traintest_by_config[cfg]['ntotal'])])
    traintest['by_config'] = traintest_by_config

    testdata = {'data': X_test, 'labels': test_true, 'predicted': test_predicted}
    traindata = {'data': X_train, 'labels': train_true}
    
    return svc, traindata, testdata, traintest #X_test, test_true, test_predicted


def get_test_data(sample_data, sample_labels, sdf, clfparams, limit_to_trained=True):
    #sdf = pd.DataFrame(clf.sconfigs).T
    
    # Filter sconfigs by const-trans/trans-value pair:
    if clfparams['const_trans'] != '' and limit_to_trained is True:
        trans_sdf = copy.copy(sdf)
        for ctrans, cval in zip(clfparams['const_trans'], clfparams['trans_value']):
            print ctrans, cval
            trans_sdf = trans_sdf[trans_sdf[ctrans].isin(cval)]
    else:
        trans_sdf = copy.copy(sdf)
    test_values = [val for val in trans_sdf[clfparams['class_name']].unique() if val not in clfparams['class_subset']]
    
    test_sdf = trans_sdf[trans_sdf[clfparams['class_name']].isin(test_values)]
    configs_included = test_sdf.index.tolist()
    
    kept_trial_ixs = np.array([fi for fi, config in enumerate(sample_labels) if config in configs_included])
    print "N kept trials:", len(kept_trial_ixs)
    
    test_data = sample_data[kept_trial_ixs, :]
    tmp_test_labels = sample_labels[kept_trial_ixs]
    test_labels = [test_sdf[test_sdf.index==config][clfparams['class_name']][0] for config in tmp_test_labels]
    
    print "Test data:", test_data.shape
    print "Test labels:", list(set(test_labels))

    return test_data, test_labels


#%%
    
def get_fov(traceid_dir):
    
    run_dir = traceid_dir.split('/traces/')[0]
    stack = []
    if 'combined' in run_dir:
        acquisition_dir = os.path.split(run_dir)[0]
        stimtype = os.path.split(run_dir)[-1].split('_')[1]
        single_runs = [d for d in glob.glob(os.path.join(acquisition_dir, '%s_run*' % stimtype)) if os.path.isdir(d)]
        tiff_fpaths = [glob.glob(os.path.join(single_run, 'processed', 'processed*', 'mcorrected_*mean_deinterleaved', 'Channel01', 'File*', '*.tif')) for single_run in single_runs]
        tiff_fpaths = [f for sublist in tiff_fpaths for f in sublist]
    else:
        tiff_fpaths = glob.glob(os.path.join(run_dir, 'processed', 'processed*', 'mcorrected_*mean_deinterleaved', 'CHannel01', 'File*', '*.tif'))
            
    for tiff_fpath in tiff_fpaths:
        im = tf.imread(tiff_fpath)
        stack.append(im)
    fov_img = np.mean(np.array(stack), axis=0)
        
    return fov_img



def load_tid_rois(mask_fpath, file_key=None):
    maskfile = h5py.File(mask_fpath, 'r')

    if file_key is None:
        file_key = maskfile.keys()[0]
    masks = np.array(maskfile[file_key]['Slice01']['maskarray'])
    dims = maskfile[file_key]['Slice01']['zproj'].shape
    masks_r = np.reshape(masks, (dims[0], dims[1], masks.shape[-1]))
    
    return masks_r
    
def get_roi_masks(traceid_dir, rootdir='/n/coxfs01/2p-data'):
    if isinstance(traceid_dir, list):
        rundir = traceid_dir[0].split('/traces/')[0]
        single_traceid = traceid_dir[0].split('/traces/')[-1]
        acquisition_dir = os.path.split(rundir)[0]
        stimtype = os.path.split(rundir)[-1].split('_')[0]
        print "STIM:", stimtype
        combo_dirs = glob.glob(os.path.join(acquisition_dir, 'combined_%s*' % stimtype, 'traces', '*%s*' % single_traceid))
        assert len(combo_dirs) > 0, "No combo dirs, but multiple traceid dirs given..."
        if len(combo_dirs) == 1:
            traceid_dir = combo_dirs[0]
        else:
            for ci, cdir in enumerate(combo_dirs):
                print ci, cdir
            traceid_dir = combo_dirs[input('Select IDX of combo dir to use: ')]
        print "USING traceid_dir", traceid_dir
        
    run_dir = traceid_dir.split('/traces/')[0]
    traceid = traceid_dir.split('/traces/')[-1]
    acquisition_dir = os.path.split(run_dir)[0]
    if rootdir not in acquisition_dir:
        session_dir = os.path.split(acquisition_dir)[0]
        session = os.path.split(session_dir)[-1]
        animalid = os.path.split(os.path.split(session_dir)[0])[-1]
        acquisition_dir = replace_root(acquisition_dir, rootdir, animalid, session)
    run = os.path.split(run_dir)[-1]
    #if 'combined' in run:
    stimtype = run.split('_')[1]
    single_runs = [b for b in os.listdir(acquisition_dir) if stimtype in b and all([mov not in b for mov in ['combined', 'movie', 'dynamic']])]
    traceid_names = traceid.split('_')[0::2]
    combined_masks = []
    for tid, singlerun in zip(traceid_names, single_runs):
        print tid, singlerun
        single_run_maskfile = glob.glob(os.path.join(acquisition_dir, '*%s*' % singlerun, 'traces', '%s*' % tid, 'MASKS.hdf5'))[0]
        
        masks = load_tid_rois(single_run_maskfile, file_key='File001')
        combined_masks.append(masks)
    rois = np.mean(np.array(combined_masks), axis=0)
    
    return rois
        

#glob.glob(os.path.join(optsE.rootdir, optsE.animalid, '2018*', 'FOV*', '*blobs*', 'traces', 'traces*', 'classifiers'))

#%%
    

def train_and_test_transforms(trans_classifiers, clfs, clf_subdir, setC='big', nfeatures_select='best', 
                              full_train=False, test_size=0.2, secondary_output_dir=None,
                              no_morphs=False, trans0=None, trans1=None, middle_morph=53, m100=106):
    if no_morphs:
        nomorph_str = '_nomorphs'
    else:
        nomorph_str = ''
    
    print "trans:", trans0, "trans:", trans1
    
    if trans0 is None:
        trans0 = clfs[clfs.keys()[0]]['classifier'].clfparams['const_trans'][0]
    if trans1 is None and len(clfs[clfs.keys()[0]]['classifier'].clfparams['const_trans']) > 1:
        trans1 = clfs[clfs.keys()[0]]['classifier'].clfparams['const_trans'][1]

    class_name = clfs[clfs.keys()[0]]['classifier'].clfparams['class_name']
    trans_types = [t for t in [trans0, trans1] if t is not None]
    print "Trans:", trans_types
    
    TEST = {'by_config': {},
            'predicted': {},
            'data': {},
            'labels': {}}
    
    TRAIN = {'traindata': {},
            'testdata': {},
            'kept_rids': {},
            'svc': {}}
    
    all_trans1 = []; all_trans0 = []
    
    for fov in sorted(clfs.keys(), key=natural_keys):
        C = copy.copy(trans_classifiers[fov]['C'])
        clf = copy.copy(clfs[fov]['classifier'])
        svc, traindataX, testdataX, traintest_results, kept_rids = train_and_validate_best_clf(clf, setC=setC, nfeatures_select=nfeatures_select, 
                                                                               full_train=full_train, test_size=test_size,
                                                                               secondary_output_dir=secondary_output_dir, 
                                                                               data_identifier=clf.data_identifier)    
        
        # Get original data and labels:
        sample_data = C.sample_data[:, kept_rids]
        sample_labels = C.sample_labels
        sdf = pd.DataFrame(clf.sconfigs).T
        if no_morphs:
            sdf = sdf[sdf[class_name].isin(clf.clfparams['class_subset'])]
            
        # Keep track of all original transforms:
        if trans0 is not None:
            all_trans0.extend(sorted([i for i in sdf[trans0].unique()]))
        if trans1 is not None:
            all_trans1.extend(sorted([i for i in sdf[trans1].unique()]))

        
        # Check that all provided trans-values are valid and get list of config
        # names to be included for TRAINING set:
        const_trans_dict = dict((k, v) for k,v in  zip(clf.clfparams['const_trans'], clf.clfparams['trans_value']))

        if const_trans_dict is not None and len(const_trans_dict.keys()) > 0:
            keys, values = zip(*const_trans_dict.items())
            #values_flat = [val[0] for val in values]
            train_list = [dict(zip(keys, v)) for v in itertools.product(*values)]
        else:
            # clf is trained on ALL configs:
            #tconfigs = traintest_results['by_configs'].keys()
            const_trans_all = [ctrans for ctrans, values in clf.run_info['transforms'].items() if len(values) > 0]
            const_trans_dict = dict((ctrans, list(sdf[ctrans].unique())) for ctrans in const_trans_all)
            keys, values = zip(*const_trans_dict.items())
            #values_flat = [val[0] for val in values]
            train_list = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        train_transforms = [tuple(tdict.values()) for tdict in train_list if tdict not in clf.clfparams['test_set']]

    
        grouped_sdf = sdf.groupby(trans_types)
        tmp_data = sample_data.copy()
        tmp_labels = sample_labels.copy()
        #tmp_labels = [sdf[sdf.index==cfg][clf.clfparams['class_name']][0] for cfg in tmp_labels]
        
        testdata = dict((tconfig, {}) for tconfig in grouped_sdf.groups.keys())
        performance = {}; counts = {}; ncorrect={};
        
        for trans_config,g in grouped_sdf:
            print trans_config
            if trans_config in train_transforms:
                testdata[trans_config]['data'] = testdataX['data']
                testdata[trans_config]['labels'] = testdataX['labels']
                testdata[trans_config]['predicted'] = testdataX['predicted']
                if testdataX['predicted'] is not None:
                    print "*****Train: %s, %i" % (str(trans_config), len(testdataX['predicted']))
            else:
                incl_ixs = np.array([i for i,label in enumerate(tmp_labels) if label in g.index.tolist()])
                testdata[trans_config]['data'] = tmp_data[incl_ixs, :]
                curr_labels = tmp_labels[incl_ixs]
                
                testdata[trans_config]['labels'] = np.array([sdf[sdf.index==cfg][clf.clfparams['class_name']][0] for cfg in curr_labels])
                testdata[trans_config]['predicted'] = svc.predict(testdata[trans_config]['data'])

            if full_train and trans_config in train_transforms:
                curr_pcorrect = np.mean([cinfo['percent_correct'] for cname, cinfo in traintest_results['by_config'].items() if cname in g.index.tolist()])
                curr_ncorrect = np.sum([cinfo['ncorrect'] for cname, cinfo in traintest_results['by_config'].items() if cname in g.index.tolist()])
                curr_counts = np.sum([cinfo['ntotal'] for cname, cinfo in traintest_results['by_config'].items() if cname in g.index.tolist()])
            else:
                left_trials = np.where(testdata[trans_config]['labels'] < middle_morph)[0]
                right_trials = np.where(testdata[trans_config]['labels'] > middle_morph)[0]
                mid_trials = np.where(testdata[trans_config]['labels'] == middle_morph)[0]
                correct = [1 if val==0 else 0 for val in testdata[trans_config]['predicted'][left_trials]]
                correct.extend([1 if val==m100 else 0 for val in testdata[trans_config]['predicted'][right_trials]])
                curr_ncorrect = np.sum(correct)
                curr_counts = len(left_trials) + len(right_trials)
                curr_pcorrect = float(curr_ncorrect) / float(curr_counts) #float(len(testdata[k]['predicted']))
                
            performance[trans_config] = curr_pcorrect
            counts[trans_config] = curr_counts
            ncorrect[trans_config] = curr_ncorrect
        
        TEST['by_config'][fov] = {'accuracy': performance, 'counts': counts, 'ncorrect': ncorrect}
        TEST['predicted'][fov] = dict((tconfig, tdata['predicted']) for tconfig, tdata in testdata.items())
        TEST['data'][fov] = dict((tconfig, tdata['data']) for tconfig, tdata in testdata.items())
        TEST['labels'][fov] = dict((tconfig, tdata['labels']) for tconfig, tdata in testdata.items())
    
        TRAIN['testdata'][fov] = testdataX
        TRAIN['traindata'][fov] = traindataX
        TRAIN['kept_rids'][fov] = kept_rids
        TRAIN['svc'][fov] = svc

    rowvals = sorted([i for i in list(set(all_trans1))]) #sdf[clf.clfparams['const_trans'][0]].unique()])
    colvals = sorted([i for i in list(set(all_trans0))])  #sdf[clf.clfparams['const_trans'][1]].unique()])
    TEST['rowvals'] = rowvals
    TEST['colvals'] = colvals
 
    train_trans_fpath = os.path.join(clf_subdir, 'TRAIN_clfs_transforms%s.pkl' % nomorph_str)
    with open(train_trans_fpath, 'wb') as f:
        pkl.dump(TRAIN, f, protocol=pkl.HIGHEST_PROTOCOL)
        
    test_trans_fpath = os.path.join(clf_subdir, 'TEST_clfs_transforms%s.pkl' % nomorph_str)
    with open(test_trans_fpath, 'wb') as f:
        pkl.dump(TEST, f, protocol=pkl.HIGHEST_PROTOCOL)
        
    pl.close('all')
#%
    return TRAIN, TEST


def transform_performance_grid(TEST):
    rowvals = TEST['rowvals']
    colvals = TEST['colvals']
    
    if len(rowvals) == 0:
        rowvals = [0]
        placehold = True
    else:
        placehold = False
    
    ncorrect_grid = np.ones((len(rowvals), len(colvals)))*np.nan
    counts_grid = np.ones((len(rowvals), len(colvals)))*np.nan
    config_grid = {}
    
    for fov in TEST['by_config'].keys():
        #performance = TEST['morph_results'][fov]['accuracy']
        ncorrect = TEST['by_config'][fov]['ncorrect']
        counts = TEST['by_config'][fov]['counts']
                
        grid_pairs = sorted(list(itertools.product(colvals, rowvals)), key=lambda x: (x[0], x[1]))
        for trans_config in grid_pairs:
            #print ncorrect_grid
            if placehold is True and not all([t in ncorrect.keys() for t in trans_config]):
                continue
            if not trans_config in ncorrect.keys():
                continue
            rix = rowvals.index(trans_config[1])
            cix = colvals.index(trans_config[0])
            print rix, cix
            if placehold: # second trans_config value is fake
                nc = ncorrect[trans_config[0]]
                cc = counts[trans_config[0]]
            else:
                nc = ncorrect[trans_config]
                cc = counts[trans_config]
            if np.isnan(ncorrect_grid[rix, cix]):
                ncorrect_grid[rix, cix] = nc #ncorrect[trans_config]
                counts_grid[rix, cix] = cc #counts[trans_config]
            else:
                ncorrect_grid[rix, cix] += nc #ncorrect[trans_config] 
                counts_grid[rix, cix] += cc #counts[trans_config]
            config_grid[trans_config] = (rix, cix)
        
    performance_grid = ncorrect_grid / counts_grid
    print config_grid
    
    return performance_grid, counts_grid


def plot_transform_grid(performance_grid, rowvals=[], colvals=[], 
                                          ylabel='rows', xlabel='columns', 
                                          cmap='hot', vmin=0.5, vmax=1.0,
                                          data_identifier='', ax=None):
    
    if ax is None:
        fig, ax = pl.subplots(1, figsize=(15,8))

        
    if len(rowvals) != performance_grid.shape[0]:
        print "Not enough ROW labels:", rowvals
        rowvals = np.arange(performance_grid.shape[0])
    if len(colvals) != performance_grid.shape[1]:
        print "Not enough COL labels:", colvals
        colvals = np.arange(performance_grid.shape[1])
        
    im = ax.imshow(performance_grid, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(colvals)))
    ax.set_xticklabels(colvals)
    ax.set_xlabel(xlabel)
    ax.set_yticks(range(len(rowvals)))
    ax.set_yticklabels(rowvals)
    ax.set_ylabel(ylabel)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    pl.colorbar(im, cax=cax)
    
    # Loop over data dimensions and create text annotations.
    for i in range(len(rowvals)):
        for j in range(len(colvals)):
            if performance_grid[i, j] >= vmax*0.75:
                text = ax.text(j, i, '%.2f' % performance_grid[i, j],
                           ha="center", va="center", color="k")
            else:
                text = ax.text(j, i, '%.2f' % performance_grid[i, j],
                           ha="center", va="center", color="w")
        
    #pl.colorbar()
    
    return fig

#%%
    
#options = ['-D', '/n/coxfs01/2p-data', '-i', 'JC022', '-S', '20181018', '-A', 'FOV2_zoom2p7x',
#           '-R', 'combined_blobs_static', '-t', 'traces001',
#           '-r', 'visual', '-d', 'stat', '-s', 'zscore',
#           '-p', 'corrected', '-N', 'morphlevel',
#           '--subset', '0,106',
#           '--nproc=1'
#           ]

rootdir = '/n/coxfs01/2p-data' #-data'



# =============================================================================
# LI -- blobs (6x4)
# =============================================================================
#options = ['-D', rootdir, '-i', 'JC022', 
#           '-S', '20181007,20181017', 
#           '-A', 'FOV1_zoom2p2x,FOV1_zoom2p7x',
#           '-R', 'combined_blobs_static,combined_blobs_static',
#           '-t', 'traces001,traces001',
#           '-r', 'visual', '-d', 'stat', '-s', 'zscore',
#           '-p', 'corrected', '-N', 'morphlevel',
#           '--subset', '0,106',
#           '-c', 'xpos,ypos',
##           '-v', '-5,0',
##           '-T', '-15,-10,0,5',
##           '-T', '-60,-30,30,60',
#           '-T', '5.6,16.8,28,5.6,16.8,28',
#           '-T', '5,5,5,15,15,15',
#           
#           '--nproc=1'
#           ]
# =============================================================================


# =============================================================================
# LI -- gratings (6x4)
# =============================================================================
#options = ['-D', rootdir, '-i', 'JC022', 
#           '-S', '20181005,20181005,20181006,20181007,20181017', 
#           '-A', 'FOV2_zoom2p7x,FOV3_zoom2p7x,FOV1_zoom2p7x,FOV1_zoom2p2x,FOV1_zoom2p7x',
#           '-R', 'combined_gratings_static,combined_gratings_static,combined_gratings_static,combined_gratings_static,combined_gratings_static',
#           '-t', 'traces001,traces001,traces001,traces001,traces001',
#           '-r', 'visual', '-d', 'stat', '-s', 'zscore',
#           '-p', 'corrected', 
#           '-N', 'ori',
##           '--subset', '0,106',
#           '-c', 'xpos,ypos',
##           '-v', '-5,0',
##           '-T', '-15,-10,0,5',
##           '-T', '-60,-30,30,60',
#           '-T', '5.6,16.8,28,5.6,16.8,28',
#           '-T', '5,5,5,15,15,15',
#           
#           '--nproc=1'
#           ]
# =============================================================================


# =============================================================================
# LI - BLOBS 5x5x5
# =============================================================================

options = ['-D', rootdir, '-i', 'JC022', 
           '-S', '20181022',
           '-A', 'FOV1_zoom4p0x',
           '-R', 'combined_blobs_static', 
           '-t', 'traces001',
           '-r', 'visual', '-d', 'stat', '-s', 'zscore',
           '-p', 'corrected', '-N', 'morphlevel',
           '--subset', '0,106',
           '-c', 'xpos,yrot',
#           '-v', '-5,0',
#           '-T', '-15,-10,0,5',
#           '-T', '-60,-30,30,60',
           '-T', '-5,-5,0,0,0,5,5,5',
           '-T', '60,30,60,30,0,60,30,0',
           
           '--nproc=1'
           ]
#


# =============================================================================
# LM - BLOBS 5x5x5
# =============================================================================

#options = ['-D', rootdir, '-i', 'CE077', 
#           '-S', '20180515,20180516,20180518,20180521,20180521,20180523,20180602,20180609,20180612', 
#           '-A', 'FOV1_zoom1x,FOV1_zoom1x,FOV1_zoom1x,FOV1_zoom1x,FOV2_zoom1x,FOV1_zoom1x,FOV1_zoom1x,FOV1_zoom1x,FOV1_zoom1x',
#           '-R', 'blobs_run3,blobs_run3,combined_blobs_static,combined_blobs_static,combined_blobs_static,combined_blobs_static,combined_blobs_static,combined_blobs_static,blobs_run1', 
#           '-t', 'traces001,traces001,traces002,traces002,traces002,traces002,traces001,traces001,traces001',
#           '-r', 'visual', '-d', 'stat', '-s', 'zscore',
#           '-p', 'corrected', '-N', 'morphlevel',
#           '--subset', '0,106',
#           '-c', 'xpos,yrot',
##           '-v', '-5,0',
##           '-T', '-15,-10,0,5',
##           '-T', '-60,-30,30,60',
#           '-T', '-5,-5,0,0,0,5,5,5',
#           '-T', '60,30,60,30,0,60,30,0',
#           
#           '--nproc=1'
#           ]
##


#options = ['-D', rootdir, '-i', 'CE077', 
#           '-S', '20180518,20180521,20180521,20180523,20180602,20180609,20180612', 
#           '-A', 'FOV1_zoom1x,FOV1_zoom1x,FOV2_zoom1x,FOV1_zoom1x,FOV1_zoom1x,FOV1_zoom1x,FOV1_zoom1x',
#           '-R', 'combined_blobs_static,combined_blobs_static,combined_blobs_static,combined_blobs_static,combined_blobs_static,combined_blobs_static,blobs_run1', 
#           '-t', 'traces002,traces002,traces002,traces002,traces001,traces001,traces001',
#           '-r', 'visual', '-d', 'stat', '-s', 'zscore',
#           '-p', 'corrected', '-N', 'morphlevel',
#           '--subset', '0,106',
#           '-c', 'xpos,yrot',
##           '-v', '-5,0',
##           '-T', '-15,-10,0,5',
##           '-T', '-60,-30,30,60',
#           '-T', '-5,-5,0,0,0,5,5,5',
#           '-T', '60,30,60,30,0,60,30,0',
#           
#           '--nproc=1'
#           ]
# =============================================================================



# =============================================================================
# LI/LL -- MORPHLINE
# =============================================================================

#options = ['-D', rootdir, '-i', 'JC022', 
#           '-S', '20181016,20181016,20181018,20181020,20181020', 
#           '-A', 'FOV1_zoom2p7x,FOV2_zoom2p7x,FOV2_zoom2p7x,FOV1_zoom2p7x,FOV2_zoom2p7x',
#           '-R', 'combined_blobs_static,combined_blobs_static,combined_blobs_static,combined_blobs_static,combined_blobs_static', 
#           '-t', 'traces001,traces001,traces001,traces001,traces001',
#           '-r', 'visual', '-d', 'stat', '-s', 'meanstim',
#           '-p', 'corrected', '-N', 'morphlevel',
#           '--subset', '0,106',
#           '-c', 'ypos',
#           '-v', '13',
#           '-V', 'LI',
#           '--nproc=1'
#           ]
#options = ['-D', rootdir, '-i', 'JC022', 
#           '-S', '20181016,20181018,20181020,20181020', 
#           '-A', 'FOV2_zoom2p7x,FOV2_zoom2p7x,FOV1_zoom2p7x,FOV2_zoom2p7x',
#           '-R', 'combined_blobs_static,combined_blobs_static,combined_blobs_static,combined_blobs_static', 
#           '-t', 'traces001,traces001,traces001,traces001',
#           '-r', 'visual', '-d', 'stat', '-s', 'meanstim',
#           '-p', 'corrected', '-N', 'morphlevel',
#           '--subset', '0,106',
##           '-c', 'ypos',
##           '-v', '13',
#           '-V', 'LI',
##           '--Cval=1.0',
#           '--nproc=1'
#           ]


# =============================================================================
# LM -- MORPHLINE
# =============================================================================


# ** morphline:  blobs_run2 was good (traces001) -- re-extract and try plotting
#options = ['-D', rootdir, '-i', 'CE077', 
#           '-S', '20180523', 
#           '-A', 'FOV1_zoom1x',
#           '-R', 'blobs_run2', 
#           '-t', 'traces001',
#           '-r', 'visual', '-d', 'stat', '-s', 'meanstim',
#           '-p', 'corrected', '-N', 'morphlevel',
#           '--subset', '0,106',
##           '-c', 'xpos,yrot',
##           '-v', '-5,0',
#           '-V', 'LM',
#           '--nproc=1'
#           ]

#options = ['-D', rootdir, '-i', 'CE077', 
#           '-S', '20180523', 
#           '-A', 'FOV1_zoom1x',
#           '-R', 'blobs_run2', 
#           '-t', 'traces001',
#           '-r', 'visual', '-d', 'stat', '-s', 'meanstim',
#           '-p', 'corrected', '-N', 'morphlevel',
#           '--subset', '0,106',
##           '-c', 'xpos,yrot',
##           '-v', '-5,0',
#           '-V', 'LM',
#           '--nproc=1'
#           ]

# ** morphline:  blobs_run2 was good (traces001) -- re-extract and try plotting
#options = ['-D', rootdir, '-i', 'CE077', 
#           '-S', '20180523', 
#           '-A', 'FOV1_zoom1x',
#           '-R', 'combined_blobs_static', 
#           '-t', 'traces002',
#           '-r', 'visual', '-d', 'stat', '-s', 'meanstim',
#           '-p', 'corrected', '-N', 'morphlevel',
#           '--subset', '0,106',
##           '-c', 'xpos,yrot',
##           '-v', '-5,0',
#           '--nproc=1'
#           ]
#
#T = lsvc.TransformClassifier(optsE.animalid, curr_session, curr_fov, curr_run, curr_traceid, rootdir=optsE.rootdir,
#                                          roi_selector=optsE.roi_selector, data_type=optsE.data_type, stat_type=optsE.stat_type,
#                                          inputdata_type=optsE.inputdata_type, 
#                                          get_null=optsE.get_null, class_name=optsE.class_name, class_subset=optsE.class_subset,
#                                          const_trans=optsE.const_trans, trans_value=optsE.trans_value,
#                                          cv_method=optsE.cv_method, cv_nfolds=optsE.cv_nfolds, cv_ngroups=optsE.cv_ngroups, 
#                                          C_val=optsE.C_val, binsize=optsE.binsize, test_set=test_set, indie=optsE.indie)
#
#T.load_dataset()
###
#T.create_classifier_dirs()
###
#T.initialize_classifiers()
##
##T.label_classifier_data()
#
#    
    
    
#%%

def main(options):
    #%%
    optsE = extract_options(options)
    visual_area = optsE.visual_area
    print "VISUAL AREA: %s" % visual_area

    test_set = []
    if len(optsE.test_values) > 0:
        tmp_value_list=[]
        for vi, val_str in enumerate(optsE.test_values):
            values = [float(v) for v in val_str.split(',')]
            tmp_value_list.append(values)
        assert len(list(set([len(values) for values in tmp_value_list]))) == 1, "Must provide corresponding value pairs if more than 1 transform varies."
        assert len(optsE.const_trans) == len(tmp_value_list), "Must provide subset of test values for EACH transform: %s" % str(optsE.const_trans)
        test_pairs = [(t1, t2) for t1, t2 in zip(tmp_value_list[0], tmp_value_list[1])] #list(itertools.product(*tmp_value_list))
        
        for cfg in test_pairs:
            test_set.append(dict((tname, tvalue) for tname, tvalue in zip(optsE.const_trans, cfg)))
    
    test_set = list(np.unique(np.array(test_set)))
    print "Holding %i configs for test_set." % len(test_set)
            
    
    traceid_dirs = get_traceids_from_lists(optsE.animalid, optsE.session_list, optsE.fov_list, optsE.run_list, optsE.traceid_list, rootdir=rootdir)
    #traceid_dir = get_traceid_dir(optsE.animalid, optsE.session, optsE.acquisition, optsE.run, optsE.traceid)


            
    # Combine data arrays -- just  use stim-period zscore? (then dont have to deal w/ different trial structures)
    #clfs = dict((fov, {}) for fov in sorted(traceid_dirs.keys(), key=natural_keys))
    trans_classifiers = dict()
    for fov, traceid_dir in sorted(traceid_dirs.items(), key=lambda x: x[0]):
        print "Getting TransformClassifiers() for: %s" % fov
        curr_session = fov.split('_')[0]
        curr_fov = '_'.join(fov.split('_')[1:3])
        curr_traceid = os.path.split(traceid_dir)[-1].split('_')[0]
        curr_run = os.path.split(traceid_dir.split('/traces/')[0])[-1]
        print "SESSION: %s | FOV: %s | RUN: %s | traceid: %s" % (curr_session, curr_fov, curr_run, curr_traceid)
        C = load_classifier_object(traceid_dir, rootdir=optsE.rootdir, animalid=optsE.animalid, session=curr_session)
        print C
        if C is None:
            C = get_transform_classifiers(optsE.animalid, curr_session, curr_fov, curr_run, curr_traceid, rootdir=optsE.rootdir,
                                          roi_selector=optsE.roi_selector, data_type=optsE.data_type, stat_type=optsE.stat_type,
                                          inputdata_type=optsE.inputdata_type, 
                                          get_null=optsE.get_null, class_name=optsE.class_name, class_subset=optsE.class_subset,
                                          const_trans=optsE.const_trans, trans_value=optsE.trans_value,
                                          cv_method=optsE.cv_method, cv_nfolds=optsE.cv_nfolds, cv_ngroups=optsE.cv_ngroups, 
                                          C_val=optsE.C_val, binsize=optsE.binsize, nprocesses=optsE.nprocesses, test_set=test_set, indie=optsE.indie)
        if optsE.rootdir not in C.classifier_dir:
            C.classifier_dir = replace_root(C.classifier_dir, optsE.rootdir, optsE.animalid, curr_session)
            for clf in C.classifiers:
                clf.classifier_dir = replace_root(clf.classifier_dir, optsE.rootdir, optsE.animalid, curr_session)

        trans_classifiers[fov] = {'C': C, 'traceid_dir': traceid_dir}
        
    
    #%
    fov_list = sorted(trans_classifiers.keys(), key=natural_keys)
    #visual_area = 'LI'
    grouped_fovs = '_'.join(sorted(fov_list, key=natural_keys))
    print grouped_fovs
    output_dir = os.path.join(optsE.rootdir, optsE.animalid, visual_area, grouped_fovs)
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    print "-- Saving all current %s output to:\n%s" % (visual_area, output_dir)
    
    clf_names = list(set([clfdict['C'].classifiers[0].classifier_dir.split('/classifiers/')[-1] for fov, clfdict in trans_classifiers.items()]))
    if len(clf_names) > 1:
        print "*** WARNING *** Looks like you're trying to combine different types of classifiers:\n    %s" % clf_names
        for ci,cname in enumerate(clf_names):
            print ci, cname
        user_choice = input("Select IDX of name to use: ")
        clf_name = clf_names[int(user_choice)]
    else:
        clf_name = clf_names[0]
    
    clf_output_dir = os.path.join(output_dir, 'classifiers', clf_name)
    if not os.path.exists(clf_output_dir): os.makedirs(clf_output_dir)
    
    print "Saving TransformClassifier output to:\n    -->%s" % clf_output_dir
    
    #%%
    # =========================================================================
    # Specify training meta parameters - this also sets the output dir:
    # =========================================================================
    setC='big'
    nfeatures_select='best'
    full_train=True
    test_size=0.0 #0.33
    train_set = '%s_%s' % ('full' if full_train else 'partial', str(test_size) if not full_train else '0')
    no_morphs = False #False
    # =========================================================================
    
    trained_labels = list(set([int(i) for fov, tdict in trans_classifiers.items() for i in tdict['C'].classifiers[0].clfparams['class_subset']]))
    
    if no_morphs:
        class_subset = '%s_%s' % (trans_classifiers[trans_classifiers.keys()[0]]['C'].classifiers[0].clfparams['class_name'], '_'.join([str(l) for l in trained_labels]))
    else:
        class_subset = '%s_all' % trans_classifiers[trans_classifiers.keys()[0]]['C'].classifiers[0].clfparams['class_name']
    
    if nfeatures_select.isdigit():
        nfeatures_select = int(nfeatures_select)
        
    
    # Create output dirs:
    # -------------------------------------------------------------------------
    clf_desc = '%s_C_%s_features_%s_%s' % (class_subset, str(setC), str(nfeatures_select), train_set)
    clf_subdir = os.path.join(clf_output_dir, clf_desc)
    if not os.path.exists(clf_subdir): os.makedirs(clf_subdir)

    test_transforms_dir = os.path.join(clf_subdir, 'test_transforms')
    test_morphs_dir = os.path.join(clf_subdir, 'test_morphs')
    if not os.path.exists(os.path.join(test_transforms_dir, 'cross_validation')): os.makedirs(os.path.join(test_transforms_dir, 'cross_validation'))
    if not os.path.exists(os.path.join(test_morphs_dir, 'cross_validation')): os.makedirs(os.path.join(test_morphs_dir, 'cross_validation'))


    # Load or create classifier dicts that combine all specified fovs:
    # -------------------------------------------------------------------------
#    clfs_fpath = os.path.join(clf_subdir, 'classifiers.pkl')
#    if os.path.exists(clfs_fpath):
#        print "--- Loading existing classifier dict.---"
#        with open(clfs_fpath, 'rb') as f:
#            clfs = pkl.load(f)
#    else:
    
    clfs_fpath = os.path.join(clf_subdir, 'classifiers.pkl')
    
    clfs = dict((fov, {}) for fov in sorted(trans_classifiers.keys(), key=natural_keys))
    print "--- Creating NEW classifier dict.---"

    for fov in trans_classifiers.keys():
        print "... ... loading %s" % fov
        C = trans_classifiers[fov]['C']
        traceid_dir = trans_classifiers[fov]['traceid_dir']
        clfs[fov]['classifier'] = copy.copy(C.classifiers[0])
        clfs[fov]['fov'] = get_fov(traceid_dir) 
#%
        clfs[fov]['RFE'] = get_clf_RFE(clfs[fov]['classifier'])
        clfs[fov]['rois'] = get_roi_masks(clfs[fov]['classifier'].run_info['traceid_dir'])
        clfs[fov]['contours'] = util.get_roi_contours(clfs[fov]['rois'], roi_axis=-1)

#    clfs[fov]['classifier'] = copy.copy(C.classifiers[0])
#    with open(clfs_fpath, 'wb') as f:
#        pkl.dump(clfs, f, protocol=pkl.HIGHEST_PROTOCOL)
        
    # Plot RFE results from original CLFs created at start:
    # -------------------------------------------------------------------------
    if not os.path.exists(os.path.join(clf_output_dir, 'best_RFE_masks.png')):
        if len(clfs.keys()) == 1:
            fig, ax = pl.subplots(1, figsize=(10,10))
            ax.imshow(clfs[fov]['fov'])
            ax.set_title(fov)
            ax.axis('off')
            util.plot_roi_contours(clfs[fov]['fov'], clfs[fov]['contours'], clip_limit=0.2, ax=ax, roi_highlight=clfs[fov]['RFE']['best']['kept_rids'], label_highlight=True, fontsize=8)
        
        else:
            fig, axes = pl.subplots(nrows=1, ncols=len(clfs.keys()), figsize=(20, 5))
            for fi, fov in enumerate(sorted(clfs.keys(), key=natural_keys)):
                axes[fi].imshow(clfs[fov]['fov'])
                axes[fi].set_title(fov)
                axes[fi].axis('off')
                util.plot_roi_contours(clfs[fov]['fov'], clfs[fov]['contours'], clip_limit=0.2, ax=axes[fi], roi_highlight=clfs[fov]['RFE']['best']['kept_rids'], label_highlight=True, fontsize=8)
            
        pl.savefig(os.path.join(clf_output_dir, 'best_RFE_masks.png'))
        pl.close()
        
#%%
    # =============================================================================
    # TEST the trained classifier -- TRANSFORMATIONS.
    # =============================================================================
    no_morphs = True

    middle_morph = None #53
    m100 = None #106 #max(clf.clfparams['class_subset'])
    if no_morphs:
        nomorph_str = '_nomorphs'
    else:
        nomorph_str = ''
    # =============================================================================

    const_trans_types = clfs[clfs.keys()[0]]['classifier'].clfparams['const_trans']
    if const_trans_types == '' or len(const_trans_types) == 1:
        trans0 = 'yrot' #clf.clfparams['const_trans'][0]
        trans1 = None #clf.clfparams['const_trans'][0]
    else:
        trans0 = const_trans_types[0]
        trans1 = const_trans_types[1]
    print "TRANS0: %s, TRANS1: %s" % (trans0, trans1)
    
    # Create secondary output dir to save all CV results from each individual fov
    # to main classifier dir:
    secondary_output_dir = os.path.join(test_transforms_dir, 'cross_validation')
    
    TRAIN, TEST = train_and_test_transforms(trans_classifiers, clfs, clf_subdir, 
                                            no_morphs=no_morphs, trans0=trans0, trans1=trans1,
                                            setC=setC, nfeatures_select=nfeatures_select,
                                            full_train=full_train, test_size=test_size, 
                                            secondary_output_dir=secondary_output_dir)

    performance_grid, counts_grid = transform_performance_grid(TEST)
            
    # Plot PERFORMANCE:
    data_identifier = '*'.join(fov_list)
    fig = plot_transform_grid(performance_grid, rowvals=TEST['rowvals'], colvals=TEST['colvals'], 
                                                ylabel=trans1, xlabel=trans0, cmap='hot', vmin=0.5, vmax=1., 
                                                data_identifier=data_identifier, ax=None)
    label_figure(fig, data_identifier)
    pl.savefig(os.path.join(test_transforms_dir, 'test_transforms_performance_grid_%s%s.png' % (train_set, nomorph_str)))
    #pl.close()
    
    # Plot COUNTS:
    fig = plot_transform_grid(counts_grid, rowvals=TEST['rowvals'], colvals=TEST['colvals'], 
                                           ylabel=trans1, xlabel=trans0, cmap='Blues_r', vmin=0, vmax=counts_grid.max(),
                                           data_identifier=data_identifier, ax=None)
    label_figure(fig, data_identifier)
    pl.savefig(os.path.join(test_transforms_dir, 'test_transforms_counts_grid_%s%s.png' % (train_set, nomorph_str)))
    #pl.close()
    
    test_set_fpath = os.path.join(test_transforms_dir, 'testset.json')
    with open(test_set_fpath, 'w') as f:
        json.dump(C.params['test_set'], f, indent=4)
        

#%%
    no_morphs = False
    # =============================================================================
    # TEST the trained classifier -- MORPH LINE.
    # =============================================================================
        
    if not no_morphs:
        
        prob_m100_list = {}
        m100 = max(clfs[clfs.keys()[0]]['classifier'].clfparams['class_subset']) #106
        
        #TRAIN = {'traindata': {}, 'testdata': {}, 'kept_rids': {}, 'svc': {}}
        #TEST = {'data': {}, 'labels': {}, 'predicted': {}}

        for fov in sorted(clfs.keys(), key=natural_keys):
            print fov
            C = trans_classifiers[fov]['C']
            clf = copy.copy(clfs[fov]['classifier'])
            svc, traindataX, testdataX, traintest_results, kept_rids = train_and_validate_best_clf(clf, setC=setC, nfeatures_select=nfeatures_select, 
                                                                                   full_train=full_train, test_size=test_size,
                                                                                   secondary_output_dir=os.path.join(test_morphs_dir, 'cross_validation'), 
                                                                                   data_identifier=clf.data_identifier)    
                
                
            #kept_rids = clfs[fov]['RFE']['best']['kept_rids']
            sample_data = C.sample_data[:, kept_rids]
            sample_labels = C.sample_labels
            sdf = pd.DataFrame(clf.sconfigs).T
            clfparams = clf.clfparams
            morph_levels = sorted(sdf[clf.clfparams['class_name']].unique())
            
#            train_config = tuple(clfparams['trans_value'])
            
            
            test_data, test_labels = get_test_data(sample_data, sample_labels, sdf, clfparams)
            trainset_sz = traindataX['data'].shape[0]
            validate_sz = 0 if testdataX['data'] is None else testdataX['data'].shape[0]
            if test_data.shape[0] + trainset_sz + validate_sz > sample_data.shape:
                print "*** @@@ ERROR @@@ ***"
                print "N train samples (%i) + N test samples (%i) >> original sample size -- Might be an overlap..."  % (trainset_sz+validate_sz+test_data.shape[0], sample_data.shape[0])
            elif test_data.shape[0] + trainset_sz + validate_sz < sample_data.shape[0]:
                print "--- warning... ---"
                print "Not all samples (%i) used:  Train set (%i) | Validate set (%i) | Test set (%i)" %  (sample_data.shape[0], trainset_sz, validate_sz, test_data.shape[0])
                
            test_predicted = svc.predict(test_data) #(test_data, fake_labels)
            

            for morph_level in morph_levels:
                if morph_level in [morph_levels[0], morph_levels[-1]]:
                    prob_m100 = (1-traintest_results[morph_level]) if morph_level==0 else traintest_results[morph_level]
                else:
                    curr_trials = [ti for ti, tchoice in enumerate(test_labels) if tchoice == morph_level]
                    print "curr morph %i (%i trials)" % (morph_level, len(curr_trials))
                    curr_choices = [test_predicted[ti] for ti in curr_trials]
                    prob_m100 = float( np.count_nonzero(curr_choices) ) / float( len(curr_trials) )
                
                if morph_level not in prob_m100_list:
                    prob_m100_list[morph_level] = []
                    
                prob_m100_list[morph_level].append(prob_m100)
            
#            TEST['data'][fov]= testdata
#            TEST['labels'][fov] = test_labels
#            TEST['predicted'][fov] = test_predicted
#            
#            TRAIN['testdata'][fov] = testdataX
#            TRAIN['traindata'][fov] = traindataX
#            TRAIN['kept_rids'] [fov] = kept_rids
#            TRAIN['svc'][fov] = svc
            
        pl.close('all')
        
        
        #%
        prob_choose_m100 = dict((k, np.mean(vals)) for k, vals in prob_m100_list.items())
        pl.figure()
        morph_choices = [prob_choose_m100[m] for m in morph_levels]
        pl.plot(morph_levels, morph_choices, 'ko')
        pl.ylim([0, 1.0])
        pl.ylabel('perc. chose %i' % m100)
        pl.xlabel('morph level')
        pl.savefig(os.path.join(test_morphs_dir, 'perc_choose_106_%s.png' % train_set))
#        
#        test_morphs_fpath = os.path.join(clf_subdir, 'TEST_clfs_morphs.pkl')
#        with open(test_morphs_fpath, 'wb') as f:
#            pkl.dump(TEST, f, protocol=pkl.HIGHEST_PROTOCOL)
#        train_morphs_fpath = os.path.join(clf_subdir, 'TRAIN_clfs_morphs.pkl')
#        with open(train_morphs_fpath, 'wb') as f:
#            pkl.dump(TRAIN, f, protocol=pkl.HIGHEST_PROTOCOL)        

#%%
    # =============================================================================
    # TEST the trained classifier -- MORPH LINE: Split test data by VIEW.
    # =============================================================================
        
    no_morphs = False
    rowvals = TEST['rowvals']
    colvals = TEST['colvals']
    
    if len(rowvals) == 0:
        rowvals = [0]
        trans1 = ''
        
    grid_pairs = sorted(list(itertools.product(rowvals, colvals)), key=lambda x: (x[0], x[1]))
    if len(grid_pairs) > 1:
        view_str = '_testview_%s' % '_'.join([trans0, trans1])
        print view_str
    else:
        view_str = ''
        
    if not no_morphs:
    
        prob_m100_list_split_trans = {}
        m100 = max(clf.clfparams['class_subset']) #106

        #TRAIN = {'traindata': {}, 'testdata': {}, 'kept_rids': {}, 'svc': {}}
        #TEST = {'data': {}, 'labels': {}, 'predicted': {}}
                
        for fov in sorted(clfs.keys(), key=natural_keys):
            print "***", fov
            C = trans_classifiers[fov]['C']
            clf = copy.copy(clfs[fov]['classifier'])
            svc, traindataX, testdataX, traintest_results, kept_rids = train_and_validate_best_clf(clf, setC=setC, nfeatures_select=nfeatures_select, 
                                                                                   full_train=full_train, test_size=test_size,
                                                                                   secondary_output_dir=os.path.join(test_morphs_dir, 'cross_validation'), 
                                                                                   data_identifier=clf.data_identifier)    
                

            train_config = tuple(clf.clfparams['trans_value'])
                
            sample_data = C.sample_data[:, kept_rids]
            sample_labels = C.sample_labels
            sdf = pd.DataFrame(clf.sconfigs).T
            clfparams = clf.clfparams
            
            for (view1,view0) in grid_pairs:
                if trans1 != '':
                    subdf = sdf[(sdf[trans0]==view0) & (sdf[trans1]==view1)]
                else:
                    subdf = sdf[(sdf[trans0]==view0)]
                if len(subdf) == 0:
                    # specified view pair does not exist in dataset
                    continue
                print "...testing:", (view1, view0)
                
                test_data, test_labels = get_test_data(sample_data, sample_labels, subdf, clfparams, limit_to_trained=False)
                test_predicted = svc.predict(test_data)

                
                morph_levels = sorted(sdf[clf.clfparams['class_name']].unique())
                for morph_level in morph_levels:
                    corresponding_config = subdf[subdf['morphlevel']==morph_level].index.tolist()[0]
                    if morph_level in [morph_levels[0], morph_levels[-1]]:
                        if corresponding_config not in traintest_results['by_config'].keys():
                            continue
                        #corresponding_config = subdf[subdf['morphlevel']==morph_level].index.tolist()[0]
                        prob_m100 = (1-traintest_results['by_config'][corresponding_config]['percent_correct']) if morph_level==0 else traintest_results['by_config'][corresponding_config]['percent_correct']
                    else:
                        curr_trials = [ti for ti, tchoice in enumerate(test_labels) if tchoice == morph_level]
                        curr_choices = [test_predicted[ti] for ti in curr_trials]
                        prob_m100 = float( np.count_nonzero(curr_choices) ) / float( len(curr_trials) )
                    
                    if morph_level not in prob_m100_list_split_trans:
                        prob_m100_list_split_trans[morph_level] = []
                        
                    prob_m100_list_split_trans[morph_level].append(prob_m100)
            
#            TEST['data'][fov]= testdata
#            TEST['labels'][fov] = test_labels
#            TEST['predicted'][fov] = test_predicted
#            
#            TRAIN['testdata'][fov] = testdataX
#            TRAIN['traindata'][fov] = traindataX
#            TRAIN['kept_rids'] [fov] = kept_rids
#            TRAIN['svc'][fov] = svc
            
        pl.close('all')
        
        #%
        prob_m100_split_means = dict((k, np.mean(vals)) for k, vals in prob_m100_list_split_trans.items())
        prob_m100_split_sems = dict((k, stats.sem(vals)) for k, vals in prob_m100_list_split_trans.items())
        
        fig, ax = pl.subplots()
        morph_choices = [prob_m100_split_means[m] for m in morph_levels]
        morph_sems = [prob_m100_split_sems[m] for m in morph_levels]
        ax.plot(morph_levels, morph_choices, 'ko')
        ax.errorbar(morph_levels, y=morph_choices, yerr=morph_sems)
        pl.ylim([0, 1.0])
        pl.ylabel('perc. chose %i' % m100)
        pl.xlabel('morph level')
        pl.savefig(os.path.join(test_morphs_dir, 'perc_choose_106_%s%s_avgviews.png' % (train_set, view_str)))
#        
#        test_morphs_fpath = os.path.join(clf_subdir, 'TEST_clfs_morphs%s.pkl' % view_str)
#        with open(test_morphs_fpath, 'wb') as f:
#            pkl.dump(TEST, f, protocol=pkl.HIGHEST_PROTOCOL)
#        train_morphs_fpath = os.path.join(clf_subdir, 'TRAIN_clfs_morphs%s.pkl' % view_str)
#        with open(train_morphs_fpath, 'wb') as f:
#            pkl.dump(TRAIN, f, protocol=pkl.HIGHEST_PROTOCOL)        

    
    
    
#%%

#def main(options):
#    optsE = extract_options
#    C = train_test_linearSVC(optsE)
#


if __name__ == '__main__':
    main(sys.argv[1:])
