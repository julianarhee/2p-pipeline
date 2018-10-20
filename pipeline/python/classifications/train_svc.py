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
import pandas as pd
import numpy as np
import multiprocessing as mp
import cPickle as pkl
import pylab as pl

from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections import Counter

from pipeline.python.classifications import linearSVC_class as lsvc
from pipeline.python.utils import print_elapsed_time, natural_keys, label_figure

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
    parser.add_option('-S', '--session', action='store', dest='session',
                          default='', help='session dir (format: YYYMMDD_ANIMALID')
    parser.add_option('-A', '--acq', action='store', dest='acquisition',
                          default='FOV1', help="acquisition folder (ex: 'FOV1_zoom3x') [default: FOV1]")
    parser.add_option('-R', '--run', action='store', dest='run',
                          default='', help="RUN name (e.g., gratings_run1)")
    parser.add_option('-t', '--traceid', action='store', dest='traceid',
                          default='', help="traceid name (e.g., traces001)")
    
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

    parser.add_option('-L', '--clf', action='store', dest='classifier', default='LinearSVC', help='Classifier type (default: LinearSVC)')
    parser.add_option('-k', '--cv', action='store', dest='cv_method', default='kfold', help='Method of cross-validation (default: kfold)')
    parser.add_option('-f', '--folds', action='store', dest='cv_nfolds', default=5, help='N folds for CV (default: 5)')
    parser.add_option('-C', '--cval', action='store', dest='C_val', default=1e9, help='Value for C param if using SVC (default: 1e9)')
    parser.add_option('-g', '--groups', action='store', dest='cv_ngroups', default=1, help='N groups for CV, relevant only for data_type=frames (default: 1)')
    parser.add_option('-b', '--bin', action='store', dest='binsize', default=10, help='Bin size, relevant only for data_type=frames (default: 10)')

    parser.add_option('--test-C', action='store', dest='setC', default=1e9, help='C value or type to use for final trained classifier')
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


def get_transform_classifiers(optsE):

    nprocs = int(optsE.nprocesses)
    
    C = lsvc.TransformClassifier(optsE.animalid, optsE.session, optsE.acquisition, optsE.run, optsE.traceid,
                                rootdir=optsE.rootdir, roi_selector=optsE.roi_selector, data_type=optsE.data_type,
                                stat_type=optsE.stat_type, inputdata_type=optsE.inputdata_type,
                                get_null=optsE.get_null, class_name=optsE.class_name, class_subset=optsE.class_subset,
                                const_trans=optsE.const_trans, trans_value=optsE.trans_value, 
                                cv_method=optsE.cv_method, cv_nfolds=optsE.cv_nfolds, cv_ngroups=optsE.cv_ngroups, binsize=optsE.binsize)
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

def get_traceid_dir(animalid, session, acquisition, run, traceid, rootdir='/n/coxfs01/2p-data'):
    found_traceid_dirs = sorted(glob.glob(os.path.join(rootdir, animalid, session, acquisition, run, 'traces', '%s*' % traceid)), key=natural_keys)
    assert len(found_traceid_dirs) > 0, "No traceids found."
    for ti, tdir in enumerate(found_traceid_dirs):
        print ti, os.path.split(tdir)[-1]
    sel = input("Select IDX of traceid to use: ")
    traceid_dir = found_traceid_dirs[int(sel)]
    
    return traceid_dir

def load_classifier_object(traceid_dir, rootdir='/n/coxfs01/2p-data'):
    C=None
    found_transform_classifiers = sorted(glob.glob(os.path.join(traceid_dir, 'classifiers', 'LinearSVC*', 'TransformClassifier.pkl')), key=natural_keys)
    assert len(found_transform_classifiers) > 0, "No classifiers found."
    for ti, tclassifier in enumerate(found_transform_classifiers):
        print ti, os.path.split(os.path.split(tclassifier)[0])[-1]
    sel = raw_input("Select IDX of TransformClassifier object to load: ")
    if len(sel) > 0:
        tclf_fpath = found_transform_classifiers[int(sel)]
    
        with open(tclf_fpath, 'rb') as f:
            C = pkl.load(f)
        if isinstance(C.dataset, str):
            loaded_dset = np.load(C.dataset)
            C.dataset = loaded_dset
        
    return C


def get_RFE_results(C):
    '''For each classifier trained in TransformClassifier object (C), get RFE resuults
    '''
    rfe_results={}
    for clf in C.classifiers:
        clf_key = os.path.split(clf.classifier_dir)[-1]
        rfe_results[clf_key] = load_RFE_results(clf)
    
    for clf_key, res in rfe_results.items():
        rfe_scores = np.array([np.mean(cv_scores) for ri, cv_scores in enumerate(res['results'])])
        best_iter = rfe_scores.argmax()
        kept_rids = res['kept_rids_by_iter'][best_iter-1]
        bestN = len(kept_rids)
        print clf_key, "Best score: %.2f" % rfe_scores.max(), "N rois: %i" % bestN
        print " --->", kept_rids
        rfe_results[clf_key]['best'] = {'iter_ix': best_iter,
                                'kept_rids': kept_rids,
                                'max_score': rfe_scores.max()}
    
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

def find_best_classifier(cX, cy, start_params, setC='best', nfeatures_select='best', 
                             full_train=True, test_size=0.33, classifier_base_dir='/tmp',
                             rfe_results={}):
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


def train_and_validate(svc, cX, cy, clf, clf_output_dir, full_train=False, test_size=0.33):
    #nfolds = clf.clfparams['cv_nfolds']
    #kfold = StratifiedKFold(n_splits=nfolds, shuffle=True)
    
    
    # Do cross-validation and plot results:
    # -------------------------------------------------------------------------
    
    # Do CV in limited dataset:
    fig = lsvc.hist_cv_permutations(svc, cX, cy, clf.clfparams)
    label_figure(fig, clf.data_identifier)
    pl.savefig(os.path.join(clf_output_dir, 'figures', 'cv_permutation.png'))
        
    #% Confusion matrix:
    predicted, true, classes = lsvc.get_cv_folds(svc, clf.clfparams, cX, cy, output_dir=None)
    cmatrix, cstr = lsvc.get_confusion_matrix(predicted, true, classes, average_iters=True)
    lsvc.plot_confusion_matrix(cmatrix, classes, ax=None, normalize=True, title='%s conf matrix (n=%i)' % (cstr, clf.clfparams['cv_nfolds']), cmap=pl.cm.Blues)
    pl.savefig(os.path.join(clf_output_dir, 'figures', 'cv_confusion.png'))
    
    # 
    # =============================================================================
    # Train the classifier with specified params:
    # =============================================================================
    
    trained_classes = {}
    if full_train:
        svc.fit(cX, cy)
        # TODO: fix this, can't just use all cv test results -- do it confusion-matrix style and get totals for each tested class
        for classix, classname in enumerate(classes):
            ncorrect = cmatrix[classix,classix]
            ntotal = np.sum(cmatrix[classix, :])
            trained_classes[classname] = float(ncorrect) / float(ntotal)
        
    else:
        X_train, X_test, train_true, test_true = train_test_split(cX, cy, test_size=test_size, random_state=0, shuffle=True)
        
        svc.fit(X_train, train_true)
        test_predicted = svc.predict(X_test)
        #test_score = svc.score(X_test, test_true)
        for classix, classname in enumerate(sorted(np.unique(test_true))):
            ncorrect = np.sum([1 if p == classname else 0 for p in test_predicted])
            ntotal = len(test_predicted)
            trained_classes[classname] = float(ncorrect) / float(ntotal)

    return svc, trained_classes


def get_test_data(sample_data, sample_labels, sdf, clfparams):
    #sdf = pd.DataFrame(clf.sconfigs).T
    
    # Filter sconfigs by const-trans/trans-value pair:
    if clfparams['const_trans'] != '':
        trans_sdf = sdf[sdf[clfparams['const_trans'][0]]==clfparams['trans_value'][0]]
    else:
        trans_sdf = copy.copy(sdf)
    test_values = [val for val in trans_sdf[clfparams['class_name']].unique() if val not in clfparams['class_subset']]
    
    test_sdf = trans_sdf[trans_sdf[clfparams['class_name']].isin(test_values)]
    configs_included = test_sdf.index.tolist()
    
    kept_trial_ixs = np.array([fi for fi, config in enumerate(sample_labels) if config in configs_included])
    
    test_data = sample_data[kept_trial_ixs, :]
    tmp_test_labels = sample_labels[kept_trial_ixs]
    test_labels = [test_sdf[test_sdf.index==config][clfparams['class_name']][0] for config in tmp_test_labels]
    
    print "Test data:", test_data.shape
    print "Test labels:", list(set(test_labels))

    return test_data, test_labels


#%%
    
options = ['-D', '/n/coxfs01/2p-data', '-i', 'JC022', '-S', '20181016', '-A', 'FOV1_zoom2p7x',
           '-R', 'combined_blobs_static', '-t', 'traces001',
           '-r', 'visual', '-d', 'stat', '-s', 'zscore',
           '-p', 'corrected', '-N', 'morphlevel',
           '--subset', '0,97',
           '--nproc=1'
           ]
   
#%%

def main(options):
    optsE = extract_options(options)
    
    traceid_dir = get_traceid_dir(optsE.animalid, optsE.session, optsE.acquisition, optsE.run, optsE.traceid)
    
    C = load_classifier_object(traceid_dir)
    
    if C is None:
        C = get_transform_classifiers(optsE)
    
    if len(C.classifiers) > 1:
        # Const-trans/ trans-value pairs were separately trained.
        print "-----------------------------------------------------"
        print "[%s]-classifier was trained at separate values of %s" % (C.params['class_name'], C.params['const_trans'])
        for ci, clf in enumerate(C.classifiers):
            print [('IDX: %i' % ci, '%s: %i' % (trans, val)) for (trans, val) in zip(clf.clfparams['const_trans'], clf.clfparams['trans_value'])]

    rfe_results = get_RFE_results(C)
    
    #%%
    
    # Look at specific classifier:
    setC='best'
    nfeatures_select='best'
    full_train=False
    test_size=0.2
    
    curr_clf = rfe_results.keys()[0]
    
    if C.params['const_trans'] != '':
        clf = copy.copy([c for c in C.classifiers if curr_clf.split('_')[0] in c.clfparams['const_trans'] and float(curr_clf.split('_')[1].replace('n', '-')) in c.clfparams['trans_value']][0])
    else:
        clf = C.classifiers[0]
    
    # =============================================================================
    # Train the classifier:
    # =============================================================================
    setC = optsE.setC #'best' # choices: 'best' (test diff values of C on log scale), 'big' (1E9), 'small' (w.0)
    nfeatures_select = int(optsE.nfeatures_select) if optsE.nfeatures_select.isdigit() else optsE.nfeatures_select # 'best' # choices: 'best' (uses RFE to find cells with max. clf accuracy), int (uses RFE to find top N cells)
    full_train = optsE.full_train #True
    test_size = float(optsE.test_size) #0.33
    
    if full_train:
        train_set = 'fulltrain'
    else:
        train_set = 'testsize%.2f' % test_size
        
    svc, cX, cy, kept_rids, clf_output_dir = find_best_classifier(clf.cX, clf.cy, clf.clf.get_params(), 
                                                                      setC=setC, 
                                                                      nfeatures_select=nfeatures_select,
                                                                      rfe_results=rfe_results[curr_clf],
                                                                      full_train=full_train, 
                                                                      test_size=test_size, 
                                                                      classifier_base_dir=clf.classifier_dir)
    
    svc, trained_classes = train_and_validate(svc, cX, cy, clf, clf_output_dir, full_train=full_train, test_size=test_size)
    
    
    #%%  
    # =============================================================================
    # TEST the trained classifier:
    # =============================================================================
    
    sample_data = C.sample_data[:, kept_rids]
    sample_labels = C.sample_labels
    sdf = pd.DataFrame(clf.sconfigs).T
    
    test_data, test_labels = get_test_data(sample_data, sample_labels, sdf, clf.clfparams)
    
    
    test_choices = svc.predict(test_data) #(test_data, fake_labels)
    prob_choose_m100 = {}
    m100 = 106
    # Plot % correct from test choices:
    morph_levels = sorted(sdf[clf.clfparams['class_name']].unique())
    for morph_level in morph_levels:
        if morph_level in [morph_levels[0], morph_levels[-1]]:
            prob_choose_m100[morph_level] = (1-trained_classes[morph_level]) if morph_level==0 else trained_classes[morph_level]
        else:
            curr_trials = [ti for ti, tchoice in enumerate(test_labels) if tchoice == morph_level]
            curr_choices = [test_choices[ti] for ti in curr_trials]
            prob_choose_m100[morph_level] = float( np.count_nonzero(curr_choices) ) / float( len(curr_choices) )
    
    
    pl.figure()
    morph_choices = [prob_choose_m100[m] for m in morph_levels]
    pl.plot(morph_levels, morph_choices, 'ko')
    pl.ylim([0, 1.0])
    pl.ylabel('perc. chose 106')
    pl.xlabel('morph level')
    pl.savefig(os.path.join(clf_output_dir, 'figures', 'perc_choose_106_%s.png' % train_set))
    
    clf_save_fpath = os.path.join(clf_output_dir, 'classifier_results.npz')
    np.savez(clf_save_fpath, svc=svc, cX=cX, cy=cy, kept_rids=kept_rids, visual_rois=C.rois, m100=m100,
             prob_choose_m100=prob_choose_m100, levels=morph_levels, class_name=clf.clfparams['class_name'],
             test_data=test_data, test_labels=test_labels)
    

#%%

#def main(options):
#    optsE = extract_options
#    C = train_test_linearSVC(optsE)
#


if __name__ == '__main__':
    main(sys.argv[1:])
