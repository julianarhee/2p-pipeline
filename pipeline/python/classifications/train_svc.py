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
    parser.add_option('--segment', action='store_true', dest='select_visual_area', default=False, help="set if selecting subset of FOV for visual area")


    parser.add_option('-L', '--clf', action='store', dest='classifier', default='LinearSVC', help='Classifier type (default: LinearSVC)')
    parser.add_option('-k', '--cv', action='store', dest='cv_method', default='kfold', help='Method of cross-validation (default: kfold)')
    parser.add_option('-f', '--folds', action='store', dest='cv_nfolds', default=5, help='N folds for CV (default: 5)')
    parser.add_option('-C', '--cval', action='store', dest='C_val', default=1e9, help='Value for C param if using SVC (default: 1e9, set to best to search)')
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


    
def initialize_classifiers_for_each_fov(traceid_dirs, optsE):
    visual_area = optsE.visual_area
    print "VISUAL AREA: %s" % visual_area
    select_visual_area = optsE.select_visual_area

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


    # Combine data arrays -- just  use stim-period zscore? (then dont have to deal w/ different trial structures)
    trans_classifiers = dict()
    for fov, traceid_dir in sorted(traceid_dirs.items(), key=lambda x: x[0]):
        
        print "Getting TransformClassifiers() for: %s" % fov
        curr_session = fov.split('_')[0]
        curr_fov = '_'.join(fov.split('_')[1:3])
        curr_traceid = os.path.split(traceid_dir)[-1].split('_')[0]
        curr_run = os.path.split(traceid_dir.split('/traces/')[0])[-1]
        print "SESSION: %s | FOV: %s | RUN: %s | traceid: %s" % (curr_session, curr_fov, curr_run, curr_traceid)
#        C = load_classifier_object(traceid_dir, rootdir=optsE.rootdir, animalid=optsE.animalid, session=curr_session)
#        print C
        
#        if C is None:
        C = initialize_transform_classifiers(optsE.animalid, curr_session, curr_fov, curr_run, curr_traceid, rootdir=optsE.rootdir,
                                      roi_selector=optsE.roi_selector, data_type=optsE.data_type, stat_type=optsE.stat_type,
                                      inputdata_type=optsE.inputdata_type, 
                                      get_null=optsE.get_null, class_name=optsE.class_name, class_subset=optsE.class_subset,
                                      const_trans=optsE.const_trans, trans_value=optsE.trans_value,
                                      cv_method=optsE.cv_method, cv_nfolds=optsE.cv_nfolds, cv_ngroups=optsE.cv_ngroups, 
                                      C_val=optsE.C_val, binsize=optsE.binsize, nprocesses=optsE.nprocesses, test_set=test_set, 
                                      indie=optsE.indie, select_visual_area=select_visual_area, visual_area=visual_area)
        trans_classifiers[fov] = {'C': C, 'traceid_dir': traceid_dir}
        
    return trans_classifiers 



def initialize_transform_classifiers(animalid, session, acquisition, run, traceid, rootdir='/n/coxfs01/2p-data',
                         roi_selector='visual', data_type='stat', stat_type='meanstim',
                         inputdata_type='corrected', 
                         get_null=False, class_name='', class_subset='',
                         const_trans='', trans_value='', test_set=[], indie=False,
                         cv_method='kfold', cv_nfolds=5, cv_ngroups=1, C_val=1e9, binsize=10,
                         nprocesses=2, select_visual_area=False, visual_area=''):

    nprocs = int(nprocesses)
    
    C = lsvc.TransformClassifier(animalid, session, acquisition, run, traceid, rootdir=rootdir,
                         roi_selector=roi_selector, data_type=data_type, stat_type=stat_type,
                         inputdata_type=inputdata_type, 
                         get_null=get_null, class_name=class_name, class_subset=class_subset,
                         const_trans=const_trans, trans_value=trans_value, test_set=test_set, indie=indie,
                         cv_method=cv_method, cv_nfolds=cv_nfolds, cv_ngroups=cv_ngroups, C_val=C_val, binsize=binsize)
    
    if select_visual_area:
        visual_areas_fpath = glob.glob(os.path.join(C.rootdir, C.animalid, C.session, C.acquisition, 'visual_areas', 'visual_areas_*.pkl'))[0]
        visual_area_info = {visual_area: visual_areas_fpath}
    else:
        visual_area_info = None

    C.load_dataset(visual_area_info=visual_area_info)
    C.create_classifier_dirs()
    
    return C

#
#def create_transform_classifiers(C, feature_select_method='rfe', feature_select_n='best', C_select='best'):
#    ##
#    '''
#    feature_select_method = 'rfe' # Choices: 'rfe', 'k_best', None
#    feature_select_n = 'best' # Choices:  'best', int, None
#    C_select = 'best' # Choices: 'best', None
#    '''
#    
#    C.create_classifiers(feature_select_method=feature_select_method, 
#                             feature_select_n=feature_select_n, 
#                             C_select=C_select)
    
#    #
#    C.label_classifier_data()
#    C.create_classifiers()
    
#def train_transform_classifiers(scoring='accuracy', full_train=False, test_size=0.33):
#    for clf in C.classifiers:
#        
#        clf.do_cv(scoring=scoring, permutation_test=True, n_jobs=4, n_permutations=500)
#
#        clf.train_classifier(full_train=full_train, test_size=test_size)
#        
#        clf.get_classifier_accuracy_by_stimconfig(full_train=full_train, test_size=test_size, row_label=None, col_label=None)
#
#    return C


#    t_train_mp = time.time()
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

#    for clf in C.classifiers:
#        clf.do_cv()
#        with open(os.path.join(clf.classifier_dir, 'classifier_%s.pkl' % clf.hash), 'wb') as f:
#            pkl.dump(clf, f, protocol=pkl.HIGHEST_PROTOCOL)
#        print "Saved object to:", os.path.join(C.classifier_dir, os.path.join(clf.classifier_dir, 'classifier_%s.pkl' % clf.hash)) #'TransformClassifier.pkl')
#
#    # Replace 'dataset':
#    C.dataset = copy.copy(C.data_fpath)
#    with open(os.path.join(C.classifier_dir, 'TransformClassifier.pkl'), 'wb') as f:
#        pkl.dump(C, f, protocol=pkl.HIGHEST_PROTOCOL)
#        
#    print "*****************************"
#    print "Finished training all %i classifiers." % len(C.classifiers)
#    print "*****************************"
#   
#    # Reload dataset to return:
#    tmp_dset = np.load(C.data_fpath)
#    C.dataset = tmp_dset
#    
#    return C

#
#T = lsvc.TransformClassifier(optsE.animalid, curr_session, curr_fov, curr_run, curr_traceid, rootdir=optsE.rootdir,
#                                          roi_selector=optsE.roi_selector, data_type=optsE.data_type, stat_type=optsE.stat_type,
#                                          inputdata_type=optsE.inputdata_type, 
#                                          get_null=optsE.get_null, class_name=optsE.class_name, class_subset=optsE.class_subset,
#                                          const_trans=optsE.const_trans, trans_value=optsE.trans_value,
#                                          cv_method=optsE.cv_method, cv_nfolds=optsE.cv_nfolds, cv_ngroups=optsE.cv_ngroups, 
#                                          C_val=optsE.C_val, binsize=optsE.binsize, test_set=test_set, indie=optsE.indie
#                                          )
#
##visual_areas_fpath = glob.glob(os.path.join(T.rootdir, T.animalid, T.session, T.acquisition, 'visual_areas', 'visual_areas_*.pkl'))[0]
##visual_area_info = {visual_area: visual_areas_fpath}
#visual_area_info = None
#
#T.load_dataset(visual_area_info=visual_area_info)
#T.create_classifier_dirs()
#
###
#feature_select_method = 'rfe' # Choices: 'rfe', 'k_best', None
#feature_select_n = 'best' # Choices:  'best', int, None
#C_select = 'best' # Choices: 'best', None
#
#T.initialize_classifiers(feature_select_method=feature_select_method, 
#                         feature_select_n=feature_select_n, 
#                         C_select=C_select)
#
##
##T.label_classifier_data()
#T.create_classifiers()
#
#
#clf = T.classifiers[0]
#
#clf.do_cv(scoring='accuracy', permutation_test=True, n_jobs=4, n_permutations=500)
#
#full_train = False
#test_size=0.33
#clf.train_classifier(full_train=full_train, test_size=test_size)
#
#clf.get_classifier_accuracy_by_stimconfig()


#%%


def get_test_data(sample_data, sample_labels, sdf, clfparams, limit_to_trained_views=True):
    #sdf = pd.DataFrame(clf.sconfigs).T
    
    # Filter sconfigs by const-trans/trans-value pair:
    if clfparams['const_trans'] != '' and limit_to_trained_views is True:
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



#glob.glob(os.path.join(optsE.rootdir, optsE.animalid, '2018*', 'FOV*', '*blobs*', 'traces', 'traces*', 'classifiers'))


    
        #%%
           
#def train_and_test_transforms(trans_classifiers, clfs, clf_subdir, setC='big', nfeatures_select='best', 
#                              full_train=False, test_size=0.2, secondary_output_dir=None,
#                              no_morphs=False, trans0=None, trans1=None, middle_morph=53, m100=106):
             
def test_classifier_on_transforms(clf, test_data, test_labels, clf_subdir, trans0=None, trans1=None, no_morphs=True):
    if no_morphs:
        nomorph_str = '_nomorphs'
    else:
        nomorph_str = ''
    
    print "trans:", trans0, "trans:", trans1
    
    if trans0 is None:
        trans0 = clf.clfparams['const_trans'][0]
    if trans1 is None and len(clf.clfparams['const_trans']) > 1:
        trans1 = clf.clfparams['const_trans'][1]

    class_name = clf.clfparams['class_name']
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
    tmp_data = test_data.copy()
    tmp_labels = test_labels.copy()
    
    # Test classifier:
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
    
    TEST['by_config'] = {'accuracy': performance, 'counts': counts, 'ncorrect': ncorrect}
    TEST['predicted'] = dict((tconfig, tdata['predicted']) for tconfig, tdata in testdata.items())
    TEST['data'] = dict((tconfig, tdata['data']) for tconfig, tdata in testdata.items())
    TEST['labels'] = dict((tconfig, tdata['labels']) for tconfig, tdata in testdata.items())

    TRAIN['testdata'] = testdataX
    TRAIN['traindata'] = traindataX
    TRAIN['kept_rids'] = kept_rids
    TRAIN['svc'] = svc

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


#%%

def get_morph_percents_m100(trans_classifiers, clfs, setC='big', nfeatures_select='best', full_train=False, test_size=0.2, secondary_output_dir=None):
 
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
        
    if average_across_view:
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
   
        else: 
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
        
    return prob_m100_list
     

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
# LM -- gratings (5x4)
# =============================================================================

# Decent:
#options = ['-D', rootdir, '-i', 'JC015', 
#           '-S', '20180915,20180917', 
#           '-A', 'FOV1_zoom2p7x,FOV1_zoom2p0x',
#           '-R', 'combined_gratings_static,combined_gratings_static',
#           '-t', 'traces002,traces002',
#           '-r', 'visual', '-d', 'stat', '-s', 'zscore',
#           '-p', 'corrected', 
#           '-N', 'ori',
##           '--subset', '0,180',
##           '--indie',
#           '-c', 'xpos,ypos',
##           '-v', '-5,0',
##           '-T', '-15,-10,0,5',
##           '-T', '-60,-30,30,60',
#           '-C', 'best', 
#           '-V', 'LM',
##           '--segment',
#           '-T', '-10,-10,-20,-20',
#           '-T', '-5,-15,-5,-15',
#           
#           '--nproc=1'
#           ]

# LM/I -- gratings (6x4)
# -----------------------------------------------------
#options = ['-D', rootdir, '-i', 'JC015', 
#           '-S', '20180919,20180924,20180925', 
#           '-A', 'FOV1_zoom2p0x,FOV1_zoom2p0x,FOV1_zoom2p0x',
#           '-R', 'combined_gratings_static,gratings_run1,combined_gratings_static',
#           '-t', 'traces003,traces002,traces003',
#           '-r', 'visual', '-d', 'stat', '-s', 'zscore',
#           '-p', 'corrected', 
#           '-N', 'ori',
##           '--subset', '0,106',
#           '--indie',
#           '-c', 'xpos,ypos',
##           '-v', '-5,0',
##           '-T', '-15,-10,0,5',
##           '-T', '-60,-30,30,60',
#           '-C', 'best', 
#           '-V', 'LM',
##           '--segment',
##           '-T', '5.6,16.8,28,5.6,16.8,28',
##           '-T', '5,5,5,15,15,15',
#           
#           '--nproc=1'
#           ]

#options = ['-D', rootdir, '-i', 'JC015', 
#           '-S', '20180919,20180924', 
#           '-A', 'FOV1_zoom2p0x,FOV1_zoom2p0x',
#           '-R', 'combined_gratings_static,gratings_run1',
#           '-t', 'traces003,traces002',
#           '-r', 'visual', '-d', 'stat', '-s', 'zscore',
#           '-p', 'corrected', 
#           '-N', 'ori',
##           '--subset', '0,106',
##           '--indie',
#           '-c', 'xpos,ypos',
##           '-v', '-5,0',
##           '-T', '-15,-10,0,5',
##           '-T', '-60,-30,30,60',
#           '-C', 'best', 
#           '-V', 'LM',
##           '--segment',
#           '-T', '16.8,16.8,28,28',
#           '-T', '5,15,5,15',
##           '-T', '16.8,16.8,28,28',
##           '-T', '5,15,5,15',
#           
#           '--nproc=1'
#           ]


# LM/I -- blobs (6x4)
# -----------------------------------------------------
#options = ['-D', rootdir, '-i', 'JC015', 
#           '-S', '20180919,20180924', 
#           '-A', 'FOV1_zoom2p0x,FOV1_zoom2p0x',
#           '-R', 'combined_blobs_static,combined_blobs_static',
#           '-t', 'traces003,traces002',
#           '-r', 'visual', '-d', 'stat', '-s', 'zscore',
#           '-p', 'corrected', 
#           '-N', 'morphlevel',
#           '--subset', '0,106',
#           '--indie',
#           '-c', 'xpos,ypos',
##           '-v', '-5,0',
##           '-T', '-15,-10,0,5',
##           '-T', '-60,-30,30,60',
#           '-C', 'best', 
#           '-V', 'LM',
##           '--segment',
##           '-T', '5.6,16.8,28,5.6,16.8,28',
##           '-T', '5,5,5,15,15,15',
#           
#           '--nproc=1'
#           ]





# =============================================================================
# LI -- gratings (6x4)
# =============================================================================
#options = ['-D', rootdir, '-i', 'JC015', 
#           '-S', '20180924,20180925', 
#           '-A', 'FOV1_zoom2p0x,FOV1_zoom2p0x',
#           '-R', 'gratings_run1,combined_gratings_static',
#           '-t', 'traces002,traces003',
#           '-r', 'visual', '-d', 'stat', '-s', 'zscore',
#           '-p', 'corrected', 
#           '-N', 'ori',
##           '--subset', '0,106',
##           '-c', 'xpos,ypos',
##           '-v', '-5,0',
##           '-T', '-15,-10,0,5',
##           '-T', '-60,-30,30,60',
#           '-V', 'LM-LI',
##           '--segment',
##           '-T', '5.6,16.8,28,5.6,16.8,28',
##           '-T', '5,5,5,15,15,15',
#           
#           '--nproc=1'
#           ]

options = ['-D', rootdir, '-i', 'JC022', 
           '-S', '20181005,20181005,20181006,20181007,20181017', 
           '-A', 'FOV2_zoom2p7x,FOV3_zoom2p7x,FOV1_zoom2p7x,FOV1_zoom2p2x,FOV1_zoom2p7x',
           '-R', 'combined_gratings_static,combined_gratings_static,combined_gratings_static,combined_gratings_static,combined_gratings_static',
           '-t', 'traces001,traces001,traces001,traces001,traces001',
           '-r', 'visual', '-d', 'stat', '-s', 'zscore',
           '-p', 'corrected', 
           '-N', 'position',
#           '--subset', '0,106',
#           '-c', 'xpos,ypos',
#           '-v', '-5,0',
#           '-T', '-15,-10,0,5',
#           '-T', '-60,-30,30,60',
#           '-T', '5.6,16.8,28,5.6,16.8,28',
#           '-T', '5,5,5,15,15,15',
           '-V', 'LI',
           '--nproc=1'
           ]
# =============================================================================


# =============================================================================
# LI - BLOBS 5x5x5
# =============================================================================

#options = ['-D', rootdir, '-i', 'JC022', 
#           '-S', '20181022',
#           '-A', 'FOV1_zoom4p0x',
#           '-R', 'combined_blobs_static', 
#           '-t', 'traces001',
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
#                                          C_val=optsE.C_val, binsize=optsE.binsize, test_set=test_set, indie=optsE.indie
#                                          )
#
#visual_areas_fpath = glob.glob(os.path.join(T.rootdir, T.animalid, T.session, T.acquisition, 'visual_areas', 'visual_areas_*.pkl'))[0]
#visual_area_info = {visual_area: visual_areas_fpath}
#
#T.load_dataset(visual_area_info=visual_area_info)
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

    # -------------------------------------------------------------------------
    # MODEL SELECTION PARAMS:
    feature_select_method='rfe'
    feature_select_n='best'
    C_select='best'
    # -------------------------------------------------------------------------
    
    
    optsE = extract_options(options)
    visual_area = optsE.visual_area

    traceid_dirs = get_traceids_from_lists(optsE.animalid, optsE.session_list, optsE.fov_list, optsE.run_list, optsE.traceid_list, rootdir=rootdir)
    trans_classifiers = initialize_classifiers_for_each_fov(traceid_dirs, optsE)

    # Create output dir(s) for all classifiers for this visual area:
    # -------------------------------------------------------------------------
    fov_list = sorted(trans_classifiers.keys(), key=natural_keys)
    grouped_fovs = '_'.join(sorted(fov_list, key=natural_keys))
    print grouped_fovs
    output_dir = os.path.join(optsE.rootdir, optsE.animalid, visual_area, grouped_fovs)
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    print "-- Saving all current %s output to:\n%s" % (visual_area, output_dir)
    
    clf_names = list(set( [os.path.split(trans_classifier['C'].classifier_dir)[-1] for fov, trans_classifier in trans_classifiers.items()] ))
    if len(clf_names) > 1:
        print "*** WARNING *** Looks like you're trying to combine different types of classifiers:\n    %s" % clf_names
        for ci,cname in enumerate(clf_names):
            print ci, cname
        user_choice = input("Select IDX of name to use: ")
        clf_name = clf_names[int(user_choice)]
    else:
        clf_name = clf_names[0]
    
    meta_params_str = '%s_%s_C_%s' % ('all' if feature_select_method is None else feature_select_method,
                                      'features' if feature_select_n is None else str(feature_select_n),
                                      C_select if isinstance(C_select, str) else '%.4f' % optsE.C_val)
    print meta_params_str
    
    clf_output_dir = os.path.join(output_dir, 'classifiers', clf_name, meta_params_str)
    if not os.path.exists(clf_output_dir): os.makedirs(clf_output_dir)
    
    

#%
    print "*******************************************************************"
    print "Created TransformClassifier() object for visual area: %s" % visual_area
    print "Saving joint TransformClassifier output to:\n    -->%s" % clf_output_dir
    for fov, trans_classifier in sorted(trans_classifiers.items(), key=lambda x: x[0]):
        print fov
        # Update classifer dir to GROUPED dir:
        trans_classifier['C'].classifier_dir = clf_output_dir  
        trans_classifier['C'].create_classifiers(feature_select_method='rfe', feature_select_n='best', C_select='best')
    print "Creating %i LinearSVM() objects for each of %i FOVs." % (len(trans_classifiers[trans_classifiers.keys()[0]]['C'].classifiers), len(trans_classifiers.keys()))
    print "*******************************************************************"


    #%%

    # -------------------------------------------------------------------------
    # TRAINING PARAMS:
    scoring = 'accuracy'
    full_train = True #False
    test_size = 0.0 #0.33
    col_label = 'xpos'
    row_label = 'ypos'
    # -------------------------------------------------------------------------

    print "*******************************************************************"
    print "TRAINING."
    for fov, trans_classifier in trans_classifiers.items():
        trans_classifier['C'].train_classifiers(scoring=scoring, full_train=full_train, test_size=test_size, col_label=col_label, row_label=row_label)

    print "*******************************************************************"

    # Save trained classifier to disk, replace dataset npz object with filepath:
    for fov, trans_classifier in trans_classifiers.items():
        trans_classifiers_fpath = os.path.join(clf_output_dir, 'transform_classifiers_%s.pkl' % fov)
        trans_classifier['C'].dataset = trans_classifier['C'].data_fpath
        
    trans_classifiers_fpath = os.path.join(clf_output_dir, 'transform_classifiers.pkl')
    with open(trans_classifiers_fpath, 'wb') as f:
        pkl.dump(trans_classifiers, f, protocol=pkl.HIGHEST_PROTOCOL)
    

    #%% 
    print "*******************************************************************"
    print "TESTING classifier:"
    
    #% Plot accuracy by stim-config across all FOVs:
    data_identifier = '_'.join(sorted(trans_classifiers.keys(), key=natural_keys))
    print data_identifier

#            test_transforms_dir = os.path.join(self.classifier_dir, 'test_transforms')
#            if not os.path.exists(test_transforms_dir): os.makedirs(test_transforms_dir)
#            clf.get_classifier_accuracy_by_stimconfig(row_label=row_label, col_label=col_label, output_dir=test_transforms_dir)
    
    nr = 4
    nc = 5
    accuracy_grid = np.zeros((nr, nc))
    counts_grid = np.zeros((nr, nc))
    nclfs = 0
    for fov, trans_classifier in sorted(trans_classifiers.items(), key=lambda x: x[0]):

        for cix in range(len(trans_classifier['C'].classifiers)):
            print "%s:  Testing %i of %i classifiers in." % (fov, (cix+1), len(trans_classifier['C'].classifiers))
                
            clf = trans_classifier['C'].classifiers[cix]
            kept_rids = clf.model_selection.features['kept_rids']
                    
            sample_data = trans_classifier['C'].sample_data[:, kept_rids]
            sample_labels = trans_classifier['C'].sample_labels
            sdf = pd.DataFrame(trans_classifier['C'].sconfigs).T
            clfparams = trans_classifier['C'].classifiers[cix].clfparams
            
            if clf.clfparams['const_trans'] != '':
                # Get all non-class transform types (xpos, ypos, etc.):
                trans_types = [trans_type for trans_type in clf.run_info['trans_types'] if trans_type != clfparams['class_name']]
                all_transforms = dict((trans, list(sdf[trans].unique())) for trans in trans_types)
                all_configs_list = [dict(zip(clfparams['const_trans'], v)) for v in itertools.product(*all_transforms.values())]
                
                # Separate all transform combos into trained set and test set:
                train_configs = [dict((trans, val) for trans, val in zip(clfparams['const_trans'], clfparams['trans_value']))]
                if isinstance(clfparams['trans_value'][0], list) and \
                    any([len(clfparams['trans_value'][t]) > 1 for t in range(len(clfparams['trans_value']))]) \
                    and len(train_configs)==1:
                    
                    keys, vals = zip(*train_configs[0].items())
                    tconfigs = [dict(zip(keys, v)) for v in itertools.product(*vals)]
                    train_configs = [tdict for tdict in tconfigs if tdict not in clfparams['test_set']]
                    
                    
                test_configs = [tdict for tdict in all_configs_list if tdict not in train_configs]
                print "N train configs: %i, N test configs: %i" % (len(train_configs), len(test_configs))
        
                # Get correponding config names (config00x) for each stim config in train and test sets:
                train_config_names = [cfg for cfg, stim in clf.sconfigs.items() if all([stim[trans]==val for tdict in train_configs for trans, val in tdict.items()])]
                test_config_names = [cfg for cfg in clf.sconfigs.keys() if cfg not in train_config_names]
                
                # Grab subset of the data corresponding to the test sets:
                test_indices = np.array([ti for ti, tval in enumerate(sample_labels) if tval in test_config_names])
                config_labels = sample_labels[test_indices]
                test_data = sample_data[test_indices, :]
                test_labels = [clf.sconfigs[cfg][clfparams['class_name']] for cfg in config_labels]
                
                
                trainset_sz = clf.train_results['train_data'].shape[0]
                validate_sz = 0 if clf.train_results['test_data'] is None else clf.train_results['test_data'].shape[0]
                if test_data.shape[0] + trainset_sz + validate_sz > sample_data.shape:
                    print "*** @@@ ERROR @@@ ***"
                    print "N train samples (%i) + N test samples (%i) >> original sample size (%i) -- Might be an overlap..."  % (trainset_sz+validate_sz, test_data.shape[0], sample_data.shape[0])
                elif test_data.shape[0] + trainset_sz + validate_sz < sample_data.shape[0]:
                    print "--- warning... ---"
                    print "Not all samples (%i) used:  Train set (%i) | Validate set (%i) | Test set (%i)" %  (sample_data.shape[0], trainset_sz, validate_sz, test_data.shape[0])
                    
            else:
                test_data = None
                test_labels = None
                config_labels = None
                
            trans_classifier['C'].classifiers[cix].test_classifier(test_data=test_data, test_labels=test_labels, config_labels=config_labels)
            
            curr_output_dir = trans_classifier['C'].classifiers[cix].classifier_dir
            pcorrect, counts, config_grid = trans_classifier['C'].classifiers[cix].get_classifier_accuracy_by_stimconfig(row_label=row_label, col_label=col_label, output_dir=curr_output_dir)
            #%
            pcorrect[np.isnan(pcorrect)] = 0.
            counts[np.isnan(counts)] = 0.
            
            accuracy_grid += pcorrect
            counts_grid += counts
            
            nclfs += 1
    
    print "*******************************************************************"

                
    #%
        
    # Plot average of ALL fovs and configs:
    # -------------------------------------
    accuracy = accuracy_grid / nclfs
    
    chance_level = 1./len(clf.class_labels)
    
    rowvals = sorted(list(set([stim[1] for stim in config_grid.keys()])))
    colvals = sorted(list(set([stim[0] for stim in config_grid.keys()])))
    fig = lsvc.plot_transform_grid(accuracy, rowvals=rowvals, colvals=colvals, ylabel=row_label, xlabel=col_label,
                             cmap='hot', vmin=chance_level, vmax=1.0)
    label_figure(fig, data_identifier)
    pl.savefig(os.path.join(clf_output_dir, 'test_transforms_performance_grid_all_datasets.png'))


    counts = counts_grid / nclfs
        
    rowvals = sorted(list(set([stim[1] for stim in config_grid.keys()])))
    colvals = sorted(list(set([stim[0] for stim in config_grid.keys()])))
    fig = lsvc.plot_transform_grid(counts, rowvals=rowvals, colvals=colvals, ylabel=row_label, xlabel=col_label,
                             cmap='Blues_r', vmin=0, vmax=counts.max())
    label_figure(fig, data_identifier)
    
    
    pl.savefig(os.path.join(clf_output_dir, 'test_transforms_counts_grid_all_datasets.png'))

#%%
    
    # If training on position (xpos, ypos), visualize coefficients for each feature (i.e., roi)
    # arranged by xpos, ypos to see if nearby positions hve higher/lower weights:
        
    if optsE.class_name == 'position':
        for fov, trans_classifier in sorted(trans_classifiers.items(), key=lambda x: x[0]):
            clf = trans_classifier['C'].classifiers[0]
            weights = clf.svc.coef_
            print "%s: classifier trained on %i labels (%i features)" % (fov, weights.shape[0], weights.shape[1])
            
            orig_roi_ids = trans_classifier['C'].rois
            best_rids = orig_roi_ids[clf.model_selection.features['kept_rids']]
            
            print "Best cells (N=%i):" % len(best_rids), best_rids
            
            position_list = [(class_ix, tuple([float(p) for p in position_str.split('_')])) for class_ix, position_str in enumerate(clf.svc.classes_)]
            nr = 4
            nc = 5
        
            ncols = 12
            nrows = 8
            nrois_to_plot = ncols * nrows
            fig = pl.figure(figsize=(20,25)) #, axes = pl.subplots(10, 10, sharex=True, sharey=True, figsize=(20,25)) #figure();
            gs1 = gridspec.GridSpec(nrows, ncols)
            gs1.update(wspace=0.1, hspace=0.3) # set the spacing between axes. 
        
            for ai, roi in enumerate(best_rids[0:nrois_to_plot]):
                curr_roi_weights = weights[:, ai]
                weights_grid = np.zeros((nr, nc))
                for class_ix, pos in position_list:
                    ri = rowvals.index(pos[1])
                    ci = colvals.index(pos[0])
                    weights_grid[ri, ci] = curr_roi_weights[class_ix]
                    
                ax = pl.subplot(gs1[ai])
                im = ax.imshow(weights_grid, cmap='hot')    
                #ax.axis('off')
                ax.set_title(roi, fontsize=8)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                if ai == nrois_to_plot - 1:
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    pl.colorbar(im, cax=cax)
            cax.yaxis.set_ticks_position('right')
        
            pl.savefig(os.path.join(clf_output_dir, 'feature_weights_by_position_top%i_%s.png' % (nrois_to_plot, fov)))
            pl.close()
             
            #%%
    
#    accuracy_grid = np.zeros((4, 5))
    for fov, trans_classifier in trans_classifiers.items():
        
        clf = trans_classifier['C'].classifiers[0]
        pcorrect, counts, config_grid = clf.get_classifier_accuracy_by_stimconfig(row_label=row_label, col_label=col_label)
    
        accuracy_grid += pcorrect
    
    accuracy = accuracy_grid / len(trans_classifiers.keys())
    
    
    rowvals = sorted(list(set([stim[1] for stim in config_grid.keys()])))
    colvals = sorted(list(set([stim[0] for stim in config_grid.keys()])))
    fig = lsvc.plot_transform_grid(accuracy, rowvals=rowvals, colvals=colvals, ylabel=row_label, xlabel=col_label,
                             cmap='hot', vmin=0.5, vmax=1.0)
    label_figure(fig, data_identifier)
    
    
    pl.savefig(os.path.join(clf_output_dir, 'test_transforms', 'test_transforms_performance_grid_all_datasets.png'))
    
    
    #%%

    
    for fov, trans_classifier in trans_classifiers.items():
        print "----------- getting fov and rois for %s" % fov
        
        trans_classifier['fov'] = get_fov(trans_classifier['traceid_dir']) 
        #if 'rois' not in trans_classifier.keys():
        #trans_classifier['rois'] = {}
        #trans_classifier['rois']['masks'] = get_roi_masks(trans_classifier['C'].classifiers[0].run_info['traceid_dir'])
        #trans_classifier['rois']['contours'] = util.get_roi_contours(trans_classifier['rois']['masks'])
    
        
#        
#        
#    for fov in trans_classifiers.keys():
#        print "... ... loading %s" % fov
#        C = trans_classifiers[fov]['C']
#        clf = clfs[fov]['classifier']
#        traceid_dir = trans_classifiers[fov]['traceid_dir']
#        clfs[fov]['classifier'] = copy.copy(C.classifiers[0])
#        clfs[fov]['fov'] = get_fov(traceid_dir) 
##%
#        clfs[fov]['RFE'] = clf.get_clf_RFE()
#        clfs[fov]['rois'] = get_roi_masks(clfs[fov]['classifier'].run_info['traceid_dir'])
#        clfs[fov]['contours'] = util.get_roi_contours(clfs[fov]['rois'], roi_axis=-1)
#
#
#    # Plot RFE results from original CLFs created at start:
#    # -------------------------------------------------------------------------
#    if not os.path.exists(os.path.join(clf_output_dir, 'best_RFE_masks.png')):
#        if len(clfs.keys()) == 1:
#            fig, ax = pl.subplots(1, figsize=(10,10))
#            ax.imshow(clfs[fov]['fov'])
#            ax.set_title(fov)
#            ax.axis('off')
#            util.plot_roi_contours(clfs[fov]['fov'], clfs[fov]['contours'], clip_limit=0.2, ax=ax, roi_highlight=clfs[fov]['RFE']['best']['kept_rids'], label_highlight=True, fontsize=8)
#        
#        else:
#            fig, axes = pl.subplots(nrows=1, ncols=len(clfs.keys()), figsize=(20, 5))
#            for fi, fov in enumerate(sorted(clfs.keys(), key=natural_keys)):
#                axes[fi].imshow(clfs[fov]['fov'])
#                axes[fi].set_title(fov)
#                axes[fi].axis('off')
#                util.plot_roi_contours(clfs[fov]['fov'], clfs[fov]['contours'], clip_limit=0.2, ax=axes[fi], roi_highlight=clfs[fov]['RFE']['best']['kept_rids'], label_highlight=True, fontsize=8)
#            
#        pl.savefig(os.path.join(clf_output_dir, 'best_RFE_masks.png'))
#        pl.close()
        
#%%
    # =============================================================================
    # TEST the trained classifier -- TRANSFORMATIONS.
    # =============================================================================
    no_morphs = False

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
        prob_choose_m100_list = get_morph_percents_m100(trans_classifiers, clfs, setC=setC, \
                                                        nfeatures_select=nfeatures_select, \
                                                        full_train=full_train, test_size=test_size, \
                                                        secondary_output_dir=secondary_output_dir)
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
