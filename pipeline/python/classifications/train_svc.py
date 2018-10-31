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
import matplotlib.gridspec as gridspec


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
        visual_areas_fpath = sorted(glob.glob(os.path.join(C.rootdir, C.animalid, C.session, C.acquisition, 'visual_areas', 'segmentation_*.pkl')), key=natural_keys)[::-1][0]
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
    
#options = ['-D', '/n/coxfs01/2p-data', '-i', 'JC022', '-S', '20181018', '-A', 'FOV2_zoom2p7x',
#           '-R', 'combined_blobs_static', '-t', 'traces001',
#           '-r', 'visual', '-d', 'stat', '-s', 'zscore',
#           '-p', 'corrected', '-N', 'morphlevel',
#           '--subset', '0,106',
#           '--nproc=1'
#           ]

rootdir = '/n/coxfs01/2p-data' #-data'
options = ['-D', rootdir, '-i', 'JC015', 
           '-S', '20180919,20180924', 
           '-A', 'FOV1_zoom2p0x,FOV1_zoom2p0x',
           '-R', 'combined_blobs_static,combined_blobs_static',
           '-t', 'traces003,traces002',
           '-r', 'visual', '-d', 'stat', '-s', 'zscore',
           '-p', 'corrected', 
           '-N', 'morphlevel',
           '--subset', '0,106',
           '-c', 'xpos,ypos',
           '--indie',
#           '-v', '0',
#           '-T', '-15,-10,0,5',
#           '-T', '-60,-30,30,60',
#           '-T', '5.6,16.8,28,5.6,16.8,28',
#           '-T', '5,5,5,15,15,15',
           '-V', 'LM-LI',
           '--nproc=1'
           ]

options = ['-D', rootdir, '-i', 'JC022', 
           '-S', '20181016,20181018', 
           '-A', 'FOV1_zoom2p7x,FOV2_zoom2p7x',
           '-R', 'combined_blobs_static,combined_blobs_static',
           '-t', 'traces001,traces001',
           '-r', 'visual', '-d', 'stat', '-s', 'zscore',
           '-p', 'corrected', 
           '-N', 'morphlevel',
           '--subset', '0,97',
           '-c', 'ypos',
           '--indie',
#           '-v', '-13',
#           '-T', '-15,-10,0,5',
#           '-T', '-60,-30,30,60',
#           '-T', '5.6,16.8,28,5.6,16.8,28',
#           '-T', '5,5,5,15,15,15',
           '-V', 'LI-LL',
           '--nproc=1'
           ]


options = ['-D', rootdir, '-i', 'JC022', 
           '-S', '20181016,20181018,20181020', 
           '-A', 'FOV1_zoom2p7x,FOV2_zoom2p7x,FOV1_zoom2p7x',
           '-R', 'combined_blobs_static,combined_blobs_static,combined_blobs_static',
           '-t', 'traces001,traces001,traces001',
           '-r', 'visual', '-d', 'stat', '-s', 'zscore',
           '-p', 'corrected', 
           '-N', 'morphlevel',
           '--subset', '0,106',
           '-c', 'ypos',
           '--indie',
#           '-v', '13',
#           '-T', '-15,-10,0,5',
#           '-T', '-60,-30,30,60',
#           '-T', '5.6,16.8,28,5.6,16.8,28',
#           '-T', '5,5,5,15,15,15',
           '-V', 'LI-LL',
           '--nproc=1'
           ]

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

#options = ['-D', rootdir, '-i', 'JC022', 
#           '-S', '20181007,20181017', 
#           '-A', 'FOV1_zoom2p2x,FOV1_zoom2p7x',
#           '-R', 'combined_blobs_static,combined_blobs_static',
#           '-t', 'traces001,traces001',
#           '-r', 'visual', '-d', 'stat', '-s', 'zscore',
#           '-p', 'corrected', 
#           '-N', 'morphlevel',
#           '--subset', '0,106',
##           '--segment',
#           '-c', 'xpos,ypos',
##           '--indie',
##           '-v', '0',
##           '-T', '-15,-10,0,5',
##           '-T', '-60,-30,30,60',
#           '-T', '16.8,28,16.8,28',
#           '-T', '-5,-5,-15,-15',
#           '-V', 'LI',
#           '--nproc=1'
#           ]

options = ['-D', rootdir, '-i', 'JC022', 
           '-S', '20181022', 
           '-A', 'FOV1_zoom4p0x',
           '-R', 'combined_blobs_static',
           '-t', 'traces001',
           '-r', 'visual', '-d', 'stat', '-s', 'zscore',
           '-p', 'corrected', 
           '-N', 'yrot',
#           '--subset', '0,106',
#           '--segment',
#           '-c', 'xpos,ypos',
#           '--indie',
#           '-v', '0',
#           '-T', '-15,-10,0,5',
#           '-T', '-60,-30,30,60',
#           '-T', '16.8,28,16.8,28',
#           '-T', '-5,-5,-15,-15',
           '-V', 'LI',
           '--nproc=1'
           ]

#%%

def main(options):
    #%%

    # -------------------------------------------------------------------------
    # MODEL SELECTION PARAMS:
    feature_select_method='rfe' #'rfe' #'rfe'
    feature_select_n='best' #'best' #'best'
    C_select='best' #'best'
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    # TRAINING PARAMS:
    scoring = 'accuracy'
    full_train = False #False
    test_size = 0.33 #0.33 #0.33 #0.20 #0.33 #0.33
    create_new = True
#    test_subset_only = True

    m50 = 53
    m100 = 106 #106
    
    col_label = 'xpos' # 'xpos'
    row_label = 'morphlevel' #'ypos'
    # -------------------------------------------------------------------------

    if full_train is False:
        training_regime = 'trainpart_testsize%.2f' % test_size
    else:
        training_regime = 'trainfull'
        
    optsE = extract_options(options)
    visual_area = optsE.visual_area

    traceid_dirs = get_traceids_from_lists(optsE.animalid, optsE.session_list, optsE.fov_list, optsE.run_list, optsE.traceid_list, rootdir=rootdir)
    
    visual_areas_dir =  os.path.join(optsE.rootdir, optsE.animalid, visual_area)
    fov_list = sorted(['%s_%s' % (session, fov) for session, fov in zip(optsE.session_list, optsE.fov_list)], key=natural_keys)
    print "Current FOVs:", fov_list
    if os.path.exists(visual_areas_dir) and create_new is False:
        found_transclassifiers = sorted(glob.glob(os.path.join(visual_areas_dir, '_'.join(fov_list), 'classifiers', 
                                                               'Linear*%s*' % optsE.class_name, '*_*', 'transform_classifiers.pkl')), key=natural_keys)
        if len(found_transclassifiers) > 0:
            for fi, trans_classifiers_fpath in enumerate(found_transclassifiers):
                print fi, trans_classifiers_fpath
            sel = raw_input("Select IDX of trans-classifier to load: ")
            if sel == '':
                create_new = True
            else:
                trans_classifiers_fpath = found_transclassifiers[int(sel)]
                with open(trans_classifiers_fpath, 'rb') as f:        
                    trans_classifiers = pkl.load(f)
                print "Loaded existing TransformClassifiers() object."
                clf_output_dir = os.path.split(trans_classifiers_fpath)[0]
                print "Current output dir:", clf_output_dir
                
        else:
            create_new = True
        
    if create_new:
        
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
    
        clf_output_dir = os.path.join(output_dir, 'classifiers', clf_name)
        if not os.path.exists(clf_output_dir): os.makedirs(clf_output_dir)

    #%
        print "*******************************************************************"
        print "Created TransformClassifier() object for visual area: %s" % visual_area
        print "Saving joint TransformClassifier output to:\n    -->%s" % clf_output_dir
        print "*******************************************************************"
        for fov, trans_classifier in sorted(trans_classifiers.items(), key=lambda x: x[0]):
            print fov
            # Update classifer dir to GROUPED dir:
            trans_classifier['C'].classifier_dir = clf_output_dir  
            trans_classifier['C'].create_classifiers(feature_select_method=feature_select_method, feature_select_n=feature_select_n, C_select=C_select)
        print "*******************************************************************"
        print "Created %i LinearSVM() objects for each of %i FOVs." % (len(trans_classifiers[trans_classifiers.keys()[0]]['C'].classifiers), len(trans_classifiers.keys()))
        print "*******************************************************************"
    
    
        #%%
#        clf_output_dir = os.path.join(clf_output_dir, training_regime)
#        if not os.path.exists(clf_output_dir): os.makedirs(clf_output_dir)

        print "*******************************************************************"
        print "TRAINING."
        print "*******************************************************************"
        for fov, trans_classifier in trans_classifiers.items():
            trans_classifier['C'].train_classifiers(scoring=scoring, full_train=full_train, test_size=test_size, col_label=col_label, row_label=row_label)
        print "*******************************************************************"
        print "Training Complete."
        print "*******************************************************************"
    
        # Save trained classifier to disk, replace dataset npz object with filepath:
        for fov, trans_classifier in trans_classifiers.items():
            trans_classifier['C'].dataset = trans_classifier['C'].data_fpath

        train_test_dir = os.path.join(clf_output_dir, training_regime)
        if not os.path.exists(train_test_dir): os.makedirs(train_test_dir)
        trans_classifiers_fpath = os.path.join(train_test_dir, 'transform_classifiers.pkl')
        with open(trans_classifiers_fpath, 'wb') as f:
            pkl.dump(trans_classifiers, f, protocol=pkl.HIGHEST_PROTOCOL)
    

        #%%
        print "*******************************************************************"
        print "TESTING classifier on generalization:"
            
        test_class_subset_only = False 
        limit_test_to_trained_views = False

        transforms_subset_str = 'class_subset_only' if test_class_subset_only else 'all_classes'

        #% Plot accuracy by stim-config across all FOVs:
        data_identifier = '_'.join(sorted(trans_classifiers.keys(), key=natural_keys))
        print data_identifier

        nr = 5
        nc = 5
        sdf = pd.DataFrame(trans_classifier['C'].classifiers[0].sconfigs).T
        
        accuracy_list = []; counts_list =[];
        accuracy_grid_all = np.zeros((nr, nc))
        counts_grid_all = np.zeros((nr, nc))
        nclfs_all = 0
        for fov, trans_classifier in sorted(trans_classifiers.items(), key=lambda x: x[0]):
            
            accuracy_grid_fov = np.zeros((nr, nc))
            counts_grid_fov = np.zeros((nr, nc))
            nclfs_fov = 0
            for cix in range(len(trans_classifier['C'].classifiers)):
                print "%s:  Testing %i of %i classifiers." % (fov, (cix+1), len(trans_classifier['C'].classifiers))
                    
                clf = trans_classifier['C'].classifiers[cix]
                kept_rids = clf.model_selection.features['kept_rids']
                
                if kept_rids is not None:       
                    sample_data = trans_classifier['C'].sample_data[:, kept_rids]
                else:
                    sample_data = trans_classifier['C'].sample_data
                sample_labels = trans_classifier['C'].sample_labels
                #sdf = pd.DataFrame(clf.sconfigs).T
                clfparams = trans_classifier['C'].classifiers[cix].clfparams
                
                # Restrict test data to those within the specified class subset (e.g., no middle morphs):
                if optsE.class_subset is not '' and test_class_subset_only is True:
                    print "----- Test set only includes classes specified in training subset."
                    sub_sdf = sdf[sdf[optsE.class_name].isin([float(s) for s in optsE.class_subset])]
                else:
                    sub_sdf = copy.copy(sdf)
                    
                train_config_names = list(set(clf.cy_labels))
                untrained_config_names = [cfg for cfg in sub_sdf.index.tolist() if cfg not in train_config_names]

                if test_class_subset_only is False:
                    assert len(untrained_config_names) > 0, "----- *** No untested stimulus configs..."
                    
                if limit_test_to_trained_views:
                    transform_keys = [trans for trans in sub_sdf.columns.tolist() if trans not in ['morphlevel', 'object', 'stimtype']]
                    train_tdict = dict((trans, list(set(sub_sdf[sub_sdf.index.isin(train_config_names)][trans])) ) for trans in transform_keys)
                    
                    test_config_names = [cfg for cfg in untrained_config_names if all([sub_sdf.loc[cfg][trans] in train_tdict[trans] for trans in transform_keys])]
                else:
                    test_config_names = untrained_config_names

                test_sconfigs = dict((cfg, cfg_dict) for cfg, cfg_dict in clf.sconfigs.items() if cfg in test_config_names) #sdf.index.tolist())
                print "----- N train configs: %i, N test configs: %i" % (len(train_config_names), len(test_config_names))

                # Include held-out test set from training/validation step, if relevant:
                heldout_train_data=None; heldout_train_labels=None; heldout_train_labels_config_names=None;
                if not clf.train_params['full_train']:
                    heldout_train_data = clf.train_results['test_data']
                    heldout_train_labels = clf.train_results['test_labels']
                    heldout_train_labels_config_names = clf.cy_labels[np.array(clf.train_results['test_labels'].index.tolist())]

                if len(test_config_names) > 0:
                    # Grab subset of the data corresponding to the test sets:
                    test_indices = np.array([ti for ti, tval in enumerate(sample_labels) if tval in test_config_names])
                    test_labels_config_names = sample_labels[test_indices]
                    test_data = sample_data[test_indices, :]
                    test_labels = np.array([test_sconfigs[cfg][clfparams['class_name']] for cfg in test_labels_config_names])
                    
                    if heldout_train_data is not None:
                        test_data = np.vstack((test_data, heldout_train_data))
                        test_labels = np.hstack((test_labels, heldout_train_labels))
                        test_labels_config_names = np.hstack((test_labels_config_names, heldout_train_labels_config_names))
                else:
                    test_data = heldout_train_data
                    test_labels = heldout_train_labels
                    test_labels_config_names = heldout_train_labels_config_names

                # For MORPHS, if trained on anchors (i.e., class_subset = [m0, m100]),
                # need to relabel middle morphs to one class or the other if they
                # are included here:
#                if not test_class_subset_only and list(set(test_labels)) != clfparams['class_subset']:
#                    print "----- Removing MIDDLE morph test samples (mlevel: %i)" % middle_morph
#                    no_midmorph_ixs = np.array([ti for ti, tlabel in enumerate(test_labels) if tlabel != middle_morph])
#                    tmp_test_labels = test_labels[no_midmorph_ixs]
#                    tmp_test_data = test_data[no_midmorph_ixs, :]
#                    tmp_test_labels_cfgs = test_labels_config_names[no_midmorph_ixs]
#                    
#                    print "----- Relabeling non-anchor morphs to class1 or class2 (trained subset: %s)" % str(clfparams['class_subset'])
#                    relabel_to_m0 = np.array([ti for ti, tlabel in enumerate(tmp_test_labels) if tlabel < middle_morph])
#                    relabel_to_m100 = np.array([ti for ti, tlabel in enumerate(tmp_test_labels) if tlabel > middle_morph])
#                    tmp_test_labels[relabel_to_m0] = 0.0
#                    tmp_test_labels[relabel_to_m100] = m100
#                    
#                    test_labels = tmp_test_labels
#                    test_data = tmp_test_data
#                    test_labels_config_names = tmp_test_labels_cfgs
                    
                # Double check data sizes, etc.
                trainset_sz = clf.train_results['train_data'].shape[0]
                testset_sz = test_data.shape[0]
                if trainset_sz + testset_sz > sample_data.shape[0]:
                    print "*** @@@ ERROR @@@ ***"
                    print "N train samples (%i) + N test samples (%i) >> original sample size (%i) -- Might be an overlap..."  % (trainset_sz, testset_sz, sample_data.shape[0])
                elif trainset_sz + testset_sz < sample_data.shape[0]:
                    print "--- warning... ---"
                    print "Not all samples (%i) used:  Train set (%i) | Test set (%i)" %  (sample_data.shape[0], trainset_sz, testset_sz)

                test_results = trans_classifier['C'].classifiers[cix].test_classifier(test_data=test_data, test_labels=test_labels, config_labels=test_labels_config_names)
                
                curr_output_dir = clf.classifier_dir
                pcorrect, counts, config_grid = clf.get_classifier_accuracy_by_stimconfig(m50=m50, m100=m100, row_label=row_label, col_label=col_label, \
                                                                                          output_dir=curr_output_dir)
                #%
                pcorrect[np.isnan(pcorrect)] = 0.
                counts[np.isnan(counts)] = 0.
                
                accuracy_grid_fov += pcorrect
                counts_grid_fov += counts
                nclfs_fov += 1
#                
#                accuracy_grid_all += pcorrect
#                counts_grid_all += counts
#                
#                nclfs_all += 1
    
            if optsE.indie:
                
                accuracy_fov = accuracy_grid_fov #/ (nclfs_fov-1)
                counts_fov = counts_grid_fov #/ (nclfs_fov-1)
            else:
                accuracy_fov = accuracy_grid_fov / (nclfs_fov)
                counts_fov = counts_grid_fov / (nclfs_fov)

                
            chance_level = 1./len(clf.class_labels)
            
            rowvals = sorted(list(set([stim[1] for stim in config_grid.keys()])))
            colvals = sorted(list(set([stim[0] for stim in config_grid.keys()])))
            
            fig, axes = pl.subplots(1,2,figsize=(15,8))
            lsvc.plot_transform_grid(accuracy_fov, ax=axes[0], rowvals=rowvals, colvals=colvals, ylabel=row_label, xlabel=col_label,
                                     cmap='hot', vmin=chance_level, vmax=1.0, title='test accuracy (%s)' % fov)
            lsvc.plot_transform_grid(counts_fov, ax=axes[1], rowvals=rowvals, colvals=colvals, ylabel=row_label, xlabel=col_label,
                                     cmap='Blues_r', vmin=0, vmax=counts.max(), title='test counts (%s)' % fov)
        
            label_figure(fig, data_identifier)
            pl.savefig(os.path.join(train_test_dir, 'test_transforms_%s_grid_%s.png' % (transforms_subset_str, fov)))
            
                    
            accuracy_grid_all += accuracy_fov
            counts_grid_all += counts_fov
            
            accuracy_list.append(accuracy_fov)
            counts_list.append(counts_fov)
            
            nclfs_all += 1
        print "*******************************************************************"
    
    #%
        # Save test results:
        with open(trans_classifiers_fpath, 'wb') as f:
            pkl.dump(trans_classifiers, f, protocol=pkl.HIGHEST_PROTOCOL)
                    
        #%
            
        # Plot average of ALL fovs and configs:
        # -------------------------------------
        accuracy_all = accuracy_grid_all / nclfs_all
        counts_all = counts_grid_all / nclfs_all
        
        chance_level = 1./len(clf.class_labels)
        
        rowvals = sorted(list(set([stim[1] for stim in config_grid.keys()])))
        colvals = sorted(list(set([stim[0] for stim in config_grid.keys()])))
        
        fig, axes = pl.subplots(1,2,figsize=(15,8))
        lsvc.plot_transform_grid(accuracy_all, ax=axes[0], rowvals=rowvals, colvals=colvals, ylabel=row_label, xlabel=col_label,
                                 cmap='hot', vmin=chance_level, vmax=1.0, title='test accuracy, all fovs')
        lsvc.plot_transform_grid(counts_all, ax=axes[1], rowvals=rowvals, colvals=colvals, ylabel=row_label, xlabel=col_label,
                                 cmap='Blues_r', vmin=0, vmax=counts.max(), title='test counts, all fovs')
        
        label_figure(fig, data_identifier)
        pl.savefig(os.path.join(train_test_dir, 'test_transforms_%s_grid_all_datasets.png' % (transforms_subset_str)))


        #pl.close('all')
        
        max_vals = np.array([max([f1, f2]) for f1, f2 in zip(accuracy_list[0].ravel(), accuracy_list[1].ravel())])
        max_grid = max_vals.reshape(accuracy_grid_all.shape)
        pl.figure(); pl.imshow(max_grid, cmap='hot', vmin=chance_level, vmax=1.0)
        
#%%
    # Plot confusion matrix for CLASS LABEL:
    # -------------------------------------------------------------------------
    
    classes = trans_classifier['C'].classifiers[0].class_labels
    confusion_matrix_cv_all = np.zeros((len(classes), len(classes)))
    confusion_matrix_test_all = np.zeros((len(classes), len(classes)))
#    
    for fov, trans_classifier in sorted(trans_classifiers.items(), key=lambda x: x[0]):

        classes = trans_classifier['C'].classifiers[0].class_labels
        confusion_matrix_cv = np.zeros((len(classes), len(classes)))
        confusion_matrix_test = np.zeros((len(classes), len(classes)))
        
        for cix in range(len(trans_classifier['C'].classifiers)):
            print "getting accuracy on class labels [%s] for %i of %i." % (str(classes), cix+1, len(trans_classifier['C'].classifiers))
            clf = trans_classifier['C'].classifiers[cix]
        #for fov, trans_classifier in sorted(trans_classifiers.items(), key=lambda x: x[0]):
                            
            # Test set:
            predicted_labels = clf.test_results['predicted_classes']
            true_labels = clf.test_results['test_labels']
            if clf.clfparams['class_name'] == 'morphlevel' and clf.clfparams['class_subset'] != '':
                # Convert true labels to category m0 or m100:
                excluded_midmorph_ixs = np.array([ti for ti, tlabel in enumerate(true_labels) if tlabel != m50])
                predicted_test = predicted_labels[excluded_midmorph_ixs]
                true_test = np.array([float(0) if p < m50 else float(m100) for p in true_labels[excluded_midmorph_ixs]])
            else:
                predicted_test = predicted_labels
                true_test = true_labels
            classes = clf.class_labels
            cmatrix, _  = lsvc.get_confusion_matrix(predicted_test, true_test, classes)
            
            confusion_matrix_test += cmatrix
            
            # CV on train set:
            predicted_cv = clf.cv_results['predicted_classes']
            true_cv = clf.cv_results['true_classes']
            classes = trans_classifier['C'].classifiers[cix].class_labels
            cmatrix, _  = lsvc.get_confusion_matrix(predicted_cv, true_cv, classes)
    
            confusion_matrix_cv += cmatrix
        
        confusion_matrix_cv_all += confusion_matrix_cv
        confusion_matrix_test_all += confusion_matrix_test
        
        clim = None #'max' #'max'# # 'max' #None #'max' # 'max' #'max'
        if clim == 'max':
            cmap_str = '_max'
        else:
            cmap_str = ''
            
        fig, axes = pl.subplots(1,2,figsize=(8,6))
        lsvc.plot_confusion_matrix(confusion_matrix_test, classes=classes, ax=axes[1], normalize=True,
                                   title='test %s' % fov, cmin=0, clim=clim)
        lsvc.plot_confusion_matrix(confusion_matrix_cv, classes=classes, ax=axes[0], normalize=True,
                                   title='CV - %s' % fov, cmin=0, clim=clim)
        pl.savefig(os.path.join(train_test_dir, 'confusion%s_%s_cv_test_%i%s_%s.png' % (cmap_str, transforms_subset_str, len(classes), optsE.class_name, fov))) #

        
        confusion_matrix_cv_all += confusion_matrix_cv
        confusion_matrix_test_all += confusion_matrix_test

    fig, axes = pl.subplots(1,2,figsize=(8, 6))
    lsvc.plot_confusion_matrix(confusion_matrix_test_all, classes=classes, ax=axes[1], normalize=True,
                               title='test - all fovs', cmin=0, clim=clim)
    lsvc.plot_confusion_matrix(confusion_matrix_cv_all, classes=classes, ax=axes[0], normalize=True,
                               title='cv - all fovs', cmin=0, clim=clim)
    label_figure(fig, data_identifier)
    pl.savefig(os.path.join(train_test_dir, 'confusion%s_%s_cv_test_%i%s_allfovs.png' % (cmap_str, transforms_subset_str, len(classes), optsE.class_name))) #

    #pl.close('all')
 
        
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
#            nr = 4
#            nc = 6
        
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
                if ai == nrois_to_plot - 1 or ai == len(best_rids) - 1:
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    pl.colorbar(im, cax=cax)
            cax.yaxis.set_ticks_position('right')
            pl.pause(1.0)
            pl.savefig(os.path.join(clf_output_dir, 'feature_weights_by_position_top%i_%s.png' % (nrois_to_plot, fov)))
            #pl.close()
    
    #%%

#    limit_test_to_trained_views = False
    
    if limit_test_to_trained_views:
        test_morphs_str = 'restrict_trained_views'
    else:
        test_morphs_str = 'average_all_views'
    morph_levels = sorted(list(set(sdf['morphlevel'])))
    
    all_prob_m100 = dict((morph_level, []) for morph_level in morph_levels)
    if optsE.class_name == 'morphlevel':
            
        for fov, trans_classifier in sorted(trans_classifiers.items(), key=lambda x: x[0]):
            
            prob_m100_list = dict((morph_level, []) for morph_level in morph_levels)
            counts = dict((morph_level, []) for morph_level in morph_levels)
            for cix in range(len(trans_classifier['C'].classifiers)):
                print "%s:  Testing %i of %i classifiers." % (fov, (cix+1), len(trans_classifier['C'].classifiers))
                    
                clf = trans_classifier['C'].classifiers[cix]
                #sdf = pd.DataFrame(clf.sconfigs).T
                #sconfigs = clf.sconfigs
                
                #predicted_classes = clf.test_results['predicted_classes']
                #original_classes = [sdf.loc[cfg]['morphlevel'] for cfg in clf.test_results['config_labels']]
                
                trained_configs = clf.train_results['results_by_config'].keys()
                
                transform_keys = [trans for trans in sdf.columns.tolist() if trans not in ['morphlevel', 'object', 'stimtype']]
                train_tdict = dict((trans, list(set(sdf[sdf.index.isin(trained_configs)][trans])) ) for trans in transform_keys)
                
                if m100 != 106:
                    morph_levels = sorted([m for m in list(set(sdf['morphlevel'])) if m < 106])
                else:
                    morph_levels = sorted(list(set(sdf['morphlevel'])))

                for morph_level in morph_levels:
                        
                    all_configs_for_morph = sdf[sdf['morphlevel'] == morph_level].index.tolist()

                    if limit_test_to_trained_views:
                        curr_config_names = [cfg for cfg in all_configs_for_morph if all([sdf.loc[cfg][trans] in train_tdict[trans] for trans in transform_keys])]
                    else:
                        curr_config_names = all_configs_for_morph
                    
                    ncorrect_from_cv=0;ntotal_from_cv=0;
                    missing_cfgs_from_test = [cfg for cfg in curr_config_names if cfg not in clf.test_results['results_by_config'].keys()]
                    for missing in missing_cfgs_from_test:
                        #cv_pcorrect = clf.train_results['results_by_config'][missing]['percent_correct']
                        #pchoose100 = (1-cv_pcorrect) if sdf.loc[missing]['morphlevel']==0 else cv_pcorrect
                        ncorrect_from_cv += clf.train_results['results_by_config'][missing]['ncorrect']
                        ntotal_from_cv += clf.train_results['results_by_config'][missing]['ntotal']
                    
                    curr_results = [clf.test_results['results_by_config'][cfg]['predicted'] for cfg in curr_config_names if cfg not in missing_cfgs_from_test]
                    curr_predictions = [p for sublist in curr_results for p in sublist ]
                    nchoose100 = sum([p==m100 for p in curr_predictions]) + ncorrect_from_cv
                    nsamples_total = sum([len(clf.test_results['results_by_config'][cfg]['indices']) for cfg in curr_config_names if cfg not in missing_cfgs_from_test]) + ntotal_from_cv
                    pchoose100 = float(nchoose100) / float(nsamples_total)
                    print "%i:  Chose %i out of %i (%i cfgs)" % (morph_level, nchoose100, nsamples_total, len(curr_config_names))
                    
                    prob_m100_list[morph_level].append(pchoose100)
                    counts[morph_level].append(nsamples_total)


            pl.figure()
            mlevel = sorted(prob_m100_list.keys())
            pchoose_mean = [np.mean(v) for k, v in sorted(prob_m100_list.items(), key=lambda x: x[0])]
            pchoose_sem = [stats.sem(v) for k, v in sorted(prob_m100_list.items(), key=lambda x: x[0])]
            pl.plot(mlevel, pchoose_mean, 'bo')
            pl.errorbar(mlevel, pchoose_mean, yerr=pchoose_sem)
            pl.ylim([0, 1.0])
            pl.ylabel('p choose 100%')
            pl.xlabel('morph level')
            pl.title(fov)
            pl.savefig(os.path.join(train_test_dir, 'prob_choose_m100_%s_%s.png' % (test_morphs_str, fov)))
                
            for ml, pc in prob_m100_list.items():
                all_prob_m100[ml].append(np.mean(pc))
                
    pl.figure()
    mlevel = sorted(all_prob_m100.keys())
    pchoose_mean = [np.mean(v) for k, v in sorted(all_prob_m100.items(), key=lambda x: x[0])]
    pchoose_sem = [stats.sem(v) for k, v in sorted(all_prob_m100.items(), key=lambda x: x[0])]
    
    pl.plot(mlevel, pchoose_mean, 'ko')
    pl.ylim([0, 1])
    pl.errorbar(mlevel, pchoose_mean, yerr=pchoose_sem)
    pl.savefig(os.path.join(train_test_dir, 'prob_choose_m100_%s_all_fovs.png' % test_morphs_str))
    
    
#%%


    
    
    
#%%

#def main(options):
#    optsE = extract_options
#    C = train_test_linearSVC(optsE)
#


if __name__ == '__main__':
    main(sys.argv[1:])
