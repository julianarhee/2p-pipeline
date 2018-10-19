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
 
def hist_cv_permutations(svc, cX_std, cy, clfparams, scoring='accuracy', 
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
    
    return fig



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
            #print train, test
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
            #print train, test
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


    # Save CV info:
    # -----------------------------------------------------------------------------
    
#    f = open(os.path.join(output_dir, 'results', 'CV_report.txt'), 'w')
#    for y_true, y_pred in zip(true, predicted):
#        f.write(metrics.classification_report(y_true, y_pred, target_names=[str(c) for c in classes]))
#    f.close()
#    
#    cv_results = {'predicted': [list(p) for p in predicted], #.tolist(), #list(y_pred),
#                  'true': [list(p) for i in true], # list(y_test),
#                  'classifier': clfparams['classifier'],
#                  'cv_method': clfparams['cv_method'],
#                  'ngroups': clfparams['cv_ngroups'],
#                  'nfolds': clfparams['cv_nfolds'],
#                  'classes': classes
#                  }
#    with open(os.path.join(output_dir, 'results', 'CV_results.json'), 'w') as f:
#        json.dump(cv_results, f, sort_keys=True, indent=4)
#    
#    print "Saved CV results: %s" % output_dir
    
    return predicted, true, classes


def get_confusion_matrix(predicted, true, classes, average_iters=True):
    # Compute confusion matrix:
    # -----------------------------------------------------------------------------
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

def plot_confusion_matrix(cmatrix, classes,
                          ax=None,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=pl.cm.Blues, cmax=1.0):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cmatrix = cmatrix.astype('float') / cmatrix.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cmatrix)

    #fig = pl.figure(figsize=(4,4))
    if ax is None:
        fig = pl.figure(figsize=(4,4))
        ax = fig.add_subplot(111)

    ax.set_title(title, fontsize=10)

    im = ax.imshow(cmatrix, interpolation='nearest', cmap=cmap, vmax=cmax)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45, fontsize=10)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=10)
    fmt = '.2f' if normalize else 'd'
    thresh = cmatrix.max() / 2.
    for i, j in itertools.product(range(cmatrix.shape[0]), range(cmatrix.shape[1])):
        ax.text(j, i, format(cmatrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cmatrix[i, j] > thresh else "black")

    #pl.tight_layout()
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    pl.colorbar(im, cax=cax)

    return fig


def get_best_C(svc, X, y, output_dir=None):
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

    if output_dir is not None:
        figname = 'crossval_scores_by_C.png'
        pl.savefig(os.path.join(output_dir, figname))
        pl.close()

    return best_C


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


def train_test_linearSVC(optsE):

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
    sel = input("Select IDX of TransformClassifier object to load: ")
    tclf_fpath = found_transform_classifiers[int(sel)]

    with open(tclf_fpath, 'rb') as f:
        C = pkl.load(f)
    if isinstance(C.dataset, str):
        loaded_dset = np.load(C.dataset)
        C.dataset = loaded_dset
        
    return C

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
    
options = ['-D', '/n/coxfs01/2p-data', '-i', 'JC022', '-S', '20181016', '-A', 'FOV1_zoom2p7x',
           '-R', 'combined_blobs_static', '-t', 'traces001',
           '-r', 'visual', '-d', 'stat', '-s', 'zscore',
           '-p', 'corrected', '-N', 'morphlevel',
           #'--subset', '0,106',
           '--nproc=1'
           ]
   
#%%
optsE = extract_options(options)

traceid_dir = get_traceid_dir(optsE.animalid, optsE.session, optsE.acquisition, optsE.run, optsE.traceid)

C = load_classifier_object(traceid_dir)

if C is None:
    C = train_test_linearSVC(optsE)

if len(C.classifiers) > 1:
    # Const-trans/ trans-value pairs were separately trained.
    print "-----------------------------------------------------"
    print "[%s]-classifier was trained at separate values of %s" % (C.params['class_name'], C.params['const_trans'])
    for ci, clf in enumerate(C.classifiers):
        print [('IDX: %i' % ci, '%s: %i' % (trans, val)) for (trans, val) in zip(clf.clfparams['const_trans'], clf.clfparams['trans_value'])]
    
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
    
#%% Look at specific classifier

#curr_clf = 'ypos_n13.0'

curr_clf = rfe_results.keys()[0]

if C.params['const_trans'] != '':
    clf = copy.copy([c for c in C.classifiers if curr_clf.split('_')[0] in c.clfparams['const_trans'] and float(curr_clf.split('_')[1].replace('n', '-')) in c.clfparams['trans_value']][0])
else:
    clf = C.classifiers[0]


#%%

setC = 'best'
nfeatures_select = 'best'
full_train = True 


# Create classifier using found params:
# -----------------------------------------------------------------------------

svc = copy.copy(clf.clf)
#svc = LinearSVC(random_state=0, dual=dual, multi_class='ovr', C=C_val)

tmp_cX = np.copy(clf.cX)
cy = np.copy(clf.cy)

#%%
    

if isinstance(nfeatures_select, int):

    rfe = RFE(svc, n_features_to_select=nfeatures_select)
    rfe.fit(tmp_cX, cy)
    
    removed_rids = np.where(rfe.ranking_!=1)[0]
    kept_rids = np.array([i for i in np.arange(0, tmp_cX.shape[-1]) if i not in removed_rids])
    feature_str = 'fit%i_RFE' % nfeatures_select
    cX = tmp_cX[:, kept_rids]

elif nfeatures_select == 'best':
    kept_rids = rfe_results[curr_clf]['best']['kept_rids']
    feature_str = 'bestRFE'
    cX = tmp_cX[:, kept_rids]

else:
    kept_rids = range(C.sample_data.shape[-1]) #C.rois
    cX = np.copy(tmp_cX)
    feature_str = 'allcells'

# Set C:
# ----------------
if setC == 'best':
    C_val = get_best_C(svc, cX, cy)
    pl.savefig(os.path.join(clf.classifier_dir, 'figures', 'C_values.png'))
    pl.close()
    
elif setC == 'big':
    C_val = 1E9 #clf.clfparams['C_val']
elif setC == 'small':
    C_val = 1.0
C_str = '%sC%.2f' % (setC, C_val)

clf_subdir = '%s_%s' % (feature_str, C_str)
clf_output_dir = os.path.join(clf.classifier_dir, clf_subdir)
if not os.path.exists(os.path.join(clf_output_dir, 'figures')): os.makedirs(os.path.join(clf_output_dir, 'figures'))
print "Saving current CLF results to:", clf_output_dir

dual = cX.shape[0] > cX.shape[1]
nfolds = clf.clfparams['cv_nfolds']
svc.C = C_val
svc.dual = dual
print svc

kfold = StratifiedKFold(n_splits=nfolds, shuffle=True)

#%

# PLOT CV results:
# -----------------------------------------------------------------------------

# Do CV in limited dataset:
hist_cv_permutations(svc, cX, cy, clf.clfparams, data_identifier=C.data_identifier)
pl.savefig(os.path.join(clf_output_dir, 'figures', 'cv_permutation.png'))
    
#% Confusion matrix:
predicted, true, classes = get_cv_folds(svc, clf.clfparams, cX, cy, output_dir='')

cmatrix, cstr = get_confusion_matrix(predicted, true, classes, average_iters=True)
plot_confusion_matrix(cmatrix, classes, ax=None, normalize=True, title='%s conf matrix (n=%i)' % (cstr, clf.clfparams['cv_nfolds']), cmap=pl.cm.Blues)
pl.savefig(os.path.join(clf_output_dir, 'figures', 'cv_confusion.png'))

#%
#full_train = False

if full_train:
    svc.fit(cX, cy)
    test_anchors = [v for sublist in predicted for v in sublist ]
    y_test = [v for sublist in true for v in sublist ]
    #test_anchors = svc.predict(cX)
    #test_score = svc.score(cX, cy)
    train_set = 'fulltrain'
else:
    X_train, X_test, y_train, y_test = train_test_split(cX, cy, test_size=0.2, random_state=0, shuffle=True)
    
    svc.fit(X_train, y_train)
    test_anchors = svc.predict(X_test)
    test_score = svc.score(X_test, y_test)

    train_set = 'halftrain'
#%


#%%  TEST TEST TEST

sample_data = C.sample_data[:, kept_rids]
sample_labels = C.sample_labels

sdf = pd.DataFrame(C.sconfigs).T

# Filter sconfigs by const-trans/trans-value pair:
if clf.clfparams['const_trans'] != '':
    trans_sdf = sdf[sdf[clf.clfparams['const_trans'][0]]==clf.clfparams['trans_value'][0]]
else:
    trans_sdf = copy.copy(sdf)
test_values = [val for val in trans_sdf[clf.clfparams['class_name']].unique() if val not in clf.clfparams['class_subset']]

test_sdf = trans_sdf[trans_sdf[clf.clfparams['class_name']].isin(test_values)]
configs_included = test_sdf.index.tolist()

kept_trial_ixs = np.array([fi for fi, config in enumerate(sample_labels) if config in configs_included])

test_data = sample_data[kept_trial_ixs, :]
tmp_test_labels = sample_labels[kept_trial_ixs]
test_labels = [test_sdf[test_sdf.index==config][clf.clfparams['class_name']][0] for config in tmp_test_labels]

print "Test data:", test_data.shape
print "Test labels:", list(set(test_labels))


test_choices = svc.predict(test_data) #(test_data, fake_labels)
prob_choose_m100 = {}
m100 = 106
# Plot % correct from test choices:
morph_levels = sorted(trans_sdf[clf.clfparams['class_name']].unique())
for morph_level in morph_levels:
    if morph_level in [morph_levels[0], morph_levels[-1]]:
        curr_trials = [ti for ti, tchoice in enumerate(y_test) if tchoice == morph_level]
        curr_choices = [test_anchors[ti] for ti in curr_trials]
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

def main(options):
    optsE = extract_options
    C = train_test_linearSVC(optsE)



if __name__ == '__main__':
    main(sys.argv[1:])
