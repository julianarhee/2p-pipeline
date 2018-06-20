#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 13:59:19 2018

@author: juliana
"""


import h5py
import os
import json
import cv2
import time
import math
import random
import itertools
import scipy.io
import optparse

import cPickle as pkl
import pandas as pd
import numpy as np
import pylab as pl
import matplotlib as mpl
import seaborn as sns
import pyvttbl as pt
import multiprocessing as mp
import tifffile as tf
from collections import namedtuple
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import scikit_posthocs as sp
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison
from ast import literal_eval

from pipeline.python.utils import natural_keys, replace_root, print_elapsed_time
import pipeline.python.traces.combine_runs as cb
import pipeline.python.paradigm.align_acquisition_events as acq
import pipeline.python.visualization.plot_psths_from_dataframe as vis
from pipeline.python.traces.utils import load_TID
from pipeline.python.paradigm import utils as fmt
from pipeline.python.classifications import utils as util


from mpl_toolkits.axes_grid1 import make_axes_locatable

from sklearn.manifold import MDS
#%
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import cross_validation

from sklearn.datasets import make_multilabel_classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelBinarizer
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA



from sklearn.linear_model import SGDClassifier, LogisticRegression, LogisticRegressionCV
from sklearn.calibration import CalibratedClassifierCV



from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold, cross_val_score
from sklearn import svm
from sklearn import cross_validation
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
#from joblib import Parallel, delayed
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import permutation_test_score


from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import LeavePGroupsOut, LeaveOneGroupOut, LeaveOneOut
from sklearn import metrics

#%%
def get_best_C(svc, X, y, output_dir='/tmp', classifier_str=''):
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

    figname = 'crossval_scores_by_C_%s.png' % classifier_str
    pl.savefig(os.path.join(output_dir, figname))

    return best_C

#%%
def plot_confusion_matrix(cm, classes,
                          ax=None,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=pl.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    #fig = pl.figure(figsize=(4,4))
    if ax is None:
        fig = pl.figure(figsize=(4,4))
        ax = fig.add_subplot(111)

    pl.title(title)

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    tick_marks = np.arange(len(classes))
    pl.xticks(tick_marks, classes, rotation=45)
    pl.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        pl.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    pl.tight_layout()
    pl.ylabel('True label')
    pl.xlabel('Predicted label')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    pl.colorbar(im, cax=cax)


#%%
    

def get_roi_list(run_info, roi_selector='visual', metric='meanstimdf'):
        
    trans_types = run_info['transforms'].keys()
    
    # Load sorted ROI info:
    # -----------------------------------------------------------------------------
#    datakey = run_info['datakey']
#    if '/' in datakey:
#        datakey = datakey[1:]
#    sort_dir = os.path.join(traceid_dir, 'sorted_%s' % datakey)

    sorted_dirs = [s for s in os.listdir(traceid_dir) if 'sorted_' in s and os.path.isdir(os.path.join(traceid_dir, s))]
    sort_dir = os.path.join(traceid_dir, sorted_dirs[0])
    
    if roi_selector == 'visual':
        
        # Get list of visually repsonsive ROIs:
        responsive_anova_fpath = os.path.join(sort_dir, 'visual_rois_anova_results.json')
        assert os.path.exists(responsive_anova_fpath), "No results found for VISUAL rois: %s" % sort_dir
        
        print "Loading existing split ANOVA results:\n", responsive_anova_fpath
        with open(responsive_anova_fpath, 'r') as f:
            responsive_anova = json.load(f)
        
        # Sort ROIs:
        responsive_rois = [r for r in responsive_anova.keys() if responsive_anova[r]['p'] < 0.05]
        sorted_visual = sorted(responsive_rois, key=lambda x: responsive_anova[x]['F'])[::-1]
        print "Loaded %i visual out of %i neurons (split-plot ANOVA (p<0.05)." % (len(sorted_visual), len(run_info['roi_list']))
        
        visual_rids = [int(r[3:])-1 for r in sorted_visual]
    
    elif roi_selector == 'selectiveanova':
    
        selective_anova_fpath = os.path.join(sort_dir, 'selective_rois_anova_results_%s.json' % metric)
        assert os.path.exists(selective_anova_fpath), "No results found for SELECTIVE rois (anova): %s" % sort_dir
        
        if os.path.exists(selective_anova_fpath):
            print "Loading existing %i-way ANOVA results: %s" % (len(trans_types), selective_anova_fpath)
            with open(selective_anova_fpath, 'r') as f:
                selective_anova = json.load(f)
            if '(' in selective_anova[selective_anova.keys()[0]].keys()[0]:
                for r in selective_anova.keys():
                    selective_anova[r] = {literal_eval(k):v for k, v in selective_anova[r].items()}
    
        if len(trans_types) > 1:    
            selective_rois_anova = [k for k in selective_anova.keys()
                                    if any([selective_anova[k][t]['p'] < 0.05
                                    for t in selective_anova[k].keys()]) ]
            # Sort by F-value, from biggest --> smallest:
            selective_rois_anova = sorted(selective_rois_anova, key=lambda x: max([selective_anova[x][f]['F'] for f in selective_anova[x].keys()]))[::-1]
            
        else:
            selective_rois_anova = [k for k in selective_anova.keys()
                                    if selective_anova[k]['p'] < 0.05]
            
        visual_rids = [int(r[3:])-1 for r in selective_rois_anova]
        
    return visual_rids

#%%
    
def split_trial_epochs(cX, cy, class_name, sconfigs, run_info, binsize=10, relabel=False):
    # Reshape (nframes_per_trial*ntrials_total X nrois) into ntrials_total x nframes_per_trial x nrois)
    nframes_per_trial = run_info['nframes_per_trial']
    ntrials_total = run_info['ntrials_total']
    nrois = len(run_info['roi_list'])
    print "Reshaping input data into trials (%i) x frames (%i) x rois (%i)" % (ntrials_total, nframes_per_trial, nrois) 
    cX_r = np.reshape(cX, (ntrials_total, nframes_per_trial, nrois))
    cy_r = np.reshape(cy, (ntrials_total, nframes_per_trial))
    
    # Bin frames into smaller chunks so we don't have to train 300 classifiers...
    # nframes_on = int(round(run_info['nframes_on']))
    # stim_on = int(run_info['stim_on_frame'])
    bin_idxs =  np.arange(0, nframes_per_trial, binsize)
    bin_idxs = np.append(bin_idxs, nframes_per_trial+1) # Add the last frame 
 
    # Make binned frame data into shape: ntrial_epochs x ntrials_total x nrois.
    cX_tmp = np.array([np.mean(cX_r[:, bstart:bin_idxs[bi+1], :], axis=1) for bi, bstart in enumerate(bin_idxs[0:-1])])
    
    # There will be cX_tmp.shape[0] classifiers, each with ntrials_total x nrois dataset:
    epochs = dict((epoch, cX_tmp[epoch, :, :]) for epoch in range(cX_tmp.shape[0])) # Each dataset is ntrials x nrois
    trial_labels = [sconfigs[cv][class_name] for cv in cy_r[:,0]]
    if relabel:
        labels = dict((epoch, trial_labels) for epoch in range(cX_tmp.shape[0]))           # Each sample in dataset (ntrials) is labeled by stim type
    else:
        labels = dict((epoch, cy_r[:,0]) for epoch in range(cX_tmp.shape[0]))           # Each sample in dataset (ntrials) is labeled by stim type

    return epochs, labels, bin_idxs[0:-1]
    
#%%
def group_classifier_data(cX, cy, class_name, sconfigs, aggregate_type='all', 
                              subset=None, nsamples=None,
                              const_trans=None, trans_value=None, relabel=False):
    sconfigs_df = pd.DataFrame(sconfigs).T

    # Check that input data is correctly formatted:
    assert len(cy)==cX.shape[0], "Incorrect input data sizes. Data (%s), labels (%s)" % (str(cX.shape), str(cy.shape))
    assert sorted(list(set(cy))) == sorted(sconfigs.keys()), "Bad configIDs for label list. Should be the same as sconfigs.keys()."
    
    if aggregate_type == 'all':
        # Assign labels, but don't group/average values by some shared transform value. 
        if const_trans is '':
            stim_grouper = [class_name]
            other_trans = list(set([k for k in sconfigs['config001'].keys() if not k == class_name]))
            stim_grouper.extend(other_trans)   
        else:
            stim_grouper = [class_name, const_trans]
        
        cy_tmp = cy.copy(); cX_tmp = cX.copy();
        
        if class_name == 'morphlevel':
            morphs_in_dataset = sorted(list(set([sconfigs[c]['morphlevel'] for c in sconfigs.keys()])))
            true_morphs = np.array(sorted([m for m in morphs_in_dataset if m >=0])) # movies have -1 label, so only get true morph *objects*
            if subset is not None:
                if subset == 'two_class':
                    included = [true_morphs[0], true_morphs[-1]] # Take first and last only (anchors)
                elif subset == 'no_morphing':
                    included = true_morphs.copy() # Take first and last only (anchors)
                    
                configs_included = [c for c in sconfigs.keys() if sconfigs[c]['morphlevel'] in included]
                cixs = np.array([ci for ci,cv in enumerate(cy) if cv in configs_included])
                cy_tmp = cy[cixs]
                cX_tmp = cX[cixs,:]
                
            
        sgroups = sconfigs_df.groupby(stim_grouper)                                               # Group stimconfigs with top-level as class-label type
        ordered_configs = {key: i for i, key in enumerate([str(g.index[0]) for k,g in sgroups])}  # Create a mapper for sorting the original configID list
        #sorted_ixs = [s for s, val in sorted(enumerate(cy_tmp), key=lambda d: ordered_configs[d[1]])]   # Get indices with which to sort original configID list
        sorted_ixs = [s for s, val in enumerate(cy_tmp)]   # Get indices with which to sort original configID list

        # check that indices are grouped properly:
#        if data_type == 'xcondsub':
#            sidx = 0
#            diffs = []
#            for k in sorted(ordered_configs.keys(), key= lambda d: ordered_configs[d]):
#                diffs.append(len(list(set(np.diff(sorted_ixs[sidx:sidx+nframes_per_trial])))))
#                sidx += nframes_per_trial
#            print diffs
#            assert len(list(set(diffs)))==1 and list(set(diffs))[0]==1, "Incorrect index grouping!"

        # Re-sort class labels and data:
        cy_tmp = cy_tmp[sorted_ixs]
        cX_tmp = cX_tmp[sorted_ixs, :]
        
    elif aggregate_type == 'half':
        
        cy_tmp = cy.copy(); cX_tmp = cX.copy();
        
        if class_name == 'morphlevel':
            morphs_in_dataset = sorted(list(set([sconfigs[c]['morphlevel'] for c in sconfigs.keys()])))
            true_morphs = np.array(sorted([m for m in morphs_in_dataset if m >=0])) # movies have -1 label, so only get true morph *objects*
            if subset is not None:
                if subset == 'two_class':
                    included = [true_morphs[0], true_morphs[-1]] # Take first and last only (anchors)
                elif subset == 'no_morphing':
                    included = true_morphs.copy() # Take first and last only (anchors)
                configs_included = [c for c in sconfigs.keys() if sconfigs[c]['morphlevel'] in included]
        else:
            configs_included = [c for c in sconfigs.keys()]
                
        # Group trials of the same condition (repetitions) and average:
        cixs = []
        for k in configs_included: #sconfigs.keys():
            curr_cys = [ci for ci,cv in enumerate(cy) if cv==k]
            random.shuffle(curr_cys)
            if nsamples is None:
                nsamples = int(round(len(curr_cys)/2))
            curr_cys = curr_cys[0:nsamples]
            cixs.append(curr_cys)
        cixs = np.hstack(cixs)
        
        cy_tmp = cy_tmp[cixs]
        cX_tmp = cX_tmp[cixs,:]
        #cX_tmp.append(np.mean(cX[curr_cys, :], axis=0))
    
        # Re-sort class labels and data:
        #cy_tmp = np.hstack(cy_tmp)
        #cX_tmp = np.vstack(cX_tmp)
        
    elif aggregate_type == 'averagereps':
        # Group trials of the same condition (repetitions) and average:
        cy_tmp = []; cX_tmp = [];
        for k in sconfigs.keys():
            curr_cys = [ci for ci,cv in enumerate(cy) if cv==k]
            cy_tmp.append(k)
            cX_tmp.append(np.mean(cX[curr_cys, :], axis=0))
    
        # Re-sort class labels and data:
        cy_tmp = np.hstack(cy_tmp)
        cX_tmp = np.vstack(cX_tmp)
            
    elif aggregate_type == 'collapse':
        # 'Collapse' views of a given object together:
        # Average values for all samples with the same class label (class_name), OR
        # If const_trans provided, average trials only across specified transform type.
        stim_grouper = [class_name, const_trans]
        if const_trans is '':
            # Just group by class_name:
            stim_grouper = [class_name]
            
        sgroups = sconfigs_df.groupby(stim_grouper)
        cy_tmp = []; cX_tmp = [];
        for k, g in sgroups:
            curr_cys = [ci for ci,cv in enumerate(cy) if cv in g.index.tolist()]
            cy_tmp.append(k)
            cX_tmp.append(np.mean(cX[curr_cys, :], axis=0))
    
        # Re-sort class labels and data:
        cy_tmp = np.hstack(cy_tmp)
        cX_tmp = np.vstack(cX_tmp)
        
    elif aggregate_type == 'single':
        # Select only those samples w/ values equal to specificed transform value.
        # 'const_trans' :  transform type desired
        # 'trans_value' :  value of const_trans to use.
        stim_grouper = [class_name, const_trans]
        
        # Check that provided trans_value is valid:
        assert trans_value in list(set([sconfigs[c][const_trans] for c in sconfigs.keys()])), "Invalid value %s for transform type %s." % (str(trans_value, const_trans))
        
        # Get configIDs with const_trans value = trans_value:
        sconfigs_included = sconfigs_df[sconfigs_df[const_trans]==trans_value]

        if class_name == 'morphlevel':
            morphs_in_dataset = sorted(list(set([sconfigs[c]['morphlevel'] for c in sconfigs_included.index.tolist()])))
            true_morphs = np.array(sorted([m for m in morphs_in_dataset if m >=0])) # movies have -1 label, so only get true morph *objects*
            if subset is not None:
                if subset == 'two_class':
                    included = [true_morphs[0], true_morphs[-1]] # Take first and last only (anchors)
                elif subset == 'no_morphing':
                    included = true_morphs.copy() # Take first and last only (anchors)
                else:
                    included = morphs_in_dataset
                    
                configs_included = [c for c in sconfigs_included.index.tolist() if sconfigs[c]['morphlevel'] in included]
                cixs = np.array([ci for ci,cv in enumerate(cy) if cv in configs_included])
                cy_tmp = cy[cixs]
                cX_tmp = cX[cixs,:]
        else:
                
            sgroups = sconfigs_included.groupby(stim_grouper)
            cy_tmp = []; cX_tmp = []
            for k, g in sgroups:
                curr_cys = [ci for ci,cv in enumerate(cy) if cv in g.index.tolist()]
                cy_tmp.append(cy[curr_cys])
                cX_tmp.append(cX[curr_cys, :])
        
    
        # Re-sort class labels and data:
        cy_tmp = np.hstack(cy_tmp)
        cX_tmp = np.vstack(cX_tmp)
        
    # Assign CLASS LABEL to each sample:
    if relabel:
        cy = np.array([sconfigs[cv][class_name] for cv in cy_tmp])
    else:
        cy = cy_tmp.copy()
    cX = cX_tmp.copy()
    
    return cX, cy

#%%
def get_trial_bins(nframes_per_trial, stim_on_frame, binsize=10):
    bins = []
    # Figure out bin size if using FRAMES:
    nframes_per_trial = run_info['nframes_per_trial']
    stim_on_frame = run_info['stim_on_frame']
    try:
        bins = np.arange(0, nframes_per_trial, binsize)
        assert stim_on_frame in bins, "Stim on frame %i not in bins: %s" % (stim_on_frame, str(bins))
    except Exception as e:
        binsize += 5
        bins = np.arange(0, nframes_per_trial, binsize)
        assert stim_on_frame in bins, "Stim on frame %i not in bins: %s" % (stim_on_frame, str(bins))
        
    return bins, binsize

#%%%
def plot_decoding_performance_trial_epochs(results, bins, names, scoring='accuracy',    
                                           stim_on_frame=None, nframes_on=None,
                                           nclasses=None):
    # boxplot algorithm comparison
    fig = pl.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    ax = sns.boxplot(data=results, showmeans=True, linewidth=0.5)
    # Add transparency to colors
    for patch in ax.artists:
     r, g, b, a = patch.get_facecolor()
     patch.set_facecolor((r, g, b, .3))
    sns.swarmplot(data=results, ax=ax) #, color="grey")
    
    #sns.violinplot(data=results)
    ax.set_xticklabels(names)
    pl.ylabel(scoring)

    pl.ylim([0, .8])
    sns.despine(offset=4, trim=True)
    if stim_on_frame is not None and nframes_on is not None:
        # Identify the time bin(s) in which the stimulus is on:
        #stim_on_frame = run_info['stim_on_frame']
        #nframes_on = int(round(run_info['nframes_on']))
        last_on_frame = stim_on_frame + nframes_on
        
        # Color the bins in which the stimulus is one:
        stimulus_bins = np.array([bins[i] for i in range(len(bins)) if stim_on_frame <= bins[i] <= last_on_frame])
        xlabel_colors = ['r' if v in stimulus_bins else 'k' for v in sorted(bins) ]
        #ax.plot(stimulus_bins, np.ones(stimulus_bins.shape)*-0.05, 'r')
        [t.set_color(i) for (i,t) in zip(xlabel_colors,ax.xaxis.get_ticklabels())]
    
    if nclasses is not None:
        chance_level = float(1./nclasses)
        ax.axhline(y=chance_level, linestyle='--', linewidth=1.0, color='k')
    
    pl.xlabel('trial epoch (bin size %i)' % binsize)
    pl.show()
    
    return fig

#%%
def plot_grand_mean_traces(dset_list, response_type='dff', label_list=[], color_list=['b','m','g'], output_dir='/tmp', save_and_close=True):
        
        
    fig, ax = pl.subplots(1) #pl.figure()
    for di,dtrace in enumerate(dset_list):
        pl.plot(dtrace['mean'], color=color_list[di], label=label_list[di])
        pl.fill_between(xrange(len(dtrace['mean'])), dtrace['mean']-dtrace['sem'], dtrace['mean']+dtrace['sem'], alpha=0.5, color=color_list[di])
        pl.plot(dtrace['stimframes'], np.ones(dtrace['stimframes'].shape) * -0.005*(di+1), color=color_list[di])
        
        #longest_kind = max([len(d1trace['mean']), len(d2trace['mean']), len(d3trace['mean'])])
        #framerates = np.mean([d1['run_info'][()]['framerate'], d2['run_info'][()]['framerate'], d2['run_info'][()]['framerate']])
        #tsecs = (xrange(longest_kind) / framerates)
        #stim_on = int(np.mean([d1trace['stimframes'][0], d2trace['stimframes'][0], d3trace['stimframes'][0]]))
        #relative_t = tsecs - tsecs[stim_on]
        #xlabels = [(i, round(t, 2)) for i, t in enumerate(relative_t)]
        #ax.set(xticks = [])
        #ax.set(xticks = [x[0] for x in xlabels])
        #ax.set(xticklabels = [x[1] for x in xlabels])
        #ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=8)
    ax.set(xticks = [])
    ax.set(xticklabels = [])
    ax.set(yticks = [0, 0.1])
    ax.set(ylabel=response_type)
    sns.despine(offset=4, trim=True, bottom=True)
    pl.legend()
    
    figname_base = 'avgtrace_%s' % (response_type)
    run_str = '_'.join([dtrace['run'] for dtrace in dset_list])
    figname = '%s_%s.pdf' % (figname_base, run_str)

    if save_and_close:
        pl.savefig(os.path.join(output_dir, figname))
        pl.close()
        
    return figname

def get_grand_mean_trace(d1, response_type='dff'):
    d = {}
    assert response_type in d1.keys(), "Specified response type (%s) not found. Choose from: %s" % (response_type, str(d1.keys()))
    d1_meanstims = d1[response_type]
    d1_run = os.path.split(d1['run_info'][()]['traceid_dir'].split('/traces')[0])[-1]
    print d1_run, d1_meanstims.shape
    
    # Get run info:
    nrois = d1_meanstims.shape[-1]
    d1_nframes = d1['run_info'][()]['nframes_per_trial']
    d1_tmat = np.reshape(d1_meanstims, (d1_meanstims.shape[0]/d1_nframes, d1_nframes, nrois))
    meantrace_rois1 = np.mean(d1_tmat, axis=0)
    mean_baselines = np.mean(meantrace_rois1[0:d1['run_info'][()]['stim_on_frame'], :], axis=0)
    
    meantrace_rois1 -= mean_baselines
    
    meantrace1 = np.mean(meantrace_rois1, axis=1)
    semtrace1 = stats.sem(meantrace_rois1, axis=1)
    d1_stim_frames = np.array([d1['run_info'][()]['stim_on_frame'], int(round(d1['run_info'][()]['stim_on_frame'] + d1['run_info'][()]['nframes_on']))])
    
    d['run'] = d1_run
    d['mean'] = meantrace1
    d['sem'] = semtrace1
    d['stimframes'] = d1_stim_frames
    return d


#%%
    

def get_input_data(dataset, roi_selector='all', data_type='stat', inputdata='meanstim'):
        
    assert inputdata in dataset.keys(), "Requested input data (%s) not found: %s" % (inputdata, str(dataset.keys()))
    
    # -----------------------------------------------------------------------------
    cX = dataset[inputdata].copy()
    ylabels = dataset['ylabels'].copy()
    
    # Use subset of ROIs, e.g., VISUAL or SELECTIVE only:
    if roi_selector == 'visual':
        visual_rids = get_roi_list(dataset['run_info'][()], roi_selector=roi_selector, metric='meanstimdf')
        cX = cX[:, visual_rids]
    #print cX.shape
    #spatial_rids = [s for s in sorted_rids if s in visual_rids]
    
    nframes_per_trial = dataset['run_info'][()]['nframes_per_trial']
    ntrials_by_cond = dataset['run_info'][()]['ntrials_by_cond']
    ntrials_total = sum([val for k,val in ntrials_by_cond.iteritems()])
    
    # Get default label list for cX (using the values originally assigned to conditions):
    if isinstance(run_info['condition_list'], str):
        cond_labels_all = sorted(run_info['condition_list'], key=natural_keys)
    else:
        cond_labels_all = sorted(run_info['condition_list'])
    
    if data_type == 'xcondsub':
        # Each sample point is a FRAME, total nsamples = nframe_per_trial * nconditions
        cy = np.hstack([np.tile(cond, (nframes_per_trial,)) for cond in sorted(cond_labels_all, key=natural_keys)]) #config_labels.copy()
    elif not data_type == 'frames':
        # Each sample is a TRIAL, total nsamples = ntrials_per_condition * nconditions.
        # Different conditions might have different num of trials, so assign by n trials:
        cy = np.reshape(ylabels, (ntrials_total, nframes_per_trial))[:,0]
    else:
        cy = ylabels.copy()
        
    print 'cX (inputdata):', cX.shape
    print 'cY (labels):', cy.shape
    
    return cX, cy


#%%
#averages_df = []
#normed_df= []
#all_zscores = {}
#all_info = {}

def load_dataset_from_opts(options_list, test=False):
    #averages_df = []
    #normed_df= []
    #all_zscores = {}
    #all_info = {}
    
    #options_idx = 1
    #options_idx = 0
    
    data_paths = {}
    for options_idx in range(len(options_list)):
        #%
        print "**********************************"
        print "Processing %i of %i runs." % (options_idx, len(options_list))
        print "**********************************"
    
        #%
        options = options_list[options_idx]
        traceid_dir = fmt.get_traceid_dir(options)
        
        # Set up output dir:
        data_basedir = os.path.join(traceid_dir, 'data_arrays')
        if not os.path.exists(data_basedir):
            os.makedirs(data_basedir)
    
        clf_basedir = os.path.join(traceid_dir, 'classifiers')
        if not os.path.exists(os.path.join(clf_basedir, 'figures')):
            os.makedirs(os.path.join(clf_basedir, 'figures'))
    
        # First check if processed datafile exists:
        reload_data = False
        data_fpath = os.path.join(data_basedir, 'datasets.npz')
        
        data_paths[options_idx] = data_fpath
    
        try:
            dataset = np.load(data_fpath)
            print "Loaded existing datafile:\n%s" % data_fpath
            print dataset.keys()
    
        except Exception as e:
            reload_data = True
    
        if reload_data:
                
            data_fpath = fmt.create_rdata_array(options)
                
            # Set up output dir:
            data_basedir = os.path.join(run_info['traceid_dir'], 'data_arrays')
    
            clf_basedir = os.path.join(run_info['traceid_dir'],  'classifiers')
            if not os.path.exists(clf_basedir):
                os.makedirs(clf_basedir)
            if not os.path.exists(os.path.join(clf_basedir, 'figures')):
                os.makedirs(os.path.join(clf_basedir, 'figures'))
                
            # Also create output dir for population-level figures:
            population_figdir = os.path.join(run_info['traceid_dir'],  'figures', 'population')
            if not os.path.exists(population_figdir):
                os.makedirs(population_figdir)
    
#            #%  Preprocessing, step 1:  Remove cross-condition mean
#            # Select color code for conditions:
#            conditions = run_info['condition_list']
#            color_codes = sns.color_palette("Greys_r", len(conditions)*2)
#            color_codes = color_codes[0::2]
#            
            
            #% Look at example ROI:
    #        if test:
    #            #ridx = 134 # 15 #44 #3 # 'roi00004' #10
    #            mean_cond_traces, mean_tsecs = util.get_mean_cond_traces(ridx, X, y, tsecs, nframes_per_trial)
    #            xcond_mean = np.mean(mean_cond_traces, axis=0)
    #    
    #            pl.figure()
    #            pl.subplot(1,2,1); pl.title('traces')
    #            for t in range(len(conditions)):
    #                pl.plot(mean_tsecs, mean_cond_traces[t, :], c=color_codes[t])
    #            pl.plot(mean_tsecs, np.mean(mean_cond_traces, axis=0), 'r')
    #    
    #            pl.subplot(1,2,2); pl.title('xcond subtracted')
    #            normed = (mean_cond_traces - xcond_mean)
    #            for t in range(len(conditions)):
    #                pl.plot(mean_tsecs, normed[t,:], c=color_codes[t])
    #            pl.plot(mean_tsecs, np.mean(normed, axis=0), 'r')
    #            pl.suptitle('average df/f for each condition, with and w/out xcond norm')
    #    
    #            figname = 'avgtrial_vs_xcondsubtracted_roi%05d.png' % int(ridx+1)
    #            pl.savefig(os.path.join(output_basedir, figname))
    #    
            #tvalues = util.format_roisXvalue(sDATA, run_info, value_type='meanstim')
            #zscores = util.format_roisXvalue(sDATA, run_info, value_type='zscore')
            
    
        dataset = np.load(data_fpath)
    #    averages_list, normed_list = util.get_xcond_dfs(roi_list, X, y, tsecs, run_info)
    #
    #    #if options_idx == 0:
    #    averages_df.extend(averages_list)
    #    normed_df.extend(normed_list)
    #
    #    # Get zscores
    #    if len(options_list) > 1:
    #        all_zscores[options_idx] = zscores
    #        all_info[options_idx] = run_info
    #
    ##%
    ##% Concat into datagrame
    #avgDF = pd.concat(averages_df, axis=1)
    #avgDF.head()
    #
    #normDF = pd.concat(normed_df, axis=1)
    #normDF.head()
    return dataset, data_paths


#%%
    
def get_classifier_id(class_labels, classifier='LinearSVC', cv_method='kfold', inputdata='meanstim', 
                          data_type='stat', binsize='', roi_selector='all', 
                          class_name='', aggregate_type='all', const_trans='', trans_value=''):
        
    class_desc = '%s_%s' % (class_name, aggregate_type)
    if const_trans is not '':
        class_desc = '%s_%s' % (class_desc, const_trans)
    if trans_value is not '':
        class_desc = '%s%i' % (class_desc, trans_value)
        
        
    classifier = 'LinearSVC'
    cv_method = 'kfold'
    #class_labels = sorted(list(set([sconfigs[c][class_name] for c in sconfigs.keys()])))
    if subset == 'two_class':
        nclasses = 2
    elif subset == 'no_morphing':
        nclasses = len([i for i in class_labels if i >= 0])
    else:
        nclasses = len(class_labels)
        
    classif_identifier = '{dtype}_{roiset}_{clf}_{cv}_{}{class_name}_{grouper}_{preprocess}'.format(
                                                                             nclasses, 
                                                                             dtype='%s' % ''.join([data_type, str(binsize)]),
                                                                             roiset='%srois' % roi_selector,
                                                                             clf=classifier,
                                                                             cv=cv_method,
                                                                             class_name=class_name,
                                                                             grouper=aggregate_type,
                                                                             preprocess=inputdata)

    return classif_identifier

#%%
def format_stat_dataset(cX, cy, sconfigs, inputdata='meanstim',
                                          class_name='morphlevel', 
                                          aggregate_type='all', 
                                          subset=None,
                                          nsamples=None,
                                          const_trans='', 
                                          trans_value='', 
                                          relabel=False):
    '''
    Specified ways to group and label classifier data.
    
    Params:
    =======
    'class_name':  (str) 
        - stimulus parameter to decode (e.g., 'morphlevel', 'yrot', 'ori', etc.)
    
    'aggregate_type':  (str)
        - method of grouping (averaging) trials together
        - options:
            'all' = label each trial 
            'single' :  Only include trials of a specific transform type and value,
                        Use this if want to decode 'morphlevel' on trials in which 'xpos'=5.0, for ex.
    
    'subset': (str) or None
        - Different from aggregate_type because no averaging or grouping is done across classes.
        - This is used for taking only a subset of classes *within* the class_name specified.
        - options:
            'two_class' :  This is specific to morphlevel, i.e., only take anchor1 and anchor2 to do a binary classification.
            'no_morphing' :  This is specific to rotating movies, i.e., don't take the funky A-B/B-A morphs (just single object movies)

    'const_trans':  (str) 
        - Transform type to keep constant (i.e., only label instances in which const_trans = trans_value)
    
    'trans_value':  (int, float)
        - Value to set 'const_trans' to.

    '''

    all_trans_types = list(set(sconfigs[sconfigs.keys()[0]].keys()))
    if aggregate_type == 'single':
        assert const_trans in all_trans_types, "Transform type to hold constant (%s), unspecified: %s" % (const_trans, str(all_trans_types)) 

    cX, cy = group_classifier_data(cX, cy, class_name, sconfigs, 
                                       subset=subset,
                                       nsamples=nsamples,
                                       aggregate_type=aggregate_type, 
                                       const_trans=const_trans, 
                                       trans_value=trans_value, 
                                       relabel=False)
    
    cy = np.array([sconfigs[cv][class_name] for cv in cy])
    
    print 'cX (input data):', cX.shape
    print 'cY (labels):', cy.shape
    
    class_labels = sorted(list(set(cy)))
    print "Labels:", class_labels
    
    return cX, cy, class_labels

#%%

def format_epoch_dataset(cX, cy, run_info, class_name, sconfigs, binsize=10, const_trans_values=[]):
#if data_type == 'frames':
#    binsize=10
    bins, binsize = get_trial_bins(run_info['nframes_per_trial'], run_info['stim_on_frame'], binsize=binsize)
    print "Dividing trial into %i epochs (binsize=%i)." % (len(bins), binsize)

#if data_type == 'frames':
    epochs, labels, bins = split_trial_epochs(cX, cy, class_name, sconfigs, run_info, binsize=binsize, relabel=False)
    
    if len(const_trans_values) > 0:
        decode_dict = dict((trans_value, {}) for trans_value in const_trans_values)
        for trans_value in const_trans_values:
            subepochs = epochs.copy() #copy.copy(epochs)
            sublabels = labels.copy()
            kept_configs = [c for c in sconfigs.keys() if sconfigs[c][const_trans]==trans_value]
            config_ixs = [ci for ci,cv in enumerate(labels[0]) if cv in kept_configs]
            # Take subset of trials:
            for c in epochs.keys():
                subepochs[c] = subepochs[c][config_ixs, :]
                sublabels[c] = sublabels[c][config_ixs]
                new_labels = np.array([sconfigs[cv][class_name] for cv in sublabels[c]])
                sublabels[c] = new_labels
                
            decode_dict[trans_value]['epochs'] = subepochs
            decode_dict[trans_value]['labels'] = sublabels
    else:
        decode_dict = {'all': {}}
        new_labels = np.array([sconfigs[cv][class_name] for cv in labels[0]]) # Each classifier has the same labels, since we are comparing epochs
        labels = dict((k, new_labels) for k in labels.keys())
        decode_dict['all']['epochs'] = epochs
        decode_dict['all']['labels'] = labels
        
    class_labels = sorted(list(set(decode_dict[decode_dict.keys()[0]]['labels'][0])))
    print "Labels:", class_labels

    return epochs, decode_dict, bins, binsize
    
#%%
    
def bootstrap_subsets_confusion(dataset, class_name, svc=None,
                                            aggregate_type='half',
                                            nsamples=None,
                                            n_iterations=100,
                                            roi_selector='all', 
                                            data_type='stat', 
                                            inputdata='meanstim',
                                            subset=None,
                                            const_trans='', 
                                            trans_value=''):
    
    cmats = []
    cX_full, cy_full = get_input_data(dataset, roi_selector=roi_selector, data_type=data_type, inputdata=inputdata)
    
    for niter in range(n_iterations):
        if niter % 100 == 0:
            print "... running iter %i of %i." % (niter, n_iterations)
                
        cX, cy, class_labels = format_stat_dataset(cX_full, cy_full, sconfigs, 
                                                     inputdata='meanstim',
                                                     class_name=class_name,
                                                     nsamples=nsamples,
                                                     subset=subset,
                                                     aggregate_type=aggregate_type, 
                                                     const_trans=const_trans, 
                                                     trans_value=trans_value, 
                                                     relabel=False)
        cX_std = StandardScaler().fit_transform(cX)
        
        cv_nfolds = 5        
        training_data = cX_std.copy()
        
        n_samples = cX_std.shape[0]
        print "N samples for CV:", n_samples
    
        # Create classifier:
        # ------------------
        if svc is None:
            if cX_std.shape[0] > cX_std.shape[1]: # nsamples > nfeatures
                dual = False
            else:
                dual = True
                
            svc = LinearSVC(random_state=0, dual=dual, multi_class='ovr', C=1E9)
            #svc = LinearSVC(random_state=0, dual=dual, C=big_C)
        
        loo = cross_validation.StratifiedKFold(cy, n_folds=cv_nfolds, shuffle=True)
        pred_results = []
        pred_true = []
        for train, test in loo: #, groups=groups):
            #print train, test
            X_train, X_test = training_data[train], training_data[test]
            y_train, y_test = cy[train], cy[test]
            y_pred = svc.fit(X_train, y_train).predict(X_test)
            pred_results.append(y_pred) #=y_test])
            pred_true.append(y_test)
    
    
        #y_pred = np.array([i[0] for i in pred_results])
        #y_test = np.array([i[0] for i in pred_true])
    
        #avg_score = np.array([int(p==t) for p,t in zip(pred_results, pred_true)]).mean()
    
        
        # Compute confusion matrix:
        # -----------------------------------------------------------------------------
        cmatrix_tframes = confusion_matrix(pred_true[0], pred_results[0], labels=class_labels)
        for iter_idx in range(len(pred_results))[1:]:
            print "adding iter %i" % iter_idx
            cmatrix_tframes += confusion_matrix(pred_true[iter_idx], pred_results[iter_idx], labels=class_labels)

        #print pred_true[0]

        cmat = cmatrix_tframes / len(pred_results)
        
        cmats.append(cmat)
#        if niter == 0:
#            cmat = cmatrix_tframes / len(pred_results)
#        else:
#            cmat += (cmatrix_tframes / len(pred_results))
            
    return cmats, class_labels



#%%
def extract_options(options):

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
    parser.add_option('-T', '--trace-type', action='store', dest='trace_type',
                          default='raw', help="trace type [default: 'raw']")
    parser.add_option('-R', '--run', dest='run_list', default=[], nargs=1,
                          action='append',
                          help="run ID in order of runs")
    parser.add_option('-t', '--traceid', dest='traceid_list', default=[], nargs=1,
                          action='append',
                          help="trace ID in order of runs")
    parser.add_option('-n', '--nruns', action='store', dest='nruns', default=1, help="Number of consecutive runs if combined")
    parser.add_option('--slurm', action='store_true', dest='slurm', default=False, help="set if running as SLURM job on Odyssey")
    parser.add_option('--par', action='store_true', dest='multiproc', default=False, help="set if want to run MP on roi stats, when possible")
    parser.add_option('--nproc', action='store', dest='nprocesses', default=4, help="N processes if running in par (default=4)")

    parser.add_option('--combo', action='store_true', dest='combined', default=False, help="Set if using combined runs with same default name (blobs_run1, blobs_run2, etc.)")


    # Pupil filtering info:
    parser.add_option('--no-pupil', action="store_false",
                      dest="filter_pupil", default=True, help="Set flag NOT to filter PSTH traces by pupil threshold params")
    parser.add_option('-s', '--radius-min', action="store",
                      dest="pupil_radius_min", default=25, help="Cut-off for smnallest pupil radius, if --pupil set [default: 25]")
    parser.add_option('-B', '--radius-max', action="store",
                      dest="pupil_radius_max", default=65, help="Cut-off for biggest pupil radius, if --pupil set [default: 65]")
    parser.add_option('-d', '--dist', action="store",
                      dest="pupil_dist_thr", default=5, help="Cut-off for pupil distance from start, if --pupil set [default: 5]")

    (options, args) = parser.parse_args(options)

    return options

#%%

#options = ['-D', '/mnt/odyssey', '-i', 'CE077', '-S', '20180425', '-A', 'FOV2_zoom1x',
#           '-T', 'np_subtracted', '--no-pupil',
#           '-R', 'gratings_run1', '-t', 'traces001',
#           '-n', '1']


# 20180518 -- CE077 data:
# -----------------------------------------------------------------------------
#opts1 = ['-D', '/mnt/odyssey', '-i', 'CE077', '-S', '20180518', '-A', 'FOV1_zoom1x',
#           '-T', 'np_subtracted', '--no-pupil',
#           '-R', 'gratings_run1', '-t', 'traces002',
#           '-n', '1']
#
#opts3 = ['-D', '/mnt/odyssey', '-i', 'CE077', '-S', '20180518', '-A', 'FOV1_zoom1x',
#           '-T', 'np_subtracted', '--no-pupil',
#           '-R', 'blobs_dynamic_run3', '-t', 'traces002',
#           '-n', '1']
#
#opts2 = ['-D', '/mnt/odyssey', '-i', 'CE077', '-S', '20180518', '-A', 'FOV1_zoom1x',
#           '-T', 'np_subtracted', '--no-pupil',
#           '-R', 'blobs_run6', '-t', 'traces002',
#           '-n', '1']
#
#
#opts1 = ['-D', '/mnt/odyssey', '-i', 'CE077', '-S', '20180518', '-A', 'FOV1_zoom1x',
#           '-T', 'np_subtracted', '--no-pupil',
#           '-R', 'blobs_run2', '-t', 'traces002',
#           '-n', '1']


# 20180521 -- CE077 datasets:
# -----------------------------------------------------------------------------
opts0 = ['-D', '/mnt/odyssey', '-i', 'CE077', '-S', '20180521', '-A', 'FOV2_zoom1x',
           '-T', 'np_subtracted', '--no-pupil',
           '-R', 'gratings_run1', '-t', 'traces001',
           '-n', '1']

opts2 = ['-D', '/mnt/odyssey', '-i', 'CE077', '-S', '20180521', '-A', 'FOV2_zoom1x',
           '-T', 'np_subtracted', '--no-pupil',
           '-R', 'blobs_run2', '-t', 'traces001',
           '-n', '1']

opts1 = ['-D', '/mnt/odyssey', '-i', 'CE077', '-S', '20180521', '-A', 'FOV2_zoom1x',
           '-T', 'np_subtracted', '--no-pupil',
           '-R', 'blobs_run1', '-t', 'traces001',
           '-n', '1']

opts3 = ['-D', '/mnt/odyssey', '-i', 'CE077', '-S', '20180521', '-A', 'FOV2_zoom1x',
           '-T', 'np_subtracted', '--no-pupil',
           '-R', 'blobs_dynamic_run1', '-t', 'traces001',
           '-n', '1']


# 20180523 -- CE077 datasets:
# -----------------------------------------------------------------------------

opts4 = ['-D', '/mnt/odyssey', '-i', 'CE077', '-S', '20180523', '-A', 'FOV1_zoom1x',
           '-T', 'np_subtracted', '--no-pupil',
           '-R', 'blobs_run2', '-t', 'traces001',
           '-n', '1']

opts3 = ['-D', '/mnt/odyssey', '-i', 'CE077', '-S', '20180523', '-A', 'FOV1_zoom1x',
           '-T', 'np_subtracted', '--no-pupil',
           '-R', 'blobs_run1', '-t', 'traces001',
           '-n', '1']

opts5 = ['-D', '/mnt/odyssey', '-i', 'CE077', '-S', '20180523', '-A', 'FOV1_zoom1x',
           '-T', 'np_subtracted', '--no-pupil',
           '-R', 'blobs_dynamic_run1', '-t', 'traces001',
           '-n', '1']



# 20180602 -- CE077 datasets:
# -----------------------------------------------------------------------------
opts6 = ['-D', '/mnt/odyssey', '-i', 'CE077', '-S', '20180602', '-A', 'FOV1_zoom1x',
           '-T', 'np_subtracted', '--no-pupil',
           '-R', 'blobs_run2', '-t', 'traces002',
           '-n', '1']

opts6 = ['-D', '/mnt/odyssey', '-i', 'CE077', '-S', '20180602', '-A', 'FOV1_zoom1x',
           '-T', 'np_subtracted', '--no-pupil',
           '-R', 'blobs_dynamic_run6', '-t', 'traces001',
           '-n', '1']

#%%


#opts = ['-D', '/mnt/odyssey', '-i', 'CE077', '-S', '20180609', '-A', 'FOV1_zoom1x',
#           '-T', 'np_subtracted', '--no-pupil',
#           '-R', 'blobs_run3', '-t', 'traces001',
#           '-n', '1']
opts = ['-D', '/mnt/odyssey', '-i', 'CE077', '-S', '20180516', '-A', 'FOV1_zoom1x',
           '-T', 'np_subtracted', '--no-pupil',
           '-R', 'objects_run2', '-t', 'traces001',
           '-n', '1']

options_list = [opts] #, opts2]






#%%
    
dataset, data_paths = load_dataset_from_opts(options_list)

#%% Look at distN of responses:

plot_grand = False

if plot_grand:
    dset_list = []
    response_type = 'dff'
    
#    if len(data_paths.keys()) > 1:
    for di, dpath in data_paths.items():
        d1 = np.load(dpath)
        # Load data:
        dtrace = get_grand_mean_trace(d1, response_type=response_type)
        dset_list.append(dtrace)
    
    # PLOT:
    a_run_dir = data_paths[0].split('/traces')[0]
    acquisition_dir = os.path.split(a_run_dir)[0]
    figname = plot_grand_mean_traces(dset_list, response_type=response_type,
                                         label_list=[dtrace['run'] for dtrace in dset_list], 
                                         output_dir=acquisition_dir, 
                                         save_and_close=False)
    
    pl.savefig(os.path.join(acquisition_dir, figname))
    


#%%
# =============================================================================
# Load ROI masks to sort ROIs by spatial distance
# =============================================================================

spatial_sort = False
optsE = extract_options(options_list[0])

if spatial_sort:
    traceid_dir = dataset['run_info'][()]['traceid_dir']
    sorted_rids, cnts, zproj = util.sort_rois_2D(traceid_dir)
    util.plot_roi_contours(zproj, sorted_rids, cnts, clip_limit=0.005, label=False)
    figname = 'spatially_sorted_rois_%s.png' % optsE.acquisition
    acquisition_dir = os.path.join(optsE.rootdir, optsE.animalid, optsE.session, optsE.acquisition)
    pl.savefig(os.path.join(acquisition_dir, figname))
    

#%%
   
run_info = dataset['run_info'][()]
sconfigs = dataset['sconfigs'][()]

print run_info['ntrials_by_cond']

traceid_dir = run_info['traceid_dir']
optsE = extract_options(options_list[0])
if optsE.rootdir not in traceid_dir:
    traceid_dir = replace_root(traceid_dir, optsE.rootdir, optsE.animalid, optsE.session)
#%%

# #############################################################################
# Randomly sample N trials per condition and compare classifier performance
# #############################################################################

data_type = 'stat'
roi_selector = 'all'
inputdata = 'meanstim'
    
class_name = 'object'

aggregate_type = 'half'
subset = None

const_trans = ''
trans_value = ''

# Create output dir:

class_labels = sorted(list(set([sconfigs[c][class_name] for c in sconfigs.keys()])))
print class_labels

classif_identifier = get_classifier_id(class_labels, inputdata=inputdata, data_type=data_type, roi_selector=roi_selector,
                                           class_name=class_name, aggregate_type=aggregate_type,
                                           const_trans=const_trans, trans_value=trans_value)

classifier_dir = os.path.join(traceid_dir, 'classifiers', classif_identifier)
if not os.path.exists(classifier_dir):
    os.makedirs(classifier_dir)
else:
    print "DIR exists!", classifier_dir

niters = 100
#nsamples_test = [5, 10, 15, 20]
nsamples_test = [4, 6, 8, 10]

#svc = LinearSVC(random_state=0, dual=dual, multi_class='ovr', C=1E9)
means = []; sems = []
for nsamples in nsamples_test:
    cmat, class_labels = bootstrap_subsets_confusion(dataset, class_name, svc=None,
                                                     aggregate_type=aggregate_type,
                                                     nsamples=nsamples,
                                                     n_iterations=niters,
                                                     roi_selector=roi_selector, 
                                                     data_type=data_type, 
                                                     inputdata=inputdata)
    
    normed_cms = [cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] for cm in cmat]
    
    gmean = np.mean(np.dstack(normed_cms), axis=-1)
    gstd = np.std(np.dstack(normed_cms), axis=-1)
    
    mean_correct = np.diagonal(gmean)
    sem_correct = np.diagonal(gstd)

    means.append(mean_correct)
    sems.append(sem_correct)

print("Normalized confusion matrix")

nconds = len(means[0])


#cmap255 = [tuple(i*255 for i in cmap[c]) for c in range(nconds)]
cmap = sns.cubehelix_palette(as_cmap=True)
if isinstance(class_labels[0], str) or isinstance(class_labels[0], unicode):
    color_levels = xrange(len(class_labels))
else:
    color_levels = class_labels
    
fig, ax = pl.subplots()
for idx, nsamples in enumerate(sorted(nsamples_test)):    
    pts = ax.scatter(np.ones(means[idx].shape)*nsamples, means[idx], c=color_levels, cmap=cmap)
    #ax.errorbar(np.ones(means[idx].shape)*nsamples, means[idx], yerr=sems[idx])
fig.colorbar(pts)


    
#cmap1 = sns.color_palette("Blues", nconds)
cmap1 = sns.color_palette("magma", nconds)
fig, ax = pl.subplots()
for idx in range(len(nsamples_test)):
    ax.errorbar(color_levels, means[idx], yerr=sems[idx], c=cmap1[idx], label=nsamples_test[idx], capsize=5)
    ax.xaxis.set_major_locator(pl.MaxNLocator(nconds))
    
pl.legend()
sns.despine(offset=4, trim=True)
pl.ylabel('accuracy')
pl.xlabel('labels')

figname = 'accuracy_%iiters_%itiers.png' % (niters, len(nsamples_test))
pl.savefig(os.path.join(classifier_dir, figname))


#%


#%%
# =============================================================================
# Assign cX -- input data for classifier (nsamples x nfeatures)
# Get corresponding labels for each sample.
# =============================================================================


roi_selector = 'all' #'all' #'selectiveanova' #'selective'
data_type = 'stat' #'zscore' #zscore' # 'xcondsub'
inputdata = 'meanstim'

cX, cy = get_input_data(dataset, roi_selector=roi_selector, data_type=data_type, inputdata=inputdata)


#%
# =============================================================================
# SPECIFIY CLASSIFIER:
# =============================================================================

# Group configIDs by selected class labels to sort labels in order:
class_name = 'xpos' #'morphlevel' #'ori' #'xpos' #morphlevel' #'ori' # 'morphlevel'
aggregate_type = 'all' #'all' #'all' #'half' #all' #'each' #'each' # 'single' #'each' #'single' #'each'
subset = None# 'two_class' #None #'two_class' # None # 'two_class' #no_morphing' #'no_morphing' # None

const_trans = '' #'xpos' #'xpos' #None #'xpos' #'xpos' #None#'xpos' #None #'xpos' #None# 'morphlevel' # 'xpos'
trans_value = '' #-5 #-5 #16 #None #'-5' #None #-5 #None #-5 #None

binsize=''

cv_method = 'kfold'
classifier = 'LinearSVC'

if data_type == 'frames':
    const_trans_values = []
    if not const_trans=='':
        const_trans_values = sorted(list(set([sconfigs[c][const_trans] for c in sconfigs.keys()])))
        print "Selected subsets of transform: %s. Found values %s" % (const_trans, str(const_trans_values))
        
    epochs, decode_dict, bins, binsize = format_epoch_dataset(cX, cy, run_info, class_name, sconfigs, 
                                                                  binsize=binsize, 
                                                                  const_trans_values=const_trans_values)
else:
     cX, cy, class_labels = format_stat_dataset(cX, cy, sconfigs, 
                                                 inputdata='meanstim',
                                                 class_name=class_name,
                                                 subset=subset,
                                                 aggregate_type=aggregate_type, 
                                                 const_trans=const_trans, 
                                                 trans_value=trans_value, 
                                                 relabel=False)
     
     cX_std = StandardScaler().fit_transform(cX)
     
#% Create output dir for current classifier:
classif_identifier = get_classifier_id(class_labels, inputdata=inputdata, data_type=data_type, roi_selector=roi_selector,
                                           class_name=class_name, aggregate_type=aggregate_type,
                                           const_trans=const_trans, trans_value=trans_value)

classifier_dir = os.path.join(traceid_dir, 'classifiers', classif_identifier)

if aggregate_type == 'single':
    clf_subdir = '%s_%s' % (const_trans, str(trans_value))
    classifier_dir = os.path.join(classifier_dir, clf_subdir)
    
if not os.path.exists(classifier_dir):
    os.makedirs(classifier_dir)
else:
    print "DIR exists!", classifier_dir


#%%
# =============================================================================
# Assign cy -- labels for classifier to learn (nsamples,)
# =============================================================================
#
#if data_type == 'frames':
#    epochs, labels, bins = split_trial_epochs(cX, cy, class_name, sconfigs, run_info, binsize=binsize, relabel=False)
#    
#    if len(const_trans_values) > 0:
#        decode_dict = dict((trans_value, {}) for trans_value in const_trans_values)
#        for trans_value in const_trans_values:
#            subepochs = epochs.copy() #copy.copy(epochs)
#            sublabels = labels.copy()
#            kept_configs = [c for c in sconfigs.keys() if sconfigs[c][const_trans]==trans_value]
#            config_ixs = [ci for ci,cv in enumerate(labels[0]) if cv in kept_configs]
#            # Take subset of trials:
#            for c in epochs.keys():
#                subepochs[c] = subepochs[c][config_ixs, :]
#                sublabels[c] = sublabels[c][config_ixs]
#                new_labels = np.array([sconfigs[cv][class_name] for cv in sublabels[c]])
#                sublabels[c] = new_labels
#                
#            decode_dict[trans_value]['epochs'] = subepochs
#            decode_dict[trans_value]['labels'] = sublabels
#    else:
#        decode_dict = {'all': {}}
#        new_labels = np.array([sconfigs[cv][class_name] for cv in labels[0]]) # Each classifier has the same labels, since we are comparing epochs
#        labels = dict((k, new_labels) for k in labels.keys())
#        decode_dict['all']['epochs'] = epochs
#        decode_dict['all']['labels'] = labels
#        
#    class_labels = sorted(list(set(decode_dict[decode_dict.keys()[0]]['labels'][0])))
#    print "Labels:", class_labels
#    
#else:
#    cX, cy = group_classifier_data(cX, cy, class_name, sconfigs, 
#                                       subset=subset,
#                                       aggregate_type=aggregate_type, 
#                                       const_trans=const_trans, 
#                                       trans_value=trans_value, 
#                                       relabel=False)
#    
#    cy = np.array([sconfigs[cv][class_name] for cv in cy])
#    
#    print 'cX (%s):' % data_type, cX.shape
#    print 'cY (labels):', cy.shape
#    
#    class_labels = sorted(list(set(cy)))
#    print "Labels:", class_labels

#%%
niters = 10
if data_type == 'frames':
    for view_key in decode_dict.keys():
        
        epochs = decode_dict[view_key]['epochs']
        labels = decode_dict[view_key]['labels']
        
        # Create pipeline:
        # ------------------
        big_C = 1e9
        if epochs[0].shape[0] > epochs[0].shape[1]: # nsamples > nfeatures
            dual = False
        else:
            dual = True
        #pipe_svc = Pipeline([('scl', StandardScaler()),
        #                     ('clf', LinearSVC(random_state=0, dual=dual, multi_class='ovr', C=big_C))]) #))])
        
        for runiter in np.arange(0,niters): #range(5):
            # evaluate each model in turn
            results = []
            names = []
            scoring = 'accuracy'
            for idx, curr_epoch in enumerate(sorted(epochs.keys())):
                model = LinearSVC(random_state=0, dual=dual, multi_class='ovr', C=big_C)
                kfold = StratifiedKFold(n_splits=5, shuffle=True)
                x_std = StandardScaler().fit_transform(epochs[curr_epoch])
                cv_results = cross_val_score(model, x_std, labels[curr_epoch], cv=kfold, scoring=scoring)
                results.append(cv_results)
                names.append(curr_epoch)
                msg = "%s: %f (%f)" % (curr_epoch, cv_results.mean(), cv_results.std())
                #print(msg)
    
            fig = plot_decoding_performance_trial_epochs(results, bins, names, 
                                                       scoring=scoring, 
                                                       stim_on_frame=run_info['stim_on_frame'],
                                                       nframes_on=run_info['nframes_on'],
                                                       nclasses=len(class_labels))
    
            fig.suptitle('Estimator Comparison: label %s (%s: %s)' % (aggregate_type, const_trans, str(view_key)))
            figname = '%s_trial_epoch_classifiers_binsize%i_%s_label_%s_iter%i.png' % (view_str, binsize, scoring, class_name, runiter)
            pl.savefig(os.path.join(classifier_dir, figname))
            pl.close()



#%%
#    data_X = dataset['meanstim']
#    data_X = StandardScaler().fit(data_X)
#    data_y = dataset['ylabels']
#    ntrials_total = dataset['run_info'][()]['ntrials_total']
#    nframes_per_trial= dataset['run_info'][()]['nframes_per_trial']
#    
#    if data_type != 'frames':
#        data_y = np.reshape(data_y, (ntrials_total, nframes_per_trial))[:,0]
#    sconfigs = dataset['sconfigs'][()]
    
#    X, y = group_classifier_data(data_X, data_y, class_name, sconfigs, 
#                                       subset=subset,
#                                       aggregate_type=aggregate_type, 
#                                       const_trans=const_trans, 
#                                       trans_value=trans_value, 
#                                       relabel=relabel)
#    
#    y = np.array([sconfigs[cv][class_name] for cv in y])
#    
#    class_labels = sorted(list(set(y)))
            
            
# Get dimensions of session dataset:
nframes_per_trial = run_info['nframes_per_trial']
#nconditions = len(run_info['condition_list'])
nrois = cX.shape[1]

            
# Also create output dir for population-level figures:
population_figdir = os.path.join(traceid_dir,  'figures', 'population')
if not os.path.exists(population_figdir):
    os.makedirs(population_figdir)
            

# Correlation between stim classes across ROIs -- compute as RDM:
# -----------------------------------------------------------------------------

zdf_cmap = 'PRGn' #pl.cm.hot

df_list = []
for label in class_labels:
    tidxs = np.where(cy==label)             # Get indices of trials (or frames) with current label
    curr_samples = np.squeeze(cX[tidxs,:])  # nsamples_current_label x nrois
    cname = str(label)
    currstim = np.reshape(curr_samples, (curr_samples.shape[0]*curr_samples.shape[1],), order='F') # Reshape into 1D vector for current label
    df_list.append(pd.DataFrame({cname: currstim}))
df = pd.concat(df_list, axis=1)
df.head()

corrs = df.corr(method='pearson')
pl.figure()
sns.heatmap(1-corrs, cmap=zdf_cmap, vmax=2.0)
#sns.heatmap(1-corrs, cmap=zdf_cmap, vmin=0.7, vmax=1.3)

pl.title('RDM (%s) - %s: %s' % (data_type, class_name, aggregate_type))
figname = 'RDM_%srois_%s_classes_%i%s_%s.png' % (roi_selector, data_type, len(class_labels), class_name, aggregate_type)

pl.savefig(os.path.join(population_figdir, figname))
pl.close()

#pl.figure()
#sns.heatmap(corrs, cmap='PRGn', vmin=-1, vmax=1)

# Look at RDM within trial:
# -----------------------------------------------------------------------------

df_list = []
cnames = []
for label in class_labels:
    tidxs = np.where(cy==label)[0]
    currtrials = np.squeeze(cX[tidxs,:])
    if data_type == 'xcondsub':
        ax_labels = np.hstack([np.tile(i, (nframes_per_trial,)) for i in range(len(tidxs)/nframes_per_trial)])
        new_ax_labels = ['%s_%s' % (str(label), ti) for ti in ax_labels]
        cnames.extend(new_ax_labels)
        cell_unit = 'frames'
    else:
        #new_ax_labels = ['%s_%s' % (str(label), ti) for ti in range(len(tidxs))]
        new_ax_labels = ['%s' % (str(label)) for ti in range(len(tidxs))]
        cnames.extend(new_ax_labels)
        cell_unit = 'trials'
    df_list.append(pd.DataFrame(data=currtrials.T, columns=new_ax_labels))

df_trials = pd.concat(df_list, axis=1)

corrs = df_trials.corr(method='pearson')
pl.figure()
ax = sns.heatmap(1-corrs, vmax=2, cmap=zdf_cmap) #, vmax=1)

indices = { value : [ i for i, v in enumerate(cnames) if v == value ] for value in list(set(cnames)) }
xtick_indices = [(k, indices[k][-1]) for k in list(set(cnames))]
ax.set(xticks = [])
ax.set(xticks = [x[1] for x in xtick_indices])
ax.set(xticklabels = [x[0] for x in xtick_indices])
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=8)
ax.set(yticks = [])
ax.set(yticks = [x[1] for x in xtick_indices])
ax.set(yticklabels = [x[0] for x in xtick_indices])
ax.set_yticklabels(ax.get_yticklabels(), rotation=45, fontsize=8)
ax.set(xlabel=cell_unit)

pl.title('RDM (%s, %s)' % (data_type, cell_unit))
figname = 'RDM_%srois_%s_classes_%i%s_%s_%s.png' % (roi_selector, data_type, len(class_labels), class_name, aggregate_type, cell_unit)
pl.savefig(os.path.join(population_figdir, figname))
pl.close()

#%% Create input data for classifier:

# -----------------------------------------------------------------------------
# Whitening:
# -----------------------------------------------------------------------------

#
#cX_std = StandardScaler().fit_transform(cX)
#print cX_std.shape
#
#pl.figure();
#sns.distplot(cX.ravel(), label=data_type)
#sns.distplot(cX_std.ravel(), label = 'standardized')
#pl.legend()
#pl.title('distN of combined ROIs for FOV1, FOV2')
#pl.savefig(os.path.join(clf_basedir, '%s_roi_zscore_distributions.png' % roi_selector))

  
#%% Train and test linear SVM using zscored response values:

# #############################################################################
# Set classifier type and CV params:
# #############################################################################

big_C = 1e9
#
scoring = 'accuracy'
cv_nfolds = 5


kfold = StratifiedKFold(n_splits=cv_nfolds, shuffle=True)

# Create classifier:
# ------------------
if cX_std.shape[0] > cX_std.shape[1]: # nsamples > nfeatures
    dual = False
else:
    dual = True
    
svc = LinearSVC(random_state=0, dual=dual, multi_class='ovr', C=big_C)
#svc = LinearSVC(random_state=0, dual=dual, C=big_C)

cv_results = cross_val_score(svc, cX_std, cy, cv=kfold, scoring=scoring)

curr_epoch = '0'
msg = "%s: %f (%f)" % (curr_epoch, cv_results.mean(), cv_results.std())

score, permutation_scores, pvalue = permutation_test_score(
    svc, cX_std, cy, scoring=scoring, cv=kfold, n_permutations=250, n_jobs=4)

print("Classification score %s (pvalue : %s)" % (score, pvalue))

# -----------------------------------------------------------------------------
# How significant is our classification score(s)?
# Calculate p-value as percentage of runs for which obtained score is greater 
# than the initial classification score (i.e., repeat classification after
# randomizing and permuting labels).
# -----------------------------------------------------------------------------
pl.figure()
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
pl.show()

figname = 'cv_permutation_test3.png'
pl.savefig(os.path.join(classifier_dir, figname))
#pl.close()

#%%
# -----------------------------------------------------------------------------
# Format cross-validation data
# -----------------------------------------------------------------------------

#grid_values = {'penalty': ['l1','l2'], 'C': [0.001,0.01,0.1,1,10,100,1000]}
#model_lr = GridSearchCV(lr, param_grid=grid_values)

cv_nfolds = 5
cv_ngroups = 1

training_data = cX_std.copy()

n_samples = cX_std.shape[0]
print "N samples for CV:", n_samples

# Cross-validate for t-series samples:
if cv_method=='splithalf':

    # Fit with subset of data:
    svc.fit(training_data[:n_samples // 2], cy[:n_samples // 2])

    # Now predict the value of the class label on the second half:
    y_test = cy[n_samples // 2:]
    y_pred = svc.predict(training_data[n_samples // 2:])

elif cv_method=='kfold':
    loo = cross_validation.StratifiedKFold(cy, n_folds=cv_nfolds, shuffle=True)
    pred_results = []
    pred_true = []
    for train, test in loo: #, groups=groups):
        print train, test
        X_train, X_test = training_data[train], training_data[test]
        y_train, y_test = cy[train], cy[test]
        y_pred = svc.fit(X_train, y_train).predict(X_test)
        pred_results.append(y_pred) #=y_test])
        pred_true.append(y_test)

else:
    if cv_method=='LOGO':
        loo = LeaveOneGroupOut()
        if data_type == 'xcondsub':
            ngroups = len(cy) / nframes_per_trial
            groups = np.hstack(np.tile(f, (nframes_per_trial,)) for f in range(ngroups))
    elif cv_method=='LOO':
        loo = LeaveOneOut()
        groups=None
    elif cv_method=='LPGO':
        loo = LeavePGroupsOut(5)

    pred_results = []
    pred_true = []

    for train, test in loo.split(training_data, cy, groups=groups):
        print train, test
        X_train, X_test = training_data[train], training_data[test]
        y_train, y_test = cy[train], cy[test]

        y_pred = svc.fit(X_train, y_train).predict(X_test)

        pred_results.append(y_pred) #=y_test])
        pred_true.append(y_test)

    
    if groups is not None:
        # Find "best fold"?
        avg_scores = []
        for y_pred, y_test in zip(pred_results, pred_true):
            pred_score = len([i for i in y_pred==y_test if i]) / float(len(y_pred))
            avg_scores.append(pred_score)
        best_fold = avg_scores.index(np.max(avg_scores))
        folds = [i for i in enumerate(loo.split(training_data, cy, groups=groups))]
        train = folds[best_fold][1][0]
        test = folds[best_fold][1][1]
        X_train, X_test = training_data[train], training_data[test]
        y_train, y_test = cy[train], cy[test]
        y_pred = pred_results[best_fold]
    
        pl.figure();
        sns.distplot(avg_scores, kde=False)
    
    else:
        y_pred = np.array([i[0] for i in pred_results])
        y_test = np.array([i[0] for i in pred_true])

    avg_score = np.array([int(p==t) for p,t in zip(pred_results, pred_true)]).mean()


print("Classification report for classifier %s:\n%s\n"
      % (svc, metrics.classification_report(y_test, y_pred)))

#%

# Compute confusion matrix:
# -----------------------------------------------------------------------------

average_iters = True

if (cv_method == 'LOO' or cv_method == 'splithalf') and (data_type != 'xcondsub'):
    cmatrix_tframes = confusion_matrix(y_test, y_pred, labels=class_labels)
    conf_mat_str = 'trials'
else:
    if average_iters:
        cmatrix_tframes = confusion_matrix(pred_true[0], pred_results[0], labels=class_labels)
        for iter_idx in range(len(pred_results))[1:]:
            print "adding iter %i" % iter_idx
            cmatrix_tframes += confusion_matrix(pred_true[iter_idx], pred_results[iter_idx], labels=class_labels)
        conf_mat_str = 'AVG'
        #cmatrix_tframes /= float(len(pred_results))
    else:
        cmatrix_tframes = confusion_matrix(pred_true[best_fold], pred_results[best_fold], labels=class_labels)
        conf_mat_str = 'best'


#% Plot confusion matrix:
# -----------------------------------------------------------------------------
sns.set_style('white')
fig = pl.figure(figsize=(10,4))
ax1 = fig.add_subplot(1,2,1)
plot_confusion_matrix(cmatrix_tframes, classes=class_labels, ax=ax1, normalize=False,
                  title='Confusion matrix (%s, %s)' % (conf_mat_str, cv_method))

ax2 = fig.add_subplot(1,2,2)
plot_confusion_matrix(cmatrix_tframes, classes=class_labels, ax=ax2, normalize=True,
                      title='Normalized')

#%%
figname = '%s__confusion_%s_iters.png' % (classif_identifier, conf_mat_str)
pl.savefig(os.path.join(classifier_dir, figname))


#%

# Save CV info:
# -----------------------------------------------------------------------------
cv_outfile = '%s__CV_report.txt' % classif_identifier

f = open(os.path.join(classifier_dir, cv_outfile), 'w')
f.write(metrics.classification_report(y_test, y_pred, target_names=[str(c) for c in class_labels]))
f.close()

cv_results = {'predicted': list(y_pred),
              'true': list(y_test),
              'classifier': classifier,
              'cv_method': cv_method,
              'ngroups': cv_ngroups,
              'nfolds': cv_nfolds
              }
cv_resultsfile = '%s__CV_results.json' % classif_identifier
with open(os.path.join(classifier_dir, cv_resultsfile), 'w') as f:
    json.dump(cv_results, f, sort_keys=True, indent=4)


#%% 
    
# Visualize feature weights:
# =============================================================================
    
# svc.coef_ :  array of shape [n_classes, n_features] (if n_classes=2, shape [n_features])
# svc.coef_ :  array of shape [n_classes, n_features] (if n_classes=2, shape [n_features])

def print_top10(vectorizer, clf, class_labels):
    """Prints features with the highest coefficient values, per class"""
    feature_names = vectorizer.get_feature_names()
    for i, class_label in enumerate(class_labels):
        top10 = np.argsort(clf.coef_[i])[-10:]
        print("%s: %s" % (class_label,
              " ".join(feature_names[j] for j in top10)))

def plot_coefficients(classifier, feature_names, class_idx=0, top_features=20):
    if len(classifier.classes_) >= 3:
        coef = classifier.coef_[class_idx, :].ravel()
    else:
        coef = classifier.coef_.ravel()
    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    # create plot
    sns.set_style('white')
    pl.figure(figsize=(15, 5))
    colors = ['r' if c < 0 else 'b' for c in coef[top_coefficients]]
    sns.barplot(np.arange(2 * top_features), coef[top_coefficients], palette=colors) #colors)
    sns.despine(offset=4, trim=True)

    feature_names = np.array(feature_names)
    pl.xticks(np.arange(0, 1 + 2 * top_features), feature_names[top_coefficients], rotation=45, ha='right')
    pl.title('Sorted weights (top %i), class %i' % (top_features, class_idx))
 #plt.show()

#plot_coefficients(svc, svc.classes_)

def plot_weight_matrix(svc, absolute_value=True):
        
    if absolute_value:
        weight_values_ref = np.abs(svc.coef_[0,:])
        cmap='magma'
    else:
        weight_values_ref = svc.coef_[0]
        cmap='RdBu'
        
    sorted_weights = np.argsort(weight_values_ref)[::-1]
    
    sorted_weight_matrix = np.empty(svc.coef_.shape)
    if len(svc.classes_) < 3:
        if absolute_value:
            sorted_weight_matrix[0,:] = np.abs(svc.coef_[0,sorted_weights])
        else:
            sorted_weight_matrix[0,:] = svc.coef_[0,sorted_weights]
    else:
        for ci,cv in enumerate(svc.classes_):
            if absolute_value:
                sorted_weight_matrix[ci,:] = np.abs(svc.coef_[ci,sorted_weights])
            else:
                sorted_weight_matrix[ci,:] = svc.coef_[ci,sorted_weights]
        
    fig, ax = pl.subplots(figsize=(50,5))
    cbar_ax = fig.add_axes([.905, .3, .005, .5])
    g = sns.heatmap(sorted_weight_matrix, cmap=cmap, ax=ax, cbar_ax = cbar_ax, cbar=True)
    for li,label in enumerate(g.get_xticklabels()):
        if li % 20 == 0:
            continue
        label.set_visible(False)
    g.set_xticklabels(g.get_xticklabels(), rotation=45, fontsize=8)
    box = g.get_position()
    g.set_position([box.x0, box.y0 + box.height * 0.2,
                     box.width, box.height * 0.8])
    g.set_xlabel('features')
    g.set_ylabel('class')
    if absolute_value:
        g.set_title('abs(coef), sorted by class 0')
    else:
        g.set_title('coefs, sorted by class 0')


#%
# Sort the weights by their strength, take out bottom N rois, iterate.

plot_weight_matrix(svc, absolute_value=True)
pl.savefig(os.path.join(classifier_dir, 'sorted_weights_abs.png'))
pl.close()

plot_weight_matrix(svc, absolute_value=False)
pl.savefig(os.path.join(classifier_dir, 'sorted_weights_raw.png'))
pl.close()

nrois = len(run_info['roi_list'])
for class_idx in range(len(svc.classes_)):
    plot_coefficients(svc, xrange(nrois), class_idx=class_idx, top_features=20)
    pl.savefig(os.path.join(classifier_dir, 'sorted_feature_weights_%s.png' % class_idx))
    pl.close()

#%%
# Save classifier:
    
clf_fpath = os.path.join(classifier_dir, '%s_datasets.npz' % classif_identifier)
np.savez(clf_fpath, cX=cX, cX_std=cX_std, cy=cy,
             data_type=data_type,
             sconfigs=sconfigs, run_info=run_info)

from sklearn.externals import joblib
joblib.dump(classifier_dir, '%s.pkl' % classif_identifier, compress=9)
 
import sys
clf_params = svc.get_params().copy()
if 'base_estimator' in clf_params.keys():
    clf_params['base_estimator'] = str(clf_params['base_estimator'] )
#clf_params['cv'] = str(clf_params['cv'])
clf_params['identifier'] = classif_identifier
clf_params_hash = hash(json.dumps(clf_params, sort_keys=True, ensure_ascii=True)) % ((sys.maxsize + 1) * 2) #[0:6]

with open(os.path.join(classifier_dir, 'params_%s.json' % clf_params_hash), 'w') as f:
    json.dump(clf_params, f, indent=4, sort_keys=True, ensure_ascii=True)

#%%

# Identify top N (most informative) features (neurons) and test classification performance:
    
from sklearn.feature_selection import RFE

nclasses = len(class_labels)
nrois_total = cX_std.shape[-1]

split_half = False
if split_half:
    datasubset = 'halfset'
    # SPlit into halves:
    train_ixs=[]; test_ixs=[]
    for c in np.unique(cy):
        curr_ixs = np.where(cy==c)[0]
        train_ixs.append(curr_ixs[0::2])
        test_ixs.append(curr_ixs[1::2])
        
    train_ixs = np.hstack(train_ixs)
    test_ixs = np.hstack(test_ixs)
    
    train_data = cX_std[train_ixs,:]
    train_labels = cy[train_ixs]
    test_data = cX_std[test_ixs, :]
    test_labels = cy[test_ixs]

else:
    datasubset = 'fullset'
    train_data = cX_std.copy()
    train_labels = cy.copy()
    test_data = cX_std.copy()
    test_labels = cy.copy()


results_topN = []
orig_rids = xrange(0, nrois_total)
print len(orig_rids)

svc = LinearSVC(random_state=0, dual=dual, multi_class='ovr', C=big_C)
kfold = StratifiedKFold(n_splits=5, shuffle=True)

# First, get accuracy with all features:
rfe = RFE(svc, n_features_to_select=nrois_total)
rfe.fit(train_data, train_labels)
cv_results = cross_val_score(rfe, test_data, test_labels, cv=kfold, scoring=scoring)
results_topN.append(cv_results)

kept_rids_by_iter = []
nfeatures_to_keep = np.arange(1, nrois_total, 1)
for nkeep in nfeatures_to_keep[::-1]:
    rfe = RFE(svc, n_features_to_select=nkeep)
    #rfe.fit(cX_std, cy)
    new_y = rfe.fit_transform(train_data, train_labels)

    removed_rids = np.where(rfe.ranking_!=1)[0]
    kept_rids = np.array([i for i in orig_rids if i not in removed_rids])
    print "Keeping %i.... (%i)" % (nkeep, len(kept_rids))
    kept_rids_by_iter.append(kept_rids)
    cv_results = cross_val_score(rfe, test_data[:, kept_rids], test_labels, cv=kfold, scoring=scoring)
    results_topN.append(cv_results)
    


# Plot number of features VS. cross-validation scores
fig, ax = pl.subplots(1) #figure()
pl.xlabel("Number of features selected")
pl.ylabel("Cross validation score (nb of correct classifications)")
#pl.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
means_by_feat = np.mean(results_topN, axis=1)[::-1]
sems_by_feat = stats.sem(results_topN, axis=1)[::-1]

ax.plot(xrange(len(results_topN)), means_by_feat, linewidth=1, color='k')
ax.fill_between(xrange(len(results_topN)), means_by_feat-sems_by_feat, means_by_feat+sems_by_feat, color='k', alpha=0.2)

pl.title("Ranked features (recursive elim.)")
ax.axhline(y=1./nclasses, linestyle='--', linewidth=1, color='k')
# draw dotted line at chance level:

#print("Optimal number of features : %d" % rfecv.n_features_)

pl.show()

pl.savefig(os.path.join(classifier_dir, 'RFE_fittransform_%s_%s.png' % (scoring, datasubset)))

# Save info:
rfe_cv_info = {'kfold': kfold,
          'rfe': rfe,
          'results': results_topN,
          'kept_rids_by_iter': kept_rids_by_iter}

with open(os.path.join(classifier_dir, 'RFE_cv_output_%s_%s.pkl' %  (scoring, datasubset)), 'wb') as f:
    pkl.dump(rfe_cv_info, f, protocol=pkl.HIGHEST_PROTOCOL)
    

#%%
    
# #############################################################################
# Visualize the data
# #############################################################################

    
# Look at a few neurons to see what the data looks like for binary classif:
ddf = pd.DataFrame(data=cX_std,
                      columns=range(nrois))
labels_df = pd.DataFrame(data=cy,
                         columns=['class'])

df = pd.concat([ddf, labels_df], axis=1).reset_index(drop=True)

_ = sns.pairplot(df[:50], vars=[8, 11, 12, 14, 19], hue="class", size=1.5)



# How correlated are the features (neurons) to class labels?  
# -----------------------------------------------------------------------------

corr = ddf.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = pl.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})



#%%

  
#%%

# #############################################################################
# Learning curves
# #############################################################################


# Modified from http://scikit-learn.org/stable/auto_examples/plot_learning_curve.html
from sklearn.learning_curve import learning_curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects
    """
    
    pl.figure()
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=1, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    pl.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    pl.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    pl.plot(train_sizes, train_scores_mean, 'o-', color="r", alpha=0.8,
             label="Training score")
    pl.plot(train_sizes, test_scores_mean, 'o-', color="g", alpha=0.8,
             label="Cross-validation score")

    pl.xlabel("Training examples")
    pl.ylabel("Score")
    pl.legend(loc="best")
    pl.grid("on") 
    if ylim:
        pl.ylim(ylim)
    pl.title(title)
    
# 
rand_order = random.sample(xrange(len(cy)), len(cy))


# 1.  Look at Training score vs. CV score.
# -----------------------------------------------------------------------------
# Training score is always at max, regardless of N training examples 
# --> OVERFITTING.
# CV score increases over time?  
# Gap between CV and training scores -- big gap ~ high variance!
# -----------------------------------------------------------------------------
plot_learning_curve(LinearSVC(C=.10), "LinearSVC(C=10.0)",
                    cX_std, cy, ylim=(0.0, 1.01),
                    cv = 5)


# 2.  Reduce the complexity -- fewer features.
# -----------------------------------------------------------------------------
# SelectKBest(f_classif, k=2) will select the k=2 best features according to their Anova F-value
 
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif

nfeatures = 10
C_val = 1E9
plot_learning_curve(Pipeline([("fs", SelectKBest(f_classif, k=nfeatures)),
                               ("svc", svc)]),
                    "SelectKBest(f_classif, k=%i) + LinearSVC(C=%.2f)" % (nfeatures, C_val),
                    cX_std, cy, ylim=(0.0, 1.0))

    
# 3.  Use better regularization term. 
# -----------------------------------------------------------------------------
# Increase classifier regularization (smaller C).
# Or, select C automatically w/ grid-search.

from sklearn.grid_search import GridSearchCV
est = GridSearchCV(LinearSVC(), 
                   param_grid={"C": [0.001, 0.01, 0.1, 1.0, 10.0]})
plot_learning_curve(est, "LinearSVC(C=AUTO)", 
                    cX_std, cy, ylim=(0.0, 1.0))
print "Chosen parameter on 100 datapoints: %s" % est.fit(cX_std[:100], cy[:100]).best_params_


# 4.  Try L1 penalty for a sparser solution.
# -----------------------------------------------------------------------------
# With L1 penalty, implicit feature selection. Can look at learned coefficients
# to see how many are 0 (ignored feature), and which have the stongest weights.

plot_learning_curve(LinearSVC(C=0.1, penalty='l1', dual=False), 
                    "LinearSVC(C=0.1, penalty='l1')", 
                    cX_std, cy, ylim=(0.0, 1.0))

est = LinearSVC(C=0.1, penalty='l1', dual=False)
est.fit(cX_std[:100], cy[:100])  # fit on 100 datapoints
print "Coefficients learned: %s" % est.coef_
print "Non-zero coefficients: %s" % np.nonzero(est.coef_)[1]

strongest_weights = np.where(np.abs(est.coef_) == np.abs(est.coef_).max())
print "Best features: %s" % str(strongest_weights)

pl.figure()
sns.heatmap(est.coef_)

#%%
# Plot decision function with all data:

svc.classes_
decisionfunc = svc.decision_function(cX_std) #print cdata.shape

pl.figure()
g = sns.heatmap(decisionfunc, cmap='PRGn')

indices = { value : [ i for i, v in enumerate(cnames) if v == value ] for value in list(set(cnames)) }
ytick_indices = [(k, indices[k][-1]) for k in list(set(cnames))]
g.set(yticks = [])
g.set(yticks = [x[1] for x in ytick_indices])
g.set(yticklabels = [x[0] for x in ytick_indices])
g.set_yticklabels(g.get_yticklabels(), rotation=45, fontsize=8)
g.set(xlabel = 'class')
g.set(ylabel = 'trials by label')

pl.title('%s (decision functions)' % data_type)

figname = '%s__decisionfunction.png' % classif_identifier

pl.savefig(os.path.join(classifier_dir, figname))


#%%
    
# Project zscores onto normals:
svc.classes_
cdata = np.array([svc.coef_[c].dot(cX_std.T) + svc.intercept_[c] for c in range(len(svc.classes_))])
print cdata.shape

pl.figure()
g = sns.heatmap(cdata, cmap='PRGn')

indices = { value : [ i for i, v in enumerate(cnames) if v == value ] for value in list(set(cnames)) }
xtick_indices = [(k, indices[k][-1]) for k in list(set(cnames))]
g.set(xticks = [])
g.set(xticks = [x[1] for x in xtick_indices])
g.set(xticklabels = [x[0] for x in xtick_indices])
g.set_xticklabels(g.get_xticklabels(), rotation=45, fontsize=8)
g.set(xlabel = cell_unit)
g.set(ylabel = 'class')

pl.title('%s data proj onto normals' % data_type)

figname = '%s__proj_normals.png' % classif_identifier

pl.savefig(os.path.join(classifier_dir, figname))


