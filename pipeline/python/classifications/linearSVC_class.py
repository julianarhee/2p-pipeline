#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 13:59:19 2018

@author: juliana
"""


import h5py
import copy
import hashlib
import os
import json
import cv2
import time
import math
import sys
import random
import itertools
import scipy.io
import optparse
import sys
import glob
import itertools
import datetime
import pprint
pp = pprint.PrettyPrinter(indent=4)

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
import scikit_posthocs as sp
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison
from ast import literal_eval

from pipeline.python.utils import natural_keys, replace_root, print_elapsed_time, label_figure
import pipeline.python.traces.combine_runs as cb
import pipeline.python.paradigm.align_acquisition_events as acq
import pipeline.python.visualization.plot_psths_from_dataframe as vis
from pipeline.python.traces.utils import load_TID
from pipeline.python.segmentation.segmentation import Segmentations
from pipeline.python.paradigm import utils as fmt
from pipeline.python.classifications import utils as util
from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections import Counter

from sklearn.manifold import MDS
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

from sklearn.feature_selection import RFE, SelectKBest, chi2, f_classif, RFECV
from sklearn.svm import SVR #LinearSVR
from sklearn.externals import joblib

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


def plot_transform_grid(performance_grid, rowvals=[], colvals=[], 
                                          ylabel='rows', xlabel='columns', 
                                          cmap='hot', vmin=0.5, vmax=1.0,
                                          data_identifier='', ax=None, title=None):
    return_fig = False
    if ax is None:
        fig, ax = pl.subplots(1, figsize=(15,8))
        return_fig = True
    if len(rowvals) != performance_grid.shape[0]:
        print "Not enough ROW labels:", rowvals
        rowvals = np.arange(performance_grid.shape[0])
    if len(colvals) != performance_grid.shape[1]:
        print "Not enough COL labels:", colvals
        colvals = np.arange(performance_grid.shape[1])
    
    
    im = ax.imshow(performance_grid, cmap=cmap, vmin=vmin, vmax=vmax)
    if title is not None:
        ax.set_title(title)
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
            textcolor='k' if performance_grid[i, j] >= vmax*0.75 else 'w'
            text = ax.text(j, i, '%.2f' % performance_grid[i, j],
                       ha="center", va="center", color=textcolor)          
#            stimkey = [k for k, v in config_grid.items() if v[0]==i and v[1]==j][0]
#            text = ax.text(j, i-0.2, '(%i, %i)' % (stimkey[0], stimkey[1]),
#                           ha='center', va='top', color=textcolor)
    
    if return_fig:
        return fig
    
        
#%%

def intersection(lst1, lst2): 
    return list(set(lst1) & set(lst2)) 


# =============================================================================
# Setting classifier params
# =============================================================================
#%
def get_classifier_params(**cparams):
    '''
    Specified ways to group and label classifier data.
    
    Params:
    =======
    'data_type': (str)
        - 'frames' :  Classify frames (or binned frames)
        - 'stat'   :  Classify some statistic taken across frames/trials (e.g., mean, zscore, etc.)
        
    'inputdata': (str)
        - 'meanstim'    :  Use mean of the stimulus period (raw/corrected values)
        - 'meanstimdff' :  Use df/f of stimulus period
    
    'inputdata_type':  (str)
        - only relevant for cnmf datasets
        - if we use a stat-based dataset, need to specify which raw input type to use (e.g., spikes, corrected, dff, etc.)
        
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

    'subset_nsamples': int or None
        - number of samples to combine together if averaging subset halves (aggregate_type='half')
        - use this param to draw specified number of repetitions to average together (default is half)

    'const_trans':  (str) 
        - Transform type to keep constant (i.e., only label instances in which const_trans = trans_value)
    
    'trans_value':  (int, float)
        - Value to set 'const_trans' to.
    '''
    
    clfparams = dict((k,v) for k,v in cparams.items())
    
#     Make sure aggregate type matches:
#    if clfparams['const_trans'] is not '' and clfparams['trans_value'] is not '':
#        optsE.aggregate_type = 'single' # TODO:  fix this so that tuples of const-trans are allowed?
    #clfparams['binsize'] = clfparams['binsize'] if clfparams['data_type'] =='frames' else ''
    
    
    return clfparams


def get_default_gratings_params():
    
    clfparams = get_classifier_params(data_type='stat', 
                                      inputdata='meanstim', 
                                      inputdata_type='',
                                      roi_selector='visual', 
                                      class_name='ori', 
                                      aggregate_type='all',
                                      subset=None, subset_nsamples=None,
                                      const_trans='', trans_value='',
                                      cv_method='kfold', 
                                      cv_nfolds=5,
                                      cv_ngroups=1,
                                      classifier='LinearSVC', 
                                      binsize=None,
                                      get_null=False,
                                      C=1e9)
    return clfparams
    

def get_best_C(svc_in, cX, cy, output_dir=None, figname=None):
    # Look at cross-validation scores as a function of parameter C
    svc = copy.copy(svc_in)
    C_s = np.logspace(-10, 10, 50)
    scores = list()
    scores_std = list()
    for C in C_s:
        svc.C = C
        this_scores = cross_val_score(svc, cX, cy, n_jobs=1)
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
        if figname is None:
            figname = 'crossval_scores_by_C.png'
        pl.savefig(os.path.join(output_dir, figname))
        pl.close()

    return best_C


def hist_cv_permutations(svc, cX, cy, clfparams, scoring='accuracy', 
                        n_permutations=500, n_jobs=4):
    
    kfold = StratifiedKFold(n_splits=clfparams['cv_nfolds'], shuffle=True)
    
    cv_results = cross_val_score(svc, cX, cy, cv=kfold, scoring=scoring)
    print "CV RESULTS: %f (%f)" % (cv_results.mean(), cv_results.std())
    
    score, permutation_scores, pvalue = permutation_test_score(
        svc, cX, cy, scoring=scoring, cv=kfold, 
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
        
    pl.title("(C: %.2f')" % svc.C) # = 'cv_permutation_test_%s.png' % Cstring
    
    return fig


#%%
    

def get_roi_list(run_info, roi_selector='visual', metric='meanstimdf'):
        
    trans_types = run_info['trans_types'].keys() #run_info['transforms'].keys()
    
    # Load sorted ROI info:
    # -----------------------------------------------------------------------------
#    datakey = run_info['datakey']
#    if '/' in datakey:
#        datakey = datakey[1:]
#    sort_dir = os.path.join(traceid_dir, 'sorted_%s' % datakey)

    traceid_dir = run_info['traceid_dir']
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

# =============================================================================
# Generic, non-classifier plotting functions (plots for training data source)
# =============================================================================

def correlation_matrix(clfparams, class_labels, cX, cy, data_identifier='', output_dir='/tmp'):
    
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
    fig = pl.figure()
    sns.heatmap(1-corrs, cmap=zdf_cmap, vmax=2.0)
    #sns.heatmap(1-corrs, cmap=zdf_cmap, vmin=0.7, vmax=1.3)
    figbase = 'RDM_%srois_%s_classes_%i%s_%s' % (clfparams['roi_selector'], clfparams['data_type'], len(class_labels), 
                                                     clfparams['class_name'], clfparams['aggregate_type'])
        
    if clfparams['aggregate_type'] == 'single':
        pl.title('RDM (%s) - %s: %s (%s %i)' % (clfparams['data_type'], 
                                                 clfparams['class_name'], clfparams['aggregate_type'],
                                                 clfparams['const_trans'], clfparams['trans_value']))
        figbase = '%s_%s%i' % (figbase, clfparams['const_trans'], clfparams['trans_value'])
    else:
        pl.title('RDM (%s) - %s: %s' % (clfparams['data_type'], clfparams['class_name'], clfparams['aggregate_type']))

    label_figure(fig, data_identifier)
    
    pl.savefig(os.path.join(output_dir, '%s.png' % figbase))
    pl.close()
    
    #pl.figure()
    #sns.heatmap(corrs, cmap='PRGn', vmin=-1, vmax=1)
    
    # Look at RDM within trial:
    # -----------------------------------------------------------------------------
    nframes_per_trial = list(set([v for k,v in Counter(cy).items()]))[0] #run_info['nframes_per_trial']
    df_list = []
    cnames = []
    for label in class_labels:
        tidxs = np.where(cy==label)[0]
        currtrials = np.squeeze(cX[tidxs,:])
        if clfparams['data_type'] == 'xcondsub':
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
    fig = pl.figure()
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
    
    pl.title('RDM (%s, %s)' % (clfparams['data_type'], cell_unit))
    
    figname = '%s_%s' % (figbase, cell_unit)
    label_figure(fig, data_identifier)
    pl.savefig(os.path.join(output_dir, figname))
    pl.close()


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
    fig = pl.figure(figsize=(15, 5))
    colors = ['r' if c < 0 else 'b' for c in coef[top_coefficients]]
    sns.barplot(np.arange(2 * top_features), coef[top_coefficients], palette=colors) #colors)
    sns.despine(offset=4, trim=True)

    feature_names = np.array(feature_names)
    pl.xticks(np.arange(0, 1 + 2 * top_features), feature_names[top_coefficients], rotation=45, ha='right')
    pl.title('Sorted weights (top %i), class %i' % (top_features, class_idx))

    return fig

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

    return fig
#%
    
#def spatially_sort_rois(traceid_dirs):
#    
#    # =========================================================================
#    # Load ROI masks to sort ROIs by spatial distance
#    # =========================================================================
#        
#    for traceid_dir in traceid_dirs: #opts in options_list:
#        #optsE = extract_options(opts)
#        run_dir = traceid_dir.split('/traces/')[0]
#        acquisition_dir = os.path.split(run_dir)[0]
#        acquisition = os.path.split(acquisition_dir)[-1]
#        
#        #traceid_dir = dataset['run_info'][()]['traceid_dir']
#        sorted_rids, cnts, zproj = util.sort_rois_2D(traceid_dir)
#        util.plot_roi_contours(zproj, sorted_rids, cnts, clip_limit=0.005, label=False)
#        
#        figname = 'spatially_sorted_rois_%s.png' % acquisition
#        pl.savefig(os.path.join(acquisition_dir, figname))
#        pl.close()
#        
#    return sorted_rids

##%
#def plot_grand_mean_traces(dset_list, response_type='dff', label_list=[], color_list=['b','m','g'], output_dir='/tmp', save_and_close=True):
#        
#    mval = []
#    fig, ax = pl.subplots(1) #pl.figure()
#    for di,dtrace in enumerate(dset_list):
#        pl.plot(dtrace['mean'], color=color_list[di], label=label_list[di])
#        pl.fill_between(xrange(len(dtrace['mean'])), dtrace['mean']-dtrace['sem'], dtrace['mean']+dtrace['sem'], alpha=0.5, color=color_list[di])
#        pl.plot(dtrace['stimframes'], np.ones(dtrace['stimframes'].shape) * -0.005*(di+1), color=color_list[di])
#        mval.append(dtrace['mean'].max())
#
#    ax.set(xticks = [])
#    ax.set(xticklabels = [])
#    ax.set(yticks = [0,max(mval)])
#    ax.set(ylabel=response_type)
#    sns.despine(offset=4, trim=True, bottom=True)
#    pl.legend()
#    
#    figname_base = 'avgtrace_%s' % (response_type)
#    run_str = '_'.join(['%s_%s' % (dtrace['run'], dtrace['traceid']) for dtrace in dset_list])
#    figname = '%s_%s.pdf' % (figname_base, run_str)
#
#    if save_and_close:
#        pl.savefig(os.path.join(output_dir, figname))
#        pl.close()
#        
#    return figname
#
#def get_grand_mean_trace(d1, response_type='dff'):
#    d = {}
#    assert response_type in d1.keys(), "Specified response type (%s) not found. Choose from: %s" % (response_type, str(d1.keys()))
#    d1_meanstims = d1[response_type]
#    d1_run = os.path.split(d1['run_info'][()]['traceid_dir'].split('/traces')[0])[-1]
#    print d1_run, d1_meanstims.shape
#    
#    # Get run info:
#    nrois = d1_meanstims.shape[-1]
#    assert len(d1['run_info'][()]['nframes_per_trial']) == 1, "More than 1 val for nframes_per_trial! -- %s" % str(d1['run_info'][()]['nframes_per_trial'])
#    assert len(d1['run_info'][()]['nframes_on']) == 1, "More than 1 val for nframes_on! -- %s" % str(d1['run_info'][()]['nframes_on'])
#
#    nframes_per_trial = d1['run_info'][()]['nframes_per_trial'][0]
#    nframes_on = d1['run_info'][()]['nframes_on'][0]
#    d1_nframes = nframes_per_trial
#    d1_tmat = np.reshape(d1_meanstims, (d1_meanstims.shape[0]/d1_nframes, d1_nframes, nrois))
#    meantrace_rois1 = np.mean(d1_tmat, axis=0)
#    mean_baselines = np.mean(meantrace_rois1[0:d1['run_info'][()]['stim_on_frame'], :], axis=0)
#    
#    meantrace_rois1 -= mean_baselines
#    
#    meantrace1 = np.mean(meantrace_rois1, axis=1)
#    semtrace1 = stats.sem(meantrace_rois1, axis=1)
#    d1_stim_frames = np.array([d1['run_info'][()]['stim_on_frame'], int(round(d1['run_info'][()]['stim_on_frame'] + nframes_on))])
#    
#    d['run'] = d1_run
#    d['mean'] = meantrace1
#    d['sem'] = semtrace1
#    d['stimframes'] = d1_stim_frames
#    d['traceid'] = d1['run_info'][()]['traceid_dir'].split('/traces/')[-1].split('/')[-1]
#    return d
#
#
#def compare_grand_mean_traces(data_paths):
#    assert len(data_paths.keys()) > 1, "Only 1 data path specified, nothing to compare..."
#    
#    dset_list = []
#    response_type = 'dff'
#
#    for di, dpath in data_paths.items():
#        d1 = np.load(dpath)
#        # Load data:
#        dtrace = get_grand_mean_trace(d1, response_type=response_type)
#        dset_list.append(dtrace)
#    
#    # PLOT:
#    a_run_dir = data_paths[0].split('/traces')[0]
#    acquisition_dir = os.path.split(a_run_dir)[0]
#    figname = plot_grand_mean_traces(dset_list, response_type=response_type,
#                                         label_list=['%s_%s' % (dtrace['run'], dtrace['traceid']) for dtrace in dset_list], 
#                                         output_dir=acquisition_dir, 
#                                         save_and_close=False)
#    pl.savefig(os.path.join(acquisition_dir, figname))
#    pl.close()
#


#%%

# =============================================================================
# FRAMES data-type classifications:
# =============================================================================
#cX = C.cX; cy = C.cy; class_name = C.clfparams['class_name']; 
#sconfigs = C.data.sconfigs; run_info = C.data.run_info;
#binsize = C.clfparams['binsize']

def split_trial_epochs(cX, cy, class_name, sconfigs, run_info, binsize=10, relabel=False):
    # Reshape (nframes_per_trial*ntrials_total X nrois) into ntrials_total x nframes_per_trial x nrois)
    nframes_per_trial = run_info['nframes_per_trial']
    if isinstance(nframes_per_trial, list):
        assert len(nframes_per_trial) == 1, "More than 1 unique value found for n frames per trial..."
        nframes_per_trial = nframes_per_trial[0]
    ntrials_total = run_info['ntrials_total']
    nrois = cX.shape[-1] # len(run_info['roi_list'])
    print "Reshaping input data into trials (%i) x frames (%i) x rois (%i)" % (ntrials_total, nframes_per_trial, nrois) 
    cX_r = np.reshape(cX, (ntrials_total, nframes_per_trial, nrois))
    cy_r = np.reshape(C.cy, (ntrials_total, nframes_per_trial))
    print cy_r.shape
    
    # Bin frames into smaller chunks so we don't have to train 300 classifiers...
    # nframes_on = int(round(run_info['nframes_on']))
    # stim_on = int(run_info['stim_on_frame'])
    bin_idxs =  np.arange(0, nframes_per_trial, binsize)
    bin_idxs = np.append(bin_idxs, nframes_per_trial+1) # Add the last frame 
 
    # Make binned frame data into shape: ntrial_epochs x ntrials_total x nrois.
    cX_tmp = np.array([np.mean(cX_r[:, bstart:bin_idxs[bi+1], :], axis=1) for bi, bstart in enumerate(bin_idxs[0:-1])])
    
    # There will be cX_tmp.shape[0] classifiers, each with ntrials_total x nrois dataset:
    epochs = dict((epoch, cX_tmp[epoch, :, :]) for epoch in range(cX_tmp.shape[0])) # Each dataset is ntrials x nrois
    if isinstance(cy_r[0, 0], (str, unicode)) and 'config' in cy_r[0, 0]:
        trial_labels = [sconfigs[cv][class_name] for cv in cy_r[:,0]]
    else:
        trial_labels = cy_r[:, 0]
    if relabel:
        labels = dict((epoch, trial_labels) for epoch in range(cX_tmp.shape[0]))           # Each sample in dataset (ntrials) is labeled by stim type
    else:
        labels = dict((epoch, cy_r[:,0]) for epoch in range(cX_tmp.shape[0]))           # Each sample in dataset (ntrials) is labeled by stim type

    return epochs, labels, bin_idxs[0:-1]
    

def get_trial_bins(run_info, binsize=10):
    bins = []
    # Figure out bin size if using FRAMES:
    nframes_per_trial = run_info['nframes_per_trial']
    if isinstance(nframes_per_trial, list):
        assert len(nframes_per_trial) == 1, "More than 1 value found for nframes_per_trial: %s" % str(nframes_per_trial)
        nframes_per_trial = nframes_per_trial[0]
        
    stim_on_frame = run_info['stim_on_frame']
    try:
        bins = np.arange(0, nframes_per_trial, binsize)
        assert stim_on_frame in bins, "Stim on frame %i not in bins: %s" % (stim_on_frame, str(bins))
    except Exception as e:
        binsize += 5
        bins = np.arange(0, nframes_per_trial, binsize)
        assert stim_on_frame in bins, "Stim on frame %i not in bins: %s" % (stim_on_frame, str(bins))
        
    return bins, binsize


def format_epoch_dataset(clfparams, cX, cy, run_info, sconfigs):

    class_name = clfparams['class_name']
    binsize = clfparams['binsize']; const_trans=clfparams['const_trans']
    bins, binsize = get_trial_bins(run_info, binsize=binsize) #run_info['nframes_per_trial'], run_info['stim_on_frame'], binsize=binsize)
    print "Dividing trial into %i epochs (binsize=%i)." % (len(bins), binsize)
    
    # Get values for specified const-trans value:
    const_trans_values = []
    if not const_trans=='':
        const_trans_values = sorted(list(set([sconfigs[c][const_trans] for c in sconfigs.keys()])))
        print "Selected subsets of transform: %s. Found values %s" % (const_trans, str(const_trans_values))
        
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
        if isinstance(labels[0][0], (str, unicode)) and 'config' in labels[0,0]:
            new_labels = np.array([sconfigs[cv][class_name] for cv in labels[0]]) # Each classifier has the same labels, since we are comparing epochs
            labels = dict((k, new_labels) for k in labels.keys())

        decode_dict['all']['epochs'] = epochs
        decode_dict['all']['labels'] = labels
        
    class_labels = sorted(list(set(decode_dict[decode_dict.keys()[0]]['labels'][0])))
    print "Labels:", class_labels

    return epochs, decode_dict, bins, binsize, class_labels


def decode_trial_epochs(class_labels, clfparams, bins, decode_dict, run_info, 
                            data_identifier='', niterations=10, scoring='accuracy',
                            output_dir='/tmp'):
    '''
    Get formatted data and bins/binsize from format_epoch_dataset(), and train
    classifier for each epoch.
    
    Plot classifier scoring method (scoring='accuracy' default) for each epoch.
    '''
    
    for view_key in decode_dict.keys():
        
        epochs = decode_dict[view_key]['epochs']
        labels = decode_dict[view_key]['labels']
        
        # Create pipeline:
        # ------------------
#        if epochs[0].shape[0] > epochs[0].shape[1]: # nsamples > nfeatures
#            dual = False
#        else:
#            dual = True
#        #pipe_svc = Pipeline([('scl', StandardScaler()),
#        #                     ('clf', LinearSVC(random_state=0, dual=dual, multi_class='ovr', C=big_C))]) #))])
#        
        for runiter in np.arange(0,niterations): #range(5):
            # evaluate each model in turn
            print "... %i of %i iterations" % (runiter, niterations)
            results = []
            names = []
            for idx, curr_epoch in enumerate(sorted(epochs.keys())):
                model = LinearSVC(random_state=0, dual=clfparams['dual'], multi_class='ovr', C=clfparams['C_val'])
                kfold = StratifiedKFold(n_splits=clfparams['cv_nfolds'], shuffle=True)
                #x_std = StandardScaler().fit_transform(epochs[curr_epoch])
                x_std = epochs[curr_epoch] # Already do standard scaling before
                cv_results = cross_val_score(model, x_std, labels[curr_epoch], cv=kfold, scoring=scoring)
                results.append(cv_results)
                names.append(curr_epoch)
                msg = "%s: %f (%f)" % (curr_epoch, cv_results.mean(), cv_results.std())
                #print(msg)
    
            fig = plot_decoding_performance_trial_epochs(results, bins, names, 
                                                       scoring=scoring, 
                                                       stim_on_frame=run_info['stim_on_frame'],
                                                       nframes_on=run_info['nframes_on'],
                                                       nclasses=len(class_labels),
                                                       binsize=clfparams['binsize'])
    
            fig.suptitle('Estimator Comparison: decode %s (%s: %s)' % (clfparams['class_name'], clfparams['const_trans'], clfparams['trans_value']))
            figname = 'trial_epoch_classifiers_binsize%i_%s_label_%s_iter%i.png' % (clfparams['binsize'], scoring, clfparams['class_name'], runiter)
            label_figure(fig, data_identifier)
            
            pl.savefig(os.path.join(output_dir, 'figures', figname))
            pl.close()
    
    #return model


def plot_decoding_performance_trial_epochs(results, bins, names, scoring='accuracy',    
                                           stim_on_frame=None, nframes_on=None,
                                           nclasses=None, binsize=10):
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

    pl.ylim([0, 1])
    sns.despine(offset=4, trim=True)
    if stim_on_frame is not None and nframes_on is not None:
        if isinstance(stim_on_frame, list):
            assert len(stim_on_frame)==1, "More than 1 val found for stim_on frame: %s" % str(stim_on_frame)
            stim_on_frame = stim_on_frame[0]
        if isinstance(nframes_on, list):
            assert len(nframes_on)==1, "More than 1 val found for nframes_on: %s" % str(nframes_on)
            nframes_on = nframes_on[0]
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
    #pl.show()
    
    return fig

#%%
            
# =============================================================================
# Cross-Validation Functions
# =============================================================================

def cv_permutation_test(svc, cX_std, cy, clfparams, scoring='accuracy', 
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
    
    pl.savefig(os.path.join(clfparams['classifier_dir'], 'figures', figname))
    print "Saved CV results to: %s" % os.path.join(clfparams['classifier_dir'], 'figures', figname)
    pl.close()
    


def get_cv_folds(svc, clfparams, cX, cy, cy_labels=None, output_dir=None, data_id=None):
    
    cv_method = clfparams['cv_method']
    cv_nfolds = clfparams['cv_nfolds']
    cv_ngroups = clfparams['cv_ngroups']
    
    training_data = cX.copy()
    
    n_samples = cX.shape[0]
    print "N samples for CV:", n_samples
    classes = sorted(np.unique(cy))

    predicted = []
    true = []
    config_names = []
    # Cross-validate for t-series samples:
    if cv_method=='splithalf':
    
        # Fit with subset of data:
        svc.fit(training_data[:n_samples // 2], cy[:n_samples // 2])
    
        # Now predict the value of the class label on the second half:
        y_test = cy[n_samples // 2:]
        y_pred = svc.predict(training_data[n_samples // 2:])
        predicted.append(y_pred) #=y_test])
        true.append(y_test)
        if cy_labels is not None:
            config_names.append(cy_labels[n_samples // 2:])
            
    elif cv_method=='kfold':
        loo = cross_validation.StratifiedKFold(cy, n_folds=cv_nfolds, shuffle=True)

        for train, test in loo: #, groups=groups):
            #print train, test
            X_train, X_test = training_data[train], training_data[test]
            y_train, y_test = cy[train], cy[test]
            y_pred = svc.fit(X_train, y_train).predict(X_test)
            predicted.append(y_pred) #=y_test])
            true.append(y_test)
            if cy_labels is not None:
                config_names.append(cy_labels[test])
    
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
            loo = LeavePGroupsOut(cv_ngroups)
    
        for train, test in loo.split(training_data, cy, groups=groups):
            #print train, test
            X_train, X_test = training_data[train], training_data[test]
            y_train, y_test = cy[train], cy[test]
    
            y_pred = svc.fit(X_train, y_train).predict(X_test)
    
            predicted.append(y_pred) #=y_test])
            true.append(y_test)
            if cy_labels is not None:
                config_names.append(cy_labels[test])
    
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
            if cy_labels is not None:
                config_names.append(cy_labels[test])
    
    
    if output_dir is not None:
        # Save CV info:
        # -----------------------------------------------------------------------------
        if data_id is None:
            data_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

        f = open(os.path.join(output_dir, 'CV_report_%s.txt' % data_id), 'w')
        for y_true, y_pred in zip(true, predicted):
            f.write(metrics.classification_report(y_true, y_pred, target_names=[str(c) for c in classes]))
        f.close()
        
        cv_results = {'predicted': [list(p) for p in predicted], #.tolist(), #list(y_pred),
                      'true': [list(p) for p in true], # list(y_test),
                      'classifier': clfparams['classifier'],
                      'cv_method': clfparams['cv_method'],
                      'ngroups': clfparams['cv_ngroups'],
                      'nfolds': clfparams['cv_nfolds'],
                      'classes': classes,
                      'cy_labels': list(cy_labels),
                      'config_names': [list(p) for p in config_names]
                      }
        with open(os.path.join(output_dir, 'CV_results_%s.json' % data_id), 'w') as f:
            json.dump(cv_results, f, sort_keys=True, indent=4)
        
        print "Saved CV results: %s" % output_dir

    return predicted, true, classes, config_names



def get_confusion_matrix(predicted, true, classes, average_iters=True):
    # Compute confusion matrix:
    # -----------------------------------------------------------------------------
    if average_iters:
        if not isinstance(predicted[0], (int, float, str)) and len(predicted[0]) > 1:
            cmatrix = confusion_matrix(true[0], predicted[0], labels=classes)
            for iter_idx in range(len(predicted))[1:]:
                print "adding iter %i" % iter_idx
                cmatrix += confusion_matrix(true[iter_idx], predicted[iter_idx], labels=classes)
            conf_mat_str = 'AVG'
        #cmatrix_tframes /= float(len(pred_results))
        else:
            cmatrix = confusion_matrix(true, predicted, labels=classes)
            conf_mat_str = 'single_test'
    else:
        avg_scores = []
        for y_pred, y_test in zip(predicted, true):
            pred_score = len([i for i in y_pred==y_test if i]) / float(len(y_pred))
            avg_scores.append(pred_score)
        best_fold = avg_scores.index(np.max(avg_scores))

        cmatrix = confusion_matrix(true[best_fold], predicted[best_fold], labels=classes)
        conf_mat_str = 'best'
                
    return cmatrix, conf_mat_str



def plot_confusion_matrix_subplots(predicted, true, classes, cv_method='kfold', 
                                   data_identifier='', output_dir=None, figname=None): #calculate_confusion_matrix(predicted, true, clfparams, data_identifier=''):

    cmatrix, conf_mat_str = get_confusion_matrix(predicted, true, classes)

    #% Plot confusion matrix:
    # -----------------------------------------------------------------------------
    sns.set_style('white')
    fig = pl.figure(figsize=(20,8))
    ax1 = fig.add_subplot(1,2,1)
    plot_confusion_matrix(cmatrix, classes=classes, ax=ax1, normalize=False,
                      title='Confusion matrix (%s, %s)' % (conf_mat_str, cv_method))
    
    ax2 = fig.add_subplot(1,2,2)
    plot_confusion_matrix(cmatrix, classes=classes, ax=ax2, normalize=True,
                          title='Normalized')
    
    #%
#    if '/classifiers' in output_dir:
#        classif_identifier = os.path.split(output_dir.split('/classifiers')[-1])[-1]
#        figname = '%s__confusion_%s_iters.png' % (classif_identifier, conf_mat_str)
#    else:
    if figname is None:
        figname = 'confusion_matrix_%s_iters.png' % conf_mat_str
    label_figure(fig, data_identifier)
    
    print "Confusion figure:", figname
    if output_dir is not None:
        pl.savefig(os.path.join(output_dir, figname))
        pl.close()
    
    return cmatrix

def plot_normed_confusion_matrix(predicted, true, classes, normalize=True, cv_method='kfold', ax=None):
    
    # Compute confusion matrix:
    # -----------------------------------------------------------------------------
    cmatrix_tframes, conf_mat_str = get_confusion_matrix(predicted, true, classes)
        
    #% Plot confusion matrix:
    # -----------------------------------------------------------------------------
    #sns.set_style('white')
    if ax is None:
        fig, ax = pl.subplots(figsize=(10,4))

    plot_confusion_matrix(cmatrix_tframes, classes=classes, ax=ax, normalize=normalize,
                          title='Normalized confusion (%s, %s)' % (conf_mat_str, cv_method))
    
    return
    
#%
#def plot_confusion_matrix(cm, classes,
#                          ax=None,
#                          normalize=False,
#                          title='Confusion matrix',
#                          cmap=pl.cm.Blues):
#    """
#    This function prints and plots the confusion matrix.
#    Normalization can be applied by setting `normalize=True`.
#    """
#    if normalize:
#        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#        print("Normalized confusion matrix")
#    else:
#        print('Confusion matrix, without normalization')
#
#    print(cm)
#
#    #fig = pl.figure(figsize=(4,4))
#    if ax is None:
#        fig = pl.figure(figsize=(4,4))
#        ax = fig.add_subplot(111)
#
#    ax.set_title(title, fontsize=10)
#
#    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
#    tick_marks = np.arange(len(classes))
#    ax.set_xticks(tick_marks)
#    ax.set_xticklabels(classes, rotation=45, fontsize=10)
#    ax.set_yticks(tick_marks)
#    ax.set_yticklabels(classes, fontsize=10)
#    fmt = '.2f' if normalize else 'd'
#    thresh = cm.max() / 2.
#    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#        ax.text(j, i, format(cm[i, j], fmt),
#                 horizontalalignment="center",
#                 color="white" if cm[i, j] > thresh else "black")
#
#    #pl.tight_layout()
#    ax.set_ylabel('True label')
#    ax.set_xlabel('Predicted label')
#
#    divider = make_axes_locatable(ax)
#    cax = divider.append_axes("right", size="5%", pad=0.05)
#    pl.colorbar(im, cax=cax)


def plot_confusion_matrix(cmatrix, classes,
                          ax=None,
                          normalize=False,
                          title='Confusion matrix', clim=None,
                          cmap=pl.cm.Blues, cmin=0, cmax=1.0):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cmatrix = cmatrix.astype('float') / cmatrix.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
        if clim=='max':
            cmax = cmatrix.max()
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
    ax.set_xticklabels(classes, rotation=45, fontsize=8)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=8)
    fmt = '.1f' if normalize else 'd'
    thresh = cmatrix.max() / 2.
    for i, j in itertools.product(range(cmatrix.shape[0]), range(cmatrix.shape[1])):
        ax.text(j, i, format(cmatrix[i, j], fmt),
                 horizontalalignment="center", fontsize=6,
                 color="white" if cmatrix[i, j] > thresh else "black")

    #pl.tight_layout()
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    pl.colorbar(im, cax=cax)

    return ax


#%%
    

def iterate_RFE(clfparams, cX_std, cy, scoring='accuracy', 
                    output_dir='/tmp', datasubset='full'):
    
    nrois_total = cX_std.shape[-1]
    
    if datasubset == 'half':
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
    
    elif datasubset == 'full':
        train_data = cX_std.copy()
        train_labels = cy.copy()
        test_data = cX_std.copy()
        test_labels = cy.copy()
    
    
    results_topN = []
    orig_rids = xrange(0, nrois_total)
    print len(orig_rids)
    
    svc = LinearSVC(random_state=0, dual=clfparams['dual'], multi_class='ovr', C=clfparams['C_val'])
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

    # Save info:
    rfe_cv_info = {'kfold': kfold,
              'rfe': rfe,
              'results': results_topN,
              'kept_rids_by_iter': kept_rids_by_iter}
    
    with open(os.path.join(output_dir, 'results', 'RFE_cv_output_%s_%s.pkl' %  (scoring, datasubset)), 'wb') as f:
        pkl.dump(rfe_cv_info, f, protocol=pkl.HIGHEST_PROTOCOL)
                
    return results_topN


        
def plot_RFE_results(results_topN, nclasses, scoring='accuracy', 
                         output_dir='/tmp', datasubset='full', data_identifier=''):
    
    #nclasses = len(clfparams['class_labels'])

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
    label_figure(fig, data_identifier)
    
    pl.savefig(os.path.join(output_dir, 'figures', 'RFE_fittransform_%s_%s.png' % (scoring, datasubset)))
    pl.close()
    
    

    #%%

def get_stat_samples(Xdata, labels_df, clfparams, multiple_durs=True):
    print "Trials are sorted by time of occurrence, not stimulus type."

    ntrials_total = len(labels_df['trial'].unique())
    ntrials_per_cond = [len(t) for t in labels_df.groupby('config')['trial'].unique()]
    #assert len(np.unique(ntrials_per_cond)) == 1, "Uneven reps per condition! %s" % str(ntrials_per_cond)
    ntrials = np.unique(ntrials_per_cond)[0]
    
    # Get baseline and stimulus indices for each trial:
    sample_labels = []
    stim_on_frame = labels_df['stim_on_frame'].unique()
    assert len(stim_on_frame) == 1, "More than 1 stim on frame found! %s" % str(stim_on_frame)
    stim_on_frame = stim_on_frame[0]
    if multiple_durs:
        tgroups = labels_df.groupby('trial')            
        #stim_durs = sorted(labels_df['stim_dur'].unique()) # longer stim durs will match with longer nframes on
        #nframes_on = sorted(labels_df['nframes_on'].unique())
        std_baseline_values=[]; mean_baseline_values=[]; mean_stimulus_values=[];
        for k,g in tgroups:
            curr_nframes_on = g['nframes_on'].unique()[0]

            curr_baseline_stds = np.nanstd(Xdata[g['tsec'][0:stim_on_frame].index.tolist(), :], axis=0)
            curr_baseline_means = np.nanmean(Xdata[g['tsec'][0:stim_on_frame].index.tolist(), :], axis=0)

            curr_stimulus_means = np.nanmean(Xdata[g['tsec'][stim_on_frame:stim_on_frame+curr_nframes_on].index.tolist(), :], axis=0)
            
            std_baseline_values.append(curr_baseline_stds)
            mean_baseline_values.append(curr_baseline_means)
            mean_stimulus_values.append(curr_stimulus_means)
            
            curr_config = g['config'].unique()[0]
            sample_labels.append(curr_config)
        
        mean_stim_on_values = np.vstack(mean_stimulus_values)
        mean_baseline_values = np.vstack(mean_baseline_values)
        std_baseline_values = np.vstack(std_baseline_values)
    else:
        nrois = Xdata.shape[-1]
        nframes_per_trial = Xdata.shape[0] / ntrials_total
        nframes_on = labels_df['nframes_on'].unique()[0]
        
        traces = np.reshape(Xdata, (ntrials_total, nframes_per_trial, nrois), order='C')
        std_baseline_values = np.nanstd(traces[:, 0:stim_on_frame], axis=1)
        mean_baseline_values = np.nanmean(traces[:, 0:stim_on_frame], axis=1)
        mean_stim_on_values = np.nanmean(traces[:, stim_on_frame:stim_on_frame+nframes_on], axis=1)
            
    if clfparams['stat_type'] == 'zscore':
        sample_array = (mean_stim_on_values - mean_baseline_values ) / std_baseline_values
    elif clfparams['stat_type'] == 'meanstimdff':
        sample_array = (mean_stim_on_values - mean_baseline_values ) / mean_baseline_values
    else:
        sample_array = mean_stim_on_values
    
    if clfparams['get_null']:
        random_draw = True
        print "Stim values:", sample_array.shape
        if random_draw:
            selected_trial_ixs = random.sample(range(0, mean_baseline_values.shape[0]), ntrials)
        bas = mean_baseline_values[selected_trial_ixs, :]
    
        sample_array = np.append(sample_array, bas, axis=0)
        print "Added null cases:", sample_array.shape
        sample_labels.extend(['bas' for _ in range(bas.shape[0])])
    
    return sample_array, np.array(sample_labels)

#%%
    
def get_frame_samples(Xdata, labels_df, clfparams):
    print "Trials are sorted by time of occurrence, not stimulus type."

    ntrials_total = len(labels_df['trial'].unique())
    ntrials_per_cond = [len(t) for t in labels_df.groupby('config')['trial'].unique()]
    assert len(np.unique(ntrials_per_cond)) == 1, "Uneven reps per condition! %s" % str(ntrials_per_cond)
    ntrials = np.unique(ntrials_per_cond)[0]
    
    
    # Get baseline and stimulus indices for each trial:
    sample_labels = []; sample_array = []; sample_bas = []
    stim_on_frame = labels_df['stim_on_frame'].unique()
    assert len(stim_on_frame) == 1, "More than 1 stim on frame found! %s" % str(stim_on_frame)
    stim_on_frame = stim_on_frame[0]
    
    tgroups = labels_df.groupby('trial')            
    #stim_durs = sorted(labels_df['stim_dur'].unique()) # longer stim durs will match with longer nframes on
    #nframes_on = sorted(labels_df['nframes_on'].unique())

    for k,g in tgroups:
        if clfparams['stat_type'] == 'trial':
            sample_ixs = g.index.tolist()
            curr_values = Xdata[sample_ixs,:]
        else:
            curr_nframes_on = g['nframes_on'].unique()[0]
            if clfparams['stat_type'] == 'stimulus':
                sample_ixs = g['tsec'][stim_on_frame:stim_on_frame+curr_nframes_on].index.tolist()
                curr_values = Xdata[sample_ixs, :]
            elif clfparams['stat_type'] == 'post':
                sample_ixs = g['tsec'][stim_on_frame+curr_nframes_on:].index.tolist()
                curr_values = Xdata[sample_ixs, :]
                
            if clfparams['get_null']:
                sample_ixs = g['tsec'][0:stim_on_frame].index.tolist()
                curr_bas = Xdata[sample_ixs, :]
                sample_bas.append(curr_bas)
                
        sample_labels.extend(g.loc[sample_ixs, 'config'].values.tolist())
        sample_array.append(curr_values)
    
    sample_array = np.vstack(sample_array)

    if clfparams['get_null']: # Can only do this if STAT_TYPE is 'stimulus' or 'post'
        random_draw = True
        print "Stim values:", sample_array.shape
        if random_draw:
            # Randomly select NTRIALS to use for baseline samples:
            selected_trials = random.sample(range(0, ntrials_total), ntrials)
        
        selected_bas_vals = np.vstack(sample_bas[selected_trials])
        
        sample_array = np.append(sample_array, selected_bas_vals, axis=0)
        print "Added null cases:", sample_array.shape
        sample_labels.extend(['bas' for _ in range(len(selected_bas_vals))])
    
    return sample_array, np.array(sample_labels)
     
    
#%%
#def extract_options(options):
#
#    def comma_sep_list(option, opt, value, parser):
#        setattr(parser.values, option.dest, value.split(','))
#
#
#    parser = optparse.OptionParser()
#
#    parser.add_option('-D', '--root', action='store', dest='rootdir',
#                          default='/nas/volume1/2photon/data',
#                          help='data root dir (dir w/ all animalids) [default: /nas/volume1/2photon/data, /n/coxfs01/2pdata if --slurm]')
#    parser.add_option('-i', '--animalid', action='store', dest='animalid',
#                          default='', help='Animal ID')
#
#    # Set specific session/run for current animal:
#    parser.add_option('-S', '--session', action='store', dest='session',
#                          default='', help='session dir (format: YYYMMDD_ANIMALID')
#    parser.add_option('-A', '--acq', action='store', dest='acquisition',
#                          default='FOV1', help="acquisition folder (ex: 'FOV1_zoom3x') [default: FOV1]")
##    parser.add_option('-T', '--trace-type', action='store', dest='trace_type',
##                          default='raw', help="trace type [default: 'raw']")
##    parser.add_option('-R', '--run', dest='run_list', default=[], nargs=1,
##                          action='append',
##                          help="run ID in order of runs")
##    parser.add_option('-t', '--traceid', dest='traceid_list', default=[], nargs=1,
##                          action='append',
##                          help="trace ID in order of runs")
##    parser.add_option('-n', '--nruns', action='store', dest='nruns', default=1, help="Number of consecutive runs if combined")
#    parser.add_option('-R', '--run', action='store', dest='run',
#                          default='', help="RUN name (e.g., gratings_run1)")
#    parser.add_option('-t', '--traceid', action='store', dest='traceid',
#                          default='', help="traceid name (e.g., traces001)")
#    
#    parser.add_option('--slurm', action='store_true', dest='slurm', default=False, help="set if running as SLURM job on Odyssey")
#    parser.add_option('--par', action='store_true', dest='multiproc', default=False, help="set if want to run MP on roi stats, when possible")
#    parser.add_option('--nproc', action='store', dest='nprocesses', default=4, help="N processes if running in par (default=4)")
#
#    # Classifier info:
#    parser.add_option('-r', '--rois', action='store', dest='roi_selector', default='all', help="(options: all, visual)")
#    parser.add_option('-d', '--dtype', action='store', dest='data_type', default='stat', help="(options: frames, stat)")
#    stat_choices = {'stat': ['meanstim', 'meanstimdff', 'zscore'],
#                    'frames': ['trial', 'stimulus', 'post']}
#    parser.add_option('-s', '--stype', action='store', dest='stat_type', default='meanstim', 
#                      help="If dtype is STAT, options: %s. If dtype is FRAMES, options: %s" % (str(stat_choices['stat']), str(stat_choices['frames'])))
#
#    parser.add_option('-p', '--indata_type', action='store', dest='inputdata_type', default='corrected', help="data processing type (dff, corrected, raw, etc.)")
#    parser.add_option('--null', action='store_true', dest='get_null', default=False, help='Include a null class in addition to stim conditions')
#    parser.add_option('-N', '--name', action='store', dest='class_name', default='', help='Name of transform to classify (e.g., ori, xpos, morphlevel, etc.)')
#    
##    choices_agg = ('all', 'single', 'averagereps', 'collapse')
##    default_agg = 'all'
##    parser.add_option('-z', '--agg', dest='aggregate_type', type="choice", choices=choices_agg, default=default_agg, 
##                      help='Aggregate method. Valid choices: %s. Default %s' % (choices_agg, default_agg)) 
##    
##    choices_subset = (None, 'two_class', 'no_morphing')
##    default_subset = None
##    parser.add_option('--subset', dest='subset', type='choice', choices=choices_subset, default=default_subset,
##                      help='Set if only want to consider subset of data. Valid choices: %s. Default %s' % (choices_subset, default_subset)) 
##    parser.add_option('--subset-samples', action='store', dest='subset_nsamples', default=None, help='N samples to draw if aggregate_type=half, but want to use N other than half')
#    
#    parser.add_option('--subset', action='store', dest='class_subset', default='', help='Subset of class_name types to learn')
#    parser.add_option('-c', '--const', dest='const_trans', default='', type='string', action='callback', 
#                          callback=comma_sep_list, help="Transform name to hold constant if classifying a different transform")
#    parser.add_option('-v', '--tval', dest='trans_value', default='', type='string', action='callback', 
#                          callback=comma_sep_list, help="Value to set const_trans to")
#
#    #parser.add_option('-c', '--const', action='store', dest='const_trans', default='', help='Transform name to hold constant if classifying a different transform')
#    #parser.add_option('-v', '--tval', action='store', dest='trans_value', default='', help='Value to set const_trans to')
#    
#    parser.add_option('-L', '--clf', action='store', dest='classifier', default='LinearSVC', help='Classifier type (default: LinearSVC)')
#    parser.add_option('-k', '--cv', action='store', dest='cv_method', default='kfold', help='Method of cross-validation (default: kfold)')
#    parser.add_option('-f', '--folds', action='store', dest='cv_nfolds', default=5, help='N folds for CV (default: 5)')
#    parser.add_option('-C', '--cval', action='store', dest='C_val', default=1e9, help='Value for C param if using SVC (default: 1e9)')
#    parser.add_option('-g', '--groups', action='store', dest='cv_ngroups', default=1, help='N groups for CV, relevant only for data_type=frames (default: 1)')
#    parser.add_option('-b', '--bin', action='store', dest='binsize', default=10, help='Bin size, relevant only for data_type=frames (default: 10)')
##    parser.add_option('--combine', action='store_true', dest='combine_data', default=False, help='Flag to combine multiple runs of the same thing')
##    parser.add_option('--combo', action='store', dest='combo_name', default='combo', help='Name of new, combined dataset (default: combined)')
#
#    (options, args) = parser.parse_args(options)
#    
#    assert options.stat_type in stat_choices[options.data_type], "Invalid STAT selected for data_type %s. Run -h for options." % options.data_type
#
#    return options

                    
            
#%%
class TransformClassifier():
    
    def __init__(self, animalid, session, acquisition, run, traceid, rootdir='/n/coxfs01/2p-data',
                         roi_selector='visual', data_type='stat', stat_type='meanstim',
                         inputdata_type='corrected', 
                         get_null=False, class_name='', class_subset='',
                         const_trans='', trans_value='', test_set=[], indie=False,
                         cv_method='kfold', cv_nfolds=5, cv_ngroups=1, C_val=1e9, binsize=10):
        
        self.rootdir = rootdir
        self.animalid = animalid
        self.session = session
        self.acquisition = acquisition
        self.run = run
        if 'cnmf' in traceid:
            tracedir_type = 'cnmf'
        else:
            tracedir_type = 'traces'
        tmp_traceid_dir = sorted(glob.glob(os.path.join(rootdir, animalid, 
                                              session, acquisition, 
                                              run, tracedir_type, 
                                              '%s*' % traceid)), key=natural_keys)
        if len(tmp_traceid_dir) > 1:
            print "Found multiple trace IDs:"
            for ti, tidir in enumerate(tmp_traceid_dir):
                print ti, tidir
            sel = input("Select IDX of tid dir to use: ")
            traceid_dir = tmp_traceid_dir[sel]
        else:
            assert len(tmp_traceid_dir) == 1, "Not TRACEIDs found!"
            traceid_dir = tmp_traceid_dir[0]
        self.traceid_dir = traceid_dir
        
        self.traceid = os.path.split(self.traceid_dir)[-1]        
        self.data_fpath = self.get_data_fpath()
        self.classifiers = []

        train_params = {'roi_selector': roi_selector,
                        'data_type': data_type,
                        'stat_type': stat_type,
                        'inputdata_type': inputdata_type,
                        'get_null': get_null,
                        'class_name': class_name,
                        'class_subset': class_subset,
                        'const_trans': const_trans,
                        'trans_value': trans_value,
                        'binsize': binsize,
                        'cv_method': cv_method,
                        'cv_nfolds': cv_nfolds,
                        'cv_ngroups': cv_ngroups,
                        'C_val': C_val,
                        'test_set': test_set,
                        'indie': indie}
        self.set_params(train_params) # Set up classifier parameters
        
                
    def get_data_fpath(self):

        # Data array dir:
        data_basedir = os.path.join(self.traceid_dir, 'data_arrays')
        data_fpath = os.path.join(data_basedir, 'datasets.npz')
        assert os.path.exists(data_fpath), "[E]: Data array not found! Did you run tifs_to_data_arrays.py?"
    
        # Create output base dir for classifier results:
        clf_basedir = os.path.join(self.traceid_dir, 'classifiers')
        if not os.path.exists(os.path.join(clf_basedir, 'figures')):
            os.makedirs(os.path.join(clf_basedir, 'figures'))
            
        return data_fpath
            

    def load_dataset(self, visual_area_info=None):
        print "------------ Loading dataset."

        # Store DATASET:            
        dt = np.load(self.data_fpath)
        if 'arr_0' in dt.keys():
            dataset = dt['arr_0'][()]
        else:
            dataset = dt           
            
#        # Check that there are equal num trials per cond:
#        rinfo = dataset['run_info'] if isinstance(dataset['run_info'], dict) else dataset['run_info'][()]
#        if len(list(set([v for k,v in rinfo['ntrials_by_cond'].items()]))) > 1:
#            dataset, data_fpath = fmt.get_equal_reps(dataset, self.data_fpath)
#            self.data_fpath = data_fpath
            
        self.dataset = dataset
        
        # Store run info:
        if isinstance(self.dataset['run_info'], dict):
            self.run_info = self.dataset['run_info']
        else:
            self.run_info = self.dataset['run_info'][()]
        
        # Make sure specified const_trans are actually tested transforms:
        if self.params['const_trans'] != '':
            tested_transforms = [t for t in self.params['const_trans'] if t in self.run_info['trans_types']]
            self.params['const_trans'] = tested_transforms
        
        
        # Store stim configs:
        if isinstance(self.dataset['sconfigs'], dict):
            orig_sconfigs = self.dataset['sconfigs']
        else:
            orig_sconfigs = self.dataset['sconfigs'][()]

        # Make sure numbers are rounded:
        for cname, cdict in orig_sconfigs.items():
            for stimkey, stimval in cdict.items():
                if isinstance(stimval, (int, float)):
                    orig_sconfigs[cname][stimkey] = round(stimval, 1)
                    
        # Add combined 'position' variable to stim configs if class_name == 'position:
        for cname, config in orig_sconfigs.items():
            pos = '_'.join([str(config['xpos']), str(config['ypos'])])
            config.update({'position': pos})
            
                
        if int(self.session) < 20180602:
            # Rename morphs:
            update_configs = [cfg for cfg, info in orig_sconfigs.items() if info['morphlevel'] > 0]
            for cfg in update_configs:
                if orig_sconfigs[cfg]['morphlevel'] == 6:
                    orig_sconfigs[cfg]['morphlevel'] = 27
                elif orig_sconfigs[cfg]['morphlevel'] == 11:
                    orig_sconfigs[cfg]['morphlevel'] = 53
                elif orig_sconfigs[cfg]['morphlevel'] == 16:
                    orig_sconfigs[cfg]['morphlevel'] = 79
                elif orig_sconfigs[cfg]['morphlevel'] == 22:
                    orig_sconfigs[cfg]['morphlevel'] = 106
                else:
                    print "Unknown morphlevel converstion: %i" % orig_sconfigs[cfg]['morphlevel']
        self.sconfigs = orig_sconfigs
#
#            
#        if isinstance(self.dataset['sconfigs'], dict):    
#            self.sconfigs = self.dataset['sconfigs']
#        else:
#            self.sconfigs = self.dataset['sconfigs'][()]

        self.data_identifier = '_'.join((self.animalid, self.session, self.acquisition, self.run, self.traceid))
        
        if visual_area_info is not None:
            (visual_area, visual_areas_fpath), = visual_area_info.items()
            print "Getting ROIs for area: %s" % visual_area
            print "Loading file:", visual_areas_fpath
            with open(visual_areas_fpath, 'rb') as f:
                areas = pkl.load(f)
            if visual_area not in areas.regions.keys():
                print "Specified visual area - %s - NOT FOUND."
                for vi, va in enumerate(areas.regions.keys()):
                    print vi, va
                sel = input("Select IDX of area to use: ")
                visual_area = areas.regions.keys()[sel]
            
#            ret_analysis_id = areas.source.retinoID_rois
#            rdictpath = glob.glob(os.path.join(self.rootdir, areas.source.animalid, areas.source.session, areas.source.acquisition,
#                                               areas.source.run, 'retino_analysis', 'analysisids*.json'))[0]
#            with open(rdictpath, 'r') as f: rdicts = json.load(f)
#            retinoID = rdicts[ret_analysis_id]
#            
#            roi_masks = areas.get_roi_masks(retinoID)
#            nrois = roi_masks.shape[-1]
#            region_mask = areas.regions[visual_area]['region_mask']
#            # Mask ROIs with area mask:
#            region_mask_copy = np.copy(region_mask)
#            region_mask_copy[region_mask==0] = np.nan
#                        
#            included_rois = [ri for ri in range(nrois) if ((roi_masks[:, :, ri] + region_mask_copy) > 1).any()]
            included_rois = [int(ri) for ri in areas.regions[visual_area]['included_rois']]
        else:
            included_rois = None
        self.sample_data, self.sample_labels = self.get_formatted_data(included_rois=included_rois)
        

    def get_formatted_data(self, included_rois=None): #get_training_data(self):
        '''
        Returns input data formatted as:
            ntrials x nrois (data_type=='stat')
            nframes x nrois (data_type = 'frames')
        Filters nrois by roi_selector.
        '''
        print "------------ Formatting data into samples."

        # Get data array:
        assert self.params['inputdata_type'] in self.dataset.keys(), "Specified dtype %s not found. Select from %s." % (self.params['data_type'], str(self.dataset.keys()))
        Xdata = np.array(self.dataset[self.params['inputdata_type']])
        
            
        selected_rois = self.load_roi_list(roi_selector=self.params['roi_selector'])
        if included_rois is not None:
            print "---> only including specified ROIs from visual area."
            roi_list = intersection(selected_rois, included_rois)
        else:
            roi_list = selected_rois
            
        # Get subset of ROIs, if roi_selector is not 'all':
        self.rois = np.array(roi_list)
        if self.rois is not None:
            print "Selecting %i out of %i ROIs (selector: %s)" % (len(self.rois), Xdata.shape[-1], self.params['roi_selector'])
            Xdata = np.squeeze(Xdata[:, self.rois])
        
        # Determine whether all trials have the same structure or not:
        multiple_durs = isinstance(self.run_info['nframes_on'], list)

        # Make sure all conds have same N trials:
        ntrials_by_cond = self.run_info['ntrials_by_cond']
        ntrials_tmp = list(set([v for k, v in ntrials_by_cond.items()]))
        #assert len(ntrials_tmp)==1, "Unequal reps per condition!"
        labels_df = pd.DataFrame(data=self.dataset['labels_data'], columns=self.dataset['labels_columns'])
        
        if self.params['data_type'] == 'stat':
            cX, cy = get_stat_samples(Xdata, labels_df, self.params, multiple_durs=multiple_durs)
        else:
            cX, cy = get_frame_samples(Xdata, labels_df, self.params)
            
        print "Ungrouped dataset cX:", cX.shape
        print "Ungrouped dataset labels cy:", cy.shape
        
        return cX, cy
            
    def load_roi_list(self, roi_selector='visual'):
        
        if roi_selector == 'all':
            roi_list = None
        else:
            roistats_results_fpath = os.path.join(self.traceid_dir, 'sorted_rois', 'roistats_results.npz')
            roistats = np.load(roistats_results_fpath)
            
            roi_subset_type = 'sorted_%s' % roi_selector
            roi_list = roistats[roi_subset_type]
        
        return roi_list
    
    def set_params(self, train_params):
        print "------------ Saving input params for linear classifier."

        # Check arg validity:
        valid_choices = {
                        'data_type': ['frames', 'stat'],
                        'stat_type': ['meanstim', 'zscore', 'meanstimdiff'],
                        'cv_method': ['kfold', 'splithalf', 'LOGO', 'LOO', 'LPGO']
                        }
        for opt, choices in valid_choices.items():
            assert train_params[opt] in choices, "Specified %s --%s-- NOT valid. Select from %s" % (opt, train_params[opt], str(choices))                
        
#        # Make sure specified const_trans are actually tested transforms:
#        if train_params['const_trans'] != '':
#            tested_transforms = [t for t in train_params['const_trans'] if t in self.run_info['trans_types']]
#            train_params['const_trans'] = tested_transforms
#        
        self.params = get_classifier_params(
                                    classifier = 'LinearSVC', 
                                    cv_method = train_params['cv_method'], 
                                    cv_nfolds = int(train_params['cv_nfolds']),
                                    cv_ngroups = train_params['cv_ngroups'],
                                    C_val = train_params['C_val'],
                                    roi_selector = train_params['roi_selector'],         # Can be 'all', or 'visual' or 'selective' (must have sorted ROIs)
                                    data_type = train_params['data_type'],               # Can be 'stat' or 'frames' 
                                    stat_type = train_params['stat_type'],               # Can be 'meanstim', 'zscore', 'meanstimdff'
                                    inputdata_type = train_params['inputdata_type'],     # is '', unless CNMF traces, in which case 'corrected', 'spikes', etc.
                                    get_null = train_params['get_null'],                 # Whether null-class should be trained
                                    class_name = train_params['class_name'],             # Should be a transform (ori, morphlevel, xpos, ypos, size)
                                    #aggregate_type = optsE.aggregate_type,     # Should be 'all', 'single', 'half' -- TOD:  add multiepl trans-constants
                                    const_trans = train_params['const_trans'],           # '' Transform type to hold at a constant value (must be different than class_name)
                                    trans_value = train_params['trans_value'],           # '' Transform value to hold const_trans at
                                    test_set = train_params['test_set'],
                                    indie = train_params['indie'], 
                                    class_subset = [float(c) if c.isdigit() else c for c in train_params['class_subset']],         # LIST of subset of class_name types to include
                                    #subset_nsamples = optsE.subset_nsamples,   #**# None; TODO:  fix this and 'subset' options -- these make no sense
                                    binsize = train_params['binsize'] if train_params['data_type'] =='frames' else '')
    
    def create_classifier_dirs(self):
        print "------------ Creating output directories for each classifier."

        # What are we classifying:
        if len(self.params['class_subset']) > 0:
            class_list = self.params['class_subset']
        else:
            class_list = pd.DataFrame(self.sconfigs).T[self.params['class_name']].unique().tolist()
        nclasses = len(class_list)
        classes_desc = '%i%s' % (nclasses, self.params['class_name'])
        if self.params['get_null']:
            classes_desc = '%s_plusnull' % classes_desc
    
        # Is there a subgroup of class_name that we want to train the classifier on:
        if self.params['const_trans'] is not '':
            train_transforms_list = self.get_constant_transforms()
            
            if self.params['indie'] is True:
                trans_info = dict((trans_name, len(list(set([tdict[trans_name] for tdict in train_transforms_list])))) for trans_name in self.params['const_trans'])
                transforms_desc = '_'.join('%i%s' % (1, k) for k, v in trans_info.items()) # in train_transforms_list for k,v in tdict.items())
                
            else:
                transforms_desc = '_'.join('%s%s' % (k, str(v)) for tdict in train_transforms_list for k,v in tdict.items())
            #transforms_desc = '_'.join('%i%s' % (len(v) if (isinstance(v, list) and len(v) > 1) else v if isinstance(v, (int, float)) else v[0], k) for tdict in train_transforms_list for k,v in tdict.items()) #train_transforms_list.items())

        else:
            train_transforms_list = [] #None
            transforms_desc = 'alltransforms'
            
        if len(self.params['test_set']) > 0:
            testkeys = sorted(self.params['test_set'][0].keys())
            tvals = []
            for tkey in testkeys:
                tvals.append(list(set([tdict[tkey] for tdict in self.params['test_set']])))
            test_str = '_'.join( ['_'.join([str(s) for s in spair]) for spair in [(t, tv) for t, tv in zip(testkeys, tvals)]] )
            transforms_desc = '%s_holdtest_%i_%s' % (transforms_desc, len(self.params['test_set']), test_str) # len(T.params['test_set']))

        transforms_desc = transforms_desc.replace(' ', '')
        transforms_desc = transforms_desc.replace('[','(').replace(']', ')')
        print transforms_desc

        # What is the input data type:
        data_desc = '%s_%s_%s' % (self.params['data_type'], self.params['inputdata_type'], self.params['stat_type'])

        classif_identifier = '{clf}_{cd}_{td}_{rs}_{dd}'.format(clf=self.params['classifier'],
                                                           cd=classes_desc,
                                                           td = transforms_desc,
                                                           rs='%srois' % self.params['roi_selector'],
                                                           dd = data_desc)
        
        # Set output dirs:
        self.classifier_dir = os.path.join(self.traceid_dir, 'classifiers', classif_identifier)
        self.train_transforms_list = train_transforms_list
        print "Creating CLF base dir:", self.classifier_dir
        print "Training classifiers on the following constant transform values:", self.train_transforms_list
            
    # CLASSIFIER CREATION:
    
    def get_constant_transforms(self):
        # Select only those samples w/ values equal to specificed transform value.
        # 'const_trans' :  transform type desired
        # 'trans_value' :  value of const_trans to use.
        train_transforms_list = []
        sconfigs_df = pd.DataFrame(self.sconfigs).T
        const_trans = [trans.strip() for trans in self.params['const_trans']]
        trans_value = self.params['trans_value']
        if trans_value == '': 
            # Get all values for specified const_trans:
            trans_value = []
            for trans in const_trans:
                trans_value.append(sorted(sconfigs_df[trans].unique().tolist()))
            train_all = True
        else:
            trans_value = [float(v) for v in trans_value]
            train_all = False
            
        # Check that all provided trans-values are valid and get config IDs to include:
        const_trans_dict = dict((k, [v]) for k,v in zip([t for t in const_trans], [v for v in trans_value]))

        if const_trans_dict is not None:
            keys, values = zip(*const_trans_dict.items())
            if train_all:
                values_flat = [val[0] for val in values]
                transforms = [dict(zip(keys, v)) for v in itertools.product(*values_flat)]
            else:
                transforms = [dict(zip(keys, v)) for v in itertools.product(*values)]

        else:
            transforms = []
            
        if len(transforms) > 0:
            # Get rid of combinations that weren't actually tested
            remove_pairs = []
            for config_set in transforms:
                found_configs = [s for s,cfg in self.sconfigs.items() if all([cfg[currkey]==currval for currkey, currval in config_set.items()])]
                if len(found_configs) == 0 and train_all is False:
                    remove_pairs.append(config_set)
            if len(remove_pairs) > 0:
                tmp_transforms = [sdict for sdict in transforms if sdict not in remove_pairs]
                transforms = tmp_transforms
            
            # Check if we need to reserve a subset of configs for TEST:
            train_transforms = [tdict for tdict in transforms if tdict not in self.params['test_set']]
    
            # Check if we are training EACH stimulus config independently, or group as 1:
            if self.params['indie'] is False:
                transform_dict = {}
                transform_names = self.params['const_trans']
                for trans_name in transform_names:
                    transform_values = sorted(list(set([tdict[trans_name] for tdict in train_transforms])))
                    transform_dict[trans_name] = transform_values
                train_transforms_list.append(transform_dict)
            else:
                train_transforms_list = train_transforms
                
        return train_transforms_list #const_trans_dict
    
    
    def initialize_classifiers(self, feature_select_method=None, feature_select_n='best', C_select='best'):            
        # If we are testing subset of the data (const_trans and trans_val are non-empty),
        # create a classifier + output subdirs for each subset:
        print "------------ Initializing all classifiers."

        print "--> transform subsets:"
        pp.pprint(self.train_transforms_list)
#        confirm = raw_input("Create classifiers for each transformation?: Enter <Y> to accept, <ENTER> to quit: ")
#        if confirm != 'Y':
#            return
        
        datasource = struct()
        datasource.animalid = self.animalid
        datasource.session = self.session
        datasource.acquisition = self.acquisition
        datasource.run = self.run
        datasource.traceid = self.traceid
        
        if len(self.train_transforms_list) > 0:
            for transform in self.train_transforms_list:
                curr_clfparams = self.params.copy()
                curr_clfparams['const_trans'] = transform.keys()
                curr_clfparams['trans_value'] = sorted([v for v in transform.values()], key=lambda x: transform.keys())
                #print curr_clfparams['const_trans'], curr_clfparams['trans_value']
                self.classifiers.append(LinearSVM(datasource,
                                                  curr_clfparams, 
                                                  self.sample_data, 
                                                  self.sample_labels, 
                                                  self.sconfigs, 
                                                  self.classifier_dir,
                                                  self.run_info,
                                                  feature_select_method=feature_select_method, 
                                                  feature_select_n=feature_select_n,
                                                  C_select=C_select,
                                                  data_identifier=self.data_identifier))
        else:
            self.classifiers.append(LinearSVM(datasource,
                                              self.params, 
                                              self.sample_data,
                                              self.sample_labels,
                                              self.sconfigs,
                                              self.classifier_dir,
                                              self.run_info,
                                              feature_select_method=feature_select_method, 
                                              feature_select_n=feature_select_n,
                                              C_select=C_select,
                                              data_identifier=self.data_identifier))
            
#    def label_classifier_data(self):
#        print "------------ Labeling classifier data."
#        for ci, clf in enumerate(self.classifiers):
#            clf.label_training_data()
#            print "Created %i of %i classifiers: %s" % (ci+1, len(self.classifiers), clf.classifier_dir)

#    def create_classifiers(self):
#        print "------------ Creating classifiers."
#        for ci, clf in enumerate(self.classifiers):
#            clf.create_classifier()


    def create_classifiers(self, feature_select_method='rfe', feature_select_n='best', C_select='best'):
        ##
        '''
        feature_select_method = ('rfe', 'k_best', None)
        feature_select_n = ('best', int, None)
        C_select = ('best', None)
        '''
        print "------------ Determining classifier meta params."
        self.initialize_classifiers(feature_select_method=feature_select_method, 
                                     feature_select_n=feature_select_n, 
                                     C_select=C_select)
#        self.label_classifier_data()
#        self.create_classifiers()
        print "------------ Labeling classifier data."
        for ci, clf in enumerate(self.classifiers):
            print "... %i of %i clfs." % ((ci+1), len(self.classifiers)) #))))
            clf.label_training_data()
            clf.create_classifier()
            print "Created %i of %i classifiers: %s" % (ci+1, len(self.classifiers), clf.classifier_dir)


    def train_classifiers(self, scoring='accuracy', full_train=False, test_size=0.33, col_label=None, row_label=None):
        print "------------ TRAINING."

        for clf in self.classifiers:
            
            clf.do_cv(scoring=scoring, permutation_test=True, n_jobs=4, n_permutations=500)
            clf.train_classifier(full_train=full_train, test_size=test_size)
#            
#            test_transforms_dir = os.path.join(self.classifier_dir, 'test_transforms')
#            if not os.path.exists(test_transforms_dir): os.makedirs(test_transforms_dir)
#            clf.get_classifier_accuracy_by_stimconfig(row_label=row_label, col_label=col_label, output_dir=test_transforms_dir)

#%%
            
class struct():
    pass

class LinearSVM():
    
    def __init__(self, source, clfparams, sample_data, sample_labels, sconfigs, classifier_dir,
                         run_info, feature_select_method=None, feature_select_n=None,
                         C_select=None, data_identifier='', full_train=False, test_size=0.33):
        self.source = source
        self.data_id = '_'.join([source.animalid, source.session, source.acquisition])
        self.svc = None
        self.clfparams = clfparams
        self.cX = sample_data
        self.cy = sample_labels
        self.sconfigs = sconfigs
        self.classifier_dir = classifier_dir
        self.run_info = run_info
        self.data_identifier = data_identifier
        self.cv_results = {'predicted_classes': None,
                           'true_classes': None,
                           'confusion_matrix': None,
                           'config_labels': None,
                           'cross_val_score': None}
        self.model_selection = struct()
        self.model_selection.features = {'method': feature_select_method,
                                         'nfeatures': feature_select_n,
                                         'kept_rids': None,
                                         'score': None}
        self.model_selection.findC = C_select
        
        self.train_results = {}
        self.train_params = {'full_train': full_train,
                             'test_size': test_size}
        self.test_results = {}
        

        if self.clfparams['const_trans'] is not '':
            # Create SUBDIR for specific const-trans and trans-val pair:
            const_trans_dict = dict((k, v) for k,v in zip([t for t in clfparams['const_trans']], [v for v in clfparams['trans_value']]))
            print "TRANSFORMS for current clf:"
            pp.pprint(const_trans_dict)
            print "Testing EACH value of const transforms."
            if any([(isinstance(v, list) and len(v) > 1) for k, v in const_trans_dict.items()]):
                transforms_desc = '_'.join('%i%s' % (len(v), k) for k,v in const_trans_dict.items())
            elif any([(isinstance(v, list) and len(v)==1) for k, v in const_trans_dict.items()]):
                transforms_desc = '_'.join('%s_n%.1f' % (k, abs(v[0])) if v[0] < 0 else '%s_%.1f' % (k, v[0]) for k,v in const_trans_dict.items())
            else:
                transforms_desc = '_'.join('%s_n%.1f' % (k, abs(v)) if v < 0 else '%s_%.1f' % (k, v) for k,v in const_trans_dict.items())
            self.classifier_dir = os.path.join(self.classifier_dir, transforms_desc)
            
        # Set output dirs:
        if not os.path.exists(os.path.join(self.classifier_dir, 'cv_model_selection')):
            os.makedirs(os.path.join(self.classifier_dir, 'cv_model_selection'))
#        if not os.path.exists(os.path.join(self.classifier_dir, 'results')):
#            os.makedirs(os.path.join(self.classifier_dir, 'results'))
            
            
    def label_training_data(self):
        
        if self.clfparams['const_trans'] != '' and len(self.clfparams['class_subset']) == 0:
            # Only group and take subset of data for specified const-trans/trans-value pair
            cX, cy, class_labels, cy_labels = self.group_by_transform_subset()
        
        elif len(self.clfparams['class_subset']) > 0 and self.clfparams['const_trans'] == '':
            # Only group and take subset of data for subset of class-to-be-trained (across all transforms)
            cX, cy, class_labels, cy_labels = self.group_by_class_subset()
        
        elif len(self.clfparams['class_subset']) > 0 and self.clfparams['const_trans'] != '':
            # Only group and take subset of data for class subset within a sub-subset of specified cons-trans/trans-value pair
            cX, cy, class_labels, cy_labels = self.group_by_class_and_transform_subset()
        
        else:
            cX, cy, class_labels, cy_labels = self.group_by_class()
        
        # Check that nsamples are the same for all groups:
        counts_by_class = Counter(cy)
        print counts_by_class
        tmp_cX = cX.copy(); tmp_cy = copy.copy(cy) #.copy();
        remove_ixs=[]
        if len(list(set([v for k,v in counts_by_class.items()]))) > 1:
            print "Uneven n samples for each class."
            print "... making equal..."
            # randomly choose N samples:
            min_num = min([v for k,v in counts_by_class.items()])
            classes_to_subsample = [k for k,v in counts_by_class.items() if v > min_num]
            for cs in classes_to_subsample:
                remove_sample_ixs = random.sample(range(0, counts_by_class[cs]), counts_by_class[cs]-min_num)
                curr_ixs = np.where(cy==cs)[0]
                remove_ixs.append(curr_ixs[remove_sample_ixs])
            discard = [val for sublist in remove_ixs for val in sublist]
            keep = np.array([ix for ix in range(len(tmp_cy)) if ix not in discard])    
            cX = tmp_cX[keep,:]
            cy = tmp_cy[keep]
            
        self.cX = StandardScaler().fit_transform(cX)
        self.cy = cy
        self.class_labels = class_labels
        self.cy_labels = cy_labels

        # Update clfparams where relevant:
        self.clfparams['dual'] = self.cX.shape[0] > self.cX.shape[1]

    
    def group_by_class(self):
        
        cX = self.cX
        cy_tmp = self.cy
        cy = np.array([self.sconfigs[cv][self.clfparams['class_name']] if cv != 'bas' else 'bas' for cv in cy_tmp])
#        if not isinstance(cy[0], (int, float, str, unicode)):
#            cy = [tuple(c) for c in cy]
        class_labels = sorted(list(set(cy))) #sorted(np.unique(cy))
        #cy = np.array(cy)
        
        return cX, cy, class_labels, cy_tmp
    
    def group_by_class_subset(self):
        '''
        Only grab samples belonging to subset of class types.
        Expects a list of included class types,
            e.g., can provide anchors if class_name is 'morphlevel', or can provide xpos values [-20, 20], etc.)
        
        For indexing purposes, cy should still be of form 'config001', 'config002', etc.
        
        Label assignment happens here.
        '''
        
        sconfigs_df = pd.DataFrame(self.sconfigs).T
        configs_tmp = sconfigs_df[sconfigs_df[self.clfparams['class_name']].isin(self.clfparams['class_subset'])].index.tolist()

        # Check if we need to hold any back for TEST SET:
        hold_for_test = []
        if len(self.clfparams['test_set']) > 0:
            for transform in self.clfparams['test_set']:
                subdf = sconfigs_df.copy()
                for transname, transvalue in transform.items():
                    subdf = subdf[subdf[transname]==transvalue]
                hold_for_test.extend(subdf.index.tolist())
            configs_included = [cfg for cfg in configs_tmp if cfg not in hold_for_test]
        else:
            configs_included = configs_tmp
            
        if self.clfparams['get_null']:
            configs_included.append('bas')

        kept_ixs = np.array([cix for cix, cname in enumerate(self.cy) if cname in configs_included])
        cX = self.cX[kept_ixs, :] 
        cy_tmp = self.cy[kept_ixs]
        
        cy = np.array([self.sconfigs[cname][self.clfparams['class_name']] if cname != 'bas' else 'bas' for cname in cy_tmp])
#        if not isinstance(cy[0], (int, float, str, unicode)):
#            cy = [tuple(c) for c in cy]
        class_labels = sorted(list(set(cy))) # sorted(np.unique(cy))
        #cy = np.array(cy)
        
        return cX, cy, class_labels, cy_tmp


    def group_by_transform_subset(self):
        const_trans_dict = dict((k, v) if isinstance(v, list) else (k, [v]) for k,v in zip([t for t in self.clfparams['const_trans']], [v for v in self.clfparams['trans_value']]))
        sconfigs_df = pd.DataFrame(self.sconfigs).T
        configs_included = []; configs_pile = self.sconfigs.keys()
        for transix, (trans_name, trans_value) in enumerate(const_trans_dict.items()):
            if isinstance(trans_value, list):
                assert all([tv in sconfigs_df[trans_name].unique() for tv in trans_value]), "Specified transvalues NOT all in transname %s: %s" % (trans_name, str(trans_value))
            else:
                assert trans_value in sconfigs_df[trans_name].unique(), "Specified trans_name, trans_value not found: %s" % str((trans_name, trans_value))
            
            if transix == 0:
                first_culling = sconfigs_df[sconfigs_df[trans_name].isin(trans_value)].index.tolist()
                configs_tmp = [c for c in configs_pile if c in first_culling]
            else:
                configs_pile = copy.copy(configs_tmp)
                first_culling = sconfigs_df[sconfigs_df[trans_name].isin(trans_value)].index.tolist()
                configs_tmp = [c for c in configs_pile if c in first_culling]

        
        # Check if we need to hold any back for TEST SET:
        hold_for_test = []
        if len(self.clfparams['test_set']) > 0:
            for transform in self.clfparams['test_set']:
                subdf = sconfigs_df.copy()
                for transname, transvalue in transform.items():
                    subdf = subdf[subdf[transname]==transvalue]
                hold_for_test.extend(subdf.index.tolist())
            configs_included = [cfg for cfg in configs_tmp if cfg not in hold_for_test]
        else:
            configs_included = configs_tmp
    
        if self.clfparams['get_null']:
            configs_included.append('bas')
            
        kept_ixs = np.array([cix for cix, cname in enumerate(self.cy) if cname in configs_included])
        cX = self.cX[kept_ixs, :] 
        cy_tmp = self.cy[kept_ixs]
        
        cy = np.array([self.sconfigs[cname][self.clfparams['class_name']] if cname != 'bas' else 'bas' for cname in cy_tmp])
#        if not isinstance(cy[0], (int, float, str, unicode)):
#            cy = [tuple(c) for c in cy]
        class_labels = sorted(list(set(cy))) #sorted(np.unique(cy))
        #cy = np.array(cy)
        
        return cX, cy, class_labels, cy_tmp


    def group_by_class_and_transform_subset(self):
        const_trans_dict = dict((k, v) if isinstance(v, list) else (k, [v]) for k,v in 
                                    zip([t for t in self.clfparams['const_trans']], [v for v in self.clfparams['trans_value']]))
        sconfigs_df = pd.DataFrame(self.sconfigs).T
        configs_subset = sconfigs_df[sconfigs_df[self.clfparams['class_name']].isin(self.clfparams['class_subset'])].index.tolist()
        sconfigs_df = sconfigs_df[sconfigs_df.index.isin(configs_subset)]
        
        configs_included = []; configs_pile = self.sconfigs.keys()
        for transix, (trans_name, trans_value) in enumerate(const_trans_dict.items()):
            if isinstance(trans_value, list):
                assert all([tv in sconfigs_df[trans_name].unique() for tv in trans_value]), "Specified transvalues NOT all in transname %s: %s" % (trans_name, str(trans_value))
            else:
                assert trans_value in sconfigs_df[trans_name].unique(), "Specified trans_name, trans_value not found: %s" % str((trans_name, trans_value))
            
            if transix == 0:
                first_culling = sconfigs_df[sconfigs_df[trans_name].isin(trans_value)].index.tolist()
                configs_tmp = [c for c in configs_pile if c in first_culling]
            else:
                configs_pile = copy.copy(configs_tmp)
                first_culling = sconfigs_df[sconfigs_df[trans_name].isin(trans_value)].index.tolist()
                configs_tmp = [c for c in configs_pile if c in first_culling]
            
            print "Trans: %s (= %s) is %i configs." % (trans_name, str(trans_value), len(configs_tmp))
        
        # Check if we need to hold any back for TEST SET:
        hold_for_test = []
        if len(self.clfparams['test_set']) > 0:
            for transform in self.clfparams['test_set']:
                subdf = sconfigs_df.copy()
                for transname, transvalue in transform.items():
                    subdf = subdf[subdf[transname]==transvalue]
                hold_for_test.extend(subdf.index.tolist())
            configs_included = [cfg for cfg in configs_tmp if cfg not in hold_for_test]
        else:
            configs_included = configs_tmp
        
    
        if self.clfparams['get_null']:
            configs_included.append('bas')
            
        kept_ixs = np.array([cix for cix, cname in enumerate(self.cy) if cname in configs_included])
        print "Training classifier on %i different stimulus configs (nsamples = %i)" % (len(configs_included), len(kept_ixs))
        #print "Current scongfig has %i trials for training." % len(kept_ixs)
        
        cX = self.cX[kept_ixs, :] 
        cy_tmp = self.cy[kept_ixs]
        
        cy = np.array([self.sconfigs[cname][self.clfparams['class_name']] if cname != 'bas' else 'bas' for cname in cy_tmp])
#        if not isinstance(cy[0], (int, float, str, unicode)):
#            cy = [tuple(c) for c in cy]
        class_labels = sorted(list(set(cy))) #sorted(np.unique(cy))
        #cy = np.array(cy)
        
        return cX, cy, class_labels, cy_tmp
    

    def iterate_feature_selection(self, method='rfe', scoring='accuracy', n_jobs=4, output_dir=None, figname=None):
        feature_selection = {}
        
        valid_choices = ['rfe', 'k_best']
        assert method in valid_choices, "Invalid feature selection method chosen. Select from: %s" % str(valid_choices)
        
        nrois_total = self.cX.shape[-1]

        roi_list = np.arange(0, nrois_total)[::-1]
        iter_results = {}
        if method == 'rfe':
            rfe = RFECV(estimator=self.svc, step=1, cv=StratifiedKFold(self.clfparams['cv_nfolds']), scoring=scoring, n_jobs=n_jobs)
            fit = rfe.fit(self.cX, self.cy)
            best_score = fit.grid_scores_[fit.n_features_ - 1]
            kept_rids = fit.get_support(indices=True)
            nfeatures_best = len(kept_rids) #fit.n_features_
            print "N features", nfeatures_best, fit.n_features_, fit.grid_scores_.shape
            all_scores = fit.grid_scores_
        elif method == 'k_best':
            scoring = 'R^2'
            for nrois in roi_list:
                rfe = SelectKBest(f_classif, k=nrois)
                fit = rfe.fit(self.cX, self.cy)
                score = fit.score(self.cX, self.cy)
                kept_rids = fit.get_support(indices=True)
                iter_results[nrois] = {'kept_rids': kept_rids, 'score': score}
            all_scores = sorted([v['score'] for k, v in iter_results.items()], key=lambda x: x[0])
            nfeatures_best = np.argmax(all_scores)
            kept_rids = iter_results[nfeatures_best]['kept_rids']
            best_score = iter_results[nfeatures_best]['score']

        fig = pl.figure()
        pl.plot(range(1, len(all_scores) + 1), all_scores)
        pl.xlabel("N features selected")
        pl.ylabel(scoring)
        pl.title("optimal: %i" % len(kept_rids))
        label_figure(fig, self.data_identifier)
        
        if output_dir is not None:
            if figname is None:
                figname = 'feature_selection_%s_%s.png' % (method, scoring)
            pl.savefig(os.path.join(output_dir, figname))
            pl.close()
        
        feature_selection['method'] = method
        feature_selection['nfeatures'] = nfeatures_best
        feature_selection['kept_rids'] = kept_rids
        feature_selection['score'] = best_score
        
        return feature_selection
            

    def do_model_selection(self, scoring='accuracy', output_dir=None, meta_params_str='metaparams'):
        #output_dir = os.path.join(self.classifier_dir, 'cv_model_selection')
        
        feature_selection = self.model_selection.features
        
        self.svc = LinearSVC(random_state=0, dual=self.clfparams['dual'], multi_class='ovr') #, C=self.clfparams['C_val'])
        
        if self.model_selection.findC == 'best':
            print "Finding optimal C value..."
            self.clfparams['C_val'] = get_best_C(self.svc, self.cX, self.cy, 
                                              output_dir=output_dir, 
                                              figname='findC_%s_%s.png' % (meta_params_str, self.data_id))
            
        print "[PARAMS]: C = %.5f" % self.clfparams['C_val']
        self.svc.C = self.clfparams['C_val']

        if self.model_selection.features['method'] is not None:
            if self.model_selection.features['nfeatures'] is None or self.model_selection.features['nfeatures'] == 'best':
                feature_selection = self.iterate_feature_selection(method=self.model_selection.features['method'],
                                                                   scoring=scoring, output_dir=output_dir,
                                                                   figname='feature_selection_%s_%s_%s.png' % (meta_params_str, 
                                                                                                               scoring, self.data_id))
                cX_tmp = self.cX[:, feature_selection['kept_rids']]
            else:
                if self.model_selection.features['nfeatures'] > self.cX.shape[0]:
                    self.model_selection.features['nfeatures'] = self.cX.shape[0]-1
                    
                if self.model_selection.features['method'] == 'k_best':
                    feature_fit = SelectKBest(f_classif, k=self.model_selection.features['nfeatures']).fit(self.cX, self.cy)
                    kids = feature_fit.get_support(indices=True)
                    kbest_scores = cross_val_score(self.svc, self.cX[:, kids], self.cy)
                    scoring = np.mean(kbest_scores)
                elif self.model_selection.features['method'] == 'RFE':
                    feature_fit = RFE(self.svc, n_features_to_select=self.model_selection.features['nfeatures'])
                    scoring = feature_fit.score(self.cX, self.cy)
                feature_selection['kept_rids'] = feature_fit.get_support(indices=True)
                feature_selection['score'] = scoring
                cX_tmp = feature_fit.transform(self.cX)
            
            self.cX = cX_tmp
            self.model_selection.features = feature_selection
        
        
    def create_classifier(self):
        '''
        Does some model selection to choose classifier parameters and data subsets.
        Creates classifier sub-dir for feature-selection and C-value    
        '''
        feature_select_method = self.model_selection.features['method']
        feature_select_n = self.model_selection.features['nfeatures']
        C_select = self.model_selection.findC
        
        meta_params_str = '%s_%s_C_%s' % ('all' if feature_select_method is None else feature_select_method,
                                          'features' if feature_select_n is None else str(feature_select_n),
                                          C_select if isinstance(C_select, str) else '%.4f' % self.clfparams['C_val'])
                
        output_dir = os.path.join(self.classifier_dir, 'cv_model_selection')
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        print "MODEL SELECTION: Saving to subdir: ./cv_model_selection"
        
        # 1.  Do model selection, if relevant:
        # --------------------------------------------
        self.do_model_selection(output_dir=output_dir, meta_params_str=meta_params_str)
        
        # Save clfparams and check file hash:
        # --------------------------------------------
        clfparams_hash = hashlib.md5(json.dumps(self.clfparams, ensure_ascii=True, indent=4, sort_keys=True)).hexdigest()
        self.hash = clfparams_hash
        
        if self.model_selection.features['method'] is not None:
            feature_str = 'select%i_%s' % (self.model_selection.features['nfeatures'], self.model_selection.features['method'])
        else:
            feature_str = 'selectall'
        C_str = 'C%.5f' % self.clfparams['C_val']
            
        # Create output dir:
        # --------------------------------------------
        model_select_str = '%s_%s' % (feature_str, C_str)
#        clf_output_dir = os.path.join(self.classifier_dir, clf_subdir)
#        if not os.path.exists(os.path.join(clf_output_dir, 'figures')): os.makedirs(os.path.join(clf_output_dir, 'figures'))
#        if not os.path.exists(os.path.join(clf_output_dir, 'results')): os.makedirs(os.path.join(clf_output_dir, 'results'))
#
#        print "Saving current CLF results to:", clf_output_dir
#        self.classifier_dir = clf_output_dir
        
        clfparams_fpath = os.path.join(output_dir, 'clfparams_%s_%s_%s_%s.json' % (clfparams_hash[0:6], self.data_id, meta_params_str, model_select_str))
        with open(clfparams_fpath, 'w') as f: 
            json.dump(self.clfparams, f, indent=4, sort_keys=True)
        

    def do_cv(self, output_dir=None, scoring='accuracy', permutation_test=True, n_jobs=4, n_permutations=500):
        
        if output_dir is None:
            output_dir = os.path.join(self.classifier_dir, 'cross_validation')
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        
        print "Training classifier.\n--- output saved to: %s" % output_dir
        if self.clfparams['data_type'] == 'frames':
            self.train_on_trial_epochs()
        else:
            self.train_on_trials(output_dir=output_dir)
            
    def train_on_trial_epochs(self):
        
        epochs, decode_dict, bins, self.clfparams['binsize'], class_labels = \
                                        format_epoch_dataset(self.clfparams, self.cX, self.cy, self.run_info, self.sconfigs)
                                        
        decode_trial_epochs(self.class_labels, self.clfparams, bins, decode_dict, 
                                self.run_info, data_identifier=self.data_identifier, 
                                niterations=10, scoring='accuracy', output_dir=self.classifier_dir)
        
    def train_on_trials(self, scoring='accuracy', output_dir=None, permutation_test=True, n_permutations=500, n_jobs=4):
        
        #print "... doing RFE."
        #self.do_RFE(scoring=scoring)
                
        print "... running permutation test for CV accuracy."
        self.cv_kfold_permutation(scoring=scoring, 
                                  permutation_test=permutation_test, 
                                  n_jobs=n_jobs,
                                  n_permutations=n_permutations,
                                  output_dir=output_dir,
                                  figname='cv_permutation_%s_%s_%s.png' % (self.clfparams['cv_method'], self.clfparams['cv_nfolds'], self.data_id))
        
        print "... plotting confusion matrix."
        self.confusion_matrix(output_dir=output_dir, figname='confusion_matrix_%s.png' % self.data_id)
        
        
        
    def cv_kfold_permutation(self, scoring='accuracy', permutation_test=True, n_jobs=4, n_permutations=500, output_dir=None, figname=None):
        # -----------------------------------------------------------------------------
        # Do cross-validation
        # -----------------------------------------------------------------------------
        kfold = StratifiedKFold(n_splits=self.clfparams['cv_nfolds'], shuffle=True)
    
        cv_results = cross_val_score(self.svc, self.cX, self.cy, cv=kfold, scoring=scoring)
        print "CV RESULTS [%s]: %.3f (%.3f)" % (scoring, cv_results.mean(), cv_results.std()*2.) # Print score and 95% CI of score estimate
        self.cv_results['cross_val_score'] = cv_results
        
        if permutation_test:
            # -----------------------------------------------------------------------------
            # How significant is our classification score(s)?
            # Calculate p-value as percentage of runs for which obtained score is greater 
            # than the initial classification score (i.e., repeat classification after
            # randomizing and permuting labels).
            # -----------------------------------------------------------------------------
            score, permutation_scores, pvalue = permutation_test_score(
                                                    self.svc, self.cX, self.cy, 
                                                    scoring=scoring, 
                                                    cv=kfold, 
                                                    n_permutations=n_permutations, 
                                                    n_jobs=n_jobs)
            
            print("Classification score %s (pvalue : %s)" % (score, pvalue))

            # View histogram of permutation scores            
            fig = pl.figure()
            n_classes = np.unique([self.cy]).size
            
            pl.hist(permutation_scores, 20, label='Permutation scores', edgecolor='black')
            ylim = pl.ylim()
            pl.plot(2 * [score], ylim, '--g', linewidth=3, label='Classification Score %.4f (p=%.4f)' % (score, pvalue))
            pl.plot(2 * [1. / n_classes], ylim, '--k', linewidth=3, label='Chance')
            pl.ylim(ylim)
            pl.legend()
            pl.xlabel('Score - %s' % scoring)
            
            if self.svc.C == 1: Cstring = 'C1';
            elif self.svc.C == 1E9: Cstring = 'bigC';
            else: Cstring = 'C%i' % self.svc.C
                
            label_figure(fig, self.data_identifier)
            if figname is None:
                figname = 'cv_permutation_test_%s_%s_%i.png' % (Cstring, self.clfparams['cv_method'], self.clfparams['cv_nfolds'])       
            print "CV permutations figure:", figname
            if output_dir is not None:
                pl.savefig(os.path.join(output_dir, figname))
                pl.close()


    def confusion_matrix(self, output_dir=None, figname=None):
#        output_dir = os.path.join(self.classifier_dir, 'cv_model_selection')
        
        predicted, true, classes, config_labels = get_cv_folds(self.svc, self.clfparams, self.cX, self.cy, cy_labels=self.cy_labels, 
                                                               output_dir=output_dir, data_id = self.data_id)
        cv_confusion = plot_confusion_matrix_subplots(predicted, true, classes, cv_method=self.clfparams['cv_method'], 
                                                      data_identifier=self.data_identifier, output_dir=output_dir, figname=figname)
        self.cv_results['predicted_classes'] = predicted
        self.cv_results['true_classes'] = true
        self.cv_results['config_labels'] = config_labels
        self.cv_results['confusion_matrix'] = cv_confusion

#    def do_RFE(self, scoring='accuracy'):
#        results_topN = iterate_RFE(self.clfparams, self.cX, self.cy, scoring=scoring, 
#                                       output_dir=self.classifier_dir)
#        plot_RFE_results(results_topN, len(self.class_labels), scoring=scoring, 
#                                     data_identifier=self.data_identifier, 
#                                     output_dir=self.classifier_dir)

#    def get_clf_RFE(self):
#        rfe_summary = {}
#        #clf_key = os.path.split(self.classifier_dir)[-1]
#        rfe_results = self.load_RFE_results()
#        rfe_scores = np.array([np.mean(cv_scores) for ri, cv_scores in enumerate(rfe_results['results'])])
#        best_iter = rfe_scores.argmax()
#        kept_rids = rfe_results['kept_rids_by_iter'][best_iter-1]
#        bestN = len(kept_rids)
#        print ">>> RFE RESULTS:  Best score: %.2f" % rfe_scores.max(), "N rois: %i" % bestN
#        print " --->", kept_rids
#        rfe_summary = {'iter_ix': best_iter,
#                       'kept_rids': kept_rids,
#                       'max_score': rfe_scores.max()}
#        return rfe_summary
#        

#    def load_RFE_results(self):
#        
#        found_rfe_results = sorted(glob.glob(os.path.join(self.classifier_dir, 'results', 'RFE*.pkl')), key=natural_keys)
#        if len(found_rfe_results) == 0:
#            print "No RFE results found in clf dir: %s" % self.classifier_dir
#            self.do_RFE(scoring='accuracy')
#        if len(found_rfe_results) > 1:
#            for ri, rfe in enumerate(found_rfe_results):
#                print ri, rfe
#            sel = input("Select IDX of RFE results to use: ")
#            rfe_fpath = found_rfe_results[sel]
#        else:
#            rfe_fpath = found_rfe_results[0]
#        
#        with open(rfe_fpath, 'rb') as f:
#            rfe_results = pkl.load(f)
#    
#        return rfe_results

    def train_classifier(self, full_train=False, test_size=0.2):
        '''
        Returns dict: traindata
            'train_data': split (or full) cX input data
            'train_labels': corresponding labels
            'test_data':  subset of cX held out for validation (None, if full_train=True)
            'test_labels': corresponding labels (None, if full_train=True)
            'predicted': classifier predictions on held out test set (None, if full_train=True)
            'results_by_config': classifier accuracy on test set, split by stim config (CV accuracy split by stim config, if full_train=True)
            'results_by_class':  classifier accuracy on test set for trained labels (CV accuracy, if full_train=True)
        '''
        self.train_params['full_train'] = full_train
        self.train_params['test_size'] = test_size
        
        if self.train_params['full_train']:
            training_regime = 'trainfull'
        else:
            training_regime = 'trainpart_testsize%.2f' % self.train_params['test_size']
        
        
        sdf = pd.DataFrame(self.sconfigs).T
        if self.clfparams['class_subset'] == '':
            classes = list(sdf[self.clfparams['class_name']].unique())
        else:
            classes = self.clfparams['class_subset']
        print "----- ----- Training classifier on %i labels: %s" % (len(classes), str(classes))
        
        X_test = None; test_true=None; test_predicted=None;
        traintest_results = {'by_class': {},
                             'by_config': {}}

        # Turn data and labels into dataframes to preserve indices:
        cX = pd.DataFrame(self.cX)
        cy = pd.Series(self.cy)
        
        if self.train_params['full_train']:
            # Fit with data:
            self.svc.fit(cX, cy)
            
            # Get CV results for trained labels:
            for classix, classname in enumerate(classes):
                ncorrect = self.cv_results['confusion_matrix'][classix,classix]
                ntotal = np.sum(self.cv_results['confusion_matrix'][classix, :])
                traintest_results['by_class'][classname] = float(ncorrect) / float(ntotal)
            X_train = cX
            train_true = cy
            
            # Split CV results by confg, since we don't have a held out test set:
            traintest_by_config = self.split_cv_by_stimconfig()
        else:
            # Split input data into train/test sets:
            X_train, X_test, train_true, test_true = train_test_split(cX, cy, test_size=self.train_params['test_size'], random_state=0, shuffle=True)
            
            # Fit with training set:
            self.svc.fit(X_train, train_true)
            
            # Predict on test set:
            test_predicted = self.svc.predict(X_test)
            
            # Get test accuracy for trained classes:
            for classix, classname in enumerate(sorted(np.unique(test_true))):
                # Use re-zero'ed indices of test_true to index into test_predicted, since test_true is indexed by original indices:
                sample_ixs = np.array([i for i, orig_ix in enumerate(test_true.index.tolist()) if orig_ix in test_true[test_true==classname].index.tolist()])
                curr_pred = test_predicted[sample_ixs]
                ncorrect = np.sum([p==classname for p in curr_pred])
                ntotal = len(sample_ixs)
                traintest_results['by_class'][classname] = float(ncorrect) / float(ntotal)
    
            # Get test accuracy for trained classes, split by stimulus config:
            all_configs = sorted(list(set(self.cy_labels)), key=natural_keys)
            orig_ixs_testset = test_true.index.tolist()
            test_labels_by_config = self.cy_labels[orig_ixs_testset]
            # if not every stimulus config was covered in test set, just use CV accuracy for those:
            missing_in_test = [cfg for cfg in all_configs if cfg not in list(set(test_labels_by_config))]
            if len(missing_in_test) > 0:
                cv_by_config = self.split_cv_by_stimconfig()
            traintest_by_config = dict((cfg, {'ncorrect': [], 'ntotal': []}) for cfg in all_configs)
            for cfg in all_configs:
                if cfg in missing_in_test:
                    traintest_by_config[cfg]['ncorrect'] = np.sum(cv_by_config[cfg]['ncorrect'])
                    traintest_by_config[cfg]['ntotal'] = np.sum(cv_by_config[cfg]['ntotal'])
                    traintest_by_config[cfg]['percent_correct'] = cv_by_config[cfg]['percent_correct']
                else:
                    curr_trials = np.where(test_labels_by_config==cfg)[0]
                    trials_by_orig_ix = np.array([orig_ix for i, orig_ix in enumerate(test_true.index.tolist()) if i in curr_trials])
                    curr_true_class = test_true[trials_by_orig_ix].unique()[0]
                    curr_pred_class = test_predicted[curr_trials]
                    traintest_by_config[cfg]['ncorrect'] = float(np.sum([p==curr_true_class for p in curr_pred_class]))
                    traintest_by_config[cfg]['ntotal'] = float(len(curr_trials))
                    traintest_by_config[cfg]['percent_correct'] = traintest_by_config[cfg]['ncorrect']  / traintest_by_config[cfg]['ntotal']
            
        traintest_results['by_config'] = traintest_by_config
    
    #    testdata = {'data': X_test, 'labels': test_true, 'predicted': test_predicted}
        traindata = {'train_data': X_train, 
                     'train_labels': train_true,
                     'test_data': X_test,
                     'test_labels': test_true,
                     'predicted': test_predicted,
                     'results_by_config': traintest_results['by_config'],
                     'results_by_class': traintest_results['by_class'],
                     'training_regime': training_regime}
        
        self.train_results = traindata
        
        return traindata
    

    def test_classifier(self, test_data=None, test_labels=None, config_labels=None):
        
        if test_data is None:
            test_results = copy.copy(self.train_results)
        else:
            test_results = {}
            predicted_classes = self.svc.predict(test_data)
            
            # Get accuracy by CLASS:
            tested_classes = list(set(test_labels))
            test_by_class = dict((cfg, {}) for cfg in tested_classes)
            for tested_class in tested_classes:
                curr_trials = np.where(np.array(test_labels) == tested_class)[0]
#                curr_predictions = predicted_classes[curr_trials]
#                test_by_class[tested_class]['ncorrect'] = float(np.sum([p==tested_class for p in curr_predictions]))
#                test_by_class[tested_class]['ntotal'] = float(len(curr_trials))
                test_by_class[tested_class]['predicted'] = predicted_classes[curr_trials]
                test_by_class[tested_class]['true'] = test_labels[curr_trials]
                test_by_class[tested_class]['indices'] = curr_trials

#                
#                # Also add held-out set tested during training/validation:
#                if self.train_params['full_train'] is False:
#                    test_by_class[tested_class]['ncorrect'] += float(np.sum([p==tested_class for p in self.train_results['predicted']]))
#                    test_by_class[tested_class]['ntotal'] += float(len(self.train_results['predicted']))

#                test_by_class[tested_class]['percent_correct'] = test_by_class[tested_class]['ncorrect'] / test_by_class[tested_class]['ntotal'] 
            
            # Save results split by stimulus config:
            tested_configs = sorted(list(set(config_labels)), key=natural_keys)
            test_by_config = dict((cfg, {}) for cfg in tested_configs)
            for cfg in tested_configs:
                curr_trials = np.where(config_labels == cfg)[0]
#                true_label = list(set([pred for pi, pred in enumerate(test_labels) if pi in curr_trials]))[0]
#                curr_predictions = predicted_classes[curr_trials]                
#                test_by_config[cfg]['ncorrect'] = float(np.sum([p==true_label for p in curr_predictions]))
#                test_by_config[cfg]['ntotal'] = float(len(curr_trials))
#                test_by_config[cfg]['percent_correct'] = test_by_config[cfg]['ncorrect']  / test_by_config[cfg]['ntotal']
                test_by_config[cfg]['predicted'] = predicted_classes[curr_trials]
                test_by_config[cfg]['true'] = test_labels[curr_trials]
                test_by_config[cfg]['indices'] = curr_trials
#            
#            if self.train_params['full_train'] is False:
#                # Add held-out test set from cross-validation, if relevant:
#                train_configs = self.train_results['results_by_config'].keys()
#                for cfg in train_configs:
#                    if cfg not in test_by_config.keys():
#                        print "Adding training held-out tested configs"
#                        test_by_config[cfg] = self.train_results['results_by_config'][cfg]

                
            test_results = {'test_data': test_data,
                            'test_labels': test_labels,
                            'predicted_classes': predicted_classes,
                            'results_by_config': test_by_config,
                            'results_by_class': test_by_class,
                            'config_labels': config_labels}
            
        self.test_results = test_results
            
        return test_results
    

    def split_cv_by_stimconfig(self):

        # Also save CV results by config, if relevant:
        trained_configs = list(set([cfg for sublist in self.cv_results['config_labels'] for cfg in sublist]))
        cv_by_config = dict((cfg, {'ncorrect': [], 'ntotal': []}) for cfg in trained_configs)
        for cfg in trained_configs:
            for fold in range(len(self.cv_results['predicted_classes'])):
                tested_ixs = np.where(self.cv_results['config_labels'][fold]==cfg)[0]
                if len(tested_ixs)> 0:
                    curr_true = self.cv_results['true_classes'][fold][tested_ixs]
                    curr_pred = self.cv_results['predicted_classes'][fold][tested_ixs]
                    curr_ncorrect = sum([tru==guess for tru, guess in zip(curr_true, curr_pred)])
                    curr_ntotal = len(tested_ixs)
                else:
                    curr_ncorrect = 0
                    curr_ntotal = 0
                cv_by_config[cfg]['ncorrect'].append(curr_ncorrect)
                cv_by_config[cfg]['ntotal'].append(curr_ntotal)
            cv_by_config[cfg]['percent_correct'] = np.nanmean([float(corr)/float(tot) if tot > 0 else np.nan \
                                                               for corr, tot in zip(cv_by_config[cfg]['ncorrect'], cv_by_config[cfg]['ntotal'])])
        
        return cv_by_config
    
    
    
    def get_classifier_accuracy_by_stimconfig(self, m50=53, m100=106, row_label=None, col_label=None, output_dir=None, figname=None): # full_train=False, test_size=0.33):
#        
#        if output_dir is None:
#            test_transforms_dir = os.path.join(self.classifier_dir, 'test_transforms')
#            if not os.path.exists(test_transforms_dir): os.makedirs(test_transforms_dir)
        sdf = pd.DataFrame(self.sconfigs).T 

        if self.train_params['full_train']:
            train_set = 'fulltrain'
        else:
            train_set = 'testsize%.2f' % self.train_params['test_size']
        
        test_results_accuracy = self.convert_predictions_to_accuracy(m50=m50, m100=m100)
        accuracy, counts = self.split_test_results_by_stimconfig(test_results_accuracy)
        
#        row_label = 'ypos'
#        col_label = 'xpos'
        accuracy_grid, counts_grid, config_grid = self.get_stimconfig_grid(accuracy, counts, row_label=row_label, col_label=col_label)
        
        # Plot PERFORMANCE:
        ignore_row = False; ignore_column = False;
        if isinstance(accuracy.keys()[0], (float, int)):
            if col_label is not None:
                colvals = sorted(list(sdf[col_label].unique()))
                if len(colvals) == 1:
                    ignore_column = True
                else:
                    ignore_column = False
            if row_label is not None:
                rowvals = sorted(list(sdf[row_label].unique()))
                if len(rowvals) == 1:
                    ignore_row = True
                else:
                    ignore_row = False
            
        if ignore_row:
            colvals = sorted(list(set([stim for stim in accuracy.keys()])))
            rowvals = np.ones((len(colvals),)) * rowvals[0]
        if ignore_column:
            rowvals = sorted(list(set([stim for stim in accuracy.keys()])))
            colvals = np.ones((len(rowvals),)) * colvals[0]
        else:
            rowvals = sorted(list(set([stim[1] for stim in accuracy.keys()])))
            colvals = sorted(list(set([stim[0] for stim in accuracy.keys()])))
            
        fig, axes = pl.subplots(1,2,figsize=(14,8)) #, sharey=True)
        
        chance_level = 1./len(self.class_labels)
        
        plot_transform_grid(accuracy_grid, rowvals=rowvals, colvals=colvals,
                            ylabel=row_label, xlabel=col_label, cmap='hot', vmin=chance_level, vmax=1., 
                            data_identifier=self.data_identifier, ax=axes[0])
        plot_transform_grid(counts_grid, rowvals=rowvals, colvals=colvals,
                            ylabel=row_label, xlabel=col_label, cmap='Blues_r', vmin=0, vmax=counts_grid.max(),
                            data_identifier=self.data_identifier, ax=axes[1])
        
        label_figure(fig, self.data_identifier)
        
        if output_dir is not None:
            if figname is None:
                figname = 'test_transforms_performance_grid_%s_%i%s_%s.png' % (train_set, len(self.class_labels), self.clfparams['class_name'], self.data_id)
            pl.savefig(os.path.join(output_dir, figname))
            pl.close()
            
        return accuracy_grid, counts_grid, config_grid
    
    def get_stimconfig_grid(self, accuracy, counts, row_label=None, col_label=None):
        sdf = pd.DataFrame(self.sconfigs).T 
        if self.clfparams['class_name'] == 'position':
            # We actually train on stimconfig, and should be averaging across object ID (or orientation):
            trans_types = ['position']
        else:
            trans_types = [trans for trans in self.run_info['trans_types'] if trans != self.clfparams['class_name']]


        if row_label is None or col_label is None:
#            trans_types = [trans for trans in clf.run_info['trans_types'] if trans != clf.clfparams['class_name']]

            if len(trans_types) > 0:
                print "No row or column label specified. Found %i transform types: %s" % (len(trans_types), ', '.join(trans_types))
                if len(trans_types) == 1:
                    col_label = trans_types[0]
                    row_label = None
                elif len(trans_types) > 1:
                    row_label = trans_types[0]
                    col_label = trans_types[1]
        
        placehold = False
        if col_label is not None:
            colvals = sorted(list(sdf[col_label].unique()))
        else:
            colvals = [0]
            placehold = True
        if row_label is not None:
            rowvals = sorted(list(sdf[row_label].unique()))
        else:
            rowvals = [0]
            placehold = True
            
        print "COLUMNS: %s %s" % (col_label, str(colvals))
        print "ROWS: %s %s" % (row_label, str(rowvals))

        ncorrect_grid = np.ones((len(rowvals), len(colvals)))*np.nan
        counts_grid = np.ones((len(rowvals), len(colvals)))*np.nan
        config_grid = {}
        grid_pairs = sorted(list(itertools.product(colvals, rowvals)), key=lambda x: (x[0], x[1]))
        for trans_config in grid_pairs:
            if placehold is True and not all([t in accuracy.keys() for t in trans_config]):
                continue
            if len(colvals) == 1:
                if trans_config[1] not in accuracy.keys():
                    continue
                else:
                    dict_key = trans_config[1]
                    
            elif len(rowvals) == 1:
                if trans_config[0] not in accuracy.keys():
                    continue
                else:
                    dict_key = trans_config[0]
                    
            elif (len(colvals) > 1 and len(rowvals) > 1):
                if trans_config not in accuracy.keys():
                    continue
                else:
                    dict_key = trans_config
            
            if 'position' in trans_types and isinstance(trans_config, str):
                # Convert str to tuple for indexing:
                trans_config = tuple([float(p) for p in dict_key.split('_')])
                dict_key = trans_config
            rix = rowvals.index(trans_config[1])
            cix = colvals.index(trans_config[0])
            if placehold: # second trans_config value is fake
                nc = accuracy[dict_key]
                cc = counts[dict_key]
            else:
                nc = accuracy[dict_key]
                cc = counts[dict_key]
                
            if np.isnan(ncorrect_grid[rix, cix]):
                ncorrect_grid[rix, cix] = nc #ncorrect[trans_config]
                counts_grid[rix, cix] = cc #counts[trans_config]
            else:
                ncorrect_grid[rix, cix] += nc #ncorrect[trans_config] 
                counts_grid[rix, cix] += cc #counts[trans_config]
            config_grid[trans_config] = (rix, cix)

        return ncorrect_grid, counts_grid, config_grid

        
    def split_test_results_by_stimconfig(self, test_results_accuracy):
        if self.clfparams['class_name'] == 'position':
            # We actually train on stimconfig, and should be averaging across object ID (or orientation):
            trans_types = ['position']
        else:
            trans_types = [trans for trans in self.run_info['trans_types'] if trans != self.clfparams['class_name']]
        print "Transform types:", trans_types
        
        sdf = pd.DataFrame(self.sconfigs).T 
    
        # Get TRAINED stimulus configs:
        if self.clfparams['const_trans'] is not '':
            # Trained on subset of configs:
            transforms = dict((k, v) for k,v in  zip(self.clfparams['const_trans'], self.clfparams['trans_value']))
            keys, values = zip(*transforms.items())
            if isinstance(values[0], list):
                trained_configs_list = [dict(zip(keys, v)) for v in itertools.product(*values)]
            else:
                values_list = [[v] for v in values]
                trained_configs_list = [dict(zip(keys, v)) for v in itertools.product(*values_list)]
        else:
            # Trained on ALL configs:
            transforms = dict((trans, list(sdf[trans].unique())) for trans in trans_types)
            keys, values = zip(*transforms.items())
            #values_flat = [val[0] for val in values]
            trained_configs_list = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        train_transforms = [tuple(tdict.values()) for tdict in trained_configs_list if tdict not in self.clfparams['test_set']]
        print "Current CLF trained on %i stim configs (%s)." % (len(train_transforms), ', '.join(trans_types))
        
        grouped_sdf = sdf.groupby(trans_types)
        accuracy={}; counts={};
        test_configs = [t for t in list(set(test_results_accuracy.keys())) if test_results_accuracy[t]['ntotal'] > 0]
        for config, g in grouped_sdf:
            if 'position' in trans_types:
                # Convert str to tuple for indexing:
                config = tuple([float(p) for p in config.split('_')])
            print '%s: %i cfgs' % (str(config), len(g.index.tolist()))
            # Get mean accuracy (averaged across cv folds) for included configs:
            tmp_curr_configs = g.index.tolist()
            curr_configs = [t for t in tmp_curr_configs if t in test_configs]
            curr_scores = np.mean([test_results_accuracy[cfg]['percent_correct'] for cfg in curr_configs]) #g.index.tolist()])
            curr_counts = np.sum([test_results_accuracy[cfg]['ntotal'] for cfg in curr_configs]) # g.index.tolist()])
            accuracy[config] = curr_scores
            counts[config] = curr_counts
            #counts[config] = clf.train_results['train_labels']
        
        return accuracy, counts


    def convert_predictions_to_accuracy(self, m50=53, m100=106):
        sdf = pd.DataFrame(self.sconfigs).T
        
        test_by_config = self.test_results['results_by_config']
        orig_configs = self.test_results['config_labels']
        tested_configs = sorted(list(set(orig_configs)), key=natural_keys)
        
        test_results_accuracy = dict((cfg, {}) for cfg in tested_configs)
        for tested_config in tested_configs:
            if self.clfparams['class_name'] == 'morphlevel':
                if sdf.loc[tested_config]['morphlevel'] == m50:
                    ncorrect = np.nan; ntotal=np.nan; pcorrect=np.nan;
                else:
                    config_names_all = orig_configs[test_by_config[tested_config]['indices']]
                    true_labels_all = np.array([sdf.loc[cfg]['morphlevel'] for cfg in config_names_all])
                    #print "----- ----- excluding middle morphs (%i) to calculate accuracy." % m50
                    excluding_midmorph_ixs = np.array([ti for ti, tvalue in enumerate(true_labels_all) if tvalue != m50])
                    
                    # Take only those tested samples that are NOT mid morphs so we can calculate %-correct:
                    curr_preds = test_by_config[tested_config]['predicted'][excluding_midmorph_ixs]
                    curr_trues = [0 if tl < m50 else m100 for tl in true_labels_all[excluding_midmorph_ixs]]
                    ncorrect = sum([p==t for p, t in zip(curr_preds, curr_trues)])
                    ntotal = len(excluding_midmorph_ixs)
                    pcorrect = float(ncorrect) / float(ntotal)
            else:
                ncorrect = sum([p==t for p, t in zip(test_by_config[tested_config]['predicted'], test_by_config[tested_config]['true'])])
                ntotal = len(test_by_config[tested_config]['indices'])
                pcorrect = float(ncorrect) / float(ntotal)
                
            if not np.isnan(ntotal):
                print "%s: Morph %i - %i, %i (%.2f)" % (tested_config, sdf.loc[tested_config]['morphlevel'], ncorrect, ntotal, pcorrect)
            test_results_accuracy[tested_config] = {'ncorrect': ncorrect, 'ntotal': ntotal, 'percent_correct': pcorrect}
            
        return test_results_accuracy
    
        
    
#%%
    
#
#    
#
#    if data_type == 'frames':
#        cX_std = StandardScaler().fit_transform(cX)
#        epochs, decode_dict, bins, clfparams['binsize'] = format_epoch_dataset(clfparams, cX, cy, run_info, sconfigs)
#        class_labels = sorted(list(set([sconfigs[c][clfparams['class_name']] for c in sconfigs.keys()])))
#
#        svc = decode_trial_epochs(clfparams, bins, decode_dict, run_info, data_identifier=data_identifier, 
#                            niterations=10, scoring='accuracy')
#    
#    
#    #%
#    # -----------------------------------------------------------------------------
#    # Save classifier and formatted data:
#    # -----------------------------------------------------------------------------
#        
#    clf_fpath = os.path.join(classifier_dir, '%s_datasets.npz' % classif_identifier)
#    np.savez(clf_fpath, cX=cX, cX_std=cX_std, cy=cy,
#                 data_type=data_type,
#                 inputdata=inputdata,
#                 inputdata_type=inputdata_type,
#                 data_fpath=data_fpath,
#                 sconfigs=sconfigs, run_info=run_info)
#
#    joblib.dump(svc, os.path.join(classifier_dir, '%s.pkl' % classif_identifier), compress=9)
#     
#    svc_params = svc.get_params().copy()
#    if 'base_estimator' in svc_params.keys():
#        svc_params['base_estimator'] = str(svc_params['base_estimator'] )
#    #clf_params['cv'] = str(clf_params['cv'])
#    svc_params['identifier'] = classif_identifier
#    svc_params_hash = hash(json.dumps(svc_params, sort_keys=True, ensure_ascii=True)) % ((sys.maxsize + 1) * 2) #[0:6]
#    
#    with open(os.path.join(classifier_dir, 'params_%s.json' % svc_params_hash), 'w') as f:
#        json.dump(svc_params, f, indent=4, sort_keys=True, ensure_ascii=True)
#    
#    clfparams['classifier_info'] = svc_params
#    
#    with open(os.path.join(classifier_dir, 'classifier_params.json'), 'w') as f:
#        json.dump(clfparams, f, indent=4, sort_keys=True, ensure_ascii=True)
#        
#        
#    #%%
#    if clfparams['data_type'] == 'stat':
#  
#        # Visualize feature weights:
#        # =============================================================================
#            
#        # svc.coef_ :  array shape [n_classes, n_features] (if n_classes=2, shape [n_features])
#        # svc.coef_ :  array shape [n_classes, n_features] (if n_classes=2, shape [n_features])
#    
#        # Sort the weights by their strength, take out bottom N rois, iterate.
#        fig = plot_weight_matrix(svc, absolute_value=True)
#        label_figure(fig, data_identifier)
#        pl.savefig(os.path.join(classifier_dir, 'figures', 'sorted_weights_abs.png'))
#        pl.close()
#        
#        fig = plot_weight_matrix(svc, absolute_value=False)
#        label_figure(fig, data_identifier)
#        pl.savefig(os.path.join(classifier_dir, 'figures', 'sorted_weights_raw.png'))
#        pl.close()
#        
#        nrois = len(run_info['roi_list'])
#        for class_idx in range(len(svc.classes_)):
#            fig = plot_coefficients(svc, xrange(nrois), class_idx=class_idx, top_features=20)
#            label_figure(fig, data_identifier)
#            pl.savefig(os.path.join(classifier_dir, 'figures', 'sorted_feature_weights_%s.png' % class_idx))
#            pl.close()
        
        #%

    #%%
        
#    # #############################################################################
#    # Visualize the data
#    # #############################################################################
#    
#        
#    # Look at a few neurons to see what the data looks like for binary classif:
#    ddf = pd.DataFrame(data=cX_std,
#                          columns=range(nrois))
#    labels_df = pd.DataFrame(data=cy,
#                             columns=['class'])
#    
#    df = pd.concat([ddf, labels_df], axis=1).reset_index(drop=True)
#    
#    _ = sns.pairplot(df[:50], vars=[8, 11, 12, 14, 19], hue="class", size=1.5)
#    
#    
#    
#    # How correlated are the features (neurons) to class labels?  
#    # -----------------------------------------------------------------------------
#    
#    corr = ddf.corr()
#    
#    # Generate a mask for the upper triangle
#    mask = np.zeros_like(corr, dtype=np.bool)
#    mask[np.triu_indices_from(mask)] = True
#    
#    # Set up the matplotlib figure
#    f, ax = pl.subplots(figsize=(11, 9))
#    
#    # Generate a custom diverging colormap
#    cmap = sns.diverging_palette(220, 10, as_cmap=True)
#    
#    # Draw the heatmap with the mask and correct aspect ratio
#    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
#                square=True, linewidths=.5, cbar_kws={"shrink": .5})
#    
#    
#      
#    #%%
#    
#    # #############################################################################
#    # Learning curves
#    # #############################################################################
#    
#    
#    # Modified from http://scikit-learn.org/stable/auto_examples/plot_learning_curve.html
#    from sklearn.learning_curve import learning_curve
#    def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
#                            train_sizes=np.linspace(.1, 1.0, 5)):
#        """
#        Generate a simple plot of the test and traning learning curve.
#    
#        Parameters
#        ----------
#        estimator : object type that implements the "fit" and "predict" methods
#            An object of that type which is cloned for each validation.
#    
#        title : string
#            Title for the chart.
#    
#        X : array-like, shape (n_samples, n_features)
#            Training vector, where n_samples is the number of samples and
#            n_features is the number of features.
#    
#        y : array-like, shape (n_samples) or (n_samples, n_features), optional
#            Target relative to X for classification or regression;
#            None for unsupervised learning.
#    
#        ylim : tuple, shape (ymin, ymax), optional
#            Defines minimum and maximum yvalues plotted.
#    
#        cv : integer, cross-validation generator, optional
#            If an integer is passed, it is the number of folds (defaults to 3).
#            Specific cross-validation objects can be passed, see
#            sklearn.cross_validation module for the list of possible objects
#        """
#        
#        pl.figure()
#        train_sizes, train_scores, test_scores = learning_curve(
#            estimator, X, y, cv=5, n_jobs=1, train_sizes=train_sizes)
#        train_scores_mean = np.mean(train_scores, axis=1)
#        train_scores_std = np.std(train_scores, axis=1)
#        test_scores_mean = np.mean(test_scores, axis=1)
#        test_scores_std = np.std(test_scores, axis=1)
#    
#        pl.fill_between(train_sizes, train_scores_mean - train_scores_std,
#                         train_scores_mean + train_scores_std, alpha=0.1,
#                         color="r")
#        pl.fill_between(train_sizes, test_scores_mean - test_scores_std,
#                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
#        pl.plot(train_sizes, train_scores_mean, 'o-', color="r", alpha=0.8,
#                 label="Training score")
#        pl.plot(train_sizes, test_scores_mean, 'o-', color="g", alpha=0.8,
#                 label="Cross-validation score")
#    
#        pl.xlabel("Training examples")
#        pl.ylabel("Score")
#        pl.legend(loc="best")
#        pl.grid("on") 
#        if ylim:
#            pl.ylim(ylim)
#        pl.title(title)
#        
#    # 
#    rand_order = random.sample(xrange(len(cy)), len(cy))
#    
#    
#    # 1.  Look at Training score vs. CV score.
#    # -----------------------------------------------------------------------------
#    # Training score is always at max, regardless of N training examples 
#    # --> OVERFITTING.
#    # CV score increases over time?  
#    # Gap between CV and training scores -- big gap ~ high variance!
#    # -----------------------------------------------------------------------------
#    plot_learning_curve(LinearSVC(C=.10), "LinearSVC(C=10.0)",
#                        cX_std, cy, ylim=(0.0, 1.01),
#                        cv = 5)
#    
#    
#    # 2.  Reduce the complexity -- fewer features.
#    # -----------------------------------------------------------------------------
#    # SelectKBest(f_classif, k=2) will select the k=2 best features according to their Anova F-value
#     
#    from sklearn.pipeline import Pipeline
#    from sklearn.feature_selection import SelectKBest, f_classif
#    
#    nfeatures = 10
#    C_val = 1E9
#    plot_learning_curve(Pipeline([("fs", SelectKBest(f_classif, k=nfeatures)),
#                                   ("svc", svc)]),
#                        "SelectKBest(f_classif, k=%i) + LinearSVC(C=%.2f)" % (nfeatures, C_val),
#                        cX_std, cy, ylim=(0.0, 1.0))
#    
#        
#    # 3.  Use better regularization term. 
#    # -----------------------------------------------------------------------------
#    # Increase classifier regularization (smaller C).
#    # Or, select C automatically w/ grid-search.
#    
#    from sklearn.grid_search import GridSearchCV
#    est = GridSearchCV(LinearSVC(), 
#                       param_grid={"C": [0.001, 0.01, 0.1, 1.0, 10.0]})
#    plot_learning_curve(est, "LinearSVC(C=AUTO)", 
#                        cX_std, cy, ylim=(0.0, 1.0))
#    print "Chosen parameter on 100 datapoints: %s" % est.fit(cX_std[:100], cy[:100]).best_params_
#    
#    
#    # 4.  Try L1 penalty for a sparser solution.
#    # -----------------------------------------------------------------------------
#    # With L1 penalty, implicit feature selection. Can look at learned coefficients
#    # to see how many are 0 (ignored feature), and which have the stongest weights.
#    
#    plot_learning_curve(LinearSVC(C=0.1, penalty='l1', dual=False), 
#                        "LinearSVC(C=0.1, penalty='l1')", 
#                        cX_std, cy, ylim=(0.0, 1.0))
#    
#    est = LinearSVC(C=0.1, penalty='l1', dual=False)
#    est.fit(cX_std[:100], cy[:100])  # fit on 100 datapoints
#    print "Coefficients learned: %s" % est.coef_
#    print "Non-zero coefficients: %s" % np.nonzero(est.coef_)[1]
#    
#    strongest_weights = np.where(np.abs(est.coef_) == np.abs(est.coef_).max())
#    print "Best features: %s" % str(strongest_weights)
#    
#    pl.figure()
#    sns.heatmap(est.coef_)

#%%
##%%
## Plot decision function with all data:
#
#svc.classes_
#decisionfunc = svc.decision_function(cX_std) #print cdata.shape
#
#pl.figure()
#g = sns.heatmap(decisionfunc, cmap='PRGn')
#
#indices = { value : [ i for i, v in enumerate(cnames) if v == value ] for value in list(set(cnames)) }
#ytick_indices = [(k, indices[k][-1]) for k in list(set(cnames))]
#g.set(yticks = [])
#g.set(yticks = [x[1] for x in ytick_indices])
#g.set(yticklabels = [x[0] for x in ytick_indices])
#g.set_yticklabels(g.get_yticklabels(), rotation=45, fontsize=8)
#g.set(xlabel = 'class')
#g.set(ylabel = 'trials by label')
#
#pl.title('%s (decision functions)' % data_type)
#
#figname = '%s__decisionfunction.png' % classif_identifier
#
#pl.savefig(os.path.join(classifier_dir, figname))
#
#
##%%
#    
## Project zscores onto normals:
#svc.classes_
#cdata = np.array([svc.coef_[c].dot(cX_std.T) + svc.intercept_[c] for c in range(len(svc.classes_))])
#print cdata.shape
#
#pl.figure()
#g = sns.heatmap(cdata, cmap='PRGn')
#
#indices = { value : [ i for i, v in enumerate(cnames) if v == value ] for value in list(set(cnames)) }
#xtick_indices = [(k, indices[k][-1]) for k in list(set(cnames))]
#g.set(xticks = [])
#g.set(xticks = [x[1] for x in xtick_indices])
#g.set(xticklabels = [x[0] for x in xtick_indices])
#g.set_xticklabels(g.get_xticklabels(), rotation=45, fontsize=8)
#g.set(xlabel = cell_unit)
#g.set(ylabel = 'class')
#
#pl.title('%s data proj onto normals' % data_type)
#
#figname = '%s__proj_normals.png' % classif_identifier
#
#pl.savefig(os.path.join(classifier_dir, figname))

#options = ['-D', '/mnt/odyssey', '-i', 'JC015', '-S', '20180915', '-A', 'FOV1_zoom2p7x',
#           '-R', 'combined_gratings_static', '-t', 'traces001',
#           '-r', 'visual', '-d', 'stat', '-s', 'meanstim',
#           '-p', 'corrected', '-N', 'ori',
#           '-c', 'xpos'
#           ]
#
#
#def train_test_linearSVC(options):
#
#    optsE = extract_options(options)
#        
#    C = LinearSVM(optsE)
#    C.get_training_data()
#    C.label_training_data()
#    C.initialize_classifier()
#    
#    if C.clfparams['data_type'] == 'frames':
#        C.train_on_trial_epochs()
#    else:
#        C.cv_kfold_permutation(scoring='accuracy', permutation_test=True, n_permutations=500)
#        C.confusion_matrix()
#        C.do_RFE(scoring='accuracy')
#    
#    
##%%
#
#def main(options):
#    train_test_linearSVC(options)
#
#
#
#if __name__ == '__main__':
#    main(sys.argv[1:])
