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

from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold, cross_val_score
from sklearn import svm
from sklearn import cross_validation
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

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
    datakey = run_info['datakey']
    if '/' in datakey:
        datakey = datakey[1:]
    sort_dir = os.path.join(traceid_dir, 'sorted_%s' % datakey)
    
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


def label_classifier_data(cX, cy, class_name, sconfigs, aggregate_type='each', const_trans=None, trans_value=None, data_type='zscores'):
    sconfigs_df = pd.DataFrame(sconfigs).T

    # Check that input data is correctly formatted:
    assert len(cy)==cX.shape[0], "Incorrect input data sizes. Data (%s), labels (%s)" % (str(cX.shape), str(cy.shape))
    assert sorted(list(set(cy))) == sorted(sconfigs.keys()), "Bad configIDs for label list. Should be the same as sconfigs.keys()."
    
    if aggregate_type == 'each':
        # Assign labels, but don't group/average values by some shared transform value. 
        if const_trans is None:
            stim_grouper = [class_name]
            other_trans = list(set([k for k in sconfigs['config001'].keys() if not k == class_name]))
            stim_grouper.extend(other_trans)   
        else:
            stim_grouper = [class_name, const_trans]
    
        sgroups = sconfigs_df.groupby(stim_grouper)                                               # Group stimconfigs with top-level as class-label type
        ordered_configs = {key: i for i, key in enumerate([str(g.index[0]) for k,g in sgroups])}  # Create a mapper for sorting the original configID list
        sorted_ixs = [s[0] for s in sorted(enumerate(cy), key=lambda d: ordered_configs[d[1]])]   # Get indices with which to sort original configID list

        # check that indices are grouped properly:
        if data_type == 'xcondsub':
            sidx = 0
            diffs = []
            for k in sorted(ordered_configs.keys(), key= lambda d: ordered_configs[d]):
                diffs.append(len(list(set(np.diff(sorted_ixs[sidx:sidx+nframes_per_trial])))))
                sidx += nframes_per_trial
            print diffs
            assert len(list(set(diffs)))==1 and list(set(diffs))[0]==1, "Incorrect index grouping!"

        # Re-sort class labels and data:
        cy_tmp = cy[sorted_ixs]
        cX_tmp = cX[sorted_ixs, :]
            
    elif aggregate_type == 'average':
        # Average values for all samples with the same class label (class_name).
        # If const_trans provided, average classes only across that transform type.
        stim_grouper = [class_name, const_trans]
        if const_trans is None:
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
        
        sgroups = sconfigs_included.groupby(stim_grouper)
        cy_tmp = []; cX_tmp = []
        for k, g in sgroups:
            curr_cys = [ci for ci,cv in enumerate(cy) if cv in g.index.tolist()]
            cy_tmp.append(cy[curr_cys])
            cX_tmp.append(cX[curr_cys, :])
    
        # Re-sort class labels and data:
        cy_tmp = np.hstack(cy_tmp)
        cX_tmp = np.vstack(cX_tmp)
        
    cy = np.array([sconfigs[cv][class_name] for cv in cy_tmp])
    cX = cX_tmp.copy()
    
    return cX, cy

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
#
#
##
##options_list = [opts1, opts2, opts3]

#opts1 = ['-D', '/mnt/odyssey', '-i', 'CE077', '-S', '20180518', '-A', 'FOV1_zoom1x',
#           '-T', 'np_subtracted', '--no-pupil',
#           '-R', 'gratings_run1', '-t', 'traces002',
#           '-n', '1']

opts2 = ['-D', '/mnt/odyssey', '-i', 'CE077', '-S', '20180518', '-A', 'FOV1_zoom1x',
           '-T', 'np_subtracted', '--no-pupil',
           '-R', 'blobs_dynamic_run3', '-t', 'traces002',
           '-n', '1']
options_list = [opts2]

opts1 = ['-D', '/mnt/odyssey', '-i', 'CE077', '-S', '20180518', '-A', 'FOV1_zoom1x',
           '-T', 'np_subtracted', '--no-pupil',
           '-R', 'blobs_run6', '-t', 'traces002',
           '-n', '1']


opts1 = ['-D', '/mnt/odyssey', '-i', 'CE077', '-S', '20180518', '-A', 'FOV1_zoom1x',
           '-T', 'np_subtracted', '--no-pupil',
           '-R', 'blobs_run2', '-t', 'traces002',
           '-n', '1']

options_list = [opts1]


opts1 = ['-D', '/mnt/odyssey', '-i', 'CE077', '-S', '20180521', '-A', 'FOV2_zoom1x',
           '-T', 'np_subtracted', '--no-pupil',
           '-R', 'blobs_run2', '-t', 'traces001',
           '-n', '1']

opts2 = ['-D', '/mnt/odyssey', '-i', 'CE077', '-S', '20180521', '-A', 'FOV2_zoom1x',
           '-T', 'np_subtracted', '--no-pupil',
           '-R', 'blobs_run1', '-t', 'traces001',
           '-n', '1']

opts3 = ['-D', '/mnt/odyssey', '-i', 'CE077', '-S', '20180521', '-A', 'FOV2_zoom1x',
           '-T', 'np_subtracted', '--no-pupil',
           '-R', 'blobs_dynamic_run1', '-t', 'traces001',
           '-n', '1']

options_list = [opts3]

#
#opts1 = ['-D', '/mnt/odyssey', '-i', 'CE077', '-S', '20180521', '-A', 'FOV2_zoom1x',
#           '-T', 'np_subtracted', '--no-pupil',
#           '-R', 'gratings_run1', '-t', 'traces001',
#           '-n', '1']
#options_list = [opts1]


test = False

#load_pretrained = True

#%%

# Load data for training classifier:
#training_type = 'xpos'
#
#if load_pretrained:
#    if training_type == 'yrot':
#        output_basedir = '/mnt/odyssey/CE077/20180518/FOV1_zoom1x/blobs_run6/traces/traces002_b55f3a/figures/population'
#    elif training_type == 'xpos':
#        output_basedir = '/mnt/odyssey/CE077/20180518/FOV1_zoom1x/blobs_run2/traces/traces002_66774c/figures/population'
#        
#    data_fpath = os.path.join(output_basedir, 'datasets.npz')
#    training_dataset = np.load(data_fpath)
#    
#    X = training_dataset['smoothedX']
#    y = training_dataset['y']
#    sDATA = training_dataset['sDATA']
#    run_info = training_dataset['run_info']
#    
#    #zscores = training_dataset['zscores']
#    
#    
##    tX_std = training_dataset['cX_std']
##    ty = training_dataset['cy']
#    training_labels = training_dataset['conditions']
#    
#    visual_rids = training_dataset['visual_rids']
#    
#    print training_labels
#
#if load_testset:
#    output_basedir = '/mnt/odyssey/CE077/20180518/FOV1_zoom1x/blobs_dynamic_run3/traces/traces002_43fc94/figures/population'
#    data_fpath = os.path.join(output_basedir, 'datasets.npz')
#    testing_dataset = np.load(data_fpath)
#    
#    tX_std = testing_dataset['cX_std']
#    ty = testing_dataset['cy']
#    training_labels = testing_dataset['conditions']
#    
#    visual_rids = testing_dataset['visual_rids']
#    
#    print training_labels
#    
#%%
averages_df = []
normed_df= []
all_zscores = {}
all_info = {}

#%%

#options_idx = 1
#options_idx = 0

test = True
for options_idx in range(len(options_list)):
    #%
    print "**********************************"
    print "Processing %i of %i runs." % (options_idx, len(options_list))
    print "**********************************"

    #%
    options = options_list[options_idx]
    traceid_dir = util.get_traceid_dir(options)
    
    # Set up output dir:
    output_basedir = os.path.join(traceid_dir, 'figures', 'population')
    if not os.path.exists(output_basedir):
        os.makedirs(output_basedir)

    # First check if processed datafile exists:
    reload_data = False
    data_fpath = os.path.join(output_basedir, 'datasets.npz')
    if os.path.exists(data_fpath):
        try:
            dataset = np.load(data_fpath)
            print "Loaded existing datafile:\n%s" % data_fpath
            print dataset.keys()
    
            sconfigs = dataset['sconfigs'][()]
            roi_list = dataset['run_info'][()]['roi_list']
            zscores = dataset['zscores']
            X = dataset['smoothedX']
            y = dataset['y']
            tsecs = dataset['tsecs']
        except Exception as e:
            reload_data = True

    if reload_data:
            
        sDATA, run_info, stimconfigs = util.get_run_details(options)
        sconfigs = util.format_stimconfigs(stimconfigs)
        
        # Set up output dir:
        output_basedir = os.path.join(run_info['traceid_dir'], 'figures', 'population')
        if not os.path.exists(output_basedir):
            os.makedirs(output_basedir)
    
        #% Load time-course data and format:
        roi_list = run_info['roi_list']
        Xdata, ylabels, groups, tsecs, roi_list = util.format_framesXrois(sDATA, roi_list, run_info['nframes_on'], run_info['framerate'], missing='none')
    
        nframes_per_trial = run_info['nframes_per_trial']
    
        #% Test smoothing on a few ROIs:
        if test:
            ridx = 0 #22 #18 #162 #3 #162
            condlabel = [c for c in sconfigs if sconfigs[c]['morphlevel']==0 and sconfigs[c]['xpos']==0][0]
            #condlabel = 135 # 'config026' # #90 #'config014' #225 #180
            cidx = int(condlabel[6:])
            ntrials_per_cond = run_info['ntrials_per_cond'][cidx]
            fmin = 0.001
            fmax = 0.005
            util.test_smoothing_fractions(ridx, Xdata, ylabels,
                                     ntrials_per_cond=ntrials_per_cond,
                                     nframes_per_trial=nframes_per_trial,
                                     condlabel=condlabel, fmin=fmin, fmax=fmax,
                                     missing='drop')
    
            pl.suptitle('roi%05d_condition%s' % (ridx+1, str(condlabel)))
            pl.savefig(os.path.join(output_basedir, 'smoothing_results_roi%05d.png' % int(ridx+1)))
    
        #% SMOOTH data, since frame rate is very high (44.75 Hz)
        frac = 0.001
        smooth = True
        if smooth:
            X = np.apply_along_axis(util.smooth_traces, 0, Xdata, frac=frac, missing='drop')
            y = ylabels.copy()
            print X.shape
    
    
        #%  Preprocessing, step 1:  Remove cross-condition mean
        # Select color code for conditions:
        conditions = run_info['condition_list']
        color_codes = sns.color_palette("Greys_r", len(conditions)*2)
        color_codes = color_codes[0::2]
        
        #% Look at example ROI:
        if test:
            #ridx = 134 # 15 #44 #3 # 'roi00004' #10
            mean_cond_traces, mean_tsecs = util.get_mean_cond_traces(ridx, X, y, tsecs, nframes_per_trial)
            xcond_mean = np.mean(mean_cond_traces, axis=0)
    
            pl.figure()
            pl.subplot(1,2,1); pl.title('traces')
            for t in range(len(conditions)):
                pl.plot(mean_tsecs, mean_cond_traces[t, :], c=color_codes[t])
            pl.plot(mean_tsecs, np.mean(mean_cond_traces, axis=0), 'r')
    
            pl.subplot(1,2,2); pl.title('xcond subtracted')
            normed = (mean_cond_traces - xcond_mean)
            for t in range(len(conditions)):
                pl.plot(mean_tsecs, normed[t,:], c=color_codes[t])
            pl.plot(mean_tsecs, np.mean(normed, axis=0), 'r')
            pl.suptitle('average df/f for each condition, with and w/out xcond norm')
    
            figname = 'avgtrial_vs_xcondsubtracted_roi%05d.png' % int(ridx+1)
            pl.savefig(os.path.join(output_basedir, figname))
    
        tvalues = util.format_roisXvalue(sDATA, run_info, value_type='meanstimdf')
        zscores = util.format_roisXvalue(sDATA, run_info, value_type='zscore')

        # Save:
        print "Saving processed data..."
        np.savez(data_fpath, 
                 sDATA=sDATA,
                 X=X, y=y,
                 smoothedX=X, frac=frac,
                 tsecs=tsecs,
                 groups=groups,
                 sconfigs=sconfigs, 
                 tvalues=tvalues, 
                 zscores=zscores,
                 run_info=run_info)


    averages_list, normed_list = util.get_xcond_dfs(roi_list, X, y, tsecs, run_info)

    #if options_idx == 0:
    averages_df.extend(averages_list)
    normed_df.extend(normed_list)

    # Get zscores
    if len(options_list) > 1:
        all_zscores[options_idx] = zscores
        all_info[options_idx] = run_info

#%
#% Concat into datagrame
avgDF = pd.concat(averages_df, axis=1)
avgDF.head()

normDF = pd.concat(normed_df, axis=1)
normDF.head()


#%%
# =============================================================================
# Load ROI masks to sort ROIs by spatial distance
# =============================================================================

#options_idx = 0
#
#single_basedir = os.path.join(info[options_idx]['traceid_dir'], 'figures', 'population')
#run_dir = info[options_idx]['traceid_dir'].split('/traces')[0]
#acquisition = os.path.split(os.path.split(run_dir)[0])[1]
#
#output_basedir = single_basedir
#traceid_dir = info[options_idx]['traceid_dir']
#
#
#data_fpath = os.path.join(output_basedir, 'datasets.npz')
#np.savez(data_fpath, 
#             normDF=normDF, 
#             avgDF=avgDF,
#             smoothedX=X, frac=frac, y=y,
#             sconfigs=sconfigs, 
#             zscores=zscores, 
#             run_info=run_info)


#%%
spatial_sort = False

if spatial_sort:
    sorted_rids, cnts, zproj = util.sort_rois_2D(traceid_dir)
    util.plot_roi_contours(zproj, sorted_rids, cnts)
    figname = 'spatially_sorted_rois_%s.png' % acquisition
    pl.savefig(os.path.join(output_basedir, figname))
    

#%%
# =============================================================================
# Assign cX -- input data for classifier (nsamples x nfeatures)
# =============================================================================

metric = 'meanstimdf'
roi_selector = 'all' #'selectiveanova' #'selective'
data_type = 'zscore' #zscore' # 'xcondsub'
is_test = False

options_idx = 0

# -----------------------------------------------------------------------------

if data_type == 'zscore':
    cX = zscores.copy().T
elif data_type == 'xcondsub':
    cX = normDF.values
elif data_type == 'avgconds':
    cX = avgDF.values

# Use subset of ROIs, e.g., VISUAL or SELECTIVE only:
if roi_selector == 'visual':
    visual_rids = get_roi_list(run_info, roi_selector=roi_selector, metric=metric)
else:
    visual_rids = xrange(cX.shape[1])
    
if is_test:
    tX = normDF.values
    tX = tX[:, visual_rids]
    print tX.shape

cX = cX[:, visual_rids]
print cX.shape

#spatial_rids = [s for s in sorted_rids if s in visual_rids]

#%%

nframes_per_trial = run_info['nframes_per_trial']
ntrials_by_cond = run_info['ntrials_per_cond']


# Get default label list for cX (using the values originally assigned to conditions):
if isinstance(run_info['condition_list'], str):
    cond_labels_all = sorted(run_info['condition_list'], key=natural_keys)
else:
    cond_labels_all = sorted(run_info['condition_list'])

if data_type == 'zscore':
    # Each sample is a TRIAL, total nsamples = ntrials_per_condition * nconditions.
    # Different conditions might have different num of trials, so assign by n trials:
    cy = np.hstack([np.tile(cond, (nt,)) for cond, nt in zip(cond_labels_all, ntrials_by_cond)]) #config_labels.copy()

elif data_type == 'xcondsub':
    # Each sample point is a FRAME, total nsamples = nframe_per_trial * nconditions
    
    cy = np.hstack([np.tile(cond, (nframes_per_trial,)) for cond in sorted(cond_labels_all, key=natural_keys)]) #config_labels.copy()
    
print 'cX (zscores):', cX.shape
print 'cY (labels):', cy.shape

#%%

# =============================================================================
# Assign cy -- labels for classifier to learn (nsamples,)
# =============================================================================
#if 'ori' in sconfigs['config001'].keys():
#    class_type = 'ori'
#    class_views = 'all'
#    trans_types = ['ori']
#else:
#    class_type = 'object' #'object' #'views'
#    trans_types = run_info['trans_types']
#    class_name = 'morphlevel' #'morphlevel'
#
#    # Set these if more than 1 transformation varying, and hold 1 constant:
#    const_trans = None #'morphlevel' #'xpos'  # transform type to hold constant
#    class_views = 'yrot' # 'morphlevel' #'%s-5' % const_trans
#
## -----------------------------------------------------------------------------
#
##class_desc = '%s_%s_%s' % (class_type, class_name, data_type)
#constant_trans_types = [t for t in trans_types if not t==class_name]
#if const_trans is None and len(constant_trans_types)==1:
#    const_trans = constant_trans_types[0]
#
#print "Found %i other varying transforms:" % len(constant_trans_types), constant_trans_types

# Get dimensions of session dataset:
nframes_per_trial = run_info['nframes_per_trial']
nconditions = len(run_info['condition_list'])
nrois = cX.shape[1]

# Group configIDs by selected class labels to sort labels in order:
#sconfigs_df = pd.DataFrame(sconfigs).T

class_name = 'morphlevel'
aggregate_type = 'each'
const_trans = None
trans_value = None

class_desc = '%s_%s_%s' % (class_name, aggregate_type, data_type)

cX, cy = label_classifier_data(cX, cy, class_name, sconfigs, aggregate_type=aggregate_type)

#%%
    
    #elif len(class_views.split('_')) > 1:
if data_type == 'xcondsub':

    if len(class_views.split('_')) > 1:
        view_to_average = class_views.split('_')[-1]
        varying_transforms = [t for t in constant_trans_types if not t==view_to_average]
        grouper = [class_name]
        grouper.extend(varying_transforms)
        sgroups = sconfigs_df.groupby(grouper)
    
        # For each object type, get average across the transform specified by view_to_avearge:
        cX_tmp = []
        cy_tmp = []
        for k,g in sgroups:
            curr_frames = np.array([ix for ix,iy in enumerate(cy) if iy in g.index.tolist()])
            ntrials_combined = len(curr_frames) / nframes_per_trial
            frame_ixs = np.reshape(curr_frames, (ntrials_combined, nframes_per_trial))
            vals_to_average = np.array([cX[frame_ixs[t],:] for t in range(ntrials_combined)])
            averaged = np.mean(vals_to_average, axis=0)
            cX_tmp.append(averaged)
            cy_tmp.append(np.tile(k[0], (averaged.shape[0],)))  # THIS ASSIGNS LABEL TO TOP-LEVEL grouper (object id, i.e., "class_name")
            
        cX = np.vstack(cX_tmp)
        cy = np.hstack(cy_tmp)
        
        print "New dims: cX %s, cy %s" % (str(cX.shape), str(cy.shape))
        
    
    elif any([t in class_views for t in constant_trans_types]) and const_trans is not None:
        # Only include sepcified value of a transform:
        const_trans_val = int(class_views.split(const_trans)[-1])
        # Identify the configs that meet the trans-value criteria:
        configs_to_include = [c for c in conditions if sconfigs[c][const_trans]==const_trans_val]
        sample_idxs = [ci for ci,cv in enumerate(cy) if cv in configs_to_include]
        
        # Use sorted cy labels to extract cX data:
        cy_tmp = cy[sample_idxs]
        cX_tmp = cX[sample_idxs,:]
        
        # Replace cy labels with object names:
        cy_tmp = np.array([sconfigs[cv][class_name] for cv in cy_tmp])
        
        # Sort labels:
        sorted_idxs = np.argsort(cy_tmp)
        cy = cy_tmp[sorted_idxs]
        
        cX = cX_tmp[sorted_idxs,:] 
             
        print "New dims: cX %s, cy %s" % (str(cX.shape), str(cy.shape))

print 'cX (%s):' % data_type, cX.shape
print 'cY (labels):', cy.shape

class_labels = sorted(list(set(cy)))


#%%

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
s#ns.heatmap(1-corrs, cmap=zdf_cmap, vmin=0.7, vmax=1.3)

pl.title('RDM (%s df) - %s: %s' % (data_type, class_name, aggregate_type))
figname = 'RDM_%srois_%s_classes_%i%s_%s.png' % (roi_selector, data_type, len(class_labels), class_name, aggregate_type)

pl.savefig(os.path.join(output_basedir, figname))


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
pl.savefig(os.path.join(output_basedir, figname))


#%% Create input data for classifier:

# -----------------------------------------------------------------------------
# Whitening:
# -----------------------------------------------------------------------------

from sklearn.preprocessing import StandardScaler

cX_std = StandardScaler(with_std=True).fit_transform(cX)
print cX_std.shape

pl.figure();
sns.distplot(cX.ravel(), label='zscores')
sns.distplot(cX_std.ravel(), label = 'standardized')
pl.legend()
pl.title('distN of combined ROIs for FOV1, FOV2')
pl.savefig(os.path.join(output_basedir, '%s_roi_zscore_distributions.png' % roi_selector))

if len(options_list) > 1:
    pl.figure();
    sns.distplot(cX1.ravel(), label = 'FOV1')
    sns.distplot(cX2.ravel(), label = 'FOV2')
    sns.distplot(cX3.ravel(), label = 'FOV3')
    sns.distplot(cX_std.ravel(), label = 'standardized')
    pl.legend()
    pl.title('zscore distN for combined FOVs')
    pl.savefig(os.path.join(output_basedir, 'roi_zscore_distributions_ALL_FOVs.png'))
  
#%% Train and test linear SVM using zscored response values:

# -----------------------------------------------------------------------------
# Set classifier type and CV params:
# -----------------------------------------------------------------------------

classifier = 'LinearSVC' #'LinearSVC' # 'OneVRest'
#sample_type = 'zscores' #'frames' #'zscores'

if data_type == 'zscore':
    cv_method = 'LOO' # "LOGO" "LPGO"
elif data_type == 'xcondsub':
    cv_method = 'LOGO' # "LOGO" "LPGO"

cv_ngroups = 1

find_C = False

#% Create output dir for current classifier:
classif_identifier = '%s_%s_%s_%s_%igroups_classes_%i%s' %  (roi_selector, classifier, data_type, cv_method, cv_ngroups, nconditions, class_desc)
print classif_identifier

output_dir = os.path.join(output_basedir, classif_identifier)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
else:
    print "Classifier dir exists!  Overwrite?"

#%%
# Create classifier:
# ------------------
#if X_train.shape[0] > X_train.shape[1]: # nsamples > nfeatures
#    dual = False
    
big_C = 1e9

from sklearn.linear_model import SGDClassifier, LogisticRegression

if classifier == 'LinearSVC':
    svc = LinearSVC(random_state=0, dual=False, multi_class='ovr', C=big_C)
    
elif classifier == 'SGD':
    svc = SGDClassifier(loss='log', penalty='l1', max_iter=1000)
    
elif classifier == 'LogReg':
    svc = LogisticRegression(multi_class='ovr', solver='liblinear', C=big_C)
    
else:
    svc = OneVsRestClassifier(SVC(kernel='linear', decision_function_shape='ovr'))

#import statsmodels.discrete.discrete_model as sm
#logit = sm.Logit(cy, cX_std)
#logit.fit().params

if find_C:
    best_C = get_best_C(svc, cX_std, cy, output_dir=output_dir, classifier_str=classif_identifier)
    C_val = best_C
else:
    C_val = big_C
    
svc = LinearSVC(random_state=0, dual=False, multi_class='ovr', C=C_val)
    

#%%
# -----------------------------------------------------------------------------
# Format cross-validation data
# -----------------------------------------------------------------------------

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
    loo = cross_validation.StratifiedKFold(cy, n_folds=5)
    pred_results = []
    pred_true = []
    for train, test in loo: #, groups=groups):
        print train, test
        X_train, X_test = training_data[train], training_data[test]
        y_train, y_test = cy[train], cy[test]
        y_pred = svc.fit(X_train, y_train).predict(X_test)
        pred_results.append(y_pred) #=y_test])
        pred_true.append(y_test)

    # Find "best fold"?
    avg_scores = []
    for y_pred, y_test in zip(pred_results, pred_true):
        pred_score = len([i for i in y_pred==y_test if i]) / float(len(y_pred))
        avg_scores.append(pred_score)
    best_fold = avg_scores.index(np.max(avg_scores))
    folds = [i for i in enumerate(loo)]
    train = folds[best_fold][1][0]
    test = folds[best_fold][1][1]
    X_train, X_test = training_data[train], training_data[test]
    y_train, y_test = cy[train], cy[test]
    y_pred = pred_results[best_fold]

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

if groups is not None:
    if average_iters:
        cmatrix_tframes = confusion_matrix(pred_true[0], pred_results[0], labels=class_labels)
        for iter_idx in range(len(pred_results))[1:]:
            print "adding iter %i" % iter_idx
            cmatrix_tframes += confusion_matrix(pred_true[iter_idx], pred_results[iter_idx], labels=class_labels)
        conf_mat_str = 'AVG'
        cmatrix_tframes /= len(pred_results)
    else:
        cmatrix_tframes = confusion_matrix(pred_true[best_fold], pred_results[best_fold], labels=class_labels)
        conf_mat_str = 'best'
else:
    cmatrix_tframes = confusion_matrix(y_test, y_pred, labels=class_labels)
    conf_mat_str = 'trials'

fig = pl.figure(figsize=(10,4))
ax1 = fig.add_subplot(1,2,1)
plot_confusion_matrix(cmatrix_tframes, classes=class_labels, ax=ax1, normalize=False,
                  title='Confusion matrix (%s, %s)' % (conf_mat_str, cv_method))

ax2 = fig.add_subplot(1,2,2)
plot_confusion_matrix(cmatrix_tframes, classes=class_labels, ax=ax2, normalize=True,
                      title='Normalized')

figname = '%s__confusion_%s_iters.png' % (classif_identifier, conf_mat_str)
pl.savefig(os.path.join(output_dir, figname))


#%

# Save, cuz takes forever:
cv_outfile = '%s__CV_report.txt' % classif_identifier

f = open(os.path.join(output_dir, cv_outfile), 'w')
f.write(metrics.classification_report(y_test, y_pred, target_names=[str(c) for c in class_labels]))
f.close()

cv_results = {'predicted': list(y_pred),
              'true': list(y_test),
              'classifier': classifier,
              'cv_method': cv_method,
              'ngroups': cv_ngroups,
              'nfolds': 1
              }
cv_resultsfile = '%s__CV_results.json' % classif_identifier
with open(os.path.join(output_dir, cv_resultsfile), 'w') as f:
    json.dump(cv_results, f, sort_keys=True, indent=4)

#%% 
    
# svc.coef_ :  array of shape [n_classes, n_features] (if n_classes=2, shape [n_features])
# svc.coef_ :  array of shape [n_classes, n_features] (if n_classes=2, shape [n_features])

def print_top10(vectorizer, clf, class_labels):
    """Prints features with the highest coefficient values, per class"""
    feature_names = vectorizer.get_feature_names()
    for i, class_label in enumerate(class_labels):
        top10 = np.argsort(clf.coef_[i])[-10:]
        print("%s: %s" % (class_label,
              " ".join(feature_names[j] for j in top10)))

def plot_coefficients(classifier, feature_names, top_features=20):
    coef = classifier.coef_.ravel()
    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    # create plot
    pl.figure(figsize=(15, 5))
    colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
    pl.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
    feature_names = np.array(feature_names)
    pl.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha='right')
 #plt.show()

plot_coefficients(svc, svc.classes_)


#%%
    
# Project zscores onto normals:
svc.classes_
cdata = np.array([svc.coef_[c].dot(training_data.T) + svc.intercept_[c] for c in range(len(svc.classes_))])
print cdata.shape

pl.figure()
g = sns.heatmap(cdata, cmap='hot')

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

pl.savefig(os.path.join(output_dir, figname))

#%%



# PCA:  reduce zscores projected onto normals
# -----------------------------------------------------------------------------

from sklearn.decomposition import PCA

#pX = cdata.T # (n_samples, n_features)
pca_data = 'projdata' #'projdata' # 'inputdata'

if pca_data == 'inputdata':
    pX = cX_std.copy()
    py = cy.copy()

elif pca_data == 'projdata':
    pX = cdata.T #copy()
    py = cy.copy() #svc.classes_
    
ncomps = 2

pca = PCA(n_components=ncomps)
principal_components = pca.fit_transform(pX)

pca_df = pd.DataFrame(data=principal_components,
                      columns=['pc%i' % int(i+1) for i in range(ncomps)])
pca_df.shape

labels_df = pd.DataFrame(data=py,
                         columns=['target'])

pdf = pd.concat([pca_df, labels_df], axis=1).reset_index()

stimtype = sconfigs['config001']['stimtype']
if class_name == 'ori':
    curr_colors = sns.color_palette("hls", len(class_labels))
else:
    curr_colors = sns.color_palette("RdBu_r", len(class_labels))

    
#colors = ['r', 'orange', 'y', 'g', 'b', 'c', 'm', 'k']

# Visualize 2D projection:
fig = pl.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('PCA-reduced k-D neural state for each tpoint', fontsize = 20)
#for target, color in zip(orientations,colors):
for target, color in zip(class_labels, curr_colors):

    indicesToKeep = pdf['target'] == target
    ax.scatter(pdf.loc[indicesToKeep, 'pc1'],
               pdf.loc[indicesToKeep, 'pc2'],
               c = color,
               s = 50,
               alpha=0.5)
ax.legend(class_labels)
ax.grid()

pl.title('PCA, %s (%s)' % (pca_data, data_type))

if pca_data == 'inputdata':
    figname = 'PCA_%s_%s.png' % (pca_data, data_type)
    pl.savefig(os.path.join(output_basedir, figname))
elif pca_data == 'projdata':
    figname = '%s__PCA_%s.png' % (classif_identifier, pca_data)
    pl.savefig(os.path.join(output_dir, figname))


#%%

# try t-SNE:
# -----------------------------------------------------------------------------

from sklearn.manifold import TSNE
import time

#pX = cdata.T # (n_samples, n_features)
pca_data ='inputdata' # 'inputdata'

if pca_data == 'inputdata':
    pX = cX_std.copy()
    py = cy.copy()

elif pca_data == 'projdata':
    pX = np.array([svc.coef_[c].dot(cX_std.T) + svc.intercept_[c] for c in range(len(svc.classes_))]).T

    py = cy.copy() #svc.classes_
    
print pX.shape


if pca_data == 'inputdata':
    tsne_df = pd.DataFrame(data=pX,
                   index=np.arange(0, pX.shape[0]),
                   columns=['r%i' % i for i in range(pX.shape[1])]) #(ori) for ori in orientations])
    
elif pca_data == 'projdata':
    # Reduce PROJECTED data
    tsne_df = pd.DataFrame(data=pX,
                       index=np.arange(0, pX.shape[0]),
                       columns=[str(label) for label in class_labels])

feat_cols = [f for f in tsne_df.columns]

# Visualize:
#target_ids = range(len(digits.target_names))
target_ids = range(len(class_labels))

multi_run = True
nruns = 4

if multi_run is False:
    nruns = 1

perplexity = 1000 #100# 40 #100 #5# 100
niter = 3000 #5000

colors = curr_colors # 'r', 'orange', 'y', 'g', 'c', 'b', 'm', 'k' #, 'purple'

if multi_run:
    fig, axes = pl.subplots(2, nruns/2, figsize=(12,8))
    axes = axes.ravel().tolist()
    for run in range(nruns):

        time_start = time.time()
        tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=niter)
        tsne_results = tsne.fit_transform(tsne_df[feat_cols].values)
        print 't-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start)

        ax = axes[run]
        print run
        for i, c, label in zip(target_ids, colors, class_labels):
            ax.scatter(tsne_results[py == int(label), 0], tsne_results[py == int(label), 1], c=c, label=label, alpha=0.5)
            box = ax.get_position()
            ax.set_position([box.x0 + box.width * 0.01, box.y0 + box.height * 0.02,
                             box.width * 0.98, box.height * 0.98])

else:

    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=niter)
    tsne_results = tsne.fit_transform(tsne_df[feat_cols].values)
    print 't-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start)

    fig, ax = pl.subplots(1, figsize=(6, 6))
    colors = curr_colors # 'r', 'orange', 'y', 'g', 'c', 'b', 'm', 'k' #, 'purple'
    #for i, c, label in zip(target_ids, colors, digits.target_names):
    #    plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], c=c, label=label)
    for i, c, label in zip(target_ids, colors, class_labels):
        pl.scatter(tsne_results[cy == int(label), 0], tsne_results[cy == int(label), 1], c=c, label=label)
        box = ax.get_position()
        ax.set_position([box.x0 + box.width * 0.01, box.y0 + box.height * 0.02,
                         box.width * 0.98, box.height * 0.98])

# Put a legend below current axis
pl.legend(loc=9, bbox_to_anchor=(-0.2, -0.15), ncol=len(class_labels))

if pca_data == 'inputdata':
    pl.suptitle('t-SNE, proj %s (%i-D rois) | px: %i, ni: %i' % (data_type, nrois, perplexity, niter))
    figname = 'tSNE_%s_%irois_orderedT_pplex%i_niter%i_%iruns.png' % (data_type, nrois, perplexity, niter, nruns)
    pl.savefig(os.path.join(output_basedir, figname))

elif pca_data == 'projdata':
    figname = 'tSNE_proj_onto_norm_%s_pplex%i_niter%i_%iruns_%s.png' % (data_type, perplexity, niter, nruns, classif_identifier)

    if data_type == 'xcondsub':
        pl.suptitle('t-SNE, proj norm (xcondsub time-series) | px: %i, ni: %i' %  (perplexity, niter))
    else:
        pl.suptitle('t-SNE: proj norm (zscores) | px: %i, ni: %i' %  (perplexity, niter))
    pl.savefig(os.path.join(output_dir, figname))



#%%


data_fpath = os.path.join(output_basedir, '%s_datasets.npz' % classif_identifier)
np.savez(data_fpath, cX=cX, cX_std=cX_std, cy=cy, c=best_C, 
             visual_rids=visual_rids,
             data_type=data_type,
             sconfigs=sconfigs, zscores=zscores, run_info=run_info)


#%%
    
# Predict labels for each time point for rotation movies:
training_type = 'xpos'

#% Plot probability of class label as function of time:
if load_pretrained:
    if training_type == 'yrot':
        output_basedir = '/mnt/odyssey/CE077/20180518/FOV1_zoom1x/blobs_run6/traces/traces002_b55f3a/figures/population'
    elif training_type == 'xpos':
        output_basedir = '/mnt/odyssey/CE077/20180518/FOV1_zoom1x/blobs_run2/traces/traces002_66774c/figures/population'
        
    data_fpath = os.path.join(output_basedir, 'datasets.npz')
    training_dataset = np.load(data_fpath)
    
    tX_std = training_dataset['cX_std']
    ty = training_dataset['cy']
    training_labels = training_dataset['conditions']
    
    visual_rids = training_dataset['visual_rids']
    
    print training_labels

from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

#clf = CalibratedClassifierCV(lsvm)
#clf.fit(cX_std, y)
#y_proba = clf.predict_proba(X_test)

clf = LogisticRegression(multi_class='ovr', solver='liblinear')
clf.fit(tX_std, ty)
#tX_std = StandardScaler(with_std=True).fit_transform(cX)
 

object_ids = [0, 11, 22]
#curr_obj_id = 22
#trialixs = np.where(cy==curr_obj_id)[0]
##all_indices = cX_std[trialixs,:]
#n_test_trials = len(trialixs) / nframes_per_trial
#test_trials = np.reshape(trialixs, (n_test_trials, nframes_per_trial))

mean_pred = {}
all_pred= {}
for curr_obj_id in object_ids:
    
    trialixs = np.where(cy==curr_obj_id)[0]
    n_test_trials = len(trialixs) / nframes_per_trial
    test_trials = np.reshape(trialixs, (n_test_trials, nframes_per_trial))

    predicted = []
    for ti in range(n_test_trials):
        true_labels = cy[test_trials[ti]]
        test_data = cX[test_trials[ti], :]

        y_proba = clf.predict_proba(test_data)
        y_proba.shape
        predicted.append(y_proba)
    
    mean_pred[curr_obj_id] = np.mean(predicted, axis=0)
    all_pred[curr_obj_id] = np.array(predicted)
    
colorvals = sns.color_palette("PRGn", len(clf.classes_))


sns.set()
# Plot AVERAGE of movie conditions:
fig, axes = pl.subplots(1, len(object_ids), figsize=(12,4), sharey=True)
for pi,obj in enumerate(object_ids):
    for ci in range(mean_pred[obj].shape[1]):
        axes[pi].plot(tpoints, mean_pred[obj][:, ci], color=colorvals[ci], label=clf.classes_[ci], linewidth=2)
    axes[pi].set(title=obj)
    # Shrink current axis's height by 10% on the bottom
    box = axes[pi].get_position()
    axes[pi].set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.8])
    if pi == 0:
        axes[pi].set(ylabel='avg p')
sns.despine(offset=True, trim=True)
# Put a legend below current axis
pl.legend(loc='lower center', bbox_to_anchor=(-1., -.3),
          fancybox=True, shadow=False, ncol=len(clf.classes_))
pl.suptitle('Predicted labels on dynamic (trained: static)')
figname = '%s_train_static_predict_dynamic_avg_trials.png' % classif_identifier
pl.savefig(os.path.join(output_dir, figname))


# Plot each trial of movie conditions:
fig, axes = pl.subplots(1, len(object_ids), figsize=(12,4), sharey=True)
for pi,obj in enumerate(object_ids):
    for ci in range(mean_pred[obj].shape[1]):
        axes[pi].plot(tpoints, all_pred[obj][0, :, ci], color=colorvals[ci], label=clf.classes_[ci], linewidth=2)
        axes[pi].plot(tpoints, all_pred[obj][1, :, ci], color=colorvals[ci], label=clf.classes_[ci], linewidth=2)
    axes[pi].set(title=obj)
    # Shrink current axis's height by 10% on the bottom
    box = axes[pi].get_position()
    axes[pi].set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.8])
    if pi == 0:
        axes[pi].set(ylabel='p')
sns.despine(offset=True, trim=True)
# Put a legend below current axis
pl.legend(loc='lower center', bbox_to_anchor=(-1., -.3),
          fancybox=True, shadow=False, ncol=len(clf.classes_))
pl.suptitle('Predicted labels on dynamic (trained: static)')
figname = '%s_train_static_predict_dynamic_each_trial.png' % classif_identifier
pl.savefig(os.path.join(output_dir, figname))











# Split frames into trials:
n_test_trials = y_proba.shape[0] / nframes_per_trial

fig, ax = pl.subplots(1, figsize=(6,6))
for ci, curr_obj_id in enumerate(sorted(object_ids)):
    trialixs = np.where(ty==curr_obj_id)[0]
    test_trial = tX_std[trialixs,:]
    true_labels = ty[trialixs]
    y_proba = clf.predict_proba(test_trial)

    class_idx = [c for c in clf.classes_].index(curr_obj_id)
    curr_preds = np.array([y_proba[t*nframes_per_trial:t*nframes_per_trial + nframes_per_trial, class_idx] for t in range(n_test_trials)])
    print curr_preds.shape
    avg_pred = np.mean(curr_preds, axis=0)
    print avg_pred.shape
    
    ax.plot(xrange(nframes_per_trial), avg_pred, label=curr_obj_id, color = colorvals[ci])
    
# Put a legend below current axis
pl.legend(loc='lower center', bbox_to_anchor=(-0., -0.3),
          fancybox=True, shadow=False, ncol=len(clf.classes_))
sns.despine(offset=4)


fig, axes = pl.subplots(1, n_test_trials, sharex=True, sharey=True, figsize=(8,4)) #pl.figure()
axes = axes.reshape(-1)
for t in range(n_test_trials):
    ax = axes[t]
    true_label = list(set(true_labels[t*nframes_per_trial:t*nframes_per_trial + nframes_per_trial]))
    for c in range(y_proba.shape[1]):
        ax.plot(y_proba[t*nframes_per_trial:t*nframes_per_trial + nframes_per_trial, c], label=clf.classes_[c],
                    linewidth=3, alpha=0.99, color=colorvals[c])
    ax.set_title('%s' % str(true_label))
    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.8])

# Put a legend below current axis
pl.legend(loc='lower center', bbox_to_anchor=(-1.8, -0.3),
          fancybox=True, shadow=False, ncol=len(clf.classes_))
sns.despine(offset=4)




# Plot predictions for each class:

fig, axes = pl.subplots(1, n_test_trials, sharex=True, sharey=True, figsize=(8,4)) #pl.figure()
axes = axes.reshape(-1)
for t in range(n_test_trials):
    ax = axes[t]
    true_label = list(set(true_labels[t*nframes_per_trial:t*nframes_per_trial + nframes_per_trial]))
    for c in range(y_proba.shape[1]):
        ax.plot(y_proba[t*nframes_per_trial:t*nframes_per_trial + nframes_per_trial, c], label=clf.classes_[c],
                    linewidth=1, alpha=0.8, color=colorvals[c])
    ax.set_title('%s' % str(true_label))
    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.8])

# Put a legend below current axis
pl.legend(loc='lower center', bbox_to_anchor=(-1.8, -0.3),
          fancybox=True, shadow=False, ncol=len(clf.classes_))
sns.despine(offset=4)

pl.suptitle('class prob over time')
figname = 'proba_LPGO_ptrials_sample_frames.png'
pl.savefig(os.path.join(output_dir, figname))

