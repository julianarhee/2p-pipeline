#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 15:03:10 2018

@author: juliana
"""
import matplotlib as mpl
mpl.use('agg')
import os
import numpy as np
import pylab as pl
import seaborn as sns
from random import shuffle
import pandas as pd
import json
import glob
import math
import datetime
from sklearn.feature_selection import RFE

from pipeline.python.paradigm import utils as util
from pipeline.python.utils import natural_keys, replace_root, label_figure
from matplotlib.colors import ListedColormap
from matplotlib.collections import LineCollection


from scipy import stats
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR, LinearSVR

def downsample_data(X_test_orig, labels_df, downsample_factor=0.1):
    chunk_size = int(round(1./downsample_factor))

    
    new_labels_df = []; downsampledX = [];
    for k,g in labels_df.groupby('trial'):
        arr_indices = g.index.tolist()
        trial_frames = X_test_orig[arr_indices] 
        chunks = np.arange(0, trial_frames.shape[0], step=chunk_size)
        nchunks = len(chunks)
        downsampled = np.array([np.sum(trial_frames[chunk:chunk+chunk_size, :], axis=0) \
                               if c < (nchunks-1) else np.sum(trial_frames[chunk:, :], axis=0) \
                                      for c,chunk in enumerate(chunks)])
        downsampledX.append(downsampled)
        
        # And get downsampled labels
        downsampled_tpoints = np.array([np.mean(labels_df['tsec'].values[chunk:chunk+chunk_size], axis=0) \
                               if c < (nchunks-1) else np.mean(labels_df['tsec'].values[chunk:], axis=0) \
                                      for c,chunk in enumerate(chunks)])
        new_stim_on_frame = np.where(abs(downsampled_tpoints-0) == min(abs(downsampled_tpoints-0)))[0][0]
        new_nframes_on = int(round(g['nframes_on'].values[0] * downsample_factor))
        
        
        newlabels = dict((col, np.tile(g[col].values[0], downsampled_tpoints.shape)) for col in g.columns.tolist() if col != 'tsec' )
        
        
        curr_labels_df = pd.DataFrame(newlabels)
        curr_labels_df['tsec'] = downsampled_tpoints
        curr_labels_df['nframes_on'] = np.tile(new_nframes_on, downsampled_tpoints.shape)
        curr_labels_df['stim_on_frame'] = np.tile(new_stim_on_frame, downsampled_tpoints.shape)
        
        new_labels_df.append(curr_labels_df)
        
    X_test_orig = np.vstack(downsampledX)
    labels_df = pd.concat(new_labels_df)
    labels_df = labels_df.reset_index(drop=True)

    return X_test_orig, labels_df



#%%

# A/B-classifier (trained well on 20180523 dataset, blobs_run2, traces001)
# -----------------------------------------------------------------------------
# For each trial in dataset, how well does the A/B-classifier do on morphs?
# Plot %-choose-A as a function of morph-level.
#
#def format_classifier_data(dataset, data_type='meanstim', class_name='morphlevel', 
#                               subset=None, aggregate_type='all', 
#                               const_trans='', trans_value='', 
#                               relabel=False):
#        
#    data_X = dataset['meanstim']
#    data_X = StandardScaler().fit(data_X)
#    data_y = dataset['ylabels']
#    ntrials_total = dataset['run_info'][()]['ntrials_total']
#    nframes_per_trial= dataset['run_info'][()]['nframes_per_trial']
#    
#    if data_type != 'frames':
#        data_y = np.reshape(data_y, (ntrials_total, nframes_per_trial))[:,0]
#    sconfigs = dataset['sconfigs'][()]
#    
#    X, y = group_classifier_data(data_X, data_y, class_name, sconfigs, 
#                                       subset=subset,
#                                       aggregate_type=aggregate_type, 
#                                       const_trans=const_trans, 
#                                       trans_value=trans_value, 
#                                       relabel=relabel)
#    
#    y = np.array([sconfigs_test[cv][class_name] for cv in y])
#    
#    class_labels = sorted(list(set(y)))
#
#    return X, y, class_labels



rootdir = '/Volumes/coxfs01/2p-data' #/mnt/odyssey'
animalid = 'CE077'
session = '20180724' #'20180713' #'20180629'
acquisition = 'FOV1_zoom1x'

acquisition_dir = os.path.join(rootdir, animalid, session, acquisition)


# #############################################################################
# Select TRAINING data and classifier:
# #############################################################################
train_runid = 'gratings_drifting_static' #'blobs_run2'
train_traceid = 'traces001'

train_data_type = 'meanstim'

if 'cnmf' in train_traceid:
    input_data_type = 'spikes'
    classif_identifier = 'stat_allrois_LinearSVC_kfold_6ori_all_%s_%s' % (train_data_type, input_data_type)
else:
    classif_identifier = 'stat_allrois_LinearSVC_kfold_8ori_all_%s' % (train_data_type)


data_identifier = '_'.join((animalid, session, acquisition))

#%%

# #############################################################################
# LOAD TRAINING DATA:
# #############################################################################

train_basedir = util.get_traceid_from_acquisition(acquisition_dir, train_runid, train_traceid)
    
train_fpath = os.path.join(train_basedir, 'classifiers', classif_identifier, '%s_datasets.npz' % classif_identifier)
train_dset = np.load(train_fpath)

train_dtype = 'cX_std'

train_X = train_dset[train_dtype]
train_y = train_dset['cy']


train_labels = sorted(list(set(train_y)))
print "Training labels:", train_labels

#%%  Classifier Parameters:
    
use_regression = False
fit_best = True
nfeatures_select = 50 #'all' #75 # 'all' #75
big_C = 1e9


#%%

# #############################################################################
# Fit classifier - get N best features, if applicable:
# #############################################################################

if train_X.shape[0] > train_X.shape[1]: # nsamples > nfeatures
    dual = False
else:
    dual = True

if 'LinearSVC' in classif_identifier:
    if use_regression is False:
        svc = LinearSVC(random_state=0, dual=dual, multi_class='ovr', C=big_C) #, C=best_C) # C=big_C)
        if fit_best:
            # First, get accuracy with all features:
            rfe = RFE(svc, n_features_to_select=nfeatures_select)
            rfe.fit(train_X, train_y)
            removed_rids = np.where(rfe.ranking_!=1)[0]
            kept_rids = np.array([i for i in np.arange(0, train_X.shape[-1]) if i not in removed_rids])
            train_X = train_X[:, kept_rids]
            print "Found %i best ROIs:" % nfeatures_select, train_X.shape
        else:
            print "Using ALL rois selected."
            
        svc.fit(train_X, train_y)
        clf = CalibratedClassifierCV(svc) 
        clf.fit(train_X, train_y)        
    else:
        print "Using regression..."
        clf = SVR(kernel='linear', C=1, tol=1e-9)
        clf.fit(train_X, train_y)
    
        classif_identifier_new = classif_identifier.replace('LinearSVC', 'SVRegression')
        
# #############################################################################

# Set output directories:
# -----------------------
classifier_dir = os.path.join(train_basedir, 'classifiers', classif_identifier)
test_results_dir = os.path.join(classifier_dir, 'testdata')
if not os.path.exists(test_results_dir): os.makedirs(test_results_dir)
train_results_dir = os.path.join(classifier_dir, 'traindata')
if not os.path.exists(train_results_dir): os.makedirs(train_results_dir)


#%%
# #############################################################################
# Select TESTING data:
# #############################################################################
test_runid = 'gratings_drifting_rotating' #'blobs_dynamic_run6' #'blobs_dynamic_run1' #'blobs_dynamic_run1'
test_traceid = 'traces001' #'cnmf_20180720_14_36_24'
test_data_type = 'smoothedDF' #'smoothedX' #'smoothedX' # 'corrected' #'smoothedX' #'smoothedDF'

#%%
# #############################################################################
# LOAD TEST DATA:
# #############################################################################

test_basedir = util.get_traceid_from_acquisition(acquisition_dir, test_runid, test_traceid)
    
test_fpath = os.path.join(test_basedir, 'data_arrays', 'datasets.npz')
test_dataset = np.load(test_fpath)
    
#%%
assert test_data_type in test_dataset.keys(), "Specified d-type (%s) not found. Choose from: %s" % (test_data_type, str(test_dataset.keys()))
assert len(test_dataset[test_data_type].shape)>0, "D-type is empty!"

# _, smoothed_X, _ = util.load_roiXtrials_df(test_dataset['run_info'][()]['traceid_dir'], trace_type='processed', dff=False, 
#                                                smoothed=True, frac=0.01, quantile=0.08)
#

X_test_orig = test_dataset[test_data_type]


test_configs = test_dataset['sconfigs'][()]
labels_df = pd.DataFrame(data=test_dataset['labels_data'], columns=test_dataset['labels_columns'])

test_info = test_dataset['run_info'][()]
framerate = test_dataset['run_info'][()]['framerate']


#%% # Format TEST data:

# Downsample to ~5 Hz (44.69 / 10)
downsample = False
threshold = False

if downsample:
    downsample_factor = 0.2
    
    X_test_orig, labels_df = downsample_data(X_test_orig, labels_df, downsample_factor=downsample_factor)

if test_data_type == 'spikes' and threshold:
    
    threshold_factor = 0.8
    
    for rid in range(X_test_orig.shape[-1]):
        ixs = np.where(X_test_orig[:, rid] < X_test_orig[:, rid].max()*threshold_factor)[0]
        X_test_orig[ixs, rid] = 0
    
    
# Whiten data:
X_test = StandardScaler().fit_transform(X_test_orig)

# Get config groups from labels matched to input data array:
cgroups = labels_df.groupby('config')

#%% # Load parsed MW file to get rotation values:

test_rundir = test_dataset['run_info'][()]['traceid_dir'].split('/traces')[0]
if rootdir not in test_rundir:
    test_rundir = replace_root(test_rundir, rootdir, animalid, session)
    
paradigm_fpath = glob.glob(os.path.join(test_rundir, 'paradigm', 'files', '*.json'))[0]
with open(paradigm_fpath, 'r') as f:
    mwtrials = json.load(f)



#%%
shuffle_frames = False

if fit_best:
    datestr = datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")
    with open(os.path.join(classifier_dir, 'fit_RFE_results_%s.txt' % datestr), 'wb') as f:
        f.write(str(svc))
        f.write('\n%s' % str({'kept_rids': kept_rids}))
    f.close()
    
#%%

# #############################################################################
# Test the classifier:
# #############################################################################

mean_pred = {}
sem_pred = {}
all_preds = {}
test_traces = {}
predicted = []
for k,g in cgroups:

    y_proba = []; y_true = [];
    for kk,gg in g.groupby('trial'):
        #print kk
        
        trial_ixs = gg.index.tolist()
        if shuffle_frames:
            shuffle(trial_ixs)

        curr_test = X_test[trial_ixs,:]
        orig_test_traces = X_test_orig[trial_ixs,:]
        
        if fit_best:
            curr_test = curr_test[:, kept_rids]
            orig_test_traces = orig_test_traces[:, kept_rids]
            
            roiset = 'best%i' % nfeatures_select
        else:
            roiset = 'all'
            
        if isinstance(clf, CalibratedClassifierCV):
            curr_proba = clf.predict_proba(curr_test)
        elif isinstance(clf, MLPRegressor):
            proba_tmp = clf.predict(curr_test)
            curr_proba = np.arctan2(proba_tmp[:, 0], proba_tmp[:, 1])
        else:
            curr_proba = clf.predict(curr_test)
        
        y_proba.append(curr_proba)
        y_true.append(orig_test_traces)
        
    y_proba = np.dstack(y_proba)
    curr_traces = np.dstack(y_true)
    
    means_by_class = np.mean(y_proba, axis=-1)
    stds_by_class = stats.sem(y_proba, axis=-1) #np.std(y_proba, axis=-1)
        
        
    mean_pred[k] = means_by_class
    sem_pred[k] = stds_by_class
    all_preds[k] = y_proba
    test_traces[k] = curr_traces


#%%
#for config in test_traces.keys():
#    tracemat = test_traces[config]
#    for rid in range(tracemat.shape[1]):
#        curr_rid_trace = tracemat[:, rid, :]
#        curr_rid_trace[curr_rid_trace < curr_rid_trace.min()] = 0
#        tracemat[:, rid, :] = curr_rid_trace
#        
#    test_traces[config] = tracemat

#%%

# #############################################################################
# Plot probability of each trained angle, if using CLASSIFIER:
# #############################################################################


# Set output dir:
decoding_dir = os.path.join(test_results_dir, '%s_%s' % (test_runid, test_data_type))
if not os.path.exists(decoding_dir): os.makedirs(decoding_dir)


plot_trials = True
if plot_trials:
    mean_lw = 2
    trial_lw=0.2
else:
    mean_lw = 1.0
    trial_lw=0.2
    
#drifting = False
framerate = 44.69
linear_legend = False


if isinstance(clf, CalibratedClassifierCV):
    
    configs_tested = sorted(list(set(labels_df['config'])), key=natural_keys)
    #maxval = 0.6 #max([all_preds[config].max() for config in mean_pred.keys()])
 
    stimdurs = sorted(list(set([test_configs[cf]['stim_dur'] for cf in test_configs.keys()])))
    print stimdurs
    if len(stimdurs) > 1:
        full_dur = stimdurs[-1]
        half_dur = stimdurs[-2]
        if len(stimdurs) > 2:
            quarter_dur = stimdurs[0]
    
    for config in configs_tested:
        #%
        print config
        minval = all_preds[config].min()
        maxval = all_preds[config].max() - 0.1
        
        # Plot each CLASS's probability on a subplot:
        # -----------------------------------------------------------------------------
        #colorvals = sns.color_palette("hsv", len(svc.classes_))
        colorvals = sns.color_palette("hls", len(svc.classes_))
        stim_on = list(set(labels_df[labels_df['config']==config]['stim_on_frame']))[0]
        nframes_on = list(set(labels_df[labels_df['config']==config]['nframes_on']))[0]
        stim_frames = np.arange(stim_on, stim_on+nframes_on)
        
        angle_step = list(set(np.diff(train_labels)))[0]
        if test_configs[config]['direction'] == 1:  # CW, values should be decreasing
            class_list = sorted(train_labels, reverse=True)
            shift_sign = -1
        else:
            class_list = sorted(train_labels, reverse=False)
            shift_sign = 1
            
        start_angle_ix = class_list.index(test_configs[config]['ori'])
        class_list = np.roll(class_list, shift_sign*start_angle_ix)

        class_indices = [[v for v in svc.classes_].index(c) for c in class_list]
        
        fig, axes = pl.subplots(len(class_list), 1, figsize=(6,15))

        nframes_in_trial = np.squeeze(all_preds[config]).shape[0]
        nclasses_total = np.squeeze(all_preds[config]).shape[1]
        ntrials_curr_config =  np.squeeze(all_preds[config]).shape[-1]

        for lix, (class_label, class_index) in enumerate(zip(class_list, class_indices)):
            #print lix
            cvals = [colorvals[class_index] if frameix in stim_frames else (0.0, 0.0, 0.0) for frameix in range(nframes_in_trial)]
    
            axes[lix].plot(np.arange(0, stim_frames[0]), mean_pred[config][0:stim_frames[0], class_index],
                                color='k', linewidth=mean_lw, alpha=1.0)
            axes[lix].plot(stim_frames, mean_pred[config][stim_frames, class_index], 
                                color=colorvals[class_index], linewidth=mean_lw, alpha=1.0)
            axes[lix].plot(np.arange(stim_frames[-1], nframes_in_trial), mean_pred[config][stim_frames[-1]:, class_index], 
                                color='k', linewidth=mean_lw, alpha=1.0)
    
            if plot_trials:
                plot_type = 'trials'
                for trialn in range(ntrials_curr_config):
    #                axes[lix].plot(range(nframes_in_trial), all_preds[config][:, class_index, trialn], 
    #                                    color=colorvals[class_index], linewidth=0.5)
                    axes[lix].plot(np.arange(0, stim_frames[0]), all_preds[config][0:stim_frames[0], class_index, trialn],
                                        color='k', linewidth=trial_lw, alpha=0.5)
                    
                    axes[lix].plot(stim_frames, all_preds[config][stim_frames, class_index, trialn], 
                                        color=colorvals[class_index], linewidth=trial_lw, alpha=0.5)
                    
                    axes[lix].plot(np.arange(stim_frames[-1], nframes_in_trial), all_preds[config][stim_frames[-1]:, class_index, trialn], 
                                        color='k', linewidth=trial_lw, alpha=0.5)
                    
            else:
                plot_type = 'fillstd'
    #            axes[lix].fill_between(range(nframes_in_trial), mean_pred[config][:,class_index]+sem_pred[config][:, class_index],
    #                                        mean_pred[config][:,class_index]-sem_pred[config][:, class_index], alpha=0.2,
    #                                        color=colorvals[class_index])
                axes[lix].fill_between(range(nframes_in_trial), mean_pred[config][:,class_index]+sem_pred[config][:, class_index],
                                            mean_pred[config][:,class_index]-sem_pred[config][:, class_index], alpha=0.2,
                                            color='k')
                
            #axes[lix].set_ylim([0.1, maxval])
            

            axes[lix].axes.xaxis.set_ticks([])
            #axes[lix].set_title(class_label)
                        
            # show stim dur, using class labels as color key:
            #pl.plot([stim_on, stim_on+nframes_on], np.ones((2,1))*0, 'r', linewidth=3)
            stimframes = np.arange(stim_on, stim_on+nframes_on)
            if test_configs[config]['stim_dur'] == quarter_dur:
                nclasses_shown = int(len(class_indices) * (1/4.)) + 1
            elif test_configs[config]['stim_dur'] == half_dur:
                nclasses_shown = int(len(class_indices) * (1/2.)) + 1
            else:
                nclasses_shown = len(class_indices)
            ordered_indices = np.array(class_indices[0:nclasses_shown])
            ordered_colors = [colorvals[c] for c in ordered_indices]
            
            bar_offset = 0.1
            curr_ylim = axes[lix].get_ylim()[0] - bar_offset
            
            axes[lix].set_ylim([minval-bar_offset*2, maxval])
            
            # Plot chance line:
            chance = 1/len(class_list)
            axes[lix].plot(np.arange(0, nframes_in_trial), np.ones((nframes_in_trial,))*chance, 'k--', linewidth=0.5)
            
                        
        sns.despine(trim=True, offset=4, ax=axes[lix])
        

        for lix in range(len(class_list)):
            # Create color bar:
            cy = np.ones(stimframes.shape) * axes[lix].get_ylim()[0]/2.0
            z = stimframes.copy()
            points = np.array([stimframes, cy]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            cmap = ListedColormap(ordered_colors)
            lc = LineCollection(segments, cmap=cmap)
            lc.set_array(z)
            lc.set_linewidth(5)
            axes[lix].add_collection(lc)
            
            if lix == len(class_list)-1:
                axes[lix].set_xticks((stim_on, stim_on + framerate))
                axes[lix].set_xticklabels([0, 1])
                axes[lix].set_xlabel('sec', horizontalalignment='right', x=0.25)        
                for axside in ['top', 'right']:
                    axes[lix].spines[axside].set_visible(False)
                sns.despine(trim=True, offset=4, ax=axes[lix])

            else:
                axes[lix].axes.xaxis.set_ticks([])
                for axside in ['bottom', 'top', 'right']:
                    axes[lix].spines[axside].set_visible(False)
                axes[lix].axes.xaxis.set_visible(False) #([])
                sns.despine(trim=True, offset=4, ax=axes[lix])
                
            axes[lix].set_ylabel('prob (%i)' % class_list[lix])

        
        #pl.plot(stimframes, np.ones(stimframes.shape), color=ordered_colors, linewidth=5)
        pl.subplots_adjust(top=0.85)
        
        # Custom legend:
        if linear_legend:
            from matplotlib.lines import Line2D
            custom_lines = []
            for lix, c_ori in enumerate(svc.classes_):
                custom_lines.append(Line2D([0], [0], color=colorvals[lix], lw=4))
            pl.legend(custom_lines, svc.classes_, loc=9, bbox_to_anchor=(0.5, -0.2), ncol=len(svc.classes_)/2)
        else:
            if 180 in class_list:
             legend = fig.add_axes([0.75, 0.75, 0.2, 0.3],
                          projection='polar') 
            else:
             legend = fig.add_axes([0.85, 0.75, 0.2, 0.3],
                          projection='polar')
        
            thetas = sorted(np.array([ori_deg*(math.pi/180) for ori_deg in class_list]))
            if max(thetas) < 300:
                thetas = np.append(thetas, 180*(math.pi/180))
            
            for tix, theta in enumerate(thetas):
                print theta
                if theta == math.pi and max(class_list) < 300:
                    color_ix = 0
                else:
                    color_ix = [t for t in thetas].index(theta)
                legend.plot([theta, theta], [0, 1], color=colorvals[color_ix], lw=3)
            legend.set_theta_zero_location("N")
            legend.set_xlim([0, math.pi])
            legend.grid(False); 
            legend.set_xticks(thetas)
            if max(class_list) < 300:
                thetas[-1] = 0
            legend.set_xticklabels([int(round(t*(180/math.pi))) for t in thetas], fontsize=8)
            legend.yaxis.grid(False); legend.set_yticklabels([])
            legend.spines["polar"].set_visible(False)


        if test_configs[config]['direction']==1:
            rot_direction = 'CW'
        else:
            rot_direction = 'CCW'
            
        if test_configs[config]['stim_dur'] == half_dur:
            rot_duration = 'half'
        elif test_configs[config]['stim_dur'] == quarter_dur:
            rot_duration = 'quarter'
        else:
            rot_duration = 'full'
        starting_rot = test_configs[config]['ori']
        config_english = 'start %i [%s, %s]' % (starting_rot, rot_direction, rot_duration)
        pl.suptitle(config_english)
        
        label_figure(fig, data_identifier)
        #%
        #train_savedir = os.path.split(train_fpath)[0]
        
        
        if shuffle_frames:
            figname = '%s_hsv_TEST_%s_%s_start_%i_%s_%s_%s_%s_shuffled.png' % (roiset, test_runid, config, starting_rot, rot_direction, rot_duration, plot_type, test_data_type)
        else:
            figname = '%s_hsv_TEST_%s_%s_start_%i_%s_%s_%s_%s.png' % (roiset, test_runid, config, starting_rot, rot_direction, rot_duration, plot_type, test_data_type)
        pl.savefig(os.path.join(decoding_dir, figname))
        pl.close()

#%%
        
# #############################################################################
# Load TRAINING DATA and plot traces:
# #############################################################################
# POLAR PLOTS?

# 
training_data_fpath = os.path.join(train_basedir, 'data_arrays', 'datasets.npz')
training_data = np.load(training_data_fpath)
print training_data.keys()
train_labels_df = pd.DataFrame(data=training_data['labels_data'], columns=training_data['labels_columns'])

#%%
train_data_type = 'smoothedDF'
trainingX = training_data[train_data_type][:, kept_rids]
print trainingX.shape
nrois = trainingX.shape[-1]

F0 = training_data['F0'][:, kept_rids]
use_dff = False


#%% FORMAT TRAINING DATA for easier plotting and comparison with true TESTING data:

threhsold = False
threshold_factor = 0.5
        
if fit_best:

    # Get trial structure:
    assert len(list(set(train_labels_df['nframes_on']))) == 1, "More than 1 nframes_on found in TRAIN set..."
    train_nframes_on = list(set(train_labels_df['nframes_on']))[0]
    assert len(list(set(train_labels_df['stim_on_frame']))) == 1, "More than 1 stim_on_frame val found in TRAIN set..."
    train_stim_on = list(set(train_labels_df['stim_on_frame']))[0]
    ntrials_by_cond = [v for k,v in training_data['run_info'][()]['ntrials_by_cond'].items()]
    assert len(list(set(ntrials_by_cond)))==1, "More than 1 rep values found in TRAIN set"
    ntrials_per_cond = list(set(ntrials_by_cond))[0]
    
    train_configs = training_data['sconfigs'][()]
    
    config_list = sorted([c for c in train_configs.keys()], key=lambda x: train_configs[x]['ori'])
    train_traces = {}
    for cf in config_list:
        print cf, train_configs[cf]['ori']
        
        cixs = train_labels_df[train_labels_df['config']==cf].index.tolist()
        curr_frames = trainingX[cixs, :]
        print curr_frames.shape
        nframes_per_trial = len(cixs) / ntrials_per_cond
        
        #tmat = np.reshape(curr_frames, (ntrials_per_cond, nframes_per_trial, nrois))
        tmat = np.reshape(curr_frames, (nframes_per_trial, ntrials_per_cond, nrois), order='f') 
        tmat = np.swapaxes(tmat, 1, 2) # Nframes x Nrois x Ntrials (to match test_traces)
        if train_data_type == 'spikes' and threshold:
            print "... thresholding spikes"
            tmat[tmat <= tmat.max()*threshold_factor] = 0.
            
        if use_dff:
#            bmat = np.reshape(F0[cixs, :], (nframes_per_trial, ntrials_per_cond, nrois), order='f')
#            bmat = np.swapaxes(bmat, 1, 2)
#            dfmat = tmat / bmat
            
            bs = np.mean(tmat[0:train_stim_on, :, :], axis=0)
            dfmat = (tmat - bs) / bs
            #dfmat = tmat / bs
            train_traces[cf] = dfmat
        else:
            train_traces[cf] = tmat
    
    responses = []; baselines = [];
    for cf in config_list:
        tmat = train_traces[cf]
        print cf
        baselines_per_trial = np.mean(tmat[0:train_stim_on, :, :], axis=0) # ntrials x nrois -- baseline val for each trial
        #meanstims_per_trial = np.mean(tmat[train_stim_on:train_stim_on+train_nframes_on, :, :], axis=0) # ntrials x nrois -- baseline val for each trial
        #mean_config = np.mean(meanstims_per_trial, axis=-1)
        baseline_config = np.max(baselines_per_trial, axis=-1)
                
        # Use Max value of the MEAN trace during stim period:
        mean_config = np.max(np.mean(tmat[train_stim_on:train_stim_on+train_nframes_on, :, :], axis=-1), axis=0) # ntrials x nrois -- baseline val for each trial
        
#        dffs_per_trial = ( meanstims_per_trial - baselines_per_trial) / baselines_per_trial
#        
#        for ridx in range(dffs_per_trial.shape[0]):
#            dffs_per_trial[ridx, :] -= dffs_per_trial[ridx, :].min()
#            
#        mean_dff_config = np.mean(dffs_per_trial, axis=-1)
#        

        responses.append(mean_config)
        baselines.append(baseline_config)
        
    responses = np.vstack(responses) # Reshape into NCONFIGS x NROIS array
    offsets = np.vstack(baselines)
    
    
    
#%%

# INPUT DATA:  Plot POLAR plots to show direction tuning:
# =============================================================================
    
thetas = [train_configs[cf]['ori'] * (math.pi/180) for cf in config_list]

print thetas

import matplotlib as mpl

nrows = 5 #4
ncols = 10 #5


fig, axes = pl.subplots(figsize=(10,10), nrows=nrows, ncols=ncols, subplot_kw=dict(polar=True))
appended_thetas = np.append(thetas, math.pi)

for ridx, ax in zip(range(nrois), axes.flat):
    #print ridx
    radii = responses[:, ridx]
    if radii.min() < 0:
        print "%i - fixing offset" % ridx
        radii -= radii.min()
    polygon = mpl.patches.Polygon(zip(thetas, radii), fill=True, alpha=0.5, color='mediumorchid')
    ax.add_line(polygon)
    #ax.autoscale()
    ax.grid(True)
    ax.set_theta_zero_location("N")
    ax.set_xticklabels([])
    #ax.set_title(ridx)
    max_angle_ix = np.where(radii==radii.max())[0][0]
    ax.plot([0, thetas[max_angle_ix]], [0, radii[max_angle_ix]],'k', lw=1)
    ax.set_title(kept_rids[ridx]+1, fontsize=8)
    ax.set_xticks([t for t in thetas])
#    ax.set_thetamin(0)
#    ax.set_thetamax(math.pi) #([0, math.pi])
#    
pl.rc('ytick', labelsize=8)
        
    #ax.set_xticklabels([int(round(t*(180/math.pi))) for t in thetas], fontsize=6)
    #ax#.yaxis.grid(False); legend.set_yticklabels([])
    #ax.spines["polar"].set_visible(False)

label_figure(fig, data_identifier)

figname = '%s_polar_plots_%s.png' % (roiset, train_data_type)
pl.savefig(os.path.join(train_results_dir, figname))
    
#%%

# Sort by direction preference:
# -----------------------------------------------------------------------------

thetas = [train_configs[cf]['ori'] for cf in config_list]

pdirections = []
for ridx in range(nrois):
    #print ridx
    radii = responses[:, ridx]
    if radii.min() < 0:
        print "ROI %i has negative response. SKIPPING." % (kept_rids[ridx]+1)
        continue
        #radii -= radii.min()
    max_angle_ix = np.where(radii==radii.max())[0][0]

    pdirections.append((ridx, thetas[max_angle_ix], radii[max_angle_ix]))
    
sorted_rois = sorted([pdir[0] for pdir in pdirections], key=lambda x: pdirections[x][1])

for ridx in sorted_rois:
    print ridx, pdirections[ridx][1]

#%%
    
# #############################################################################
# Plot mean traces for ROIs sorted by ANGLE: TRAINING DATA
# #############################################################################

plot_trials = True
trial_lw = 0.2
mean_lw = 1

config_list = sorted([c for c in train_configs.keys()], key=lambda x: train_configs[x]['ori'])
class_indices = np.arange(0, len(config_list))

fig, axes = pl.subplots(len(config_list), 1, figsize=(6,15))
for lix, (class_label, class_index) in enumerate(zip(config_list, class_indices)):
    #print lix
    
    #grand_meantrace_across_rois = np.mean(np.mean(traces[class_label], axis=-1), axis=0)
    rois_preferred = [pdir[0] for pdir in pdirections if pdir[1]==train_configs[class_label]['ori']]
    print "Found %i cells with preferred dir %i" % (len(rois_preferred), train_configs[class_label]['ori'])
    stim_on = list(set(train_labels_df[train_labels_df['config']==class_label]['stim_on_frame']))[0]
    nframes_on = list(set(train_labels_df[train_labels_df['config']==class_label]['nframes_on']))[0]
    stim_frames = np.arange(stim_on, stim_on+nframes_on)
        
    for ridx in rois_preferred:
        roi_traces = train_traces[class_label][:, ridx, :]
        mean_roi_trace = np.mean(roi_traces, axis=-1)
        std_roi_trace = np.std(roi_traces, axis=-1)
        ntrials_curr_config = roi_traces.shape[-1] # original array is:  Ntrials x Nframes x Nrois

        cvals = [colorvals[class_index] if frameix in stim_frames else (0.0, 0.0, 0.0) for frameix in range(len(mean_roi_trace))]

        axes[lix].plot(np.arange(0, stim_frames[0]), mean_roi_trace[0:stim_frames[0]],
                            color='k', linewidth=mean_lw, alpha=1.0)
        axes[lix].plot(stim_frames, mean_roi_trace[stim_frames], 
                            color=colorvals[class_index], linewidth=mean_lw, alpha=1.0)
        axes[lix].plot(np.arange(stim_frames[-1], len(mean_roi_trace)), mean_roi_trace[stim_frames[-1]:], 
                            color='k', linewidth=mean_lw, alpha=1.0)

        # PLOT std:
        if plot_trials:
            plot_type = 'trials'
            for trialn in range(ntrials_curr_config):
                axes[lix].plot(np.arange(0, stim_frames[0]), roi_traces[0:stim_frames[0], trialn],
                                    color='k', linewidth=trial_lw, alpha=0.3)
                
                axes[lix].plot(stim_frames, roi_traces[stim_frames, trialn], 
                                    color=colorvals[class_index], linewidth=trial_lw, alpha=0.3)
                
                axes[lix].plot(np.arange(stim_frames[-1], len(mean_roi_trace)), roi_traces[stim_frames[-1]:, trialn], 
                                    color='k', linewidth=trial_lw, alpha=0.3)
        else:
            axes[lix].fill_between(range(len(mean_roi_trace)), mean_roi_trace+std_roi_trace,
                                    mean_roi_trace-std_roi_trace, alpha=0.2,
                                    color='k')
    if lix < len(config_list):
        axes[lix].axes.xaxis.set_visible(True) #([])
        axes[lix].axes.xaxis.set_ticks([])
        
    axes[lix].set_title(str([kept_rids[r]+1 for r in rois_preferred]))
        

sns.despine(trim=True, offset=4, bottom=True)

# Custom legend:
from matplotlib.lines import Line2D
custom_lines = []
for lix, c_ori in enumerate(svc.classes_):
    custom_lines.append(Line2D([0], [0], color=colorvals[lix], lw=4))
pl.legend(custom_lines, svc.classes_, loc=9, bbox_to_anchor=(0.5, -0.2), ncol=len(svc.classes_)/2)
pl.suptitle('%s: Training set' % roiset)

label_figure(fig, data_identifier)


figname = '%s_TRAIN_traces_%s_%s.png' % (roiset, train_runid, train_data_type)
pl.savefig(os.path.join(train_results_dir, figname))


#%%

# #############################################################################
# Plot mean traces for ROIs sorted by ANGLE: TESTING DATA
# #############################################################################

# Create decoder-style plots for ROI responses to test data
# Each group of cells in preferred-direction group serves as "classifier"

plot_trials = True
if plot_trials:
    mean_lw = 1|
    trial_lw=0.1
else:
    mean_lw = 1.0
    trial_lw=0.2
    
#drifting = False
framerate = 44.69
linear_legend = False
if 'DF' in test_data_type:
    yaxis_label = 'df/f'
else:
    yaxis_label = 'intensity'
    
response_classif_dir = os.path.join(train_results_dir, 'test_data_responses')
if not os.path.exists(response_classif_dir):
    os.makedirs(response_classif_dir)
    
    
configs_tested = sorted(list(set(labels_df['config'])), key=natural_keys)
#maxval = 0.5 #max([all_preds[config].max() for config in mean_pred.keys()])

stimdurs = sorted(list(set([test_configs[cf]['stim_dur'] for cf in test_configs.keys()])))
print stimdurs
if len(stimdurs) > 1:
    full_dur = stimdurs[-1]
    half_dur = stimdurs[-2]
    if len(stimdurs) > 2:
        quarter_dur = stimdurs[0]

for test_config in configs_tested:
    #%
    print test_config
    #minval = test_traces[test_config].min()
    #maxval = test_traces[test_config].max()
    
    # Plot each CLASS's probability on a subplot:
    # -----------------------------------------------------------------------------
    #colorvals = sns.color_palette("hsv", len(svc.classes_))
    colorvals = sns.color_palette("hls", len(svc.classes_))
    stim_on = list(set(labels_df[labels_df['config']==test_config]['stim_on_frame']))[0]
    nframes_on = list(set(labels_df[labels_df['config']==test_config]['nframes_on']))[0]
    stim_frames = np.arange(stim_on, stim_on+nframes_on)
        
    angle_step = list(set(np.diff(train_labels)))[0]
    if test_configs[test_config]['direction'] == 1:  # CW, values should be decreasing
        class_list = sorted(train_labels, reverse=True)
        shift_sign = -1
    else:
        class_list = sorted(train_labels, reverse=False)
        shift_sign = 1
        
    start_angle_ix = class_list.index(test_configs[test_config]['ori'])
    class_list = np.roll(class_list, shift_sign*start_angle_ix)
        
    class_indices = [[v for v in svc.classes_].index(c) for c in class_list]
    
    #%
    fig, axes = pl.subplots(len(class_list), 1, figsize=(6,15))
    #for lix in range(8):
    #    curr_ori = svc.classes_[lix]
    nframes_in_trial = np.squeeze(all_preds[test_config]).shape[0]
    nclasses_total = np.squeeze(all_preds[test_config]).shape[1]
    ntrials_curr_config =  np.squeeze(all_preds[test_config]).shape[-1]

    for lix, (class_label, class_index) in enumerate(zip(class_list, class_indices)):
        #print lix, class_label

        #grand_meantrace_across_rois = np.mean(np.mean(traces[class_label], axis=-1), axis=0)
        rois_preferred = [pdir[0] for pdir in pdirections if pdir[1]==class_label]
        
        print "Found %i cells with preferred dir %i" % (len(rois_preferred), class_label)
        
        for ridx in rois_preferred:
            roi_traces = test_traces[test_config][:, ridx, :] # Test mat is:  Nframes x Nrois x Ntrials
            mean_roi_trace = np.mean(roi_traces, axis=1)
            std_roi_trace = np.std(roi_traces, axis=1)
    
            cvals = [colorvals[class_index] if frameix in stim_frames else (0.0, 0.0, 0.0) for frameix in range(len(mean_roi_trace))]
    
            axes[lix].plot(np.arange(0, stim_frames[0]), mean_roi_trace[0:stim_frames[0]], color='k', linewidth=1.5, alpha=1.0)
            axes[lix].plot(stim_frames, mean_roi_trace[stim_frames], color=colorvals[class_index], linewidth=1.5, alpha=1.0)
            axes[lix].plot(np.arange(stim_frames[-1]+1, len(mean_roi_trace)), mean_roi_trace[stim_frames[-1]+1:], color='k', linewidth=1.5, alpha=1.0)

            if plot_trials:
                plot_type = 'trials'
                for trialn in range(ntrials_curr_config):
                    axes[lix].plot(np.arange(0, stim_frames[0]), test_traces[test_config][0:stim_frames[0], ridx, trialn],
                                        color='k', linewidth=0.4, alpha=0.3)
                    
                    axes[lix].plot(stim_frames, test_traces[test_config][stim_frames, ridx, trialn], 
                                        color=colorvals[class_index], linewidth=0.4, alpha=0.3)
                    
                    axes[lix].plot(np.arange(stim_frames[-1], nframes_in_trial), test_traces[test_config][stim_frames[-1]:, ridx, trialn], 
                                        color='k', linewidth=0.4, alpha=0.3)
                    
            else:
                plot_type = 'fillstd'
                axes[lix].fill_between(range(nframes_in_trial), mean_roi_trace + std_roi_trace,
                                            mean_roi_trace - std_roi_trace, alpha=0.2,
                                            color='k')

        axes[lix].set_title(str([kept_rids[r]+1 for r in rois_preferred]))
        axes[lix].axes.xaxis.set_ticks([])
        #axes[lix].set_title(class_label)
                    
        # show stim dur, using class labels as color key:
        #pl.plot([stim_on, stim_on+nframes_on], np.ones((2,1))*0, 'r', linewidth=3)
        stimframes = np.arange(stim_on, stim_on+nframes_on)
        if test_configs[test_config]['stim_dur'] == quarter_dur:
            nclasses_shown = int(len(class_indices) * (1/4.)) + 1
        elif test_configs[test_config]['stim_dur'] == half_dur:
            nclasses_shown = int(len(class_indices) * (1/2.)) + 1
        else:
            nclasses_shown = len(class_indices)
        ordered_indices = np.array(class_indices[0:nclasses_shown])
        ordered_colors = [colorvals[c] for c in ordered_indices]
        
#        bar_offset = 0.1 * 1000
#        curr_ylim = axes[lix].get_ylim()[0] - bar_offset
#        axes[lix].set_ylim([minval-bar_offset*2, axes[lix].get_ylim()[1]])
#        
        # Plot chance line:
        chance = 1/len(class_list)
        axes[lix].plot(np.arange(0, nframes_in_trial), np.ones((nframes_in_trial,))*chance, 'k--', linewidth=0.5)
        
                  
        sns.despine(trim=True, offset=4, ax=axes[lix])
    

    for lix in range(len(class_list)):
        # Create color bar:
        cy = np.ones(stimframes.shape) * axes[lix].get_ylim()[0]/2.0
        z = stimframes.copy()
        points = np.array([stimframes, cy]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        cmap = ListedColormap(ordered_colors)
        lc = LineCollection(segments, cmap=cmap)
        lc.set_array(z)
        lc.set_linewidth(5)
        axes[lix].add_collection(lc)
        
        if lix == len(class_list)-1:
            axes[lix].set_xticks((stim_on, stim_on + framerate))
            axes[lix].set_xticklabels([0, 1])
            axes[lix].set_xlabel('sec', horizontalalignment='right', x=0.25)        
            for axside in ['top', 'right']:
                axes[lix].spines[axside].set_visible(False)
            sns.despine(trim=True, offset=4, ax=axes[lix])

        else:
            axes[lix].axes.xaxis.set_ticks([])
            for axside in ['bottom', 'top', 'right']:
                axes[lix].spines[axside].set_visible(False)
            axes[lix].axes.xaxis.set_visible(False) #([])
        axes[lix].set_ylabel('%s (%i)' % (yaxis_label, class_list[lix]))

    
    #pl.plot(stimframes, np.ones(stimframes.shape), color=ordered_colors, linewidth=5)
    pl.subplots_adjust(top=0.85)
    
    # Custom legend:
    if linear_legend:
        from matplotlib.lines import Line2D
        custom_lines = []
        for lix, c_ori in enumerate(svc.classes_):
            custom_lines.append(Line2D([0], [0], color=colorvals[lix], lw=4))
        pl.legend(custom_lines, svc.classes_, loc=9, bbox_to_anchor=(0.5, -0.2), ncol=len(svc.classes_)/2)
    else:
        legend = fig.add_axes([0.85, 0.75, 0.2, 0.3],
                      projection='polar')
    
        thetas = sorted(np.array([ori_deg*(math.pi/180) for ori_deg in class_list]))
        thetas = np.append(thetas, 180*(math.pi/180))
        
        for tix, theta in enumerate(thetas):
            print theta
            if theta == math.pi:
                color_ix = 0
            else:
                color_ix = [t for t in thetas].index(theta)
            legend.plot([theta, theta], [0, 1], color=colorvals[color_ix], lw=3)
        legend.set_theta_zero_location("N")
        legend.set_xlim([0, math.pi])
        legend.grid(False); 
        legend.set_xticks(thetas)
        thetas[-1] = 0
        legend.set_xticklabels([int(round(t*(180/math.pi))) for t in thetas], fontsize=8)
        legend.yaxis.grid(False); legend.set_yticklabels([])
        legend.spines["polar"].set_visible(False)


    if test_configs[test_config]['direction']==1:
        rot_direction = 'CW'
    else:
        rot_direction = 'CCW'
        
    if test_configs[test_config]['stim_dur'] == half_dur:
        rot_duration = 'half'
    elif test_configs[test_config]['stim_dur'] == quarter_dur:
        rot_duration = 'quarter'
    else:
        rot_duration = 'full'
    starting_rot = test_configs[test_config]['ori']
    config_english = 'start %i [%s, %s]' % (starting_rot, rot_direction, rot_duration)
    
    label_figure(fig, data_identifier)
    
    pl.suptitle(config_english)
    
    #%
    #train_savedir = os.path.split(train_fpath)[0]
    
    
    if shuffle_frames:
        figname = '%s_traces_TEST_%s_%s_start_%i_%s_%s_%s_%s_shuffled.png' % (roiset, test_runid, test_config, starting_rot, rot_direction, rot_duration, plot_type, test_data_type)
    else:
        figname = '%s_traces_TEST_%s_%s_start_%i_%s_%s_%s_%s.png' % (roiset, test_runid, test_config, starting_rot, rot_direction, rot_duration, plot_type, test_data_type)
    
    pl.savefig(os.path.join(response_classif_dir, figname))
    pl.close()
    
