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
import ast
from sklearn.feature_selection import RFE
from sklearn.externals import joblib

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

#%%
def filter_noise(X, labels_df, min_spikes=0.0002):    
    
    X[X <= min_spikes] = 0.

    return X

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

def load_training_data(training_data_fpath):
    train_runid = os.path.split(training_data_fpath.split('/traces/')[0])[-1]
    if 'comb' in train_runid:
        tmpd = np.load(training_data_fpath)
        training_data = tmpd['arr_0'][()]
        train_runinfo = training_data['run_info']
    else:
        training_data = np.load(training_data_fpath)
        train_runinfo = training_data['run_info'][()]
    
    return training_data, train_runinfo


def load_training_set(train_fpath, train_dtype='cX_std'):
        
    # Load data:
    train_dset = np.load(train_fpath)
    assert train_dtype in train_dset.keys(), "Specified training dset %s not found!" % train_dtype
    
        
    train_X = train_dset[train_dtype]
    train_y = [int(t) if t != 'bas' else 'bas' for t in train_dset['cy']]

    
    train_labels = sorted(list(set(train_y)))
    #train_labels = [t for t in train_labels if t != 'bas']
    
    return train_X, train_y, train_labels


def get_testing_set(test_fpath, test_data_type='smoothedX'):
    # Load test dataset:    
    test_dataset = np.load(test_fpath)
    #%
    assert test_data_type in test_dataset.keys(), "Specified d-type (%s) not found. Choose from: %s" % (test_data_type, str(test_dataset.keys()))
    assert len(test_dataset[test_data_type].shape)>0, "D-type is empty!"
    
    # _, smoothed_X, _ = util.load_roiXtrials_df(test_dataset['run_info'][()]['traceid_dir'], trace_type='processed', dff=False, 
    #                                                smoothed=True, frac=0.01, quantile=0.08)
    #
    
    test_X = test_dataset[test_data_type]

    
    test_sconfigs = test_dataset['sconfigs'][()]
    test_labels_df = pd.DataFrame(data=test_dataset['labels_data'], columns=test_dataset['labels_columns'])
    
    test_runinfo = test_dataset['run_info'][()]
    
    return test_X, test_labels_df, test_sconfigs, test_runinfo


#%


def refit_SVC(svc, train_X, train_y, classifier_dir, fit_best=False, nfeatures_select='all'):
        
    #if 'LinearSVC' in classif_identifier:
    if fit_best:
        roiset = 'best%i' % nfeatures_select
        # First check if we've done this already:
        fit_results = sorted(glob.glob(os.path.join(classifier_dir, 'results', '%s_fit_RFE_results*.txt' % roiset)), key=natural_keys)
        if len(fit_results) > 0:
            # Load the most recent one:
            print "Loading most recent fit: %s" % os.path.split(fit_results[-1])[-1]
            with open(fit_results[-1], 'rb') as f:
                for texti, textline in enumerate(f):
                    if 'kept_rids' in textline:
                        kept_rids_tmp = ast.literal_eval(textline)
                        break
            kept_rids = np.array(kept_rids_tmp['kept_rids'])
        else:
            # Recursively fit to get top N features:
            rfe = RFE(svc, n_features_to_select=nfeatures_select)
            rfe.fit(train_X, train_y)
            removed_rids = np.where(rfe.ranking_!=1)[0]
            kept_rids = np.array([i for i in np.arange(0, train_X.shape[-1]) if i not in removed_rids])
            datestr = datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")
            with open(os.path.join(classifier_dir, 'results', '%s_fit_RFE_results_%s_subset.txt' % (roiset, datestr)), 'wb') as f:
                f.write('%s\n' % str(svc))
                f.write('\n%s' % str({'kept_rids': list(kept_rids)}))
                
        # Only take top N cells:
        train_X = train_X[:, kept_rids]
        print "Found %i best ROIs:" % nfeatures_select, train_X.shape
    
    else:
        print "Using ALL rois selected."
        roiset = 'all'
        kept_rids = np.arange(0, train_X.shape[-1])
    
    # Re-fit classifier:
    svc.fit(train_X, train_y)
    
    # Convert to get probabilities
    clf = CalibratedClassifierCV(svc) 
    clf.fit(train_X, train_y)    

    return clf, train_X, kept_rids, roiset


def fit_regressor(train_X, train_y, classif_identifier):
    print "Using regression..."
    clf = SVR(kernel='linear', C=1, tol=1e-9)
    clf.fit(train_X, train_y)

    classif_identifier = classif_identifier.replace('LinearSVC', 'SVRegression')
    
    return clf, classif_identifier

#%%


def decode_traces(clf, cgroups, X, shuffle_frames=False):

    all_class_predictions = {}
    test_traces = {}
    
    for k,g in cgroups:
        predictions = []; trace_values = [];
        for kk,gg in g.groupby('trial'):
    
            trial_ixs = gg.index.tolist()
            if shuffle_frames:
                shuffle(trial_ixs)
                
            # Get subset of input test data corresponding to current config:
            curr_test = X[trial_ixs, :]
            #orig_test_traces = test_X_orig[trial_ixs,:]
    
            if isinstance(clf, CalibratedClassifierCV):
                curr_proba = clf.predict_proba(curr_test)
            elif isinstance(clf, MLPRegressor):
                proba_tmp = clf.predict(curr_test)
                curr_proba = np.arctan2(proba_tmp[:, 0], proba_tmp[:, 1])
            else:
                curr_proba = clf.predict(curr_test)
            
            predictions.append(curr_proba)
            trace_values.append(curr_test) #(orig_test_traces)
            
        predictions = np.dstack(predictions)
        trace_values = np.dstack(trace_values)
        
        all_class_predictions[k] = predictions
        test_traces[k] = trace_values
    
    return all_class_predictions, test_traces


#%%

def get_ordered_classes(config, test_sconfigs, class_names, direction_label):
    
    if test_sconfigs[config][direction_label] > 0: #== 1:  # CW, values should be decreasing
        class_list = sorted(class_names, reverse=True)
        shift_sign = -1
        start_angle_ix = class_list[0]
    else:
        class_list = sorted(class_names, reverse=False)
        shift_sign = 1
        start_angle_ix = class_list[0]
    
    if is_grating:
        if max(class_list) < 180 and test_sconfigs[config]['ori'] == 180: # Not a drifting case, 180 = 0
            start_angle_ix = class_list.index(0)
        else:
            start_angle_ix = class_list.index(test_sconfigs[config]['ori'])      
        class_list = np.roll(class_list, shift_sign*start_angle_ix)

    if any([isinstance(v, str) for v in class_names]):
        class_indices = [[v for v in class_names].index(str(c)) for c in class_list]
    else:
        class_indices = [[v for v in class_names].index(c) for c in class_list]
    
    return class_list, class_indices

#%%


def draw_subplot_stimbars(axes, config, stim_frames, class_names, class_indices, 
                          sconfigs, colorvals, is_static=False, y_unit=None, bar_offset=0.1):
    
    if any('stim_dur' in sconfigs[cf].keys() for cf in sconfigs.keys()): 
        stimdurs = sorted(list(set([sconfigs[cf]['stim_dur'] for cf in sconfigs.keys()])))
        vary_stimdur = True
        if len(stimdurs) > 1:
            full_dur = stimdurs[-1]
            half_dur = stimdurs[-2]
            if len(stimdurs) > 2:
                quarter_dur = stimdurs[0]
    else:
        stimdurs = None
        vary_stimdur = False
        
    for lix in range(len(axes)):
        # Adjust ylimit and set stimbar location:
        ymin, ymax = axes[lix].get_ylim()
        curr_ylim = ymin - bar_offset
        axes[lix].set_ylim([ymin-bar_offset*2, ymax])
        axes[lix].axes.xaxis.set_ticks([])
        if y_unit is None:
            axes[lix].set_ylabel('prob (%i)' % class_names[class_indices[lix]])
        else:
            axes[lix].set_ylabel('%s' % y_unit)            
        if vary_stimdur:
            if sconfigs[config]['stim_dur'] == quarter_dur:
                nclasses_shown = int(len(class_indices) * (1/4.)) + 1
            elif sconfigs[config]['stim_dur'] == half_dur:
                nclasses_shown = int(len(class_indices) * (1/2.)) + 1
            else:
                nclasses_shown = len(class_indices)
        else:
            nclasses_shown = len(class_indices)
        ordered_indices = np.array(class_indices[0:nclasses_shown])
        
        if is_static:
            ordered_colors = colorvals[lix]
        else:
            if 'bas' in class_names:
                #ordered_colors = [colorvals[train_labels.index(int(svc.classes_[c]))] for c in ordered_indices]
                ordered_colors = [colorvals[class_names.index(int(class_names[c]))] for c in ordered_indices]
            else:
                ordered_colors = [colorvals[c] for c in ordered_indices]
            
        # Create color bar:
        cy = np.ones(stim_frames.shape) * axes[lix].get_ylim()[0]/2.0
        z = stim_frames.copy()
        points = np.array([stim_frames, cy]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        cmap = ListedColormap(ordered_colors)
        lc = LineCollection(segments, cmap=cmap)
        lc.set_array(z)
        lc.set_linewidth(5)
        axes[lix].add_collection(lc)
        
        
def format_xaxis_subplots(axes, stim_on, framerate):
    for lix in range(len(axes)):
        if lix == len(axes)-1:
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


def draw_legend(fig, class_list, colorvals , polar=False):         
    if len(class_list) > 5: #180 in class_list:
        legend_axes = [0.75, 0.75, 0.2, 0.3]
    else:
        legend_axes = [0.80, 0.75, 0.2, 0.3]

    if polar:
        legend = fig.add_axes(legend_axes, projection='polar') 
    
        thetas = sorted(np.array([ori_deg*(math.pi/180) for ori_deg in class_list]))
        if max(thetas) < 300:
            # Add 180 to legend, since that can be a starting/end point
            thetas = np.append(thetas, 180*(math.pi/180))
        
        # Assign colors to each class:
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
            # Make sure 180 is just 0:
            thetas[-1] = 0
        legend.set_xticklabels([int(round(t*(180/math.pi))) for t in thetas], fontsize=8)
        legend.yaxis.grid(False); legend.set_yticklabels([])
        legend.spines["polar"].set_visible(False)
    else:
        from matplotlib.lines import Line2D
        custom_lines = []
        for lix, c_ori in enumerate(class_list):
            custom_lines.append(Line2D([0], [0], color=colorvals[lix], lw=4))
        legend = fig.add_axes(legend_axes)
        legend.legend(custom_lines, class_list, loc='center')
        legend.axis('off')
        
        
def configs_to_english(config, sconfigs, direction_label, class_list):
    if any('stim_dur' in sconfigs[cf].keys() for cf in sconfigs.keys()): 
        stimdurs = sorted(list(set([sconfigs[cf]['stim_dur'] for cf in sconfigs.keys()])))
        vary_stimdur = True
        if len(stimdurs) > 1:
            full_dur = stimdurs[-1]
            half_dur = stimdurs[-2]
            if len(stimdurs) > 2:
                quarter_dur = stimdurs[0]
    else:
        stimdurs = None
        vary_stimdur = False
        
        
    rot_duration = 'full'
    rot_direction = 'static'
    if config == '':
        starting_rot = ''
    else:
        starting_rot = sconfigs[config][direction_label]
        
        if vary_stimdur:
            if sconfigs[config]['stim_dur'] == half_dur:
                rot_duration = 'half'
            elif sconfigs[config]['stim_dur'] == quarter_dur:
                rot_duration = 'quarter'
                
        if direction_label == 'ori':
            if sconfigs[config]['direction']==1:
                rot_direction = 'CW'
            else:
                rot_direction = 'CCW'
            starting_rot = sconfigs[config]['ori']
        else:
            if direction_label=='xpos':
                if sconfigs[config]['xpos'] < 0:
                    rot_direction = 'rightward'
                else:
                    rot_direction = 'leftward'
                starting_rot = sconfigs[config]['xpos']
            elif direction_label == 'yrot':
                if sconfigs[config]['yrot'] == -1:
                    rot_direction = 'posneg'
                else:
                    rot_direction = 'reverse'
                starting_rot = class_list[0]
    
    return starting_rot, rot_direction, rot_duration


#%%


def plot_decoded_traces(cf, traces=None, trained_classes=[], pdirections=None,
                        direction_label='', sconfigs=dict(), labels_df=None, 
                        decode=True, shuffle_frames=False, plot_trials=True):
    
    if decode:
        mean_traces = dict((k, np.mean(v, axis=-1)) for k, v in traces.items())
        sem_traces = dict((k, stats.sem(v, axis=-1)) for k, v in traces.items())
    else:
        # Just plotting responses, so instead of dim 2 = classes, dim2 is ROIs:
        assert pdirections is not None, "For plotting response traces, must provide ROIs sorted by preferred feature."

            
    # Order subplots by class-order in dynamic stimulus:
    class_list, class_indices = get_ordered_classes(cf, sconfigs, trained_classes, direction_label)
    
    # PLOT ------------------------------------------------------
    fig, axes = pl.subplots(len(class_list), 1, figsize=(6,15))
    
    # Draw stimulus frames in the color of the predicted label:
    nframes_in_trial, nclasses_total, ntrials_curr_config = traces[cf].shape
    stim_on = list(set(labels_df[labels_df['config']==cf]['stim_on_frame']))[0]
    nframes_on = list(set(labels_df[labels_df['config']==cf]['nframes_on']))[0]
    stim_frames = np.arange(stim_on, stim_on+nframes_on)    
    
    for lix, (class_label, class_index) in enumerate(zip(class_list, class_indices)):
        color_index = train_labels.index(class_label)
        if decode:
            mean_trace = mean_traces[cf][:, class_index]
            std_trace = sem_traces[cf][:, class_index]
            rois_preferred = class_index
        else:
            rois_preferred = np.array([pdir[0] for pdir in pdirections if pdir[1]==class_label])
            print "Found %i cells with preferred dir %i" % (len(rois_preferred), class_label)
            roi_traces = traces[cf][:, rois_preferred, :] # Test mat is:  Nframes x Nrois x Ntrials
            mean_trace = np.mean(roi_traces, axis=-1)
            std_trace = np.std(roi_traces, axis=-1)
        
        
        axes[lix].plot(np.arange(0, stim_frames[0]), 
                            mean_trace[0:stim_frames[0]],
                            color='k', linewidth=mean_lw, alpha=1.0)
        axes[lix].plot(stim_frames, 
                            mean_trace[stim_frames], 
                            color=colorvals[color_index], linewidth=mean_lw, alpha=1.0)
        axes[lix].plot(np.arange(stim_frames[-1], nframes_in_trial), 
                            mean_trace[stim_frames[-1]:], 
                            color='k', linewidth=mean_lw, alpha=1.0)
        
        if decode and 'bas' in trained_classes:
            null_index = [v for v in trained_classes].index('bas')
            axes[lix].plot(np.arange(0, nframes_in_trial), 
                            mean_traces[cf][:, null_index], 
                            color='gray', linestyle=':', linewidth=mean_lw, alpha=1.0)

        if plot_trials:
            plot_type = 'trials'
            for trialn in range(ntrials_curr_config):
                axes[lix].plot(np.arange(0, stim_frames[0]+1), 
                                    traces[cf][0:stim_frames[0]+1, rois_preferred, trialn],
                                    color='k', linewidth=trial_lw, alpha=0.5)
                axes[lix].plot(stim_frames, 
                                    traces[cf][stim_frames[0]:stim_frames[-1]+1, rois_preferred, trialn], 
                                    color=colorvals[color_index], linewidth=trial_lw, alpha=0.5)
                axes[lix].plot(np.arange(stim_frames[-1], nframes_in_trial), 
                                    traces[cf][stim_frames[-1]:, rois_preferred, trialn], 
                                    color='k', linewidth=trial_lw, alpha=0.5)
                
        else:
            plot_type = 'fillstd'
            if len(mean_trace.shape) == 1:
                axes[lix].fill_between(np.arange(0, nframes_in_trial), 
                                    mean_trace + std_trace,
                                    mean_trace - std_trace, 
                                    alpha=0.2, color='k')
            else:
                for l in range(mean_trace.shape[-1]):
                    axes[lix].fill_between(np.arange(0, nframes_in_trial), 
                                    mean_trace[:, l] + std_trace[:, l],
                                    mean_trace[:, l] - std_trace[:, l], 
                                    alpha=0.2, color='k')
        # Plot chance line:
        chance = 1/len(class_list)
        axes[lix].plot(np.arange(0, nframes_in_trial), np.ones((nframes_in_trial,))*chance, 'k--', linewidth=0.5)

    # Format subplots, add labels, legend:
    sns.despine(trim=True, offset=4, ax=axes[lix])
    draw_subplot_stimbars(axes, cf, stim_frames, clf.classes_, class_indices, test_sconfigs, colorvals, 
                          is_static=False)
    format_xaxis_subplots(axes, stim_on, framerate, class_list)
    # Put legend in upper right:
    pl.subplots_adjust(top=0.85)
    draw_legend(fig, class_list, colorvals, polar=polar_legend)
    # Give a sensible title:
    starting_rot, rot_direction, rot_duration = configs_to_english(cf, test_sconfigs, direction_label, class_list)
    config_english = 'start %i [%s, %s]' % (starting_rot, rot_direction, rot_duration)
    pl.suptitle(config_english)
  
    if shuffle_frames:
        figbase = '%s_start_%i_%s_%s_%s_%s_shuffled.png' % (cf, starting_rot, rot_direction, rot_duration, plot_type, test_data_type)
    else:
        figbase = '%s_start_%i_%s_%s_%s_%s.png' % (cf, starting_rot, rot_direction, rot_duration, plot_type, test_data_type)
    
    return fig, figbase


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



rootdir = '/Volumes/coxfs01/2p-data' #'/mnt/odyssey' #'/Volumes/coxfs01/2p-data' #/mnt/odyssey'
animalid = 'CE077'
session = '20180713' #'20180713' #'20180629'
acquisition = 'FOV1_zoom1x'

acquisition_dir = os.path.join(rootdir, animalid, session, acquisition)


# #############################################################################
# Select TRAINING data and classifier:
# -----------------------------------------------------------------------------
train_runid = 'combined_gratings_static' #_static' #'blobs_run2'
train_traceid = 'cnmf_' #'traces001'

train_data_type = 'meanstim' #_plusnull'
input_data_type = 'corrected'
class_desc = '6ori'

# Select TESTING data:
# -----------------------------------------------------------------------------
test_runid = 'gratings_rotating' 
test_traceid =  'cnmf_' 
test_data_type = 'corrected' 

# #############################################################################


if 'cnmf' in train_traceid:
    classif_identifier = 'stat_allrois_LinearSVC_kfold_%s_all_%s_%s' % (class_desc, train_data_type, input_data_type)
else:
    classif_identifier = 'stat_allrois_LinearSVC_kfold_%s_all_%s' % (class_desc, train_data_type)

train_basedir = util.get_traceid_from_acquisition(acquisition_dir, train_runid, train_traceid)
train_fpath = os.path.join(train_basedir, 'classifiers', classif_identifier, '%s_datasets.npz' % classif_identifier)
test_basedir = util.get_traceid_from_acquisition(acquisition_dir, test_runid, test_traceid)
test_fpath = os.path.join(test_basedir, 'data_arrays', 'datasets.npz')
training_data_fpath = os.path.join(train_basedir, 'data_arrays', 'datasets.npz')

data_identifier = '_'.join((animalid, session, acquisition, 
                            train_runid, os.path.split(train_basedir)[-1], 
                            test_runid, os.path.split(test_basedir)[-1]))
print data_identifier


# Set output directories:
# -----------------------
classifier_dir = os.path.join(train_basedir, 'classifiers', classif_identifier)
test_results_dir = os.path.join(classifier_dir, 'testdata')
if not os.path.exists(test_results_dir): os.makedirs(test_results_dir)
train_results_dir = os.path.join(classifier_dir, 'traindata')
if not os.path.exists(train_results_dir): os.makedirs(train_results_dir)


# Get training feature keys for indexing into sconfigs dict()
direction_label = ''.join([s for s in class_desc if not s.isdigit()])
is_grating = False
if 'ori' in class_desc:
    direction_label = 'direction'
    is_grating = True

# If GRATINGS: CW=1, values should be decreasing; CCW = -1, values are increasing
# If XPOS:  direction < 0 -- start LEFT; direction > 0 -- start RIGHT
# If YROT:  direction == 1 -- standard movie (positive to negative); direction == -1 -- _reverse movie  (negative to positive)

#%%
# #############################################################################
# LOAD TRAINING DATA:
# #############################################################################

train_X, train_y, train_labels = load_training_set(train_fpath)
print "Training labels:", train_labels

# Load trained classifier:
svc = joblib.load(os.path.join(train_basedir, 'classifiers', classif_identifier, '%s.pkl' % classif_identifier))

#%  Classifier Parameters: 
use_regression = False
fit_best = True
nfeatures_select = 30 #'all' #50 # 'all' #150 #'all' #50 # 'all' #20 #50 #'all' #75 # 'all' #75

# Fit classifier - get N best features, if applicable:
clf, train_X, kept_rids, roiset = refit_SVC(svc, train_X, train_y, classifier_dir, 
                                            fit_best=fit_best, nfeatures_select=nfeatures_select)

# Create color palette:
if 'bas' in clf.classes_:
    colorvals = sns.color_palette("hls", len(clf.classes_)-1)
else:
    colorvals = sns.color_palette("hls", len(clf.classes_))
    
#%%
# #############################################################################
# LOAD TEST DATA:
# #############################################################################

# Load testing data:
test_X, test_labels_df, test_sconfigs, test_runinfo = get_testing_set(test_fpath, test_data_type=test_data_type)
framerate = test_runinfo['framerate']


#% Data cleanup: --------------------------------------------------------------
threshold = False
min_spikes = 0.0002

downsample = False
downsample_factor = 0.2

const_trans = 'xpos'
trans_value = -5
# -----------------------------------------------------------------------------

# Filter and/or downsample:
if threshold:
    test_X = filter_noise(test_X, test_labels_df, min_spikes=min_spikes)
if downsample:
    test_X, test_labels_df = downsample_data(test_X, test_labels_df, downsample_factor=downsample_factor)
    
# Only grab subset of dataset if relevant:
if const_trans is not '':
    assert trans_value in [test_sconfigs[c][const_trans] for c in test_sconfigs.keys()], \
        "Specified trans_value %i for const_trans = %s, Not found." % (trans_value, const_trans)
    configs_to_test = [c for c in test_sconfigs.keys() if test_sconfigs[c][const_trans] == trans_value]
else:
    configs_to_test = test_sconfigs.keys()
    
label_ixs = np.array(test_labels_df[test_labels_df['config'].isin(configs_to_test)].index.tolist())
test_X = test_X[label_ixs, :]
test_X = test_X[:, kept_rids]
test_labels_df = test_labels_df.iloc[label_ixs].reset_index(drop=True)

# Whiten data:
test_X = StandardScaler().fit_transform(test_X)

# Get config groups from labels matched to input data array:
cgroups = test_labels_df.groupby('config')

#%% # Load parsed MW file to get frame-by-frame stimulus values from test set:

test_rundir = test_runinfo['traceid_dir'].split('/traces')[0]
if rootdir not in test_rundir:
    test_rundir = replace_root(test_rundir, rootdir, animalid, session)
    
paradigm_fpath = glob.glob(os.path.join(test_rundir, 'paradigm', 'files', '*.json'))[0]
with open(paradigm_fpath, 'r') as f:
    mwtrials = json.load(f)

#%%

# #############################################################################
# Test the classifier:
# #############################################################################

shuffle_frames = False
all_predictions_by_class, test_traces = decode_traces(clf, cgroups, test_X, shuffle_frames=shuffle_frames)
mean_predictions_by_class = dict((k, np.mean(v, axis=-1)) for k, v in all_predictions_by_class.items())
sem_predictions_by_class = dict((k, stats.sem(v, axis=-1)) for k, v in all_predictions_by_class.items())


#%%

# #############################################################################
# Plot probability of each trained angle, if using CLASSIFIER:
# #############################################################################

# Set plotting params:
plot_trials = True
if plot_trials:
    mean_lw = 2
    trial_lw=0.2
else:
    mean_lw = 1.0
    trial_lw=0.2  
polar_legend = False


# Set output dir:
decoding_dir = os.path.join(test_results_dir, '%s_%s' % (test_runid, test_data_type))
if const_trans is not None and trans_value is not None:
    trans_string = '%s_%i' % (const_trans, trans_value)
    decoding_dir = '%s_%s' % (decoding_dir, trans_string)
if not os.path.exists(decoding_dir): os.makedirs(decoding_dir)
  
#%

#train_labels = [int(t) for t in train_labels]

#if isinstance(clf, CalibratedClassifierCV):

configs_tested = sorted(list(set(test_labels_df['config'])), key=natural_keys)

for cf in configs_tested:

    fig, figbase = plot_decoded_traces(cf, traces=all_predictions_by_class, 
                                       trained_classes=clf.classes_, 
                                       direction_label=direction_label, 
                                       sconfigs=test_sconfigs, 
                                       labels_df=test_labels_df, 
                                       shuffle_frames=False, decode=True,
                                       plot_trials=True)
    label_figure(fig, data_identifier)
    figname = '0decode_%s_TEST_%s_%s' % (roiset, test_runid, figbase)
    pl.savefig(os.path.join(decoding_dir, figname))
    pl.close()

#
#
#for cf in configs_tested:
#    
#    # Order subplots by class-order in dynamic stimulus:
#    class_list, class_indices = get_ordered_classes(cf, test_sconfigs, clf.classes_, direction_label)
#
#    # PLOT ------------------------------------------------------
#    fig, axes = pl.subplots(len(class_list), 1, figsize=(6,15))
# 
#    # Draw stimulus frames in the color of the predicted label:
#    nframes_in_trial, nclasses_total, ntrials_curr_config = all_predictions_by_class[cf].shape
#    stim_on = list(set(test_labels_df[test_labels_df['config']==cf]['stim_on_frame']))[0]
#    nframes_on = list(set(test_labels_df[test_labels_df['config']==cf]['nframes_on']))[0]
#    stim_frames = np.arange(stim_on, stim_on+nframes_on)
#        
#    for lix, (class_label, class_index) in enumerate(zip(class_list, class_indices)):
#        color_index = train_labels.index(class_label)
#
#        axes[lix].plot(np.arange(0, stim_frames[0]), 
#                            mean_predictions_by_class[cf][0:stim_frames[0], class_index],
#                            color='k', linewidth=mean_lw, alpha=1.0)
#        axes[lix].plot(stim_frames, 
#                            mean_predictions_by_class[cf][stim_frames, class_index], 
#                            color=colorvals[color_index], linewidth=mean_lw, alpha=1.0)
#        axes[lix].plot(np.arange(stim_frames[-1], nframes_in_trial), 
#                            mean_predictions_by_class[cf][stim_frames[-1]:, class_index], 
#                            color='k', linewidth=mean_lw, alpha=1.0)
#        
#        if 'bas' in clf.classes_:
#            null_index = [v for v in clf.classes_].index('bas')
#            axes[lix].plot(np.arange(0, nframes_in_trial), 
#                            mean_predictions_by_class[cf][:, null_index], 
#                            color='gray', linestyle=':', linewidth=mean_lw, alpha=1.0)
#
#        if plot_trials:
#            plot_type = 'trials'
#            for trialn in range(ntrials_curr_config):
#                axes[lix].plot(np.arange(0, stim_frames[0]), 
#                                    all_predictions_by_class[cf][0:stim_frames[0], class_index, trialn],
#                                    color='k', linewidth=trial_lw, alpha=0.5)
#                axes[lix].plot(stim_frames, 
#                                    all_predictions_by_class[cf][stim_frames, class_index, trialn], 
#                                    color=colorvals[color_index], linewidth=trial_lw, alpha=0.5)
#                axes[lix].plot(np.arange(stim_frames[-1], nframes_in_trial), 
#                                    all_predictions_by_class[cf][stim_frames[-1]:, class_index, trialn], 
#                                    color='k', linewidth=trial_lw, alpha=0.5)
#                
#        else:
#            plot_type = 'fillstd'
#            axes[lix].fill_between(range(nframes_in_trial), 
#                                    mean_predictions_by_class[cf][:,class_index]+sem_predictions_by_class[cf][:, class_index],
#                                    mean_predictions_by_class[cf][:,class_index]-sem_predictions_by_class[cf][:, class_index], 
#                                    alpha=0.2, color='k')
#        # Plot chance line:
#        chance = 1/len(class_list)
#        axes[lix].plot(np.arange(0, nframes_in_trial), np.ones((nframes_in_trial,))*chance, 'k--', linewidth=0.5)
#
#    # Format subplots, add labels, legend:
#    sns.despine(trim=True, offset=4, ax=axes[lix])
#    draw_subplot_stimbars(axes, cf, stim_frames, clf.classes_, class_indices, test_sconfigs, colorvals, 
#                          is_static=False)
#    format_xaxis_subplots(axes, stim_on, framerate)
#    # Put legend in upper right:
#    pl.subplots_adjust(top=0.85)
#    draw_legend(fig, class_list, colorvals, polar=polar_legend)
#    # Give a sensible title:
#    starting_rot, rot_direction, rot_duration = configs_to_english(cf, test_sconfigs, direction_label, class_list)
#    config_english = 'start %i [%s, %s]' % (starting_rot, rot_direction, rot_duration)
#    pl.suptitle(config_english)
#    label_figure(fig, data_identifier)
#    
#    #%
#    if shuffle_frames:
#        figname = 'decode_%s_TEST_%s_%s_start_%i_%s_%s_%s_%s_shuffled.png' % (roiset, test_runid, cf, starting_rot, rot_direction, rot_duration, plot_type, test_data_type)
#    else:
#        figname = 'decode_%s_TEST_%s_%s_start_%i_%s_%s_%s_%s.png' % (roiset, test_runid, cf, starting_rot, rot_direction, rot_duration, plot_type, test_data_type)
#    pl.savefig(os.path.join(decoding_dir, figname))
#    pl.close()

#%%
        
# #############################################################################
# Load TRAINING DATA and plot traces:
# #############################################################################

training_data, train_runinfo = load_training_data(training_data_fpath)
train_labels_df = pd.DataFrame(data=training_data['labels_data'], columns=training_data['labels_columns'])

aggregate_configs = True 

#%%
train_data_type = test_data_type #'smoothedX'
if 'df' not in train_data_type and 'DF' not in train_data_type:
    train_data_units = 'intensity'
else:
    train_data_units = 'dff'
    
trainingX = training_data[train_data_type][:, kept_rids]
print trainingX.shape
nrois = trainingX.shape[-1]

train_sconfigs = training_data['sconfigs'][()]
train_feature = ''.join([s for s in class_desc if not s.isdigit()]) #'yrot' #'xpos'

use_dff = False
if use_dff:
    F0 = training_data['F0'][:, kept_rids]


#%% Group training configs by trained feature (may need to aggregate across objects)

train_configs_df = pd.DataFrame(train_sconfigs).T
config_groups = train_configs_df.groupby(train_feature)
configs_by_feature = config_groups.indices

for cval, cindices in configs_by_feature.items():
    configs_by_feature[cval] = ['config%03d' % (i+1) for i in cindices]
    

#%%
train_cgroups = train_labels_df.groupby('config')
trainingX_std = StandardScaler().fit_transform(trainingX)

all_preds_train, train_traces = decode_traces(clf, train_cgroups, trainingX_std, shuffle_frames=False)
mean_preds_train = dict((k, np.mean(v, axis=-1)) for k, v in all_preds_train.items())
sem_preds_train = dict((k, stats.sem(v, axis=-1)) for k, v in all_preds_train.items())


#%% #############################################################################
# Plot DECODED traces for ROIs sorted by ANGLE: TRAINING DATA
# #############################################################################


trained_classes = list(np.copy(train_labels))
trained_classes.append('bas')


if any([isinstance(v, str) for v in clf.classes_]):
    class_indices = [[v for v in clf.classes_].index(str(c)) for c in train_labels]
else:
    class_indices = [[v for v in clf.classes_].index(c) for c in train_labels]
    

fig, axes = pl.subplots(len(train_labels), 1, figsize=(6,15))

for lix, (class_label, class_index) in enumerate(zip(train_labels, class_indices)):
    curr_configs = [k for k,v in train_sconfigs.items() if v[train_feature] == class_label] #[0]
    
    stim_on = list(set(train_labels_df[train_labels_df['config'].isin(curr_configs)]['stim_on_frame']))[0]
    nframes_on = list(set(train_labels_df[train_labels_df['config'].isin(curr_configs)]['nframes_on']))[0]
    stim_frames = np.arange(stim_on, stim_on+nframes_on)

    for vi, class_name in enumerate(clf.classes_):
        ix_in_svc = [i for i in clf.classes_].index(class_name)

        for curr_config in curr_configs:
            nframes_in_trial = all_preds_train[curr_config].shape[0]
            nclasses_total = all_preds_train[curr_config].shape[1]
            ntrials_curr_config =  all_preds_train[curr_config].shape[-1]
            if class_name == 'bas':
                axes[lix].plot(np.arange(0, nframes_in_trial), 
                                mean_preds_train[curr_config][:, int(ix_in_svc)], 
                                color='gray', linestyle=':')
            else:
                color_index = train_labels.index(int(class_name))
                axes[lix].plot(np.arange(0, nframes_in_trial), 
                                mean_preds_train[curr_config][:, int(ix_in_svc)], 
                                color=colorvals[color_index], alpha=0.8)
            

    axes[lix].axes.xaxis.set_ticks([])
                
    bar_offset = 0.1
    curr_ylim = axes[lix].get_ylim()[0] - bar_offset
    
    #axes[lix].set_ylim([minval-bar_offset*2, maxval])
    
    # Plot chance line:
    chance = 1/len(trained_classes)
    axes[lix].plot(np.arange(0, nframes_in_trial), np.ones((nframes_in_trial,))*chance, 'k--', linewidth=0.5)
    
                
# Format subplots, add labels, legend:
sns.despine(trim=True, offset=4)
draw_subplot_stimbars(axes, '', stim_frames, clf.classes_, class_indices, train_sconfigs, colorvals, is_static=True)
format_xaxis_subplots(axes, stim_on, framerate, class_list)
# Put legend in upper right:
pl.subplots_adjust(top=0.85)
draw_legend(fig, clf.classes_, colorvals, polar=polar_legend)
# Give a sensible title:
_, rot_direction, rot_duration = configs_to_english('', test_sconfigs, direction_label, class_list)
pl.suptitle('%s: Decoding training set (all configs: %s - %s)' % (roiset, rot_direction, rot_duration))
label_figure(fig, data_identifier)

figname = 'decode_%s_TRAIN_traces_%s_%s_%s.png' % (roiset, train_runid, train_feature, train_data_type)
pl.savefig(os.path.join(train_results_dir, figname))



#%% FORMAT TRAINING DATA for easier plotting and comparison with true TESTING data:



threshold = False
threshold_factor = 0.5
min_spikes = 0.0005

if threshold:
    figs_append = '_filtered'
else:
    figs_append = ''
    
#if fit_best:

# Get trial structure:
assert len(list(set(train_labels_df['nframes_on']))) == 1, "More than 1 nframes_on found in TRAIN set..."
train_nframes_on = list(set(train_labels_df['nframes_on']))[0]
assert len(list(set(train_labels_df['stim_on_frame']))) == 1, "More than 1 stim_on_frame val found in TRAIN set..."
train_stim_on = list(set(train_labels_df['stim_on_frame']))[0]
ntrials_by_cond = [v for k,v in train_runinfo['ntrials_by_cond'].items()]
assert len(list(set(ntrials_by_cond)))==1, "More than 1 rep values found in TRAIN set"
ntrials_per_cond = list(set(ntrials_by_cond))[0]


config_list = sorted([c for c in train_sconfigs.keys()], key=lambda x: train_sconfigs[x][train_feature])
train_traces = {}
for cf in config_list:
    curr_config = [k for k,v in train_sconfigs.items() if v[train_feature] == class_label][0]
    
    stim_on = list(set(train_labels_df[train_labels_df['config']==curr_config]['stim_on_frame']))[0]
    nframes_on = list(set(train_labels_df[train_labels_df['config']==curr_config]['nframes_on']))[0]
    stim_frames = np.arange(stim_on, stim_on+nframes_on)

    nframes_per_trial = np.squeeze(all_preds_train[curr_config]).shape[0]
    ntrials_per_cond =  np.squeeze(all_preds_train[curr_config]).shape[-1]


    cixs = train_labels_df[train_labels_df['config']==cf].index.tolist()
    curr_frames = trainingX[cixs, :]
    
    tmat = np.reshape(curr_frames, (nframes_per_trial, ntrials_per_cond, nrois), order='f') 
    tmat = np.swapaxes(tmat, 1, 2) # Nframes x Nrois x Ntrials (to match test_traces)
    if train_data_type == 'spikes' and threshold:
        print "... thresholding spikes"
        tmat[tmat <= min_spikes] = 0.
        
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
if train_feature == 'ori':
    thetas = [train_sconfigs[cf][direction_label] * (math.pi/180) for cf in config_list]
    use_polar = True
else:
    thetas = [train_sconfigs[cf][train_feature] for cf in config_list]
    use_polar = False

import matplotlib as mpl

if nfeatures_select == 50:
    nrows = 5 #4
    ncols = 10 #5
elif nfeatures_select == 80:
    nrows = 8
    ncols = 10
else:
    nrows = 4
    ncols = 5

fig, axes = pl.subplots(figsize=(12,12), nrows=nrows, ncols=ncols, subplot_kw=dict(polar=use_polar))

for ridx, ax in zip(range(len(kept_rids)), axes.flat):
    #print ridx
    radii = responses[:, ridx]
    if radii.min() < 0:
        print "%i - fixing offset" % ridx
        radii -= radii.min()

    if use_polar:
        ax.grid(True)
        polygon = mpl.patches.Polygon(zip(thetas, radii), fill=True, alpha=0.5, color='mediumorchid')
        ax.add_line(polygon)
        #ax.autoscale()
        ax.set_theta_zero_location("N")
        ax.set_xticklabels([])
        max_angle_ix = np.where(radii==radii.max())[0][0]
        ax.plot([0, thetas[max_angle_ix]], [0, radii[max_angle_ix]],'k', lw=1)
    else:
        for val in np.unique(thetas):
            ixs = np.where(thetas == val)[0]
            ax.plot(val, np.mean(radii[ixs]), '_', color='mediumorchid', markeredgewidth=2, markersize=10)
        ax.scatter(thetas, radii, s=2, c='mediumorchid', alpha=0.5)        
    radii[np.isnan(radii)] = 0

    ax.set_title(kept_rids[ridx]+1, fontsize=8)
    ax.set_xticks([t for t in thetas])
pl.suptitle("tuning: %s" % roiset)
sns.despine(trim=True, offset=4)
fig.text(0.5, 0.04, '%s' % train_feature, ha='center')
fig.text(0.04, 0.5, '%s' % train_data_units, va='center', rotation='vertical')

if not use_polar:
    pl.subplots_adjust(wspace=0.4, hspace=0.4)
pl.rc('ytick', labelsize=8)
label_figure(fig, data_identifier)

figname = '%s_tuning_plots_%s%s.png' % (roiset, train_data_type, figs_append)
pl.savefig(os.path.join(train_results_dir, figname))
    
#%%

# Sort by direction preference:
# -----------------------------------------------------------------------------

if len(thetas) != len(np.unique(thetas)):
    # Either aggregate configs with shared featuer value, or keep separate
    if aggregate_configs:
        responses_by_feature=[]; feature_names=[];
        for val in sorted(np.unique(thetas)):
            ixs = np.where(thetas==val)[0]
            avg_configs = np.mean(responses[ixs,:], axis=0)
            responses_by_feature.append(avg_configs)
            feature_names.append(val)
        responses_by_feature = np.array(responses_by_feature)
    else:
        responses_by_feature = responses.copy()
        feature_names = np.copy(thetas)
        
#thetas = [train_configs[cf][direction_label] for cf in config_list]

pdirections = []
for ridx in range(nrois):
    #print ridx
    radii = responses_by_feature[:, ridx]
    if radii.min() < 0:
        print "ROI %i has negative response. SKIPPING." % (kept_rids[ridx]+1)
        continue
        #radii -= radii.min()
    radii[np.isnan(radii)] = 0
    max_angle_ix = np.where(radii==radii.max())[0][0]

    pdirections.append((ridx, feature_names[max_angle_ix], radii[max_angle_ix]))
    
sorted_rois = sorted([pdir[0] for pdir in pdirections], key=lambda x: pdirections[x][1])

#for ridx in sorted_rois:
#    print ridx, pdirections[ridx][1]

#%%
    
# #############################################################################
# Plot mean traces for ROIs sorted by ANGLE: TRAINING DATA
# #############################################################################

config_list = sorted([c for c in configs_by_feature.keys()]) # key=lambda x: train_configs[x][direction_label])
class_indices = np.arange(0, len(config_list))

fig, axes = pl.subplots(len(config_list), 1, figsize=(6,15))
for lix, (class_name, class_index) in enumerate(zip(config_list, class_indices)):
    color_index = train_labels.index(class_name)

    #grand_meantrace_across_rois = np.mean(np.mean(traces[class_label], axis=-1), axis=0)
    rois_preferred = [pdir[0] for pdir in pdirections if pdir[1]==class_name] #train_configs[class_label][direction_label]]
    print "Found %i cells with preferred dir %i" % (len(rois_preferred), class_name) #train_configs[class_label][direction_label])
    stim_on = list(set(train_labels_df[train_labels_df['config'].isin(configs_by_feature[class_name])]['stim_on_frame']))[0]
    nframes_on = list(set(train_labels_df[train_labels_df['config'].isin(configs_by_feature[class_name])]['nframes_on']))[0]
    stim_frames = np.arange(stim_on, stim_on+nframes_on)
        
    for ridx in rois_preferred:
        if aggregate_configs:
            roi_traces = np.mean(np.dstack(train_traces[class_label][:, ridx, :] for class_label in configs_by_feature[class_name]), axis=1)
        else:
            roi_traces = train_traces[class_label][:, ridx, :]
        mean_roi_trace = np.mean(roi_traces, axis=-1)
        std_roi_trace = np.std(roi_traces, axis=-1)
        ntrials_curr_config = roi_traces.shape[-1] # original array is:  Ntrials x Nframes x Nrois

        axes[lix].plot(np.arange(0, stim_frames[0]), mean_roi_trace[0:stim_frames[0]],
                            color='k', linewidth=mean_lw, alpha=1.0)
        axes[lix].plot(stim_frames, mean_roi_trace[stim_frames], 
                            color=colorvals[color_index], linewidth=mean_lw, alpha=1.0)
        axes[lix].plot(np.arange(stim_frames[-1], len(mean_roi_trace)), mean_roi_trace[stim_frames[-1]:], 
                            color='k', linewidth=mean_lw, alpha=1.0)

        # PLOT std:
        if plot_trials:
            plot_type = 'trials'
            for trialn in range(ntrials_curr_config):
                axes[lix].plot(np.arange(0, stim_frames[0]), roi_traces[0:stim_frames[0], trialn],
                                    color='k', linewidth=trial_lw, alpha=0.3)
                
                axes[lix].plot(stim_frames, roi_traces[stim_frames, trialn], 
                                    color=colorvals[color_index], linewidth=trial_lw, alpha=0.3)
                
                axes[lix].plot(np.arange(stim_frames[-1], len(mean_roi_trace)), roi_traces[stim_frames[-1]:, trialn], 
                                    color='k', linewidth=trial_lw, alpha=0.3)
        else:
            axes[lix].fill_between(range(len(mean_roi_trace)), mean_roi_trace+std_roi_trace,
                                    mean_roi_trace-std_roi_trace, alpha=0.2,
                                    color='k')
    #if lix < len(config_list):
        #axes[lix].axes.xaxis.set_visible(True) #([])
        #axes[lix].axes.xaxis.set_ticks([])
        
    axes[lix].set_title(str([kept_rids[r]+1 for r in rois_preferred]), fontsize=8)


# Format subplots, add labels, legend:
sns.despine(trim=True, offset=4)
draw_subplot_stimbars(axes, '', stim_frames, clf.classes_, class_indices, train_sconfigs, colorvals, is_static=True)
format_xaxis_subplots(axes, stim_on, framerate, class_list)
# Put legend in upper right:
pl.subplots_adjust(top=0.85)
draw_legend(fig, clf.classes_, colorvals, polar=polar_legend)
pl.suptitle('Training set responses')

label_figure(fig, data_identifier)

if aggregate_configs:
    aggregate_str = '_AGG%s' % train_feature
else:
    aggregate_str = ''
figname = '%s_train_traces_%s_%s%s%s.png' % (roiset, train_runid, train_data_type, aggregate_str, figs_append)
pl.savefig(os.path.join(train_results_dir, figname))


#%%

# #############################################################################
# Plot mean traces for ROIs sorted by ANGLE: TESTING DATA
# #############################################################################

# Create decoder-style plots for ROI responses to test data
# Each group of cells in preferred-direction group serves as "classifier"

if 'DF' in test_data_type:
    yaxis_label = 'df/f'
else:
    yaxis_label = 'intensity'
    
response_classif_dir = os.path.join(train_results_dir, 'test_data_responses')
if not os.path.exists(response_classif_dir):
    os.makedirs(response_classif_dir)
    
    
configs_tested = sorted(list(set(test_labels_df['config'])), key=natural_keys)
#maxval = 0.5 #max([all_preds[config].max() for config in mean_pred.keys()])

for test_config in configs_tested:

    fig, figbase = plot_decoded_traces(test_config, traces=test_traces, 
                                       trained_classes=clf.classes_, 
                                       direction_label=direction_label, 
                                       sconfigs=test_sconfigs, 
                                       labels_df=test_labels_df, 
                                       shuffle_frames=False, decode=False,
                                       pdirections=pdirections,
                                       plot_trials=False)
    label_figure(fig, data_identifier)
    figname = '0%s_traces_TEST_%s_%s.png' % (roiset, test_runid, figbase)
    pl.savefig(os.path.join(response_classif_dir, figname))
    pl.close()
    
#
#
#for test_config in configs_tested:
#    #%
#    print test_config
#    # Plot each CLASS's probability on a subplot:
#    # -----------------------------------------------------------------------------
#    stim_on = list(set(test_labels_df[test_labels_df['config']==test_config]['stim_on_frame']))[0]
#    nframes_on = list(set(test_labels_df[test_labels_df['config']==test_config]['nframes_on']))[0]
#    stim_frames = np.arange(stim_on, stim_on+nframes_on)
#        
#    angle_step = list(set(np.diff(train_labels)))[0]
#    class_list, class_indices = get_ordered_classes(cf, test_sconfigs, clf.classes_, direction_label)
#    nframes_in_trial, nclasses_total, ntrials_curr_config = all_predictions_by_class[test_config].shape
#
#    #%
#    fig, axes = pl.subplots(len(class_list), 1, figsize=(6,15))
#    
#    for lix, (class_label, class_index) in enumerate(zip(class_list, class_indices)):
#        
#        color_index = train_labels.index(class_label)
#        rois_preferred = np.array([pdir[0] for pdir in pdirections if pdir[1]==class_label])
#        
#        print "Found %i cells with preferred dir %i" % (len(rois_preferred), class_label)
#        
#        #for ridx in rois_preferred:
#        roi_traces = test_traces[test_config][:, rois_preferred, :] # Test mat is:  Nframes x Nrois x Ntrials
#        mean_roi_trace = np.mean(roi_traces, axis=-1)
#        std_roi_trace = np.std(roi_traces, axis=-1)
#            
#        axes[lix].plot(np.arange(0, stim_frames[0]), 
#                        mean_roi_trace[0:stim_frames[0]], color='k', 
#                        linewidth=1.5, alpha=1.0)
#        axes[lix].plot(stim_frames, 
#                        mean_roi_trace[stim_frames], color=colorvals[color_index], 
#                        linewidth=1.5, alpha=1.0)
#        axes[lix].plot(np.arange(stim_frames[-1]+1, len(mean_roi_trace)), 
#                        mean_roi_trace[stim_frames[-1]+1:], color='k', 
#                        linewidth=1.5, alpha=1.0)
#
#        if plot_trials:
#            plot_type = 'trials'
#            for trialn in range(ntrials_curr_config):
#                axes[lix].plot(np.arange(0, stim_frames[0]+1),
#                                test_traces[test_config][0:stim_frames[0]+1, rois_preferred, trialn],
#                                color='k', linewidth=0.4, alpha=0.3)
#                
#                axes[lix].plot(stim_frames, 
#                                test_traces[test_config][stim_frames[0]:stim_frames[-1]+1, rois_preferred, trialn], 
#                                color=colorvals[color_index], linewidth=0.4, alpha=0.3)
#                
#                axes[lix].plot(np.arange(stim_frames[-1], nframes_in_trial), 
#                                test_traces[test_config][stim_frames[-1]:, rois_preferred, trialn], 
#                                color='k', linewidth=0.4, alpha=0.3)
#                
#        else:
#            plot_type = 'fillstd'
#            axes[lix].fill_between(range(nframes_in_trial), mean_roi_trace + std_roi_trace,
#                                        mean_roi_trace - std_roi_trace, 
#                                        alpha=0.2, color='k')
#
#        axes[lix].set_title(str([kept_rids[r]+1 for r in rois_preferred]))
#        axes[lix].axes.xaxis.set_ticks([])
#
#        # Plot chance line:
#        chance = 1/len(class_list)
#        axes[lix].plot(np.arange(0, nframes_in_trial), np.ones((nframes_in_trial,))*chance, 'k--', linewidth=0.5)
#                  
#    # Format subplots, add labels, legend:
#    sns.despine(trim=True, offset=4, ax=axes[lix])
#    draw_subplot_stimbars(axes, cf, stim_frames, clf.classes_, class_indices, test_sconfigs, colorvals, 
#                          is_static=False)
#    format_xaxis_subplots(axes, stim_on, framerate)
#    # Put legend in upper right:
#    pl.subplots_adjust(top=0.85)
#    draw_legend(fig, clf.classes_, colorvals, polar=polar_legend)
#    # Give a sensible title:
#    starting_rot, rot_direction, rot_duration = configs_to_english(cf, test_sconfigs, direction_label, class_list)
#    config_english = 'start %i [%s, %s]' % (starting_rot, rot_direction, rot_duration)
#    pl.suptitle(config_english)
#    label_figure(fig, data_identifier)
#    #%
    
#    if shuffle_frames:
#        figname = '%s_traces_TEST_%s_%s_start_%i_%s_%s_%s_%s_shuffled.png' % (roiset, test_runid, test_config, starting_rot, rot_direction, rot_duration, plot_type, test_data_type)
#    else:
#        figname = '%s_traces_TEST_%s_%s_start_%i_%s_%s_%s_%s.png' % (roiset, test_runid, test_config, starting_rot, rot_direction, rot_duration, plot_type, test_data_type)
#    
#    pl.savefig(os.path.join(response_classif_dir, figname))
#    pl.close()
#    
