#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 15:03:10 2018

@author: juliana
"""

import os
import numpy as np
import pylab as pl
import seaborn as sns
from random import shuffle
import pandas as pd
import json
import glob
import math

from pipeline.python.paradigm import utils as util
from pipeline.python.utils import natural_keys, replace_root

from scipy import stats
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVR #LinearSVR



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



rootdir = '/mnt/odyssey'
animalid = 'CE077'
session = '20180704'
acquisition = 'FOV1_zoom1x'

acquisition_dir = os.path.join(rootdir, animalid, session, acquisition)


# #############################################################################
# Select TRAINING data and classifier:
# #############################################################################
train_runid = 'gratings_phasemod' #'blobs_run2'
train_traceid = 'traces001'

classif_identifier = 'stat_allrois_LinearSVC_kfold_4ori_all_meanstim'

#classif_identifier = 'stat_allrois_LinearSVC_kfold_5morphlevel_all_meanstim'

clf_pts = classif_identifier.split('_')
decoder = clf_pts[4][1:]
print "Decoding: %s" % decoder

# LOAD TRAINING DATA:
# -------------------
train_basedir = util.get_traceid_from_acquisition(acquisition_dir, train_runid, train_traceid)
train_fpath = os.path.join(train_basedir, 'classifiers', classif_identifier, '%s_datasets.npz' % classif_identifier)
train_dset = np.load(train_fpath)

train_dtype = 'cX_std'

train_X = train_dset[train_dtype]
train_y = train_dset['cy']


#if orientations:
#    tmp_y = train_y.copy()
#    train_y = 

#replace_mnum = np.where(cy == 53)[0] # make this 11
#cy[replace_mnum] = 11

train_labels = sorted(list(set(train_y)))
print "Training labels:", train_labels
# #############################################################################

#%%

from sklearn.feature_selection import RFE


use_regression = True
fit_best = False
nfeatures_select = 'all' #75 # 'all' #75

# FIT CLASSIFIER: #############################################################
if train_X.shape[0] > train_X.shape[1]: # nsamples > nfeatures
    dual = False
else:
    dual = True

if 'LinearSVC' in classif_identifier:
    if use_regression is False:
        svc = LinearSVC(random_state=0, dual=dual, multi_class='ovr', C=1000) #, C=best_C) # C=big_C)
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
        output_dir = os.path.join(train_basedir, 'classifiers', classif_identifier, 'testdata')
        
    else:
        print "Using regression..."
        clf = SVR(kernel='linear', C=1, tol=1e-9)
        clf.fit(train_X, train_y)
    
        classif_identifier_new = classif_identifier.replace('LinearSVC', 'SVRegression')
        
        output_dir = os.path.join(train_basedir, 'classifiers', classif_identifier_new, 'testdata')


#if not os.path.exists(output_dir):
#    os.makedirs(output_dir)
#print "Saving TEST results to %s" % output_dir        
# #############################################################################



#%%
        
    
full_configs = [c for c in test_configs.keys() if test_configs[c]['stim_dur'] == 8]
print full_configs    
xdata = test_dataset['smoothedX']
ixs = np.array(labels_df[labels_df['config'].isin(full_configs)].index.tolist())
train_x = xdata[ixs, :]
print "train X:", train_x.shape

# Label full-rotation trial(s) to regress out angle from train trials:


#    currtrials = list(set(labels_df[labels_df['config']==config]['trial']))
#    currconfig_angles = np.mean(np.vstack([mwtrials[trial]['rotation_values'] for trial in currtrials]), axis=0)
#    
#    start_rot = currconfig_angles[0]
#    end_rot = currconfig_angles[-1]
#    
#    if start_rot == 360 and end_rot > start_rot:
#        start_rot -= 360
#        end_rot -= 360
#    elif start_rot == 0 and end_rot < start_rot:
#        start_rot += 180
#        end_rot += 180
#    elif start_rot == -90:
#        start_rot += 180
#        end_rot += 180
#    
#    
#    stim_on = list(set(labels_df[labels_df['config']==config]['stim_on_frame']))[0]
#    nframes_on = list(set(labels_df[labels_df['config']==config]['nframes_on']))[0]
#    nframes_in_trial = np.squeeze(mean_pred[config]).shape[0]
#    
#    interp_angles = np.ones(predicted_vals.shape) * np.nan
#    interp_angles[stim_on:stim_on + nframes_on] = np.linspace(start_rot, end_rot, num=nframes_on)


    
#%%

# #############################################################################
# Select TESTING data:
# #############################################################################
test_runid = 'gratings_rotating_phasemod' #'blobs_dynamic_run6' #'blobs_dynamic_run1' #'blobs_dynamic_run1'
test_traceid = 'traces001'

output_dir = os.path.join(output_dir, test_runid)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
print "Saving TEST results to:\n%s" % output_dir      


#%%

# LOAD TEST DATA:
# -----------------------------------------------------------------------------

test_data_type = 'smoothedX' #'smoothedX' #'smoothedX' # 'corrected' #'smoothedX' #'smoothedDF'

#rootdir = '/mnt/odyssey'
#animalid = 'CE077'
#session = '20180523'
#acquisition = 'FOV1_zoom1x'
#test_dtype = 'meanstim'


#acquisition_dir = os.path.join(rootdir, animalid, session, acquisition)

test_basedir = util.get_traceid_from_acquisition(acquisition_dir, test_runid, test_traceid)
test_fpath = os.path.join(test_basedir, 'data_arrays', 'datasets.npz')
test_dataset = np.load(test_fpath)
assert test_data_type in test_dataset.keys(), "Specified d-type (%s) not found. Choose from: %s" % (test_data_type, str(test_dataset.keys()))
assert len(test_dataset[test_data_type].shape)>0, "D-type is empty!"

# _, smoothed_X, _ = util.load_roiXtrials_df(test_dataset['run_info'][()]['traceid_dir'], trace_type='processed', dff=False, 
#                                                smoothed=True, frac=0.01, quantile=0.08)
#


#%%

# Format TEST data:

X_test = test_dataset[test_data_type]

#X_test = smoothed_X
#X_test = np.array(smoothed_X)
#X_test = X_test[:, xrange(0, cX_std.shape[-1])]

X_test = StandardScaler().fit_transform(X_test)

test_configs = test_dataset['sconfigs'][()]
labels_df = pd.DataFrame(data=test_dataset['labels_data'], columns=test_dataset['labels_columns'])

# just look at 1 config for now:
cgroups = labels_df.groupby('config')

#%%
# Get rotation values:
test_rundir = test_dataset['run_info'][()]['traceid_dir'].split('/traces')[0]
if rootdir not in test_rundir:
    test_rundir = replace_root(test_rundir, rootdir, animalid, session)
    
paradigm_fpath = glob.glob(os.path.join(test_rundir, 'paradigm', 'files', '*.json'))[0]
with open(paradigm_fpath, 'r') as f:
    mwtrials = json.load(f)


#%%
shuffle_frames = False

mean_pred = {}
sem_pred = {}
all_preds = {}
test_traces = {}
predicted = []
for k,g in cgroups:

    y_proba = []; tvals = [];
    for kk,gg in g.groupby('trial'):
        #print kk
        
        trial_ixs = gg.index.tolist()
        if shuffle_frames:
            shuffle(trial_ixs)

        curr_test = X_test[trial_ixs,:]
        if fit_best:
            curr_test = curr_test[:, kept_rids]
            roiset = 'best%i' % nfeatures_select
        else:
            roiset = 'all'
            
        if isinstance(clf, SVR):
            curr_proba = clf.predict(curr_test)
        else:
            curr_proba = clf.predict_proba(curr_test)
        
        y_proba.append(curr_proba)
        tvals.append(curr_test)
        
    y_proba = np.dstack(y_proba)
    curr_traces = np.dstack(tvals)
    
    means_by_class = np.mean(y_proba, axis=-1)
    stds_by_class = stats.sem(y_proba, axis=-1) #np.std(y_proba, axis=-1)
        
        
    mean_pred[k] = means_by_class
    sem_pred[k] = stds_by_class
    all_preds[k] = y_proba
    test_traces[k] = curr_traces
    

#
#fig, axes = pl.subplots(1, len(mean_pred.keys()) + 1)
#for ax,config in zip(axes.flat, sorted(mean_pred.keys())):
#    for lix in range(8):
#        ax.plot(range(len(mean_pred[config])), mean_pred[config][:,lix], color=colorvals[lix])
#
#        #ax.errorbar(range(len(mean_pred[morph])), mean_pred[morph][:,lix], yerr=sem_pred[morph][:,lix], color=colorvals[lix])

#%%

# #############################################################################
# Plot SIN of predicted angle, if using regression:
# #############################################################################

plot_std = True
predicted_color = 'cornflowerblue'
true_color = 'k'
use_sin = False

all_angles = sorted(list(set([test_configs[c]['ori'] for c in test_configs.keys()])))


if isinstance(clf, SVR):
    configs_tested = sorted(list(set(labels_df['config'])), key=natural_keys)
    
    row1_configs = sorted([c for c in configs_tested if test_configs[c]['direction'] == 1], key=lambda x: test_configs[x]['stim_dur'])
    row2_configs = sorted([c for c in configs_tested if test_configs[c]['direction'] == -1], key=lambda x: test_configs[x]['stim_dur'])
    
    
    plot_configs = sorted(configs_tested, key=lambda x: (test_configs[x]['direction'], test_configs[x]['ori'], test_configs[x]['stim_dur']))
        
    pix = 0
    nconfigs = len(configs_tested)
    
    fig, axes = pl.subplots(nrows=2, ncols=nconfigs, sharex=True, sharey=True, squeeze=False, figsize=(5*len(row1_configs),6))
    
    
    for config in plot_configs:
        
        if config in row1_configs:
            rindex = row1_configs.index(config)
            ax = axes[0][rindex]
        else:
            rindex = row2_configs.index(config)
            ax = axes[1][rindex]
        #ax = axes[pix]
        
        #predicted_vals = np.array([np.sin( ang * (math.pi / 180.) ) + np.cos( ang * (math.pi / 180.) ) for ang in np.squeeze(mean_pred[config])])
        if use_sin:
            predicted_vals = np.array([np.sin( ang * (math.pi / 180.) )for ang in np.squeeze(mean_pred[config])])
            std_vals = np.array([np.sin( ang * (math.pi / 180.) ) for ang in np.squeeze(sem_pred[config])])
        else:
            predicted_vals = np.array([ang for ang in np.squeeze(mean_pred[config])])
            std_vals = np.array([ang for ang in np.squeeze(sem_pred[config])])
            
        #std_vals = np.squeeze(sem_pred[config])
        
        currtrials = list(set(labels_df[labels_df['config']==config]['trial']))
        currconfig_angles = np.mean(np.vstack([mwtrials[trial]['rotation_values'] for trial in currtrials]), axis=0)
        
        start_rot = currconfig_angles[0]
        end_rot = currconfig_angles[-1]
        
        if start_rot == 360 and end_rot > start_rot:
            start_rot -= 360
            end_rot -= 360
        elif start_rot == 0 and end_rot < start_rot:
            start_rot += 180
            end_rot += 180
        elif start_rot == -90:
            start_rot += 180
            end_rot += 180
        
        
        stim_on = list(set(labels_df[labels_df['config']==config]['stim_on_frame']))[0]
        nframes_on = list(set(labels_df[labels_df['config']==config]['nframes_on']))[0]
        nframes_in_trial = np.squeeze(mean_pred[config]).shape[0]
        
        interp_angles = np.ones(predicted_vals.shape) * np.nan
        interp_angles[stim_on:stim_on + nframes_on] = np.linspace(start_rot, end_rot, num=nframes_on)

        if use_sin:
            true_vals = np.array([np.sin( ang * (math.pi / 180.) ) if not np.isnan(ang) else np.nan for ang in interp_angles])
        else:
            if do_scatter:
#                colorvals = sns.color_palette("hsv", len(svc.classes_))
#                stim_on = list(set(labels_df[labels_df['config']==config]['stim_on_frame']))[0]
#                nframes_on = list(set(labels_df[labels_df['config']==config]['nframes_on']))[0]
#                stim_frames = np.arange(stim_on, stim_on+nframes_on)
                # cvals = [colorvals[1] if frameix in stim_frames else (0.0, 0.0, 0.0) for frameix in range(nframes_in_trial)]
                # pl.figure(); pl.scatter(true_vals, predicted_vals, c=cvals)
                for ai, ang in enumerate(interp_angles):
                    if ai <= stim_on and np.isnan(ang):
                        true_vals[ai] = interp_angles[stim_on]
                    elif ai >= (stim_on + nframes_on - 1) and np.isnan(ang):
                        print ai
                        true_vals[ai] = interp_angles[stim_on+nframes_on-1]
                    else:
                        true_vals[ai] = interp_angles[ai]
                    
                
            true_vals = np.array([ang if not np.isnan(ang) else np.nan for ai,ang in enumerate(interp_angles)])

        #print "True:", true_vals.shape
        #print "Pred:", predicted_vals.shape
        
                
        if pix < 7:
            ax.axes.xaxis.set_visible(False) #([])
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
    
        ax.plot(range(nframes_in_trial), predicted_vals, label='predicted', color=predicted_color)
        if plot_std:
            ax.fill_between(range(nframes_in_trial), predicted_vals+std_vals, predicted_vals-std_vals,
                                        color=predicted_color, alpha=0.2)
        ax.plot(range(nframes_in_trial), true_vals, label='true', color=true_color)
        if test_configs[config]['direction'] == -1:
            CWstring = 'CCW'
        else:
            CWstring = 'CW'
            
        pix += 1
        
        ax.set_title('start %i, %s (%is)' % (test_configs[config]['ori'], CWstring, test_configs[config]['stim_dur']))
        
    sns.despine(offset=4, trim=True, bottom=True, left=True)
    pl.legend(loc=9, bbox_to_anchor=(0, 0.1), ncol=2)
    
    pl.savefig(os.path.join(output_dir, 'predicted_v_true.png'))


#%%%



#pl.figure()
#colorvals = sns.color_palette("hsv", len(svc.classes_))
#stim_on = list(set(labels_df[labels_df['config']==config]['stim_on_frame']))[0]
#nframes_on = list(set(labels_df[labels_df['config']==config]['nframes_on']))[0]
#stim_frames = np.arange(stim_on, stim_on+nframes_on)
#
#cvals = [colorvals[class_index] if frameix in stim_frames else (0,0,0) for frameix in range(nframes_in_trial)]
#
#pl.scatter(range(nframes_in_trial), mean_pred[config][:, class_index], c=cvals)


#%%
    
# #############################################################################
# Plot probability of each trained angle, if using CLASSIFIER:
# #############################################################################

plot_trials = True

    
if isinstance(clf, CalibratedClassifierCV):
    
    print [c for c in test_configs if test_configs[c]['ori']==0 and test_configs[c]['direction']==1]
    
    #config = 'config004' #'config001' -- start 180, CW
    #config = 'config008' # -- start 180, CCW
    
    configs_tested = sorted(list(set(labels_df['config'])), key=natural_keys)
    maxval = max([all_preds[config].max() for config in mean_pred.keys()])
    
    for config in configs_tested:
        #%
        print config
            
        # Plot each CLASS's probability on a subplot:
        # -----------------------------------------------------------------------------
        #colorvals = sns.color_palette("hsv", len(svc.classes_))
        colorvals = sns.color_palette("hsv", len(svc.classes_))
        stim_on = list(set(labels_df[labels_df['config']==config]['stim_on_frame']))[0]
        nframes_on = list(set(labels_df[labels_df['config']==config]['nframes_on']))[0]
        stim_frames = np.arange(stim_on, stim_on+nframes_on)
        
    
        if test_configs[config]['ori'] == 180:
            if test_configs[config]['direction'] == 1: #CW
                class_list = [180, 135, 90, 45, 0, 315, 270, 225] #225, 270, 315, 0, 45, 90, 135]
            else: #test_configs[config]['direction'] == -1: #CCW
                class_list = [180, 225, 270, 315, 0, 45, 90, 135]
        elif test_configs[config]['ori'] == 0:
            if test_configs[config]['direction'] == 1: #CW
                #angle_cycles = [0, 315, 270, 225, 180, 135, 90, 45]  #225, 270, 315, 0, 45, 90, 135]
                class_list = [0, 135, 90, 45]
            else: # test_configs[config]['direction'] == -1: #CCW
                #class_list = [0, 45, 90, 135, 180, 225, 270, 315]
                class_list = [0, 45, 90, 135]
        elif test_configs[config]['ori'] == -90:
            if test_configs[config]['direction'] == 1: #CW
                #angle_cycles = [0, 315, 270, 225, 180, 135, 90, 45]  #225, 270, 315, 0, 45, 90, 135]
                class_list = [90, 45, 0, 135]
            else: # test_configs[config]['direction'] == -1: #CCW
                #class_list = [0, 45, 90, 135, 180, 225, 270, 315]
                class_list = [90, 135, 0, 45]
        elif test_configs[config]['ori'] == 90:
            if test_configs[config]['direction'] == 1: #CW
                #angle_cycles = [0, 315, 270, 225, 180, 135, 90, 45]  #225, 270, 315, 0, 45, 90, 135]
                class_list = [90, 45, 0, 135]
            else: # test_configs[config]['direction'] == -1: #CCW
                #class_list = [0, 45, 90, 135, 180, 225, 270, 315]
                class_list = [90, 135, 0, 45]
                
        #class_list = [0, 45, 90, 135]
        class_indices = [[v for v in svc.classes_].index(c) for c in class_list]
        
        fig, axes = pl.subplots(len(class_list), 1, figsize=(8,15))
        #for lix in range(8):
        #    curr_ori = svc.classes_[lix]
        nframes_in_trial = np.squeeze(all_preds[config]).shape[0]
        nclasses_total = np.squeeze(all_preds[config]).shape[1]
        ntrials_curr_config =  np.squeeze(all_preds[config]).shape[-1]

        for lix, (class_label, class_index) in enumerate(zip(class_list, class_indices)):
            print lix
    #        axes[lix].plot(range(nframes_in_trial), mean_pred[config][:, class_index], color=colorvals[class_index])
            cvals = [colorvals[class_index] if frameix in stim_frames else (0.0, 0.0, 0.0) for frameix in range(nframes_in_trial)]
    
            #axes[lix].scatter(range(nframes_in_trial), mean_pred[config][:, class_index], c=cvals, s=1)
            axes[lix].plot(np.arange(0, stim_frames[0]), mean_pred[config][0:stim_frames[0], class_index],
                                color='k', linewidth=1.0, alpha=1.0)
            
            axes[lix].plot(stim_frames, mean_pred[config][stim_frames, class_index], 
                                color=colorvals[class_index], linewidth=1.0, alpha=1.0)
            
            axes[lix].plot(np.arange(stim_frames[-1]+1, nframes_in_trial), mean_pred[config][stim_frames[-1]+1:, class_index], 
                                color='k', linewidth=0.5, alpha=1.0)
    
            if plot_trials:
                plot_type = 'trials'
                for trialn in range(ntrials_curr_config):
    #                axes[lix].plot(range(nframes_in_trial), all_preds[config][:, class_index, trialn], 
    #                                    color=colorvals[class_index], linewidth=0.5)
                    axes[lix].plot(np.arange(0, stim_frames[0]), all_preds[config][0:stim_frames[0], class_index, trialn],
                                        color='k', linewidth=0.2, alpha=0.5)
                    
                    axes[lix].plot(stim_frames, all_preds[config][stim_frames, class_index, trialn], 
                                        color=colorvals[class_index], linewidth=0.2, alpha=0.5)
                    
                    axes[lix].plot(np.arange(stim_frames[-1]+1, nframes_in_trial), all_preds[config][stim_frames[-1]+1:, class_index, trialn], 
                                        color='k', linewidth=0.2, alpha=0.5)
                    
            else:
                plot_type = 'fillstd'
    #            axes[lix].fill_between(range(nframes_in_trial), mean_pred[config][:,class_index]+sem_pred[config][:, class_index],
    #                                        mean_pred[config][:,class_index]-sem_pred[config][:, class_index], alpha=0.2,
    #                                        color=colorvals[class_index])
                axes[lix].fill_between(range(nframes_in_trial), mean_pred[config][:,class_index]+sem_pred[config][:, class_index],
                                            mean_pred[config][:,class_index]-sem_pred[config][:, class_index], alpha=0.2,
                                            color='k')
                
            #axes[lix].set_frame_on(False)
            
            axes[lix].set_ylim([0, maxval])
            if lix < 7:
                axes[lix].axes.xaxis.set_visible(True) #([])
            axes[lix].axes.xaxis.set_ticks([])
            #axes[lix].set_title(class_label)
                
        sns.despine(trim=True, offset=4, bottom=True)
        
        # show stim dur:
    
        pl.plot([stim_on, stim_on+nframes_on], np.ones((2,1))*0, 'r', linewidth=3)
        
        # Custom legend:
        from matplotlib.lines import Line2D
        custom_lines = []
        for lix, c_ori in enumerate(svc.classes_):
            custom_lines.append(Line2D([0], [0], color=colorvals[lix], lw=4))
        pl.legend(custom_lines, svc.classes_, loc=9, bbox_to_anchor=(0.5, -0.2), ncol=len(svc.classes_))
        
        #%
        if test_configs[config]['direction']==1:
            rot_direction = 'CW'
        else:
            rot_direction = 'CCW'
            
        if test_configs[config]['stim_dur'] == 4:
            rot_duration = 'half'
        elif test_configs[config]['stim_dur'] == 2:
            rot_duration = 'quarter'
        else:
            rot_duration = 'full'
        starting_rot = test_configs[config]['ori']
        config_english = 'start %i [%s, %s]' % (starting_rot, rot_direction, rot_duration)
        pl.suptitle(config_english)
        
        #%
        #train_savedir = os.path.split(train_fpath)[0]
        
        
        if shuffle_frames:
            figname = '%s_hsv_TEST_%s_%s_start_%i_%s_%s_%s_%s_shuffled.png' % (roiset, test_runid, config, starting_rot, rot_direction, rot_duration, plot_type, test_data_type)
        else:
            figname = '%s_hsv_TEST_%s_%s_start_%i_%s_%s_%s_%s.png' % (roiset, test_runid, config, starting_rot, rot_direction, rot_duration, plot_type, test_data_type)
        pl.savefig(os.path.join(output_dir, figname))
        #pl.close()
    



#%%
# #############################################################################
# Plot grand-mean traces pair-wise between conditions:
# #############################################################################
        
        
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator

#fig, axes = pl.subplots(n_classes, ntrials_per_stim, sharex=True, sharey=True, figsize=(20,20))
config = 'config006'
nframes_in_trial = max([all_preds[config].shape[0] for config in all_preds.keys()])

nclasses_total = all_preds[config].shape[1]
ntrials_curr_config = all_preds[config].shape[2]

configs_tested = sorted(list(set(labels_df['config'])), key=natural_keys)

    
# Figure settings:
nr = len(configs_tested)
nc = len(configs_tested)
aspect = 1
bottom = 0.1; left=0.1
top=1.-bottom; right = 1.-left
fisasp = 1 # (1-bottom-(1-top))/float( 1-left-(1-right) )
wspace=0.15  # widthspace, relative to subplot size; set to zero for no spacing
hspace=wspace/float(aspect)
figheight= 8 # fix the figure height
figwidth = (nc + (nc-1)*wspace)/float((nr+(nr-1)*hspace)*aspect)*figheight*fisasp



# Colormap settings:
tag = np.arange(0, nframes_in_trial)
cmap = pl.cm.jet                                        # define the colormap
cmaplist = [cmap(i) for i in range(cmap.N)]             # extract all colors from the .jet map
#cmaplist[0] = (.5,.5,.5,1.0)                            # force the first color entry to be grey
cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)  # create the new map
bounds = [int(i) for i in np.linspace(0,len(tag),len(tag)+1)]             # define the bins and normalize
norm = mpl.colors.BoundaryNorm(boundaries=bounds, ncolors=cmap.N)




fig, axes = pl.subplots(nr, nc, sharex=True, sharey=True, figsize=(figwidth, figheight))
pl.subplots_adjust(top=top, bottom=bottom, left=left, right=right,
                    wspace=wspace, hspace=hspace)


plot_configs = sorted(configs_tested, key=lambda x: (test_configs[x]['stim_dur'], test_configs[x]['ori'], test_configs[x]['direction']))
#plot_configs = svc.classes_

for row in range(nr):
    row_config = plot_configs[row]
    for col in range(nc):
        col_config = plot_configs[col]
        row_label = '%i_%s_%i' % (test_configs[row_config]['ori'], test_configs[row_config]['direction'], test_configs[row_config]['stim_dur'])
        col_label = '%i_%s_%i' % (test_configs[col_config]['ori'], test_configs[col_config]['direction'], test_configs[col_config]['stim_dur'])
        ax = axes[row][col]
        
#        traceX = np.mean(np.mean(test_traces[row_config], axis=2), axis=1)
#        traceY = np.mean(np.mean(test_traces[col_config], axis=2), axis=1)
#        traceX = svc.coef_[row, :].dot(np.mean(test_traces[row_config], axis=2).T) + svc.intercept_[row]
#        traceY = svc.coef_[col, :].dot(np.mean(test_traces[col_config], axis=2).T) + svc.intercept_[col]
        traceX = np.mean(test_traces[row_config], axis=1)
        traceY = np.mean(test_traces[col_config], axis=1)

        for rep in range(traceX.shape[1]):
            currx = traceX[:, rep]
            curry = traceY[:, rep]
            
            if currx.shape[0] > curry.shape[0]:
                curry = np.pad(curry, (0, currx.shape[0]-curry.shape[0]), mode='constant', constant_values=np.nan)
            elif traceY.shape[0] > traceX.shape[0]:
                currx = np.pad(currx, (0, curry.shape[0]-currx.shape[0]), mode='constant', constant_values=np.nan)
            
        
            im = ax.scatter(currx, curry, c=tag[0:len(currx)], cmap=cmap, norm=norm, vmax=nframes_in_trial, s=1, alpha=0.1)

        if col == 0:
            ax.set(ylabel=row_label)
        if row == nr-1:
            ax.set(xlabel = col_label)
        #ax.set_xlim(minval, maxval)
        #ax.set_ylim(minval, maxval)
        ax.set(aspect='equal') #adjustable='box-forced', aspect='equal')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=3))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=3))



    















#%%



#######
y_test = test_dataset['ylabels']
if test_data_type == 'meanstim':
    y_test = np.reshape(y_test, (test_dataset['run_info'][()]['ntrials_total'], test_dataset['run_info'][()]['nframes_per_trial']))[:,0]

sconfigs_test = test_dataset['sconfigs'][()]
runinfo_test = test_dataset['run_info'][()]
trial_nframes_test = runinfo_test['nframes_per_trial']

all_tsecs = np.reshape(test_dataset['tsecs'][:], (sum(runinfo_test['ntrials_by_cond'].values()), trial_nframes_test))
print all_tsecs.shape

stimtype = 'gratings'


data_subset = 'all' #'single' # 'xpos-5' # 'alltrials' #'xpos-5'
const_trans = '' #'xpos'
trans_value = '' #-5

class_name = decoder

if 'all' in data_subset:
    if stimtype == 'gratings':
        test_labels = np.array([sconfigs_test[c][class_name] for c in y_test])  # y_test.copy()
        colorvals = sns.color_palette("cubehelix", len(svc.classes_))

    else:
        test_labels = np.array([sconfigs_test[c][class_name] for c in y_test])
        colorvals = sns.color_palette("PRGn", len(svc.classes_))

    test_data = X_test.copy()
    tsecs = all_tsecs.copy()
# Get subset of test data (position matched, if trained on y-rotation at single position):
elif 'single' in data_subset:
    assert len([i for i in sconfigs_test.keys() if sconfigs_test[i][const_trans]==trans_value]) > 0, "Specified stim config does not exist (const_trans=%s, trans_value=%s)" % (str(const_trans), str(trans_value))
    #included = np.array([yi for yi, yv in enumerate(y_test) if sconfigs_test[yv][const_trans]==trans_value]) # and sconfigs_test[yv]['yrot']==1])
    included = np.array([yi for yi, yv in enumerate(y_test) if sconfigs_test[yv][const_trans]==trans_value \
                         and sconfigs_test[yv]['morphlevel'] > 0]) # and sconfigs_test[yv]['yrot']==1])

    test_labels = np.array([sconfigs_test[c][class_name] for c in y_test[included]])
    test_data = X_test[included, :]
    sub_tsecs = test_dataset['tsecs'][included]
    tsecs = np.reshape(sub_tsecs, (len(sub_tsecs)/trial_nframes_test, trial_nframes_test))
    #tsecs = all_tsecs[included,:] # Only need 1 column, since all ROIs have the same tpoints
print "TEST: %s, labels: %s" % (str(test_data.shape), str(test_labels.shape))


#object_ids = [label for label in list(set(test_labels)) if label in train_labels]
object_ids = sorted([label for label in list(set(test_labels))])

#object_ids = [-1, 0, 53, 22]

print "LABELS:", object_ids


###############################################################################

mean_pred = {}
sem_pred = {}
predicted = []
for tru in sorted(list(set(test_labels))):
    trial_ixs = np.where(test_labels==tru)[0]
    curr_test = X_test[trial_ixs,:]
    y_proba = clf.predict_proba(curr_test)
    
    means_by_class = np.mean(y_proba, axis=0)
    stds_by_class = np.std(y_proba, axis=0)
    
    
    mean_pred[tru] = means_by_class
    sem_pred[tru] = stds_by_class

fig, axes = pl.subplots(1, len(mean_pred.keys()))
for ax,morph in zip(axes.flat, sorted(mean_pred.keys())):
    ax.errorbar(xrange(len(mean_pred[morph])), mean_pred[morph], yerr=sem_pred[morph])
    
    
    
# Get probabilities as a function of time #####################################
colorvals = sns.color_palette("PRGn", len(clf.classes_))
if len(colorvals) == 3:
    # Replace the middle!
    colorvals[1] = 'gray'

mean_pred = {}
sem_pred = {}
all_pred= {}
timesecs= {}

for curr_obj_id in object_ids:
    
    frame_ixs = np.where(test_labels==curr_obj_id)[0]
    n_test_trials = int(len(frame_ixs) / float(trial_nframes_test))
    #print curr_obj_id, n_test_trials
    
    test_trialmat_fixs = np.reshape(frame_ixs, (n_test_trials, trial_nframes_test))  # Reshape indices into nframes x ntrials array:
    #data_r = np.reshape(test_data[frame_ixs,:], (n_test_trials, trial_nframes_test, test_data.shape[-1]))
    predicted = []
    for ti in range(n_test_trials):
        true_labels = test_labels[test_trialmat_fixs[ti]]  # True labels for frames of current trial
        test = test_data[test_trialmat_fixs[ti], :]        # Test data (predict labels for each frame of current trial) -- nframes_per_trial x nrois
        tpoints = tsecs[ti, :] #tsecs[test_trialmat_fixs[ti]]            # Time stamps for frames of current trial
        y_proba = clf.predict_proba(test)                  # Predict label
        #pl.plot(tpoints, y_proba[:,0], color=colorvals[0])
        #pl.plot(tpoints, y_proba[:,4], color=colorvals[4])
        predicted.append(y_proba)
    predicted = np.array(predicted)
    mean_pred[curr_obj_id] = np.mean(predicted, axis=0)
    sem_pred[curr_obj_id] = stats.sem(predicted, axis=0)
    timesecs[curr_obj_id] = tpoints #np.mean(tsecs, axis=1) #np.array(tpoints)
    #all_pred[curr_obj_id] = np.array(predicted)
    
    
# PLOT ########################################################################


    
sns.set_style('white')

# Plot AVERAGE of movie conditions:
fig, axes = pl.subplots(1, len(object_ids), figsize=(20,8), sharey=True)
for pi,obj in enumerate(object_ids):
    for ci in range(mean_pred[obj].shape[1]):
        tpoints = timesecs[obj]
        stim_on = np.where(tpoints==0)[0]
        stim_bar = np.array([tpoints[int(stim_on)], tpoints[int(stim_on+round(runinfo_test['nframes_on']))]])
        
        preds_mean = mean_pred[obj][:, ci]
        preds_sem = sem_pred[obj][:, ci]
        
        axes[pi].plot(tpoints, preds_mean, color=colorvals[ci], label=clf.classes_[ci], linewidth=2)
        axes[pi].fill_between(tpoints, preds_mean-preds_sem, preds_mean+preds_sem, color=colorvals[ci], alpha=0.2)
        axes[pi].get_xaxis().set_visible(False)
        
        # stimulus bar:
        axes[pi].plot(stim_bar, np.ones(stim_bar.shape) * 0.3, 'k')
        
    axes[pi].set(title=obj)
    # Shrink current axis's height by 10% on the bottom
    box = axes[pi].get_position()
    axes[pi].set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.8])
    if pi == 0:
        axes[pi].set(ylabel='avg p')
        

sns.despine(offset=True, bottom=True)

# Put a legend below current axis
pl.legend(loc='lower center', bbox_to_anchor=(-1., -.3),
          fancybox=True, shadow=False, ncol=len(clf.classes_))
pl.suptitle('Prob of trained classes on trial frames')

#output_dir = os.path.join(acquisition_dir, 'tests', 'figures')
#if not os.path.exists(output_dir):
#    os.makedirs(output_dir)

#pl.ylim(0.3, 0.7)

figname = 'TRAIN_%s_%s_C_%s_TEST_%s_%s_%s_%s_darkercols.png' % (train_runid, train_traceid, classif_identifier, test_runid, test_traceid, data_subset, test_data_type)
print figname


classifier_dir = os.path.join(train_basedir, 'classifiers', classif_identifier)
pl.savefig(os.path.join(classifier_dir, figname))








#%%

roi_selector = 'all' #'all' #'selectiveanova' #'selective'
data_type = 'stat' #'zscore' #zscore' # 'xcondsub'
inputdata = 'meanstim'

testX, testy, test_labels = format_classifier_data(test_dataset, data_type=test_dtype, 
                                                       class_name=class_name,
                                                       subset=subset,
                                                       aggregate_type=aggregate_type,
                                                       const_trans=const_trans,
                                                       trnas_value=trans_value,
                                                       relabel=relabel)

print "Possible test labels:", test_labels
print "Known trained labels:", class_labels 

svc = LinearSVC(random_state=0, dual=dual, multi_class='ovr', C=big_C)
cX_train, cX_test, cy_train, cy_test = train_test_split(cX_std, cy, test_size=0.25, shuffle=True)
svc.fit(cX_train, cy_train)


m100 = 22
choices = []
for tlabel in test_labels:
    if tlabel in class_labels:
        curr_ypred = svc.predict(cX_test)
        avg_pred = np.mean([1 if pred==tru else 0 for i,(pred,tru) in enumerate(zip(curr_ypred, cy_test))])
        
    else:
        tdata_ixs = np.where(testy == tlabel)[0]
        curr_ytest = testy[tdata_ixs]
        curr_xtest = testX[tdata_ixs,:]
        # Look at accuracy for each trial?
        curr_ypred = svc.predict(curr_xtest)
        avg_pred = np.mean([1 if p==m100 else 0 for p in curr_ypred])
    choices.append((tlabel, avg_pred))
    
# plot choice percentage:
pl.figure()
m100 = 22
if m100==22:
    choices[0] = (0, 1-choices[-1][1])


choices = morph_test['choices']
fig, ax = pl.subplots(1) #pl.figure();
pl.plot([c[0] for c in choices], [c[1] for c in choices], 'ro', markersize=10)
pl.ylabel('perc. choose %i' % m100)
ax.set_xticks([c[0] for c in choices])
pl.xlabel('morph level')
pl.ylim([0, 1])
sns.despine(offset=4, trim='bottom')

pl.savefig(os.path.join(classifier_dir, 'pchoose_%i_test25perc.pdf' % m100))

from scipy.optimize import curve_fit
def sigmoid(x, x0, k):
     #y = chance + (1 - chance) / (1 + np.exp(-k*(x-x0)))
     y = 0.5 + (0.5)/(1. + np.exp(-k*(x-x0)))
     return y

xs = [c[0] for c in choices]
tofit = [c[1] for c in choices]

popt, pcov = curve_fit(sigmoid, xs, tofit)
print popt
x = np.linspace(-1, 15, 50)
chance_level = 0.5
y = sigmoid(x, *popt)

pl.figure()
pl.plot(xs, tofit, 'ro', markersize=10, label='data')
pl.plot(x,y, 'k', linewidth=2, label='fit')
pl.ylim(0, 1.05)
pl.legend(loc='best')
pl.show()


morph_test = {'train_x': cX_train,
              'train_y': cy_train,
              'test_x': cX_test,
              'test_y': cy_test,
              'choices': choices,
              'svc': svc}

with open(os.path.join(classifier_dir, 'pchoose_%i_samedataset.pkl'), 'wb') as f:
    pkl.dump(morph_test, f, protocol=pkl.HIGHEST_PROTOCOL)

with open(os.path.join(classifier_dir, 'pchoose_%i_samedataset.pkl' % m100), 'rb') as f:
    morph_test = pkl.load(f)
#%




#%%
# Try using XPOS data for test set:
opts_static = ['-D', '/mnt/odyssey', '-i', 'CE077', '-S', '20180523', '-A', 'FOV1_zoom1x',
           '-T', 'np_subtracted', '--no-pupil',
           '-R', 'blobs_run1', '-t', 'traces001',
           '-n', '1']

opts_dynamic = ['-D', '/mnt/odyssey', '-i', 'CE077', '-S', '20180602', '-A', 'FOV1_zoom1x',
           '-T', 'np_subtracted', '--no-pupil',
           '-R', 'blobs_dynamic_run6', '-t', 'traces001',
           '-n', '1']

traceid_dir_static = util.get_traceid_dir(opts_static)
data_basedir_static = os.path.join(traceid_dir_static, 'data_arrays')


svc = LinearSVC(random_state=0, dual=dual, multi_class='ovr', C=big_C)
svc.fit(cX_std, cy)


# First check if processed datafile exists:
static_fpath = os.path.join(data_basedir_static, 'datasets.npz')
static = np.load(static_fpath)


testdata_X = static['meanstim']
testdata_Xstd = StandardScaler().fit_transform(testdata_X)
testdata_y = np.reshape(static['ylabels'], (static['run_info'][()]['ntrials_total'], static['run_info'][()]['nframes_per_trial']))[:,0]
sconfigs_test = static['sconfigs'][()]

test_morphs = sorted(list(set(np.array([sconfigs_test[cv]['morphlevel'] for cv in sconfigs_test.keys()]))))


pos_values = [-15, -10, -5, 0, 5]
all_choices = {}
choices = dict((k, []) for k in test_morphs)

for pos in pos_values:
    testX, testy = group_classifier_data(testdata_Xstd, testdata_y, 'morphlevel', sconfigs_test, 
                                   subset=None,
                                   aggregate_type='single', 
                                   const_trans='xpos', 
                                   trans_value=pos, 
                                   relabel=False)

    testy = np.array([sconfigs_test[cv]['morphlevel'] for cv in testy])
    
    test_labels = sorted(list(set(testy)))
    print "Possible test labels:", test_labels
    print "Known trained labels:", class_labels 

    for tlabel in test_labels:
        tdata_ixs = np.where(testy == tlabel)[0]
        curr_ytest = testy[tdata_ixs]
        curr_xtest = testX[tdata_ixs,:]
        # Look at accuracy for each trial?
        curr_ypred = svc.predict(curr_xtest)
        choices[tlabel].append(curr_ypred)

averages = dict((k, {}) for k in choices.keys())
for mlevel in choices.keys():
    mean_choices = [np.mean([1 if p==m100 else 0 for p in choices[mlevel][posi]]) for posi in range(len(choices[mlevel]))]
    averages[mlevel]['mean'] = np.mean(mean_choices)
    averages[mlevel]['sem'] = stats.sem(mean_choices)

# plot choice percentage:
means = [averages[mlevel]['mean'] for mlevel in sorted(averages.keys())]
sems = [averages[mlevel]['sem'] for mlevel in sorted(averages.keys())]


fig, ax = pl.subplots(1) #pl.figure();

pl.errorbar(sorted(averages.keys()), means, yerr=sems)
pl.ylabel('perc. choose %i' % m100)
ax.set_xticks(sorted(averages.keys()))
pl.xlabel('morph level')
pl.ylim([0.25, 0.75])
sns.despine(offset=4, trim=True)
pl.savefig(os.path.join(classifier_dir, 'pchoose_%i_xposdata_avgxpos.pdf' % m100))

#%%

#%% Assign data:


X_test = test_dataset[test_data_type]

#X_test = smoothed_X
X_test = np.array(smoothed_X)
X_test = X_test[:, xrange(0, cX_std.shape[-1])]


X_test = StandardScaler().fit_transform(X_test)
y_test = test_dataset['ylabels']
if test_data_type == 'meanstim':
    y_test = np.reshape(y_test, (test_dataset['run_info'][()]['ntrials_total'], test_dataset['run_info'][()]['nframes_per_trial']))[:,0]

sconfigs_test = test_dataset['sconfigs'][()]
runinfo_test = test_dataset['run_info'][()]
trial_nframes_test = runinfo_test['nframes_per_trial']

all_tsecs = np.reshape(test_dataset['tsecs'][:], (sum(runinfo_test['ntrials_by_cond'].values()), trial_nframes_test))
print all_tsecs.shape




data_subset = 'all' # 'xpos-5' # 'alltrials' #'xpos-5'

if 'all' in data_subset:
    if stimtype == 'gratings':
        test_labels = np.array([sconfigs_test[c][class_name] for c in y_test])  # y_test.copy()
        colorvals = sns.color_palette("cubehelix", len(svc.classes_))

    else:
        test_labels = np.array([sconfigs_test[c][class_name] for c in y_test])
        colorvals = sns.color_palette("PRGn", len(svc.classes_))

    test_data = X_test.copy()
    tsecs = all_tsecs.copy()
# Get subset of test data (position matched, if trained on y-rotation at single position):
else:
    included = np.array([yi for yi, yv in enumerate(y_test) if sconfigs_test[yv]['xpos']==-5]) # and sconfigs_test[yv]['yrot']==1])
    test_labels = np.array([sconfigs_test[c][class_name] for c in y_test[included]])
    test_data = X_test[included, :]
    sub_tsecs = test_dataset['tsecs'][included]
    tsecs = np.reshape(sub_tsecs, (len(sub_tsecs)/trial_nframes_test, trial_nframes_test))
    #tsecs = all_tsecs[included,:] # Only need 1 column, since all ROIs have the same tpoints
print "TEST: %s, labels: %s" % (str(test_data.shape), str(test_labels.shape))


#object_ids = [label for label in list(set(test_labels)) if label in train_labels]
object_ids = sorted([label for label in list(set(test_labels))])

#object_ids = [-1, 0, 53, 22]

print "LABELS:", object_ids

mean_pred = {}
sem_pred = {}
predicted = []
for tru in sorted(list(set(test_labels))):
    trial_ixs = np.where(test_labels==tru)[0]
    curr_test = X_test[trial_ixs,:]
    y_proba = clf.predict_proba(curr_test)
    
    means_by_class = np.mean(y_proba, axis=0)
    stds_by_class = np.std(y_proba, axis=0)
    
    
    mean_pred[tru] = means_by_class
    sem_pred[tru] = stds_by_class

fig, axes = pl.subplots(1, len(mean_pred.keys()))
for ax,morph in zip(axes.flat, sorted(mean_pred.keys())):
    ax.errorbar(xrange(3), mean_pred[morph], yerr=sem_pred[morph])
    
    
    
# Get probabilities as a function of time #####################################
colorvals = sns.color_palette("PRGn", len(clf.classes_))
if len(colorvals) == 3:
    # Replace the middle!
    colorvals[1] = 'gray'

mean_pred = {}
sem_pred = {}
all_pred= {}
timesecs= {}

for curr_obj_id in object_ids:
    
    frame_ixs = np.where(test_labels==curr_obj_id)[0]
    n_test_trials = int(len(frame_ixs) / float(trial_nframes_test))
    #print curr_obj_id, n_test_trials
    
    test_trialmat_fixs = np.reshape(frame_ixs, (n_test_trials, trial_nframes_test))  # Reshape indices into nframes x ntrials array:
    #data_r = np.reshape(test_data[frame_ixs,:], (n_test_trials, trial_nframes_test, test_data.shape[-1]))
    predicted = []
    for ti in range(n_test_trials):
        true_labels = test_labels[test_trialmat_fixs[ti]]  # True labels for frames of current trial
        test = test_data[test_trialmat_fixs[ti], :]        # Test data (predict labels for each frame of current trial) -- nframes_per_trial x nrois
        tpoints = tsecs[ti, :] #tsecs[test_trialmat_fixs[ti]]            # Time stamps for frames of current trial
        y_proba = clf.predict_proba(test)                  # Predict label
        #pl.plot(tpoints, y_proba[:,0], color=colorvals[0])
        #pl.plot(tpoints, y_proba[:,4], color=colorvals[4])
        predicted.append(y_proba)
    predicted = np.array(predicted)
    mean_pred[curr_obj_id] = np.mean(predicted, axis=0)
    sem_pred[curr_obj_id] = stats.sem(predicted, axis=0)
    timesecs[curr_obj_id] = tpoints #np.mean(tsecs, axis=1) #np.array(tpoints)
    #all_pred[curr_obj_id] = np.array(predicted)
    
    
# PLOT ########################################################################
    
sns.set_style('white')
# Plot AVERAGE of movie conditions:
fig, axes = pl.subplots(1, len(object_ids), figsize=(20,8), sharey=True)
for pi,obj in enumerate(object_ids):
    for ci in range(mean_pred[obj].shape[1]):
        tpoints = timesecs[obj]
        preds_mean = mean_pred[obj][:, ci]
        preds_sem = sem_pred[obj][:, ci]
        
        axes[pi].plot(tpoints, preds_mean, color=colorvals[ci], label=clf.classes_[ci], linewidth=2)
        axes[pi].fill_between(tpoints, preds_mean-preds_sem, preds_mean+preds_sem, color=colorvals[ci], alpha=0.2)
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
pl.suptitle('Prob of trained classes on trial frames')

#output_dir = os.path.join(acquisition_dir, 'tests', 'figures')
#if not os.path.exists(output_dir):
#    os.makedirs(output_dir)

pl.ylim(0, 0.7)

figname = 'TRAIN_%s_%s_C_%s_TEST_%s_%s_%s_%s_darkercols.png' % (train_runid, train_traceid, classif_identifier, test_runid, test_traceid, data_subset, test_data_type)
print figname

pl.savefig(os.path.join(classifier_dir, figname))

#%%






pl.figure()
dfunc = svc.decision_function(cX_std)
sns.heatmap(dfunc, cmap='hot')
pl.title('decision function')
figname = 'TRAIN_%s_%s_C_%s_decisionfunction.png' % (train_runid, train_traceid, classif_identifier)
pl.savefig(os.path.join(output_dir, figname))



pl.figure()
sns.heatmap(svc.decision_function(test_data), cmap='hot')
pl.title('decision function -- test data')
figname = 'TRAIN_%s_%s_C_%s_TEST_%s_%s_%s_decisionfunction.png' % (train_runid, train_traceid, classif_identifier, test_runid, test_traceid, data_subset)
pl.savefig(os.path.join(output_dir, figname))





pl.figure()
sns.heatmap(svc.decision_function(stimvalues.T), cmap='hot')
pl.title('decision function - mean stim values (test set)')
figname = 'TRAIN_%s_%s_C_%s_TEST_%s_%s_%s_decisionfunction_meanstim_testvals.png' % (train_runid, train_traceid, classif_identifier, test_runid, test_traceid, data_subset)
pl.savefig(os.path.join(output_dir, figname))


