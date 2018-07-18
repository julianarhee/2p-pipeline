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
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR, LinearSVR



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
session = '20180629' #'20180713' #'20180629'
acquisition = 'FOV1_zoom1x'

acquisition_dir = os.path.join(rootdir, animalid, session, acquisition)


# #############################################################################
# Select TRAINING data and classifier:
# #############################################################################
train_runid = 'gratings_drifting' #'blobs_run2'
train_traceid = 'traces001'

classif_identifier = 'stat_allrois_LinearSVC_kfold_8ori_all_meanstim'

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


use_regression = False
fit_best = True
nfeatures_select = 50 #'all' #75 # 'all' #75



#%%

from sklearn.feature_selection import RFE

# FIT CLASSIFIER: #############################################################
if train_X.shape[0] > train_X.shape[1]: # nsamples > nfeatures
    dual = False
else:
    dual = True

if 'LinearSVC' in classif_identifier:
    if use_regression is False:
        svc = LinearSVC(random_state=0, dual=dual, multi_class='ovr', C=1) #, C=best_C) # C=big_C)
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

# #############################################################################
# Select TESTING data:
# #############################################################################
test_runid = 'gratings_rotating_drifting' #'blobs_dynamic_run6' #'blobs_dynamic_run1' #'blobs_dynamic_run1'
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


#%% # Format TEST data:

X_test_orig = test_dataset[test_data_type]

X_test = StandardScaler().fit_transform(X_test_orig)

test_configs = test_dataset['sconfigs'][()]
labels_df = pd.DataFrame(data=test_dataset['labels_data'], columns=test_dataset['labels_columns'])

# just look at 1 config for now:
cgroups = labels_df.groupby('config')

#%% # Load parsed MW file to get rotation values:

test_rundir = test_dataset['run_info'][()]['traceid_dir'].split('/traces')[0]
if rootdir not in test_rundir:
    test_rundir = replace_root(test_rundir, rootdir, animalid, session)
    
paradigm_fpath = glob.glob(os.path.join(test_rundir, 'paradigm', 'files', '*.json'))[0]
with open(paradigm_fpath, 'r') as f:
    mwtrials = json.load(f)



#%% Fit regressor with FULL ROTATION trials, predict half and quarter turns:

fit_on_full = True
      
if fit_on_full and use_regression:   
    full_configs = [c for c in test_configs.keys() if test_configs[c]['stim_dur'] == 8]
    print full_configs    
    xdata = test_dataset['smoothedX']
    
    # Get indices of data array in which FULL rotation shown:
    ixs = np.array(labels_df[labels_df['config'].isin(full_configs)].index.tolist())
    train_x = xdata[ixs, :]
    print "train X:", train_x.shape
    nrois = train_x.shape[-1]
    
    # Get trials in which FULL rotation shown:
    train_trials = sorted(list(set( labels_df.loc[ixs, 'trial'] )), key=natural_keys)
    
    # For Full rotation trials, same duration:
    nframes_on = list(set(labels_df[labels_df['trial'].isin(train_trials)]['nframes_on']))
    assert len(nframes_on)==1, "More than 1 val found for nframes_on."
    nframes_on = nframes_on[0]
    
    stim_on = list(set(labels_df[labels_df['trial'].isin(train_trials)]['stim_on_frame']))
    assert len(stim_on)==1, "More than 1 val found for stim_on."
    stim_on = stim_on[0]
    
    
    # Get STIMULUS period and labels:
    framelabels = []
    for trial in train_trials:
        mw_rotations = mwtrials[trial]['rotation_values']
        
        start_rot = mw_rotations[0]
        end_rot = mw_rotations[-1]
        
#        if start_rot == 0 and end_rot < start_rot:
#            start_rot += 180
#            end_rot += 180
#    
        stim_on = list(set(labels_df[labels_df['trial']==trial]['stim_on_frame']))[0]
        nframes_on = list(set(labels_df[labels_df['trial']==trial]['nframes_on']))[0]
        nframes_in_trial = labels_df[labels_df['trial']==trial].shape[0]
        interp_angles = np.linspace(start_rot, end_rot, num=nframes_on)
    
        framelabels.append(interp_angles)
    
    train_y = np.array(np.hstack(framelabels))
    
    # Only get stimulus frames:
    train_x_tmp = np.reshape(train_x, (len(train_trials), nframes_in_trial, nrois))
    train_X = train_x_tmp[:, stim_on:stim_on+nframes_on, :]
    train_X = np.reshape(train_X, (len(train_trials)*nframes_on, nrois))
    print train_X.shape
    train_X_std = StandardScaler().fit_transform(train_X)
    
    # Label full-rotation trial(s) to regress out angle from train trials and fit regressor:
    print "Using regression..."
    #clf = SVR(kernel='linear', C=1, tol=1e-9)
    clf = LinearSVR(dual=False, C=1, tol=1e-9, fit_intercept=False, loss='squared_epsilon_insensitive')
    clf.fit(train_X_std, train_y)






#%%
shuffle_frames = False

if fit_best:
    with open(os.path.join(output_dir, 'fit_RFE_results.txt'), 'wb') as f:
        f.write(str(svc))
        f.write('\n%s' % str({'kept_rids': kept_rids}))
    f.close()
    
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
        tvals.append(orig_test_traces)
        
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
    
#from sklearn.neural_network import MLPRegressor
#from sklearn.model_selection import cross_val_predict
#from sklearn.metrics import r2_score
#
#
#X = train_X_std.copy() #np.random.normal(size=(100, 2))
#y = np.array([v*(math.pi/180) for v in train_y]) #np.arctan2(np.dot(X, [1,2]), np.dot(X, [3,0.4]))
#
## simple prediction
#clf = MLPRegressor(activation='tanh', max_iter=10000)
#clf.fit(X, np.column_stack([np.sin(y), np.cos(y)]))
#
#

#
#
#
#y_simple_pred = cross_val_predict(model, X, y)
#
## transformed prediction
#joint = cross_val_predict(model, X, np.column_stack([np.sin(y), np.cos(y)]))
#y_trig_pred = np.arctan2(joint[:,0], joint[:,1])
#
## compare
#def align(y_true, y_pred):
#    """ Add or remove 2*pi to predicted angle to minimize difference from GT"""
#    y_pred = y_pred.copy()
#    y_pred[y_true-y_pred >  np.pi] += np.pi*2
#    y_pred[y_true-y_pred < -np.pi] -= np.pi*2
#    return y_pred
#
#print(r2_score(y, align(y, y_simple_pred))) # R^2 about 0.57
#print(r2_score(y, align(y, y_trig_pred)))   # R^2 about 0.99
#
#
#yr = np.reshape(y, (len(train_trials), nframes_on))
#y_simpler = np.reshape(align(y, y_simple_pred), (len(train_trials), nframes_on))
#
#ty = y_trig_pred[0:nframes_in_trial]
#ty[ty < 0] *= -1
#
#y_trigr = np.reshape(y_trig_pred, (len(train_trials), nframes_on))
#
#pl.subplot(1,2,1)
#pl.scatter(y_simpler[0,:], yr[0,:])
#pl.title('Direct model'); pl.xlabel('prediction'); pl.ylabel('actual')
#pl.subplot(1,2,2)
#pl.scatter(align(yr[0,:], y_trigr[0,:]), yr[0,:])
#pl.title('Sine-cosine model'); pl.xlabel('prediction'); pl.ylabel('actual')
#
#
#
#
#
#
## Look at CCW
#config_list = [c for c in test_configs.keys() if test_configs[c]['direction']==1 and test_configs[c]['ori']==0]
#config_list = sorted(config_list, key=lambda x: test_configs[x]['stim_dur'])
#color_list =  sns.color_palette("hsv", len(config_list))
#
#pl.figure()
#for lcolor, config in zip(color_list, config_list):
#    pl.plot(np.squeeze(mean_pred[config]), color=lcolor, label=test_configs[config]['stim_dur'])
#pl.plot([stim_on, stim_on+nframes_on], [-45, -45])
#pl.legend()

#%%

# #############################################################################
# Plot SIN of predicted angle, if using regression:
# #############################################################################

plot_std = True
predicted_color = 'cornflowerblue'
true_color = 'k'
use_sin = False

all_angles = sorted(list(set([test_configs[c]['ori'] for c in test_configs.keys()])))


if isinstance(clf, SVR) or isinstance(clf, LinearSVR):
    
    #%%
    configs_tested = sorted(list(set(labels_df['config'])), key=natural_keys)
    
    row1_configs = sorted([c for c in configs_tested if test_configs[c]['direction'] == 1], key=lambda x: test_configs[x]['stim_dur'])
    row2_configs = sorted([c for c in configs_tested if test_configs[c]['direction'] == -1], key=lambda x: test_configs[x]['stim_dur'])
    
    
    plot_configs = sorted(configs_tested, key=lambda x: (test_configs[x]['direction'], test_configs[x]['ori'], test_configs[x]['stim_dur']))
        
    pix = 0
    nconfigs = len(configs_tested)
    
    fig, axes = pl.subplots(nrows=2, ncols=nconfigs/2, sharex=True, sharey=True, squeeze=False, figsize=(5*len(row1_configs),6))
    
    
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
        
#        if start_rot == 360 and end_rot > start_rot:
#            start_rot -= 360
#            end_rot -= 360
#        elif start_rot == 0 and end_rot < start_rot:
#            start_rot += 180
#            end_rot += 180
#        elif start_rot == -90:
#            start_rot += 180
#            end_rot += 180
#        
        
        stim_on = list(set(labels_df[labels_df['config']==config]['stim_on_frame']))[0]
        nframes_on = list(set(labels_df[labels_df['config']==config]['nframes_on']))[0]
        nframes_in_trial = np.squeeze(mean_pred[config]).shape[0]
        
        interp_angles = np.ones(predicted_vals.shape) * np.nan
        interp_angles[stim_on:stim_on + nframes_on] = np.linspace(start_rot, end_rot, num=nframes_on)
        
        # Convert to rads:
        interp_angles = np.array([ v*(math.pi/180) for v in interp_angles])
        #predicted_vals[predicted_vals < 0] -= np.pi*2
        
        if use_sin:
            true_vals = np.array([np.sin( ang * (math.pi / 180.) ) if not np.isnan(ang) else np.nan for ang in interp_angles])
        else:
            if do_scatter:
                colorvals = sns.color_palette("hsv")
                stim_on = list(set(labels_df[labels_df['config']==config]['stim_on_frame']))[0]
                nframes_on = list(set(labels_df[labels_df['config']==config]['nframes_on']))[0]
                stim_frames = np.arange(stim_on, stim_on+nframes_on)
                cvals = [colorvals[1] if frameix in stim_frames else (0.0, 0.0, 0.0) for frameix in range(nframes_in_trial)]
                #pl.figure(); pl.scatter(true_vals, predicted_vals, c=cvals)
                true_vals = np.empty(interp_angles.shape)
                for ai, ang in enumerate(interp_angles):
                    if ai <= stim_on and np.isnan(ang):
                        true_vals[ai] = interp_angles[stim_on]
                    elif ai >= (stim_on + nframes_on - 1) and np.isnan(ang):
                        #print ai
                        true_vals[ai] = interp_angles[stim_on+nframes_on-1]
                    else:
                        true_vals[ai] = interp_angles[ai]
                    
                
            true_vals = np.array([ang if not np.isnan(ang) else np.nan for ai,ang in enumerate(interp_angles)])

        #print "True:", true_vals.shape
        #print "Pred:", predicted_vals.shape
        
                
#        if pix < 7:
#            ax.axes.xaxis.set_visible(False) #([])
#        ax.axes.xaxis.set_ticks([])
#        ax.axes.yaxis.set_ticks([])
#    
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
        
    #sns.despine(offset=4, trim=True, bottom=True, left=True)
    pl.legend(loc=9, bbox_to_anchor=(0, 0.1), ncol=2)
    
    #pl.savefig(os.path.join(output_dir, 'predicted_v_true.png'))


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
from matplotlib.colors import ListedColormap
from matplotlib.collections import LineCollection


# #############################################################################
# Plot probability of each trained angle, if using CLASSIFIER:
# #############################################################################

plot_trials = True
if plot_trials:
    mean_lw = 2
    trial_lw=0.4
else:
    mean_lw = 1.0
    trial_lw=0.2
    
#drifting = False
framerate = 44.69
linear_legend = False

if isinstance(clf, CalibratedClassifierCV):
    
    print [c for c in test_configs if test_configs[c]['ori']==0 and test_configs[c]['direction']==1]
    
    #config = 'config004' #'config001' -- start 180, CW
    #config = 'config008' # -- start 180, CCW
    
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
            axes[lix].plot(np.arange(stim_frames[-1]+1, nframes_in_trial), mean_pred[config][stim_frames[-1]+1:, class_index], 
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
                    
                    axes[lix].plot(np.arange(stim_frames[-1]+1, nframes_in_trial), all_preds[config][stim_frames[-1]+1:, class_index, trialn], 
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
        
        #%
        #train_savedir = os.path.split(train_fpath)[0]
        
        
        if shuffle_frames:
            figname = '%s_hsv_TEST_%s_%s_start_%i_%s_%s_%s_%s_shuffled.png' % (roiset, test_runid, config, starting_rot, rot_direction, rot_duration, plot_type, test_data_type)
        else:
            figname = '%s_hsv_TEST_%s_%s_start_%i_%s_%s_%s_%s.png' % (roiset, test_runid, config, starting_rot, rot_direction, rot_duration, plot_type, test_data_type)
        pl.savefig(os.path.join(output_dir, figname))
        pl.close()
    

#%%

# Look at traces of Top N neurons:
        
#if fit_best:
#    train_data_type = 'smoothedX'
#    # Also load Training Data to look at traces:
#    training_data_fpath = os.path.join(train_basedir, 'data_arrays', 'datasets.npz')
#    training_data = np.load(training_data_fpath)
#    print training_data.keys()
#    train_labels_df = pd.DataFrame(data=training_data['labels_data'], columns=training_data['labels_columns'])
#    trainingX = training_data[train_data_type][:, kept_rids]
#    print trainingX.shape
#    nrois = trainingX.shape[-1]
#    
#    # Get trial structure:
#    assert len(list(set(train_labels_df['nframes_on']))) == 1, "More than 1 nframes_on found in TRAIN set..."
#    train_nframes_on = list(set(train_labels_df['nframes_on']))[0]
#    assert len(list(set(train_labels_df['stim_on_frame']))) == 1, "More than 1 stim_on_frame val found in TRAIN set..."
#    train_stim_on = list(set(train_labels_df['stim_on_frame']))[0]
#    ntrials_by_cond = [v for k,v in training_data['run_info'][()]['ntrials_by_cond'].items()]
#    assert len(list(set(ntrials_by_cond)))==1, "More than 1 rep values found in TRAIN set"
#    ntrials_per_cond = list(set(ntrials_by_cond))[0]
#    
#    train_configs = training_data['sconfigs'][()]
#    
#    config_list = sorted([c for c in train_configs.keys()], key=lambda x: train_configs[x]['ori'])
#    traces = {}
#    for cf in config_list:
#        print cf, train_configs[cf]['ori']
#        
#        cixs = train_labels_df[train_labels_df['config']==cf].index.tolist()
#        curr_frames = trainingX[cixs, :]
#        print curr_frames.shape
#        nframes_per_trial = len(cixs) / ntrials_per_cond
#        
#        tmat = np.reshape(curr_frames, (ntrials_per_cond, nframes_per_trial, nrois))
#        traces[cf] = tmat
    
#%%

# #############################################################################
# Plot traces for top N neurons (similar to decoding traces):
# #############################################################################

#plot_trials = True
#drifting = True
#
#framerate = training_data['run_info'][()]['framerate']
#
#if isinstance(clf, CalibratedClassifierCV):
#    
#    #configs_tested = sorted(list(set(labels_df['config'])), key=natural_keys)
#    #maxval = max([traces[config].max() for config in traces.keys()])
#    stimulus_duration = round(nframes_on / framerate)
#        
#    # Plot each CLASS's probability on a subplot:
#    # -----------------------------------------------------------------------------
#    #colorvals = sns.color_palette("hsv", len(svc.classes_))
#    colorvals = sns.color_palette("hsv", len(svc.classes_))
#    stim_on = list(set(train_labels[train_labels['config']==config]['stim_on_frame']))[0]
#    nframes_on = list(set(train_labels[train_labels['config']==config]['nframes_on']))[0]
#    stim_frames = np.arange(stim_on, stim_on+nframes_on)
#
#    class_indices = [[v for v in svc.classes_].index(train_configs[c]['ori']) for c in config_list]
#    
#    fig, axes = pl.subplots(len(config_list), 1, figsize=(6,15))
#    #for lix in range(8):
#    #    curr_ori = svc.classes_[lix]
#    nframes_in_trial = nframes_per_trial
#    nclasses_total = len(config_list)
#    ntrials_curr_config =  ntrials_per_cond
#
#    for lix, (class_label, class_index) in enumerate(zip(config_list, class_indices)):
#        #print lix
#        cvals = [colorvals[class_index] if frameix in stim_frames else (0.0, 0.0, 0.0) for frameix in range(nframes_in_trial)]
#        
#        grand_meantrace_across_rois = np.mean(np.mean(traces[class_label], axis=-1), axis=0)
#        
#        axes[lix].plot(np.arange(0, stim_frames[0]), grand_meantrace_across_rois[0:stim_frames[0]],
#                            color='k', linewidth=1.5, alpha=1.0)
#        axes[lix].plot(stim_frames, grand_meantrace_across_rois[stim_frames], 
#                            color=colorvals[class_index], linewidth=1.5, alpha=1.0)
#        axes[lix].plot(np.arange(stim_frames[-1]+1, nframes_in_trial), grand_meantrace_across_rois[stim_frames[-1]+1:], 
#                            color='k', linewidth=1.5, alpha=1.0)
#
#        if plot_trials:
#            plot_type = 'trials'
#            meantrace_across_rois = np.mean(traces[class_label], axis=-1).T
#            for trialn in range(ntrials_curr_config):
##                axes[lix].plot(range(nframes_in_trial), all_preds[config][:, class_index, trialn], 
##                                    color=colorvals[class_index], linewidth=0.5)
#                axes[lix].plot(np.arange(0, stim_frames[0]), meantrace_across_rois[0:stim_frames[0], trialn],
#                                    color='k', linewidth=0.2, alpha=0.6)
#                
#                axes[lix].plot(stim_frames, meantrace_across_rois[stim_frames, trialn], 
#                                    color=colorvals[class_index], linewidth=0.2, alpha=0.6)
#                
#                axes[lix].plot(np.arange(stim_frames[-1]+1, nframes_in_trial), meantrace_across_rois[stim_frames[-1]+1:, trialn], 
#                                    color='k', linewidth=0.2, alpha=0.6)
#                
#        else:
#            plot_type = 'fillstd'
##            axes[lix].fill_between(range(nframes_in_trial), mean_pred[config][:,class_index]+sem_pred[config][:, class_index],
##                                        mean_pred[config][:,class_index]-sem_pred[config][:, class_index], alpha=0.2,
##                                        color=colorvals[class_index])
#            axes[lix].fill_between(range(nframes_in_trial), mean_pred[config][:,class_index]+sem_pred[config][:, class_index],
#                                        mean_pred[config][:,class_index]-sem_pred[config][:, class_index], alpha=0.2,
#                                        color='k')
#        #axes[lix].set_frame_on(False)
#        
#        #axes[lix].set_ylim([0, maxval])
#        if lix < 7:
#            axes[lix].axes.xaxis.set_visible(True) #([])
#        axes[lix].axes.xaxis.set_ticks([])
#        #axes[lix].set_title(class_label)
#            
#    #sns.despine(trim=True, offset=4, bottom=True)
#    
#    # show stim dur, using class labels as color key:
#
#    #pl.plot([stim_on, stim_on+nframes_on], np.ones((2,1))*0, 'r', linewidth=3)
#    stimframes = np.arange(stim_on, stim_on+nframes_on)
##        if test_configs[config]['stim_dur'] == quarter_dur:
##            nclasses_shown = int(len(class_indices) * (1/4.)) + 1
##        elif test_configs[config]['stim_dur'] == half_dur:
##            nclasses_shown = int(len(class_indices) * (1/2.)) + 1
##        else:
#    nclasses_shown = len(class_indices)
#        
#    ordered_indices = np.array(class_indices[0:nclasses_shown])
#        
#    ordered_colors = [colorvals[c] for c in ordered_indices]
#    cy = np.zeros(stimframes.shape) - 0.01
#    z = stimframes.copy()
#    points = np.array([stimframes, cy]).T.reshape(-1, 1, 2)
#    segments = np.concatenate([points[:-1], points[1:]], axis=1)
#    cmap = ListedColormap(ordered_colors)
#    lc = LineCollection(segments, cmap=cmap)
#    lc.set_array(z)
#    lc.set_linewidth(5)
#    pl.gca().add_collection(lc)
#    #axes[lix].axes.add_collection(lc)
#    pl.show()
#    
#    sns.despine(trim=True, offset=4, bottom=True)
#    #pl.plot(stimframes, np.ones(stimframes.shape), color=ordered_colors, linewidth=5)
#    
#    
#    # Custom legend:
#    from matplotlib.lines import Line2D
#    custom_lines = []
#    for lix, c_ori in enumerate(svc.classes_):
#        custom_lines.append(Line2D([0], [0], color=colorvals[lix], lw=4))
#    pl.legend(custom_lines, svc.classes_, loc=9, bbox_to_anchor=(0.5, -0.2), ncol=len(svc.classes_)/2)
#    
#    #%
##        if test_configs[config]['direction']==1:
##            rot_direction = 'CW'
##        else:
##            rot_direction = 'CCW'
#    rot_direction = 'static'
#        
##        if test_configs[config]['stim_dur'] == half_dur:
##            rot_duration = 'half'
##        elif test_configs[config]['stim_dur'] == quarter_dur:
##            rot_duration = 'quarter'
##        else:
##            rot_duration = 'full'
#    rot_duration = int(stimulus_duration)
#    starting_rot = test_configs[config]['ori']
#    config_english = 'start %i [%s, %s]' % (starting_rot, rot_direction, rot_duration)
#    pl.suptitle(config_english)
#    
#    #%
#    #train_savedir = os.path.split(train_fpath)[0]
#    
#    #%
##        if shuffle_frames:
##            figname = '%s_hsv_TEST_%s_%s_start_%i_%s_%s_%s_%s_shuffled.png' % (roiset, test_runid, config, starting_rot, rot_direction, rot_duration, plot_type, test_data_type)
##        else:
#    figname = '%s_hsv_TRAIN_%s_%s_start_%i_%s_%s_%s_%s.png' % (roiset, train_runid, config, starting_rot, rot_direction, rot_duration, plot_type, train_data_type)
#    
#    
#    pl.savefig(os.path.join(train_figdir, figname))
#    pl.close()
#    
        
#%%
        
train_figdir = os.path.join(train_basedir, 'classifiers', classif_identifier, 'traindata')
if not os.path.exists(train_figdir):
    os.makedirs(train_figdir)
            
# #############################################################################
# Load TRAINING DATA and plot traces:
# #############################################################################
# POLAR PLOTS?

train_traces = {}
if fit_best:
    train_data_type = 'corrected'
    # Also load Training Data to look at traces:
    training_data_fpath = os.path.join(train_basedir, 'data_arrays', 'datasets.npz')
    training_data = np.load(training_data_fpath)
    print training_data.keys()
    train_labels_df = pd.DataFrame(data=training_data['labels_data'], columns=training_data['labels_columns'])
    trainingX = training_data[train_data_type][:, kept_rids]
    print trainingX.shape
    nrois = trainingX.shape[-1]
    
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
    traces = {}
    for cf in config_list:
        print cf, train_configs[cf]['ori']
        
        cixs = train_labels_df[train_labels_df['config']==cf].index.tolist()
        curr_frames = trainingX[cixs, :]
        print curr_frames.shape
        nframes_per_trial = len(cixs) / ntrials_per_cond
        
        #tmat = np.reshape(curr_frames, (ntrials_per_cond, nframes_per_trial, nrois))
        tmat = np.reshape(curr_frames, (nframes_per_trial, ntrials_per_cond, nrois), order='f') 
        tmat = np.swapaxes(tmat, 1, 2) # Nframes x Nrois x Ntrials (to match test_traces)
        train_traces[cf] = tmat
    
    responses = []; baselines = [];
    for cf in config_list:
        tmat = train_traces[cf]
        print cf
        baselines_per_trial = np.mean(tmat[0:train_stim_on, :, :], axis=0) # ntrials x nrois -- baseline val for each trial
        meanstims_per_trial = np.mean(tmat[train_stim_on:train_stim_on+train_nframes_on, :, :], axis=0) # ntrials x nrois -- baseline val for each trial

        dffs_per_trial = ( meanstims_per_trial - baselines_per_trial) / baselines_per_trial
        for ridx in range(dffs_per_trial.shape[0]):
            dffs_per_trial[ridx, :] -= dffs_per_trial[ridx, :].min()
            
        mean_dff_config = np.mean(dffs_per_trial, axis=-1)
        
        mean_config = np.mean(meanstims_per_trial, axis=-1)
        baseline_config = np.mean(baselines_per_trial, axis=-1)
        
        responses.append(mean_config)
        baselines.append(baseline_config)
        
    responses = np.vstack(responses) # Reshape into NCONFIGS x NROIS array
    offsets = np.vstack(baselines)
    
#        tmat = np.reshape(curr_frames, (ntrials_per_cond, nframes_per_trial, nrois))
#        traces[cf] = tmat
#    
#    responses = []
#    for cf in config_list:
#        tmat = traces[cf]
#        print cf
#        baselines_per_trial = np.mean(tmat[:, 0:train_stim_on, :], axis=1) # ntrials x nrois -- baseline val for each trial
#        meanstims_per_trial = np.mean(tmat[:, train_stim_on:train_stim_on+train_nframes_on, :], axis=1) # ntrials x nrois -- baseline val for each trial
#
#        dffs_per_trial = ( meanstims_per_trial - baselines_per_trial) / baselines_per_trial
#        mean_dff_config = np.mean(dffs_per_trial, axis=0)
#        
#        responses.append(mean_dff_config)
#        
#    
#    responses = np.vstack(responses) # Reshape into NCONFIGS x NROIS array
#    
#%%

# Plot POLAR plots to show direction tuning:
# =============================================================================
    
thetas = [train_configs[cf]['ori'] * (math.pi/180) for cf in config_list]

print thetas

import matplotlib as mpl

nrows = 5 #4
ncols = 10 #5

#
#ridx = 4
#radii = responses[:, ridx]
#np.append(radii, responses[0, ridx])
#fig = pl.figure()
#polygon = mpl.patches.Polygon(zip(thetas, radii), fill=False)
#polar = fig.add_subplot(111, projection='polar')
#polar.add_line(polygon)
#polar.set_rmax(radii.max())
##polar.autoscale()
#polar.set_theta_zero_location("N")
#pl.plot([0, thetas[max_angle_ix]*(180/math.pi)], [0, radii[max_angle_ix]],'r', lw=5)
#pl.show()


#train_figdir = os.path.join(train_basedir, 'classifiers', classif_identifier, 'traindata')
#if not os.path.exists(train_figdir):
#    os.makedirs(train_figdir)
#    

fig, axes = pl.subplots(figsize=(10,10), nrows=nrows, ncols=ncols, subplot_kw=dict(polar=True))
appended_thetas = np.append(thetas, math.pi)

for ridx, ax in zip(range(nrois), axes.flat):
    print ridx
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

    

figname = '%s_polar_plots_%s.png' % (roiset, train_data_type)
pl.savefig(os.path.join(train_figdir, figname))
    
#%%

# Sort by direction preference:
# -----------------------------------------------------------------------------

thetas = [train_configs[cf]['ori'] for cf in config_list]

pdirections = []
for ridx in range(nrois):
    #print ridx
    radii = responses[:, ridx]
    if radii.min() < 0:
        radii -= radii.min()
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
                            color='k', linewidth=1.5, alpha=1.0)
        axes[lix].plot(stim_frames, mean_roi_trace[stim_frames], 
                            color=colorvals[class_index], linewidth=1.5, alpha=1.0)
        axes[lix].plot(np.arange(stim_frames[-1]+1, len(mean_roi_trace)), mean_roi_trace[stim_frames[-1]+1:], 
                            color='k', linewidth=1.5, alpha=1.0)

        # PLOT std:
        if plot_trials:
            plot_type = 'trials'
            for trialn in range(ntrials_curr_config):
                axes[lix].plot(np.arange(0, stim_frames[0]), roi_traces[0:stim_frames[0], trialn],
                                    color='k', linewidth=0.3, alpha=0.5)
                
                axes[lix].plot(stim_frames, roi_traces[stim_frames, trialn], 
                                    color=colorvals[class_index], linewidth=0.3, alpha=0.5)
                
                axes[lix].plot(np.arange(stim_frames[-1]+1, len(mean_roi_trace)), roi_traces[stim_frames[-1]+1:, trialn], 
                                    color='k', linewidth=0.3, alpha=0.5)
        else:
            axes[lix].fill_between(range(len(mean_roi_trace)), mean_roi_trace+std_roi_trace,
                                    mean_roi_trace-std_roi_trace, alpha=0.2,
                                    color='k')
    if lix < len(config_list):
        axes[lix].axes.xaxis.set_visible(True) #([])
        axes[lix].axes.xaxis.set_ticks([])
        

sns.despine(trim=True, offset=4, bottom=True)

# Custom legend:
from matplotlib.lines import Line2D
custom_lines = []
for lix, c_ori in enumerate(svc.classes_):
    custom_lines.append(Line2D([0], [0], color=colorvals[lix], lw=4))
pl.legend(custom_lines, svc.classes_, loc=9, bbox_to_anchor=(0.5, -0.2), ncol=len(svc.classes_)/2)
pl.suptitle('%s: Training set' % roiset)



figname = '%s_TRAIN_traces_%s_%s.png' % (roiset, train_runid, train_data_type)
pl.savefig(os.path.join(train_figdir, figname))


#%%

# #############################################################################
# Plot mean traces for ROIs sorted by ANGLE: TESTING DATA
# #############################################################################


# Create decoder-style plots for ROI responses to test data
# Each group of cells in preferred-direction group serves as "classifier"

plot_trials = True
if plot_trials:
    mean_lw = 2
    trial_lw=0.4
else:
    mean_lw = 1.0
    trial_lw=0.2
    
#drifting = False
framerate = 44.69
linear_legend = False

response_classif_dir = os.path.join(train_figdir, 'test_data_responses')
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
    minval = test_traces[test_config].min()
    maxval = test_traces[test_config].max()
    
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
                                        color='k', linewidth=0.4, alpha=0.5)
                    
                    axes[lix].plot(stim_frames, test_traces[test_config][stim_frames, ridx, trialn], 
                                        color=colorvals[class_index], linewidth=0.4, alpha=0.5)
                    
                    axes[lix].plot(np.arange(stim_frames[-1]+1, nframes_in_trial), test_traces[test_config][stim_frames[-1]+1:, ridx, trialn], 
                                        color='k', linewidth=0.4, alpha=0.5)
                    
            else:
                plot_type = 'fillstd'
                axes[lix].fill_between(range(nframes_in_trial), mean_roi_trace + std_roi_trace,
                                            mean_roi_trace - std_roi_trace, alpha=0.2,
                                            color='k')

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
        
        bar_offset = 0.1 * 1000
        curr_ylim = axes[lix].get_ylim()[0] - bar_offset
        
        axes[lix].set_ylim([minval-bar_offset*2, axes[lix].get_ylim()[1]])
        
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
        axes[lix].set_ylabel('intensity (%i)' % class_list[lix])

    
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
    pl.suptitle(config_english)
    
    #%
    #train_savedir = os.path.split(train_fpath)[0]
    
    
    if shuffle_frames:
        figname = '%s_traces_TEST_%s_%s_start_%i_%s_%s_%s_%s_shuffled.png' % (roiset, test_runid, test_config, starting_rot, rot_direction, rot_duration, plot_type, test_data_type)
    else:
        figname = '%s_traces_TEST_%s_%s_start_%i_%s_%s_%s_%s.png' % (roiset, test_runid, test_config, starting_rot, rot_direction, rot_duration, plot_type, test_data_type)
    
    pl.savefig(os.path.join(response_classif_dir, figname))
    pl.close()
    
