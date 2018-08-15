#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 13:49:50 2018

@author: julianarhee
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
from pipeline.python.utils import natural_keys, replace_root, label_figure

from scipy import stats
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.feature_selection import RFE
from scipy.optimize import curve_fit

from matplotlib.colors import ListedColormap
from matplotlib.collections import LineCollection


#%%

rootdir = '/mnt/odyssey'
animalid = 'CE077'
session = '20180713' #'20180713' #'20180629'
acquisition = 'FOV1_zoom1x'

acquisition_dir = os.path.join(rootdir, animalid, session, acquisition)


# #############################################################################
# Select TRAINING data and classifier:
# #############################################################################
train_runid = 'combined_gratings_static' #'blobs_run2'
train_traceid = 'cnmf_'
traceid_dir = util.get_traceid_from_acquisition(acquisition_dir, train_runid, train_traceid)

#%%

train_data_type = 'corrected'
classif_identifier = 'stat_allrois_LinearSVC_kfold_6ori_all_meanstim_%s' % train_data_type

clf_pts = classif_identifier.split('_')
decoder = clf_pts[4][1:]
print "Decoding: %s" % decoder

# LOAD TRAINING DATA:
# -------------------
train_fpath = os.path.join(traceid_dir, 'classifiers', classif_identifier, '%s_datasets.npz' % classif_identifier)
train_dset = np.load(train_fpath)

train_dtype = 'cX_std'

train_X = train_dset[train_dtype]
train_y = train_dset['cy']


train_labels = sorted(list(set(train_y)))
print "Training labels:", train_labels
# #############################################################################

data_identifier = '_'.join((animalid, session, acquisition, os.path.split(traceid_dir)[-1]))

#%%

# #############################################################################
# Select TESTING data:
# #############################################################################
test_runid = 'gratings_rotating' #'blobs_dynamic_run6' #'blobs_dynamic_run1' #'blobs_dynamic_run1'
test_traceid = 'cnmf_'

#%

# LOAD TEST DATA:
# -----------------------------------------------------------------------------

test_data_type = 'corrected' #'smoothedX' #'smoothedX' # 'corrected' #'smoothedX' #'smoothedDF'
test_basedir = util.get_traceid_from_acquisition(acquisition_dir, test_runid, test_traceid)
test_fpath = os.path.join(test_basedir, 'data_arrays', 'datasets.npz')
test_dataset = np.load(test_fpath)
assert test_data_type in test_dataset.keys(), "Specified d-type (%s) not found. Choose from: %s" % (test_data_type, str(test_dataset.keys()))
assert len(test_dataset[test_data_type].shape)>0, "D-type is empty!"

#% # Format TEST data:

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


#%% output paths:
sim_dir = os.path.join(traceid_dir, 'simulations')
if not os.path.exists(sim_dir):
    os.makedirs(sim_dir)
    

sim_data_dir  = os.path.join(sim_dir, 'simulated_calcium_traces')
if not os.path.exists(sim_data_dir):
    os.makedirs(sim_data_dir)

kernel_dir = os.path.join(sim_data_dir, 'test_kernel')

curr_sim_tuning_dir = os.path.join(sim_data_dir, 'psth_testdata')
if not os.path.exists(curr_sim_tuning_dir): os.makedirs(curr_sim_tuning_dir)
if not os.path.exists(os.path.join(sim_data_dir, 'test_data')): os.makedirs(os.path.join(sim_data_dir, 'test_data'))
if not os.path.exists(os.path.join(sim_data_dir, 'train_data')): os.makedirs(os.path.join(sim_data_dir, 'train_data'))


sim_classifier_dir = os.path.join(sim_dir, 'classifiers', classif_identifier)
if not os.path.exists(sim_classifier_dir): os.makedirs(sim_classifier_dir)

if not os.path.exists(os.path.join(sim_classifier_dir, 'testdata')): os.makedirs(os.path.join(sim_classifier_dir, 'testdata'))

if not os.path.exists(os.path.join(sim_classifier_dir, 'traindata')): os.makedirs(os.path.join(sim_classifier_dir, 'traindata'))

#%%

use_regression = False
fit_best = True
nfeatures_select = 50 #50 #'all' #75 # 'all' #75
best_C = 1 #1e9

# FIT CLASSIFIER: #############################################################
if train_X.shape[0] > train_X.shape[1]: # nsamples > nfeatures
    dual = False
else:
    dual = True

if 'LinearSVC' in classif_identifier:
    svc = LinearSVC(random_state=0, dual=dual, multi_class='ovr', C=best_C) #, C=best_C) # C=big_C)
    if fit_best:
        # First, get accuracy with all features:
        rfe = RFE(svc, n_features_to_select=nfeatures_select)
        rfe.fit(train_X, train_y)
        removed_rids = np.where(rfe.ranking_!=1)[0]
        kept_rids = np.array([i for i in np.arange(0, train_X.shape[-1]) if i not in removed_rids])
        train_X = train_X[:, kept_rids]
        print "Found %i best ROIs:" % nfeatures_select, train_X.shape
        roiset = 'best%i' % nfeatures_select
    else:
        print "Using ALL rois selected."
        roiset = 'all'

    svc.fit(train_X, train_y)
    clf = CalibratedClassifierCV(svc) 
    clf.fit(train_X, train_y)
    #output_dir = os.path.join(train_basedir, 'classifiers', classif_identifier, 'testdata')
    


#%%

            
# #############################################################################
# Load TRAINING DATA and plot traces:
# #############################################################################
# Also load Training Data to look at traces:
training_data_fpath = os.path.join(traceid_dir, 'data_arrays', 'datasets.npz')
training_data = np.load(training_data_fpath)
if 'arr_0' in training_data.keys():
    training_data = training_data['arr_0'][()]
    train_run_info = training_data['run_info']
print training_data.keys()
train_labels_df = pd.DataFrame(data=training_data['labels_data'], columns=training_data['labels_columns'])



#%%

fit_best = True
use_dff = True

#train_data_type = 'corrected'
trainingX = training_data[train_data_type][:, kept_rids]
print trainingX.shape
nrois = trainingX.shape[-1]

F0 = training_data['F0'][:, kept_rids]

train_configs = training_data['sconfigs'][()]


#%%
        
if fit_best:

    # Get trial structure:
    assert len(list(set(train_labels_df['nframes_on']))) == 1, "More than 1 nframes_on found in TRAIN set..."
    train_nframes_on = list(set(train_labels_df['nframes_on']))[0]
    assert len(list(set(train_labels_df['stim_on_frame']))) == 1, "More than 1 stim_on_frame val found in TRAIN set..."
    train_stim_on = list(set(train_labels_df['stim_on_frame']))[0]
    ntrials_by_cond = [v for k,v in train_run_info['ntrials_by_cond'].items()]
    assert len(list(set(ntrials_by_cond)))==1, "More than 1 rep values found in TRAIN set"
    ntrials_per_cond = list(set(ntrials_by_cond))[0]
        
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
from scipy import signal

# For each neuron, take the responses to each angle from its tuning curve:

ridx = 0
config = 'config002'
config = 'config002'

currtrials = list(set(labels_df[labels_df['config']==config]['trial']))
currconfig_angles = np.mean(np.vstack([mwtrials[trial]['rotation_values'] for trial in currtrials]), axis=0)

start_rot = currconfig_angles[0]
end_rot = currconfig_angles[-1]

stim_on = list(set(labels_df[labels_df['config']==config]['stim_on_frame']))[0]
nframes_on = list(set(labels_df[labels_df['config']==config]['nframes_on']))[0]
        
#nframes_in_trial = np.squeeze(test_traces[config]).shape[0]

# Simulation params:
framerate = 44.69
nsecs = round(nframes_on / framerate) #test_configs[config]['stim_dur']
tau_rise = 0.0156
tau_decay = 0.150 #0.150 #0.076 #0.48 #0.76 

tpoints = np.linspace(0, 1, 44.69*nsecs)

tpoints = np.arange(0, 1, 1./framerate)

#%%
def exp2(tau_rise=0.068, tau_decay=0.135, ksize=50):    
#def double_exponential(tpoints, A0=1, A1=1, tau_rise=0.068, tau_decay=0.135):
    # g(t) = exp(−t/tau_decay) − exp(−t/tau_rise)
    tpoints = np.arange(0, ksize)
    
    response = np.array([( np.exp(-t/tau_decay) - np.exp(-t/tau_rise) ) \
                / ( (1-np.exp(-ksize/tau_decay)) / (1-np.exp(-1/tau_decay)) \
                   - (1-np.exp(-ksize/tau_rise)) / (1-np.exp(-1/tau_rise)) ) \
                   for t in tpoints])
    return response

tmat = train_traces[config]
traces = tmat[:, ridx, :]
meantrace = np.mean(traces, axis=-1) 
meantrace /= sum(meantrace)

stim_function = np.zeros(nframes_per_trial) + 0.01
stim_function[stim_on:stim_on+nframes_on] = np.ones((nframes_on,)) 


A0 = responses[cf_idx, ridx] / offsets[cf_idx, ridx] #mean_data[stim_on:stim_on+nframes_on].max() #/ np.mean(mean_data[0:stim_on], axis=0)

gcamp = exp2(tau_rise=1.0, tau_decay=6., ksize=100) + offsets[cf_idx, ridx]
pl.plot(gcamp)

filtered = np.convolve(stim_function, gcamp, mode='full') / sum(gcamp)
pl.figure(); pl.plot(filtered)

impulse_response, _ = signal.deconvolve(filtered, gcamp)
pl.plot(impulse_response)


fig, axes = pl.subplots(3,1)
axes[0].plot(stim_function, 'k'); axes[0].set_title('stimulus')
axes[1].plot(meantrace, 'b'); axes[1].set_title('mean response (roi %i)' % int(kept_rids[ridx]+1))
axes[2].plot(impulse_response, 'r'); axes[2].set_title('impulse response')



fig, axes = pl.subplots(3, 1)
axes[0].plot(responses[:, ridx]); 
axes[0].set_xticks(list(np.arange(0, len(train_labels))))
axes[0].set_xticklabels(train_labels)
axes[0].set_ylim([0, responses.max()])



ntrials = train_traces['config001'].shape[-1]

fig, axes = pl.subplots(1, len(config_list), sharex=True, sharey=True)
i = 0
for config in config_list:
    for t in range(ntrials):
        axes[i].plot(train_traces[config][:, ridx, t], 'k', linewidth=0.3, alpha=0.5)
    axes[i].plot(np.mean(train_traces[config][:, ridx, :], axis=-1), 'r', linewidth=1)
    i += 1

#%%
 # Just make linear fit:
thetas = [train_configs[cf]['ori'] for cf in config_list]
upsampled_thetas = np.linspace(thetas[0], thetas[-1], num=1000)

if nfeatures_select == 50:
    nrows = 5
    ncols = 10
else:
    nrows = 4 #5 #5
    ncols = 5 #10 #10 #10

nrois = responses.shape[-1]
fig, axes = pl.subplots(figsize=(25,10), nrows=nrows, ncols=ncols)

max_val = responses.max()

interp_curves = []
for ridx, ax in zip(range(nrois), axes.flat):
    tuning = responses[:, ridx]
    
    y_fit = np.interp(upsampled_thetas, thetas, tuning)
    interp_curves.append(y_fit)

    ax.plot(thetas, tuning, 'ko')
    ax.plot(thetas, offsets[:, ridx], 'bo')
    ax.plot(upsampled_thetas, y_fit)
    
    ax.set_title(kept_rids[ridx]+1)
    ax.set_ylim([0, max_val]) # ax.get_ylim()[1]])
    
sns.despine(offset=4, trim=True)

label_figure(fig, data_identifier)

pl.savefig(os.path.join(sim_dir, '%s_linear_interp_meanstim_zerod_%s.png' % (roiset, train_data_type)))

interp_curves = np.vstack(interp_curves) # Nrois x N-sampled points


#%%

import matplotlib as mpl


thetas_r = [theta*(math.pi/180) for theta in thetas]
#thetas_r = thetas

if nfeatures_select == 50:
    nrows = 5
    ncols = 10
else:
    nrows = 4 #5 #5
    ncols = 5 #10 #10 #10
    
fig, axes = pl.subplots(figsize=(10,10), nrows=nrows, ncols=ncols, subplot_kw=dict(polar=True))
appended_thetas = np.append(thetas, math.pi)

max_val = responses.max()

for ridx, ax in zip(range(nrois), axes.flat):
    radii = responses[:, ridx]
#    if radii.min() < 0:
#        radii -= radii.min()
    polygon = mpl.patches.Polygon(zip(thetas_r, radii), fill=True, alpha=0.5, color='mediumorchid')
    ax.add_line(polygon)
    #ax.autoscale()
    ax.grid(True)
    ax.set_theta_zero_location("N")
    ax.set_xticklabels([])
    #ax.set_title(ridx)
    max_angle_ix = np.where(radii==radii.max())[0][0]
    ax.plot([0, thetas_r[max_angle_ix]], [0, radii[max_angle_ix]],'k', lw=1)
    ax.set_title(kept_rids[ridx]+1, fontsize=8)
    ax.set_xticks([t for t in thetas_r])
    #ax.set_ylim([0, max_val])
#    ax.set_thetamin(0)
#    ax.set_thetamax(math.pi) #([0, math.pi])
pl.rc('ytick', labelsize=8)
        
    #ax.set_xticklabels([int(round(t*(180/math.pi))) for t in thetas], fontsize=6)
    #ax#.yaxis.grid(False); legend.set_yticklabels([])
    #ax.spines["polar"].set_visible(False)

    
label_figure(fig, data_identifier)
figname = '%s_polar_plots_traindata_%s.png' % (roiset, train_data_type)
pl.savefig(os.path.join(sim_dir, figname))
    


#%%

# =============================================================================
# Get TEST DATA traces and predictions from trained classifier:
# =============================================================================
    
shuffle_frames = False
#
#if fit_best:
#    with open(os.path.join(output_dir, 'fit_RFE_results.txt'), 'wb') as f:
#        f.write(str(svc))
#        f.write('\n%s' % str({'kept_rids': kept_rids}))
#    f.close()
    
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

#%% 
##use_dff =  True

# Calculate DFF if not using mean_stim_dff:
sorted_test_configs = sorted([c for c in test_configs.keys()], key=lambda x: test_configs[c]['ori'])

if use_dff:
    for ix, config in enumerate(sorted_test_configs):
                
        currtrials = list(set(labels_df[labels_df['config']==config]['trial']))
        currconfig_angles = np.mean(np.vstack([mwtrials[trial]['rotation_values'] for trial in currtrials]), axis=0)
        
        start_rot = currconfig_angles[0]
        end_rot = currconfig_angles[-1]
        
        stim_on = list(set(labels_df[labels_df['config']==config]['stim_on_frame']))[0]
        nframes_on = list(set(labels_df[labels_df['config']==config]['nframes_on']))[0]
                
        nframes_in_trial = np.squeeze(test_traces[config]).shape[0]
        
        # Simulation params:
        framerate = 44.69
        nsecs = test_configs[config]['stim_dur']
        tpoints = np.linspace(0, 1, 44.69*nsecs)
    
        # Generate fake instantaneous response for each frame in trial using tuning curve:
        
        # Get angle shown for each frame:
        stimulus_values = np.ones((nframes_in_trial,)) * np.nan
        stimulus_values[stim_on:stim_on + nframes_on] = np.linspace(start_rot, end_rot, num=nframes_on)
    
        traces_real_data = test_traces[config][:, :, :]
        baselines = np.mean(test_traces[config][0:stim_on, :, :], axis=0)
        #test_traces[config] = (traces_real_data - baselines) / baselines
        test_traces[config] =  traces_real_data / baselines
    
    
#%%
    
#%%
def double_exponential(tpoints, A0=1, tau_rise=0.068, tau_decay=0.135):    
#def double_exponential(tpoints, A0=1, A1=1, tau_rise=0.068, tau_decay=0.135):
    # g(t) = exp(−t/tau_decay) − exp(−t/tau_rise)
    t0 = tpoints[0]
    #response = np.array([ A0*np.exp(-t/tau_decay) - A1*np.exp(-t/tau_rise) for t in tpoints])
    response = np.array([ A0 * ( 1 - np.exp(-(t-t0)/tau_rise) ) * np.exp( -(t-t0) / tau_decay) for t in tpoints])
    return response
#
##A0 = 64.1 # 2.0 #dff_real_data.max()
##A1 = 0.3 #curr_offset
##tau_rise = 0.068 #* framerate # sec
##tau_decay = 0.135 #* framerate # sec
#
#tau_rise = 0.0156
#tau_decay = 0.076 
#
#tpoints = np.linspace(0, 1, 44.69*6)
#
#response = double_exponential(tpoints, A0=A1, A1=A0, tau_rise=tau_rise, tau_decay=tau_decay)
#
#pl.figure(); pl.plot(response)
#    

from scipy import signal

#%%
# Current config / trial info:
config = 'config002'

currtrials = list(set(labels_df[labels_df['config']==config]['trial']))
currconfig_angles = np.mean(np.vstack([mwtrials[trial]['rotation_values'] for trial in currtrials]), axis=0)

start_rot = currconfig_angles[0]
end_rot = currconfig_angles[-1]

stim_on = list(set(labels_df[labels_df['config']==config]['stim_on_frame']))[0]
nframes_on = list(set(labels_df[labels_df['config']==config]['nframes_on']))[0]
        
nframes_in_trial = np.squeeze(test_traces[config]).shape[0]

# Simulation params:
framerate = 44.69
nsecs = test_configs[config]['stim_dur']
#tau_rise = 0.068 #* framerate # sec
#tau_decay = 0.135 #* framerate # sec
tau_rise = 0.0156
tau_decay = 0.150 #0.150 #0.076 #0.48 #0.76 

tpoints = np.linspace(0, 1, 44.69*nsecs)



#%%
# Generate fake instantaneous response for each frame in trial using tuning curve:

sim_params = {'tau_rise': tau_rise, 
              'tau_decay': tau_decay, 
              'tuning': 'linear',
              'nsecs': nsecs,
              'framerate': framerate}

with open(os.path.join(sim_dir, 'simulated_calcium_traces', 'simulation_params_%s.json' % config), 'w') as f:
    json.dump(sim_params, f, indent=4)
    
#%%
# Get real response for neuron:


# Get angle shown for each frame:
stimulus_values = np.ones((nframes_in_trial,)) * np.nan
stimulus_values[stim_on:stim_on + nframes_on] = np.linspace(start_rot, end_rot, num=nframes_on)



cf_idx = config_list.index(config)

for ridx in range(nrois):
    #bas_real_data = np.mean(test_traces[config][0:stim_on, ridx, :], axis=0)
    #dff_real_data = (test_traces[config][:, ridx, :] - bas_real_data) / bas_real_data
    #true_response = np.mean(dff_real_data, axis=1)

    traces_real_data = test_traces[config][:, ridx, :] #- np.mean(test_traces[config][0:stim_on, ridx, :], axis=0)        
    mean_data = np.mean(traces_real_data, axis=1)
    if mean_data.min() < 0:
        mean_data -= mean_data.min()
    #curr_offset = np.mean(offsets[:, ridx])

    interp_angle_ixs = [np.where(abs(upsampled_thetas-s) == min(abs(upsampled_thetas-s)))[0][0] if not np.isnan(s) else np.nan for s in stimulus_values]
    #inst_response = [interp_curves[ridx, i] if not np.isnan(i) else 0 for i in interp_angle_ixs]
    inst_response = [interp_curves[ridx, i] if not np.isnan(i) else max(offsets[:, ridx]) for i in interp_angle_ixs]
    
    
    A0 = responses[cf_idx, ridx] / np.abs(offsets[cf_idx, ridx]) #mean_data[stim_on:stim_on+nframes_on].max() #/ np.mean(mean_data[0:stim_on], axis=0)
    #true_response.max()
    #A1 = A0 #np.mean(mean_data[0:stim_on]) #A0 #0
    
    gcamp = double_exponential(tpoints, A0=A0, tau_rise=tau_rise, tau_decay=tau_decay)
    sim_trace = signal.convolve(inst_response, gcamp, mode='full') / sum(gcamp)
    #sim_trace = sim_trace[0:len(inst_response)]
    #sim_trace = np.pad(sim_trace, (stim_on-1,), mode='constant', constant_values=(curr_offset,) )
    
    
    fig, (ax_orig, ax_win, ax_filt) = pl.subplots(3, 1, sharex=True)
    ax_orig.plot(inst_response); ax_orig.set_title('Interpolated inst. response')
    ax_win.plot(gcamp); ax_win.set_title('GCaMP kernel')
    ax_filt.plot(sim_trace, label='simulated'); ax_filt.set_title('Simulated Ca2+ trace')
    ax_filt.plot(mean_data, label='data')
    ax_filt.plot([stim_on, stim_on+nframes_on], np.zeros(2,)-0.01, 'r')
    fig.tight_layout()
    
    label_figure(fig, data_identifier)
    pl.savefig(os.path.join(sim_data_dir, 'test_kernel', '%i_roi%i_simulated_testconfig%s.png' % (ridx, kept_rids[ridx]+1, config)))
    pl.close()


#
#dff_real_data = test_traces[config][:, ridx, :]
#true_response = np.mean(dff_real_data, axis=1)
#
#pl.figure();
#for t in range(dff_real_data.shape[1]):
#    pl.plot(dff_real_data[:, t], 'k', linewidth=0.5)
#pl.plot(true_response, 'k', linewidth=1)

#%%
    
# Plot "simulated" trace against real trace for each neuron:
    
sorted_test_configs = sorted([c for c in test_configs.keys()], \
                              key=lambda x: (test_configs[x]['stim_dur'], test_configs[x]['direction'], test_configs[x]['ori']))
for c in sorted_test_configs: print c, test_configs[c]['ori'], test_configs[c]['stim_dur'], test_configs[c]['direction']


for ridx in range(nrois):
    #bas_real_data = np.mean(test_traces[config][0:stim_on, ridx, :], axis=0)
    #dff_real_data = (test_traces[config][:, ridx, :] - bas_real_data) / bas_real_data
    #true_response = np.mean(dff_real_data, axis=1)
    #%
    fig, axes = pl.subplots(1, len(sorted_test_configs), sharex=True, sharey=True, figsize=(24,4))

    for ix, (config, ax) in enumerate(zip(sorted_test_configs, axes.flat)):
                
        currtrials = list(set(labels_df[labels_df['config']==config]['trial']))
        currconfig_angles = np.mean(np.vstack([mwtrials[trial]['rotation_values'] for trial in currtrials]), axis=0)
        
        start_rot = currconfig_angles[0]
        end_rot = currconfig_angles[-1]
        
        stim_on = list(set(labels_df[labels_df['config']==config]['stim_on_frame']))[0]
        nframes_on = list(set(labels_df[labels_df['config']==config]['nframes_on']))[0]
                
        nframes_in_trial = np.squeeze(test_traces[config]).shape[0]
        
        # Simulation params:
        framerate = 44.69
        nsecs = test_configs[config]['stim_dur']
        tpoints = np.linspace(0, 1, 44.69*nsecs)

        # Generate fake instantaneous response for each frame in trial using tuning curve:
        
        # Get angle shown for each frame:
        stimulus_values = np.ones((nframes_in_trial,)) * np.nan
        stimulus_values[stim_on:stim_on + nframes_on] = np.linspace(start_rot, end_rot, num=nframes_on)

        traces_real_data = test_traces[config][:, ridx, :] #- np.mean(test_traces[config][0:stim_on, ridx, :], axis=0)
        mean_data = np.mean(traces_real_data, axis=1)
        #curr_offset = np.mean(offsets[:, ridx])
    
        interp_angle_ixs = [np.where(abs(upsampled_thetas-s) == min(abs(upsampled_thetas-s)))[0][0] if not np.isnan(s) else np.nan for s in stimulus_values]
        inst_response = [interp_curves[ridx, i] if not np.isnan(i) else 0 for i in interp_angle_ixs]
        
        A0 = mean_data.max() #/ np.mean(np.mean(test_traces[config][0:stim_on, ridx, :], axis=0)) #true_response.max()
        #print "%s:"% config, A0
        A1 = A0 #0
        
        #gcamp = double_exponential(tpoints, A0=A0, A1=A1, tau_rise=tau_rise, tau_decay=tau_decay)
        gcamp = double_exponential(tpoints, A0=A0, tau_rise=tau_rise, tau_decay=tau_decay)

        sim_trace = signal.convolve(inst_response, gcamp, mode='full') / sum(gcamp)
        sim_trace = sim_trace[0:len(inst_response)]
        #sim_trace = np.pad(sim_trace, (stim_on-1,), mode='constant', constant_values=(curr_offset,) )
        
    
        # Plot simulated and real trace(s):
        ax.plot(sim_trace, 'cornflowerblue', linewidth=2); ax_filt.set_title('Simulated Ca2+ trace')
        ax.plot(mean_data, 'k', linewidth=2)
        for t in range(traces_real_data.shape[1]):
            ax.plot(traces_real_data[:, t], 'k', linewidth=0.5, alpha=0.3)
        
        # stimbar    
        ax.plot([stim_on, stim_on+nframes_on], np.zeros(2,)-0.01, 'r')

        # Use x-axis for time scale bar
        ax.set_xticks((stim_on, stim_on+nframes_on))
        ax.set_xticklabels((0, test_configs[config]['stim_dur']))
        ax.tick_params(axis='x', which='both',length=0)
        
        if ix > 0:
            for label in ax.get_yticklabels():
                label.set_visible(False)
            ax.yaxis.offsetText.set_visible(False)
            ax.yaxis.set_visible(False)
            for label in ax.get_xticklabels():
                label.set_visible(False)
            ax.xaxis.offsetText.set_visible(False)
            ax.xaxis.set_visible(False)
            
        if test_configs[config]['direction'] == 1:
            rot_direction = 'CW'
        else:
            rot_direction = 'CCW'
        
        ax.set_title('%i %s (%.1f s)' % (test_configs[config]['ori'], rot_direction, test_configs[config]['stim_dur']), fontsize=8)
        
    sns.despine(offset=4, trim=True)
    fig.tight_layout()
    pl.subplots_adjust(top=0.85)
    pl.suptitle('roi %i' % (kept_rids[ridx]+1))
    
    label_figure(fig, data_identifier)

#%
    
    pl.savefig(os.path.join(curr_sim_tuning_dir, '%i_roi%i_simulated.png' % (ridx, kept_rids[ridx]+1)))
    pl.close()
    


#%%

from random import gauss, seed
seed(1) # seed random number generator


framerate = 44.69
#ntrials = 10


#%%  
# =============================================================================
# Generate simulated TESTING DATASET:
# =============================================================================

max_noise = max([np.std(test_traces[c][0:stim_on, :])*2 for c,stim_on in \
                 zip(test_traces.keys(), [list(set(labels_df[labels_df['config']==c]['stim_on_frame']))[0] \
                     for c in test_traces.keys()])]) 

sim_traces = {}
#ridx = 18
ntrials = list(set([test_traces[c].shape[-1] for c in test_traces.keys()]))[0]
class_list = sorted([test_configs[c]['ori'] for c in sorted_test_configs])
print class_list

for config in test_configs.keys():
    # Get STIMULUS info:
    currtrials = list(set(labels_df[labels_df['config']==config]['trial']))
    currconfig_angles = np.mean(np.vstack([mwtrials[trial]['rotation_values'] for trial in currtrials]), axis=0)
    start_rot = currconfig_angles[0]
    end_rot = currconfig_angles[-1]
    
    if end_rot < 0:
        print config
        if len([i for i in class_list if i> 300]) > 0:
            start_rot += 360
            end_rot += 360
        else:
            start_rot += 180
            end_rot += 180
    
    # Get TRIAL info for current config:
    stim_on = list(set(labels_df[labels_df['config']==config]['stim_on_frame']))[0]
    nframes_on = list(set(labels_df[labels_df['config']==config]['nframes_on']))[0]
    nframes_in_trial = np.squeeze(test_traces[config]).shape[0]

    # Get angle shown for each frame:
    stimulus_values = np.ones((nframes_in_trial,)) * np.nan
    stimulus_values[stim_on:stim_on + nframes_on] = np.linspace(start_rot, end_rot, num=nframes_on)
        
    # Simulation params:
    nsecs = test_configs[config]['stim_dur']
    tpoints = np.linspace(0, 1, 44.69*nsecs)
    
    # Generate fake instantaneous response for each frame in trial using tuning curve:
    traces_real_data = test_traces[config][:, :, :] #/ np.mean(test_traces[config][0:stim_on, :, :], axis=0) # Subtract offset
    #traces_real_data /= np.mean(test_traces[config][0:stim_on, :, :], axis=0)
    mean_data = np.mean(traces_real_data, axis=-1) # Shape Nframes_in_trial x Nrois)
    #curr_offset = np.mean(offsets[:, ridx])
    
    # Get indices of upsampled angle values to match currently shown angles:
    #interp_angle_ixs = [np.where(abs(upsampled_thetas-s) == min(abs(upsampled_thetas-s)))[0][0] if not np.isnan(s) else np.nan for s in stimulus_values]
    interp_angle_ixs = [np.where(abs(upsampled_thetas-s) == min(abs(upsampled_thetas-s)))[0][0] for s in stimulus_values[stim_on:stim_on+nframes_on]]


    # Select a big noise level that isn't totally random:
    #max_noise = test_traces[config][0:stim_on, ridx, :].max()  # Using actual STD is too clean...
    #max_noise = max([np.std(test_traces[c][0:stim_on, :])*2 for c in test_traces.keys()])
    roi_noise = np.array([np.std(test_traces[config][stim_on:stim_on+nframes_on, ridx, :])*2 for ridx in range(nrois)])
    
    sim_traces_roi = []
    for t in range(ntrials):
        # Get "instantaneous response" from tuning curve:
        inst_response = interp_curves[:, interp_angle_ixs]
        inst_response = np.pad(inst_response, [(0, 0), (stim_on, nframes_in_trial-nframes_on-stim_on)], mode='constant', constant_values=0) # Pad pre- and post-stim
        
        # add generic noise:
        noise = np.random.normal(0, max_noise, (inst_response.shape))
        inst_response += noise #[i+n for i,n in zip(inst_response, noise)]
             
        # Add noise to generated trace
        #noise = [gauss(0, max_noise) for i in range(len(inst_response))]
        roi_variability = np.array([np.random.normal(0, roi_noise_level, (nframes_in_trial,)) for roi_noise_level in roi_noise])
        #noise = np.random.normal(0, max_noise, (inst_response.shape))

        simulated_response = inst_response + roi_variability #[s+n for s,n in zip(inst_response, noise)]   # Shape (Nrois, Nframes_in_trial)
        #pl.figure(); pl.plot(inst_response); pl.plot(simulated_response)

        # For each neuron, create GCaMP kernel using current config params and each neuron's A0:
        A0 = mean_data[stim_on:stim_on+nframes_on].max(axis=0) #/ np.mean(np.mean(test_traces[config][0:stim_on, :, :], axis=0), axis=-1) #true_response.max()
        gcamp_kernels = np.array([double_exponential(tpoints, A0=amplitude_val, tau_rise=tau_rise, tau_decay=tau_decay) for amplitude_val in A0])

        # Convolve noisy-instantaneous trace w/ kernel:
        sim_trace = np.array([signal.convolve(simulated_response[r, :], gcamp_kernels[r, :], mode='full') / sum(gcamp_kernels[r,:]) for r in range(simulated_response.shape[0])])
        sim_trace = sim_trace[:, 0:nframes_in_trial].T #.T # swap axes to fit with test_traces convention
        sim_traces_roi.append(sim_trace)
        
    sim_traces_roi = np.dstack(sim_traces_roi)
    
#    pl.figure()
#    for t in range(ntrials):
#        pl.plot(sim_traces_roi[:, 18, t], 'k', linewidth=0.3)
#    pl.plot(np.mean(sim_traces_roi[:, 18, :], axis=-1), 'k', linewidth=1.0)
#        
    sim_traces[config] = sim_traces_roi

#%%
    
# #############################################################################
# Plot all generated TEST data to comapre against real data:
# #############################################################################
    

sorted_test_configs = sorted([c for c in test_configs.keys()], \
                              key=lambda x: (test_configs[x]['stim_dur'], test_configs[x]['direction'], test_configs[x]['ori']))
for c in sorted_test_configs: print c, test_configs[c]['ori'], test_configs[c]['stim_dur'], test_configs[c]['direction']


for ridx in range(nrois):
    
    fig, axes = pl.subplots(2, len(sorted_test_configs), sharex=True, sharey=True, figsize=(24,4))
    
    for ix, config in enumerate(sorted_test_configs):
                
        currtrials = list(set(labels_df[labels_df['config']==config]['trial']))
        currconfig_angles = np.mean(np.vstack([mwtrials[trial]['rotation_values'] for trial in currtrials]), axis=0)
        
        start_rot = currconfig_angles[0]
        end_rot = currconfig_angles[-1]
        
        stim_on = list(set(labels_df[labels_df['config']==config]['stim_on_frame']))[0]
        nframes_on = list(set(labels_df[labels_df['config']==config]['nframes_on']))[0]
                
        nframes_in_trial = np.squeeze(test_traces[config]).shape[0]
        
        # Simulation params:
        framerate = 44.69
        nsecs = test_configs[config]['stim_dur']
        tpoints = np.linspace(0, 1, 44.69*nsecs)

        # Plot REAL traces:
        axes[0, ix].plot(np.mean(test_traces[config][:, ridx, :], axis=-1), 'k', linewidth=2)
        for t in range(test_traces[config].shape[-1]):
            axes[0, ix].plot(test_traces[config][:, ridx, t], 'k', linewidth=0.5, alpha=0.3)
        
        # Plot SIMULATED traces:
        axes[1, ix].plot(np.mean(sim_traces[config][:, ridx, :], axis=-1), 'cornflowerblue', linewidth=2)
        for t in range(sim_traces[config].shape[-1]):
            axes[1, ix].plot(sim_traces[config][:, ridx, t], 'cornflowerblue', linewidth=0.5, alpha=0.3)
        
        # stimbar    
        axes[0, ix].plot([stim_on, stim_on+nframes_on], np.zeros(2,)-0.01, 'r')
        axes[1, ix].plot([stim_on, stim_on+nframes_on], np.zeros(2,)-0.01, 'r')

        # Use x-axis for time scale bar
        axes[0,ix].set_xticks((stim_on, stim_on+nframes_on))
        axes[0,ix].set_xticklabels((0, test_configs[config]['stim_dur']))
        axes[0,ix].tick_params(axis='x', which='both',length=0)
        
        if ix > 0:
            for label in ax.get_yticklabels():
                label.set_visible(False)
            axes[0,ix].yaxis.offsetText.set_visible(False); axes[1,ix].yaxis.offsetText.set_visible(False);
            axes[0,ix].yaxis.set_visible(False); axes[1,ix].yaxis.set_visible(False);
            for label1, label2 in zip(axes[0, ix].get_xticklabels(), axes[1, ix].get_xticklabels()):
                label1.set_visible(False); label2.set_visible(False)
            axes[0,ix].xaxis.offsetText.set_visible(False); axes[1,ix].xaxis.offsetText.set_visible(False)
            axes[0,ix].xaxis.set_visible(False); axes[1,ix].xaxis.set_visible(False)
                
        if test_configs[config]['direction'] == 1:
            rot_direction = 'CW'
        else:
            rot_direction = 'CCW'
        
        axes[0,ix].set_title('%i %s (%.1f s)' % (test_configs[config]['ori'], rot_direction, test_configs[config]['stim_dur']), fontsize=8)
        
    sns.despine(offset=4, trim=True)
    fig.tight_layout()
    pl.subplots_adjust(top=0.85)
    pl.suptitle('roi %i' % (kept_rids[ridx]+1))

    pl.savefig(os.path.join(sim_data_dir, 'test_data', '%i_roi%i_simulated.png' % (ridx, kept_rids[ridx]+1)))
    pl.close()
    

#%%
    
# #############################################################################
# Generate TRAINING data:
# #############################################################################
 
ntrials = list(set([v for k,v in train_run_info['ntrials_by_cond'].items()]))[0]

stim_on = list(set([list(set(train_labels_df[train_labels_df['config']==config]['stim_on_frame']))[0] for config in train_traces.keys()]))
assert len(stim_on) == 1, "more than 1 stim ON frame found in TRAIN data set"
stim_on = stim_on[0]

max_noise = max([np.std(train_traces[c][0:stim_on, :])*2 for c in train_traces.keys()])



sim_traces_train = {}
#ridx = 18

for ci, config in enumerate(sorted(train_configs.keys(), key=lambda x: train_configs[x]['ori'])):
    
    # Get STIMULUS info:
    currtrials = list(set(train_labels_df[train_labels_df['config']==config]['trial']))

    # Get TRIAL info for current config:
    stim_on = list(set(train_labels_df[train_labels_df['config']==config]['stim_on_frame']))[0]
    nframes_on = list(set(train_labels_df[train_labels_df['config']==config]['nframes_on']))[0]
    nframes_in_trial = train_traces[config].shape[0]

    # Get angle shown for each frame:
    stimulus_values = np.ones((nframes_in_trial,)) * np.nan
    stimulus_values[stim_on:stim_on + nframes_on] = np.ones((nframes_on,)) * train_configs[config]['ori']
        
    # Simulation params:
    nsecs = round(nframes_on/framerate) # train_configs[config]['stim_dur']
    tpoints = np.linspace(0, 1, 44.69*nsecs)
    
    # Generate fake instantaneous response for each frame in trial using tuning curve:
    traces_real_data = train_traces[config][:, :, :] #- np.mean(train_traces[config][0:stim_on, :, :], axis=0) # Subtract offset
    mean_data = np.mean(traces_real_data, axis=-1) # Shape Nframes_in_trial x Nrois)
    #curr_offset = np.mean(offsets[:, ridx])
    
    # Get indices of upsampled angle values to match currently shown angles:
    #interp_angle_ixs = [np.where(abs(upsampled_thetas-s) == min(abs(upsampled_thetas-s)))[0][0] if not np.isnan(s) else np.nan for s in stimulus_values]
    #interp_angle_ixs = [np.where(abs(upsampled_thetas-s) == min(abs(upsampled_thetas-s)))[0][0] for s in stimulus_values[stim_on:stim_on+nframes_on]]


    # Select a big noise level that isn't totally random:
    #max_noise = test_traces[config][0:stim_on, ridx, :].max()  # Using actual STD is too clean...
    #max_noise = max([np.std(test_traces[c][0:stim_on, :])*2 for c in test_traces.keys()])
    roi_noise = np.array([np.std(train_traces[config][stim_on:stim_on+nframes_on, ridx, :])*2 for ridx in range(nrois)])
    
    sim_traces_roi = []
    for t in range(ntrials):
        # Get "instantaneous response" from tuning curve:
        inst_response = np.squeeze(np.dstack([[float(responses[ci, roi] + np.random.normal(0, roi_noise[roi], 1)) for roi in range(nrois)] for f in range(nframes_on)]))

        inst_response = np.pad(inst_response, [(0, 0), (stim_on, nframes_in_trial-nframes_on-stim_on)], mode='constant', constant_values=0) # Pad pre- and post-stim
        
        # add generic noise:
        noise = np.random.normal(0, max_noise, (inst_response.shape))
        inst_response += noise #[i+n for i,n in zip(inst_response, noise)]
             
        # Add noise to generated trace
        #noise = [gauss(0, max_noise) for i in range(len(inst_response))]
        roi_variability = np.array([np.random.normal(0, roi_noise_level, (nframes_in_trial,)) for roi_noise_level in roi_noise])
        #noise = np.random.normal(0, max_noise, (inst_response.shape))

        simulated_response = inst_response + roi_variability #[s+n for s,n in zip(inst_response, noise)]   # Shape (Nrois, Nframes_in_trial)
        #pl.figure(); pl.plot(inst_response); pl.plot(simulated_response)

        # For each neuron, create GCaMP kernel using current config params and each neuron's A0:
        A0 = mean_data[stim_on:stim_on+nframes_on].max(axis=0) #/ np.mean(np.mean(train_traces[config][0:stim_on, :, :], axis=0), axis=-1) #true_response.max()
        gcamp_kernels = np.array([double_exponential(tpoints, A0=amplitude_val, tau_rise=tau_rise, tau_decay=tau_decay) for amplitude_val in A0])

        # Convolve noisy-instantaneous trace w/ kernel:
        sim_trace = np.array([signal.convolve(simulated_response[r, :], gcamp_kernels[r, :], mode='full') / sum(gcamp_kernels[r,:]) for r in range(simulated_response.shape[0])])
        sim_trace = sim_trace[:, 0:nframes_in_trial].T #.T # swap axes to fit with test_traces convention
        sim_traces_roi.append(sim_trace)
        
    sim_traces_roi = np.dstack(sim_traces_roi)
    
#    pl.figure()
#    for t in range(ntrials):
#        pl.plot(sim_traces_roi[:, 18, t], 'k', linewidth=0.3)
#    pl.plot(np.mean(sim_traces_roi[:, 18, :], axis=-1), 'k', linewidth=1.0)
        
    sim_traces_train[config] = sim_traces_roi

#%%
# #############################################################################
# Plot all generated TRAIN data to comapre against real data:
# #############################################################################


sorted_train_configs = sorted([c for c in train_configs.keys()], \
                              key=lambda x: train_configs[x]['ori'])
for c in sorted_train_configs: print c, train_configs[c]['ori']


for ridx in range(nrois):
    
    fig, axes = pl.subplots(2, len(sorted_train_configs), sharex=True, sharey=True, figsize=(24/2,4))
    
    for ix, config in enumerate(sorted_train_configs):
                
        currtrials = list(set(train_labels_df[train_labels_df['config']==config]['trial']))

        stim_on = list(set(train_labels_df[train_labels_df['config']==config]['stim_on_frame']))[0]
        nframes_on = list(set(train_labels_df[train_labels_df['config']==config]['nframes_on']))[0]
                
        nframes_in_trial = train_traces[config].shape[0]
        
        # Simulation params:
        framerate = 44.69
        nsecs = round(nframes_on/framerate)
        tpoints = np.linspace(0, 1, 44.69*nsecs)

        # Plot REAL traces:
        axes[0, ix].plot(np.mean(train_traces[config][:, ridx, :], axis=-1), 'k', linewidth=2)
        for t in range(train_traces[config].shape[-1]):
            axes[0, ix].plot(train_traces[config][:, ridx, t], 'k', linewidth=0.5, alpha=0.3)
        
        # Plot SIMULATED traces:
        axes[1, ix].plot(np.mean(sim_traces_train[config][:, ridx, :], axis=-1), 'cornflowerblue', linewidth=2)
        for t in range(sim_traces_train[config].shape[-1]):
            axes[1, ix].plot(sim_traces_train[config][:, ridx, t], 'cornflowerblue', linewidth=0.5, alpha=0.3)
        
        # stimbar    
        axes[0, ix].plot([stim_on, stim_on+nframes_on], np.zeros(2,)-0.01, 'r')
        axes[1, ix].plot([stim_on, stim_on+nframes_on], np.zeros(2,)-0.01, 'r')

        # Use x-axis for time scale bar
        axes[0,ix].set_xticks((stim_on, stim_on+nframes_on))
        axes[0,ix].set_xticklabels((0, nsecs))
        axes[0,ix].tick_params(axis='x', which='both',length=0)
        
        if ix > 0:
            for label in ax.get_yticklabels():
                label.set_visible(False)
            axes[0,ix].yaxis.offsetText.set_visible(False); axes[1,ix].yaxis.offsetText.set_visible(False);
            axes[0,ix].yaxis.set_visible(False); axes[1,ix].yaxis.set_visible(False);
            for label1, label2 in zip(axes[0, ix].get_xticklabels(), axes[1, ix].get_xticklabels()):
                label1.set_visible(False); label2.set_visible(False)
            axes[0,ix].xaxis.offsetText.set_visible(False); axes[1,ix].xaxis.offsetText.set_visible(False)
            axes[0,ix].xaxis.set_visible(False); axes[1,ix].xaxis.set_visible(False)
                
        if train_configs[config]['direction'] == 1:
            rot_direction = 'CW'
        else:
            rot_direction = 'CCW'
        
        axes[0,ix].set_title('%i' % train_configs[config]['ori'], fontsize=8)
        
    sns.despine(offset=4, trim=True)
    fig.tight_layout()
    pl.subplots_adjust(top=0.85)
    pl.suptitle('roi %i' % (kept_rids[ridx]+1))

    label_figure(fig, data_identifier)
    pl.savefig(os.path.join(sim_data_dir, 'train_data', '%i_roi%i_simulated.png' % (ridx, kept_rids[ridx]+1)))
    pl.close()
    


#%% 

# =============================================================================
# SIMULATED DATA -- TRAIN classifier:
# =============================================================================
print train_dset[train_dtype].shape
if train_dtype == 'cX_std':
    use_mean_stim = True
else:
    use_mean_stim = False
    
# First, reformat trace arrays to be nsamples x nfeatures:
sim_train_X = []; sim_train_y = [];
for cf in sorted(train_configs.keys(), key = lambda x: train_configs[x]['ori']):
    curr_traces = sim_traces_train[cf]
    if use_mean_stim:
       curr_train_values = np.mean(curr_traces[stim_on:stim_on+nframes_on, :, :], axis=0)
    
    sim_train_X.append(curr_train_values)
    sim_train_y.append(np.array([train_configs[cf]['ori'] for s in range(curr_train_values.shape[-1])]))
    #train_labels_df 

sim_train_X = np.hstack(sim_train_X).T  # Shape:  Ntrials x Nrois
sim_train_y = np.hstack(sim_train_y)

#%%

use_regression = False

# FIT CLASSIFIER: #############################################################
if sim_train_X.shape[0] > sim_train_X.shape[1]: # nsamples > nfeatures
    dual = False
else:
    dual = True

# DON'T REFIT for top cells, since we want to use the same ROIs:
nrois = sim_train_X.shape[-1]
if 'LinearSVC' in classif_identifier:
    if use_regression is False:
        svc = LinearSVC(random_state=0, dual=dual, multi_class='ovr', C=1) #, C=best_C) # C=big_C)
        if fit_best:
            # First, get accuracy with all features:
            print "Using all %i ROIs selected from RFE step." % nrois
            
        svc.fit(sim_train_X, sim_train_y)
        clf = CalibratedClassifierCV(svc) 
        clf.fit(sim_train_X, sim_train_y)

        print "Saving output for simulations to:\n%s" % sim_classifier_dir
        
        
#%%

# SIMULATION:  Re-format TEST data:
        
sim_X_test = []; sim_labels_list = [];
for config in sorted_test_configs:
    # Get STIMULUS info:
    currtrials = list(set(labels_df[labels_df['config']==config]['trial']))
    currconfig_angles = np.mean(np.vstack([mwtrials[trial]['rotation_values'] for trial in currtrials]), axis=0)
    start_rot = currconfig_angles[0]
    end_rot = currconfig_angles[-1]
    
    if end_rot < 0:
        if max(class_list) < 300:
            start_rot += 180
            end_rot += 180
        else:
            start_rot += 360
            end_rot += 360
    
    # Get TRIAL info for current config:
    stim_on = list(set(labels_df[labels_df['config']==config]['stim_on_frame']))[0]
    nframes_on = list(set(labels_df[labels_df['config']==config]['nframes_on']))[0]
    nframes_in_trial = test_traces[config].shape[0]

    # Get angle shown for each frame:
    stimulus_values = np.ones((nframes_in_trial,)) * np.nan
    stimulus_values[stim_on:stim_on + nframes_on] = np.linspace(start_rot, end_rot, num=nframes_on)
        
    # Simulation params:
    nsecs = test_configs[config]['stim_dur']
    tpoints = np.linspace(0, 1, 44.69*nsecs)
    
    tarray_tmp = np.swapaxes(sim_traces[config], 1, 2) # Re-swap axes so that shape is Nframes x Ntrials Nrois
    curr_nframes, curr_ntrials, nrois = tarray_tmp.shape
    tarray = np.reshape(tarray_tmp, (curr_nframes*curr_ntrials, nrois), order='f')
    sim_X_test.append(tarray)
    
    sim_labels_list.append(labels_df[labels_df['config']==config].sort_values(['trial', 'tsec']))
    
sim_labels_df = pd.concat(sim_labels_list).reset_index(drop=True)
X_test_orig = np.vstack(sim_X_test)
X_test = StandardScaler().fit_transform(X_test_orig)


#
#X_test_orig = test_dataset[test_data_type]
#X_test = StandardScaler().fit_transform(X_test_orig)
#
#test_configs = test_dataset['sconfigs'][()]
#labels_df = pd.DataFrame(data=test_dataset['labels_data'], columns=test_dataset['labels_columns'])
#
## just look at 1 config for now:
sim_cgroups = sim_labels_df.groupby('config')


#%%
# =============================================================================
# SIMULATION:  Get TEST DATA traces and predictions from trained classifier:
# =============================================================================
    
shuffle_frames = False

mean_pred = {}
sem_pred = {}
all_preds = {}
test_traces = {}
predicted = []
for k,g in sim_cgroups:

    y_proba = []; tvals = [];
    for kk,gg in g.groupby('trial'):
        #print kk
        
        trial_ixs = gg.index.tolist()
        if shuffle_frames:
            shuffle(trial_ixs)

        curr_test = X_test[trial_ixs,:]
        orig_test_traces = X_test_orig[trial_ixs,:]
        
#        if fit_best:
#            curr_test = curr_test[:, kept_rids]
#            orig_test_traces = orig_test_traces[:, kept_rids]
#            
#            roiset = 'best%i' % nfeatures_select
#        else:
#            roiset = 'all'
            
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


#%%

print "Saving SIMULATED probability decoding curves to:\n%s" % os.path.join(sim_classifier_dir, 'testdata')


#%%
# #############################################################################
# SIMULATION - TEST SET:  Plot probability of each trained angle, if using CLASSIFIER:
# #############################################################################

plot_trials = True
if plot_trials:
    mean_lw = 2
    trial_lw=0.4
else:
    mean_lw = 1.0
    trial_lw=0.2
    
#drifting = False
linear_legend = False
    
if isinstance(clf, CalibratedClassifierCV):

    configs_tested = sorted(list(set(sim_labels_df['config'])), key=natural_keys)
    #maxval = 0.2 #max([all_preds[config].max() for config in mean_pred.keys()])
    
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
        maxval = all_preds[config].max()
        
        # Plot each CLASS's probability on a subplot:
        # -----------------------------------------------------------------------------

        # Get trial structure:
        stim_on = list(set(sim_labels_df[sim_labels_df['config']==config]['stim_on_frame']))[0]
        nframes_on = list(set(sim_labels_df[sim_labels_df['config']==config]['nframes_on']))[0]
        stim_frames = np.arange(stim_on, stim_on+nframes_on)

        nframes_in_trial = np.squeeze(all_preds[config]).shape[0]
        nclasses_total = np.squeeze(all_preds[config]).shape[1]
        ntrials_curr_config =  np.squeeze(all_preds[config]).shape[-1]
        
        # Get color-cycle order based on angle:
        colorvals = sns.color_palette("hls", len(svc.classes_))
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


        # PLOT:
        fig, axes = pl.subplots(len(class_list), 1, figsize=(6,15))

        for lix, (class_label, class_index) in enumerate(zip(class_list, class_indices)):
            
            # Mean trace:  Plot stimulus frames in class color:
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
                    axes[lix].plot(np.arange(0, stim_frames[0]), all_preds[config][0:stim_frames[0], class_index, trialn],
                                        color='k', linewidth=trial_lw, alpha=0.5)
                    axes[lix].plot(stim_frames, all_preds[config][stim_frames, class_index, trialn], 
                                        color=colorvals[class_index], linewidth=trial_lw, alpha=0.5)
                    axes[lix].plot(np.arange(stim_frames[-1]+1, nframes_in_trial), all_preds[config][stim_frames[-1]+1:, class_index, trialn], 
                                        color='k', linewidth=trial_lw, alpha=0.5)
                    
            else:
                plot_type = 'fillstd'
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
#            currmax = axes[lix].get_ylim()[1]
#            currmin = axes[lix].get_ylim()[0]
#            axes[lix].set_ylim([currmin-0.02, currmax])
            cy = np.ones(stimframes.shape) * (axes[lix].get_ylim()[0]/2)
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
            legend = fig.add_axes([0.82, 0.75, 0.2, 0.3],
                          projection='polar')
        
            thetas = sorted(np.array([ori_deg*(math.pi/180) for ori_deg in class_list]))
            if max(class_list) < 300: # not in class_list:
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


        #%
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
            figname = '%s_SIM_TEST_%s_%s_start_%i_%s_%s_%s_%s_shuffled.png' % (roiset, test_runid, config, starting_rot, rot_direction, rot_duration, plot_type, test_data_type)
        else:
            figname = '%s_sim_TEST_%s_%s_start_%i_%s_%s_%s_%s.png' % (roiset, test_runid, config, starting_rot, rot_direction, rot_duration, plot_type, test_data_type)
        pl.savefig(os.path.join(sim_classifier_dir, 'testdata', figname))
        pl.close()
    

#%%
        
# #############################################################################
# SIMULATED DATA - TRAINING SET :  PLOT Average and Trial curves for ROIs sorted by orientation preference:
# #############################################################################

        
thetas = [train_configs[cf]['ori'] for cf in config_list]

pdirections = []
for ridx in range(nrois):
    #print ridx
    radii = responses[:, ridx]
    if radii.min() < 0:
        radii -= radii.min()
    max_angle_ix = np.where(radii==radii.max())[0][0]

    pdirections.append((ridx, thetas[max_angle_ix], radii[max_angle_ix]))
        

print "Saving SIMULATED training data to:\n%s" % os.path.join(sim_classifier_dir, 'traindata')

        
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
        roi_traces = sim_traces_train[class_label][:, ridx, :]
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
        
    axes[lix].set_title(str([kept_rids[r]+1 for r in rois_preferred]), fontsize=8)

sns.despine(trim=True, offset=4, bottom=True)

# Custom legend:
from matplotlib.lines import Line2D
custom_lines = []
for lix, c_ori in enumerate(svc.classes_):
    custom_lines.append(Line2D([0], [0], color=colorvals[lix], lw=4))
pl.legend(custom_lines, svc.classes_, loc=9, bbox_to_anchor=(0.5, -0.2), ncol=len(svc.classes_)/2)
pl.suptitle('%s: Training set' % roiset)

label_figure(fig, data_identifier)

figname = '%s_SIM_TRAIN_traces_%s_%s.png' % (roiset, train_runid, train_data_type)

pl.savefig(os.path.join(sim_classifier_dir, 'traindata', figname))

#%
#%%
# Get training data in format frames x nrois:


sim_train_labels = []
# First, reformat trace arrays to be nsamples x nfeatures:
sim_train_X_test = []; sim_train_y_test = [];
for cf in sorted(train_configs.keys(), key = lambda x: train_configs[x]['ori']):
    print cf
    curr_traces_tmp = sim_traces_train[cf]
    nfr, nr, nt = curr_traces_tmp.shape
    curr_traces = np.reshape(np.swapaxes(curr_traces_tmp, 1, 2), (nfr*nt, nr), order='f')  # Nframes x Nrois x Ntrials (to match test_traces)
    
    sim_train_X_test.append(curr_traces)
    sim_train_y_test.append(np.array([train_configs[cf]['ori'] for s in range(curr_traces.shape[0])]))
    #train_labels_df 
    sim_train_labels.append(train_labels_df[train_labels_df['config']==cf])

sim_train_labels_df = pd.concat(sim_train_labels).reset_index(drop=True)

sim_train_X_test = np.vstack(sim_train_X_test)  # Shape:  Nframes total x Nrois
sim_train_y_test = np.hstack(sim_train_y_test)



#%%

train_cgroups_sim = sim_train_labels_df.groupby('config')

sim_train_X_test_std = StandardScaler().fit_transform(sim_train_X_test)

# How good can the classifier "decode" its own training data:
mean_preds_train = {}
sem_preds_train = {}
all_preds_train = {}

predicted = []
for k,g in train_cgroups_sim:

    y_proba = []; y_true = [];
    for kk,gg in g.groupby('trial'):
        #print kk
        
        trial_ixs = gg.index.tolist()
        if shuffle_frames:
            shuffle(trial_ixs)

        curr_test = sim_train_X_test_std[trial_ixs,:]
        orig_test_traces = sim_train_X_test[trial_ixs,:]
        
            
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
        
        
    mean_preds_train[k] = means_by_class
    sem_preds_train[k] = stds_by_class
    all_preds_train[k] = y_proba
    
    
#%%
#% #############################################################################
# Plot mean traces for ROIs sorted by ANGLE: TRAINING DATA
# #############################################################################

plot_trials = True
trial_lw = 0.2
mean_lw = 1

trained_classes = list(np.copy(train_labels))
trained_classes.append('bas')


if any([isinstance(v, str) for v in clf.classes_]):
    class_indices = [[v for v in clf.classes_].index(str(c)) for c in train_labels]
else:
    class_indices = [[v for v in clf.classes_].index(c) for c in train_labels]
    
if 'bas' in clf.classes_:
    colorvals = sns.color_palette("hls", len(clf.classes_)-1)
else:
    colorvals = sns.color_palette("hls", len(clf.classes_))
    
    
fig, axes = pl.subplots(len(train_labels), 1, figsize=(6,15))


#class_indices = [[i for i in svc.classes_].index(str(v)) for v in train_labels]

for lix, (class_label, class_index) in enumerate(zip(train_labels, class_indices)):
    curr_config = [k for k,v in train_configs.items() if v['ori'] == class_label][0]
    
    stim_on = list(set(train_labels_df[train_labels_df['config']==curr_config]['stim_on_frame']))[0]
    nframes_on = list(set(train_labels_df[train_labels_df['config']==curr_config]['nframes_on']))[0]
    stim_frames = np.arange(stim_on, stim_on+nframes_on)

    nframes_in_trial = np.squeeze(all_preds_train[curr_config]).shape[0]
    nclasses_total = np.squeeze(all_preds_train[curr_config]).shape[1]
    ntrials_curr_config =  np.squeeze(all_preds_train[curr_config]).shape[-1]

   
    for vi, class_name in enumerate(clf.classes_):
        ix_in_svc = [i for i in svc.classes_].index(class_name)

        if class_name == 'bas':
            axes[lix].plot(np.arange(0, nframes_in_trial), mean_preds_train[curr_config][:, int(ix_in_svc)], color='gray', linestyle=':')
        else:
            color_index = train_labels.index(int(class_name))
            axes[lix].plot(np.arange(0, nframes_in_trial), mean_preds_train[curr_config][:, int(ix_in_svc)], color=colorvals[color_index], alpha=0.8)
        

    axes[lix].axes.xaxis.set_ticks([])
                
    bar_offset = 0.1
    curr_ylim = axes[lix].get_ylim()[0] - bar_offset
    
    #axes[lix].set_ylim([minval-bar_offset*2, maxval])
    
    # Plot chance line:
    chance = 1/len(trained_classes)
    axes[lix].plot(np.arange(0, nframes_in_trial), np.ones((nframes_in_trial,))*chance, 'k--', linewidth=0.5)
    
                
sns.despine(trim=True, offset=4, ax=axes[lix])


for lix in range(len(train_labels)):
    # Create color bar:
    cy = np.ones(stim_frames.shape) * axes[lix].get_ylim()[0]/2.0
    z = stim_frames.copy()
    points = np.array([stim_frames, cy]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    cmap = ListedColormap(colorvals[lix])
    lc = LineCollection(segments, cmap=cmap)
    lc.set_array(z)
    lc.set_linewidth(5)
    axes[lix].add_collection(lc)
    
    if lix == len(trained_classes)-1:
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
        
    axes[lix].set_ylabel('prob') # % trained_classes[lix])


#pl.plot(stimframes, np.ones(stimframes.shape), color=ordered_colors, linewidth=5)
pl.subplots_adjust(top=0.85)
    

# Custom legend:
from matplotlib.lines import Line2D
custom_lines = []
for lix, c_ori in enumerate(svc.classes_):
    if c_ori == 'bas':
        custom_lines.append(Line2D([0], [0], color='gray', lw=4))
    else:
        custom_lines.append(Line2D([0], [0], color=colorvals[lix], lw=4))
pl.legend(custom_lines, trained_classes, loc=9, bbox_to_anchor=(0.5, -0.2), ncol=len(svc.classes_)/2)
pl.suptitle('%s: Decoding training set' % roiset)

label_figure(fig, data_identifier)


figname = '%s_decoded_training_traces_%s_%s.png' % (roiset, train_runid, train_data_type)
pl.savefig(os.path.join(sim_classifier_dir, 'traindata', figname))


