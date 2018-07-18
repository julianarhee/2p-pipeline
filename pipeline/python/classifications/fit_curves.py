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
from pipeline.python.utils import natural_keys, replace_root

from scipy import stats
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.feature_selection import RFE


#%%

rootdir = '/Volumes/coxfs01/2p-data' #'/mnt/odyssey'
animalid = 'CE077'
session = '20180713' #'20180629'
acquisition = 'FOV1_zoom1x'

acquisition_dir = os.path.join(rootdir, animalid, session, acquisition)


# #############################################################################
# Select TRAINING data and classifier:
# #############################################################################
train_runid = 'gratings_static' #'blobs_run2'
train_traceid = 'traces001'
traceid_dir = util.get_traceid_from_acquisition(acquisition_dir, train_runid, train_traceid)

#%%

classif_identifier = 'stat_allrois_LinearSVC_kfold_6ori_all_meanstim'

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


#%%

use_regression = False
fit_best = True
nfeatures_select = 50 #'all' #75 # 'all' #75

# FIT CLASSIFIER: #############################################################
if train_X.shape[0] > train_X.shape[1]: # nsamples > nfeatures
    dual = False
else:
    dual = True

if 'LinearSVC' in classif_identifier:
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
    #output_dir = os.path.join(train_basedir, 'classifiers', classif_identifier, 'testdata')
    


#%%
sim_dir = os.path.join(acquisition_dir, train_runid, train_traceid, 'simulations')
if not os.path.exists(sim_dir):
    os.makedirs(sim_dir)
            
# #############################################################################
# Load TRAINING DATA and plot traces:
# #############################################################################

fit_best = True
        
if fit_best:
    train_data_type = 'corrected'
    # Also load Training Data to look at traces:
    training_data_fpath = os.path.join(traceid_dir, 'data_arrays', 'datasets.npz')
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
        
        tmat = np.reshape(curr_frames, (ntrials_per_cond, nframes_per_trial, nrois))
        traces[cf] = tmat
    
    responses = []; baselines = [];
    for cf in config_list:
        tmat = traces[cf]
        print cf
        baselines_per_trial = np.mean(tmat[:, 0:train_stim_on, :], axis=1) # ntrials x nrois -- baseline val for each trial
        meanstims_per_trial = np.mean(tmat[:, train_stim_on:train_stim_on+train_nframes_on, :], axis=1) # ntrials x nrois -- baseline val for each trial

        dffs_per_trial = ( meanstims_per_trial - baselines_per_trial) / baselines_per_trial
        mean_dff_config = np.mean(dffs_per_trial, axis=0)
        mean_config = np.mean(meanstims_per_trial, axis=0)
        baseline_config = np.mean(baselines_per_trial, axis=0)
        
        responses.append(mean_config)
        baselines.append(baseline_config)
        
    responses = np.vstack(responses) # Reshape into NCONFIGS x NROIS array
    offsets = np.vstack(baselines)
    
#%%

def wrapped_gaussian(theta, *params):
    (Rpreferred, Rnulll, theta_preferred, sigma, offset) = params
#    a = theta - theta_preferred 
#    b = theta + 180 - theta_preferred
#    wrap_a = min( [a, a-360, a+360] )
#    wrap_b = min( [b, b-360, b+360] )

    wrap_as = theta[:, 0]
    wrap_bs = theta[:, 1]
    pref_term = Rpreferred * np.exp( - (wrap_as**2) / (2.0 * sigma**2.0) )
    null_term = Rnull * np.exp( - (wrap_bs**2) / (2.0 * sigma**2.0) )
    R = offset + pref_term +  null_term
    
    return R


def wrapped_curve(theta, *params):
    (Rpreferred, theta_preferred, sigma) = params
    v = 0
    for n in [-2, 2]:
        v += np.exp( -((theta - theta_preferred + 180*n)**2) / (2 * sigma**2) )
    return Rpreferred * v


def von_mises_double(theta, *params):
    (Rpreferred, Rnull, theta_preferred, k1, k2) = params
    pterm = Rpreferred * np.exp(k1 * ( np.cos( thetas*(math.pi/180.)- theta_preferred*(math.pi/180.) ) - 1))
    nterm = Rnull * np.exp(k2 * ( np.cos( theta*(math.pi/180.)- theta_preferred*(math.pi/180.) ) - 1))
    return pterm + nterm


def von_mises(theta, *params):
    (Rpreferred, theta_preferred, sigma, offset) = params
    R = Rpreferred * np.exp( sigma * (np.cos( 2*(theta*(math.pi/180.) - theta_preferred*(math.pi/180.)) ) - 1) )
    return R + offset


#def double_gaussian( x, params ):
#    (c1, mu1, sigma1, c2, mu2, sigma2) = params
#    res =   c1 * np.exp( - (x - mu1)**2.0 / (2.0 * sigma1**2.0) ) \
#          + c2 * np.exp( - (x - mu2)**2.0 / (2.0 * sigma2**2.0) )
#    return res

def double_gaussian_fit( params ):
    fit = wrapped_gaussian( xvals, params )
    return (fit - y_proc)


from scipy.optimize import curve_fit
from scipy.optimize import leastsq


#%%

# TRY von mises for NON-drifiting (0, 180)

drifting = False
thetas = [train_configs[cf]['ori'] for cf in config_list]


nrows = 5
ncols = 10

nrois = responses.shape[-1]
fig, axes = pl.subplots(figsize=(12,12), nrows=nrows, ncols=ncols)

nrois = responses.shape[-1]
#ridx = 2

for ridx, ax in zip(range(nrois), axes.flat):
    tuning = responses[:, ridx]
    offset = np.mean(offsets[:, ridx]) #np.mean(offsets[:, ridx])
    max_ix = np.where(tuning==tuning.max())[0][0]
    Rpreferred = tuning[max_ix]
    theta_preferred = thetas[max_ix]
    
    if drifting:
        if theta_preferred >=180:
            theta_null = theta_preferred - 180
        else:
            theta_null = theta_preferred + 180
        null_ix = thetas.index(theta_null)
        Rnull = tuning[null_ix]
    
        print "Rpref: %.3f (%i) | Rnull: %.3f (%i)" % (Rpreferred, theta_preferred, Rnull, theta_null)
    
    else:
        print "Rpref: %.3f (%i) " % (Rpreferred, theta_preferred)
    
    #pl.figure()
    sigma_diffs = []
    test_sigmas = np.arange(5, 60, 5)
    for sigma in np.arange(5, 30, 5):
        
        try:
            popt=None; pcov=None;
            params = [Rpreferred, theta_preferred, sigma, offset]
            popt, pcov = curve_fit(von_mises, thetas, tuning, p0=params )
            sigma_diffs.append(np.abs(popt[2] - sigma))
            
            y_fit = von_mises(upsampled_thetas, *popt)
        except Exception as e:
            print "%i no fit" % ridx
        #pl.plot(thetas, tuning, 'ko')
        #pl.plot(upsampled_thetas, y_fit, label=sigma)
        
    #pl.legend()
    if popt is not None:
        best_ix = sigma_diffs.index(min(sigma_diffs))
        best_sigma = test_sigmas[best_ix]  
        print best_sigma
    
        params = [Rpreferred, theta_preferred, best_sigma, offset]
        popt, pcov = curve_fit(von_mises, thetas, tuning, p0=params )
        sigma_diffs.append(np.abs(popt[2] - sigma))
        
        y_fit = von_mises(upsampled_thetas, *popt)
        ax.plot(thetas, tuning, 'ko')
        ax.plot(upsampled_thetas, y_fit, label=sigma)
        

#def get_wrap(thetas):
#    xvals = np.empty((len(thetas), 2))
#    #print xvals.shape
#    for ti, theta in enumerate(thetas):
#        a = theta - theta_preferred 
#        b = theta + 180 - theta_preferred
#        wrap_a = min( [a, a-360, a+360] )
#        wrap_b = min( [b, b-360, b+360] )
#        xvals[ti, :] = np.array([wrap_a, wrap_b])
#    #print xvals[:, 0]
#    #print xvals[:, 1]
#    return xvals
#   
#ofset = 0
#for sigma in np.arange(45*(math.pi/180)/4, 45*(math.pi/180)*2, 10*(math.pi/180)):
#    
#    params = [Rpreferred, Rnull, theta_preferred, sigma, offset]
#    popt, pcov = curve_fit(wrapped_gaussian, xvals, tuning, p0=params )
#    
#    y_fit = wrapped_gaussian(get_wrap(upsampled_thetas), *popt)
#    pl.figure()
#    pl.plot(thetas, tuning, 'ko')
#    pl.plot(upsampled_thetas, y_fit)


#%%

# Try bimodal with drifting:
    
upsampled_thetas = np.linspace(thetas[0], thetas[-1], num=100)

nrows = 5
ncols = 10

nrois = responses.shape[-1]
fig, axes = pl.subplots(figsize=(12,12), nrows=nrows, ncols=ncols)

for ridx, ax in zip(range(nrois), axes.flat):
    
    tuning = responses[:, ridx]
    max_ix = np.where(tuning==tuning.max())[0][0]
    Rpreferred = tuning[max_ix]
    theta_preferred = thetas[max_ix]
#    if theta_preferred >=180:
#        theta_null = theta_preferred - 180
#    else:
#        theta_null = theta_preferred + 180
#    null_ix = thetas.index(theta_null)
#    Rnull = tuning[null_ix]
    print "%i - Rpref: %.3f (%i)" % (ridx, Rpreferred, theta_preferred)
    
    
    # Find best sigma:
    k1_fits=[]; k2_fits=[];
    #test_sigmas = np.arange(2.5*(math.pi/180), 45*(math.pi/180)*2, 2.5*(math.pi/180))
    test_sigmas = np.arange(5, 45*2, 5)
    for k in test_sigmas:
        try:
            popt=None; pcov=None;
            k1 = k; k2 = k;
            params = [Rpreferred, theta_preferred, k1, k2]
            popt, pcov = curve_fit(bimodal_curve, np.array(thetas), tuning, p0=params )
            k1_fits.append(k1-popt[3])
            k2_fits.append(k2-popt[4])        
        except Exception as e:
            print "%i - NO FIT" % ridx
            continue
    pl.legend()  
    
    if popt is not None:
        small_k = min([min(k1_fits), min(k2_fits)])
        if small_k in k1_fits:
            best_k = k1_fits.index(small_k)
        else:
            best_k = k2_fits.index(small_k)
        #print test_sigmas[best_k] * (180./math.pi)
        
        del popt
        del pcov
        best_sigma = test_sigmas[best_k] 
        k1 = best_sigma; k2 = best_sigma;
        params = [Rpreferred, Rnull, theta_preferred, k1, k2]
        popt, pcov = curve_fit(bimodal_curve, np.array(thetas), tuning, p0=params , maxfev=5000)
        y_fit = bimodal_curve(np.array(upsampled_thetas), *popt)
        ax.plot(thetas, tuning, 'ko')
        ax.plot(upsampled_thetas, y_fit) #, label=k*(180./math.pi))
        ax.set_title("sigma = %.2f" % best_sigma, fontsize=8)
        ax.set_xticks(thetas)

sns.despine(trim=True, offset=4)
    



#%%

# Thin plate spline interp?

ridx = 15
tuning = responses[:, ridx]
from scipy.interpolate import splprep, splev


tck, u = splprep([thetas, tuning], k=5, t=-1, s=1000)
new_points = splev(u, tck)

fig, ax = pl.subplots()
ax.plot(thetas, tuning, 'ro')
ax.plot(new_points[0], new_points[1], 'r-')
pl.show()

# Just fit a polynomial...

def _polynomial(x, *p):
    """Polynomial fitting function of arbitrary degree."""
    poly = 0.
    for i, n in enumerate(p):
        poly += n * x**i
    return poly

p0 = np.ones(6,)

coeff, var_matrix = curve_fit(_polynomial, thetas, tuning, p0=p0)

yfit = [_polynomial(xx, *tuple(coeff)) for xx in upsampled_thetas] # I'm sure there is a better
                                                    # way of doing this
pl.figure()
pl.plot(thetas, tuning, 'ko', label='Test data', )
pl.plot(upsampled_thetas, yfit, label='fitted data')



#%%

#%%

# #############################################################################
# Select TESTING data:
# #############################################################################
test_runid = 'gratings_rotating' #'blobs_dynamic_run6' #'blobs_dynamic_run1' #'blobs_dynamic_run1'
test_traceid = 'traces001'

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


#%%
# g(t) = exp(−t/tau_decay) − exp(−t/tau_rise)
        
        currtrials = list(set(labels_df[labels_df['config']==config]['trial']))
        currconfig_angles = np.mean(np.vstack([mwtrials[trial]['rotation_values'] for trial in currtrials]), axis=0)
        
        start_rot = currconfig_angles[0]
        end_rot = currconfig_angles[-1]
        
        stim_on = list(set(labels_df[labels_df['config']==config]['stim_on_frame']))[0]
        nframes_on = list(set(labels_df[labels_df['config']==config]['nframes_on']))[0]
        
        nframes_in_trial = np.squeeze(mean_pred[config]).shape[0]
        
        interp_angles = np.ones(predicted_vals.shape) * np.nan
        interp_angles[stim_on:stim_on + nframes_on] = np.linspace(start_rot, end_rot, num=nframes_on)
        
        # Convert to rads:
        interp_angles = np.array([ v*(math.pi/180) for v in interp_angles])


        # Generate a fake repsonse trace:
        curr_trace = 
        [v + 0.1*(random.random()*2. - 1.) for v in currtrace]














