#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  1 13:13:27 2018

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

from pipeline.python.utils import natural_keys, replace_root, print_elapsed_time
import pipeline.python.traces.combine_runs as cb
import pipeline.python.paradigm.align_acquisition_events as acq
import pipeline.python.visualization.plot_psths_from_dataframe as vis
from pipeline.python.traces.utils import load_TID

from mpl_toolkits.axes_grid1 import make_axes_locatable


#%%
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
def load_roi_dataframe(roidata_filepath):

    fn_parts = os.path.split(roidata_filepath)[-1].split('_')
    roidata_hash = fn_parts[1]
    trace_type = os.path.splitext(fn_parts[-1])[0]

    df_list = []

    #DATA = pd.read_hdf(combined_roidata_fpath, key=datakey, mode='r')

    df = pd.HDFStore(roidata_filepath, 'r')
    datakeys = df.keys()
    if 'roi' in datakeys[0]:
        for roi in datakeys:
            if '/' in roi:
                roiname = roi[1:]
            else:
                roiname = roi
            dfr = df[roi]
            dfr['roi'] = pd.Series(np.tile(roiname, (len(dfr .index),)), index=dfr.index)
            df_list.append(dfr)
        DATA = pd.concat(df_list, axis=0, ignore_index=True)
        datakey = '%s_%s' % (trace_type, roidata_hash)
    else:
        print "Found %i datakeys" % len(datakeys)
        datakey = datakeys[0]
        #df.close()
        #del df
        DATA = pd.read_hdf(roidata_filepath, key=datakey, mode='r')
        #DATA = df[datakey]
        df.close()
        del df

    return DATA, datakey

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

options = ['-D', '/mnt/odyssey', '-i', 'CE077', '-S', '20180425', '-A', 'FOV2_zoom1x',
           '-T', 'np_subtracted', '--no-pupil',
           '-R', 'gratings_run1', '-t', 'traces001',
           '-n', '1']





options = extract_options(options)

rootdir = options.rootdir
animalid = options.animalid
session = options.session
acquisition = options.acquisition
slurm = options.slurm
if slurm is True:
    rootdir = '/n/coxfs01/2p-data'

trace_type = options.trace_type

run_list = options.run_list
traceid_list = options.traceid_list

filter_pupil = options.filter_pupil
pupil_radius_max = float(options.pupil_radius_max)
pupil_radius_min = float(options.pupil_radius_min)
pupil_dist_thr = float(options.pupil_dist_thr)
pupil_max_nblinks = 0

multiproc = options.multiproc
nprocesses = int(options.nprocesses)
combined = options.combined
nruns = int(options.nruns)

acquisition_dir = os.path.join(rootdir, animalid, session, acquisition)
if combined is False:
    runfolder = run_list[0]
    traceid = traceid_list[0]
    with open(os.path.join(acquisition_dir, runfolder, 'traces', 'traceids_%s.json' % runfolder), 'r') as f:
        tdict = json.load(f)
    tracefolder = '%s_%s' % (traceid, tdict[traceid]['trace_hash'])
    traceid_dir = os.path.join(rootdir, animalid, session, acquisition, runfolder, 'traces', tracefolder)
else:
    assert len(run_list) == nruns, "Incorrect runs or number of runs (%i) specified!\n%s" % (nruns, str(run_list))
    if len(run_list) > 2:
        runfolder = '_'.join([run_list[0], 'thru', run_list[-1]])
    else:
        runfolder = '_'.join(run_list)
    if len(traceid_list)==1:
        if len(run_list) > 2:
            traceid = traceid_list[0]
        else:
            traceid = '_'.join([traceid_list[0] for i in range(nruns)])
    traceid_dir = os.path.join(rootdir, animalid, session, acquisition, runfolder, traceid)


print(traceid_dir)
assert os.path.exists(traceid_dir), "Specified traceid-dir does not exist!"


#%% # Load ROIDATA file:
print "Loading ROIDATA file..."

roidf_fn = [i for i in os.listdir(traceid_dir) if i.endswith('hdf5') and 'ROIDATA' in i and trace_type in i][0]
roidata_filepath = os.path.join(traceid_dir, roidf_fn) #'ROIDATA_098054_626d01_raw.hdf5')
DATA, datakey = load_roi_dataframe(roidata_filepath)

transform_dict, object_transformations = vis.get_object_transforms(DATA)
trans_types = object_transformations.keys()

#%% Set filter params:

if filter_pupil is True:
    pupil_params = acq.set_pupil_params(radius_min=pupil_radius_min,
                                        radius_max=pupil_radius_max,
                                        dist_thr=pupil_dist_thr,
                                        create_empty=False)
elif filter_pupil is False:
    pupil_params = acq.set_pupil_params(create_empty=True)


#%% Calculate metrics & get stats ---------------------------------------------

print "Getting ROI STATS..."
STATS, stats_filepath = cb.get_combined_stats(DATA, datakey, traceid_dir, trace_type=trace_type, filter_pupil=filter_pupil, pupil_params=pupil_params)


#%% Get stimulus config info:assign_roi_selectivity
# =============================================================================

rundir = os.path.join(rootdir, animalid, session, acquisition, runfolder)

if combined is True:
    stimconfigs_fpath = os.path.join(traceid_dir, 'stimulus_configs.json')
else:
    stimconfigs_fpath = os.path.join(rundir, 'paradigm', 'stimulus_configs.json')

with open(stimconfigs_fpath, 'r') as f:
    stimconfigs = json.load(f)

print "Loaded %i stimulus configurations." % len(stimconfigs.keys())


#%%
# =============================================================================
# Look at population activity:
# =============================================================================

output_dir = os.path.join(traceid_dir, 'figures', 'population')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print output_dir

#%%

configs = sorted([k for k in stimconfigs.keys()], key=lambda x: stimconfigs[x]['rotation'])
orientations = [stimconfigs[c]['rotation'] for c in configs]


#%% Use Z-SCORED responses :
# =============================================================================
#
#stats = STATS[['roi', 'config', 'trial', 'baseline_df', 'stim_df', 'zscore']] #STATS['zscore']
#
#std_baseline = stats['stim_df'].values / stats['zscore'].values
#zscored_resp = (stats['stim_df'].values - stats['baseline_df'].values ) /std_baseline
#
#zscore_vals = stats['zscore'].values

assert len(list(set(DATA['first_on'])))==1, "More than 1 frame idx found for stimulus ON"
assert len(list(set(DATA['nframes_on'])))==1, "More than 1 value found for nframes on."

stim_on_frame = int(list(set(DATA['first_on']))[0])
nframes_on = int(round(list(set(DATA['nframes_on']))[0]))

# Turn DF values into matrix with rows=trial, cols=df value for each frame:
roi_list = sorted(list(set(DATA['roi'])), key=natural_keys)
nrois = len(roi_list)


sDATA = DATA[['roi', 'config', 'trial', 'raw', 'df', 'tsec']].reset_index()
sDATA.loc[:, 'config'] = [stimconfigs[c]['rotation'] for c in sDATA.loc[:,'config'].values]
sDATA = sDATA.sort_values(by=['config', 'trial'], inplace=False)
sDATA.head()

nframes_per_trial = len(sDATA[sDATA['trial']=='trial00001']['tsec']) / nrois
ntrials_per_stim = len(list(set(sDATA[sDATA['config']==0]['trial']))) # Assumes all stim have same # trials!




#%%
# z-score each neuron's repsonse:
# zscore = ( mean(stimulus) - mean(baseline) ) / std(baseline)

trace = 'raw'
rawtraces = np.vstack((sDATA.groupby(['roi', 'config', 'trial'])[trace].apply(np.array)).as_matrix())

std_baseline_values = np.nanstd(rawtraces[:, 0:stim_on_frame], axis=1)
mean_baseline_values = np.nanmean(rawtraces[:, 0:stim_on_frame], axis=1)
mean_stim_on_values = np.nanmean(rawtraces[:, stim_on_frame:stim_on_frame+nframes_on], axis=1)

zscore_values_raw = np.array([meanval/stdval for (meanval, stdval) in zip(mean_stim_on_values, std_baseline_values)])
zscore_values_df = STATS.groupby(['roi', 'config', 'trial'])['zscore'].apply(np.mean).values
zscored_df = (mean_stim_on_values - mean_baseline_values ) / std_baseline_values
mean_stimdf_values =  STATS.groupby(['roi', 'config', 'trial'])['stim_df'].apply(np.mean).values

trial_labels = [t[2] for t in sDATA.groupby(['roi', 'config', 'trial'])[trace].apply(np.array).keys().tolist()]
config_labels = [t[1] for t in sDATA.groupby(['roi', 'config', 'trial'])[trace].apply(np.array).keys().tolist()]
roi_labels = [t[0] for t in sDATA.groupby(['roi', 'config', 'trial'])[trace].apply(np.array).keys().tolist()]
trial_list = list(set(trial_labels))
ntrials = len(trial_list)

# Reshape metrics as N x T, where N = nrois, T = ntrials total:
rois_by_stimraw = np.reshape(mean_stim_on_values, (nrois, ntrials))
rois_by_stimdf = np.reshape(mean_stimdf_values, (nrois, ntrials))
rois_by_zscore = np.reshape(zscored_df, (nrois, ntrials))
rois_by_zscore_raw = np.reshape(zscore_values_raw, (nrois, ntrials))
rois_by_zscore_df = np.reshape(zscore_values_df, (nrois, ntrials))
#np.array_equal(np.around(rois_by_zscore, decimals=5), np.around(rois_by_zscore_df, decimals=5))

# Get labels:
trial_labels = np.reshape(trial_labels, (nrois, ntrials))[0,:]
config_labels = np.reshape(config_labels, (nrois, ntrials))[0,:]


#%% Plot all ROIs' zscores:

zdf_cmap = mpl.cm.magma
zdf_vmax = 4.0

fig = pl.figure()
g = sns.heatmap(rois_by_zscore_df, cmap=zdf_cmap, vmax=zdf_vmax, cbar_kws={'label': 'zscored df/f'})

# label last trial with orientation of that trial group (trials grped by stimulus):
indices = { value : [ i for i, v in enumerate(config_labels) if v == value ] for value in orientations }
xtick_indices = [(k, indices[k][-1]) for k in indices.keys()]
g.set(xticks = [x[1] for x in xtick_indices])
g.set(xticklabels = [x[0] for x in xtick_indices])
g.set_xticklabels(g.get_xticklabels(), rotation=45, fontsize=8)
g.set(xlabel='trial')

# rotate ytick labels:
g.set_yticklabels(g.get_yticklabels(), rotation=45, fontsize=8)

pl.title('Z-scored responses - all trials')
pl.savefig(os.path.join(output_dir, 'zscored_dff_all_trials.png'))

#%% Train and test linear SVM using zscored response values:


from sklearn.model_selection import LeavePGroupsOut, LeaveOneGroupOut, LeaveOneOut
from sklearn import metrics


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
    pl.title('best C: %.4f' % best_C)

    figname = 'crossval_scores_by_C_%s.png' % classifier_str
    pl.savefig(os.path.join(output_dir, figname))

    return best_C


#%% ZSCORE --format data


# Format input data:
# ------------------
cX = rois_by_zscore.T
cy = config_labels
groups = [int(t[5:]) for t in trial_labels]
print "X:", cX.shape
print "y:", cy.shape


#%% ZSCORE -- Train classifier with ZSCORED data:
# -----------------------------------------------------------------------------

classifier = 'LinearSVC' #'LinearSVC' # 'OneVRest'
classifier_str = '%s_zscores' % (classifier)

# Create classifier:
# ------------------
#if X_train.shape[0] > X_train.shape[1]: # nsamples > nfeatures
#    dual = False
multi_class = 'ovr' # one-vs-rest classifier

if classifier == 'LinearSVC':
    svc = LinearSVC(random_state=0, dual=False, multi_class=multi_class)
else:
    svc = OneVsRestClassifier(SVC(kernel='linear'))


# Find best value for C:
# ----------------------
find_C = False
if find_C:
    svc.C = get_best_C(svc, cX, cy, output_dir=output_dir, classifier_str=classifier_str)

#%%  ZSCORE -- cross-validate
# -----------------------------------------------------------------------------

# Split Test/Training data:

splitter = 'LOO' # "LOGO" "LPGO"

if splitter=='LOGO':
    loo = LeaveOneGroupOut()
elif splitter=='LOO':
    loo = LeaveOneOut()
elif splitter=='LPGP':
    loo = LeavePGroupsOut()

pred_results = []
pred_true = []
for train, test in loo.split(cX, cy, groups=groups):
    #print train, test
    X_train, X_test = cX[train], cX[test]
    y_train, y_test = cy[train], cy[test]

    y_pred = svc.fit(X_train, y_train).predict(X_test)

    pred_results.append(y_pred) #=y_test])
    pred_true.append(y_test)

y_pred = np.array([int(i) for i in pred_results])
y_test = np.array([int(i) for i in pred_true])

avg_score = np.array([int(p==t) for p,t in zip(pred_results, pred_true)]).mean()

cv = loo.split(cX, cy, groups=groups)
cross_val_scores = cross_val_score(svc, cX, cy, cv=cv)
print "LOGO cross-validation scores:", cross_val_scores
print("Accuracy: %0.2f (+/- %0.2f)" % (cross_val_scores.mean(), cross_val_scores.std() * 2))

#%%  ZSCORE -- compute confusion matrix and save fig:
# -------------------------------------------

classif_identifier = '%ioris_zscored_%s_%s' %  (len(orientations), classifier, splitter)

cnf_matrix = confusion_matrix(y_test, y_pred, labels=orientations)

fig = pl.figure(figsize=(10,4))
ax = fig.add_subplot(1,2,1)
plot_confusion_matrix(cnf_matrix, classes=orientations, ax=ax, normalize=False,
                      title='Confusion matrix (zscored resp, %s)' % splitter)
ax = fig.add_subplot(1,2,2)
plot_confusion_matrix(cnf_matrix, classes=orientations, ax=ax, normalize=True,
                      title='Normalized')

figname = 'confusion_matrix_%s.png' % classif_identifier

pl.savefig(os.path.join(output_dir, figname))

print metrics.classification_report(y_test, y_pred, target_names=[str(c) for c in orientations])


# Save cross-validation results:
cv_outfile = 'CV_report_%s.txt' % classif_identifier

f = open(os.path.join(output_dir, cv_outfile), 'w')
f.write(metrics.classification_report(y_test, y_pred, target_names=[str(c) for c in orientations]))
f.close()

cv_results = {'predicted': list(y_pred),
              'true': list(y_test),
              'classifier': classifier,
              'splitter': splitter
              }
cv_resultsfile = 'CV_results_%s.json' % classif_identifier
with open(os.path.join(output_dir, cv_resultsfile), 'w') as f:
    json.dump(cv_results, f, sort_keys=True, indent=4)




#%%  TIMECOURSE

# =============================================================================
# Use a 2D sample for a given trial (EXAMPLE using DIGITS dataset):
# =============================================================================

# Each TRIAL is a sample composed of N rois and t frames, i.e., each sample is
# of size [N,t]. We need to "flatten" this to turn the data into matrix
# of shape [nsamples, nfeatures]:

#X = np.vstack([vals for vals in sDATA.groupby(['trial'])['df'].apply(np.array).values])
#trial_names = sDATA.groupby(['trial'])['df'].apply(np.array).index.tolist()
#y = np.array([sDATA[sDATA['trial']==trial]['config'].values[0] for trial in trial_names])

#%% TIMECOURSE --format input data

sample_frames = False #True

trial_list = sorted(list(set(sDATA['trial'])), key=natural_keys)

sample_list = []
class_list = []
for trial in trial_list:
    img = np.vstack([vals for vals in sDATA[sDATA['trial']==trial].groupby(['roi'])['df'].apply(np.array).values])
    #roi_names = sDATA[sDATA['trial']==trial].groupby(['roi'])['df'].apply(np.array).index.tolist()
    curr_config = sDATA[sDATA['trial']==trial]['config'].values[0]  #[0]
    sample_list.append(img)
    class_list.append(curr_config)


y = np.hstack([np.tile(c, (nframes_per_trial, )) for c in class_list])
X = np.vstack([s.T for s in sample_list])
groups = np.hstack([np.tile(c, (nframes_per_trial, )) for c in range(len(trial_list))]) #y])

# Check that PSTHs match:
troi = 10
for o in orientations:
    tmat = []
    pl.figure()
    ixs = np.where(y==o)
    tr2 = np.squeeze(X[ixs,troi])
    print tr2.shape
    for t in range(ntrials_per_stim):
        currtrace = tr2[t*nframes_per_trial:t*nframes_per_trial+nframes_per_trial]
        pl.plot(currtrace, 'k', linewidth=0.5)
        tmat.append(currtrace)
    tmat = np.array(tmat)
    pl.plot(np.mean(tmat, axis=0), 'r')
    pl.title(o)



# Reshape:

#if sample_frames:
#    y = np.hstack([np.tile(c, (nframes_per_trial, )) for c in cy])
#    X = np.vstack([s.T for s in sample_list])
#    groups = np.hstack([np.tile(c, (nframes_per_trial, )) for c in range(len(trial_list))]) #y])
#
#else:
#    y = np.array(class_list)
#    X = np.array(sample_list)
#    groups = None
#
#    # Reshape each sample (which is of shape [n_rois, n_frames] for each trial):
#    n_samples = len(sample_list)
#    X = X.reshape((n_samples, -1))

print "N samples: %i, N features: %i" % (X.shape[0], X.shape[1])

#%% TIMESERIES -- Create a classifier:

if sample_frames:
    classifier = 'LinearSVC' # 'OneVRest'
    classifier_str = '%s_tpoints' % (classifier)

    if classifier == 'LinearSVC':
        svc = LinearSVC(random_state=0, dual=False, multi_class='ovr')
    else:
        svc = OneVsRestClassifier(SVC(kernel='linear'))


    find_C = False
    if find_C:
        svc.C = get_best_C(svc, X, y, output_dir=output_dir, classifier_str=classifier_str)


#%% We learn the digits on the first half of the digits

if sample_frames:
    # Cross-validate for t-series samples:

    splitter = 'LPGO' #'LOO' #'splithalf' #'LOO'
    groups = None
    n_samples = X.shape[0]


    if splitter=='splithalf':
        svc.fit(X[:n_samples // 2], y[:n_samples // 2])

        # Now predict the value of the digit on the second half:
        y_test = y[n_samples // 2:]
        y_pred = svc.predict(X[n_samples // 2:])

    elif splitter=='kfold':
        loo = cross_validation.StratifiedKFold(y, n_folds=5)
        pred_results = []
        pred_true = []
        for train, test in loo: #, groups=groups):
            print train, test
            X_train, X_test = X[train], X[test]
            y_train, y_test = y[train], y[test]

            y_pred = svc.fit(X_train, y_train).predict(X_test)

            pred_results.append(y_pred) #=y_test])
            pred_true.append(y_test)

        #y_pred = np.array([int(i) for i in pred_results])
        #y_test = np.array([int(i) for i in pred_true])

        # Find "best fold"?
        avg_scores = []
        for y_pred, y_test in zip(pred_results, pred_true):
            pred_score = len([i for i in y_pred==y_test if i]) / float(len(y_pred))
            avg_scores.append(pred_score)
        best_fold = avg_scores.index(np.max(avg_scores))
        folds = [i for i in enumerate(loo)]
        train = folds[best_fold][1][0]
        test = folds[best_fold][1][1]
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        y_pred = pred_results[best_fold]

    else:
        if splitter=='LOGO':
            loo = LeaveOneGroupOut()
        elif splitter=='LOO':
            loo = LeaveOneOut()
        elif splitter=='LPGO':
            loo = LeavePGroupsOut(5)

        pred_results = []
        pred_true = []
        for train, test in loo.split(X, y, groups=groups): #, groups=groups):
            print train, test
            X_train, X_test = X[train], X[test]
            y_train, y_test = y[train], y[test]

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
            folds = [i for i in enumerate(loo.split(X, y, groups=groups))]
            train = folds[best_fold][1][0]
            test = folds[best_fold][1][1]
            X_train, X_test = X[train], X[test]
            y_train, y_test = y[train], y[test]
            y_pred = pred_results[best_fold]

            pl.figure();
            sns.distplot(avg_scores, kde=False)

            isgroup = 'grouped'

        else:

            y_pred = np.array([int(i) for i in pred_results])
            y_test = np.array([int(i) for i in pred_true])

            isgroup = '1sample'

    print("Classification report for classifier %s:\n%s\n"
          % (svc, metrics.classification_report(y_test, y_pred)))

    # Visualize PREDICTED img:
    # ------------------------------------
    #pl.figure()
    #images_and_predictions = list(zip(sample_list[n_samples // 2:], predicted))
    #for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    #    pl.subplot(2, 4, index + 5)
    #    pl.axis('off')
    #    pl.imshow(image, cmap=pl.cm.gray_r, interpolation='nearest')
    #    pl.title('Prediction: %i' % prediction)

    #cv_resultsfile = 'cross_validation_results_%s_%s_%s.json' % (classifier, splitter, isgroup)
    #with open(os.path.join(output_dir, cv_resultsfile), 'r') as f:
    #    cv_results = json.load(f)


#%% Compute confusion matrix:
# --------------------------------------

if sample_frames:

    classif_identifier = '%ioris_tseries_%s_%s' %  (len(orientations), classifier, splitter)


    average_iters = False
    if groups is not None:
        if average_iters:
            cmatrix_tframes = confusion_matrix(pred_true[0], pred_results[0], labels=orientations)
            for iter_idx in range(len(pred_results))[1:]:
                cmatrix_tframes += confusion_matrix(pred_true[iter_idx], pred_results[iter_idx], labels=orientations)
            conf_mat_str = 'AVG'
        else:
            cmatrix_tframes = confusion_matrix(pred_true[best_fold], pred_results[best_fold], labels=orientations)
            conf_mat_str = 'best'
    else:
        cmatrix_tframes = confusion_matrix(y_test, y_pred, labels=orientations)
        conf_mat_str = 'single'

    fig = pl.figure(figsize=(10,4))
    ax1 = fig.add_subplot(1,2,1)
    plot_confusion_matrix(cmatrix_tframes, classes=orientations, ax=ax1, normalize=False,
                      title='Confusion matrix (all tpoints, %s)' % splitter)
    ax2 = fig.add_subplot(1,2,2)
    plot_confusion_matrix(cmatrix_tframes, classes=orientations, ax=ax2, normalize=True,
                          title='Normalized')

    figname = 'confusion_matrix_%ioris_tpoints_%s_%s_%s_iters.png' % (len(orientations), splitter, isgroup, conf_mat_str)
    pl.savefig(os.path.join(output_dir, figname))


    #%

    # Save, cuz takes forever:
    cv_outfile = 'CV_report_%s.txt' % classif_identifier

    f = open(os.path.join(output_dir, cv_outfile), 'w')
    f.write(metrics.classification_report(y_test, y_pred, target_names=[str(c) for c in orientations]))
    f.close()

    cv_results = {'predicted': list(y_pred),
                  'true': list(y_test),
                  'classifier': classifier,
                  'splitter': splitter
                  }
    cv_resultsfile = 'CV_results_%s.json' % classif_identifier
    with open(os.path.join(output_dir, cv_resultsfile), 'w') as f:
        json.dump(cv_results, f, sort_keys=True, indent=4)


#%%

# smooth that shit:
from statsmodels.nonparametric.smoothers_lowess import lowess


frac = 0.002

def smooth_traces(trace):
    xvals = np.arange(len(trace))
    filtered = lowess(trace, xvals, is_sorted=True, frac=0.002, it=0)
    return filtered[:, 1]

trace = X[0:358, 10]

filtered = lowess(trace, xvals, is_sorted=True, frac=0.002, it=0)

trace_test = X[:, 10:13]
print trace_test.shape


trace_test_filt = np.apply_along_axis(smooth_traces, 0, trace_test)
print trace_test_filt.shape


pl.figure();
pl.plot(xvals, trace, 'r')
pl.plot(filtered[:,0], filtered[:,1], 'k')

pl.plot(xvals, trace_test_filt[:, 0])


#%%

Xf = np.apply_along_axis(smooth_traces, 0, X)
print Xf.shape

#%%

from sklearn.preprocessing import StandardScaler


cX_std = StandardScaler(with_std=True).fit_transform(cX)
#lsvm = LinearSVC(random_state=0, dual=False, multi_class='ovr')

X_std = StandardScaler().fit_transform(Xf)


sample_frames = True

if classifier == 'LinearSVC':
    lsvm = LinearSVC(random_state=0, dual=False, multi_class='ovr')

else:
    lsvm = OneVsRestClassifier(SVC(kernel='linear', decision_function_shape='ovr'))

if sample_frames:
    lsvm.fit(X_std, y)
else:
    lsvm.fit(cX_std, cy)


#%%
# Predict confidence scores for samples:  this is the signed distance of a given
# sample to the hyperplane.

# lsvm.decition_function(X)
# - returns confidence scores per (sample,class) combo.
# - in OvR, this will be of shape [n_samples, n_classes].

dec = lsvm.decision_function(X_std)
w_norm = np.linalg.norm(lsvm.coef_)
dist = dec / w_norm

for idx, ori in enumerate(lsvm.classes_):
    confidence_scores = dec[:, idx] # Predicted score for current class (for all trials)
    print [i for i,v in enumerate(confidence_scores) if v > 0] == [i for i,v in enumerate(y) if v == ori]


# Take 2 class example:
# Class 1:  w . x + b > 0
# Class 2:  w . x + b < 0
# margin:  x . x + b = 0

'''
lsvm.coef_      : (n_classes, n_features)
lsvm.intercept_ : (n_classes,)
X               : (n_samples, n_features)
'''

# Look at class 0 vs. rest:
class_assignment = [i for i, v in enumerate((lsvm.coef_[0,:].dot(cX_std.T) + lsvm.intercept_[0]) > 0) if v]
true_label = [i for i, v in enumerate(cy) if v == 0]
class_assignment == true_label


#%%

# Project population timecourse as a sanity check:
p0 = lsvm.coef_[0,:].dot(cX_std.T) + lsvm.intercept_[0]
p1 = lsvm.coef_[1,:].dot(cX_std.T) + lsvm.intercept_[1]
pl.figure()
pl.scatter(p0, p1)


# Project population timecourse as a sanity check:
p0 = lsvm.coef_[0,:].dot(X_std.T) + lsvm.intercept_[0]
p1 = lsvm.coef_[1,:].dot(X_std.T) + lsvm.intercept_[1]
pl.figure()
pl.scatter(p0, p1)


#%%

# Project zscore data onto vectors normal to each hyperplane

# This is essentially a "distance" matrix centered around 0, since we are
# projecting each sample in 105-D space onto the normal:

if sample_frames:
    cdata = np.array([lsvm.coef_[c].dot(X_std.T) + lsvm.intercept_[c] for c in range(len(lsvm.classes_))])

else:
    cdata = np.array([lsvm.coef_[c].dot(cX_std.T) + lsvm.intercept_[c] for c in range(len(lsvm.classes_))])

print cdata.shape

pl.figure()
g = sns.heatmap(cdata, cmap='hot')

indices = { value : [ i for i, v in enumerate(config_labels) if v == value ] for value in orientations }
xtick_indices = [(k, indices[k][-1]) for k in indices.keys()]
g.set(xticks = [x[1] for x in xtick_indices])
g.set(xticklabels = [x[0] for x in xtick_indices])
g.set_xticklabels(g.get_xticklabels(), rotation=45, fontsize=8)
g.set(xlabel='trial')

pl.title('zscored data proj onto normals')

figname = 'proj_normals_all_zscored.png'
pl.savefig(os.path.join(output_dir, figname))


#
#np.random.seed(0)
#testX = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
#testY = [0] * 20 + [1] * 20


## fit the model
#testclf = svm.SVC(kernel='linear')
#testclf.fit(testX, testY)
#
## get the separating hyperplane
#w = testclf.coef_[0]
#a = -w[0] / w[1]
#xx = np.linspace(-5, 5)
#yy = a * xx - (clf.intercept_[0]) / w[1]


#%%

from sklearn.decomposition import PCA

# PCA the distance matrix:

pX = cdata.T # (n_samples, n_features)

pca = PCA(n_components=2)

principal_components = pca.fit_transform(pX)

pca_df = pd.DataFrame(data=principal_components,
                      columns=['pc1', 'pc2'])
labels_df = pd.DataFrame(data=cy,
                         columns=['target'])

pdf = pd.concat([pca_df, labels_df], axis=1)


# Visualize 2D projection:
fig = pl.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('Hyperplane distances PCA', fontsize = 20)
colors = ['r', 'y', 'g', 'b', 'm', 'k', 'c', 'orange']
for target, color in zip(orientations,colors):
    indicesToKeep = pdf['target'] == target
    ax.scatter(pdf.loc[indicesToKeep, 'pc1']
               , pdf.loc[indicesToKeep, 'pc2']
               , c = color
               , s = 50)
ax.legend(orientations)
ax.grid()

if sample_frames:
    pl.title('PCA, proj norm (time-series)')
else:
    pl.title('PCA, proj norm (zscores)')

figname = 'PCA_proj_norm_%s.png' % classif_identifier
pl.savefig(os.path.join(output_dir, figname))



#%%

from sklearn.manifold import TSNE
import time

# try t-SNE:

tsne_df = pd.DataFrame(data=pX,
                       index=np.arange(0, pX.shape[0]),
                       columns=[str(ori) for ori in orientations])

feat_cols = [f for f in tsne_df.columns]

time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(tsne_df[feat_cols].values)
print 't-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start)

# Visualize:
#target_ids = range(len(digits.target_names))
target_ids = range(len(orientations))


fig, ax = pl.subplots(1, figsize=(6, 6))
colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange' #, 'purple'
#for i, c, label in zip(target_ids, colors, digits.target_names):
#    plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], c=c, label=label)
for i, c, label in zip(target_ids, colors, orientations):
    pl.scatter(tsne_results[cy == int(label), 0], tsne_results[cy == int(label), 1], c=c, label=label)
    box = ax.get_position()
    ax.set_position([box.x0 + box.width * 0.01, box.y0 + box.height * 0.02,
                     box.width * 0.98, box.height * 0.98])

# Put a legend below current axis
pl.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=len(orientations)/2)

if sample_frames:
    pl.title('t-SNE, proj norm (time-series)')
else:
    pl.title('t-SNE: proj norm (zscores)')

figname = 'tSNE_proj_norm_%s.png' % classif_identifier
pl.savefig(os.path.join(output_dir, figname))

#%% Project time-courses onto hyperplane from classifier:


# z-score time-courses:
#X_std = StandardScaler(with_std=True).fit_transform(X)

# Reshape so that each "trial" is n_neurons x n_tpoints:
n_samples = X_std.shape[0]
ntrials = 80

Xr = np.reshape(X_std, (ntrials, nframes_per_trial, nrois))
Xr = np.swapaxes(Xr, 1, 2)
yr = np.reshape(y, (ntrials, nframes_per_trial))
yr = yr[:,0]

#
#sample_frames = True
#if sample_frames:
#    Xr = np.reshape(X_std, (ntrials, nframes_per_trial, nrois))
#    Xr = np.swapaxes(Xr, 1, 2)
#    yr = np.reshape(y, (ntrials, nframes_per_trial))
#    yr = yr[:,0]
#else:
#    Xr = np.reshape(X_std, (ntrials, nrois, nframes_per_trial))
#    yr = y.copy()

print "X reshaped:", Xr.shape
print "Y reshaped:", yr.shape

#%% Check PSTH for reshaped:
troi = 10
o = 90
#for o in orientations:

tmat = []
pl.figure()
ixs = np.where(y==o)
tr2 = np.squeeze(Xf[ixs,10])
print tr2.shape
for t in range(10):
    currtrace = tr2[t*nframes_per_trial:t*nframes_per_trial+nframes_per_trial]
    pl.plot(currtrace, 'k', linewidth=0.5)
    tmat.append(currtrace)
tmat = np.array(tmat)
pl.plot(np.mean(tmat, axis=0), 'r')
pl.title(o)


ixsr = np.where(yr==o)
tr_r = np.squeeze(Xr[ixsr,troi, :])
print tr_r.shape
tmat = []

pl.figure()
for t in range(10):
    currtrace = tr2[t*nframes_per_trial:t*nframes_per_trial+nframes_per_trial]
    pl.plot(currtrace, 'k', linewidth=0.5)
    tmat.append(currtrace)
tmat = np.array(tmat)
pl.plot(np.mean(tmat, axis=0), 'r')
pl.title(o)


#%%

# Project time series for each trial onto norm vector of hyperplane:

stim0 = 0
stim1 = 90

stim0_idx = [i for i in lsvm.classes_].index(stim0)
stim1_idx = [i for i in lsvm.classes_].index(stim1)


# Get indices for trials of each stim type:
idx0 = np.where(yr==stim0)
idx1 = np.where(yr==stim1)

# Get the traces for each stimulus:
trace0 = np.squeeze(Xr[idx0, :, :])
trace1 = np.squeeze(Xr[idx1, :, :])
print trace1.shape


p0_list = [lsvm.coef_[stim0_idx,:].dot(trace0[t, :]) + lsvm.intercept_[stim0_idx] for t in range(ntrials_per_stim)]
p1_list = [lsvm.coef_[stim1_idx,:].dot(trace1[t, :]) + lsvm.intercept_[stim1_idx] for t in range(ntrials_per_stim)]

nframes = len(p0_list[0])
tag = np.arange(0, nframes)

cmap = pl.cm.jet                                        # define the colormap
cmaplist = [cmap(i) for i in range(cmap.N)]             # extract all colors from the .jet map
cmaplist[0] = (.5,.5,.5,1.0)                            # force the first color entry to be grey
cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)  # create the new map


# define the bins and normalize
bounds = np.linspace(0,len(tag),len(tag)+1)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

fig, axes = pl.subplots(2,5, sharex=True, sharey=True, figsize=(10,4))
pidx = 0
axes = axes.reshape(-1)

for p0, p1 in zip(p0_list, p1_list):
    ax = axes[pidx]
    ax.scatter(p0, p1, c=tag, cmap=cmap, norm=norm, vmin=0, vmax=nframes, s=4, alpha=0.5)
    if pidx==5: # bottom left corner plot
        ax.set(xlabel='%i' % stim0)
        ax.set(ylabel='%i' % stim1)

    pidx += 1
# create a second axes for the colorbar
ax2 = fig.add_axes([0.91, 0.1, 0.005, 0.8])
cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm, spacing='proportional', boundaries=bounds) #, format='%1i')
ax2.set_ylabel('time', size=10)

pl.suptitle('proj norm of hyperplane x%i, y%i' % (stim0, stim1))

figname = '%s_fit_proj_tseries_normals_x%i_y%i_lowess%.3f.png' % (classif_identifier, stim0, stim1, frac)

pl.savefig(os.path.join(output_dir, figname))

#%%

# Project time courses, use white noise:

Nr = np.random.uniform(low=Xr.min(), high=Xr.max(), size=Xr.shape)


# Get the traces for each stimulus:
trace0 = np.squeeze(Nr[idx0, :, :])
trace1 = np.squeeze(Nr[idx1, :, :])
print trace1.shape


p0_list = [lsvm.coef_[stim0_idx,:].dot(trace0[t, :]) + lsvm.intercept_[stim0_idx] for t in range(ntrials_per_stim)]
p1_list = [lsvm.coef_[stim1_idx,:].dot(trace1[t, :]) + lsvm.intercept_[stim1_idx] for t in range(ntrials_per_stim)]

nframes = len(p0_list[0])
tag = np.arange(0, nframes)

cmap = pl.cm.jet                                        # define the colormap
cmaplist = [cmap(i) for i in range(cmap.N)]             # extract all colors from the .jet map
cmaplist[0] = (.5,.5,.5,1.0)                            # force the first color entry to be grey
cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)  # create the new map


# define the bins and normalize
bounds = np.linspace(0,len(tag),len(tag)+1)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

fig, axes = pl.subplots(2,5, sharex=True, sharey=True, figsize=(10,4))
pidx = 0
axes = axes.reshape(-1)

for p0, p1 in zip(p0_list, p1_list):
    ax = axes[pidx]
    ax.scatter(p0, p1, c=tag, cmap=cmap, norm=norm, vmin=0, vmax=nframes, s=4, alpha=0.5)
    if pidx==5: # bottom left corner plot
        ax.set(xlabel='%i' % stim0)
        ax.set(ylabel='%i' % stim1)

    pidx += 1
# create a second axes for the colorbar
ax2 = fig.add_axes([0.91, 0.1, 0.005, 0.8])
cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm, spacing='proportional', boundaries=bounds) #, format='%1i')
ax2.set_ylabel('time', size=10)

pl.suptitle('proj norm of hyperplane x%i, y%i (rand)' % (stim0, stim1))

figname = '%s_fit_proj_tseries_normals_x%i_y%i_lowess%.3f_NOISE.png' % (classif_identifier, stim0, stim1, frac)

pl.savefig(os.path.join(output_dir, figname))


#%%

# Project ENTIRE series (don't split up trials):

proj_all = lsvm.coef_.dot(X_std.T)
print proj_all.shape

nframes_total = proj_all.shape[1]

tag = np.arange(0, nframes_total)

cmap = pl.cm.jet                                        # define the colormap
cmaplist = [cmap(i) for i in range(cmap.N)]             # extract all colors from the .jet map
cmaplist[0] = (.5,.5,.5,1.0)                            # force the first color entry to be grey
cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)  # create the new map

# define the bins and normalize
bounds = np.linspace(0,len(tag),len(tag)+1)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)



fig = pl.figure();
pl.scatter(proj_all[stim0_idx,:], proj_all[stim1_idx,:], c=tag, cmap=cmap, norm=norm, vmin=0, vmax=nframes_total, s=4, alpha=0.5)
pl.xlabel('0')
pl.ylabel('90')
ax2 = fig.add_axes([0.91, 0.1, 0.005, 0.8])
cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm, spacing='proportional', boundaries=bounds) #, format='%1i')
ax2.set_ylabel('time', size=10)

figname = '%s_fit_zscore_proj_tseries_normals_x%i_y%i_lowess%.3f_FULL.png' % (classif_identifier, stim0, stim1, frac)
pl.savefig(os.path.join(output_dir, figname))

#%%

# Project time-series data onto hyperplanes found by LinearSVM trained on zscores:


#pl.figure()
#pl.scatter(proj, proj1)
#
#
#stim0 = 0
#stim1 = 90
#
#stim0_idx = [i for i in lsvm.classes_].index(stim0)
#stim1_idx = [i for i in lsvm.classes_].index(stim1)
#
#
## Project timecourses onto hyperplanes:
#proj0 = lsvm.coef_[stim0_idx,:].dot(X_std.T) + lsvm.intercept_[stim0_idx]
#proj1 = lsvm.coef_[stim1_idx,:].dot(X_std.T) + lsvm.intercept_[stim1_idx]
#
#
## Get indices for trials of each stim type:
#idx0 = np.where(y==stim0)
#idx1 = np.where(y==stim1)
#
#X0 = np.squeeze(proj0[idx0])
#X1 = np.squeeze(proj1[idx1])
#print "Data subset for current angle:", X0.shape
#
#print "N trials per stim:", ntrials_per_stim
#
#
##colorvals = sns.color_palette("Blues", nframes)
#tag = np.arange(0, nframes_per_trial)
## define the colormap
#cmap = pl.cm.jet
## extract all colors from the .jet map
#cmaplist = [cmap(i) for i in range(cmap.N)]
## force the first color entry to be grey
#cmaplist[0] = (.5,.5,.5,1.0)
## create the new map
#cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
## define the bins and normalize
#bounds = np.linspace(0,len(tag),len(tag)+1)
#norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
#
#
#
#fig, axes = pl.subplots(2,5, sharex=True, sharey=True, figsize=(10,4))
#pidx = 0
#axes = axes.reshape(-1)
#for ti in range(ntrials_per_stim):
#    ax = axes[pidx]
#
##    vals = X_std[ti*nframes_per_trial:ti*nframes_per_trial+nframes_per_trial, 10]
##    ax.plot(vals)
##
#
#    x0_proj_values = X0[ti*nframes_per_trial:ti*nframes_per_trial+nframes_per_trial]
#    x1_proj_values = X1[ti*nframes_per_trial:ti*nframes_per_trial+nframes_per_trial]
#
#    ax.scatter(x0_proj_values, x1_proj_values, c=tag, cmap=cmap, norm=norm, vmin=0, vmax=nframes, s=4, alpha=0.5)
#    if pidx==5: # bottom left corner plot
#        ax.set(xlabel='%i' % stim0)
#        ax.set(ylabel='%i' % stim1)
#    pidx += 1
## create a second axes for the colorbar
#ax2 = fig.add_axes([0.91, 0.1, 0.005, 0.8])
#cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm, spacing='proportional', boundaries=bounds) #, format='%1i')
#ax2.set_ylabel('time', size=10)
#
#pl.suptitle('popN timecourse projected onto hyperplanes')
#
#figname = 'tcourse_proj_zscore_hplanes_x%i_y%i.png' % (stim0, stim1)
#pl.savefig(os.path.join(output_dir, figname))



#%%

# Plot "average" time-series of ROI population:
# Get indices for trials of each stim type:

idx0 = np.where(yr==stim0)
idx1 = np.where(yr==stim1)

# Get the traces for each stimulus:
trace0 = np.squeeze(Xr[idx0, :, :])
trace1 = np.squeeze(Xr[idx1, :, :])
print trace1.shape


d0_list = [np.mean(trace0[t, :], axis=0) for t in range(ntrials_per_stim)]
d1_list = [np.mean(trace1[t, :], axis=0) for t in range(ntrials_per_stim)]


tag = np.arange(0, nframes)

cmap = pl.cm.jet                                        # define the colormap
cmaplist = [cmap(i) for i in range(cmap.N)]             # extract all colors from the .jet map
cmaplist[0] = (.5,.5,.5,1.0)                            # force the first color entry to be grey
cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)  # create the new map

# define the bins and normalize
bounds = np.linspace(0,len(tag),len(tag)+1)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)



fig, axes = pl.subplots(2,5, sharex=True, sharey=True, figsize=(10,4))
pidx = 0
axes = axes.reshape(-1)

for d0, d1 in zip(d0_list, d1_list):
    ax = axes[pidx]
    ax.scatter(d0, d1, c=tag, cmap=cmap, norm=norm, vmin=0, vmax=nframes, s=4, alpha=0.5)
    if pidx==5: # bottom left corner plot
        ax.set(xlabel='%i' % stim0)
        ax.set(ylabel='%i' % stim1)
    pidx += 1
# create a second axes for the colorbar
ax2 = fig.add_axes([0.91, 0.1, 0.005, 0.8])
cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm, spacing='proportional', boundaries=bounds) #, format='%1i')
ax2.set_ylabel('time', size=10)

pl.suptitle('average popN response')

figname = 'tseries_by_trials_x%i_y%i_lowess%.3f.png' % (stim0, stim1, frac)


pl.savefig(os.path.join(output_dir, figname))


#%% Plot probability of class label as function of time:


from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

#clf = CalibratedClassifierCV(lsvm)
#clf.fit(cX_std, y)
#y_proba = clf.predict_proba(X_test)

clf = LogisticRegression(multi_class='ovr', solver='liblinear')
clf.fit(X_train, y_train)
y_proba = clf.predict_proba(X_test)
y_proba.shape

# Split frames into trials:
n_test_trials = y_proba.shape[0] / nframes_per_trial

colorvals = sns.color_palette("Spectral", len(clf.classes_))

fig, axes = pl.subplots(1, n_test_trials, sharex=True, sharey=True, figsize=(10,4)) #pl.figure()
axes = axes.reshape(-1)
for t in range(n_test_trials):
    ax = axes[t]
    true_label = list(set(y_test[t*nframes_per_trial:t*nframes_per_trial + nframes_per_trial]))
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








#clf = LogisticRegression(multi_class='ovr', solver='liblinear')
clf.fit(X, y)
y_proba_full = clf.predict_proba(X)
print y_proba_full.shape
n_trials_split = y_proba_full.shape[0] / nframes_per_trial

# look at one angle first:
stim = 90
stim_idx = [i for i in clf.classes_].index(stim)
label_idxs = [yi for yi, ori in enumerate(y) if ori==stim]
proba_subset = y_proba_full[label_idxs,:]
print proba_subset.shape
n_trials_split = proba_subset.shape[0] / nframes_per_trial
print "Curr angle: %i (%i trials)" % (stim, n_trials_split)


fig, axes = pl.subplots(2, n_trials_split/2, sharex=True, sharey=True, figsize=(10,4)) #pl.figure()
axes = axes.reshape(-1)
for t in range(n_trials_split):
    ax = axes[t]
    true_label = list(set(y[t*nframes_per_trial:t*nframes_per_trial + nframes_per_trial]))
    for c in range(proba_subset.shape[1]):
        ax.plot(proba_subset[t*nframes_per_trial:t*nframes_per_trial + nframes_per_trial, c], label=clf.classes_[c],
                    linewidth=1, alpha=0.8, color=colorvals[c])
    ax.set_title('%s' % str(stim))
    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.8])

# Put a legend below current axis
pl.legend(loc='lower center', bbox_to_anchor=(-1.8, -0.3),
          fancybox=True, shadow=False, ncol=len(clf.classes_))
sns.despine(offset=4)


















#%%
pl.figure()
colors = "rygb"
for i, color in zip(lsvm.classes_[1:5], colors[1:5]):
    idx0 = np.where(y == 0)
    idx1 = np.where(y == i)
    pl.scatter(X[idx0, :], X[idx1, :], c=color, cmap=pl.cm.Paired,
                edgecolor='black', s=20)

# Plot the three one-against-all classifiers
xmin, xmax = pl.xlim()
ymin, ymax = pl.ylim()
coef = lsvm.coef_
intercept = lsvm.intercept_

def plot_hyperplane(c, color):
    def line(x0):
        return (-(x0 * coef[c, :]) - intercept[c]) / coef[c, :]

    pl.plot([xmin, xmax], [line(xmin), line(xmax)],
             ls="--", color=color)

for c, (ori, color) in enumerate(zip(lsvm.classes_, colors)):
    plot_hyperplane(c, color)

#%%
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()
cv.fit(X)
print len(cv.vocabulary_)
print cv.get_feature_names()
X_train = cv.transform(data)

svm = LinearSVC()
svm.fit(X_train, target)


#def plot_coefficients(classifier, feature_names, top_features=20):
top_features = 20
coef = lsvm.coef_.ravel()
top_positive_coefficients = np.argsort(coef)[-top_features:]
top_negative_coefficients = np.argsort(coef)[:top_features]
top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])

# create plot
pl.figure(figsize=(15, 5))
colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
pl.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
feature_names = np.array(feature_names)
pl.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha='right')
pl.show()


#%%
def project_points(x, y, z, a, b, c):
    """
    Projects the points with coordinates x, y, z onto the plane
    defined by a*x + b*y + c*z = 1
    """
    vector_norm = a*a + b*b + c*c
    normal_vector = np.array([a, b, c]) / np.sqrt(vector_norm)
    point_in_plane = np.array([a, b, c]) / vector_norm

    points = np.column_stack((x, y, z))
    points_from_point_in_plane = points - point_in_plane
    proj_onto_normal_vector = np.dot(points_from_point_in_plane,
                                     normal_vector)
    proj_onto_plane = (points_from_point_in_plane -
                       proj_onto_normal_vector[:, None]*normal_vector)

    return point_in_plane + proj_onto_plane

vector_norm = sum([v*v for v in lsvm.coef_])

#%%
line = np.linspace(-10, 10)
for coef, intercept in zip(lsvm.coef_, lsvm.intercept_):
    pl.plot(line, -(line * coef[0] + intercept) / coef[1])

X0, X1 = X[:, 0], X[:, 1]

pl.scatter(X0, X1, c=y)


#%% SUBSET?

trials_0 = list(set(sDATA[sDATA['config']==0]['trial']))
trials_90 = list(set(sDATA[sDATA['config']==90]['trial']))

trial_idxs0 = [ti for ti, tname in enumerate(trial_labels) if tname in trials_0]
trial_idxs90 = [ti for ti, tname in enumerate(trial_labels) if tname in trials_90]

trial_idxs = trial_idxs0 + trial_idxs90
subX = X[trial_idxs,:]
suby = y[trial_idxs]