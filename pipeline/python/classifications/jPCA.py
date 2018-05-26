#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 12:09:51 2018

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

from pipeline.python.utils import natural_keys, replace_root, print_elapsed_time
import pipeline.python.traces.combine_runs as cb
import pipeline.python.paradigm.align_acquisition_events as acq
import pipeline.python.visualization.plot_psths_from_dataframe as vis
from pipeline.python.traces.utils import load_TID
from pipeline.python.classifications import utils as util


from mpl_toolkits.axes_grid1 import make_axes_locatable

from sklearn.manifold import MDS

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

from sklearn.preprocessing import StandardScaler


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
def get_frame_labels(sDATA, traces='df', sort_by_config=False):
    # TODO:  Add option to groupby stim-config, or just return order of occurrence

    # =============================================================================
        # FORMAT DATA:
        # -----------------------------------------------------------------------------

    #    trace = 'df'

    #    trial_labels = [t[2] for t in sDATA.groupby(['roi', 'config', 'trial'])[trace].apply(np.array).keys().tolist()]
    #    config_labels = [t[1] for t in sDATA.groupby(['roi', 'config', 'trial'])[trace].apply(np.array).keys().tolist()]
    #
    #    # Get labels:
    #    ntrials_total = len(list(set(trial_labels)))
    #    #trial_labels = np.reshape(trial_labels, (nrois, ntrials_total))[0,:]    # DON'T SORT trials, since these are ordered by stimulus angle
    #    config_labels = np.reshape(config_labels, (nrois, ntrials_total))[0,:]  # These are already sorted, by increasing angle
    #
    #    roi_labels = [t[0] for t in sDATA.groupby(['roi', 'config', 'trial'])[trace].apply(np.array).keys().tolist()]
    #
    #    # Get list of features and class labels:
    #    roi_list = sorted(list(set(roi_labels)), key=natural_keys)  # Sort by name, since dataframe grouped by ROI first

        #%
        # Rename ROIs if combing multiple FOVs:
    #    if options_idx > 0:
    #        curr_roi_start = len(averages_df)
    #        curr_nrois = len(roi_list)
    #        roi_list_updated = ['roi%05d' % int(i) for i in np.arange(curr_roi_start, curr_roi_start+curr_nrois)]
    #        assert len(roi_list_updated) == len(roi_list), "Bad indexing for ROI updating..."
    #        nindices = len(sDATA['roi']==roi_list[0])
    #        sDATA.loc[:, 'roi'] = [np.tile(roi, (nindices,)) for roi in roi_list_updated]
    #

    # Trial labels (unsorted) follow the grouping of STIM CONFIG:
    if sort_by_config:
        trial_labels = [t[2] for t in sDATA.groupby(['roi', 'config', 'trial'])[trace].apply(np.array).keys().tolist()]
        config_labels = [t[1] for t in sDATA.groupby(['roi', 'config', 'trial'])[trace].apply(np.array).keys().tolist()]
    else:
        trial_labels = [t[1] for t in sDATA.groupby(['roi', 'trial', 'config'])[trace].apply(np.array).keys().tolist()]
        config_labels = [t[2] for t in sDATA.groupby(['roi', 'trial', 'config'])[trace].apply(np.array).keys().tolist()]

        # Get labels:
    labels['trials'] = np.reshape(trial_labels, (nrois, ntrials_total))[0,:]    # DON'T SORT trials, since these are ordered by stimulus angle
    labels['config'] = np.reshape(config_labels, (nrois, ntrials_total))[0,:]  # These are already sorted, by increasing angle

    labels['roi'] = [t[0] for t in sDATA.groupby(['roi', 'config', 'trial'])[trace].apply(np.array).keys().tolist()]

    return labels



#%%

#opts1 = ['-D', '/mnt/odyssey', '-i', 'CE077', '-S', '20180425', '-A', 'FOV2_zoom1x',
#           '-T', 'np_subtracted', '--no-pupil',
#           '-R', 'gratings_run1', '-t', 'traces001',
#           '-n', '1']
#
#
#opts2 = ['-D', '/mnt/odyssey', '-i', 'CE077', '-S', '20180425', '-A', 'FOV1_zoom1x',
#           '-T', 'np_subtracted', '--no-pupil',
#           '-R', 'gratings_run2', '-t', 'traces001',
#           '-n', '1']
#
#options_list = [opts1, opts2]


#opts1 = ['-D', '/mnt/odyssey', '-i', 'CE084', '-S', '20180511', '-A', 'FOV1_zoom1x',
#           '-T', 'np_subtracted', '--no-pupil',
#           '-R', 'gratings_run1', '-t', 'traces001',
#           '-n', '1']
#
#opts2 = ['-D', '/mnt/odyssey', '-i', 'CE084', '-S', '20180511', '-A', 'FOV2_zoom1x',
#           '-T', 'np_subtracted', '--no-pupil',
#           '-R', 'gratings_run1', '-t', 'traces001',
#           '-n', '1']
#
#opts3 = ['-D', '/mnt/odyssey', '-i', 'CE084', '-S', '20180511', '-A', 'FOV3_zoom1x',
#           '-T', 'np_subtracted', '--no-pupil',
#           '-R', 'gratings_run1', '-t', 'traces001',
#           '-n', '1']

#options_list = [opts1, opts2, opts3]
opts1 = ['-D', '/mnt/odyssey', '-i', 'CE077', '-S', '20180518', '-A', 'FOV1_zoom1x',
           '-T', 'np_subtracted', '--no-pupil',
           '-R', 'blobs_run6', '-t', 'traces002',
           '-n', '1']
    
options_list = [opts1]

test = True

#%%
averages_df = []
normed_df= []
zscores = {}
info = {}
sdata = {}
#%%

options_idx = 0


for options_idx in range(len(options_list)):
    #%%
    print "**********************************"
    print "Processing %i of %i runs." % (options_idx, len(options_list))
    print "**********************************"

    #%
    options = options_list[options_idx]

    sDATA, run_info, stimconfigs = util.get_run_details(options)
    
    sconfigs = util.format_stimconfigs(stimconfigs)
    
    sconfigs = util.format_stimconfigs(stimconfigs)

    # Set up output dir:
    output_basedir = os.path.join(run_info['traceid_dir'], 'figures', 'population')
    if not os.path.exists(output_basedir):
        os.makedirs(output_basedir)

    #% Load time-course data and format:
    roi_list = run_info['roi_list']
    Xdata, ylabels, groups, tsecs, roi_list = util.format_framesXrois(sDATA, roi_list, missing='none')

    nframes_per_trial = run_info['nframes_per_trial']
    ntrials_per_cond = run_info['ntrials_per_cond']

    #% Test smoothing on a few ROIs:
    if test:
        ridx = 15 #162 #3 #162
        [c for c in sconfigs if sconfigs[c]['morphlevel']==6 and sconfigs[c]['yrot']==60]
        condlabel = 'config014' #225 #180
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
    smooth = True
    if smooth:
        X = np.apply_along_axis(util.smooth_traces, 0, Xdata, frac=0.001, missing='drop')
        y = ylabels.copy()
        print X.shape


    #%  Preprocessing, step 1:  Remove cross-condition mean

    # Select color code for conditions:
    conditions = run_info['condition_list']
    color_codes = sns.color_palette("Greys_r", len(conditions)*2)
    color_codes = color_codes[0::2]

    #% Look at example ROI:
    if test:
        ridx = 15 #44 #3 # 'roi00004' #10
        mean_cond_traces, mean_tsecs = util.get_mean_cond_traces(ridx, X, y, tsecs, ntrials_per_cond, nframes_per_trial)
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

    averages_list, normed_list = util.get_xcond_dfs(roi_list, X, y, tsecs, run_info)

    averages_df.extend(averages_list)
    normed_df.extend(normed_list)

    # Get zscores
    zscores[options_idx] = util.format_roisXzscore(sDATA, run_info)
    info[options_idx] = run_info
    sdata[options_idx] = sDATA


#% Concat into datagrame
avgDF = pd.concat(averages_df, axis=1)
avgDF.head()

normDF = pd.concat(normed_df, axis=1)
normDF.head()


#%%

traceid_dir = run_info['traceid_dir']

# Create OUTPUT DIR:
# -----------------------------------------------------------------------------
if len(options_list) > 1:
    options = options_list[options_idx]
    options = extract_options(options_list[options_idx])
    rootdir = options.rootdir; animalid = options.animalid; session = options.session
    session_dir = os.path.join(rootdir, animalid, session)
    combo_basedir = os.path.join(session_dir, 'combined_acquisitions')
    if not os.path.exists(combo_basedir):
        os.makedirs(combo_basedir)
else:
    options_idx = 0
    run_info = info[options_idx]
    sDATA = sdata[options_idx]
    zscores_by_roi = zscores[options_idx]

    single_basedir = os.path.join(traceid_dir, 'figures', 'population')
    if not os.path.exists(single_basedir):
        os.makedirs(single_basedir)

#%%
if len(options_list) > 1:
    output_basedir = combo_basedir
else:
    output_basedir = single_basedir

print  "Current output dir:\n", output_basedir

nframes_per_trial = info[0]['nframes_per_trial']
conditions = sorted(info[0]['condition_list'], key=natural_keys)

#%% #%% Format for matlab jPCA:

if jPCA:

    jpca_dir = os.path.join(output_basedir, 'jPCA')
    if not os.path.exists(jpca_dir):
        os.makedirs(jpca_dir)

#    # Since "mean_subtracted" traces are in order of stimulus, can just tile a vec for conditions:
#    norm_labels = np.hstack([np.tile(c, (nframes_per_trial,)) for c in orientations])
#
#    D = {}
#    for cond in orientations:
#        ixs = np.where(norm_labels==cond)[0]
#        curr_cond_data = normDF.values[ixs, :]
#        D['angle_%s' % str(cond)] = curr_cond_data
#    D['time_ms'] = mean_tsecs * 1000
#
#    mfile_path = os.path.join(jpca_dir, 'jpca_xcond_mean_subtracted.mat')
#    scipy.io.savemat(mfile_path, D)

    #%
    jX = avgDF.values
    
    # Get defualt label list for cX:
    if isinstance(run_info['condition_list'], str):
        cond_labels_all = sorted(run_info['condition_list'], key=natural_keys)
    else:
        cond_labels_all = sorted(run_info['condition_list'])
        
    jy = np.hstack([np.tile(cond, (info[0]['nframes_per_trial'],)) for cond in cond_labels_all]) #config_labels.copy()
    # Sort config labels as morphlevel:
    jy_ids = np.array([sconfigs[c]['morphlevel'] for c in jy])
    sorted_idxs = np.argsort(jy_ids)
    jy = jy_ids[sorted_idxs]
    jX = jX[sorted_idxs,:]
    
    #norm_labels = np.hstack([np.tile(c, (nframes_per_trial,)) for c in conditions])
    
    print conditions
    
    D = {}
    nan_ixs, nan_vals = np.where(np.isnan(jX))
    for cond in conditions:
        ixs = np.where(jy==cond)[0]
        ivals = np.array([i for i in range(jX.shape[1]) if i not in nan_vals])
        curr_cond_data = jX[ixs, :]
        curr_cond_data = curr_cond_data[:, ivals]
        print curr_cond_data.shape
        D['angle_%s' % str(cond)] = curr_cond_data
    D['time_ms'] = mean_tsecs * 1000

    mfile_path = os.path.join(jpca_dir, 'jpca_avg_trials.mat')
    scipy.io.savemat(mfile_path, D)

    #%

    # jPCA with EACH TRIAL, rather than average trial?
#    D = {}
#    for cond in orientations:
#        ixs = np.where(y==cond)[0]
#        curr_cond_data = X[ixs, :]
#        D['angle_%s' % str(cond)] = curr_cond_data
#    D['time_ms'] = tsecs[ixs,0] * 1000
#
#    mfile_path = os.path.join(jpca_dir, 'jpca_Xdata.mat')
#    scipy.io.savemat(mfile_path, D)


#%%

# =============================================================================
# Load ROI masks to sort ROIs by spatial distance
# =============================================================================

options_idx = 0
single_basedir = os.path.join(info[options_idx]['traceid_dir'], 'figures', 'population')
run_dir = info[options_idx]['traceid_dir'].split('/traces')[0]
acquisition = os.path.split(os.path.split(run_dir)[0])[1]

output_basedir = single_basedir
traceid_dir = info[options_idx]['traceid_dir']
sorted_rids, cnts, zproj = util.sort_rois_2D(traceid_dir)
util.plot_roi_contours(zproj, sorted_rids, cnts)
figname = 'spatially_sorted_rois_%s.png' % acquisition
pl.savefig(os.path.join(output_basedir, figname))


#%% Classify some shit:

from sklearn.preprocessing import StandardScaler

#output_basedir = combo_basedir

if len(options_list) > 1:
    cX1 = zscores[0].copy().T
    cX2 = zscores[1].copy().T
    cX3 = zscores[2].copy().T
    cX = np.hstack((cX1, cX2, cX3))
    print cX.shape
else:
    cX = zscores[0].copy().T


cX_std = StandardScaler(with_std=True).fit_transform(cX)
cy = np.hstack([np.tile(cond, (info[0]['ntrials_per_cond'],)) for cond in conditions]) #config_labels.copy()
print 'cX (zscores):', cX.shape
print 'cY (labels):', cy.shape

pl.figure();
sns.distplot(cX.ravel(), label='zscores')
sns.distplot(cX_std.ravel(), label = 'standardized')
pl.legend()
pl.title('distN of combined ROIs for FOV1, FOV2')
pl.savefig(os.path.join(output_basedir, 'roi_zscore_distributions.png'))

if len(options_list) > 1:
    pl.figure();
    sns.distplot(cX1.ravel(), label = 'FOV1')
    sns.distplot(cX2.ravel(), label = 'FOV2')
    sns.distplot(cX3.ravel(), label = 'FOV3')
    sns.distplot(cX_std.ravel(), label = 'standardized')
    pl.legend()
    pl.title('zscore distN for combined FOVs')
    pl.savefig(os.path.join(output_basedir, 'roi_zscore_distributions_ALL_FOVs.png'))



#%% Project zscore data onto normals of hyperplanes:

svc = LinearSVC(random_state=0, dual=False, multi_class='ovr')
svc.fit(cX_std, cy)

n_samples = cX_std.shape[0]


#  1.  Look at projection of all zscores (the data we fit our classifier with)
#  onto the normal vectors of each separating hyperplane:

cdata = np.array([svc.coef_[c].dot(cX_std.T) + svc.intercept_[c] for c in range(len(svc.classes_))])
pl.figure()
g = sns.heatmap(cdata, cmap='hot')

indices = { value : [ i for i, v in enumerate(cy) if v == value ] for value in svc.classes_ }
xtick_indices = [(k, indices[k][-1]) for k in indices.keys()]
g.set(xticks = [x[1] for x in xtick_indices])
g.set(xticklabels = [x[0] for x in xtick_indices])
g.set_xticklabels(g.get_xticklabels(), rotation=45, fontsize=8)
g.set(xlabel='trial')
g.set(ylabel = 'class')

pl.title('zscored trials proj onto normals')

figname = 'LinearSVC_fitall_proj2normals_zscores.png'

svc_output_dir = os.path.join(output_basedir, 'linearSVC')
if not os.path.exists(svc_output_dir):
    os.makedirs(svc_output_dir)
print "Currently saving to:", svc_output_dir
pl.savefig(os.path.join(svc_output_dir, figname))


#%%  Project data onto the HYPERPLANES (not the norm)

#
#stim1 = np.where(cy == 90)[0]
#stim0 = np.where(cy != 90)[0]

from math import sqrt
from sklearn.decomposition import PCA

def dot_product(x, y):
    #return sum([x[i] * y[i] for i in range(len(x))])
    return y.dot(x.T)

def norm(x):
    return sqrt(dot_product(x, x))

def normalize(x):
    return [x[i] / norm(x) for i in range(len(x))]

def project_onto_plane(x, n):
    d = dot_product(x, n) / norm(n)
    p = [d * normalize(n)[i] for i in range(len(n))]
    #p1 = [d / norm(n) * normalize(n)[i] for i in range(len(n))]
    return [x[i] - p[i] for i in range(len(x))]
#
#
## Look at 0-hyperplane:
#norm_vec = svc.coef_[2]
#
#ppoints = np.empty((cX.shape))
#for t in range(cX.shape[0]):
#    ppoints[t,:] = project_onto_plane(cX[t,:], norm_vec)
#print ppoints.shape
#
#pl.figure();
#sns.heatmap(ppoints)

# ppoints is an nTrials x nNeurons matrix (normal is in n-D space, where n = num neurons)

#%%
nconds = len(svc.classes_)
nrows = 2
ncols = nconds/nrows
cmap = mpl.cm.magma

fig1, axes1 = pl.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(15,5)) #pl.figure();
axes1 = axes1.flat

fig2, axes2 = pl.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(15,5)) #pl.figure();
axes2 = axes2.flat

hyp_fpath = os.path.join(svc_output_dir, 'hyperplane_projections.pkl')
if os.path.exists(hyp_fpath):
    with open(hyp_fpath, 'rb') as f:
        hyp_projections = pkl.load(f)
else:
    create_new = True
    if create_new:
        hyp_projections = {}

for ci in range(len(svc.classes_)):

    norm_vec = svc.coef_[ci]

    if create_new:
        ppoints = np.empty((cX.shape))
        for t in range(cX.shape[0]):
            ppoints[t,:] = project_onto_plane(cX_std[t,:], norm_vec)
        hyp_projections[svc.classes_[ci]] = ppoints
    else:
        ppoints = hyp_projections[svc.classes_[ci]]

    pX = ppoints.copy()
    py = cy.copy()

    g = sns.heatmap(ppoints, ax=axes1[ci], cmap=cmap)
    indices = { value : [ i for i, v in enumerate(cy) if v == value ] for value in svc.classes_ }
    ytick_indices = [(k, indices[k][0]) for k in indices.keys()]
    g.set(yticks = [yt[1] for yt in ytick_indices])
    #if ci % ncols == 0:
    g.set(yticklabels = [yt[0] for yt in ytick_indices])
    g.set_yticklabels(g.get_yticklabels(), rotation=45, fontsize=8)
    g.set(ylabel='trial')
    g.set(xticks = [])
    g.set(xticklabels = [])
    if ci % nrows == 0 and ci==ncols:
        g.set(xlabel = 'roi')
    g.set(title='%s' % svc.classes_[ci])

    g2 = sns.heatmap(cX_std - ppoints, ax=axes2[ci], cmap=cmap)
    indices = { value : [ i for i, v in enumerate(cy) if v == value ] for value in svc.classes_ }
    ytick_indices = [(k, indices[k][0]) for k in indices.keys()]
    g2.set(yticks = [yt[1] for yt in ytick_indices])
    #if ci % ncols == 0:
    g2.set(yticklabels = [yt[0] for yt in ytick_indices])
    g2.set_yticklabels(g.get_yticklabels(), rotation=45, fontsize=8)
    g2.set(ylabel='trial')
    g2.set(xticks = [])
    g2.set(xticklabels = [])
    if ci % nrows == 0 and ci==ncols:
        g2.set(xlabel = 'roi')
    g2.set(title='%s' % svc.classes_[ci])


fig1.suptitle('zscores projected onto hyperplanes')
fig2.suptitle('zscore minus projections')

fig1.savefig(os.path.join(svc_output_dir, 'project_zscores2hyperplanes.png'))
fig2.savefig(os.path.join(svc_output_dir, 'zscores_minus_hyperplane_projections.png'))


#%% How


def corr2_coeff(A,B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:,None]
    B_mB = B - B.mean(1)[:,None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1);
    ssB = (B_mB**2).sum(1);

    # Finally get corr coeff
    return np.dot(A_mA,B_mB.T)/np.sqrt(np.dot(ssA[:,None],ssB[None]))

ci = 0
norm_vec = svc.coef_[ci]

proj = np.empty((cX.shape))
for t in range(cX.shape[0]):
    proj[t,:] = project_onto_plane(cX_std[t,:], norm_vec)



corrmat = corr2_coeff(cX_std.T, proj.T)
pl.figure();
g0 = sns.heatmap(corrmat)
pl.title('Corr b/w zscore data and h-projections (%i)' % svc.classes_[ci])
g0.set(xticks = [])
g0.set(xticklabels = [])
g0.set(xlabel = 'roi')
g0.set(ylabel = 'roi')

pl.savefig(os.path.join(svc_output_dir, 'corrcoef_zscores_hyperplane_projections_%degrees.png' % svc.classes_[ci]))



#%% Reduce this with PCA (2 components to view) -- input data should be (n_samples, n_features)
#   -- we want nROI-dimension space to reduce down somehow..



#%%


    ncomps = 6
    pdata = cX_std - ppoints
    notC = np.where(cy!=90)[0]
    pdata = pdata[notC, :]

    corrmat = corr2_coeff(cX_std.T, proj.T)

    pca = PCA(n_components=ncomps, whiten=True)
    pcomps = pca.fit_transform(pdata)
    # Store PCA results into dataframe and plot:
    # -----------------------------------------------------------------------------
    pca_df = pd.DataFrame(data=pcomps,
                          columns=['pc%i' % int(i+1) for i in range(ncomps)])
    labels_df = pd.DataFrame(data=py,
                             columns=['target'])
    pdf = pd.concat([pca_df, labels_df], axis=1).reset_index()

    paired = False
    if paired:
        curr_oris = [0, 180, 90, 270, 45, 225, 135, 315]
        curr_colors = ['r', 'r', 'g', 'g', 'y', 'y', 'b', 'b']
    else:
        curr_oris = orientations #[0, 90, 180, 270]
        curr_colors = ['r', 'orange', 'y', 'g', 'b', 'c', 'm', 'k']

    #curr_oris = curr_oris[0::2]
    #curr_colors = curr_colors[0::2]

    # Visualize 2D projection:
    fig = pl.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('PCA-reduced k-D projection onto %i-hyperplane' % svc.classes_[ci], fontsize = 20)
    #for target, color in zip(orientations,colors):
    for target, color in zip(curr_oris, curr_colors):

        indicesToKeep = pdf['target'] == target
        ax.scatter(pdf.loc[indicesToKeep, 'pc2'],
                   pdf.loc[indicesToKeep, 'pc3'],
                   c = color,
                   s = 50,
                   alpha=0.5)
    ax.legend(curr_oris)
    ax.grid()


#%%
# =============================================================================
# Plot smoothed, xcond-mean-subtracted trials by condition for EACH roi:
# =============================================================================

#color_codes = ['r', 'orange', 'y', 'g', 'b', 'c', 'm', 'k']
#alpha_codes = np.linspace(0.2, 1, num=len(orientations))

# Colormap settings:
#tag = np.arange(0, 8)
#cmap = pl.cm.Greys                                        # define the colormap
#cmaplist = [cmap(i) for i in range(cmap.N)]             # extract all colors from the .jet map
#cmaplist[0] = (.5,.5,.5,1.0)                            # force the first color entry to be grey
#cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)  # create the new map
#bounds = np.linspace(0,len(tag),len(tag)+1)             # define the bins and normalize
#norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

color_codes = sns.color_palette("Greys_r", len(configs)*2)
color_codes = color_codes[0::2]
stimbar_loc = -20

# First, let's look at mean across-trial firing rats for each condition by ROI:
if smooth:
    roi_psth_dir = os.path.join(output_basedir, 'roi_psths_smooth')
else:
    roi_psth_dir = os.path.join(output_basedir, 'roi_psths')

if not os.path.exists(roi_psth_dir):
    os.makedirs(roi_psth_dir)

plot_average = True
plot_legend = False
for roi in avgDF.columns:

    #fig = pl.figure()
    #traces = np.reshape(avgDF[roi].values, (nconds, nframes_per_trial))
    tracevec = avgDF[roi].values

    psth_from_full_trace(roi, tracevec, mean_tsecs, nconds, nframes_per_trial,
                                  color_codes=color_codes, orientations=orientations,
                                  stim_on_frame=stim_on_frame, nframes_on=nframes_on,
                                  plot_legend=True, as_percent=True,
                                  save_and_close=True, roi_psth_dir=roi_psth_dir)

                                  #roi_psth_dir='/tmp', save_and_close=True):


#%%


# Save legend to file:
# -----------------------------------------------------------------------------

SMALL_SIZE = 8
MEDIUM_SIZE = 12
BIGGER_SIZE = 16

pl.rc('font', size=SMALL_SIZE)          # controls default text sizes
pl.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
pl.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
pl.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
pl.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
pl.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
pl.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

fig = pl.figure()
rad = 1
thetas = (np.pi/180.0) * np.array(orientations)
for c, theta in enumerate(thetas):
    pl.polar([0, theta], [0, rad], lw=5, c=color_codes[c])
ax = pl.gca()
ax.set_theta_zero_location("N")
ax.set_yticklabels([])
ax.spines['polar'].set_visible(False)
pl.grid(False)

pl.savefig(os.path.join(output_basedir, 'angles_legend.png'))


#%%

# =============================================================================
# PCA xcond-mean subtracted "trials"
# =============================================================================

from sklearn.decomposition import PCA

#pX = cdata.T # (n_samples, n_features)
pX = normDF.values #.X_std.copy()

py = np.array([np.tile(ori, (nframes_per_trial,)) for ori in orientations]) #y.copy()
py = np.reshape(py, (nconds * nframes_per_trial))


ncomps = 6
pca = PCA(n_components=ncomps, whiten=True)
pcomps = pca.fit_transform(pX)

pca.fit(pX)
pca_coefs = pca.transform(pX)

#% Plot reconstructed traces:
Ahat = pca_coefs.dot(pca.components_)
print "reconstructed:", Ahat.shape

recon_psth_dir = os.path.join(output_basedir, 'roi_psths_recon_%ipcs' % ncomps)
if not os.path.exists(recon_psth_dir):
    os.makedirs(recon_psth_dir)

for ridx in range(Ahat.shape[1]):
    tracevec = Ahat[:, ridx]
    psth_from_full_trace('roi%05d' % int(ridx+1), tracevec, mean_tsecs, nconds, nframes_per_trial,
                                  color_codes=color_codes, orientations=orientations,
                                  stim_on_frame=stim_on_frame, nframes_on=nframes_on,
                                  plot_legend=False, as_percent=False,
                                  save_and_close=True, roi_psth_dir=recon_psth_dir)
#

# Store PCA results into dataframe and plot:
# -----------------------------------------------------------------------------
pca_df = pd.DataFrame(data=pcomps,
                      columns=['pc%i' % int(i+1) for i in range(ncomps)])
pca_df.shape

labels_df = pd.DataFrame(data=py,
                         columns=['target'])

pdf = pd.concat([pca_df, labels_df], axis=1).reset_index()

#curr_oris = [0, 90] #orientations #[0, 90, 180, 270]
#curr_oris = orientations #[0, 90, 180, 270]
#curr_colors = ['r', 'g']
#curr_colors = ['r', 'orange', 'y', 'g', 'b', 'c', 'm', 'k']

paired = True
if paired:
    curr_oris = [0, 180, 90, 270, 45, 225, 135, 315]
    curr_colors = ['r', 'r', 'g', 'g', 'y', 'y', 'b', 'b']
else:
    curr_oris = orientations #[0, 90, 180, 270]
    curr_colors = ['r', 'orange', 'y', 'g', 'b', 'c', 'm', 'k']

#curr_oris = curr_oris[0::2]
#curr_colors = curr_colors[0::2]

# Visualize 2D projection:
fig = pl.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('PCA-reduced k-D neural state for each tpoint', fontsize = 20)
#for target, color in zip(orientations,colors):
for target, color in zip(curr_oris, curr_colors):

    indicesToKeep = pdf['target'] == target
    ax.scatter(pdf.loc[indicesToKeep, 'pc1'],
               pdf.loc[indicesToKeep, 'pc2'],
               c = color,
               s = 50,
               alpha=0.5)
ax.legend(curr_oris)
ax.grid()

if len(curr_oris) == len(orientations):
    stim_str = 'all_angles'
else:
    stim_str = '_'.join([str(i) for i in curr_oris])
if paired:
    stim_str = '%s_paired' % stim_str

if smooth:
    figname = 'PCA_smoothed_%icomps_%s.png' % (ncomps, stim_str)
else:
    figname = 'PCA_%icomps_%s.png' % (ncomps, stim_str)
pl.savefig(os.path.join(output_basedir, figname))

#%%

# Plot "PSTH" for first k components:
# =============================================================================

pca_psth_dir = os.path.join(output_basedir, 'pca_psths_smooth')
print pca_psth_dir
if not os.path.exists(pca_psth_dir):
    os.makedirs(pca_psth_dir)

plot_legend = False
for p in range(pca_coefs.shape[1]):

    #fig = pl.figure()
    #traces = np.reshape(pca_coefs[:,p], (nconds, nframes_per_trial))
    tracevec = pca_coefs[:,p]

    psth_from_full_trace('pc%i' % p, tracevec, mean_tsecs, nconds, nframes_per_trial,
                                  color_codes=color_codes, orientations=orientations,
                                  stim_on_frame=stim_on_frame, nframes_on=nframes_on,
                                  plot_legend=False, as_percent=False,
                                  save_and_close=True, roi_psth_dir=roi_psth_dir)

#%%
# =============================================================================
# tSNE xcond-mean subtracted traces:
# =============================================================================

from sklearn.manifold import TSNE
import time


pX = normDF.values

#
tsne_df = pd.DataFrame(data=pX,
               index=np.arange(0, pX.shape[0]),
               columns=['r%i' % i for i in range(pX.shape[1])]) #(ori) for ori in orientations])


feat_cols = [f for f in tsne_df.columns]

# Visualize:
#target_ids = range(len(digits.target_names))
target_ids = range(len(orientations))

multi_run = False
nruns = 8

if multi_run is False:
    nruns = 1

perplexity = 1000 #5# 100
niter = 3000 #5000

colors = 'r', 'orange', 'y', 'g', 'c', 'b', 'm', 'k' #, 'purple'

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
        for i, c, label in zip(target_ids, colors, orientations):
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
    colors = 'r', 'orange', 'y', 'g', 'c', 'b', 'm', 'k' #, 'purple'
    #for i, c, label in zip(target_ids, colors, digits.target_names):
    #    plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], c=c, label=label)
    for i, c, label in zip(target_ids, colors, orientations):
        pl.scatter(tsne_results[py == int(label), 0], tsne_results[py == int(label), 1], c=c, label=label)
        box = ax.get_position()
        ax.set_position([box.x0 + box.width * 0.01, box.y0 + box.height * 0.02,
                         box.width * 0.98, box.height * 0.98])

# Put a legend below current axis
pl.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=len(orientations)/2)


pl.suptitle('t-SNE, x-cond sub (%i-D rois) | px: %i, ni: %i' % (nrois, perplexity, niter))
figname = 'tSNE_xcondmeansub_%irois_orderedT_pplex%i_niter%i_%iruns.png' % (nrois, perplexity, niter, nruns)


pl.savefig(os.path.join(output_basedir, figname))


#%%
# =============================================================================
# RDM xcond-mean-sub traces:
# =============================================================================
#    - Pearson's corr b/w CONDITIONS
#    -input data pX: [nconds * nframes_per_trial, n_neurons]

spatially_sort_rois = True

zdf_cmap = 'PRGn' #pl.cm.hot
pX = normDF.values

if spatially_sort_rois:
    pX = pX[:, sorted_rids]
    order = 'spatial'
else:
    order = 'unordered'

py = np.array([np.tile(ori, (nframes_per_trial,)) for ori in orientations]) #y.copy()
py = np.reshape(py, (nconds * nframes_per_trial))

df_list = []
for ori in orientations:
    tidxs = np.where(py==ori)
    currtrials = np.squeeze(pX[tidxs,:])
    cname = str(ori)
    currstim = np.reshape(currtrials, (currtrials.shape[0]*currtrials.shape[1],), order='F')
    df_list.append(pd.DataFrame({cname: currstim}))
df = pd.concat(df_list, axis=1)
df.head()

corrs = df.corr(method='pearson')
pl.figure()
sns.heatmap(1-corrs, cmap=zdf_cmap, vmax=2.0)
pl.title('RDM (xcond mean-subtracted)')
figname = 'RDM_conds_xcond_mean_subtracted_all_frames.png'
pl.savefig(os.path.join(output_basedir, figname))

#%%
# =============================================================================
# PCA xcond-mean subtracted RDM to visualize:
# =============================================================================

rdm = normDF.values #1-corrs
rdm_labels = py ##orientations

ncomps = 8
pca = PCA(n_components=ncomps, whiten=False)
pcomps = pca.fit_transform(rdm)

print pcomps.shape

#pca.fit(pX)
#pca_coefs = pca.transform(pX)

# Store PCA results into dataframe and plot:
# -----------------------------------------------------------------------------
pca_df = pd.DataFrame(data=pcomps,
                      columns=['pc%i' % int(i+1) for i in range(ncomps)])
pca_df.shape

labels_df = pd.DataFrame(data=rdm_labels,
                         columns=['target'])

pdf = pd.concat([pca_df, labels_df], axis=1).reset_index()

#curr_oris = [0, 90] #orientations #[0, 90, 180, 270]
#curr_oris = orientations #[0, 90, 180, 270]
#curr_colors = ['r', 'g']
#curr_colors = ['r', 'orange', 'y', 'g', 'b', 'c', 'm', 'k']

paired = False
if paired:
    curr_oris = [0, 180, 90, 270, 45, 225, 135, 315]
    curr_colors = ['r', 'r', 'g', 'g', 'y', 'y', 'b', 'b']
else:
    curr_oris = orientations #[0, 90, 180, 270]
    curr_colors = ['r', 'orange', 'y', 'g', 'b', 'c', 'm', 'k']

#curr_oris = curr_oris[0::2]
#curr_colors = curr_colors[0::2]

# Visualize 2D projection:
fig = pl.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)

#ax.scatter(pcomps[:, 0], pcomps[:,1])


ax.set_title('PCA-reduced k-D neural state for each tpoint', fontsize = 20)
#for target, color in zip(orientations,colors):
for target, color in zip(curr_oris, color_codes): #curr_colors):

    indicesToKeep = pdf['target'] == target
    ax.scatter(pdf.loc[indicesToKeep, 'pc1'],
               pdf.loc[indicesToKeep, 'pc2'],
               c = color,
               s = 50,
               alpha=0.5)
ax.legend(curr_oris)
ax.grid()

if len(curr_oris) == len(orientations):
    stim_str = 'all_angles'
else:
    stim_str = '_'.join([str(i) for i in curr_oris])
if paired:
    stim_str = '%s_paired' % stim_str

data_str = 'xcond_subtracted'
pl.savefig(os.path.join(output_basedir, 'PCA_%s_%s.pdf' % (data_str, stim_str)))


#%%
# =============================================================================
# Look at RDM of ROI popN at each TIME POINT in trial:
# =============================================================================
# Correlation between frame t for all trials of stim X, averaged over stimuli
# for each neuron pair.

spatially_sort_rois = True
plot_rdm = False  # Set to False if want to plot p-corr instead


bin_frames = True
nbins = 12
bin_size = None

pcorr_cmap = 'PRGn' #pl.cm.hot
rdm_cmap = 'bwr'

pX = normDF.values
if spatially_sort_rois:
    pX = pX[:, sorted_rids]
    order = 'spatial'
else:
    order = 'unordered'

nrois = pX.shape[1]

py = np.array([np.tile(ori, (nframes_per_trial,)) for ori in orientations]) #y.copy()
py = np.reshape(py, (nconds * nframes_per_trial))

# First, look at ALL rois together across all trials:
# -----------------------------------------------------------------------------
tdf = pd.DataFrame(data=pX, columns=np.arange(0, pX.shape[1]), index=xrange(0, pX.shape[0]))
tcorr = tdf.corr(method='pearson')
fig, ax = pl.subplots(1, 1, sharex=True, sharey=True, figsize=(6,6))
pl.title('1-corr (%s rois)' % order)

cbar_ax = fig.add_axes([.91, .4, .002, .2])
sns.heatmap(1-tcorr, cmap=rdm_cmap, vmin=0, vmax=2.0, ax=ax, cbar=1,
                 cbar_ax=cbar_ax)
# only label every 5th ROI:
ax.set_xticks([])
ax.set_yticks([])
ax.set_xticks(np.arange(0, nrois, 5))
ax.set_yticks(np.arange(0, nrois, 5))
ax.set_xticklabels(sorted_rids[::5], rotation=45)
ax.set_yticklabels(sorted_rids[::5], rotation=45)

figname = 'roi_rdm_full_%s.png' % order
pl.savefig(os.path.join(output_basedir, figname))



#%%
from sklearn.manifold import MDS

# Get RDMs for each TIME CHUNK (either binned or per-frame):
# If BINNING frames, use this:
if bin_frames:
    tchunks = np.linspace(0, nframes_per_trial, num=nbins)
    tchunks = [int(round(t)) for t in tchunks[0:-1]]
    bin_size = int(round(np.mean(np.diff(tchunks))))

    corr_dict = {}
    for ti in range(len(tchunks)):
        tcorr = np.zeros((nrois, nrois))
        tchunk = tchunks[ti]
        print tchunk
        #ax.set(aspect='equal')
        for c in orientations:
            ixs = np.where(py==c)
            t = np.squeeze(pX[ixs, :])
            tdf = pd.DataFrame(data=t, columns=np.arange(0, pX.shape[1]), index=xrange(0, nframes_per_trial))
            tcorr_tmp = tdf.loc[tchunk:tchunk+bin_size, :].corr(method='pearson')
            tcorr += tcorr_tmp
        tcorr /= len(orientations)

        corr_dict['chunk_%i' % tchunk] = tcorr
else:
    tchunks = np.linspace(0, nframes_per_trial-1, num=nframes_per_trial)
    tchunks = [int(round(t)) for t in tchunks]
    interval = int(round(np.mean(np.diff(tchunks))))

    corr_dict = {}
    for ti in range(len(tchunks)):
        tcorr = np.zeros((nrois, nrois))
        tchunk = tchunks[ti]
        print tchunk
        #ax.set(aspect='equal')
        for c in orientations:
            ixs = np.where(py==c)
            t = np.squeeze(pX[ixs, :])
            tdf = pd.DataFrame(data=t, columns=roi_list, index=xrange(0, nframes_per_trial))
            tcorr_tmp = tdf.loc[tchunk:tchunk+interval, :].corr(method='pearson')
            tcorr += tcorr_tmp
        tcorr /= len(orientations)

        corr_dict['frame_%i' % tchunk] = tcorr


# Get frame / timechunk indices and labels:
frame_keys = sorted(corr_dict.keys(), key=natural_keys)
if bin_frames:
    frames_to_plot = [int(f.split('_')[1]) for f in frame_keys]
    frame_names_included = frame_keys
    bin_str = '%ibins' % nbins
else:
    frame_onset = stim_on_frame
    pre_frames = np.arange(0, frame_onset)
    post_frames = np.arange(frame_onset+1, nframes_per_trial)

    frames_to_plot = []
    frames_to_plot.extend(pre_frames[1::10])
    frames_to_plot.extend([frame_onset])
    frames_to_plot.extend(post_frames[10::10])
    print len(frames_to_plot)
    frame_names_included = [frame_keys[i] for i in frames_to_plot]
    bin_str = 'perframe'


plot_type = 'pcorr'
plot_rdm = True
# Plot RDMs:
fig, axes = pl.subplots(1, len(frames_to_plot), sharex=True, sharey=True, figsize=(15,5))
cbar_ax = fig.add_axes([.91, .4, .002, .2])
for ti, ax in enumerate(axes):
    frame_idx = frames_to_plot[ti]
    tcorr = corr_dict[frame_names_included[ti]]
    ax.set(adjustable='box-forced', aspect='equal')

#    if plot_rdm:
#        sns.heatmap(1-tcorr, cmap=rdm_cmap, vmin=0, vmax=2.0, ax=ax, cbar=ti==0,
#                         cbar_ax=None if ti else cbar_ax)
#    else:
#        sns.heatmap(tcorr, cmap=pcorr_cmap, vmin=-1, vmax=1.0, ax=ax, cbar=ti==0,
#                         cbar_ax=None if ti else cbar_ax)


    rdm = 1-tcorr

    if plot_type == 'mds':
        mds = MDS(2, max_iter=100, n_init=1, dissimilarity='precomputed')
        M = mds.fit_transform(rdm)
        ax.scatter(M[:, 0], M[:, 1], s=2, alpha=0.5)
    elif plot_type == 'pca':
        pca = PCA(n_components=2)
        pcs = pca.fit_transform(rdm)
        ax.scatter(pcs[:, 0], pcs[:, 1], s=2, alpha=0.5)
    else:
        if plot_rdm:
            sns.heatmap(rdm, cmap=zdf_cmap, vmin=0, vmax=2.0, ax=ax, cbar=ti==0,
                         cbar_ax=None if ti else cbar_ax)
        else:
            sns.heatmap(tcorr, cmap=pcorr_cmap, vmin=-1, vmax=1.0, ax=ax, cbar=ti==0,
                         cbar_ax=None if ti else cbar_ax)

    if bin_frames:
        twindow = '%.2f to %.2f s' % (mean_tsecs[frame_idx], mean_tsecs[frame_idx + bin_size-1])
    else:
        twindow = '%.2f s' % mean_tsecs[frame_idx]


    ax.set_title(twindow, fontsize=8)
    ax.set_xticks(())
    ax.set_yticks(())

pl.suptitle('%s RDM across trial - %s' % (plot_type, order))

figname = 'ROI_RDM_xtrial_%s_%s_%s.png' % (order, bin_str, plot_type)
print figname
pl.savefig(os.path.join(output_basedir, figname))


#%%
#
#def find_s(m):
#    d=(m+m.T)
#    off_diag_indices=np.tril_indices(len(d),-1)
#    if 0 in d[off_diag_indices]:
#        return 'NA'
#    else:
#        numerator=(m-m.T)**2
#        denominator=m+m.T
#        return np.sum(numerator[off_diag_indices]/denominator[off_diag_indices])


# Plot sum of off-diagonals for each time point:
if not bin_frames:
    frame_names_included = sorted(corr_dict.keys(), key=natural_keys)
    frames_to_plot =  range(len(frame_names_included))

sum_rdm = False
pcorr_sums = []
for frame_name in frame_names_included:
    if sum_rdm is True:
        print "summing RDM values"
        curr_pcorr = 1-corr_dict[frame_name]
    else:
        # sum pcorrs:
        curr_pcorr = corr_dict[frame_name]

    off_diag_indices=np.tril_indices(len(curr_pcorr),-1)
    #offdiag_sum = abs(curr_pcorr[off_diag_indices]).sum()
    offdiag_sum = (curr_pcorr[off_diag_indices]**2).sum()
    #offdiag_sum = find_s(corr_dict[chunk])
    pcorr_sums.append(offdiag_sum)

fig, ax = pl.subplots(1, figsize=(20,5)) #()
ax.plot(mean_tsecs[frames_to_plot], pcorr_sums)
ax.set_xticklabels([])
ax.set_xticks(mean_tsecs[frames_to_plot][0::10])
ax.set_xticklabels(['%.2f' % i for i in mean_tsecs[frames_to_plot][0::10]], rotation=45)
pl.ylabel('sum ( abs( corr ))')
pl.title('popN correlation over trial - %s' % order)
sns.despine(offset=4, trim=True)

figname = 'ROI_RDM_xtrial_ssq__%s_%s.png' % (order, bin_str)
pl.savefig(os.path.join(output_basedir, figname))



#%%

tchunks = np.linspace(0, nframes_per_trial, num=12)
tchunks = [int(round(t)) for t in tchunks]
interval = round(np.mean(np.diff(tchunks)))

# Get correlation of t-bin frames for each trial:

plot_type = 'rdm'
zdf_cmap = 'bwr'


fig, axes = pl.subplots(1,len(tchunks)-1, sharex=True, sharey=True, figsize=(20,1.5))
cbar_ax = fig.add_axes([.91, .3, .002, .3])

#for c in orientations:
#    ixs = np.where(py==c)
#    t = np.squeeze(pX[ixs, :])
#    tdf = pd.DataFrame(data=t, columns=roi_list, index=xrange(0, nframes_per_trial))
for ti, ax in enumerate(axes.flat):

    tcorr = np.zeros((nrois, nrois))
    tchunk = tchunks[ti]
    print tchunk
    #ax.set(aspect='equal')
    for c in orientations:
        ixs = np.where(y==c)[0]                                                # Get all indices of frame-vec for condition C
        ixs_r = np.reshape(ixs, (ntrials_per_stim, nframes_per_trial))         # Reshape into ntrials x nframes for cur condition trials
        tpoints = [int(i) for i in np.arange(tchunk, tchunk+interval)]
        trial_chunk = ixs_r[:, tpoints]                                        # Get current time-bin for all trials
        trial_chunk_ixs = np.reshape(trial_chunk, (trial_chunk.shape[0]*trial_chunk.shape[1]))

        t = X[trial_chunk_ixs,:]

        tdf = pd.DataFrame(data=t, columns=roi_list, index=xrange(0, t.shape[0]))
        tcorr_tmp = tdf.corr(method='pearson')
        tcorr += tcorr_tmp
    tcorr /= len(orientations)


    rdm = 1-tcorr

    if plot_type == 'mds':
        mds = MDS(2, max_iter=100, n_init=1, dissimilarity='precomputed')
        M = mds.fit_transform(rdm)
        ax.scatter(M[:, 0], M[:, 1], s=2, alpha=0.5)
    elif plot_type == 'pca':
        pca = PCA(n_components=2)
        pcs = pca.fit_transform(rdm)
        ax.scatter(pcs[:, 0], pcs[:, 1], s=2, alpha=0.5)
    else:
        sns.heatmap(rdm, cmap=zdf_cmap, vmin=0, vmax=2.0, ax=ax, cbar=ti==0,
                         cbar_ax=None if ti else cbar_ax)


    twindow = '%.2f to %.2f s' % (mean_tsecs[tchunk], mean_tsecs[tchunk + interval-1])

    ax.set_title(twindow, fontsize=8)
    ax.set_xticks(())
    ax.set_yticks(())



rdm = 1-tcorr
mds = MDS(2, max_iter=100, n_init=1)
M = mds.fit_transform(rdm)
pl.figure();

pl.scatter(M[:, 0], M[:, 1])

pca = PCA(n_components=2)
pcs = pca.fit_transform(rdm)
pl.figure()
pl.scatter(pcs[:, 0], pcs[:,1])