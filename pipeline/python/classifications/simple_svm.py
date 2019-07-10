#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 12:13:30 2019

@author: julianarhee
"""

import os
import glob
import random

import numpy as np
import pandas as pd
import pylab as pl
import seaborn as sns

from pipeline.python.classifications import test_responsivity as resp
from pipeline.python.utils import label_figure, natural_keys
from pipeline.python.classifications import utils as utils

from scipy import stats
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC
from sklearn import preprocessing
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    fig = pl.figure()
    pl.title(title)
    if ylim is not None:
        pl.ylim(*ylim)
    pl.xlabel("Training examples")
    pl.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    pl.grid()

    pl.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    pl.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    pl.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    pl.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    pl.legend(loc="best")
    return fig


#%%

def get_filtered_roi_dfs(traces, labels, response_type='meanstim', response_thr=0.2,
                         goodness_type='snr', goodness_thr=1.5):
        
    bdf = resp.group_roidata_stimresponse(traces, labels)
    nrois_total = len(bdf.groups)
    print("... Total of %i rois selected for this FOV" % nrois_total)
    
    # Get responsive cells:
    #response_type = 'meanstim'
    #response_thr = 0.2
    #goodness_type = 'snr'
    #goodness_thr = 1.5
    roi_list = [k for k, g in bdf if round(g.groupby(['config']).mean()[response_type].max(), 1) >= response_thr\
                and g.groupby(['config']).mean()[goodness_type].max() >= goodness_thr] #\
    print("... %i out of %i cells meet min %s req. of %.2f" % (len(roi_list), nrois_total, response_type, response_thr))
    
    # remake DF with only included rois:
    bdf = resp.group_roidata_stimresponse(traces[roi_list], labels)
    
    return bdf, roi_list

#%%
# V1 --------------------------------------------------------
# JC076: 20190420, *20190501 

# JC083: 20190507, 20190510, 20190511
# JC084: 20190522
# JC085:  20190622
# JC097: 20190613, 20190616, 20190617


# Li --------------------------------------------------------
# JC076: 20190502
# JC090: 20190605
# JC091:  '20190602', '20190606', '20190607', '20190614'
# JC099: 20190609, 20190612, 20190617
# ------------------------------------------------------------

#animalid = 'JC083'
#session_list = ['20190507', '20190510', '20190511'] # 
animalid = 'JC091'
session_list = ['20190602', '20190606', '20190607', '20190614'] # 

rootdir = '/n/coxfs01/2p-data'
visual_area = 'Li'

fov = 'FOV1_zoom2p0x'
run = 'combined_blobs_static'
traceid = 'traces001' #
segment_visual_area = False
trace_type = 'dff'
metric_type = 'zscore'


# Filter params:
response_type='meanstim' # if using trace_type=='dff'
response_thr=0.2
goodness_type='snr'
goodness_thr=1.5


#%%
fig, axes = pl.subplots(1, len(session_list), figsize=(6*len(session_list), 5))

for session, ax in zip(session_list, axes.flat):
        
    # Load data:
    blobs = utils.Experiment('blobs', animalid, session, fov, traceid, trace_type='dff')
    blobs.data.traces, blobs.data.labels = utils.check_counts_per_condition(blobs.data.traces, blobs.data.labels)
    
    # Exclude non-morph controls for now:
    blobs.data.sdf = blobs.data.sdf[blobs.data.sdf['morphlevel']!=-1] # get rid of controls for now
    tested_sizes = sorted(blobs.data.sdf ['size'].unique())
    tested_morphs = sorted(blobs.data.sdf ['morphlevel'].unique())
    print('sizes:', tested_sizes)
    print('morphs:', tested_morphs)
    
    # Remove indices corresponding to excluded configs:
    tmplabels = blobs.data.labels
    tmptraces = blobs.data.traces
    ixs = np.array(tmplabels[tmplabels['config'].isin(blobs.data.sdf.index.tolist())].index.tolist())
    blobs.data.traces = tmptraces.loc[ixs].reset_index(drop=True)
    blobs.data.labels = tmplabels[tmplabels['config'].isin(blobs.data.sdf.index.tolist())].reset_index(drop=True)
    del tmplabels
    del tmptraces
    
    #% Create data frame groups by ROI:
    bdf, roi_list = get_filtered_roi_dfs(blobs.data.traces, blobs.data.labels,
                                         response_type=response_type, response_thr=response_thr,
                                         goodness_type=goodness_type, goodness_thr=goodness_thr)
    
    data_id = '|'.join([blobs.animalid, blobs.session, blobs.fov, blobs.traceid, blobs.trace_type])
    fig_label = '%s\n%s' % (data_id, '_'.join(['filter: %s %.2f, %s %.2f' % (response_type, response_thr, goodness_type, goodness_thr), 'response vals = %s' % metric_type]))
    
    
    #%
    sample_data = pd.concat([pd.DataFrame(roidf[metric_type].values, columns=[roi],
                                          index=roidf['config']) for roi, roidf in bdf], axis=1)
    sample_labels = np.array(sample_data.index.tolist())
    sample_data = sample_data.reset_index(drop=True)
    sdf = blobs.data.sdf
    
    
    #% Specify train/test conditions:     
    class_name = 'morphlevel'
    class_types = [0, 106]
    restrict_transform = True
    constant_transform = 'size'
    
    C = 1e3
    m0 = 0
    m100 = 106
    #fig, ax = pl.subplots()
    
    size_colors = sns.cubehelix_palette(len(tested_sizes))
    lw=2
    for curr_sz, curr_color in zip(tested_sizes, size_colors):
        if restrict_transform:
            constant_transform_val = curr_sz
            train_configs = sdf[((sdf[class_name].isin(class_types)) & (sdf[constant_transform]==constant_transform_val))].index.tolist()
        else:
            train_configs = sdf[sdf[class_name].isin(class_types)].index.tolist()
        
        # Set train/test set:
        train_ixs = [i for i, l in enumerate(sample_labels) if l in train_configs]
        
        X = sample_data.iloc[train_ixs].values #[train_configs]
        y = np.array([sdf[class_name][c] for c in sample_labels[train_ixs]])
        
        # Set validation set:
        untrained_class_types = [c for c in sdf[class_name].unique() if c not in class_types]
        test_configs = sdf[( (sdf[constant_transform]==constant_transform_val) & (sdf[class_name].isin(untrained_class_types)) )].index.tolist()
        test_ixs = [i for i, l in enumerate(sample_labels) if l in test_configs]
        X_test = sample_data.iloc[test_ixs].values
        y_test_labels = sample_labels[test_ixs]
        
        
        #% Train/test split
        n_splits = len(X)
        
        kf = KFold(n_splits=n_splits)
        kf.get_n_splits(X)
        
        scores=[]
        test_scores = dict((sdf[class_name][tc], []) for tc in test_configs)
        choices = dict((tc, []) for tc in tested_morphs)
        
        for train_index, validate_index in kf.split(X):
            # Get current train/test split data:
            X_train, X_validate = X[train_index], X[validate_index]
            y_train, y_validate = y[train_index], y[validate_index]
            scaler = preprocessing.StandardScaler().fit(X_train)
            X_train_transformed = scaler.transform(X_train)
            
            # Fit SVM:
            trained_svc = LinearSVC(multi_class='ovr', C=C).fit(X_train_transformed, y_train)
            
            # Test:
            X_validate_transformed = scaler.transform(X_validate)
            curr_score = trained_svc.score(X_validate_transformed, y_validate)  
            y_pred = trained_svc.predict(X_validate_transformed)
            
            # Test 2:
            for true_val, pred_val in zip(y_validate, y_pred):
                choices[true_val].append(pred_val)
                    
            for tc in test_configs:
                X_test_transformed = scaler.transform(X_test[np.where(y_test_labels==tc)[0], :])
                curr_score_test = trained_svc.score(X_test_transformed, [m100 for _ in range(X_test_transformed.shape[0])])  
                test_scores[sdf[class_name][tc]].append(curr_score_test)
                y_pred_test = trained_svc.predict(X_test_transformed)
                
                choices[sdf[class_name][tc]].append(y_pred_test)
        
            scores.append(curr_score)
            
        print("mean score: %.2f (+/- %.2f)" % (np.mean(scores), stats.sem(scores)))
        #for k, v in sorted(test_scores.items(), key=lambda x: x[0]):
        #    print k, np.nanmean(v)
        
        #fig, ax = pl.subplots()
        pchoose100={}
        pchoose100_sem={}
        for k, v in choices.items():
            if k in [m0, m100]:
                pchoose100[k] = np.sum([i==m100 for i in v]) / float(len(v))
                pchoose100_sem[k] = 0
            else:
                pchoose100[k] = np.mean([np.sum([i==m100 for i in vv]) / float(len(vv)) for vv in v])
                pchoose100_sem[k] = np.std([np.sum([i==m100 for i in vv]) / float(len(vv)) for vv in v])
        
        
        curr_label = "[sz %i] %.2f (+/- %.2f)" % (curr_sz, np.mean(scores), stats.sem(scores))
        ax.plot(sorted(pchoose100.keys()), [pchoose100[k] for k in sorted(pchoose100.keys())], '-', \
                markersize=.5, lw=lw, color=curr_color, label=curr_label)
        ax.errorbar(sorted(pchoose100.keys()), [pchoose100[k] for k in sorted(pchoose100.keys())],\
                    yerr=[pchoose100_sem[k] for k in sorted(pchoose100_sem.keys())], fmt='none',
                    ecolor=curr_color)
        
        ax.set_ylim([0, 1])
        ax.set_ylabel('% choose morph100')
        ax.set_xlabel('morph')
        ax.set_title('%i-fold cross val (+ test morphs)' % n_splits)
        sns.despine(trim=True, offset=4, ax=ax)
        
    #pl.subplots_adjust(top=0.8)
    ax.legend(fontsize=6)

    #ax.text(0, 0.85, "mean CV score: %.2f (+/- %.2f)" % (np.mean(scores), stats.sem(scores)), fontsize=6)



pl.subplots_adjust(top=0.8)
label_figure(fig, fig_label)



#pl.legend(fontsize=6)

#%% Learning curves

X_transformed = scaler.transform(X)

svc = LinearSVC(random_state=0, multi_class='ovr', C=1e3)

# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
n_splits = 5
test_size = 0.1
cv = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=0)
title = "Learning Curves (%i features, %i splits, test sz %.2f)" % (len(roi_list), n_splits, test_size)

fig = plot_learning_curve(svc, title, X_transformed, y, ylim=(0., 1.01), cv=cv, n_jobs=4)

label_figure(fig, fig_label)




#roi = 43 #207 #43
#meanr = df[roi].groupby(['config']).mean()
#sns.heatmap(np.reshape(meanr, (9, 5), order='C').T, annot=True)
#pl.colorbar()

###
#roi = 207 #149 #343
#curr_metric = 'zscore'
#meanr1 = gdf.get_group(roi).groupby(['config']).mean()[curr_metric]
#meanr2 = gdf_f.get_group(roi).groupby(['config']).mean()[curr_metric]
#fig, ax = pl.subplots(1,2)
#sns.heatmap(np.reshape(meanr1, (9, 5), order='C').T, annot=True, ax=ax[0])
#sns.heatmap(np.reshape(meanr2[5:], (9, 5), order='C').T, annot=True, ax=ax[1])
#
#pl.figure()
#curr_metric = 'meanstim'
#roi = 147
#meanr2f = gdf_f.get_group(roi).groupby(['config']).mean()[curr_metric]
#sns.heatmap(np.reshape(meanr2, (10, 5), order='C').T, annot=True) # ax=ax[1])
###
#l2 = pd.DataFrame(data=dset['labels_data'], columns=dset['labels_columns'])
#r2 = pd.DataFrame(dset['dff'])
#
#fig, ax= pl.subplots(1,2)
#mean_trace = []
#for ti, tix in l2[l2['config']=='config001'].groupby(['trial']):
#    print ti
#    mean_trace.append(r2[roi][np.array(tix.index.tolist())].values)
#    ax[0].plot(r2[roi][np.array(tix.index.tolist())].values, alpha=0.5, c='k')
#mean_trace1 = np.array(mean_trace)
#print(mean_trace1.shape)
#ax[0].plot(np.nanmean(mean_trace1, axis=0), lw=2)
#
#mean_trace = []
#for ti, tix in l2[l2['config']=='config006'].groupby(['trial']):
#    print ti
#    mean_trace.append(r2[roi][np.array(tix.index.tolist())].values)
#    ax[1].plot(r2[roi][np.array(tix.index.tolist())].values, alpha=0.5, c='k')
#mean_trace2 = np.array(mean_trace)
#print(mean_trace2.shape)
#ax[1].plot(np.nanmean(mean_trace2, axis=0), lw=2)
#
#


