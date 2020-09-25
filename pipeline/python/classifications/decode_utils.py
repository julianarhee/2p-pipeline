#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on  Apr 24 19:56:32 2020

@author: julianarhee
"""
import os
import json
import glob
import copy
import copy
import itertools
import datetime
import pprint 
pp = pprint.PrettyPrinter(indent=4)

import numpy as np
import pylab as pl
import seaborn as sns
import pandas as pd
import statsmodels as sm
import cPickle as pkl

from scipy import stats as spstats

from pipeline.python.classifications import experiment_classes as util
from pipeline.python.classifications import aggregate_data_stats as aggr
from pipeline.python.classifications import rf_utils as rfutils
from pipeline.python import utils as putils

from pipeline.python.retinotopy import fit_2d_rfs as fitrf
from pipeline.python.rois.utils import load_roi_coords


from matplotlib.lines import Line2D
import matplotlib.patches as patches

import scipy.stats as spstats
import sklearn.metrics as skmetrics
import random

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC
from sklearn import preprocessing
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn import svm


def load_aggregate_rfs(rf_dsets, traceid='traces001', 
                        fit_desc='fit-2dgaus_dff-no-cutoff', verbose=False):
    rf_dpaths, no_fits = rfutils.get_fit_dpaths(rf_dsets, traceid=traceid, fit_desc=fit_desc)
    rfdf = rfutils.aggregate_rf_data(rf_dpaths, reliable_only=reliable_only, 
                                        fit_desc=fit_desc, traceid=traceid, verbose=verbose)
    rfdf = rfdf.reset_index(drop=True)
    return rfdf

def get_rf_positions(rf_dsets, df_fpath):
    rfdf = load_aggregate_rfs(rf_dsets, traceid=traceid, fit_desc=fit_desc)
    get_positions = False
    if os.path.exists(df_fpath) and get_positions is False:
        print("Loading existing RF coord conversions...")
        try:
            with open(df_fpath, 'rb') as f:
                df= pkl.load(f)
            rfdf = df['df']
        except Exception as e:
            get_positions = True

    if get_positions:
        print("Calculating RF coord conversions...")
        pos_params = ['fov_xpos', 'fov_xpos_pix', 'fov_ypos', 'fov_ypos_pix', 'ml_pos','ap_pos']
        for p in pos_params:
            rfdf[p] = ''
        p_list=[]
        for (animalid, session, fovnum), g in rfdf.groupby(['animalid', 'session', 'fovnum']):
            fcoords = load_roi_coords(animalid, session, 'FOV%i_zoom2p0x' % fovnum, 
                                      traceid=traceid, create_new=False)

            for ei, e_df in g.groupby(['experiment']):
                cell_ids = e_df['cell'].unique()
                p_ = fcoords['roi_positions'].loc[cell_ids]
                for p in pos_params:
                    rfdf[p][e_df.index] = p_[p].values
        with open(df_fpath, 'wb') as f:
            pkl.dump(rfdf, f, protocol=pkl.HIGHEST_PROTOCOL)
    return rfdf

def pick_rfs_with_most_overlap(rfdf, MEANS):
    r_list=[]
    for datakey, expdf in MEANS.items(): #corrs.groupby(['datakey']):
        # Get active blob cells
        exp_rids = [r for r in expdf.columns if putils.isnumber(r)]     
        # Get current fov's RFs
        rdf = rfdf[rfdf['datakey']==datakey].copy()
        
        # If have both rfs/rfs10, pick the best one
        if len(rdf['experiment'].unique())>1:
            rf_rids = rdf[rdf['experiment']=='rfs']['cell'].unique()
            rf10_rids = rdf[rdf['experiment']=='rfs10']['cell'].unique()
            same_as_rfs = np.intersect1d(rf_rids, exp_rids)
            same_as_rfs10 = np.intersect1d(rf10_rids, exp_rids)
            rfname = 'rfs' if len(same_as_rfs) > len(same_as_rfs10) else 'rfs10'
            print("%s: Selecting %s, overlappig rfs, %i | rfs10, %i (of %i cells)" 
                  % (datakey, rfname, len(same_as_rfs), len(same_as_rfs10), len(exp_rids)))
            r_list.append(rdf[rdf['experiment']==rfname])
        else:
            r_list.append(rdf)
    RFs = pd.concat(r_list, axis=0)

    return RFs

def plot_all_rfs(RFs, cmap='cubehelix'):
    '''
    Plot ALL receptive field pos, mark CoM by FOV. Colormap = datakey.
    One subplot per visual area.
    '''
    visual_areas = ['V1', 'Lm', 'Li']
    fig, axn = pl.subplots(1,3, figsize=(10,6), dpi=dpi)
    for visual_area, v_df in RFs.groupby(['visual_area']):
        ai = visual_areas.index(visual_area)
        ax = axn[ai]
        dcolors = sns.color_palette(cmap, n_colors=len(v_df['datakey'].unique()))
        for di, (datakey, d_df) in enumerate(v_df.groupby(['datakey'])):
            
            exp_rids = [r for r in MEANS[datakey] if putils.isnumber(r)]     
            rf_rids = d_df['cell'].unique()
            common_to_rfs_and_blobs = np.intersect1d(rf_rids, exp_rids)
            curr_df = d_df[d_df['cell'].isin(common_to_rfs_and_blobs)].copy()
            
            sns.scatterplot('x0', 'y0', data=curr_df, ax=ax, color=dcolors[di],
                           s=10, marker='o', alpha=0.5) 

            x = curr_df['x0'].values
            y=curr_df['y0'].values
            
            ncells_rfs = len(rf_rids)
            ncells_common = len(common_to_rfs_and_blobs) #curr_df.shape[0]
            m=np.ones(curr_df['x0'].shape)
            cgx = np.sum(x*m)/np.sum(m)
            cgy = np.sum(y*m)/np.sum(m)
            #print('The center of mass: (%.2f, %.2f)' % (cgx, cgy))
            ax.plot(cgx, cgy, marker='+', markersize=20, color=dcolors[di], 
                    label='%s (%s, %i/%i)' 
                            % (visual_area, datakey, ncells_common, ncells_rfs), lw=3) 
        ax.set_title(visual_area)
        ax.legend(bbox_to_anchor=(0.95, -0.4), fontsize=8) #1))

    for ax in axn:
        ax.set_xlim([screenleft, screenright])
        ax.set_ylim([screenbottom, screentop])
        ax.set_aspect('equal')
        ax.set_ylabel('')
        ax.set_xlabel('')
        
    pl.suptitle("RF positions (+ CoM), responsive cells (%s)" % experiment)
    pl.subplots_adjust(top=0.9, bottom=0.4)

    return fig

def calculate_overlaps(RFs, datakeys, experiment='blobs'):
    rf_fit_params = ['cell', 'std_x', 'std_y', 'theta', 'x0', 'y0']

    o_list=[]
    for (visual_area, animalid, session, fovnum, datakey), g in RFs.groupby(['visual_area', 'animalid', 'session', 'fovnum', 'datakey']):  
        if datakey not in datakeys: #MEANS.keys():
            continue
        
        # Convert RF fit params to polygon
        g.index = g['cell'].values
        rf_polys = rfutils.rfs_to_polys(g[rf_fit_params])

        S = util.Session(animalid, session, 'FOV%i_zoom2p0x' % fovnum)
        stim_xpos, stim_ypos = S.get_stimulus_coordinates(experiments=[experiment])
        stim_sizes = S.get_stimulus_sizes(size_tested=[experiment])

        # Convert stimuli to polyon bounding boxes
        stim_polys = [(blob_sz, rfutils.stimsize_poly(blob_sz, xpos=stim_xpos, ypos=stim_ypos))                   for blob_sz in stim_sizes[experiment]]
        
        # Get all pairwise overlaps (% of smaller ellipse that overlaps larger ellipse)
        overlaps = pd.concat([rfutils.get_proportion_overlap(rf_poly, stim_poly) \
                    for stim_poly in stim_polys \
                    for rf_poly in rf_polys]).rename(columns={'row': 'cell', 'col': 'stim_size'})
        metadict={'visual_area': visual_area, 'animalid': animalid, 'rfname': rfname,
                  'session': session, 'fovnum': fovnum, 'datakey': datakey}
        o_ = putils.add_meta_to_df(overlaps, metadict)
        o_list.append(o_)

    stim_overlaps = pd.concat(o_list, axis=0).reset_index(drop=True)
    return stim_overlaps

# Decoding funcs
def computeMI(x, y):
    sum_mi = 0.0
    x_value_list = np.unique(x)
    y_value_list = np.unique(y)
    Px = np.array([ len(x[x==xval])/float(len(x)) for xval in x_value_list ]) #P(x)
    Py = np.array([ len(y[y==yval])/float(len(y)) for yval in y_value_list ]) #P(y)
    for i in xrange(len(x_value_list)):
        if Px[i] ==0.:
            continue
        sy = y[x == x_value_list[i]]
        if len(sy)== 0:
            continue
        pxy = np.array([len(sy[sy==yval])/float(len(y))  for yval in y_value_list]) #p(x,y)
        t = pxy[Py>0.]/Py[Py>0.] /Px[i] # log(P(x,y)/( P(x)*P(y))
        sum_mi += sum(pxy[t>0]*np.log2( t[t>0]) ) # sum ( P(x,y)* log(P(x,y)/( P(x)*P(y)) )
    return sum_mi

def filter_rois(rfs_and_blobs, overlap_thr=0.50, return_counts=False):
    visual_areas=['V1', 'Lm', 'Li']
    
    nocells=[]; notrials=[];
    global_rois = dict((v, []) for v in visual_areas)
    roi_counters = dict((v, 0) for v in visual_areas)
    
    roidf = []
    datakeys = dict((v, []) for v in visual_areas)
    for (visual_area, datakey), g in rfs_and_blobs[rfs_and_blobs['perc_overlap']>=overlap_thr].groupby(['visual_area', 'datakey']):

        roi_counter = roi_counters[visual_area]
        datakeys[visual_area].append(datakey)

        roi_list = sorted(g['cell'].unique()) #[int(r) for r in ddf.columns if r != 'config']

        # Reindex roi ids for global
        roi_ids = [i+roi_counter for i, r in enumerate(roi_list)]
        nrs = len(roi_list)

        global_rois[visual_area].extend(roi_ids)
        
        roidf.append(pd.DataFrame({'roi': roi_ids,
                                   'dset_roi': roi_list,
                                   'visual_area': [visual_area for _ in np.arange(0, nrs)],
                                   'datakey': [datakey for _ in np.arange(0, nrs)]}))

        # Update global roi id counter
        roi_counters[visual_area] += len(roi_ids)

    roidf = pd.concat(roidf, axis=0) #.groupby(['visual_area']).count()
    for k, v in global_rois.items():
        print(k, len(v))
    
    if return_counts:
        return roidf, roi_counters
    else:
        return roidf


def get_trials_for_N_cells(curr_ncells, gdf, MEANS):
    '''
    Randomly select N cells from global roi list (gdf), get cell's responses to all trials.
    
    gdf = dataframe (subset of global_rois dataframe), contains 
    - all rois for a given visual area
    - corresponding within-datakey roi IDs
    '''

    # Get current global RIDs
    ncells_t = gdf.shape[0]                      
    roi_ids = np.array(gdf['roi'].values.copy()) 

    # Random sample w/ replacement
    rand_ixs = np.array([random.randint(0, ncells_t-1) for _ in np.arange(0, curr_ncells)])
    curr_roi_list = roi_ids[rand_ixs]
    curr_roidf = gdf[gdf['roi'].isin(curr_roi_list)].copy()

    # Make sure equal num trials per condition for all dsets
    min_ntrials_by_config = min([MEANS[k]['config'].value_counts().min() for k in curr_roidf['datakey'].unique()])

    # Get data samples for these cells
    d_list=[]; c_list=[];
    for datakey, dkey_rois in curr_roidf.groupby(['datakey']):
        # Get subset of trials per cond to match min N trials
        tmpd_list=[]
        for cfg, trialmat in MEANS[datakey].groupby(['config']):
            # Get indices of trials in current dataset
            trial_ixs = trialmat.index.tolist() 
            # Shuffle them to get random order
            np.random.shuffle(trial_ixs)                
            # Select min_ntrials randomly
            curr_cfg_trials = trialmat.loc[trial_ixs[0:min_ntrials_by_config]].copy() 
            # Add current trials of current config to list
            tmpd_list.append(curr_cfg_trials)        
        tmpd = pd.concat(tmpd_list, axis=0) 

        # For each RID sample belonging to current dataset, get RID order
        sampled_cells = pd.concat([dkey_rois[dkey_rois['roi']==globalid][['roi', 'dset_roi']]                                          for globalid in curr_roi_list])
        sampled_dset_rois = sampled_cells['dset_roi'].values
        sampled_global_rois = sampled_cells['roi'].values

        # Get trial responses (some columns are repeats)
        curr_roidata = tmpd[sampled_dset_rois].copy().reset_index(drop=True)
        curr_roidata.columns = sampled_global_rois # Rename ROI columns to global-rois
        config_list = tmpd['config'].reset_index(drop=True)  # Get configs on selected trials
        d_list.append(curr_roidata)
        c_list.append(config_list)
    curr_neuraldf = pd.concat(d_list, axis=1).reset_index(drop=True)

    cfg_df = pd.concat(c_list, axis=1)
    cfg_df = cfg_df.T.drop_duplicates().T
    assert cfg_df.shape[0]==curr_neuraldf.shape[0], "Bad trials"
    assert cfg_df.shape[1]==1, "Bad configs"
    df = pd.concat([curr_neuraldf, cfg_df], axis=1)

    return df


def tune_C(sample_data, target_labels, scoring_metric='accuracy', cv_nfolds=5, test_split=0.2, verbose=False):
    
    train_data, test_data, train_labels, test_labels = train_test_split(sample_data, target_labels,
                                                                        test_size=test_split)
    
    #### DATA - Fit classifier
    scaler = StandardScaler()
    scaler.fit(train_data)
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)

    # Set the parameters by cross-validation
    tuned_parameters = [{'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}]

    #scores = ['accuracy', 'precision_macro', 'recall_macro']
    scoring = ('accuracy') #, 'precision_macro', 'recall_macro')
    # scoring_metric = 'accuracy' 
    results ={}
    #for scorer in scoring:
    scorer = scoring_metric
    
    if verbose:
        print("# Tuning hyper-parameters for %s" % scorer)
    #print()
    clf = GridSearchCV(
        svm.SVC(kernel='linear'), tuned_parameters, scoring=scorer, cv=cv_nfolds #scoring='%s_macro' % score
    )
    clf.fit(train_data, train_labels)
    if verbose:
        print("Best parameters set found on development set:")
        print(clf.best_params_)
    if verbose:
        print("Grid scores on development set (scoring=%s):" % scoring_metric)
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))

    y_true, y_pred = test_labels, clf.predict(test_data)
    if verbose:
        print("Detailed classification report:")
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print(classification_report(y_true, y_pred))
    test_score = clf.score(test_data, test_labels)
    if verbose:
        print("Held out test score: %.2f" % test_score)
    results.update({'%s' % scorer: {'C': clf.best_params_['C'], 'test_score': test_score}})
    return results #clf.best_params_


def fit_svm(zdata, targets, test_split=0.2, cv_nfolds=5, verbose=False, cv=True, C_value=None):

    #### For each transformation, split trials into 80% and 20%
    train_data, test_data, train_labels, test_labels = train_test_split(zdata, targets['label'].values, 
                                                        test_size=test_split, stratify=targets['group'])
    #print("CV:", cv)
    #### Cross validate (tune C w/ train data)
    if cv:
        cv_results = tune_C(train_data, train_labels, scoring_metric='accuracy', cv_nfolds=cv_nfolds, 
                           test_split=test_split, verbose=verbose)
        C_value = cv_results['accuracy']['C']
    else:
        assert C_value is not None, "Provide value for hyperparam C..."

    #### Fit SVM
    scaler = StandardScaler().fit(train_data)
    train_data = scaler.transform(train_data)
    trained_svc = svm.SVC(kernel='linear', C=C_value, random_state=10)
    scores = cross_validate(trained_svc, train_data, train_labels, cv=5,
                            scoring=('precision_macro', 'recall_macro', 'accuracy'),
                            return_train_score=True)
    iterdict = dict((s, values.mean()) for s, values in scores.items())
    trained_svc = svm.SVC(kernel='linear', C=C_value, random_state=10).fit(train_data, train_labels)
        
    #### DATA - Test with held-out data
    test_data = scaler.transform(test_data)
    test_score = trained_svc.score(test_data, test_labels)

    #### DATA - Calculate MI
    predicted_labels = trained_svc.predict(test_data)
    mi = skmetrics.mutual_info_score(test_labels, predicted_labels)
    ami = skmetrics.adjusted_mutual_info_score(test_labels, predicted_labels)
    log2_mi = computeMI(test_labels, predicted_labels)
    iterdict.update({'heldout_test_score': test_score, 
                     'heldout_MI': mi, 'heldout_aMI': ami, 'heldout_log2MI': log2_mi,
                     'C': C_value})
    # ------------------------------------------------------------------
    # Shuffle LABELS to calculate chance level
    train_labels_chance = train_labels.copy()
    np.random.shuffle(train_labels_chance)
    test_labels_chance = test_labels.copy()
    np.random.shuffle(test_labels_chance)

    #### CHANCE - Fit classifier
    chance_svc = svm.SVC(kernel='linear', C=C_value, random_state=10)
    scores_chance = cross_validate(chance_svc, train_data, train_labels_chance, cv=5,
                            scoring=('precision_macro', 'recall_macro', 'accuracy'),
                            return_train_score=True)
    iterdict_chance = dict((s, values.mean()) for s, values in scores_chance.items())

    # CHANCE - Test with held-out data
    trained_svc_chance = chance_svc.fit(train_data, train_labels_chance)
    test_score_chance = trained_svc_chance.score(test_data, test_labels_chance)  

    # Chance - Calculate MI
    predicted_labels = trained_svc_chance.predict(test_data)
    mi = skmetrics.mutual_info_score(test_labels, predicted_labels)
    ami = skmetrics.adjusted_mutual_info_score(test_labels, predicted_labels)
    log2_mi = computeMI(test_labels, predicted_labels)

    iterdict_chance.update({'heldout_test_score': test_score_chance, 
                            'heldout_MI': mi, 'heldout_aMI': ami, 'heldout_log2MI': log2_mi})

    return iterdict, iterdict_chance


# In[93]:


def do_fit(iter_num, global_rois=None, MEANS=None, sdf=None, sample_ncells=None,
           C_value=None, test_size=0.2, cv_nfolds=5, class_a=0, class_b=106):
    #[gdf, MEANS, sdf, sample_ncells, cv] * n_times)
    '''
    Resample w/ replacement from pooled cells (across datasets). Assumes 'sdf' is same for all datasets.
    Do n_iterations, return mean/sem/std over iterations as dict of results.
    Classes (class_a, class_b) should be the actual labels of the target (i.e., value of morph level)
    '''
    #iter_list=[]
    #chance_list=[]
    #for iteration in np.arange(0, n_iterations): #n_iterations):
    
    # Get new sample set
    curr_data = get_trials_for_N_cells(sample_ncells, global_rois, MEANS)

    #### Select train/test configs for clf A vs B
    object_configs = sdf[sdf['morphlevel'].isin([class_a, class_b])].index.tolist() 
    curr_roi_list = [int(c) for c in curr_data.columns if c != 'config']
    sample_data = curr_data[curr_data['config'].isin(object_configs)]

    #### Equalize df/f across neurons:  Normalize each neuron to have same (zero) mean, (unit) SD across stimuli
    zdata = sample_data.drop('config', 1) #sample_data[curr_roi_list].copy()
    #zdata = (data - data.mean()) / data.std()

    #### Get labels
    targets = pd.DataFrame(sample_data['config'].copy(), columns=['config'])
    targets['label'] = [sdf['morphlevel'][cfg] for cfg in targets['config'].values]
    targets['group'] = [sdf['size'][cfg] for cfg in targets['config'].values]

    #### Fit
    curr_iter, _ = fit_svm(zdata, targets, cv=cv, C_value=C_value,
                                          test_split=test_split, cv_nfolds=cv_nfolds)

    return pd.DataFrame(curr_iter, index=[iter_num])


def plot_by_ncells(pooled, metric='heldout_test_score', area_colors=None,
        lw=2, ls='-', capsize=3, ax=None, dpi=150):

    if area_colors is None:
        visual_area, area_colors = putils.set_threecolor_palette()
        dpi = putils.set_plot_params()
       
    if ax is None:
        fig, ax = pl.subplots(figsize=(5,4), sharex=True, sharey=True, dpi=dpi)

    for ai, (visual_area, g) in enumerate(pooled.groupby(['visual_area'])):
        mean_scores = g.sort_values(by='n_units')[metric]
        std_scores = g.sort_values(by='n_units')['%s_sem' % metric]
        n_units_per = g.groupby(['n_units'])[metric].mean().index.tolist()
        ax.plot(n_units_per, mean_scores, color=area_colors[visual_area], 
                alpha=1, lw=lw,
                label='%s' % (visual_area))
        ax.errorbar(n_units_per, mean_scores, yerr=std_scores, color=area_colors[visual_area], 
                    capthick=lw, capsize=capsize, label=None, alpha=1, lw=lw, linestyle=ls)
    ax.legend(bbox_to_anchor=(1., 1))
    ax.set_xlabel("N units")
    ax.set_ylabel(metric)

    return ax



