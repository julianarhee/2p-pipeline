#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 15:10:00 2018

@author: juliana
"""


import h5py
import os
import json
import cv2
import time
import math
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

##%%
#def kw_dunn(groups, to_compare=None, alpha=0.05, method='bonf'):
#    """
#    Kruskal-Wallis 1-way ANOVA with Dunn's multiple comparison test
#    Arguments:
#    ---------------
#    groups: sequence
#        arrays corresponding to k mutually independent samples from
#        continuous populations
#    to_compare: sequence
#        tuples specifying the indices of pairs of groups to compare, e.g.
#        [(0, 1), (0, 2)] would compare group 0 with 1 & 2. by default, all
#        possible pairwise comparisons between groups are performed.
#    alpha: float
#        family-wise error rate used for correcting for multiple comparisons
#        (see statsmodels.stats.multitest.multipletests for details)
#    method: string
#        method used to adjust p-values to account for multiple corrections (see
#        statsmodels.stats.multitest.multipletests for options)
#    Returns:
#    ---------------
#    H: float
#        Kruskal-Wallis H-statistic
#    p_omnibus: float
#        p-value corresponding to the global null hypothesis that the medians of
#        the groups are all equal
#    Z_pairs: float array
#        Z-scores computed for the absolute difference in mean ranks for each
#        pairwise comparison
#    p_corrected: float array
#        corrected p-values for each pairwise comparison, corresponding to the
#        null hypothesis that the pair of groups has equal medians. note that
#        these are only meaningful if the global null hypothesis is rejected.
#    reject: bool array
#        True for pairs where the null hypothesis can be rejected for the given
#        alpha
#    Reference:
#    ---------------
#    Gibbons, J. D., & Chakraborti, S. (2011). Nonparametric Statistical
#    Inference (5th ed., pp. 353-357). Boca Raton, FL: Chapman & Hall.
#    """
#
#    # omnibus test (K-W ANOVA)
#    # -------------------------------------------------------------------------
#
#    groups = [np.array(gg) for gg in groups]
#
#    k = len(groups)
#
#    n = np.array([len(gg) for gg in groups])
#    if np.any(n < 5):
#        warnings.warn("Sample sizes < 5 are not recommended (K-W test assumes "
#                      "a chi square distribution)")
#
#    allgroups = np.concatenate(groups)
#    N = len(allgroups)
#    ranked = stats.rankdata(allgroups)
#
#    # correction factor for ties
#    T = stats.tiecorrect(ranked)
#    if T == 0:
#        raise ValueError('All numbers are identical in kruskal')
#
#    # sum of ranks for each group
#    j = np.insert(np.cumsum(n), 0, 0)
#    R = np.empty(k, dtype=np.float)
#    for ii in range(k):
#        R[ii] = ranked[j[ii]:j[ii + 1]].sum()
#
#    # the Kruskal-Wallis H-statistic
#    H = (12. / (N * (N + 1.))) * ((R ** 2.) / n).sum() - 3 * (N + 1)
#
#    # apply correction factor for ties
#    H /= T
#
#    df_omnibus = k - 1
#    p_omnibus = stats.chisqprob(H, df_omnibus)
#
#    # multiple comparisons
#    # -------------------------------------------------------------------------
#
#    # by default we compare every possible pair of groups
#    if to_compare is None:
#        to_compare = tuple(combinations(range(k), 2))
#
#    ncomp = len(to_compare)
#
#    Z_pairs = np.empty(ncomp, dtype=np.float)
#    p_uncorrected = np.empty(ncomp, dtype=np.float)
#    Rmean = R / n
#
#    for pp, (ii, jj) in enumerate(to_compare):
#
#        # standardized score
#        Zij = (np.abs(Rmean[ii] - Rmean[jj]) /
#               np.sqrt((1. / 12.) * N * (N + 1) * (1. / n[ii] + 1. / n[jj])))
#        Z_pairs[pp] = Zij
#
#    # corresponding p-values obtained from upper quantiles of the standard
#    # normal distribution
#    p_uncorrected = stats.norm.sf(Z_pairs) * 2.
#
#    # correction for multiple comparisons
#    reject, p_corrected, alphac_sidak, alphac_bonf = multipletests(
#        p_uncorrected, method=method
#    )
#
#    return H, p_omnibus, Z_pairs, p_corrected, reject



#%%
def load_roi_dataframe(roidata_filepath):

    fn_parts = os.path.split(roidata_filepath)[-1].split('_')
    roidata_hash = fn_parts[1]
    trace_type = os.path.splitext(fn_parts[-1])[0]

    df_list = []

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
        datakey =datakeys[0]
        DATA = df[datakey]

    return DATA, datakey

#%%

def pyvt_raw_epochXoriXsf(rdata, trans_types, save_fig=False, output_dir='/tmp', fname='boxplot(intensity~epoch_X_config).png'):

    # Make sure trans_types are sorted:
    trans_types = sorted(trans_types, key=natural_keys)

    assert len(list(set(rdata['nframes_on'])))==1, "More than 1 idx found for nframes on... %s" % str(list(set(rdata['nframes_on'])))
    assert len(list(set(rdata['first_on'])))==1, "More than 1 idx found for first frame on... %s" % str(list(set(rdata['first_on'])))

    nframes_on = int(round(list(set(rdata['nframes_on']))[0]))
    first_on =  int(round(list(set(rdata['first_on']))[0]))

    df_groups = np.copy(trans_types).tolist()
    df_groups.extend(['trial', 'raw'])
    groupby_list = np.copy(trans_types).tolist()
    groupby_list.extend(['trial'])

    currdf = rdata[df_groups] #.sort_values(trans_types)
    grp = currdf.groupby(groupby_list)
    config_trials = {} # dict((config, []) for config in list(set(currdf['config'])))
    for k,g in grp: #config_trials.keys():
        if k[0] not in config_trials.keys():
            config_trials[k[0]] = {}
        if k[1] not in config_trials[k[0]].keys():
            config_trials[k[0]][k[1]] = {}

        config_trials[k[0]][k[1]] = sorted(list(set(currdf.loc[(currdf['ori']==k[0])
                                                        & (currdf['sf']==k[1])]['trial'])), key=natural_keys)

    idx = 0
    df_list = []
    for k,g in grp:
        #print k
        base_mean= g['raw'][0:first_on].mean()
        base_std = g['raw'][0:first_on].std()
        stim_mean = g['raw'][first_on:first_on+nframes_on].mean()

        df_list.append(pd.DataFrame({'ori': k[0],
                                     'sf': k[1],
                                     'trial': 'trial%05d' % int(config_trials[k[0]][k[1]].index(k[2]) + 1),
                                     'epoch': 'baseline',
                                     'intensity': base_mean}, index=[idx]))
        df_list.append(pd.DataFrame({'ori': k[0],
                                     'sf': k[1],
                                     'trial': 'trial%05d' % int(config_trials[k[0]][k[1]].index(k[2]) + 1),
                                     'epoch': 'stimulus',
                                     'intensity': stim_mean}, index=[idx+1]))
        idx += 2
    df = pd.concat(df_list, axis=0)
    df = df.sort_values(['epoch', 'ori', 'sf'])
    df = df.reset_index(drop=True)
    #pdf.pivot_table(index=['trial'], columns=['config', 'epoch'], values='intensity')

    # Format pandas df into pyvttbl dataframe:
    df_factors = np.copy(trans_types).tolist()
    df_factors.extend(['trial', 'epoch', 'intensity'])

    Trial = namedtuple('Trial', df_factors)
    pdf = pt.DataFrame()
    for idx in xrange(df.shape[0]):
        pdf.insert(Trial(df.loc[idx, 'ori'],
                         df.loc[idx, 'sf'],
                         df.loc[idx, 'trial'],
                         df.loc[idx, 'epoch'],
                         df.loc[idx, 'intensity'])._asdict())

    if save_fig:
        factor_list = np.copy(trans_types).tolist()
        factor_list.extend(['epoch'])
        pdf.box_plot('intensity', factors=factor_list, fname=fname, output_dir=output_dir)

    return pdf
#%%
def pyvt_stimdf_oriXsf(rdata, trans_types, save_fig=False, output_dir='/tmp', fname='dff_boxplot(intensity~epoch_X_config).png'):

    # Make sure trans_types are sorted:
    trans_types = sorted(trans_types, key=natural_keys)

    assert len(list(set(rdata['nframes_on'])))==1, "More than 1 idx found for nframes on... %s" % str(list(set(rdata['nframes_on'])))
    assert len(list(set(rdata['first_on'])))==1, "More than 1 idx found for first frame on... %s" % str(list(set(rdata['first_on'])))

    nframes_on = int(round(list(set(rdata['nframes_on']))[0]))
    first_on =  int(round(list(set(rdata['first_on']))[0]))

    df_groups = np.copy(trans_types).tolist()
    df_groups.extend(['trial', 'df'])
    groupby_list = np.copy(trans_types).tolist()
    groupby_list.extend(['trial'])

    currdf = rdata[df_groups] #.sort_values(trans_types)
    grp = currdf.groupby(groupby_list)
    config_trials = {} # dict((config, []) for config in list(set(currdf['config'])))
    for k,g in grp: #config_trials.keys():
        if k[0] not in config_trials.keys():
            config_trials[k[0]] = {}
        if k[1] not in config_trials[k[0]].keys():
            config_trials[k[0]][k[1]] = {}

        config_trials[k[0]][k[1]] = sorted(list(set(currdf.loc[(currdf['ori']==k[0])
                                                        & (currdf['sf']==k[1])]['trial'])), key=natural_keys)

    idx = 0
    df_list = []
    for k,g in grp:
        #print k
        base_mean= g['df'][0:first_on].mean()
        base_std = g['df'][0:first_on].std()
        stim_mean = g['df'][first_on:first_on+nframes_on].mean()
        zscore_val = stim_mean / base_std

        df_list.append(pd.DataFrame({'ori': k[0],
                                     'sf': k[1],
                                     'trial': k[2], #'trial%05d' % int(config_trials[k[0]][k[1]].index(k[2]) + 1),
                                     'dff': stim_mean,
                                     'zscore': zscore_val}, index=[idx]))

        idx += 1
    df = pd.concat(df_list, axis=0)
    df = df.sort_values(trans_types)
    df = df.reset_index(drop=True)
    #pdf.pivot_table(index=['trial'], columns=['config', 'epoch'], values='intensity')

    # Format pandas df into pyvttbl dataframe:
    df_factors = np.copy(trans_types).tolist()
    df_factors.extend(['trial', 'dff'])

    Trial = namedtuple('Trial', df_factors)
    pdf = pt.DataFrame()
    for idx in xrange(df.shape[0]):
        pdf.insert(Trial(df.loc[idx, 'ori'],
                         df.loc[idx, 'sf'],
                         df.loc[idx, 'trial'],
                         df.loc[idx, 'dff'])._asdict())

    if save_fig:
        factor_list = np.copy(trans_types).tolist()
        pdf.box_plot('dff', factors=factor_list, fname=fname, output_dir=output_dir)

    return pdf

#%%
def pyvt_stimdf_configs(rdata, save_fig=False, output_dir='/tmp', fname='boxplot(intensity~epoch_X_config).png'):

    '''
    Take single ROI as a datatset, do split-plot rmANOVA:
        within-trial factor :  baseline vs. stimulus epoch
        between-trial factor :  stimulus condition
    '''

    assert len(list(set(rdata['nframes_on'])))==1, "More than 1 idx found for nframes on... %s" % str(list(set(rdata['nframes_on'])))
    assert len(list(set(rdata['first_on'])))==1, "More than 1 idx found for first frame on... %s" % str(list(set(rdata['first_on'])))

    nframes_on = int(round(list(set(rdata['nframes_on']))[0]))
    first_on =  int(round(list(set(rdata['first_on']))[0]))

    df_groups = ['config', 'trial', 'df']
    groupby_list = ['config', 'trial']

    currdf = rdata[df_groups] #.sort_values(trans_types)
    grp = currdf.groupby(groupby_list)
    config_trials = {} # dict((config, []) for config in list(set(currdf['config'])))
    for k,g in grp: #config_trials.keys():
        if k[0] not in config_trials.keys():
            config_trials[k[0]] = {}

        config_trials[k[0]] = sorted(list(set(currdf.loc[currdf['config']==k[0]]['trial'])), key=natural_keys)

    idx = 0
    df_list = []
    for k,g in grp:
        #print k
        base_mean= g['df'][0:first_on].mean()
        base_std = g['df'][0:first_on].std()
        stim_mean = g['df'][first_on:first_on+nframes_on].mean()

        df_list.append(pd.DataFrame({'config': k[0],
                                     'trial': k[1], #'trial%05d' % int(config_trials[k[0]].index(k[1]) + 1),
                                     'dff': stim_mean}, index=[idx]))
        idx += 1
    df = pd.concat(df_list, axis=0)
    df = df.sort_values(['config'])
    df = df.reset_index(drop=True)

    #pdf.pivot_table(index=['trial'], columns=['config', 'epoch'], values='intensity')

    # Format pandas df into pyvttbl dataframe:
    df_factors = ['config', 'trial', 'dff']

    Trial = namedtuple('Trial', df_factors)
    pdf = pt.DataFrame()
    for idx in xrange(df.shape[0]):
        pdf.insert(Trial(df.loc[idx, 'config'],
                         df.loc[idx, 'trial'],
                         df.loc[idx, 'dff'])._asdict())

    if save_fig:
        factor_list = ['config']
        pdf.box_plot('dff', factors=factor_list, fname=fname, output_dir=output_dir)

    return pdf

#%%
def roidata_to_df_configs(rdata):

    '''
    Take subset of full ROIDATA dataframe using specified columns.
    Convert DF to make compatible w/ pyvttbl (and other).
    '''

    assert len(list(set(rdata['nframes_on'])))==1, "More than 1 idx found for nframes on... %s" % str(list(set(rdata['nframes_on'])))
    assert len(list(set(rdata['first_on'])))==1, "More than 1 idx found for first frame on... %s" % str(list(set(rdata['first_on'])))

    nframes_on = int(round(list(set(rdata['nframes_on']))[0]))
    first_on =  int(round(list(set(rdata['first_on']))[0]))

    df_groups = ['config', 'trial', 'df']
    groupby_list = ['config', 'trial']

    currdf = rdata[df_groups] #.sort_values(trans_types)
    grp = currdf.groupby(groupby_list)
    config_trials = {} # dict((config, []) for config in list(set(currdf['config'])))
    for k,g in grp: #config_trials.keys():
        if k[0] not in config_trials.keys():
            config_trials[k[0]] = {}

        config_trials[k[0]] = sorted(list(set(currdf.loc[currdf['config']==k[0]]['trial'])), key=natural_keys)

    idx = 0
    df_list = []
    for k,g in grp:
        #print k
        base_mean= g['df'][0:first_on].mean()
        base_std = g['df'][0:first_on].std()
        stim_mean = g['df'][first_on:first_on+nframes_on].mean()

        df_list.append(pd.DataFrame({'config': k[0],
                                     'trial': k[1], #'trial%05d' % int(config_trials[k[0]].index(k[1]) + 1),
                                     'dff': stim_mean,
                                     'zscore': stim_mean / base_std}, index=[idx]))
        idx += 2
    df = pd.concat(df_list, axis=0)
    df = df.sort_values(['config'])
    df = df.reset_index(drop=True)

    return df

#%%

def roidata_to_df_transforms(rdata, trans_types):

    '''
    Take subset of full ROIDATA dataframe using specified columns.
    Convert DF to make compatible w/ pyvttbl (and other).
    '''

    assert len(list(set(rdata['nframes_on'])))==1, "More than 1 idx found for nframes on... %s" % str(list(set(rdata['nframes_on'])))
    assert len(list(set(rdata['first_on'])))==1, "More than 1 idx found for first frame on... %s" % str(list(set(rdata['first_on'])))

    nframes_on = int(round(list(set(rdata['nframes_on']))[0]))
    first_on =  int(round(list(set(rdata['first_on']))[0]))

    # Check if 'xpos' or 'ypos' in trans_types, replace with 'position':
#    if 'xpos' in trans_types or 'ypos' in trans_types:
#        trans_types.extend(['position'])
#        trans_types = [t for t in trans_types if not (t == 'xpos') and not (t == 'ypos')]
    # Make sure trans_types sorted:
    trans_types = sorted(trans_types, key=natural_keys)

    df_groups = np.copy(trans_types).tolist()
    df_groups.extend(['trial', 'df'])
    currdf = rdata[df_groups] #.sort_values(trans_types)

    groupby_list = np.copy(trans_types).tolist()
    groupby_list.extend(['trial'])
    grp = currdf.groupby(groupby_list)
#    config_trials = {} # dict((config, []) for config in list(set(currdf['config'])))
#    for k,g in grp: #config_trials.keys():
#        if k[0] not in config_trials.keys():
#            config_trials[k[0]] = {}
#        config_trials[k[0]] = sorted(list(set(currdf.loc[currdf['config']==k[0]]['trial'])), key=natural_keys)

    idx = 0
    df_list = []
    for k,g in grp:
        #print k
        base_mean= g['df'][0:first_on].mean()
        base_std = g['df'][0:first_on].std()
        stim_mean = g['df'][first_on:first_on+nframes_on].mean()

        tdict = {'trial': k[-1],
                 'dff': stim_mean,
                 'zscore': stim_mean / base_std}
        for dkey in range(len(k)-1):
            tdict[trans_types[dkey]] = k[dkey]

        df_list.append(pd.DataFrame(tdict, index=[idx]))

        idx += 1

    df = pd.concat(df_list, axis=0)
    df = df.sort_values(trans_types)
    df = df.reset_index(drop=True)

    return df


def pd_to_pyvtt_transforms(df, trans_types):

    # Format pandas df into pyvttbl dataframe:
    #df_factors = ['config', 'trial', 'dff']
    df_factors = np.copy(trans_types).tolist()
    df_factors.extend(['trial', 'dff'])

    Trial = namedtuple('Trial', df_factors)
    pdf = pt.DataFrame()
    for idx in xrange(df.shape[0]):
        if len(trans_types)==1:
            pdf.insert(Trial(df.loc[idx, trans_types[0]],
                             df.loc[idx, 'trial'],
                             df.loc[idx, 'dff'])._asdict())
        elif len(trans_types)==2:
            pdf.insert(Trial(df.loc[idx, trans_types[0]],
                             df.loc[idx, trans_types[1]],
                             df.loc[idx, 'trial'],
                             df.loc[idx, 'dff'])._asdict())
        elif len(trans_types)== 3:
            pdf.insert(Trial(df.loc[idx, trans_types[0]],
                             df.loc[idx, trans_types[1]],
                             df.loc[idx, trans_types[2]],
                             df.loc[idx, 'trial'],
                             df.loc[idx, 'dff'])._asdict())

    return pdf



#%%
def pyvt_raw_epochXconfig(rdata, save_fig=False, output_dir='/tmp', fname='boxplot(intensity~epoch_X_config).png'):

    '''
    Take single ROI as a datatset, do split-plot rmANOVA:
        within-trial factor :  baseline vs. stimulus epoch
        between-trial factor :  stimulus condition
    '''

    assert len(list(set(rdata['nframes_on'])))==1, "More than 1 idx found for nframes on... %s" % str(list(set(rdata['nframes_on'])))
    assert len(list(set(rdata['first_on'])))==1, "More than 1 idx found for first frame on... %s" % str(list(set(rdata['first_on'])))

    nframes_on = int(round(list(set(rdata['nframes_on']))[0]))
    first_on =  int(round(list(set(rdata['first_on']))[0]))

    df_groups = ['config', 'trial', 'raw']
    groupby_list = ['config', 'trial']

    currdf = rdata[df_groups] #.sort_values(trans_types)
    grp = currdf.groupby(groupby_list)
    config_trials = {} # dict((config, []) for config in list(set(currdf['config'])))
    for k,g in grp: #config_trials.keys():
        if k[0] not in config_trials.keys():
            config_trials[k[0]] = {}

        config_trials[k[0]] = sorted(list(set(currdf.loc[currdf['config']==k[0]]['trial'])), key=natural_keys)

    idx = 0
    df_list = []
    for k,g in grp:
        #print k
        base_mean= g['raw'][0:first_on].mean()
        base_std = g['raw'][0:first_on].std()
        stim_mean = g['raw'][first_on:first_on+nframes_on].mean()

        df_list.append(pd.DataFrame({'config': k[0],
                                     'trial': k[1], #'trial%05d' % int(config_trials[k[0]].index(k[1]) + 1),
                                     'epoch': 'baseline',
                                     'intensity': base_mean}, index=[idx]))
        df_list.append(pd.DataFrame({'config': k[0],
                                     'trial': k[1], #'trial%05d' % int(config_trials[k[0]].index(k[1]) + 1),
                                     'epoch': 'stimulus',
                                     'intensity': stim_mean}, index=[idx+1]))
        idx += 2
    df = pd.concat(df_list, axis=0)
    df = df.sort_values(['epoch', 'config'])
    df = df.reset_index(drop=True)

    #pdf.pivot_table(index=['trial'], columns=['config', 'epoch'], values='intensity')

    # Format pandas df into pyvttbl dataframe:
    df_factors = ['config', 'trial', 'epoch', 'intensity']

    Trial = namedtuple('Trial', df_factors)
    pdf = pt.DataFrame()
    for idx in xrange(df.shape[0]):
        pdf.insert(Trial(df.loc[idx, 'config'],
                         df.loc[idx, 'trial'],
                         df.loc[idx, 'epoch'],
                         df.loc[idx, 'intensity'])._asdict())

    if save_fig:
        factor_list = ['config', 'epoch']
        pdf.box_plot('intensity', factors=factor_list, fname=fname, output_dir=output_dir)

    return pdf


#%%
def pyvt_raw_epochXsinglecond(rdata, curr_config='config001'):

    '''
    Treat single condition for single ROI as dataset, and do ANOVA with 'epoch' as factor.
    Test for main-effect of trial epoch -- but this is just 1-way...?
    '''

    assert len(list(set(rdata['nframes_on'])))==1, "More than 1 idx found for nframes on... %s" % str(list(set(rdata['nframes_on'])))
    assert len(list(set(rdata['first_on'])))==1, "More than 1 idx found for first frame on... %s" % str(list(set(rdata['first_on'])))

    nframes_on = int(round(list(set(rdata['nframes_on']))[0]))
    first_on =  int(round(list(set(rdata['first_on']))[0]))

    df_groups = ['config', 'trial', 'raw']
    groupby_list = ['config', 'trial']

    currdf = rdata[df_groups]

    # Split DF by current stimulus config:
    currdf = currdf[currdf['config']==curr_config]

    grp = currdf.groupby(groupby_list)
    trial_list = sorted(list(set(currdf['trial'])), key=natural_keys)

    idx = 0
    df_list = []
    for k,g in grp:
        print k
        base_mean= g['raw'][0:first_on].mean()
        base_std = g['raw'][0:first_on].std()
        stim_mean = g['raw'][first_on:first_on+nframes_on].mean()

        df_list.append(pd.DataFrame({'config': k[0],
                                     'trial': 'trial%05d' % int(trial_list.index(k[1]) + 1),
                                     'epoch': 'baseline',
                                     'intensity': base_mean}, index=[idx]))
        df_list.append(pd.DataFrame({'config': k[0],
                                     'trial': 'trial%05d' % int(trial_list.index(k[1]) + 1),
                                     'epoch': 'stimulus',
                                     'intensity': stim_mean}, index=[idx+1]))
        idx += 2
    df = pd.concat(df_list, axis=0)
    idxs = pd.Index(xrange(0, len(df)))
    df = df.sort_values(['epoch', 'config'])
    df = df.reset_index(drop=True)

    #pdf.pivot_table(index=['trial'], columns=['config', 'epoch'], values='intensity')

    # Format pandas df into pyvttbl dataframe:
    df_factors = ['trial', 'epoch', 'intensity']

    Trial = namedtuple('Trial', df_factors)
    pdf = pt.DataFrame()
    for idx in xrange(df.shape[0]):
        pdf.insert(Trial(df.loc[idx, 'trial'],
                         df.loc[idx, 'epoch'],
                         df.loc[idx, 'intensity'])._asdict())

    return pdf


#%%

def extract_apa_anova2(factor, aov, values = ['F', 'mse', 'eta', 'p']):
    results = {}
    for key,result in aov[(factor)].iteritems():
        if key in values:
            results[key] = result

    return results

#%%
def id_visual_cells_mp(DATA, output_dir='/tmp', nprocs=4):

    roi_list = sorted(list(set(DATA['roi'])), key=natural_keys)
    print("Calculating split-plot ANOVA (factors=epoch, config) for %i rois." % len(roi_list))

    t_eval_mp = time.time()

    def worker(roi_list, DATA, output_dir, out_q):
        """
        Worker function is invoked in a process. 'roi_list' is a list of
        roi names to evaluate [rois00001, rois00002, etc.]. Results are placed
        in a dict that is pushed to a queue.
        """
        outdict = {}
        for roi in roi_list:
            rdata = DATA[DATA['roi']==roi]
            outdict[roi] = splitplot_anova2(rdata, output_dir, asdict=True)
        out_q.put(outdict)

    # Each process gets "chunksize' filenames and a queue to put his out-dict into:
    out_q = mp.Queue()
    chunksize = int(math.ceil(len(roi_list) / float(nprocs)))
    procs = []

    for i in range(nprocs):
        p = mp.Process(target=worker,
                       args=(roi_list[chunksize * i:chunksize * (i + 1)],
                                       DATA,
                                       out_q))
        procs.append(p)
        p.start()

    # Collect all results into single results dict. We should know how many dicts to expect:
    resultdict = {}
    for i in range(nprocs):
        resultdict.update(out_q.get())

    # Wait for all worker processes to finish
    for p in procs:
        print "Finished:", p
        p.join()

    print_elapsed_time(t_eval_mp)

    return resultdict

#%%
def splitplot_anova2(rdata, output_dir, asdict=True):

    pdf = pyvt_raw_epochXconfig(rdata, save_fig=False)

    # Calculate ANOVA split-plot:
    aov = pdf.anova('intensity', sub='trial',
                       wfactors=['epoch'],
                       bfactors=['config'])
    #print(aov)

    aov_results_fpath = os.path.join(output_dir, 'visual_anova_results_%s.txt' % roi)
    with open(aov_results_fpath,'wb') as f:
        f.write(str(aov))

    #etas = get_effect_sizes(aov, factor_a='epoch', factor_b='config')
    res_epoch = extract_apa_anova2(('epoch',), aov)
    #res_interaction = extract_apa_anova2(('epoch', 'config'), aov)
    if res_epoch['p'] < 0.1: # or res_interaction['p'] < 0.1:
        responsive_rois[roi] = {'F': res_epoch['F'], 'p': res_epoch['p']} #.append(roi)

    if asdict is True:
        return res_epoch
    else:
        return res_epoch['F'], res_epoch['p']

#%%
def id_visual_cells(DATA, save_figs=False, output_dir='/tmp'):
    '''
    For each ROI, do split-plot ANOVA --
        between-groups factor :  config
        within-groups factor :  epoch
    Use raw intensity to avoid depence of trial-epoch values.
    Save ANOVA results to disk.

    Returns:
        dict() -- keys are rois with p-value < 0.1, values are 'F' and 'p'
    '''

    roi_list = sorted(list(set(DATA['roi'])), key=natural_keys)
    print("Calculating split-plot ANOVA (factors=epoch, config) for %i rois." % len(roi_list))

    #curr_config = 'config006'

    #pdf = pyvt_raw_epochXsinglecond(rdata, curr_config=curr_config)
    #aov = pdf.anova('intensity', sub='trial', wfactors=['epoch'])
    #aov1 = pdf.anova1way('intensity', 'epoch')

    responsive_rois = {} #[]
    for roi in roi_list:
        print roi

        rdata = DATA[DATA['roi']==roi]
        pdf = pyvt_raw_epochXconfig(rdata, save_fig=False)

        # Calculate ANOVA split-plot:
        aov = pdf.anova('intensity', sub='trial',
                           wfactors=['epoch'],
                           bfactors=['config'])
        #print(aov)

        aov_results_fpath = os.path.join(output_dir, 'visual_anova_results_%s.txt' % roi)
        with open(aov_results_fpath,'wb') as f:
            f.write(str(aov))

        #etas = get_effect_sizes(aov, factor_a='epoch', factor_b='config')
        res_epoch = extract_apa_anova2(('epoch',), aov)
        #res_interaction = extract_apa_anova2(('epoch', 'config'), aov)
        if res_epoch['p'] < 0.1: # or res_interaction['p'] < 0.1:
            responsive_rois[roi] = {'F': res_epoch['F'], 'p': res_epoch['p']} #.append(roi)

#        if roi in responsive_rois and save_figs is True:
#            factor_list = ['config', 'epoch']
#            fname = '%s_boxplot(intensity~epoch_X_config).png' % roi
#            pdf.box_plot('intensity', factors=factor_list, fname=fname, output_dir=output_dir)

    return responsive_rois


#%%
def plot_box_raw(DATA, roi_list, output_dir='/tmp'):

    for roi in roi_list:
        rdata = DATA[DATA['roi']==roi]
        pdf = pyvt_raw_epochXconfig(rdata, save_fig=False)
        factor_list = ['config', 'epoch']
        fname = '%s_boxplot(intensity~epoch_X_config).png' % roi
        pdf.box_plot('intensity', factors=factor_list, fname=fname, output_dir=output_dir)

#%%

def selectivity_KW(rdata, post_hoc='dunn', asdict=True):

    # Get standard dataframe (not pyvttbl):
    df = roidata_to_df_configs(rdata)

    # Format dataframe and do KW test:
    groupedconfigs = {}
    for grp in df['config'].unique():
        groupedconfigs[grp] = df[df['config']==grp]['dff'].values
    args = groupedconfigs.values()
    H, p = stats.kruskal(*args)

    # Do post-hoc test:
    if post_hoc == 'dunn':
        pc = sp.posthoc_dunn(df, val_col='dff', group_col='config')
    elif post_hoc == 'conover':
        pc = sp.posthoc_conover(df, val_col='dff', group_col='config')

    # Save ROI info:
    posthoc_results = {'H': H,
                       'p': p,
                       'post_hoc': post_hoc,
                       'p_rank': pc}
    if asdict is True:
        return posthoc_results
    else:
        return posthoc_results['H'], posthoc_results['p'], pc

#%%
def id_selective_cells_mp(DATA, nprocs=4):

    roi_list = sorted(list(set(DATA['roi'])), key=natural_keys)
    print("Calculating split-plot ANOVA (factors=epoch, config) for %i rois." % len(roi_list))

    t_eval_mp = time.time()

    def worker(roi_list, DATA, out_q):
        """
        Worker function is invoked in a process. 'roi_list' is a list of
        roi names to evaluate [rois00001, rois00002, etc.]. Results are placed
        in a dict that is pushed to a queue.
        """
        outdict = {}
        for roi in roi_list:
            print roi
            rdata = DATA[DATA['roi']==roi]
            outdict[roi] = selectivity_KW(rdata, post_hoc='dunn', asdict=True)
        out_q.put(outdict)

    # Each process gets "chunksize' filenames and a queue to put his out-dict into:
    out_q = mp.Queue()
    chunksize = int(math.ceil(len(roi_list) / float(nprocs)))
    procs = []

    for i in range(nprocs):
        p = mp.Process(target=worker,
                       args=(roi_list[chunksize * i:chunksize * (i + 1)],
                                       DATA,
                                       out_q))
        procs.append(p)
        p.start()

    # Collect all results into single results dict. We should know how many dicts to expect:
    resultdict = {}
    for i in range(nprocs):
        resultdict.update(out_q.get())

    # Wait for all worker processes to finish
    for p in procs:
        print "Finished:", p
        p.join()

    print_elapsed_time(t_eval_mp)

    return resultdict

#%%
def id_selective_cells(DATA, roi_list, topn=10, test_normal=False, post_hoc='dunn', save_figs=False, output_dir='/tmp'):

    ph_results = {}

    for ridx,roi in enumerate(roi_list):
        rdata = DATA[DATA['roi']==roi]

        # Get standard dataframe (not pyvttbl):
        df = roidata_to_df_configs(rdata)

        if ridx < topn and save_figs:
            print roi
            #% Sort configs by mean value:
            grped = df.groupby(['config']) #.mean()
            df2 = pd.DataFrame({col:vals['dff'] for col,vals in grped})
            meds = df2.median().sort_values(ascending=False)
            df2 = df2[meds.index]
            pl.figure(figsize=(10,5))
            ax = sns.boxplot(data=df2)
            pl.title(roi)
            pl.ylabel('df/f')
            ax.set_xticklabels(['%i deg\n%.2f cpd\n%s' % (stimconfigs[t.get_text()]['rotation'],
                                                          stimconfigs[t.get_text()]['frequency'],
                                                          t.get_text()) for t in ax.get_xticklabels()])

            figname = 'box_mediandff_%s.png' % roi
            pl.savefig(os.path.join(output_dir, figname))
            pl.close()

        normality = False
        if test_normal:
            k2, pn = stats.mstats.normaltest(df['dff'])
            if pn < 0.05:
                print("Normal test: p < 0.05, k=%.2f" % k2)
                normality = False
            else:
                print("Normal test: p > 0.05, k=%.2f" % k2)
                normality = True

            # Check for normality:
            if ridx < topn and save_figs:
                pl.figure()
                qq_res = stats.probplot(df['dff'], dist="norm", plot=pl)
                pl.title('P-P plot %s' % roi)
                pl.text(-2, 0.3, 'p=%s' % str(pn))
                pl.show()
                figname = 'PPplot_%s.png' % roi
                pl.savefig(os.path.join(output_dir, figname))
                pl.close()

            # Check if STDs are equal (ANOVA):
            #df.groupby(['config']).std()

        if normality is False:
            # Format dataframe and do KW test:
            groupedconfigs = {}
            for grp in df['config'].unique():
                groupedconfigs[grp] = df[df['config']==grp]['dff'].values
            args = groupedconfigs.values()
            H, p = stats.kruskal(*args)

            # Do post-hoc test:
            if post_hoc == 'dunn':
                pc = sp.posthoc_dunn(df, val_col='dff', group_col='config')
            elif post_hoc == 'conover':
                pc = sp.posthoc_conover(df, val_col='dff', group_col='config')

            if ridx < topn and save_figs:
                # Plot heatmap of p-values from post-hoc test:
                pl.figure(figsize=(10,8))
                pl.title('%s test, %s' % (post_hoc, roi))
                cmap = ['1', '#fb6a4a',  '#08306b',  '#4292c6', '#c6dbef']
                heatmap_args = {'cmap': cmap, 'linewidths': 0.25, 'linecolor': '0.5',
                                'clip_on': False, 'square': True,
                                'cbar_ax_bbox': [0.90, 0.35, 0.02, 0.3]}
                sp.sign_plot(pc, **heatmap_args)
                figname = 'pvalues_%s_%s.png' % (roi, post_hoc)
                pl.savefig(os.path.join(output_dir, figname))
                pl.close()
#        else:
#
#            # 1-way ANOVA (only valid under condNs):
#            pdf = pyvt_stimdf_configs(rdata)
#            aov = pdf.anova1way('dff', 'config') #
#            print(aov)
#
#            tukey = pairwise_tukeyhsd(df['dff'], df['config'])
#            print(tukey)

        # Save ROI info:
        ph_results[roi] = {'H': H,
                           'p': p,
                           'post_hoc': post_hoc,
                           'p_rank': pc}

    return ph_results

#%%
def uint16_to_RGB(img):
    im = img.astype(np.float64)/img.max()
    im = 255 * im
    im = im.astype(np.uint8)
    rgb = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    return rgb


def assign_roi_selectivity(DATA, roi_list, stimconfigs):

    selectivity = {}
    for roi in roi_list:
        ridx = int(roi[3:]) - 1

        rdata = DATA[DATA['roi']==roi]

        # Get standard dataframe (not pyvttbl):
        df = roidata_to_df_configs(rdata)

        stimdf_means = df.groupby(['config'])['dff'].mean()
        ordered_configs = stimdf_means.sort_values(ascending=False).index
        Rmost = stimdf_means[ordered_configs[0]]
        Rleast = stimdf_means[ordered_configs[-1]]
        SI = (Rmost - Rleast) / (Rmost + Rleast)

        # If > 1 SF, use best one:
        sfs = list(set([stimconfigs[config]['frequency'] for config in stimconfigs.keys()]))
        sort_config_types = {}
        for sf in sfs:
            sort_config_types[sf] = sorted([config for config in stimconfigs.keys()
                                                if stimconfigs[config]['frequency']==sf],
                                                key=lambda x: stimconfigs[x]['rotation'])

        oris = [stimconfigs[config]['rotation'] for config in sort_config_types[sf]]

        orientation_list = sort_config_types[stimconfigs[ordered_configs[0]]['frequency']]

        OSI = np.abs( sum([stimdf_means[cfg]*np.exp(2j*theta) for theta, cfg in zip(oris, orientation_list)]) / sum([stimdf_means[cfg] for cfg in  orientation_list]) )

        selectivity[roi] = {'ridx': ridx,
                            'SI': SI,
                            'OSI': OSI,
                            'ori': stimconfigs[ordered_configs[0]]['rotation']}

    return selectivity

#%%
def get_reference_config(Cmax_overall, trans_types, transform_dict):
    Cref = {}
    if 'xpos' in trans_types:
        xpos_tidx = trans_types.index('xpos')
        ref_xpos = Cmax_overall[xpos_tidx]
    else:
        ref_xpos = transform_dict['xpos'][0]

    if 'ypos' in trans_types:
        ypos_tidx = trans_types.index('ypos')
        ref_ypos = Cmax_overall[ypos_tidx]
    else:
        ref_ypos = transform_dict['ypos'][0]

    if 'size' in trans_types:
        size_tidx = trans_types.index('size')
        ref_size = Cmax_overall[size_tidx]
    else:
        ref_size = transform_dict['size'][0]

    Cref = {'xpos': ref_xpos,
            'ypos': ref_ypos,
            'size': ref_size}

    return Cref

#%% SELECTIVITY -- calculate a sparseness measure:

# Case:  Transformations in xpos, ypos
# Take as "reference" position, the x- and y-position eliciting the max response

def calc_sparseness(df, trans_types, transform_dict):

    stimdf_means = df.groupby(trans_types)['dff'].mean()
    ordered_configs = stimdf_means.sort_values(ascending=False).index
    if isinstance(ordered_configs, pd.MultiIndex):
        ordered_configs = ordered_configs.tolist()
    Cmax_overall = ordered_configs[0]

    Cref = get_reference_config(Cmax_overall, trans_types, transform_dict)


    object_resp_df = stimdf_means.copy()
    if 'xpos' in trans_types:
        object_resp_df = object_resp_df.xs(Cref['xpos'], level='xpos')
    if 'ypos' in trans_types:
        object_resp_df = object_resp_df.xs(Cref['ypos'], level='ypos')
    if 'size' in trans_types:
        object_resp_df = object_resp_df.xs(Cref['size'], level='size')

    object_list = object_resp_df.index.tolist()
    nobjects = len(object_list)
    t1a = (sum([(object_resp_df[i] / nobjects) for i in object_list])**2)
    t1b = sum([object_resp_df[i]/nobjects for i in object_list])
    S = (1 - (t1a / t1b)) / (1-(1/nobjects))

    sparseness_ref = {'S': S, 'object_responses': object_resp_df}

    return sparseness_ref

#%%
def assign_selectivity_index_mp(roi_list, DATA, trans_types, transform_dict, nprocs=4):

    print("Calculating SPARSENESS index for %i rois." % len(roi_list))

    t_eval_mp = time.time()

    def worker(roi_list, DATA, trans_types, transform_dict, out_q):
        """
        Worker function is invoked in a process. 'roi_list' is a list of
        roi names to evaluate [rois00001, rois00002, etc.]. Results are placed
        in a dict that is pushed to a queue.
        """
        outdict = {}
        for roi in roi_list:
            print roi
            rdata = DATA[DATA['roi']==roi]
            df = roidata_to_df_transforms(rdata, trans_types)
            outdict[roi] = calc_sparseness(df, trans_types, transform_dict)
        out_q.put(outdict)

    # Each process gets "chunksize' filenames and a queue to put his out-dict into:
    out_q = mp.Queue()
    chunksize = int(math.ceil(len(roi_list) / float(nprocs)))
    procs = []

    for i in range(nprocs):
        p = mp.Process(target=worker,
                       args=(roi_list[chunksize * i:chunksize * (i + 1)],
                                       DATA,
                                       trans_types,
                                       transform_dict,
                                       out_q))
        procs.append(p)
        p.start()

    # Collect all results into single results dict. We should know how many dicts to expect:
    resultdict = {}
    for i in range(nprocs):
        resultdict.update(out_q.get())

    # Wait for all worker processes to finish
    for p in procs:
        print "Finished:", p
        p.join()

    print_elapsed_time(t_eval_mp)

    return resultdict


#%%
#
#roi = 'roi00006'
#rdata = DATA[DATA['roi']==roi]
#df = roidata_to_df_transforms(rdata, trans_types)
#
#stimdf_means = df.groupby(trans_types)['dff'].mean()
#

#%% Load ROI DATA:

#    S = {1 − [(ΣRi/n)2/Σ(Ri2/n)]}/[1 − (1/n)],
#
#    MT = [n − (ΣRi/Rmax)]/(n − 1),
#    ST = 〈Rtest size/max(Rtest size)〉

multiproc = False
nprocesses = 4


rootdir = '/mnt/odyssey'
animalid = 'CE077' #'CE077'
session = '20180329' #'20180321'
acquisition = 'FOV2_zoom1x' #'FOV1_zoom1x'

combined = True
nruns = 4
run_basename = 'gratings'
traces_basename = 'traces001'
if 'combined' is True:
    runfolder = ['%s_run%i' % int(i+1) for i in range(nruns)]
    runfolder = '_'.join(runfolder)
    tracedir = [traces_basename for i in range(nruns)]
    tracedir = '_'.join(tracedir)
else:
    runfolder = 'gratings_run1_gratings_run2_gratings_run3_gratings_run4' #'gratings_phasemod'# 'blobs_run3_blobs_run4'
    tracedir =  'traces001_traces001_traces001_traces001' #'traces/traces005_69d8ec' #'traces002_traces002'

trace_type = 'raw'

# TODO:  all trials included to keep N samples the same for each group...
# Need to adjust for differing trial nums b/w groups
filter_pupil = False
pupil_radius_min = 20
pupil_radius_max = 60
pupil_dist_thr = 3.0

traceid_dir = os.path.join(rootdir, animalid, session, acquisition, runfolder, tracedir)
print(traceid_dir)

#%% # Load ROIDATA file:
roidf_fn = [i for i in os.listdir(traceid_dir) if i.endswith('hdf5') and 'ROIDATA' in i][0]
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

STATS, stats_filepath = cb.get_combined_stats(DATA, datakey, traceid_dir, trace_type=trace_type, filter_pupil=filter_pupil, pupil_params=pupil_params)


#%%  Create output dir for ROI selection:
# =============================================================================

if '/' in datakey:
    datakey = datakey[1:]
sort_dir = os.path.join(traceid_dir, 'sorted_%s' % datakey)
sort_resultsdir = os.path.join(sort_dir, 'anova_results')
sort_figdir = os.path.join(sort_dir, 'figures')

responsive_resultsdir = os.path.join(sort_dir, 'anova_results', 'responsive_tests')
selective_resultsdir = os.path.join(sort_dir, 'anova_results', 'selectivity_tests')

if not os.path.exists(sort_figdir):
    os.makedirs(sort_figdir)

if not os.path.exists(responsive_resultsdir):
    os.makedirs(responsive_resultsdir)
if not os.path.exists(selective_resultsdir):
    os.makedirs(selective_resultsdir)

tolerance_figdir = os.path.join(sort_dir, 'figures', 'tolerance')
if not os.path.exists(tolerance_figdir):
    os.makedirs(tolerance_figdir)

#%% Get stimulus config info:
# =============================================================================

rundir = os.path.join(rootdir, animalid, session, acquisition, runfolder)

if combined is True:
    stimconfigs_fpath = os.path.join(traceid_dir, 'stimulus_configs.json')
else:
    stimconfigs_fpath = os.path.join(rundir, 'paradigm', 'stimulus_configs.json')

with open(stimconfigs_fpath, 'r') as f:
    stimconfigs = json.load(f)

#%% Identify visually repsonsive cells:
# =============================================================================

if multiproc is True:
    responsive_rois = id_visual_cells_mp(DATA, output_dir=responsive_resultsdir, nprocs=nprocesses)
else:
    responsive_rois = id_visual_cells(DATA, output_dir=responsive_resultsdir, save_figs=False)

roi_list = sorted(list(set(DATA['roi'])), key=natural_keys)
print("%i out of %i cells pass split-plot ANOVA test for visual responses." % (len(responsive_rois), len(roi_list)))

# Save responsive roi list to disk:
with open(os.path.join(sort_dir, 'visual_rois.json'), 'w') as f:
    json.dump(responsive_rois, f, indent=4, sort_keys=True)

# Sort ROIs by F ratio:
sorted_rois_visual = sorted(responsive_rois, key=lambda x: responsive_rois[x]['F'])[::-1]

# Plot top 10 "most responsive":
topn = 10
print("Plotting box plots for factors EPOCH x CONFIG for %i rois." % topn)
vis_responsive_dir = os.path.join(sort_figdir, 'responsive')
if not os.path.exists(vis_responsive_dir):
    os.makedirs(vis_responsive_dir)

plot_box_raw(DATA, sorted_rois_visual[0:topn], output_dir=vis_responsive_dir)

#%% Identify selective cells:
# =============================================================================

#roi = sorted_rois[6]
#print(roi)
post_hoc = 'dunn'
vis_selective_dir = os.path.join(sort_figdir, 'selectivity')
if not os.path.exists(vis_selective_dir):
    os.makedirs(vis_selective_dir)

multiproc = True
nprocesses = 4
if multiproc is True:
    ph_results = id_selective_cells_mp(DATA, nprocs=nprocesses)
else:
    ph_results = id_selective_cells(DATA, sorted_rois_visual,
                                    post_hoc=post_hoc, topn=10, save_figs=False,
                                    output_dir=vis_selective_dir)

posthoc_fpath = os.path.join(sort_dir, 'selectivity_KW_posthoc_%s.npz' % post_hoc)
np.savez(posthoc_fpath, ph_results)
# Note, to reload as dict:
# ph_results = {key:ph_results[key].item() for key in ph_results}

#ph_tojson = ph_results.copy()
## Save responsive roi list to disk:
#for k in ph_tojson.keys():
#    ph_tojson[k]['p_rank'] = ph_tojson[k]['p_rank'].to_json(orient='values')
#with open(os.path.join(sort_dir, 'posthoc_%s.json' % post_hoc), 'w') as f:
#    json.dump(ph_tojson, f, indent=4, sort_keys=True)


# Get ROIs that pass KW test:
selective_rois = [r for r in sorted_rois_visual if ph_results[r]['p'] < 0.05]
sorted_rois_selective = sorted(selective_rois, key=lambda x: ph_results[x]['H'])[::-1]


H_mean = np.mean([ph_results[r]['H'] for r in sorted_rois_selective])
H_std = np.std([ph_results[r]['H'] for r in sorted_rois_selective])

print("**********************************************************************")
print("%i out of %i cells are visually responsive (split-plot ANOVA, p < 0.05)" % (len(responsive_rois), len(roi_list)))
print("%i out of %i visual are stimulus selective (Kruskal-Wallis, p < 0.05)" % (len(selective_rois), len(responsive_rois)))
print("Mean H=%.2f (std=%.2f)" % (H_mean, H_std))
print("**********************************************************************")


#%% Look at p-values heatmap of specific ROI:

#roi = 'roi00001'
#pl.figure(figsize=(10,8))
#pl.title('%s test, %s' % (post_hoc, roi))
#cmap = ['1', '#fb6a4a',  '#08306b',  '#4292c6', '#c6dbef']
#heatmap_args = {'cmap': cmap, 'linewidths': 0.25, 'linecolor': '0.5',
#                'clip_on': False, 'square': True,
#                'cbar_ax_bbox': [0.90, 0.35, 0.02, 0.3]}
#sp.sign_plot(ph_results[roi]['p_rank'], **heatmap_args)
#
#figname = 'pvalues_%s_%s.png' % (roi, post_hoc)
#pl.savefig(os.path.join(output_dir, figname))


#%% Look for sig diffs between SHAPE vs. TRANSFORM:

#roi = 'roi00006'


shape_selective_rois = {}
anova_results = {}
for roi in sorted_rois_visual:
    rdata = DATA[DATA['roi']==roi]
    #rdata['position'] = list(zip(rdata['xpos'], rdata['ypos']))

    df = roidata_to_df_transforms(rdata, trans_types)

    pdf = pd_to_pyvtt_transforms(df, trans_types)

    # Calculate ANOVA split-plot:
    aov = pdf.anova('dff', sub='trial',
                       bfactors=trans_types)
    #print(aov)

    aov_results_fpath = os.path.join(selective_resultsdir, 'selectivity_2wayanova_results_%s.txt' % roi)
    with open(aov_results_fpath,'wb') as f:
        f.write(str(aov))

    #etas = get_effect_sizes(aov, factor_a='epoch', factor_b='config')
    # Identify which, if any, factors are significant:
    factor_types = aov.keys()
    res_epoch = {}
    for factor in factor_types:
        res_epoch[factor] = extract_apa_anova2(factor, aov)
    #res_epoch = extract_apa_anova2(('ori',), aov)

    #res_interaction = extract_apa_anova2(('epoch', 'config'), aov)
    sig_factors = [k for k in res_epoch.keys() if res_epoch[k]['p'] < 0.05]
    if len(sig_factors) > 0:
    #if res_epoch['p'] < 0.1: # or res_interaction['p'] < 0.1:
        shape_selective_rois[roi] = sig_factors

    anova_results[roi] = res_epoch #{'F': res_epoch['F'], 'p': res_epoch['p']} #.append(roi)


#%%

from statsmodels.stats.libqsturng import psturng

roi = 'roi00006'
rdata = DATA[DATA['roi']==roi]
#df = roidata_to_df_configs(rdata)
df = roidata_to_df_transforms(rdata, trans_types)

aov = pdf.anova('dff', sub='trial', bfactors=trans_types)
#res_epoch = extract_apa_anova2(('ori',), aov)

res2 = pairwise_tukeyhsd(df['dff'], df['xpos'])

res2 = pairwise_tukeyhsd(df['dff'], df['ori']) # df['xpos'], df['ypos'])
#res2 = pairwise_tukeyhsd(df['dff'], df['config']) # df['xpos'], df['ypos'])
print(res2)

#mod = MultiComparison(df['dff'], df['ori'])
#mod = MultiComparison(df['dff'], df['config'])
mod = MultiComparison(df['dff'], df['xpos'])
print(mod.tukeyhsd())




st_range = np.abs(res2.meandiffs) / res2.std_pairs
pvalues = psturng(st_range, len(res2.groupsunique), res2.df_total)

group_labels = res2.groupsunique

# plot:
res2.plot_simultaneous()

pl.figure()
pl.plot(xrange(len(res2.meandiffs)), np.abs(res2.meandiffs), 'o')
pl.errorbar(xrange(len(res2.meandiffs)), np.abs(res2.meandiffs), yerr=np.abs(res2.std_pairs.T - np.abs(res2.meandiffs)))
xlim = -0.5, len(res2.meandiffs)-0.5
pl.hlines(0, *xlim)
pl.xlim(*xlim)
pair_labels = res2.groupsunique[np.column_stack(mod.pairindices)] #[1][0])]
pl.xticks(xrange(len(res2.meandiffs)), pair_labels)
pl.title('Multiple Comparison of Means - Tukey HSD, FWER=0.05' +
          '\n Pairwise Mean Differences')


# Pairwise t-tests:
# TukeyHSD uses joint variance across all samples
# Pairwise ttest calculates joint variance estimate for each pair of samples separately

# Paired samples:  Assumes that samples are paired (i.e., each subject goes through all X treatments)

rtp = mod.allpairtest(stats.ttest_rel, method='Holm')
print(rtp[0])
rtp_bon = mod.allpairtest(stats.ttest_rel, method='b')
print(rtp_bon[0])

# Indep samples:  Assumes that samples are paired (i.e., each subject goes through all X treatments)
# Pairwise ttest calculates joint variance estimate for each pair of samples separately
# ... but stats.ttest_ind looks at one pair at a time --
# So, calculate a pairwise ttest that takes a joint variance as given and feed
# it to mod.allpairtest?
itp_bon = mod.allpairtest(stats.ttest_ind, method='b')
print(itp_bon[0])

# TukeyHSD returns mean and variance for studentized range statistic
# studentized ranged statistic is t-statistic except scaled by np.sqrt(2)
#t_stat = res2[1][2] / res2[1][3] / np.sqrt(2)
t_stat = np.abs(res2.meandiffs) / res2.std_pairs / np.sqrt(2)
print(t_stat)
my_pvalues = stats.t.sf(np.abs(t_stat), res2.df_total) * 2   #two-sided
print(my_pvalues) # Uncorrected p-values (same as R)

# Do multiple-testing p-value correction (Bonferroni):
from statsmodels.stats.multitest import multipletests
res_b = multipletests(my_pvalues, method='b')

# False discvoery rate correction:
res_fdr = multipletests(my_pvalues, method='fdr_bh')


#%% Assign selectivity:
# =============================================================================

# SI = 0 (no selectivity) --> 1 (high selectivity)
# 1. Split trials in half, assign shapes to which neuron shows highest/lowest activity
# 2. Use other half to measure activity to assigned shapes and calculate SI:
#    SI = (Rmost - Rleast) / (Rmost + Rleast)

#SI = (Rmost - Rleast) / (Rmost + Rleast)


#S = assign_roi_selectivity(DATA, sorted_rois_visual, stimconfigs)

S = assign_selectivity_index_mp(sorted_rois_visual, DATA, trans_types, transform_dict, nprocs=4)

# Histogram of sparseness:
# --------------------------------------------
sparseness_values = [S[r]['S'] for r in S.keys()] #S.values()

highS_color = 'mediumvioletred'
lowS_color = 'forestgreen'

sns.set_style("white")
pl.figure()
sns.distplot(sparseness_values, bins=50, kde=False, rug=True, color="#34495e")
pl.xlabel('Sparseness (S)')
pl.ylabel('Cell counts')
s_bounds = np.linspace(0, 1, num=4, endpoint=True) # Divide range into 3 bins
pl.axvline(x=s_bounds[1], linewidth=2, linestyle='--', color=lowS_color)
pl.axvline(x=s_bounds[2], linewidth=2, linestyle='--', color=highS_color) #'r')

#pl.plot([s_bounds[1], s_bounds[1]], [0, 20], linewidth=2, linestyle='-', color='b')
#pl.plot([s_bounds[2], s_bounds[2]], [0, 20], linewidth=2, linestyle='-', color='r')
figname = 'hist_sparseness.png'
pl.savefig(os.path.join(sort_figdir, figname))
pl.close()

# For all ROIs in FOV, plot normalized firing rate (normalized to MAX) as a
# function of (ranked) objects:
pl.figure()
for r in S.keys():
    normed_dff = S[r]['object_responses'].values / S[r]['object_responses'].max()
    pl.plot(xrange(len(normed_dff)), sorted(normed_dff)[::-1], 'k', alpha=0.5, linewidth=0.5)
pl.xticks(xrange(len(normed_dff)))
pl.xlabel('Ranked objects')
pl.ylabel('Normalized df/f')
pl.gca().set_xticklabels(np.arange(1, len(normed_dff)+1))

# Choose 2 examples showing low-sparseness and high-sparseness
lowS_ex = [k for k in S.keys() if S[k]['S'] < 0.2][0]
highS_ex = 'roi00006' # [k for k in S.keys() if S[k]['S'] > 0.8][0]

sorted_lowS_vals = sorted(S[lowS_ex]['object_responses'].values / S[lowS_ex]['object_responses'].max())[::-1]
sorted_highS_vals = sorted(S[highS_ex]['object_responses'].values / S[highS_ex]['object_responses'].max())[::-1]

pl.plot(xrange(len(normed_dff)), sorted_lowS_vals, color=lowS_color, linestyle='--', linewidth=2)
pl.plot(xrange(len(normed_dff)), sorted_highS_vals, color=highS_color, linestyle='--', linewidth=2)

figname = 'ranked_normed_dff.png'
pl.savefig(os.path.join(sort_figdir, figname))
pl.close()

#%% Assign TOLERANCE index:

tolerance_figdir = os.path.join(sort_dir, 'figures', 'tolerance')
if not os.path.exists(tolerance_figdir):
    os.makedirs(tolerance_figdir)

# Case:  POSITION (XPOS)

roi = 'roi00006'
rdata = DATA[DATA['roi']==roi]
#df = roidata_to_df_configs(rdata)
df = roidata_to_df_transforms(rdata, trans_types)

stimdf_means = df.groupby(trans_types)['dff'].mean()
ordered_configs = stimdf_means.sort_values(ascending=False).index
if isinstance(ordered_configs, pd.MultiIndex):
    ordered_configs = ordered_configs.tolist()
Cmax_overall = ordered_configs[0]

Cref = get_reference_config(Cmax_overall, trans_types, transform_dict)

transform = 'xpos'

Tref = Cref[transform]
Tflankers = [tval for tval in transform_dict[transform] if not tval==Tref]

const_trans_types = [trans for trans in Cref.keys() if not trans==transform and trans in df.keys()]

objectid = 'ori'
transdf = df[[i for i in df[c]==Cref[c] for c in const_trans_types]]
means = transdf.groupby([objectid, transform]).mean()

fig, ax = pl.subplots()
count = 0
def func(group, objectid, transform):
    grp = group.groupby(transform)
    xvals = grp.mean().index.get_level_values(transform).unique().tolist()
    obj_name = grp[objectid].unique().tolist()[0][0]
    print(obj_name)
    global ax, count
    if count > 0:
        #ax.plot(xvals, grp.dff.mean(), label=obj_name)
        ax.errorbar(xvals, grp.dff.mean(), yerr=grp.dff.sem(), label=obj_name)
    count += 1
    return group

xvals = means.index.get_level_values(transform).unique()
#means.groupby(objectid).apply(func, xvals)
transdf.groupby([objectid]).apply(func, objectid, transform)
ax.legend(loc='best', numpoints=1)
pl.xticks(xvals)
pl.xlabel(transform)
pl.ylabel('df/f')
pl.title(roi)
figname = '%s_%s_across_%s.png' % (roi, objectid, transform)
pl.savefig(os.path.join(tolerance_figdir, figname))
pl.close()
#%%

# Identify reference file and load MASKS:
session_dir = os.path.join(rootdir, animalid, session)
trace_id = os.path.split(traceid_dir)[-1].split('_')[0]

if combined is True:
    tmp_runfolder = '_'.join([runfolder.split('_')[0], runfolder.split('_')[1]])
    tmp_rundir = os.path.join(session_dir, acquisition, tmp_runfolder)
    TID = load_TID(tmp_rundir, trace_id)
    tmp_traceid_dir = TID['DST']
else:
    TID = load_TID(rundir, trace_id)
    tmp_traceid_dir = traceid_dir

with open(os.path.join(session_dir, 'ROIs', 'rids_%s.json' % session), 'r') as f:
    rdict = json.load(f)

RID = rdict[TID['PARAMS']['roi_id']]
ref_file = 'File%03d' % int(RID['PARAMS']['options']['ref_file'])

if rootdir not in tmp_traceid_dir:
    tmp_traceid_dir = replace_root(tmp_traceid_dir, rootdir, animalid, session)

maskpath = os.path.join(tmp_traceid_dir, 'MASKS.hdf5')
masks = h5py.File(maskpath, 'r')




#%% Plot best ORIENTATION on overlay of fov:

colors = {0: (255, 0, 0),
          45: (255, 255, 0),
          90: (0, 102, 0),
          135: (0, 0, 255)}

img_src = masks[ref_file]['Slice01']['zproj'].attrs['source']
if 'std' in img_src:
    img_src = img_src.replace('std', 'mean')
if rootdir not in img_src:
    img_src = replace_root(img_src, rootdir, animalid, session)

img = tf.imread(img_src)
#img = masks[ref_file]['Slice01']['zproj'][:]
dims = img.shape
maskarray = masks[ref_file]['Slice01']['maskarray']

label_rois = True
imgrgb = uint16_to_RGB(img)

# Plot average img, overlay ROIs with OSI (visual only):
sns.set_style('white')
fig = pl.figure()
ax = fig.add_subplot(111)
pl.imshow(imgrgb, cmap='gray')
output =  imgrgb.copy()
alpha=0.5
for roi in S.keys():
    ridx = S[roi]['ridx']

    # Select color based on ORIENTATION preference:
    osi = S[roi]['ori'] # TODO:  deal with OSI
    col = colors[osi]
    if not roi in selective_rois:
        col = (127, 127, 127)
    msk = np.reshape(maskarray[:, ridx], dims)
    msk[msk>0] = 1
    msk = msk.astype('uint8')

    # Draw contour for ORIG rois on reference:
    ret,thresh = cv2.threshold(msk,.5,255,0)
    orig2,contours,hierarchy = cv2.findContours(thresh, 1, 2)
    cv2.drawContours(imgrgb, contours, 0, color=col, thickness=-1)
    cv2.addWeighted(imgrgb, alpha, output, 1 - alpha, 0, output)

    # Label ROI
    if label_rois:
        [ys, xs] = np.where(msk>0)
        ax.text(int(xs[0]), int(ys[-1]), str(ridx+1), fontsize=8, weight='light', color='w')
    ax.imshow(imgrgb, cmap=cmap, alpha=0.5)
pl.axis('off')

if label_rois:
    figname = 'visual_rois_orientation_labeled.png'
else:
    figname = 'visual_rois_orientation.png'

pl.savefig(os.path.join(sort_figdir, figname))
pl.close()

#%% Check tuning curves (compare stimdf vs zscore):

sorted_tuningdir_visual = os.path.join(sort_figdir, 'tuning_responsive')
if not os.path.exists(sorted_tuningdir_visual):
    os.makedirs(sorted_tuningdir_visual)

#roi = 'roi00030'

for roi in sorted_rois_visual[0:10]:
    ridx = int(roi[3:]) - 1

    # Get standard dataframe (not pyvttbl):
    rdata = DATA[DATA['roi']==roi]
    df = roidata_to_df_configs(rdata)

    #grped = df.groupby(['config']) #.mean()
    #df2_dff = pd.DataFrame({col:vals['dff'] for col,vals in grped})

    stimdf_means = df.groupby(['config'])['dff'].mean() # df2_dff.mean() #.sort_values(ascending=False)
    stimdf_sems = df.groupby(['config'])['dff'].sem() #df2_dff.sem()

    zscore_means = df.groupby(['config'])['zscore'].mean()
    zscore_sems = df.groupby(['config'])['zscore'].sem()


    sf1 = sorted([config for config in stimconfigs.keys() if stimconfigs[config]['frequency']==0.1], key=lambda x: stimconfigs[x]['rotation'])
    sf2 = sorted([config for config in stimconfigs.keys() if stimconfigs[config]['frequency']==0.5], key=lambda x: stimconfigs[x]['rotation'])
    oris = [stimconfigs[config]['rotation'] for config in sf1]


    sns.set()
    pl.figure(figsize=(10,5))
    pl.subplot(1,2,1)
    pl.plot(oris, stimdf_means[sf1], color='g', label=0.1)
    pl.errorbar(oris, stimdf_means[sf1], yerr=stimdf_sems[sf1], color='g', label=None)
    pl.plot(oris, stimdf_means[sf2], color='m', label=0.5)
    pl.errorbar(oris, stimdf_means[sf2], yerr=stimdf_sems[sf2], color='m', label=None)
    pl.xticks(oris)
    pl.ylabel('df/f')
    pl.legend()

    pl.subplot(1,2,2)
    pl.plot(oris, zscore_means[sf1], color='g', label=0.1)
    pl.errorbar(oris, zscore_means[sf1], yerr=zscore_sems[sf1], color='g', label=None)
    pl.plot(oris, zscore_means[sf2], color='m', label=0.5)
    pl.errorbar(oris, zscore_means[sf2], yerr=zscore_sems[sf2], color='m', label=None)
    pl.xticks(oris)
    pl.ylabel('zscore')
    pl.legend()

    pl.suptitle(roi)

    figname = 'tuning_%s.png' % roi
    pl.savefig(os.path.join(sorted_tuningdir_visual, figname))

    pl.close()

#%%

##% Sort configs by mean value:
#grped = df.groupby(['config']) #.mean()
#df2 = pd.DataFrame({col:vals['dff'] for col,vals in grped})
#meds = df2.median().sort_values(ascending=False)
#df2 = df2[meds.index]
#pl.figure(figsize=(10,5))
#ax = sns.boxplot(data=df2)
#pl.title(roi)
#pl.ylabel('df/f')
#ax.set_xticklabels(['%i deg\n%.2f cpd' % (stimconfigs[t.get_text()]['rotation'], stimconfigs[t.get_text()]['frequency'])
#                        for t in ax.get_xticklabels()])
#
## statistical annotation
#sigpairs = [s for s in list(pc[pc < 0.05].stack().index) if not s[0]==s[1]]
#sigpairs = [sorted(s) for s in {frozenset(x) for x in sigpairs}]
#
#ordered_meds = meds.index.tolist()
#i = 1
#sigA = sorted(list(set([s[0] for s in sigpairs])), key=natural_keys)
#spairs = {}
#for skey in sigA:
#    spairs[skey] = [s[1] for s in sigpairs if s[0]==skey]
#
#
#
#for spair in sorted(sigpairs, key=lambda x: (x[0], x[1])):
#    x1 = ordered_meds.index(spair[0]) # column indices
#    x2 = ordered_meds.index(spair[1])
#    y, h, col = df['dff'].max() + 0.05*i, 0, 'k'
#    pl.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1, c=col)
#    pl.text((x1)*.5, y+h, "*", ha='center', va='bottom', color=col)
#    i += 1
#


#%%

pdf = pyvt_stimdf_oriXsf(rdata, trans_types, save_fig=True,
                   output_dir=sort_figdir,
                   fname='DFF_%s_box(dff~epoch_X_config).png' % roi)


#pdf.interaction_plot('dff','ori',seplines='sf', yerr='ci')




#
#sub='trial',
#bfactors=['config'])
#curr_config = 'config006'

    #pdf = pyvt_raw_epochXsinglecond(rdata, curr_config=curr_config)
    #aov = pdf.anova('intensity', sub='trial', wfactors=['epoch'])
    #aov1 = pdf.anova1way('intensity', 'epoch')




#pdf = get_pyvt_dataframe_raw(rdata, trans_types, save_fig=False,
#                             output_dir=sort_figdir,
#                             fname='%s_boxplot(intensity~epoch_X_config).png' % roi)
#



#%%
def get_effect_sizes(aov):
    #factor_a='', factor_b=''):
    for factor in aov.keys():
        aov[factor]['ss']

    ssq_a = aov[(factor_a,)]['ss']
    ssq_b = aov[(factor_b,)]['ss']
    ssq_ab = aov[(factor_a,factor_b)]['ss']

    ssq_error = aov[('WITHIN',)]['ss']
    ss_total = aov[('TOTAL',)]['ss']

    etas = {}
    etas[factor_a] = ssq_a / ss_total
    etas[factor_b] = ssq_b / ss_total
    etas['%sx%s' % (factor_a, factor_b)] = ssq_ab / ss_total
    etas['error'] = ssq_error / ss_total

    return etas

#%%
def get_pyvt_dataframe(STATS, roi):
    roiSTATS = STATS[STATS['roi']==roi]
    grouped = roiSTATS.groupby(['config', 'trial']).agg({'stim_df': 'mean',
                                                         'baseline_df': 'mean'
                                                         }).dropna()

    #rstats_df = roiSTATS[['config', 'trial', 'baseline_df', 'stim_df', 'xpos', 'ypos', 'morphlevel', 'yrot', 'size']]
    rstats_df = roiSTATS[['trial', 'config', 'baseline_df', 'stim_df']].dropna()
    newtrials_names = ['trial%05d' % int(i+1) for i in rstats_df.index]
    rstats_df.loc[:, 'trial'] = newtrials_names

    tmpd = rstats_df.pivot_table(['stim_df', 'baseline_df'], ['config', 'trial']).T

    data = []
    data.append(pd.DataFrame({'epoch': np.tile('baseline', (len(tmpd.loc['baseline_df'].values),)),
                  'df': tmpd.loc['baseline_df'].values,
                  'config': [cfg[0] for cfg in tmpd.loc['baseline_df'].index.tolist()],
                  'trial': [cfg[1] for cfg in tmpd.loc['baseline_df'].index.tolist()]
                  }))
    data.append(pd.DataFrame({'epoch': np.tile('stimulus', (len(tmpd.loc['stim_df'].values),)),
                  'df': tmpd.loc['stim_df'].values,
                  'config': [cfg[0] for cfg in tmpd.loc['baseline_df'].index.tolist()],
                  #'trial': ['trial%05d' % int(i+1) for i in range(len(tmpd.loc['baseline_df'].index.tolist()))]
                  'trial': [cfg[1] for cfg in tmpd.loc['baseline_df'].index.tolist()]
                  }))
    data = pd.concat(data)

