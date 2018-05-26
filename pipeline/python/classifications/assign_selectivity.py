#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 15:10:00 2018

@author: juliana
"""

import matplotlib
matplotlib.use('agg')
import h5py
import os
import sys
import json
import cv2
import time
import math
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
import pprint
pp = pprint.PrettyPrinter(indent=4)

from pipeline.python.utils import natural_keys, replace_root, print_elapsed_time
import pipeline.python.traces.combine_runs as cb
import pipeline.python.paradigm.align_acquisition_events as acq
import pipeline.python.visualization.plot_psths_from_dataframe as vis
from pipeline.python.traces.utils import load_TID
from skimage import exposure
from collections import Counter

#from pipeline.python.classification import utils


#%%
def uint16_to_RGB(img):
    im = img.astype(np.float64)/img.max()
    im = 255 * im
    im = im.astype(np.uint8)
    rgb = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    return rgb


def find_barval_index(bar_value_to_label, p):
    min_distance = float("inf")  # initialize min_distance with infinity
    index_of_bar_to_label = 0
    for i, rectangle in enumerate(p.patches):  # iterate over every bar
        tmp = abs(  # tmp = distance from middle of the bar to bar_value_to_label
            (rectangle.get_x() +
                (rectangle.get_width() * (1 / 2))) - bar_value_to_label)
        if tmp < min_distance:  # we are searching for the bar with x cordinate
                                # closest to bar_value_to_label
            min_distance = tmp
            index_of_bar_to_label = i
    return index_of_bar_to_label



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
        datakey = datakeys[0]
        df.close()
        del df
        DATA = pd.read_hdf(roidata_filepath, datakey)
        #DATA = df[datakey]

    return DATA, datakey


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
                                     'meanstimdf': stim_mean,
                                     'zscore': stim_mean / base_std}, index=[idx]))
        idx += 2
    df = pd.concat(df_list, axis=0)
    df = df.sort_values(['config'])
    df = df.reset_index() #drop=True)

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

    trans_types = sorted(trans_types, key=natural_keys)

    df_groups = np.copy(trans_types).tolist()
    df_groups.extend(['trial', 'df'])
    currdf = rdata[df_groups] #.sort_values(trans_types)

    groupby_list = np.copy(trans_types).tolist()
    groupby_list.extend(['trial'])
    grp = currdf.groupby(groupby_list)

    idx = 0
    df_list = []
    for k,g in grp:
        #print k
        base_mean= np.nanmean(g['df'][0:first_on]) #.mean()
        base_std = np.nanmean(g['df'][0:first_on]) #.std()
        stim_mean = np.nanmean(g['df'][first_on:first_on+nframes_on]) #mean()

        tdict = {'trial': k[-1],
                 'meanstimdf': stim_mean,
                 'zscore': stim_mean / base_std}
        for dkey in range(len(k)-1):
            tdict[trans_types[dkey]] = k[dkey]

        df_list.append(pd.DataFrame(tdict, index=[idx]))

        idx += 1

    df = pd.concat(df_list, axis=0)
    df = df.sort_values(trans_types)
    df = df.reset_index() #drop=True)

    return df

#%%
def pd_to_pyvtt_transforms(df, trans_types, metric='meanstimdf'):

    '''
    Take single ROI as a datatset, N-way rmANOVA (N = num of transform types that are varying):
        between-trial factor :  trans_type(s)
    '''
    
    # Format pandas df into pyvttbl dataframe:
    #df_factors = ['config', 'trial', 'dff']
    df_factors = np.copy(trans_types).tolist()
    df_factors.extend(['trial', metric])

    Trial = namedtuple('Trial', df_factors)
    pdf = pt.DataFrame()
    for idx in xrange(df.shape[0]):
        if len(trans_types)==1:
            pdf.insert(Trial(df.loc[idx, trans_types[0]],
                             df.loc[idx, 'trial'],
                             df.loc[idx, metric])._asdict())
        elif len(trans_types)==2:
            pdf.insert(Trial(df.loc[idx, trans_types[0]],
                             df.loc[idx, trans_types[1]],
                             df.loc[idx, 'trial'],
                             df.loc[idx, metric])._asdict())
        elif len(trans_types)== 3:
            pdf.insert(Trial(df.loc[idx, trans_types[0]],
                             df.loc[idx, trans_types[1]],
                             df.loc[idx, trans_types[2]],
                             df.loc[idx, 'trial'],
                             df.loc[idx, metric])._asdict())

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

def extract_apa_anova2(factor, aov, values = ['F', 'mse', 'eta', 'p', 'df']):

    results = {}

    if not isinstance(factor, list):
        factor = [factor]

    for fac in factor:
        fmtresults = {}
        for key,result in aov[(fac)].iteritems():
            if key in values:
                fmtresults[key] = result

        fmtresults['dim'] = aov.D

        # Calculate partial-eta2:
        fmtresults['eta2_p'] = aov[(fac)]['ss'] / ( aov[(fac)]['ss'] + aov[(fac)]['sse'] )

        results[fac] = fmtresults

    if len(results.keys()) == 1:
        results = results[results.keys()[0]]

    return results

#%%
def splitplot_anova2_pyvt(roi, rdata, output_dir='/tmp', asdict=True):
#    responsive_rois = {}

    pdf = pyvt_raw_epochXconfig(rdata, save_fig=False)
    # Calculate ANOVA split-plot:
    aov = pdf.anova('intensity', sub='trial',
                       wfactors=['epoch'],
                       bfactors=['config'])
    #print(aov)

    aov_results_fpath = os.path.join(output_dir, 'visual_anova_results_%s.txt' % roi)
    with open(aov_results_fpath,'wb') as f:
        f.write(str(aov))
    f.close()
#    print aov_results_fpath

    #etas = get_effect_sizes(aov, factor_a='epoch', factor_b='config')
    results_epoch = extract_apa_anova2(('epoch',), aov)
    #res_interaction = extract_apa_anova2(('epoch', 'config'), aov)
#    if res_epoch['p'] < 0.1: # or res_interaction['p'] < 0.1:
#        responsive_rois[roi] = {'F': res_epoch['F'], 'p': res_epoch['p']} #.append(roi)

    if asdict is True:
        return results_epoch
    else:
        return results_epoch['F'], results_epoch['p']


#%%
def test_for_responsive_mp(DATA, output_dir='/tmp', nprocs=4):

    roi_list = sorted(list(set(DATA['roi'])), key=natural_keys)
    print("Calculating split-plot ANOVA (factors=epoch, config) for %i rois." % len(roi_list))

    t_eval_mp = time.time()

    def worker(rlist, DATA, output_dir, out_q):
        """
        Worker function is invoked in a process. 'roi_list' is a list of
        roi names to evaluate [rois00001, rois00002, etc.]. Results are placed
        in a dict that is pushed to a queue.
        """
        outdict = {}
        print len(rlist)
        for roi in rlist:
            #print roi
            rdata = DATA[DATA['roi']==roi]
            outdict[roi] = splitplot_anova2_pyvt(roi, rdata, output_dir=output_dir, asdict=True)
        out_q.put(outdict)

    # Each process gets "chunksize' filenames and a queue to put his out-dict into:
    out_q = mp.Queue()
    chunksize = int(math.ceil(len(roi_list) / float(nprocs)))
    procs = []

    for i in range(nprocs):
        p = mp.Process(target=worker,
                       args=(roi_list[chunksize * i:chunksize * (i + 1)],
                                       DATA,
                                       output_dir,
                                       out_q))
        procs.append(p)
        print "Starting:", p
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
def test_for_responsive(DATA, save_figs=False, output_dir='/tmp'):
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

    responsive_rois = {} #[]
    for roi in roi_list:
        #print roi
        rdata = DATA[DATA['roi']==roi]
        responsive_rois[roi] = splitplot_anova2_pyvt(roi, rdata, output_dir=output_dir, asdict=True)

    return responsive_rois



#%%


#%%
#def test_for_selectiveKW(DATA, topn=10, test_normal=False, post_hoc='dunn', save_figs=False, selective_dir='/tmp'):
#
#    ph_results = {}
#    
#    roi_list = sorted(list(set(DATA['roi'])), key=natural_keys)
#    print("Calculating KW selectivity test for %i rois." % len(roi_list))
#
#    for ridx,roi in enumerate(roi_list):
#        rdata = DATA[DATA['roi']==roi]
#
#        # Get standard dataframe (not pyvttbl):
#        df = roidata_to_df_configs(rdata)
#
##        if ridx < topn and save_figs:
##            print roi
##            #% Sort configs by mean value:
##            grped = df.groupby(['config']) #.mean()
##            df2 = pd.DataFrame({col:vals['dff'] for col,vals in grped})
##            meds = df2.median().sort_values(ascending=False)
##            df2 = df2[meds.index]
##            pl.figure(figsize=(10,5))
##            ax = sns.boxplot(data=df2)
##            pl.title(roi)
##            pl.ylabel('df/f')
##            ax.set_xticklabels(['%i deg\n%.2f cpd\n%s' % (stimconfigs[t.get_text()]['rotation'],
##                                                          stimconfigs[t.get_text()]['frequency'],
##                                                          t.get_text()) for t in ax.get_xticklabels()])
##            figname = 'box_mediandff_%s.png' % roi
##            pl.savefig(os.path.join(output_dir, figname))
##            pl.close()
#
#        normality = False
#        if test_normal:
#            k2, pn = stats.mstats.normaltest(df['dff'])
#            if pn < 0.05:
#                print("Normal test: p < 0.05, k=%.2f" % k2)
#                normality = False
#            else:
#                print("Normal test: p > 0.05, k=%.2f" % k2)
#                normality = True
#
#            # Check for normality:
#            if ridx < topn and save_figs:
#                pl.figure()
#                qq_res = stats.probplot(df['dff'], dist="norm", plot=pl)
#                pl.title('P-P plot %s' % roi)
#                pl.text(-2, 0.3, 'p=%s' % str(pn))
#                pl.show()
#                figname = 'PPplot_%s.png' % roi
#                pl.savefig(os.path.join(output_dir, figname))
#                pl.close()
#
#            # Check if STDs are equal (ANOVA):
#            #df.groupby(['config']).std()
#
#        if normality is False:
#            # Format dataframe and do KW test:
#            groupedconfigs = {}
#            for grp in df['config'].unique():
#                groupedconfigs[grp] = df[df['config']==grp]['dff'].values
#            args = groupedconfigs.values()
#            H, p = stats.kruskal(*args)
#
#            # Do post-hoc test:
#            if post_hoc == 'dunn':
#                pc = sp.posthoc_dunn(df, val_col='dff', group_col='config')
#            elif post_hoc == 'conover':
#                pc = sp.posthoc_conover(df, val_col='dff', group_col='config')
#
#            if ridx < topn and save_figs:
#                # Plot heatmap of p-values from post-hoc test:
#                pl.figure(figsize=(10,8))
#                pl.title('%s test, %s' % (post_hoc, roi))
#                cmap = ['1', '#fb6a4a',  '#08306b',  '#4292c6', '#c6dbef']
#                heatmap_args = {'cmap': cmap, 'linewidths': 0.25, 'linecolor': '0.5',
#                                'clip_on': False, 'square': True,
#                                'cbar_ax_bbox': [0.90, 0.35, 0.02, 0.3]}
#                sp.sign_plot(pc, **heatmap_args)
#                figname = 'pvalues_%s_%s.png' % (roi, post_hoc)
#                pl.savefig(os.path.join(selective_dir, figname))
#                pl.close()
#
#        # Save ROI info:
#        ph_results[roi] = {'H': H,
#                           'p': p,
#                           'post_hoc': post_hoc,
#                           'p_rank': pc}
#
#    return ph_results

#%%
    
# -----------------------------------------------------------------------------
# Test if there is a significant interaction between transformation types:
# -----------------------------------------------------------------------------
from ast import literal_eval

def find_selective_cells_ANOVA(trans_types, options, DATA, sort_dir, sorted_visual, metric='meanstimdf'):
    
    options = extract_options(options)
    multiproc = False # options.multiproc
    nprocesses = int(options.nprocesses)
    if multiproc is False:
        nprocesses = 1
        
    selective_resultsdir = os.path.join(sort_dir, 'anova_results', 'selectivity_tests')
    if not os.path.exists(selective_resultsdir):
        os.makedirs(selective_resultsdir)
    
    #transform_dict, object_transformations = vis.get_object_transforms(DATA)
    #trans_types = [trans for trans in transform_dict.keys() if len(transform_dict[trans]) > 1]
        
    selective_anova_fpath = os.path.join(sort_dir, 'selective_rois_anova_results_%s.json' % metric)
    if os.path.exists(selective_anova_fpath):
        print "Loading existing %i-way ANOVA results: %s" % (len(trans_types), selective_anova_fpath)
        with open(selective_anova_fpath, 'r') as f:
            selective_anova = json.load(f)
        if '(' in selective_anova[selective_anova.keys()[0]].keys()[0]:
            for r in selective_anova.keys():
                selective_anova[r] = {literal_eval(k):v for k, v in selective_anova[r].items()}
            
    else:
        if multiproc:
            selective_anova = test_transform_selectivity_mp(sorted_visual, DATA, trans_types,
                                                            metric=metric,
                                                            output_dir=selective_resultsdir,
                                                            nprocs=nprocesses)
        else:
            selective_anova = test_transform_selectivity(sorted_visual, DATA, trans_types,
                                                         metric=metric,
                                                         output_dir=selective_resultsdir)
        
        if isinstance(selective_anova[selective_anova.keys()[0]].keys()[0], tuple):
            for r in selective_anova.keys():
                selective_anova[r] = {str(k):v for k, v in selective_anova[r] .items()}
                
        # Save responsive roi list to disk:
        print "Saving %i-way ANOVA results to:\n%s" % (len(trans_types), selective_anova_fpath)
        with open(selective_anova_fpath, 'w') as f:
            json.dump(selective_anova, f, indent=4, sort_keys=True)

    if '(' in selective_anova[selective_anova.keys()[0]].keys()[0]:
        for r in selective_anova.keys():
            selective_anova[r] = {literal_eval(k):v for k, v in selective_anova[r].items()}
                
    if len(trans_types) > 1:    
        selective_rois_anova = [k for k in selective_anova.keys()
                                if any([selective_anova[k][t]['p'] < 0.05
                                for t in selective_anova[k].keys()]) ]
        # Sort by F-value, from biggest --> smallest:
        selective_rois_anova = sorted(selective_rois_anova, key=lambda x: max([selective_anova[x][f]['F'] for f in selective_anova[x].keys()]))[::-1]
        
    else:
        selective_rois_anova = [k for k in selective_anova.keys()
                                if selective_anova[k]['p'] < 0.05]
    
        # Sort by F-value, from biggest --> smallest:
        selective_rois_anova = sorted(selective_rois_anova, key=lambda x: selective_anova[x]['F'])[::-1]
        
    return selective_anova, selective_rois_anova


def selectivity_nwayANOVA(roi, rdata, trans_types, output_dir='/tmp', metric='meanstimdf'):

    df = roidata_to_df_transforms(rdata, trans_types) #.dropna().reset_index()
    pdf = pd_to_pyvtt_transforms(df, trans_types, metric=metric)

    # Calculate ANOVA split-plot:
    aov = pdf.anova(metric, sub='trial',
                       bfactors=trans_types)
    #print(aov)

    aov_results_fpath = os.path.join(output_dir, '%s_selectivity_%iwayanova_results_%s.txt' % (metric, len(trans_types), roi))
    with open(aov_results_fpath,'wb') as f:
        f.write(str(aov))
    f.close()

    factor_types = aov.keys()
    res_epoch = extract_apa_anova2(factor_types, aov)

    return res_epoch


def test_transform_selectivity_mp(sorted_visual, DATA, trans_types, output_dir='/tmp', nprocs=4, metric='meanstimdf'):

    print "Calculating %i-way ANOVA for %i rois (factors: %s)" % (len(trans_types), len(roi_list), str(trans_types))

    t_eval_mp = time.time()

    def worker(rlist, DATA, trans_types, output_dir, metric, out_q):
        """
        Worker function is invoked in a process. 'roi_list' is a list of
        roi names to evaluate [rois00001, rois00002, etc.]. Results are placed
        in a dict that is pushed to a queue.
        """
        print len(rlist)
        outdict = {}
        for roi in rlist:
            # print roi
            rdata = DATA[DATA['roi']==roi]
            outdict[roi] = selectivity_nwayANOVA(roi, rdata, trans_types, output_dir=output_dir, metric=metric)
        out_q.put(outdict)

    # Each process gets "chunksize' filenames and a queue to put his out-dict into:
    out_q = mp.Queue()
    chunksize = int(math.ceil(len(sorted_visual) / float(nprocs)))
    procs = []

    for i in range(nprocs):
        p = mp.Process(target=worker,
                       args=(sorted_visual[chunksize * i:chunksize * (i + 1)],
                                       DATA,
                                       trans_types,
                                       output_dir,
                                       metric,
                                       out_q))
        procs.append(p)
        print "Starting:", p
        p.start()

    # Collect all results into single results dict. We should know how many dicts to expect:
    resultdict = {}
    for i in range(nprocs):
        print "Getting results:", i
        resultdict.update(out_q.get())

    # Wait for all worker processes to finish
    for p in procs:
        print "Finished:", p
        p.join()

    print_elapsed_time(t_eval_mp)

    return resultdict


def test_transform_selectivity(sorted_visual, DATA, trans_types, metric='meanstimdf', output_dir='/tmp'):
    results = {}

    for roi in sorted_visual:
        #print roi
        rdata = DATA[DATA['roi']==roi]
        results[roi] = selectivity_nwayANOVA(roi, rdata, trans_types, metric=metric, output_dir=output_dir)

    return results

#%%
def get_OSI(DATA, roi_list, stimconfigs, metric='meanstimdf'):
    
    '''
    OSI:  
        OSI = (umax - uorth) / (umax + uorth), 
            umax = mean response to preferred orientation
            uorth = mean response to orthogonal orientation (average of both directions)
            
    cvOSI:
        1 - circular variance = sum( R(theta) * exp(2 * i * theta) ) / sum( R(theta) ),
            where R(theta) is avearged repsonse to gratings with orientation theta
        
        Note: This takes into account both tuning width and depth of modulation.
            
    '''

    selectivity = {}
    for ri, roi in enumerate(roi_list):
        ridx = int(roi[3:]) - 1
        #print roi
        
        if ri % 20 == 0:
            print "Calculating OSI for %i of %i rois..." % (ri, len(roi_list))

        rdata = DATA[DATA['roi']==roi]

        # Get standard dataframe (not pyvttbl):
        df = roidata_to_df_configs(rdata)

        stimdf_means = df.groupby(['config'])[metric].mean()
        stim_values = pd.DataFrame({'angle': [stimconfigs[c]['rotation'] for c in stimdf_means.index.tolist()]}, index=stimdf_means.index)
        stimdf = pd.concat([stimdf_means, stim_values], axis=1)
    
        ordered_configs = stimdf_means.sort_values(ascending=False).index
        Rmax = stimdf_means[ordered_configs[0]]
        max_ori = stimdf.loc[ordered_configs[0], 'angle']
        if 225 >= max_ori >= 90:
            orth_ori1 = max_ori + 90
            orth_ori2 = orth_ori1 - 180
        else:
            orth_ori1 = max_ori + 90 # 360 - max_ori + 90
            orth_ori2 = orth_ori1 + 180#orth_ori1 + 180
        if orth_ori1 >= 360: orth_ori1 -= 360
        if orth_ori2 >= 360: orth_ori2 -= 360

        Rorthog = ( float(stimdf[stimdf['angle']==orth_ori1][metric]) + float(stimdf[stimdf['angle']==orth_ori2][metric]) ) / 2.0
        #Rleast = stimdf_means[ordered_configs[-1]]
        #OSI = (Rmost - Rleast) / (Rmost + Rleast)
        OSI_orth = (Rmax - Rorthog) / (Rmax + Rorthog)


        # If > 1 SF, use best one:
        sfs = list(set([stimconfigs[config]['frequency'] for config in stimconfigs.keys()]))
        if len(sfs) > 1:
            sort_config_types = {}
            for sf in sfs:
                sort_config_types[sf] = sorted([config for config in stimconfigs.keys()
                                                    if stimconfigs[config]['frequency']==sf],
                                                    key=lambda x: stimconfigs[x]['rotation'])
            oris = [stimconfigs[config]['rotation'] for config in sort_config_types[sf]]
            orientation_list = sort_config_types[stimconfigs[ordered_configs[0]]['frequency']]
            OSI_cv = np.abs( sum([stimdf_means[cfg]*np.exp(2j*theta) for theta, cfg in zip(oris, orientation_list)]) / sum([stimdf_means[cfg] for cfg in  orientation_list]) )
        else:
            OSI_cv = np.abs( sum([stimdf.loc[cfg, metric]*np.exp(2j*stimdf.loc[cfg, 'angle']) for cfg in stimconfigs.keys()]) / sum([stimdf_means[cfg] for cfg in stimconfigs.keys()]) )

        selectivity[roi] = {'ridx': ridx,
                            'value_type': metric,
                            'OSI_orth': OSI_orth,
                            'OSI_cv': OSI_cv,
                            'pref_ori': max_ori}

    return selectivity


def assign_OSI(stimconfigs, DATA, sorted_visual, sort_dir, metric='meanstimdf'):
    
    osi_fpath = os.path.join(sort_dir, 'OSI_%s.json' % metric)
    if os.path.exists(osi_fpath):
        print "Loading OSI results from:\n%s" % (osi_fpath)
        with open(osi_fpath, 'r') as f:
            OSI = json.load(f)
    else:
        OSI = get_OSI(DATA, sorted_visual, stimconfigs, metric=metric)
        
        print "Saving OSI results to:\n%s" % (osi_fpath)
        with open(osi_fpath, 'w') as f:
            json.dump(OSI, f)
            
    return OSI

    
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

    if 'sf' in trans_types:
        sf_tidx = trans_types.index('sf')
        ref_sf = Cmax_overall[sf_tidx]
    else:
        if 'sf' in transform_dict.keys():
            ref_sf = transform_dict['sf'][0]

    if 'yrot' in trans_types:
        yrot_tidx = trans_types.index('yrot')
        ref_yrot = Cmax_overall[yrot_tidx]
    else:
        if 'yrot' in transform_dict.keys():
            ref_yrot = transform_dict['yrot'][0]
            
    Cref = {'xpos': ref_xpos,
            'ypos': ref_ypos,
            'size': ref_size}

    if 'sf' in trans_types:
        Cref['sf'] = ref_sf
        
    if 'yrot' in trans_types:
        Cref['yrot'] = ref_yrot

    return Cref

#%% SELECTIVITY -- calculate a sparseness measure:

def assign_sparseness(transform_dict, options, DATA, sort_dir, sorted_visual, 
                          metric='meanstimdf', plot=True):
    
    trans_types = [trans for trans in transform_dict.keys() if len(transform_dict[trans]) > 1]
    options = extract_options(options)
    create_new = options.create_new
    multiproc = options.multiproc
    if multiproc is False:
        nprocesses = 1
    else:
        nprocesses = int(options.nprocesses)
        
    new_sorting = False
    sparseness_fpath = os.path.join(sort_dir, 'sparseness_index_results.npz')
    if os.path.exists(sparseness_fpath) and create_new is False:
        print "Loading previously calculated SPARSENESS index..."
        sparseness = np.load(sparseness_fpath)
        sparseness = {key:sparseness[key].item() for key in sparseness}
        if len(sparseness)==1 or 'roi' not in sparseness.keys()[0]:
            sparseness = sparseness[sparseness.keys()[0]]
    else:
        new_sorting = True
        print "*** Assigning SPARSENESS index to each ROI..."
        sparseness = assign_sparseness_index_mp(sorted_visual, DATA, trans_types, transform_dict, metric=metric, nprocs=nprocesses)

        # Save results to file:
        np.savez(sparseness_fpath, sparseness)
        print "Saved SPARSENESS value assignments to:\n%s" % sparseness_fpath
        # Note, to reload as dict:  ph_results = {key:ph_results[key].item() for key in ph_results}
    
    if new_sorting or plot:
        hist_spareness_index(sparseness, metric=metric, sort_dir=sort_dir, save_and_close=True)
        normalize_rank_ordered_responses(sparseness, sorted_visual, sort_dir=sort_dir, save_and_close=True)
        
    return sparseness


# Case:  Transformations in xpos, ypos
# Take as "reference" position, the x- and y-position eliciting the max response

def calc_sparseness(df, trans_types, transform_dict, metric='meanstimdf'):

    stimdf_means = df.groupby(trans_types)[metric].mean()
    ordered_configs = stimdf_means.sort_values(ascending=False).index
    if isinstance(ordered_configs, pd.MultiIndex):
        ordered_configs = ordered_configs.tolist()
    Cmax_overall = ordered_configs[0]

    Cref = get_reference_config(Cmax_overall, trans_types, transform_dict)

    # Get responses to each "OBJECT" at the reference transform values (i.e., transform values at BEST object response):
    object_resp_df = stimdf_means.copy()
    for best_trans in Cref.keys():
        if best_trans in object_resp_df.keys().names:
            object_resp_df = object_resp_df.xs(Cref[best_trans], level=best_trans)  
#    if 'xpos' in trans_types:
#        object_resp_df = object_resp_df.xs(Cref['xpos'], level='xpos')
#    if 'ypos' in trans_types:
#        object_resp_df = object_resp_df.xs(Cref['ypos'], level='ypos')
#    if 'size' in trans_types:
#        object_resp_df = object_resp_df.xs(Cref['size'], level='size')
#    if 'sf' in trans_types:
#        object_resp_df = object_resp_df.xs(Cref['sf'], level='sf')

    # TODO:  what to do if stim_df values are negative??
    if all(object_resp_df.values < 0):
        S = 0
    else:
        object_list = object_resp_df.index.tolist()
        nobjects = len(object_list)
        t1a = (sum([(object_resp_df[i] / nobjects) for i in object_list])**2)
        t1b = sum([object_resp_df[i]**2/nobjects for i in object_list])
        S = (1 - (t1a / t1b)) / (1-(1/nobjects))

    sparseness_ref = {'S': S, 'object_responses': object_resp_df}

    return sparseness_ref

#%
def assign_sparseness_index_mp(roi_list, DATA, trans_types, transform_dict, metric='meanstimdf', nprocs=4):

    print("Calculating SPARSENESS index for %i rois." % len(roi_list))

    t_eval_mp = time.time()

    def worker(roi_list, DATA, trans_types, transform_dict, metric, out_q):
        """
        Worker function is invoked in a process. 'roi_list' is a list of
        roi names to evaluate [rois00001, rois00002, etc.]. Results are placed
        in a dict that is pushed to a queue.
        """
        print len(roi_list)
        outdict = {}
        for roi in roi_list:
            #print roi
            rdata = DATA[DATA['roi']==roi]
            df = roidata_to_df_transforms(rdata, trans_types)
            outdict[roi] = calc_sparseness(df, trans_types, transform_dict, metric=metric)
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
                                       metric,
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

def hist_spareness_index(sparseness, metric='meanstimdf', sort_dir='/tmp', 
                             save_and_close=True,
                             lowS_color='forestgreen', 
                             highS_color='mediumvioletred'):

    # Histogram of sparseness:
    # --------------------------------------------    
    sparseness_values = [sparseness[r]['S'] for r in sparseness.keys()] #S.values()
#    highS_color = 'mediumvioletred'
#    lowS_color = 'forestgreen'
    sns.set_style("white")
    pl.figure()
    sns.distplot(sparseness_values, bins=50, kde=False, rug=True, color="#34495e")
    pl.xlabel('Sparseness (S)')
    pl.ylabel('Cell counts')
    s_bounds = np.linspace(0, 1, num=4, endpoint=True) # Divide range into 3 bins
    pl.axvline(x=s_bounds[1], linewidth=2, linestyle='--', color=lowS_color)
    pl.axvline(x=s_bounds[2], linewidth=2, linestyle='--', color=highS_color) #'r')
    pl.xlim([0, 1])
    figname = 'hist_sparseness_%s.png' % metric
    
    if save_and_close:
        pl.savefig(os.path.join(sort_dir, 'figures', figname))
        pl.close()
    
    return figname
    
def normalize_rank_ordered_responses(sparseness, sorted_visual, 
                                         metric='meanstimdf',
                                         sort_dir='/tmp', 
                                         save_and_close=True, 
                                         lowS_color='forestgreen', 
                                         highS_color='mediumvioletred'):
    
    # For all ROIs in FOV, plot normalized firing rate (normalized to MAX) as a
    # function of (ranked) objects:
    excluded_rois = []
    pl.figure()
    for r in sparseness.keys():
        normed_dff = sparseness[r]['object_responses'].values / sparseness[r]['object_responses'].max()
        #print r, normed_dff.min()
        if any(normed_dff < -1) or any(normed_dff > 1):
            excluded_rois.append(r)
            continue
        pl.plot(xrange(len(normed_dff)), sorted(normed_dff)[::-1], 'k', alpha=0.5, linewidth=0.5)
    pl.xticks(xrange(len(normed_dff)))
    pl.xlabel('Ranked objects')
    pl.ylabel('Normalized df/f')
    pl.gca().set_xticklabels(np.arange(1, len(normed_dff)+1))
    
    # Choose 2 examples showing low-sparseness and high-sparseness
    kept_rois = [r for r in sparseness.keys() if r not in excluded_rois]
    print "%i (out of %i) ROIs excluded because of negative stim_df values..." % (len(excluded_rois), len(sparseness.keys()))
    
    strong_rois = [r for r in sorted_visual if sparseness[r]['object_responses'].max() >= 0.2]
    sparseness_vals_topn = [sparseness[r]['S'] for r in strong_rois]
    highS_ex = [r for r in kept_rois if sparseness[r]['S'] == max(sparseness_vals_topn)][0]
    lowS_ex = [r for r in kept_rois if sparseness[r]['S'] == min(sparseness_vals_topn)][0]
    
    #sparseness_vals_topn = [sparseness[r]['S'] for r in kept_rois]
    #highS_ex = [r for r in kept_rois if sparseness[r]['S'] >=0.8 ][0]
    #lowS_ex = [r for r in kept_rois if sparseness[r]['S'] <= 0.1 ][0]
    
    sorted_lowS_vals = sorted(sparseness[lowS_ex]['object_responses'].values / sparseness[lowS_ex]['object_responses'].max())[::-1]
    sorted_highS_vals = sorted(sparseness[highS_ex]['object_responses'].values / sparseness[highS_ex]['object_responses'].max())[::-1]
    
    pl.plot(xrange(len(normed_dff)), sorted_lowS_vals, color=lowS_color, linestyle='--', linewidth=2)
    pl.plot(xrange(len(normed_dff)), sorted_highS_vals, color=highS_color, linestyle='--', linewidth=2)
    pl.text(0., -0.1, "high S: %.2f (%s)" % (sparseness[highS_ex]['S'], highS_ex), color=highS_color)
    pl.text(0, -0.3, "low S: %.2f (%s)" % (sparseness[lowS_ex]['S'], lowS_ex), color=lowS_color)
    
    pl.title('Normalized %s across ranked objects' % metric)
    figname = 'ranked_normed_dff_%s.png' % metric
    
    if save_and_close:
        pl.savefig(os.path.join(sort_dir, 'figures', figname))
        pl.close()

    return figname


#%%
def get_dataset(options, verbose=True):
    INFO = {}
    
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
    #pupil_max_nblinks = 0
    
    #multiproc = options.multiproc
    #nprocesses = int(options.nprocesses)
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
        runfolder = '_'.join(run_list)
        if len(traceid_list)==1:
            traceid = '_'.join([traceid_list[0] for i in range(nruns)])
        traceid_dir = os.path.join(rootdir, animalid, session, acquisition, runfolder, traceid)
    
    print(traceid_dir)
    assert os.path.exists(traceid_dir), "Specified traceid-dir does not exist!"


    #% # Load ROIDATA file:
    print "Loading ROIDATA file..."
    roidf_fn = [i for i in os.listdir(traceid_dir) if i.endswith('hdf5') and 'ROIDATA' in i and trace_type in i][0]
    roidata_filepath = os.path.join(traceid_dir, roidf_fn) #'ROIDATA_098054_626d01_raw.hdf5')
    DATA, datakey = load_roi_dataframe(roidata_filepath)
    

    #% Set filter params:
    if filter_pupil is True:
        pupil_params = acq.set_pupil_params(radius_min=pupil_radius_min,
                                            radius_max=pupil_radius_max,
                                            dist_thr=pupil_dist_thr,
                                            create_empty=False)
    elif filter_pupil is False:
        pupil_params = acq.set_pupil_params(create_empty=True)

    # =============================================================================
    # Extract data summary info:
    # =============================================================================

    assert len(list(set(DATA['first_on'])))==1, "More than 1 frame idx found for stimulus ON"
    assert len(list(set(DATA['nframes_on'])))==1, "More than 1 value found for nframes on."

    stim_on_frame = int(list(set(DATA['first_on']))[0])
    nframes_on = int(round(list(set(DATA['nframes_on']))[0]))
    
    # CHeck for np.nans:
    nan_ixs = np.where(np.isnan(DATA['df'].values))[0]
    #trials_to_fix = list(set(DATA.loc[nan_ixs, 'trial']))
    rois_to_fix = list(set(DATA.loc[nan_ixs, 'roi']))
    for roi in rois_to_fix:
        rdata = DATA[DATA['roi']==roi]
        ixs = np.where(np.isnan(rdata['df'].values))[0]
        dixs = rdata.index[ixs]
        if len(ixs) > 0:
            trials_to_fix = list(set(rdata.loc[dixs, 'trial']))
            for trial in trials_to_fix:
                print "re-calc df/f: %s, %s" % (roi, trial)
                raw = rdata[rdata['trial']==trial]['raw']
                bas = raw[0:stim_on_frame].mean()
                dft = (raw - bas) / bas
                tixs = rdata[rdata['trial']==trial].index
                DATA.loc[tixs, 'df'] = dft
                
    configs = list(set(DATA['config']))
    transform_dict, object_transformations = vis.get_object_transforms(DATA)

    roi_list = sorted(list(set(DATA['roi'])), key=natural_keys)
    nrois = len(roi_list)

    nframes_per_trial = len(DATA[DATA['trial']=='trial00001']['tsec']) / nrois
    ntrials_per_stim = [len(list(set(DATA[DATA['config']==config]['trial']))) for config in configs] # Assumes all stim have same # trials!
    if len(list(set(ntrials_per_stim))) == 1:
        ntrials_per_stim = ntrials_per_stim[0]
    ntrials_total = len(list(set(DATA['trial'])))

    if verbose:
        print "-------------------------------------------"
        print "Run summary:"
        print "-------------------------------------------"
        print "N rois:", len(roi_list)
        print "N trials:", ntrials_total
        print "N frames per trial:", nframes_per_trial
        print "N trials per stimulus:", ntrials_per_stim
        print "-------------------------------------------"

    INFO['nrois'] = len(roi_list)
    INFO['ntrials_total'] = ntrials_total
    INFO['nframes_per_trial'] = nframes_per_trial
    INFO['ntrials_per_cond'] = ntrials_per_stim
    INFO['condition_list'] = configs
    INFO['stim_on_frame'] = stim_on_frame
    INFO['nframes_on'] = nframes_on
    INFO['traceid_dir'] = traceid_dir
    INFO['trace_type'] = trace_type
    INFO['transforms'] = object_transformations
    INFO['combined_dataset'] = combined
    INFO['nruns'] = nruns


    return DATA, datakey, INFO, pupil_params

#%%
def extract_options(options):

    parser = optparse.OptionParser()

    parser.add_option('-D', '--root', action='store', dest='rootdir',
                          default='/nas/volume1/2photon/data',
                          help='data root dir (dir containing all animalids) [default: /nas/volume1/2photon/data, /n/coxfs01/2pdata if --slurm]')
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
    parser.add_option('--new', action="store_true",
                      dest="create_new", default=False, help="Set flag to create new output files (/paradigm/parsed_frames.hdf5, roi_trials.hdf5")
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

#%% Load ROI DATA:

#    S = {1 − [(ΣRi/n)2/Σ(Ri2/n)]}/[1 − (1/n)],
#
#    MT = [n − (ΣRi/Rmax)]/(n − 1),
#    ST = 〈Rtest size/max(Rtest size)〉



#options = ['-D', '/mnt/odyssey', '-i', 'CE077', '-S', '20180413', '-A', 'FOV1_zoom1x',
#           '-T', 'np_subtracted', '--no-pupil',
#           '-R', 'gratings_run1', '-t', 'traces001',
#           '-n', '1']
#
#options = ['-D', '/mnt/odyssey', '-i', 'CE074', '-S', '20180215', '-A', 'FOV1_zoom1x_V1',
#           '-T', 'np_subtracted', '--no-pupil',
#           '-R', 'gratings_phasemod', '-t', 'traces005',
#           '-n', '1']

#options = ['-D', '/mnt/odyssey', '-i', 'CE077', '-S', '20180423', '-A', 'FOV2_zoom1x',
#           '-T', 'np_subtracted', '--no-pupil',
#           '-R', 'blobs_run1', '-t', 'traces001',
#           '-n', '1']


#options = ['-D', '/mnt/odyssey', '-i', 'CE077', '-S', '20180425', '-A', 'FOV1_zoom1x',
#           '-T', 'np_subtracted', '--no-pupil',
#           '-R', 'gratings_run2', '-t', 'traces001',
#           '-n', '1']

#options = ['-D', '/mnt/odyssey', '-i', 'CE077', '-S', '20180425', '-A', 'FOV1_zoom1x',
#           '-T', 'np_subtracted', '--no-pupil',
#           '-R', 'blobs_run1', '-t', 'traces002',
#           '-n', '1']

#options = ['-D', '/mnt/odyssey', '-i', 'CE077', '-S', '20180515', '-A', 'FOV1_zoom1x',
#           '-T', 'np_subtracted', '--no-pupil',
#           '-R', 'blobs_run3', '-t', 'traces001',
#           '-n', '1', '--par', '--nproc=4']
    
options = ['-D', '/mnt/odyssey', '-i', 'CE077', '-S', '20180518', '-A', 'FOV1_zoom1x',
       '-T', 'np_subtracted', '--no-pupil',
       '-R', 'blobs_dynamic_run3', '-t', 'traces002',
       '-n', '1', '--par', '--nproc=4']

#%% 

# -----------------------------------------------------------------------------
# Functions for identifying and testing VISUAL RESPONSIVENESS of cells:
# -----------------------------------------------------------------------------

def find_visual_cells(options, DATA, sort_dir, plot=True):
    
    responsive_resultsdir = os.path.join(sort_dir, 'anova_results', 'responsive_tests')
    if not os.path.exists(responsive_resultsdir):
        os.makedirs(responsive_resultsdir)
        
    # Kind of redundant, but re-extract options to get some other info:
    options = extract_options(options)
    multiproc = options.multiproc
    nprocesses = int(options.nprocesses)
    create_new = options.create_new
    
    # Load responsive rois, if exist:
    new_sorting = False
    if create_new:
        new_sorting = True
    responsive_anova_fpath = os.path.join(sort_dir, 'visual_rois_anova_results.json')
    if os.path.exists(responsive_anova_fpath) and new_sorting is False:
        print "Loading existing split ANOVA results:\n", responsive_anova_fpath
        with open(responsive_anova_fpath, 'r') as f:
            responsive_anova = json.load(f)
    else:
        new_sorting = True
        
        if multiproc is True:
            responsive_anova = test_for_responsive_mp(DATA, output_dir=responsive_resultsdir, nprocs=nprocesses)
        else:
            responsive_anova = test_for_responsive(DATA, output_dir=responsive_resultsdir)
    
        # Save responsive roi list to disk:
        print "Saving split ANOVA results to:\n", responsive_anova_fpath
        with open(responsive_anova_fpath, 'w') as f:
            json.dump(responsive_anova, f, indent=4, sort_keys=True)

    # Sort ROIs:
    responsive_rois = [r for r in responsive_anova.keys() if responsive_anova[r]['p'] < 0.05]
    sorted_visual = sorted(responsive_rois, key=lambda x: responsive_anova[x]['F'])[::-1]
    
    if new_sorting and plot:
        boxplots_visual_cells(DATA, responsive_anova, sorted_visual, topn=10, sort_dir=sort_dir)
            
    return responsive_anova, sorted_visual


def boxplots_visual_cells(DATA, responsive_anova, sorted_visual, topn=10, sort_dir='/tmp'):
    
    # Box plots for top N rois (sortedb y ANOVA F-value):
    print("Plotting box plots for factors EPOCH x CONFIG for top %i rois." % topn)
    vis_responsive_dir = os.path.join(sort_dir, 'figures', 'responsive')
    if not os.path.exists(vis_responsive_dir):
        os.makedirs(vis_responsive_dir)
    
    boxplot_epochXconfig(DATA, sorted_visual[0:topn], output_dir=vis_responsive_dir)

    # View histogram of partial eta-squared values:
    eta2p_vals = [responsive_anova[r]['eta2_p'] for r in responsive_anova.keys()]
    eta2p = pd.Series(eta2p_vals)
    
    pl.figure()
    nrois = len(responsive_anova)
    count, division = np.histogram(eta2p, bins=nrois/5)
    p = eta2p.hist(bins=division, color='gray')
    
    # Highlight p-eta2 vals for significant neurons:
    sig_etas = [responsive_anova[r]['eta'] for r in responsive_anova.keys() if responsive_anova[r]['p'] < 0.05]
    sig_bins = list(set([binval for ix,binval in enumerate(division) for etaval in sig_etas if division[ix] < etaval <= division[ix+1]]))
    
    indices_to_label = [find_barval_index(v, p) for v in sig_bins]
    for ind in indices_to_label:
        p.patches[ind].set_color('m')
    
    pl.xlabel('eta-squared')
    pl.title("Partial eta-squared (split-plot ANOVA2, p<0.05)")
    figname = 'vis_responsive_eta2_partial.png'
    pl.savefig(os.path.join(os.path.split(vis_responsive_dir)[0], figname))
    
    pl.close()
    
#%
def boxplot_epochXconfig(DATA, roi_list, output_dir='/tmp'):

    for roi in roi_list:
        rdata = DATA[DATA['roi']==roi]
        pdf = pyvt_raw_epochXconfig(rdata, save_fig=False)
        factor_list = ['config', 'epoch']
        fname = '%s_boxplot(intensity~epoch_X_config).png' % roi
        pdf.box_plot('intensity', factors=factor_list, fname=fname, output_dir=output_dir)

#%%
def find_selective_cells_KW(options, DATA, sort_dir, sorted_visual, post_hoc='dunn', metric='meanstimdf', stimlabels={}, plot=True):
    
    #post_hoc = 'dunn'
    options = extract_options(options)
    multiproc = options.multiproc
    nprocesses = int(options.nprocesses)
    create_new = options.create_new
    
    new_sorting = False
    posthoc_fpath = os.path.join(sort_dir, 'selectivity_KW_posthoc_%s.npz' % post_hoc)
    if os.path.exists(posthoc_fpath) and create_new is False:
        print "Loading previously calculated SELECTIVITY test results (KW, post-hoc: %s)" % post_hoc
        selectivityKW_results = np.load(posthoc_fpath)
        selectivityKW_results = {key:selectivityKW_results[key].item() for key in selectivityKW_results}
        if len(selectivityKW_results)==1 or 'roi' not in selectivityKW_results.keys()[0]:
            selectivityKW_results = selectivityKW_results[selectivityKW_results.keys()[0]]
    else:
        new_sorting = True
        if multiproc is False and nprocesses > 1:
            nprocesses = 1
        selectivityKW_results = test_for_selectiveKW_mp(sorted_visual, DATA, metric=metric, nprocs=nprocesses)

        # Save results to file:
        posthoc_fpath = os.path.join(sort_dir, 'selectivity_KW_posthoc_%s.npz' % post_hoc)
        np.savez(posthoc_fpath, selectivityKW_results)
        print "Saved SELECTIVITY test results (KW, post-hoc: %s) to: %s" % (post_hoc, posthoc_fpath)
        # Note, to reload as dict:  ph_results = {key:ph_results[key].item() for key in ph_results}

    # Get ROIs that pass KW test:
    selective_rois = [r for r in sorted_visual if selectivityKW_results[r]['p'] < 0.05]
    sorted_selective = sorted(selective_rois, key=lambda x: selectivityKW_results[x]['H'])[::-1]
   
    if new_sorting or plot:
        # Plot p-values and median-ranked box plots for top N rois:
        # ---------------------------------------------------------
        for roi in sorted_selective[0:10]:
            rdata = DATA[DATA['roi']==roi]
            boxplot_roi_selectivity(rdata, sort_dir, metric='meanstimdf', stimlabels=stimlabels)
            heatmap_roi_stimulus_pval(selectivityKW_results[roi], roi, sort_dir)
        
    
    return selectivityKW_results, sorted_selective


#%
def test_for_selectiveKW_mp(roi_list, DATA, metric='meanstimdf', nprocs=4):

    #roi_list = sorted(list(set(DATA['roi'])), key=natural_keys)
    print("Calculating KW selectivity test for %i rois." % len(roi_list))

    t_eval_mp = time.time()

    def worker(roi_list, DATA, metric, out_q):
        """
        Worker function is invoked in a process. 'roi_list' is a list of
        roi names to evaluate [rois00001, rois00002, etc.]. Results are placed
        in a dict that is pushed to a queue.
        """
        outdict = {}
        for roi in roi_list:
            #print roi
            rdata = DATA[DATA['roi']==roi]
            outdict[roi] = selectivity_KW(rdata, post_hoc='dunn', metric=metric, asdict=True)
        out_q.put(outdict)

    # Each process gets "chunksize' filenames and a queue to put his out-dict into:
    out_q = mp.Queue()
    chunksize = int(math.ceil(len(roi_list) / float(nprocs)))
    procs = []

    for i in range(nprocs):
        p = mp.Process(target=worker,
                       args=(roi_list[chunksize * i:chunksize * (i + 1)],
                                       DATA,
                                       metric,
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


def selectivity_KW(rdata, post_hoc='dunn', metric='meanstimdf', asdict=True):

    # Get standard dataframe (not pyvttbl):
    df = roidata_to_df_configs(rdata)

    # Format dataframe and do KW test:
    groupedconfigs = {}
    for grp in df['config'].unique():
        print grp
        groupedconfigs[grp] = df[df['config']==grp][metric].values
    args = groupedconfigs.values()
    H, p = stats.kruskal(*args)

    # Do post-hoc test:
    if post_hoc == 'dunn':
        pc = sp.posthoc_dunn(df, val_col=metric, group_col='config')
    elif post_hoc == 'conover':
        pc = sp.posthoc_conover(df, val_col=metric, group_col='config')

    # Save ROI info:
    posthoc_results = {'H': H,
                       'p': p,
                       'posthoc_test': post_hoc,
                       'p_rank': pc}
    if asdict is True:
        return posthoc_results
    else:
        return posthoc_results['H'], posthoc_results['p'], pc

#%
    
def boxplot_roi_selectivity(rdata, sort_dir, metric='meanstimdf', stimlabels={}):

    selective_dir = os.path.join(sort_dir, 'figures', 'selectivity')
    if not os.path.exists(selective_dir):
        os.makedirs(selective_dir)
    
    df = roidata_to_df_configs(rdata)
    roi = list(set(rdata['roi']))[0]

    #% Sort configs by mean value:
    grped = df.groupby(['config']) #.mean()
    df2 = pd.DataFrame({col:vals[metric] for col,vals in grped})
    meds = df2.median().sort_values(ascending=False)
    df2 = df2[meds.index]
    pl.figure(figsize=(10,5))
    ax = sns.boxplot(data=df2)
    pl.title(roi)
    pl.ylabel('df/f')
#    ax.set_xticklabels(['%i deg\n%.2f cpd\n%s' % (stimconfigs[t.get_text()]['rotation'],
#                                                  stimconfigs[t.get_text()]['frequency'],
#                                                  t.get_text()) for t in ax.get_xticklabels()])
    if len(stimlabels.keys()) > 0:
        if isinstance(stimlabels[stimlabels.keys()[0]], int):
            ax.set_xticklabels(['%i deg\n%.2f cpd\n%s' % (stimlabels[t.get_text()],
                                                  stimlabels[t.get_text()],
                                                  t.get_text()) for t in ax.get_xticklabels()],
                                rotation=45)
        else:
            ax.set_xticklabels(['%s\n%s' % (stimlabels[t.get_text()],
                                      t.get_text()) for t in ax.get_xticklabels()],
                                rotation=45)
    
    figname = 'box_mediandff_%s.png' % roi
    pl.savefig(os.path.join(selective_dir, figname))
    pl.close()

#
def heatmap_roi_stimulus_pval(posthoc_results, roi, sort_dir):

    selective_dir = os.path.join(sort_dir, 'figures', 'selectivity')
    if not os.path.exists(selective_dir):
        os.makedirs(selective_dir)
        
    pc = posthoc_results['p_rank']
    posthoc_test = posthoc_results['posthoc_test']
    
    # Plot heatmap of p-values from post-hoc test:
    pl.figure(figsize=(10,8))
    pl.title('%s test, %s' % (posthoc_test, roi))
    cmap = ['1', '#fb6a4a',  '#08306b',  '#4292c6', '#c6dbef']
    heatmap_args = {'cmap': cmap, 'linewidths': 0.25, 'linecolor': '0.5',
                    'clip_on': False, 'square': True,
                    'cbar_ax_bbox': [0.90, 0.35, 0.02, 0.3]}
    sp.sign_plot(pc, **heatmap_args)
    figname = 'pvalues_%s_%s.png' % (roi, posthoc_test)
    pl.savefig(os.path.join(selective_dir, figname))
    pl.close()
                
    #%
    

   #%% 
def color_rois_by_OSI(img, maskarray, OSI, stimconfigs, sorted_selective=[], cmap='hls',
                          sort_dir='/tmp', metric='meanstimdf', save_and_close=True):
    
    dims = img.shape
    if len(sorted_selective) == 0:
        sorted_selective = [r for r in OSI.keys() if OSI['OSI_orth'] >= 0.33]
        color_code = 'all'
    else:
        color_code = 'OS'
        
    configs = sorted(stimconfigs.keys(), key=lambda x: stimconfigs[x]['rotation'])
    colorvals = sns.color_palette(cmap, len(configs))
    colors = dict()
    for ci, config in enumerate(sorted([stimconfigs[k]['rotation'] for k in stimconfigs.keys()])):
        print config
        colors[config] = tuple(c*255 for c in colorvals[ci])

    
    label_rois = True
    imgrgb = uint16_to_RGB(img)
    zproj = exposure.equalize_adapthist(imgrgb, clip_limit=0.001)
    zproj *= 256
    zproj= zproj.astype('uint8')
        
    # Plot average img, overlay ROIs with OSI (visual only):
    sns.set_style('white')
    fig = pl.figure()
    ax = fig.add_subplot(111)
    pl.imshow(zproj, cmap='gray')
    outimg = zproj.copy()
    alpha=0.8
    for roi in OSI.keys():
        ridx = int(roi[3:])-1 #S[roi]['ridx']
    
        # Select color based on ORIENTATION preference:
        osi = OSI[roi]['pref_ori'] # TODO:  deal with OSI
        #print osi
        col = colors[osi]
        if not roi in sorted_selective:
            col = (127, 127, 127)
        msk = np.reshape(maskarray[:, ridx], dims)
        msk[msk>0] = 1
        msk[msk==0] = np.nan
        msk = msk.astype('uint8')
    
        # Draw contour for ORIG rois on reference:
        ret,thresh = cv2.threshold(msk,.5,1,0)
        orig2,contours,hierarchy = cv2.findContours(thresh, 1, 2)
        cv2.drawContours(outimg, contours, 0, color=col, thickness=-1)
        cv2.addWeighted(outimg, alpha, outimg, 1 - alpha, 0, outimg)
    
        # Label ROI
        if label_rois and roi in sorted_selective and OSI[roi]['pref_ori'] >= 0.33:
            [ys, xs] = np.where(msk>0)
            ax.text(int(xs[0]), int(ys[-1]), str(ridx+1), fontsize=8, weight='light', color='w')
        ax.imshow(outimg, alpha=0.5)
    pl.axis('off')
    
    if label_rois:
        figname = 'visual_rois_OSI_%s_%s_labeled.png' %  (metric, color_code)
    else:
        figname = 'visual_rois_OSI_%s_%s.png' % (metric, color_code)
        
    if save_and_close:
        pl.savefig(os.path.join(sort_dir, 'figures', figname))
        pl.close()
        
    return figname
    


def hist_preferred_oris(OSI, colorvals, metric='meanstimdf', sort_dir='/tmp', save_and_close=True):
    
    best_ori_vals = [OSI[r]['pref_ori'] for r in OSI.keys()]
    best_ori_vals_selective = [OSI[r]['pref_ori'] for r in OSI.keys() if OSI[r]['OSI_orth'] >=0.33]
    ori_counts_all = Counter(best_ori_vals)
    ori_counts_selective = Counter(best_ori_vals_selective)
    for ori in ori_counts_all.keys():
        if ori not in ori_counts_selective.keys():
            ori_counts_selective[ori] = 0
   
    bar_palette = colorvals.as_hex()
    fig, ax = pl.subplots()
    ax2 = ax.twinx()
    sns.barplot(sorted(ori_counts_all.keys()), [ori_counts_all[c] for c in sorted(ori_counts_all.keys())], palette=bar_palette, ax=ax)
    sns.barplot(sorted(ori_counts_all.keys()), [ori_counts_selective[c] for c in sorted(ori_counts_all.keys())], palette=bar_palette, ax=ax2)
    ax2.set_ylim(ax.get_ylim())
    hatch = '//' #itertools.cycle(['/', '//', '+', '-', 'x', '\\', '*', 'o', 'O', '.'])
    for i, bar in enumerate(ax2.patches):
        bar.set_hatch(hatch)
    ax2.set_yticklabels([])
    figname = 'counts_per_cond_OSI_%s.png' % metric
    if save_and_close:
        pl.savefig(os.path.join(sort_dir, 'figures', figname))
        pl.close()
    
    # legend:
    sns.palplot(colorvals)
    if save_and_close:
        pl.savefig(os.path.join(sort_dir, 'figures', 'legend_OSI.png'))
        pl.close()
    
    return figname

def format_stimulus_transforms(stimconfigs):
    # Split position into x,y:
    for config in stimconfigs.keys():
        stimconfigs[config]['xpos'] = stimconfigs[config]['position'][0]
        stimconfigs[config]['ypos'] = stimconfigs[config]['position'][1]
        stimconfigs[config]['size'] = stimconfigs[config]['scale'][0]
        stimconfigs[config].pop('position', None)
        stimconfigs[config].pop('scale', None)
        if 'frequency' in stimconfigs[config].keys():
            stimconfigs[config]['sf'] = stimconfigs[config]['frequency']
            stimconfigs[config].pop('frequency', None)
        if 'rotation' in stimconfigs[config].keys():
            stimconfigs[config]['ori'] = stimconfigs[config]['rotation']
            stimconfigs[config].pop('rotation', None)
        if 'filename' in stimconfigs[config].keys():
            fn = os.path.splitext(stimconfigs[config]['filename'])[0] # remove extension
            if '_CamRot_y' in fn:
                stimconfigs[config]['object'] = fn.split('_CamRot_y')[0]
                stimconfigs[config]['yrot'] = int(fn.split('_CamRot_y')[-1])
            else: # RW objects are formatted differently:
                stimconfigs[config]['object'] = fn.split('_')[0]
                stimconfigs[config]['yrot'] = int(fn.split('zRot')[-1])
                
            stimconfigs[config].pop('filename', None)        
            stimconfigs[config].pop('filepath', None)        
            if 'morph' in stimconfigs[config]['object']:
                stimconfigs[config]['morphlevel'] = int(stimconfigs[config]['object'].split('morph')[-1])
            elif '_N1' in stimconfigs[config]['object']:
                stimconfigs[config]['morphlevel'] = 0
            elif '_N2' in stimconfigs[config]['object']:
                stimconfigs[config]['morphlevel'] = 22
            else: # RW object
                stimconfigs[config]['morphlevel'] = -1
    return stimconfigs

def compare_meandf_zscore(stimconfigs, transform_dict, DATA, sorted_visual, sorted_selective, sort_dir, metric='meanstimdf'):
    print "------ Creating tuning curves from mean stim ON values and zscore values"
    
    sorted_tuningdir_visual = os.path.join(sort_dir, 'figures', 'tuning_responsive')
    if not os.path.exists(sorted_tuningdir_visual):
        os.makedirs(sorted_tuningdir_visual)
    
    #roi = 'roi00030'
    id_trans = ['xpos', 'ypos', 'yrot', 'size']
    id_preserving_transforms = [t for t in transform_dict.keys() if t in id_trans and len(transform_dict[t]) > 1]
    
    # Format MW stimulus config labels for easy access:
    sconfigs = format_stimulus_transforms(stimconfigs)
    
    # Turn into DF with row indices as 'config' to concat with mean/sem dfs:
    config_df = pd.DataFrame(stimconfigs).T.reset_index()
    config_df = config_df.rename(columns={'index': 'config'}).sort_values('config')
    print config_df.head()
    sns.set_palette(sns.color_palette("hls", len(sconfigs)))
    
    # Only plot top 10 visual/selective ROIs
    rois_to_plot = sorted_selective[0:10]
    rois_to_plot.extend(sorted_visual[0:10])
    rois_to_plot = list(set(rois_to_plot))
    
    for roi in rois_to_plot:
        ridx = int(roi[3:]) - 1
    
        # Get standard dataframe (not pyvttbl):
        rdata = DATA[DATA['roi']==roi]
        df = roidata_to_df_configs(rdata)
    
        #grped = df.groupby(['config']) #.mean()
        #df2_dff = pd.DataFrame({col:vals['dff'] for col,vals in grped})
        df.config = df.config.astype("category")
        #df.config.cat.set_categories(INFO['condition_list'], inplace=True)
        df = df.sort_values(['config'])
        
        stimdf_means = df.groupby(['config'])['meanstimdf'].mean().reset_index().rename(columns={'meanstimdf': 'mean'}) 
        stimdf_sems  = df.groupby(['config'])['meanstimdf'].sem().reset_index().rename(columns={'meanstimdf': 'sem'}) 
        stimdf = pd.merge(stimdf_means, stimdf_sems, on='config')
        stimdf = pd.merge(stimdf, config_df, on='config')
    
        zscore_means = df.groupby(['config'])['zscore'].mean().reset_index().rename(columns={'zscore': 'mean'}) 
        zscore_sems = df.groupby(['config'])['zscore'].sem().reset_index().rename(columns={'zscore': 'sem'}) 
        zscoredf = pd.merge(zscore_means, zscore_sems, on='config')
        zscoredf = pd.merge(zscoredf, config_df, on='config')
    
        trans_list = []
        print id_preserving_transforms
        print stimdf.head()
        #if len(id_preserving_transforms) == 1:
        if len(id_preserving_transforms)==0 and 'sf' in config_df.columns:
            id_preserving_transforms = ['ori']
	# For gratings, this is ORI. For objects, this can be Yrot, Xpos, etc.
	config_idx = dict((g[0], g[1]['config'].values) for g in config_df.groupby(id_preserving_transforms))
        print config_idx
	if 'object' in config_df.columns:
	    objects_to_color = list(set(config_df.sort_values(['morphlevel'])['object']))
	elif 'sf' in config_df.columns:
	    objects_to_color = list(set(config_df['sf']))
        print "COLORS:", objects_to_color
 
        labels = objects_to_color
        trans_values = sorted(config_idx.keys())
        colors = ['g', 'm', 'b', 'orange', 'purple']
#            sf2 = sorted([config for config in stimconfigs.keys() if stimconfigs[config]['frequency']==0.5], key=lambda x: stimconfigs[x]['rotation'])
#            oris = [stimconfigs[config]['rotation'] for config in sf1]
#            sfs = [sf1, sf2]
#            colors = ['g', 'm']
#            labels = [0.1, 0.5]
#        else:
#            oris = sorted([stimconfigs[config]['rotation'] for config in stimconfigs])
#    
        sns.set_style('white')

        fig, (ax1, ax2) = pl.subplots(1, 2, figsize=(12,5))
        
        trans = id_preserving_transforms[0]
        for idx, obj in enumerate(objects_to_color):
            if 'object' in stimdf.columns:
                mdf = stimdf[stimdf['object']==obj]
                zdf = zscoredf[zscoredf['object']==obj]
            elif 'sf' in stimdf.columns:
                mdf = stimdf[stimdf['sf']==obj]
                zdf = zscoredf[zscoredf['sf']==obj]
            ax1.plot(trans_values, mdf.sort_values(trans)['mean'].values, label=obj, color=colors[idx])
            ax1.errorbar(trans_values, mdf.sort_values(trans)['mean'].values, \
                             yerr=mdf.sort_values(trans)['sem'].values, label=None, color=colors[idx])
            ax1.set(xticks=trans_values)
            ax1.set(ylabel='df/f')
            ax1.set(xlabel=trans)
            #ax1.legend(loc='best')
                
            ax2.plot(trans_values, zdf.sort_values(trans)['mean'].values, label=obj, color=colors[idx])
            ax2.errorbar(trans_values, zdf.sort_values(trans)['mean'].values, \
                             yerr=zdf.sort_values(trans)['sem'].values, label=None, color=colors[idx])
            ax2.set(xticks=trans_values)
            ax2.set(ylabel='zscore')
            ax2.set(xlabel=trans)
            ax2.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
        sns.despine(offset=4, trim=True)
            
        pl.suptitle(roi)
        figname = 'tuning_%s_%s.png' % (metric, roi)
        pl.savefig(os.path.join(sorted_tuningdir_visual, figname))
    
        pl.close()

#%%
    
def load_masks_and_img(options, INFO):
    
    options = extract_options(options)
    rootdir = options.rootdir
    animalid = options.animalid
    session = options.session
    acquisition = options.acquisition
    
    # Identify reference file and load MASKS:
    session_dir = os.path.join(rootdir, animalid, session)
    traceid_dir = INFO['traceid_dir']
    trace_id = os.path.split(traceid_dir)[-1].split('_')[0]
    run_dir = traceid_dir.split('/traces')[0]
    
    if INFO['combined_dataset'] is True:
        tmp_runfolder = '_'.join([run_dir.split('_')[0], run_dir.split('_')[1]])
        tmp_rundir = os.path.join(session_dir, acquisition, tmp_runfolder)
        TID = load_TID(tmp_rundir, trace_id)
        tmp_traceid_dir = TID['DST']
    else:
        TID = load_TID(run_dir, trace_id)
        tmp_traceid_dir = traceid_dir
    
    with open(os.path.join(session_dir, 'ROIs', 'rids_%s.json' % session), 'r') as f:
        rdict = json.load(f)
    
    RID = rdict[TID['PARAMS']['roi_id']]
    ref_file = 'File%03d' % int(RID['PARAMS']['options']['ref_file'])
    
    if rootdir not in tmp_traceid_dir:
        tmp_traceid_dir = replace_root(tmp_traceid_dir, rootdir, animalid, session)
    
    maskpath = os.path.join(tmp_traceid_dir, 'MASKS.hdf5')
    masks = h5py.File(maskpath, 'r')
    
    # Plot on MEAN img:
    if ref_file not in masks.keys():
        ref_file = masks.keys()[0]
    img_src = masks[ref_file]['Slice01']['zproj'].attrs['source']
    if 'std' in img_src:
        img_src = img_src.replace('std', 'mean')
    if rootdir not in img_src:
        img_src = replace_root(img_src, rootdir, animalid, session)
    
    img = tf.imread(img_src)
    maskarray = masks[ref_file]['Slice01']['maskarray']
    
    return maskarray, img


#%%
def plot_roi_tolerance(transform_dict, DATA, sorted_visual, sort_dir, metric='meanstimdf'):
    
    tolerance_figdir = os.path.join(sort_dir, 'figures', 'tolerance')
    if not os.path.exists(tolerance_figdir):
        os.makedirs(tolerance_figdir)
    
#    def average_resp_transforms(group, objectid, transform, metric_type='dff'):
#        grp = group.groupby(transform) # All entries belong to a single object ID (from original groupby)
#        xvals = grp.mean().index.get_level_values(transform).unique().tolist() # These are the varying transform values
#        assert len(list(set(np.hstack(grp[objectid].apply(np.array))))) == 1, "More than 1 object id found!"
#        obj_name = grp[objectid].unique().tolist()[0][0] # There should only be a single object ID
#        print(obj_name)
#        global ax, count
#        if count > 0:
#            #ax.plot(xvals, grp.dff.mean(), label=obj_name)
#            ax.errorbar(xvals, grp[metric_type].mean(), yerr=grp[metric_type].sem(), label=obj_name)
#        count += 1
#        return group
    
    trans_types = [trans for trans in transform_dict.keys() if len(transform_dict[trans]) > 1]
    id_trans = ['xpos', 'ypos', 'yrot', 'size']
    id_preserving_transforms = [t for t in transform_dict.keys() if t in id_trans and len(transform_dict[t]) > 1]
            
    # Select an ID-preserving transform to view:
    transform = id_preserving_transforms[0] #'xpos'
    
    print "Plotting TOLERANCE to %s..." % transform
                
    for roi in sorted_visual:
        rdata = DATA[DATA['roi']==roi]
        df = roidata_to_df_transforms(rdata, trans_types)
    
        stimdf_means = df.groupby(trans_types)[metric].mean()
        ordered_configs = stimdf_means.sort_values(ascending=False).index
        if isinstance(ordered_configs, pd.MultiIndex):
            ordered_configs = ordered_configs.tolist()
        Cmax_overall = ordered_configs[0]
    
        # Get transform values using the stimulus config with the BEST response:
        Cref = get_reference_config(Cmax_overall, trans_types, transform_dict)
        
        # Assign the "reference" value for current transform type based on 
        # the stim-config w/ best reference:
        Tref = Cref[transform]
        
        # Assign the other values of the current transform type as "flankers"
        Tflankers = [tval for tval in transform_dict[transform] if not tval==Tref]
    
        # Determine if there are any other varying transform types that we
        # are keeping constant while viewing the current transform:
        const_trans_types = [trans for trans in Cref.keys() if not trans==transform and trans in df.keys()]
    
        objectid = 'morphlevel' #'ori' # 'morphlevel' #'ori'
        if len(const_trans_types) == 0:
            transdf = df.copy()
        else:
            transdf = df[[i for i in df[c]==Cref[c] for c in const_trans_types]]
        
#        means = transdf.groupby([objectid, transform]).mean()
#        xvals = means.index.get_level_values(transform).unique()
#        count = 0
        #transdf.groupby([objectid]).apply(average_resp_transforms, objectid, transform, metric_type=metric)
        means = transdf.groupby([objectid, transform])[metric].mean().reset_index().rename(columns={metric: 'means'})
        sems = transdf.groupby([objectid, transform])[metric].sem().reset_index().rename(columns={metric: 'sems'})
        tdf = pd.merge(means, sems)
        xvals = sorted(list(set(tdf[transform])))

        fig, ax = pl.subplots(figsize=(8,6))   
        grped_means = tdf.groupby([objectid])
        for k,g in grped_means:
            g.plot(x=transform, y='means', yerr='sems', kind='line', ax=ax, label=k)
        
        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        
        # Put a legend to the right of the current axis
        legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.2), title=objectid, fontsize=10)
        pl.setp(legend.get_title(), fontsize='10')

        # Customize axes:
        pl.xticks(xvals)
        pl.xlabel(transform)
        if metric=='meanstimdf':
            ylabel = 'df/f'
            ax.set(ylim=[0, .5])
            ax.set(yticks=[0, 0.20])
            ylabel_pos = 0.25
        else:
            ylabel = 'zscore'
            ax.set(ylim=[-1, 4])
            ax.set(yticks=[0, 4])
            ylabel_pos = 0.5
        pl.ylabel(ylabel, verticalalignment='bottom')
        ax.get_yaxis().set_label_coords(-0.1, ylabel_pos)
        sns.despine(offset=8, trim=True)
        
        pl.title(roi)
        figname = '%s_%s_%s_across_%s.png' % (metric, roi, objectid, transform)
        pl.savefig(os.path.join(tolerance_figdir, figname))
        pl.close()

#%%

def calculate_selectivity_measures(options):
    # =============================================================================
    # LOAD everything.
    # =============================================================================
    
    DATA, datakey, INFO, pupil_params = get_dataset(options)
    
    #% Get stimulus config info:assign_roi_selectivity
    # =============================================================================
    traceid_dir = INFO['traceid_dir']
    run_dir = traceid_dir.split('/traces')[0]
    
    if INFO['combined_dataset'] is True:
        stimconfigs_fpath = os.path.join(traceid_dir, 'stimulus_configs.json')
    else:
        stimconfigs_fpath = os.path.join(run_dir, 'paradigm', 'stimulus_configs.json')
    with open(stimconfigs_fpath, 'r') as f:
        stimconfigs = json.load(f)
    print "Loaded %i stimulus configurations." % len(stimconfigs.keys())
    
    #% TODO:  Set up stimulus-specific info about conditions here:
    if 'gratings' in traceid_dir:
        configs = sorted([k for k in stimconfigs.keys()], key=lambda x: stimconfigs[x]['rotation'])
        INFO['condition_list'] = configs
        #orientations = [stimconfigs[c]['rotation'] for c in configs]
    
    # =========================================================================
    # Create output dir for ROI selection:
    # =========================================================================
    print "Creating OUTPUT DIRS for ROI analyses..."
    if '/' in datakey:
        datakey = datakey[1:]
    sort_dir = os.path.join(traceid_dir, 'sorted_%s' % datakey)
    if not os.path.exists(sort_dir):
        os.makedirs(sort_dir)

    # Create output text file for quick-viewing of summary info:
    # -------------------------------------------------------------------------
    summary_fpath = os.path.join(sort_dir, 'roi_summary.txt')
    with open(summary_fpath, 'w') as f:
        pprint.pprint(INFO, f)

    # =========================================================================
    # Identify visually repsonsive cells:
    # =========================================================================

    # Sort ROIs by F ratio:
    responsive_anova, sorted_visual = find_visual_cells(options, DATA, sort_dir, plot=True)
    roi_list = sorted(list(set(DATA['roi'])), key=natural_keys)
    print("%i out of %i cells pass split-plot ANOVA test for visual responses." % (len(sorted_visual), len(responsive_anova)))

    # =========================================================================
    # Identify selective cells -- KRUSKAL WALLIS
    # =========================================================================
    post_hoc = 'dunn'
    metric = 'meanstimdf'
    # Rename stim labels so that "config" is replaced w/ sth informative...
    if 'frequency' in stimconfigs[stimconfigs.keys()[0]].keys():
        stimtype = 'gratings'
        stimlabels = dict((c, stimconfigs[c]['rotation']) for c in stimconfigs.keys())
    else:
        # each config is some combination of various transforms:
        stim_fnames = [stimconfigs[c]['filename'] for c in stimconfigs.keys()]
        if stim_fnames[0].endswith('png') and any(['morph' in f for f in stim_fnames]):
            stimtype = 'blobs'
        elif 'movie' in stim_fnames[0]:
            stimtype = 'movie'
            
        if len(stim_fnames) == len(stimconfigs): 
            # Prob only YROTATION tested:
            stimlabels = dict((c, os.path.splitext(stimconfigs[c]['filename'])[0]) for c in stimconfigs.keys())
        
    selectivityKW_results, sorted_selective = find_selective_cells_KW(options, DATA, sort_dir, sorted_visual, 
                                                                           post_hoc=post_hoc, metric=metric,
                                                                           stimlabels=stimlabels, plot=True)

    # Update roi stats summary file:
    # ---------------------------------------------------------
    H_mean = np.mean([selectivityKW_results[r]['H'] for r in selectivityKW_results.keys()])
    H_std = np.std([selectivityKW_results[r]['H'] for r in selectivityKW_results.keys()])
    with open(summary_fpath, 'a') as f:
        pprint.pprint(INFO, f)
        print >> f, '**********************************************************************'
        print >> f, '%i out of %i cells are visually responsive (split-plot ANOVA, p < 0.05)' % (len(sorted_visual), len(roi_list))
        print >> f, '%i out of %i visual are stimulus selective (Kruskal-Wallis, p < 0.05)' % (len(sorted_selective), len(sorted_visual))
        print >> f, 'Mean H=%.2f (std=%.2f)' % (H_mean, H_std)
        print >> f, '**********************************************************************'
        print >> f, 'VISUAL -- Top 10 neuron ids: %s' % str([int(r[3:]) for r in sorted_visual[0:10]])
        print >> f, 'SELECTIVE -- Top 10 neuron ids: %s' % str([int(r[3:]) for r in sorted_selective[0:10]])
    
    #%
    # =========================================================================
    # Look for sig diffs between SHAPE vs. TRANSFORM 
    # This test is an N-way ANOVA, where N = transformations that are tested.
    # =========================================================================
    transform_dict, object_transformations = vis.get_object_transforms(DATA)
    trans_types = [trans for trans in transform_dict.keys() if len(transform_dict[trans]) > 1]
    selective_anova, selective_rois_anova = find_selective_cells_ANOVA(trans_types, options, DATA, sort_dir, sorted_visual)
    
    with open(summary_fpath, 'a') as f:
        print >> f, '---- Test effects of transforms (%i-way ANOVA: %s) ---------------' % (len(trans_types), str(trans_types))
        print >> f, ".... Found %i out of %i visual ROIs with a significant effect of 1 or more factors (p < 0.05)" % (len(selective_rois_anova), len(sorted_visual))
        print >> f, '.... Top 10 neuron ids: %s' % str([int(r[3:]) for r in selective_rois_anova[0:10]])    
    
    #%
    # =========================================================================
    # Assign SELECTIVITY / SPARSENESS.
    # =========================================================================
    # SI = 0 (no selectivity) --> 1 (high selectivity)
    # 1. Split trials in half, assign shapes to which neuron shows highest/lowest activity
    # 2. Use other half to measure activity to assigned shapes and calculate SI:
    #    SI = (Rmost - Rleast) / (Rmost + Rleast)
    
    #SI = (Rmost - Rleast) / (Rmost + Rleast)
    
    sparseness = assign_sparseness(transform_dict, options, DATA, sort_dir, sorted_visual, metric=metric, plot=True)

    # =========================================================================
    # Visualize FOV with color-coded ROIs
    # =========================================================================
    maskarray, img = load_masks_and_img(options, INFO)


    if stimtype == 'gratings':
        # =====================================================================
        # Assign OSI for all (visual) ROIs in FOV:
        # =====================================================================
        #% Best orientation
        # TODO:  calculate standard OSI / DSI instead?
        OSI = assign_OSI(stimconfigs, DATA, sorted_visual, sort_dir, metric=metric)
            
        cmap='hls'
        color_rois_by_OSI(img, maskarray, OSI, stimconfigs, cmap=cmap,
                              sorted_selective=sorted_selective, 
                              sort_dir=sort_dir, 
                              metric=metric, save_and_close=True)
        
        colorvals = sns.color_palette(cmap, len(configs))
        hist_preferred_oris(OSI, colorvals, metric='meanstimdf', sort_dir=sort_dir, save_and_close=True)
        
    elif stimtype == 'blobs':
        # =====================================================================
        # Assign TOLERANCE.
        # =====================================================================
        plot_roi_tolerance(transform_dict, DATA, sorted_visual, sort_dir, metric=metric)


    # =========================================================================
    # Check tuning curves (compare stimdf vs zscore):
    # =========================================================================
    compare_meandf_zscore(stimconfigs, transform_dict, DATA, sorted_visual, sorted_selective, sort_dir, metric=metric)
    
    print "SELECTIVITY:  @@@DONE@@@"
    
    return sort_dir


    #%
def main(options):
    
    sort_dir = calculate_selectivity_measures(options)
    
    print "*******************************************************************"
    print "DONE!"
    print "All output saved to: %s" % sort_dir
    print "*******************************************************************"
    
    
#%%

if __name__ == '__main__':
    main(sys.argv[1:])


#%% 
    
# =============================================================================
# STATS FOR SELECTIVITY vs TOLERANCE:
# =============================================================================
    
#from statsmodels.stats.libqsturng import psturng
#from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison
#
##roi = 'roi00006'
#roi = 'roi00008'
#rdata = DATA[DATA['roi']==roi]
##df = roidata_to_df_configs(rdata)
#df = roidata_to_df_transforms(rdata, trans_types)
#
##aov = df.anova('dff', sub='trial', bfactors=trans_types)
##res_epoch = extract_apa_anova2(('ori',), aov)
#
#curr_trans = 'morphlevel'
#res2 = pairwise_tukeyhsd(df['dff'], df[curr_trans])
#
##res2 = pairwise_tukeyhsd(df['dff'], df['config']) # df['xpos'], df['ypos'])
#print(res2)
#
##mod = MultiComparison(df['dff'], df['ori'])
##mod = MultiComparison(df['dff'], df['config'])
#mod = MultiComparison(df['dff'], df['xpos'])
#print(mod.tukeyhsd())
#
#
#st_range = np.abs(res2.meandiffs) / res2.std_pairs
#pvalues = psturng(st_range, len(res2.groupsunique), res2.df_total)
#
#group_labels = res2.groupsunique
#
## plot:
#res2.plot_simultaneous()
#
#pl.figure()
#pl.plot(xrange(len(res2.meandiffs)), np.abs(res2.meandiffs), 'o')
#pl.errorbar(xrange(len(res2.meandiffs)), np.abs(res2.meandiffs), yerr=np.abs(res2.std_pairs.T - np.abs(res2.meandiffs)))
#xlim = -0.5, len(res2.meandiffs)-0.5
#pl.hlines(0, *xlim)
#pl.xlim(*xlim)
#pair_labels = res2.groupsunique[np.column_stack(mod.pairindices)] #[1][0])]
#pl.xticks(xrange(len(res2.meandiffs)), pair_labels)
#pl.title('Multiple Comparison of Means - Tukey HSD, FWER=0.05' +
#          '\n Pairwise Mean Differences')
#
#
## Pairwise t-tests:
## TukeyHSD uses joint variance across all samples
## Pairwise ttest calculates joint variance estimate for each pair of samples separately
#
## Paired samples:  Assumes that samples are paired (i.e., each subject goes through all X treatments)
#
#rtp = mod.allpairtest(stats.ttest_rel, method='Holm')
#print(rtp[0])
#rtp_bon = mod.allpairtest(stats.ttest_rel, method='b')
#print(rtp_bon[0])
#
## Indep samples:  Assumes that samples are paired (i.e., each subject goes through all X treatments)
## Pairwise ttest calculates joint variance estimate for each pair of samples separately
## ... but stats.ttest_ind looks at one pair at a time --
## So, calculate a pairwise ttest that takes a joint variance as given and feed
## it to mod.allpairtest?
#itp_bon = mod.allpairtest(stats.ttest_ind, method='b')
#print(itp_bon[0])
#
## TukeyHSD returns mean and variance for studentized range statistic
## studentized ranged statistic is t-statistic except scaled by np.sqrt(2)
##t_stat = res2[1][2] / res2[1][3] / np.sqrt(2)
#t_stat = np.abs(res2.meandiffs) / res2.std_pairs / np.sqrt(2)
#print(t_stat)
#my_pvalues = stats.t.sf(np.abs(t_stat), res2.df_total) * 2   #two-sided
#print(my_pvalues) # Uncorrected p-values (same as R)
#
## Do multiple-testing p-value correction (Bonferroni):
#from statsmodels.stats.multitest import multipletests
#res_b = multipletests(my_pvalues, method='b')
#
## False discvoery rate correction:
#res_fdr = multipletests(my_pvalues, method='fdr_bh')


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

#pdf = pyvt_stimdf_oriXsf(rdata, trans_types, save_fig=True,
#                   output_dir=sort_figdir,
#                   fname='DFF_%s_box(dff~epoch_X_config).png' % roi)


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



##%%
#def get_effect_sizes(aov):
#    #factor_a='', factor_b=''):
#    for factor in aov.keys():
#        aov[factor]['ss']
#
#    ssq_a = aov[(factor_a,)]['ss']
#    ssq_b = aov[(factor_b,)]['ss']
#    ssq_ab = aov[(factor_a,factor_b)]['ss']
#
#    ssq_error = aov[('WITHIN',)]['ss']
#    ss_total = aov[('TOTAL',)]['ss']
#
#    etas = {}
#    etas[factor_a] = ssq_a / ss_total
#    etas[factor_b] = ssq_b / ss_total
#    etas['%sx%s' % (factor_a, factor_b)] = ssq_ab / ss_total
#    etas['error'] = ssq_error / ss_total
#
#    return etas
#
##%%
#def get_pyvt_dataframe(STATS, roi):
#    roiSTATS = STATS[STATS['roi']==roi]
#    grouped = roiSTATS.groupby(['config', 'trial']).agg({'stim_df': 'mean',
#                                                         'baseline_df': 'mean'
#                                                         }).dropna()
#
#    #rstats_df = roiSTATS[['config', 'trial', 'baseline_df', 'stim_df', 'xpos', 'ypos', 'morphlevel', 'yrot', 'size']]
#    rstats_df = roiSTATS[['trial', 'config', 'baseline_df', 'stim_df']].dropna()
#    newtrials_names = ['trial%05d' % int(i+1) for i in rstats_df.index]
#    rstats_df.loc[:, 'trial'] = newtrials_names
#
#    tmpd = rstats_df.pivot_table(['stim_df', 'baseline_df'], ['config', 'trial']).T
#
#    data = []
#    data.append(pd.DataFrame({'epoch': np.tile('baseline', (len(tmpd.loc['baseline_df'].values),)),
#                  'df': tmpd.loc['baseline_df'].values,
#                  'config': [cfg[0] for cfg in tmpd.loc['baseline_df'].index.tolist()],
#                  'trial': [cfg[1] for cfg in tmpd.loc['baseline_df'].index.tolist()]
#                  }))
#    data.append(pd.DataFrame({'epoch': np.tile('stimulus', (len(tmpd.loc['stim_df'].values),)),
#                  'df': tmpd.loc['stim_df'].values,
#                  'config': [cfg[0] for cfg in tmpd.loc['baseline_df'].index.tolist()],
#                  #'trial': ['trial%05d' % int(i+1) for i in range(len(tmpd.loc['baseline_df'].index.tolist()))]
#                  'trial': [cfg[1] for cfg in tmpd.loc['baseline_df'].index.tolist()]
#                  }))
#    data = pd.concat(data)
#
