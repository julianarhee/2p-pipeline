#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 12:19:15 2018

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
import imutils
import datetime

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

from pipeline.python.paradigm import tifs_to_data_arrays as fmt
from pipeline.python.paradigm import utils as util

from pipeline.python.utils import natural_keys, replace_root, print_elapsed_time, uint16_to_RGB, label_figure
import pipeline.python.traces.combine_runs as cb
import pipeline.python.paradigm.align_acquisition_events as acq
import pipeline.python.visualization.plot_psths_from_dataframe as vis
from pipeline.python.traces.utils import load_TID
from skimage import exposure
from collections import Counter


options = ['-D', '/mnt/odyssey', '-i', 'CE077', '-S', '20180817', '-A', 'FOV2_zoom1x',
           '-d', 'corrected',
           '-R', 'gratings_drifting', '-t', 'traces001',
           ]


def extract_options(options):

    parser = optparse.OptionParser()

    parser.add_option('-D', '--root', action='store', dest='rootdir',
                          default='/nas/volume1/2photon/data',
                          help='data root dir (dir w/ all animalids) [default: /nas/volume1/2photon/data, /n/coxfs01/2pdata if --slurm]')
    parser.add_option('--slurm', action='store_true', dest='slurm', default=False, help="set if running as SLURM job on Odyssey")

    parser.add_option('-i', '--animalid', action='store', dest='animalid',
                          default='', help='Animal ID')

    # Set specific session/run for current animal:
    parser.add_option('-S', '--session', action='store', dest='session',
                          default='', help='session dir (format: YYYMMDD_ANIMALID')
    parser.add_option('-A', '--acq', action='store', dest='acquisition',
                          default='FOV1', help="acquisition folder (ex: 'FOV1_zoom3x') [default: FOV1]")
    parser.add_option('-d', '--trace-type', action='store', dest='trace_type',
                          default='corrected', help="trace type [default: 'corrected']")
#    parser.add_option('-R', '--run', dest='run_list', default=[], nargs=1,
#                          action='append',
#                          help="run ID in order of runs")
#    parser.add_option('-t', '--traceid', dest='traceid_list', default=[], nargs=1,
#                          action='append',
#                          help="trace ID in order of runs")
    parser.add_option('-R', '--run', dest='run', default='', action='store', help="run name")
    parser.add_option('-t', '--traceid', dest='traceid', default=None, action='store', help="datestr YYYYMMDD_HH_mm_SS")

    parser.add_option('--par', action='store_true', dest='multiproc', default=False, help="set if want to run MP on roi stats, when possible")
    parser.add_option('--nproc', action='store', dest='nprocesses', default=4, help="N processes if running in par (default=4)")
    parser.add_option('--new', action='store_true', dest='create_new', default=False, help="set to run anew")

    (options, args) = parser.parse_args(options)

    return options

#%% f_by_rois.get_group(int(roi))

def find_selective_cells(roidata, labels_df, roi_list=[], sort_dir='/tmp', 
                         metric='meanstim', stimlabels={}, data_identifier='',
                         nprocs=4, post_hoc='dunn', create_new=False):
    
    pvalue = 0.05
    
    posthoc_fpath = os.path.join(sort_dir, 'selectivity_KW_posthoc_%s.npz' % post_hoc)
    if create_new is False:
        try:
            print "Loading previously calculated SELECTIVITY test results (KW, post-hoc: %s)" % post_hoc
            selectivityKW_results = np.load(posthoc_fpath)
            selectivityKW_results = {key:selectivityKW_results[key].item() for key in selectivityKW_results}
            if len(selectivityKW_results)==1 or 'roi' not in selectivityKW_results.keys()[0]:
                selectivityKW_results = selectivityKW_results[selectivityKW_results.keys()[0]]
        except Exception as e:
            print "Unable to find previous results... Creating new."
            create_new = True

    if metric == 'meanstim':
        df_by_rois = group_roidata_stimresponse(roidata, labels_df)
    
    if create_new:
        selectivityKW_results = selectivity_KW(df_by_rois, roi_list=roi_list, 
                                               post_hoc=post_hoc, nprocs=nprocs,
                                               metric=metric)

        # Save results to file:
        posthoc_fpath = os.path.join(sort_dir, 'selectivity_KW_posthoc_%s.npz' % post_hoc)
        np.savez(posthoc_fpath, selectivityKW_results)
        print "Saved SELECTIVITY test results (KW, post-hoc: %s) to: %s" % (post_hoc, posthoc_fpath)
        # Note, to reload as dict:  ph_results = {key:ph_results[key].item() for key in ph_results}
        
    # Get ROIs that pass KW test:
    if len(roi_list) == 0:
        roi_list = selectivityKW_results.keys()
    selective_rois = [r for r in roi_list if selectivityKW_results[r]['p'] < pvalue]
    sorted_selective = sorted(selective_rois, key=lambda x: selectivityKW_results[x]['H'])[::-1]
   
    if create_new:
        summary_fpath = os.path.join(sort_dir, 'roi_summary.txt')
        top10 = ['roi%05d' % int(r+1) for r in sorted_selective[0:10]]
        with open(summary_fpath, 'a') as f:
            f.write('----------------------------------------------------------\n')
            f.write('Kruskal-Wallis test for selectivity:\n')
            f.write('----------------------------------------------------------\n')
            f.write('%i out of %i pass selectivity test (p < %.2f).\n' % (len(selective_rois), len(roi_list), pvalue))
            f.write('Top 10 (sorted by H val):\n    %s' % str(top10))
            
            
        # Plot p-values and median-ranked box plots for top N rois:
        # ---------------------------------------------------------
        boxplots_selectivity(df_by_rois, sorted_selective, metric=metric, stimlabels=stimlabels, topn=10, sort_dir=sort_dir, data_identifier=data_identifier)
        
            #heatmap_roi_stimulus_pval(selectivityKW_results[roi], roi, sort_dir)
        
    
    return selectivityKW_results, sorted_selective


    
def selectivity_KW(df_by_rois, roi_list=[], post_hoc='dunn', metric='meanstim', nprocs=4):
    
    if len(roi_list) == 0:
        roi_list = df_by_rois.groups.keys()
    print("Calculating KW selectivity test for %i rois." % len(roi_list))

    t_eval_mp = time.time()
    def worker(rlist, df_by_rois, out_q, post_hoc, metric):
        """
        Worker function is invoked in a process. 'roi_list' is a list of
        roi names to evaluate [rois00001, rois00002, etc.]. Results are placed
        in a dict that is pushed to a queue.
        """
        outdict = {}
        for roi in rlist:
            # Format pandas df into pyvttbl dataframe:
            rdata = df_by_rois.get_group(int(roi))
            outdict[roi] = do_KW_test(rdata, post_hoc=post_hoc, metric=metric, asdict=True)  
        out_q.put(outdict)


    # Each process gets "chunksize' filenames and a queue to put his out-dict into:
    out_q = mp.Queue()
    chunksize = int(math.ceil(len(roi_list) / float(nprocs)))
    procs = []

    for i in range(nprocs):
        p = mp.Process(target=worker,
                       args=(roi_list[chunksize * i:chunksize * (i + 1)],
                                       df_by_rois,
                                       out_q,
                                       post_hoc,
                                       metric))
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


def do_KW_test(rdata, post_hoc='dunn', metric='meanstim', asdict=True):

    # Format dataframe and do KW test:
    values_by_cond = list(rdata.groupby('config')[metric].apply(np.array))
    H, p = stats.kruskal(*values_by_cond)

    # Do post-hoc test:
    if post_hoc == 'dunn':
        pc = sp.posthoc_dunn(rdata, val_col=metric, group_col='config')
    elif post_hoc == 'conover':
        pc = sp.posthoc_conover(rdata, val_col=metric, group_col='config')

    # Save ROI info:
    posthoc_results = {'H': H,
                       'p': p,
                       'posthoc_test': post_hoc,
                       'p_rank': pc}
    if asdict is True:
        return posthoc_results
    else:
        return posthoc_results['H'], posthoc_results['p'], pc



def group_roidata_stimresponse(roidata, labels_df):
    
    try:
        stimdur_vary = False
        assert len(labels_df['nframes_on'].unique())==1, "More than 1 idx found for nframes on... %s" % str(list(set(labels_df['nframes_on'])))
        assert len(labels_df['stim_on_frame'].unique())==1, "More than 1 idx found for first frame on... %s" % str(list(set(labels_df['stim_on_frame'])))
        nframes_on = int(round(labels_df['nframes_on'].unique()[0]))
        stim_on_frame =  int(round(labels_df['stim_on_frame'].unique()[0]))
    except Exception as e:
        stimdur_vary = True
        
    groupby_list = ['config', 'trial']
    config_groups = labels_df.groupby(groupby_list)

    df_list = []
    for (config, trial), trial_ixs in config_groups:
        if stimdur_vary:
            # Get stim duration info for this config:
            assert len(labels_df[labels_df['config']==config]['nframes_on'].unique())==1, "Something went wrong! More than 1 unique stim dur for config: %s" % config
            assert len(labels_df[labels_df['config']==config]['stim_on_frame'].unique())==1, "Something went wrong! More than 1 unique stim ON frame for config: %s" % config
            nframes_on = labels_df[labels_df['config']==config]['nframes_on'].unique()[0]
            stim_on_frame = labels_df[labels_df['config']==config]['stim_on_frame'].unique()[0]
             
        trial_frames = roidata[trial_ixs.index.tolist(), :]
        nrois = trial_frames.shape[-1]
        #base_mean= trial_frames[0:stim_on_frame, :].mean(axis=0)
        base_std = trial_frames[0:stim_on_frame].std()
        stim_mean = trial_frames[stim_on_frame:stim_on_frame+nframes_on, :].mean(axis=0)
        zscore = stim_mean / base_std
        df_list.append(pd.DataFrame({'config': np.tile(config, (nrois,)),
                                     'trial': np.tile(trial, (nrois,)), 
                                     'meanstim': stim_mean,
                                     'zscore': zscore}))

    df = pd.concat(df_list, axis=0) # size:  ntrials * 2 * nrois
    df_by_rois = df.groupby(df.index)
    
    return df_by_rois



def boxplots_selectivity(df_by_rois, roi_list, metric='meanstim', stimlabels={}, topn=10, sort_dir='/tmp', data_identifier=''):

    selective_dir = os.path.join(sort_dir, 'selectivity', 'figures')
    if not os.path.exists(selective_dir):
        os.makedirs(selective_dir)
    
    if data_identifier == '':
        data_identifier = os.path.split(sort_dir)[0]
    
    print "Plotting box plots for top %i selective cells out of %i total (selective) cells." % (topn, len(roi_list))
        
    for roi in roi_list[0:topn]:
        rdata = df_by_rois.get_group(int(roi))
    
        #% Sort configs by mean value:
        df2 = pd.DataFrame({col:vals[metric] for col,vals in rdata.groupby('config')})
        meds = df2.median().sort_values(ascending=False)
        df2 = df2[meds.index]
        fig = pl.figure(figsize=(10,5))
        ax = sns.boxplot(data=df2)
        pl.title('%roi %05d' % int(roi+1))
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
        
        figname = 'selectivity_%s_roi%05d.png' % (metric, int(roi+1))
        label_figure(fig, data_identifier)
        pl.savefig(os.path.join(selective_dir, figname))
        pl.close()



#%%
def find_visual_cells(roidata, labels_df, sort_dir='/tmp', nprocs=4, create_new=False, data_identifier=''):
    
    # Create output dir for ANOVA2 results:
    responsive_resultsdir = os.path.join(sort_dir, 'responsivity', 'spanova2_results')
    if not os.path.exists(responsive_resultsdir):
        os.makedirs(responsive_resultsdir)
        
    responsive_anova_fpath = os.path.join(sort_dir, 'visual_rois_spanova2_results.json')
    if create_new is False:
        try:
            print "Loading existing split ANOVA results:\n", responsive_anova_fpath
            with open(responsive_anova_fpath, 'r') as f:
                responsive_anova = json.load(f)
        except Exception as e:
            print "[E]: No previous results found for SP Anova2 test for visual responsivity. Creating new."
            create_new = True
    
    df_by_rois = group_roidata_trialepoch(roidata, labels_df)
    
    if create_new:
        responsive_anova = visually_responsive_spanova2(df_by_rois, output_dir=responsive_resultsdir, nprocs=nprocs)

        # Save responsive roi list to disk:
        print "Saving split ANOVA results to:\n", responsive_anova_fpath
        with open(responsive_anova_fpath, 'w') as f:
            json.dump(responsive_anova, f, indent=4, sort_keys=True)

    # Sort ROIs:
    responsive_rois = [r for r in responsive_anova.keys() if responsive_anova[r]['p'] < 0.05]
    sorted_visual = sorted(responsive_rois, key=lambda x: responsive_anova[x]['F'])[::-1]
    
    if create_new:
        summary_fpath = os.path.join(sort_dir, 'roi_summary.txt')
        top10 = ['roi%05d' % int(r+1) for r in sorted_visual[0:10]]
        with open(summary_fpath, 'a') as f:
            f.write('----------------------------------------------------------\n')
            f.write('Split-plot ANOVA2 results:\n')
            f.write('----------------------------------------------------------\n')
            f.write('%i out of %i pass visual responsivity test (p < 0.05).\n' % (len(responsive_rois), len(responsive_anova.keys())))
            f.write('Top 10 (sorted by F val):\n    %s' % str(top10))
            
        boxplots_responsivity(df_by_rois, responsive_anova, sorted_visual, topn=10, sort_dir=sort_dir)
                
    return responsive_anova, sorted_visual


def visually_responsive_spanova2(df_by_rois, nprocs=4, output_dir='/tmp', fname='boxplot(intensity~epoch_X_config).png'):

    '''
    Take single ROI as a datatset, do split-plot rmANOVA:
        within-trial factor :  baseline vs. stimulus epoch
        between-trial factor :  stimulus condition
    '''

    roi_list = df_by_rois.groups.keys()
    print("Calculating split-plot ANOVA (factors=epoch, config) for %i rois." % len(roi_list))

    t_eval_mp = time.time()

    def worker(rlist, df_by_rois, output_dir, out_q):
        """
        Worker function is invoked in a process. 'roi_list' is a list of
        roi names to evaluate [rois00001, rois00002, etc.]. Results are placed
        in a dict that is pushed to a queue.
        """
        outdict = {}
        print len(rlist)
        for roi in rlist:
            # Format pandas df into pyvttbl dataframe:
            rdata = df_by_rois.get_group(int(roi))
            pdf = pyvt_format_trialepoch_df(rdata)
            outdict[roi] = pyvt_splitplot_anova2(roi, pdf, output_dir=output_dir, asdict=True)  
        out_q.put(outdict)

    # Each process gets "chunksize' filenames and a queue to put his out-dict into:
    out_q = mp.Queue()
    chunksize = int(math.ceil(len(roi_list) / float(nprocs)))
    procs = []

    for i in range(nprocs):
        p = mp.Process(target=worker,
                       args=(roi_list[chunksize * i:chunksize * (i + 1)],
                                       df_by_rois,
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


def group_roidata_trialepoch(roidata, labels_df):
    
    try:
        stimdur_vary = False
        assert len(labels_df['nframes_on'].unique())==1, "More than 1 idx found for nframes on... %s" % str(list(set(labels_df['nframes_on'])))
        assert len(labels_df['stim_on_frame'].unique())==1, "More than 1 idx found for first frame on... %s" % str(list(set(labels_df['stim_on_frame'])))
        nframes_on = int(round(labels_df['nframes_on'].unique()[0]))
        stim_on_frame =  int(round(labels_df['stim_on_frame'].unique()[0]))
    except Exception as e:
        stimdur_vary = True

    groupby_list = ['config', 'trial']
    config_groups = labels_df.groupby(groupby_list)

    df_list = []
    for (config, trial), trial_ixs in config_groups:
        if stimdur_vary:
            # Get stim duration info for this config:
            assert len(labels_df[labels_df['config']==config]['nframes_on'].unique())==1, "Something went wrong! More than 1 unique stim dur for config: %s" % config
            assert len(labels_df[labels_df['config']==config]['stim_on_frame'].unique())==1, "Something went wrong! More than 1 unique stim ON frame for config: %s" % config
            nframes_on = labels_df[labels_df['config']==config]['nframes_on'].unique()[0]
            stim_on_frame = labels_df[labels_df['config']==config]['stim_on_frame'].unique()[0]
            
        trial_frames = roidata[trial_ixs.index.tolist(), :]
        nrois = trial_frames.shape[-1]
        base_mean= trial_frames[0:stim_on_frame, :].mean(axis=0)
        #base_std = trial_frames[0:stim_on_frame].std()
        stim_mean = trial_frames[stim_on_frame:stim_on_frame+nframes_on, :].mean(axis=0)

        df_list.append(pd.DataFrame({'config': np.tile(config, (nrois,)),
                                     'trial': np.tile(trial, (nrois,)), 
                                     'epoch': 'baseline',
                                     'intensity': base_mean}))
        df_list.append(pd.DataFrame({'config': np.tile(config, (nrois,)),
                                     'trial': np.tile(trial, (nrois,)), 
                                     'epoch': 'stimulus',
                                     'intensity': stim_mean}))

    df = pd.concat(df_list, axis=0) # size:  ntrials * 2 * nrois
    df_by_rois = df.groupby(df.index)
    
    return df_by_rois

def pyvt_format_trialepoch_df(rdata):
    '''
    PYVT has its own dataframe structure, so just reformat pandas DF.
    '''
    df_factors = ['config', 'trial', 'epoch', 'intensity']
    Trial = namedtuple('Trial', df_factors)
    pdf = pt.DataFrame()
    [pdf.insert(Trial(cf, tr, ep, val)._asdict()) for cf, tr, ep, val in zip(rdata['config'], rdata['trial'], rdata['epoch'], rdata['intensity'])]
    return pdf


#%
# =============================================================================
# STATISTICS:
# =============================================================================


def pyvt_splitplot_anova2(roi, pdf, output_dir='/tmp', asdict=True):
    '''
    Calculate splt-plot ANOVA.
    Return results for this ROI as a dict.
    '''
    # Calculate ANOVA split-plot:
    aov = pdf.anova('intensity', sub='trial',
                       wfactors=['epoch'],
                       bfactors=['config'])

    aov_results_fpath = os.path.join(output_dir, 'visual_anova_results_%s.txt' % roi)
    with open(aov_results_fpath,'wb') as f:
        f.write(str(aov))
    f.close()
    results_epoch = extract_apa_anova2(('epoch',), aov)
    if asdict is True:
        return results_epoch
    else:
        return results_epoch['F'], results_epoch['p']
    
    
def extract_apa_anova2(factor, aov, values = ['F', 'mse', 'eta', 'p', 'df']):
    '''
    Returns ANOVA results as dict holding standard reported values.
    '''
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

#%
def boxplots_responsivity(df_by_rois, responsive_anova, sorted_visual, topn=10, sort_dir='/tmp', data_identifier=''):
    
    # Box plots for top N rois (sortedb y ANOVA F-value):
    print("Plotting box plots for factors EPOCH x CONFIG for top %i rois." % topn)
    vis_responsive_dir = os.path.join(sort_dir, 'responsivity', 'figures')
    if not os.path.exists(vis_responsive_dir):
        os.makedirs(vis_responsive_dir)
    
    pyvt_boxplot_epochXconfig(df_by_rois, sorted_visual[0:topn], output_dir=vis_responsive_dir)

    # View histogram of partial eta-squared values:
    eta2p_vals = [responsive_anova[r]['eta2_p'] for r in responsive_anova.keys()]
    eta2p = pd.Series(eta2p_vals)
    
    fig = pl.figure()
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
    datestring = datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")
    figname = 'responsivity_eta2_partial_%s.png' % datestring
    pl.savefig(os.path.join(os.path.split(vis_responsive_dir)[0], figname))
    if data_identifier == '':
        data_identifier = os.path.split(sort_dir)[0]
    label_figure(fig, data_identifier)
    pl.close()
    
#%
def pyvt_boxplot_epochXconfig(df_by_rois, roi_list, output_dir='/tmp'):

    for roi in roi_list:
        rdata = df_by_rois.get_group(int(roi))
        pdf = pyvt_format_trialepoch_df(rdata)
        factor_list = ['config', 'epoch']
        fname = 'roi%05d_boxplot(intensity~epoch_X_config).png' % (int(roi)+1)
        pdf.box_plot('intensity', factors=factor_list, fname=fname, output_dir=output_dir)

    print "Done plotting boxplots for top %i visually responsive ROIs." % len(roi_list)


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

def calculate_roi_responsivity(options):
    optsE = extract_options(options)
    create_new = optsE.create_new
    nprocs = optsE.nprocesses
    
    acquisition_dir = os.path.join(optsE.rootdir, optsE.animalid, optsE.session, optsE.acquisition)
    traceid_dir = util.get_traceid_from_acquisition(acquisition_dir, optsE.run, optsE.traceid)
    traceid = os.path.split(traceid_dir)[-1]
    trace_type = optsE.trace_type
    data_identifier = '_'.join((optsE.animalid, optsE.session, optsE.acquisition, traceid, trace_type))
    
    # Create output dir for ROI selection:
    # =========================================================================
    print "Creating OUTPUT DIRS for ROI analyses..."
    sort_dir = os.path.join(traceid_dir, 'sorted_rois')
    if not os.path.exists(sort_dir):
        os.makedirs(sort_dir)

    # Load data array:
    # -------------------------------------------------------------------------
    data_fpath = os.path.join(traceid_dir, 'data_arrays', 'datasets.npz')
    print "Loaded data from: %s" % traceid_dir
    dataset = np.load(data_fpath)
    print dataset.keys()
    
    # Get trace array (nframes x nrois) and experiment info labels (nframes x nfeatures)
    assert trace_type in dataset.keys(), "[W] Specified trace_type %s does not exist in dataset." % trace_type
    roidata = dataset[trace_type]
    labels_df = pd.DataFrame(data=dataset['labels_data'], columns=dataset['labels_columns'])
    assert roidata.shape[0] == labels_df.shape[0], "[W] trace data shape (%s) does not match labels (%s)" % (str(roidata.shape), str(labels_df.shape))
    sconfigs = dataset['sconfigs'][()]
    nrois_total = roidata.shape[-1]
    
    # Create output text file for quick-viewing of summary info:
    # -------------------------------------------------------------------------
    summary_fpath = os.path.join(sort_dir, 'roi_summary.txt')
    with open(summary_fpath, 'w') as f:
        f.write('----------------------------------------------------------\n')
        f.write('ANIMAL ID: %s - SESSION: %s - FOV: %s - RUN: %s\n' % (optsE.animalid, optsE.session, optsE.acquisition, optsE.run))

    # =========================================================================
    # RESPONSIVITY:
    # =========================================================================
    responsive_anova, sorted_visual = find_visual_cells(roidata, labels_df, 
                                                        sort_dir=sort_dir, 
                                                        nprocs=nprocs, 
                                                        create_new=create_new, 
                                                        data_identifier=data_identifier)
    print("%i out of %i cells pass split-plot ANOVA test for visual responses." % (len(sorted_visual), len(responsive_anova)))


    # =========================================================================
    # SELECTIVTY:
    # =========================================================================
    post_hoc = 'dunn'
    metric = 'meanstim'
    # Rename stim labels so that "config" is replaced w/ sth informative...
    if sconfigs[sconfigs.keys()[0]]['stimtype']=='gratings':
        stimlabels = dict((c, sconfigs[c]['ori']) for c in sconfigs.keys())
    elif sconfigs[sconfigs.keys()[0]]['stimtype']=='image':
        stimlabels = dict((c, sconfigs[c]['object']) for c in sconfigs.keys())
    elif sconfigs[sconfigs.keys()[0]]['stimtype']=='movie':
        stimlabels = dict((c, sconfigs[c]['object']) for c in sconfigs.keys())
        
    selectivityKW_results, sorted_selective = find_selective_cells(roidata, labels_df, 
                                                                   roi_list=sorted_visual, 
                                                                   post_hoc='dunn', 
                                                                   sort_dir=sort_dir, 
                                                                   metric=metric, 
                                                                   stimlabels=stimlabels, 
                                                                   data_identifier=data_identifier,
                                                                   nprocs=nprocs, 
                                                                   create_new=create_new)
    
    # Update roi stats summary file:
    # ---------------------------------------------------------
    H_mean = np.mean([selectivityKW_results[r]['H'] for r in selectivityKW_results.keys()])
    H_std = np.std([selectivityKW_results[r]['H'] for r in selectivityKW_results.keys()])
    with open(summary_fpath, 'a') as f:
        print >> f, '**********************************************************************'
        print >> f, '%i out of %i cells are visually responsive (split-plot ANOVA, p < 0.05)' % (len(sorted_visual), nrois_total)
        print >> f, '%i out of %i visual are stimulus selective (Kruskal-Wallis, p < 0.05)' % (len(sorted_selective), len(sorted_visual))
        print >> f, 'Mean H=%.2f (std=%.2f)' % (H_mean, H_std)
        print >> f, '**********************************************************************'
        print >> f, 'VISUAL -- Top 10 neuron ids: %s' % str([int(r) for r in sorted_visual[0:10]])
        print >> f, 'SELECTIVE -- Top 10 neuron ids: %s' % str([int(r) for r in sorted_selective[0:10]])
        
    
    # Save ROI stats to file for easy access later:
    roistats_fpath = os.path.join(sort_dir, 'roistats_results.npz')
    np.savez(roistats_fpath, 
             animalid=optsE.animalid,
             session=optsE.session,
             acquisition=optsE.acquisition,
             traceid=traceid,
             nrois_total=nrois_total,
             responsivity_test='pyvt_splitplot_anova2',
             sorted_visual=sorted_visual,
             sorted_selective=sorted_selective,
             selectivity_test = 'kruskal_wallis',
             selectivity_posthoc=post_hoc,
             metric=metric
             )
    print "Saved ROI stat results to: %s" % roistats_fpath
    
    return roistats_fpath

#%%
    

def main(options):
    
    roistats_fpath = calculate_roi_responsivity(options)
    
    print "*******************************************************************"
    print "DONE!"
    print "*******************************************************************"
    
    
#%%

if __name__ == '__main__':
    main(sys.argv[1:])
