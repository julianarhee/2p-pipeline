#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 17:01:20 2019

@author: julianarhee
"""

import os
import glob
import json
import h5py
import optparse
import sys
import math
import time
import cPickle as pkl
import json

import matplotlib as mpl
mpl.use('agg')

import numpy as np
import pandas as pd
import pylab as pl
import seaborn as sns
import multiprocessing as mp

from sklearn.utils import shuffle

#from pipeline.python.classifications import experiment_classes as util #utils as util
from pipeline.python.classifications import test_responsivity as resp
from pipeline.python.utils import label_figure, natural_keys

def get_hits_and_fas(resp_stim, resp_bas):
    
    curr_cfg_ixs = range(resp_stim.shape[0])
    n_conditions, n_trials = resp_stim.shape
    min_val = resp_stim.min()
    max_val = resp_stim.max()
    
    crit_vals = np.linspace(min_val, max_val)
    
    p_hits = np.empty((len(curr_cfg_ixs), len(crit_vals)))
    p_fas = np.empty((len(curr_cfg_ixs), len(crit_vals)))
    for ci in range(n_conditions): #range(n_conditions):
        p_hit = [sum(resp_stim[ci, :] > crit) / float(n_trials) for crit in crit_vals]
        p_fa = [sum(resp_bas[ci, :] > crit) / float(n_trials) for crit in crit_vals]
        p_hits[ci, :] = p_hit
        p_fas[ci, :] = p_fa
        
    return p_hits, p_fas, crit_vals


def load_experiment_data(experiment_name, animalid, session, fov, traceid, 
                        trace_type='corrected', rootdir='/n/coxfs01/2p-data'):
    
    from pipeline.python.classifications import experiment_classes as util #utils as util

    if 'gratings' in experiment_name:
        exp = util.Gratings(animalid, session, fov, traceid=traceid, rootdir=rootdir)
    elif 'blobs' in experiment_name:
        exp = util.Objects(animalid, session, fov, traceid=traceid, rootdir=rootdir)
    else: 
        exp = util.Experiment(experiment_name, animalid, session, fov, traceid) 
    exp.load(trace_type='corrected') #trace_type)
   
    print("... loaded data") 
    #exp.data.traces, exp.data.labels = util.check_counts_per_condition(exp.data.traces, exp.data.labels)
    gdf = resp.group_roidata_stimresponse(exp.data.traces, exp.data.labels)
    
    # Reformat/rename stimulus params:
    excluded_params = []
    if experiment_name == 'gratings':
        excluded_params = ['position', 'xpos', 'ypos']
    elif experiment_name == 'blobs':
        excluded_params = ['color', 'xpos', 'ypos', 'object']
        fix_cfgs = exp.data.sdf[np.isnan(exp.data.sdf['size'])].index.tolist()
        for cfg in fix_cfgs:
            exp.data.sdf.loc[cfg, 'size'] = 0
    elif experiment_name in ['rfs', 'rfs10']:
        excluded_params = ['position']
            
    all_params = [c for c in exp.data.sdf.columns if c not in excluded_params]
    tested_params = [c for c in all_params if len(exp.data.sdf[c].unique()) > 1]
    stim_params = dict((str(p), sorted(exp.data.sdf[p].unique())) for p in tested_params)
    #print("Tested stim params:")
    #for param, vals in stim_params.items():
    #    print('%s: %i' % (param, len(vals)))

    return exp, gdf        

def calculate_roc_bootstrap(roi_df, n_iters=1000):

    resp_stim = np.vstack(roi_df.groupby(['config'])['stim_mean'].apply(np.array).values)
    resp_bas = np.vstack(roi_df.groupby(['config'])['base_mean'].apply(np.array).values)

    # Generate ROC curve 
    n_conditions, n_trials = resp_stim.shape
    p_hits, p_fas, crit_vals = get_hits_and_fas(resp_stim, resp_bas)
    true_auc = []
    for ci in range(n_conditions):
        true_auc.append(-np.trapz(p_hits[ci, :], x=p_fas[ci, :]))
    max_true_auc = np.max(true_auc)
    
    #### Shuffle
    all_values = np.vstack([resp_stim, resp_bas])
    #print(all_values.shape)

    # Shuffle values, group into stim and bas again
    shuff_auc = []
    print("... getting shuffle")
    for i in range(n_iters):
        print(i)
        X = shuffle(all_values.ravel())
        X = np.reshape(X, (n_conditions*2, n_trials))
        shuff_stim = X[0:n_conditions, :]
        shuff_bas = X[n_conditions:, :]
        shuff_p_hits, shuff_p_fas, shuff_crit_vals = get_hits_and_fas(shuff_stim, shuff_bas)
        shuff_auc.append(np.max([-np.trapz(shuff_p_hits[ci, :], x=shuff_p_fas[ci, :]) for ci in range(n_conditions)]))

    pval = sum(shuff_auc >= max_true_auc)/ float(len(shuff_auc))

    roc_results = {'p_hits': p_hits,
                   'p_fas': p_fas, 
                   'resp_stim': resp_stim,
                    'res_bas': resp_bas,
                    'auc': true_auc,
                    'shuffled_auc': shuff_auc,
                    'pval': pval,
                    'n_iters': n_iters}

    return roc_results

def plot_roc_bootstrap_results(roc_results):
    n_conditions, n_trials = roc_results['resp_stim'].shape
    
    p_fas = roc_results['p_fas']
    p_hits = roc_results['p_hits']
    true_auc = roc_results['auc']
    shuff_auc = roc_results['shuffled_auc']
    max_true_auc = np.max(true_auc)
    
    colors = sns.color_palette('cubehelix', n_conditions)
    fig, ax = pl.subplots(1,3, figsize=(10,4))
    fig.patch.set_alpha(1.0)

    # TRUE: plot sorted ROC curves ----------------------------------------
    for ci in range(n_conditions):
        ax[0].plot(p_fas[ci, :], p_hits[ci, :], color=colors[ci])
    ax[0].set_xlim([0, 1])
    ax[0].set_ylim([0, 1])
    ax[0].set_xlabel('p (bas > crit)')
    ax[0].set_ylabel('p (stim > crit)')
    sns.despine(offset=4, trim=True, ax=ax[0])

    # TRUE: plot sorted AUC values ----------------------------------------
    ax[1].plot(sorted(true_auc, reverse=True))
    ax[1].set_xticks([]) 
    ax[1].set_ylabel('auc')
    ax[1].set_xlabel('sorted conditions')
    ax[1].set_ylim([0, 1.0])
    sns.despine(offset=4, trim=True, ax=ax[1])

    # SHUFFLED: plot sorted AUC distN -------------------------------------
    ax[2].hist(shuff_auc, color='k', alpha=0.5)
    ax[2].axvline(x=max_true_auc, color='r')
    ax[2].set_xlim([0.5, 1])
    ax[2].set_title('N iter=%i, p=%.3f' % (roc_results['n_iters'], roc_results['pval']), fontsize=8)
    ax[2].set_xlabel('max AUC')
    ax[2].set_ylabel('counts')
    sns.despine(trim=True, offset=4, ax=ax[2])

    pl.subplots_adjust(wspace=0.5, top=0.8, left=0.1, bottom=0.2)
    
    return fig

    
def do_roc_bootstrap_mp(gdf, dst_dir='/tmp', n_iters=1000, n_processes=1, plot_rois=False,
                        data_identifier='DATAID'):
    
    # Create output dirs:
    #traces_basedir = exp.source.split('/data_arrays/')[0]
    #output_dir = os.path.join(traces_basedir, 'summary_stats')
    
    #roc_dir = os.path.join(output_dir, 'ROC')
    #if not os.path.exists(roc_dir):
    #    os.makedirs(roc_dir)

    # create output dir for roi figures:
    roi_figdir = os.path.join(dst_dir, 'rois')
    if not os.path.exists(roi_figdir):
        os.makedirs(roi_figdir)
        
#    pval_dict = {}
#    for roi in range(len(gdf.groups)):
#    
#        roi_df = gdf.get_group(roi)
#
#        roc_results = calculate_roc_bootstrap(roi_df, n_iters=n_iters)
#        # PLOT:
#        if plot_rois:
#            
#            fig = plot_roc_bootstrap_results(roi, roc_results)
#            fig.suptitle('cell %i' % (int(roi+1)))
#            label_figure(fig, data_identifier)
#            pl.savefig(os.path.join(roi_figdir, 'roi%05d.png' % (int(roi+1))))
#            pl.close()
#        pval_dict.update({roi: {'max_auc': np.max(roc_results['auc']),
#                                'pval': roc_results['pval']}})
        

    def worker(roi_list, gdf, n_iters, plot_rois, roi_figdir, out_q):
        
        curr_results = {}
        for roi in roi_list:
            roi_df = gdf.get_group(roi)
            print(roi, roi_df.shape)
            roc_results = calculate_roc_bootstrap(roi_df, n_iters=n_iters)
            # PLOT:
            if plot_rois:
                fig = plot_roc_bootstrap_results(roc_results)
                fig.suptitle('cell %i' % (int(roi+1)))
                label_figure(fig, data_identifier)
                pl.savefig(os.path.join(roi_figdir, 'roi%05d.png' % (int(roi+1))))
                pl.close()
            curr_results[roi] = {'max_auc': np.max(roc_results['auc']),
                                'pval': roc_results['pval']}
                
        out_q.put(curr_results)
        
        
    # Each process gets "chunksize' filenames and a queue to put his out-dict into:
    roi_list = gdf.groups.keys()
    out_q = mp.Queue()
    chunksize = int(math.ceil(len(roi_list) / float(n_processes)))
    procs = []
    for i in range(n_processes):
        p = mp.Process(target=worker,
                       args=(roi_list[chunksize * i:chunksize * (i + 1)],
                                      gdf, n_iters, plot_rois, roi_figdir, out_q))
        procs.append(p)
        p.start()

    # Collect all results into single results dict. We should know how many dicts to expect:
    results = {}
    for i in range(n_processes):
        results.update(out_q.get())

    # Wait for all worker processes to finish
    for p in procs:
        print "Finished:", p
        p.join()
        
    return results, dst_dir
        

def main(options):

    opts = extract_options(options)
    n_iters = int(opts.n_iterations)
    n_processes = int(opts.n_processes)
    plot_rois = opts.plot_rois
    create_new = opts.create_new
    try:
        bootstrap_roc_func(opts.animalid, opts.session, opts.fov, opts.traceid, opts.experiment, 
                            trace_type=opts.trace_type, create_new=create_new,
                            rootdir=opts.rootdir, n_processes=n_processes, 
                            plot_rois=plot_rois, n_iters=n_iters)
    except Exception as e:
        print(e)
    print("******DONE BOOTSTRAP ROC ANALYSIS.")
 

def bootstrap_roc_func(animalid, session, fov, traceid, experiment, trace_type='corrected', rootdir='/n/coxfs01/2p-data',
                        n_processes=1, plot_rois=True, n_iters=1000, create_new=False):
 

    print(".... starting boot.", animalid, experiment, session, fov, traceid) #
    exp, gdf = load_experiment_data(experiment, animalid, session, fov, traceid, 
                                trace_type=trace_type, rootdir=rootdir) 
    data_identifier = '|'.join([animalid, session, fov, traceid, experiment, trace_type])
    print("... data id: %s" % data_identifier)

    traces_basedir = exp.source.split('/data_arrays/')[0]
    #output_dir = os.path.join(traces_basedir, 'summary_stats')
   
    #traces_basedir = glob.glob(os.path.join(rootdir, animalid, session, fov, 
    #                                        'combined_%s*' % experiment,
    #                                        'traces', '%s*' % traceid))[0]
    stats_dir = os.path.join(traces_basedir, 'summary_stats')
    roc_dir = os.path.join(stats_dir, 'ROC')
    if not os.path.exists(roc_dir):
        os.makedirs(roc_dir)

    #raw_traces, labels, sdf, run_info = load_dataset(soma_fpath, trace_type='corrected')
    # Each group is roi's trials x metrics
    #gdf = resp.group_roidata_stimresponse(raw_traces.values, labels, return_grouped=True) 

    print("STARTING BOOTSTRAP ANALYSIS.")
    start_t = time.time()
    results, roc_dir = do_roc_bootstrap_mp(gdf, dst_dir=roc_dir, n_iters=n_iters, 
                                  n_processes=n_processes, plot_rois=plot_rois,
                                  data_identifier=data_identifier) 
    print("FINISHED CALCULATING ROC BOOTSTRAP ANALYSIS.")
    end_t = time.time() - start_t
    print("--> Elapsed time: {0:.2f}sec".format(end_t))

    fmts = ['pkl', 'json']
    for fmt in fmts:
        results_outfile = os.path.join(roc_dir, 'roc_results.%s' % fmt)
        if fmt == 'pkl':
            with open(results_outfile, 'wb') as f:
                pkl.dump(results, f, protocol=pkl.HIGHEST_PROTOCOL)
        elif fmt == 'json':
            with open(results_outfile, 'w') as f:
                json.dump(results, f, sort_keys=True, indent=4)
                
    print("-- saved results to: %s" % results_outfile)
    
    thr_rois = [r for r, res in results.items() if res['pval'] < 0.05]
    sig_aucs = [res['max_auc'] for r, res in results.items() if r in thr_rois]
    nonsig_aucs = [res['max_auc'] for r, res in results.items() if r not in thr_rois]
    
    fig = pl.figure()
    fig.patch.set_alpha(1)
    weights1 = np.ones_like(sig_aucs) / float(len(results.keys()))
    weights2 = np.ones_like(nonsig_aucs) / float(len(results.keys())) #float(len(nonsig_aucs))
    
    pl.hist(sig_aucs, alpha=0.5, label='sig. (%i)' % len(sig_aucs), weights=weights1, normed=0)
    pl.hist(nonsig_aucs, alpha=0.5, label='non-sig. (%i)' % len(nonsig_aucs), weights=weights2, normed=0)
    pl.legend()
    
    pl.ylabel('frac of all selected cells')
    pl.xlabel('max AUC')
    pl.xlim([0, 1])
    
    sns.despine(offset=4, trim=True)
    label_figure(fig, data_identifier)
    pl.savefig(os.path.join(roc_dir, 'max_aucs.png'))
    pl.close()
    
    # Save list of rois that pass for quick ref:
    with open(os.path.join(roc_dir, 'significant_rois.json'), 'w') as f:
        json.dump(thr_rois, f)
    
    print("-- %i out if %i cells are responsive." % (len(thr_rois), len(results.keys())))
    
    
    #return results


def extract_options(options):

    parser = optparse.OptionParser()

    # PATH opts:
    parser.add_option('-D', '--root', action='store', dest='rootdir', default='/n/coxfs01/2p-data', help='data root dir (root project dir containing all animalids) [default: /nas/volume1/2photon/data, /n/coxfs01/2pdata if --slurm]')
    parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')
    parser.add_option('-S', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID')
    parser.add_option('-A', '--acq', action='store', dest='fov', default='FOV1_zoom2p0x', help="acquisition folder (ex: 'FOV1_zoom2p0x') [default: FOV1_zoom2p0x]")
    parser.add_option('-E', '--exp', action='store', dest='experiment', default='', help="Name of experiment (stimulus type), e.g., rfs")
    parser.add_option('--default', action='store_true', dest='default', default='store_false', help="Use all DEFAULT params, for params not specified by user (no interactive)")
    parser.add_option('--slurm', action='store_true', dest='slurm', default=False, help="set if running as SLURM job on Odyssey")
    parser.add_option('-t', '--trace-id', action='store', dest='traceid', default='traces001', help="Trace ID for current trace set (created with set_trace_params.py, e.g., traces001, traces020, etc.)")

    parser.add_option('-n', '--nproc', action="store",
                      dest="n_processes", default=2, help="N processes [default: 1]")
    parser.add_option('-d', '--trace-type', action="store",
                      dest="trace_type", default='corrected', help="Trace type to use for calculating stats [default: corrected]")

    parser.add_option('-N', '--niter', action="store",
                      dest="n_iterations", default=1000, help="N iterations for bootstrap [default: 1000]")
    parser.add_option('--plot', action='store_true', dest='plot_rois', default=False, help="set to plot results of each roi's analysis")

    parser.add_option('--new', action='store_true', dest='create_new', default=False, help="set to run bootstrap roc anew")


    (options, args) = parser.parse_args(options)

    return options

if __name__ == '__main__':
    main(sys.argv[1:])

 
