#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 08 12:18:13 2020

@author: julianarhee
"""

import os
import json
import glob
import copy
import sys
import optparse

import numpy as np
import pylab as pl
import seaborn as sns
import pandas as pd
import cPickle as pkl

from scipy import stats as spstats

from pipeline.python.classifications import experiment_classes as util
from pipeline.python.classifications import aggregate_data_stats as aggr
from pipeline.python.utils import natural_keys, label_figure
# from pipeline.python.classifications import get_dataset_stats as gd
from pipeline.python.retinotopy import fit_2d_rfs as fitrf
from pipeline.python.classifications import evaluate_receptivefield_fits as evalrfs

def get_excluded_datasets(filter_by='drop_repeats', excluded_sessions=[]):

    # Blobs runs w/ incorrect stuff
    always_exclude = ['20190426_JC078']

    if filter_by=='drop_repeats':
        # Sessions with repeat FOVs
        lm_repeats = ['20190513_JC078', '20190504_JC078', '20190509_JC078', '20190506_JC080', 
                      '20190512_JC083', '20190517_JC083']
        
        li_repeats = ['20190609_JC099', '20190606_JC091', '20190607_JC091', '20191108_JC113']

        v1_repeats = ['20190501_JC076', '20190510_JC083', '20190511_JC083']
        
        also_exclude = [v1_repeats, lm_repeats, li_repeats]

    for excl in also_exclude:
        excluded_sessions.extend(excl)
    excluded_sessions = list(set(excluded_sessions))

    return excluded_sessions


def get_metadata(traceid='traces001', fov_type='zoom2p0x', state='awake',
                    filter_by='drop_repeats',
                    aggregate_dir='/n/coxfs01/julianarhee/aggregate-visual-areas'):
       
    # Get all datasets
    sdata = aggr.get_aggregate_info(traceid=traceid, fov_type=fov_type, state=state,
                             aggregate_dir=aggregate_dir)

    # Filter excluded datasets
    excluded_sessions = get_excluded_datasets(filter_by=filter_by)
    dsets = pd.concat([g for v, g in sdata.groupby(['visual_area', 'animalid', 'session', 'fovnum']) \
            if '%s_%s' % (v[2], v[1]) not in excluded_sessions])

    return dsets


def create_deviant_dataframe(dsets, traceid='traces001', n_processes=1,
                             response_type='dff', fit_thr=0.50, ci=0.95,
                             n_bootstrap_iters=1000, n_resamples=10, transform_fov=True,
                             plot_boot_distns=True, deviant_color='dodgerblue', 
                             filter_weird=False, plot_all_cis=False):
    
    stats_load_errors= []
    devdf = []  
    bad_fit_results = []
    for (visual_area, animalid, session, fovnum), g in dsets.groupby(['visual_area', 'animalid', 'session', 'fovnum']):

        datakey = '%s_%s_fov%i' % (session, animalid, fovnum)
        fov = 'FOV%i_zoom2p0x' % fovnum

        rfnames = [r for r in g['experiment'].values if 'rf' in r]
        for rfname in rfnames:
            exp = util.ReceptiveFields(rfname, animalid, session, fov,
                                       traceid=traceid) #, trace_type='dff')

            # Create output dirs
            statsdir, stats_desc = util.create_stats_dir(exp.animalid, exp.session, exp.fov,
                                                          traceid=exp.traceid, trace_type=exp.trace_type,
                                                          response_type=response_type, 
                                                          responsive_test=None, responsive_thr=0)
            if not os.path.exists(os.path.join(statsdir, 'receptive_fields')):
                os.makedirs(os.path.join(statsdir, 'receptive_fields'))
            print("Saving stats output to: %s" % statsdir)    

            # Get RF fit stats
            try:
                estats = exp.get_stats(response_type=response_type, fit_thr=fit_thr)
            except Exception as e:
                stats_load_errors.append(datakey)
                continue
            rfdf = estats.fits
            fovinfo = estats.fovinfo
            rois_rfs = rfdf.index.tolist()

            # Evaluate RF results        
            regresults = evalrfs.do_rf_fits_and_evaluation(animalid, session, fov, 
                                                            rfname=rfname,
                                                            n_processes=n_processes, 
                                                            fit_thr=fit_thr,
                                                            ci=ci, n_bootstrap_iters=n_bootstrap_iters,
                                                            n_resamples=n_resamples,
                                                            transform_fov=transform_fov,
                                                            plot_boot_distns=plot_boot_distns,
                                                            deviant_color=deviant_color,
                                                            filter_weird=filter_weird,
                                                            plot_all_cis=plot_all_cis)       

            if len(regresults)==0:
                bad_fit_results.append(datakey)
                continue
                
            # Identify good fits and deviants
            bad_either = np.union1d(regresults['azimuth']['bad_fits'], regresults['elevation']['bad_fits'])
            pass_rois_rfs = np.array([r for r in rois_rfs if r not in bad_either])
            #deviants_either = np.union1d(regresults['azimuth']['deviants'], regresults['elevation']['deviants'])
            #deviant_rois = np.intersect1d(pass_rois_rfs, deviants_either)
            #print(len(deviant_rois), len(pass_rois_rfs), len(rois_rfs), len(bad_either))

            for cond in regresults.keys():
                nsamples = len(regresults[cond]['deviants'])
                deviant_rois = [None] if nsamples==0 else regresults[cond]['deviants']
                ns = len(deviant_rois)

                cond_df = pd.DataFrame({'deviants': deviant_rois,
                                        'cond': [cond for _ in np.arange(0, ns)],
                                        'n_rois': [len(rois_rfs) for _ in np.arange(0, ns)],
                                        'n_rois_pass': [len(pass_rois_rfs) for _ in np.arange(0, ns)],
                                        'experiment': [rfname for _ in np.arange(0, ns)],
                                        'datakey': [datakey for _ in np.arange(0, ns)]})
                devdf.append(cond_df)
            
    devdf = pd.concat(devdf, axis=0).reset_index(drop=True)
        
    return devdf
   

def get_deviant_data(traceid='traces001', fov_type='zoom2p0x', state='awake', n_processes=1,
                     response_type='dff', filter_by='drop_repeats', fit_thr=0.5, transform_fov=True,
                     ci=0.95, n_bootstrap_iters=1000, n_resamples=10, create_new=False, 
                     plot_boot_distns=True, filter_weird=False, plot_all_cis=False, deviant_color='dodgerblue',
                     aggregate_dir='/n/coxfs01/julianarhee/aggregate-visual-areas'):

    '''
    Get dataframe containing RF fit evaluation stats and ROI position info.
    Will run evaluation if needed.
    Save to: <aggregate_dir>/receptive-fields/<traceid>-<fit_desc>/scatter/deviants_dfile.pkl
    '''
    # Get datasets to process
    dsets = get_metadata(traceid=traceid, fov_type=fov_type, state=state, filter_by=filter_by)

    # Set output dir
    fit_desc = fitrf.get_fit_desc(response_type=response_type)
    outdir = os.path.join(aggregate_dir, 'receptive-fields', '%s-%s' % (traceid, fit_desc), 'scatter')
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    print(outdir)

    deviants_dfile = os.path.join(outdir, 'deviants_dfile.pkl')
    # Load or create deviant df
    if os.path.exists(deviants_dfile) and create_new is False:
        print("Loading deviant info")
        with open(deviants_dfile, 'rb') as f:
            devdf = pkl.load(f)
    else:
        print("Creating deviant df")
        devdf = create_deviant_dataframe(dsets, traceid=traceid, 
                                        response_type=response_type, fit_thr=fit_thr,
                                        n_processes=n_processes, ci=ci, 
                                        n_bootstrap_iters=n_bootstrap_iters, n_resamples=n_resamples,
                                        transform_fov=transform_fov, filter_weird=filter_weird,
                                        plot_boot_distns=plot_boot_distns, plot_all_cis=plot_all_cis,
                                        deviant_color=deviant_color)


        with open(deviants_dfile, 'wb') as f:
            pkl.dump(devdf, f, protocol=pkl.HIGHEST_PROTOCOL)
        

def extract_options(options):
    
    parser = optparse.OptionParser()
    parser.add_option('-D', '--root', action='store', dest='rootdir', default='/n/coxfs01/2p-data',\
                      help='data root dir (root dir containing all animalids) [default: /n/coxfs01/2pdata]')
    
    parser.add_option('-A', '--aggregate-dir', action='store', dest='aggregate_dir', default='/n/coxfs01/julianarhee/aggregate-visual-areas',\
                      help='dir for aggregatd data [default: /n/coxfs01/julianarhee/aggregate-visual-areas]')
    
    parser.add_option('-F', '--fov-type', action='store', dest='fov_type', default='zoom2p0x', \
                      help="acquisition type (ex: 'FOV1_zoom3x') [default: zoom2p0x]")
    parser.add_option('--new', action='store_true', dest='create_new', default=False, \
                      help="flag to fit linear regr anew to calculate deviants")

    parser.add_option('-t', '--traceid', action='store', dest='traceid', default='traces001', \
                      help="name of traces ID [default: traces001]")

    parser.add_option('-r', '--resp', action='store', dest='response_type', default='dff', \
                      help="Response metric to use for creating RF maps (default: dff)")    
    parser.add_option('-f', '--fit-thr', action='store', dest='fit_thr', default=0.5, \
                      help="Threshold for RF fits (default: 0.5)")

    parser.add_option('-b', '--n-boot', action='store', dest='n_bootstrap_iters', default=1000, \
                      help="N bootstrap iterations for evaluating RF param fits (default: 1000)")
    parser.add_option('-s', '--n-resamples', action='store', dest='n_resamples', default=10, \
                      help="N trials to sample with replacement (default: 10)")
    parser.add_option('-n', '--n-processes', action='store', dest='n_processes', default=1, \
                      help="N processes (default: 1)")
    
    parser.add_option('-C', '--ci', action='store', dest='ci', default=0.95, \
                      help="CI percentile(default: 0.95)")

    parser.add_option('--no-boot-plot', action='store_false', dest='plot_boot_distns', default=True, \
                      help="flag to not plot bootstrapped distNs of x0, y0 for each roi")

    parser.add_option('--pixels', action='store_false', dest='transform_fov', default=True, \
                      help="flag to not convert fov space into microns (keep as pixels)")

    parser.add_option('--remove-weird', action='store_true', dest='filter_weird', default=False, \
                      help="[plotting only] flag to remove really funky fits")
    parser.add_option('--all-cis', action='store_true', dest='plot_all_cis', default=False, \
                      help="[plotting] flag to plot CIs for all cells (not just deviants)")

    parser.add_option('-c', '--color', action='store', dest='deviant_color', default='dodgerblue', \
            help="color to plot deviants to stand out (default: dodgerblue)")

    parser.add_option('--sigma', action='store', dest='sigma_scale', default=2.35, \
                      help="sigma scale factor for FWHM (default: 2.35)")

 
    parser.add_option('--filter', action='store', dest='filter_by', default='drop_repeats', \
                      help="filter for excluding datasets (default: drop_repeats)")
    parser.add_option('--state', action='store', dest='state', default='awake', \
                      help="anesthetized or awake (default: awake)")


   

    (options, args) = parser.parse_args(options)

    return options


   

def main(options):
    opts = extract_options(options)
    devdf = get_deviant_data(filter_by=opts.filter_by, traceid=opts.traceid, fov_type=opts.fov_type,
                                state=opts.state, n_processes=int(opts.n_processes),
                                response_type=opts.response_type,
                                fit_thr=float(opts.fit_thr), ci=float(opts.ci), 
                                n_bootstrap_iters = int(opts.n_bootstrap_iters), 
                                n_resamples=int(opts.n_resamples), transform_fov=opts.transform_fov,
                                plot_boot_distns=opts.plot_boot_distns, 
                                plot_all_cis=opts.plot_all_cis,
                                filter_weird=opts.filter_weird,
                                deviant_color=opts.deviant_color,
                                create_new=opts.create_new, aggregate_dir=opts.aggregate_dir)



    print("Got all deviants")

if __name__ == '__main__':
    main(sys.argv[1:])


