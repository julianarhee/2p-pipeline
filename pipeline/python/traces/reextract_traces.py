#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 08 16:27:30 2020

@author: julianarhee
"""

import os
import glob
import json
import h5py
import cv2
import traceback
import optparse
import sys
import matplotlib as mpl
mpl.use('agg')

import numpy as np
import pylab as pl

from pipeline.python.utils import natural_keys, label_figure
from pipeline.python.traces import realign_epochs as realign
from pipeline.python.traces import remake_neuropil_masks as rmasks
from pipeline.python.retinotopy import do_retinotopy_analysis as retino
from pipeline.python.retinotopy import fit_2d_rfs as fitrf
from pipeline.python.classifications import bootstrap_roc as roc


def extract_options(options):
    parser = optparse.OptionParser()

    # PATH opts:
    parser.add_option('-D', '--root', action='store', dest='rootdir', default='/n/coxfs01/2p-data', 
                      help='root project dir containing all animalids [default: /n/coxfs01/2pdata]')
    parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', 
                      help='Animal ID')
    parser.add_option('-S', '--session', action='store', dest='session', default='', 
                      help='Session (format: YYYYMMDD)')
    
    # Set specific session/run for current animal:
    parser.add_option('-A', '--fov', action='store', dest='fov', default='FOV1_zoom2p0x', 
                      help="fov name (default: FOV1_zoom2p0x)")
    parser.add_option('-E', '--experiment', action='store', dest='experiment', default='', 
                      help="experiment name (e.g,. gratings, rfs, rfs10, or blobs)") #: FOV1_zoom2p0x)")
    
    parser.add_option('-t', '--traceid', action='store', dest='traceid', default='traces001', 
                      help="traceid (default: traces001)")

    # Neuropil mask params
    parser.add_option('-N', '--np-outer', action='store', dest='np_niterations', default=24, 
                      help="Num cv dilate iterations for outer annulus (default: 24, ~50um for zoom2p0x)")
    parser.add_option('-g', '--np-inner', action='store', dest='gap_niterations', default=4, 
                      help="Num cv dilate iterations for inner annulus (default: 4, gap ~8um for zoom2p0x)")
    parser.add_option('-c', '--factor', action='store', dest='np_correction_factor', default=0.7, 
                      help="Neuropil correction factor (default: 0.7)")

    # Alignment params
    parser.add_option('-p', '--iti-pre', action='store', dest='iti_pre', default=1.0, 
                      help="pre-stim amount in sec (default: 1.0)")
    parser.add_option('-P', '--iti-post', action='store', dest='iti_post', default=1.0, 
                      help="post-stim amount in sec (default: 1.0)")

    parser.add_option('--plot', action='store_true', dest='plot_masks', default=False, 
                      help="set flat to plot soma and NP masks")

    parser.add_option('--masks', action='store_true', dest='create_masks', default=False, 
                      help="set flag to remake NP masks")
    parser.add_option('--apply', action='store_true', dest='apply_masks', default=False, 
                      help="set flag to apply NP masks")

    parser.add_option('--align', action='store_true', dest='align_trials', default=False, 
                      help="set flag to align trial epochs")
    parser.add_option('--retino', action='store_true', dest='do_retino', default=False, 
                      help="set flag to do retino analysis on runs")
    parser.add_option('--rfs', action='store_true', dest='fit_rfs', default=False, 
                      help="set flag to fit RFs")
    parser.add_option('--roc', action='store_true', dest='roc_test', default=False, 
                      help="set flag to do ROC test")

    parser.add_option('-n', '--nproc', action='store_true', dest='n_processes', default=1,
                      help="N processes (default: 1)")


#    parser.add_option('-r', '--rows', action='store', dest='rows',
#                          default=None, help='Transform to plot along ROWS (only relevant if >2 trans_types)')
#    parser.add_option('-c', '--columns', action='store', dest='columns',
#                          default=None, help='Transform to plot along COLUMNS')
#    parser.add_option('-H', '--hue', action='store', dest='subplot_hue',
#                          default=None, help='Transform to plot by HUE within each subplot')
#    parser.add_option('-d', '--response', action='store', dest='response_type',
#                          default='dff', help='Traces to plot (default: dff)')
#
#    parser.add_option('-f', '--filetype', action='store', dest='filetype',
#                          default='svg', help='File type for images [default: svg]')
#
    (options, args) = parser.parse_args(options)

    return options


def redo_manual_extraction(options):

    opts = extract_options(options)

    animalid = opts.animalid
    session = opts.session
    traceid = opts.traceid   
    fov = opts.fov
    rootdir = opts.rootdir

    np_niterations = int(opts.np_niterations)
    gap_niterations = int(opts.gap_niterations)
    np_correction_factor = float(opts.np_correction_factor)
    plot_masks = opts.plot_masks

    iti_pre = float(opts.iti_pre)
    iti_post = float(opts.iti_post)

    create_masks = opts.create_masks
    apply_masks = opts.apply_masks   
    align_trials = opts.align_trials
    do_retino = opts.do_retino
    fit_rfs = opts.fit_rfs
    roc_test = opts.roc_test

    n_processes = int(opts.n_processes)

    if create_masks: 
        print("1. Creating neuropil masks")
        rmasks.create_masks_for_all_runs(animalid, session, fov, traceid=traceid, 
                               np_niterations=np_niterations, gap_niterations=gap_niterations, 
                                rootdir=rootdir, plot_masks=plot_masks)

    if apply_masks:
        print("2. Applying neuropil masks")
        rmasks.apply_masks_for_all_runs(animalid, session, fov, traceid=traceid, 
                                rootdir=rootdir, np_correction_factor=np_correction_factor)
        
    # Get event-basd runs to extract
    session_dir = os.path.join(rootdir, animalid, session)
    nonretino_rundirs = [r for r in sorted(glob.glob(os.path.join(session_dir, fov, '*_run*')), key=natural_keys)\
                             if 'retino' not in r and 'compare' not in r]

    experiment_types = np.unique([os.path.split(r)[-1].split('_')[0] for r in nonretino_rundirs])
   
    if align_trials: 
        for experiment in experiment_types:
            if 'blobs' in experiment:
                continue
            print("3. Parsing trials - %s" % experiment) 
            realign.parse_trial_epochs(animalid, session, fov, experiment, traceid, 
                                        iti_pre=iti_pre, iti_post=iti_post)
            
            print("4. Aligning traces to trials - %s" % experiment)
            realign.align_traces(animalid, session, fov, experiment, 
                                    traceid, rootdir=rootdir)

    if do_retino:
        # Get retino runs and extract
        retino_rundirs = sorted(glob.glob(os.path.join(session_dir, fov, 'retino_run*')),
                                key=natural_keys)
        print("5.  Doiing retino anaysis for %i runs." % len(retino_rundirs))
        for retino_rundir in retino_rundirs:
            curr_retino_run = os.path.split(retino_rundir)[1]
            print("--> %s" % curr_retino_run)
            # extract
            retino_opts = ['-i', animalid, '-S', session, '-A', fov, 
                            '-g', gap_niterations, '-a', np_niterations, 
                            '--new', '--masks',
                            '-d', traceid, '-R', curr_retino_run]
            retino.do_analysis(retino_opts)

    if fit_rfs:
        response_type = 'dff'
        post_stimulus_sec = 0.5
        sigma_scale=2.35 #True
        scale_sigma=True
        ci=0.95

        # Do RF fits
        if int(session) < 20190511 and 'gratings' in experiment_types:
                rf_runs = ['rfs']
        else:
            rf_runs = [r for r in experiment_types if 'rfs' in r]

        for rf_run in rf_runs:
            print("[%s] 6a.  Fitting RF runs." % rf_run)
            # fit RFs
            fit_thr = 0.5
            res_, params_ = fitrf.fit_2d_receptive_fields(animalid, session, fov, 
                                                        rf_run, traceid, 
                                                        fit_thr=fit_thr, 
                                                        make_pretty_plots=True,
                                                        create_new=True,
                                                        reload_data=True,
                                                        sigma_scale=sigma_scale, 
                                                        scale_sigma=scale_sigma,
                                                        post_stimulus_sec=post_stimulus_sec,
                                                        response_type=response_type     
                                                        )

            # evaluate  
            print("[%s] 6b. Evaluating RF fits." % rf_run)
            eval_, params_ = evalrfs.do_rf_fits_and_evaluation(animalid, session, fov, 
                                                    rfname=rf_run, traceid=traceid, 
                                                    response_type=response_type, 
                                                    fit_thr=fit_thr, 
                                                    sigma_scale=sigma_scale, 
                                                    scale_sigma=scale_sigma,
                                                    ci=ci,
                                                    post_stimulus_sec=post_stimulus_sec,
                                                    n_processes=n_processes,
                                                    reload_data=False,
                                                    create_stats=True,
                                                    do_evaluation=True) 

    if roc_test:
        # Do ROC responsivity test
        if 'blobs' in experiment_types:
            print("[blobs] 7.  Doing ROC test.")
            roc.bootstrap_roc_func(animalid, session, fov, traceid, 'blobs', n_processes=n_processes)

        if 'gratings' in experiment_types and int(session) < 20190511:
            print("[gratings] 8.  Doing ROC test.")
            roc.bootstrap_roc_func(animalid, session, fov, traceid, 'gratings', n_processes=n_processes)

            print("[gratings] 9.  Doing OSI fits, using nstds")
            osi_thr = 0.66
            exp = cutils.Gratings(animalid, session, fov, traceid=traceid)
            bootr_, fitparams = exp.get_tuning(n_processes=n_processes, response_type='dff',
                                               make_plots=True, create_new=True,
                                               responsive_test='nstds', responsive_thr=10, n_stds=2.5)
            evalosi_, goodfits = exp.evaluate_fits(bootr_, fitparams, goodness_thr=osi_thr, make_plots=True)
            print("--- done: %i cells with good fits (thr %.2f)" % (len(goodfits), osi_thr))
       
    print("********************************")
    print("Finished.")
    print("********************************")


def main(options):
    redo_manual_extraction(options)


if __name__ == '__main__':
    main(sys.argv[1:])
    


