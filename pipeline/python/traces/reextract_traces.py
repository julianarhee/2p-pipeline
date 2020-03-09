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
    
    print("1. Creating neuropil masks")
    rmasks.create_masks_for_all_runs(animalid, session, fov, traceid=traceid, 
                           np_niterations=np_niterations, gap_niterations=gap_niterations, 
                            rootdir=rootdir, plot_masks=plot_masks)

    print("2. Applying neuropil masks")
    rmasks.apply_masks_for_all_runs(animalid, session, fov, traceid=traceid, 
                            rootdir=rootdir, np_correction_factor=np_correction_factor)
    
    # Get event-basd runs to extract
    session_dir = os.path.join(rootdir, animalid, session)
    nonretino_rundirs = [r for r in sorted(glob.glob(os.path.join(session_dir, fov, '*_run*')), key=natural_keys) if 'retino' not in r]

    experiment_types = np.unique([os.path.split(r)[-1].split('_')[0] for r in nonretino_rundirs])
    for experiment in experiment_types:
        print("3. Parsing trials - %s" % experiment) 
        realign.parse_trial_epochs(animalid, session, fov, experiment, traceid, 
                            iti_pre=iti_pre, iti_post=iti_post)
        
        print("4. Aligning traces to trials - %s" % experiment)
        relign.align_traces(animalid, session, fov, experiment, traceid, rootdir=rootdir)

    # Get retino runs and extract
    retino_rundirs = sorted(glob.glob(os.path.join(session_dir, fov, 'retino_run*')), key=natural_keys)
    print("5.  Doiing retino anaysis for %i runs." % len(retino_rundirs))
    for retino_rundir in retino_rundirs:
        # extract
        retino_opts = ['-i', animalid, '-S', session, '-A', fov, '-g', gap_niterations, '-a', np_niterations, '--new', '--masks']
        retino.do_analysis(retino_opts)

    # Do RF fits
    if int(session) < 20190511 and 'gratings' in experiment_types:
            rf_runs = ['rfs']
    else:
        rf_runs = [r for r in experiment_types if 'rfs' in r]

    for rf_run in rf_runs:
        print("[%s] 6a.  Fitting RF runs." % rf_run)
        # fit RFs
        res_, fov_ = fitrfs.fit_2d_receptive_fields(animalid, session, fov, rf_run, traceid, fit_thr=fit_thr, 
                             make_pretty_plots=True)

        # evaluate  
        print("[%s] 6b. Evaluating RF fits." % rf_run)
        devs_ = evalrfs.do_rf_fits_and_evaluation(animalid, session, fov, rfname=rf_run,
                              traceid=traceid, response_type=response_type, fit_thr=fit_thr, n_processes=n_processes) 

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
        bootr_, fitparams = exp.get_tuning(n_processes=n_processes, make_plots=True, create_new=True)
        evalosi_, goodfits = exp.evaluate_fits(bootr_, fitparams, goodness_thr=osi_thr, make_plots=True)
        print("--- done: %i cells with good fits (thr %.2f)" % (len(goodfits), osi_thr)
    
    print("********************************")
    print("Finished.")
    print("********************************")


def main(options):
    redo_manual_extraction(options)


if __name__ == '__main__':
    main(sys.argv[1:])
    


