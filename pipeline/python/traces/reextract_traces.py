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

import numpy as np
import pylab as pl

from pipeline.python.utils import natural_keys, label_figure
from pipeline.python.traces import realign_epochs as realign
from pipeline.python.traces import remake_neuropil_masks as rmasks



def redo_manual_extraction(options):

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
    
    # Get runs to extract
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


    print("********************************")
    print("Finished.")
    print("********************************")

