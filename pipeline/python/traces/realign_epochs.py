#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  21 12:17:55 2020

@author: julianarhee
"""

import h5py
import glob
import os
import json
import copy
import traceback
import re
import optparse
import sys

import pandas as pd
import numpy as np
from pipeline.python.utils import natural_keys, get_frame_info
from pipeline.python.paradigm import align_acquisition_events as alignacq
from pipeline.python.traces import trial_alignment as talignment
from pipeline.python.paradigm.plot_responses import make_clean_psths
from pipeline.python.classifications import experiment_classes as cutils



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

    parser.add_option('-p', '--iti-pre', action='store', dest='iti_pre', default=1.0, 
                      help="pre-stim amount in sec (default: 1.0)")
    parser.add_option('-P', '--iti-post', action='store', dest='iti_post', default=1.0, 
                      help="post-stim amount in sec (default: 1.0)")

    parser.add_option('--plot', action='store_true', dest='plot_psth', default=False, 
                      help="set flat to plot psths (specify row, col, hue)")
    parser.add_option('-r', '--rows', action='store', dest='rows',
                          default=None, help='Transform to plot along ROWS (only relevant if >2 trans_types)')
    parser.add_option('-c', '--columns', action='store', dest='columns',
                          default=None, help='Transform to plot along COLUMNS')
    parser.add_option('-H', '--hue', action='store', dest='subplot_hue',
                          default=None, help='Transform to plot by HUE within each subplot')
    parser.add_option('-d', '--response', action='store', dest='response_type',
                          default='dff', help='Traces to plot (default: dff)')

    parser.add_option('-f', '--filetype', action='store', dest='filetype',
                          default='svg', help='File type for images [default: svg]')

    (options, args) = parser.parse_args(options)

    return options

def parse_trial_epochs(animalid, session, fov, experiment, traceid, 
                        iti_pre=1.0, iti_post=1.0, rootdir='/n/coxfs01/2p-data'):
    fovdir = os.path.join(rootdir, animalid, session, fov)
    rundirs = [os.path.split(p)[0] for p in glob.glob(os.path.join(fovdir, '%s_run*' % experiment, 'paradigm'))]
    for rundir in rundirs:
        si_info = get_frame_info(rundir) 
        paradigm_dir = os.path.join(rundir, 'paradigm')  
        trial_info = alignacq.get_alignment_specs(paradigm_dir, si_info, iti_pre=iti_pre, iti_post=iti_post)
        
        # Save alignment info
        traceid_dir = glob.glob(os.path.join(rundir, 'traces', '%s*' % traceid))
        assert len(traceid_dir)==1, "More than 1 tracedid path found..."
        traceid_dir = traceid_dir[0]
        alignment_info_filepath = os.path.join(traceid_dir, 'event_alignment.json')
        with open(alignment_info_filepath, 'w') as f:
            json.dump(trial_info, f, sort_keys=True, indent=4)       

        # Update extraction_params.json
        extraction_info_filepath = os.path.join(traceid_dir, 'extraction_params.json')
        if os.path.exists(extraction_info_filepath):
            with open(extraction_info_filepath, 'r') as f:
                eparams = json.load(f)
            for k, v in trial_info.items():
                if k in eparams:
                    eparams[k] = v
        else:
            eparams = trial_info
        with open(extraction_info_filepath, 'w') as f:
            json.dump(eparams, f, sort_keys=True, indent=4)       

        # Get parsed frame indices
        parsed_frames_filepath = alignacq.assign_frames_to_trials(si_info, trial_info, paradigm_dir, create_new=True)

    print("Done!")
   
def align_traces(animalid, session, fov, experiment, traceid, rootdir='/n/coxfs01/2p-data'):

    talignment.aggregate_experiment_runs(animalid, session, fov, experiment, traceid=traceid)

#    if experiment == 'blobs':
#        exp = cutils.Objects(animalid, session, fov, traceid=traceid, rootdir=rootdir)
#    elif experiment == 'gratings':
#        exp = cutils.Gratings(animalid, session, fov, traceid=traceid, rootdir=rootdir)
#
#    exp.load(trace_type='dff', update_self=True, make_equal=False, create_new=True)
    print("Aligned traces!") 


def main(options):
    opts = extract_options(options)
    animalid = opts.animalid
    session = opts.session
    fov = opts.fov
    experiment = opts.experiment
    traceid = opts.traceid
    iti_pre = float(opts.iti_pre)
    iti_post = float(opts.iti_post)
    rootdir = opts.rootdir
    plot_psth = opts.plot_psth    
   
    print("1. Parsing") 
    parse_trial_epochs(animalid, session, fov, experiment, traceid, 
                        iti_pre=iti_pre, iti_post=iti_post)

    print("2. Aligning - %s" % experiment)

    align_traces(animalid, session, fov, experiment, traceid, rootdir=rootdir)

    if plot_psth:
        print("3. Plotting")
        row_str = opts.rows
        col_str = opts.columns
        hue_str = opts.subplot_hue
        response_type = opts.response_type
        file_type = opts.filetype

        plot_opts = ['-i', animalid, '-S', session, '-A', fov, '-t', traceid, '-R', 'combined_%s_static' % experiment, 
                     '--shade', '-r', row_str, '-c', col_str, '-H', hue_str, '-d', response_type, '-f', file_type]
        make_clean_psths(plot_opts) 
    
if __name__ == '__main__':
    main(sys.argv[1:])
    


