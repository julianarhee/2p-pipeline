#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 18:32:03 2018

@author: juliana
"""

import os
import optparse
import numpy as np

from pipeline.python.traces import get_traces as traces
from pipeline.python.paradigm import tifs_to_data_arrays as tdf
from pipeline.python.paradigm import plot_responses as rplot
from pipeline.python.paradigm import process_mw_files as mw
from pipeline.python.paradigm import extract_stimulus_events as mwe



def create_function_opts(rootdir='', animalid='', session='', acquisition='',
                         run='', traceid=''):
    options = ['-D', rootdir, '-i', animalid, '-S', session, '-A', acquisition,
            '-R', run, '-t', traceid]
    
    return options

def parse_mw(mw_opts):
    
    mw_optsE = mwe.extract_options(mw_opts)
    retinobar = False
    if 'retino' in mw_optsE.run:
        mw_opts.extend(['--retinobar'])
        retinobar = True
        
    paradigm_outdir = mw.parse_mw_trials(mw_opts)
    print "----------------------------------------"
    print "Extracted MW events!"
    print "Outfile saved to:\n%s" % paradigm_outdir
    print "----------------------------------------"
    
    if retinobar is False: 
        run_dir = os.path.join(mw_optsE.rootdir, mw_optsE.animalid, mw_optsE.session, mw_optsE.acquisition, mw_optsE.run)
        parsed_run_outfile = mwe.parse_acquisition_events(run_dir, blank_start=mw_optsE.blank_start)
        print "----------------------------------------"
        print "ACQUISITION INFO saved to:\n%s" % parsed_run_outfile
        print "----------------------------------------"

    return paradigm_outdir


def get_raw_traces(trace_opts):
    TID, RID, si_info, filearrays_dir, _, _ = traces.extract_traces(trace_opts)
    print "DONE extracting traces!"
    print "Output saved to:\n---> %s" % filearrays_dir

    return TID, RID, si_info, filearrays_dir


#%%
    
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
    
    
    
    (options, args) = parser.parse_args(options)
    if options.slurm is True and '/n/coxfs01' not in options.rootdir:
        options.rootdir = '/n/coxfs01/2p-data'
    return options

#%%

    optsE = extract_options(options)
    np_factor = optsE.np_factor
    np_annulus = optsE.np_annulus
    baseline_quantile = optsE.baseline_quantile
    smoothing_window = optsE.smoothing_window
    smoothing_fraction = optsE.smoothing_fraction
    pre_iti = optsE.pre_iti
    post_iti = optsE.post_iti
    
    data_type = optsE.data_type
    psth_rows = optsE.psth_rows
    psth_cols = optsE.psth_cols
    psth_hue = optsE.psth_hue
    
    # Parse MW:
    mw_opts = create_function_opts(options)
    if optsE.dynamic:
        mw_opts.extend(['--dynamic'])
        
    # =========================================================================
    # Extract traces from each TIF.
    # =========================================================================
    trace_opts = create_function_opts(options)
    trace_opts.extend(['-c', np_factor, '-a', np_annulus, '--neuropil'])

        
    # =========================================================================        
    # Create data arrays for each file, then concatenate into one DataFrame.
    # =========================================================================

    darray_opts = create_function_opts(options)
    darray_opts.extend(['-q', baseline_quantile, '-w', smoothing_window, '--frac=%.2f' % smoothing_fraction,
                        '--iti=%.2f' % pre_iti, '--post=%.2f' % post_iti])
        
    data_fpath = tdf.create_rdata_array(darray_opts)
    print "*******************************************************************"
    print "DONE!"
    print "New ROIDATA array saved to: %s" % data_fpath
    print "*******************************************************************"

    # =========================================================================    
    # Plot PSTHs:
    # =========================================================================

    psth_opts = create_function_opts(options)
    psth_opts.extend(['-d', data_type])
    if psth_rows is not None:
        psth_opts.extend(['-r', psth_rows])
    if psth_cols is not None:
        psth_opts.extend(['-c', psth_cols])
    if psth_hue is not None:
        psth_opts.extend(['-H', psth_hue])
        
    psth_dir = rplot.make_clean_psths(psth_opts)
    
    