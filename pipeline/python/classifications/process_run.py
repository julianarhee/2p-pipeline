#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 12:53:03 2018

@author: juliana
"""
import optparse
import os
import glob
import shutil
import copy
import sys
import pprint
pp = pprint.PrettyPrinter(indent=4)
import cPickle as pkl
import numpy as np
import pandas as pd

from pipeline.python.visualization import get_session_summary as ss
from pipeline.python.paradigm import utils as util
from pipeline.python.utils import natural_keys, label_figure
from pipeline.python.classifications import test_responsivity as resp #import calculate_roi_responsivity, group_roidata_stimresponse, find_barval_index

animalid = optsE.animalid
session = optsE.session
acquisition = optsE.acquisition
run_list = optsE.run_list
traceid_list = optsE.traceid_list

def process_run_data(animalid, session, acquisition, run_list, traceid_list, 
                         rootdir='/n/coxfs01/2p-data',
                         stimtype='', data_type='corrected', metric='zscore',
                         make_equal=True, create_new=False, nproc=2):

    acquisition_dir = os.path.join(rootdir, animalid, session, acquisition)
    if stimtype == '':
        stimtype = run_list[0].split('_')[0]
        
    traceid_dir = ss.get_traceid_dir_from_lists(acquisition_dir, run_list, traceid_list, stimtype=stimtype, make_equal=make_equal, create_new=create_new)

    
    data_fpath = glob.glob(os.path.join(traceid_dir, 'data_arrays', '*.npz'))[0]
    dataset = np.load(data_fpath)
    
    rinfo = dataset['run_info'] if isinstance(dataset['run_info'], dict) else dataset['run_info'][()]
    pp.pprint(rinfo['ntrials_by_cond'])
    if len(list(set([v for k,v in rinfo['ntrials_by_cond'].items()]))) > 1:
        dataset, data_fpath = util.get_equal_reps(dataset, data_fpath)


    traceid = os.path.split(traceid_dir)[-1]
    run = os.path.split(traceid_dir.split('/traces/')[0])[-1]
    print "Run: %s | traceid: %s" % (run, traceid)
    
    # Get stats for responsivity and selectivity:
    roistats = ss.get_roi_stats(rootdir, animalid, session, acquisition, 
                                   run, traceid, create_new=create_new, nproc=nproc) #blobs_traceid.split('_')[0])
    # Group data by ROIs:
    roidata, labels_df, sconfigs = ss.get_data_and_labels(dataset, data_type=data_type)
    df_by_rois = resp.group_roidata_stimresponse(roidata, labels_df)
    data, transforms_tested = ss.get_object_transforms(df_by_rois, roistats, sconfigs, metric=metric)
     
    object_dict = {'source': run,
                   'traceid': traceid,
                   'data_fpath': data_fpath,
                   'roistats': roistats,
                   'roidata': df_by_rois,
                   'sconfigs': sconfigs,
                   'transforms': data,
                   'transforms_tested': transforms_tested,
                   'metric': metric}
    
    processed_run_fpath = os.path.join(traceid_dir, 'processed_run.pkl')
    with open(processed_run_fpath, 'wb') as f:
        pkl.dump(object_dict, f, protocol=pkl.HIGHEST_PROTOCOL)



def extract_options(options):

    def comma_sep_list(option, opt, value, parser):
        setattr(parser.values, option.dest, value.split(','))

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

    # Run specific info:
    #parser.add_option('-g', '--gratings-traceid', dest='gratings_traceid_list', default=[], action='append', nargs=1, help="traceid for GRATINGS [default: []]")
    parser.add_option('-t', '--traceid', dest='traceid_list', default=[], type='string', action='callback', callback=comma_sep_list, help="traceids for corresponding runs [default: []]")

    parser.add_option('-R', '--run', dest='run_list', default=[], type='string', action='callback', callback=comma_sep_list, help='list of run IDs [default: []')
    parser.add_option('-s', '--stim', dest='stimtype', default='', action='store', help='stimulus type (must be: gratings, blobs, or objects in run name)')

    parser.add_option('-d', '--dtype', dest='data_type', default='corrected', action='store', help='data_type (corrected, dff, etc.) for raw data (default: corrected)')
    parser.add_option('-m', '--metric', dest='metric', default='zscore', action='store', help='metric (e.g., zscore, meastim) to use for first-pass visualiztion of stim responses')
    
    parser.add_option('--equal', action='store_true', dest='make_equal', default=False, help="set if make ntrials per cond equal (need for roistats)")
    parser.add_option('--new', action='store_true', dest='create_new', default=False, help="set to re-extract data summaries")
    parser.add_option('-n', '--nproc', action='store', dest='nprocesses', default=4, help="N processes if running in par (default=4)")

    (options, args) = parser.parse_args(options)
    if options.slurm is True and '/n/coxfs01' not in options.rootdir:
        options.rootdir = '/n/coxfs01/2p-data'
    return options


#%%
    
options = ['-D', '/n/coxfs01/2p-data', '-i', 'CE077', '-S', '20180521', '-A', 'FOV2_zoom1x',
           '-R', 'blobs_run1,blobs_run2',
           '-t', 'traces002,traces002',
           '-s', 'blobs',
           '--new',
           '-n', 2
           ]

#metric = 'zscore'
#data_type = 'corrected'

def main(options):
    optsE = extract_options(options)

    processed_run_fpath = process_run_data(optsE.animalid, optsE.session, optsE.acquisition, optsE.run_list, optsE.traceid_list, rootdir=optsE.rootdir,
                                           stimtype=optsE.stimtype, data_type=optsE.data_type, metric=optsE.metric,
                                           make_equal=optsE.make_equal, create_new=optsE.create_new, nproc=int(optsE.nprocesses))

    print "Finished processing data!"
    print "Saved processed run to:\n--> %s" % processed_run_fpath
    
if __name__ == '__main__':
    main(sys.argv[1:]) 
