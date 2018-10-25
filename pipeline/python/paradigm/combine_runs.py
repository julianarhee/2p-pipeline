#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 15:30:42 2018

@author: juliana
"""

import optparse
import os
import glob
import shutil
import copy

import cPickle as pkl
import numpy as np
import pandas as pd

from pipeline.python.visualization import get_session_summary as ss
from pipeline.python.paradigm import utils as util
from pipeline.python.utils import natural_keys, label_figure
from pipeline.python.classifications import test_responsivity as resp 

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
    
    (options, args) = parser.parse_args(options)
    if options.slurm is True and '/n/coxfs01' not in options.rootdir:
        options.rootdir = '/n/coxfs01/2p-data'
    return options


#%%
    
options = ['-D', '/n/coxfs01/2p-data', '-i', 'JC022', '-S', '20181016', '-A', 'FOV1_zoom2p7x',
           '-R', 'blobs,blobs_run2,blobs_run3,blobs_run4,blobs_run5', 
           '-t', 'traces001,traces001,traces001,traces001,traces001',
           '-s', 'blobs'
           ]

optsE = extract_options(options)
acquisition_dir = os.path.join(optsE.rootdir, optsE.animalid, optsE.session, optsE.acquisition)
run_list = optsE.run_list
traceid_list = optsE.traceid_list
stimtype = optsE.stimtype
if stimtype == '':
    stimtype = run_list[0].split('_')[0]
    
traceid_dir = ss.get_traceid_dir_from_lists(acquisition_dir, run_list, traceid_list, stimtype=stimtype, make_equal=False, create_new=False)

#%%


data_fpath = glob.glob(os.path.join(traceid_dir, 'data_arrays', '*.npz'))[0]

dset = np.load(data_fpath)

sdf = pd.DataFrame(dset['sconfigs'][()]).T

# Take subset of data
configs_included = [c for c in sdf.index.tolist() if c not in sdf[sdf['morphlevel']==106].index.tolist()]
kept_ixs = np.array([fi for fi,cfg in enumerate(dset['ylabels']) if cfg in configs_included])


sconfigs = dict((cname, cvalue) for cname, cvalue in dset['sconfigs'][()].items() if cname in configs_included)
trial_col_ix = [col for col in dset['labels_columns']].index('trial')

all_trials = sorted(list(set(dset['labels_data'][:, trial_col_ix])), key=natural_keys) # trials aren't incrementing in 1-intervals... (20181016 FOV1 data) - but they ARE in order of size
kept_trials = sorted(list(set(dset['labels_data'][kept_ixs, trial_col_ix])), key=natural_keys)
trial_ixs = np.array([all_trials.index(t) for t in kept_trials])

run_info = copy.copy(dset['run_info'][()])
run_info['condition_list'] = sorted(sconfigs.keys(), key=natural_keys)
updated_ntrials = dict((cfg, ntrials) for cfg, ntrials in run_info['ntrials_by_cond'].items() if cfg in configs_included)
run_info['ntrials_by_cond'] = updated_ntrials
run_info['ntrials_total'] = len(trial_ixs)

#shutil.move(data_fpath, '%s.orig' % data_fpath)


np.savez(data_fpath, corrected=dset['corrected'][kept_ixs, :], 
                     ylabels = dset['ylabels'][kept_ixs],
                     labels_columns=dset['labels_columns'],
                     labels_data=dset['labels_data'][kept_ixs, :],
                     meanstim=dset['meanstim'][trial_ixs, :],
                     zscore=dset['zscore'][trial_ixs, :],
                     sconfigs=sconfigs,
                     run_info=run_info)

del dset

#%%


dataset = np.load(data_fpath)

# Get sorted ROIs:
traceid = os.path.split(traceid_dir)[-1]
run = 'combined_blobs_static'
metric = 'zscore'

roistats = ss.get_roi_stats(optsE.rootdir, optsE.animalid, optsE.session, optsE.acquisition, 
                               run, traceid, create_new=False, nproc=4) #blobs_traceid.split('_')[0])


# Group data by ROIs:
roidata, labels_df, sconfigs = ss.get_data_and_labels(dataset, data_type='corrected')
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