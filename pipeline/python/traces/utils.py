#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 20:23:06 2018

@author: juliana
"""
import matplotlib
matplotlib.use('Agg')
import os
import sys
import json
import optparse
import operator
import h5py
import pprint
import itertools
import time
import shutil
import datetime
import traceback

from scipy import stats
import pandas as pd
import seaborn as sns
import pylab as pl
import numpy as np
from pipeline.python.utils import natural_keys, hash_file_read_only, print_elapsed_time, hash_file
pp = pprint.PrettyPrinter(indent=4)


def get_metric_set(traceid_dir, filter_pupil=True, pupil_size_thr=None,
                       pupil_dist_thr=None, pupil_max_nblinks=None, auto=False):

    metrics_dir = os.path.join(traceid_dir, 'metrics')
    metric_list = [m for m in os.listdir(metrics_dir) if os.path.isdir(os.path.join(metrics_dir, m))]
    if filter_pupil is True:
        metric_desc_base = 'pupil_size%i-dist%i-blinks%i_' % (pupil_size_thr, pupil_dist_thr, pupil_max_nblinks)
    else:
        metric_desc_base = 'unfiltered_'

    # First check for requested metrics set:
    metric_matches = [m for m in metric_list if metric_desc_base in m]
    try:
        assert len(metric_matches) == 1, "Unable to find unique metrics set with base: %s" % metric_desc_base
        selected_metric = metric_matches[0]
        if auto is True:
            return selected_metric
        else:
            while True:
                # Get confirmation first:
                confirm = raw_input('Selected metric: %s\nUse?  Press <Y> to confirm, <n> to reset: ' % selected_metric)
                if confirm == 'Y':
                    break
                elif confirm == 'n':
                    selected_metric = user_select_metric(metrics_dir)
                    break

    except Exception as e:
        if auto is True:
            # Sort by modified date, and select most-recent:
            metric_list.sort(key=lambda s: os.path.getmtime(os.path.join(metrics_dir, s)), reverse=True)
            selected_metric = metric_list[0]
            print "Using most recent metric set: %s" % selected_metric
            return selected_metric
        else:
            selected_metric = user_select_metric(metrics_dir)

    return selected_metric



def user_select_metric(metrics_dir):
    # Load particular metrics set:
    metric_list = [m for m in os.listdir(metrics_dir) if os.path.isdir(os.path.join(metrics_dir, m))]

    while True:
        print "Found %i metric sets:" % len(metric_list)
        for mi, metric in enumerate(metric_list):
            print mi, metric
        user_choice = input('Select IDX of metric set to view: ')
        selected_metric = metric_list[int(user_choice)]
        print "Viewing metric: %s" % selected_metric
        with open(os.path.join(metrics_dir, selected_metric, 'pupil_params.json'), 'r') as f:
            pupil_params = json.load(f)
        pp.pprint(pupil_params)
        confirm = raw_input("\nUse this metric? Select <Y> to use, or enter to try again: ")
        if confirm == 'Y':
            break

    return selected_metric


#%
def load_TID(run_dir, trace_id):
    run = os.path.split(run_dir)[-1]
    trace_basedir = os.path.join(run_dir, 'traces')
    try:
        tracedict_path = os.path.join(trace_basedir, 'traceids_%s.json' % run)
        with open(tracedict_path, 'r') as tr:
            tracedict = json.load(tr)
        TID = tracedict[trace_id]
        print "USING TRACE ID: %s" % TID['trace_id']
        pp.pprint(TID)
    except Exception as e:
        print "Unable to load TRACE params info: %s:" % trace_id
        print "Aborting with error:"
        print e

    return TID


def get_frame_info(run_dir):
    si_info = {}

    run = os.path.split(run_dir)[-1]
    runinfo_path = os.path.join(run_dir, '%s.json' % run)
    with open(runinfo_path, 'r') as fr:
        runinfo = json.load(fr)
    nfiles = runinfo['ntiffs']
    file_names = sorted(['File%03d' % int(f+1) for f in range(nfiles)], key=natural_keys)

    # Get frame_idxs -- these are FRAME indices in the current .tif file, i.e.,
    # removed flyback frames and discard frames at the top and bottom of the
    # volume should not be included in the indices...
    frame_idxs = runinfo['frame_idxs']
    if len(frame_idxs) > 0:
        print "Found %i frames from flyback correction." % len(frame_idxs)
    else:
        frame_idxs = np.arange(0, runinfo['nvolumes'] * len(runinfo['slices']))

    ntiffs = runinfo['ntiffs']
    file_names = sorted(['File%03d' % int(f+1) for f in range(ntiffs)], key=natural_keys)
    volumerate = runinfo['volume_rate']
    framerate = runinfo['frame_rate']
    nvolumes = runinfo['nvolumes']
    nslices = int(len(runinfo['slices']))
    nchannels = runinfo['nchannels']


    nslices_full = int(round(runinfo['frame_rate']/runinfo['volume_rate']))
    nframes_per_file = nslices_full * nvolumes

    # =============================================================================
    # Get VOLUME indices to assign frame numbers to volumes:
    # =============================================================================
    vol_idxs_file = np.empty((nvolumes*nslices_full,))
    vcounter = 0
    for v in range(nvolumes):
        vol_idxs_file[vcounter:vcounter+nslices_full] = np.ones((nslices_full, )) * v
        vcounter += nslices_full
    vol_idxs_file = [int(v) for v in vol_idxs_file]


    vol_idxs = []
    vol_idxs.extend(np.array(vol_idxs_file) + nvolumes*tiffnum for tiffnum in range(nfiles))
    vol_idxs = np.array(sorted(np.concatenate(vol_idxs).ravel()))

    si_info['nslices_full'] = nslices_full
    si_info['nframes_per_file'] = nframes_per_file
    si_info['vol_idxs'] = vol_idxs
    si_info['volumerate'] = volumerate
    si_info['framerate'] = framerate
    si_info['nslices'] = nslices
    si_info['nchannels'] = nchannels
    si_info['ntiffs'] = ntiffs
    si_info['frames_tsec'] = runinfo['frame_tstamps_sec']
    si_info['nvolumes'] = nvolumes

    return si_info