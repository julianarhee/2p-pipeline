#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 12:42:10 2017

@author: julianarhee
"""

import os
import json
import pprint
import re
import pkg_resources
import optparse
import sys
import hashlib
import shutil
from pipeline.python.utils import write_dict_to_json, get_tiff_paths
import numpy as np
from checksumdir import dirhash

pp = pprint.PrettyPrinter(indent=4)

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]

#%%
def extract_options(options):
    choices_sourcetype = ('raw', 'mcorrected', 'bidi')
    default_sourcetype = 'mcorrected'

    parser = optparse.OptionParser()

    # PATH opts:
    parser.add_option('-D', '--root', action='store', dest='rootdir', default='/nas/volume1/2photon/data', help='data root dir (root project dir containing all animalids) [default: /nas/volume1/2photon/data, /n/coxfs01/2pdata if --slurm]')
    parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')
    parser.add_option('-S', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID')
    parser.add_option('-A', '--acq', action='store', dest='acquisition', default='FOV1', help="acquisition folder (ex: 'FOV1_zoom3x') [default: FOV1]")
    parser.add_option('-R', '--run', action='store', dest='run', default='', help="name of run dir containing tiffs to be processed (ex: gratings_phasemod_run1)")
    parser.add_option('--slurm', action='store_true', dest='slurm', default=False, help="Set if running as SLURM job on Odyssey")

    parser.add_option('-s', '--tiffsource', action='store', dest='tiffsource', default=None, help="Name of folder containing tiffs to be processed (ex: processed001). Should be child of <run>/processed/")
    parser.add_option('-t', '--source-type', type='choice', choices=choices_sourcetype, action='store', dest='sourcetype', default=default_sourcetype, help="Type of tiff source. Valid choices: %s [default: %s]" % (choices_sourcetype, default_sourcetype))
    parser.add_option('-o', '--roi-set', action='store', dest='roi_name', default=None, help="Roi id name (e.g., 'rois001' or 'rois011')")
    parser.add_option('-c', '--channel', action='store', dest='signal_channel', default=1, help="Signal channel [default: 1]")
    parser.add_option('--default', action='store_true', dest='default', default='store_false', help="Use all DEFAULT params, for params not specified by user (prevent interactive)")

    (options, args) = parser.parse_args(options)

    if options.slurm is True:
        if 'coxfs01' not in options.rootdir:
            options.rootdir = '/n/coxfs01/2p-data'

    return options

#%%
def create_tid(options):

    options = extract_options(options)

    # Set USER INPUT options:
    rootdir = options.rootdir
    animalid = options.animalid
    session = options.session
    acquisition = options.acquisition
    run = options.run

    tiffsource = options.tiffsource
    sourcetype = options.sourcetype
    roi_name = options.roi_name

    auto = options.default

    signal_channel = options.signal_channel

    # Get paths to tiffs from which to create ROIs:
    tiffpaths = get_tiff_paths(rootdir=rootdir, animalid=animalid, session=session,
                               acquisition=acquisition, run=run,
                               tiffsource=tiffsource, sourcetype=sourcetype, auto=auto)

    # Load specified ROI set (stored in session dir):
    session_dir = os.path.join(rootdir, animalid, session)
    with open(os.path.join(session_dir, 'ROIs', 'rids_%s.json' % session), 'r') as f:
        roidict = json.load(f)
    RID = roidict[roi_name]

    # Set trace dir:
    run_dir = os.path.join(session_dir, acquisition, run)
    trace_dir = os.path.join(run_dir, 'traces')
    if not os.path.exists(trace_dir):
        os.makedirs(trace_dir)

    # Create trace-params dict with source and ROI set:
    tiff_sourcedir = os.path.split(tiffpaths[0])[0]
    # Check if there are any TIFFs to exclude:
    orig_roi_dst = RID['DST']
    if rootdir not in orig_roi_dst:
        orig_root = orig_roi_dst.split('/%s/%s' % (animalid, session))[0]
        print "ORIG root:", orig_root
        rparams_dir = orig_roi_dst.replace(orig_root, rootdir)
    else:
        rparams_dir = orig_roi_dst
    print "Loading PARAM info... Looking in ROI dst dir: %s" % rparams_dir
    with open(os.path.join(rparams_dir, 'roiparams.json'), 'r') as f:
        roiparams = json.load(f)
    excluded_tiffs = roiparams['excluded_tiffs']
    
    PARAMS = get_params_dict(signal_channel, tiff_sourcedir, RID, excluded_tiffs=excluded_tiffs)

    # Create TRACE ID (TID):
    TID = initialize_tid(PARAMS, run_dir, auto=auto)

    # Create TRACE output directory:
    trace_name = '_'.join((TID['trace_id'], TID['trace_hash']))
    curr_trace_dir = os.path.join(trace_dir, trace_name)
    if not os.path.exists(curr_trace_dir):
        os.makedirs(curr_trace_dir)

    # Check RID fields to include RID hash, and save updated to ROI DICT:
    if TID['trace_hash'] not in TID['DST']:
        TID['DST'] = TID['DST'] + '_' + TID['trace_hash']
    update_tid_records(TID, run_dir)

    # Write to tmp_rid folder in current run source:
    tmp_tid_dir = os.path.join(trace_dir, 'tmp_tids')
    if not os.path.exists(tmp_tid_dir):
        os.makedirs(tmp_tid_dir)

    tmp_tid_path = os.path.join(tmp_tid_dir, 'tmp_tid_%s.json' % TID['trace_hash'])
    write_dict_to_json(TID, tmp_tid_path)

    print "********************************************"
    print "Created params for TRACE SET, hash: %s" % TID['trace_hash']
    print "********************************************"

    return TID


def get_params_dict(signal_channel, tiff_sourcedir, RID, excluded_tiffs=[]):
    PARAMS = dict()
    PARAMS['tiff_source'] = tiff_sourcedir 
    PARAMS['excluded_tiffs'] = excluded_tiffs
    PARAMS['signal_channel'] = signal_channel
    PARAMS['roi_id'] = RID['roi_id']
    PARAMS['rid_hash'] = RID['rid_hash']
    PARAMS['roi_type'] = RID['roi_type']
    PARAMS['hashid'] = hashlib.sha1(json.dumps(PARAMS, sort_keys=True)).hexdigest()[0:6]

    return PARAMS


def initialize_tid(PARAMS, run_dir, auto=False):

    print "************************"
    print "Initializing trace ID."
    print "************************"
    trace_id = get_trace_id(PARAMS, run_dir, auto=auto)

    tid = dict()
    version = pkg_resources.get_distribution('pipeline').version

    tid['version'] = version
    tid['trace_id'] = trace_id
    tid['PARAMS'] = PARAMS
    tid['SRC'] = PARAMS['tiff_source']
    tid['DST'] = os.path.join(run_dir, 'traces', trace_id)

    tid['trace_hash'] = hashlib.sha1(json.dumps(tid, sort_keys=True)).hexdigest()[0:6]

    return tid


def load_tracedict(run_dir):

    tracedict = None

    run = os.path.split(run_dir)[1]
    tracedict_filepath = os.path.join(run_dir, 'traces', 'traceids_%s.json' % run)

    # Load analysis "processdict" file:
    if os.path.exists(tracedict_filepath):
        print "exists!"
        with open(tracedict_filepath, 'r') as f:
            tracedict = json.load(f)

    return tracedict


def get_trace_id(PARAMS, run_dir, auto=False):

    trace_id = None

    print "********************************"
    print "Checking previous TRACE IDs..."
    print "********************************"

    # Load previously created PIDs:
    tracedict = load_tracedict(run_dir)

    #Check current PARAMS against existing PId params by hashid:
    if tracedict is None or len(tracedict.keys()) == 0:
        tracedict = dict()
        existing_tids = []
        is_new_tid = True
        print "No existing TRACE IDs found."
    else:
        existing_tids = sorted([str(k) for k in tracedict.keys()], key=natural_keys)
        print existing_tids
        matching_tids = sorted([tid for tid in existing_tids if tracedict[tid]['PARAMS']['hashid'] == PARAMS['hashid']], key=natural_keys)
        is_new_tid = False
        if len(matching_tids) > 0:
            while True:
                print "WARNING **************************************"
                print "Current param config matches existing trace ID:"
                for tidx, tid in enumerate(matching_tids):
                    print "******************************************"
                    print tidx, tid
                    pp.pprint(tracedict[tid])
                if auto is True:
                    check_tidx = ''
                else:
                    check_tidx = raw_input('Enter IDX of trace id to re-use, or hit <ENTER> to create new: ')
                if len(check_tidx) == 0:
                    is_new_tid = True
                    break
                else:
                    confirm_reuse = raw_input('Re-use trace ID %s? Press <Y> to confirm, any key to try again:' % existing_tids[int(check_tidx)])
                    if confirm_reuse == 'Y':
                        is_new_tid = False
                        break
        else:
            is_new_tid = True

    if is_new_tid is True:
        # Create new PID by incrementing num of process dirs found:
        trace_id = 'traces%03d' % int(len(existing_tids)+1)
        print "Creating NEW trace ID: %s" % trace_id
    else:
        # Re-using an existing PID:
        trace_id = existing_tids[int(check_tidx)]
        print "Reusing existing trace id: %s" % trace_id

    return trace_id


def update_tid_records(TID, run_dir):

    print "************************"
    print "Updating JSONS..."
    print "************************"
    run = os.path.split(run_dir)[1]
    tracedict_filepath = os.path.join(run_dir, 'traces', 'traceids_%s.json' % run)

    if os.path.exists(tracedict_filepath):
        with open(tracedict_filepath, 'r') as f:
            tracedict = json.load(f)
    else:
        tracedict = dict()

    trace_id = TID['trace_id']
    tracedict[trace_id] = TID

    #% Update Process Info DICT:
    write_dict_to_json(tracedict, tracedict_filepath)

    print "Trace Set Info UPDATED."


def post_tid_cleanup(run_dir, trace_hash):

    run = os.path.split(run_dir)[1]
    trace_dir = os.path.join(run_dir, 'traces')
    print "Cleaning up TID info: %s" % trace_hash
    tmp_tid_dir = os.path.join(trace_dir, 'tmp_tids')
    tmp_tid_fn = 'tmp_tid_%s.json' % trace_hash
    tid_path = os.path.join(tmp_tid_dir, tmp_tid_fn)
    with open(tid_path, 'r') as f:
        TID = json.load(f)

    tracedict_fn = 'traceids_%s.json' % run
    # UPDATE PID entry in dict:
    with open(os.path.join(trace_dir, tracedict_fn), 'r') as f:
        tracedict = json.load(f)
    trace_id = [p for p in tracedict.keys() if tracedict[p]['trace_hash'] == trace_hash][0]
    tracedict[trace_id] = TID

    # Save updated PID dict:
    path_to_tracedict = os.path.join(trace_dir, tracedict_fn)
    write_dict_to_json(tracedict, path_to_tracedict)

    finished_dir = os.path.join(tmp_tid_dir, 'completed')
    if not os.path.exists(finished_dir):
        os.makedirs(finished_dir)
    shutil.move(tid_path, os.path.join(finished_dir, tmp_tid_fn))
    print "Moved tmp tid file to completed."

def main(options):

    tid = create_tid(options)

    print "****************************************************************"
    print "Created TRACE ID."
    print "----------------------------------------------------------------"
    pp.pprint(tid)
    print "****************************************************************"


if __name__ == '__main__':
    main(sys.argv[1:])














