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
import pandas as pd
import optparse
import sys
import hashlib
import copy
from pipeline.python.utils import write_dict_to_json
from caiman.source_extraction.cnmf import utilities as cmu
import caiman as cm
import numpy as np
from checksumdir import dirhash
from stat import S_IREAD, S_IRGRP, S_IROTH, S_IWRITE, S_IWGRP, S_IWOTH
import shutil

pp = pprint.PrettyPrinter(indent=4)

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]


def extract_options(options):
    choices_sourcetype = ('raw', 'mcorrected', 'bidi')
    default_sourcetype = 'mcorrected'

    choices_roi = ('caiman2D', 'manual2D_circle', 'manual2D_square', 'manual2D_polygon')
    default_roitype = 'caiman2D'

    parser = optparse.OptionParser()

    # PATH opts:
    parser.add_option('-R', '--root', action='store', dest='rootdir', default='/nas/volume1/2photon/data', help='data root dir (root project dir containing all animalids) [default: /nas/volume1/2photon/data, /n/coxfs01/2pdata if --slurm]')
    parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')

    # Set specific session/run for current animal:
    parser.add_option('-S', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID')
    parser.add_option('-A', '--acq', action='store', dest='acquisition', default='FOV1', help="acquisition folder (ex: 'FOV1_zoom3x') [default: FOV1]")
    parser.add_option('-r', '--run', action='store', dest='run', default='', help="name of run dir containing tiffs to be processed (ex: gratings_phasemod_run1)")
    parser.add_option('--default', action='store_true', dest='default', default='store_false', help="Use all DEFAULT params, for params not specified by user (no interactive)")

    parser.add_option('-s', '--tiffsource', action='store', dest='tiffsource', default=None, help="name of folder containing tiffs to be processed (ex: processed001). should be child of <run>/processed/")
    parser.add_option('-t', '--source-type', type='choice', choices=choices_sourcetype, action='store', dest='sourcetype', default=default_sourcetype, help="Type of tiff source. Valid choices: %s [default: %s]" % (choices_sourcetype, default_sourcetype))
    parser.add_option('-o', '--roi-type', type='choice', choices=choices_roi, action='store', dest='roi_type', default=default_roitype, help="Roi type. Valid choices: %s [default: %s]" % (choices_roi, default_roitype))

    parser.add_option('--slurm', action='store_true', dest='slurm', default=False, help="set if running as SLURM job on Odyssey")

    parser.add_option('--deconv', action='store', dest='nmf_deconv', default='oasis', help='method deconvolution if using cNMF [default: oasis]')
    parser.add_option('--gSig', action='store', dest='nmf_gsig', default=8, help='half size of neurons if using cNMF [default: 8]')
    parser.add_option('--K', action='store', dest='nmf_K', default=10, help='N expected components per patch [default: 10]')
    parser.add_option('--patch', action='store', dest='nmf_rf', default=30, help='Half size of patch if using cNMF [default: 30]')
    parser.add_option('--overlap', action='store', dest='nmf_stride', default=15, help='Amount of patch overlap if using cNMF [default: 15]')
    parser.add_option('--nmf-order', action='store', dest='nmf_p', default=2, help='Order of autoregressive system if using cNMF [default: 2]')


    (options, args) = parser.parse_args(options) 
    
    if options.slurm is True:
        if 'coxfs01' not in options.rootdir:
            options.rootdir = '/n/coxfs01/julianarhee/testdata'

    return options

def create_rid(options):
    
    options = extract_options(options)
    
    # Set USER INPUT options:
    rootdir = options.rootdir
    animalid = options.animalid
    session = options.session
    acquisition = options.acquisition
    run = options.run

    tiffsource = options.tiffsource
    sourcetype = options.sourcetype
    roi_type = options.roi_type

    auto = options.default

    # cNMF-specific opts:
    nmf_deconv = options.nmf_deconv
    nmf_gsig = [int(options.nmf_gsig), int(options.nmf_gsig)]
    nmf_K = int(options.nmf_K)
    nmf_rf = int(options.nmf_rf)
    nmf_stride = int(options.nmf_stride)
    nmf_p = int(options.nmf_p)

    session_dir = os.path.join(rootdir, animalid, session)
    roi_dir = os.path.join(session_dir, 'ROIs')
    if not os.path.exists(roi_dir):
        os.makedirs(roi_dir)
        
    # Get paths to tiffs from which to create ROIs:
    tiffpaths = get_tiff_paths(rootdir=rootdir, animalid=animalid, session=session,
                               acquisition=acquisition, run=run,
                               tiffsource=tiffsource, sourcetype=sourcetype)
    
    # Get roi-type specific options:
    if roi_type == 'caiman2D':
        print "Creating param set for caiman2D ROIs."
        movie_idxs = []
        roi_options = set_options_cnmf(rootdir=rootdir, animalid=animalid, session=session,
                                       acquisition=acquisition, run=run,
                                       movie_idxs=movie_idxs, method_deconv=nmf_deconv, K=nmf_K,
                                       gSig=nmf_gsig, rf=nmf_rf, stride=nmf_stride, p=nmf_p)
    elif 'manual' in roi_type:
        roi_options = set_options_manual(rootdir=rootdir, animalid=animalid, session=session,
                                         acquisition=acquisition, run=run,
                                         roi_type=roi_type)
    
    # Create roi-params dict with source and roi-options:
    tiff_sourcedir = os.path.split(tiffpaths[0])[0]
    print "SRC: %s, found %i tiffs." % (tiff_sourcedir, len(tiffpaths))
    PARAMS = get_params_dict(tiff_sourcedir, roi_options, roi_type=roi_type, mmap_dir=None, check_hash=False)
    
    # Create ROI ID (RID):
    RID = initialize_rid(PARAMS, session_dir)

    # Create ROI output directory:
    roi_name = '_'.join((RID['roi_id'], RID['rid_hash']))
    curr_roi_dir = os.path.join(roi_dir, roi_name)
    if not os.path.exists(curr_roi_dir):
        os.makedirs(curr_roi_dir)

    # Check RID fields to include RID hash, and save updated to ROI DICT:
    if RID['rid_hash'] not in RID['DST']:
        RID['DST'] = RID['DST'] + '_' + RID['rid_hash']
    update_roi_records(RID, session_dir)
    
    # Write to tmp_rid folder in current run source:
    tmp_rid_dir = os.path.join(roi_dir, 'tmp_rids')
    if not os.path.exists(tmp_rid_dir):
        os.makedirs(tmp_rid_dir)
        
    tmp_rid_path = os.path.join(tmp_rid_dir, 'tmp_rid_%s.json' % RID['rid_hash'])
    write_dict_to_json(RID, tmp_rid_path)
        
    print "********************************************"
    print "Created params for ROI SET, hash: %s" % RID['rid_hash']
    print "********************************************"
    
    return RID


def get_tiff_paths(rootdir='', animalid='', session='', acquisition='', run='', tiffsource=None, sourcetype=None, auto=False):

    tiffpaths = []
 
    rundir = os.path.join(rootdir, animalid, session, acquisition, run)
    processed_dir = os.path.join(rundir, 'processed')
    
    if tiffsource is None:
        while True:
            if auto is True:
                tiffsource = 'raw'
                break
            tiffsource_idx = raw_input('No tiffsource specified. Enter <R> for raw, or <P> for processed: ')
            processed_dirlist = sorted([p for p in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, p))], key=natural_keys)
            if len(processed_dirlist) == 0 or tiffsource_idx == 'R':
                tiffsource = 'raw'
                if len(processed_dirlist) == 0:
                    print "No processed dirs... Using raw."
                confirm_tiffsource = raw_input('Press <Y> to use raw.')
                if confirm_tiffsource == 'Y':
                    break  
            elif len(processed_dirlist) > 0:
                for pidx, pfolder in enumerate(sorted(processed_dirlist, key=natural_keys)):
                    print pidx, pfolder
                tiffsource_idx = int(input("Enter IDX of processed source to use: "))
                tiffsource = processed_dirlist[tiffsource_idx]
                confirm_tiffsource = raw_input('Tiffs are %s? Press <Y> to confirm. ' % tiffsource)
                if confirm_tiffsource == 'Y':
                    break
    
    if 'processed' in tiffsource: 
        tiffsource_name = [t for t in os.listdir(processed_dir) if tiffsource in t and os.path.isdir(os.path.join(processed_dir, t))][0]
        tiff_parent = os.path.join(processed_dir, tiffsource_name)
    else:
        tiffsource_name = [t for t in os.listdir(rundir) if tiffsource in t and os.path.isdir(os.path.join(rundir, t))][0]
        tiff_parent = os.path.join(rundir, tiffsource_name)

    print "Using tiffsource:", tiffsource_name
 
    if sourcetype is None:
        while True:
            if auto is True or tiffsource == 'raw':
                sourcetype = 'raw'
                break
            print "Specified PROCESSED tiff source, but not process type."
            curr_processed_dir = os.path.join(rundir, 'processed', tiffsource)
            processed_typlist = sorted([t for t in os.listdir(curr_processed_dir) if os.path.isdir(os.path.join(curr_processed_dir, t))], key=natural_keys)
            for tidx, tname in enumerate(processed_typlist):
                print tidx, tname
            sourcetype_idx = int(input('Enter IDX of processed dir to use: '))
            sourcetype = processed_typlist[sourcetype_idx]
            confirm_sourcetype = raw_input('Tiffs are from %s? Press <Y> to confirm. ' % sourcetype)
            if confirm_sourcetype == 'Y':
                break

    if 'processed' in tiffsource_name:
        sourcetype_name = [s for s in os.listdir(tiff_parent) if sourcetype in s and os.path.isdir(os.path.join(tiff_parent, s))][0]
        tiff_path = os.path.join(tiff_parent, sourcetype_name)
    else:
        tiff_path = tiff_parent
    
    
    print "Looking for tiffs in tiff_path: %s" % tiff_path 
    tiff_fns = [t for t in os.listdir(tiff_path) if t.endswith('tif')]
    tiffpaths = sorted([os.path.join(tiff_path, fn) for fn in tiff_fns], key=natural_keys)
    print "Found %i TIFFs for cNMF ROI extraction." % len(tiff_fns)

    return tiffpaths


def load_roidict(session_dir):
    
    roidict = None
    
    session = os.path.split(session_dir)[1]
    roidict_filepath = os.path.join(session_dir, 'ROIs', 'rids_%s.json' % session)
    
    # Load analysis "processdict" file:
    if os.path.exists(roidict_filepath):
        print "exists!"
        with open(roidict_filepath, 'r') as f:
            roidict = json.load(f)
            
    return roidict


def set_options_cnmf(rootdir='', animalid='', session='', acquisition='', run='',
                    movie_idxs=[], fr=None, signal_channel=1, 
                    gSig=[8,8], rf=30, stride=15, K=10, p=2, gnb=1, merge_thr=0.8,
                    init_method='greedy_roi', method_deconv='oasis',
                    rval_thr=0.8, min_SNR=2, decay_time=1.0, use_cnn=False, cnn_thr=0.8,
                    use_average=True, save_movies=True, thr_plot=0.8):
    
    # Load run meta info:
    rundir = os.path.join(rootdir, animalid, session, acquisition, run)
    runinfo_fn = os.path.join(rundir, '%s.json' % run)
    
    with open(runinfo_fn, 'r') as f:
        runinfo = json.load(f)
    
    params = dict()
    
    params['info'] = {
            'nmovies': runinfo['ntiffs'],
            'nchannels': runinfo['nchannels'],
            'signal_channel': signal_channel,
            'volumerate': runinfo['volume_rate'],
            'is_3D': len(runinfo['slices']) > 1
            }
    
    # parameters for source extraction and deconvolution
    params['extraction'] = {
            'p': p,                       # order of the autoregressive system
            'gnb': gnb,                   # number of global background components
            'merge_thresh': merge_thr,    # merging threshold, max correlation allowed
            'gSig': gSig,                 # expected half size of neurons
            'init_method': init_method,   # initialization method (if analyzing dendritic data using 'sparse_nmf')
            'method_deconv': method_deconv,
            'K': K,                       # number of components per patch
            'is_dendrites': False,        # flag for analyzing dendritic data
            'low_rank_background': True
            }
    
    params['patch'] = {
            'init_method': init_method, #initialization method (if analyzing dendritic data using 'sparse_nmf')
            'rf': rf,                     # half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
            'stride': stride,            # amount of overlap between the patches in pixels
            'only_init_patch': True
            }
    
    params['full'] = {
            'rf': None,
            'stride': None,
            'only_init_patch': False
            }
    
    # parameters for component evaluation
    if fr is None:
        fr = params['info']['volumerate']
    if len(movie_idxs) == 0 and save_movies is True:
        movie_idxs = [i for i in np.arange(1, params['info']['nmovies']+1, 2)]
        
    params['eval'] = {
            'min_SNR': min_SNR,               # signal to noise ratio for accepting a component
            'rval_thr': rval_thr,              # space correlation threshold for accepting a component
            'decay_time': decay_time,            # length of typical transient (sec)
            'use_cnn': use_cnn,             # CNN classifier designed for 2d data ?
            'cnn_thr': cnn_thr,               # threshold for CNN based classifier
            'final_frate': fr     # demos downsample - not sure if this matters for evaluation
            }
    
    # parameters for displaying stuff
    params['display'] = {
        'downsample_ratio': .2,
        'thr_plot': thr_plot,
        'use_average': use_average,
        'save_movies': save_movies, 
        'movie_files': ['File%03d' % int(i) for i in movie_idxs]
    }

    return params

def set_options_manual(rootdir='', animalid='', session='', acquisition='', run='',
                    roi_type=''): 
    
    params = dict()
    params['roi_type'] = roi_type
    
    return params


def get_params_dict(tiff_sourcedir, roi_options, roi_type='', mmap_dir=None, check_hash=False):
    
    '''mmap_dir: <rundir>/processed/<processID_processHASH>/mcorrected_<subprocessHASH>_mmap_<mmapHASH>/*.mmap
    '''
    
    PARAMS = dict()
    
    PARAMS['tiff_sourcedir'] = tiff_sourcedir
    
    if mmap_dir is None:
        tiffparent = os.path.split(tiff_sourcedir)[0]
        mmap_dirs = [m for m in os.listdir(tiffparent) if '_mmap' in m]
        if len(mmap_dirs) == 1:
            mmap_dir =  os.path.join(tiffparent, mmap_dirs[0])
            mmap_hash = os.path.split(mmap_dir)[1].split('_')[-1]
        else:
            mmap_dir = tiff_sourcedir + '_mmap'
            check_hash = True
             
        if check_hash is True:
            if os.path.isdir(mmap_dir):
                excluded_files = [f for f in os.listdir(mmap_dir) if not f.endswith('mmap')]
                mmap_hash = dirhash(mmap_dir, 'sha1', excluded_files=excluded_files)[0:6]
            else:
                mmap_hash = None
            
        if mmap_hash is not None and mmap_hash not in mmap_dir:
            PARAMS['mmap_source'] = mmap_dir + '_' + mmap_hash
            os.rename(mmap_dir, PARAMS['mmap_source'])
            print "Renamed mmap with hash:", PARAMS['mmap_source']
        else:
            PARAMS['mmap_source'] = mmap_dir
            
    PARAMS['options'] = roi_options
    PARAMS['roi_type'] = roi_type
   
    PARAMS['hashid'] = hashlib.sha1(json.dumps(PARAMS, sort_keys=True)).hexdigest()[0:6]
    print "PARAMS hashid is:", PARAMS['hashid']

    return PARAMS


def get_roi_id(PARAMS, session_dir, auto=False):
    
    roi_id = None
    
    print "********************************"
    print "Checking previous ROI IDs..."
    print "********************************"
    
    # Load previously created PIDs:
    roidict = load_roidict(session_dir)
    
    #Check current PARAMS against existing PId params by hashid:
    if roidict is None or len(roidict.keys()) == 0:
        roidict = dict()
        existing_rids = []
        is_new_rid = True
        print "No existing RIDs found."
    else:
        existing_rids = sorted([str(k) for k in roidict.keys()], key=natural_keys)
        print existing_rids
        matching_rids = sorted([rid for rid in existing_rids if roidict[rid]['PARAMS']['hashid'] == PARAMS['hashid']], key=natural_keys)
        is_new_rid = False
        if len(matching_rids) > 0:            
            while True:
                print "WARNING **************************************"
                print "Current param config matches existing RID:"
                for ridx, rid in enumerate(matching_rids):
                    print "******************************************"
                    print ridx, rid
                    pp.pprint(roidict[rid])
                if auto is True:
                    check_ridx = ''
                else:
                    check_ridx = raw_input('Enter IDX of rid to re-use, or hit <ENTER> to create new: ')
                if len(check_ridx) == 0:
                    is_new_rid = True
                    break
                else:
                    confirm_reuse = raw_input('Re-use RID %s? Press <Y> to confirm, any key to try again:' % existing_rids[int(check_ridx)])
                    if confirm_reuse == 'Y':
                        is_new_rid = False
                        break
        else:
            is_new_rid = True

    if is_new_rid is True:
        # Create new PID by incrementing num of process dirs found:
        roi_id = 'rois%03d' % int(len(existing_rids)+1)
        print "Creating NEW roi ID: %s" % roi_id
    else:
        # Re-using an existing PID:
        roi_id = existing_rids[int(check_ridx)]
        print "Reusing existing rid: %s" % roi_id
 
    return roi_id


def initialize_rid(PARAMS, session_dir, auto=False):
    
    print "************************"
    print "Initialize ROI ID."
    print "************************"
    roi_id = get_roi_id(PARAMS, session_dir, auto=auto)

    rid = dict()
    version = pkg_resources.get_distribution('pipeline').version
    
    rid['version'] = version 
    rid['roi_id'] = roi_id
    rid['SRC'] = PARAMS['tiff_sourcedir'] #source_dir
    rid['DST'] = os.path.join(session_dir, 'ROIs', roi_id)
    rid['roi_type'] = PARAMS['roi_type']

    rid['PARAMS'] = PARAMS
    
    # deal with jsonify:
    #curropts = to_json(PARAMS['options'])
    #rid['PARAMS']['options'] = curropts

    # TODO:  Generate hash for full PID dict
    rid['rid_hash'] = hashlib.sha1(json.dumps(rid, sort_keys=True)).hexdigest()[0:6]
        
    return rid


def update_roi_records(rid, session_dir):

    print "************************"
    print "Updating JSONS..."
    print "************************"
    session = os.path.split(session_dir)[1]
    roidict_filepath = os.path.join(session_dir, 'ROIs', 'rids_%s.json' % session)
    
    if os.path.exists(roidict_filepath):
        with open(roidict_filepath, 'r') as f:
            roidict = json.load(f)
    else:
        roidict = dict()

    roi_id = rid['roi_id'] 
    roidict[roi_id] = rid

    #% Update Process Info DICT:
    write_dict_to_json(roidict, roidict_filepath)

    print "ROI Info UPDATED."
    


def main(options):
    
    rid = create_rid(options)
    
    print "****************************************************************"
    print "Created RID."
    print "----------------------------------------------------------------"
    pp.pprint(rid)
    print "****************************************************************"
    
    
if __name__ == '__main__':
    main(sys.argv[1:]) 
