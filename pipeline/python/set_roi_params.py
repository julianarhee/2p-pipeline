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
import traceback
import h5py
from pipeline.python.utils import write_dict_to_json, get_tiff_paths
import numpy as np
from checksumdir import dirhash

pp = pprint.PrettyPrinter(indent=4)

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]

def post_rid_cleanup(session_dir, rid_hash):

    session = os.path.split(session_dir)[1]
    roi_dir = os.path.join(session_dir, 'ROIs')
    print "Cleaning up RID info: %s" % rid_hash
    tmp_rid_dir = os.path.join(roi_dir, 'tmp_rids')
    tmp_rid_fn = 'tmp_rid_%s.json' % rid_hash
    rid_path = os.path.join(tmp_rid_dir, tmp_rid_fn)
    if not os.path.exists(rid_path):
        print "No files to cleanup!"
        return

    with open(rid_path, 'r') as f:
        RID = json.load(f)

    roidict_fn = 'rids_%s.json' % session
    # UPDATE PID entry in dict:
    with open(os.path.join(roi_dir, roidict_fn), 'r') as f:
        roidict = json.load(f)
    roi_id = [p for p in roidict.keys() if roidict[p]['rid_hash'] == rid_hash][0]
    roidict[roi_id] = RID

    # Save updated PID dict:
    path_to_roidict = os.path.join(roi_dir, roidict_fn)
    write_dict_to_json(roidict, path_to_roidict)

    finished_dir = os.path.join(tmp_rid_dir, 'completed')
    if not os.path.exists(finished_dir):
        os.makedirs(finished_dir)
    if os.path.exists(rid_path):
        os.rename(rid_path, os.path.join(finished_dir, tmp_rid_fn))
    #shutil.move(rid_path, os.path.join(finished_dir, tmp_rid_fn))
    print "Moved tmp rid file to completed."


def extract_options(options):
    choices_sourcetype = ('raw', 'mcorrected', 'bidi')
    default_sourcetype = 'mcorrected'

    choices_roi = ('caiman2D', 'manual2D_circle', 'manual2D_square', 'manual2D_polygon', 'coregister')
    default_roitype = 'caiman2D'

    parser = optparse.OptionParser()

    # PATH opts:
    parser.add_option('-D', '--root', action='store', dest='rootdir', default='/nas/volume1/2photon/data', help='data root dir (root project dir containing all animalids) [default: /nas/volume1/2photon/data, /n/coxfs01/2pdata if --slurm]')
    parser.add_option('-H', '--home', action='store', dest='homedir', default='/nas/volume1/2photon/data', help='current data root dir (if creating params with path-root different than what will be used for actually doing the processing.')
    parser.add_option('--notnative', action='store_true', dest='notnative', default=False, help="Set flag if not setting params on same system as processing. MUST rsync data sources.")

    parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')

    # Set specific session/run for current animal:
    parser.add_option('-S', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID')
    parser.add_option('-A', '--acq', action='store', dest='acquisition', default='FOV1', help="acquisition folder (ex: 'FOV1_zoom3x') [default: FOV1]")
    parser.add_option('-R', '--run', action='store', dest='run', default='', help="name of run dir containing tiffs to be processed (ex: gratings_phasemod_run1)")
    parser.add_option('--slurm', action='store_true', dest='slurm', default=False, help="set if running as SLURM job on Odyssey")
    parser.add_option('--default', action='store_true', dest='default', default='store_false', help="Use all DEFAULT params, for params not specified by user (prevent interactive)")

    parser.add_option('-s', '--tiffsource', action='store', dest='tiffsource', default=None, help="name of folder containing tiffs to be processed (ex: processed001). should be child of <run>/processed/")
    parser.add_option('-t', '--source-type', type='choice', choices=choices_sourcetype, action='store', dest='sourcetype', default=default_sourcetype, help="Type of tiff source. Valid choices: %s [default: %s]" % (choices_sourcetype, default_sourcetype))
    parser.add_option('-o', '--roi-type', type='choice', choices=choices_roi, action='store', dest='roi_type', default=default_roitype, help="Roi type. Valid choices: %s [default: %s]" % (choices_roi, default_roitype))
    parser.add_option('--mc', action='store_true', dest='check_motion', default=False, help="Exclude tiffs that fail motion-correction evaluation metric.")
    parser.add_option('--mcmetric', action='store', dest='mcmetric', default='zproj_corrcoefs', help='Motion-correction metric to determine tiffs to exclude [default: zproj_corrcoefs]')
    parser.add_option('-x', '--exclude', action="store", dest="excluded_tiffs", default='', help="User-selected tiff numbers to exclude (comma-separated) - 1 indexed")

    # MANUAL OPTS:
    parser.add_option('-f', '--ref-file', action='store', dest='ref_file', default=1, help="[man]: File NUM of tiff to use as reference, if applicable [default: 1]")
    parser.add_option('-c', '--ref-channel', action='store', dest='ref_channel', default=1, help="[man]: Channel NUM of tiff to use as reference, if applicable [default: 1]")
    parser.add_option('-z', '--slices', action='store', dest='slices', default='', help="[man]: Comma-separated list of slice numbers (1-indexed) for ROI extraction [default: all slices in run tiffs]")
    parser.add_option('-g', '--zproj', action='store', dest='zproj_type', default='mean', help="[man]: Type of z-projection to use as image for ROI extraction, if applicable [default: mean]")

    # cNMF OPTS:
    parser.add_option('--deconv', action='store', dest='nmf_deconv', default='oasis', help='[nmf]: method deconvolution if using cNMF [default: oasis]')
    parser.add_option('--gSig', action='store', dest='nmf_gsig', default=8, help='[nmf]: Half size of neurons if using cNMF [default: 8]')
    parser.add_option('--K', action='store', dest='nmf_K', default=10, help='[nmf]: N expected components per patch [default: 10]')
    parser.add_option('--patch', action='store', dest='nmf_rf', default=30, help='[nmf]: Half size of patch if using cNMF [default: 30]')
    parser.add_option('--stride', action='store', dest='nmf_stride', default=5, help='[nmf]: Amount of patch overlap if using cNMF [default: 5]')
    parser.add_option('--nmf-order', action='store', dest='nmf_p', default=2, help='[nmf]: Order of autoregressive system if using cNMF [default: 2]')
    parser.add_option('--border', action='store', dest='border_pix', default=0, help='[nmf]: N pixels to exclude for border (from motion correcting)[default: 0]')
    parser.add_option('--mmap', action='store_true', dest='mmap_new', default=False, help="[nmf]: set if want to make new mmap set")

    # COREG OPTS:
    parser.add_option('-u', '--roi-source', action='store', dest='roi_source_id', default='', help='[coreg]: Name of ROI ID that is the source of coregsitered ROIs (TODO: allow for multiple sources)')
    parser.add_option('--good', action="store_true",
                      dest="keep_good_rois", default=False, help="[coreg]: Set flag to only keep good components (useful for avoiding computing massive ROI sets)")
    parser.add_option('--max', action="store_true",
                      dest="use_max_nrois", default=False, help="[coreg]: Set flag to use file with max N components (instead of reference file) [default uses reference]")
    parser.add_option('-f', '--ref', action="store",
                      dest="coreg_fidx", default=None, help="Reference file for coregistration if use_max_nrois==False")

    parser.add_option('-E', '--eval-key', action="store",
                      dest="eval_key", default=None, help="[coreg]: Evaluation key from ROI source <rid_dir>/evaluation (format: evaluation_YYYY_MM_DD_hh_mm_ss)")
    parser.add_option('-b', '--maxthr', action='store', dest='dist_maxthr', default=0.1, help="[coreg]: threshold for turning spatial components into binary masks [default: 0.1]")
    parser.add_option('-n', '--power', action='store', dest='dist_exp', default=0.1, help="[coreg]: power n for distance between masked components: dist = 1 - (and(M1,M2)/or(M1,M2)**n [default: 1]")
    parser.add_option('-d', '--dist', action='store', dest='dist_thr', default=0.5, help="[coreg]: threshold for setting a distance to infinity, i.e., illegal matches [default: 0.5]")
    parser.add_option('-v', '--overlap', action='store', dest='dist_overlap_thr', default=0.8, help="[coreg]: overlap threshold for detecting if one ROI is subset of another [default: 0.8]")

    (options, args) = parser.parse_args(options)

    if options.slurm is True:
        if 'coxfs01' not in options.rootdir:
            options.rootdir = '/n/coxfs01/2p-data'
    if options.notnative is False:
        print "NATIVE~~"
        options.homedir = options.rootdir

    return options


def get_roi_eval_path(src_roi_dir, eval_key, auto=False):
    src_eval_filepath = None
    src_eval = None
    try:
        print "-----------------------------------------------------------"
        print "Loading evaluation results for src roi set"
        # Load eval info:
        src_eval_filepath = os.path.join(src_roi_dir, 'evaluation', 'evaluation_%s' % eval_key, 'evaluation_results_%s.hdf5' % eval_key)
        assert os.path.exists(src_eval_filepath), "Specfied EVAL src file does not exist!\n%s" % src_eval_filepath
        src_eval = h5py.File(src_eval_filepath, 'r')
    except Exception as e:
        print "Error loading specified eval file:\n%s" % src_eval_filepath
        traceback.print_exc()
        print "-----------------------------------------------------------"
        try:
            evaldict_filepath = os.path.join(src_roi_dir, 'evaluation', 'evaluation_info.json')
            with open(evaldict_filepath, 'r') as f:
                evaldict = json.load(f)
            eval_list = sorted(evaldict.keys(), key=natural_keys)
            print "Found evaluation keys:"
            if auto is False:
                while True:
                    if len(eval_list) > 1:
                        for eidx, ekey in enumerate(eval_list):
                            print eidx, ekey
                            eval_select_idx = input('Select IDX of evaluation key to view: ')
                    else:
                        eval_select_idx = 0
                        print "Only 1 evaluation set found: %s" % eval_list[eval_select_idx]
                    pp.pprint(evaldict[eval_list[eval_select_idx]])
                    confirm_eval = raw_input('Enter <Y> to use this eval set, or <n> to return: ')
                    if confirm_eval == 'Y':
                        eval_key = eval_list[eval_select_idx].split('evaluation_')[-1]
                        print "Using key: %s" % eval_key
                        break
            else:
                print "Auto is ON, using most recent evaluation set: %s" % eval_key
                eval_key = eval_list[-1].split('evaluation_')[-1]
                pp.pprint(evaldict[eval_list[-1]])

            src_eval_filepath = os.path.join(src_roi_dir, 'evaluation', 'evaluation_%s' % eval_key, 'evaluation_results_%s.hdf5' % eval_key)
            src_eval = h5py.File(src_eval_filepath, 'r')
        except Exception as e:
            print "ERROR: Can't load source evaluation file - %s" % eval_key
            traceback.print_exc()
            print "Aborting..."
            print "-----------------------------------------------------------"

    return src_eval_filepath

def create_rid(options):

    options = extract_options(options)

    # Set USER INPUT options:
    rootdir = options.rootdir
    homedir = options.homedir
    notnative = options.notnative
    print "ROOT:", rootdir
    print "HOME:", homedir
    animalid = options.animalid
    session = options.session
    acquisition = options.acquisition
    run = options.run

    tiffsource = options.tiffsource
    sourcetype = options.sourcetype
    roi_type = options.roi_type
    auto = options.default

    check_motion = options.check_motion
    mcmetric = options.mcmetric
    exclude_str = options.excluded_tiffs

    # cNMF-specific opts:
    nmf_deconv = options.nmf_deconv
    nmf_gsig = [int(options.nmf_gsig), int(options.nmf_gsig)]
    nmf_K = int(options.nmf_K)
    nmf_rf = int(options.nmf_rf)
    nmf_stride = int(options.nmf_stride)
    nmf_p = int(options.nmf_p)
    border_pix = int(options.border_pix)
    mmap_new = options.mmap_new

    # COREG specific opts:
    roi_source_str = options.roi_source_id
    if len(roi_source_str) > 0:
        roi_source_ids = ['rois%03d' % int(r) for r in roi_source_str.split(',')]
    print "ROI SOURCES:", roi_source_ids
    keep_good_rois = options.keep_good_rois
    use_max_nrois = options.use_max_nrois

    if use_max_nrois is False:
        coreg_fidx = int(options.coreg_fidx) - 1
        reference_filename = "File%03d" % int(options.coreg_fidx)
    else:
        reference_filename = None

    eval_key = options.eval_key

    dist_maxthr = options.dist_maxthr
    dist_exp = options.dist_exp
    dist_thr = options.dist_thr
    dist_overlap_thr = options.dist_overlap_thr


    # manual options:
    ref_file = int(options.ref_file)
    ref_channel = int(options.ref_channel)
    zproj_type = options.zproj_type
    slices_str = options.slices
    if not roi_type=='coregister':
        if len(slices_str)==0:
            # Use all slices
            runmeta_path = os.path.join(homedir, animalid, session, acquisition, run, '%s.json' % run)
            with open(runmeta_path, 'r') as f:
                runinfo = json.load(f)
            slices = runinfo['slices']
        else:
            slices = slices_str.split(',')
            slices = [int(s) for s in slices]

    # Create ROI output dir:
    session_dir = os.path.join(homedir, animalid, session)
    roi_dir = os.path.join(session_dir, 'ROIs')
    if not os.path.exists(roi_dir):
        os.makedirs(roi_dir)

    # Get paths to tiffs from which to create ROIs:
    if not roi_type == 'coregister':
        tiffpaths = get_tiff_paths(rootdir=homedir,
                                   animalid=animalid, session=session,
                                   acquisition=acquisition, run=run,
                                   tiffsource=tiffsource, sourcetype=sourcetype,
                                   auto=auto)
        tiff_sourcedir = os.path.split(tiffpaths[0])[0]
        print "SRC: %s, found %i tiffs." % (tiff_sourcedir, len(tiffpaths))

    # Get roi-type specific options:
    if roi_type == 'caiman2D':
        print "Creating param set for caiman2D ROIs."
        movie_idxs = []
        roi_options = set_options_cnmf(rootdir=homedir, animalid=animalid, session=session,
                                       acquisition=acquisition, run=run,
                                       movie_idxs=movie_idxs,
                                       method_deconv=nmf_deconv,
                                       K=nmf_K,
                                       gSig=nmf_gsig,
                                       rf=nmf_rf,
                                       stride=nmf_stride,
                                       p=nmf_p,
                                       border_pix=border_pix)
        src_roi_type = roi_type
    elif 'manual' in roi_type:
        roi_options = set_options_manual(rootdir=homedir, animalid=animalid, session=session,
                                         acquisition=acquisition, run=run,
                                         roi_type=roi_type,
                                         zproj_type=zproj_type,
                                         ref_file=ref_file,
                                         ref_channel=ref_channel,
                                         slices=slices)
        src_roi_type = roi_type
    elif roi_type == 'coregister':
        roi_options = set_options_coregister(rootdir=homedir, animalid=animalid, session=session,
                                         roi_source=roi_source_ids,
                                         roi_type=roi_type,
                                         keep_good_rois=keep_good_rois,
                                         use_max_nrois=use_max_nrois,
                                         coreg_fidx=coreg_fidx,
                                         reference_filename=reference_filename,
                                         eval_key=eval_key,
                                         dist_maxthr=dist_maxthr,
                                         dist_exp=dist_exp,
                                         dist_thr=dist_thr,
                                         dist_overlap_thr=dist_overlap_thr,
                                         auto=auto)
        if len(roi_source_ids) > 1:
            tiff_sourcedir = sorted([roi_options['source'][k]['tiff_dir'] for k in roi_options['source'].keys()], key=natural_keys)
            src_roi_type = roi_options['source'][0]['roi_type']
        else:
            tiff_sourcedir = roi_options['source']['tiff_dir']
            src_roi_type = roi_options['source']['roi_type']


    # Create roi-params dict with source and roi-options:
    PARAMS = get_params_dict(tiff_sourcedir, roi_options, roi_type=roi_type,
                             notnative=notnative, rootdir=rootdir, homedir=homedir, auto=auto,
                             mmap_new=mmap_new, check_hash=False,
                             check_motion=check_motion, mcmetric=mcmetric, exclude_str=exclude_str)

    # Create ROI ID (RID):
    RID = initialize_rid(PARAMS, session_dir, notnative=notnative, rootdir=rootdir, homedir=homedir)

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
                    gSig=[8,8], rf=30, stride=5, K=10, p=2, gnb=1, merge_thr=0.8,
                    init_method='greedy_roi', method_deconv='oasis',
                    rval_thr=0.8, min_SNR=2, decay_time=1.0, use_cnn=False, cnn_thr=0.8,
                    use_average=True, save_movies=True, thr_plot=0.8,
                    border_pix=0):

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
            'is_3D': len(runinfo['slices']) > 1,
            'max_shifts': border_pix
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
                    roi_type='', zproj_type='', ref_file=1, ref_channel=1, slices=[1]):

    params = dict()
    params['roi_type'] = roi_type
    params['zproj_type'] = zproj_type
    params['ref_file'] = int(ref_file)
    params['ref_channel'] = int(ref_channel)
    params['slices'] = slices

    return params

def set_options_coregister(rootdir='', animalid='', session='', auto=False,
                           roi_source='', roi_type='',
                           use_max_nrois=True, keep_good_rois=True, coreg_fidx=None, reference_filename=None,
                           eval_key="",
                           dist_maxthr=0.1, dist_exp=0.1, dist_thr=0.5, dist_overlap_thr=0.8):

    # TODO:  Allow multiple ROI sets from 1 session to be coregistered
    # TODO:  Allow multiple sessions to be coregistered...

    params = dict()
    params['roi_type'] = roi_type
    params['roi_source'] = roi_source

    # Load ROI info from source set:
    rid_info_path = os.path.join(rootdir, animalid, session, 'ROIs', 'rids_%s.json' % session)
    with open(rid_info_path, 'r') as f:
        rdict = json.load(f)
    params['source'] = dict()
    for ridx,roi_source_id in enumerate(roi_source):
        src_rid = rdict[roi_source_id]
        params['source'][ridx] = dict()
        params['source'][ridx]['roi_dir'] = src_rid['DST']
        params['source'][ridx]['tiff_dir'] = src_rid['SRC']
        params['source'][ridx]['rid_hash'] = src_rid['rid_hash']
        params['source'][ridx]['roi_id'] = src_rid['roi_id']
        params['source'][ridx]['roi_type'] = src_rid['roi_type']
        evalpath = get_roi_eval_path(src_rid['DST'], eval_key, auto=auto)
        params['source'][ridx]['roi_eval_path'] = evalpath

    if len(roi_source) == 1:
        params['source'] = params['source'][0]

    params['keep_good_rois'] = keep_good_rois
    params['use_max_nrois'] = use_max_nrois
    params['ref_filename'] = reference_filename
    params['coreg_fidx'] = coreg_fidx

    params['dist_maxthr'] = dist_maxthr
    params['dist_exp'] = dist_exp
    params['dist_thr'] = dist_thr
    params['dist_overlap_thr'] = dist_overlap_thr


    return params

def get_params_dict(tiff_sourcedir, roi_options, roi_type='',
                    notnative=False, rootdir='', homedir='', auto=False,
                    mmap_new=False, check_hash=False,
                    check_motion=False, mcmetric='zproj_corrcoef', exclude_str=''):

    '''mmap_dir: <rundir>/processed/<processID_processHASH>/mcorrected_<subprocessHASH>_mmap_<mmapHASH>/*.mmap
    '''
    if roi_type == 'coregister':
        if 'roi_type' not in roi_options['source'].keys():
            roi_src_type = roi_options['source'][0]['roi_type']
        else:
            roi_src_type = roi_options['source']['roi_type']
    else:
        roi_src_type = None

    PARAMS = dict()
    if isinstance(tiff_sourcedir, list): # > 1:
        # Multiple sources (roi_type==coregister), so store each src's info:
        PARAMS['tiff_sourcedir'] = {}
        if roi_type == 'caiman2D':
            PARAMS['mmap_source'] = {}
        for tidx, tiff_source in enumerate(sorted(tiff_sourcedir, key=natural_keys)):
            PARAMS['tiff_sourcedir'][tidx] = tiff_source
            if roi_type=='caiman2D' or (roi_type=='coregister' and roi_src_type=='caiman2D'):
                mmap_dir = get_mmap_dirname(tiff_source, mmap_new=mmap_new, check_hash=check_hash, auto=auto)
                PARAMS['mmap_source'][tidx] = mmap_dir
    else:
        # Single source, don't store as dict:
        if isinstance(tiff_sourcedir, list):
            tiff_sourcedir = tiff_sourcedir[0]
        PARAMS['tiff_sourcedir'] = tiff_sourcedir
        if roi_type == 'caiman2D'  or (roi_type=='coregister' and roi_src_type=='caiman2D'):
            mmap_dir = get_mmap_dirname(tiff_sourcedir, mmap_new=mmap_new, check_hash=check_hash, auto=auto)
            PARAMS['mmap_source'] = mmap_dir

    # Replace PARAM paths with processing-machine paths:
    if notnative is True:
        for dirtype in PARAMS.keys():
            if isinstance(PARAMS[dirtype], dict):
                for k in PARAMS[dirtype]:
                    PARAMS[dirtype][k] = PARAMS[dirtype][k].replace(homedir, rootdir)
            else:
                PARAMS[dirtype] = PARAMS[dirtype].replace(homedir, rootdir)

    PARAMS['options'] = roi_options
    PARAMS['eval'] = dict()
    PARAMS['eval']['check_motion'] = check_motion
    PARAMS['eval']['mcmetric'] = mcmetric
    if len(exclude_str) > 0:
        PARAMS['eval']['manual_excluded'] = ['File%03d' % int(x) for x in exclude_str.split(',')]
    else:
        PARAMS['eval']['manual_excluded'] = []
    PARAMS['roi_type'] = roi_type
    PARAMS['hashid'] = hashlib.sha1(json.dumps(PARAMS, sort_keys=True)).hexdigest()[0:6]
    print "PARAMS hashid is:", PARAMS['hashid']

    return PARAMS

def get_mmap_dirname(tiff_sourcedir, mmap_new=False, check_hash=False, auto=False):
    mmap_dir = None

    # First check if mmap-ed dir exists:
    tiffparent = os.path.split(tiff_sourcedir)[0]
    mmap_dirs = [m for m in os.listdir(tiffparent) if '_mmap' in m]
    if len(mmap_dirs) == 1 and mmap_new is False:
        mmap_dir =  os.path.join(tiffparent, mmap_dirs[0])
        mmap_hash = os.path.split(mmap_dir)[1].split('_')[-1]
        check_hash = False
    elif len(mmap_dirs) > 1 and mmap_new is False:
        if auto is True:
            mmap_dir =  os.path.join(tiffparent, mmap_dirs[0])
            mmap_hash = os.path.split(mmap_dir)[1].split('_')[-1]
            check_hash = False
        else:
            print "Found multiple mmap dirs in source: %s" % tiffparent
            for midx, mdir in enumerate(mmap_dirs):
                print midx, mdir
            user_selected = raw_input("Select IDX of mmap dir to use, or press <N> to make new memmap dir: ")
            if user_selected == 'N':
                mmap_new = True
                check_hash = True
            else:
                mmap_dir = os.path.join(tiffparent, mmap_dirs[int(user_selected)])
                mmap_hash = os.path.split(mmap_dir)[1].split('_')[-1]
                check_hash = False

    if mmap_dir is None or mmap_new is True:
        mmap_dir = tiff_sourcedir + '_mmap'
        check_hash = True

    if check_hash is True:
        if os.path.isdir(mmap_dir): # Get hash for mmap files to rename mmap dir
            excluded_files = [f for f in os.listdir(mmap_dir) if not f.endswith('mmap')]
            mmap_hash = dirhash(mmap_dir, 'sha1', excluded_files=excluded_files)[0:6]
        else:
            mmap_hash = None

    if mmap_hash is not None and mmap_hash not in mmap_dir:
        mmap_source = mmap_dir + '_' + mmap_hash
        os.rename(mmap_dir, mmap_source)
        print "Renamed mmap with hash:", mmap_source
    else:
        mmap_source = mmap_dir

    return mmap_source

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


def initialize_rid(PARAMS, session_dir, notnative=False, rootdir='', homedir='', auto=False):

    print "************************"
    print "Initialize ROI ID."
    print "************************"
    roi_id = get_roi_id(PARAMS, session_dir, auto=auto)

    rid = dict()
    version = pkg_resources.get_distribution('pipeline').version

    rid['version'] = version
    rid['roi_id'] = roi_id
    if notnative is True:
        if rootdir not in PARAMS['tiff_sourcedir']:
            PARAMS['tiff_sourcedir'] = PARAMS['tiff_sourcedir'].replace(homedir, rootdir)
        if rootdir not in session_dir:
            session_dir = session_dir.replace(homedir, rootdir)

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
