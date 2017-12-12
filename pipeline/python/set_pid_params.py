#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 17:19:12 2017

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

from checksumdir import dirhash
from stat import S_IREAD, S_IRGRP, S_IROTH, S_IWRITE, S_IWGRP, S_IWOTH
import shutil


pp = pprint.PrettyPrinter(indent=4)

# GENERAL METHODS:
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]

def convert_bytes(num):
    """
    this function will convert bytes to MB.... GB... etc
    """
    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if num < 1024.0:
            return "%3.1f %s" % (num, x)
        num /= 1024.0

def get_file_size(file_path):
    """
    this function will return the file size
    """
    if os.path.isfile(file_path):
        file_info = os.stat(file_path)
        return convert_bytes(file_info.st_size)

def post_pid_cleanup(acquisition_dir, run, pid_hash):

    processed_dir = os.path.join(acquisition_dir, run, 'processed')
    print "Cleaning up PID info: %s" % pid_hash
    tmp_pid_dir = os.path.join(processed_dir, 'tmp_pids')
    tmp_pid_fn = 'tmp_pid_%s.json' % pid_hash
    pid_path = os.path.join(tmp_pid_dir, tmp_pid_fn)
    with open(pid_path, 'r') as f:
        PID = json.load(f)

    processdict_fn = 'pids_%s.json' % run
    # UPDATE PID entry in dict:
    with open(os.path.join(processed_dir, processdict_fn), 'r') as f:
        processdict = json.load(f)
    process_id = [p for p in processdict.keys() if processdict[p]['pid_hash'] == pid_hash][0]
#    process_id_basename = PID['process_id']
#    new_process_id_key = '_'.join((process_id_basename, pid_hash))
#    processdict[new_process_id_key] = processdict.pop(PID['process_id'])
#    print "Updated main process dict, with key: %s" % new_process_id_key
    processdict[process_id] = PID
    
    # Save updated PID dict:
    path_to_processdict = os.path.join(processed_dir, processdict_fn)
    write_dict_to_json(processdict, path_to_processdict)

    finished_dir = os.path.join(tmp_pid_dir, 'completed')
    if not os.path.exists(finished_dir):
        os.makedirs(finished_dir)
    shutil.move(pid_path, os.path.join(finished_dir, tmp_pid_fn))
    print "Moved tmp pid file to completed."


def change_permissions_recursive(path, mode):
    for root, dirs, files in os.walk(path, topdown=False):
        #for dir in [os.path.join(root,d) for d in dirs]:
            #os.chmod(dir, mode)
        for file in [os.path.join(root, f) for f in files]:
            os.chmod(file, mode)
            
def write_hash_readonly(write_dir, PID=None, step='', label=''):
    
    # Before changing anything, check if there is a corresponding 'slices" dir:
    adjust_slicedir = False
    if os.path.isdir(write_dir + '_slices'):
        adjust_slicedir = True             # WRITE-DIR has companion _slices dir.
        slice_dir = write_dir + '_slices'

    write_hash = None
    excluded_files = [str(f) for f in os.listdir(write_dir) if not f.endswith('tif')]
    print "Checking %s hash, excluded_files:" % label, excluded_files
    write_hash = dirhash(write_dir, 'sha1', excluded_files=excluded_files)[0:6]
    
    # Rename dir if hash is not included:
    if write_hash not in write_dir:
        newwrite_dir = write_dir + '_%s' % write_hash
        os.rename(write_dir, newwrite_dir)
    else:
        newwrite_dir = write_dir
    
    # Set READ-ONLY permissions:
    change_permissions_recursive(newwrite_dir, S_IREAD|S_IRGRP|S_IROTH)
    # for f in os.listdir(newwrite_dir):
    #     os.chmod(os.path.join(newwrite_dir, f), S_IREAD|S_IRGRP|S_IROTH)  
        
    if PID is not None:
        if write_hash not in PID['PARAMS'][step]['destdir']:
            PID['PARAMS'][step]['destdir'] = newwrite_dir
        print "PID %s: Renamed output dir: %s" % (PID['pid_hash'], PID['PARAMS'][step]['destdir'])
    
    # Adjust slice dir, too, if needed:
    if adjust_slicedir is True:
        print "Also adding hash to _slices dir:", slice_dir
        if write_hash not in slice_dir:
            newwrite_dir_slices = write_dir + '_%s_slices' % write_hash
            os.rename(slice_dir, newwrite_dir_slices)
        else:
            newwrite_dir_slices = slice_dir
        # Set READ-ONLY permissions:
        change_permissions_recursive(newwrite_dir_slices, S_IREAD|S_IRGRP|S_IROTH)
        # for f in os.listdir(newwrite_dir_slices):
        #     os.chmod(os.path.join(newwrite_dir_slices, f), S_IREAD|S_IRGRP|S_IROTH)  

        
    return write_hash, PID

def append_hash_to_paths(PID, pid_hash, step=''):
    correct_flyback = PID['PARAMS']['preprocessing']['correct_flyback']
    correct_bidir = PID['PARAMS']['preprocessing']['correct_bidir']
    correct_motion = PID['PARAMS']['motion']['correct_motion']

    # If get_scanimage_data.py not run, <rundir>/raw/ does not have hash:
    if PID['PARAMS']['source']['tiffsource'] == 'raw':
        rawtiff_dir = PID['SRC']
        rawtiff_hashdir = [r for r in os.listdir(os.path.split(PID['SRC'])[0]) if 'raw_' in r]
        if len(rawtiff_hashdir) == 1:
            # Raw-hashed renamed dir exists
            rawtiff_dir = os.path.join(os.path.split(PID['SRC'])[0], rawtiff_hashdir[0])
        elif os.path.exists(rawtiff_dir):
            # No hash created, rawtiff folder is just 'raw':
            print "Checking tiffdir hash..."
            rawdir_hash, PID = write_hash_readonly(rawtiff_hashdir, PID=None, step='simeta')
            if rawdir_hash not in rawtiff_dir:
                rawtiff_dir = rawtiff_dir + '_%s' % rawdir_hash
                print "Got hash for RAW dir:", rawtiff_dir
        PID['SRC'] = rawtiff_dir
    elif 'processed' in PID['PARAMS']['source']['tiffsource']:
        acquisition_dir = os.path.join(PID['PARAMS']['source']['rootdir'], PID['PARAMS']['source']['animalid'],\
                            PID['PARAMS']['source']['session'], PID['PARAMS']['source']['acquisition'])
        processed_dir = os.path.join(acquisition_dir, PID['PARAMS']['source']['run'], 'processed')
        processed_name = PID['PARAMS']['source']['tiffsource'].split('/')[0]
        processed_hashid = [p for p in os.listdir(processed_dir) if processed_name in p][0]
        if processed_hashid not in PID['PARAMS']['source']['tiffsource']:
            # Replace all processed src paths with hashed src:
            print "Adding processed dir hash to all sources."
            PID['SRC'] = PID['SRC'].split(processed_name)[0] + processed_hashid + PID['SRC'].split(processed_name)[1] 
            for pkey in PID['PARAMS'].keys():
                if not isinstance(PID['PARAMS'][pkey], dict):
                    continue
                src_pkey = [skey for skey in PID['PARAMS'][pkey].keys() if 'source' in skey]
                if len(src_pkey) > 0:
                    for skey in src_pkey:
                        PID['PARAMS'][pkey][skey] = PID['PARAMS'][pkey][skey].split(processed_name)[0]\
                                                        + processed_hashid + PID['PARAMS'][pkey][skey].split(processed_name)[1]
    if pid_hash not in PID['DST']:
        PID['DST'] = PID['DST'] + '_%s' % pid_hash
        
    print " ----STEP: %s ---------------------------------" % step
    print 'PID %s: SRC is %s' % (pid_hash, PID['SRC'])
    print 'PID %s: DST is %s' % (pid_hash, PID['DST'])
    print "------------------------------------------------------"

    # Update source/dest, depending on current step and params:
    if step == 'flyback':
        PID['PARAMS']['preprocessing']['sourcedir'] = PID['SRC']
        if correct_flyback is True:
            PID['PARAMS']['preprocessing']['destdir'] = os.path.join(PID['DST'], 'raw')

    if step == 'bidir':
        if correct_flyback is True:
            bidi_destdir = PID['PARAMS']['preprocessing']['destdir'] 
            if PID['pid_hash'] in bidi_destdir and 'raw' in bidi_destdir:
                # Bidir-correction is on flyback-corrected tiffs:
                PID['PARAMS']['preprocessing']['sourcedir'] = copy.copy(PID['PARAMS']['preprocessing']['destdir'])
            else:
                # Get flyback-corrected tiffs in current processing dir:
                raw_flyback_dir = [r for r in os.listdir(PID['DST']) if 'raw' in r and os.path.isdir(os.path.join(PID['DST'], r))][0]
                PID['PARAMS']['preprocessing']['sourcedir'] = os.path.join(PID['DST'], raw_flyback_dir)
        else:
            # Bidir-correction is raw/SRC tiffs:
            print "processing on RAW"
            PID['PARAMS']['preprocessing']['sourcedir'] = PID['SRC']
        if correct_bidir is True:
            PID['PARAMS']['preprocessing']['destdir'] = os.path.join(PID['DST'], 'bidi')

    if step == 'motion':
        if correct_bidir is True or correct_flyback is True:
            PID['PARAMS']['motion']['sourcedir'] = copy.copy(PID['PARAMS']['preprocessing']['destdir'])
        else:
            PID['PARAMS']['motion']['sourcedir'] = copy.copy(PID['SRC'])
        if correct_motion is True:
            PID['PARAMS']['motion']['destdir'] = os.path.join(PID['DST'], 'mcorrected')

    return PID

def set_motion_params(correct_motion=False,
                 ref_channel=0,
                 ref_file=0,
                 method=None,
                 algorithm=None,
                 auto=False
                ):

    mc_methods = ['Acquisition2P', 'NoRMCorre']
    mc_algos = dict((mc, []) for mc in mc_methods)
    mc_algos = {'Acquisition2P': ['@withinFile_withinFrame_lucasKanade', '@lucasKanade_plus_nonrigid'],
                'NoRMCorre': ['rigid', 'nonrigid']}

    if correct_motion is True:
        if method is None:
            while True:
                if auto is True:
                    method = 'Acquisition2P'
                    break
                print "No MC method specified. Use default [Acquisition2P]?"
                mc_choice = raw_input('Enter <Y> to use default, or <o> to see options:')
                if mc_choice == 'Y':
                    print "Using default."
                    method = 'Acquisition2P'
                    break
                elif mc_choice == 'o':
                    for mcid, mcname in enumerate(mc_methods):
                        print mcid, mcname
                    mc_select = input('Enter IDX of motion-correction method to use:')
                    method = mc_methods[mc_select]
                    break
        if algorithm is None or (algorithm not in mc_algos[method]):
            if auto is True:
                algorithm = mc_algos['Acquisition2P'][0]
            else:
                print "No MC algorithm specified... Here are the options:"
                for algonum, algoname in enumerate(mc_algos[method]):
                    print algonum, algoname
                algo_choice = input('Enter IDX of mc algorithm to use:')
                algorithm = mc_algos[method][algo_choice] 
    else:
        ref_channel = 0
        ref_file = 0
        method = None
        algorithm = None
        
    motion_params = dict()
    motion_params['correct_motion'] = correct_motion
    motion_params['method'] = method
    motion_params['algorithm'] = algorithm
    motion_params['ref_channel'] = ref_channel
    motion_params['ref_file'] = ref_file
    
    return motion_params

def set_preprocessing_params(correct_flyback=False,
                  nflyback_frames=0,
                  correct_bidir=False,
                  split_channels=False
                 ):
    
    preprocess_params = dict()
    if correct_flyback is False:
        nflyback_frames = 0
    
    preprocess_params['correct_flyback'] = correct_flyback
    preprocess_params['nflyback_frames'] = nflyback_frames
    preprocess_params['correct_bidir'] = correct_bidir
    preprocess_params['split_channels'] = split_channels
   
    return preprocess_params


def set_params(rootdir='', animalid='', session='', acquisition='', run='', tiffsource=None, sourcetype='raw',
               correct_bidir=False, correct_flyback=False, nflyback_frames=None,
               split_channels=False, correct_motion=False, ref_file=1, ref_channel=1,
               mc_method=None, mc_algorithm=None, auto=False):
    
    '''
    Create a dictionary of PARAMS used for processing TIFFs.
    PARAMS = {
        'source' : {
            'rootdir' (str)
                Root base of all data
            'animalid' : str
                Animal name (letters and numbers only, e.g., CE052)
            'session' (str)
                Session name (standard format: YYYYMMDD_<animalid>, e.g., 20171026_CE052)
            'acquisition': str
                Name of subdir in session directory for tiffs of a given FOV (e.g., FOV1_zoom3x')
            'run' (str)
                Name of experimental run (e.g., 'gratings_run1'). Child dir is 'raw' (acquired tiffs, read-only).
            'tiffsource' (str)
                Name of folder containing tiffs to be processed
                Full file-tree ex, <rootdir>/<animalid>/<session>/<acquisition>/<run>/raw
                }
        'motion' : {
            'correct_motion' (bool)
                True/False for whether to correct motion or not.
            'method' (str)
                Package to use for motion-correction (default: Acquisition2P)
                Options: Acquisition2P, NoRMCorre
            'algorithm' (str)
                Algorithm to use for motion-correction (default: '@withinFile_withinFrame_lucasKanade')
                Options:
                    Acquisition2P: '@withinFile_withinFrame_lucasKanade', '@lucasKanade_plus_nonrigid'
                    NoRMCorre: 'rigid', 'nonrigid'
            'ref_channel' (str)
                Reference channel to use for motion-correction (default: 1)
            'ref_file' (str)
                Reference file to use for motion-correction (default: 1)
            }
        'preprocessing' : {
            'correct_flyback' (bool)
                Whether to correct for bad flyback settings by removing extra frames from top of stack.
                Assumes stack acquisition starts from top (default: False)
            'nflyback_frames' (int)
                Number of extra frames to remove from the stack to correct flyback error (default: 1)
            'correct_bidir' (bool)
                Whether to correct scan phase if acquired with bidirectional scanning (default: False)
            'split_channels' (bool)
                Whether to split channels for extra large files (default: False)
                Will automatically check file size and set this to True if doing bidi- and motion-correction,
                since Matlab limit is 4GB.
                }
        'hashid' (str)
            Generated hash using the above key-value pairs, specific to PARAMS.
            Note: full PID dict includes extra house-keeping info that gets its own hash id.       
    }
    '''
    
    PARAMS = dict()
    PARAMS['source'] = dict()

    acquisition_dir = os.path.join(rootdir, animalid, session, acquisition)
    print acquisition_dir
    
    # Check tiffs to see if should split-channels for Matlab-based MC:
    if correct_motion is True or correct_bidir is True:
        print os.listdir(os.path.join(acquisition_dir, run))
        rawdir_name = [r for r in os.listdir(os.path.join(acquisition_dir, run)) if 'raw' in r][0]
        rawtiff_dir = os.path.join(acquisition_dir, run, rawdir_name)
        tiffs = [t for t in os.listdir(rawtiff_dir) if t.endswith('tif')]
        file_sizes = [get_file_size(os.path.join(rawtiff_dir, t)) for t in tiffs]
        gb_files = [f for f in file_sizes if 'GB' in f]
        toobig = [g for g in gb_files if float(g.split('GB')[0]) > 4.0]
        if len(toobig) > 0:
            split_channels = True
        
    # ----------------------------------------------------------
    # Get source params:
    # ----------------------------------------------------------
    PARAMS['source']['rootdir'] = rootdir
    PARAMS['source']['animalid'] = animalid
    PARAMS['source']['session'] = session
    PARAMS['source']['acquisition'] = acquisition
    PARAMS['source']['run'] = run

    if not os.path.exists(os.path.join(acquisition_dir, run, 'processed')): 
        os.makedirs(os.path.join(acquisition_dir, run, 'processed'))
        
    rundir = os.path.join(acquisition_dir, run)
    
    processed_dirs = sorted([p for p in os.listdir(os.path.join(rundir, 'processed'))
                              if 'processed' in p], key=natural_keys)
    
    process_dict = load_processdict(acquisition_dir, run)
    
    rawdir_name = [r for r in os.listdir(rundir) if 'raw' in r and os.path.isdir(os.path.join(rundir, r))]
    
    if tiffsource is None or len(tiffsource) == 0:
        while True:
            if auto is True:
                tiffsource = rawdir_name[0]
                break
            print "TIFF SOURCE was not specified."
            raw_or_processed = raw_input('Enter <P> if source is processed, <R> if raw: ')
            if raw_or_processed == 'R':
                tiffsource = rawdir_name[0]
                break
            elif raw_or_processed =='P':
                print "Selected PROCESSED source."
                pp.pprint(processed_dirs)
                if process_dict is None or (len(process_dict.keys()) == 0 and len(processed_dirs) == 0):
                    tiffsource = 'processed%03d' % int(len(processed_dirs) + 1)
                else:
                    candidate_sources = sorted(process_dict.keys(), key=natural_keys)
                    for pdix, pdir in enumerate(candidate_sources):
                        print pdix, pdir
                    tiffsource_idx = input('Select IDX of processed dir to use as source: ')
                    if tiffsource_idx < len(candidate_sources):
                        tiffsource = candidate_sources[int(tiffsource_idx)]
                confirm_selection = raw_input('Selected %s. Press <Y> to confirm.' % tiffsource)
                if confirm_selection == 'Y':
                    break
                    
    if 'processed' in tiffsource:
        tiffsource_name = [p for p in os.listdir(os.path.join(rundir, 'processed')) if tiffsource in p][0]
        tiffsource_path = os.path.join(rundir, 'processed', tiffsource_name)
        if sourcetype is None or len(sourcetype) == 0:
            while True:
                if auto is True:
                    sourcetype = 'raw'
                    break
                processed_types = sorted([p for p in os.listdir(tiffsource_path)], key=natural_keys)
                print "Specified PROCESSED tiff source, but not which type:"
                for pidx,ptype in enumerate(processed_types):
                    print pidx, ptype
                processed_type_idx = raw_input('Enter IDX of processed source to use: ')
                if processed_type_idx in range(len(processed_types)):
                    processed_type = processed_types[processed_type_idx]
                    confirm_type = raw_input('Selected source %s/%s. Press <Y> to confirm: ' % (tiffsource_name, processed_type))
                    if confirm_type == 'Y':
                        sourcetype = processed_type
                        break
                
        tiffsource_child = [d for d in os.listdir(tiffsource_path) if sourcetype in d and os.path.isdir(os.path.join(tiffsource_path, d))][0]
        tiffsource_for_run = '/'.join(('processed', tiffsource_name, tiffsource_child))
    else:
        if not tiffsource == rawdir_name[0]:
            tiffsource = rawdir_name[0]
        tiffsource_for_run = tiffsource
       
    PARAMS['source']['tiffsource'] = tiffsource_for_run
    
    # ----------------------------------------------------------
    # Get preprocessing opts:
    # ----------------------------------------------------------
    PARAMS['preprocessing'] = set_preprocessing_params(correct_flyback=correct_flyback,
                             nflyback_frames=nflyback_frames,
                             correct_bidir=correct_bidir,
                             split_channels=split_channels)
    
    # ------------------------------------------
    # Check user-provided MC params:
    # ------------------------------------------
    print correct_motion
    PARAMS['motion'] = set_motion_params(correct_motion=correct_motion,
                            ref_channel=ref_channel,
                            ref_file=ref_file,
                            method=mc_method,
                            algorithm=mc_algorithm,
                            auto=auto)

    PARAMS['hashid'] = hashlib.sha1(json.dumps(PARAMS, sort_keys=True)).hexdigest()
    
    return PARAMS

def initialize_pid(PARAMS, acquisition_dir, run, auto=False):
    
    print "************************"
    print "Initialize PID."
    print "************************"
    process_id, is_new_pid, create_pid_file = get_process_id(PARAMS, acquisition_dir, run, auto=auto)

    pid = dict()
    version = pkg_resources.get_distribution('pipeline').version
    
    pid['version'] = version 
    pid['process_id'] = process_id
    pid['PARAMS'] = PARAMS
    pid['SRC'] = os.path.join(acquisition_dir, run, PARAMS['source']['tiffsource']) #source_dir
    pid['DST'] = os.path.join(acquisition_dir, run, 'processed', process_id)
    
    correct_flyback = pid['PARAMS']['preprocessing']['correct_flyback']
    correct_bidir = pid['PARAMS']['preprocessing']['correct_bidir']
    correct_motion = pid['PARAMS']['motion']['correct_motion']
    
    # Set source/dest dirs for Preprocessing tiffs:
    pid['PARAMS']['preprocessing']['sourcedir'] = pid['SRC']
    pid['PARAMS']['preprocessing']['destdir'] = None
    if correct_flyback is True:
        pid['PARAMS']['preprocessing']['destdir'] = os.path.join(pid['DST'], 'raw')
    if correct_bidir is True:
        pid['PARAMS']['preprocessing']['sourcedir'] = os.path.join(pid['DST'], 'raw')
        pid['PARAMS']['preprocessing']['destdir'] = os.path.join(pid['DST'], 'bidi')
       
    # Set source/dest dirs for Motion-Correction:
    pid['PARAMS']['motion']['sourcedir'] =  pid['PARAMS']['preprocessing']['destdir']
    pid['PARAMS']['motion']['destdir'] = None
    if correct_motion is True:
        if pid['PARAMS']['motion']['sourcedir'] is None:
            pid['PARAMS']['motion']['sourcedir'] = pid['SRC']
        pid['PARAMS']['motion']['destdir'] = os.path.join(pid['DST'], 'mcorrected')

    # TODO:  Generate hash for full PID dict
    pid['pid_hash'] = hashlib.sha1(json.dumps(pid, sort_keys=True)).hexdigest()[0:6]
    
    return pid

def load_processdict(acquisition_dir, run):
    processdict = None
    processdict_filepath = os.path.join(acquisition_dir, run, 'processed', 'pids_%s.json' % run)
    
    # Load analysis "processdict" file:
    if os.path.exists(processdict_filepath):
        print "exists!"
        with open(processdict_filepath, 'r') as f:
            processdict = json.load(f)
            
    return processdict

def get_process_id(PARAMS, acquisition_dir, run, auto=False):
    
    process_id = None
    
    print "********************************"
    print "Checking previous PROCESS IDs..."
    print "********************************"
    
    # Load previously created PIDs:
    processdict = load_processdict(acquisition_dir, run)
    
    #Check current PARAMS against existing PId params by hashid:
    if processdict is None or len(processdict.keys()) == 0:
        processdict = dict()
        existing_pids = []
        is_new_pid = True
        print "No existing PIDs found."
        create_pid_file = True
    else:
        create_pid_file = False
        existing_pids = sorted([str(k) for k in processdict.keys()], key=natural_keys)
        matching_pids = sorted([pid for pid in existing_pids if processdict[pid]['PARAMS']['hashid'] == PARAMS['hashid']], key=natural_keys)
        is_new_pid = False
        if len(matching_pids) > 0:            
            while True:
                print "WARNING **************************************"
                print "Current param config matches existing PID:"
                for pidx, pid in enumerate(matching_pids):
                    print "******************************************"
                    print pidx, pid
                    pp.pprint(processdict[pid])
                if auto is True:
                    check_pidx = ''
                else:
                    check_pidx = raw_input('Enter IDX of pid to re-use, or hit <ENTER> to create new: ')
                if len(check_pidx) == 0:
                    is_new_pid = True
                    break
                else:
                    confirm_reuse = raw_input('Re-use PID %s? Press <Y> to confirm, any key to try again:' % existing_pids[int(check_pidx)])
                    if confirm_reuse == 'Y':
                        is_new_pid = False
                        break
        else:
            is_new_pid = True

    if is_new_pid is True:
        # Create new PID by incrementing num of process dirs found:
        process_id = 'processed%03d' % int(len(existing_pids)+1)
        print "Creating NEW pid: %s" % process_id
    else:
        # Re-using an existing PID:
        process_id = existing_pids[int(check_pidx)]
        print "Reusing existing pid: %s" % process_id
 
    return process_id, is_new_pid, create_pid_file


def get_default_pid(rootdir='', animalid='', session='', acquisition='', run='',
                    correct_flyback=None, nflyback_frames=0):
    
    acquisition_dir = os.path.join(rootdir, animalid, session, acquisition)
    rundir = os.path.join(acquisition_dir, run)
    rawdir_name = [r for r in os.listdir(rundir) if 'raw' in r and os.path.isdir(os.path.join(rundir, r))]
    rawdir = rawdir_name[0]
    
    DEFPARAMS = set_params(rootdir=rootdir, animalid=animalid, session=session,
                         acquisition=acquisition, run=run, tiffsource=rawdir, sourcetype='raw',
                         correct_motion=False, correct_bidir=False,
                         correct_flyback=correct_flyback, nflyback_frames=nflyback_frames)
    
    # Generate new process_id based on input params:
    pid = initialize_pid(DEFPARAMS, acquisition_dir, run, auto=True)
    
    # UPDATE RECORDS:
    update_pid_records(pid, acquisition_dir, run)
    
    # STORE TMP FILE OF CURRENT PARAMS:
    tmp_pid_fn = 'tmp_pid_%s.json' % pid['pid_hash'][0:6]
    tmp_pid_dir = os.path.join(acquisition_dir, run, 'processed', 'tmp_pids')
    if not os.path.exists(tmp_pid_dir):
        os.makedirs(tmp_pid_dir)
    tmp_pid_path = os.path.join(tmp_pid_dir, tmp_pid_fn)
    write_dict_to_json(pid, tmp_pid_path)

    return pid

def update_pid_records(pid, acquisition_dir, run):

    print "************************"
    print "Updating JSONS..."
    print "************************"

    processdict_filepath = os.path.join(acquisition_dir, run, 'processed', 'pids_%s.json' % run)
    if os.path.exists(processdict_filepath):
        with open(processdict_filepath, 'r') as f:
            processdict = json.load(f)
    else:
        processdict = dict()

    process_id = pid['process_id'] 
    processdict[process_id] = pid

    #% Update Process Info DICT:
    write_dict_to_json(processdict, processdict_filepath)

    print "Process Info UPDATED."
    

def extract_options(options):
    parser = optparse.OptionParser()

    # PATH opts:
    parser.add_option('-P', '--sipath', action='store', dest='path_to_si_reader', default='~/Downloads/ScanImageTiffReader-1.1-Linux/share/python', help='path to dir containing ScanImageTiffReader.py')
    parser.add_option('-R', '--root', action='store', dest='rootdir', default='/nas/volume1/2photon/data', help='data root dir (root project dir containing all animalids) [default: /nas/volume1/2photon/data, /n/coxfs01/2pdata if --slurm]')
    parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')

    # Set specific session/run for current animal:
    parser.add_option('-S', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID')
    parser.add_option('-A', '--acq', action='store', dest='acquisition', default='FOV1', help="acquisition folder (ex: 'FOV1_zoom3x') [default: FOV1]")
    parser.add_option('-r', '--run', action='store', dest='run', default='', help="name of run dir containing tiffs to be processed (ex: gratings_phasemod_run1)")
    parser.add_option('--default', action='store_true', dest='default', default='store_false', help="Use all DEFAULT params, for params not specified by user (no interactive)")

    parser.add_option('-s', '--tiffsource', action='store', dest='tiffsource', default=None, help="name of folder containing tiffs to be processed (ex: processed001). should be child of <run>/processed/")
    parser.add_option('-t', '--sourcetype', action='store', dest='sourcetype', default='raw', help="type of source tiffs (e.g., bidi, raw, mcorrected) [default: 'raw']")
    parser.add_option('--slurm', action='store_true', dest='slurm', default=False, help="set if running as SLURM job on Odyssey")

    # Preprocessing params:
    parser.add_option('--bidi', action='store_true', dest='bidi', default=False, help='Set flag if correct bidirectional scanning phase offset.')
    parser.add_option('--flyback', action='store_true', dest='flyback', default=False, help='Set flag if need to correct extra flyback frames')
    parser.add_option('-F', '--nflyback', action='store', dest='nflyback_frames', default=0, help='Number of flyback frames to remove from top of each volume [default: 0]')

    # MOTION params:
    parser.add_option('--motion', action='store_true', dest='do_mc', default=False, help='Set flag if should run motion-correction.')
    parser.add_option('-c', '--channel', action='store', dest='ref_channel', default=1, help='Index of CHANNEL to use for reference if doing motion correction [default: 1]')
    parser.add_option('-f', '--file', action='store', dest='ref_file', default=1, help='Index of FILE to use for reference if doing motion correction [default: 1]')
    parser.add_option('-M', '--method', action='store', dest='mc_method', default=None, help='Method for motion-correction. OPTS: Acquisition2P, NoRMCorre [default: Acquisition2P]')
    parser.add_option('-a', '--algo', action='store', dest='algorithm', default=None, help='Algorithm to use for motion-correction, e.g., @withinFile_withinFrame_lucasKanade if method=Acquisition2P, or nonrigid if method=NoRMCorre')

    (options, args) = parser.parse_args(options) 
    
    if options.slurm is True:
        if 'coxfs01' not in options.rootdir:
            options.rootdir = '/n/coxfs01/julianarhee/testdata'
        if 'coxfs01' not in options.path_to_si_reader:
            options.path_to_si_reader = '/n/coxfs01/2p-pipeline/pkgs/ScanImageTiffReader-1.1-Linux/share/python'

    return options

def create_pid(options):
   
    options = extract_options(options)

    rootdir = options.rootdir
    animalid = options.animalid
    session = options.session
    acquisition = options.acquisition
    run = options.run

    tiffsource = options.tiffsource
    sourcetype = options.sourcetype
    
    auto = options.default

    # PREPROCESSING params:
    correct_bidir = options.bidi
    correct_flyback = options.flyback
    nflyback_frames = options.nflyback_frames

    # MOTION params:
    correct_motion = options.do_mc
    mc_method = options.mc_method
    mc_algorithm = options.algorithm
    ref_file = int(options.ref_file)
    ref_channel = int(options.ref_channel)

    # -------------------------------------------------------------
    # Set basename for files created containing meta/reference info:
    # -------------------------------------------------------------
    raw_simeta_basename = 'SI_%s' % run #functional_dir
    run_info_basename = '%s' % run #functional_dir
    pid_info_basename = 'pids_%s' % run
    # -------------------------------------------------------------

    # Create config of PARAMS:
    PARAMS = set_params(rootdir=rootdir, animalid=animalid, session=session,
                            acquisition=acquisition, run=run, tiffsource=tiffsource, sourcetype=sourcetype,
                            correct_bidir=correct_bidir, correct_flyback=correct_flyback, nflyback_frames=nflyback_frames,
                            correct_motion=correct_motion, ref_file=ref_file, ref_channel=ref_channel, 
                            mc_method=mc_method, mc_algorithm=mc_algorithm, auto=auto)
    
    # Generate new process_id based on input params:
    acquisition_dir = os.path.join(rootdir, animalid, session, acquisition)
    print "ACQ DIR:", acquisition_dir
    pid = initialize_pid(PARAMS, acquisition_dir, run, auto=auto)
    
    # UPDATE RECORDS:
    update_pid_records(pid, acquisition_dir, run)
    
    # STORE TMP FILE OF CURRENT PARAMS:
    tmp_pid_fn = 'tmp_pid_%s.json' % pid['pid_hash'][0:6]
    tmp_pid_dir = os.path.join(acquisition_dir, run, 'processed', 'tmp_pids')
    if not os.path.exists(tmp_pid_dir):
        os.makedirs(tmp_pid_dir)
    tmp_pid_path = os.path.join(tmp_pid_dir, tmp_pid_fn)
    write_dict_to_json(pid, tmp_pid_path)
       
    print "Params set for PID: %s" % pid['pid_hash']

    return pid

#%%

# if __name__ == '__main__':
#     main(sys.argv[1:])
    
def main(options):
    
    pid = create_pid(options)
    
    print "****************************************************************"
    print "Created PID."
    print "----------------------------------------------------------------"
    pp.pprint(pid)
    print "****************************************************************"
    
if __name__ == '__main__':
    main(sys.argv[1:]) 

