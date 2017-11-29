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
    
# PID-SPECIFIC METHODS:
def load_processdict(acquisition_dir, run):
    processdict = None
    processdict_filepath = os.path.join(acquisition_dir, run, 'processed', 'pid_info_%s.json' % run)
    
    # Load analysis "processdict" file:
    if os.path.exists(processdict_filepath):
        print "exists!"
        with open(processdict_filepath, 'r') as f:
            processdict = json.load(f)
            
    return processdict
              
def get_process_id(PARAMS, acquisition_dir, run)
    
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
        if len(matching_pids) > 0:
            while True:
                print "WARNING **********************************"
                print "Current param config matches existing PID:"
                for pidx, pid in enumerate(matching_pids):
                    print "******************************************"
                    print pidx, pid
                    pp.pprint(processdict[pid])
                check_pidx = raw_input('Enter IDX of pid to re-use, or hit <ENTER> to create new: ')
                if len(check_pidx) == 0:
                    is_new_pid = True
                    break
                else:
                    
                    confirm_reuse = raw_input('Re-use PID %s? Press <Y> to confirm, any key to try again:' % existing_pids[int(check_pidx)])
                    if confirm_reuse == 'Y':
                        is_new_pid = False
                        break

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
    
    DEFPARAMS = set_params(rootdir=rootdir, animalid=animalid, session=session,
                         acquisition=acquisition, run=run, tiffsource='raw',
                         correct_motion=False, correct_bidir=False,
                         correct_flyback=correct_flyback, nflyback_frames=nflyback_frames)
    
    # Generate new process_id based on input params:
    pid = initialize_pid(DEFPARAMS, acquisition_dir, run):
    
    # UPDATE RECORDS:
    update_records(pid, acquisition_dir, run)
    
    # STORE TMP FILE OF CURRENT PARAMS:
    tmp_pid_fn = 'tmp_pid_params_%s.json' % pid['hashid'][0:8]
    with open(os.path.join(acquisition_dir, run, 'processed', tmp_pid_fn), 'w') as f:
        json.dump(pid, f, indent=4, sort_keys=True)
        
    return pid

def initialize_pid(PARAMS, acquisition_dir, run):
    
    print "************************"
    print "Initialize PID."
    print "************************"
    process_id, is_new_pid, create_pid_file = get_process_id(PARAMS, acquisition_dir, run)

    pid = dict()
    version = pkg_resources.get_distribution('pipeline').version
    
    pid['version'] = version 
    pid['process_id'] = process_id
    pid['PARAMS'] = PARAMS
    pid['SRC'] = os.path.join(acquisition_dir, run, PARAMS['source']['tiffsource']) #source_dir
    pid['DST'] = os.path.join(acquisition_dir, run, 'processed', process_id)
    
    correct_flyback = pid['PARAMS']['preprocessing']['correct_flyback']
    correct_bidir = pid['PARAMS']['preprocessing']['correct_bidir']
    correct_motion = pid['PARAMS']['preprocessing']['correct_motion']
    
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
    pid['hashid'] = hashlib.sha1(json.dumps(pid, sort_keys=True)).hexdigest()
    
    return pid
     
    
def set_mcparams(correct_motion=False,
                 ref_channel=0,
                 ref_file=0,
                 method=None,
                 algorithm=None
                ):

    mc_methods = ['Acquisition2P', 'NoRMCorre']
    mc_algos = dict((mc, []) for mc in mc_methods)
    mc_algos = {'Acquisition2P': ['@withinFile_withinFrame_lucasKanade', '@lucasKanade_plus_nonrigid'],
                'NoRMCorre': ['rigid', 'nonrigid']}

    if correct_motion is True:
        if method is None:
            while True:
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


def set_preparams(correct_flyback=False,
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



def set_params(rootdir='', animalid='', session='', acquisition='', run='', tiffsource=None,
               correct_bidir=False, correct_flyback=False, nflyback_frames=None,
               split_channels=False, correct_motion=False, ref_file=1, ref_channel=1,
               mc_method=None, mc_algorithm=None):
    
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
    
    # Check tiffs to see if should split-channels for Matlab-based MC:
    if correct_motion is True or correct_bidir is True:
        rawtiff_dir = os.path.join(acquisition_dir, run, 'raw')
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

    processed_dirs = sorted([p for p in os.listdir(os.path.join(acquisition_dir, run, 'processed'))
                              if 'processed' in p], key=natural_keys)
 
    if tiffsource is None or len(tiffsource) == 0:
        while True:
            print "TIFF SOURCE was not specified."
            tiffsource_type = raw_input('Enter <P> if source is processed, <R> if raw: ')
            if tiffsource_type == 'R':
                tiffsource = 'raw'
                break
            elif tiffsource_type =='P':
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
                    
    PARAMS['source']['tiffsource'] = tiffsource
    
    # ----------------------------------------------------------
    # Get preprocessing opts:
    # ----------------------------------------------------------
    preparams = set_preparams(correct_flyback=correct_flyback,
                             nflyback_frames=nflyback_frames,
                             correct_bidir=correct_bidir,
                             split_channels=split_channels)
    #pp.pprint(preparams)
    PARAMS['preprocessing'] = preparams
    
    # ------------------------------------------
    # Check user-provided MC params:
    # ------------------------------------------
    print correct_motion
    mcparams = set_mcparams(correct_motion=correct_motion,
                            ref_channel=ref_channel,
                            ref_file=ref_file,
                            method=mc_method,
                            algorithm=mc_algorithm)
    #pp.pprint(mcparams)
    PARAMS['motion'] = mcparams
    
    PARAMS['hashid'] = hashlib.sha1(json.dumps(PARAMS, sort_keys=True)).hexdigest()
    
    return PARAMS


def update_records(pid, acquisition_dir, run):

    print "************************"
    print "Updating JSONS..."
    print "************************"

    processdict_filepath = os.path.join(acquisition_dir, run, 'processed', 'pid_info_%s.json' % run)
    if os.path.exists(processdict_filepath):
        with open(processdict_filepath, 'r') as f:
            processdict = json.load(f)
    else:
        processdict = dict()

    process_id = pid['process_id'] 
    processdict[process_id] = pid

    #% Update Process Info DICT:
    with open(processdict_filepath, 'w') as f:
        json.dump(processdict, f, sort_keys=True, indent=4)

    print "Process Info UPDATED."



def main(options):
    parser = optparse.OptionParser()

    # PATH opts:
    parser.add_option('-P', '--sipath', action='store', dest='path_to_si_reader', default='~/Downloads/ScanImageTiffReader-1.1-Linux/share/python', help='path to dir containing ScanImageTiffReader.py')
    parser.add_option('-R', '--root', action='store', dest='rootdir', default='/nas/volume1/2photon/data', help='data root dir (root project dir containing all animalids) [default: /nas/volume1/2photon/data, /n/coxfs01/2pdata if --slurm]')
    parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')

    # Set specific session/run for current animal:
    parser.add_option('-S', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID')
    parser.add_option('-A', '--acq', action='store', dest='acquisition', default='FOV1', help="acquisition folder (ex: 'FOV1_zoom3x') [default: FOV1]")
    parser.add_option('-r', '--run', action='store', dest='run', default='', help="name of run dir containing tiffs to be processed (ex: gratings_phasemod_run1)")

    parser.add_option('-s', '--tiffsource', action='store', dest='tiffsource', default=None, help="name of folder containing tiffs to be processed (ex: processed001). should be child of <run>/processed/")

    # Preprocessing params:
    parser.add_option('--bidi', action='store_true', dest='bidi', default=False, help='Set flag if correct bidirectional scanning phase offset.')
    parser.add_option('--flyback', action='store_true', dest='flyback', default=False, help='Set flag if need to correct extra flyback frames')
    parser.add_option('-F', '--nflyback', action='store', dest='nflyback_frames', default=0, help='Number of flyback frames to remove from top of each volume [default: 0]')

    # MOTION params:
    parser.add_option('--motion', action='store_true', dest='mc', default=False, help='Set flag if should run motion-correction.')
    parser.add_option('-c', '--channel', action='store', dest='ref_channel', default=1, help='Index of CHANNEL to use for reference if doing motion correction [default: 1]')
    parser.add_option('-f', '--file', action='store', dest='ref_file', default=1, help='Index of FILE to use for reference if doing motion correction [default: 1]')
    parser.add_option('-M', '--method', action='store', dest='mc_method', default=None, help='Method for motion-correction. OPTS: Acquisition2P, NoRMCorre [default: Acquisition2P]')
    parser.add_option('-a', '--algo', action='store', dest='algorithm', default=None, help='Algorithm to use for motion-correction, e.g., @withinFile_withinFrame_lucasKanade if method=Acquisition2P, or nonrigid if method=NoRMCorre')

    (options, args) = parser.parse_args(options) 

    rootdir = options.rootdir
    animalid = options.animalid
    session = options.session
    acquisition = options.acquisition
    run = options.run

    tiffsource = options.tiffsource

    # PREPROCESSING params:
    correct_bidir = options.bidi
    correct_flyback = options.flyback
    nflyback_frames = options.nflyback_frames

    # MOTION params:
    correct_motion = options.mc
    mc_method = options.mc_method
    mc_algorithm = options.algorithm
    ref_file = options.ref_file
    ref_channel = options.ref_channel

    # Create config of PARAMS:
    PARAMS = set_params(rootdir=rootdir, animalid=animalid, session=session,
                            acquisition=acquisition, run=run, tiffsource=tiffsource,
                            correct_bidir=correct_bidir, correct_flyback=correct_flyback, nflyback_frames=nflyback_frames,
                            correct_motion=correct_motion, ref_file=ref_file, ref_channel=ref_channel, 
                            mc_method=mc_method, mc_algorithm=mc_algorithm)
    
    # Generate new process_id based on input params:
    pid = initialize_pid(PARAMS, acquisition_dir, run):
    
    # UPDATE RECORDS:
    update_records(pid, acquisition_dir, run)
    
    # STORE TMP FILE OF CURRENT PARAMS:
    tmp_pid_fn = 'tmp_pid_params_%s.json' % pid['hashid'][0:6]
    tmp_pid_dir = os.path.join(acquisition_dir, run, 'processed', 'tmp_pids')
    if not os.path.exists(tmp_pid_dir):
        os.makedirs(tmp_pid_dir)
    with open(os.path.join(tmp_pid_dir, tmp_pid_fn), 'w') as f:
        json.dump(pid, f, indent=4, sort_keys=True)

#%%

if __name__ == '__main__':
    main(sys.argv[1:])
    

