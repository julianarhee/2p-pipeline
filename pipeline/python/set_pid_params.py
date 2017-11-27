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

pp = pprint.PrettyPrinter(indent=4)

def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]



def load_processdict(acquisition_dir, run):
    processdict = None
    processdict_filepath = os.path.join(acquisition_dir, run, 'processed', 'pid_info_%s.json' % run)
    
    # Load analysis "processdict" file:
    if os.path.exists(processdict_filepath):
        print "exists!"
        with open(processdict_filepath, 'r') as f:
            processdict = json.load(f)
            
    return processdict
              
    
def get_process_id(processdict):
    
    print "************************"
    print "Generating PROCESS ID..."
    print "************************"
    
    process_id = None
    
    # First check current params against existing analyses:
    if processdict is None or len(processdict.keys()) == 0:
        processdict = dict()
        existing_pids = []
        is_new_pid = True
        print "No existing PIDs found."
        create_pid_file = True
    else:
        create_pid_file = False
        existing_pids = sorted([str(k) for k in processdict.keys()], key=natural_keys)

        # Show existing PIDs:
        while True:
            print "Found existing PIDs:"
            for pidx, pid in enumerate(existing_pids):
                print pidx, pid
            
            check_pidx = raw_input('Enter IDX of pid to view or hit <ENTER> to create new: ')
            if len(check_pidx) == 0:
                is_new_pid = True
                break
            else:
                print "Viewing PID: %s" % existing_pids[int(check_pidx)]
                pp.pprint(processdict[existing_pids[int(check_pidx)]])
                reuse_idx = raw_input('Press <R> to re-use current pid, or hit <ENTER> to view list.')
                if reuse_idx == 'R':
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


def get_basic_pid(rootdir=rootdir, animalid=animalid, session=session,
                         acquisition=acquisition, run=run):
    tiffsource = 'raw'
    correct_motion = False
    correct_bidir = False
    
    pid = set_processing_params(rootdir=rootdir, animalid=animalid, session=session,
                         acquisition=acquisition, run=run, tiffsource=tiffsource,
                         correct_motion=correct_motion, correct_bidir=correct_bidir, correct_flyback=correct_flyback)
    
    return pid


def initialize_pid(PARAMS, process_dict, acquisition_dir, run, tiffsource):
    
    print "************************"
    print "Initialize PID."
    print "************************"

    pid = dict()
    version = pkg_resources.get_distribution('pipeline').version
    pid['version'] = version 
    
    processed_dirs = sorted([p for p in os.listdir(os.path.join(acquisition_dir, run, 'processed'))
                              if 'processed' in p], key=natural_keys)
    
    if tiffsource is None:
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

    # SET SOURCE:
    if tiffsource is 'raw':
        pid['SRC'] = os.path.join(acquisition_dir, run, tiffsource) #source_dir
    else:
        pid['SRC'] = os.path.join(acquisition_dir, run, 'processed', tiffsource) 
        
    # SET DEST:

    process_id, is_new_pid, create_pid_file = get_process_id(process_dict)
    
    pid['DST'] = os.path.join(acquisition_dir, run, 'processed', process_id)
    
    pid['process_id'] = process_id

    pid['PARAMS'] = PARAMS
 
    return pid
     
    
def set_mcparams(acquisition_dir, run,
                    correct_motion=True,
                    ref_channel=1,
                    ref_file=1,
                    method=None,
                    algorithm=None,
                    ):

    mcparams = dict()
    mcparams['correct_motion'] = correct_motion
    mcparams['method'] = method
    mcparams['algorithm'] = algorithm
    mcparams['ref_channel'] = ref_channel
    mcparams['ref_file'] = ref_file
#     mcparams['source_dir'] = source
#     mcparams['dest_dir'] = destination

#     if source is None:
#         mcparams['source_dir'] = os.path.join(acquisition_dir, run, 'raw')
#     if destination is None:
#         mcparams['dest_dir'] = os.path.join(acquisition_dir, run, 'processed', process_id)

    return mcparams


def update_records(pid, processdict, acquisition_dir, run):

    print "************************"
    print "Updating JSONS..."
    print "************************"

    processdict_filepath = os.path.join(acquisition_dir, run, 'processed', 'pid_info_%s.json' % run)
    #processdict_tablepath = os.path.join(acquisition_dir, run, 'processed', 'pid_info_%s.txt' % run)

    if processdict is None:
        processdict = dict()
    process_id = pid['process_id'] 
    processdict[process_id] = pid

    #% Update Process Info DICT:
    with open(processdict_filepath, 'w') as f:
        json.dump(processdict, f, sort_keys=True, indent=4)

    print "Process Info UPDATED."


def set_processing_params(rootdir=rootdir, animalid=animalid, session=session,
                         acquisition=acquisition, run=run, tiffsource=None,
                         correct_bidir=False, correct_flyback=False, nflyback_frames=None,
                         correct_motion=False, ref_file=1, ref_channel=1,
                         mc_method=None, mc_algorithm=None):
    
    mc_methods = ['Acquisition2P', 'NoRMCorre']
    mc_algos = dict((mc, []) for mc in mc_methods)
    mc_algos = {'Acquisition2P': ['@withinFile_withinFrame_lucasKanade', '@lucasKanade_plus_nonrigid'],
                'NoRMCorre': ['rigid', 'nonrigid']}

    PARAMS = dict()
    PARAMS['source'] = dict()
    PARAMS['preprocessing'] = dict()
    PARAMS['motion'] = dict()

    acquisition_dir = os.path.join(rootdir, animalid, session, acquisition)
    
    # ----------------------------------------------------------
    # Get preprocessing opts:
    # ----------------------------------------------------------
    PARAMS['source']['rootdir'] = rootdir
    PARAMS['source']['animalid'] = animalid
    PARAMS['source']['session'] = session
    PARAMS['source']['acquisition'] = acquisition
    PARAMS['source']['run'] = run

    PARAMS['preprocessing']['flyback_corrected'] = correct_flyback
    PARAMS['preprocessing']['nflyback_frames'] = nflyback_frames
    PARAMS['preprocessing']['bidir_corrected'] = correct_bidir

    # ------------------------------------------
    # Check user-provided MC params:
    # ------------------------------------------
    if correct_motion is True:
        if mc_method is None:
            while True:
                print "No MC method specified. Use default [Acquisition2P]?"
                mc_choice = raw_input('Enter <Y> to use default, or <o> to see options:')
                if mc_choice == 'Y':
                    print "Using default."
                    mc_method = 'Acquisition2P'
                    break
                elif mc_choice == 'o':
                    for mcid, mcname in enumerate(mc_methods):
                        print mcid, mcname
                    mc_select = input('Enter IDX of motion-correction method to use:')
                    mc_method = mc_methods[mc_select]
                    break
        if mc_algorithm is None or (mc_algorithm not in mc_algos[mc_method]):
            print "No MC algorithm specified... Here are the options:"
            for algonum, algoname in enumerate(mc_algos[mc_method]):
                print algonum, algoname
            algo_choice = input('Enter IDX of mc algorithm to use:')
            mc_algorithm = mc_algos[mc_method][algo_choice] 
    else:
        ref_channel = None
        ref_file = None
        mc_method = None
        mc_algorithm = None

    mcparams = set_mcparams(acquisition_dir, run,
                            correct_motion=correct_motion,
                            ref_channel=ref_channel,
                            ref_file=ref_file,
                            method=mc_method,
                            algorithm=mc_algorithm)
    #pp.pprint(mcparams)
    PARAMS['motion'] = mcparams

    # INITIALIZE PID:
    processdict = load_processdict(acquisition_dir, run)
    #pp.pprint(processdict)
    pid = initialize_pid(PARAMS, processdict, acquisition_dir, run, tiffsource)
    
    # UPDATE RECORDS:
    update_records(pid, processdict, acquisition_dir, run)
    
    # STORE TMP FILE OF CURRENT PARAMS:
    with open(os.path.join(acquisition_dir, run, 'processed', 'tmp_processparams.json'), 'w') as f:
        json.dump(pid, f, indent=4, sort_keys=True)
        
    return pid


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

    parser.add_option('-s', '--tiffsource', action='store', dest='tiffsource', default='', help="name of folder containing tiffs to be processed (ex: processed001). should be child of <run>/processed/")

    # Preprocessing params:
    parser.add_option('--bidi', action='store_true', dest='bidi', default=False, help='Set flag if correct bidirectional scanning phase offset.')
    parser.add_option('--flyback', action='store_true', dest='flyback', default=False, help='Set flag if need to correct extra flyback frames during')
    parser.add_option('-F', '--nflyback' action='store', dest='nflyback_frames', default=0, help='Number of flyback frames to remove from top of each volume [default: 0]')

    # MOTION params:
    parser.add_option('--motion', action='store_true', dest='mc', default=False, help='Set flag if should run motion-correction.')
    parser.add_option('-c', '--channel', action='store', dest='ref_channel', default=1, help='Index of CHANNEL to use for reference if doing motion correction [default: 1]')
    parser.add_option('-f', '--file', action='store', dest='ref_file', default=1, help='Index of FILE to use for reference if doing motion correction [default: 1]')
    parser.add_option('-M', '--method', action='store', dest='mc_method', default=None, help='Method for motion-correction. OPTS: Acquisition2P, NoRMCorre [default: Acquisition2P]')
    parser.add_option('-a', '--algo', action='store', dest='algorithm', default=None, help='Algorithm to use for motion-correction, e.g., @withinFile_withinFrame_lucasKanade if method=Acquisition2P, or nonrigid if method=NoRMCorre')

    (options, args) = parser.parse_args() 

    mc_methods = ['Acquisition2P', 'NoRMCorre']
    mc_algos = dict((mc, []) for mc in mc_methods)
    mc_algos = {'Acquisition2P': [@withinFile_withinFrame_lucasKanade, @lucasKanade_plus_nonrigid],
                'NoRMCorre': ['rigid', 'nonrigid']}

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

    pid = set_processing_params(rootdir=rootdir, animalid=animalid, session=session,
                               acquisition=acquisition, run=run, tiffsource=tiffsource,
                               correct_bidir=correct_bidir, correct_flyback=correct_flyback, nflyback_frames=nflyback_frames,
                             correct_motion=correct_motion, ref_file=ref_file, ref_channel=ref_channel,
                             mc_method=mc_method, mc_algorithm=mc_algorithm)
    pp.pprint(pid)


#%%
if __name__ == '__main__':
    main(sys.argv[1:])
    

