#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 15:23:13 2018

@author: julianarhee
"""

import sys
import os
import json
import optparse

from pipeline.python.preprocessing.process_raw import process_pid
import logging

def fake_process_run(options):
    parser = optparse.OptionParser()

    # PATH opts:
    parser.add_option('-D', '--root', action='store', dest='rootdir', default='/nas/volume1/2photon/data', help='source dir (root project dir containing all expts) [default: /nas/volume1/2photon/data]')
    parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')
    parser.add_option('-S', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID') 
    parser.add_option('-A', '--acq', action='store', dest='acquisition', default='', help="acquisition folder (ex: 'FOV1_zoom3x')")
    parser.add_option('-R', '--run', action='store', dest='run', default='', help='name of run to process') 
    parser.add_option('-p', '--pid', action='store', dest='pid_hash', default='', help="PID hash of current processing run (6 char), default will create new if set_pid_params.py not run")

    parser.add_option('--slurm', action='store_true', dest='slurm', default=False, help="set if running as SLURM job on Odyssey")
    parser.add_option('--default', action='store_true', dest='default', default='store_false', help="Use all DEFAULT params, for params not specified by user (no interactive)")
    tiffsource = 'raw'

    (options, args) = parser.parse_args(options) 

    # -------------------------------------------------------------
    # INPUT PARAMS:
    # -------------------------------------------------------------
    new_acquisition = options.new_acquisition
    save_tiffs = options.save_tiffs

    rootdir = options.rootdir #'/nas/volume1/2photon/projects'
    animalid = options.animalid
    session = options.session #'20171003_JW016' #'20170927_CE059'
    acquisition = options.acquisition #'FOV1' #'FOV1_zoom3x'
    run = options.run
    pid_hash = options.pid_hash
    repo_path = options.repo_path

    acquisition_dir = os.path.join(rootdir, animalid, session, acquisition)
    print acquisition_dir

    # ===========================================================================
    # If PID specified, that takes priority:
    # ===========================================================================
    execute_flyback = False
    execute_bidi = False
    execute_motion = False
    if len(pid_hash) > 0:
        tmp_pid_fn = 'tmp_pid_%s.json' % pid_hash
        with open(os.path.join(acquisition_dir, run, 'processed', 'tmp_pids', tmp_pid_fn), 'r') as f:
            PID = json.load(f)
        execute_flyback = PID['PARAMS']['preprocessing']['correct_flyback']
        nflyback = int(PID['PARAMS']['preprocessing']['nflyback_frames'])
        execute_bidi = PID['PARAMS']['preprocessing']['correct_bidir']
        execute_motion = PID['PARAMS']['motion']['correct_motion']
    print "PID %s -- Flyback: %s" % (pid_hash, str(execute_flyback))
    print "PID %s -- Bidir: %s" % (pid_hash, str(execute_bidi))
    print "PID %s -- Motion: %s" % (pid_hash, str(execute_motion))

    return pid_hash
    
def process_run_pid(pid_filepath):

    if 'tmp_spids' in pid_filepath:  # This means create_session_pids.py was run to format to set of run PIDs in current session 
        with open(pid_filepath, 'r') as f:
            pinfo = json.load(f)
    else:
        with open(pid_filepath, 'r') as f:
            tmppid = json.load(f)
        pinfo = tmppid['PARAMS']['source']
        pinfo['pid'] = tmppid['pid_hash']       
    
    logging.info('PID opts:')
    logging.info(pinfo)
       
    popts = ['-D', pinfo['rootdir'], '-i', pinfo['animalid'], '-S', pinfo['session'], '-A', pinfo['acquisition'], '-R', pinfo['run'], '-p', pinfo['pid'], '--slurm', '--zproject'] 
    pidhash = process_pid(popts)
        
    return pidhash

def main(pid_filepath):

    logging.info(pid_filepath)
    pidhash = process_run_pid(pid_filepath)

    logging.info("FINISHED PROCESSING PID %s." % pidhash)

if __name__ == "__main__":

    pid_path = sys.argv[1]
    pid_id = os.path.splitext(os.path.split(pid_path)[-1])[0].split('_')[-1]

    logging.basicConfig(level=logging.DEBUG, filename="logfile_%s" % pid_id, filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    logging.info("PID %s -- starting..." % pid_id)
    
    main(pid_path)
    
    # Clean up session info dict:
    tmp_session_info_dir = os.path.split(pid_path)[0]
    completed_session_info_dir = os.path.join(tmp_session_info_dir, 'completed')
    if not os.path.exists(completed_session_info_dir):
        os.makedirs(completed_session_info_dir)

    completed_pinfo_path = os.path.join(completed_session_info_dir, os.path.split(pid_path)[-1])
    if os.path.exists(pid_path):
        os.rename(pid_path, completed_pinfo_path)
        logging.info("Cleaned up session info file: %s" % completed_pinfo_path)
    
    logging.info("PID %s -- done!" % pid_id)
