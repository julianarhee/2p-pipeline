#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 14:34:35 2018

@author: julianarhee
"""

import sys
import os
import json
import logging
from pipeline.python.evaluation.evaluate_motion_correction import evaluate_motion

def evaluate_motion_pid(pid_filepath, zproj='mean', nprocs=12):
    
    if 'tmp_spids' in pid_filepath:  # This means create_session_pids.py was run to format to set of run PIDs in current session 
        with open(pid_filepath, 'r') as f:
            pinfo = json.load(f)
    else:
        tmp_rid_dir = os.path.split(pid_filepath)[0]
        if not os.path.exists(pid_filepath):
            rid_fn = os.path.split(pid_filepath)[1]
            completed_path = os.path.join(tmp_rid_dir, 'completed', rid_fn)
            assert os.path.exists(os.path.join(completed_path)), "No such RID file exists in either %s or %s." % (pid_filepath, completed_path)
            pid_filepath = completed_path
       
        if os.path.exists(pid_filepath):
            with open(pid_filepath, 'r') as f:
                PID = json.load(f)
            
        pinfo = PID['PARAMS']['source']
        pinfo['process_id'] = PID['process_id']       
        
    #roi_hash = os.path.splitext(os.path.split(rid_filepath)[-1])[0].split('_')[-1]
        
    eval_opts = ['-D', pinfo['rootdir'],
                 '-i', pinfo['animalid'],
                 '-A', pinfo['acquisition'],
                 '-S', pinfo['session'],
                 '-R', pinfo['run'],
                 '-P', pinfo['process_id'],
                 '--zproj=%s' % zproj,
                 '--par',
                 '-n', nprocs]
        
    eval_filepath, roi_source_basedir, tiff_source_basedir, excluded_tiffs = evaluate_motion(eval_opts)
        
    return eval_filepath


def main():

    pid_path = sys.argv[1]
    if len(sys.argv) > 2:
        zproj = sys.argv[2]
    if len(sys.argv) > 3:
        nprocs = int(sys.argv[3])
    else:
        nprocs = 12
    
    # Open LOGGING dir in tmp PID dir:
    pid_id = os.path.splitext(os.path.split(pid_path)[-1])[0].split('_')[-1]
    logdir = os.path.join(os.path.split(pid_path)[0], "logging_%s" % pid_id)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    print "Logging to: %s" % logdir
    logging.basicConfig(level=logging.DEBUG, filename="%s/logfile_%s_mceval" % (logdir, pid_id), filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    logging.info("PID %s -- starting..." % pid_id)
    logging.info(pid_path)
    
    pidhash = evaluate_motion_pid(pid_path, zproj=zproj, nprocs=nprocs)

    logging.info("FINISHED PROCESSING PID %s." % pidhash)

     # Clean up session info dict:
    tmp_session_info_dir = os.path.split(pid_path)[0]
    completed_session_info_dir = os.path.join(tmp_session_info_dir, 'completed')
    if not os.path.exists(completed_session_info_dir):
        os.makedirs(completed_session_info_dir)

    completed_pinfo_path = os.path.join(completed_session_info_dir, os.path.split(pid_path)[-1])
    if os.path.exists(pid_path):
        os.rename(pid_path, completed_pinfo_path)
        logging.info("Cleaned up session info file: %s" % completed_pinfo_path)
    
    logging.info("PID %s -- MC EVALUATION!" % pid_id)

if __name__ == "__main__":
    main()