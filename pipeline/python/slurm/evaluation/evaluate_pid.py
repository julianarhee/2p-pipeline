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
        # Get process ID NAME:
        rundir = os.path.join(pinfo['rootdir'], pinfo['animalid'], pinfo['session'], pinfo['acquisition'], pinfo['run'])
        with open(os.path.join(rundir, 'processed', 'pids_%s.json' % pinfo['run']), 'r') as f:
            pdict = json.load(f)
        pkey = [k for k in pdict.keys() if pdict[k]['pid_hash'] == pinfo['pid']][0]
        pinfo['process_id'] = pkey
        del pdict
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
        pinfo['pid'] = PID['pid_hash']
 
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
         
    eval_outfile = evaluate_motion(eval_opts)

    print "----------------------------------------------------------------"
    print "Finished evulation motion-correction."
    print "Saved output to:"
    print eval_outfile
    print "----------------------------------------------------------------"
    return eval_outfile, pinfo['pid']


def main():

    pid_path = sys.argv[1]
    if len(sys.argv) > 2:
        zproj = str(sys.argv[2])
    else:
        zproj = 'mean'
    if len(sys.argv) > 3:
        nprocs = int(sys.argv[3])
    else:
        nprocs = 12
    
    # Open LOGGING dir in tmp PID dir:
    pid_id = os.path.splitext(os.path.split(pid_path)[-1])[0].split('_')[-1]
    if 'completed' in pid_path:
        spid_dir = os.path.split(os.path.split(pid_path)[0])[0]
    else:
        spid_dir = os.path.split(pid_path)[0] 
    logdir = os.path.join(spid_dir, "logging_%s" % pid_id)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    print "Logging to: %s" % logdir
    logging.basicConfig(level=logging.DEBUG, filename="%s/logfile_%s_mceval" % (logdir, pid_id), filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    logging.info("PID %s -- starting..." % pid_id)
    logging.info(pid_path)
    
    outfilepath, pidhash = evaluate_motion_pid(pid_path, zproj=zproj, nprocs=nprocs)

    logging.info("FINISHED evaluating MOTION for PID %s." % pidhash)
    logging.info("MC evaluation results saved to: %s" % outfilepath)

     # Clean up session info dict:
    #tmp_session_info_dir = os.path.split(pid_path)[0]
    pid_fn = os.path.split(pid_path)[-1]
    completed_session_info_dir = os.path.join(spid_dir, 'completed')
    if not os.path.exists(completed_session_info_dir):
        os.makedirs(completed_session_info_dir)

    #completed_pinfo_path = os.path.join(completed_session_info_dir, os.path.split(pid_path)[-1])
    if 'completed' not in pid_path:
        shutil.move(os.path.join(spid_dir, pid_fn), os.path.join(completed_session_info_dir, pid_fn))
        logging.info("Cleaned up session info file: %s" % os.path.join(completed_session_info_dir, pid_fn))
    
    logging.info("PID %s -- MC EVALUATION!" % pid_id)

    # Move log files to completed, if no error:
    
if __name__ == "__main__":
    main()
