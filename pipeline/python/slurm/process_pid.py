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
import logging
from pipeline.python.preprocessing.process_raw import process_pid

def process_pid_from_file(pid_filepath):

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

def main():

    pid_path = sys.argv[1]
    pid_id = os.path.splitext(os.path.split(pid_path)[-1])[0].split('_')[-1]
    logdir = os.path.join(os.path.split(pid_path)[0], "logging_%s" % pid_id)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    print "Logging to: %s" % logdir

    logging.basicConfig(level=logging.DEBUG, filename="%s/logfile_%s_pid" % (logdir, pid_id), filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    logging.info("PID %s -- starting..." % pid_id)
 
    logging.info(pid_path)
    pidhash = process_pid_from_file(pid_path)

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
    
    logging.info("PID %s -- done!" % pid_id)

if __name__ == "__main__":
    main()
