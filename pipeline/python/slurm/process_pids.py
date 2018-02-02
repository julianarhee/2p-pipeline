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


def main():

    pid_path = sys.argv[1]
    pid_id = os.path.splitext(os.path.split(pid_path)[-1])[0].split('_')[-1]

    logging.basicConfig(level=logging.DEBUG, filename="logfile_%s" % pid_id, filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    logging.info("PID %s -- starting..." % pid_id)
 
    logging.info(pid_filepath)
    pidhash = process_run_pid(pid_filepath)

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
