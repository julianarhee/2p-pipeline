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
import optparse
import sys
from pipeline.python.preprocessing.process_raw import process_pid
from multiprocessing import Process

pp = pprint.PrettyPrinter(indent=4)

# GENERAL METHODS:
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]

# PATH opts:
# -------------------------------------------------------------
parser = optparse.OptionParser()

parser.add_option('-R', '--root', action='store', dest='rootdir', default='/nas/volume1/2photon/data', help='source dir (root project dir containing all expts) [default: /nas/volume1/2photon/data]')
parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')
parser.add_option('-S', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID') 
parser.add_option('--slurm', action='store_true', dest='slurm', default=False, help="set if running as SLURM job on Odyssey")

(options, args) = parser.parse_args() 

# -------------------------------------------------------------
# INPUT PARAMS:
# -------------------------------------------------------------
rootdir = options.rootdir #'/nas/volume1/2photon/projects'
animalid = options.animalid
session = options.session #'20171003_JW016' #'20170927_CE059'
slurm = options.slurm

session_dir = os.path.join(rootdir, animalid, session)
pid_dir = os.path.join(session_dir, 'tmp_spids')
if not os.path.exists(pid_dir) or len(os.listdir(pid_dir)) == 0:
    print "No PIDs to process."
    process_batch = False
else:
    process_batch = True
    pid_paths = [os.path.join(pid_dir, p) for p in os.listdir(pid_dir) if p.endswith('json') and 'pid_' in p]


for pid_path in pid_paths:
    with open(pid_path, 'r') as f:
        pinfo = json.load(f)
    opts = ['-R', pinfo['rootdir'], '-i', pinfo['animalid'], '-S', pinfo['session'], '-A', pinfo['acquisition'], '-r', pinfo['run'], '-p', pinfo['pid']]
    Process = target=process_pid, args=(opts,)).start()
 

