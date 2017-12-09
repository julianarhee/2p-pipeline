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
from pipeline.python.set_pid_params import create_pid
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
parser.add_option('--flyback', action='store_true', dest='correct_flyback', default=False, help="do flyback correction on acquisitions with _volume_ in name")
parser.add_option('-F', '--nflyback', action='store', dest='nflyback_frames', default=0, help="Number of flyback frames to remove from top of volumes")
parser.add_option('--motion', action='store_true', dest='correct_motion', default=False, help="do motion correction")
parser.add_option('-M', '--mcmethod', action='store', dest='mcmethod', default='Acquisition2P', help="Method of motion-correction to use [default: Acquisition2P]")
parser.add_option('-a', '--mcalgorithm', action='store', dest='mcalgorithm', default='@withinFile_withinFrame_lucasKanade', help="Algorithm to use for motion-correction [default: @withinFile_withinFrame_lucasKanade]")
parser.add_option('--bidi', action='store_true', dest='correct_bidir', default=False, help="do bidirectional scan correction")


(options, args) = parser.parse_args() 

# -------------------------------------------------------------
# INPUT PARAMS:
# -------------------------------------------------------------
rootdir = options.rootdir #'/nas/volume1/2photon/projects'
animalid = options.animalid
session = options.session #'20171003_JW016' #'20170927_CE059'
slurm = options.slurm

tiffsource = 'raw'

correct_flyback = options.correct_flyback
nflyback_frames = options.nflyback_frames
correct_bidir = options.correct_bidir
correct_motion = options.correct_motion
mc_method = options.mcmethod
mc_algorithm = options.mcalgorithm

# -------------------------------------------------------------
# Set paths:
# -------------------------------------------------------------
session_dir = os.path.join(rootdir, animalid, session)
acquisitions = [a for a in os.listdir(session_dir) if os.path.isdir(os.path.join(session_dir, a))]
session_dict = dict((acq, []) for acq in acquisitions)
for acq in session_dict.keys():
    session_dict[acq] = dict((r, []) for r in os.listdir(os.path.join(session_dir, acq))\
                                    if os.path.isdir(os.path.join(session_dir, acq, r))\
                                    and 'anat' not in r)
pp.pprint(session_dict)

base_opts = ['-R', rootdir, '-i', animalid, '-S', session, '-t', tiffsource, '--default']
if correct_bidir is True:
    base_opts.extend(['--bidi'])
if correct_motion is True:
    base_opts.extend(['--motion', '-M', mc_method, '-a', mc_algorithm])

for curr_acq in session_dict.keys():
    for curr_run in session_dict[acq].keys():
        base_opts.extend(['-A', curr_acq, '-r', curr_run])
        if 'volume' in curr_acq and correct_flyback is True:
            base_opts.extend(['--flyback', '-F', nflyback_frames])
        pid = create_pid(base_opts)
        session_dict[curr_acq][curr_run] = pid['pid_hash']
print "Created PIDs for session %s | acquisitions: runs --" % session
pp.pprint(session_dict)

for curr_acq in session_dict.keys():
    for curr_run in session_dict[curr_acq].keys():
        pid = session_dict[curr_acq][curr_run]
        opts = ['-R', rootdir, '-i', animalid, '-S', session, '-A', curr_acq, '-r', curr_run, '-p', pid]
        Process(target=process_pid, args=(opts,)).start()

