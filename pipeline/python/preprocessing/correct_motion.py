#!/usr/bin/env python2
'''
This script calls MATLAB function do_bidi_correction.m

inputs : 
    sourcedir :  fullpath to dir containing tiffs to be bidi corrected
    destdir : fullpath to writedir of bidi-corrected tiffs
    A :  reference struct containing meta info about current run
'''

import os
import json
import optparse
from stat import S_IREAD, S_IRGRP, S_IROTH
from os.path import expanduser
home = expanduser("~")

import matlab.engine

parser = optparse.OptionParser()

# PATH opts:
parser.add_option('-R', '--root', action='store', dest='rootdir', default='/nas/volume1/2photon/data', help='source dir (root project dir containing all expts) [default: /nas/volume1/2photon/data]')
parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')
parser.add_option('-S', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID') 
parser.add_option('-A', '--acq', action='store', dest='acquisition', default='', help="acquisition folder (ex: 'FOV1_zoom3x')")
parser.add_option('-r', '--run', action='store', dest='run', default='', help='name of run to process') 
parser.add_option('-P', '--repo', action='store', dest='repo_path', default='~/Repositories/2p-pipeline', help='Path to 2p-pipeline repo. [default: ~/Repositories/2p-pipeline. If --slurm, default: /n/coxfs01/2p-pipeline/repos/2p-pipeline]')
parser.add_option('--slurm', action='store_true', dest='slurm', default=False, help='flag to use SLURM default opts')

(options, args) = parser.parse_args() 

rootdir = options.rootdir #'/nas/volume1/2photon/projects'
animalid = options.animalid
session = options.session #'20171003_JW016' #'20170927_CE059'
acquisition = options.acquisition #'FOV1' #'FOV1_zoom3x'
run = options.run

repo_path = options.repo_path
slurm = options.slurm

if slurm is True and 'coxfs01' not in repo_path:
    repo_path = '/n/coxfs01/2p-pipeline/repos/2p-pipeline'
if '~' in repo_path:
    repo_path = repo_path.replace('~', hom)
repo_path_matlab = os.path.join(repo_path, 'pipeline', 'matlab')

acquisition_dir = os.path.join(rootdir, animalid, session, acquisition)
paramspath = os.path.join(acquisition_dir, run, 'processed', 'tmp_processparams.json')
refpath = os.path.join(acquisition_dir, run, 'reference_%s.json' % run)

eng = matlab.engine.start_matlab()
eng.cd(repo_path_matlab, nargout=0)
eng.add_repo_paths(nargout=0)
eng.do_motion_correction(paramspath, nargout=0)
eng.quit()

print "FINISHED MOTION CORRECTION STEP."
