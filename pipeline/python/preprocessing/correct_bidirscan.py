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
import matlab.engine

parser = optparse.OptionParser()

# PATH opts:
parser.add_option('-R', '--root', action='store', dest='rootdir', default='/nas/volume1/2photon/data', help='source dir (root project dir containing all expts) [default: /nas/volume1/2photon/data]')
parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')
parser.add_option('-S', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID') 
parser.add_option('-A', '--acq', action='store', dest='acquisition', default='', help="acquisition folder (ex: 'FOV1_zoom3x')")
parser.add_option('-r', '--run', action='store', dest='run', default='', help='name of run to process') 

(options, args) = parser.parse_args() 

rootdir = options.rootdir #'/nas/volume1/2photon/projects'
animalid = options.animalid
session = options.session #'20171003_JW016' #'20170927_CE059'
acquisition = options.acquisition #'FOV1' #'FOV1_zoom3x'
run = options.run

acquisition_dir = os.path.join(rootdir, animalid, session, acquisition)
paramspath = os.path.join(acquisition_dir, run, 'processed', 'tmp_processparams.json')
refpath = os.path.join(acquisition_dir, run, 'reference_%s.json' % run)

eng = matlab.engine.start_matlab('-nojvm')
eng.cd(repo_path_matlab, nargout=0)
eng.add_repo_paths(nargout=0)
eng.do_bidi_correction(paramspath, refpath, nargout=0)
eng.quit()

