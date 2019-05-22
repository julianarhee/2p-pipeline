#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 14:52:33 2019

@author: julianarhee
"""

import os
import glob
import optparse
import re
import shutil
import sys

def extract_options(options):
    
    parser = optparse.OptionParser()

    parser.add_option('-D', '--root', action='store', dest='rootdir', default='/n/coxfs01/2p-data',\
                      help='data root dir (root project dir containing all animalids) [default: /n/coxfs01/2pdata]')
    parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')

    # Set specific session/run for current animal:
    parser.add_option('-S', '--session', action='store', dest='session', default='', \
                      help='session dir (format: YYYMMDD_ANIMALID')

    (options, args) = parser.parse_args(options)

    return options

options = ['-i', 'JC089', '-S', '20190520']


def main(options):
        
    optsE = extract_options(options)
    
    session_dir = os.path.join(optsE.rootdir, optsE.animalid, optsE.session)
    fovs = glob.glob(os.path.join(session_dir, 'FOV*'))
    print("[%s - %s] Found %i FOVs." % (optsE.animalid, optsE.session, len(fovs)))
    for fov in fovs:
        print("... %s" % fov)
    
    raw_paradigm_dir = os.path.join(session_dir, 'all_paradigm_files')
    assert os.path.exists(raw_paradigm_dir), "No raw paradigm files found! Did you transfer the files?"
    
    for fov in fovs:
        fov_num = int(re.search("FOV\d{1}", fov).group()[-1])
        all_runs = [os.path.split(runpath)[-1] for runpath in glob.glob(os.path.join(fov, '*run*'))]
        experiment_types = list(set([r.split('_')[0] for r in all_runs]))
        print("Found %i experiment types:" % len(experiment_types))
        for exp in experiment_types:
            print "- organizing runs: %s" % exp
            curr_runs = glob.glob(os.path.join(fov, '%s_run*' % exp))
            for run_dir in curr_runs:
                run_num = int(re.search("_run\d{1}", run_dir).group()[-1])
                dest_paradigm_dir = os.path.join(run_dir, 'raw', 'paradigm_files')
                dest_eyetracker_dir = os.path.join(run_dir, 'raw', 'eyetracker_files')
                if not os.path.exists(dest_paradigm_dir):
                    os.makedirs(dest_paradigm_dir)
                    os.makedirs(dest_eyetracker_dir)
                curr_para_files = glob.glob(os.path.join(raw_paradigm_dir, '*fov%i_%s_f%i*' % (fov_num, exp, run_num)))
                
                if exp == 'retino' and len(curr_para_files) == 0:
                    curr_para_files = glob.glob(os.path.join(raw_paradigm_dir, '*fov%i_retinobar_f%i*' % (fov_num, run_num)))
                
                assert len(curr_para_files) > 0, "No paradigm files found for exp: %s, run_dir: %s" % (exp, run_dir)
                for cpath in curr_para_files:
                    fname = os.path.split(cpath)[-1]
                    shutil.move(cpath, os.path.join(dest_paradigm_dir, fname))
    
    print("Moved paradigm files. There are %i files unaccounted for." % len(os.listdir(raw_paradigm_dir)))
    
if __name__ == '__main__':
    main(sys.argv[1:])

   
            