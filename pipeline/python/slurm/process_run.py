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

def fake_process_run(options):
    parser = optparse.OptionParser()

    # PATH opts:
    parser.add_option('-D', '--root', action='store', dest='rootdir', default='/nas/volume1/2photon/data', help='source dir (root project dir containing all expts) [default: /nas/volume1/2photon/data]')
    parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')
    parser.add_option('-S', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID') 
    parser.add_option('-A', '--acq', action='store', dest='acquisition', default='', help="acquisition folder (ex: 'FOV1_zoom3x')")
    parser.add_option('-R', '--run', action='store', dest='run', default='', help='name of run to process') 
    parser.add_option('-P', '--repo', action='store', dest='repo_path', default='', help='Path to 2p-pipeline repo. [default: ~/Repositories/2p-pipeline. If --slurm, default: /n/coxfs01/2p-pipeline/repos/2p-pipeline]')

    parser.add_option('-p', '--pid', action='store', dest='pid_hash', default='', help="PID hash of current processing run (6 char), default will create new if set_pid_params.py not run")

    #parser.add_option('-H', '--hash', action='store', dest='source_hash', default='', help="hash of source dir (8 char). default uses output of get_scanimage_data()")

    parser.add_option('--flyback', action='store_true', dest='do_fyback_correction', default=False, help="Correct incorrect flyback frames (remove from top of stack). [default: false]")
    parser.add_option('-F', '--nflyback', action='store', dest='flyback', default=0, help="Num extra frames to remove from top of each volume to correct flyback [default: 0]")
    parser.add_option('--notiffs', action='store_false', dest='save_tiffs', default=True, help="Set if not to write TIFFs after flyback-correction.")
    parser.add_option('--rerun', action='store_false', dest='new_acquisition', default=True, help="set if re-running to get metadata for previously-processed acquisition")
    parser.add_option('--slurm', action='store_true', dest='slurm', default=False, help="set if running as SLURM job on Odyssey")
    parser.add_option('--default', action='store_true', dest='default', default='store_false', help="Use all DEFAULT params, for params not specified by user (no interactive)")
    parser.add_option('--zproject', action='store_true', dest='get_zproj', default='store_false', help="Set flag to create z-projection slices for processed tiffs.")
    parser.add_option('-Z', '--zproj', action='store', dest='zproj_type', default='mean', help="Method of zprojection to create slice images [default: mean].")

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
    #source_hash = options.source_hash

    execute_flyback = options.do_fyback_correction 
    nflyback = int(options.flyback)

    slurm = options.slurm
    default = options.default
    
    get_zproj = options.get_zproj
    zproj_type = options.zproj_type

    # -------------------------------------------------------------
    # Set basename for files created containing meta/reference info:
    # -------------------------------------------------------------
    raw_simeta_basename = 'SI_%s' % run #functional_dir
    run_info_basename = '%s' % run #functional_dir
    pid_info_basename = 'pids_%s' % run
    # -------------------------------------------------------------

    acquisition_dir = os.path.join(rootdir, animalid, session, acquisition)

    # ===========================================================================
    # If PID specified, that takes priority:
    # ===========================================================================
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
    print "PID %s -- Motion:" % (pid_hash, str(execute_motion))

    return pid_hash
    
def process_run(pid_filepath):
    
    with open(pid_filepath, 'r') as f:
        pidinfo = json.load(f)
    
    # Load current PID info from tmp file:
#    tmp_pid_fpath = os.path.join(pidinfo['rootdir'], pidinfo['animalid'],
#                                 pidinfo['session'], pidinfo['run'],
#                                 'processed', 'tmp_pids', 'tmp_pid_%s.json' % pidinfo['pid'])
#    with open(tm_pid_fpath, 'r') as f:
#        pid = json.load(f)
    
    popts = ['-R', pinfo['rootdir'], '-i', pinfo['animalid'], '-S', pinfo['session'], '-A', pinfo['acquisition'], '-r', pinfo['run'], '-p', pinfo['pid']]
    
    pidhash = fake_process_run(popts)
        
    return pidhash

def main():
    
    pid_filepath = sys.argv[1]
    pidhash = process_run(pid_filepath)
    
    print "FINISHED PROCESSING PID %s." % pidhash
    