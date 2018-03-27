#!/usr/bin/env python2
'''
This script calls MATLAB function do_bidi_correction.m

inputs : 
    sourcedir :  fullpath to dir containing tiffs to be bidi corrected
    destdir : fullpath to writedir of bidi-corrected tiffs
    A :  reference struct containing meta info about current run
'''

import os
import sys
import json
import optparse
from stat import S_IREAD, S_IRGRP, S_IROTH
import matlab.engine
import copy
from checksumdir import dirhash
from pipeline.python.set_pid_params import create_pid, write_hash_readonly, append_hash_to_paths, post_pid_cleanup, update_pid_records
from pipeline.python.utils import sort_deinterleaved_tiffs, interleave_tiffs, deinterleave_tiffs, write_dict_to_json, zproj_tseries
from memory_profiler import profile

from os.path import expanduser
home = expanduser("~")

import matlab.engine

def extract_options(options):

    parser = optparse.OptionParser()

    # PATH opts:
    parser.add_option('-D', '--root', action='store', dest='rootdir', default='/nas/volume1/2photon/data', help='source dir (root project dir containing all expts) [default: /nas/volume1/2photon/data]')
    parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')
    parser.add_option('-S', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID') 
    parser.add_option('-A', '--acq', action='store', dest='acquisition', default='', help="acquisition folder (ex: 'FOV1_zoom3x')")
    parser.add_option('-R', '--run', action='store', dest='run', default='', help='name of run to process')
    parser.add_option('-p', '--pid', action='store', dest='pid_hash', default='', help="PID hash of current processing run (6 char), default will create new if set_pid_params.py not run")
    parser.add_option('-P', '--repo', action='store', dest='repo_path', default='~/Repositories/2p-pipeline', help='Path to 2p-pipeline repo. [default: ~/Repositories/2p-pipeline. If --slurm, default: /n/coxfs01/2p-pipeline/repos/2p-pipeline]')
    parser.add_option('-C', '--cvx', action='store', dest='cvx_path', default='~/MATLAB/cvx', help='Path to cvx install dir [default: ~/MATLAB/cvx. If --slurm, default: /n/coxfs01/2p-pipeline/pkgs/cvx]')
    parser.add_option('--slurm', action='store_true', dest='slurm', default=False, help='flag to use SLURM default opts')

    # MOTION params:
    parser.add_option('--motion', action='store_true', dest='do_mc', default=False, help='Set flag if should run motion-correction.')
    parser.add_option('-c', '--channel', action='store', dest='ref_channel', default=1, help='Index of CHANNEL to use for reference if doing motion correction [default: 1]')
    parser.add_option('-f', '--file', action='store', dest='ref_file', default=1, help='Index of FILE to use for reference if doing motion correction [default: 1]')
    parser.add_option('-M', '--method', action='store', dest='mc_method', default=None, help='Method for motion-correction. OPTS: Acquisition2P, NoRMCorre [default: Acquisition2P]')
    parser.add_option('-a', '--algo', action='store', dest='algorithm', default=None, help='Algorithm to use for motion-correction, e.g., @withinFile_withinFrame_lucasKanade if method=Acquisition2P, or nonrigid if method=NoRMCorre')
    parser.add_option('--default', action='store_true', dest='default', default='store_false', help="Use all DEFAULT params, for params not specified by user (no interactive)")

    parser.add_option('-s', '--tiffsource', action='store', dest='tiffsource', default=None, help="name of folder containing tiffs to be processed (ex: processed001). should be child of <run>/processed/")
    parser.add_option('-t', '--sourcetype', action='store', dest='sourcetype', default='raw', help="type of source tiffs (e.g., bidi, raw, mcorrected) [default: 'raw']")

    parser.add_option('-Z', '--zproj', action='store', dest='zproj_type', default='mean', help="Method of zprojection to create slice images [default: mean].")

    (options, args) = parser.parse_args(options) 

    if options.slurm is True:
        if 'coxfs01' not in options.rootdir:
            options.rootdir = '/n/coxfs01/2p-data'
        if 'coxfs01' not in options.repo_path:
            options.repo_path = '/n/coxfs01/2p-pipeline/repos/2p-pipeline' 
        if 'coxfs01' not in options.cvx_path:
	        options.cvx_path = '/n/coxfs01/2p-pipeline/pkgs/cvx'
    if '~' in options.rootdir:
        options.rootdir = options.rootdir.replace('~', home)
    if '~' in options.repo_path:
        options.repo_path = options.repo_path.replace('~', home)
    if '~' in options.cvx_path:
        options.cvx_path = options.cvx_path.replace('~', home)
    
    return options

#@profile
def do_motion(options):

    options = extract_options(options)

    rootdir = options.rootdir #'/nas/volume1/2photon/projects'
    animalid = options.animalid
    session = options.session #'20171003_JW016' #'20170927_CE059'
    acquisition = options.acquisition #'FOV1' #'FOV1_zoom3x'
    run = options.run
    pid_hash = options.pid_hash
    repo_path = options.repo_path
    cvx_path = options.cvx_path
    slurm = options.slurm
    do_mc = options.do_mc
    
    default = options.default
    tiffsource = options.tiffsource
    sourcetype = options.sourcetype

    zproj_type = options.zproj_type

    repo_path_matlab = os.path.join(repo_path, 'pipeline', 'matlab') 
    repo_prefix = os.path.split(repo_path)[0]

    # MOTION params:
    correct_motion = options.do_mc
    mc_method = options.mc_method
    mc_algorithm = options.algorithm
    ref_file = int(options.ref_file)
    ref_channel = int(options.ref_channel)
 
    # -------------------------------------------------------------
    # Set basename for files created containing meta/reference info:
    # -------------------------------------------------------------
    raw_simeta_basename = 'SI_%s' % run #functional_dir
    run_info_basename = '%s' % run #functional_dir
    pid_info_basename = 'pids_%s' % run

    # -------------------------------------------------------------
    # Set paths:
    # -------------------------------------------------------------
    acquisition_dir = os.path.join(rootdir, animalid, session, acquisition)
    assert os.path.exists(acquisition_dir), "Acquisition dir %s not found. Check inputs." % acquisition_dir

    tmp_pid_dir = os.path.join(acquisition_dir, run, 'processed', 'tmp_pids')
    if not os.path.exists(tmp_pid_dir):
        os.makedirs(tmp_pid_dir)
    
    tmp_pid_fns = [j for j in os.listdir(tmp_pid_dir) if pid_hash in j]
    if len(pid_hash) == 0 or (len(pid_hash) > 0 and len([j for j in os.listdir(tmp_pid_dir) if pid_hash in j]) == 0):
        # NO VALID PID, create default with input opts:
        print "Creating default PID with specified MCORRECTION input opts:"
        mc_opts = ['-D', rootdir, '-i', animalid, '-S', session, '-A', acquisition, '-R', run, '-t', tiffsource, '-s', sourcetype]
        if default is True:
            mc_opts.extend(['--default'])
        if do_mc is True:
            mc_opts.extend(['--motion', '-c', ref_channel, '-f', ref_file, '-M', mc_method, '-a', mc_algorithm])
        PID = create_pid(mc_opts)
        pid_hash = PID['pid_hash']
        print "New PID:", pid_hash
        tmp_pid_fn = 'tmp_pid_%s.json' % pid_hash 
    else:
        # -------------------------------------------------------------
        # Load PID:
        # -------------------------------------------------------------
        tmp_pid_fn = 'tmp_pid_%s.json' % pid_hash
        with open(os.path.join(tmp_pid_dir, tmp_pid_fn), 'r') as f:
            PID = json.load(f)
        do_mc = PID['PARAMS']['motion']['correct_motion']
 
    paramspath = os.path.join(tmp_pid_dir, tmp_pid_fn)
    runmeta_path = os.path.join(acquisition_dir, run, '%s.json' % run_info_basename)
    
    # -------------------------------------------------------------
    # Load run info:
    # -------------------------------------------------------------
    with open(runmeta_path, 'r') as f:
        runinfo = json.load(f)
    if len(runinfo['slices']) > 1 or runinfo['nchannels'] > 1:
        multiplanar = True
    else:
        multiplanar = False
        
#    # -------------------------------------------------------------
#    # Load PID:
#    # -------------------------------------------------------------
#    tmp_pid_fn = 'tmp_pid_%s.json' % pid_hash
#    paramspath = os.path.join(tmp_pid_dir, tmp_pid_fn)
#    with open(paramspath, 'r') as f:
#        print "Loading run PID file:", paramspath 
#        PID = json.load(f)
#        
    # -----------------------------------------------------------------------------
    # Update SOURCE/DEST paths for current PID, if needed:
    # -----------------------------------------------------------------------------
    # Make sure preprocessing sourcedir/destdir are correct:
    PID = append_hash_to_paths(PID, pid_hash, step='motion')
    
    interleave_write_tiffs = False
    if PID['PARAMS']['motion']['method'] == 'Acquisition2P' and multiplanar is True:
        # Default is to write deinterleaved slices to write_dir
        if 'deinterleaved' not in PID['PARAMS']['motion']['destdir']:
            PID['PARAMS']['motion']['destdir'] = PID['PARAMS']['motion']['destdir'] + '_deinterleaved'
        interleave_write_tiffs = True
       
    write_dict_to_json(PID, paramspath)  
    # And update process dict entry:
    update_pid_records(PID, acquisition_dir, run)
    
    source_dir = PID['PARAMS']['motion']['sourcedir']
    write_dir = PID['PARAMS']['motion']['destdir']
    
    print "======================================================="
    print "PID: %s -- MOTION" % pid_hash
    #pp.pprint(PID)
    print "SOURCE:", source_dir
    print "DEST:", write_dir
    print "======================================================="

    # -------------------------------------------------------------
    # Do correction:
    # -------------------------------------------------------------
    if do_mc is True:
        if not os.path.exists(write_dir):
            os.makedirs(write_dir)
        print "================================================="
        print "Doing MOTION correction."
        print "================================================="
        completed_mc = False
        try:
            eng = matlab.engine.start_matlab()
            eng.cd(repo_path_matlab, nargout=0)
            eng.add_repo_paths(cvx_path, repo_prefix, nargout=0)
            eng.do_motion_correction(paramspath, nargout=0)
            completed_mc = True
            eng.quit()
        except SystemError as e:
            print "ERROR terminating... -----------------------"
            print "MC step did not exit cleanly."
            if completed_mc is True:
                print "MC completed, but unable to quit engine."

    # -------------------------------------------------------------
    # Check for Interleaving/Deinterleaving:
    # -------------------------------------------------------------
    if do_mc is True:
        if interleave_write_tiffs is True:
            slice_dir = copy.copy(write_dir)
            volume_dir = slice_dir.split('_deinterleaved')[0]
            interleave_tiffs(slice_dir, volume_dir, runmeta_path)
        else:
            volume_dir = copy.copy(write_dir)
            slice_dir = volume_dir + '_deinterleaved'
            
        if multiplanar is True:
            print "Multiple slices/channels found. Sorting deinterleaved tiffs."
            sort_deinterleaved_tiffs(slice_dir, runmeta_path)
    
    # ========================================================================================
    # UPDATE PREPROCESSING SOURCE/DEST DIRS, if needed:
    # ========================================================================================
    write_hash = None
    if do_mc is True:
        write_hash, PID = write_hash_readonly(volume_dir, PID=PID, step='motion', label='mc')

    print paramspath    
    write_dict_to_json(PID, paramspath) 
    # And update process dict entry:
    update_pid_records(PID, acquisition_dir, run)
 
#    with open(paramspath, 'w') as f:
#        print paramspath
#        json.dump(PID, f, indent=4, sort_keys=True)
    # ========================================================================================

    return write_hash, pid_hash #PID #pid_hash


def main(options):
   
    # Do motion-correction: 
    mc_hash, pid_hash = do_motion(options) 
    #pid_hash = PID['pid_hash']
    print "PID %s: Finished motion-correction step: output dir hash %s" % (pid_hash, mc_hash)
  
    options = extract_options(options)
    acquisition_dir = os.path.join(options.rootdir, options.animalid, options.session, options.acquisition)
    run = options.run

    # Create average slices for viewing:
    with open(os.path.join(acquisition_dir, run, 'processed', 'pids_%s.json' % run), 'r') as f:
        currpid = json.load(f)
    curr_process_id = [p for p in currpid.keys() if currpid[p]['pid_hash'] == pid_hash][0]
    source_dir = currpid[curr_process_id]['PARAMS']['motion']['destdir']
    runmeta_fn = os.path.join(acquisition_dir, run, '%s.json' % run)
    if os.path.isdir(source_dir):
        zproj_tseries(source_dir, runmeta_fn, zproj_type=options.zproj_type)
    print "Finished creating ZPROJ slice images from motion-corrected tiffs."
 
    # Clean up tmp files and udpate meta info: 
    post_pid_cleanup(acquisition_dir, run, pid_hash)
    print "PID %s -- Finished cleaning up tmp files, updated dicts." % pid_hash


if __name__ == '__main__':
    main(sys.argv[1:]) 

    
