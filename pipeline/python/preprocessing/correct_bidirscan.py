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
import copy
from checksumdir import dirhash

from os.path import expanduser
home = expanduser("~")

def do_bidir_correction(options):

    parser = optparse.OptionParser()

    # PATH opts:
    parser.add_option('-R', '--root', action='store', dest='rootdir', default='/nas/volume1/2photon/data', help='source dir (root project dir containing all expts) [default: /nas/volume1/2photon/data]')
    parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')
    parser.add_option('-S', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID') 
    parser.add_option('-A', '--acq', action='store', dest='acquisition', default='', help="acquisition folder (ex: 'FOV1_zoom3x')")
    parser.add_option('-r', '--run', action='store', dest='run', default='', help='name of run to process') 
    #parser.add_option('-H', '--hash', action='store', dest='process_hash', default='', help="hash of source dir (6 char)")
    parser.add_option('-p', '--pid', action='store', dest='pid_hash', default='', help="PID hash of current processing run (6 char), default will create new if set_pid_params.py not run")

    parser.add_option('-P', '--repo', action='store', dest='repo_path', default='~/Repositories/2p-pipeline', help='Path to 2p-pipeline repo. [default: ~/Repositories/2p-pipeline. If --slurm, default: /n/coxfs01/2p-pipeline/repos/2p-pipeline]')
    parser.add_option('--slurm', action='store_true', dest='slurm', default=False, help='flag to use SLURM default opts')
    parser.add_option('--bidi', action='store_true', dest='do_bidi', default=False, help='flag to actually do bidi-correction')
    
    (options, args) = parser.parse_args() 

    rootdir = options.rootdir #'/nas/volume1/2photon/projects'
    animalid = options.animalid
    session = options.session #'20171003_JW016' #'20170927_CE059'
    acquisition = options.acquisition #'FOV1' #'FOV1_zoom3x'
    run = options.run
    pid_hash = options.pid_hash

    repo_path = options.repo_path
    slurm = options.slurm
    do_bidi = options.do_bidi
    
    if slurm is True and 'coxfs01' not in repo_path:
        repo_path = '/n/coxfs01/2p-pipeline/repos/2p-pipeline'
    if '~' in repo_path:
        repo_path = repo_path.replace('~', home)
    repo_path_matlab = os.path.join(repo_path, 'pipeline', 'matlab')


    acquisition_dir = os.path.join(rootdir, animalid, session, acquisition)
    
    # -------------------------------------------------------------
    # Set basename for files created containing meta/reference info:
    # -------------------------------------------------------------
    raw_simeta_basename = 'SI_%s' % run #functional_dir
    run_info_basename = '%s' % run #functional_dir
    pid_info_basename = 'pids_%s' % run
    # -------------------------------------------------------------
    
    tmp_pid_dir = os.path.join(acquisition_dir, run, 'processed', 'tmp_pids')
    paramspath = os.path.join(tmp_pid_dir, 'tmp_pid_%s.json' % pid_hash)
    runmeta_path = os.path.join(acquisition_dir, run, '%s.json' % run_info_basename)

    # -----------------------------------------------------------------------------
    # Update SOURCE/DEST paths for current PID, if needed:
    # -----------------------------------------------------------------------------
    with open(paramspath, 'r') as f:
        PID = json.load(f)
    if pid_hash not in PID['DST']:
        PID['DST'] = os.path.join(acquisition_dir, run, 'processed', '%s_%s' % (PID['process_id'], pid_hash))
    
    # If done, update preprocessing SOURCEDIR to output of flyback-correction:
    correct_flyback = PID['PARAMS']['preprocessing']['correct_flyback']
    if correct_flyback is True and 'raw' in PID['PARAMS']['preprocessing']['destdir']:
        PID['PARAMS']['preprocessing']['sourcedir'] = copy.copy(PID['PARAMS']['preprocessing']['destdir'])
    # IF doing bidi, update preprocessing DESTDIR to current PID path, plus 'bidi':
    if do_bidi is True:
        PID['PARAMS']['preprocessing']['destdir'] = os.path.join(PID['DST'], 'bidi')
    with open(paramspath, 'w') as f:
        json.dump(PID, f, indent=4, sort_keys=True)
    # -----------------------------------------------------------------------------

    
    if do_bidi is True:
        # 2. Run BIDIR-CORRECTION (Matlab):              
        eng = matlab.engine.start_matlab('-nojvm')
        eng.cd(repo_path_matlab, nargout=0)
        eng.add_repo_paths(nargout=0)
        eng.do_bidi_correction(paramspath, runmeta_path, nargout=0)
        eng.quit()
    
    # ========================================================================================
    # UPDATE PREPROCESSING SOURCE/DEST DIRS, if needed:
    # ========================================================================================
    write_hash = None
    if do_bidi is True:
        write_dir = PID['PARAMS']['preprocessing']['destdir']
        excluded_files = [str(f) for f in os.listdir(write_dir) if not f.endswith('tif')]
        write_hash = dirhash(write_dir, 'sha1', excluded_files=excluded_files)[0:6]
        if write_hash not in write_dir:
            newdir_name = write_dir + '_%s' % write_hash
            if not os.path.exists(newdir_name):
                os.rename(write_dir, newdir_name)
            PID['PARAMS']['preprocessing']['destdir'] = newdir_name
            print "Renamed bidi:", newdir_name

            # Make sure newly created TIFFs are READ-ONLY:
            for f in os.listdir(newdir_name):
                os.chmod(os.path.join(newdir_name, f), S_IREAD|S_IRGRP|S_IROTH)  

            if os.path.exists(write_dir + '_slices'):
                newslicedir_name = write_dir + '_%s_slices' % write_hash
                if not os.path.exists(newslicedir_name):
                    os.rename(write_dir + '_slices', newslicedir_name)
                print "Renamed bidi-slices:", newslicedir_name

                # Make sure newly created TIFFs are READ-ONLY:
                for f in os.listdir(newdir_name):
                    os.chmod(os.path.join(newdir_name, f), S_IREAD|S_IRGRP|S_IROTH)  

    with open(paramspath, 'w') as f:
        print paramspath
        json.dump(PID, f, indent=4, sort_keys=True)
    # ========================================================================================
    
    # if write_hash is None:
    #     write_hash = source_hash
        
    return write_hash, pid_hash


def main(options):
    
    bidir_hash, pid_hash = do_bidir_correction(options)
    
    print "PID %s: Finished bidir-correction step: output dir hash %s" % (pid_hash, bidir_hash)
    
if __name__ == '__main__':
    main(sys.argv[1:]) 

    
    