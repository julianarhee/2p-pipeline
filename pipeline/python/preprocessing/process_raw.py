#!/usr/bin/env python2
'''
STEP 1 of analysis pipeline:  process raw ScanImage TIFFs.

1.  get_scanimage_data.py

Saves SI struct and TIFF image descriptions in .json to: <acquisition_dir>/<run>/raw/
RAW tiff info is read-only.
    
2.  correct_flyback.py

If FastZ params are badly set during acquisition, this creates new TIFFs that corrects by removing bad frames.
Meta info is also updated for relevant params.

If --correct-flyback is flagged, flyback-corrected TIFFs are saved to: <acquisition_dir>/<run>/processed/<process_id>/raw/
'''

import os
import json
import optparse
import sys
from stat import S_IREAD, S_IRGRP, S_IROTH, S_IWRITE, S_IWGRP, S_IWOTH
from pipeline.python.set_pid_params import update_pid_records, post_pid_cleanup
from pipeline.python.utils import zproj_tseries
import get_scanimage_data as sim
import correct_flyback as fb
import correct_motion as mc

import time
from functools import wraps
 
def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print ("Total time running %s: %s seconds" %
               (function.func_name, str(t1-t0))
               )
        return result
    return function_timer

@fn_timer
def process_pid(options):

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
    print "Flyback:", execute_flyback
    print "Bidir:", execute_bidi
    print "Motion:", execute_motion

    # ===========================================================================
    # 1.  Get SI meta data from raw tiffs:
    # ===========================================================================
    if slurm is True:
        if 'coxfs01' not in rootdir:
            rootdir = '/n/coxfs01/2p-data'
        sireader_path = '/n/coxfs01/2p-pipeline/pkgs/ScanImageTiffReader-1.1-Linux'
        print sireader_path

    simeta_options = ['-R', rootdir, '-i', animalid, '-S', session, '-A', acquisition, '-r', run]

    print "Getting SI meta data"
    if new_acquisition is False:
        simeta_options.extend(['--rerun'])
    if slurm is True:
        simeta_options.extend(['-T', sireader_path, '--slurm']) 

    print simeta_options

    raw_hashid = sim.get_meta(simeta_options)
    print "Finished getting SI metadata!"
    print "Raw hash: %s" % raw_hashid


    # ===========================================================================
    # 2.  Correct flyback, if needed:
    # ===========================================================================
    # Only need to do this if creating substacks due to incorret flyback frames in volumes.

    # NOTE:  This assumes all tiffs in current run have the same acquisition params.
    rawdir = 'raw_%s' % raw_hashid
    simeta_fn = "%s.json" % raw_simeta_basename
    with open(os.path.join(acquisition_dir, run, rawdir,  simeta_fn), 'r') as fr:
        simeta = json.load(fr)

    ndiscard = int(simeta['File001']['SI']['hFastZ']['numDiscardFlybackFrames'])
    nvolumes = int(simeta['File001']['SI']['hFastZ']['numVolumes'])
    nslices = int(simeta['File001']['SI']['hStackManager']['numSlices'])
    tmp_channels = simeta['File001']['SI']['hChannels']['channelSave']
    if isinstance(tmp_channels, int):
        nchannels = simeta['File001']['SI']['hChannels']['channelSave']
    else:
        nchannels = len(simeta['File001']['SI']['hChannels']['channelSave']) 
    print "Raw SI info:"
    print "N channels: {nchannels}, N slices: {nslices}, N volumes: {nvolumes}".format(nchannels=nchannels, nslices=nslices, nvolumes=nvolumes)
    print "Num discarded frames for flyback:", ndiscard

    flyback_options = ['-R', rootdir, '-i', animalid, '-S', session, '-A', acquisition, '-r', run,
                      '-z', nslices, '-c', nchannels, '-v', nvolumes]
    if len(pid_hash) > 0:
        flyback_options.extend(['-p', pid_hash])

    if execute_flyback is True:
        print "Correcting incorrect flyback frames in volumes."
        flyback_options.extend(['--flyback', '--nflyback=%i' % nflyback, '--discard=%i' % ndiscard])

    if save_tiffs is False:
        flyback_options.extend(['--notiffs'])

    if slurm is True:
        flyback_options.extend(['--slurm'])

    flyback_hash, pid_hash = fb.do_flyback_correction(flyback_options)
    #pid_hash = PID['pid_hash']
    print "Flyback hash: %s" % flyback_hash
    print "PID %s: Flyback finished." % pid_hash

    # ===========================================================================
    # 3.  Correct bidir scanning, if needed:
    # ===========================================================================
    import correct_bidirscan as bd
    bidir_options = ['-R', rootdir, '-i', animalid, '-S', session, '-A', acquisition, '-r', run, '-p', pid_hash]
    if default is True:
        bidir_otions.extend(['--default'])
    if slurm is True:
        bidir_options.extend(['--slurm'])
    if execute_bidi is True:
        bidir_options.extend(['--bidi'])
    if len(repo_path) > 0:
        bidir_options.extend(['-P', repo_path])
    print bidir_options

    bidir_hash, pid_hash = bd.do_bidir_correction(bidir_options)
    #pid_hash = PID['pid_hash']
    print "Bidir hash: %s" % bidir_hash
    print "PID %s: BIDIR finished." % pid_hash

    # Create average slices for viewing:
    if get_zproj is True and execute_bidi is True:
        print "PID %s -- Done with MC. Getting z-projection (%s) slice images." % (pid_hash, zproj_type)

        with open(os.path.join(acquisition_dir, run, 'processed', 'pids_%s.json' % run), 'r') as f:
            currpid = json.load(f)
        curr_process_id = [p for p in currpid.keys() if currpid[p]['pid_hash'] == pid_hash][0]
        source_dir = currpid[curr_process_id]['PARAMS']['preprocessing']['destdir']
        runmeta_fn = os.path.join(acquisition_dir, run, '%s.json' % run)
        #if os.path.isdir(source_dir):
        zproj_tseries(source_dir, runmeta_fn, zproj_type=zproj_type)
        print "PID %s -- Finished creating ZPROJ slice images from bidi-corrected tiffs." % pid_hash
     


    # ===========================================================================
    # 4.  Correct motion, if needed:
    # ===========================================================================
    mc_options = ['-R', rootdir, '-i', animalid, '-S', session, '-A', acquisition, '-r', run, '-p', pid_hash]
    if default is True:
        mc_options.extend(['--default'])
    if slurm is True:
        mc_options.extend(['--slurm'])
    if execute_motion is True:
        mc_options.extend(['--motion'])
    if len(repo_path) > 0:
        mc_options.extend(['-P', repo_path])

    mcdir_hash, pid_hash = mc.do_motion(mc_options)
    #pid_hash = PID['pid_hash']
    print "MC hash: %s" % mcdir_hash
    print "PID %s: MC finished." % pid_hash
 
    # Create average slices for viewing:
    if get_zproj is True and execute_motion is True:
        print "PID %s -- Done with MC. Getting z-projection (%s) slice images." % (pid_hash, zproj_type)
        with open(os.path.join(acquisition_dir, run, 'processed', 'pids_%s.json' % run), 'r') as f:
            currpid = json.load(f)
        curr_process_id = [p for p in currpid.keys() if currpid[p]['pid_hash'] == pid_hash][0]
        source_dir = currpid[curr_process_id]['PARAMS']['motion']['destdir']
        runmeta_fn = os.path.join(acquisition_dir, run, '%s.json' % run)
        print "SOURCE dir is:", source_dir
        #if os.path.isdir(source_dir):
        zproj_tseries(source_dir, runmeta_fn, zproj_type=zproj_type)
        print "PID %s -- Finished creating ZPROJ slice images from motion-corrected tiffs." % pid_hash

 
    # ===========================================================================
    # 4.  Clean up and update meta files:
    # ===========================================================================
    post_pid_cleanup(acquisition_dir, run, pid_hash)     
    print "FINISHED PROCESSING PID %s" % pid_hash
   
    return pid_hash
    
def main(options):
    
    pid_hash = process_pid(options)
    
    print "**************************************************"
    print "PID %s: COMPLETED." % pid_hash
    print "**************************************************"

if __name__ == '__main__':
    main(sys.argv[1:]) 

    
