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
from stat import S_IREAD, S_IRGRP, S_IROTH, S_IWRITE, S_IWGRP, S_IWOTH

parser = optparse.OptionParser()

# PATH opts:
parser.add_option('-R', '--root', action='store', dest='rootdir', default='/nas/volume1/2photon/data', help='source dir (root project dir containing all expts) [default: /nas/volume1/2photon/data]')
parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')
parser.add_option('-S', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID') 
parser.add_option('-A', '--acq', action='store', dest='acquisition', default='', help="acquisition folder (ex: 'FOV1_zoom3x')")
parser.add_option('-r', '--run', action='store', dest='run', default='', help='name of run to process') 
parser.add_option('-p', '--pid', action='store', dest='pid_hash', default='', help="PID hash of current processing run (6 char), default will create new if set_pid_params.py not run")

parser.add_option('-H', '--hash', action='store', dest='source_hash', default='', help="hash of source dir (8 char). default uses output of get_scanimage_data()")

parser.add_option('--correct-flyback', action='store_true', dest='do_fyback_correction', default=False, help="Correct incorrect flyback frames (remove from top of stack). [default: false]")
parser.add_option('--flyback', action='store', dest='flyback', default=0, help="Num extra frames to remove from top of each volume to correct flyback [default: 0]")
parser.add_option('--notiffs', action='store_false', dest='save_tiffs', default=True, help="Set if not to write TIFFs after flyback-correction.")
parser.add_option('--rerun', action='store_false', dest='new_acquisition', default=True, help="set if re-running to get metadata for previously-processed acquisition")
parser.add_option('--slurm', action='store_true', dest='slurm', default=False, help="set if running as SLURM job on Odyssey")

tiffsource = 'raw'

(options, args) = parser.parse_args() 

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
source_hash = options.source_hash

execute_flyback = options.do_fyback_correction 
nflyback = int(options.flyback)

slurm = options.slurm

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
print "Getting SI meta data"
simeta_options = ['-R', rootdir, '-i', animalid, '-S', session, '-A', acquisition, '-r', run]
if new_acquisition is False:
    simeta_options.extend(['--rerun'])
if slurm is True:
    sireader_path = '/n/coxfs01/2p-pipeline/pkgs/ScanImageTiffReader-1.1-Linux'
    simeta_options.extend(['-P', sireader_path])

print simeta_options

import get_scanimage_data as sim
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

flyback_options = ['-R', rootdir, '-i', animalid, '-S', session, '-A', acquisition, '-r', run, '-H', raw_hashid,
                  '-z', nslices, '-c', nchannels, '-v', nvolumes]
if len(pid_hash) > 0:
    flyback_options.extend(['-p', pid_hash])

if execute_flyback is True:
    print "Correcting incorrect flyback frames in volumes."
    flyback_options.extend(['--correct-flyback', '--flyback=%i' % nflyback, '--discard=%i' % ndiscard])
    
if save_tiffs is False:
    flyback_options.extend(['--notiffs'])

import correct_flyback as fb
flyback_hash, pid_hash = fb.do_flyback_correction(flyback_options)
print "Flyback hash: %s" % flyback_hash
print "PID %s: Flyback finished." % pid_hash

# ===========================================================================
# 3.  Correct bidir scanning, if needed:
# ===========================================================================
import correct_bidirscan as bd
bidir_options = ['-R', rootdir, '-i', animalid, '-S', session, '-A', acquisition, '-r', run, '-p', pid_hash]
if slurm is True:
    bidir_options.extend(['--slurm'])
if execute_bidi is True:
    bidir_options.extend(['--bidi'])
print bidir_options

bidir_hash, pid_hash = bd.do_bidir_correction(bidir_options)
print "Bidir hash: %s" % bidir_hash
print "PID %s: BIDIR finished." % pid_hash

# ===========================================================================
# 4.  Correct motion, if needed:
# ===========================================================================
import correct_motion as mc
mc_options = ['-R', rootdir, '-i', animalid, '-S', session, '-A', acquisition, '-r', run, '-p', pid_hash]
if slurm is True:
    mc_options.extend(['--slurm'])
if execute_motion is True:
    mc_options.extend(['--motion'])
    
mcdir_hash, pid_hash = mc.do_motion(mc_options)
print "MC hash: %s" % mcdir_hash
print "PID %s: MC finished." % pid_hash


# ===========================================================================
# 4.  Clean up and update meta files:
# ===========================================================================
processed_dir = os.path.join(acquisition_dir, run, 'processed')
print "PI %s: DONE PROCESSING RAW." % pid_hash
pid_path = os.path.join(processed_dir, 'tmp_pids', 'tmp_pid_%s.json' % pid_hash)
with open(pid_path, 'r') as f:
    PID = json.load(f)
        
# UPDATE PID entry in dict:
with open(os.path.join(processed_dir, '%s.json' % pid_info_basename), 'r') as f:
    processdict = json.load(f)
processdict[PID['process_id']] = PID
print "Final PID: %s | tmp PID: %s." % (PID['tmp_hashid'], pid_hash)

# DELETE TMP PID FILE:
print "Removing tmp PID file: %s" % pid_path
os.remove(pid_path)