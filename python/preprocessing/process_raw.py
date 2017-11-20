#!/usr/bin/env python2
'''
STEP 1 of analysis pipeline:  process raw ScanImage TIFFs.

1.  get_scanimage_data.py

Saves SI struct (in TIFF metdata) to json for easy access. 
Saves hdf5 containing SI parameters relevant for rest of pipeline.
    
2.  correct_flyback.py

If FastZ params are badly set during acquisition, this creates new TIFFs that corrects by removing bad frames.
Saves a new TIFF for each raw tiff in ./acquistion_dir/DATA/ 

If used, mcparams.correct_flyback = True.

'''

# TODO:  do SI metadata adjustment here?? (instead of parseSIdata.m, etc. in MATLAB)
# TODO:  if NOT creating substacks, i.e., the raw acquisition files are ready to go for motion-correction, etc., substitute flyback-correction step with proper saving of structs and file-paths to match standard.

import os
import json
import optparse

parser = optparse.OptionParser()

# PATH opts:
parser.add_option('-S', '--source', action='store', dest='source', default='/nas/volume1/2photon/data', help='source dir (root project dir containing all expts) [default: /nas/volume1/2photon/data]')
parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')
parser.add_option('-s', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID') 
parser.add_option('-A', '--acq', action='store', dest='acquisition', default='', help="acquisition folder (ex: 'FOV1_zoom3x')")
parser.add_option('-r', '--run', action='store', dest='run', default='', help='name of run to process') 

#parser.add_option('-E', '--experiment', action='store', dest='experiment', default='', help='experiment type (parent of session dir)') 
#parser.add_option('-f', '--functional', action='store', dest='functional_dir', default='functional', help="folder containing functional TIFFs. [default: 'functional']")

parser.add_option('--correct-flyback', action='store_true', dest='do_fyback_correction', default=False, help="Correct incorrect flyback frames (remove from top of stack). [default: false]")
parser.add_option('--flyback', action='store', dest='flyback', default=0, help="Num extra frames to remove from top of each volume to correct flyback [default: 0]")
parser.add_option('--notiffs', action='store_false', dest='save_tiffs', default=True, help="Set if not to write TIFFs after flyback-correction.")
parser.add_option('--rerun', action='store_false', dest='new_acquisition', default=True, help="set if re-running to get metadata for previously-processed acquisition")

tiffsource = 'raw'

(options, args) = parser.parse_args() 

new_acquisition = options.new_acquisition
save_tiffs = options.save_tiffs

source = options.source #'/nas/volume1/2photon/projects'
animalid = options.animalid
session = options.session #'20171003_JW016' #'20170927_CE059'
acquisition = options.acquisition #'FOV1' #'FOV1_zoom3x'
run = options.run

acquisition_dir = os.path.join(source, animalid, session, acquisition)

#acquisition_dir = os.path.join(source, experiment, session, acquisition)


# -------------------------------------------------------------
# Set basename for files created containing meta/reference info:
# -------------------------------------------------------------
raw_simeta_basename = 'SI_%s' % run #functional_dir
reference_info_basename = 'reference_%s' % run #functional_dir
# -------------------------------------------------------------
# -------------------------------------------------------------

do_flyback_correction = options.do_fyback_correction #True #True #True
flyback = int(options.flyback) #0 #1       # Num flyback frames at top of stack [default: 8]


# ----------------------------------------------------------------------------
# 1.  Get SI meta data from raw tiffs:
# ----------------------------------------------------------------------------
print "Getting SI meta data"

if new_acquisition is True:
    simeta_options = ['-S', source, '-i', animalid, '-s', session, '-A', acquisition, '-r', run]
else:
    simeta_options = ['-S', source, '-i', animalid, '-s', session, '-A', acquisition, '-r', run, '--rerun']

import get_scanimage_data
get_scanimage_data.main(simeta_options)


# ----------------------------------------------------------------------------
# 2.  Optional:  Correct flyback, if needed:
# ----------------------------------------------------------------------------
# Only need to do this if creating substacks due to incorret flyback frames in volumes:

# flyback = 2      # Num flyback frames at top of stack [default: 8]

# discard = 1      # Num discard frames at end of stack [default: 8]
# nchannels = 2    # Num interleaved channels in raw tiffs to be processed [default: 2]
# nvolumes = 1080  # Num volumes acquired [default: 340]
# nslices = 15     # Num slices specified in FastZ control, not including discard [default: 30]

# Load raw SI meta info for relevant params (so don't need to specify):
simeta_fn = "%s.json" % raw_simeta_basename
with open(os.path.join(acquisition_dir, run, 'raw',  simeta_fn), 'r') as fr:
    simeta = json.load(fr)
 
discard = int(simeta['File001']['SI']['hFastZ']['numDiscardFlybackFrames'])
nvolumes = int(simeta['File001']['SI']['hFastZ']['numVolumes'])
# nslices = int(simeta['File001']['SI']['hFastZ']['numFramesPerVolume'])
nslices = int(simeta['File001']['SI']['hStackManager']['numSlices'])
tmp_channels = simeta['File001']['SI']['hChannels']['channelSave']
if isinstance(tmp_channels, int):
    nchannels = simeta['File001']['SI']['hChannels']['channelSave']
else:
    nchannels = len(simeta['File001']['SI']['hChannels']['channelSave']) 
#nchannels = #len([int(i) for i in simeta['File001']['SI']['hChannels']['channelSave']]) # if i.isnumeric()])
print "Raw SI info:"
print "N channels: {nchannels}, N slices: {nslices}, N volumes: {nvolumes}".format(nchannels=nchannels, nslices=nslices, nvolumes=nvolumes)
print "Num discarded frames for flyback:", discard

if do_flyback_correction:
    print "Correcting incorrect flyback frames in volumes."
    if save_tiffs is False:
        flyback_options = ['-S', source, '-i', animalid, '-s', session, '-A', acquisition, \
                       '-r', run, '--flyback=%i' % flyback, '--discard=%i' % discard, \
                       '-z', nslices, '-c', nchannels, '-v', nvolumes, \
                       '--native', '--correct-flyback', '--notiffs']
    else:
        flyback_options = ['-S', source, '-i', animalid, '-s', session, '-A', acquisition, \
                       '-r', run, '--flyback=%i' % flyback, '--discard=%i' % discard, \
                       '-z', nslices, '-c', nchannels, '-v', nvolumes, \
                       '--native', '--correct-flyback']
else:
    print "Not doing flyback correction."
    if save_tiffs is False:
        flyback_options = ['-S', source, '-i', animalid, '-s', session, '-A', acquisition, \
                       '-r', run, '--flyback=%i' % flyback, '--discard=%i' % discard, \
                       '-z', nslices, '-c', nchannels, '-v', nvolumes, \
                       '--native', '--notiffs']
    else: 
        flyback_options = ['-S', source, '-i', animalid, '-s', session, '-A', acquisition, \
                           '-r', run, \
                           '-z', nslices, '-c', nchannels, '-v', nvolumes, \
                           '--native']

import correct_flyback
correct_flyback.main(flyback_options)

