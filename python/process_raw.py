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

source = '/nas/volume1/2photon/projects'
experiment = 'scenes' #'gratings_phaseMod' #'retino_bar' #'gratings_phaseMod'
session = '20171003_JW016' #'20170927_CE059' #'20170902_CE054' #'20170825_CE055'
acquisition = 'FOV1' #'FOV1_zoom3x' #'FOV1_zoom3x_run2' #'FOV1_planar'
functional_dir = 'functional' #'functional_subset'

acquisition_dir = os.path.join(source, experiment, session, acquisition)
# -------------------------------------------------------------
# Set basename for files created containing meta/reference info:
# -------------------------------------------------------------
raw_simeta_basename = 'SI_raw_%s' % functional_dir
reference_info_basename = 'reference_%s' % functional_dir
# -------------------------------------------------------------
# -------------------------------------------------------------


do_flyback_correction = True #True #True
flyback = 0 #1       # Num flyback frames at top of stack [default: 8]


# ----------------------------------------------------------------------------
# 1.  Get SI meta data from raw tiffs:
# ----------------------------------------------------------------------------
simeta_options = ['-S', source, '-E', experiment, '-s', session, '-A', acquisition, '-f', functional_dir]

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
with open(os.path.join(acquisition_dir, simeta_fn), 'r') as fr:
    simeta = json.load(fr)
 
discard = int(simeta['File001']['SI']['hFastZ']['numDiscardFlybackFrames'])
nvolumes = int(simeta['File001']['SI']['hFastZ']['numVolumes'])
# nslices = int(simeta['File001']['SI']['hFastZ']['numFramesPerVolume'])
nslices = int(simeta['File001']['SI']['hStackManager']['numSlices'])
nchannels = len([int(i) for i in simeta['File001']['SI']['hChannels']['channelSave'] if i.isnumeric()])
print "Raw SI info:"
print "N channels: {nchannels}, N slices: {nslices}, N volumes: {nvolumes}".format(nchannels=nchannels, nslices=nslices, nvolumes=nvolumes)
print "Num discarded frames for flyback:", discard

if do_flyback_correction:
    print "Correcting incorrect flyback frames in volumes."

    flyback_options = ['-S', source, '-E', experiment, '-s', session, '-A', acquisition, \
                       '-f', functional_dir, '--flyback=%i' % flyback, '--discard=%i' % discard, \
                       '-z', nslices, '-c', nchannels, '-v', nvolumes, \
                       '--native', '--correct-flyback']
else:
    print "Not doing flyback correction."
    flyback_options = ['-S', source, '-E', experiment, '-s', session, '-A', acquisition, \
                       '-f', functional_dir, \
                       '-z', nslices, '-c', nchannels, '-v', nvolumes, \
                       '--native']

import correct_flyback
correct_flyback.main(flyback_options)

