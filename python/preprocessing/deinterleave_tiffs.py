#!/usr/bin/env python2 
''' 
Run this script to interleave deinterleaved files. Assumes standard SliceNN_ChannelNN_FileNNN.tif format. 
Native format is int16. Loads reference .json file created in Step 1 of pipeline (process_raw.py)
 
Run python reinterleave_tiffs.py -h for all input options. 
''' 
import os 
import numpy as np 
import tifffile as tf 
from skimage import img_as_uint 
from skimage import exposure 
import optparse 
import shutil
import json
import re


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]


parser = optparse.OptionParser() 
parser.add_option('-S', '--source', action='store', dest='source', default='/nas/volume1/2photon/projects', help="source dir (parent directory of sessions) [default: '/nas/volume1/2photon/projects']")
parser.add_option('-E', '--experiment', action='store', dest='experiment', default='', help="stimulus-type (gratings_phaseMod, gratings_static, etc.)")
parser.add_option('-s', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID')
parser.add_option('-A', '--acq', action='store', dest='acquisition', default='', help="acquisition folder (ex: 'FOV1_zoom3x')")
parser.add_option('-f', '--functional', action='store', dest='functional_dir', default='functional', help="folder containing functional TIFFs. [default: 'functional']")
parser.add_option('-i', '--indir', action='store', dest='interleaved_dir', default='', help="folder containing tiffs to be interleaved [default: <path-to-functional-dir>/DATA/]")
parser.add_option('-o', '--outdir', action='store', dest='write_dir', default='Parsed', help="path to dir in which to write deinterleaved tiffs [default: './DATA/Parsed']")


(options, args) = parser.parse_args()

source = options.source
experiment = options.experiment
session = options.session
acquisition = options.acquisition
functional_dir = options.functional_dir
in_dir = options.interleaved_dir
out_dir = options.write_dir

# raw_simeta_basename = 'SI_raw_%s' % functional_dir
reference_info_fn = 'reference_%s.json' % functional_dir

acquisition_dir = os.path.join(source, experiment, session, acquisition)
data_dir =  os.path.join(acquisition_dir, functional_dir, 'DATA')
write_dir = os.path.join(data_dir, out_dir)
if not os.path.exists(write_dir):
    os.mkdir(write_dir)

if len(in_dir)==0:
    interleaved_dir = data_dir
else:
    interleaved_dir = os.path.join(data_dir, in_dir)

# Get ref info:
with open(os.path.join(acquisition_dir, reference_info_fn), 'r') as fr:
    ref = json.load(fr)

nfiles = ref['ntiffs']
nchannels = ref['nchannels']
nslices = len(ref['slices'])
nvolumes = ref['nvolumes']
ntotalframes = nslices*nvolumes*nchannels

# Get interleaved tiffs:
print "Deinterleaving TIFFs found in:", interleaved_dir
tiffs = os.listdir(interleaved_dir)
tiffs = sorted([t for t in tiffs if t.endswith('.tif')], key=natural_keys)

if not len(tiffs)==nfiles:
    print "Mismatch in num TIFFs. Started with %i files." % nfiles
    print "Found %i tiffs in dir: %s" % (len(tiffs), data_dir)
else:
    print "Found %i tiffs." % len(tiffs)


# Load in each TIFF and deinterleave:
for fidx,filename in enumerate(sorted(tiffs, key=natural_keys)):
    print "Deinterleaving File %i of %i [%s]" % (int(fidx+1), nfiles, filename)
    stack = tf.imread(os.path.join(data_dir, filename))
    print "Size:", stack.shape
    curr_file = "File%03d" % int(fidx+1)
    for ch_idx in range(nchannels):
        curr_channel = "Channel%02d" % int(ch_idx+1)
        for sl_idx in range(nslices):
            curr_slice = "Slice%02d" % int(sl_idx+1)
            frame_idx = ch_idx + sl_idx*nchannels
            slice_indices = np.arange(frame_idx, ntotalframes, (nslices*nchannels))
            print "nslices:", len(slice_indices)
            curr_slice_fn = "{basename}_{currslice}_{currchannel}_{currfile}.tif".format(basename=ref['base_filename'], currslice=curr_slice, currchannel=curr_channel, currfile=curr_file)
            tf.imsave(os.path.join(write_dir, curr_slice_fn), stack[slice_indices, :, :])


