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
parser.add_option('-i', '--indir', action='store', dest='deinterleaved_dir', default='', help="folder containing tiffs to be interleaved")
parser.add_option('-o', '--outdir', action='store', dest='write_dir', default='', help="path to dir in which to write interleaved tiffs")


(options, args) = parser.parse_args()

source = options.source
experiment = options.experiment
session = options.session
acquisition = options.acquisition
functional_dir = options.functional_dir
deinterleaved_dir = options.deinterleaved_dir
write_dir = options.write_dir

# raw_simeta_basename = 'SI_raw_%s' % functional_dir
reference_info_fn = 'reference_%s.json' % functional_dir

acquisition_dir = os.path.join(source, experiment, session, acquisition)
data_dir =  os.path.join(acquisition_dir, functional_dir, 'DATA')
source_tiff_dir = os.path.join(data_dir, deinterleaved_dir)
if len(write_dir)==0:
    write_dir = data_dir
else:
    write_dir = os.path.join(data_dir, write_dir)
if not os.path.exists(write_dir):
    os.mkdir(write_dir)

tiffs = os.listdir(source_tiff_dir)
tiffs = sorted([t for t in tiffs if t.endswith('.tif')], key=natural_keys)
print "Found %i tiffs." % len(tiffs)

with open(os.path.join(acquisition_dir, reference_info_fn), 'r') as fr:
    ref = json.load(fr)

nfiles = ref['ntiffs']
nchannels = ref['nchannels']
nslices = len(ref['slices'])
nvolumes = ref['nvolumes']
ntotalframes = nslices*nvolumes*nchannels

sample = tf.imread(os.path.join(source_tiff_dir, tiffs[0]))
print sample.shape, sample.dtype

for fidx in range(nfiles):
    print "Interleaving File %i  f %i." % (int(fidx+1), nfiles)
    curr_file = "File%03d" % int(fidx+1)
    interleaved_tiff_fn = "{basename}_{currfile}.tif".format(basename=ref['base_filename'], currfile=curr_file)
    print "New tiff name:", interleaved_tiff_fn
    curr_file_fns = [t for t in tiffs if curr_file in t]
    print "Found %i tiffs for current file." % len(curr_file_fns)
    stack = np.empty((ntotalframes, sample.shape[1], sample.shape[2]), dtype=sample.dtype)
    for fn in curr_file_fns:
        curr_tiff = tf.imread(os.path.join(source_tiff_dir, fn))
        sl_idx = int(fn.split('Slice')[1][0:2]) - 1
        ch_idx = int(fn.split('Channel')[1][0:2]) - 1 
        slice_indices = np.arange((sl_idx*nchannels)+ch_idx, ntotalframes)
        idxs = slice_indices[::(nslices*nchannels)]
        stack[idxs,:,:] = curr_tiff
    
    tf.imsave(os.path.join(write_dir, interleaved_tiff_fn), stack)

