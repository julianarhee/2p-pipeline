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
import cv2

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
parser.add_option('-i', '--indir', action='store', dest='average_source_dir', default='', help="folder from which to create averaged slices [default: Parsed]")
parser.add_option('-o', '--outdir', action='store', dest='write_dir', default='', help="path to save averaged slices [default: Averaged_Slices_Parsed")


(options, args) = parser.parse_args()

source = options.source
experiment = options.experiment
session = options.session
acquisition = options.acquisition
functional_dir = options.functional_dir
average_source_dir = options.average_source_dir
write_dir = options.write_dir

# raw_simeta_basename = 'SI_raw_%s' % functional_dir
reference_info_fn = 'reference_%s.json' % functional_dir

acquisition_dir = os.path.join(source, experiment, session, acquisition)
data_dir =  os.path.join(acquisition_dir, functional_dir, 'DATA')
if len(average_source_dir)==0:
    average_source_dir = os.path.join(data_dir, 'Parsed')
    if not os.path.exists(average_source_dir):
        print "No parsed time-series found..."
    write_dir = 'Averaged_Slices_Parsed'
else:
    write_dir = 'Averaged_Slices_%s' % average_source_dir
    average_source_dir = os.path.join(data_dir, average_source_dir)

write_dir = os.path.join(data_dir, write_dir)
if not os.path.exists(write_dir):
    os.mkdir(write_dir)

# Get reference struct info:
with open(os.path.join(acquisition_dir, reference_info_fn), 'r') as fr:
    ref = json.load(fr)

nfiles = ref['ntiffs']
nchannels = ref['nchannels']
nslices = len(ref['slices'])
nvolumes = ref['nvolumes']
ntotalframes = nslices*nvolumes*nchannels

# Get interleaved tiffs:
channel_dirs = os.listdir(average_source_dir)
channel_dirs = sorted([c for c in channel_dirs if 'Channel' in c], key=natural_keys)
for ch_dir in channel_dirs:
    file_dirs = os.listdir(os.path.join(average_source_dir, ch_dir))
    file_dirs = sorted([f for f in file_dirs if 'File' in f], key=natural_keys)
    for fi_dir in file_dirs:
        slice_fns = os.listdir(os.path.join(average_source_dir, ch_dir, fi_dir))
        slice_fns = sorted([t for t in slice_fns if t.endswith('.tif')], key=natural_keys)
        curr_write_dir = os.path.join(write_dir, ch_dir, fi_dir)
        curr_write_dir_vis = os.path.join(write_dir, ch_dir, '%s_visible' % fi_dir)
        if not os.path.exists(curr_write_dir):
            os.makedirs(curr_write_dir)
        if not os.path.exists(curr_write_dir_vis): 
            os.makedirs(curr_write_dir_vis)

        for sl_idx,sl_fn in enumerate(sorted(slice_fns, key=natural_keys)):
            curr_slice = "Slice%02d" % int(sl_idx+1)
            curr_tiff = tf.imread(os.path.join(average_source_dir, ch_dir, fi_dir, sl_fn))
            avg = np.mean(curr_tiff, axis=0, dtype=curr_tiff.dtype)
            
            curr_tiff_fn = "average_{currslice}_{currchannel}_{currfile}.tif".format(currslice=curr_slice, currchannel=ch_dir, currfile=fi_dir)
            tf.imsave(os.path.join(curr_write_dir, curr_tiff_fn), avg)
            
            # Save visible, too:   
            curr_tiff_fn_vis = "average_{currslice}_{currchannel}_{currfile}_vis.tif".format(currslice=curr_slice, currchannel=ch_dir, currfile=fi_dir)
            #tmp = (avg - avg.min()) / (avg.max()-avg.min());
            #avg_vis_eq = cv2.equalizeHist(np.uint8(tmp*255))
            # print avg_vis_eq.min(), avg_vis_eq.max()
            #avg_vis = (avg_vis_eq - avg_vis_eq.min()) / (avg_vis_eq.max()-avg_vis_eq.min());
            avg_vis = exposure.rescale_intensity(avg, in_range=(avg.min(), avg.max()))
            tf.imsave(os.path.join(curr_write_dir_vis, curr_tiff_fn_vis), avg_vis*((2**16)-1))

