#!/usr/bin/env python2 
''' 
Run this script to interleave deinterleaved files. Assumes standard SliceNN_ChannelNN_FileNNN.tif format. 
Native format is int16. Loads reference .json file created in Step 1 of pipeline (process_raw.py)
 
Run python reinterleave_tiffs.py -h for all input options. 
''' 
import os 
import optparse 
import shutil
import json
import re
import cv2
import numpy as np 
import tifffile as tf 
from skimage import img_as_uint 
from skimage import exposure 
from pipeline.python.utils import zproj_tseries

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]


parser = optparse.OptionParser() 
# PATH opts:
parser.add_option('-D', '--root', action='store', dest='rootdir', default='/nas/volume1/2photon/data', help='source dir (root project dir containing all expts) [default: /nas/volume1/2photon/data]')
parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')
parser.add_option('-S', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID') 
parser.add_option('-A', '--acq', action='store', dest='acquisition', default='', help="acquisition folder (ex: 'FOV1_zoom3x')")
parser.add_option('-R', '--run', action='store', dest='run', default='', help='name of run to process') 
parser.add_option('--slurm', action='store_true', dest='slurm', default=False, help="set if running as SLURM job on Odyssey")

parser.add_option('-s', '--source', action='store', dest='source_dir', default=None, help="folder from which to create z-projected slice images")
parser.add_option('-o', '--outdir', action='store', dest='write_dir', default=None, help="path to save averaged slices [default appends <sourcedir>_<zprojtype>_deinterleaved/")
parser.add_option('-z', '--zproj', action='store', dest='zproj_type', default='mean', help="Method of z-projection to get summary slice image [default: mean]")
parser.add_option('--filter', action='store_true', dest='filter3D', default=False, help="Filter 3D tif for ROI extraction (default, false)")
parser.add_option('-f', '--ftype', action='store', dest='filter_type', default='median', help="Filter type, if --filter set [default: median]")
parser.add_option('-g', '--fsize', action='store', dest='filter_size', default=4, help="Filter size, if --filter set [default: 4]")




(options, args) = parser.parse_args() 

# -------------------------------------------------------------
# INPUT PARAMS:
# -------------------------------------------------------------
rootdir = options.rootdir #'/nas/volume1/2photon/projects'
slurm = options.slurm
if slurm is True and 'coxfs01' not in rootdir:
    rootdir = '/n/coxfs01/2p-data'
animalid = options.animalid
session = options.session #'20171003_JW016' #'20170927_CE059'
acquisition = options.acquisition #'FOV1' #'FOV1_zoom3x'
run = options.run

source_dir = options.source_dir
write_dir = options.write_dir
zproj_type = options.zproj_type
filter3D = options.filter3D
filter_type = options.filter_type
filter_size = int(options.filter_size)
runmeta_path = os.path.join(rootdir, animalid, session, acquisition, run, '%s.json' % run)

zproj_tseries(source_dir, runmeta_path, zproj_type=zproj_type, write_dir=write_dir, filter3D=filter3D, filter_type=filter_type, filter_size=filter_size)
