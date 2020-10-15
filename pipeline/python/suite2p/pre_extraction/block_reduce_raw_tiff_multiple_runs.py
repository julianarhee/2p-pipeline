#!/usr/bin/env python
# coding: utf-8

'''
Pre-extraction steps for block-reducing and moving files.
Adapted from CE jupyter notebook 

Created May 19, 2020
'''

#import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
import os,glob,sys
import re
import scipy.io
import argparse

import numpy as np
from natsort import natsorted

#sys.path.insert(0, '/n/coxfs01/cechavarria/repos/suite2p')
#sys.path.insert(0, '/home/julianarhee/Repositories/suite2p')

from scipy import ndimage
from suite2p.io.utils import get_tif_list, list_files #list_tifs

from skimage.external.tifffile import imread, TiffFile, imsave
from skimage.measure import block_reduce

def findOccurrences(s, ch):
    return [i for i, letter in enumerate(s) if letter == ch]

def block_mean_stack(stack0, ds_block):
    im0 = block_reduce(stack0[0,:,:],ds_block) 
    print(im0.shape)
    stack1 = np.zeros((stack0.shape[0],im0.shape[0],im0.shape[1]))
    for i in range(0,stack0.shape[0]):
        stack1[i,:,:] = block_reduce(stack0[i,:,:],ds_block) 
    return stack1

def reduce_and_save_tif(fpath, dst_dir='/scratch'):
 
    #i0 = findOccurrences(fn,'/')[-1]
    #i1 = findOccurrences(fn,'_')[-1]

    print("reducing, save to %s" % dst_dir)

    fname = os.path.split(fpath)[-1]
    tif_num = re.findall(r'_\d{5}', fname)[0][1:]

    # Get src info
    run_dir = fpath.split('/raw')[0]
    fov_dir, run = os.path.split(run_dir)
    fov = os.path.split(fov_dir)[-1]

    new_fn = '%s_%s_%s' % (fov, run, tif_num) #fn[i1+1:])
    print(new_fn)

    # Read tif stack
    stack0 = imread(fpath)

    # Black reduce spatially
    stack1 = block_mean_stack(stack0, (2,2))
    print(stack1.shape)

    # Save
    imsave(os.path.join(dst_dir, new_fn), stack1)

    print("--saved: %s" % new_fn)


#provide some info
parser = argparse.ArgumentParser()
parser.add_argument('-fovnum', type=int, default=1)
parser.add_argument('-zoom', type=int, default=2)
parser.add_argument('-animalid')
parser.add_argument('-session')
parser.add_argument('-rootdir', default='/n/coxfs01/2p-data')

opts = parser.parse_args()


#rootdir = '/n/coxfs01/2p-data'
#animalid = 'JC084' #'JC120'
#session = '20190522' #'20191120'
#acquisition = 'FOV1_zoom2p0x' #'FOV1_zoom4p0x'

# Get fov dir
acquisition = 'FOV%i_zoom%ip0x' % (opts.fovnum, opts.zoom)
fov_dir = os.path.join(opts.rootdir, opts.animalid, opts.session, acquisition)
assert os.path.exists(fov_dir), "Unable to find fov dir: %s" % fov_dir
print(fov_dir)

# Set relative output dir 
dst_dir = os.path.join(fov_dir, 'all_combined', 'block_reduced')
if not os.path.isdir(dst_dir):
    os.makedirs(dst_dir)

# Get run list
#run_list = [r for r in os.listdir(os.path.join(rootdir, animalid, session, acquisition)) if re.findall(r'_run\d+', r)]

print(os.listdir(fov_dir))
run_dirs = [os.path.join(fov_dir, r) for r in os.listdir(fov_dir) if re.findall(r'_run\d+', r)]
print(run_dirs)
#run_list = ['scenes_run1','scenes_run2','scenes_run3','scenes_run4','scenes_run5','scenes_run6']
#print(run_list)

for run_dir in run_dirs:
    #run_dir = os.path.join(rootdir, animalid, session, acquisition, run) 
    #src_dir = glob.glob(os.path.join(run_dir, 'raw*'))[0]
    #print(src_dir)
    fpaths = glob.glob(os.path.join(run_dir, 'raw*', '*.tif'))

    for fpath in fpaths:      
        print(fpath)
        #reduce_and_save_tif(fpath, dst_dir=dst_dir)





