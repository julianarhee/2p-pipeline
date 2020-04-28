#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 14:34:46 2019

@author: julianarhee
"""


import cv2
import glob
import logging
import matplotlib as mpl
mpl.use('agg')

import matplotlib.pyplot as plt
import numpy as np
import os
import re
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]

try:
    cv2.setNumThreads(0)
except():
    pass

try:
    if __IPYTHON__:
        # this is used for debugging purposes only. allows to reload classes
        # when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    pass

import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf import params as params
from caiman.utils.utils import download_demo
from caiman.utils.visualization import plot_contours, nb_view_patches, nb_plot_contour
#bpl.output_notebook()

import pylab as pl

import tifffile as tf
import json
import time

logging.basicConfig(format=
                          "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s] [%(process)d] %(message)s",
                    # filename="/tmp/caiman.log",
                    level=logging.WARNING)


#%%

rootdir = '/n/coxfs01/2p-data'
animalid = 'JC084'
session = '20190525' #'20190505_JC083'
session_dir = os.path.join(rootdir, animalid, session)
fov = 'FOV1_zoom2p0x'
run_list = ['gratings']
run_label = 'gratings'

motion_correct = True

fnames = []
for run in run_list:
    if motion_correct:
        fnames_tmp = glob.glob(os.path.join(session_dir, fov, '%s*' % run, 'raw*', '*.tif'))
    else:
        fnames_tmp = [f for f in glob.glob(os.path.join(session_dir, fov, '%s*' % run, 
                                                        'processed', 'processed001*','mcorrected*', '*.tif'))\
                     if len(os.path.split(os.path.split(f)[0])[-1].split('_'))==2]
    fnames.extend(fnames_tmp)
    print("[%s]: added %i tifs to queue." % (run, len(fnames_tmp)))

#print(fnames)

##fnames = glob.glob(os.path.join('/n/coxfs01/2p-data/JC083/20190505/FOV1_zoom2p0x/gratings_run1/raw*', '*.tif'))
#print(fnames)

fnames = sorted(fnames, key=natural_keys)



data_identifier = '|'.join([animalid, session, fov, run_label])

print("*** Dataset: %s ***" % data_identifier)

#%%
results_dir = os.path.join(session_dir, fov, 'caiman_results', run_label)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
print(results_dir)




# dataset dependent parameters
fr = 44.65                             # imaging rate in frames per second
decay_time = 0.4                    # length of a typical transient in seconds

# motion correction parameters
strides = (48, 48)          # start a new patch for pw-rigid motion correction every x pixels
overlaps = (24, 24)         # overlap between pathes (size of patch strides+overlaps)
max_shifts = (6,6)          # maximum allowed rigid shifts (in pixels)
max_deviation_rigid = 3     # maximum shifts deviation allowed for patch with respect to rigid shifts
pw_rigid = False             # flag for performing non-rigid motion correction

# parameters for source extraction and deconvolution
p = 1                       # order of the autoregressive system
gnb = 2                     # number of global background components
merge_thr = 0.85            # merging threshold, max correlation allowed
rf = 15                     # half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
stride_cnmf = 6             # amount of overlap between the patches in pixels
K = 4                       # number of components per patch
gSig = [2, 2]               # expected half size of neurons in pixels
method_init = 'greedy_roi'  # initialization method (if analyzing dendritic data using 'sparse_nmf')
ssub = 1                    # spatial subsampling during initialization
tsub = 5                    # temporal subsampling during intialization

# parameters for component evaluation
min_SNR = 2.0               # signal to noise ratio for accepting a component
rval_thr = 0.85              # space correlation threshold for accepting a component
cnn_thr = 0.99              # threshold for CNN based classifier
cnn_lowest = 0.1 # neurons with cnn probability lower than this value are rejected




#%%

opts_dict = {'fnames': fnames,
            'fr': fr,
            'decay_time': decay_time,
            'strides': strides,
            'overlaps': overlaps,
            'max_shifts': max_shifts,
            'max_deviation_rigid': max_deviation_rigid,
            'pw_rigid': pw_rigid,
            'p': 1,
            'nb': gnb,
            'rf': rf,
            'K': K, 
            'stride': stride_cnmf,
            'method_init': method_init,
            'rolling_sum': True,
            'only_init': True,
            'ssub': ssub,
            'tsub': tsub,
            'merge_thr': merge_thr, 
            'min_SNR': min_SNR,
            'rval_thr': rval_thr,
            'use_cnn': True,
            'min_cnn_thr': cnn_thr,
            'cnn_lowest': cnn_lowest}

opts = params.CNMFParams(params_dict=opts_dict)


#%% start a cluster for parallel processing (if a cluster already exists it will be closed and a new session will be opened)
if 'dview' in locals():
    cm.stop_server(dview=dview)
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)

#%%

# first we create a motion correction object with the parameters specified
mc = MotionCorrect(fnames, dview=dview, **opts.get_group('motion'))
# note that the file is not loaded in memory

#%% Run piecewise-rigid motion correction using NoRMCorre
mc.motion_correct(save_movie=True)
border_to_0 = 0 if mc.border_nan is 'copy' else mc.border_to_0 
# maximum shift to be used for trimming against NaNs

if not os.path.exists(os.path.join(results_dir, 'memmap')):
    os.makedirs(os.path.join(results_dir, 'memmap'))
    
base_name = '%s/memmap/Yr' % results_dir
print(base_name)

# memory map the file in order 'C'
fname_new = cm.save_memmap(mc.mmap_file, base_name=base_name, order='C',
                       border_to_0=border_to_0) # exclude borders


np.savez(os.path.join(results_dir, 'mc_rigid.npz'),
        fname=mc.fname, max_shifts=mc.max_shifts, min_mov=mc.min_mov,
        border_nan=mc.border_nan,
        fname_tot_rig=mc.fname_tot_rig,
        total_template_rig=mc.total_template_rig,
        templates_rig=mc.templates_rig,
        shifts_rig=mc.shifts_rig,
        mmap_file=mc.mmap_file,
        border_to_0=mc.border_to_0)
print("--- saved MC results: %s" % os.path.join(results_dir, 'mc_rigid.npz'))


m_rig = cm.load(mc.fname_tot_rig)
                          
                           # Load the file
Yr, dims, T = cm.load_memmap(fname_new)
#images = np.reshape(Yr.T, [T] + list(dims), order='F') # load frames in python format (T x X x Y)
images = np.reshape(Yr, dims + (T,), order='F')
print(images.shape)
corrim_fpath = os.path.join(results_dir, 'corrimg.npz')
# if os.path.exists(corrim_fpath):
#     saveddata = np.load(corrim_fpath)
#     Cn = saveddata['Cn']
# else:
Cn = cm.local_correlations(images)
# Save:
print("Saving correlation image...", corrim_fpath)
np.savez(corrim_fpath, Cn=Cn)


plt.imshow(Cn.max(0) if len(Cn.shape) == 3 else Cn, cmap='gray',
           vmin=np.percentile(Cn, 1), vmax=np.percentile(Cn, 99))
plt.show()

plt.savefig(os.path.join(results_dir, 'Cn.png'))

print("*** DONE ***")
print(fname_new)

