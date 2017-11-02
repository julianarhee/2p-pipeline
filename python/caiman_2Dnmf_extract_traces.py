#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 18:30:20 2017

@author: julianarhee
"""

from __future__ import division
from __future__ import print_function
import matplotlib
matplotlib.use('TkAgg')
from builtins import zip
from builtins import str
from builtins import map
from builtins import range
from past.utils import old_div
import cv2
import glob
import numpy as np
import os
from scipy.sparse import spdiags, issparse

# import caiman
from caiman.base.rois import com

import time
import pylab as pl

import re
import json
import h5py
import cPickle as pkl
import scipy.io

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def serialize_json(instance=None, path=None):
    dt = {}
    dt.update(vars(instance))


source = '/nas/volume1/2photon/projects'
experiment = 'gratings_phaseMod'
session = '20171009_CE059'
acquisition = 'FOV1_zoom3x'
functional = 'functional'

roi_id = 'caiman2Dnmf001'
inspect_components = False
display_average = True
reuse_reference = True
#ref_file = 6
#ref_filename = 'File%03d' % ref_file

acquisition_dir = os.path.join(source, experiment, session, acquisition)

acquisition_meta_fn = os.path.join(acquisition_dir, 'reference_%s.json' % functional)
with open(acquisition_meta_fn, 'r') as f:
    acqmeta = json.load(f)
    
#%%
# Load mcparams.mat:
mcparams = scipy.io.loadmat(acqmeta['mcparams_path'])
mc_methods = sorted([m for m in mcparams.keys() if 'mcparams' in m], key=natural_keys)
if len(mc_methods)>1:
    for mcidx,mcid in enumerate(sorted(mc_methods, key=natural_keys)):
        print(mcidx, mcid)
    mc_method_idx = raw_input('Select IDX of mc-method to use: ')
    mc_method = mc_methods[int(mc_method_idx)]
    print("Using MC-METHOD: ", mc_method)
else:
    mc_method = mc_methods[0]

mcparams = mcparams[mc_method] #mcparams['mcparams01']
reference_file_idx = int(mcparams['ref_file'])
signal_channel_idx = int(mcparams['ref_channel'])

signal_channel = 'Channel%02d' % int(signal_channel_idx)
reference_file = 'File%03d' % int(reference_file_idx)
if signal_channel_idx==0:
    signal_channel_idx = input('No ref channel found. Enter signal channel idx (1-indexing): ')
if reference_file_idx==0:
    reference_file_idx = input('No ref file found. Enter file idx (1-indexing): ')

signal_channel = 'Channel%02d' % int(signal_channel_idx)
reference_file = 'File%03d' % int(reference_file_idx)
print("Specified signal channel is:", signal_channel)
print("Selected reference file:", reference_file)
del mcparams

    
   #%% 
roi_dir = os.path.join(acqmeta['roi_dir'], roi_id)

# source of NMF output run:
nmf_output_dir = os.path.join(roi_dir, 'nmf_output')
nmf_fns = sorted([n for n in os.listdir(nmf_output_dir) if n.endswith('npz')], key=natural_keys)

ref_nmf_fn = [f for f in nmf_fns if reference_file in f][0]

file_names = sorted(['File%03d' % int(f+1) for f in range(acqmeta['ntiffs'])], key=natural_keys)
if not len(file_names)==len(nmf_fns):
    print('***ALERT***')
    print('Found NMF results does not match num tiff files.')

# Create dirs for TRACES:
trace_dir = os.path.join(acqmeta['trace_dir'], roi_id, mc_method)
if not os.path.exists(trace_dir):
    os.makedirs(trace_dir)


if isinstance(acqmeta['slices'], int):
    nslices = acqmeta['slices']
else:
    nslices = len(acqmeta['slices'])
    
#currslice = 0

for currslice in range(nslices):
    tracestruct = dict()
    tracestruct['file'] = dict() #np.array((int(acqmeta['ntiffs']),))
       
    for fid,curr_file in enumerate(['File001']): #roiparams.keys():
        tracestruct['file'][fid] = dict()

        print("Extracting ROI STRUCT from %s" % curr_file)
        curr_nmf_fn = [n for n in nmf_fns if curr_file in n][0]
        nmf = np.load(os.path.join(nmf_output_dir, curr_nmf_fn))
        
        nr = nmf['A'].all().shape[1]
        d1 = int(nmf['d1'])
        d2 = int(nmf['d2'])
        
        A = nmf['A'].all()
        C = nmf['C']
        if nmf['YrA'].dtype=='float64':
            YrA = nmf['YrA']
        elif nmf['YrA']=='O':
            YrA = nmf['YrA'].all()
        
        f = nmf['f']
        b = nmf['b']
        
        tracestruct['file'][fid]['tracematDC'] = YrA + C
        tracestruct['file'][fid]['rawtracemat'] = A.dot(C) + b.dot(f)       
   
    
    base_trace_fn = 'traces_Slice%02d_%s' % (currslice+1, signal_channel)
    
    # Save as .mat:
    scipy.io.savemat(os.path.join(trace_dir, '%s.mat' % base_trace_fn), mdict=tracestruct)
    
    # Save as .pkl:
    with open(os.path.join(trace_dir, '%s.pkl' % base_trace_fn), 'wb') as f:
        pkl.dump(tracestruct, f, protocol=pkl.HIGHEST_PROTOCOL)
    
    
   
