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
import pprint

pp = pprint.PrettyPrinter(indent=4)


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def serialize_json(instance=None, path=None):
    dt = {}
    dt.update(vars(instance))


def byteify(input):
    if isinstance(input, dict):
        return {byteify(key): byteify(value)
                for key, value in input.iteritems()}
    elif isinstance(input, list):
        return [byteify(element) for element in input]
    elif isinstance(input, unicode):
        return input.encode('utf-8')
    else:
        return input

source = '/nas/volume1/2photon/projects'
experiment = 'gratings_phaseMod'
session = '20171009_CE059'
acquisition = 'FOV1_zoom3x'
functional = 'functional'

roi_id = 'caiman2Dnmf001'
inspect_components = False
display_average = True
use_kept_only = True


#reuse_reference = True
#ref_file = 6
#ref_filename = 'File%03d' % ref_file

acquisition_dir = os.path.join(source, experiment, session, acquisition)

acquisition_meta_fn = os.path.join(acquisition_dir, 'reference_%s.json' % functional)
with open(acquisition_meta_fn, 'r') as f:
    acqmeta = json.load(f)

roi_dir = os.path.join(acqmeta['roi_dir'], roi_id)

    
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

if isinstance(acqmeta['slices'], int):
    nslices = acqmeta['slices']
else:
    nslices = len(acqmeta['slices'])
print("Processing %i slices." % nslices)

   #%% 

# source of NMF output run:
nmf_output_dir = os.path.join(roi_dir, 'nmf_output')
nmf_fns = sorted([n for n in os.listdir(nmf_output_dir) if n.endswith('npz')], key=natural_keys)

ref_nmf_fn = [f for f in nmf_fns if reference_file in f][0]

file_names = sorted(['File%03d' % int(f+1) for f in range(acqmeta['ntiffs'])], key=natural_keys)
if not len(file_names)==len(nmf_fns):
    print('***ALERT***')
    print('Found NMF results does not match num tiff files.')

# Create dirs for ROIs:
roi_fig_dir = os.path.join(roi_dir, 'figures')
roi_mask_dir = os.path.join(roi_dir, 'masks')
if not os.path.exists(roi_fig_dir):
    os.mkdir(roi_fig_dir)
if not os.path.exists(roi_mask_dir):
    os.mkdir(roi_mask_dir)

sourcepaths = [] #np.zeros((nslices,), dtype=np.object)
maskpaths = [] #np.zeros((nslices,), dtype=np.object)
maskpaths_mat = [] #np.zeros((nslices,), dtype=np.object)
roi_info = [] #np.zeros((nslices,), dtype=np.object)  # For blobs, this is center/radii; for NMF...?
nrois_by_slice = [] #np.zeros((nslices,))

roiparams_path = os.path.join(roi_dir, 'roiparams.json')
if os.path.exists(os.path.join(roi_dir, 'roiparams.json')):
    try:
        with open(os.path.join(roi_dir, 'roiparams.json'), 'r') as f:
            roiparams = json.load(f)
    except:
        roiparams = dict()
else:
    roiparams = dict()

if 'params' not in roiparams.keys() or 'use_reference' not in roiparams['params'].keys():
    print("No ROIPARAMS found. Creating new with ROI_ID: ", roi_id)
    roiparams['roi_id'] = roi_id
    roiparams['params'] = dict()
    roiparams['params']['reference_file'] = reference_file
    roiparams['params']['signal_channel'] = signal_channel
    while True: 
        print("Was a reference file used to constrain cNMF ROI extraction?")
        used_ref = raw_input('Press Y/n: ')
        if used_ref=='Y':
            roiparams['params']['use_reference'] = True
            break
        elif used_ref=='n':
            roiparams['params']['use_reference'] = False 
            break

roiparams['params']['use_kept_only'] = use_kept_only
use_reference = roiparams['params']['use_reference']

        
#%%
#currslice = 0

for currslice in range(nslices):
    maskstruct = dict((f, dict()) for f in file_names)

    if use_reference is True and use_kept_only is True:
        if 'kept' in roiparams['params'].keys():
            kept = roiparams['params']['kept_rois']
        else:
            ref_nmf_fn = [n for n in nmf_fns if reference_file in n][0]
            ref_nmf = np.load(os.path.join(nmf_output_dir, ref_nmf_fn))
            kept = [i for i in ref_nmf['idx_components']]
            for fid,curr_file in enumerate(sorted(file_names, key=natural_keys)): 
                curr_nmf_fn = [n for n in nmf_fns if curr_file in n][0]
                nmf = np.load(os.path.join(nmf_output_dir, curr_nmf_fn))
                curr_kept = [i for i in ref_nmf['idx_components']]
                kept = list(set(kept) & set(curr_kept))
            roiparams['params']['kept_rois'] = kept
        nrois = len(kept)
    else:
        nrois = np.copy(nr)

   
    for curr_file in file_names: #['File001']: #roiparams.keys():
        print("Extracting ROI STRUCT from %s" % curr_file)
        curr_nmf_fn = [n for n in nmf_fns if curr_file in n][0]
        nmf = np.load(os.path.join(nmf_output_dir, curr_nmf_fn))
        
        nr = nmf['A'].all().shape[1]
        d1 = int(nmf['d1'])
        d2 = int(nmf['d2'])
        
        A = nmf['A'].all()
        A2 = A.copy()
        A2.data **= 2
        nA = np.sqrt(np.array(A2.sum(axis=0))).squeeze()
        
        rA = A * spdiags(old_div(1, nA), 0, nr, nr)
        rA = rA.todense()
        masks = np.reshape(np.array(rA), (d1, d2, nr), order='F')
        if use_reference is True and use_kept_only is True:
            masks = masks[:,:,kept]
            print("Keeping %i out of %i ROIs." % (len(kept), nr))
    
        print('Mask array:', masks.shape)
        
        if not 'Av' in nmf.keys():
            img = np.mean(masks[:,:,:-1], axis=-1)
        else: 
            img = nmf['Av']

        vmax = np.percentile(img, 98)
        pl.figure()
        pl.imshow(img, interpolation='None', cmap=pl.cm.gray, vmax=vmax)
        for roi in range(nrois):
            masktmp = masks[:,:,roi]
            msk = masktmp.copy() 
            msk[msk==0] = np.nan
            pl.imshow(msk, interpolation='None', alpha=0.3, cmap=pl.cm.hot)
            [ys, xs] = np.where(masktmp>0)
            pl.text(xs[int(round(len(xs)/4))], ys[int(round(len(ys)/4))], str(roi), weight='bold')

            pl.axis('off')
        pl.tight_layout()
 
        imname = '%s_%s_Slice%02d_%s_%s_ROI.png' % (session,acquisition,currslice+1,signal_channel, curr_file)
        print(imname) 
        pl.savefig(os.path.join(roi_fig_dir, imname))
        pl.close()
        
        maskstruct[curr_file] = masks
   
    
    base_mask_fn = '%s_%s_Slice%02d_%s_masks' % (session, acquisition, currslice+1, signal_channel)
    
    # Save as .mat:
    scipy.io.savemat(os.path.join(roi_mask_dir, '%s.mat' % base_mask_fn), mdict=maskstruct)
    
    # Save as .pkl:
    with open(os.path.join(roi_mask_dir, '%s.pkl' % base_mask_fn), 'wb') as f:
        pkl.dump(maskstruct, f, protocol=pkl.HIGHEST_PROTOCOL)
    
    
    maskpaths_mat.append(str(os.path.join(roi_mask_dir, '%s.mat' % base_mask_fn)))
    maskpaths.append(str(os.path.join(roi_mask_dir, '%s.pkl' % base_mask_fn)))
    nrois_by_slice.append(nrois)
    #roi_info[currslice] = cm
    sourcepaths.append(str(os.path.join(nmf_output_dir, ref_nmf_fn)))
    

#%%

#print maskpaths
#dkeys = [k for k in params_dict.keys() if not k=='fname']
roiparams['nrois']=nrois_by_slice
roiparams['roi_info']=[] #roi_info
roiparams['sourcepaths']=sourcepaths
roiparams['maskpaths_mat']=maskpaths_mat
roiparams['maskpaths']=maskpaths
roiparams['maskpath3d']=[]


# Save main ROIPARAMS as mat:
tmpr = byteify(roiparams)
scipy.io.savemat(os.path.join(roi_dir,'roiparams.mat'), {'roiparams': tmpr}, oned_as='row') #, mdict={'roiparams': roiparams})

# Save as .PKL:
with open(os.path.join(roi_dir, 'roiparams.pkl'), 'wb') as f:
    pkl.dump(roiparams, f, protocol=pkl.HIGHEST_PROTOCOL)

with open(roiparams_path, 'w') as f:
    json.dump(roiparams, f, indent=True, sort_keys=True)
