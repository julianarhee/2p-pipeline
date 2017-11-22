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
#from caiman.base.rois import com
from caiman.utils.visualization import get_contours

import time
import pylab as pl

import re
import json
import h5py
import cPickle as pkl
import scipy.io
import pprint
import optparse

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

#source = '/nas/volume1/2photon/projects'
#experiment = 'gratings_phaseMod'
#session = '20171009_CE059'
#acquisition = 'FOV1_zoom3x'
#functional = 'functional'
#
#roi_id = 'caiman2Dnmf004'
#
use_kept_only = True # Value matters if not using thresholded/matched subset (coregister_rois.py)

parser = optparse.OptionParser()

parser.add_option('-S', '--source', action='store', dest='source', default='/nas/volume1/2photon/projects', help='source dir (root project dir containing all expts) [default: /nas/volume1/2photon/projects]')
parser.add_option('-E', '--experiment', action='store', dest='experiment', default='', help='experiment type (parent of session dir)')
parser.add_option('-s', '--sess', action='store', dest='session', default='', help='session name')
parser.add_option('-A', '--acq', action='store', dest='acquisition', default='FOV1', help='acquisition folder')
parser.add_option('-f', '--func', action='store', dest='functional', default='functional', help="folder containing functional tiffs [default: 'functional']")
parser.add_option('-R', '--roi', action='store', dest='roi_id', default='', help="unique ROI ID (child of <acquisition_dir>/ROIs/")


(options, args) = parser.parse_args() 

source = options.source #'/nas/volume1/2photon/projects'
experiment = options.experiment #'gratings_phaseMod'
session = options.session #'20171009_CE059'
acquisition = options.acquisition #'FOV1_zoom3x'
functional = options.functional # 'functional'

roi_id = options.roi_id #'caiman2Dnmf003'


#reuse_reference = True
#ref_file = 6
#ref_filename = 'File%03d' % ref_file
#%%
acquisition_dir = os.path.join(source, experiment, session, acquisition)

acquisition_meta_fn = os.path.join(acquisition_dir, 'reference_%s.json' % functional)
with open(acquisition_meta_fn, 'r') as f:
    acqmeta = json.load(f)

roi_dir = os.path.join(acqmeta['roi_dir'], roi_id)
roiparams_path = os.path.join(roi_dir, 'roiparams.json')
with open(roiparams_path, 'r') as f:
    roiparams = json.load(f)

roiparams = byteify(roiparams)
if not roi_id==roiparams['roi_id']:
    print("***WARNING***")
    print("Loaded ROIPARAMS id doesn't match user-specified roi_id.")
    pp.pprint(roiparams)
    use_loaded = raw_input('Use loaded ROIPARAMS? Press Y/n: ')
    if use_loaded=='Y':
        roi_id = roiparams['roi_id']
        

if 'subsets' in roiparams.keys():
    for sidx,subset in enumerate(roiparams['subsets']):
        print(sidx, str(subset.split('/')[-1]))
    
    roi_set_choice = raw_input("Enter IDX of subset to use: ")
    roi_subdir = str(roiparams['subsets'][int(roi_set_choice)].split('/')[-1])
    
else:
    print("Using ALL ROIs...")
    roi_subdir = ''


#%% Create unique ROI dir:

roi_dir = os.path.join(acqmeta['roi_dir'], roi_id) #, roi_subdir)    
    
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

#%%  GET ROI SUBSET info, if relevant:
    
if len(roi_subdir)>0:
    with open(os.path.join(roi_dir, roi_subdir, 'threshold_params.json'), 'r') as f:
        thr_params = json.load(f)
    roi_ref = thr_params['roi_ref']
    thresholded = thr_params['threshold']


else:
    thr_params = dict()
    roi_ref = reference_file
    thresholded = use_kept_only

print("Threshold params:")
pp.pprint(thr_params)
print("Thresholding:", thresholded)
print("Reference file:", reference_file)


#%% 
all_file_names = sorted(['File%03d' % int(f+1) for f in range(acqmeta['ntiffs'])], key=natural_keys)

#%% only run on good MC files:

metrics_path = os.path.join(acqmeta['acquisition_base_dir'], functional, 'DATA', 'mcmetrics.json')
print(metrics_path)
bad_files = []
bad_fids = []
if os.path.exists(metrics_path):
    with open(metrics_path, 'r') as f:
        metrics_info = json.load(f)
        
    mcmetrics = metrics_info[mc_method]
    print(mcmetrics)
    if len(mcmetrics['bad_files'])>0:
        bad_fids = [int(i)-1 for i in mcmetrics['bad_files']]
        bad_files = ['File%03d' % int(i) for i in mcmetrics['bad_files']]
        print("Bad MC files excluded:", bad_files)

file_names = [t for i,t in enumerate(sorted(all_file_names, key=natural_keys)) if i not in bad_fids]

print("Files that passed MC:", file_names)

#%% 

# source of NMF output run:
nmf_output_dir = os.path.join(roi_dir, 'nmf_output')
all_nmf_fns = sorted([n for n in os.listdir(nmf_output_dir) if n.endswith('npz')], key=natural_keys)
nmf_fns = []
for f in file_names:
    match_nmf = [m for m in all_nmf_fns if f in m][0]
    nmf_fns.append(match_nmf)
    
ref_nmf_fn = [f for f in nmf_fns if roi_ref in f][0]

# Create dirs for ROIs:
if len(roi_subdir)==0:
    new_roi_dir = roi_dir
else:
    new_roi_dir = roi_dir + '_' + roi_subdir

if not os.path.exists(new_roi_dir):
    os.makedirs(new_roi_dir)
    
roi_fig_dir = os.path.join(new_roi_dir, 'figures')
roi_mask_dir = os.path.join(new_roi_dir, 'masks')
if not os.path.exists(roi_fig_dir):
    os.mkdir(roi_fig_dir)
if not os.path.exists(roi_mask_dir):
    os.mkdir(roi_mask_dir)

sourcepaths = [] #np.zeros((nslices,), dtype=np.object)
maskpaths = [] #np.zeros((nslices,), dtype=np.object)
maskpaths_mat = [] #np.zeros((nslices,), dtype=np.object)
roi_info = [] #np.zeros((nslices,), dtype=np.object)  # For blobs, this is center/radii; for NMF...?
nrois_by_slice = [] #np.zeros((nslices,))

        
#%% Exclude really bad runs:
    
if 'excluded' in thr_params.keys():
    exclude = [str(i) for i in thr_params['excluded']]
else:
    exclude = []

print("Excluding files:", exclude)
file_names = sorted([f for f in file_names if f not in exclude], key=natural_keys)
print ("Extracting ROIs for Files:")
print(file_names)

#%%
#currslice = 0
if len(roi_subdir)>0:
    with open(os.path.join(roi_dir, roi_subdir, 'matchedROIs_ref%s.json' % thr_params['roi_ref']), 'r') as f:
        roidict = json.load(f)
else:
    roidict = {}
    
#%%
    
for currslice in range(nslices):
    
    abort = False
    maskstruct = dict((f, dict()) for f in file_names)
    
    if len(roidict.keys())==0:
        print("No file found for matched ROIs. Did you run coregister.py?")
        print("Assuming common reference and roi IDs from CNMF extraction...")
        
        ref_nmf_fn = [n for n in nmf_fns if roi_ref in n][0]
        ref_nmf = np.load(os.path.join(nmf_output_dir, ref_nmf_fn))
        kept = [i for i in ref_nmf['idx_components']]
    
        if thresholded is True and roiparams['params']['use_reference'] is True:
            print("Taking KEPT subset")
            #if 'kept' in roiparams['params'].keys():
            ref_nmf_fn = [n for n in nmf_fns if roi_ref in n][0]
            ref_nmf = np.load(os.path.join(nmf_output_dir, ref_nmf_fn))
            kept = [i for i in ref_nmf['idx_components']]
            for fid,curr_file in enumerate(sorted(file_names, key=natural_keys)):
                if curr_file in exclude:
                    continue
                curr_nmf_fn = [n for n in nmf_fns if curr_file in n][0]
                print(curr_nmf_fn)
                nmf = np.load(os.path.join(nmf_output_dir, curr_nmf_fn))
                curr_kept = [i for i in nmf['idx_components']]
                kept = list(set(kept) & set(curr_kept))
                print(kept)
                
            roiparams['params']['kept_rois'] = kept
            print("KEPT:", kept)
            roidict = dict((curr_f, kept) for curr_f in sorted(file_names, key=natural_keys))
        
        else:
            print("Common reference was not used in CNMF extraction, ID: %s", roi_id)
            print("Aborting.")
            abort = True
    
    if abort is True:
        continue
    
    
    #%%
    for curr_file in file_names: #['File001']: #roiparams.keys():

        if curr_file in exclude:
            continue
        
        #
        print("Extracting ROI STRUCT from %s" % curr_file)
        
        
        curr_nmf_fn = [n for n in nmf_fns if curr_file in n][0]
        nmf = np.load(os.path.join(nmf_output_dir, curr_nmf_fn))
        
        nr = nmf['A'].all().shape[1]
        d1 = int(nmf['d1'])
        d2 = int(nmf['d2'])
        dims = (d1, d2)
        
        A = nmf['A'].all()
        if thresholded is True:
            A = A.tocsc()[:, nmf['idx_components']]
            nr = A.shape[-1]

#
        A2 = A.copy()
        A2.data **= 2
        nA2 = np.sqrt(np.array(A2.sum(axis=0))).squeeze()
        
        #normedA = old_div(np.array(A[:, i]), nA2[i])
        
        rA = A * spdiags(old_div(1, nA2), 0, nr, nr)
        rA = rA.todense()

#
        masks = np.reshape(np.array(rA), (d1, d2, nr), order='F')
        
        if thresholded is True:
            kept = roidict[curr_file]
            masks = masks[:,:,kept]
            print("Keeping %i out of %i ROIs." % (len(kept), nr))
            nrois = len(kept)
        else:
            nrois = nr
            
            
        print('Mask array:', masks.shape)
                
        if not 'Av' in nmf.keys():
            img = np.mean(masks[:,:,:-1], axis=-1)
        else: 
            img = nmf['Av']
#
        coors = get_contours(A, dims, thr=0.9)
        if thresholded is True:
            coors = [coors[i] for i in kept]
            
        cc1 = [[l[0] for l in n['coordinates']] for n in coors]
        cc2 = [[l[1] for l in n['coordinates']] for n in coors]
        coords = [[(x,y) for x,y in zip(cc1[n], cc2[n])] for n in range(len(cc1))] 
        com = [list(n['CoM']) for n in coors]
#
        vmax = np.percentile(img, 98)
        pl.figure()
        pl.imshow(img, interpolation='None', cmap=pl.cm.gray, vmax=vmax)
        for roi in range(nrois):
            masktmp = masks[:,:,roi]
            msk = masktmp.copy() 
            msk[msk==0] = np.nan
            pl.imshow(msk, interpolation='None', alpha=0.3, cmap=pl.cm.hot)
            [ys, xs] = np.where(masktmp>0)
            pl.text(xs[int(round(len(xs)/4))], ys[int(round(len(ys)/4))], str(kept[roi]), weight='bold')
            pl.axis('off')
        pl.colorbar()
        pl.tight_layout()
    
 
        imname = '%s_%s_Slice%02d_%s_%s_ROI.png' % (session,acquisition,currslice+1,signal_channel, curr_file)
        print(imname) 
        pl.savefig(os.path.join(roi_fig_dir, imname))
        pl.close()
        
        maskstruct[curr_file] = masks
        #
   
#    if len(roi_subdir)==0:
#        roisubstr = '_all'
#    else:
#        roisubstr = '_' + re.sub('_', '', roi_subdir)
        
    base_mask_fn = '%s_%s_Slice%02d_%s_masks' % (session, acquisition, currslice+1, signal_channel)
    
    #
    # Save as .mat:
    scipy.io.savemat(os.path.join(roi_mask_dir, '%s.mat' % base_mask_fn), mdict=maskstruct)
    
    # Save as .pkl:
    with open(os.path.join(roi_mask_dir, '%s.pkl' % base_mask_fn), 'wb') as f:
        pkl.dump(maskstruct, f, protocol=pkl.HIGHEST_PROTOCOL)
    
    
    maskpaths_mat.append(str(os.path.join(roi_mask_dir, '%s.mat' % base_mask_fn)))
    maskpaths.append(str(os.path.join(roi_mask_dir, '%s.pkl' % base_mask_fn)))
    nrois_by_slice.append(nrois)
    sourcepaths.append(str(os.path.join(nmf_output_dir, ref_nmf_fn)))
    
    # TODO:  Need to adjust this to work with 3D NMF (currently, for 2D):
        
    roi_info.append({'com': com, 'coords': coords})

    

#%%

#print maskpaths
#dkeys = [k for k in params_dict.keys() if not k=='fname']
roiparamdict = dict()
roiparamdict['nrois']=nrois_by_slice
roiparamdict['roi_info']= roi_info
roiparamdict['sourcepaths']=sourcepaths
roiparamdict['maskpaths_mat']=maskpaths_mat
roiparamdict['maskpaths']=maskpaths
roiparamdict['maskpath3d']=[]

#%%
# Save main ROIPARAMS as mat:
tmpr = byteify(roiparamdict)
scipy.io.savemat(os.path.join(new_roi_dir,'roiparams.mat'), {'roiparams': tmpr}, oned_as='row') #, mdict={'roiparams': roiparams})

# Save as .PKL:
with open(os.path.join(new_roi_dir, 'roiparams.pkl'), 'wb') as f:
    pkl.dump(roiparamdict, f, protocol=pkl.HIGHEST_PROTOCOL)


with open(os.path.join(new_roi_dir, 'roiparams.json'), 'w') as f:
    json.dump(roiparamdict, f, indent=True, sort_keys=True)
