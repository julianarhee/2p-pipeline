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
import caiman as cm

import time
import pylab as pl

import re
import json
import h5py
import cPickle as pkl
import scipy.io
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
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

#%%
#source = '/nas/volume1/2photon/projects'
#experiment = 'gratings_phaseMod'
#session = '20171009_CE059'
#acquisition = 'FOV1_zoom3x'
#functional = 'functional'
#
#roi_id = 'caiman2Dnmf004'
#roi_method = 'caiman2D'
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

roi_method = 'caiman2D'



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
mc_ids = sorted([m for m in mcparams.keys() if 'mcparams' in m], key=natural_keys)
if len(mc_ids)>1:
    for mcidx,mcid in enumerate(sorted(mc_ids, key=natural_keys)):
        print(mcidx, mcid)
    mc_id_idx = raw_input('Select IDX of mc-method to use: ')
    mc_id = mc_ids[int(mc_id_idx)]
    print("Using MC-METHOD: ", mc_id)
else:
    mc_id = mc_ids[0]

mcparams = mcparams[mc_id] #mcparams['mcparams01']
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
#del mcparams


if isinstance(acqmeta['slices'], int):
    nslices = acqmeta['slices']
else:
    nslices = len(acqmeta['slices'])
    
print(nslices)

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

#%% Create TRACE dir, and update ROI-ID if needed:
    
if len(roi_subdir)>0 and roi_subdir not in roi_id:
    roi_id = roi_id + '_' + roi_subdir

print("ROI ID:", roi_id)

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
        
    mcmetrics = metrics_info[mc_id]
    print(mcmetrics)
    if len(mcmetrics['bad_files'])>0:
        bad_fids = [int(i)-1 for i in mcmetrics['bad_files']]
        bad_files = ['File%03d' % int(i) for i in mcmetrics['bad_files']]
        print("Bad MC files excluded:", bad_files)

file_names = [t for i,t in enumerate(sorted(all_file_names, key=natural_keys)) if i not in bad_fids]

print("Files that passed MC:", file_names)

#%% Exclude really bad runs:
    
if 'excluded' in thr_params.keys():
    exclude = [str(i) for i in thr_params['excluded']]
else:
    exclude = []

print("Excluding files:", exclude)
file_names = sorted([f for f in file_names if f not in exclude], key=natural_keys)
print ("Extracting ROIs for Files:")
print(file_names)

#%% Get NMF output files:
nmf_output_dir = os.path.join(roi_dir, 'nmf_output')
all_nmf_fns = sorted([n for n in os.listdir(nmf_output_dir) if n.endswith('npz')], key=natural_keys)
nmf_fns = []
for f in file_names:
    match_nmf = [m for m in all_nmf_fns if f in m][0]
    nmf_fns.append(match_nmf)
    
ref_nmf_fn = [f for f in nmf_fns if roi_ref in f][0]

print("Using reference: %s" % roi_ref)

#%% Get source tiffs (mmap):
    
tiff_source = str(mcparams['dest_dir'][0][0][0])
tiff_dir = os.path.join(acquisition_dir, functional, 'DATA', tiff_source)
#tiff_dir

# Get mmap tiffs:
all_memmapped_fns = sorted([m for m in os.listdir(tiff_dir) if m.endswith('mmap')], key=natural_keys)
memmapped_fns = []
for cf in file_names:
    match_mmap = [f for f in all_memmapped_fns if cf in f][0]
    memmapped_fns.append(match_mmap)


#%% Create output dirs:
    
# Create dirs for TRACES:
trace_dir = os.path.join(acqmeta['trace_dir'], roi_id, mc_id)
if not os.path.exists(trace_dir):
    os.makedirs(trace_dir)

trace_dir_movs = os.path.join(trace_dir, 'movs')
trace_dir_figs = os.path.join(trace_dir, 'figs')
if not os.path.exists(trace_dir_movs):
    os.mkdir(trace_dir_movs)
if not os.path.exists(trace_dir_figs):
    os.mkdir(trace_dir_figs)


#%% Check ACQMETA fields (rolodex updating...)

I = dict()
I['average_source'] = tiff_source
I['corrected'] = int(mcparams['corrected'])
I['functional'] = functional
I['mc_id'] = mc_id
I['mc_method'] = str(mcparams['method'][0][0][0])
I['roi_id'] = roi_id
I['roi_method'] = roi_method
I['signal_channel'] = signal_channel_idx
I['slices'] = len(np.arange(0,nslices))
    
#%% Create and update ANALYSIS structs:
import initialize_analysis

infodict = {'I': I, 'acquisition_dir': acquisition_dir, 'functional': functional}

I = initialize_analysis.main(**infodict)
pp.pprint(I)

#%%

if len(roi_subdir)>0:
    with open(os.path.join(roi_dir, roi_subdir, 'matchedROIs_ref%s.json' % thr_params['roi_ref']), 'r') as f:
        roidict = json.load(f)
else:
    roidict = {}

print("Found coregistered ROIs:")
pp.pprint(roidict)


#%% #currslice = 0

for currslice in range(nslices):
    tracestruct = dict()
    tracestruct['file'] = dict() #np.array((int(acqmeta['ntiffs']),))
    
    abort = False
    
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
    for fid,curr_file in enumerate(sorted(file_names, key=natural_keys)): #['File001']): #roiparams.keys():
        #%%
        fid = 0; curr_file = file_names[0]
        
        tracestruct['file'][fid] = dict()

        curr_nmf_fn = [n for n in nmf_fns if curr_file in n][0]
        nmf = np.load(os.path.join(nmf_output_dir, curr_nmf_fn))
        
        nr = nmf['A'].all().shape[1]
        d1 = int(nmf['d1'])
        d2 = int(nmf['d2'])
        dims = (d1, d2)
        
        A = nmf['A'].all()
        if thresholded is True:
            kept = nmf['idx_components']
            A = A.tocsc()[:, kept]
            nr = A.shape[-1]
            C = nmf['C'][kept, :]
            nmf['idx_components']
            Cdf = nmf['Cdf'][kept,:]
            S = nmf['S'][kept,:]
        else:
            A = nmf['A'].all()
            C = nmf['C']
            nmf['YrA'] = nmf['YrA']
            Cdf = nmf['Cdf']

        print(Cdf.shape)       
        f = nmf['f']
        b = nmf['b']
        
        #%%
        if thresholded is True:
            rois = roidict[curr_file]
            print("Keeping %i out of %i ROIs." % (len(rois), nr))
            nrois = len(rois)
        else:
            rois = range(nr)
            nrois = nr
            
        #%%
        curr_mmap = [m for m in memmapped_fns if curr_file in m][0]
        Yr, dims, T = cm.load_memmap(os.path.join(tiff_dir, curr_mmap))

        #%%
        #Av = nmf['Av']
        #view_patches_bar(Yr, scipy.sparse.coo_matrix(A), C, b, f, dims[0], dims[1], YrA=YrA, img=Av)
        
        #%%
        nr = np.shape(A)[-1]
        nb = b.shape[1]
        
        # Keep background as components:
        Ab = scipy.sparse.hstack((A, b)).tocsc()
        Cf = np.vstack((C, f))

        A2 = Ab.copy()
        A2.data **= 2
        nA2 = np.sqrt(np.array(A2.sum(axis=0))).squeeze()
        
        #normedA = old_div(np.array(A[:, i]), nA2[i])
        
        rA = Ab * spdiags(old_div(1, nA2), 0, nr+nb, nr+nb)

        # Apply spatial components to raw tiff:
        #raw = (A.T.dot(Yr).T)
        raw = rA.T.dot(Yr) # normed A applied to raw
        #raw = raw[rois,:]
        
        #%% Normalize A in standard way:
            
#        normedAb = Ab/np.sum(Ab, axis=0)
#        normedAb_raw = normedAb.T.dot(Yr)
#        normedAb_raw = np.array(normedAb_raw[rois,:])

        #%% Get "reconstructed" traces by adding background components:
            
        # Apply spatial components to denoised tiff:
        #extracted = A.dot(C) + b.dot(f)
        #applied = A.T.dot(extracted)
        reconstructed = rA.dot(Cf)
        recon_traces = rA.T.dot(reconstructed)
        #recon_traces = recon_traces[rois,:]

        #%% Get extracted traces (no background)
        
        extracted = rA[:, 0:-1].dot(C)
        extr_traces = rA[:, 0:-1].T.dot(extracted)
        #extr_traces = extr_traces[rois,:]


        #%%
        # Extracted df/f:
        #Cdf = Cdf #.toarray()
        tracestruct['file'][fid]['df'] = Cdf[rois,:]
        tracestruct['file'][fid]['tracematDC'] = recon_traces[rois,:] #extr_traces
        #tracestruct['file'][fid]['reconstructed'] = extr_traces #recon_traces
        tracestruct['file'][fid]['rawtracemat'] = raw[rois,:]
        tracestruct['file'][fid]['nrois'] = len(rois)
        tracestruct['file'][fid]['roi_idxs'] = rois
        tracestruct['file'][fid]['masks'] = np.reshape(rA.toarray(), (d1, d2, nr+nb), order='F')
        
        if 'S' in nmf.keys():
            tracestruct['file'][fid]['deconvolved'] = S[rois,:]
            
   

        #%%
        dfmat_reconstructed = np.empty((nr+nb, T))
        dfmat_Cdf = np.empty((nr, T))
        dfmat_raw = np.empty((nr+nb, T))
        for r in range(nr+nb): #range(nr+nb):
            
            dfmat_raw[r,:] = (raw[r,:] - np.mean(raw[r,:])) / np.mean(raw[r,:])
            dfmat_reconstructed[r,:] = (recon_traces[r,:] - np.mean(recon_traces[r,:])) / np.mean(recon_traces[r,:])
            
            if r<nr:
                dfmat_Cdf[r,:] = Cdf[r,:]

        #%%

        fig = pl.figure()
        ax = fig.add_subplot(111)
        im = ax.imshow(dfmat_Cdf)
        ax.set_aspect('auto')
        pl.title('Cdf - %s' % curr_file)
        pl.xlabel('frames')
        pl.ylabel('roi')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        pl.colorbar(im, cax=cax)
        pl.tight_layout()
        pl.savefig(os.path.join(trace_dir_figs, '%s_%s_dfmat_Cdf.png' % (I['analysis_id'], curr_file)))
        #pl.close()

        fig = pl.figure()
        ax = fig.add_subplot(111)
        im = ax.imshow(dfmat_raw)
        ax.set_aspect('auto')
        pl.title('raw - %s' % curr_file)
        pl.xlabel('frames')
        pl.ylabel('roi')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        pl.colorbar(im, cax=cax)
        pl.tight_layout()
        pl.savefig(os.path.join(trace_dir_figs, '%s_%s_dfmat_raw.png' % (I['analysis_id'], curr_file)))
        #pl.close()

        fig = pl.figure()
        ax = fig.add_subplot(111)
        im = ax.imshow(dfmat_reconstructed)
        ax.set_aspect('auto')
        pl.title('reconstructed - %s' % curr_file)
        pl.xlabel('frames')
        pl.ylabel('roi')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        pl.colorbar(im, cax=cax)
        pl.tight_layout()
        pl.savefig(os.path.join(trace_dir_figs, '%s_%s_dfmat_reconstructed.png' % (I['analysis_id'], curr_file)))
        #pl.close()
        
        #%%
        if save_movies is True:
            if curr_file==roi_ref: # or curr_file=='File001':
                # %% reconstruct denoised movie
                dfmovie = cm.movie(extracted).reshape(dims + (-1,), order='F').transpose([2, 0, 1])
                dfmovie.save(os.path.join(trace_dir_movs, '%s_%s_DFF_mov_extracted.tif' % (I['analysis_id'], curr_file)))
                
                #%%
                dfmovie = cm.movie(reconstructed).reshape(dims + (-1,), order='F').transpose([2, 0, 1])
                dfmovie.save(os.path.join(trace_dir_movs, '%s_%s_DFF_mov_reconstructed.tif' % (I['analysis_id'], curr_file)))


    #%% SAVE:
        
    base_trace_fn = '%s_traces_Slice%02d_%s' % (I['analysis_id'], currslice+1, signal_channel)
    
    # Save as .mat:
    scipy.io.savemat(os.path.join(trace_dir, '%s.mat' % base_trace_fn), mdict=tracestruct)
    
    # Save as .pkl:
    with open(os.path.join(trace_dir, '%s.pkl' % base_trace_fn), 'wb') as f:
        pkl.dump(tracestruct, f, protocol=pkl.HIGHEST_PROTOCOL)
    
    
#%%
#
#nr = np.shape(A)[-1]
#nb = b.shape[1]
#
#Ab = scipy.sparse.hstack((A, b)).tocsc()
#Cf = np.vstack((C, f))
#
#print("nr:", nr, "nb:", nb)
#print("Ab:", Ab.shape, "Cf:", Cf.shape)
#
##%%
#
#Cdf = nmf['Cdf']
#
#raw = (Ab.T.dot(Yr).T)
#binaryA = A.toarray().astype('bool')
#
#rawbinary = (binaryA.T.dot(Yr).T)
#
#nAb = np.ravel(Ab.power(2).sum(axis=0))
#
#diagsb = spdiags(old_div(1., nAb), 0, nr+nb, nr+nb)
#AAb = ((Ab.T.dot(Ab)) * diagsb).tocsr()
#AAb_unweighted = (Ab.T.dot(Ab))
#
#pl.figure(); pl.subplot(1,2,1); pl.title('AAb = [Ab]T.dot([Ab])'); pl.imshow(AAb_unweighted.toarray()); pl.colorbar()
#pl.subplot(1,2,2); pl.title('AAb = [Ab]T.dot([Ab]) * diagsb'); pl.imshow(AAb.toarray()); pl.colorbar()
#
##%%
#
#trace = raw[:,0]
#trace_df = (trace - np.mean(trace))/np.mean(trace)
#
#applied_trace = applied[0,:]
#applied_df = (applied_trace - np.mean(applied_trace))/np.mean(applied_trace)
#
#extracted_cdf = Cdf[0,:]
#pl.figure(); pl.subplot(3,1,1); pl.title('raw df/f'); pl.plot(range(T), trace_df)
#pl.subplot(3,1,2); pl.title('apply A to AC+bf'); pl.plot(range(T), applied_df)
#pl.subplot(3,1,3); pl.title('extracted df/f from nmf'); pl.plot(range(T), extracted_cdf)
#
##%%
#
#YAb = ((Ab.T.dot(Yr).T) * diagsb)
#YrAb = YAb - AAb.T.dot(Cf).T
#
#pl.figure(); pl.subplot(3,1,1); pl.title('raw: [Ab]T.dot(Yr).T'); pl.plot(range(T), raw[:,0]);
#pl.subplot(3,1,2); pl.title('YAb: raw * diagsb'); pl.plot(range(T), YAb[:,0])
#pl.subplot(3,1,3); pl.title('YrA: YAb - [AA]T.dot([Cf]).T'); pl.plot(range(T), YrAb[:, 0])
#
##%%
#pl.figure(); pl.subplot(2,1,1); pl.title('YrA'); pl.plot(range(T), YrA[0,:])
#
#traces = YrA + C
#pl.subplot(2,1,2); pl.title('YrA + C'); pl.plot(range(T), traces[0,:])
#
##%% 
#
#comps = AAb.T.dot(Cf).T
#extracted = A.dot(C) + b.dot(f)
#applied = A.T.dot(extracted)
#
##%%
#pl.figure(); pl.subplot(3,1,1); pl.title('[Ab].dot(Cf)'); pl.plot(range(T), comps[:,0])
#pl.subplot(3,1,2); pl.title('C'); pl.plot(range(T), C[0,:])
#pl.subplot(3,1,3); pl.title('[A]T.dot(AC + bf)'); pl.plot(range(T), applied[0,:])
#
##%%
#nA = np.ravel(A.power(2).sum(axis=0))
#diags = spdiags(old_div(1., nA), 0, nr, nr)
#AA = ((A.T.dot(A)) * diags)
#
#applied_weighted = AA.T.dot(extracted)
#print(applied_weighted.shape)

