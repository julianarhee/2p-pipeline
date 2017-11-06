#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 12:42:50 2017

Adapted from demo_pipeline.py (CaImAn)

In caimain env: python setup.py install (in order to import caimain not in ipython)

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

try:
    cv2.setNumThreads(1)
except:
    print('Open CV is naturally single threaded')

try:
    if __IPYTHON__:
        print(1)
        # this is used for debugging purposes only. allows to reload classes
        # when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    print('Not IPYTHON')
    pass


import os
# from os.path import expanduser
# home = expanduser("~")
# 
# caiman_dir = '~/Repositories/CaImAn'
# if '~' in caiman_dir:
#     caiman_dir = caiman_dir.replace('~', home)
# os.chdir(caiman_dir)
# cwd = os.getcwd()
# print(cwd)

import caiman as cm
import numpy as np

import time
import pylab as pl
import psutil
import sys
from ipyparallel import Client
from skimage.external.tifffile import TiffFile
import scipy
import copy

from caiman.utils.utils import save_object, load_object
from caiman.base.rois import extract_binary_masks_blob
from caiman.utils.visualization import plot_contours, view_patches_bar
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.motion_correction import MotionCorrect
from caiman.components_evaluation import estimate_components_quality
from caiman.source_extraction.cnmf.utilities import extract_DF_F

from caiman.components_evaluation import evaluate_components,evaluate_components_CNN

from caiman.tests.comparison import comparison
from caiman.motion_correction import tile_and_correct, motion_correction_piecewise

import re
import json
import h5py

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]


#%%
source = '/nas/volume1/2photon/projects'
experiment = 'gratings_phaseMod'
session = '20171009_CE059'
acquisition = 'FOV1_zoom3x'
functional = 'functional'

roi_id = 'caiman2Dnmf002'
inspect_components = False
display_average = True
reuse_reference = False

use_reference = False
if use_reference is False:
    get_reference = False

acquisition_dir = os.path.join(source, experiment, session, acquisition)

acquisition_meta_fn = os.path.join(acquisition_dir, 'reference_%s.json' % functional)
with open(acquisition_meta_fn, 'r') as f:
    acqmeta = json.load(f)

#%% Load MCPARAMS struct to set TIFF source:

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


#%% Create ROI output dir:


roi_dir = os.path.join(acqmeta['roi_dir'], roi_id)
if not os.path.exists(roi_dir):
    os.mkdir(roi_dir)
    
# Also create an NMF-output dir:
nmf_output_dir = os.path.join(roi_dir, 'nmf_output')
if not os.path.exists(nmf_output_dir):
    os.mkdir(nmf_output_dir)

# Create separate dirs for figures and movies:
nmf_fig_dir = os.path.join(nmf_output_dir, 'figures')
nmf_mov_dir = os.path.join(nmf_output_dir, 'movies')
if not (os.path.exists(nmf_fig_dir) and os.path.exists(nmf_mov_dir)):
    os.mkdir(nmf_fig_dir)
    os.mkdir(nmf_mov_dir)
    
#%% Set TIFF SOURCE
#for aidx,s in enumerate(acqmeta['average_source']):
#    print(aidx, acqmeta['average_source'][s])
#selected_source_idx = int(raw_input('Select IDX of tiff source to use: '))
#selected_source = acqmeta['average_source'].keys()[selected_source_idx]

tiff_source = str(mcparams['dest_dir'][0][0][0])
tiff_dir = os.path.join(acquisition_dir, functional, 'DATA', tiff_source)
tiff_dir


tiffpaths = sorted([str(os.path.join(tiff_dir, fn)) for fn in os.listdir(tiff_dir) if fn.endswith('.tif')], key=natural_keys)
tiffpaths


#%% 

params_movie = {'fname': tiffpaths,                         # List of .tif files in current NMF extraction (acquisition)
               'p': 2,                                      # order of the autoregressive fit to calcium imaging in general one (slow gcamps) or two (fast gcamps fast scanning)
               'merge_thresh': 0.8,                         # merging threshold, max correlation allowed
               'rf': 50,                                    # half-size of the patches in pixels. rf=25, patches are 50x50
               'stride_cnmf': 20,                           # amount of overlap between the patches in pixels
               'K': 4,                                      # number of components per patch
               'is_dendrites': False,                       # if dendritic. In this case you need to set init_method to sparse_nmf
               'init_method': 'greedy_roi',                 # init method can be greedy_roi for round shapes or sparse_nmf for denritic data
               'gSig': [20, 20],                            # expected half size of neurons
               'alpha_snmf': None,                          # this controls sparsity
               'final_frate': 30,                           # frame rate of movie (even considering eventual downsampling)
               'r_values_min_patch': .7,                    # threshold on space consistency
               'fitness_min_patch': -20,                    # threshold on time variability
               'fitness_delta_min_patch': -20,              # threshold on time variability (if nonsparse activity)
               'Npeaks': 10,
               'r_values_min_full': .8,
               'fitness_min_full': - 40,
               'fitness_delta_min_full': - 40,
               'only_init_patch': True,
               'gnb': 1,
               'memory_fact': 1,
               'n_chunks': 10,
               'update_background_components': True,        # whether to update the background components in the spatial phase
               'low_rank_background': True,                 # whether to update the using a low rank approximation. If FalseFalse, all nonzero elements of the background components updated using hals
                                                            # (to be used with one background per patch)
                'num_of_channels': 1,
                'channel_of_neurons': 1
                }
#,
#                
#                'deconv_flag': True
#               }

params_display = {
    'downsample_ratio': .2,
    'thr_plot': 0.8
}
#%% Get REFERENCE nmf:


#ref_file = 6
#ref_filename = 'File%03d' % ref_file

if use_reference is True:
    refname = [str(f) for f in tiffpaths if reference_file in f]
    ref_file_idx = [t for t in tiffpaths].index(os.path.join(tiff_dir, refname[0]))
    print(ref_file_idx, refname)


#%% Check for memmapped files:
memmapped_fns = sorted([m for m in os.listdir(tiff_dir) if m.endswith('mmap')], key=natural_keys)
expected_filenames = sorted(['File%03d' % int(f+1) for f in range(len(tiffpaths))], key=natural_keys)
ntiffs = len(expected_filenames)

if len(memmapped_fns)==len(expected_filenames):
    matching_mmap_fns = [m for m,f in zip(memmapped_fns, expected_filenames) if f in m]
    if len(matching_mmap_fns)==ntiffs:
        do_memmapping = False
    else:
        do_memmapping = True
else:
    do_memmapping = True
    
#%%
# do_memmapping = True
curr_fns = params_movie['fname']


#%% Start cluster:
    
# TODO: todocument
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)


#%% 
def memmap_tiffs(fnames, ref_idx=0):
    
    border_to_0 = 0
        
    idx_xy = None
    
    # TODO: needinfo
    #add_to_movie = min_value #-np.nanmin(m_els) + 1  # movie must be positive
    # if you need to remove frames from the beginning of each file
    remove_init = 0
    # downsample movie in time: use .2 or .1 if file is large and you want a quick answer
    downsample_factor = 1
    #base_name = acqmeta['base_filename'] #fname[0].split('/')[-1][:-4]  # omit this in save_memmap_each() to use original filenames as base for mmap files

    # estimate offset:
    print("Memmapping: %s" % fnames[ref_idx])
    m_orig = cm.load_movie_chain([fnames[ref_idx]])
    m_orig.shape
    
    add_to_movie = -np.nanmin(m_orig)
    
    del m_orig
    
    mmap_fnames = cm.save_memmap_each(
            fnames, dview=dview,
            resize_fact=(1, 1, downsample_factor), remove_init=remove_init, 
            idx_xy=idx_xy, add_to_movie=add_to_movie, border_to_0=border_to_0)

    return mmap_fnames


#%% Do memmapping if needed:
if do_memmapping is True:
    # estimate offset:
    mmap_fnames = memmap_tiffs(curr_fns, ref_idx=ref_file_idx)

else:
    mmap_fnames = [os.path.join(tiff_dir, m) for m in memmapped_fns]

#
mmap_fnames = sorted(mmap_fnames, key=natural_keys)
print(mmap_fnames)

#%% CHECK nmf_output_dir to see if REFERENCE file already processed:

nmf_results = [n for n in os.listdir(nmf_output_dir) if n.endswith('.npz')]

if use_reference is True:
    nmf_results_ref = [n for n in os.listdir(nmf_output_dir) if n.endswith('.npz') and reference_file in n]
    
    if len(nmf_results_ref)==0:
        # No output for reference file found
        print("NO output found for reference file %s. Getting NMF blobs for ref." % reference_file)
        get_reference = True
    else:
        print("NMF ouput for reference file %s found. Processing rest of the files now..." % reference_file)
        reuse_input = raw_input("Do you want to re-do the reference file NMF? Press Y/n: ")
        if reuse_input=='Y':
            reuse_reference = False
            print("Re-doing reference.")
        else:
            reuse_reference = True
            print("Skipping reference file.")
        if reuse_reference is True:
            get_reference = False
        else:
            print("REUSE_REFERENCE set to False.")
            get_reference = True

# %% LOAD MEMMAP FILE
# fname_new='Yr_d1_501_d2_398_d3_1_order_F_frames_369_.mmap'

if use_reference is True:
    ref_mmap = [m for m in mmap_fnames if reference_file in m][0]
    mmap_list = [ref_mmap]
    file_list = [reference_file]
    other_mmaps = sorted([m for m,f in zip(mmap_fnames, expected_filenames) if f in m and reference_file not in m], key=natural_keys)
    other_files = sorted([f for m,f in zip(mmap_fnames, expected_filenames) if f in m and reference_file not in m], key=natural_keys)
    
    for mm,ff in zip(other_mmaps, other_files):
        mmap_list.append(mm)
        file_list.append(ff)
    
    # Load REF results, if they exist:
    if len(nmf_results_ref)>0 and get_reference is False:
        ref = np.load(os.path.join(nmf_output_dir, nmf_results_ref[0]))
        refA = ref['A'].all().astype('bool').toarray()
        
else:
    mmap_list = sorted([m for m,f in zip(mmap_fnames, expected_filenames) if f in m], key=natural_keys)
    file_list = sorted([f for m,f in zip(mmap_fnames, expected_filenames) if f in m], key=natural_keys)
    
#if get_reference is True:
#    mmaps_to_process = [m for m in mmap_fnames if ref_filename in m]
#    corr_filenames = [f for f in expected_filenames if f==ref_filename]
#else:
#    mmaps_to_process = sorted([m for m,f in zip(mmap_fnames, expected_filenames) if f in m], key=natural_keys)
#    corr_filenames = expected_filenames
#    
#    ref = np.load(os.path.join(nmf_output_dir, nmf_results_ref[0]))
#    refA = ref['A'].all().astype('bool')
#%% Use subset of files:
files_todo = [] #['File006'] #['File002', 'File003', 'File005', 'File009']
mmaps_todo = [] #i for i in mmap_list if any(f in i for f in files_todo)]
#print(mmaps_todo)

if len(files_todo)==0:
    files_todo = np.copy(file_list)
    mmaps_todo = np.copy(mmap_list)

print("FILES:", files_todo)

print("MMAP:", mmaps_todo)
#%% Process all files:

#for curr_file,curr_mmap in zip(file_list,mmap_list):
for curr_file,curr_mmap in zip(files_todo,mmaps_todo):

    #curr_file = corr_filenames[0]
    #curr_mmap = mmaps_to_process[0]
    if use_reference is True and get_reference is False and curr_file==reference_file:
        continue
    
    print("Loading: %s" % curr_file)

    #%%
    Yr, dims, T = cm.load_memmap(curr_mmap)
    d1, d2 = dims
    
    images = np.reshape(Yr.T, [T] + list(dims), order='F')

    # %%  checks on movies
    # computationnally intensive
    if np.min(images) < 0:
        # TODO: should do this in an automatic fashion with a while loop at the 367 line
        print("Re-adding to make movie positive. Min is %i." % np.min(images))
        curr_tiff = [os.path.join(tiff_dir, t) for t in tiffpaths if curr_file in t and t.endswith('tif')][0]
        print("Memmapping tiff %s..." % curr_tiff)
        curr_mmap = memmap_tiffs([curr_tiff], ref_idx=0)
        curr_mmap = curr_mmap[0] # Returns list
        reload_memmap = True
        #raise Exception('Movie too negative, add_to_movie should be larger')
    else:
        reload_memmap = False
        
    #if np.sum(np.isnan(images)) > 0:
    #    # TODO: same here
    #    raise Exception('Movie contains nan! You did not remove enough borders')

    #%% reload
    if reload_memmap is True:
        
        Yr, dims, T = cm.load_memmap(curr_mmap)
        d1, d2 = dims
        images = np.reshape(Yr.T, [T] + list(dims), order='F')
        print('Reloaded memmap %s. Min value: %i' % (curr_mmap, np.min(images)))
    
    if np.sum(np.isnan(images)) > 0:
        # TODO: same here
        raise Exception('Movie contains nan! You did not remove enough borders')
    
    Y = np.reshape(Yr, dims + (T,), order='F')
    m_images = cm.movie(images)


    # %% correlation image
    Cn = cm.local_correlations(Y)
    Cn[np.isnan(Cn)] = 0
    
    Av = np.mean(m_images, axis=0)
    pl.subplot(1,2,1)
    pl.imshow(Av, cmap='gray')
    pl.subplot(1,2,2)
    pl.imshow(Cn, cmap='gray', vmax=.95)
    
    # TODO: show screenshot 11
    pl.savefig(os.path.join(nmf_fig_dir, '%s_localcorrs.png' % curr_file))
    pl.close()


    #%% GET CNMF BLOBS:
    
        
    # %% Extract spatial and temporal components on patches
    t1 = time.time()
    
    border_pix = 0 # if motion correction introduces problems at the border remove pixels from border
    if use_reference is True:
        if (get_reference is True):
            cnm = cnmf.CNMF(n_processes=1, k=params_movie['K'], gSig=params_movie['gSig'], merge_thresh=params_movie['merge_thresh'],
                            p=params_movie['p'],
                            dview=dview, rf=params_movie['rf'], stride=params_movie['stride_cnmf'], memory_fact=1,
                            method_init=params_movie['init_method'], alpha_snmf=params_movie['alpha_snmf'],
                            only_init_patch=params_movie['only_init_patch'],
                            gnb=params_movie['gnb'], method_deconvolution='oasis', border_pix=border_pix,
                            low_rank_background=params_movie['low_rank_background']) 
        else:
            cnm = cnmf.CNMF(n_processes=1, rf=None, Ain=refA, skip_refinement=False, only_init_patch=False,
                            gSig=params_movie['gSig'], merge_thresh=params_movie['merge_thresh'],
                            p=params_movie['p'],
                            dview=dview, memory_fact=1,
                            method_init=params_movie['init_method'], alpha_snmf=params_movie['alpha_snmf'],
                            gnb=params_movie['gnb'], method_deconvolution='oasis', border_pix=border_pix,
                            low_rank_background = params_movie['low_rank_background']) 
    else:
        cnm = cnmf.CNMF(n_processes=1, k=params_movie['K'], gSig=params_movie['gSig'], merge_thresh=params_movie['merge_thresh'],
                        p=params_movie['p'],
                        dview=dview, rf=params_movie['rf'], stride=params_movie['stride_cnmf'], memory_fact=1,
                        method_init=params_movie['init_method'], alpha_snmf=params_movie['alpha_snmf'],
                        only_init_patch=params_movie['only_init_patch'],
                        gnb=params_movie['gnb'], method_deconvolution='oasis', border_pix=border_pix,
                        low_rank_background=params_movie['low_rank_background']) 

#%%    
    cnm = cnm.fit(images)
    
    A_tot = cnm.A
    C_tot = cnm.C
    YrA_tot = cnm.YrA
    b_tot = cnm.b
    f_tot = cnm.f
    sn_tot = cnm.sn
    print(('Number of components:' + str(A_tot.shape[-1])))
    
    # %%
    pl.figure()
    # TODO: show screenshot 12`
    # TODO : change the way it is used
    if display_average is True:
        crd = plot_contours(A_tot, Av, thr=params_display['thr_plot'])
    else:
        crd = plot_contours(A_tot, Cn, thr=params_display['thr_plot'])
    
    pl.savefig(os.path.join(nmf_fig_dir, '%s_contours_iter1_%s.png' % (curr_file, roi_id)))
    #pl.close()

    # %% DISCARD LOW QUALITY COMPONENT
    final_frate = params_movie['final_frate']
    r_values_min = params_movie['r_values_min_patch']  # threshold on space consistency
    fitness_min = params_movie['fitness_delta_min_patch']  # threshold on time variability
    # threshold on time variability (if nonsparse activity)
    fitness_delta_min = params_movie['fitness_delta_min_patch']
    Npeaks = params_movie['Npeaks']
    traces = C_tot + YrA_tot
    # TODO: todocument
    idx_components, idx_components_bad = estimate_components_quality(
        traces, Y, A_tot, C_tot, b_tot, f_tot, final_frate=final_frate, Npeaks=Npeaks, r_values_min=r_values_min,
        fitness_min=fitness_min, fitness_delta_min=fitness_delta_min)
    
    print(('Keeping ' + str(len(idx_components)) +
           ' and discarding  ' + str(len(idx_components_bad))))

    # %%
    # TODO: show screenshot 13
#    pl.figure()
#    if display_average is True:
#        crd = plot_contours(A_tot.tocsc()[:, idx_components], Av, thr=params_display['thr_plot'])
#    else:
#        crd = plot_contours(A_tot.tocsc()[:, idx_components], Cn, thr=params_display['thr_plot'])
#        
#    pl.savefig(os.path.join(nmf_fig_dir, '%s_contours_kept_%s.png' % (curr_file, roi_id)))
#    pl.close()

    # %% Don't overwrite bad components, just store separately:

    if (get_reference is True) or (use_reference is False):
        A_tot = A_tot.tocsc()[:, idx_components]
        C_tot = C_tot[idx_components]
    else:
        A_tot_kept = A_tot.tocsc()[:, idx_components]
        C_tot_kept = C_tot[idx_components]


    # %% rerun updating the components to refine
    t1 = time.time()
    if use_reference is True:
        if (get_reference is True):
            cnm = cnmf.CNMF(n_processes=1, k=A_tot.shape, gSig=params_movie['gSig'], merge_thresh=params_movie['merge_thresh'],
                            p=params_movie['p'], dview=dview, Ain=A_tot, Cin=C_tot, b_in = b_tot,
                            f_in=f_tot, rf=None, stride=None, method_deconvolution='oasis', gnb=params_movie['gnb'],
                            low_rank_background=params_movie['low_rank_background'],
                            update_background_components=params_movie['update_background_components'], check_nan = True)
    else:
        cnm = cnmf.CNMF(n_processes=1, k=A_tot.shape, gSig=params_movie['gSig'], merge_thresh=params_movie['merge_thresh'],
                        p=params_movie['p'], dview=dview, Ain=A_tot, Cin=C_tot, b_in = b_tot,
                        f_in=f_tot, rf=None, stride=None, method_deconvolution='oasis', gnb=params_movie['gnb'],
                        low_rank_background=params_movie['low_rank_background'],
                        update_background_components=params_movie['update_background_components'], check_nan = True)
            
#    else:
#        cnm = cnmf.CNMF(n_processes=1, k=A_tot.shape, gSig=params_movie['gSig'], merge_thresh=params_movie['merge_thresh'],
#                        p=params_movie['p'], dview=dview, Ain=A_tot, Cin=C_tot, b_in = b_tot, skip_refinement=True, only_init_patch=False,
#                        f_in=f_tot, rf=None, stride=None, method_deconvolution='oasis', gnb = params_movie['gnb'],
#                        low_rank_background=params_movie['low_rank_background'],
#                        update_background_components = params_movie['update_background_components'], check_nan = True)
#        
        cnm = cnm.fit(images)
    
    #cnm_kept = cnmf.CNMF(n_processes=1, k=A_tot_kept.shape, gSig=params_movie['gSig'], merge_thresh=params_movie['merge_thresh'],
    #                p=params_movie['p'], dview=dview, Ain=A_tot_kept, Cin=C_tot_kept, b_in = b_tot,
    #                f_in=f_tot, rf=None, stride=None, method_deconvolution='oasis', gnb = params_movie['gnb'],
    #                low_rank_background = params_movie['low_rank_background'], update_background_components = params_movie['update_background_components'], check_nan = True)
    #
    #cnm_kept = cnm_kept.fit(images)

    #%% Replot:
    pl.figure()
    if display_average is True:
        crd = plot_contours(cnm.A, Av, thr=params_display['thr_plot'])
    else:
        crd = plot_contours(cnm.A, Cn, thr=params_display['thr_plot'])
    
    pl.savefig(os.path.join(nmf_fig_dir, '%s_contours_iter2_%s.png' % (curr_file, roi_id)))

    #%%
    A, C, b, f, YrA, sn = cnm.A, cnm.C, cnm.b, cnm.f, cnm.YrA, cnm.sn
    
    # To save deconvolved:
    S = cnm.S

    # %% again recheck quality of components, stricter criteria
    final_frate = params_movie['final_frate']
    r_values_min = params_movie['r_values_min_full']  # threshold on space consistency
    fitness_min = params_movie['fitness_min_full']  # threshold on time variability
    # threshold on time variability (if nonsparse activity)
    fitness_delta_min = params_movie['fitness_delta_min_full']
    Npeaks = params_movie['Npeaks']
    traces = C + YrA
    idx_components, idx_components_bad, fitness_raw, fitness_delta, r_values = estimate_components_quality(
        traces, Y, A, C, b, f, final_frate=final_frate, Npeaks=Npeaks, r_values_min=r_values_min, fitness_min=fitness_min,
        fitness_delta_min=fitness_delta_min, return_all=True)
    print(' ***** ')
    print((len(traces)))
    print((len(idx_components)))
    
    # %% save results
    Cdf = extract_DF_F(Yr=Yr, A=A, C=C, bl=cnm.bl)
    np.savez(os.path.join(nmf_output_dir, os.path.split(curr_mmap)[1][:-4] + 'results_analysis.npz'), Cn=Cn,
             A=A, Cdf = Cdf, C=C, b=b, f=f, YrA=YrA, sn=sn, d1=d1, d2=d2, idx_components=idx_components,
             idx_components_bad=idx_components_bad,
             fitness_raw=fitness_raw, fitness_delta=fitness_delta, r_values=r_values, Av=Av, S=S)
            

    print("FINAL N COMPONENTS:", A.shape[1])

    # %%
    # TODO: show screenshot 14
    if display_average is True:
        pl.subplot(1, 2, 1)
        crd = plot_contours(A.tocsc()[:, idx_components], Av, thr=params_display['thr_plot'])
        pl.title('kept')
        pl.subplot(1, 2, 2)
        crd = plot_contours(A.tocsc()[:, idx_components_bad], Av, thr=params_display['thr_plot'])
        pl.title('bad')
    else:
        pl.subplot(1, 2, 1)
        crd = plot_contours(A.tocsc()[:, idx_components], Cn, thr=params_display['thr_plot'])
        pl.title('kept')
        pl.subplot(1, 2, 2)
        crd = plot_contours(A.tocsc()[:, idx_components_bad], Cn, thr=params_display['thr_plot'])
        pl.title('bad')
    
    pl.savefig(os.path.join(nmf_fig_dir, '%s_contours_kept_%s.png' % (curr_file, roi_id)))
    pl.close()

#%%

    #YrA = np.array(A.T * np.matrix(Yr) - (A.T * np.matrix(b[:, np.newaxis])) * np.matrix(f[np.newaxis]) - (A.T.dot(A)) * np.matrix(C) + C)

    # %%
    # TODO: needinfo
    if inspect_components is True:
        if display_average is True:
            view_patches_bar(Yr, scipy.sparse.coo_matrix(A.tocsc()[:, idx_components]), C[idx_components, :], b, f, dims[0], dims[1],
                         YrA=YrA[idx_components, :], img=Av)
        else:
            view_patches_bar(Yr, scipy.sparse.coo_matrix(A.tocsc()[:, idx_components]), C[idx_components, :], b, f, dims[0], dims[1],
                         YrA=YrA[idx_components, :], img=Cn)
            
        # %%
        if display_average is True:
            view_patches_bar(Yr, scipy.sparse.coo_matrix(A.tocsc()[:, idx_components_bad]), C[idx_components_bad, :], b, f, dims[0],
                         dims[1], YrA=YrA[idx_components_bad, :], img=Av)
        else:
            view_patches_bar(Yr, scipy.sparse.coo_matrix(A.tocsc()[:, idx_components_bad]), C[idx_components_bad, :], b, f, dims[0],
                         dims[1], YrA=YrA[idx_components_bad, :], img=Cn)
        
    # %% reconstruct denoised movie
    denoised = cm.movie(A.dot(C) + b.dot(f)).reshape(dims + (-1,), order='F').transpose([2, 0, 1])
    
    #%% save denoised movie:
        
    denoised.save(os.path.join(nmf_mov_dir, '%s_denoisedmov.tif' % curr_file))

    #%% background only 
    background = cm.movie(b.dot(f)).reshape(dims + (-1,), order='F').transpose([2, 0, 1])
    #denoised.play(gain=2, offset=0, fr=50, magnification=4)
    #%% save denoised movie:
        
    background.save(os.path.join(nmf_mov_dir, '%s_backgroundmov.tif' % curr_file))


    # %% reconstruct denoised movie without background
    denoised = cm.movie(A.dot(C)).reshape(dims + (-1,), order='F').transpose([2, 0, 1])
    
    # %%
    # TODO: show screenshot 16
    #denoised.play(gain=10, offset=0, fr=100, magnification=2)
    
    #%% save denoised movie:
        
    denoised.save(os.path.join(nmf_mov_dir, '%s_denoised_nobackground_mov.tif' % curr_file))

    #%% show background(s)
    BB  = cm.movie(b.reshape(dims+(-1,), order = 'F').transpose(2,0,1))
    #BB.play(gain=2, offset=0, fr=2, magnification=4)
    pl.figure()
    BB.zproject()
    pl.savefig(os.path.join(nmf_mov_dir, '%s_background_project.png' % curr_file))

    #%% Save params:
    #import pprint
    #pp = pprint.PrettyPrinter(indent=4)
    
    # Create paramsdir:
    nmf_params_dir = os.path.join(roi_dir, 'nmf_params')
    if not os.path.exists(nmf_params_dir):
        os.mkdir(nmf_params_dir)
    
    params_fn_base = 'nmfopts_%s_%s' % (roi_id, curr_file)
    save_object(cnm.options, os.path.join(nmf_params_dir, '%s.pkl' % params_fn_base))
    
    # Reformat 2D arrays into list for json viewing:
    for k in cnm.options.keys():
        for i in cnm.options[k].keys():
            if type(cnm.options[k][i])==np.ndarray:
                cnm.options[k][i] = cnm.options[k][i].tolist()
                
    with open(os.path.join(nmf_params_dir, '%s.json' % params_fn_base), 'w') as f:
        #pprint.pprint(cnm.options, stream=f)
        #fwriteKeyVals(cnm.options, f, indent=4)
        json.dump(cnm.options, f, indent=4, sort_keys=True) #, separators=(',','\n'))

    if curr_file == reference_file:
        with open(os.path.join(roi_dir, '%s.json' % params_fn_base), 'w') as f:
            json.dump(cnm.options, f, indent=4, sort_keys=True)

# %% STOP CLUSTER and clean up log files
# TODO: todocument
cm.stop_server()

log_files = glob.glob('*_LOG_*')
for log_file in log_files:
    os.remove(log_file)

