#!/usr/bin/env python
# coding: utf-8

# In[1]:

import matplotlib
matplotlib.use('agg')

import cv2
import glob
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import h5py

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


import pylab as pl
from functools import partial
import tifffile as tf
import multiprocessing as mp
import json
import time
import re
import optparse
import sys


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]



from caiman.source_extraction.cnmf.initialization import downscale as cmdownscale



def extract_options(options):

    parser = optparse.OptionParser()

    # PATH opts:
    parser.add_option('-D', '--root', action='store', dest='rootdir', default='/n/coxfs01/2p-data', help='data root dir (root project dir containing all animalids) [default: /n/coxfs01/2pdata]')
    parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')
    parser.add_option('-S', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID')
    parser.add_option('-A', '--acq', action='store', dest='fov', default='FOV1_zoom2p0x', help="acquisition folder (ex: 'FOV1_zoom2p0x') [default: FOV1_zoom2p0x]")
    parser.add_option('-E', '--exp', action='store', dest='experiment', default='', help="Name of experiment (stimulus type), e.g., rfs")
    parser.add_option('-t', '--traceid', action='store', dest='traceid', default='traces001', help="Traceid from which to get seeded rois (default: traces001)")


    parser.add_option('-n', '--nproc', action="store",
                      dest="n_processes", default=2, help="N processes [default: 1]")
    parser.add_option('-d', '--downsample', action="store",
                      dest="ds_factor", default=5, help="Downsample factor (int, default: 5)")

    parser.add_option('--destdir', action="store",
                      dest="destdir", default='/n/scratchlfs/cox_lab/julianarhee/downsampled', help="output dir for movie files [default: /n/scratchlfs/cox_lab/julianarhee/downsampled]")
    parser.add_option('--plot', action='store_true', dest='plot_rois', default=False, help="set to plot results of each roi's analysis")
    parser.add_option('--processed', action='store_false', dest='use_raw', default=True, help="set to downsample on non-raw source")

    parser.add_option('--new', action='store_true', dest='create_new', default=False, help="Set to downsample and motion correct anew")
    parser.add_option('--mmap', action='store', dest='mmap_prefix', default='Yr', help="Prefix for sourced memmap/mc files (default: Yr)")
    parser.add_option('--prefix', action='store', dest='save_prefix', default=None, help="Prefix for saveing caiman results (default: 'seeded-/patch-Yr')")

    parser.add_option('--seed', action='store_true', dest='seed_rois', default=False, help="Set flag to seed ROIs with manual (must provide traceid)")

    (options, args) = parser.parse_args(options)

    return options

def caiman_params(fnames, kwargs=None):
    # dataset dependent parameters
    fr = 44.65                             # imaging rate in frames per second
    decay_time = 0.4                    # length of a typical transient in seconds

    # motion correction parameters
    strides = (96, 96)          # start a new patch for pw-rigid motion correction every x pixels
    overlaps = (48, 48)         # overlap between pathes (size of patch strides+overlaps)
    max_shifts = (6,6)          # maximum allowed rigid shifts (in pixels)
    max_deviation_rigid = 3     # maximum shifts deviation allowed for patch with respect to rigid shifts
    pw_rigid = False             # flag for performing non-rigid motion correction

    # parameters for source extraction and deconvolution
    p = 2                       # order of the autoregressive system
    gnb = 2                     # number of global background components
    merge_thr = 0.85            # merging threshold, max correlation allowed
    rf = 25                     # half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
    stride_cnmf = 12             # amount of overlap between the patches in pixels
    K = 8                       # number of components per patch
    gSig = [2, 2]               # expected half size of neurons in pixels
    method_init = 'greedy_roi'  # initialization method (if analyzing dendritic data using 'sparse_nmf')
    ssub = 1                    # spatial subsampling during initialization
    tsub = 1                    # temporal subsampling during intialization

    # parameters for component evaluation
    min_SNR = 2.0               # signal to noise ratio for accepting a component
    rval_thr = 0.85              # space correlation threshold for accepting a component
    cnn_thr = 0.99              # threshold for CNN based classifier
    cnn_lowest = 0.1 # neurons with cnn probability lower than this value are rejected

    opts_dict = {'fnames': fnames,
                'fr': fr,
                'decay_time': decay_time,
                'strides': strides,
                'overlaps': overlaps,
                'max_shifts': max_shifts,
                'max_deviation_rigid': max_deviation_rigid,
                'pw_rigid': pw_rigid,
                'p': p,
                'nb': gnb,
                'rf': rf,
                'K': K, 
                'gSig': gSig,
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

   
    if kwargs is not None:
        for k, v in kwargs.iteritems():
            opts_dict.update({k: v})
            print("... updating opt %s to value %s" % (k, str(v)))

    if opts_dict['ssub'] != 1:
        print("Updating params for spatial ds factor: %i" % opts_dict['ssub'])
        ssub_val = float(opts_dict['ssub'])
        gsig_val = float(opts_dict['gSig'][0])
        rf_val = float(opts_dict['rf'])

        strides = (int(round(opts_dict['strides'][0]/ssub_val)), int(round(opts_dict['strides'][1]/ssub_val)))
        overlaps = (int(round(opts_dict['overlaps'][0]/ssub_val)), int(round(opts_dict['overlaps'][1]/ssub_val)))
        gSig = (int(round(gsig_val/ssub_val)), int(round(gsig_val/ssub_val)))
        rf = (int(round(rf_val/ssub_val)), int(round(rf_val/ssub_val)))
        strid_cnmf = int(round(opts_dict['stride']))

        opts_dict.update({'strides': strides, 
                          'overlaps': overlaps,
                          'gSig': gSig,
                          'rf': rf,
                          'stride': stride_cnmf})

    opts_dict.update({'p_ssub': opts_dict['ssub'],
                      'p_tsub': opts_dict['tsub']}) 


    opts = params.CNMFParams(params_dict=opts_dict)

    return opts

def save_mc_results(results_dir, prefix='Yr'):
    np.savez(os.path.join(results_dir, '%s_mc-rigid.npz' % prefix),
            mc=mc,
            fname=mc.fname, max_shifts=mc.max_shifts, min_mov=mc.min_mov,
            border_nan=mc.border_nan,
            fname_tot_rig=mc.fname_tot_rig,
            total_template_rig=mc.total_template_rig,
            templates_rig=mc.templates_rig,
            shifts_rig=mc.shifts_rig,
            mmap_file=mc.mmap_file,
            border_to_0=mc.border_to_0)
    print("--- saved MC results: %s" % os.path.join(results_dir, '%s_mc-rigid.npz' % prefix))
 
def load_mc_results(results_dir, prefix='Yr'):
    try:
        mc_results = np.load(os.path.join(results_dir, '%s_mc-rigid.npz' % prefix))
        mc = mc_results[mc] 
#            fname=mc.fname, max_shifts=mc.max_shifts, min_mov=mc.min_mov,
#            border_nan=mc.border_nan,
#            fname_tot_rig=mc.fname_tot_rig,
#            total_template_rig=mc.total_template_rig,
#            templates_rig=mc.templates_rig,
#            shifts_rig=mc.shifts_rig,
#            mmap_file=mc.mmap_file,
#            border_to_0=mc.border_to_0)
    except Exception as e:
        return None

    return mc 

def get_file_paths(results_dir, mm_prefix='Yr'):
    try:
        mparams_fpath = os.path.join(results_dir, '%s_memmap-params.json' % mm_prefix)
        print("Loading memmap params...")
        with open(mparams_fpath, 'r') as f:
            mparams = json.load(f)
        fnames = mparams['fnames']
    except Exception as e:
        print("Unable to load memmap params, trying alt.")
        try:
            fnames = []
            if 'downsample' in mm_prefix:
                print("checking scratch")
                fovdir = results_dir.split('/caiman_results/')[0]
                fov = os.path.split(fovdir)[-1]
                sessiondir = os.path.split(fovdir)[0]
                session = os.path.split(sessiondir)[-1]
                animalid = os.path.split(os.path.split(sessiondir)[0])[-1]
                print("Animal: %s, Fov: %s, Session: %s" % (animalid, fov, session))
                fnames = sorted(glob.glob(os.path.join('/n/scratchlfs/cox_lab/julianarhee/downsampled/*%s*/*.tif' % mm_prefix)), key=natural_keys)
                print(fnames[0:5]) 
            if len(fnames)==0 and 'downsample' not in mm_prefix: 
                dpath = glob.glob(os.path.join(results_dir, 'memmap', '*%s*.npz' % mm_prefix))[0]#) [0]) 
                minfo = np.load(dpath)
                fnames = sorted(list(minfo['mmap_fnames']))
        except Exception as e:
            print("unable to load file names.")
            return None
    
    return fnames #fnames = mparams['fnames']


#def get_full_memmap_path(results_dir, mm_prefix='Yr'):
#    print("Getting full mmap path for prefix: %s" % mm_prefix)
#    print("-- dir: %s" % results_dir)
#    print(glob.glob(os.path.join(results_dir, 'memmap', '*%s*.mmap')))
#    fname_new = glob.glob(os.path.join(results_dir, 'memmap', '*%s*_d*_.mmap' % mm_prefix))[0]
#    mm_prefix = os.path.splitext(os.path.split(fname_new)[-1])[0].split('_d1_')[0]
#    print("CORRECTED PREFIX: %s" % mm_prefix)
#    return fname_new, mm_prefix

def get_full_memmap_path(results_dir, framestr='order_C_frames', prefix='Yr'):
    print("Getting full mmap path for prefix: %s" % prefix)
    print("-- dir: %s" % results_dir)
    print(glob.glob(os.path.join(results_dir, 'memmap', '*%s*.mmap'))) 
    try:
        fname_new = glob.glob(os.path.join(results_dir, 'memmap', '*%s*_d*%s*_.mmap' % (prefix, framestr)))
        if len(fname_new) > 1:
            nframes = max([int(i.split('_')[-2]) for i in fname_new])
            framestr = '_frames_%i_' % nframes
            fname_new = glob.glob(os.path.join(results_dir, 'memmap', '*%s*_d*%s*_.mmap' % (prefix, framestr)))[0]
        else:
            assert len(fname_new)==1, "Unique fname_new not found: %s" % str(fname_new)
            fname_new = fname_new[0] 
        mm_prefix = os.path.splitext(os.path.split(fname_new)[-1])[0].split('_d1_')[0]
        print("CORRECTED PREFIX: %s" % mm_prefix)
    except Exception as e:
        print(e)
        return None
    return fname_new, mm_prefix


def get_roiid_from_traceid(animalid, session, fov, run_type=None, traceid='traces001', rootdir='/n/coxfs01/2p-data'):
    
    if run_type is not None:
        if int(session) < 20190511 and run_type == 'gratings':
            a_traceid_dict = glob.glob(os.path.join(rootdir, animalid, session, fov, '*run*', 'traces', 'traceids*.json'))[0]
        else:
            a_traceid_dict = glob.glob(os.path.join(rootdir, animalid, session, fov, '*%s*' % run_type, 'traces', 'traceids*.json'))[0]
    else:
        a_traceid_dict = glob.glob(os.path.join(rootdir, animalid, session, fov, '*run*', 'traces', 'traceids*.json'))[0]
    with open(a_traceid_dict, 'r') as f:
        tracedict = json.load(f)
    
    tid = tracedict[traceid]
    roiid = tid['PARAMS']['roi_id']
    
    return roiid


def load_roi_masks(animalid, session, fov, rois=None, rootdir='/n/coxfs01/2p-data'):
    masks=None; zimg=None;
    mask_fpath = glob.glob(os.path.join(rootdir, animalid, session, 'ROIs', '%s*' % rois, 'masks.hdf5'))[0]
    try:
        mfile = h5py.File(mask_fpath, 'r')

        # Load and reshape masks
        fkey = list(mfile.keys())[0]
        masks = mfile[fkey]['masks']['Slice01'][:] #.T
        #print(masks.shape)
        #mfile[mfile.keys()[0]].keys()

        zimg = mfile[fkey]['zproj_img']['Slice01'][:] #.T
        zimg.shape
    except Exception as e:
        print("error loading masks")
    finally:
        mfile.close()
        
    return masks, zimg

# def reshape_and_binarize_masks(masks):
#     # Binarze and reshape:
#     nrois, d1, d2 = masks.shape
#     Ain = np.reshape(masks, (nrois, d1*d2))
#     Ain[Ain>0] = 1
#     Ain = Ain.astype(bool).T 
    
#     return Ain

def reshape_and_binarize_masks(masks):
    # Binarze and reshape:
    nrois, d1, d2 = masks.shape
    #masks2 = np.swapaxes(masks, 1, 2)
    Ain = np.reshape(masks, (nrois, d1*d2))
    Ain[Ain>0] = 1
    Ain = Ain.astype(bool).T 
    
    return Ain


def run_cnmf_seeded(animalid, session, fov, experiment='', traceid='traces001', rootdir='/n/coxfs01/2p-data', mm_prefix='Yr', prefix=None, n_processes=1, opts_kws=None):

    # Load manual ROIs and format
    print("Getting seeds...")
    roiid = get_roiid_from_traceid(animalid, session, fov, run_type=experiment, traceid=traceid)
    masks, zimg = load_roi_masks(animalid, session, fov, rois=roiid)
    uimg = zimg.T
    Ain = reshape_and_binarize_masks(masks)

    # Load memmapped file(s)
    fovdir = glob.glob(os.path.join(rootdir, animalid, session, fov))[0]
    results_dir = os.path.join(fovdir, 'caiman_results', experiment)

    fname_tot, mm_prefix = get_full_memmap_path(results_dir, prefix=mm_prefix)
    print("Extracting CNMF from: %s" % fname_tot)

    # Load data
    Yr, dims, T = cm.load_memmap(fname_tot)
    images = np.reshape(Yr.T, [T] + list(dims), order='F') #np.reshape(Yr, dims + (T,), order='F')
    print("Loaded data:", images.shape)

    # Create opts for cnmf
    print("Preparing for CNMF extraction...") 
    fnames = get_file_paths(results_dir, mm_prefix=mm_prefix) 
    print("--> got %i files for extraction" % len(fnames))
    
#    if 'fov' in fnames[0]:
#        fnames = sorted(glob.glob(os.path.join(rootdir, animalid, session, fov, '*%s*' % experiment,
#                                'raw*', '*.tif')))
        
    opts = caiman_params(fnames, opts_kws)
    if prefix is None:
         prefix = mm_prefix
    prefix = 'seeded_%s' % prefix

    #%% start a cluster for parallel processing 
    #(if a cluster already exists it will be closed and a new session will be opened)
    if 'dview' in locals():
        cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=n_processes, single_thread=False)
    print("--- running on %i processes ---" % n_processes)
    print("--- dview: ", dview)
    #dview=None 
    
    # Reset default patch params to run on full
    rf = None          # half-size of the patches in pixels. `None` when seeded CNMF is used.
    only_init = False  # has to be `False` when seeded CNMF is used
    #gSig = (2, 2)      # expected half size of neurons in pixels, v important for proper component detection

    # params object
    opts_dict = {'fnames': fnames,
                'decay_time': 0.4,
                'p': 2,
                'nb': 2,
                'rf': rf,
                'only_init': only_init,
                'merge_thr': 0.85,
                'n_pixels_per_process': 100}

    opts.change_params(opts_dict)

    # Run cnmf
    print("Extracting...")
    start_t = time.time()

    cnm = cnmf.CNMF(n_processes, params=opts, dview=dview, Ain=Ain)
    cnm.fit(images)
    end_t = time.time() - start_t
    print("--> Elapsed time: {0:.2f}sec".format(end_t))
    print("A:", cnm.estimates.A.shape)

    #Cn = cm.local_correlations(images.transpose(1,2,0))
    #fig = pl.figure()
    #pl.imshow(Cn, cmap='gray')
    #pl.savefig(os.path.join(results_dir, '%s_Cn.png' % prefix))
    #pl.close()
   
    # Evaluate components 
    print("Evaluatnig components...")
    # parameters for component evaluation
    min_SNR = 1.5               # signal to noise ratio for accepting a component
    rval_thr = 0.85              # space correlation threshold for accepting a component
    min_cnn_thr = 0.99          # threshold for CNN based classifier
    cnn_lowest = 0.05           # neurons with cnn probability lower than this value are rejected
    #cnm_seeded.estimates.restore_discarded_components()
    cnm.params.set('quality', {'min_SNR': min_SNR,
                               'rval_thr': rval_thr,
                               'use_cnn': True,
                               'min_cnn_thr': min_cnn_thr,
                               'cnn_lowest': cnn_lowest})

    #%% COMPONENT EVALUATION
    # the components are evaluated in three ways:
    #   a) the shape of each component must be correlated with the data
    #   b) a minimum peak SNR is required over the length of a transient
    #   c) each shape passes a CNN based classifier
    start_t = time.time()
    cnm.estimates.evaluate_components(images, cnm.params, dview=dview)
    end_t = time.time() - start_t
    print("--> evaluation - Elapsed time: {0:.2f}sec".format(end_t))

    #%% Extract DF/F values
    print("Extracting df/f...")
    quantileMin = 10 # 8
    frames_window_sec = 30.
    if 'downsample' in mm_prefix:
        ds_factor = float(mm_prefix.split('downsample-')[-1].split('-')[0]) #opts.init['tsub']
    else:
        ds_factor = float(opts.init['tsub'])
    fr = float(opts.data['fr'])
    frames_window = int(round(frames_window_sec * (fr / ds_factor))) # 250
    dff_params = {'quantileMin': quantileMin,
                  'frames_window_sec': frames_window_sec,
                  'ds_factor': ds_factor,
                  'fr': fr,
                  'frames_window': frames_window,
                  'source': fname_tot}

    with open(os.path.join(results_dir, 'seeded_%s_processing-params.json' % prefix), 'w') as f:
        json.dump(dff_params, f, indent=4)

    start_t = time.time()
    cnm.estimates.detrend_df_f(quantileMin=quantileMin, frames_window=frames_window)
    end_t = time.time() - start_t
    print("--> Elapsed time: {0:.2f}sec".format(end_t))

    # save results
    save_results = True
    if save_results:
        cnm.save(os.path.join(results_dir, 'seeded_%s_results.hdf5' % prefix))
    print("Saved results: %s" % os.path.join(results_dir, 'seeded_%s_results.hdf5' % prefix))

    print("******DONE!**********")

def run_cnmf_patches(animalid, session, fov, experiment='', traceid='traces001', rootdir='/n/coxfs01/2p-data', mm_prefix='Yr', prefix=None, n_processes=1, opts_kws=None):

    # Load memmapped file(s)
    fovdir = glob.glob(os.path.join(rootdir, animalid, session, fov))[0]
    results_dir = os.path.join(fovdir, 'caiman_results', experiment)
    try:
        fname_tot, mm_prefix = get_full_memmap_path(results_dir, mm_prefix=mm_prefix)
    except Exception as e:
        print("Unable to find .mmap.  Creating new.")
        create_memmap()
    print("Extracting CNMF from: %s" % fname_tot)

    # Load data
    Yr, dims, T = cm.load_memmap(fname_tot)
    images = np.reshape(Yr.T, [T] + list(dims), order='F') #np.reshape(Yr, dims + (T,), order='F')
    print("Loaded data:", images.shape)

    # Create opts for cnmf
    print("Preparing for CNMF extraction...") 
    fnames = get_file_paths(results_dir, mm_prefix=mm_prefix) 
    print("--> got %i files for extraction" % len(fnames))
      
    opts = caiman_params(fnames, opts_kws)
       
    if prefix is None:
         prefix = mm_prefix
    prefix = 'patches_%s' % prefix

    #%% start a cluster for parallel processing 
    #(if a cluster already exists it will be closed and a new session will be opened)
    if 'dview' in locals():
        cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=n_processes, single_thread=False)
    print("--- running on %i processes ---" % n_processes)
    print("--- dview: ", dview)
    #dview=None 

    # First extract spatial and temporal components on patches and combine them
    # for this step deconvolution is turned off (p=0)
    #opts.change_params({'p': 0})

    # Run cnmf
    print("Extracting from patches...")
    start_t = time.time() 
    cnm = cnmf.CNMF(n_processes, params=opts, dview=dview)
    cnm = cnm.fit(images)
    end_t = time.time() - start_t
    print("--> patches - Elapsed time: {0:.2f}sec".format(end_t))

    #%% RE-RUN seeded CNMF on accepted patches to refine and perform deconvolution 
    print("Refitting on accepted patches")
    start_t = time.time() 
    cnm.params.change_params({'p': 2})
    cnm = cnm.refit(images, dview=dview)
    end_t = time.time() - start_t
    print("--> refit - Elapsed time: {0:.2f}sec".format(end_t))

    #print("Getting local correlations...")
    #Cn = cm.local_correlations(images.transpose(1,2,0))
    #fig = pl.figure()
    #pl.imshow(Cn, cmap='gray')
    #pl.savefig(os.path.join(results_dir, '%s_Cn.png' % prefix))
    #pl.close()
 
 
    # Evaluate components 
    print("Evaluatnig components...")
    # parameters for component evaluation
    min_SNR = 1.5               # signal to noise ratio for accepting a component
    rval_thr = 0.85              # space correlation threshold for accepting a component
    min_cnn_thr = 0.99          # threshold for CNN based classifier
    cnn_lowest = 0.05           # neurons with cnn probability lower than this value are rejected
    #cnm_seeded.estimates.restore_discarded_components()
    cnm.params.set('quality', {'min_SNR': min_SNR,
                               'rval_thr': rval_thr,
                               'use_cnn': True,
                               'min_cnn_thr': min_cnn_thr,
                               'cnn_lowest': cnn_lowest})

    #%% COMPONENT EVALUATION
    # the components are evaluated in three ways:
    #   a) the shape of each component must be correlated with the data
    #   b) a minimum peak SNR is required over the length of a transient
    #   c) each shape passes a CNN based classifier
    start_t = time.time()
    cnm.estimates.evaluate_components(images, cnm.params, dview=dview)
    end_t = time.time() - start_t
    print("--> evaluation - Elapsed time: {0:.2f}sec".format(end_t))
    cnm.estimates.select_components(use_object=True)
    print("Discarding %i of %i initially selected components." % (len(cnm.estimates.idx_components), cnm.estimates.A.shape[-1]))

    #%% Extract DF/F values
    print("Extracting df/f...")
    start_t = time.time()
    quantileMin = 10 # 8
    frames_window_sec = 30.
    if 'downsample' in mm_prefix:
        ds_factor = float(mm_prefix.split('downsample-')[-1].split('-')[0]) #opts.init['tsub']
        #ds_factor = float(mm_prefix.split('downsample-')[-1]) #opts.init['tsub']
    else:
        ds_factor = float(opts.init['tsub'])
    fr = float(opts.data['fr'])
    frames_window = int(round(frames_window_sec * (fr / ds_factor))) # 250
    dff_params = {'quantileMin': quantileMin,
                  'frames_window_sec': frames_window_sec,
                  'ds_factor': ds_factor,
                  'fr': fr,
                  'frames_window': frames_window,
                  'source': fname_tot}

    with open(os.path.join(results_dir, 'patches_%s_processing-params.json' % prefix), 'w') as f:
        json.dump(dff_params, f, indent=4)
    cnm.estimates.detrend_df_f(quantileMin=quantileMin, frames_window=frames_window)
    end_t = time.time() - start_t
    print("--> dF_F - Elapsed time: {0:.2f}sec".format(end_t))

    # save results
    save_results = True
    if save_results:
        cnm.save(os.path.join(results_dir, 'patches_%s_results.hdf5' % prefix))
    print("Saved results 2: %s" % os.path.join(results_dir, 'patches_%s_results.hdf5' % prefix))


    print("******DONE!**********")


def main(options):
    opts = extract_options(options) 
    rootdir = opts.rootdir #'/n/coxfs01/2p-data'
    animalid = opts.animalid #'JC084'
    session = opts.session #'20190525' #'20190505_JC083'
    fov = opts.fov
    experiment = opts.experiment
    ds_factor = int(opts.ds_factor)
    destdir = opts.destdir
    use_raw = opts.use_raw
    n_processes = int(opts.n_processes) 
    create_new = opts.create_new
    mm_prefix = opts.mmap_prefix
    prefix = opts.save_prefix
    traceid=opts.traceid
    seed_rois = opts.seed_rois

    cparams = [a for a in options if 'c_' in a]
    if len(cparams) > 0:
        if ',' in cparams and len(cparams)==1:
            cparams = cparams[0].split(',')
        c_args = dict([a[2:].split('=', maxsplit=1) for a in cparams])
    else:
        c_args = None

    print(c_args)



    if seed_rois:
        run_cnmf_seeded(animalid, session, fov, experiment=experiment, traceid=traceid, mm_prefix=mm_prefix,
                        rootdir=rootdir, prefix=prefix, n_processes=n_processes, opts_kws=c_args)
    else:
        run_cnmf_patches(animalid, session, fov, experiment=experiment, traceid=traceid, mm_prefix=mm_prefix,
                         rootdir=rootdir, prefix=prefix, n_processes=n_processes, opts_kws=c_args)


if __name__=='__main__':
    main(sys.argv[1:])


