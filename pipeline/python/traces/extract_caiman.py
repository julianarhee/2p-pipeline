#!/usr/bin/env python
# coding: utf-8

# In[1]:



import cv2
import glob
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import time

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

    parser.add_option('-n', '--nproc', action="store",
                      dest="n_processes", default=2, help="N processes [default: 1]")
    parser.add_option('-d', '--downsample', action="store",
                      dest="ds_factor", default=5, help="Downsample factor (int, default: 5)")

    parser.add_option('--destdir', action="store",
                      dest="destdir", default='/n/scratchlfs/cox_lab/julianarhee/downsampled', help="output dir for movie files [default: /n/scratchlfs/cox_lab/julianarhee/downsampled]")
    parser.add_option('--plot', action='store_true', dest='plot_rois', default=False, help="set to plot results of each roi's analysis")
    parser.add_option('--processed', action='store_false', dest='use_raw', default=True, help="set to downsample on non-raw source")

    parser.add_option('--new', action='store_true', dest='create_new', default=False, help="Set to downsample and motion correct anew")
    parser.add_option('--prefix', action='store', dest='prefix', default='Yr', help="Prefix for sourced memmap/mc files (default: Yr)")


    (options, args) = parser.parse_args(options)

    return options


def caiman_params(fnames):
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
    p = 2                       # order of the autoregressive system
    gnb = 2                     # number of global background components
    merge_thr = 0.85            # merging threshold, max correlation allowed
    rf = 25                     # half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
    stride_cnmf = 12             # amount of overlap between the patches in pixels
    K = 8                      # number of components per patch
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
    return opts_dict

def save_mc_results(results_dir, prefix='Yr'):
    np.savez(os.path.join(results_dir, 'mc_rigid.npz'),
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

def get_original_tifs(results_dir):
    mparams_fpath = os.path.join(results_dir, 'memmap', 'memmap_params.json')
    with open(mparams, fpath, 'r') as f:
        mparams = json.load(f)

    return mparams #fnames = mparams['fnames']

def get_full_memmap_path(results_dir, prefix='Yr'):
    fname_new = glob.glob(os.path.join(results_dir, 'memmap', '%s_d*_.mmap' % prefix))[0]
    return fname_new


def run_seeded_cnmf(animalid, session, fov, experiment='', rootdir='/n/coxfs01/2p-data', prefix='gratings'):

    # Load manual ROIs and format
    print("Getting seeds...")
    roiid = get_roiid_from_traceid(animalid, session, fov, run_type=None, traceid=traceid)
    masks, zimg = load_roi_masks(animalid, session, fov, rois=roiid)
    Ain = reshape_and_binarize_masks(masks)

    # Load memmapped file(s)
    fovdir = glob.glob(os.path.join(rootdir, animalid, session, fov))[0]
    results_dir = os.path.join(fovdir, 'caiman_results', experiment)
    fname_tot = get_full_memmap_path(results_dir, prefix=prefix)
    print("Extracting CNMF from: %s" % fname_tot)

    # Load data
    Yr, dims, T = cm.load_memmap(fname_tot)
    images = np.reshape(Yr, dims + (T,), order='F')
    print("Loaded data:", images.shape)

    # Create opts for cnmf
    print("Preparing for CNMF extraction...") 
    memparams = get_original_tifs(results_dir) 
    opts = caiman_params(memparams['fnames'])

    #%% start a cluster for parallel processing 
    #(if a cluster already exists it will be closed and a new session will be opened)
    if 'dview' in locals():
        cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=2, single_thread=False)


    # Reset default patch params to run on full
    rf = None          # half-size of the patches in pixels. `None` when seeded CNMF is used.
    only_init = False  # has to be `False` when seeded CNMF is used
    gSig = (2, 2)      # expected half size of neurons in pixels, v important for proper component detection

    # params object
    opts_dict = {'fnames': memparams['fnames'],
                'decay_time': 0.4,
                'p': 2,
                'nb': 2,
                'rf': rf,
                'only_init': only_init,
                'gSig': gSig,
                'ssub': 1,
                'tsub': 1,
                'merge_thr': 0.85}

    opts.change_params(opts_dict)

    # Run cnmf
    print("Extracting...")
    start_t = time.time()

    cnm = cnmf.CNMF(n_processes, params=opts, dview=dview, Ain=Ain)
    cnm.fit(images)

    end_t = time.time()
    print("--> Elapsed time: {0:.2f}sec".format(end_t))

    #%% Extract DF/F values
    print("Extracting df/f...")
    quantileMin = 10 # 8
    frames_window_sec = 30.
    ds_factor = opts.init['tsub']
    frames_window = frames_window_sec * (fr * ds_factor) # 250

    dff_params = {'quantileMin': quantileMin,
                  'frames_window_sec': frames_window_sec,
                  'ds_factor': ds_factor,
                  'fr': fr,
                  'frames_window': frames_window,
                  'source': fname_tot}
    with open(os.path.join(results_dir, 'processing_params.json'), 'w') as f:
        json.dump(dff_params, f, indent=4)

    start_t = time.time()
    cnm.estimates.detrend_df_f(quantileMin=quantileMin, frames_window=frames_window)
    end_t = time.time()
    print("--> Elapsed time: {0:.2f}sec".format(end_t))

    # save results
    save_results = True
    if save_results:
        cnm.save(os.path.join(results_dir, 'cnm_analysis_results.hdf5'))
    print("Saved results: %s" % os.path.join(results_dir, 'cnm_analysis_results.hdf5'))


    print("******DONE!**********")


def get_roiid_from_traceid(animalid, session, fov, run_type=None, traceid='traces001', rootdir='/n/coxfs01/2p-data'):
    
    if run_type is not None:
        a_traceid_dict = glob.glob(os.path.join(rootdir, animalid, session, fov, '*%s*' % run_type, 'traces', 'traceids*.json'))[0]
    else:
        a_traceid_dict = glob.glob(os.path.join(rootdir, animalid, session, fov, '*run*', 'traces', 'traceids*.json'))[0]
    with open(a_traceid_dict, 'r') as f:
        tracedict = json.load(f)
    
    tid = tracedict[traceid]
    roiid = tid['PARAMS']['roi_id']
    
    return roiid


def load_roi_masks(animalid, session, fov, rois=None, rootdir='/n/coxfs01/2p-data'):
    mask_fpath = glob.glob(os.path.join(rootdir, animalid, session, 'ROIs', '%s*' % rois, 'masks.hdf5'))[0]
    mfile = h5py.File(mask_fpath, 'r')

    # Load and reshape masks
    masks = mfile[mfile.keys()[0]]['masks']['Slice01'][:] #.T
    #print(masks.shape)
    mfile[mfile.keys()[0]].keys()

    zimg = mfile[mfile.keys()[0]]['zproj_img']['Slice01'][:] #.T
    zimg.shape
    
    return masks, zimg

def reshape_and_binarize_masks(masks):
    # Binarze and reshape:
    nrois, d1, d2 = masks.shape
    Ain = np.reshape(masks, (nrois, d1*d2))
    Ain[Ain>0] = 1
    Ain = Ain.astype(bool).T 
    
    return Ain

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
    prefix = opts.prefix

    run_seeded_cnmf(animalid, session, fov, experiment=experiment, rootdir=rootdir, prefix=prefix)

if __name__=='__main__':
    main(sys.argv[1:])


