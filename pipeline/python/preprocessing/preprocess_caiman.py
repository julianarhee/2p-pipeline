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

import downsample_movies as preproc


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
    parser.add_option('--motion', action='store_true', dest='do_motion', default=False, help="Set to do motion correction")

    parser.add_option('--prefix', action="store",
                      dest="prefix", default='Yr', help="Prefix for memmapped files [default: Yr]")


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
    p = 1                       # order of the autoregressive system
    gnb = 2                     # number of global background components
    merge_thr = 0.85            # merging threshold, max correlation allowed
    rf = 15                     # half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
    stride_cnmf = 6             # amount of overlap between the patches in pixels
    K = 4                       # number of components per patch
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
        print(mc.border_to_0)
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

def get_full_memmap_path(results_dir, prefix='Yr'):
    fname_new = glob.glob(os.path.join(results_dir, 'memmap', '%s_d*_.mmap' % prefix))[0]
    return fname_new



def do_motion_correction(animalid, session, fov, run_label='res', srcdir='/tmp', rootdir='/n/coxfs01/2p-data', n_processes=None, prefix=None, save_total=True):
    fnames = glob.glob(os.path.join(srcdir, '*.tif')) 
    fnames = sorted(fnames, key=natural_keys)
    print("Found %i movies." % len(fnames))

    data_identifier = '|'.join([animalid, session, fov, run_label])
    print("*** Dataset: %s ***" % data_identifier)
    
    # Check for existing memmaped/processed files
    source_key = os.path.split(srcdir)[-1]
    print("... Checking for existing processed mmaps in src: %s" % source_key)
    memfiles = glob.glob(os.path.join(srcdir, '*_.mmap'))
    if len(fnames) == len(memfiles):
        print("... Found %i existing MC memmaped files. Skipping memmap step." % len(memfiles))
        do_memmap = False
    else:
        do_memmap = True

    #%%
    fovdir = os.path.join(rootdir, animalid, session, fov)
    results_dir = os.path.join(fovdir, 'caiman_results', run_label)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    prefix = source_key #run_label if prefix is None else prefix

    #%% start a cluster for parallel processing
    # (if a cluster already exists it will be closed and a new session will be opened)
    if 'dview' in locals():
        cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=None, single_thread=False)


    #mc = load_mc_results(results_dir, prefix=prefix)
    if do_memmap:

        opts_dict = caiman_params(fnames)
        opts = params.CNMFParams(params_dict=opts_dict)

        # first we create a motion correction object with the parameters specified
        mc = MotionCorrect(fnames, dview=dview, **opts.get_group('motion'))
        # note that the file is not loaded in memory

        #%% Run piecewise-rigid motion correction using NoRMCorre
        mc.motion_correct(save_movie=True)
        save_mc_results(results_dir, prefix=prefix)
        memfiles = mc.mmap_file
        print("... memmaped %i MC files (prefix: %s)." % (len(memfiles), prefix))

    # memory map the file in order 'C'
    if save_total:
        base_name = '%s/memmap/%s' % (results_dir, prefix)
        print("... saving total result to: %s" % base_name)
        if not os.path.exists(os.path.join(results_dir, 'memmap')):
            os.makedirs(os.path.join(results_dir, 'memmap'))
        fname_new = cm.save_memmap_join(memfiles, base_name=base_name, dview=dview)

        #fname_new = get_full_memmap_path(results_dir, prefix=prefix)
 
        print("DONE! MMAP and MC results saved to: %s" % fname_new)

    return mc #results_dir, Cn

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
    do_motion = opts.do_motion
    prefix = opts.prefix

    outdir = preproc.downsample_experiment_movies(animalid, session, fov, experiment=experiment,
                                ds_factor=ds_factor, destdir=destdir, use_raw=use_raw, n_processes=n_processes, create_new=create_new) 
    print("[Downsampling] Complete!")
    print("... all movies saved to:\n... %s" % outdir)


    if do_motion:
        mc = do_motion_correction(animalid, session, fov, run_label=experiment, srcdir=outdir, rootdir=rootdir, n_processes=n_processes, prefix=prefix)

        print("--- finished motion correction ----")
        #print("All results saved to: %s" % results_dir)

if __name__=='__main__':
    mp.freeze_support()
    main(sys.argv[1:])


