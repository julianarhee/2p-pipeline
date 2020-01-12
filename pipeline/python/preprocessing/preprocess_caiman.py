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


#%%
from caiman.source_extraction.cnmf.initialization import downscale as cmdownscale
import downsample_movies as preproc

import pylab as pl
from functools import partial
import tifffile as tf
import multiprocessing as mp
import json
import time
import re
import optparse
import sys
import _pickle as pkl
import traceback

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]



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
    parser.add_option('-d', '--ds-factor', action="store",
                      dest="ds_factor", default=5, help="Downsample factor (int, default: 5)")

    parser.add_option('--destdir', action="store",
                      dest="destdir", default='/n/scratchlfs02/cox_lab/julianarhee/downsampled', help="output dir for movie files [default: /n/scratchlfs/cox_lab/julianarhee/downsampled]")
    parser.add_option('--plot', action='store_true', dest='plot_rois', default=False, help="set to plot results of each roi's analysis")
    parser.add_option('--processed', action='store_false', dest='use_raw', default=True, help="set to downsample on non-raw source")

    parser.add_option('--new', action='store_true', dest='create_new', default=False, help="Set to downsample and motion correct anew")
    parser.add_option('--motion', action='store_true', dest='do_motion', default=False, help="Set to do motion correction")
    parser.add_option('--downsample', action='store_true', dest='do_downsample', default=False, help="Set to temporally downsample tifs before caiman memmap and MC")
    parser.add_option('--memmap', action='store_true', dest='do_memmap', default=False, help="Set to do memmap only (assumes prev MC-corrected)")

    parser.add_option('--prefix', action="store",
                      dest="prefix", default=None, help="Prefix for memmapped files [default: Yr]")
    parser.add_option('--source-file', action="store",
                      dest="source_file", default=None, help="Full path to full memmaped file [default: None]")



    (options, args) = parser.parse_args(options)

    return options



def caiman_params(fnames, **kwargs):
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

   
    if kwargs is not None and kwargs != {}:
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





def save_memmap_params(results_dir, resize_fact=(), add_to_movie=None, border_to_0=None, fnames=[], prefix='Yr'):
    # create mmap params file
    mmap_fpath = os.path.join(results_dir, '%s_memmap-params.json' % prefix)
    mmap_params = {'resize_fact': list(resize_fact), 
                   'add_to_movie': add_to_movie,
                   'border_to_0': border_to_0,
                   'fnames': fnames}
    for k, m in mmap_params.items():
        if isinstance(m, list):
            if not isinstance(m[0], (str, bytes)):
                mmap_params[k] = [float(mi) for mi in m]
            #print
        else:
            mmap_params[k] = float(m)

    with open(mmap_fpath, 'w') as f:
        json.dump(mmap_params, f, indent=4)
     

def save_mc_results(mc, results_dir, prefix='Yr'):
    np.savez(os.path.join(results_dir, '%s_mc-rigid.npz' % prefix),
            #mc=mc,
            fname=mc.fname, 
            max_shifts=mc.max_shifts, 
            min_mov=mc.min_mov,
            border_nan=mc.border_nan,
            fname_tot_rig=mc.fname_tot_rig,
            total_template_rig=mc.total_template_rig,
            templates_rig=mc.templates_rig,
            shifts_rig=mc.shifts_rig,
            mmap_file=mc.mmap_file,
            border_to_0=mc.border_to_0,
            gSig_filt=mc.gSig_filt)

    print("--- saved MC results: %s" % os.path.join(results_dir, '%s_mc-rigid.npz' % prefix))
    
    # Also save params as json 
    if 'downsample' in prefix:
        ds_fact = int(prefix.split('downsample-')[-1])
        resize_fact = (1, 1, 1./ds_fact)
    else:
        resize_fact = (1, 1, 1)
    save_memmap_params(results_dir, resize_fact=resize_fact, add_to_movie=mc.min_mov, 
                   border_to_0=mc.border_to_0, fnames=mc.fname, prefix=prefix)
    print("--- saved MC params to JSON")


def load_mc_results(results_dir, mc, prefix='Yr'):
    try:
        res = np.load(os.path.join(results_dir, '%s_mc-rigid.npz' % prefix), allow_pickle=True)
        #mc = mc_results[mc] 
        print(list(res.keys()))
        mc.fname = res['fname']
        mc.max_shifts = res['max_shifts']
        mc.min_mov = res['min_mov']
        mc.border_nan = res['border_nan']
        mc.fname_tot_rig = res['fname_tot_rig']
        mc.total_template_rig = res['total_template_rig']
        mc.templates_rig = res['templates_rig']
        mc.shifts_rig = res['shifts_rig']
        mc.mmap_file = res['mmap_file']
        mc.border_to_0 = res['border_to_0']
        mc.gSig_filt = res['gSig_filt']

    except Exception as e:
        print(e)
        print("Error loading MC")
        return None

    return mc 

def get_full_memmap_path(results_dir, framestr='order_C_frames', prefix='Yr'):
    try:
        fname_new = glob.glob(os.path.join(results_dir, 'memmap', '*%s*_d*%s*_.mmap' % (prefix, framestr)))
        assert len(fname_new)==1, "Unique fname_new not found: %s" % str(fname_new)
        fname_new = fname_new[0] 
    except Exception as e:
        return None
    return fname_new


def do_memmap_no_mc(animalid, session, fov, run_label='res', base_name='Yr', resize_fact=(1, 1, 1), remove_init=0, srcdir=None,
                    add_to_movie=0., border_to_0=0, dview=None, rootdir='/n/coxfs01/2p-data', n_processes=8):
    fovdir = os.path.join(rootdir, animalid, session, fov)
    results_dir = os.path.join(fovdir, 'caiman_results', run_label)

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    start_t = time.time()
    #%% start a cluster for parallel processing
    # (if a cluster already exists it will be closed and a new session will be opened)
    if 'dview' in locals():
        cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=n_processes, single_thread=False)

    if srcdir is None:
        if 'run' in run_label:
            srcfiles = sorted(glob.glob(os.path.join(fovdir, '%s*' % run_label, 'processed', 'processed001*', '*mcorrected_*', 'fov*.tif')), key=natural_keys)
        else:
            srcfiles = sorted(glob.glob(os.path.join(fovdir, '%s_*' % run_label, 'processed', 'processed001*', '*mcorrected_*', 'fov*.tif')), key=natural_keys)
    else:
        srcfiles = sorted(glob.glob(os.path.join(srcdir, '*.tif')), key=natural_keys)
    print("Found %i previously corrected src files to memmap." % len(srcfiles))

    mmapdir = os.path.join(results_dir, 'memmap')
    prefix = base_name
    base_name = '%s/%s' % (mmapdir, prefix)
 
    fname_tot = cm.save_memmap(srcfiles, base_name=base_name, resize_fact=resize_fact,
                            remove_init=remove_init, add_to_movie=add_to_movie, border_to_0=border_to_0,
                            dview=dview)

    end_t = time.time() - start_t
    print("... MMAP total - Elapsed time: {0:.2f}sec".format(end_t))

    # Also save params as json 
    if 'downsample' in prefix:
        ds_fact = int(prefix.split('downsample-')[-1])
        resize_fact = (1, 1, 1./ds_fact)
    else:
        resize_fact = (1, 1, 1)
    save_memmap_params(results_dir, resize_fact=resize_fact, add_to_movie=mc.min_mov, 
                   uorder_to_0=mc.border_to_0, fnames=mc.fname, prefix=prefix)
    print("--- saved MC params to JSON")
    print("PREFIX: %s" % prefix)

    return fname_tot

def do_motion_correction(animalid, session, fov, run_label='res', srcdir=None, rootdir='/n/coxfs01/2p-data', n_processes=None, prefix=None, save_total=True, opts_kws=None):

    source_key = '-'.join([animalid, session, fov, run_label]) #, prefix])
    data_identifier = '|'.join([animalid, session, fov, run_label])
    print("*** Dataset: %s ***" % data_identifier)
    
    #%% Create output dirs
    fovdir = os.path.join(rootdir, animalid, session, fov)
    results_dir = os.path.join(fovdir, 'caiman_results', run_label)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if not os.path.exists(os.path.join(results_dir, 'memmap')):
        os.makedirs(os.path.join(results_dir, 'memmap'))

    # Get input source
    if prefix is None:
        prefix = 'Yr'
    #if prefix not in source_key:
    prefix = '%s-%s' % (source_key, prefix)
    print("***PROCESSING: %s ***" % prefix)

    print("SOURCE for mc is: ", srcdir)
    if srcdir is None:
        runstr = '%s' % run_label if 'run' in run_label else '%s_run' % run_label
        fnames = sorted(glob.glob(os.path.join(rootdir, animalid, session, fov, '%s*' % runstr, 'raw*', '*.tif')), key=natural_keys)
        print("... No sourcedir provided. Found %i .tif files to process." % len(fnames))

    else:
        fnames = sorted(glob.glob(os.path.join(srcdir, '*.tif')), key=natural_keys)
        print("... Found %i movies" % len(fnames)) 

    # Check for existing MC file (and saved mmap files)
    opts = caiman_params(fnames, **opts_kws)
    print('opts:', opts_kws)

    try:
        print("... Checking for existing .mmap files")
        mc = MotionCorrect(fnames, **opts.get_group('motion'))
        mc = load_mc_results(results_dir, mc, prefix=prefix)
        assert mc is not None, "---> No completed MC found. Running now..."
        assert len(mc.mmap_file)==len(fnames), "--> Incorrect .mmap files found (%i - should be %i)" % (len(mc.mmap_file), len(fnames))
        motion_correct = False
    except Exception as e:
        print(e)
        motion_correct = True
    
    # first we create a motion correction object with the parameters specified
    print("Creating MotionCorrect object and preprocessing data...")
    start_t = time.time()
    # note that the file is not loaded in memory
 
    #%% start a cluster for parallel processing
    # (if a cluster already exists it will be closed and a new session will be opened)
    if 'dview' in locals():
        cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=n_processes, single_thread=False)
 
    if motion_correct:
        print("... running motion correction step")
        mc = MotionCorrect(fnames, dview=dview, **opts.get_group('motion'))
        #%% Run piecewise-rigid motion correction using NoRMCorre
        mc.motion_correct(save_movie=True)
        print("... memmapped %i MC files (prefix: %s)." % (len(mc.mmap_file), prefix))
        print("... motion correction - Elapsed time: {0:.2f}sec".format(time.time()-start_t))
 
        # Save MC info 
        save_mc_results(mc, results_dir, prefix=prefix)
       
    # memory map the file in order 'C'
    #if len(mc.fname_tot_rig)>1:
    nframes_total = sum([int(mf.split('_')[-2]) for mf in mc.mmap_file])
    d1, d2 = mc.total_template_rig.shape
    dimstr = 'd1_%i_d2_%i' % (d1, d2)
    framestr = 'order_C_frames_%i' % nframes_total
    print('... checking for existing mmap total: %s, %s' % (dimstr, framestr))
    bord_px = np.ceil(np.max(np.abs(mc.shifts_rig))).astype(np.int)

    fname_new = get_full_memmap_path(results_dir, framestr=framestr, prefix=prefix)
    save_total = True
    if fname_new is not None: 
        print("... Got mmap! %s" % fname_new)
        Yr, dims, T = cm.load_memmap(fname_new)
        save_total = T!=nframes_total
        print("Expected %i frames, found %i in mmap" % (nframes_total, T))

    if save_total:
        start_t = time.time() 
        base_name = '%s/memmap/%s' % (results_dir, prefix)
        print("... saving total result to: %s" % base_name)
        fname_new = cm.save_memmap(mc.fname_tot_rig, base_name=base_name, border_to_0=bord_px, order='C', dview=dview)     
        end_t = time.time() - start_t
        print("... MMAP total - Elapsed time: {0:.2f}sec".format(end_t))

    print("***DONE!*********")
    print("Results saved to: %s" % fname_new)

    return prefix #results_dir, Cn

def main(options):
    opts = extract_options(options) 
    cparams = [a for a in options if 'c_' in a]
    if ',' in cparams and len(cparams)==1:
        cparams = cparams[0].split(',')
    c_args = dict([a[2:].split('=', maxsplit=1) for a in cparams])
    print(c_args)

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
    do_downsample = opts.do_downsample
    do_memmap = opts.do_memmap

    srcdir = None
    outdir = None
    if do_downsample:
        outdir = preproc.downsample_experiment_movies(animalid, session, fov, experiment=experiment,
                                ds_factor=ds_factor, destdir=destdir, use_raw=use_raw, n_processes=n_processes, create_new=create_new) 
        srcdir = outdir
        print("[Downsampling] Complete!")
        print("... all movies saved to:\n... %s" % outdir)


    if do_motion:
        print("[Motion correction] Starting...")
        #if srcdir is None:
        #    srctifs = glob.glob(os.path.join(rootdir, animalid, session, fov, '%s_*' % experiment, 'raw_*', '*.tif'))
            #srcdir = os.path.split(srctifs[0])[0]
        print("... src: %s" % srcdir)
        
        prefix = do_motion_correction(animalid, session, fov, run_label=experiment, srcdir=srcdir, rootdir=rootdir, n_processes=n_processes, prefix=prefix, opts_kws=c_args)

        print("--- finished motion correction ----")
        #print("All results saved to: %s" % results_dir)

    elif do_memmap:
        print("[Memmap] Starting...")
        fname_tot = do_memmap_no_mc(animalid, session, fov, run_label=experiment, base_name=prefix, resize_fact=(1, 1, 1), remove_init=0, srcdir=srcdir,
                                    add_to_movie=0., border_to_0=0, dview=None, rootdir='/n/coxfs01/2p-data')

        print("---- finished memmapping previously MC files")

if __name__=='__main__':
    mp.freeze_support()
    main(sys.argv[1:])


