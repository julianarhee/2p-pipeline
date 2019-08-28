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
import math

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


    (options, args) = parser.parse_args(options)

    return options



def downsample_movie(fn, fi=None, ds_factor=(5, 1, 1), outdir='/tmp'):
    if isinstance(ds_factor, int):
        ds_factor = (ds_factor, 1, 1)

    mov = tf.imread(fn)
    #print(mov.shape)
    ds_mov = cmdownscale(mov, ds_factor)
    #print(ds_mov.shape)
    #print(ds_mov.dtype)
    fname = os.path.splitext(os.path.split(fn)[-1])[0]
    if fi == None:
        fname = fname
    else:
        fname = 'file%05d_%s' % (int(fi+1), fname) 
    tf.imsave(os.path.join(outdir, '%s.tif' % fname), ds_mov)
    print("... finished %s" % fname)
    
    return fname

def downsample_experiment_movies(animalid, session, fov, experiment='', ds_factor=5, destdir='/tmp', rootdir='/n/coxfs01/2p-data', use_raw=True, n_processes=2, create_new=False): 
    
    fovdir = glob.glob(os.path.join(rootdir, animalid, session, fov))[0]
    all_experiments = [os.path.split(d)[-1] for d in glob.glob(os.path.join(fovdir, '*_run*')) if 'combined' not in d and 'anatomical' not in d]
    experiment_types = list(set([d.split('_')[0] for d in all_experiments]))
    print(experiment_types)
    assert experiment in experiment_types, "Specified exp - %s - not found." % experiment
    experiment_dirs = sorted(glob.glob(os.path.join(fovdir, '%s*' % experiment)), key=natural_keys)
    print("Found %i blocks for experiment: %s." % (len(experiment_dirs), experiment))
    for edir in experiment_dirs:
        print(edir)

    fnames = []
    if use_raw:
        fnames = glob.glob(os.path.join(fovdir, '%s*' % experiment, 'raw*', '*.tif'))
    else:
        fnames = [f for f in glob.glob(os.path.join(fovdir, '%s*' % experiment, 
                                                        'processed', 'processed001*','mcorrected*', '*.tif'))\
                     if len(os.path.split(os.path.split(f)[0])[-1].split('_'))==2]
    print("[%s]: added %i tifs to queue." % (experiment, len(fnames)))
    fnames = sorted(fnames, key=natural_keys)

    ds = ds_factor[0] if isinstance(ds_factor, tuple) else ds_factor
    skey = '-'.join([animalid, session, fov, experiment, 'downsample-%i' % ds])

    if not use_raw:
        skey = '%s-mcorrected' % skey

    outdir = os.path.join(destdir, skey)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
   
    do_downsample = create_new 
    found_movs = glob.glob(os.path.join(outdir, '*.tif'))
    print("Found %i downsampled movies. Expecting %i." % (len(found_movs), len(fnames)))
    if len(found_movs) == len(fnames) and create_new is False:
        print("Downsampled movies found. Skipping...")
        do_downsample = False
    else:
        print("Saving downsampled movies to: %s" % outdir)
        do_downsample = True 
   
    if do_downsample: 
        start_t = time.time()
        #pool = mp.Pool(processes=n_processes)# as pool:
        #pool.map_async(partial(downsample_movie, ds_factor=ds_factor, tmpdir=tmpdir, use_raw=use_raw), fnames)
        #pool.close()
        #pool.join()
        if n_processes == 1:
            for fi, fn in enumerate(sorted(fnames, key=natural_keys)):
                if fi % 10. == 0:
                    print("... processed %i of %i movies." % (int(fi+1), len(fnames)))
                outname = downsample_movie(fn, fi=fi, ds_factor=ds_factor, outdir=outdir) 
        else:
            outdir = downsample_movie_list_mp(fnames, ds_factor=ds_factor, outdir=outdir, n_processes=n_processes) 
        end_t = time.time() - start_t
        print("Downsample: {0:.2f}sec".format(end_t))
    print("**** done! ****")

    return outdir


def downsample_movie_list_mp(file_list, ds_factor=5, outdir='/tmp', n_processes=1):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
   
    file_list = sorted(file_list, key=natural_keys)
    file_ids = [i for i, fn in enumerate(sorted(file_list, key=natural_keys))]
 
    def downsampler(file_list, file_ids, ds_factor, outdir, out_q):
        
        outdict = {}
        for fi, fn in zip(file_ids, file_list):
            outname = downsample_movie(fn, fi=fi, ds_factor=ds_factor, outdir=outdir)
            outdict['%i_%s' % (fi, fn)] = outname 
        out_q.put(outdict)
    
    # Each process gets "chunksize' filenames and a queue to put his out-dict into:
    out_q = mp.Queue()
    chunksize = int(math.ceil(len(file_list) / float(n_processes)))
    procs = []
    for i in range(n_processes):
        p = mp.Process(target=downsampler,
                       args=(file_list[chunksize * i:chunksize * (i + 1)],
                             file_ids[chunksize * i:chunksize * (i+1)],
                             ds_factor,
                             outdir,
                             out_q))
        procs.append(p)
        p.start()
        
    # Collect all results into single results dict. We should know how many dicts to expect:
    resultdict = {}
    for i in range(n_processes):
        resultdict.update(out_q.get())

    # Wait for all worker processes to finish
    for p in procs:
        print("Finished:", p)
        p.join()

    return outdir #resultdict


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

    outdir = downsample_experiment_movies(animalid, session, fov, experiment=experiment,
                                ds_factor=ds_factor, destdir=destdir, use_raw=use_raw, n_processes=n_processes, create_new=create_new) 

    print("--- all movies saved to:\n---%s" % outdir)

if __name__=='__main__':
    mp.freeze_support()
    main(sys.argv[1:])


