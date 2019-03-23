#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 16:50:26 2018

@author: juliana
"""

import sys
import os
import glob
import cv2
import re
import math
import json
import optparse
import copy
import numpy as np
import pylab as pl
import tifffile as tf
import multiprocessing as mp
from scipy.ndimage import zoom

    
def get_downsampled_std_images(run_dir, downsample_factor=(0.1, 1, 1), interpolation='bilinear',
                               pid='processed001', channel='Channel01'):
    print "RUN:", run_dir 
    
    # Load run info:
    runinfo_fpath = glob.glob(os.path.join(run_dir, '*.json'))[0]
    with open(runinfo_fpath, 'r') as f: runinfo = json.load(f)
    nchannels = runinfo['nchannels']

    tif_paths = glob.glob(os.path.join(run_dir, 'processed', '%s*' % pid, 'mcorrected_*', '*.tif'))
    # create output dir:
    zproj_dir = '%s_std_deinterleaved/%s' % (os.path.split(tif_paths[0])[0], channel)
    if not os.path.exists(zproj_dir): os.makedirs(zproj_dir)
    print "Checking dir for existing tifs in: %s" % zproj_dir
 
    # First check if already done:
    existing_std_paths = glob.glob(os.path.join(zproj_dir, 'File*', '*std*.tif'))
    existing_files = [str(re.search('File(\d{3})', p).group(0)) for p in existing_std_paths]
    print "Found %i existing STD images." % len(existing_std_paths)
    tif_paths = [p for p in tif_paths if str(re.search('File(\d{3})', p).group(0)) not in existing_files]
    print "Missing %i images." % len(tif_paths)
 
    for ti, tf_path in enumerate(tif_paths):
        print "Processing %i of %i files." % (int(ti+1), len(tif_paths))
        print "... %s" % tf_path
        write_dir = os.path.join(zproj_dir, str(re.search('File(\d{3})', tf_path).group(0)))
        curr_file = str(re.search('File(\d{3})', tf_path).group(0))
        if 'Channel' not in tf_path or 'Slice' not in tf_path: # interleaved
            deint_fpath = glob.glob(os.path.join('%s_deinterleaved' % os.path.split(tf_path)[0], channel, curr_file, '*.tif'))[0]
            fn_append = str(re.search('Slice(\d{2})_Channel(\d{2})_File(\d{3})', deint_fpath).group(0)) 
        else:
            fn_append = str(re.search('Slice(\d{2})_Channel(\d{2})_File(\d{3})', tf_path).group(0))
        if not os.path.exists(write_dir): os.makedirs(write_dir)
        tif_tmp = tf.imread(tf_path)
        print "orig tif shape: %s" % str(tif_tmp.shape)
        if nchannels==2:
            if channel=='Channel02':
                tif = tif_tmp[1::nchannels, :, :]
            else:
                tif = tif_tmp[0::nchannels, :, :]
        else:
            tif = copy.copy(tif_tmp)
        print fn_append, tif.shape, tif.dtype
        if interpolation == 'bilinear':
            order = 1
        elif interpolation == 'cubic':
            order = 3
        elif interpolation == 'nearest':
            order = 0
        tif_r = zoom(tif, downsample_factor, order=order)
        std_img = np.empty((tif_r.shape[1], tif_r.shape[2]), dtype=tif_r.dtype)
        std_img[:] = np.std(tif_r, axis=0)
        tf.imsave(os.path.join(write_dir, 'std_%s.tif' % fn_append), std_img)

       # tf.imsave(os.path.join(write_dir, 'std_%s.tif' % fn_append), tif_r)
    
    return zproj_dir

def get_downsampled_std_images_mp(run_dir, downsample_factor=(0.1, 1, 1), interpolation='bilinear',
                               pid='processed001', channel='Channel01', nprocs=4):

    if interpolation == 'bilinear':
        order = 1
    elif interpolation == 'cubic':
        order = 3
    elif interpolation == 'nearest':
        order = 0
 
    # Load run info:
    runinfo_fpath = glob.glob(os.path.join(run_dir, '*.json'))[0]
    with open(runinfo_fpath, 'r') as f: runinfo = json.load(f)
    nchannels = runinfo['nchannels']
   
    tif_paths = glob.glob(os.path.join(run_dir, 'processed', '%s*' % pid, 'mcorrected_*', '*.tif'))
    # create output dir:
    zproj_dir = '%s_std_deinterleaved/%s' % (os.path.split(tif_paths[0])[0], channel)
    if not os.path.exists(zproj_dir): os.makedirs(zproj_dir)

    print "Checking dir for existing tifs in: %s" % zproj_dir
 
    # First check if already done:
    existing_std_paths = glob.glob(os.path.join(zproj_dir, 'File*', '*std*.tif'))
    existing_files = [str(re.search('File(\d{3})', p).group(0)) for p in existing_std_paths]
    print "Found %i existing STD images." % len(existing_std_paths)
    tif_paths = [p for p in tif_paths if str(re.search('File(\d{3})', p).group(0)) not in existing_files]
    print "Missing %i images." % len(tif_paths)
    
    def downsampler(tif_list, downsample_factor, order, zproj_dir, out_q):
        
        outdict = {}
        for ti, tf_path in enumerate(tif_list):
            print "... Processing %i of %i tifs in chunk." % (int(ti+1), len(tif_list))
            write_dir = os.path.join(zproj_dir, str(re.search('File(\d{3})', tf_path).group(0)))
            curr_file = str(re.search('File(\d{3})', tf_path).group(0))
            if 'Channel' not in tf_path or 'Slice' not in tf_path: # interleaved
                deint_fpath = glob.glob(os.path.join('%s_deinterleaved' % os.path.split(tf_path)[0], channel, curr_file, '*.tif'))[0]
                fn_append = str(re.search('Slice(\d{2})_Channel(\d{2})_File(\d{3})', deint_fpath).group(0)) 
            else:
                fn_append = str(re.search('Slice(\d{2})_Channel(\d{2})_File(\d{3})', tf_path).group(0))
            if not os.path.exists(write_dir): os.makedirs(write_dir)
            tif_tmp = tf.imread(tf_path)
            print "orig shape: %s" % str(tif_tmp.shape)
            if nchannels==2:
                if channel=='Channel02':
                    tif = tif_tmp[1::nchannels, :, :]
                else:
                    tif = tif_tmp[0::nchannels, :, :]
            else:
                tif = copy.copy(tif_tmp)
            print "... new tif: %s" % str(tif.shape)
#
#            fn_append = str(re.search('Slice(\d{2})_Channel(\d{2})_File(\d{3})', tf_path).group(0))
#            if not os.path.exists(write_dir): os.makedirs(write_dir)
#            tif = tf.imread(tf_path)
            #print fn_append, tif.shape
            tif_r = zoom(tif, downsample_factor, order=order)
            std_img = np.empty((tif_r.shape[1], tif_r.shape[2]), dtype=tif_r.dtype)
            std_img[:] = np.std(tif_r, axis=0)
            tf.imsave(os.path.join(write_dir, 'std_%s.tif' % fn_append), std_img)
            outdict[fn_append] = write_dir
            print 'Done:', fn_append, tif.shape

            
        out_q.put(outdict)
    
    # Each process gets "chunksize' filenames and a queue to put his out-dict into:
    out_q = mp.Queue()
    chunksize = int(math.ceil(len(tif_paths) / float(nprocs)))
    procs = []

    for i in range(nprocs):
        p = mp.Process(target=downsampler,
                       args=(tif_paths[chunksize * i:chunksize * (i + 1)],
                                       downsample_factor,
                                       order,
                                       zproj_dir,
                                       out_q))
        procs.append(p)
        p.start()
        
    # Collect all results into single results dict. We should know how many dicts to expect:
    resultdict = {}
    for i in range(nprocs):
        resultdict.update(out_q.get())

    # Wait for all worker processes to finish
    for p in procs:
        print "Finished:", p
        p.join()

    return zproj_dir #resultdict

def extract_options(options):

    parser = optparse.OptionParser()

    parser.add_option('-D', '--root', action='store', dest='rootdir',
                          default='/nas/volume1/2photon/data',
                          help='data root dir (dir w/ all animalids) [default: /nas/volume1/2photon/data, /n/coxfs01/2pdata if --slurm]')
    parser.add_option('--slurm', action='store_true', dest='slurm', default=False, help="set if running as SLURM job on Odyssey")

    parser.add_option('-i', '--animalid', action='store', dest='animalid',
                          default='', help='Animal ID')

    # Set specific session/run for current animal:
    parser.add_option('-S', '--session', action='store', dest='session',
                          default='', help='session dir (format: YYYMMDD_ANIMALID')
    parser.add_option('-A', '--acq', action='store', dest='acquisition',
                          default='FOV1', help="acquisition folder (ex: 'FOV1_zoom3x') [default: FOV1]")

    parser.add_option('-R', '--run', dest='run', default='', action='store', help="run name") 
    parser.add_option('-p', '--pid', dest='pid', default='processed001', action='store', help='Process PID name (default: processed001)')
    parser.add_option('--c', '--channel', dest='channel', default=1, action='store', help='Channel 1 or 2 (default: 1)')
    parser.add_option('-I', '--interp', dest='interpolation', action='store', default='bilinear', help='Method of interpolation for downsampled tifs (default: bilinear)')
    parser.add_option('-d', '--downsample', dest='downsample_factor', action='store', default='0.1, 1, 1', help="Downsample factors (z, y, x) - default is (0.1, 1, 1)")

    parser.add_option('-n', '--nproc', dest='nprocesses', default=1, action='store', help='N processes to use (default: 1)')

    (options, args) = parser.parse_args(options)
    if options.slurm:
        options.rootdir = '/n/coxfs01/2p-data' 

    return options

def deinterleaved_to_stack(zproj_dir):
    img_paths = glob.glob(os.path.join(zproj_dir, 'File*', '*.tif'))
    print "Found %i deinterleaved images." % len(img_paths)
    stack = []
    for i, imgp in enumerate(img_paths):
        img = tf.imread(imgp)
        print i, imgp
        print "... D img: %s" % str(img.shape), img.dtype
        if len(img.shape) == 3:
            std_img = np.empty((img.shape[1], img.shape[2]), dtype=img.dtype)
            std_img[:] = np.std(img, axis=0)
            tf.imsave(imgp, std_img)
            stack.append(std_img)
        else:
            stack.append(img)
    stack_r = np.dstack(stack).T
    
    stack_r = np.swapaxes(stack_r, 1, 2) # swap x,y to match natural

    print stack_r.shape, stack_r.dtype, stack[0].dtype
    all_imgs = np.empty(stack_r.shape, dtype=stack[0].dtype)
    all_imgs[:] = stack_r #np.dstack(stack)
    std_base_dir = os.path.split(zproj_dir)[0]
    stack_outfile = os.path.join(std_base_dir, 'std_images.tif')
    print "Saving stack to:", stack_outfile
    tf.imsave(stack_outfile, all_imgs)
 

def main(options):

    optsE = extract_options(options)
    run_dir = os.path.join(optsE.rootdir, optsE.animalid, optsE.session, optsE.acquisition, optsE.run)
    downsample_factor = tuple(float(i) for i in optsE.downsample_factor.split(','))
    print "Downsample factor:", downsample_factor 
    channel = 'Channel%02d' % int(optsE.channel)
    nprocs = int(optsE.nprocesses)
    if nprocs > 1:
        zproj_dir = get_downsampled_std_images_mp(run_dir, downsample_factor=downsample_factor, pid=optsE.pid, channel=channel, interpolation=optsE.interpolation, nprocs=nprocs)
    else:
        zproj_dir = get_downsampled_std_images(run_dir, downsample_factor=downsample_factor, pid=optsE.pid, channel=channel, interpolation=optsE.interpolation)

    print "*** DONE ***"
    print "Downsample, STD-projected images saved to:\n", zproj_dir
    deinterleaved_to_stack(zproj_dir)

if __name__ == '__main__':
    main(sys.argv[1:])
