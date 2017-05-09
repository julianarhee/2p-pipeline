#!/usr/bin/env python2

import os
import numpy as np
import tifffile as tf

source = '/nas/volume1/2photon/RESDATA/TEFO'
session = '20161219_JR030W'
run = 'retinotopyFinal'

tiffpath = os.path.join(source, session, run)
tiffs = os.listdir(tiffpath)
tiffs = [t for t in tiffs if t.endswith('.tif')]

savepath = os.path.join(tiffpath, 'DATA')
if not os.path.exists(savepath):
    os.mkdir(savepath)


nflyback = 8
ndiscard = 8

nslices_full = 38

nchannels = 2                         # n channels in vol (ch1, ch2, ch1, ch2, ...)
nvolumes = 340


for tiffidx in range(len(tiffs)):
    stack = tf.imread(os.path.join(tiffpath, tiffs[tiffidx]))

    print "TIFF: %s" % tiffs[tiffidx]
    print "size: ", stack.shape
    print "dtype: ", stack.dtype


   
    # First, remove DISCARD frames:
    nslices_orig = nslices_full - ndiscard #30 # single-channel n z-slices

    start_idxs = np.arange(0, stack.shape[0], nslices_full*nchannels)
    substack = np.empty((nslices_orig*nchannels*nvolumes, stack.shape[1], stack.shape[2]), dtype=stack.dtype)
    newstart = 0
    for x in range(len(start_idxs)):    
	substack[newstart:newstart+(nslices_orig*nchannels),:,:] = stack[start_idxs[x]:start_idxs[x]+(nslices_orig*nchannels), :, :]
	newstart = newstart + (nslices_orig*nchannels)
    
    print "Removed discard frames. New substack shape is: ", substack.shape    

    # Next, crop off FLYBACK frames: 
    nslices_crop = nslices_orig - nflyback 

    start_idxs = np.arange(nflyback*nchannels, substack.shape[0], nslices_orig*nchannels)
    final = np.empty((nslices_crop*nchannels*nvolumes, substack.shape[1], substack.shape[2]), dtype=stack.dtype)
    newstart = 0
    for x in range(len(start_idxs)):
	final[newstart:newstart+(nslices_crop*nchannels),:,:] = substack[start_idxs[x]:start_idxs[x]+(nslices_crop*nchannels), :, :]
	newstart = newstart + (nslices_crop*nchannels)
    
    print "Removed flyback frames. Final shape is: ", final.shape

    newtiff_fn = 'File%05d.tif' % int(tiffidx+1)
    tf.imsave(os.path.join(savepath, newtiff_fn), final)


 


