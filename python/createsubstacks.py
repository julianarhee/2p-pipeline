#!/usr/bin/env python2

import os
import numpy as np
import tifffile as tf
from skimage import img_as_uint
from skimage import exposure
import optparse

#source = '/nas/volume1/2photon/RESDATA/TEFO'
#session = '20161219_JR030W'
#run = 'retinotopyFinalMask'
#
#displaymin = 10
#displaymax = 2000
#

#nflyback = 8
#ndiscard = 8
#
#nslices_full = 38
#
#nchannels = 2                         # n channels in vol (ch1, ch2, ch1, ch2, ...)
#nvolumes = 340
#

parser = optparse.OptionParser()
parser.add_option('-S', '--source', action='store', dest='source', default='', help='source dir (parent of session dir)')
parser.add_option('-s', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID')
parser.add_option('-r', '--run', action='store', dest='run', default='', help='run name (ex: retinotopyFinal)')
parser.add_option('--visible', action='store_true', dest='visible', default=False, help='Also create tiffs with adjusted display-range to make visible.')
parser.add_option('-m', '--min', action='store', dest='displaymin', default=10, help='min display range value [default: 10]')
parser.add_option('-M', '--max', action='store', dest='displaymax', default=2000, help='max display range value [default: 2000]')
parser.add_option('-P', '--savepath', action='store', dest='savepath', default='', help='path to save new TIFFs.')
parser.add_option('--flyback', action='store', dest='nflyback', default=8, help='Num flyback frames at top of stack [default: 8]')
parser.add_option('--discard', action='store', dest='ndiscard', default=8, help='Num discard frames at end of stack [default: 8]')
parser.add_option('--channels', action='store', dest='nchannels', default=2, help='Num interleaved channels in raw tiffs to be processed [default: 2]')
parser.add_option('--volumes', action='store', dest='nvolumes', default=340, help='Num volumes acquired [default: 340]')
parser.add_option('--nslices', action='store', dest='nslices', default=30, help='Num slices specified, no discard [default: 30]')


(options, args) = parser.parse_args() 

nflyback = options.nflyback
ndiscard = options.ndiscard
nchannels = options.nchannels
nvolumes = options.nvolumes
nslices_full = options.nslices + options.ndiscard


source = options.source 
session = options.session 
run = options.run
visible = options.visible
if visible:
    displaymin = options.displaymin
    displaymax = options.displaymax

savepath = options.savepath

tiffpath = os.path.join(source, session, run)

tiffs = os.listdir(tiffpath)
tiffs = [t for t in tiffs if t.endswith('.tif')]

if len(savepath)==0:
    savepath = os.path.join(tiffpath, 'DATA')
if not os.path.exists(savepath):
    os.mkdir(savepath)



for tiffidx in range(len(tiffs)):
    stack = tf.imread(os.path.join(tiffpath, tiffs[tiffidx]))
    stack = img_as_uint(stack)

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

    newtiff_fn = 'File%03d.tif' % int(tiffidx+1)
    tf.imsave(os.path.join(savepath, newtiff_fn), final)


    ranged = exposure.rescale_intensity(final, in_range=(displaymin, displaymax))
    rangetiff_fn = 'File%03d_visible.tif' % int(tiffidx+1)
    tf.imsave(os.path.join(savepath, rangetiff_fn), ranged)



