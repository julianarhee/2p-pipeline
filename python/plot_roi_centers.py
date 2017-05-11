#!/usr/bin/env python2

import os
import numpy as np
import tifffile as tf

source = '/nas/volume1/2photon/RESDATA/TEFO'
session = '20161219_JR030W'
run = 'retinotopyFinal'
didx = 5
datastruct = 'datastruct_%03d' % didx

# Specifiy TIFF paths (if not raw):
runpath = os.path.join(source, session, run)
tiffdir = 'DATA/int16'
tiffpath = os.path.join(runpath, tiffdir)

tiffs = os.listdir(tiffpath)
tiffs = [t for t in tiffs if t.endswith('.tif')]

savepath = os.path.join(tiffpath, 'DATA')
if not os.path.exists(savepath):
    os.mkdir(savepath)


tiffidx = 4
stack = tf.imread(os.path.join(tiffpath, tiffs[tiffidx]))

print "TIFF: %s" % tiffs[tiffidx]
print "size: ", stack.shape
print "dtype: ", stack.dtype

# Just get green channel:
nvolumes = 340
nslices = 22
nframes_single_channel = nvolumes * nslices
channel = 1
if stack.shape[0] > nframes_single_channel:
    print "Getting single channel: Channel %02d" % channel
    volume = stack[::2, :, :]
else:
    volume = copy.deepcopy(stack)
del stack

# Get averaged volume:
if volume.shape[0] > nslices:
    print "Averaging across time to produce average volume."
    avg = np.empty((nslices, volume.shape[1], volume.shape[2]))
    for zslice in range(avg.shape[0]):
        avg[zslice,:,:] = volume[zslice::nslices,:,:]

