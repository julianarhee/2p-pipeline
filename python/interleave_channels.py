#!/usr/bin/env python2 
''' 
Run this script to parse RAW acquisition files (.tif) from SI. 
Native format is int16. This script will remove all artifact and discard frames, and s
ave to new .tif (int16, uint1`6). 
 
Append _visible.tif to crop display range for viewing on BOSS. 
 
Run python createsubstacks.py -h for all input options. 
''' 
import os 
import numpy as np 
import tifffile as tf 
from skimage import img_as_uint 
from skimage import exposure 
import optparse 
import shutil

parser = optparse.OptionParser() 
parser.add_option('-S', '--source', action='store', dest='source', default='', help='source dir (parent of session dir)')

(options, args) = parser.parse_args()

source = options.source

channel_dir = os.path.join(source, 'Channels')
if not os.path.exists(channel_dir):
    os.mkdir(channel_dir)

tiffs = os.listdir(source)
tiffs = [t for t in tiffs if t.endswith('.tif')]
print "Found %i tiffs." % len(tiffs)

if len(tiffs)%2 > 0:
    print "Found uneven number of tiffs to interleave in source:\n%s" % source

else:
    fids = np.arange(0, len(tiffs), 2)
    print "Interleaving %i files with 2 channels." % len(fids)
    fidx = 1
    for f in fids:
	channel1 = tf.imread(os.path.join(source, tiffs[f]))
	stack = np.empty((channel1.shape[0]*2, channel1.shape[1], channel1.shape[2]), dtype=channel1.dtype)
	stack[0::2,:,:] = channel1
        del channel1
	channel2 = tf.imread(os.path.join(source, tiffs[f+1]))
	stack[1::2,:,:] = channel2
        del channel2
	filename = "File%03d.tif" % fidx
	print "Writing interleaved tiff: %s" % filename
	tf.imsave(os.path.join(source, filename), stack)
	fidx += 1

        shutil.move(os.path.join(source, tiffs[f]), os.path.join(channel_dir, tiffs[f]))
        shutil.move(os.path.join(source, tiffs[f+1]), os.path.join(channel_dir, tiffs[f+1]))
    
    print "Done interleaving tiffs!"
       
   
