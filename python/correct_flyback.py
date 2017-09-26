#!/usr/bin/env python2
'''
Run this script to parse RAW acquisition files (.tif) from SI.
Native format is int16. This script will remove all artifact and discard frames, and save to new .tif (int16, uint1`6).

Append _visible.tif to crop display range for viewing on BOSS.

Run python correct_flyback.py -h for all input options.
'''
import os
import numpy as np
import tifffile as tf
from skimage import img_as_uint
from skimage import exposure
import optparse

parser = optparse.OptionParser()

# PATH opts:
parser.add_option('-S', '--source', action='store', dest='source', default='', help='source dir (parent of session dir)')
parser.add_option('-s', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID')
parser.add_option('-A', '--acq', action='store', dest='acquisition', default='', help="acquisition folder (ex: 'FOV1_zoom3x')")
parser.add_option('-f', '--functional', action='store', dest='functional_dir', default='functional', help="folder containing functional TIFFs. [default: 'functional']")

# TIFF saving opts:
parser.add_option('--native', action='store_false', dest='uint16', default=True, help='Keep int16 tiffs as native [default: convert to uint16]')
parser.add_option('--visible', action='store_true', dest='visible', default=False, help='Also create tiffs with adjusted display-range to make visible.')
parser.add_option('-m', '--min', action='store', dest='displaymin', default=10, help='min display range value [default: 10]')
parser.add_option('-M', '--max', action='store', dest='displaymax', default=2000, help='max display range value [default: 2000]')
#parser.add_option('-P', '--savepath', action='store', dest='savepath', default='', help='path to save new TIFFs.')

# SUBSTACK opts (for correcting flyback):
parser.add_option('--substack', action='store_true', dest='create_substacks', default=False, help='Create substacks of data-only by removing all artifact/discard frames [default: False]')

parser.add_option('--flyback', action='store', dest='nflyback', default=8, help='Num flyback frames at top of stack [default: 8]')
parser.add_option('--discard', action='store', dest='ndiscard', default=8, help='Num discard frames at end of stack [default: 8]')
parser.add_option('-c', '--nchannels', action='store', dest='nchannels', default=2, help='Num interleaved channels in raw tiffs to be processed [default: 2]')
parser.add_option('-v', '--nvolumes', action='store', dest='nvolumes', default=340, help='Num volumes acquired [default: 340]')
parser.add_option('-z', '--nslices', action='store', dest='nslices', default=30, help='Num slices specified, no discard [default: 30]')

# FOV-cropping opts (prob shouldn't go here):
parser.add_option('--crop', action='store_true', dest='crop_fov', default=False, help='Crop FOV in x,y for smaller data size [default: False]')
parser.add_option('-X', '--xstart', action='store', dest='x_startidx', default=0, help='Starting idx for x-dimension, i.e., columns [default: 0]')
parser.add_option('-Y', '--ystart', action='store', dest='y_startidx', default=0, help='Starting idx for y-dimension, i.e., rows [default: 0]')
parser.add_option('-W', '--width', action='store', dest='width', default='', help='Width of FOV, i.e., cols [default: '']')
parser.add_option('-H', '--height', action='store', dest='height', default='', help='Height of FOV, i.e., rows [default: '']')


(options, args) = parser.parse_args() 

crop_fov = options.crop_fov
x_startidx = int(options.x_startidx)
y_startidx = int(options.y_startidx)

create_substacks = options.create_substacks
uint16 = options.uint16
nflyback = int(options.nflyback)
ndiscard = int(options.ndiscard)
nchannels = int(options.nchannels)
nvolumes = int(options.nvolumes)
nslices = int(options.nslices)
nslices_full = nslices + ndiscard
print "nvolumes:", nvolumes
print "nslices specified:", nslices
print "n expected after substack:", nslices - nflyback

source = options.source 
session = options.session 
acquisition = options.run
functional_dir = options.functional_dir

visible = options.visible
if visible:
    displaymin = options.displaymin
    displaymax = options.displaymax

tiffpath = os.path.join(source, session, acquisition, functional_dir)

# Set and create default output-directory:
#savepath = options.savepath
savepath = os.path.join(tiffpath, 'DATA')
if not os.path.exists(savepath):
    os.mkdir(savepath)


tiffs = os.listdir(tiffpath)
tiffs = [t for t in tiffs if t.endswith('.tif')]


for tiffidx,tiffname in enumerate(tiffs):
    stack = tf.imread(os.path.join(tiffpath, tiffs[tiffidx]))
    if uint16:
        stack = img_as_uint(stack)

    print "TIFF: %s" % tiffs[tiffidx]
    print "size: ", stack.shape
    print "dtype: ", stack.dtype
    if create_substacks:
       
	# First, remove DISCARD frames:
	nslices_orig = nslices_full - ndiscard #30 # single-channel n z-slices

	start_idxs = np.arange(0, stack.shape[0], nslices_full*nchannels)
	substack = np.empty((nslices_orig*nchannels*nvolumes, stack.shape[1], stack.shape[2]), dtype=stack.dtype)
	print "Removing SI discard frames. Tmp stack shape:", substack.shape
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
    
    else:
        print "Not creating substacks from input tiffs."
        
        final = tf.imread(os.path.join(tiffpath, tiffs[tiffidx]))
        final = img_as_uint(final)
   
    if crop_fov:
	if len(options.width)==0:
	    width = int(input('No width specified. Starting idx is: %i.\nEnter image width: ' % x_startidx))
	else:
	    width = int(options.width)
	if len(options.height)==0:
	    height = int(input('No height specified. Starting idx is: %i.\nEnter image height: ' % y_startidx))
	else:
	    height = int(options.height)
	x_endidx = x_startidx + width
	y_endidx = y_startidx + height
	
	final = final[:, y_startidx:y_endidx, x_startidx:x_endidx]
	print "Cropped FOV. New size: ", final.shape

    origname = tiffname.split('.')[0]
    prefix = '_'.join(origname.split('_')[0:-1])
    newtiff_fn = '%s_File%03d.tif' % (prefix, int(tiffidx+1)) #'File%03d.tif' % int(tiffidx+1)
    tf.imsave(os.path.join(savepath, newtiff_fn), final)
    
    if visible: 
        ranged = exposure.rescale_intensity(final, in_range=(displaymin, displaymax))
        rangetiff_fn = '%s_visible.tif' % newtiff_fn.split('.')[0] #'File%03d_visible.tif' % int(tiffidx+1)
        tf.imsave(os.path.join(savepath, rangetiff_fn), ranged)


