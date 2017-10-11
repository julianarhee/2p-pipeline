#!/usr/bin/env python2
'''
Combine repetitions of a given run-type/trial-type if corresponding TIFF files are the same in stucture and size.  Need to provide matching file indices from list.

Native format is int16. This script will remove all artifact and discard frames, and save to new .tif (int16, uint1`6).

Append _visible.tif to crop display range for viewing on BOSS.

Run python create_averaged_tiffs.py -h for all input options.
'''
import os
import numpy as np
import tifffile as tf
from skimage import img_as_uint
from skimage import exposure
import optparse

#TODO:  no hard-cording
sourcepath = '/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/retinotopyFinal/memfiles/tiffs'
tiffs = os.listdir(sourcepath)
tiffs = [t for t in tiffs if t.endswith('.tif')]

acquisition_names = ['fov6_retinobar_037Hz_final_nomask', 'fov6_retinobar_037Hz_final_bluemask']
tiffs1 = [i for i in tiffs if acquisition_names[0] in i]
tiffs2 = [i for i in tiffs if acquisition_names[1] in i]

savepath = '/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/retinotopyFinal/memfiles/averaged/tiffs'
if not os.path.exists(savepath):
    os.mkdirs(savepath)

matchtrials = np.array([[1, 1], [2, 3], [3, 2], [4, 4]]) - 1 # bec 0-indexing

parser = optparse.OptionParser()
#parser.add_option('-S', '--source', action='store', dest='source', default='', help='source dir (parent of session dir)')
#parser.add_option('-s', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID')
#parser.add_option('-r', '--run', action='store', dest='run', default='', help='run name (ex: retinotopyFinal)')
parser.add_option('--visible', action='store_true', dest='visible', default=False, help='Also create tiffs with adjusted display-range to make visible.')
parser.add_option('--uint16', action='store_true', dest='uint16', default=False, help='Convert native int16 to uint16.')

parser.add_option('-m', '--min', action='store', dest='displaymin', default=10, help='min display range value [default: 10]')
parser.add_option('-M', '--max', action='store', dest='displaymax', default=2000, help='max display range value [default: 2000]')
#parser.add_option('-P', '--savepath', action='store', dest='savepath', default='', help='path to save new TIFFs.')
#parser.add_option('--channels', action='store', dest='nchannels', default=2, help='Num interleaved channels in raw tiffs to be processed [default: 2]')
#parser.add_option('--volumes', action='store', dest='nvolumes', default=340, help='Num volumes acquired [default: 340]')
#parser.add_option('--nslices', action='store', dest='nslices', default=30, help='Num slices specified, no discard [default: 30]')
parser.add_option('--native', action='store_false', dest='uint16', default=True, help='Keep int16 tiffs as native [default: convert to uint16]')
#parser.add_option('--substack', action='store_true', dest='create_substacks', default=False, help='Create substacks of data-only by removing all artifact/discard frames [default: False]')
#
(options, args) = parser.parse_args() 

#create_substacks = options.create_substacks
uint16 = options.uint16
#nflyback = options.nflyback
#ndiscard = options.ndiscard
#nchannels = options.nchannels
#nvolumes = options.nvolumes
#nslices_full = options.nslices + options.ndiscard


#source = options.source
#session = options.session
#run = options.run
visible = options.visible
if visible:
    displaymin = float(options.displaymin) # For processed, 15000 good
    displaymax = float(options.displaymax) # For processed, 22000 good


tiffs_to_join = [[tiffs1[matchtrials[trial,0]], tiffs2[matchtrials[trial,1]]] for trial in range(matchtrials.shape[0])]

for trial in range(len(tiffs_to_join)):
    currtiffs = tiffs_to_join[trial]
    stack1 = tf.imread(os.path.join(sourcepath, currtiffs[0]))
    stack2 = tf.imread(os.path.join(sourcepath, currtiffs[1]))
    print "STACK 1: ", currtiffs[0] 
    print "stack1 (size, dtype): (%s, %s)" % (str(stack1.shape), stack1.dtype)

    print "STACK 2: ", currtiffs[1] 
    print "stack2 (size, dtype): (%s, %s)" % (str(stack2.shape), stack2.dtype)
 
    
    stack = np.empty((stack1.shape[0], stack1.shape[1], stack1.shape[2], 2), dtype=stack1.dtype)
    stack[:,:,:,0] = stack1
    stack[:,:,:,1] = stack2

    avgstack = np.mean(stack, axis=3, dtype=stack1.dtype)
    avgstack_fn = 'File%03d.tif' % int(trial+1)
    print "AVERAGE STACK fn: ", avgstack_fn
    print "size: ", avgstack.shape
    print "dtype: ", avgstack.dtype

    tf.imsave(os.path.join(savepath, avgstack_fn), avgstack)

    if not avgstack.dtype=='uint16':
        if uint16:
            uint16_stack = img_as_uint(avgstack)
            uint16tiff_fn = 'File%03d_uint16.tif' % int(trial+1)
            tf.imsave(os.path.join(savepath, uint16tiff_fn), uint16_stack)

    if visible:
	ranged = exposure.rescale_intensity(avgstack, in_range=(displaymin, displaymax))
	rangetiff_fn = 'File%03d_visible.tif' % int(trial+1)
	tf.imsave(os.path.join(savepath, rangetiff_fn), ranged)

