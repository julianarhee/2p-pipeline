#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import matplotlib as mpl
mpl.use('agg')

import glob
import os
import cv2
import numpy as np
import tifffile as tf
import pylab as pl

def get_gradient(im) :
    # Calculate the x and y gradients using Sobel operator
    grad_x = cv2.Sobel(im,cv2.CV_32F,1,0,ksize=3)
    grad_y = cv2.Sobel(im,cv2.CV_32F,0,1,ksize=3)

    # Combine the two gradients
    grad = cv2.addWeighted(np.absolute(grad_x), 0.5, np.absolute(grad_y), 0.5, 0)
    return grad

#%%
def uint16_to_RGB(img):
    im = img.astype(np.float64)/img.max()
    im = 255 * im
    im = im.astype(np.uint8)
    rgb = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    return rgb


#%%
def align_sample_to_ref(ref, img):

    # Allocate space for aligned image
    sample_aligned = np.zeros((height,width), dtype=ref.dtype) #dtype=np.uint8 )
    
    # Define motion model
    warp_mode = cv2.MOTION_HOMOGRAPHY
    
    # Set the warp matrix to identity.
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else :
        warp_matrix = np.eye(2, 3, dtype=np.float32)
    
    # Set the stopping criteria for the algorithm.
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000,  1e-6)
    
    print "WARPING!"
    sample = img.copy()
    
    # Warp REFERENCE image into sample:
    (cc, warp_matrix) = cv2.findTransformECC (get_gradient(ref), get_gradient(sample), warp_matrix, warp_mode, criteria)
        
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        # Use Perspective warp when the transformation is a Homography
        sample_aligned = cv2.warpPerspective (sample, warp_matrix, (width,height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        mode_str = 'MOTION_HOMOGRAPHY'
    else :
        # Use Affine warp when the transformation is not a Homography
        sample_aligned = cv2.warpAffine(sample, warp_matrix, (width, height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
        mode_str = 'WARP_AFFINE'
        
    return sample_aligned

#%%

rootdir = '/Volumes/coxfs01/2p-data'
animalid = 'CE077'
session_list = ['20180515', '20180516', '20180523', '20180602', '20180609']
acquisition = 'FOV1'
stimulus = 'gratings'

img_paths = []
for session in session_list:
    ipath = glob.glob(os.path.join(rootdir, animalid, session, \
                                   '%s*' % acquisition, '%s*' % stimulus, \
                                   'processed', 'processed001*', \
                                   'mcorrected_*mean_deinterleaved', \
                                   'Channel01', 'File001', '*.tif'))
    img_paths.append(ipath[0])
print img_paths

# Find the width and height of the color image
ref_session = '20180523'
ref_path = [ipath for ipath in img_paths if ref_session in ipath][0]
print "REFERENCE: %s" % ref_path

ref = tf.imread(ref_path)

height, width = ref.shape

#%%

nsessions = len(session_list)
stack = np.empty((nsessions, height, width), dtype=ref.dtype)
for ix, img_path in enumerate(img_paths):
    if img_path == ref_path:
        stack[ix, :, :] = ref
    else:
        print img_path
        img = tf.imread(img_path)
        sample_aligned = align_sample_to_ref(ref, img)
        stack[ix, :, :] = sample_aligned
    
    
#%% Save stacked mean images to .tif:
    
output_dir = os.path.join(rootdir, animalid, 'test_registration')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
tf.imsave(os.path.join(output_dir, 'warp_merge_r%s_%s_%s.tif' % (ref_session, session_list[0], session_list[-1])), stack)


#%% Save best 3 as RGB channels:
    
test_merge = np.zeros((height, width, 3), dtype=np.uint8)
test_merge[:,:,0] = uint16_to_RGB(ref)[:,:,0]
test_merge[:,:,1] = uint16_to_RGB(stack[3,:])[:,:,0]
test_merge[:,:,2] = uint16_to_RGB(stack[4,:])[:,:,0]

#%
pl.figure()
pl.imshow(test_merge)
pl.savefig(os.path.join(output_dir, 'RGB_merge.png'))
pl.close()


#
