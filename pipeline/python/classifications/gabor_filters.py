#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 12:25:43 2019

@author: julianarhee
"""

import cv2
import sys
import glob
import os
import math

import numpy as np
import pylab as pl
from skimage.util import img_as_float



# Get image paths:
stimulus_dir = '/home/julianarhee/Repositories/protocols/physiology/stimuli/images'
#object_list = ['D1', 'M14', 'M27', 'M53', 'M66', 'M9', 'M93', 'D2']
object_list = ['D1', 'M53', 'D2']
image_paths = []
for obj in object_list:
    stimulus_type = 'Blob_%s_Rot_y_fine' % obj
    image_paths.extend(glob.glob(os.path.join(stimulus_dir, stimulus_type, '*_y0.png')))
print("%i images found for %i objects" % (len(image_paths), len(object_list)))


# Load images and mask:
shrink = (slice(0, None, 3), slice(0, None, 3))
x_vals = []
y_vals = []
image_list = []
for i, img_fpath in enumerate(image_paths):
    print img_fpath
    img = cv2.imread(img_fpath)
    
    # Mask of non-black pixels (assuming image has a single channel).
    img = img[:, :, 0]
    mask = img > 0
    
    # Coordinates of non-black pixels.
    coords = np.argwhere(mask)
    
    # Bounding box of non-black pixels.
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1   # slices are exclusive at the top
    
    # Get the contents of the bounding box.
    x_vals.append((x0, x1))
    y_vals.append((y0, y1))
    
    image_list.append(img_as_float(img))
    
x0 = min([x[0] for x in x_vals])
x1 = max([x[1] for x in x_vals])
y0 = min([y[0] for y in y_vals])
y1 = max([y[1] for y in y_vals])
fig, axes = pl.subplots(1,3)
for ix, img in enumerate(image_list):
    cropped = img[x0-2:x1+2, y0-2:y1+2]
    image_list[ix] = cropped
    axes[ix].imshow(cropped)

    
image_names = tuple(object_list) #('brick', 'grass', 'wall')
images = tuple(image_list) #(brick, grass, wall)


#%%

 
def build_filters(ksize=31, thetas=None, sigmas=None, lambdas=None):
    filters = []
    params = []
    
    if thetas is None:
        thetas = np.linspace(0, np.pi, num=4, endpoint=False)
    if sigmas is None:
        sigmas = [4.0]
    if lambdas is None:
        lambdas = [10.0]
        
    ksize = 31
#    sigma = 4.0
#    lambda_f = 10.0
    gamma = 1
    psi = 0
    
    for theta in thetas: #np.arange(0, np.pi, np.pi / 16):
        print(np.rad2deg(theta))
        for lambda_f in lambdas:
            for sigma in sigmas:
            #for lambda_f in lambdas:
                kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambda_f, gamma, psi, ktype=cv2.CV_32F)
                kern /= math.sqrt((kernel * kernel).sum()) # Normalize to have L2 norm #1.5*kern.sum()
                filters.append(kern)
                param_str = 'theta=%d,\nsigma=%.2f,\nlambda=%.2f' % (theta * 180 / np.pi, sigma, lambda_f)
                params.append(param_str)
                
    return filters, params
     
def process(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum
 
    
thetas = np.linspace(0, np.pi, num=4, endpoint=False)
sfs = [0.1, 0.5]
sigmas = [3,4,5]
lambdas = [1./s for s in sfs]
filters, params = build_filters(thetas=thetas, sigmas=sigmas, lambdas=lambdas)

nrows = len(thetas) #len(kernel_params)+1
ncols = len(sigmas) * len(lambdas) #len(object_list)+1

pl.figure()
gs = mpl.gridspec.GridSpec(nrows, ncols)
gs.update(wspace=0.1, hspace=0.1, left=0.2, right=0.8, bottom=0.1, top=0.9) 
pl.gray()

fig.suptitle('Image responses for Gabor filter kernels', fontsize=12)


#for label, (kernel, powers), ax_row in zip(kernel_params, results, axes[1:]):
for ai, (label, kernel) in enumerate(zip(params, filters)): #, (kernel, powers)) in enumerate(zip(kernel_params, results)):

    # Plot Gabor kernel
    #ax_row = gs[ai+1:]
    ax = pl.subplot(gs[ai])
    #ax = ax_row[0]
    ax.imshow(np.real(kernel), interpolation='nearest')
    ax.set_title(label, fontsize=6)
    ax.set_xticks([])
    ax.set_yticks([])
#    
#    if nrows==1:
#        ax.set_xlabel(label, fontsize=7, rotation=0, horizontalalignment='center')
#    else:
#        if nrows>1 and ncols>1:
#            if ai % ncols == 0:
#                ax.set_ylabel(label, fontsize=7, rotation=0, horizontalalignment='right', y=0.25)
#            if ai >= (nrows-1)*ncols:
#                ax.set_xlabel(label, fontsize=7, rotation=0, horizontalalignment='center')
#                
#    ax.set_xticks([])
#    ax.set_yticks([])


img_fpath = image_paths[0]
res1 = process(img, filters)


img = cv2.imread(img_fpath)


res1 = process(img, filters)
cv2.imshow('result', res1)
cv2.waitKey(0)
cv2.destroyAllWindows()


#%%

import matplotlib.pyplot as pl
import numpy as np
from scipy import ndimage as ndi

from skimage.filters import gabor_kernel


def compute_feats(image, kernels):
    feats = np.zeros((len(kernels), 2), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(image, kernel, mode='wrap')
        feats[k, 0] = filtered.mean()
        feats[k, 1] = filtered.var()
    return feats


def match(feats, ref_feats):
    min_error = np.inf
    min_i = None
    for i in range(ref_feats.shape[0]):
        error = np.sum((feats - ref_feats[i, :])**2)
        if error < min_error:
            min_error = error
            min_i = i
    return min_i



# prepare filter bank kernels
kernels = []
for theta in range(4):
    theta = theta / 4. * np.pi
    for sigma in (1, 3):
        for frequency in (0.05, 0.25):
            kernel = np.real(gabor_kernel(frequency, theta=theta,
                                          sigma_x=sigma, sigma_y=sigma))
            kernels.append(kernel)

# prepare reference features
ref_feats = np.zeros((3, len(kernels), 2), dtype=np.double)
ref_feats[0, :, :] = compute_feats(images[0], kernels)
ref_feats[1, :, :] = compute_feats(images[1], kernels)
ref_feats[2, :, :] = compute_feats(images[2], kernels)

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

print('Rotated images matched against references using Gabor filter banks:')

for obj_name, obj_image in zip(image_names, images):
    
    feats = compute_feats(ndi.rotate(obj_image, 190, reshape=False), kernels)
    print('original: %s, rotated: 30deg, match result: %s' % ( obj_name, image_names[match(feats, ref_feats)] ) )
    feats = compute_feats(ndi.rotate(obj_image, angle=70, reshape=False), kernels)
    print('original: %s, rotated: 70deg, match result: %s' % ( obj_name, image_names[match(feats, ref_feats)] ) )
    feats = compute_feats(ndi.rotate(obj_image, angle=145, reshape=False), kernels)
    print('original: %s, rotated: 145deg, match result: %s' % ( obj_name, image_names[match(feats, ref_feats)] ) )




def power(image, kernel):
    # Normalize images for better comparison.
    image = (image - image.mean()) / image.std()
    return np.sqrt(ndi.convolve(image, np.real(kernel), mode='wrap')**2 +
                   ndi.convolve(image, np.imag(kernel), mode='wrap')**2)



# Plot a selection of the filter bank kernels and their responses.
results = []
kernel_params = []
sigma=3
for theta in range(4):
    theta = theta / 4. * np.pi
    print(np.rad2deg(theta))
    
    for frequency in (0.1, 0.5):
        kernel = gabor_kernel(frequency, theta=theta, sigma_x=sigma, sigma_y=sigma)
        params = 'theta=%d,\nfrequency=%.2f' % (theta * 180 / np.pi, frequency)
        kernel_params.append(params)
        # Save kernel and the power image for each image
        results.append((kernel, [power(img, kernel) for img in images]))

#fig, axes = pl.subplots(nrows=len(kernel_params)+1, ncols=len(object_list)+1, figsize=(5, 10), aspect='auto')

import matplotlib as mpl
nrows = len(kernel_params)+1
ncols = len(object_list)+1


pl.figure()
gs = mpl.gridspec.GridSpec(nrows, ncols)
gs.update(wspace=0.1, hspace=0.1, left=0.2, right=0.8, bottom=0.1, top=0.9) 
pl.gray()

fig.suptitle('Image responses for Gabor filter kernels', fontsize=12)

ax = pl.subplot(gs[0, 0])
ax.axis('off')

# Plot original images
#for label, img, gspec in zip(image_names, images, gs[0, 1:]): #axes[0][1:]):
for ai, (label, img) in enumerate(zip(image_names, images)): #axes[0][1:]):

    ax = pl.subplot(gs[0, ai+1])
    
    ax.imshow(img)
    ax.set_title(label, fontsize=9)
    ax.axis('off')

#for label, (kernel, powers), ax_row in zip(kernel_params, results, axes[1:]):
for ai, (label, (kernel, powers)) in enumerate(zip(kernel_params, results)):

    # Plot Gabor kernel
    #ax_row = gs[ai+1:]
    ax = pl.subplot(gs[ai+1, 0])
    #ax = ax_row[0]
    ax.imshow(np.real(kernel), interpolation='nearest')
    ax.set_ylabel(label, fontsize=7, rotation=0,horizontalalignment='right', y=0.25)
    ax.set_xticks([])
    ax.set_yticks([])

    # Plot Gabor responses with the contrast normalized for each filter
    vmin = np.min(powers)
    vmax = np.max(powers)
    #for patch, ax in zip(powers, ax_row[1:]):
    for oc, patch in enumerate(powers): #zip(powers, ax_row[1:]):
        ax = pl.subplot(gs[ai+1, oc+1])
        ax.imshow(patch, vmin=vmin, vmax=vmax)
        ax.axis('off')


