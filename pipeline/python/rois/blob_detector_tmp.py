#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 15:37:54 2018

@author: juliana
"""

import numpy as np
import tifffile as tf
from scipy.ndimage.filters import gaussian_filter
from skimage import data, img_as_float
from skimage import exposure

# Adapted from the skimage blob detector example code
import tifffile as tf
from math import sqrt
from skimage import filters
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
from scipy.misc import imread
import matplotlib.pyplot as plt
from skimage.exposure import equalize_adapthist
from numpy.core.records import fromarrays

import os
import numpy as np
import scipy.io
import itertools

import optparse
import re
import json
from json_tricks.np import dump, dumps, load, loads

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]


def createCircularMask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask


def plot_img_and_hist(image, axes, bins=256):
    """Plot an image along with its histogram and cumulative histogram.

    """
    #image = img_as_float(image)
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(image, cmap=plt.cm.gray)
    ax_img.set_axis_off()

    # Display histogram
    ax_hist.hist(image.ravel(), bins=bins, histtype='step', color='black')
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yticks([])

    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(image, bins)
    ax_cdf.plot(bins, img_cdf, 'r')
    ax_cdf.set_yticks([])

    return ax_img, ax_hist, ax_cdf

#%%
    
if use_corr:
    
    cn_path = os.path.join('/mnt/odyssey/CE077/20180523/FOV1_zoom1x/caiman_test_data/results', 'Cn.npz')
    d = np.load(cn_path)
    image = d['Cn']
    image[np.isnan(image)] = 0
    print image.shape
    uimg = image.copy()
    
else:
    
    image = tf.imread(tiff_path)
    
    #%
    uimg = ((image/image.max()) * 255).round().astype(np.uint8)

#%%
d1, d2 = image.shape

border_to_0 = 16
#simg = uimg[border_to_0:d2-border_to_0+1, border_to_0:d1-border_to_0+1]

gaussian_sigma = 2

image_gaussian = gaussian_filter(uimg, gaussian_sigma)

hist_kernel = 50 #50#10
image_processed = equalize_adapthist(image_gaussian, kernel_size=hist_kernel)

image_hq = equalize_adapthist(uimg, kernel_size=hist_kernel)

fig, axes = pl.subplots(1,3)
axes[0].imshow(uimg)
axes[1].imshow(image_gaussian); axes[1].set_title('gaussian: %i' % gaussian_sigma)
axes[2].imshow(image_processed); axes[2].set_title('adapt hist: %i' % hist_kernel)



# Difference of Gaussians
max_sigma_val = 3
min_sigma_val = 1
blob_threshold= 0.05 #0.03
blobs_dog = blob_dog(image_processed, max_sigma=max_sigma_val, min_sigma=min_sigma_val, threshold=blob_threshold)
blobs_dog = np.array([b for b in blobs_dog if border_to_0 <= b[0] <= d1-border_to_0 and border_to_0 <= b[1] <= d1-border_to_0])

blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)


# Plot and save figures
blobs_list = [blobs_dog]
colors = ['red']
titles = ['Difference of Gaussian %i' % len(blobs_dog)] 
# blobs_list = [blobs_log, blobs_dog]
# colors = ['yellow', 'red']
# titles = ['Laplacian of Gaussian', 'Difference of Gaussian']
sequence = zip(blobs_list, colors, titles)

#     fig, axes = plt.subplots(1, 2, figsize=(20, 10), sharex=True, sharey=True,
#                              subplot_kw={'adjustable': 'box-forced'})
#     ax = axes.ravel()

fig = plt.figure(figsize=(20, 10))
ax = plt.gca()
for idx, (blobs, color, title) in enumerate(sequence):
    plt.title(title)
    plt.imshow(uimg, interpolation='nearest', cmap='gray')
#         ax[idx].set_title(title)
#         ax[idx].imshow(image, interpolation='nearest', cmap='gray')
    for blob in blobs:
        y, x, r = blob
        c = plt.Circle((x, y), r, color=color, linewidth=1, fill=False) 
        ax.add_patch(c)
        # ax[idx].add_patch(c)
    ax.set_axis_off()  
    # ax[idx].set_axis_off()

plt.tight_layout()


#%%
#
# Display results
fig = plt.figure(figsize=(8, 5))
axes = np.zeros((2, 3), dtype=np.object)
axes[0, 0] = fig.add_subplot(2, 3, 1)
for i in range(1, 3):
    axes[0, i] = fig.add_subplot(2, 3, 1+i, sharex=axes[0,0], sharey=axes[0,0])
for i in range(0, 3):
    axes[1, i] = fig.add_subplot(2, 3, 4+i)

hist_kernels = [25, 50, 100]
for axi, hist_kernel in enumerate(hist_kernels):
    img_adapthist = equalize_adapthist(image_gaussian, kernel_size=hist_kernel)
    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_adapthist, axes[:, axi])
    ax_img.set_title('Adaptive equalization')

