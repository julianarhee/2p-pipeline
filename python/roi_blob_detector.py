
# coding: utf-8

# In[1]:


# Adapted from the skimage blob detector example code

from math import sqrt
from skimage import filters
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
from scipy.misc import imread
import matplotlib.pyplot as plt
from skimage.exposure import equalize_adapthist

import os
import cv2 
import numpy as np
import scipy.io
import itertools

import optparse

# get_ipython().magic(u'matplotlib inline')
parser = optparse.OptionParser()
parser.add_option('-S', '--source', action='store', dest='source', default='', help='source dir (parent of slice dir)')
parser.add_option('-s', '--slicedir', action='store', dest='slicedir', default='', help='folder name containing averaged slices to use for blob detection (child of source)')
parser.add_option('-M', '--maxsigma', action='store', dest='max_sigma', default=2.0, help='Max val for blob detector [default: 2.0]')
parser.add_option('-m', '--minsigma', action='store', dest='min_sigma', default=0.2, help='Min val for blob detector [default: 0.2]')
parser.add_option('-t', '--threshold', action='store', dest='blob_threshold', default=0.005, help='Threshold for blob detector [default: 0.005]')
parser.add_option('-H', '--histkernel', action='store', dest='hist_kernel', default=10., help='Kernel size for histogram equalization step [default: 10]')
parser.add_option('-G', '--gauss', action='store', dest='gaussian_sigma', default=1, help='Sigma for initial gaussian blur[default: 1]')
(options, args) = parser.parse_args() 


nslices = int(options.nslices)
max_sigma_val = float(options.max_sigma)
min_sigma_val = float(otions.min_sigma)
blob_threshold = float(options.blob_threshold)
hist_kernel = float(options.hist_kernel)
gaussian_sigma = float(options.gauss_sigma)

source_dir = options.source
avg_vol_dir = options.slicedir

# In[2]:

# nslices = 12 #30

# max_sigma_val = 2
# min_sigma_val = 0.2

# source_dir = '/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/average_volumes'
# avg_vol_dir = 'avg_frames_conditions_channel01'

# source_dir = '/nas/volume1/2photon/RESDATA/TEFO/20161218_CE024/retinotopy5/DATA'
# avg_vol_dir = 'average_slices' #'avg_frames_conditions_channel01'



avg_slices = os.listdir(os.path.join(source_dir, avg_vol_dir))
avg_slices = [i for i in avg_slices if i.endswith('.tif')]
print "N slices: ", len(avg_slices)


# In[3]:




# In[4]:


# Read the image

currslice = 5 #13 #20
image = cv2.imread(os.path.join(source_dir, avg_vol_dir, avg_slices[currslice]), -1)
print "Slice image shape: ", image.shape


# In[5]:


plt.imshow(image, cmap='gray'); plt.axis('off'); plt.title('average slice %i' % currslice); plt.show()


# In[6]:


# Gaussian blur
#gaussian_sigma = 1
image_gaussian = filters.gaussian(image, sigma=gaussian_sigma)


# In[7]:


plt.imshow(image_gaussian, cmap='gray'); plt.axis('off'); plt.title('Gaussian blur, sigma %i' % gaussian_sigma); plt.show()


# In[8]:


# CLAHE
# hist_kernel = 10
image_processed = equalize_adapthist(image_gaussian, kernel_size=hist_kernel)


# In[9]:


plt.imshow(image_processed, cmap='gray'); plt.axis('off'); plt.imshow('Equalize hist, kernel %i' % hist_kernel); plt.show()


# In[10]:



# In[11]:


# Laplacian of Gaussians
# blobs_log = blob_log(image_processed, max_sigma=6, min_sigma=3, threshold=.01)
# blob_threshold = 0.005
blobs_log = blob_log(image_processed, max_sigma=max_sigma_val, min_sigma=min_sigma_val, threshold=blob_threshold)

blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2) # Compute radii in the 3rd column.


# In[12]:


# Difference of Gaussians
blobs_dog = blob_dog(image_processed, max_sigma=max_sigma_val, min_sigma=min_sigma_val, threshold=blob_threshold)
blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)


# In[13]:


# Plot that shit

blobs_list = [blobs_log, blobs_dog]
colors = ['yellow', 'red']
titles = ['Laplacian of Gaussian', 'Difference of Gaussian']
sequence = zip(blobs_list, colors, titles)

fig, axes = plt.subplots(1, 2, figsize=(20, 10), sharex=True, sharey=True,
                         subplot_kw={'adjustable': 'box-forced'})
ax = axes.ravel()

for idx, (blobs, color, title) in enumerate(sequence):
    ax[idx].set_title(title)
    ax[idx].imshow(image, interpolation='nearest', cmap='gray')
    for blob in blobs:
        y, x, r = blob
        c = plt.Circle((x, y), r, color=color, linewidth=1, fill=False)
        ax[idx].add_patch(c)
    ax[idx].set_axis_off()

plt.tight_layout()

imname = 'Slice%02d_ROIs_%s.png' % (currslice, avg_vol_dir)
print imname

roi_dir = 'ROIs_%s' % avg_vol_dir
if not os.path.exists(os.path.join(source_dir, roi_dir)):
    os.mkdir(os.path.join(source_dir, roi_dir))
    
plt.savefig(os.path.join(source_dir, roi_dir, imname))

plt.show()


# ## Process all slices:

# In[14]:



rois = dict()

for currslice in range(nslices):

    currslice_name = 'slice%i' % int(currslice+1)
    print currslice_name
    rois[currslice_name] = dict()

    # Read the image
    image = cv2.imread(os.path.join(source_dir, avg_vol_dir, avg_slices[currslice]), -1)
    print image.shape

    # Gaussian blur
    image_gaussian = filters.gaussian(image, sigma=1)

    # CLAHE
    image_processed = equalize_adapthist(image_gaussian, kernel_size=10)

    # Laplacian of Gaussians
    # blobs_log = blob_log(image_processed, max_sigma=6, min_sigma=3, threshold=.01)
    blobs_log = blob_log(image_processed, max_sigma=max_sigma_val, min_sigma=min_sigma_val, threshold=.005)

    blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2) # Compute radii in the 3rd column.

    # Difference of Gaussians
    blobs_dog = blob_dog(image_processed, max_sigma=max_sigma_val, min_sigma=min_sigma_val, threshold=.005)
    blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)

    # Save in array
    rois['slice%i' % (currslice+1)]['LoG'] = blobs_log
    rois['slice%i' % (currslice+1)]['DoG'] = blobs_dog


    # Plot that shit

    blobs_list = [blobs_log, blobs_dog]
    colors = ['yellow', 'red']
    titles = ['Laplacian of Gaussian', 'Difference of Gaussian']
    sequence = zip(blobs_list, colors, titles)

    fig, axes = plt.subplots(1, 2, figsize=(20, 10), sharex=True, sharey=True,
                             subplot_kw={'adjustable': 'box-forced'})
    ax = axes.ravel()

    for idx, (blobs, color, title) in enumerate(sequence):
        ax[idx].set_title(title)
        ax[idx].imshow(image, interpolation='nearest', cmap='gray')
        for blob in blobs:
            y, x, r = blob
            c = plt.Circle((x, y), r, color=color, linewidth=1, fill=False)
            ax[idx].add_patch(c)
        ax[idx].set_axis_off()

    plt.tight_layout()

    imname = 'Slice%02d_ROIs_%s.png' % (currslice+1, avg_vol_dir)
    print imname

    roi_dir = 'ROIs_%s' % avg_vol_dir
    if not os.path.exists(os.path.join(source_dir, roi_dir, 'images')):
        os.mkdir(os.path.join(source_dir, roi_dir))

    plt.savefig(os.path.join(source_dir, roi_dir, imname))

    # plt.show()
    


# In[16]:


roi_mat_fn = 'rois_%s.mat' % avg_vol_dir
scipy.io.savemat(os.path.join(source_dir, roi_dir, roi_mat_fn), mdict=rois)
print "ROI save dir: ", os.path.join(source_dir, roi_mat_fn)


# In[20]:


# print rois.keys()[0]
# print int(rois.keys()[0][5:])


# In[21]:


all_centroids_log = [(rois[slicekey]['LoG'][:,1], rois[slicekey]['LoG'][:,0], int(slicekey[5:])) for slicekey in rois.keys()]


# In[22]:


centroids = dict()
centroids['LoG'] = []
centroids['DoG'] = []

for slicekey in rois.keys():
    centroids1 = [[rois[slicekey]['LoG'][r,1], rois[slicekey]['LoG'][r,0], int(slicekey[5:])] for r in range(rois[slicekey]['LoG'].shape[0])]
    centroids2 = [[rois[slicekey]['DoG'][r,1], rois[slicekey]['DoG'][r,0], int(slicekey[5:])] for r in range(rois[slicekey]['DoG'].shape[0])]

    centroids['LoG'].append(centroids1)
    centroids['DoG'].append(centroids2)


# In[23]:


centroids['LoG'] = list(itertools.chain.from_iterable(centroids['LoG']))
centroids['DoG'] = list(itertools.chain.from_iterable(centroids['DoG']))


# In[24]:


print "N ROIs: ", len(centroids['LoG'])


# In[25]:



centroids_mat_fn = 'centroids_%s.mat' % avg_vol_dir
scipy.io.savemat(os.path.join(source_dir, roi_dir, centroids_mat_fn), mdict=centroids)
print "CENTROID save dir: ", os.path.join(source_dir, centroids_mat_fn)


# In[ ]:




