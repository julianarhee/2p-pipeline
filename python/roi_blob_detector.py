
# coding: utf-8


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

def createCircularMask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

# get_ipython().magic(u'matplotlib inline')
parser = optparse.OptionParser()
parser.add_option('-S', '--source', action='store', dest='source', default='', help='source dir (parent of slice dir)')
parser.add_option('-s', '--sess', action='store', dest='sess', default='', help='session name')
parser.add_option('-M', '--maxsigma', action='store', dest='max_sigma', default=2.0, help='Max val for blob detector [default: 2.0]')
parser.add_option('-m', '--minsigma', action='store', dest='min_sigma', default=0.2, help='Min val for blob detector [default: 0.2]')
parser.add_option('-t', '--threshold', action='store', dest='blob_threshold', default=0.005, help='Threshold for blob detector [default: 0.005]')
parser.add_option('-H', '--histkernel', action='store', dest='hist_kernel', default=10., help='Kernel size for histogram equalization step [default: 10]')
parser.add_option('-G', '--gauss', action='store', dest='gaussian_sigma', default=1, help='Sigma for initial gaussian blur[default: 1]')
parser.add_option('--nslices', action='store', dest='nslices', default='', help='N slices containing ROIs')
parser.add_option('-r', '--reference', action='store', dest='reference', default='', help='File number to use as refernce [default: 0]')
parser.add_option('-c', '--channel', action='store', dest='channel', default='1', help='channel to use for ROI detection [default: 1]')

(options, args) = parser.parse_args() 


nslices = options.nslices
max_sigma_val = float(options.max_sigma)
min_sigma_val = float(options.min_sigma)
blob_threshold = float(options.blob_threshold)
hist_kernel = float(options.hist_kernel)
gaussian_sigma = float(options.gaussian_sigma)


source_dir = options.source
sess = options.sess
subdir_str = 'functional/DATA/Averaged_Slices'
print(options.channel)
print(type(options.channel))
signal_ch = 'Channel%02d' % int(options.channel)
file_idx = 'File%03d' % int(options.reference)

#define directories
slice_directory = os.path.join(source_dir,sess,subdir_str,signal_ch,file_idx)
log_roi_dir = os.path.join(source_dir,sess,'ROIs/blobs_LoG/')
if not os.path.exists(log_roi_dir):
    os.mkdir(log_roi_dir)

dog_roi_dir = os.path.join(source_dir,sess,'ROIs/blobs_DoG/')
if not os.path.exists(dog_roi_dir):
    os.mkdir(dog_roi_dir)
    
log_fig_dir = os.path.join(log_roi_dir,'figures')
if not os.path.exists(log_fig_dir):
    os.mkdir(log_fig_dir)

dog_fig_dir = os.path.join(dog_roi_dir,'figures')
if not os.path.exists(dog_fig_dir):
    os.mkdir(dog_fig_dir)

log_mask_dir = os.path.join(log_roi_dir,'masks')
if not os.path.exists(log_mask_dir):
    os.mkdir(log_mask_dir)

dog_mask_dir = os.path.join(dog_roi_dir,'masks')
if not os.path.exists(dog_mask_dir):
    os.mkdir(dog_mask_dir)

#save parameter names and values used for ROI creation
#to a record structure to be saved as a MATLAB structure
roiparams = dict()
params_dict = dict()
params_dict['max_sigma_val'] = max_sigma_val
params_dict['min_sigma_val'] = min_sigma_val
params_dict['blob_threshold'] = blob_threshold
params_dict['hist_kernel'] = hist_kernel
params_dict['gaussian_sigma'] = gaussian_sigma

#TO DO:output a tab-delimited text file with parameters used 


#get averaged slices
avg_slices = os.listdir(slice_directory)
avg_slices = [i for i in avg_slices if i.endswith('.tif')]

print "N slices: ", len(avg_slices)
if len(nslices)==0:
    nslices = len(avg_slices)
else:
    nslices = int(nslices)    

#initialize empty dictionaries to populate as we go on
source_paths = np.zeros((nslices,), dtype=np.object)
log_maskpaths = np.zeros((nslices,), dtype=np.object)
dog_maskpaths = np.zeros((nslices,), dtype=np.object)
log_rois = np.zeros((nslices,), dtype=np.object)
dog_rois = np.zeros((nslices,), dtype=np.object)
log_nrois = np.zeros((nslices,))
dog_nrois = np.zeros((nslices,))

for currslice in range(nslices):
    rois = dict()
    currslice_name = 'slice%i' % int(currslice+1)
    print currslice_name


    #get tif file name
    tiff_path = os.path.join(slice_directory, avg_slices[currslice])
    #save for later
    source_paths[currslice] = tiff_path
    #Read in
    with tf.TiffFile(tiff_path) as tif:
        image = tif.asarray()#numpy array

    print image.shape

    # Gaussian blur
    image_gaussian = filters.gaussian(image, sigma=gaussian_sigma)

    # CLAHE
    image_processed = equalize_adapthist(image_gaussian, kernel_size=hist_kernel)

    # Laplacian of Gaussians
    # blobs_log = blob_log(image_processed, max_sigma=6, min_sigma=3, threshold=.01)
    blobs_log = blob_log(image_processed, max_sigma=max_sigma_val, min_sigma=min_sigma_val, threshold=blob_threshold)

    blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2) # Compute radii in the 3rd column.

    # Difference of Gaussians
    blobs_dog = blob_dog(image_processed, max_sigma=max_sigma_val, min_sigma=min_sigma_val, threshold=blob_threshold)
    blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)


    # Plot and save figures
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

    #write image to file
    imname = '%s_Slice%02d_%s_%s_ROI.png' % (sess,currslice+1,signal_ch,file_idx)
    plt.savefig(os.path.join(log_fig_dir, imname))
    plt.savefig(os.path.join(dog_fig_dir, imname))

    # plt.show()

    #record roi info in structure
    log_rois[currslice] = blobs_log
    dog_rois[currslice] = blobs_dog
    log_nrois[currslice] = blobs_log.shape[0]
    dog_nrois[currslice] = blobs_dog.shape[0]

    #aggregate LoG masks
    masks = np.zeros((image.shape[0],image.shape[1],blobs_log.shape[0]))
    for blob_idx in range(0,blobs_log.shape[0]):
        masks[:,:,blob_idx] = createCircularMask(image.shape[0],image.shape[1],(blobs_log[blob_idx,1],blobs_log[blob_idx,0]),blobs_log[blob_idx,2])

    #save to structure
    rois['masks']=masks
    #save structure to file, record path in structure
    mat_filename = '%s_Slice%02d_%s_masks.mat' % (sess,currslice+1,signal_ch)
    mat_filepath = os.path.join(log_roi_dir,'masks', mat_filename)
    log_maskpaths[currslice] = mat_filepath
    scipy.io.savemat(mat_filepath, mdict=rois)

    #aggregate DoG masks
    masks = np.zeros((image.shape[0],image.shape[1],blobs_dog.shape[0]))
    for blob_idx in range(0,blobs_dog.shape[0]):
        masks[:,:,blob_idx] = createCircularMask(image.shape[0],image.shape[1],(blobs_dog[blob_idx,1],blobs_dog[blob_idx,0]),blobs_dog[blob_idx,2])

    #save to structure
    rois['masks']=masks
    #save structure to file, record path in structure
    mat_filename = '%s_Slice%02d_%s_masks.mat' % (sess,currslice+1,signal_ch)
    mat_filepath = os.path.join(dog_roi_dir,'masks', mat_filename)
    dog_maskpaths[currslice] = mat_filepath
    scipy.io.savemat(mat_filepath, mdict=rois)

print(log_maskpaths)
#populate fields of params strucutre
roiparams['params']=params_dict
roiparams['nrois']=log_nrois
roiparams['roi_info']=log_rois
roiparams['sourcepaths']=source_paths
roiparams['maskpaths']=log_maskpaths
roiparams['maskpath3d']=[]

#save roiparams structure to file
scipy.io.savemat(os.path.join(log_roi_dir,'roiparams'), {'roiparams': roiparams})

#do it again for DoG methods
roiparams['params']=params_dict
roiparams['nrois']=dog_nrois
roiparams['roi_info']=dog_rois
roiparams['sourcepaths']=source_paths
roiparams['maskpaths']=dog_maskpaths
roiparams['maskpath3d']=[]
scipy.io.savemat(os.path.join(dog_roi_dir,'roiparams'), {'roiparams': roiparams})