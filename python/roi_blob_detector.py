#!/usr/bin/env python2

'''
Given parameters for blurring, thresholding, etc., this script automatically detects blobs. Adapted from the skimage blob detector example code. Outputs figures displaying detected blobs on each slice. Saves parameters and paths to masks to roiparams.mat.

Example:

python roi_blob_detector.py -S'/nas/volume1/2photon/projects' -E'gratings_phaseMod' -s'20170901_CE054' -A'FOV1_zoom3x' -f'functional_sub' -r3 -m3 -M10 -H100 -t.05 -G3
'''

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



# get_ipython().magic(u'matplotlib inline')
parser = optparse.OptionParser()

# PATH options:

parser.add_option('-S', '--source', action='store', dest='source', default='/nas/volume1/2photon/projects', help='source dir (root project dir containing all expts) [default: /nas/volume1/2photon/projects]')
parser.add_option('-E', '--experiment', action='store', dest='experiment', default='', help='experiment type (parent of session dir)')
parser.add_option('-s', '--sess', action='store', dest='sess', default='', help='session name')
parser.add_option('-A', '--acq', action='store', dest='acquisition', default='', help='acquisition folder')
parser.add_option('-f', '--func', action='store', dest='functional_dir', default='functional', help="folder containing functional tiffs [default: 'functional']")
parser.add_option('-a', '--slicedir', action='store', dest='avgsource', default='', help="folder name from which to get averaged slices, e.g., 'Corrected', 'Corrected_Bidi', 'Parsed' [default: '', or 'source_to_average' field of ref struct]")

# ACQUISITION-specific options:
parser.add_option('--nslices', action='store', dest='nslices', default='', help='N slices containing ROIs')
#parser.add_option('-r', '--reference', action='store', dest='reference', default=1, help='File number to use as refernce [default: 1]')
#parser.add_option('-c', '--channel', action='store', dest='channel', default='1', help='channel to use for ROI detection [default: 1]')

# BLOB param options:
parser.add_option('-M', '--maxsigma', action='store', dest='max_sigma', default=2.0, help='Max val for blob detector [default: 2.0]')
parser.add_option('-m', '--minsigma', action='store', dest='min_sigma', default=0.2, help='Min val for blob detector [default: 0.2]')
parser.add_option('-t', '--threshold', action='store', dest='blob_threshold', default=0.005, help='Threshold for blob detector [default: 0.005]')

# IMAGE processing options:
parser.add_option('-H', '--histkernel', action='store', dest='hist_kernel', default=10., help='Kernel size for histogram equalization step [default: 10]')
parser.add_option('-G', '--gauss', action='store', dest='gaussian_sigma', default=1, help='Sigma for initial gaussian blur[default: 1]')

# SAVE OPTIONS:
parser.add_option('-R', '--roiname', action='store', dest='roi_id', default='', help="unique ID name for current ROI set (ex: 'blobsDoG01')")
#parser.add_option('-C', '--mcname', action='store', dest='mc_id', default='', help="unique ID name of mc-params to use (ex: 'mcparams01')")




(options, args) = parser.parse_args() 


#nslices = options.nslices
#reference_file_idx = options.reference
#signal_channel_idx = options.channel

roi_id = options.roi_id
#mc_id = options.mc_id

max_sigma_val = float(options.max_sigma)
min_sigma_val = float(options.min_sigma)
blob_threshold = float(options.blob_threshold)
hist_kernel = float(options.hist_kernel)
gaussian_sigma = float(options.gaussian_sigma)


source = options.source
experiment = options.experiment
sess = options.sess
acquisition = options.acquisition
functional_subdir = options.functional_dir
avgsource = options.avgsource

acquisition_dir = os.path.join(source, experiment, sess, acquisition)
data_dir = os.path.join(acquisition_dir, functional_subdir, 'DATA')

# Load mcparams.mat:
mcparams = scipy.io.loadmat(os.path.join(data_dir, 'mcparams.mat'))
mc_methods = sorted([m for m in mcparams.keys() if 'mcparams' in m], key=natural_keys)
if len(mc_methods)>1:
    for mcidx,mcid in enumerate(sorted(mc_methods, key=natural_keys)):
        print mcidx, mcid
    mc_method_idx = raw_input('Select IDX of mc-method to use: ')
    mc_method = mc_methods[int(mc_method_idx)]
    print "Using MC-METHOD: ", mc_method
else:
    mc_method = mc_methods[0] 

mcparams = mcparams[mc_method] #mcparams['mcparams01']
reference_file_idx = int(mcparams['ref_file']) 
signal_channel_idx = int(mcparams['ref_channel'])

signal_channel = 'Channel%02d' % int(signal_channel_idx)
reference_file = 'File%03d' % int(reference_file_idx)
if signal_channel_idx==0:
    signal_channel_idx = input('No ref channel found. Enter signal channel idx (1-indexing): ')
if reference_file_idx==0:
    reference_file_idx = input('No ref file found. Enter file idx (1-indexing): ')
    
signal_channel = 'Channel%02d' % int(signal_channel_idx)
reference_file = 'File%03d' % int(reference_file_idx)
print "Specified signal channel is:", signal_channel
print "Selected reference file:", reference_file


# Specify and create directories:
if len(avgsource)==0:
    avgsource = ''
else:
    avgsource = '_%s' % avgsource
    
subdir_str = '{tiffstr}/DATA/Averaged_Slices{avgsource}'.format(avgsource=avgsource, tiffstr=functional_subdir)
# subdir_str = '{tiffstr}/DATA/Averaged_Slices'.format(tiffstr=functional_subdir)

slice_directory = os.path.join(source, experiment, sess, acquisition, subdir_str, signal_channel, reference_file)

# Define output directories:
acquisition_dir = os.path.join(source, experiment, sess, acquisition)
roi_dir = os.path.join(acquisition_dir, 'ROIs')
if not os.path.exists(roi_dir):
    os.mkdir(roi_dir)

existing_rois = sorted(os.listdir(roi_dir), key=natural_keys)

#existing_blob_rois = sorted([r for r in existing_rois if 'LoG' in r or 'DoG' in r], key=natural_keys)
existing_blob_rois = sorted([r for r in existing_rois if os.path.isdir(os.path.join(roi_dir, r))], key=natural_keys)

if len(existing_rois)>0:
    print "Found existing blob ROIs:"
    for ridx,rid in enumerate(sorted(existing_rois, key=natural_keys)):
        print ridx, rid
else:
    print "No existing blob ROIs found"

if roi_id in existing_blob_rois or len(roi_id)==0:
    while True:
        roi_id = raw_input('Enter unique roi_id name:')
        if roi_id in existing_blob_rois:
            print('Exists! Pick a new name.')
            overwrite = raw_input('To overwrite, press Y/n')
            if overwrite is 'Y':
                break
        else:
            break
print "Creating new ROI ID: ", roi_id
    
# user_id = raw_input('Enter IDX for LoG/DoG folders: ')
# if len(user_id)==0:
#     log_folder = 'blobsLoG'
#     dog_folder = 'blobsDoG'
# else:
#     log_folder = 'blobsLoG%i' % int(user_id)
#     dog_folder = 'blobsDoG%i' % int(user_id)
# 
roi_reference_path = os.path.join(acquisition_dir, 'ROIs', 'roiparams.json')

# Check for MC-params to save:
# roiparams = loadmat(os.path.join(acquisition_dir, 'ROIs', 'roiparams.mat'))
# roi_ids = [r for r in roiparams.keys() if '__' not in r]
# for ridx,roiname in enumerate(roi_ids):
#     if params_dict==roiparams[roiname]['params']:
#         new_roiparams = False
#         roiparams_idx = rid + 1
#     else:
#         new_roiparams = True
# if new_roiparams is True:
#     roiparams_idx = len(roi_ids) + 1
#     curr_roi_id = 'roiparams%02d' % int(roiparams_idx) 
#     roiparams[curr_roi_id] = params_dict
# 

curr_roi_dir = os.path.join(acquisition_dir, 'ROIs', roi_id)
if not os.path.exists(curr_roi_dir):
    os.mkdir(curr_roi_dir)

fig_dir = os.path.join(curr_roi_dir, 'figures')
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

mask_dir = os.path.join(curr_roi_dir, 'masks')
if not os.path.exists(mask_dir):
    os.makedirs(mask_dir)

# log_roi_dir = os.path.join(base_roi_dir, 'blobs_LoG')
# if not os.path.exists(log_roi_dir):
#     os.makedirs(log_roi_dir)
# 
# dog_roi_dir = os.path.join(base_roi_dir, 'blobs_DoG')
# if not os.path.exists(dog_roi_dir):
#     os.makedirs(dog_roi_dir)
#     
# log_fig_dir = os.path.join(log_roi_dir,'figures')
# if not os.path.exists(log_fig_dir):
#     os.mkdir(log_fig_dir)
# 
# dog_fig_dir = os.path.join(dog_roi_dir,'figures')
# if not os.path.exists(dog_fig_dir):
#     os.mkdir(dog_fig_dir)
# 
# log_mask_dir = os.path.join(log_roi_dir,'masks')
# if not os.path.exists(log_mask_dir):
#     os.mkdir(log_mask_dir)
# 
# dog_mask_dir = os.path.join(dog_roi_dir,'masks')
# if not os.path.exists(dog_mask_dir):
#     os.mkdir(dog_mask_dir)
# 

# Save parameter names and values used for ROI creation to a record structure 
# to be saved as a MATLAB structure
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
nslices = len(avg_slices)

print "N slices: ", len(avg_slices)
# if len(nslices)==0:
#     nslices = len(avg_slices)
# else:
#     nslices = int(nslices)    
 
#initialize empty dictionaries to populate as we go on
source_paths = np.zeros((nslices,), dtype=np.object)
maskpaths = np.zeros((nslices,), dtype=np.object)
curr_rois = np.zeros((nslices,), dtype=np.object)
nrois = np.zeros((nslices,))

#log_maskpaths = np.zeros((nslices,), dtype=np.object)
#dog_maskpaths = np.zeros((nslices,), dtype=np.object)
#log_rois = np.zeros((nslices,), dtype=np.object)
#dog_rois = np.zeros((nslices,), dtype=np.object)
#log_nrois = np.zeros((nslices,))
#dog_nrois = np.zeros((nslices,))
#
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
    # blobs_log = blob_log(image_processed, max_sigma=max_sigma_val, min_sigma=min_sigma_val, threshold=blob_threshold)
    # blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2) # Compute radii in the 3rd column.

    # Difference of Gaussians
    blobs_dog = blob_dog(image_processed, max_sigma=max_sigma_val, min_sigma=min_sigma_val, threshold=blob_threshold)
    blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)


    # Plot and save figures
    blobs_list = [blobs_dog]
    colors = ['red']
    titles = ['Difference of Gaussian'] 
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
        plt.imshow(image, interpolation='nearest', cmap='gray')
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

    #write image to file
    imname = '%s_%s_Slice%02d_%s_%s_ROI.png' % (sess,acquisition,currslice+1,signal_channel,reference_file)
    #plt.savefig(os.path.join(log_fig_dir, imname))
    #plt.savefig(os.path.join(dog_fig_dir, imname))
    plt.savefig(os.path.join(fig_dir, imname))


    # plt.show()

    #record roi info in structure
    curr_rois[currslice] = blobs_dog
    nrois[currslice] = blobs_dog.shape[0]
#     log_rois[currslice] = blobs_log
#     dog_rois[currslice] = blobs_dog
#     log_nrois[currslice] = blobs_log.shape[0]
#     dog_nrois[currslice] = blobs_dog.shape[0]
# 
    #aggregate LoG masks
#     masks = np.zeros((image.shape[0],image.shape[1],blobs_log.shape[0]))
#     for blob_idx in range(0,blobs_log.shape[0]):
#         masks[:,:,blob_idx] = createCircularMask(image.shape[0],image.shape[1],(blobs_log[blob_idx,1],blobs_log[blob_idx,0]),blobs_log[blob_idx,2])
# 
#     #save to structure
#     rois['masks']=masks
#     #save structure to file, record path in structure
#     mat_filename = '%s_%s_Slice%02d_%s_masks.mat' % (sess,acquisition,currslice+1,signal_channel)
#     mat_filepath = os.path.join(log_roi_dir,'masks', mat_filename)
#     log_maskpaths[currslice] = mat_filepath
#     scipy.io.savemat(mat_filepath, mdict=rois)
# 
    #aggregate DoG masks
    masks = np.zeros((image.shape[0],image.shape[1],blobs_dog.shape[0]))
    for blob_idx in range(0,blobs_dog.shape[0]):
        masks[:,:,blob_idx] = createCircularMask(image.shape[0],image.shape[1],(blobs_dog[blob_idx,1],blobs_dog[blob_idx,0]),blobs_dog[blob_idx,2])

    #save to structure
    rois['masks']=masks
    #save structure to file, record path in structure
    mat_filename = '%s_%s_Slice%02d_%s_masks.mat' % (sess,acquisition,currslice+1,signal_channel)
    mat_filepath = os.path.join(curr_roi_dir,'masks', mat_filename)
    maskpaths[currslice] = mat_filepath
#     mat_filepath = os.path.join(dog_roi_dir,'masks', mat_filename)
#     dog_maskpaths[currslice] = mat_filepath
    scipy.io.savemat(mat_filepath, mdict=rois)

# print(log_maskpaths)
# #populate fields of params strucutre
# roiparams['params']=params_dict
# roiparams['nrois']=log_nrois
# roiparams['roi_info']=log_rois
# roiparams['sourcepaths']=source_paths
# roiparams['maskpaths']=log_maskpaths
# roiparams['maskpath3d']=[]
# 
# #save roiparams structure to file
# scipy.io.savemat(os.path.join(log_roi_dir,'roiparams'), {'roiparams': roiparams})
# 
#do it again for DoG methods
print maskpaths
roiparams['params']=params_dict
roiparams['nrois']=nrois
roiparams['roi_info']=curr_rois
roiparams['sourcepaths']=source_paths
roiparams['maskpaths']=maskpaths
roiparams['maskpath3d']=[]
scipy.io.savemat(os.path.join(curr_roi_dir,'roiparams'), {'roiparams': roiparams})
# print maskpaths
# roiparams['params']=params_dict
# roiparams['nrois']=dog_nrois
# roiparams['roi_info']=dog_rois
# roiparams['sourcepaths']=source_paths
# roiparams['maskpaths']=dog_maskpaths
# roiparams['maskpath3d']=[]
# scipy.io.savemat(os.path.join(dog_roi_dir,'roiparams'), {'roiparams': roiparams})
# 

# Save pertinent roi params for quick views:
if os.path.exists(roi_reference_path):
    with open(roi_reference_path, 'r') as f:
        roiref = json.load(f)
else:
    roiref = dict()

if roi_id not in roiref:
    roiref[roi_id] = dict()

roiref[roi_id]['params'] = params_dict
roiref[roi_id]['source'] = os.path.split(source_paths[0])[0]
roiref[roi_id]['nrois'] = nrois
with open(roi_reference_path, 'w') as f:
    dump(roiref, f, indent=4)

