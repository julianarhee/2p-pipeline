#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 13:49:03 2018

@author: juliana
"""
import os
import json
import h5py
import traceback
import datetime
import optparse
import cv2
import pylab as pl
import numpy as np
import tifffile as tf

from pipeline.python.rois.utils import load_RID, get_source_paths, check_mc_evaluation, get_info_from_tiff_dir
from pipeline.python.rois.get_rois import standardize_rois, save_roi_params

#%%
def get_gradient(im) :
    # Calculate the x and y gradients using Sobel operator
    grad_x = cv2.Sobel(im,cv2.CV_32F,1,0,ksize=3)
    grad_y = cv2.Sobel(im,cv2.CV_32F,0,1,ksize=3)

    # Combine the two gradients
    grad = cv2.addWeighted(np.absolute(grad_x), 0.5, np.absolute(grad_y), 0.5, 0)
    return grad


def uint16_to_RGB(img):
    im = img.astype(np.float64)/img.max()
    im = 255 * im
    im = im.astype(np.uint8)
    rgb = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    return rgb

#%%
#rootdir = '/mnt/odyssey'
#animalid = 'CE074'
#session = '20180215'
#acquisition = 'FOV1_zoom1x_V1'
#run = 'blobs'
#roi_id = 'rois024'


#%%

parser = optparse.OptionParser()

# PATH opts:
parser.add_option('-D', '--root', action='store', dest='rootdir', default='/nas/volume1/2photon/data', help='data root dir (root project dir containing all animalids) [default: /nas/volume1/2photon/data, /n/coxfs01/2pdata if --slurm]')
parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')
parser.add_option('-S', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID')
parser.add_option('--default', action='store_true', dest='default', default=False, help="Use all DEFAULT params, for params not specified by user (no interactive)")
parser.add_option('--slurm', action='store_true', dest='slurm', default=False, help="set if running as SLURM job on Odyssey")
parser.add_option('-r', '--roi-id', action='store', dest='roi_id', default='', help="ROI ID for rid param set to use (created with set_roi_params.py, e.g., rois001, rois005, etc.)")


parser.add_option('-z', '--zproj', action='store', dest='zproj_type', default="mean", help="zproj to use for display [default: mean]")
parser.add_option('-C', '--coreg-path', action="store",
                  dest="coreg_results_path", default=None, help="Path to coreg results if standardizing ROIs only")
parser.add_option('--par', action="store_true",
                  dest='multiproc', default=False, help="Use mp parallel processing to extract from tiffs at once, only if not slurm")
parser.add_option('--format', action="store_true",
                  dest='format_only', default=False, help="Only format ROIs to standard (already extracted).")

(options, args) = parser.parse_args()

if options.slurm is True:
    if 'coxfs01' not in options.rootdir:
        options.rootdir = '/n/coxfs01/2p-data'

#%%

rootdir = options.rootdir
animalid = options.animalid
session = options.session
roi_id = options.roi_id
slurm = options.slurm
if slurm is True and 'coxfs01' not in rootdir:
    rootdir = '/n/coxfs01/2p-data'
auto = options.default


#%%
session_dir = os.path.join(rootdir, animalid, session)

rdict_path = os.path.join(session_dir, 'ROIs', 'rids_%s.json' % session)
with open(rdict_path, 'r') as f:
    rdict = json.load(f)

RID = load_RID(session_dir, roi_id)

#%%

# Set output paths:
warp_output_dir = os.path.join(RID['DST'], 'warp_results')
if not os.path.exists(warp_output_dir):
    os.makedirs(warp_output_dir)


# Load mean image of REFERENCE:
src_tiff_dir = RID['PARAMS']['options']['source']['tiff_dir']
src_proj_dir = [os.path.join(os.path.split(src_tiff_dir)[0], d) for d in os.listdir(os.path.split(src_tiff_dir)[0]) if
                    '_mean_slices' in d and os.path.split(src_tiff_dir)[-1] in d][0]
ref_img_dir = os.path.join(src_proj_dir,
                                'Channel%02d' % RID['PARAMS']['options']['source']['ref_channel'],
                                'File%03d' % RID['PARAMS']['options']['source']['ref_file'])
ref_img_path = [os.path.join(ref_img_dir, t) for t in os.listdir(ref_img_dir) if t.endswith('tif')][0]

# Load mean image of SAMPLE to warp to:
rid_tiff_dir = RID['SRC']
rid_proj_dir = [os.path.join(os.path.split(rid_tiff_dir)[0], d) for d in os.listdir(os.path.split(rid_tiff_dir)[0]) if
                    '_mean_slices' in d and os.path.split(rid_tiff_dir)[-1] in d][0]
rid_img_dir = os.path.join(rid_proj_dir,
                                 'Channel%02d' % RID['PARAMS']['options']['ref_channel'],
                                 'File%03d' % RID['PARAMS']['options']['ref_file'])
rid_img_path = [os.path.join(rid_img_dir, t) for t in os.listdir(rid_img_dir) if t.endswith('tif')][0]

#%%
ref = tf.imread(ref_img_path)
img = tf.imread(rid_img_path)

#%% Load masks:
mask_path = os.path.join(RID['PARAMS']['options']['source']['roi_dir'], 'masks.hdf5')

maskfile = h5py.File(mask_path, 'r')
ref_key = 'File%03d' % RID['PARAMS']['options']['source']['ref_file']
masks = maskfile[ref_key]['masks']
if type(masks) == h5py._hl.group.Group:
    if len(masks.keys()) == 1:  # SINGLE SLICE
        masks = np.array(masks['Slice01']).T


# Find the width and height of the color image
sz = img.shape
print sz
height = sz[0]
width = sz[1]

#%% WARP REF TO SAMPLE:

# Allocate space for aligned image
ref_aligned = np.zeros((height,width), dtype=ref.dtype) #dtype=np.uint8 )

# Define motion model
warp_mode = cv2.MOTION_HOMOGRAPHY

# Set the warp matrix to identity.
if warp_mode == cv2.MOTION_HOMOGRAPHY :
    warp_matrix = np.eye(3, 3, dtype=np.float32)
else :
    warp_matrix = np.eye(2, 3, dtype=np.float32)

# Set the stopping criteria for the algorithm.
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000,  1e-6)

sample = img.copy()
# Warp REFERENCE image into sample:
(cc, warp_matrix) = cv2.findTransformECC (get_gradient(sample), get_gradient(ref),warp_matrix, warp_mode, criteria)
if warp_mode == cv2.MOTION_HOMOGRAPHY :
    # Use Perspective warp when the transformation is a Homography
    ref_aligned = cv2.warpPerspective (ref, warp_matrix, (width,height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    mode_str = 'MOTION_HOMOGRAPHY'
else :
    # Use Affine warp when the transformation is not a Homography
    ref_aligned = cv2.warpAffine(ref, warp_matrix, (width, height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
    mode_str = 'WARP_AFFINE'

#%% Visualize img diff with and without warp:
pl.figure()
pl.subplot(2,2,1); pl.imshow(ref); pl.title('ref'); pl.axis('off')
pl.subplot(2,2,2); pl.imshow(img); pl.title('sample'); pl.axis('off')
pl.subplot(2,2,3); pl.imshow(img-ref); pl.title('difference'); pl.axis('off')
pl.subplot(2,2,4); pl.imshow(img-ref_aligned); pl.title('warp + diff'); pl.axis('off')

figname = 'ref_to_sample_warp.png'
pl.savefig(os.path.join(warp_output_dir, figname))
pl.close()

# Check out one mask:
#fig = pl.figure()
#ax = fig.add_subplot(1,1,1)
#pl.imshow(ref); pl.axis('off')
#masktmp = masks[:,:,0]
#msk = masktmp.copy()
#msk[msk==0] == np.nan
#ax.imshow(msk, interpolation='None', alpha=0.2, cmap=pl.cm.hot)


# Show original ROIs:
#refRGB = uint16_to_RGB(ref_std)
#imRGB = uint16_to_RGB(sample)
#
#pl.figure()
#for ridx in range(nrois):
#    roinum = ridx + 1
#    orig = masks[:,:,ridx].copy().astype('uint8')
#    ret,thresh = cv2.threshold(orig,.5,255,0)
#    orig2,contours,hierarchy = cv2.findContours(thresh, 1, 2)
#    cv2.drawContours(refRGB, contours, 0, (0,255,0), 1)
#    pl.imshow(refRGB)
#for ridx in range(nrois):
#    orig = masks[:,:,ridx].copy().astype('float')
#    orig[orig == 0] = np.nan
#    pl.imshow(orig, interpolation='None', alpha=0.2, cmap=pl.cm.YlGn_r)
#pl.axis('off')
#pl.title('ref: orig roi contours')


#%% Warp masks with same transform:
masks_aligned = np.zeros(masks.shape, dtype=masks.dtype)
nrois = masks.shape[-1]
for r in xrange(0, nrois):
    masks_aligned[:,:,r] = cv2.warpPerspective (masks[:,:,r], warp_matrix, (width,height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

#%% Save WARP info:
dtstamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
warp_filepath = os.path.join(warp_output_dir, 'warp_results.hdf5')
warp_file = h5py.File(warp_filepath, 'w')
try:
    warp_file.attrs['roi_id'] = RID['roi_id']
    warp_file.attrs['creation_time'] = dtstamp
    warp_file.attrs['criteria'] = criteria
    warp_file.attrs['mode'] = mode_str

    # Save warp matrix output:
    m = warp_file.create_dataset('warp_matrix', warp_matrix.shape, warp_matrix.dtype)
    m[...] = warp_matrix
    m.attrs['width'] = width
    m.attrs['height'] = height

    # Save sample img and info:
    samd = warp_file.create_dataset('sample', sample.shape, sample.dtype)
    samd[...] = sample
    samd.attrs['source'] = rid_img_path

    # Save ref img and info:
    refd = warp_file.create_dataset('reference', ref.shape, ref.dtype)
    refd[...] = ref
    refd.attrs['source'] = ref_img_path

    # Save orig mask info:
    md = warp_file.create_dataset('orig_masks', masks.shape, masks.dtype)
    md.attrs['source'] = mask_path
except Exception as e:
    traceback.print_exc()
finally:
    warp_file.close()


#%% Show original, uncorrected, and corrected ROIs on ref and sample:
refRGB = uint16_to_RGB(ref)
imRGB = uint16_to_RGB(sample)
wimRGB = uint16_to_RGB(sample)

fig = pl.figure(figsize=(15,5))
fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.1, wspace=0.1)
ax1 = fig.add_subplot(1,3,1); pl.imshow(refRGB, cmap='gray'); pl.title('ref rois'); pl.axis('off')
ax2 = fig.add_subplot(1,3,2); pl.imshow(imRGB, cmap='gray'); pl.title('sample, orig rois'); pl.axis('off')
ax3 = fig.add_subplot(1,3,3); pl.imshow(imRGB, cmap='gray'); pl.title('sample, warped rois'); pl.axis('off')
for ridx in range(nrois):
    roinum = ridx + 1
    orig = masks[:,:,ridx].copy().astype('uint8')
    # Draw contour for ORIG rois on reference:
    ret,thresh = cv2.threshold(orig,.5,255,0)
    orig2,contours,hierarchy = cv2.findContours(thresh, 1, 2)
    cv2.drawContours(refRGB, contours, 0, (0,255,0), 1)
    ax1.imshow(refRGB)
    # Draw orig ROIs on sample:
    cv2.drawContours(imRGB, contours, 0, (0,255,0), 1)
    ax2.imshow(imRGB)
    # Draw orig ROIs + warped ROIs on sample (i.e., ref rois warped to match sample)
    alig = masks_aligned[:,:,ridx].copy().astype('uint8')
    ret,thresh = cv2.threshold(alig,.5,255,0)
    aligC,contours2,hierarchy = cv2.findContours(thresh, 1, 2)
    cv2.drawContours(wimRGB, contours, 0, (0,255,0), 1)
    cv2.drawContours(wimRGB, contours2, 0, (255,0,0), 1)
    ax3.imshow(wimRGB)
figname = 'aligned_rois.png'
pl.savefig(os.path.join(warp_output_dir, figname))
pl.close()

#%% STANDARDIZE

# Save figure:
fig = pl.figure()
ax = fig.add_subplot(1,1,1)
pl.imshow(sample, cmap='gray')
for ridx in range(nrois):
    masktmp = masks_aligned[:,:,ridx]
    msk = masktmp.copy() #.copy().astype('float')
    msk[msk == 0] = np.nan
    ax.imshow(msk, interpolation='None', alpha=0.2, cmap=pl.cm.Greens_r)
    [ys, xs] = np.where(masktmp>0)
    ax.text(xs[int(round(len(xs)/4))], ys[int(round(len(ys)/4))], str(ridx+1), fontsize=8, weight='light', color='w')
    ax.axis('off')

std_figname = '%s_%s_Slice01_Channel%03d_File%03d_masks.tif' % (RID['roi_id'], RID['rid_hash'], RID['PARAMS']['options']['ref_channel'], RID['PARAMS']['options']['ref_file'])
rid_figdir = os.path.join(RID['DST'], 'figures')
if not os.path.exists(rid_figdir):
    os.makedirs(rid_figdir)

pl.savefig(os.path.join(rid_figdir, std_figname))
pl.close()

#%
# Save masks in standard MANUAL2D format:
mask_outpath = os.path.join(RID['DST'], 'masks.hdf5')
outmasks = h5py.File(mask_outpath, 'w')
try:
    outmasks.attrs['roi_type'] = RID['roi_type']
    outmasks.attrs['roi_id'] = RID['roi_id']
    outmasks.attrs['roi_hash'] = RID['rid_hash']
    outmasks.attrs['animal'] = animalid
    outmasks.attrs['session'] = session
    outmasks.attrs['ref_file'] = RID['PARAMS']['options']['ref_file']
    outmasks.attrs['cretion_date'] = dtstamp
    outmasks.attrs['keep_good_rois'] = True
    outmasks.attrs['ntiffs_in_set'] = 1
    outmasks.attrs['zproj'] = RID['PARAMS']['options']['zproj_type']
    outmasks.attrs['is_3D'] = False

    filegrp = outmasks.create_group('File%03d' % RID['PARAMS']['options']['ref_file'])
    mgroup = filegrp.create_group('masks')
    mgroup.attrs['source'] = rid_img_dir

    savemasks = masks_aligned.T # Re-transpose because manual2D are usually flipped
    slicemasks = mgroup.create_dataset('Slice01', savemasks.shape, savemasks.dtype)
    slicemasks[...] = savemasks
    slicemasks.attrs['source_file'] = rid_img_path
    slicemasks.attrs['nrois'] =  nrois
    slicemasks.attrs['src_roi_idxs'] = np.arange(1, nrois+1)

    zgroup = filegrp.create_group('zproj_img')
    zsliceimg = zgroup.create_dataset('Slice01', sample.shape, sample.dtype)
    zsliceimg.attrs['source_file'] = rid_img_path
except Exception as e:
    traceback.print_exc()
finally:
    outmasks.close()

#
#% Save roi param info
check_motion = RID['PARAMS']['eval']['check_motion']
mcmetric = RID['PARAMS']['eval']['mcmetric']
manual_excluded = RID['PARAMS']['eval']['manual_excluded']

if check_motion is True:
    roi_source_paths, tiff_source_paths, filenames, mc_excluded_tiffs, mcmetrics_filepath = get_source_paths(session_dir, RID, check_motion=check_motion, mcmetric=mcmetric, rootdir=rootdir)
else:
    mc_excluded_tiffs = []

excluded_tiffs = list(set(manual_excluded + mc_excluded_tiffs))
exclude_str = ','.join([str(int(fn[4:])) for fn in excluded_tiffs])
print "TIFFS EXCLUDED:", excluded_tiffs

roiparams = save_roi_params(RID, evalparams={}, keep_good_rois=True, excluded_tiffs=excluded_tiffs, rootdir=rootdir)

