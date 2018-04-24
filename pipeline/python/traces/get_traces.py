#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Extracted time courses using a set of trace params (TID) defined with set_trace_params.py

Outputs:

    a.  Images of masks for each tif file:

        <TRACEID_DIR>/figures/roi_FILEXXX_SliceXX_<RID_NAME>_<RID_HASH>.png
        -- Masks overlaid on zprojected reference slice img with labeled rois.

    b. HDF5 files containing extracted traces from specified ROI set:

        <TRACEID_DIR>/files/FileXXX_rawtraces_<TRACEID_HASH>.hdf5
        -- Extracted raw traces using specified trace params (and specified ROI set)-- this means, .tif files excluded in ROI set are excluded here, too.
        -- File hierarchy is:

        /FileXXX    - group
            - attrs:
                source_file        : TIF file path from which traces were extracted
                file_id            : filename (i.e., File006, for ex.)
                dims               : (array) dimension as d1 x d2 x nslices x nframes
                masks              : <SESSION_DIR>/ROIs/<RID_DIR>/masks.hdf5 <-- standardized mask file (TODO:  make sure this is always produced)

            /SliceXX - group
                /traces - group
                    /raw - dataset
                        [Txnrois] array
                        - atttrs:
                            nframes : T (num of frames in file)
                            dims    : original dimensions of movie for reshaping (d1xd2)
                    /denoised_nmf - dataset
                        [Txnrois] array
                        - attrs:
                            nb      :  number of background components (relevant only for cNMF)
                            nr      :  number of rois

                /masks - dataset
                    [d1xd2xnrois] array
                    - attrs:
                        roi_id         : RID name,
                        rid_hash       : RID hash,
                        roi_type       : roi type,
                        nr             : num rois in set,
                        nb             : num background components, if any
                        src_roi_idxs   : original indices of rois in set (some ROIs are subsets of some other ROI set)

                /zproj - dataset
                    [d1xd2] zprojected image
                    - attrs:
                        img_source     : file path to zprojected reference image

                /frames_tsec - dataset
                    [T,] array -- corresponding time in secs for each frame in timecourse
                    --> Note:  this info is extracted from SI META info (preprocessing/get_scanimage_data.py), and saved to run meta info
                    -- runmeta info is saved to:  <RUN_DIR>/<RUN_NAME>.json
                    -- frame times and indices should be re-indexed based on discard frames/skipped frames

    c.  Traces separated for each ROI:

        <TRACEID_DIR>/roi_timecourses_YYYYMMDD_HH_MM_SS_<filehash>.hdf5
        -- Trace arrays split up by ROI.
        -- File hierarchy is:

        /roi00001 - Group (length-5 name of ROI in set, 1-indexed; may include background components as 'bg01', 'bg02', etc.)
            - attrs:
                slice        :  slice on which current roi (or com) is
                roi_img_path :  source of zprojected slice image
                id_in_set    :  ROI id in current ROI set (should match Group name) -- relative to entire set (across slices, too)
                id_in_src    :  ROI id in source from which ROI was originally created (can be different than id_in_set)
                idx_in_slice :  index of ROI in current slice (in case multiple slices exist)

            /mask - dataset
                [d1xd2] array
                - attrs:
                    slice         :  'SliceXX'
                    is_background :  (bool) whether mask image is background (really only relevant for cNMF rois)

            /timecourse - Group
                /raw - dataset
                    [T,] timecourse of roi
                    - attrs:
                        source_file:  path to file from which trace is extracted
                /denoised_nmf - dataset
                    [T,] timecourse of roi
                    - attrs:
                        source_file:  path to file from which trace is extracted

Created on Tue Dec 12 11:57:24 2017

@author: julianarhee
"""

import matplotlib
matplotlib.use('Agg')
import os
import sys
import h5py
import json
import re
import datetime
import optparse
import pprint
import traceback
import time
import skimage
import shutil
import fissa
import cv2
import tifffile as tf
import pylab as pl
import numpy as np
import cPickle as pkl
from skimage import img_as_uint
from pipeline.python.utils import natural_keys, hash_file_read_only, load_sparse_mat, print_elapsed_time, hash_file, replace_root
from pipeline.python.set_trace_params import post_tid_cleanup
from pipeline.python.rois.utils import get_info_from_tiff_dir
from pipeline.python.traces.utils import get_frame_info, get_metric_set
from pipeline.python.paradigm import align_acquisition_events as acq
pp = pprint.PrettyPrinter(indent=4)

#%%

def load_TID(run_dir, trace_id, auto=False):
    run = os.path.split(run_dir)[-1]
    trace_dir = os.path.join(run_dir, 'traces')
    tmp_tid_dir = os.path.join(trace_dir, 'tmp_tids')
    tracedict_path = os.path.join(trace_dir, 'traceids_%s.json' % run)
    try:
        print "Loading params for TRACE SET, id %s" % trace_id
        with open(tracedict_path, 'r') as f:
            tracedict = json.load(f)
        TID = tracedict[trace_id]
        pp.pprint(TID)
    except Exception as e:
        print "No TRACE SET entry exists for specified id: %s" % trace_id
        print "TRACE DIR:", tracedict_path
        try:
            print "Checking tmp trace-id dir..."
            if auto is False:
                while True:
                    tmpfns = [t for t in os.listdir(tmp_tid_dir) if t.endswith('json')]
                    for tidx, tidfn in enumerate(tmpfns):
                        print tidx, tidfn
                    userchoice = raw_input("Select IDX of found tmp trace-id to view: ")
                    with open(os.path.join(tmp_tid_dir, tmpfns[int(userchoice)]), 'r') as f:
                        tmpTID = json.load(f)
                    print "Showing tid: %s, %s" % (tmpTID['trace_id'], tmpTID['trace_hash'])
                    pp.pprint(tmpTID)
                    userconfirm = raw_input('Press <Y> to use this trace ID, or <q> to abort: ')
                    if userconfirm == 'Y':
                        TID = tmpTID
                        break
                    elif userconfirm == 'q':
                        break
        except Exception as e:
            traceback.print_exc()
            print "---------------------------------------------------------------"
            print "No tmp trace-ids found either... ABORTING with error:"
            print e
            print "---------------------------------------------------------------"

    return TID

#%%
def get_mask_info(TID, RID, nslices=1, rootdir='/n/coxfs01/2p-data'):

    mask_path = os.path.join(RID['DST'], 'masks.hdf5')
    excluded_tiffs = TID['PARAMS']['excluded_tiffs']


    maskinfo = dict()
    try:
        maskfile = h5py.File(mask_path, 'r')
        is_3D = maskfile.attrs['is_3D'] in ['True']

        # Identify tiff source for ROIs:
        roidict_path = os.path.join(rootdir, maskfile.attrs['animal'], maskfile.attrs['session'], 'ROIs', 'rids_%s.json' % maskfile.attrs['session'])
        with open(roidict_path, 'r') as f:
            roidict = json.load(f)
        roi_tiff_src = roidict[maskfile.attrs['roi_id']]['SRC']
        if rootdir not in roi_tiff_src:
            roi_tiff_src = replace_root(roi_tiff_src, rootdir, maskfile.attrs['animal'], maskfile.attrs['session'])

        # Check whether ROI tiffs are same src as TRACE ID tiffs:
        trace_tiff_src = TID['SRC']
        if rootdir not in trace_tiff_src:
            trace_tiff_src = replace_root(trace_tiff_src, rootdir, maskfile.attrs['animal'], maskfile.attrs['session'])

        # Get n tiffs from TRACE source:
        ntiffs = len([f for f in os.listdir(trace_tiff_src) if f.endswith('tif')])

        # Get files from which ROIs were extracted in this set:
        maskfiles = maskfile.keys()
        print "MASK FILES:", len(maskfiles)
        if len(maskfiles) == 1:
            ref_file = maskfiles[0]
            single_reference = True
        else:
            ref_file = None
            single_reference = False

        # Get zproj source base dir:
        # For now, assuming preprocessing + motion-correction output of fmt:
        # <...>_ZPROJ_deinterleaved/Channel01/File003 -- only take up to the Channel-dir
        if 'source' not in maskfile[maskfile.keys()[0]].attrs.keys():
            mask_source_dir = maskfile[maskfile.keys()[0]]['masks'].attrs['source']
        else:
            mask_source_dir = maskfile[maskfile.keys()[0]].attrs['source']
        if rootdir not in mask_source_dir:
            mask_source_dir = replace_root(mask_source_dir, rootdir, maskfile.attrs['animal'], maskfile.attrs['session'])
        rid_zproj_basedir = os.path.split(mask_source_dir)[0]
        sigchannel_dirname = os.path.split(rid_zproj_basedir)[-1]

        # Get reference file in current trace id set (just use reference from processed dir)
        if roi_tiff_src == trace_tiff_src:
            print "Extracting traces from ROI source"
            matched_sources = True
            if len(maskfiles) == 1:
                ref_file = maskfiles[0]  # REF FILE just is the one used to extract ROIs
            else:
                ref_file = None          # REF FILE doesn't exist, since ROIs extracted from each tif in set
            zproj_source_dir = rid_zproj_basedir
        else:
            print "Extracting traces from ALT run roi src"
            matched_sources = False
            # Identify which file was used as reference, assuming tiffs were preprocessed and motion-corrected:
            if 'mcorrected' in trace_tiff_src:
                # Walk backward from standard motion-correction output-dir formatting
                # to get filepath parts we need:
                processed_dir = os.path.split(trace_tiff_src.split('/mcorrected')[0])[0]
                process_name =  os.path.split(trace_tiff_src.split('/mcorrected')[0])[1]
                run_name = os.path.split(os.path.split(processed_dir)[0])[-1]
                with open(os.path.join(processed_dir, 'pids_%s.json' % run_name), 'r') as f:
                    pdict = json.load(f)
                ref_file = 'File%03d' % int(pdict[process_name.split('_')[0]]['PARAMS']['motion']['ref_file'])
            # Get corresponding zproj source dir:
            zproj_source_dir = '%s_mean_deinterleaved/%s' % (trace_tiff_src, sigchannel_dirname)

        # Get list of files in current trace set:
        filenames = sorted(['File%03d' % int(i+1) for i in range(ntiffs)], key=natural_keys)
        filenames = sorted([ f for f in filenames if f not in excluded_tiffs], key=natural_keys)
        print "Using reference file %s on %i total tiffs." % (ref_file, len(filenames))


        # Check if masks are split up by slices: (Matlab, manual2D methods are diff)
        if type(maskfile[maskfiles[0]]['masks']) == h5py.Dataset:
            slice_masks = False
        else:
            slice_keys = [s for s in maskfile[maskfiles[0]]['masks'].keys() if 'Slice' in s]
            if len(slice_keys) > 0:
                slice_masks = True
            else:
                slice_masks = False

        # Get slices for which there are ROIs in this set:
        if slice_masks:
            roi_slices = sorted([str(s) for s in maskfile[maskfiles[0]]['masks'].keys()], key=natural_keys)
        else:
            roi_slices = sorted(["Slice%02d" % int(s+1) for s in range(nslices)], key=natural_keys)
    except Exception as e:
        traceback.print_exc()
        print "Error loading mask info..."
        print "Mask path was: %s" % mask_path
    #finally:
        #maskfile.close()

    maskinfo['filenames'] = filenames
    maskinfo['ref_file'] = ref_file
    maskinfo['is_single_reference'] = single_reference
    maskinfo['is_3D'] = is_3D
    maskinfo['is_slice_format'] = slice_masks
    maskinfo['roi_slices'] = roi_slices
    maskinfo['filepath'] = mask_path
    maskinfo['matched_sources'] = matched_sources
    maskinfo['zproj_source'] = zproj_source_dir
    maskinfo['roi_source_dir'] = mask_source_dir

    return maskinfo

#def get_mask_info(mask_path, nslices=1, excluded_tiffs=[], rootdir='/n/coxfs01/2p-data'):
#    maskinfo = dict()
#    try:
#        maskfile = h5py.File(mask_path, "r")
#        is_3D = bool(maskfile.attrs['is_3D'])
#
#        # identify tiff source:
#        roidict_path = os.path.join(rootdir, maskfile.attrs['animal'], maskfile.attrs['session'], 'ROIs', 'rids_%s.json' % maskfile.attrs['session'])
#        with open(roidict_path, 'r') as f:
#            roidict = json.load(f)
#        roi_tiff_src = roidict[maskfile.attrs['roi_id']]['SRC']
#
#        if rootdir not in roi_tiff_src:
#            roi_tiff_src = replace_root(roi_tiff_src, rootdir, maskfile.attrs['animal'], maskfile.attrs['session'])
#        ntiffs = len([f for f in os.listdir(roi_tiff_src) if f.endswith('tif')])
#
#
#        # Get files for which there are ROIs in this set:
#        maskfiles = maskfile.keys()
#        print "MASK FILES:", len(maskfiles)
#        if len(maskfiles) == 1:
#            #ntiffs = maskfile.attrs['ntiffs_in_set']
#            filenames = sorted(['File%03d' % int(i+1) for i in range(ntiffs)], key=natural_keys)
#            filenames = sorted([ f for f in filenames if f not in excluded_tiffs], key=natural_keys)
#            ref_file = maskfiles[0]
#            print "Using reference file %s on %i total tiffs." % (ref_file, len(filenames))
#            single_reference = True
#        else:
#            filenames = maskfile.keys()
#            single_reference = False
#            ref_file = None #RID['PARAMS']['options']['ref_file']
#
#        # Check if masks are split up by slices: (Matlab, manual2D methods are diff)
#        if type(maskfile[maskfiles[0]]['masks']) == h5py.Dataset:
#            slice_masks = False
#        else:
#            slice_keys = [s for s in maskfile[maskfiles[0]]['masks'].keys() if 'Slice' in s]
#            if len(slice_keys) > 0:
#                slice_masks = True
#            else:
#                slice_masks = False
#
#        # Get slices for which there are ROIs in this set:
#        if slice_masks:
#            roi_slices = sorted([str(s) for s in maskfile[maskfiles[0]]['masks'].keys()], key=natural_keys)
#        else:
#            roi_slices = sorted(["Slice%02d" % int(s+1) for s in range(nslices)], key=natural_keys)
#    except Exception as e:
#        traceback.print_exc()
#        print "Error loading mask info..."
#        print "Mask path was: %s" % mask_path
#    #finally:
#        #maskfile.close()
#
#    maskinfo['filenames'] = filenames
#    maskinfo['ref_file'] = ref_file
#    maskinfo['is_single_reference'] = single_reference
#    maskinfo['is_3D'] = is_3D
#    maskinfo['is_slice_format'] = slice_masks
#    maskinfo['roi_slices'] = roi_slices
#    maskinfo['filepath'] = mask_path
#
#    return maskinfo

#%%
def masks_to_normed_array(masks):
    '''
    Assumes masks.shape = (d1, d2, nrois)

    Returns:
        maskarray of shape (d, nrois), where d = d1*d2
        values are normalized by size of mask
    '''
    d1, d2 = masks[:,:,0].shape
    d = d1*d2

    nrois = masks.shape[-1]

    masks_arr = np.empty((d, nrois))
    for r in range(nrois):
        masks_arr[:, r] = np.reshape(masks[:,:,r], (d,), order='C') /  len(np.nonzero(masks[:,:,r])[0])

    return masks_arr


#%%
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

def plot_warped_rois(ref, sample, masks, masks_aligned, save_warp_images=True, out_fpath='/tmp/aligned_rois.png'):
    refRGB = uint16_to_RGB(ref)
    imRGB = uint16_to_RGB(sample)
    wimRGB = uint16_to_RGB(sample)
    nrois = masks.shape[-1]

    fig = pl.figure(figsize=(15,5))
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.1, wspace=0.1)
    ax1 = fig.add_subplot(1,3,1); pl.imshow(refRGB, cmap='gray'); pl.title('ref rois'); pl.axis('off')
    ax2 = fig.add_subplot(1,3,2); pl.imshow(imRGB, cmap='gray'); pl.title('sample, orig rois'); pl.axis('off')
    ax3 = fig.add_subplot(1,3,3); pl.imshow(imRGB, cmap='gray'); pl.title('sample, warped rois'); pl.axis('off')
    for ridx in range(nrois):
        #roinum = ridx + 1
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

    #figname = 'aligned_rois.png'
    pl.savefig(out_fpath)
    pl.close()

#%%
def warp_masks(masks, ref, img, warp_mode=cv2.MOTION_HOMOGRAPHY, save_warp_images=False, out_fpath='/tmp/warped.png'):

    height, width = ref.shape

    # Allocate space for aligned image
    ref_aligned = np.zeros((height,width), dtype=ref.dtype) #dtype=np.uint8 )

    # Define motion model
    #warp_mode = cv2.MOTION_HOMOGRAPHY

    # Set the warp matrix to identity.
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
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
    else:
        # Use Affine warp when the transformation is not a Homography
        ref_aligned = cv2.warpAffine(ref, warp_matrix, (width, height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
        mode_str = 'WARP_AFFINE'


    #% Warp masks with same transform:
    masks_aligned = np.zeros(masks.shape, dtype=masks.dtype)
    nrois = masks.shape[-1]
    for r in xrange(0, nrois):
        masks_aligned[:,:,r] = cv2.warpPerspective (masks[:,:,r], warp_matrix, (width,height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    # Save warp alignment, if requested:
    if save_warp_images:
        plot_warped_rois(ref, sample, masks, masks_aligned, save_warp_images=save_warp_images, out_fpath=out_fpath)

    return masks_aligned

#%%

def create_neuropil_masks(masks, niterations=3):
    # Create kernel for dilating ROIs:
    kernel = np.ones((3,3),masks.dtype)

    nrois = masks.shape[-1]
    np_masks = np.empty(masks.shape, dtype=masks.dtype)

    for ridx in range(nrois):
        rmask = masks[:,:,ridx]
        dilated = cv2.dilate(rmask, kernel, iterations=niterations)

        # Subtract to get annulus region:
        annulus = (dilated - rmask)

        # Get full mask image to subtract overlaps:
        allmasks = np.sum(masks, axis=-1)
        summed = annulus + allmasks
        summed[summed>1] = 0

        # Add annulus back in to make neuropil area = 2, everythign else = 1:
        summed += annulus
        neuropil = summed - allmasks
        np_masks[:,:,ridx] = neuropil.astype('bool')

    return np_masks


#%%
def get_masks(mask_write_path, maskinfo, RID, save_warp_images=False, do_neuropil_correction=True, niter=3, rootdir=''):
    '''
    This function takes a masks.hdf5 file and formats it into a dict that standardizes how
    mask arrays are called. This was a tmp fix to deal with the fact that
        - ROI types differ in saving/indexing ROIs by slice or across the whole volume (only relevant for 3D)
        - if ROI type only uses a single reference (manual, for ex.), we want to apply those ROIs to all other files
        - cNMF trace extraction offers non-raw traces, so just bring that along at this step

    Save MASKS.pkl at end.
    This only needs to be done once, unless create_new = True (i.e., if re-doing masks prior to extracting traces).
    '''

    curr_file=None; curr_slice=None

    MASKS = h5py.File(mask_write_path, 'w')
    MASKS.attrs['source_file'] = maskinfo['filepath']
    MASKS.attrs['roi_id'] = str(RID['roi_id'])
    MASKS.attrs['rid_hash'] = str(RID['rid_hash'])
    MASKS.attrs['roi_type'] = str(RID['roi_type'])

    try:
        maskfile = h5py.File(str(maskinfo['filepath']), 'r')
        for fidx, curr_file in enumerate(maskinfo['filenames']):

            if curr_file not in MASKS.keys():
                filegrp = MASKS.create_group(curr_file)
            else:
                filegrp = MASKS[curr_file]

            if maskinfo['is_single_reference'] is True:
                if maskinfo['matched_sources'] is True:
                    maskfile_key = maskinfo['ref_file']
                else:
                    # ROI_SOURCE_DIR is the deinterleaved dir containing the
                    # source img(s) from which ROIs were created -- this is the image
                    # set we want to warp to the CURRENT set of tiffs' reference img...
                    maskfile_key = os.path.split(maskinfo['roi_source_dir'])[-1]
                ref_zproj_dir = maskinfo['roi_source_dir']
            else:
                maskfile_key = curr_file

            # Get base dir of zprojected images -- this is the DETINERLEAVED
            # zproj-dir in 'processed' folder, specifying Channel and File.
            # This format is assumed since we save the REFERENCE info as source.
            zproj_source_dir = maskinfo['zproj_source']

            # Tmp fix:  renamed zproj-dirs with "deinterleaved" (old way: "slices")
            if '_slices' in zproj_source_dir:
                if '_mean_' in zproj_source_dir:
                    zproj_source_dir = zproj_source_dir.replace('_mean_slices', '_mean_deinterleaved')
                elif '_std_' in zproj_source_dir:
                    zproj_source_dir = zproj_source_dir.replace('_std_slices', '_std_deinterleaved')

            curr_zproj_dir = os.path.join(zproj_source_dir, curr_file)
            # Check root:
            if rootdir not in curr_zproj_dir:
                session_dir = RID['DST'].split('/ROIs')[0]; session = os.path.split(session_dir)[-1]; animalid = os.path.split(os.path.split(session_dir)[0])[-1]
                curr_zproj_dir = replace_root(curr_zproj_dir, rootdir, animalid, session)

            # Make sure all masks are saved by file, slice:
            for sidx, curr_slice in enumerate(maskinfo['roi_slices']):

                # Get average image of CURRENT TIFF:
                zproj_img_fn = [m for m in os.listdir(curr_zproj_dir) if curr_slice in m][0]
                zproj_img_path = os.path.join(curr_zproj_dir, zproj_img_fn)
                img = tf.imread(zproj_img_path)
                print "... loaded TIFF %i of %i to warp ROIs onto..." % (fidx, len(maskinfo['filenames']))
                zproj = filegrp.create_dataset('/'.join([curr_slice, 'zproj']), img.shape, img.dtype)
                zproj[...] = img
                zproj.attrs['source'] = zproj_img_path

                # Get masks from ROI SOURCE:
                if maskinfo['is_slice_format']: #slice_masks:
                    src_roi_idxs = maskfile[maskfile_key]['masks'][curr_slice].attrs['src_roi_idxs']
                    masks = maskfile[maskfile_key]['masks'][curr_slice][:].T.copy()
                else:
                    src_roi_idxs = maskfile[maskfile_key]['masks'].attrs['src_roi_idxs']
                    masks = maskfile[maskfile_key]['masks'][:].T.copy()


                # Warp reference tiff to current tiff so that ROIs line up:
                if maskinfo['is_single_reference']:
                    # If ROI source == tiff source, just need to make sure REF FILE is normed:
                    if curr_file == maskinfo['ref_file'] and maskinfo['matched_sources'] is True:
                        mask_arr = masks_to_normed_array(masks)
                    else:
                        # If ROI source and TIFF source differ, reference image
                        # of ROI source should be loaded, to then be aligned to
                        # all current tiff files:
                        if maskinfo['matched_sources'] is False:
                            print "... loading ROI src reference img to warp."
                            ref_img_fpath = [os.path.join(maskinfo['roi_source_dir'], f)
                                                for f in os.listdir(maskinfo['roi_source_dir']) if f.endswith('tif')][0]
                            print "... %s" % ref_img_fpath
                        else:
                            zproj_base = os.path.split(curr_zproj_dir)[0]; ref = maskinfo['ref_file']
                            ref_img_fpath = [os.path.join(zproj_base, ref, m)
                                                for m in os.listdir(os.path.join(zproj_base, ref)) if curr_slice in m][0]

                        ref_img = tf.imread(ref_img_fpath)

                        # Get warp matrix to transform reference to current file:
                        if save_warp_images:
                            traceid_dir = os.path.split(mask_write_path)[0]
                            warp_img_dir = os.path.join(traceid_dir, 'figures', 'masks', 'warps')
                            if not os.path.exists(warp_img_dir):
                                os.makedirs(warp_img_dir)
                            print "%s - %s: Saving warp imgs to %s" % (curr_slice, curr_file, warp_img_dir)
                            warp_img_path = os.path.join(warp_img_dir, 'warped_rois_r%s_to_%s_%s.png' % (maskinfo['ref_file'], curr_file, curr_slice))
                        else:
                            warp_img_path = ''
                        print "... warping original REF to current tif image for aligned ROIs."
                        masks = warp_masks(masks, ref_img, img, save_warp_images=save_warp_images, out_fpath=warp_img_path)
                        mask_arr = masks_to_normed_array(masks)
                    mref_name = maskinfo['ref_file']
                else:
                    if len(masks.shape) > 2:
                        mask_arr = masks_to_normed_array(masks)
                    else:
                        mask_arr = masks.copy()
                    mref_name = curr_file

                # Check if extra backround:
                if 'background' in maskfile[maskfile_key]['masks'].attrs.keys():
                    nb = maskfile[maskfile_key]['masks'].attrs['background']
                else:
                    nb = 0

                # Save mask info:
                nrois = mask_arr.shape[-1]
                m = filegrp.create_dataset('/'.join([curr_slice, 'maskarray']), mask_arr.shape, mask_arr.dtype)
                m[...] = mask_arr
                m.attrs['nb'] = nb
                m.attrs['nr'] = nrois - nb
                m.attrs['src_roi_idxs'] = src_roi_idxs
                m.attrs['rois'] = sorted(["roi%05d" % int(ridx+1) for ridx in range(len(src_roi_idxs))], key=natural_keys)


                # Check if should create neuropil masks:
                if do_neuropil_correction:
                    np_masks = create_neuropil_masks(masks, niterations=niter)
                    npil_arr = masks_to_normed_array(np_masks)
                    npil = filegrp.create_dataset('/'.join([curr_slice, 'np_maskarray']), npil_arr.shape, npil_arr.dtype)
                    npil[...] = npil_arr
                    npil.attrs['niterations'] = niter


                # Check if have nmf traces:
                if 'Ab_data' in maskfile[maskfile.keys()[0]].keys():
                    Ab = load_sparse_mat('%s/Ab' % curr_file, maskinfo['filepath']).todense()
                    Cf = load_sparse_mat('%s/Cf' % curr_file, maskinfo['filepath']).todense()
                    mab = filegrp.create_dataset('/'.join([curr_slice, 'Ab']), Ab.shape, Ab.dtype)
                    mab[...] = Ab
                    mac = filegrp.create_dataset('/'.join([curr_slice, 'Cf']), Cf.shape, Cf.dtype)
                    mac[...] = Cf

    except Exception as e:
        print "------------------------------------------"
        print "*** ERROR creating masks: %s, %s ***" % (curr_file, curr_slice)
        traceback.print_exc()
        print "------------------------------------------"

    finally:
        MASKS.close()
        maskfile.close()

    return mask_write_path #, curr_rois



#%%
def plot_roi_masks(TID, RID, plot_neuropil=True, mask_figdir='/tmp', rootdir=''):
    '''
    This also only needs to be done once, unless create_new = True or incorrect
    number of mask images found in <TRACEID_DIR>/figures/masks/.

    It's a little slow..

    '''
    maskdict_path = os.path.join(TID['DST'], 'MASKS.hdf5')
    if rootdir not in maskdict_path:
        session_dir = RID['DST'].split('/ROIs')[0]
        info = get_info_from_tiff_dir(TID['SRC'], session_dir)
        maskdict_path = replace_root(maskdict_path, rootdir, info['animalid'], info['session'])
    MASKS = h5py.File(maskdict_path, 'r')

    print "Plotting masks for %i files." % len(MASKS.keys())
    filenames = [k for k in MASKS.keys() if 'File' in k]

    #Save slices in separate dirs:
    multislice = len(MASKS[filenames[0]].keys()) > 1

    for curr_file in sorted(filenames, key=natural_keys): #sorted(MASKS.keys(), key=natural_keys):
        for curr_slice in sorted(MASKS[curr_file].keys(), key=natural_keys):
            if multislice is True:
                save_dir = os.path.join(mask_figdir, curr_slice)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
            else:
                save_dir = mask_figdir

            curr_rois = MASKS[curr_file][curr_slice]['maskarray'].attrs['rois']
            nrois = len(curr_rois)

            avg = MASKS[curr_file][curr_slice]['zproj'][:]
            dims = avg.shape

            nb = MASKS[curr_file][curr_slice]['maskarray'].attrs['nb']
            maskarray= MASKS[curr_file][curr_slice]['maskarray'][:]

            print "--- Mask array: %i ROIs on %s, %s" % (len(curr_rois), curr_file, curr_slice)
            fig = pl.figure()
            ax = fig.add_subplot(1,1,1)
            p2, p98 = np.percentile(avg, (2, 99.98))
            avgimg = skimage.exposure.rescale_intensity(avg, in_range=(p2, p98)) #avg *= (1.0/avg.max())
            ax.imshow(avgimg, cmap='gray')

            if plot_neuropil is True:
                if 'np_maskarray' not in MASKS[curr_file][curr_slice].keys():
                    print "--- No neuropil mask array found!"
                    plot_neuropil = False
                else:
                    np_maskarray = MASKS[curr_file][curr_slice]['np_maskarray'][:]
                    np_niterations = MASKS[curr_file][curr_slice]['np_maskarray'].attrs['niterations']
                    print "--- ...plus neuropil."


            bgidx = 0
            for ridx in range(nrois):
                masktmp = np.reshape(maskarray[:, ridx], dims, order='C')
                msk = masktmp.copy()
                msk[msk==0] = np.nan

                if (nb > 0) and (ridx >= (nrois-nb)):
                    bgidx += 1
                    bgfig = pl.figure()
                    bgax = bgfig.add_subplot(1,1,1)
                    bgax.imshow(avg, cmap='gray')
                    bgax.imshow(msk, interpolation='None', alpha=0.3, cmap=pl.cm.hot)
                    pl.title('background %i' % bgidx)
                    pl.savefig(os.path.join(mask_figdir, 'bg%i_%s_%s_%s_%s.png' % (bgidx, curr_file, curr_slice, RID['roi_id'], RID['rid_hash'])))
                    pl.close()
                    is_bg = True
                else:
                    if 'caiman' in RID['roi_type'] or (RID['roi_type']=='coregister' and 'caiman' in RID['PARAMS']['options']['source']['roi_type']):
                        ax.imshow(msk, interpolation='None', alpha=0.2, cmap=pl.cm.hot)
                        [ys, xs] = np.where(masktmp>0)
                        ax.text(xs[int(round(len(xs)/4))], ys[int(round(len(ys)/4))], str(ridx+1), fontsize=8, weight='light', color='w')
                    else:
                        if np.isnan(masktmp).all():
                            ax.text(1, 1, '%i - no mask' % int(ridx+1), fontsize=8, weight='light', color='r')
                        else:
                            ax.imshow(msk, interpolation='None', alpha=0.5, cmap=pl.cm.Greens_r, vmin=0, vmax=1.0)
                            [ys, xs] = np.where(masktmp>0)
                            ax.text(xs[int(round(len(xs)/4))], ys[int(round(len(ys)/4))], str(ridx+1), fontsize=8, weight='light', color='w')
                    is_bg = False

                if plot_neuropil is True and not is_bg:
                    masktmp = np.reshape(np_maskarray[:, ridx], dims, order='C')
                    msk = masktmp.copy()
                    msk[msk==0] = np.nan
                    ax.imshow(msk, interpolation='None', alpha=0.2, cmap=pl.cm.Blues_r)
                    #[ys, xs] = np.where(masktmp>0)
                    #ax2.text(xs[int(round(len(xs)/4))], ys[int(round(len(ys)/4))], str(ridx+1), fontsize=8, weight='light', color='w')

            ax.axis('off')

            if plot_neuropil is True:
                figname = 'rois_%s_%s_%s_%s_np_iter%i.png' % (curr_file, curr_slice, RID['roi_id'], RID['rid_hash'], np_niterations)
            else:
                figname = 'rois_%s_%s_%s_%s.png' % (curr_file, curr_slice, RID['roi_id'], RID['rid_hash'])

            pl.savefig(os.path.join(save_dir, figname))
            pl.close()


    MASKS.close()

#%%
def hash_filetraces(filetraces_dir, traceid_hash):
    file_hashdict = {}

    filetraces_fns = sorted([f for f in os.listdir(filetraces_dir) if f.endswith('hdf5')], key=natural_keys)

    for filetraces_fname in filetraces_fns:
        filetraces_fpath = os.path.join(filetraces_dir, filetraces_fname)
        # Create hash of current raw tracemat:
        rawfile_hash = hash_file(filetraces_fpath)

        file_hashdict[os.path.splitext(filetraces_fname)[0]] = rawfile_hash

    with open(os.path.join(filetraces_dir, 'filetraces_info_%s.json' % traceid_hash), 'w') as f:
        json.dump(file_hashdict, f, indent=4, sort_keys=True)

    print "Saved hash info for file-traces files."

#%%

def apply_masks_to_tiff(currtiff_path, TID, si_info, do_neuropil_correction=True, cfactor=0.6, output_filedir='/tmp', rootdir=''):
    nchannels = si_info['nchannels']
    nslices = si_info['nslices']
    nvolumes = si_info['nvolumes']
    frames_tsec = si_info['frames_tsec']
    signal_channel_idx = int(TID['PARAMS']['signal_channel']) - 1 # 0-indexing into tiffs

    traceid_dir = TID['DST']

    curr_file = str(re.search(r"File\d{3}", currtiff_path).group())
    if curr_file in TID['PARAMS']['excluded_tiffs']:
        print "***Skipping %s -- excluded from ROI set" % (curr_file)
        return
    print "-- Extracting traces: %s" % curr_file

    # Load MASKS info:
    maskdict_path = os.path.join(traceid_dir, 'MASKS.hdf5')
    MASKS = h5py.File(maskdict_path, 'r')

    roi_slices = [k for k in MASKS[curr_file].keys() if 'Slice' in k] #maskinfo['roi_slices']

    # Create outfile:
    filetraces_fn = '%s_rawtraces_%s.hdf5' % (curr_file, TID['trace_hash'])
    filetraces_filepath = os.path.join(traceid_dir, 'files', filetraces_fn)

    try:
        # Load input tiff file:
        print "-- -- Reading tiff..."
        tiff = tf.imread(currtiff_path)
        T, d1, d2 = tiff.shape
        d = d1*d2
        tiffR = np.reshape(tiff, (T, d), order='C'); del tiff

        # First get signal channel only:
        tiffR = tiffR[signal_channel_idx::nchannels,:]

        # Apply masks to each slice:
        file_grp = h5py.File(filetraces_filepath, 'w')
        file_grp.attrs['source_file'] = currtiff_path
        file_grp.attrs['file_id'] = curr_file
        file_grp.attrs['signal_channel'] = TID['PARAMS']['signal_channel']
        file_grp.attrs['dims'] = (d1, d2, nslices, T/nslices)
        file_grp.attrs['mask_sourcefile'] = MASKS.attrs['source_file']  #MASKS['original_source'] #mask_path

        for sl, curr_slice in enumerate(sorted(roi_slices, key=natural_keys)):

            print "-- -- -- Extracting ROI time course from %s" % curr_slice
            maskarray = MASKS[curr_file][curr_slice]['maskarray'][:]

            # Get frame tstamps:
            curr_tstamps = np.array(frames_tsec[sl::nslices])

            # Save tstamps:
            fset = file_grp.create_dataset('/'.join([curr_slice, 'frames_tsec']), curr_tstamps.shape, curr_tstamps.dtype)
            fset[...] = curr_tstamps

            # Get current frames:
            tiffslice = tiffR[sl::nslices, :]
            tracemat = tiffslice.dot(maskarray)
            dims = (d1, d2, T/nvolumes)

            # Save RAW trace:
            tset = file_grp.create_dataset('/'.join([curr_slice, 'traces', 'raw']), tracemat.shape, tracemat.dtype)
            tset[...] = tracemat
            tset.attrs['nframes'] = tracemat.shape[0]
            tset.attrs['dims'] = dims

            # Save NEUROPIL traces and neurpil-CORRECTED traces, if relevant:
            if do_neuropil_correction is True and 'np_maskarray' in MASKS[curr_file][curr_slice].keys():
                print "-- -- -- + neuropil subtraction"
                np_maskarray = np.array(MASKS[curr_file][curr_slice]['np_maskarray']) #[:]
                np_tracemat = tiffslice.dot(np_maskarray)
                np_corrected = tracemat - (cfactor * np_tracemat)
                np_traces = file_grp.create_dataset('/'.join([curr_slice, 'traces', 'neuropil']), np_tracemat.shape, np_tracemat.dtype)
                np_traces[...] = np_tracemat
                np_corrected = file_grp.create_dataset('/'.join([curr_slice, 'traces', 'np_subtracted']), np_corrected.shape, np_corrected.dtype)
                np_corrected.attrs['correction_factor'] = cfactor


            # Extract NMF-denoised traces, if relevant:
            if 'Ab' in MASKS[curr_file][curr_slice].keys():
                Ab = MASKS[curr_file][curr_slice]['Ab'][:]
                Cf = MASKS[curr_file][curr_slice]['Cf'][:]
                extracted_traces = Ab.T.dot(Ab.dot(Cf))
                extracted_traces = np.array(extracted_traces.T) # trans to get same format as other traces (NR x Tpoints)
                ext = file_grp.create_dataset('/'.join([curr_slice, 'traces', 'denoised_nmf']), extracted_traces.shape, extracted_traces.dtype)
                ext[...] = extracted_traces
                ext.attrs['nb'] = MASKS[curr_file][curr_slice].attrs['nb']
                ext.attrs['nr'] = MASKS[curr_file][curr_slice].attrs['nr']

        print "-- Done extracting: %s" % curr_file

    except Exception as e:
        print "--- TID %s: Error extracting traces from file %s ---" % (TID['trace_hash'], curr_file)
        traceback.print_exc()
        print "---------------------------------------------------------------"
    finally:
        if file_grp is not None:
            file_grp.close()
        #maskfile.close()

    return filetraces_filepath


#%%

def apply_masks_by_tid(tmp_tid_path, filenum=1, rootdir='', do_neuropil_correction=True, cfactor=0.5):
    filetraces_filepath = None

    # Load tmp rid file for coreg:
    with open(tmp_tid_path, 'r') as f:
        TID = json.load(f)

    rundir = tmp_tid_path.split('/traces')[0]
    session_dir = os.path.split(os.path.split(rundir)[0])[0]
    info = get_info_from_tiff_dir(TID['SRC'], session_dir)

    if rootdir not in rundir:
        rundir = replace_root(rundir, rootdir, info['animalid'], info['session'])
    print "RUN:", rundir
    si_info = get_frame_info(rundir)

    if rootdir not in TID['DST']:
        TID['DST'] = replace_root(TID['DST'], rootdir, info['animalid'], info['session'])

    filetraces_dir = os.path.join(TID['DST'], 'files')
    print "Saving file traces to:", filetraces_dir
    if not os.path.exists(filetraces_dir):
        os.makedirs(filetraces_dir)

    if rootdir not in TID['SRC']:
        TID['SRC'] = replace_root(TID['SRC'], rootdir, info['animalid'], info['session'])
    tiff_files = sorted([t for t in os.listdir(TID['SRC']) if t.endswith('tif')], key=natural_keys)

    # Then, get matches to sample:
    curr_file = 'File%03d' % filenum
    if curr_file in TID['PARAMS']['excluded_tiffs']:
        return None

    try:
        tfn = [p for p in tiff_files if str(re.search('File(\d{3})', p).group(0)) == curr_file][0]
        currtiff_path = os.path.join(TID['SRC'], tfn)
        filetraces_filepath = apply_masks_to_tiff(currtiff_path, TID, si_info,
                                                  do_neuropil_correction=do_neuropil_correction,
                                                  cfactor=cfactor,
                                                  output_filedir=filetraces_dir,
                                                  rootdir=rootdir)
    except Exception as e:
        if filetraces_filepath is None:
            print 'Unable to find tiff src path for: %s' % curr_file
            for r in tiff_files:
                print r
        traceback.print_exc()

    return filetraces_filepath


#%%
# =============================================================================
# Extract ROIs for each specified slice for each file:
# =============================================================================
def apply_masks_to_movies(TID, RID, si_info, do_neuropil_correction=True, cfactor=0.6, output_filedir='/tmp', rootdir=''):
    '''
    For each .tif in this trace id set, load .tif movie and apply masks.
    Save traces as .hdf5 for each .tif file in <TRACEID_DIR>/files/.
    '''
    session_dir = RID['DST'].split('/ROIs/')[0]
    info = get_info_from_tiff_dir(TID['SRC'], session_dir)

    if rootdir not in TID['SRC']:
        TID['SRC'] = replace_root(TID['SRC'], rootdir, info['animalid'], info['session'])
    tiff_files = sorted([t for t in os.listdir(TID['SRC']) if t.endswith('tif')], key=natural_keys)

    print "TID %s -- Applying masks to traces..." % TID['trace_hash']
    t_extract = time.time()

    # Load MASKDICT:
    if rootdir not in TID['DST']:
        TID['DST'] = replace_root(TID['DST'], rootdir, info['animalid'], info['session'])

    for tfn in tiff_files:
        curr_file = str(re.search(r"File\d{3}", tfn).group())
        if curr_file in TID['PARAMS']['excluded_tiffs']:
            print "***Skipping %s -- excluded from ROI set %s" % (curr_file, RID['roi_id'])
            continue
        print "Extracting traces: %s" % curr_file
        currtiff_path = os.path.join(TID['SRC'], tfn)
        filetraces_filepath = apply_masks_to_tiff(currtiff_path, TID, si_info,
                                                  do_neuropil_correction=do_neuropil_correction,
                                                  cfactor=cfactor,
                                                  output_filedir=output_filedir, rootdir=rootdir)
        print "Saved %s traces: %s" % (curr_file, filetraces_filepath)

    # Hash filetraces files:
    filetraces_dir = os.path.join(TID['DST'], 'files')
    hash_filetraces(filetraces_dir, TID['trace_hash'])

    print "TID %s -- Finished compiling trace arrays across files" % TID['trace_hash']
    print_elapsed_time(t_extract)
    print "-----------------------------------------------------------------------"

    return os.path.join(TID['DST'], 'files')

#%%
# =============================================================================
# Create time courses for ROIs:
# =============================================================================

def get_roi_timecourses(TID, RID, si_info, input_filedir='/tmp', rootdir='', create_new=False):
    '''
    Concatenate extracted traces from each file into one set of traces. At this level,
    organization is by ROI, not by .tif file.
    '''

    session_dir = RID['DST'].split('/ROIs')[0]
    info = get_info_from_tiff_dir(TID['SRC'], session_dir)
    print "Got trace info:"
    pp.pprint(info)

    traceid_dir = TID['DST']
    if rootdir not in traceid_dir:
        traceid_dir = replace_root(traceid_dir, rootdir, info['animalid'], info['session'])
    traceid_dir = TID['DST']
    if rootdir not in traceid_dir:
        traceid_dir = replace_root(traceid_dir, rootdir, info['animalid'], info['session'])

    print "-----------------------------------------------------------------------"
    print "TID %s -- sorting traces by ROI..." % TID['trace_hash']
    t_roi = time.time()

    #% Load raw traces:
    try:
        filetraces_fns = [f for f in os.listdir(os.path.join(traceid_dir, 'files')) if f.endswith('hdf5') and 'rawtraces' in f]
        print "Found traces by file for %i tifs in dir: %s" % (len(filetraces_fns), os.path.join(traceid_dir, 'files'))
    except Exception as e:
        print "Unable to find extracted tracestructs from trace set: %s" % TID['trace_id']
        print "Aborting with error:"
        traceback.print_exc()

    # Create output file for parsed traces:
    # -----------------------------------------------------------------------------
    # Check if time courses file already exists:
    existing_tcourse_fns = sorted([t for t in os.listdir(traceid_dir) if 'roi_timecourses' in t and t.endswith('hdf5')], key=natural_keys)
    if len(existing_tcourse_fns) > 0 and create_new is False:
        roi_tcourse_filepath = os.path.join(traceid_dir, existing_tcourse_fns[-1])
        print "Loaded existing ROI time courses file: %s" % roi_tcourse_filepath
    else:
        tstamp = datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")
        roi_tcourse_fn = 'roi_timecourses_%s.hdf5' % tstamp
        roi_tcourse_filepath = os.path.join(traceid_dir, roi_tcourse_fn)

        roi_outfile = h5py.File(roi_tcourse_filepath, 'w')
        roi_outfile.attrs['tiff_source'] = TID['SRC']
        roi_outfile.attrs['trace_id'] = TID['trace_id']
        roi_outfile.attrs['trace_hash'] = TID['trace_hash']
        roi_outfile.attrs['trace_source'] = traceid_dir
        roi_outfile.attrs['roiset_id'] = RID['roi_id']
        roi_outfile.attrs['roiset_hash'] = RID['rid_hash']
        roi_outfile.attrs['run'] = info['run']
        roi_outfile.attrs['session'] = info['session']
        roi_outfile.attrs['acquisition'] = info['acquisition']
        roi_outfile.attrs['animalid'] = info['animalid']
        roi_outfile.attrs['creation_time'] = tstamp #datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Concatenate time courses across all files to create single time-source for RUN:
        # -----------------------------------------------------------------------------
        tmp_tracestruct = h5py.File(os.path.join(traceid_dir, 'files', filetraces_fns[0]), 'r')
        ntiffs = si_info['ntiffs']
        nslices = si_info['nslices']
        nslices_full = si_info['nslices_full']
        dims = tmp_tracestruct.attrs['dims']
        all_frames_tsec = np.array(si_info['frames_tsec'])
        #total_nframes_in_run = tmp_tracestruct.attrs['dims'][-1] * ntiffs * nslices_full
        tmp_tracestruct.close(); del tmp_tracestruct

        #nframes_in_file = si_info['nframes_per_file'] #uninfo['nvolumes']
        nframes_in_file_raw = si_info['nframes_per_file']
        nframes_per_chunk = int(round(nframes_in_file_raw / nslices_full))
        total_nframes_in_run = nframes_per_chunk * ntiffs

        full_file_dur = all_frames_tsec[-1] + (1/si_info['framerate'])

        curr_frame_idx = 0
        tiff_start_fridxs = []
        for fi in range(ntiffs):
            tiff_start_fridxs.append(fi * nframes_per_chunk)

        try:
            for fi, filetraces_fn in enumerate(sorted(filetraces_fns, key=natural_keys)):
                fidx_in_run = int(filetraces_fn[4:7]) - 1 # Get IDX of tiff file in run (doesn't assume all tiffs continuous)
                curr_frame_idx = tiff_start_fridxs[fidx_in_run]
                #print "Loading file:", filetraces_fn
                filetraces = h5py.File(os.path.join(traceid_dir, 'files', filetraces_fn), 'r')         # keys are SLICES (Slice01, Slice02, ...)

                roi_counter = 0
                for sidx, currslice in enumerate(sorted(filetraces.keys(), key=natural_keys)):
                    #print "Loading slice:", currslice
#                    maskarray = filetraces[currslice]['maskarray'][:]
#                    d1, d2 = dims[0:2] #tracefile[currslice]['traces']'rawtraces'].attrs['dims'][0:-1]
#                    T = dims[-1] #tracefile[currslice]['rawtraces'].attrs['nframes']
#                    src_roi_idxs = filetraces[currslice]['maskarray'].attrs['src_roi_idxs']
#                    nr = filetraces[currslice]['maskarray'].attrs['nr'] #maskarray.shape[1]
#                    nb = filetraces[currslice]['maskarray'].attrs['nb']
#                    ncomps = nr + nb
#                    masks = np.reshape(maskarray, (d1, d2, nr+nb), order='C')
#
                    ncomps = filetraces[currslice]['traces']['raw'].shape[-1]
                    bgidx = 0
                    for ridx in range(ncomps): #, roi in enumerate(src_roi_idxs):
#                        if (nb > 0) and (ridx >= nr):
#                            bgidx += 1
#                            is_background = True
#                            roiname = 'bg%02d' % bgidx
#                        else:
                        is_background = False
                        roi_counter += 1
                        roiname = 'roi%05d' % int(roi_counter)

                        # Create unique ROI group:
                        if roiname not in roi_outfile.keys():
                            roi_grp = roi_outfile.create_group(roiname)
                            #roi_grp.attrs['slice'] = currslice
                            #roi_grp.attrs['roi_img_path'] = filetraces[currslice]['zproj'].attrs['img_source']
                        else:
                            roi_grp = roi_outfile[roiname]

#                        if 'mask' not in roi_grp.keys():
#                            roi_mask = roi_grp.create_dataset('mask', masks[:,:,ridx].shape, masks[:,:,ridx].dtype)
#                            roi_mask[...] = masks[:,:,ridx]
                        roi_grp.attrs['id_in_set'] = roi_counter #roi
#                            roi_grp.attrs['id_in_src'] = src_roi_idxs[ridx] #ridx
                        roi_grp.attrs['idx_in_slice'] = ridx
                        roi_grp.attrs['slice'] = currslice
                        roi_grp.attrs['is_background'] = is_background
#                            roi_mask.attrs['slice'] = currslice
#                            roi_mask.attrs['is_background'] = is_background
#
                        # Add time courses:
                        trace_types = filetraces[currslice]['traces'].keys()
                        if 'timecourse' not in roi_grp.keys():
                            tcourse_grp = roi_grp.create_group('timecourse')
                        else:
                            tcourse_grp = roi_grp['timecourse']

                        for trace_type in trace_types:
                            #print "---> Sorting %s" % trace_type
                            curr_tcourse = filetraces[currslice]['traces'][trace_type][:, ridx]
                            if trace_type not in tcourse_grp.keys():
                                roi_tcourse = tcourse_grp.create_dataset(trace_type, (total_nframes_in_run,), curr_tcourse.dtype)
                            else:
                                roi_tcourse = tcourse_grp[trace_type]
                            roi_tcourse[curr_frame_idx:curr_frame_idx+nframes_per_chunk] = curr_tcourse
                            roi_tcourse.attrs['source_file'] = os.path.join(traceid_dir, 'files', filetraces_fn)

                        # Add raw time stamps:
                        if 'frames_tsec' not in roi_grp.keys():
                            tsec = roi_grp.create_dataset('frames_tsec', (total_nframes_in_run,), all_frames_tsec.dtype)
                        else:
                            tsec = roi_grp['frames_tsec']
                        tsec[curr_frame_idx:curr_frame_idx+nframes_per_chunk] = all_frames_tsec[sidx::nslices_full] + (full_file_dur * fi)

                        #print "%s: added frames %i:%i, from %s." % (roiname, curr_frame_idx, curr_frame_idx+nframes_in_file, filetraces_fn)
        except Exception as e:
            print "--- TID %s: ERROR extracting traces from file %s..." % (TID['trace_hash'], filetraces_fn)
            traceback.print_exc()
        finally:
            roi_outfile.close()

        ## Rename FRAME file with hash:
        roi_tcourse_filehash = hash_file(roi_tcourse_filepath)

        # Check if existing files:
        outdir = os.path.split(roi_tcourse_filepath)[0]
        existing_files = [f for f in os.listdir(outdir) if 'roi_timecourses_' in f and f.endswith('hdf5') and roi_tcourse_filehash not in f and tstamp not in f]
        if len(existing_files) > 0:
            old = os.path.join(outdir, 'old')
            if not os.path.exists(old):
                os.makedirs(old)
            for f in existing_files:
                shutil.move(os.path.join(outdir, f), os.path.join(old, f))

        roi_tcourse_filepath = hash_file_read_only(roi_tcourse_filepath)

        print "TID %s -- Finished extracting time course for run %s by roi." % (TID['trace_hash'], info['run'])
        print_elapsed_time(t_roi)
        print "Saved ROI TIME COURSE file to:", roi_tcourse_filepath

    return roi_tcourse_filepath



#%%
def format_masks_fissa(rois_fpath):
    '''
    Currently, only tested with manually extract ROIs.
    ONE set of ROI contours is applied to all files in set (which FISSA calls "trials").
    Create a boolean array for each ROI and append as list for the single file source.
    '''
    maskfile = h5py.File(rois_fpath, 'r')
    ref_filename = maskfile.keys()[0]
    masks = maskfile[ref_filename]['masks']
    if type(masks) == h5py._hl.group.Group:
        if len(masks.keys()) == 1:  # SINGLE SLICE
            masks = np.array(masks['Slice01']).T

    # Format masks into list of arrays:
    masks = masks.astype(bool)
    roi_masks = []
    for roi in range(masks.shape[-1]):
        roi_masks.append(masks[:,:,roi])
    roi_list = [roi_masks]

    return roi_list

#%%
def collate_fissa_traces(exp, tiff=0, region='cell', trace_type='corrected'):
    nrois = exp.nCell
    nframes = exp.result[0][tiff].shape[1]
    nregions = exp.result[0][tiff].shape[0]
    nregions_np = nregions - 1

    tracemat = np.empty((nframes, nrois))
    for ridx in range(len(exp.result)):
        meandiffs = [abs(np.mean(exp.raw[ridx][tiff][0,:]) - np.mean(exp.result[ridx][tiff][i,:]))
                                for i in range(len(exp.result[ridx][tiff]))]
        if region == 'cell':
            region_index = meandiffs.index(np.min(meandiffs))
            if trace_type == 'corrected':
                tracemat[:, ridx] = exp.result[ridx][tiff][region_index,:]
            elif trace_type == 'raw':
                tracemat[:, ridx] = exp.raw[ridx][tiff][region_index,:]
        else:
            region_index = [i for i in range(len(meandiffs)) if not i==meandiffs.index(np.min(meandiffs))]
            if trace_type == 'corrected':
                tracemat[:, ridx] = np.mean(exp.result[ridx][tiff][region_index,:], axis=0)
            elif trace_type == 'raw':
                tracemat[:, ridx] = np.mean(exp.raw[ridx][tiff][region_index,:], axis=0)

    return tracemat

#%%
def make_nonnegative(images_dir):

    tiff_list = sorted([t for t in os.listdir(images_dir) if t.endswith('tif')], key=natural_keys)

    images_dir_nonneg = '%s_nonnegative' % images_dir
    if not os.path.exists(images_dir_nonneg):
        os.makedirs(images_dir_nonneg)

    for i in range(len(tiff_list)):
        print "Processing %i of %i tiffs." % (int(i+1), len(tiff_list))
        tiff = tf.imread(os.path.join(images_dir, tiff_list[i]))

        # Make tif nonnegative:
        tiff = tiff - tiff.min()

        # Write tif to new directory:
        tf.imsave(os.path.join(images_dir_nonneg, tiff_list[i]), tiff)

    return images_dir_nonneg

#%%
def make_unsigned(images_dir):

    tiff_list = sorted([t for t in os.listdir(images_dir) if t.endswith('tif')], key=natural_keys)

    images_dir_nonneg = '%s_unsigned' % images_dir
    if not os.path.exists(images_dir_nonneg):
        os.makedirs(images_dir_nonneg)

    for i in range(len(tiff_list)):
        print "Processing %i of %i tiffs." % (int(i+1), len(tiff_list))
        tiff = tf.imread(os.path.join(images_dir, tiff_list[i]))

        # Make tif nonnegative:
        #if tiff.min() < 0:
        tiff = tiff + 32768 #- tiff.min()

        # Write tif to new directory:
        tf.imsave(os.path.join(images_dir_nonneg, tiff_list[i]), tiff)

    return images_dir_nonneg

def convert_uint16(images_dir):

    tiff_list = sorted([t for t in os.listdir(images_dir) if t.endswith('tif')], key=natural_keys)

    images_dir_nonneg = '%s_uint16' % images_dir
    if not os.path.exists(images_dir_nonneg):
        os.makedirs(images_dir_nonneg)

    for i in range(len(tiff_list)):
        print "Processing %i of %i tiffs." % (int(i+1), len(tiff_list))
        tiff = tf.imread(os.path.join(images_dir, tiff_list[i]))

        # Make tif nonnegative:
        #if tiff.min() < 0:
        tiff = img_as_uint(tiff)

        # Write tif to new directory:
        tf.imsave(os.path.join(images_dir_nonneg, tiff_list[i]), tiff)

    return images_dir_nonneg

#%%

def get_fissa_object(TID, RID, rootdir='', ncores_prep=2, ncores_sep=4, redo_prep=False, redo_sep=False, append_only=False):
    session_dir = RID['DST'].split('/ROIs/')[0]
    info = get_info_from_tiff_dir(TID['SRC'], session_dir)

    # Get masks:
    rid_dst = RID['DST']
    if rootdir not in rid_dst:
        rid_dst = replace_root(rid_dst, rootdir, info['animalid'], info['session'])
    rois_fpath = os.path.join(rid_dst, 'masks.hdf5')

    # Set output dir:
    tiff_dst_dir = TID['DST']
    if rootdir not in tiff_dst_dir:
        tiff_dst_dir = replace_root(tiff_dst_dir, rootdir, info['animalid'], info['session'])

    output_dir = os.path.join(tiff_dst_dir, 'np_fissa_results')

    # Format roi input as boolean array list:
    roi_list = format_masks_fissa(rois_fpath)

    # Format .tif files to be NON-NEG:

    # Get trace info:
    tiff_src_dir = TID['SRC']
    if rootdir not in tiff_src_dir:
        tiff_src_dir = replace_root(tiff_src_dir, rootdir, info['animalid'], info['session'])
    if not os.path.exists(tiff_src_dir):
        ntiffs_in_src = 0
    else:
        ntiffs_in_src = len([t for t in os.listdir(tiff_src_dir) if t.endswith('tif')])

    #images_dir = '%s_nonnegative' % tiff_src_dir
    images_dir = tiff_src_dir
    if not os.path.exists(images_dir) or not len([t for t in os.listdir(images_dir) if t.endswith('tif')]) == ntiffs_in_src:
        if '_nonnegative' in tiff_src_dir:
            print "Making tif files NONNEGATIVE..."
            src_img_dir = images_dir.split('_nonnegative')[0]
            images_dir = make_nonnegative(src_img_dir)
        elif '_unsigned' in tiff_src_dir:
            print "Making tif files unsigned..."
            src_img_dir = images_dir.split('_unsigned')[0]
            images_dir = make_unsigned(src_img_dir)
        elif '_uint16' in tiff_src_dir:
            print "Making tif files UINT16..."
            src_img_dir = images_dir.split('_uint16')[0]
            images_dir = convert_uint16(src_img_dir)

    # Extract raw & corrected traces with FISSA:
    exp = fissa.Experiment(str(images_dir), roi_list, output_dir, ncores_preparation=ncores_prep, ncores_separation=ncores_sep)
    exp.separate(redo_prep=redo_prep, redo_sep=redo_sep) # To redo:  experiment.separate(redo_prep=True, redo_sep=True)

    # Check that 'means" is stored (not saved if redo):
    if len(exp.means) == 0 and append_only is False:
        #exp.separate(redo_prep=True)
        for trial in range(exp.nTrials):
            curdata = fissa.datahandler.image2array(exp.images[trial])
            exp.means.append(np.array(fissa.datahandler.getmean(curdata)))

    return exp

#%%
def roi_list_to_array(roi_list, normalize_rois=True):

    nrois = len(roi_list) # Only care about 1 file, since they are all the same

    d1, d2 = roi_list[0].shape

    maskarray = np.empty((d1*d2, nrois))

    listarray = np.array(roi_list)
    maskarray = np.reshape(listarray, (nrois, d1*d2), order='C').T

    maskarray = maskarray.astype('float')
    if normalize_rois is True:
        for ridx in range(nrois):
            npixels = len(np.nonzero(maskarray[:,ridx])[0])
            maskarray[:, ridx] = maskarray[:,ridx]/npixels

    return maskarray

#%%
def create_filetraces_from_fissa(exp, TID, RID, si_info, filetraces_dir, rootdir=''):
    file_grp = None

    session_dir = RID['DST'].split('/ROIs/')[0]
    info = get_info_from_tiff_dir(TID['SRC'], session_dir)

    nchannels = si_info['nchannels']
    nslices = 1 #si_info['nslices']
    nvolumes = si_info['nvolumes']
    frames_tsec = si_info['frames_tsec']

    #roi_slices = maskinfo['roi_slices']
    tiff_dir = TID['SRC']
    if rootdir not in tiff_dir:
        tiff_dir = replace_root(tiff_dir, rootdir, info['animalid'], info['session'])

    print "TID %s -- FISSA:  Applying masks to traces..." % TID['trace_hash']
    t_extract = time.time()

    # Set trace output:
    traceid_dir = TID['DST']
    if rootdir not in traceid_dir:
        traceid_dir = replace_root(traceid_dir, rootdir, info['animalid'], info['session'])

    d1, d2 = exp.rois[0][0].shape
    T = exp.raw[0][0].shape[-1]
    dims = (d1, d2, T/nvolumes)

    tiff_fpaths = exp.images
    file_hashdict = dict()
    try:
        for fidx, tfn in enumerate(tiff_fpaths):
            sl = 0

            curr_file = str(re.search(r"File\d{3}", tfn).group())
            if curr_file in TID['PARAMS']['excluded_tiffs']:
                print "***Skipping %s -- excluded from ROI set %s" % (curr_file, RID['roi_id'])
                continue

            print "Formatting traces file (idx %i): %s" % (fidx, curr_file)

            # Create outfile:
            filetraces_fn = '%s_rawtraces_%s.hdf5' % (curr_file, TID['trace_hash'])
            filetraces_filepath = os.path.join(traceid_dir, 'files', filetraces_fn)

            file_grp = h5py.File(filetraces_filepath, 'w')
            file_grp.attrs['source_file'] = tfn
            file_grp.attrs['signal_channel'] = TID['PARAMS']['signal_channel']
            file_grp.attrs['file_id'] = curr_file
            file_grp.attrs['dims'] = (d1, d2, nslices, T/nslices)
            file_grp.attrs['mask_sourcefile'] = exp.folder #MASKS['original_source'] #mask_path

            # Only done on 2D with FISSA, but keep slice group for consistency:
            curr_slice = 'Slice%02d' % int(sl+1)
            if curr_slice not in file_grp.keys():
                slice_grp = file_grp.create_group(curr_slice)
            else:
                slice_grp = file_grp[curr_slice]

            # Get frame tstamps:
            curr_tstamps = np.array(frames_tsec)

            # Save masks:
            #if 'masks' not in slice_grp.keys():
            roi_list = exp.rois
            maskarray = roi_list_to_array(roi_list[0], normalize_rois=True)
            mset = slice_grp.create_dataset('masks', maskarray.shape, maskarray.dtype)
            mset[...] = maskarray
            mset.attrs['roi_id'] = str(RID['roi_id'])
            mset.attrs['rid_hash'] = str(RID['rid_hash'])
            mset.attrs['roi_type'] = str(RID['roi_type'])
            mset.attrs['nr'] = maskarray.shape[-1]
            mset.attrs['nb'] = 0
            mset.attrs['src_roi_idxs'] = np.arange(0, maskarray.shape[-1])

            # Save zproj img:
            print "NFILES: %i" % len(exp.means)
            zproj = exp.means[fidx]
            if 'zproj' not in slice_grp.keys():
                zset = slice_grp.create_dataset('zproj', zproj.shape, zproj.dtype)
            zset[...] = zproj
            zset.attrs['img_source'] = tfn

            # Save fluor trace:
            tracemat = collate_fissa_traces(exp, tiff=fidx, region='cell', trace_type='raw')
            tset = slice_grp.create_dataset('/'.join(['traces', 'raw_fissa']), tracemat.shape, tracemat.dtype)
            tset[...] = tracemat
            tset.attrs['nframes'] = tracemat.shape[0]
            tset.attrs['dims'] = dims

            # Save tstamps:
            if 'frames_tsec' not in slice_grp.keys():
                fset = slice_grp.create_dataset('frames_tsec', curr_tstamps.shape, curr_tstamps.dtype)
            fset[...] = curr_tstamps

            # Save corrected trace:
            tracemat_corr= collate_fissa_traces(exp, tiff=fidx, region='cell', trace_type='corrected')
            tset_corrected = slice_grp.create_dataset('/'.join(['traces', 'np_corrected_fissa']), tracemat_corr.shape, tracemat_corr.dtype)
            tset_corrected[...] = tracemat_corr
            tset_corrected.attrs['nframes'] = tracemat_corr.shape[0]
            tset_corrected.attrs['dims'] = dims
            tset_corrected.attrs['source'] = exp.folder
            tset_corrected.attrs['nregions'] = exp.result[0][fidx].shape[0] - 1
            tset_corrected.attrs['tiff'] = exp.images[fidx]

            # Also save averaged neuropil trace:
            tracemat_np = collate_fissa_traces(exp, tiff=fidx, region='neuropil', trace_type='corrected')
            npil = slice_grp.create_dataset('/'.join(['traces', 'neuropil_fissa']), tracemat_np.shape, tracemat_np.dtype)
            npil[...] = tracemat_np

            # Create hash of current raw tracemat:
            rawfile_hash = hash_file(filetraces_filepath)
            file_hashdict[os.path.splitext(filetraces_fn)[0]] = rawfile_hash

        with open(os.path.join(traceid_dir, 'files', 'filetraces_info_%s.json' % TID['trace_hash']), 'w') as f:
            json.dump(file_hashdict, f, indent=4, sort_keys=True)


    except Exception as e:
        print "--- TID %s: Error extracting traces from file %s ---" % (TID['trace_hash'], curr_file)
        traceback.print_exc()
        print "---------------------------------------------------------------"
    finally:
        if file_grp is not None:
            file_grp.close()
        #maskfile.close()

    #%
    print "TID %s -- Finished compiling trace arrays across files" % TID['trace_hash']
    print_elapsed_time(t_extract)
    print "-----------------------------------------------------------------------"

    return os.path.join(traceid_dir, 'files')

#%%
def append_corrected_fissa(exp, filetraces_dir):

    filetraces_fpaths = sorted([os.path.join(filetraces_dir, t) for t in os.listdir(filetraces_dir) if t.endswith('hdf5')], key=natural_keys)

    #%
    for tfpath in filetraces_fpaths:
        #tfpath = trace_file_paths[0]
        traces_currfile = h5py.File(tfpath, 'r+')
        fidx = int(os.path.split(tfpath)[-1].split('_')[0][4:]) - 1
        print "FISSA -- Appending neurpil corrected traces: File%03d" % int(fidx+1)

        nvolumes = 1
        d1, d2 = exp.rois[0][0].shape
        T = exp.raw[0][0].shape[-1]
        dims = (d1, d2, T/nvolumes)

        try:
            # Append traces:
            for slicekey in traces_currfile.keys():
                if 'raw_fissa' not in traces_currfile[slicekey]['traces'].keys():
                    print "-----> Appending RAW output from FISSA..."
                    # Save fluor trace:
                    tracemat = collate_fissa_traces(exp, tiff=fidx, region='cell', trace_type='raw')
                    tset = traces_currfile[slicekey]['traces'].create_dataset('raw_fissa', tracemat.shape, tracemat.dtype)
                    tset[...] = tracemat
                    tset.attrs['nframes'] = tracemat.shape[0]
                    tset.attrs['dims'] = dims
                else:
                    print "-----> Raw FISSA output already saved."

                if 'np_corrected_fissa' not in traces_currfile[slicekey]['traces'].keys():
                    print "-----> Appending CORRECTED output from FISSA..."
                    # Collate FISSA results to standard trace  mat form: MxN, where M = frame tpoints, N = roi
                    tracemat_corr = collate_fissa_traces(exp, tiff=fidx, region='cell', trace_type='corrected')
                    corr = traces_currfile[slicekey]['traces'].create_dataset('np_corrected_fissa', tracemat_corr.shape, tracemat_corr.dtype)
                    corr[...] = tracemat_corr
                    corr.attrs['nframes'] = tracemat.shape[0]
                    corr.attrs['dims'] = dims
                    corr.attrs['source'] = exp.folder
                    corr.attrs['nregions'] = exp.result[0][fidx].shape[0] - 1
                    corr.attrs['tiff'] = exp.images[fidx]

                    tracemat_np = collate_fissa_traces(exp, tiff=fidx, region='neuropil', trace_type='corrected')
                    npil = traces_currfile[slicekey]['traces'].create_dataset('neuropil_fissa', tracemat_np.shape, tracemat_np.dtype)
                    npil[...] = tracemat_np
                else:
                    print "-----> Corrected FISSA output already saved."

        except:
            print "** ERROR appending NP-corrected traces: %s" % traces_currfile
            traceback.print_exc()
        finally:
            traces_currfile.close()

    return filetraces_dir

#%%

def append_neuropil_subtraction(maskdict_path, cfactor, filetraces_dir, create_new=False, rootdir=''):

    #signal_channel_idx = int(TID['PARAMS']['signal_channel']) - 1 # 0-indexing into tiffs

    MASKS = h5py.File(maskdict_path, 'r')

    filetraces_fpaths = sorted([os.path.join(filetraces_dir, t) for t in os.listdir(filetraces_dir) if t.endswith('hdf5')], key=natural_keys)

    #
    print "Appending subtrated NP traces to %i files." % len(filetraces_fpaths)
    for tfpath in filetraces_fpaths:
        #tfpath = trace_file_paths[0]
        traces_currfile = h5py.File(tfpath, 'r+')
        #print "FILETRACES attrs:"
        #print traces_currfile.attrs.keys()
        fidx = int(os.path.split(tfpath)[-1].split('_')[0][4:]) - 1
        curr_file = "File%03d" % int(fidx+1)
        print "CFACTOR -- Appending neurpil corrected traces: %s" % curr_file
        try:
            for curr_slice in traces_currfile.keys():

                tracemat = np.array(traces_currfile[curr_slice]['traces']['raw'])

                # First check that neuropil traces don't already exist:
                if 'neuropil' in traces_currfile[curr_slice]['traces'].keys() and create_new is False:
                    np_tracemat = np.array(traces_currfile[curr_slice]['traces']['neuropil'])
                    overwrite_neuropil = False
                else:
                    overwrite_neuropil = True

                if 'np_subtracted' in traces_currfile[curr_slice]['traces'].keys() and create_new is False:
                    np_correctedmat = np.array(traces_currfile[curr_slice]['traces']['np_subtracted'])
                    if np.mean(np_correctedmat) == 0:
                        overwrite_correctedmat = True
                    else:
                        overwrite_correctedmat = False

                if overwrite_neuropil is True:
                    overwrite_correctedmat = True # always overwrite tracemat if new neuropil

                    # Load tiff:
                    tiffpath = traces_currfile.attrs['source_file']
                    print "Calculating neuropil from src: %s" % tiffpath

                    if rootdir not in tiffpath:
                        session_dir = os.path.split(os.path.split(filetraces_dir.split('/traces')[0])[0])[0]
                        info = get_info_from_tiff_dir(os.path.split(tiffpath)[0], session_dir)
                        tiffpath = replace_root(tiffpath, rootdir, info['animalid'], info['session'])

                    tiff = tf.imread(tiffpath)
                    T, d1, d2 = tiff.shape
                    d = d1*d2
                    orig_mat_shape = traces_currfile[curr_slice]['traces']['raw'].shape
                    #orig_dims = traces_currfile.attrs['dims'] # (d1, d2, nslices, T)
                    nchannels = T/orig_mat_shape[0]
                    signal_channel_idx = int(traces_currfile.attrs['signal_channel']) - 1

                    tiffR = np.reshape(tiff, (T, d), order='C'); del tiff
                    tiffslice = tiffR[signal_channel_idx::nchannels,:]
                    print "SLICE shape is:", tiffslice.shape

                    np_maskarray = MASKS[curr_file][curr_slice]['np_maskarray'][:]
                    np_tracemat = tiffslice.dot(np_maskarray)

                if overwrite_correctedmat is True:
                    np_correctedmat = tracemat - (cfactor * np_tracemat)

                if 'neuropil' not in traces_currfile[curr_slice]['traces'].keys():
                    np_traces = traces_currfile.create_dataset('/'.join([curr_slice, 'traces', 'neuropil']), np_tracemat.shape, np_tracemat.dtype)
                else:
                    np_traces = traces_currfile[curr_slice]['traces']['neuropil']
                np_traces[...] = np_tracemat
                if 'np_subtracted' not in traces_currfile[curr_slice]['traces'].keys():
                    np_corrected = traces_currfile.create_dataset('/'.join([curr_slice, 'traces', 'np_subtracted']), np_correctedmat.shape, np_correctedmat.dtype)
                else:
                    np_corrected = traces_currfile[curr_slice]['traces']['np_subtracted']
                np_corrected[...] = np_correctedmat
                np_corrected.attrs['correction_factor'] = cfactor
        except Exception as e:
            print "** ERROR appending NP-subtracted traces: %s" % traces_currfile
            print traceback.print_exc()
        finally:
            traces_currfile.close()

    return filetraces_dir


#%%
def extract_options(options):
    choices_npmethod = ('fissa', 'subtract')
    default_npmethod = 'subtract'

    choices_tracetype = ('raw', 'raw_fissa', 'denoised_nmf', 'np_corrected_fissa', 'neuropil_fissa', 'np_subtracted', 'neuropil')
    default_tracetype = 'raw'

    parser = optparse.OptionParser()

    # PATH opts:
    parser.add_option('-D', '--root', action='store', dest='rootdir', default='/nas/volume1/2photon/data', help='data root dir (root project dir containing all animalids) [default: /nas/volume1/2photon/data, /n/coxfs01/2pdata if --slurm]')
    parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')
    parser.add_option('-S', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID')
    parser.add_option('-A', '--acq', action='store', dest='acquisition', default='FOV1', help="acquisition folder (ex: 'FOV1_zoom3x') [default: FOV1]")
    parser.add_option('-R', '--run', action='store', dest='run', default='', help="name of run dir containing tiffs to be processed (ex: gratings_phasemod_run1)")
    parser.add_option('--default', action='store_true', dest='default', default='store_false', help="Use all DEFAULT params, for params not specified by user (no interactive)")
    parser.add_option('--slurm', action='store_true', dest='slurm', default=False, help="set if running as SLURM job on Odyssey")
    parser.add_option('-t', '--trace-id', action='store', dest='trace_id', default='', help="Trace ID for current trace set (created with set_trace_params.py, e.g., traces001, traces020, etc.)")

    parser.add_option('--new', action="store_true",
                      dest="create_new", default=False, help="Set flag to create new output files (/paradigm/parsed_frames.hdf5, roi_trials.hdf5")
    parser.add_option('--append', action="store_true",
                      dest="append_trace_type", default=False, help="Set flag to append non-default trace type to trace structs (e.g., neuropil correction).")

    parser.add_option('--np', type='choice', choices=choices_npmethod, action='store', dest='np_method', default=default_npmethod, help="Method for neuropil correction. Valid choices: %s [default: %s]" % (choices_npmethod, default_npmethod))

    parser.add_option('-N', '--ncores', action="store",
                      dest="ncores", default=2, help="[np-fissa]: N cores to use for FISSA prep and separation [default: 2, 4. If slurm, 1]")

    parser.add_option('-a', '--halo', action="store",
                      dest="np_niterations", default=3, help="[np-subtract]:  N iterations for ROI dilation when creating annulus for neuropil [default: 3]")
    parser.add_option('-c', '--cfactor', action="store",
                      dest="np_correction_factor", default=0.5, help="[np-subtract]: Correction factor for neuropil subtraction [default: 0.5]")

    parser.add_option('--neuropil', action="store_true",
                      dest="neuropil", default=False, help="Set flag to extract neuropil.")


    parser.add_option('--collate', action="store_true",
                      dest="create_dataframe", default=False, help="Set flag to collate traces into dataframe (and extract filtered traces, if params set).")
    parser.add_option('-T', '--trace-type', type='choice', choices=choices_tracetype, action='store', dest='trace_type', default=default_tracetype, help="Type of timecourse to plot PSTHs. Valid choices: %s [default: %s]" % (choices_tracetype, default_tracetype))

    parser.add_option('--warp', action="store_true",
                      dest="save_warp_images", default=False, help="Set flag to save output plots of warped ROIs (manual warp only).")


    # Pupil filtering info:
    parser.add_option('--no-pupil', action="store_false",
                      dest="filter_pupil", default=True, help="Set flag NOT to filter PSTH traces by pupil threshold params")
    parser.add_option('-s', '--radius-min', action="store",
                      dest="pupil_radius_min", default=25, help="Cut-off for smnallest pupil radius, if --pupil set [default: 25]")
    parser.add_option('-B', '--radius-max', action="store",
                      dest="pupil_radius_max", default=65, help="Cut-off for biggest pupil radius, if --pupil set [default: 65]")
    parser.add_option('-d', '--dist', action="store",
                      dest="pupil_dist_thr", default=5, help="Cut-off for pupil distance from start, if --pupil set [default: 5]")
#

    (options, args) = parser.parse_args(options)

    return options

#%%

def get_tiff_source(TID, rootdir, animalid, session):
    trace_hash = TID['trace_hash']
    tiff_dir = TID['SRC']
    roi_name = TID['PARAMS']['roi_id']
    if rootdir not in tiff_dir:
        tiff_dir = replace_root(tiff_dir, rootdir, animalid, session)

    if '_nonnegative' in tiff_dir and not os.path.exists(tiff_dir):
        print "Making tif files NONNEGATIVE..."
        orig_tiff_dir = tiff_dir.split('_nonnegative')[0]
        tiff_dir = make_nonnegative(orig_tiff_dir)
    elif '_unsigned' in tiff_dir and not os.path.exists(tiff_dir):
        print "Making tif files psuedo unsigned..."
        orig_tiff_dir = tiff_dir.split('_unsigned')[0]
        tiff_dir = make_unsigned(orig_tiff_dir)
    elif '_uint16' in tiff_dir and not os.path.exists(tiff_dir):
        print "Making tif files UINT16..."
        orig_tiff_dir = tiff_dir.split('_uint16')[0]
        tiff_dir = convert_uint16(orig_tiff_dir)

    return tiff_dir


#%%
def load_TID_roiset(TID, rootdir):

    rundir = TID['DST'].split('/traces')[0]
    sessiondir =  os.path.split(os.path.split(rundir)[0])[0]
    session = os.path.split(sessiondir)[-1]
    animalid = os.path.split(os.path.split(sessiondir)[0])[-1]

    # Load ROI set specified in Traces param set:
    roi_dir = os.path.join(rootdir, animalid, session, 'ROIs')
    roidict_path = os.path.join(roi_dir, 'rids_%s.json' % session)
    with open(roidict_path, 'r') as f:
        roidict = json.load(f)
    RID = roidict[TID['PARAMS']['roi_id']]

    if rootdir not in RID['DST']:
        RID['DST'] = replace_root(RID['DST'], rootdir, animalid, session)
        notnative = True

    return RID

#%%
def create_formatted_maskfile(TID, RID, nslices=1, save_warp_images=True,
                              np_method=None, do_neuropil_correction=True,
                              plot_neuropil=True, niter=3, rootdir='',
                              create_new=False):
    '''
    For each specified SLICE in this ROI set, create 2D mask array.
    Save as standardized format to disk...

    Note:  This may be a bit redundant with standard_rois() steps in rois/get_rois.py
    but addresses slight differences in how hdf5 mask files are structured across
    roi types.

    TODO:  Need to make MATLAB (manual methods) HDF5 output structure the same
    as python-based methods... Ambiguity comes from 2D vs. 3D Rois, and whether
    to define ROIs per-slice, or per-volume.

    '''



    # Get mask info (is3D, normalize, by-slice, etc.):
    #mask_path = os.path.join(RID['DST'], 'masks.hdf5')
    #maskinfo = get_mask_info(mask_path, nslices=nslices, rootdir=rootdir, excluded_tiffs=TID['PARAMS']['excluded_tiffs'])
    maskinfo = get_mask_info(TID, RID, nslices=nslices, rootdir=rootdir)

    # Check if formatted MASKS dict exists and load, otherwise, create new:
    maskdict_path = os.path.join(TID['DST'], 'MASKS.hdf5')
    if create_new is True or not os.path.exists(maskdict_path):
        maskdict_path = get_masks(maskdict_path, maskinfo, RID,
                                  save_warp_images=save_warp_images,
                                  do_neuropil_correction=do_neuropil_correction,
                                  niter=niter,
                                  rootdir=rootdir)

    # Check if alrady have plotted masks, if not, create new:
    mask_figdir = os.path.join(TID['DST'], 'figures', 'masks')
    if not os.path.exists(mask_figdir):
        os.makedirs(mask_figdir)

    maskfigs = [i for i in os.listdir(mask_figdir) if 'rois_File' in i and i.endswith('png') and 'np_' not in i]
    if create_new is True or not len(maskfigs)==len(maskinfo['filenames']):
        if len(maskfigs) > 0:
            print "Removing old mask files..."
        for f in maskfigs:
            os.remove(os.path.join(mask_figdir, f))
        print "Plotting new mask figures."
        plot_roi_masks(TID, RID, plot_neuropil=False, mask_figdir=mask_figdir, rootdir=rootdir)
        maskfigs = [i for i in os.listdir(mask_figdir) if 'rois_File' in i and i.endswith('png')]

    if np_method=='subtract' and plot_neuropil is True:
        np_maskfigs = [i for i in os.listdir(mask_figdir) if 'rois_File' in i and i.endswith('png') and 'np_iter%i' % niter in i]
        if create_new is True or len(np_maskfigs) != len(maskfigs):
            plot_roi_masks(TID, RID, plot_neuropil=plot_neuropil, mask_figdir=mask_figdir, rootdir=rootdir)

    return maskinfo, maskdict_path

#%%
#options = ['-D', '/mnt/odyssey', '-i', 'CE077', '-S', '20180412', '-A', 'FOV1_zoom1x', '-R', 'blobs_run3',
#        '-t', 'traces001', '--np=subtract', '--neuropil', '--append', '--no-pupil', '--warp']

#%%
def extract_traces(options):
#    options = ['-D', '/mnt/odyssey', '-i', 'CE074', '-S', '20180215', '-A', 'FOV2_zoom1x_LI', '-R', 'blobs',
#            '-t', 'traces003', '--np-method=subtract', '--neuropil', '--append', '--no-pupil']
#
    # Set USER INPUT options:
    options = extract_options(options)

    rootdir = options.rootdir
    animalid = options.animalid
    session = options.session
    acquisition = options.acquisition
    run = options.run
    trace_id = options.trace_id
    slurm = options.slurm
    auto = options.default
    create_new = options.create_new

    if slurm is True:
        if 'coxfs01' not in rootdir:
            rootdir = '/n/coxfs01/2p-data'
        ncores = 1
        ncores_sep = 1
    else:
        ncores = int(options.ncores)
        ncores_sep = ncores * 2

    # Neuropil correction:
    append_trace_type = options.append_trace_type
    np_method = options.np_method
    np_niterations = options.np_niterations
    do_neuropil_correction = options.neuropil
    if do_neuropil_correction:
        plot_neuropil = True
    else:
        plot_neuropil = False

    if np_method == 'subtract':
        subtract_neuropil = True
    else:
        subtract_neuropil = False
    np_correction_factor = float(options.np_correction_factor)

    if np_method == 'fissa':
        print "Requesting %i cores prep, %i cores sep for FISSA." % (ncores, ncores_sep)
    elif np_method == 'subtract':
        print "Requesting NP subtraction method."
        print "... Specified %i iterations for annulus size." % np_niterations
        print "... Correction factor = %.2f" % np_correction_factor
    save_warp_images = options.save_warp_images

    # Trace alignment params:
    create_dataframe = options.create_dataframe
    trace_type = options.trace_type
    filter_pupil = options.filter_pupil
    pupil_radius_max = float(options.pupil_radius_max)
    pupil_radius_min = float(options.pupil_radius_min)
    pupil_dist_thr = float(options.pupil_dist_thr)


    print "======================================================================="
    print "Trace Set: %s -- Starting trace extraction..." % trace_id
    t_start = time.time()


    #% Get meta info for run:
    # =============================================================================
    run_dir = os.path.join(rootdir, animalid, session, acquisition, run)
    if rootdir not in run_dir:
        run_dir = replace_root(run_dir, rootdir, animalid, session)
    print "RUN:", run_dir
    si_info = get_frame_info(run_dir)

    # Load specified trace-ID parameter set:
    # =============================================================================
    TID = load_TID(run_dir, trace_id, auto=auto)
    if rootdir not in TID['DST']:
        TID['DST'] = replace_root(TID['DST'], rootdir, animalid, session)

    #%
    # Get source tiff paths using trace-ID params:
    # =============================================================================
    tiff_dir = get_tiff_source(TID, rootdir, animalid, session)
    tiff_files = sorted([t for t in os.listdir(tiff_dir) if t.endswith('tif')], key=natural_keys)
    print "Found %i tiffs in dir %s.\nExtracting traces with ROI set %s." % (len(tiff_files), tiff_dir, TID['PARAMS']['roi_id'])

    #%
    # Create output dirs and files:
    # =============================================================================
    filetraces_dir = os.path.join(TID['DST'], 'files')
    if not os.path.exists(filetraces_dir):
        os.makedirs(filetraces_dir)

    trace_figdir = os.path.join(TID['DST'], 'figures')
    if not os.path.exists(trace_figdir):
        os.makedirs(trace_figdir)

    #%
    # Create mask array and save mask images for each slice specified in ROI set:
    # =============================================================================
    RID = load_TID_roiset(TID, rootdir)

    #% For each specified SLICE in this ROI set, create 2D mask array:
    # TODO:  Need to make MATLAB (manual methods) HDF5 output structure the same
    # as python-based methods... if-checks hacked for now...
    print "TID %s -- Getting mask info..." % TID['trace_hash']
    print "--------------------------------------------------------------------"
    t_mask = time.time()
    maskinfo, maskdict_path = create_formatted_maskfile(TID, RID,
                                                        nslices=si_info['nslices'],
                                                        save_warp_images=save_warp_images,
                                                        np_method=np_method,
                                                        do_neuropil_correction=do_neuropil_correction,
                                                        plot_neuropil=plot_neuropil,
                                                        niter=np_niterations,
                                                        rootdir=rootdir,
                                                        create_new=create_new)

    print "TID %s - Got mask info from ROI set %s." % (TID['trace_hash'], RID['roi_id'])
    print_elapsed_time(t_mask)
    print "-----------------------------------------------------------------------"

    #%
    # Apply masks to .tif files:
    print "*** Extracting traces from each file."
    print "-----------------------------------------------------------------------"
    filetraces_fns = [f for f in os.listdir(filetraces_dir) if f.endswith('hdf5')]
    mismatch= []
    if si_info['nslices'] > 1:
        for slicename in ['Slice%02d' % int(i+1) for i in range(si_info['nslices'])]:
            maskfig_dir = os.path.join(trace_figdir, 'masks', slicename)
            maskfigs = [i for i in os.listdir(maskfig_dir) if i.endswith('png') and 'np_' not in i]
            print "...... N=%i trace files found (expecting %i)." % (len(filetraces_fns), len(maskfigs))
            mismatch.append(len(filetraces_fns) != len(maskfigs))
    else:
        maskfigs = [i for i in os.listdir(os.path.join(trace_figdir, 'masks')) if i.endswith('png') and 'np_' not in i]
        print "...... N=%i trace files found (expecting %i)." % (len(filetraces_fns), len(maskfigs))
        mismatch.append(len(filetraces_fns) != len(maskfigs))

    if any(mismatch):
        create_new = True

    # 1)  If np_method == 'fissa', raw trace extraction and neuropil-subtraction
    # happens simultaneously (unless "--append" is specified, i.e., raw traces
    # were already extracted).
    if np_method!='fissa' and create_new is True:
        if len(filetraces_fns)==0:
            print "...... Creating new file-trace files."
        else:
            print "...... Rewriting file-trace files."
        filetraces_dir = apply_masks_to_movies(TID, RID, si_info,
                                                  do_neuropil_correction=do_neuropil_correction,
                                                  cfactor=np_correction_factor,
                                                  output_filedir=filetraces_dir,
                                                  rootdir=rootdir)
        create_new = False # Re-toggle create-new, since traces now extracted.
        if np_method=='subtract':
            append_trace_type = True


    # 2)  Do neuropil correction (& raw trace extraction) using FISSA method:
    if do_neuropil_correction is True and np_method == 'fissa':
        if create_new is True:
            redo_prep=True; redo_sep=True
        else:
            redo_prep=False; redo_sep=False
        exp = get_fissa_object(TID, RID, rootdir=rootdir,
                               ncores_prep=ncores, ncores_sep=ncores_sep,
                               redo_prep=redo_prep, redo_sep=redo_sep)
        print "N files:", len(exp.rois)
        print "N rois:", len(exp.rois[0])
        print "N results:", len(exp.result)
        print "N means:", len(exp.means)

        # Create TRACEFILE files for each .tif, if none exist yet:
        filetraces_fpaths = sorted([os.path.join(filetraces_dir, t)
                                        for t in os.listdir(filetraces_dir)
                                        if t.endswith('hdf5')], key=natural_keys)
        if not len(filetraces_fpaths) == len(maskfigs):
            filetraces_dir = create_filetraces_from_fissa(exp, TID, RID, si_info, filetraces_dir, rootdir=rootdir)

            # Set append NP-correction flag to FALSE:
            append_trace_type = False
            create_new = True
        else:
            append_trace_type = True
            create_new = True


    # Append non-raw traces, if relevant:
    update_roi_timecourses = False
    if append_trace_type:
        update_roi_timecourses = True
        print "Appending trace type!"
        if np_method == 'fissa':
            exp = get_fissa_object(TID, RID, rootdir=rootdir,
                                   ncores_prep=ncores, ncores_sep=ncores_sep,
                                   append_only=True)
            filetraces_dir = append_corrected_fissa(exp, filetraces_dir)
        elif np_method == 'subtract':
            # First make sure that MASKS.hdf5 in traceid dir contains np info:
            print "Checking MASKS.hdf5 for neuropil masks..."
            mtmp = h5py.File(maskdict_path, 'r')
            lev1 = mtmp.keys()[0]
            lev2 = mtmp[lev1].keys()[0]
            if 'np_maskarray' not in mtmp[lev1][lev2].keys():
                mtmp.close()
                print "--- Unable to find NEUROPIL mask info. Creating new."
                maskdict_path = get_masks(maskdict_path, maskinfo, RID,
                                          do_neuropil_correction=True,
                                          niter=np_niterations,
                                          rootdir=rootdir)

            print "--- Using SUBTRACTION method, (global) correction-factor: ", np_correction_factor
            filetraces_dir = append_neuropil_subtraction(maskdict_path,
                                                         np_correction_factor,
                                                         filetraces_dir,
                                                         create_new=create_new,
                                                         rootdir=rootdir)

    #%
    # Organize timecourses by stim-type for each ROI:
    # -----------------------------------------------
    print "*** Creating ROI-TCOURSE file...."
    print "-----------------------------------------------------------------------"
    roi_tcourse_filepath = get_roi_timecourses(TID, RID, si_info,
                                               input_filedir=filetraces_dir,
                                               rootdir=rootdir,
                                               create_new=update_roi_timecourses)
    print "-----------------------------------------------------------------------"

    #% move tmp file and clean up:
    tmp_tid_fn = 'tmp_tid_%s.json' % TID['trace_hash']
    tmp_tid_dir = os.path.join(run_dir, 'traces', 'tmp_tids')

    completed_tid_dir = os.path.join(tmp_tid_dir, 'completed')
    if not os.path.exists(completed_tid_dir):
        os.makedirs(completed_tid_dir)
    if os.path.exists(os.path.join(tmp_tid_dir, tmp_tid_fn)):
        os.rename(os.path.join(tmp_tid_dir, tmp_tid_fn), os.path.join(completed_tid_dir, tmp_tid_fn))
    print "Cleaned up tmp tid files."

    #%
    print "*** TID %s *** COMPLETED TRACE EXTRACTION!" % TID['trace_hash']
    print_elapsed_time(t_start)
    print "======================================================================="
    traceid_dir = TID['DST']

    roidata_filepath = None
    if create_dataframe is True:
        print "*** Creating ROI dataframes ***"
        if append_trace_type is True:
            create_new = True

        # Assign frame indices for specified trial epochs:
        # =====================================================================
        roidata_filepath, roistats_filepath = acq.align_roi_traces(trace_type, TID, si_info, traceid_dir, run_dir,
                                                           create_new=create_new,
                                                           filter_pupil=filter_pupil,
                                                           pupil_radius_min=pupil_radius_min,
                                                           pupil_radius_max=pupil_radius_max,
                                                           pupil_dist_thr=pupil_dist_thr)

    return roi_tcourse_filepath, roidata_filepath


#%%
#rdata = h5py.File(roi_tcourse_filepath, 'r')
#pl.figure(); sns.distplot(np.diff(rdata['roi00001']['frames_tsec']))
#Out[321]: <matplotlib.axes._subplots.AxesSubplot at 0x7f3dc962ead0>
#
#np.diff(rdata['roi00001']['frames_tsec'])
#Out[322]:
#array([ 0.06721005,  0.0672101 ,  0.0672102 , ...,  0.0672498 ,
#        0.0672498 ,  0.0672498 ])
#np.diff(rdata['roi00001']['frames_tsec']).min()
#Out[323]: 0.067208268739022969
#np.diff(rdata['roi00001']['frames_tsec']).max()
#Out[324]: 0.067249800000013238
#
#1/si_info['framerate']
#Out[325]: 0.022375018739078194
#
#1/si_info['volumerate']
#Out[326]: 0.06712490602513156

#%% GET PLOTS:
def main(options):
    #options = extract_options(options)
    roi_tcourse_filepath, roidata_filepath = extract_traces(options)
    print "DONE extracting traces!"
    print "Output saved to:\n---> %s" % roi_tcourse_filepath
    if roidata_filepath is not None:
        print "Aligned traces to trial events. Saved dataframe to:\n%s" % roidata_filepath

if __name__ == '__main__':
    main(sys.argv[1:])

