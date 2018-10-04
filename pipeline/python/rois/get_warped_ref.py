#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 16:50:26 2018

@author: juliana
"""

import matplotlib
matplotlib.use('agg')
import os
import glob
import cv2
import re
import sys
import hashlib
import shutil
import json
import optparse
import numpy as np
import pylab as pl
import cPickle as pkl
import tifffile as tf
from scipy.ndimage import zoom
from pipeline.python.utils import write_dict_to_json, get_tiff_paths, replace_root


def get_gradient(im):
    # Calculate the x and y gradients using Sobel operator
    grad_x = cv2.Sobel(im,cv2.CV_32F,1,0,ksize=3)
    grad_y = cv2.Sobel(im,cv2.CV_32F,0,1,ksize=3)

    # Combine the two gradients
    grad = cv2.addWeighted(np.absolute(grad_x), 0.5, np.absolute(grad_y), 0.5, 0)
    return grad

def warp_images(stack, ref, warp_mode=cv2.MOTION_HOMOGRAPHY):

    height, width = stack[0].shape
    
    # Allocate space for aligned image

    # Define motion mode. # Set the warp matrix to identity.
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Set the stopping criteria for the algorithm.
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000,  1e-6)
    warps = {}
    for image_ix, img in enumerate(stack):
        print image_ix
        sample = img.copy()
        if (img == ref).all():
            aligned_sample = sample.copy()
            mode_str = 'None'
        else:
            aligned_sample = np.zeros((height,width), dtype=sample.dtype) #dtype=np.uint8 )
            
            # Warp sample to ref:
            (cc, warp_matrix) = cv2.findTransformECC (get_gradient(ref), get_gradient(sample), warp_matrix, warp_mode, criteria)
            
            if warp_mode == cv2.MOTION_HOMOGRAPHY :
                # Use Perspective warp when the transformation is a Homography
                aligned_sample = cv2.warpPerspective(sample, warp_matrix, (width,height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                mode_str = 'MOTION_HOMOGRAPHY'
            else:
                # Use Affine warp when the transformation is not a Homography
                aligned_sample = cv2.warpAffine(sample, warp_matrix, (width, height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
                mode_str = 'WARP_AFFINE'
        
        warps[image_ix] = {'aligned': aligned_sample,
                           'mode_str': mode_str,
                           'corrcoef': np.corrcoef(sample.ravel(), ref.ravel())[0,1],
                           'corrcoef_aligned': np.corrcoef(aligned_sample.ravel(), ref.ravel())[0,1]}
        
    return warps
        

#%%
#rootdir = '/mnt/odyssey'
#animalid = 'JC015'
#session = '20180917'
#acquisition = 'FOV1_zoom2p0x'
##channel = 'Channel01'
##pid = 'processed001'
##zproj = 'std'
#roi_id = 'rois002'
    

from PIL import Image, ImageEnhance

def convert_range(img, min_new=0, max_new=255):
    img_new = (img - img.min()) * ((max_new - min_new) / (img.max() - img.min())) + min_new
    return img_new

def enhance_image_and_save(warped_mean, warped_mean_image_path, factor=2.0):
    # Get brightness range - i.e. darkest and lightest pixels
    #img = np.array(Image.fromarray(warped_mean).convert("L"))
    img = convert_range(warped_mean)
    img = np.array(Image.fromarray(img).convert("L"))
    
    minval=np.min(img)        # result=144
    maxval=np.max(img)        # result=216
    
    # Make a LUT (Look-Up Table) to translate image values
    LUT=np.zeros(256,dtype=np.uint8)
    LUT[minval:maxval+1]=np.linspace(start=0,stop=255,num=(maxval-minval)+1,endpoint=True,dtype=np.uint8)
    
    # Apply LUT, enhance contrast, and save resulting image
    enhancer_object = ImageEnhance.Contrast(Image.fromarray(LUT[img]).convert("L"))
    out = enhancer_object.enhance(factor)
    print "SAVING:", warped_mean_image_path
    out.save(warped_mean_image_path)
        


def warp_runs_in_fov(acquisition_dir, roi_id, warp_threshold=0.7, enhance_factor=2.0, create_new=False):
    
    # Load RID:
    session_dir = os.path.split(acquisition_dir)[0]
    print "Getting ROI info from session dir: %s" % session_dir
    roidict_filepath = glob.glob(os.path.join(os.path.split(acquisition_dir)[0], 'ROIs', 'rids_*.json'))[0]
    with open(roidict_filepath, 'r') as f: rids = json.load(f)
    RID = rids[roi_id]
    
    channel = 'Channel%02d' % RID['PARAMS']['options']['ref_channel']
    zproj = RID['PARAMS']['options']['zproj_type']
    pid = str(re.search('processed(\d{3})', RID['SRC']).group(0))
    roi_output_dir = RID['DST']
    
    warped_mean_image_path = os.path.join(RID['DST'], 'warped_mean_reference.tif')
    session = os.path.split(session_dir)[1]
    animalid_dir = os.path.split(session_dir)[0]
    animalid = os.path.split(animalid_dir)[1]
    rootdir = os.path.split(animalid_dir)[0]
    print "ROOTDIR:", rootdir
    print "ANIMALID:", animalid
    if rootdir not in warped_mean_image_path:
        warped_mean_image_path = replace_root(warped_mean_image_path, rootdir, animalid, session)
    warp_results_path = os.path.join(RID['DST'], 'warp_results.pkl')
    if rootdir not in warp_results_path:
       warp_results_path = replace_root(warp_results_path, rootdir, animalid, session)

    if os.path.exists(warp_results_path) and create_new is False:
        action = raw_input("Warp results exist. Press <R> to re-warp, <I> to remake image, and <ENTER> to escape: ")
        if action == 'R':
            create_new = True
        elif action == 'I':
            # Load warp results, and remake image.
            redraw = raw_input("Do you want to re-draw reference img? Press <Y> to re-enhance, and <ENTER> to escape: ")
            if redraw == 'Y':
                new_factor = float(raw_input("... Enter enhancing factor (default: 1.2): "))
                with open(warp_results_path, 'rb') as f: warp_results = pkl.load(f)
                aligned_stack = np.dstack([results['aligned'] for ix, results in warp_results['warps'].items()])
                warped_mean = np.mean(aligned_stack, axis=-1)
                warped_mean = enhance_image_and_save(warped_mean, warped_mean_image_path, factor=new_factor)
                return warp_results
            else:
                return warp_results #None
    else:
        creat_new = True #return None
            
    # Warp all ZPROJ images to a single reference.
    # -----------------------------------------------------------------------------

    if create_new:
        primary_warp = cv2.MOTION_HOMOGRAPHY
        secondary_warp = cv2.MOTION_AFFINE
        
        # Get a list of ALL zproj images for all runs in acquisition:
        img_paths = glob.glob(os.path.join(acquisition_dir, '*run*', 'processed', '%s*' % pid, \
                                           'mcorrected_*_%s_deinterleaved' % zproj, channel,  'File*', '*.tif'))
        nfiles_total = len(img_paths)
        print "TOTAL N IMAGES (across all runs): %i" % nfiles_total

        stack = []
        for i,imgp in enumerate(img_paths):
            img = tf.imread(imgp)
            if len(img.shape) == 3:
                std_img = np.empty((img.shape[1], img.shape[2]), dtype=img.dtype)
                std_img[:] = np.std(img, axis=0)
                tf.imsave(imgp, std_img)
                stack.append(std_img)
            else:
                stack.append(img)
        
        # Select reference and warp each image to it:
        reference_ix = nfiles_total/2
        ref = stack[reference_ix]
        warps = warp_images(stack, ref=ref, warp_mode=primary_warp)
        
        # Check for bad warps:
        bad_warps = [image_ix for image_ix, results in warps.items() \
                     if (results['corrcoef_aligned'] < results['corrcoef'] \
                     and results['corrcoef_aligned'] < warp_threshold)]
        
        retry_warps = {}; still_bad_warps= [];
        if len(bad_warps) > 0:
            print "Retrying warps using secondary warp mode for %i images." % len(bad_warps)
            retry_warps = warp_images(stack[np.array(bad_warps)], ref=ref, warp_mode=secondary_warp)
            still_bad_warps = [bad_warps[image_ix] for image_ix, results in retry_warps.items() \
                                   if results['corrcoef_aligned'][0,1] < warp_threshold]
        if len(still_bad_warps) > 0:
            print "Found %i files that fail %.2f threshold using both warp modes." 
        
        # Keep good warps:
        for bad_ix, image_ix in enumerate(bad_warps):
            warps[image_ix] = retry_warps[bad_ix]
        
        aligned_stack = np.dstack([results['aligned'] for ix, results in warps.items()])
        
        # Save summed warp image for ROI extraction img:
        #warped_mean = np.zeros((aligned_stack.shape[0], aligned_stack.shape[1]), dtype='uint16')
        warped_mean = np.mean(aligned_stack, axis=-1)
        warped_mean = enhance_image_and_save(warped_mean, warped_mean_image_path, factor=enhance_factor)
        
        fig, axes = pl.subplots(1,3, figsize=(15,5))
        ref_run = img_paths[reference_ix].split(acquisition_dir)[1].split('/')[1]
        ref_file = img_paths[reference_ix].split(acquisition_dir)[1].split('/')[-2]
        axes[0].imshow(ref); axes[0].set_title('reference\n(%s, %s)' % (ref_run, ref_file))
        axes[1].imshow(np.mean(aligned_stack, axis=-1)); axes[1].set_title('mean aligned')
        axes[2].imshow(np.sum(aligned_stack, axis=-1)); axes[2].set_title('sum aligned')
        pl.savefig(os.path.join(roi_output_dir, 'reference_to_aligned.png'))
        pl.close()
        
        # Save warp results to file:
        warp_results = {'warps': warps,
                        'warps_attempt2': retry_warps,
                        'ref': ref,
                        'reference_file': img_paths[reference_ix],
                        'warp_threshold': warp_threshold,
                        'failed_warps': [img_paths[bi] for bi in still_bad_warps],
                        'roi_reference_file': warped_mean_image_path}
        
        with open(warp_results_path, 'wb') as f:
            pkl.dump(warp_results, f, protocol=pkl.HIGHEST_PROTOCOL)
            
        # Save ROI INFO:
        # -------------------------------------------------------------------------
        # EFfectively usinga specific STD file of a specific run as "reference":
        reference_filepath = warp_results['reference_file']
        tiff_source_dir = reference_filepath.split('_std')[0]
        RID['SRC'] = warped_mean_image_path
        RID['PARAMS']['tiff_sourcedir'] = tiff_source_dir
        RID['PARAMS']['options']['ref_file'] = int(str(re.search('File(\d{3})', reference_filepath).group(0))[4:])
        RID['PARAMS']['options']['zproj_type'] = '%s_warped_sum' % zproj
        
        # Update RID hashes:
        RID['PARAMS']['hashid'] = hashlib.sha1(json.dumps(RID['PARAMS'], sort_keys=True)).hexdigest()[0:6]
        
        # Update hash-included dirs
        old_hash = RID['rid_hash']
        old_dir = RID['DST']
        RID['rid_hash'] = hashlib.sha1(json.dumps(RID, sort_keys=True)).hexdigest()[0:6]
        RID['DST'] = RID['DST'].replace(old_hash, RID['rid_hash'])
        RID['SRC'] = warped_mean_image_path.replace(old_hash, RID['rid_hash'])
        shutil.move(old_dir, RID['DST']) # Rename ROI ID dir
        print "Updated ROI ID from %s\n --> to %s" % (old_dir, RID['DST'])
        
        rids[RID['roi_id']] = RID
        write_dict_to_json(rids, roidict_filepath)
        # -------------------------------------------------------------------------
    
    return warp_results



def extract_options(options):
    parser = optparse.OptionParser()

    parser.add_option('-D', '--root', action='store', dest='rootdir',
                          default='/nas/volume1/2photon/data',
                          help='data root dir (dir w/ all animalids) [default: /nas/volume1/2photon/data, /n/coxfs01/2pdata if --slurm]')
    parser.add_option('--slurm', action='store_true', dest='slurm', default=False, help="set if running as SLURM job on Odyssey")

    parser.add_option('-i', '--animalid', action='store', dest='animalid',
                          default='', help='Animal ID')

    # Set specific session/run for current animal:
    parser.add_option('-S', '--session', action='store', dest='session',
                          default='', help='session dir (format: YYYMMDD_ANIMALID')
    parser.add_option('-A', '--acq', action='store', dest='acquisition',
                          default='FOV1', help="acquisition folder (ex: 'FOV1_zoom3x') [default: FOV1]")


    #parser.add_option('-R', '--run', dest='run', default='', action='store', help="run name")
    #parser.add_option('-p', '--pid', dest='pid', default='processed001', action='store', help="PID for all runs (default: processed001)")
    parser.add_option('-r', '--rid', dest='rid', default='', action='store', help="ROI ID for all runs (default: '')")
    parser.add_option('-w', '--warp-thr', dest='warp_threshold', default=0.70, action='store', help='Threshold for aligned image correlation (default: 0.70)')
    parser.add_option('-e', '--enhance', dest='enhance_factor', default=2.0, action='store', help='Factor for enhancing grand mean img for ROI ref (default: 2.0)')
    
    parser.add_option('--new', action='store_true', dest='create_new', default=False, help="set flag if making warps anew")

    (options, args) = parser.parse_args(options)
    if options.slurm:
        options.rootdir = '/n/coxfs01/2p-data'
    
    return options

    
    
def get_roi_reference(options):
    optsE = extract_options(options)
    acquisition_dir = os.path.join(optsE.rootdir, optsE.animalid, optsE.session, optsE.acquisition)
    roi_id = optsE.rid
    warp_threshold = float(optsE.warp_threshold)
    enhance_factor = float(optsE.enhance_factor)
    
    warp_results = warp_runs_in_fov(acquisition_dir, roi_id, 
                                        warp_threshold=warp_threshold, 
                                        enhance_factor=enhance_factor,
                                        create_new=optsE.create_new)
    
    if len(warp_results['failed_warps']) > 0:
        print "----- WARNING ----- Unable to warp %i files." % (len(warp_results['failed_results']))
        print warp_results['failed_results']
    else:
        print "!! All successful warps !!"
        
    return warp_results['roi_reference_file'] 
        
#%%

def main(options):
    
    roi_reference_file = get_roi_reference(options)
    
    print "*******************************************************************"
    print "DONE!"
    print "Saved warped, summed STD image to:\n... %s" % roi_reference_file
    print "*******************************************************************"
    
    
#%%

if __name__ == '__main__':
    main(sys.argv[1:])
    
    
