#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 18:19:52 2018

@author: juliana
"""

#!/usr/bin/env python2
import matplotlib
#matplotlib.use('agg')
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import math
import json
import os
import cv2
import h5py
import traceback
import hashlib
import datetime
import re
import copy
import optparse
import glob
import pprint
pp = pprint.PrettyPrinter(indent=4)

import pylab as pl
import cPickle as pkl
import numpy as np
import tifffile as tf

from pipeline.python.utils import natural_keys
from dateutil.parser import parse
import cv2
#%%
def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat


def transform2p_to_macro(avg, zoom_factor, acquisition_dir, channel_ix=0, plot=False):
    '''
    Does standard Fiji steps:
        1. Scale slow-angle (if needed)
        2. Rotate image leftward, and flip L/R ('horizontally' in Fiji)
        3. Convert to 8-bit and adjust contrast
    '''
    # Scale:
    d1, d2 = avg.shape
    scaled = cv2.resize(avg, dsize=(d1, int(d2*zoom_factor)), interpolation=cv2.INTER_CUBIC)  #, dtype=avg.dtype)
     
    # Rotate leftward:
    rotated = rotate_image(scaled, 90)
    
    # Flip L/R:
    transformed = np.fliplr(rotated)
    
    # Cut off super low vals, Convert range from 0, 255
    transformed[transformed<-50] = 0 
    normed = cv2.normalize(transformed, None, 0, 255, cv2.NORM_MINMAX)
    
    # Convert to 8-bit
    img8 = cv2.convertScaleAbs(normed)
    
    # Equalize hist:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10,10))
    eq = clahe.apply(img8)
    
    if plot:
        pl.figure()
        pl.subplot(2,2,1); pl.title('orig'); pl.imshow(rotated)
        pl.subplot(2,2,2); pl.title('normalized'); pl.imshow(normed)
        pl.subplot(2,2,3); pl.title('8 bit'); pl.imshow(img8)
        pl.subplot(2,2,4); pl.title('equalize'); pl.imshow(eq)
        pl.savefig(os.path.join(acquisition_dir, 'anatomical', 'transform_steps_Ch%i.png' % int(channel_ix+1)))
        pl.close()
        
    transformed_img_path = os.path.join(acquisition_dir, 'anatomical', 'anatomical_Channel%02d_transformed.tif' % int(channel_ix+1))
    tf.imsave(transformed_img_path, eq)
    
    return transformed_img_path
    

def transform_anatomicals(acquisition_dir):
    image_paths = []
    anatomical_fpath = glob.glob(os.path.join(acquisition_dir, 'anatomical', 'processed', 'processed*', 'mcorrected*', '*.tif'))
    assert len(anatomical_fpath) == 1, "More than 1 anatomical .tif found: %s" % str(anatomical_fpath)
    anatomical_fpath = anatomical_fpath[0]
    
    # Load corrected tif stack:
    img = tf.imread(anatomical_fpath)
    
    # Load SI meta data:
    si_fpath = glob.glob(os.path.join(acquisition_dir, 'anatomical', 'raw*', 'SI*.json'))[0]
    with open(si_fpath, 'r') as f: SI = json.load(f)
    SI = SI['File001']['SI']
    
    # Get volume dimensions:
    nchannels = len(SI['hChannels']['channelSave'])
    nvolumes = SI['hFastZ']['numVolumes']
    nslices = SI['hFastZ']['numFramesPerVolume'] - SI['hFastZ']['numDiscardFlybackFrames']
    
    # Determine zoom factor
    zoom_factor = SI['hRoiManager']['scanAngleMultiplierSlow']
    
    for channel_ix in range(nchannels):
        # Split channels
        channel_img = img[channel_ix::nchannels, :, :]
            
        # Get the group-averaged sum of single channel:
        avg = np.sum(np.dstack([np.mean(channel_img[i::nslices, :, :], axis=0) for i in range(nslices)]), axis=-1, dtype=channel_img.dtype)
        
        # Transform img:
        image_path = transform2p_to_macro(avg, zoom_factor, acquisition_dir, channel_ix=channel_ix, plot=True)
        image_paths.append(image_path)
        
    return image_paths


#%%
    
def transformation_from_points(points1, points2):
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)

    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = np.linalg.svd(points1.T * points2)
    R = (U * Vt).T

    return np.vstack([np.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         np.matrix([0., 0., 1.])])


def warp_im(im, M, dshape):
    output_im = np.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im

#%

def point_and_click(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping, refPt_pre, cropROI, image

    # Append clicked point:
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt.append((x, y))
        cv2.circle(image, refPt[-1], 1, (0,0,255), -1)
        cv2.imshow("image", image)
        #cv2.putText(image, '%i' % len(refPt), (refPt[-1][0]-5, refPt[-1][1]+5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

def get_registration_points(sample):

    #image = copy.copy(sample)
    #refPt = []
    #cropping = False
    global refPt, image

    clone = image.copy()

    cv2.startWindowThread()
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("image", point_and_click)

    # keep looping until the 'q' key is pressed
    while True:
        # display the image and wait for a keypress
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF

        # if the 'r' key is pressed, reset the cropping region
        if key == ord("r"):
            image = clone.copy()
            refPt = []

        # if the 'c' key is pressed, break from the loop
        elif key == ord("c"):
            break

    # close all open windows
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    return image, refPt

#%%
def align_to_reference(sample, reference, outdir, sample_name='sample'):

    global refPt, image

    # GET SAMPLE:
    print "Sample that will be aligned to ref is: ", sample.shape

    refPt = []
    image = copy.copy(sample)
    sample_pts_img, sample_pts = get_registration_points(image)
    #sample_pts = copy.copy(refPt)
    npoints = len(sample_pts)
    print "GOT %i SAMPLE POINTS: " % npoints
    print sample_pts

    # Save chosen SAMPLE points:
    sample_points_path = os.path.join(outdir, 'sample_points_%s.png' % sample_name)
    cv2.imwrite(sample_points_path, sample_pts_img)
    print "Saved SAMPLE points to:\n%s" % sample_points_path


    #% GET corresponding reference points:

    refPt = []
    image = copy.copy(reference)
    ref_pts_img, reference_pts = get_registration_points(reference)
    npoints = len(reference_pts)

    # DISPLAY REF IMAGE:
    print "GOT %i reference test POINTS: " % npoints
    print reference_pts
    # Save chosen REF points:
    ref_points_path = os.path.join(outdir, 'reference_points_%s.png' % sample_name)
    cv2.imwrite(ref_points_path, ref_pts_img)
    print "Saved REFERENCE points to:\n%s" % ref_points_path
    cv2.destroyAllWindows()

    #%
    # Use SAMPLTE and TEST points to align:
    sample_mat = np.matrix([i for i in sample_pts])
    reference_mat = np.matrix([i for i in reference_pts])
    M = transformation_from_points(reference_mat, sample_mat)

    #Re-read sample image as grayscale for warping:
    out = warp_im(sample, M, reference.shape)

    coreg_info = dict()
    coreg_info['reference_points_x'] = tuple(p[0] for p in reference_pts)
    coreg_info['reference_points_y'] = tuple(p[1] for p in reference_pts)
    coreg_info['sample_points_x'] = tuple(p[0] for p in sample_pts)
    coreg_info['sample_points_y'] = tuple(p[1] for p in sample_pts)
    #coreg_info['transform_mat'] = M

    coreg_hash = hash(frozenset(coreg_info.items()))
    print "COREG HASH: %s" % coreg_hash

    return M, out, coreg_info, coreg_hash

#%%

def plot_transforms(sampleimg, referenceimg, out, npoints, out_path='/tmp'):

    print "Making figure..."
    # plt.figure(figsize=(10,0))
    pl.figure()

    pl.subplot(221)
    pl.imshow(sampleimg, cmap='gray')
    pl.axis('off')
    pl.title('original sample')

    pl.subplot(222)
    pl.imshow(referenceimg, cmap='gray')
    pl.axis('off')
    pl.title('original reference')

    pl.subplot(223)
    pl.imshow(out, cmap='gray')
    pl.axis('off')
    pl.title('warped sample')

    pl.subplot(224)
    merged = np.zeros((referenceimg.shape[0], referenceimg.shape[1], 3), dtype=np.uint8)
    merged[:,:,0] = referenceimg #cv2.cvtColor(reference)#, cv2.COLOR_RGB2GRAY)
    merged[:,:,1] = out #cv2.cvtColor(outi) #, cv2.COLOR_RGB2GRAY)
    pl.imshow(merged)
    pl.axis('off')
    pl.title('combined')
    pl.tight_layout()
    
    imname = 'warp_transforms_npoints%i.png' % (npoints)
    print imname
    pl.savefig(os.path.join(out_path, imname))
    #pl.show()

#%
def plot_merged(reference, out, npoints=None, out_path='/tmp'):
    print "Getting MERGED figure..."

    pl.figure()
    merged = np.zeros((reference.shape[0], reference.shape[1], 3), dtype=np.uint8)
    merged[:,:,0] = reference
    merged[:,:,1] = out
    pl.imshow(merged)
    pl.axis('off')

    if npoints is None:
        imname = 'overlay_ALL_FOVs'
    else:
        imname = 'overlay_npoints%i' % npoints #(sample_fn, reference_fn, npoints)

    print os.path.join(out_path, imname)
    pl.savefig(os.path.join(out_path, imname))

    #pl.show()
    
#%%

class Animal():
    def __init__(self, optsE):
        self.rootdir = optsE.rootdir
        self.animalid = optsE.animalid
        self.session_list = {}

        self.coreg_dir = os.path.join(self.rootdir, self.animalid, 'coreg')
        if not os.path.exists(self.coreg_dir):
            os.makedirs(self.coreg_dir)
        print "Saving coreg results to: %s" % self.coreg_dir
                
        self.get_reference(path_to_macro=optsE.macro_path)
        
    def save_me(self):
        out_fpath = os.path.join(self.coreg_dir, 'FOVs.pkl')
        f = open(out_fpath, 'wb')
        #with open(out_fpath, 'wb') as f:
        pkl.dump(self, f, protocol=pkl.HIGHEST_PROTOCOL)
        f.close()
    
    def get_reference(self, path_to_macro=None):
        if path_to_macro is None:
            try:
                macro_paths = glob.glob(os.path.join(self.rootdir, self.animalid, 'macro_maps', '16bitSurf.tif'))
                assert len(macro_paths) == 1, "Not sure which image to use as reference..."
                path_to_macro = macro_paths[0]
            except Exception as e:
                macro_paths = glob.glob(os.path.join(self.rootdir, self.animalid, 'macro_maps', '20*', '*.tiff'))
                assert len(macro_paths) > 0, "Nothing in macro dir: %s. ABORING." % macro_paths
                
                print "Found these images..."
                for i, mpath in enumerate(macro_paths):
                    print i, mpath
                select = input("Select IDX: ")
                path_to_macro = macro_paths[select]
                
        reference = tf.imread(path_to_macro)
        self.reference = reference
        self.reference_fpath = path_to_macro
        
    
    def add_fov(self, optsE):
        curr_fov = '%s_%s' % (optsE.session, optsE.acquisition)
        if curr_fov not in self.session_list.keys():
            print "... Adding new FOV -- %s -- to list." % curr_fov
            self.session_list.update({curr_fov: FOV(optsE)})
        else:
            redo = raw_input("... FOV -- %s -- exists. Re-do alignment?\nPress [Y] to redo, Enter to escape: " % curr_fov)
            if redo == 'Y':
                print "Re-adding FOV to list."
                self.session_list.update({curr_fov: FOV(optsE)})
            else:
                return
        
        # Get anatomical image if this is  a new or re-do FOV:
        self.session_list[curr_fov].get_transformed_image()
        
    def align_fov(self, curr_fov):
        
        print "... ALIGNING:  current FOV is %s" % curr_fov
        M, out, coreg_info, coreg_hash = align_to_reference(self.session_list[curr_fov].image, self.reference, self.session_list[curr_fov].coreg_dir, sample_name=curr_fov)
        #npoints = len(coreg_info['sample_points_x'])

        # Plot figures:
        #plot_transforms(self.session_list[curr_fov].image, self.reference, out, npoints, out_path=self.session_list[curr_fov].coreg_dir)
        #plot_merged(self.reference, out, npoints, out_path=self.session_list[curr_fov].coreg_dir)
        
        self.session_list[curr_fov].alignment = {'aligned': out,
                                                 'transform_matrix': M,
                                                 'coreg_points': coreg_info}
        
    def plot_alignment(self, curr_fov):
        
        aligned = self.session_list[curr_fov].alignment['aligned']
        a = np.ma.masked_where(aligned==0, aligned)

        pl.figure()
        pl.imshow(self.reference, cmap='gray')
        pl.imshow(a, alpha=0.5)
        pl.axis('off')
        pl.title(curr_fov)
        
        pl.savefig(os.path.join(self.session_list[curr_fov].coreg_dir, 'overlay_npoints%i.png' % len(self.session_list[curr_fov].alignment['coreg_points']['sample_points_x'])))
        pl.show(block=False)
        
    
    def check_alignment(self, curr_fov):
        reselect = raw_input("Keep current alignment for -- %s --? Press [A] to accept, [R] to reselect points: " % curr_fov)
        if reselect == 'R':
            #pl.close(fig)
            reselect_points = True
        else:
            #pl.savefig(os.path.join(self.session_list[curr_fov].coreg_dir, 'overlay_npoints%i.png' % len(self.session_list[curr_fov].alignment['coreg_points']['sample_points_x'])))
            #pl.close(fig)
            reselect_points = False
            
        return reselect_points  
    
        
        
        
#%%
class FOV():
    
    def __init__(self, optsE):
        self.rootdir = optsE.rootdir
        self.animalid = optsE.animalid
        self.session = optsE.session
        self.acquisition = optsE.acquisition
        self.image_fpath = None
        self.image = None
        self.meta = {'nchannels': None, 'zoom_factor': None}
        self.coreg_dir = os.path.join(optsE.rootdir, optsE.animalid, 'coreg', '%s_%s' % (self.session, self.acquisition))
        if not os.path.exists(self.coreg_dir): os.makedirs(self.coreg_dir)
        
    def get_transformed_image(self):
        acquisition_dir = os.path.join(self.rootdir, self.animalid, self.session, self.acquisition)
        
        # Get transformed, 8bit anatomical image to align to macro map:
        image_paths = sorted(glob.glob(os.path.join(acquisition_dir, 'anatomical', 'anatomical_Channel*_transformed.tif')), key=natural_keys)
        if len(image_paths) == 0:
            print "No anatomical transformed images found. Creating now."
            image_paths = self.transform_anatomicals()
        
        if len(image_paths) > 1:
            print "More than 1 channel img found:"
            ch1 = tf.imread([f for f in image_paths if 'Channel01' in f][0])
            ch2 = tf.imread([f for f in image_paths if 'Channel02' in f][0])
            fig = pl.figure(figsize=(10,4))
            pl.subplot(1,2,1); pl.imshow(ch1, cmap='gray'); pl.title('Channel01'); pl.axis('off')
            pl.subplot(1,2,2); pl.imshow(ch2, cmap='gray'); pl.title('Channel02'); pl.axis('off')
            pl.tight_layout()
            pl.show(block=False)
            
            for i, img_path in enumerate(image_paths):
                print i, img_path
            select = input("Choose IDX of channel to use: ")
            self.image_path = image_paths[select]
            pl.close(fig)
        else:
            self.image_path = image_paths[0]
            
        self.image = tf.imread(self.image_path)
        
        
    def transform_anatomicals(self):
        acquisition_dir = os.path.join(self.rootdir, self.animalid, self.session, self.acquisition)

        image_paths = []
        anatomical_fpath = glob.glob(os.path.join(acquisition_dir, 'anatomical', 'processed', 'processed*', 'mcorrected*', '*.tif'))
        assert len(anatomical_fpath) == 1, "More than 1 anatomical .tif found: %s" % str(anatomical_fpath)
        anatomical_fpath = anatomical_fpath[0]
        
        # Load corrected tif stack:
        img = tf.imread(anatomical_fpath)
        
        # Load SI meta data:
        si_fpath = glob.glob(os.path.join(acquisition_dir, 'anatomical', 'raw*', 'SI*.json'))[0]
        with open(si_fpath, 'r') as f: SI = json.load(f)
        SI = SI['File001']['SI']
        
        # Get volume dimensions:
        if isinstance(SI['hChannels']['channelSave'], int):
            nchannels = 1
        else:
            nchannels = len(SI['hChannels']['channelSave'])
            
        nvolumes = SI['hFastZ']['numVolumes']
        nslices = SI['hFastZ']['numFramesPerVolume'] - SI['hFastZ']['numDiscardFlybackFrames']
        
        # Determine zoom factor
        zoom_factor = SI['hRoiManager']['scanAngleMultiplierSlow']
        
        for channel_ix in range(nchannels):
            # Split channels
            channel_img = img[channel_ix::nchannels, :, :]
                
            # Get the group-averaged sum of single channel:
            avg = np.sum(np.dstack([np.mean(channel_img[i::nslices, :, :], axis=0) for i in range(nslices)]), axis=-1, dtype=channel_img.dtype)
            
            # Transform img:
            image_path = transform2p_to_macro(avg, zoom_factor, acquisition_dir, channel_ix=channel_ix, plot=True)
            image_paths.append(image_path)
        
        self.meta['nchannels'] = nchannels
        self.meta['zoom_factor'] = zoom_factor
        
        return image_paths
    


    
#%%

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
    parser.add_option('-M', '--macro-path', action='store', dest='macro_path',
                          default=None, help="Full path to image to use as reference (macro image) [default: /path/to/macro_maps/16bitSurf.tiff]")
    
    
    (options, args) = parser.parse_args(options)
    if options.slurm is True and '/n/coxfs01' not in options.rootdir:
        options.rootdir = '/n/coxfs01/2p-data'
        
    return options


#%%
    
options = ['-D', '/mnt/odyssey', '-i', 'JC015', '-S', '20180913', '-A', 'FOV1_zoom2p7x']

#%%



    

def main(options):
    optsE = extract_options(options)
    animal_fpath = os.path.join(optsE.rootdir, optsE.animalid, 'FOVs.pkl')
    if not os.path.exists(animal_fpath):
        print "--- Creating NEW animal object! ---"
        A = Animal(optsE)
        A.save_me()
    else:
        with open(animal_fpath, 'rb') as f: A = pkl.load(f)
        
        
    A.add_fov(optsE)
    curr_fov = '%s_%s' % (optsE.session, optsE.acquisition)
    A.align_fov(curr_fov)
    A.plot_alignment(curr_fov)
    reselect_points = A.check_alignment(curr_fov)
    while reselect_points:
        #pl.close(fig)
        A.align_fov(curr_fov)
        A.plot_alignment(curr_fov)
        reselect_points = A.check_alignment(curr_fov)

    A.save_me()
    print A.session_list


if __name__ == '__main__':
    main(sys.argv[1:])
    
    


#%%
def get_file_hash(file_path):
    hasher = hashlib.md5()
    with open(file_path, 'r') as f:
        buf = f.read()
        hasher.update(buf)
        filehash = hasher.hexdigest()
    return filehash


#%%
def update_alignments(alignments_filepath, FOV):
    if os.path.exists(alignments_filepath):
        alignments = h5py.File(alignments_filepath, 'a')
        new_file = False
    else:
        alignments = h5py.File(alignments_filepath, 'w')
        new_file = True

    try:
        # Save reference to ALIGNMENT file, if doesn't exist:
        if new_file is True:
            sources = alignments.create_group('source_files')
        else:
            sources = alignments['source_files']

        sessions = FOV.keys()
        for session in sessions:
            for acquisition in FOV[session].keys():
                fov_key = '%s_%s' % (session, acquisition)
                img, imghash = load_image(FOV[session][acquisition]['filepath'])
                if img is None:
                    print "UMMM no image found: %s" % FOV[session][acquisition]['filepath']
                    continue
                if fov_key not in sources.keys():
                    dset = sources.create_dataset('%s' % fov_key, img.shape, img.dtype)
                    dset[...] = img
                    dset.attrs['filepath'] = FOV[session][acquisition]['filepath']
                    dset.attrs['filehash'] = FOV[session][acquisition]['filehash']
                elif fov_key in sources.keys() and not sources[fov_key].attrs['filehash'] == FOV[session][acquisition]['filehash']:
                    print "For session %s, acq %s -- different img hashes!"
                    pl.figure()
                    pl.subplot(1,2,1); pl.title('stored file'); pl.imshow(sources[fov_key], cmap='gray')
                    pl.subplot(1,2,2); pl.title('requested file'); pl.imshow(img, cmap='gray')
                    pl.show(block=False)
                    while True:
                        user_select = raw_input("Select O to overwrite with new img (requested file), or C to create new: ")
                        if user_select == 'N':
                            dset = sources[fov_key]
                            dset.attrs['filepath'] = FOV[session][acquisition]['filepath']
                            dset.attrs['filehash'] = FOV[session][acquisition]['filehash']
                            break
                        elif user_select == 'O':
                            nimgs = len([i for i in sources.keys() if fov_key in i]) + 1
                            dset = sources.create_dataset('%s_%i' % (fov_key, nimgs), img.shape, img.dtype)
                            dset[...] = img
                            dset.attrs['filepath'] = FOV[session][acquisition]['filepath']
                            dset.attrs['filehash'] = FOV[session][acquisition]['filehash']
                            break
                    pl.close()
    except Exception as e:
        print "***ERROR: Unable to update FOVs for session %s, acq %s." % (session, acquisition)
        print "FOV KEY: %s" % fov_key
        traceback.print_exc()
    finally:
        alignments.close()

    print "UPDATE COMPLETE."


#%%
def load_image(image_path):
    imghash = get_file_hash(image_path)
    image = cv2.imread(image_path)
    # Make sure images are gray-scale:
    if image is None:
        return None, None
    if len(image.shape)==2: # not RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) # make it 3D
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image[:,:,1]
    print "Image size is: ", image.shape

    return image, imghash
#%%
def is_date(string):
    try:
        parse(string)
        return True
    except ValueError:
        return False
#%%
def get_sample_paths(animal_dir, verbose=False):
    '''For each SESSION for current animal, return dict:

        FOV[SESSION][ACQUISITION] = '/path/to/transformed/corrected/BV/image.tif'

        NOTE:  image should be 8-bit, corrected, summed, and transformed to match
        reference FOV (for 12k-2p, this is rotated-left, flipped horizontally in Fiji).
    '''
    # Find session list, for each ACQUISITION (i.e., FOV), find the corrected, transformed, 8-bit image
    sessions = [s for s in os.listdir(animal_dir) if os.path.isdir(os.path.join(animal_dir, s)) and is_date(s)]
    if verbose is True:
        print "SESSIONS:"
        print sessions

    #non_acquisitions = ['coregistration', 'macro_fullfov', 'ROIs']
    FOV = {}
    for session in sessions:
        session_dir = os.path.join(animal_dir, session)
        acquisition_list = [a for a in os.listdir(session_dir) if os.path.isdir(os.path.join(session_dir, a)) and 'anatomical' in os.listdir(os.path.join(session_dir, a))]
        if verbose is True:
            print 'SESSION: %s -- Found %i acquisitions.' % (session, len(acquisition_list))

        if len(acquisition_list) > 0:
            FOV[session] = dict() #dict((acquisition, dict()) for acquisition in acquisition_list)
        #sample_paths = dict()
        for acquisition in acquisition_list:
            if verbose is True:
                print "-- ACQ: %s" % acquisition
            curr_anatomical_filepath = None
            bv_images = [f for f in os.listdir(os.path.join(session_dir, acquisition, 'anatomical')) if f.endswith('tif')]
            if len(bv_images) == 1:
                fn = bv_images[0]
                curr_anatomical_filepath = os.path.join(session_dir, acquisition, 'anatomical', fn)
            elif len(bv_images) > 1:
                print "Found multiple anatomicals for Acq. %s, session %s." % (acquisition, session)
                for idx,imgfn in enumerate(bv_images):
                    print idx, imgfn
                while True:
                    user_selection = input('Select IDX of image file to use: ')
                    fn = bv_images[int(user_selection)]
                    confirmation = raw_input('Use file: %s?  Press Y to confirm.' % fn)
                    if confirmation == 'Y':
                        break
                curr_anatomical_filepath = os.path.join(session_dir, acquisition, 'anatomical', fn)
            else:
                print "**WARNING** No anatomical image found in session %s, for acquisition %s." % (session, acquisition)
                print "Create processed blood vessel image, transform to MACRO fov, and save to dir:\n%s" % os.path.join(session_dir, acquisition, 'anatomical')

            if curr_anatomical_filepath is not None:
                if acquisition not in FOV[session].keys():
                    FOV[session][acquisition] = dict()
                FOV[session][acquisition]['filepath'] = curr_anatomical_filepath
                FOV[session][acquisition]['filehash'] = get_file_hash(curr_anatomical_filepath)
            #sample_paths[acquisition] = curr_anatomical_filepath
    return FOV



#%%
#
#parser = optparse.OptionParser()
#parser.add_option('-D', '--root', action='store', dest='rootdir', default='/nas/volume1/2photon/data', help='data root dir (root project dir containing all animalids) [default: /nas/volume1/2photon/data, /n/coxfs01/2pdata if --slurm]')
#parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')
#parser.add_option('--new', action='store_true', dest='align_new', default=False, help='Flag if there is a new FOV to align')
#parser.add_option('--merge', action='store_true', dest='merge_all', default=False, help="Flag to create merged image of ALL found FOVs for current animal")
##parser.add_option('-R', '--run', action='store', dest='curr_run', default='', help="custom run name [e.g., 'barflash']")
#
##parser.add_option('-r', '--reference', action="store", dest="reference",
##                  default="", help="Path to reference image (to align to)")
##parser.add_option('-s', '--sample', action="store", dest="sample",
##                  default="", help="Path to sample image (to align to the reference")
##parser.add_option('-o', '--outpath', action="store", dest="outpath",
##                  default="/tmp", help="Path to the save ROIs")
##parser.add_option('--C', '--crop', action="store_true", dest="crop", default=False, help="Path to save ROI")
#
#(options, args) = parser.parse_args()


#%%
#
#
#rootdir = options.rootdir
#animalid = options.animalid
#
#align_new = options.align_new
#merge_all = options.merge_all

#%%

## First check animal dir to see if coregistration info already exists:
#animal_dir = os.path.join(rootdir, animalid)
#coreg_dir = os.path.join(animal_dir, 'coregistration')
#if not os.path.exists(coreg_dir):
#    os.makedirs(coreg_dir)
#
#alignments_filepath = os.path.join(coreg_dir, 'alignments.hdf5')
#if os.path.exists(alignments_filepath):
#    alignments = h5py.File(alignments_filepath, 'a')
#    new_file = False
#else:
#    alignments = h5py.File(alignments_filepath, 'w')
#    new_file = True
#
## Get reference image -- look in "macro_fov" dir:
#reference_path = os.path.join(coreg_dir, 'REFERENCE.tif')
#try:
#    reference, refhash = load_image(reference_path)
#
#    # Save reference to ALIGNMENT file, if doesn't exist:
#    if new_file is True:
#        sources = alignments.create_group('source_files')
#    else:
#        sources = alignments['source_files']
#    if 'reference' not in sources.keys():
#        ref = sources.create_dataset('reference', reference.shape, reference.dtype)
#        ref[...] = reference
#        ref.attrs['filepath'] = reference_path
#        ref.attrs['filehash'] = refhash
#    else:
#        ref = sources['reference']
#
#    # Get list of paths to anatomical img for each acquisition:
#    FOV = get_sample_paths(animal_dir, verbose=False)
#    print "Coregistering %i acquisitions." % len(FOV.keys())
#    pp.pprint(FOV)
#
#    update_alignments(alignments_filepath, FOV)
#
#except Exception as e:
#    if not os.path.exists(reference_path):
#        print "***ERROR: Unable to find REFERENCE.tif"
#        print "Save 8-bit gray-scale image 'REFERENCE.tif' to:\n%s" % reference_path
#    traceback.print_exc()
#    print "Aborting."
#    print "-------------------------------------------"
#finally:
#    alignments.close()


#%%


#%%
#coreg_info = dict()
#coreg_info['reference_file'] = reference_path
#coreg_info['sample_file'] = sample_path
#coreg_info['reference_points_x'] = tuple(p[0] for p in reference_pts)
#coreg_info['reference_points_y'] = tuple(p[1] for p in reference_pts)
#coreg_info['sample_points_x'] = tuple(p[0] for p in sample_pts)
#coreg_info['sample_points_y'] = tuple(p[1] for p in sample_pts)
##coreg_info['transform_mat'] = M
#
#coreg_hash = hash(frozenset(coreg_info.items()))
#print "COREG HASH: %s" % coreg_hash

#
#if align_new == True:
#    alignments = h5py.File(alignments_filepath, 'r')
#
#
#    fov_keys = [k for k in alignments['source_files'].keys() if not k=='reference']
#    outdir = os.path.join(coreg_dir, 'results')
#    if not os.path.exists(outdir):
#        os.makedirs(outdir)
#
#    for fov in fov_keys:
#        curr_alignment_path = os.path.join(outdir, 'align_%s.hdf5' % fov)
#        if os.path.exists(curr_alignment_path):
#            results = h5py.File(curr_alignment_path, 'a')
#        else:
#            results = h5py.File(curr_alignment_path, 'w')
#            make_new = True
#
#        # Add coregistration info and results:
#        if 'transforms' not in results.keys():
#            transforms = results.create_group('transforms')
#            make_new = True
#        else:
#            transforms = results['transforms']
#            existing_registers = sorted(transforms.keys(), key=natural_keys)
#            if len(existing_registers) > 0:
#                while True:
#                    print "Found existing transforms:"
#                    for i, trans in enumerate(sorted(existing_registers, key=natural_keys)):
#                        print i, trans
#                    user_choice = raw_input('Create new alignment for fov: %s?\nPress <Y> to create new, or <n> to continue: ' % fov)
#                    if user_choice == 'Y':
#                        make_new = True
#                        break
#                    elif user_choice == 'n':
#                        make_new = False
#                        break
#            else:
#                make_new = True
#        try:
#            if make_new is True:
#                if 'reference' not in results.keys():
#                    refimg = alignments['source_files']['reference'][:]
#                    reference = results.create_dataset('reference', refimg.shape, refimg.dtype)
#                    reference[...] = refimg
#                else:
#                    reference = results['reference']
#                if 'sample' not in results.keys():
#                    fovimg = alignments['source_files'][fov][:]
#                    sample = results.create_dataset('sample', fovimg.shape, fovimg.dtype)
#                    sample[...] = fovimg
#                else:
#                    sample = results['sample']
#
#                print "Current FOV: %s" % fov
#                M, out, coreg_info, coreg_hash = align_to_reference(sample[:], reference[:], outdir, sample_name=fov)
#                npoints = len(coreg_info['sample_points_x'])
#
#                if coreg_hash not in transforms.keys():
#                    tstamp = datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")
#                    transf_key = "%s_%s" % (fov, tstamp)
#                    match = transforms.create_dataset(transf_key, M.shape, M.dtype)
#                    match[...] = M
#                    for info in coreg_info.keys():
#                        match.attrs[info] = coreg_info[info]
#                    match.attrs['coreg_hash'] = coreg_hash
#
#                # Plot figures:
#                curr_outdir = os.path.join(outdir, 'figures_%s' % fov)
#                if not os.path.exists(curr_outdir):
#                    os.makedirs(curr_outdir)
#                plot_transforms(sample[:], reference[:], out, npoints, out_path=curr_outdir)
#                plot_merged(reference[:], out, npoints, out_path=curr_outdir)
#            else:
#                continue
#
#        except Exception as e:
#            print "***ERROR in aligning sample to reference."
#            print "--- SAMPLE: %s" % fov
#            traceback.print_exc()
#            print "----------------------------------------"
#        finally:
#            results.close()

#%%
#
#if merge_all is True:
#    alignments = h5py.File(alignments_filepath, 'r')
#    reference = alignments['source_files']['reference']
#    fov_files = [os.path.join(coreg_dir, 'results', f) for f in os.listdir(os.path.join(coreg_dir, 'results')) if f.endswith('hdf5')]
#    for fov_fn in fov_files:
#        results = h5py.File(fov_fn, 'a')
#        try:
#            transforms = results['transforms']
#            sample = results['sample'][:]
#            if 'warps' not in results.keys():
#                warps = results.create_group('warps')
#            else:
#                warps = results['warps']
#            transf_keys = [k for k in transforms.keys() if len(transforms[k].attrs['sample_points_x']) > 1]
#            for key in transf_keys:
#                if key not in warps.keys():
#                    warpim = warp_im(sample[:], transforms[key][:], reference[:].shape)
#                    dset = warps.create_dataset(key, warpim.shape, warpim.dtype)
#                    dset[...] = warpim
#        except Exception as e:
#            print "***ERROR warping transform."
#            print "--- FOV: %s" % fov_fn
#            traceback.print_exc()
#        finally:
#            results.close()
#
#    overlay = np.zeros(reference.shape, reference.dtype)
#    min_vals = []
#    max_vals = []
#    for fov_fn in fov_files:
#        results = h5py.File(fov_fn, 'r')
#
#        try:
#            if len(results['warps'].keys()) > 1:
#                warp_keys = sorted(results['warps'].keys(), key=natural_keys) # For now, just take most recent...
#                print "Found %i transforms for %s" % (len(warp_keys), fov_fn)
#                print "Taking the most recent one..."
#                warp_key = warp_keys[-1]
#            else:
#                warp_key = results['warps'].keys()[0]
#
#            curr_warp = results['warps'][warp_key][:]
#            #pl.figure(); pl.imshow(curr_warp); pl.colorbar()
#            min_vals.append(curr_warp.min())
#            max_vals.append(curr_warp.max())
#            overlay += curr_warp
#        except Exception as e:
#            print "---- Error combining FOV to reference."
#            print "---- Curr file: %s" % fov_fn
#            traceback.print_exc()
#        finally:
#            results.close()
#
#
#    plot_merged(reference, overlay, npoints=None, out_path=coreg_dir)
#