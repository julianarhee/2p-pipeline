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


def transform_2p_fov(img, pixel_size, zoom_factor=1., normalize=True):
    '''
    First, left/right reflection and rotation of 2p image to match orientation of widefield view.
    Then, scale image to pixel size as measured by PSF.
    '''
    transf_ = orient_2p_to_macro(img, zoom_factor=zoom_factor, save=False, normalize=normalize)
    scaled_ = scale_2p_fov(transf_, pixel_size=pixel_size)
    return scaled_


def orient_2p_to_macro(avg, zoom_factor, normalize=True, 
                    acquisition_dir='/tmp', channel_ix=0, plot=False, save=True): #,
                        #xaxis_conversion=2.312, yaxis_conversion=1.904):
    '''
    Does standard Fiji steps:
        1. Scale slow-angle (if needed)
        2. Rotate image leftward, and flip L/R ('horizontally' in Fiji)
        3. Convert to 8-bit and adjust contrast
    '''
    # Scale:
    d1, d2 = avg.shape # (img height, img width)
    #print("Input img shape: (%i, %i)" % (d1, d2))
    #scaled = cv2.resize(avg, dsize=(d1, int(d2*zoom_factor)), interpolation=cv2.INTER_CUBIC)  #, dtype=avg.dtype)
    #new_d1 = d1*xaxis_conversion
    #new_d2 = d2*yaxis_conversion
    #scaled_pix = cv2.resize(avg, (new_d1, new_d2))
    
    # dsize: (v1, v2) -- v1 specifies COLS, v2 specifies ROWS (i.e., img_w, img_h)
    scaled = cv2.resize(avg, dsize=(int(d1*zoom_factor), d2), interpolation=cv2.INTER_CUBIC)  #, dtype=avg.dtype)
     
    # Rotate leftward:
    rotated = rotate_image(scaled, 90)
    
    # Flip L/R:
    transformed = np.fliplr(rotated)
    
    # Cut off super low vals, Convert range from 0, 255
    if normalize:
        transformed[transformed<-50] = 0 
        normed = cv2.normalize(transformed, None, 0, 255, cv2.NORM_MINMAX)
        
        # Convert to 8-bit
        img8 = cv2.convertScaleAbs(normed)
        
        # Equalize hist:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10,10))
        eq = clahe.apply(img8)
            
        if plot:
            pl.figure()
            pl.subplot(2,2,1); pl.title('orig'); pl.imshow(rotated); pl.colorbar();
            pl.subplot(2,2,2); pl.title('normalized'); pl.imshow(normed); pl.colorbar();
            pl.subplot(2,2,3); pl.title('8 bit'); pl.imshow(img8); pl.colorbar();
            pl.subplot(2,2,4); pl.title('equalize'); pl.imshow(eq); pl.colorbar();
            pl.savefig(os.path.join(acquisition_dir, 'anatomical', 'transform_steps_Ch%i.png' % int(channel_ix+1)))
            pl.close()
       
    else:
        eq = transformed.copy()

    if save:
        transformed_img_path = os.path.join(acquisition_dir, 'anatomical', 'anatomical_Channel%02d_transformed.tif' % int(channel_ix+1))
        tf.imsave(transformed_img_path, eq)
        return transformed_img_path
    else:
        if normalize:
            return img8 #eq #transformed_img_path
        else:
            return eq
    
 
def scale_2p_fov(transformed_image, pixel_size=(2.312, 1.888)):
    xaxis_conversion, yaxis_conversion = pixel_size 

    d1, d2 = transformed_image.shape # d1=HEIGHT, d2=WIDTH
    new_d1 = int(round(d1*xaxis_conversion,1)) # yaxis corresponds to M-L axis (now along )
    new_d2 = int(round(d2*yaxis_conversion,1)) # xaxis corresopnds to A-P axis (d2 is iamge width) 
    im_r = cv2.resize(transformed_image, (new_d2, new_d1))

    return im_r

    
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
        image_path = orient_2p_to_macro(avg, zoom_factor, 
                                    acquisition_dir=acquisition_dir, 
                                    channel_ix=channel_ix, plot=True)
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
    #print("warping...", im.min(), im.max())
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
    print("ref:", reference_mat.dtype)
    print("sample:", sample_mat.dtype)
    M = transformation_from_points(reference_mat, sample_mat)

    #Re-read sample image as grayscale for warping:
    out = warp_im(sample.astype(float), M, reference.shape)
    print('output:', out.min(), out.max())
    print('wrap matrix:', M.min(), M.max())

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
        print("...saving")
        out_fpath = os.path.join(self.coreg_dir, 'FOVs.pkl')
        f = open(out_fpath, 'wb')
        #with open(out_fpath, 'wb') as f:
        pkl.dump(self, f, protocol=pkl.HIGHEST_PROTOCOL)
        f.close()
    
    def get_reference(self, path_to_macro=None):
        if path_to_macro is None:
            try:
                macro_paths = glob.glob(os.path.join(self.rootdir, self.animalid, 'macro_maps', '20*', '*urf.png'))
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
               
        if path_to_macro.endswith('png'):
            reference = cv2.imread(path_to_macro, cv2.IMREAD_GRAYSCALE)
        else: 
            reference = tf.imread(path_to_macro)
        print "REF:",  reference.shape
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
                return -1
       
        # Get anatomical image if this is  a new or re-do FOV:
        self.session_list[curr_fov].get_transformed_image(create_new=optsE.create_new)
       
        return None
        
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
        print(aligned.min(), aligned.max())
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
        self.pixel_size = (2.312, 1.888) # um per pixel
        self.meta = {'nchannels': None, 'zoom_factor': None}
        self.coreg_dir = os.path.join(optsE.rootdir, optsE.animalid, 'coreg', '%s_%s' % (self.session, self.acquisition))
        if not os.path.exists(self.coreg_dir): os.makedirs(self.coreg_dir)
        
    def get_transformed_image(self, create_new=False): #1.904):
        acquisition_dir = os.path.join(self.rootdir, self.animalid, self.session, self.acquisition)
        
        # Get transformed, 8bit anatomical image to align to macro map:
        image_paths = sorted(glob.glob(os.path.join(acquisition_dir, 'anatomical', 'anatomical_Channel*_transformed.tif')), key=natural_keys)
        if len(image_paths) == 0:
            print "No anatomical transformed images found. Creating now."
            image_paths = self.transform_anatomicals()
       
        # check if scaled for pixels
        xaxis_conversion, yaxis_conversion = self.pixel_size
        scaled_paths = sorted(glob.glob(os.path.join(acquisition_dir, 'anatomical', 'anatomical_Channel*_transformed_scaled.tif')), key=natural_keys)
        if len(scaled_paths)==0:
            scaled_paths = self.scale_anatomicals(image_paths) 
                                                  #xaxis_conversion=xaxis_conversion,
                                                  #yaxis_conversion=yaxis_conversion)

        if len(scaled_paths) > 1:
            print "More than 1 channel img found:"
            ch1 = tf.imread([f for f in scaled_paths if 'Channel01' in f][0])
            ch2 = tf.imread([f for f in scaled_paths if 'Channel02' in f][0])
            fig = pl.figure(figsize=(10,4))
            pl.subplot(1,2,1); pl.imshow(ch1, cmap='gray'); pl.title('Channel01'); pl.axis('off')
            pl.subplot(1,2,2); pl.imshow(ch2, cmap='gray'); pl.title('Channel02'); pl.axis('off')
            pl.tight_layout()
            pl.show(block=False)
            
            for i, img_path in enumerate(scaled_paths):
                print i, img_path
            select = input("Choose IDX of channel to use: ")
            self.image_path = scaled_paths[select]
            pl.close(fig)
        else:
            self.image_path = scaled_paths[0]
            
        self.image = tf.imread(self.image_path)
        
        return

    def scale_anatomicals(self, image_paths): #, xaxis_conversion=2.312, yaxis_conversion=1.904):
        
        print("... scaling pixels")
        xaxis_conversion, yaxis_conversion = self.pixel_size 
        new_paths = []
        for impath in image_paths:
            img_outpath = '%s_scaled.tif' % (os.path.splitext(impath)[0])
            im = tf.imread(impath)
#            d1, d2 = im.shape # d1=HEIGHT, d2=WIDTH
#            new_d1 = int(round(d1*xaxis_conversion,1)) # yaxis corresponds to M-L axis (now along )
#            new_d2 = int(round(d2*yaxis_conversion,1)) # xaxis corresopnds to A-P axis (d2 is iamge width 
#            im_r = cv2.resize(im, (new_d2, new_d1))

            im_r = scale_2p_fov(im, pixel_size=self.pixel_size)
            tf.imsave(img_outpath, im_r)
            print(img_outpath)
            new_paths.append(img_outpath)

        return new_paths

    def transform_anatomicals(self):
        acquisition_dir = os.path.join(self.rootdir, self.animalid, self.session, self.acquisition)

        image_paths = []
        no_anatomical=False
        anatomical_fpath = glob.glob(os.path.join(acquisition_dir, 'anatomical', 'processed', 'processed*', 'mcorrected*', '*.tif'))
        if len(anatomical_fpath)==0:
            anatomical_fpath = [glob.glob(os.path.join(acquisition_dir, '*_run*', 'processed', 'processed*', 'mcorrected*', '*.tif'))[0]]
            no_anatomical=True

        assert len(anatomical_fpath) == 1, "More than 1 anatomical .tif found: %s" % str(anatomical_fpath)
        anatomical_fpath = anatomical_fpath[0]
        
        # Load corrected tif stack:
        img = tf.imread(anatomical_fpath)
        
        # Load SI meta data:
        if no_anatomical:
            si_fpath = glob.glob(os.path.join(acquisition_dir, '*_run*', 'raw*', 'SI*.json'))[0]
        else:
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

        # Craete anatomical outdir, if nec
        if no_anatomical:
            if not os.path.exists(os.path.join(acquisition_dir, 'anatomical')):
                os.makedirs(os.path.join(acquisition_dir, 'anatomical'))


        for channel_ix in range(nchannels):
            # Split channels
            channel_img = img[channel_ix::nchannels, :, :]
                
            # Get the group-averaged sum of single channel:
            avg = np.sum(np.dstack([np.mean(channel_img[i::nslices, :, :], axis=0) for i in range(nslices)]), axis=-1, dtype=channel_img.dtype)
            
            # Transform img:
            image_path = orient_2p_to_macro(avg, zoom_factor, 
                                            acquisition_dir=acquisition_dir, 
                                            channel_ix=channel_ix, plot=True)
            image_paths.append(image_path)
        
        self.meta['nchannels'] = nchannels
        self.meta['zoom_factor'] = zoom_factor
        
        return image_paths
    


    
#%%

def extract_options(options):

    parser = optparse.OptionParser()

    parser.add_option('-D', '--root', action='store', dest='rootdir',
                          default='/n/coxfs01/2p-data',
                          help='data root dir (dir w/ all animalids) [default: /nas/volume1/2photon/data, /n/coxfs01/2pdata if --slurm]')
    # parser.add_option('--slurm', action='store_true', dest='slurm', default=False, help="set if running as SLURM job on Odyssey")

    parser.add_option('-i', '--animalid', action='store', dest='animalid',
                          default='', help='Animal ID')

    # Set specific session/run for current animal:
    parser.add_option('-S', '--session', action='store', dest='session',
                          default='', help='session dir (format: YYYMMDD_ANIMALID')
    parser.add_option('-A', '--acq', action='store', dest='acquisition',
                          default='FOV1_zoom2p0x', help="acquisition folder (ex: 'FOV1_zoom3x') [default: FOV1]")
    parser.add_option('-M', '--macro-path', action='store', dest='macro_path',
                          default=None, help="Full path to image to use as reference (macro image) [default: /path/to/macro_maps/16bitSurf.tiff]")
    
    parser.add_option('--new', action='store_true', dest='create_new',
                          default=False, help="flag to remake fov images")
    
    (options, args) = parser.parse_args(options)
       
    return options


#%%
    
options = ['-D', '/mnt/odyssey', '-i', 'JC015', '-S', '20180913', '-A', 'FOV1_zoom2p7x']

#%%



    

def main(options):
    optsE = extract_options(options)
    animal_fpath = os.path.join(optsE.rootdir, optsE.animalid, 'coreg', 'FOVs.pkl')
    if not os.path.exists(animal_fpath):
        print "--- Creating NEW animal object! ---"
        A = Animal(optsE)
        A.save_me()
    else:
        with open(animal_fpath, 'rb') as f: A = pkl.load(f)
       
#        for fkey, currfv in A.session_list.items():
#            if not hasattr(currfv, 'pixel_size'):
#                fv_ = FOV(optsE) #setattr(xx, vv, 'test')
#                print('%s - adding px' % fkey)
#                for vv in dir(currfv):
#                    if _ in vv:
#                        continue
##                    if callable(getattr(currfv, vv)):
##                        continue
##                    setattr(fv_, vv, getattr(currfv, vv))
#                A.session_list[fkey] = fv_
#
    state = A.add_fov(optsE)
    if state is None:
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

