#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 01 17:51:27 2020

@author: julianarhee
"""
import matplotlib as mpl
mpl.use('agg')
import os
import cv2
import sys
import optparse

import numpy as np
import pylab as pl
import tifffile as tf
import cPickle as pkl

from pipeline.python.coregistration.align_fov import Animal, FOV, warp_im, transform_2p_fov
from pipeline.python.rois import utils as roi_util
from pipeline.python import utils as putils

import numpy.ma as ma
import matplotlib.gridspec as gridspec


'''
Saves:
    fov2p_transformed : 2p fov rotated, flipped, and pixel-scaled.
    fov2p_warped : 2p fov warped to align to widefield vasculature image with warp_mat.
    warp_mat : Transformation matrix to align 2p to widefield (wrap_im()).
    vasculature : Surface image (common to retino + 2p fov coreg.)    
'''

#
class struct():
    pass


def adjust_image_contrast(img, clip=2.0, tile=5):
    '''Adjust grayscale fov image to make cells more visiible
    '''
    img_int8 = img.astype(np.uint8)
    clh = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile,tile))
    img_eq = clh.apply(img_int8)
    return img_eq

def plot_roi_overlay(roi_img, roi_zproj, ax=None, cmap='jet', vmin=None, vmax=None):
    '''Combine one image with another as overlay
    '''
    
    if vmin is None or vmax is None:
        vmin, vmax = (roi_img.min(), roi_img.max())
        
    roi_img_overlay = np.ma.masked_where(roi_img == 0, roi_img)
    
    if ax is None:
        fig, ax = pl.subplots()
        
    ax.imshow(roi_zproj, cmap='gray')
    ax.imshow(roi_img_overlay, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis('off')
    
    return 

def assign_int_to_masks(roi_masks):
    '''
    Assign roi index to each mask for unique color identification (visualization).
    '''
    d1, d2, nrois = roi_masks.shape

    int_rois = np.dstack([roi_masks[:, :, r].astype(bool).astype(int)*(r+1) for r in np.arange(0, nrois)])
    
    int_rois_sum = np.zeros((d1, d2))
    for ri in np.arange(0, nrois):
        curr_msk = int_rois[:, :, ri].copy()
        int_rois_sum[curr_msk>0] = ri+1
        
    #int_rois_sum = int_rois.sum(axis=-1)
    int_roi_overlay = int_rois_sum.copy().astype(float)
    int_roi_overlay[int_rois_sum==0] = np.nan
    #np.ma.masked_array(int_rois_sum==0, int_rois_sum)
    print(nrois, int_rois_sum.min(), int_rois_sum.max())
    
    return int_roi_overlay


# COREGISTRATION INFO.
def get_coregistration_results(animalid, session, rootdir='/n/coxfs01/2p-data', verbose=False):

    coreg_dfile = os.path.join(rootdir, animalid, 'coreg', 'FOVs.pkl')
    vasculature, fov_ = load_coregistration(coreg_dfile, session, verbose=verbose)

    assert fov_ is not None, "[%s|%s]: error loading coreg results: %s" % (animalid, session, coreg_dfile)
    coreg_fov2p_transformed = fov_.image.copy() # transformed + pixel-scaled image (pre-alignment)
    coreg_fov2p_warped = fov_.alignment['aligned'] # aligned 2p fov 

    transform_mat = fov_.alignment['transform_matrix'].copy()

    coreg_d = {'BV_transformed': coreg_fov2p_transformed,
               'BV_warped': coreg_fov2p_warped,
               'warp_mat': transform_mat,
               'vasculature': vasculature,
               'pixel_size': fov_.pixel_size, 
               'source': coreg_dfile,
               'fov_key': '%s_%s' % (fov_.session, fov_.acquisition)}
    
    return coreg_d

def load_coregistration(coreg_dfile, session, verbose=False):

    with open(coreg_dfile, 'rb') as f:
        A = pkl.load(f)
         
    vasculature = A.reference.copy()  # surface img for WF retino map (+ 2p fov alignment)
    print("Found %i fovs." % len(A.session_list.keys()))

    if verbose:
        for f in A.session_list.keys():
            print(f)

    fov_ = None
    try:
        fov_key = [f for f in A.session_list.keys() if session in f][0]
        #fov_name = fov_key.split('%s_' % session)[-1]
        fov_ = A.session_list[fov_key]
    except Exception as e:
        print("Unable to find session %s in datafile: %s" % (session, coreg_dfile))


    return vasculature, fov_



def plot_coregistration_overlay(vasculature, coreg_fov2p_warped, ax=None):
    if ax is None:
        fig, ax = pl.subplots()
    fov2p_aligned_overlay = np.ma.masked_where(coreg_fov2p_warped==0, coreg_fov2p_warped)
    ax.imshow(vasculature, cmap='gray')
    ax.imshow( fov2p_aligned_overlay, cmap='jet', alpha=0.3)
    return ax

def load_and_plot_coregistration(animalid, session, verbose=False,
                                 rootdir='/n/coxfs01/2p-data', outdir='/tmp'):
    coreg_d = get_coregistration_results(animalid, session, rootdir=rootdir, verbose=verbose)
    #vasculature = coreg_d['vasculature'].copy()
    fov_key = coreg_d['fov_key'] #'%s_%s' % (session, acquisition)
    fov_name = fov_key.split('%s_' % session)[-1]

    # Plot coregistration images
    fig, ax = pl.subplots()
    plot_coregistration_overlay(coreg_d['vasculature'], coreg_d['BV_warped'], ax=ax)
    ax.set_title(fov_key)
    pl.savefig(os.path.join(outdir, '%s_%s_%s_coregistration.png' % (session, animalid, fov_name)))
    pl.close()

    return coreg_d


def get_rois(animalid, session, fov_name, traceid='traces001', roiid=None):
    
    if roiid is None:
        roiid = roi_utils.get_roiid_from_traceid(animalid, session, fov_name, traceid=traceid)
    roi_masks, roi_zproj = roi_utils.load_roi_masks(animalid, session, fov_name, rois=roiid)
    print("Loaded rois: %s" % roiid)
    return roi_masks, roi_zproj, roiid


def warp_rois(coreg_d, roi_masks, roi_zproj, clip_lim=2.0, tile=5):
    print("warping 2d tranform to roi masks")
    # Format input images
    roi_zproj_eq = adjust_image_contrast(roi_zproj, clip=clip_lim, tile=5) 

    # Transform 2p fov to match WF orientation
    transf_zproj = transform_2p_fov(roi_zproj_eq, coreg_d['pixel_size']) 

    # Apply BV warp from coregistration to ROI images
    transform_mat = coreg_d['warp_mat'].copy() #fov_.alignment['transform_matrix'].copy()
    warped_zproj = warp_im(transf_zproj.astype(float), transform_mat, coreg_d['vasculature'].shape)

    # Apply warp to EACH roi
    d1, d2, nrois = roi_masks.shape  
    transf_rois = np.dstack([transform_2p_fov(roi_masks[:, :, i].astype(float), coreg_d['pixel_size'], normalize=False)                for i in np.arange(0, nrois)])
    warped_rois = np.dstack([warp_im(transf_rois[:, :, i], transform_mat, coreg_d['vasculature'].shape)\
                            for i in np.arange(0, nrois)])
       
    zproj_d = struct() #
    zproj_d.original = roi_zproj
    zproj_d.transformed = transf_zproj
    zproj_d.warped = warped_zproj
    zproj_d.equalized = roi_zproj_eq

    rois_d = struct()
    rois_d.original = roi_masks
    rois_d.transformed = transf_rois
    rois_d.warped = warped_rois

    return rois_d, zproj_d#, roi_zproj_eq


def plot_transformations(coreg_d, rois_d, zproj_d, cmap='jet'):

    roi_zproj_eq = zproj_d.equalized.copy()
    transf_zproj = zproj_d.transformed.copy()

    coreg_fov2p_warped = coreg_d['BV_warped'].copy()
    fov2p_aligned_overlay = np.ma.masked_where(coreg_fov2p_warped==0, coreg_fov2p_warped)
    nrois = rois_d.original.shape[-1]

    #### Create arrays for visualization
    transf_rois = rois_d.transformed
    warped_rois = rois_d.warped
    roi_masks = rois_d.original

    rois_orig_overlay = assign_int_to_masks(roi_masks.astype(bool).astype(int)) # original masks
    rois_transf_overlay = assign_int_to_masks(transf_rois.astype(bool).astype(int)) # transformed fov
    rois_warped_overlay = assign_int_to_masks(warped_rois.astype(bool).astype(int)) # wraped to WF

    # ## Plot all the steps
    fig = pl.figure(figsize=(12, 8), dpi=150)
    gs = gridspec.GridSpec(ncols=4, nrows=2) #, figure=fig)

    ax = fig.add_subplot(gs[0, 0])
    plot_roi_overlay(rois_orig_overlay, roi_zproj_eq, ax=ax,vmin=0, vmax=nrois, cmap=cmap)
    ax.set_title('Original 2p fov')

    ax = fig.add_subplot(gs[1, 0])
    plot_roi_overlay(rois_transf_overlay, transf_zproj, ax=ax, vmin=0, vmax=nrois, cmap=cmap)
    ax.set_title('Warped to WF')

    ax = fig.add_subplot(gs[0:, 1:])
    ax.imshow(coreg_d['vasculature'], cmap='gray')
    ax.imshow( fov2p_aligned_overlay, cmap='Greens', alpha=0.2)
    ax.imshow( rois_warped_overlay, cmap=cmap, alpha=1.0,  vmin=0, vmax=nrois)
    ax.axis('off')
    ax.set_title('2p/WF overlay')

    return fig


def save_results(outfile, coreg_d=dict(), zproj_d=None, rois_d=None):

    D = {'coreg': coreg_d,
        'zproj': zproj_d,
        'rois': rois_d}

    with open(outfile, 'wb') as f:
        pkl.dump(D, f, protocol=pkl.HIGHEST_PROTOCOL)
 

def do_fov_alignment(animalid, session, roiid=None, traceid='traces001', verbose=False,
                    rootdir='/n/coxfs01/2p-data', outdir='/tmp', clip_lim=2.0, cmap='jet',
                    plot=True):

    # Load coregistration results for animal
    # --------------------------------------------------
    coreg_d = load_and_plot_coregistration(animalid, session, verbose=verbose,
                                            rootdir=rootdir, outdir=outdir)

    # Select ROI ID to warp 
    # --------------------------------------------------
    fov_name = coreg_d['fov_key'].split('%s_' % session)[-1]
    roi_masks, roi_zproj, roiid = get_rois(animalid, session, fov_name, roiid=roiid, traceid=traceid)
    data_id = '%s_%s_%s_%s_%s' % (session, animalid, fov_name, roiid, traceid)
    print(data_id)

    # Warp 2p ROIs using transf matrix
    # ------------------------------------------
    rois_d, zproj_d = warp_rois(coreg_d, roi_masks, roi_zproj, clip_lim=clip_lim)
    roi_zproj_eq = zproj_d.equalized.copy()

 
    if plot:
        fig = plot_transformations(coreg_d, rois_d, zproj_d, cmap=cmap)
        putils.label_figure(fig, data_id)
        pl.subplots_adjust(left=0.01, right=0.99, hspace=0.2)
        pl.savefig(os.path.join(outdir, '%s.png' % data_id))
        #print(outdir, data_id)
        pl.close()

    #### Save
    outfile = os.path.join(outdir, '%s_results.pkl' % data_id)
    save_results(outfile, coreg_d=coreg_d, zproj_d=zproj_d, rois_d=rois_d)
    print("--- done ---\n--- saved to: ---\n%s" % outfile)
       
    return outfile



def extract_options(options):

    parser = optparse.OptionParser()

    parser.add_option('-D', '--root', action='store', dest='rootdir',
                          default='/n/coxfs01/2p-data',
                          help='data root dir (dir w/ all animalids) [default: /n/coxfs01/2pdata]')
    parser.add_option('-d', '--dest', action='store', dest='aggregate_dir',
                          default='/n/coxfs01/julianarhee/aggregate-visual-areas',
                          help='aggregate analysis base dir[default: /n/coxfs01/julianarhee/aggregate-visual-areas]')

    parser.add_option('-i', '--animalid', action='store', dest='animalid',
                          default='', help='Animal ID')
    # Set specific session/run for current animal:
    parser.add_option('-S', '--session', action='store', dest='session',
                          default='', help='session dir (format: YYYMMDD_ANIMALID')
    parser.add_option('-A', '--acq', action='store', dest='acquisition',
                          default='FOV1_zoom2p0x', help="acquisition folder (ex: 'FOV1_zoom3x') [default: FOV1]")
    parser.add_option('-r', '--roi-id', action='store', dest='roiid',
                          default=None, help="roi id of rois to warp (e.g., rois001)")
    parser.add_option('-t', '--trace-id', action='store', dest='traceid',
             default='traces001', help="trace id from which to get roi id to warp [default: traces001]")
 
    parser.add_option('-c', '--clip', action='store', dest='clip_lim',
             default=2.0, help="Clip limit for equalizing grayscale img for visualization [default: 2.0, higher for darker imgs]")
 

    parser.add_option('--new', action='store_true', dest='create_new',
                          default=False, help="flag to remake fov images")
    parser.add_option('-v', '--verbose', action='store_true', dest='verbose',
                          default=False, help="flag to print extra stuff")

    parser.add_option('--no-plot', action='store_false', dest='plot',
                          default=True, help="flag to not plot transform steps")
    parser.add_option('--cmap', action='store', dest='cmap',
                          default='jet', help="Color map to use for rois")
    
    (options, args) = parser.parse_args(options)
       
    return options


#rootdir = '/n/coxfs01/2p-data'
#animalid = 'JC113'
#session = '20191018'


def main(options):

    opts = extract_options(options)
    rootdir = opts.rootdir
    aggregate_dir = opts.aggregate_dir
    animalid = opts.animalid
    session = opts.session
    traceid = opts.traceid
    roiid = opts.roiid
    plot = opts.plot
    verbose = opts.verbose
    fov_name = opts.acquisition
    clip_lim = float(opts.clip_lim)
    cmap = opts.cmap

    # Set output dir
    outdir = os.path.join(aggregate_dir, 'data-stats', 'area-assignment', 'retinotopic-mapper')
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    print(outdir)

    outfile = do_fov_alignment(animalid, session, roiid=roiid, traceid=traceid,
                                verbose=verbose, rootdir=rootdir, outdir=outdir)

    
if __name__ == '__main__':
    main(sys.argv[1:])

# # outf = h5py.File(alignment_outfile, 'a')
# try:
#     #if fkey in outf.keys():
#     f = outf[fkey] if fkey in outf.keys() else outf.create_group(fkey)
#     #f = outf.create_group(fkey)

#     grp = f.create_dataset('coreg/fov2p_transformed', coreg_fov2p_transformed.shape, dtype=coreg_fov2p_transformed.dtype)
#     grp[...] = coreg_fov2p_transformed
#     grp = f.create_dataset('coreg/fov2p_warped', coreg_fov2p_warped.shape, dtype=coreg_fov2p_warped.dtype)
#     grp[...] = coreg_fov2p_warped
#     grp = f.create_dataset('coreg/warp_mat', transform_mat.shape, dtype=transform_mat.dtype)
#     grp[...] = transform_mat
#     grp = f.create_dataset('coreg/vasculature', vasculature.shape, dtype=vasculature.dtype)
#     grp[...] = vasculature
#     grp.attrs['fov2p_transformed'] = '2p fov rotated, flipped, and pixel-scaled'
#     grp.attrs['fov2p_warped'] = '2p fov warped to align to widefield vasculature image with warp_mat'
#     grp.attrs['warp_mat'] = 'Transformation matrix to align 2p to widefield (wrap_im())'
#     grp.attrs['source'] = coreg_dfile
#     grp.attrs['pixel_size'] = fov_.pixel_size


#     grp = f.create_dataset('zproj/image', roi_zproj.shape, dtype=roi_zproj.dtype)
#     grp[...] = roi_zproj
#     grp = f.create_dataset('zproj/transformed', roi_zproj_transf.shape, dtype=roi_zproj_transf.dtype)
#     grp[...] = roi_zproj_transf
#     grp = f.create_dataset('zproj/warped', roi_zproj_warped.shape, dtype=roi_zproj_warped.dtype)
#     grp[...] = roi_zproj_warped


#     grp = f.create_dataset('rois/masks', roi_masks.shape, dtype=roi_masks.dtype)
#     grp[...] = roi_masks
#     grp = f.create_dataset('rois/transformed', transf_rois.shape, dtype=transf_rois.dtype)
#     grp[...] = transf_rois
#     grp = f.create_dataset('rois/warped', warped_rois.shape, dtype=warped_rois.dtype)
#     grp[...] = warped_rois

#     grp.attrs['roiid'] = roi_id
#     grp.attrs['traceid'] = traceid
# except Exception as e:
#     traceback.print_exc()
# finally:
#     outf.close()


# np.savez(alignment_outfile, 
#          'coreg_transformed': fov2p_transformed,
#          'coreg_aligned': fov2p_to_widefield,
         
#          'roi_zproj': roi_zproj,
#          'roi_summed': roi_img,
#          'roi_masks': roi_masks,
         
#          'transform_mat': transform_mat,
         
#          )



