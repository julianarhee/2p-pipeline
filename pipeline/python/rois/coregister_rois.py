#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 16:31:56 2017

@author: julianarhee
"""


from __future__ import division
#from __future__ import print_function
import matplotlib
#matplotlib.use('TkAgg')
from builtins import zip
from builtins import str
from builtins import map
from builtins import range
from past.utils import old_div
import datetime
import cv2
import glob
import numpy as np
import os
from scipy.sparse import spdiags, issparse
from matplotlib import gridspec
from pipeline.python.evaluation.evaluate_motion_correction import get_source_info

# import caiman
from caiman.base.rois import com
import caiman as cm
from caiman.utils.visualization import plot_contours #get_contours

import time
import pylab as pl

import re
import json
import h5py
import cPickle as pkl
import scipy.io
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pprint
from scipy import ndimage
import optparse

pp = pprint.PrettyPrinter(indent=4)

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def serialize_json(instance=None, path=None):
    dt = {}
    dt.update(vars(instance))

def byteify(input):
    if isinstance(input, dict):
        return {byteify(key): byteify(value)
                for key, value in input.iteritems()}
    elif isinstance(input, list):
        return [byteify(element) for element in input]
    elif isinstance(input, unicode):
        return input.encode('utf-8')
    else:
        return input

def minimumWeightMatching(costSet):
    '''
    Computes a minimum-weight matching in a bipartite graph
    (A union B, E).

    costSet:
    An (m x n)-matrix of real values, where costSet[i, j]
    is the cost of matching the i:th vertex in A to the j:th 
    vertex of B. A value of numpy.inf is allowed, and is 
    interpreted as missing the (i, j)-edge.

    returns:
    A minimum-weight matching given as a list of pairs (i, j), 
    denoting that the i:th vertex of A be paired with the j:th 
    vertex of B.
    '''

    m, n = costSet.shape
    nMax = max(m, n)

    # Since the choice of infinity blocks later choices for that index, 
    # it is important that the cost matrix is square, so there
    # is enough space to shift the choices for infinity to the unused 
    # part of the cost-matrix.
    costSet_ = np.full((nMax, nMax), np.inf)
    costSet_[0 : m, 0 : n] = costSet
    assert costSet_.shape[0] == costSet_.shape[1]

    # We allow a cost to be infinity. Since scipy does not
    # support this, we use a workaround. We represent infinity 
    # by M = 2 * maximum cost + 1. The point is to choose a distinct 
    # value, greater than any other cost, so that choosing an 
    # infinity-pair is the last resort. The 2 times is for large
    # values for which x + 1 == x in floating point. The plus 1
    # is for zero, for which 2 x == x.
    try:
        practicalInfinity = 2 * costSet[costSet < np.inf].max() + 1
    except ValueError:
        # This is thrown when the indexing set is empty;
        # then all elements are infinities.
        practicalInfinity = 1

    # Replace infinitites with our representation.
    costSet_[costSet_ == np.inf] = practicalInfinity

    # Find a pairing of minimum total cost between matching second-level contours.
    iSet, jSet = scipy.optimize.linear_sum_assignment(costSet_)
    assert len(iSet) == len(jSet)

    # Return only pairs with finite cost.
    return [(iSet[k], jSet[k]) 
        for k in range(len(iSet)) 
        if costSet_[iSet[k], jSet[k]] != practicalInfinity]

#%%
def get_distance_matrix(A1, A2, dims, dist_maxthr=0.1, dist_exp=0.1, dist_overlap_thr=0.8):
    
    d1 = dims[0]
    d2 = dims[1]
    
    #% first transform A1 and A2 into binary masks
    M1 = np.zeros(A1.shape).astype('bool') #A1.astype('bool').toarray()        
    M2 = np.zeros(A2.shape).astype('bool') #A2.astype('bool').toarray()

    K1 = A1.shape[-1]
    K2 = A2.shape[-1]
    # print("K1", K1, "K2", K2)

    #%
    s = ndimage.generate_binary_structure(2,2)
    for i in np.arange(0, max(K1,K2)):
        if i < K1:
            A_temp = A1.toarray()[:,i]
            M1[A_temp>dist_maxthr*max(A_temp),i] = True
            labeled, nr_objects = ndimage.label(np.reshape(M1[:,i], (d1,d2), order='F'), s)  # keep only the largest connected component
            sizes = ndimage.sum(np.reshape(M1[:,i], (d1,d2), order='F'), labeled, range(1,nr_objects+1)) 
            maxp = np.where(sizes==sizes.max())[0] + 1 
            max_index = np.zeros(nr_objects + 1, np.uint8)
            max_index[maxp] = 1
            BW = max_index[labeled]
            M1[:,i] = np.reshape(BW, M1[:,i].shape, order='F')
        if i < K2:
            A_temp = A2.toarray()[:,i];
            M2[A_temp>dist_maxthr*max(A_temp),i] = True
            labeled, nr_objects = ndimage.label(np.reshape(M2[:,i], (d1,d2), order='F'), s)  # keep only the largest connected component
            sizes = ndimage.sum(np.reshape(M2[:,i], (d1,d2), order='F'), labeled, range(1,nr_objects+1)) 
            maxp = np.where(sizes==sizes.max())[0] + 1 
            max_index = np.zeros(nr_objects + 1, np.uint8)
            max_index[maxp] = 1
            BW = max_index[labeled]
            M2[:,i] = np.reshape(BW, M2[:,i].shape, order='F')

    #% determine distance matrix between M1 and M2
    D = np.zeros((K1,K2));
    for i in np.arange(0, K1):
        for j in np.arange(0, K2):
            
            overlap = float(np.count_nonzero(M1[:,i] & M2[:,j]))
            #print overlap
            totalarea = float(np.count_nonzero(M1[:,i] | M2[:,j]))
            #print totalarea
            smallestROI = min(np.count_nonzero(M1[:,i]),np.count_nonzero(M2[:,j]));
            #print smallestROI
                
            D[i,j] = 1 - (overlap/totalarea)**dist_exp
    
            if overlap >= dist_overlap_thr*smallestROI:
                #print('Too small!')
                D[i,j] = 0   
                
    return D

def find_matches_nmf(params_thr, output_dir, idxs_to_keep=None, save_output=True):
     # TODO:  Add 3D compatibility...
    coreg_outpath = None
    if save_output is True:
        coreg_outpath = os.path.join(output_dir, 'coreg_results_{}.hdf5'.format(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")))
        coreg_outfile = h5py.File(coreg_outpath, 'w')
        for k in params_thr.keys():
            if k == 'eval': # eval is a dict, will be saved in roiparams.json (no need to save as attr for coreg)
                continue
            coreg_outfile.attrs[k] = params_thr[k]
        coreg_outfile.attrs['creation_date'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    output_dir_figs = os.path.join(output_dir, 'figures')
    if not os.path.exists(output_dir_figs):
        os.makedirs(output_dir_figs)
        
    all_matches = dict()
    ref_file = str(params_thr['ref_filename'])
    try:
        # Load reference file info:
        ref = np.load(params_thr['ref_filepath'])
        nr = ref['A'].all().shape[1]
        dims = ref['dims']   
        A1 = ref['A'].all()
        if params_thr['keep_good_rois'] is True:
            if idxs_to_keep is None:
                ref_idx_components = ref['idx_components']
            else:
                ref_idx_components = idxs_to_keep[ref_file]
            A1 = A1[:, ref_idx_components]
            nr = A1.shape[-1]
            
        # For each file, find best matching ROIs to ref:
        nmf_src_dir = os.path.split(params_thr['ref_filepath'])[0]
        nmf_fns = [n for n in os.listdir(nmf_src_dir) if n.endswith('npz')]
        for nmf_fn in nmf_fns:
            
            curr_file = str(re.search('File(\d{3})', nmf_fn).group(0))

            if nmf_fn == os.path.basename(params_thr['ref_filepath']):
                if save_output is True:
                    idx_components = np.array(ref_idx_components)
                    kpt = coreg_outfile.create_dataset('/'.join([curr_file, 'roi_idxs']), idx_components.shape, idx_components.dtype)
                    kpt[...] = idx_components
                continue

            nmf = np.load(os.path.join(nmf_src_dir, nmf_fn))
            print "Loaded %s..." % curr_file
            nr = nmf['A'].all().shape[1]
            A2 = nmf['A'].all()
            if params_thr['keep_good_rois'] is True:
                if idxs_to_keep is None:
                    idx_components = nmf['idx_components']
                else:
                    idx_components = idxs_to_keep[curr_file]
                print("Keeping %i out of %i components." % (len(idx_components), nr))
                A2 = A2[:,  idx_components]
                nr = A2.shape[-1]
            
            # Calculate distance matrix between ref and all other files:
            D = get_distance_matrix(A1, A2, dims, 
                                    dist_maxthr=params_thr['dist_maxthr'], 
                                    dist_exp=params_thr['dist_exp'], 
                                    dist_overlap_thr=params_thr['dist_overlap_thr'])

            if save_output is True:
                idx_components = np.array(idx_components)
                kpt = coreg_outfile.create_dataset('/'.join([curr_file, 'roi_idxs']), idx_components.shape, idx_components.dtype)
                kpt[...] = idx_components
                d = coreg_outfile.create_dataset('/'.join([curr_file, 'distance']), D.shape, D.dtype)
                d[...] = D
                d.attrs['d1'] = dims[0]
                d.attrs['d2'] = dims[1]
                if len(dims) > 2:
                    d.attrs['d3'] = dims[2]
                d.attrs['source'] = os.path.join(nmf_src_dir, nmf_fn)
                
            # Set illegal matches (distance vals greater than dist_thr):
            D[D>params_thr['dist_thr']] = np.inf #1E100 #np.nan #1E9
            if save_output is True:
                dthr = coreg_outfile.create_dataset('/'.join([curr_file, 'distance_thr']), D.shape, D.dtype)
                dthr[...] = D
                dthr.attrs['dist_thr'] = params_thr['dist_thr']
                
            # Save distance matrix for curr file:
            pl.figure()
            pl.imshow(D); pl.colorbar();
            pl.title('%s - dists to ref (%s, overlap_thr %s)' % (curr_file, ref_file, str(params_thr['dist_overlap_thr'])))
            pl.savefig(os.path.join(output_dir_figs, 'distancematrix_%s.png' % curr_file))
            pl.close()

            #% Get matches using thresholds on distance matrix:
            matches = minimumWeightMatching(D)  # Use modified linear_sum_assignment to allow np.inf
            print("Found %i ROI matches in %s" % (len(matches), curr_file))
            
            if save_output is True:
                matches = np.array(matches)
                match = coreg_outfile.create_dataset('/'.join([curr_file, 'matches']), matches.shape, matches.dtype)
                match[...] = matches
                
            # Store matches for file:
            if not isinstance(matches, list):
                all_matches[curr_file] = matches.tolist()

        # Also save to json for easy viewing:
        match_fn_base = 'matches_byfile_r%s' % str(params_thr['ref_filename'])
        with open(os.path.join(output_dir, '%s.json' % match_fn_base), 'w') as f:
            json.dump(all_matches, f, indent=4, sort_keys=True)
         
    except Exception as e:
        print "-- ERROR: in finding matches to ref. --------------------------"
        print traceback.print_exc()
        print "---------------------------------------------------------------"
    finally:
        coreg_outfile.close()
    
    if save_output is True:
        coreg_outfile.close()
        
    return all_matches, coreg_outpath


def plot_matched_rois(all_matches, params_thr, savefig_dir, idxs_to_keep=None):
    # TODO:  Add 3D compatibility...
    if not os.path.exists(savefig_dir):
        os.makedirs(savefig_dir)
    
    src_nmf_dir = os.path.split(params_thr['ref_filepath'])[0]
    source_nmf_paths = sorted([os.path.join(src_nmf_dir, n) for n in os.listdir(src_nmf_dir) if n.endswith('npz')], key=natural_keys) # Load nmf files
        
    # Load reference:
    ref_file = str(params_thr['ref_filename'])
    ref = np.load(params_thr['ref_filepath'])
    nr = ref['A'].all().shape[1]
    dims = ref['dims']
    if len(ref['dims']) > 2:
        is3D = True
        d1 = int(ref['dims'][0])
        d2 = int(ref['dims'][1])
        d3 = int(ref['dims'][2])
    else:
        is3D = False
        d1 = int(ref['dims'][0])
        d2 = int(ref['dims'][1])
    A1 = ref['A'].all()
    if params_thr['keep_good_rois'] is True:
        if idxs_to_keep is None:
            idx_components = ref['idx_components']
        else:
            idx_components = idxs_to_keep[ref_file]
        A1 = A1[:,  idx_components]
        nr = A1.shape[-1]
    masks = np.reshape(np.array(A1.todense()), (d1, d2, nr), order='F')
    print "Loaded reference masks with shape:", masks.shape
    img = ref['Av']
        
    for curr_file in all_matches.keys():

        if curr_file==params_thr['ref_filename']:
            continue
 
        nmf_path = [f for f in source_nmf_paths if curr_file in f][0]
        nmf = np.load(os.path.join(nmf_path))
        A2 = nmf['A'].all()
        nr = A2.shape[-1]
 
        #% Save overlap of REF with curr-file matches:
        if params_thr['keep_good_rois'] is True:
            if idxs_to_keep is None:
                idx_components = nmf['idx_components']
            else:
                idx_components = idxs_to_keep[curr_file]
            print("Keeping %i out of %i components." % (len(idx_components), nr))
            A2 = A2[:,  idx_components]
            nr = A2.shape[-1]
        masks2 = np.reshape(np.array(A2.todense()), (d1, d2, nr), order='F')

        # Plot contours overlaid on reference image:
        pl.figure()
        pl.imshow(img, cmap='gray')
            
        if issparse(A1): 
            A1 = np.array(A1.todense()) 
        A2 = np.array(A2.todense())
        matches = all_matches[curr_file]
        for ridx,match in enumerate(matches):
            roi1=match[0]; roi2=match[1]
            
            x, y = np.mgrid[0:d1:1, 0:d2:1]
    
            # Draw contours for REFERENCE:
            indx = np.argsort(A1[:,roi1], axis=None)[::-1]
            cumEn = np.cumsum(A1[:,roi1].flatten()[indx]**2)
            cumEn /= cumEn[-1] # normalize
            Bvec = np.zeros(d1*d2)
            Bvec[indx] = cumEn
            Bmat = np.reshape(Bvec, (d1,d2), order='F')
            cs = pl.contour(y, x, Bmat, [0.9], colors='b') #[colorvals[fidx]]) #, cmap=colormap)

            # Draw contours for CURRFILE matches:
            indx = np.argsort(A2[:,roi2], axis=None)[::-1]
            cumEn = np.cumsum(A2[:,roi2].flatten()[indx]**2)
            cumEn /= cumEn[-1] # normalize
            Bvec = np.zeros(d1*d2)
            Bvec[indx] = cumEn
            Bmat = np.reshape(Bvec, (d1,d2), order='F')
            cs = pl.contour(y, x, Bmat, [0.9], colors='r') #[colorvals[fidx]]) #, cmap=colormap)

            # Label ROIs with original roi nums:
            masktmp1 = masks[:,:,roi1]; masktmp2 = masks2[:,:,roi2]
            [ys, xs] = np.where(masktmp1>0)
            pl.text(xs[int(round(len(xs)/4))], ys[int(round(len(ys)/4))], str(roi1), color='b') #, weight='bold')
            [ys, xs] = np.where(masktmp2>0)
            pl.text(xs[int(round(len(xs)/4))], ys[int(round(len(ys)/4))], str(roi2), color='r') #, weight='bold')
                    
        pl.savefig(os.path.join(savefig_dir, 'matches_%s_%s.png' % (str(ref_file), str(curr_file))))
        pl.close()


def coregister_rois_nmf(params_thr, coreg_output_dir, excluded_tiffs=[], idxs_to_keep=None):
    
    ref_rois = []
    
    if not os.path.exists(coreg_output_dir):
        os.makedirs(coreg_output_dir)
    
    # Get matches:
    all_matches, coreg_results_path = find_matches_nmf(params_thr, coreg_output_dir, idxs_to_keep=idxs_to_keep, save_output=True)
    
    # Plot matches over reference:
    coreg_figdir = os.path.join(coreg_output_dir, 'figures')
    plot_matched_rois(all_matches, params_thr, coreg_figdir, idxs_to_keep=idxs_to_keep)

    #% Find intersection of all matches with reference:
    filenames = all_matches.keys()
    filenames.extend([str(params_thr['ref_filename'])])
    filenames = sorted(filenames, key=natural_keys)

    ref_idxs = [[comp[0] for comp in all_matches[f]] for f in all_matches.keys() if f not in excluded_tiffs]
    print "REF idxs:", len(ref_idxs)
    #file_match_max = [len(r) for r in ref_idxs].index(max(len(r) for r in ref_idxs))
    
    ref_rois = set(ref_idxs[0])
    for s in ref_idxs[1:]:
        ref_rois.intersection_update(s)
    ref_rois = list(ref_rois)
    
    return ref_rois, coreg_results_path

                    
def plot_coregistered_rois(matchedROIs, params_thr, src_filepaths, save_dir, idxs_to_keep=None, cmap='jet', plot_by_file=True):
    
    src_nmf_dir = os.path.split(params_thr['ref_filepath'])[0]
    source_nmf_paths = sorted([os.path.join(src_nmf_dir, n) for n in os.listdir(src_nmf_dir) if n.endswith('npz')], key=natural_keys) # Load nmf files
        
    # Load ref img:
    ref_fn = [f for f in source_nmf_paths if str(params_thr['ref_filename']) in f and f.endswith('npz')][0]
    ref = np.load(ref_fn)
    refimg = ref['Av']
    
    colormap = pl.get_cmap(cmap)
    ref_rois = matchedROIs[params_thr['ref_filename']]
    nrois = len(ref_rois)
    print "Plotting %i coregistered ROIs from each file..." % nrois
    
    file_names = matchedROIs.keys();
    if plot_by_file is True:
        plot_type = 'byfile'
        colorvals = colormap(np.linspace(0, 1, len(file_names))) #get_spaced_colors(nrois)
        print "Plotting coregistered by file. Found %i files." % len(file_names)
    else:
        plot_type = 'byroi'
        colorvals = colormap(np.linspace(0, 1, len(ref_rois))) #get_spaced_colors(nrois)
        print "Plotting coregistered by ROI. Found %i rois." % len(ref_rois)

    colorvals[:,3] *= 0.5
    
    fig = pl.figure(figsize=(12, 10)) 
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1]) 
    gs.update(wspace=0.05, hspace=0.05)
    ax1 = pl.subplot(gs[0])
    
    ax1.imshow(refimg, cmap='gray')
    pl.axis('equal')
    pl.axis('off')
    blank = np.ones(refimg.shape)*np.nan
    
    for fidx,curr_file in enumerate(sorted(matchedROIs.keys(), key=natural_keys)):
        
        src_path = [f for f in src_filepaths if curr_file in f][0]
        nmf = np.load(src_path)
        nr = nmf['A'].all().shape[1]
        d1 = int(nmf['dims'][0])
        d2 = int(nmf['dims'][1])
        dims = (d1, d2)
        x, y = np.mgrid[0:d1:1, 0:d2:1]
        A = nmf['A'].all()
        nr = A.shape[-1]
        
        if params_thr['keep_good_rois'] is True:
            if idxs_to_keep is None:
                idx_components = nmf['idx_components']
            else:
                idx_components = idxs_to_keep[curr_file]
            A = A[:, idx_components]
            nr = A.shape[-1]
      
        curr_rois = matchedROIs[curr_file]
        A = np.array(A.todense()) 
        
        for ridx, roi in enumerate(curr_rois):
            #print roi
            # compute the cumulative sum of the energy of the Ath component that 
            # has been ordered from least to highest:
            indx = np.argsort(A[:,roi], axis=None)[::-1]
            cumEn = np.cumsum(A[:,roi].flatten()[indx]**2)
            cumEn /= cumEn[-1] # normalize
            Bvec = np.zeros(d1*d2)
            Bvec[indx] = cumEn
            Bmat = np.reshape(Bvec, (d1,d2), order='F')
            #currcolor = (colorvals[fidx][0], colorvals[fidx][1], colorvals[fidx][2], 0.5)
            if plot_by_file is True:
                cs = pl.contour(y, x, Bmat, [0.9], colors=[colorvals[fidx]]) #, cmap=colormap)
            else:
                cs = pl.contour(y, x, Bmat, [0.9], colors=[colorvals[ridx]]) #, cmap=colormap)
    #pl.axis('equal')
    #pl.axis('off')
    #pl.savefig(os.path.join(save_dir, 'contours_%s_r%s_rois.png' % (plot_type,  str(params_thr['ref_filename']))))
    
    nfiles = len(matchedROIs.keys())
    nrois = len(ref_rois)
    print "N files:", nfiles
    print "N rois:", nrois
    gap = 1
    #%
    #pl.figure()
    ax2 = pl.subplot(gs[1])
    if plot_by_file is True:
        interval = np.arange(0., 1., 1./nfiles)
        for fidx,curr_file in enumerate(sorted(matchedROIs.keys(), key=natural_keys)):
            ax2.plot(1, interval[fidx], c=colorvals[fidx], marker='.', markersize=20)
            pl.text(1.1, interval[fidx], str(curr_file), fontsize=12)
            pl.xlim([0.95, 2])
    else:
        interval = np.arange(0., 1., 1./nrois)
        for ridx, roi in enumerate(ref_rois):
            ax2.plot(1, interval[ridx], c=colorvals[ridx], marker='.', markersize=20)
            pl.text(1.1, interval[ridx], str(roi), fontsize=12)
            pl.xlim([0.95, 2])
    pl.axis('equal')
    pl.axis('off')
    
    #%
    pl.savefig(os.path.join(save_dir, 'contours_%s_r%s.png' % (plot_type,  str(params_thr['ref_filename'])))) #matchedrois_fn_base))
    pl.close()
    
    
def load_rid(session_dir, roi_id, auto=False):
    
    RID = None
    
    session = os.path.split(session_dir)[1]
    print "SESSION:", session
    
    roi_base_dir = os.path.join(session_dir, 'ROIs') #acquisition, run)
    tmp_rid_dir = os.path.join(roi_base_dir, 'tmp_rids')
    
    try:
        print "Loading params for ROI SET, id %s" % roi_id
        roidict_path = os.path.join(roi_base_dir, 'rids_%s.json' % session)
        with open(roidict_path, 'r') as f:
            roidict = json.load(f)
        RID = roidict[roi_id]
        pp.pprint(RID)
    except Exception as e:
        print "No ROI SET entry exists for specified id: %s" % roi_id
        traceback.print_exc()
        try:
            print "Checking tmp roi-id dir..."
            if auto is False:
                while True:
                    tmpfns = [t for t in os.listdir(tmp_rid_dir) if t.endswith('json')]
                    for ridx, ridfn in enumerate(tmpfns):
                        print ridx, ridfn
                    userchoice = raw_input("Select IDX of found tmp roi-id to view: ")
                    with open(os.path.join(tmp_rid_dir, tmpfns[int(userchoice)]), 'r') as f:
                        tmpRID = json.load(f)
                    print "Showing tid: %s, %s" % (tmpRID['roi_id'], tmpRID['rid_hash'])
                    pp.pprint(tmpRID)
                    userconfirm = raw_input('Press <Y> to use this roi ID, or <q> to abort: ')
                    if userconfirm == 'Y':
                        RID = tmpRID
                        break
                    elif userconfirm == 'q':
                        break
        except Exception as e:
            print "---------------------------------------------------------------"
            print "No tmp roi-ids found either... ABORTING with error:"
            traceback.print_exc()
            print "---------------------------------------------------------------"
    
    return RID, roidict

            
#%%
def run_coregistration(options):

    parser = optparse.OptionParser()
    
    parser.add_option('-R', '--root', action='store', dest='rootdir', default='/nas/volume1/2photon/data', help='data root dir (root project dir containing all animalids) [default: /nas/volume1/2photon/data, /n/coxfs01/2pdata if --slurm]')
    parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')
    parser.add_option('-S', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID')
    parser.add_option('--slurm', action='store_true', dest='slurm', default=False, help="set if running as SLURM job on Odyssey")
    parser.add_option('--default', action='store_true', dest='default', default='store_false', help="Use all DEFAULT params, for params not specified by user (no interactive)")
    
    parser.add_option('-r', '--roi-id', action='store', dest='roi_id', default='', help="ROI ID for rid param set to use (created with set_roi_params.py, e.g., rois001, rois005, etc.)")
    
    parser.add_option('-t', '--maxthr', action='store', dest='dist_maxthr', default=0.1, help="threshold for turning spatial components into binary masks [default: 0.1]")
    parser.add_option('-n', '--power', action='store', dest='dist_exp', default=0.1, help="power n for distance between masked components: dist = 1 - (and(M1,M2)/or(M1,M2)**n [default: 1]")
    parser.add_option('-d', '--dist', action='store', dest='dist_thr', default=0.5, help="threshold for setting a distance to infinity, i.e., illegal matches [default: 0.5]")
    parser.add_option('-o', '--overlap', action='store', dest='dist_overlap_thr', default=0.8, help="overlap threshold for detecting if one ROI is subset of another [default: 0.8]")
    
    
    parser.add_option('-x', '--exclude', action="store",
                      dest="exclude_file_ids", default='', help="comma-separated list of files to exclude")
    parser.add_option('--good', action="store_true",
                      dest="keep_good_rois", default=False, help="Set flag to only keep good components (useful for avoiding computing massive ROI sets)")
    parser.add_option('--max', action="store_true",
                      dest="use_max_nrois", default=False, help="Set flag to use file with max N components (instead of reference file) [default uses reference]")
    parser.add_option('--roipath', action="store",
                      dest="roipath", default="", help="If keep_good_rois is True, path to .json with ROI idxs for each file (if using cNMF). Default uses nmf-extraction ROIs.")
    parser.add_option('-O', '--outdir', action="store",
                      dest="coreg_output_dir", default="", help="Output dir to save coreg results to. Default uses curr ROI dir + 'src_coreg_results'")


    (options, args) = parser.parse_args(options)
    
    if options.slurm is True:
        if 'coxfs01' not in options.rootdir:
            options.rootdir = '/n/coxfs01/2p-data'
    
    # Set USER INPUT options:
    rootdir = options.rootdir
    animalid = options.animalid
    session = options.session
    #acquisition = options.acquisition
    #run = options.run
    roi_id = options.roi_id
    slurm = options.slurm
    auto = options.default

    tmp_exclude = options.exclude_file_ids
    if len(tmp_exclude)==0:
        exclude_manual = []
    else:
        exclude_files = tmp_exclude.split(',')
        exclude_files = [int(f) for f in exclude_files]
        exclude_manual = ['File%03d' % f for f in exclude_files]
    print("Excluding files: ", exclude_manual)
    
    keep_good_rois = options.keep_good_rois
    use_max_nrois = options.use_max_nrois
    
    roipath = options.roipath
    if len(roipath) == 0:
        idxs_to_keep = None
    else:
        try:
            idxs_to_keep = dict()
            print "Loaded ROI info for files:",
            print roipath
            roi_eval_info = h5py.File(roipath, 'r')
            for f in roi_eval_info.keys():
                print f, len(roi_eval_info[f]['idxs_to_keep'])
                idxs_to_keep[str(f)] = np.array(roi_eval_info[f]['idxs_to_keep'])
            roi_eval_info.close()
            pp.pprint(idxs_to_keep)
        except Exception as e:
            roi_eval_info.close()
            print "ERROR LOADING ROI idxs ------------------------------------"
            print traceback.print_exc()
            print "User provided ROI idx path:"
            print roipath
            print "-----------------------------------------------------------"
    
    coreg_output_dir = options.coreg_output_dir
    
    #%% ca-source-extraction options:
        
    params_thr = dict()
    
    # dist_maxthr:      threshold for turning spatial components into binary masks (default: 0.1)
    # dist_exp:         power n for distance between masked components: dist = 1 - (and(m1,m2)/or(m1,m2))^n (default: 1)
    # dist_thr:         threshold for setting a distance to infinity. (default: 0.5)
    # dist_overlap_thr: overlap threshold for detecting if one ROI is a subset of another (default: 0.8)
        
    params_thr['dist_maxthr'] = options.dist_maxthr #0.1
    params_thr['dist_exp'] = options.dist_exp # 1
    params_thr['dist_thr'] = options.dist_thr #0.5
    params_thr['dist_overlap_thr'] = options.dist_overlap_thr #0.8
    params_thr['keep_good_rois'] = keep_good_rois
    if use_max_nrois is True:
        params_thr['filter_type'] = 'max'
    else:
        params_thr['filter_type'] = 'ref'
    
    # =============================================================================
    # Load specified ROI-ID parameter set:
    # =============================================================================
    session_dir = os.path.join(rootdir, animalid, session)
    
    RID, roidict = load_rid(session_dir, roi_id, auto=auto)


    #%%
    # =============================================================================
    # Get meta info for current run and source tiffs using trace-ID params:
    # =============================================================================
    #rid_hash = RID['rid_hash']
    tiff_dir = RID['SRC']
    roi_id = RID['roi_id']
    roi_type = RID['roi_type']
    tiff_files = sorted([t for t in os.listdir(tiff_dir) if t.endswith('tif')], key=natural_keys)
    print "Found %i tiffs in dir %s.\nExtracting %s ROIs...." % (len(tiff_files), tiff_dir, roi_type)
    
    acquisition = tiff_dir.split(session)[1].split('/')[1]
    run = tiff_dir.split(session)[1].split('/')[2]
    process_id = tiff_dir.split(session)[1].split('/')[4]
    
    filenames = ['File%03d' % int(ti+1) for ti, t in enumerate(os.listdir(tiff_dir)) if t.endswith('tif')]
    print "Source tiffs:"
    for f in filenames:
        print f
        
    #%% If motion-corrected (standard), check evaluation:
    print "Loading Motion-Correction Info...======================================="
    mcmetrics_filepath = None
    mcmetric_type = None
    excluded_tiffs = []
    if 'mcorrected' in RID['SRC']:
        try:
            mceval_dir = '%s_evaluation' % RID['SRC']
            assert 'mc_metrics.hdf5' in os.listdir(mceval_dir), "MC output file not found!"
            mcmetrics_filepath = os.path.join(mceval_dir, 'mc_metrics.hdf5')
            mcmetrics = h5py.File(mcmetrics_filepath, 'r')
            print "Loaded MC eval file. Found metric types:"
            for ki, k in enumerate(mcmetrics):
                print ki, k
        except Exception as e:
            traceback.print_exc()
            print "Unable to load motion-correction evaluation info."
            
        # Use zprojection corrs to find bad files:
        mcmetric_type = 'zproj_corrcoefs'
        bad_files = mcmetrics['zproj_corrcoefs'].attrs['bad_files']
        if len(bad_files) > 0:
            print "Found %i files that fail MC metric %s:" % (len(bad_files), mcmetric_type)
            for b in bad_files:
                print b
            fidxs_to_exclude = [int(f[4:]) for f in bad_files]
            if len(fidxs_to_exclude) > 1:
                exclude_str = ','.join([i for i in fidxs_to_exclude])
            else:
                exclude_str = str(fidxs_to_exclude[0])
        else:
            exclude_str = ''
    
        # Get process info from attrs stored in metrics file:
        mc_ref_channel = mcmetrics.attrs['ref_channel']
        mc_ref_file = mcmetrics.attrs['ref_file']
    else:
        acquisition_dir = os.path.join(session_dir, acquisition)
        info = get_source_info(acquisition_dir, run, process_id)
        mc_ref_channel = info['ref_channel']
        mc_ref_file = info['ref_file']
        del info
    
    if len(exclude_str) > 0:
        filenames = [f for f in filenames if int(f[4:]) not in [int(x) for x in exclude_str.split(',')]]
        excluded_tiffs = ['File%03d' % int(fidx) for fidx in exclude_str.split(',')]
    
    if len(exclude_manual) > 0:
        excluded_tiffs.extend(exclude_manual)
        
    print "Motion-correction info:"
    print "MC reference is %s, %s." % (mc_ref_file, mc_ref_channel)
    print "Found %i tiff files to exclude based on MC EVAL: %s." % (len(excluded_tiffs), mcmetric_type)
    print "======================================================================="
    
    if len(exclude_manual) > 0:
        excluded_tiffs.extend(exclude_manual)
    print "Additionally excluding manully-selected tiffs."
    print "Excluded:", excluded_tiffs
    
    params_thr['excluded_tiffs'] = excluded_tiffs
    
    #%%
    # Determine which file should be used as "reference" for coregistering ROIs:
    roi_ref_type = RID['PARAMS']['options']['source']['roi_type']
    roi_source_dir = RID['PARAMS']['options']['source']['roi_dir']
    
    if roi_ref_type == 'caiman2D':
        src_nmf_dir = os.path.join(roi_source_dir, 'nmfoutput')
        source_nmf_paths = sorted([os.path.join(src_nmf_dir, n) for n in os.listdir(src_nmf_dir) if n.endswith('npz')], key=natural_keys) # Load nmf files
        if use_max_nrois is True:
            src_nrois = []
            for src_nmf_path in source_nmf_paths:
                snmf = np.load(src_nmf_path)
                fname = re.search('File(\d{3})', src_nmf_path).group(0)
                nall = snmf['A'].all().shape[1]
                npass = len(snmf['idx_components'])
                src_nrois.append((fname, nall, npass))
            if keep_good_rois is True:
                nmax_idx = [s[2] for s in src_nrois].index(max([s[2] for s in src_nrois]))
                nrois_max = src_nrois[nmax_idx][2]
            else:
                nmax_idx = [s[1] for s in src_nrois].index(max([s[1] for s in src_nrois]))
                nrois_max = src_nrois[nmax_idx][1]
            params_thr['ref_filename'] = src_nrois[nmax_idx][0]
            params_thr['ref_filepath'] = source_nmf_paths[nmax_idx]
            print "Using source %s as reference. Max N rois: %i" % (params_thr['ref_filename'], nrois_max)
        else:
            params_thr['ref_filename'] = mc_ref_file
            params_thr['ref_filepath'] = source_nmf_paths.index([i for i in source_nmf_paths if mc_ref_file in i][0])
        
        #% Load eval params from src: 
        if len(coreg_output_dir) == 0:
            coreg_output_dir = os.path.join(RID['DST'], 'src_coreg_results')            
        print "Saving COREG results to:", coreg_output_dir
        if not os.path.exists(coreg_output_dir):
            os.makedirs(coreg_output_dir)
            
        if keep_good_rois is True:
            with open(os.path.join(roi_source_dir, 'roiparams.json'), 'r') as f:
                src_roiparams = json.load(f)
                src_evalparams = src_roiparams['eval']
            print "-----------------------------------------------------------"
            print "Coregistering ROIS from source..."
            print "Source ROIs have been filtered with these eval params:"
            for k in src_evalparams.keys():
                print k, ':', src_evalparams[k]
            print "-----------------------------------------------------------"
        # Save info to current coreg dir:
        params_thr['eval'] = src_evalparams
        with open(os.path.join(coreg_output_dir, 'coreg_params.json'), 'w') as f:
            json.dump(params_thr, f, indent=4, sort_keys=True)
                
        # COREGISTER ROIS:
        ref_rois, coreg_results_path = coregister_rois_nmf(params_thr, coreg_output_dir, excluded_tiffs=excluded_tiffs, idxs_to_keep=idxs_to_keep)
        print("Found %i common ROIs matching reference." % len(ref_rois))
        
        #% Load coregistered roi matches and save universal matches:
        coreg_info = h5py.File(coreg_results_path, 'r')
        filenames = [str(i) for i in coreg_info.keys()]
        filenames.append(str(params_thr['ref_filename']))
        filenames = sorted(filenames, key=natural_keys)
        
        # Save ROI idxs for each file that matches ref and is common to all:
        matchedROIs = dict()
        for curr_file in filenames: #all_matches.keys():
            print curr_file
            if curr_file in excluded_tiffs:
                continue
            if curr_file==str(params_thr['ref_filename']):
                matchedROIs[curr_file] = ref_rois
            else:
                curr_matches = coreg_info[curr_file]['matches']
                matchedROIs[curr_file] = [curr_matches[[i[0] for i in curr_matches].index(r)][1] for r in ref_rois]
        
        coreg_info.close()
        
        # Save ROI idxs of unviersal matches:
        matchedrois_fn_base = 'coregistered_r%s' % str(params_thr['ref_filename'])
        print("Saving matches to: %s" % os.path.join(coreg_output_dir, matchedrois_fn_base))
        with open(os.path.join(coreg_output_dir, '%s.json' % matchedrois_fn_base), 'w') as f:
            json.dump(matchedROIs, f, indent=4, sort_keys=True)
        
        # Save plots of universal matches:
        if len(ref_rois) > 0:
            plot_coregistered_rois(matchedROIs, params_thr, source_nmf_paths, coreg_output_dir, idxs_to_keep=idxs_to_keep, plot_by_file=True)
            plot_coregistered_rois(matchedROIs, params_thr, source_nmf_paths, coreg_output_dir, idxs_to_keep=idxs_to_keep, plot_by_file=False)
    
    return ref_rois, params_thr, coreg_results_path
        
#%%
def main(options):

    ref_rois, params_thr, coreg_results_path = run_coregistration(options)

    print "----------------------------------------------------------------"
    print "Finished coregistration."
    print "Found %i matches across files to reference." % len(ref_rois)
    print "Saved output to:"
    print coreg_results_path
    print "----------------------------------------------------------------"


    #%% Get NMF output files:

if __name__ == '__main__':
    main(sys.argv[1:])
