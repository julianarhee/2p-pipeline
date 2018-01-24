#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This script finds matches between ROIs in a specified reference file and all other files.
User can choose to coregister only subsets of ROIs by providing a key that identifies which evaluation result set to use.

# 1.  Load ROI evaluation info, if relevant (evalparams).
- User-provided key specifying a particular evaluation result
- Evaluation set is keyed with a date-time string (output of evaulate_roi_extraction.py)

# 2.  Populate coregistration params (params_thr).
    - dist_maxthr <float>
        	threshold for turning spatial components into binary masks (default: 0.1)
    - dist_exp <float>
        	power n for distance between masked components: dist = 1 - (and(m1,m2)/or(m1,m2))^n (default: 1.0)
    - dist_thr <float>
        	threshold for setting a distance to infinity (illegal matches) (default: 0.5)
    - dist_overlap_thr <float>
        	overlap threshold for detecting if one ROI is a subset of another (default: 0.8)

# 3.  Load ROI set info and get roi/tiff source paths.

# 4.  Identify file to be used as coregistration reference.

# 5.  Save coregistration parameters to output dir:
	- params_thr saved to: 
	<rid_dir>/coreg_results/coreg_params.json

# 6.  Run coregistration (coregister_rois_nmf)

	a. Find matches to reference for each file and save results (coreg_results_path):
	   [outfile]:  <rid_dir>/coreg_results/coreg_results_<datetimestr>.hdf5
	   --> file groups (File001, File002, etc.) each contain datasets:
		roi_idxs :  indices of rois (source indices)
		distance :  distance matrix between ref and current file
		distance_thr :  thresholded distance matrix (params_thr['dist_thr'])
		matches :  list of matches between ref rois and current file rois

	b. Plot distance matrix and contour plots for each file's matches to reference:
	   [outfiles]:  <rid_dir>/coreg_results/figures/files/distancematrix_FileXXX.png

	c. Save matches to json for easy acccess:
	   [outfile]:  <rid_dir>/coreg_results/matches_byfile_rFileRRR.json,
	   where rFileRRR is the reference file.

	d. Plot ROI matches for each file:
	   [outfiles]:  <rid_dir>/coreg_results/figures/matches_rFileRRR_FileXXX.png

	e. Identify universal ROIs, i.e., ROI idxs common across all files (ref_rois)

	f. Save univeral ROIs to json file:
	   [outfile]:  <rid_dir>/coreg_results/coregistered_rFileRRR.json

	g. Plot universal ROIs on contour plots:
	   [outfiles]: <rid_dir>/coreg_results/figures/contours_byfile_rFileRRR.png
		       <rid_dir>/coreg_results/figures/contours_byroi_rFileRRR.png


Created on Tue Nov  7 16:31:56 2017

@author: julianarhee
"""


from __future__ import division
#import __builtin__
#from __future__ import print_function
import matplotlib
from future.utils import native
matplotlib.use('TkAgg')
#from builtins import zip
#from builtins import str
#from builtins import map
#from builtins import range
import datetime
import traceback
import numpy as np
import os
from scipy.sparse import issparse
from matplotlib import gridspec
from pipeline.python.rois.utils import load_RID, get_source_paths

import pylab as pl

import re
import json
import h5py
import scipy.io
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

#%%
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

#%%
def find_matches_nmf(params_thr, output_dir, idxs_to_keep=None, save_output=True):
     # TODO:  Add 3D compatibility...
    coreg_outpath = None
    if save_output is True:
        coreg_outpath = os.path.join(output_dir, 'coreg_results_{}.hdf5'.format(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")))
        coreg_outfile = h5py.File(coreg_outpath, 'w')
        for k in params_thr.keys():
            if k == 'eval': # eval is a dict, will be saved in roiparams.json (no need to save as attr for coreg)
                continue
            print k, type(params_thr[k])
            coreg_outfile.attrs[k] = native(params_thr[k])
        coreg_outfile.attrs['creation_date'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    output_dir_figs = os.path.join(output_dir, 'figures', 'files')
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
        traceback.print_exc()
        print "---------------------------------------------------------------"
    finally:
        coreg_outfile.close()
    
    if save_output is True:
        coreg_outfile.close()
        
    return all_matches, coreg_outpath

#%%
def plot_matched_rois_by_file(all_matches, params_thr, savefig_dir, idxs_to_keep=None):
    # TODO:  Add 3D compatibility...
    if not os.path.exists(savefig_dir):
        os.makedirs(savefig_dir)
    
    src_nmf_dir = os.path.split(params_thr['ref_filepath'])[0]
    source_nmf_paths = sorted([os.path.join(src_nmf_dir, n) for n in os.listdir(src_nmf_dir) if n.endswith('npz')], key=natural_keys) # Load nmf files
        
    # Load reference:
    ref_file = str(params_thr['ref_filename'])
    ref = np.load(params_thr['ref_filepath'])
    nr = ref['A'].all().shape[1]
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
        A1 = A1[:, idx_components]
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
            print("Plotting %i out of %i components." % (len(idx_components), nr))
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

#%%            
def plot_coregistered_rois(matchedROIs, params_thr, src_filepaths, save_dir, idxs_to_keep=None, cmap='jet', plot_by_file=True):
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
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
    pl.figure(figsize=(12, 10)) 
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1]) 
    gs.update(wspace=0.05, hspace=0.05)
    ax1 = pl.subplot(gs[0])
    
    ax1.imshow(refimg, cmap='gray')
    pl.axis('equal')
    pl.axis('off')
    
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
            # compute the cumulative sum of the energy of the Ath component that 
            # has been ordered from least to highest:
            indx = np.argsort(A[:,roi], axis=None)[::-1]
            cumEn = np.cumsum(A[:,roi].flatten()[indx]**2)
            cumEn /= cumEn[-1] # normalize
            Bvec = np.zeros(d1*d2)
            Bvec[indx] = cumEn
            Bmat = np.reshape(Bvec, (d1,d2), order='F')
            if plot_by_file is True:
                cs = pl.contour(y, x, Bmat, [0.9], colors=[colorvals[fidx]]) #, cmap=colormap)
            else:
                cs = pl.contour(y, x, Bmat, [0.9], colors=[colorvals[ridx]]) #, cmap=colormap)

    nfiles = len(matchedROIs.keys())
    nrois = len(ref_rois)
    print "N files:", nfiles
    print "N rois:", nrois

    #%
    # PLOT labels and legend:
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
    
#%%
def coregister_rois_nmf(params_thr, coreg_output_dir, excluded_tiffs=[], idxs_to_keep=None):
    
    ref_rois = []
    
    if not os.path.exists(coreg_output_dir):
        os.makedirs(coreg_output_dir)
    
    # Get matches:
    all_matches, coreg_results_path = find_matches_nmf(params_thr, coreg_output_dir, idxs_to_keep=idxs_to_keep, save_output=True)
    
    # Plot matches over reference:
    coreg_figdir = os.path.join(coreg_output_dir, 'figures', 'files')
    plot_matched_rois_by_file(all_matches, params_thr, coreg_figdir, idxs_to_keep=idxs_to_keep)

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
    parser.add_option('-M', '--mcmetric', action="store",
                  dest="mcmetric", default='zproj_corrcoefs', help="Motion-correction metric to use for identifying tiffs to exclude [default: zproj_corrcoefs]")

    parser.add_option('--good', action="store_true",
                      dest="keep_good_rois", default=False, help="Set flag to only keep good components (useful for avoiding computing massive ROI sets)")
    parser.add_option('--max', action="store_true",
                      dest="use_max_nrois", default=False, help="Set flag to use file with max N components (instead of reference file) [default uses reference]")
    parser.add_option('--roipath', action="store",
                      dest="roipath", default="", help="If keep_good_rois is True, path to .json with ROI idxs for each file (if using cNMF). Default uses nmf-extraction ROIs.")
    parser.add_option('-O', '--outdir', action="store",
                      dest="coreg_output_dir", default=None, help="Output dir to save coreg results to. Default uses curr ROI dir + 'coreg_results'")
    
    parser.add_option('-f', '--ref', action="store",
                      dest="coreg_fidx", default=1, help="Reference file for coregistration if use_max_nrois==False [default: 1]")


    (options, args) = parser.parse_args(options)
    
    if options.slurm is True:
        if 'coxfs01' not in options.rootdir:
            options.rootdir = '/n/coxfs01/2p-data'
    
    # Set USER INPUT options:
    rootdir = options.rootdir
    animalid = options.animalid
    session = options.session
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
    mcmetric = options.mcmetric
    
    #%%
    # =========================================================================
    # Load ROI evaluation results, if relevant:
    # =========================================================================
    if len(roipath) == 0:
        idxs_to_keep = None
        nrois_total = None
        evalparams = None
    else:
        try:
            idxs_to_keep = dict()
            nrois_total = dict()
            evalparams = dict()
            print "Loaded ROI info for files:",
            print roipath
            src_eval = h5py.File(roipath, 'r')
            for f in src_eval.keys():
                print "%s: %i rois" % (f, len(src_eval[f]['pass_rois']))
                idxs_to_keep[str(f)] = np.array(src_eval[f]['pass_rois'])
                nrois_total[str(f)] = int(len(src_eval[f]['pass_rois']) + len(src_eval[f]['fail_rois']))
                print "idxs:", idxs_to_keep[str(f)]
                print "ntotal:", nrois_total[str(f)]
                
            for eparam in src_eval.attrs.keys():
                if eparam == 'creation_date':
                    continue
                if isinstance(src_eval.attrs[eparam], np.ndarray):
                    evalparams[eparam] = src_eval.attrs[eparam].tolist()
                else:
                    evalparams[eparam] = src_eval.attrs[eparam].item()
                #print eparam, src_eval.attrs[eparam], src_eval.attrs[eparam].dtype
                evalparams[eparam] = src_eval.attrs[eparam].tolist()
        except Exception as e:
            print "ERROR LOADING ROI idxs ------------------------------------"
            print traceback.print_exc()
            print "User provided ROI idx path:"
            print roipath
            print "-----------------------------------------------------------"
        finally:
            src_eval.close()
            
    coreg_output_dir = options.coreg_output_dir
    
    coreg_fidx = int(options.coreg_fidx) - 1
    reference_filename = "File%03d" % int(options.coreg_fidx)
    
    #%%
    # =========================================================================
    # Set Coregistration parameters:
    # =========================================================================   
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
    
    #%%    
    # =========================================================================
    # Load specified ROI-ID parameter set:
    # =========================================================================
    session_dir = os.path.join(rootdir, animalid, session)

    try:
        RID = load_RID(session_dir, roi_id)
        print "Coregistering ROIs from set: %s" % RID['roi_id']
    except Exception as e:
        print "-- ERROR: unable to open source ROI dict. ---------------------"
        traceback.print_exc()
        print "---------------------------------------------------------------"
    
    #%%
    # =========================================================================
    # Get info for ROI source files and TIFF/mmapped source files:
    # =========================================================================
    tiff_sourcedir = RID['SRC']
    path_parts = tiff_sourcedir.split(session_dir)[-1].split('/')
    acquisition = path_parts[1]
    run = path_parts[2]
    process_dirname = path_parts[4]
    process_id = process_dirname.split('_')[0]
    
    roi_source_paths, tiff_source_paths, filenames, excluded_tiffs, mcmetrics_path = get_source_paths(session_dir, RID, check_motion=True, 
                                                                                                      mcmetric=mcmetric,
                                                                                                      acquisition=acquisition,
                                                                                                      run=run,
                                                                                                      process_id=process_id)
    if len(exclude_manual) > 0:
        excluded_tiffs.extend(exclude_manual)
        
    print "Additionally excluding manully-selected tiffs."
    print "Excluded:", excluded_tiffs
    
    params_thr['excluded_tiffs'] = excluded_tiffs
    
    roi_ref_type = RID['PARAMS']['options']['source']['roi_type']
    roi_source_dir = RID['PARAMS']['options']['source']['roi_dir'] 
   
    
    # =========================================================================
    # Create output dir:
    # =========================================================================
    if coreg_output_dir is None:
        coreg_output_dir = os.path.join(RID['DST'], 'coreg_results')            
    print "Saving COREG results to:", coreg_output_dir
    if not os.path.exists(coreg_output_dir):
        os.makedirs(coreg_output_dir)
        
    #%%
    # =========================================================================
    # Determine which file should be used as "reference" for coregistering ROIs:
    # =========================================================================
    
    if idxs_to_keep is not None:
        src_nrois = [(str(fkey), nrois_total[fkey], len(idxs_to_keep[fkey])) for fkey in sorted(idxs_to_keep.keys(), key=natural_keys)]    
        
    else:
        if roi_ref_type == 'caiman2D':
            # Go through all files to select the one that has the MOST number of ROIs:
            src_nrois = []
            for roi_source in roi_source_paths:
                snmf = np.load(roi_source)
                fname = re.search('File(\d{3})', roi_source).group(0)
                nall = snmf['A'].all().shape[1]
                npass = len(snmf['idx_components'])
                src_nrois.append((str(fname), nall, npass))
    
    if use_max_nrois is True:
        # Either select the file that has the MAX number of "good" ROIs:
        if keep_good_rois is True:
            nmax_idx = [s[2] for s in src_nrois].index(max([s[2] for s in src_nrois]))
            nrois_max = src_nrois[nmax_idx][2]
        else:
            # ... or, select file that has the MAX number of ROIs total:
            nmax_idx = [s[1] for s in src_nrois].index(max([s[1] for s in src_nrois]))
            nrois_max = src_nrois[nmax_idx][1]
        params_thr['ref_filename'] = native(src_nrois[nmax_idx][0])
        params_thr['ref_filepath'] = roi_source_paths[nmax_idx]
    else:
        # Use a reference file (either MC reference or default, File001):
        print "Using reference:", reference_filename
        params_thr['ref_filename'] = reference_filename
        params_thr['ref_filepath'] = roi_source_paths[int(coreg_fidx)]
        if keep_good_rois is True:
            nrois_max = src_nrois[coreg_fidx][2]
        else:
            nrois_max = src_nrois[coreg_fidx][1]
    print "Using source %s as reference. Max N rois: %i" % (params_thr['ref_filename'], nrois_max)
    
    # Show evaluation params, if filtered:
    if keep_good_rois is True:
        if evalparams is None:
            with open(os.path.join(roi_source_dir, 'roiparams.json'), 'r') as f:
                src_roiparams = json.load(f)
            evalparams = src_roiparams['eval']
        print "-----------------------------------------------------------"
        print "Coregistering ROIS from source..."
        print "Source ROIs were filtered with the following eval params:"
        for eparam in evalparams.keys():
            print eparam, evalparams[eparam]
        print "-----------------------------------------------------------"
    
    params_thr['eval'] = evalparams
    
    #%%
    # =========================================================================
    # Save coreg params info to current coreg dir:
    # =========================================================================
    pp.pprint(params_thr)
    with open(os.path.join(coreg_output_dir, 'coreg_params.json'), 'w') as f:
        json.dump(params_thr, f, indent=4, sort_keys=True)
        
    #%%
    # =========================================================================
    # COREGISTER ROIs:
    # =========================================================================
    ref_rois, coreg_results_path = coregister_rois_nmf(params_thr, coreg_output_dir, excluded_tiffs=excluded_tiffs, idxs_to_keep=idxs_to_keep)
    print("Found %i common ROIs matching reference." % len(ref_rois))

    #%%
    # =========================================================================
    # Identify and save universal matches:
    # =========================================================================
    #% Re-load coregistered roi matches and save universal matches:
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
    
    #% Save ROI idxs of unviersal matches:
    matchedrois_fn_base = 'coregistered_r%s' % str(params_thr['ref_filename'])
    print("Saving matches to: %s" % os.path.join(coreg_output_dir, matchedrois_fn_base))
    with open(os.path.join(coreg_output_dir, '%s.json' % matchedrois_fn_base), 'w') as f:
        json.dump(matchedROIs, f, indent=4, sort_keys=True)
    
    # Save plots of universal matches:
    coreg_fig_dir = os.path.join(coreg_output_dir, 'figures')
    if len(ref_rois) > 0:
        plot_coregistered_rois(matchedROIs, params_thr, roi_source_paths, coreg_fig_dir, idxs_to_keep=idxs_to_keep, plot_by_file=True)
        plot_coregistered_rois(matchedROIs, params_thr, roi_source_paths, coreg_fig_dir, idxs_to_keep=idxs_to_keep, plot_by_file=False)
    
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
