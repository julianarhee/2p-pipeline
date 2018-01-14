#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Select from a set of methods for ROI extraction.
Currently supports:
    - manual2D_circle
    - manual2D_square
    - manual2D_polygon
    - caiman2D
    - blob_detector

User-provided option with specified ROI ID (e.g., 'rois002') OR roi_type.
If <roi_type> is provided, RID set is created with set_roid_params.py.
ROI ID will always take precedent.

INPUTS:
    rootdir
    animalid
    session
    (roi_id)
    (roi params)
    
OUTPUTS:
    <path/to/roi/id>/masks.hdf5
        masks :  hdf5 file with a group for each file source from which ROIs are extracted
            masks['File001']['com'] = hdf5 dataset of array of coords for each roi's center-of-mass
            masks['File001']['masks'] = hdf5 dataset of array containg masks, size is d1xd2(xd3?) x nrois
            masks['File001'].attrs['source_file'] :  
        
        masks has attributes:
            masks.attrs['roi_id']             :  name identifier for ROI ID set
            masks.attrs['rid_hash']           :  hash identifier for ROI ID set
            masks.attrs['keep_good_rois']     :  whether to keep subset of ROIs that pass some "roi evalation" threshold (currently, only for caiman2D)
            masks.attrs['ntiffs_in_set']      :  number of tiffs included in current ROI set (excludes bad files)
            masks.attrs['mcmetrics_filepath'] :  full path to mc_metrics.hdf5 file (if motion-corrected tiff source is used for ROI extraction)
            masks.attrs['mcmetric_type']      :  metric used to determine bad files (i.e., 'excluded_tiffs')
            masks.attrs['creation_date']      :  date string created, format 'YYYY-MM-DD hh:mm:ss'
    
    <path/to/roi/id>/roiparams.json
        - info about extracted roi set
        - evaluation params
        - excluded tiffs
        - whether ROIs were filtered by some metric
        - coregisteration parms, if applicable


Created on Thu Jan  4 11:54:38 2018
@author: julianarhee
"""
import matplotlib
matplotlib.use('Agg')
import os
import h5py
import json
import re
import datetime
import optparse
import pprint
import time
import scipy
import tifffile as tf
import pylab as pl
import numpy as np
from pipeline.python.utils import natural_keys, hash_file, write_dict_to_json
from pipeline.python.rois import extract_rois_caiman as rcm
from pipeline.python.set_roi_params import post_rid_cleanup
from pipeline.python.evaluate_motion_correction import get_source_info
from caiman.components_evaluation import evaluate_components, estimate_components_quality_auto
from caiman.utils.visualization import plot_contours
import caiman as cm
from matplotlib import gridspec

from scipy.sparse import spdiags, issparse
from caiman.utils.visualization import get_contours
from past.utils import old_div
from scipy import ndimage

def timer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    formatted_time = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)
    return formatted_time
   
pp = pprint.PrettyPrinter(indent=4)

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
def find_matches_nmf(params_thr, output_dir, idxs_to_keep=None, save_output=True):
     # TODO:  Add 3D compatibility...
    if save_output is True:
        coreg_outpath = os.path.join(output_dir, 'coreg_results.h5py')
        coreg_outfile = h5py.File(coreg_outpath, 'w')
        for k in params_thr.keys():
            coreg_outfile.attrs[k] = params_thr[k]
        coreg_outfile.attrs['creation_date'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    output_dir_figs = os.path.join(output_dir, 'figures')
    if not os.path.exists(output_dir_figs):
        os.makedirs(output_dir_figs)
        
    all_matches = dict()
    ref_file = str(params_thr['coreg_ref_file'])
    try:
        # Load reference file info:
        ref = np.load(params_thr['coreg_ref_path'])
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
        nmf_src_dir = os.path.split(params_thr['coreg_ref_path'])[0]
        nmf_fns = [n for n in os.listdir(nmf_src_dir) if n.endswith('npz')]
        for nmf_fn in nmf_fns:
            
            curr_file = str(re.search('File(\d{3})', nmf_fn).group(0))

            if nmf_fn == os.path.basename(params_thr['coreg_ref_path']):
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
                d.attrs['dims'] = dims
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
        match_fn_base = 'matches_byfile_r%s' % str(params_thr['coreg_ref_file'])
        with open(os.path.join(output_dir, '%s.json' % match_fn_base), 'w') as f:
            json.dump(all_matches, f, indent=4, sort_keys=True)
         
    except Exception as e:
        print e
        coreg_outfile.close()
    
    if save_output is True:
        coreg_outfile.close()
        
    return all_matches

#%%
def plot_matched_rois(all_matches, params_thr, savefig_dir, idxs_to_keep=None):
    # TODO:  Add 3D compatibility...
    if not os.path.exists(savefig_dir):
        os.makedirs(savefig_dir)
        
    # Load reference:
    ref_file = str(params_thr['coreg_ref_file'])
    ref = np.load(params_thr['coreg_ref_path'])
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

        if curr_file==params_thr['coreg_ref_file']:
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

#%%
def coregister_rois_nmf(params_thr, coreg_output_dir, excluded_tiffs=[], idxs_to_keep=None):
    
    ref_rois = None
    
    if not os.path.exists(coreg_output_dir):
        os.makedirs(coreg_output_dir)
    
    # Get matches:
    all_matches = find_matches_nmf(params_thr, coreg_output_dir, idxs_to_keep=idxs_to_keep, save_output=True)
    
    # Plot matches over reference:
    coreg_figdir = os.path.join(coreg_output_dir, 'figures')
    plot_matched_rois(all_matches, params_thr, coreg_figdir, idxs_to_keep=idxs_to_keep)

    #% Find intersection of all matches with reference:
    filenames = all_matches.keys()
    filenames.extend([str(params_thr['coreg_ref_file'])])
    filenames = sorted(filenames, key=natural_keys)

    ref_idxs = [[comp[0] for comp in all_matches[f]] for f in all_matches.keys() if f not in excluded_tiffs]
    #file_match_max = [len(r) for r in ref_idxs].index(max(len(r) for r in ref_idxs))
    
    ref_rois = set(ref_idxs[0])
    for s in ref_idxs[1:]:
        ref_rois.intersection_update(s)
    ref_rois = list(ref_rois)
    
    return ref_rois
                    

#%%
def evaluate_rois_nmf(mmap_path, nmfout_path, evalparams, dview=None, eval_outdir='', save_output=True):
    
    eval_outdir_figs = os.path.join(eval_outdir, 'figures')
    if not os.path.exists(eval_outdir_figs):
        os.makedirs(eval_outdir_figs)
        
    curr_file = str(re.search('File(\d{3})', nmfout_path).group(0))
    
    nmf = np.load(nmfout_path)
    
    # Evaluate components and save output:
    final_frate = evalparams['final_frate']
    rval_thr = evalparams['rval_thr']       # accept components with space corr threshold or higher
    decay_time = evalparams['decay_time']   # length of typical transient (sec)
    use_cnn = evalparams['use_cnn']         # CNN classifier designed for 2d data ?
    min_SNR = evalparams['min_SNR']         # accept components with peak-SNR of this or higher
    gSig = [int(evalparams['gSig']), int(evalparams['gSig'])]
    
    Yr, dims, T = cm.load_memmap(mmap_path)
    images = np.reshape(Yr.T, [T] + list(dims), order='F')

    idx_components, idx_components_bad, SNR_comp, r_values, cnn_preds = \
        estimate_components_quality_auto(images, nmf['A'].all(), nmf['C'], nmf['b'], nmf['f'], nmf['YrA'],
                                         final_frate, decay_time,
                                         gSig, nmf['dims'], 
                                         dview=dview,
                                         min_SNR=min_SNR, 
                                         r_values_min=rval_thr,
                                         use_cnn=use_cnn)
    
    print "%s: evalulation results..." % curr_file
    print(('Should keep ' + str(len(idx_components)) +
       ' and discard  ' + str(len(idx_components_bad))))
    
    # Save evaluation output:
    try:
        if save_output is True:
            eval_outfile_path = os.path.join(eval_outdir, 'eval_results.hdf5')
            eval_outfile = h5py.File(eval_outfile_path, 'w')
            if len(eval_outfile.attrs.keys()) == 0:
                for k in evalparams.keys():
                    eval_outfile.attrs[k] = evalparams[k]
                eval_outfile.attrs['creation_date'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
            if curr_file not in eval_outfile.keys():
                file_grp = eval_outfile.create_group(curr_file)
                file_grp.attrs['source'] = nmfout_path
            else:
                file_grp = eval_outfile[curr_file]
            
            kept = file_grp.create_dataset('kept_components', idx_components.shape, idx_components.dtype)
            kept[...] = idx_components
            
            bad = file_grp.create_dataset('bad_components', idx_components_bad.shape, idx_components_bad.dtype)
            bad[...] = idx_components_bad
            
            snr = file_grp.create_dataset('SNR', SNR_comp.shape, SNR_comp.dtype)
            snr[...] = SNR_comp
            
            rvals = file_grp.create_dataset('r_values', r_values.shape, r_values.dtype)
            rvals[...] = r_values
            
            eval_outfile.close()
    except Exception as e:
        print e
        eval_outfile.close()
        
    #% PLOT: Iteration 2 - Visualize Spatial and Temporal component evaluation ----------
        
    pl.figure(figsize=(5,15))
    pl.subplot(2,1,1); pl.title('r values (spatial)'); pl.plot(r_values); pl.plot(range(len(r_values)), np.ones(r_values.shape)*rval_thr, 'r')
    pl.subplot(2,1,2); pl.title('SNR_comp'); pl.plot(SNR_comp); pl.plot(range(len(SNR_comp)), np.ones(r_values.shape)*min_SNR, 'r')
    pl.xlabel('roi')
    pl.suptitle(curr_file)
    pl.savefig(os.path.join(eval_outdir_figs, 'eval_results_%s.png' % curr_file))
    pl.close()
    # -----------------------------------------------------------------------------------
    
    
    # PLOT: Iteration 2 - Show components that pass/fail evaluation metric --------------
    pl.figure();
    pl.subplot(1,2,1); pl.title('pass'); plot_contours(nmf['A'].all()[:, idx_components], nmf['Av'], thr=0.85); pl.axis('off')
    pl.subplot(1,2,2); pl.title('fail'); plot_contours(nmf['A'].all()[:, idx_components_bad], nmf['Av'], thr=0.85); pl.axis('off')
    pl.savefig(os.path.join(eval_outdir_figs, 'contours_%s.png' % curr_file))
    pl.close()
    # -----------------------------------------------------------------------------------

    return idx_components, idx_components_bad, SNR_comp, r_values

#%%
def plot_coregistered_rois(matchedROIs, params_thr, src_filepaths, save_dir, idxs_to_keep=None, cmap='jet', plot_by_file=True):
    
    # Load ref img:
    ref_fn = [f for f in source_nmf_paths if str(params_thr['coreg_ref_file']) in f and f.endswith('npz')][0]
    ref = np.load(ref_fn)
    refimg = ref['Av']
    
    colormap = pl.get_cmap(cmap)
    ref_rois = matchedROIs[params_thr['coreg_ref_file']]
    nrois = len(ref_rois)
    print "Plotting %i coregistered ROIs from each file..." % nrois
    
    file_names = matchedROIs.keys();
    if plot_by_file is True:
        plot_type = 'byfile'
        colorvals = colormap(np.linspace(0, 1, len(file_names))) #get_spaced_colors(nrois)
    else:
        plot_type = 'byroi'
        colorvals = colormap(np.linspace(0, 1, len(ref_rois))) #get_spaced_colors(nrois)
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
    #pl.savefig(os.path.join(save_dir, 'contours_%s_r%s_rois.png' % (plot_type,  str(params_thr['coreg_ref_file']))))
    
    nfiles = len(matchedROIs.keys())
    print "N files:", nfiles
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
        for ridx,roi in enumerate(ref_rois):
            ax2.plot(1, interval[ridx], c=colorvals[ridx], marker='.', markersize=20)
            pl.text(1.1, interval[ridx], str(roi), fontsize=12)
            pl.xlim([0.95, 2])
    pl.axis('equal')
    pl.axis('off')
    
    #%
    pl.savefig(os.path.join(save_dir, 'contours_%s_r%s.png' % (plot_type,  str(params_thr['coreg_ref_file'])))) #matchedrois_fn_base))
    pl.close()
            
#%%

#rootdir = '/nas/volume1/2photon/data'
#animalid = 'CE059' #'JR063'
#session = '20171009_CE059' #'20171202_JR063'
#acquisition = 'FOV1_zoom3x' #'FOV1_zoom1x_volume'
#run = 'gratings_phasemod' #'scenes'
#slurm = False
#
#trace_id = 'traces001'
#auto = False
#
#if slurm is True:
#    if 'coxfs01' not in rootdir:
#        rootdir = '/n/coxfs01/2p-data'

#%%

parser = optparse.OptionParser()

# PATH opts:
parser.add_option('-R', '--root', action='store', dest='rootdir', default='/nas/volume1/2photon/data', help='data root dir (root project dir containing all animalids) [default: /nas/volume1/2photon/data, /n/coxfs01/2pdata if --slurm]')
parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')
parser.add_option('-S', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID')
#parser.add_option('-A', '--acq', action='store', dest='acquisition', default='FOV1', help="acquisition folder (ex: 'FOV1_zoom3x') [default: FOV1]")
#parser.add_option('-r', '--run', action='store', dest='run', default='', help="name of run dir containing tiffs to be processed (ex: gratings_phasemod_run1)")
parser.add_option('--default', action='store_true', dest='default', default='store_false', help="Use all DEFAULT params, for params not specified by user (no interactive)")
parser.add_option('--slurm', action='store_true', dest='slurm', default=False, help="set if running as SLURM job on Odyssey")
parser.add_option('-r', '--roi-id', action='store', dest='roi_id', default='', help="ROI ID for rid param set to use (created with set_roi_params.py, e.g., rois001, rois005, etc.)")

# Coregistration options:
parser.add_option('-t', '--maxthr', action='store', dest='dist_maxthr', default=0.1, help="[coreg]: threshold for turning spatial components into binary masks [default: 0.1]")
parser.add_option('-n', '--power', action='store', dest='dist_exp', default=0.1, help="[coreg]: power n for distance between masked components: dist = 1 - (and(M1,M2)/or(M1,M2)**n [default: 1]")
parser.add_option('-d', '--dist', action='store', dest='dist_thr', default=0.5, help="[coreg]: threshold for setting a distance to infinity, i.e., illegal matches [default: 0.5]")
parser.add_option('-o', '--overlap', action='store', dest='dist_overlap_thr', default=0.8, help="[coreg]: overlap threshold for detecting if one ROI is subset of another [default: 0.8]")

(options, args) = parser.parse_args()

# Set USER INPUT options:
rootdir = options.rootdir
animalid = options.animalid
session = options.session
#acquisition = options.acquisition
#run = options.run
roi_id = options.roi_id
slurm = options.slurm
auto = options.default

if slurm is True:
    if 'coxfs01' not in rootdir:
        rootdir = '/n/coxfs01/2p-data'

#%%
rootdir = '/nas/volume1/2photon/data'
animalid = 'JR063' #'JR063'
session = '20171128_JR063' #'20171128_JR063'
roi_id = 'rois002'
slurm = False
auto = False

keep_good_rois = True       # Only keep "good" ROIs from a given set (TODO:  add eval for ROIs -- right now, only have eval for NMF and coregister)

# COREG-SPECIFIC opts:
use_max_nrois = True        # Use file which has the max N ROIs as reference (alternative is to use reference file)
dist_maxthr = 0.1
dist_exp = 0.1
dist_thr = 0.5
dist_overlap_thr = 0.8

#%%
# =============================================================================
# Load specified trace-ID parameter set:
# =============================================================================
session_dir = os.path.join(rootdir, animalid, session)
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
    print e
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
    except Exception as E:
        print "---------------------------------------------------------------"
        print "No tmp roi-ids found either... ABORTING with error:"
        print e
        print "---------------------------------------------------------------"

#%%
# =============================================================================
# Get meta info for current run and source tiffs using trace-ID params:
# =============================================================================
rid_hash = RID['rid_hash']
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
        print e
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

print "Motion-correction info:"
print "MC reference is %s, %s." % (mc_ref_file, mc_ref_channel)
print "Found %i tiff files to exclude based on MC EVAL: %s." % (len(excluded_tiffs), mcmetric_type)
print "======================================================================="


#%%
# =============================================================================
# Extract ROIs using specified method:
# =============================================================================
print "Extracting ROIs...====================================================="

format_roi_output = False
t_start = time.time()

if roi_type == 'caiman2D':
    #%
    roi_opts = ['-R', rootdir, '-i', animalid, '-S', session, '-A', acquisition, '-r', run, '-p', rid_hash]
    if slurm is True:
        roi_opts.extend(['--slurm'])
    if len(exclude_str) > 0:
        roi_opts.extend(['-x', exclude_str])
        
    nmf_hash, rid_hash = rcm.extract_cnmf_rois(roi_opts)
    
    # Clean up tmp RID files:
    session_dir = os.path.join(rootdir, animalid, session)
    post_rid_cleanup(session_dir, rid_hash)
    
    format_roi_output = True
    #%
    
elif roi_type == 'blob_detector':
    #% Do some other stuff
    print "blobs"
    format_roi_output = False

elif 'manual' in roi_type:
    # Do some matlab-loading stuff ?
    print "manual"
    format_roi_output = False

elif roi_type == 'coregister':
    #%
    params_thr = dict()
    params_thr['keep_good_rois'] = keep_good_rois
    
    params_thr['dist_maxthr'] = dist_maxthr            # threshold for turning spatial components into binary masks (default: 0.1)
    params_thr['dist_exp'] = dist_exp                  # power n for distance between masked components: dist = 1 - (and(m1,m2)/or(m1,m2))^n (default: 1)
    params_thr['dist_thr'] = dist_thr                  # threshold for setting a distance to infinity. (default: 0.5)
    params_thr['dist_overlap_thr'] = dist_overlap_thr  # overlap threshold for detecting if one ROI is a subset of another (default: 0.8)
    if use_max_nrois is True:
        params_thr['filter_type'] = 'max'
    else:
        params_thr['filter_type'] = 'ref'
    
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
            params_thr['coreg_ref_file'] = src_nrois[nmax_idx][0]
            params_thr['coreg_ref_path'] = source_nmf_paths[nmax_idx]
            print "Using source %s as reference. Max N rois: %i" % (params_thr['coreg_ref_file'], nrois_max)
        else:
            params_thr['coreg_ref_file'] = mc_ref_file
            params_thr['coreg_ref_path'] = source_nmf_paths.index([i for i in source_nmf_paths if mc_ref_file in i][0])
        
        #% Evaluate?
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
            
            src_rid = roidict[RID['PARAMS']['options']['roi_source']]
            evalparams = src_evalparams.copy()
            evalparams['gSig'] = src_rid['PARAMS']['options']['extraction']['gSig'][0]
            idxs_to_keep = None # Set to None, since if first eval (during nmf extraction) is good, no need to provide new/alt roi idxs
            
        #% Coregister ROIs using specified reference:
            
        coreg_output_dir = os.path.join(RID['DST'], 'src_coreg_output')
        rois_to_keep = coregister_rois_nmf(params_thr, coreg_output_dir, excluded_tiffs=excluded_tiffs)
        print("Found %i common ROIs matching reference." % len(rois_to_keep))
        
        # Save info to current coreg dir:
        params_thr['eval'] = evalparams
        with open(os.path.join(coreg_output_dir, 'coreg_params.json'), 'w') as f:
            json.dump(params_thr, f, indent=4, sort_keys=True)
            
        #% Re-evaluate ROIs to less stringest thresholds?
        if len(rois_to_keep) == 0:
                        
            print "Evaluating NMF components with less stringent eval params..."
            roi_eval_outdir = os.path.join(RID['DST'], 'src_evaluation')
            if not os.path.exists(roi_eval_outdir):
                os.makedirs(roi_eval_outdir)
            
            # ================================================================
            evalparams['min_SNR'] = 1.5
            evalparams['rval_thr'] = 0.7
            # ================================================================

            src_mmap_dir = src_rid['PARAMS']['mmap_source']
            mem_paths = sorted([os.path.join(src_mmap_dir, f) for f in os.listdir(src_mmap_dir) if f.endswith('mmap')], key=natural_keys)
            src_file_list = []
            for fn in filenames:
                match_nmf = [f for f in source_nmf_paths if fn in f][0]
                match_mmap = [f for f in mem_paths if fn in f][0]
                src_file_list.append((match_mmap, match_nmf))
                
            #%start a cluster for parallel processing
            try:
                dview.terminate() # stop it if it was running
            except:
                pass
            
            c, dview, n_processes = cm.cluster.setup_cluster(backend='local', # use this one
                                                             n_processes=None,  # number of process to use, reduce if out of mem
                                                             single_thread = False)
            idxs_to_keep = dict()
            for src_file in src_file_list:
                curr_mmap_path = src_file[0]
                curr_nmfout_path = src_file[1]
                curr_file = str(re.search('File(\d{3})', curr_nmfout_path).group(0))
                
                good, bad, snr_vals, r_vals = evaluate_rois_nmf(curr_mmap_path, curr_nmfout_path, 
                                                                      evalparams, dview=dview,
                                                                      eval_outdir=roi_eval_outdir, save_output=True)
                idxs_to_keep[curr_file] = good
                
            #% Try finding NEW matches:
            coreg_output_dir = os.path.join(RID['DST'], 'reeval_coreg_output')
            ref_rois = coregister_rois_nmf(params_thr, coreg_output_dir, excluded_tiffs=excluded_tiffs, idxs_to_keep=idxs_to_keep)
            print("Found %i common ROIs matching reference." % len(ref_rois))
            
            # Save info to current coreg dir:
            params_thr['eval'] = evalparams
            with open(os.path.join(coreg_output_dir, 'coreg_params.json'), 'w') as f:
                json.dump(params_thr, f, indent=4, sort_keys=True)
            
        #% Load coregistered roi matches and save universal matches:
        coreg_info = h5py.File(os.path.join(coreg_output_dir, 'coreg_results.h5py'), 'r')
        filenames = [str(i) for i in coreg_info.keys()]
        filenames.append(str(params_thr['coreg_ref_file']))
        filenames = sorted(filenames, key=natural_keys)
        
        # Save ROI idxs for each file that matches ref and is common to all:
        matchedROIs = dict()
        for curr_file in filenames: #all_matches.keys():
            print curr_file
            if curr_file in excluded_tiffs:
                continue
            if curr_file==str(params_thr['coreg_ref_file']):
                matchedROIs[curr_file] = ref_rois
            else:
                curr_matches = coreg_info[curr_file]['matches']
                matchedROIs[curr_file] = [curr_matches[[i[0] for i in curr_matches].index(r)][1] for r in ref_rois]
        
        coreg_info.close()
        
        # Save ROI idxs of unviersal matches:
        matchedrois_fn_base = 'coregistered_r%s' % str(params_thr['coreg_ref_file'])
        print("Saving matches to: %s" % os.path.join(coreg_output_dir, matchedrois_fn_base))
        with open(os.path.join(coreg_output_dir, '%s.json' % matchedrois_fn_base), 'w') as f:
            json.dump(matchedROIs, f, indent=4, sort_keys=True)
        
        # Save plots of universal matches:
        plot_coregistered_rois(matchedROIs, params_thr, source_nmf_paths, coreg_output_dir, idxs_to_keep=idxs_to_keep, plot_by_file=True)
        plot_coregistered_rois(matchedROIs, params_thr, source_nmf_paths, coreg_output_dir, idxs_to_keep=idxs_to_keep, plot_by_file=False)
        
        
    format_roi_output = True
    
    
        #%
else:
    print "ERROR: %s -- roi type not known..." % roi_type

roi_dur = timer(t_start, time.time())
print "RID %s -- Finished ROI extration!" % rid_hash
print "Total duration was:", roi_dur

print "======================================================================="


#%% Save ROI params info:
    
# TODO: Include ROI eval info for other methods?
# TODO: If using NMF eval methods, make func to do evaluation at post-extraction step (since extract_rois_caiman.py keeps all when saving anyway)
roiparams = dict()
rid_dir = RID['DST']

if roi_type == 'caiman2D':
    roiparams['eval'] = RID['PARAMS']['options']['eval']
elif roi_type == 'coregister':
    roiparams['eval'] = evalparams
    
roiparams['keep_good_rois'] = keep_good_rois
roiparams['excluded_tiffs'] = excluded_tiffs
roiparams['roi_type'] = roi_type
roiparams['roi_id'] = roi_id
roiparams['rid_hash'] = rid_hash

roiparams_filepath = os.path.join(rid_dir, 'roiparams.json') # % (str(roi_id), str(rid_hash)))
with open(roiparams_filepath, 'w') as f:
    write_dict_to_json(roiparams, roiparams_filepath)
    
    
#%%
def get_masks_and_coms(nmf_filepath, roiparams, kept_rois=None, coreg_rois=None):
    
    nmf = np.load(nmf_filepath)
    nr = nmf['A'].all().shape[1]
    d1 = int(nmf['dims'][0])
    d2 = int(nmf['dims'][1])
    dims = (d1, d2)
    
    A = nmf['A'].all()
    A2 = A.copy()
    A2.data **= 2
    nA2 = np.sqrt(np.array(A2.sum(axis=0))).squeeze()
    rA = A * spdiags(old_div(1, nA2), 0, nr, nr)
    rA = rA.todense()
    nr = A.shape[0]
    
    # Create masks:
    masks = np.reshape(np.array(rA), (d1, d2, nr), order='F')
    if roiparams['keep_good_rois'] is True:
        if kept_rois is None:
            kept_rois = nmf['idx_components']
        masks = masks[:,:,kept_rois]
    if coreg_rois is not None:
        masks = masks[:,:,coreg_rois]
    
    print("Keeping %i out of %i ROIs." % (len(kept_rois), nr))
    
    # Get center of mass for each ROI:
    coors = get_contours(A, dims, thr=0.9)
    if roiparams['keep_good_rois'] is True:
        if kept_rois is None:
            kept_rois = nmf['idx_components']
        coors = [coors[i] for i in kept_rois]
    if coreg_rois is not None:
        coors = [coors[i] for i in coreg_rois]
        
    cc1 = [[l[0] for l in n['coordinates']] for n in coors]
    cc2 = [[l[1] for l in n['coordinates']] for n in coors]
    coords = [[(x,y) for x,y in zip(cc1[n], cc2[n])] for n in range(len(cc1))] 
    coms = np.array([list(n['CoM']) for n in coords])
    
    return masks, coms

#%%
# =============================================================================
# Format ROI output to standard, if applicable:
# =============================================================================

rid_figdir = os.path.join(rid_dir, 'figures')
if not os.path.exists(rid_figdir):
    os.makedirs(rid_figdir)

mask_filepath = os.path.join(rid_dir, 'masks.hdf5')
maskfile = h5py.File(mask_filepath, 'w')
maskfile.attrs['roi_id'] = roi_id
maskfile.attrs['rid_hash'] = rid_hash
maskfile.attrs['keep_good_rois'] = keep_good_rois
maskfile.attrs['ntiffs_in_set'] = len(filenames)
maskfile.attrs['mcmetrics_filepath'] = mcmetrics_filepath
maskfile.attrs['mcmetric_type'] = mcmetric_type
maskfile.attrs['creation_date'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

if format_roi_output is True :
    if roi_type == 'caiman2D':
        nmf_output_dir = os.path.join(rid_dir, 'nmfoutput')
        all_nmf_fns = sorted([n for n in os.listdir(nmf_output_dir) if n.endswith('npz')], key=natural_keys)
        nmf_fns = []
        for f in filenames:
            nmf_fns.append([m for m in all_nmf_fns if f in m][0])
        assert len(nmf_fns) == len(filenames), "Unable to find matching nmf file for expected files."
        
        for fidx, nmf_fn in enumerate(sorted(nmf_fns, key=natural_keys)):
            print "Creating ROI masks for %s" % filenames[fidx]
            # Create group for current file:
            if filenames[fidx] not in maskfile.keys():
                filegrp = maskfile.create_group(filenames[fidx])
                filegrp.attrs['source_file'] = os.path.join(nmf_output_dir, nmf_fn)
            else:
                filegrp = maskfile[filenames[fidx]]
                
            # Get NMF output info:
            nmf_filepath = os.path.join(nmf_output_dir, nmf_fn)
            nmf = np.load(nmf_filepath)
            img = nmf['Av']
            
            masks, coms = get_masks_and_coms(nmf_filepath, roiparams)
            kept_idxs = nmf['idx_components']
            
            print('Mask array:', masks.shape)
            currmasks = filegrp.create_dataset('masks', masks.shape, masks.dtype)
            currmasks[...] = masks
            if roiparams['keep_good_ros'] is True:
                currmasks.attrs['nrois'] = len(kept_idxs)
                currmasks.attrs['roi_idxs'] = kept_idxs
            else:
                currmasks.attrs['nrois'] = masks.shape[-1]
                    
            currcoms = filegrp.create_dataset('coms', coms.shape, coms.dtype)
            currcoms[...] = coms
            
            # Plot figure with ROI masks:
            vmax = np.percentile(img, 98)
            pl.figure()
            pl.imshow(img, interpolation='None', cmap=pl.cm.gray, vmax=vmax)
            for roi in range(len(kept_idxs)):
                masktmp = masks[:,:,roi]
                msk = masktmp.copy() 
                msk[msk==0] = np.nan
                pl.imshow(msk, interpolation='None', alpha=0.3, cmap=pl.cm.hot)
                [ys, xs] = np.where(masktmp>0)
                pl.text(xs[int(round(len(xs)/4))], ys[int(round(len(ys)/4))], str(kept_idxs[roi]), weight='bold')
                pl.axis('off')
            pl.colorbar()
            pl.tight_layout()
            
            # Save image:
            imname = '%s_%s_%s_masks.png' % (roi_id, rid_hash, filenames[fidx])
            print(imname) 
            pl.savefig(os.path.join(rid_figdir, imname))
            pl.close()
            
    elif roi_type == 'coregister':
            
        roi_ref_type = RID['PARAMS']['options']['source']['roi_type']
        roi_source_dir = RID['PARAMS']['options']['source']['roi_dir']
        
        if roi_ref_type == 'caiman2D':
            src_nmf_dir = os.path.join(roi_source_dir, 'nmfoutput')
            source_nmf_paths = sorted([os.path.join(src_nmf_dir, n) for n in os.listdir(src_nmf_dir) if n.endswith('npz')], key=natural_keys) # Load nmf files
            # Load coregistration info for each file:
            coreg_info = h5py.File(os.path.join(coreg_output_dir, 'coreg_results.h5py'), 'r')
            filenames = [str(i) for i in coreg_info.keys()]
            filenames.append(str(params_thr['coreg_ref_file']))
            filenames = sorted(filenames, key=natural_keys)
            
            # Load universal match info:
            matchedrois_fn_base = 'coregistered_r%s' % str(params_thr['coreg_ref_file'])
            with open(os.path.join(coreg_output_dir, '%s.json' % matchedrois_fn_base), 'r') as f:
                matchedROIs = json.load(f)
            
            idxs_to_keep = dict()
            for curr_file in filenames:
                
                idxs_to_keep[curr_file] = coreg_info[curr_file]['roi_idxs']
                nmf = np.load([n for n in source_nmf_paths if curr_file in n][0])
                img = nmf['Av']
                
                print "Creating ROI masks for %s" % filenames[fidx]
                # Create group for current file:
                if filenames[fidx] not in maskfile.keys():
                    filegrp = maskfile.create_group(curr_file)
                else:
                    filegrp = maskfile[curr_file]
                
                masks, coms = get_masks_and_coms(nmf_filepath, roiparams, kept_rois=idxs_to_keep[curr_file], coreg_rois=matchedROIs[curr_file])
    
                print('Mask array:', masks.shape)
                currmasks = filegrp.create_dataset('masks', masks.shape, masks.dtype)
                currmasks[...] = masks
                if roiparams['keep_good_ros'] is True:
                    currmasks.attrs['nrois'] = len(kept_idxs)
                    currmasks.attrs['roi_idxs'] = kept_idxs
                else:
                    currmasks.attrs['nrois'] = masks.shape[-1]
                        
                currcoms = filegrp.create_dataset('coms', coms.shape, coms.dtype)
                currcoms[...] = coms
                
                zproj = filegrp.create_datatset('avg_img', img.shape, img.dtype)
                zproj[...] = img
                
                # Plot figure with ROI masks:
                vmax = np.percentile(img, 98)
                pl.figure()
                pl.imshow(img, interpolation='None', cmap=pl.cm.gray, vmax=vmax)
                for roi in range(len(kept_idxs)):
                    masktmp = masks[:,:,roi]
                    msk = masktmp.copy() 
                    msk[msk==0] = np.nan
                    pl.imshow(msk, interpolation='None', alpha=0.3, cmap=pl.cm.hot)
                    [ys, xs] = np.where(masktmp>0)
                    pl.text(xs[int(round(len(xs)/4))], ys[int(round(len(ys)/4))], str(kept_idxs[roi]), weight='bold')
                    pl.axis('off')
                pl.colorbar()
                pl.tight_layout()
                
                # Save image:
                imname = '%s_%s_%s_masks.png' % (roi_id, rid_hash, filenames[fidx])
                print(imname) 
                pl.savefig(os.path.join(rid_figdir, imname))
                pl.close()

maskfile.close()


#%%
print "*************************************************"
print "FINISHED EXTRACTING ROIs!"
print "*************************************************"
