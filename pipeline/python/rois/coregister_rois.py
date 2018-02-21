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
import time
import traceback
import numpy as np
import os
import sys
from scipy.sparse import issparse
from matplotlib import gridspec
from pipeline.python.rois.utils import load_RID, get_source_paths, replace_root

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
    """
        params_thr (dict)
            'dist_maxthr'      :  (float) threshold for turning spatial components into binary masks (default: 0.1)
            'dist_exp'         :  (float) power n for distance between masked components: dist = 1 - (and(m1,m2)/or(m1,m2))^n (default: 1)
            'dist_thr'         :  (float) threshold for setting a distance to infinity, i.e., illegal matches (default: 0.5)
            'dist_overlap_thr' :  (float) overlap threshold for detecting if one ROI is a subset of another (default: 0.8)
    """
    d1 = dims[0]
    d2 = dims[1]

    #% first transform A1 and A2 into binary masks
    M1 = np.zeros(A1.shape).astype('bool') #A1.astype('bool').toarray()
    M2 = np.zeros(A2.shape).astype('bool') #A2.astype('bool').toarray()

    K1 = A1.shape[-1]
    K2 = A2.shape[-1]

    s = ndimage.generate_binary_structure(2,2)
    print "Generating binary structure..."
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
    print "Determining distance between REF and FILE"
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

def setup_coreg_params(RID, rootdir=''):
    """
    All parameters for coregistration are collated here:

    Returns:

        params_thr (dict)
            'dist_maxthr'      :  (float) threshold for turning spatial components into binary masks (default: 0.1)
            'dist_exp'         :  (float) power n for distance between masked components: dist = 1 - (and(m1,m2)/or(m1,m2))^n (default: 1)
            'dist_thr'         :  (float) threshold for setting a distance to infinity, i.e., illegal matches (default: 0.5)
            'dist_overlap_thr' :  (float) overlap threshold for detecting if one ROI is a subset of another (default: 0.8)
            'filter_type'      :  (str) options are 'max' (use file with most nrois as reference) or 'ref' (use user-specified reference file, or MC reference file)
            'keep_good_rois'   :  (bool) first filter all ROIs from each file using some evaluation criteria
            'excluded_tiffs'   :  (list) files/tiffs to exclude, includes MC-excluded tiffs and user-selected tiffs to exclude [File001, File002, etc.]
            'ref_filename'     :  (str) file to use as reference (e.g., 'File003')
            'ref_filepath'     :  (str) path to roi source file that will be used as the reference for coreg
            'eval'             :  (dict) evaluation parameters if keep_good_rois = True

        pass_rois_dict (dict)
            key:  FileXXX,  val:  indices of ROIs in file that pass evaluation

        roi_source_paths (list)
            paths to roi source files
    """
    # =========================================================================
    # Set Coregistration parameters:
    # =========================================================================
    keep_good_rois = RID['PARAMS']['options']['keep_good_rois']
    use_max_nrois = RID['PARAMS']['options']['use_max_nrois']


    params_thr = dict()
    # dist_maxthr:      threshold for turning spatial components into binary masks (default: 0.1)
    # dist_exp:         power n for distance between masked components: dist = 1 - (and(m1,m2)/or(m1,m2))^n (default: 1)
    # dist_thr:         threshold for setting a distance to infinity. (default: 0.5)
    # dist_overlap_thr: overlap threshold for detecting if one ROI is a subset of another (default: 0.8)
    params_thr['dist_maxthr'] = RID['PARAMS']['options']['dist_maxthr'] #options.dist_maxthr #0.1
    params_thr['dist_exp'] = RID['PARAMS']['options']['dist_exp'] #options.dist_exp # 1
    params_thr['dist_thr'] = RID['PARAMS']['options']['dist_thr'] #options.dist_thr #0.5
    params_thr['dist_overlap_thr'] = RID['PARAMS']['options']['dist_overlap_thr'] #options.dist_overlap_thr #0.8
    params_thr['keep_good_rois'] = keep_good_rois
    if use_max_nrois is True:
        params_thr['filter_type'] = 'max'
    else:
        params_thr['filter_type'] = 'ref'

    exclude_manual = RID['PARAMS']['eval']['manual_excluded']
    roi_source_dir = RID['PARAMS']['options']['source']['roi_dir']
    session_dir = roi_source_dir.split('/ROIs')[0]

    session = os.path.split(session_dir)[-1]
    animalid = os.path.split(os.path.split(session_dir)[0])[-1]

    # Get ROI source files and their sources (.tif, .mmap) -- use MC evaluation
    # results, if relevant.
    roi_source_paths, tiff_source_paths, filenames, mc_excluded_tiffs, mcmetrics_path = get_source_paths(session_dir, RID, check_motion=True, rootdir=rootdir)

    # Get list of .tif files to exclude (from MC-eval fail or user-choice):
    excluded_tiffs = list(set(exclude_manual + mc_excluded_tiffs))
    print "Additionally excluding manully-selected tiffs:", mc_excluded_tiffs
    print "Excluded:", excluded_tiffs
    params_thr['excluded_tiffs'] = [str(t) for t in excluded_tiffs]

    # Get list of NROIS (all, and "pass" rois) from roi sources -- this is where
    # load_roi_eval is called with eval_key:
    src_nrois, evalparams, pass_rois_dict = get_evaluated_roi_list(RID, roi_source_paths, rootdir=rootdir)

    # Filter source roi list by MAX num or selected REFERENCE:
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
        reference_filename = RID['PARAMS']['options']['reference_filename']
        coreg_fidx = int(RID['PARAMS']['options']['coreg_ridx'])
        print "Using reference: %s" % reference_filename
        params_thr['ref_filename'] = reference_filename
        params_thr['ref_filepath'] = roi_source_paths[coreg_fidx]
        if keep_good_rois is True:
            nrois_max = src_nrois[coreg_fidx][2]
        else:
            nrois_max = src_nrois[coreg_fidx][1]
    print "Using source %s as reference. Max N rois: %i" % (params_thr['ref_filename'], nrois_max)

    # Show evaluation params, if filtered:
    if keep_good_rois is True and evalparams is None:
        if rootdir not in roi_source_dir:
            roi_source_dir = replace_root(roi_source_dir, rootdir, animalid, session)
            print "NEW ROI SRC:", roi_source_dir
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

    return params_thr, pass_rois_dict, roi_source_paths


#%%
def get_evaluated_roi_list(RID, roi_source_paths, rootdir=''):
    """
    This method is currently limited to roi_type='coregister'.
    For a specified RID parameter set that uses an existing ROI set as its source,
    returns a summary list of ROIs that pass the user-specified evaluation criteria.

    Returns:

        src_nrois (list)
            Format is (FileXXX, NTOTAL, NPASS) for each file in ROI source set.

        evalparams (dict)
            key, val pairs are all criteria used in original evaluation.

        pass_rois_dict (dict)
            key: FileXXX, val: indices of ROIs that pass evaluation.

    """
    roi_eval_path = RID['PARAMS']['options']['source']['roi_eval_path']
    if rootdir not in roi_eval_path:
        session_dir = roi_eval_path.split('/ROIs')[0]
        session = os.path.split(session_dir)[-1]
        animalid = os.path.split(os.path.split(session_dir)[0])[-1]
        roi_eval_path = replace_root(roi_eval_path, rootdir, animalid, session)
    if not os.path.exists(roi_eval_path):
        pass_rois_dict = None
        nrois_total = None
        evalparams = None
    else:
        pass_rois_dict, nrois_total, evalparams = load_roi_eval(roi_eval_path)
    print "Loaded EVALPARAMS from roi-eval file: %s" % roi_eval_path
    pp.pprint(evalparams)

    #%
    # =========================================================================
    # Determine which file or ROI-subset should be used as the reference for COREG.
    # =========================================================================
    roi_ref_type = RID['PARAMS']['options']['source']['roi_type']
    if pass_rois_dict is not None:
        # Use user-specified ROI evaluation to get N-pass, N-total for each relevant tiff file:
        src_nrois = [(str(fkey), nrois_total[fkey], len(pass_rois_dict[fkey])) for fkey in sorted(pass_rois_dict.keys(), key=natural_keys)]
    else:
        if roi_ref_type == 'caiman2D':
            # Create a list of N-pass, N-total for each tiff in set:
            src_nrois = []
            for roi_source in roi_source_paths:
                snmf = np.load(roi_source)
                fname = re.search('File(\d{3})', roi_source).group(0)
                nall = snmf['A'].all().shape[1]
                npass = len(snmf['idx_components'])
                src_nrois.append((str(fname), nall, npass))

    return src_nrois, evalparams, pass_rois_dict

#%%
def load_roi_eval(roi_eval_path):
    """
    Loads hdf5 file specified in input roi_eval_path (str).

    Returns:
        pass_rois_dict (dict)
            key: FileXXX, val: indices of ROIs that pass evaluation

        nrois_total (dict)
            key: FileXXX, val: nrois that pass + nrois that fail

        evalparams (dict)
            key, value pairs are all criteria used for evaluation.
    """
    pass_rois_dict = dict()
    nrois_total = dict()
    evalparams = dict()
    try:
        print "Loading ROI info for files:\n",
        src_eval = h5py.File(roi_eval_path, 'r')
        for f in src_eval.keys():
            pass_rois_dict[f] = np.array(src_eval[f]['pass_rois'])
            nrois_total[f] = int(len(src_eval[f]['pass_rois']) + len(src_eval[f]['fail_rois']))
            print "%s: %i out of %i rois passed evaluation." % (f, len(pass_rois_dict[f]), nrois_total[f])

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
        print roi_eval_path
        print "-----------------------------------------------------------"
    finally:
        src_eval.close()

    return pass_rois_dict, nrois_total, evalparams


#%%
def coregister_file_by_rid(tmp_rid_path, filenum=1, nprocs=12, rootdir=''):
    tmp_filepath = None

    # Load tmp rid file for coreg:
    with open(tmp_rid_path, 'r') as f:
        RID = json.load(f)

    # Create dir for coregistration output:
    coreg_output_dir = os.path.join(RID['DST'], 'coreg_results')
    print "Saving COREG results to:", coreg_output_dir
    if not os.path.exists(coreg_output_dir):
        os.makedirs(coreg_output_dir)

    # =========================================================================
    # Set Coregistration parameters:
    # =========================================================================
    params_thr, pass_rois_dict, roi_source_paths = setup_coreg_params(RID, rootdir=rootdir)

    if RID['roi_type'] == 'caiman2D' and not (roi_source_paths[0].endswith('npz')):
        nmf_src_dir = os.path.split(params_thr['ref_filepath'])[0]
        nmf_fns = sorted([n for n in os.listdir(nmf_src_dir) if n.endswith('npz')], key=natural_keys)
        roi_source_paths = sorted([os.path.join(nmf_src_dir, fn) for fn in nmf_fns], key=natural_keys)

    # Save coreg params info to current coreg dir:
    pp.pprint(params_thr)
    with open(os.path.join(coreg_output_dir, 'coreg_params.json'), 'w') as f:
        json.dump(params_thr, f, indent=4, sort_keys=True)

    # Get list of ROIs of keep_good_rois=True and evalulation set used:
    if pass_rois_dict is None:
        filenames = [str(re.search('File(\d{3})', fn).group(0)) for fn in roi_source_paths]
        filenames = sorted([f for f in filenames if f not in params_thr['excluded_tiffs']], key=natural_keys)
        pass_rois_dict = dict((k, None) for k in filenames)

    pass_rois = pass_rois_dict[params_thr['ref_filename']]
    A1, dims, ref_pass_rois, img = load_source_rois(params_thr['ref_filepath'], keep_good_rois=params_thr['keep_good_rois'], pass_rois=pass_rois)
    REF = dict()
    REF['roimat'] = A1
    REF['pass_roi_idxs'] = ref_pass_rois
    REF['dims'] = dims
    REF['img'] = img

    # Then, get matches to sample:
    curr_file = 'File%03d' % filenum
    if curr_file in params_thr['excluded_tiffs']:
        return None

    curr_filepath = [p for p in roi_source_paths if str(re.search('File(\d{3})', p).group(0)) == curr_file][0]
    results = match_file_against_ref(REF, curr_filepath, params_thr, pass_rois_dict=pass_rois_dict, asdict=True)

    # Save tmp results for current file:
    tmpdir = os.path.join(coreg_output_dir, 'tmp')
    if not os.path.exists(tmpdir):
        os.makedirs(tmpdir)
    tmp_filepath = os.path.join(tmpdir, '%s_matches_to_ref.npz' % curr_file)
    np.savez(tmp_filepath,
             distance=results['distance_thr'],
             source=results['source'],
             matches_to_ref=results['matches_to_ref'],
             pass_roi_idxs=results['pass_roi_idxs'],
             img=results['img'],
             A=results['A'],
             dims=results['dims']
             )

#    tmp = h5py.File(tmp_filepath, 'w')
#    dist = tmp.create_dataset('distance_thr', results['distance_thr'].shape, results['distance_thr'].dtype)
#    dist[...] = results['distance_thr']
#    dist.attrs['source'] = results['source']
#    dist.attrs['matches'] = results['matches_to_ref']
#    dist.attrs['pass_roi_idxs'] = results['pass_roi_idxs']

    return tmp_filepath

#%%
def load_source_rois(roi_filepath, keep_good_rois=True, pass_rois=None):

    # First get reference file info:
    ref = np.load(roi_filepath) # np.load(params_thr['ref_filepath'])
    nr = ref['A'].all().shape[1]
    A1 = ref['A'].all()
    dims = ref['dims']
    img = ref['Av']
    if keep_good_rois is True:
        if pass_rois is None:
            pass_rois = ref['idx_components']
        A1 = A1[:, pass_rois]
        print "Loaded SRC rois. Keeping %i out of %i components." % (len(pass_rois), nr)

    return A1, dims, pass_rois, img

#%%
def match_file_against_ref(REF, file_path, params_thr, pass_rois_dict=None, asdict=True):

    # Then, get comparison file:
    curr_file = str(re.search('File(\d{3})', file_path).group(0))
    print "*****CURR FILE: %s*****" % curr_file

    if file_path == params_thr['ref_filepath']: #os.path.basename(params_thr['ref_filepath']):
        print "Skipping REFERENCE."
        pass_roi_idxs = np.array(REF['pass_roi_idxs'])
        D = np.zeros((2,2))
        matches = [[v, v] for v in pass_roi_idxs] #pass_roi_idxs.copy()
        A2 = REF['roimat'].copy()
        img = REF['img'].copy()
        dims = REF['dims']
    else:
        # Assign reference:
        A1 = REF['roimat']
        dims = REF['dims']

        # Load file to match:
        pass_rois = pass_rois_dict[curr_file]
        A2, dims, pass_roi_idxs, img = load_source_rois(file_path, keep_good_rois=params_thr['keep_good_rois'], pass_rois=pass_rois)

        # Calculate distance matrix between ref and all other files:
        print "%s: Calculating DISTANCE MATRIX." % curr_file
        # A1 should always be FIRST:
        D = get_distance_matrix(A1, A2, dims,
                                dist_maxthr=params_thr['dist_maxthr'],
                                dist_exp=params_thr['dist_exp'],
                                dist_overlap_thr=params_thr['dist_overlap_thr'])

        # Set illegal matches (distance vals greater than dist_thr):
        D[D>params_thr['dist_thr']] = np.inf #1E100 #np.nan #1E9

        #% Get matches using thresholds on distance matrix:
        matches = minimumWeightMatching(D)  # Use modified linear_sum_assignment to allow np.inf
        print("Found %i ROI matches in %s" % (len(matches), curr_file))


    if asdict is True:
        results = dict()
        results['source'] = file_path
        results['pass_roi_idxs'] = pass_roi_idxs
        #results['distance'] = fullD
        results['distance_thr'] = D
        results['matches_to_ref'] = np.array(matches)
        results['A'] = A2
        results['dims'] = dims
        results['img'] = img
        return results
    else:
        return D, np.array(matches)


#%%
def find_matches_nmf(RID, coreg_output_dir, rootdir='', nprocs=12):
     # TODO:  Add 3D compatibility...
    coreg_outpath = None

    # Create figure dir:
    output_dir_figs = os.path.join(coreg_output_dir, 'figures', 'files')
    if not os.path.exists(output_dir_figs):
        os.makedirs(output_dir_figs)

    # =========================================================================
    # Set Coregistration parameters:
    # =========================================================================
    params_thr, pass_rois_dict, roi_source_paths = setup_coreg_params(RID, rootdir=rootdir)

    if RID['roi_type'] == 'caiman2D' and not (roi_source_paths[0].endswith('npz')):
        nmf_src_dir = os.path.split(params_thr['ref_filepath'])[0]
        nmf_fns = sorted([n for n in os.listdir(nmf_src_dir) if n.endswith('npz')], key=natural_keys)
        roi_source_paths = sorted([os.path.join(nmf_src_dir, fn) for fn in nmf_fns], key=natural_keys)
    # =========================================================================

    # Save coreg params info to current coreg dir:
    pp.pprint(params_thr)
    with open(os.path.join(coreg_output_dir, 'coreg_params.json'), 'w') as f:
        json.dump(params_thr, f, indent=4, sort_keys=True)

    # Create outfile:
    coreg_outpath = os.path.join(coreg_output_dir, 'coreg_results_{}.hdf5'.format(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")))
    coreg_outfile = h5py.File(coreg_outpath, 'w')
    for k in params_thr.keys():
        print k
        if k == 'eval': # eval is a dict, will be saved in roiparams.json (no need to save as attr for coreg)
            continue
        coreg_outfile.attrs[k] = native(params_thr[k])
    coreg_outfile.attrs['creation_date'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # For each file, find matches to reference ROIs:
    filenames = [str(re.search('File(\d{3})', fn).group(0)) for fn in roi_source_paths]
    filenames = sorted([f for f in filenames if f not in params_thr['excluded_tiffs']], key=natural_keys)
    print "COREGISTERING ACROSS %i FILES." % len(filenames)

    all_matches = dict()
    ref_file = str(params_thr['ref_filename'])
    try:
        # Get list of ROIs of keep_good_rois=True and evalulation set used:
        if pass_rois_dict is None:
            pass_rois_dict = dict((k, None) for k in filenames)

        # Get REFERENCE:
        pass_rois = pass_rois_dict[params_thr['ref_filename']]
        A1, dims, ref_pass_rois, img = load_source_rois(params_thr['ref_filepath'], keep_good_rois=params_thr['keep_good_rois'], pass_rois=pass_rois)
        REF = dict()
        REF['mat'] = A1
        REF['pass_roi_idxs'] = ref_pass_rois
        REF['dims'] = dims

        # Then, get matches to sample:
        for curr_file in filenames:
            curr_filepath = [p for p in roi_source_paths if str(re.search('File(\d{3})', p).group(0)) == curr_file][0]
            results = match_file_against_ref(REF, curr_filepath, params_thr, pass_rois_dict=pass_rois_dict, asdict=True)

            zproj = coreg_outfile.create_dataset('/'.join([curr_file, 'img']), results['img'].shape, results['img'].dtype)
            zproj[...] = results['img']

            src = coreg_outfile.create_dataset('/'.join([curr_file, 'roimat']), results['A'].shape, results['A'].dtype)
            src[...] = results['A'].todense()
            src.attrs['source'] = curr_filepath

            kpt = coreg_outfile.create_dataset('/'.join([curr_file, 'roi_idxs']), results['pass_roi_idxs'].shape, results['pass_roi_idxs'].dtype)
            kpt[...] = results['pass_roi_idxs']

            dist = coreg_outfile.create_dataset('/'.join([curr_file, 'distance']), results['distance_thr'].shape, results['distance_thr'].dtype)
            dist[...] = results['distance_thr']
            dist.attrs['d1'] = dims[0]
            dist.attrs['d2'] = dims[1]
            if len(dims) > 2:
                dist.attrs['d3'] = dims[2]
            dist.attrs['dist_thr'] = params_thr['dist_thr']

            # Plot distance mat curr file:
            pl.figure()
            pl.imshow(results['distance_thr']); pl.colorbar();
            pl.title('%s - dists to ref (%s, overlap_thr %s)' % (curr_file, ref_file, str(params_thr['dist_overlap_thr'])))
            pl.savefig(os.path.join(output_dir_figs, 'distancematrix_%s.png' % curr_file))
            pl.close()

            # Save matches to reference for current file:
            match = coreg_outfile.create_dataset('/'.join([curr_file, 'matches_to_ref']), results['matches_to_ref'].shape, results['matches_to_ref'].dtype)
            match[...] = results['matches_to_ref']
            match.attrs['ref_filename'] = params_thr['ref_filename']
            match.attrs['ref_filepath'] = params_thr['ref_filepath']

            if not isinstance(results['matches_to_ref'], list):
                all_matches[curr_file] = results['matches_to_ref'].tolist()

        # Also save to json for easy viewing:
        match_fn_base = 'matches_byfile_r%s' % str(params_thr['ref_filename'])
        with open(os.path.join(coreg_output_dir, '%s.json' % match_fn_base), 'w') as f:
            json.dump(all_matches, f, indent=4, sort_keys=True)

    except Exception as e:
        print "-- ERROR: in finding matches to ref. --------------------------"
        traceback.print_exc()
        print "---------------------------------------------------------------"
    finally:
        coreg_outfile.close()

    return all_matches, coreg_outpath


#%%
def plot_roi_contours(roi_list, roi_mat, dims, color=['b']):
    d1 = dims[0]
    d2 = dims[1]
    nr = roi_mat.shape[-1]
    masks = np.reshape(np.array(roi_mat), (d1, d2, nr), order='F')

    for ridx,roi in enumerate(roi_list):
        x, y = np.mgrid[0:d1:1, 0:d2:1]
        indx = np.argsort(roi_mat[:,roi], axis=None)[::-1]
        cumEn = np.cumsum(roi_mat[:,roi].flatten()[indx]**2)
        cumEn /= cumEn[-1] # normalize
        Bvec = np.zeros(d1*d2)
        Bvec[indx] = cumEn
        Bmat = np.reshape(Bvec, (d1,d2), order='F')
        cs = pl.contour(y, x, Bmat, [0.9], colors=color[ridx]) #[colorvals[fidx]]) #, cmap=colormap)

        # Label it:
        masktmp = masks[:,:,roi]
        [ys, xs] = np.where(masktmp>0)
        pl.text(xs[int(round(len(xs)/4))], ys[int(round(len(ys)/4))], str(roi), color=color) #, weight='bold')

#%%
#coreg_results_path = coreg_outpath

def plot_matched_rois_by_file(all_matches, coreg_results_path):
    # TODO:  Add 3D compatibility...

    # Create output dir for figures:
    coreg_output_dir = os.path.split(coreg_results_path)[0]
    coreg_fig_dir = os.path.join(coreg_output_dir, 'figures', 'files')
    if not os.path.exists(coreg_fig_dir):
        os.makedirs(coreg_fig_dir)

    # Load coreg params info:
    coreg_paramspath = os.path.join(os.path.split(coreg_results_path)[0], 'coreg_params.json')
    with open(coreg_paramspath, 'r') as f:
        params_thr = json.load(f)

    # Load COREG results:
    results = h5py.File(coreg_results_path, 'r')

    # First, get reference:
    A1 = np.array(results['%s/roimat' % params_thr['ref_filename']])
    d1 = results['%s/distance' % params_thr['ref_filename']].attrs['d1']
    d2 = results['%s/distance' % params_thr['ref_filename']].attrs['d2']
    dims = [d1, d2]
    nr = A1.shape[-1]

#    masks = np.reshape(np.array(A1), (d1, d2, nr), order='F')
#    print "Loaded reference masks with shape:", masks.shape
#    img = results['%s/img' % params_thr['ref_filename']]

    ref_file = str(params_thr['ref_filename'])

    for curr_file in all_matches.keys():
        print "--- Plotting %s ---" % curr_file

        roi_mat = np.array(results['%s/roimat' % curr_file])
        img = results['%s/img' % curr_file]

#        masks2 = np.reshape(A2, (d1,d2,nr), order='F') #np.reshape(np.array(A2.todense()), (d1, d2, nr), order='F')

        # Plot contours overlaid on reference image:
        pl.figure()
        pl.imshow(img, cmap='gray')

        ref_rois = [m[0] for m in all_matches[curr_file]]
        matches = [m[1] for m  in all_matches[curr_file]]

        if curr_file == ref_file:
            plot_roi_contours(np.arange(0, nr), A1, dims, color='b')
            pl.title('REFERENCE: %s' % ref_file)
        else:
            plot_roi_contours(ref_rois, A1, dims, color='b')
            plot_roi_contours(matches, roi_mat, dims, color='r')
            pl.title('%s coreg to %s' % (curr_file, ref_file))

        pl.axis('off')
        pl.savefig(os.path.join(coreg_fig_dir, 'matches_%s_%s.png' % (str(ref_file), str(curr_file))))
        pl.close()

#%%
def plot_coregistered_rois(coregistered_rois, coreg_results_path, cmap='jet', plot_by_file=True):

    # Create output dir for figures:
    coreg_output_dir = os.path.split(coreg_results_path)[0]
    coreg_fig_dir = os.path.join(coreg_output_dir, 'files')
    if not os.path.exists(coreg_fig_dir):
        os.makedirs(coreg_fig_dir)

    # Load coreg params info:
    coreg_paramspath = os.path.join(os.path.split(coreg_results_path)[0], 'coreg_params.json')
    with open(coreg_paramspath, 'r') as f:
        params_thr = json.load(f)

    # Load COREG results:
    results = h5py.File(coreg_results_path, 'r')


    # First, get reference:
    A1 = np.array(results['%s/roimat' % params_thr['ref_filename']])
    d1 = results['%s/distance' % params_thr['ref_filename']].attrs['d1']
    d2 = results['%s/distance' % params_thr['ref_filename']].attrs['d2']
    dims = [d1, d2]
    nr = A1.shape[-1]
    refimg = results['%s/img' % params_thr['ref_filename']]

    colormap = pl.get_cmap(cmap)
    ref_rois = coregistered_rois[params_thr['ref_filename']]
    nrois = len(ref_rois)
    print "Plotting %i coregistered ROIs from each file..." % nrois

    file_names = coregistered_rois.keys();
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

    for fidx,curr_file in enumerate(sorted(coregistered_rois.keys(), key=natural_keys)):
        print "--- Plotting %s ---" % curr_file

        roi_mat = np.array(results['%s/roimat' % curr_file])

        curr_rois = coregistered_rois[curr_file]
        if plot_by_file is True:
            colorlist = np.tile(colorvals[fidx], [len(curr_rois,),1])
        else:
            colorlist = [colorvals[ridx] for ridx in range(len(curr_rois))]

        plot_roi_contours(curr_rois, roi_mat, dims, color=colorlist)

#        for ridx, roi in enumerate(curr_rois):
#            # compute the cumulative sum of the energy of the Ath component that
#            # has been ordered from least to highest:
#            indx = np.argsort(A[:,roi], axis=None)[::-1]
#            cumEn = np.cumsum(A[:,roi].flatten()[indx]**2)
#            cumEn /= cumEn[-1] # normalize
#            Bvec = np.zeros(d1*d2)
#            Bvec[indx] = cumEn
#            Bmat = np.reshape(Bvec, (d1,d2), order='F')
#            if plot_by_file is True:
#                cs = pl.contour(y, x, Bmat, [0.9], colors=[colorvals[fidx]]) #, cmap=colormap)
#            else:
#                cs = pl.contour(y, x, Bmat, [0.9], colors=[colorvals[ridx]]) #, cmap=colormap)

    nfiles = len(coregistered_rois.keys())
    nrois = len(ref_rois)
    print "N files:", nfiles
    print "N rois:", nrois

    #%
    # PLOT labels and legend:
    ax2 = pl.subplot(gs[1])
    if plot_by_file is True:
        interval = np.arange(0., 1., 1./nfiles)
        for fidx,curr_file in enumerate(sorted(coregistered_rois.keys(), key=natural_keys)):
            ax2.plot(1, interval[fidx], c=colorvals[fidx], marker='.', markersize=20)
            pl.text(1.1, interval[fidx], str(curr_file), fontsize=12)
            pl.xlim([0.95, 2])
    else:
        interval = np.arange(0., 1., 1./nrois)
        for ridx, roi in enumerate(ref_rois):
            ax2.plot(1, interval[ridx], c=colorvals[ridx], marker='.', markersize=20)
            pl.text(1.1, interval[ridx], "%s_%s" % ('roi%05d' % int(ridx+1), str(roi)), fontsize=12)
            pl.xlim([0.95, 2])
    pl.axis('equal')
    pl.axis('off')

    #%
    pl.savefig(os.path.join(coreg_fig_dir, 'contours_%s_r%s.png' % (plot_type,  str(params_thr['ref_filename'])))) #matchedrois_fn_base))
    pl.close()



#%%
#
#with open(os.path.join(rootdir, animalid, session, 'ROIs', 'rids_%s.json' % session), 'r') as f:
#    roidict = json.load(f)
#rid_hash = roidict[roi_id]['rid_hash']
#
#tmp_rid_path = os.path.join(rootdir, animalid, session, 'ROIs', 'tmp_rids', 'tmp_rid_%s.json' % rid_hash)
#
#tmp_results_paths = []
#for fn in np.arange(1, 11, 1):
#    tmp_fpath = coregister_file_by_rid(tmp_rid_path, filenum=fn, nprocs=12, rootdir=rootdir)
#    tmp_results_paths.append(tmp_fpath)

#coreg_results_path = collate_slurm_output(tmp_rid_path, rootdir='')


#%% SLURM SCRIPT:


# tmp_fpath = coregister_file_by_rid(tmp_rid_path, filenum=fn, nprocs=12, rootdir=rootdir)

def collate_slurm_output(tmp_rid_path, rootdir=''):

    all_matches, coreg_results_path = collate_coreg_results(tmp_rid_path, rootdir=rootdir)

    plot_matched_rois_by_file(all_matches, coreg_results_path)

    ref_rois, ref_file = find_universal_matches(coreg_results_path, all_matches)

    # Update COREG RESULTS FILE:
    coregistered_rois = append_universal_matches(coreg_results_path, ref_rois)

    print "COMPLETED COREGISTRATION."
    print "Output file saved to: %s" % coreg_results_path
    print "Found %i universal matches to reference: %s" % (len(ref_rois), ref_file)
    print coregistered_rois

    ncoreg_rois = len(coregistered_rois[coregistered_rois.keys()[0]])
    pp.pprint(coregistered_rois)
    print "Total %i Universal Matches found." % ncoreg_rois
    print "Output saved to:", coreg_results_path

    # Save plots of universal matches:
    # =========================================================================
    if len(ncoreg_rois) > 0:
        plot_coregistered_rois(coregistered_rois, coreg_results_path, plot_by_file=True)
        plot_coregistered_rois(coregistered_rois, coreg_results_path, plot_by_file=False)


    return coreg_results_path

#%%
def collate_coreg_results(tmp_rid_path, rootdir=''):

    with open(tmp_rid_path, 'r') as f:
        RID = json.load(f)

    # Set COREG output dir:
    coreg_output_dir = os.path.join(RID['DST'], 'coreg_results')
    if rootdir not in coreg_output_dir:
        session_dir = coreg_output_dir.split('/ROIs')[0]
        session = os.path.split(session_dir)[-1]
        animalid = os.path.split(os.path.split(session_dir)[0])[-1]
        coreg_output_dir = replace_root(coreg_output_dir, rootdir, animalid, session)

    # Create figure dir:
    output_dir_figs = os.path.join(coreg_output_dir, 'figures', 'files')
    if not os.path.exists(output_dir_figs):
        os.makedirs(output_dir_figs)

    # Load coreg params:
    with open(os.path.join(coreg_output_dir, 'coreg_params.json'), 'r') as f:
        params_thr = json.load(f)
        if len(params_thr['excluded_tiffs']) > 0:
            params_thr['excluded_tiffs'] = [str(f) for f in params_thr['excluded_tiffs']]


    # Get tmp results files:
    tmp_results_dir = os.path.join(coreg_output_dir, 'tmp')
    tmp_results_paths = sorted([os.path.join(tmp_results_dir, fn) for fn in os.listdir(tmp_results_dir) if fn.endswith('npz')], key=natural_keys)

    # Create outfile:
    coreg_outpath = os.path.join(coreg_output_dir, 'coreg_results_{}.hdf5'.format(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")))
    coreg_outfile = h5py.File(coreg_outpath, 'w')
    for k in params_thr.keys():
        print k
        if k == 'eval': # eval is a dict, will be saved in roiparams.json (no need to save as attr for coreg)
            continue
        coreg_outfile.attrs[k] = native(params_thr[k])
    coreg_outfile.attrs['creation_date'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # For each file, write results to main coreg results file:
    filenames = [str(re.search('File(\d{3})', fn).group(0)) for fn in tmp_results_paths]
    filenames = sorted([f for f in filenames if f not in params_thr['excluded_tiffs']], key=natural_keys)
    print "COLLATING results of coregistration ACROSS %i FILES." % len(filenames)

    ref_file = str(params_thr['ref_filename'])
    all_matches = dict()

    for curr_file, tmp_results_path in zip(sorted(filenames, key=natural_keys), sorted(tmp_results_paths, key=natural_keys)):

        print "--- COLLATING: %s" % curr_file

        results = np.load(tmp_results_path)

        img = coreg_outfile.create_dataset('/'.join([curr_file, 'img']), results['img'].shape, results['img'].dtype)
        img[...] = results['img']

        roimat = np.array(results['A'].all().todense())
        src = coreg_outfile.create_dataset('/'.join([curr_file, 'roimat']), roimat.shape, roimat.dtype)
        src[...] = roimat
        src.attrs['source'] = str(results['source'])

        kpt = coreg_outfile.create_dataset('/'.join([curr_file, 'roi_idxs']), results['pass_roi_idxs'].shape, results['pass_roi_idxs'].dtype)
        kpt[...] = results['pass_roi_idxs']

        dist = coreg_outfile.create_dataset('/'.join([curr_file, 'distance']), results['distance'].shape, results['distance'].dtype)
        dist[...] = results['distance']
        dist.attrs['d1'] = results['dims'][0]
        dist.attrs['d2'] = results['dims'][1]
        if len(results['dims']) > 2:
            dist.attrs['d3'] = results['dims'][2]
        dist.attrs['dist_thr'] = params_thr['dist_thr']

        # Plot distance mat curr file:
        pl.figure()
        pl.imshow(results['distance']); pl.colorbar();
        pl.title('%s - dists to ref (%s, overlap_thr %s)' % (curr_file, ref_file, str(params_thr['dist_overlap_thr'])))
        pl.savefig(os.path.join(output_dir_figs, 'distancematrix_%s.png' % curr_file))
        pl.close()

        # Save matches to reference for current file
        if not isinstance(results['matches_to_ref'][0], list): # == 1:
            if isinstance(results['matches_to_ref'][0], int):
                # This is the old version of saving reference idxs
                matched_pairs = [[v, v] for v in results['matches_to_ref']]
            else:
                matched_pairs = results['matches_to_ref'].tolist()
        ref_idxs = np.array([m[0] for m in matched_pairs])
        matched_idxs = np.array([m[1] for m in matched_pairs])

        match = coreg_outfile.create_dataset('/'.join([curr_file, 'matches_to_ref']), matched_idxs.shape, matched_idxs.dtype)
        match[...] = matched_idxs #results['matches_to_ref']
        match.attrs['ref_idxs'] = ref_idxs
        match.attrs['ref_filename'] = params_thr['ref_filename']
        match.attrs['ref_filepath'] = params_thr['ref_filepath']

        #if not isinstance(matched_pairs, list):
        all_matches[curr_file] = matched_pairs #.tolist()

    # Also save to json for easy viewing:
    match_fn_base = 'matches_byfile_r%s' % str(params_thr['ref_filename'])
    with open(os.path.join(coreg_output_dir, '%s.json' % match_fn_base), 'w') as f:
        json.dump(all_matches, f, indent=4, sort_keys=True)

    return all_matches, coreg_outpath

#%%
def find_universal_matches(coreg_results_path, all_matches):

    # Load coreg params:
    coreg_output_dir = os.path.split(coreg_results_path)[0]
    with open(os.path.join(coreg_output_dir, 'coreg_params.json'), 'r') as f:
        params_thr = json.load(f)
    if len(params_thr['excluded_tiffs']) > 0:
        params_thr['excluded_tiffs'] = [str(f) for f in params_thr['excluded_tiffs']]

    ref_file = str(params_thr['ref_file'])
    #% Find intersection of all matches with reference (aka, "universal matches"):
    filenames = all_matches.keys()
    filenames.extend([str(params_thr['ref_filename'])])
    filenames = sorted(list(set(filenames)), key=natural_keys)
    ref_idxs = [[comp[0] for comp in all_matches[f]] for f in all_matches.keys() if f not in params_thr['excluded_tiffs']]
    ref_rois = set(ref_idxs[0])
    for s in ref_idxs[1:]:
        ref_rois.intersection_update(s)
    ref_rois = list(ref_rois)

    return ref_rois, ref_file

#%%
def append_universal_matches(coreg_results_path, ref_rois):

    # Load coreg params:
    coreg_output_dir = os.path.split(coreg_results_path)[0]
    with open(os.path.join(coreg_output_dir, 'coreg_params.json'), 'r') as f:
        params_thr = json.load(f)
    try:
        coregistered_rois = dict()
        coreg_info = h5py.File(coreg_results_path, 'a')
        for fn in coreg_info.keys():
            if fn == params_thr['ref_filename']:
                curr_match_idxs = np.array([int(i) for i in ref_rois])
            else:
                match_idxs = np.array([[m for m in coreg_info[fn]['matches_to_ref'].attrs['ref_idxs']].index(refroi) for refroi in ref_rois])
                curr_match_idxs = np.empty((len(match_idxs,)), dtype=int)
                for m in range(len(match_idxs)):
                    curr_match_idxs[m] = int(coreg_info[fn]['matches_to_ref'][match_idxs[m]]) #np.array([m[1] for m in coreg_info[fn]['matches_to_ref'][match_idxs,:]])
            if 'universal_matches' in coreg_info[fn].keys():
                umatches = coreg_info[fn]['universal_matches']
            else:
                umatches = coreg_info[fn].create_dataset('universal_matches', curr_match_idxs.shape, curr_match_idxs.dtype)
            umatches[...] = curr_match_idxs
            coregistered_rois[fn] = curr_match_idxs.tolist()

    except Exception as e:
        print "---------------------------------------"
        print "Error saving universal matches, %s" % fn
        traceback.print_exc()
    finally:
        coreg_info.close()
        print "---------------------------------------"

    #% Save ROI idxs of unviersal matches:
    coregistered_rois_fn = 'coregistered_r%s.json' % str(params_thr['ref_filename'])
    print("Saving coregistered ROIs to: %s" % os.path.join(coreg_output_dir, coregistered_rois_fn))
    with open(os.path.join(coreg_output_dir, coregistered_rois_fn), 'w') as f:
        json.dump(coregistered_rois, f, indent=4, sort_keys=True)

    return coregistered_rois

#%%
def coregister_rois_nmf(RID, coreg_output_dir, excluded_tiffs=[], rootdir='', coreg_fig_dir=None):

    ref_rois = []

    if not os.path.exists(coreg_output_dir):
        os.makedirs(coreg_output_dir)

    if coreg_fig_dir is None:
        coreg_fig_dir = os.path.join(coreg_output_dir, 'figures')
    if not os.path.exists(coreg_fig_dir):
        os.makedirs(coreg_fig_dir)

    # Load coreg params:
    with open(os.path.join(coreg_output_dir, 'coreg_params.json'), 'r') as f:
        params_thr = json.load(f)
    if len(params_thr['excluded_tiffs']) > 0:
        params_thr['excluded_tiffs'] = [str(f) for f in params_thr['excluded_tiffs']]

    # Get matches:
    print "FINDING MATCHES...."
    all_matches, coreg_results_path = find_matches_nmf(RID, coreg_output_dir, rootdir=rootdir)

    plot_matched_rois_by_file(all_matches, coreg_results_path)

    ref_rois, ref_file = find_universal_matches(coreg_results_path, all_matches)

    # Update COREG RESULTS FILE:
    coregistered_rois = append_universal_matches(coreg_results_path, ref_rois)

    return coregistered_rois, coreg_results_path


#%%


#%%

#%%
def run_coregistration(options):

    parser = optparse.OptionParser()

    parser.add_option('-D', '--root', action='store', dest='rootdir', default='/nas/volume1/2photon/data', help='data root dir (root project dir containing all animalids) [default: /nas/volume1/2photon/data, /n/coxfs01/2pdata if --slurm]')
    parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')
    parser.add_option('-S', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID')
    parser.add_option('--slurm', action='store_true', dest='slurm', default=False, help="set if running as SLURM job on Odyssey")
    parser.add_option('--default', action='store_true', dest='default', default='store_false', help="Use all DEFAULT params, for params not specified by user (no interactive)")

    parser.add_option('-r', '--roi-id', action='store', dest='roi_id', default='', help="ROI ID for rid param set to use (created with set_roi_params.py, e.g., rois001, rois005, etc.)")

#    parser.add_option('-t', '--maxthr', action='store', dest='dist_maxthr', default=0.1, help="threshold for turning spatial components into binary masks [default: 0.1]")
#    parser.add_option('-n', '--power', action='store', dest='dist_exp', default=0.1, help="power n for distance between masked components: dist = 1 - (and(M1,M2)/or(M1,M2)**n [default: 1]")
#    parser.add_option('-d', '--dist', action='store', dest='dist_thr', default=0.5, help="threshold for setting a distance to infinity, i.e., illegal matches [default: 0.5]")
#    parser.add_option('-o', '--overlap', action='store', dest='dist_overlap_thr', default=0.8, help="overlap threshold for detecting if one ROI is subset of another [default: 0.8]")


#    parser.add_option('-x', '--exclude', action="store",
#                      dest="exclude_file_ids", default='', help="comma-separated list of files to exclude")
#    parser.add_option('-M', '--mcmetric', action="store",
#                  dest="mcmetric", default='zproj_corrcoefs', help="Motion-correction metric to use for identifying tiffs to exclude [default: zproj_corrcoefs]")

#    parser.add_option('--good', action="store_true",
#                      dest="keep_good_rois", default=False, help="Set flag to only keep good components (useful for avoiding computing massive ROI sets)")
#    parser.add_option('--max', action="store_true",
#                      dest="use_max_nrois", default=False, help="Set flag to use file with max N components (instead of reference file) [default uses reference]")
#    parser.add_option('--roipath', action="store",
#                      dest="roipath", default="", help="If keep_good_rois is True, path to .json with ROI idxs for each file (if using cNMF). Default uses nmf-extraction ROIs.")

#    parser.add_option('-O', '--outdir', action="store",
#                      dest="coreg_output_dir", default=None, help="Output dir to save coreg results to. Default uses curr ROI dir + 'coreg_results'")

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

#    tmp_exclude = options.exclude_file_ids
#    if len(tmp_exclude)==0:
#        exclude_manual = []
#    else:
#        exclude_files = tmp_exclude.split(',')
#        exclude_files = [int(f) for f in exclude_files]
#        exclude_manual = ['File%03d' % f for f in exclude_files]
#    print("Excluding files: ", exclude_manual)

    #keep_good_rois = options.keep_good_rois
    #use_max_nrois = options.use_max_nrois

    #roi_eval_path = options.roipath
    #mcmetric = options.mcmetric

#    coreg_output_dir = options.coreg_output_dir
#    coreg_fidx = int(options.ref_file) - 1
#    reference_filename = "File%03d" % int(options.ref_file)
#
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

    exclude_manual = RID['PARAMS']['eval']['manual_excluded']
    mcmetric = RID['PARAMS']['eval']['mcmetric']
    check_motion = RID['PARAMS']['eval']['check_motion']


    #%%
    # =========================================================================
    # Get ROI source(s) info:
    # =========================================================================
#    tiff_sourcedir = RID['SRC']
#    path_parts = tiff_sourcedir.split(session_dir)[-1].split('/')
#    acquisition = path_parts[1]
#    run = path_parts[2]
#    process_dirname = path_parts[4]
#    process_id = process_dirname.split('_')[0]
#    roi_ref_type = RID['PARAMS']['options']['source']['roi_type']
#    roi_source_dir = RID['PARAMS']['options']['source']['roi_dir']

    # Get ROI source files and their sources (.tif, .mmap) -- use MC evaluation
    # results, if relevant.
    roi_source_paths, tiff_source_paths, filenames, mc_excluded_tiffs, mcmetrics_path = get_source_paths(session_dir, RID, check_motion=check_motion,
                                                                                                      mcmetric=mcmetric, rootdir=rootdir)
#    # Get list of .tif files to exclude (from MC-eval fail or user-choice):
    excluded_tiffs = list(set(exclude_manual + mc_excluded_tiffs))
    print "Additionally excluding manully-selected tiffs:", mc_excluded_tiffs
    print "Excluded:", excluded_tiffs
#    params_thr['excluded_tiffs'] = excluded_tiffs


    # Create dir for coregistration output:
    coreg_output_dir = os.path.join(RID['DST'], 'coreg_results')
    print "Saving COREG results to:", coreg_output_dir
    if not os.path.exists(coreg_output_dir):
        os.makedirs(coreg_output_dir)
    #%
    # =========================================================================
    # COREGISTER ROIs:
    # =========================================================================
    coregistered_rois, coreg_results_path = coregister_rois_nmf(RID, coreg_output_dir, excluded_tiffs=excluded_tiffs, rootdir=rootdir)
    print "COREGISTRATION COMPLETE!"
    ncoreg_rois = len(coregistered_rois[coregistered_rois.keys()[0]])
    pp.pprint(coregistered_rois)
    print "Total %i Universal Matches found." % ncoreg_rois
    print "Output saved to:", coreg_results_path

    # Save plots of universal matches:
    # =========================================================================
    if len(ncoreg_rois) > 0:
        plot_coregistered_rois(coregistered_rois, coreg_results_path, plot_by_file=True)
        plot_coregistered_rois(coregistered_rois, coreg_results_path, plot_by_file=False)

    return coregistered_rois, coreg_results_path

#%%
def main(options):

    coregistered_rois, coreg_results_path = run_coregistration(options)
    ncoreg_rois = len(coregistered_rois[coregistered_rois.keys()[0]])

    print "----------------------------------------------------------------"
    print "Finished coregistration."
    print "Found %i matches across files to reference." % ncoreg_rois #len(ref_rois)
    print "Saved output to:"
    print coreg_results_path
    print "----------------------------------------------------------------"


    #%% Get NMF output files:

if __name__ == '__main__':
    main(sys.argv[1:])
