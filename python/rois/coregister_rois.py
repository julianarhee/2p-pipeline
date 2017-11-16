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
import cv2
import glob
import numpy as np
import os
from scipy.sparse import spdiags, issparse

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

parser = optparse.OptionParser()

parser.add_option('-S', '--source', action='store', dest='source', default='/nas/volume1/2photon/projects', help='source dir (root project dir containing all expts) [default: /nas/volume1/2photon/projects]')
parser.add_option('-E', '--experiment', action='store', dest='experiment', default='', help='experiment type (parent of session dir)')
parser.add_option('-s', '--sess', action='store', dest='session', default='', help='session name')
parser.add_option('-A', '--acq', action='store', dest='acquisition', default='FOV1', help='acquisition folder')
parser.add_option('-f', '--func', action='store', dest='functional', default='functional', help="folder containing functional tiffs [default: 'functional']")
parser.add_option('-R', '--roi', action='store', dest='roi_id', default='', help="unique ROI ID (child of <acquisition_dir>/ROIs/")

parser.add_option('-t', '--maxthr', action='store', dest='dist_maxthr', default=0.1, help="threshold for turning spatial components into binary masks [default: 0.1]")
parser.add_option('-n', '--power', action='store', dest='dist_exp', default=0.1, help="power n for distance between masked components: dist = 1 - (and(M1,M2)/or(M1,M2)**n [default: 1]")
parser.add_option('-d', '--dist', action='store', dest='dist_thr', default=0.5, help="threshold for setting a distance to infinity, i.e., illegal matches [default: 0.5]")
parser.add_option('-o', '--overlap', action='store', dest='dist_overlap_thr', default=0.8, help="overlap threshold for detecting if one ROI is subset of another [default: 0.8]")


parser.add_option('-x', '--exclude', action="store",
                  dest="exclude_file_ids", default='', help="comma-separated list of files to exclude")
parser.add_option('--threshold', action="store_true",
                  dest="threshold", default=False, help="Set flag to only keep good components (useful for avoiding computing massive ROI sets)")
parser.add_option('--max', action="store_true",
                  dest="use_max_filter", default=False, help="Set flag to use file with max N components (instead of reference file) [default uses reference]")


(options, args) = parser.parse_args() 

source = options.source #'/nas/volume1/2photon/projects'
experiment = options.experiment #'gratings_phaseMod'
session = options.session #'20171009_CE059'
acquisition = options.acquisition #'FOV1_zoom3x'
functional = options.functional # 'functional'

roi_id = options.roi_id #'caiman2Dnmf003'

tmp_exclude = options.exclude_file_ids
if len(tmp_exclude)==0:
    exclude = []
else:
    exclude_files = tmp_exclude.split(',')
    exclude_files = [int(f) for f in exclude_files]
    exclude = ['File%03d' % f for f in exclude_files]
print("Excluding files: ", exclude)

threshold = options.threshold
use_max_filter = options.use_max_filter

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

params_thr['excluded'] = exclude
params_thr['threshold'] = threshold
if use_max_filter is True:
    params_thr['filter_type'] = 'max'
else:
    params_thr['filter_type'] = 'ref'

# In[3]:

acquisition_dir = os.path.join(source, experiment, session, acquisition)

acquisition_meta_fn = os.path.join(acquisition_dir, 'reference_%s.json' % functional)
with open(acquisition_meta_fn, 'r') as f:
    acqmeta = json.load(f)

roi_dir = os.path.join(acqmeta['roi_dir'], roi_id)
roiparams_path = os.path.join(roi_dir, 'roiparams.json')
with open(roiparams_path, 'r') as f:
    roiparams = json.load(f)

roiparams = byteify(roiparams)
if not roi_id==roiparams['roi_id']:
    print("***WARNING***")
    print("Loaded ROIPARAMS id doesn't match user-specified roi_id.")
    pp.pprint(roiparams)
    use_loaded = raw_input('Use loaded ROIPARAMS? Press Y/n: ')
    if use_loaded=='Y':
        roi_id = roiparams['roi_id']


#%%
# Load mcparams.mat:
mcparams = scipy.io.loadmat(acqmeta['mcparams_path'])
mc_ids = sorted([m for m in mcparams.keys() if 'mcparams' in m], key=natural_keys)
if len(mc_ids)>1:
    for mcidx,mcid in enumerate(sorted(mc_ids, key=natural_keys)):
        print(mcidx, mcid)
    mc_id_idx = raw_input('Select IDX of mc-method to use: ')
    mc_id = mc_ids[int(mc_id_idx)]
    print("Using MC-METHOD: ", mc_id)
else:
    mc_id = mc_ids[0]

mcparams = mcparams[mc_id] #mcparams['mcparams01']
reference_file_idx = int(mcparams['ref_file'])
signal_channel_idx = int(mcparams['ref_channel'])

signal_channel = 'Channel%02d' % int(signal_channel_idx)
reference_file = 'File%03d' % int(reference_file_idx)
if signal_channel_idx==0:
    signal_channel_idx = input('No ref channel found. Enter signal channel idx (1-indexing): ')
if reference_file_idx==0:
    reference_file_idx = input('No ref file found. Enter file idx (1-indexing): ')

signal_channel = 'Channel%02d' % int(signal_channel_idx)
reference_file = 'File%03d' % int(reference_file_idx)
print("Specified signal channel is:", signal_channel)
print("Selected reference file:", reference_file)
#del mcparams


if isinstance(acqmeta['slices'], int):
    nslices = acqmeta['slices']
else:
    nslices = len(acqmeta['slices'])
    
print(nslices)


#%% 
all_file_names = sorted(['File%03d' % int(f+1) for f in range(acqmeta['ntiffs'])], key=natural_keys)

#%% only run on good MC files:

metrics_path = os.path.join(acqmeta['acquisition_base_dir'], functional, 'DATA', 'mcmetrics.json')
print(metrics_path)
bad_files = []
bad_fids = []
if os.path.exists(metrics_path):
    with open(metrics_path, 'r') as f:
        metrics_info = json.load(f)
        
    mcmetrics = metrics_info[mc_id]
    print(mcmetrics)
    if len(mcmetrics['bad_files'])>0:
        bad_fids = [int(i)-1 for i in mcmetrics['bad_files']]
        bad_files = ['File%03d' % int(i) for i in mcmetrics['bad_files']]
        print("Bad MC files excluded:", bad_files)

file_names = [t for i,t in enumerate(sorted(all_file_names, key=natural_keys)) if i not in bad_fids]

print("Files that passed MC:", file_names)

#%% Get NMF output files:

# source of NMF output run:
nmf_output_dir = os.path.join(roi_dir, 'nmf_output')
all_nmf_fns = sorted([n for n in os.listdir(nmf_output_dir) if n.endswith('npz')], key=natural_keys)
nmf_fns = []
for f in file_names:
    match_nmf = [m for m in all_nmf_fns if f in m][0]
    nmf_fns.append(match_nmf)
    
ref_nmf_fn = [f for f in nmf_fns if reference_file in f][0]



#%% Get source tiffs (mmap):
tiff_source = str(mcparams['dest_dir'][0][0][0])
tiff_dir = os.path.join(acquisition_dir, functional, 'DATA', tiff_source)
#tiff_dir

# Get mmap tiffs:
memmapped_fns = sorted([m for m in os.listdir(tiff_dir) if m.endswith('mmap')], key=natural_keys)

#%% Create output dir for matches:
if params_thr['threshold'] is True:
    metrics_subdir = 'metrics_%s_thr' % params_thr['filter_type']
else:
    metrics_subdir = 'metrics_%s_all' % params_thr['filter_type']

metrics_dir = os.path.join(roi_dir, metrics_subdir)
metrics_dir_figs = os.path.join(metrics_dir, 'figures')
if not os.path.exists(metrics_dir):
    os.mkdir(metrics_dir)
if not os.path.exists(metrics_dir_figs):
    os.mkdir(metrics_dir_figs)
print("Saving output figures to: %s" % metrics_dir_figs)


if use_max_filter is True:
    # Get file that has max N components:
    tmp_nrs = []
    for fn,f in zip(file_names, nmf_fns):
        nmf = np.load(os.path.join(nmf_output_dir, f))
        if threshold is True:
            tmp_nr = len(nmf['idx_components'])
        else:
            tmp_nr = nmf['A'].all().shape[-1]
            
        tmp_nrs.append(tmp_nr)
    
    max_idx = tmp_nrs.index(max(tmp_nrs))
    roi_reference_file = file_names[max_idx]
else:
    roi_reference_file = reference_file

params_thr['roi_ref'] = roi_reference_file
  
#%% Load REF-NMF to get first set of ROIs:

ref_nmf_fn = [f for f in nmf_fns if roi_reference_file in f][0]
refnmf = np.load(os.path.join(nmf_output_dir, ref_nmf_fn))
print(refnmf.keys())

#%% Get average image
img = refnmf['Av']

#pl.figure()
#pl.imshow(img)

#%% Get components from REFERENCE file:
    
nr = refnmf['A'].all().shape[1]
d1 = int(refnmf['d1'])
d2 = int(refnmf['d2'])

A1 = refnmf['A'].all()
#nA1 = np.array(np.sqrt(A1.power(2).sum(0)).T)
#A1 = scipy.sparse.coo_matrix(A1 / nA1.T)

threshold = True

#%%
from caiman.components_evaluation import estimate_components_quality

#%%
if threshold is True:
#    ref_mmap = [m for m in memmapped_fns if reference_file in m][0]
#    Yr, dims, T = cm.load_memmap(os.path.join(tiff_dir, ref_mmap))
#    d1, d2 = dims
#    Y = np.reshape(Yr, dims + (T,), order='F')
#    
#    #%%
#    final_frate = 44.7027 #params_movie['final_frate'] #44.7027 #params_movie['final_frate']
#    r_values_min = 0.8 #params_movie['r_values_min_patch']  # threshold on space consistency
#    fitness_min = -20 #params_movie['fitness_delta_min_patch']  # threshold on time variability
#    # threshold on time variability (if nonsparse activity)
#    fitness_delta_min = -10 #params_movie['fitness_delta_min_patch']
#    Npeaks = 10 #params_movie['Npeaks']
#    traces = refnmf['C'] + refnmf['YrA'] #C_tot + YrA_tot
#    # TODO: todocument
#    idx_components, idx_components_bad, fitness_raw, fitness_delta, r_values =          estimate_components_quality(
#        traces, Y, refnmf['A'].all(), refnmf['C'], refnmf['b'], refnmf['f'], final_frate=final_frate, Npeaks=Npeaks, r_values_min=r_values_min,
#        fitness_min=fitness_min, fitness_delta_min=fitness_delta_min, return_all=True)
#    
#    print(('Keeping ' + str(len(idx_components)) +
#           ' and discarding  ' + str(len(idx_components_bad))))
#    
#    #%% Check out thresholded components:
#    pl.figure();
#    pl.subplot(1,2,1); pl.title('kept'); plot_contours(refnmf['A'].all().tocsc()[:, idx_components], img, thr=0.9); pl.axis('off')
#    pl.subplot(1,2,2); pl.title('bad'); plot_contours(refnmf['A'].all().tocsc()[:, idx_components_bad], img, thr=0.9); pl.axis('off')
    
    A1 = A1[:,  refnmf['idx_components']]
    nr = A1.shape[-1]
    
    #%%
    
masks = np.reshape(np.array(A1.todense()), (d1, d2, nr), order='F')
print(masks.shape)



#%% Loop thru all Files and match pairwise:
    
#curr_file = 'File001'
#exclude = ['File009', 'File010']

match_fn_base = 'matches_byfile_ref%s' % reference_file
if os.path.exists(os.path.join(metrics_dir, '%s.pkl' % match_fn_base)):
    with open(os.path.join(metrics_dir, '%s.pkl' % match_fn_base), 'rb') as f:
        all_matches = pkl.load(f)

else:
    all_matches = dict()

    for curr_file in file_names:
        
        if curr_file==roi_reference_file:
            continue
        
        print("Finding matches between reference %s and %s." % (roi_reference_file, curr_file))

        # In[16]:
        
        
        nmf_fn = [f for f in nmf_fns if curr_file in f][0]
        nmf = np.load(os.path.join(nmf_output_dir, nmf_fn))
        #print nmf.keys()


        # In[23]:
        
        nr = nmf['A'].all().shape[1]
        d1 = int(nmf['d1'])
        d2 = int(nmf['d2'])
        
        A2 = nmf['A'].all()
        
        if threshold is True:
            print("Keeping %i out of %i components." % (len(nmf['idx_components']), nr))
            A2 = A2[:,  nmf['idx_components']]
            nr = A2.shape[-1]
        #nA2 = np.array(np.sqrt(A2.power(2).sum(0)).T)
        #A2 = scipy.sparse.coo_matrix(A2 / nA2.T)
        masks2 = np.reshape(np.array(A2.todense()), (d1, d2, nr), order='F')
        #print masks2.shape

        
        #%% first transform A1 and A2 into binary masks
        
        
        M1 = np.zeros(A1.shape).astype('bool') #A1.astype('bool').toarray()
        
        M2 = np.zeros(A2.shape).astype('bool') #A2.astype('bool').toarray()


        # In[33]:
        
        K1 = A1.shape[-1]
        K2 = A2.shape[-1]
        print("K1", K1, "K2", K2)

        #%%
        s = ndimage.generate_binary_structure(2,2)
        
        for i in np.arange(0, max(K1,K2)):
            if i < K1:
                A_temp = A1.toarray()[:,i]
                M1[A_temp>params_thr['dist_maxthr']*max(A_temp),i] = True
                labeled, nr_objects = ndimage.label(np.reshape(M1[:,i], (d1,d2), order='F'), s)  # keep only the largest connected component
                sizes = ndimage.sum(np.reshape(M1[:,i], (d1,d2), order='F'), labeled, range(1,nr_objects+1)) 
                maxp = np.where(sizes==sizes.max())[0] + 1 
                max_index = np.zeros(nr_objects + 1, np.uint8)
                max_index[maxp] = 1
                BW = max_index[labeled]
                M1[:,i] = np.reshape(BW, M1[:,i].shape, order='F')
            if i < K2:
                A_temp = A2.toarray()[:,i];
                M2[A_temp>params_thr['dist_maxthr']*max(A_temp),i] = True
                labeled, nr_objects = ndimage.label(np.reshape(M2[:,i], (d1,d2), order='F'), s)  # keep only the largest connected component
                sizes = ndimage.sum(np.reshape(M2[:,i], (d1,d2), order='F'), labeled, range(1,nr_objects+1)) 
                maxp = np.where(sizes==sizes.max())[0] + 1 
                max_index = np.zeros(nr_objects + 1, np.uint8)
                max_index[maxp] = 1
                BW = max_index[labeled]
                M2[:,i] = np.reshape(BW, M2[:,i].shape, order='F')


        #%% determine distance matrix between M1 and M2
        D = np.zeros((K1,K2));
        for i in np.arange(0, K1):
            for j in np.arange(0, K2):
                
                overlap = float(np.count_nonzero(M1[:,i] & M2[:,j]))
                #print overlap
                totalarea = float(np.count_nonzero(M1[:,i] | M2[:,j]))
                #print totalarea
                smallestROI = min(np.count_nonzero(M1[:,i]),np.count_nonzero(M2[:,j]));
                #print smallestROI
                    
                D[i,j] = 1 - (overlap/totalarea)**params_thr['dist_exp']
        
                if overlap >= params_thr['dist_overlap_thr']*smallestROI:
                    #print('Too small!')
                    D[i,j] = 0   
                

        #%% Set illegal matches (distance vals greater than dist_thr):

        D[D>params_thr['dist_thr']] = np.inf #1E100 #np.nan #1E9

        # In[125]:
        
        pl.figure()
        pl.imshow(D)
        pl.colorbar()
        pl.savefig(os.path.join(metrics_dir_figs, 'distancematrix_%s.png' % curr_file))
        pl.close()

        #%%
        #match_1, match_2 = scipy.optimize.linear_sum_assignment(D)

        matches = minimumWeightMatching(D)  # Use modified linear_sum_assignment to allow np.inf

        print("Found %i ROI matches in %s" % (len(matches), curr_file))

        #matched_ROIs = [match_1, match_2]
        #nonmatched_1 = np.setdiff1d(np.arange(0, K1), match_1)
        #nonmatched_2 = np.setdiff1d(np.arange(0, K2), match_2)
        #
        #print len(match_1)

        #%% Save overlap of REF with curr-file matches:
        
        pl.figure()
        pl.imshow(img, cmap='gray')
        #pl.subplot(1,2,1); pl.imshow(img, cmap='gray')
        #pl.subplot(1,2,2); pl.imshow(img, cmap='gray')
        #for ridx,(roi1,roi2) in enumerate(zip(match_1, match_2)):
        for ridx,match in enumerate(matches):
            roi1=match[0]; roi2=match[1]
            masktmp1 = masks[:,:,roi1]; masktmp2 = masks2[:,:,roi2]
            msk1 = masktmp1.copy(); msk2 = masktmp2.copy()  
            msk1[msk1==0] = np.nan; msk2[msk2==0] = np.nan
            
            #pl.subplot(1,2,1); pl.title('match1')
            #cs1a = pl.contour(msk1, interpolation='None', alpha=0.3, cmap=pl.cm.Blues_r)
            #cs1b = pl.contour(cs1a, levels=cs1a.levels[::4], interpolation='None', alpha=0.3, cmap=pl.cm.Blues_r)
            pl.imshow(msk1, interpolation='None', alpha=0.3, cmap=pl.cm.Blues_r)
            pl.clim(masktmp1.max()*0.7, masktmp1.max())
            [ys, xs] = np.where(masktmp1>0)
            pl.text(xs[int(round(len(xs)/4))], ys[int(round(len(ys)/4))], str(roi1), weight='bold')
            pl.axis('off')
            
            #pl.subplot(1,2,2); pl.title('match2')
            #pl.contour(msk2, interpolation='None', alpha=0.3, cmap=pl.cm.Reds)
            #cs2a = pl.contour(msk2, interpolation='None', alpha=0.3, cmap=pl.cm.Reds_r)
            #cs2b = pl.contour(cs2a, levels=cs2a.levels[::4], interpolation='None', alpha=0.3, cmap=pl.cm.Reds_r)
            pl.imshow(msk2, interpolation='None', alpha=0.3, cmap=pl.cm.Reds_r)
            pl.clim(masktmp2.max()*0.7, masktmp2.max())
            [ys, xs] = np.where(masktmp2>0)
            pl.text(xs[int(round(len(xs)/4))], ys[int(round(len(ys)/4))], str(roi2), weight='bold')
            pl.axis('off')
        
        pl.savefig(os.path.join(metrics_dir_figs, 'matches_%s_%s.png' % (roi_reference_file, curr_file)))
        #pl.close()
        
        #%% Store matches for file:
        all_matches[curr_file] = matches
        
    #%% Save matches:
        
    match_fn_base = 'matches_byfile_ref%s' % roi_reference_file
    with open(os.path.join(metrics_dir, '%s.pkl' % match_fn_base), 'wb') as f:
        pkl.dump(all_matches, f, protocol=pkl.HIGHEST_PROTOCOL)

    with open(os.path.join(metrics_dir, '%s.json' % match_fn_base), 'w') as f:
        json.dump(all_matches, f, indent=4, sort_keys=True)
        
#%% Find intersection of all matches with reference:
       
#rois_to_keep = [r for r in range(nr)]
rois_to_keep = []
for curr_file in all_matches.keys():
    if curr_file in exclude:
        continue
    
    matchable_refs = [r[0] for r in all_matches[curr_file]]
    if len(rois_to_keep)==0:
        rois_to_keep = np.copy(matchable_refs)
        
    rois_to_keep = list(set(rois_to_keep) & set(matchable_refs))

print("Found %i common ROIs matching reference." % len(rois_to_keep))

#%%
matchedROIs = dict()

for curr_file in file_names: #all_matches.keys():
    
    if curr_file in exclude:
        continue
    
    if curr_file==roi_reference_file:
        matchedROIs[curr_file] = rois_to_keep
    else:
        matchedROIs[curr_file] = [all_matches[curr_file][[i[0] for i in all_matches[curr_file]].index(r)][1] for r in rois_to_keep]

matchedrois_fn_base = 'matchedROIs_ref%s' % roi_reference_file
print("Saving matches to: %s" % os.path.join(metrics_dir, matchedrois_fn_base))
with open(os.path.join(metrics_dir, '%s.pkl' % matchedrois_fn_base), 'wb') as f:
    pkl.dump(matchedROIs, f, protocol=pkl.HIGHEST_PROTOCOL)
with open(os.path.join(metrics_dir, '%s.json' % matchedrois_fn_base), 'w') as f:
    json.dump(matchedROIs, f, indent=4, sort_keys=True)
    
    
#%% Draw all matching ROIs on top of e/o:
from matplotlib import gridspec

cmaptype='jet'
colormap = pl.get_cmap(cmaptype)
#nrois = len(matchedROIs[reference_file])
colorvals = colormap(np.linspace(0, 1, len(file_names))) #get_spaced_colors(nrois)
colorvals[:,3] *= 0.5

fig = pl.figure(figsize=(8, 6)) 
gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1]) 
gs.update(wspace=0.05, hspace=0.05)
ax1 = pl.subplot(gs[0])

ax1.imshow(img, cmap='gray')
pl.axis('equal')
pl.axis('off')
blank = np.ones(img.shape)*np.nan
#%%
for fidx,curr_file in enumerate(sorted(matchedROIs.keys(), key=natural_keys)):
    
    nmf_fn = [f for f in nmf_fns if curr_file in f][0]
    nmf = np.load(os.path.join(nmf_output_dir, nmf_fn))

    nr = nmf['A'].all().shape[1]
    d1 = int(nmf['d1'])
    d2 = int(nmf['d2'])
    dims = (d1, d2)

    x, y = np.mgrid[0:d1:1, 0:d2:1]
    
    A = nmf['A'].all()
    if threshold is True:
        #print("Keeping %i out of %i components." % (len(nmf['idx_components']), nr))
        A = A[:,  nmf['idx_components']]
        nr = A.shape[-1]
            
#     coors = get_contours(A, dims, thr=0.9)
#     cc1 = [[l[0] for l in n['coordinates']] for n in coors]
#     cc2 = [[l[1] for l in n['coordinates']] for n in coors]
#     coords = [[(cx,cy) for cx,cy in zip(cc1[n], cc2[n])] for n in range(len(cc1))]
#  
    masks = np.reshape(np.array(A.todense()), (d1, d2, nr), order='F')
    curr_rois = matchedROIs[curr_file]
    A = np.array(A.todense()) 
    
    for roi in curr_rois:
        #print roi
        # compute the cumulative sum of the energy of the Ath component that 
        # has been ordered from least to highest:
        indx = np.argsort(A[:,roi], axis=None)[::-1]
        cumEn = np.cumsum(A[:,roi].flatten()[indx]**2)
        cumEn /= cumEn[-1] # normalize
        Bvec = np.zeros(d1*d2)
        Bvec[indx] = cumEn
        Bmat = np.reshape(Bvec, (d1,d2), order='F')
        currcolor = (colorvals[fidx][0], colorvals[fidx][1], colorvals[fidx][2], 0.5)
        cs = pl.contour(y, x, Bmat, [0.9], colors=[colorvals[fidx]]) #, cmap=colormap)
#pl.axis('equal')
#pl.axis('off')

pl.suptitle(roi_id)

pl.savefig(os.path.join(metrics_dir, 'matchedROIs_reference_%s_rois.png' % roi_reference_file))
        

nfiles = len(matchedROIs.keys())
gap = 1
#%%
#pl.figure()
ax2 = pl.subplot(gs[1])
interval = np.arange(0, 1, 1/nfiles)
for fidx,curr_file in enumerate(sorted(matchedROIs.keys(), key=natural_keys)):
    ax2.plot(1, interval[fidx], c=colorvals[fidx], marker='.', markersize=20)
    pl.text(1.1, interval[fidx], str(curr_file), fontsize=12)
    pl.xlim([0.95, 2])
pl.axis('equal')
pl.axis('off')


#%%
pl.suptitle(roi_id)
pl.savefig(os.path.join(metrics_dir, '%s.png' % matchedrois_fn_base))

#%%
if 'subsets' not in roiparams.keys():
    roiparams['subsets'] = [metrics_dir]
elif metrics_dir not in roiparams['subsets']:
    roiparams['subsets'].append(metrics_dir)

with open(os.path.join(roiparams_path), 'w') as f:
    json.dump(roiparams, f, indent=4)

#%% Save threshold params:
    
with open(os.path.join(metrics_dir, 'threshold_params.json'), 'w') as f:
    json.dump(params_thr, f, indent=4)