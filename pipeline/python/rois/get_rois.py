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
import tifffile as tf
import pylab as pl
import numpy as np
from pipeline.python.utils import natural_keys, hash_file, write_dict_to_json
from pipeline.python.rois import extract_rois_caiman as rcm
from pipeline.python.set_roi_params import post_rid_cleanup
from scipy.sparse import spdiags, issparse
from caiman.utils.visualization import get_contours
from past.utils import old_div
import time

def timer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    formatted_time = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)
    return formatted_time
   
pp = pprint.PrettyPrinter(indent=4)

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
animalid = 'JR066' #'JR063'
session = '20180103_JR066_test' #'20171128_JR063'
roi_id = 'rois001'
slurm = False
auto = False


dist_maxthr = 0.1
dis_exp = 0.1
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
    ref_channel = mcmetrics.attrs['ref_channel']
    ref_file = mcmetrics.attrs['ref_file']
else:
    acquisition_dir = os.path.join(session_dir, acquisition)
    info = get_source_info(acquisition_dir, run, process_id)
    ref_channel = info['ref_channel']
    ref_file = info['ref_file']

if len(exclude_str) > 0:
    filenames = [f for f in filenames if int(f[4:]) not in [int(x) for x in exclude_str.split(',')]]
    excluded_tiffs = ['File%03d' % int(fidx) for fidx in exclude_str.split(',')]
    
#%%
# =============================================================================
# Extract ROIs using specified method:
# =============================================================================
format_roi_output = False
t_start = time.time()
if roi_type == 'caiman2D':
        
    roi_opts = ['-R', rootdir, '-i', animalid, '-S', session, '-A', acquisition, '-r', run, '-p', rid_hash]
    if slurm is True:
        roi_opts.extend(['--slurm'])
    if len(exclude_str) > 0:
        roi_opts.extend(['-x', exclude_str])
        
    RID = rcm.extract_cnmf_rois(roi_opts)
    
    # Clean up tmp RID files:
    session_dir = os.path.join(rootdir, animalid, session)
    post_rid_cleanup(session_dir, rid_hash)
    
    format_roi_output = True
    
elif roi_type == 'blob_detector':
    # Do some other stuff
    print "blobs"

elif 'manual' in roi_type:
    # Do some matlab-loading stuff ?
    print "manual"

else:
    print "ERROR: %s -- roi type not known..." % roi_type

roi_dur = timer(t_start, time.time())
print "RID %s -- Finished ROI extration!" % rid_hash
print "Total duration was:", roi_dur

#%%
# =============================================================================
# Format ROI output to standard, if applicable:
# =============================================================================
keep_good_rois = True       # Only keep "good" ROIs from a given set
roiparams = dict()
rid_dir = RID['DST']
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
            nmf = np.load(os.path.join(nmf_output_dir, nmf_fn))
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
            
            # Create masks:
            masks = np.reshape(np.array(rA), (d1, d2, nr), order='F')
            if keep_good_rois is True:
                kept = nmf['idx_components']
                masks = masks[:,:,kept]
                print("Keeping %i out of %i ROIs." % (len(kept), nr))
                nrois = len(kept)
            else:
                nrois = nr
            print('Mask array:', masks.shape)
            currmasks = filegrp.create_dataset('masks', masks.shape, masks.dtype)
            currmasks[...] = masks
            currmasks.attrs['nrois'] = nrois
            currmasks.attrs['roi_idxs'] = kept
            
                    
            # Get center of mass for each ROI:
            coors = get_contours(A, dims, thr=0.9)
            if keep_good_rois is True:
                coors = [coors[i] for i in kept]
            cc1 = [[l[0] for l in n['coordinates']] for n in coors]
            cc2 = [[l[1] for l in n['coordinates']] for n in coors]
            coords = [[(x,y) for x,y in zip(cc1[n], cc2[n])] for n in range(len(cc1))] 
            com = np.array([list(n['CoM']) for n in coors])
            currcoms = filegrp.create_dataset('com', com.shape, com.dtype)
            currcoms[...] = com
            
            # Plot figure with ROI masks:
            img = nmf['Av']
            vmax = np.percentile(img, 98)
            pl.figure()
            pl.imshow(img, interpolation='None', cmap=pl.cm.gray, vmax=vmax)
            for roi in range(nrois):
                masktmp = masks[:,:,roi]
                msk = masktmp.copy() 
                msk[msk==0] = np.nan
                pl.imshow(msk, interpolation='None', alpha=0.3, cmap=pl.cm.hot)
                [ys, xs] = np.where(masktmp>0)
                pl.text(xs[int(round(len(xs)/4))], ys[int(round(len(ys)/4))], str(kept[roi]), weight='bold')
                pl.axis('off')
            pl.colorbar()
            pl.tight_layout()
            
            # Save image:
            imname = '%s_%s_%s_masks.png' % (roi_id, rid_hash, filenames[fidx])
            print(imname) 
            pl.savefig(os.path.join(rid_figdir, imname))
            pl.close()
                    

#%% Save ROI params info:
if roi_type == 'caiman2D':
    roiparams['eval'] = RID['PARAMS']['options']['eval']
roiparams['keep_good_rois'] = keep_good_rois
roiparams['excluded_tiffs'] = excluded_tiffs
roiparams['roi_type'] = roi_type
roiparams['roi_id'] = roi_id
roiparams['rid_hash'] = rid_hash

roiparams_filepath = os.path.join(rid_dir, 'roiparams.json') # % (str(roi_id), str(rid_hash)))
with open(roiparams_filepath, 'w') as f:
    write_dict_to_json(roiparams, roiparams_filepath)
    
maskfile.close()

#%%
# =============================================================================
# Coregister ROIs, if applicable:
# =============================================================================

# options:
use_max_nrois = True        # Use file which has the max N ROIs as reference
params_thr = dict()

if coregister is True:

    params_thr['dist_maxthr'] = dist_maxthr            # threshold for turning spatial components into binary masks (default: 0.1)
    params_thr['dist_exp'] = dist_exp                  # power n for distance between masked components: dist = 1 - (and(m1,m2)/or(m1,m2))^n (default: 1)
    params_thr['dist_thr'] = dist_thr                  # threshold for setting a distance to infinity. (default: 0.5)
    params_thr['dist_overlap_thr'] = dist_overlap_thr  # overlap threshold for detecting if one ROI is a subset of another (default: 0.8)
    if use_max_nrois is True:
        params_thr['filter_type'] = 'max'
    else:
        params_thr['filter_type'] = 'ref'

# Save ROI params info for coreg:
roiparams['coreg'] = params_thr



#%%
print "*************************************************"
print "FINISHED EXTRACTING ROIs!"
print "*************************************************"
