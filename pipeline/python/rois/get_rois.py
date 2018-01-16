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
import traceback
import tifffile as tf
import pylab as pl
import numpy as np
from pipeline.python.utils import natural_keys, hash_file, write_dict_to_json
from pipeline.python.rois import extract_rois_caiman as rcm
from pipeline.python.rois import coregister_rois as reg
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
def run_roi_evaluation(session_dir, src_roi_id, roi_eval_dir, roi_type='caiman2D', evalparams=None):
    
    session = os.path.split(session_dir)[1]
    roidict_path = os.path.join(session_dir, 'ROIs', 'rids_%s.json' % session)
    try:
        with open(roidict_path, 'r') as f:
            roidict = json.load(f)
        src_roi_key = [k for k in roidict if src_roi_id in k][0]
        src_rid = roidict[src_roi_key]
        roi_source_dir = src_rid['DST']
        print "Evaluating ROIs from source:", roi_source_dir
    except Exception as e:
        print "-- ERROR: unable to open source ROI dict. ---------------------"
        traceback.print_exc()
        print "---------------------------------------------------------------"
    
    if roi_type == 'caiman2D':
        src_nmf_dir = os.path.join(roi_source_dir, 'nmfoutput')
        source_nmf_paths = sorted([os.path.join(src_nmf_dir, n) for n in os.listdir(src_nmf_dir) if n.endswith('npz')], key=natural_keys) # Load nmf files
        
        src_mmap_dir = src_rid['PARAMS']['mmap_source']
        mem_paths = sorted([os.path.join(src_mmap_dir, f) for f in os.listdir(src_mmap_dir) if f.endswith('mmap')], key=natural_keys)
        src_file_list = []
        for fn in filenames:
            match_nmf = [f for f in source_nmf_paths if fn in f][0]
            match_mmap = [f for f in mem_paths if fn in f][0]
            src_file_list.append((match_mmap, match_nmf))
                    
        roi_idx_filepath = os.path.join(roi_eval_dir, 'roi_idxs_to_keep.hdf5')
        roifile = h5py.File(roi_idx_filepath, 'w')
        for k in evalparams.keys():
            roifile.attrs[k] = evalparams[k]
        roifile.attrs['creation_date'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        try:
            #idxs_to_keep = dict()
            for src_file in src_file_list:
                curr_mmap_path = src_file[0]
                curr_nmfout_path = src_file[1]
                
                curr_file = str(re.search('File(\d{3})', curr_nmfout_path).group(0))
                
                good, bad, snr_vals, r_vals = evaluate_rois_nmf(curr_mmap_path, curr_nmfout_path, 
                                                                      evalparams, dview=dview,
                                                                      eval_outdir=roi_eval_dir, save_output=True)
                #idxs_to_keep[curr_file] = good
                        
                rois = roifile.create_dataset('/'.join([curr_file, 'idxs_to_keep']), good.shape, good.dtype)
                rois[...] = good
                rois.attrs['tiff_source'] = curr_mmap_path
                rois.attrs['roi_source'] = curr_nmfout_path
            
            roifile.close()
        except Exception as e:
            print "--- Error evaulating ROIs. Curr file: %s ---" % src_file
            traceback.print_exc()
            print "-----------------------------------------------------------"
        finally:
            roifile.close()

        
#        with open(roi_idx_filepath, 'w') as f:
#            json.dump(idxs_to_keep, f, indent=4, sort_keys=True)
            
    print "Finished ROI evaluation step. ROI eval info saved to:"
    print roi_idx_filepath
    
    return roi_idx_filepath

#%%

parser = optparse.OptionParser()

# PATH opts:
parser.add_option('-R', '--root', action='store', dest='rootdir', default='/nas/volume1/2photon/data', help='data root dir (root project dir containing all animalids) [default: /nas/volume1/2photon/data, /n/coxfs01/2pdata if --slurm]')
parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')
parser.add_option('-S', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID')

parser.add_option('--default', action='store_true', dest='default', default='store_false', help="Use all DEFAULT params, for params not specified by user (no interactive)")
parser.add_option('--slurm', action='store_true', dest='slurm', default=False, help="set if running as SLURM job on Odyssey")
parser.add_option('-r', '--roi-id', action='store', dest='roi_id', default='', help="ROI ID for rid param set to use (created with set_roi_params.py, e.g., rois001, rois005, etc.)")

# Eval opts:
parser.add_option('--good', action="store_true",
                  dest="keep_good_rois", default=False, help="Set flag to only keep good components (useful for avoiding computing massive ROI sets)")
parser.add_option('--max', action="store_true",
                  dest="use_max_nrois", default=False, help="Set flag to use file with max N components (instead of reference file) [default uses reference]")
    
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
roi_id = options.roi_id
slurm = options.slurm
auto = options.default

if slurm is True:
    if 'coxfs01' not in rootdir:
        rootdir = '/n/coxfs01/2p-data'

keep_good_rois = options.keep_good_rois
use_max_nrois = options.use_max_nrois

dist_maxthr = options.dist_maxthr
dist_exp = options.dist_exp
dist_thr = options.dist_thr
dist_overlap_thr = options.dist_overlap_thr

#%%
#rootdir = '/nas/volume1/2photon/data'
#animalid = 'JR063' #'JR063'
#session = '20171128_JR063' #'20171128_JR063'
#roi_id = 'rois002'
#slurm = False
#auto = False
#
#keep_good_rois = True       # Only keep "good" ROIs from a given set (TODO:  add eval for ROIs -- right now, only have eval for NMF and coregister)
#
## COREG-SPECIFIC opts:
#use_max_nrois = True        # Use file which has the max N ROIs as reference (alternative is to use reference file)
#dist_maxthr = 0.1
#dist_exp = 0.1
#dist_thr = 0.5
#dist_overlap_thr = 0.8

#%%
# =============================================================================
# Load specified ROI-ID parameter set:
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
    
    
    coreg_opts = ['-R', rootdir, '-i', animalid, '-S', session, '-r', roi_id,
                  '-t', params_thr['dist_maxthr'], '-n', params_thr['dist_exp'],
                  '-d', params_thr['dist_thr'], '-o', params_thr['dist_overlap_thr']]
    if params_thr['filter_type'] == 'max':
        coreg_opts.extend(['--max'])
    if params_thr['keep_good_rois'] is True:
        coreg_opts.extend(['--good'])
    
    ref_rois, params_thr, coreg_outpath = reg.run_coregistration(coreg_opts)
    
    #%% Re-evaluate ROIs to less stringest thresholds
    if len(ref_rois) == 0:
                    
        roi_eval_outdir = os.path.join(RID['DST'], 'src_evaluation')        
        if not os.path.exists(roi_eval_outdir):
            os.makedirs(roi_eval_outdir)
            
        #% Load eval params from src: 
        roi_source_dir = RID['PARAMS']['options']['source']['roi_dir']
        src_roi_id = RID['PARAMS']['options']['source']['roi_id']
        if keep_good_rois is True:
            src_evalparams = params_thr['eval']
            evalparams = src_evalparams.copy()
            src_rid = roidict[RID['PARAMS']['options']['source']['roi_id']]
            evalparams['gSig'] = src_rid['PARAMS']['options']['extraction']['gSig'][0] # Need gSig to run NMF roi evaluation

        # ================================================================
        evalparams['min_SNR'] = 1.5
        evalparams['rval_thr'] = 0.7
        # ================================================================
        
        print "-----------------------------------------------------------"
        print "Evaluating NMF components with less stringent eval params..."
        for k in evalparams.keys():
            print k, ':', evalparams[k]
        print "-----------------------------------------------------------"

        #%start a cluster for parallel processing
        try:
            dview.terminate() # stop it if it was running
        except:
            pass
        c, dview, n_processes = cm.cluster.setup_cluster(backend='local', # use this one
                                                         n_processes=None,  # number of process to use, reduce if out of mem
                                                         single_thread = False)
                 
        roi_idx_filepath = run_roi_evaluation(session_dir, src_roi_id, roi_eval_outdir, roi_type='caiman2D', evalparams=evalparams)

        
        #%% Re-run coregistration with new ROI idxs:
            
        coreg_output_dir = os.path.join(RID['DST'], 'reeval_coreg_results')
        
        coreg_opts.extend(['--roipath=%s' % roi_idx_filepath])
        coreg_opts.extend(['-O', coreg_output_dir])
        
        ref_rois, params_thr, coreg_outpath = reg.run_coregistration(coreg_opts)

        #% Save new evaluation info:
            
        print("Found %i common ROIs matching reference." % len(ref_rois))
        
        # Overwrite SRC eval info to current coreg dir:
        params_thr['eval'] = evalparams
        with open(os.path.join(coreg_output_dir, 'coreg_params.json'), 'w') as f:
            json.dump(params_thr, f, indent=4, sort_keys=True)

    format_roi_output = True

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
def format_rois_nmf(nmf_filepath, roiparams, kept_rois=None, coreg_rois=None):
    
    nmf = np.load(nmf_filepath)
    nr = nmf['A'].all().shape[1]

    d1 = int(nmf['dims'][0])
    d2 = int(nmf['dims'][1])
    if len(nmf['dims']) > 2:
        is_3D = True
        d3 = int(nmf['dims'][2])
        dims = (d1, d2)
    else:
        is_3D = False
        dims = (d1, d2)
    
    A = nmf['A'].all()
    A2 = A.copy()
    A2.data **= 2
    nA2 = np.sqrt(np.array(A2.sum(axis=0))).squeeze()
    rA = A * spdiags(old_div(1, nA2), 0, nr, nr)
    rA = rA.todense()
    nr = A.shape[1]
    
    # Create masks:
    if is_3D:
        masks = np.reshape(np.array(rA), (d1, d2, d3, nr), order='F')
        if roiparams['keep_good_rois'] is True:
            if kept_rois is None:
                kept_rois = nmf['idx_components']
            masks = masks[:,:,:,kept_rois]
        if coreg_rois is not None:
            masks = masks[:,:,:,coreg_rois]
    else:
        masks = np.reshape(np.array(rA), (d1, d2, nr), order='F')
        if roiparams['keep_good_rois'] is True:
            if kept_rois is None:
                kept_rois = nmf['idx_components']
            masks = masks[:,:,kept_rois]
        if coreg_rois is not None:
            masks = masks[:,:,coreg_rois]
    
    #print("Keeping %i out of %i ROIs." % (len(kept_rois), nr))
    
    # Get center of mass for each ROI:
    coors = get_contours(A, dims, thr=0.9)
    if roiparams['keep_good_rois'] is True:
        if kept_rois is None:
            kept_rois = nmf['idx_components']
        coors = [coors[i] for i in kept_rois]
    if coreg_rois is not None:
        coors = [coors[i] for i in coreg_rois]
        
    #cc1 = [[l[0] for l in n['coordinates']] for n in coors]
    #cc2 = [[l[1] for l in n['coordinates']] for n in coors]
    #coords = [[(x,y) for x,y in zip(cc1[n], cc2[n])] for n in range(len(cc1))]
    #coms = np.array([np.array(n) for n in coords])
    
    return masks, coors, is_3D

#%%
# =============================================================================
# Format ROI output to standard, if applicable:
# =============================================================================

if format_roi_output is True :
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

    try:
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
                
                masks, coord_info, is_3D= format_rois_nmf(nmf_filepath, roiparams)
                maskfile.attrs['is_3D'] = is_3D
                kept_idxs = nmf['idx_components']
                
                print('Mask array:', masks.shape)
                currmasks = filegrp.create_dataset('masks', masks.shape, masks.dtype)
                currmasks[...] = masks
                if roiparams['keep_good_rois'] is True:
                    currmasks.attrs['nrois'] = len(kept_idxs)
                    currmasks.attrs['roi_idxs'] = kept_idxs
                else:
                    currmasks.attrs['nrois'] = masks.shape[-1]
                
                coms = np.array([r['CoM'] for r in coord_info])
                currcoms = filegrp.create_dataset('coms', coms.shape, coms.dtype)
                currcoms[...] = coms
                
                for ridx, roi in enumerate(coord_info):
                    curr_roi = filegrp.create_dataset('/'.join(['coords', 'roi%04d' % ridx]), roi['coordinates'].shape, roi['coordinates'].dtype)
                    curr_roi[...] = roi['coordinates']
                    curr_roi.attrs['roi_source'] = nmf_filepath
                    curr_roi.attrs['idx_in_src'] = roi['neuron_id']
                    curr_roi.attrs['idx_in_kept'] = kept_idxs[ridx]
                    
                
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
                coreg_info = h5py.File(coreg_outpath, 'r')
                filenames = [str(i) for i in coreg_info.keys()]
                filenames = sorted(filenames, key=natural_keys)
                
                # Load universal match info:
                matchedrois_fn_base = 'coregistered_r%s' % str(params_thr['coreg_ref_file'])
                with open(os.path.join(coreg_output_dir, '%s.json' % matchedrois_fn_base), 'r') as f:
                    matchedROIs = json.load(f)
                
                idxs_to_keep = dict()
                for curr_file in filenames:
                    
                    idxs_to_keep[curr_file] = coreg_info[curr_file]['roi_idxs']
                    nmf_filepath = [n for n in source_nmf_paths if curr_file in n][0]
                    nmf = np.load(nmf_filepath)
                    img = nmf['Av']
                    
                    print "Creating ROI masks for %s" % curr_file
                    
                    # Create group for current file:
                    if curr_file not in maskfile.keys():
                        filegrp = maskfile.create_group(curr_file)
                    else:
                        filegrp = maskfile[curr_file]
                    
                    # Get masks:
                    masks, coord_info, is_3D = format_rois_nmf(nmf_filepath, roiparams, kept_rois=idxs_to_keep[curr_file], coreg_rois=matchedROIs[curr_file])
                    maskfile.attrs['is_3D'] = is_3D
                    final_rois = matchedROIs[curr_file]
                    
                    print('Mask array:', masks.shape)
                    currmasks = filegrp.create_dataset('masks', masks.shape, masks.dtype)
                    currmasks[...] = masks
                    if roiparams['keep_good_rois'] is True:
                        currmasks.attrs['nrois'] = len(final_rois) #len(kept_idxs)
                        currmasks.attrs['roi_idxs'] = final_rois
                    else:
                        currmasks.attrs['nrois'] = masks.shape[-1]
                    
                    print "Saving coms..."
                    coms = np.array([r['CoM'] for r in coord_info])
                    currcoms = filegrp.create_dataset('coms', coms.shape, coms.dtype)
                    currcoms[...] = coms
                    
                    print "Saving ROI info..."
                    for ridx, roi in enumerate(coord_info):
                        curr_roi = filegrp.create_dataset('/'.join(['coords', 'roi%04d' % ridx]), roi['coordinates'].shape, roi['coordinates'].dtype)
                        curr_roi[...] = roi['coordinates']
                        curr_roi.attrs['roi_source'] = nmf_filepath
                        curr_roi.attrs['idx_in_src'] = roi['neuron_id'] - 1 # 0-indexed
                        curr_roi.attrs['idx_in_kept'] = idxs_to_keep[curr_file][ridx]
                        curr_roi.attrs['idx_in_coreg'] = final_rois[ridx]
                        
                    zproj = filegrp.create_dataset('avg_img', img.shape, img.dtype)
                    zproj[...] = img
                    
                    # Plot figure with ROI masks:
                    vmax = np.percentile(img, 98)
                    pl.figure()
                    pl.imshow(img, interpolation='None', cmap=pl.cm.gray, vmax=vmax)
                    for roi in range(len(final_rois)):
                        masktmp = masks[:,:,roi]
                        msk = masktmp.copy() 
                        msk[msk==0] = np.nan
                        pl.imshow(msk, interpolation='None', alpha=0.3, cmap=pl.cm.hot)
                        [ys, xs] = np.where(masktmp>0)
                        pl.text(xs[int(round(len(xs)/4))], ys[int(round(len(ys)/4))], str(final_rois[roi]), weight='bold')
                        pl.axis('off')
                    pl.colorbar()
                    pl.tight_layout()
                    
                    # Save image:
                    imname = '%s_%s_%s_masks.png' % (roi_id, rid_hash, curr_file)
                    print(imname) 
                    pl.savefig(os.path.join(rid_figdir, imname))
                    pl.close()
        else:
            # do sth ?
            print "Formatting for roi_type %s unknown..." % roi_type
            
    except Exception as e:
        print "--ERROR: formatting ROIs to standard! -------------------------"
        traceback.print_exc()
        print "Unable to format ROIs for type: %s" % roi_type
        print "ABORTING."
        print "---------------------------------------------------------------"
    finally:
        maskfile.close()



#%%
print "*************************************************"
print "FINISHED EXTRACTING ROIs!"
print "*************************************************"
