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
import re
import sys
import h5py
import json
import datetime
import optparse
import pprint
import time
import traceback
import scipy
import skimage
import pylab as pl
import numpy as np
from pipeline.python.utils import natural_keys, write_dict_to_json, save_sparse_hdf5, print_elapsed_time
from pipeline.python.rois import caiman2D as rcm
from pipeline.python.rois import coregister_rois as reg
from pipeline.python.set_roi_params import post_rid_cleanup
from pipeline.python.rois.utils import load_RID, get_source_paths, check_mc_evaluation
from scipy.sparse import spdiags
from caiman.utils.visualization import get_contours
from past.utils import old_div

def timer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    formatted_time = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)
    return formatted_time
   
pp = pprint.PrettyPrinter(indent=4)

#%%
def load_eval_results(src_roi_dir, eval_key, auto=False):
    src_eval_filepath = None
    src_eval = None
    try:
        print "-----------------------------------------------------------"
        print "Loading evaluation results for src roi set"
        # Load eval info:
        src_eval_filepath = os.path.join(src_roi_dir, 'evaluation', 'evaluation_%s' % eval_key, 'evaluation_results_%s.hdf5' % eval_key)
        assert os.path.exists(src_eval_filepath), "Specfied EVAL src file does not exist!\n%s" % src_eval_filepath
        src_eval = h5py.File(src_eval_filepath, 'r')
    except Exception as e:
        print "Error loading specified eval file:\n%s" % src_eval_filepath
        traceback.print_exc()
        print "-----------------------------------------------------------"
        try:
            evaldict_filepath = os.path.join(src_roi_dir, 'evaluation', 'evaluation_info.json')
            with open(evaldict_filepath, 'r') as f:
                evaldict = json.load(f)
            eval_list = sorted(evaldict.keys(), key=natural_keys)
            print "Found evaluation keys:"
            if auto is False:
                while True:
                    if len(eval_list) > 1:
                        for eidx, ekey in enumerate(eval_list):
                            print eidx, ekey
                            eval_select_idx = input('Select IDX of evaluation key to view: ')
                    else:
                        eval_select_idx = 0
                        print "Only 1 evaluation set found: %s" % eval_list[eval_select_idx]
                    pp.pprint(evaldict[eval_list[eval_select_idx]])
                    confirm_eval = raw_input('Enter <Y> to use this eval set, or <n> to return: ')
                    if confirm_eval == 'Y':
                        eval_key = eval_list[eval_select_idx].split('evaluation_')[-1]
                        print "Using key: %s" % eval_key
                        break
            else:
                print "Auto is ON, using most recent evaluation set: %s" % eval_key
                eval_key = eval_list[-1].split('evaluation_')[-1]
                pp.pprint(evaldict[eval_list[-1]])
            
            src_eval_filepath = os.path.join(src_roi_dir, 'evaluation', 'evaluation_%s' % eval_key, 'evaluation_results_%s.hdf5' % eval_key)
            src_eval = h5py.File(src_eval_filepath, 'r')
        except Exception as e:
            print "ERROR: Can't load source evaluation file - %s" % eval_key
            traceback.print_exc()
            print "Aborting..."
            print "-----------------------------------------------------------"
            
    return src_eval, src_eval_filepath
    
#%%
def format_rois_nmf(nmf_filepath, roiparams, zproj_type='mean', pass_rois=None, coreg_rois=None):
    """
    Get shaped masks (filtered, if specified) and coordinate list for ROIs.
    Also return original indices of final ROIs (0-indexed).
    """
    nmf = np.load(nmf_filepath)

    d1 = int(nmf['dims'][0])
    d2 = int(nmf['dims'][1])
    if len(nmf['dims']) > 2:
        is_3D = True
        d3 = int(nmf['dims'][2])
        dims = (d1, d2)
    else:
        is_3D = False
        dims = (d1, d2)
        
    # Get zprojected image:
    if zproj_type == 'corr':
        img = nmf['Cn']
    else:
        img = nmf['Av']

    A = nmf['A'].all().tocsc()
    b = nmf['b']
    C = nmf['C']
    f = nmf['f']
#    A2 = A.copy()
#    A2.data **= 2
#    nA2 = np.sqrt(np.array(A2.sum(axis=0))).squeeze()
#    rA = A * spdiags(old_div(1, nA2), 0, nr, nr)
#    rA = rA.todense()
#    nr = A.shape[1]

    nr = np.shape(A)[-1]
    nb = b.shape[1]
    
    # Keep background as components:
    Ab = scipy.sparse.hstack((A, b)).tocsc()
    Cf = np.vstack((C, f))

    A2 = Ab.copy()
    A2.data **= 2.
    nA2 = np.sqrt(np.array(A2.sum(axis=0))).squeeze()
    
    # normalize by each pixel's contribution to spatial component:
    rA = Ab * spdiags(old_div(1., nA2), 0., nr+nb, nr+nb)
    AB = rA.todense()
    
    # Get center of mass for each ROI:
    coors = get_contours(A, dims, thr=0.9)
    roi_idxs = np.arange(0, nr)

    # Create masks:
    if is_3D:
        masks = np.reshape(np.array(AB), (d1, d2, d3, nr+nb), order='F')
    else:
        masks = np.reshape(np.array(AB), (d1, d2, nr+nb), order='F')
        
    # Filter coors and masks:
    if roiparams['keep_good_rois'] is True:
        if pass_rois is None:
            pass_rois = nmf['idx_components']    # Get idxs of ROIs that "pass" evaluation
        roi_idxs = roi_idxs[pass_rois]           # Update ROI index list
        
    if coreg_rois is not None:                   # coreg_rois = indices into either "pass" rois (if keep_good_rois==True) or just the org src 
        if not isinstance(coreg_rois[0], int):
            coreg_rois = [int(c) for c in coreg_rois]
        roi_idxs = roi_idxs[coreg_rois]   

    roi_idxs = np.append(roi_idxs, nr)           # Append the "background component" to the ROI list:
    
    # Only return selected masks and coord info:
    if is_3D: 
        final_masks = np.empty((d1, d2, d3, len(roi_idxs)))
    else:
        final_masks = np.empty((d1, d2, len(roi_idxs)))
    final_rA = scipy.sparse.csc_matrix((rA.shape[0], len(roi_idxs)), dtype=rA.dtype)
    final_Cf = np.empty((len(roi_idxs), Cf.shape[1]), dtype=Cf.dtype)
    for ridx in range(len(roi_idxs)):
        if is_3D:
            final_masks[:,:,:,ridx] = masks[:,:,:,roi_idxs[ridx]]
        else:
            final_masks[:,:,ridx] = masks[:,:,roi_idxs[ridx]]
            
        final_rA[:, ridx] = rA[:, roi_idxs[ridx]] #.toarray().squeeze()
        final_Cf[ridx, :] = Cf[roi_idxs[ridx], :]
    
#    if is_3D:
#        masks = masks[:, :, :, roi_idxs]
#    else:
#        masks = masks[:, :, roi_idxs]
    coors = [coors[i] for i in roi_idxs if not i==nr]
#    rA = rA[:, roi_idxs]
#    Cf = Cf[roi_idxs, :]
    
    #cc1 = [[l[0] for l in n['coordinates']] for n in coors]
    #cc2 = [[l[1] for l in n['coordinates']] for n in coors]
    #coords = [[(x,y) for x,y in zip(cc1[n], cc2[n])] for n in range(len(cc1))]
    #coms = np.array([np.array(n) for n in coords])
    
    return final_masks, img, coors, roi_idxs, is_3D, nb, final_rA, final_Cf

#%
#def standardize_rois(session_dir, roi_id, auto=False, check_motion=True, zproj_type='mean', mcmetric='zproj_corrfcoefs', coreg_results_path=None, keep_good_rois=True):
def standardize_rois(session_dir, roi_id, auto=False, zproj_type='mean', coreg_results_path=None, keep_good_rois=True):

    RID = load_RID(session_dir, roi_id, auto=auto)
    rid_dir = RID['DST']
    roi_type = RID['roi_type']
    if roi_type == 'coregister':
        src_roi_type = RID['PARAMS']['source']['roi_type']
    
    rid_figdir = os.path.join(rid_dir, 'figures')
    if not os.path.exists(rid_figdir):
        os.makedirs(rid_figdir)

    check_motion = RID['PARAMS']['eval']['check_motion']
    mcmetric = RID['PARAMS']['eval']['mcmetric']
    
    roi_source_paths, tiff_source_paths, filenames, mc_excluded_tiffs, mcmetrics_filepath = get_source_paths(session_dir, RID, check_motion=check_motion, mcmetric=mcmetric)   
    if mcmetrics_filepath is None:
        mcmetrics_filepath = "None"
 
    roiparams_path = os.path.join(rid_dir, 'roiparams.json')
    if not os.path.exists(roiparams_path):
        if roi_type == 'caiman2D':
            evalparams = RID['PARAMS']['options']['eval']
        roiparams = save_roi_params(RID, evalparams=evalparams, keep_good_rois=keep_good_rois, mc_excluded_tiffs=mc_excluded_tiffs)
    else:
        with open(roiparams_path, 'r') as f:
            roiparams = json.load(f)
        
    mask_filepath = os.path.join(rid_dir, 'masks.hdf5')
    maskfile = h5py.File(mask_filepath, 'w')
    maskfile.attrs['roi_type'] = roi_type
    maskfile.attrs['roi_id'] = roi_id
    maskfile.attrs['rid_hash'] = RID['rid_hash']
    maskfile.attrs['animal'] = os.path.split(os.path.split(session_dir)[0])[1]
    maskfile.attrs['session'] = os.path.split(session_dir)[1]
    #maskfile.attrs['ref_file'] = params_thr['ref_filename']
    maskfile.attrs['creation_date'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    maskfile.attrs['keep_good_rois'] = roiparams['keep_good_rois']
    maskfile.attrs['ntiffs_in_set'] = len(filenames)
    maskfile.attrs['mcmetrics_filepath'] = mcmetrics_filepath
    maskfile.attrs['mcmetric_type'] = mcmetric
    maskfile.attrs['zproj'] = zproj_type
    
    try:
        if roi_type == 'caiman2D' or (roi_type == 'coregister' and src_roi_type == 'caiman2D'):
            
            if coreg_results_path is None and roi_type == 'coregister':
                coreg_dir = os.path.join(rid_dir, 'coreg_results')
                coreg_file = sorted([c for c in os.listdir(coreg_dir) if 'coreg_results' in c and c.endswith('hdf5')], key=natural_keys)[-1]
                print "Using most recent coreg file: %s" % coreg_file
                print "Unable to load coregistration file..."
                
            for fidx, nmfpath in enumerate(sorted(roi_source_paths, key=natural_keys)):
                
                curr_file = filenames[fidx]
                print "Creating ROI masks for %s" % curr_file
                
                # Create group for current file:
                filegrp = maskfile.create_group(filenames[fidx])
                filegrp.attrs['source'] = os.path.split(nmfpath)[0]

                # Format NMF output to standard masks:
                print "Formatting masks..."
                if roi_type == 'coregister':
                    # Load coreg results:
                    coreg_byfile = h5py.File(coreg_results_path, 'r')
                    
                    # Get masks:
                    masks, img, coord_info, roi_idxs, is_3D, nb, Ab, Cf = format_rois_nmf(nmfpath, roiparams, 
                                                                         pass_rois=coreg_byfile[curr_file]['roi_idxs'], 
                                                                         coreg_rois=coreg_byfile[curr_file]['universal_matches'])
                else:
                    masks, img, coord_info, roi_idxs, is_3D, nb, Ab, Cf = format_rois_nmf(nmfpath, roiparams, zproj_type=zproj_type)
                
                maskfile.attrs['is_3D'] = is_3D

                #sorted_roi_idxs = np.argsort(roi_idxs) # background comp(s) will always be last anyway
                #masks = masks[:,:,sorted_roi_idxs]                
                roi_names = sorted(["roi%04d" % int(ridx+1) for ridx in range(len(coord_info))], key=natural_keys) # BG not included for coordinate list

                # Save masks for current file (TODO: separate slices?)
                print('Mask array:', masks.shape)
                currmasks = filegrp.create_dataset('masks', masks.shape, masks.dtype)
                currmasks[...] = masks
                currmasks.attrs['src_roi_idxs'] = roi_idxs
                #currmasks.attrs['roi_idxs'] = sorted_roi_idxs
                currmasks.attrs['nrois'] = len(roi_names) #len(roi_idxs) - nb
                currmasks.attrs['background'] = nb
                
                # Save spatial and temporal comps:
                save_sparse_hdf5(Ab, '%s/Ab' % curr_file, mask_filepath)
                save_sparse_hdf5(scipy.sparse.csc_matrix(Cf, dtype=Cf.dtype), '%s/Cf' % curr_file, mask_filepath)

                # Save CoM for each ROI:
                coms = np.array([r['CoM'] for r in coord_info])
                currcoms = filegrp.create_dataset('coms', coms.shape, coms.dtype)
                currcoms[...] = coms
                
                # Save coords for each ROI:
                for ridx, roi in enumerate(coord_info):
                    curr_roi = filegrp.create_dataset('/'.join(['coords', roi_names[ridx]]), roi['coordinates'].shape, roi['coordinates'].dtype)
                    curr_roi[...] = roi['coordinates']
                    curr_roi.attrs['roi_source'] = nmfpath
                    curr_roi.attrs['id_in_set'] = roi_names[ridx]
                    curr_roi.attrs['id_in_src'] = roi['neuron_id']
                    curr_roi.attrs['idx_in_src'] = roi_idxs[ridx]
                    if roi_type == 'coregister':
                        curr_roi.attrs['idx_in_coreg'] = coreg_byfile[curr_file]['universal_matches']
                
                # Save zproj image:
                zproj = filegrp.create_dataset('zproj_img', img.shape, img.dtype)
                zproj[...] = img
                zproj.attrs['zproj_type'] = zproj_type

                # Plot figure with ROI masks: (1-indexed for naming)
                p2, p98 = np.percentile(img, (2, 99.98))
                avgimg = skimage.exposure.rescale_intensity(img, in_range=(p2, p98)) #avg *= (1.0/avg.max())
                print "Plotting final ROIs..."
                pl.figure()
                pl.imshow(avgimg, interpolation='None', cmap=pl.cm.gray)
                for ridx in range(len(roi_names)):
                    print "plot roi: %i" % int(ridx+1)
                    masktmp = masks[:,:,ridx]
                    msk = masktmp.copy() 
                    msk[msk==0] = np.nan
                    pl.imshow(msk, interpolation='None', alpha=0.3, cmap=pl.cm.hot)
                    [ys, xs] = np.where(masktmp>0)
                    pl.text(xs[int(round(len(xs)/4))], ys[int(round(len(ys)/4))], str(ridx+1), weight='light', fontsize=8, color='w')
                    pl.axis('off')
                pl.colorbar()
                pl.tight_layout()
                #imname = '%s_%s_%s_masks.png' % (roi_id, rid_hash, filenames[fidx])
                imname = 'rois_%s_masks.png' % filenames[fidx]
                print(imname) 
                pl.savefig(os.path.join(rid_figdir, imname))
                pl.close()
                
                # Also plot BG, if relevant -- for nb background comps, they will also be the last nb idxs of roi_idxs
                if not len(roi_names) == masks.shape[-1]:
                    print "Plotting background componbents for tiff."
                    n_bg_comps = masks.shape[-1] - len(roi_names)
                    bg_names = ['bg%02d' % int(b+1) for b in range(n_bg_comps)]
                    for bidx, bg in enumerate(bg_names):
                        bgtmp = masks[:,:,(len(roi_idxs)-1+bidx)]
                        bgtmp[bgtmp==0] = np.nan
                        pl.figure()
                        #pl.imshow(avgimg, cmap=pl.cm.gray, alpha=0)
                        pl.imshow(bgtmp, alpha=1, cmap=pl.cm.hot); pl.axis('off'); pl.colorbar()
                        pl.title(bg_names[bidx])
                        imname = '%s_%s_masks.png' % (bg_names[bidx], filenames[fidx])
                        print imname
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

    return mask_filepath

def save_roi_params(RID, evalparams=None, keep_good_rois=True, mc_excluded_tiffs=[]):
    roiparams = dict()
    rid_dir = RID['DST']
    
    roiparams['eval'] = evalparams       
    roiparams['keep_good_rois'] = keep_good_rois
    roiparams['excluded_tiffs'] = mc_excluded_tiffs
    roiparams['roi_type'] = RID['roi_type']
    roiparams['roi_id'] = RID['roi_id']
    roiparams['rid_hash'] = RID['rid_hash']
    
    roiparams_filepath = os.path.join(rid_dir, 'roiparams.json') # % (str(roi_id), str(rid_hash)))
    write_dict_to_json(roiparams, roiparams_filepath)
    
    print "Saved ROI params to: %s" % roiparams_filepath
    
    return roiparams 

#%%

def extract_options(options):
    parser = optparse.OptionParser()
    
    # PATH opts:
    parser.add_option('-D', '--root', action='store', dest='rootdir', default='/nas/volume1/2photon/data', help='data root dir (root project dir containing all animalids) [default: /nas/volume1/2photon/data, /n/coxfs01/2pdata if --slurm]')
    parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')
    parser.add_option('-S', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID')
    
    parser.add_option('--default', action='store_true', dest='default', default=False, help="Use all DEFAULT params, for params not specified by user (no interactive)")
    parser.add_option('--slurm', action='store_true', dest='slurm', default=False, help="set if running as SLURM job on Odyssey")
    parser.add_option('-r', '--roi-id', action='store', dest='roi_id', default='', help="ROI ID for rid param set to use (created with set_roi_params.py, e.g., rois001, rois005, etc.)")
    
    parser.add_option('-z', '--zproj', action='store', dest='zproj_type', default="mean", help="zproj to use for display [default: mean]")
    
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
    
    parser.add_option('-E', '--eval-key', action="store",
                      dest="eval_key", default=None, help="Evaluation key from ROI source <rid_dir>/evaluation (format: evaluation_YYYY_MM_DD_hh_mm_ss)")
    parser.add_option('-C', '--coreg-path', action="store",
                      dest="coreg_results_path", default=None, help="Path to coreg results if standardizing ROIs only")

#    parser.add_option('-M', '--mcmetric', action="store",
#                      dest="mcmetric", default='zproj_corrcoefs', help="Motion-correction metric to use for identifying tiffs to exclude [default: zproj_corrcoefs]")
#    
    parser.add_option('--par', action="store_true",
                      dest='multiproc', default=False, help="Use mp parallel processing to extract from tiffs at once, only if not slurm")
#    parser.add_option('--mc', action="store_true",
#                      dest='check_motion', default=False, help="Check MC evaluation for bad tiffs.")

    parser.add_option('-x', '--exclude', action="store",
                  dest="excluded_tiffs", default='', help="Tiff numbers to exclude (comma-separated)")

    parser.add_option('--format', action="store_true",
                      dest='format_only', default=False, help="Only format ROIs to standard (already extracted).")

    (options, args) = parser.parse_args(options)
    
    if options.slurm is True:
        if 'coxfs01' not in options.rootdir:
            options.rootdir = '/n/coxfs01/2p-data'
 
    return options
    #%%
    
    #rootdir = '/nas/volume1/2photon/data'
    #animalid = 'JR063' #'JR063'
    #session = '20171128_JR063' #'20171128_JR063'
    #roi_id = 'rois002'
    #slurm = False
    #auto = False
    ##
    #keep_good_rois = True       # Only keep "good" ROIs from a given set (TODO:  add eval for ROIs -- right now, only have eval for NMF and coregister)
    ##
    ### COREG-SPECIFIC opts:
    #use_max_nrois = True        # Use file which has the max N ROIs as reference (alternative is to use reference file)
    #dist_maxthr = 0.1
    #dist_exp = 0.1
    #dist_thr = 0.5
    #dist_overlap_thr = 0.8
    ##
    #eval_key = '2018_01_22_18_50_59'
    #mcmetric = 'zproj_corrcoefs'
    #zproj_type = 'mean'
    
#%%

def just_format_rois(options):
    #options = extract_options(options)

    rootdir = options.rootdir
    animalid = options.animalid
    session = options.session
    roi_id = options.roi_id
    slurm = options.slurm
    auto = options.default
    
    zproj_type= options.zproj_type
    #mcmetric = options.mcmetric
    coreg_results_path = options.coreg_results_path
    #check_motion = options.check_motion
 
    session_dir = os.path.join(rootdir, animalid, session)
    #mask_filepath = standardize_rois(session_dir, roi_id, auto=auto, check_motion=check_motion, zproj_type=zproj_type, mcmetric=mcmetric, coreg_results_path=coreg_results_path)
    mask_filepath = standardize_rois(session_dir, roi_id, auto=auto, zproj_type=zproj_type, coreg_results_path=coreg_results_path)


    print "Standardized ROIs, mask file saved to: %s" % mask_filepath

    return session_dir, mask_filepath 

     
def do_roi_extraction(options):
    #options = extract_options(options)
        
    # Set USER INPUT options:
    rootdir = options.rootdir
    animalid = options.animalid
    session = options.session
    roi_id = options.roi_id
    slurm = options.slurm
    auto = options.default
    
    keep_good_rois = options.keep_good_rois
    use_max_nrois = options.use_max_nrois
    
    dist_maxthr = options.dist_maxthr
    dist_exp = options.dist_exp
    dist_thr = options.dist_thr
    dist_overlap_thr = options.dist_overlap_thr
    
    zproj_type= options.zproj_type
    
    eval_key = options.eval_key
    #mcmetric = options.mcmetric
    
    multiproc = options.multiproc
    #check_motion = options.check_motion
    exclude_str = options.excluded_tiffs
    coreg_results_path = options.coreg_results_path
 

    #%%
    session_dir = os.path.join(rootdir, animalid, session)
    
    # =============================================================================
    # Load specified ROI-ID parameter set:
    # =============================================================================
    try:
        RID = load_RID(session_dir, roi_id, auto=auto)
        print "Evaluating ROIs from set: %s" % RID['roi_id']
    except Exception as e:
        print "-- ERROR: unable to open source ROI dict. ---------------------"
        traceback.print_exc()
        print "---------------------------------------------------------------"
        

    #%%
    # =============================================================================
    # Get meta info for current run and source tiffs using trace-ID params:
    # =============================================================================
    tiff_sourcedir = RID['SRC']
    path_parts = tiff_sourcedir.split(session_dir)[-1].split('/')
    acquisition = path_parts[1]
    run = path_parts[2]
    process_dirname = path_parts[4]
    process_id = process_dirname.split('_')[0]
    
    tiffs = sorted([t for t in os.listdir(tiff_sourcedir) if t.endswith('tif')], key=natural_keys) 
    filenames = sorted([str(re.search('File(\d{3})', tf).group(0)) for tf in tiffs], key=natural_keys)
    print "FILES:", filenames
    
    check_motion = RID['PARAMS']['eval']['check_motion'] 
    mcmetric = RID['PARAMS']['eval']['mcmetric']
 
    if check_motion is True: 
        filenames, excluded_tiffs, mcmetrics_filepath = check_mc_evaluation(RID, filenames, mcmetric_type=mcmetric, 
                                                       acquisition=acquisition, run=run, process_id=process_id)
    else:
        mc_excluded_tiffs = []
    
     
    #%%
    # =============================================================================
    # Extract ROIs using specified method:
    # =============================================================================
    print "Extracting ROIs...====================================================="
    roi_type = RID['roi_type']
    rid_hash = RID['rid_hash']
 
    format_roi_output = False
    #src_roi_type = None
    t_start = time.time()
    if len(mc_excluded_tiffs) > 0:
        exclude_str = ','.join([int(fn[4:]) for fn in mc_excluded_tiffs])
     
    if roi_type == 'caiman2D':
        #%
        roi_opts = ['-D', rootdir, '-i', animalid, '-S', session, '-A', acquisition, '-R', run, '-p', rid_hash]
        if slurm is True:
            roi_opts.extend(['--slurm'])
        if len(exclude_str) > 0:
            roi_opts.extend(['-x', exclude_str])
        if multiproc is True:
            roi_opts.extend(['--par'])
            
        nmf_hash, rid_hash = rcm.extract_cnmf_rois(roi_opts)
        
        # Clean up tmp RID files:
        session_dir = os.path.join(rootdir, animalid, session)
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
        roi_source_paths, tiff_source_paths, filenames, mc_excluded_tiffs, mcmetrics_filepath = get_source_paths(session_dir, RID, check_motion=check_motion, 
                                                                                                             mcmetric=mcmetric, 
                                                                                                             acquisition=acquisition,
                                                                                                             run=run,
                                                                                                             process_id=process_id)


        #%
        src_roi_id = RID['PARAMS']['options']['source']['roi_id']
        src_roi_dir = RID['PARAMS']['options']['source']['roi_dir']
        
        #% Set COREG opts:
        coreg_opts = ['-D', rootdir, '-i', animalid, '-S', session, '-r', roi_id,
                      '-t', dist_maxthr,
                      '-n', dist_exp,
                      '-d', dist_thr,
                      '-o', dist_overlap_thr]
        
        if use_max_nrois is True: # == 'max':
            coreg_opts.extend(['--max'])
        if keep_good_rois is True:
            coreg_opts.extend(['--good'])
        
        #% RUN COREGISTRATION
        print "==========================================================="
        print "RID %s -- Running coregistration..." % rid_hash
        print "RID %s -- Source ROI set is: %s" % (rid_hash, src_roi_id)
        if eval_key is None and keep_good_rois is False:
            # Just run coregistration on default (if nmf rois, will use source eval-params if "keep_good_rois" is True)
            ref_rois, params_thr, coreg_outpath = reg.run_coregistration(coreg_opts)
            src_eval_filepath = None
        else:
            # Load ROI info for "good" rois to include:
            src_eval, src_eval_filepath = load_eval_results(src_roi_dir, eval_key, auto=False)
            coreg_opts.extend(['--roipath=%s' % src_eval_filepath])            
            #%
            ref_rois, params_thr, coreg_results_path = reg.run_coregistration(coreg_opts)
    
        print("Found %i common ROIs matching reference." % len(ref_rois))
        format_roi_output = True
        #src_roi_type = RID['PARAMS']['options']['source']['roi_type']
        #%
    else:
        print "ERROR: %s -- roi type not known..." % roi_type
    
    print "RID %s -- Finished ROI extration!" % rid_hash
    print_elapsed_time(t_start)
    print "======================================================================="


    #%% Save ROI params info:
        
    # TODO: Include ROI eval info for other methods?
    # TODO: If using NMF eval methods, make func to do evaluation at post-extraction step (since extract_rois_caiman.py keeps all when saving anyway)
    if roi_type == 'caiman2D':
        evalparams = RID['PARAMS']['options']['eval']
    elif roi_type == 'coregister':
        evalparams = params_thr['eval']
 
    roiparams = save_roi_params(RID, evalparams=evalparams, keep_good_rois=keep_good_rois, mc_excluded_tiffs=mc_excluded_tiffs)
    
    #%%
    # =============================================================================
    # Format ROI output to standard, if applicable:
    # =============================================================================
        
    if format_roi_output is True:
        mask_filepath = standardize_rois(session_dir, roi_id, auto=auto, check_motion=check_motion, zproj_type=zproj_type, mcmetric=mcmetric, coreg_results_path=coreg_results_path)
        print "Standardized ROIs, mask file saved to: %s" % mask_filepath
    
    return session_dir, rid_hash

def select_roi_action(options):
    options = extract_options(options)
    
    mask_filepath = None
    rid_hash = None
    if options.format_only is True:
        session_dir, mask_filepath = just_format_rois(options)
    else: 
        session_dir, rid_hash = do_roi_extraction(options)
    
    if mask_filepath is not None and rid_hash is None:
        formatting_only = True
        optargout = mask_filepath
    else:
        formatting_only = False
        optargout = rid_hash
    
    return session_dir, optargout, formatting_only 

def main(options):
    session_dir, optargout, formatting_only = select_roi_action(options) 
    
    #session_dir, rid_hash = do_roi_extraction(options)
    if formatting_only is True:
        print "Formatted ROIs! Masks saved to:\n%s" % optargout
    else:
        print "RID %s -- Finished formatting ROI output to standard." % optargout 
        post_rid_cleanup(session_dir, rid_hash)
        print "Cleaned up tmp rid files." 
        print "*************************************************"
        print "FINISHED EXTRACTING ROIs!"
        print "*************************************************"

#%%

if __name__ == '__main__':
    main(sys.argv[1:])
