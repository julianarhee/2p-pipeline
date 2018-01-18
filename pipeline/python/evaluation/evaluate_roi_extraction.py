#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 12:12:03 2018

@author: julianarhee
"""
import os
import re
import h5py
import datetime
import traceback
import json
import caiman as cm

import numpy as np

from pipeline.python.utils import natural_keys
from caiman.components_evaluation import evaluate_components, estimate_components_quality_auto

def evaluate_rois_nmf(mmap_path, nmfout_path, evalparams, dview=None, eval_outdir='', save_output=True):
    """
    Uses component quality evaluation from caiman.
    mmap_path : (str)
        path to mmap files of movies (.mmap)
    nmfout_path : (str)
        path to nmf output files from initial nmf extraction (.npz)
    evalparams : (dict)
        params for evaluation function
        'final_frate' :  default uses acquisition rate (not sure if this matters)
        'rval_thr' :  accept components with space correlation threshold of this or higher
        'min_SNR' :  accept components with peak-SNR of this or higher
        'decay_time' :  length of typical transient (in secs)
        'gSig' :  (int) half-size of neuron
        'use_cnn' :  CNN classifier (unused)
        
    Returns:
        idx_components :  components that pass
        idx_components_bad :  components that fail
        SNR_comp :  SNR val for each comp
        r_values :  spatial corr values for each comp
    """
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
        
        filenames = sorted([str(re.search('File(\d{3})', nmffile).group(0)) for nmffile in source_nmf_paths], key=natural_keys)

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
                
                #%start a cluster for parallel processing
                try:
                    dview.terminate() # stop it if it was running
                except:
                    pass
                c, dview, n_processes = cm.cluster.setup_cluster(backend='local', # use this one
                                                                 n_processes=None,  # number of process to use, reduce if out of mem
                                                                 single_thread = False)
        
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

    print "Finished ROI evaluation step. ROI eval info saved to:"
    print roi_idx_filepath
    
    return roi_idx_filepath

