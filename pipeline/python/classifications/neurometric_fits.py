#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 19:56:49 2021

@author: julianarhee
"""
import sys
import traceback
import math
import multiprocessing as mp
import os
import optparse
import psignifit as ps
import glob
# import pickle as pkl
import dill as pkl
import numpy as np
import pylab as pl
import seaborn as sns
import scipy.stats as spstats
import pandas as pd
import importlib
import scipy as sp
import itertools


# #########################
def decode_analysis_id(visual_area=None, prefix='split_pupil', 
                       response_type='dff', responsive_test='ROC',
                       overlap_str='noRF', trial_epoch='plushalf', C_str='tuneC'):
    
    results_id = '%s_%s__%s-%s_%s__%s__%s' \
            % (prefix, visual_area, response_type, responsive_test, overlap_str, trial_epoch, C_str)

    return results_id

def add_morph_info(ndf, sdf, morph_lut, a_morphs, b_morphs, midp=53):
    # add stimulus info
    morphlevels = sdf['morphlevel'].unique()
    max_morph = max(morphlevels)

    assert midp in  morphlevels, "... Unknown midpoint in morphs: %s" % str(morphlevels)
        
    ndf['size'] = [sdf['size'][c] for c in ndf['config']]
    ndf['morphlevel'] = [sdf['morphlevel'][c] for c in ndf['config']]
    ndf = ndf[(ndf['morphlevel']!=-1)].copy()

    morph_lut, a_morphs, b_morphs = get_morph_levels(levels=morphlevels, midp=midp)
    # update neuraldata
    ndf['morphstep'] = [morph_lut[m] for m in ndf['morphlevel']]
    ndf['morph_ix'] = [m/float(max_morph) for m in ndf['morphlevel']]

    ndf['object'] = None
    ndf.loc[ndf.morphlevel.isin(a_morphs), 'object'] = 'A'
    ndf.loc[ndf.morphlevel.isin(b_morphs), 'object'] = 'B'
    ndf.loc[ndf.morphlevel==midp, 'object'] = 'M'

    return ndf


# SPLIT_PUPIL (Calculate AUCs across iterations).
# -----------------------------------------------------------
class WorkerStop(Exception):
    traceback.print_exc() 
    pass

def auc_split_pupil_worker(out_q, iternums, ndf, trialdf, param, allow_negative):
    '''
    trialdf: All iterations of trial indices selected for various pupil states
    ndf: data for all cells in current FOV (group by)
    '''
    i_list = []
    for ni in iternums:
        curr_trials = trialdf[trialdf['iteration']==ni]
        for (arousal_label, shuffle_cond), a_df in curr_trials.groupby(['arousal', 'true_labels']):
            print(ni, arousal_label, shuffle_cond)
            try:
                curr_auc = ndf[ndf['trial'].isin(a_df['trial'].values)].groupby('cell')\
                            .apply(get_auc_AB, param=param, allow_negative=allow_negative)\
                            .reset_index().drop('level_1', axis=1)
                curr_auc['arousal'] = arousal_label
                curr_auc['iteration'] = ni
                curr_auc['true_labels'] = shuffle_cond
                curr_auc['n_chooseB'] = curr_auc['AUC']*curr_auc['n_trials']
                i_list.append(curr_auc)
            except Exception as e:
                out_q.put(None)
                raise WorkerStop("error!")
                
    curr_iterdf = pd.concat(i_list, axis=0)
    out_q.put(curr_iterdf)
    
def iterate_auc_split_pupil(ndf, trialdf, n_iterations=100, param='morphlevel', 
                            allow_negative=True, n_processes=1):
    iterdf=None
   
    procs=[] 
    results = []
    terminating = mp.Event()
    try:
        iter_list = np.arange(0, n_iterations)
        out_q = mp.Queue()
        chunksize = int(math.ceil(len(iter_list) / float(n_processes)))

        procs = []
        for i in range(n_processes):
            p = mp.Process(target=auc_split_pupil_worker, 
                           args=(out_q, iter_list[chunksize * i:chunksize * (i + 1)],
                                 ndf, trialdf, param, allow_negative))
            p.start()
        results=[]
        for i in range(n_processes):
            res = out_q.get(99999)
            results.append(res)
        for p in procs:
            p.join()
    except WorkerStop:
        print("No results (terminating)")
        terminating.set()
    except KeyboardInterrupt:
        terminating.set()
    except Exception as e:
        traceback.print_exc()
    finally:
        for p in procs:
            p.join()
            print('%s.exitcode = %s' % (p.name, p.exitcode))

    res_ = [i for i in results if i is not None]
    if len(res_)>0:
        iterdf = pd.concat(res_,axis=0)

    return iterdf


def split_pupil_calculate_auc(va, dk, ndf, sdf, trialdf, decode_id='results_id', 
                    param='morphlevel', n_processes=1, allow_negative=True,
                     n_iterations=None):
    traceid_dir = get_tracedir_from_datakey(dk)
    
    # Setup output file
    results_outfile = os.path.join(traceid_dir, 'neurometric', 'split_pupil', 'AUC_%s.pkl' % (decode_id))
    if os.path.exists(results_outfile):
        os.remove(results_outfile)
    
    # Do iterations 
    iterdf = iterate_auc_split_pupil(ndf, trialdf, n_iterations=n_iterations, 
                    param=param, n_processes=n_processes, allow_negative=allow_negative) 
    if iterdf is None:
        print("None returned -- %s, %s" % (va, dk))
        return None
    
    iterdf['visual_area'] = va
    iterdf['datakey'] = dk
    # save
    with open(results_outfile, 'wb') as f:
        pkl.dump(iterdf, f, protocol=pkl.HIGHEST_PROTOCOL)
 
    print("============ done. %s|%s ============" % (va, dk))  
    print(iterdf.groupby(['true_labels', 'arousal']).mean())

    print(results_outfile)
    
    return


def do_split_pupil_auc(curr_visual_area, curr_datakey, param='morphlevel', 
                    sigmoid='gauss', fit_experiment='2AFC', 
                    max_auc=0.70, fit_new=False, create_auc=False, 
                    traceid='traces001', responsive_test='ROC', responsive_thr=0.05,
                    experiment='blobs',
                    src_response_type='dff', overlap_thr=None, trial_epoch='plushalf',
                    n_processes=1, allow_negative=True, n_iterations=None):

    # Load source data
    DATA, SDF, selective_df = get_data(traceid=traceid, 
                    responsive_test=responsive_test, responsive_thr=responsive_thr)
    assert curr_visual_area in DATA['visual_area'].unique(), \
            "Visual area <%s> not in DATA." % curr_visual_area
    assert curr_datakey in DATA['datakey'].unique(), \
            "Datakey <%s> not in DATA" % curr_datakey

    sdf = SDF[curr_datakey].copy()
    ndf = DATA[(DATA.visual_area==curr_visual_area) & (DATA.datakey==curr_datakey)].copy()

    morph_lut, a_morphs, b_morphs = get_morph_levels() 
    ndf = add_morph_info(ndf, sdf, morph_lut, a_morphs, b_morphs)
       
    overlap_str = 'noRF' if overlap_thr is None else 'overlap%.2f' % overlap_thr
 
    curr_decode_id = decode_analysis_id(visual_area=curr_visual_area, 
                        responsive_test=responsive_test, response_type=src_response_type, 
                        overlap_str=overlap_str, trial_epoch=trial_epoch)

    print("Loading SPLIT_PUPIL results w ID: %s" % curr_decode_id)
    
    # Set output dir
    traceid_dir = get_tracedir_from_datakey(curr_datakey)
    dst_dir = os.path.join(traceid_dir, 'neurometric', 'split_pupil')
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    decoding_outfile = os.path.join(traceid_dir, 'decoding', 'inputdata_%s.pkl' % curr_decode_id)
    assert os.path.exists(decoding_outfile),\
            "(%s, %s) No split_pupil results: %s" % (curr_datakey, curr_visual_area, curr_decode_id)

    with open(decoding_outfile, 'rb') as f:
        indata = pkl.load(f, encoding='latin1')
    trialdf = indata['input_trials']

    if n_iterations is None:
        n_iterations = trialdf['iteration'].max()+1
    print("~~~~~~~~~~ N iterations=%i" % n_iterations)
 
    split_pupil_calculate_auc(curr_visual_area, curr_datakey, ndf, sdf, 
            trialdf, decode_id=curr_decode_id, param=param, n_processes=n_processes, 
            allow_negative=allow_negative, n_iterations=n_iterations)

    print("@@@@@@yaaa done!!@@@@@@@@@@")

    return



# SPLIT_PUPIL (Fit curves across iterations).
# -----------------------------------------------------------s
def fit_iter_pupil_worker(out_q, iternums, auc_fov_iters, opts, sigmoid, param, allow_negative):
    '''
    auc_fov_iters:  df, loaded from traceid dir per fov
    For each iteration, has AUCs calculated for sie, arousal, shuffle cond.
    '''
    curr_iterdf=None
    i_list = []
#    try:
    for (ni, ri) in iternums:
        curr_trials = auc_fov_iters[(auc_fov_iters['iteration']==ni) 
                                  & (auc_fov_iters['cell']==ri)]
        #for (sz, a_cond, s_cond), a_df in curr_trials.groupby(['size', 'arousal', 'true_labels']):
        print(ni, ri) #a_cond, s_cond)
        try:
            curr_fiter = curr_trials.groupby(['size', 'arousal', 'true_labels'], as_index=False)\
                        .apply(group_fit_psignifit, opts, ni=ni, 
                        sigmoid=sigmoid, param=param, allow_negative=allow_negative)
            curr_fiter['iteration'] = ni
            #curr_fiter['true_labels'] = s_cond
            #curr_fiter['arousal'] = a_cond 
            #curr_fiter['size']= sz
            curr_fiter['cell']= ri
            i_list.append(curr_fiter)
        except Exception as e:
            out_q.put(None)
            raise WorkerStop("error!")

    curr_iterdf = pd.concat(i_list, axis=0)
    out_q.put(curr_iterdf)
 
def xfit_split_pupil_worker(out_q, iternums, auc_fov_iters, opts, sigmoid, param, allow_negative):
    '''
    auc_fov_iters:  df, loaded from traceid dir per fov
    For each iteration, has AUCs calculated for sie, arousal, shuffle cond.
    '''
    curr_iterdf=None
    i_list = []
#    try:
    for ni in iternums:
        curr_trials = auc_fov_iters[auc_fov_iters['iteration']==ni]
        for (sz, a_cond, s_cond), a_df in curr_trials.groupby(['size', 'arousal', 'true_labels']):
            print(ni, a_cond, s_cond)
            try:
                curr_fiter = a_df.groupby(['cell'], as_index=False)\
                            .apply(group_fit_psignifit, opts, ni=ni, 
                            sigmoid=sigmoid, param=param, allow_negative=allow_negative)
                curr_fiter['arousal'] = a_cond 
                curr_fiter['iteration'] = ni
                curr_fiter['true_labels'] = s_cond
                curr_fiter['size']= sz
                i_list.append(curr_fiter)
            except Exception as e:
                out_q.put(None)
                raise WorkerStop("error!")

    curr_iterdf = pd.concat(i_list, axis=0)
    out_q.put(curr_iterdf)
 
#                
##            curr_fiter = curr_trials.groupby(['cell', 'size', 'arousal', 'true_labels'], as_index=False)\
##                .apply(group_fit_psignifit, opts, ni=ni, sigmoid=sigmoid, param=param, allow_negative=allow_negative)
##            curr_fiter['iteration'] = ni
##            i_list.append(curr_fiter)
#        curr_iterdf = pd.concat(i_list, axis=0)
#        out_q.put(curr_iterdf)
#
#    except Exception as e:
#        traceback.print_exc()
#        out_q.put(None)
#        raise WorkerStop("error!") 
#        return 
#    #curr_iterdf = pd.concat(i_list, axis=0)
#    #out_q.put(curr_iterdf)
#

def iterate_fit_split_pupil(auc_fov_iters, n_iterations=100, n_processes=1,
                            fitopts={}, sigmoid='gauss', param='morphlevel', allow_negative=True):
    iterdf=None 
    procs=[] 
    results = []
    terminating = mp.Event()
    try:
        #iter_list = np.arange(0, n_iterations)
        iter_nums = np.arange(0, n_iterations)
        cell_nums = auc_fov_iters['cell'].unique()
        iter_list = list(itertools.product(*[iter_nums, cell_nums]))
        out_q = mp.Queue()
        chunksize = int(math.ceil(len(iter_list) / float(n_processes)))
        for i in range(n_processes):
            p = mp.Process(target=fit_iter_pupil_worker, 
                           args=(out_q, iter_list[chunksize * i:chunksize * (i + 1)],
                                 auc_fov_iters, fitopts, sigmoid, param, allow_negative))
            p.start()
        for i in range(n_processes):
            res = out_q.get(99999)
            results.append(res)
        for p in procs:
            p.join()
    except WorkerStop:
        print("No results (terminating)")
        terminating.set()
    except KeyboardInterrupt:
        terminating.set()
    except Exception as e:
        traceback.print_exc()
    finally:
        for p in procs:
            p.join()
            print('%s.exitcode = %s' % (p.name, p.exitcode))

    res_ = [i for i in results if i is not None]
    if len(res_)>0:
        iterdf = pd.concat(res_,axis=0)

    return iterdf


def split_pupil_fit_curve(va, dk, auc_fov_iters, n_processes=1, 
                     n_iterations=None, decode_id='resultsid',
                    fit_experiment='2AFC', sigmoid='gauss', param='morphlevel', 
                    allow_negative=True):

    fitopts = dict()
    fitopts['expType'] = fit_experiment
    fitopts['threshPC'] = 0.5 
    fitopts['sigmoidName'] = sigmoid 

    traceid_dir = get_tracedir_from_datakey(dk)
    
    # Setup output file
    results_outfile = os.path.join(traceid_dir, 'neurometric', 'split_pupil', 'FIT_%s.pkl' % (decode_id))
    if os.path.exists(results_outfile):
        os.remove(outfile)
    if n_iterations is None:
        n_iterations = auc_fov_iters['iteration'].max()+1 
   
    print("~~~~~~~~N iterations: %i (n_proc=%i)" % (n_iterations, n_processes)) 
    # Do iterations 
    iterdf = iterate_fit_split_pupil(auc_fov_iters, fitopts=fitopts, 
                    sigmoid=sigmoid, param=param, allow_negative=allow_negative, 
                    n_processes=n_processes, n_iterations=n_iterations) 
    if iterdf is None:
        print("None returned -- %s, %s" % (va, dk))
        return None
    
    iterdf['visual_area'] = va
    iterdf['datakey'] = dk
    # save
    with open(results_outfile, 'wb') as f:
        pkl.dump(iterdf, f, protocol=pkl.HIGHEST_PROTOCOL)
 
    print("============ done. %s|%s ============" % (va, dk))  
    print(iterdf.groupby(['true_labels', 'arousal']).mean())

    print(results_outfile)
    
    return


 
def split_pupil_fit_curve_meanAUC(curr_visual_area, curr_datakey, auc_fov, 
                            param='morphlevel', sigmoid='gauss', fit_experiment='2AFC',
                            allow_negative=True, fit_new=False, 
                            traceid='traces001', experiment='blobs'):
    fitopts = dict()
    fitopts['expType'] = fit_experiment
    fitopts['threshPC'] = 0.5 

    # Set output dir
    traceid_dir = get_tracedir_from_datakey(curr_datakey, 
                    traceid=traceid, experiment=experiment)
    curr_dst_dir = os.path.join(traceid_dir, 'neurometric', 'split_pupil', 'fits')
    if not os.path.exists(curr_dst_dir):
        os.makedirs(curr_dst_dir)
    print("Saving ROI results to:\n   %s" % curr_dst_dir)

    if fit_new:
        old_fns = glob.glob(os.path.join(curr_dst_dir, 'meanAUC_*%s_rid*.pkl' %sigmoid))
        print("Removing %i old files" % len(old_fns))
        for f in old_fns:
            os.remove(f)
     
    ncells = len(auc_fov['cell'].unique())
    for ri, (rid, auc_r) in enumerate(auc_fov.groupby(['cell'])):
        if ri%10==0:
            print("... fitting %i of %i cells" % (int(ri+1), ncells))
        
        fn = 'meanAUC_%s_rid%03d.pkl' % (sigmoid, rid)
        outfile = os.path.join(curr_dst_dir, fn)
        fitparams = auc_r.groupby(['size', 'arousal', 'true_labels'], as_index=False)\
                        .apply(group_fit_psignifit, fitopts, \
                                param=param, ni=rid, allow_negative=allow_negative)
        with open(outfile, 'wb') as f:
            pkl.dump(fitparams, f, protocol=2)
        print("... saved: %s" % fn)

    print("DONE!")
    
    return None


def split_pupil_fit_curve_meanAUC0(curr_visual_area, curr_datakey, auc_fov, 
                            param='morphlevel', sigmoid='gauss', fit_experiment='2AFC',
                            allow_negative=True, fit_new=False, 
                            traceid='traces001', experiment='blobs', n_processes=1):
    fitopts = dict()
    fitopts['expType'] = fit_experiment
    fitopts['threshPC'] = 0.5 

    # Set output dir
    traceid_dir = get_tracedir_from_datakey(curr_datakey, 
                    traceid=traceid, experiment=experiment)
    curr_dst_dir = os.path.join(traceid_dir, 'neurometric', 'split_pupil', 'fits')
    if not os.path.exists(curr_dst_dir):
        os.makedirs(curr_dst_dir)
    print("Saving ROI results to:\n   %s" % curr_dst_dir)
    all_cells = auc_fov['cell'].unique()

    if fit_new:
        old_fns = glob.glob(os.path.join(curr_dst_dir, 'meanAUC_*%s_rid*.pkl' %sigmoid))
        print("Removing %i old files" % len(old_fns))
        for f in old_fns:
            os.remove(f)
        cells_to_run = all_cells
    else:
        old_fns = [os.path.split(fn)[-1] for fn in \
                glob.glob(os.path.join(curr_dst_dir, 'meanAUC_%s_rid*.pkl' %sigmoid))]
        cells_to_run = [rid for rid in all_cells if \
                'meanAUC_%s_rid%03d.pkl' % (sigmoid, rid) not in old_fns]

    print("... fitting %i of %i cells" % (len(cells_to_run), len(all_cells))) #ncells) 

    iterdf=None 
    procs=[] 
    results = []
    terminating = mp.Event()
    try:
        out_q = mp.Queue()
        iter_list =cells_to_run
        chunksize = int(math.ceil(len(iter_list) / float(n_processes)))
        for i in range(n_processes):
            p = mp.Process(target=split_pupil_worker, 
                    args=(out_q, iter_list[chunksize * i:chunksize * (i + 1)],
                auc_fov, curr_dst_dir, fitopts, sigmoid, param, allow_negative))
            p.start()
        for i in range(n_processes):
            res = out_q.get(99999)
            results.append(res)
        for p in procs:
            p.join()
    except WorkerStop:
        print("No results (terminating)")
        terminating.set()
    except KeyboardInterrupt:
        terminating.set()
    except Exception as e:
        traceback.print_exc()
    finally:
        for p in procs:
            p.join()
            print('%s.exitcode = %s' % (p.name, p.exitcode))

    res_ = [i for i in results if i is not None]
    if len(res_)>0:
        iterdf = pd.concat(res_,axis=0)
    
    print(iterdf.shape)
   
    return iterdf 



def split_pupil_worker(out_q, roi_list, auc_fov, curr_dst_dir, 
                        fitopts, sigmoid, param, allow_negative):
    '''
    auc_fov_iters:  df, loaded from traceid dir per fov
    For each iteration, has AUCs calculated for sie, arousal, shuffle cond.
    '''
    curr_iterdf=None
    i_list = []
#    try:
    for ri, (rid, auc_r) in enumerate(auc_fov[auc_fov['cell'].isin(roi_list)].groupby(['cell'])):
        try:
            print(ri, rid)
            fn = 'meanAUC_%s_rid%03d.pkl' % (sigmoid, rid)
            outfile = os.path.join(curr_dst_dir, fn)
            fitparams = auc_r.groupby(['Eff', 'size', 'arousal', 'true_labels'], as_index=False)\
                            .apply(group_fit_psignifit, fitopts, \
                                    ni=rid, param=param, sigmoid=sigmoid, 
                                    allow_negative=allow_negative)
            with open(outfile, 'wb') as f:
                pkl.dump(fitparams, f, protocol=2)
            i_list.append(fitparams)
 
        except Exception as e:
            out_q.put(None)
            raise WorkerStop("error!")
   
    fitd = pd.concat(i_list) 
    out_q.put(fitd)
           

def group_fit_psignifit(auc_, opts, ni=0, sigmoid='gauss', 
                    param='morphlevel', allow_negative=True):
    '''
    auc_ is curve for 1 cell, 1 size (and 1 arousal state, 1 shuffle cond)
    '''
    param_names = ['threshold', 'width', 'lambda', 'gamma', 'eta']
    at_pc = 0.75 if opts['expType']=='2AFC' else 0.5

    if allow_negative:
        try:
            eff = int(auc_['Eff'].unique())
        except Exception as e:
            print(auc_['Eff'].unique())
            return None

        opts['sigmoidName'] = 'neg_%s' % sigmoid if int(auc_['Eff'].unique())<53 else sigmoid
        # print(opts['sigmoidName'])

    data_ = data_matrix_from_auc(auc_, param=param, normalize=False)        
    res_ = ps.psignifit(data_, opts)
    try:
        thr = ps.getThreshold(res_, at_pc)[0] # Value at which function reaches at_pc correct
        slp = ps.getSlope(res_, ps.getThreshold(res_, at_pc)[0]) # Slope at given stimulus level
    except Exception as e:
        thr=None
        slp=None
        
    df_ = pd.DataFrame(res_['Fit'], index=param_names, columns=[ni]).T        
    df_['slope'] = slp
    df_['thr'] = thr
    
    add_cols= ['visual_area', 'datakey', 'cell', 'size', 'Eff']
    if  'arousal' in auc_.columns:
        add_cols.extend([ 'arousal', 'true_labels'])
    
    add_ = auc_[add_cols].drop_duplicates()
    add_.index = df_.index
    df_[add_cols] = add_

    return df_


def do_split_pupil_fits(curr_visual_area, curr_datakey, param='morphlevel', 
                    sigmoid='gauss', fit_experiment='2AFC', 
                    max_auc=0.70, fit_new=False, by_iter=True,allow_negative=True,
                    traceid='traces001', responsive_test='ROC', responsive_thr=0.05,
                    experiment='blobs', src_response_type='dff', 
                    overlap_thr=None, trial_epoch='plushalf',
                    n_processes=1,  n_iterations=100):

    # Get split_arousal AUCs (auc per iter, size, arousal state, shuffle cond)      
    overlap_str = 'noRF' if overlap_thr is None else 'overlap%.2f' % overlap_thr 
    curr_decode_id = decode_analysis_id(visual_area=curr_visual_area, 
                        responsive_test=responsive_test, response_type=src_response_type,
                        overlap_str=overlap_str, trial_epoch=trial_epoch)
    print("Loading SPLIT_PUPIL results w ID: %s" % curr_decode_id)
    
    # Set output dir
    traceid_dir = get_tracedir_from_datakey(curr_datakey, 
                    traceid=traceid, experiment=experiment)
    curr_dst_dir = os.path.join(traceid_dir, 'neurometric', 'split_pupil')
    if not os.path.exists(curr_dst_dir):
        os.makedirs(curr_dst_dir)
    print("Saving ROI results to:\n   %s" % curr_dst_dir)

    # Get input aucs
    auc_outfile = os.path.join(curr_dst_dir, 'AUC_%s.pkl' % curr_decode_id)
    assert os.path.exists(auc_outfile),\
        "(%s, %s) No file for: AUC_%s" % (curr_datakey, curr_visual_area, curr_decode_id)

    with open(auc_outfile, 'rb') as f:
        auc_fov_iters = pkl.load(f, encoding='latin1')

    # mean AUC over iters 
    group_cols = ['visual_area', 'datakey', 'cell', 'arousal', 'true_labels', 'morphlevel', 'size', 'Eff']
    auc_fov = auc_fov_iters.groupby(group_cols).mean().reset_index()
    # Save mean
    meanauc_outfile = os.path.join(os.path.split(auc_outfile)[0], 'mean_aucs.pkl')
    with open(meanauc_outfile, 'wb') as f:
        pkl.dump(auc_fov, f, protocol=2)

    pass_cells = auc_fov[auc_fov['AUC']>=max_auc]['cell'].unique() 

#    if by_iter:
#        pass_cells = auc_fov[auc_fov['AUC']>=max_auc]['cell'].unique() 
#    else:
#        pass_cells = [r for r, g in auc_fov[auc_fov['AUC']>=max_auc].groupby(['cell']) if len(g['Eff'].unique())==1]
#
    print("%i of %i cells pass (crit>=%.2f)" \
            % (len(pass_cells), len(auc_fov['cell'].unique()), max_auc))


    if by_iter:
        split_pupil_fit_curve(curr_visual_area, curr_datakey, 
                            auc_fov_iters[auc_fov_iters['cell'].isin(pass_cells)], 
                             decode_id=curr_decode_id,
                            param=param, sigmoid=sigmoid, fit_experiment=fit_experiment,
                            allow_negative=allow_negative, 
                            n_iterations=n_iterations, n_processes=n_processes)

       
    else:
        split_pupil_fit_curve_meanAUC0(curr_visual_area, curr_datakey, 
                            auc_fov[auc_fov['cell'].isin(pass_cells)], 
                            param=param, sigmoid=sigmoid, fit_experiment=fit_experiment,
                            allow_negative=allow_negative, fit_new=fit_new,
                            n_processes=n_processes)

        print("@@@@@@yaaa done!!@@@@@@@@@@")

    return





# ####################3

def split_datakey_str(s):
    session, animalid, fovn = s.split('_')
    fovnum = int(fovn[3:])
    return session, animalid, fovnum



def get_tracedir_from_datakey(datakey, experiment='blobs', rootdir='/n/coxfs01/2p-data', traceid='traces001'):
    session, animalid, fovn = split_datakey_str(datakey)
    
    darray_dirs = glob.glob(os.path.join(rootdir, animalid, session, 'FOV%i_zoom2p0x' % fovn,
                            'combined_%s_static' % experiment, 'traces', 
                             '%s*' % traceid, 'data_arrays', 'np_subtracted.npz'))
    
    assert len(darray_dirs)==1, "More than 1 data array dir found: %s" % str(darray_dirs)
    
    traceid_dir = darray_dirs[0].split('/data_arrays')[0]
    return traceid_dir

                
def load_source_dataframes(traceid='traces001', 
                responsive_test='ROC', responsive_thr=0.05, 
                aggregate_dir = '/n/coxfs01/julianarhee/aggregate-visual-areas'):
    D = None
    stats_dir = os.path.join(aggregate_dir, 'data-stats')
    src_data_dir = os.path.join(stats_dir, 'tmp_data')

    fbasename = 'neuraldata_%s_corrected_%s-thr-%.2f' \
                % (traceid, responsive_test, responsive_thr)
    dfns = glob.glob(os.path.join(src_data_dir, '%s_*.pkl' % fbasename))
    print(fbasename)
    print(dfns)

    src_datafile = os.path.join(src_data_dir, dfns[0])
    with open(src_datafile, 'rb') as f:
        D = pkl.load(f, encoding='latin1')

    return D

def get_data(traceid='traces001', responsive_test='ROC', responsive_thr=0.05):
    D = load_source_dataframes(traceid=traceid, 
            responsive_test=responsive_test, responsive_thr=responsive_thr)

    DATA = D['DATA']
    sdata = D['sdata']
    SDF = D['SDF']
    selective_df = None

    return DATA, SDF, selective_df



def get_morph_levels(midp=53, levels=[-1, 0, 14, 27, 40, 53, 66, 79, 92, 106]):

    a_morphs = np.array(sorted([m for m in levels if m<midp and m!=-1]))[::-1]
    b_morphs = np.array(sorted([m for m in levels if m>midp and m!=-1]))

    d1 = dict((k, list(a_morphs).index(k)+1) for k in a_morphs)
    d2 = dict((k, list(b_morphs).index(k)+1) for k in b_morphs)
    morph_lut = d1.copy()
    morph_lut.update(d2)
    morph_lut.update({midp: 0, -1: -1})

    return morph_lut, a_morphs, b_morphs

def split_sample_half(g):
    nt = int(np.floor(len(g['trial'].unique())/2.))
    d1 = g[['trial', 'response']].sample(n=nt, replace=False)
    d2 = g[~g['trial'].isin(d1['trial'])].sample(n=nt)[['trial', 'response']]

    #pB = len(np.where(d1['response'].values>d2['response'].values)[0])/float(len(d1))
    return d1, d2



def get_hits_and_fas(resp_stim, resp_bas, n_crit=50):
    
    # Get N conditions
    n_conditions = len(resp_stim) #curr_cfg_ixs)
 
    # Set criterion (range between min/max response)
    min_val = min(list(itertools.chain(*resp_stim))) #resp_stim.min()
    max_val = max(list(itertools.chain(*resp_stim))) #[r.max() for r in resp_stim]) #resp_stim.max() 
    crit_vals = np.linspace(min_val, max_val, n_crit)
   
    # For each crit level, calculate p > crit (out of N trials)   
    p_hits = np.array([[sum(rs > crit)/float(len(rs)) for crit in crit_vals] for rs in resp_stim])
    p_fas = np.array([[sum(rs > crit)/float(len(rs)) for crit in crit_vals] for rs in resp_bas])

    return p_hits, p_fas, crit_vals


def split_AB_morphstep(rdf, param='morphstep', Eff=None, include_ref=True, class_a=0, class_b=106):
    '''
    rdf_sz = responses at each morph level (including diff sizes)
    Eff = effective class overall (0 or 106)
    resp_B, corresonds to response distn of Effective stimulus
    resp_A, corresponds to response distn if Ineffective stimulus
    '''
    # Split responses into A and B distns at each morph step
    Eff_obj = 'A' if Eff==class_a else 'B'
    Ineff_obj = 'B' if Eff_obj=='A' else 'B'
    resp_A=[]; resp_B=[]; resp_cfgs=[]; resp_counts=[];
    
    if include_ref:
       # Split responded to morphstep=0 in half:
        split_halves = [split_sample_half(g) for c, g in rdf[rdf.object=='M'].groupby(['size', param])]
        resp_A_REF = [t[0]['response'].values for t in split_halves]
        resp_B_REF = [t[1]['response'].values for t in split_halves]
        resp_cfgs_REF = [c for c, g in rdf[rdf.object=='M'].groupby(['size', param])]
        resp_counts_REF = [g.shape[0] for c, g in rdf[rdf.object=='M'].groupby(['size', param])]
        
        # Add to resp
        resp_A.extend(resp_A_REF)
        resp_B.extend(resp_B_REF)
        resp_counts.extend(resp_counts_REF)
        resp_cfgs.extend(resp_cfgs_REF)

    # Get all the other responses
    resp_A_ = [g['response'].values for c, g in rdf[rdf.object==Ineff_obj].groupby(['size', param])]
    resp_B_ = [g['response'].values for c, g in rdf[rdf.object==Eff_obj].groupby(['size', param])]
    
    # Corresponding configs
    resp_cfgs1_ = [c for c, g in rdf[rdf.object==Ineff_obj].groupby(['size', param])]
    resp_cfgs2_ = [c for c, g in rdf[rdf.object==Eff_obj].groupby(['size', param])]
    assert resp_cfgs1_==resp_cfgs2_, \
        "ERR: morph levels and sizes don't match for object A and B"
    # Corresponding counts
    resp_counts1_ = [g.shape[0] for c, g in rdf[rdf.object==Ineff_obj].groupby(['size', param])]
    resp_counts2_ = [g.shape[0] for c, g in rdf[rdf.object==Eff_obj].groupby(['size', param])]
    assert resp_counts1_==resp_counts2_, \
        "ERR: Trial counts don't match for object A and B"
    
    resp_cfgs.extend(resp_cfgs1_)
    resp_counts.extend(resp_counts1_)
    
    return resp_A, resp_B, resp_cfgs, resp_counts


def split_AB_morphlevel(rdf, param='morphlevel', Eff=None, include_ref=True, class_a=0, class_b=106):
    '''
    rdf_sz = responses at each morph level (~ including diff sizes)
    Eff = effective class overall (0 or 106)
    resp_B, corresonds to response distn of Effective stimulus
    resp_A, corresponds to response distn if Ineffective stimulus
    '''
    Ineff = class_b if Eff==class_a else class_a
    resp_A=[]; resp_B=[]; resp_cfgs=[]; resp_counts=[];

    # Responses to Everythign that's not "Ineffective" stimuli
    resp_B = [g['response'].values for c, g in rdf[rdf['morphlevel']!=Ineff].groupby(['size', param])]
    resp_cfgs = [c for c, g in rdf[rdf['morphlevel']!=Ineff].groupby(['size', param])]
    resp_counts = [g.shape[0] for c, g in rdf[rdf['morphlevel']!=Ineff].groupby(['size', param])]

    # Responses to "Ineffective" (baseline distN)
    if include_ref:
        # Split responses to Eff in half
        split_halves = [split_sample_half(g) for c, g in rdf[rdf['morphlevel']==Ineff].groupby(['size', param])]
        resp_A = [t[0]['response'].values for t in split_halves]
        resp_A_REF = [t[1]['response'].values for t in split_halves]
        resp_cfgs_REF = [c for c, g in rdf[rdf['morphlevel']==Ineff].groupby(['size', param])]
        resp_counts_REF = [g.shape[0] for c, g in rdf[rdf['morphlevel']==Ineff].groupby(['size', param])]

        # Add to list of comparison DistNs
        resp_B.extend(resp_A_REF)
        resp_cfgs.extend(resp_cfgs_REF)
        resp_counts.extend(resp_counts_REF)
    else:
        resp_A = [g['response'].values for c, g in rdf[rdf['morphlevel']==Ineff].groupby(['size', param])]
        

    return resp_A, resp_B, resp_cfgs, resp_counts

            
def split_signal_distns(rdf, param='morphlevel', n_crit=50, include_ref=True, Eff=None,
                       class_a=0, class_b=106):
    '''
    param=='morphstep':
        Compare objectA vs B distributions at each morph step 
        Levels: 0=53/53, 1=40/66, 2=27/79, 3=4=14/92, 4=0/106
        Split trials into halves to calculate "chance" distN (evens/odds)
    param=='morph_ix' or 'morphlevel'
        Compare Ineffective distN against Effective distNs.
        Eff = prefered A or B, Ineff, the other object.
        Split trials into halfs for Ineff distN (evens/odds)
        
    '''
    Ineff=class_b if Eff==class_a else class_a
    if param=='morphstep':
        resp_A, resp_B, resp_cfgs, resp_counts = split_AB_morphstep(rdf, param=param, Eff=Eff, include_ref=include_ref)
        
    else:
        resp_A, resp_B, resp_cfgs, resp_counts = split_AB_morphlevel(rdf, param=param, 
                                                                    Eff=Eff, include_ref=include_ref)

    p_hits, p_fas, crit_vals = get_hits_and_fas(resp_B, resp_A, n_crit=n_crit)
    
    return p_hits, p_fas, resp_cfgs, resp_counts
    
    
def calculate_auc(p_hits, p_fas, resp_cfgs1, reverse_eff=False, Eff=None):
    if p_fas.shape[0] < p_hits.shape[0]:
        altconds = list(np.unique([c[0] for c in resp_cfgs1]))
        true_auc = [-np.trapz(p_hits[ci, :], x=p_fas[altconds.index(sz), :]) \
                            for ci, (sz, mp) in enumerate(resp_cfgs1)]
    else:
        true_auc = [-np.trapz(p_hits[ci, :], x=p_fas[ci, :]) for ci in np.arange(0, len(resp_cfgs1))]
        
    aucs = pd.DataFrame({'AUC': true_auc, 
                          param: [r[1] for r in resp_cfgs1], 
                         'size': [r[0] for r in resp_cfgs1]}) 
    if reverse_eff and Eff==0:
        # flip
        for sz, a in aucs.groupby(['size']):
            aucs.loc[a.index, 'AUC'] = a['AUC'].values[::-1]
        
    return aucs


def get_auc_AB(rdf, param='morphlevel', n_crit=50, include_ref=True, allow_negative=True,
                  class_a=0, class_b=106, return_probs=False):
    '''
    Calculate AUCs for A vs. B at each size. 
    Note:  rdf must contain columns 'morphstep' and 'size' (morphstep LUT from: get_morph_levels())

    include_ref: include morphlevel=0 (or morph_ixx) by splitting into half.
    Compare p_hit (morph=0) to p_fa (morph=106), calculate AUC.
    '''
    # Get Eff/Ineff
    means = rdf[rdf.morphlevel.isin([class_a, class_b])].groupby(['object']).mean()
    Eff = class_a if means['response']['A'] > means['response']['B'] else class_b

    p_hits, p_fas, resp_cfgs, counts = split_signal_distns(rdf, param=param, n_crit=n_crit, 
                                                        include_ref=include_ref, Eff=Eff)
        
    aucs =  calculate_auc(p_hits, p_fas, resp_cfgs, reverse_eff=not(allow_negative), Eff=Eff)
    aucs['n_trials'] = counts
    aucs['Eff'] = Eff
    
    if return_probs:
        return aucs, p_hits, p_fas, resp_cfgs
    else:
        return aucs.sort_values(by=['size', 'morphlevel']).reset_index()

def plot_auc_for_cell(rdf, param='morphlevel', class_a=0, class_b=106, n_crit=50, include_ref=True, cmap='RdBu'):
    
    means = rdf[rdf.morphlevel.isin([class_a, class_b])].groupby(['object']).mean()
    print(means)
    # Get Eff/Ineff
    Eff = class_a if means['response']['A'] > means['response']['B'] else class_b

    # Calculate p_hits/p_fas for plot
    p_hits, p_fas, resp_cfgs1 = split_signal_distns(rdf, param=param, n_crit=n_crit, 
                                                    include_ref=include_ref, Eff=Eff)
    print(p_hits.shape, p_fas.shape, len(resp_cfgs1))
    aucs = calculate_auc(p_hits, p_fas, resp_cfgs1, reverse_eff=False, Eff=Eff)

    # Plot----
    mdiffs = sorted(aucs[param].unique())
    mdiff_colors = sns.color_palette(cmap, n_colors=len(mdiffs))
    colors = dict((k, v) for k, v in zip(mdiffs, mdiff_colors))

    fig, axn = pl.subplots(1, len(sizes), figsize=(8,3))    
    for ci, (sz, mp) in enumerate(resp_cfgs1):
        si = sizes.index(sz)
        ax=axn[si]
        if param=='morphstep':
            ax.plot(p_fas[ci, :], p_hits[ci, :], color=colors[mp], label=mp)
        else:
            ax.plot(p_fas[si, :], p_hits[ci, :], color=colors[mp], label=mp)
        ax.set_title(sz)
        ax.set_aspect('equal')
        ax.plot([0, 1], [0, 1], linestyle=':', color='k', lw=1)
    ax.legend(bbox_to_anchor=(1, 1.2), loc='lower right', title=param, ncol=5)
    pl.subplots_adjust(left=0.1, right=0.9, bottom=0.2, hspace=0.5, wspace=0.5, top=0.7)
    return fig

def data_matrix_from_auc(auc_, param='morphlevel', normalize=False):
    auc_['n_chooseB'] = auc_['AUC'] * auc_['n_trials']
    auc_['n_chooseB'] = np.ceil(auc_['n_chooseB']) #.astype(int)
    
    if normalize:
        maxv = float(auc_[param].max())
        auc_[param] = auc_[param]/maxv
    
    sort_cols = [param]
    if 'size' in auc_.columns:
        sort_cols.append('size')
        
    data = auc_.sort_values(by=sort_cols)[[param, 'n_chooseB', 'n_trials']].values

    return data

def aggregate_AUC(DATA, SDF, param='morphlevel', midp=53, reverse_eff=False,
                  selective_only=False, selective_df=None, create_new=False):
    tmp_res = '/n/coxfs01/julianarhee/aggregate-visual-areas/data-stats/tmp_data/AUC.pkl'
    if not create_new:
        try:
            with open(tmp_res, 'rb') as f:
                AUC = pkl.load(f, encoding='latin1')
                print(AUC.head())
        except Exception as e:
            create_new = True

    if create_new:
        print("... creating new AUC dfs")
        AUC = create_auc_dataframe(DATA, SDF, param=param, midp=midp)
        with open(tmp_res, 'wb') as f:
            pkl.dump(AUC, f, protocol=2)

    if selective_only:
        print("... getting selective only")
        assert selective_df is not None, \
            "[ERR]. Requested SELECTIVE ONLY. Must provide selective_df. Aborting."
        
        mAUC = AUC.copy()
        AUC = pd.concat([mAUC[(mAUC.visual_area==va) & (mAUC.datakey==dk) & (mAUC['cell'].isin(sg['cell'].unique()))] \
                for (va, dk), sg in selective_df.groupby(['visual_area', 'datakey'])])
        
    if reverse_eff:
        print("... reversing")
        mAUC = AUC.copy()
        to_reverse = mAUC[mAUC['Eff']==0].copy()
        for (va, dk, c, sz), auc_ in to_reverse.groupby(['visual_area', 'datakey', 'cell', 'size']):
            # flip
            AUC.loc[auc_.index, 'AUC'] = auc_['AUC'].values[::-1]

    return AUC


def create_auc_dataframe(DATA, SDF, param='morphlevel', midp=53):
    '''
    DATA: neuraldata dataframe (all data)
    SDF: dict of stimconfig dfs
    reverse_eff:  set False to allow negative sigmoid
    selective_only:  Must provide dataframe of selective cells if True
    '''
    a_=[]
    DATA['cell'] = DATA['cell'].astype(int)
    for (va, dk), nd in DATA.groupby(['visual_area', 'datakey']):
        # add stimulus info
        sdf = SDF[dk].copy()
        morphlevels = sdf['morphlevel'].unique()
        max_morph = max(morphlevels)

        sizes = list(sdf['size'].unique())
        if midp not in morphlevels:
            print("... (%s, %s) Unknown midp in morphs: %s" % (dk, va, str(morphlevels)))
            continue
        nd['size'] = [sdf['size'][c] for c in nd['config']]
        nd['morphlevel'] = [sdf['morphlevel'][c] for c in nd['config']]
        ndf = nd[(nd['morphlevel']!=-1)].copy()

        morph_lut, a_morphs, b_morphs = get_morph_levels(levels=morphlevels, midp=midp)
        # update neuraldata
        ndf['morphstep'] = [morph_lut[m] for m in ndf['morphlevel']]
        ndf['morph_ix'] = [m/float(max_morph) for m in ndf['morphlevel']]

        ndf['object'] = None
        ndf.loc[ndf.morphlevel.isin(a_morphs), 'object'] = 'A'
        ndf.loc[ndf.morphlevel.isin(b_morphs), 'object'] = 'B'
        ndf.loc[ndf.morphlevel==midp, 'object'] = 'M'

        # calculate AUCs
        # reverse_eff = param!='morphstep'
        AUC0 = ndf.groupby('cell').apply(get_auc_AB, param=param, reverse_eff=False)
        AUC0['visual_area'] = va
        AUC0['datakey'] = dk
        a_.append(AUC0)

    mAUC = pd.concat(a_).reset_index() #.drop('level_1', axis=1)
    
    return mAUC
        

def do_neurometric(curr_visual_area, curr_datakey, param='morphlevel', 
                    sigmoid='gauss', fit_experiment='2AFC', allow_negative=True,
                    max_auc=0.70, fit_new=False, create_auc=False, 
                    traceid='traces001', responsive_test='ROC', responsive_thr=0.05,
                    experiment='blobs'):
    # Load source data
    DATA, SDF, selective_df = get_data(traceid=traceid, 
                    responsive_test=responsive_test, responsive_thr=responsive_thr)
    assert curr_visual_area in DATA['visual_area'].unique(), "Visual area <%s> not in DATA." % curr_visual_area
    assert curr_datakey in DATA['datakey'].unique(), "Datakey <%s> not in DATA" % curr_datakey

    fit_neurometric_curves(curr_visual_area, curr_datakey, DATA, SDF, 
                            param=param, sigmoid=sigmoid, max_auc=max_auc, 
                            fit_experiment=fit_experiment, allow_negative=allow_negative,
                            fit_new=fit_new, create_auc=create_auc,
                            traceid=traceid, experiment=experiment)
    return

#
#def fit_neurometric_curves(curr_visual_area, curr_datakey, DATA, SDF, 
#                            param='morphlevel', sigmoid='gauss', fit_experiment='2AFC',
#                            allow_negative=True,
#                            max_auc=0.7, fit_new=False, create_auc=False,
#                            traceid='traces001', experiment='blobs'):
#    fitopts = dict()
#    fitopts['expType'] = fit_experiment
#    fitopts['threshPC'] = 0.5 
#
#    # Load AUC
#    AUC = aggregate_AUC(DATA,SDF, param=param, midp=53,reverse_eff=False,
#                        selective_only=False, selective_df=None, create_new=create_auc)
#
#    currAUC = AUC[(AUC.visual_area==curr_visual_area) 
#                & (AUC.datakey==curr_datakey)].copy()
#
#    # Set output dir
#    traceid_dir = get_tracedir_from_datakey(curr_datakey, 
#                    traceid=traceid, experiment=experiment)
#    curr_dst_dir = os.path.join(traceid_dir, 'neurometric', 'fits')
#    if not os.path.exists(curr_dst_dir):
#        os.makedirs(curr_dst_dir)
#    print("Saving ROI results to:\n   %s" % curr_dst_dir)
#
#    # Cells that pass performance criterion
#    pass_cells = currAUC[currAUC['AUC']>=max_auc]['cell'].unique()
#
#    print("%i of %i cells pass crit (%.2f)" \
#        % (len(pass_cells), len(currAUC['cell'].unique()), max_auc))
#    pass_auc = currAUC[currAUC['cell'].isin(pass_cells)].copy()
#    if len(pass_cells)==0:
#        print("****[%s, %s] no cells." % (va, dk))
#        return 
#
#    if create_auc or fit_new:
#        old_fns = glob.glob(os.path.join(curr_dst_dir, '*_rid*.pkl'))
#        print("~~~~ deleting %i old files ~~~~~" % len(old_fns))
#        for o_ in old_fns:
#            os.remove(o_)
#
#    for ri, (rid, auc_r) in enumerate(pass_auc.groupby(['cell'])):
#        if ri%10==0:
#            print("... fitting %i of %i cells" % (int(ri+1), len(pass_cells)))
#
#        Eff = int(auc_r['Eff'].unique())
#        sigmoid_ = 'neg_%s' % sigmoid if (Eff==0 and allow_negative) else sigmoid
#        fitopts['sigmoidName'] = sigmoid_
#
#        fn = '%s_rid%03d.pkl' % (sigmoid_, rid)
#        outfile = os.path.join(curr_dst_dir, fn)
#        if os.path.exists(outfile):
#            continue
#
#        results={}
#        for sz, auc_sz in auc_r.groupby(['size']):
#            # format data
#            data = data_matrix_from_auc(auc_sz, param=param)
#
#            # fit
#            res = ps.psignifit(data, fitopts)
#            results[sz] = res
#
#        fn = '%s_rid%03d.pkl' % (sigmoid_, rid)
#        outfile = os.path.join(curr_dst_dir, fn)
#
#        with open(outfile, 'wb') as f:
#            pkl.dump(results, f, protocol=2)
#        print("... saved: %s" % fn)
#
#    print("DONE!")
#    return None
#
#
def fit_neurometric_curves(curr_visual_area, curr_datakey, DATA, SDF, 
                            param='morphlevel', sigmoid='gauss', fit_experiment='2AFC',
                            allow_negative=True,
                            max_auc=0.7, fit_new=False, create_auc=False,
                            traceid='traces001', experiment='blobs'):
    fitopts = dict()
    fitopts['expType'] = fit_experiment
    fitopts['threshPC'] = 0.5 

    # Load AUC
    AUC = aggregate_AUC(DATA,SDF, param=param, midp=53,reverse_eff=False,
                        selective_only=False, selective_df=None, create_new=create_auc)

    currAUC = AUC[(AUC.visual_area==curr_visual_area) 
                & (AUC.datakey==curr_datakey)].copy()

    # Set output dir
    traceid_dir = get_tracedir_from_datakey(curr_datakey, 
                    traceid=traceid, experiment=experiment)
    curr_dst_dir = os.path.join(traceid_dir, 'neurometric', 'fits')
    if not os.path.exists(curr_dst_dir):
        os.makedirs(curr_dst_dir)
    print("Saving ROI results to:\n   %s" % curr_dst_dir)

    # Cells that pass performance criterion
    pass_cells = currAUC[currAUC['AUC']>=max_auc]['cell'].unique()

    print("%i of %i cells pass crit (%.2f)" \
        % (len(pass_cells), len(currAUC['cell'].unique()), max_auc))
    pass_auc = currAUC[currAUC['cell'].isin(pass_cells)].copy()
    if len(pass_cells)==0:
        print("****[%s, %s] no cells." % (va, dk))
        return 

    if create_auc or fit_new:
        old_fns = glob.glob(os.path.join(curr_dst_dir, '*_rid*.pkl'))
        print("~~~~ deleting %i old files ~~~~~" % len(old_fns))
        for o_ in old_fns:
            os.remove(o_)

    for ri, (rid, auc_r) in enumerate(pass_auc.groupby(['cell'])):
        if ri%10==0:
            print("... fitting %i of %i cells" % (int(ri+1), len(pass_cells)))

        Eff = int(auc_r['Eff'].unique())
        sigmoid_ = 'neg_%s' % sigmoid if (Eff==0 and allow_negative) else sigmoid
        fitopts['sigmoidName'] = sigmoid_

        fn = '%s_rid%03d.pkl' % (sigmoid_, rid)
        outfile = os.path.join(curr_dst_dir, fn)
        if os.path.exists(outfile):
            continue

        results={}
        for sz, auc_sz in auc_r.groupby(['size']):
            # format data
            data = data_matrix_from_auc(auc_sz, param=param)
            # fit
            res = ps.psignifit(data, fitopts)
            results[sz] = res

        fn = '%s_rid%03d.pkl' % (sigmoid_, rid)
        outfile = os.path.join(curr_dst_dir, fn)

        with open(outfile, 'wb') as f:
            pkl.dump(results, f, protocol=2)
        print("... saved: %s" % fn)

    print("DONE!")
    
    return None



def extract_options(options):
    parser = optparse.OptionParser()

    # PATH opts:
    parser.add_option('-D', '--root', action='store', dest='rootdir', default='/n/coxfs01/2p-data', 
                      help='root project dir containing all animalids [default: /n/coxfs01/2pdata]')
  
    # Set specific session/run for current animal:
    parser.add_option('-E', '--experiment', action='store', dest='experiment', default='blobs', 
                      help="experiment name (e.g,. gratings, rfs, rfs10, or blobs)") 

    choices_e = ('stimulus', 'firsthalf', 'plushalf', 'baseline')
    default_e = 'plushalf'
    parser.add_option('-e', '--epoch', action='store', dest='trial_epoch', 
            default=default_e, type='choice', choices=choices_e,
            help="Trial epoch to average, choices: %s. (default: %s" % (choices_e, default_e))

    parser.add_option('-t', '--traceid', action='store', dest='traceid', default='traces001', 
                      help="traceid (default: traces001)")

    choices_c = ('all', 'ROC', 'nstds', None, 'None')
    default_c = 'ROC'
    parser.add_option('-R', '--responsive_test', action='store', dest='responsive_test', 
            default=default_c, type='choice', choices=choices_c,
            help="Responsive test, choices: %s. (default: %s" % (choices_c, default_c))

    parser.add_option('--thr', action='store', dest='responsive_thr', default=0.05, 
                      help="responsive test thr (default: 0.05 for ROC)")

    parser.add_option('-d', '--response', action='store', dest='response_type', default='corrected', 
                      help="response type (default: corrected, for AUC)")
    parser.add_option('--nstds', action='store', dest='nstds_above', default=2.5, 
                      help="only for test=nstds, N stds above (default: 2.5)")
    parser.add_option('--auc', action='store_true', dest='create_auc', default=False, 
                      help="flag to create AUC dataframes anew")
    parser.add_option('--new', action='store_true', dest='fit_new', default=False, 
                      help="flag to fit each cell anew")
    
    parser.add_option('-n', '--nproc', action='store', dest='n_processes', 
                      default=1)
    parser.add_option('-N', '--niter', action='store', dest='n_iterations', 
                      default=100)



    parser.add_option('-p', '--param', action='store', dest='param',
                      default='morphlevel', help='Physical param for neurometric curve fit (default: morphlevel)')
    parser.add_option('-s', '--sigmoid', action='store', dest='sigmoid',
                      default='gauss', help='Sigmoid to fit (default: gauss)')
    parser.add_option('-c', '--max-auc', action='store', dest='max_auc',
                      default=0.7, help='Criterion value, max AUC (default: 0.7)')
    parser.add_option('-f', '--fit-experiment', action='store', dest='fit_experiment',
                      default='2AFC', help='Experiment type to fit (default: 2AFC)')
    parser.add_option('--reverse', action='store_false', dest='allow_negative',
                      default=True, help='Set flag to fit all to same func (default: allows neg sigmoid fits)')


    parser.add_option('-V', '--visual-area', action='store', dest='curr_visual_area',
                      default=None, help='Visual area to fit (default: None)')
    parser.add_option('-k', '--datakey', action='store', dest='curr_datakey',
                      default=None, help='Datakey to fit (default: None)')

    parser.add_option('-o', '--overlap-thr', action='store', dest='overlap_thr',
                      default=None, help='[SPLIT_PUPIL]: Overlap thr (default: None)')
    parser.add_option('--pupil-resp-type', action='store', dest='split_pupil_response_type',
                      default='dff', help='[SPLIT_PUPIL]: Response type (default: dff)')
    
    parser.add_option('--pupil-auc', action='store_true', dest='run_auc_split_pupil',
                      default=False, help='Run AUC for all pupil split conditions')
    parser.add_option('--pupil', action='store_true', dest='split_pupil',
                      default=False, help='Fit curves from averaged AUC curves (over split_pupil iterations)')

    parser.add_option('--iter', action='store_true', dest='by_iter',
                      default=False, help='Fit curves to each iter (over split_pupil iterations)')



    (options, args) = parser.parse_args(options)

    return options


responsive_test='ROC'
responsive_thr=10. if responsive_test=='nstds' else 0.05
experiment='blobs'

traceid = 'traces001'
rootdir = '/n/coxfs01/2p-data'

fit_new=False
create_auc=False
param = 'morphlevel'
max_auc = 0.70

curr_visual_area = 'V1'
curr_datakey = '20190616_JC097_fov2'

def main(options):
    opts = extract_options(options)
    rootdir = opts.rootdir
    experiment = opts.experiment
    responsive_test=opts.responsive_test
    responsive_thr = float(opts.responsive_thr)
    traceid = opts.traceid
    
    max_auc = float(opts.max_auc)
    curr_visual_area = opts.curr_visual_area
    curr_datakey = opts.curr_datakey
    create_auc = opts.create_auc
    fit_new = opts.fit_new
    
    param = opts.param
    sigmoid = opts.sigmoid
    fit_experiment = opts.fit_experiment
    allow_negative = opts.allow_negative

    run_auc_split_pupil = opts.run_auc_split_pupil
    trial_epoch = opts.trial_epoch
    overlap_thr = opts.overlap_thr
    src_response_type = opts.split_pupil_response_type
    n_processes = int(opts.n_processes)
    n_iterations = None if opts.n_iterations in [None, 'None'] else int(opts.n_iterations)
    split_pupil = opts.split_pupil
    by_iter = opts.by_iter

    if run_auc_split_pupil:
        do_split_pupil_auc(curr_visual_area, curr_datakey, param=param, sigmoid=sigmoid, 
                        fit_experiment=fit_experiment, allow_negative=allow_negative,
                        max_auc=max_auc, fit_new=fit_new, create_auc=create_auc,
                        traceid=traceid, experiment=experiment,
                        responsive_test=responsive_test, responsive_thr=responsive_thr,
                        src_response_type=src_response_type, 
                        overlap_thr=overlap_thr, trial_epoch=trial_epoch, 
                        n_processes=n_processes, n_iterations=n_iterations)
    else:
        if split_pupil:
            do_split_pupil_fits(curr_visual_area, curr_datakey, param=param,sigmoid=sigmoid, 
                        fit_experiment=fit_experiment, allow_negative=allow_negative,
                        max_auc=max_auc, fit_new=fit_new, by_iter=by_iter,  
                        traceid=traceid, experiment=experiment,
                        responsive_test=responsive_test, responsive_thr=responsive_thr,
                        src_response_type=src_response_type, 
                        overlap_thr=overlap_thr, trial_epoch=trial_epoch,
                        n_processes=n_processes, n_iterations=n_iterations)
        else:
            do_neurometric(curr_visual_area, curr_datakey, param=param, sigmoid=sigmoid, 
                        fit_experiment=fit_experiment, allow_negative=allow_negative,
                        max_auc=max_auc, fit_new=fit_new, create_auc=create_auc,
                        traceid=traceid, experiment=experiment,
                        responsive_test=responsive_test, responsive_thr=responsive_thr)


    print('.done.')


if __name__ == '__main__':
    main(sys.argv[1:])
