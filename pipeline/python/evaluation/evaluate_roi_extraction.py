#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 12:12:03 2018

@author: julianarhee
"""
import matplotlib
matplotlib.use('Agg')
import os
import sys
import re
import h5py
import datetime
import traceback
import json
import pprint
import optparse
import math
import time
import caiman as cm
import pylab as pl
import numpy as np
import multiprocessing as mp
from pipeline.python.rois.utils import load_RID, get_source_paths, get_source_info
from caiman.utils.visualization import get_contours, plot_contours
from pipeline.python.utils import natural_keys, print_elapsed_time
from caiman.components_evaluation import evaluate_components, estimate_components_quality_auto

pp = pprint.PrettyPrinter(indent=4)


#%%
def evaluate_rois_nmf(mmap_path, nmfout_path, evalparams, eval_outdir, asdict=False):
    """
    Uses component quality evaluation from caiman.
    mmap_path : (str)
        path to mmap files of movies (.mmap)
    nmfout_path : (str)
        path to nmf output files from initial nmf extraction (.npz)
    evalparams : (dict)
        params for evaluation function
        'frame_rate' :  frame rate in Hz
        'rval_thr' :  accept components with space correlation threshold of this or higher (spatial consistency)
        'min_SNR' :  accept components with peak-SNR of this or higher
        'decay_time' :  length of typical transient (in secs)
        'gSig' :  (int) half-size of neuron
        'use_cnn' :  CNN classifier (unused)
        'Npeaks': number of local maxima to consider
        'N': N number of consecutive events (# number of timesteps to consider when testing new neuron candidates)
        
        
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
    
    print "----------------------------------"
    print "EVALUATING ROIS with params:"
    for k in evalparams.keys():
        print k, ':', evalparams[k]
    print "----------------------------------"

    #%start a cluster for parallel processing
    try:
        dview.terminate() # stop it if it was running
    except:
        pass
    
    c, dview, n_processes = cm.cluster.setup_cluster(backend='local', # use this one
                                                     n_processes=None,  # number of process to use, reduce if out of mem
                                                     single_thread = False)
    try:
        Yr, dims, T = cm.load_memmap(mmap_path)
        images = np.reshape(Yr.T, [T] + list(dims), order='F')
    
        pass_comps, fail_comps, snr_vals, r_vals, cnn_preds = \
            estimate_components_quality_auto(images, nmf['A'].all(), nmf['C'], 
                                             nmf['b'], nmf['f'], nmf['YrA'],
                                             evalparams['frame_rate'],
                                             evalparams['decay_time'],
                                             evalparams['decay_time'],
                                             dims, 
                                             dview=dview,
                                             min_SNR=evalparams['min_SNR'],        # accept components with peak-SNR of this or higher
                                             r_values_min=evalparams['rval_thr'],  # accept components with space corr threshold or higher
                                             Npeaks=evalparams['Npeaks'],
                                             use_cnn=evalparams['use_cnn'])
        
        print "%s: evalulation results..." % curr_file
        print(('Should keep ' + str(len(pass_comps)) +
           ' and discard  ' + str(len(fail_comps))))
        
        #% PLOT: Visualize Spatial and Temporal component evaluation ----------
        print "Plotting evaluation output..."
        pl.figure(figsize=(5,15))
        pl.subplot(2,1,1); pl.title('r values (spatial)'); 
        pl.plot(r_vals); pl.plot(range(len(r_vals)), np.ones(r_vals.shape)*evalparams['rval_thr'], 'r')
        pl.subplot(2,1,2); pl.title('SNR_comp'); 
        pl.plot(snr_vals); pl.plot(range(len(snr_vals)), np.ones(r_vals.shape)*evalparams['min_SNR'], 'r')
        pl.xlabel('roi')
        pl.suptitle(curr_file)
        pl.savefig(os.path.join(eval_outdir_figs, 'eval_metrics_%s.png' % curr_file))
        pl.close()
        # ---------------------------------------------------------------------
        
        
        # PLOT: Iteration 1 - Show components that pass/fail evaluation metric --------------
        print "Plotting contours..."
        pl.figure();
        #pl.subplot(1,2,1); 
        pl.title('pass'); 
        plot_contours(nmf['A'].all()[:, pass_comps], nmf['Av'], thr=0.85); pl.axis('off')
        #pl.subplot(1,2,2); pl.title('fail'); 
        #plot_contours(nmf['A'].all()[:, idx_components_bad], nmf['Av'], thr=0.85); pl.axis('off')
        pl.savefig(os.path.join(eval_outdir_figs, 'eval_contours_%s.png' % curr_file))
        pl.close()
        # ---------------------------------------------------------------------
        
    except Exception as e:
        print "ERROR evaluating NMF components."
        traceback.print_exc()
        print "Aborting"
    finally:
        try:
            dview.terminate() # stop it if it was running
        except:
            pass
    
    if asdict is True:
        evalresults = dict()
        evalresults['pass_rois'] = pass_comps
        evalresults['fail_rois'] = fail_comps
        evalresults['snr_vals'] = snr_vals
        evalresults['r_vals'] = r_vals
        evalresults['cnn_preds'] = cnn_preds
        return evalresults
    else:
        return pass_comps, fail_comps, snr_vals, r_vals, cnn_preds

#%%
def mp_evaluator_nmf(srcfiles, evalparams, roi_eval_dir, nprocs=12):
    
    t_eval_mp = time.time()
    
    filenames = sorted(srcfiles.keys(), key=natural_keys)
    def worker(filenames, srcfiles, evalparams, roi_eval_dir, out_q):
        """
        Worker function is invoked in a process. 'filenames' is a list of 
        filenames to evaluate [File001, File002, etc.]. The results are placed
        in a dict that is pushed to a queue.
        """
        outdict = {}
        for fn in filenames:
            outdict[fn] = evaluate_rois_nmf(srcfiles[fn][0], srcfiles[fn][1], evalparams, roi_eval_dir, asdict=True)
            print "Worker: Done with %s" % fn
        out_q.put(outdict)
    
    # Each process gets "chunksize' filenames and a queue to put his out-dict into:
    out_q = mp.Queue()
    chunksize = int(math.ceil(len(filenames) / float(nprocs)))
    procs = []
    
    for i in range(nprocs):
        p = mp.Process(target=worker,
                       args=(filenames[chunksize * i:chunksize * (i + 1)],
                                       srcfiles,
                                       evalparams,
                                       roi_eval_dir,
                                       out_q))
        procs.append(p)
        p.start()
        
    # Wait for all worker processes to finish
    for p in procs:
        print "Finished:", p
        p.join()
        
    # Collect all results into single results dict. We should know how many dicts to expect:
    resultdict = {}
    for i in range(nprocs):
        resultdict.update(out_q.get())
    
#    # Wait for all worker processes to finish
#    for p in procs:
#        print "Finished:", p
#        p.join()
#    
    print_elapsed_time(t_eval_mp)
    
    return resultdict

#%%
def set_evalparams_nmf(RID, frame_rate=None, decay_time=1.0, min_SNR=1.5, rval_thr=0.6, Nsamples=10, Npeaks=5, use_cnn=False, cnn_thr=0.8):
    evalparams = {}
    evalparams['min_SNR'] = min_SNR
    evalparams['rval_thr'] = rval_thr
    evalparams['decay_time'] = decay_time
    evalparams['frame_rate'] = frame_rate
    evalparams['Nsamples'] = Nsamples
    evalparams['Npeaks'] = Npeaks
    evalparams['use_cnn'] = use_cnn
    evalparams['cnn_thr'] = cnn_thr

    if RID['roi_type'] == 'caiman2D':
        evalparams['gSig'] = RID['PARAMS']['options']['extraction']['gSig']
        if frame_rate is None:
            evalparams['frame_rate'] = RID['PARAMS']['options']['eval']['final_frate']
    else:
        session_dir = os.path.split(os.path.split(RID['DST'])[0])[0]
        session = os.path.split(session_dir)[-1]
        with open(os.path.join(session_dir, 'ROIs', 'rids_%s.json' % session), 'r') as f:
            roidict = json.load(f)
        src_rid = RID['PARAMS']['options']['source']['roi_id']
        evalparams['gSig'] = roidict[src_rid]['PARAMS']['options']['extraction']['gSig']
        if frame_rate is None:
            evalparams['frame_rate'] = roidict[src_rid]['PARAMS']['options']['eval']['final_frate']
        
    return evalparams

#%%
def par_evaluate_rois(RID, evalparams=None, nprocs=12):
    
    session_dir = RID['DST'].split('/ROIs')[0]
    roi_type = RID['roi_type']
    roi_id = RID['roi_id']
    roi_source_dir = RID['DST']
    
    # Get ROI and TIFF source paths:
    roi_source_paths, tiff_source_paths, filenames, excluded_tiffs, mcmetrics_filepath = get_source_paths(session_dir, RID, subset=True)
    
    # Create output dir and OUTFILE:
    try: 
        tstamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        roi_eval_dir = os.path.join(roi_source_dir, 'evaluation', 'evaluation_%s' % tstamp)
        if not os.path.exists(roi_eval_dir):
            os.makedirs(roi_eval_dir)
        eval_filename = 'evaluation_results_{tstamp}.hdf5'.format(tstamp=tstamp)
        eval_filepath = os.path.join(roi_eval_dir, eval_filename)
        print "Saving roi evaluation results to file: %s" % eval_filepath
    except Exception as e:
        print "-- ERROR: Unable to create ROI evaluation dir for source:\n%s" % roi_source_dir
        traceback.print_exc()
        print "---------------------------------------------------------------"
    
    # Get file-matched list of ROI and TIFF paths:
    srcfiles = {}
    for fn in filenames:
        match_nmf = [f for f in roi_source_paths if fn in f][0]
        match_mmap = [f for f in tiff_source_paths if fn in f][0]
        srcfiles[fn] = ((match_mmap, match_nmf))

    print "-----------------------------------------------------------"
    print "Evalating ROIS from set: %s" % roi_id
    print "EVAL parameters:"
    for k in evalparams.keys():
        print k, ':', evalparams[k]
    print "-----------------------------------------------------------"

    #%
    t_par = time.time()
    try:
        evalfile = h5py.File(eval_filepath, 'a')
        for k in evalparams.keys():
            print k, evalparams[k]
            evalfile.attrs[k] = evalparams[k]
        evalfile.attrs['creation_date'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if roi_type == 'caiman2D':
            RESULTS = mp_evaluator_nmf(srcfiles, evalparams, roi_eval_dir, nprocs=nprocs)
        
        for fn in RESULTS.keys():
            filegrp = evalfile.create_group(fn)
            filegrp.attrs['tiff_source'] = srcfiles[fn][1]
            filegrp.attrs['roi_source'] = srcfiles[fn][0]
            
            good_dset = filegrp.create_dataset('pass_rois', RESULTS[fn]['pass_rois'].shape, RESULTS[fn]['pass_rois'].dtype)
            good_dset[...] = RESULTS[fn]['pass_rois']
            bad_dset = filegrp.create_dataset('fail_rois', RESULTS[fn]['fail_rois'].shape, RESULTS[fn]['fail_rois'].dtype)
            bad_dset[...] = RESULTS[fn]['fail_rois']
            snr_dset = filegrp.create_dataset('snr_values', RESULTS[fn]['snr_vals'].shape, RESULTS[fn]['snr_vals'].dtype)
            snr_dset[...] = RESULTS[fn]['snr_vals']
            snr_dset.attrs['min_SNR'] = evalparams['min_SNR']
            rcorr_dset = filegrp.create_dataset('rcorr_values', RESULTS[fn]['r_vals'].shape, RESULTS[fn]['r_vals'].dtype)
            rcorr_dset[...] = RESULTS[fn]['r_vals']
            rcorr_dset.attrs['min_rval'] = evalparams['rval_thr']
            
    except Exception as e:
        print "MP evaluator bugged out."
        traceback.print_exc()
        print "Aborting evaluation..."
    finally:
        evalfile.close()
    
    print "Finished ROI evaluation step. ROI eval info saved to:"
    print eval_filepath
    print_elapsed_time(t_par)
    
    roi_source_basedir = os.path.split(roi_source_paths[0])[0]
    tiff_source_basedir = os.path.split(tiff_source_paths[0])[0]
    
    return eval_filepath, roi_source_basedir, tiff_source_basedir, excluded_tiffs

#%%
def evaluate_roi_set(RID, evalparams=None):
    
    session_dir = RID['DST'].split('/ROIs')[0]
    roi_id = RID['roi_id']
    
    # Create output dir:
    try: 
        roi_type = RID['roi_type']
        roi_id = RID['roi_id']
        roi_source_dir = RID['DST']
        tstamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        
        roi_eval_dir = os.path.join(roi_source_dir, 'evaluation', 'evaluation_%s' % tstamp)
        if not os.path.exists(roi_eval_dir):
            os.makedirs(roi_eval_dir)
        print "Saving evaluation output to: %s" % roi_eval_dir
    except Exception as e:
        print "-- ERROR: Unable to create ROI evaluation dir-------------------"
        traceback.print_exc()
        print "---------------------------------------------------------------"
    
    roi_source_paths, tiff_source_paths, filenames, excluded_tiffs, mcmetrics_filepath = get_source_paths(session_dir, RID)
    
    # Set up output file:
    eval_filename = 'evaluation_results_{tstamp}.hdf5'.format(tstamp=tstamp)
    eval_filepath = os.path.join(roi_eval_dir, eval_filename)
    print "Saving roi evaluation results to file: %s" % eval_filepath
    
    # Get file-matched list of ROI and TIFF paths:
    src_file_list = []
    for fn in filenames:
        match_nmf = [f for f in roi_source_paths if fn in f][0]
        match_mmap = [f for f in tiff_source_paths if fn in f][0]
        src_file_list.append((match_mmap, match_nmf))

    print "-----------------------------------------------------------"
    print "Evalating ROIS from set: %s" % roi_id
    print "Parameters:"
    for k in evalparams.keys():
        print k, ':', evalparams[k]
    print "-----------------------------------------------------------"

    #%
    try:
        evalfile = h5py.File(eval_filepath, 'a')
        for k in evalparams.keys():
            print k, evalparams[k]
            evalfile.attrs[k] = evalparams[k]
        evalfile.attrs['creation_date'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        #evalfile.attrs['excluded_tiffs'] = excluded_tiffs
        
        for src_file in src_file_list:
            
            if roi_type == 'caiman2D':
                curr_mmap_path = src_file[0]
                curr_nmfout_path = src_file[1]
                
                curr_file = str(re.search('File(\d{3})', curr_nmfout_path).group(0))
                if curr_file in evalfile.keys():
                    filegrp = evalfile[curr_file]
                else:
                    filegrp = evalfile.create_group(curr_file)
                    filegrp.attrs['tiff_source'] = curr_mmap_path
                    filegrp.attrs['roi_source'] = curr_nmfout_path
            
                good, bad, snr_vals, r_vals, cnn_preds = evaluate_rois_nmf(curr_mmap_path, curr_nmfout_path, evalparams, roi_eval_dir)
    
                # Save Eval function output to file:
                good_dset = filegrp.create_dataset('pass_rois', good.shape, good.dtype)
                good_dset[...] = good
                bad_dset = filegrp.create_dataset('fail_rois', bad.shape, bad.dtype)
                bad_dset[...] = bad
                snr_dset = filegrp.create_dataset('snr_values', snr_vals.shape, snr_vals.dtype)
                snr_dset[...] = snr_vals
                snr_dset.attrs['min_SNR'] = evalparams['min_SNR']
                rcorr_dset = filegrp.create_dataset('rcorr_values', r_vals.shape, r_vals.dtype)
                rcorr_dset[...] = r_vals
                rcorr_dset.attrs['min_rval'] = evalparams['rval_thr']
                
    except Exception as e:
        print "--- Error evaulating ROIs. Curr file: %s ---" % curr_file
        print "--> tiff source: %s" % src_file[0]
        print "--> roi source: %s" % src_file[1]
        traceback.print_exc()
        print "-----------------------------------------------------------"
    finally:
        evalfile.close()
    
    print "Finished ROI evaluation step. ROI eval info saved to:"
    print eval_filepath
    
    roi_source_basedir = os.path.split(roi_source_paths[0])[0]
    tiff_source_basedir = os.path.split(tiff_source_paths[0])[0]
    
    return eval_filepath, roi_source_basedir, tiff_source_basedir, excluded_tiffs

#%% 
def run_rid_eval(rid_filepath, nprocs=12):
    #roi_hash = os.path.splitext(os.path.split(rid_filepath)[-1])[0].split('_')[-1]
    tmp_rid_dir = os.path.split(rid_filepath)[0]
    if not os.path.exists(rid_filepath):
        rid_fn = os.path.split(rid_filepath)[1]
        completed_path = os.path.join(tmp_rid_dir, 'completed', rid_fn)
        assert os.path.exists(os.path.join(completed_path)), "No such RID file exists in either %s or %s." % (rid_filepath, completed_path)
        rid_filepath = completed_path
   
    if os.path.exists(rid_filepath):
        with open(rid_filepath, 'r') as f:
            RID = json.load(f)
        
        evalparams = set_evalparams_nmf(RID)
        eval_filepath, roi_source_basedir, tiff_source_basedir, excluded_tiffs = par_evaluate_rois(RID, evalparams=evalparams, nprocs=nprocs)
        
    return eval_filepath

#%%
def run_evaluation(options):
    # PATH opts:
    parser = optparse.OptionParser()

    parser.add_option('-D', '--root', action='store', dest='rootdir', default='/nas/volume1/2photon/data', help='data root dir (root project dir containing all animalids) [default: /nas/volume1/2photon/data, /n/coxfs01/2pdata if --slurm]')
    parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')
    parser.add_option('-S', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID')
    
    parser.add_option('--default', action='store_true', dest='default', default='store_false', help="Use all DEFAULT params, for params not specified by user (no interactive)")
    parser.add_option('--slurm', action='store_true', dest='slurm', default=False, help="set if running as SLURM job on Odyssey")
    parser.add_option('--par', action='store_true', dest='multiproc', default=False, help="set if parallel proc eval on files")
    parser.add_option('-n', '--nprocs', action='store', dest='nprocs', default=12, help="n cores for parallel processing [default: 12]")

    parser.add_option('-r', '--roi-id', action='store', dest='roi_id', default='', help="ROI ID for rid param set to use (created with set_roi_params.py, e.g., rois001, rois005, etc.)")
    
    # Evluation params:
    parser.add_option('-s', '--snr', action='store', dest='min_SNR', default=1.5, help="[nmf]: min SNR for re-evalation [default: 1.5]")
    parser.add_option('-c', '--rval', action='store', dest='rval_thr', default=0.6, help="[nmf]: min rval thresh for re-evalation [default: 0.6]")
    parser.add_option('-N', '--N', action='store', dest='Nsamples', default=10, help="[nmf]: N number of consecutive events probability multiplied [default: 10]")
    parser.add_option('-P', '--npeaks', action='store', dest='Npeaks', default=5, help="[nmf]: Number of local maxima to consider [default: 5]")
    parser.add_option('-d', '--decay', action='store', dest='decay_time', default=1.0, help="[nmf]: decay time of transients/indicator [default: 1.0]")
    parser.add_option('--cnn', action='store_true', dest='use_cnn', default=False, help="[nmf]: whether to use CNN to filter components")
    parser.add_option('-v', '--cnn_thr', action='store', dest='cnn_thr', default=0.8, help="[nmf]: all samples with probabilities larger than this are accepted [default: 0.8]")

   
    (options, args) = parser.parse_args(options)

    # Set USER INPUT options:
    rootdir = options.rootdir
    animalid = options.animalid
    session = options.session
    roi_id = options.roi_id
    
    slurm = options.slurm
    auto = options.default
    if options.slurm is True:
        if 'coxfs01' not in options.rootdir:
            options.rootdir = '/n/coxfs01/2p-data'
            
    min_SNR = float(options.min_SNR)
    rval_thr = float(options.rval_thr)
    Nsamples = int(options.Nsamples)
    Npeaks = int(options.Npeaks)
    decay_time = float(options.decay_time)
    use_cnn = options.use_cnn
    cnn_thr = float(options.cnn_thr)
    
    multiproc = options.multiproc
    nprocs = options.nprocs
    
    #%% OPTPARSE:
    rootdir = '/nas/volume1/2photon/data'
    animalid = 'JR063' #'JR063'
    session = '20171128_JR063' #'20171128_JR063'
    roi_id = 'rois008'
    slurm = False
    auto = False
    multiproc = True

    # Eval params (specific for NMF)
    min_SNR = 1.5
    rval_thr = 0.6
    use_cnn = False
    cnn_thr = 0.8
    decay_time = 1.0
    frame_rate = 44.6828
    Nsamples = 5
    Npeaks = 5
    
    #%%
    session_dir = os.path.join(rootdir, animalid, session)
    
    # Load ROI set info:
    try:
        RID = load_RID(session_dir, roi_id)
        print "Evaluating ROIs from set: %s" % RID['roi_id']
    except Exception as e:
        print "-- ERROR: unable to open source ROI dict. ---------------------"
        traceback.print_exc()
        print "---------------------------------------------------------------"
        
    # Get frame rate from runmeta info from tiff source:
    tiff_sourcedir = RID['SRC']
    acquisition = tiff_sourcedir.split(session_dir)[-1].split('/')[1]
    run = tiff_sourcedir.split(session_dir)[-1].split('/')[2]
    runinfo_filepath = os.path.join(session_dir, acquisition, run, '%s.json' % run)
    with open(runinfo_filepath, 'r') as f:
        runinfo = json.load(f)
    frame_rate = runinfo['volume_rate'] # Use volume rate since this will be the correct sample rate for planar or volumetric

    #% Get evalparams and evaluate:
    if RID['roi_type'] == 'caiman2D' or (RID['roi_type'] == 'coregister' and RID['PARAMS']['options']['source']['roi_type'] == 'caiman2D'):
        evalparams = set_evalparams_nmf(RID, frame_rate=frame_rate, min_SNR=min_SNR, rval_thr=rval_thr, decay_time=decay_time,
                                        Nsamples=Nsamples, Npeaks=Npeaks, use_cnn=use_cnn, cnn_thr=cnn_thr)
    
        if multiproc is True:
            eval_filepath, roi_source_basedir, tiff_source_basedir, excluded_tiffs = par_evaluate_rois(RID, evalparams=evalparams, nprocs=nprocs)
        else:
            eval_filepath, roi_source_basedir, tiff_source_basedir, excluded_tiffs = evaluate_roi_set(RID, evalparams=evalparams)
        
    #% Save Eval info to dict:
    evalinfo = dict()
    evalinfo['output_path'] = eval_filepath
    evalinfo['params'] = evalparams
    evalinfo['roi_type'] = RID['roi_type']
    evalinfo['roi_id'] = roi_id
    evalinfo['rid_hash'] = RID['rid_hash']
    evalinfo['roi_source'] = roi_source_basedir
    evalinfo['tiff_source'] = tiff_source_basedir
    evalinfo['nrois'] = []
    evalfile_read = h5py.File(eval_filepath, 'r')
    for fn in sorted(evalfile_read.keys(), key=natural_keys):
        curr_nrois = len(evalfile_read[fn]['pass_rois'])
        evalinfo['nrois'].append(curr_nrois)
        print "%s: %i rois" % (fn, curr_nrois)
    evalfile_read.close()
    evalinfo['excluded_files'] = excluded_tiffs
   
    min_good_rois = min(evalinfo['nrois'])
    max_good_rois = max(evalinfo['nrois'])
    
    roi_eval_dir = os.path.split(eval_filepath)[0]
    eval_base_dir = os.path.split(roi_eval_dir)[0]
    eval_info_path = os.path.join(eval_base_dir, 'evaluation_info.json')
    if os.path.exists(eval_info_path):
        with open(eval_info_path, 'r') as fr:
            evaldict = json.load(fr)
    else:
        evaldict = dict()
    eval_key = os.path.split(os.path.split(eval_filepath)[0])[1]
    print "Saving new eval key: %s" % eval_key
    #eval_filename = str(os.path.splitext(os.path.basename(eval_filepath))[0]) 
    if eval_key not in evaldict.keys():
        evaldict[eval_key] = evalinfo
    with open(eval_info_path, 'w') as fw:
        json.dump(evaldict, fw, sort_keys=True, indent=4)
    print "Updated eval info dict with entry: %s\nSaved update to %s." % (eval_key, eval_info_path)    
    
    return eval_filepath, max_good_rois, min_good_rois

#%%
def main(options):

    eval_filepath, max_good_rois, min_good_rois = run_evaluation(options)

    print "----------------------------------------------------------------"
    print "Finished ROI evaluation."
    print "Fewest N rois: %i" % min_good_rois
    print "Most N rois: %i" % max_good_rois
    print "Saved output to:"
    print eval_filepath
    print "----------------------------------------------------------------"


    #%% Get NMF output files:

if __name__ == '__main__':
    main(sys.argv[1:])
