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
import caiman as cm
import pylab as pl
import numpy as np
from pipeline.python.evaluation.evaluate_motion_correction import get_source_info
from caiman.utils.visualization import get_contours, plot_contours
from pipeline.python.utils import natural_keys
from caiman.components_evaluation import evaluate_components, estimate_components_quality_auto

pp = pprint.PrettyPrinter(indent=4)


#%%
               
#%%
#rootdir = '/nas/volume1/2photon/data'
#animalid = 'JR063' #'JR063'
#session = '20171128_JR063' #'20171128_JR063'
#roi_id = 'rois001'
#slurm = False
#auto = False
#session_dir = os.path.join(rootdir, animalid, session)
#
## Eval params (specific for NMF)
#min_SNR = 1.5
#rval_thr = 0.6
#use_cnn = False
#cnn_thr = 0.8
#decay_time = 1.0
#frame_rate = 44.6828
#Nsamples = 5
#Npeaks = 5


#%%
# =============================================================================
# Load specified ROI-ID parameter set:
# =============================================================================
#session_dir = os.path.join(rootdir, animalid, session)
#roi_base_dir = os.path.join(session_dir, 'ROIs') #acquisition, run)
#tmp_rid_dir = os.path.join(roi_base_dir, 'tmp_rids')

def load_RID(session_dir, roi_id):
    
    roi_base_dir = os.path.join(session_dir, 'ROIs') #acquisition, run)
    session = os.path.split(session_dir)[1]
    roidict_path = os.path.join(session_dir, 'ROIs', 'rids_%s.json' % session)
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
            print E
            print "---------------------------------------------------------------"

    return RID
    
#%% Load specified ROI set:
#rid_hash = RID['rid_hash']
#tiff_dir = RID['SRC']
#roi_id = RID['roi_id']
#roi_type = RID['roi_type']
#tiff_files = sorted([t for t in os.listdir(tiff_dir) if t.endswith('tif')], key=natural_keys)
#print "Found %i tiffs in dir %s.\nEvaluating %s ROIs...." % (len(tiff_files), tiff_dir, roi_type)
#
#acquisition = tiff_dir.split(session)[1].split('/')[1]
#run = tiff_dir.split(session)[1].split('/')[2]
#process_id = tiff_dir.split(session)[1].split('/')[4]
#
#filenames = ['File%03d' % int(ti+1) for ti, t in enumerate(os.listdir(tiff_dir)) if t.endswith('tif')]
#print "Source tiffs:"
#for f in filenames:
#    print f

#%% If motion-corrected (standard), check evaluation:
def check_mc_evlatuion(RID, filenames, mcmetric_type='zproj_corrcoefs'):
    
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
        tiff_dir = RID['SRC']
        session_dir = RID['DST'].split('/ROIs')      
        session = os.path.split(session_dir)[1]
        acquisition = tiff_dir.split(session)[1].split('/')[1]
        run = tiff_dir.split(session)[1].split('/')[2]
        process_id = tiff_dir.split(session)[1].split('/')[4]

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
    
    return filenames, excluded_tiffs

#%%
def evaluate_rois_nmf(mmap_path, nmfout_path, evalparams, eval_outdir):
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
    
        idx_components, idx_components_bad, SNR_comp, r_values, cnn_preds = \
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
        print(('Should keep ' + str(len(idx_components)) +
           ' and discard  ' + str(len(idx_components_bad))))
        
        #% PLOT: Visualize Spatial and Temporal component evaluation ----------
        print "Plotting evaluation output..."
        pl.figure(figsize=(5,15))
        pl.subplot(2,1,1); pl.title('r values (spatial)'); 
        pl.plot(r_values); pl.plot(range(len(r_values)), np.ones(r_values.shape)*evalparams['rval_thr'], 'r')
        pl.subplot(2,1,2); pl.title('SNR_comp'); 
        pl.plot(SNR_comp); pl.plot(range(len(SNR_comp)), np.ones(r_values.shape)*evalparams['min_SNR'], 'r')
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
        plot_contours(nmf['A'].all()[:, idx_components], nmf['Av'], thr=0.85); pl.axis('off')
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
        
    return idx_components, idx_components_bad, SNR_comp, r_values, cnn_preds

#%%
def get_source_paths(RID, check_motion=True):
    
    roi_source_dir = RID['DST']
    roi_type = RID['roi_type']
    roi_id = RID['roi_id']
    excluded_tiffs = []
    
    if roi_type == 'caiman2D':
        
        # Get ROI source paths:
        src_nmf_dir = os.path.join(roi_source_dir, 'nmfoutput')
        roi_source_paths = sorted([os.path.join(src_nmf_dir, n) for n in os.listdir(src_nmf_dir) if n.endswith('npz')], key=natural_keys) # Load nmf files
        
        # Get TIFF/mmap source from which ROIs were extracted:
        src_mmap_dir = RID['PARAMS']['mmap_source']
        tiff_source_paths = sorted([os.path.join(src_mmap_dir, f) for f in os.listdir(src_mmap_dir) if f.endswith('mmap')], key=natural_keys)
        
        # Get filenames for matches between roi source and tiff source:
        assert len(roi_source_paths) == len(tiff_source_paths), "Mismatch in N tiffs (%i) and N roi sources (%i)." % (len(roi_source_paths), len(tiff_source_paths))
        filenames = []
        for roi_src in roi_source_paths:
            # Get filename base
            # filenames = sorted([str(re.search('File(\d{3})', nmffile).group(0)) for nmffile in roi_source_paths], key=natural_keys)
            filenames.append(str(re.search('File(\d{3})', roi_src).group(0)))

    if check_motion is True:
        filenames, excluded_tiffs = check_mc_evlatuion(RID, filenames, mcmetric_type='zproj_corrcoefs')
        if len(excluded_tiffs) > 0:
            bad_roi_fns = []
            bad_tiff_fns = []
            for badfn in excluded_tiffs:
                bad_roi_fns.append([r for r in roi_source_paths if badfn in r][0])
                bad_tiff_fns.append([r for r in tiff_source_paths if badfn in r][0])
            roi_source_paths = [r for r in roi_source_paths if r not in bad_roi_fns]
            tiff_source_paths = [r for r in tiff_source_paths if r not in bad_tiff_fns]
    
    return roi_source_paths, tiff_source_paths, filenames, excluded_tiffs

#%%
def evaluate_roi_set(RID, evalparams=None):
    
    session_dir = RID['DST'].split('/ROIs')[0]
    roi_id = RID['roi_id']
    
    session = os.path.split(session_dir)[1]

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
    
    
    roi_source_paths, tiff_source_paths, filenames, excluded_tiffs = get_source_paths(RID)
    tstamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    
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
def run_evaluation(options):
    # PATH opts:
    parser = optparse.OptionParser()

    parser.add_option('-R', '--root', action='store', dest='rootdir', default='/nas/volume1/2photon/data', help='data root dir (root project dir containing all animalids) [default: /nas/volume1/2photon/data, /n/coxfs01/2pdata if --slurm]')
    parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')
    parser.add_option('-S', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID')
    
    parser.add_option('--default', action='store_true', dest='default', default='store_false', help="Use all DEFAULT params, for params not specified by user (no interactive)")
    parser.add_option('--slurm', action='store_true', dest='slurm', default=False, help="set if running as SLURM job on Odyssey")
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
    
    #%% OPTPARSE:
#    rootdir = '/nas/volume1/2photon/data'
#    animalid = 'JR063' #'JR063'
#    session = '20171128_JR063' #'20171128_JR063'
#    roi_id = 'rois001'
#    slurm = False
#    auto = False
#    
#    # Eval params (specific for NMF)
#    min_SNR = 1.5
#    rval_thr = 0.6
#    use_cnn = False
#    cnn_thr = 0.8
#    decay_time = 1.0
#    frame_rate = 44.6828
#    Nsamples = 5
#    Npeaks = 5
    
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

    #% Define params dict:
    evalparams = dict()
    if RID['roi_type'] == 'caiman2D':
        evalparams['min_SNR'] = min_SNR
        evalparams['rval_thr'] = rval_thr
        evalparams['decay_time'] = decay_time
        evalparams['frame_rate'] = frame_rate
        evalparams['Nsamples'] = Nsamples
        evalparams['Npeaks'] = Npeaks
        evalparams['use_cnn'] = use_cnn
        evalparams['cnn_thr'] = cnn_thr
        evalparams['gSig'] = RID['PARAMS']['options']['extraction']['gSig']
            
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
    eval_filename = str(os.path.splitext(os.path.basename(eval_filepath))[0]) 
    if eval_filename not in evaldict.keys():
        evaldict[eval_filename] = evalinfo
    with open(eval_info_path, 'w') as fw:
        json.dump(evaldict, fw, sort_keys=True, indent=4)
    print "Updated eval info dict with entry: %s\nSaved update to %s." % (eval_filename, eval_info_path)    
    
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
