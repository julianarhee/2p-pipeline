#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 11:36:30 2017

@author: julianarhee
"""
import matplotlib
matplotlib.use('Agg')
import sys
import optparse
import os
import re
import json
import shutil
import scipy
import tifffile
import hashlib
import time
import pprint
import glob
import pylab as pl
import caiman as cm
import numpy as np
import multiprocessing as mp
from checksumdir import dirhash
from mpl_toolkits.axes_grid1 import make_axes_locatable

from caiman.utils.visualization import plot_contours, view_patches_bar
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.components_evaluation import evaluate_components, estimate_components_quality_auto
from caiman.source_extraction.cnmf.utilities import extract_DF_F

from pipeline.python.set_roi_params import create_rid, update_roi_records # get_tiff_paths, set_params, initialize_rid, update_roi_records
from pipeline.python.utils import write_dict_to_json, jsonify_array

pp = pprint.PrettyPrinter(indent=4)



#%%
#if self.stride is None:
#    self.stride = np.int(self.rf * 2 * .1)
#    print(
#        ('**** Setting the stride to 10% of 2*rf automatically:' + str(self.stride)))

#%%
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def timer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    formatted_time = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)
    return formatted_time
   
#%%

def save_memmap2(filenames, base_name='Yr', resize_fact=(1, 1, 1), remove_init=0, idx_xy=None,
                order='F', xy_shifts=None, is_3D=False, add_to_movie=0, border_to_0=0):
    """ Saves efficiently a list of tif files into a memory mappable file
    Parameters:
    ----------
        filenames: list
            list of tif files or list of numpy arrays
        base_name: str
            the base used to build the file name. IT MUST NOT CONTAIN "_"
        resize_fact: tuple
            x,y, and z downampling factors (0.5 means downsampled by a factor 2)
        remove_init: int
            number of frames to remove at the begining of each tif file
            (used for resonant scanning images if laser in rutned on trial by trial)
        idx_xy: tuple size 2 [or 3 for 3D data]
            for selecting slices of the original FOV, for instance
            idx_xy = (slice(150,350,None), slice(150,350,None))
        order: string
            whether to save the file in 'C' or 'F' order
        xy_shifts: list
            x and y shifts computed by a motion correction algorithm to be applied before memory mapping
        is_3D: boolean
            whether it is 3D data
    Returns:
    -------
        fname_new: the name of the mapped file, the format is such that
            the name will contain the frame dimensions and the number of f
    """

    # TODO: can be done online
    Ttot = 0
    for idx, f in enumerate(filenames):
        if isinstance(f, str):
            print(f)

        if is_3D:
            #import tifffile
            #            print("Using tifffile library instead of skimage because of  3D")
            Yr = f if isinstance(f, basestring) else tifffile.imread(f)
            if idx_xy is None:
                Yr = Yr[remove_init:]
            elif len(idx_xy) == 2:
                Yr = Yr[remove_init:, idx_xy[0], idx_xy[1]]
            else:
                Yr = Yr[remove_init:, idx_xy[0], idx_xy[1], idx_xy[2]]

        else:
            Yr = cm.load(f, fr=1, in_memory=True) if isinstance(f, basestring) else cm.movie(f)
            if xy_shifts is not None:
                Yr = Yr.apply_shifts(xy_shifts, interpolation='cubic', remove_blanks=False)

            if idx_xy is None:
                if remove_init > 0:
                    Yr = np.array(Yr)[remove_init:]
            elif len(idx_xy) == 2:
                Yr = np.array(Yr)[remove_init:, idx_xy[0], idx_xy[1]]
            else:
                raise Exception('You need to set is_3D=True for 3D data)')
                Yr = np.array(Yr)[remove_init:, idx_xy[0], idx_xy[1], idx_xy[2]]

        if border_to_0 > 0:

            min_mov = Yr.calc_min()
            Yr[:, :border_to_0, :] = min_mov
            Yr[:, :, :border_to_0] = min_mov
            Yr[:, :, -border_to_0:] = min_mov
            Yr[:, -border_to_0:, :] = min_mov

        fx, fy, fz = resize_fact
        if fx != 1 or fy != 1 or fz != 1:

            if 'movie' not in str(type(Yr)):
                Yr = cm.movie(Yr, fr=1)

            Yr = Yr.resize(fx=fx, fy=fy, fz=fz)
            
        T, dims = Yr.shape[0], Yr.shape[1:]
        Yr = np.transpose(Yr, list(range(1, len(dims) + 1)) + [0])
        Yr = np.reshape(Yr, (np.prod(dims), T), order='F')

        if idx == 0:
            fname_tot = base_name + '_d1_' + str(dims[0]) + '_d2_' + str(dims[1]) + '_d3_' + str(
                1 if len(dims) == 2 else dims[2]) + '_order_' + str(order)
            if isinstance(f, str):
                fname_tot = os.path.join(os.path.split(f)[0], os.path.splitext(os.path.split(f)[1])[0] + '_' + fname_tot)
            big_mov = np.memmap(fname_tot, mode='w+', dtype=np.float32,
                                shape=(np.prod(dims), T), order=order)
        else:
            big_mov = np.memmap(fname_tot, dtype=np.float32, mode='r+',
                                shape=(np.prod(dims), Ttot + T), order=order)

        big_mov[:, Ttot:Ttot + T] = np.asarray(Yr, dtype=np.float32) + 1e-10 + add_to_movie
        big_mov.flush()
        del big_mov
        Ttot = Ttot + T

    fname_new = fname_tot + '_frames_' + str(Ttot) + '_.mmap'
    print fname_new
    print fname_tot
    try:
        # need to explicitly remove destination on windows
        os.unlink(fname_new)
    except OSError:
        pass
    os.rename(fname_tot, fname_new)

    return fname_new

#%%
def memmap_tiff(filepath, outpath, is_3D, border_to_0, basename='Yr'):
    
    #border_to_0 = 0
        
    idx_xy = None
    
    # TODO: needinfo
    remove_init = 0           # if you need to remove frames from the beginning of each file
    downsample_factor = 1     # downsample movie in time: use .2 or .1 if file is large and you want a quick answer

    #base_name = acqmeta['base_filename'] #fname[0].split('/')[-1][:-4]  # omit this in save_memmap_each() to use original filenames as base for mmap files

    # estimate offset:
    m_orig = cm.load_movie_chain([filepath])
    add_to_movie = -np.nanmin(m_orig)
    print "min val:", add_to_movie
    del m_orig

    mmap_fpath = save_memmap2(
            [filepath], base_name=basename, is_3D=is_3D,
            resize_fact=(1, 1, downsample_factor), remove_init=remove_init, 
            idx_xy=idx_xy, add_to_movie=add_to_movie, border_to_0=border_to_0)

    if isinstance(mmap_fpath, list):
        mmap_fpath = mmap_fpath[0]

    mmap_fn = os.path.split(mmap_fpath)[1]
    print mmap_fn
        
    shutil.move(mmap_fpath, os.path.join(outpath, mmap_fn))
    
    mmap_outpath = os.path.join(outpath, mmap_fn)
    
    return mmap_outpath


#%% MEMORY-MAPPING (plus make non-negative)
def check_memmapped_tiffs(tiffpaths, mmap_dir, is_3D, border_pix=0):
    expected_filenames = [os.path.splitext(os.path.split(tpath)[1])[0].split('_')[-1] for tpath in tiffpaths]
    if os.path.isdir(mmap_dir):
        tmp_mmap_fns = sorted([m for m in os.listdir(mmap_dir) if m.endswith('mmap')], key=natural_keys)
    else:
        tmp_mmap_fns = []
        os.makedirs(mmap_dir)

    missing_mmap_fns = []
    for f in expected_filenames:
        matched_mmap = [m for m in tmp_mmap_fns if f in m]
        if len(matched_mmap) == 0:
            missing_mmap_fns.append(f)

    tiffs_to_mmap = [[t for t in tiffpaths if m in t][0] for m in missing_mmap_fns]
    print "Need to mmap %i tiffs." % len(tiffs_to_mmap)
    
    if len(tiffs_to_mmap) > 0:
        output = mp.Queue()
        processes = [mp.Process(target=memmap_tiff, args=(str(tpath), mmap_dir, is_3D, border_pix,)) for tpath in tiffs_to_mmap]
        for p in processes:
            p.start()
        for p in processes:
            p.join()
        print "Finished memmapping..."
        
    mmap_paths = sorted([os.path.join(mmap_dir, m) for m in os.listdir(mmap_dir) if m.endswith('mmap')], key=natural_keys)
    
    return expected_filenames, mmap_paths

#%%

def extract_cnmf_rois(options):

    # PATH opts:
    choices_sourcetype = ('raw', 'mcorrected', 'bidi')
    default_sourcetype = 'mcorrected'

    parser = optparse.OptionParser()

    # PATH opts:
    parser.add_option('-D', '--root', action='store', dest='rootdir', default='/nas/volume1/2photon/data', help='data root dir (root project dir containing all animalids) [default: /nas/volume1/2photon/data, /n/coxfs01/2pdata if --slurm]')
    parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')

    # Set specific session/run for current animal:
    parser.add_option('-S', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID')
    parser.add_option('-A', '--acq', action='store', dest='acquisition', default='FOV1', help="acquisition folder (ex: 'FOV1_zoom3x') [default: FOV1]")
    parser.add_option('-R', '--run', action='store', dest='run', default='', help="name of run dir containing tiffs to be processed (ex: gratings_phasemod_run1)")
    parser.add_option('--default', action='store_true', dest='default', default='store_false', help="Use all DEFAULT params, for params not specified by user (no interactive)")

    parser.add_option('-s', '--tiffsource', action='store', dest='tiffsource', default=None, help="name of folder containing tiffs to be processed (ex: processed001). should be child of <run>/processed/")
    parser.add_option('-t', '--source-type', type='choice', choices=choices_sourcetype, action='store', dest='sourcetype', default=default_sourcetype, help="Type of tiff source. Valid choices: %s [default: %s]" % (choices_sourcetype, default_sourcetype))

    parser.add_option('-p', '--rid', action='store', dest='rid_hash', default='', help="RID hash of current ROI extraction (6 char), default will create new if set_roi_params.py not run")

    parser.add_option('--slurm', action='store_true', dest='slurm', default=False, help="set if running as SLURM job on Odyssey")

    parser.add_option('-x', '--exclude', action="store",
                  dest="excluded_tiffs", default='', help="Tiff numbers to exclude (comma-separated)")

    (options, args) = parser.parse_args(options) 


    # dataset dependent parameters
    #rootdir = '/nas/volume1/2photon/data'
    #animalid = 'JR063'
    #session = '20171202_JR063'
    #acquisition = 'FOV1_zoom1x'
    #run = 'static_gratings'
    #slurm = False
    #
    #tiffsource = 'processed001'
    #sourcetype = 'mcorrected'
    #rid_hash = 'fa5f0f'
    
    # Set USER INPUT options:
    rootdir = options.rootdir
    animalid = options.animalid
    session = options.session
    acquisition = options.acquisition
    run = options.run

    tiffsource = options.tiffsource
    sourcetype = options.sourcetype
    rid_hash = options.rid_hash
    slurm = options.slurm
    
    exclude_str = options.excluded_tiffs
    
    #%%
    if len(exclude_str) > 0:
        excluded_fids = exclude_str.split(',')
        excluded_tiffs = ['File%03d' % int(f) for f in excluded_fids]
    else:
        excluded_tiffs = []
    
    print "Excluding files:", excluded_tiffs
 
    if slurm is True:
        if 'coxfs01' not in rootdir:
            rootdir = '/n/coxfs01/2p-data'

    #% CHECK TIFF paths and get corresponding MMAP paths:
    session_dir = os.path.join(rootdir, animalid, session)
    roi_dir = os.path.join(session_dir, 'ROIs')
    if not os.path.exists(roi_dir):
        os.makedirs(roi_dir)
        
        
    if len(rid_hash) == 0:
        print "Creating new RID..."
        rid_options = ['-R', rootdir, '-i', animalid, '-S', session, '-A', acquisition, '-r', run,
                           '-s', tiffsource, '-t', sourcetype, '-o', 'caiman2D']
        if slurm is True:
            rid_options.extend(['--slurm'])
        RID = create_rid(rootdir)
        rid_hash = RID['rid_hash']
        tmp_rid_path = os.path.join(roi_dir, 'tmp_rids', 'tmp_rid_%s.json' % rid_hash)
        
    else:
        print "RID %s -- Loading params..." % rid_hash
        tmp_rid_path = os.path.join(roi_dir, 'tmp_rids', 'tmp_rid_%s.json' % rid_hash)
        with open(tmp_rid_path, 'r') as f:
            RID = json.load(f)
        

    #%% Set output paths:
    curr_roi_dir = RID['DST']
    nmf_outdir = os.path.join(curr_roi_dir, 'nmfoutput')
    nmf_figdir = os.path.join(nmf_outdir, 'figures')
    nmf_movdir = os.path.join(nmf_outdir, 'movies')

    if not os.path.exists(nmf_figdir):
        os.makedirs(nmf_figdir)
    if not os.path.exists(nmf_movdir):
        os.makedirs(nmf_movdir)

    print "RID %s -- writing cNMF output files to %s." % (rid_hash, nmf_outdir)
        
    #%%
    params = RID['PARAMS']['options']

    nmovies = params['info']['nmovies']
    nchannels = params['info']['nchannels']
    signal_channel = params['info']['signal_channel']
    volumerate = params['info']['volumerate']
    is_3D = params['info']['is_3D']
    border_pix = params['info']['max_shifts']
    frate = params['eval']['final_frate']          # imaging rate in frames per second
    movie_files = params['display']['movie_files']

    display_average = params['display']['use_average']
    inspect_components = False
    save_movies = params['display']['save_movies']
    remove_bad = False

    # Get TIFF paths, and create memmapped files, if needed:
    # =========================================================================
    tiffpaths = sorted([os.path.join(RID['SRC'], t) for t in os.listdir(RID['SRC']) if t.endswith('tif')], key=natural_keys)
    print "RID %s -- Extracting cNMF ROIs from %i tiffs." % (rid_hash, len(tiffpaths))
    for t in tiffpaths:
        print t
    mmap_dir = RID['PARAMS']['mmap_source']
    expected_filenames, mmap_paths = check_memmapped_tiffs(tiffpaths, mmap_dir, is_3D, border_pix=border_pix)

    #%% 
    # Update mmap dir with hashed mmap files, update RID:
    # =========================================================================
    check_mmap_hash = False
    if '_mmap_' not in mmap_dir:
        check_mmap_hash = True
    else:
        mmap_hash = os.path.split(mmap_dir)[1].split('_')[-1]

    if check_mmap_hash is True:
        print "RID %s -- Checking mmap dir hash..." % rid_hash
        hash_ignore_files = [m for m in os.listdir(mmap_dir) if not m.endswith('mmap')]
        mmap_hash = dirhash(mmap_dir, 'sha1', excluded_files=hash_ignore_files)[0:6]

    if mmap_hash not in mmap_dir: 
        mmap_dir_hash =  mmap_dir + '_' + mmap_hash
        os.rename(mmap_dir, mmap_dir_hash)
        RID['PARAMS']['mmap_source'] = mmap_dir_hash
        mmap_dir = mmap_dir_hash
        write_dict_to_json(RID, tmp_rid_path)
        update_roi_records(RID, session_dir)
        print "RID %s -- ROI entry updated." % rid_hash
        mmap_paths = [os.path.join(mmap_dir, m) for m in os.listdir(mmap_dir) if m.endswith('mmap')]

    print "******************************"
    print "Done MEMMAPPING tiffs:"
    print "******************************"
    for m in mmap_paths:
        print m
     
    #%%
    durations = dict()
    
    #%%
    #fidx = 0
    #curr_filename = expected_filenames[fidx]
    for curr_filename in expected_filenames:
        #%
        
        if curr_filename in excluded_tiffs:
            continue
        
        #%%
        durations[curr_filename] = dict()
        t_start = time.time()
        
        print "Extracting ROIs:", curr_filename
        curr_mmap = [m for m in mmap_paths if curr_filename in m][0]
        Yr, dims, T = cm.load_memmap(curr_mmap)
        if is_3D:
            d1, d2, d3 = dims
        else:
            d1, d2 = dims
        images = np.reshape(Yr.T, [T] + list(dims), order='F')
        
        print "RID %s -- %s: ITER 1 -- RUNNING CNMF FIT..." % (rid_hash, curr_filename)
        
        #%% start a cluster for parallel processing
        try:
            dview.terminate() # stop it if it was running
        except:
            pass
        
        c, dview, n_processes = cm.cluster.setup_cluster(backend='local', # use this one
                                                         n_processes=None,  # number of process to use, reduce if out of mem
                                                         single_thread = False)

        #%% Create CNMF object:
        # =====================================================================
        cnm = cnmf.CNMF(k=params['extraction']['K'],
                        gSig=params['extraction']['gSig'],
                        p=params['extraction']['p'],
                        merge_thresh=params['extraction']['merge_thresh'],
                        dview=dview, n_processes=n_processes, memory_fact=1,
                        rf=params['patch']['rf'],
                        stride=params['patch']['stride'],
                        method_init=params['patch']['init_method'],
                        only_init_patch=params['patch']['only_init_patch'],
                        gnb=params['extraction']['gnb'],
                        low_rank_background=params['extraction']['low_rank_background'],
                        method_deconvolution=params['extraction']['method_deconv'],
                        border_pix=params['info']['max_shifts'])                #deconv_flag = True) 
        
        # adjust opts:
        cnm.options['temporal_params']['memory_efficient'] = True
        cnm.options['temporal_params']['method'] = params['extraction']['method_deconv']
        cnm.options['temporal_params']['verbosity'] = True
        #cnm.options['init_params']['rolling_sum'] = False #True
        #cnm.options['init_params']['normalize_init'] = False
        #cnm.options['init_params']['center_psf'] = True
        
        # Save CNMF options:
        # =========================================================================
        if not 'nmf_hashid' in RID['PARAMS'].keys(): #curr_filename == 'File001':
            print "RID %s -- Updating tmp RID file with cnmf options." % rid_hash
            
            # Save CNMF options with has:
            nmfoptions = jsonify_array(cnm.options)
            nmfoptions['excluded_tiffs'] = excluded_tiffs
            nmfopts_hashid = hashlib.sha1(json.dumps(nmfoptions, sort_keys=True)).hexdigest()[0:6]
            write_dict_to_json(nmfoptions, os.path.join(nmf_outdir, 'nmfoptions_%s.json' % nmfopts_hashid))
            
            # Update tmp RID dict:
            RID['PARAMS']['nmf_hashid'] = nmfopts_hashid
            write_dict_to_json(RID, tmp_rid_path)
            
            # Update main RID dict:
            roidict_path = os.path.join(session_dir, 'ROIs', 'rids_%s.json' % session)
            with open(roidict_path, 'r') as f:
                roidict = json.load(f)
            roidict[RID['roi_id']] = RID
            write_dict_to_json(roidict, roidict_path)
            

        #%% Extract ROIs with specified params:
        # =====================================================================
        t = time.time()
        cnm = cnm.fit(images)
        #elapsed = time.time() - t
        curr_dur = timer(t, time.time())
        print "ITERATION 1 -- Time elapsed:", curr_dur #elapsed
        durations[curr_filename]['iter1'] = curr_dur
        write_dict_to_json(durations, os.path.join(nmf_outdir, 'durations.json'))
        
        #%% Look at local correlations:
        # =====================================================================
        Y = np.reshape(Yr, dims + (T,), order='F')
        Cn = cm.local_correlations(Y)
        Cn[np.isnan(Cn)] = 0
        
        m_images = cm.movie(images)    
        Av = np.mean(m_images, axis=0)
        
        #% PLOT: Correlation img and Average img ------------------------------
        pl.figure()
        pl.subplot(1,2,1); pl.title('Average'); pl.imshow(Av, cmap='gray'); pl.axis('off')
        pl.subplot(1,2,2); pl.title('Corr'); pl.imshow(Cn.max(0) if len(Cn.shape) == 3 else Cn, cmap='gray',
                   vmin=np.percentile(Cn, 1), vmax=np.percentile(Cn, 99)); pl.axis('off')
        pl.suptitle(curr_filename)
        pl.savefig(os.path.join(nmf_figdir, 'zproj_%s.png' % curr_filename))
        pl.close()
        # ---------------------------------------------------------------------


        # PLOT: Iteration 1 -- view initial spatial footprints ----------------
        pl.figure()
        if display_average is True:
            crd = plot_contours(cnm.A, Av, thr=params['display']['thr_plot'])
        else:
            crd = plot_contours(cnm.A, Cn, thr=params['display']['thr_plot'])
        
        pl.savefig(os.path.join(nmf_figdir, 'iter1_contours_%s.png' % curr_filename))
        pl.close()
        # ---------------------------------------------------------------------
        
        #%%
        # ITERATION 1:  Evaluate components
        # =====================================================================
        
        final_frate = volumerate
        rval_thr = params['eval']['rval_thr']   # accept components with space corr threshold or higher
        decay_time = 1.0                        # length of typical transient (sec)
        use_cnn = False                         # CNN classifier designed for 2d data ?
        min_SNR = params['eval']['min_SNR']     # accept components with peak-SNR of this or higher
        
        idx_components, idx_components_bad, SNR_comp, r_values, cnn_preds = \
            estimate_components_quality_auto(images, cnm.A, cnm.C, cnm.b, cnm.f, cnm.YrA,
                                             final_frate, decay_time,
                                             params['extraction']['gSig'], dims, 
                                             dview=dview,
                                             min_SNR=min_SNR, 
                                             r_values_min=rval_thr,
                                             use_cnn=use_cnn)
        
        #% PLOT: Iteration 1 - Visualize Spatial and Temporal component evaluation ----------
        pl.figure(figsize=(5,15))
        pl.subplot(2,1,1); pl.title('r values (spatial)'); pl.plot(r_values); pl.plot(range(len(r_values)), np.ones(r_values.shape)*rval_thr, 'r')
        pl.subplot(2,1,2); pl.title('SNR_comp'); pl.plot(SNR_comp); pl.plot(range(len(SNR_comp)), np.ones(r_values.shape)*min_SNR, 'r')
        pl.xlabel('roi')
        pl.suptitle(curr_filename)
        pl.savefig(os.path.join(nmf_figdir, 'iter1_eval_metrics_%s.png' % curr_filename))
        pl.close()
        # ---------------------------------------------------------------------
        
        
        # PLOT: Iteration 1 - Show components that pass/fail evaluation metric --------------
        pl.figure();
        pl.subplot(1,2,1); pl.title('pass'); plot_contours(cnm.A.tocsc()[:, idx_components], Av, thr=params['display']['thr_plot']); pl.axis('off')
        pl.subplot(1,2,2); pl.title('fail'); plot_contours(cnm.A.tocsc()[:, idx_components_bad], Av, thr=params['display']['thr_plot']); pl.axis('off')
        pl.savefig(os.path.join(nmf_figdir, 'iter1_eval_contours_%s.png' % curr_filename))
        pl.close()
        # ---------------------------------------------------------------------
        
        #%%
        # [INTERACTIVE] Page through components:
        # =====================================================================
        if inspect_components is True:
            if display_average is True:
                view_patches_bar(Yr, scipy.sparse.coo_matrix(cnm.A.tocsc()[:, idx_components]), cnm.C[idx_components, :], cnm.b, cnm.f,
                                 dims[0], dims[1], YrA=cnm.YrA[idx_components, :], img=Av)
            else:
                view_patches_bar(Yr, scipy.sparse.coo_matrix(cnm.A.tocsc()[:, idx_components]), cnm.C[idx_components, :], cnm.b, cnm.f,
                                 dims[0], dims[1], YrA=cnm.YrA[idx_components, :], img=Cn)
        
        #%% Filter REALLY BAD components:
        # =====================================================================
        if remove_bad is True:
            A_tot = cnm.A[:, idx_components]
            C_tot = cnm.C[idx_components]
        else:
            A_tot = cnm.A
            C_tot = cnm.C
        
        #YrA_tot = cnm.YrA
        #b_tot = cnm.b
        f_tot = cnm.f
        #sn_tot = cnm.sn
        
        print(('Number of components:' + str(A_tot.shape[-1])))
        
        #%% ITERATION 2:  re-run seeded cNMF:
        # =====================================================================
        cnm = cnmf.CNMF(k=A_tot.shape,
                        gSig=params['extraction']['gSig'],
                        p=params['extraction']['p'],
                        merge_thresh=params['extraction']['merge_thresh'],
                        dview=dview, n_processes=n_processes, memory_fact=1,
                        rf=params['full']['rf'],
                        stride=params['full']['stride'],
                        method_deconvolution=params['extraction']['method_deconv'],
                        Ain=A_tot,
                        Cin=C_tot,
                        f_in=f_tot,
                        border_pix=params['info']['max_shifts'])
        # adjust opts:
        cnm.options['temporal_params']['memory_efficient'] = True
        cnm.options['temporal_params']['method'] = params['extraction']['method_deconv']
        cnm.options['temporal_params']['verbosity'] = True

        t = time.time()
        cnm = cnm.fit(images)
        #elapsed = time.time() - t
        curr_dur = timer(t, time.time())
        print "ITERATION 2 -- Time elapsed:", curr_dur #elapsed
        durations[curr_filename]['iter2'] = curr_dur #elapsed
        write_dict_to_json(durations, os.path.join(nmf_outdir, 'durations.json'))
        
        # PLOT: Iteration 2 -- view initial spatial footprints ----------------
        pl.figure()
        if display_average is True:
            crd = plot_contours(cnm.A, Av, thr=params['display']['thr_plot'])
        else:
            crd = plot_contours(cnm.A, Cn, thr=params['display']['thr_plot'])
        pl.savefig(os.path.join(nmf_figdir, 'iter2_contours_%s.png' % curr_filename))
        pl.close()
        # ---------------------------------------------------------------------

        #%%
        # ITER 2:  Evaluate components and save output:
        # =====================================================================
        final_frate = params['eval']['final_frate']
        rval_thr = params['eval']['rval_thr']       # accept components with space corr threshold or higher
        decay_time = params['eval']['decay_time']   # length of typical transient (sec)
        use_cnn = params['eval']['use_cnn']         # CNN classifier designed for 2d data ?
        min_SNR = params['eval']['min_SNR']         # accept components with peak-SNR of this or higher
        
        idx_components, idx_components_bad, SNR_comp, r_values, cnn_preds = \
            estimate_components_quality_auto(images, cnm.A, cnm.C, cnm.b, cnm.f, cnm.YrA,
                                             final_frate, decay_time,
                                             params['extraction']['gSig'], dims, 
                                             dview=dview,
                                             min_SNR=min_SNR, 
                                             r_values_min=rval_thr,
                                             use_cnn=use_cnn) 
        
        print(('Should keep ' + str(len(idx_components)) +
           ' and discard  ' + str(len(idx_components_bad))))


        #% PLOT: Iteration 2 - Visualize Spatial and Temporal component evaluation ----------
        pl.figure(figsize=(5,15))
        pl.subplot(2,1,1); pl.title('r values (spatial)'); pl.plot(r_values); pl.plot(range(len(r_values)), np.ones(r_values.shape)*rval_thr, 'r')
        pl.subplot(2,1,2); pl.title('SNR_comp'); pl.plot(SNR_comp); pl.plot(range(len(SNR_comp)), np.ones(r_values.shape)*min_SNR, 'r')
        pl.xlabel('roi')
        pl.suptitle(curr_filename)
        pl.savefig(os.path.join(nmf_figdir, 'iter2_eval_metrics_%s.png' % curr_filename))
        pl.close()
        # -----------------------------------------------------------------------------------
        
        
        # PLOT: Iteration 2 - Show components that pass/fail evaluation metric --------------
        pl.figure();
        pl.subplot(1,2,1); pl.title('pass'); plot_contours(cnm.A.tocsc()[:, idx_components], Av, thr=params['display']['thr_plot']); pl.axis('off')
        pl.subplot(1,2,2); pl.title('fail'); plot_contours(cnm.A.tocsc()[:, idx_components_bad], Av, thr=params['display']['thr_plot']); pl.axis('off')
        pl.savefig(os.path.join(nmf_figdir, 'iter2_eval_contours_%s.png' % curr_filename))
        pl.close()
        # -----------------------------------------------------------------------------------
        
        #%% 
        # Save NMF outupt:
        # =====================================================================
        A, C, b, f, YrA, sn, S = cnm.A, cnm.C, cnm.b, cnm.f, cnm.YrA, cnm.sn, cnm.S
        
#        print(S.max())

        #% Extract DF/F:
        Cdf = extract_DF_F(Yr=Yr, A=A, C=C, bl=cnm.bl)
        
        #% save results:
        np.savez(os.path.join(nmf_outdir, os.path.split(curr_mmap)[1][:-4] + 'results_analysis.npz'),
                 A=A, Cdf=Cdf, C=C, b=b, f=f, YrA=YrA, sn=sn, S=S, dims=cnm.dims, 
                 idx_components=idx_components, idx_components_bad=idx_components_bad,
                 r_values=r_values, SNR_comp=SNR_comp, Av=Av, Cn=Cn,
                 bl=cnm.bl, g=cnm.g, c1=cnm.c1, neurons_sn=cnm.neurons_sn, lam=cnm.lam)
                
    
        print("FINAL N COMPONENTS:", A.shape[1])
    
        #%% Plot SN:
        pl.figure()
        pl.subplot(1,3,1); pl.title('avg'); pl.imshow(Av, cmap='gray'); pl.axis('off')
        pl.subplot(1,3,2); pl.title('cn'); pl.imshow(Cn, cmap='gray'); pl.axis('off')
        ax = pl.subplot(1,3,3); pl.title('sn'); im = pl.imshow(np.reshape(sn, (d1,d2), order='F')); pl.axis('off')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        pl.colorbar(im, cax=cax); 
        pl.savefig(os.path.join(nmf_figdir, 'zproj_final_%s.png' % curr_filename))
        pl.close()
        
        #%% 
        # [INTERACTIVE] Iter 2 -- Page through components
        # =====================================================================
        if inspect_components is True:
            if display_average is True:
                view_patches_bar(Yr, scipy.sparse.coo_matrix(A.tocsc()[:, idx_components]), C[idx_components, :], b, f, dims[0], dims[1],
                             YrA=YrA[idx_components, :], img=Av)
            else:
                view_patches_bar(Yr, scipy.sparse.coo_matrix(A.tocsc()[:, idx_components]), C[idx_components, :], b, f, dims[0], dims[1],
                             YrA=YrA[idx_components, :], img=Cn)
                
            # %
            if display_average is True:
                view_patches_bar(Yr, scipy.sparse.coo_matrix(A.tocsc()[:, idx_components_bad]), C[idx_components_bad, :], b, f, dims[0],
                             dims[1], YrA=YrA[idx_components_bad, :], img=Av)
            else:
                view_patches_bar(Yr, scipy.sparse.coo_matrix(A.tocsc()[:, idx_components_bad]), C[idx_components_bad, :], b, f, dims[0],
                             dims[1], YrA=YrA[idx_components_bad, :], img=Cn)
            
        # %% 
        # Reconstruct denoised movie:
        # =====================================================================
        if save_movies is True:
            if curr_filename in movie_files:
                #%% save denoised movie:
                currmovie = cm.movie(A.dot(C) + b.dot(f)).reshape(dims + (-1,), order='F').transpose([2, 0, 1])                
                currmovie.save(os.path.join(nmf_movdir, 'denoisedmov_plusbackground_%s.tif' % curr_filename))
            
                #%% background only 
                currmovie = cm.movie(b.dot(f)).reshape(dims + (-1,), order='F').transpose([2, 0, 1])
                currmovie.save(os.path.join(nmf_movdir, 'backgroundmov_%s.tif' % curr_filename))
            
                # %% reconstruct denoised movie without background
                currmovie = cm.movie(A.dot(C)).reshape(dims + (-1,), order='F').transpose([2, 0, 1])                    
                currmovie.save(os.path.join(nmf_movdir, 'denoisedmov_nobackground_%s.tif' % curr_filename))

        #%% show background(s)
        BB  = cm.movie(b.reshape(dims+(-1,), order = 'F').transpose(2,0,1))
        #BB.play(gain=2, offset=0, fr=2, magnification=4)
        pl.figure()
        BB.zproject()
        pl.savefig(os.path.join(nmf_figdir, 'background_zproj_%s.png' % curr_filename))
        pl.close()
    
        #elapsed_full = time.time() - t_start
        full_dur = timer(t_start, time.time())
        durations[curr_filename]['full'] = full_dur #elapsed_full
        write_dict_to_json(durations, os.path.join(nmf_outdir, 'durations.json'))
        
            #%%
#    except Exception as e:
#        print "RID %s -- EXCEPTION during processing of %s" % (rid_hash, curr_filename)
#        print(e)
#    finally:
#        print "RID %s -- No Errors. Completed ROI extraction from %s" % (rid_hash, curr_filename)

    # SAVE HASH for current nmf options set:
    pp.pprint(RID)
    nmfopts_hash = RID['PARAMS']['nmf_hashid']
    
    print "-------------------------------------------------------------------"
    print "RID %s: COMPLETED!" % rid_hash
    print "-------------------------------------------------------------------"
    print "Each iteration lasted:"
    pp.pprint(durations)
    print "-------------------------------------------------------------------"
    
    #% STOP CLUSTER and clean up log files
    cm.stop_server(dview=dview)
    log_files = glob.glob('*_LOG_*')
    for log_file in log_files:
        os.remove(log_file)
    
    return nmfopts_hash, rid_hash

#%%
def main(options):

    #nmf_hash, rid_hash = extract_cnmf_rois(options)
    nmf_hash, rid_hash = extract_cnmf_rois(options)
    #rid_hash = RID['rid_hash']
    #nmf_hash = RID['PARAMS']['nmf_hashid']
    print "RID %s: Finished cNMF ROI extraction: nmf options were %s" % (rid_hash, nmf_hash)
    
#    options = extract_options(options)     
#    acquisition_dir = os.path.join(options.rootdir, options.animalid, options.session, options.acquisition)
#    run = options.run 
#    post_pid_cleanup(acquisition_dir, run, pid_hash)
#    print "FINISHED FLYBACK, PID: %s" % pid_hash
    
if __name__ == '__main__':
    main(sys.argv[1:]) 
