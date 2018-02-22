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

from pipeline.python.set_roi_params import create_rid, update_roi_records, post_rid_cleanup # get_tiff_paths, set_params, initialize_rid, update_roi_records
from pipeline.python.utils import write_dict_to_json, jsonify_array, natural_keys, print_elapsed_time
from pipeline.python.rois.utils import check_mc_evaluation
pp = pprint.PrettyPrinter(indent=4)

import traceback
import re
import math

#%%
#if self.stride is None:
#    self.stride = np.int(self.rf * 2 * .1)
#    print(
#        ('**** Setting the stride to 10% of 2*rf automatically:' + str(self.stride)))

#%%
def timer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    formatted_time = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)
    return formatted_time

#%%
def extract_options(options):

    # PATH opts:
    choices_sourcetype = ('raw', 'mcorrected', 'bidi')
    default_sourcetype = 'mcorrected'
    choices_cluster_backend = ('local', 'multiprocessing', 'SLURM', 'ipyparallel')
    default_cluster_backend = 'local'

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
    parser.add_option('-n', '--nproc', action="store",
                  dest="nproc", default=12, help="N cores [default: 12]")
    parser.add_option('--par', action="store_true",
                  dest='multiproc', default=False, help="Use mp parallel processing to extract from tiffs at once, only if not slurm")

    parser.add_option('-C', '--cluster', type='choice', choices=choices_cluster_backend, action='store',
                      dest='cluster_backend', default=default_cluster_backend,
                      help="Which cluster backend to use. Valid choices: %s [default: %s]" % (choices_cluster_backend, default_cluster_backend))

    (options, args) = parser.parse_args(options)

    return options

#%%
def extract_cnmf_rois(options):
    t_serial = time.time()

    options = extract_options(options)

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
    nproc = int(options.nproc)
    multiproc = options.multiproc

    cluster_backend = options.cluster_backend

    if slurm is True:
        if 'coxfs01' not in rootdir:
            rootdir = '/n/coxfs01/2p-data'
        #cluster_backend = 'SLURM'

    if len(exclude_str) > 0:
        excluded_fids = exclude_str.split(',')
        excluded_tiffs = ['File%03d' % int(f) for f in excluded_fids]
    else:
        excluded_tiffs = []

    print "Excluding files:", excluded_tiffs

    # Set dirs:
    session_dir = os.path.join(rootdir, animalid, session)
    roi_basedir = os.path.join(session_dir, 'ROIs')
    if not os.path.exists(roi_basedir):
        os.makedirs(roi_basedir)

    # GET RID and RID path:
    if len(rid_hash) == 0:
        print "Creating new RID..."
        rid_options = ['-R', rootdir, '-i', animalid, '-S', session, '-A', acquisition, '-r', run,
                           '-s', tiffsource, '-t', sourcetype, '-o', 'caiman2D']
        if slurm is True:
            rid_options.extend(['--slurm'])
        RID = create_rid(rid_options)
        rid_hash = RID['rid_hash']
        tmp_rid_path = os.path.join(roi_basedir, 'tmp_rids', 'tmp_rid_%s.json' % rid_hash)
    else:
        print "RID %s -- Loading params..." % rid_hash
        tmp_rid_path = os.path.join(roi_basedir, 'tmp_rids', 'tmp_rid_%s.json' % rid_hash)
        RID = load_RID(tmp_rid_path, infostr='initialization')

    # Memmap tiffs, get mmapped tiff paths:
    print "Getting mmapped files."
    mmap_paths = par_mmap_tiffs(tmp_rid_path)
    print "DONE MEMMAPPING! There are %i mmap files for current run." % len(mmap_paths)

    files_to_run = sorted([str(re.search('File(\d{3})', m).group(0)) for m in mmap_paths], key=natural_keys)

    # Check for any tiffs to exclude:
    manual_excluded = RID['PARAMS']['eval']['manual_excluded']
    if RID['PARAMS']['eval']['check_motion'] is True:
        print "Requesting NMF extraction for %i TIFFs. Checking MC evaluation..." % len(files_to_run)
        files_to_run, mc_excluded_tiffs, mcmetrics_filepath = check_mc_evaluation(RID, files_to_run, mcmetric_type=RID['PARAMS']['eval']['mcmetric'],
                                                                                      rootdir=rootdir, animalid=animalid, session=session)
        excluded_tiffs = list(set(manual_excluded + mc_excluded_tiffs + excluded_tiffs))
    files_to_run = sorted([f for f in files_to_run if f not in excluded_tiffs])


    if multiproc is True:
        nmf_output_dict = mp_extract_nmf(files_to_run, tmp_rid_path, nproc=nproc, cluster_backend=cluster_backend, rootdir=rootdir)
        for f in nmf_output_dict.keys():
            print f, nmf_output_dict[f]['ngood_rois']
    else:
        for fidx, filename in enumerate(files_to_run):
            filenum = int(fidx + 1)
            print "Extracting from FILE %i..." % filenum
            nmfopts_hash, ngood_rois = extract_nmf_from_rid(tmp_rid_path, filenum, nproc=nproc, cluster_backend=cluster_backend, rootdir=rootdir)
            print "Finished FILE %i. Found %i components that pass initial evaluation." % (filenum, ngood_rois)

    if multiproc is True:
        print "DONE PROCESSING ALL FILES (par)!"
    else:
        print "DONE PROCESSING ALL FILES (serial)!"
    print "Total duration..."
    print_elapsed_time(t_serial)

    return nmfopts_hash, rid_hash

#%%
class nmfworker(mp.Process):
    def __init__(self, in_q, out_q, cluster_backend, nproc, rootdir):
        super(nmfworker, self).__init__()
        self.in_q = in_q
        self.out_q = out_q
        self.cluster_backend = cluster_backend
        self.nproc = nproc
        self.rootdir = rootdir

    def run(self):
        proc_name = self.name
        print "Starting NMF worker: %s" % proc_name

        print 'Computing things!'
        print "Worker resources: %s, nprocs %i" % (self.cluster_backend, self.nproc)
        outdict= {}
        while True:
            task = self.in_q.get()
            if task is None:
                # Poison pill to shutdown:
                print "%s: Exiting. Task done." % proc_name
                self.in_q.task_done()
                break
            print '%s: extracting %s.' % (proc_name, task[0])
            rid_path = task[1]
            fn = task[0]
            outdict[fn] = extract_nmf_from_rid(rid_path, int(fn[4:]), cluster_backend=self.cluster_backend, nproc=self.nproc, rootdir=self.rootdir, asdict=True)
            print "Worker: Extracted %s." % fn
            self.in_q.task_done()
            self.out_q.put(outdict)
            self.out_q.put(None)
        print "Worker: Finished %s." % fn
#        return
#        for fnkey in iter( self.in_q.get, None ):
#            # Use data
#            rid_path = fnkey[1]
#            fn = fnkey[0]
#            outdict[fn] = extract_nmf_from_rid(rid_path, int(fn[4:]), nproc=4, asdict=True)
#
#        self.out_q.put(outdict)

#%%
def mp_extract_nmf(files_to_run, tmp_rid_path, nproc=12, cluster_backend='local', rootdir=''): #, cluster_backend='local'):
    t_nmf = time.time()

    request_queue = mp.JoinableQueue()
    out_q = mp.Queue()

    # Start workers:
    arglist = [(fn, tmp_rid_path) for fn in files_to_run]
    nworkers = len(arglist)
    print "Creating %i workers..." % nworkers
    workers = [ nmfworker(request_queue, out_q, cluster_backend, nproc, rootdir) for i in xrange(nworkers) ]
    for w in workers:
        w.start()

    # Queue jobs
    nworkers = len(arglist)
    for fnkey in arglist:
        request_queue.put(fnkey)

    # Poison pill to allow clean shutdown (1 per worker):
    for i in xrange(nworkers):
        request_queue.put(None)

    # Wait for tasks to finish:
    print "Waiting for submitted tasks to complete..."
    request_queue.join()

    # Collate worker results
    print "Collating worker results..."
    resultdict = {}
    n_proc_end = 0
    while not n_proc_end == nworkers:
        res = out_q.get()
        if res is None:
            n_proc_end += 1
        else:
            resultdict.update(res)

    # Wait for all worker processes to finish
    for w in workers:
        w.join()
        print "Finished:", w
    print "Done joining."

#
    print_elapsed_time(t_nmf)

    return resultdict


#    for i in range(4):
#        nmfworker( request_queue, out_q ).start()
#    arglist = [(fn, tmp_rid_path) for fn in files_to_run]
#    for fnkey in arglist:
#        request_queue.put( fnkey )
#    # Sentinel objects to allow clean shutdown: 1 per worker.
#    for i in range(4):
#        request_queue.put( None )

#def mp_extract_nmf(files_to_run, tmp_rid_path, nproc=12, cluster_backend='local'):
#
#    t_eval_mp = time.time()
#
#    def worker(files_to_run, tmp_rid_path, cluster_backend, out_q):
#        """
#        Worker function is invoked in a process. 'filenames' is a list of
#        filenames to evaluate [File001, File002, etc.]. The results are placed
#        in a dict that is pushed to a queue.
#        """
#        outdict = {}
#        for fn in files_to_run:
#            outdict[fn] = extract_nmf_from_rid(tmp_rid_path, int(fn[4:]), nproc=len(files_to_run), cluster_backend=cluster_backend, asdict=True)
#        out_q.put(outdict)
#
#    # Each process gets "chunksize' filenames and a queue to put his out-dict into:
#    out_q = mp.Queue()
#    chunksize = int(math.ceil(len(files_to_run) / float(nproc)))
#    procs = []
#
#    for i in range(nproc):
#        p = mp.Process(target=worker,
#                       args=(files_to_run[chunksize * i:chunksize * (i + 1)],
#                                       tmp_rid_path,
#                                       cluster_backend,
#                                       out_q))
#        procs.append(p)
#        p.start()
#
#    # Collect all results into single results dict. We should know how many dicts to expect:
#    resultdict = {}
#    for i in range(nproc):
#        resultdict.update(out_q.get())
#
#    # Wait for all worker processes to finish
#    for p in procs:
#        print "Finished:", p
#        p.join()
#
#    print_elapsed_time(t_eval_mp)
#
#    return resultdict

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

    if not os.path.exists(outpath):
        os.makedirs(outpath)

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
def load_RID(tmp_rid_path, infostr='placeholder'):

    RID = None
    try:
        with open(tmp_rid_path, 'r') as f:
            RID = json.load(f)
    except Exception as e:
        print "Invalid RID path: %s" % tmp_rid_path
        traceback.print_exc()
        print "Aborting %s step." % infostr
        print "---------------------------------------------------------------"

    return RID

#%%
def par_mmap_tiffs(tmp_rid_path):

    mmap_paths = []

    # Load RID:
    RID = load_RID(tmp_rid_path, infostr='memmap')

    rid_hash = RID['rid_hash']
    roi_basedir = os.path.split(RID['DST'])[0]

    tmp_rid_path = os.path.join(roi_basedir, 'tmp_rids', 'tmp_rid_%s.json' % rid_hash)

    session_dir = os.path.split(roi_basedir)[0]


    if RID is not None:
        params = RID['PARAMS']['options']
        is_3D = params['info']['is_3D']
        border_pix = params['info']['max_shifts']

        # Get TIFF paths, and create memmapped files, if needed:
        # =========================================================================
        tiffpaths = sorted([os.path.join(RID['SRC'], t) for t in os.listdir(RID['SRC']) if t.endswith('tif')], key=natural_keys)
        print "RID %s -- Checking mmap files for each tif (%i tifs total)." % (rid_hash, len(tiffpaths))
        mmap_dir = RID['PARAMS']['mmap_source']
        expected_filenames, mmap_paths = check_memmapped_tiffs(tiffpaths, mmap_dir, is_3D, border_pix=border_pix)

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
        print "Output saved to: %s" % mmap_dir
        print "******************************"

#        for m in mmap_paths:
#            print m

    return mmap_paths

#%%
def create_cnm_object(params, patch=True, A=None, C=None, f=None, dview=None, n_processes=None):
    if patch is True:
        rf=params['patch']['rf']
        stride=params['patch']['stride']
        method_init=params['patch']['init_method']
        only_init_patch=params['patch']['only_init_patch']
    else:
        rf=params['full']['rf']
        stride=params['full']['stride']
        only_init_patch = False
    
    k=params['extraction']['K']
    gSig=params['extraction']['gSig']
    gSiz = (int((3 * gSig[0]) + 1), int((3 * gSig[0]) + 1))
    p=params['extraction']['p']
    merge_thresh=params['extraction']['merge_thresh']
    memory_fact=1
    gnb=params['extraction']['gnb']
    low_rank_background=params['extraction']['low_rank_background']
    method_deconvolution=params['extraction']['method_deconv']
    border_pix=params['info']['max_shifts']                #deconv_flag = True)
    update_bg = True 
    method_deconvolution = params['extraction']['method_deconv']
    
    cnm = cnmf.CNMF(
    	n_processes=n_processes, 
    	k=k,                                        # neurons per patch
    	gSig=gSig,                                  # half size of neuron
    	gSiz=gSiz,                                  # in general 3*gSig+1
    	Ain=A,
    	Cin=C,
    	f_in=f, 
    	merge_thresh=merge_thresh,                  # threshold for merging
    	p=p,                                        # order of autoregressive process to fit
    	dview=dview,                                # if None it will run on a single thread
    	tsub=2,                                     # downsampling factor in time for initialization, increase if you have memory problems             
    	ssub=2,                                     # downsampling factor in space for initialization, increase if you have memory problems
    	rf=rf,                                      # half size of the patch (final patch will be 100x100)
    	stride=stride,                              # overlap among patches (keep it at least large as 4 times the neuron size)
    	only_init_patch=only_init_patch,            # just leave it as is
    	gnb=gnb,                                    # number of background components
    	#nb_patch=gnb,                               # number of background components per patch
    	method_deconvolution=method_deconvolution,  # could use 'cvxpy' alternatively
    	low_rank_background=low_rank_background) #,    #leave as is
    	#update_background_components=update_bg,     # sometimes setting to False improve the results
    	#del_duplicates=True) #,                        # whether to remove duplicates from initialization
    	#deconv_flag=True
        #)

#    if patch is True:
#        cnm = cnmf.CNMF(k=params['extraction']['K'],
#                        gSig=params['extraction']['gSig'],
#                        p=params['extraction']['p'],
#                        merge_thresh=params['extraction']['merge_thresh'],
#                        dview=dview, n_processes=n_processes, memory_fact=1,
#                        rf=params['patch']['rf'],
#                        stride=params['patch']['stride'],
#                        method_init=params['patch']['init_method'],
#                        only_init_patch=params['patch']['only_init_patch'],
#                        gnb=params['extraction']['gnb'],
#                        low_rank_background=params['extraction']['low_rank_background'],
#                        method_deconvolution=params['extraction']['method_deconv'],
#                        border_pix=params['info']['max_shifts'])                #deconv_flag = True)
#    else:
#        cnm = cnmf.CNMF(k=A.shape,
#                        gSig=params['extraction']['gSig'],
#                        p=params['extraction']['p'],
#                        merge_thresh=params['extraction']['merge_thresh'],
#                        dview=dview, n_processes=n_processes, memory_fact=1,
#                        rf=params['full']['rf'],
#                        stride=params['full']['stride'],
#                        method_deconvolution=params['extraction']['method_deconv'],
#                        Ain=A,
#                        Cin=C,
#                        f_in=f,
#                        border_pix=params['info']['max_shifts'])
#
    # adjust opts:
    cnm.options['temporal_params']['memory_efficient'] = True
    cnm.options['temporal_params']['method'] = params['extraction']['method_deconv']
    cnm.options['temporal_params']['verbosity'] = False #True

    return cnm

#%%
def update_RID_nmf(nmfopts, RID):

    nmf_outdir = os.path.join(RID['DST'], 'nmfoutput')
    roi_basedir = os.path.split(RID['DST'])[0]

    # Save CNMF options with hash:
    nmfoptions = jsonify_array(nmfopts)
    nmfopts_hashid = hashlib.sha1(json.dumps(nmfoptions, sort_keys=True)).hexdigest()[0:6]
    write_dict_to_json(nmfoptions, os.path.join(nmf_outdir, 'nmfoptions_%s.json' % nmfopts_hashid))

    # Update tmp RID dict:
    RID['PARAMS']['nmf_hashid'] = nmfopts_hashid
    tmp_rid_path = os.path.join(roi_basedir, 'tmp_rids', 'tmp_rid_%s.json' % RID['rid_hash'])
    write_dict_to_json(RID, tmp_rid_path)

    # Update main RID dict:
    session = os.path.split(os.path.split(roi_basedir)[0])[-1]
    roidict_path = os.path.join(roi_basedir, 'rids_%s.json' % session)
    with open(roidict_path, 'r') as f:
        roidict = json.load(f)
    roidict[RID['roi_id']] = RID
    write_dict_to_json(roidict, roidict_path)

    return RID

#%%
def evaluate_cnm(images, cnm, params, dims, iteration=1, img=None, curr_filename='default', nmf_figdir='/tmp', dview=None):

    if not os.path.exists(nmf_figdir):
        os.makedirs(nmf_figdir)

    final_frate = params['eval']['final_frate']
    rval_thr = params['eval']['rval_thr']   # accept components with space corr threshold or higher
    decay_time = params['eval']['decay_time']                        # length of typical transient (sec)
    use_cnn = params['eval']['use_cnn']                         # CNN classifier designed for 2d data ?
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
    pl.savefig(os.path.join(nmf_figdir, 'iter%i_eval_metrics_%s.png' % (iteration, curr_filename)))
    pl.close()
    # ---------------------------------------------------------------------


    # PLOT: Iteration 1 - Show components that pass/fail evaluation metric --------------
    if img is None:
        m_images = cm.movie(images)
        img = np.mean(m_images, axis=0)

    pl.figure();
    pl.subplot(1,2,1); pl.title('pass'); plot_contours(cnm.A.tocsc()[:, idx_components], img, thr=params['display']['thr_plot']); pl.axis('off')
    pl.subplot(1,2,2); pl.title('fail'); plot_contours(cnm.A.tocsc()[:, idx_components_bad], img, thr=params['display']['thr_plot']); pl.axis('off')
    pl.savefig(os.path.join(nmf_figdir, 'iter%i_eval_contours_%s.png' % (iteration, curr_filename)))
    pl.close()
    # ---------------------------------------------------------------------

    return idx_components, idx_components_bad, SNR_comp, r_values

#%%
def run_nmf_on_file(tiffpath, tmp_rid_path, nproc=12, cluster_backend='local'):

    curr_filename = str(re.search('File(\d{3})', tiffpath).group(0))
    RID = load_RID(tmp_rid_path, infostr='nmf')
    rid_hash = RID['rid_hash']

    #% Set output paths:
    curr_roi_dir = RID['DST']
    nmf_outdir = os.path.join(curr_roi_dir, 'nmfoutput')
    nmf_figdir = os.path.join(nmf_outdir, 'figures')
    nmf_movdir = os.path.join(nmf_outdir, 'movies')
    if not os.path.exists(nmf_figdir):
        os.makedirs(nmf_figdir)
    if not os.path.exists(nmf_movdir):
        os.makedirs(nmf_movdir)

    print "RID %s -- writing cNMF output files to %s." % (rid_hash, nmf_outdir)

    #% Set NMF options from ROI params:
    params = RID['PARAMS']['options']
    border_to_0 = params['info']['max_shifts']
    is_3D = params['info']['is_3D']
    display_average = params['display']['use_average']
    inspect_components = False
    save_movies = params['display']['save_movies']
    remove_bad = False

    #% Get corresponding MMAP file:
    mmap_dir = RID['PARAMS']['mmap_source']
    mmap_path = None
    if not os.path.isdir(mmap_dir) or len(os.listdir(mmap_dir)) == 0:
        print "Attempting to create memmap file for tiff."
        print "Writing to dir: %s" % mmap_dir
        mmap_path = memmap_tiff(tiffpath, mmap_dir, is_3D, border_to_0, basename='Yr')

    if mmap_path is None:
        mmap_fn_matches = [m for m in os.listdir(mmap_dir) if m.endswith('mmap') and curr_filename in m]
        print mmap_fn_matches
        assert len(mmap_fn_matches) == 1, "Unable to find .MMAP file match for %s." % curr_filename
        mmap_path = os.path.join(mmap_dir, mmap_fn_matches[0])

#    try:
#        assert os.path.isdir(mmap_dir), "Specified MMAP dir does not exist,\n%s" % mmap_dir
#        mmap_fn_matches = [m for m in os.listdir(mmap_dir) if m.endswith('mmap') and curr_filename in m]
#        assert len(mmap_fn_matches) == 1, "Unable to find .MMAP file match for %s." % curr_filename
#        mmap_path = os.path.join(mmap_dir, mmap_fn_matches[0])
#    except Exception as e:
#        traceback.print_exc()
#        try:
#            print "Attempting to create memmap file for tiff."
#            print "Writing to dir: %s" % mmap_dir
#            mmap_path = memmap_tiff(tiffpath, mmap_dir, is_3D, border_to_0, basename='Yr')
#        except Exception as e:
#            print "Unable to create memmap file for tiff:\n%s" % tiffpath
#            traceback.print_exc()
#            print "Problem getting MMAP'ed file."
#            print "Aborting. Check MMAP files."
#            print "---------------------------------------------------------------"

    if mmap_path is not None:
        t_start = time.time()

        print "Loading MMAP file for %s." % curr_filename
        Yr, dims, T = cm.load_memmap(mmap_path)
        if is_3D:
            d1, d2, d3 = dims
        else:
            d1, d2 = dims
        images = np.reshape(Yr.T, [T] + list(dims), order='F')

        print "RID %s -- %s: ITER 1 -- RUNNING CNMF FIT..." % (rid_hash, curr_filename)

        #% start a cluster for parallel processing
        try:
            dview.terminate() # stop it if it was running
        except:
            pass


        c, dview, n_processes = cm.cluster.setup_cluster(backend=cluster_backend, # use this one
                                                         n_processes=nproc,  # number of process to use, reduce if out of mem
                                                         single_thread = False)

        #% Create CNMF object:
        # =====================================================================
        cnm = create_cnm_object(params, patch=True)


        # Save CNMF options:
        # =====================================================================
        if not 'nmf_hashid' in RID['PARAMS'].keys(): #curr_filename == 'File001':
            print "RID %s -- Updating tmp RID file with cnmf options." % rid_hash
            RID = update_RID_nmf(cnm.options, RID)

        # Fit CNM:
        t = time.time()
        cnm = cnm.fit(images)
        print "%s [RID %s]:  ITERATION 1 -- Finished fit." % (curr_filename, rid_hash)
        print_elapsed_time(t)

        #% Look at local correlations:
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

        #%
        # ITERATION 1:  Evaluate components
        # =====================================================================
        pass_components = []
        if remove_bad is True:
            pass_components, fail_components, r_values, SNR_comp = evaluate_cnm(images, cnm, params, dims, iteration=1,
                                                                                img=Av,
                                                                                curr_filename=curr_filename,
                                                                                nmf_figdir=nmf_figdir,
                                                                                dview=dview)

            A_tot = cnm.A[:, pass_components]
            C_tot = cnm.C[pass_components]
        else:
            A_tot = cnm.A
            C_tot = cnm.C
        f_tot = cnm.f
        print(('Number of components:' + str(A_tot.shape[-1])))

        #% ITERATION 2:  re-run seeded cNMF:
        # =====================================================================
        cnm = create_cnm_object(params, patch=False, A=A_tot, C=C_tot, f=f_tot)

        t = time.time()
        cnm = cnm.fit(images)
        print "%s [RID %s]:  ITERATION 2 -- Finished FIT." % (curr_filename, rid_hash)
        print_elapsed_time(t)

        # PLOT: Iteration 2 -- view initial spatial footprints ----------------
        pl.figure()
        if display_average is True:
            crd = plot_contours(cnm.A, Av, thr=params['display']['thr_plot'])
        else:
            crd = plot_contours(cnm.A, Cn, thr=params['display']['thr_plot'])
        pl.savefig(os.path.join(nmf_figdir, 'iter2_contours_%s.png' % curr_filename))
        pl.close()
        # ---------------------------------------------------------------------

        # ITER 2:  Evaluate components and save output:
        # =====================================================================
        print "%s [RID %s]:  Running initial evaluation." % (curr_filename, rid_hash)
        t_eval = time.time()
        pass_components, fail_components, r_values, SNR_comp = evaluate_cnm(images, cnm, params, dims, iteration=2,
                                                                            img=Av,
                                                                            curr_filename=curr_filename,
                                                                            nmf_figdir=nmf_figdir,
                                                                            dview=dview)

        print(('Should keep ' + str(len(pass_components)) +
           ' and discard  ' + str(len(fail_components))))
        print_elapsed_time(t_eval)

        #% Extract df/f:
        print "%s [RID %s]:  Extracting DF/F." % (curr_filename, rid_hash)
        t_df = time.time()
        Cdf = extract_DF_F(Yr=Yr, A=cnm.A, C=cnm.C, bl=cnm.bl)
        print_elapsed_time(t_df)

        # Save NMF outupt:
        # =====================================================================
        print "Saving results..."
        #A, C, b, f, YrA, sn, S = cnm.A, cnm.C, cnm.b, cnm.f, cnm.YrA, cnm.sn, cnm.S

        #% save results:
        np.savez(os.path.join(nmf_outdir, os.path.split(mmap_path)[1][:-4] + 'results_analysis.npz'),
                 Yr_path=mmap_path, dims=dims, T=T,
                 A=cnm.A, C=cnm.C, b=cnm.b, f=cnm.f, YrA=cnm.YrA, sn=cnm.sn, S=cnm.S, Cdf=Cdf,
                 idx_components=pass_components, idx_components_bad=fail_components,
                 r_values=r_values, SNR_comp=SNR_comp, Av=Av, Cn=Cn,
                 bl=cnm.bl, g=cnm.g, c1=cnm.c1, neurons_sn=cnm.neurons_sn, lam=cnm.lam)


        print("FINAL N COMPONENTS:", cnm.A.shape[1])

#        #%% Plot SN:
#        pl.figure()
#        pl.subplot(1,3,1); pl.title('avg'); pl.imshow(Av, cmap='gray'); pl.axis('off')
#        pl.subplot(1,3,2); pl.title('cn'); pl.imshow(Cn, cmap='gray'); pl.axis('off')
#        ax = pl.subplot(1,3,3); pl.title('sn'); im = pl.imshow(np.reshape(sn, (d1,d2), order='F')); pl.axis('off')
#        divider = make_axes_locatable(ax)
#        cax = divider.append_axes("right", size="5%", pad=0.05)
#        pl.colorbar(im, cax=cax);
#        pl.savefig(os.path.join(nmf_figdir, 'zproj_final_%s.png' % curr_filename))
#        pl.close()

        #%
        # [INTERACTIVE] Iter 2 -- Page through components
        # =====================================================================
        if inspect_components is True:
            if display_average is True:
                view_patches_bar(Yr, scipy.sparse.coo_matrix(cnm.A.tocsc()[:, pass_components]), cnm.C[pass_components, :], cnm.b, cnm.f, dims[0], dims[1],
                             YrA=cnm.YrA[pass_components, :], img=Av)
            else:
                view_patches_bar(Yr, scipy.sparse.coo_matrix(cnm.A.tocsc()[:, pass_components]), cnm.C[pass_components, :], cnm.b, cnm.f, dims[0], dims[1],
                             YrA=cnm.YrA[pass_components, :], img=Cn)

            # %
            if display_average is True:
                view_patches_bar(Yr, scipy.sparse.coo_matrix(A.tocsc()[:, fail_components]), cnm.C[fail_components, :], cnm.b, cnm.f, dims[0],
                             dims[1], YrA=cnm.YrA[fail_components, :], img=Av)
            else:
                view_patches_bar(Yr, scipy.sparse.coo_matrix(cnm.A.tocsc()[:, fail_components]), cnm.C[fail_components, :], cnm.b, cnm.f, dims[0],
                             dims[1], YrA=cnm.YrA[fail_components, :], img=Cn)

        # %
        # Reconstruct denoised movie:
        # =====================================================================
        if save_movies is True:
            t_mov = time.time()
            #if curr_filename in movie_files:
            #% save denoised movie:
            currmovie = cm.movie(cnm.A.dot(cnm.C) + cnm.b.dot(cnm.f)).reshape(dims + (-1,), order='F').transpose([2, 0, 1])
            currmovie.save(os.path.join(nmf_movdir, 'denoisedmov_plusbackground_%s.tif' % curr_filename))

            #% background only
            currmovie = cm.movie(cnm.b.dot(cnm.f)).reshape(dims + (-1,), order='F').transpose([2, 0, 1])
            currmovie.save(os.path.join(nmf_movdir, 'backgroundmov_%s.tif' % curr_filename))

            #% reconstruct denoised movie without background
            currmovie = cm.movie(cnm.A.dot(cnm.C)).reshape(dims + (-1,), order='F').transpose([2, 0, 1])
            currmovie.save(os.path.join(nmf_movdir, 'denoisedmov_nobackground_%s.tif' % curr_filename))

            print "Saved movie for %s" % curr_filename
            print_elapsed_time(t_mov)

        #% show background(s)
        BB  = cm.movie(cnm.b.reshape(dims+(-1,), order = 'F').transpose(2,0,1))
        #BB.play(gain=2, offset=0, fr=2, magnification=4)
        pl.figure()
        BB.zproject()
        pl.savefig(os.path.join(nmf_figdir, 'background_zproj_%s.png' % curr_filename))
        pl.close()

        print "FINISHED!"
        print "TOTAL TIME:"
        print_elapsed_time(t_start)

    # SAVE HASH for current nmf options set:
    pp.pprint(RID)
    nmfopts_hash = RID['PARAMS']['nmf_hashid']

    #% STOP CLUSTER and clean up log files
    cm.stop_server(dview=dview)
    log_files = glob.glob('*_LOG_*')
    for log_file in log_files:
        os.remove(log_file)

    return nmfopts_hash, len(pass_components), rid_hash

#%%
def extract_nmf_from_rid(tmp_rid_path, file_num, nproc=12, cluster_backend='local', asdict=False, rootdir=''):
    nmfopts_hash = "None"
    ngood_rois = 0

    RID = load_RID(tmp_rid_path)

    tiffpaths = sorted([os.path.join(RID['SRC'], t) for t in os.listdir(RID['SRC']) if t.endswith('tif')], key=natural_keys)

    currfile = 'File%03d' % int(file_num)

    # Check for any tiffs to exclude:
    excluded_tiffs = RID['PARAMS']['eval']['manual_excluded']
    if RID['PARAMS']['eval']['check_motion'] is True:
        filenames = [os.path.splitext(os.path.split(tpath)[1])[0].split('_')[-1] for tpath in tiffpaths]
        print "Requesting NMF extraction for %i TIFFs. Checking MC evaluation..." % len(filenames)
        filenames, mc_excluded_tiffs, mcmetrics_filepath = check_mc_evaluation(RID, filenames, mcmetric_type=RID['PARAMS']['eval']['mcmetric'],
                                                                                   rootdir=rootdir)
        excluded_tiffs = list(set(excluded_tiffs + mc_excluded_tiffs))

    if currfile in excluded_tiffs:
        print "***Skipping EXCLUDED TIFF: %s" % currfile
        return nmfopts_hash, ngood_rois


#    print "Getting mmapped files."
#    mmap_paths = mmap_tiffs(tmp_rid_path)
#    print "DONE MEMMAPPING! There are %i mmap files for current run." % len(mmap_paths)

    try:
        tiffmatches = [t for t in tiffpaths if currfile in t]
        assert len(tiffmatches) == 1, "Unable to find correct tiff match for specified filenum: %s" % currfile
        tiffpath =  tiffmatches[0]

        print "EXTRACTING ROIS"
        nmfopts_hash, ngood_rois, rid_hash = run_nmf_on_file(tiffpath, tmp_rid_path, nproc=nproc, cluster_backend=cluster_backend)

        print "RID %s: Finished cNMF ROI extraction: nmf options were %s" % (rid_hash, nmfopts_hash)
        print "%s-- Initialial evalation found %i ROIs that pass." % (currfile, ngood_rois)

        if os.path.exists(tmp_rid_path):
            print "Cleaning up tmp files..."
            session_dir = os.path.split(os.path.split(tmp_rid_path)[0])[0]
            post_rid_cleanup(session_dir, rid_hash)

    except Exception as e:
        print "Failed while extracting cnmf ROIs for %s" % currfile
        traceback.print_exc()

    if asdict is True:
        print "Returning as dict."
        nmf_file_output = dict()
        nmf_file_output['nmfopts_hash'] = nmfopts_hash
        nmf_file_output['ngood_rois'] = ngood_rois
        return nmf_file_output
    else:
        print "nmf hash: %s, ngood rois: %i" % (nmfopts_hash, ngood_rois)
        return nmfopts_hash, ngood_rois


def main(options):

    nmf_hash, rid_hash = extract_cnmf_rois(options)
    print "RID %s: Finished cNMF ROI extraction: nmf options were %s" % (rid_hash, nmf_hash)

if __name__ == '__main__':
    main(sys.argv[1:])
