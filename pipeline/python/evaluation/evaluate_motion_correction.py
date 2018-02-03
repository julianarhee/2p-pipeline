#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 11:54:38 2018
@author: julianarhee
"""
import os
import sys
import h5py
import json
import time
import datetime
import optparse
import math
import re
import multiprocessing as mp
import matplotlib
matplotlib.use('Agg')
import pylab as pl
import tifffile as tf
import numpy as np
import seaborn as sns
import pandas as pd

from pipeline.python.utils import natural_keys, get_source_info, print_elapsed_time

#%%      
    
def zproj_corr_file(filename, ref_image, info, ref_channel_dir, asdict=False):

    file_corrvals = []
    slice_files = sorted([t for t in os.listdir(os.path.join(ref_channel_dir, filename)) if t.endswith('tif')], key=natural_keys)
    for slice_idx, slice_file in enumerate(sorted(slice_files, key=natural_keys)):
        slice_img = tf.imread(os.path.join(ref_channel_dir, filename, slice_file))
        corr = np.corrcoef(ref_image[:,:,slice_idx].flat, slice_img.flat)
        file_corrvals.append(corr[0,1])
    file_corrvals = np.array(file_corrvals)
    
    if asdict is True:
        zproj = dict()
        zproj['source_images'] = os.path.join(ref_channel_dir, filename)
        zproj['slice_corrcoefs'] = file_corrvals
        zproj['nslices'] = len(slice_files)
        return zproj
    else:
        source_img_path = os.path.join(ref_channel_dir, filename)
        slice_corrcoefs = file_corrvals
        nslices = len(slice_files)
        return source_img_path, slice_corrcoefs, nslices
    
    
def mp_zproj_corr(filenames, ref_image, info, ref_channel_dir, nprocs=12):
    
    t_eval_mp = time.time()
    
    def worker(filenames, ref_image, info, ref_channel_dir, out_q):
        """
        Worker function is invoked in a process. 'filenames' is a list of 
        filenames to evaluate [File001, File002, etc.]. The results are placed
        in a dict that is pushed to a queue.
        """
        outdict = {}
        for fn in filenames:
            outdict[fn] = zproj_corr_file(fn, ref_image, info, ref_channel_dir, asdict=True)
        out_q.put(outdict)
    
    # Each process gets "chunksize' filenames and a queue to put his out-dict into:
    out_q = mp.Queue()
    chunksize = int(math.ceil(len(filenames) / float(nprocs)))
    procs = []
    
    for i in range(nprocs):
        p = mp.Process(target=worker,
                       args=(filenames[chunksize * i:chunksize * (i + 1)],
                                       ref_image,
                                       info,
                                       ref_channel_dir,
                                       out_q))
        procs.append(p)
        p.start()
    
    # Collect all results into single results dict. We should know how many dicts to expect:
    resultdict = {}
    for i in range(nprocs):
        resultdict.update(out_q.get())
    
    # Wait for all worker processes to finish
    for p in procs:
        print "Finished:", p
        p.join()
    
    print_elapsed_time(t_eval_mp)
    
    return resultdict

#%%
def get_zproj_correlations(info, zproj='mean', multiproc=True, nprocs=12):
    
    zproj_results = dict()
    
    process_dir = info['process_dir']
    ref_channel = info['ref_channel']
    ref_filename = info['ref_filename']
    d1 = info['d1']; d2 = info['d2']; d3 = info['d3']

    # Get zproj slices:
    mean_slice_dirs = [m for m in os.listdir(process_dir) if zproj in m and 'mcorrected' in m and os.path.isdir(os.path.join(process_dir, m))]
    try:
        assert len(mean_slice_dirs)==1, "No zproj dirs for type %s found." % zproj
        mean_slices_dir = os.path.join(process_dir, mean_slice_dirs[0])
        assert ref_channel in os.listdir(mean_slices_dir), "DIR NOT FOUND: %s, in parent %s." % (ref_channel, mean_slices_dir)
        ref_channel_dir = os.path.join(mean_slices_dir, ref_channel)
        filenames = sorted([i for i in os.listdir(ref_channel_dir) if 'File' in i], key=natural_keys)
        print "STEP 1: Checking zproj across time."
        print "Found %i files to evaluate." % len(filenames)
    except Exception as e:
        print "Failed to find correct dirs for zproj evaluation."
        print e
        print "Aborting."

    # Get corrs to reference zproj img:
    try:
        print "Loading REFERENCE file image..."
        ref_slices = sorted([t for t in os.listdir(os.path.join(ref_channel_dir, ref_filename)) if t.endswith('tif')], key=natural_keys)
        assert len(ref_slices) == d3, "Expected %i slices, only found %i in zproj dir." % (d3, len(ref_slices))
        
        ref_image = np.empty((d1, d2, d3))
        for sidx, sfn in enumerate(sorted(ref_slices, key=natural_keys)):
            ref_img_tmp = tf.imread(os.path.join(ref_channel_dir, ref_filename, sfn))
            ref_image[:, :, sidx] = ref_img_tmp #np.ravel(ref_img_tmp, order='C')
        
        files_to_corr = sorted([f for f in filenames if not f == ref_filename], key=natural_keys)
        
        if multiproc is True:
            zproj_results['files'] = mp_zproj_corr(files_to_corr, ref_image, info, ref_channel_dir, nprocs=nprocs)
        else:
            zproj_results['files'] = dict((fname, dict()) for fname in sorted(filenames, key=natural_keys) if not fname==ref_filename)
            for filename in files_to_corr:
                print filename
                if filename == ref_filename:
                    continue
                zproj_results['files'][filename] = zproj_corr_file(filename, ref_image, info, ref_channel_dir, asdict=True)
    
        assert len(zproj_results['files'].keys()) == len(filenames)-1, "Incomplete zproj for each files."

    except Exception as e:
        print e
        print "Error calculating corrcoefs for each slice to reference."
        print "Aborting"
    
    return zproj_results

def plot_zproj_results(zproj_results, info, zproj='mean'):
    ref_filename = info['ref_filename']
    mc_evaldir = info['output_dir']
    
    # Plot zproj corr results:
    try:
        # PLOT:  for each file, plot each slice's correlation to reference slice
        df = pd.DataFrame(dict((k, zproj_results['files'][k]['slice_corrcoefs']) for k in zproj_results['files'].keys()))
        sns.set(style="whitegrid", color_codes=True)
        pl.figure(figsize=(12, 4)); sns.stripplot(data=df, jitter=True, split=True)
        pl.title('%s across time, corrcoef by slice (reference: %s)' % (zproj, ref_filename))
        figname = 'corrcoef_%s_across_time.png' % zproj
        pl.savefig(os.path.join(mc_evaldir, figname))
        pl.close()
    except Exception as e:
        print e
        print "Unable to plot per-slice zproj correlations for each file."

def evaluate_zproj(zproj_results, info, nstds=2, zproj='mean'):
    ref_filename = info['ref_filename']
    d3 = info['d3']
    mc_evaldir = info['output_dir']
    
    print "Identifying bad files using zproj method, nstds = %i" % nstds
    
    df = pd.DataFrame(dict((k, zproj_results['files'][k]['slice_corrcoefs']) for k in zproj_results['files'].keys()))
    
    # Identify "bad" tiffs:    
    try:
        # Use some metric to determine while TIFFs might be "bad":
        bad_files = [(df.columns[i], sl) for sl in range(df.values.shape[0]) for i,d in enumerate(df.values[sl,:]) if abs(d-np.mean(df.values[sl,:])) >= np.std(df.values[sl,:])*nstds]
        zproj_results['metric'] = '%i std' % nstds
        zproj_results['bad_files'] = bad_files
        
        # Collapse across slices to get a single correlation val for each tiff:
        means_by_file = np.mean(df.values, axis=0)
        pl.figure(figsize=(12,4))
        pl.plot(means_by_file)
        pl.xticks(range(len(means_by_file)), list(df.columns))
        pl.title('mean corr across %s-projected slices (reference: %s)' % (zproj, ref_filename))        
        if len(bad_files) > 0:
            for f in range(len(bad_files)):
                print "%s: %s slice img is %i stds >= mean correlation" % (bad_files[f][0], zproj, nstds)
                dfidx = list(df.columns).index(bad_files[f][0])
                pl.plot(dfidx, df[bad_files[f][0]][bad_files[f][1]], 'r*', label='%i stds' % nstds)
        pl.legend()
        figname = 'corrcoef_%s_across_time_volume.png' % zproj
        pl.savefig(os.path.join(mc_evaldir, figname))
        pl.close()
        zproj_results['mean_corrcoefs'] = means_by_file
        
        # Re-plot corr to ref by slice, but connect slices:
        pl.figure(figsize=(12,4))
        for sl in range(d3):
            currslice = [zproj_results['files'][k]['slice_corrcoefs'][sl] for k in sorted(zproj_results['files'].keys(), key=natural_keys)]
            pl.plot(currslice, label='slice%02d' % int(sl+1))
        pl.xticks(range(len(currslice)), [k for k in sorted(zproj_results['files'].keys(), key=natural_keys)])
        pl.legend()
        figname = 'corrcoef_%s_across_time2.png' % zproj
        pl.savefig(os.path.join(mc_evaldir, figname))
        pl.close()
    except Exception as e:
        print e
        print "Unable to calculate bad files..."
        
    
    return zproj_results
    
#%%
def frame_corr_file(tiffpath, info, nstds=4, ref_frame=0, asdict=True):
                    
    T = info['T']
    d1 = info['d1']; d2 = info['d2']; d3 = info['d3']
    mc_evaldir = info['output_dir']
 
    mov = tf.imread(tiffpath)
    curr_filename = str(re.search('File(\d{3})', tiffpath).group(0))
    
    
    print "Loaded %s. Mov size:" % curr_filename, mov.shape      # T*d3, d1, d2 (d1 = lines/fr, d2 = pix/lin)
    mov = np.squeeze(np.reshape(mov, [T, d3, d1, d2], order='C'))  # Reshape so that slices in vol are grouped together
    movR = np.reshape(mov, [T, d1*d2*d3], order='C')
    df = pd.DataFrame(movR)
    fr_idxs = [fr for fr in np.arange(0, T) if not fr==ref_frame]
    corrcoefs = np.array([df.T[ref_frame].corr(df.T[fr]) for fr in fr_idxs])
    bad_frames, metric = evaluate_frame_corrs(corrcoefs, currfile=curr_filename, nstds=nstds, ref_frame=ref_frame, mc_evaldir=mc_evaldir)
 
    if asdict is True:
        framecorr = dict()
        framecorr['frame_corrcoefs'] = corrcoefs
        framecorr['file_source'] = tiffpath
        framecorr['dims'] = mov.shape
        framecorr['metric'] = metric
        framecorr['bad_frames'] = bad_frames
        return framecorr
    else:
        return corrcoefs, tiffpath, mov.shape, metric, bad_frames

####
def mp_frame_corr(filepaths, info, nstds=4, ref_frame=0, nprocs=12):
    """filepaths is a dict: key=File001, val=path/to/tiff
    """
    t_eval_mp = time.time()
    
    filenames = sorted(filepaths.keys(), key=natural_keys)
    
    def worker(filenames, filepaths, info, nstds, ref_frame, out_q):
        """
        Worker function is invoked in a process. 'filenames' is a list of 
        filenames to evaluate [File001, File002, etc.]. The results are placed
        in a dict that is pushed to a queue.
        """
        outdict = {}
        for fn in filenames:
            outdict[fn] = frame_corr_file(filepaths[fn], info, nstds=nstds, ref_frame=ref_frame, asdict=True)
        out_q.put(outdict)
    
    # Each process gets "chunksize' filenames and a queue to put his out-dict into:
    out_q = mp.Queue()
    chunksize = int(math.ceil(len(filenames) / float(nprocs)))
    procs = []
    
    for i in range(nprocs):
        p = mp.Process(target=worker,
                       args=(filenames[chunksize * i:chunksize * (i + 1)],
                                       filepaths,
                                       info,
                                       nstds, 
                                       ref_frame,
                                       out_q))
        procs.append(p)
        p.start()
    
    # Collect all results into single results dict. We should know how many dicts to expect:
    resultdict = {}
    for i in range(nprocs):
        resultdict.update(out_q.get())
    
    # Wait for all worker processes to finish
    for p in procs:
        print "Finished:", p
        p.join()
    
    print_elapsed_time(t_eval_mp)
    
    return resultdict
####

def evaluate_frame_corrs(corrcoefs, currfile="placeholder", nstds=4, ref_frame=0, mc_evaldir='/tmp'):
    
    T = len(corrcoefs)
    fr_idxs = [fr for fr in np.arange(0, T) if not fr==ref_frame]
    print "fr_idxs: %i, corrcoefs: %i" % (len(fr_idxs), len(corrcoefs))
    bad_frames = [idx for idx,val in zip(fr_idxs, corrcoefs) if abs(val - np.mean(corrcoefs)) > (np.std(corrcoefs)*nstds)]
    metric = '%i stds' % nstds

    pl.figure()
    pl.plot(corrcoefs)
    if len(bad_frames) > 0:
        print "%s: Found %i frames >= %s from mean correlation val." % (currfile, len(bad_frames), metric)
        pl.plot(bad_frames, [corrcoefs[b] for b in bad_frames], 'r*')
    pl.title("%s: correlation to first frame" % currfile)
    figname = 'corr_to_frame1_%s.png' % currfile
    pl.savefig(os.path.join(mc_evaldir, figname))
    pl.close()
    
    return bad_frames, metric
    
def get_frame_correlations(info, ref_frame=0, nstds=4, multiproc=True, nprocs=12):

    
    mc_sourcedir = info['source_dir']
    mc_evaldir = info['output_dir']
    nexpected_tiffs = info['ntiffs']

    tiff_fns = sorted([t for t in os.listdir(mc_sourcedir) if 'File' in t and t.endswith('tif')], key=natural_keys)
    assert len(tiff_fns) == info['ntiffs'], "Expected %i tiffs, found %i in dir\n%s" % (nexpected_tiffs, len(tiff_fns), mc_sourcedir)
    tiff_paths = sorted([os.path.join(mc_sourcedir, f) for f in tiff_fns], key=natural_keys) # if filename in f for filename in filenames]
    filenames = sorted([str(re.search('File(\d{3})', m).group(0)) for m in tiff_paths], key=natural_keys)
    #%
    
    
    t = time.time()
    if multiproc is True:
        tiffpath_dict = dict((k, v) for k, v in zip(filenames, tiff_paths))
        framecorr_results = mp_frame_corr(tiffpath_dict, info, nstds=nstds, ref_frame=ref_frame, nprocs=nprocs)
    else:
        framecorr_results = dict()
        for fidx,fn in enumerate(sorted(tiff_paths, key=natural_keys)):
            curr_filename = str(re.search('File(\d{3})', fn.group(0)))
            framecorr = frame_corr_file(fn, info, nstds=nstds, ref_frame=ref_frame, asdict=True)
#            bad_frames, metric = evaluate_frame_corrs(framecorr['corrcoefs'], currfile=curr_filename, nstds=nstds, ref_frame=ref_frame, mc_evaldir=mc_evaldir)
#            framecorr['metric'] = metric
#            framecorr['bad_frames'] = bad_frames
#            
            framecorr_results[curr_filename] = framecorr
        
    elapsed = time.time() - t
    print "Time elapsed:", elapsed
    
    return framecorr_results

#%%
def parse_options(options):
    
    parser = optparse.OptionParser()

    # PATH opts:
    parser.add_option('-D', '--root', action='store', dest='rootdir', default='/nas/volume1/2photon/data', help='data root dir (root project dir containing all animalids) [default: /nas/volume1/2photon/data, /n/coxfs01/2pdata if --slurm]')
    parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')

    # Set specific session/run for current animal:
    parser.add_option('-S', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID')
    parser.add_option('-A', '--acq', action='store', dest='acquisition', default='FOV1', help="acquisition folder (ex: 'FOV1_zoom3x') [default: FOV1]")
    parser.add_option('-R', '--run', action='store', dest='run', default='', help="name of run dir containing tiffs to be processed (ex: gratings_phasemod_run1)")
    parser.add_option('--slurm', action='store_true', dest='slurm', default=False, help="set if running as SLURM job on Odyssey")
    parser.add_option('-P', '--process-id', action='store', dest='process_id', default='', help="Process ID (for ex: processed001, or processed005, etc.")
    parser.add_option('-Z', '--zproj', action='store', dest='zproj', default='mean', help="Z-projection across time for comparing slice images for each file to reference [default: mean]")
    parser.add_option('-f', '--ref-frame', action='store', dest='ref_frame', default=0, help="Frame within tiff to use as reference for frame-to-frame correlations [default: 0]")
    parser.add_option('--stds-zp', action='store', dest='nstds_zproj', default=2.0, help="Num stds over which zproj-corr to ref is considered a failure [default: 2]")
    parser.add_option('--stds-fr', action='store', dest='nstds_frames', default=4.0, help="Num stds over frame-to-frame corr is considered a failure [default: 4]")

    parser.add_option('--par', action='store_true', dest='multiproc', default=False, help="set if parallel proc eval on files")
    parser.add_option('-n', '--nprocs', action='store', dest='nprocs', default=12, help="n cores for parallel processing [default: 12]")
    
    (options, args) = parser.parse_args(options)
    
    if options.slurm is True:
        if 'coxfs' not in options.rootdir:
            options.rootdir = '/n/coxfs01/2p-data'
            
    return options

    
def evaluate_motion(options):
    #%%
    options = parse_options(options)
    
#    rootdir = '/nas/volume1/2photon/data'
#    animalid = 'JR063'
#    session = '20171128_JR063'
#    acquisition = 'FOV2_zoom1x'
#    run = 'gratings_static'
#    process_id = 'processed001'
#    zproj = 'mean'
#    ref_frame = 0
    
    #%%
    rootdir = options.rootdir
    animalid = options.animalid
    session = options.session
    acquisition = options.acquisition
    run = options.run
    #slurm = options.slurm           
    process_id = options.process_id
    zproj = options.zproj
    ref_frame = options.ref_frame
    nstds_zproj = options.nstds_zproj
    nstds_frames = options.nstds_frames
    
    multiproc = options.multiproc
    nprocs = options.nprocs
    
    #%% Get info about MC to evaluate:
    acquisition_dir = os.path.join(rootdir, animalid, session, acquisition)
    info = get_source_info(acquisition_dir, run, process_id)

    #%% Create HDF5 file to save evaluation data:
    eval_outfile = os.path.join(info['output_dir'], 'mc_metrics.hdf5')
    metrics = h5py.File(eval_outfile, 'w')
    metrics.attrs['source'] = info['source_dir']
    metrics.attrs['ref_file'] = info['ref_filename']
    metrics.attrs['ref_channel'] = info['ref_channel']
    metrics.attrs['dims'] = [info['d1'], info['d2'], info['d3'], info['T']]
    metrics.attrs['creation_time'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    #%%
    # -------------------------------------------------------------------------
    # 1. Check zproj across time for each slice. Plot corrcoefs to reference.
    # -------------------------------------------------------------------------
    zproj_results = get_zproj_correlations(info, zproj=zproj, multiproc=multiproc, nprocs=nprocs)
    plot_zproj_results(zproj_results, info, zproj=zproj)
    zproj_results = evaluate_zproj(zproj_results, info, nstds=nstds_zproj, zproj=zproj)

    
    #% Save results:
    if 'zproj_corrcoefs' not in metrics.keys():
        zproj_corr_grp = metrics.create_group('zproj_corrcoefs')
        zproj_corr_grp.attrs['nslices'] = list(set([zproj_results['files'][k]['nslices'] for k in zproj_results['files'].keys()]))
        zproj_corr_grp.attrs['nfiles'] = len(zproj_results['files'].keys())
        zproj_corr_grp.attrs['zproj'] = zproj
        zproj_corr_grp.attrs['metric'] = zproj_results['metric']
        zproj_corr_grp.attrs['bad_files'] = [str(bfile[0]) for bfile in zproj_results['bad_files']]
        #print [str(bfile[0]) for bfile in zproj_results['bad_files']]
    else:
        zproj_corr_grp = metrics['zproj_corrcoefs']
    
    for fn in zproj_results['files'].keys():
        if fn not in zproj_corr_grp.keys():
            file_grp = zproj_corr_grp.create_group(fn)
            file_grp.attrs['source_images'] = zproj_results['files'][fn]['source_images']
        else:
            file_grp = zproj_corr_grp[fn]
        corrvals_for_file = zproj_results['files'][fn]['slice_corrcoefs']
        slice_corr_vals = file_grp.create_dataset('corrcoefs_by_slice', corrvals_for_file.shape,  corrvals_for_file.dtype)
        slice_corr_vals[...] = corrvals_for_file
        slice_corr_vals.attrs['nslices'] =  zproj_results['files'][fn]['nslices']

    means_by_file = zproj_results['mean_corrcoefs']
    mean_corr_vals = zproj_corr_grp.create_dataset('mean_corrcoefs', means_by_file.shape, means_by_file.dtype)
    mean_corr_vals[...] = means_by_file
    
    #%%
    # -------------------------------------------------------------------------
    # 2. Within each movie, check frame-to-frame corr. Plot corrvals across time.
    # -------------------------------------------------------------------------
    framecorr_results = get_frame_correlations(info, nstds=nstds_frames, ref_frame=ref_frame, multiproc=multiproc, nprocs=nprocs)

    #%%
    if 'within_file' not in metrics.keys():
        frame_corr_grp = metrics.create_group('within_file')
        frame_corr_grp.attrs['nslices'] = info['d3']
        frame_corr_grp.attrs['nframes'] = info['T']
        frame_corr_grp.attrs['ref_frame'] = ref_frame
    else:
        frame_corr_grp = metrics['within_file']
    
    for fn in framecorr_results.keys():
        if fn not in frame_corr_grp.keys():
            curr_corrcoefs = framecorr_results[fn]['frame_corrcoefs']
            frame_corr_file = frame_corr_grp.create_dataset(fn, curr_corrcoefs.shape, curr_corrcoefs.dtype)
            frame_corr_file[...] = curr_corrcoefs
            frame_corr_file.attrs['file_source'] = framecorr_results[fn]['file_source']
            frame_corr_file.attrs['dims'] = framecorr_results[fn]['dims']
            frame_corr_file.attrs['metric'] = framecorr_results[fn]['metric']
            frame_corr_file.attrs['bad_frames'] = framecorr_results[fn]['bad_frames']
        else:
            frame_corr_file = frame_corr_grp[fn]
    
    #%%
    # -------------------------------------------------------------------------
    # 3. Identify border pixels:
    # -------------------------------------------------------------------------
    # The max N pixels on border should be set as input param to set_roi_params.py
    # Currently, just using the same border pix size for all 4 borders
    
    
    #%%
    metrics.close()
    
    return eval_outfile

#%%
def main(options):

    eval_outfile = evaluate_motion(options)

    print "----------------------------------------------------------------"
    print "Finished evulation motion-correction."
    print "Saved output to:"
    print eval_outfile
    print "----------------------------------------------------------------"


if __name__ == '__main__':
    main(sys.argv[1:])
