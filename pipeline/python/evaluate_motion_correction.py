
import os
import h5py
import json
import time
import datetime
import matplotlib
matplotlib.use('Agg')
import pylab as pl
import tifffile as tf
import numpy as np
import seaborn as sns
import pandas as pd

from pipeline.python.utils import natural_keys

#%%
def get_source_info(acquisition_dir, run)
    # Set basename for files created containing meta/reference info:
    # -------------------------------------------------------------
    run_info_basename = '%s' % run #functional_dir
    pid_info_basename = 'pids_%s' % run

    # Set paths:
    # -------------------------------------------------------------
    #acquisition_dir = os.path.join(rootdir, animalid, session, acquisition)
    pidinfo_path = os.path.join(acquisition_dir, run, 'processed', '%s.json' % pid_info_basename)
    runmeta_path = os.path.join(acquisition_dir, run, '%s.json' % run_info_basename)
    
    # Load run meta info:
    # -------------------------------------------------------------
    with open(runmeta_path, 'r') as r:
        runmeta = json.load(r)
    
    # Load PID:
    # -------------------------------------------------------------
    with open(pidinfo_path, 'r') as f:
        pdict = json.load(f)

    if len(process_id) == 0 and len(pdict.keys()) > 0:
        process_id = pdict.keys()[0]
    
    PID = pdict[process_id]

    mc_sourcedir = PID['PARAMS']['motion']['destdir']
    mc_evaldir = '%s_evaluation' % mc_sourcedir
    if not os.path.exists(mc_evaldir):
        os.makedirs(mc_evaldir)
        
    # Get correlation of MEAN image (mean slice across time) to reference:
    ref_filename = 'File%03d' % PID['PARAMS']['motion']['ref_file']
    ref_channel = 'Channel%02d' % PID['PARAMS']['motion']['ref_channel']

    # Create dict to pass around to methods
    info['process_dir'] = PID['DST']
    info['source_dir'] = mc_sourcedir
    info['output_dir'] = mc_evaldir
    info['ref_filename'] = ref_filename
    info['ref_channel'] = ref_channel
    info['d1'] = runmeta['lines_per_frame']
    info['d2'] = runmeta['pixels_per_line']
    info['d3'] = d3
    info['T'] = runmeta['nvolumes']
    info['ntiffs'] = runmeta['ntiffs']
    
    return info

#%%
def get_zproj_correlations(info, nstds=2, zproj='mean'):
    
    zproj_results = dict()
    
    process_dir = info['process_dir']
    ref_channel = info['ref_channel']
    ref_filename = info['ref_filename']
    d1 = info['d1']; d2 = info['d2']; d3 = info['d3']
    mc_evaldir = info['output_dir']

    mean_slice_dirs = [m for m in os.listdir(process_dir) if zproj in m and os.path.isdir(os.path.join(process_dir, m))]
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

    try:
        print "Loading REFERENCE file image..."
        ref_slices = sorted([t for t in os.listdir(os.path.join(ref_channel_dir, ref_filename)) if t.endswith('tif')], key=natural_keys)
        assert len(ref_slices) == d3, "Expected %i slices, only found %i in zproj dir." % (d3, len(ref_slices))
        
        ref_image = np.empty((d1, d2, d3))
        for sidx, sfn in enumerate(sorted(ref_slices, key=natural_keys)):
            ref_img_tmp = tf.imread(os.path.join(ref_channel_dir, ref_filename, sfn))
            ref_image[:, :, sidx] = ref_img_tmp #np.ravel(ref_img_tmp, order='C')
        
        zproj_results['files'] = dict((fname, dict()) for fname in sorted(filenames, key=natural_keys))
        for filename in filenames:
            if filename == ref_filename:
                continue
                
            file_corrvals = []
            slice_files = sorted([t for t in os.listdir(os.path.join(ref_channel_dir, filename)) if t.endswith('tif')], key=natural_keys)
            for slice_idx, slice_file in enumerate(sorted(slice_files, key=natural_keys)):
                slice_img = tf.imread(os.path.join(ref_channel_dir, filename, slice_file))
                corr = np.corrcoef(ref_image[:,:,slice_idx].flat, slice_img.flat)
                file_corrvals.append(corr[0,1])
            file_corrvals = np.array(file_corrvals)

            zproj_results['files'][filename]['source_images'] = os.path.join(ref_channel_dir, filename)
            zproj_results['files'][filename]['slice_corrcoefs'] = file_corrvals
            zproj_results['files'][filename]['nslices'] = len(slice_files)
            
        # PLOT:  for each file, plot each slice's correlation to reference slice
        df = pd.DataFrame(dict((k, zproj_results['files'][k]['slice_corrcoefs']) for k in zproj_results['files'].keys()))
        sns.set(style="whitegrid", color_codes=True)
        pl.figure(figsize=(12, 4)); sns.stripplot(data=df, jitter=True, split=True)
        pl.title('%s across time, corrcoef by slice (reference: %s)' % (zproj, ref_filename))
        figname = 'corrcoef_%s_across_time.png' % zproj
        pl.savefig(os.path.join(mc_evaldir, figname))
        pl.close()
        
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
        print "Error calculating corrcoefs for each slice to reference."
        print e
        print "Aborting"
    
    return zproj_results
    
#%%
def get_frame_correlations(info, nstds=4, ref_frame=0):

    framecorr_results = dict()
    
    mc_sourcedir = info['source_dir']
    mc_evaldir = info['output_dir']
    nexpected_tiffs = info['ntiffs']
    T = info['T']
    d1 = info['d1']; d2 = info['d2']; d3 = info['d3']

    tiff_fns = sorted([t for t in os.listdir(mc_sourcedir) if 'File' in t and t.endswith('tif')], key=natural_keys)
    assert len(tiff_fns) == runmeta['ntiffs'], "Expected %i tiffs, found %i in dir\n%s" % (nexpected_tiffs, len(tiff_fns), mc_sourcedir)
    tiff_paths = [os.path.join(mc_sourcedir, f) for f in tiff_fns] # if filename in f for filename in filenames]
    #%
    
    t = time.time()
    for fidx,fn in enumerate(sorted(tiff_paths, key=natural_keys)):
        mov = tf.imread(fn)
        curr_filename = filenames[fidx]
        framecorr_results[curr_filename] = dict()
        print "Loaded %s. Mov size:" % curr_filename, mov.shape      # T*d3, d1, d2 (d1 = lines/fr, d2 = pix/lin)
        mov = np.squeeze(np.reshape(mov, [T, d3, d1, d2], order='C'))  # Reshape so that slices in vol are grouped together
        movR = np.reshape(mov, [T, d1*d2*d3], order='C')
        df = pd.DataFrame(movR)
        fr_idxs = [fr for fr in np.arange(0, T) if not fr==ref_frame]
        corrcoefs = np.array([df.T[ref_frame].corr(df.T[fr]) for fr in fr_idxs])
        bad_frames = [idx for idx,val in zip(fr_idxs, corrcoefs) if abs(val - np.mean(corrcoefs)) > (np.std(corrcoefs)*nstds)]
        
        framecorr_results[curr_filename]['frame_corrcoefs'] = corrcoefs
        framecorr_results[curr_filename]['file_source'] = fn
        framecorr_results[curr_filename]['dims'] = mov.shape
        framecorr_results[curr_filename]['metric'] = '%i stds' % nstds
        framecorr_results[curr_filename]['bad_frames'] = bad_frames
                
        pl.figure()
        pl.plot(corrcoefs)
        if len(bad_frames) > 0:
            print "%s: Found %i frames >= %i stds from mean correlation val." % (filenames[fidx], len(bad_frames), nstds)
            pl.plot(bad_frames, [corrcoefs[b] for b in bad_frames], 'r*')
        pl.title("%s: correlation to first frame" % filenames[fidx])
        figname = 'corr_to_frame1_%s.png' % filenames[fidx]
        pl.savefig(os.path.join(mc_evaldir, figname))
        pl.close()
        
    elapsed = time.time() - t
    print "Time elapsed:", elapsed
    
    return framecorr_results


#%%
def main(options):
    #%%
    rootdir = '/nas/volume1/2photon/data'
    animalid = 'JR063'
    session = '20171128_JR063'
    acquisition = 'FOV2_zoom1x'
    
    run = 'gratings_static'
    
    process_id = 'processed001'
    zproj = 'mean'
    #%%
    rootdir = options.rootdir
    animalid = options.animalid
    session = options.session
    acquisition = options.session
    run = options.run
    slurm = options.slurm
    if slurm is True:
        if 'coxfs' not in rootdir:
            rootdir = '/n/coxfs01/...'
            
    process_id = options.process_id
    zproj = options.zproj
    
    ref_frame = 0
    
    #%% Get info about MC to evaluate:
    acquisition_dir = os.path.join(rootdir, animalid, session, acquisition)
    info = get_source_info(acquisition_dir, run)

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
    
    zproj_results = get_zproj_correlations(info, nstds=nstds, zproj=zproj)
    
    if 'zproj_slice' not in metrics.keys():
        slice_corr_grp = metrics.create_group('zproj_slice')
        slice_corr_grp.attrs['nslices'] = list(set([zproj_results['files'][k]['nslices'] for k in zproj_results['files'].keys()]))
        slice_corr_grp.attrs['nfiles'] = len(zproj_results['files'].keys())
        slice_corr_grp.attrs['zproj'] = zproj
    else:
        slice_corr_grp = metrics['slice_correlations']
    
    for fn in zproj_results['files'].keys():
        if filename not in slice_corr_grp.keys():
            file_grp = slice_corr_grp.create_group(fn)
            file_grp.attrs['source_images'] = zproj_results['files'][fn]['source_images']
        else:
            file_grp = slice_corr_grp[fn]
        corrvals_for_file = zproj_results['files'][fn]['slice_corrcoefs']
        slice_corr_vals = file_grp.create_dataset('corrcoefs_by_slice', corrvals_for_file.shape,  corrvals_for_file.dtype)
        slice_corr_vals[...] = corrvals_for_file
        slice_corr_vals.attrs['nslices'] =  zproj_results['files'][fn]['nslices']
        slice_corr_vals.attrs['metric'] = zproj_results['metric']
        slice_corr_vals.attrs['bad_files'] = zproj_results['bad_files']
    
    means_by_file = zproj_results['mean_corrcoefs']
    mean_corr_vals = file_grp.create_dataset('mean_corrcoefs', means_by_file.shape, means_by_file.dtype)
    mean_corr_vals[...] = means_by_file
    
    #%%
    # -------------------------------------------------------------------------
    # 2. Within each movie, check frame-to-frame corr. Plot corrvals across time.
    # -------------------------------------------------------------------------
    framecorr_results = get_frame_correlations(info, nstds=4, ref_frame=ref_frame)

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
            frame_corr_file = frame_corr_grp.create_dataset(fn, corrcoefs.shape, corrcoefs.dtype)
            frame_corr_file[...] = corrcoefs
            frame_corr_file.attrs['file_source'] = framecorr_results[fn]['file_source']
            frame_corr_file.attrs['dims'] = framecorr_results[fn]['dims']
            frame_corr_file.attrs['metric'] = framecorr_results['metric']
            frame_corr_file.attrs['bad_frames'] = framecorr_results[fn]['bad_frames']
        else:
            frame_corr_file = frame_corr_grp[fn]
    
    #%%
    # -------------------------------------------------------------------------
    # 3. Identify border pixels:
    # -------------------------------------------------------------------------
    
    #%
    metrics.close()
        
