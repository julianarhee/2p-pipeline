#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 16:41:33 2018

@author: juliana
"""

import os
import glob
import optparse
import json
import h5py
import datetime
import random
import itertools
import matplotlib
matplotlib.use('agg')

import pandas as pd
import pylab as pl
import numpy as np
import pandas as pd
import seaborn as sns
import tifffile as tf
from PIL import Image, ImageChops
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from pipeline.python.utils import natural_keys, replace_root, label_figure
from pipeline.python.paradigm import utils as util
from pipeline.python.classifications import test_responsivity as resp #import calculate_roi_responsivity, group_roidata_stimresponse, find_barval_index
from pipeline.python.classifications import osi_dsi as osi
from pipeline.python.classifications import linear_svc as lsvc
from pipeline.python.traces.utils import load_TID

from collections import Counter



#%%


    
from skimage.measure import block_reduce
from mpl_toolkits.axes_grid1 import make_axes_locatable

def create_activity_map(acquisition_dir, run, rootdir=''):
    
    # TODO:  calculate activity map by using the baseline frames and subtracting from the stimulus frames
    # Can just use first File in run to not deal with frame indices 

    # Use motion-corrected files:
    pid_path = glob.glob(os.path.join(acquisition_dir, run, 'processed', 'pids_*.json'))[0]
    with open(pid_path, 'r') as f: pids = json.load(f)
    if len(pids.keys()) > 1:
        print "Multiple processing IDs found:"
        pkeys = []
        for pix, pkey, pid in enumerate(pids.items()):
            print pix, pkey
            print '%    ', pid
            pkeys.append(pkey)
        pix = int(raw_input("Select IDX of pkey to use: "))
        pid = pids[pkeys[pix]]
    else:
        pid = pids[pids.keys()[0]]
    ref_file = pid['PARAMS']['motion']['ref_file']
    tif_source_dir = pid['PARAMS']['motion']['destdir']
    if rootdir not in tif_source_dir:
        tif_source_dir = replace_root(tif_source_dir, rootdir, pid['PARAMS']['source']['animalid'], pid['PARAMS']['source']['session'])
    
    # Downsample:
    tif_fpath = glob.glob(os.path.join(tif_source_dir, '*File%03d.tif' % ref_file))[0]
    print "Loading ref img from: %s" % tif_fpath
    stack = tf.imread(tif_fpath)
    frames_tmp = np.empty((stack.shape[0], stack.shape[1]/2, stack.shape[2]/2), dtype='uint16')
    for fr in range(stack.shape[0]):
        frames_tmp[fr,:,:] = block_reduce(stack[fr, :, :], (2, 2), func=np.mean) + 10000

    # Get frame info for trial epochs:
    parsed_frames_fpath = glob.glob(os.path.join(acquisition_dir, run, 'paradigm', 'parsed_frames_*.hdf5'))[-1]
    parsed_frames = h5py.File(parsed_frames_fpath, 'r')

    # Get all trials contained in current .tif file:
    trials_in_block = sorted([t for t in parsed_frames.keys() \
                              if parsed_frames[t]['frames_in_run'].attrs['aux_file_idx'] == ref_file-1], key=natural_keys)
    print "N trials in block: %i" % len(trials_in_block)
    frame_indices = np.hstack([np.array(parsed_frames[t]['frames_in_run']) \
                               for t in trials_in_block])
    stim_onset_idxs = np.array([parsed_frames[t]['frames_in_run'].attrs['stim_on_idx'] \
                                for t in trials_in_block])
    
    # Re-index frame indices for current tif file (i.e., block)
    frame_indices = frame_indices - stack.shape[0] * (ref_file-1)
    stim_onset_idxs = stim_onset_idxs - stack.shape[0] * (ref_file-1)
    stim_on_relative = np.array([list(frame_indices).index(i) for i in stim_onset_idxs])
    
    # Make "nonnegative" so no funky df/f
    frames = np.zeros(frames_tmp.shape, dtype='uint16')
    frames[:] = frames_tmp + 10000 #32768
    
    with open(os.path.join(acquisition_dir, run, '%s.json' % run), 'r') as f:
        run_info = json.load(f)
    framerate = run_info['frame_rate']
    n_bas_frames = int(round(parsed_frames[trials_in_block[0]]['frames_in_run'].attrs['baseline_dur_sec'] * framerate))
    n_stim_frames = int(round(parsed_frames[trials_in_block[0]]['frames_in_run'].attrs['stim_dur_sec'] * framerate))
    bas_frames = [np.arange(stim_on - n_bas_frames, stim_on) for stim_on in stim_on_relative]
    stim_frames = [np.arange(stim_on - n_stim_frames, stim_on) for stim_on in stim_on_relative]

    # Just use average of baseline values:
    grandmean_baseline = frames[bas_frames,:,:].mean()
    mean_frames = np.empty((len(stim_frames), frames.shape[1], frames.shape[2]), dtype=float)
    for trial, stim in enumerate(stim_frames):
        im = (frames[stim,:,:] - grandmean_baseline) / grandmean_baseline
        mean_frames[trial,:,:] = im.mean(axis=0)

    activity_map = mean_frames.mean(axis=0)
    
    return activity_map
    
    
    
def load_traceid_zproj(traceid_dir, rootdir=''):
    run_dir = traceid_dir.split('/traces')[0]
    traceid = os.path.split(traceid_dir)[-1].split('_')[0]
    TID = load_TID(run_dir, traceid)
    
    session_dir = os.path.split(os.path.split(run_dir)[0])[0]
    session = os.path.split(session_dir)[-1]
    with open(os.path.join(session_dir, 'ROIs', 'rids_%s.json' % session), 'r') as f:
        rdict = json.load(f)
    
    RID = rdict[TID['PARAMS']['roi_id']]
    ref_file = 'File%03d' % int(RID['PARAMS']['options']['ref_file'])
    
    maskpath = os.path.join(traceid_dir, 'MASKS.hdf5')
    masks = h5py.File(maskpath, 'r')
    
    # Plot on MEAN img:
    if ref_file not in masks.keys():
        ref_file = masks.keys()[0]
    img_src = masks[ref_file]['Slice01']['zproj'].attrs['source']
    if 'std' in img_src:
        img_src = img_src.replace('std', 'mean')
    if rootdir not in img_src:
        animalid = traceid_dir.split(optsE.rootdir)[-1].split('/')[1]
        session = traceid_dir.split(optsE.rootdir)[-1].split('/')[2]
        img_src = replace_root(img_src, rootdir, animalid, session)
    
    img = tf.imread(img_src)

    return img    

#%%
def plot_image(image_fpath, scale=8, ax=None):
    if ax is None:
        fig, ax = pl.subplots()
    
    img = Image.open(image_fpath)
    img = trim_whitespace(img, scale=scale)
    ax.imshow(img)
    ax.axis('off')
    
    
def trim_whitespace(im, scale=2):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, scale, -1)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)
    
    
def colorcode_histogram(bins, ppatches, color='m'):

    indices_to_label = [resp.find_barval_index(v, ppatches) for v in bins]
    for ind in indices_to_label:
        ppatches.patches[ind].set_color(color)
        ppatches.patches[ind].set_alpha(0.5)
    
#%%
#blobs_roistats = get_roi_stats(optsE.rootdir, optsE.animalid, optsE.session, optsE.acquisition, blobs_run, blobs_traceid) #blobs_traceid.split('_')[0])

#rootdir=optsE.rootdir; animalid=optsE.animalid; session=optsE.session; acquisition=optsE.acquisition;
#run = blobs_run; traceid = blobs_traceid;

def get_roi_stats(rootdir, animalid, session, acquisition, run, traceid, create_new=False):
    
    acquisition_dir = os.path.join(rootdir, animalid, session, acquisition) 

    if create_new is False:
        try:
            roistats_fpath = sorted(glob.glob(os.path.join(acquisition_dir, run, 'traces', '%s*' % traceid, 'sorted_rois', 'roistats_results*.npz')))[-1]
            roistats = np.load(roistats_fpath)
        except Exception as e:
            print "** No roi stats found... Testing responsivity now."
            create_new = True
            
    if create_new:
        responsivity_opts = create_function_opts(rootdir=rootdir, animalid=animalid, 
                                                 session=session, 
                                                 acquisition=acquisition, 
                                                 run=run, traceid=traceid)
        responsivity_opts.extend(['-d', 'corrected', '--nproc=4', '--par', '--new'])
        roistats_fpath = resp.calculate_roi_responsivity(responsivity_opts)
        roistats = np.load(roistats_fpath)
        
    visual_test = str(roistats['responsivity_test'])
    selective_test = str(roistats['selectivity_test'])
    rois_visual = roistats['sorted_visual']
    rois_selective = roistats['sorted_selective']   
    
    roistats = {'visual_test': visual_test,
                'selective_test': selective_test,
                'rois_visual': rois_visual,
                'rois_selective': rois_selective
                }
    
    return roistats

#%%

def hist_roi_stats(df_by_rois, roistats, ax=None):
        
    if ax is None:
        fig, ax = pl.subplots()
        
    max_zscores_all = pd.Series([roidf.groupby('config')['zscore'].mean().max() for roi, roidf in df_by_rois])
    max_zscores_visual = [df_by_rois.get_group(roi).groupby('config')['zscore'].mean().max() for roi in roistats['rois_visual']]
    max_zscores_selective = [df_by_rois.get_group(roi).groupby('config')['zscore'].mean().max() for roi in roistats['rois_selective']]
    
    count, division = np.histogram(max_zscores_all, bins=100)
    ppatches = max_zscores_all.hist(bins=division, color='gray', ax=ax, grid=False, alpha=0.5)
    
    # Highlight p-eta2 vals for significant neurons:
    visual_bins = list(set([binval for ix,binval in enumerate(division) for zscore in max_zscores_visual if division[ix] < zscore <= division[ix+1]]))
    selective_bins = list(set([binval for ix,binval in enumerate(division) for zscore in max_zscores_selective if division[ix] < zscore <= division[ix+1]]))
    
    colorcode_histogram(visual_bins, ppatches, color='magenta')
    colorcode_histogram(selective_bins, ppatches, color='cornflowerblue')
    #axes_flat[hist_ix].set_xlabel('zscore')
    ax.set_ylabel('counts')
    ax.set_title('distN of zscores (all rois)', fontsize=10)
    
    # Label histogram colors:
    visual_pval= 0.05; visual_test = roistats['visual_test'].split('_')[-1];
    visual_str = 'visual: %i/%i (p<%.2f, %s)' % (len(roistats['rois_visual']), nrois_total, visual_pval, visual_test)
    texth = ax.get_ylim()[-1] + 2
    textw = ax.get_xlim()[0] + .1
    ax.text(textw, texth, visual_str, fontdict=dict(color='magenta', size=10))
    
    selective_pval = 0.05; #selective_test = str(roistats['selectivity_test']);
    selective_str = 'sel: %i/%i (p<%.2f, %s)' % (len(roistats['rois_selective']), len(roistats['rois_visual']), selective_pval, roistats['selective_test'])
    ax.text(textw, texth-1, selective_str, fontdict=dict(color='cornflowerblue', size=10))
    ax.set_ylim([0, texth + 2])
    ax.set_xlim([0, ax.get_xlim()[-1]])
    
    sns.despine(trim=True, offset=4, ax=ax)
    
    return

#%%
def combine_run_info(D, identical_fields=[], combined_fields=[]):
    
    run_info = {}
    
    info_keys = D[D.keys()[0]]['run_info'].keys()
    
    for info_key in info_keys:
        print info_key
        run_vals = [D[curr_run]['run_info'][info_key] for curr_run in D.keys()]
        if info_key in identical_fields:
            if isinstance(run_vals[0], list):
                assert all([run_vals[0] == D[curr_run]['run_info'][info_key] for curr_run in D.keys()]), "%s: All vals not equal!" % info_key
                run_info[info_key] = run_vals[0]
            elif isinstance(run_vals[0], (int, long, float)):
                uvals = np.unique(list(set(run_vals)))
                assert len( uvals ) == 1, "** %s: runs do not match!" % info_key
            run_info[info_key] = uvals[0]
        elif info_key in combined_fields:

            if isinstance(run_vals[0], dict):
                uvals = dict((k,v) for rd in run_vals for k,v in rd.items())
            else:
                if isinstance(run_vals[0], list):
                    uvals = np.unique(list(itertools.chain.from_iterable(run_vals))) 
                else:
                    uvals = list(set(run_vals))
                if isinstance(uvals[0], (str, unicode)):
                    run_info[info_key] = list(uvals)
                elif isinstance(uvals[0], (int, float, long)):
                    run_info[info_key] = sum(uvals)
        else:
            print "%s: Not sure what to do with this..." % info_key
            run_info[info_key] = None
            
    return run_info

def combine_static_runs(check_blobs_dir, combined_name='combined', create_new=False):
    
    # First check if specified combo run exists:
    traceid_string = '_'.join([blobdir.split('/traces/')[-1] for blobdir in sorted(check_blobs_dir)])
    acquisition_dir = os.path.split(check_blobs_dir[0].split('/traces/')[0])[0]
    combined_darray_dir = os.path.join(acquisition_dir, combined_name, 'traces', traceid_string, 'data_arrays')
    if not os.path.exists(combined_darray_dir): os.makedirs(combined_darray_dir)
    combo_dpath = os.path.join(combined_darray_dir, 'datasets.npz')    
    
    if create_new is False:
        try:
            assert os.path.exists(combo_dpath), "Combined dset %s does not exist!" % combined_name
        except Exception as e:
            print "Creating new combined dataset: %s" % combined_name
            print "Combining data from dirs:\n", check_blobs_dir
            create_new = True
        
    if create_new:
        
        D = dict()
        for blobdir in check_blobs_dir:
            curr_run = os.path.split(blobdir.split('/traces')[0])[-1]
            print "Getting data array for run: %s" % curr_run
            darray_fpath = glob.glob(os.path.join(blobdir, 'data_arrays', 'datasets.npz'))[0]
            curr_dset = np.load(darray_fpath)
            
            D[curr_run] = {'data':  curr_dset['corrected'],
                           'labels_df':  pd.DataFrame(data=curr_dset['labels_data'], columns=curr_dset['labels_columns']),
                           'sconfigs':  curr_dset['sconfigs'][()],
                           'run_info': curr_dset['run_info'][()]
                           }
            
        unique_sconfigs = list(np.unique(np.array([D[curr_run]['sconfigs'].values() for curr_run in D.keys()])))
        sconfigs = dict(('config%03d' % int(cix+1), cfg) for cix, cfg in enumerate(unique_sconfigs))
        
        # Remap config names for each run:
        last_trial_prev_run = 0
        prev_run = None
        for ridx, curr_run in enumerate(sorted(D.keys(), key=natural_keys)):
            print curr_run
            # Get the correspondence between current run's original keys, and the new keys from the combined stim list
            remapping = dict((oldkey, newkey) for oldkey, oldval in D[curr_run]['sconfigs'].items() for newkey, newval in sconfigs.items() 
                                                if newval==oldval)
            # Create dict of DF indices <--> new config key to replace each index once
            ixs_to_replace = dict((ix, remapping[oldval]) for ix, oldval in 
                                      zip(D[curr_run]['labels_df']['config'].index.tolist(), D[curr_run]['labels_df']['config'].values))
            # Replace old config with new config at the correct index
            D[curr_run]['labels_df']['config'].put(ixs_to_replace.keys(), ixs_to_replace.values())
            
            # Also replace trial names so that they have unique values between the two runs:
            if prev_run is not None:        
                last_trial_prev_run = int(sorted(D[prev_run]['labels_df']['trial'].unique(), key=natural_keys)[-1][5:]) #len(D[curr_run]['labels_df']['trial'].unique())
    
                trials_to_replace = dict((ix, 'trial%05d' % int(int(oldval[5:]) + last_trial_prev_run)) for ix, oldval in 
                                     zip(D[curr_run]['labels_df']['trial'].index.tolist(), D[curr_run]['labels_df']['trial'].values))
                D[curr_run]['labels_df']['trial'].put(trials_to_replace.keys(), trials_to_replace.values())
                
            prev_run = curr_run
            
    
        # Combine runs in order of their alphanumeric name:
        tmp_data = np.vstack([D[curr_run]['data'] for curr_run in sorted(D.keys(), key=natural_keys)])
        tmp_labels_df = pd.concat([D[curr_run]['labels_df'] for curr_run in sorted(D.keys(), key=natural_keys)], axis=0).reset_index(drop=True)
        
        # Get run_info dict:
        identical_fields = ['trace_type', 'roi_list', 'nframes_on', 'framerate', 'stim_on_frame', 'nframes_per_trial']
        combined_fields = ['traceid_dir', 'trans_types', 'transforms', 'nfiles', 'ntrials_total']
        
        rinfo = combine_run_info(D, identical_fields=identical_fields, combined_fields=combined_fields)
        
        replace_fields = ['condition_list', 'ntrials_by_cond']
        replace_keys = [k for k,v in rinfo.items() if v is None]
        assert replace_fields == replace_keys, "Replace fields (%s) and None keys (%s) do not match!" % (str(replace_fields), str(replace_keys))
        rinfo['condition_list'] = sorted(tmp_labels_df['config'].unique())
        rinfo['ntrials_by_cond'] = dict((cf, len(tmp_labels_df[tmp_labels_df['config']==cf]['trial'].unique())) for cf in rinfo['condition_list'])
        
        # CHeck N trials per condition:
        ntrials_by_cond = list(set([v for k,v in rinfo['ntrials_by_cond'].items()]))
        if len(ntrials_by_cond) > 1:
            print "Uneven numbers of trials per cond. Making equal."
            configs_with_more = [k for k,v in rinfo['ntrials_by_cond'].items() if v==max(ntrials_by_cond)]
            ntrials_target = min(ntrials_by_cond)
            remove_ixs = []
            for cf in configs_with_more:
                curr_trials = tmp_labels_df[tmp_labels_df['config']==cf]['trial'].unique()
                rand_trial_ixs = random.sample(range(0, len(curr_trials)), ntrials_target)
                selected_trials = curr_trials[rand_trial_ixs] 
                ixs = tmp_labels_df[tmp_labels_df['trial'].isin(selected_trials)].index.tolist()
                remove_ixs.extend(ixs)
            
            all_ixs = np.arange(0, tmp_labels_df.shape[0])
            kept_ixs = np.delete(all_ixs, remove_ixs)
            
            labels_df = tmp_labels_df.iloc[kept_ixs, :].reset_index(drop=True)
            data = tmp_data[kept_ixs, :]
            rinfo['ntrials_by_cond'] = dict((cf, len(labels_df[labels_df['config']==cf]['trial'].unique())) for cf in rinfo['condition_list'])

        else:
            labels_df = tmp_labels_df
            data = tmp_data
            
        
        # Save it:
        np.savez(combo_dpath,
                 corrected=data,
                 labels_data=labels_df,
                 labels_columns=labels_df.columns.tolist(),
                 run_info = rinfo,
                 sconfigs=sconfigs
                 )
    
    return combo_dpath

#%%
def get_data_and_labels(dataset, data_type='corrected'):
    #print "Loading data..."
    
    #dataset = np.load(data_fpath)
    assert data_type in dataset.keys(), "Specified dtype - %s - not found!\nChoose from: %s" % (data_type, str(dataset.keys()))
    roidata = dataset[data_type]
    labels_df = pd.DataFrame(data=dataset['labels_data'], columns=dataset['labels_columns'])
    sconfigs = dataset['sconfigs'][()]

    return roidata, labels_df, sconfigs

#%
    

def create_function_opts(rootdir='', animalid='', session='', acquisition='',
                         run='', traceid=''):
    opts = ['-D', rootdir, '-i', animalid, '-S', session, '-A', acquisition,
            '-R', run, '-t', traceid]
    
    return opts

def extract_options(options):

    parser = optparse.OptionParser()

    parser.add_option('-D', '--root', action='store', dest='rootdir',
                          default='/nas/volume1/2photon/data',
                          help='data root dir (dir w/ all animalids) [default: /nas/volume1/2photon/data, /n/coxfs01/2pdata if --slurm]')
    parser.add_option('--slurm', action='store_true', dest='slurm', default=False, help="set if running as SLURM job on Odyssey")

    parser.add_option('-i', '--animalid', action='store', dest='animalid',
                          default='', help='Animal ID')

    # Set specific session/run for current animal:
    parser.add_option('-S', '--session', action='store', dest='session',
                          default='', help='session dir (format: YYYMMDD_ANIMALID')
    parser.add_option('-A', '--acq', action='store', dest='acquisition',
                          default='FOV1', help="acquisition folder (ex: 'FOV1_zoom3x') [default: FOV1]")
    parser.add_option('-d', '--data-type', action='store', dest='data_type',
                          default='corrected', help="trace type [default: 'corrected']")


    parser.add_option('--par', action='store_true', dest='multiproc', default=False, help="set if want to run MP on roi stats, when possible")
    parser.add_option('--nproc', action='store', dest='nprocesses', default=4, help="N processes if running in par (default=4)")
    parser.add_option('--new', action='store_true', dest='create_new', default=False, help="set to run anew")
    
    
    # Run specific info:
    parser.add_option('-g', '--gratings', dest='gratings_traceid', default='traces001', action='store', help="traceid for GRATINGS [default: traces001]")
    parser.add_option('-r', '--retino', dest='retino_traceid', default=None, action='store', help='analysisid for RETINO [default assumes only 1 roi-based analysis]')
    parser.add_option('-b', '--objects', dest='blobs_traceid_list', default=[], action='append', nargs=1, help='list of blob traceids [default: [traces001]')
    
    #parser.add_option('-t', '--traceid', dest='traceid', default=None, action='store', help="datestr YYYYMMDD_HH_mm_SS")

    (options, args) = parser.parse_args(options)

    return options

#%%
options = ['-D', '/mnt/odyssey', '-i', 'CE077', '-S', '20180523', '-A', 'FOV1_zoom1x',
           '-d', 'corrected',
           '-g', 'traces003', '-b', 'traces002', '-b', 'traces002', '-r', 'analysis001'
           ]


optsE = extract_options(options)

data_type = optsE.data_type
acquisition_dir = os.path.join(optsE.rootdir, optsE.animalid, optsE.session, optsE.acquisition)


traceid_dirs = {'gratings': None, 
                'blobs': None}

# Get gratings traceid dir:
check_gratings_dir = glob.glob(os.path.join(acquisition_dir, 'gratings*', 'traces', '%s*' % optsE.gratings_traceid))
if len(check_gratings_dir) > 1:
    combo_gratings_dpath = combine_static_runs(check_gratings_dir, combined_name = 'combined_gratings_static')
    traceid_dirs['gratings'] = combo_gratings_dpath.split('/data_arrays')[0]
else:
    traceid_dirs['gratings'] = check_gratings_dir[0]

# Get static-blobs traceid dir(s):
check_blobs_dir = list(set([item for sublist in [glob.glob(os.path.join(acquisition_dir, 'blobs*', 'traces', '%s*' % b)) 
                                for b in optsE.blobs_traceid_list] for item in sublist]))
check_blobs_dir = [b for b in check_blobs_dir if 'dynamic' not in b]
if len(check_blobs_dir) > 1:
    combo_blobs_dpath = combine_static_runs(check_blobs_dir, combined_name = 'combined_blobs_static')
    traceid_dirs['blobs'] = combo_blobs_dpath.split('/data_arrays')[0]
else:
    traceid_dirs['blobs'] = check_blobs_dir[0]

data_identifier ='_'.join([optsE.rootdir, optsE.animalid, optsE.session, optsE.acquisition])


#%%

use_dff = True
zproj_run = os.path.split(traceid_dirs['gratings'].split('/traces/')[0])[-1] #[0])[-1]

if use_dff:
    zproj = create_activity_map(acquisition_dir, zproj_run, rootdir=optsE.rootdir)
else:
    zproj = load_traceid_zproj(traceid_dirs['gratings'], rootdir=optsE.rootdir)


#%%
# a)  Load data for gratings (w/ specified traceid) - do stats, plot summary figs
# b)  Load data for static blobs " "
#       - need to combine 5x5 for each transform if tested separately 
#       - 

fig, axes = pl.subplots(3,3, figsize=(15,20)) #pl.figure()
axes_flat = axes.flat

zproj_ix = 0
retino_screen_ix = 1
rf_ix = 2

ori_hist_ix = 3
ori_osi_ix = 4
ori_decode_ix = 5

blobs_hist_ix = 6
transforms_ix = 7


# SUBPLOT 1:  Mean / zproj image
# -----------------------------------------------------------------------------
im = axes_flat[zproj_ix].imshow(zproj, cmap='gray')
axes_flat[zproj_ix].axis('off')
divider = make_axes_locatable(axes_flat[zproj_ix])
cax = divider.append_axes('right', size='5%', pad=0.05)
pl.colorbar(im, cax=cax)
axes_flat[zproj_ix].set_title('dF/F map')

axes_flat[rf_ix].axis('off')

#%%
create_new = False

# GRATINGS:
gratings_traceid = os.path.split(traceid_dirs['gratings'])[-1]
gratings_run = os.path.split(traceid_dirs['gratings'].split('/traces/')[0])[-1] #[0])[-1]

# Load data array:
data_fpath = os.path.join(traceid_dirs['gratings'], 'data_arrays', 'datasets.npz')
gratings_dataset = np.load(data_fpath)

gratings_roistats = get_roi_stats(optsE.rootdir, optsE.animalid, optsE.session, optsE.acquisition, gratings_run, gratings_traceid.split('_')[0], create_new=create_new)

gratings_roidata, gratings_labels_df, gratings_sconfigs = get_data_and_labels(gratings_dataset, data_type=data_type)
gratings_df_by_rois = resp.group_roidata_stimresponse(gratings_roidata, gratings_labels_df)
nrois_total = gratings_roidata.shape[-1]


# SUBPLOT 2:  Histogram of visual vs. selective ROIs (use zscores)
# -----------------------------------------------------------------------------
axes_flat[ori_hist_ix].clear()
hist_roi_stats(gratings_df_by_rois, gratings_roistats, ax=axes_flat[ori_hist_ix])
#bb = axes_flat[ori_hist_ix].get_position().bounds
#new_bb = [bb[0], bb[1], bb[2], bb[3]*0.95]
#axes_flat[ori_hist_ix].set_position(new_bb)

axes_flat[ori_hist_ix].set_title('gratings: distN of zscores')
axes_flat[ori_hist_ix].set_xlabel('zscore')


# SUBPLOT 3:  Retinotopy:
# -----------------------------------------------------------------------------
retinovis_fpath = glob.glob(os.path.join(optsE.rootdir, optsE.animalid, optsE.session, optsE.acquisition, 
                                         'retino_*', 'retino_analysis', 'analysis*', 'visualization', '*.png'))[0]
axes_flat[retino_screen_ix].clear()
plot_image(retinovis_fpath, scale=20, ax=axes_flat[retino_screen_ix])
bb = axes_flat[retino_screen_ix].get_position().bounds
new_bb = [bb[0]*0.95, bb[1]*0.98, bb[2]*1.3, bb[3]*1.2]
axes_flat[retino_screen_ix].set_position(new_bb)

#retino = Image.open(retinovis_fpath)
#retino = trim_whitespace(retino, scale=scale)
#axes_flat[retino_screen_ix].imshow(retino)
#axes_flat[retino_screen_ix].axis('off')

#%

# SUBPLOT 4:  DistN of preferred orientations:
# -----------------------------------------------------------------------------
metric = 'meanstim'
selectivity = osi.get_OSI_DSI(gratings_df_by_rois, gratings_sconfigs, roi_list=gratings_roistats['rois_visual'], metric=metric)
cmap = 'hls'
colorvals = sns.color_palette(cmap, len(gratings_sconfigs))
osi.hist_preferred_oris(selectivity, colorvals, metric=metric, save_and_close=False, ax=axes_flat[ori_osi_ix])
sns.despine(trim=True, offset=4, ax=axes_flat[ori_osi_ix])
#bb = axes_flat[ori_osi_ix].get_position().bounds
#new_bb = [bb[0], bb[1], bb[2], bb[3]]
#axes_flat[ori_osi_ix].set_position(new_bb)
axes_flat[ori_osi_ix].set_title('orientation selectivity')


# SUBPLOT 5:  Decoding performance for linear classifier on orientations:
# -----------------------------------------------------------------------------
clfparams = lsvc.get_default_gratings_params()
cX, cy, inputdata, is_cnmf = lsvc.get_formatted_traindata(clfparams, gratings_dataset, gratings_traceid)
cX_std = StandardScaler().fit_transform(cX)
if cX_std.shape[0] > cX_std.shape[1]: # nsamples > nfeatures
    clfparams['dual'] = False
else:
    clfparams['dual'] = True
cX, cy, clfparams['class_labels'] = lsvc.format_stat_dataset(clfparams, cX_std, cy, gratings_sconfigs, relabel=False)
    
if clfparams['classifier'] == 'LinearSVC':
    svc = LinearSVC(random_state=0, dual=clfparams['dual'], multi_class='ovr', C=clfparams['C'])

axes_flat[ori_decode_ix].clear()
predicted, true = lsvc.do_cross_validation(svc, clfparams, cX_std, cy, data_identifier=data_identifier)
lsvc.plot_normed_confusion_matrix(predicted, true, clfparams, ax=axes_flat[ori_decode_ix])
sns.despine(trim=True, offset=4, ax=axes_flat[ori_decode_ix])
bb = axes_flat[ori_decode_ix].get_position().bounds
new_bb = [bb[0], bb[1]*1.05, bb[2]*0.9, bb[3]*0.9]
axes_flat[ori_decode_ix].set_position(new_bb)
        
#%%

# SUBPLOT 6:  Complex stimuli...
# -----------------------------------------------------------------------------

#%
#
#options = ['-D', '/mnt/odyssey', '-i', 'CE077', '-S', '20180817', '-A', 'FOV2_zoom1x',
#           '-d', 'corrected',
#           '-R', 'blobs_static_xpos', '-t', 'traces001',
#           ]
#optsE = extract_options(options)
#acquisition_dir = os.path.join(optsE.rootdir, optsE.animalid, optsE.session, optsE.acquisition) 
#traceid_dir = util.get_traceid_from_acquisition(acquisition_dir, optsE.run, optsE.traceid)
#traceid = os.path.split(traceid_dir)[-1]

# BLOBS
blobs_traceid = os.path.split(traceid_dirs['blobs'])[-1]
blobs_run = os.path.split(traceid_dirs['blobs'].split('/traces/')[0])[-1] #[0])[-1]


# Load data array:
data_fpath = os.path.join(traceid_dirs['blobs'], 'data_arrays', 'datasets.npz')
blobs_dataset = np.load(data_fpath)

blobs_roistats = get_roi_stats(optsE.rootdir, optsE.animalid, optsE.session, optsE.acquisition, blobs_run, blobs_traceid, create_new=create_new) #blobs_traceid.split('_')[0])


blobs_roidata, blobs_labels_df, blobs_sconfigs = get_data_and_labels(blobs_dataset, data_type=data_type)
blobs_df_by_rois = resp.group_roidata_stimresponse(blobs_roidata, blobs_labels_df)
nrois_total = blobs_roidata.shape[-1]


# SUBPLOT 6:  Complex stimuli...
# -----------------------------------------------------------------------------
roistats = get_roi_stats(optsE.rootdir, optsE.animalid, optsE.session, optsE.acquisition, blobs_run, blobs_traceid)
axes_flat[blobs_hist_ix].clear()
hist_roi_stats(blobs_df_by_rois, roistats, ax=axes_flat[blobs_hist_ix])
axes_flat[blobs_hist_ix].set_title('blobs: distN of zscores')

bb = axes_flat[blobs_hist_ix].get_position().bounds
new_bb = [bb[0], bb[1]*0.70, bb[2], bb[3]]
axes_flat[blobs_hist_ix].set_position(new_bb)
axes_flat[blobs_hist_ix].set_xlabel('zscore')

#%%


# SUBPLOT 7:  Complex stimuli 2...
# -----------------------------------------------------------------------------

metric = 'zscore'
responses_by_config = dict((roi, blobs_df_by_rois.get_group(roi).groupby('config')[metric].mean()) for roi in roistats['rois_visual'])


transform_dict, object_transformations = util.get_transforms(blobs_sconfigs)
transforms_tested = [k for k, v in object_transformations.items() if len(v) > 0]

sconfigs_df = pd.DataFrame(blobs_sconfigs).T
responses = []
for roi, rdf in responses_by_config.items():
    responses.append(pd.concat([sconfigs_df, rdf, pd.Series(data=[roi for _ in range(rdf.shape[0])], index=rdf.index, name='roi')], axis=1))
responses = pd.concat(responses, axis=0)    

df_columns = ['object', 'roi', metric]
df_columns.extend(transforms_tested)

data = responses[df_columns]


#for tix, trans in enumerate(transforms_tested):
#    new_trans_str = 'transval%i' % tix
#    data.rename(columns={trans: new_trans_str}, inplace=True)
#    data.loc[:, 'transform%i' % tix] = [trans for _ in range(data.shape[0])]

rois = data.groupby('roi')

# Colors = cells
rois_to_plot = roistats['rois_selective'][0:10]
nrois_plot = len(rois_to_plot) 
colors = sns.color_palette('husl', nrois_plot*2)

# Shapes = objects
nobjects = len(responses['object'].unique())
markers = ['o', 'P', '*', '^', 's', 'd']
marker_kws = {'markersize': 10, 'linewidth': 1, 'alpha': 0.3}
    
for trans_ix, transform in enumerate(transforms_tested):
    tix = transforms_ix + trans_ix
    plot_list = []
    for roi, df in rois:
        if roi not in rois_to_plot:
            continue
        
        df2 = df.pivot_table(index='object', columns=transform, values=metric)
        #new_df = pd.concat([df2, pd.Series(data=[roi for _ in range(df2.shape[0])], index=df2.index, name='roi')], axis=1)
        plot_list.append(df2)
        
    data = pd.concat(plot_list, axis=0)
    #%
    axes_flat[tix].clear()

    for ridx, r in enumerate(np.arange(0, data.shape[0], nobjects)):
        for object_ix in range(nobjects):
            axes_flat[tix].plot(data.iloc[r+object_ix, :], color=colors[ridx], marker=markers[object_ix], **marker_kws) #'.-')
    axes_flat[tix].set_xticks(data.keys().tolist())
    axes_flat[tix].set_ylabel(metric)
    axes_flat[tix].set_xlabel(transforms_tested[0])
    axes_flat[tix].set_title(transform)
    
    bb = axes_flat[tix].get_position().bounds
    new_bb = [bb[0], bb[1]*0.8, bb[2], bb[3]]
    axes_flat[tix].set_position(new_bb)
    axes_flat[tix].set_xlabel('zscore')

    sns.despine(ax=axes_flat[tix])

from matplotlib.lines import Line2D
legend_objects = []
object_names = data.iloc[0:nobjects].index.tolist()
for object_ix in range(nobjects):
    legend_objects.append(Line2D([0], [0], color='w', markerfacecolor='k', 
                                 marker=markers[object_ix], label=object_names[object_ix], 
                                 linewidth=2, markersize=10))
    
axes_flat[tix].legend(handles=legend_objects, loc=9, bbox_to_anchor=(-0.2, -0.2), ncol=nobjects) # loc='upper right')


label_figure(fig, data_identifier)
figname = '%s_acquisition_summary.png' % optsE.acquisition

pl.savefig(os.path.join(os.path.split(acquisition_dir)[0], figname))

pl.close()

## get the tick label font size
#fontsize_pt = 10 #pl.rcParams['ytick.labelsize']
#dpi = 72.27
#
## comput the matrix height in points and inches
#matrix_height_pt = fontsize_pt * data.shape[0]
#matrix_height_in = matrix_height_pt / dpi
#
## compute the required figure height 
#top_margin = 0.02  # in percentage of the figure height
#bottom_margin = 0.02 # in percentage of the figure height
#figure_height = matrix_height_in / (1 - top_margin - bottom_margin)
#
#fig, ax = pl.subplots(
#        figsize=(6,figure_height), 
#        gridspec_kw=dict(top=1-top_margin, bottom=bottom_margin))

#cbar_ax = fig.add_axes([0.91, 0.1, 0.02, 0.4])
#ax = axes_flat[7]
#ax.clear()
#nobjects = len(responses['object'].unique())



#sns.heatmap(data, ax=ax, cbar_kws={'label': metric}, cmap='magma')
#
#ax.hlines(np.arange(0, data.shape[0], nobjects), *ax.get_xlim(), color='w', linewidth=1)
#ax.set_yticklabels([])
#ax.set_yticks(np.arange(0, nobjects)+0.5)
#ax.set_yticklabels(data.index.tolist()[0:nobjects], fontsize=5, rotation=0)
#ax.set_ylabel('cells')
#pl.subplots_adjust(top=0.9, right=0.85)
