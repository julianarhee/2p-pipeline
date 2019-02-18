#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 16:41:33 2018

@author: juliana
"""

import os
import sys
import glob
import optparse
import json
import h5py
import datetime
import random
import itertools
import shutil
import pprint
pp = pprint.PrettyPrinter(indent=4)
import matplotlib
matplotlib.use('agg')
import traceback
import pandas as pd
import pylab as pl
import numpy as np
import pandas as pd
import seaborn as sns
import tifffile as tf
import cPickle as pkl
from PIL import Image, ImageChops
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from pipeline.python.utils import natural_keys, replace_root, label_figure
from pipeline.python.paradigm import utils as util
from pipeline.python.classifications import test_responsivity as resp #import calculate_roi_responsivity, group_roidata_stimresponse, find_barval_index
from pipeline.python.classifications import osi_dsi as osi
from pipeline.python.classifications import linear_svc as lsvc
from pipeline.python.traces.utils import load_TID
from pipeline.python.retinotopy import estimate_RF_size as RF

from collections import Counter
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec

    
from skimage.measure import block_reduce
from mpl_toolkits.axes_grid1 import make_axes_locatable


#%%



def create_activity_map(acquisition_dir, run, rootdir=''):
    
    # TODO:  calculate activity map by using the baseline frames and subtracting from the stimulus frames
    # Can just use first File in run to not deal with frame indices 

    # Use motion-corrected files:
    print os.listdir(os.path.join(acquisition_dir, run, 'processed'))
    pid_path = glob.glob(os.path.join(acquisition_dir, run, 'processed', 'pids_*.json'))[0]
    with open(pid_path, 'r') as f: pids = json.load(f)
    if len(pids.keys()) > 1:
        print "Multiple processing IDs found:"
        pkeys = []
        for pix, (pkey, pid) in enumerate(pids.items()):
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

    # Make "nonnegative" so no funky df/f
    frames = np.zeros(frames_tmp.shape, dtype='uint16')
    frames[:] = frames_tmp + 10000 #32768

    if 'retino' not in run:
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
   
    else:
        grandmean_baseline = frames.mean()
        activity_map =  (frames - grandmean_baseline) / grandmean_baseline
        activity_map = activity_map.mean(axis=0)
        print "ACTIVITY MAP:", activity_map.shape
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
        animalid = traceid_dir.split(rootdir)[-1].split('/')[1]
        session = traceid_dir.split(rootdir)[-1].split('/')[2]
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
def get_roi_stats(rootdir, animalid, session, acquisition, run, traceid, create_new=False, nproc=4):
    
    acquisition_dir = os.path.join(rootdir, animalid, session, acquisition) 

    if create_new is False:
        try:
            roistats_fpath = sorted(glob.glob(os.path.join(acquisition_dir, run, 'traces', '%s*' % traceid, 'sorted_rois', 'roistats_results*.npz')))[-1]
            roistats = np.load(roistats_fpath)
        except Exception as e:
            print "** No roi stats found... Testing responsivity now."
            #create_new = True
            
    responsivity_opts = create_function_opts(rootdir=rootdir, animalid=animalid, 
                                             session=session, 
                                             acquisition=acquisition, 
                                             run=run, traceid=traceid)
    responsivity_opts.extend(['-d', 'corrected', '--nproc=%i' % nproc, '--par'])
    if create_new:
        responsivity_opts.extend(['--new'])
    print responsivity_opts
        
    roistats_fpath = resp.calculate_roi_responsivity(responsivity_opts)
    roistats = np.load(roistats_fpath)
        
    visual_test = str(roistats['responsivity_test'])
    selective_test = str(roistats['selectivity_test'])
    rois_visual = [int(r) for r in roistats['sorted_visual']]
    rois_selective = [int(r) for r in roistats['sorted_selective']]   
    
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
    print "DF ROIS:", df_by_rois.groups.keys()
   
    max_zscores_all = pd.Series([roidf.groupby('config')['zscore'].mean().max() for roi, roidf in df_by_rois]) 
                                        # if not np.where(np.isnan(roidf['zscore']))[0].any()])
    max_zscores_visual = [df_by_rois.get_group(roi).groupby('config')['zscore'].mean().max() for roi in roistats['rois_visual']] 
                                        # if not np.where(np.isnan(df_by_rois.get_group(roi)['zscore']))[0].any()]
    max_zscores_selective = [df_by_rois.get_group(roi).groupby('config')['zscore'].mean().max() for roi in roistats['rois_selective']] 
                                        #if not np.where(np.isnan(df_by_rois.get_group(roi)['zscore']))[0].any()]
    
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
    nrois_total = len(df_by_rois.groups.keys())
    visual_pval= 0.05; visual_test = roistats['visual_test'].split('_')[-1];
    visual_str = 'visual: %i/%i (p<%.2f, %s)' % (len(roistats['rois_visual']), nrois_total, visual_pval, visual_test)
    texth = ax.get_ylim()[-1] #- 0.5 #+ 2
    textw = 0.01 #ax.get_xlim()[0] + .001
    ax.text(textw, texth-0.5, visual_str, fontdict=dict(color='magenta', size=16))
    
    selective_pval = 0.05; #selective_test = str(roistats['selectivity_test']);
    selective_str = 'sel: %i/%i (p<%.2f, %s)' % (len(roistats['rois_selective']), len(roistats['rois_visual']), selective_pval, roistats['selective_test'])
    ax.text(textw, texth-1.0, selective_str, fontdict=dict(color='cornflowerblue', size=16))
    #ax.set_ylim([0, texth + 2])
    ax.set_xlim([0, ax.get_xlim()[-1]])
    
    sns.despine(trim=False, offset=4, ax=ax)
    
    return


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


#%%
    
def get_data_sources(optsE):

    acquisition_dir = os.path.join(optsE.rootdir, optsE.animalid, optsE.session, optsE.acquisition)

    traceid_dirs = {'gratings': None, 
                    'blobs': None,
                    'objects': None}
                
    
    # Get gratings traceid dir:
    if len(optsE.gratings_traceid_list) > 0:
        print "Getting gratings..."
        traceid_dirs['gratings'] = get_traceid_dir_from_lists(acquisition_dir, optsE.gratings_run_list, optsE.gratings_traceid_list, stimtype='gratings', create_new=optsE.create_new)
    else:
        traceid_dirs.pop('gratings')

    # Get static-blobs traceid dir(s):
    if len(optsE.blobs_traceid_list) > 0:
        print "Getting blobs..."
        traceid_dirs['blobs'] = get_traceid_dir_from_lists(acquisition_dir, optsE.blobs_run_list, optsE.blobs_traceid_list, stimtype='blobs', create_new=optsE.create_new)               
    else:
        traceid_dirs.pop('blobs')
       
    if len(optsE.objects_traceid_list) > 0:
        print "Getting objects..."
        traceid_dirs['objects'] = get_traceid_dir_from_lists(acquisition_dir, optsE.objects_run_list, optsE.objects_traceid_list, stimtype='objects', create_new=optsE.create_new)               
    else:
        traceid_dirs.pop('objects')
 
    return traceid_dirs

def get_traceid_dir_from_lists(acquisition_dir, run_list, traceid_list, stimtype='', create_new=False, make_equal=True):
    print "Runs:", run_list
    print "TraceIDs:", traceid_list
    if len(run_list) > 0:
        check_run_dir = sorted([glob.glob(os.path.join(acquisition_dir, '*%s*' % run, 'traces', '%s*' % traceid))[0] for run, traceid in zip(run_list, traceid_list)], key=natural_keys)
    else:
        check_run_dir = sorted(list(set([item for sublist in [glob.glob(os.path.join(acquisition_dir, '*%s_run*' % stimtype, 'traces', '%s*' % traceid)) for traceid in traceid_list] for item in sublist if 'combined' not in item])), key=natural_keys)
    print "Found -- %s --  dirs:" % stimtype, check_run_dir

    # Check if should combine runs:
    if len(check_run_dir) > 1:
        print "Combining runs:", check_run_dir
        combo_dpath = util.combine_static_runs(check_run_dir, combined_name='combined_%s_static' % stimtype, create_new=create_new, make_equal=make_equal)
        traceid_dirs = combo_dpath.split('/data_arrays')[0]
    elif len(check_run_dir) == 1 and 'combined' in check_run_dir[0] or 'dynamic' in check_run_dir[0]:
        traceid_dirs = check_run_dir[0]
    else:
        #print os.listdir(glob.glob(os.path.join(acquisition_dir, '*%s*' % stimtype))[0])
        if any(['combined' in d for d in run_list]):
            check_run_dir = sorted(list(set([item for sublist in [glob.glob(os.path.join(acquisition_dir, '*%s*' % stimtype, 'traces', '%s*' % traceid)) for traceid in traceid_list] for item in sublist])), key=natural_keys)
        check_runs = [d for d in check_run_dir if 'movies' not in d and 'dynamic' not in d]
        print "1 run:", check_runs
        traceid_dirs = check_runs[0]

    return traceid_dirs

#%%

def run_gratings_classifier(dataset, sconfigs, traceid):
    clfparams = lsvc.get_default_gratings_params()
    cX, cy, inputdata, is_cnmf = lsvc.get_formatted_traindata(clfparams, dataset, traceid)
    # Check for NaNs:
    frames, bad_roi_ixs = np.where(np.isnan(cX))
    bad_rois = list(set(bad_roi_ixs))
    if len(bad_rois) > 0:
        cX = np.delete(cX, bad_rois, axis=1) 
    cX_std = StandardScaler().fit_transform(cX)
    if cX_std.shape[0] > cX_std.shape[1]: # nsamples > nfeatures
        clfparams['dual'] = False
    else:
        clfparams['dual'] = True
    cX, cy, clfparams['class_labels'] = lsvc.format_stat_dataset(clfparams, cX_std, cy, sconfigs, relabel=False)
        
    if clfparams['classifier'] == 'LinearSVC':
        svc = LinearSVC(random_state=0, dual=clfparams['dual'], multi_class='ovr', C=clfparams['C'])
    
    predicted, true = lsvc.do_cross_validation(svc, clfparams, cX_std, cy, data_identifier='')
    cmatrix, classes, _ = lsvc.get_confusion_matrix(predicted, true, clfparams)

    return cmatrix, classes, clfparams


def get_object_transforms(df_by_rois, roistats, sconfigs, is_gratings=False, metric='zscore'):

    responses_by_config = dict((roi, df_by_rois.get_group(roi).groupby('config')[metric].mean()) for roi in roistats['rois_visual'])

    transform_dict, object_transformations = util.get_transforms(sconfigs)
    transforms_tested = [k for k, v in object_transformations.items() if len(v) > 0]
    
    sconfigs_df = pd.DataFrame(sconfigs).T
    responses = []
    for roi, rdf in responses_by_config.items():
        responses.append(pd.concat([sconfigs_df, rdf, pd.Series(data=[roi for _ in range(rdf.shape[0])], index=rdf.index, name='roi')], axis=1))
    responses = pd.concat(responses, axis=0)    
   
    if is_gratings:
        df_columns = ['ori', 'roi', metric]
        transforms_tested = [t for t in transforms_tested if t != 'ori']
    else: 
        df_columns = ['object', 'roi', metric]
    df_columns.extend(transforms_tested)
    
    data = responses[df_columns]
    
    return data, transforms_tested


class SessionSummary():
    
    def __init__(self, optsE):
        self.rootdir = optsE.rootdir
        self.animalid = optsE.animalid
        self.session = optsE.session
        self.acquisition = optsE.acquisition
        self.data_type = optsE.data_type
        self.create_new = optsE.create_new
        self.nproc = int(optsE.nprocesses)
        self.traceid_dirs = get_data_sources(optsE)
        self.zproj = {'source': None, 'type': 'dff' if optsE.use_dff else 'mean', 'data': None, 'retinorun_name': None}
        if optsE.retino_run is None:
            self.zproj['retinorun_name'] = 'retino*'
        else:
            self.zproj['retinorun_name'] = optsE.retino_run
        self.retinotopy = {'source': None, 'traceid': optsE.retino_traceid, 'data': None}
        self.gratings = {'source': None, 'traceid': None, 'roistats': None, 'roidata': None, 'sconfigs': None}
        self.blobs = {'source': None, 'traceid': None, 'roistats': None, 'roidata': None, 'sconfigs': None}
        self.objects = {'source': None, 'traceid': None, 'roistats': None, 'roidata': None, 'sconfigs': None}
        self.data_identifier = None
        self.traceset = ''.join([tid for tid in [self.retinotopy['traceid'], self.gratings['traceid'], self.blobs['traceid'], self.objects['traceid']] if tid is not None])
    
        #self.get_data()


    def get_data(self):
        self.get_zproj_image()
        self.get_retinotopy()
        info_str = [self.animalid, self.session, self.acquisition, self.retinotopy['source'], self.retinotopy['traceid']]
       
        print "*********************************************"
        print "Getting data:"
        for k,t in self.traceid_dirs.items():
            print k, t

        if 'gratings' in self.traceid_dirs.keys():# is not None:
            self.get_gratings(metric='meanstim')
            info_str.extend([str(self.gratings['source']), ''.join(self.gratings['traceid'].split('_')[0])])

        if 'blobs' in self.traceid_dirs.keys():
            self.get_objects(object_type='blobs', metric='zscore')
            info_str.extend([str(self.blobs['source']), ''.join(self.blobs['traceid'].split('_')[0])])

        if 'objects' in self.traceid_dirs.keys():
            self.get_objects(object_type='objects', metric='zscore')
            info_str.extend([str(self.objects['source']), ''.join(self.objects['traceid'].split('_')[0])])

        print info_str
        self.data_identifier ='_'.join(info_str)
        
        # Update tmp_ss.pkl file:
#        acquisition_dir = os.path.join(self.rootdir, self.animalid, self.session, self.acquisition)
#        tmp_fpath = os.path.join(acquisition_dir, 'tmp_%s.pkl' % self.data_identifier)
#        if not os.path.exists(tmp_fpath):
#            shutil.move(os.path.join(acquisition_dir, 'tmp_ss.pkl'), tmp_fpath)   

    def plot_summary(self, ignore_null=False, selective=True):
        
        if 'blobs' in self.traceid_dirs.keys() and 'objects' in self.traceid_dirs.keys(): 
            ntransforms_plot = max([2, len(self.blobs['transforms_tested']), len(self.objects['transforms_tested']) ])
            nrows = 4

        elif 'blobs' in self.traceid_dirs.keys() or 'objects' in self.traceid_dirs.keys():
            if 'blobs' in self.traceid_dirs.keys():
                ntransforms_plot = max([2, len(self.blobs['transforms_tested'])])
            else:
                ntransforms_plot = max([2, len(self.objects['transforms_tested'])])

            if 'gratings' not in self.traceid_dirs.keys():
                nrows = 2
            else:
                nrows = 3
             
        elif 'gratings' in self.traceid_dirs.keys():
            ntransforms_plot = 3
            nrows = 2
        else:
            # Only have retino and FOV:
            ntransforms_plot = 3
            nrows = 1

        fig = pl.figure(figsize=(40,10*nrows))
        print "++++++++++GRID SPEC+++++++++++++++++"
        print "Ncols: %i, Nrows: %i" % (ntransforms_plot+1, nrows)

        self.fig = fig
        ax1 = pl.subplot2grid((nrows, ntransforms_plot+1), (0, 0), colspan=1) 
        self.plot_zproj_image(ax=ax1) #fig.axes, aix=0)

        ax2 = pl.subplot2grid((nrows, ntransforms_plot+1), (0, 1), colspan=2)
        self.plot_retinotopy_to_screen(ax=ax2) #fig.axes, aix=1)
        
        ax3 = pl.subplot2grid((nrows, ntransforms_plot+1), (0, 3), colspan=1)
        self.plot_estimated_RF_size(ax=ax3, ignore_null=ignore_null)  #fig.axes, aix=2, ignore_null=ignore_null)
        if 'gratings' in self.traceid_dirs.keys():
            ax4 = pl.subplot2grid((nrows, ntransforms_plot+1), (1, 0), colspan=1)
            self.plot_responsivity_gratings(ax=ax4)

            ax5 = pl.subplot2grid((nrows, ntransforms_plot+1), (1, 1), colspan=1)
            self.plot_OSI_histogram(ax=ax5)

            ax6 = pl.subplot2grid((nrows, ntransforms_plot+1), (1, 2), colspan=1)
            self.plot_confusion_gratings(ax=ax6)

            obj_start_row = 2
        else:
            obj_start_row = 1

        if 'blobs' in self.traceid_dirs.keys():
            ax7 = pl.subplot2grid((nrows, ntransforms_plot+1), (obj_start_row, 0), colspan=1)
            self.plot_responsivity_objects(ax=ax7, object_type='blobs')
            ntransforms_blobs = len(self.blobs['transforms_tested'])
            ax8 = pl.subplot2grid((nrows, ntransforms_plot+1), (obj_start_row, 1), colspan=1) 
            self.plot_transforms_objects(ax=ax8, selective=selective, object_type='blobs', nrows=nrows, ncols=ntransforms_plot+1, row_start=obj_start_row, col_start=1)
            obj_start_row += 1

        if 'objects' in self.traceid_dirs.keys():
            ax9 = pl.subplot2grid((ntransforms_plot+1, nrows), (obj_start_row, 0), colspan=1)
            self.plot_responsivity_objects(ax=ax9, object_type='objects')
            ntransforms_objects = len(self.objects['transforms_tested'])
            ax10 = pl.subplot2grid((ntransforms_plot+1, nrows), (obj_start_row, 1), colspan=ntransforms_objects) 
            self.plot_transforms_objects(ax=ax10, selective=selective, object_type='objects', nrows=nrows, ncols=ntransforms_plot+1, row_start=obj_start_row, col_start=1)
           


    def load_sessionsummary_step(self, key='', traceset=''):
        acquisition_dir = os.path.join(self.rootdir, self.animalid, self.session, self.acquisition)
        if self.data_identifier is None:
            print "Loading step: %s" % key
            tmp_fpath = os.path.join(acquisition_dir, 'tmp_ss_%s.pkl' % traceset)
        else:
            tmp_fpath = os.path.join(acquisition_dir, 'tmp_%s.pkl' % self.data_identifier)

        if not os.path.exists(tmp_fpath):
            print"No temp file exists. Redo step: %s" % key
            return None
        else:
            with open(tmp_fpath, 'rb') as f: tmpdict = pkl.load(f)
            if key in tmpdict.keys():
                return tmpdict[key]
            else:
                print "Specified key %s does not exist. Create new." % key
                return None
            
    def save_sessionsummary_step(self, key='', val=None, traceset=''):
        acquisition_dir = os.path.join(self.rootdir, self.animalid, self.session, self.acquisition)
        if self.data_identifier is None:
            tmp_fpath = os.path.join(acquisition_dir, 'tmp_ss_%s.pkl' % traceset)
        else:
            tmp_fpath = os.path.join(acquisition_dir, 'tmp_%s.pkl' % self.data_identifier)

        if not os.path.exists(tmp_fpath):
            print "No temp file exists yet. Creating it!"
            tmpdict = {}
            tmpdict.update({key: val})
            with open(tmp_fpath, 'wb') as f: pkl.dump(tmpdict, f, protocol=pkl.HIGHEST_PROTOCOL)
        else:
            with open(tmp_fpath, 'rb') as f: tmpdict = pkl.load(f)
            tmpdict.update({key: val})
            with open(tmp_fpath, 'wb') as f: pkl.dump(tmpdict, f, protocol=pkl.HIGHEST_PROTOCOL)

    
    def get_zproj_image(self):
        zproj = None
        
        if not self.create_new:
            zproj = self.load_sessionsummary_step(key='zproj', traceset=self.traceset)
        if zproj is not None:
            for k in zproj:
                if k not in self.zproj.keys() or self.zproj[k] is None:
                    self.zproj[k] = zproj[k]
                    
        else:
            acquisition_dir = os.path.join(self.rootdir, self.animalid, self.session, self.acquisition)
            
            self.zproj['source'] = os.path.split(glob.glob(os.path.join(acquisition_dir, '%s' % self.zproj['retinorun_name']))[0])[-1] 
    
            if self.zproj['type'] == 'dff':
                self.zproj['data'] = create_activity_map(acquisition_dir, self.zproj['source'], rootdir=self.rootdir)
            else:
                self.zproj['data'] = load_traceid_zproj(self.traceid_dirs['gratings'], rootdir=self.rootdir)
            
            # Save this step for now:
            self.save_sessionsummary_step(key='zproj', val=self.zproj, traceset=self.traceset)
        
    def get_retinotopy(self, fitness_thr=0.5, size_thr=0.1):
        
        retino = None
        if not self.create_new:
            retino = self.load_sessionsummary_step(key='retinotopy', traceset=self.traceset)
        if retino is not None:
            for k in retino:
                if k not in self.retinotopy.keys() or self.retinotopy[k] is None:
                    self.retinotopy[k] = retino[k]
                    
        else:
            acquisition_dir = os.path.join(self.rootdir, self.animalid, self.session, self.acquisition)
            if self.retinotopy['traceid'] is None:
                # just take the first found ROI analysis
                traceid = 'analysis*'
            else:
                traceid = '%s*' % self.retinotopy['traceid']
                
            retinovis_fpath = glob.glob(os.path.join(self.rootdir, self.animalid, self.session, self.acquisition, 
                                                 '%s' % self.zproj['retinorun_name'], 'retino_analysis', traceid, 'visualization', '*.png'))[0]
            
            retino_run = os.path.split(retinovis_fpath.split('/retino_analysis')[0])[1]
            retino_traceid = retinovis_fpath.split('/retino_analysis')[1].split('/')[1]
            
            ROIs, retinoid = RF.get_RF_size_estimates(acquisition_dir, 
                                     fitness_thr=fitness_thr, 
                                     size_thr=size_thr, 
                                     analysis_id=retino_traceid, 
                                     retino_run=retino_run)
            
            self.retinotopy['source'] = retino_run
            self.retinotopy['data'] = ROIs
            self.retinotopy['traceid'] = retino_traceid
            self.retinotopy['fitness_thr'] = fitness_thr
            self.retinotopy['size_thr'] = size_thr
            
            # Save this step for now:
            self.save_sessionsummary_step(key='retinotopy', val=self.retinotopy, traceset=self.traceset)
        
    def get_gratings(self, metric='meanstim'):
        gratings = None
        if not self.create_new:
            gratings = self.load_sessionsummary_step(key='gratings', traceset=self.traceset)
        if gratings is not None:
            for k in gratings:
                if k not in self.gratings.keys() or self.gratings[k] is None:
                    self.gratings[k] = gratings[k]
                    
        else:
            # GRATINGS:
            gratings_traceid = os.path.split(self.traceid_dirs['gratings'])[-1]
            gratings_run = os.path.split(self.traceid_dirs['gratings'].split('/traces/')[0])[-1] #[0])[-1]
            
            # Load data array:
            data_fpath = os.path.join(self.traceid_dirs['gratings'], 'data_arrays', 'datasets.npz')
            gratings_dataset = np.load(data_fpath)
            
            # Get sorted ROIs:
            gratings_roistats = get_roi_stats(self.rootdir, self.animalid, self.session, self.acquisition, 
                                                  gratings_run, gratings_traceid, create_new=self.create_new, nproc=self.nproc)
                                                  #gratings_traceid.split('_')[0], create_new=optsE.create_new)
            
            # Group data by ROIs:
            gratings_roidata, gratings_labels_df, gratings_sconfigs = get_data_and_labels(gratings_dataset, data_type=self.data_type)
            gratings_df_by_rois = resp.group_roidata_stimresponse(gratings_roidata, gratings_labels_df)
            #nrois_total = gratings_roidata.shape[-1]
            oris = np.unique([v['ori'] for k,v in gratings_sconfigs.items()])
            if max(oris) > 180:
                selectivity = osi.get_OSI_DSI(gratings_df_by_rois, gratings_sconfigs, 
                                              roi_list=gratings_roistats['rois_visual'], 
                                              metric=metric)
            else:
                selectivity = {}
                
            self.gratings['source'] = gratings_run 
            self.gratings['traceid'] = gratings_traceid
            self.gratings['data_fpath'] = data_fpath
            self.gratings['roistats'] = gratings_roistats
            self.gratings['roidata'] = gratings_df_by_rois
            self.gratings['sconfigs'] = gratings_sconfigs
            self.gratings['selectivity'] = selectivity
            
            cmatrix, classes, clfparams = run_gratings_classifier(gratings_dataset, gratings_sconfigs, gratings_traceid)
    
            self.gratings['metric'] = metric
            self.gratings['SVC'] = {'cmatrix': cmatrix, 'classes': classes, 'clfparams': clfparams}
            
            # Save this step for now:
            self.save_sessionsummary_step(key='gratings', val=self.gratings, traceset=self.traceset)
        
        
    def get_objects(self, object_type='blobs', metric='zscore'):
        blobs = None
        if not self.create_new:
            blobs = self.load_sessionsummary_step(key=object_type, traceset=self.traceset)
        if blobs is not None:
            for k in blobs:
                if object_type == 'blobs':
                    if k not in self.blobs.keys() or self.blobs[k] is None:
                        self.blobs[k] = blobs[k]
                else:
                    if k not in self.objects.keys() or self.objects[k] is None:
                        self.objects[k] = blobs[k]
                   
        else:
                
            blobs_traceid = os.path.split(self.traceid_dirs[object_type])[-1]
            blobs_run = os.path.split(self.traceid_dirs[object_type].split('/traces/')[0])[-1] #[0])[-1]
    
            # Load data array:
            data_fpath = os.path.join(self.traceid_dirs[object_type], 'data_arrays', 'datasets.npz')
            blobs_dataset = np.load(data_fpath)
            
            # Get sorted ROIs:
            blobs_roistats = get_roi_stats(self.rootdir, self.animalid, self.session, self.acquisition, 
                                           blobs_run, blobs_traceid, create_new=self.create_new, nproc=self.nproc) #blobs_traceid.split('_')[0])
            
            # Group data by ROIs:
            blobs_roidata, blobs_labels_df, blobs_sconfigs = get_data_and_labels(blobs_dataset, data_type=self.data_type)
            blobs_df_by_rois = resp.group_roidata_stimresponse(blobs_roidata, blobs_labels_df)
            data, transforms_tested = get_object_transforms(blobs_df_by_rois, blobs_roistats, blobs_sconfigs, metric=metric, is_gratings=False)
         
            object_dict = {'source': blobs_run,
                           'traceid': blobs_traceid,
                           'data_fpath': data_fpath,
                           'roistats': blobs_roistats,
                           'roidata': blobs_df_by_rois,
                           'sconfigs': blobs_sconfigs,
                           'transforms': data,
                           'transforms_tested': transforms_tested,
                           'metric': metric}
 
            if object_type == 'blobs': 
                self.blobs = object_dict
            else:
                self.objects = object_dict               

            # Save this step for now:
            self.save_sessionsummary_step(key=object_type, val=object_dict, traceset=self.traceset)
        
        
    def plot_zproj_image(self, ax=None): #axes_flat=None, aix=0):
        # SUBPLOT 0:  Mean / zproj image
        # -----------------------------------------------------------------------------
        if ax is None: # axes_flat is None:
            fig, ax = pl.subplots()
            #axes_flat = fig.axes
            
        im = ax.imshow(self.zproj['data'], cmap='gray')
        ax.axis('off')
        ax.set_title('dF/F map')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.5)
        cb = pl.colorbar(im, cax=cax, orientation='vertical', ticklocation='right')
        cax.yaxis.set_ticks_position('right')
        cax.yaxis.set_label_position('right')
       
        
    def plot_retinotopy_to_screen(self, ax=None):
        if ax is None: 
            fig, ax = pl.subplots()
            
        # SUBPLOT 1:  Retinotopy:
        # -------------------------------------------------------------------------
        acquisition_dir = os.path.join(self.rootdir, self.animalid, self.session, self.acquisition)
        RF.plot_RF_position_and_size(self.retinotopy['data'], acquisition_dir, 
                                         self.retinotopy['source'], self.retinotopy['traceid'], 
                                         ax=ax) 
        
        
    def plot_estimated_RF_size(self, ax=None, ignore_null=False):
        if ax is None:
            fig, ax = pl.subplots()
            
        # SUBPLOT 2:  Estimated RF sizes:
        # ------------------------------------------------------------------------  
        cond0_name = self.retinotopy['data'][0].conditions[0].name
        cond1_name = self.retinotopy['data'][0].conditions[1].name

        nrois = len(self.retinotopy['data'])
        fit_rois = [ri for ri in range(nrois) if self.retinotopy['data'][ri].conditions[0].RF_degrees > 0 
                                 and self.retinotopy['data'][ri].conditions[1].RF_degrees > 0]
        if ignore_null:
            plot_rois = fit_rois
        else:
            plot_rois = range(nrois)
        
        el_rfs = [roi.conditions[0].RF_degrees for ri, roi in enumerate(self.retinotopy['data']) if ri in plot_rois]
        az_rfs = [roi.conditions[1].RF_degrees for ri, roi in enumerate(self.retinotopy['data']) if ri in plot_rois]
        
        n_badfits = nrois - len(fit_rois)
        
        print "RF FIT: fit %i rois." % len(fit_rois)
        if len(fit_rois) > 0: 
            sns.distplot(az_rfs, kde=False, bins=len(plot_rois), ax=ax, label=cond0_name, color='orange')
            sns.distplot(el_rfs, kde=False, bins=len(plot_rois), ax=ax, label=cond1_name, color='cornflowerblue')
        ax.legend()
        ax.set_title('distN of estimated RF sizes')
        
        texth = ax.get_ylim()[-1] + 2
        textw = ax.get_xlim()[0] + .1
        ax.text(textw, texth-1, "N no fit: %i" % n_badfits, fontdict=dict(color='k', size=10))
        
    def plot_responsivity_gratings(self, ax=None):
        if ax is None:
            fig, ax = pl.subplots()
            
        # SUBPLOT 3:  Histogram of visual vs. selective ROIs (use zscores)
        # ------------------------------------------------------------------------
        hist_roi_stats(self.gratings['roidata'], self.gratings['roistats'], ax=ax)
        ax.set_title('gratings', fontsize=24)
        ax.set_xlabel('zscore', fontsize=24)
#        
        #%
        
        # SUBPLOT 4:  DistN of preferred orientations:
        # -------------------------------------------------------------------------
        
    def plot_OSI_histogram(self, ax=None): #axes_flat=None, aix=0):
        if ax is None:
            fig, ax = pl.subplots()
            #axes_flat = fig.axes
            
        cmap = 'hls'
        noris = len(np.unique([v['ori'] for k, v in self.gratings['sconfigs'].items()]))
        colorvals = sns.color_palette(cmap, noris) # len(gratings_sconfigs))
        if len(self.gratings['selectivity'].keys()) > 0:
            osi.hist_preferred_oris(self.gratings['selectivity'], colorvals, metric=self.gratings['metric'], save_and_close=False, ax=ax) 
        ax.set_title('orientation selectivity', fontsize=24)
        
        
    def plot_confusion_gratings(self, ax=None):
        if ax is None:
            fig, ax = pl.subplots()
            
        # SUBPLOT 5:  Decoding performance for linear classifier on orientations:
        # -------------------------------------------------------------------------
    
        lsvc.plot_confusion_matrix(self.gratings['SVC']['cmatrix'], classes=self.gratings['SVC']['classes'], ax=ax, normalize=True)
    
    def plot_responsivity_objects(self, ax=None, object_type='blobs'): 
        if ax is None:
            fig, ax = pl.subplots()

        # SUBPLOT 6:  Complex stimuli...
        # -------------------------------------------------------------------------
        if object_type == 'blobs': #in self.traceid_dirs.keys():
            hist_roi_stats(self.blobs['roidata'], self.blobs['roistats'], ax=ax)
            ax.set_title('%s' % self.blobs['source'], fontsize=24)
            ax.set_xlabel('zscore', fontsize=24)
        else:
            hist_roi_stats(self.objects['roidata'], self.objects['roistats'], ax=ax)
            ax.set_title('%s' % self.objects['source'], fontsize=24)

    def plot_transforms_objects(self, ax=None, selective=True, object_type='blobs', row_start=0, col_start=0, ncols=1, nrows=1):
        if ax is None:
            fig, ax = pl.subplots()
        if object_type == 'blobs':
            plotdata = self.blobs
        elif object_type == 'objects':
            plotdata = self.objects
       
        ylabel = plotdata['metric']
        metric = plotdata['metric']
        transforms_tested = plotdata['transforms_tested']
        object_list = plotdata['transforms']['object'].unique()
        # Colors = cells
        if selective:
            rois_to_plot = plotdata['roistats']['rois_selective'][0:10]
            if len(rois_to_plot) == 0:
                selective = False
        if not selective:
            rois_to_plot = plotdata['roistats']['rois_visual'][0:10]
        rois = plotdata['transforms'].groupby('roi')
      
        nrois_plot = len(rois_to_plot) 
        colors = sns.color_palette('husl', nrois_plot)
        
        # Shapes = objects
        nobjects = len(object_list)  #len(responses['object'].unique())
        markers = ['o', 'P', '*', 'X', 's', 'd', 'p', 'H', '1', '2', '3', '4','<','>','_']
        marker_kws = {'markersize': 15, 'linewidth': 2, 'alpha': 0.3}
        print "Plotting %i rois" % nrois_plot

        curr_col = col_start
        for trans_ix, transform in enumerate(transforms_tested):
            plot_list = []
            for roi, df in rois:
                if roi not in rois_to_plot:
                    continue
                
                df2 = df.pivot_table(index='object', columns=transform, values=metric)
                plot_list.append(df2)
                
            data = pd.concat(plot_list, axis=0)
            print "***(%i, %i) - row: %i, col: %i" % (ncols, nrows, row_start, curr_col)

            ax0 = pl.subplot2grid((nrows, ncols), (row_start, curr_col), colspan=1) 
            for ridx, r in enumerate(np.arange(0, data.shape[0], nobjects)):
                for object_ix in range(nobjects):
                    ax0.plot(data.iloc[r+object_ix, :], color=colors[ridx], marker=markers[object_ix], **marker_kws) #'.-')
            ax0.set_xticks(data.keys().tolist())
            ax0.set_ylabel(ylabel) #self.blobs['metric'])
            ax0.set_title(transform, fontsize=24)
            curr_col += 1            
            sns.despine(ax=ax0, trim=True, offset=4)
        
        legend_objects = []
        object_names = data.iloc[0:nobjects].index.tolist()
        for object_ix in range(nobjects):
            legend_objects.append(Line2D([0], [0], color='w', markerfacecolor='k', 
                                         marker=markers[object_ix], label=object_names[object_ix], 
                                         linewidth=2, markersize=15))
            
        ax0.legend(handles=legend_objects, loc=2, bbox_to_anchor=(0.0, 0.99), ncol=nobjects) # loc='upper right')
        
        
        
#%%

def extract_options(options):

    def comma_sep_list(option, opt, value, parser):
        setattr(parser.values, option.dest, value.split(','))

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
    parser.add_option('-n', '--nproc', action='store', dest='nprocesses', default=4, help="N processes if running in par (default=4)")
    parser.add_option('--new', action='store_true', dest='create_new', default=False, help="set to run anew")
    parser.add_option('--redo', action='store_true', dest='redo', default=False, help="set to (re-)create SessionSummary object")
    parser.add_option('--mean', action='store_false', dest='use_dff', default=True, help="set to use MEAN image for zproj instead of df/f (default)")
    parser.add_option('--ignore-null-RF', action='store_true', dest='ignore_null_RF', default=False, help="set to plot all ROIs in RF size historgram (even ones with RF 0 due to poor fit")
   
    # Run specific info:
    #parser.add_option('-g', '--gratings-traceid', dest='gratings_traceid_list', default=[], action='append', nargs=1, help="traceid for GRATINGS [default: []]")
    parser.add_option('-g', '--gratings-traceid', dest='gratings_traceid_list', default=[], type='string', action='callback', callback=comma_sep_list, help="traceids for GRATINGS [default: []]")

    parser.add_option('-G', '--gratings-run', dest='gratings_run_list', default=[], type='string', action='callback', callback=comma_sep_list, help='list of gratings run IDs [default: []')
    parser.add_option('-r', '--retino', dest='retino_traceid', default=None, action='store', help='analysisid for RETINO [default assumes only 1 roi-based analysis]')
    parser.add_option('-b', '--blobs-traceid', dest='blobs_traceid_list', default=[], type='string', action='callback', callback=comma_sep_list, help='list of blob traceids [default: []')
    parser.add_option('-B', '--blobs-run', dest='blobs_run_list', default=[], type='string', action='callback', callback=comma_sep_list, help='list of blob run IDs [default: []')
    parser.add_option('-o', '--objects-traceid', dest='objects_traceid_list', default=[], type='string', action='callback', callback=comma_sep_list, help='list of RW object traceids [default: []')
    parser.add_option('-O', '--objects-run', dest='objects_run_list', default=[], type='string', action='callback', callback=comma_sep_list, help='list of RW object run IDs [default: []')
    parser.add_option('-R', '--retino-run', dest='retino_run', default=None, action='store', help='Specific retino_run to use')
  
    #parser.add_option('-t', '--traceid', dest='traceid', default=None, action='store', help="datestr YYYYMMDD_HH_mm_SS")
     
    (options, args) = parser.parse_args(options)
    if options.slurm is True and '/n/coxfs01' not in options.rootdir:
        options.rootdir = '/n/coxfs01/2p-data'
    return options
options = ['-D', '/n/coxfs01/2p-data', '-i', 'CE077', '-S', '20180612', '-A', 'FOV1_zoom1x',
           '-r', 'analysis001',
           '-b', 'traces001',
           '-n', 4,
           '--redo']

def load_session_summary(optsE, redo=False):
    acquisition_dir = os.path.join(optsE.rootdir, optsE.animalid, optsE.session, optsE.acquisition)
    #%
    # First check if saved session summary info exists:
    
    ss_fpaths = glob.glob(os.path.join(acquisition_dir, 'session_summary*.pkl'))
    print "Found SS:", ss_fpaths
    if len(ss_fpaths) > 0 and optsE.create_new is False and redo is False:
        session_summary_fpath = None
        while session_summary_fpath is None:
            if len(ss_fpaths) > 1:
                print "More than 1 SS fpath found:"
                for si, spath in enumerate(ss_fpaths):
                    print si, spath
                selected = raw_input("Select IDX to plot: ")
                if selected == '':
                    redo = True
                    session_summary_fpath = 'NA'
                else:     
                    session_summary_fpath = ss_fpaths[int(selected)]
            else:
                session_summary_fpath = ss_fpaths[0]
        if session_summary_fpath == 'NA':
            pass
        else:
            with open(session_summary_fpath, 'rb') as f:
                S = pkl.load(f)

    if optsE.create_new or redo:
       print "*** Creating new SessionSummary() object!"
       S = SessionSummary(optsE)
       #datestr = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
       try:
           S.get_data()
       except Exception as e:
           traceback.print_exc()
       finally:
           with open(os.path.join(acquisition_dir, 'session_summary_%s.pkl' % S.data_identifier), 'wb') as f:
               pkl.dump(S, f, protocol=pkl.HIGHEST_PROTOCOL)

    return S

def plot_session_summary(options):
    optsE = extract_options(options)
    print "Getting session summary..."
    S = load_session_summary(optsE, redo=optsE.redo)
    if optsE.rootdir != S.rootdir:
        S.rootdir = optsE.rootdir 
    #data_identifier ='_'.join([S.animalid, S.session, S.acquisition, S.retinotopy['traceid'], S.gratings['traceid'], S.blobs['traceid']])

    print "PLOTTING..."
    S.plot_summary(ignore_null=optsE.ignore_null_RF, selective=True)
    label_figure(S.fig, S.data_identifier)
    
    figname = 'acquisition_summary_%s.png' % (S.data_identifier)
    
    pl.savefig(os.path.join(S.rootdir, S.animalid, S.session, figname))
    pl.close()
    

#%%
    

def main(options):
    plot_session_summary(options)



if __name__ == '__main__':
    main(sys.argv[1:])
    
    
