#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 16:37:13 2019

@author: julianarhee
"""

#import matplotlib as mpl
#mpl.use('Agg')
import os
import h5py
import json
import traceback
#import re
#import sys
#import datetime
#import optparse
import pprint
import copy

import cPickle as pkl
#import tifffile as tf
import pylab as pl
import numpy as np
#from scipy import ndimage
#import cv2
import glob
#from scipy.optimize import curve_fit
#import seaborn as sns
#from pipeline.python.retinotopy import visualize_rois as vis
#from matplotlib.patches import Ellipse
#from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy as sp
import pandas as pd


from pipeline.python.utils import natural_keys, label_figure, replace_root
#from matplotlib.patches import Ellipse, Rectangle

pp = pprint.PrettyPrinter(indent=4)
#from scipy.signal import argrelextrema

from pipeline.python.retinotopy import do_retinotopy_analysis as ra

#%%

# -----------------------------------------------------------------------------
# Data processing funcs 
# -----------------------------------------------------------------------------

# preprocessing ---------------

def load_retino_traces(retino_dpath, scaninfo, trace_type='corrected', temporal_ds=None, frame_rate=44.65):
    '''
    Loads ./traces/extracted_traces.h5 (contains data for each tif file).
    Pre-processes raw extracted traces by adding back in neuropil offsets and F0 offset from drift correction.
    Averages traces for each condition. Downsamples final array.
    '''
    frame_rate = scaninfo['stimulus']['frame_rate']
    stim_freq = scaninfo['stimulus']['stim_freq']
    trials_by_cond = scaninfo['trials']

    traces = {}
    try:
        tfile = h5py.File(retino_dpath, 'r')
        for condition, trialnums in trials_by_cond.items():
            print("... loading cond: %s" % condition)
            dlist = tuple([process_data(tfile, trialnum, trace_type=trace_type, frame_rate=frame_rate, stim_freq=stim_freq) for trialnum in trialnums])
            dfcat = pd.concat(dlist)
            df_rowix = dfcat.groupby(dfcat.index)
            meandf = df_rowix.mean()
            if temporal_ds is not None:
                meandf = downsample_array(meandf, temporal_ds=temporal_ds)
            traces[condition] = meandf
    except Exception as e:
        traceback.print_exc()
    finally:
        tfile.close()
        
    return traces

def process_data(tfile, trialnum, trace_type='corrected', add_offset=True,
                frame_rate=44.65, stim_freq=0.13):
    #print(tfile['File001'].keys())
    if trace_type != 'neuropil' and add_offset:
        # Get raw soma traces and raw neuropil -- add neuropil offset to soma traces
        soma = pd.DataFrame(tfile['File%03d' % int(trialnum)][trace_type][:].T)
        neuropil = pd.DataFrame(tfile['File%03d' % int(trialnum)]['neuropil'][:].T)
        np_offset = neuropil.mean(axis=0) #neuropil.mean().mean()
        xd = soma.subtract(neuropil) + np_offset
        del neuropil
        del soma
    else:
        xd = pd.DataFrame(tfile['File%03d' % int(trialnum)][trace_type][:].T)
    
    f0 = xd.mean().mean()
    drift_corrected = detrend_array(xd, frame_rate=frame_rate, stim_freq=stim_freq)
    xdata = drift_corrected + f0
    #if temporal_ds is not None:
    #    xdata = downsample_array(xdata, temporal_ds=temporal_ds)
    
    return xdata

def subtract_rolling_mean(trace, windowsz):
    #print(trace.shape)
    tmp1 = np.concatenate((np.ones(windowsz)*trace.values[0], trace, np.ones(windowsz)*trace.values[-1]),0)
    rolling_mean = np.convolve(tmp1, np.ones(windowsz)/windowsz, 'same')
    rolling_mean=rolling_mean[windowsz:-windowsz]
    return np.subtract(trace, rolling_mean)

def detrend_array(roi_trace, frame_rate=44.65, stim_freq=0.24):
    #print('Removing rolling mean from traces...')
    windowsz = int(np.ceil((np.true_divide(1,stim_freq)*3)*frame_rate))
    detrend_roi_trace = roi_trace.apply(subtract_rolling_mean, args=(windowsz,), axis=0)
    return detrend_roi_trace #pd.DataFrame(detrend_roi_trace)
        
def temporal_downsample(trace, windowsz):
    tmp1=np.concatenate((np.ones(windowsz)*trace.values[0], trace, np.ones(windowsz)*trace.values[-1]),0)
    tmp2=np.convolve(tmp1, np.ones(windowsz)/windowsz, 'same')
    tmp2=tmp2[windowsz:-windowsz]
    return tmp2

        
def downsample_array(roi_trace, temporal_ds=5):
    print('Performing temporal smoothing on traces...')
    windowsz = int(temporal_ds)
    smooth_roi_trace = roi_trace.apply(temporal_downsample, args=(windowsz,), axis=0)
    return smooth_roi_trace
    

# averaging -------------------

def get_condition_averaged_traces(RID, retinoid_dir, mwinfo, runinfo, tiff_fpaths, create_new=False):

    # Set output dir:
    output_dir = os.path.join(retinoid_dir,'traces')
    if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    print output_dir

    avg_trace_fpath = os.path.join(output_dir, 'averaged_roi_traces.pkl')
    redo=False
    if os.path.exists(avg_trace_fpath) and create_new is False:
        try:
            with open(avg_trace_fpath, 'rb') as f:
                traces = pkl.load(f)
        except Exception as e:
            redo=True
    if create_new or redo:
        acquisition_dir = os.path.split(retinoid_dir.split('/retino_analysis')[0])[0]
        session_dir = os.path.split(acquisition_dir)[0]
        masks = load_roi_masks(session_dir, RID)

        # Block reduce masks to match downsampled tiffs:
        dsample = RID['PARAMS']['downsample_factor']
        if dsample is None:
            dsample = 1
        else:
            dsample = int(dsample)
        masks = ra.block_mean_stack(masks, dsample, along_axis=0)
        print masks.shape
        
        # Combine reps of the same condition.
        # Reshape masks and averaged tiff stack, extract ROI traces
        traces = average_retino_traces(RID, mwinfo, runinfo, tiff_fpaths, masks, output_dir=output_dir)
        
    return traces

def average_retino_traces(RID, mwinfo, runinfo, tiff_fpaths, masks, output_dir='/tmp'):
    
    rep_list = [(k, v['stimuli']['stimulus']) for k,v in mwinfo.items()]
    unique_conditions = np.unique([rep[1] for rep in rep_list])
    conditions = dict((cond, [int(run) for run,config in rep_list if config==cond]) for cond in unique_conditions)
    print("CONDITIONS:", conditions)
    
    rtraces = {}

    # First check if extracted_traces file exists:
    traces_fpath = glob.glob(os.path.join(output_dir, 'extracted_traces*.h5'))
    extract_from_stack = False
    try:
        assert len(traces_fpath) == 1, "*** unable to find unique extracted_traces.h5 in dir:\n%s" % output_dir
        traces_fpath = traces_fpath[0]
        print("... Loading extracted traces: %s" % traces_fpath)
        extracted = h5py.File(traces_fpath, 'r')
        print("Found %i files of extracted traces." % len(extracted.keys()))
        for curr_cond in conditions.keys():
            curr_tstack = np.array([extracted['File%03d' % int(rep)]['corrected'][:] for rep in conditions[curr_cond]])
            print("... cond: %s (stack size: %s)" % (curr_cond, str(curr_tstack.shape)))
            rtraces[curr_cond] = np.mean(curr_tstack, axis=0)
            print rtraces[curr_cond].shape

    except Exception as e:
        print e
        print("Extracting ROI traces from tiff stacks...")
        extract_from_stack = True
    finally:
        extracted.close()

    if extract_from_stack: 
        cstack = get_averaged_condition_stack(conditions, tiff_fpaths, RID)

        for curr_cond in cstack.keys():
            roi_traces = apply_masks_to_tifs(masks, cstack[curr_cond])
            #print roi_traces.shape
            rtraces[curr_cond] = roi_traces
      
 
    # Smooth roi traces:
    traceinfo = dict((cond, dict()) for cond in rtraces.keys())
    for curr_cond in rtraces.keys():
        # get some info from paradigm and run file
        stack_info = dict()
        stack_info['stimulus'] = curr_cond
        stack_info['stimfreq'] = np.unique([v['stimuli']['scale'] for k,v in mwinfo.items() if v['stimuli']['stimulus']==curr_cond])[0]
        stack_info['frame_rate'] = runinfo['frame_rate']
        stack_info['n_reps'] = len(conditions[curr_cond])
        pp.pprint(stack_info)

        traces = ra.process_array(rtraces[curr_cond], RID, stack_info)

        traceinfo[curr_cond]['traces'] = traces
        traceinfo[curr_cond]['info'] = stack_info

    traces = {'mwinfo': mwinfo,
             'conditions': conditions,
             'source_tifs': tiff_fpaths,
             'RETINOID': RID,
              'masks': masks,
              'traces': traceinfo
             }
    avg_trace_fpath = os.path.join(output_dir, 'averaged_roi_traces.pkl')
    with open(avg_trace_fpath, 'wb') as f: pkl.dump(traces, f, protocol=pkl.HIGHEST_PROTOCOL)
    print "Saved processed ROI traces to:\n%s\n" % avg_trace_fpath

    return traces



# Masks
def load_retinoanalysis(run_dir, traceid):
    run = os.path.split(run_dir)[-1]
    trace_dir = os.path.join(run_dir, 'retino_analysis')
    tracedict_path = os.path.join(trace_dir, 'analysisids_%s.json' % run)
    with open(tracedict_path, 'r') as f:
        tracedict = json.load(f)

    if 'traces' in traceid:
        fovdir = os.path.split(run_dir)[0]
        tmp_tdictpath = glob.glob(os.path.join(fovdir, '*run*', 'traces', 'traceids*.json'))[0]
        with open(tmp_tdictpath, 'r') as f:
            tmptids = json.load(f)
        roi_id = tmptids[traceid]['PARAMS']['roi_id']
        analysis_id = [t for t, v in tracedict.items() if v['PARAMS']['roi_type']=='manual2D_circle' and v['PARAMS']['roi_id'] == roi_id][0]
        print("Corresponding ANALYSIS ID (for %s with %s) is: %s" % (traceid, roi_id, analysis_id))

    else:
        analysis_id = traceid 
    TID = tracedict[analysis_id]
    pp.pprint(TID)
    return TID

 
def load_roi_masks(session_dir, RID):
    assert RID['PARAMS']['roi_type'] != 'pixels', "ROI type for analysis should not be pixels. This is: %s" % RID['PARAMS']['roi_type']
    print 'Getting masks'
    # Load ROI set specified in analysis param set:
    roidict_fpath = glob.glob(os.path.join(session_dir, 'ROIs', 'rids_*.json'))[0]
    with open(roidict_fpath, 'r') as f: roidict = json.load(f)

    roi_dir = roidict[RID['PARAMS']['roi_id']]['DST']
    session = os.path.split(session_dir)[-1]
    animalid = os.path.split(os.path.split(session_dir)[0])[-1]
    rootdir = os.path.split(os.path.split(session_dir)[0])[0]
    
    if rootdir not in roi_dir:
        roi_dir = replace_root(roi_dir, rootdir, animalid, session)
    mask_fpath = os.path.join(roi_dir, 'masks.hdf5')
    maskfile = h5py.File(mask_fpath,  'r')#read
    masks = maskfile[maskfile.keys()[0]]['masks']['Slice01']
    print masks.shape
    
    return masks

def get_averaged_condition_stack(conditions, tiff_fpaths, RID):
    cstack = {}
    condition_list = conditions.keys()
    curr_cond = 'right'
    for curr_cond in condition_list:
        curr_tiff_fpaths = [tiff_fpaths[int(i)-1] for i in conditions[curr_cond]]
        for tidx, tiff_fpath in enumerate(curr_tiff_fpaths):    
            print "Loading: ", tiff_fpath
            tiff_stack = ra.get_processed_stack(tiff_fpath, RID)
            szx, szy, nframes = tiff_stack.shape
            #print szx, szy, nframes
            if tidx == 0:
                # initiate stack
                stack = np.empty(tiff_stack.shape, dtype=tiff_stack.dtype)
            stack =  stack + tiff_stack

        # Get stack average:
        cstack[curr_cond] = stack / len(curr_tiff_fpaths)

    return cstack

def apply_masks_to_tifs(masks, stack):
    szx, szy, nframes = stack.shape
    nrois = masks.shape[0]
    maskr = np.reshape(masks, (nrois, szx*szy))
    stackr = np.reshape(stack, (szx*szy, nframes))
    #print "masks:", maskr.shape
    #print "stack:", stackr.shape
    roi_traces = np.dot(maskr, stackr)
    return roi_traces





# -----------------------------------------------------------------------------
# General utitilies often used for retino-related analysis
# -----------------------------------------------------------------------------

def convert_values(oldval, newmin=None, newmax=None, oldmax=None, oldmin=None):
    oldrange = (oldmax - oldmin)
    newrange = (newmax - newmin)
    newval = (((oldval - oldmin) * newrange) / oldrange) + newmin
    return newval


def correct_phase_wrap(phase):
        
    corrected_phase = phase.copy()
    
    corrected_phase[phase<0] =- phase[phase<0]
    corrected_phase[phase>0] = (2*np.pi) - phase[phase>0]
    
    return corrected_phase

def select_rois(mean_magratios, mag_thr=0.02, mag_thr_stat='max'):

    if mag_thr_stat == 'mean':
        best_mags = mean_magratios.loc[mean_magratios.mean(axis=1)>mag_thr].index.tolist()
    elif mag_thr_stat == 'max':
        best_mags = mean_magratios.loc[mean_magratios.max(axis=1)>mag_thr].index.tolist()
    else:
        mag_thr_stat == 'allconds'
        best_mags = [rid for rid in mean_magratios.index.tolist() if all(mean_magratios.iloc[rid] > mag_thr)]
    top_rois = copy.copy(np.array(best_mags))

    return top_rois

def do_fft_analysis(avg_traces, sorted_idxs, stim_freq_idx):
    n_frames = avg_traces.shape[0]

    fft_results = np.fft.fft(avg_traces, axis=0) #avg_traces.apply(np.fft.fft, axis=1)

    # get phase and magnitude
    mag_data = abs(fft_results)
    phase_data = np.angle(fft_results)

    # sort mag and phase by freq idx:
    mag_data = mag_data[sorted_idxs]
    phase_data = phase_data[sorted_idxs]

    # exclude DC offset from data
    mag_data = mag_data[int(np.round(n_frames/2.))+1:, :]
    phase_data = phase_data[int(np.round(n_frames/2.))+1:, :]

    #unpack values from frequency analysis
    mag_array = mag_data[stim_freq_idx, :]
    phase_array = phase_data[stim_freq_idx, :]

    #get magnitude ratio
    tmp = np.copy(mag_data)
    #tmp = np.delete(tmp,freq_idx,0)
    nontarget_mag_array=np.sum(tmp,0)
    magratio_array=mag_array/nontarget_mag_array

    return magratio_array, phase_array
#%%

# -----------------------------------------------------------------------------
# Data formatting
# -----------------------------------------------------------------------------

def trials_to_dataframes(processed_fpaths, conditions_fpath):
    
    # Get condition / trial info:
    with open(conditions_fpath, 'r') as f:
        conds = json.load(f)
    cond_list = list(set([cond_dict['stimuli']['stimulus'] for trial_num, cond_dict in conds.items()]))
    trials_by_cond = dict((cond, [int(k) for k, v in conds.items() if v['stimuli']['stimulus']==cond]) for cond in cond_list)

    excluded_tifs = [] 
    for cond, tif_list in trials_by_cond.items():
        for tifnum in tif_list:
            processed_tif = [f for f in processed_fpaths if 'File%03d' % tifnum in f]
            if len(processed_tif) == 0:
                print("No analysis found for file: %s" % tifnum)
                excluded_tifs.append(tifnum)
        trials_by_cond[cond] = [t for t in tif_list if t not in excluded_tifs]
    print("TRIALS BY COND:")
    print(trials_by_cond) 
    trial_list = [int(t) for t in conds.keys() if int(t) not in excluded_tifs]
    print("Trials:", trial_list)

    fits = []
    phases = []
    mags = []
    for trial_num, trial_fpath in zip(sorted(trial_list), sorted(processed_fpaths, key=natural_keys)):
        
        print("%i: %s" % (trial_num, os.path.split(trial_fpath)[-1]))
        df = h5py.File(trial_fpath, 'r')
        fits.append(pd.Series(data=df['var_exp_array'][:], name=trial_num))
        phases.append(pd.Series(data=df['phase_array'][:], name=trial_num))
        mags.append(pd.Series(data=df['mag_ratio_array'][:], name=trial_num))
        df.close()
        
    fit = pd.concat(fits, axis=1)
    magratio = pd.concat(mags, axis=1)
    phase = pd.concat(phases, axis=1)
    
    return fit, magratio, phase, trials_by_cond


def load_retino_analysis_info(animalid, session, fov, run, retinoid, use_pixels=False, rootdir='/n/coxfs01/2p-data'):
    
    run_dir = glob.glob(os.path.join(rootdir, animalid, session, '%s*' % fov, run))[0]
    fov = os.path.split(os.path.split(run_dir)[0])[-1]
    print("FOV: %s, run: %s" % (fov, run))
    retinoids_fpath = glob.glob(os.path.join(run_dir, 'retino_analysis', 'analysisids_*.json'))[0]
    with open(retinoids_fpath, 'r') as f:
        rids = json.load(f)
    if use_pixels:
        roi_analyses = [r for r, rinfo in rids.items() if rinfo['PARAMS']['roi_type'] == 'pixels']
    else:
        roi_analyses = [r for r, rinfo in rids.items() if rinfo['PARAMS']['roi_type'] != 'pixels']
    if retinoid not in roi_analyses:
        retinoid = sorted(roi_analyses, key=natural_keys)[-1] # use most recent roi analysis
        print("Fixed retino id to most recent: %s" % retinoid)
        
    return retinoid, rids[retinoid]

#%%
# Convert degs to centimeters:
def get_linear_coords(width, height, resolution, leftedge=None, rightedge=None, bottomedge=None, topedge=None):
    #width = 103 # in cm
    #height = 58 # in cm
    #resolution = [1920, 1080]

    if leftedge is None:
        leftedge = -1*width/2.
    if rightedge is None:
        rightedge = width/2.
    if bottomedge is None:
        bottomedge = -1*height/2.
    if topedge is None:
        topedge = height/2.

    #print("center 2 Top/Anterior:", topedge, rightedge)


    mapx = np.linspace(leftedge, rightedge, resolution[0] * ((rightedge-leftedge)/float(width)))
    mapy = np.linspace(bottomedge, topedge, resolution[1] * ((topedge-bottomedge)/float(height)))

    lin_coord_x, lin_coord_y = np.meshgrid(mapx, mapy, sparse=False)

    return lin_coord_x, lin_coord_y


def get_retino_info(animalid, session, fov=None, interactive=True, rootdir='/n/coxfs01/2p-data',
                    azimuth='right', elevation='top',
                    leftedge=None, rightedge=None, bottomedge=None, topedge=None):

    screen_info = get_screen_info(animalid, session, fov=fov, interactive=interactive,
                                  rootdir=rootdir)

    lin_coord_x, lin_coord_y = get_linear_coords(screen_info['azimuth'], 
                                                 screen_info['elevation'], 
                                                 screen_info['resolution'], 
                                                 leftedge=leftedge, rightedge=rightedge, 
                                                 bottomedge=bottomedge, topedge=topedge)
    
    linminW = lin_coord_x.min(); linmaxW = lin_coord_x.max()
    linminH = lin_coord_y.min(); linmaxH = lin_coord_y.max()

        
        
    retino_info = {}
    retino_info['width'] = screen_info['azimuth']
    retino_info['height'] = screen_info['elevation']
    retino_info['resolution'] = screen_info['resolution']
    #aspect_ratio = float(height)/float(width)
    retino_info['aspect'] = retino_info['height'] / retino_info['width']#aspect_ratio
    retino_info['azimuth'] = azimuth
    retino_info['elevation'] = elevation
    retino_info['linminW'] = linminW
    retino_info['linmaxW'] = linmaxW
    retino_info['linminH'] = linminH
    retino_info['linmaxH'] = linmaxH
    retino_info['bounding_box'] = [leftedge, bottomedge, rightedge, topedge]

    return retino_info

def format_retino_traces(data_fpath, info=None, trace_type='corrected'):
    f = h5py.File(data_fpath, 'r')
    
    labels_list = []
    nrois, nframes_per = f[f.keys()[0]][trace_type].shape
    dtype = f[f.keys()[0]][trace_type].dtype
    nfiles = len(f.keys())
    tmat = np.empty((nframes_per*nfiles, nrois), dtype=dtype)
    print("retino: nframes per trial (%i) for %i files" % (nframes_per, nfiles))
    s_ix = 0
    for tix, tfile in enumerate(sorted(f.keys(), key=natural_keys)):
        print("retino file %s: %i" % (tfile, s_ix))
        tmat[s_ix:s_ix+nframes_per, :] = f[tfile][trace_type][:].T
        
        if info is not None:
            trials_by_cond = info['trials']
            # Condition info:
            curr_cond = [cond for cond, trial_list in trials_by_cond.items() if int(tix+1) in trial_list][0]
            curr_trial = 'trial%05d' % int(tix+1)
            frame_tsecs = info['frame_tstamps_sec']
            if info['nchannels'] == 2:
                frame_tsecs = frame_tsecs[0::2]
                
            curr_labels = pd.DataFrame({'tsec': frame_tsecs,
                                        'config': [curr_cond for _ in range(len(frame_tsecs))],
                                        'trial': [curr_trial for _ in range(len(frame_tsecs))],
                                        'frame': [i for i in range(len(frame_tsecs))]
                                        })
            labels_list.append(curr_labels)
            
        s_ix = s_ix + nframes_per
        
    traces = pd.DataFrame(tmat)
    labels = pd.concat(labels_list, axis=0).reset_index(drop=True)
    
    return traces, labels
    
    
    
def get_protocol_info(animalid, session, fov, run='retino_run1', rootdir='/n/coxfs01/2p-data'):
    
    run_dir = os.path.join(rootdir, animalid, session, fov, run)
    mw_fpath = glob.glob(os.path.join(run_dir, 'paradigm', 'files', 'parsed*.json'))[0]
    with open(mw_fpath, 'r') as f:
        mwinfo = json.load(f)
    
    si_fpath = glob.glob(os.path.join(run_dir, '*.json'))[0]
    with open(si_fpath, 'r') as f:
        scaninfo = json.load(f)
    
    conditions = list(set([cdict['stimuli']['stimulus'] for trial_num, cdict in mwinfo.items()]))
    trials_by_cond = dict((cond, [int(k) for k, v in mwinfo.items() if v['stimuli']['stimulus']==cond]) \
                           for cond in conditions)
    print(trials_by_cond)
    n_frames = scaninfo['nvolumes']
    fr = scaninfo['frame_rate']
        

    stiminfo = dict((cond, dict()) for cond in conditions)
    curr_cond = conditions[0]
    # get some info from paradigm and run file
    stimfreq = np.unique([v['stimuli']['scale'] for k,v in mwinfo.items() if v['stimuli']['stimulus']==curr_cond])[0]
    stimperiod = 1./stimfreq # sec per cycle
    
    n_cycles = int(round((n_frames/fr) / stimperiod))
    n_frames_per_cycle = int(np.floor(stimperiod * fr))
    cycle_starts = np.round(np.arange(0, n_frames_per_cycle * n_cycles, n_frames_per_cycle)).astype('int')

    # Get frequency info
    freqs = np.fft.fftfreq(n_frames, float(1./fr))
    sorted_idxs = np.argsort(freqs)
    freqs = freqs[sorted_idxs] # sorted
    freqs = freqs[int(np.round(n_frames/2.))+1:] # exclude DC offset from data
    stim_freq_idx = np.argmin(np.absolute(freqs - stimfreq)) # Index of stimulation frequency

    stiminfo = {'stim_freq': stimfreq,
               'frame_rate': fr,
               'n_reps': len(trials_by_cond[curr_cond]),
               'n_frames': n_frames,
               'n_cycles': n_cycles,
               'n_frames_per_cycle': n_frames_per_cycle,
               'cycle_start_ixs': cycle_starts,
               'stim_freq_idx': stim_freq_idx,
               'freqs': freqs
                }
    
    scaninfo.update({'stimulus': stiminfo})
    scaninfo.update({'trials': trials_by_cond})


    return scaninfo


def get_retino_stimulus_info(mwinfo, runinfo):
    conditions = list(set([cdict['stimuli']['stimulus'] for trial_num, cdict in mwinfo.items()]))
    trials_by_cond = dict((cond, [int(k) for k, v in mwinfo.items() if v['stimuli']['stimulus']==cond]) \
                           for cond in conditions)
    
    stiminfo = dict((cond, dict()) for cond in conditions)
    for curr_cond in conditions:
        # get some info from paradigm and run file
        stimfreq = np.unique([v['stimuli']['scale'] for k,v in mwinfo.items() if v['stimuli']['stimulus']==curr_cond])[0]
        stimperiod = 1./stimfreq # sec per cycle
        
        n_frames = runinfo['nvolumes']
        fr = runinfo['frame_rate']
        
        n_cycles = int(round((n_frames/fr) / stimperiod))
        #print n_cycles

        n_frames_per_cycle = int(np.floor(stimperiod * fr))
        cycle_starts = np.round(np.arange(0, n_frames_per_cycle * n_cycles, n_frames_per_cycle)).astype('int')

        freqs = np.fft.fftfreq(n_frames, float(1./fr))
        sorted_idxs = np.argsort(freqs)
        freqs = freqs[sorted_idxs] # sorted
        freqs = freqs[int(np.round(n_frames/2.))+1:] # exclude DC offset from data
        stim_freq_idx = np.argmin(np.absolute(freqs - stimfreq)) # Index of stimulation frequency
        
        stiminfo[curr_cond] = {'stim_freq': stimfreq,
                               'frame_rate': fr,
                               'n_reps': len(trials_by_cond[curr_cond]),
                               'n_frames': n_frames,
                               'n_cycles': n_cycles,
                               'n_frames_per_cycle': n_frames_per_cycle,
                               'cycle_start_ixs': cycle_starts,
                               #'trials_by_cond': trials_by_cond,
                                'stim_freq_idx': stim_freq_idx,
                               'frame_tstamps': runinfo['frame_tstamps_sec']
                              }

    return stiminfo, trials_by_cond


# Interpolate bar position for found SI frame using upsampled MW tstamps and positions:
def get_interp_positions(condname, mwinfo, stiminfo, trials_by_cond):
    mw_fps = 1./np.diff(np.array(mwinfo[str(trials_by_cond[condname][0])]['stiminfo']['tstamps'])/1E6).mean()
    si_fps = stiminfo[condname]['frame_rate']
    print("[%s]: Downsampling MW positions (sampled at %.2fHz) to SI frame rate (%.2fHz)" % (condname, mw_fps, si_fps))

    si_cyc_ixs = stiminfo[condname]['cycle_start_ixs']
    assert len(np.unique(np.diff(si_cyc_ixs))) == 1, "Uneven cycle durs found!\n--> %s" % str(np.diff(si_cyc_ixs))
    fr_per_cyc = int(np.unique(np.diff(si_cyc_ixs))[0])
    
    si_tstamps = stiminfo[condname]['frame_tstamps']


    #fig, axes = pl.subplots(1, len(trials_by_cond[condname]))

    stim_pos_list = []
    stim_tstamp_list = []

    for ti, trial in enumerate(trials_by_cond[condname]):
        #ax = axes[ti]

        pos_list = []
        tstamp_list = []
        mw_cyc_ixs = mwinfo[str(trial)]['stiminfo']['start_indices']
        for cix in np.arange(0, len(mw_cyc_ixs)):
            if cix==len(mw_cyc_ixs)-1:
                mw_ts = [t/1E6 for t in mwinfo[str(trial)]['stiminfo']['tstamps'][mw_cyc_ixs[cix]:]]
                xs = mwinfo[str(trial)]['stiminfo']['values'][mw_cyc_ixs[cix]:]
                si_ts = si_tstamps[si_cyc_ixs[cix]:si_cyc_ixs[cix]+fr_per_cyc]
            else:
                mw_ts = np.array([t/1E6 for t in mwinfo[str(trial)]['stiminfo']['tstamps'][mw_cyc_ixs[cix]:mw_cyc_ixs[cix+1]]])
                xs = np.array(mwinfo[str(trial)]['stiminfo']['values'][mw_cyc_ixs[cix]:mw_cyc_ixs[cix+1]])
                si_ts = si_tstamps[si_cyc_ixs[cix]:si_cyc_ixs[cix+1]]

            recentered_mw_ts = [t-mw_ts[0] for t in mw_ts]
            recentered_si_ts = [t-si_ts[0] for t in si_ts]

            # Since MW tstamps are linear, SI tstamps linear, interpolate position values down to SI's lower framerate:
            interpos = sp.interpolate.interp1d(recentered_mw_ts, xs, fill_value='extrapolate')
            resampled_xs = interpos(recentered_si_ts)

            pos_list.append(pd.Series(resampled_xs, name=trial))
            tstamp_list.append(pd.Series(recentered_si_ts, name=trial))

            #ax.plot(recentered_mw_ts, xs, 'ro', alpha=0.5, markersize=2)
            #ax.plot(recentered_si_ts, resampled_xs, 'bx', alpha=0.5, markersize=2)

        pos_vals = pd.concat(pos_list, axis=0).reset_index(drop=True) 
        tstamp_vals = pd.concat(tstamp_list, axis=0).reset_index(drop=True)

        stim_pos_list.append(pos_vals)
        stim_tstamp_list.append(tstamp_vals)

    stim_positions = pd.concat(stim_pos_list, axis=1)
    stim_tstamps = pd.concat(stim_tstamp_list, axis=1)


    return stim_positions, stim_tstamps


#%%
def get_screen_info(animalid, session, fov=None, interactive=True, rootdir='/n/coxfs01/2p-data'):
    
    print("... Getting screen info")    
    screen = {}
    
    try:
        # Get bounding box values from epi:
        epi_session_paths = sorted(glob.glob(os.path.join(rootdir, animalid, 'epi_maps', '20*')), key=natural_keys)
        epi_sessions = sorted([os.path.split(s)[-1].split('_')[0] for s in epi_session_paths], key=natural_keys)
        #print("Found epi sessions: %s" % str(epi_sessions))
        if len(epi_sessions) > 0:
            epi_sesh = [datestr for datestr in sorted(epi_sessions, key=natural_keys) if int(datestr) <= int(session)][-1] # Use most recent session
            print("Most recent: %s" % str(epi_sesh))

            epi_fpaths = glob.glob(os.path.join(rootdir, animalid, 'epi_maps', '*%s*' % epi_sesh, 'screen_boundaries*.json'))
            if len(epi_fpaths) == 0:
                epi_fpaths = glob.glob(os.path.join(rootdir, animalid, 'epi_maps', '*%s*' % epi_sesh, '*', 'screen_boundaries*.json'))

        else:
            #print("No EPI maps found for session: %s * (using tmp session boundaries file)" % session)
            epi_fpaths = glob.glob(os.path.join(rootdir, animalid, 'epi_maps', 'screen_boundaries*.json'))
        
        assert len(epi_fpaths) > 0, "No epi screen info found!"
        
        # Each epi run should have only 2 .json files (1 for each condition):
        if len(epi_fpaths) > 2:
            print("-- found %i screen boundaries files: --" % len(epi_fpaths))
            repeat_epi_sessions = sorted(list(set( [os.path.split(s)[0] for s in epi_fpaths] )), key=natural_keys)
            for ei, repeat_epi in enumerate(sorted(repeat_epi_sessions, key=natural_keys)):
                print(ei, repeat_epi)
            if interactive:
                selected_epi = input("Select IDX of epi run to use: ")
            else:
                assert fov is not None, "ERROR: not interactive, but no FOV specified and multiple epis for session %s" % session
                
                selected_fovs = [fi for fi, epi_session_name in enumerate(repeat_epi_sessions) if fov in epi_session_name]
                print("Found FOVs: %s" % str(selected_fovs))

                if len(selected_fovs) == 0:
                    selected_epi = sorted(selected_fovs, key=natural_keys)[-1]
                else:
                    selected_epi = selected_fovs[0]
                
            epi_fpaths = [s for s in epi_fpaths if repeat_epi_sessions[selected_epi] in s]
        
        #print("-- getting screen info from:", epi_fpaths)
        
        for epath in epi_fpaths:
            with open(epath, 'r') as f:
                epi = json.load(f)
            print("getting screen info") 
            screen['azimuth'] = 59.7782*2. #epi['screen_params']['screen_size_x_degrees']
            screen['elevation'] = 33.6615*2. #epi['screen_params']['screen_size_t_degrees']
            screen['resolution'] = [1920, 1080] ##[epi['screen_params']['screen_size_x_pixels'], epi['screen_params']['screen_size_y_pixels']]

            if 'screen_boundaries' in epi.keys():
                if 'boundary_left_degrees' in epi['screen_boundaries'].keys():
                    screen['bb_left'] = epi['screen_boundaries']['boundary_left_degrees']
                    screen['bb_right'] = epi['screen_boundaries']['boundary_right_degrees']
                elif 'boundary_down_degrees' in epi['screen_boundaries'].keys():
                    screen['bb_lower'] = epi['screen_boundaries']['boundary_down_degrees']
                    screen['bb_upper'] = epi['screen_boundaries']['boundary_up_degrees']
            
            else:
                screen['bb_lower'] = -1*screen['elevation']/2.0
                screen['bb_upper'] = screen['elevation']/2.0
                screen['bb_left']  = -1*screen['azimuth']/2.0
                screen['bb_right'] = screen['azimuth']/2.0


        #print("*********************************")
        #pp.pprint(screen)
        #print("*********************************")
      
    except Exception as e:
        traceback.print_exc()
        
    return screen

#%%

# -----------------------------------------------------------------------------
# plotting
# -----------------------------------------------------------------------------


def plot_roi_traces_by_cond(roi_traces, stiminfo):
        
    fig, axes = pl.subplots(2,2, sharex=True, sharey=True, figsize=(10,6)) #, figsize=(8,))
    
    for aix, cond in enumerate(['left', 'right', 'bottom', 'top']):
        ax = axes.flat[aix]
        for repnum in np.arange(0, roi_traces[cond].shape[0]):
            ax.plot(roi_traces[cond][repnum, :], 'k', alpha=0.5, lw=0.5)
        ax.plot(roi_traces[cond].mean(axis=0), 'k', linewidth=1)
        ax.set_title(cond)
        for cyc in stiminfo[cond]['cycle_start_ixs']:
            ax.axvline(x=cyc, color='k', linestyle='--', linewidth=0.5)
            
    return fig


