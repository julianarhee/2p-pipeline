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
import tifffile as tf
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
import seaborn as sns

from mpl_toolkits.axes_grid1 import make_axes_locatable
from pipeline.python.utils import natural_keys, label_figure, replace_root, convert_range, get_screen_dims
#from matplotlib.patches import Ellipse, Rectangle

pp = pprint.PrettyPrinter(indent=4)
#from scipy.signal import argrelextrema

from pipeline.python.retinotopy import do_retinotopy_analysis as ra


from scipy import ndimage
import cv2
from scipy import misc,interpolate,stats,signal
from matplotlib.colors import LinearSegmentedColormap

import matplotlib.colors as mcolors
import cPickle as pkl
#%%
# -----------------------------------------------------------------------------
# Map funcs 
# -----------------------------------------------------------------------------
def arrays_to_maps(magratio, phase, trials_by_cond, use_cont=False,
                            dims=(512, 512), ds_factor=2, cond='right', 
                            mag_thr=None, mag_perc=0.05):
    if mag_thr is None:
        mag_thr = magratio.max().max()*mag_perc
        
    currmags = magratio[trials_by_cond[cond]]
    currmags[currmags<mag_thr] = np.nan
    currmags_mean = np.nanmean(currmags, axis=1)
    #d1 = int(np.sqrt(currmags_mean.shape[0]))
    d1 = dims[0] / ds_factor
    d2 = dims[1] / ds_factor
    currmags_map = np.reshape(currmags_mean, (d1, d2))
    
    currphase = phase[trials_by_cond[cond]]
    currphase_mean = stats.circmean(currphase, low=-np.pi, high=np.pi, axis=1)
    currphase_mean_c = correct_phase_wrap(currphase_mean)

    currphase_mean_c[np.isnan(currmags_mean)] = np.nan
    currphase_map_c = np.reshape(currphase_mean_c, (d1, d2))
    
    return currmags_map, currphase_map_c, mag_thr

def absolute_maps_from_conds(magratio, phase, trials_by_cond, mag_thr=0.01,
                                dims=(512, 512), ds_factor=2, outdir='/tmp', 
                                plot_conditions=False, data_id='dataid'):
    use_cont=False # doens't matter, should be equiv now
    magmaps = {}
    phasemaps = {}
    magthrs = {}
    for cond in trials_by_cond.keys():    
        magmaps[cond], phasemaps[cond], magthrs[cond] = arrays_to_maps(
                                                    magratio, phase, trials_by_cond,
                                                    cond=cond, use_cont=use_cont,
                                                    mag_thr=mag_thr, dims=dims,
                                                    ds_factor=ds_factor)
        if plot_conditions:
            fig = plot_filtered_maps(cond, magmaps[cond], 
                                        phasemaps[cond], magthrs[cond])
            label_figure(fig, data_id)
            figname = 'maps_%s_magthr-%.3f' % (cond, mag_thr)
            pl.savefig(os.path.join(outdir, '%s.png' % figname)) 
    ph_left = phasemaps['left'].copy()
    ph_right = phasemaps['right'].copy()
    ph_top = phasemaps['top'].copy()
    ph_bottom = phasemaps['bottom'].copy()
    print("got phase:", np.nanmin(ph_left), np.nanmax(ph_left)) # (0, 2*np.pi)

    absolute_az = (ph_left - ph_right) / 2.
    delay_az = (ph_left + ph_right) / 2.

    absolute_el = (ph_bottom - ph_top) / 2.
    delay_el = (ph_bottom + ph_top) / 2.

    vmin, vmax = (-np.pi, np.pi) # Now in range (-np.pi, np.pi)
    print("got absolute:", np.nanmin(absolute_az), np.nanmax(absolute_az))
    print("Delay:", np.nanmin(delay_az), np.nanmax(delay_az))

    return absolute_az, absolute_el, delay_az, delay_el
 

# -----------------------------------------------------------------------------
# Data processing funcs 
# -----------------------------------------------------------------------------

# preprocessing ---------------
def load_traces(animalid, session, fov, run='retino_run1', analysisid='analysis002',
                trace_type='raw', rootdir='/n/coxfs01/2p-data'):
    print("... loading traces (%s)" % trace_type)
    retinoid_path = glob.glob(os.path.join(rootdir, animalid, session, fov, '%s*' % run,
                                'retino_analysis', 'analysisids_*.json'))[0]
    with open(retinoid_path, 'r') as f:
        RIDS = json.load(f)
    eligible = [r for r, res in RIDS.items() if res['PARAMS']['roi_type']!='pixels']
    if analysisid not in eligible:
        print("Specified ID <%s> not eligible. Selecting 1st of %s" 
                    % (analysisid, str(eligible)))
        analysisid = eligible[0]

    analysis_dir = RIDS[analysisid]['DST']
    retino_dpath = os.path.join(analysis_dir, 'traces', 'extracted_traces.h5')
    scaninfo = get_protocol_info(animalid, session, fov, run=run)
    temporal_ds = RIDS[analysisid]['PARAMS']['downsample_factor']
    traces = load_traces_from_file(retino_dpath, scaninfo, trace_type=trace_type, 
                                    temporal_ds=temporal_ds)
    return traces


def load_traces_from_file(retino_dpath, scaninfo, trace_type='corrected', 
                            temporal_ds=None):
    '''
    Pre-processes raw extracted traces by:
        - adding back in neuropil offsets, and 
        - F0 offset from drift correction.
    Loads: ./traces/extracted_traces.h5 (contains data for each tif file).
    Averages traces for each condition. Downsamples final array.
    '''
    frame_rate = scaninfo['stimulus']['frame_rate']
    stim_freq = scaninfo['stimulus']['stim_freq']
    trials_by_cond = scaninfo['trials']

    traces = {}
    try:
        tfile = h5py.File(retino_dpath, 'r')
        for condition, trialnums in trials_by_cond.items():
            #print("... loading cond: %s" % condition)
            dlist = tuple([process_data(tfile, trialnum, trace_type=trace_type, frame_rate=frame_rate, stim_freq=stim_freq) for trialnum in trialnums])
            dfcat = pd.concat(dlist)
            df_rowix = dfcat.groupby(dfcat.index)
            meandf = df_rowix.mean()
            if temporal_ds is not None:
                #print("Temporal ds: %.2f" % temporal_ds)
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
    #print('Performing temporal smoothing on traces...')
    windowsz = int(temporal_ds)
    smooth_roi_trace = roi_trace.apply(temporal_downsample, args=(windowsz,), axis=0)
    return smooth_roi_trace
    

# smoothing ------------------
def smooth_neuropil(azim_r, smooth_fwhm=21):
    V=azim_r.copy()
    V[np.isnan(azim_r)]=0
    VV=ndimage.gaussian_filter(V,sigma=smooth_fwhm)

    W=0*azim_r.copy()+1
    W[np.isnan(azim_r)]=0
    WW=ndimage.gaussian_filter(W,sigma=smooth_fwhm)

    azim_smoothed = VV/WW
    return azim_smoothed

def smooth_phase_nans(inputArray, sigma, sz):
    
    V=inputArray.copy()
    V[np.isnan(inputArray)]=0
    VV=smooth_phase_array(V,sigma,sz)

    W=0*inputArray.copy()+1
    W[np.isnan(inputArray)]=0
    WW=smooth_phase_array(W,sigma,sz)

    Z=VV/WW

    return Z
   
def smooth_phase_array(theta,sigma,sz):
    #build 2D Gaussian Kernel
    kernelX = cv2.getGaussianKernel(sz, sigma); 
    kernelY = cv2.getGaussianKernel(sz, sigma); 
    kernelXY = kernelX * kernelY.transpose(); 
    kernelXY_norm=np.true_divide(kernelXY,np.max(kernelXY.flatten()))
    
    #get x and y components of unit-length vector
    componentX=np.cos(theta)
    componentY=np.sin(theta)
    
    #convolce
    componentX_smooth=signal.convolve2d(componentX, kernelXY_norm, 
                                            mode='same',boundary='symm')
    componentY_smooth=signal.convolve2d(componentY, kernelXY_norm, 
                                            mode='same',boundary='symm')

    theta_smooth=np.arctan2(componentY_smooth,componentX_smooth)
    return theta_smooth



# averaging -------------------
def block_mean(ar, fact):
    assert isinstance(fact, int), type(fact)
    sx, sy = ar.shape
    X, Y = np.ogrid[0:sx, 0:sy]
    regions = sy/fact * (X/fact) + Y/fact
    res = ndimage.mean(ar, labels=regions, index=np.arange(regions.max() + 1))
    res.shape = (sx/fact, sy/fact)
    return res

def get_condition_averaged_traces(RID, retinoid_dir, mwinfo, runinfo, tiff_fpaths, create_new=False):
    '''This only works for roi_type NOT pixels (otherwise creates ridick huge files
    '''
    # Set output dir:
    output_dir = os.path.join(retinoid_dir,'traces')
    if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    print output_dir

    avg_trace_fpath = os.path.join(output_dir, 'averaged_roi_traces.pkl')
    redo=False
    if os.path.exists(avg_trace_fpath) or create_new is False:
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

def average_retino_traces(RID, mwinfo, runinfo, tiff_fpaths, masks, 
                            output_dir='/tmp'):
    
    rep_list = [(k, v['stimuli']['stimulus']) for k,v in mwinfo.items()]
    unique_conditions = np.unique([rep[1] for rep in rep_list])
    conditions = dict((cond, 
                    [int(run) for run,config in rep_list if config==cond]) \
                                                    for cond in unique_conditions)
    print("CONDITIONS:", conditions)
    
    rtraces = {}

    # First check if extracted_traces file exists:
    traces_fpath = glob.glob(os.path.join(output_dir, 'extracted_traces*.h5'))
    extract_from_stack = False
    try:
        assert len(traces_fpath) == 1, "*No extracted_traces.h5:\n%s" % output_dir
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
def load_retinoanalysis(run_dir, traceid, verbose=False):
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
        if verbose:
            print("Corresponding ANALYSIS ID (for %s with %s) is: %s" % (traceid, roi_id, analysis_id))

    else:
        analysis_id = traceid 
    TID = tracedict[analysis_id]
    if verbose:
        pp.pprint(TID)
    return TID

def load_soma_and_np_masks(RETID):

    # Get ROIID and projection image
    roiid = RETID['PARAMS']['roi_id']
    ds_factor = int(RETID['PARAMS']['downsample_factor'])
    

    zimg = load_fov_image(RETID)
    d1, d2 = zimg.shape

    # Get roi extraction info
#    session_dir = RETID['DST'].split('FOV')[0]
#    rid_fpath = glob.glob(os.path.join(session_dir, 'ROIs', 'rids*.json'))[0]
#    with open(rid_fpath, 'r') as f:
#        rids = json.load(f)
#    reffile = rids[roiid]['PARAMS']['options']['ref_file']
#
    # Get reference file for run
    rdir, procid = RETID['SRC'].split('/processed/')
    pid = procid.split('_')[0]
    pidpath = glob.glob(os.path.join(rdir, 'processed', 'pids_*.json'))[0]
    with open(pidpath, 'r') as f:
        pids = json.load(f)
    reffile = pids[pid]['PARAMS']['motion']['ref_file']

    # Load masks
    retino_dpath = os.path.join(RETID['DST'], 'traces', 'extracted_traces.h5')
    tfile = h5py.File(retino_dpath, 'r')

    # Reshape masks
    masks_np = tfile['File%03d' % int(reffile)]['np_masks'][:].T
    masks_soma = tfile['File%03d' % int(reffile)]['masks'][:].copy()
    #print("NP masks:", masks_np.shape)

    nrois_total, _ = masks_soma.shape
    masks_np = np.reshape(masks_np, (nrois_total, d1, d2))
    masks_np[masks_np>0] = 1

    masks_soma = np.reshape(masks_soma, (nrois_total, d1, d2))
    masks_soma[masks_soma>0] = 1
    #print(masks_soma.shape)

    #print( tfile['File%03d' % int(reffile)].keys())
    
    return masks_soma, masks_np, zimg


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

def make_continuous(mapvals):
    map_c = mapvals.copy()
    map_c = -1*map_c
    map_c = map_c % (2*np.pi)
    return map_c

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

#def average_trial_dataframes(df, trials_by_cond, is_circular=False):
#    for 

def fft_results_by_trial(RETID):

    run_dir = RETID['DST'].split('/retino_analysis/')[0]
    processed_filepaths = glob.glob(os.path.join(RETID['DST'], 'files', '*h5'))
    trialinfo_filepath = glob.glob(os.path.join(run_dir, 'paradigm', 
                                    'files', 'parsed_trials*.json'))[0]
    _, magratio, phase, trials_by_cond = trials_to_dataframes(processed_filepaths, 
                                                trialinfo_filepath)
    return magratio, phase, trials_by_cond

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
        
        #print("%i: %s" % (trial_num, os.path.split(trial_fpath)[-1]))
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

    screen_info = get_screen_dims()

    lin_coord_x, lin_coord_y = get_linear_coords(screen_info['azimuth_deg'], 
                                                 screen_info['altitude_deg'], 
                                                 screen_info['resolution'], 
                                                 leftedge=leftedge, rightedge=rightedge, 
                                                 bottomedge=bottomedge, topedge=topedge)
    
    linminW = lin_coord_x.min(); linmaxW = lin_coord_x.max()
    linminH = lin_coord_y.min(); linmaxH = lin_coord_y.max()

            
    retino_info = {}
    retino_info['width'] = screen_info['azimuth_deg']
    retino_info['height'] = screen_info['altitude_deg']
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
    mwinfo = load_mw_info(animalid, session, fov, run, rootdir=rootdir)
   
    si_fpath = glob.glob(os.path.join(run_dir, '*.json'))[0]
    with open(si_fpath, 'r') as f:
        scaninfo = json.load(f)
    
    conditions = list(set([cdict['stimuli']['stimulus'] for trial_num, cdict in mwinfo.items()]))
    trials_by_cond = dict((cond, [int(k) for k, v in mwinfo.items() if v['stimuli']['stimulus']==cond]) \
                           for cond in conditions)
    #print(trials_by_cond)
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

def load_mw_info(animalid, session, fov, run_name, rootdir='/n/coxfs01/2p-data'):
    parsed_fpaths = glob.glob(os.path.join(rootdir, animalid, session, fov, 
                                    '%s*' % run_name, 
                                    'paradigm', 'files', 'parsed_trials*.json'))
    assert len(parsed_fpaths)==1, "Unable to find correct parsed trials path: %s" % str(parsed_fpaths)
    with open(parsed_fpaths[0], 'r') as f:
        mwinfo = json.load(f)

    return mwinfo



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

# -----------------------------------------------------------------------------
# plotting
# -----------------------------------------------------------------------------
# Load colormap

def get_retino_legends(cmap_name='nic_edge', zero_center=True, return_cmap=False,
                    cmap_dir='/n/coxfs01/julianarhee/aggregate-visual-areas/colormaps', 
                    dst_dir='/n/coxfs01/julianarhee/aggregate-visual-areas/retinotopy'):
    #colormap = 'nic_Edge'
    #cmapdir = os.path.join(aggr_dir, 'colormaps')
    cdata = np.loadtxt(os.path.join(cmap_dir, cmap_name) + ".txt")
    cmap_phase = LinearSegmentedColormap.from_list('my_colormap', cdata[::-1])
    screen = make_legends(cmap=cmap_phase, cmap_name=cmap_name, zero_center=zero_center,
                            dst_dir=dst_dir)
    if return_cmap:
        return screen, cmap_phase
    else:
        return screen
    
def load_fov_image(RETID):
    
    ds_factor = int(RETID['PARAMS']['downsample_factor'])

    # Load reference image
    imgs = glob.glob(os.path.join('%s*' % RETID['SRC'], 'std_images.tif'))[0]
    #imgs = glob.glob(os.path.join(rootdir, animalid, session, fov, retinorun, 'processed',\
    #                      'processed001*', 'mcorrected_*', 'std_images.tif'))[0]
    zimg = tf.imread(imgs)
    zimg = zimg.mean(axis=0)

    if ds_factor is not None:
        zimg = block_mean(zimg, int(ds_factor))

    print("... FOV size: %s (downsample factor=%i)" % (str(zimg.shape), ds_factor))
    
    return zimg

def create_legend(screen, zero_center=False):
    screen_x = screen['azimuth_deg']
    screen_y = screen['azimuth_deg'] #screen['altitude_deg']

    x = np.linspace(0, 2*np.pi, int(round(screen_x)))
    y = np.linspace(0, 2*np.pi, int(round(screen_y)) )
    xv, yv = np.meshgrid(x, y)

    az_legend = (2*np.pi) - xv
    el_legend = yv

    newmin = -0.5*screen_x if zero_center else 0
    newmax = 0.5*screen_x if zero_center else screen_x
    
    az_screen = convert_range(az_legend, newmin=newmin, newmax=newmax, 
                                oldmin=0, oldmax=2*np.pi)
    el_screen = convert_range(el_legend, newmin=newmin, newmax=newmax, 
                                oldmin=0, oldmax=2*np.pi)

    return az_screen, el_screen


def save_legend(az_screen, screen, cmap, cmap_name='cmap_name', cond='cond', dst_dir='/tmp'):
    screen_min = int(round(az_screen.min()))
    screen_max = int(round(az_screen.max()))
    #print("min/max:", screen_min, screen_max)
    
    fig, ax = pl.subplots()
    im = ax.imshow(az_screen, cmap=cmap)
    #ax.invert_xaxis()
   
    # Max value is twice the 0-centered value, or just the full value if not 0-cent
    max_v = screen['azimuth_deg'] #az_screen.max()*2.0 if screen_min < 0 else az_screen.max() #screen_max
  
    # Get actual screen edges
    midp = max_v/2.
    yedge_from_bottom = midp + screen['altitude_deg']/2.
    yedge_from_top = midp - screen['altitude_deg']/2.
    screen_edges_y = (-screen['altitude_deg']/2., screen['altitude_deg']/2.)

    if cond=='azimuth':

        ax.set_xticks(np.linspace(0, max_v, 5))
        ax.set_xticklabels([int(round(i)) for i in np.linspace(screen_min, screen_max, 5)][::-1])

        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.tick_params(axis='x', length=0)
        ax.set_xlim(ax.get_xlim()[::-1])
    
    else:

        ax.set_yticks(np.linspace(0, min(az_screen.shape), 5))
        ax.set_yticklabels([int(round(i)) for i in np.linspace(screen_min, screen_max, 5)])
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.tick_params(axis='y', length=0)

        #ax.axhline(y=yedge_from_bottom, color='w', lw=2)
        #ax.axhline(y=yedge_from_top, color='w', lw=2)
        #print(screen_edges_y)
        ax.set_ylim(ax.get_ylim()[::-1])

    ax.axhline(y=yedge_from_bottom, color='w', lw=2)
    ax.axhline(y=yedge_from_top, color='w', lw=2)

    ax.set_frame_on(False)
    pl.colorbar(im, ax=ax, shrink=0.7)

    figname = '%s_pos_%s_LEGEND_abs' % (cond, cmap_name)
    pl.savefig(os.path.join(dst_dir, '%s.svg' % figname))

    print(dst_dir, figname)
    return

def make_legends(cmap='nipy_spectral', cmap_name='nipy_spectral', zero_center=False,
                 dst_dir='/n/coxfs01/julianarhee/aggregate-data/retinotopy'):

    screen = get_screen_dims()
    azi_legend, alt_legend = create_legend(screen, zero_center=zero_center)
   
    if dst_dir is not None:
        save_legend(azi_legend, screen, cmap=cmap, 
                        cmap_name=cmap_name, cond='azimuth', dst_dir=dst_dir)
        save_legend(alt_legend, screen, cmap=cmap, 
                        cmap_name=cmap_name, cond='elevation', dst_dir=dst_dir)
        
    screen.update({'azi_legend': azi_legend,
                   'alt_legend': alt_legend})
    return screen

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

def plot_some_example_traces(soma_traces, np_traces, plot_rois=[],
                            dst_dir='/tmp', data_id='dataid'):
    if not os.path.exists(os.path.join(dst_dir, 'example_traces')):
        os.makedirs(os.path.join(dst_dir, 'example_traces'))
    
    for rid in plot_rois: #sorted_rois_soma[0:3]:
        fig, axn = pl.subplots(4, 1, sharex=True, sharey=True, figsize=(5, 6))
        for ai, (ax, cond) in enumerate(zip(axn.flat, ['left', 'right', 'top', 'bottom'])):
            ax = plot_example_traces(soma_traces, np_traces, rid=rid, cond=cond, ax=ax)
            if ai==0:
                ax.legend(bbox_to_anchor=(1, 1))
            ax.set_title(cond, loc='left', fontsize=12)
        pl.subplots_adjust(left=0.1, right=0.75, hspace=0.5, top=0.9)
        sns.despine(trim=True)
        pl.suptitle('RID %i' % rid)
        
        label_figure(fig, data_id)
        pl.savefig(os.path.join(dst_dir, 'example_traces', 
                                    'np_v_soma_roi%05d.svg' % int(rid+1)))
        return
        
def plot_example_traces(soma_traces, np_traces, rid=0, cond='right',
                       soma_color='k', np_color='r', ax=None):
    
    if ax is None:
        fig, ax = pl.subplots()
    ax.plot(soma_traces[cond][rid], soma_color, label='soma')
    ax.plot(np_traces[cond][rid], np_color, label='neuropil')
    
    return ax


def plot_phase_and_delay_maps(absolute_az, absolute_el, delay_az, delay_el, 
                                cmap='nipy_spectral', vmin=-np.pi, vmax=np.pi, 
                                elev_cutoff=0.56):
    fig, axes = pl.subplots(2,2)
    im1 = axes[0,0].imshow(absolute_az, cmap=cmap, vmin=vmin, vmax=vmax)
    im2 = axes[0,1].imshow(absolute_el, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1,0].imshow(delay_az, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1,1].imshow(delay_el, cmap=cmap, vmin=vmin, vmax=vmax)

    cbar1_orientation='horizontal'
    cbar1_axes = [0.35, 0.85, 0.1, 0.1]
    cbar2_orientation='vertical'
    cbar2_axes = [0.75, 0.85, 0.1, 0.1]

    cbaxes = fig.add_axes(cbar1_axes) 
    cb = pl.colorbar(im1, cax = cbaxes, orientation=cbar1_orientation)  
    cb.ax.axis('off')
    cb.outline.set_visible(False)

    cbaxes = fig.add_axes(cbar2_axes) 
    cb = pl.colorbar(im2, cax = cbaxes, orientation=cbar2_orientation)
    #cb.ax.set_ylim([cb.norm(-np.pi*top_cutoff), cb.norm(np.pi*top_cutoff)])
    cb.ax.axhline(y=cb.norm(vmin*elev_cutoff), color='w', lw=1)
    cb.ax.axhline(y=cb.norm(vmax*elev_cutoff), color='w', lw=1)
    cb.ax.axis('off')
    cb.outline.set_visible(False)
    pl.subplots_adjust(top=0.8)

    for ax in axes.flat:
        ax.axis('off')
 
    return fig

def plot_filtered_maps(cond, currmags_map, currphase_map_c, mag_thr):
    '''
    For given cond, plots mag-ratio and phase maps.
    '''
    fig, axes = pl.subplots(1, 2) #pl.figure()
    im = axes[0].imshow(currmags_map)
    divider = make_axes_locatable(axes[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    
    im2 = axes[1].imshow(currphase_map_c, cmap='nipy_spectral', vmin=0, vmax=2*np.pi)
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im2, cax=cax, orientation='vertical')
    
    pl.subplots_adjust(wspace=0.5)
    fig.suptitle('%s (mag_thr: %.4f)' % (cond, mag_thr))

    return fig


