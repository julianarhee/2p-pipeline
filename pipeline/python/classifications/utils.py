#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 12:31:35 2018

@author: juliana
"""

import matplotlib as mpl
mpl.use('agg')
import h5py
import os
import json
import cv2
import time
import math
import random
import itertools
import copy
import scipy.io
import optparse
import cPickle as pkl
import pandas as pd
import numpy as np
import pylab as pl
import seaborn as sns
import pyvttbl as pt
import multiprocessing as mp
import tifffile as tf
from collections import namedtuple
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import scikit_posthocs as sp
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison
from statsmodels.nonparametric.smoothers_lowess import lowess
from skimage import exposure
from collections import Counter

from pipeline.python.utils import natural_keys, replace_root, print_elapsed_time
import pipeline.python.traces.combine_runs as cb
import pipeline.python.paradigm.align_acquisition_events as acq
import pipeline.python.visualization.plot_psths_from_dataframe as vis
from pipeline.python.traces.utils import load_TID

from mpl_toolkits.axes_grid1 import make_axes_locatable
from pipeline.python.traces.utils import get_frame_info

from pipeline.python.retinotopy import utils as retinotools
from pipeline.python.retinotopy import fit_2d_rfs as fitrf
from pipeline.python.classifications import test_responsivity as resp

#%%


import glob
import os
import json
import re
import pandas as pd


#### GENERAL FUNCTIONS ########################################################
class Struct():
    pass

def get_roi_id(animalid, session, fov, traceid, run_name='', rootdir='/n/coxfs01/2p-data'):
    extraction_type = re.sub('[0-9]+', '', traceid) if 'traces' in traceid else 'retino_analysis'
    #extraction_num = int(re.findall(r'\d+', traceid)[0])
    
    if 'retino' in run_name and extraction_type=='traces': #using traceid in reference to other run types
        traceid_info_fpath = glob.glob(os.path.join(rootdir, animalid, session, fov, '*', \
                                         'traces', 'traceids_*.json'))[0] # % traceid, ))
    else:
        traceid_info_fpath = glob.glob(os.path.join(rootdir, animalid, session, fov, '*%s*' % run_name, \
                                             '%s' % extraction_type, '*.json'))[0] # % traceid, ))
    with open(traceid_info_fpath, 'r') as f:
        traceids = json.load(f)
        
    roi_id = traceids[traceid]['PARAMS']['roi_id']
    
    if 'retino' in run_name: #extraction_type == 'retino_analysis':
        extraction_type = 'retino_analysis'
        retinoid_info_fpath = glob.glob(os.path.join(rootdir, animalid, session, fov, '*%s*' % run_name, \
                                             '%s' % extraction_type, '*.json'))[0] # % traceid, ))
        with open(retinoid_info_fpath, 'r') as f:
            retino_ids = json.load(f)
        found_ids = [t for t, tinfo in retino_ids.items() if 'roi_id' in tinfo['PARAMS'].keys()\
                     and tinfo['PARAMS']['roi_id'] == roi_id]
        if len(found_ids) > 1:
            for fi, fid in enumerate(found_ids):
                print fi, fid
            sel = input("More than 1 retino analysis using [%s]. Select IDX to use: " % roi_id)
            traceid = found_ids[int(sel)]
        else:
            traceid = found_ids[0]
            
        
    return roi_id, traceid


def get_anatomical(animalid, session, fov, channel_num=2, rootdir='/n/coxfs01/2p-data'):
    anatomical = None
    fov_dir = os.path.join(rootdir, animalid, session, fov)
    anatomical_dirs = glob.glob(os.path.join(fov_dir, 'anatomical'))
    print("[%s] %s - %s:  Getting anatomicals..." % (animalid, session, fov))
    try:
        assert len(anatomical_dirs) > 0, "No anatomicals for current session: (%s | %s | %s)" % (animalid, session, fov)
        anatomical_dir = anatomical_dirs[0]
        print("... Found %i anatomical runs." % len(anatomical_dirs))
        anatomical_imgs = glob.glob(os.path.join(anatomical_dir, 'processed',
                                                 'processed*', 'mcorrected_*_mean_deinterleaved',
                                                 'Channel%02d' % channel_num, 'File*', '*.tif'))
        assert len(anatomical_imgs) > 0, "... No processed anatomicals found!"
        images=[]
        for fpath in anatomical_imgs:
            im = tf.imread(fpath)
            images.append(im)
        anatomical = np.array(images).sum(axis=0)
    except Exception as e:
        print e
        
    return anatomical
        

def load_roi_masks(animalid, session, fov, rois=None, rootdir='/n/coxfs01/2p-data'):
    mask_fpath = glob.glob(os.path.join(rootdir, animalid, session, 'ROIs', '%s*' % rois, 'masks.hdf5'))[0]
    mfile = h5py.File(mask_fpath, 'r')

    # Load and reshape masks
    masks = mfile[mfile.keys()[0]]['masks']['Slice01'][:].T
    print(masks.shape)
    mfile[mfile.keys()[0]].keys()

    zimg = mfile[mfile.keys()[0]]['zproj_img']['Slice01'][:].T
    zimg.shape
    
    return masks, zimg


#### RETINOTOPY EXPERIMENT FUNCTIONS ##########################################
def get_retino_analysis(animalid, session, fov, run='retino_run1', rois=None, rootdir='/n/coxfs01/2p-data'):
    
    run_dir = os.path.join(rootdir, animalid, session, fov, run)
    analysis_info_fpath = glob.glob(os.path.join(run_dir, 'retino_analysis', 'analysisids*.json'))[0]
    with open(analysis_info_fpath, 'r') as f:
        ainfo = json.load(f)
        
    # Find analysis id using roi type specified:
    if rois == 'pixels':
        found_ids = sorted([a for a, info in ainfo.items() if info['PARAMS']['roi_type']==rois], key=natural_keys)
    else:
        found_ids = sorted([a for a, info in ainfo.items() if 'roi_id' in info['PARAMS'].keys()\
                            and info['PARAMS']['roi_id'] == rois], key=natural_keys)
    assert len(found_ids) > 0, "No analysis ids found of type: %s (run dir:\n%s)" % (rois, run_dir)
    if len(found_ids) > 1:
        for fi, fr in enumerate(found_ids):
            print fi, fr
        sel = input("Select ID of analysis to use: ")
        analysis_id = found_ids[int(sel)]
    else:
        analysis_id = found_ids[0]
        
    data_fpath = glob.glob(os.path.join(run_dir, 'retino_analysis', '%s*' % analysis_id, 'traces', 'extracted_traces.h5'))[0]
    
    return data_fpath

def get_retino_stats(expdata, responsive_thr=0.01):
    magratios, phases, traces, trials_by_cond = do_retino_analysis_on_raw(expdata)
    roi_list = [r for r in magratios.index.tolist() if any(magratios.loc[r] > responsive_thr)]
    nrois_total = len(magratios.index.tolist())
    rstats = {'magratios': magratios, 'phases': phases, 'traces': traces}
    
    return rstats, roi_list, nrois_total, trials_by_cond
        

def do_retino_analysis_on_raw(expdata):
    n_frames = expdata.info['stimulus']['nframes']
    n_files = expdata.info['ntiffs']
    fr = expdata.info['stimulus']['frame_rate']
    stimfreq = expdata.info['stimulus']['stimfreq']

    # label frequency bins
    freqs = np.fft.fftfreq(n_frames, float(1/fr))
    sorted_freq_ixs = np.argsort(freqs)
    freqs=freqs[sorted_freq_ixs]
    #print(freqs)

    # exclude DC offset from data
    freqs=freqs[int(np.round(n_frames/2.))+1:]

    # Identify freq idx:
    stim_freq_ix=np.argmin(np.absolute(freqs-stimfreq))#find out index of stimulation freq
    top_freq_ix=np.where(freqs>1)[0][0]#find out index of 1Hz, to cut-off zoomed out plot
    print("Target freq: %.3f Hz" % (freqs[stim_freq_ix]))
    

    trials_by_cond = expdata.info['trials']
    trial_nums = np.array([v for k,v in trials_by_cond.items()])
    trial_nums = sorted(trial_nums.flatten())

    nframes_total, nrois = expdata.traces.shape
    magratios=[]
    phases=[]
    conds=[]
    traces={}
    for curr_cond in trials_by_cond.keys():
        avg_traces = []
        for rid in expdata.traces.columns:
            tracemat = pd.DataFrame(np.reshape(expdata.traces[rid], (n_frames, n_files), order='F'),\
                                    columns=trial_nums)
            avg = tracemat[trials_by_cond[curr_cond]].mean(axis=1)
            avg_traces.append(avg)
        avg_traces = pd.DataFrame(np.array(avg_traces).T, columns=expdata.traces.columns)
        traces[curr_cond] = avg_traces

        magratio_array, phase_array = do_fft_analysis(avg_traces, sorted_freq_ixs, stim_freq_ix, n_frames)

        magratios.append(magratio_array)
        phases.append(phase_array)
        conds.append(curr_cond)
        
    magratios = pd.DataFrame(np.array(magratios).T, columns=conds)
    phases = pd.DataFrame(np.array(phases).T, columns=conds)
    
    return magratios, phases, traces, trials_by_cond


def do_fft_analysis(avg_traces, sorted_freq_ixs, stim_freq_ix, n_frames):
    fft_results = np.fft.fft(avg_traces, axis=0) #avg_traces.apply(np.fft.fft, axis=1)

    # get phase and magnitude
    mag_data = abs(fft_results)
    phase_data = np.angle(fft_results)

    # sort mag and phase by freq idx:
    mag_data = mag_data[sorted_freq_ixs]
    phase_data = phase_data[sorted_freq_ixs]

    # exclude DC offset from data
    mag_data = mag_data[int(np.round(n_frames/2.))+1:, :]
    phase_data = phase_data[int(np.round(n_frames/2.))+1:, :]

    #unpack values from frequency analysis
    mag_array = mag_data[stim_freq_ix, :]
    phase_array = phase_data[stim_freq_ix, :]

    #get magnitude ratio
    tmp = np.copy(mag_data)
    #tmp = np.delete(tmp,freq_idx,0)
    nontarget_mag_array=np.sum(tmp,0)
    magratio_array=mag_array/nontarget_mag_array

    return magratio_array, phase_array


#### RECEPTIVE FIELD EXPERIMENT FUNCTIONS #####################################
def get_receptive_field_fits(animalid, session, fov, receptive_field_fit='zscore0.00_no_trim',
                             run='combined_rfs*_static', traceid='traces001', rootdir='/n/coxfs01/2p-data'):
    #assert 'rfs' in S.experiments['rfs'].name, "This is not a RF experiment object! %s" % exp.name
    rfits = None
    fov_dir = os.path.join(rootdir, animalid, session, fov)
    do_fits = False
    try:
        combined_rf_dirs = glob.glob(os.path.join(fov_dir, run, 'traces', '%s*' % traceid))
        #assert len(combined_rf_dirs) == 1, "---> [%s] warning: No unique traceid dir found (%s)" % (run, traceid)
        rf_traceid_dir = combined_rf_dirs[0]
        
        rf_fits = sorted(glob.glob(os.path.join(rf_traceid_dir,
                                         'figures', 'receptive_fields', 
                                         'rfs_2dgaus*%s*' % receptive_field_fit, '*.pkl')), key=natural_keys)
        if len(rf_fits) > 1:
            print("RFs: %s - more than 1 RF fit result found:" % run)
            for r, ri in enumerate(rf_fits):
                print(r, ri)
            sel = input("-- Select IDX of fits to use: ")
            rfs_fpath = rf_fits[int(sel)]
        elif len(rf_fits) == 1:
            rfs_fpath = rf_fits[0]
        else:
            do_fits = True
        
        if do_fits:
            print("... specified RF fit method not found, running now.")
            rfits = fitrf.fit_2d_receptive_fields(animalid, session, fov, run, traceid, 
                            trace_type='corrected', visual_area='', select_rois=False, segment=False,
                            response_type='zscore', response_thr=None, rootdir='/n/coxfs01/2p-data')
        
        else:
            print("... loading RF fits: %s" % receptive_field_fit)
            with open(rfs_fpath, 'rb') as f:
                rfits = pkl.load(f)
    except Exception as e:
        print("*** NO receptive field fits found: %s ***" % '|'.join([animalid, session, fov, run, traceid]))
        print e
        
    return rfits


#### EVENT PROTOCOL FUNCTIONS #################################################


def get_responsive_cells(animalid, session, fov, run=None, traceid='traces001',
                         responsive_test='ROC', responsive_thr=0.05,
                         rootdir='/n/coxfs01/2p-data'):
        
    traceid_dir =  glob.glob(os.path.join(rootdir, animalid, session, fov, run, 'traces', '%s*' % traceid))[0]

    if responsive_test == 'ROC':
        stats_dir = os.path.join(traceid_dir, 'summary_stats', responsive_test)
        stats_fpath = glob.glob(os.path.join(stats_dir, '*results*.pkl'))
        assert len(stats_fpath) > 0, "No stats results found for: %s" % stats_dir
        with open(stats_fpath[0], 'rb') as f:
            rstats = pkl.load(f)

    roi_list = [r for r, res in rstats.items() if res['pval'] < responsive_thr]
    
    return roi_list
    


def get_roi_stats(animalid, session, fov, exp_name=None, traceid='traces001', 
                  responsive_test='ROC', responsive_thr=0.01, 
                  receptive_field_fit='zscore0.00_no_trim', rootdir='/n/coxfs01/2p-data'):
    rstats = None
    roi_list = None
    nrois_total = None
    # Load list of "visually responsive" cells
    if ('blobs' in exp_name) or ('gratings' in exp_name and int(session) >= 20190511):
        print("... loading ROI stats: %s" % responsive_test)
        curr_traceid_dir = glob.glob(os.path.join(rootdir, animalid, session, fov, \
                                                  exp_name, 'traces', '%s*' % traceid))[0]
        #print("...", curr_traceid_dir)
        #exp.source.split('/data_arrays/')[0]
        try:
            curr_stats_dir = os.path.join(curr_traceid_dir, 'summary_stats', responsive_test)
            stats_fpath = glob.glob(os.path.join(curr_stats_dir, '*results*.pkl'))
            assert len(stats_fpath) > 0, "No stats results found for: %s" % curr_stats_dir
            with open(stats_fpath[0], 'rb') as f:
                rstats = pkl.load(f)
            roi_list = [r for r, res in rstats.items() if res['pval'] < responsive_thr]
            nrois_total = len(rstats.keys())
        except Exception as e:
            print e
            print("-- Unable to load stats: %s [%s]" % (responsive_test, exp_name))
            
    else:
        if 'rfs' in exp_name or ('gratings' in exp_name and int(session) < 20190511):
#            if (exp_name == 'gratings' and int(session) < 20190511):
#                print "OLD"
            try:
                rf_fit_thr = 0.5
                rstats = get_receptive_field_fits(animalid, session, fov,
                                                 run=exp_name, traceid=traceid, 
                                                 receptive_field_fit=receptive_field_fit,
                                                 rootdir=rootdir) #(S.experiments[exp_name])
                print("... loaded rf fits")
                roi_list = [r for r, res in rstats['fits'].items() if res['fit_r']['r2'] >= rf_fit_thr]
                #if exp_name == 'gratings':
                #    exp_name = 'rfs'
                nrois_total = len(rstats['fits'].keys())
            except Exception as e:
                print e
                print("-- No RF fits! [%s]" % exp_name)
        elif 'retino' in exp_name:
            print("-- Not implemented -- [%s] run get_retino_stats()" % exp_name)
            
    return rstats, roi_list, nrois_total


def check_counts_per_condition(raw_traces, labels):
    # Check trial counts / condn:
    min_n = labels.groupby(['config'])['trial'].unique().apply(len).min()
    conds_to_downsample = np.where( labels.groupby(['config'])['trial'].unique().apply(len) != min_n)[0]
    if len(conds_to_downsample) > 0:
        print("incorrect reps / condn...")
        d_cfgs = [sorted(labels.groupby(['config']).groups.keys())[i]\
                  for i in conds_to_downsample]
        trials_kept = []
        for cfg in labels['config'].unique():
            c_trialnames = labels[labels['config']==cfg]['trial'].unique()
            if cfg in d_cfgs:
                #ntrials_remove = len(c_trialnames) - min_n
                #print("... removing %i trials" % ntrials_remove)
    
                # In-place shuffle
                random.shuffle(c_trialnames)
    
                # Take the first 2 elements of the now randomized array
                trials_kept.extend(c_trialnames[0:min_n])
            else:
                trials_kept.extend(c_trialnames)
    
        ixs_kept = labels[labels['trial'].isin(trials_kept)].index.tolist()
        
        tmp_traces = raw_traces.loc[ixs_kept].reset_index(drop=True)
        tmp_labels = labels[labels['trial'].isin(trials_kept)].reset_index(drop=True)
    else:
        return raw_traces, labels
    
    return tmp_traces, tmp_labels
    

#%%

class Session():
    def __init__(self, animalid, session, fov, visual_area=None, state=None, rootdir='/n/coxfs01/2p-data'):
        self.animalid = animalid
        self.session = session
        self.fov = fov
        
        if visual_area is None or state is None:
            with open(os.path.join(rootdir, animalid, 'sessionmeta.json'), 'r') as f:
                sessionmeta = json.load(f)
            skey = [k for k in sessionmeta.keys() if k.split('_')[0] == session and k.split('_')[1] in fov][0]
            visual_area = sessionmeta[skey]['visual_area']
            state = sessionmeta[skey]['state']
            
        self.visual_area = visual_area
        self.state = state
        
        self.anatomical = get_anatomical(animalid, session, fov, rootdir=rootdir)
        
        self.rois = None
        self.traceid = None
        self.trace_type = None
        self.experiments = {}
        self.experiment_list = self.get_experiment_list(rootdir=rootdir)
        
        self.screen = retinotools.get_retino_info(animalid, session, fov=fov, rootdir=rootdir)

    def get_stimulus_coordinates(self, update_self=False):

        # Get stimulus positions - blobs and gratings only
        xpositions=[]; ypositions=[];
        for ex in ['blobs', 'gratings']:
            if ex not in self.experiment_list: #.keys():
                print("[%s|%s] No experiment exists for: %s" % (self.animalid, self.session, ex))
                continue
            if ex not in self.experiments.keys():
                expdict = self.load_data(experiment=ex, update_self=update_self)
                expdata = expdict[ex]
            else:
                expdata = self.experiments[ex]
                
            sdf = expdata.data.sdf.copy()
            if ex == 'gratings': # deal with FF stimuli
                sdf = sdf[sdf['size']<200]
                sdf.pop('luminance')
            curr_xpos = sdf.dropna()['xpos'].unique()
            assert len(curr_xpos)==1, "[%s] more than 1 xpos found! %s" % (ex, str(curr_xpos))
            curr_ypos = sdf.dropna()['ypos'].unique()
            assert len(curr_ypos)==1, "[%s] more than 1 ypos found! %s" % (ex, str(curr_ypos))
            xpositions.append(curr_xpos[0])
            ypositions.append(curr_ypos[0])
        
        xpos = list(set(xpositions))
        assert len(xpos)==1, "blobs and gratings have different XPOS: %s" % str(xpos)
        ypos = list(set(ypositions))
        assert len(ypos)==1, "blobs and gratings have different YPOS: %s" % str(ypos)
        xpos = xpos[0]
        ypos = ypos[0]
        print("Stimuli presented at coords: (%i, %i)" % (xpos, ypos))
        
        return xpos, ypos
    

    def get_stimulus_sizes(self, size_tested = ['gratings', 'blobs']):
        
        tested_exps = [e for e in self.experiment_list if e in size_tested]
    
        stimsizes = {}
        for exp in tested_exps:
            stimsizes[exp] = self.experiments[exp].data.sdf.dropna()['size'].unique()
            
    #    gratings_sz = S.experiments['gratings'].data.sdf['size'].unique().min()
    #    print("Gratings: min apertured size %i" % gratings_sz)
    #    
    #    # Get blob size(s):
    #    blobs_sz_min = S.experiments['blobs'].data.sdf.dropna()['size'].unique().min()
    #    blobs_sz_max = S.experiments['blobs'].data.sdf.dropna()['size'].unique().max()
    #    print("Blobs: min/max size = %i/%i" % (blobs_sz_min, blobs_sz_max))
        return stimsizes
 
    
    def save_session(self, rootdir='/n/coxfs01/2p-data'):
        outdir = os.path.join(rootdir, self.animalid, self.session, self.fov, 'summaries')
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        with open(os.path.join(outdir, 'sessiondata.pkl'), 'wb') as f:
            pkl.dump(self, f, protocol=pkl.HIGHEST_PROTOCOL)
            
    
    def load_masks(self, rootdir='/n/coxfs01/2p-data'):
        masks, zimg = load_roi_masks(self.animalid, self.session, self.fov, rois=self.rois, rootdir=rootdir)
        return masks, zimg
    
    
    def load_data(self, traceid='traces001', trace_type='corrected', make_equal=True,\
                  experiment=None, rootdir='/n/coxfs01/2p-data', update_self=True):
        
        '''Set experiment = None to load all data'''
        
        if update_self:
            self.traceid = traceid
            self.trace_type = trace_type
        if experiment is not None:
            print("... Loading data (%s - %s - %s)" % (experiment, traceid, trace_type))
        else:
            print("... Loading all experiments (%s - %s)" % (traceid, trace_type))
#            
#        self.rois, tmp_tid = get_roi_id(self.animalid, self.session, self.fov, traceid, rootdir=rootdir)
#        if tmp_tid != self.traceid:
#            self.traceid = tmp_tid
            
        expdict = self.get_experiment_data(experiment=experiment,\
                                           traceid=traceid,
                                           trace_type=trace_type,\
                                           rootdir=rootdir, 
                                          make_equal=make_equal)
        
        if update_self and expdict is not None:
            self.experiments.update(expdict)
        return expdict
    

    def get_experiment_list(self, traceid='traces001', trace_type='corrected',\
                            rootdir='/n/coxfs01/2p-data'):

        fov_dir = os.path.join(rootdir, self.animalid, self.session, self.fov)
        run_list = sorted(glob.glob(os.path.join(fov_dir, '*_run[0-9]')), key=natural_keys)
        experiment_list = list(set([os.path.split(f)[-1].split('_run')[0] for f in run_list]))
        
        if int(self.session) < 20190511 and 'gratings' in experiment_list:
            # Old experiment, where "gratings" were actually RFs
            experiment_list = [e for e in experiment_list if e != 'gratings']
            experiment_list.append('rfs') # These are always 5 degree res        
        
        return experiment_list
    
    def get_experiment_data(self, experiment=None, traceid='traces001', trace_type='corrected',\
                            rootdir='/n/coxfs01/2p-data', make_equal=True):
        experiment_dict = {}

        all_experiments = self.get_experiment_list(traceid=traceid, trace_type=trace_type, rootdir=rootdir)
         
        if experiment is None: # Get ALL experiments
            experiment_types = all_experiments
        else:
            if not isinstance(experiment, list):
                experiment_types = [experiment]
            else:
                experiment_types = experiment
        print("EXP:", experiment_types)

       
        #try:
            # Create object for each experiment:
        for experiment_type in experiment_types:     
            if int(self.session) < 20190511 and experiment_type == 'rfs':
                experiment_type = 'gratings' # Temporarily revert back to old-name since get_experiment_list() changed
        
            try:        
                self.rois, tmp_tid = get_roi_id(self.animalid, self.session, self.fov, traceid, run_name=experiment_type, rootdir=rootdir)
                print("-- %s: got rois" % experiment_type)
                if tmp_tid != self.traceid:
                    self.traceid = tmp_tid
                    print("renamed traceid")
                exp = Experiment(experiment_type, self.animalid, self.session, self.fov, self.traceid, rootdir=rootdir)
                print("-- %s: got source" % (experiment_type)) 
                if exp.source is None:
                    continue
                exp.load(trace_type=trace_type, make_equal=make_equal)
                print("loaded")
                if 'gratings' in experiment_type and int(self.session) < 20190511:
                    experiment_type = 'rfs'
            except Exception as e:
                print e
                print("--- %s skipping ---" % experiment_type)
                exp = None
                #experiment_dict = None
            
            experiment_dict[experiment_type] = exp
               
        return experiment_dict
    
    
    def get_grouped_stats(self, experiment_type, responsive_thr=0.01, responsive_test='ROC', 
                          receptive_field_fit='zscore0.00_no_trim',
                          update=True, get_grouped=True, make_equal=True,
                          traceid='traces001', trace_type='corrected',
                          rootdir='/n/coxfs01/2p-data'):
        
        #assert exp in [v for k, v in self.experiments.items()], "*ERROR* - specified experiment (%s) not found in Session object." % exp.name
        expdict=None; estats_dict=None;
        
        experiment_names = [k for k, v in self.experiments.items()]
        # see if already loaded data:
        print("Getting stats:", experiment_type)
        if experiment_type is not None:
            found_exp_names = [k for k in experiment_names if experiment_type == k]
        else:
            found_exp_names = experiment_names
        
        print("... getting grouped stats  for experiments: ", found_exp_names)
        if isinstance(found_exp_names, list) and len(found_exp_names) > 0:
            print("loading found experiments")
            if len(found_exp_names) > 1:
                for fi, fname in enumerate(found_exp_names):
                    print fi, fname
                sel = raw_input("Select IDX of exp to use: ")
                if sel == '':
                    expdict = dict((exp_name, self.experiments[exp_name]) for exp_name in found_exp_names)
                else:
                    exp_name = found_exp_names[int(sel)]
                    expdict = {exp_name: self.experiments[exp_name]}
            elif len(found_exp_names) == 1:
                exp_name = found_exp_names[0]
                expdict = {exp_name: self.experiments[exp_name]}
        else:
            # Load just this experiment type:
            print("no exp data saved, loading now...")
            expdict = self.load_data(experiment=experiment_type, traceid=traceid,
                                     trace_type=trace_type, rootdir=rootdir, 
                                     update_self=update, make_equal=make_equal)
            #exp = None if expdict is None else expdict[expdict.keys()[0]] 
        
        if expdict is not None:
            estats_dict = {}
            for exp_name, exp in expdict.items():
                if exp is None:
                    continue
                print("%s: calculating stats" % exp_name)
                #print("... [%s] Loading roi stats and cell list..." % exp.name)
                tmp_estats_dict = exp.get_stats(responsive_test=responsive_test, responsive_thr=responsive_thr,
                                                get_grouped=get_grouped, receptive_field_fit=receptive_field_fit,
                                               make_equal=make_equal)
                if tmp_estats_dict is not None:
                    estats_dict.update(tmp_estats_dict)
                
        return estats_dict
        
    
class Experiment():
    def __init__(self, experiment_type, animalid, session, fov, \
                 traceid='traces001', rootdir='/n/coxfs01/2p-data'):
        print(" [%s | %s | %s] creating experiment object - %s" % (animalid, session, fov, experiment_type))
        self.name = experiment_type
        self.animalid = animalid
        self.session = session
        self.fov = fov
        self.traceid = traceid
        self.rois, tmp_tid = get_roi_id(animalid, session, fov, traceid, run_name=experiment_type, rootdir=rootdir)
        if tmp_tid != self.traceid:
            self.traceid = tmp_tid
            
        paths = self.get_data_paths(rootdir=rootdir)
        self.source = paths
        self.trace_type = None #trace_type
        self.data = Struct() #self.load()
        
        
        if 'gratings' in self.name and int(self.session) < 20190511:
            self.experiment_type = 'rfs'
        else:
            self.experiment_type = self.name
                    
    def get_responsive_cells(self, responsive_test='ROC', responsive_thr=0.01, 
                             receptive_field_fit='zscore0.00_no_trim'):
        
        roi_list=None; rstats=None; nrois_total=None; trials_by_cond=None;
        nattrs = [i for i in dir(self.data) if '__' not in i]
        assert len(nattrs) > 0, "ERROR:  no data loaded."
        #print self.name
        if 'retino' in self.name:
            print("... loading retino data")
            rstats, roi_list, nrois_total, trials_by_cond = get_retino_stats(self.data, responsive_thr=responsive_thr)
        else:
            print("... loading event data")
            rstats, roi_list, nrois_total = get_roi_stats(self.animalid, self.session, self.fov, traceid=self.traceid,
                                             exp_name=self.name, responsive_test=responsive_test, 
                                             receptive_field_fit=receptive_field_fit)
            trials_by_cond = None
        if rstats is not None:
            print("... Found %i responsive cells." % (len(roi_list)))
        
        return rstats, roi_list, nrois_total, trials_by_cond
               
    def get_stats(self, responsive_test='ROC', responsive_thr=0.01, make_equal=True,
                  receptive_field_fit='zscore0.00_no_trim', get_grouped=True):
        print("... [%s] Loading roi stats and cell list..." % self.name)
        rstats, roi_list, nrois_total, trials_by_cond = self.get_responsive_cells(responsive_test=responsive_test,
                                                                 responsive_thr=responsive_thr,
                                                                 receptive_field_fit=receptive_field_fit)
        
        if rstats is None and roi_list is None:
            print("--- NO STATS (%s)" % responsive_test)
            return None
    
    
        if 'gratings' in self.name and int(self.session) < 20190511:
            print("... renaming experiment to 'rfs' (old, pre-20190511)")
            experiment_id = 'rfs'
        else:
            if 'combined' in self.name:
                experiment_id = self.name.split('_')[1]
            else:
                experiment_id = self.name.split('_')[0]
            
        estats = Struct()
        estats.experiment_id = experiment_id
        estats.rois = roi_list
        estats.nrois = nrois_total
        estats.gdf = None
        estats.sdf = None
        
        if 'retino' not in experiment_id:
            self.load(trace_type='dff', make_equal=make_equal)
            estats.gdf = resp.group_roidata_stimresponse(self.data.traces[roi_list], self.data.labels, 
                                                         roi_list=roi_list,
                                                         return_grouped=get_grouped)
            if 'rf' in experiment_id:
                estats.fits = fitrf.rfits_to_df(rstats, roi_list=sorted(roi_list))
                rstats.pop('fits')
                print("*** got rf fits***")
                
                estats.finfo = rstats
            estats.sdf = self.data.sdf
        else:
            estats.gdf = rstats #rstats['magratios'].max(axis=1)
            assert trials_by_cond is not None, "Retino trial data failed to return"
            estats.sdf = trials_by_cond
            
        return {experiment_id: estats}
        
                
    
    def load(self, trace_type='corrected', make_equal=True, rootdir='/n/coxfs01/2p-data'):
        '''
        Populates trace_type and data
        '''
        
        self.trace_type=trace_type
        if not(isinstance(self.source, list)):
            assert os.path.exists(self.source), "File path does not exist! -- %s" % self.source
            print("... loading data array (%s - %s)" % (self.name, os.path.split(self.source)[-1] ))
            try:
                if self.source.endswith('npz'):
                    dset = np.load(self.source)
                    print("... loaded")
                    self.data.traces  = pd.DataFrame(dset[self.trace_type])
                    self.data.labels = pd.DataFrame(data=dset['labels_data'], columns=dset['labels_columns'])
                    print self.data.labels.shape
                    sdf = pd.DataFrame(dset['sconfigs'][()]).T
                    round_sz = [int(round(s)) if s is not None else s for s in sdf['size']]
                    sdf['size'] = round_sz
                    self.data.sdf = sdf
                    self.data.info = dset['run_info'][()]
                    
                    if make_equal:
                        print("... making equal")
                        self.data.traces, self.data.labels = check_counts_per_condition(self.data.traces, self.data.labels)

                elif self.source.endswith('h5'):
                    #dfile = h5py.File(self.source, 'r')
                    # TODO: formatt retino data in sensible way with rutils
                    self.data.info = retinotools.get_protocol_info(self.animalid, self.session, self.fov, run=self.name,
                                                                   rootdir=rootdir)
                    self.data.traces, self.data.labels = retinotools.format_retino_traces(self.source, info=self.data.info)      
            except Exception as e:
                print("ERROR LOADING DATA")
                print e
    
        else:
            print("*** NOT IMPLEMENTED ***\n--%s--" % self.source)

        #return data
    
            
    def get_data_paths(self, rootdir='/n/coxfs01/2p-data'):
        print("... getting data paths")
        fov_dir = os.path.join(rootdir, self.animalid, self.session, self.fov)
        if 'retino' in self.name:
            all_runs = glob.glob(os.path.join(fov_dir, '*%s*' % self.name, 'retino_analysis', 'anaylsis*', 'traces', '*.h5'))
            trace_extraction = 'retino_analysis'
        else:
            all_runs = glob.glob(os.path.join(fov_dir, '*%s_*' % self.name, 'traces', 'traces*', 'data_arrays', '*.npz'))
            trace_extraction = 'traces'
        if len(all_runs) == 0:
            print("[%s|%s|%s] No extracted traces: %s" % (self.animalid, self.session, self.fov, self.name))
            return None
        
        all_runs = [s.split('/%s' % trace_extraction)[0] for s in all_runs]
        combined_runs = [r for r in all_runs if 'combined' in r]
        single_runs = []
        for crun in combined_runs:
            stim_type = re.search('combined_(.+?)_static', os.path.split(crun)[-1]).group(1)
            #print stim_type
            single_runs.extend(glob.glob(os.path.join(fov_dir, '%s_run*' % stim_type)))
        run_list = list(set([r for r in all_runs if r not in single_runs and 'compare' not in r]))
        #print run_list
        
        data_fpaths = []
        for run_dir in run_list:
            run_name = os.path.split(run_dir)[-1]
            print("... ... %s" % run_name)
            try:
                if 'retino' in run_name:
                    # Select analysis ID that corresponds to current ROI set:
                    extraction_name = 'retino_analysis'
                    fpath = get_retino_analysis(self.animalid, self.session, self.fov,\
                                                run=run_name, rois=self.rois, rootdir=rootdir) # retrun extracted raw tracs (.h5)
                    
                else:
                    extraction_name = 'traces'
                    fpath = glob.glob(os.path.join(run_dir, 'traces', '%s*' % self.traceid, \
                                                   'data_arrays', 'datasets.npz'))[0]
                data_fpaths.append(fpath)
            except IndexError:
                print("... no data arrays found for: %s" % run_name)

            data_fpaths = list(set(data_fpaths))
            if len(data_fpaths) > 1:
                data_fpath = [f for f in data_fpaths if self.name in f][0]
#                print("More than 1 file found for %s" % self.name)
#                for fi, fpath in enumerate(data_fpaths):
#                    print fi, fpath
#                sel = raw_input("Select IDX of file path to use: ")
#                if sel=='':
#                    data_fpath = data_fpaths
#                else:
#                    data_fpath = data_fpaths[int(sel)]
            else:
                data_fpath = data_fpaths[0]

            if not isinstance(data_fpath, list):
                corresp_run_name = os.path.split(data_fpath.split('/%s/' % extraction_name)[0])[-1]
                if self.name != corresp_run_name:
                    print("... ... renaming experiment to run name: %s" % corresp_run_name)
                    self.name = corresp_run_name


        return data_fpath



#%% Load Datasets:
#
#def extract_options(options):
#
#    parser = optparse.OptionParser()
#
#    parser.add_option('-D', '--root', action='store', dest='rootdir',
#                          default='/nas/volume1/2photon/data',
#                          help='data root dir (dir w/ all animalids) [default: /nas/volume1/2photon/data, /n/coxfs01/2pdata if --slurm]')
#    parser.add_option('-i', '--animalid', action='store', dest='animalid',
#                          default='', help='Animal ID')
#
#    # Set specific session/run for current animal:
#    parser.add_option('-S', '--session', action='store', dest='session',
#                          default='', help='session dir (format: YYYMMDD_ANIMALID')
#    parser.add_option('-A', '--acq', action='store', dest='acquisition',
#                          default='FOV1', help="acquisition folder (ex: 'FOV1_zoom3x') [default: FOV1]")
#    parser.add_option('-T', '--trace-type', action='store', dest='trace_type',
#                          default='raw', help="trace type [default: 'raw']")
#    parser.add_option('-R', '--run', dest='run_list', default=[], nargs=1,
#                          action='append',
#                          help="run ID in order of runs")
#    parser.add_option('-t', '--traceid', dest='traceid_list', default=[], nargs=1,
#                          action='append',
#                          help="trace ID in order of runs")
#    parser.add_option('-n', '--nruns', action='store', dest='nruns', default=1, help="Number of consecutive runs if combined")
#    parser.add_option('--slurm', action='store_true', dest='slurm', default=False, help="set if running as SLURM job on Odyssey")
#    parser.add_option('--combo', action='store_true', dest='combined', default=False, help="Set if using combined runs with same default name (blobs_run1, blobs_run2, etc.)")
#    parser.add_option('-q', '--quant', action='store', dest='quantile', default=0.08, help="Quantile of trace to include for drift calculation (default: 0.08)")
#
#
#    # Pupil filtering info:
#    parser.add_option('--no-pupil', action="store_false",
#                      dest="filter_pupil", default=True, help="Set flag NOT to filter PSTH traces by pupil threshold params")
#    parser.add_option('-s', '--radius-min', action="store",
#                      dest="pupil_radius_min", default=25, help="Cut-off for smnallest pupil radius, if --pupil set [default: 25]")
#    parser.add_option('-B', '--radius-max', action="store",
#                      dest="pupil_radius_max", default=65, help="Cut-off for biggest pupil radius, if --pupil set [default: 65]")
#    parser.add_option('-d', '--dist', action="store",
#                      dest="pupil_dist_thr", default=5, help="Cut-off for pupil distance from start, if --pupil set [default: 5]")
#
#    (options, args) = parser.parse_args(options)
#
#    return options


        

#%%


#%%

#
#def format_framesXrois(Xdf, Sdf, nframes_on, framerate, trace='raw', verbose=True, missing='drop'):
##def format_framesXrois(sDATA, roi_list, nframes_on, framerate, trace='raw', verbose=True, missing='drop'):
#
#    # Format data: rows = frames, cols = rois
#    #raw_xdata = np.array(sDATA.sort_values(['trial', 'tsec']).groupby(['roi'])[trace].apply(np.array).tolist()).T
#    raw_xdata = np.array(Xdf)
#    
#    # Make data non-negative:
#    if raw_xdata.min() < 0:
#        print "Making data non-negative"
#        raw_xdata = raw_xdata - raw_xdata.min()
#
#    #roi_list = sorted(list(set(sDATA['roi'])), key=natural_keys) #sorted(roi_list, key=natural_keys)
#    roi_list = sorted([r for r in Xdf.columns.tolist() if not r=='index'], key=natural_keys) #sorted(roi_list, key=natural_keys)
#    Xdf = pd.DataFrame(raw_xdata, columns=roi_list)
#
#    # Calculate baseline for RUN:
#    # decay_constant = 71./1000 # in sec -- this is what Romano et al. bioRxiv 2017 do for Fsmooth (decay_constant of indicator * 40)
#    # vs. Dombeck et al. Neuron 2007 methods (15 sec +/- tpoint 8th percentile)
#    
#    window_size_sec = (nframes_on/framerate) * 4 # decay_constant * 40
#    decay_frames = window_size_sec * framerate # decay_constant in frames
#    window_size = int(round(decay_frames))
#    quantile = 0.08
#    
#    Fsmooth = Xdf.apply(rolling_quantile, args=(window_size, quantile))
#    Xdata_tmp = (Xdf - Fsmooth)
#    Xdata = np.array(Xdata_tmp)
#    
##    fig, axes = pl.subplots(2,1, figsize=(20,5))
##    axes[0].plot(raw_xdata[0:nframes_per_trial*20, 0], label='raw')
##    axes[0].plot(fsmooth.values[0:nframes_per_trial*20, 0], label='baseline')
##    axes[1].plot(Xdata[0:nframes_per_trial*20,0], label='Fmeasured')
#    
#    
##    # Get rid of "bad rois" that have np.nan on some of the trials:
##    # NOTE:  This is not exactly the best way, but if the df/f trace is wild, np.nan is set for df value on that trial
##    # Since this is done in traces/get_traces.py, for now, just deal with this by ignoring ROI
##    bad_roi = None
##    if missing == 'drop':
##        ix, iv = np.where(np.isnan(Xdata))
##        bad_roi = list(set(iv))
##        if len(bad_roi) == 0:
##            bad_roi = None
##
##    if bad_roi is not None:
##        Xdata = np.delete(Xdata, bad_roi, 1)
##        roi_list = [r for ri,r in enumerate(roi_list) if ri not in bad_roi]
#
#    #tsecs = np.array(sDATA.sort_values(['trial', 'tsec']).groupby(['roi'])['tsec'].apply(np.array).tolist()).T
#    tsecs = np.array(Sdf['tsec'].values)
##    if bad_roi is not None:
##        tsecs = np.delete(tsecs, bad_roi, 1)
#
#    # Get labels: # only need one col, since trial id same for all rois
#    ylabels = np.array(sDATA.sort_values(['trial', 'tsec']).groupby(['roi'])['config'].apply(np.array).tolist()).T[:,0]
#    groups = np.array(sDATA.sort_values(['trial']).groupby(['roi'])['trial'].apply(np.array).tolist()).T[:,0]
#
#    if verbose:
#        print "-------------------------------------------"
#        print "Formatting summary:"
#        print "-------------------------------------------"
#        print "X:", Xdata.shape
#        print "y (labels):", ylabels.shape
#        print "N groupings of trials:", len(list(set(groups)))
#        print "N samples: %i, N features: %i" % (Xdata.shape[0], Xdata.shape[1])
#        print "-------------------------------------------"
#
#    return Xdata, ylabels, groups, tsecs, Fsmooth # roi_list, Fsmooth

#%%
def format_roisXvalue(Xdata, run_info, fsmooth=None, sorted_ixs=None, value_type='meanstim', trace='raw'):

    #if isinstance(Xdata, pd.DataFrame):
    Xdata = np.array(Xdata)
        
    # Make sure that we only get ROIs in provided list (we are dropping ROIs w/ np.nan dfs on any trials...)
    #sDATA = sDATA[sDATA['roi'].isin(roi_list)]
    stim_on_frame = run_info['stim_on_frame']
    nframes_on = int(round(run_info['nframes_on']))
    ntrials_total = run_info['ntrials_total']
    nframes_per_trial = run_info['nframes_per_trial']
    nrois = Xdata.shape[-1] #len(run_info['roi_list'])
    
    if sorted_ixs is None:
        print "Trials are sorted by time of occurrence, not stimulus type."
        sorted_ixs = xrange(ntrials_total) # Just sort in trial order

    #trace = 'raw'
    traces = np.reshape(Xdata, (ntrials_total, nframes_per_trial, nrois), order='C')
    traces = traces[sorted_ixs,:,:]
    #rawtraces = np.vstack((sDATA.groupby(['roi', 'config', 'trial'])[trace].apply(np.array)).as_matrix())

    
#    if value_type == 'meanstimdff' and fsmooth is not None:
#        dftraces = np.array(Xdata/fsmooth)
#        dftraces = np.reshape(dftraces, (ntrials_total, nframes_per_trial, nrois), order='C')
#        dftraces = dftraces[sorted_ixs,:,:]
#        mean_stim_dff_values = np.nanmean(dftraces[:, stim_on_frame:stim_on_frame+nframes_on], axis=1)

    std_baseline_values = np.nanstd(traces[:, 0:stim_on_frame], axis=1)
    mean_baseline_values = np.nanmean(traces[:, 0:stim_on_frame], axis=1)
    mean_stim_on_values = np.nanmean(traces[:, stim_on_frame:stim_on_frame+nframes_on], axis=1)
    

    #zscore_values_raw = np.array([meanval/stdval for (meanval, stdval) in zip(mean_stim_on_values, std_baseline_values)])
    if value_type == 'zscore':
        values_df = (mean_stim_on_values - mean_baseline_values ) / std_baseline_values
    elif value_type == 'meanstim':
        values_df = mean_stim_on_values #- mean_baseline_values ) / std_baseline_values
#    elif value_type == 'meanstimdff':
#        values_df = mean_stim_dff_values #- mean_baseline_values ) / std_baseline_values
        
    #rois_by_value = np.reshape(values_df, (nrois, ntrials_total))
        
#    if bad_roi is not None:
#        rois_by_zscore = np.delete(rois_by_zscore, bad_roi, 0)

    return values_df #rois_by_value


#%%
        
#% Get mean trace for each condition:
def get_mean_cond_traces(ridx, Xdata, ylabels, tsecs, nframes_per_trial):
    '''For each ROI, get average trace for each condition.
    '''
    if isinstance(ylabels[0], str):
        conditions = sorted(list(set(ylabels)), key=natural_keys)
    else:
        conditions = sorted(list(set(ylabels)))

    mean_cond_traces = []
    mean_cond_tsecs = []
    for cond in conditions:
        ixs = np.where(ylabels==cond)                                          # Get sample indices for current condition
        curr_trace = np.squeeze(Xdata[ixs, ridx])                              # Grab subset of sample data 
        ntrials_in_cond = curr_trace.shape[0]/nframes_per_trial                # Identify the number of trials for current condition
        
        # Reshape both traces and corresponding time stamps:  
        # Shape (ntrials, nframes) to get average:
        curr_tracemat = np.reshape(curr_trace, (ntrials_in_cond, nframes_per_trial))
        curr_tsecs = np.reshape(np.squeeze(tsecs[ixs,ridx]), (ntrials_in_cond, nframes_per_trial))

        mean_ctrace = np.mean(curr_tracemat, axis=0)
        mean_cond_traces.append(mean_ctrace)
        mean_tsecs = np.mean(curr_tsecs, axis=0)
        mean_cond_tsecs.append(mean_tsecs)

    mean_cond_traces = np.array(mean_cond_traces)
    mean_cond_tsecs = np.array(mean_tsecs)
    #print mean_cond_traces.shape
    return mean_cond_traces, mean_cond_tsecs


#%%

def get_xcond_dfs(roi_list, X, y, tsecs, run_info):
    nconds = len(run_info['condition_list'])
    averages_list = []
    normed_list = []
    for ridx, roi in enumerate(sorted(roi_list, key=natural_keys)):
        mean_cond_traces, mean_tsecs = get_mean_cond_traces(ridx, X, y, tsecs, run_info['nframes_per_trial']) #get_mean_cond_traces(ridx, X, y)
        xcond_mean = np.mean(mean_cond_traces, axis=0)
        normed = mean_cond_traces - xcond_mean

        averages_list.append(pd.DataFrame(data=np.reshape(mean_cond_traces, (nconds*run_info['nframes_per_trial'],)),
                                        columns = [roi],
                                        index=np.array(range(nconds*run_info['nframes_per_trial']))
                                        ))

        normed_list.append(pd.DataFrame(data=np.reshape(normed, (nconds*run_info['nframes_per_trial'],)),
                                        columns = [roi],
                                         index=np.array(range(nconds*run_info['nframes_per_trial']))
                                        ))
    return averages_list, normed_list


#%% Visualization:

# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import imutils

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def uint16_to_RGB(img):
    im = img.astype(np.float64)/img.max()
    im = 255 * im
    im = im.astype(np.uint8)
    rgb = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    return rgb

def sort_rois_2D(traceid_dir):

    run_dir = traceid_dir.split('/traces')[0]
    acquisition_dir = os.path.split(run_dir)[0]; acquisition = os.path.split(acquisition_dir)[1]
    session_dir = os.path.split(acquisition_dir)[0]; session = os.path.split(session_dir)[1]
    animalid = os.path.split(os.path.split(session_dir)[0])[1]
    rootdir = session_dir.split('/%s' % animalid)[0]

    # Load formatted mask file:
    mask_fpath = os.path.join(traceid_dir, 'MASKS.hdf5')
    maskfile =h5py.File(mask_fpath, 'r')

    # Get REFERENCE file (file from which masks were made):
    mask_src = maskfile.attrs['source_file']
    if rootdir not in mask_src:
        mask_src = replace_root(mask_src, rootdir, animalid, session)
    tmp_msrc = h5py.File(mask_src, 'r')
    ref_file = tmp_msrc.keys()[0]
    tmp_msrc.close()

    # Load masks and reshape to 2D:
    if ref_file not in maskfile.keys():
        ref_file = maskfile.keys()[0]
    masks = np.array(maskfile[ref_file]['Slice01']['maskarray'])
    dims = maskfile[ref_file]['Slice01']['zproj'].shape
    masks_r = np.reshape(masks, (dims[0], dims[1], masks.shape[-1]))
    print "Masks: (%i, %i), % rois." % (masks_r.shape[0], masks_r.shape[1], masks_r.shape[-1])

    # Load zprojection image:
    zproj = np.array(maskfile[ref_file]['Slice01']['zproj'])


    # Cycle through all ROIs and get their edges
    # (note:  tried doing this on sum of all ROIs, but fails if ROIs are overlapping at all)
    cnts = []
    for ridx in range(masks_r.shape[-1]):
        im = masks_r[:,:,ridx]
        im[im>0] = 1
        im[im==0] = np.nan #1
        im = im.astype('uint8')
        edged = cv2.Canny(im, 0, 0.9)
        tmp_cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        tmp_cnts = tmp_cnts[0] if imutils.is_cv2() else tmp_cnts[1]
        cnts.append((ridx, tmp_cnts[0]))
    print "Created %i contours for rois." % len(cnts)

    # Sort ROIs b y x,y position:
    sorted_cnts =  sorted(cnts, key=lambda ctr: (cv2.boundingRect(ctr[1])[1] + cv2.boundingRect(ctr[1])[0]) * zproj.shape[1] )
    cnts = [c[1] for c in sorted_cnts]
    sorted_rids = [c[0] for c in sorted_cnts]

    return sorted_rids, cnts, zproj

#
def plot_roi_contours(zproj, sorted_rids, cnts, clip_limit=0.008, label=True, label_rois=[],
                          draw_box=False, thickness=1, roi_color=(0, 255, 0), transform=False,
                          single_color=False, ax=None):

    # Create ZPROJ img to draw on:
    refRGB = uint16_to_RGB(zproj)

    # Use some color map to indicate distance from upper-left corner:
    sorted_colors = sns.color_palette("Spectral", len(sorted_rids)) #masks.shape[-1])

    if ax is None:
        fig, ax = pl.subplots(1, figsize=(10,10))
        
    
    
#    p2, p98 = np.percentile(refRGB, (1, 99))
#    img_rescale = exposure.rescale_intensity(refRGB, in_range=(p2, p98))
    im_adapthist = exposure.equalize_adapthist(refRGB, clip_limit=clip_limit)
    im_adapthist *= 256
    im_adapthist= im_adapthist.astype('uint8')
    ax.imshow(im_adapthist) #pl.figure(); pl.imshow(refRGB) # cmap='gray')

    refObj = None
    orig = im_adapthist.copy()
    distances = []
    # loop over the contours individually
    for cidx, (rid, cnt) in enumerate(zip(sorted_rids, cnts)):

        # compute the rotated bounding box of the contour
        box = cv2.minAreaRect(cnt)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")

    	# order the points in the contour such that they appear
    	# in top-left, top-right, bottom-right, and bottom-left order
        box = perspective.order_points(box)

    	# compute the center of the bounding box
        cX = np.average(box[:, 0])
        cY = np.average(box[:, 1])

        # if this is the first contour we are examining (i.e.,
        # the left-most contour), we presume this is the
        # reference object
        if refObj is None:
            # unpack the ordered bounding box, then compute the
            # midpoint between the top-left and top-right points,
            # followed by the midpoint between the top-right and
            # bottom-right
            (tl, tr, br, bl) = box
            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)

            # compute the Euclidean distance between the midpoints,
            # then construct the reference object
            D = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
            refObj = (cidx, (cX, cY)) #(box, (cX, cY), D) # / args["width"])
            continue

        # draw the contours on the image
        #orig = refRGB.copy()
        if len(label_rois) > 1 and rid not in label_rois:
            col255 = 0
        else:
            if single_color:
                col255 = roi_color
            else:
                col255 = tuple([cval*255 for cval in sorted_colors[cidx]])
                
        if draw_box:
            cv2.drawContours(orig, [box.astype("int")], -1, col255, thickness)
        else:
            cv2.drawContours(orig, cnt, -1, col255, thickness)
            
        if label:
            cv2.putText(orig, str(rid+1), cv2.boundingRect(cnt)[:2], cv2.FONT_HERSHEY_COMPLEX, .5, [0])
        
        if transform:
            img = imutils.rotate(orig, 90)  
            #imageROI = orig.copy()
            #img = imutils.rotate_bound(imageROI, -90)
            
            ax.imshow(img)
            ax.invert_xaxis()
        else:
            ax.imshow(orig)
        
        
        # stack the reference coordinates and the object coordinates
        # to include the object center
        refCoords = refObj[1] #np.vstack([refObj[0], refObj[1]])
        objCoords = (cX, cY) #np.vstack([box, (cX, cY)])

        D = dist.euclidean((cX, cY), (refCoords[0], refCoords[1])) #/ refObj[2]
        distances.append(D)

    pl.axis('off')
    
    return ax
    
    
#
def psth_from_full_trace(roi, tracevec, mean_tsecs, nr, nc,
                                  color_codes=None, orientations=None,
                                  stim_on_frame=None, nframes_on=None,
                                  plot_legend=True, plot_average=True, as_percent=False,
                                  roi_psth_dir='/tmp', save_and_close=True):

    '''Pasre a full time-series (of a given run) and plot as stimulus-aligned
    PSTH for a given ROI.
    '''

    pl.figure()
    traces = np.reshape(tracevec, (nr, nc))

    if as_percent:
        multiplier = 100
        units_str = ' (%)'
    else:
        multiplier = 1
        units_str = ''

    if color_codes is None:
        color_codes = sns.color_palette("Greys_r", nr*2)
        color_codes = color_codes[0::2]
    if orientations is None:
        orientations = np.arange(0, nr)

    for c in range(traces.shape[0]):
        pl.plot(mean_tsecs, traces[c,:] * multiplier, c=color_codes[c], linewidth=2, label=orientations[c])

    if plot_average:
        pl.plot(mean_tsecs, np.mean(traces, axis=0)*multiplier, c='r', linewidth=2.0)
    sns.despine(offset=4, trim=True)

    if stim_on_frame is not None and nframes_on is not None:
        stimbar_loc = traces.min() - (0.1*traces.min()) #8.0

        stimon_frames = mean_tsecs[stim_on_frame:stim_on_frame + nframes_on]
        pl.plot(stimon_frames, stimbar_loc*np.ones(stimon_frames.shape), 'g')

    pl.xlabel('tsec')
    pl.ylabel('mean df/f%s' % units_str)
    pl.title(roi)

    if plot_legend:
        pl.legend(orientations)

    if save_and_close:
        pl.savefig(os.path.join(roi_psth_dir, '%s_psth_mean.png' % roi))
        pl.close()
