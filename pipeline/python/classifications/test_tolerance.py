#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 12:24:04 2018

@author: juliana
"""

import os
import json
import glob
import sys
import optparse
import cPickle as pkl
import copy

import numpy as np
import pandas as pd
from scipy import stats

import math

import tifffile as tf
import pylab as pl
import seaborn as sns
from pipeline.python.classifications import linearSVC_class as cls

from pipeline.python.utils import replace_root, label_figure, natural_keys

#%%

def extract_options(options):

    def comma_sep_list(option, opt, value, parser):
        setattr(parser.values, option.dest, value.split(','))


    parser = optparse.OptionParser()

    parser.add_option('-D', '--root', action='store', dest='rootdir',
                          default='/nas/volume1/2photon/data',
                          help='data root dir (dir w/ all animalids) [default: /nas/volume1/2photon/data, /n/coxfs01/2pdata if --slurm]')
    parser.add_option('-i', '--animalid', action='store', dest='animalid',
                          default='', help='Animal ID')

    # Set specific session/run for current animal:
#    parser.add_option('-S', '--session', action='store', dest='session',
#                          default='', help='session dir (format: YYYMMDD_ANIMALID')
#    parser.add_option('-A', '--acq', action='store', dest='acquisition',
#                          default='FOV1', help="acquisition folder (ex: 'FOV1_zoom3x') [default: FOV1]")
#    parser.add_option('-R', '--run', action='store', dest='run',
#                          default='', help="RUN name (e.g., gratings_run1)")
#    parser.add_option('-t', '--traceid', action='store', dest='traceid',
#                          default='', help="traceid name (e.g., traces001)")

    # Run specific info:
    parser.add_option('-S', '--session', dest='session_list', default=[], type='string', action='callback', callback=comma_sep_list, help="SESSIONS for corresponding runs [default: []]")

    parser.add_option('-A', '--fov', dest='fov_list', default=[], type='string', action='callback', callback=comma_sep_list, help="FOVs for corresponding runs [default: []]")

    parser.add_option('-t', '--traceid', dest='traceid_list', default=[], type='string', action='callback', callback=comma_sep_list, help="TRACEIDs for corresponding runs [default: []]")

    parser.add_option('-R', '--run', dest='run_list', default=[], type='string', action='callback', callback=comma_sep_list, help='list of run IDs [default: []')
    parser.add_option('-p', '--indata_type', action='store', dest='inputdata_type', default='corrected', help="data processing type (dff, corrected, raw, etc.)")


    parser.add_option('--slurm', action='store_true', dest='slurm', default=False, help="set if running as SLURM job on Odyssey")
    parser.add_option('--par', action='store_true', dest='multiproc', default=False, help="set if want to run MP on roi stats, when possible")
    parser.add_option('--nproc', action='store', dest='nprocesses', default=4, help="N processes if running in par (default=4)")

    # Classifier info:
    parser.add_option('-r', '--rois', action='store', dest='roi_selector', default='all', help="(options: all, visual)")
    parser.add_option('-d', '--dtype', action='store', dest='data_type', default='stat', help="(options: frames, stat)")

    stat_choices = {'stat': ['meanstim', 'meanstimdff', 'zscore'],
                    'frames': ['trial', 'stimulus', 'post']}
    parser.add_option('-s', '--stype', action='store', dest='stat_type', default='meanstim', 
                      help="If dtype is STAT, options: %s. If dtype is FRAMES, options: %s" % (str(stat_choices['stat']), str(stat_choices['frames'])))

    parser.add_option('-N', '--name', action='store', dest='class_name', default='', help='Name of transform to classify (e.g., ori, xpos, morphlevel, etc.)')

    parser.add_option('-V', '--area', action='store', dest='visual_area', default='', help='Name of visual area (e.g., LI, LL, etc.)')
    parser.add_option('--segment', action='store_true', dest='select_visual_area', default=False, help="set if selecting subset of FOV for visual area")

    
    (options, args) = parser.parse_args(options)
    
    assert options.stat_type in stat_choices[options.data_type], "Invalid STAT selected for data_type %s. Run -h for options." % options.data_type

    return options


#%%

def get_stat_samples(Xdata, labels_df, stat_type='meanstim', multiple_durs=True, return_type=None):
    print "Trials are sorted by time of occurrence, not stimulus type."

    ntrials_total = len(labels_df['trial'].unique())
    ntrials_per_cond = [len(t) for t in labels_df.groupby('config')['trial'].unique()]
    #assert len(np.unique(ntrials_per_cond)) == 1, "Uneven reps per condition! %s" % str(ntrials_per_cond)
    ntrials = np.unique(ntrials_per_cond)[0]
    
    # Get baseline and stimulus indices for each trial:
    sample_labels = []
    stim_on_frame = labels_df['stim_on_frame'].unique()
    assert len(stim_on_frame) == 1, "More than 1 stim on frame found! %s" % str(stim_on_frame)
    stim_on_frame = stim_on_frame[0]
    if multiple_durs:
        tgroups = labels_df.groupby('trial')            
        #stim_durs = sorted(labels_df['stim_dur'].unique()) # longer stim durs will match with longer nframes on
        #nframes_on = sorted(labels_df['nframes_on'].unique())
        std_baseline_values=[]; mean_baseline_values=[]; mean_stimulus_values=[];
        std_stimulus_values = []
        for k,g in tgroups:
            curr_nframes_on = g['nframes_on'].unique()[0]

            curr_baseline_stds = np.nanstd(Xdata[g['tsec'][0:stim_on_frame].index.tolist(), :], axis=0)
            curr_baseline_means = np.nanmean(Xdata[g['tsec'][0:stim_on_frame].index.tolist(), :], axis=0)

            curr_stimulus_means = np.nanmean(Xdata[g['tsec'][stim_on_frame:stim_on_frame+curr_nframes_on].index.tolist(), :], axis=0)
            curr_stimulus_stds = np.nanstd(Xdata[g['tsec'][stim_on_frame:stim_on_frame+curr_nframes_on].index.tolist(), :], axis=0)
            
            std_baseline_values.append(curr_baseline_stds)
            mean_baseline_values.append(curr_baseline_means)
            mean_stimulus_values.append(curr_stimulus_means)
            std_stimulus_values.append(curr_stimulus_stds)
            
            curr_config = g['config'].unique()[0]
            sample_labels.append(curr_config)
        
        mean_stim_on_values = np.vstack(mean_stimulus_values)
        std_stim_on_values = np.vstack(std_stimulus_values)
        
        mean_baseline_values = np.vstack(mean_baseline_values)
        std_baseline_values = np.vstack(std_baseline_values)
        
    else:
        nrois = Xdata.shape[-1]
        nframes_per_trial = Xdata.shape[0] / ntrials_total
        nframes_on = labels_df['nframes_on'].unique()[0]
        
        traces = np.reshape(Xdata, (ntrials_total, nframes_per_trial, nrois), order='C')
        std_baseline_values = np.nanstd(traces[:, 0:stim_on_frame], axis=1)
        mean_baseline_values = np.nanmean(traces[:, 0:stim_on_frame], axis=1)
        mean_stim_on_values = np.nanmean(traces[:, stim_on_frame:stim_on_frame+nframes_on], axis=1)
        std_stim_on_values = np.nanstd(traces[:, stim_on_frame:stim_on_frame+nframes_on], axis=1)
        
    if stat_type == 'zscore':
        sample_array = (mean_stim_on_values - mean_baseline_values ) / std_baseline_values
    elif stat_type == 'meanstimdff':
        sample_array = (mean_stim_on_values - mean_baseline_values ) / mean_baseline_values
    else:
        sample_array = mean_stim_on_values
    
    std_array = std_stim_on_values
#    if clfparams['get_null']:
#        random_draw = True
#        print "Stim values:", sample_array.shape
#        if random_draw:
#            selected_trial_ixs = random.sample(range(0, mean_baseline_values.shape[0]), ntrials)
#        bas = mean_baseline_values[selected_trial_ixs, :]
#    
#        sample_array = np.append(sample_array, bas, axis=0)
#        print "Added null cases:", sample_array.shape
#        sample_labels.extend(['bas' for _ in range(bas.shape[0])])
    if return_type is not None:
        if return_type == 'std':
            return sample_array, std_array, np.array(sample_labels)
        elif return_type == 'bas':
            return sample_array, mean_baseline_values, np.array(sample_labels)
    
    else:
        return sample_array, np.array(sample_labels)

#%
class DataSet():
    def __init__(self, animalid, session, acquisition, run, traceid, rootdir='/n/coxfs01/2p-data',
                         roi_selector='visual', data_type='stat', stat_type='meanstim',
                         inputdata_type='corrected', class_name=''):
        
        self.rootdir = rootdir
        self.animalid = animalid
        self.session = session
        self.acquisition = acquisition
        self.run = run
        if 'cnmf' in traceid:
            tracedir_type = 'cnmf'
        else:
            tracedir_type = 'traces'
            
        tmp_traceid_dir = sorted(glob.glob(os.path.join(rootdir, animalid, 
                                              session, acquisition, 
                                              run, tracedir_type, 
                                              '%s*' % traceid)), key=natural_keys)
        if len(tmp_traceid_dir) > 1:
            print "Found multiple trace IDs:"
            for ti, tidir in enumerate(tmp_traceid_dir):
                print ti, tidir
            sel = input("Select IDX of tid dir to use: ")
            traceid_dir = tmp_traceid_dir[sel]
        else:
            assert len(tmp_traceid_dir) == 1, "Not TRACEIDs found!"
            traceid_dir = tmp_traceid_dir[0]
        self.traceid_dir = traceid_dir
        
        self.traceid = os.path.split(self.traceid_dir)[-1]        
        self.data_fpath = self.get_data_fpath()
        

        self.data_identifier = ''
        self.inputdata_type = inputdata_type
        self.data_type = data_type
        self.stat_type = stat_type
        self.roi_selector = roi_selector
        self.data = None
        self.bas= None
        self.std = None
        
        self.labels = None
        self.run_info = None
        self.sconfigs = None
        self.rois = None
        
    def get_data_fpath(self):

        # Data array dir:
        data_basedir = os.path.join(self.traceid_dir, 'data_arrays')
        data_fpath = os.path.join(data_basedir, 'datasets.npz')
        assert os.path.exists(data_fpath), "[E]: Data array not found! Did you run tifs_to_data_arrays.py?"
    
        # Create output base dir for classifier results:
        clf_basedir = os.path.join(self.traceid_dir, 'classifiers')
        if not os.path.exists(os.path.join(clf_basedir, 'figures')):
            os.makedirs(os.path.join(clf_basedir, 'figures'))
            
        return data_fpath

            
    def load_roi_list(self, roi_selector='visual'):
        
        if roi_selector == 'all':
            roi_list = None
        else:
            roistats_results_fpath = os.path.join(self.traceid_dir, 'sorted_rois', 'roistats_results.npz')
            roistats = np.load(roistats_results_fpath)
            
            roi_subset_type = 'sorted_%s' % roi_selector
            roi_list = roistats[roi_subset_type]
        
        return roi_list
    
    def load_dataset(self, visual_area_info=None, return_type=None):
        print "------------ Loading dataset."

        # Store DATASET:            
        dt = np.load(self.data_fpath)
        if 'arr_0' in dt.keys():
            dataset = dt['arr_0'][()]
        else:
            dataset = dt           
            
        self.dataset = dataset
        
        # Store run info:
        if isinstance(self.dataset['run_info'], dict):
            self.run_info = self.dataset['run_info']
        else:
            self.run_info = self.dataset['run_info'][()]

        
        # Store stim configs:
        if isinstance(self.dataset['sconfigs'], dict):
            orig_sconfigs = self.dataset['sconfigs']
        else:
            orig_sconfigs = self.dataset['sconfigs'][()]

        # Make sure numbers are rounded:
        for cname, cdict in orig_sconfigs.items():
            for stimkey, stimval in cdict.items():
                if isinstance(stimval, (int, float)):
                    orig_sconfigs[cname][stimkey] = round(stimval, 1)
                    
        # Add combined 'position' variable to stim configs if class_name == 'position:
        for cname, config in orig_sconfigs.items():
            pos = '_'.join([str(config['xpos']), str(config['ypos'])])
            config.update({'position': pos})
            
                
        if int(self.session) < 20180602:
            # Rename morphs:
            update_configs = [cfg for cfg, info in orig_sconfigs.items() if info['morphlevel'] > 0]
            for cfg in update_configs:
                if orig_sconfigs[cfg]['morphlevel'] == 6:
                    orig_sconfigs[cfg]['morphlevel'] = 27
                elif orig_sconfigs[cfg]['morphlevel'] == 11:
                    orig_sconfigs[cfg]['morphlevel'] = 53
                elif orig_sconfigs[cfg]['morphlevel'] == 16:
                    orig_sconfigs[cfg]['morphlevel'] = 79
                elif orig_sconfigs[cfg]['morphlevel'] == 22:
                    orig_sconfigs[cfg]['morphlevel'] = 106
                else:
                    print "Unknown morphlevel converstion: %i" % orig_sconfigs[cfg]['morphlevel']
        self.sconfigs = orig_sconfigs

        self.data_identifier = '_'.join((self.animalid, self.session, self.acquisition, self.run, self.traceid))
        
        if visual_area_info is not None:
            (visual_area, visual_areas_fpath), = visual_area_info.items()
            print "Getting ROIs for area: %s" % visual_area
            print "Loading file:", visual_areas_fpath
            with open(visual_areas_fpath, 'rb') as f:
                areas = pkl.load(f)
            if visual_area not in areas.regions.keys():
                print "Specified visual area - %s - NOT FOUND."
                for vi, va in enumerate(areas.regions.keys()):
                    print vi, va
                sel = input("Select IDX of area to use: ")
                visual_area = areas.regions.keys()[sel]
            
            included_rois = [int(ri) for ri in areas.regions[visual_area]['included_rois']]
        else:
            included_rois = None
        
        if return_type is None:
            self.data, self.labels = self.get_formatted_data(included_rois=included_rois, return_type=return_type)
        elif return_type == 'std':
            self.data, self.std, self.labels = self.get_formatted_data(included_rois=included_rois, return_type=return_type)
        elif return_type == 'bas':
            self.data, self.bas, self.labels = self.get_formatted_data(included_rois=included_rois, return_type=return_type)



    def get_formatted_data(self, included_rois=None, return_type=None): #get_training_data(self):
        '''
        Returns input data formatted as:
            ntrials x nrois (data_type=='stat')
            nframes x nrois (data_type = 'frames')
        Filters nrois by roi_selector.
        '''
        print "------------ Formatting data into samples."

        # Get data array:
        Xdata = np.array(self.dataset[self.inputdata_type])
            
        selected_rois = self.load_roi_list(roi_selector=self.roi_selector)
        if included_rois is not None:
            print "---> only including specified ROIs from visual area."
            roi_list = cls.intersection(selected_rois, included_rois)
        else:
            roi_list = selected_rois
            
        # Get subset of ROIs, if roi_selector is not 'all':
        self.rois = np.array(roi_list)
        if self.rois is not None:
            print "Selecting %i out of %i ROIs (selector: %s)" % (len(self.rois), Xdata.shape[-1], self.roi_selector)
            Xdata = np.squeeze(Xdata[:, self.rois])
        
        # Determine whether all trials have the same structure or not:
        multiple_durs = isinstance(self.run_info['nframes_on'], list)

        # Make sure all conds have same N trials:
        ntrials_by_cond = self.run_info['ntrials_by_cond']
        ntrials_tmp = list(set([v for k, v in ntrials_by_cond.items()]))
        #assert len(ntrials_tmp)==1, "Unequal reps per condition!"
        labels_df = pd.DataFrame(data=self.dataset['labels_data'], columns=self.dataset['labels_columns'])
        
        if self.data_type == 'stat':
            cX, bas, cy = get_stat_samples(Xdata, labels_df, self.stat_type, multiple_durs=multiple_durs, return_type=return_type)
        else:
            cX, cy = cls.get_frame_samples(Xdata, labels_df, self.params)
            
        print "Ungrouped dataset cX:", cX.shape
        print "Ungrouped dataset labels cy:", cy.shape
        
        return cX, bas, cy
    



def calculate_relative_dprime(df, sdf, ref_obj, flank_obj, tested_transforms, stat_type='meanstim', object_type='ori'):
    d_values = {}
    g_sdf = sdf.groupby(tested_transforms)
    for k,g in g_sdf:
        if isinstance(k, (float, int)):
            mean_ref = df[ ( (df[object_type]==ref_obj) & (df[tested_transforms[0]]==k) ) ][stat_type].mean()
            std_ref = df[ ( (df[object_type]==ref_obj) & (df[tested_transforms[0]]==k) ) ][stat_type].std()
            mean_flank = df[ ( (df[object_type]==flank_obj) & (df[tested_transforms[0]]==k) ) ][stat_type].mean()
            std_flank = df[ ( (df[object_type]==flank_obj) & (df[tested_transforms[0]]==k) ) ][stat_type].std()
               
            
        elif len(k) == 2:
            mean_ref = df[ ( (df[object_type]==ref_obj) & (df[tested_transforms[0]]==k[0]) & (df[tested_transforms[1]]==k[1]) ) ][stat_type].mean()
            std_ref = df[ ( (df[object_type]==ref_obj) & (df[tested_transforms[0]]==k[0]) & (df[tested_transforms[1]]==k[1]) ) ][stat_type].std()
            mean_flank = df[ ( (df[object_type]==flank_obj) & (df[tested_transforms[0]]==k[0]) & (df[tested_transforms[1]]==k[1]) ) ][stat_type].mean()
            std_flank = df[ ( (df[object_type]==flank_obj) & (df[tested_transforms[0]]==k[0]) & (df[tested_transforms[1]]==k[1]) ) ][stat_type].std()

        dprime = (mean_ref - mean_flank) / np.mean((std_ref, std_flank))
        
        d_values[k] = dprime
            
    return d_values
        

def get_dprime(df, sdf, tested_transforms, roi_id=-1, stat_type='meanstim', object_type='ori'):
    grouper = copy.copy(tested_transforms)
    grouper.extend([object_type])
    
    grouped = df.groupby(grouper).mean()
    
    resp_max = grouped.apply(np.argmax)[D.stat_type]
    resp_min = grouped.apply(np.argmin)[D.stat_type]
    
    ref_obj = resp_max[grouper.index(object_type)]
    
    no_x = False; no_y = False;
    
    if 'xpos' not in tested_transforms:
        ref_x = list(set(sdf['xpos']))[0]
        no_x = True
    else:
        ref_x = resp_max[grouper.index('xpos')]
    if 'ypos' not in tested_transforms:
        ref_y = list(set(sdf['ypos']))[0]
        no_y = True
    else:
        ref_y = resp_max[grouper.index('ypos')]
    
    if no_x and no_y and len(tested_transforms) > 0:
        if 'size' in tested_transforms:
            ref_size = resp_max[grouper.index('size')]
        else:
            ref_size = list(set(sdf['size']))[0]
            
    
    flank_obj = resp_min[grouper.index(object_type)]
    
    print "ROI %i:  best/worst object: %i/%i (refX, refY) = (%.1f, %.1f)" % (roi_id, ref_obj, flank_obj, ref_x, ref_y)
    if ref_obj == flank_obj:
        print "-- warning:  Best and worst are the SAME object..."
        return None
        
    d_values = calculate_relative_dprime(df, sdf, ref_obj, flank_obj, tested_transforms, object_type=object_type, stat_type=stat_type)

    return d_values


def assign_configs(row, trans, sdf):
    print row
    return sdf.loc[row][trans]

#%%
#rootdir = '/n/coxfs01/2p-data' #-data'
#LI_options = ['-D', rootdir, 
#              '-i', 'JC022', 
#           '-S', '20181005', 
#           '-A', 'FOV3_zoom2p7x',
#           '-R', 'combined_gratings_static',
#           '-t', 'traces001',
#           '-r', 'visual', 
#           '-s', 'meanstim',
#           '-p', 'corrected',
#           '-V', 'LI',
##           '--segment',
#           '--nproc=1'
#           ]
#
#LI_options = ['-D', rootdir, 
#              '-i', 'JC015', 
#           '-S', '20180925', 
#           '-A', 'FOV1_zoom2p0x',
#           '-R', 'combined_gratings_static',
#           '-t', 'traces003',
#           '-r', 'visual', 
#           '-s', 'meanstim',
#           '-p', 'corrected',
#           '-V', 'LI',
#           '--segment',
#           '--nproc=1'
#           ]

#LM_options = ['-D', rootdir, '-i', 'JC015', 
#           '-S', '20180925', 
#           '-A', 'FOV1_zoom2p0x',
#           '-R', 'combined_gratings_static',
#           '-t', 'traces003',
#           '-r', 'visual', 
#           '-s', 'meanstim',
#           '-p', 'corrected',
#           '-V', 'LM',
#           '--segment',
#           '--nproc=1'
#           ]

#%%
    

color_dict = {'V1': 'b',
              'LM': 'g',
              'LI': 'm',
              'LL': 'o'}
dprime_by_area = dict((vkey, {}) for vkey in color_dict.keys())

#%%


rootdir = '/n/coxfs01/2p-data' #-data'


#LM_options = ['-D', rootdir, '-i', 'JC015', 
#           '-S', '20180919', 
#           '-A', 'FOV1_zoom2p0x',
#           '-R', 'combined_gratings_static',
#           '-t', 'traces003',
#           '-r', 'visual', 
#           '-s', 'meanstim',
#           '-p', 'corrected',
#           '-V', 'LM',
##           '--segment',
#           '--nproc=1'
#           ]

options_list = [
 ['-D', rootdir, '-i', 'JC013', 
           '-S', '20180907', 
           '-A', 'FOV2_zoom1x',
           '-R', 'gratings_run1',
           '-t', 'traces001',
           '-r', 'visual', 
           '-s', 'meanstim',
           '-p', 'corrected',
           '-V', 'LI',
#           '--segment',
           '--nproc=1'
           ],
  
['-D', rootdir, '-i', 'JC023', 
           '-S', '20180929', 
           '-A', 'FOV1_zoom2p0x',
           '-R', 'combined_gratings_static',
           '-t', 'traces001',
           '-r', 'visual', 
           '-s', 'meanstim',
           '-p', 'corrected',
           '-V', 'LI',
#           '--segment',
           '--nproc=1'
           ],
           
['-D', rootdir, '-i', 'JC022', 
           '-S', '20181005', 
           '-A', 'FOV2_zoom2p7x',
           '-R', 'combined_gratings_static',
           '-t', 'traces001',
           '-r', 'visual', 
           '-s', 'meanstim',
           '-p', 'corrected',
           '-V', 'LI',
#           '--segment',
           '--nproc=1'
           ],


['-D', rootdir, '-i', 'JC022', 
           '-S', '20181017', 
           '-A', 'FOV1_zoom2p7x',
           '-R', 'combined_gratings_static',
           '-t', 'traces001',
           '-r', 'visual', 
           '-s', 'meanstim',
           '-p', 'corrected',
           '-V', 'LI',
#           '--segment',
           '--nproc=1'
           ],


['-D', rootdir, '-i', 'JC022', 
           '-S', '20181006', 
           '-A', 'FOV1_zoom2p7x',
           '-R', 'combined_gratings_static',
           '-t', 'traces001',
           '-r', 'visual', 
           '-s', 'meanstim',
           '-p', 'corrected',
           '-V', 'LI',
#           '--segment',
           '--nproc=1'
           ],
 
        ['-D', rootdir, 
              '-i', 'JC022', 
           '-S', '20181005', 
           '-A', 'FOV3_zoom2p7x',
           '-R', 'combined_gratings_static',
           '-t', 'traces001',
           '-r', 'visual', 
           '-s', 'meanstim',
           '-p', 'corrected',
           '-V', 'LI',
#           '--segment',
           '--nproc=1'
           ],
         
         ['-D', rootdir, 
              '-i', 'JC015', 
           '-S', '20180925', 
           '-A', 'FOV1_zoom2p0x',
           '-R', 'combined_gratings_static',
           '-t', 'traces003',
           '-r', 'visual', 
           '-s', 'meanstim',
           '-p', 'corrected',
           '-V', 'LI',
           '--segment',
           '--nproc=1'
           ],
          
          ['-D', rootdir, '-i', 'JC015', 
           '-S', '20180925', 
           '-A', 'FOV1_zoom2p0x',
           '-R', 'combined_gratings_static',
           '-t', 'traces003',
           '-r', 'visual', 
           '-s', 'meanstim',
           '-p', 'corrected',
           '-V', 'LM',
           '--segment',
           '--nproc=1'
           ],
           
           ['-D', rootdir, '-i', 'JC015', 
           '-S', '20180919', 
           '-A', 'FOV1_zoom2p0x',
           '-R', 'combined_gratings_static',
           '-t', 'traces003',
           '-r', 'visual', 
           '-s', 'meanstim',
           '-p', 'corrected',
           '-V', 'LM',
#           '--segment',
           '--nproc=1'
           ],
            
           ['-D', rootdir, '-i', 'JC015', 
           '-S', '20180917', 
           '-A', 'FOV1_zoom2p0x',
           '-R', 'combined_gratings_static',
           '-t', 'traces002',
           '-r', 'visual', 
           '-s', 'meanstim',
           '-p', 'corrected',
           '-V', 'LM',
#           '--segment',
           '--nproc=1'
           ],
            
           ['-D', rootdir, '-i', 'JC015', 
           '-S', '20180915', 
           '-A', 'FOV1_zoom2p7x',
           '-R', 'combined_gratings_static',
           '-t', 'traces002',
           '-r', 'visual', 
           '-s', 'meanstim',
           '-p', 'corrected',
           '-V', 'LM',
#           '--segment',
           '--nproc=1'
           ]
           

        ]
           
           #%%
           
#curr_options = ['-D', rootdir, '-i', 'JC023', 
#           '-S', '20180929', 
#           '-A', 'FOV1_zoom2p0x',
#           '-R', 'combined_gratings_static',
#           '-t', 'traces001',
#           '-r', 'visual', 
#           '-s', 'meanstim',
#           '-p', 'corrected',
#           '-V', 'LI',
##           '--segment',
#           '--nproc=1'
#           ]
           
#curr_options = ['-D', rootdir, '-i', 'JC022', 
#           '-S', '20181005', 
#           '-A', 'FOV2_zoom2p7x',
#           '-R', 'combined_gratings_static',
#           '-t', 'traces001',
#           '-r', 'visual', 
#           '-s', 'meanstim',
#           '-p', 'corrected',
#           '-V', 'LI',
##           '--segment',
#           '--nproc=1'
#           ]

#
#curr_options = ['-D', rootdir, '-i', 'JC022', 
#           '-S', '20181017', 
#           '-A', 'FOV1_zoom2p7x',
#           '-R', 'combined_gratings_static',
#           '-t', 'traces001',
#           '-r', 'visual', 
#           '-s', 'meanstim',
#           '-p', 'corrected',
#           '-V', 'LI',
##           '--segment',
#           '--nproc=1'
#           ]

#
#curr_options = ['-D', rootdir, '-i', 'JC022', 
#           '-S', '20181006', 
#           '-A', 'FOV1_zoom2p7x',
#           '-R', 'combined_gratings_static',
#           '-t', 'traces001',
#           '-r', 'visual', 
#           '-s', 'meanstim',
#           '-p', 'corrected',
#           '-V', 'LI',
##           '--segment',
#           '--nproc=1'
#           ]

#
#curr_options = ['-D', rootdir, '-i', 'JC013', 
#           '-S', '20180907', 
#           '-A', 'FOV2_zoom1x',
#           '-R', 'gratings_run1',
#           '-t', 'traces001',
#           '-r', 'visual', 
#           '-s', 'meanstim',
#           '-p', 'corrected',
#           '-V', 'LI',
##           '--segment',
#           '--nproc=1'
#           ]


#%%

for curr_options in options_list:
    
    #%
    optsE = extract_options(curr_options)
    visual_area = optsE.visual_area
    
    #traceid_dirs = cls.get_traceids_from_lists(optsE.animalid, optsE.session_list, optsE.fov_list, optsE.run_list, optsE.traceid_list, rootdir=rootdir)
    #(fov, traceid_dir), = sorted(traceid_dirs.items(), key=lambda x: x[0])
    #print "%s | %s" % (fov, os.path.split(traceid_dir)[-1].split('_')[0])
    
    D = DataSet(optsE.animalid, optsE.session_list[0], optsE.fov_list[0],  optsE.run_list[0],  optsE.traceid_list[0], rootdir=optsE.rootdir)
        
    if optsE.select_visual_area:
        visual_areas_fpath = sorted(glob.glob(os.path.join(D.rootdir, D.animalid, D.session, D.acquisition, 'visual_areas', 'segmentation_*.pkl')), key=natural_keys)[0]
        visual_area_info = {optsE.visual_area: visual_areas_fpath}
    else:
        visual_area_info = None
            
    D.load_dataset(visual_area_info=visual_area_info, return_type='std')
    
    
    fov_key = '%s_%s_%s' % (D.animalid, D.session, D.acquisition)
    print fov_key
    
    
    #if fov_key not in dprime_by_area[visual_area].keys():
    #    dprime_by_area[visual_area][fov_key] 
    ##%
    #dataset = np.load(data_fpath)
    
    area_color = color_dict[optsE.visual_area]
    
    sdf = pd.DataFrame(D.sconfigs).T
    
    trans_types = D.run_info['trans_types']
    object_id_types = ['morphlevel', 'ori', 'object']
    
    tested_transforms = [t for t in trans_types if t not in object_id_types]
    print "Tested transforms: %s" % str(tested_transforms)
    object_type = 'ori'
    
    
    config_labels = pd.DataFrame(data=D.labels, columns=['config'], index=D.labels)
    
    for trans in trans_types:
        config_labels = config_labels.assign(trans = lambda x:assign_configs(x['config'], trans, sdf)) #, axis=0)
        config_labels.rename(columns={'trans': '%s' % trans}, inplace=True)
    
    
    all_dprime_values = []
    for roi_id in D.rois:
        rid = list(D.rois).index(roi_id)
        
        stim_data = pd.DataFrame(data=D.data[:, rid], columns=[D.stat_type], index=D.labels)
        std_data = pd.DataFrame(data=D.std[:, rid], columns=['std'], index=D.labels)
        
        df = pd.concat([stim_data, std_data, config_labels], axis=1)      
        
        dprime_by_config = get_dprime(df, sdf, tested_transforms, roi_id=roi_id, stat_type=D.stat_type, object_type=object_type)
    
    #    if plot:
    #        plot_dprime_by_config(dprime_by_config, roi_id=roi_id)
        
        if dprime_by_config is not None:
            all_dprime_values.extend(dprime_by_config.values())
    
        
    #dprime_by_area[visual_area].extend(all_dprime_values)
    dprime_by_area[visual_area][fov_key] = all_dprime_values
    

    #%%
#area_color = 'g'
subdict = dict((k, v) for k,v in dprime_by_area.items() if len(v.keys()) > 0)
print subdict.keys()


DPRIME = dict((vkey, []) for vkey in subdict.keys())


excluded = {'LI':  ['20181006', '20181007', '20180929'],
            'LM':  ['20180925']}

for vkey, fovinfo in subdict.items():
    tmpfov = dict((fov, vals) for fov, vals in fovinfo.items() if fov.split('_')[1] not in excluded[vkey])
    
    
    DPRIME[vkey].extend([l for sublist in tmpfov.values() for l in sublist])
    
    

#%%
    
    
fig, ax = pl.subplots(figsize=(10,10))

for visual_area, dprime_values in DPRIME.items():
    area_color = color_dict[visual_area]
    sns.distplot(dprime_values, label=visual_area,
                 kde=True, kde_kws={"color": area_color, "lw": 3, "alpha": 1.0, "label": "KDE"},
#                 rug=True, rug_kws={"color": area_color},
                 hist_kws={"histtype": "step", "linewidth": 3, "alpha": 0.5, "color": area_color})

pl.legend()
pl.xlabel('d prime')
pl.ylabel('counts')
#pl.ylim([0, 1.2])

sns.despine(offset=4, trim=True)

pl.savefig(os.path.join(rootdir, 'VISUAL_AREAS', 'comparisons', 'hist_kde_dprime_LM_LI_plot1_normed.pdf'))
pl.savefig(os.path.join(rootdir, 'VISUAL_AREAS', 'comparisons', 'hist_kde_dprime_LM_LI_plot1_normed.png'))

#R_ref = df[ ( (df[object_type]==ref_obj) & (df['xpos']==ref_x) & (df['ypos']==ref_y) ) ][D.stat_type]
#R_flank = df[ ( (df[object_type]==flank_obj) & (df['xpos']==ref_x) & (df['ypos']==ref_y) ) ][D.stat_type]
#
#mean_ref = np.mean(R_ref.values)
#std_ref = np.std(R_ref.values) #df[ ( (df[object_type]==ref_obj) & (df['xpos']==ref_x) & (df['ypos']==ref_y) ) ]['std']
#
#mean_flank = np.mean(R_flank.values)
#std_flank = np.std(R_flank.values)
#
#roi_dprime = (mean_ref - mean_flank) / np.mean((std_ref, std_flank))
#
#print roi_dprime

def plot_dprime_by_config(dprime_by_config, roi_id=-1):
    
    colvals = sorted(list(set([r[0] for r in dprime_by_config.keys()])))
    rowvals = sorted(list(set([r[1] for r in dprime_by_config.keys()])))
    
    dprime_grid = np.empty((len(rowvals), len(colvals)))
    for ri in range(len(rowvals)):
        for ci in range(len(colvals)):
            stimkey = (colvals[ci], rowvals[ri])
            dprime_grid[ri, ci] = dprime_by_config[stimkey]
            
    pl.figure(); pl.imshow(dprime_grid, cmap='PRGn'); pl.colorbar()
    pl.title('roi%05d' % (roi_id+1))
    
    
    
        
        
        
    
    
    
    
    
    
    
    