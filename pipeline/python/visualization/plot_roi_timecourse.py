#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 16:24:17 2018

@author: julianarhee
"""

import os
import json
import traceback
import h5py
import matplotlib
matplotlib.use('Agg')
import pylab as pl
import seaborn as sns
import pandas as pd
import numpy as np
from pipeline.python.utils import natural_keys

#%%
rootdir = '/nas/volume1/2photon/data'
animalid = 'JR063'
session = '20171128_JR063'
acquisition = 'FOV2_zoom1x'
run = 'gratings_static'
trace_id = 'traces001'

roi_idx = 4  # This var matters
slice_idx = 1 # This one, too, if nslices > 1
min_val = 2000 # Make this flex

#%%

#roi_idx = 1
#slice_idx = 1
#file_idx = 1
#min_val = 2000

#%%
run_dir = os.path.join(rootdir, animalid, session, acquisition, run)

# Load parsed paradigm files:
paradigm_dir = os.path.join(run_dir, 'paradigm')
paradigm_files = sorted([f for f in os.listdir(os.path.join(paradigm_dir, 'files')) if f.endswith('json')], key=natural_keys)

# Load extracted traces:
trace_dir = os.path.join(run_dir, 'traces')
trace_name = [t for t in os.listdir(trace_dir) if os.path.isdir(os.path.join(trace_dir, t)) and trace_id in t][0]
traceid_dir = os.path.join(trace_dir, trace_name)
trace_files = sorted([t for t in os.listdir(os.path.join(traceid_dir, 'files')) if t.endswith('hdf5')], key=natural_keys)
file_names = ["File%03d" % int(i+1) for i in range(len(trace_files))]

#%% Match paradigm files and trace files:
assert len(paradigm_files) == len(trace_files), "Mismatch in N paradigm files (%i) and N trace files (%i)" % (len(paradigm_files), len(trace_files))
try:
    file_list = [(par, tra) for par, tra in zip(paradigm_files, trace_files)]
    for fname in file_names:
        pidx = paradigm_files.index([f for f in paradigm_files if fname in f or fname.lower() in f][0])
        tidx = trace_files.index([t for t in trace_files if fname in t][0])
        assert pidx == tidx, "Paradigm and Trace files are not sorted properly!"
except Exception as e:
    print "--- ERROR: Unable to match TIFF files to paradigm files..."
    traceback.print_exc()
    print "----------------------------------------------------------"
    for p, pfile in enumerate(paradigm_files):
        print p, pfile
    for t, tfile in enumerate(trace_files):
        print t, tfile
    print "ABORTING." 

#%%
#curr_save_dir = os.path.join(traceid_dir, 'figures', 'roi%04d' % roi_idx)
#if not os.path.exists(curr_save_dir):
#    os.makedirs(curr_save_dir)
trace_type = 'raw'

trace_dfs = []
trial_dfs = []
for fidx, filepair in enumerate(file_list):
    curr_file = file_names[fidx]
    
    # Load parsed TRIAL info from behavior parsing:
    paradigm_fn = filepair[0]
    with open(os.path.join(paradigm_dir, 'files', paradigm_fn), 'r') as fp:
        trials = json.load(fp)
    for idx, trial in enumerate(sorted(trials.keys(), key=natural_keys)):
        if trials[trial]['stimuli']['type'] == 'drifting_grating':
            trial_dfs.append(pd.DataFrame({'trial': str(trial),
                                           'file': curr_file,
                                           'stim_on': trials[trial]['stim_on_times']/1E3,
                                           'stim_off': trials[trial]['stim_off_times']/1E3,
                                           'ori': int(trials[trial]['stimuli']['rotation']),
                                           'sf': trials[trial]['stimuli']['frequency']},
                                           index=[idx]
                                           ))
        else:
            # do sth else:
            trial_dfs.append(pd.DataFrame({'trial': str(trial),
                                           'file': curr_file,
                                           'stim_on': trials[trial]['stim_on_times']/1E3,
                                           'stim_off': trials[trial]['stim_off_times']/1E3,
                                           'img': int(trials[trial]['stimuli']['name'])},
                                           index=[idx]
                                           ))  
            
    # Load extrated TRACES from extracted timecourse mats [T,nrois]:
    trace_fn = filepair[1]
    traces = h5py.File(os.path.join(traceid_dir, 'files', trace_fn), 'r')
    for sidx, curr_slice in enumerate(traces.keys()):
        nrois = traces[curr_slice]['masks'].attrs['nr'] + traces[curr_slice]['masks'].attrs['nb']
        
        roi_list = ['roi%05d' % int(ridx+1) for ridx in range(nrois)] #traces[curr_slice]['masks'].attrs['roi_idxs']]

        for ridx, roi in enumerate(roi_list):
            if len(traces.keys()) > 1:
                trace_dfs.append(pd.DataFrame({'tsec': traces[curr_slice]['frames_tsec'],
                                         'file': curr_file,
                                         'slice': curr_slice,
                                         'values': traces[curr_slice]['traces'][trace_type][:, ridx],
                                         'roi': roi_list[ridx]
                                         }))
            else:
                trace_dfs.append(pd.DataFrame({'tsec': traces[curr_slice]['frames_tsec'],
                                         'file': curr_file,
                                         'values': traces[curr_slice]['traces'][trace_type][:, ridx],
                                         'roi': roi_list[ridx]
                                         }))

    traces.close()
    
DATA = pd.concat(trace_dfs, axis=0)
TRIALS = pd.concat(trial_dfs, axis=0)


#%%
tcourse_figdir = os.path.join(traceid_dir, 'figures', 'timecourses')
if not os.path.exists(tcourse_figdir):
    os.makedirs(tcourse_figdir)
    
#%%
#roi_idx = 4  # This var matters
#slice_idx = 1 # This one, too, if nslices > 1


file_idx = 1

selected_roi = 'roi%05d' % int(roi_idx)
selected_slice = 'Slice%02d' % int(slice_idx)
selected_file = 'File%03d' % int(file_idx)

curr_save_dir = os.path.join(tcourse_figdir, selected_roi)
if not os.path.exists(curr_save_dir):
    os.makedirs(curr_save_dir)
    
roi_df = DATA.loc[DATA['roi'] == selected_roi]
file_df = roi_df.loc[roi_df['file'] == selected_file]

#%%
# GET STIM CONFIG dict:
curr_trials_df = TRIALS.loc[TRIALS['file'] == selected_file]
if trials[trials.keys()[0]]['stimuli']['type'] == 'drifting_grating':
    oris = curr_trials_df.ori.unique()
    sfs = curr_trials_df.sf.unique()
    config_idx = 1
    configs = dict()
    for ori in oris:
        for sf in sfs:
            stimconfig = 'Ori: %.0f, SF: %.2f' % (ori, sf)
            
            configs['config%03d' % int(config_idx)] = stimconfig
            config_idx += 1
else:
    config_idx = 1
    configs = dict()
    imgs = curr_trials_df.img.unique()
    for img in imgs:
        stimconfig = 'Img: %i' % img
        configs['config%03d' % int(config_idx)] = stimconfig
        config_idx += 1
        
#%%
# Plot time course of EACG file, highlight EACH stim config for specfied ROI:
fnames = roi_df.file.unique()
sns.set_style("darkgrid", {"axes.facecolor": ".9"})

for selected_file in fnames:
    file_df = roi_df.loc[roi_df['file'] == selected_file]
    curr_trials_df = TRIALS.loc[TRIALS['file'] == selected_file]
    
    oris = curr_trials_df.ori.unique()
    sfs = curr_trials_df.sf.unique()
    
    config_idx = 0
    for ori in oris:
        for sf in sfs:
            stimconfig = 'Ori: %.0f, SF: %.2f' % (ori, sf)
            print stimconfig
            config_key = [k for k in configs.keys() if configs[k] == stimconfig][0]
            
            key_trials = curr_trials_df.loc[(curr_trials_df['ori'] == ori) & (curr_trials_df['sf'] == sf)]['trial']
            
            grid = sns.FacetGrid(file_df, row="file", size=3, aspect=15)
            #g = (grid.map(pl.plot, "tsec", "values", linewidth=0.5).fig.subplots_adjust(wspace=-.001, hspace=-.001))
            grid.map(pl.plot, "tsec", "values", linewidth=0.5)
            
            for ax in grid.axes.flat:
                got_legend = False
                for trial in trials.keys():
                    if trial in [k for k in key_trials]: #[int(file_idx-1)]:
                        if got_legend is False:
                            ax.plot([trials[trial]['stim_on_times']/1E3, trials[trial]['stim_off_times']/1E3], np.array([1,1])*min_val, 'r', label=stimconfig)
                        else:
                            ax.plot([trials[trial]['stim_on_times']/1E3, trials[trial]['stim_off_times']/1E3], np.array([1,1])*min_val, 'r', label=None)
                        got_legend = True
                    else:
                        ax.plot([trials[trial]['stim_on_times']/1E3, trials[trial]['stim_off_times']/1E3], np.array([1,1])*min_val, 'k', label=None)
                
                ax.legend() #{stimconfig})        
                
            sns.despine(trim=True, offset=1)

            fname = '%s_%s_%s_timecourse_%s.png' % (selected_roi, selected_slice, selected_file, config_key)
            pl.savefig(os.path.join(curr_save_dir, fname), bbox_inches='tight')
            pl.close()
   
#%% Plot traces, sublots are files:
    
#grid = sns.FacetGrid(roi_df, row="file", sharex=True, margin_titles=False)
##grid.map(pl.plot, "tsec", "values")
#g = (grid.map(pl.plot, "tsec", "values", linewidth=0.5).fig.subplots_adjust(wspace=.001, hspace=.001))
#
#file_counter = 0
#for ax in grid.axes.flat:
#    
#    for trial in trials.keys():
#        if trial in key_trials[file_counter]:
#            color = 'r'
#        else:
#            color = 'k'
#        ax.plot([trials[trial]['stim_on_times']/1E3, trials[trial]['stim_off_times']/1E3], np.array([1,1])*min_val, color)
#    
#    file_counter += 1
#    
#sns.despine(trim=True)

#%%

#subdf = pd.DataFrame({'tsec': })
#tidy = pd.melt(DATA, id_vars=['roi', 'slice', 'file', 'tsec'], value_vars=['values'], value_name='trace')
#
#g = sns.factorplot(data=roi_df, # from your Dataframe
#                   col="file", # Make a subplot in columns for each variable in "animal"
#                   col_wrap=5, # Maximum number of columns per row 
#                   x="tsec", # on x-axis make category on the variable "variable" (created by the melt operation)
#                   y="values", # The corresponding y values
#                   #hue="file", # color according to the column gender
#                   kind="strip", # the kind of plot, the closest to what you want is a stripplot, 
#                   legend_out=False, # let the legend inside the first subplot.
#                   )
# grid = sns.FacetGrid(roi_df, col="file", hue="file", col_wrap=5, size=1.5)
#
#melted = DATA.melt(id_vars=['roi', 'file', 'tsec'], value_vars=['values'])
#g = sns.FacetGrid(melted, col='file', hue='roi', sharex='col', margin_titles=True)
#g.map(pl.plot, 'tsec', 'value')

        #tr = traces[curr_slice]['rawtraces']
        #df = pd.DataFrame(np.reshape(tr, tr.shape), columns=roi_list)
        #df['frame_tsec'] = traces[curr_slice]['frames_tsec'] #.tolist()
        
        
        #curr_traces = traces[curr_slice]['rawtraces'][:,roi_idx]
        #curr_tstamps = traces[curr_slice]['frames_tsec']
        #min_val = curr_traces.min() - curr_traces.min()*0.10
        
#        pl.figure()
#        pl.plot(curr_tstamps, curr_traces)
#        pl.xlabel('sec')


#        melted = df.melt('frame_tsec', value_vars=['rois0001', 'rois0002']) #var_name='cols',  value_name='vals')
#        g = sns.FacetGrid(melted, row='variable', sharex='col', margin_titles=True)
#        g.map(pl.plot, 'value')
##        g = sns.factorplot(x="X_Axis", y="vals", hue='cols', data=df)
#
#        for trial in trials.keys():
#            pl.plot([trials[trial]['stim_on_times'], trials[trial]['stim_off_times']], np.array([1,1])*min_val, 'r')
#        
#        figname = 'roi%04d_timecourse_%s_%s.png' % (roi_idx, curr_file, curr_slice)
#        pl.savefig(os.path.join(curr_save_dir, figname))
#        pl.close()
