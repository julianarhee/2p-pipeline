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
import optparse
import matplotlib
matplotlib.use('Agg')
import pylab as pl
import seaborn as sns
import pandas as pd
import numpy as np
from pipeline.python.utils import natural_keys

#%%
def plot_movie_timecourse_for_stimconfig(roi_df, TRIALS, configs, curr_roi_figdir='/tmp'): #, roi_trials=None):
    """
    For user-specified ROI, create a plot of timecourse for EACH file. Create a new
    plot for each stimulus-config, s.t. each repetition of a given stim-config in a given file
    is highlighted on the timecourse.
    """
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})

    fnames = sorted(roi_df.file.unique(), key=natural_keys)

    for blockidx, selected_file in enumerate(fnames):
        print "Plotting traces for BLOCK %i" % int(blockidx+1)
        file_df = roi_df.loc[roi_df['file'] == selected_file]
        curr_trials_df = TRIALS.loc[TRIALS['block'] == blockidx+1]
        stim_bar_loc = file_df['values'].min() - 500

        for config_key in configs.keys():
            if experiment == 'gratings':
                ori = configs[config_key]['rotation']
                sf = configs[config_key]['frequency']
                xpos = configs[config_key]['position'][0]
                ypos = configs[config_key]['position'][1]
                size = configs[config_key]['scale'][0]
                stimconfig = 'Ori: %.0f, SF: %.2f' % (ori, sf)
                key_trials = curr_trials_df.loc[(curr_trials_df['ori'] == ori) &
                                                (curr_trials_df['sf'] == sf) &
                                                (curr_trials_df['xpos'] == xpos) &
                                                (curr_trials_df['ypos'] == ypos) &
                                                (curr_trials_df['size'] == size)
                                                ]['trial']
            else:
                currimg = os.path.split(configs[config_key]['filepath'])[1]
                currxpos = configs[config_key]['position'][0]
                currypos = configs[config_key]['position'][1]
                currsize = configs[config_key]['scale'][0]
                stimconfig = '%s, pos (%.1f, %.1f), size %.1f' % (str(currimg), currxpos, currypos, currsize)
                key_trials = curr_trials_df.loc[(curr_trials_df['img'] == currimg) &
                                            (curr_trials_df['xpos'] == currxpos) &
                                            (curr_trials_df['ypos'] == currypos) &
                                            (curr_trials_df['size'] == currsize)
                                            ]['trial']

            print stimconfig

            grid = sns.FacetGrid(file_df, row="file", size=3, aspect=15)  # Create subplot(s) based on TIF file (i.e., block)
            grid.map(pl.plot, "tsec", "values", linewidth=0.5)            # Plot trace for current block

            for ax in grid.axes.flat:
                got_legend = False  # Keep track of whether legend-label created, so repeated trials of the same config are not all labeled
                for trial in sorted(curr_trials_df['trial'], key=natural_keys): #trials.keys():
                    #print trial
                    stim_on_sec = float(curr_trials_df.loc[curr_trials_df['trial'] == trial, 'stim_on'])
                    stim_off_sec = float(curr_trials_df.loc[curr_trials_df['trial'] == trial, 'stim_off'])

                    if trial in [k for k in key_trials]: #[int(file_idx-1)]:
                        if got_legend is False:
                            ax.plot([stim_on_sec, stim_off_sec], np.array([1,1])*stim_bar_loc, 'r', label=stimconfig)
                        else:
                            ax.plot([stim_on_sec, stim_off_sec], np.array([1,1])*stim_bar_loc, 'r', label=None)
                        got_legend = True
                    else:
                        ax.plot([stim_on_sec, stim_off_sec], np.array([1,1])*stim_bar_loc, 'k', label=None)

                ax.legend() #{stimconfig})
                sns.despine(trim=True, offset=1)
                curr_save_dir = os.path.join(curr_roi_figdir, config_key, trace_type)
                if not os.path.exists(curr_save_dir):
                    os.makedirs(curr_save_dir)
                fname = '%s_%s_%s_timecourse_%s.png' % (selected_roi, selected_slice, selected_file, config_key)
                pl.savefig(os.path.join(curr_save_dir, fname), bbox_inches='tight')
                pl.close()

                print "Saved timecourse plots to: %s" % curr_save_dir

    # Save a copy of configs info file to roi dir, for easy access:
    with open(os.path.join(curr_roi_figdir, 'configkey.json'), 'w') as f:
        json.dump(configs, f, sort_keys=True, indent=4)



#%%
def plot_movie_timecourse(roi_df, TRIALS, configs, trace_type='raw', roi_df_raw=None, compare_trace_types=False, curr_roi_figdir='/tmp'): #, roi_trials=None):
    """
    For user-specified ROI, create a plot of timecourse for EACH file. Create a new
    plot for each stimulus-config, s.t. each repetition of a given stim-config in a given file
    is highlighted on the timecourse.
    """
    #colormap = pl.get_cmap('jet')
    #colorvals = colormap(np.linspace(0, 1, len(configs.keys())))
    colorvals = sns.color_palette("hls", len(configs.keys()))

    sns.set_style("darkgrid", {"axes.facecolor": ".9"})

    fnames = sorted(roi_df.file.unique(), key=natural_keys)
    for blockidx, selected_file in enumerate(fnames):
        print "Plotting traces for BLOCK %i" % int(blockidx+1)
        file_df = roi_df.loc[roi_df['file'] == selected_file]

        curr_trials_df = TRIALS.loc[TRIALS['block'] == blockidx+1]
        stim_bar_loc = file_df['values'].min() - 500

        fig = pl.figure(figsize=(15,2))
        pl.plot("tsec", "values", linewidth=0.5, data=file_df, label=trace_type)            # Plot trace for current block
        if compare_trace_types is True:
            file_df_raw = roi_df_raw.loc[roi_df_raw['file'] == selected_file]
            pl.plot("tsec", "values", linewidth=0.5, data=file_df_raw, label='raw')

        dfs = []
        for cidx, config_key in enumerate(sorted(configs.keys(), key=natural_keys)):
            stimcolor = colorvals[cidx]
            if experiment == 'gratings':
                ori = configs[config_key]['rotation']
                sf = configs[config_key]['frequency']
                xpos = configs[config_key]['position'][0]
                ypos = configs[config_key]['position'][1]
                size = configs[config_key]['scale'][0]
                stimconfig = 'Ori: %.0f, SF: %.2f' % (ori, sf)
                key_trials = curr_trials_df.loc[(curr_trials_df['ori'] == ori) &
                                            (curr_trials_df['sf'] == sf) &
                                            (curr_trials_df['xpos'] == xpos) &
                                            (curr_trials_df['ypos'] == ypos) &
                                            (curr_trials_df['size'] == size)
                                            ]['trial']
            else:
                currimg = os.path.split(configs[config_key]['filepath'])[1]
                currxpos = configs[config_key]['position'][0]
                currypos = configs[config_key]['position'][1]
                currsize = configs[config_key]['scale'][0]
                stimconfig = '%s, pos (%.1f, %.1f), size %.1f' % (str(currimg), currxpos, currypos, currsize)
                key_trials = curr_trials_df.loc[(curr_trials_df['img'] == currimg) &
                                            (curr_trials_df['xpos'] == currxpos) &
                                            (curr_trials_df['ypos'] == currypos) &
                                            (curr_trials_df['size'] == currsize)
                                            ]['trial']


            got_legend = False
            for trial in sorted(key_trials, key=natural_keys): #trials.keys():
                print trial
                stim_on_sec = float(curr_trials_df.loc[curr_trials_df['trial'] == trial, 'stim_on'])
                stim_off_sec = float(curr_trials_df.loc[curr_trials_df['trial'] == trial, 'stim_off'])
                if got_legend is False:
                    pl.plot([stim_on_sec, stim_off_sec], np.array([1,1])*stim_bar_loc, color=stimcolor, label=stimconfig)
                    got_legend = True
                else:
                    pl.plot([stim_on_sec, stim_off_sec], np.array([1,1])*stim_bar_loc, color=stimcolor, label=None)

        pl.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        sns.despine(trim=True, offset=1)
        curr_save_dir = os.path.join(curr_roi_figdir, trace_type)
        if not os.path.exists(curr_save_dir):
            os.makedirs(curr_save_dir)
        fname = '%s_%s_%s_timecourse_allconfigs.png' % (selected_roi, selected_slice, selected_file)
        pl.savefig(os.path.join(curr_save_dir, fname), bbox_inches='tight')
        pl.close()

        print "Saved timecourse plots to: %s" % curr_save_dir

    # Save a copy of configs info file to roi dir, for easy access:
    with open(os.path.join(curr_roi_figdir, 'configkey.json'), 'w') as f:
        json.dump(configs, f, sort_keys=True, indent=4)

#%%
#rootdir = '/mnt/odyssey' #'/nas/volume1/2photon/data'
#animalid = 'CE074'
#session = '20180215'
#acquisition = 'FOV1_zoom1x_V1'
#run = 'blobs'
#trace_id = 'traces001'
#
#roi_id = 4  # This var matters
#slice_id = 1 # This one, too, if nslices > 1
#trace_type = 'raw'
#is_background = False

#%%

#choices_tracetype = ('raw', 'denoised_nmf')
choices_tracetype = ('raw', 'raw_fissa', 'denoised_nmf', 'np_corrected_fissa', 'neuropil_fissa', 'neuropil', 'np_subtracted')

default_tracetype = 'raw'

parser = optparse.OptionParser()

parser.add_option('-D', '--root', action='store', dest='rootdir', default='/nas/volume1/2photon/data', help='data root dir (root project dir containing all animalids) [default: /nas/volume1/2photon/data, /n/coxfs01/2pdata if --slurm]')
parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')

# Set specific session/run for current animal:
parser.add_option('-S', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID')
parser.add_option('-A', '--acq', action='store', dest='acquisition', default='FOV1', help="acquisition folder (ex: 'FOV1_zoom3x') [default: FOV1]")
parser.add_option('-R', '--run', action='store', dest='run', default='', help="name of run dir containing tiffs to be processed (ex: gratings_phasemod_run1)")
parser.add_option('--slurm', action='store_true', dest='slurm', default=False, help="set if running as SLURM job on Odyssey")

parser.add_option('-t', '--trace-id', action='store', dest='trace_id', default='', help="Trace ID for current trace set (created with set_trace_params.py, e.g., traces001, traces020, etc.)")
parser.add_option('-T', '--trace-type', type='choice', choices=choices_tracetype, action='store', dest='trace_type', default=default_tracetype, help="Type of timecourse to plot PSTHs. Valid choices: %s [default: %s]" % (choices_tracetype, default_tracetype))

parser.add_option('-s', '--slice', action='store', dest='slice_id', default=1, help="Slice num of ROI to plot [default: 1]")
parser.add_option('-r', '--roi', action='store', dest='roi_id', default=1, help="ROI num to plot [default: 1]")
parser.add_option('--background', action='store_true', dest='is_background', default=False, help="set if ROI selected is a background component")
parser.add_option('--config', action='store_true', dest='plot_by_config', default=False, help="set if want to plot timecourses per file, per stimulus config also.")
parser.add_option('--compare', action='store_true', dest='compare_traces', default=False, help="set if want to plot non-raw trace against raw.")

(options, args) = parser.parse_args()

# Set USER INPUT options:
rootdir = options.rootdir
slurm = options.slurm
if slurm is True and 'coxfs01' not in rootdir:
    rootdir = '/n/coxfs01/2p-data'

animalid = options.animalid
session = options.session
acquisition = options.acquisition
run = options.run
compare_traces = options.compare_traces

trace_id = options.trace_id
trace_type = options.trace_type
if not trace_type == 'raw' and compare_traces is True:
    compare_trace_types = True
else:
    compare_trace_types = False

roi_id = int(options.roi_id)
slice_id = int(options.slice_id)

is_background = options.is_background
plot_by_config = options.plot_by_config

#%%
if is_background is True:
    selected_roi = 'bg%02d' % int(roi_id)
else:
    selected_roi = 'roi%05d' % int(roi_id)

selected_slice = 'Slice%02d' % int(slice_id)

print "Plotting %s PSTHs for roi %i, slice %i." % (trace_type, roi_id, slice_id)

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

# First check whether AUX and TIF files are one-to-one:
with open(os.path.join(paradigm_dir, 'files', paradigm_files[0]), 'r') as fp:
    tmptrials = json.load(fp)
    blocknums = list(set([tmptrials[t]['block_idx'] for t in tmptrials.keys()]))
    if len(blocknums) > 1:
        # More than 1 block found in curr aux file.
        one_to_one = False

        # Load run meta info to get N frames per file (need in order to offset tstamps for tifs):
        runinfo_path = os.path.join(run_dir, '%s.json' % run)
        with open(runinfo_path, 'r') as fr:
            runinfo = json.load(fr)
        nvolumes = runinfo['nvolumes']
        nslices_full = int(round(runinfo['frame_rate']/runinfo['volume_rate']))
        nframes_per_file = nslices_full * nvolumes
        secs_per_file = nframes_per_file / runinfo['volume_rate'] + (1/runinfo['volume_rate'])

    else:
        # Only 1 block found in curr aux file.
        one_to_one = True

if one_to_one is True:
    # Make sure there is an AUX file for each extract-trace file (of which there is 1 per TIF):
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

#%% CREATE TRIAL dataframe:
trial_dfs = []
for fidx, paradigm_fn in enumerate(sorted(paradigm_files, key=natural_keys)):
    curr_file = file_names[fidx]
    with open(os.path.join(paradigm_dir, 'files', paradigm_fn), 'r') as fp:
        trials = json.load(fp)
    for idx, trial in enumerate(sorted(trials.keys(), key=natural_keys)):
        if trials[trial]['stimuli']['type'] == 'drifting_grating':
            experiment = 'gratings'
            trial_dfs.append(pd.DataFrame({'trial': str(trial),
                                           'file': curr_file,
                                           'block': trials[trial]['block_idx'] + 1,
                                           'stim_on': trials[trial]['stim_on_time_block']/1E3,
                                           'stim_off': trials[trial]['stim_off_time_block']/1E3,
                                           'ori': int(trials[trial]['stimuli']['rotation']),
                                           'sf': trials[trial]['stimuli']['frequency'],
                                           'xpos': trials[trial]['stimuli']['position'][0],
                                           'ypos': trials[trial]['stimuli']['position'][1],
                                           'size': trials[trial]['stimuli']['scale'][0]},
                                           index=[idx]
                                           ))
        else:
            # do sth else:
            experiment = 'images'
            trial_dfs.append(pd.DataFrame({'trial': str(trial),
                                           'file': curr_file,
                                           'block': trials[trial]['block_idx'] + 1,
                                           'stim_on': trials[trial]['stim_on_time_block']/1E3,
                                           'stim_off': trials[trial]['stim_off_time_block']/1E3,
                                           'img': os.path.split(trials[trial]['stimuli']['filepath'])[1],
                                           'xpos': trials[trial]['stimuli']['position'][0],
                                           'ypos': trials[trial]['stimuli']['position'][1],
                                           'size': trials[trial]['stimuli']['scale'][0]}, # int(trials[trial]['stimuli']['name'])},
                                           index=[idx]
                                           ))
TRIALS = pd.concat(trial_dfs, axis=0)

#%%  Create TRACES dataframe:

# Get mask info from MASKS.hdf5 in traceid dir:
maskinfo = h5py.File(os.path.join(traceid_dir, 'MASKS.hdf5'), 'r')

rawtrace_dfs = []
trace_dfs = []
nframes = []
for fidx, trace_fn in enumerate(sorted(trace_files, key=natural_keys)):
    curr_file = file_names[fidx]

    # Load extrated TRACES from extracted timecourse mats [T,nrois]:
    traces = h5py.File(os.path.join(traceid_dir, 'files', trace_fn), 'r')
    nframes.append(traces['Slice01']['traces'][trace_type].shape[0])

nvolumes_unique = list(set(nframes))
max_nframes = max(nvolumes_unique)


for fidx, trace_fn in enumerate(sorted(trace_files, key=natural_keys)):
    curr_file = file_names[fidx]

    # Load extrated TRACES from extracted timecourse mats [T,nrois]:
    traces = h5py.File(os.path.join(traceid_dir, 'files', trace_fn), 'r')

    for sidx, curr_slice in enumerate(traces.keys()):
        print "trace types:", traces[curr_slice]['traces'].keys()

        nr = maskinfo[curr_file][curr_slice]['maskarray'].attrs['nr'] #traces[curr_slice]['masks'].attrs['nr']
        nb = maskinfo[curr_file][curr_slice]['maskarray'].attrs['nb'] #traces[curr_slice]['masks'].attrs['nb']

        roi_list = ['roi%05d' % int(ridx+1) for ridx in range(nr)] #traces[curr_slice]['masks'].attrs['roi_idxs']]
        roi_list.extend(['bg%02d' % int(bidx+1) for bidx in range(nb)]) # Make sure background comps are at END Of list, since indexing thru:
        values = traces[curr_slice]['traces'][trace_type][:, ridx]
        raw_values = traces[curr_slice]['traces']['raw'][:, ridx]
        if len(values) < max_nframes:
            # Pad:
            pad_size = max_nframes - len(values)
            values = np.pad(values, (pad_size,0), 'constant', constant_values=np.nan) 
            raw_values = np.pad(raw_values, (pad_size,0), 'constant', constant_values=np.nan) 
        for ridx, roi in enumerate(roi_list):
#            if len(traces.keys()) > 1:
            trace_dfs.append(pd.DataFrame({'tsec': traces[curr_slice]['frames_tsec'][:], # + fidx*secs_per_file,
                                     'file': curr_file,
                                     'block': fidx + 1,
                                     'slice': curr_slice,
                                     'values': values, #traces[curr_slice]['traces'][trace_type][:, ridx],
                                     'roi': roi_list[ridx]
                                     }))
            if not trace_type == 'raw':
                 rawtrace_dfs.append(pd.DataFrame({'tsec': traces[curr_slice]['frames_tsec'][:], # + fidx*secs_per_file,
                                     'file': curr_file,
                                     'block': fidx + 1,
                                     'slice': curr_slice,
                                     'values': raw_values, #traces[curr_slice]['traces']['raw'][:, ridx],
                                     'roi': roi_list[ridx]
                                     }))
    
#            else:
#                trace_dfs.append(pd.DataFrame({'tsec': traces[curr_slice]['frames_tsec'][:] + fidx*secs_per_file,
#                                         'file': curr_file,
#                                         'block': fidx + 1,
#                                         'values': traces[curr_slice]['traces'][trace_type][:, ridx],
#                                         'roi': roi_list[ridx]
#                                         }))

    traces.close()

DATA = pd.concat(trace_dfs, axis=0)
if len(rawtrace_dfs) > 0:
    rawDATA = pd.concat(rawtrace_dfs, axis=0)

#TRIALS = pd.concat(trial_dfs, axis=0)

#%%
def print_attrs(name, obj):
    print name
    for key, val in obj.attrs.iteritems():
        print "    %s: %s" % (key, val)

#%% Make output dir:

tcourse_figdir = os.path.join(traceid_dir, 'figures', 'timecourses')
if not os.path.exists(tcourse_figdir):
    os.makedirs(tcourse_figdir)

curr_roi_figdir = os.path.join(tcourse_figdir, selected_roi)
if not os.path.exists(curr_roi_figdir):
    os.makedirs(curr_roi_figdir)

#%%
#roi_idx = 4  # This var matters
#slice_idx = 1 # This one, too, if nslices > 1

file_id = 1 # this is tmp

#selected_roi = 'roi%05d' % int(roi_id)
#selected_slice = 'Slice%02d' % int(slice_id)
selected_file = 'File%03d' % int(file_id)

curr_save_dir = os.path.join(tcourse_figdir, selected_roi)
if not os.path.exists(curr_save_dir):
    os.makedirs(curr_save_dir)

# Only grab entries for selected ROI:
roi_df = DATA.loc[DATA['roi'] == selected_roi]
if compare_trace_types:
    roi_df_raw = rawDATA.loc[rawDATA['roi'] == selected_roi]
else:
    roi_df_raw = None

# Only grab a given file, tmp:
file_df = roi_df.loc[roi_df['file'] == selected_file]

#%%
# GET STIM CONFIG dict -- this is created in align_acquisition_events.py
with open(os.path.join(paradigm_dir, 'stimulus_configs.json'), 'r') as f:
    configs = json.load(f)

#curr_trials_df = TRIALS.loc[TRIALS['file'] == selected_file]
#if experiment == 'gratings':
#    oris = curr_trials_df.ori.unique()
#    sfs = curr_trials_df.sf.unique()
#    config_idx = 1
#    configs = dict()
#    for ori in oris:
#        for sf in sfs:
#            stimconfig = 'Ori: %.0f, SF: %.2f' % (ori, sf)
#
#            configs['config%03d' % int(config_idx)] = stimconfig
#            config_idx += 1
#else:
#    with open(os.path.join(paradigm_dir, 'stimulus_configs.json'), 'r') as f:
#        configs = json.load(f)

#    config_idx = 1
#    configs = dict()
#    imgs = curr_trials_df.img.unique()
#    for img in imgs:
#        stimconfig = 'Img:', img
#        configs['config%03d' % int(config_idx)] = stimconfig
#        config_idx += 1

#%%
# Plot time course of EACG file, highlight EACH stim config for specfied ROI:
if plot_by_config:
    plot_movie_timecourse_for_stimconfig(roi_df, TRIALS, configs, trace_type=trace_type, curr_roi_figdir=curr_roi_figdir) #, roi_trials=roi_trials)

plot_movie_timecourse(roi_df, TRIALS, configs, trace_type=trace_type, roi_df_raw=roi_df_raw, compare_trace_types=compare_trace_types, curr_roi_figdir=curr_roi_figdir)


#%%


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
