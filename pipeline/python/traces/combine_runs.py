#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 17:52:45 2018

@author: juliana
"""
import optparse
import h5py
import sys
import os
import json
import pprint
import optparse
import time
import datetime
import pandas as pd
import seaborn as sns
import pylab as pl
import numpy as np


import pipeline.python.visualization.plot_psths_from_dataframe as vis
import pipeline.python.paradigm.align_acquisition_events as acq
from scipy import stats
from pipeline.python.utils import natural_keys, replace_root, print_elapsed_time
from pipeline.python.visualization import plot_psths_from_dataframe as psth
from pipeline.python.traces.utils import load_TID, get_metric_set
from pipeline.python.paradigm.align_acquisition_events import get_stimulus_configs, set_pupil_params
pp = pprint.PrettyPrinter(indent=4)

#%%
def get_info_for_runs(run_list, traceid_list, trace_type, acquisition_dir):


    acquisition = os.path.split(acquisition_dir)[-1]
    session = os.path.split(os.path.split(acquisition_dir)[0])[-1]
    animalid = os.path.split(os.path.split(os.path.split(acquisition_dir)[0])[0])[-1]
    rootdir = os.path.split(os.path.split(os.path.split(acquisition_dir)[0])[0])[0]
    print "Root: %s | Animal: %s | Session: %s | Acq: %s" % (rootdir, animalid, session, acquisition)

    TIDs = {}
    alignment_info = {}
    data_fpaths = dict((run, dict()) for run in run_list)
    stimconfigs = {}
    for run, trace_id in zip(run_list, traceid_list):
        run_dir = os.path.join(rootdir, animalid, session, acquisition, run)

        # Load TRACE ID info:
        # =========================================================================
        TID = load_TID(run_dir, trace_id)
        traceid_dir = TID['DST']
        if rootdir not in traceid_dir:
            orig_root = traceid_dir.split('/%s/%s' % (animalid, session))[0]
            traceid_dir = traceid_dir.replace(orig_root, rootdir)
            print "Replacing orig root with dir:", traceid_dir
            TID['DST'] = traceid_dir
        TIDs[run] = TID

        # Get path to ROIDATA_ for each run:
        # =========================================================================
        roidata_filepath = [os.path.join(traceid_dir, f)
                            for f in os.listdir(traceid_dir)
                            if 'ROIDATA_' in f and trace_type in f and f.endswith('hdf5')][0]
        data_fpaths[run]['roidata_filepath'] = roidata_filepath
        data_fpaths[run]['roidata_hash'] = os.path.splitext(os.path.split(roidata_filepath)[-1])[0].split('_')[1]

        # Get aligned + parsed stimulus/paradigm info for each run:
        # =========================================================================
        event_info_fpath = [os.path.join(traceid_dir, f) for f in os.listdir(traceid_dir) if 'event_alignment' in f and f.endswith('json')][0]
        with open(event_info_fpath, 'r') as f:
            trial_info = json.load(f)
        if rootdir not in trial_info['parsed_trials_source']:
            trial_info['parsed_trials_source'] = replace_root(trial_info['parsed_trials_source'], rootdir, animalid, session)
        configs, stimtype = get_stimulus_configs(trial_info)
        stimconfigs[run] = configs

        alignment_info[run] = {k: v for k, v in trial_info.items() if k not in ['parsed_trials_source']}


    return TIDs, data_fpaths, stimconfigs, alignment_info

#%%
def roidata_collate_runs(data_fpaths):

    run_list = data_fpaths.keys()
    print "Collating data across %i runs." % len(run_list)

    t_collate = time.time()
    df_list = []
    for run in run_list:
        print data_fpaths[run]['roidata_filepath']
        df = pd.HDFStore(data_fpaths[run]['roidata_filepath'], 'r')
        for roi in df.keys():
            if '/' in roi:
                roiname = roi[1:]
            else:
                roiname = roi
            dfr = df[roi]
            dfr['run'] = pd.Series(np.tile(run, (len(dfr .index),)), index=dfr.index)
            dfr['roi'] = pd.Series(np.tile(roiname, (len(dfr .index),)), index=dfr.index)
            df_list.append(dfr)
    DATA = pd.concat(df_list, axis=0, ignore_index=True)
    print_elapsed_time(t_collate)

    return DATA

#%%

def reindex_data_fields(DATA, run_list, stimconfigs, combined_tracedir):
    # arbitrarily choose run (default is chronologically first):
    ntrials = {}
    for run in run_list:
        ntrials[run] = len(list(set(DATA[DATA['run']==run]['trial'])))

    ntrials_to_add = 0
    for idx,run in enumerate(run_list):
        if idx == 0:
            continue

        trial_list = sorted(list(set([str(t) for t in list(set(DATA[DATA['run']==run]['trial']))])), key=natural_keys)
        ntrials_to_add += ntrials[run_list[idx-1]]
        print ntrials_to_add

        for trial in trial_list:
            old_trial_num = int(trial.split('trial')[-1])
            new_trial_num = old_trial_num + ntrials_to_add
            new_trial_name = 'trial%05d' % new_trial_num
            print "%s --> %s" % (trial, new_trial_name)
            trial_idxs = DATA.index[((DATA['run'] == run) & (DATA['trial'] == trial))].tolist()

            DATA.loc[trial_idxs, 'trial'] = new_trial_name

    # Rename config names to match across runs (default, use FIRST run's stimconfigs)
    replace_config_ids = {}
    refrun = run_list[0]
    for run in run_list[1:]:
        for configname in sorted(stimconfigs[refrun].keys(), key=natural_keys):
            corresp_configname = [c for c in stimconfigs[run].keys() if stimconfigs[run][c] == stimconfigs[refrun][configname]][0]
            print "%s --> %s" % (configname, corresp_configname)

            if not configname == corresp_configname:
                replace_config_ids[configname] =  DATA.index[((DATA['run'] == run) & (DATA['config'] == corresp_configname))].tolist()

    for configname in replace_config_ids.keys():
        DATA.loc[replace_config_ids[configname], 'config'] = configname

    # Save reference stimconfig info:
    configs = stimconfigs[refrun]
    with open(os.path.join(combined_tracedir, 'stimulus_configs.json'), 'w') as f:
        json.dump(configs, f, sort_keys=True, indent=4)

    return DATA, configs

#%%
def get_combined_data(run_list, traceid_list, combined_tracedir, trace_type, combine_new=False):

    acquisition_dir = os.path.split(os.path.split(combined_tracedir)[0])[0]
    TIDs, data_fpaths, stimconfigs, alignment_info = get_info_for_runs(run_list, traceid_list, trace_type, acquisition_dir)

    # Check if existing dataframe:
    combined_roidata_fpath = os.path.join(combined_tracedir, 'ROIDATA_%s_%s.hdf5' % ('_'.join([data_fpaths[r]['roidata_hash'] for r in data_fpaths.keys()]), trace_type))
    datakey = '%s_%s' % (trace_type, '_'.join([data_fpaths[r]['roidata_hash'] for r in data_fpaths.keys()]))

    if os.path.exists(combined_roidata_fpath) and combine_new is False:
        print "Loading existing file..."
        try:
            DATA = pd.read_hdf(combined_roidata_fpath, key=datakey, mode='r')
            assert len(DATA) > 0, "Empty ROIDATA file. Creating new..."
            print "Retrieved existing ROIDATA file:"
            print combined_roidata_fpath
            return DATA, datakey
        except Exception as e:
            combine_new = True
    else:
        combine_new = True

    if combine_new is True:
        print "Combining %i runs into single dataframe." % len(run_list)

        #% Combine all data from all runs into single dataframe:
        DATA = roidata_collate_runs(data_fpaths)

        #% Rename trials based on run:
        DATA, configs = reindex_data_fields(DATA, run_list, stimconfigs, combined_tracedir)

        #% Save collated data:
        print "Saving combined ROIDATA..."
        t_datastore = time.time()

        # Save new combined dataframe to disk:
        DATA.to_hdf(combined_roidata_fpath, datakey,  mode='w')

        print_elapsed_time(t_datastore)

    return DATA, datakey

#%%
def get_combined_stats(DATA, datakey, combined_tracedir, trace_type='raw', filter_pupil=False, pupil_params=None):

    if filter_pupil is False:
        metric_desc = 'unfiltered_%d' % pupil_params['hash']
    else:
        metric_desc = 'pupil_rmin%.2f-rmax%.2f-dist%.2f_%s' % (pupil_params['radius_min'], pupil_params['radius_max'], pupil_params['dist_thr'], pupil_params['hash'])

    print "Requested METRIC:", metric_desc

    # Check to see if stats file already exists:
    combined_metrics_dir = os.path.join(combined_tracedir, 'metrics', metric_desc)
    if not os.path.exists(combined_metrics_dir):
        os.makedirs(combined_metrics_dir)
    metric_hash = metric_desc.split('_')[-1]

    existing_stats = sorted([f for f in os.listdir(combined_metrics_dir) if 'roi_stats_%s_%s_' % (metric_hash, trace_type) in f and f.endswith('hdf5')])
    if len(existing_stats) > 0:
        existing_stats_path = os.path.join(combined_metrics_dir, existing_stats[-1])
        STATS = pd.read_hdf(existing_stats_path, datakey, mode='r')
        print "Retrieved existing STATS file:"
        print existing_stats_path
        return STATS

    # Otherwise, check if metrics file exists:
    existing_metrics = sorted([f for f in os.listdir(combined_metrics_dir) if 'roi_metrics_%s_%s_' % (metric_hash, trace_type) in f and f.endswith('hdf5')])
    if len(existing_metrics) > 0:
        print "Loading existing combined METRICS..."
        existing_metrics_path = os.path.join(combined_metrics_dir, existing_metrics[-1])
        metrics = pd.read_hdf(existing_metrics_path, datakey, mode='r')
    else:
        print "Creating combined METRICS..."
        metrics, passtrials = acq.calculate_metrics(DATA, filter_pupil=filter_pupil, pupil_params=pupil_params)

        # METRICS output is a dict of dataframes, collate into single:-----------------
#        df_list = []
#        for roi in metrics.keys():
#            if '/' in roi:
#                roiname = roi[1:]
#            else:
#                roiname = roi
#            dfr = metrics[roi]
#            dfr['roi'] = pd.Series(np.tile(roiname, (len(dfr .index),)), index=dfr.index)
#            df_list.append(dfr)
#        METRICS = pd.concat(df_list, axis=0, ignore_index=True)

        # Save METRICS/STATS files:-----------------------------------------------------------
        datestr = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        combined_metrics_filepath = os.path.join(combined_metrics_dir, 'roi_metrics_%s_%s_%s.hdf5' % (metric_hash, trace_type, datestr))
        combined_stats_filepath = os.path.join(combined_metrics_dir, 'roi_stats_%s_%s_%s.hdf5' % (metric_hash, trace_type, datestr))
#        METRICS.to_hdf(combined_metrics_filepath, datakey,  mode='w')
        datastore = pd.HDFStore(combined_metrics_filepath, 'w')
        for roi in metrics.keys():
            datastore[str(roi)] = metrics[roi]
        datastore.close()
        #os.chmod(combined_metrics_filepath, S_IREAD|S_IRGRP|S_IROTH)
        with open(os.path.join(combined_metrics_dir, 'pass_trials.json'), 'w') as f:
            json.dump(passtrials, f, indent=4, sort_keys=True)


#    if isinstance(METRICS, pd.DataFrame):
#        metrics = {}
#        roi_list = list(set(METRICS['roi']))
#        for roi in roi_list:
#            metrics[roi] = METRICS[METRICS['roi']==roi].drop(['roi'], axis=1)
#    else:
#        metrics = METRICS

    # Get STATS:
    # Load common stim-config info:
    with open(os.path.join(combined_tracedir, 'stimulus_configs.json'), 'r') as f:
        configs = json.load(f)

    STATS = acq.collate_roi_stats(metrics, configs) # expects dict of DFs, so pass 'metrics' not 'METRICS'
    STATS.to_hdf(combined_stats_filepath, datakey,  mode='w')

    return STATS


#%%

def extract_options(options):

    parser = optparse.OptionParser()

    parser.add_option('-D', '--root', action='store', dest='rootdir',
                          default='/nas/volume1/2photon/data',
                          help='data root dir (dir containing all animalids) [default: /nas/volume1/2photon/data, /n/coxfs01/2pdata if --slurm]')
    parser.add_option('-i', '--animalid', action='store', dest='animalid',
                          default='', help='Animal ID')

    # Set specific session/run for current animal:
    parser.add_option('-S', '--session', action='store', dest='session',
                          default='', help='session dir (format: YYYMMDD_ANIMALID')
    parser.add_option('-A', '--acq', action='store', dest='acquisition',
                          default='FOV1', help="acquisition folder (ex: 'FOV1_zoom3x') [default: FOV1]")
    parser.add_option('-T', '--trace-type', action='store', dest='trace_type',
                          default='raw', help="trace type [default: 'raw']")

    parser.add_option('-R', '--run', dest='run_list', default=[], nargs=1,
                          action='append',
                          help="run ID in order of runs")
    parser.add_option('-t', '--traceid', dest='traceid_list', default=[], nargs=1,
                          action='append',
                          help="trace ID in order of runs")

    parser.add_option('--slurm', action='store_true', dest='slurm', default=False, help="set if running as SLURM job on Odyssey")
    parser.add_option('--default', action='store_true', dest='auto', default=False, help="set if want to use all defaults")
    parser.add_option('--new', action='store_true', dest='combine_new', default=False, help="set if want to combine anew")
#
#    parser.add_option('-z', '--zscore', action="store",
#                      dest="zscore_thr", default=2.0, help="Cut-off min zscore value [default: 2.0]")

    # Pupil filtering info:
    parser.add_option('--no-pupil', action="store_false",
                      dest="filter_pupil", default=True, help="Set flag NOT to filter PSTH traces by pupil threshold params")
    parser.add_option('-s', '--radius-min', action="store",
                      dest="pupil_radius_min", default=25, help="Cut-off for smnallest pupil radius, if --pupil set [default: 25]")
    parser.add_option('-B', '--radius-max', action="store",
                      dest="pupil_radius_max", default=65, help="Cut-off for biggest pupil radius, if --pupil set [default: 65]")
    parser.add_option('-d', '--dist', action="store",
                      dest="pupil_dist_thr", default=5, help="Cut-off for pupil distance from start, if --pupil set [default: 5]")

    (options, args) = parser.parse_args(options)


    return options #, fop.run_list



#%% Run info:


#rootdir = '/mnt/odyssey'
#animalid = 'CE077'
#session = '20180321'
#acquisition = 'FOV1_zoom1x'
##run_list = ['blobs_run3', 'blobs_run4']
##traceid_list = ['traces001', 'traces001']
#
#trace_type = 'np_subtracted'
#
#options = ['-D', rootdir, '-i', animalid, '-S', session, '-A', acquisition,
#           '-T', trace_type,
#           '-R', 'blobs_run3', '-t', 'traces002', '-R', 'blobs_run4', '-t', 'traces002']


#options = ['-D', '/mnt/odyssey', '-i', 'CE077', '-S', '20180329', '-A', 'FOV2_zoom1x',
#           '-T', 'raw', '--no-pupil', '--new',
#           '-R', 'gratings_run1', '-t', 'traces001',
#           '-R', 'gratings_run2', '-t', 'traces001',
#           '-R', 'gratings_run3', '-t', 'traces001',
#           '-R', 'gratings_run4', '-t', 'traces001',]

#%%

def combine_runs_and_plot(options):
    options = extract_options(options)

    rootdir = options.rootdir
    animalid = options.animalid
    session = options.session
    acquisition = options.acquisition
    slurm = options.slurm
    if slurm is True:
        rootdir = '/n/coxfs01/2p-data'

    trace_type = options.trace_type
    combine_new = options.combine_new

    run_list = options.run_list
    traceid_list = options.traceid_list

    filter_pupil = options.filter_pupil
    pupil_radius_max = float(options.pupil_radius_max)
    pupil_radius_min = float(options.pupil_radius_min)
    pupil_dist_thr = float(options.pupil_dist_thr)


    fov = acquisition.split('_')[0]
    stimulus = run_list[0].split('_')[0]

    #%% Create output dir:
    session_dir = os.path.join(rootdir, animalid, session, acquisition)
    combined_rundir = os.path.join(session_dir, '_'.join(run_list))
    combined_tracedir = os.path.join(combined_rundir, '_'.join(traceid_list))
    if not os.path.exists(combined_tracedir):
        os.makedirs(combined_tracedir)

    # Load combined dataframe, or create new, if none exists:
    DATA, datakey = get_combined_data(run_list, traceid_list, combined_tracedir, trace_type, combine_new=combine_new)

    # Load common stim-config info:
    with open(os.path.join(combined_tracedir, 'stimulus_configs.json'), 'r') as f:
        configs = json.load(f)


    #%% Set pupil filtering info:

    #filter_pupil = True
    #pupil_radius_min = 20
    #pupil_radius_max = 60
    #pupil_dist_thr = 3
    #pupil_max_nblinks = 0

    if filter_pupil is True:
        pupil_params = acq.set_pupil_params(radius_min=pupil_radius_min,
                                            radius_max=pupil_radius_max,
                                            dist_thr=pupil_dist_thr,
                                            create_empty=False)
    elif filter_pupil is False:
        pupil_params = acq.set_pupil_params(create_empty=True)

    #% Calculate metrics & get stats ---------------------------------------------
    STATS = get_combined_stats(DATA, datakey, combined_tracedir, trace_type=trace_type, filter_pupil=filter_pupil, pupil_params=pupil_params)

    #%% if filter pupil, get subset of DATA:
    if filter_pupil is True:
        DATA = DATA.query('pupil_size_stimulus > @pupil_radius_min \
                               & pupil_size_baseline > @pupil_radius_min \
                               & pupil_size_stimulus < @pupil_radius_max \
                               & pupil_size_baseline < @pupil_radius_max \
                               & pupil_dist_stimulus < @pupil_dist_thr \
                               & pupil_dist_baseline < @pupil_dist_thr \
                               & pupil_nblinks_stim <= @pupil_max_nblinks \
                               & pupil_nblinks_baseline >= @pupil_max_nblinks')

        stimbar_color = 'k'
        trace_color = 'b'
    else:
        stimbar_color = 'r'
        trace_color = 'k'

    #%%
    selected_metric = vis.get_metric_set(combined_tracedir, filter_pupil=filter_pupil,
                                         pupil_radius_min=pupil_radius_min,
                                         pupil_radius_max=pupil_radius_max,
                                         pupil_dist_thr=pupil_dist_thr
                                         )

    visualization_method = 'separate_transforms'
    metric_type = 'zscore'

    combined_runs_figdir_tuning = os.path.join(combined_tracedir, 'figures', 'tuning', trace_type, metric_type, selected_metric, visualization_method)
    if not os.path.exists(combined_runs_figdir_tuning):
        os.makedirs(combined_runs_figdir_tuning)

    roi_list = sorted(list(set(STATS['roi'])), key=natural_keys)
    transform_dict, object_transformations = vis.get_object_transforms(DATA)
    for roi in roi_list:
        print roi
        roiDF = STATS[STATS['roi']==roi]

        fignames = vis.plot_tuning_by_transforms(roiDF, transform_dict, object_transformations,
                                                 metric_type=metric_type, save_and_close=True,
                                                 output_dir = combined_runs_figdir_tuning,
                                                 include_trials=False) #output_dir='/tmp', include_trials=True)

    combined_runs_figdir_psth = os.path.join(combined_tracedir, 'figures', 'psth', trace_type, metric_type, selected_metric, visualization_method)
    if not os.path.exists(combined_runs_figdir_psth):
        os.makedirs(combined_runs_figdir_psth)

    for roi in roi_list:
        roiDF = DATA[DATA['roi']==roi]
        vis.plot_roi_psth(roi, roiDF, object_transformations,
                          figdir=combined_runs_figdir_psth, prefix='psth',
                          trace_color=trace_color, stimbar_color=stimbar_color,
                          save_and_close=True)

    #%% Look at MAX ZSCORE:

    # Get stats on ROIs:
    trans_types = object_transformations.keys()
    grouped = STATS.groupby([trans_types], as_index=False)         # Group dataframe by variables-of-interest

    zscores = grouped[metric_type].mean()                                                # Get mean of 'metric_type' for each combination of transforms
    zscores['sem_%s' % metric_type] = grouped[metric_type].aggregate(stats.sem)[metric_type]             # Get SEM
    zscores = zscores.rename(columns={metric_type: 'mean_%s' % metric_type})                # Rename 'zscore' column to 'mean_zscore' so we can merge
    STATS = STATS.merge(zscores)#.sort_values([xval_trans])                         # Merge summary stats to each corresponding row (indexed by columns values in that row)

    # Get max zscore across all configs for each ROI:
    max_config_zscores = [max(list(set(STATS[STATS['roi']==roi]['mean_%s' % metric_type]))) for roi in roi_list]
    pl.figure()
    sns.distplot(max_config_zscores)
    pl.xlabel('max zscore')
    pl.title("%s %s %s %s" % (animalid, session, fov, stimulus))

    curr_tuning_dir = os.path.join(combined_tracedir, 'figures', 'tuning')
    figname = "hist_rois_max_%s_%s_%s.png" % (trace_type, metric_type, selected_metric)
    figpath = os.path.join(curr_tuning_dir, figname)
    pl.savefig(figpath)

    #%% Look at STIM_DF:

    metric_type = 'stim_df'
    zscores = grouped[metric_type].mean()                                                # Get mean of 'metric_type' for each combination of transforms
    zscores['sem_%s' % metric_type] = grouped[metric_type].aggregate(stats.sem)[metric_type]             # Get SEM
    zscores = zscores.rename(columns={metric_type: 'mean_%s' % metric_type})                # Rename 'zscore' column to 'mean_zscore' so we can merge
    STATS = STATS.merge(zscores)#.sort_values([xval_trans])                         # Merge summary stats to each corresponding row (indexed by columns values in that row)

    STATS[STATS['roi']==roi]['mean_stim_df']

    max_config_stimdfs = [max(list(set(STATS[STATS['roi']==roi]['mean_stim_df']))) for roi in roi_list]
    pl.figure()
    sns.distplot(max_config_stimdfs)
    pl.xlabel('max df/f during stimulus')
    pl.title("%s %s %s %s" % (animalid, session, fov, stimulus))

    figname = "hist_rois_max_%s_%s_%s.png" % (trace_type, metric_type, selected_metric)
    figpath = os.path.join(curr_tuning_dir, figname)
    pl.savefig(figpath)

    return combined_tracedir
#%%

def main(options):

    combined_tracedir = combine_runs_and_plot(options)

    print "*******************************************************************"
    print "Done combining runs!"
    print "-------------------------------------------------------------------"
    print "Outputs saved to: %s" % combined_tracedir
    print "*******************************************************************"



if __name__ == '__main__':
    main(sys.argv[1:])
