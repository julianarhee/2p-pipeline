#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 18:47:08 2018

@author: juliana
"""

import os
import sys
import datetime
import h5py
import math
import numpy as np
import json
import operator
import optparse
import seaborn as sns
import pylab as pl
import pandas as pd
import cPickle as pkl
from optparse import OptionParser
import matplotlib as mpl
from matplotlib.lines import Line2D

from pipeline.python.utils import natural_keys
from pipeline.python.traces.utils import get_metric_set
import pipeline.python.retinotopy.visualize_rois as ret

#%%
#rootdir = '/mnt/odyssey'
#animalid = 'CE074'
#session = '20180215'
##acquisition = 'FOV1_zoom1x_V1'
#acquisition = 'FOV2_zoom1x_LI'
#
#acquisition_dir = os.path.join(rootdir, animalid, session, acquisition)
#
##
#run1 = 'gratings_phasemod'
#run1_traceid = 'traces004'
#run1_traceid = 'traces002'
#
#run2 = 'blobs'
#run2_traceid = 'traces003'
#run2_traceid = 'traces003'

#%%


class RunBase(object):
    def __init__(self, run):
        print run
        self.run = run
        self.traceid = None
        self.pupil_radius_min = None
        self.pupil_radius_max = None
        self.pupil_dist_thr = None
        self.pupil_max_nblinks = 1

    def set_params(self, paramslist):
        #params = getattr(parservalues, 'trace_info')
        self.traceid = paramslist[0]
        if len(paramslist) > 1:
            self.pupil_radius_min = paramslist[1]
            self.pupil_radius_max = paramslist[2]
            self.pupil_dist_thr = paramslist[3]


class FileOptionParser(object):
    def __init__(self):
        self.last_run = None
        self.run_list = []

    def set_info(self, option, opt, value, parser):
        if option.dest=="run":
            print "Creating"
            cls = RunBase
        else:
            assert False

        print value
        self.last_run = cls(value)
        self.run_list.append(self.last_run)
        setattr(parser.values, option.dest, self.last_run)


def extract_options(options):
    fop = FileOptionParser()

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

    parser.add_option('-R', '--run', dest='run', type='string',
                          action='callback', callback=fop.set_info, help="Supply multiple runs for comparison (currently, expects only 2 runs)")

    parser.add_option('-t', '--traces', dest='trace_info', default=[], nargs=1,
                          action='append',
                          help="Comma-sep string of run, traceid, pupil rad min, pupil rad max, pupil dist thr (for ex, -t blobs,traces001,30,50,5)")


    parser.add_option('--slurm', action='store_true', dest='slurm', default=False, help="set if running as SLURM job on Odyssey")
    parser.add_option('--default', action='store_true', dest='auto', default=False, help="set if want to use all defaults")
#    parser.add_option('-z', '--zscore', action="store",
#                      dest="zscore_thr", default=2.0, help="Cut-off min zscore value [default: 2.0]")
    parser.add_option('--positions', action='store_true', dest='colormap_position', default=False, help="set if want to view position responses as color map (retinotopy)")

    parser.add_option('-B', '--bbox', dest='boundingbox_runs', default=[], nargs=1, action='append', help="RUN that is a bounding box run (only for retino)")
    parser.add_option('-l', '--left', dest='leftedge', default=None, action='store', help="left edge of bounding box")
    parser.add_option('-r', '--right', dest='rightedge', default=None, action='store', help="right edge of bounding box")
    parser.add_option('-u', '--upper', dest='topedge', default=None, action='store', help="upper edge of bounding box")
    parser.add_option('-b', '--lower', dest='bottomedge', default=None, action='store', help="bottom edge of bounding box")

    #    parser.add_option('-t', '--trace-id', action='store', dest='trace_id', default='', help="Trace ID for current trace set (created with set_trace_params.py, e.g., traces001, traces020, etc.)")

    # Pupil filtering info:
#    parser.add_option('--no-pupil', action="store_false",
#                      dest="filter_pupil", default=True, help="Set flag NOT to filter PSTH traces by pupil threshold params")
#    parser.add_option('-r', '--rad', action="store",
#                      dest="pupil_size_thr", default=25, help="Cut-off for pupil radius, if --pupil set [default: 30]")
#    parser.add_option('-d', '--dist', action="store",
#                      dest="pupil_dist_thr", default=15, help="Cut-off for pupil distance from start, if --pupil set [default: 5]")
#    parser.add_option('-b', '--blinks', action="store",
#                      dest="pupil_max_nblinks", default=1, help="Cut-off for N blinks allowed in trial, if --pupil set [default: 1 (i.e., 0 blinks allowed)]")

    (options, args) = parser.parse_args(options)

    for f in fop.run_list:
        run_params = [t for t in options.trace_info if f.run in t][0]
        print run_params
        params_list = [p for p in run_params.split(',') if not p==f.run]
        f.set_params(params_list)
    #print [(f.run, f.traceid, f.pupil_max_nblinks) for f in fop.run_list]

    return options, fop.run_list

#%%

def get_dataframe_paths(acquisition_dir, trace_info, trace_type='raw'):
    dfpaths = dict()
    for idx, info in enumerate(trace_info):
        dfilepath = None
        rkey = 'run%i' % int(idx+1)

        #runs[rkey]['run'] = info.run
        is_combo = False
        if 'analysis' in info.traceid:
            # This is likely a retino run
            trace_basename = 'retino_analysis'
            tdict_path = os.path.join(acquisition_dir, info.run, trace_basename, 'analysisids_%s.json' % info.run)
            hash_type = 'analysis_hash'
            is_retino = True
        else:
            is_retino = False
            trace_basename = 'traces'
            if len(info.traceid.split('_'))>1:
                traceid_dir = os.path.join(acquisition_dir, info.run, info.traceid)
                is_combo = True
            else:
                tdict_path = os.path.join(acquisition_dir, info.run, trace_basename, 'traceids_%s.json' % info.run)
                hash_type = 'trace_hash'
                is_combo = False

        if is_combo is False:
            with open(tdict_path, 'r') as f:
                tdict = json.load(f)
            trace_idname = '%s_%s' % (info.traceid, tdict[info.traceid][hash_type])
            traceid_dir = os.path.join(acquisition_dir, info.run, trace_basename, trace_idname)

        if is_retino:
            dfilepath = [os.path.join(traceid_dir, 'files', f) for f in os.listdir(os.path.join(traceid_dir, 'files')) if 'retino_data' in f]
        else:
            if info.pupil_dist_thr is None or info.pupil_radius_min is None:
                pupil_str = 'unfiltered_'
            else:
                pupil_str = 'pupil_rmin%.2f-rmax%.2f-dist%.2f' % (float(info.pupil_radius_min), float(info.pupil_radius_max), int(info.pupil_dist_thr))
            pupil_dir = [os.path.join(traceid_dir, 'metrics', p) for p in os.listdir(os.path.join(traceid_dir, 'metrics')) if pupil_str in p][0]

            dfilepath = [os.path.join(pupil_dir, f) for f in os.listdir(pupil_dir) if 'roi_stats_' in f and trace_type in f][0]
        dfpaths[rkey] = dfilepath

    return dfpaths

#%%
def create_zscore_df(dfpaths):
    all_dfs = []

    retino_runs = [f for f in dfpaths.keys() if 'retino_analysis' in dfpaths[f][0]]
    event_runs = [f for f in dfpaths.keys() if 'traces' in dfpaths[f]]

    for retino_run in retino_runs:
        #retino_run = retino_runs[0]
        rundir = (dfpaths[retino_run][0]).split('/retino_analysis')[0]
        paradigmdir = os.path.join(rundir, 'paradigm', 'files')
        paradigm_info_fn = [f for f in os.listdir(paradigmdir) if 'parsed_' in f][0]
        with open(os.path.join(paradigmdir, paradigm_info_fn), 'r') as f:
            pinfo = json.load(f)
        ratiomat = []
        condlist= []
        phasemat =[]
        for di,df in enumerate(dfpaths[retino_run]):
            if os.stat(df).st_size == 0:
                continue
            rundf = h5py.File(df, 'r')
            if len(rundf) == 0:
                continue
            run_name = os.path.split(df.split('/retino_analysis')[0])[-1]
            print "Getting mag-ratios for each ROI in run: %s, trial %i" % (run_name, di+1)
            roi_list = ['roi%05d' % int(r+1) for r in range(rundf['mag_ratio_array'].shape[0])]
            ratiomat.append(rundf['mag_ratio_array'][:])
            phasemat.append(rundf['phase_array'][:])
            condlist.append(pinfo[str(di+1)]['stimuli']['stimulus'])
        phasemat = np.array(phasemat)
        ratiomat = np.array(ratiomat)
        max_zscores_by_trial = np.max(ratiomat, axis=0)

        # Get corresponding stim info for TRIAL based:
        nrois = len(roi_list)
        trial_idxs_by_trial = [np.where(ratiomat[:,ridx]==ratiomat[:,ridx].max())[0][0] for ridx in range(nrois)]
        conds_by_trial = [condlist[cidx] for cidx in trial_idxs_by_trial]

        # Get corresponding stim info for mean across stim configs:
        mats_by_stim = []
        unique_conds = sorted(list(set(condlist)))
        for cond in unique_conds:
            idxs = np.where(np.array(condlist)==cond)[0]
            currmat = np.mean(ratiomat[idxs,:], axis=0)
            mats_by_stim.append(currmat)
        mats_by_stim = np.array(mats_by_stim)
        max_zscores_by_stim = np.max(mats_by_stim, axis=0)

        trial_idxs_by_stim = [np.where(mats_by_stim[:,ridx]==mats_by_stim[:,ridx].max())[0][0] for ridx in range(nrois)]
        conds_by_stim = [unique_conds[cidx] for cidx in trial_idxs_by_stim]

        curr_df = pd.DataFrame({'roi': roi_list,
                                'run': np.tile(run_name, (nrois,)),
                                'max_zscore_trial': np.array(max_zscores_by_trial),
                                'trialnum_trial': trial_idxs_by_trial,
                                'condition_trial': conds_by_trial,
                                'max_zscore_stim':np.array(max_zscores_by_stim),
                                #'trialnum_stim': trial_idxs_by_stim,
                                'condition_stim': conds_by_stim
                                })

        # Concatenate all info for this current trial:
        all_dfs.append(curr_df)

    for event_run in event_runs: #df in dfpaths.values():
        df = dfpaths[event_run]
        traceid = os.path.split(df.split('/metrics')[0])[-1]
        if len(traceid.split('_')) > 1:
            is_combo = True
        else:
            is_combo = False
        if is_combo:
            rundf = pd.HDFStore(df, 'r')
            rundf = rundf[rundf.keys()[0]]
        else:
            rundf = pd.HDFStore(df, 'r')['/df']

        run_name = os.path.split(df.split('/traces')[0])[-1]
        run_name = os.path.split(df.split('/traces')[0])[-1]
        if len(run_name.split('_')) > 3:
            run_name = run_name.split('_')[0]

        print "Compiling zscores for each ROI in run: %s" % run_name
        roi_list = sorted(list(set(rundf['roi'])), key=natural_keys)

        trial_list = sorted(list(set(rundf['trial'])), key=natural_keys)
        confg_list = sorted(list(set(rundf['config'])), key=natural_keys)
        if 'mean_zscore' in rundf.keys():
            max_zscores_by_trial = [max(list(set((rundf[rundf['roi']==roi]['zscore'])))) for roi in sorted(roi_list, key=natural_keys)]
            max_zscores_by_stim = [max(list(set((rundf[rundf['roi']==roi]['mean_zscore'])))) for roi in sorted(roi_list, key=natural_keys)]

            #trial_idxs_by_stim =  [str(rundf[((rundf['roi']==roi) & (rundf['mean_zscore']==zval))]['trial'].values[0]) for roi,zval in zip(roi_list, max_zscores_by_stim)]
            conds_by_stim =  [str(rundf[((rundf['roi']==roi) & (rundf['mean_zscore']==zval))]['config'].values[0]) for roi,zval in zip(roi_list, max_zscores_by_stim)]
        else:
            max_zscores_by_trial = [max([np.float(rundf[((rundf['roi']==roi) & (rundf['trial']==trial))]['zscore'])
                                    for trial in trial_list]) for roi in sorted(roi_list, key=natural_keys)]
#            max_zscores_by_stim = [max([np.nanmean(rundf[((rundf['roi']==roi) & (rundf['config']==config))]['zscore'])
#                                    for config in confg_list]) for roi in sorted(roi_list, key=natural_keys)]

            cfg_dict = dict((roi,
                             dict((config, np.nanmean(rundf[((rundf['roi']==roi) & (rundf['config']==config))]['zscore'])) for config in confg_list))
                                for roi in roi_list)
            max_zscores_by_stim = [max(cfg_dict[roi].items(), key=operator.itemgetter(1))[1] for roi in roi_list]
            conds_by_stim = [max(cfg_dict[roi].items(), key=operator.itemgetter(1))[0] for roi in roi_list]

        # Get corresponding stim info for TRIAL based:
        nrois = len(roi_list)
        trial_idxs_by_trial = [str(rundf[((rundf['roi']==roi) & (rundf['zscore']==zval))]['trial'].values[0]) for roi,zval in zip(roi_list, max_zscores_by_trial)]
        conds_by_trial = [str(rundf[((rundf['roi']==roi) & (rundf['zscore']==zval))]['config'].values[0]) for roi,zval in zip(roi_list, max_zscores_by_trial)]

        if '/' in roi_list[0]:
            roi_list = [r[1:] for r in roi_list]


        nrois = len(roi_list)
        curr_df = pd.DataFrame({'roi': roi_list,
                                'run': np.tile(run_name, (nrois,)),
                                'max_zscore_trial': np.array(max_zscores_by_trial),
                                'trialnum_trial': trial_idxs_by_trial,
                                'condition_trial': conds_by_trial,
                                'max_zscore_stim':np.array(max_zscores_by_stim),
                                #'trialnum_stim': trial_idxs_by_stim,
                                'condition_stim': conds_by_stim
                                })

        # Concatenate all info for this current trial:
        all_dfs.append(curr_df)

    # Finally, concatenate all trials across all configs for current ROI dataframe:
    DF = pd.concat(all_dfs, axis=0,  ignore_index=True)

    return DF


#%%

def phase_to_grid(nrows, ncols, start_right=True, start_top=True):
    azimuths = np.linspace(0.0, 2*math.pi, num=ncols+1)
    elevations = np.linspace(0.0, 2*math.pi, num=nrows+1)
    if start_right is True:
        azimuths = azimuths[::-1]  # Flip l/r so that 2pi is left, 0 is right
    if start_top is False:
        elevations = elevations[::-1]

    return azimuths, elevations

def position_to_grid(xpositions, ypositions):
    positions = []
    if any([y < 0 for y in ypositions]) and ypositions[0] < 0:
        ypositions = ypositions[::-1]

    for x in xpositions:
        for y in ypositions:
            positions.append((x, y))
    positions = sorted(positions , key=lambda k: [k[1], k[0]])

    return positions

def get_grid_info(nrows, ncols, xpositions, ypositions):
    azimuths, elevations = phase_to_grid(nrows, ncols)
    #positions =  position_to_grid(xpositions, ypositions)

    xpositions = sorted(xpositions)
    xpos_idxs = list(np.copy(xpositions))
    xpos_idxs.extend(xpositions[::-1])
    ypositions = sorted(ypositions)[::-1]
    ypos_idxs = list(np.tile(ypositions[0], (len(xpositions),)))
    ypos_idxs.extend(list(np.tile(ypositions[1], (len(xpositions),))))
    gridinfo = {}
    i=0
    for row in range(nrows):
        for col in range(ncols):
            gridinfo[i] = {}
            gridinfo[i]['coords'] = (row, col)
            gridinfo[i]['xpos'] = xpos_idxs[i] #xpositions[col]
            gridinfo[i]['ypos'] = ypos_idxs[i] #ypositions[row]
            gridinfo[i]['right'] = sorted((azimuths[col], azimuths[col+1]))
            gridinfo[i]['top'] = sorted((elevations[row], elevations[row+1]))
            i+=1
    return gridinfo



#%%
def get_metricdf_stimpositions(dfpaths):
    #all_dfs = []
    #retino_runs = [f for f in dfpaths.keys() if 'retino_analysis' in dfpaths[f][0]]
    #event_runs = [f for f in dfpaths.keys() if 'traces' in dfpaths[f]]
    run_list = dfpaths.keys()
    dataframes = {}

    currdf= []
    for run in run_list:
        if 'retino_analysis' in dfpaths[run][0]:
            is_retino = True
            rundir = (dfpaths[run][0]).split('/retino_analysis')[0]
            paradigmdir = os.path.join(rundir, 'paradigm', 'files')
            paradigm_info_fn = [f for f in os.listdir(paradigmdir) if 'parsed_' in f][0]
            with open(os.path.join(paradigmdir, paradigm_info_fn), 'r') as f:
                paradigm_info = json.load(f)
            #stimconfigs = sorted(list(set([paradigm_info[f]['stimuli']['stimulus'] for f in paradigm_info.keys()])), key=natural_keys)

            for di,df in enumerate(dfpaths[run]):
                if os.stat(df).st_size == 0:
                    continue
                rundf = h5py.File(df, 'r')
                if len(rundf) == 0:
                    continue
                currconfig = paradigm_info[str(di+1)]['stimuli']['stimulus']
                currtrial = 'trial%05d' % int(di+1)
                run_name = os.path.split(df.split('/retino_analysis')[0])[-1]
                if len(run_name.split('_')) > 3:
                    run_name = run_name.split('_')[0]

                print "Getting mag-ratios for each ROI in run: %s, trial %i" % (run_name, di+1)
                roi_list = ['roi%05d' % int(r+1) for r in range(rundf['mag_ratio_array'].shape[0])]
                phase_convert = -1 * rundf['phase_array'][:]
                phase_convert = phase_convert % (2*np.pi)
                currdf.append(pd.DataFrame({'roi': roi_list,
                                       'config': np.tile(currconfig, (len(roi_list),)),
                                       'trial': np.tile(currtrial, (len(roi_list),)),
                                       'magratio': rundf['mag_ratio_array'][:],
                                       'phase': phase_convert,
                                       'run': run_name
                                       }))
            data = pd.concat(currdf, axis=0)
            metricdf = data.groupby(['config', 'roi']).agg({'magratio': {'magratio_mean': 'mean', 'magratio_max': 'max'},
                                                            'phase': {'phase_mean': 'mean'}
                                                            })
            # Get phases at max mag-ratio:
            phases_at_max = [data[(data['roi']==ind[1]) & (data['config']==ind[0]) & (data['magratio']==val)]['phase'].values[0]
                                for ind, val in zip(metricdf['magratio']['magratio_max'].index.tolist(), metricdf['magratio']['magratio_max'].values)]
            metricdf.columns = metricdf.columns.get_level_values(1)
            metricdf['phase_atmax'] = phases_at_max

        elif 'traces' in dfpaths[run]:
            is_retino = False
            df = dfpaths[run]
            traceid = os.path.split(df.split('/metrics')[0])[-1]
            if len(traceid.split('_')) > 1:
                is_combo = True
            else:
                is_combo = False
            if is_combo:
                rundf = pd.HDFStore(df, 'r')
                rundf = rundf[rundf.keys()[0]]
            else:
                rundf = pd.HDFStore(df, 'r')['/df']

            run_name = os.path.split(df.split('/traces')[0])[-1]
            if len(run_name.split('_')) > 3:
                run_name = run_name.split('_')[0]

            print "Compiling zscores for each ROI in run: %s" % run_name
            roi_list = sorted(list(set(rundf['roi'])), key=natural_keys)

            subdf = rundf[['roi', 'config', 'trial', 'stim_df', 'zscore', 'xpos', 'ypos']]
            metricdf = subdf.groupby(['config', 'xpos', 'ypos', 'roi']).agg({'zscore': {'zscore_mean': 'mean', 'zscore_max': 'max'},
                                                             'stim_df': {'stimdf_mean': 'mean', 'stimdf_max': 'max'}
                                                             })
            # Concatenate all info for this current trial:
            metricdf.columns = metricdf.columns.get_level_values(1)

        dataframes[run_name] = {}
        dataframes[run_name]['df'] = metricdf
        dataframes[run_name]['is_phase'] = is_retino

    # Get grid info:
#    nrows = len(list(set(rundf['ypos'])))
#    ncols = len(list(set(rundf['xpos'])))
#    xpositions = list(set(rundf['xpos']))
#    ypositions = list(set(rundf['ypos']))
#    gridinfo = get_grid_info(nrows, ncols, xpositions, ypositions)
#
#    for run in dataframes.keys():
#        d = dataframes[run]
#        if 'zscore_mean' in d.keys():
#            d['gridpos'] = [[k for k in gridinfo.keys() if combo[1]==gridinfo[k]['xpos'] and combo[2]==gridinfo[k]['ypos']][0] for combo in d.index.tolist()]
#        elif 'magratio_mean' in d.keys():
#            d['gridpos'] = [[k for k in gridinfo.keys() if  gridinfo[k][combo[0]][1] >= d.loc[combo]['phase_mean'] >= gridinfo[k][combo[0]][0]] [0]
#                                for combo in d.index.tolist()]
#        dataframes[run] = d

    return dataframes #, gridinfo

#%%

def plot_joint_distn_metric(zdf, metric, combo_basename='', output_dir='', save_and_close=True):

    roi_list = sorted(list(set(zdf['roi'])), key=natural_keys)
    nrois = len(roi_list)

    runs = list(set(zdf['run']))
    run1 = runs[0]
    run2 = runs[1]
    if len(run2.split('_')) > 3:
        run2_label = run2.split('_')[0]
    else:
        run2_label = run2
    if len(run1.split('_')) > 3:
        run1_label = run1.split('_')[0]
    else:
        run1_label = run1

    run_title = 'R1: %s | R2 %s' % (run1_label, run2_label)

    # Plot as scatter:
    #min_zscore = float(options.zscore_thr) # 2.0

#    with sns.axes_style("white"):
#        g1 = sns.jointplot(x=zdf[zdf['run']==run1][metric], y=zdf[zdf['run']==run2][metric], kind="scatter", dropna = True,
#                      marginal_kws=dict(bins=5, rug=True),
#                      linewidth=1)
#        g1.ax_joint.set_xlabel('%s: %s' % (run1, metric))
#        g1.ax_joint.set_ylabel('%s: %s' % (run2, metric))
#
#    ax1_min = zdf[zdf['run']==run1][metric].min()
#    ax1_max = zdf[zdf['run']==run1][metric].max()
#    ax2_min = zdf[zdf['run']==run2][metric].min()
#    ax2_max = zdf[zdf['run']==run2][metric].max()
#
#    #g1.line([min_zscore, ax2_min], [min_zscore, ax2_max], 'k', linewidth=1, alpha=0.5)
#
#    g1.fig.suptitle(run_title)
#    figname = '%s_jointdistN.png' % combo_basename
#    pl.savefig(os.path.join(output_dir, figname))
#    #pl.close()

    pl.figure()
    x = zdf[zdf['run']==run1][metric]
    y = zdf[zdf['run']==run2][metric]
    sns.regplot(x=x, y=y, marker="+", fit_reg=False)
    pl.subplots_adjust(top=0.9)
    pl.xlabel('%s: %s' % (run1_label, metric))
    pl.ylabel('%s: %s' % (run2_label, metric))
    pl.suptitle(run_title)

    if save_and_close:
        figname = '%s_jointdistN.png' % combo_basename
        pl.savefig(os.path.join(output_dir, figname))
        pl.close()
        print "Saved fig to: %s" % os.path.join(output_dir, figname)

#%%

def assign_roi_metric(dataframes, run1_stat_type='mean', run2_stat_type='mean'):
    '''
    If retinotopy data, metric options are:
        'magratio_mean'
        'magratio_max'
    If trial-based data and looking at xpos, ypos:
        'zscore_mean'
        'zscore_max'

    Select 'mean' to get average metric across all trials/reps of a given stimulus config.
    Select 'max' to take the best trial/rep of a given stimulus config.
    '''

    run_list = dataframes.keys()

    # X-axis data:-------------------------------------------------------------
    run1 = run_list[0]
    xrun = dataframes[run1]['df']

    # Check if we want ZSCORE or MAG-RATIO:
    if dataframes[run1]['is_phase']:
        metric_type = 'magratio'
    else:
        metric_type = 'zscore'
    metric_run1 = '%s_%s' % (metric_type, run1_stat_type)
    best_zscores_run1 = xrun.groupby(['roi']).agg({metric_run1: {'%s1' % metric_run1: 'max'}})

    best_zscores_run1.columns = best_zscores_run1.columns.get_level_values(1)
    #best_zscores_run1['pos_run1'] = coresponding_pos_run1
    #corresponding_gridpos = [xrun[xrun[metric_run1]==val[0]]['gridpos'][0] for val in best_zscores_run1.values]
    #best_zscores_run1['gridpos_run1'] = corr_gridpos_run1


    # Y-axis data:-------------------------------------------------------------
    run2 = run_list[1]
    yrun = dataframes[run2]['df'] #[[metric_run2, metric_run2_value, 'gridpos']] #.loc['right'] #.loc['top'] #.loc['right']
    # Check if we want ZSCORE or MAG-RATIO:
    if dataframes[run2]['is_phase']:
        metric_type = 'magratio'
    else:
        metric_type = 'zscore'
    metric_run2 = '%s_%s' % (metric_type, run1_stat_type)
    #best_zscores_run2 = yrun.groupby(['roi']).agg({metric_run2: 'max'})
    best_zscores_run2 = yrun.groupby(['roi']).agg({metric_run2: {'%s2' % metric_run2: 'max'}})
    best_zscores_run2.columns = best_zscores_run2.columns.get_level_values(1)

    #corr_gridpos_run2 = [yrun[yrun[metric_run2]==val[0]]['gridpos'][0] for val in best_zscores_run2.values]
    #best_zscores_run2['gridpos_run2'] = corr_gridpos_run2
    #coresponding_pos_run2 = [yrun[yrun[metric_run2]==val[0]][metric_run2_value][0] for val in best_zscores_run2.values]
    #best_zscores_run2['pos_run2'] = coresponding_pos_run2

    # Concatenate into dataframe:
    zdf = pd.concat([best_zscores_run1, best_zscores_run2], axis=1)

    return zdf


#%%
#def convert_values(oldval, newmin, newmax, oldmax=None, oldmin=None):
#    oldrange = (oldmax - oldmin)
#    newrange = (newmax - newmin)
#    newval = (((oldval - oldmin) * newrange) / oldrange) + newmin
#    return newval
#
## Convert degs to centimeters:
#def get_linear_coords(width, height, resolution, leftedge=None, rightedge=None, bottomedge=None, topedge=None):
#    #width = 103 # in cm
#    #height = 58 # in cm
#    #resolution = [1920, 1080]
#
#    if leftedge is None:
#        leftedge = -1*width/2.
#    if rightedge is None:
#        rightedge = width/2.
#    if bottomedge is None:
#        bottomedge = -1*height/2.
#    if topedge is None:
#        topedge = height/2.
#
#    print "center 2 Top/Anterior:", topedge, rightedge
#
#
#    mapx = np.linspace(leftedge, rightedge, resolution[0] * ((rightedge-leftedge)/float(width)))
#    mapy = np.linspace(bottomedge, topedge, resolution[1] * ((topedge-bottomedge)/float(height)))
#
#    lin_coord_x, lin_coord_y = np.meshgrid(mapx, mapy, sparse=False)
#
#    return lin_coord_x, lin_coord_y
#
#def get_retino_info(width=80, height=44, resolution=[1920, 1080],
#                    azimuth='right', elevation='top',
#                    leftedge=None, rightedge=None, bottomedge=None, topedge=None):
#
#    lin_coord_x, lin_coord_y = get_linear_coords(width, height, resolution, leftedge=leftedge, rightedge=rightedge, bottomedge=bottomedge, topedge=topedge)
#    linminW = lin_coord_x.min(); linmaxW = lin_coord_x.max()
#    linminH = lin_coord_y.min(); linmaxH = lin_coord_y.max()
#
#    retino_info = {}
#    retino_info['width'] = width
#    retino_info['height'] = height
#    retino_info['resolution'] = resolution
#    aspect_ratio = float(height)/float(width)
#    retino_info['aspect'] = aspect_ratio
#    retino_info['azimuth'] = azimuth
#    retino_info['elevation'] = elevation
#    retino_info['linminW'] = linminW
#    retino_info['linmaxW'] = linmaxW
#    retino_info['linminH'] = linminH
#    retino_info['linmaxH'] = linmaxH
#    retino_info['bounding_box'] = [leftedge, bottomedge, rightedge, topedge]
#
#    return retino_info

#%%
def assign_lincoords_lincolors(rundf, metric_values, stat_type='mean'):
    metric_name = 'zscore_%s' % stat_type
    coresponding_pos_run1 = [(rundf[rundf[metric_name]==val].index.tolist()[0][1], rundf[rundf[metric_name]==val].index.tolist()[0][2])
                            for val in metric_values]

    # Get color list for grid positions:
    linX = [p[0] for p in coresponding_pos_run1]
    linY = [p[1] for p in coresponding_pos_run1]
    linC = np.arctan2(linY,linX)

    return linX, linY, linC

#
#def convert_lincoords_lincolors(rundf, rinfo, stat_type='mean'):
#
#    angX = rundf.loc[slice(rinfo['azimuth']), 'phase_%s' % stat_type].values
#    angY = rundf.xs(rinfo['elevation'], axis=0)['phase_%s' % stat_type].values
#
#    # Convert phase range to linear-coord range:
#    linX = convert_values(angX, rinfo['linminW'], rinfo['linmaxW'], oldmax=0, oldmin=2*np.pi)  # If cond is 'right':  positive values = 0, negative values = 2pi
#    linY = convert_values(angY, rinfo['linminH'], rinfo['linmaxH'], oldmax=2*np.pi, oldmin=0)  # If cond is 'top':  positive values = 0, negative values = 2pi
#    linC = np.arctan2(linY,linX)
#
#    return linX, linY, linC
#
#
#def plot_roi_retinotopy(linX, linY, rgbas, retino_info, curr_metric='magratio_mean',
#                        alpha_min=0, alpha_max=1, color_position=False,
#                        output_dir='', figname='roi_retinotopy.png', save_and_close=True):
#    sns.set()
#    fig = pl.figure(figsize=(10,8))
#    ax = fig.add_subplot(111) #, aspect=retino_info['aspect'])
#    if color_position is True:
#        pl.scatter(linX, linY, s=150, c=rgbas, cmap='hsv', vmin=-np.pi, vmax=np.pi) #, vmin=0, vmax=2*np.pi)
#        magcmap = mpl.cm.Greys
#    else:
#        pl.scatter(linX, linY, s=150, c=rgbas, cmap='inferno', alpha=0.75, edgecolors='w') #, vmin=0, vmax=2*np.pi)
#        magcmap=mpl.cm.inferno
#
#    pl.gca().invert_xaxis()  # Invert x-axis so that negative values are on left side
##    pl.xlim([retino_info['linminW'], retino_info['linmaxW']])
##    pl.ylim([retino_info['linminH'], retino_info['linmaxH']])
#    pl.xlim([-1*retino_info['width']/2., retino_info['width']/2.])
#    pl.ylim([-1*retino_info['height']/2., retino_info['height']/2.])
#
#    pl.xlabel('x position')
#    pl.ylabel('y position')
#    pl.title('ROI position selectivity (%s)' % curr_metric)
#    pos = ax.get_position()
#    ax2 = fig.add_axes([pos.x0+.8, pos.y0, 0.01, pos.height])
#    #magcmap = mpl.cm.Greys
##    if alpha_max < 0.05:
##        alpha_max = 0.05
#    magnorm = mpl.colors.Normalize(vmin=alpha_min, vmax=alpha_max)
#    cb = mpl.colorbar.ColorbarBase(ax2, cmap=magcmap, norm=magnorm, orientation='vertical')
#
#    if save_and_close is True:
#        pl.savefig(os.path.join(output_dir, figname))
#        pl.close()

def store_session_info(rootdir, animalid, session, acquisition):
    session_info = {}
    session_info['rootdir'] = rootdir
    session_info['animalid'] = animalid
    session_info['session'] = session
    session_info['acquisition'] = acquisition

    return session_info

#%%
def visualize_position_data(dataframes, zdf, retino_info,
                            set_response_alpha=True, stat_type='mean', color_position=False,
                            acquisition_str='', output_dir='/tmp', save_and_close=True):

    # Get RGBA mapping normalized to mag-ratio values:
    norm = mpl.colors.Normalize(vmin=-np.pi, vmax=np.pi)
    cmap = mpl.cm.get_cmap('hsv')
    mapper = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    retino_runs = [k for k in retino_info.keys() if 'retino' in k]
    if color_position is False and len(retino_runs)>1:
        same_alpha=True
        alpha_min = min([zdf[k].min() for k in zdf.keys()])
        alpha_max = min([zdf[k].max() for k in zdf.keys()])
    else:
        same_alpha=False
        alpha_min=None; alpha_max=None

    visinfo = dict((run, dict()) for run in dataframes.keys())
    for runnum, run in enumerate(dataframes.keys()):
        rundf = dataframes[run]['df']
        if dataframes[run]['is_phase']:
            curr_metric = 'magratio_%s%i' % (stat_type, int(runnum+1))
            linX, linY, linC = ret.convert_lincoords_lincolors(rundf, retino_info[run], stat_type=stat_type)
        else:
            curr_metric = 'zscore_%s%i' % (stat_type, int(runnum+1))
            linX, linY, linC = assign_lincoords_lincolors(rundf, zdf[curr_metric].values, stat_type=stat_type)

        visinfo[run]['linX'] = linX
        visinfo[run]['linY'] = linY
        visinfo[run]['linC'] = linC
        visinfo[run]['metric'] = curr_metric

        rgbas = np.array([mapper.to_rgba(v) for v in linC])
        magratios = zdf[curr_metric]
        if alpha_min is None:
            alpha_min = magratios.min()
        if alpha_max is None:
            alpha_max = magratios.max()

        if set_response_alpha is True:
            alphas = np.array(magratios / magratios.max())
            rgbas[:, 3] = alphas
        visinfo[run]['rgbas'] = rgbas

        if dataframes[run]['is_phase']:
            if set_response_alpha is False:
                alphas = np.array(zdf[curr_metric] / zdf[curr_metric].max())
                rgbas[:, 3] = alphas

            # Plot each ROI's "position" color-coded with angle map:
            if color_position is True:
                figname = '%s_R%i-%s_position_selectivity_%s_Cpos.png' % (acquisition_str, int(runnum+1), run, curr_metric)
                ret.plot_roi_retinotopy(linX, linY, rgbas, retino_info[run], curr_metric=curr_metric,
                                    alpha_min=zdf[curr_metric].min(), alpha_max=zdf[curr_metric].max(),
                                    output_dir=output_dir, figname=figname, save_and_close=save_and_close)
            else:
                figname = '%s_R%i-%s_position_selectivity_%s_Cmagr.png' % (acquisition_str, int(runnum+1), run, curr_metric)
                ret.plot_roi_retinotopy(linX, linY, magratios, retino_info[run], curr_metric=curr_metric,
                                    alpha_min=alpha_min, alpha_max=alpha_max, color_position=color_position,
                                    output_dir=output_dir, figname=figname, save_and_close=save_and_close)

    return visinfo

#%%
def main(options):

#
#    options = ['-D', '/mnt/odyssey', '-i', 'CE074', '-S', '20180215', '-A', 'FOV1_zoom1x_V1', '-R', 'gratings_phasemod', '-t', 'gratings_phasemod,traces004,30,8', '-R', 'blobs', '-t', 'blobs,traces003,25,15']
#    options = ['-D', '/mnt/odyssey', '-i', 'CE074', '-S', '20180221', '-A', 'FOV1_zoom1x', '-R', 'gratings', '-t', 'gratings,traces002', '-R', 'blobs_run3', '-t', 'blobs_run3,traces002']

#    options = ['-D', '/mnt/odyssey', '-i', 'CE077', '-S', '20180316', '-A', 'FOV1_zoom1x',
#               '-R', 'gratings', '-t', 'gratings,traces002,30,60,8',
#               '-R', 'retino_run1', '-t', 'retino_run1,analysis001',
#               '-R', 'retino_run2', '-t', 'retino_run2,analysis001']

#    options = ['-D', '/mnt/odyssey', '-i', 'CE077', '-S', '20180326', '-A', 'FOV2_zoom1x',
#               '-R', 'gratings', '-t', 'gratings,traces001,10,80,12',
#               '-R', 'retino', '-t', 'retino,analysis001']
#
#    options = ['-D', '/mnt/odyssey', '-i', 'CE077', '-S', '20180329', '-A', 'FOV2_zoom1x',
#               '-R', 'gratings_run1_gratings_run2_gratings_run3_gratings_run4',
#               '-t', 'gratings_run1_gratings_run2_gratings_run3_gratings_run4,traces001_traces001_traces001_traces001',
#               '-R', 'retino', '-t', 'retino,analysis001']

#    options = ['-D', '/mnt/odyssey', '-i', 'CE077', '-S', '20180331', '-A', 'FOV1_zoom1x',
#               '-R', 'gratings_run1',
#               '-t', 'gratings_run1,traces001,20,80,8',
#               '-R', 'gratings_run2',
#               '-t', 'gratings_run2,traces001,20,80,8']


#    options = ['-D', '/mnt/odyssey', '-i', 'CE077', '-S', '20180316', '-A', 'FOV1_zoom1x',
#               '-R', 'retino_run1', '-t', 'retino_run1,analysis001',
#               '-R', 'retino_run2', '-t', 'retino_run2,analysis001']


    options, trace_info = extract_options(options)
    trace_info = list(trace_info)

    rootdir = options.rootdir
    animalid = options.animalid
    session = options.session
    acquisition = options.acquisition
    trace_type = options.trace_type
    colormap_position = options.colormap_position

    boundingbox_runs = options.boundingbox_runs
    leftedge = options.leftedge
    rightedge = options.rightedge
    topedge = options.topedge
    bottomedge = options.bottomedge

    acquisition_dir = os.path.join(rootdir, animalid, session, acquisition)

    # Get dataframe paths for runs to be compared:
    dfpaths = get_dataframe_paths(acquisition_dir, trace_info, trace_type=trace_type)

    # Create otuput dir:
    output_dir = os.path.join(rootdir, animalid, 'session_summaries', "%s_%s" % (animalid, session))
    output_figdir = os.path.join(output_dir, 'figures')
    if not os.path.exists(output_figdir):
        os.makedirs(output_figdir)

    # Use unique timestamp to disambiguate...
    tstamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    combo_info =  {}
    for idx, run in enumerate(dfpaths.keys()):
        runinfo = {}
        if 'pupil' in dfpaths[run]:
            metricstr =  '_'.join(os.path.split(os.path.split(dfpaths[run])[0])[-1].split('_')[0:-1])
        else:
            metricstr = 'unfiltered'
        runinfo['run'] = trace_info[idx].run
        runinfo['traceid'] = trace_info[idx].traceid
        runinfo['metric'] = metricstr
        combo_info['run%i' % int(idx+1)] = runinfo

    infodict_path = os.path.join(output_dir, 'combo_info.json')
    if os.path.exists(infodict_path):
        with open(infodict_path, 'r') as f:
            infodict = json.load(f)
            infodict[tstamp] = combo_info
    else:
        infodict = {}
        infodict[tstamp] = combo_info

    with open(infodict_path, 'w') as f:
        json.dump(infodict, f, indent=4, sort_keys=True)

    combo_basename = '_'.join(['R%i-%s' % (idx+1, runname) for idx, runname in enumerate([trace_info[i].run for i in range(len(trace_info))])])
    combo_basename = '%s_%s' % (tstamp, combo_basename)

    # Create DF for easy plotting:
    print "Getting DF..."
    # Create DF for easy plotting:
    animal_dir = os.path.join(rootdir, animalid)

    #% If retinotopy vs. gratings, look at position-selectivity:
#    width = 80
#    height = 44
#    resolution = [1920, 1080]

#    leftedge = -28; rightedge = 0
#    bottomedge = -15; topedge = 13
#    boundingbox_runs = ['retino_run2']


#    retino_info = get_retino_info(azimuth='right', elevation='top',
#                                  leftedge=leftedge, rightedge=rightedge, bottomedge=bottomedge, topedge=topedge)
#    #session_info = store_session_info(rootdir, animalid, session, acquisition)

    stat_type = 'mean'
    set_response_alpha = False

    output_figdir = os.path.join(output_dir, 'figures')
    if not os.path.exists(output_figdir):
        os.makedirs(output_figdir)

    #%%
    if colormap_position is True:

        zdf_path = os.path.join(animal_dir, output_dir, '%s_stats_tileposition.pkl' % combo_basename)

        dataframes = get_metricdf_stimpositions(dfpaths)
        zdf = assign_roi_metric(dataframes, run1_stat_type=stat_type, run2_stat_type=stat_type)

        run_list = dataframes.keys()
        retino_info = {}
        for run in run_list:
            if run in boundingbox_runs:
                retino_info[run] = ret.get_retino_info(azimuth='right', elevation='top',
                                              leftedge=leftedge, rightedge=rightedge,
                                              bottomedge=bottomedge, topedge=topedge)
            else:
                retino_info[run] = ret.get_retino_info(azimuth='right', elevation='top')

        # Get conversions for retinotopy & grid protocols:
        acquisition_str = '%s_%s_%s' % (animalid, session, acquisition)
        visinfo = visualize_position_data(dataframes, zdf, retino_info,
                                          set_response_alpha=set_response_alpha,
                                          stat_type=stat_type,
                                          color_position=False,
                                          acquisition_str=acquisition_str,
                                          output_dir=output_figdir,
                                          save_and_close=True)

        # Save monitor/retino info:
        with open(os.path.join(output_dir, 'retinotopy.json'), 'w') as f:
            json.dump(retino_info, f, indent=4, sort_keys=True)

        # Scatter plot:  phase-encoded response vs. grid-position response
        sns.set()
        fig, ax = pl.subplots(figsize=(8,8))
        run1=run_list[0]; run2=run_list[1]

        metric1 = visinfo[run1]['metric']
        metric2 = visinfo[run2]['metric']
        points = pl.scatter(zdf[metric1], zdf[metric2], cmap='hsv',
                             c=visinfo[run1]['rgbas'], s=200, vmin=-np.pi, vmax=np.pi, edgecolors='w')
        points = pl.scatter(zdf[metric1], zdf[metric2],
                             c=visinfo[run2]['rgbas'], s=60, vmin=-np.pi, vmax=np.pi, edgecolors='w', linewidth=1)

        # Create custom legend for spot sizes:
        if 'retino' in run1:
            legend1 = 'phase-code (run1)'
            run1_metric = 'mag ratio'
        else:
            legend1 = 'grid pos (run1)'
            run1_metric = 'zscore'

        if 'retino' in run2:
            legend2 = 'phase-code (run2)'
            run2_metric = 'mag ratio'
        else:
            legend2 = 'grid-position (run2)'
            run2_metric = 'zscore'

        legend_elements = [Line2D([0], [0], marker='o', color='w', label=legend1,
                                  markerfacecolor='k', markersize=15),
                           Line2D([0], [0], marker='o', color='w', label=legend2,
                                  markerfacecolor='k', markersize=8) ]
        ax.legend(handles=legend_elements, loc='lower right')

        sns.despine()
        pl.xlabel('%s: %s' % (run1, run1_metric))
        pl.ylabel('%s: %s' % (run2, run2_metric))
        run_title = '%s - R1: %s | R2 %s' % (acquisition, run1, run2)
        pl.subplots_adjust(top=0.9)
        pl.suptitle(run_title)

        figname = '%s_jointdistN.png' % combo_basename
        pl.savefig(os.path.join(output_figdir, figname))
        pl.close()

        # Create custom legend for color & alpha:
        ##### generate data grid
        N = 100
        x = np.linspace(-2,2,N)
        y = np.linspace(-2,2,N*retino_info[run]['aspect'])
        z = np.zeros((len(y),len(x))) # make cartesian grid
        for ii in range(len(y)):
            z[ii] = np.arctan2(y[ii],x) # simple angular function

        fig = pl.figure();
        ax = fig.add_subplot(111, aspect=retino_info[run]['aspect'])
        pl.imshow(z, cmap='hsv'); pl.gca().invert_yaxis()
        pl.axis('off')
        figname = '%s_jointdistN_legend.png' % combo_basename
        pl.savefig(os.path.join(output_figdir, figname))
        pl.close()

    #%% Look at repsonse magnitudes by their relevant measures:

    # TODO:  Save all combo stats for session in 1 file, store w/ keys matching
    # combo_info.json file.

    compare_metrics = True

    if compare_metrics is True:
        zdf_path = os.path.join(animal_dir, output_dir, '%s_stats.pkl' % combo_basename)
        if not os.path.exists(zdf_path):
            zdf = create_zscore_df(dfpaths)
            with open(zdf_path, 'wb') as f:
                pkl.dump(zdf, f, protocol=pkl.HIGHEST_PROTOCOL)
        else:
            with open(zdf_path, 'rb') as f:
                zdf = pkl.load(f)

        #% Look at 2 runs at a time:
        metric = 'max_zscore_stim'
        plot_joint_distn_metric(zdf, metric, combo_basename=combo_basename, output_dir=output_figdir, save_and_close=True)

        #% Get list of good rois:
        roi_list = sorted(list(set(zdf['roi'])), key=natural_keys)
        runs = list(set(zdf['run']))
        run1 = runs[0]
        run2 = runs[1]

        if 'retino' in run1:
            run1_metric_thr = 0.05 # zscore
            run1_metric = 'magratio'
        else:
            run1_metric_thr = 2.0 # mag-ratio
            run1_metric = 'zscore'
        if 'retino' in run2:
            run2_metric_thr = 0.05
            run2_metric = 'magratio'
        else:
            run2_metric_thr = 2.0
            run2_metric = 'zscore'

        run1_rois = [roi for roi in roi_list if max(zdf[((zdf['roi']==roi) & (zdf['run']==run1))][metric]) >= run1_metric_thr]
        run2_rois = [roi for roi in roi_list if max(zdf[((zdf['roi']==roi) & (zdf['run']==run2))][metric]) >= run2_metric_thr]

        print "Run: %s -- Found %i with %s >= %.2f." % (run1, len(run1_rois), run1_metric, run1_metric_thr)
        print "Run: %s -- Found %i with %s >= %.2f." % (run2, len(run2_rois), run2_metric, run2_metric_thr)

        passrois = {}
        passrois[run1] = run1_rois
        passrois[run2] = run2_rois

        pass_roi_fname = '%s_rois.json' % combo_basename
        pass_roi_fpath = os.path.join(output_dir, pass_roi_fname)
        with open(pass_roi_fpath, 'w') as f:
            json.dump(passrois, f, indent=4, sort_keys=True)


#%%
#        #% Bin to compare distN:
#        with sns.axes_style("white"):
#    #        g2 = sns.jointplot(xfeat, yfeat, data=zdf, kind="hex", color="k",
#    #                          marginal_kws=dict(bins=10, rug=True),
#    #                          joint_kws={'gridsize' : 10},
#    #                          linewidth=1)
#            g2 = sns.jointplot(x=zdf[zdf['run']==run1][metric], y=zdf[zdf['run']==run2][metric],
#                           kind="hex", color="k",
#                          marginal_kws=dict(bins=10, rug=True),
#                          joint_kws={'gridsize' : 10},
#                          linewidth=1)
#            g2.ax_joint.set_xlabel('%s: %s' % (run1, metric))
#            g2.ax_joint.set_ylabel('%s: %s' % (run2, metric))
#        pl.subplots_adjust(top=0.9)
#        g2.fig.suptitle(run_title)


        #%% Create dataframe from runs using metric
#        metric = 'max_zscore_stim'
#
#        runs = sorted(list(set(zdf['run'])), key=natural_keys)
#        run_zdf_list_all = []
#        for run in runs:
#            run_idxs = zdf[zdf['run']==run].index.tolist()
#            run_zdf_list_all.append(pd.DataFrame({
#                                         '%s_zscore_stim' % run: zdf.loc[run_idxs, 'max_zscore_stim'],
#                                         '%s_zscore_trial' % run: zdf.loc[run_idxs, 'max_zscore_trial']
#                                         }).set_index(zdf.loc[run_idxs, 'roi'][:])
#                                        )
#        runzdf_all = pd.concat(run_zdf_list_all, axis=1, ignore_index=False)
#        sns.pairplot(runzdf_all)
#
#
#        #%%
#        run_zdf_list = []
#        for run in runs:
#            run_idxs = zdf[zdf['run']==run].index.tolist()
#            if 'retino' in run:
#                label = 'mag_ratio'
#            else:
#                label = 'zscore_stim'
#            run_zdf_list.append(pd.DataFrame({
#                                         '%s_%s' % (run, label): zdf.loc[run_idxs, 'max_zscore_stim'],
#                                         }).set_index(zdf.loc[run_idxs, 'roi'][:])
#                                        )
#        runzdf = pd.concat(run_zdf_list, axis=1, ignore_index=False)
#
#        sns.set()
#        sns.pairplot(runzdf)
#        pl.suptitle("%s %s" % (animalid, session))
#        pl.subplots_adjust(top=0.9)
#
##        figname = '%s_pairwise_best_stimulus.png' % combo_basename
##        pl.savefig(os.path.join(output_figdir, figname))

#%%

if __name__ == '__main__':
    main(sys.argv[1:])



#%%

#t1_path = os.path.join(acquisition_dir, run1, 'traces', 'traceids_%s.json' % run1)
#with open(t1_path, 'r') as f:
#    t1dict = json.load(f)
#run1_tracename = '%s_%s' % (run1_traceid, t1dict[run1_traceid]['trace_hash'])
#run1_tracedir = os.path.join(acquisition_dir, run1, 'traces', run1_tracename)
#
#
#t2_path = os.path.join(acquisition_dir, run2, 'traces', 'traceids_%s.json' % run2)
#with open(t2_path, 'r') as f:
#    t2dict = json.load(f)
#run2_tracename = '%s_%s' % (run2_traceid, t2dict[run2_traceid]['trace_hash'])
#run2_tracedir = os.path.join(acquisition_dir, run2, 'traces', run2_tracename)
#
#
## First, load datasets:
##R1_datafile = [os.path.join(run1_tracedir, f) for f in os.listdir(run1_tracedir) if 'ROIDATA_' in f and f.endswith('hdf5')][0]
#size_thr = 25
#dist_thr = 13
#blinks = 1
#pupil_str = 'pupil_size%i-dist%i-blinks%i' % (size_thr, dist_thr, blinks)
#pupil_dir = [os.path.join(run1_tracedir, 'metrics', p) for p in os.listdir(os.path.join(run1_tracedir, 'metrics')) if pupil_str in p][0]
#
#R1_datafile = [os.path.join(pupil_dir, f) for f in os.listdir(pupil_dir) if 'roi_stats_' in f][0]
#R1 = pd.HDFStore(R1_datafile, 'r')['/df']
#
#
#size_thr = 25
#dist_thr = 8
#blinks = 1
#pupil_str = 'pupil_size%i-dist%i-blinks%i' % (size_thr, dist_thr, blinks)
#pupil_dir = [os.path.join(run2_tracedir, 'metrics', p) for p in os.listdir(os.path.join(run2_tracedir, 'metrics')) if pupil_str in p][0]
#R2_datafile = [os.path.join(pupil_dir, f) for f in os.listdir(pupil_dir) if 'roi_stats_' in f][0]
#R2 = pd.HDFStore(R2_datafile, 'r')['/df']
#
##R2_datafile = [os.path.join(run2_tracedir, f) for f in os.listdir(run2_tracedir) if 'ROIDATA_' in f and f.endswith('hdf5')][0]
#R2 = pd.HDFStore(R2_datafile, 'r')

#%%
#roi_list = sorted(list(set(R1['roi'])), key=natural_keys)
#nrois = len(roi_list)
#trial_list1 = list(set(R1['trial']))
#trial_list2 = list(set(R2['trial']))
#
#config_list1 = list(set(R1['config']))
#config_list2 = list(set(R2['config']))
#
#
#R1_max_zscores = [max([np.float(R1[((R1['roi']==roi) & (R1['trial']==trial))]['zscore']) for trial in trial_list1]) for roi in sorted(roi_list, key=natural_keys)]
#R2_max_zscores = [max([np.float(R2[((R2['roi']==roi) & (R2['trial']==trial))]['zscore']) for trial in trial_list2]) for roi in sorted(roi_list, key=natural_keys)]
#
#R1_max_zscore_stim = [max([np.nanmean(R1[((R1['roi']==roi) & (R1['config']==config))]['zscore']) for config in config_list1]) for roi in sorted(roi_list, key=natural_keys)]
#R2_max_zscore_stim = [max([np.nanmean(R2[((R2['roi']==roi) & (R2['config']==config))]['zscore']) for config in config_list1]) for roi in sorted(roi_list, key=natural_keys)]
#
#
#
##x = np.array(R1_max_zscores) #[s for s in R2[curr_roi][R2['roi']==curr_roi]['zscore'] if not np.isnan(s)] #.sort_values(['trial'])
##y = np.array(R2_max_zscores) #@[s for s in R2[R2['roi']==curr_roi]['zscore'] if not np.isnan(s)] #.sort_values(['trial'])
#
#
#zdf = pd.DataFrame({'roi': roi_list,
#              'R1_max_zscore': np.array(R1_max_zscores),
#              'R2_max_zscore': np.array(R2_max_zscores),
#              'R1_max_zscore_stim': np.array(R1_max_zscore_stim),
#              'R2_max_zscore_stim': np.array(R2_max_zscore_stim)
#              })
#
#run_title = '%s - R1: %s | R2 %s' % (acquisition, run1, run2)
#
## Plot as scatter:
#xfeat = "R1_max_zscore_stim"
#yfeat = "R2_max_zscore_stim"
#
#with sns.axes_style("white"):
#    g1 = sns.jointplot(xfeat, yfeat, data=zdf, kind="scatter",
#                      marginal_kws=dict(bins=nrois, rug=True),
#                      linewidth=1)
#pl.subplots_adjust(top=0.9)
#g1.fig.suptitle(run_title)
#
## Bin to compare distN:
#with sns.axes_style("white"):
#    g2 = sns.jointplot(xfeat, yfeat, data=zdf, kind="hex", color="k",
#                      marginal_kws=dict(bins=10, rug=True),
#                      joint_kws={'gridsize' : 10},
#                      linewidth=1)
#pl.subplots_adjust(top=0.9)
#g2.fig.suptitle(run_title)
#
#run1_rois = [roi for roi in roi_list if max(R1[R1['roi']==roi]['zscore']) > 4]
#run2_rois = [roi for roi in roi_list if max(R2[R2['roi']==roi]['zscore']) > 4]
#
#run1_DF = R1[R1['roi'].isin(run1_rois)]
#run2_DF = R2[R2['roi'].isin(run2_rois)]


#%%

