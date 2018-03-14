#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 18:47:08 2018

@author: juliana
"""

import os
import sys
import numpy as np
import json
import optparse
import seaborn as sns
import pylab as pl
import pandas as pd
from optparse import OptionParser

from pipeline.python.utils import natural_keys
from pipeline.python.traces.utils import get_metric_set

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
        self.pupil_size_thr = None
        self.pupil_dist_thr = None
        self.pupil_max_nblinks = 1

    def set_params(self, paramslist):
        #params = getattr(parservalues, 'trace_info')
        self.traceid = paramslist[0]
        self.pupil_size_thr = paramslist[1]
        self.pupil_dist_thr = paramslist[2]


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

    parser.add_option('-R', '--run', dest='run', type='string',
                          action='callback', callback=fop.set_info, help="Supply multiple runs for comparison, all runs used otherwise")

    parser.add_option('-t', '--traces', dest='trace_info', default=[], nargs=1,
                          action='append', help="Corresponding trace ID to specified runs.")


    parser.add_option('--slurm', action='store_true', dest='slurm', default=False, help="set if running as SLURM job on Odyssey")
    parser.add_option('--default', action='store_true', dest='auto', default=False, help="set if want to use all defaults")

    #    parser.add_option('-t', '--trace-id', action='store', dest='trace_id', default='', help="Trace ID for current trace set (created with set_trace_params.py, e.g., traces001, traces020, etc.)")

    # Pupil filtering info:
    parser.add_option('--no-pupil', action="store_false",
                      dest="filter_pupil", default=True, help="Set flag NOT to filter PSTH traces by pupil threshold params")
    parser.add_option('-r', '--rad', action="store",
                      dest="pupil_size_thr", default=25, help="Cut-off for pupil radius, if --pupil set [default: 30]")
    parser.add_option('-d', '--dist', action="store",
                      dest="pupil_dist_thr", default=15, help="Cut-off for pupil distance from start, if --pupil set [default: 5]")
    parser.add_option('-b', '--blinks', action="store",
                      dest="pupil_max_nblinks", default=1, help="Cut-off for N blinks allowed in trial, if --pupil set [default: 1 (i.e., 0 blinks allowed)]")

    (options, args) = parser.parse_args(options)

    for f in fop.run_list:
        run_params = [t for t in options.trace_info if f.run in t][0]
        print run_params
        params_list = [p for p in run_params.split(',') if not p==f.run]
        f.set_params(params_list)
    #print [(f.run, f.traceid, f.pupil_max_nblinks) for f in fop.run_list]

    return options, fop.run_list

#%%

def get_dataframe_paths(acquisition_dir, trace_info):
    dfpaths = dict()
    for idx, info in enumerate(trace_info):
        dfilepath = None
        rkey = 'run%i' % int(idx+1)

        #runs[rkey]['run'] = info.run
        tdict_path = os.path.join(acquisition_dir, info.run, 'traces', 'traceids_%s.json' % info.run)
        with open(tdict_path, 'r') as f:
            tdict = json.load(f)
        tracename = '%s_%s' % (info.traceid, tdict[info.traceid]['trace_hash'])
        traceid_dir = os.path.join(acquisition_dir, info.run, 'traces', tracename)

        pupil_str = 'pupil_size%i-dist%i-blinks%i' % (float(info.pupil_size_thr), float(info.pupil_dist_thr), int(info.pupil_max_nblinks))
        pupil_dir = [os.path.join(traceid_dir, 'metrics', p) for p in os.listdir(os.path.join(traceid_dir, 'metrics')) if pupil_str in p][0]

        dfilepath = [os.path.join(pupil_dir, f) for f in os.listdir(pupil_dir) if 'roi_stats_' in f][0]
        dfpaths[rkey] = dfilepath

    return dfpaths

#%%
def create_zscore_df(dfpaths):
    all_dfs = []
    for df in dfpaths.values():
        rundf = pd.HDFStore(df, 'r')['/df']
        run_name = os.path.split(df.split('/traces/')[0])[-1]
        print "Compiling zscores for each ROI in run: %s" % run_name
        roi_list = sorted(list(set(rundf['roi'])), key=natural_keys)
        nrois = len(roi_list)
        trial_list = sorted(list(set(rundf['trial'])), key=natural_keys)
        confg_list = sorted(list(set(rundf['config'])), key=natural_keys)

        max_zscores_by_trial = [max([np.float(rundf[((rundf['roi']==roi) & (rundf['trial']==trial))]['zscore'])
                                    for trial in trial_list]) for roi in sorted(roi_list, key=natural_keys)]
        max_zscores_by_stim = [max([np.nanmean(rundf[((rundf['roi']==roi) & (rundf['config']==config))]['zscore'])
                                    for config in confg_list]) for roi in sorted(roi_list, key=natural_keys)]

        curr_df = pd.DataFrame({'roi': roi_list,
                                'max_zscore_trial': np.array(max_zscores_by_trial),
                                'max_zscore_stim': np.array(max_zscores_by_stim),
                                'run': np.tile(run_name, (nrois,))
                                })

        # Concatenate all info for this current trial:
        all_dfs.append(curr_df)
        #rundf.close()

    # Finally, concatenate all trials across all configs for current ROI dataframe:
    DF = pd.concat(all_dfs, axis=0)

    return DF


#%%
def main(options):

    opts = ['-D', '/mnt/odyssey', '-i', 'CE074', '-S', '20180215', '-A', 'FOV1_zoom1x_V1', '-R', 'gratings_phasemod', '-t', 'gratings_phasemod,traces004,30,8', '-R', 'blobs', '-t', 'blobs,traces003,25,15']

    options, trace_info = extract_options(opts)
    trace_info = list(trace_info)

    rootdir = options.rootdir
    animalid = options.animalid
    session = options.session
    acquisition = options.acquisition

    acquisition_dir = os.path.join(rootdir, animalid, session, acquisition)

    # Get dataframe paths for runs to be compared:
    dfpaths = get_dataframe_paths(acquisition_dir, trace_info)

    # Create DF for easy plotting:
    zdf = create_zscore_df(dfpaths)

    roi_list = sorted(list(set(zdf['roi'])), key=natural_keys)
    nrois = len(roi_list)

    runs = list(set(zdf['run']))
    run1 = runs[0]
    run2 = runs[1]
    run_title = '%s - R1: %s | R2 %s' % (acquisition, run1, run2)

    # Plot as scatter:
#    xfeat = "R1_max_zscore_stim"
#    yfeat = "R2_max_zscore_stim"

    metric = 'max_zscore_stim'
    min_zscore = 1.6

    with sns.axes_style("white"):
#        g1 = sns.jointplot(xfeat, yfeat, data=zdf, kind="scatter",
#                          marginal_kws=dict(bins=nrois, rug=True),
#                          linewidth=1)
        g1 = sns.jointplot(x=zdf[zdf['run']==run1][metric], y=zdf[zdf['run']==run2][metric], kind="scatter",
                      marginal_kws=dict(bins=nrois, rug=True),
                      linewidth=1)
        g1.ax_joint.set_xlabel('%s: %s' % (run1, metric))
        g1.ax_joint.set_ylabel('%s: %s' % (run2, metric))

    pl.subplots_adjust(top=0.9)
    g1.fig.suptitle(run_title)

    # Bin to compare distN:
    with sns.axes_style("white"):
#        g2 = sns.jointplot(xfeat, yfeat, data=zdf, kind="hex", color="k",
#                          marginal_kws=dict(bins=10, rug=True),
#                          joint_kws={'gridsize' : 10},
#                          linewidth=1)
        g2 = sns.jointplot(x=zdf[zdf['run']==run1][metric], y=zdf[zdf['run']==run2][metric],
                       kind="hex", color="k",
                      marginal_kws=dict(bins=10, rug=True),
                      joint_kws={'gridsize' : 10},
                      linewidth=1)
        g2.ax_joint.set_xlabel('%s: %s' % (run1, metric))
        g2.ax_joint.set_ylabel('%s: %s' % (run2, metric))
    pl.subplots_adjust(top=0.9)
    g2.fig.suptitle(run_title)

    run1_rois = [roi for roi in roi_list if max(zdf[((zdf['roi']==roi) & (zdf['run']==run1))][metric]) >= min_zscore]
    run2_rois = [roi for roi in roi_list if max(zdf[((zdf['roi']==roi) & (zdf['run']==run2))][metric]) >= min_zscore]

    print "Run: %s -- Found %i with zscore >= 2." % (run1, len(run1_rois))
    print "Run: %s -- Found %i with zscore >= 2." % (run2, len(run2_rois))


    #run1_DF = R1[R1['roi'].isin(run1_rois)]
    #run2_DF = R2[R2['roi'].isin(run2_rois)]





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


def plot_transform_tuning(roi, DF, trans_type, object_list, object_type='object', metric_type='zscore', include_trials=True):
    '''
    trans_type = feature that varies for a given object ID (i.e., how we want to color-code)
    object_sorter = how we want to define object ID
        -- for objects, this is the object ID (e.g., Blobs_N1, Blobs_N2, morph5, or just morph, if only 1 morph)
        -- for gratings, this is the spatial frequency, since we want to look at variation in ORI (trans_type) within a given S.F.
    hue = should be the same as object_sorter (except if only a single morph, since there is only 1 morphID)
    '''

    nrows = len(object_transformations.keys())
    fig, axes = pl.subplots(nrows=nrows, ncols=1, sharex=False, squeeze=True, figsize=(6,10))
    transform_str = '_'.join(object_transformations.keys())
    for trans, ax in zip(object_transformations.keys(), axes):
        #print trans
        if object_type == 'grating':
            if trans == 'ori':
                object_color = 'sf'
                object_sorter = 'sf'
            else:
                object_color = 'ori'
                object_sorter = 'ori'
        else:
            if trans == 'morphlevel':
                object_color = None
                object_sorter = trans
            else:
                object_color = 'object'
                object_sorter = 'object'

        # Make sure only plotting lines for objects that are tested with current transform:
        if object_type == 'object':
            ROI = DF[DF['object'].isin(object_transformations[trans])]
        else:
            ROI = DF
        # Draw MEAN metric:
        sns.pointplot(x=trans, y=metric_type, hue=object_color, data=ROI.sort_values([object_sorter]),
                      ci=None, join=True, markers='_', scale=2, ax=ax, legend_out=True)
        axlines = ax.get_lines()
        pl.setp(axlines, linewidth=1)

        # Add dots for individual trial z-score values:
        if include_trials is True:
            if trans == 'morphlevel':
                # Adjust colors for trans=morphlevel, since we aren't really testing any morphs except "default view" yet:
                sns.stripplot(x=trans, y=metric_type, hue=object_color, data=ROI.sort_values([object_sorter]),
                      edgecolor="w", split=True, size=2, ax=ax, color=sns.color_palette()[0])
            else:
                sns.stripplot(x=trans, y=metric_type, hue=object_color, data=ROI.sort_values([object_sorter]),
                      edgecolor="w", split=True, size=2, ax=ax)

        transforms = sorted(list(set(ROI[trans])))
        for oi, obj in enumerate(sorted(list(set(ROI[object_sorter])))): #enumerate(sorted(curr_objects)): #, key=natural_keys)):
            curr_zmeans = [np.nanmean(ROI[((ROI[object_sorter] == obj) & (ROI[trans] == t))][metric_type]) for t in sorted(transforms)]
            curr_zmeans_yerr = [stats.sem(ROI[((ROI[object_sorter] == obj) & (ROI[trans] == t))][metric_type], nan_policy='omit') for t in sorted(transforms)]
            # Adjust colors for trans=morphlevel, since we aren't really testing any morphs except "default view" yet:
            if trans == 'morphlevel':
                ax.errorbar(np.arange(0, len(curr_zmeans)), curr_zmeans, yerr=curr_zmeans_yerr,
                                capsize=5, elinewidth=1, color=sns.color_palette()[0])
            else:
                ax.errorbar(np.arange(0, len(curr_zmeans)), curr_zmeans, yerr=curr_zmeans_yerr,
                                capsize=5, elinewidth=1, color=sns.color_palette()[oi])

    # Format, title, and save:
    pl.subplots_adjust(top=0.9)
    fig.suptitle(roi)
    sns.despine(offset=1, trim=True, bottom=True)
    if '/' in roi:
        roi = roi[1:]
