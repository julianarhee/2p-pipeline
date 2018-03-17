#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""



    d.  PSTHs, if relevant for each ROI:

    <TRACEID_DIR>/figures/psths/<TRACE_TYPE>/all/roiXXXXX_SliceXX_IDX_<TRACE_TYPE>_<TRANSFORM_STR>.png
    -- TRACE_TYPE = 'raw' or 'denoised_nmf' for now
    -- IDX = roi idx in current roi set (0-indexed)
    -- TRANSFORM_STR = short '_'-separated string describing transforms

    * if --pupil is True, also plots PSTHs that show 'included' and 'excluded' traces based on pupil_params:

        <TRACEID_DIR>/figures/psths/<TRACE_TYPE>/size<SIZE>-dist<DIST>-blinks<BLINKS>/
        -- SIZE = opt-arg, size of pupil radius, below which trials are excluded
        -- DIST = opt-arg, pupil distance from start frame, below which trials are excluded
        -- BLINKS = opt-arg, number of blinks allowed




Created on Fri Mar  9 12:35:05 2018

@author: juliana
"""

import os
import json
import sys
import h5py
import itertools
import optparse
import pprint
import traceback
import matplotlib
matplotlib.use('Agg')
import pylab as pl
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats

from pipeline.python.utils import natural_keys, replace_root
from pipeline.python.traces.utils import load_TID, get_metric_set
from pipeline.python.paradigm.align_acquisition_events import get_stimulus_configs, set_pupil_params
pp = pprint.PrettyPrinter(indent=4)


#def set_subplot_order(configs, stimtype, universal_scale=False):  # in percentage of the figure height
#    plot_info = {}
#    stiminfo = {}
#
#    if 'grating' in stimtype:
#        sfs = list(set([configs[c]['frequency'] for c in configs.keys()]))
#        oris = list(set([configs[c]['rotation'] for c in configs.keys()]))
#
#        noris = len(oris)
#        nsfs = len(sfs)
#
#        nrows = min([noris, nsfs])
#        ncols = (nsfs * noris) / nrows
#
#        tmp_list = []
#        for sf in sorted(sfs):
#            for oi in sorted(oris):
#                match = [k for k in configs.keys() if configs[k]['rotation']==oi and configs[k]['frequency']==sf][0]
#                tmp_list.append(match)
#        stimid_only = True
#        subplot_stimlist = dict()
#        subplot_stimlist['defaultgratings'] = tmp_list
#        #print subplot_stimlist
#
#        stiminfo['sfs'] = sfs
#        stiminfo['oris'] = oris
#
#    else:
#        #configparams = configs[configs.keys()[0]].keys()
#
#        # Create figure(s) based on stim configs:
#        position_vals = list(set([tuple(configs[k]['position']) for k in configs.keys()]))
#        size_vals = list(set([configs[k]['scale'][0] for k in configs.keys()]))
#        img_vals = list(set([configs[k]['filename'] for k in configs.keys()]))
#        if len(position_vals) > 1 or len(size_vals) > 1:
#            stimid_only = False
#        else:
#            stimid_only = True
#
#        if stimid_only is True:
#            nfigures = 1
#            nrows = int(np.ceil(np.sqrt(len(configs.keys()))))
#            ncols = len(configs.keys()) / nrows
#            img = img_vals[0]
#            subplot_stimlist = dict()
#            subplot_stimlist[img] = sorted(configs.keys(), key=lambda x: configs[x]['filename'])
#        else:
#            nfigures = len(img_vals)
#            nrows = len(position_vals)
#            ncols = len(size_vals)
#            subplot_stimlist = dict()
#            for img in img_vals:
#                curr_img_configs = [c for c in configs.keys() if configs[c]['filename'] == img]
#                subplot_stimlist[img] = sorted(curr_img_configs, key=lambda x: (configs[x].get('scale'), configs[x].get('position')))
#
#        stiminfo['position_vals'] = position_vals
#        stiminfo['size_vals'] = size_vals
#        stiminfo['img_vals'] = img_vals
#
#    plot_info['stimuli'] = stiminfo
#    plot_info['stimid_only'] = stimid_only
#    plot_info['subplot_stimlist'] = subplot_stimlist
#    plot_info['nrows'] = nrows
#    plot_info['ncols'] = ncols
#
#    figure_height, top_margin, bottom_margin =  set_figure_params(nrows, ncols)
#    plot_info['figure_height'] = figure_height
#    plot_info['top_margin'] = top_margin
#    plot_info['bottom_margin'] = bottom_margin
#    plot_info['universal_scale'] = universal_scale
##
#
#    return plot_info #subplot_stimlist, nrows, ncols

#%%

def set_figure_params(nrows, ncols, fontsize_pt=20, dpi=72.27, spacer=20, top_margin=0.01, bottom_margin=0.05):

    # comput the matrix height in points and inches
    matrix_height_pt = fontsize_pt * nrows * spacer
    matrix_height_in = matrix_height_pt / dpi

    # compute the required figure height  # in percentage of the figure height
    figure_height = matrix_height_in / (1 - top_margin - bottom_margin)

    return figure_height, top_margin, bottom_margin



#%%
def get_facet_stats(DF, config_list, g):
    plotstats = {'dfmats': {}, 'indices': {}}
    dfmats = dict((config, []) for config in config_list)
    plotindices = dict((config, []) for config in config_list)
    for (row_i, col_j, hue_k), data_ijk in g.facet_data():
        if len(data_ijk) == 0:
            continue
        dfmats[list(set(data_ijk['config']))[0]].append(data_ijk['df'])
        plotindices[list(set(data_ijk['config']))[0]].append((row_i, col_j))

    for config in dfmats.keys():
        plotstats['dfmats'][config] = np.array(dfmats[config])
        plotstats['indices'][config] = list(set(plotindices[config]))[0]

    return plotstats

#%%
def plot_roi_psth(roi, DF, object_transformations, figdir='/tmp', prefix='psth', trace_color='k', stimbar_color='r'):

    if 'ori' in DF.keys():
        stimtype = 'grating'
    else:
        stimtype = 'image'

    trans_types = object_transformations.keys()
    #config_list = list(set(DF['config']))
#
#    first_on = list(set(DF['first_on']))[0]
#    nsecs_on = list(set(DF['nsecs_on']))[0]
#    tsecs = sorted(list(set(DF['tsec'])))

    single_object_figure = False
    if 'size' in trans_types and ('xpos' in trans_types or 'ypos' in trans_types):
        ordered_rows = sorted(list(set(DF['position'])))
        ordered_cols = sorted(list(set(DF['size'])))
        rows = 'position'
        columns = 'size'
        single_object_figure = True
        figbase = 'pos%i_size%i' % (len(ordered_rows), len(ordered_cols))
    else:
        figbase = 'all_objects_default_pos_size'
        if stimtype == 'grating':
            ordered_rows = sorted(list(set(DF['sf'])))
            ordered_cols = sorted(list(set(DF['ori'])))
            rows = 'sf'
            columns = 'ori'
        else:
            ordered_rows = sorted(list(set(DF['morphlevel'])))
            ordered_cols = sorted(list(set(DF['yrot'])))
            rows = 'morphlevel'
            columns = 'yrot'

    if single_object_figure is True:
        for objectid in list(set(DF['object'])):
            currdf = DF[DF['object'] == objectid]
            figpath = os.path.join(figdir, '%s_%s_%s.png' % (prefix, objectid, figbase))
            draw_psth(roi, currdf, rows, columns, ordered_rows, ordered_cols, trace_color, stimbar_color, figpath)
    else:
        figpath = os.path.join(figdir, '%s_%s.png' % (prefix, figbase))
        draw_psth(roi, DF, rows, columns, ordered_rows, ordered_cols, trace_color, stimbar_color, figpath)

#%%
def draw_psth(roi, DF, rows, columns, row_order, col_order, trace_color, stimbar_color, figpath):

    config_list = list(set(DF['config']))

    first_on = list(set(DF['first_on']))[0]
    nsecs_on = list(set(DF['nsecs_on']))[0]
    tsecs = sorted(list(set(DF['tsec'])))

    g1 = sns.FacetGrid(DF, row=rows, col=columns, sharex=True, sharey=True, hue='trial', row_order=row_order, col_order=col_order)
    g1.map(pl.plot, "tsec", "df", linewidth=0.2, color=trace_color, alpha=0.5)
    plotstats = get_facet_stats(DF, config_list, g1)

    # Get mean trace:
    meandfs = {}
    for config in plotstats['dfmats'].keys():
        meandfs[config] = np.mean(plotstats['dfmats'][config], axis=0)
        currax = g1.facet_axis(plotstats['indices'][config][0], plotstats['indices'][config][1])
        currax.plot(tsecs, meandfs[config], trace_color, linewidth=1, alpha=1)
        currax.plot([tsecs[first_on], tsecs[first_on]+nsecs_on], [0, 0], stimbar_color, linewidth=2, alpha=1)

    sns.despine(offset=2, trim=True)
    #%
    pl.subplots_adjust(top=0.9)
    g1.fig.suptitle(roi)
    g1.savefig(figpath)
    pl.close()


#%%

def get_object_transforms(DF):
    '''
    Returns 2 dicts:
        transform_dict = lists all transforms tested in dataset for each transform type
        object_transformations = lists all objects tested on each transform type
    '''

    if 'ori' in DF.keys():
        stimtype = 'grating'
    else:
        stimtype = 'image'

    transform_dict = {'xpos': list(set(DF['xpos'])),
                       'ypos': list(set(DF['ypos'])),
                       'size': list(set((DF['size'])))
                       }
    if stimtype == 'image':
        transform_dict['yrot'] = list(set(DF['yrot']))
        transform_dict['morphlevel'] = list(set(DF['morphlevel']))
    else:
        transform_dict['ori'] = sorted(list(set(DF['ori'])))
        transform_dict['sf'] = sorted(list(set(DF['sf'])))
    trans_types = [t for t in transform_dict.keys() if len(transform_dict[t]) > 1]

    object_transformations = {}
    for trans in trans_types:
        if stimtype == 'image':
            curr_objects = [list(set(DF[DF[trans] == t]['object'])) for t in transform_dict[trans]]
            if len(list(itertools.chain(*curr_objects))) == len(transform_dict[trans]):
                # There should be a one-to-one correspondence between object id and the transformation (i.e., morphs)
                included_objects = list(itertools.chain(*curr_objects))
            else:
                included_objects = list(set(curr_objects[0]).intersection(*curr_objects[1:]))
        else:
            included_objects = transform_dict[trans]
            print included_objects
        object_transformations[trans] = included_objects

    return transform_dict, object_transformations


#%%
def plot_psths(roidata_filepath, trial_info, configs, roi_psth_dir='/tmp', trace_type='raw',
                   filter_pupil=True, pupil_params=None, plot_all=True, universal_scale=False,
                   ylim_min=-1.0, ylim_max=1.0):

    if plot_all is False and filter_pupil is False:
        print "No PSTH types specified. Exiting."
        return

    if filter_pupil is True and pupil_params is None:
        pupil_params = set_pupil_params()
    pupil_max_nblinks = pupil_params['max_nblinks']
    pupil_size_thr = pupil_params['size_thr']
    pupil_dist_thr = pupil_params['dist_thr']

    roi_psth_dir_all = os.path.join(roi_psth_dir, 'unfiltered')
    if not os.path.exists(roi_psth_dir_all):
        os.makedirs(roi_psth_dir_all)
    if filter_pupil is True:
        pupil_thresh_str = 'pupil_size%i-dist%i-blinks%i' % (pupil_size_thr, pupil_dist_thr, int(pupil_max_nblinks))
        roi_psth_dir_include = os.path.join(roi_psth_dir, pupil_thresh_str, 'include')
        if not os.path.exists(roi_psth_dir_include):
            os.makedirs(roi_psth_dir_include)
        roi_psth_dir_exclude = os.path.join(roi_psth_dir, pupil_thresh_str, 'exclude')
        if not os.path.exists(roi_psth_dir_exclude):
            os.makedirs(roi_psth_dir_exclude)

    DATA = pd.HDFStore(roidata_filepath, 'r')

    transform_dict, object_transformations = get_object_transforms(DATA[DATA.keys()[0]])

    roi_list = sorted(DATA.keys(), key=natural_keys)
    if '/' in DATA.keys()[0]:
        roi_list = sorted([r[1:] for r in roi_list], key=natural_keys)


    roi=None; configname=None; trial=None
    try:
        for roi in roi_list:
            print roi

            DF = DATA[roi] #[DATA[roi]['config'].isin(curr_subplots)]
            DF['position'] = list(zip(DF['xpos'], DF['ypos']))

            curr_slice = list(set(DF['slice']))[0] #roi_trials[configname][roi].attrs['slice']
            roi_in_slice = list(set(DF['roi_in_slice']))[0] #roi_trials[configname][roi].attrs['idx_in_slice']

            # PLOT ALL:
            if plot_all is True:
                prefix = '%s_%s_%s_%s_ALL' % (roi, curr_slice, roi_in_slice, trace_type) #, figname)
                plot_roi_psth(roi, DF, object_transformations, figdir=roi_psth_dir_all, prefix=prefix, trace_color='k', stimbar_color='r')

                # Plot df values:
            if filter_pupil is True:
                filtered_DF = DF.query('pupil_size_stimulus > @pupil_size_thr \
                                       & pupil_size_baseline > @pupil_size_thr \
                                       & pupil_dist_stimulus < @pupil_dist_thr \
                                       & pupil_dist_baseline < @pupil_dist_thr \
                                       & pupil_nblinks_stim < @pupil_max_nblinks \
                                       & pupil_nblinks_baseline < @pupil_max_nblinks')
                pass_trials = list(set(filtered_DF['trial']))

                # INCLUDED trials:
                prefix = '%s_%s_%s_%s_PUPIL_%s_pass.png' % (roi, curr_slice, roi_in_slice, trace_type, pupil_thresh_str)
                plot_roi_psth(roi, filtered_DF, object_transformations,
                                  figdir=roi_psth_dir_include, prefix=prefix, trace_color='b', stimbar_color='k')

                # EXCLUDED trials:
                excluded_DF = DF[~DF['trial'].isin(pass_trials)]
                prefix = '%s_%s_%s_%s_PUPIL_%s_fail.png' % (roi, curr_slice, roi_in_slice, trace_type, pupil_thresh_str)
                plot_roi_psth(roi, excluded_DF, object_transformations,
                                  figdir=roi_psth_dir_exclude, prefix=prefix, trace_color='r', stimbar_color='k')


    except Exception as e:

        print "--- Error plotting PSTH ---------------------------------"
        print roi, configname, trial
        traceback.print_exc()
    #    print "---------------------------------------------------------"
#        finally:
#            roi_trials.close()
    #parsed_frames.close()
    finally:
        DATA.close()

    print "PSTHs saved to: %s" % roi_psth_dir





#%%
def plot_tuning_curves(roistats_filepath, configs, curr_tuning_figdir, metric_type='zscore', include_trials=True):

    STATS = pd.HDFStore(roistats_filepath, 'r')['/df']


    if 'frequency' in configs[configs.keys()[0]].keys():
        stimtype = 'grating'
    else:
        stimtype = 'image'

    roi_list = sorted(list(set(STATS['roi'])), key=natural_keys)

    transform_dict, object_transformations = get_object_transforms(STATS)
    if stimtype == 'image':
        trans_types = [t for t in transform_dict.keys() if len(transform_dict[t]) > 1]
    else:
        trans_types = ['ori', 'sf']

    for roi in sorted(roi_list, key=natural_keys):
        print roi
        DF = STATS[STATS['roi'] == roi]

        if stimtype == 'image':
            #objectid_list = sorted(list(set(STATS['object'])))

            # ----- First, plot TUNING curves for each OBJECT ID --------
            plot_transform_tuning(roi, DF, object_transformations, object_type='object', metric_type=metric_type, output_dir=curr_tuning_figdir, include_trials=include_trials)

            #plot_transform_tuning(roi, curr_df, trans_type="yrot", object_type='object', metric_type=metric_type, output_dir=output_dir, include_trials=include_trials)

            # ----- Now, plot TUNING curves for MORPHS (color by morph level) --------
            #curr_df = STATS[((STATS['roi'] == roi) & (STATS['yrot'] == 0))]
            #plot_transform_tuning(roi, curr_df, trans_type="morph", object_type='morph', metric_type=metric_type, output_dir=output_dir, include_trials=include_trials)

        elif 'grating' in stimtype:

            object_transformations = {}
            for trans in trans_types:
                object_transformations[trans] = []
            #print roi
            #objectid_list = sorted(list(set(STATS['ori'])))
            #curr_df = STATS[((STATS['roi'] == roi) & (STATS['ori'].isin(objectid_list)))]
            plot_transform_tuning(roi, DF, object_transformations, object_type='grating', metric_type=metric_type, output_dir=curr_tuning_figdir, include_trials=include_trials)

    #STATS.close()

#%%
def plot_transform_tuning(roi, DF, object_transformations, object_type='object', metric_type='zscore', output_dir='/tmp', include_trials=True):
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
    if include_trials is True:
        figname = '%s_tuning_%s_trials_%s.png' % (transform_str, metric_type, roi)
    else:
        figname = '%s_tuning_%s_%s.png' % (transform_str, metric_type, roi)
    pl.savefig(os.path.join(output_dir, figname))
    pl.close()

#

#%%

def extract_options(options):

    choices_tracetype = ('raw', 'raw_fissa', 'denoised_nmf', 'np_corrected_fissa', 'neuropil_fissa')
    default_tracetype = 'raw'

    parser = optparse.OptionParser()

    parser.add_option('-D', '--root', action='store', dest='rootdir', default='/nas/volume1/2photon/data', help='data root dir (root project dir containing all animalids) [default: /nas/volume1/2photon/data, /n/coxfs01/2pdata if --slurm]')
    parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')

    # Set specific session/run for current animal:
    parser.add_option('-S', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID')
    parser.add_option('-A', '--acq', action='store', dest='acquisition', default='FOV1', help="acquisition folder (ex: 'FOV1_zoom3x') [default: FOV1]")
    parser.add_option('-R', '--run', action='store', dest='run', default='', help="name of run dir containing tiffs to be processed (ex: gratings_phasemod_run1)")
    parser.add_option('--slurm', action='store_true', dest='slurm', default=False, help="set if running as SLURM job on Odyssey")
    parser.add_option('--default', action='store_true', dest='auto', default=False, help="set if want to use all defaults")

    parser.add_option('-t', '--trace-id', action='store', dest='trace_id', default='', help="Trace ID for current trace set (created with set_trace_params.py, e.g., traces001, traces020, etc.)")
    parser.add_option('-T', '--trace-type', type='choice', choices=choices_tracetype, action='store', dest='trace_type', default=default_tracetype, help="Type of timecourse to plot PSTHs. Valid choices: %s [default: %s]" % (choices_tracetype, default_tracetype))


    parser.add_option('--psth', action="store_true",
                      dest="psth", default=False, help="Set flag to plot (any) PSTH figures.")
    parser.add_option('--tuning', action="store_true",
                      dest="tuning", default=False, help="Set flag to plot(any) tuning curves.")


    parser.add_option('--scale', action="store_true",
                      dest="universal_scale", default=False, help="Set flag to plot all PSTH plots with same y-axis scale")
    parser.add_option('-y', '--ylim_min', action="store",
                      dest="ylim_min", default=-1.0, help="min lim for Y axis, df/f plots [default: -1.0]")
    parser.add_option('-Y', '--ylim_max', action="store",
                      dest="ylim_max", default=1.0, help="max lim for Y axis, df/f plots [default: 1.0]")

    parser.add_option('--filter', action="store_false",
                      dest="plot_all_psths", default=True, help="Set flag to only plot PSTHs for filtered traces (don't plot PSTHS for ALL, unfiltered)")
#    parser.add_option('--omit-err', action="store_false",
#                      dest="use_errorbar", default=True, help="Set flag to plot PSTHs without error bars (default: std)")
    parser.add_option('--omit-trials', action="store_false",
                      dest="include_trials", default=True, help="Set flag to plot PSTHS without individual trial values")
    parser.add_option('--metric', action="store",
                      dest="roi_metric", default="zscore", help="ROI metric to use for tuning curves [default: 'zscore']")

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

    return options

#%%

#
#options = ['-D', '/mnt/odyssey', '-i', 'CE074', '-S', '20180220',
#        '-A', 'FOV1_zoom1x', '-R', 'blobs', '-t', 'traces004', '-r', '15', '-d8',
#        '--omit-trials']


def plot_roi_figures(options):
    options = extract_options(options)

    rootdir = options.rootdir
    slurm = options.slurm
    if slurm is True and 'coxfs01' not in rootdir:
        rootdir = '/n/coxfs01/2p-data'
    animalid = options.animalid
    session = options.session
    acquisition = options.acquisition
    run = options.run
    trace_id = options.trace_id
    trace_type = options.trace_type
    auto = options.auto

    # Plotting param info:
    universal_scale = options.universal_scale
    ylim_min = float(options.ylim_min)
    ylim_max = float(options.ylim_max)
    plot_all_psths = options.plot_all_psths
    include_trials = options.include_trials
    roi_metric = options.roi_metric

    filter_pupil = options.filter_pupil
    pupil_size_thr = float(options.pupil_size_thr)
    pupil_dist_thr = float(options.pupil_dist_thr)
    pupil_max_nblinks = float(options.pupil_max_nblinks)

    plot_psth = options.psth
    plot_tuning = options.tuning

    # Get acquisition info:
    run_dir = os.path.join(rootdir, animalid, session, acquisition, run)

    # Load TRACE ID info:
    # =========================================================================
    TID = load_TID(run_dir, trace_id)
    traceid_dir = TID['DST']
    if rootdir not in traceid_dir:
        orig_root = traceid_dir.split('/%s/%s' % (animalid, session))[0]
        traceid_dir = traceid_dir.replace(orig_root, rootdir)
        print "Replacing orig root with dir:", traceid_dir
        #trace_hash = TID['trace_hash']

    # Get paradigm/AUX info:
    # =========================================================================

    event_info_fpath = [os.path.join(traceid_dir, f) for f in os.listdir(traceid_dir) if 'event_alignment' in f and f.endswith('json')][0]
    with open(event_info_fpath, 'r') as f:
        trial_info = json.load(f)
    if rootdir not in trial_info['parsed_trials_source']:
        trial_info['parsed_trials_source'] = replace_root(trial_info['parsed_trials_source'], rootdir, animalid, session)
    configs, stimtype = get_stimulus_configs(trial_info)



    # Load particular metrics set:
    selected_metric = get_metric_set(traceid_dir, filter_pupil=filter_pupil,
                                         pupil_size_thr=pupil_size_thr,
                                         pupil_dist_thr=pupil_dist_thr,
                                         pupil_max_nblinks=pupil_max_nblinks)

    # Load associated pupil_params set:
    with open(os.path.join(traceid_dir, 'metrics', selected_metric, 'pupil_params.json'), 'r') as f:
        pupil_params = json.load(f)
    pp.pprint(pupil_params)

    # Set paths toi ROIDATA_, roi_metrics_, and roi_stats_:
    roistats_filepath = [os.path.join(traceid_dir, 'metrics', selected_metric, f)
                            for f in os.listdir(os.path.join(traceid_dir, 'metrics', selected_metric))
                            if 'roi_stats_' in f  and trace_type in f and f.endswith('hdf5')][0]

    roidata_filepath = [os.path.join(traceid_dir, f)
                            for f in os.listdir(traceid_dir)
                            if 'ROIDATA_' in f and trace_type in f and f.endswith('hdf5')][0]


    # PSTHS:
    # Set plotting params for trial average plots for each ROI:
    # =============================================================================
    roi_psth_dir = os.path.join(traceid_dir, 'figures', 'psths', trace_type)

    if plot_psth is True:
        print "-------------------------------------------------------------------"
        print "Plotting PSTHS."
        print "-------------------------------------------------------------------"
        #% For each ROI, plot PSTH for all stim configs:
        if not os.path.exists(roi_psth_dir):
            os.makedirs(roi_psth_dir)
        print "Saving PSTH plots to: %s" % roi_psth_dir
        if plot_all_psths is True:
            print "--- saving to ./unfiltered"
        if filter_pupil is True:
            print "--- saving to ./%s" % selected_metric

        print "Plotting PSTHs.........."
        plot_psths(roidata_filepath, trial_info, configs, roi_psth_dir=roi_psth_dir, trace_type=trace_type,
                       filter_pupil=filter_pupil, pupil_params=pupil_params, plot_all=plot_all_psths,
                       universal_scale=universal_scale, ylim_min=ylim_min, ylim_max=ylim_max)


    # PLOT TUNING CURVES
    tuning_figdir_base = os.path.join(traceid_dir, 'figures', 'tuning', trace_type)

    # First, plot with ALL trials included:
    tuning_figdir = os.path.join(tuning_figdir_base, roi_metric)

    if plot_tuning is True:
        print "-------------------------------------------------------------------"
        print "Plotting tuning curves."
        print "-------------------------------------------------------------------"


        if filter_pupil is True:
            curr_tuning_figdir = os.path.join(tuning_figdir, selected_metric)
        else:
            curr_tuning_figdir = os.path.join(tuning_figdir, 'unfiltered')
        if not os.path.exists(curr_tuning_figdir):
            os.makedirs(curr_tuning_figdir)

        plot_tuning_curves(roistats_filepath, configs, curr_tuning_figdir, metric_type=roi_metric, include_trials=include_trials)


    print "==================================================================="

    return roi_psth_dir, tuning_figdir, selected_metric

#%%

def main(options):

    roi_psth_dir, tuning_figdir, selected_metric = plot_roi_figures(options)

    print "*******************************************************************"
    print "Done creating ROI plots!"
    print "-------------------------------------------------------------------"
    print "PSTHs are saved to base dir: %s" % roi_psth_dir
    print "Tuning curves are saved to base dir: %s" % tuning_figdir
    print "Selected pupil-filter metric was: %s" % selected_metric
    print "*******************************************************************"


#%%

if __name__ == '__main__':
    main(sys.argv[1:])



