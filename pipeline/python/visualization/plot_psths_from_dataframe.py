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

#%%
def get_axis_limits(ax, xscale=.9, yscale=0.9):
    return ax.get_xlim()[1]*xscale, ax.get_ylim()[1]*yscale

#%

def set_figure_params(nrows, ncols, fontsize_pt=20, dpi=72.27, spacer=20, top_margin=0.01, bottom_margin=0.05):

    # comput the matrix height in points and inches
    matrix_height_pt = fontsize_pt * nrows * spacer
    matrix_height_in = matrix_height_pt / dpi

    # compute the required figure height  # in percentage of the figure height
    figure_height = matrix_height_in / (1 - top_margin - bottom_margin)

    return figure_height, top_margin, bottom_margin



#%%
def get_facet_stats(config_list, g, value='df'):
    plotstats = {'dfmats': {}, 'indices': {}}
    dfmats = dict((config, []) for config in config_list)
    plotindices = dict((config, []) for config in config_list)
    for (row_i, col_j, hue_k), data_ijk in g.facet_data():
        if len(data_ijk) == 0:
            continue
        dfmats[list(set(data_ijk['config']))[0]].append(data_ijk[value])
        plotindices[list(set(data_ijk['config']))[0]].append((row_i, col_j))

    for config in dfmats.keys():
        plotstats['dfmats'][config] = np.array(dfmats[config])
        plotstats['indices'][config] = list(set(plotindices[config]))[0]

    return plotstats

#%%
def plot_roi_psth(roi, roiDF, object_transformations, figdir='/tmp', prefix='psth', trace_color='k', stimbar_color='r', save_and_close=True):

    #roi = list(set(roiDF['roi']))[0]

    if 'ori' in roiDF.keys():
        stimtype = 'grating'
    else:
        stimtype = 'image'

    trans_types = object_transformations.keys()

    if stimtype == 'image':
        object_sorter = 'object'
    else:
        if 'ori' in trans_types and not 'sf' in trans_types:
            object_sorter = 'ori'
        elif 'ori' in trans_types and 'sf' in trans_types:
            object_sorter = 'sf'

    single_object_figure = False
    if 'size' in trans_types and ('xpos' in trans_types or 'ypos' in trans_types):
        # ---- Transform description ---------------------------
        # Change SIZE & change POSITION
        # NOTE:  only tested with size + horizontal translation
        # ------------------------------------------------------
        row_order = sorted(list(set(roiDF['position'])))
        col_order = sorted(list(set(roiDF['size'])))
        rows = 'position'
        columns = 'size'
        single_object_figure = True
        figbase = 'pos%i_size%i' % (len(row_order), len(col_order))
    elif 'xpos' in trans_types and 'ypos' in trans_types:
        # ---- Transform description ---------------------------
        # Change POSITIONS only, grid w/ x and y positions
        # NOTE:  only tested with 4 orientations + x-,y-position grid
        # ------------------------------------------------------
        row_order = sorted(list(set(roiDF['ypos'])))[::-1] # Reverse order so POS are on top, NEG on bottom
        col_order = sorted(list(set(roiDF['xpos'])))
        rows = 'ypos'
        columns = 'xpos'
        single_object_figure = True
        figbase = 'xpos%i_ypos%i' % (len(row_order), len(col_order))
    else:
        # ---- Transform description ---------------------------
        # POS and SIZE at single value, only changing [morph or yrot
        # NOTE:  only tested with 4 orientations + x-,y-position grid
        # ------------------------------------------------------
        figbase = 'all_objects_default_pos_size'
        if stimtype == 'grating':
            row_order = sorted(list(set(roiDF['sf'])))
            col_order = sorted(list(set(roiDF['ori'])))
            rows = 'sf'
            columns = 'ori'
        else:
            row_order = sorted(list(set(roiDF['morphlevel'])))
            col_order = sorted(list(set(roiDF['yrot'])))
            rows = 'morphlevel'
            columns = 'yrot'

    if single_object_figure is True:
        for objectid in list(set(roiDF[object_sorter])):
            currdf = roiDF[roiDF[object_sorter] == objectid]
            figpath = os.path.join(figdir, '%s_%s%s_%s.png' % (prefix, object_sorter, objectid, figbase))
            draw_psth(roi, currdf, objectid, trans_types, rows, columns, row_order, col_order, trace_color, stimbar_color, figpath, save_and_close=save_and_close)
    else:
        objectid = 'all'
        figpath = os.path.join(figdir, '%s_%s.png' % (prefix, figbase))
        draw_psth(roi, roiDF, trans_types, rows, columns, row_order, col_order, trace_color, stimbar_color, figpath, save_and_close=save_and_close)

#%%
def draw_psth(roi, roiDF, objectid, trans_types, rows, columns, row_order, col_order, trace_color, stimbar_color, figpath, save_and_close=True):

    #roi = list(set(roiDF['roi']))[0]
    config_list = list(set(roiDF['config']))

#    funky_trials = [trial for trial in list(set(roiDF['trial'])) if max(roiDF[roiDF['trial']==trial]['df'])>10]
#    funky_idxs = roiDF.index[roiDF['trial'].isin(funky_trials)].tolist()
#    roiDF.loc[funky_idxs, 'df'] = np.nan

    # Add n trials to each subplot:
    ntrials = dict((c, len(list(set(roiDF[roiDF['config']==c]['trial'])))) for c in config_list)

    first_on = list(set(roiDF['first_on']))[0]
    nsecs_on = list(set(roiDF['nsecs_on']))[0]
    tsecs = sorted(list(set(roiDF['tsec'])))

    #pl.figure()
    plot_vars = ['trial', 'df', 'tsec', rows, columns]
    plot_vars.extend([t for t in trans_types if t not in plot_vars])

    subDF = roiDF[plot_vars]
    g1 = sns.FacetGrid(subDF, row=rows, col=columns, sharex=True, sharey=True, hue='trial', row_order=row_order, col_order=col_order)
    g1.map(pl.plot, "tsec", "df", linewidth=0.2, color=trace_color, alpha=0.5)
    #plotstats = get_facet_stats(config_list, g1, value='df')

    nrows = len(g1.row_names)
    ncols = len(g1.col_names)
    for ri, rowval in enumerate(g1.row_names):
        for ci, colval in enumerate(g1.col_names):
            currax = g1.facet_axis(ri, ci)
            configDF = subDF[((subDF[rows]==rowval) & (subDF[columns]==colval))]
            dfmat = []
            tmat = []
            for trial in list(set(configDF['trial'])):
                dfmat.append(np.array(configDF[configDF['trial']==trial]['df'][:]))
                tmat.append(np.array(configDF[configDF['trial']==trial]['tsec'][:]))
            dfmat = np.array(dfmat); tmat = np.array(tmat);
            curr_ntrials = dfmat.shape[0]
            mean_df = np.mean(dfmat, axis=0)
            mean_tsec = np.mean(tmat, axis=0)
            if ri == 0 and ci == 0:
                stimbar_pos = dfmat.min() - dfmat.min()*-.25
            currax.plot(mean_tsec, mean_df, trace_color, linewidth=1, alpha=1)
            currax.plot([mean_tsec[first_on], mean_tsec[first_on]+nsecs_on], [stimbar_pos, stimbar_pos], stimbar_color, linewidth=2, alpha=1)
            currax.annotate("n = %i" % curr_ntrials, xy=get_axis_limits(currax, xscale=0.2, yscale=0.8))

    # Get mean trace:
#    meandfs = {}
#    for config in plotstats['dfmats'].keys():
#        meandfs[config] = np.mean(plotstats['dfmats'][config], axis=0)
#        currax = g1.facet_axis(plotstats['indices'][config][0], plotstats['indices'][config][1])
#        currax.plot(tsecs, meandfs[config], trace_color, linewidth=1, alpha=1)
#        currax.plot([tsecs[first_on], tsecs[first_on]+nsecs_on], [0, 0], stimbar_color, linewidth=2, alpha=1)
#        currax.annotate("n = %i" % ntrials[config], xy=get_axis_limits(currax, xscale=0.2, yscale=0.8))

    sns.despine(offset=2, trim=True)
    #%
    pl.subplots_adjust(top=0.9)
    g1.fig.suptitle("%s - stim%s" % (roi, objectid))

    if save_and_close is True:
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
                   ylim_min=-1.0, ylim_max=1.0, visualization_method='separate_transforms'):

    if plot_all is False and filter_pupil is False:
        print "No PSTH types specified. Exiting."
        return

    if filter_pupil is True:
        if pupil_params is None:
            pupil_params = set_pupil_params()
        pupil_max_nblinks = pupil_params['max_nblinks']
        pupil_radius_min = float(pupil_params['radius_min'])
        pupil_radius_max = float(pupil_params['radius_max'])
        #pupil_size_thr = pupil_params['size_thr']
        pupil_dist_thr = float(pupil_params['dist_thr'])

    # UNFILTERED psth plot dir:
    roi_psth_dir_all = os.path.join(roi_psth_dir, 'unfiltered', visualization_method)
    if not os.path.exists(roi_psth_dir_all):
        os.makedirs(roi_psth_dir_all)

    # Excluded/Included psth plot dirs:
    if filter_pupil is True:
        #pupil_thresh_str = 'pupil_size%i-dist%i-blinks%i' % (pupil_size_thr, pupil_dist_thr, int(pupil_max_nblinks))
        pupil_thresh_str = 'pupil_rmin%.2f-rmax%.2f-dist%.2f' % (pupil_radius_min, pupil_radius_max, pupil_dist_thr)
        roi_psth_dir_include = os.path.join(roi_psth_dir, pupil_thresh_str, visualization_method, 'include')
        if not os.path.exists(roi_psth_dir_include):
            os.makedirs(roi_psth_dir_include)
        roi_psth_dir_exclude = os.path.join(roi_psth_dir, pupil_thresh_str, visualization_method, 'exclude')
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

            roiDF = DATA[roi] #[DATA[roi]['config'].isin(curr_subplots)]
            roiDF['position'] = list(zip(roiDF['xpos'], roiDF['ypos']))

            curr_slice = list(set(roiDF['slice']))[0] #roi_trials[configname][roi].attrs['slice']
            roi_in_slice = list(set(roiDF['roi_in_slice']))[0] #roi_trials[configname][roi].attrs['idx_in_slice']

            # PLOT ALL:
            if plot_all is True:
                prefix = '%s_%s_%s_%s_ALL' % (roi, curr_slice, roi_in_slice, trace_type) #, figname)
                plot_roi_psth(roi, roiDF, object_transformations, figdir=roi_psth_dir_all, prefix=prefix, trace_color='k', stimbar_color='r')

                # Plot df values:
            if filter_pupil is True:
                filtered_DF = roiDF.query('pupil_size_stimulus > @pupil_radius_min \
                                       & pupil_size_baseline > @pupil_radius_min \
                                       & pupil_size_stimulus < @pupil_radius_max \
                                       & pupil_size_baseline < @pupil_radius_max \
                                       & pupil_dist_stimulus < @pupil_dist_thr \
                                       & pupil_dist_baseline < @pupil_dist_thr \
                                       & pupil_nblinks_stim <= @pupil_max_nblinks \
                                       & pupil_nblinks_baseline >= @pupil_max_nblinks')
#
#                filtered_DF = DF.query('pupil_size_stimulus > @pupil_size_thr \
#                                       & pupil_size_baseline > @pupil_size_thr \
#                                       & pupil_dist_stimulus < @pupil_dist_thr \
#                                       & pupil_dist_baseline < @pupil_dist_thr \
#                                       & pupil_nblinks_stim < @pupil_max_nblinks \
#                                       & pupil_nblinks_baseline < @pupil_max_nblinks')
                pass_trials = list(set(filtered_DF['trial']))

                # INCLUDED trials:
                prefix = '%s_%s_%s_%s_PUPIL_%s_pass.png' % (roi, curr_slice, roi_in_slice, trace_type, pupil_thresh_str)
                plot_roi_psth(roi, filtered_DF, object_transformations,
                                  figdir=roi_psth_dir_include, prefix=prefix, trace_color='b', stimbar_color='k')

                # EXCLUDED trials:
                excluded_DF = roiDF[~roiDF['trial'].isin(pass_trials)]
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
def plot_tuning_curves(roistats_filepath, configs, curr_tuning_figdir,
                       metric_type='zscore', include_trials=True,
                       visualization_method='separate_transforms',
                       save_and_close=True):

    STATS = pd.HDFStore(roistats_filepath, 'r')['/df']


    if 'frequency' in configs[configs.keys()[0]].keys():
        stimtype = 'grating'
    else:
        stimtype = 'image'

    roi_list = sorted(list(set(STATS['roi'])), key=natural_keys)

    transform_dict, object_transformations = get_object_transforms(STATS)
    trans_types = object_transformations.keys()

    for roi in sorted(roi_list, key=natural_keys):
        print roi
        roiDF = STATS[STATS['roi'] == roi]
        # Check if valid traces exist:
        if len(roiDF[np.isfinite(roiDF['zscore'])]) == 0:
            print "***WARNING: -- tuning curves --***"
            print "-- No valid traces found for ROI: %s" % roi
            print "-- skipping..."
            continue

        if visualization_method == 'separate_transforms':
            plot_tuning_by_transforms(roiDF, transform_dict, object_transformations,
                                      metric_type=metric_type,
                                      output_dir=curr_tuning_figdir,
                                      include_trials=include_trials,
                                      save_and_close=save_and_close)
        else:
            if stimtype == 'grating':
                object_transformations = {}
                for trans in trans_types:
                    object_transformations[trans] = []
                plot_tuning_collapse_transforms(roiDF, object_transformations,
                                                object_type='grating',
                                                metric_type=metric_type,
                                                output_dir=curr_tuning_figdir,
                                                include_trials=include_trials,
                                                save_and_close=save_and_close)
            else:
                plot_tuning_collapse_transforms(roiDF, object_transformations,
                                                object_type='object',
                                                metric_type=metric_type,
                                                output_dir=curr_tuning_figdir,
                                                include_trials=include_trials,
                                                save_and_close=save_and_close)




    #STATS.close()

#%%


def plot_tuning_by_transforms(roiDF, transform_dict, object_transformations, metric_type='zscore',
                              save_and_close=True, output_dir='/tmp', include_trials=True):
    '''
    trans_type = feature that varies for a given object ID (i.e., how we want to color-code)
    object_sorter = how we want to define object ID
        -- for objects, this is the object ID (e.g., Blobs_N1, Blobs_N2, morph5, or just morph, if only 1 morph)
        -- for gratings, this is the spatial frequency, since we want to look at variation in ORI (trans_type) within a given S.F.
    hue = should be the same as object_sorter (except if only a single morph, since there is only 1 morphID)
    '''
    fignames = []
    stim_subsets = False

    roi = list(set(roiDF['roi']))[0]

    colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple"]
    cmapcolors = itertools.cycle(sns.xkcd_palette(colors))

    trans_types = object_transformations.keys()

    # Grid x,y positions, and vary other trans_type(s??) as curves:
    grid_variables = ['xpos', 'ypos']
    nrows = len(transform_dict['ypos'])
    rows = 'ypos'
    row_order = sorted(list(set(roiDF['ypos'])))[::-1]

    ncols = len(transform_dict['xpos'])
    columns = 'xpos'
    col_order = sorted(list(set(roiDF['xpos'])))

    other_trans_types = [t for t in object_transformations.keys() if not t in grid_variables]
    desc = 'grid_xypos_%s' % '_'.join(other_trans_types)

    if len(other_trans_types) == 1:
        xval_trans = other_trans_types[0]
        hue_trans = None

    elif 'ori' in other_trans_types and 'sf' in other_trans_types:
        xval_trans = 'ori'   # Plot orientation along x-axis
        hue_trans = 'sf'     # Set hue by spatial frequency

    elif 'morphlevel' in other_trans_types and ('yrot' in other_trans_types or 'size' in other_trans_types):
        # Actually want to plot a figure for EACH transform...
        desc = {}
        hues = {}
        stim_subsets = True
        plotconfig = []
        for trans_type in other_trans_types:
            desc[trans_type] = 'grid_xypos_%s' % trans_type
            hues[trans_type] = {}
            trans_vals = roiDF.groupby([trans_type]).groups.keys()                                      # Get all values of current trans_type
            comparison_trans_types = [i for i in other_trans_types if not i == trans_type]           # Identify which other transform there is (only 2)
            for comp_trans in comparison_trans_types:
                vals_bytrans = [list(set(roiDF[roiDF[trans_type]==tr][comp_trans])) for tr in trans_vals]  # Get other-transform vals for each value of current trans_type
                all_vals = list(set(list(itertools.chain.from_iterable(vals_bytrans))))
                common_vals = list(reduce(set.intersection, map(set, vals_bytrans)))                  # Only include other-trans vals common to all vals of trans_type

#                if not len(common_vals) == len(all_vals):
#                    stim_subsets = True
#                else:
#                    stim_subsets = False
                plotconfig.append(stim_subsets)
                hues[trans_type][comp_trans] = common_vals



    # Set up colors for transforms with multiple combos with other transforms
    color_dict = {}
    if stim_subsets is True:
        # There multiple transforms we want for the 'x-axis' in a given subplot
        for xval_trans in hues.keys():
            for comp_trans in hues[xval_trans].keys():
                comparison_values = hues[xval_trans][comp_trans]
                color_dict[xval_trans] = [next(cmapcolors) for v in comparison_values]
    else:
        # There is only one main transform to look at, with the other trans
        # denoted by hue
        if hue_trans is not None:
            comparison_values = list(set(roiDF[hue_trans]))
            color_dict[xval_trans] = [next(cmapcolors) for v in comparison_values] #'xkcd:%s' % cmapcolors[t]
        else:
            color_dict[xval_trans] = [next(cmapcolors)]


    # Append summary stats for metric_type = 'zscore' to dataframe for plotting:

    dfz = roiDF[np.isfinite(roiDF[metric_type])]     # Get rid of NaNs
    plot_variables = list(trans_types)      # List of transforms (variables-of-interest) to pull from dataframe (e.g.,'ori')
    plot_variables.extend([metric_type])    # Append metric to show in list of variables-of-interest
    if 'xpos' not in plot_variables:
        plot_variables.extend(['xpos'])
    if 'ypos' not in plot_variables:
        plot_variables.extend(['ypos'])

    dfz = dfz[plot_variables]               # Get subset dataframe of variables-of-interest

    grouped = dfz.groupby(trans_types, as_index=False)                         # Group dataframe subset by variables-of-interest
    zscores = grouped.zscore.mean()                                            # Get mean of 'metric_type' for each combination of transforms
    zscores['sem'] = grouped.zscore.aggregate(stats.sem)[metric_type]             # Get SEM
    zscores = zscores.rename(columns={metric_type: 'mean_%s' % metric_type})                # Rename 'zscore' column to 'mean_zscore' so we can merge
    dfz = dfz.merge(zscores).sort_values([xval_trans])                         # Merge summary stats to each corresponding row (indexed by columns values in that row)

    # Plot tuning curves:

    #sns.set_style("white")
    sns.set()
    for xval_trans in color_dict.keys():

        if isinstance(desc, str):
            transform_str = desc
        elif isinstance(desc, dict):
            transform_str = desc[xval_trans]

        print "Plotting tuning for: %s" % transform_str

        if stim_subsets is True and isinstance(hues, dict):
            hue_trans = hues[xval_trans].keys()[0]  # Should only be one for now...
            plotdf = dfz[dfz[hue_trans].isin(hues[xval_trans][hue_trans])]
        else:
            plotdf = dfz.copy()
        #sns.boxplot(x="size", y="zscore", hue="morphlevel", data=plotdf)
        g1 = sns.FacetGrid(plotdf.sort_values([xval_trans]), row=rows, col=columns, sharex=True, sharey=True,
                               hue=hue_trans, row_order=row_order, col_order=col_order,
                               legend_out=True, size=6)
        g1.map(pl.plot, xval_trans, 'mean_%s' % metric_type, marker="_", linestyle='-')
        g1.map(pl.errorbar, xval_trans, 'mean_%s' % metric_type, 'sem', elinewidth=1, linewidth=1)
        if include_trials:
            g1.map(pl.scatter, xval_trans, metric_type, s=1.0)

        # Format ticks, title, etc.
        g1.set(xticks=sorted(plotdf[xval_trans]))
        sns.despine(offset=2, trim=True, bottom=True)
        pl.subplots_adjust(top=0.8)
        g1.fig.suptitle(roi)

        # resize figure box to -> put the legend out of the figure
        if hasattr(g1, 'ax'):
            box = g1.ax.get_position() # get position of figure
            g1.ax.set_position([box.x0, box.y0, box.width * 0.8, box.height]) # resize position
            g1.ax.legend(loc='center right', bbox_to_anchor=(1.2, 0.8), ncol=1)
        else:
            box = g1.facet_axis(-1, -1).get_position() # get position of figure
            g1.facet_axis(-1, -1).set_position([box.x0, box.y0, box.width * 0.8, box.height]) # resize position
            g1.facet_axis(-1, -1).legend(loc='center right', bbox_to_anchor=(1.2, 0.8), ncol=1)
        if '/' in roi:
            roi = roi[1:]

        if include_trials is True:
            figname = '%s_tuning_%s_trials_%s.png' % (transform_str, metric_type, roi)
        else:
            figname = '%s_tuning_%s_%s.png' % (transform_str, metric_type, roi)

        if save_and_close is True:
            pl.savefig(os.path.join(output_dir, figname))
            pl.close()
        else:
            fignames.append(figname)

    return fignames


#%%
def plot_tuning_collapse_transforms(roiDF, object_transformations, object_type='object',
                                    save_and_close=True, metric_type='zscore', output_dir='/tmp', include_trials=True):
    '''
    trans_type = feature that varies for a given object ID (i.e., how we want to color-code)
    object_sorter = how we want to define object ID
        -- for objects, this is the object ID (e.g., Blobs_N1, Blobs_N2, morph5, or just morph, if only 1 morph)
        -- for gratings, this is the spatial frequency, since we want to look at variation in ORI (trans_type) within a given S.F.
    hue = should be the same as object_sorter (except if only a single morph, since there is only 1 morphID)
    '''
    fignames = []

    roi = list(set(roiDF['roi']))[0]

    nrows = len(object_transformations.keys())
    fig, axes = pl.subplots(nrows=nrows, ncols=1, sharex=False, squeeze=True, figsize=(6,10))
    transform_str = '_'.join(object_transformations.keys())
    for trans, ax in zip(object_transformations.keys(), axes):
        #print trans
        other_trans_types = [t for t in object_transformations.keys() if not t==trans]
        if object_type == 'grating':
            if trans == 'ori':
                object_color = 'sf'
                object_sorter = 'sf'
            elif trans == 'sf':
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
            ROI = roiDF[roiDF['object'].isin(object_transformations[trans])]

            # TODO:  tmp filter to exclude YROT objects -30 and 30
            if trans == 'morphlevel':
                ROI = ROI[ROI['yrot']==0]

        else:
            ROI = roiDF
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


    if save_and_close is True:
        pl.savefig(os.path.join(output_dir, figname))
        pl.close()
    else:
        fignames.append(figname)

    return fignames


#

#%%

def extract_options(options):

    choices_tracetype = ('raw', 'raw_fissa', 'denoised_nmf', 'np_corrected_fissa', 'neuropil_fissa', 'np_subtracted', 'neuropil')
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

    parser.add_option('--collapse', action="store_false",
                      dest="separate_transforms", default=True, help="Set flag to collapse across all other transforms for a given transform (default separates)")

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
    parser.add_option('-s', '--radius-min', action="store",
                      dest="pupil_radius_min", default=25, help="Cut-off for smnallest pupil radius, if --pupil set [default: 25]")
    parser.add_option('-B', '--radius-max', action="store",
                      dest="pupil_radius_max", default=65, help="Cut-off for biggest pupil radius, if --pupil set [default: 65]")
    parser.add_option('-d', '--dist', action="store",
                      dest="pupil_dist_thr", default=15, help="Cut-off for pupil distance from start, if --pupil set [default: 5]")

    (options, args) = parser.parse_args(options)

    return options

#%%

# Test basic pplotting for ORI/SF only:
#options = ['-D', '/mnt/odyssey', '-i', 'CE077', '-S', '20180323',
#        '-A', 'FOV1_zoom1x', '-R', 'gratings_run1', '-t', 'traces001',
#        '-s', '20', '-B', '60', '-d', '8',
#        '--omit-trials', '--psth', '--tuning',
#        '-T', 'raw']

# Test plotting with varying x-,y-positions and 1 feature (ORI):
#options = ['-D', '/mnt/odyssey', '-i', 'CE077', '-S', '20180321',
#        '-A', 'FOV1_zoom1x', '-R', 'gratings', '-t', 'traces001',
#        '-s', '35', '-B', '60', '-d', '3',
#        '--omit-trials', '--psth', '--tuning',
#        '-T', 'np_subtracted']

# Test plotting with morphs + yrot, default position/size:

#options = ['-D', '/mnt/odyssey', '-i', 'CE077', '-S', '20180321',
#        '-A', 'FOV1_zoom1x', '-R', 'blobs_run4', '-t', 'traces001',
#        '-s', '30', '-B', '60', '-d', '3',
#        '--omit-trials', '--psth', '--tuning',
#        '-T', 'np_subtracted']

# Test plotting wtih morphs + vary pos/size:
#options = ['-D', '/mnt/odyssey', '-i', 'CE074', '-S', '20180221',
#        '-A', 'FOV1_zoom1x', '-R', 'blobs_run3', '-t', 'traces002',
#        '--no-pupil',
#        '--omit-trials', '--psth', '--tuning',
#        '-T', 'raw']

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
    pupil_radius_max = float(options.pupil_radius_max)
    pupil_radius_min = float(options.pupil_radius_min)
    pupil_dist_thr = float(options.pupil_dist_thr)

    plot_psth = options.psth
    plot_tuning = options.tuning

    separate_transforms = options.separate_transforms
    if separate_transforms is True:
        visualization_method = 'separate_transforms'
    else:
        visualization_method = 'collapse_transforms'


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
                                         pupil_radius_min=pupil_radius_min,
                                         pupil_radius_max=pupil_radius_max,
                                         pupil_dist_thr=pupil_dist_thr
                                         )

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
                       universal_scale=universal_scale, ylim_min=ylim_min, ylim_max=ylim_max,
                       visualization_method=visualization_method)


    # PLOT TUNING CURVES
    tuning_figdir_base = os.path.join(traceid_dir, 'figures', 'tuning', trace_type)

    # First, plot with ALL trials included:
    tuning_figdir = os.path.join(tuning_figdir_base, roi_metric)

    if plot_tuning is True:
        print "-------------------------------------------------------------------"
        print "Plotting tuning curves."
        print "-------------------------------------------------------------------"


        if filter_pupil is True:
            curr_tuning_figdir = os.path.join(tuning_figdir, selected_metric, visualization_method)
        else:
            curr_tuning_figdir = os.path.join(tuning_figdir, 'unfiltered', visualization_method)
        if not os.path.exists(curr_tuning_figdir):
            os.makedirs(curr_tuning_figdir)

        plot_tuning_curves(roistats_filepath, configs, curr_tuning_figdir,
                           metric_type=roi_metric, include_trials=include_trials, visualization_method=visualization_method)


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



