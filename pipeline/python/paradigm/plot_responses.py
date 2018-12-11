#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 20:36:25 2018

@author: juliana
"""


import matplotlib as mpl
mpl.use('agg')
import matplotlib.patches as patches
import os
import sys
import optparse
import itertools
import glob
import seaborn as sns
import numpy as np
import pandas as pd
import pylab as pl
from scipy import stats
from pipeline.python.paradigm import utils as util

from pipeline.python.paradigm import align_acquisition_events as acq
from pipeline.python.traces.utils import get_frame_info
from pipeline.python.utils import label_figure

#rootdir = '/mnt/odyssey'
#animalid = 'CE077'
#session = '20180521'
#acquisition = 'FOV2_zoom1x'
#run = 'gratings_run1'
#traceid = 'traces001'
#run_dir = os.path.join(rootdir, animalid, session, acquisition, run)
#iti_pre = 1.0


#options = ['-D', '/mnt/odyssey', '-i', 'CE077', '-S', '20180523', '-A', 'FOV1_zoom1x',
#           '-T', 'np_subtracted',
#           '-R', 'blobs_run2', '-t', 'traces001', '-d', 'dff']

#options = ['-D', '/mnt/odyssey', '-i', 'CE077', '-S', '20180602', '-A', 'FOV1_zoom1x',
#           '-T', 'np_subtracted',
#           '-R', 'blobs_dynamic_run7', '-t', 'traces001', '-d', 'dff']

#options = ['-D', '/mnt/odyssey', '-i', 'CE077', '-S', '20180612', '-A', 'FOV1_zoom1x',
#           '-T', 'np_subtracted',
#           '-R', 'blobs_run1', '-t', 'traces001', '-d', 'dff', 
#           '-r', 'yrot', '-c', 'xpos', '-H', 'morphlevel']

options = ['-D', '/n/coxfs01/2p-data','-i', 'JC031', '-S', '20181119', '-A', 'FOV1_zoom2p0x',
           '-R', 'blobs_run1', '-t', 'traces001', 
           '-d', 'dff',
           '-r', 'ypos', '-c', 'xpos', '-H', 'morphlevel']

def extract_options(options):

    parser = optparse.OptionParser()

    parser.add_option('-D', '--root', action='store', dest='rootdir',
                          default='/nas/volume1/2photon/data',
                          help='data root dir (dir w/ all animalids) [default: /nas/volume1/2photon/data, /n/coxfs01/2pdata if --slurm]')
    parser.add_option('--slurm', action='store_true', dest='slurm', default=False, help="set if running as SLURM job on Odyssey")

    parser.add_option('-i', '--animalid', action='store', dest='animalid',
                          default='', help='Animal ID')

    # Set specific session/run for current animal:
    parser.add_option('-S', '--session', action='store', dest='session',
                          default='', help='session dir (format: YYYMMDD_ANIMALID')
    parser.add_option('-A', '--acq', action='store', dest='acquisition',
                          default='FOV1', help="acquisition folder (ex: 'FOV1_zoom3x') [default: FOV1]")
    parser.add_option('-T', '--trace-type', action='store', dest='trace_type',
                          default='raw', help="trace type [default: 'raw']")
#    parser.add_option('-R', '--run', dest='run_list', default=[], nargs=1,
#                          action='append',
#                          help="run ID in order of runs")
#    parser.add_option('-t', '--traceid', dest='traceid_list', default=[], nargs=1,
#                          action='append',
#                          help="trace ID in order of runs")
    parser.add_option('-R', '--run', dest='run', default='', action='store', help="run name")
    parser.add_option('-t', '--traceid', dest='traceid', default=None, action='store', help="datestr YYYYMMDD_HH_mm_SS")

    # Set specific session/run for current animal:
    parser.add_option('-d', '--datatype', action='store', dest='datatype',
                          default='corrected', help='Traces to plot (must be in dataset.npz [default: corrected]')
    parser.add_option('--offset', action='store_true', dest='correct_offset',
                          default=False, help='Set to correct df/f offset after drift correction')           
    parser.add_option('-f', '--filetype', action='store', dest='filetype',
                          default='png', help='File type for images [default: png]')
    parser.add_option('--scale', action='store_true', dest='scale_y',
                          default=False, help='Set to scale y-axis across roi images')
    parser.add_option('-y', '--ymax', action='store', dest='dfmax',
                          default=None, help='Set value for y-axis scaling (if not provided, and --scale, uses max across rois)')
    parser.add_option('--shade', action='store_false', dest='plot_trials',
                          default=True, help='Set to plot mean and sem as shaded (default plots individual trials)')
    parser.add_option('--median', action='store_true', dest='plot_median',
                          default=False, help='Set to plot MEDIAN (default plots mean across trials)')


    parser.add_option('-r', '--rows', action='store', dest='rows',
                          default=None, help='Transform to plot along ROWS (only relevant if >2 trans_types) - default uses objects or morphlevel')
    parser.add_option('-c', '--columns', action='store', dest='columns',
                          default=None, help='Transform to plot along COLUMNS')
    parser.add_option('-H', '--hue', action='store', dest='subplot_hue',
                          default=None, help='Transform to plot by HUE within each subplot')
    parser.add_option('--filter', action='store_true', dest='filter_noise',
                          default=False, help='Set to filter our noisy spikes') 

    (options, args) = parser.parse_args(options)
    if options.slurm:
        options.rootdir = '/n/coxfs01/2p-data'
    
    return options


##%%
#    
#    if len(trans_types) == 1:
#        stim_grid = (transform_dict[trans_types[0]],)
#        sgroups = sconfigs_df.groupby(sorted(trans_types))
#        ncols = len(stim_grid[0])
#        columns = trans_types[0]
#        col_order = sorted(stim_grid[0])
#        nrows = 1; rows = None; row_order=None
#        
#    elif len(trans_types) >= 2:
#        object_list = list(set([sconfigs[c]['object'] for c in sconfigs.keys()]))
#        nobjects = len(object_list)
#        
#        if 'morphlevel' in trans_types or nobjects > 1:
#            # We always do 5x5xN, where one axis is morphlevel, so use rows for morphs
#            # Use the other axis (xpos, yrot, etc.) as columns.
#            # Can plot 3rd axis, if exists, as hue, if multi_plot specified.
#            if multi_plot is None:
#                if 'morphlevel' in trans_types:
#                    plot_by = 'morphlevel'
#                else:
#                    plot_by = 'object'
#                other_transforms = [t for i,t in enumerate(trans_types) if t != plot_by]  # Transform types that are NOT morphlevel=
#                plot_columns = other_transforms[0]
#            else:
#                plot_by = [i for i in trans_types if i != multi_plot][0]
#                other_transforms = [t for i,t in enumerate(trans_types) if t != plot_by and t != multi_plot]  # Transform types that are NOT morphlevel=
#                plot_columns = [t for t in other_transforms if t != multi_plot][0]
#                
#            print "COLUMNS:", plot_columns #trans_types[plot_columns]
#            stim_grid = (sorted(transform_dict[plot_by]), sorted(transform_dict[plot_columns]))
#            if len(other_transforms) > 0:
#                # Use 1 other-trans for grid columns, use the 2nd trans for color:
#                if multi_plot is None:
#                    multi_plot = [t for t in other_transforms if t != plot_columns][0] #trans_types[other_indices[-1]]
#                sgroups = sconfigs_df.groupby(sorted([plot_by, plot_columns]))
#            else:
#                # Only 1 other trans_type, use as other axis on grid:
#                sgroups = sconfigs_df.groupby(sorted(trans_types))
#                
#        elif alt_axis is not None:
#            real_transform = [t for t in trans_types if t != alt_axis][0]
#            alt_axis_values = object_transformations[real_transform]
#            
#            other_indices = [i for i,t in enumerate(trans_types) if t != alt_axis]  # Transform types that are NOT morphlevel
#            if multi_plot is None:
#                transform_columns = other_indices[0]
#            else:
#                transform_columns = [i for i in other_indices if trans_types[i] != multi_plot][0]
#                
#            print "COLUMNS:", trans_types[transform_columns]
#            stim_grid = (sorted(alt_axis_values), sorted(transform_dict[trans_types[transform_columns]]))
#            if len(other_indices) > 1:
#                # Use 1 other-trans for grid columns, use the 2nd trans for color:
#                if multi_plot is None:
#                    multi_plot = trans_types[other_indices[-1]]
#                sgroups = sconfigs_df.groupby(sorted(['morphlevel', trans_types[transform_columns]]))
#            else:
#                # Only 1 other trans_type, use as other axis on grid:
#                sgroups = sconfigs_df.groupby(sorted(trans_types))
#                
#                
#                
#        ncols = len(stim_grid[1])
#        columns = plot_columns #trans_types[transform_columns]
#        col_order = sorted(stim_grid[1])
#        nrows = 1; rows = None; row_order=None
#        if len(stim_grid) == 2:
#            nrows = len(stim_grid[0])
#            rows = stim_grid[0]
#            row_order = sorted(stim_grid[0])
            
#%%
                          
                          
def make_clean_psths(options):
    
    optsE = extract_options(options)
    
    #traceid_dir = util.get_traceid_dir(options)
    run = optsE.run #run_list[0]
    traceid = optsE.traceid #traceid_list[0]
    acquisition_dir = os.path.join(optsE.rootdir, optsE.animalid, optsE.session, optsE.acquisition)
    if 'cnmf' in traceid:
        traceid_dir = glob.glob(os.path.join(acquisition_dir, run, 'cnmf', '%s*' % traceid))[0]
    else:
        traceid_dir = [t for t in glob.glob(os.path.join(acquisition_dir, run, 'traces', '%s*' % traceid)) if 'ORIG' not in t][0]

#    if '_' not in traceid: 
#        traceid_dir = util.get_traceid_from_acquisition(acquisition_dir, run, traceid)
#    else:
#        traceid_dir = os.path.join(acquisition_dir, run, 'traces', traceid)

    data_fpath = os.path.join(traceid_dir, 'data_arrays', 'datasets.npz')
    print "Loaded data from: %s" % data_fpath #traceid_dir
#    dataset = np.load(data_fpath)
#    print dataset.keys()
#        
    #%
    #optsE = extract_options(options)
    inputdata = optsE.datatype
    correct_offset = optsE.correct_offset
    filetype = optsE.filetype
        
    filter_noise = optsE.filter_noise
    
    dfmax = optsE.dfmax
    scale_y = optsE.scale_y
    plot_trials = optsE.plot_trials
    plot_median = optsE.plot_median

    subplot_hue = optsE.subplot_hue
    rows = optsE.rows
    columns = optsE.columns
    
    data_identifier = '_'.join((optsE.animalid, optsE.session, optsE.acquisition, optsE.traceid))

    
    dataset = np.load(data_fpath)
    print dataset.keys()
        
    #ridx = 0
    #inputdata = 'dff' #corrected'
    assert inputdata in dataset.keys(), "Specified data type (%s) not found! Choose from: %s" % (inputdata, str(dataset.keys()))
    if inputdata == 'corrected' or inputdata=='smoothedX':
        ylabel = 'intensity'
    elif inputdata == 'dff' or inputdata=='smoothedDF':
        ylabel = 'df/f'
    elif inputdata == 'spikes':
        ylabel = 'inferred'
        
        
    xdata = dataset[inputdata]
    
    # Filter out noisy spikes:
    figdir_append = ''
    if inputdata == 'spikes' and filter_noise:
        xdata[xdata<=0.0004] = 0.
        figdir_append = '_filtered'
    if plot_median:
        figdir_append = '%s_median' % figdir_append 
    
    #ydata = dataset['ylabels']
    #tsecs = dataset['tsecs']
    
    run_info = dataset['run_info'][()]
    nframes_per_trial = run_info['nframes_per_trial']
    ntrials_by_cond = run_info['ntrials_by_cond']
    ntrials_total = sum([val for k,val in ntrials_by_cond.iteritems()])
    #trial_labels = np.reshape(ydata, (ntrials_total, nframes_per_trial))[:,0]

    nrois = xdata.shape[-1]
#    xdata = dataset['corrected']
#    F0 = dataset['raw'] - dataset['corrected']
    
    
    labels_df = pd.DataFrame(data=dataset['labels_data'], columns=dataset['labels_columns'])
    
#    
#    xmat = np.reshape(xdata, (ntrials_total, nframes_per_trial, nrois))    
#    print xmat.shape
#    stim_on = dataset['run_info'][()]['stim_on_frame']
#    bas_frames = xmat[:, 0:stim_on, :]
#    bas_means = np.mean(bas_frames, axis=1)
#    bas_grand_mean = np.mean(bas_means, axis=0)

    
    # Get stimulus info:
    sconfigs = dataset['sconfigs'][()]
    transform_dict, object_transformations = util.get_transforms(sconfigs)
   
    # replace duration:
    if 'duration' in transform_dict.keys():
        transform_dict['stim_dur'] = transform_dict['duration']
        transform_dict.pop('duration')
    trans_types = sorted([trans for trans in transform_dict.keys() if len(transform_dict[trans]) > 1])        
    print "Trans:", trans_types
    print object_transformations #transform_dict
#
#    if alt_axis is not None:
#        trans_types.extend([alt_axis])

#    tpoints = np.reshape(tsecs, (ntrials_total, nframes_per_trial))[0,:]
#    labeled_trials = np.reshape(ydata, (ntrials_total, nframes_per_trial))[:,0]
    
    tested_configs = list(set(labels_df['config']))
    for c in sconfigs.keys():
        if c not in tested_configs:
            sconfigs.pop(c)
            
    sconfigs_df = pd.DataFrame(sconfigs).T
    print sconfigs_df.head()    

    # -------------------------------------------------------------------------
    # Create PSTH plot grid:
    # -------------------------------------------------------------------------
    # 1.  No rows or columns, i.e., SINGLE transform (e.g., orientation):
    if len(trans_types) == 1 and rows is None and columns is None:
        stim_grid = (transform_dict[trans_types[0]],)
        sgroups = sconfigs_df.groupby(sorted(trans_types))
        ncols = len(stim_grid[0])
        columns = trans_types[0]
        col_order = sorted(stim_grid[0])
        nrows = 1; rows = None; row_order=None
        unspecified_trans_types = []
    else:
        # 2.  If more than 1 transform, default is to make ROWS object/morphlevel:
        if rows is None:
            if 'morphlevel' in trans_types and subplot_hue != 'morphlevel':
                rows = 'morphlevel'
            elif 'ori' in trans_types:
                rows = [t for t in trans_types if t != 'ori'][0]
            else:
                rows = 'object'
        # And make the COLS whatever other transform there is:
        if rows == 'object':
            transform_dict['object'] = [i.value for i in sconfigs_df['object'].unique()]
        nonrow_trans_types = [t for t in trans_types if t not in t != rows and t != subplot_hue]

        if columns is None:
            columns = nonrow_trans_types[0]
        if columns == 'object':
            transform_dict['object'] = [i for i in sconfigs_df['object'].unique()]
        noncol_trans_types =  [t for t in trans_types if t not in [rows, columns, subplot_hue]]

        if subplot_hue is None and len(noncol_trans_types) > 1:
            subplot_hue = [t for t in noncol_trans_types if t != columns and t != rows][0]

        unspecified_trans_types = [t for t in trans_types if t not in [rows, columns, subplot_hue]]
        
        
        nrows = len(transform_dict[rows])
        ncols = len(transform_dict[columns])        
        stim_grid = (transform_dict[rows], transform_dict[columns])

        sgroups = sconfigs_df.groupby([rows, columns])
    
    if len(sgroups.groups) == 3:
        nrows = 1; ncols=3;

    print "N stimulus combinations:", len(stim_grid)
    if len(stim_grid)==1:
        grid_pairs = sorted([x for x in stim_grid[0]])
    else:
        grid_pairs = sorted(list(itertools.product(stim_grid[0], stim_grid[1])), key=lambda x: (x[0], x[1]))
    print grid_pairs
    #%
    # Set output dir for nice(r) psth:
    if plot_trials:
        plot_type = 'trials'
    else:
        plot_type = 'shade'
    psth_dir = os.path.join(traceid_dir, 'figures', 'psth_%s_%s%s' % (inputdata, plot_type, figdir_append))
    if filetype == 'pdf':
        psth_dir = '%s_hq' % psth_dir
        
    if not os.path.exists(psth_dir):
        os.makedirs(psth_dir)
    print "Saving PSTHs to: %s" % psth_dir
    
    if dfmax is None:
        dfmax = xdata.max()
    else:
        dfmax = float(dfmax)
    
    
    # Set COLORS for subplots:
    if len(trans_types)<=2 and subplot_hue is None:
        trace_colors = ['k']
        trace_labels = ['']
        
    elif len(trans_types) > 2 and len(unspecified_trans_types) > 0:
        # Use different color gradients for each object to plot the unspecified transform, too:
        # X and Y already specified, so color-gradient will be the unspecified.
        assert len(unspecified_trans_types) == 1, "More than 1 unspecified trans: %s" % str(unspecified_trans_types)
        colorbank = ['Purples', 'Greens', 'Blues', 'Reds']
        if subplot_hue in transform_dict.keys():
            hue_labels = transform_dict[subplot_hue]
        else:
            hue_labels = object_transformations[rows]
        
        unspec_trans_type = unspecified_trans_types[0]
        unspec_trans_values = sorted(transform_dict[unspec_trans_type])
        n_unspec_trans_levels = len(unspec_trans_values)
        
        trace_colors = dict((hue_base, dict((gvalue, sns.color_palette(colorbank[hi], n_unspec_trans_levels)[glevel]) \
                                            for glevel, gvalue in enumerate(unspec_trans_values)) ) \
                                            for hi, hue_base in enumerate(hue_labels))
        trace_labels = dict((hue_base, dict((translevel, '_'.join([str(hue_base), str(translevel)])) for translevel in unspec_trans_values) ) \
                                             for hue_base in hue_labels)
#        trace_colors = dict((hue_base, sns.color_palette(colorbank[hi], n_unspec_trans_levels)) for hi, hue_base in enumerate(hue_labels))
#        trace_labels = dict((hue_base, ['_'.join([str(hue_base), str(translevel)]) for translevel in unspec_trans_values]) for hue_base in hue_labels)
    
        
    else:
        print "Subplot hue: %s" % subplot_hue
        if subplot_hue in transform_dict.keys():
            hues = transform_dict[subplot_hue]
            trace_labels = ['%s %i' % (subplot_hue, v) for v in sorted(transform_dict[subplot_hue])]
        else:
            print "Hue is OBJECT"
            hues = object_transformations[rows]
            trace_labels = hues
            
        if len(hues) <= 2:
            trace_colors = ['g', 'b']
        else:
            trace_colors = sns.color_palette('hls', len(hues))

        

    print trace_labels #rows, object_transformations[rows] #trace_labels

    for ridx in range(xdata.shape[-1]):
        #%%
        if ridx % 20 == 0:
            print "Plotting %i of %i rois." % (ridx, xdata.shape[-1])
        roi_id = 'roi%05d' % int(ridx+1)

    
        rdata = labels_df.copy()
        rdata['data'] = xdata[:, ridx]
        
        stim_on = list(set(rdata['stim_on_frame']))[0]
        bas_grand_mean = np.mean([np.mean(vals[0:stim_on]) for vals in rdata.groupby('trial')['data'].apply(np.array)])
        
        #stim_on = run_info['stim_on_frame']
        #nframes_on = run_info['nframes_on']
        #tracemat = np.reshape(xdata[:, ridx], (ntrials_total, nframes_per_trial))
        
        sns.set_style('ticks')
        fig, axes = pl.subplots(nrows, ncols, sharex=False, sharey=True, figsize=(20,3*nrows+5))
        axesf = axes.flat    
        traces_list = []
        skipped_axes = []
        pi = 0
        
        #for k,g in sgroups:
        for k in grid_pairs:
            if k not in sgroups.groups.keys(): 
                axesf[pi].axis('off')
                skipped_axes.append(pi)
                pi += 1
                continue
            #print k
            g = sgroups.get_group(k)
            if subplot_hue is not None:
                curr_configs = g.sort_values(subplot_hue).index.tolist()
            else:
                curr_configs = sorted(g.index.tolist())
                
            for cf_idx, curr_config in enumerate(curr_configs):
                sub_df = rdata[rdata['config']==str(curr_config)]
                tracemat = np.vstack(sub_df.groupby('trial')['data'].apply(np.array))
                tpoints = np.mean(sub_df.groupby('trial')['tsec'].apply(np.array), axis=0)
                assert len(list(set(sub_df['nframes_on']))) == 1, "More than 1 stimdur parsed for current config..."
                
                nframes_on = list(set(sub_df['nframes_on']))[0]
                if correct_offset:
                    subdata = tracemat - bas_grand_mean
                else:
                    subdata = tracemat

                if plot_median:
                    trace_mean = np.median(subdata, axis=0)
                else:    
                    trace_mean = np.mean(subdata, axis=0) #- bas_grand_mean[ridx]
                trace_sem = stats.sem(subdata, axis=0) #stats.sem(subdata, axis=0)
                
                
                if isinstance(trace_colors, list):
                    curr_color = trace_colors[cf_idx]
                    curr_label = trace_labels[cf_idx]
                else:
                    # There is an additional axis (plotted as GRADIENT) -- unspecified_tra
                    # Identify which is the corresponding object for subplot_hue:
                    hue_key = sconfigs_df.loc[curr_config][subplot_hue]
                    unspec_key = sconfigs_df.loc[curr_config][unspec_trans_type]
                    curr_color = trace_colors[hue_key][unspec_key]
                    curr_label = trace_labels[hue_key][unspec_key]
                    
                    
                axesf[pi].plot(tpoints, trace_mean, color=curr_color, linewidth=2, 
                             label=curr_label, alpha=1.0)
                
                if plot_trials:
                    for ti in range(subdata.shape[0]):
                        if len(np.where(np.isnan(subdata[ti, :]))[0]) > 0:
                            print "-- NaN: Trial %i, %i" % (ti+1, len(np.where(np.isnan(subdata[ti, :]))[0]))
                            continue
                        axesf[pi].plot(tpoints, subdata[ti,:], color=curr_color, linewidth=0.5, alpha=0.5)
                else:
                    # fill between with sem:
                    axesf[pi].fill_between(tpoints, trace_mean-trace_sem, trace_mean+trace_sem, color=curr_color, alpha=0.5)
                
                
                
                # Set x-axis to only show stimulus ON bar:
                start_val = tpoints[stim_on]
                end_val = tpoints[stim_on + int(round(nframes_on))]
                if pi==len(axes)-1:
                    axesf[pi].set_xticks((start_val, 1.0)) 
                else:
                    axesf[pi].set_xticks((0, 0))
                axesf[pi].set_xticklabels(())
                axesf[pi].tick_params(axis='x', which='both',length=0)
                if isinstance(k, int):
                    axesf[pi].set_title('(%.1f)' % (k), fontsize=10)
                else:
                    if isinstance(k[0], (int, float)) or k[0].isdigit():
                        k0 = float(k[0])
                        k0_str = '%.1f' % k0
                    else:
                        k0_str = str(k[0])
                    if isinstance(k[1], (int, float)) or k[1].isdigit():
                        k1 = float(k[1])
                        k1_str = '%.1f' % k1
                    else:
                        k1_str = str(k[1])
                    axesf[pi].set_title('(%s, %s)' % (k0_str, k1_str), fontsize=10)


            # Set y-axis to be the same, if specified:
            if scale_y:
                axesf[pi].set_ylim([0, dfmax])
            if 'df' in inputdata:
                axesf[pi].set_yticks((0, 1))
            sns.despine(offset=4, trim=True, ax=axesf[pi])
          
            # Add annotation for n trials in stim config:    
            axesf[pi].text(-0.8, axesf[pi].get_ylim()[-1]*0.8, 'n=%i' % subdata.shape[0])   

            
            #pl.legend(loc=9, bbox_to_anchor=(-0.5, -0.1), ncol=len(trace_labels))

            if pi==0:
                axesf[pi].set_ylabel(ylabel)
            pi += 1

        #sns.despine(offset=4, trim=True)
        #loop over the non-left axes:
        for ai,ax in enumerate(axes.flat):
            if ai in skipped_axes:
                continue
            ymin = min([ax.get_ylim()[0], ax.get_yticks()[0]])
            ymax = max([ax.get_ylim()[-1], ax.get_yticks()[-1]])
            stimpatch = patches.Rectangle((start_val, ymin), end_val, ymax, linewidth=0, fill=True, color='k', alpha=0.2)
            ax.add_patch(stimpatch)
            if 'df' in inputdata:
                ax.set_yticks((0, 1))
            sns.despine(offset=4, trim=True, ax=ax)


#        for ax in axes.flat[1:]:
#            # get the yticklabels from the axis and set visibility to False
#            for label in ax.get_yticklabels():
#                label.set_visible(False)
#            ax.yaxis.offsetText.set_visible(False)
#            ax.yaxis.set_visible(False)
#                
    
        pl.subplots_adjust(bottom=0.12, top=0.95)
        pl.suptitle("%s" % (roi_id))
        pl.legend(loc=9, bbox_to_anchor=(0, 0), ncol=len(trace_labels))
        label_figure(fig, data_identifier)
        
        #%%
        figname = '%s_psth_%s.%s' % (roi_id, inputdata, filetype)
        pl.savefig(os.path.join(psth_dir, figname))
        pl.close()
    
    return psth_dir


#%%
#
#data_array_dir = os.path.join(traceid_dir, 'data_arrays')
#
#rdata_path = os.path.join(data_array_dir, 'ROIDATA.pkl')
#with open(rdata_path, 'rb') as f:
#    DATA = pkl.load(f)
#
## Group by MORPH:
#class_key = 'config'
#first_on = DATA['first_on'].unique()[0]
#nframes_on = DATA['nframes_on'].unique()[0]
#frame_ixs = [int(i) for i in np.arange(first_on, first_on+nframes_on)]
#classes = sorted(DATA[class_key].unique())
#grps = DATA.groupby(['roi', 'trial', class_key])['dff']
#
#
#means = pd.DataFrame({'roi': [k[0] for k,g in grps],
#              'morphlevel': [k[-1] for k,g in grps],
#              'meandff': [np.mean(g.values[frame_ixs]) for k,g in grps]})
#
#morphs = means.groupby(['morphlevel', 'roi']).mean()
#G = morphs.groupby(level='morphlevel').apply(np.array)
#groups = G.index.tolist()
#
##fig, axes = pl.subplots(1, len(groups))
##for ax,group in zip(axes, groups):
##    sns.distplot(G[group], ax=ax)
#
#
#popn_sparse = []; frac_responsive= []; all_values= [];
#for group in groups:
#    roi_values = np.array([float(i)-v for i,v in zip(G[group], bas_grand_mean)])
#    #roi_values = np.array([float(i) for i in G[group]])
#
#    nrois = float(len(roi_values))
#
##    num = G[group].sum()**2
##    denom = sum(G[group]**2)
##    s = ( 1. - (1./nrois)*(num/denom) ) / (1. - (1./nrois))
#    num = sum( [ float(i)/nrois for i in G[group] ]) **2
#    denom = sum( (v**2 / nrois) for v in G[group] )
#    s = (nrois / (nrois - 1.)) * (1. - (num/denom))
#
#    nresp = len(np.where(roi_values >= 0.05)[0])
#    popn_sparse.append(float(s))
#    frac_responsive.append(float(nresp)/nrois)
#    all_values.append(roi_values)
#    
#all_values = np.array(all_values)
#
#
## Look at max response value for ROIs:
#max_resp_by_roi = np.max(all_values, axis=0)
#pl.figure()
#sns.distplot(max_resp_by_roi, hist=True, kde=False)
#pl.title('response')
#pl.xlabel('df/f')
#pl.ylabel('cell counts')
#
## Compare frac. reposnsive and popN sparseness index:
#pl.figure()
#pl.scatter(popn_sparse, frac_responsive)
#pl.xlabel('population sparseness')
#pl.ylabel('fraction of cells > 10% df/f')
#
#
#curr_trans = 'xpos'
#
#mlevels = sorted(list(set(DATA['morphlevel'])))
#popn_sparse = np.array(popn_sparse)
#
## Plot sparseness index for each combination fo blobs and view:
#morph_ixs = []
#for m in mlevels:
#    cfigs = sorted([c for c,v in sconfigs.items() if v['morphlevel']==m], key=lambda x: sconfigs[x][curr_trans])
#    m_ixs = np.array(sorted([ci for ci,cv in enumerate(groups) if cv in cfigs], key=lambda x: cfigs))
#    morph_ixs.append(m_ixs)
#    
##morph_ixs = [np.array([ci for ci,cv in enumerate(groups) if sconfigs[cv]['morphlevel']==m]) for m in sorted(mlevels)]
#yrots = sorted(list(set(DATA[curr_trans])))
#
#fig, axes = pl.subplots(1, len(mlevels), sharey=True, sharex=True)
#for m,ax in zip(range(len(morph_ixs)), axes.flat):
#    currvals = popn_sparse[morph_ixs[m]]
##    hist, edges = np.histogram(currvals, bins)
##    ax.bar(bins[:-1], hist, width=.001, align='edge', ec='k')
#    ax.plot(range(len(currvals)), list(currvals), 'bo', markersize=5)
#    ax.set_title(mlevels[m])
#    ax.set_ylabel('S')
#    locs, labels = pl.xticks()           # Get locations and labels
#    pl.xticks(range(len(currvals)), yrots)
#
#    
## Plot frac responsive for each combination fo blobs and view:
#
#frac_responsive = np.array(frac_responsive)
#fig, axes = pl.subplots(1, len(mlevels), sharey=True, sharex=True)
#for m,ax in zip(range(len(morph_ixs)), axes.flat):
#    currvals = frac_responsive[morph_ixs[m]]
##    hist, edges = np.histogram(currvals, bins)
##    ax.bar(bins[:-1], hist, width=.001, align='edge', ec='k')
#    ax.plot(range(len(currvals)), list(currvals), 'bo', markersize=5)
#    ax.set_title(mlevels[m])
#    ax.set_ylabel('fraction of cells')
#    locs, labels = pl.xticks()           # Get locations and labels
#
#    pl.xticks(range(len(currvals)), yrots)
#
#pl.figure()
#sns.distplot(popn_sparse, norm_hist=True)
#
##%%
#
#ridx = 114
#vmax = 0.2
#roi_id = 'roi%05d' % int(ridx+1)
#print roi_id
#
#rdata = DATA[DATA['roi']==roi_id]
#grped = rdata.groupby(['morphlevel', curr_trans])['dff']
#
#ids = []; dffs=[];
#for k,g in grped:
#    vals = g - bas_grand_mean[ridx]
#    ntrials = len(vals) / nframes_per_trial
#    tmat = np.reshape(vals, (ntrials, nframes_per_trial))
#    stimvals = tmat[:, stim_on:stim_on+int(round(nframes_on))]
#    meanval = np.mean(np.mean(stimvals, axis=1))
#    ids.append(k)
#    dffs.append(meanval)
#    
##t = range(len(ids))
##np.reshape(t, (5,5))
#
#curr_trans_vals = sorted(list(set(DATA[curr_trans])))
#val_array = np.reshape(np.array(dffs), (5,5))
#xlabels = curr_trans_vals
#ylabels = mlevels
#
#pl.figure()
#g = sns.heatmap(val_array, vmax=vmax,cmap='hot')
#g.set_yticklabels(ylabels)
#g.set_xticklabels(xlabels)
#
#
#    
#    
#
#
#
#
#As = np.array([ci for ci,cv in enumerate(groups) if sconfigs[cv]['morphlevel']==0])
#Bs = np.array([ci for ci,cv in enumerate(groups) if sconfigs[cv]['morphlevel']==22])
#
#popn_sparse = []
#object_ids = sorted(list(set([sconfigs[c]['morphlevel'] for c in sconfigs.keys()])))
#fig, axes = pl.subplots(1, len(object_ids))
#for ax,obj in zip(axes.flat, object_ids):
#    As = np.array([cv for ci,cv in enumerate(groups) if sconfigs[cv]['morphlevel']==obj])
#    
#    mean_across_views = np.mean(np.hstack(G[As].values), axis=1)
#    # Calc sparseness:
#    num = sum( [ float(i)/nrois for i in mean_across_views ]) **2
#    denom = sum( (v**2 / nrois) for v in mean_across_views )
#    s = (nrois / (nrois - 1.)) * (1. - (num/denom))
#    
#    popn_sparse.append(s)
#
#fig, ax = pl.subplots(1)
#ax.plot(popn_sparse)
#sns.despine()
#
#pl.scatter(popn_sparse[As], popn_sparse[Bs])


#        t1a = object_resp_df.mean()**2.
#        t1b = sum([ (object_resp_df[i]**2) for i in object_list]) * nobjects
#        #t1b = sum([ (object_resp_df[i]**2)/nobjects for i in object_list])
#        a = t1a/t1b
#        S = (1. - a) / (1. - (1./nobjects) )
        
#%%
    #%
def main(options):
    
    psth_dir = make_clean_psths(options)
    
    print "*******************************************************************"
    print "DONE!"
    print "All output saved to: %s" % psth_dir
    print "*******************************************************************"
    
    
#%%

if __name__ == '__main__':
    main(sys.argv[1:])
