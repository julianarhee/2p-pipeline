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

def extract_options(options):

    parser = optparse.OptionParser()

    parser.add_option('-D', '--root', action='store', dest='rootdir',
                          default='/n/coxfs01/2p-data',
                          help='data root dir (dir w/ all animalids) [default: /nas/volume1/2photon/data, /n/coxfs01/2pdata if --slurm]')
    parser.add_option('--slurm', action='store_true', dest='slurm', default=False, help="set if running as SLURM job on Odyssey")

    parser.add_option('-i', '--animalid', action='store', dest='animalid',
                          default='', help='Animal ID')
    parser.add_option('--auto', action='store_true', dest='auto',
                          default=False, help='Set to always use auto options')
    parser.add_option('--compare', action='store_true', dest='compare_runs',
                          default=False, help='Set to compare 2 runs (assumes fixed stim params for grid pos, uses HUE as comparison bw runs')
    parser.add_option('-p', action='store', dest='compare_param',
                          default=None, help='Stimulus param to compare bw runs (e.g., backlight)')

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

#

#options = ['-D', '/n/coxfs01/2p-data','-i', 'JC078', '-S', '20190426', '-A', 'FOV1_zoom2p0x',
#           '-R', 'combined_blobs_static', '-t', 'traces001', '--shade',
#           '--compare','-p', 'color',
#           '-r', 'size', '-c', 'morphlevel']

#options = ['-D', '/n/coxfs01/2p-data','-i', 'JC078', '-S', '20190427', '-A', 'FOV1_zoom2p0x',
#           '-R', 'combined_gratings_static', '-t', 'traces001', '--shade',
#           '-r', 'ypos', '-c', 'xpos']

options = ['-D', '/n/coxfs01/2p-data','-i', 'JC086', '-S', '20190515', '-A', 'FOV1_zoom2p0x',
           '-R', 'gratings_run2', '-t', 'traces001', '--shade',
           '-r', 'size', '-c', 'ori', '-H', 'sf']

#%%
def get_data_id_from_tracedir(traceid_dir, rootdir='/n/coxfs01/2p-data/'):
    trace_info = traceid_dir.split(rootdir)[-1]
    data_identifier = trace_info.replace('/', '|')
    return data_identifier
    
def load_traces_and_configs(traceid_dir, inputdata='corrected'):
    data_fpath = os.path.join(traceid_dir, 'data_arrays', 'datasets.npz')
    print "Loaded data from: %s" % data_fpath #traceid_dir
    dataset = np.load(data_fpath)
    assert inputdata in dataset.keys(), "Specified data type (%s) not found! Choose from: %s" % (inputdata, str(dataset.keys()))
    xdata = pd.DataFrame(dataset[inputdata])
    labels = pd.DataFrame(data=dataset['labels_data'], columns=dataset['labels_columns'])
    sdf = pd.DataFrame(dataset['sconfigs'][()]).T
    return xdata, labels, sdf
                          
      #%%                    
def make_clean_psths(options):
    #%%
    optsE = extract_options(options)
    
    #traceid_dir = util.get_traceid_dir(options)
    run = optsE.run #run_list[0]
    traceid = optsE.traceid #traceid_list[0]
    acquisition_dir = os.path.join(optsE.rootdir, optsE.animalid, optsE.session, optsE.acquisition)
    if 'cnmf' in traceid:
        traceid_dirs = glob.glob(os.path.join(acquisition_dir, run, 'cnmf', '%s*' % traceid))[0]
    else:
        traceid_dirs = [t for t in glob.glob(os.path.join(acquisition_dir, run, 'traces', '%s*' % traceid)) if 'ORIG' not in t] #[0]
    
    compare_runs = optsE.compare_runs
    auto  = optsE.auto
    if len(traceid_dirs) > 1 and compare_runs is False:
        print "More than 1 traceid found:"
        for ti, traceid in enumerate(traceid_dirs):
            print ti, traceid
        if auto is False:
            sel = input("Select IDX of traceid to use: ")
            traceid_dirs = [traceid_dirs[int(sel)]]
        else:
            traceid_dirs = [traceid_dirs[0]  ]          


    #%%   
    subplot_hue = optsE.subplot_hue
    rows = optsE.rows
    columns = optsE.columns
    compare_param = optsE.compare_param
    if compare_runs:
        subplot_hue = compare_param
    plot_params = {'hue': subplot_hue,
                   'rows': rows,
                   'cols': columns}
    
    if compare_param == 'color':
        compare_param_name = 'backlight'
    else:
        compare_param_name = compare_param
    
    #if compare_runs:
    xdata_list=[]; labels_list=[]; sdf_list=[]; data_ids=[];
    for traceid_dir in traceid_dirs:
        xdata, labels, sdf = load_traces_and_configs(traceid_dir, inputdata=optsE.datatype)
        data_identifier = get_data_id_from_tracedir(traceid_dir)
        xdata_list.append(xdata)
        labels_list.append(labels)
        sdf_list.append(sdf)
        data_ids.append(data_identifier)
    
    data_id_list = data_ids[0].split('|')
    if len(data_ids) > 1:
        data_id_list.extend([s for s in data_ids[1].split('|') if s not in data_id_list])
    data_identifier = '|'.join(data_id_list)
    print "*** %s" % data_identifier
    
    for si, sdf in enumerate(sdf_list):
        if compare_param_name == 'backlight' and compare_runs:
            if round(sdf['color'].min(), 2) == 0.06: 
                compare_condn = 50
            elif round(sdf['color'].min(), 2) == 0.08:
                compare_condn = 100
            print compare_condn
            sdf[compare_param_name]  = [compare_condn for _ in np.arange(sdf.shape[0])]
    
        # adjust "sconfigs" to deal with funky controls:
        # 20190422:  this is assigning fake morph value to -1, which is really a control stimulus
        if 'morphlevel' in plot_params.values() and 'control' in sdf['object'].unique():
            sizes = sorted([s for s in sdf['size'].unique() if s is not None])
            lums = sorted([s for s in sdf['color'].unique() if s not in [None, '']])
            print "---> Assigning luminance as psuedoe-size for CONTROL stim"
            lum_lut = dict((lm, sz) for lm, sz in zip(lums, sizes))
            for cfg in sdf.index.tolist():
                if sdf['object'][cfg] == 'control':
                    sdf['size'][cfg] = lum_lut[sdf['color'][cfg]]
        
        last_cfg_n = int(sdf.index.tolist()[-1][7:])
        if si > 0:
            sdf.index = ['config%03d' % int(last_cfg_n + int(c[7:])) for c in sdf.index.tolist()]
            tmp_labels = labels_list[si]
            tmp_labels['config'] = ['config%03d' % int(last_cfg_n + int(c[7:])) for c in tmp_labels['config']]
            labels_list[si] = tmp_labels
        
        sdf_list[si] = sdf

#%% 
    # Combine data:
    sdf_c = pd.concat(sdf_list, axis=0) # Get rid of config labels
    xdata_c = pd.concat(xdata_list, axis=0).reset_index(drop=True)
    labels_c = pd.concat(labels_list, axis=0).reset_index(drop=True) # Get rid of config labels        
    stim_on = labels_c['stim_on_frame'].unique()[0]
    nframes_on = labels_c['nframes_on'].unique()[0]
    
    for pparam in plot_params.values():
        if pparam not in labels_c.columns and pparam is not None:
            #print pparam
            labels_c[pparam] = [sdf_c[pparam][cfg] for cfg in labels_c['config']]
    if 'size' in labels_c.columns:
        labels_c = labels_c.round({'size': 0}).astype({'size': int})
        sdf_c = sdf_c.round({'size': 0})

#%%
    if compare_runs:
        output_figdir = os.path.join(traceid_dirs[0].split('/traces/')[0], 'compare_runs', 'figures')
    else:
        output_figdir = os.path.join(traceid_dir, 'figures')
    if not os.path.exists(output_figdir):
        os.makedirs(output_figdir)
    print "OUTPUT saved to:", output_figdir    
    
    #%%
    

    # Plotting options:
    filetype = optsE.filetype
    dfmax = optsE.dfmax
    scale_y = optsE.scale_y
    
    transform_dict, object_transformations = util.get_transforms(sdf_c.T.to_dict())
   
    # replace duration:
    if 'duration' in transform_dict.keys():
        transform_dict['stim_dur'] = transform_dict['duration']
        transform_dict.pop('duration')

            
    if 'position' in plot_params.values() and 'position' not in sdf_c.columns.tolist():
        posvals = list(set(zip(sdf_c['xpos'].values, sdf_c['ypos'].values)))
        print "Found %i unique positions." % len(posvals)
        transform_dict['position'] = posvals
        sdf['position'] = list(zip(sdf_c['xpos'], sdf_c['ypos']))


    trans_types = sorted([trans for trans in transform_dict.keys() if len(transform_dict[trans]) > 1])         
    print "Trans:", trans_types
    #print transform_dict
    print sdf_c.head()    
   
   
    # -------------------------------------------------------------------------
    # Create PSTH plot grid:
    # -------------------------------------------------------------------------
    # 1.  No rows or columns, i.e., SINGLE transform (e.g., orientation):
    if len(trans_types) == 1 and rows is None and columns is None:
        stim_grid = (transform_dict[trans_types[0]],)
        sgroups = sdf.groupby(sorted(trans_types))
        ncols = len(stim_grid[0])
        columns = trans_types[0]
        col_order = sorted(stim_grid[0])
        nrows = 1; rows = None; row_order=None
        unspecified_trans_types = []
    elif rows is None or columns is None:
        assert len(trans_types) > 1, "No transforms found!  what to plot??"
        if rows is None and cols is None:
            rows = trans_types[0] 
            cols = trans_types[1]
        elif rows is None:
            rows = [t for t in trans_types if t != cols]
        elif cols is None:
            cols = [t for t in trans_types if t != rows]
        if subplot_hue is None:
            subplot_hue = None if len(trans_types)==2 else trans_types[2]
    else:
        nrows = len(transform_dict[rows])
        ncols = len(transform_dict[columns])        
        stim_grid = (transform_dict[rows], transform_dict[columns])
        sgroups = sdf.groupby([rows, columns])

    
    # Update plot_params:    
    plot_params = {'hue': subplot_hue,
                   'rows': rows,
                   'cols': columns}
        
    unspecified_trans_types = [t for t in trans_types if t not in plot_params.values()]
    
    if len(sgroups.groups) == 3:
        nrows = 1; ncols=3;

    print "N stimulus combinations:", len(stim_grid)
    if len(stim_grid)==1:
        grid_pairs = sorted([x for x in stim_grid[0]])
    else:
        grid_pairs = sorted(list(itertools.product(stim_grid[0], stim_grid[1])), key=lambda x: (x[0], x[1]))
    #print grid_pairs

    
    row_vals = None if plot_params['rows'] is None else sorted(sdf_c[plot_params['rows']].unique())
    col_vals = None if plot_params['cols'] is None else sorted(sdf_c[plot_params['cols']].unique())
    hue_vals = None if plot_params['hue'] is None else sorted(sdf_c[plot_params['hue']].unique())


    #%%

    if optsE.datatype in ['corrected', 'smoothedX']:
        ylabel = 'intensity'
    elif optsE.datatype in ['dff', 'smoothedDF']:
        ylabel = 'df/f'
    elif optsE.datatype == 'spikes':
        ylabel = 'inferred'

    #%
    # Filter out noisy spikes:
    figdir_append = ''
    if optsE.datatype == 'spikes' and optsE.filter_noise:
        xdata[xdata<=0.0004] = 0.
        figdir_append = '_filtered'
    if optsE.plot_median:
        figdir_append = '%s_median' % figdir_append 
    
    
    # Set output dir for nice(r) psth:
    if optsE.plot_trials:
        plot_type = 'trials'
    else:
        plot_type = 'shade'
    psth_dir = os.path.join(output_figdir, 'psth_%s_%s%s' % (optsE.datatype, plot_type, figdir_append))
    if optsE.filetype == 'pdf':
        psth_dir = '%s_hq' % psth_dir
        
    if not os.path.exists(psth_dir):
        os.makedirs(psth_dir)
    print "Saving PSTHs to: %s" % psth_dir
    
    if optsE.dfmax is None:
        dfmax = xdata.max()
    else:
        dfmax = float(dfmax)
    
    #%%
    # Set COLORS for subplots:
    print "Trans Types:", trans_types
    if len(trans_types)<=2 and plot_params['hue'] is None:
        print "PLOTTING 1 color"
        trace_colors = ['k']
        trace_labels = ['']
        
    elif len(trans_types) > 2 and len(unspecified_trans_types) > 0:
        # Use different color gradients for each object to plot the unspecified transform, too:
        # X and Y already specified, so color-gradient will be the unspecified.
        if len(unspecified_trans_types) > 1:
            print "--- WARNING --- More than 1 unspecified trans: %s" % str(unspecified_trans_types)
        colorbank = ['Purples', 'Greens', 'Blues', 'Reds']
        if plot_params['hue'] in sdf_c.columns:
            hue_labels = sorted(sdf_c[plot_params['hue']].unique())
        else:
            hue_labels = object_transformations[rows]
        
        unspec_trans_type = unspecified_trans_types[0]
        unspec_trans_values = sorted(sdf_c[unspec_trans_type].unique())
        n_unspec_trans_levels = len(unspec_trans_values)
        
        trace_colors = dict((hue_base, dict((gvalue, sns.color_palette(colorbank[hi], n_unspec_trans_levels)[glevel]) \
                                            for glevel, gvalue in enumerate(unspec_trans_values)) ) \
                                            for hi, hue_base in enumerate(hue_labels))
        trace_labels = dict((hue_base, dict((translevel, '_'.join([str(hue_base), str(translevel)])) for translevel in unspec_trans_values) ) \
                                             for hue_base in hue_labels)
        
    else:
        print "Subplot hue: %s" % plot_params['hue']
        if plot_params['hue'] in sdf_c.columns:
            hues = sorted(sdf_c[plot_params['hue']].unique())
            trace_labels = ['%s %s' % (plot_params['hue'], str(v)) for v in sorted(sdf_c[plot_params['hue']].unique())]
        else:
            hues = object_transformations[rows]
            trace_labels = hues
            
        if len(hues) <= 2:
            trace_colors = ['g', 'b']
        else:
            trace_colors = sns.color_palette('hls', len(hues))

        
#%%
    print trace_labels #rows, object_transformations[rows] #trace_labels

    for ridx in range(xdata.shape[-1]):
        #%%
        if ridx % 20 == 0:
            print "Plotting %i of %i rois." % (ridx, xdata.shape[-1])
        roi_id = 'roi%05d' % int(ridx+1)

        sns.set_style('ticks')
        
        rdata = labels_c.copy()
        if 'size' in rdata.columns:
            rdata = rdata.round({'size': 0})
            rdata['size'] = rdata.astype({'size': int})
        rdata[ylabel] = xdata_c[ridx]
                        
        if optsE.plot_trials:
            p = sns.FacetGrid(rdata, col=plot_params['cols'], row=plot_params['rows'], hue=plot_params['hue'], sharex=True, sharey=True)
            p.map(pl.plot, "tsec", ylabel, lw=0.5, alpha=0.5)
        else:
            for k, g in rdata.groupby(['config']):
                mean_trace = np.array(g.groupby(['trial'])[ylabel].groups.values()).mean(axis=0) # Get mean trace across trials
                mean_tsec = np.array(g.groupby(['trial'])['tsec'].groups.values()).mean(axis=0)
                sem_trace = stats.sem(np.array(g.groupby(['trial'])[ylabel].groups.values()), axis=0)
                
            meandfs = []
            for k, g in rdata.groupby(['config']):
                nreps = len(g['trial'].unique())
                mean_trace = g.groupby(['trial'])[ylabel].apply(np.array).mean(axis=0) # Get mean trace across trials
                mean_tsec = g.groupby(['trial'])['tsec'].apply(np.array).mean(axis=0)
                sem_trace = stats.sem(np.vstack(g.groupby(['trial'])[ylabel].apply(np.array)), axis=0)
                mdf = pd.DataFrame({'%s' % ylabel: mean_trace,
                                      'tsec': mean_tsec,
                                      'sem': sem_trace,
                                      'fill_minus': mean_trace - sem_trace,
                                      'fill_plus': mean_trace + sem_trace,
                                      'config': [k for _ in range(len(mean_trace))],
                                      'nreps': [nreps for _ in range(len(mean_trace))]
                                      })
                for p in plot_params.values():
                    if p not in sdf_c.columns:
                        continue
                    mdf[p] = [round(sdf_c[p][cfg]) for cfg in mdf['config']]
                meandfs.append(mdf)
            meandfs = pd.concat(meandfs, axis=0)
        
    #        ylim = meandfs['data'].max()
    #        meandfs['annot_x'] = [-0.999 for _ in range(meandfs.shape[0])]
    #        meandfs['annot_y'] = [ylim*0.9 for _ in range(meandfs.shape[0])]
    #        meandfs['annot_str'] = ['n=%i' % i for i in meandfs['nreps']]
            p = sns.FacetGrid(meandfs, col=plot_params['cols'], row=plot_params['rows'], hue=plot_params['hue'], size=1)
            if len(meandfs[plot_params['rows']].unique()) == 1:
                p.fig.set_figheight(3)
                p.fig.set_figwidth(20)
            if plot_params['hue'] is None:
                p = p.map(pl.fill_between, "tsec", "fill_minus", "fill_plus", alpha=0.5, color='k')
                p = p.map(pl.plot, "tsec", ylabel, lw=1, alpha=1, color='k')
            else:
                p = p.map(pl.fill_between, "tsec", "fill_minus", "fill_plus", alpha=0.5)
                p = p.map(pl.plot, "tsec", ylabel, lw=1, alpha=1)
            p = p.set_titles(col_template="{col_name}", size=6)
#        for xi in range(p.axes.shape[0]):
#            for yi in range(p.axes.shape[1]):
#                p.axes[xi, yi].text(-0.999, ylim*0.9, 'n=%i' % nreps)
           
            if 'xpos' in plot_params.values() and 'ypos' in plot_params.values(): 
                pl.subplots_adjust(wspace=0.05, hspace=0.3, top=0.85, bottom=0.1, left=0.05)
            else:
                pl.subplots_adjust(wspace=0.8, hspace=0.8, top=0.85, bottom=0.1, left=0.1)
            
        ymin = meandfs[ylabel].min()
        ymax = meandfs[ylabel].max()
        start_val = 0.0
        end_val = mdf['tsec'][stim_on + int(round(nframes_on))]
        for ri in range(p.axes.shape[0]):
            for ci in range(p.axes.shape[1]):
                #print ri, ci
                p.axes[ri, ci].add_patch(patches.Rectangle((start_val, ymin), end_val, ymax, linewidth=0, fill=True, color='k', alpha=0.2))
                p.axes[ri, ci].text(-0.999, ymax+(ymax*0.2), 'n=%i' % nreps, fontsize=6)
                if len(col_vals) > 10:
                    p.axes[ri, ci].set_title('')
                    
                if ri == 0 and ci == 0:
                    p.axes[ri, ci].yaxis.set_major_locator(pl.MaxNLocator(2))
                    p.axes[ri, ci].set_xticks(())
                    sns.despine(trim=True, offset=4, bottom=True, left=False, ax=p.axes[ri, ci])
                    p.axes[ri, ci].set_xlabel('time (s)', fontsize=8)
                    p.axes[ri, ci].set_ylabel('%s' % ylabel, fontsize=8)
                else:
                    sns.despine(trim=True, offset=4, bottom=True, left=True, ax=p.axes[ri, ci])
                    p.axes[ri, ci].tick_params(
                                            axis='both',          # changes apply to the x-axis
                                            which='both',      # both major and minor ticks are affected
                                            bottom='off',      # ticks along the bottom edge are off
                                            left='off',
                                            top='off',         # ticks along the top edge are off
                                            labelbottom='off',
                                            labelleft='off') # labels along the bottom edge are off)
                    p.axes[ri, ci].set_xlabel('')
                    p.axes[ri, ci].set_ylabel('')

        pl.legend(bbox_to_anchor=(0, -0.0), loc=2, borderaxespad=0.1, labels=trace_labels, fontsize=8)

        label_figure(p.fig, data_identifier)
        
        p.fig.suptitle('roi %i' % (int(ridx+1)))
        
        figname = '%s_psth_%s.%s' % (roi_id, optsE.datatype, filetype)
        p.savefig(os.path.join(psth_dir, figname))
        pl.close()
        
        #%%
#        #%%
#        traces_list = []
#        skipped_axes = []
#        pi = 0
#        
#        #for k,g in sgroups:
#        for k, g in sgroups: #in grid_pairs:
##            if k not in sgroups.groups.keys(): 
##                axesf[pi].axis('off')
##                skipped_axes.append(pi)
##                pi += 1
##                continue
#            #print k
#            #g = sgroups.get_group(k)
#            if subplot_hue is not None:
#                curr_configs = g.sort_values(subplot_hue).index.tolist()
#            else:
#                curr_configs = sorted(g.index.tolist())
#                
#            for cf_idx, curr_config in enumerate(curr_configs):
#                sub_df = rdata[rdata['config']==str(curr_config)]
#                tracemat = np.vstack(sub_df.groupby('trial')['data'].apply(np.array))
#                tpoints = np.mean(sub_df.groupby('trial')['tsec'].apply(np.array), axis=0)
#                assert len(list(set(sub_df['nframes_on']))) == 1, "More than 1 stimdur parsed for current config..."
#                
#                nframes_on = list(set(sub_df['nframes_on']))[0]
#                if optsE.correct_offset:
#                    bas_grand_mean = np.mean([np.mean(vals[0:stim_on]) for vals in rdata.groupby('trial')['data'].apply(np.array)])
#                    subdata = tracemat - bas_grand_mean
#                else:
#                    subdata = tracemat
#
#                if optsE.plot_median:
#                    trace_mean = np.median(subdata, axis=0)
#                else:    
#                    trace_mean = np.mean(subdata, axis=0) #- bas_grand_mean[ridx]
#                trace_sem = stats.sem(subdata, axis=0) #stats.sem(subdata, axis=0)
#                
#                
#                if isinstance(trace_colors, list):
#                    curr_color = trace_colors[cf_idx]
#                    curr_label = trace_labels[cf_idx]
#                else:
#                    # There is an additional axis (plotted as GRADIENT) -- unspecified_tra
#                    # Identify which is the corresponding object for subplot_hue:
#                    hue_key = sdf.loc[curr_config][subplot_hue]
#                    unspec_key = sdf.loc[curr_config][unspec_trans_type]
#                    curr_color = trace_colors[hue_key][unspec_key]
#                    curr_label = trace_labels[hue_key][unspec_key]
#                    
#                    
#                axesf[pi].plot(tpoints, trace_mean, color=curr_color, linewidth=2, 
#                             label=curr_label, alpha=1.0)
#                
#                if optsE.plot_trials:
#                    for ti in range(subdata.shape[0]):
#                        if len(np.where(np.isnan(subdata[ti, :]))[0]) > 0:
#                            print "-- NaN: Trial %i, %i" % (ti+1, len(np.where(np.isnan(subdata[ti, :]))[0]))
#                            continue
#                        axesf[pi].plot(tpoints, subdata[ti,:], color=curr_color, linewidth=0.2, alpha=0.5)
#                else:
#                    # fill between with sem:
#                    axesf[pi].fill_between(tpoints, trace_mean-trace_sem, trace_mean+trace_sem, color=curr_color, alpha=0.5)
#                
#                
#                
#                # Set x-axis to only show stimulus ON bar:
#                start_val = tpoints[stim_on]
#                end_val = tpoints[stim_on + int(round(nframes_on))]
#                if pi==len(axes)-1:
#                    axesf[pi].set_xticks((start_val, 1.0)) 
#                else:
#                    axesf[pi].set_xticks((0, 0))
#                axesf[pi].set_xticklabels(())
#                axesf[pi].tick_params(axis='x', which='both',length=0)
#                if isinstance(k, int):
#                    axesf[pi].set_title('(%.1f)' % (k), fontsize=10)
#                else:
#                    if isinstance(k[0], (int, float)) or k[0].isdigit():
#                        k0 = float(k[0])
#                        k0_str = '%.1f' % k0
#                    else:
#                        k0_str = str(k[0])
#                    if isinstance(k[1], (int, float)) or k[1].isdigit():
#                        k1 = float(k[1])
#                        k1_str = '%.1f' % k1
#                    else:
#                        k1_str = str(k[1])
#                    axesf[pi].set_title('(%s, %s)' % (k0_str, k1_str), fontsize=10)
#
#
#            # Set y-axis to be the same, if specified:
#            if scale_y:
#                axesf[pi].set_ylim([0, dfmax])
#            if 'df' in optsE.inputdata:
#                axesf[pi].set_yticks((0, 1))
#            sns.despine(offset=4, trim=True, ax=axesf[pi])
#          
#            # Add annotation for n trials in stim config:    
#            axesf[pi].text(-0.999, axesf[pi].get_ylim()[-1]*0.9, 'n=%i' % subdata.shape[0])   
#
#            
#            #pl.legend(loc=9, bbox_to_anchor=(-0.5, -0.1), ncol=len(trace_labels))
#
#            if pi==0:
#                axesf[pi].set_ylabel(ylabel)
#            pi += 1
#
#        #sns.despine(offset=4, trim=True)
#        #loop over the non-left axes:
#        for ai,ax in enumerate(axes.flat):
#            if ai in skipped_axes:
#                continue
#            ymin = min([ax.get_ylim()[0], ax.get_yticks()[0]])
#            ymax = max([ax.get_ylim()[-1], ax.get_yticks()[-1]])
#            stimpatch = patches.Rectangle((start_val, ymin), end_val, ymax, linewidth=0, fill=True, color='k', alpha=0.2)
#            ax.add_patch(stimpatch)
#            if 'df' in inputdata:
#                ax.set_yticks((0, 1))
#            sns.despine(offset=4, trim=True, ax=ax)
#
#
##        for ax in axes.flat[1:]:
##            # get the yticklabels from the axis and set visibility to False
##            for label in ax.get_yticklabels():
##                label.set_visible(False)
##            ax.yaxis.offsetText.set_visible(False)
##            ax.yaxis.set_visible(False)
##                
#    
#        pl.subplots_adjust(bottom=0.12, top=0.95)
#        pl.suptitle("%s" % (roi_id))
#        pl.legend(loc=9, bbox_to_anchor=(0, 0), ncol=len(trace_labels))
#        label_figure(fig, data_identifier)
#        
#        #%%
#        figname = '%s_psth_%s.%s' % (roi_id, inputdata, filetype)
#        pl.savefig(os.path.join(psth_dir, figname))
#        pl.close()
#    
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
