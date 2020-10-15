#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 20:36:25 2018

@author: juliana
"""

#%%
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
#from pipeline.python.classifications import experiment_classes as util

from pipeline.python.paradigm import align_acquisition_events as acq
#from pipeline.python.traces.utils import get_frame_info
from pipeline.python.utils import label_figure, get_frame_info
from pipeline.python import utils as util


def extract_options(options):

    parser = optparse.OptionParser()

    parser.add_option('-D', '--root', action='store', dest='rootdir',
                          default='/n/coxfs01/2p-data',
                          help='data root dir (dir w/ all animalids) [default: /n/coxfs01/2pdata]')
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
    parser.add_option('-R', '--run', dest='run', default='', action='store', help="run name")
    parser.add_option('-t', '--traceid', dest='traceid', default=None, action='store', help="e.g., traces001")

    # Set specific session/run for current animal:
    parser.add_option('-d', '--datatype', action='store', dest='datatype',
                          default='corrected', help='Traces to plot (must be in dataset.npz [default: corrected]')
    parser.add_option('-f', '--filetype', action='store', dest='filetype',
                          default='svg', help='File type for images [default: svg]')

    parser.add_option('--scale', action='store_true', dest='scale_y',
                          default=False, help='Set to scale y-axis across roi images')
    parser.add_option('-y', '--ymax', action='store', dest='dfmax', default=None,
                          help='Set value for y-axis scaling (if not provided, and --scale, uses max across rois)')
    parser.add_option('--shade', action='store_false', dest='plot_trials',
                          default=True, help='Set to plot mean/sem as shaded (default plots individual trials)')
    parser.add_option('--median', action='store_true', dest='plot_median',
                          default=False, help='Set to plot MEDIAN (default plots mean across trials)')


    parser.add_option('-r', '--rows', action='store', dest='rows', default=None,
                        help='Transform to plot along ROWS')
    parser.add_option('-c', '--columns', action='store', dest='columns', default=None,
                        help='Transform to plot along COLUMNS')
    parser.add_option('-H', '--hue', action='store', dest='subplot_hue', default=None,
                        help='Transform to plot by HUE within each subplot')
    parser.add_option('--filter', action='store_true', dest='filter_noise', default=False,
                        help='Set to filter our noisy spikes') 
    parser.add_option('--no-offset', action='store_false', dest='add_offset',
                          default=True, help='Set to do offset old way') 
   
    (options, args) = parser.parse_args(options)

    
    return options

#

#options = ['-D', '/n/coxfs01/2p-data','-i', 'JC078', '-S', '20190426', '-A', 'FOV1_zoom2p0x',
#           '-R', 'combined_blobs_static', '-t', 'traces001', '--shade',
#           '--compare','-p', 'color',
#           '-r', 'size', '-c', 'morphlevel']

#options = ['-D', '/n/coxfs01/2p-data','-i', 'JC078', '-S', '20190427', '-A', 'FOV1_zoom2p0x',
#           '-R', 'combined_gratings_static', '-t', 'traces001', '--shade',
#           '-r', 'ypos', '-c', 'xpos']

options = ['-D', '/n/coxfs01/2p-data','-i', 'JC084', '-S', '20190522', 
           '-A', 'FOV1_zoom2p0x', '-d', 'dff',
           '-R', 'combined_rfs_static', '-t', 'traces001', '--shade',
           '-r', 'ypos', '-c', 'xpos'] #, '-H', 'sf']

#%%
def get_data_id_from_tracedir(traceid_dir, rootdir='/n/coxfs01/2p-data/'):
    trace_info = traceid_dir.split(rootdir)[-1]
    data_identifier = trace_info.replace('/', '|')
    return data_identifier
#    
def get_mean_and_sem_timecourse(rdata, sdf, plot_params, trace_type='dff'):

    assert trace_type in rdata.columns, "[ERR]: <%s> not found in columns" % trace_type    

    meandfs = []
    for k, g in rdata.groupby(['config']):
        nreps = len(g['trial'].unique())
        mean_trace = g.groupby(['trial'])[trace_type].apply(np.array).mean(axis=0)
        mean_tsec = g.groupby(['trial'])['tsec'].apply(np.array).mean(axis=0)
        sem_trace = stats.sem(np.vstack(g.groupby(['trial'])[trace_type].apply(np.array)), axis=0, nan_policy='omit')
        mdf = pd.DataFrame({'%s' % trace_type: mean_trace,
                            'tsec': mean_tsec,
                            'sem': sem_trace,
                            'fill_minus': mean_trace - sem_trace,
                            'fill_plus': mean_trace + sem_trace,
                            'config': [k for _ in range(len(mean_trace))],
                            'nreps': [nreps for _ in range(len(mean_trace))]

                           })
        for p in plot_params.values():
            if p is None: 
                continue
            mdf[p] = [round(sdf[p][cfg], 1) if isinstance(sdf[p][cfg], (float)) else sdf[p][cfg] for cfg in mdf['config']]
        meandfs.append(mdf)
    meandfs = pd.concat(meandfs, axis=0)
    
    return meandfs

def add_text(x, huep, **kwargs):
    ax = pl.gca()
    if huep.unique() == 0.1:
        ax.text(0, 1, 'n=%i' % x.unique(), ha='left', va='top',  transform=ax.transAxes, **kwargs)
    else:
        ax.text(0, 0.95, 'n=%i' % x.unique(), ha='left', va='top',  transform=ax.transAxes, **kwargs)
        
def add_stimulus_bar(x, start_val=0, end_val=1, color='k', alpha=0.5, **kwargs):
    ax = pl.gca()
    ymin, ymax = ax.get_ylim()
    ax.add_patch(patches.Rectangle((start_val, ymin), end_val, (ymax-ymin), linewidth=0, 
                                  fill=True, color=color, alpha=alpha))
    
    
def fix_grid_labels(p, trace_type='trace_type'):
    '''
    p is output of sns.FacetGrid
    '''
    for ri in np.arange(0, p.axes.shape[0]):
        for ci in np.arange(0, p.axes.shape[1]):
            if ci==0:
                p.axes[ri, ci].set_ylabel(trace_type)
            else:
                p.axes[ri, ci].set_ylabel('')
            if ri == p.axes.shape[0]-1:
                p.axes[ri, ci].set_xlabel('time (s)')
            else:
                p.axes[ri, ci].set_xlabel('')


def plot_psth_grid(meandfs, plot_params, trace_type='trace_type', palette='colorblind', 
                       stim_start_tsec=0, stim_end_tsec=1):
    
    p = sns.FacetGrid(meandfs, col=plot_params['cols'], row=plot_params['rows'], hue=plot_params['hue'],\
                      sharex=True, sharey=True, palette=palette)


    if len(meandfs[plot_params['rows']].unique()) == 1:
        p.fig.set_figheight(3)
        p.fig.set_figwidth(20)

    if plot_params['hue'] is None:
        p = p.map(pl.fill_between, "tsec", "fill_minus", "fill_plus", alpha=0.5, color='k')
        p = p.map(pl.plot, "tsec", trace_type, lw=1, alpha=1, color='k')
    else:
        p = p.map(pl.fill_between, "tsec", "fill_minus", "fill_plus", alpha=0.5)
        p = (p.map(pl.plot, "tsec", trace_type, lw=1, alpha=1).add_legend())

    p = p.set_titles(col_template="{col_name}", size=12)   
    p.map(add_stimulus_bar, 'ntrials',  start_val=stim_start_tsec, end_val=stim_end_tsec, color='k', alpha=0.1)
    if plot_params['hue'] is not None:
        p.map(add_text, 'ntrials', plot_params['hue'])
    pl.subplots_adjust(top=0.9, right=0.9, wspace=0.1, hspace=0.4)
    sns.despine(trim=True) #, bottom=True) 
    fix_grid_labels(p, trace_type=trace_type)

    if 'xpos' in plot_params.values() and 'ypos' in plot_params.values(): 
        pl.subplots_adjust(wspace=0.05, hspace=0.3, top=0.85, bottom=0.1, left=0.05) 
    else:
        pl.subplots_adjust(wspace=0.1, hspace=0.4, top=0.9, right=0.9)

    return p
    

def plot_psth_and_save(rid, meandfs, plot_params, trace_type='trace_type', palette='colorblind', 
                       stim_start_tsec=0, stim_end_tsec=1, fig_id='figureID', dst_dir='/tmp', filetype='svg'):
    
    p = plot_psth_grid(meandfs, plot_params, trace_type=trace_type, palette=palette, 
                       stim_start_tsec=stim_start_tsec, stim_end_tsec=stim_end_tsec)
    
    p.fig.suptitle('roi %i' % int(rid+1))
    label_figure(p.fig, fig_id)

    figname = 'roi%05d_%s' % (int(rid+1), trace_type)
    pl.savefig(os.path.join(dst_dir, '%s.%s' % (figname, filetype)))
    #print(figname)
    pl.close()
    
    

#%%                    
def make_clean_psths(options):
    #%%
    optsE = extract_options(options)
    run = optsE.run #run_list[0]
    traceid = optsE.traceid #traceid_list[0]
    inputdata = optsE.datatype
    filetype = optsE.filetype
    filter_noise = optsE.filter_noise
    trace_type = optsE.datatype

    # Plotting options:
    filetype = optsE.filetype
    scale_y = optsE.scale_y

    # Set data source
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
    add_offset = optsE.add_offset

    #%%  Set plotting styles and params 
    subplot_hue = optsE.subplot_hue.split(',') if optsE.subplot_hue not in [None, 'None'] else [None]
    rows = optsE.rows.split(',') if optsE.rows is optsE.rows not in [None, 'None'] else [None]
    columns = optsE.columns.split(',') if optsE.columns not in [None, 'None'] else [None]
    compare_param = optsE.compare_param
    if compare_runs:
        subplot_hue = compare_param
    plot_params = {'hue': subplot_hue if len(subplot_hue) > 1 else subplot_hue[0],
                   'rows': rows if len(rows) > 1 else rows[0],
                   'cols': columns if len(columns) > 1 else columns[0]}
    
    compare_param_name = 'backlight' if compare_param=='color' else compare_param
   
    # Aggregate the data 
    xdata_list=[]; labels_list=[]; sdf_list=[]; data_ids=[];
    for traceid_dir in traceid_dirs:
        dataset_name = 'np_subtracted' if trace_type in ['dff', 'df', 'corrected'] else trace_type

        soma_fpath = os.path.join(traceid_dir, 'data_arrays', '%s.npz' % dataset_name)
        xdata, labels, sdf, run_info = util.load_dataset(soma_fpath, trace_type=trace_type)
        sdf = sdf.assign(ix=[int(i.split('config')[-1]) for i in sdf.index]).sort_values('ix')
        
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
   
    # Fix some naming stuff for easier labeling/plotting 
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
#% 
    # Combine data:
    sdf_c = pd.concat(sdf_list, axis=0) # Get rid of config labels
    xdata_c = pd.concat(xdata_list, axis=0).reset_index(drop=True)
    labels_c = pd.concat(labels_list, axis=0).reset_index(drop=True) # Get rid of config labels        
    stim_on = labels_c['stim_on_frame'].unique()[0]
    nframes_on = labels_c['nframes_on'].unique()[0]
    mean_tsecs = labels_c.groupby(['trial'])['tsec'].apply(np.array).mean(axis=0)
    print(xdata_c.head())

    for pparam in plot_params.values():
        if isinstance(pparam, list):
            for pp in pparam:
                if pp not in labels_c.columns and pp is not None:
                    #print pparam
                    labels_c[pp] = [sdf_c[pp][cfg] for cfg in labels_c['config']]
        else:
            if pparam not in labels_c.columns and pparam is not None:
                #print pparam
                labels_c[pparam] = [sdf_c[pparam][cfg] for cfg in labels_c['config']]
    if 'size' in labels_c.columns:
        labels_c = labels_c.round({'size': 0}).astype({'size': int})
        sdf_c = sdf_c.round({'size': 0})

#%
    # Set output dirs
    if compare_runs:
        output_figdir = os.path.join(traceid_dirs[0].split('/traces/')[0], 'compare_runs', 'figures')
    else:
        output_figdir = os.path.join(traceid_dir, 'figures')
    if not os.path.exists(output_figdir):
        os.makedirs(output_figdir)
    print "OUTPUT saved to:", output_figdir    
    #%%
  
    #%% Set what gets plotted where and houes
    # Get varying transforms:
    ignore_params = ['position', 'aspect', 'stimtype']
    transform_params = [p for p in sdf_c.columns if p not in ignore_params]
    transform_dict = dict((param, sdf_c[param].unique()) for param in transform_params)
    for k, v in transform_dict.items():
        if len(v) == 1:
            transform_dict.pop(k)

    # replace duration:
    if 'duration' in transform_dict.keys():
        transform_dict['stim_dur'] = transform_dict['duration']
        transform_dict.pop('duration')

    if 'position' in plot_params.values() and 'position' not in sdf_c.columns.tolist():
        posvals = list(set(zip(sdf_c['xpos'].values, sdf_c['ypos'].values)))
        print "Found %i unique positions." % len(posvals)
        transform_dict['position'] = posvals
        sdf['position'] = list(zip(sdf_c['xpos'], sdf_c['ypos']))


    if isinstance(plot_params['rows'], list) and len(plot_params['rows']) > 1:
        combo_param_name = '_'.join(plot_params['rows'])
        sdf_c[combo_param_name] = ['_'.join([str(c) for c in list(combo[0])]) for combo in list(zip(sdf_c[plot_params['rows']].values))]
        plot_params['rows'] = combo_param_name
        
    if isinstance(plot_params['cols'], list) and len(plot_params['cols']) > 1:
        combo_param_name = '_'.join(plot_params['cols'])
        sdf_c[combo_param_name] = ['_'.join([str(c) for c in list(combo[0])]) for combo in list(zip(sdf_c[plot_params['cols']].values))]
        plot_params['cols'] = combo_param_name

    if isinstance(plot_params['hue'], list) and len(plot_params['hue']) > 1:
        combo_param_name = '_'.join(plot_params['hue'])
        sdf_c[combo_param_name] = ['_'.join([str(c) for c in list(combo[0])]) for combo in list(zip(sdf_c[plot_params['hue']].values))]
        plot_params['hue'] = combo_param_name
        
        
    trans_types = sorted([trans for trans in transform_dict.keys() if len(transform_dict[trans]) > 1 and trans!='ix'])         
    print "Trans:", trans_types
    #print sdf_c.head()    
   
    # -------------------------------------------------------------------------
    # Create PSTH plot grid:
    # -------------------------------------------------------------------------
    all_plot_params = []
    for k, v in plot_params.items():
        if isinstance(v, list):
            all_plot_params.extend(v)
        else:
            all_plot_params.append(v)
    unspecified_trans_types = [t for t in trans_types if t not in all_plot_params]
    #%
    # Filter out noisy spikes:
    figdir_append = ''
    if optsE.datatype == 'spikes' and optsE.filter_noise:
        xdata[xdata<=0.0004] = 0.
        figdir_append = '_filtered'
    if optsE.plot_median:
        figdir_append = '%s_median' % figdir_append 
    
    # Set output dir for nice(r) psth:
    plot_type = 'trials' if optsE.plot_trials else 'shade' 
    psth_dir = os.path.join(output_figdir, 'psth_%s_%s%s' % (optsE.datatype, plot_type, figdir_append))
    if optsE.filetype == 'pdf':
        psth_dir = '%s_hq' % psth_dir
       
    if not os.path.exists(psth_dir):
        os.makedirs(psth_dir)
    print "Saving PSTHs to: %s" % psth_dir

    # Set COLORS for subplots:
    print "Trans Types:", trans_types
    if len(trans_types)<=2 and plot_params['hue'] is None:
        print "PLOTTING 1 color"
        trace_colors = ['k']
        trace_labels = ['']
    else: 
        print "Subplot hue: %s" % plot_params['hue']
        if plot_params['hue'] in sdf_c.columns:
            hues = sorted(sdf_c[plot_params['hue']].unique())
            trace_labels = ['%s %s' % (plot_params['hue'], str(v)) for v in sorted(sdf_c[plot_params['hue']].unique())]
        else:
            trace_labels = []
    print trace_labels #rows, object_transformations[rows] #trace_labels

    # pick some colors:
    # = ["forest green", "purple"] #["windows blue", "amber", "greyish", "faded green", "dusty purple"]
    palette = 'colorblind' #sns.xkcd_palette(colors) 
    #sns.set()
    #palette = sns.color_palette('colorblind')
    
    trial_counts = labels_c[['config', 'trial']].drop_duplicates().groupby(['config']).count()
    stim_start_tsec = 0.0
    stim_end_tsec = mean_tsecs[stim_on + int(round(nframes_on))]

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
        rdata[trace_type] = xdata_c[ridx]
        
        if optsE.plot_trials:
            p = sns.FacetGrid(rdata, col=plot_params['cols'], 
                              row=plot_params['rows'], hue=plot_params['hue'], 
                              sharex=True, sharey=True, palette=palette)
            p.map(pl.plot, "tsec", trace_type, lw=0.5, alpha=0.5)
        else:
            meandfs = get_mean_and_sem_timecourse(rdata, sdf_c, plot_params, 
                                                  trace_type=trace_type)
            meandfs['ntrials'] = [trial_counts.loc[cfg]['trial'] for cfg in meandfs['config']]
            
            plot_psth_and_save(ridx, meandfs, plot_params, trace_type=trace_type, palette='colorblind', 
                                   stim_start_tsec=stim_start_tsec, stim_end_tsec=stim_end_tsec, 
                                   fig_id=data_identifier, dst_dir=psth_dir, filetype=filetype)


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
def main(options):
    
    psth_dir = make_clean_psths(options)
    
    print "*******************************************************************"
    print "DONE!"
    print "All output saved to: %s" % psth_dir
    print "*******************************************************************"
    
    
#%%

if __name__ == '__main__':
    main(sys.argv[1:])
