#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 20:36:25 2018

@author: juliana
"""
import os
import sys
import optparse
import seaborn as sns
import numpy as np
import pandas as pd
import pylab as pl
from scipy import stats
from pipeline.python.classifications import utils as util

from pipeline.python.paradigm import align_acquisition_events as acq
from pipeline.python.traces.utils import get_frame_info

#rootdir = '/mnt/odyssey'
#animalid = 'CE077'
#session = '20180521'
#acquisition = 'FOV2_zoom1x'
#run = 'gratings_run1'
#traceid = 'traces001'
#run_dir = os.path.join(rootdir, animalid, session, acquisition, run)
#iti_pre = 1.0


options = ['-D', '/mnt/odyssey', '-i', 'CE077', '-S', '20180521', '-A', 'FOV2_zoom1x',
           '-T', 'np_subtracted',
           '-R', 'blobs_run1', '-t', 'traces001', '-d', 'dff']

def extract_options(options):

    parser = optparse.OptionParser()

    parser.add_option('-D', '--root', action='store', dest='rootdir',
                          default='/nas/volume1/2photon/data',
                          help='data root dir (dir w/ all animalids) [default: /nas/volume1/2photon/data, /n/coxfs01/2pdata if --slurm]')
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
    
    # Set specific session/run for current animal:
    parser.add_option('-d', '--datatype', action='store', dest='datatype',
                          default='corrected', help='Traces to plot (must be in dataset.npz [default: corrected]')
    parser.add_option('-f', '--filetype', action='store', dest='filetype',
                          default='png', help='File type for images [default: png]')
    parser.add_option('--scale', action='store_true', dest='scale_y',
                          default=False, help='Set to scale y-axis across roi images')
    parser.add_option('-y', '--ymax', action='store', dest='dfmax',
                          default=None, help='Set value for y-axis scaling (if not provided, and --scale, uses max across rois)')
    
    (options, args) = parser.parse_args(options)

    return options

#%%
                          
                          
def make_clean_psths(options):
    
    optsE = extract_options(options)
    
    #traceid_dir = util.get_traceid_dir(options)
    run = optsE.run_list[0]
    traceid = optsE.traceid_list[0]
    acquisition_dir = os.path.join(optsE.rootdir, optsE.animalid, optsE.session, optsE.acquisition)
    
    traceid_dir = util.get_traceid_from_acquisition(acquisition_dir, run, traceid)
    data_fpath = os.path.join(traceid_dir, 'data_arrays', 'datasets.npz')
    print "Loaded data from: %s" % traceid_dir
    dataset = np.load(data_fpath)
    print dataset.keys()
        
    #%
    #optsE = extract_options(options)
    inputdata = optsE.datatype
    filetype = optsE.filetype
        
    dfmax = optsE.dfmax
    scale_y = optsE.scale_y
    
    
    #ridx = 0
    #inputdata = 'dff' #corrected'
    assert inputdata in dataset.keys(), "Specified data type (%s) not found! Choose from: %s" % (inputdata, str(dataset.keys()))
    if inputdata == 'corrected' or inputdata=='smoothedX':
        ylabel = 'intensity'
    elif inputdata == 'dff' or inputdata=='smoothedDF':
        ylabel = 'df/f'
        
    xdata = dataset[inputdata]
    ydata = dataset['ylabels']
    tsecs = dataset['tsecs']
    
    run_info = dataset['run_info'][()]
    nframes_per_trial = run_info['nframes_per_trial']
    ntrials_by_cond = run_info['ntrials_by_cond']
    ntrials_total = sum([val for k,val in ntrials_by_cond.iteritems()])
    #trial_labels = np.reshape(ydata, (ntrials_total, nframes_per_trial))[:,0]
    
    # Get stimulus info:
    sconfigs = dataset['sconfigs'][()]
    transform_dict, object_transformations = util.get_transforms(sconfigs)
    trans_types = [trans for trans in transform_dict.keys() if len(transform_dict[trans]) > 1]
    
    # Get trial and timing info:
    #    trials = np.hstack([np.tile(i, (nframes_per_trial, )) for i in range(ntrials_total)])
    tsecs = np.reshape(tsecs, (ntrials_total, nframes_per_trial))
    print tsecs.shape
    #
    if len(trans_types) == 1:
        stim_grid = (transform_dict[trans_types[0]],)
    elif len(trans_types) == 2:
        stim_grid = (transform_dict[trans_types[0]], transform_dict[trans_types[1]])
    
    ncols = len(stim_grid[0])
    columns = trans_types[0]
    col_order = sorted(stim_grid[0])
    nrows = 1; rows = None; row_order=None
    if len(stim_grid) == 2:
        nrows = len(stim_grid[1])
        rows = stim_grid[1]
        row_order = sorted(stim_grid[1])
        
    #%
    # Set output dir for nice(r) psth:
    psth_dir = os.path.join(traceid_dir, 'figures', 'psth_%s' % inputdata)
    if filetype == 'pdf':
        psth_dir = '%s_hq' % psth_dir
        
    if not os.path.exists(psth_dir):
        os.makedirs(psth_dir)
    print "Saving PSTHs to: %s" % psth_dir
    
#    
#    dfmax = xdata.max()
#    #scale_y = False
    if dfmax is None:
        dfmax = xdata.max()
    else:
        dfmax = float(dfmax)
        
        
    for ridx in range(xdata.shape[-1]):
        #%
        roi_id = 'roi%05d' % int(ridx+1)
    
        tpoints = tsecs[0,:]
        labeled_trials = np.reshape(ydata, (ntrials_total, nframes_per_trial))[:,0]
        sconfigs_df = pd.DataFrame(sconfigs).T
        sgroups = sconfigs_df.groupby(trans_types)
    
        tracemat = np.reshape(xdata[:, ridx], (ntrials_total, nframes_per_trial))
        
        sns.set_style('ticks')
        fig, axes = pl.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(15,4*ncols))
        axesf = axes.flat    
        traces_list = []
        pi = 0
        for k,g in sgroups:
            curr_configs = g.index.tolist()
            config_ixs = [ci for ci,cv in enumerate(labeled_trials) if cv in curr_configs]
            subdata = tracemat[config_ixs, :]
            trace_mean = np.mean(subdata, axis=0)
            trace_sem = stats.sem(subdata, axis=0)
            
            axesf[pi].plot(tpoints, trace_mean, color='k', linewidth=1)
            axesf[pi].fill_between(tpoints, trace_mean-trace_sem, trace_mean+trace_sem, color='k', alpha=0.2)
            
            # Set x-axis to only show stimulus ON bar:
            start_val = tpoints[run_info['stim_on_frame']]
            end_val = tpoints[run_info['stim_on_frame'] + int(round(run_info['nframes_on']))]
            axesf[pi].set_xticks((start_val, int(round(end_val))))
            axesf[pi].set_xticklabels(())
            axesf[pi].tick_params(axis='x', which='both',length=0)
            axesf[pi].set_title(k, fontsize=8)
            
            # Set y-axis to be the same, if specified:
            if scale_y:
                axesf[pi].set_ylim([0, dfmax])
    #        else:
    #            if trace_mean.max() < .5:
    #                axesf[pi].set_ylim([0, 1.0])
    #            else:
    #                axesf[pi].set_ylim([0, trace_mean.max()+0.2])
                
            if pi%ncols==0:
                axesf[pi].set_ylabel(ylabel)
            pi += 1
        sns.despine(offset=4, trim=True)
    
        # loop over the non-left axes:
        for ax in axes[1:].flat:
            # get the yticklabels from the axis and set visibility to False
            for label in ax.get_yticklabels():
                label.set_visible(False)
            ax.yaxis.offsetText.set_visible(False)
            ax.yaxis.set_visible(False)
        
        if len(axes.shape) > 1:
            # loop over the top axes:
            for ax in axes[:-1, :].flat:
                # get the xticklabels from the axis and set visibility to False
                for label in ax.get_xticklabels():
                    label.set_visible(False)
                ax.xaxis.offsetText.set_visible(False)
    
        pl.subplots_adjust(top=0.78)
        pl.suptitle("%s" % (roi_id))
        #%
        figname = '%s_psth_%s.%s' % (roi_id, inputdata, filetype)
        pl.savefig(os.path.join(psth_dir, figname))
        pl.close()
    
    return psth_dir

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
