#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 13:22:15 2018

@author: juliana
"""

import sys
import optparse
import os
import cPickle as pkl
import pandas as pd
import numpy as np
import pylab as pl
import seaborn as sns

from pipeline.python.paradigm import align_acquisition_events as acq
from pipeline.python.traces.utils import get_frame_info
from pipeline.python.classifications import utils as util

def test_drift_correction(raw_df, F0_df, corrected_df, run_info, test_roi='roi00001',):
    
    #test_roi = 'roi%05d' % int(roi_id)
    # Check data:
    nframes_per_trial = run_info['nframes_per_trial']
    ntrials_per_file = run_info['ntrials_total'] / run_info['nfiles']
    fig = pl.figure(figsize=(60, 10))
    pl.plot(raw_df[test_roi][0:nframes_per_trial*ntrials_per_file], label='raw')
    pl.plot(F0_df[test_roi][0:nframes_per_trial*ntrials_per_file], label='drift')
    pl.plot(corrected_df[test_roi][0:nframes_per_trial*ntrials_per_file], label='corrected')
    pl.legend()
    pl.savefig(os.path.join(run_info['traceid_dir'], '%s_drift_correction.png' % test_roi))
    pl.show()
    
    
def test_smooth_frac(trace_df, smoothed_df, run_info, is_dff=True, test_roi='roi00001'):
    
    if is_dff:
        trace_label = 'df/f'
    else:
        trace_label = 'raw (F0)'
        
    nframes_per_trial = run_info['nframes_per_trial']
    ntrials_per_file = run_info['ntrials_total'] / run_info['nfiles']
    
    # Check smoothing
    fig = pl.figure(figsize=(20, 5))
    pl.plot(trace_df[test_roi][0:nframes_per_trial*ntrials_per_file], 'k', label=trace_label)
    pl.plot(smoothed_df[test_roi][0:nframes_per_trial*ntrials_per_file], 'r', label='smoothed')
    pl.legend()
    pl.savefig(os.path.join(run_info['traceid_dir'], '%s_dff_smoothed.png' % test_roi))
    pl.show()
   
    fig = pl.figure(figsize=(20, 5))
    pl.plot(trace_df[test_roi][0:nframes_per_trial*ntrials_per_file], 'k', label=trace_label)
    pl.plot(smoothed_df[test_roi][0:nframes_per_trial*ntrials_per_file], 'r', label='smoothed')
    pl.legend()
    pl.savefig(os.path.join(run_info['traceid_dir'], '%s_dff_smoothed.png' % test_roi))
    pl.show()
   

def create_data_arrays(options, test_drift=False, test_smoothing=False, frac=0.01, smooth=False, test_roi='roi00001', quantile=0.08):
    
    # Extract raw trace arrays from the hdf5 files (created in traces/get_traces.py)
    run_info, stimconfigs, labels_df, raw_df = util.get_run_details(options, create_new=True)
    
    # Set data array output dir:
    data_basedir = os.path.join(run_info['traceid_dir'], 'data_arrays')
    if not os.path.exists(data_basedir):
        os.makedirs(data_basedir)
    data_fpath = os.path.join(data_basedir, 'datasets.npz')
    
    # Also create output dir for population-level figures:
    population_figdir = os.path.join(run_info['traceid_dir'], 'figures', 'population')
    if not os.path.exists(population_figdir):
        os.makedirs(population_figdir)
        
    
    # Get processed traces:
    print "Getting processed traces!"
    _, corrected_df, F0_df = util.load_roiXtrials_df(run_info['traceid_dir'], trace_type='processed', dff=False, smoothed=False, quantile=quantile)

    # Check data:
    print "Showing initial drift correction (quantile: %.2f)" % quantile
    print "Mean baseline for all ROIs:", np.mean(np.mean(F0_df, axis=0))
    test_drift_correction(raw_df, F0_df, corrected_df, run_info, test_roi=test_roi)
    if test_drift is False:
        pl.close()
    
    if test_drift:
        confirm_drift = raw_input('Press <Y> if F0 drift is good. Otherwise, select NTRIALS to use as window (default: 3).')
        quantile_sel = raw_input('Press <Y> if F0 drift is good. Otherwise, select QUANTILE to use as baseline (default: 0.08).')

        if confirm_drift=='Y':
            pl.close()
        else:
            while True:
                window_ntrials = int(confirm_drift)
                quantile = float(quantile_sel)
                print "Selected %i trials to use for calculating rolling %.3f percentile..." % (window_ntrials, quantile)
                _, corrected_df, F0_df = util.load_roiXtrials_df(run_info['traceid_dir'], trace_type='processed', dff=False, smoothed=False, 
                                                                     window_ntrials=window_ntrials, quantile=quantile, create_new=True)
                test_drift_correction(raw_df, F0_df, corrected_df, run_info, test_roi=test_roi)
                print "Showing initial drift correction (quantile: %.2f)" % quantile
                print "Mean baseline for all ROIs:", np.mean(np.mean(F0_df, axis=0))
                confirm_drift = raw_input('Press <Y> if F0 calculation is good. Otherwise, select ntrials to use as window.')
                quantile_sel = raw_input('Press <Y> if F0 drift is good. Otherwise, select QUANTILE to use as baseline (default: 0.08).')

                if confirm_drift=='Y':
                    break

    # Get df/f (F-measured and F-0 already calculated from PROCESSED traces extraction)
    _, dff_df, _ = util.load_roiXtrials_df(run_info['traceid_dir'], trace_type='processed', dff=True, smoothed=False)

    # Test fraction ranges for smoothing trace:
    if test_smoothing:                
        util.test_file_smooth(run_info['traceid_dir'], use_raw=False, ridx=0, fmin=0.001, fmax=0.02, save_and_close=False) #, output_dir=data_basedir)
        get_frac = input('Enter fraction to use for smoothing: ')
        print get_frac
        frac = float(get_frac)
        print "Smoothing traces with fraction: ", frac
    else:
        util.test_file_smooth(run_info['traceid_dir'], use_raw=False, ridx=0, fmin=0.001, fmax=0.02, save_and_close=True, output_dir=data_basedir)
        pl.close()
        
    smoothed_df = None; smoothed_X = None;
    if smooth:
        print "smoothing..."
        # Smooth df/f and measured traces using test value above:
        _, smoothed_df, _ = util.load_roiXtrials_df(run_info['traceid_dir'], trace_type='processed', dff=True, smoothed=True, frac=frac)
        _, smoothed_X, _ = util.load_roiXtrials_df(run_info['traceid_dir'], trace_type='processed', dff=False, smoothed=True, frac=frac)

        test_smooth_frac(dff_df, smoothed_df, run_info, is_dff=True, test_roi=test_roi)

    
    # Get label info:
    sconfigs = util.format_stimconfigs(stimconfigs)

    ylabels = labels_df['config'].values
    groups = labels_df['trial'].values
    tsecs = labels_df['tsec']
    
    
    # Get single-valued training data for each trial:
    meanstim_values = util.format_roisXvalue(corrected_df, run_info, value_type='meanstim')
    zscore_values = util.format_roisXvalue(corrected_df, run_info, value_type='zscore')
    meanstimdff_values = util.format_roisXvalue(dff_df, run_info, value_type='meanstim')
    
    pl.figure(figsize=(20,5)); 
    pl.subplot(1,3,1); ax=sns.heatmap(meanstim_values); pl.title('mean stim (raw)')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=45)
    ax.set_ylabel('trial')
    ax.set_xlabel('roi')
    pl.subplot(1,3,2); ax=sns.heatmap(zscore_values); pl.title('zscored')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=45)
    pl.subplot(1,3,3); ax=sns.heatmap(meanstimdff_values); pl.title('mean stim (dff)')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=45)
    pl.savefig(os.path.join(population_figdir, 'roiXtrial_values.png'))
    pl.close()
        
    # Save:
    print "Saving processed data...", data_fpath
    np.savez(data_fpath, 
             raw=raw_df,
             smoothedDF=smoothed_df,
             smoothedX=smoothed_X,
             frac=frac,
             tsecs=tsecs,
             groups=groups,
             ylabels=ylabels,
             dff=dff_df,
             corrected=corrected_df,
             sconfigs=sconfigs, 
             meanstim=meanstim_values, 
             zscore=zscore_values,
             meanstimdff=meanstimdff_values,
             run_info=run_info)

    dataset = np.load(data_fpath)
        
    return dataset


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
    parser.add_option('-n', '--nruns', action='store', dest='nruns', default=1, help="Number of consecutive runs if combined")
    parser.add_option('--slurm', action='store_true', dest='slurm', default=False, help="set if running as SLURM job on Odyssey")
    parser.add_option('--combo', action='store_true', dest='combined', default=False, help="Set if using combined runs with same default name (blobs_run1, blobs_run2, etc.)")

    parser.add_option('--frac', action='store', dest='frac', default=0.01, help="Fraction of trace to use for lowess smoothing [default: 0.01]")
    parser.add_option('-r', '--roi', action='store', dest='test_roi_id', default=1, help="Roi ID to use for tests [default: 1]")
    parser.add_option('--smooth', action='store_true', dest='smooth', default=False, help="Set flag to smooth traces")
    parser.add_option('--test-smooth', action='store_true', dest='test_smoothing', default=False, help="Set flag to test frac ranges for smoothing traces")
    parser.add_option('--test-drift', action='store_true', dest='test_drift', default=False, help="Set flag to inspect drift correction for F0 calculation")

    parser.add_option('--new', action='store_true', dest='create_new', default=False, help="Set flag to create data arrays from new.")
    parser.add_option('--align', action='store_true', dest='align_frames', default=False, help="Set flag to (re)-align frames to trials.")
    parser.add_option('--iti', action='store', dest='iti_pre', default=1.0, help="Num seconds to use as pre-stimulus period [default: 1.0]")
    parser.add_option('--post', action='store', dest='iti_post', default=None, help="Num seconds to use as pre-stimulus period [default: tue ITI - iti_pre]")
    parser.add_option('-q', '--quant', action='store', dest='quantile', default=0.08, help="Quantile of trace to include for drift calculation (default: 0.08)")



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

    return options

#%%
    
#rootdir = '/mnt/odyssey'
#animalid = 'CE077'
#session = '20180521'
#acquisition = 'FOV2_zoom1x'
#run = 'gratings_run1'
#traceid = 'traces001'
#run_dir = os.path.join(rootdir, animalid, session, acquisition, run)
#iti_pre = 1.0
#

opts = ['-D', '/mnt/odyssey', '-i', 'CE077', '-S', '20180521', '-A', 'FOV2_zoom1x',
           '-T', 'np_subtracted', '--no-pupil',
           '-R', 'blobs_run1', '-t', 'traces001',
           '-n', '1']
#
#opts = ['-D', '/mnt/odyssey', '-i', 'CE077', '-S', '20180523', '-A', 'FOV1_zoom1x',
#           '-T', 'np_subtracted', '--no-pupil',
#           '-R', 'gratings_run1', '-t', 'traces001',
#           '-r', '18', '--smooth', '--iti=1.0', '--test-smooth']


opts = ['-D', '/mnt/odyssey', '-i', 'CE077', '-S', '20180602', '-A', 'FOV1_zoom1x',
           '-T', 'np_subtracted', '--no-pupil',
           '-R', 'blobs_dynamic_run6', '-t', 'traces001',
           '-r', '18', '--smooth', '--iti=4.0', '--post=12.0', '--test-smooth']

#%%

def create_rdata_array(opts):
        
    optsE = extract_options(opts)
    print optsE
    test_roi = 'roi%05d' % int(optsE.test_roi_id) 
    frac = float(optsE.frac)
    test_smoothing = optsE.test_smoothing
    test_drift = optsE.test_drift
    smooth = optsE.smooth
    align_frames = optsE.align_frames
    create_new = optsE.create_new
    iti_pre = float(optsE.iti_pre)
    iti_post = optsE.iti_post
    if iti_post is not None:
        iti_post = float(iti_post)
        
    quantile = float(optsE.quantile)
    
    #
    options = ['-D', optsE.rootdir, '-i', optsE.animalid, '-S', optsE.session, 
               '-A', optsE.acquisition,
               '-T', optsE.trace_type,
               '-R', optsE.run_list[0], '-t', optsE.traceid_list[0],
               '-n', optsE.nruns,
               '-q', optsE.quantile]
    if optsE.filter_pupil is False:
        options.extend(['--no-pupil'])
        


    #test_roi = 'roi00001'
    #frac = 0.01
    #test_smoothing = True
    #test_drift = True
    
    traceid_dir = util.get_traceid_dir(options)
    
    # Set data array output dir:
    data_basedir = os.path.join(traceid_dir, 'data_arrays')
    if not os.path.exists(data_basedir):
        os.makedirs(data_basedir)
    
    # Also create output dir for population-level figures:
    population_figdir = os.path.join(traceid_dir, 'figures', 'population')
    if not os.path.exists(population_figdir):
        os.makedirs(population_figdir)
        
    # First check if processed datafile exists:
    data_fpath = os.path.join(data_basedir, 'datasets.npz')
    if create_new is False:
        try:
            dataset = np.load(data_fpath)
            print "Loaded existing datafile:\n%s" % data_fpath
            print dataset.keys()
        except Exception as e:
            create_new = True
            
    if create_new:
        
        run_dir = traceid_dir.split('/traces')[0]
        
        # Get paradigm/AUX info:
        # =============================================================================
        paradigm_dir = os.path.join(run_dir, 'paradigm')
        si_info = get_frame_info(run_dir)
        trial_info = acq.get_alignment_specs(paradigm_dir, si_info, iti_pre=iti_pre, iti_post=iti_post)
        configs, stimtype = acq.get_stimulus_configs(trial_info)
    
        print "-------------------------------------------------------------------"
        print "Getting frame indices for trial epochs..."
        parsed_frames_filepath = acq.assign_frames_to_trials(si_info, trial_info, paradigm_dir, create_new=align_frames)
        print "Finished aligning frames to trial structure (iti pre = %i)" % iti_pre
        print "Saved parsed frames to: %s" % parsed_frames_filepath
        
        dataset = create_data_arrays(options, test_drift=test_drift, test_smoothing=test_smoothing, frac=frac, test_roi=test_roi, smooth=smooth)

    return data_fpath
#%

def get_rdata_dataframe(data_fpath):
    
    # Set data array output dir:
    data_basedir = os.path.split(data_fpath)[0]
    #data_basedir = os.path.join(traceid_dir, 'data_arrays')
    roidata_fpath = os.path.join(data_basedir, 'ROIDATA.pkl')
    labels_fpath = os.path.join(data_basedir, 'runinfo.pkl')
    
    try:
        with open(roidata_fpath, 'rb') as f:
            DATA = pkl.load(f)
        print "Loaded data frame."
        print DATA.head()
        
        with open(labels_fpath, 'rb') as f:
            paradigm_info = pkl.load(f)
        return DATA, paradigm_info['run_info'], paradigm_info['sconfigs']
    except Exception as e:
        print "Unable to find ROIDATA array df."
        try:
            dataset_fpath = os.path.join(data_basedir, 'datasets.npz')
            dataset = np.load(dataset_fpath)
            print "Loaded dataset arrays. Creating new ROIDATA dataframe."
        except Exception as e:
            print "Unable to find data arrays..."
            print "Did you run create_rdata_array()?"
            print "Expected file in: %s" % dataset_fpath
            return None
            

    #inputdata = 'corrected' #corrected'
    #assert inputdata in dataset.keys(), "Specified data type (%s) not found! Choose from: %s" % (inputdata, str(dataset.keys()))
    rawdata = dataset['corrected']
    dffdata = dataset['dff']
    nrois = rawdata.shape[-1]
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
    trials = np.hstack([np.tile('trial%05d' % int(i+1), (nframes_per_trial, )) for i in range(ntrials_total)])
    nframes_on = np.hstack([np.tile(run_info['nframes_on'], (nframes_per_trial, )) for i in range(ntrials_total)])
    first_on = np.hstack([np.tile(run_info['stim_on_frame'], (nframes_per_trial, )) for i in range(ntrials_total)])
    
    
    #tsecs = np.reshape(tsecs, (ntrials_total, nframes_per_trial))
    #print tsecs.shape
        
    # Create raw dataframe:
    new_columns=[]
    for trans in trans_types:
        trans_vals = [sconfigs[c][trans] for c in ydata]
        new_columns.append(pd.DataFrame(data=trans_vals, columns=[trans], index=xrange(len(ydata))))
    new_columns.append(pd.DataFrame(data=trials, columns=['trial'], index=xrange(len(ydata))))
    new_columns.append(pd.DataFrame(data=tsecs, columns=['tsec'], index=xrange(len(ydata))))
    new_columns.append(pd.DataFrame(data=nframes_on, columns=['nframes_on'], index=xrange(len(ydata))))
    new_columns.append(pd.DataFrame(data=first_on, columns=['first_on'], index=xrange(len(ydata))))
    new_columns.append(pd.DataFrame(data=ydata, columns=['config'], index=xrange(len(ydata))))
    
    config_df = pd.concat(new_columns, axis=1)
    
    df_list = []
    for ridx in range(nrois):
        raw_df = pd.DataFrame(data=rawdata[:,ridx], columns=['raw'], index=xrange(len(ydata)))
        dff_df = pd.DataFrame(data=dffdata[:,ridx], columns=['dff'], index=xrange(len(ydata)))
        df = pd.concat([raw_df, dff_df, config_df], axis=1).reset_index(drop=True)
    
        roiname = 'roi%05d' % int(ridx+1)
        df['roi'] = pd.Series(np.tile(roiname, (len(df.index),)), index=df.index)
        df_list.append(df)
    
    DATA = pd.concat(df_list, axis=0, ignore_index=True)
    print DATA.head()
    
    with open(roidata_fpath, 'wb') as f:
        pkl.dump(DATA, f, protocol=pkl.HIGHEST_PROTOCOL)
        
    runinfo_fpath = os.path.join(data_basedir, 'runinfo.pkl')
    info = {'run_info': run_info,
            'sconfigs': sconfigs}
    with open(runinfo_fpath, 'wb') as f:
        pkl.dump(info, f, protocol=pkl.HIGHEST_PROTOCOL)
    
    print "DONE!"
        
    return DATA, run_info, sconfigs


#%%
def main(options):
    
    data_fpath = create_rdata_array(options)
    get_rdata_dataframe(data_fpath)
    
    print "*******************************************************************"
    print "DONE!"
    print "New ROIDATA array saved to: %s" % data_fpath
    print "*******************************************************************"
    
    
#%%

if __name__ == '__main__':
    main(sys.argv[1:])

