#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 14:50:56 2019

@author: julianarhee
"""

import os
import glob
import json
import h5py
import optparse
import sys

import pandas as pd
import pylab as pl
import seaborn as sns
import numpy as np

from pipeline.python.utils import natural_keys, label_figure

#%%
def convert_values(oldval, newmin, newmax, oldmax=None, oldmin=None):
    oldrange = (oldmax - oldmin)
    newrange = (newmax - newmin)
    newval = (((oldval - oldmin) * newrange) / oldrange) + newmin
    return newval

def trials_to_dataframes(processed_fpaths, conditions_fpath):
    
    # Get condition / trial info:
    with open(conditions_fpath, 'r') as f:
        conds = json.load(f)
    cond_list = list(set([cond_dict['stimuli']['stimulus'] for trial_num, cond_dict in conds.items()]))
    trials_by_cond = dict((cond, [int(k) for k, v in conds.items() if v['stimuli']['stimulus']==cond]) for cond in cond_list)


    fits = []
    phases = []
    mags = []
    for trial_num, trial_fpath in zip(sorted([int(k) for k in conds.keys()]), sorted(processed_fpaths, key=natural_keys)):
        
        print("%i: %s" % (trial_num, os.path.split(trial_fpath)[-1]))
        df = h5py.File(trial_fpath, 'r')
        fits.append(pd.Series(data=df['var_exp_array'][:], name=trial_num))
        phases.append(pd.Series(data=df['phase_array'][:], name=trial_num))
        mags.append(pd.Series(data=df['mag_ratio_array'][:], name=trial_num))
        df.close()
        
    fit = pd.concat(fits, axis=1)
    magratio = pd.concat(mags, axis=1)
    phase = pd.concat(phases, axis=1)
    
    return fit, magratio, phase, trials_by_cond


def plot_signal_fits_by_roi(fit, magratio, threshold=0.2, 
                            data_identifier='', fov='FOV', retinoid='retinoID', output_dir='/tmp'):
    
    fig = pl.figure(figsize=(20,10))
    nr = 2
    nc = 3
    
    #%
    
    rnums = np.arange(1, fit.shape[0]+1)
    mean_fits = fit.mean(axis=1)
    mean_fits_std = fit.std(axis=1)
    
    max_fits = fit.max(axis=1)
    
    #%
    ax1 = pl.subplot2grid((nr,nc), (0, 0), colspan=3)
    ax1.plot(rnums, mean_fits, '.')
    ax1.set_xticks(rnums)
    if len(rnums) > 50:
        rnum_labels = [r if (r==1 or r%10==0) else '' for r in rnums]
    else:
        rnum_labels = rnums.copy()
    ax1.set_xticklabels(rnum_labels)
    ax1.errorbar(x=rnums, y=mean_fits, yerr=mean_fits_std, color='k', fmt='.', elinewidth=1, label='var exp')
    ax1.plot(rnums, max_fits, '.', color='b', label='max fit')
    
    ax1.set_ylabel('mean fit (+/- std)')
    ax1.set_xlabel('roi')
    ax1.axhline(y=threshold, color='r', linewidth=0.5, label='threshold')
    
    ax1.legend()
    
    #%
    mean_magratios = magratio.mean(axis=1)
    max_magratios = magratio.max(axis=1)
    
    ax2 = pl.subplot2grid((nr, nc), (1, 0), colspan=1)
    ax2.scatter(mean_fits, mean_magratios)
    ax2.set_xlabel('mean fit')
    ax2.set_ylabel('mean mag ratio')
    #ax2.set_ylim([0, mean_magratios.max()])
    
    ax3 = pl.subplot2grid((nr, nc), (1, 1), colspan=1)
    ax3.scatter(max_fits, mean_magratios)
    ax3.set_xlabel('max fit')
    #ax3.set_ylim([0, mean_magratios.max()])
    
    ax4 = pl.subplot2grid((nr, nc), (1, 2), colspan=1)
    ax4.scatter(max_fits, max_magratios)
    ax4.set_xlabel('max fit')
    ax4.set_ylabel('max mag ratio')
    
    #% Save figure
    
    label_figure(fig, data_identifier)
    figname = 'var_exp_by_roi_%s_%s.png' % (fov, retinoid)
    
    pl.savefig(os.path.join(output_dir, 'visualization', figname))

    return fig


def get_screen_info(animalid, session, rootdir='/n/coxfs01/2p-data'):
        
    screen = {}
    
    # Get bounding box values from epi:
    epi_session_paths = sorted(glob.glob(os.path.join(rootdir, animalid, 'epi_maps', '20*')), key=natural_keys)
    epi_sessions = sorted([os.path.split(s)[-1].split('_')[0] for s in epi_session_paths], key=natural_keys)
    if len(epi_sessions) > 0:
        epi_sesh = [datestr for datestr in sorted(epi_sessions, key=natural_keys) if int(datestr) <= int(session)][-1] # Use most recent session
        epi_fpaths = glob.glob(os.path.join(rootdir, animalid, 'epi_maps', '*%s*' % epi_sesh, '*', 'screen_boundaries*.json'))
    else:
        print("No EPI maps found for session: %s * (trying to use tmp session boundaries file)")
        epi_fpaths = glob.glob(os.path.join(rootdir, animalid, 'epi_maps', 'screen_boundaries*.json'))
    
    assert len(epi_fpaths) > 0, "No epi screen info found!"
    
    # Each epi run should have only 2 .json files (1 for each condition):
    if len(epi_fpaths) > 2:
        print("-- found %i screen boundaries files: --")
        repeat_epi_sessions = sorted(list(set( [os.path.split(s)[0] for s in epi_fpaths] )), key=natural_keys)
        for ei, repeat_epi in enumerate(sorted(repeat_epi_sessions, key=natural_keys)):
            print(ei, repeat_epi)
        selected_epi = input("Select IDX of epi run to use: ")
        epi_fpaths = [s for s in epi_fpaths if repeat_epi_sessions[selected_epi] in s]
    
    print("-- getting screen info from:", epi_fpaths)
    
    for epath in epi_fpaths:
        with open(epath, 'r') as f:
            epi = json.load(f)
        
        screen['azimuth'] = epi['screen_params']['screen_size_x_degrees']
        screen['elevation'] = epi['screen_params']['screen_size_t_degrees']
        
        if 'screen_boundaries' in epi.keys():
            if 'boundary_left_degrees' in epi['screen_boundaries'].keys():
                screen['bb_left'] = epi['screen_boundaries']['boundary_left_degrees']
                screen['bb_right'] = epi['screen_boundaries']['boundary_right_degrees']
            elif 'boundary_down_degrees' in epi['screen_boundaries'].keys():
                screen['bb_lower'] = epi['screen_boundaries']['boundary_down_degrees']
                screen['bb_upper'] = epi['screen_boundaries']['boundary_up_degrees']
        
        else:
            screen['bb_lower'] = -1*screen['elevation']/2.0
            screen['bb_upper'] = screen['elevation']/2.0
            screen['bb_left']  = -1*screen['azimuth']/2.0
            screen['bb_right'] = screen['azimuth']/2.0
            
    return screen

#%%

def visualize_fits_by_condition(fit, magratio, corrected_phase, trials_by_cond, screen,
                                use_linear=True, fit_thresh_az=0.2, fit_thresh_el=0.2,
                                data_identifier='', fov='FOV', retinoid='retinoID', output_dir='/tmp'):
    
    fig = pl.figure(figsize=(20,20))
    nr = 3
    nc = 3
    
    mean_fits = fit.mean(axis=1)
    mean_magratios = magratio.mean(axis=1)
    
    
    mean_fits_az = fit[trials_by_cond['right']].mean(axis=1)
    mean_fits_el = fit[trials_by_cond['top']].mean(axis=1)
    
    std_fits_az = fit[trials_by_cond['right']].std(axis=1)
    std_fits_el = fit[trials_by_cond['top']].std(axis=1)
    
    #%
    ax1 = pl.subplot2grid((nr, nc), (0, 0), colspan=2, rowspan=2)
    ax1.scatter(mean_fits_az, mean_fits_el, alpha=0.5, s=100)
    ax1.set_xlabel('mean fits (az)')
    ax1.set_ylabel('mean fits (el)')
    ax1.errorbar(mean_fits_az, mean_fits_el, yerr=std_fits_el, xerr=std_fits_az, fmt='.', alpha=0.5)
    
    #% Label "good" rois:
    
    #fit_thresh_el = 0.20
    #fit_thresh_az = 0.20
    good_fits = [ri for ri in mean_fits_az.index.tolist() if mean_fits_el.loc[ri] >= fit_thresh_el\
                                                             and mean_fits_az.loc[ri] >= fit_thresh_az]
    print good_fits
    
    for ri in good_fits:
        ax1.text(mean_fits_az.iloc[ri], mean_fits_el.iloc[ri], '%s' % str(ri+1))
    
    ax1.text(mean_fits_az.max(), mean_fits_el.max(), 'az thr: %.2f\nel thr: %.2f' % (fit_thresh_az, fit_thresh_el), fontsize=16)
    
    #%
    phase_std_az = corrected_phase[trials_by_cond['right']].std(axis=1)
    fit_std_az = fit[trials_by_cond['right']].std(axis=1)
    
    phase_std_el = corrected_phase[trials_by_cond['top']].std(axis=1)
    fit_std_el = fit[trials_by_cond['top']].std(axis=1)
    
    ax4 = pl.subplot2grid((nr, nc), (0, 2), colspan=1)
    ax4.scatter(fit_std_az, phase_std_az, alpha=0.5, s=50)
    ax4.set_ylabel('std PHASE')
    ax4.set_xlabel('std FIT')
    ax4.set_title('azimuth')
    
    
    ax5 = pl.subplot2grid((nr, nc), (1, 2), colspan=1)
    ax5.scatter(fit_std_el, phase_std_el, alpha=0.5, s=50)
    ax5.set_ylabel('std PHASE')
    ax5.set_xlabel('std FIT')
    ax5.set_title('elevation')
    
    for ri in good_fits:
        ax4.text(fit_std_az.iloc[ri], phase_std_az.iloc[ri], '%s' % str(ri+1))
        ax5.text(fit_std_el.iloc[ri], phase_std_el.iloc[ri], '%s' % str(ri+1))
    
    #%
    #use_linear = True
    
    mean_phase_az = corrected_phase[trials_by_cond['right']].mean(axis=1)
    mean_phase_el = corrected_phase[trials_by_cond['top']].mean(axis=1)
    
    
    # Convert phase to linear coords:
    
    az_extent = screen['bb_right'] - screen['bb_left']
    el_extent = screen['bb_upper'] - screen['bb_lower']
    
    print("AZ extent: %.2f" % az_extent)
    print("EL extend: %.2f" % el_extent)
    
    ax2 = pl.subplot2grid((nr, nc), (2, 0), colspan=1)
    ax3 = pl.subplot2grid((nr, nc), (2, 1), colspan=1)
    
    if use_linear:
        screen_left = -1*screen['azimuth']/2.
        screen_right = screen['azimuth']/2. #screen['azimuth']/2.
        screen_lower = -1*screen['elevation']/2.
        screen_upper = screen['elevation']/2. #screen['elevation']/2.
        
        linX = convert_values(mean_phase_az, screen_left, screen_right,
                              oldmax=0, oldmin=2*np.pi)  # If cond is 'right':  positive values = 0, negative values = 2pi
        linY = convert_values(mean_phase_el, screen_lower, screen_upper,
                              oldmax=0, oldmin=2*np.pi)  # If cond is 'right':  positive values = 0, negative values = 2pi
        
        # Draw azimuth value as a function of mean fit (color code by standard cmap, too)
        ax2.scatter(mean_fits, linX, c=linX*-1, cmap='nipy_spectral', vmin=screen_left, vmax=screen_right)
        ax2.set_ylim([screen_left, screen_right])
        ax2.invert_yaxis()
        ax2.set_xlabel('mean fit')
        ax2.set_ylabel('mean xpos')     
        
        # Draw BB from epi:
        ax2.axhline(y=screen['bb_right'], color='k', linestyle='--', linewidth=1)
        ax2.axhline(y=screen['bb_left'], color='k', linestyle='--', linewidth=1)
        
        # Draw elevation value as a function of mean fit (color code by standard cmap, too)
        ax3.scatter(mean_fits, linY, c=linY*-1, cmap='nipy_spectral', vmin=screen_lower, vmax=screen_upper)
        ax3.set_ylim([screen_lower, screen_upper])
        ax3.invert_yaxis()
        ax3.set_xlabel('mean fit')
        ax3.set_ylabel('mean ypos')  
        
        # Draw BB from epi:
        ax3.axhline(y=screen['bb_upper'], color='k', linestyle='--', linewidth=1)
        ax3.axhline(y=screen['bb_lower'], color='k', linestyle='--', linewidth=1)
        
        # Label good cells:
        for ri in good_fits:
            ax3.text(mean_fits.iloc[ri], linY.iloc[ri], '%s' % str(ri+1))
            ax2.text(mean_fits.iloc[ri], linX.iloc[ri], '%s' % str(ri+1))
    
    
    else:
        #%
        ax2.scatter(mean_fits, mean_phase_az, c=mean_phase_az, cmap='nipy_spectral', vmin=0, vmax=np.pi*2)
        ax2.set_ylim([0, np.pi*2])
        ax2.set_xlabel('mean fit')
        ax2.set_ylabel('average phase (AZ)')
        
        ax3.scatter(mean_fits, mean_phase_el, c=mean_phase_el, cmap='nipy_spectral', vmin=0, vmax=np.pi*2)
        ax3.set_ylim([0, np.pi*2])
        ax3.set_xlabel('mean fit')
        ax3.set_ylabel('average phase (EL)')
        
    #%
    
    ax0 = pl.subplot2grid((nr, nc), (2, 2), colspan=1)
    ax0.scatter(mean_fits, mean_magratios)
    ax0.set_xlabel('mean fit')
    ax0.set_ylabel('mean mag ratio')
    for ri in good_fits:
        ax0.text(mean_fits.iloc[ri], mean_magratios.iloc[ri], '%s' % str(ri+1))
    
    #%
    
    
    label_figure(fig, data_identifier)
    figname = 'compare_fit_and_pos_by_condition_%s_%s.png' % (fov, retinoid)
    
    pl.savefig(os.path.join(output_dir, 'visualization', figname))
    
    return fig, good_fits


#%%

#rootdir = '/n/coxfs01/2p-data'
#animalid = 'JC022'
#session = '20181005'
#fov = 'FOV1_zoom1p7x'
#run = 'retino_run1'
#retinoid = 'analysis002'

def extract_options(options):
    
    parser = optparse.OptionParser()

    parser.add_option('-D', '--root', action='store', dest='rootdir', default='/n/coxfs01/2p-data', help='data root dir (root project dir containing all animalids) [default: /nas/volume1/2photon/data, /n/coxfs01/2pdata if --slurm]')
    parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')

    # Set specific session/run for current animal:
    parser.add_option('-S', '--session', action='store', dest='session', default='', \
                      help='session dir (format: YYYMMDD_ANIMALID')
    parser.add_option('-A', '--acq', action='store', dest='acquisition', default='FOV1', \
                      help="acquisition folder (ex: 'FOV1_zoom3x') [default: FOV1]")
    parser.add_option('-R', '--run', action='store', dest='run', default='retino_run1', \
                      help="name of run dir containing tiffs to be processed (ex: gratings_phasemod_run1)")
    parser.add_option('-t', '--retinoid', action='store', dest='retinoid', default='analysis001', \
                      help="name of retino ID (roi analysis) [default: analysis001]")
    
    parser.add_option('--angular', action='store_false', dest='use_linear', default=True, \
                      help="Plot az/el coordinates in angular spce [default: plots linear coords]")
    parser.add_option('-e', '--thr-el', action='store', dest='fit_thresh_el', default=0.2, \
                      help="fit threshold for elevation [default: 0.2]")
    parser.add_option('-a', '--thr-az', action='store', dest='fit_thresh_az', default=0.2, \
                      help="fit threshold for azimuth [default: 0.2]")
    
    (options, args) = parser.parse_args(options)

    return options

#%%

options = ['-i', 'JC053', '-S', '20190220', '-A', 'FOV1']
#%%

def main(options):
#%%
    opts = extract_options(options)
    
    rootdir = opts.rootdir
    animalid = opts.animalid
    session = opts.session
    fov = opts.acquisition
    run = opts.run
    retinoid = opts.retinoid
    use_linear = opts.use_linear
    fit_thresh_az = float(opts.fit_thresh_az)
    fit_thresh_el = float(opts.fit_thresh_el) #0.2
    
    #%%
    
    run_dir = glob.glob(os.path.join(rootdir, animalid, session, '%s*' % fov, run))[0]
    fov = os.path.split(os.path.split(run_dir)[0])[-1]
    print("FOV: %s, run: %s" % (fov, run))
    retinoids_fpath = glob.glob(os.path.join(run_dir, 'retino_analysis', 'analysisids_*.json'))[0]
    with open(retinoids_fpath, 'r') as f:
        rids = json.load(f)
    roi_analyses = [r for r, rinfo in rids.items() if rinfo['PARAMS']['roi_type'] != 'pixels']
    if retinoid not in roi_analyses:
        retinoid = sorted(roi_analyses, key=natural_keys)[-1] # use most recent roi analysis
        print("Fixed retino id to most recent: %s" % retinoid)
    
    data_identifier = '|'.join([animalid, session, fov, run, retinoid])
    
    print("*** Dataset: %s ***" % data_identifier)
    
    #%%
    
    processed_dir = glob.glob(os.path.join(run_dir, 'retino_analysis', '%s*' % retinoid))[0]
    
    processed_fpaths = glob.glob(os.path.join(processed_dir, 'files', '*.h5'))
    print("Found %i processed retino runs." % len(processed_fpaths))
    
    conditions_fpath = glob.glob(os.path.join(run_dir, 'paradigm', 'files', '*.json'))[0]


    #%%
    
    # Comine all trial data into data frames:
    fit, magratio, phase, trials_by_cond = trials_to_dataframes(processed_fpaths, conditions_fpath)
    print fit.head()
    print trials_by_cond
    
    #%%
    
    threshold = 0.2
    fig = plot_signal_fits_by_roi(fit, magratio, threshold=threshold, data_identifier=data_identifier,
                                  fov=fov, retinoid=retinoid, output_dir=processed_dir)
    
    screen = get_screen_info(animalid, session, rootdir=rootdir)
    
    
    #%%
    
    # Correct phase to wrap around:
        
    corrected_phase = phase.copy()
    
    corrected_phase[phase<0] =- phase[phase<0]
    corrected_phase[phase>0] = (2*np.pi) - phase[phase>0]

    #%% 
    
    # -----------------------------------------------------------------------------
    # Visualize FITS by condition:
    # -----------------------------------------------------------------------------
#    use_linear = True
#    fit_thresh_az = 0.2
#    fit_thresh_el = 0.2
    
    fig, good_fits = visualize_fits_by_condition(fit, magratio, corrected_phase, trials_by_cond, screen, 
                                                 use_linear=use_linear,
                                                 fit_thresh_az=fit_thresh_az, fit_thresh_el=fit_thresh_el,
                                                 data_identifier=data_identifier, fov=fov, retinoid=retinoid,
                                                 output_dir=processed_dir)
    
    #%%


if __name__ == '__main__':
    main(sys.argv[1:])


    

#%%


