#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 21:16:54 2019

@author: julianarhee
"""

import os
import glob
import json
import h5py
import copy
import optparse
import sys
import matplotlib as mpl
mpl.use('agg')

import numpy as np
import pandas as pd
import pylab as pl
import seaborn as sns
import cPickle as pkl
import matplotlib.gridspec as gridspec

from pipeline.python.classifications import experiment_classes as util
from pipeline.python.classifications import test_responsivity as resp
from pipeline.python.utils import label_figure, natural_keys, convert_range
from pipeline.python.classifications import responsivity_stats as respstats
from pipeline.python.classifications.analyze_retino_structure import do_rf_fits_and_evaluation


from pipeline.python.retinotopy import fit_2d_rfs as fitrf
#from pipeline.python.classifications import bootstrap_fit_tuning_curves as osi
from pipeline.python.classifications import bootstrap_osi as osi

from matplotlib.patches import Ellipse, Rectangle

from shapely.geometry.point import Point
from shapely import affinity
from matplotlib.patches import Polygon

from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import MaxNLocator

import matplotlib_venn as mpvenn
import itertools




def extract_options(options):
    parser = optparse.OptionParser()

    # PATH opts:
    parser.add_option('-D', '--root', action='store', dest='rootdir', default='/n/coxfs01/2p-data', 
                      help='root project dir containing all animalids [default: /n/coxfs01/2pdata]')
    parser.add_option('-o', '--aggregate-dir', action='store', dest='aggregate_dir', default='/n/coxfs01/julianarhee/aggregate-visual-areas', 
                      help='output dir for saving aggregated data and figures [default: /n/coxfs01/julianarhee/aggregate-visual-areas]')
   
    # Data selection parameters
    parser.add_option('-F', '--fov-type', action='store', dest='fov_type', default='zoom2p0x', 
                      help="fov type (default: zoom2p0x)")
    parser.add_option('-t', '--traceid', action='store', dest='traceid', default='traces001', 
                      help="traceid (default: traces001)")
    parser.add_option('-x', '--exclude', action='append', dest='blacklist', default=['20190514', '20190530'], nargs=1,
                      help="session to exclude (default includes 20190514, 20190530)")
    parser.add_option('-v', '--area', action='append', dest='visual_areas', default=[], nargs=1,
                      help="visual areas (default = V1, Lm, Li, if not provided)")
    parser.add_option('-a', '--state', action='store', dest='state', default='awake', 
                      help="Behavior state of rats (default: awake)")
    
    # Trace type parameters
    parser.add_option('-d', '--trace-type', action='store', dest='trace_type', default='corrected', 
                      help="trace type (default: corrected, for traces and calculating stats)")
    parser.add_option('-m', '--response-type', action='store', dest='response_type', default='dff', 
                      help="Metric to use for comparing responses per trial (default: dff, stat to compare)")


    choices_resptest = ('ROC','nstds', None, 'None')
    default_resptest = None
    
    parser.add_option('--response-test', type='choice', choices=choices_resptest,
                      dest='responsive_test', default=default_resptest, 
                      help="Stat to get. Valid choices are %s. Default: %s" % (choices_resptest, str(default_resptest)))
    parser.add_option('--response-thr', action='store', dest='responsive_thr', default=0.05, 
                      help="Responsivity threshold (default: 0.05)")
    parser.add_option('-s', '--n-stds', action='store', dest='n_stds', default=2.5, 
                      help="n stds above/below baseline to count frames, if test=nstds (default: 2.5)")    
    
    # Processing parameters
    parser.add_option('--new', action='store_true', dest='create_new', default=False, 
                      help="Create all session objects from scratch")
    parser.add_option('-n', '--n-processes', action='store', dest='n_processes', default=1, 
                      help="N processes (default: 1)")
    parser.add_option('--plot-rois', action='store_true', dest='plot_rois', default=False, 
                      help="Plot fit results for each roi (takes longer)")
    
    
    # Bootstrap parameters
    parser.add_option('-b', '--n-boot', action='store', dest='n_bootstrap_iters', type='int', default=1000, \
                      help="N bootstrap iterations for evaluating RF param fits (default: 1000)")
    parser.add_option('-k', '--n-resamples', action='store', dest='n_resamples', type='int', default=20, \
                      help="N trials to sample with replacement (default: 20)")
    
    # Gratings-specific parameters
    parser.add_option('-p', '--interp', action='store', dest='n_intervals_interp', default=3, 
                      help="[gratings only] N intervals to interp between tested angles (default: 3)")
    parser.add_option('-G', '--goodness-thr', action='store', dest='goodness_thr', default=0.66, 
                      help="[gratings only] Goodness-of-fit threshold (default: 0.66)")
    parser.add_option('-c', '--min-configs', action='store', dest='min_cfgs_above', default=2, 
                      help="[n_stds only] Min N configs in which min-n-frames threshold is met, if responsive_test=nstds (default: 2)")   

    # RF-specific parameters
    parser.add_option('--sigma', action='store', dest='sigma_scale', default=2.35, 
                      help="[rfs only] Sigma scale for RF 2d gaus fits (default: 2.35 for FWHM)")
    parser.add_option('--ci', action='store', dest='ci', default=0.95, 
                      help="[rfs only] CI for bootstrapping RF params (default: 0.95 for 95% CI)")
    parser.add_option('--pixels', action='store_false', dest='transform_fov', default=True, 
                      help="[rfs only] flag to not transform FOV from pixels to microns")
    parser.add_option('-r', '--rf-thr', action='store', dest='rf_fit_thr', default=0.5, 
                      help="[rfs only] Threshold for coeff. of determination for RF fits (default: 0.5)")
                        

    choices_stat = ('gratings','responsivity', 'rfs')
    default_stat = 'responsivity'
    
    parser.add_option('-S', '--stat', type='choice', choices=choices_stat,
                      dest='stats', default=default_stat, 
                      help="Stat to get. Valid choices are %s. Default: %s" % (choices_stat, default_stat))
 
    (options, args) = parser.parse_args(options)
    
    if len(options.visual_areas) == 0:
        options.visual_areas = ['V1', 'Lm', 'Li']

    return options



#%% Get list of all datasets





def aggregate_session_info(traceid='traces001', trace_type='corrected', 
                           state='awake', fov_type='zoom2p0x', 
                           visual_areas=['V1', 'Lm', 'Li'],
                           blacklist=['20190426', '20190514', '20190530'], 
                           rootdir='/n/coxfs01/2p-data'):
                           #aggregate_dir='/n/coxfs01/julianarhee/aggregate-visual-areas'):

    all_rats = [os.path.split(os.path.split(f)[0])[-1] \
                for f in glob.glob(os.path.join(rootdir, 'JC*', 'sessionmeta.json'))]
    
    sessiondata = []
    dcounter = 0
    for animalid in all_rats:
    
            
        # Get metadata for this rat's sessions:
        with open(os.path.join(rootdir, animalid, 'sessionmeta.json'), 'r') as f:
            sessionmeta = json.load(f)
        
        # Get session data paths, if exist:
        for visual_area in visual_areas:
            curr_session_list = [str(k) for k, v in sessionmeta.items()\
                                 if v['state'] == state and v['visual_area'] == visual_area]
            
            if len(curr_session_list) > 0:
                for s in curr_session_list:
    
                    session_str = s.split('_')[0]
                    fov_str = s.split('_')[-1]
                    if session_str in blacklist:
                        continue
                        
                    found_fovs = glob.glob(os.path.join(rootdir, animalid, session_str, '%s*' % fov_str))
                    for fov_dir in found_fovs:
#                        session_fpath = os.path.join(fov_dir, 'summaries', 'sessiondata.pkl')    
#    
#                        if os.path.exists(session_fpath):
#                            with open(session_fpath, 'rb') as f:
#                                S = pkl.load(f)
#                        else:
                        print("Creating new session object...") #% (animalid, session_name))
                        S = util.Session(animalid, session_str, '%s_%s' % (fov_str, fov_type), 
                                         visual_area=visual_area, state=state,
                                         rootdir=rootdir)
#                            with open(session_fpath, 'wb') as f:
#                                pkl.dump(S, f, protocol=pkl.HIGHEST_PROTOCOL)
#                            print("... created session object!")
#    
                        experiment_list = S.get_experiment_list(traceid=traceid, trace_type=trace_type)
    
                        #sessiondatapaths[visual_area][animalid].update({s: experiment_list})
    
                        for e in experiment_list:
                            if 'dyn' in e:
                                continue
                            sessiondata.append(pd.DataFrame({'visual_area': visual_area, 
                                                               'animalid': animalid, 
                                                               'experiment': e,
                                                               'session': session_str,
                                                               'fov': '%s_%s' % (fov_str, fov_type)}, index=[dcounter]) )
                            dcounter += 1
    
            else:
                print("[%s] %s - skipping" % (animalid, visual_area))
    
    sessiondata = pd.concat(sessiondata, axis=0)
    
    return sessiondata


def get_dataset_info(aggregate_dir='/n/coxfs01/2p-data/aggregate-visual-areas',
                      traceid='traces001', trace_type='corrected', state='awake',
                      fov_type='zoom2p0x', visual_areas=['V1', 'Lm', 'Li'],
                      blacklist = ['20190514', '20190530'], rootdir='/n/coxfs01/2p-data', create_new=False):
    dataset_info_fpath = os.path.join(aggregate_dir, 'dataset_info.pkl')
    if os.path.exists(dataset_info_fpath) and create_new is False:
        with open(dataset_info_fpath, 'rb') as f:
            sessiondata = pkl.load(f)
    else:
        
        sessiondata = aggregate_session_info(traceid=traceid, trace_type=trace_type, 
                                               state=state, fov_type=fov_type, 
                                               visual_areas=visual_areas,
                                               blacklist=blacklist, 
                                               rootdir=rootdir)
                                               #aggregate_dir=optsE.aggregate_dir)
                
        experiment_types = sorted(sessiondata['experiment'].unique(), key=natural_keys)
        experiment_ids = dict((exp, i) for i, exp in enumerate(experiment_types))
        animal_names = sorted(sessiondata['animalid'].unique(), key=natural_keys)
        animal_ids = dict((animal, i) for i, animal in enumerate(animal_names))
        
        sessiondata['exp_no'] = [int(experiment_ids[exp]) for exp in sessiondata['experiment']]
        sessiondata['animal_no'] = [int(animal_ids[animal]) for animal in sessiondata['animalid']]

        #%
        with open(dataset_info_fpath, 'wb') as f:
            pkl.dump(sessiondata, f, protocol=pkl.HIGHEST_PROTOCOL)
            
    return sessiondata

#%%

options = ['-t', 'traces001', '--response-test', 'nstds', '--response-thr', '10', '-S', 'responsivity']

#%%
def main(options):
    
    #%%
    optsE = extract_options(options)
    
    # Create output aggregate dir:
    #aggregate_dir = '/n/coxfs01/julianarhee/aggregate-visual-areas'
    aggregate_dir = optsE.aggregate_dir
    

    #%%
    sessiondata = get_dataset_info(aggregate_dir=aggregate_dir, traceid=optsE.traceid,
                                   fov_type=optsE.fov_type, state=optsE.state)
    
    sessiondata.describe()        
    sessions_by_animal = sessiondata.groupby(['animalid', 'session'])['fov'].unique()

    #%%
    stats = optsE.stats
    n_processes= int(optsE.n_processes)
    traceid = optsE.traceid
    rootdir = optsE.rootdir
    n_bootstrap_iters = int(optsE.n_bootstrap_iters)
    n_resamples = int(optsE.n_resamples)
    n_intervals_interp = int(optsE.n_intervals_interp)
    responsive_test = optsE.responsive_test
    if responsive_test == 'None':
        responsive_test = None
    responsive_thr = float(optsE.responsive_thr)
    n_stds = float(optsE.n_stds)
    plot_rois = optsE.plot_rois

    goodness_thr = float(optsE.goodness_thr)
    sigma_scale = float(optsE.sigma_scale)
    ci = float(optsE.ci)
    min_cfgs_above = int(optsE.min_cfgs_above)
    
    transform_fov = optsE.transform_fov
    create_new = optsE.create_new
    response_type = optsE.response_type
    trace_type = optsE.trace_type
    
    if stats=='responsivity':
        '''
        Responsivity stats:
            - RF comparisons (if relevant)
            - distribution of peak dF/F values
            - comparison of N responsive cells (for RFs, this is N fit rois)
            - Fits 2d-gaussian for RF data using default params
        '''
#%%
    
        #dtype_str = '%s-%s-%s-%s' % (optsE.traceid, optsE.trace_type, optsE.response_type, optsE.responsive_test)
        stats_desc = util.get_stats_desc(traceid=optsE.traceid,
                                              trace_type= trace_type,
                                              response_type = response_type,
                                              responsive_test = responsive_test,
                                              responsive_thr = responsive_thr,
                                              n_stds = n_stds)
        
        aggregate_stats_dir = os.path.join(aggregate_dir, 'responsivity', stats_desc)
        if not os.path.exists(aggregate_stats_dir):
            os.makedirs(aggregate_stats_dir)
        print(aggregate_stats_dir)
        
        emptystats = {}

        for (visual_area, animalid, session, fov), g in sessiondata.groupby(['visual_area', 'animalid', 'session', 'fov']):
            skey = '%s_%s' % (visual_area, '-'.join([animalid, session, fov]))
            #print skey
            nostats = respstats.visualize_session_stats(animalid, session, fov, 
                                                     create_new=create_new, altdir=aggregate_stats_dir, 
                                                     traceid=traceid, trace_type=trace_type,
                                                     plot_rois=plot_rois,
                                                     response_type=response_type, 
                                                     responsive_test=responsive_test,
                                                     responsive_thr=responsive_thr,
                                                     n_stds = n_stds)
            
            dset_key = '_'.join([animalid, session, fov])
            emptystats[dset_key] = nostats

        for k, checklist in emptystats.items():
            if len(checklist) == 0:
                emptystats.pop(k)

        error_fpath = os.path.join(aggregate_stats_dir, 'check_stats.json')
        with open(error_fpath, 'w') as f:
            json.dump(emptystats, f, indent=4, sort_keys=True)


    #%%
    else:
        '''
        Get tuning / fit stats:
            
        GRATINGS:
            responsive_test = 'nstds'
            responsive_thr = 10 (N frames above baseline*n_std to count as responsive)
            n_stds = 2.5
            
            # Tuning params:
            n_bootstrap_iters = 1000
            n_resamples = 20
            n_intervals_interp = 3
            min_cfgs_above = 2 (# of stim configs (orientations) that meet reposnsive_thr for cell to count as responsive to a given stim-config)
            
        RFS:
            responsive_test = None
            responsive_thr = 0.05
    
            # Tuning params:
            n_bootstrap_iters = 1000
            n_resamples = 10
        '''
        if stats == 'rfs':
            dsets = sessiondata[sessiondata['experiment'].isin(['rfs', 'rfs10'])]
        else: 
            dsets = sessiondata[sessiondata['experiment']==stats]
        #%%
        tuning_counts = {}

        for (visual_area, animalid, session, fov), g in dsets.groupby(['visual_area', 'animalid', 'session', 'fov']):
            skey = '%s_%s' % (visual_area, '-'.join([animalid, session, fov]))
            if stats=='gratings':
                n_resamples = 20
                plot_rois = True
                exp = util.Gratings(animalid, session, fov, traceid=traceid, rootdir=rootdir)
                bootresults, fitparams = exp.get_tuning(create_new=create_new, n_processes=n_processes,
                                                           responsive_test=responsive_test, responsive_thr=responsive_thr,
                                                           n_stds=n_stds, min_cfgs_above=min_cfgs_above,
                                                           n_bootstrap_iters=n_bootstrap_iters, n_resamples=n_resamples,
                                                           n_intervals_interp=n_intervals_interp, make_plots=plot_rois)
                
                rmetrics, goodrois = exp.evaluate_fits(bootresults, fitparams, goodness_thr=goodness_thr, 
                                                       make_plots=plot_rois, rootdir=rootdir)
                
                tuning_counts[skey] = goodrois if goodrois is not None else 0
                del bootresults
                del fitparams
                del rmetrics
                
            elif stats == 'rfs':
                n_resamples = 10
                fit_thr = 0.05
                # fit_thr = float(optsE.rf_fit_thr)
                rfnames = g['experiment'].unique()
                print("Found %i rf experiments." % len(rfnames))
                for rfname in rfnames: 
                    deviants = do_rf_fits_and_evaluation(animalid, session, fov, rfname=rfname,
                                          traceid=traceid, response_type=response_type, fit_thr=fit_thr,
                                          n_bootstrap_iters=n_bootstrap_iters, n_resamples=n_resamples, ci=ci,
                                          transform_fov=transform_fov, plot_boot_distns=plot_rois, sigma_scale=sigma_scale,
                                          n_processes=n_processes, create_new=plot_rois, rootdir=rootdir)
                
                    tuning_counts['%s_%s' % (skey, rfname)] = deviants
                
        
        if stats == 'gratings':
            #fit_desc = osi.get_fit_desc(response_type, responsive_test, responsive_thr, n_stds)
            fit_desc = osi.get_fit_desc(response_type=response_type, responsive_test=responsive_test, n_stds=n_stds,
                                    responsive_thr=responsive_thr, n_bootstrap_iters=n_bootstrap_iters,
                                    n_resamples=n_resamples)
            aggregate_stats_dir = os.path.join(aggregate_dir, 'orientation-tuning', '%s-%s' % (traceid, fit_desc))
        elif stats == 'rfs':
            fit_desc = fitrf.get_fit_desc(response_type=response_type)
            aggregate_stats_dir = os.path.join(aggregate_dir, 'receptive-fields', '%s-%s' % (traceid, fit_desc))
            
        if not os.path.exists(aggregate_stats_dir):
            os.makedirs(aggregate_stats_dir)
 
        with open(os.path.join(aggregate_stats_dir, 'aggregated_stats.pkl'), 'wb') as f:
            pkl.dump(tuning_counts, f, protocol=pkl.HIGHEST_PROTOCOL)
       
        print("DONE!")                

        
    #%%
            
if __name__ == '__main__':
    main(sys.argv[1:])
    
            
            
            
            
#%%

#animalid = 'JC084'
#session = '20190522'
#fov = 'FOV1_zoom2p0x'
#create_new=True
#aggregate_session_dir=None
