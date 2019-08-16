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
from pipeline.python.classifications import responsivity_stats as estats

from pipeline.python.retinotopy import fit_2d_rfs as fitrf
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
   
#
#    parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', 
#                      help='Animal ID')

    # Set specific session/run for current animal:
    parser.add_option('-f', '--fov-type', action='store', dest='fov_type', default='zoom2p0x', 
                      help="fov type (default: zoom2p0x)")
    
    parser.add_option('-t', '--traceid', action='store', dest='traceid', default='traces001', 
                      help="traceid (default: traces001)")
    
    parser.add_option('-T', '--trace-type', action='store', dest='trace_type', default='corrected', 
                      help="trace type (default: corrected, for traces and calculating stats)")
    parser.add_option('-r', '--response-type', action='store', dest='response_type', default='dff', 
                      help="trace type (default: dff, stat to compare)")
    parser.add_option('-R', '--response-test', action='store', dest='responsive_test', default='ROC', 
                      help="Responsivity test (default: ROC)")
    
    
    parser.add_option('-x', '--exclude', action='append', dest='blacklist', default=['20190514', '20190530'], nargs=1,
                      help="session to exclude (default includes 20190514, 20190530)")
    
    parser.add_option('-v', '--area', action='append', dest='visual_areas', default=[], nargs=1,
                      help="visual areas (default = V1, Lm, Li, if not provided)")
    parser.add_option('-s', '--state', action='store', dest='state', default='awake', 
                      help="Behavior state of rats (default: awake)")
    
    parser.add_option('--new', action='store_true', dest='create_new', default=False, 
                      help="Create all session objects from scratch")

    parser.add_option('-S', '--stat', action='store', dest='stats', default='responsivity', 
                      help="Stats to run across datasets (default: responsivity)")
    parser.add_option('-n', '--n-processes', action='store', dest='n_processes', default=1, 
                      help="N processes (default: 1)")
 
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
                        session_fpath = os.path.join(fov_dir, 'summaries', 'sessiondata.pkl')    
    
                        if os.path.exists(session_fpath):
                            with open(session_fpath, 'rb') as f:
                                S = pkl.load(f)
                        else:
                            print("Creating new session object...") #% (animalid, session_name))
                            S = util.Session(animalid, session_str, '%s_%s' % (fov_str, fov_type), 
                                             visual_area=visual_area, state=state,
                                             rootdir=rootdir)
                            with open(session_fpath, 'wb') as f:
                                pkl.dump(S, f, protocol=pkl.HIGHEST_PROTOCOL)
                            print("... created session object!")
    
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

#%%

options = ['-t', 'traces001']

#%%
def main(options):
    
    #%%
    optsE = extract_options(options)
    
    # Create output aggregate dir:
    #aggregate_dir = '/n/coxfs01/julianarhee/aggregate-visual-areas'
    aggregate_dir = optsE.aggregate_dir
                
    dataset_info_fpath = os.path.join(aggregate_dir, 'dataset_info.pkl')
    if os.path.exists(dataset_info_fpath) and optsE.create_new is False:
        with open(dataset_info_fpath, 'rb') as f:
            sessiondata = pkl.load(f)
    else:
        
        sessiondata = aggregate_session_info(traceid=optsE.traceid, trace_type=optsE.trace_type, 
                                               state=optsE.state, fov_type=optsE.fov_type, 
                                               visual_areas=optsE.visual_areas,
                                               blacklist=optsE.blacklist, 
                                               rootdir=optsE.rootdir)
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
                      
    #%%
    
    sessiondata.describe()        
    sessions_by_animal = sessiondata.groupby(['animalid', 'session'])['fov'].unique()

    responsive_dir = os.path.join(aggregate_dir, 'responsivity')
    if not os.path.exists(responsive_dir):
        os.makedirs(responsive_dir)
    
    dtype_str = '%s-%s-%s-%s' % (optsE.traceid, optsE.trace_type, optsE.response_type, optsE.responsive_test)
    
    #%%
    stats = optsE.stats
    n_processes= int(optsE.n_processes)

    if stats=='responsivity':
        '''
        Responsivity stats:
            - RF comparisons (if relevant)
            - distribution of peak dF/F values
            - comparison of N responsive cells (for RFs, this is N fit rois)
            - Fits 2d-gaussian for RF data using default params
        '''
#%%
        curr_output_dir = os.path.join(responsive_dir, dtype_str)
        if not os.path.exists(curr_output_dir):
            os.makedirs(curr_output_dir)
        print(curr_output_dir)
        
        emptystats = {}
        for animalid in sessiondata['animalid'].unique():
            session_list = sessions_by_animal[animalid].index.tolist()
            for session in session_list:
                fovs = sessions_by_animal[animalid][session]

                for fov in fovs:
                    nostats = estats.visualize_session_stats(animalid, session, fov, 
                                                             create_new=True, altdir=curr_output_dir,
                                                             response_type=optsE.response_type, responsive_test=optsE.responsive_test,
                                                             traceid=optsE.traceid, trace_type=optsE.trace_type)

                    dset_key = '_'.join([animalid, session, fov])
                    emptystats[dset_key] = nostats

        for k, checklist in emptystats.items():
            if len(checklist) == 0:
                emptystats.pop(k)

        error_fpath = os.path.join(responsive_dir, 'check_stats.json')
        with open(error_fpath, 'w') as f:
            json.dump(emptystats, f, indent=4, sort_keys=True)


    #%%
    elif stats=='gratings':
        '''
        # Get responsivity stats:
        responsive_test = 'ROC'
        responsive_thr = 0.05

        # Tuning params:
        n_bootstrap_iters = 1000
        n_resamples = 60
        n_intervals_interp = 3
        '''

        traceid = optsE.traceid
        rootdir = optsE.rootdir

        dsets = sessiondata[sessiondata['experiment']=='gratings']
        #%%
        tuning_counts = {}

        for animalid in dsets['animalid'].unique():
            session_list = dsets[dsets['animalid']==animalid]['session'].unique()
            for session in session_list:
                fovs = dsets[(dsets['animalid']==animalid) & (dsets['session']==session)]['fov'].unique()
                for fov in fovs:
                    exp = util.Gratings(animalid, session, fov, traceid=traceid, rootdir=rootdir)
                    fitdf, fitparams, fitdata = exp.get_tuning(create_new=False, n_processes=n_processes)
                    fitdf, goodfits = exp.evaluate_fits(fitdf, fitparams, fitdata)
                    skey = '_'.join([animalid, session, fov])
                    tuning_counts[skey] = goodfits
                    del fitdf
                    del fitparams
                    del fitdata
                    del exp
                    
        with open(os.path.join(aggregate_session_dir, 'gratings_summary.pkl')):
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
