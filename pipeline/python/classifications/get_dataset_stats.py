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

import numpy as np
import pandas as pd
import pylab as pl
import seaborn as sns
import cPickle as pkl
import matplotlib.gridspec as gridspec

from pipeline.python.classifications import utils as util
from pipeline.python.classifications import test_responsivity as resp
from pipeline.python.utils import label_figure, natural_keys, convert_range
from pipeline.python.classifications import run_experiment_stats as stats

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
   

    parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', 
                      help='Animal ID')

    # Set specific session/run for current animal:
    parser.add_option('-f', '--fov-type', action='store', dest='fov_type', default='zoom2p0x', 
                      help="fov type (default: zoom2p0x)")
    
    parser.add_option('-t', '--traceid', action='store', dest='traceid', default='traces001', 
                      help="traceid (default: traces001)")
    
    parser.add_option('-T', '--trace_type', action='store', dest='trace_type', default='corrected', 
                      help="trace type (default: dff, for calculating mean stats)")
    
    parser.add_option('-x', '--exclude', action='append', dest='blacklist', default=['20190514', '20190530'], nargs=1,
                      help="session to exclude (default includes 20190514, 20190530)")
    
    parser.add_option('-v', '--area', action='append', dest='visual_areas', default=[], nargs=1,
                      help="visual areas (default = V1, Lm, Li, if not provided)")
    parser.add_option('-s', '--state', action='store', dest='state', default='awake', 
                      help="Behavior state of rats (default: awake)")
    
    parser.add_option('--new', action='store_true', dest='create_new', default=False, 
                      help="Create all session objects from scratch")

    (options, args) = parser.parse_args(options)
    
    if len(options.visual_areas) == 0:
        options.visual_areas = ['V1', 'Lm', 'Li']

    return options

#%% Get list of all datasets





def aggregate_session_info(traceid='traces001', trace_type='corrected', 
                           state='awake', fov_type='zoom2p0x', 
                           visual_areas=['V1', 'Lm', 'Li'],
                           blacklist=['20190514', '20190530'], 
                           rootdir='/n/coxfs01/2p-data',
                           aggregate_dir='/n/coxfs01/julianarhee/aggregate-visual-areas'):

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

def main(options):
    
    optsE = extract_options(options)
    
    sessiondata = aggregate_session_info(traceid=optsE.traceid, trace_type=optsE.trace_type, 
                                           state=optsE.state, fov_type=optsE.fov_type, 
                                           visual_areas=optsE.visual_areas,
                                           blacklist=optsE.blacklist, 
                                           rootdir=optsE.rootdir,
                                           aggregate_dir=optsE.aggregate_dir)
    
    sessiondata.describe()
    
    experiment_types = sorted(sessiondata['experiment'].unique(), key=natural_keys)
    experiment_ids = dict((exp, i) for i, exp in enumerate(experiment_types))
    animal_names = sorted(sessiondata['animalid'].unique(), key=natural_keys)
    animal_ids = dict((animal, i) for i, animal in enumerate(animal_names))
    
    sessiondata['exp_no'] = [int(experiment_ids[exp]) for exp in sessiondata['experiment']]
    sessiondata['animal_no'] = [int(animal_ids[animal]) for animal in sessiondata['animalid']]

    
    #%%
    
    # Create output figure dir:
    
    #aggregate_dir = '/n/coxfs01/julianarhee/aggregate-visual-areas'
    datasetdir = os.path.join(optsE.aggregate_dir, 'dataset_info')
    if not os.path.exists(datasetdir):
        os.makedirs(datasetdir)
        
    sessions_by_animal = sessiondata.groupby(['animalid', 'session'])['fov'].unique()
    
    aggregate_session_dir = os.path.join(datasetdir, 'session_stats')
    if not os.path.exists(aggregate_session_dir):
        os.makedirs(aggregate_session_dir)
        
    #%%
        
    for animalid in sessiondata['animalid'].unique():
        session_list = sessions_by_animal[animalid].index.tolist()
        for session in session_list:
            fovs = sessions_by_animal[animalid][session]
    
            for fov in fovs:
                stats.visualize_session_stats(animalid, session, fov, create_new=False, altdir=aggregate_session_dir)
                
                
                
            
if __name__ == '__main__':
    main(sys.argv[1:])
    
            
            
            
            
            
            