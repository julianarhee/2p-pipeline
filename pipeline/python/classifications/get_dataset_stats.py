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

rootdir = '/n/coxfs01/2p-data'
fov_type = 'zoom2p0x'

all_rats = [os.path.split(os.path.split(f)[0])[-1] for f in glob.glob(os.path.join(rootdir, 'JC*', 'sessionmeta.json'))]



#%% Get list of all datasets
visual_areas = ['V1', 'Lm', 'Li']
state = 'awake'
traceid = 'traces001'
trace_type = 'corrected'
#sessiondatapaths = dict((visual_area, {}) for visual_area in visual_areas)
sessiondata = []
blacklist = ['20190514', '20190530']

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
sessiondata.describe()

experiment_types = sorted(sessiondata['experiment'].unique(), key=natural_keys)
experiment_ids = dict((exp, i) for i, exp in enumerate(experiment_types))
animal_names = sorted(sessiondata['animalid'].unique(), key=natural_keys)
animal_ids = dict((animal, i) for i, animal in enumerate(animal_names))

sessiondata['exp_no'] = [int(experiment_ids[exp]) for exp in sessiondata['experiment']]
sessiondata['animal_no'] = [int(animal_ids[animal]) for animal in sessiondata['animalid']]


#%%

n_animals_per = sessiondata.groupby(['visual_area'])['animalid'].unique()
    
group_by_animal = [len(s) for s in n_animals_per] #.reset_index().pivot(columns='animalid', index='visual_area', values=0)
print(group_by_animal)
n_animals_per

#df_plot = sessiondata.groupby(['visual_area', 'experiment']).size().reset_index().pivot(columns='visual_area', index='experiment', values=0)
group_by_area = sessiondata.groupby(['visual_area', 'experiment']).size().reset_index().pivot(columns='experiment', index='visual_area', values=0)
group_by_experiment = sessiondata.groupby(['visual_area', 'experiment']).size().reset_index().pivot(columns='visual_area', index='experiment', values=0)
#group_by_animal = sessiondata.groupby(['visual_area', 'animalid']).unique().reset_index().pivot(columns='animalid', index='visual_area', values=0)

#group_by_animal

#%%

# Create output figure dir:

outputdir = '/n/coxfs01/julianarhee/aggregate-visual-areas'
datasetdir = os.path.join(outputdir, 'dataset_info')
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
            
            
            
            
            
            
            
            
            
            
            
            