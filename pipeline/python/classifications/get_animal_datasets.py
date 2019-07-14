#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 13:54:28 2019

@author: julianarhee
"""

import os
import glob
import json 

import numpy as np
import cPickle as pkl

from pipeline.python.utils import natural_keys
from pipeline.python.classifications import utils as util

rootdir = '/n/coxfs01/2p-data'

animalids = ['JC076', 'JC078', 'JC080', 'JC083', 'JC084', 'JC085', 'JC090', 'JC091', 'JC097', 'JC099']

animalid = 'JC097'
fov_type = 'zoom2p0x'

#%%
class MetaData():
    def __init__(self, animalid, rootdir='/n/coxfs01/2p-data'):
        self.animalid = animalid
        self.anesthetized_session_list = []
        self.sessions = {}
    
    def get_sessions(self, fov_type='zoom2p0x', create_new=False, rootdir='/n/coxfs01/2p-data'):

        # Check if anesthetized info / visual area info stored:
        create_meta = False
        meta_info_file = os.path.join(rootdir, animalid, 'sessionmeta.json')
        if os.path.exists(meta_info_file):
            try:
                with open(meta_info_file, 'r') as f:
                    meta_info = json.load(f)
            except Exception as e:
                print("...creating new meta file")
                create_meta = True
        else:
            create_new = True
            create_meta = True
            
        if create_meta:
            meta_info = {}
                
        session_paths = sorted(glob.glob(os.path.join(rootdir, animalid,  '*', 'FOV*_%s' % fov_type)), key=natural_keys)
        print("Found %i acquisitions." % len(session_paths))
        for si, session_path in enumerate(session_paths):
            session_name = os.path.split(session_path.split('/FOV')[0])[-1]
            fov_name = os.path.split(session_path)[-1]
            print("[%s]: %s - %s" % (animalid, session_name, fov_name))
            skey = '%s_%s' % (session_name, fov_name.split('_')[0])

            output_dir = os.path.join(rootdir, animalid, session_name, fov_name, 'summaries')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            session_outfile = os.path.join(output_dir, 'sessiondata.pkl')                
                
            if not create_new:
                try:
                    assert os.path.exists(session_outfile), "... session object does not exist, creating new."
                    print("... loading session object...") #% (animalid, session_name))
                    with open(session_outfile, 'rb') as f:
                        S = pkl.load(f)
                        assert 'visual_area' in dir(S), "... No visual area found, creating new."
                except Exception as e:
                    print e
                    create_new = True
                    
            if skey not in meta_info.keys():
                user_input = raw_input('Was this session anesthetized? [Y/n]')
                if user_input == 'Y':
                    #self.anesthetized_session_list.append(session_name)
                    state = 'anesthetized'
                else:
                    state = 'awake'
                visual_area = raw_input('Enter visual area recorded: ')
                meta_info.update({skey: {'state': state,
                                        'visual_area': visual_area}})
            else:
                state = meta_info[skey]['state']
                visual_area = meta_info[skey]['visual_area']
                

            if create_new:
                print("... creating new session object...") #% (animalid, session_name))
                    
                S = util.Session(animalid, session_name, fov_name, visual_area=visual_area, rootdir=rootdir)
                #S.load_data(traceid=traceid, trace_type='corrected')
                # Save session data object
                with open(session_outfile, 'wb') as f:
                    pkl.dump(S, f, protocol=pkl.HIGHEST_PROTOCOL)
                
            self.sessions[skey] = (S)
            if state == 'anesthetized' and skey not in self.anesthetized_session_list:
                self.anesthetized_session_list.append(skey)
                
            with open(meta_info_file, 'w') as f:
                json.dump(meta_info, f, sort_keys=True, indent=4)
            
            
    def load_experiments(self, experiment, state=None, visual_area=None,
                         traceid='traces001', trace_type='corrected', load_raw=False,
                         rootdir='/n/coxfs01/2p-data'):
        loaded_experiments = {}
        assert len(self.sessions.keys()) > 0, "** no sessions found! **"
        for skey, sobj in self.sessions.items():
            
            if state == 'awake' and skey in self.anesthetized_session_list:
                continue
            if state == 'anesthetized' and skey not in self.anesthetized_session_list:
                continue
            if visual_area is not None and sobj.visual_area != visual_area:
                continue
            
            if experiment == 'rfs' and int(sobj.session) < 20190511:
                experiment_name = 'gratings'
            else:
                experiment_name = experiment
                
            if load_raw:
                expdict = sobj.load_data(experiment=experiment_name, traceid=traceid, trace_type=trace_type)
            else:
                expdict = sobj.get_grouped_stats(experiment_type=experiment_name, responsive_thr=0.01, responsive_test='ROC',
                          traceid=traceid, trace_type=trace_type, rootdir=rootdir)
            
            loaded_experiments[skey] = expdict
        
        return loaded_experiments
            
#%%


#%%

import seaborn as sns
import pylab as pl
import pandas as pd

import itertools

palette = itertools.cycle(sns.color_palette())

rstats = {}
for animalid in animalids:
    A = MetaData(animalid)
    A.get_sessions(fov_type=fov_type, create_new=True)
    
    curr_sessions = A.load_experiments('gratings', state='awake', visual_area='V1')
    
    rstats[animalid] = curr_sessions
    
    for si, (skey, edata) in enumerate(curr_sessions.items()):
        print si
        fitdf = edata.fits
        avg_rfs = edata.fits[['sigma_x', 'sigma_y']].mean(axis=1)
        
        dfs.append(pd.DataFrame({'avg_rfs': avg_rfs,
                      'session': [skey for _ in range(len(avg_rfs))],
                      'animalid': [animalid for _ in range(len(avg_rfs))]
                      }))
        
    
df = pd.concat(dfs, axis=0)

fig, ax = pl.subplots()
    
sns.violinplot(x='session', y='avg_rfs', ax=ax, data=df)

    
    


