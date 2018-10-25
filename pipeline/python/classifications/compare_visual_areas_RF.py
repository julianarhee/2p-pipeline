#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 20:47:06 2018

@author: juliana
"""

import json
import os


import numpy as np
import seaborn as sns
import pylab as pl
import cPickle as pkl

from pipeline.python.classifications import visualarea_class as vis


#%%

rootdir = '/n/coxfs01/2p-data'

fit_thr = 0.5


fig, ax = pl.subplots()



#%%


visual_area = 'V1'
color = 'b'
kde = False

V1_fovs = {'JC024': {'exclude':[],
                     'include': [],
                     'ignore': []
                     },
           'CE074': {'exclude': ['20180215'],
                     'include': [],
                     'ignore': ['20180215_FOV2',
                                '20180220',
                                '20180221_FOV2']
                     },
           'JC007': {'exclude': [],
                     'include': [],
                     'ignore': []
                     }
           }
  
                  
V1 ={}
for animalid, sessioninfo in V1_fovs.items():
    
    A = vis.get_animal(animalid, excluded_sessions=sessioninfo['exclude'], rootdir=rootdir)
    
    V1_rf_dicts = vis.get_good_RFs(A, fit_thr=fit_thr, rootdir=rootdir)

    for fovkey, rdict in V1_rf_dicts.items():
        if '_'.join(fovkey.split('_')[0:-1]) in sessioninfo['ignore']:
            continue
        if len(sessioninfo['include']) > 0 and fovkey.split('_')[0] not in sessioninfo['include']:
            continue
        
        V1.update({fovkey: rdict})
    

vis.hist_rf_widths(V1, visual_area=visual_area, ax=ax, color=color, kde=kde)



#%%

#animalid = 'CE077'
#excluded_sessions = ['20180525','20180321', '20180331', '20180423']

#A = vis.get_animal(animalid, excluded_sessions=excluded_sessions, rootdir=rootdir)
#
#LM_rfs = vis.get_good_RFs(A, fit_thr=fit_thr, rootdir=rootdir)

visual_area = 'LM'
color = 'g'
#kde = False

LM_fovs = {'CE077': {'exclude':['20180525',
                                '20180321',
                                '20180331',
                                '20180423'],
                     'include': [],
                     'ignore': []
                     }
           }
                    
LM = {}
for animalid, sessioninfo in LM_fovs.items():
    
    A = vis.get_animal(animalid, excluded_sessions=sessioninfo['exclude'], rootdir=rootdir)
    
    LM_rf_dicts = vis.get_good_RFs(A, fit_thr=fit_thr, rootdir=rootdir)

    for fovkey, rdict in LM_rf_dicts.items():
        if '_'.join(fovkey.split('_')[0:-1]) in sessioninfo['ignore']:
            continue
        if len(sessioninfo['include']) > 0 and fovkey.split('_')[0] not in sessioninfo['include']:
            continue
        
        LM.update({fovkey: rdict})
        
        
vis.hist_rf_widths(LM, visual_area=visual_area, ax=ax, color=color, kde=kde)

#%%


#rootdir = '/n/coxfs01/2p-data'
#animalid = 'JC022'
#excluded_sessions = ['20181009']

visual_area = 'LI/LL'
color = 'magenta'


LI_fovs = {'JC022': {'exclude':['20181009'],
                     'include': [],
                     'ignore': []
                     },
           'JC023': {'exclude': [],
                     'include': [],
                     'ignore': []
                     },
           'CE074': {'exclude': ['20180215'],
                     'include': [],
                     'ignore': ['20180221_FOV1*']
                     },
           'CE084': {'exclude': [],
                     'include': ['20180507'],
                     'ignore': []
                     }
           }
                    
LI = {}
for animalid, sessioninfo in LI_fovs.items():
    
    A = vis.get_animal(animalid, excluded_sessions=sessioninfo['exclude'], rootdir=rootdir)
    
    LI_rf_dicts = vis.get_good_RFs(A, fit_thr=fit_thr, rootdir=rootdir)

    for fovkey, rdict in LI_rf_dicts.items():
        if '_'.join(fovkey.split('_')[0:-1]) in sessioninfo['ignore']:
            continue
        if len(sessioninfo['include']) > 0 and fovkey.split('_')[0] not in sessioninfo['include']:
            continue
            
        LI.update({fovkey: rdict})
    
#    
#A = vis.get_animal(animalid, excluded_sessions=excluded_sessions, rootdir=rootdir)
#LI_rfs = vis.get_good_RFs(A, fit_thr=fit_thr, rootdir=rootdir)
vis.hist_rf_widths(LI, visual_area=visual_area, ax=ax, color=color, kde=kde)

#%%

visual_area = 'P/POR'
color = 'orange'


POR_fovs = {'JC017': {'exclude':[],
                     'include': [],
                     'ignore': []
                     },
           'CE084':  {'exclude': [],
                     'include': ['20180511'],
                     'ignore': []
                     }
           }
                    
POR = {}
for animalid, sessioninfo in POR_fovs.items():
    
    A = vis.get_animal(animalid, excluded_sessions=sessioninfo['exclude'], rootdir=rootdir)
    
    POR_rf_dicts = vis.get_good_RFs(A, fit_thr=fit_thr, rootdir=rootdir)

    for fovkey, rdict in POR_rf_dicts.items():
        if '_'.join(fovkey.split('_')[0:-1]) in sessioninfo['ignore']:
            continue
        if len(sessioninfo['include']) > 0 and fovkey.split('_')[0] not in sessioninfo['include']:
            continue
        
        POR.update({fovkey: rdict})
    
#    
#A = vis.get_animal(animalid, excluded_sessions=excluded_sessions, rootdir=rootdir)
#LI_rfs = vis.get_good_RFs(A, fit_thr=fit_thr, rootdir=rootdir)
vis.hist_rf_widths(POR, visual_area=visual_area, ax=ax, color=color, kde=kde)

#%%

ax.legend()


pl.savefig(os.path.join(rootdir, 'VISUAL_AREAS', 'all_fovs_RFs_noPOR.png'))