#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 11:40:31 2019

@author: julianarhee
"""

import glob
import os
import json
import sys

import numpy as np
import pylab as pl
import seaborn as sns
import pandas as pd
import cPickle as pkl


from pipeline.python.classifications import experiment_classes as util

from pipeline.python.rois.utils import get_roiid_from_traceid

import scipy.stats as spstats

from pipeline.python.utils import label_figure, natural_keys
from pipeline.python.classifications import bootstrap_fit_tuning_curves as osi
from pipeline.python.classifications import get_dataset_stats as gd
from pipeline.python.retinotopy import convert_coords as cc


    
def get_roi_positions(sdata, traceid='traces001', rootdir='/n/coxfs01/2p-data'):
    skipped = []
    pos_list = []
    for (visual_area, animalid, session, fov), g in sdata.groupby(['visual_area', 'animalid', 'session', 'fov']):
        
        datakey = '_'.join([visual_area, animalid, session, fov])
        print("DSET: %s" % '|'.join([animalid, session, fov]))

        S = util.Session(animalid, session, fov, rootdir=rootdir)
        
        try: 
            roiid = get_roiid_from_traceid(animalid, session, fov, run_type=None, traceid=traceid)
        except Exception as e:
            print(e)
            print("... skipping")
            skipped.append(datakey)
            continue

        masks, zimg = S.load_masks(rois=roiid)
        fovinfo = cc.get_roi_fov_info(masks, zimg, roi_list=None)
        pos = fovinfo['positions']
        pos['visual_area'] = [visual_area for _ in range(len(pos))]
        pos['animalid'] = [animalid for _ in range(len(pos))]
        pos['session'] = [session for _ in range(len(pos))]
        pos['fov'] = [fov for _ in range(len(pos))]
        
        pos_list.append(pos)
        
    
    posdf = pd.concat(pos_list, axis=0).reset_index()
    print posdf.shape
    print posdf.head()
    
    fovinfo = {'positions': posdf,
               'dims': zimg.shape,
               'ap_lim': fovinfo['ap_lim'],
               'ml_lim': fovinfo['ml_lim']}
    
    return fovinfo, skipped
        
def main(options):
    
    optsE = gd.extract_options(options)
    
    
    rootdir = optsE.rootdir
    aggregate_dir = optsE.aggregate_dir
    fov_type = optsE.fov_type
    traceid = optsE.traceid
    state  = optsE.state
    
    print aggregate_dir
    
    sdata = gd.get_dataset_info(aggregate_dir=aggregate_dir, traceid=traceid,
                           fov_type=fov_type, state=state)
    
    fovinfo, skipped = get_roi_positions(sdata, traceid=traceid, rootdir=rootdir)
    
    with open(os.path.join(aggregate_dir, 'roi_positions.pkl'), 'wb') as f:
        pkl.dump(fovinfo, f, protocol=pkl.HIGHEST_PROTOCOL)
    
   
    print("Got position info for all ROIs.")
    print("Skipped %i datasets." % len(skipped))
    if len(skipped) > 0:
        print("*** WARNING (missing) ***")
        for s in skipped:
            print(s)

 
if __name__ == '__main__':
    main(sys.argv[1:])
    
