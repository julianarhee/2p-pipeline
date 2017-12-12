#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 11:57:24 2017

@author: julianarhee
"""

import os
import h5py
import json
import tifffile as tf

#%%
rootdir = '/nas/volume1/2photon/data'
animalid = 'JR063'
session = '20171202_JR063'

session_dir = os.path.join(rootdir, animalid, session)
roi_dir = os.path.join(session_dir, 'ROIs')

rid_hash = 'e4893c'
roi_name = 'rois002'

with open(os.path.join(roi_dir, 'rids_%s.json' % session), 'r') as f:
    roidict = json.load(f)
    
RID = roidict[roi_name]

#%%

