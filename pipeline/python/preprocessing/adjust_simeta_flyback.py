#!/usr/bin/env python2
'''
Run this script to adjust SI metadata for flyback-corrected tiffs.
'''

import sys
import os
import numpy as np
import tifffile as tf
import re

rootdir = '/nas/volume1/2photon/data'
animalid = 'CE059'
session = '20171009_CE059'
acquisition = 'FOV1_zoom3x'
run = 'gratings_phasemod_run4'

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]


acquisition_dir = os.path.join(rootdir, animalid, session, acquisition)
rawtiff_dir = os.path.join(acquisition_dir, run, 'raw')
raw_simeta_fn = [j for j in os.listdir(rawtiff_dir) if j.endswith('json')][0]
with open(os.path.join(rawtiff_dir, raw_simeta_fn), 'r') as f:
    raw_simeta = json.load(f)

#tiffnames = sorted(raw_simeta['filenames'], key=natural_keys)
filenames = sorted([k for k in raw_simeta.keys() if 'File' in k], key=natural_keys)

for fi in filenames:
    frame_idxs = refmeta['frame_idxs']
    nslices_orig = raw_simeta[fi]['SI']['hStackManager']['numSlices']
    ndiscard_orig = raw_simeta[fi]['SI']['hFastZ']['numDiscardFlybackFrames']
   
    nslices_selected = refmeta['nslices']
    ndiscarded_extra = nslices_orig - nslices_selected
    
    # Rewrite relevant SI fields:
    raw_simeta[fi]['SI']['hStackManager']['nSlices'] = nslices_selected
    raw_simeta[fi]['SI']['hStackManager']['zs'] = raw_simeta['SI']['hStackManager']['zs'][ndiscarded_extra:]
    raw_simeta[fi]['SI']['hFastZ']['numDiscardFlybackFrames'] = 0
    raw_simeta[fi]['SI']['hFastZ']['numFramesPerVolume'] = nslices_selected
    raw_simeta[fi]['SI']['hFastZ']['discardFlybackFames'] = 0 # flag this so Acquisition2P's parseScanImageTiff tkaes correct n slices
   
    if len(frame_idxs) > 0: 
        raw_simeta[fi]['imgdescr'] = [raw_simeta[fi]['imgdescr'][i] for i in frame_idxs] 
    
    adj_simeta[fi]['SI'] = raw_simeta[fi]['SI']
    adj_simeta[fi]['imgdescr'] = raw_simeta[fi]['imgdescr']


with open(os.path.join(acquisition_dir, run, 'processed', process_id), 'w') as fw:
    json.dump(adj_simeta, fw, indent=4, sort_keys=True)


      
    
