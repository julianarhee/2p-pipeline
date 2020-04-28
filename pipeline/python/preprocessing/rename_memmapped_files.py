#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 16:52:57 2019

@author: julianarhee
"""

import glob
import os
import numpy as np

from pipeline.python.utils import natural_keys

animalid = 'JC084'
session = '20190522'
fov = 'FOV1_zoom2p0x'
experiment = 'gratings'
rootdir = '/n/coxfs01/2p-data'

fovdir = os.path.join(rootdir, animalid, session, fov)
old_prefix = 'Yr'


import json
    
#prefix = 'JC084-20190525-FOV1_zoom2p0x-gratings-downsample-1'
#prefix = 'JC084-20190522-FOV1_zoom2p0x-gratings-downsample-1-mcorrected'
prefix = 'JC084-20190522-FOV1_zoom2p0x-gratings-downsample-5'


memfiles = sorted(glob.glob(os.path.join(fovdir, 'caiman_results', experiment, 'memmap', '%s0*.mmap' % old_prefix)), key=natural_keys)
print(len(memfiles))

# Load MC info
mcinfo = glob.glob(os.path.join(fovdir, 'caiman_results', experiment, '*_mc-rigid.npz'))[0]
minfo = np.load(mcinfo)
list(minfo.keys())
tiffiles = sorted(list(minfo['fnames']), key=natural_keys)
len(tiffiles)


# Load mmap info (if no mc)
meminfo = glob.glob(os.path.join(fovdir, 'caiman_results', experiment, '*_memmap-params.json'))[0]
with open(meminfo, 'r') as f:
    minfo = json.load(f)
tiffiles = sorted(list(minfo['fnames']), key=natural_keys)
len(tiffiles)



fn = memfiles[0]

for fi, (tfn, fn) in enumerate(zip(tiffiles, memfiles)):
    print fi, tfn, fn
    tif_name = os.path.splitext(os.path.split(tfn)[-1])[0]
    if 'Slice' in tif_name:
        process_str = 'mcorrected'
    else:
        process_str = 'rig'
    append = '_d1_%s' % os.path.split(fn)[-1].split('_d1_')[-1]
    newname = 'file%05d_%s_%s_%s' % (int(fi+1), tif_name, process_str, append)
    parentdir = os.path.split(fn)[0]
    os.rename(fn, os.path.join(parentdir, newname))




memparams_path = os.path.join(fovdir, 'caiman_results', experiment, '%s_memmap-params.json' % prefix)

resize_fact = (1, 1, 1)
mmap_params = {'resize_fact': list(resize_fact),
               'add_to_movie': float(minfo['min_mov']),
               'border_to_0': int(minfo['border_to_0']),
               'fnames': sorted(list(minfo['fname']), key=natural_keys)}
for k, m in mmap_params.items():
    if isinstance(m, list):
        if not isinstance(m[0], (str, unicode)):
            mmap_params[k] = [float(mi) for mi in m]
    else:
        mmap_params[k] = float(m)

with open(memparams_path, 'w') as f:
    json.dump(mmap_params, f, indent=4)

