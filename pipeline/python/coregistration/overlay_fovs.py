#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 15:26:27 2019

@author: julianarhee
"""

import os
import numpy as np
import cPickle as pkl
import pylab as pl


from pipeline.python.coregistration.align_fov import Animal, FOV
from pipeline.python.utils import label_figure, natural_keys

rootdir = '/n/coxfs01/2p-data'
animalid = 'JC076'

coreg_dir = os.path.join(rootdir, animalid, 'coreg')
fn = os.path.join(coreg_dir, 'FOVs.pkl')


with open(fn, 'rb') as f:
    fovs = pkl.load(f)
#%%
ref = fovs.reference
d1, d2  = ref.shape

merge = np.zeros((d1, d2, 3), dtype=ref.dtype)
merge[:, :, 1] = ref

for session in sorted(fovs.session_list.keys(), key=natural_keys):
    img = fovs.session_list[session].alignment['aligned']
    merge[:, :, 0] += img


fig = pl.figure()
pl.imshow(merge)
pl.axis('off')

fovnames = ['_'.join(f.split('_')[0:2]) for f in sorted(fovs.session_list.keys(), key=natural_keys)]
data_identifier = '%s|%s' % (animalid, '|'.join(fovnames))
print(data_identifier)

label_figure(fig, data_identifier)


pl.savefig(os.path.join(coreg_dir, 'all_fovs.png'))
