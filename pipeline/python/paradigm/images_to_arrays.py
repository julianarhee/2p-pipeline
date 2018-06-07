#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 21:14:21 2018

@author: juliana
"""

import os
import h5py
import datetime
import numpy as np
from PIL import Image as pil
from pipeline.python.utils import natural_keys


output_dir = '/mnt/odyssey/behavior/stimuli/movie_arrays'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    

movie_dir = '/home/juliana/Repositories/protocols/physiology/stimuli/movies'
movie_types = [m for m in os.listdir(movie_dir) if os.path.isdir(os.path.join(movie_dir, m))]
print movie_types

objects = ['Blob_D1', 'Blob_M53', 'Blob_D2']
movies_to_make = ['%s_Rot_y_movie_slow' % obj for obj in objects]
movies_to_make_rev = ['%s_reverse' % mov for mov in movies_to_make]
movies = movies_to_make + movies_to_make_rev

def pngs_to_array(img_dir, fmt='png'):
    fns = sorted([i for i in os.listdir(img_dir) if i.endswith(fmt)], key=natural_keys)
    sample = np.array(pil.open(os.path.join(img_dir, fns[0])))
    nframes = len(fns)
    d1, d2 = sample.shape
    marray = np.empty((d1, d2, nframes), dtype=sample.dtype)
    for fi, fn in enumerate(fns):    
        im = pil.open(os.path.join(img_dir, fn))
        marray[:,:,fi] = im
    return marray
        

for mov in movies:
    print "Creating array: %s" % mov
    img_dir = os.path.join(movie_dir, mov)
    marray = pngs_to_array(img_dir)
    
    f = h5py.File(os.path.join(output_dir, '%s.hdf5' % mov), 'a')
    if 'frames' in f.keys():
        del f['frames']
    f.create_dataset('frames', data=marray, dtype=marray.dtype)
    f.attrs['name'] = mov
    f.attrs['source'] = img_dir
    f.attrs['creation_time'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    f.close()