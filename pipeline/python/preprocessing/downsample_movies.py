#!/usr/bin/env python
# coding: utf-8

# In[1]:



import cv2
import glob
import logging
import matplotlib.pyplot as plt
import numpy as np
import os

try:
    cv2.setNumThreads(0)
except():
    pass

try:
    if __IPYTHON__:
        # this is used for debugging purposes only. allows to reload classes
        # when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    pass

import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf import params as params
from caiman.utils.utils import download_demo
from caiman.utils.visualization import plot_contours, nb_view_patches, nb_plot_contour

import pylab as pl


import tifffile as tf
import json
import time


# In[2]:


import re
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]


# In[ ]:





# In[3]:


rootdir = '/n/coxfs01/2p-data'
animalid = 'JC084'
session = '20190525' #'20190505_JC083'
session_dir = os.path.join(rootdir, animalid, session)
fov = 'FOV1_zoom2p0x'
run_list = ['gratings']
run_label = 'gratings'


# In[4]:


motion_correct = True
fnames = []
for run in run_list:
    if motion_correct:
        fnames_tmp = glob.glob(os.path.join(session_dir, fov, '%s*' % run, 'raw*', '*.tif'))
    else:
        fnames_tmp = [f for f in glob.glob(os.path.join(session_dir, fov, '%s*' % run, 
                                                        'processed', 'processed001*','mcorrected*', '*.tif'))\
                     if len(os.path.split(os.path.split(f)[0])[-1].split('_'))==2]
    fnames.extend(fnames_tmp)
    print("[%s]: added %i tifs to queue." % (run, len(fnames_tmp)))
fnames = sorted(fnames, key=natural_keys)


# In[5]:


fn = fnames[0]
print(fn)


# In[6]:


fovdir = glob.glob(os.path.join(session_dir, fov))[0]


# In[7]:


tmpdir = os.path.join(fovdir, 'tmp')
print(tmpdir)


# In[ ]:


mov = tf.imread(fn)
print(mov.shape)


# In[ ]:


from caiman.source_extraction.cnmf.initialization import resize as cmresize


# In[ ]:


ds_factor = (1, 1, 0.1)
ds_mov = cmresize(mov, ds_factor)


# In[ ]:


tf.imsave(os.path.join(tmpdir, 'ds.tif'))


# In[ ]:




