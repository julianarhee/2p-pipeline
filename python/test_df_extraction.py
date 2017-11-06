
# coding: utf-8

# In[18]:


from __future__ import division
#from __future__ import print_function
import matplotlib
matplotlib.use('TkAgg')
from builtins import zip
from builtins import str
from builtins import map
from builtins import range
from past.utils import old_div
import cv2
import glob
import numpy as np
import os
from scipy.sparse import spdiags, issparse

# import caiman
from caiman.base.rois import com
import caiman as cm
from caiman.mmapping import parallel_dot_product

import time
import pylab as pl

import re
import json
import h5py
import cPickle as pkl
import scipy.io
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pprint

pp = pprint.PrettyPrinter(indent=4)

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def serialize_json(instance=None, path=None):
    dt = {}
    dt.update(vars(instance))


# In[19]:


source = '/nas/volume1/2photon/projects'
experiment = 'gratings_phaseMod'
session = '20171009_CE059'
acquisition = 'FOV1_zoom3x'
functional = 'functional'

roi_id = 'caiman2Dnmf001'
roi_method = 'caiman2D'

save_movies = False #True

inspect_components = False
display_average = True
reuse_reference = True


# In[3]:


acquisition_dir = os.path.join(source, experiment, session, acquisition)

acquisition_meta_fn = os.path.join(acquisition_dir, 'reference_%s.json' % functional)
with open(acquisition_meta_fn, 'r') as f:
    acqmeta = json.load(f)


# In[4]:


roi_dir = os.path.join(acqmeta['roi_dir'], roi_id)
roiparams_path = os.path.join(roi_dir, 'roiparams.json')
with open(roiparams_path, 'r') as f:
    roiparams = json.load(f)

if not roi_id==roiparams['roi_id']:
    print("***WARNING***")
    print("Loaded ROIPARAMS id doesn't match user-specified roi_id.")
    pp.pprint(roiparams)
    use_loaded = raw_input('Use loaded ROIPARAMS? Press Y/n: ')
    if use_loaded=='Y':
        roi_id = roiparams['roi_id']
        roiparams['params']['use_kept_ony'] = use_kept_only
    else:
        print("Not a valid entry. Re-start with correct ROI_ID.")        
else:
    use_kept_only = roiparams['params']['use_kept_only'] 
   


# In[5]:


# Load mcparams.mat:
mcparams = scipy.io.loadmat(acqmeta['mcparams_path'])
mc_ids = sorted([m for m in mcparams.keys() if 'mcparams' in m], key=natural_keys)
if len(mc_ids)>1:
    for mcidx,mcid in enumerate(sorted(mc_ids, key=natural_keys)):
        print(mcidx, mcid)
    mc_id_idx = raw_input('Select IDX of mc-method to use: ')
    mc_id = mc_ids[int(mc_id_idx)]
    print("Using MC-METHOD: ", mc_id)
else:
    mc_id = mc_ids[0]
    


# In[6]:


mcparams = mcparams[mc_id] #mcparams['mcparams01']
reference_file_idx = int(mcparams['ref_file'])
signal_channel_idx = int(mcparams['ref_channel'])

signal_channel = 'Channel%02d' % int(signal_channel_idx)
reference_file = 'File%03d' % int(reference_file_idx)
if signal_channel_idx==0:
    signal_channel_idx = input('No ref channel found. Enter signal channel idx (1-indexing): ')
if reference_file_idx==0:
    reference_file_idx = input('No ref file found. Enter file idx (1-indexing): ')

signal_channel = 'Channel%02d' % int(signal_channel_idx)
reference_file = 'File%03d' % int(reference_file_idx)
print("Specified signal channel is:", signal_channel)
print("Selected reference file:", reference_file)


# In[7]:


if isinstance(acqmeta['slices'], int):
    nslices = acqmeta['slices']
else:
    nslices = len(acqmeta['slices'])
    
print(nslices)


# In[8]:


# source of NMF output run:
nmf_output_dir = os.path.join(roi_dir, 'nmf_output')
nmf_fns = sorted([n for n in os.listdir(nmf_output_dir) if n.endswith('npz')], key=natural_keys)

ref_nmf_fn = [f for f in nmf_fns if reference_file in f][0]

file_names = sorted(['File%03d' % int(f+1) for f in range(acqmeta['ntiffs'])], key=natural_keys)
if not len(file_names)==len(nmf_fns):
    print('***ALERT***')
    print('Found NMF results does not match num tiff files.')

# Get source tiffs (mmap):
tiff_source = str(mcparams['dest_dir'][0][0][0])
tiff_dir = os.path.join(acquisition_dir, functional, 'DATA', tiff_source)
#tiff_dir

# Get mmap tiffs:
memmapped_fns = sorted([m for m in os.listdir(tiff_dir) if m.endswith('mmap')], key=natural_keys)


# In[9]:


currslice = 0


# In[10]:



if use_kept_only is True:
    if 'kept_rois' in roiparams['params'].keys():
        kept = roiparams['params']['kept_rois']
    else:
        ref_nmf_fn = [n for n in nmf_fns if reference_file in n][0]
        ref_nmf = np.load(os.path.join(nmf_output_dir, ref_nmf_fn))
        kept = [i for i in ref_nmf['idx_components']]
        
        for fid,curr_file in enumerate(sorted(file_names, key=natural_keys)):
            #print("Extracting ROI STRUCT from %s" % curr_file)
            curr_nmf_fn = [n for n in nmf_fns if curr_file in n][0]
            nmf = np.load(os.path.join(nmf_output_dir, curr_nmf_fn))
            curr_kept = [i for i in ref_nmf['idx_components']]
            kept = list(set(kept) & set(curr_kept))
            #print(kept)
        roiparams['params']['kept_rois'] = kept
    if not roiparams['nrois'][currslice]==len(kept):
        roiparams['nrois'][currslice] = len(kept)
pp.pprint(roiparams)


# In[11]:


fid = 0
curr_file = file_names[fid]


# In[12]:


print("Extracting ROI STRUCT from %s" % curr_file)
curr_nmf_fn = [n for n in nmf_fns if curr_file in n][0]
nmf = np.load(os.path.join(nmf_output_dir, curr_nmf_fn))


d1 = int(nmf['d1'])
d2 = int(nmf['d2'])
      
if use_kept_only:
    print("Keeping %i ROIs." % len(kept))
    A = nmf['A'].all().tocsc()[:, kept]
    C = nmf['C'][kept, :]
    YrA = nmf['YrA'][kept, :]
    Cdf = nmf['Cdf'][kept,:]
else:
    A = nmf['A'].all()
    C = nmf['C']
    nmf['YrA'] = nmf['YrA']
    Cdf = nmf['Cdf']

print(Cdf.shape)       
f = nmf['f']
b = nmf['b']

curr_mmap = [m for m in memmapped_fns if curr_file in m][0]
Yr, dims, T = cm.load_memmap(os.path.join(tiff_dir, curr_mmap))


# In[13]:


bl = np.copy(b)


# In[14]:


nA = np.array(np.sqrt(A.power(2).sum(0)).T)
A = scipy.sparse.coo_matrix(A / nA.T)
C = C * nA
bl = (bl * nA.T).squeeze()
nA = np.array(np.sqrt(A.power(2).sum(0)).T)
T = C.shape[-1]


# In[15]:


print("nA:", nA.shape)
print("A:", A.shape)
print("C:", C.shape)
print("bl:", bl.shape)


# In[21]:


c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)


# In[24]:


AY = parallel_dot_product(Yr, A, dview=dview, block_size=400,
                                  transpose=True).T
# AY = A.T.dot(Yr)
print("AY:", AY.shape)


# In[25]:


bas_val = bl[None, :]
print(bas_val.shape)


# In[26]:


Bas = np.repeat(bas_val, T, 0).T
print(Bas.shape)


# In[ ]:


AA = A.T.dot(A)
AA.setdiag(0)
Cf = (C - Bas) * (nA**2)
C2 = AY - AA.dot(C)

