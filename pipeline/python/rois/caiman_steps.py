
# coding: utf-8

# In[3]:

import matplotlib as mpl
mpl.use('agg')

try:
    if  __IPYTHON____IPYTH :
        # this is used for debugging purposes only. allows to reload classes when changed
        get_ipython().magic(u'load_ext autoreload')
        get_ipython().magic(u'autoreload 2')
except NameError:       
    print('Not IPYTHON')    
    pass

import sys
import numpy as np
from time import time
from scipy.sparse import coo_matrix
import psutil
import glob
import os
import scipy
import pprint
pp = pprint.PrettyPrinter(indent=4)

from ipyparallel import Client
import pylab as pl
import caiman as cm
from caiman.components_evaluation import evaluate_components
from caiman.utils.visualization import plot_contours,view_patches_bar,nb_plot_contour,nb_view_patches
from caiman.base.rois import extract_binary_masks_blob
import caiman.source_extraction.cnmf as cnmf

from caiman.source_extraction.cnmf.utilities import detrend_df_f


# In[4]:



#import bokeh.plotting as bp#import 
import bokeh.plotting as bpl
try:
       from bokeh.io import vform, hplot
except:
       # newer version of bokeh does not use vform & hplot, instead uses column & row
       from bokeh.layouts import column as vform
       from bokeh.layouts import row as hplot
from bokeh.models import CustomJS, ColumnDataSource, Slider
from IPython.display import display, clear_output
import matplotlib.cm as cmap
import numpy as np

bpl.output_notebook()


# In[24]:


import os
import glob
import json
import h5py
import datetime
import copy
import cPickle as pkl
import numpy as np
import tifffile as tf
import pandas as pd
from pipeline.python.utils import natural_keys

from pipeline.python.paradigm import utils as util



#%% start cluster for efficient computation
single_thread=False


# frame rate in Hz
#final_frate=10 
##backend='SLURM'
#backend='local'
#if backend == 'SLURM':
#    n_processes = np.int(os.environ.get('SLURM_NPROCS'))
#else:
#    # roughly number of cores on your machine minus 1
#    n_processes = np.maximum(np.int(psutil.cpu_count()),1) 
#print('using ' + str(n_processes) + ' processes')


#if single_thread:
#    dview=None
#else:    
#    try:
#        c.close()
#    except:
#        print('C was not existing, creating one')
#    print("Stopping  cluster to avoid unnencessary use of memory....")
#    sys.stdout.flush()  
##     if backend == 'SLURM':
##         try:
##             cm.stop_server(is_slurm=True)
##         except:
##             print('Nothing to stop')
##         slurm_script='/mnt/xfs1/home/agiovann/SOFTWARE/Constrained_NMF/SLURM/slurmStart.sh'
##         cm.start_server(slurm_script=slurm_script)
##         pdir, profile = os.environ['IPPPDIR'], os.environ['IPPPROFILE']
##         c = Client(ipython_dir=pdir, profile=profile)        
##     else:
#    cm.stop_server()
#    cm.start_server()        
#    c=Client()
#
#    print('Using '+ str(len(c)) + ' processes')
#    dview=c[:len(c)]



# Start cluster:
n_processes = 4

c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=n_processes, single_thread=False)
    

print n_processes
print dview


#%% ### Get TIF source

rootdir = '/mnt/odyssey'

#rootdir = '/n/coxfs01/2p-data'

animalid = 'CE077'
session = '20180523'
acquisition = 'FOV1_zoom1x'
run = 'gratings_run1'

acquisition_dir = os.path.join(rootdir, animalid, session, acquisition)

tif_src_dir = os.path.join(acquisition_dir, 'caiman_test_data')
fnames = glob.glob(os.path.join(tif_src_dir, '*.tif'))
print "Found %i tifs in src: %s" % (len(fnames), tif_src_dir)

run_dir = os.path.join(acquisition_dir, run)

fr = 44.69

#%% Set output dir for figures and results:

datestr = datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")
#output_dir = os.path.join(tif_src_dir, 'cnmf_%s' % datestr)

output_dir = os.path.join(tif_src_dir, 'results_seed')

if not os.path.exists(os.path.join(output_dir, 'figures')):
    os.makedirs(os.path.join(output_dir, 'figures'))
if not os.path.exists(os.path.join(output_dir, 'results')):
    os.makedirs(os.path.join(output_dir, 'results'))
print output_dir
    
#%% ### Create memmapped files:

do_mmap = False
mmap_dir = os.path.join(acquisition_dir, 'caiman_test_data', 'memmap2')

if do_mmap:
    mmap_dir = os.path.join(acquisition_dir, 'caiman_test_data', 'memmap2')
    if not os.path.exists(mmap_dir):
        os.makedirs(mmap_dir)
        
    # Determind min value of dataset:
    add_to_movie = np.min([-np.min(tf.imread(fname, pages=[5])) for fname in fnames])
    print add_to_movie

    # Params for mmap:
    downsample_factor = 1 # use .2 or .1 if file is large and you want a quick answer
    border_to_0 = 2
    base_name = os.path.join(mmap_dir, 'gratings') #'Yr'
    print base_name

    
    final_frate = fr * downsample_factor
    
    name_new=cm.save_memmap_each(fnames
             , dview=dview, base_name=base_name, resize_fact=(1, 1, downsample_factor)
             , remove_init=0, idx_xy=None, border_to_0=border_to_0 )
    name_new.sort()


#% In[33]:


# Run on cluster, rename (memory...)
name_new = glob.glob(os.path.join(mmap_dir, '*.mmap'))
print "Found mmap files:", name_new

if len(name_new)==0 or do_mmap:
    for ni, n in enumerate(name_new):
        mdir = os.path.split(n)[0]
        mfile = '%s%s' % ('gratings_%i' % int(ni+1000), os.path.split(n)[1].split('File%03d' % int(ni+1))[-1])
        print(mfile)
        os.rename(n, os.path.join(mdir, mfile))
    # 

    ##%% Join into one giant mmap if needed
    name_new = glob.glob(os.path.join(mmap_dir, '*.mmap'))
    if len(name_new) > 1:
        fname_new = cm.save_memmap_join(
            name_new, base_name='Yr', n_chunks=20, dview=dview)
    else:
        print('One file only, not saving!')
        fname_new = name_new[0]


#% In[38]:


fname_new = glob.glob(os.path.join(mmap_dir, 'Yr_*'))[0]
print "Full mmapped file for run: ", fname_new


#%% ### Load memmapped file (all runs):

#% LOAD MEMMAP FILE
Yr, dims, T = cm.load_memmap(fname_new)
d1, d2 = dims
images = np.reshape(Yr.T, [T] + list(dims), order='F')
Y = np.reshape(Yr, dims + (T,), order='F')


#%% Get correlation image:

cn_path = os.path.join(output_dir, 'Cn.npz')
if not os.path.exists(cn_path):
    print "Cn not found, calculating new..."
    # Look at correlation image:
    Cn = cm.movie(images).local_correlations(swap_dim=False) #cm.local_correlations(Y)
    #Cn[np.isnan(Cn)] = 0
    
    # #In[60]:
    np.savez(os.path.join(output_dir, 'Cn.npz'), Cn=Cn)
else:
    tmpd = np.load(cn_path)
    Cn = tmpd['Cn']
    print "Loaded Cn, shape:", Cn.shape

fig, ax = pl.subplots(1)
ax.imshow(Cn, cmap='gray') #, vmax=.35)
fig.savefig(os.path.join(output_dir, 'figures', 'Cn.png'))
pl.close()


#%% 

# #############################################################################
# cNMF: Extract from patches:
# #############################################################################

K=20
gSig=[3, 3]
#dview=None
Ain=None
rf=25
stride_cnmf = 6
init_method='greedy_roi'
alpha_snmf=None
final_frate=fr
p=2
merge_thresh=0.8
gnb = 1

cnmf_params = {'K': K, 
               'gSig': gSig,
               'rf': rf,
               'stride_cnmf': stride_cnmf,
               'init_method': init_method,
               'alpha_snmf': alpha_snmf,
               'final_frate': final_frate,
               'p': p,
               'merge_thresh': merge_thresh,
               'gnb': gnb,
               'spatial_dist': 2,
               'spatial_method': 'ellipse',
               'temporal_method': 'cvxpy',
               'fname_new': fname_new}

with open(os.path.join(output_dir, 'results', 'cnmf_params.json'), 'w') as f:
    json.dump(cnmf_params, f, indent=4, sort_keys=True)
    
#%%

def fit_cnmf_patches(images, cnmf_params, output_dir, n_processes=1, dview=None, verbose=True):
    

    #% Extract spatial and temporal components on patches
#    K=20
#    gSig=[3, 3]
#    #dview=None
#    Ain=None
#    rf=25
#    stride_cnmf = 6
#    init_method='greedy_roi'
#    alpha_snmf=None
#    final_frate=fr
#    p=2
#    merge_thresh=0.8
#    gnb = 2
        
    # Create cnmf object:
    cnm = cnmf.CNMF(n_processes, 
                    k=cnmf_params['K'],
                    gSig=cnmf_params['gSig'], 
                    merge_thresh=cnmf_params['merge_thresh'], 
                    p=0, 
                    dview=dview, 
                    Ain=None, 
                    rf=cnmf_params['rf'],
                    stride=cnmf_params['stride_cnmf'], 
                    memory_fact=1,
                    method_init=cnmf_params['init_method'], 
                    alpha_snmf=cnmf_params['alpha_snmf'], 
                    only_init_patch=True, 
                    gnb=cnmf_params['gnb']) #, 
                    #method_deconvolution='cvxpy')
    cnm.options['spatial_params']['dist'] = cnmf_params['spatial_dist'] # 2
    cnm.options['spatial_params']['method'] = cnmf_params['spatial_method'] # 'ellipse'
    cnm.options['temporal_params']['method'] = cnmf_params['temporal_method'] #'cvxpy'
    if verbose:
        pp.pprint(cnm.options)
    
    # Save cNMF input params:
    js_cnm_opts = copy.deepcopy(cnm.options)
    for k in cnm.options.keys():
        jsonify_keys = [skey for skey, sval in cnm.options[k].items() if type(sval) == np.ndarray]
        if len(jsonify_keys) > 0:
            for skey in jsonify_keys:
                js_cnm_opts[k][skey] = cnm.options[k][skey].tolist()
    
    with open(os.path.join(output_dir, 'results', 'cnmf_options_full.json'), 'w') as f:
        json.dump(js_cnm_opts, f, indent=4, sort_keys=False)

    
    # Extract components:
    cnm = cnm.fit(images)
    
    # Save because can take awhile, and want to be able to play around with params:
    cnmf_results_path = os.path.join(output_dir, 'results', 'results_patch.npz')
    np.savez(cnmf_results_path, 
             A_tot=cnm.A, C_tot=cnm.C, YrA_tot=cnm.YrA, b_tot=cnm.b, f_tot=cnm.f, sn_tot=cnm.sn,
             S_tot=cnm.S, Cn=Cn,
            options=cnm.options)
    
    print "cNMF patch results saved to: %s" % cnmf_results_path
    
    return cnm


def seed_cnmf(images, cnmf_params, A_in, C_in, b_in, f_in, n_processes=1, dview=None, verbose=True):
    
    cnm = cnmf.CNMF(n_processes, 
                    k=A_in.shape[-1], 
                    gSig=cnmf_params['gSig'], 
                    p=cnmf_params['p'], 
                    dview=dview,
                    merge_thresh=cnmf_params['merge_thresh'], 
                    Ain=A_in, 
                    Cin=C_in, 
                    b_in=b_in,
                    f_in=f_in, 
                    rf=None, 
                    stride=None, 
                    gnb=cnmf_params['gnb'],
                    check_nan = True) #,
                 #method_deconvolution='oasis', check_nan=True)

    cnm.options['temporal_params']['method'] = 'cvxpy'
    cnm.options['spatial_params']['dist'] = 2
    cnm.options['spatial_params']['method'] = 'ellipse'
            
    if verbose:
        print cnm.options

    
    # #In[114]:
    
    print "*** FITTING REFINED ***"
    
    cnm = cnm.fit(images)
    
    return cnm


#%%
    
patch_path = os.path.join(output_dir, 'results_test_1a.npz')

if not os.path.exists(patch_path):
    print "Patch results not found! Creating new."
    

    cnm = fit_cnmf_patches(images, cnmf_params, output_dir, n_processes=n_processes, dview=dview)
        
    ##%%
    A_tot = cnm.A
    C_tot = cnm.C
    YrA_tot = cnm.YrA
    b_tot = cnm.b
    f_tot = cnm.f
    sn_tot = cnm.sn
    #t2 = time.time() - t1
    
    #%%
else:
    cnm = np.load(patch_path)
    A_tot = cnm['A_tot'][()]
    C_tot = cnm['C_tot']
    YrA_tot = cnm['YrA_tot']
    b_tot = cnm['b_tot']
    f_tot = cnm['f_tot']
    sn_tot = cnm['sn_tot']

print(('Number of components:' + str(A_tot.shape[-1])))


#%%
# #### Evaluate components:


fr = 44.69             # approximate frame rate of data
decay_time = 4.0 #.0    # length of transient
min_SNR = 2.0      # peak SNR for accepted components (if above this, acept)
rval_thr = 0.5     # space correlation threshold (if above this, accept)
use_cnn = False # use the CNN classifier
min_cnn_thr = None  # if cnn classifier predicts below this value, reject
dims = [d1, d2]
gSig = (3, 3)

idx_components, idx_components_bad, SNR_comp, r_values, cnn_preds =     cm.components_evaluation.estimate_components_quality_auto(
                                    images, A_tot, C_tot, b_tot, f_tot,
                                     YrA_tot, fr, decay_time, gSig, dims,
                                     dview=dview, min_SNR=min_SNR,
                                     r_values_min=rval_thr, use_cnn=use_cnn,
                                     thresh_cnn_min=min_cnn_thr)

#crd = plot_contours(A_tot.tocsc()[:, idx_components], Cn, thr=0.9)

eval_params = {'fr': fr, 'decay_time': decay_time,
               'min_SNR': min_SNR, 'rval_thr': rval_thr,
               'use_cnn': use_cnn, 'min_cnn_thr': min_cnn_thr,
               'dims': dims, 'gSig': gSig}


eval_datestr = datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")

# Save evaluation results (plus all info we want to keep):
np.savez(os.path.join(output_dir, 'evaluation_patch_%s.npz' % eval_datestr), 
             A_tot=A_tot, C_tot=C_tot, YrA_tot=YrA_tot, b_tot=b_tot, f_tot=f_tot, sn_tot=sn_tot,
             idx_components=idx_components, idx_components_bad=idx_components_bad,
             SNR_comp=SNR_comp, r_values=r_values, 
            options=cnm.options, evalparams=eval_params)

# Save params in easy-read format:
with open(os.path.join(output_dir, 'figures', 'evalparams_patch_%s.json' % eval_datestr), 'w') as f:
    json.dump(eval_params, f, indent=4, sort_keys=True)

# Save correposnding eval results figure:
pl.figure(figsize=(15,5))
pl.subplot(1,2,1)
crd = plot_contours(A_tot.tocsc()[:, idx_components], Cn, thr=0.9)
pl.subplot(1,2,2)
crd = plot_contours(A_tot.tocsc()[:, idx_components_bad], Cn, thr=0.9)

pl.savefig(os.path.join(output_dir, 'figures', 'evaluation_patch_%s.png' % eval_datestr))
pl.close()


print "Keeping %i components." % len(idx_components)

#%%

#
#cm.utils.visualization.view_patches_bar(Yr, cnm.A[:, idx_components],
#                                        cnm.C[idx_components, :], cnm.b, cnm.f,
#                                        d1, d2,
#                                        YrA=cnm.YrA[idx_components, :], img=Cn)




##%% restart cluster to clean up memory
#dview.terminate()
#c, dview, n_processes = cm.cluster.setup_cluster(
#    backend='local', n_processes=None, single_thread=False)



#%%
# #############################################################################
# Load MANUAL rois, and use these to seed cNMF
# (instead of using patches to initilaize):
# #############################################################################

if manual_seed:
    results_basename = 'results_seed'
    rid = 'rois001'
    rid_dir = glob.glob(os.path.join(rootdir, animalid, session, 'ROIs', '%s*' % rid))[0]
    print rid_dir
    
    mask_fpath = os.path.join(rid_dir, 'masks.hdf5')
    
    # Load masks:
    maskfile = h5py.File(mask_fpath, 'r')
    print "Loaded maskfile:", maskfile.keys()
    ref_file = maskfile.keys()[0]
    masks = maskfile[ref_file]['masks']['Slice01']
    print masks.shape
    
    # Binarze and reshape:
    nrois, d1, d2 = masks.shape
    
    A_in = np.reshape(masks, (nrois, d1*d2)).astype(bool).T # A_in should be of shape (npixels, nrois)
    print "Reshaped seeded spatial comps:", A_in.shape
    
    C_in = None; b_in = None; f_in = None; 
    
#    gnb = 1
#    gSig = (3, 3)
#    p = 2

else:
    # Run patches above to get initial components:
    results_basename = 'results_cnmf'
    A_in, C_in, b_in, f_in = cnm.A[:, idx_components], cnm.C[idx_components], cnm.b, cnm.f
    
#%% 

crd = plot_contours(A_in, Cn, thr=0.9)
pl.savefig(os.path.join(output_dir, 'figures', 'seed_components.png'))
pl.close()


cnm2 = seed_cnmf(images, cnmf_params, A_in, C_in, b_in, f_in, n_processes=n_processes, dview=dview)

# =============================================================================
# RE-RUN seeded CNMF on accepted patches to refine and perform deconvolution
# =============================================================================

#A, C, b, f, YrA, sn = cnm.A, cnm.C, cnm.b, cnm.f, cnm.YrA, cnm.sn

refined_datestr = datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")


#% #### Visualize refined run:

pl.figure()
crd = plot_contours(cnm2.A, Cn, thr=0.6)
pl.savefig(os.path.join(output_dir, 'figures', 'spatial_comps_refined_%s.png' % refined_datestr))
pl.close()

print "*** Plotted all final components ***"
print "Final C:", cnm2.C.shape
print "Final YrA:", cnm2.YrA.shape
print "S:", cnm2.S.shape

# #### Extract DFF: 

quantile = 8
frames_window = int(round(15.*fr))
F_dff = detrend_df_f(cnm2.A, cnm2.b, cnm2.C, cnm2.f, YrA=cnm2.YrA,
                     quantileMin=quantile, frames_window=frames_window)


# #### Save refined run:

np.savez(os.path.join(output_dir, 'results', 'results_refined_%s.npz' % refined_datestr), 
         Cn=Cn, 
         A=cnm2.A.todense(), C=cnm2.C, b=cnm2.b, f=cnm2.f, 
         YrA=cnm2.YrA, 
         sn=cnm2.sn, 
         bl = cnm2.bl,
         d1=d1, d2=d2, 
         S=cnm2.S, F_dff=F_dff, quantileMin=quantile, frames_window=frames_window)

#%%
# #### Evaluate refined run:


## In[ ]:

print "*** Evaluating final run ***"
fr = 44.69             # approximate frame rate of data
decay_time = 4.0    # length of transient
min_SNR = 2.0      # peak SNR for accepted components (if above this, acept)
rval_thr = 0.6     # space correlation threshold (if above this, accept)

use_cnn = False # use the CNN classifier
min_cnn_thr = None  # if cnn classifier predicts below this value, reject
dims = [d1, d2]
gSig = (3, 3)

eval_params = {'fr': fr, 'decay_time': decay_time,
               'min_SNR': min_SNR, 'rval_thr': rval_thr,
               'use_cnn': use_cnn, 'min_cnn_thr': min_cnn_thr,
               'dims': dims, 'gSig': gSig}

idx_components, idx_components_bad, SNR_comp, r_values, cnn_preds =     cm.components_evaluation.estimate_components_quality_auto(
                                    images, cnm2.A, cnm2.C, cnm2.b, cnm2.f,
                                     cnm2.YrA, fr, decay_time, gSig, dims,
                                     dview=dview, min_SNR=min_SNR,
                                     r_values_min=rval_thr, use_cnn=use_cnn,
                                     thresh_cnn_min=min_cnn_thr)

eval2_datestr = datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")


with open(os.path.join(output_dir, 'figures', 'evalparams_refined_%s.json' % eval2_datestr), 'w') as f:
    json.dump(eval_params, f, indent=4, sort_keys=True)

plot_thr = 0.75
pl.figure(figsize=(15,5))
pl.subplot(1,2,1)
crd = plot_contours(cnm2.A.tocsc()[:, idx_components], Cn, thr=plot_thr)
pl.subplot(1,2,2)
crd = plot_contours(cnm2.A.tocsc()[:, idx_components_bad], Cn, thr=plot_thr)

pl.savefig(os.path.join(output_dir, 'figures', 'evaluation_refined_%s.png' % eval2_datestr))
pl.close()


print "Keeping %i components." % len(idx_components)

#% Save results

np.savez(os.path.join(output_dir, 'evaluation_refined_%s.npz' % eval2_datestr), 
         Cn=Cn, 
         A=cnm2.A.todense(), C=cnm2.C, b=cnm2.b, f=cnm2.f, YrA=cnm2.YrA, sn=cnm2.sn, 
         bl = cnm2.bl,
         d1=d1, d2=d2, 
         eval_params = eval_params,
         idx_components=idx_components, idx_components_bad=idx_components_bad,
         SNR_comp=SNR_comp, r_values=r_values,
         S=cnm2.S, F_dff=F_dff, quantileMin=quantile, frames_window=frames_window)


#%%

# ### Play around with contour visualization...


#results_fpath = os.path.join(output_dir, 'results', 'eval_final.npz')
results_fpath = glob.glob(os.path.join(output_dir, 'results_final_*.npz'))[0]

remove_bad_components = False

cnmd = np.load(results_fpath)


if '_patch' in results_fpath:
    append = '_tot'
else:
    append = ''
    
if remove_bad_components:
    A = cnmd['A%s' % append][()][:, cnmd['idx_components']]
    C = cnmd['C%s' % append][cnmd['idx_components']]
else:
    A = cnmd['A%s' % append][()]
    C = cnmd['C%s' % append]


b = cnmd['b%s' % append]
f = cnmd['f%s' % append]
    
Cn = cnmd['Cn']

crd = plot_contours(A, Cn, thr_method='nrg', nrgthr=0.5)
pl.savefig(os.path.join(output_dir, 'figures', 'final_idxcomps_nrg.png'))
pl.close()



#%% reconstruct denoised movie
#
#results_fpath = glob.glob(os.path.join(output_dir, 'results_final_*.npz'))[0]
#print results_fpath
#cnmd = np.load(results_fpath)

ntiffs = 4
nvolumes = T/ntiffs
C = C[:, 0:int(nvolumes)]
f = f[:, 0:int(nvolumes)]

#A = cnm2.A.tocsc()[:, idx_components]
#C = cnm2.C[idx_components]
#C = C[:, 0:int(nvolumes)]
#
#b = cnm2.b
#f = cnm2.f[0:int(nvolumes)]



#denoised = cm.movie(cnm2.A.dot(cnm2.C) +
#                    cnm2.b.dot(cnm2.f)).reshape(dims + (-1,), order='F').transpose([2, 0, 1])
#

if remove_bad_components:
#    A = cnm2.A.tocsc()[:, idx_components]
#    C = cnm2.C[idx_components, 0:nvolumes]
#    b = cnm2.b
#    f = cnm2.f[:, 0:int(nvolumes)]
    mov_fn_append = '_idxcomp'
else:
#    A = cnm2.A.tocsc()
#    C = cnm2.C[:, 0:nvolumes]
#    b = cnm2.b
#    f = cnm2.f[:, 0:int(nvolumes)]
    mov_fn_append = '_all'

denoised = cm.movie(A.dot(C) + b.dot(f)).reshape(dims + (-1,), order='F').transpose([2, 0, 1])

denoised.save(os.path.join(output_dir, 'figures', 'denoised_refined_File005%s.tif' % mov_fn_append))





#% #### Visualize "good" components:

# In[ ]:


#crd = plot_contours(A[:, idx_components], Cn, thr=0.9)

#pl.figure(figsize=(15,5))
#pl.subplot(1,2,1)
#crd = plot_contours(cnm2.A[:, idx_components], Cn, thr=0.9)
#pl.subplot(1,2,2)
#crd = plot_contours(cnm2.A[:, idx_components_bad], Cn, thr=0.9)
#
#pl.savefig(os.path.join(output_dir, 'figures', 'evaluation_final.png'))
#pl.close()

# In[ ]:

A = A_tot
C = C_tot
YrA = YrA_tot
f = f_tot
b = b_tot
cm.utils.visualization.view_patches_bar(Yr, cnm2.A[:, idx_components],
                                        cnm2.C[idx_components, :], cnm2.b, cnm2.f,
                                        d1, d2,
                                        YrA=cnm2.YrA[idx_components, :], img=Cn)



# In[ ]:

#%% Show final traces
cnm2.view_patches(Yr, dims=dims, img=Cn)

#%% STOP CLUSTER and clean up log files
cm.stop_server(dview=dview)
log_files = glob.glob('*_LOG_*')
for log_file in log_files:
    os.remove(log_file)


#%%

# #############################################################################
# #### Align traces to trials
# #############################################################################

#results_path = os.path.join(output_dir, 'results_analysis1b.npz')

#results_fpath = os.path.join(output_dir, 'results', 'eval_final.npz')

results_fpath = glob.glob(os.path.join(output_dir, 'evaluation_refined_*.npz'))[0]
print "Loading results: %s" % results_fpath

c = np.load(results_fpath)

A = c['A'][()]
C = c['C']
YrA = c['YrA']
idx_components = c['idx_components']

#traces = cnm.C[idx_components] + cnm.YrA[idx_components]


# Get RAW df:
if remove_bad_components:
    nrois = len(idx_components)
    F_raw = C[idx_components] + YrA[idx_components]
    A = A[:, idx_components]
    C = C[idx_components]
    S = c['S'][idx_components]
    F_dff = c['F_dff'][idx_components]
    YrA = YrA[idx_components]
else:
    nrois = A.shape[-1]
    F_raw = C + YrA
    F_dff = c['F_dff']
    S = c['S']
print F_raw.shape
raw_df = pd.DataFrame(data=F_raw.T, columns=[['roi%05d' % int(i+1) for i in range(nrois)]])



b = c['b']
f = c['f']

# Get baseline:
B = A.T.dot(b).dot(f)
print B.shape
F0_df = pd.DataFrame(data=B.T, columns=raw_df.columns)


# Get DFF:
dFF_df = pd.DataFrame(data=F_dff.T, columns=[['roi%05d' % int(i+1) for i in range(nrois)]])
print dFF_df.shape


# Get Drift-Corrected "raw":
quantileMin = 8 # c['quantile']
frames_window =  int(round(15.*fr)) #c['frames_window']

Fd = scipy.ndimage.percentile_filter(
            F_raw, quantileMin, (frames_window, 1))
corrected = F_raw - Fd
print corrected.shape

corrected_df = pd.DataFrame(data=corrected.T, columns=raw_df.columns)


# Get deconvolved traces (spikes):
print S.shape
S_df = pd.DataFrame(data=S.T, columns=raw_df.columns)


run_dir = os.path.join(acquisition_dir, run)

# Turn all this info into "standard" data frame arrays:
labels_df, raw_df, corrected_df, F0_df, dFF_df, spikes_df = caiman_to_darrays(run_dir, raw_df, corrected_df=corrected_df, dFF_df=dFF_df, F0_df=F0_df, S_df=S_df, output_dir=output_dir, ntiffs=4)


run_info = util.run_info_from_dfs(run_dir, raw_df, labels_df, traceid_dir=output_dir, trace_type='caiman', ntiffs=4)

stimconfigs_fpath = os.path.join(run_dir, 'paradigm', 'stimulus_configs.json')
with open(stimconfigs_fpath, 'r') as f:
    stimconfigs = json.load(f)
    
        
# Get label info:
sconfigs = util.format_stimconfigs(stimconfigs)
ylabels = labels_df['config'].values
groups = labels_df['trial'].values
tsecs = labels_df['tsec']
    

# Set data array output dir:
data_basedir = os.path.join(output_dir, 'data_arrays')
if not os.path.exists(data_basedir):
    os.makedirs(data_basedir)
data_fpath = os.path.join(data_basedir, 'datasets.npz')

#%%

print "Saving processed data...", data_fpath
np.savez(data_fpath, 
         raw=raw_df,
#             smoothedDF=smoothed_DF,
#             smoothedX=smoothed_X,
         dff=dFF_df,
         corrected=corrected_df,
         F0=F0_df,
#             frac=frac,
#             quantile=quantile,
         spikes=spikes_df,
         tsecs=tsecs,
         groups=groups,
         ylabels=ylabels,
         sconfigs=sconfigs,
         run_info=run_info)







#%%
def caiman_to_darrays(run_dir, raw_df, corrected_df=None, dFF_df=None, F0_df=None, S_df=None, output_dir='tmp', ntiffs=None, fmt='hdf5', trace_arrays_type='caiman'):
    
    xdata_df=None; labels_df=None; #F0_df=None; dFF_df=None; S_df=None;
    
    roixtrial_fn = 'roiXtrials_%s.%s' % (trace_arrays_type, fmt) # Procssed or SMoothed

    # Get SCAN IMAGE info for run:
    # -------------------------------------------------------------------------
    #run_dir = os.path.join(acquisition_dir, run)
    run = os.path.split(run_dir)[-1]
    with open(os.path.join(run_dir, '%s.json' % run), 'r') as fr:
        scan_info = json.load(fr)
    framerate = scan_info['frame_rate']
    frame_tsecs = np.array(scan_info['frame_tstamps_sec'])
    ntiffs_total = scan_info['ntiffs']
    if ntiffs is None:
        ntiffs = int(ntiffs_total)
        
    nvolumes_per_tiff = scan_info['nvolumes']
    
    # Need to make frame_tsecs span all TIFs:
    frame_tsecs_ext = np.hstack([frame_tsecs for i in range(ntiffs)])


    # Load MW info to get stimulus details:
    # -------------------------------------------------------------------------
    paradigm_dir = os.path.join(run_dir, 'paradigm')
    mw_fpath = [os.path.join(paradigm_dir, m) for m in os.listdir(paradigm_dir) if 'trials_' in m and m.endswith('json')][0]
    with open(mw_fpath,'r') as m:
        mwinfo = json.load(m)
    with open(os.path.join(paradigm_dir, 'stimulus_configs.json'), 'r') as s:
        stimconfigs = json.load(s)
    if 'frequency' in stimconfigs[stimconfigs.keys()[0]].keys():
        stimtype = 'gratings'
    elif 'fps' in stimconfigs[stimconfigs.keys()[0]].keys():
        stimtype = 'movie'
    else:
        stimtype = 'image'
    
    for conf, params in stimconfigs.items():
        if 'filename' in params.keys():
            params.pop('filename')
        stimconfigs[conf] = params
        
    
    # Load aligned frames--trial info:
    # -------------------------------------------------------------------------
    parsed_frames_fpath = [os.path.join(paradigm_dir, pfn) for pfn in os.listdir(paradigm_dir) if 'parsed_frames_' in pfn][0]
    parsed_frames = h5py.File(parsed_frames_fpath, 'r')
    
    # Get trial info:
    trial_list = sorted(parsed_frames.keys(), key=natural_keys)
    print "There are %i total trials across all .tif files." % len(trial_list)
    
    stimdurs = list(set([parsed_frames[t]['frames_in_run'].attrs['stim_dur_sec'] for t in trial_list]))
    assert len(stimdurs)==1, "More than 1 unique value for stim dur found in parsed_frames_ file!"
    nframes_on = round(int(stimdurs[0] * framerate))
    

#    
#    frame_df_list = []
#    drift_df_list = []
#    frame_times = []
#    trial_ids = []
#    config_ids = []

#    file_F0_df = None; #file_df = None;

    # Get all trials contained in current .tif file:
    trials_in_block = sorted([t for t in trial_list if parsed_frames[t]['frames_in_file'].attrs['aux_file_idx'] >= 4], key=natural_keys)


    # Get frame indices of the full trial 
    # -------------------------------------------------------------------------
    # (this includes PRE-stim baseline, stim on, and POST-stim iti):
    frame_indices = np.hstack([np.array(parsed_frames[t]['frames_in_file']) \
                                   for t in trials_in_block])

    # Check if frame indices are indexed relative to full run (all .tif files)
    # or relative to within-tif frames (i.e., a "block")
    block_indexed = True
    if all([all(parsed_frames[t]['frames_in_run'][:] == parsed_frames[t]['frames_in_file'][:]) for t in trial_list]):
        block_indexed = False


    # If frame indices are relative, make them absolute (full run):
    if block_indexed is True:
        full_file_dur = frame_tsecs[-1] + (1./framerate)
        frame_indices = np.hstack([frame_tsecs + (full_file_dur * fi) for fi in range(ntiffs_total)])
        #frame_indices = frame_indices - len(frame_tsecs)*fidx
    
    frame_index_adjust = 0
    if ntiffs_total != ntiffs:
        assert ntiffs < ntiffs_total, "Funky n tiffs specified. Total is %i, you specified too many (%i)" % (ntiffs_total, ntiffs)
        frame_index_adjust = len(frame_tsecs) * ntiffs
        frame_indices -= frame_index_adjust

    # Check that we have all frames needed (no cut off frames from end):
    assert frame_indices[-1] <= len(frame_tsecs) * ntiffs_total
    
    
    
    # Get relevant info for trial labels:
    # -------------------------------------------------------------------------
    excluded_params = ['filehash', 'stimulus', 'type']
    curr_trial_stimconfigs = [dict((k,v) for k,v in mwinfo[t]['stimuli'].iteritems() if k not in excluded_params) for t in trials_in_block]
    curr_config_ids = [k for trial_configs in curr_trial_stimconfigs for k,v in stimconfigs.iteritems() if v==trial_configs]
    config_labels = np.hstack([np.tile(conf, parsed_frames[t]['frames_in_file'].shape) for conf,trial in zip(curr_config_ids, trials_in_block)])
    
    trial_labels = np.hstack([np.tile(parsed_frames[t]['frames_in_run'].attrs['trial'], parsed_frames[t]['frames_in_file'].shape) for t in trials_in_block])
    stim_onset_idxs = np.array([parsed_frames[t]['frames_in_file'].attrs['stim_on_idx'] for t in trials_in_block]) - frame_index_adjust
    
    currtrials_df = raw_df.loc[frame_indices,:]  # DF (nframes_per_trial*ntrials_in_tiff X nrois)
#    if file_F0_df is not None:
#        currbaseline_df = file_F0_df.loc[frame_indices,:]
#    
    # Turn time-stamp array into (ntrials x nframes_per_trial) array:
    #trial_tstamps = frame_tsecs[frame_indices]        
    trial_tstamps = frame_tsecs_ext[frame_indices]
    nframes_per_trial = len(frame_indices) / len(trials_in_block)
    tsec_mat = np.reshape(trial_tstamps, (len(trials_in_block), nframes_per_trial))
    
    # Subtract frame_onset timestamp from each frame for each trial to get
    # time relative to stim ON:
    #tsec_mat -= np.tile(frame_tsecs[stim_onset_idxs].T, (tsec_mat.shape[1], 1)).T
    tsec_mat -= np.tile(frame_tsecs_ext[stim_onset_idxs].T, (tsec_mat.shape[1], 1)).T
    relative_tsecs = np.reshape(tsec_mat, (len(trials_in_block)*nframes_per_trial, ))
    
    # Get corresponding STIM CONFIG ids:
    if stimtype == 'grating' or stimtype=='gratings':
        excluded_params = ['stimulus', 'type']
    else:
        excluded_params = ['filehash', 'stimulus', 'type']
    curr_trial_stimconfigs = [dict((k,v) for k,v in mwinfo[t]['stimuli'].iteritems() if k not in excluded_params) for t in trials_in_block]
    curr_config_ids = [k for trial_configs in curr_trial_stimconfigs for k,v in stimconfigs.iteritems() if v==trial_configs]
    config_labels = np.hstack([np.tile(conf, parsed_frames[t]['frames_in_file'].shape) for conf,trial in zip(curr_config_ids, trials_in_block)])
    
    # Add current block of trial info:
#    frame_df_list.append(currtrials_df)
#    if file_F0_df is not None:
#        drift_df_list.append(currbaseline_df)
    
#    frame_times.append(relative_tsecs)
#    trial_ids.append(trial_labels)
#    config_ids.append(config_labels)

    xdata_df = currtrials_df.reset_index(drop=True)
    if dFF_df is not None:
        dFF_df = dFF_df.loc[frame_indices,:]  # DF (nframes_per_trial*ntrials_in_tiff X nrois)
        dFF_df = dFF_df.reset_index(drop=True)
    if F0_df is not None:
        F0_df = F0_df.loc[frame_indices,:]
        F0_df = F0_df.reset_index(drop=True)
    if S_df is not None:
        S_df = S_df.loc[frame_indices,:]
        S_df = S_df.reset_index(drop=True)        
    if corrected_df is not None:
        corrected_df = corrected_df.loc[frame_indices,:]
        corrected_df = corrected_df.reset_index(drop=True)   
        
    # Also collate relevant frame info (i.e., labels):
    tstamps = relative_tsecs
    trials = trial_labels
    configs = config_labels 
    
    stim_dur_sec = list(set([round(mwinfo[t]['stim_dur_ms']/1e3) for t in trial_list]))
    assert len(stim_dur_sec)==1, "more than 1 unique stim duration found in MW file!"
    stim_dur = stim_dur_sec[0]
        
    # Turn paradigm info into dataframe:
    labels_df = pd.DataFrame({'tsec': tstamps, 
                              'config': configs,
                              'trial': trials,
                              'stim_dur': np.tile(stim_dur, trials.shape)
                              }, index=xdata_df.index)
    
    if fmt == 'pkl':
        dfile = {'xdata': xdata_df, 'labels': labels_df}
        with open(os.path.join(output_dir, roixtrial_fn), 'wb') as f:
            pkl.dump(dfile, f, protocol=pkl.HIGHEST_PROTOCOL)
            
    elif fmt == 'hdf5':
        f = h5py.File(os.path.join(output_dir, roixtrial_fn),'a')
        for k in f.keys():
            del f[k]
        sa, saType = util.df_to_sarray(xdata_df)
        f.create_dataset('xdata', data=sa, dtype=saType)
        sa_labels, saType_labels = util.df_to_sarray(labels_df)
        f.create_dataset('labels', data=sa_labels, dtype=saType_labels)
        
        if F0_df is not None:
            sa_labels, saType_labels = util.df_to_sarray(F0_df)
            f.create_dataset('F0', data=sa_labels, dtype=saType_labels)
        if dFF_df is not None:
            sa_labels, saType_labels = util.df_to_sarray(dFF_df)
            f.create_dataset('dff', data=sa_labels, dtype=saType_labels)
        if S_df is not None:
            sa_labels, saType_labels = util.df_to_sarray(S_df)
            f.create_dataset('spikes', data=sa_labels, dtype=saType_labels)
        if corrected_df is not None:
            sa_labels, saType_labels = util.df_to_sarray(corrected_df)
            f.create_dataset('corrected', data=sa_labels, dtype=saType_labels)
        
    
        f.close()
    

    return labels_df, xdata_df, corrected_df, F0_df, dFF_df, S_df

# In[ ]:




#
## # Look at EACH step
#
## ## cNMF:  Preprocessing
#
## In[ ]:
#
#
#Yr,sn,g,psx = cnmf.pre_processing.preprocess_data(Yr
#            ,dview=dview
#            ,n_pixels_per_process=100,  noise_range = [0.25,0.5]
#            ,noise_method = 'logmexp', compute_g=False,  p = 2,
#             lags = 5, include_noise = False, pixels = None
#            ,max_num_samples_fft=3000, check_nan = True)
#
#
## # cNMF: Initialize components
#
## In[ ]:
#
#
#Ain, Cin, b_in, f_in, center=cnmf.initialization.initialize_components(Y
#            ,K=K, gSig=gSig, gSiz=None, ssub=1, tsub=1, nIter=5, maxIter=5, nb=1
#            , use_hals=False, normalize_init=True, img=None, method='greedy_roi'
#            , max_iter_snmf=500, alpha_snmf=10e2, sigma_smooth_snmf=(.5, .5, .5)
#            , perc_baseline_snmf=20)
#
#p1=nb_plot_contour(Cn,Ain,dims[0],dims[1],thr=0.9,face_color=None,
#                   line_color='black',alpha=0.4,line_width=2)
#bpl.show(p1)
#
#
## In[ ]:
#
#
#### initialize w/ HALS:
#
#AinAin,,  CinCin,,  b_inb_in,,  f_inf_in  ==  cnmfcnmf.initializationinitial.hals(Y, Ain, Cin, b_in, f_in, maxIter=5)
#p1=nb_plot_contour(Cn,Ain,dims[0],dims[1],thr=0.9,face_color=None,
#                   line_color='black',alpha=0.4,line_width=2)
#bpl.show(p1)
#
