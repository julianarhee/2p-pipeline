
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

#bpl.output_notebook()


# In[24]:


import os
import glob
import json
import h5py
import datetime
import copy
import hashlib

import cPickle as pkl
import numpy as np
import tifffile as tf
import pandas as pd
from pipeline.python.utils import natural_keys

from pipeline.python.paradigm import utils as util


def tifs_to_mmaps(fnames, dview=None, base_name='Yr', downsample_factor=(1, 1, 1), border_to_0=0):
    # read first few pages of each tif and find the min value to add:
    print "Finding min value to add..."
    min_values = [-np.min(tf.imread(fname, pages=[5])) for fname in fnames]
    add_to_movie = np.min([m for m in min_values if m > 0]) # Make sure we are only adding if min value is negative
    print add_to_movie
    name_new=cm.save_memmap_each(fnames, dview=dview,
                                 base_name=base_name, 
                                 resize_fact=downsample_factor, 
                                 remove_init=0, 
                                 idx_xy=None, 
                                 add_to_movie=add_to_movie,
                                 border_to_0=border_to_0 )
    return name_new
    
#%%
def get_mmap_file(fnames, excluded_files=[], 
                  dview=None, file_base='run', 
                  downsample_factor=(1, 1, 1), border_to_0=0, create_new=False):
    
    # Get tifs to convert into mmapped files:
    #fnames = glob.glob(os.path.join(tif_src_dir, '*.tif'))
    tif_src_dir = os.path.split(fnames[0])[0]
    print "Found %i tifs in src: %s" % (len(fnames), tif_src_dir)
    
    # Re-use exiting mmapped files, but may want to concatenate differing .tif files
    # Add prefix showing which files excluded, if any, so as not to overwrite.
    if len(excluded_files) > 0:
        print "** Excluding tif idxs:", excluded_files
    prefix = 'Yr%s' % 'x'.join([str(f) for f in excluded_files])
    
    # Use memmap input params as unique id for current files:
    mmap_info = {'source': tif_src_dir, 
                 'downsample_factor': downsample_factor,
                 'border_to_0': border_to_0}
    
    mhash = hashlib.md5(json.dumps(mmap_info, sort_keys=True, ensure_ascii=True)).hexdigest()
        
    # Create output dir:
    mmap_basedir = '%s_memmap_%s' % (os.path.split(fnames[0])[0], mhash[0:6])
    mmap_filedir = os.path.join(mmap_basedir, 'files')
    if not os.path.exists(mmap_filedir):
        os.makedirs(mmap_filedir)

    # Check for existing mmap files, and create them if not found:
    existing_files = glob.glob(os.path.join(mmap_filedir, '%s*.mmap' % file_base))
    if create_new is False:
        try:
            assert len(existing_files) == len(fnames), "Incorrect num .mmap files found (%i)" % len(existing_files)
            mmap_files = False
        except:
            mmap_files = True # If assertion fails, mmap_files

    if mmap_files:
        
        # Basename should include path, otherwise will be saved in current dir
        base_name = os.path.join(mmap_filedir, file_base) 
        print "Creating mmap files with base: %s" % base_name
        mmap_names = tifs_to_mmaps(fnames, dview=dview, base_name=base_name, 
                                   downsample_factor=downsample_factor,
                                   border_to_0=border_to_0)
    
    try:
        final_mmap = glob.glob(os.path.join(mmap_basedir, '%s*.mmap' % prefix))
        assert len(final_mmap)==1, "Full concatenated .mmap not found."
        return final_mmap[0]
    
    except Exception as e:
        # Join mmap files into 1:
        mmap_names = sorted(glob.glob(os.path.join(mmap_filedir, '%s*.mmap' % file_base)), key=natural_keys)
        mmap_names = sorted([m for mi, m in enumerate(mmap_names) if mi not in excluded_files], key=natural_keys)
        print "*** Combining %i of %i files. ***" % (len(mmap_names), len(fnames))
        
        if len(mmap_names) > 1:
            final_mmap = cm.save_memmap_join(mmap_names, base_name=os.path.join(mmap_basedir, prefix), n_chunks=20, dview=dview)
        else:
            print('One file only, not saving!')
            final_mmap = mmap_names[0]
        return final_mmap

#%%

def fit_cnmf_patches(images, cnmf_params, output_dir, n_processes=1, dview=None, verbose=True):
    
    '''
    Extract spatial and temporal components on patches
    '''
    
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
    '''
    Run cNMF extraction on seeded components
    '''
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
def get_cnmf_seeds(output_dir, cnmf_params, n_processes=None, dview=None, create_new=False):

    patch_fpath = os.path.join(output_dir, 'results_patch.pkl')

    if create_new or not os.path.exists(patch_fpath):
        print "Patch results not found! Creating new."
        
        cnm = fit_cnmf_patches(images, cnmf_params, output_dir, n_processes=n_processes, dview=dview)
        
        cnm.dview = None
        with open(patch_fpath, 'wb') as f:
            pkl.dump(cnm, f, protocol=pkl.HIGHEST_PROTOCOL)
        print "Save CNMF object to disk: %s" % patch_fpath
        
        # reassign dview before returning:
        cnm.dview = dview

    else:
        with open(patch_fpath, 'rb') as f:
            cnm = pkl.load(f)
    
    print(('Number of components:' + str(A_tot.shape[-1])))

    return cnm #A_tot, C_tot, YrA_tot, b_tot, f_tot, sn_tot


#%% start cluster for efficient computation


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


single_thread=False

# Start cluster:
n_processes = 12

c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=n_processes, single_thread=False)
    

print n_processes
print dview


#%% ### Get TIF source

rootdir = '/mnt/odyssey'

rootdir = '/n/coxfs01/2p-data'

#rootdir = '/Volumes/coxfs01/2p-data'

animalid = 'CE077'
session = '20180629'
acquisition = 'FOV1_zoom1x'
run = 'gratings_rotating_drifting'
create_new = False

acquisition_dir = os.path.join(rootdir, animalid, session, acquisition)

#tif_src_dir = os.path.join(acquisition_dir, 'caiman_test_data')


fnames = sorted(glob.glob(os.path.join(acquisition_dir, run, 'processed', 'processed001*', 'mcorrected_*', '*.tif')), key=natural_keys)

#tif_src_dir = os.path.split(fnames[0])[0]
#
#
#fnames = glob.glob(os.path.join(tif_src_dir, '*.tif'))
#print "Found %i tifs in src: %s" % (len(fnames), tif_src_dir)
#
#run_dir = os.path.join(acquisition_dir, run)

si_info_path = (os.path.join(acquisition_dir, run, '%s.json' % run))
with open(si_info_path, 'r') as f:
    si_info = json.load(f)
fr = si_info['frame_rate'] # 44.69


#%%

#datestr = '20180622_17_24_42' #None #'20180622_15_26_53'
datestr = None # '20180720_12_10_07'
excluded_files = []


#% Set output dir for figures and results:

output_basedir = os.path.join(acquisition_dir, run, 'traces', 'cnmf')
if not os.path.exists(output_basedir):
    os.makedirs(output_basedir)
    
if datestr is None:
    datestr = datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")
#output_dir = os.path.join(tif_src_dir, 'cnmf_%s' % datestr)

output_dir = os.path.join(output_basedir, 'cnmf_%s' % datestr)

if not os.path.exists(os.path.join(output_dir, 'figures')):
    os.makedirs(os.path.join(output_dir, 'figures'))
if not os.path.exists(os.path.join(output_dir, 'results')):
    os.makedirs(os.path.join(output_dir, 'results'))
print output_dir
    
#%% ### Create memmapped files:
    
downsample_factor = (1,1,1) # Use fractions if want to downsample
border_to_0 = 2

#fnames = [f for f in fnames if fi not in excluded_files]
fbase = run.split('_')[0]

fname_new = get_mmap_file(fnames, excluded_files=excluded_files, file_base=fbase, 
                              dview=dview, 
                              downsample_factor=downsample_factor, 
                              border_to_0=border_to_0)


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
manual_seed = True
if manual_seed:
    #results_basename = 'results_seed'
    
    if original_source:
        rid = 'rois001'
        rid_dir = glob.glob(os.path.join(rootdir, animalid, session, 'ROIs', '%s*' % rid))[0]
        print rid_dir
        
        mask_fpath = os.path.join(rid_dir, 'masks.hdf5')
    else:
        mask_fpath = glob.glob(os.path.join(acquisition_dir, run, 'traces', 'traces001*', 'MASKS.hdf5'))[0]
    
    # Load masks:
    maskfile = h5py.File(mask_fpath, 'r')
    print "Loaded maskfile:", maskfile.keys()
    if original_source:
        ref_file = maskfile.keys()[0]
        masks = maskfile[ref_file]['masks']['Slice01']
        # Binarze and reshape:
        nrois, d1, d2 = masks.shape
        A_in = np.reshape(masks, (nrois, d1*d2))
        A_in[A_in>0] = 1
        A_in = A_in.astype(bool).T # A_in should be of shape (npixels, nrois)
    else:
        ref_file = 'File005'
        A_in = maskfile[ref_file]['Slice01']['maskarray'][:] # Already in shape npixels x nrois
        A_in[A_in>0] = 1
        A_in = A_in.astype(bool)
    print A_in.shape
    

    

    print "Reshaped seeded spatial comps:", A_in.shape
    
    C_in = None; b_in = None; f_in = None; 
    
#    gnb = 1
#    gSig = (3, 3)
#    p = 2

else:
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
                   'fname_new': fname_new,
                   'excluded_files': excluded_files}
    
    with open(os.path.join(output_dir, 'results', 'cnmf_params.json'), 'w') as f:
        json.dump(cnmf_params, f, indent=4, sort_keys=True)
        
    cnm = get_cnmf_seeds(output_dir, cnmf_params, n_processes=n_processes, dview=dview, create_new=create_new)
    
    #%%
    # #### Evaluate components:
    
    #fr = 44.69             # approximate frame rate of data
    decay_time = 4.0 #.0    # length of transient
    min_SNR = 1.2      # peak SNR for accepted components (if above this, acept)
    rval_thr = 0.7     # space correlation threshold (if above this, accept)
    use_cnn = False # use the CNN classifier
    min_cnn_thr = None  # if cnn classifier predicts below this value, reject
    dims = [d1, d2]
    gSig = (3, 3)
    
    
    # NOTE:  additional methods not pulled from source as of 2018/06/22...
    #cnm.evaluate_components(fr=fr, decay_time=decay_time, min_SNR=min_SNR,
    #                        rval_thr=rval_thr, use_cnn=use_cnn, min_cnn_thr=min_cnn_thr)
    
    idx_components, idx_components_bad, SNR_comp, r_values, cnn_preds = cm.components_evaluation.estimate_components_quality_auto(
                                        images, cnm.A, cnm.C, cnm.b, cnm.f,
                                         cnm.YrA, fr, decay_time, gSig, dims,
                                         dview=dview, min_SNR=min_SNR,
                                         r_values_min=rval_thr, use_cnn=use_cnn,
                                         thresh_cnn_min=min_cnn_thr)
    
    # Save results, params, and figure with unique datestr:
    eval_datestr = datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")
    
    # Save json for params:
    eval_params = {'fr': fr, 
                   'decay_time': decay_time,
                   'min_SNR': min_SNR, 
                   'rval_thr': rval_thr,
                   'use_cnn': use_cnn, 
                   'min_cnn_thr': min_cnn_thr,
                   'dims': dims, 
                   'gSig': gSig, 
                   'n_passed': len(idx_components)}
    
    # Save params in easy-read format:
    with open(os.path.join(output_dir, 'figures', 'evalparams_patch_%s.json' % eval_datestr), 'w') as f:
        json.dump(eval_params, f, indent=4, sort_keys=True)
    
    # Save evaluation results (plus all info we want to keep):
    np.savez(os.path.join(output_dir, 'evaluation_patch_%s.npz' % eval_datestr), 
                 A_tot=A_tot, C_tot=C_tot, YrA_tot=YrA_tot, b_tot=b_tot, f_tot=f_tot, sn_tot=sn_tot,
                 idx_components=idx_components, idx_components_bad=idx_components_bad,
                 SNR_comp=SNR_comp, r_values=r_values, 
                options=cnm.options, evalparams=eval_params)
    
    
    # Save correposnding eval results figure:
    pl.figure(figsize=(15,5))
    pl.subplot(1,2,1)
    crd = plot_contours(A_tot.tocsc()[:, idx_components], Cn, thr=0.9)
    pl.subplot(1,2,2)
    crd = plot_contours(A_tot.tocsc()[:, idx_components_bad], Cn, thr=0.9)
    pl.savefig(os.path.join(output_dir, 'figures', 'evaluation_patch_%s.png' % eval_datestr))
    pl.close()
    
    
    print "Keeping %i components." % len(idx_components)

    # Run patches above to get initial components:
    #results_basename = 'results_cnmf'
    A_in, C_in, b_in, f_in = cnm.A[:, idx_components], cnm.C[idx_components], cnm.b, cnm.f
    
#%% 

# #############################################################################
# REFINE seeded components:
# #############################################################################


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

quantileMin = 10
frames_window = int(round(30.*fr))
F_dff = detrend_df_f(cnm2.A, cnm2.b, cnm2.C, cnm2.f, YrA=cnm2.YrA,
                     quantileMin=quantileMin, frames_window=frames_window)

pl.figure(figsize=(20,5))
pl.plot(F_dff[0,:])
pl.savefig(os.path.join(output_dir, 'figures', 'example_roi0_dff.png'));
pl.close()

cnm2.F_dff = F_dff
cnm2.quantileMin = quantileMin
cnm2.frames_window = frames_window


#% Save refined cnm with dff and S:
cnm2.dview = None
with open(os.path.join(output_dir, 'results', 'results_refined_%s.pkl' % refined_datestr), 'wb') as f:
    pkl.dump(cnm2, f, protocol=pkl.HIGHEST_PROTOCOL)
cnm2.dview = dview

# #### Save refined run:
cnm2.dview=None
np.savez(os.path.join(output_dir, 'results', 'results_refined_%s.npz' % refined_datestr), 
         cnm=cnm2)
cnm2.dview=dview



#%%
# #### Evaluate refined run:


## In[ ]:

print "*** Evaluating final run ***"
#fr = 44.69             # approximate frame rate of data
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


cnm2.idx_components = idx_components
cnm2.idx_components_bad = idx_components_bad
cnm2.SNR_comp = SNR_comp
cnm2.r_values = r_values
cnm2.options['quality'] = eval_params
cnm2.Cn = Cn

cnm2.dview=None
results_fpath = os.path.join(output_dir, 'results', 'results_refined_%s.npz' % refined_datestr)
np.savez(results_fpath, cnm=cnm2)
cnm2.dview=dview

#
#np.savez(os.path.join(output_dir, 'evaluation_refined_%s.npz' % eval2_datestr), 
#         Cn=Cn, 
#         A=cnm2.A.todense(), C=cnm2.C, b=cnm2.b, f=cnm2.f, YrA=cnm2.YrA, sn=cnm2.sn, 
#         bl = cnm2.bl,
#         d1=d1, d2=d2, 
#         eval_params = eval_params,
#         idx_components=idx_components, idx_components_bad=idx_components_bad,
#         SNR_comp=SNR_comp, r_values=r_values,
#         S=cnm2.S, F_dff=F_dff, quantileMin=quantile, frames_window=frames_window)


#%%

# ### Play around with contour visualization...


#results_fpath = os.path.join(output_dir, 'results', 'eval_final.npz')
results_fpath = glob.glob(os.path.join(output_dir, 'results', 'results_refined_*.npz'))[0]

remove_bad_components = False

cnmd = np.load(results_fpath)
cnm = cnmd['cnm'][()]


if remove_bad_components:
    A = cnm.A[:, cnm.idx_components]
    C =cnm.C[cnm.idx_components]
    mov_fn_append = '_idxcomp'
else:
    A = cnm.A
    C = cnm.C
    mov_fn_append = '_all'

b = cnm.b
f = cnm.f
    
Cn = cnm.Cn

crd = plot_contours(A, Cn, thr_method='nrg', nrgthr=0.5)
pl.savefig(os.path.join(output_dir, 'figures', 'final_idxcomps_nrg.png'))
pl.close()



#%% reconstruct denoised movie


ntiffs = si_info['ntiffs'] - len(excluded_files)
print "Creating movie for 1 out of %i tiffs." % ntiffs
nvolumes = T/ntiffs
C = C[:, 0:int(nvolumes)]
f = f[:, 0:int(nvolumes)]

#denoised = cm.movie(cnm2.A.dot(cnm2.C) +
#                    cnm2.b.dot(cnm2.f)).reshape(dims + (-1,), order='F').transpose([2, 0, 1])
#


denoised = cm.movie(A.dot(C) + b.dot(f)).reshape(tuple(dims) + (-1,), order='F').transpose([2, 0, 1])

denoised.save(os.path.join(output_dir, 'figures', 'denoised_refined_File001%s.tif' % mov_fn_append))





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
    print log_file
    os.remove(log_file)


#%%

# #############################################################################
# #### Align traces to trials
# #############################################################################

#results_path = os.path.join(output_dir, 'results_analysis1b.npz')

#results_fpath = os.path.join(output_dir, 'results', 'eval_final.npz')

results_fpath = glob.glob(os.path.join(output_dir, 'results', 'results_refined_*.npz'))[0]
print "Loading results: %s" % results_fpath

cnmd = np.load(results_fpath)
cnm = cnmd['cnm'][()]



nrois = cnm.A.shape[-1]
F_raw = cnm.C + cnm.YrA
A = cnm.A
C = cnm.C
YrA = cnm.YrA
S = cnm.S
F_dff = cnm.F_dff
b = cnm.b
f = cnm.f

if remove_bad_components:
    idx_components = cnm.idx_components
    nrois = len(idx_components)
    F_raw = C[idx_components] + YrA[idx_components]
    A = A[:, idx_components]
    C = C[idx_components]
    S = S[idx_components]
    F_dff = F_dff[idx_components]
    YrA = YrA[idx_components]

print F_raw.shape
raw_df = pd.DataFrame(data=F_raw.T, columns=[['roi%05d' % int(i+1) for i in range(nrois)]])


# Get baseline:
B = A.T.dot(b).dot(f)
print B.shape
F0_df = pd.DataFrame(data=B.T, columns=raw_df.columns)


# Get DFF:
dFF_df = pd.DataFrame(data=F_dff.T, columns=[['roi%05d' % int(i+1) for i in range(nrois)]])
print dFF_df.shape


# Get Drift-Corrected "raw":
quantileMin = cnm.quantileMin # c['quantile']
frames_window = cnm.frames_window # int(round(15.*fr)) #c['frames_window']

Fd = scipy.ndimage.percentile_filter(
            F_raw, quantileMin, (frames_window, 1))
corrected = F_raw - Fd
print corrected.shape

corrected_df = pd.DataFrame(data=corrected.T, columns=raw_df.columns)


# Get deconvolved traces (spikes):
print S.shape
S_df = pd.DataFrame(data=S.T, columns=raw_df.columns)


run_dir = os.path.join(acquisition_dir, run)

ntiffs = si_info['ntiffs'] - len(excluded_files)



#%% Turn all this info into "standard" data frame arrays:
labels_df, raw_df, corrected_df, F0_df, dFF_df, spikes_df = caiman_to_darrays(run_dir, raw_df, 
                                                                              corrected_df=corrected_df, 
                                                                              dFF_df=dFF_df, 
                                                                              F0_df=F0_df, 
                                                                              S_df=S_df, 
                                                                              output_dir=output_dir, 
                                                                              ntiffs=ntiffs,
                                                                              excluded_files=excluded_files)

# Get stimulus / trial info:
run_info = util.run_info_from_dfs(run_dir, raw_df, labels_df, traceid_dir=output_dir, trace_type='caiman', ntiffs=ntiffs)

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

#%

print "Saving processed data...", data_fpath
#np.savez(data_fpath, 
#         raw=raw_df,
##             smoothedDF=smoothed_DF,
##             smoothedX=smoothed_X,
#         dff=dFF_df,
#         corrected=corrected_df,
#         F0=F0_df,
##             frac=frac,
##             quantile=quantile,
#         spikes=spikes_df,
#         tsecs=tsecs,
#         groups=groups,
#         ylabels=ylabels,
#         sconfigs=sconfigs,
#         run_info=run_info)

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
#             meanstim=meanstim_values, 
#             zscore=zscore_values,
#             meanstimdff=meanstimdff_values,
         labels_data=labels_df,
         labels_columns=labels_df.columns.tolist(),
         run_info=run_info)






#%%
def caiman_to_darrays(run_dir, raw_df, corrected_df=None, dFF_df=None, 
                      F0_df=None, S_df=None, output_dir='tmp', ntiffs=None, excluded_files=[],
                      fmt='hdf5', trace_arrays_type='caiman'):
    
    xdata_df=None; labels_df=None; #F0_df=None; dFF_df=None; S_df=None;
    
    roixtrial_fn = 'roiXtrials_%s.%s' % (trace_arrays_type, fmt) # Procssed or SMoothed

    # Get SCAN IMAGE info for run:
    # -------------------------------------------------------------------------
    run = os.path.split(run_dir)[-1]
    with open(os.path.join(run_dir, '%s.json' % run), 'r') as fr:
        scan_info = json.load(fr)
    framerate = scan_info['frame_rate']
    frame_tsecs = np.array(scan_info['frame_tstamps_sec'])
    ntiffs_total = scan_info['ntiffs']
    if ntiffs is None:
        ntiffs = int(ntiffs_total)
            
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
    #parsed_frames_fpath = [os.path.join(paradigm_dir, pfn) for pfn in os.listdir(paradigm_dir) if 'parsed_frames_' in pfn][0]
    parsed_frames_fpath = glob.glob(os.path.join(paradigm_dir, 'parsed_frames_*.hdf5'))[0]
    parsed_frames = h5py.File(parsed_frames_fpath, 'r')
    
    # Get trial info:
    trial_list = sorted(parsed_frames.keys(), key=natural_keys)
    print "There are %i total trials across all .tif files." % len(trial_list)
    
    stimdurs = list(set([parsed_frames[t]['frames_in_run'].attrs['stim_dur_sec'] for t in trial_list]))
    #assert len(stimdurs)==1, "More than 1 unique value for stim dur found in parsed_frames_ file!"
    if len(stimdurs) > 1:
        print "Multiple stim durations found. This is a MOVING stim expmt."
    #nframes_on = round(int(stimdurs[0] * framerate))
    
    frame_df_list = []
    frame_times = []
    trial_ids = []
    config_ids = []
    
    
    
    # Get all trials contained in current .tif file:
    #trials_in_block = sorted([t for t in trial_list if parsed_frames[t]['frames_in_file'].attrs['aux_file_idx'] >= 4], key=natural_keys)
    trials_in_block = sorted([t for t in trial_list if parsed_frames[t]['frames_in_file'].attrs['aux_file_idx'] not in excluded_files], key=natural_keys)
        
  

    # Get frame indices of the full trial 
    # -------------------------------------------------------------------------
    # (this includes PRE-stim baseline, stim on, and POST-stim iti):
    frame_indices = np.hstack([np.array(parsed_frames[t]['frames_in_file']) \
                               for t in trials_in_block])
    stim_onset_idxs = np.array([parsed_frames[t]['frames_in_file'].attrs['stim_on_idx'] \
                                for t in trials_in_block])

    stim_offset_idxs = np.array([mwinfo[t]['frame_stim_off'] for t in trials_in_block])
    
    sdurs = []
    for on,off in zip(stim_onset_idxs, stim_offset_idxs):
        dur = round((off - on) / framerate)
        sdurs.append(dur)
    print "unique durs:", list(set(sdurs))

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
    
    
    
    # Get Stimulus info for each trial:        
    #excluded_params = [k for k in mwinfo[t]['stimuli'].keys() if k in stimconfigs[stimconfigs.keys()[0]].keys()]
    excluded_params = ['filehash', 'stimulus', 'type', 'rotation_range', 'phase'] # Only include param keys saved in stimconfigs
    #print "---- ---- EXCLUDED PARAMS:", excluded_params

    curr_trial_stimconfigs = [dict((k,v) for k,v in mwinfo[t]['stimuli'].iteritems() \
                                   if k not in excluded_params) for t in trials_in_block]
    varying_stim_dur = False
    # Add stim_dur if included in stim params:
    if 'stim_dur' in stimconfigs[stimconfigs.keys()[0]].keys():
        varying_stim_dur = True
        for ti, trial in enumerate(sorted(trials_in_block, key=natural_keys)):
            #print curr_trial_stimconfigs[ti]
            curr_trial_stimconfigs[ti]['stim_dur'] = round(mwinfo[trial]['stim_dur_ms']/1E3)
    
    # Get corresponding configXXX label:
    #print curr_trial_stimconfigs
    curr_config_ids = [k for trial_configs in curr_trial_stimconfigs \
                           for k,v in stimconfigs.iteritems() if v==trial_configs]
    # Combine into array that matches size of curr file_df:
    config_labels = np.hstack([np.tile(conf, parsed_frames[t]['frames_in_file'].shape) \
                               for conf, t in zip(curr_config_ids, trials_in_block)])
    trial_labels = np.hstack(\
                    [np.tile(parsed_frames[t]['frames_in_run'].attrs['trial'], parsed_frames[t]['frames_in_file'].shape) \
                     for t in trials_in_block])
    
    assert len(config_labels)==len(trial_labels), "Mismatch in n frames per trial, %s" % dfn

    # #### Parse full file into "trial" structure using frame indices:
    currtrials_df = raw_df.loc[frame_indices,:]  # DF (nframes_per_trial*ntrials_in_tiff X nrois)

    # Turn time-stamp array into RELATIVE time stamps (relative to stim onset):
    #pre_iti = 1.0
    trial_tstamps = frame_tsecs_ext[frame_indices]  
    if varying_stim_dur:
        

#        iti = list(set([round(mwinfo[t]['iti_dur_ms']/1E3) for t in trials_in_block]))[0] - pre_iti - 0.05
#        stimdurs = [int(round((mwinfo[t]['stim_dur_ms']/1E3) * framerate)) for t in trials_in_block]
#        
#        # Each trial has a different stim dur structure:
#        trial_ends = [ stimoff + int(round(iti * framerate)) for stimoff in stim_offset_idxs]
#        trial_end_idxs = []
#        for te in trial_ends:
#            if te not in frame_indices:
#                eix = np.where(abs(frame_indices-te)==min(abs(frame_indices-te)))[0][0]
#                #print frame_indices[eix]
#            else:
#                eix = [i for i in frame_indices].index(te)
#            trial_end_idxs.append(eix)
        
        trial_end_idxs = np.where(np.diff(frame_indices) > 1)[0]
        trial_end_idxs = np.append(trial_end_idxs, len(trial_tstamps)-1)
        trial_start_idxs = [0]
        trial_start_idxs.extend([i+1 for i in trial_end_idxs[0:-1]])
        
        
        relative_tsecs = []; #curr_trial_start_ix = 0;
        #prev_trial_end = 0;
        #for ti, (stim_on_ix, trial_end_ix) in enumerate(zip(sorted(stim_onset_idxs), sorted(trial_end_idxs))):    
        for ti, (trial_start_ix, trial_end_ix) in enumerate(zip(sorted(trial_start_idxs), sorted(trial_end_idxs))):
            
            stim_on_ix = stim_onset_idxs[ti]
            #assert curr_trial_start_ix > prev_trial_end, "Current start is %i frames prior to end!" % (prev_trial_end - curr_trial_start_ix)
            #print ti, trial_end_ix - trial_start_ix
            
            
#            if trial_end_ix==trial_ends[-1]:
#                curr_tstamps = trial_tstamps[trial_start_ix:]
#            else:
            curr_tstamps = trial_tstamps[trial_start_ix:trial_end_ix+1]
            
            zeroed_tstamps = curr_tstamps - frame_tsecs_ext[stim_on_ix]
            print ti, zeroed_tstamps[0]
            relative_tsecs.append(zeroed_tstamps)
            #prev_trial_end = trial_end_ix
            
        relative_tsecs = np.hstack(relative_tsecs)

    else:
        # All trials have the same structure:
        nframes_per_trial = len(frame_indices) / len(trials_in_block)
        tsec_mat = np.reshape(trial_tstamps, (len(trials_in_block), nframes_per_trial))
        
        # Subtract frame_onset timestamp from each frame for each trial to get
        # time relative to stim ON:
        tsec_mat -= np.tile(frame_tsecs_ext[stim_onset_idxs].T, (tsec_mat.shape[1], 1)).T
        relative_tsecs = np.reshape(tsec_mat, (len(trials_in_block)*nframes_per_trial, ))
    

    # Add current block of trial info:
    frame_df_list.append(currtrials_df)

    frame_times.append(relative_tsecs)
    trial_ids.append(trial_labels)
    config_ids.append(config_labels)
    
    

    xdata_df = pd.concat(frame_df_list, axis=0).reset_index(drop=True)

    # Also collate relevant frame info (i.e., labels):
    tstamps = np.hstack(frame_times)
    trials = np.hstack(trial_ids)
    configs = np.hstack(config_ids)
    if 'stim_dur' in stimconfigs[stimconfigs.keys()[0]].keys():
        stim_durs = np.array([stimconfigs[c]['stim_dur'] for c in configs])
    else:
        stim_durs = list(set([round(mwinfo[t]['stim_dur_ms']/1e3) for t in trial_list]))
    nframes_on = np.array([int(round(dur*framerate)) for dur in stim_durs])
   
    
    
#    
#    
#    # Get relevant info for trial labels:
#    # -------------------------------------------------------------------------
#    #excluded_params = ['filehash', 'stimulus', 'type']
#    excluded_params = ['stimulus', 'type', 'rotation_range', 'phase']
#
#    curr_trial_stimconfigs = [dict((k,v) for k,v in mwinfo[t]['stimuli'].iteritems() if k not in excluded_params) for t in trials_in_block]
#    curr_config_ids = [k for trial_configs in curr_trial_stimconfigs for k,v in stimconfigs.iteritems() if v==trial_configs]
#    config_labels = np.hstack([np.tile(conf, parsed_frames[t]['frames_in_file'].shape) for conf,trial in zip(curr_config_ids, trials_in_block)])
#    
#    trial_labels = np.hstack([np.tile(parsed_frames[t]['frames_in_run'].attrs['trial'], parsed_frames[t]['frames_in_file'].shape) for t in trials_in_block])
#    stim_onset_idxs = np.array([parsed_frames[t]['frames_in_file'].attrs['stim_on_idx'] for t in trials_in_block]) - frame_index_adjust
#    
#    currtrials_df = raw_df.loc[frame_indices,:]  # DF (nframes_per_trial*ntrials_in_tiff X nrois)
##    if file_F0_df is not None:
##        currbaseline_df = file_F0_df.loc[frame_indices,:]
##    
#    # Turn time-stamp array into (ntrials x nframes_per_trial) array:
#    #trial_tstamps = frame_tsecs[frame_indices]        
#    trial_tstamps = frame_tsecs_ext[frame_indices]
#    nframes_per_trial = len(frame_indices) / len(trials_in_block)
#    tsec_mat = np.reshape(trial_tstamps, (len(trials_in_block), nframes_per_trial))
#    
#    # Subtract frame_onset timestamp from each frame for each trial to get
#    # time relative to stim ON:
#    #tsec_mat -= np.tile(frame_tsecs[stim_onset_idxs].T, (tsec_mat.shape[1], 1)).T
#    tsec_mat -= np.tile(frame_tsecs_ext[stim_onset_idxs].T, (tsec_mat.shape[1], 1)).T
#    relative_tsecs = np.reshape(tsec_mat, (len(trials_in_block)*nframes_per_trial, ))
#    
#    # Get corresponding STIM CONFIG ids:
#    if stimtype == 'grating' or stimtype=='gratings':
#        excluded_params = ['stimulus', 'type', 'rotation_range', 'phase']
#    else:
#        excluded_params = ['filehash', 'stimulus', 'type']
#    curr_trial_stimconfigs = [dict((k,v) for k,v in mwinfo[t]['stimuli'].iteritems() if k not in excluded_params) for t in trials_in_block]
#    curr_config_ids = [k for trial_configs in curr_trial_stimconfigs for k,v in stimconfigs.iteritems() if v==trial_configs]
#    config_labels = np.hstack([np.tile(conf, parsed_frames[t]['frames_in_file'].shape) for conf,trial in zip(curr_config_ids, trials_in_block)])
#    
    # Format pd DataFrames:
    xdata_df = currtrials_df.reset_index(drop=True)
    if dFF_df is not None:
        currtrials_df = dFF_df.loc[frame_indices,:]  # DF (nframes_per_trial*ntrials_in_tiff X nrois)
        dFF_df = currtrials_df.reset_index(drop=True)
    if F0_df is not None:
        currtrials_df = F0_df.loc[frame_indices,:]
        F0_df = currtrials_df.reset_index(drop=True)
    if S_df is not None:
        currtrials_df = S_df.loc[frame_indices,:]
        S_df = currtrials_df.reset_index(drop=True)        
    if corrected_df is not None:
        currtrials_df = corrected_df.loc[frame_indices,:]
        corrected_df = currtrials_df.reset_index(drop=True)   
        
#    # Also collate relevant frame info (i.e., labels):
#    tstamps = relative_tsecs
#    trials = trial_labels
#    configs = config_labels 
#    
#    if 'stim_dur' in stimconfigs[stimconfigs.keys()[0]].keys():
#        stim_durs = np.array([stimconfigs[c]['stim_dur'] for c in configs])
#    else:
#        stim_durs = list(set([round(mwinfo[t]['stim_dur_ms']/1e3) for t in trial_list]))
#        assert len(stim_durs)==1, "more than 1 unique stim duration found in MW file!"
#    
#    nframes_on = np.array([int(round(dur*framerate)) for dur in stim_durs])
#   
#    

        
    # Turn paradigm info into dataframe:
    labels_df = pd.DataFrame({'tsec': tstamps, 
                              'config': configs,
                              'trial': trials,
                              'stim_dur': stim_durs #np.tile(stim_dur, trials.shape)
                              }, index=xdata_df.index)
    
    ons = [int(np.where(t==0)[0]) for t in labels_df.groupby('trial')['tsec'].apply(np.array)]
    assert len(list(set(ons))) == 1, "Stim onset index has multiple values..."
    stim_on_frame = list(set(ons))[0]
    stim_ons_df = pd.DataFrame({'stim_on_frame': np.tile(stim_on_frame, (len(tstamps),)),
                                'nframes_on': nframes_on,
                                }, index=labels_df.index)
    labels_df = pd.concat([stim_ons_df, labels_df], axis=1)
    
    
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
