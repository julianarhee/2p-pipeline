
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

import optparse
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
from pipeline.python.paradigm import plot_responses as pplot


#%%

def tifs_to_mmaps(fnames, dview=None, base_name='Yr', downsample_factor=(1, 1, 1), border_to_0=0, add_offset=False):
    # read first few pages of each tif and find the min value to add:
    add_to_movie = 0
    if add_offset:
        print "Finding min value to add..."
        min_values = [-np.min(tf.imread(fname, pages=[5])) for fname in fnames]
        add_to_movie = np.min([m for m in min_values if m > 0]) # Make sure we are only adding if min value is negative
        print add_to_movie
    name_new=cm.save_memmap_each(fnames, dview=dview,
                                 base_name=base_name, 
                                 resize_fact=downsample_factor, 
                                 remove_init=0, 
                                 idx_xy=None, 
                                 #add_to_movie=add_to_movie,
                                 border_to_0=border_to_0 )
    return name_new
    
#%%
def get_mmap_file(fnames, excluded_files=[], 
                  dview=None, file_base='run', 
                  downsample_factor=(1, 1, 1), border_to_0=0, create_new=False, add_offset=False):
    
    # Check fnames list to see if this is multiple runs combined:
    run_list = list(set([ os.path.split(f.split('/processed/')[0])[-1] for f in fnames ]))
    if len(run_list) > 1:
        combined_runs = True
        acquisition_dir = os.path.split(fnames[0].split('/processed/')[0])[0]
        processid, processtype, _ = fnames[0].split('/processed/')[-1].split('/')
        stim_type = run_list[0].split('_run')[0]
        combined_run_name = 'combined_%s_static' % stim_type
        mmap_subdir = '_'.join([processid.split('_')[0], processtype.split('_')[0], 'memmap'])
        mmap_base = os.path.join(acquisition_dir, combined_run_name, 'processed', mmap_subdir)
        tif_src_dir = sorted(list(set([ os.path.split(f)[0] for f in fnames ])), key=natural_keys)
    else:
        combined_runs = False
        tif_src_dir = os.path.split(fnames[0])[0]
        mmap_base = '%s_memmap' % (os.path.split(fnames[0])[0])

    print "Memmapping %i tifs to output dir: %s" % (len(fnames), mmap_base)

    # Re-use exiting mmapped files, but may want to concatenate differing .tif files
    # Add prefix showing which files excluded, if any, so as not to overwrite.
    if len(excluded_files) > 0:
        print "** Excluding tif idxs:", excluded_files
    prefix = 'Yr%s' % 'x'.join([str(f) for f in excluded_files])
    
    # Use memmap input params as unique id for current files:
    mmap_info = {'source': tif_src_dir, 
                 'downsample_factor': downsample_factor,
                 'border_to_0': border_to_0,
                 'excluded_files': list(excluded_files),
                 'add_offset': add_offset}
 
    # Create output dir:
    mhash = hashlib.md5(json.dumps(mmap_info, sort_keys=True, ensure_ascii=True)).hexdigest()

    mmap_basedir = '%s_%s' % (mmap_base, mhash[0:6]) #'%s_memmap_%s' % (os.path.split(fnames[0])[0], mhash[0:6])
    mmap_filedir = os.path.join(mmap_basedir, 'files')
    if not os.path.exists(mmap_filedir):
        os.makedirs(mmap_filedir)
        
    with open(os.path.join(mmap_basedir, 'mmap_info.json'), 'w') as f:
        json.dump(mmap_info, f, indent=4, sort_keys=True)
        
    # Check for existing mmap files, and create them if not found:
    existing_files = glob.glob(os.path.join(mmap_filedir, '%s*.mmap' % file_base))
    do_memmap = True
    if create_new is False:
        try:
            assert len(existing_files) == len(fnames), "Incorrect num .mmap files found (%i)" % len(existing_files)
            do_memmap = False
        except:
            do_memmap = True # If assertion fails, mmap_files

    if do_memmap:
        # Basename should include path, otherwise will be saved in current dir
        base_name = os.path.join(mmap_filedir, file_base) 
        print "Creating mmap files with base: %s" % base_name
        mmap_names = tifs_to_mmaps(fnames, dview=dview, base_name=base_name, add_offset=add_offset, 
                                   downsample_factor=downsample_factor,
                                   border_to_0=border_to_0)
    
    try:
        final_mmap = glob.glob(os.path.join(mmap_basedir, '%s*.mmap' % prefix))
        assert len(final_mmap)==1, "Full, concatenated master .mmap not found."
        return final_mmap[0], mmap_basedir
    
    except Exception as e:
        # Join mmap files into 1:
        mmap_names = sorted(glob.glob(os.path.join(mmap_filedir, '%s*.mmap' % file_base)), key=natural_keys)
        mmap_names = sorted([m for mi, m in enumerate(mmap_names) if mi not in excluded_files], key=natural_keys)
        print "*** Combining %i of %i files. ***" % (len(mmap_names), len(fnames))
        
        if len(mmap_names) > 1:
            final_mmap = cm.save_memmap_join(mmap_names, base_name=os.path.join(mmap_basedir, prefix), n_chunks=20, dview=dview)
        else:
            print('One file only, not saving master memmap file.')
            final_mmap = mmap_names[0]
            
        return final_mmap, mmap_basedir

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
                    #noise_range=cnmf_params['noise_range'],
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
    cnm.options['preprocess_params']['noise_range'] = cnmf_params['noise_range']
    cnm.options['temporal_params']['noise_range'] = cnmf_params['noise_range']
    cnm.options['temporal_params']['s_min'] = cnmf_params['s_min']


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
             S_tot=cnm.S, #Cn=Cn,
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
                    #noise_range=cnmf_params['noise_range'],
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
    cnm.options['preprocess_params']['noise_range'] = cnmf_params['noise_range']
    cnm.options['temporal_params']['noise_range'] = cnmf_params['noise_range']
    cnm.options['temporal_params']['s_min'] = cnmf_params['s_min']

    if verbose:
        print cnm.options

    
    # #In[114]:
    
    print "*** FITTING REFINED ***"
    
    cnm = cnm.fit(images)
    
    return cnm

#%%
def get_cnmf_seeds(images, cnmf_params, output_dir, n_processes=None, dview=None, create_new=False):

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
    
    #print(('Number of components:' + str(A_tot.shape[-1])))

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


#single_thread=False
#
## Start cluster:
#n_processes = 12
#
#c, dview, n_processes = cm.cluster.setup_cluster(
#    backend='local', n_processes=n_processes, single_thread=False)
#    
#
#print n_processes
#print dview


#%%


def extract_options(options):

    parser = optparse.OptionParser()

    parser.add_option('-D', '--root', action='store', dest='rootdir',
                          default='/nas/volume1/2photon/data',
                          help='data root dir (dir w/ all animalids) [default: /nas/volume1/2photon/data, /n/coxfs01/2pdata if --slurm]')
    parser.add_option('--slurm', action='store_true', dest='slurm', default=False, help="set if running as SLURM job on Odyssey")

    parser.add_option('-i', '--animalid', action='store', dest='animalid',
                          default='', help='Animal ID')

    # Set specific session/run for current animal:
    parser.add_option('-S', '--session', action='store', dest='session',
                          default='', help='session dir (format: YYYYMMDD')
    parser.add_option('-A', '--acq', action='store', dest='acquisition',
                          default='FOV1_zoom1x', help="acquisition folder [default: FOV1_zoom1x]")
    parser.add_option('-T', '--trace-type', action='store', dest='trace_type',
                          default='raw', help="trace type [default: 'raw']")
    parser.add_option('-R', '--run', dest='run', default='', action='store', help="run name")
    parser.add_option('--new', action='store_true', dest='create_new', default=False, help="set if run a new source extraction instance")
    parser.add_option('-d', '--datestr', dest='datestr', default=None, action='store', help="datestr YYYYMMDD_HH_mm_SS")
    parser.add_option('-t', '--cnmf-id', dest='cnmf_id', default=None, action='store', help="cnmf ID: e.g., cnmf001")

    parser.add_option('-x', '--exclude', dest='excluded_files', default=[], nargs=1,
                          action='append',
                          help="Index of files to exclude (0-indexed)")
    
    parser.add_option('--downsample', action='store', dest='downsample_factor', default=1.0, help='[nmf]: Downsample factor (default=1, use 0.5 or 0.2 if huge files)')
    
    parser.add_option('--gSig', action='store', dest='gSig', default=3, help='[nmf]: Half size of neurons [default: 3]')
    parser.add_option('--K', action='store', dest='K', default=20, help='[nmf]: N expected components per patch [default: 20]')
    parser.add_option('--patch', action='store', dest='rf', default=25, help='[nmf]: Half size of patch [default: 25]')
    parser.add_option('--stride', action='store', dest='stride_cnmf', default=6, help='[nmf]: Amount of patch overlap (keep it at least large as 4 times the neuron size) [default: 6]')
    parser.add_option('--bg', action='store', dest='gnb', default=1, help='[nmf]: Number of background components [default: 1]')
    parser.add_option('--p', action='store', dest='p', default=2, help='[nmf]: Order of autoregressive system [default: 2]')
    parser.add_option('--border', action='store', dest='border_to_0', default=0, help='[nmf]: N pixels to exclude for border (from motion correcting)[default: 0]')
    parser.add_option('--noise', action='store', dest='noise_range', default='0.25,0.5', help='[nmf]: Noise range for PSD [default: 0.25,0.5]')
    parser.add_option('--smin', action='store', dest='s_min', default=None, help='[nmf]: Min spike level [default: None]')
    parser.add_option('--gnb', action='store', dest='gnb', default=1, help='[nmf]: N background components [default: 1]')
    
    parser.add_option('--rval', action='store', dest='rval_thr', default=0.7, help='[nmf]: space correlation threshold (if above this, accept) [default: 0.7]')
    parser.add_option('--snr', action='store', dest='min_SNR', default=1.5, help='[nmf]: peak SNR for accepted components (if above this, acept) [default: 1.5]')
    parser.add_option('--decay', action='store', dest='decay_time', default=3.0, help='[nmf]: length of transient [default: 3.0]')
    
    parser.add_option('--nproc', action='store', dest='n_processes', default=4, help='[nmf]: N processes (default: 4)')
    parser.add_option('--seed', action='store_true', dest='manual_seed', default=False, help='Set true if seed ROIs with manual')
    parser.add_option('--offset', action='store_true', dest='add_offset', default=False, help='Set true if add min value to all movies to make (mostly) nonneg')
    parser.add_option('--suffix', action='store', dest='tif_suffix', default='', help='suffix to add to mcorrected_ dir, if relevant (e.g., unsigned)')
  
    parser.add_option('-o', '--rois', action='store', dest='roi_source', default='rois001', help='Manual ROI set to use as seeds (must set --seed, default: rois001)')
    parser.add_option('-q', action='store', dest='quantile_min', default=10, help='Quantile min for drift correction (default: 10)')
    parser.add_option('-w', action='store', dest='window_size', default=30, help='Window size for drift correction (default: 30 sec)')

    # PLOTTING PSTH opts:
    parser.add_option('--psth', action='store_true', dest='plot_psth', default=False, help='Set flag to plot PSTHs for all ROIs. Set plotting grid opts.')
    parser.add_option('-p', action='store', dest='psth_dtype', default='corrected', help='Data type to plot for PSTHs.')

    parser.add_option('-r', '--rows', action='store', dest='psth_rows', default=None, help='PSTH: transform to plot on ROWS of grid')
    parser.add_option('-C', '--cols', action='store', dest='psth_cols', default=None, help='PSTH: transform to plot on COLS of grid')
    parser.add_option('-H', '--hues', action='store', dest='psth_hues', default=None, help='PSTH: transform to plot for HUES of each subplot')


    (options, args) = parser.parse_args(options)
    if options.slurm:
        options.rootdir = '/n/coxfs01/2p-data'
    if options.datestr == 'None':
        options.datestr = None
    if options.cnmf_id == 'None':
        options.cnmf_id = None
    
    return options

def get_mmap(fnames, fbase=None, excluded_files=[], dview=None, border_to_0=0, downsample_factor=(1,1,1), add_offset=False):
    
    
    #downsample_factor = (1,1,1) # Use fractions if want to downsample
    #border_to_0 = 2
    if fbase is None:
        fbase = os.path.split(fnames[0].split('/processed/')[0])[-1].split('_')[0] # Just use stim-type <stim>_runx
        
    #fbase = run.split('_')[0]
    
    fname_new, mmap_basedir = get_mmap_file(fnames, excluded_files=excluded_files, file_base=fbase, 
                                  dview=dview, 
                                  downsample_factor=downsample_factor, 
                                  border_to_0=border_to_0, add_offset=add_offset)

    return fname_new, mmap_basedir

def get_Cn(images, traceid_dir):
    cn_path = os.path.join(traceid_dir, 'Cn.npz')
    if not os.path.exists(cn_path):
        print "Cn not found, calculating new..."
        # Look at correlation image:
        Cn = cm.movie(images).local_correlations(swap_dim=False) #cm.local_correlations(Y)
        #Cn[np.isnan(Cn)] = 0
        
        # #In[60]:
        np.savez(os.path.join(traceid_dir, 'Cn.npz'), Cn=Cn)
    else:
        tmpd = np.load(cn_path)
        Cn = tmpd['Cn']
        print "Loaded Cn, shape:", Cn.shape
    
    fig, ax = pl.subplots(1)
    ax.imshow(Cn, cmap='gray') #, vmax=.35)
    fig.savefig(os.path.join(traceid_dir, 'figures', 'Cn.png'))
    pl.close()
    
    return Cn

def get_cnmf_params(fname_new, excluded_files=[], final_frate=44.69, 
                        K=20, gSig=[3,3],
                        rf=25, stride_cnmf=6,
                        init_method='greedy_roi', alpha_snmf=None, p=2,
                        merge_thresh=0.8, gnb=1,
                        noise_range=[0.25, 0.5],
                        s_min=None,
                        manual_seed=False, roi_source=None):
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
#    gnb = 1
    
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
                   'excluded_files': excluded_files,
                   'manual_seeds': manual_seed,
                   'roi_source': roi_source,
                   'noise_range': noise_range,
                   's_min': s_min}
    
    return cnmf_params

def get_cnmf_outdirs(acquisition_dir, run, cnmf_id=None, datestr=None):
    traceid_basedir = os.path.join(acquisition_dir, run, 'traces')
    if not os.path.exists(traceid_basedir):
        os.makedirs(traceid_basedir)
        
    new_cnmf=False; cnmf_num=None;
    if cnmf_id is not None:
        existing_cnmfs = glob.glob(os.path.join(traceid_basedir, '%s*' % cnmf_id))
        try:
            assert len(existing_cnmfs) > 0, "No cnmf dirs with ID -- %s -- found." % cnmf_id
            if datestr is None:
                current_cnmf = sorted(existing_cnmfs, key=natural_keys)[::-1][-1]
            else:
                current_cnmf = [c for c in existing_cnmfs if datestr in c][0]
            cnmf_num = int(os.path.split(current_cnmf)[-1].split('_')[0][4:])
            datestr = '_'.join(os.path.split(current_cnmf)[-1].split('_')[1:])
        except Exception as e:
            print "Creating new CNMF id."
            new_cnmf = True
    else:
        new_cnmf = True
            
    if new_cnmf:
        existing_cnmfs = glob.glob(os.path.join(traceid_basedir, 'cnmf*'))
        if len(existing_cnmfs) > 0:
            last_cnmf_id = os.path.split(sorted(existing_cnmfs, key=natural_keys)[-1])[-1]
            last_cnmf_num = int(last_cnmf_id.split('_')[0][4:])
            cnmf_num = last_cnmf_num + 1
        else:
            cnmf_num = 1
                
    if datestr is None:
        datestr = datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")    
    
    
    traceid_dir = os.path.join(traceid_basedir, 'cnmf%03d_%s' % (cnmf_num, datestr))
    
    if not os.path.exists(os.path.join(traceid_dir, 'figures')):
        os.makedirs(os.path.join(traceid_dir, 'figures'))
    if not os.path.exists(os.path.join(traceid_dir, 'results')):
        os.makedirs(os.path.join(traceid_dir, 'results'))
    print traceid_dir
    
    return traceid_dir
        

def filter_bad_seeds(cnm, images, fr, dims, gSig, traceid_dir, dview=None, 
                     decay_time=4.0, min_SNR=1.2,
                     rval_thr=0.7, use_cnn=False, min_cnn_thr=None, Cn=None):
    
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
    with open(os.path.join(traceid_dir, 'figures', 'evalparams_patch_%s.json' % eval_datestr), 'w') as f:
        json.dump(eval_params, f, indent=4, sort_keys=True)
    
    # Save evaluation results (plus all info we want to keep):
    A_tot, C_tot, YrA_tot, b_tot, f_tot, sn_tot = cnm.A, cnm.C, cnm.YrA, cnm.b, cnm.f, cnm.sn
    
    np.savez(os.path.join(traceid_dir, 'evaluation_patch_%s.npz' % eval_datestr), 
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
    pl.savefig(os.path.join(traceid_dir, 'figures', 'evaluation_patch_%s.png' % eval_datestr))
    pl.close()
    
    
    print "Keeping %i components." % len(idx_components)

    # Run patches above to get initial components:
    #results_basename = 'results_cnmf'
    A_in, C_in, b_in, f_in = cnm.A[:, idx_components], cnm.C[idx_components], cnm.b, cnm.f

    return A_in, C_in, b_in, f_in


def get_seeds(fname_new, fr, optsE, traceid_dir, decay_time=4.0,
                  min_SNR=1.2, rval_thr=0.7, n_processes=1, 
                  dview=None, images=None):
#        decay_time = 4.0 #.0    # length of transient
#        min_SNR = 1.2      # peak SNR for accepted components (if above this, acept)
#        rval_thr = 0.7     # space correlation threshold (if above this, accept)
    
    original_source = True
    acquisition_dir = os.path.join(optsE.rootdir, optsE.animalid, optsE.session, optsE.acquisition)
    run = optsE.run

    K = optsE.K
    p = int(optsE.p)
    noise_range_str = optsE.noise_range
    noise_range = [float(s) for s in noise_range_str.split(',')]
    print "Noise Range: ", noise_range
    s_min = optsE.s_min
    if s_min is not None:
        s_min = float(s_min)
    gnb = int(optsE.gnb)
     
    gSig = (int(optsE.gSig), int(optsE.gSig))
    rf = optsE.rf
    stride_cnmf = optsE.stride_cnmf
    roi_source = None
    if optsE.manual_seed:
        roi_source = optsE.roi_source
        
    cnmf_params = get_cnmf_params(fname_new, final_frate=fr, K=K, gSig=gSig,
                              rf=rf, stride_cnmf=stride_cnmf, p=p, noise_range=noise_range, s_min=s_min, gnb=gnb,
                              manual_seed=optsE.manual_seed, roi_source=roi_source)
    
    
    with open(os.path.join(traceid_dir, 'cnmf_params.json'), 'w') as f:
        json.dump(cnmf_params, f, indent=4, sort_keys=True)
        
    if optsE.manual_seed:
        #results_basename = 'results_seed'
        if 'rois' in roi_source:
            if original_source:
                rid_dirs = glob.glob(os.path.join(optsE.rootdir, optsE.animalid, optsE.session, 'ROIs', '%s*' % roi_source))
                assert len(rid_dirs) > 0, "Specified ROI src not found: %s" % optsE.roi_source
                rid_dir = sorted(rid_dirs)[-1]
                mask_fpath = os.path.join(rid_dir, 'masks.hdf5')
                print "Seeding NMF from manual ROIs: %s" % roi_source 
                print "Source ROIs from: %s" % rid_dir
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
                # Use motion-corrected tiff:
                pid_fpath = glob.glob(os.path.join(acquisition_dir, run, 'processed', 'pids_%s.json' % run))[0]
                with open(pid_fpath, 'r') as f: pids = json.load(f)
                ref_file = pids['processed001']['PARAMS']['motion']['ref_file'] # Just always default to processed001...
                A_in = maskfile[ref_file]['Slice01']['maskarray'][:] # Already in shape npixels x nrois -- but x-y dims are flipped@**
                A_in[A_in>0] = 1
                A_in = A_in.astype(bool)
            
        elif 'cnmf_' in roi_source:
            assert os.path.exists(roi_source), "Specified ROI src not found: %s" % optsE.roi_source
            cnmf_fpath = roi_source # sorted(rid_dirs)[-1]
            cnmf = np.load(cnmf_fpath)
            cnm = cnmf['cnm'][()]

            print "Loaded %s" % cnmf_fpath
            A_in = np.array(cnm.A.todense())
            A_in[A_in>0] = 1
            A_in = A_in.astype(bool)
   
        print A_in.shape
        print "Reshaped seeded spatial comps:", A_in.shape
        
        C_in = None; b_in = None; f_in = None; 
        
    else:
        #%
        # #############################################################################
        # cNMF: Extract from patches:
        # #############################################################################
        print "Estimating components with patches."
        Yr, dims, T = cm.load_memmap(fname_new)
        d1, d2 = dims
        images = np.reshape(Yr.T, [T] + list(dims), order='F') 
        
        Cn = get_Cn(images, traceid_dir)

        cnm = get_cnmf_seeds(images, cnmf_params, traceid_dir, 
                             n_processes=n_processes, dview=dview, 
                             create_new=optsE.create_new)

        # #### Evaluate components:
        #fr = 44.69             # approximate frame rate of data
#        decay_time = 4.0 #.0    # length of transient
#        min_SNR = 1.2      # peak SNR for accepted components (if above this, acept)
#        rval_thr = 0.7     # space correlation threshold (if above this, accept)
        #use_cnn = False # use the CNN classifier
        #min_cnn_thr = None  # if cnn classifier predicts below this value, reject
        dims = [d1, d2]
        gSig = (optsE.gSig, optsE.gSig) #(3, 3)
      
        A_in, C_in, b_in, f_in = filter_bad_seeds(cnm, images, fr, dims, gSig, traceid_dir,
                                                  dview=dview,
                                                  decay_time=decay_time,
                                                  min_SNR=min_SNR, rval_thr=rval_thr, Cn=Cn)
        
        crd = plot_contours(A_in.astype(int), Cn, thr=0.9)
        pl.savefig(os.path.join(traceid_dir, 'figures', 'seed_components.png'))
        pl.close()

    return A_in, C_in, b_in, f_in, cnmf_params


def evaluate_cnmf(cnm2, images, fr, dims, gSig, traceid_dir,
                  dview=None,
                  decay_time=4.0, min_SNR=2.0, rval_thr=0.6,
                  use_cnn=False, min_cnn_thr=None, Cn=None):
    
    #fr = 44.69             # approximate frame rate of data
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
    
    refined_datestr = datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")
    
    
    with open(os.path.join(traceid_dir, 'figures', 'evalparams_refined_%s.json' % refined_datestr), 'w') as f:
        json.dump(eval_params, f, indent=4, sort_keys=True)
    
    plot_thr = 0.75
    pl.figure(figsize=(15,5))
    pl.subplot(1,2,1)
    crd = plot_contours(cnm2.A.tocsc()[:, idx_components], Cn, thr=plot_thr)
    pl.subplot(1,2,2)
    crd = plot_contours(cnm2.A.tocsc()[:, idx_components_bad], Cn, thr=plot_thr)
    
    pl.savefig(os.path.join(traceid_dir, 'figures', 'evaluation_refined_%s.png' % refined_datestr))
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
    results_fpath = os.path.join(traceid_dir, 'results', 'results_refined_%s.npz' % refined_datestr)
    cnm2.fpath = results_fpath
    
    np.savez(results_fpath, cnm=cnm2)
    cnm2.dview=dview
    
    print "Saved final evaluation results:\n%s" % results_fpath

    return cnm2

#%% ### Get TIF source

#rootdir = '/mnt/odyssey'

#rootdir = '/n/coxfs01/2p-data'
#rootdir = '/Volumes/coxfs01/2p-data'

#animalid = 'CE077'
#session = '20180629'
#acquisition = 'FOV1_zoom1x'
#run = 'gratings_rotating_drifting'
#create_new = False
#excluded_files = []
#datestr = None
#
#options = ['-D', '/n/coxfs01/2p-data', '-i', 'CE077', '-S', '20180702',
#           '-R', 'gratings_rotating_drifting', '--nproc=12', '--seed', '-r', 'rois001',
#           '--border=2']
#    
#options = ['-D', '/n/coxfs01/2p-data', '-i', 'CE077', '-S', '20180629',
#'-A', 'FOV1_zoom1x', '-R', 'gratings_drifting', '--seed',
#'-r', '/n/coxfs01/2p-data/CE077/20180629/FOV1_zoom1x/gratings_rotating_drifting/traces/cnmf/cnmf_20180803_17_04_21/results/results_refined_20180803_17_13_07.npz',
#'--border=2', '--nproc=4', '--suffix=offset', '-q', 20, '-w', 30, '-t', '20180807_10_38_14']
#
#

#%%
    
def run_cnmf(options):
    
    optsE = extract_options(options)
        
    # Start cluster:
    single_thread=False
    n_processes = int(optsE.n_processes)
    excluded_files = optsE.excluded_files
    add_offset = optsE.add_offset
    tif_suffix = optsE.tif_suffix
 
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=n_processes, single_thread=False)
        
    print n_processes
    print dview

    acquisition_dir = os.path.join(optsE.rootdir, optsE.animalid, optsE.session, optsE.acquisition)

#%  ### Create memmapped files:
    # -------------------------------------------------------------------------
#    fnames = sorted(glob.glob(os.path.join(acquisition_dir, optsE.run, 'processed', 'processed001*', 'mcorrected_*%s' % tif_suffix, '*.tif')), key=natural_keys)
    
    fnames = sorted(glob.glob(os.path.join(acquisition_dir, '%s*' % optsE.run, 'processed', 'processed001*', 'mcorrected_*%s' % tif_suffix, '*.tif')), key=natural_keys)
    fnames = sorted([f for f in fnames if '_deinterleaved' not in f], key=natural_keys)
    
    border_to_0 = int(optsE.border_to_0)
    downsample_factor = (1, 1, float(optsE.downsample_factor)) # 0.5 #(1,1,1)
    fname_new, mmap_basedir = get_mmap(fnames, fbase=optsE.run, excluded_files=excluded_files, 
                                       dview=dview, border_to_0=border_to_0, 
                                       downsample_factor=downsample_factor, add_offset=add_offset)
    
    run_name = os.path.split(mmap_basedir.split('/processed/')[0])[-1]
    
    # Get SI info:
    # -------------------------------------------------------------------------
    si_info_path = glob.glob(os.path.join(acquisition_dir, '%s*' % optsE.run, '*.json'))[0]
    with open(si_info_path, 'r') as f:
        si_info = json.load(f)
    
    fr = si_info['frame_rate'] * downsample_factor[-1] # 44.69
    
    #% Set output dir for figures and results:
    # -------------------------------------------------------------------------
    traceid_dir = get_cnmf_outdirs(acquisition_dir, run_name, cnmf_id=optsE.cnmf_id, datestr=optsE.datestr)
    

    #%% ### Load memmapped file (all runs):
    
    #% LOAD MEMMAP FILE
    Yr, dims, T = cm.load_memmap(fname_new)
    d1, d2 = dims
    images = np.reshape(Yr.T, [T] + list(dims), order='F')
    Y = np.reshape(Yr, dims + (T,), order='F')
    print "dims:", dims
    print "Getting correlation..." 
    #% Get correlation image:    
    Cn = get_Cn(images, traceid_dir)
        
    #%%
    # #############################################################################
    # Get seeds for cNMF (either patches or manual ROIs)
    # #############################################################################
    decay_time = float(optsE.decay_time)
    min_SNR = float(optsE.min_SNR)
    rval_thr = float(optsE.rval_thr)
    A_in, C_in, b_in, f_in, cnmf_params = get_seeds(fname_new, fr, optsE, 
                                                    traceid_dir, n_processes=n_processes, dview=dview,
                                                    images=images,
                                                    decay_time=decay_time,
                                                    min_SNR=min_SNR, 
                                                    rval_thr=rval_thr)

    
    #%% 
    
    # #############################################################################
    # REFINE seeded components:
    # #############################################################################

    cnm2 = seed_cnmf(images, cnmf_params, A_in, C_in, b_in, f_in, n_processes=n_processes, dview=dview)
        
    refined_datestr = datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")
        
    #% #### Visualize refined run:
    pl.figure()
    crd = plot_contours(cnm2.A, Cn, thr=0.6)
    pl.savefig(os.path.join(traceid_dir, 'figures', 'spatial_comps_refined_%s.png' % refined_datestr))
    pl.close()
    
    print "*** Plotted all final components ***"


    # #### Extract DFF: 
    
    quantileMin = float(optsE.quantile_min)
    frames_window = int(round(float(optsE.window_size)*fr))
    print "quant MIN: %i, window size %i" % (quantileMin, frames_window) 

    F_dff = detrend_df_f(cnm2.A, cnm2.b, cnm2.C, cnm2.f, YrA=cnm2.YrA,
                         quantileMin=quantileMin, frames_window=frames_window)
    print "Extracted DFF"
 
#    pl.figure(figsize=(20,5))
#    pl.plot(F_dff[0,:])
#    pl.savefig(os.path.join(traceid_dir, 'figures', 'example_roi0_dff.png'));
#    pl.close()
#    
    cnm2.F_dff = F_dff
    cnm2.quantileMin = quantileMin
    cnm2.frames_window = frames_window
    cnm2.Cn = Cn
    
    #% Save refined cnm with dff and S:
    cnm2.dview = None
    with open(os.path.join(traceid_dir, 'results', 'results_refined_%s.pkl' % refined_datestr), 'wb') as f:
        pkl.dump(cnm2, f, protocol=pkl.HIGHEST_PROTOCOL)
    cnm2.dview = dview
    print "Saved pkl..."
    
    # #### Save refined run:
    cnm2.dview=None
    np.savez(os.path.join(traceid_dir, 'results', 'results_refined_%s.npz' % refined_datestr), 
             cnm=cnm2)
    cnm2.dview=dview
    print "... and npz"
    
    #%%
    # #### Evaluate refined run:
    print "*** Evaluating final run ***"
    decay_time = float(optsE.decay_time)    # length of transient
    min_SNR = 2.0      # peak SNR for accepted components (if above this, acept)
    rval_thr = 0.8     # space correlation threshold (if above this, accept)
#    use_cnn = False # use the CNN classifier
#    min_cnn_thr = None  # if cnn classifier predicts below this value, reject
#    dims = [d1, d2]
    gSig = (int(optsE.gSig), int(optsE.gSig))
    
    cnm2 = evaluate_cnmf(cnm2, images, fr, dims, gSig, traceid_dir, dview=dview,
                         decay_time=decay_time, min_SNR=min_SNR, rval_thr=rval_thr, Cn=Cn)
    
    #%%
    
    # ### Play around with contour visualization...
    results_fpath = glob.glob(os.path.join(traceid_dir, 'results', 'results_refined_*.npz'))[0]
    
    remove_bad_components = False
    
    if remove_bad_components:
        A = cnm2.A[:, cnm2.idx_components]
        C =cnm2.C[cnm2.idx_components]
        mov_fn_append = '_idxcomp'
    else:
        A = cnm2.A
        C = cnm2.C
        mov_fn_append = '_all'
    
    b = cnm2.b
    f = cnm2.f
        
    Cn = cnm2.Cn
    
    #%% reconstruct denoised movie

    ntiffs = si_info['ntiffs'] - len(excluded_files)
    print "Creating movie for 1 out of %i tiffs." % ntiffs
    nvolumes = T/ntiffs
    
#    nruns = len(list(set([fname.split('/processed/')[0] for fname in fnames])))
#    nvolumes = T / nruns
    C = C[:, 0:int(nvolumes)]
    f = f[:, 0:int(nvolumes)]
    
    denoised = cm.movie(A.dot(C) + b.dot(f)).reshape(tuple(dims) + (-1,), order='F').transpose([2, 0, 1])
    
    denoised.save(os.path.join(traceid_dir, 'figures', 'denoised_refined_File001%s.tif' % mov_fn_append))
    
    
    #% #### Visualize "good" components:
    
    # In[ ]:
    
    
    #crd = plot_contours(A[:, idx_components], Cn, thr=0.9)
    
    #pl.figure(figsize=(15,5))
    #pl.subplot(1,2,1)
    #crd = plot_contours(cnm2.A[:, idx_components], Cn, thr=0.9)
    #pl.subplot(1,2,2)
    #crd = plot_contours(cnm2.A[:, idx_components_bad], Cn, thr=0.9)
    #
    #pl.savefig(os.path.join(traceid_dir, 'figures', 'evaluation_final.png'))
    #pl.close()
    
    ## In[ ]:
    #
    #A = A_tot
    #C = C_tot
    #YrA = YrA_tot
    #f = f_tot
    #b = b_tot
#    cm.utils.visualization.view_patches_bar(Yr, cnm2.A[:, cnm2.idx_components],
#                                            cnm2.C[cnm2.idx_components, :], cnm2.b, cnm2.f,
#                                            d1, d2,
#                                            YrA=cnm2.YrA[cnm2.idx_components, :], img=Cn)
#    
#        
    #
    ## In[ ]:
    #
    ##%% Show final traces
    #cnm2.view_patches(Yr, dims=dims, img=Cn)

    #%% STOP CLUSTER and clean up log files
    cm.stop_server(dview=dview)
    log_files = glob.glob('*_LOG_*')
    for log_file in log_files:
        print log_file
        os.remove(log_file)
    ##%% restart cluster to clean up memory
    #dview.terminate()
    #c, dview, n_processes = cm.cluster.setup_cluster(
    #    backend='local', n_processes=None, single_thread=False)
    

    return cnm2.fpath


#%%

# #############################################################################
# #### Align traces to trials
# #############################################################################

#results_path = os.path.join(traceid_dir, 'results_analysis1b.npz')

#results_fpath = os.path.join(traceid_dir, 'results', 'eval_final.npz')

#results_fpath = glob.glob(os.path.join(traceid_dir, 'results', 'results_refined_*.npz'))[0]

def format_cnmf_results(results_fpath, excluded_files=[], remove_bad_components=False):
    print "Loading results: %s" % results_fpath
    
    
    traceid_dir = results_fpath.split('/results')[0]
    run_dir = traceid_dir.split('/traces')[0]
    run = os.path.split(run_dir)[-1]
    acquisition_dir = os.path.split(run_dir)[0]
    
    
    cnmd = np.load(results_fpath)
    cnm = cnmd['cnm'][()]
    
    
    
    nrois = cnm.A.shape[-1]
    F_raw = cnm.C + cnm.YrA
    A = cnm.A
    C = cnm.C
    b = cnm.b
    f = cnm.f
    
#    rec = A.dot(C) 
#    fig, axes = pl.subplots(3, 1) 
#    axes[0].plot(F_raw[1, 0:3000]); axes[0].set_title('raw');
#    axes[1].plot(rec[1, 0:3000]); axes[1].set_title('denoised')
#    rec += b.dot(f)
#    axes[2].plot(rec[1, 0:3000]); axes[2].set_title('+ noise')
    
    
    YrA = cnm.YrA
    S = cnm.S
    F_dff = cnm.F_dff
    
    if remove_bad_components:
        idx_components = cnm.idx_components
        nrois = len(idx_components)
        F_raw = C[idx_components] + YrA[idx_components]
        A = A[:, idx_components]
        C = C[idx_components]
        S = S[idx_components]
        F_dff = F_dff[idx_components]
        YrA = YrA[idx_components]
    
    raw_df = pd.DataFrame(data=F_raw.T, columns=[['roi%05d' % int(i+1) for i in range(nrois)]])
    
    # Get baseline:
    B = A.T.dot(b).dot(f)
    F0_df = pd.DataFrame(data=B.T, columns=raw_df.columns)
    
    # Get DFF:
    dFF_df = pd.DataFrame(data=F_dff.T, columns=[['roi%05d' % int(i+1) for i in range(nrois)]])
    
    # Get Drift-Corrected "raw":
    quantileMin = cnm.quantileMin # c['quantile']
    frames_window = cnm.frames_window # int(round(15.*fr)) #c['frames_window']
    
#    Fd = scipy.ndimage.percentile_filter(
#                F_raw, quantileMin, (frames_window, 1))
#    corrected = F_raw - Fd    
    corrected_df = pd.DataFrame(data=C.T, columns=raw_df.columns)

    # Get deconvolved traces (spikes):
    S_df = pd.DataFrame(data=S.T, columns=raw_df.columns)
    
    if 'combined' in run: 
        run_base = run.split('_static')[0].split('combined_')[-1]
    else:
        run_base = run
        
    si_info_path = glob.glob(os.path.join(acquisition_dir, '%s*' % run_base, '*.json'))[0]
    with open(si_info_path, 'r') as f:
        si_info = json.load(f)
        
    ntiffs = si_info['ntiffs'] - len(excluded_files)
    
    #% Turn all this info into "standard" data frame arrays:
    labels_df, raw_df, corrected_df, F0_df, dFF_df, spikes_df = caiman_to_darrays(run_dir, raw_df, 
                                                                              corrected_df=corrected_df, 
                                                                              dFF_df=dFF_df, 
                                                                              F0_df=F0_df, 
                                                                              S_df=S_df, 
                                                                              output_dir=traceid_dir, 
                                                                              ntiffs=ntiffs,
                                                                              excluded_files=excluded_files)
    
    # Get stimulus / trial info:
    run_info = util.run_info_from_dfs(run_dir, raw_df, labels_df, traceid_dir=traceid_dir, trace_type='caiman', ntiffs=ntiffs)
    
    stimconfigs_fpath = os.path.join(run_dir, 'paradigm', 'stimulus_configs.json')
    with open(stimconfigs_fpath, 'r') as f:
        stimconfigs = json.load(f)
        
            
    # Get label info:
    sconfigs = util.format_stimconfigs(stimconfigs)
    ylabels = labels_df['config'].values
    groups = labels_df['trial'].values
    tsecs = labels_df['tsec']
        
    
    # Set data array output dir:
    data_basedir = os.path.join(traceid_dir, 'data_arrays')
    if not os.path.exists(data_basedir):
        os.makedirs(data_basedir)
    data_fpath = os.path.join(data_basedir, 'datasets.npz')
    
    #%
    
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
    #             meanstim=meanstim_values, 
    #             zscore=zscore_values,
    #             meanstimdff=meanstimdff_values,
             labels_data=labels_df,
             labels_columns=labels_df.columns.tolist(),
             run_info=run_info)


    return data_fpath




#%%

def arrays_to_trials(trials_in_block, frame_tsecs, parsed_frames, mwinfo, framerate=44.69, ntiffs=None):
        
    ntiffs_total = len(list(set([parsed_frames[t]['frames_in_file'].attrs['aux_file_idx'] for t in parsed_frames.keys()])))
    
    if ntiffs is None:
        ntiffs = ntiffs_total
        
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
        dur = round(float(off - on) / framerate, 1)
        sdurs.append(dur)
    print "unique durs:", list(set(sdurs))

    # Check if frame indices are indexed relative to full run (all .tif files)
    # or relative to within-tif frames (i.e., a "block")
    block_indexed = True
    if all([all(parsed_frames[t]['frames_in_run'][:] == parsed_frames[t]['frames_in_file'][:]) for t in trials_in_block]):
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
        stim_onset_idxs -= frame_index_adjust ####

    # Check that we have all frames needed (no cut off frames from end):
    assert frame_indices[-1] <= len(frame_tsecs) * ntiffs_total
    
    return frame_indices, stim_onset_idxs


def caiman_to_darrays(run_dir, raw_df, downsample_factor=(1, 1, 1),
                      corrected_df=None, dFF_df=None, 
                      F0_df=None, S_df=None, output_dir='tmp', ntiffs=None, excluded_files=[],
                      fmt='hdf5', trace_arrays_type='caiman'):
    
    xdata_df=None; labels_df=None; #F0_df=None; dFF_df=None; S_df=None;
    
    roixtrial_fn = 'roiXtrials_%s.%s' % (trace_arrays_type, fmt) # Procssed or SMoothed

    # Get SCAN IMAGE info for run:
    # -------------------------------------------------------------------------
    run = os.path.split(run_dir)[-1]
    fov_dir = os.path.split(run_dir)[0]
    if 'combined' in run:
        run_base = run.split('_static')[0].split('combined_')[-1]
    else:
        run_base = run
    
    runinfo_fpath = glob.glob(os.path.join(fov_dir, '%s*' % run_base, '*.json'))[0]
    with open(runinfo_fpath, 'r') as fr:
        scan_info = json.load(fr)
    framerate = scan_info['frame_rate'] * downsample_factor[-1]
    frame_tsecs = np.array(scan_info['frame_tstamps_sec'])
    ds = int(1/downsample_factor[-1])
    frame_tsecs = frame_tsecs[0::ds]
    ntiffs_in_run = scan_info['ntiffs']
    if ntiffs is None:
        ntiffs = int(ntiffs_in_run)
            
    # Need to make frame_tsecs span all TIFs:
    frame_tsecs_ext = np.hstack([frame_tsecs for i in range(ntiffs)])
    print "N frame tstamps to align, TOTAL:", len(frame_tsecs_ext)

    # Load MW info to get stimulus details:
    # -------------------------------------------------------------------------
    paradigm_dir = os.path.join(run_dir, 'paradigm')
    #mw_fpaths = glob.glob(os.path.join(fov_dir, '%s*' % run_base, 'paradigm', 'trials_*.json'))
    
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
    frame_indices, stim_onset_idxs = arrays_to_trials(trials_in_block, frame_tsecs, parsed_frames, mwinfo, framerate=framerate, ntiffs=ntiffs)
    
    
    # Get Stimulus info for each trial:        
    #excluded_params = [k for k in mwinfo[t]['stimuli'].keys() if k in stimconfigs[stimconfigs.keys()[0]].keys()]
    excluded_params = ['filehash', 'stimulus', 'type', 'rotation_range'] #, 'phase'] # Only include param keys saved in stimconfigs
    if 'phase' not in stimconfigs['config001'].keys():
        excluded_params.append('phase')
        
    #print "---- ---- EXCLUDED PARAMS:", excluded_params

    curr_trial_stimconfigs = [dict((k,v) for k,v in mwinfo[t]['stimuli'].iteritems() \
                                   if k not in excluded_params) for t in trials_in_block]
    #print curr_trial_stimconfigs
    
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
    
    assert len(config_labels)==len(trial_labels), "Mismatch in n frames per trial, %s" % parsed_frames_fpath

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
        stim_durs = list(set([round(mwinfo[t]['stim_dur_ms']/1e3, 1) for t in trial_list]))
    nframes_on = np.array([int(round(dur*framerate)) for dur in stim_durs])
   
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


#options = ['-D', '/n/coxfs01/2p-data', '-i', 'JC026', '-S', '20181209', '-A', 'FOV1_zoom2p0x',
#           '-R', 'gratings', '--nproc=2', '--seed', '-r', 'rois001',
#           '--border=4']

options = ['-D', '/n/coxfs01/2p-data', '-i', 'JC026', '-S', '20181209', '-A', 'FOV1_zoom2p0x',
           '-R', 'gratings_run1', '--nproc=16', '--gSig=5', 
           '--border=4']

#%%
def main(options):
    
    # First check that we don't need to re-extract:
    optsE = extract_options(options)
    create_new = optsE.create_new
    check_results = [] 
    if optsE.datestr is not None:
        check_results = glob.glob(os.path.join(optsE.rootdir, optsE.animalid, optsE.session,
                                           optsE.acquisition, optsE.run, 'traces', 'cnmf*', 
                                           'results', 'results_refined_*.npz'))
        if len(check_results) > 0:
            print "Found results:\n%s" % str(check_results)
            create_new = False
        else:
            print "No results found. Creating new."
            create_new = True
        if len(check_results) == 1 and optsE.create_new is False:
            results_fpath = check_results[0]
        elif len(check_results) > 1 and optsE.create_new is False:
            results_fpath = sorted(check_results)[-1]
            print "*** Loading: %s" % results_fpath
            
    if create_new or len(check_results)==0: #lse:
        # Extract rois and traces:
        results_fpath = run_cnmf(options)
        
    data_fpath = format_cnmf_results(results_fpath, excluded_files=[], remove_bad_components=False)
    
    # Plot df/f PSTH figures for each ROI:
    cnmf_traceid = os.path.split(results_fpath.split('/results')[0])[-1]

    run_opts = ['-D', optsE.rootdir, '-i', optsE.animalid, '-S', optsE.session,
		'-A', optsE.acquisition, '-R', optsE.run, '-t', cnmf_traceid]
    if optsE.slurm:
        run_opts.extend(['--slurm'])
        
    if optsE.plot_psth:
        psth_opts = copy.copy(run_opts)
        psth_opts.extend(['-d', optsE.psth_dtype])
        if optsE.psth_dtype == 'corrected': # and optsE.psth_calc_dff != 'None':
            psth_opts.extend(['--calc-dff'])
            
        if optsE.psth_rows is not None and optsE.psth_rows != 'None':
            psth_opts.extend(['-r', optsE.psth_rows])
        if optsE.psth_cols is not None and optsE.psth_cols != 'None':
            psth_opts.extend(['-c', optsE.psth_cols])
        if optsE.psth_hues is not None and optsE.psth_hues!='None':
            print "Specified HUE:", optsE.psth_hues
            psth_opts.extend(['-H', optsE.psth_hues])
        psth_opts.extend(['--shade'])

#    plot_opts = ['-D', optsE.rootdir, '-i', optsE.animalid, '-S', optsE.session,
#                 '-A', optsE.acquisition, '-R', optsE.run, '-t', cnmf_traceid,
#                 '-d','dff']
#    if 'rotating' in optsE.run:
#        plot_opts.extend(['-r', 'stim_dur', '-c', 'ori', '-H', 'direction'])
#    psth_dir = pplot.make_clean_psths(plot_opts)
#
#    # Plot PSTHs with inferred spikes to compare:
#    plot_opts = ['-D', optsE.rootdir, '-i', optsE.animalid, '-S', optsE.session,
#                 '-A', optsE.acquisition, '-R', optsE.run, '-t', cnmf_traceid,
#                 '-d', 'spikes']
#    if 'rotating' in optsE.run:
#        plot_opts.extend(['-r', 'stim_dur', '-c', 'ori', '-H', 'direction'])
        
    psth_dir = pplot.make_clean_psths(psth_opts)

    print "*******************************************************************"
    print "DONE!"
    print "All output saved to: %s" % psth_dir
    print "*******************************************************************"

if __name__ == '__main__':
    main(sys.argv[1:])


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
