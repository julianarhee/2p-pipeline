#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on  Apr 24 17:16:28 2020

@author: julianarhee
"""
import matplotlib as mpl
mpl.use('agg')

import sys
import optparse
import os
import json
import glob
import copy
import copy
import itertools
import datetime
import pprint 
pp = pprint.PrettyPrinter(indent=4)

import numpy as np
import pylab as pl
import seaborn as sns
import pandas as pd
import statsmodels as sm
import cPickle as pkl

from scipy import stats as spstats

from pipeline.python.classifications import experiment_classes as util
from pipeline.python.classifications import aggregate_data_stats as aggr
from pipeline.python.classifications import rf_utils as rfutils
from pipeline.python import utils as putils

from pipeline.python.retinotopy import fit_2d_rfs as fitrf
from pipeline.python.rois.utils import load_roi_coords


from matplotlib.lines import Line2D
import matplotlib.patches as patches

import scipy.stats as spstats
import sklearn.metrics as skmetrics
import random

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC
from sklearn import preprocessing
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn import svm


from pipeline.python.classifications import decode_utils as dutils

import multiprocessing as mp
from functools import partial
from contextlib import contextmanager


import multiprocessing as mp
from functools import partial
from contextlib import contextmanager

@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()
    pool.join()

def initializer(terminating_):
    # This places terminating in the global namespace of the worker subprocesses.
    # This allows the worker function to access `terminating` even though it is
    # not passed as an argument to the function.
    global terminating
    terminating = terminating_


def pool_bootstrap(global_rois, MEANS, sdf, sample_ncells, n_iterations=50, n_processes=1,
                   test=None, single=False, n_train_configs=4):   
    '''
    test (string, None)
        None  : Classify A/B only 
                single=True to train/test on each size
        morph : Train on anchors, test on intermediate morphs
                single=True to train/test on each size
        size  : Train on specific size(s), test on un-trained sizes
                single=True to train/test on each size
    '''
    
    results = []
    terminating = mp.Event()

    pool = mp.Pool(initializer=initializer, initargs=(terminating, ), processes=n_processes)  
    try:
        print("... n: %i (%i procs)" % (sample_ncells, n_processes))
        if test=='morph':
            if single: # train on 1 size, test on other sizes
                func = partial(dutils.do_fit_train_single_test_morph, global_rois=global_rois, 
                               MEANS=MEANS, sdf=sdf, sample_ncells=sample_ncells)
            else: # combine data across sizes
                func = partial(dutils.do_fit_train_test_morph, global_rois=global_rois, 
                               MEANS=MEANS, sdf=sdf, sample_ncells=sample_ncells)             
        elif test=='size':
            if single:
                func = partial(dutils.do_fit_train_test_single, global_rois=global_rois, 
                               MEANS=MEANS, sdf=sdf, sample_ncells=sample_ncells)
            else:
                func = partial(dutils.cycle_train_sets, global_rois=global_rois, 
                               MEANS=MEANS, sdf=sdf, sample_ncells=sample_ncells, 
                                n_train_configs=n_train_configs)
        else:
            func = partial(dutils.do_fit, global_rois=global_rois, MEANS=MEANS, 
                           sdf=sdf, sample_ncells=sample_ncells)
        results = pool.map_async(func, range(n_iterations)).get(99999999)
        pool.close()
        pool.join()
    except KeyboardInterrupt:
        print("**interupt")
        pool.terminate()
        print("***Terminating!")
    finally:
        pool.close()
        pool.join()

    return results

#%%

def decode_vs_ncells(rfs_and_blobs, stim_datakeys, MEANS, sdf, train_str='clf-by-ncells',
                    n_iterations=100, overlap_thr=0.8, n_processes=1, 
                    test_split=0.2, cv_nfolds=5, C_value=None, cv=True, 
                    class_a=0, class_b=106, data_id='DATAID', 
                    dst_dir='/n/coxfs01/julianarhee/aggregate-data/decoding'):

    # Filter by RF overlap
    overlap_int = 0.2
    overlap_thr_values = np.arange(0, 1+overlap_int, overlap_int)

    #### Linear separability, by RF overlap
    #### Run for 1 overlap_thr, 1 iter, select M0 / M100
    globalcells_df, cell_counts = dutils.filter_rois(
                                    rfs_and_blobs[rfs_and_blobs['datakey'].isin(curr_dkeys)], 
                                    overlap_thr=overlap_thr, return_counts=True)

    # Make sure have SAME N trials total
    keys_with_min_reps = [k for k, v in MEANS.items() if v['config'].value_counts().min() < 29]
    filt_globaldf = globalcells_df[~globalcells_df['datakey'].isin(keys_with_min_reps)]
    print(filt_globaldf['visual_area'].value_counts())

    # SET N cells, plot.
    if overlap_thr==0:
        NCELLS = [2, 4, 8, 16, 32, 64, 82, 123, 186, 237, 448, 556, 652]
    elif overlap_thr==0.8:
        NCELLS = [2, 4, 8, 16, 32, 64, 82, 112, 164, 201, 448, 556, 652]
    print("NCELLS: %s" % (str(NCELLS)))
    ncells_dict = dict((k, NCELLS) for k in overlap_thr_values)

    popdf = []
    #for overlap_thr, NCELLS in ncells_dict.items():
    print("-------- Overlap: %.2f --------" % overlap_thr)
    i=0
    for visual_area, global_rois in filt_globaldf.groupby(['visual_area']):
        for sample_ncells in NCELLS: #[0::2]:
            print("... [%s] popn size: %i" % (visual_area, sample_ncells))
            if sample_ncells > cell_counts[visual_area]:
                continue 
            iter_list = pool_bootstrap(global_rois, MEANS, sdf, sample_ncells, test=None,
                                       n_iterations=n_iterations, n_processes=n_processes)
            # DATA - get mean across iters
            iter_results = pd.concat(iter_list, axis=0)
            iterd = dict(iter_results.mean())
            iterd.update( dict(('%s_std' % k, v) \
                    for k, v in zip(iter_results.std().index, iter_results.std().values)) )
            iterd.update( dict(('%s_sem' % k, v) \
                    for k, v in zip(iter_results.sem().index, iter_results.sem().values)) )
            iterd.update({'n_units': sample_ncells, 
                          'overlap': overlap_thr, 'visual_area': visual_area})
            popdf.append(pd.DataFrame(iterd, index=[i]))
            i += 1
    pooled = pd.concat(popdf, axis=0)
    pooled.head()

    # Save data
    print("SAVING.....")
    datestr = datetime.datetime.now().strftime("%Y%m%d")
    results_outfile = os.path.join(dst_dir, 
                        '%s_overlap-%.2f_results_%s.pkl' % (train_str, overlap_thr, datestr))
    params_outfile = os.path.join(dst_dir, 
                        '%s_overlap-%.2f_params_%s.json' % (train_str, overlap_thr, datestr))

    with open(results_outfile, 'wb') as f:
        pkl.dump(pooled, f, protocol=pkl.HIGHEST_PROTOCOL) 
    print("-- results: %s" % results_outfile)

    params = {'test_split': test_split, 'cv_nfolds': cv_nfolds, 'C_value': C_value, 'cv':cv,
              'n_iterations': n_iterations, 'overlap_thr': overlap_thr,
              'class_a': m0, 'class_b': m100, filter_fovs=True, }
    with open(params_outfile, 'w') as f:
        json.dump(params, f,  indent=4, sort_keys=True)
    print("-- params: %s" % params_outfile)
       
    # Plot
    plot_str = '%s_overlap-%.2f' % (train_str, overlap_thr)
    dutils.default_classifier_by_ncells(pooled, plot_str=plot_str, dst_dir=dst_dir, 
                        data_id=data_id, area_colors=area_colors, datestr=datestr)
    print("DONE!")

    return


traceid = 'traces001'
fov_type = 'zoom2p0x'
state = 'awake'
aggregate_dir = '/n/coxfs01/julianarhee/aggregate-visual-areas'

response_type = 'dff'
responsive_test = 'nstds' # 'nstds' #'ROC' #None
responsive_thr = 10

# CV stuff
experiment = 'blobs'
m0=0
m100=106
n_iterations=100 
n_processes = 2
print(m0, m100, '%i iters' % n_iterations)

test_split=0.2
cv_nfolds=5
C_value=None
cv=True

min_ncells = 20
overlap_thr=0.
filter_fovs = True
remove_too_few = False


def extract_options(options):
    
    parser = optparse.OptionParser()

    parser.add_option('-D', '--root', action='store', dest='rootdir', 
                      default='/n/coxfs01/2p-data',\
                      help='data root dir [default: /n/coxfs01/2pdata]')

    # Set specific session/run for current animal:
    parser.add_option('-E', '--experiment', action='store', dest='experiment', 
                        default='blobs', help="experiment type [default: blobs]")
    parser.add_option('-t', '--traceid', action='store', dest='traceid', default='traces001', \
                      help="name of traces ID [default: traces001]")
    parser.add_option('-d', '--response-type', action='store', dest='response_type', 
                        default='dff', help="response type [default: dff]")
       
    # data filtering 
    choices_c = ('all', 'roc', 'nstds')
    default_c = 'nstds'
    parser.add_option('-R', '--responsive_test', action='store', dest='responsive_test', 
            default=default_c, type='choice', choices=choices_c,
            help="Responsive test, choices: %s. (default: %s" % (choices_c, default_c))
    parser.add_option('-r', '--responsive-thr', action='store', dest='responsive_thr', 
                        default=10, help="response type [default: 10, nstds]")
 
    # plotting
    parser.add_option('-a', action='store', dest='class_a', 
            default=0, help="m0 (default: 0 morph)")
    parser.add_option('-b', action='store', dest='class_b', 
            default=106, help="m100 (default: 106 morph)")
    parser.add_option('-n', action='store', dest='n_processes', 
            default=1, help="N processes (default: 1)")
    parser.add_option('-i', action='store', dest='n_iterations', 
            default=100, help="N iterations (default: 100)")

    parser.add_option('-o', action='store', dest='overlap_thr', 
            default=0.8, help="% overlap between RF and stimulus (default: 0.8)")

    (options, args) = parser.parse_args(options)

    return options



def main(options):

    opts = extract_options(options)
    fov_type = 'zoom2p0x'
    state = 'awake'
    aggregate_dir = '/n/coxfs01/julianarhee/aggregate-visual-areas'

    traceid = opts.traceid #'traces001'
    response_type = opts.response_type #'dff'
    responsive_test = opts.responsive_test #'nstds' # 'nstds' #'ROC' #None
    responsive_thr = float(opts.responsive_thr) #10

    # CV stuff
    experiment = opts.experiment #'blobs'
    m0=int(opts.class_a) #0
    m100=int(opts.class_b) #106
    n_iterations=int(opts.n_iterations) #100 
    n_processes=int(opts.n_processes) #2
    print(m0, m100, '%i iters' % n_iterations)
    overlap_thr = float(opts.overlap_thr)

    stim_filterby = 'first'
    has_gratings = experiment!='blobs'

    # -------------------------------------------------
    cv =True 
    test_split=0.2
    cv_nfolds=5
    C_value=None

    filter_fovs = True
    remove_too_few = False
    min_ncells = 20 if remove_too_few else 0
    # -------------------------------------------------                              
    
    train_str = 'traintest_by-ncells_iter-%i' % (n_iterations)
 
    # Set colors
    visual_area, area_colors = putils.set_threecolor_palette()
    dpi = putils.set_plot_params()

    #### Responsive params
    n_stds = None if responsive_test=='ROC' else 2.5 #None
    response_str = '%s_%s-thr-%.2f' % (response_type, responsive_test, responsive_thr) 

    #### Output dir
    stats_dir = os.path.join(aggregate_dir, 'data-stats')
    decoding_dir = os.path.join(aggregate_dir, 'decoding')

    # Drop duplicates and whatnot fovs
    if experiment=='blobs':
        g_str = 'hasgratings' if has_gratings else 'blobsonly'
        exp_dkeys = aggr.get_blob_datasets(filter_by=stim_filterby, 
                                            has_gratings=has_gratings, as_dict=True)
    else:
        g_str = 'gratingsonly'
        exp_dkeys = aggr.get_gratings_datasets(filter_by=stim_filterby, as_dict=True)

    # Create data ID for labeling figures with data-types
    filter_str = 'filter_%s_%s' % (stim_filterby, g_str)
    data_id = '|'.join([traceid, filter_str, respones_str])

    #### Get metadata for experiment type
    sdata = aggr.get_aggregate_info(traceid=traceid, fov_type=fov_type, state=state)

    # Get blob metadata only - and only if have RFs
    sdata_exp = pd.concat([g for k, g in sdata.groupby(['animalid', 'session', 'fov']) if 
                            (experiment in g['experiment'].values 
                             and ('rfs' in g['experiment'].values \
                                     or 'rfs10' in g['experiment'].values)) ])

    dictkeys = [d for d in list(itertools.chain(*exp_dkeys.values()))]
    stim_datakeys = ['%s_%s_fov%i' % (s.split('_')[0], s.split('_')[1], 
                       sdata[(sdata['animalid']==s.split('_')[1]) 
                           & (sdata['session']==s.split('_')[0])]['fovnum'].unique()[0]) \
                                   for s in dictkeys]
    expmeta = dict((k, [dv for dv in stim_datakeys for vv in v \
                    if vv in dv]) for k, v in exp_dkeys.items())

    #### Load neural responses
    aggr_trialmeans_dfile = glob.glob(os.path.join(stats_dir, 
                                'aggr_%s_trialmeans_*%s-thr-%.2f*_%s_stimulus.pkl' 
                                % (experiment, responsive_test, responsive_thr, response_type)))[0]
    print(aggr_trialmeans_dfile)
    with open(aggr_trialmeans_dfile, 'rb') as f:
        MEANS = pkl.load(f)
        

    #### Check that all datasets have same stim configs
    SDF={}
    for datakey in stim_datakeys:
        session, animalid, fov_ = datakey.split('_')
        fovnum = int(fov_[3:])
        obj = util.Objects(animalid, session, 'FOV%i_zoom2p0x' %  fovnum, traceid=traceid)
        sdf = obj.get_stimuli()
        SDF[datakey] = sdf
    nonpos_params = [p for p in sdf.columns if p not in ['xpos', 'ypos', 'position']] 
    assert all([all(sdf[nonpos_params]==d[nonpos_params]) for k, d in SDF.items()]), "Incorrect stimuli..."

    #### Get screen and stimulus info
    screeninfo = putils.get_screen_dims() #aggr.get_aggregate_stimulation_info(curr_sdata) #, experiment='blobs')
    screenright = float(screeninfo['azimuth_deg']/2)
    screenleft = -1*screenright #float(screeninfo['screen_right'].unique())
    screentop = float(screeninfo['altitude_deg']/2)
    screenbottom = -1*screentop
    screenaspect = float(screeninfo['resolution'][0]) / float(screeninfo['resolution'][1])

    #### Load RF fits -------------------------------------
    rf_filter_by=None
    reliable_only = True
    rf_fit_thr = 0.05
    # -----------------------------------------------------
    fit_desc = fitrf.get_fit_desc(response_type=response_type)
    reliable_str = 'reliable' if reliable_only else ''
    rf_str = 'match%s_%s' % (experiment, reliable_str)

    # Get position info for RFs 
    rf_dsets = sdata_exp[(sdata_exp['datakey'].isin(stim_datakeys))
                         & (sdata_exp['experiment'].isin(['rfs', 'rfs10']))].copy()
    aggr_rf_dir = os.path.join(aggregate_dir, 'receptive-fields', '%s__%s' % (traceid, fit_desc))
    df_fpath =  os.path.join(aggr_rf_dir, 
                                'fits_and_coords_%s_%s.pkl' % (rf_filter_by, reliable_str))
    rfdf = dutils.get_rf_positions(rf_dsets, df_fpath)

    # Select RFs, whichever (rfs/rfs10) in common with blob rids
    RFs = dutils.pick_rfs_with_most_overlap(rfdf, MEANS)

    print("All RFs-----------------------------------")
    pp.pprint(rfdf[['visual_area', 'datakey']].drop_duplicates().groupby(['visual_area']).count())
    print("RFs with blobs -----------------------------------")
    pp.pprint(RFs[['visual_area', 'datakey']].drop_duplicates().groupby(['visual_area']).count())

    # Plot
    fig = dutils.plot_all_rfs(RFs, MEANS, screeninfo, cmap='cubehelix')
    pl.suptitle("RF positions (+ CoM), responsive cells (%s)" % experiment)
    putils.label_figure(fig, data_id)
    figname = 'CoM_label-fovs_common_to_blobs_and_rfs'
    pl.savefig(os.path.join(aggr_rf_dir, '%s.svg' % figname))
    print(aggr_rf_dir, figname)


    #### Calculate overlap with stimulus
    stim_overlaps = dutils.calculate_overlaps(RFs, MEANS.keys(), experiment=experiment)

    # Get data common to RFs + blobs
    c_list=[]
    d_list = []
    i=0
    for (visual_area, datakey, rfname), g in stim_overlaps.groupby(['visual_area', 'datakey', 'rfname']):
        if datakey not in MEANS.keys():
            print("no %s: %s" % (experiment, datakey))
            continue
            
        exp_rids = [r for r in MEANS[datakey].columns if putils.isnumber(r)]
        rf_rids = sorted(g['cell'].unique())
        common_rids = np.intersect1d(exp_rids, rf_rids)
        print("[%s] %s, (%s) %i common cells" % (visual_area, datakey, rfname, len(common_rids)))
        c_list.append(pd.DataFrame({'visual_area': visual_area, 'datakey': datakey, 
                                    'rfname': rfname, 'n_cells': len(common_rids)}, index=[i])) 
        d_list.append(g[g['cell'].isin(common_rids)].copy())
        i+=1    
    rfs_and_blobs = pd.concat(d_list, axis=0)   
    common_counts = pd.concat(c_list, axis=0)
    
    min_ncells=20
    curr_datakeys = stim_datakeys if filter_fovs else rfs_and_blobs['datakey'].unique()
    if remove_too_few:
        too_few = [datakey for (visual_area, datakey), g in 
                rfs_and_blobs[rfs_and_blobs['perc_overlap']>=overlap_thr].groupby(['visual_area', 'datakey']) if len(g['cell'].unique()) < min_ncells]
        curr_datakeys = [s for s in curr_datakeys if s not in too_few]

    fig_str = '%s_%s' % (response_str, filter_str) if filter_fovs else '%s_all' % response_str
    fig_str = '%s_%s' % (fig_str, 'min-%i-cells' % min_ncells) if remove_too_few else fig_str
    print(fig_str)

    decode_vs_ncells(rfs_and_blobs, curr_datakeys, MEANS, sdf, train_str=train_str,
                    n_iterations=n_iterations, overlap_thr=overlap_thr, 
                    n_processes=n_processes, 
                    test_split=test_split, cv_nfolds=cv_nfolds, C_value=C_value, cv=cv, 
                    class_a=m0, class_b=m100, data_id='%s|%s' % (traceid, fig_str), #data_id,
                    dst_dir=decoding_dir)



if __name__ == '__main__':
    main(sys.argv[1:])
