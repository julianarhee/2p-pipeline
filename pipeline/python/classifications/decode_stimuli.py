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
import traceback
import math

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
from pipeline.python.retinotopy import segment_retinotopy as seg
from pipeline.python.classifications import decode_utils as dc

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

def initializer(terminating_):
    # This places terminating in the global namespace of the worker subprocesses.
    # This allows the worker function to access `terminating` even though it is
    # not passed as an argument to the function.
    global terminating
    terminating = terminating_


def pool_bootstrap(neuraldf, sdf, n_iterations=50, n_processes=1, C_value=None,
                   test=None, single=False, n_train_configs=4, verbose=False):   
    '''
    test (string, None)
        None  : Classify A/B only 
                single=True to train/test on each size
        morph : Train on anchors, test on intermediate morphs
                single=True to train/test on each size
        size  : Train on specific size(s), test on un-trained sizes
                single=True to train/test on each size
    '''
    C=C_value
    vb = verbose 
    results = []
    terminating = mp.Event()

    pool = mp.Pool(initializer=initializer, initargs=(terminating, ), processes=n_processes)  
    try:
        ntrials, sample_ncells = neuraldf.shape
        print("... n: %i (%i procs)" % (int(sample_ncells-1), n_processes))
#        if test=='morph':
#            if single: # train on 1 size, test on other sizes
#                func = partial(dutils.do_fit_train_single_test_morph, global_rois=global_rois, 
#                               MEANS=MEANS, sdf=sdf, sample_ncells=sample_ncells)
#            else: # combine data across sizes
#                func = partial(dutils.do_fit_train_test_morph, global_rois=global_rois, 
#                               MEANS=MEANS, sdf=sdf, sample_ncells=sample_ncells)             
#        elif test=='size':
#            if single:
#                func = partial(dutils.do_fit_train_test_single, global_rois=global_rois, 
#                               MEANS=MEANS, sdf=sdf, sample_ncells=sample_ncells)
#            else:
#                func = partial(dutils.cycle_train_sets, global_rois=global_rois, 
#                               MEANS=MEANS, sdf=sdf, sample_ncells=sample_ncells, 
#                                n_train_configs=n_train_configs)
#        else:
#        func = partial(dutils.do_fit_within_fov, curr_data=neuraldf, sdf=sdf, verbose=verbose,
#                                                    C_value=C_value)
#        results = pool.map_async(func, range(n_iterations)).get(99999999)
        results = [pool.apply_async(dutils.do_fit_within_fov, args=(i, neuraldf, sdf, vb, C)) \
                    for i in range(n_iterations)]
        output= [p.get(99999999) for p in results]
    except KeyboardInterrupt:
        terminating.set()
        print("**interupt")
        pool.terminate()
        print("***Terminating!")
    finally:
        pool.close()
        pool.join()

    return output #results


def fit_svm_mp(neuraldf, sdf, n_iterations=50, n_processes=1, 
                   C_value=None, cv_nfolds=5, test_split=0.2, 
                   test=None, single=False, n_train_configs=4, verbose=False,
                   class_a=0, class_b=106):   
    results = []
    terminating = mp.Event()
    
    #### Select train/test configs for clf A vs B
    train_configs = sdf[sdf['morphlevel'].isin([class_a, class_b])].index.tolist() 

    def worker(n_iters, neuraldf, sdf, C_value, verbose, out_q):
        r_ = []        
        for ni in n_iters:
            print('... %i' % ni)
            curr_iter = dutils.do_fit_within_fov(ni, curr_data=neuraldf, sdf=sdf, verbose=verbose,
                                                    C_value=C_value)
            #### Fit
            r_.append(curr_iter)
        curr_iterdf = pd.concat(r_, axis=0)
        out_q.put(curr_iterdf)
        
    try:        
        # Each process gets "chunksize' filenames and a queue to put his out-dict into:
        iter_list = np.arange(0, n_iterations) #gdf.groups.keys()
        out_q = mp.Queue()
        chunksize = int(math.ceil(len(iter_list) / float(n_processes)))
        procs = []
        for i in range(n_processes):
            p = mp.Process(target=worker,
                           args=(iter_list[chunksize * i:chunksize * (i + 1)],
                                          neuraldf, sdf, C_value, verbose, out_q))
            procs.append(p)
            p.start()

        # Collect all results into single results dict. We should know how many dicts to expect:
        results = []
        for i in range(n_processes):
            results.append(out_q.get())

        # Wait for all worker processes to finish
        for p in procs:
            print "Finished:", p
            p.join()
    except KeyboardInterrupt:
        terminating.set()
        print("***Terminating!")
    except Exception as e:
        traceback.print_exc()
    finally:
        for p in procs:
            p.join    

    return results
#%%

def decode_from_fov(datakey, visual_area, cells, MEANS, min_ncells=5,
                    C_value=None,
                    n_iterations=50, n_processes=2, tmp_results_str='iter_results',
                    rootdir='/n/coxfs01/2p-data', create_new=False, verbose=False):
    # tmp save
    session, animalid, fov_ = datakey.split('_')
    fovnum = int(fov_[3:])
    fov = 'FOV%i_zoom2p0x' % fovnum
    traceid_dir = glob.glob(os.path.join(rootdir, animalid, session, fov, 'combined_blobs*', 
                            'traces', '%s*' % traceid))[0]
    curr_dst_dir = os.path.join(traceid_dir, 'decoding')
    if not os.path.exists(curr_dst_dir):
        os.makedirs(curr_dst_dir)
        print("... saving tmp results to:\n  %s" % curr_dst_dir)

    results_outfile = os.path.join(curr_dst_dir, '%s.pkl' % tmp_results_str)
    if create_new is False: 
        try:
            with open(results_outfile, 'rb') as f:
                iter_results = pkl.load(f)
        except Exception as e:
            create_new=True 

    if create_new:    
        #### Get neural means
        print("... Stating decoding analysis")
        neuraldf = aggr.get_neuraldf_for_cells_in_area(cells, MEANS, 
                                                       datakey=datakey, visual_area=visual_area)
        if int(neuraldf.shape[1]-1)<min_ncells:
            return None

        n_cells = int(neuraldf.shape[1]-1) 
        print("... [%s] %s, n=%i cells" % (visual_area, datakey, n_cells))
        # ------ STIMULUS INFO -----------------------------------------
        session, animalid, fov_ = datakey.split('_')
        fovnum = int(fov_[3:])
        obj = util.Objects(animalid, session, 'FOV%i_zoom2p0x' %  fovnum, traceid=traceid)
        sdf = obj.get_stimuli()

        # Decodinng -----------------------------------------------------
        #print(neuraldf.head())
        iter_list = fit_svm_mp(neuraldf, sdf, C_value=C_value, 
                                    n_iterations=n_iterations, 
                                    n_processes=n_processes, verbose=verbose)

        print("%i items in mp list" % len(iter_list))

        # DATA - get mean across items
        iter_results = pd.concat(iter_list, axis=0)
    
        with open(results_outfile, 'wb') as f:
            pkl.dump(iter_results, f, protocol=pkl.HIGHEST_PROTOCOL)

    # Pool mean
    #print(iter_results)
    print("... finished all iters: %s" % str(iter_results.shape))
    iterd = dict(iter_results.mean())
    iterd.update( dict(('%s_std' % k, v) \
            for k, v in zip(iter_results.std().index, iter_results.std().values)) )
    iterd.update( dict(('%s_sem' % k, v) \
            for k, v in zip(iter_results.sem().index, iter_results.sem().values)) )
    iterd.update({'n_units': n_cells, 
                  'visual_area': visual_area, 'datakey': datakey})
    print("::FINAL::")
    pp.pprint(iterd)

    return iterd


def do_decoding(dsets, MEANS, min_ncells=5, n_iterations=50, n_processes=2,
                results_str='iter_results',
                dst_dir='/n/coxfs01/julianarhee/aggregate-visual-areas/decoding'):

    rois = seg.get_cells_by_area(dsets)
    cells = aggr.get_active_cells_in_current_datasets(rois, MEANS)

    popdf = []
    for (visual_area, datakey), g in dsets.groupby(['visual_area', 'datakey']): 
        print("[%s]: %s" % (visual_area, datakey))
        iterd = decode_from_fov(datakey, visual_area, cells, MEANS, min_ncells=min_ncells,
                                n_iterations=n_iterations, n_processes=n_processes, 
                                tmp_results_str=results_str)
        popdf.append(pd.DataFrame(iterd, index=[i]))
        i += 1
    pooled = pd.concat(popdf, axis=0)

    # Save data
    print("SAVING.....")
    datestr = datetime.datetime.now().strftime("%Y%m%d")

    # Save classifier results
    results_outfile = os.path.join(dst_dir, 
                        '%s_results_%s.pkl' % (train_str, datestr))
    results = {'results': pooled, 'sdf': sdf}
    with open(results_outfile, 'wb') as f:
        pkl.dump(pooled, f, protocol=pkl.HIGHEST_PROTOCOL) 
    print("-- results: %s" % results_outfile)

    # Save params
    #params_outfile = os.path.join(dst_dir, 
    #                    '%s_params_%s.json' % (train_str, datestr))

    #params = {'test_split': test_split, 'cv_nfolds': cv_nfolds, 'C_value': C_value,
    #          'n_iterations': n_iterations, 'overlap_thr': overlap_thr,
    #          'class_a': m0, 'class_b': m100, 'train_str': train_str, 'data_id': data_id}
    #with open(params_outfile, 'w') as f:
    #    json.dump(params, f,  indent=4, sort_keys=True)
    #print("-- params: %s" % params_outfile)
       
    # Plot
    #plot_str = '%s' % (train_str)
    #dutils.default_classifier_by_ncells(pooled, plot_str=plot_str, dst_dir=dst_dir, 
    #                    data_id=data_id, area_colors=area_colors, datestr=datestr)
    print("DONE!")

    return pooled



#%%

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
#print(m0, m100, '%i iters' % n_iterations)

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

    parser.add_option('-i', '--animalid', action='store', dest='animalid', 
                        default='', help="animalid")
    parser.add_option('-A', '--fov', action='store', dest='fov', 
                        default='FOV1_zoom2p0x', help="fov (default: FOV1_zoom2p0x)")
    parser.add_option('-S', '--session', action='store', dest='session', 
                        default='', help="session (YYYYMMDD)")
 
       
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
    parser.add_option('-N', action='store', dest='n_iterations', 
            default=100, help="N iterations (default: 100)")

    parser.add_option('-o', action='store', dest='overlap_thr', 
            default=0.8, help="% overlap between RF and stimulus (default: 0.8)")
    parser.add_option('-V', action='store_true', dest='verbose', 
            default=False, help="verbose printage")
    parser.add_option('--new', action='store_true', dest='create_new', 
            default=False, help="re-do decode")

    parser.add_option('--cv', action='store_true', dest='do_cv', 
            default=False, help="tune for C")
    parser.add_option('-C','--cvalue', action='store', dest='C_value', 
            default=1.0, help="tune for C (default: 1)")

    (options, args) = parser.parse_args(options)

    return options


def main(options):
    opts = extract_options(options)
    fov_type = 'zoom2p0x'
    state = 'awake'
    aggregate_dir = '/n/coxfs01/julianarhee/aggregate-visual-areas'
    rootdir = opts.rootdir

    create_new = opts.create_new

    animalid = opts.animalid
    session = opts.session
    fov = opts.fov

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

    stim_filterby = None # 'first'
    has_gratings = experiment!='blobs'
    trial_epoch = 'stimulus'
    verbose=opts.verbose
    print(verbose)


    # -------------------------------------------------
    test_split=0.2
    cv_nfolds=5
    C_value=opts.C_value
    do_cv = opts.do_cv
    C_value = None if do_cv else float(opts.C_value)

    filter_fovs = True
    remove_too_few = False
    min_ncells = 20 if remove_too_few else 0
    # -------------------------------------------------                              
    
    # Set colors
    visual_area, area_colors = putils.set_threecolor_palette()
    dpi = putils.set_plot_params()

    #### Responsive params
    n_stds = None if responsive_test=='ROC' else 2.5 #None
    response_str = '%s_%s-thr-%.2f' % (response_type, responsive_test, responsive_thr) 

    #### Output dir
    stats_dir = os.path.join(aggregate_dir, 'data-stats')
    #dst_dir = os.path.join(aggregate_dir, 'decoding')
    MEANS = aggr.load_aggregate_data(experiment, 
                responsive_test=responsive_test, responsive_thr=responsive_thr, 
                response_type=response_type, epoch=trial_epoch)

    # Get all data sets
    sdata = aggr.get_aggregate_info(traceid=traceid, fov_type=fov_type, state=state)

    rois = seg.get_cells_by_area(sdata)
    cells = aggr.get_active_cells_in_current_datasets(rois, MEANS, verbose=False)

    fovnum = int(fov.split('_')[0][3:])
    datakey = '%s_%s_fov%i' % (session, animalid, fovnum)
    try:
        assert datakey in cells['datakey'].unique(), "Dataset %s not segmented. Aborting." % datakey

        visual_areas = cells[cells['datakey']==datakey]['visual_area'].unique()
        print("[%s] %i visul areas in current fov." % (datakey, len(visual_areas)))

        for visual_area in visual_areas:    
            decode_from_fov(datakey, visual_area, cells, MEANS, 
                            min_ncells=5, C_value=C_value,
                            n_iterations=n_iterations, n_processes=n_processes, 
                            tmp_results_str='fov_results_%s' % visual_area,
                            rootdir=rootdir, create_new=create_new, verbose=verbose)
            print("... finished %s (%s)" % (datakey, visual_area))
    except Exception as e:
        traceback.print_exc()
 
    #pooled = do_decoding(sdata, MEANS, min_ncells=5, n_iterations=n_iterations, 
    #            n_processes=n_processes,
    #            dst_dir='/n/coxfs01/julianarhee/aggregate-visual-areas/decoding')
    #print("pooled: %s" % (pooled.shape))

    return 


if __name__ == '__main__':
    main(sys.argv[1:])
