#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 19:44:31 2020

@author: julianarhee
"""

#%%
import os
import glob
import json
import copy
import optparse
import sys
import traceback
import cv2
from scipy import interpolate
import math

import matplotlib as mpl
mpl.use('agg')
import pylab as pl
import seaborn as sns
import cPickle as pkl
import numpy as np

from functools import partial
import multiprocessing as mp

def sphr_correct_maps_mp(avg_resp_by_cond, fit_params, n_processes=2, test_subset=False):
    
    if test_subset:
        roi_list=[92, 249, 91, 162, 61, 202, 32, 339]
        df_ = avg_resp_by_cond[roi_list]
    else:
        df_ = avg_resp_by_cond.copy()
    print("Parallel", df_.shape)

    df = parallelize_dataframe(df_.T, sphr_correct_maps, fit_params, n_processes=n_processes)

    return df

def initializer(terminating_):
    # This places terminating in the global namespace of the worker subprocesses.
    # This allows the worker function to access `terminating` even though it is
    # not passed as an argument to the function.
    global terminating
    terminating = terminating_

def parallelize_dataframe(df, func, fit_params, n_processes=4):
    #cart_x=None, cart_y=None, sphr_th=None, sphr_ph=None,
    #                      row_vals=None, col_vals=None, resolution=None, n_processes=4):
    results = []
    terminating = mp.Event()
    
    df_split = np.array_split(df, n_processes)
    pool = mp.Pool(processes=n_processes, initializer=initializer, initargs=(terminating,))
    try:
        results = pool.map(partial(func, fit_params=fit_params), df_split)
        print("done!")
    except KeyboardInterrupt:
        pool.terminate()
        print("terminating")
    finally:
        pool.close()
        pool.join()
  
    print(results[0].shape)
    df = pd.concat(results, axis=1)
    print(df.shape)
    return df #results

#
