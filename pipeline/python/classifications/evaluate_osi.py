#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 18:22:01 2019

@author: julianarhee
"""

import os
import glob

import numpy as np
import seaborn as sns
import pylab as pl
import pandas as pd

from scipy import stats as spstats

from pipeline.python.utils import natural_keys, label_figure
from pipeline.python.classifications import bootstrap_fit_tuning_curves as bs 
from pipeline.python.classifications import utils as util

#%%

def get_average_fit_params(fiters):
    fparams = pd.DataFrame({'r_pref':  [fiter['popt'][0] for fiter in fiters],
                            'r_null': [fiter['popt'][1] for fiter in fiters],
                            'theta_pref': [fiter['popt'][2] for fiter in fiters],
                            'sigma': [fiter['popt'][3] for fiter in fiters],
                            'r_offset': [fiter['popt'][4] for fiter in fiters]
                            })
    
    r_pref = fparams.mean()['r_pref']
    r_null = fparams.mean()['r_null']
    theta_pref = fparams.mean()['theta_pref']
    sigma = fparams.mean()['sigma']
    r_offset = fparams.mean()['r_offset']
    
    popt = (r_pref, r_null, theta_pref, sigma, r_offset)
    
    return popt

#%%

rootdir = '/n/coxfs01/2p-data'
animalid = 'JC084' 
session = '20190522' #'20190319'
fov = 'FOV1_zoom2p0x' 
run = 'combined_gratings_static'
traceid = 'traces001' #'traces002'

response_type = 'dff'
make_plots = True
n_bootstrap_iters = 100
n_intervals_interp = 3

responsive_test = 'ROC'
responsive_thr = 0.05

create_new = False


#%%
run_name = bs.get_gratings_run(animalid, session, fov, traceid=traceid, rootdir=rootdir) 
roi_list = util.get_responsive_cells(animalid, session, fov, run=run_name, traceid=traceid,
                                     responsive_test=responsive_test, responsive_thr=responsive_thr,
                                     rootdir=rootdir)
                    

#% Select fit set:
roi_fitdir = glob.glob(os.path.join(rootdir, animalid, session, fov, run_name,
                                    'traces', '%s*' % traceid, 'tuning', 
                                    'bootstrap*%s-thr%.2f' % (responsive_test, responsive_thr)))[0]

#%

#% Load fit results, examine each cell's iterations:  
fitdf, fitparams, fitdata = bs.get_tuning(animalid, session, fov, run_name, roi_list=roi_list, create_new=create_new,
                             n_bootstrap_iters=n_bootstrap_iters, n_intervals_interp=n_intervals_interp,
                             response_type=response_type, make_plots=make_plots, roi_fitdir=roi_fitdir, return_iters=True)
fitdf['cell'] = [int(i) for i in fitdf['cell'].values]

#%%

#r_pref, r_null, theta_pref, sigma, r_offset = i['popt']

roi = 30

fiters = fitdata['results_by_iter'][roi]



#%%  Look at residuals

fig, axes = pl.subplots(2,3)
mean_residuals = []
for fiter in fitdata['results_by_iter'][roi]:
    xs = fiter['x']
    fity = fiter['y']
    origy = fiter['fit_y']
    residuals = origy - fity
    
    axes[0,0].plot(xs, fity, lw=0.5)
    axes[0,1].scatter(origy, residuals, alpha=0.5)
    
    mean_residuals.append(np.mean(residuals))

axes[0,0].set_xticks(xs[0::n_intervals_interp][0:-1])
axes[0,0].set_xticklabels(xs[0::n_intervals_interp][0:-1])
axes[0,0].set_xlabel('thetas')
axes[0,0].set_ylabel('fit')

axes[0,1].axhline(y=0, linestyle=':', color='k')
axes[0,1].set_ylabel('residuals')
axes[0,1].set_xlabel('fitted value')

axes[0,2].hist(mean_residuals, bins=20)


#%%

# Compare the original direction tuning curve with the fitted curve derived 
# using the average of each fitting parameter (across 100 iterations):


popt = get_average_fit_params(fiters)
thetas = fiters[0]['x'][0::n_intervals_interp][0:-1]

responses = fitdata['original_data'][roi]['responses']
origr = responses.mean().values
fitr = bs.double_gaussian( thetas, *popt)

# Get residual sum of squares 
residuals = origr - fitr
ss_res = np.sum(residuals**2)
ss_tot = np.sum((origr - np.mean(origr))**2)
r2_comb = 1 - (ss_res / ss_tot)

ax = axes[1,0]
ax.plot(thetas, origr, 'k', label='orig')
ax.plot(thetas, fitr, 'r:', label='fit')
ax.set_title('r2-comb: %.2f' % r2_comb)

#%%

def get_gfit(r2_values):
    iqr = spstats.iqr(r2_values)
    gfit = np.mean(r2_values) * (1-iqr) * np.sqrt(r2_comb)
    return gfit

gfits = fitdf.groupby(['cell'])['r2'].apply(get_gfit)

pl.figure()
pl.hist(gfits, alpha=0.5)

goodness_thr = 0.66
good_fits = [int(r) for r in gfits.index.tolist() if gfits[r] > goodness_thr]
nrois_fit = len(gfits)
nrois_pass = len(good_fits)
print("%i out of %i fit cells pass goodness-thr %.2f" % (nrois_pass, nrois_fit, goodness_thr))













