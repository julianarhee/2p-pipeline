#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 11:10:13 2019

@author: julianarhee
"""

#!/usr/bin/env python2
# coding: utf-8

# In[1]:


import os
import glob
import json
import h5py
import copy
import cv2
import imutils

import numpy as np
import pandas as pd
import pylab as pl
import seaborn as sns
import cPickle as pkl
import matplotlib.gridspec as gridspec

from pipeline.python.classifications import experiment_classes as util
from pipeline.python.classifications import test_responsivity as resp
from pipeline.python.classifications import responsivity_stats as rstats
from pipeline.python.utils import label_figure, natural_keys, convert_range
from pipeline.python.retinotopy import convert_coords as coor
from pipeline.python.retinotopy import fit_2d_rfs as fitrf
from matplotlib.patches import Ellipse, Rectangle

from shapely.geometry.point import Point
from shapely import affinity
from matplotlib.patches import Polygon

from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import MaxNLocator

from shapely.geometry import box


import matplotlib_venn as mpvenn
import itertools
import time
import multiprocessing as mp

#%%

# ############################################
# Functions for processing visual field coverage
# ############################################

def create_ellipse(center, lengths, angle=0):
    """
    create a shapely ellipse. adapted from
    https://gis.stackexchange.com/a/243462
    """
    circ = Point(center).buffer(1)
    ell = affinity.scale(circ, int(lengths[0]), int(lengths[1]))
    ellr = affinity.rotate(ell, angle)
    return ellr


#%%


def get_session_object(animalid, session, fov, traceid='traces001', trace_type='corrected',
                       create_new=True, rootdir='/n/coxfs01/2p-data'):
        
    
    # # Creat session object
    summarydir = os.path.join(rootdir, animalid, session, fov, 'summaries')
    session_outfile = os.path.join(summarydir, 'sessiondata.pkl')
    if os.path.exists(session_outfile) and create_new is False:
        print("... loading session object")
        with open(session_outfile, 'rb') as f:
            S = pkl.load(f)
        
    else:
        print("... creating new session object")
        S = util.Session(animalid, session, fov, rootdir=rootdir)
    
        # Save session data object
        if not os.path.exists(summarydir):
            os.makedirs(summarydir)
            
        with open(session_outfile, 'wb') as f:
            pkl.dump(S, f, protocol=pkl.HIGHEST_PROTOCOL)
            print("... new session object to: %s" % session_outfile)
            
    print("... got session object w/ experiments:", S.experiments)
    
#    data_identifier = '|'.join([S.animalid, S.session, S.fov, S.traceid, S.rois])
#    print("(*** %s ***" % data_identifier)

    try:
        print("Found %i experiments in current session:" % len(S.experiment_list), S.experiment_list)
        assert 'rfs' in S.experiment_list or 'rfs10' in S.experiment_list, "ERROR:  No receptive field mapping found for current dataset: [%s|%s|%s]" % (S.animalid, S.session, S.fov)
    except Exception as e:
        print e
        return None
    
    
    return S


def group_configs(group, response_type):
    '''
    Takes each trial's reponse for specified config, and puts into dataframe
    '''
    config = group['config'].unique()[0]
    group.index = np.arange(0, group.shape[0])

    return pd.DataFrame(data={'%s' % config: group[response_type]})
    

#%%

# ############################################################################
# Functions for receptive field fitting and evaluation
# ############################################################################

def get_empirical_ci(stat, alpha=0.95):
    p = ((1.0-alpha)/2.0) * 100
    lower = np.percentile(stat, p) #max(0.0, np.percentile(stat, p))
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = np.percentile(stat, p) # min(1.0, np.percentile(x0, p))
    #print('%.1f confidence interval %.2f and %.2f' % (alpha*100, lower, upper))
    return lower, upper

def plot_bootstrapped_position_estimates(x0, y0, true_x, true_y, alpha=0.9):
    lower_x0, upper_x0 = get_empirical_ci(x0, alpha=alpha)
    lower_y0, upper_y0 = get_empirical_ci(y0, alpha=alpha)

    fig, axes = pl.subplots(1, 2, figsize=(5,3))
    ax=axes[0]
    ax.hist(x0, color='k', alpha=0.5)
    ax.axvline(x=lower_x0, color='k', linestyle=':')
    ax.axvline(x=upper_x0, color='k', linestyle=':')
    ax.axvline(x=true_x, color='r', linestyle='-')
    ax.set_title('x0 (n=%i)' % len(x0))
    
    ax=axes[1]
    ax.hist(y0, color='k', alpha=0.5)
    ax.axvline(x=lower_y0, color='k', linestyle=':')
    ax.axvline(x=upper_y0, color='k', linestyle=':')
    ax.axvline(x=true_y, color='r', linestyle='-')
    ax.set_title('y0 (n=%i)' % len(y0))
    pl.subplots_adjust(wspace=0.5, top=0.8)
    
    return fig

#%%


from functools import partial
from contextlib import contextmanager

@contextmanager
def poolcontext(*args, **kwargs):
    pool = mp.Pool(*args, **kwargs)
    yield pool
    pool.terminate()


def bootstrap_rf_params(rdf, rows=[], cols=[], sigma_scale=2.35,
                            n_resamples=10, n_bootstrap_iters=1000):

    xres = np.unique(np.diff(row_vals))[0]
    yres = np.unique(np.diff(col_vals))[0]
    min_sigma=5; max_sigma=50;

    grouplist = [group_configs(group, response_type) for config, group in rdf.groupby(['config'])]
    responses_df = pd.concat(grouplist, axis=1) # indices = trial reps, columns = conditions

    # Get mean response across re-sampled trials for each condition, do this n-bootstrap-iters times
    bootdf = pd.concat([responses_df.sample(n_resamples, replace=True).mean(axis=0) for ni in range(n_bootstrap_iters)], axis=1)
    
    bparams = []; #x0=[]; y0=[];
    for ii in bootdf.columns:

        response_vector = bootdf[ii].values
        rfmap = fitrf.get_rf_map(response_vector, len(col_vals), len(row_vals))
        fitr, fit_y = fitrf.do_2d_fit(rfmap, nx=len(col_vals), ny=len(row_vals))
        if fitr['success']:
            amp_f, x0_f, y0_f, sigx_f, sigy_f, theta_f, offset_f = fitr['popt']
            if any(s < min_sigma for s in [abs(sigx_f)*xres*sigma_scale, abs(sigy_f)*yres*sigma_scale])\
                or any(s > max_sigma for s in [abs(sigx_f)*xres*sigma_scale, abs(sigy_f)*yres*sigma_scale]):
                fitr['success'] = False
                
        if fitr['success']:
            amp_f, x0_f, y0_f, sigx_f, sigy_f, theta_f, offset_f = fitr['popt']
            bparams.append(fitr['popt'])

    #%    
    bparams = np.array(bparams)  
    paramsdf = pd.DataFrame(data=bparams, columns=['amp', 'x0', 'y0', 'sigma_x', 'sigma_y', 'theta', 'offset'])
    paramsdf['cell'] = [rdf.index[0] for _ in range(bparams.shape[0])]
    
    return paramsdf

#def pool_bootstrap(rdf_list, n_processes=1):
#    pool = mp.Pool(processes=n_processes)
#    results = pool.map(bootstrap_roi_responses, rdf_list)
#    return results

def pool_bootstrap(rdf_list, params, n_processes=1):
    
    with poolcontext(processes=n_processes) as pool:
        results = pool.map(partial(bootstrap_rf_params, 
                                   rows=params['row_vals'], 
                                   cols=params['col_vals'],
                                   n_resamples=params['n_resamples'], 
                                   n_bootstrap_iters=params['n_bootstrap_iters']), rdf_list)
    return results
    
def merge_names(a, b, c=1, d=2):
    return '{} & {} - ({}, {})'.format(a, b, c, d)

#names = ['Brown', 'Wilson', 'Bartlett', 'Rivera', 'Molloy', 'Opie']
#with poolcontext(processes=2) as pool:
#    results = pool.map(partial(merge_names, b='Sons', c=100), names)
#print(results)
#

# In[2]:


rootdir = '/n/coxfs01/2p-data'

animalid = 'JC084' #JC076'
session = '20190522' #'20190501'
fov = 'FOV1_zoom2p0x'


traceid = 'traces001'
trace_type = 'corrected'

response_type = 'dff'
fit_thr = 0.5

create_new = True
convert_um = True


# Create session and experiment objects
S = util.Session(animalid, session, fov)
experiment_list = S.get_experiment_list(traceid=traceid)
assert 'rfs' in experiment_list or 'rfs10' in experiment_list, "NO receptive field experiments found!"
rfname = 'rfs10' if 'rfs10' in experiment_list else 'rfs'
exp = util.ReceptiveFields(rfname, S.animalid, S.session, S.fov, traceid=traceid)
exp.print_info()

# Get output dir for stats
estats = exp.get_stats(response_type=response_type, fit_thr=fit_thr)
statsdir, stats_desc = rstats.create_stats_dir(exp.animalid, exp.session, exp.fov,
                                              traceid=exp.traceid, trace_type=exp.trace_type,
                                              response_type=response_type)
print("Saving stats output to: %s" % statsdir)

# Get RF dir for current fit type
fit_desc = fitrf.get_fit_desc(response_type=response_type)
rfdir = glob.glob(os.path.join(rootdir, exp.animalid, exp.session, exp.fov, 
                               exp.name, 'traces/%s*' % exp.traceid,
                               'receptive_fields', fit_desc))[0]

data_identifier = '|'.join([exp.animalid, exp.session, exp.fov, exp.traceid, exp.rois, exp.trace_type, response_type])
        

# Get RF protocol info
row_vals, col_vals = estats.fitinfo['row_vals'], estats.fitinfo['col_vals']



#%% Bootstrap RF parameter estimates

# Parameters
n_bootstrap_iters=1000
n_resamples = 10
plot_distns = True
alpha = 0.95
n_processes=2
sigma_scale = 2.35

# Create output dir for bootstrap results
bootstrapdir = os.path.join(rfdir, 'evaluation')
if not os.path.exists(os.path.join(bootstrapdir, 'rois')):
    os.makedirs(os.path.join(bootstrapdir, 'rois'))

    
roi_list = estats.rois  # Get list of all cells that pass fit-thr
rdf_list = [estats.gdf.get_group(roi)[['config', 'trial', response_type]] for roi in roi_list]
bparams = {'row_vals': row_vals, 'col_vals': col_vals,
           'n_bootstrap_iters': n_bootstrap_iters, 'n_resamples': n_resamples}

start_t = time.time()
bootstrap_results = pool_bootstrap(rdf_list, bparams, n_processes=n_processes)
end_t = time.time() - start_t
print "Multiple processes: {0:.2f}sec".format(end_t)


bootdf = pd.concat(bootstrap_results)

xx, yy, sigx, sigy = fitrf.convert_fit_to_coords(bootdf, row_vals, col_vals)
bootdf['x0'] = xx
bootdf['y0'] = yy
bootdf['sigma_x'] = sigx * sigma_scale
bootdf['sigma_y'] = sigy * sigma_scale

#%%

alpha=0.95

# Plot bootstrapped distn of x0 and y0 parameters for each roi (w/ CIs)

counts = bootdf.groupby(['cell']).count()['x0']
unreliable = counts[counts < n_bootstrap_iters*0.5].index.tolist()
print("%i cells seem to have unreliable estimates." % len(unreliable))

# Plot distribution of params w/ 95% CI
if plot_distns:
    for roi, paramsdf in bootdf.groupby(['cell']):
        
        true_x = estats.fits['x0'][roi]
        true_y = estats.fits['y0'][roi]
        fig = plot_bootstrapped_position_estimates(paramsdf['x0'], paramsdf['y0'], true_x, true_y, alpha=alpha)
        fig.suptitle('roi %i' % int(roi+1))
        
        pl.savefig(os.path.join(bootstrapdir, 'rois', 'roi%05d_%i-bootstrap-iters_%i-resample' % (int(roi+1), n_bootstrap_iters, n_resamples)))
        pl.close()
            
#%% Plot estimated x0, y0 as a function of r2 rank (plot top 30 neurons)

sorted_r2 = estats.fits['r2'].argsort()[::-1]
sorted_rois = np.array(roi_list)[sorted_r2.values]
for roi in sorted_rois:
    print roi, estats.fits['r2'][roi]

dflist = []
for roi, d in bootdf.groupby(['cell']): #.items():
    if roi not in sorted_rois[0:30]:
        continue
    tmpd = d.copy()
    tmpd['cell'] = [roi for _ in range(len(tmpd))]
    tmpd['r2_rank'] = [sorted_r2[roi] for _ in range(len(tmpd))]
    dflist.append(tmpd)
df = pd.concat(dflist, axis=0)

fig, axes = pl.subplots(1,2)
sns.boxplot(x='r2_rank', y='x0', data=df, ax=axes[0])
sns.boxplot(x='r2_rank', y='y0', data=df, ax=axes[1])


#%% Load session's rois:
S.load_data(rfname, traceid=traceid) # Load data to get traceid and roiid


masks, zimg = S.load_masks(rois=exp.rois)
npix_y, npix_x = zimg.shape

# Create contours from maskL
roi_contours = coor.contours_from_masks(masks)

# Convert to brain coords
fov_pos_x, rf_xpos, xlim, fov_pos_y, rf_ypos, ylim = coor.get_roi_position_um(estats.fits, roi_contours, 
                                                                     rf_exp_name=rfname,
                                                                     convert_um=True,
                                                                     npix_y=npix_y,
                                                                     npix_x=npix_x)

posdf = pd.DataFrame({'xpos_fov': fov_pos_y,
                       'xpos_rf': rf_xpos,
                       'ypos_fov': fov_pos_x,
                       'ypos_rf': rf_ypos}, index=roi_list)

#%%
params = [p for p in bootdf.columns if p != 'cell']

CI = {}
for p in params:
    CI[p] = dict((roi, get_empirical_ci(bdf[p].values, alpha=alpha)) for roi, bdf in bootdf.groupby(['cell']))

cis = {}
for p in params:
    cvals = np.array([get_empirical_ci(bdf[p].values, alpha=alpha) for roi, bdf in bootdf.groupby(['cell'])])
    cis['%s_lower' % p] = cvals[:, 0]
    cis['%s_upper' % p] = cvals[:, 1]
cis = pd.DataFrame(cis, index=[roi_list])


#%% Fit linear regression for brain coords vs VF coords

from sklearn.linear_model import LinearRegression
import scipy.stats as spstats
import sklearn.metrics as skmetrics #import mean_squared_error

#
#df = pd.DataFrame({'fov': fov_pos_y,
#                   'vf': rf_xpos}, index=roi_list)

def fit_linear_regr(xvals, yvals):
    regr = LinearRegression()
    if len(xvals.shape) == 1:
        xvals = np.array(xvals).reshape(-1, 1)
        yvals = np.array(yvals).reshape(-1, 1)
    else:
        xvals = np.array(xvals)
        yvals = np.array(yvals)
    regr.fit(xvals, yvals)
    fitv = regr.predict(xvals)
    return fitv.reshape(-1)


y_pred = fit_linear_regr(posdf['xpos_fov'], posdf['xpos_rf'])
y_true = posdf['xpos_rf'].values

mse = skmetrics.mean_squared_error(y_true, y_pred)
var = skmetrics.r2_score(y_true, y_pred)
print("Mean squared error: %.2f" % mse)
print('Variance score: %.2f' % var)

pl.figure()

df['dist'] = posdf['xpos_rf']- fitv

    
fig, axes = pl.subplots(1, 3, figsize=(10, 3))
ax=axes[0]
ax.scatter(posdf['xpos_fov'], posdf['xpos_rf'], c='k', alpha=0.5)
ax.set_title('Azimuth')
ax.set_ylabel('RF position (rel. deg.)')
ax.set_xlabel('FOV position (um)')
ax.set_xlim([0, ylim])
sns.despine(offset=1, trim=True, ax=ax)
ax.plot(xv, fitv, 'r')
r, p = spstats.pearsonr(posdf['xpos_fov'], posdf['xpos_rf'].abs())
ax.text(0.5, ax.get_ylim()[-1]-1, 'pearson=%.2f (p=%.2f)' % (r, p), fontsize=8)

ax = axes[1]
ax.hist(df['dist'], histtype='step', color='k')
sns.despine(offset=1, trim=True, ax=ax)
ax.set_xlabel('distance')
ax.set_ylabel('counts')

ax = axes[2]
r2_vals = estats.fits['r2']
ax.scatter(r2_vals, df['dist'].abs(), c='k', alpha=0.5)
ax.set_xlabel('r2')
ax.set_ylabel('abs(distance)')
testregr = LinearRegression()
testregr.fit(r2_vals.reshape(-1, 1), df['dist'].abs().values.reshape(-1, 1)) #, yv)
r2_dist_corr = testregr.predict(r2_vals.reshape(-1, 1))
ax.plot(r2_vals, r2_dist_corr, 'r')
sns.despine(offset=1, trim=True, ax=ax)
r, p = spstats.pearsonr(r2_vals.values, df['dist'].abs())
ax.text(0.5, ax.get_ylim()[-1], 'pearson=%.2f (p=%.2f)' % (r, p), fontsize=8)

pl.subplots_adjust(top=0.8, bottom=0.2)