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
from pipeline.python.classifications import run_experiment_stats as rstats
from pipeline.python.utils import label_figure, natural_keys, convert_range

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
# Functions for processing ROIs (masks)
# ############################################

def get_roi_position_um(rffits, tmp_roi_contours, rf_exp_name='rfs', convert_um=True, npix_y=512, npix_x=512,
                        xaxis_conversion=2.312, yaxis_conversion=1.904):
    
    '''
    From 20190605 PSF measurement:
        xaxis_conversion = 2.312
        yaxis_conversion = 1.904
    '''
    
    # Sort ROIs b y x,y position:
    sorted_roi_indices_xaxis, sorted_roi_contours_xaxis = spatially_sort_contours(tmp_roi_contours, sort_by='x')
    sorted_roi_indices_yaxis, sorted_roi_contours_yaxis = spatially_sort_contours(tmp_roi_contours, sort_by='y')

    _, sorted_roi_centroids_xaxis = spatially_sort_contours(tmp_roi_contours, sort_by='x', get_centroids=True)
    _, sorted_roi_centroids_yaxis = spatially_sort_contours(tmp_roi_contours, sort_by='y', get_centroids=True)

    # Convert pixels to um:
    xlinspace = np.linspace(0, npix_x*xaxis_conversion, num=npix_x)
    ylinspace = np.linspace(0, npix_y*yaxis_conversion, num=npix_y)
    
    # ---- Spatially sorted ROIs vs. RF position -----------------------
    rf_rois = rffits.index.tolist() 
    #colors = ['k' for _ in range(len(rf_rois))]
    # Get values for azimuth:
    spatial_rank_x = [sorted_roi_indices_xaxis.index(roi) for roi in rf_rois] # Get sorted rank for indexing
    pixel_order_x = [sorted_roi_centroids_xaxis[s] for s in spatial_rank_x]    # Get corresponding spatial position in FOV
    pixel_order_xvals = [p[0] for p in pixel_order_x]
    if convert_um:
        fov_pos_x = [xlinspace[p] for p in pixel_order_xvals]
        xlim=xlinspace.max()
    else:
        fov_pos_x = pixel_order_xvals
        xlim = npix_x
    rf_xpos = rffits['x0'][rf_rois] #[gdfs[rf_exp_name].fits.loc[roi]['x0'] for roi in rf_rois]

    # Get values for elevation
    spatial_rank_y = [sorted_roi_indices_yaxis.index(roi) for roi in rf_rois] # Get sorted rank for indexing
    pixel_order_y = [sorted_roi_centroids_yaxis[s] for s in spatial_rank_y]    # Get corresponding spatial position in FOV
    pixel_order_yvals = [p[1] for p in pixel_order_y]
    if convert_um:
        fov_pos_y = [ylinspace[p] for p in pixel_order_yvals]
        ylim = ylinspace.max()
    else:
        fov_pos_y = pixel_order_yvals
        ylim = npix_y
    rf_ypos = rffits['y0'][rf_rois] #[gdfs[rf_exp_name].fits.loc[roi]['y0'] for roi in rf_rois]

    return fov_pos_x, rf_xpos, xlim, fov_pos_y, rf_ypos, ylim


def contours_from_masks(masks):
    # Cycle through all ROIs and get their edges
    # (note:  tried doing this on sum of all ROIs, but fails if ROIs are overlapping at all)
    tmp_roi_contours = []
    for ridx in range(masks.shape[-1]):
        im = masks[:,:,ridx]
        im[im>0] = 1
        im[im==0] = np.nan #1
        im = im.astype('uint8')
        edged = cv2.Canny(im, 0, 0.9)
        tmp_cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        tmp_cnts = tmp_cnts[0] if imutils.is_cv2() else tmp_cnts[1]
        tmp_roi_contours.append((ridx, tmp_cnts[0]))
    print "Created %i contours for rois." % len(tmp_roi_contours)
    
    return tmp_roi_contours


def spatially_sort_contours(tmp_roi_contours, sort_by='xy', get_centroids=False):

    if sort_by == 'xy':
        sorted_rois =  sorted(tmp_roi_contours, key=lambda ctr: (cv2.boundingRect(ctr[1])[1] + cv2.boundingRect(ctr[1])[0]) * zimg.shape[1])  
    elif sort_by == 'x':
        sorted_rois = sorted(tmp_roi_contours, key=lambda ctr: cv2.boundingRect(ctr[1])[0])
    elif sort_by == 'y':
        sorted_rois = sorted(tmp_roi_contours, key=lambda ctr: cv2.boundingRect(ctr[1])[1])
    else:
        print("Unknown sort-by: %s" % sort_by)
        return None
    
    sorted_ixs = [c[0] for c in sorted_rois]
    
    if get_centroids:
        sorted_contours = [get_contour_center(cnt[1]) for cnt in sorted_rois]
    else:
        sorted_contours = [cnt[1] for cnt in sorted_rois]
        
    return sorted_ixs, sorted_contours

def get_contour_center(cnt):

    M = cv2.moments(cnt)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return (cX, cY)


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



# In[2]:


rootdir = '/n/coxfs01/2p-data'

animalid = 'JC084' #JC076'
session = '20190522' #'20190501'
fov = 'FOV1_zoom2p0x'


traceid = 'traces001'

create_new = True
trace_type = 'corrected'
responsive_test = 'ROC'
convert_um = True


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
        
#        if int(S.session) < 20190511 and 'rfs' in S.experiment_list:
#            # Old experiment, where "gratings" were actually RFs
#            S.load_data(experiment='gratings', traceid=traceid, trace_type=trace_type)
#        else:
#            if 'rfs' in S.experiment_list:
#                S.load_data(experiment='rfs', traceid=traceid, trace_type=trace_type)
#            if 'rfs10' in S.experiment_list:
#                S.load_data(experiment='rfs', traceid=traceid, trace_type=trace_type)
#                
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
    

#%%

def group_configs(group, response_type):
    '''
    Takes each trial's reponse for specified config, and puts into dataframe
    '''
    config = group['config'].unique()[0]
    group.index = np.arange(0, group.shape[0])

    return pd.DataFrame(data={'%s' % config: group[response_type]})

def get_empirical_ci(stat, alpha=0.95):
    p = ((1.0-alpha)/2.0) * 100
    lower = np.percentile(stat, p) #max(0.0, np.percentile(stat, p))
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = np.percentile(stat, p) # min(1.0, np.percentile(x0, p))
    print('%.1f confidence interval %.2f and %.2f' % (alpha*100, lower, upper))
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



def bootstrap_roi_responses(rdf, n_resamples=10, n_bootstrap_iters=1000):
    grouplist = [group_configs(group, response_type) for config, group in rdf.groupby(['config'])]
    responses_df = pd.concat(grouplist, axis=1) # indices = trial reps, columns = conditions

    # Get mean response across re-sampled trials for each condition, do this n-bootstrap-iters times
    bootdf = pd.concat([responses_df.sample(n_resamples, replace=True).mean(axis=0) for ni in range(n_bootstrap_iters)], axis=1)
    
    bparams = []; #x0=[]; y0=[];
    for ii in bootdf.columns:
#        if ii % 100 == 0:
#            print("%i of %i iters." % (ii, n_bootstrap_iters))
#            
        response_vector = bootdf[ii].values
        rfmap = fitrf.get_rf_map(response_vector, len(col_vals), len(row_vals))
        fitr, fit_y = fitrf.do_2d_fit(rfmap, nx=len(col_vals), ny=len(row_vals))
        # {'popt': popt, 'pcov': pcov, 'init': initial_guess, 'r2': r2, 'success': success}, fitr
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

def pool_bootstrap(rdf_list):
    pool = mp.Pool()
    results = pool.map(bootstrap_roi_responses, rdf_list)
    return results

    
#%%

trace_type = 'corrected'
S = get_session_object(animalid, session, fov, traceid=traceid, trace_type=trace_type,
                       create_new=True, rootdir=rootdir)

if S is not None:
    
    
    # Get Receptive Field measures:
    rf_exp_name = 'rfs10' if 'rfs10' in S.experiment_list else 'rfs'

    # Get grouped roi stat metrics:
    gdfs, statsdir, stats_desc, nostats = rstats.get_session_stats(S, response_type='dff', 
                                                          experiment_list=S.experiment_list,
                                                          responsive_test=responsive_test, 
                                                          traceid=traceid, trace_type=trace_type,
                                                          create_new=False, rootdir=rootdir,
                                                          pretty_plots=False)
    
    
    # Create output dir for figures:
    statsfigdir = os.path.join(statsdir, 'figures')
    if not os.path.exists(statsfigdir):
        os.makedirs(statsfigdir)
        
        
    #%%
    response_type = 'dff'

    row_vals = gdfs[rf_exp_name].fitinfo['row_vals']
    col_vals = gdfs[rf_exp_name].fitinfo['col_vals']
    xres = np.unique(np.diff(row_vals))[0]
    yres = np.unique(np.diff(col_vals))[0]
    sigma_scale = 2.35
    min_sigma=5; max_sigma=50;
    
    #%%
    
    rfdir = glob.glob(os.path.join(rootdir, S.animalid, S.session, S.fov, '*rfs*', 'traces', '%s*' % traceid,
                                   'receptive_fields', 'fit-2dgaus_%s-no-cutoff' % response_type))[0]
            
    bootstrapdir = os.path.join(rfdir, 'evaluation')
    if not os.path.exists(os.path.join(bootstrapdir, 'rois')):
        os.makedirs(os.path.join(bootstrapdir, 'rois'))
    
        
    data_identifier = '|'.join([S.animalid, S.session, S.fov, S.traceid, S.rois, S.trace_type, responsive_test])
    rf_rois = gdfs[rf_exp_name].rois #gdfs[rf_exp_name].fits.index.tolist() # These are all ROIs w/ r2 fit > 0.5

        #%%
    
    
    n_bootstrap_iters=1000
    n_resamples = 10
    plot_distns = True
    alpha = 0.95
    
    
    rdf_list = [gdfs[rf_exp_name].gdf.get_group(roi)[['config', 'trial', response_type]] for roi in rf_rois]
    start_t = time.time()
    bootstrap_results = pool_bootstrap(rdf_list)
    end_t = time.time() - start_t
    
    print "Multiple processes: {0:.2f}sec".format(end_t)
    
    
    #%%
    bootdf = pd.concat(bootstrap_results)
    
    xx, yy, sigx, sigy = fitrf.convert_fit_to_coords(bootdf, row_vals, col_vals)
    bootdf['x0'] = xx
    bootdf['y0'] = yy
    bootdf['sigma_x'] = sigx
    bootdf['sigma_y'] = sigy

    #%%
    counts = bootdf.groupby(['cell']).count()['x0']
    unreliable = counts[counts < n_bootstrap_iters*0.5].index.tolist()
    
    # Plot distribution of params w/ 95% CI
    alpha=0.95
    if plot_distns:
        for roi, paramsdf in bootdf.groupby(['cell']):
            
            true_x = gdfs[rf_exp_name].fits['x0'][roi]
            true_y = gdfs[rf_exp_name].fits['y0'][roi]
            fig = plot_bootstrapped_position_estimates(paramsdf['x0'], paramsdf['y0'], true_x, true_y, alpha=alpha)
            fig.suptitle('roi %i' % int(roi+1))
            
            pl.savefig(os.path.join(bootstrapdir, 'rois', 'roi%05d_%i-bootstrap-iters_%i-resample' % (int(roi+1), n_bootstrap_iters, n_resamples)))
            pl.close()
                

    #%% box plot of top 30 r2 cells for distn of estimated param
    
    rffits = gdfs[rf_exp_name].fits
    rffits[rffits['r2'] > 0.5]
       
    # Plot estimated x0, y0 as a function of r2 rank (plot top 30 neurons):
    sorted_r2 = rffits['r2'].argsort()[::-1]
    sorted_rois = np.array(rf_rois)[sorted_r2.values]
    for roi in sorted_rois:
        print roi, rffits['r2'][roi]
    
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
        

    #%%
    # Load session's rois:
    S.load_data(rf_exp_name, traceid=traceid) # Load data to get traceid and roiid
    masks, zimg = S.load_masks()
    
    # Create contours from maskL
    roi_contours = contours_from_masks(masks)
    
    # Convert to brain coords
    fov_pos_x, rf_xpos, xlim, fov_pos_y, rf_ypos, ylim = get_roi_position_um(rffits, roi_contours, 
                                                                         rf_exp_name=rf_exp_name,
                                                                         convert_um=True)
    
    
    
    posdf = pd.DataFrame({'xpos_fov': fov_pos_y,
                       'xpos_rf': rf_xpos,
                       'ypos_fov': fov_pos_x,
                       'ypos_rf': rf_ypos
                       }, index=rf_rois)
            
    #%%
    params = [p for p in bootdf.columns if p != 'cell']
    
    CI = {}
    for p in params:
        CI[p] = dict((roi, get_empirical_ci(bdf[p].values, alpha=0.95)) for roi, bdf in bootdf.groupby(['cell']))
    
    cis = {}
    for p in params:
        cvals = np.array([get_empirical_ci(bdf[p].values, alpha=0.95) for roi, bdf in bootdf.groupby(['cell'])])
        cis['%s_lower' % p] = cvals[:, 0]
        cis['%s_upper' % p] = cvals[:, 1]
    cis = pd.DataFrame(cis, index=[rf_rois])


    #%% Fit linear regression for brain coords vs VF coords
    
    from sklearn.linear_model import LinearRegression
    import scipy.stats as spstats
    
    df = pd.DataFrame({'fov': fov_pos_y,
                       'vf': rf_xpos}, index=rf_rois)
    
    regr = LinearRegression()
    xv = np.array(df['fov']).reshape(-1, 1) 
    yv = np.array(df['vf']).reshape(-1, 1)
    regr.fit(xv, yv)
    fitv = regr.predict(xv)
    df['dist'] = yv - fitv
    
        
    fig, axes = pl.subplots(1, 3, figsize=(10, 3))
    ax=axes[0]
    ax.scatter(df['fov'], df['vf'], c='k', alpha=0.5)
    ax.set_title('Azimuth')
    ax.set_ylabel('RF position (rel. deg.)')
    ax.set_xlabel('FOV position (um)')
    ax.set_xlim([0, ylim])
    sns.despine(offset=1, trim=True, ax=ax)
    ax.plot(xv, fitv, 'r')
    r, p = spstats.pearsonr(df['fov'], df['vf'].abs())
    ax.text(0.5, ax.get_ylim()[-1]-1, 'pearson=%.2f (p=%.2f)' % (r, p), fontsize=8)
    
    ax = axes[1]
    ax.hist(df['dist'], histtype='step', color='k')
    sns.despine(offset=1, trim=True, ax=ax)
    ax.set_xlabel('distance')
    ax.set_ylabel('counts')
    
    ax = axes[2]
    r2_vals = gdfs[rf_exp_name].fits['r2']
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
    
#    sns.jointplot(r2_vals, df['dist'].abs())

    #%%
    fig, ax = pl.subplots()
    
    ax.set_title('Azimuth')
    ax.set_ylabel('RF position (rel. deg.)')
    ax.set_xlabel('FOV position (um)')
    ax.set_xlim([0, ylim])
    
    ax.plot(xv, fitv, 'r') # regression line
    
    # Get rois sorted by position:
    plot_rois = [r for r in rf_rois if r not in unreliable]
    sortby_fov = posdf['xpos_fov'][plot_rois].argsort()
    sorted_rois_fov = np.array(plot_rois)[sortby_fov]
    
    xvals = posdf['xpos_fov'][sorted_rois_fov].values
    x0_meds = [bootdf.groupby(['cell']).get_group(roi)['x0'].median() for roi in sorted_rois_fov]
    x0_lower = cis['x0_lower'][sorted_rois_fov]
    x0_upper = cis['x0_upper'][sorted_rois_fov]
    
    ax.scatter(xvals, x0_meds, c='k', marker='_')
    ax.errorbar(xvals, x0_meds, yerr=np.array(zip(x0_meds-x0_lower, x0_upper-x0_meds)).T, 
            fmt='none', color='k', alpha=0.5)
    
    ax.scatter(posdf['xpos_fov'][sorted_rois_fov], posdf['xpos_rf'][sorted_rois_fov], c='cornflowerblue', marker='_', alpha=1.0)
    ax.set_xlim([0, xlim])
    ax.set_xticks(np.arange(0, xlim, 100))
    sns.despine(offset=1, trim=True, ax=ax)
    
    for roi in sorted_rois_fov[-10:]:
        ix = list(sorted_rois_fov).index(roi)
        print roi, x0_lower.loc[roi], posdf['xpos_rf'][roi], x0_upper.loc[roi]
    
    rs = []
    for roi,lo,up,med in zip(sorted_rois_fov, x0_lower, x0_upper, x0_meds):
        if lo <= med <= up:
            continue
        else:
            rs.append(roi)
    
#    #%%
#    
#    
#    fig, ax = pl.subplots()
#    #bp = pl.boxplot([bootdf.groupby(['cell']).get_group(roi)['x0'],])
#    bdata = [bootdf.groupby(['cell']).get_group(roi)['x0'] for roi in sorted_rois_fov]
#    bp = pl.boxplot(bdata, sym='.', whis=[2.5, 97.5], 
#                    usermedians=x0_meds, 
#                    conf_intervals=np.array(zip(x0_lower, x0_upper)),
#                    positions=posdf['xpos_fov'][sorted_rois_fov].values,
#                    medianprops={'color': 'cornflowerblue', 'lw': 2, 'alpha':1}, notch=False,
#                    boxprops={'lw': 0.5, 'color': 'k', 'alpha': 0.5},
#                    flierprops={'markersize': 1},
#                    widths=10,
#                    showfliers=False
#                    )
#    
#    ax.set_xlim([0, xlim])
#    ax.set_xticks([])
#    ax.plot(xv, fitv, 'k', alpha=0.5) # regression line
#    
#    ax.scatter(posdf['xpos_fov'], posdf['xpos_rf'], c='r', marker='o', s=5, alpha=1.0)







    #%%

    # Which cells' CI contain the regression line, and which don't?
    

    
    #%%
    
    
    sorted_dist = np.argsort(df['dist'].abs())[::-1]
    sortby_dist = np.array(rf_rois)[sorted_dist.values]
#    for roi in sortby_dist:
#        print roi, df['dist'].abs()[roi]

    pl.figure()
    pl.scatter(sortby_dist, [df['dist'].abs()[r] for r in sortby_dist])
    
    
    

    #%%
    # # Sort ROIs by x,y position:
    sorted_roi_indices_xaxis, sorted_roi_contours_xaxis = spatially_sort_contours(roi_contours, sort_by='x')
    sorted_roi_indices_yaxis, sorted_roi_contours_yaxis = spatially_sort_contours(roi_contours, sort_by='y')
    
    _, sorted_roi_centroids_xaxis = spatially_sort_contours(roi_contours, sort_by='x', get_centroids=True)
    _, sorted_roi_centroids_yaxis = spatially_sort_contours(roi_contours, sort_by='y', get_centroids=True)

    # x-axis in FOV = posterior to anterior, from left to right (0-->512)
    # y-axis in FOV = lateral to medial, from top to bottom (0-->512)
    
    # Color by RF position:

    #%% #### Plot

    convert_um = True
    transform = True

    fig, axes = pl.subplots(2,2)
    fig.patch.set_alpha(1)
    ### Plot ALL sorted rois:
    ax = axes[0,1] 
    util.plot_roi_contours(zimg, sorted_roi_indices_xaxis, sorted_roi_contours_xaxis, 
                           label_rois=rf_rois, label=False, single_color=False, overlay=False,
                           clip_limit=0.02, draw_box=False, thickness=2, 
                           ax=ax, transform=transform)
    ax.axis('off')
                        
    ax = axes[0,0]
    util.plot_roi_contours(zimg, sorted_roi_indices_yaxis, sorted_roi_contours_yaxis, label_rois=rf_rois,
                           clip_limit=0.008, label=False, draw_box=False, 
                           thickness=2, roi_color=(255, 255, 255), single_color=False, ax=ax, transform=transform)
    ax.axis('off')
                    
    ### Plot corresponding RF centroids:
    colors = ['k' for roi in rf_rois]
    units = 'um' if convert_um else 'pixels'
    # Get values for azimuth:    
    ax = axes[1,0]
    ax.scatter(fov_pos_y, rf_xpos, c=colors, alpha=0.5)
    ax.set_title('Azimuth')
    ax.set_ylabel('RF position (rel. deg.)')
    ax.set_xlabel('FOV position (%s)' % units)
    ax.set_xlim([0, ylim])
    sns.despine(offset=4, trim=True, ax=ax)
    # Get values for elevation:
    ax = axes[1,1]
    ax.scatter(fov_pos_x, rf_ypos, c=colors, alpha=0.5)
    ax.set_title('Elevation')
    ax.set_ylabel('RF position (rel. deg.)')
    ax.set_xlabel('FOV position (%s)' % units)
    ax.set_xlim([0, xlim])
    ax.axis('on')
    sns.despine(offset=4, trim=True, ax=ax)
    
    #pl.subplots_adjust(wspace=0.5, top=0.9, hspace=0.5, bottom=0.2)
    
    #%
    label_figure(fig, data_identifier)
    if transform:
        pl.savefig(os.path.join(statsfigdir, 'transformed_spatially_sorted_rois_%s_only_%s.png' % (rf_exp_name, units)))
    else:
        pl.savefig(os.path.join(statsfigdir, 'spatially_sorted_rois_%s_only_%s.png' % (rf_exp_name, units)))
    

#%%
# # Calculate RF sizes/overlap with stimuli


# In[27]:



def create_ellipse(center, lengths, angle=0):
    """
    create a shapely ellipse. adapted from
    https://gis.stackexchange.com/a/243462
    """
    circ = Point(center).buffer(1)
    ell = affinity.scale(circ, int(lengths[0]), int(lengths[1]))
    ellr = affinity.rotate(ell, angle)
    return ellr

from shapely.geometry import box



# Identify stimulus location for current session:
xpos, ypos = S.get_stimulus_coordinates()
    
# Get stimulus size(s):
stimsizes = S.get_stimulus_sizes()

    
# In[28]:


# Draw RFs
#rf_fits_df = gdfs[rf_exp_name].fits
xres = list(set(np.diff(gdfs[rf_exp_name].finfo['col_vals'])))[0]
yres = list(set(np.diff(gdfs[rf_exp_name].finfo['row_vals'])))[0]
sigma_scale = 2.36

print("X- and Y-res: (%i, %i)" % (xres, yres))


# In[29]:


gratings_color = 'orange'
blobs_color = 'blue'
colordict = {'gratings': gratings_color,
            'blobs-min': blobs_color,
            'blobs-max': blobs_color}


## create GRATINGS patch
s_gratings = create_ellipse((xpos, ypos), (gratings_sz/2., gratings_sz/2.), 0)
v_gratings = np.array(s_gratings.exterior.coords.xy)

## create BLOBS patch(es) - min/max
ry_min = ypos - blobs_sz_min/2.
rx_min = xpos - blobs_sz_min/2.
ry_max = ypos + blobs_sz_min/2.
rx_max = xpos + blobs_sz_min/2.
s_blobs_min = box(rx_min, ry_min, rx_max, ry_max)
v_blobs_min = np.array(s_blobs_min.exterior.coords.xy)

ry_min = ypos - blobs_sz_max/2.
rx_min = xpos - blobs_sz_max/2.
ry_max = ypos + blobs_sz_max/2.
rx_max = xpos + blobs_sz_max/2.
s_blobs_max = box(rx_min, ry_min, rx_max, ry_max)
v_blobs_max = np.array(s_blobs_max.exterior.coords.xy)


# #### Plot

# In[30]:




# Get screen bounds: [bottom left upper right]
screen_bounds = [S.screen['linminH'], S.screen['linminW'], S.screen['linmaxH'], S.screen['linmaxW']]
screen_aspect = S.screen['resolution'][0] / S.screen['resolution'][1]


# PLOT
fig = pl.figure(figsize=(8,6))
fig.patch.set_alpha(1)

# Screen visualization ----------------------------------------------------
ax0 = pl.subplot2grid((3, 4), (0, 0), colspan=2, rowspan=1)
ax0.set_xlim([screen_bounds[1], screen_bounds[3]])
ax0.set_ylim([screen_bounds[0], screen_bounds[2]])
ax0.set_aspect(screen_aspect)
ax0.tick_params(axis='both', which='both', length=0, labelsize=6)


# Draw receptive fields, calculate overlap(s):
in_gratings = []
in_blobs_min = []
in_blobs_max = []
rf_dist_from_center = []
rf_avg_size = []
for roi in sorted(gdfs[rf_exp_name].rois):
    sx, sy, th, x0, y0 = gdfs[rf_exp_name].fits.loc[roi]
    s_ell = create_ellipse((x0, y0), (abs(sx)*sigma_scale/2., abs(sy)*sigma_scale/2.), np.rad2deg(th))
    v_ell = np.array(s_ell.exterior.coords.xy)
    p_ell = Polygon(v_ell.T, edgecolor='k', alpha=0.5, facecolor='none', lw=0.2)
    ax0.add_patch(p_ell)
    
    ## get intersection and compute areas/ratios:
    intersect_wgratings = s_ell.intersection(s_gratings)
    intersect_wblobs_min = s_ell.intersection(s_blobs_min)
    intersect_wblobs_max = s_ell.intersection(s_blobs_max)
    
    in_gratings.append(intersect_wgratings.area / s_ell.area)
    in_blobs_min.append(intersect_wblobs_min.area / s_ell.area)
    in_blobs_max.append(intersect_wblobs_max.area / s_ell.area)
    
    ## get distance bw RF centers and stimulus location:
    rf_dist_from_center.append(np.sqrt((x0 - xpos)**2 + (y0 - ypos)**2))
    rf_avg_size.append(np.mean([abs(sx)*sigma_scale/2., abs(sy)*sigma_scale/2.]))
    

# Draw patches:
p_gratings = Polygon(v_gratings.T, edgecolor=gratings_color, alpha=0.5, lw=2, facecolor='none', label='gratings')
ax0.add_patch(p_gratings)
p_blobs_min = Polygon(v_blobs_min.T, edgecolor=blobs_color, alpha=0.5, lw=2, facecolor='none', label='blobs-min')
ax0.add_patch(p_blobs_min)
p_blobs_max = Polygon(v_blobs_max.T, edgecolor=blobs_color, alpha=0.5, lw=2, facecolor='none', label='blobs-max')
ax0.add_patch(p_blobs_max)


# ---- Proportion of RF overlapping with stimulus bounds ----
all_overlap_values = copy.copy(in_gratings)
stimulus_labels = ['gratings' for _ in range(len(in_gratings))]
all_overlap_values.extend(in_blobs_min)
stimulus_labels.extend(['blobs-min' for _ in range(len(in_blobs_min))])
all_overlap_values.extend(in_blobs_max)
stimulus_labels.extend(['blobs-max' for _ in range(len(in_blobs_max))])
stimulus_colors = [colordict[e] for e in stimulus_labels]

overlap_df = pd.DataFrame({'stimulus': stimulus_labels,
                          'overlap': np.array(all_overlap_values)*100.,
                          'color': stimulus_colors})

ax = pl.subplot2grid((3, 4), (0, 2), colspan=2, rowspan=1)
ax = sns.boxplot(x="stimulus", y="overlap", data=overlap_df, ax=ax, 
                 palette=colordict, saturation=1.0, notch=True, boxprops=dict(alpha=.5))
ax.set_xlabel('')
ax.set_ylabel('Overlap area\n(% of RF)', fontsize=8)
ax.set_ylim([0, 100])
ax.tick_params(axis='x', which='both', length=0, labelsize=8)
ax.tick_params(axis='y', which='both', length=0, labelsize=8)

sns.despine(trim=True, offset=4, ax=ax, bottom=True)



# ---- Average RF size -----------------------
ax2a = pl.subplot2grid((3, 4), (1, 2), colspan=1, rowspan=1)
sns.distplot(rf_avg_size, ax=ax2a, color='k')
ax2a.set_xlim([0, max(rf_avg_size)+10])
ax2a.set_xlabel('Average RF size\n(deg)', fontsize=8)
ax2a.set_ylabel('kde', fontsize=8)
ax2a.tick_params(axis='both', which='both', length=3, labelsize=8)
ax2a.yaxis.set_major_locator(MaxNLocator(2))
ax2a.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
sns.despine(trim=True, offset=4, ax=ax2a)

# ---- Distance from stimulus center -----------------------
ax2b = pl.subplot2grid((3, 4), (1, 3), colspan=1, rowspan=1)
sns.distplot(rf_dist_from_center, ax=ax2b, color='k')
ax2b.set_xlabel('RF distance from\nstimulus center', fontsize=8)
ax2b.set_ylabel('kde', fontsize=8)
ax2b.tick_params(axis='both', which='both', length=3, labelsize=8)
ymax = max([ax2a.get_ylim()[-1], ax2b.get_ylim()[-1]])
ax2b.set_ylim([0, ymax])
ax2b.yaxis.set_major_locator(MaxNLocator(2))
ax2b.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
sns.despine(trim=True, offset=4, ax=ax2b)


# ---- Spatially sorted ROIs vs. RF position -----------------------
rf_rois = gdfs[rf_exp_name].fits.index.tolist() 
colors = ['k' for _ in range(len(rf_rois))]

# Get values for azimuth/elevation:
fov_pos_x, rf_xpos, xlim, fov_pox_y, rf_ypos, ylim = get_roi_position_um(tmp_roi_contours,
                                                                         rf_exp_name=rf_exp_name,
                                                                         convert_um=True)


ax3a = pl.subplot2grid((3, 4), (1, 0), colspan=1, rowspan=1)
ax3a.scatter(fov_pos_y, rf_xpos, c=colors, alpha=0.3) # FOV y-axis is left-right on brain
#ax3a.set_title('Azimuth', fontsize=8)
ax3a.set_ylabel('Azimuth\n(rel. deg.)', fontsize=8)
ax3a.set_xlabel('FOV position\n(%s)' % units, fontsize=8)
ax3a.set_xlim([0, ylim])
ax3a.yaxis.set_major_locator(MaxNLocator(5))
sns.despine(trim=True, offset=4, ax=ax3a)

ax3b = pl.subplot2grid((3, 4), (1, 1), colspan=1, rowspan=1)
ax3b.scatter(fov_pos_x, rf_ypos, c=colors, alpha=0.3) # FOV x-axis is posterior-anterior on brain
ax3b.set_ylabel('Elevation\n(rel. deg.)', fontsize=8)
ax3b.set_xlabel('FOV position\n(%s)' % units, fontsize=8)
ax3b.set_xlim([0, xlim])
#ax.set_aspect('auto')
ax3b.yaxis.set_major_locator(MaxNLocator(5))
sns.despine(trim=True, offset=4, ax=ax3b)


pl.subplots_adjust(left=0.1, top=0.9, right=0.99, wspace=0.5, hspace=0.5)


bbox_s = ax2b.get_position()
bbox_s2 = [bbox_s.x0 - 0.01, bbox_s.y0,  bbox_s.width, bbox_s.height] 
ax2b.set_position(bbox_s2) # set a new position

# Move upper-left plot over to reduce white space
bbox = ax0.get_position()
print(bbox)
bbox2 = [bbox.x0 - 0.04, bbox.y0+0.0,  bbox.width-0.04, bbox.height+0.05] 
ax0.set_position(bbox2) # set a new position
ax0.legend(fontsize=6, ncol=3, loc='lower center', bbox_to_anchor=[0.5, -0.25])

label_figure(fig, data_identifier)

pl.savefig(os.path.join(statsfigdir, 'visual_field_coverage_w%s_%s.png' % (rf_exp_name, units)))


# In[31]:


print(output_dir)


# In[ ]:





def get_percentiles_from_box_plots(bp):
    percentiles = []
    for i in range(len(bp['boxes'])):
        percentiles.append((bp['caps'][2*i].get_ydata()[0],
                           bp['boxes'][i].get_ydata()[0],
                           bp['medians'][i].get_ydata()[0],
                           bp['boxes'][i].get_ydata()[2],
                           bp['caps'][2*i + 1].get_ydata()[0],
                           (bp['fliers'][i].get_xdata(),
                            bp['fliers'][i].get_ydata())))
    return percentiles

def customized_box_plot(box_plot, percentiles, axes, redraw = True, *args, **kwargs):
    """
    Generates a customized boxplot based on the given percentile values
    """

    #box_plot = axes.boxplot([[-9, -4, 2, 4, 9],]*n_box, *args, **kwargs) 
    # Creates len(percentiles) no of box plots

    min_y, max_y = float('inf'), -float('inf')

    for box_no, (q1_start, 
                 q2_start,
                 q3_start,
                 q4_start,
                 q4_end,
                 fliers_xy) in enumerate(percentiles):

        # Lower cap
        box_plot['caps'][2*box_no].set_ydata([q1_start, q1_start])
        # xdata is determined by the width of the box plot

        # Lower whiskers
        box_plot['whiskers'][2*box_no].set_ydata([q1_start, q2_start])

        # Higher cap
        box_plot['caps'][2*box_no + 1].set_ydata([q4_end, q4_end])

        # Higher whiskers
        box_plot['whiskers'][2*box_no + 1].set_ydata([q4_start, q4_end])

        # Box
        box_plot['boxes'][box_no].set_ydata([q2_start, 
                                             q2_start, 
                                             q4_start,
                                             q4_start,
                                             q2_start])

        # Median
        box_plot['medians'][box_no].set_ydata([q3_start, q3_start])

        # Outliers
        if fliers_xy is not None and len(fliers_xy[0]) != 0:
            # If outliers exist
            box_plot['fliers'][box_no].set(xdata = fliers_xy[0],
                                           ydata = fliers_xy[1])

            min_y = min(q1_start, min_y, fliers_xy[1].min())
            max_y = max(q4_end, max_y, fliers_xy[1].max())

        else:
            min_y = min(q1_start, min_y)
            max_y = max(q4_end, max_y)

        # The y axis is rescaled to fit the new box plot completely with 10% 
        # of the maximum value at both ends
        axes.set_ylim([min_y*1.1, max_y*1.1])

    # If redraw is set to true, the canvas is updated.
    if redraw:
        ax.figure.canvas.draw()

    return box_plot

#%

percentiles = get_percentiles_from_box_plots(bp)


fig, ax = pl.subplots()
b = customized_box_plot(bp, percentiles, ax, redraw=True, notch=0, sym='+', vert=1, whis=1.5)

