#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 11:49:39 2018

@author: juliana
"""
import datetime
import os
import imutils
import cv2
import glob
import h5py
import sys
import optparse

import pylab as pl
from collections import Counter
import seaborn as sns

import cPickle as pkl
import numpy as np
import pylab as pl
import pandas as pd
import seaborn as sns
import tifffile as tf

from pipeline.python.classifications import test_responsivity as resp #import calculate_roi_responsivity, group_roidata_stimresponse, find_barval_index
#from pipeline.python.classifications import osi_dsi as osi
#from pipeline.python.visualization import get_session_summary as ss
from pipeline.python.utils import natural_keys, label_figure

from pipeline.python.retinotopy import fit_2d_rfs as rf

from pipeline.python.utils import uint16_to_RGB
from skimage import exposure
from matplotlib import patches
    #%%
from scipy.interpolate import interp1d
import scipy.optimize as spopt

#%%



def cleanup_axes(axes_list, which_axis='y'):    
    for ax in axes_list: 
        if which_axis=='y':
            # get the yticklabels from the axis and set visibility to False
            for label in ax.get_yticklabels():
                label.set_visible(False)
            ax.yaxis.offsetText.set_visible(False)
        elif which_axis=='x':
            # get the xticklabels from the axis and set visibility to False
            for label in ax.get_xticklabels():
                label.set_visible(False)
            ax.xaxis.offsetText.set_visible(False)



#%%
#def angdir90(x):
#    '''wraps anguar diff values to interval 0, 180'''
#    return min(np.abs([x, x-180, x+180]))
#
#def single_gaussian( x, c1, mu, sigma, C ):
#    #(c1, c2, mu, sigma) = params
#    x1vals = np.array([angdir90(xi - mu) for xi in x])
#    res =   C + c1 * np.exp( - x1vals**2.0 / (2.0 * sigma**2.0) )
#    
##    res =   C + c1 * np.exp( - ((x - mu) % 360.)**2.0 / (2.0 * sigma**2.0) ) \
##            + c2 * np.exp( - ((x + 180 - mu) % 360.)**2.0 / (2.0 * sigma**2.0) )
#
##        res =   c1 * np.exp( - (x - mu)**2.0 / (2.0 * sigma**2.0) ) \
##                #+ c2 * np.exp( - (x - mu2)**2.0 / (2.0 * sigma2**2.0) )
#    return res
#
##
#
#def fit_orientation_selectivity(x, y, init_params=[0, 0, 0, 0]):
#    roi_fit = None
#    
#    popt, pcov = spopt.curve_fit(single_gaussian, x, y, p0=init_params, maxfev=1000)
#    fitr = single_gaussian( x, *popt)
#        
#    # Get residual sum of squares 
#    residuals = y - fitr
#    ss_res = np.sum(residuals**2)
#    ss_tot = np.sum((y - np.mean(y))**2)
#    r2 = 1 - (ss_res / ss_tot)
#    #print(r2)
#    
#    
#    if pcov.max() == np.inf: # or r2 == 1: #round(r2, 3) < 0.15 or 
#        success = False
#    else:
#        success = True
#    assert success
#    
#    if success:
#        roi_fit = {'pcov': pcov,
#                     'popt': popt,
#                     'fit_y': fitr,
#                     'r2': r2,
#                     'x': x,
#                     'y': y,
#                     'init': init_params,
#                     'configs': curr_cfgs,
#                     'success': success}
#    return roi_fit


#%%
def fit_direction_selectivity(x, y, init_params=[0, 0, 0, 0, 0], bounds=[np.inf, np.inf, np.inf, np.inf, np.inf]):
    roi_fit = None
    
    popt, pcov = spopt.curve_fit(double_gaussian, x, y, p0=init_params, maxfev=1000, bounds=bounds)
    fitr = double_gaussian( x, *popt)
        
    # Get residual sum of squares 
    residuals = y - fitr
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - (ss_res / ss_tot)
    #print(r2)
    
    
    if pcov.max() == np.inf: # or r2 == 1: #round(r2, 3) < 0.15 or 
        success = False
    else:
        success = True
    #assert success
    
    if success:
        roi_fit = {'pcov': pcov,
                     'popt': popt,
                     'fit_y': fitr,
                     'r2': r2,
                     'x': x,
                     'y': y,
                     'init': init_params,
                     'success': success}
    return roi_fit

#%
def angdir180(x):
    '''wraps anguar diff values to interval 0, 180'''
    return min(np.abs([x, x-360, x+360]))

def double_gaussian( x, c1, c2, mu, sigma, C ):
    #(c1, c2, mu, sigma) = params
    x1vals = np.array([angdir180(xi - mu) for xi in x])
    x2vals = np.array([angdir180(xi + 180 - mu) for xi in x])
    res =   C + c1 * np.exp( - x1vals**2.0 / (2.0 * sigma**2.0) ) \
            + c2 * np.exp( - x2vals**2.0 / (2.0 * sigma**2.0) )

#    res =   C + c1 * np.exp( - ((x - mu) % 360.)**2.0 / (2.0 * sigma**2.0) ) \
#            + c2 * np.exp( - ((x + 180 - mu) % 360.)**2.0 / (2.0 * sigma**2.0) )

#        res =   c1 * np.exp( - (x - mu)**2.0 / (2.0 * sigma**2.0) ) \
#                #+ c2 * np.exp( - (x - mu2)**2.0 / (2.0 * sigma2**2.0) )
    return res

#
#def double_gaussian2( x, params ):
#    c1, c2, mu, sigma, C = params
##    x1vals = np.array([angdir(xi - mu) for xi in x])
##    x2vals = np.array([angdir(xi + 180 - mu) for xi in x])
##    res =   c1 * np.exp( - x1vals**2.0 / (2.0 * sigma**2.0) ) \
##            + c2 * np.exp( - x2vals**2.0 / (2.0 * sigma**2.0) )
#    #print x    
#    #print( int(mu), ( (x- int(mu)) % 360 ))
#    mu = int(mu)
#    
#    res =   C + c1 * np.exp( - ( ((x-mu) % 360.)**2.0 ) / (2.0 * sigma**2.0) ) \
#            + c2 * np.exp( - ( ((x+180-mu) % 360.)**2.0) / (2.0 * sigma**2.0) )
#            
##        res =   c1 * np.exp( - (x - mu)**2.0 / (2.0 * sigma**2.0) ) \
##                #+ c2 * np.exp( - (x - mu2)**2.0 / (2.0 * sigma2**2.0) )
#    return res
#
#
#def double_gaussian_fit( params ):
#    fit = double_gaussian2( x, params )
#    return fit-y




def fit_leastsq(p0, datax, datay, function):

    errfunc = lambda p, x, y: function(x,p) - y

    pfit, pcov, infodict, errmsg, success = \
        spopt.leastsq(errfunc, p0, args=(datax, datay), \
                          full_output=1, epsfcn=0.0001)

    if (len(datay) > len(p0)) and pcov is not None:
        s_sq = (errfunc(pfit, datax, datay)**2).sum()/(len(datay)-len(p0))
        pcov = pcov * s_sq
    else:
        pcov = np.inf

    error = [] 
    for i in range(len(pfit)):
        try:
          error.append(np.absolute(pcov[i][i])**0.5)
        except:
          error.append( 0.00 )
    pfit_leastsq = pfit
    perr_leastsq = np.array(error) 
    return pfit_leastsq, perr_leastsq 


#%%
#
#oris = np.arange(0, 360, 45)
#for ori in oris:
#    orthog = (ori+90) % 360.
#    print ori, orthog
    


def hist_gratings_stats(statsdf, colorvals, ax=None,
                        thresh=0.33, ori_metric='OSI', show_selective=False):
    
    #datestring = datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")
    
    best_ori_vals = statsdf['pref_ori']
    best_ori_vals_selective = statsdf[statsdf[ori_metric] >= thresh]['pref_ori']
    
    ori_counts_all = Counter(best_ori_vals)
    ori_counts_selective = Counter(best_ori_vals_selective)
    for ori in ori_counts_all.keys():
        if ori not in ori_counts_selective.keys():
            ori_counts_selective[ori] = 0
   
    bar_palette = colorvals.as_hex()
    
    if ax is None:
        fig, ax = pl.subplots()
    
    sns.barplot(sorted(ori_counts_all.keys()), [ori_counts_all[c] for c in sorted(ori_counts_all.keys())], palette=bar_palette, ax=ax)
    ax.tick_params(axis='x', which='both', length=0)

    if show_selective:    
        ax2 = ax.twinx()
        sns.barplot(sorted(ori_counts_all.keys()), [ori_counts_selective[c] for c in sorted(ori_counts_all.keys())], palette=bar_palette, ax=ax2)
        ax2.set_ylim(ax.get_ylim())
        hatch = '//' #itertools.cycle(['/', '//', '+', '-', 'x', '\\', '*', 'o', 'O', '.'])
        for i, bar in enumerate(ax2.patches):
            bar.set_hatch(hatch)
        ax2.set_yticklabels([])
        ax2.tick_params(axis='x', which='both', length=0)
    
        hatch1 = patches.Patch( facecolor='white', hatch=r'\\\\',label='%s > %.2f' % (ori_metric, thresh))
        ax2.legend(handles = [hatch1], loc=1)
        
    sns.despine(trim=True, offset=4)
        
    return ax

#%%
def get_gratings_stats(df_by_rois, sdf, roi_list=[], metric='meanstim'):
    
    '''
    OSI:  
        OSI = (umax - uorth) / (umax + uorth), 
            umax = mean response to preferred orientation
            uorth = mean response to orthogonal orientation (average of both directions)
            
    cvOSI:
        1 - circular variance = sum( R(theta) * exp(2 * i * theta) ) / sum( R(theta) ),
            where R(theta) is avearged repsonse to gratings with orientation theta
        
        This takes into account both tuning width and depth of modulation, "global OSI"
        
    DSI:  
        DSI = (umax - uopp) / (umax + uopp), 
            umax = mean response to preferred direction
            uopp = mean response to opposite direction
              
    '''
    
    if len(roi_list)==0:
        roi_list = df_by_rois.groups.keys()
        
    print "Calculating OSI/DSI for %i rois." % len(roi_list)
    selectivity = {}
    for roi in roi_list:
        
        stim_stats =  df_by_rois.get_group(roi).groupby(['config'])[metric].mean() # Get mean stat across trial reps for each config type 
        if 'rotation' in sdf.columns.tolist():
            rkey = 'rotation'; fkey = 'frequency';
        else:
            rkey = 'ori'; fkey = 'sf';
            
        stim_values = pd.DataFrame({'angle': [sdf[rkey][c] for c in stim_stats.index.tolist()]}, index=stim_stats.index)
        stimdf = pd.concat([stim_stats, stim_values], axis=1)
    
        ordered_configs = stim_stats.sort_values(ascending=False).index
        R_max = stim_stats[ordered_configs[0]]
        #R_least = stim_stats[ordered_configs[-1]]
        
        max_ori = stimdf.loc[ordered_configs[0], 'angle']
        max_ori_opp = (max_ori + 180) % 360. # Same orientation, opposite direction
        max_ori_orthog = (max_ori + 90) % 360. # Orthogonal direction
        max_ori_orthog_opp = (max_ori_orthog + 180) % 360.
        
        best_sf = sdf['sf'][ordered_configs[0]]
        best_sz = sdf['size'][ordered_configs[0]]
        best_sp = sdf['speed'][ordered_configs[0]]
        
        opp_config = sdf[((sdf['sf']==best_sf) & (sdf['size']==best_sz) & (sdf['speed']==best_sp) & (sdf[rkey]==max_ori_opp))].index[0]
        orthog_config = sdf[((sdf['sf']==best_sf) & (sdf['size']==best_sz) & (sdf['speed']==best_sp) & (sdf[rkey]==max_ori_orthog))].index[0]
        orthog_opp_config = sdf[((sdf['sf']==best_sf) & (sdf['size']==best_sz) & (sdf['speed']==best_sp) & (sdf[rkey]==max_ori_orthog_opp))].index[0]
        
        
        #R_pref = stimdf[stimdf['angle'].isin([max_ori, max_ori_opp])][metric].mean()
        R_pref = abs(stimdf[metric][ordered_configs[0]])
        R_opp = abs(stimdf[metric][opp_config])
        R_orthog = abs(stimdf[metric][orthog_config])
        R_orthog_opp = abs(stimdf[metric][orthog_opp_config]) #[stimdf['angle'] == max_ori_opp][metric].values[0]
        
#        if max_ori < 180: #in [0, 45, 90, 135]:
#            max_ori2 = max_ori + 180
#            orth_ori1 = max_ori + 90
#        else: #max_ori in [180, 225, 270, 315]:
#            max_ori2 = max_ori - 180
#            orth_ori1 = max_ori - 90
#        if orth_ori1 < 180:
#            orth_ori2 = orth_ori1 + 180
#        else:
#            orth_ori2 = orth_ori1 - 180
#        
#        R_pref = stimdf[stimdf['angle'].isin([max_ori, max_ori2])][metric].mean()
#        R_orthog = stimdf[stimdf['angle'].isin([orth_ori1, orth_ori2])][metric].mean()
#        R_opp = stimdf[stimdf['angle'] == max_ori2][metric].values[0]
        
        #OSI_use_least = (R_max - R_least) / (R_max + R_least)
        OSI = (R_pref - R_orthog) / (R_pref + np.mean([R_orthog, R_orthog_opp]))

        if any(stimdf['angle'] > 180):
            DSI = (R_pref - R_opp) / (R_pref + R_opp)
        else:
            DSI = None

        # If > 1 SF, use best one:
        sfs = list(set([sdf[fkey][config] for config in sdf.index.tolist()]))
        if len(sfs) > 1:
            sort_config_types = {}
            for sf in sfs:
                sort_config_types[sf] = sorted([config for config in sdf.index.tolist()
                                                    if sdf[fkey][config]==sf],
                                                    key=lambda x: sdf[rkey][x])
            oris = [sdf[rkey][config] for config in sort_config_types[sf]]
            
            orientation_list = sort_config_types[sdf[fkey][ordered_configs[0]]] # take SF at preferred ori
            OSI_cv = np.abs( sum([stim_stats[cfg]*np.exp(2j*theta) for theta, cfg in 
                                  zip(oris, orientation_list)]) / sum([stim_stats[cfg] for cfg in  orientation_list]) )
                    
            DSI_cv = 1 - np.abs( sum([stim_stats[cfg]*np.exp(1j*theta) for theta, cfg in 
                                  zip(oris, orientation_list)]) / sum([stim_stats[cfg] for cfg in  orientation_list]) )
        else:
            OSI_cv = 1 - np.abs( sum([stimdf.loc[cfg, metric]*np.exp(2j*stimdf.loc[cfg, 'angle']) for cfg in 
                                  sdf.index.tolist()]) / sum([stim_stats[cfg] for cfg in sdf.index.tolist()]) )
                    
            DSI_cv = 1 - np.abs( sum([stimdf.loc[cfg, metric]*np.exp(1j*stimdf.loc[cfg, 'angle']) for cfg in 
                                  sdf.index.tolist()]) / sum([stim_stats[cfg] for cfg in sdf.index.tolist()]) )

        selectivity[roi] = {'roi': roi,
                            'roi_label': 'roi%05d' % int(roi+1),
                            'metric': metric,
                            'OSI': OSI,
                            'DSI': DSI,
                            'OSI_cv': OSI_cv,
                            'DSI_cv': DSI_cv,
                            'pref_ori': max_ori}
        
    statsdf = pd.DataFrame(selectivity).T
    
    return statsdf


def get_OSI_DSI(df_by_rois, sconfigs, roi_list=[], metric='meanstim'):
    
    '''
    OSI:  
        OSI = (umax - uorth) / (umax + uorth), 
            umax = mean response to preferred orientation
            uorth = mean response to orthogonal orientation (average of both directions)
            
    cvOSI:
        1 - circular variance = sum( R(theta) * exp(2 * i * theta) ) / sum( R(theta) ),
            where R(theta) is avearged repsonse to gratings with orientation theta
        
        Note: This takes into account both tuning width and depth of modulation.
        
    DSI:  
        DSI = (umax - uopp) / (umax + uopp), 
            umax = mean response to preferred direction
            uopp = mean response to opposite direction
              
    '''
    if len(roi_list)==0:
        roi_list = df_by_rois.groups.keys()
        
    print "Calculating OSI/DSI for %i rois." % len(roi_list)
    selectivity = {}
    for roi in roi_list:
        
        stim_stats =  df_by_rois.get_group(roi).groupby(['config'])[metric].mean()
        if 'rotation' in sconfigs[sconfigs.keys()[0]].keys():
            rkey = 'rotation'; fkey = 'frequency';
        else:
            rkey = 'ori'; fkey = 'sf';
            
        stim_values = pd.DataFrame({'angle': [sconfigs[c][rkey] for c in stim_stats.index.tolist()]}, index=stim_stats.index)
        stimdf = pd.concat([stim_stats, stim_values], axis=1)
    
        ordered_configs = stim_stats.sort_values(ascending=False).index
        R_max = stim_stats[ordered_configs[0]]
        #R_least = stim_stats[ordered_configs[-1]]
        max_ori = stimdf.loc[ordered_configs[0], 'angle']

        if max_ori < 180: #in [0, 45, 90, 135]:
            max_ori2 = max_ori + 180
            orth_ori1 = max_ori + 90
        else: #max_ori in [180, 225, 270, 315]:
            max_ori2 = max_ori - 180
            orth_ori1 = max_ori - 90
        if orth_ori1 < 180:
            orth_ori2 = orth_ori1 + 180
        else:
            orth_ori2 = orth_ori1 - 180
        
        R_pref = stimdf[stimdf['angle'].isin([max_ori, max_ori2])][metric].mean()
        R_orthog = stimdf[stimdf['angle'].isin([orth_ori1, orth_ori2])][metric].mean()
        R_opp = stimdf[stimdf['angle'] == max_ori2][metric].values[0]
        
        #OSI_use_least = (R_max - R_least) / (R_max + R_least)
        OSI = (R_pref - R_orthog) / (R_pref + R_orthog)
        
        if any(stimdf['angle'] > 180):
            DSI = (R_max - R_opp) / (R_max + R_opp)
        else:
            DSI = None

        # If > 1 SF, use best one:
        sfs = list(set([sconfigs[config][fkey] for config in sconfigs.keys()]))
        if len(sfs) > 1:
            sort_config_types = {}
            for sf in sfs:
                sort_config_types[sf] = sorted([config for config in sconfigs.keys()
                                                    if sconfigs[config][fkey]==sf],
                                                    key=lambda x: sconfigs[x][rkey])
            oris = [sconfigs[config][rkey] for config in sort_config_types[sf]]
            orientation_list = sort_config_types[sconfigs[ordered_configs[0]][fkey]]
            OSI_cv = np.abs( sum([stim_stats[cfg]*np.exp(2j*theta) for theta, cfg in 
                                  zip(oris, orientation_list)]) / sum([stim_stats[cfg] for cfg in  orientation_list]) )
        else:
            OSI_cv = np.abs( sum([stimdf.loc[cfg, metric]*np.exp(2j*stimdf.loc[cfg, 'angle']) for cfg in 
                                  sconfigs.keys()]) / sum([stim_stats[cfg] for cfg in sconfigs.keys()]) )

        selectivity[roi] = {'roi': roi,
                            'roi_label': 'roi%05d' % int(roi+1),
                            'metric': metric,
                            'OSI': OSI,
                            'DSI': DSI,
                            'OSI_cv': OSI_cv,
                            'pref_ori': max_ori}

    return selectivity


#%%



def hist_preferred_oris(selectivity, colorvals, metric='meanstim', sort_dir='/tmp',  ax=None,
                        save_and_close=True, thresh=0.33, ori_metric='OSI', show_selective=False):
    
    datestring = datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")
    
    best_ori_vals = [selectivity[r]['pref_ori'] for r in selectivity.keys()]
    best_ori_vals_selective = [selectivity[r]['pref_ori'] for r in selectivity.keys() 
                                    if selectivity[r]['%s' % ori_metric] >= thresh]
    
    ori_counts_all = Counter(best_ori_vals)
    ori_counts_selective = Counter(best_ori_vals_selective)
    for ori in ori_counts_all.keys():
        if ori not in ori_counts_selective.keys():
            ori_counts_selective[ori] = 0
   
    bar_palette = colorvals.as_hex()
    
    if ax is None:
        fig, ax = pl.subplots()
    
    sns.barplot(sorted(ori_counts_all.keys()), [ori_counts_all[c] for c in sorted(ori_counts_all.keys())], palette=bar_palette, ax=ax)
    ax.tick_params(axis='x', which='both', length=0)

    if show_selective:    
        ax2 = ax.twinx()
        sns.barplot(sorted(ori_counts_all.keys()), [ori_counts_selective[c] for c in sorted(ori_counts_all.keys())], palette=bar_palette, ax=ax2)
        ax2.set_ylim(ax.get_ylim())
        hatch = '//' #itertools.cycle(['/', '//', '+', '-', 'x', '\\', '*', 'o', 'O', '.'])
        for i, bar in enumerate(ax2.patches):
            bar.set_hatch(hatch)
        ax2.set_yticklabels([])
        ax2.tick_params(axis='x', which='both', length=0)
    
        hatch1 = patches.Patch( facecolor='white', hatch=r'\\\\',label='%s > %.2f' % (ori_metric, thresh))
        ax2.legend(handles = [hatch1], loc=1)
        
    sns.despine(trim=True, offset=4)
    

    figname = 'counts_per_cond_OSI_%s_%s.png' % (metric, datestring)
    if save_and_close:
        pl.savefig(os.path.join(sort_dir, 'figures', figname))
        pl.close()
    
    # legend:
    #sns.palplot(colorvals)
    if save_and_close:
        pl.savefig(os.path.join(sort_dir, 'figures', 'legend_OSI.png'))
        #pl.close()
    
    return figname


#%%


def get_all_contours(mask_array):
    # Cycle through all ROIs and get their edges
    # (note:  tried doing this on sum of all ROIs, but fails if ROIs are overlapping at all)
    cnts = []
    for ridx in range(mask_array.shape[0]):
        im = mask_array[ridx,:,:]
        im[im>0] = 1
        im[im==0] = np.nan #1
        im = im.astype('uint8')
        edged = cv2.Canny(im, 0, 0.9)
        tmp_cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        tmp_cnts = tmp_cnts[0] if imutils.is_cv2() else tmp_cnts[1]
        cnts.append(tmp_cnts[0])
    print "Created %i contours for rois." % len(cnts)
    
    return cnts


def color_rois_by_OSI(img, maskarray, OSI, colordict, sorted_selective=[], cmap='hls', thickness=1,
                          label_rois=True, labels=[]):
    

    #dims = img.shape
    if len(sorted_selective) == 0:
        sorted_selective = [r for r in OSI.keys() if OSI['OSI_orth'] >= 0.33]
        color_code = 'all'
    else:
        color_code = 'OS'
    

    #label_rois = True
    imgrgb = uint16_to_RGB(img)
    zproj = exposure.equalize_adapthist(imgrgb, clip_limit=0.03)
    zproj *= 256
    zproj= zproj.astype('uint8')
        
    # Plot average img, overlay ROIs with OSI (visual only):
    sns.set_style('white')
    fig, ax = pl.subplots(1, figsize=(10,10))
    ax.imshow(zproj, cmap='gray')
    outimg = zproj.copy()
    #alpha=1.0

    roi_contours = get_all_contours(maskarray) 
    print "MASK ARR:", maskarray.shape
    print len(roi_contours)
    
    # loop over the contours individually
    for ridx in np.arange(0, len(roi_contours)):
        cnt = roi_contours[ridx]
        if not ridx in sorted_selective:
            col = (127, 127, 127)
        else:
            pref_ori = OSI[ridx]['pref_ori'] # TODO:  deal with OSI
            col = colordict[pref_ori]
            
        cv2.drawContours(outimg, cnt, -1, col, thickness)

        # Label ROI
        if label_rois:
            if len(labels) > 0 and ridx in labels:
                label_this = True
            elif ridx in sorted_selective and OSI[ridx]['pref_ori'] >= 0.33:
                label_this = True
            else:
                label_this = False
            if label_this:
                cv2.putText(outimg, str(ridx+1), cv2.boundingRect(cnt)[:2], cv2.FONT_HERSHEY_COMPLEX, .5, [0])
        ax.imshow(outimg, alpha=1.0)
    
    pl.axis('off')
 
    return fig

#%%
def overlay_osi_rois(statsdf, roi_dir, conditions, cmap='hls', output_dir='/tmp'):

    #roi_dir = '/n/coxfs01/2p-data/JC084/20190522/ROIs/rois001_efa39d'
    
    fov_fpath = os.path.join(roi_dir, 'warped_max_reference.tif')
    fov_img = tf.imread(fov_fpath)
    
    
    mask_fpath = os.path.join(roi_dir, 'masks.hdf5')
    mfile = h5py.File(mask_fpath, 'r')
    
    ref_img = mfile[mfile.keys()[0]]['zproj_img']['Slice01'][:].astype('uint16')
    masks =  mfile[mfile.keys()[0]]['masks']['Slice01'][:]
    print(masks.shape)
    
    
    #
    #clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
    #cl1 = clahe.apply(ref_img)
    #
    #pl.figure()
    #ax.imshow(cl1, cmap='gray')
    #
    #from matplotlib.colors import LinearSegmentedColormap
    #cm = LinearSegmentedColormap.from_list('hls', colorvals, N=len(oris))
    #
    #
    #fig, ax = pl.subplots()
    #ax.imshow(cl1, cmap='gray')
    #for rid in roi_list:
    #    ori = selectivity[rid]['pref_ori']
    #    rmask = masks[rid, :, :] * ori
    #    
    #    msk = np.ma.masked_where(rmask==0, rmask)
        #    ax.imshow(msk, cmap=cm)
    
    #%
    colorvals = sns.color_palette(cmap, len(conditions)) # len(gratings_sconfigs))
    colordict = dict((ori, tuple([ci*255 for ci in cval])) for ori, cval in zip(conditions, colorvals))
    
    fig = color_rois_by_OSI(ref_img, masks, statsdf, colordict, sorted_selective=statsdf.index.tolist(), label_rois=True)
    label_figure(fig, data_identifier)
    
    figname = 'rois_by_ori_labeled'
    pl.savefig(os.path.join(output_dir, '%s.png' % figname))
    pl.close()
    
    fig = color_rois_by_OSI(ref_img, masks, statsdf, colordict, sorted_selective=statsdf.index.tolist(), label_rois=False, thickness=2)
    label_figure(fig, data_identifier)
    
    figname = 'rois_by_ori'
    pl.savefig(os.path.join(output_dir, '%s.png' % figname))

#%%
def get_roi_resp_by_condition(raw_traces, labels, metric_type='snr', nframes_post_onset=45):
    
    zscored_traces, zscores, snrs = rf.zscore_traces(raw_traces, labels, nframes_post_onset=nframes_post_onset)
    
    #metric_type = 'snr'
    #response_thr = 1.5
    
    trials_by_cond = rf.get_trials_by_cond(labels)
    if metric_type == 'zscore':
        zscores_by_cond = rf.group_zscores_by_cond(zscores, trials_by_cond)
    elif metric_type == 'snr':
        zscores_by_cond = rf.group_zscores_by_cond(snrs, trials_by_cond)
    
    # Sort ROIs by zscore by cond
    # -----------------------------------------------------------------------------
    avg_resp_by_cond = pd.DataFrame([zscores_by_cond[cfg].mean(axis=0) \
                                        for cfg in sorted(zscores_by_cond.keys(), key=natural_keys)]) # nconfigs x nrois
    
    return avg_resp_by_cond

#%%
def calculate_gratings_stats(animalid, session, fov, run, traceid, 
                             metric_type='snr', metric_thr=1.2,
                             response_type='dff',
                             rootdir='/n/coxfs01/2p-data', create_new=False, 
                             n_processes=1):
    #%%
    traceid_dir =  glob.glob(os.path.join(rootdir, animalid, session, fov, run, 'traces', '%s*' % traceid))[0]
    data_fpath = glob.glob(os.path.join(traceid_dir, 'data_arrays', 'datasets.npz'))[0]
    dset = np.load(data_fpath)
    
    data_identifier = '|'.join([animalid, session, fov, run, traceid])

    
    raw_traces = pd.DataFrame(dset['corrected'])
    labels = pd.DataFrame(data=dset['labels_data'], columns=dset['labels_columns'])

    
    fr = 44.65 #dset['run_info'][()]['framerate']
    nframes_per_trial = int(dset['run_info'][()]['nframes_per_trial'][0])
    nframes_on = labels['nframes_on'].unique()[0]
    stim_on_frame = labels['stim_on_frame'].unique()[0]
    
    # zscore the traces:
    nframes_post_onset = nframes_on + int(round(1.*fr))

    trials_by_cond = rf.get_trials_by_cond(labels)

    sdf = pd.DataFrame(dset['sconfigs'][()]).T

    
    #%%

#    avg_resp_by_condition = get_roi_resp_by_condition(raw_traces, labels, 
#                                                      metric_type=metric_type, 
#                                                      nframes_post_onset=nframes_post_onset)
#
#    roi_list = [r for r in avg_resp_by_condition.columns.tolist() if avg_resp_by_condition[r].max() >= response_thr]
#    print("%i out of %i cells meet min req. of %.2f" % (len(roi_list), avg_resp_by_condition.shape[1], response_thr))
    
    
        
    #%%
    
    gdf = resp.group_roidata_stimresponse(raw_traces.values, labels) # Each group is roi's trials x metrics
    #gratings_df_by_rois.get_group(roi_list[0])
    nrois_total = len(gdf.groups)
    
    #%%
    metric_type = 'dff'
    metric_thr = 0.5
    goodness_type = ['zscore', 'snr']
    goodness_thr = [0.5, 1.5]

    
    roi_list = [k for k, g in gdf if g.groupby(['config']).mean()[metric_type].max() >= metric_thr\
                and g.groupby(['config']).mean()[goodness_type[0]].max() >= goodness_thr[0]\
                and g.groupby(['config']).mean()[goodness_type[1]].max() >= goodness_thr[1]]
    print("%i out of %i cells meet min %s req. of %.2f" % (len(roi_list), nrois_total, metric_type, metric_thr))

    goodness_str = '_'.join(['%s%.2f' % (gt, gthr) for gt, gthr in zip(goodness_type, goodness_thr)])
    
    fit_str = 'responsemin_%s%.2f_goodness_%s' % (metric_type, metric_thr, goodness_str)
    
    
    #%%
    plot_interpolate = True
    make_plots =  True
    response_type = 'zscore'
    fit_color = 'b'
    
    #roi_fitdir = os.path.join(traceid_dir, 'figures', 'fits', 'tuning_by_roi_%s' % response_type)
    roi_fitdir = os.path.join(traceid_dir, 'figures', 'tuning', 'fit_%s_%s' % (response_type, fit_str))

    if not os.path.exists(roi_fitdir):
        os.makedirs(roi_fitdir)
    print("Saving roi fits to: %s" % roi_fitdir)
    
    
    osi_results_fpath = os.path.join(roi_fitdir, 'roistats.pkl') #% osi_dsi_str
    if not os.path.exists(osi_results_fpath) or create_new is True:
        do_fits = True
        
    else:
        do_fits = False
        print("Loading existing results...")
        with open(osi_results_fpath, 'rb') as f:
            statsdf = pkl.load(f)    
    
    
    #%%
    
#    roi = 17
#    g = gdf.get_group(roi)
#             
#    quants = [c for c in g.columns if g[c].dtype==np.float64]
#    g_df = g[quants]
#    g_std = (g_df - g_df.mean()) / g_df.std()
#    
##    
##    pl.figure()
##    for c in quants:
##        print c
##        sns.distplot(g_std[c].values)
#        
#    sns.pairplot(g_std)
#
#
#    lgroups = labels.groupby(['config', 'trial'])
#    for (config, trial), trial_ixs in config_groups:
#        raw_traces[roi]


    #%%
    
    if do_fits:
            
        from scipy import stats
    
        #
        '''
        Mazurek, M., Kager, M., & Van Hooser, S. D. (2014). Robust quantification 
        of orientation selectivity and direction selectivity. Frontiers in neural 
        circuits, 8, 92. doi:10.3389/fncir.2014.00092
        '''
    
    
        oris = sorted(sdf['ori'].unique())
    
        if plot_interpolate:
            new_length = 20 
            oris_interp = []
            for orix, ori in enumerate(oris[0:-1]):
                if ori == oris[-2]:
                    oris_interp.extend(np.linspace(ori, oris[orix+1], endpoint=True, num=new_length+1))
                else:
                    oris_interp.extend(np.linspace(ori, oris[orix+1], endpoint=False, num=new_length))          
        
        dir_fit_color = 'cornflowerblue'
        ori_fit_color = 'red'
        
        # Fit each roi's response, and plot:
        roi_figdir = os.path.join(roi_fitdir, 'roi_fits')
        if not os.path.exists(roi_figdir):
            os.makedirs(roi_figdir)
        
        fit_results = {}
        all_metrics_list = []
        #roi = 4 #5
        
        for roi in roi_list:
                #%
            g = gdf.get_group(roi)
            
    #        if g[response_type].min() < 0:
    #            g_offset = g[response_type] - g[response_type].min()
    #            g[response_type] = g_offset
                
            mean_responses = g.groupby(['config']).mean()[response_type]
            sem_responses = g.groupby(['config']).sem()[response_type]
            sorted_config_ixs = mean_responses.values.argsort()[::-1]
            sorted_configs = [mean_responses.index[s] for s in sorted_config_ixs]
            
            constant_params = ['aspect', 'luminance', 'position', 'stimtype']
            params = [c for c in sdf.columns if c not in constant_params]
            stimdf = sdf[params]
            
            best_cfg = sorted_configs[0]
            best_cfg_params = stimdf.loc[best_cfg][[p for p in params if p!='ori']]
            curr_cfgs = sorted([c for c in stimdf.index.tolist() \
                                if all(stimdf.loc[c][[p for p in params if p!='ori']] == best_cfg_params)],\
                                key = lambda x: stimdf['ori'][x])
            
                
            fig = pl.figure(figsize=(12,8))
    
            # ---------------------------------------------------------------------
            #% plot raw traces:
            cfg_groups = labels[labels['config'].isin(curr_cfgs)].groupby(['config'])
            mean_traces = np.array([np.nanmean(np.array([raw_traces[roi][trial_df.index]\
                                                         for cfg_rep, trial_df in config_df.groupby(['trial'])]), axis=0) \
                                    for cfg, config_df in sorted(cfg_groups, key=lambda x: stimdf['ori'][x[0]])])
            std_traces = np.array([stats.sem(np.array([raw_traces[roi][trial_df.index]\
                                                       for cfg_rep, trial_df in config_df.groupby(['trial'])]), axis=0, nan_policy='omit') \
                                    for cfg, config_df in sorted(cfg_groups, key=lambda x: stimdf['ori'][x[0]])])
            tpoints = np.array([np.array([trial_df['tsec'] for cfg_rep, trial_df in config_df.groupby(['trial'])]).mean(axis=0) \
                                    for cfg, config_df in sorted(cfg_groups, key=lambda x: stimdf['ori'][x[0]])]).mean(axis=0).astype(float)
                
            ymin = (mean_traces - std_traces ).min()
            ymax = (mean_traces + std_traces ).max()
            
            for icfg in range(len(curr_cfgs)):
                ax = pl.subplot2grid((2, 8), (0, icfg), colspan=1) #pl.subplots(1, 3) #pl.figure()
                ax.plot(tpoints, mean_traces[icfg, :], color='k')
                ax.set_xticks([tpoints[stim_on_frame], round(tpoints[stim_on_frame+nframes_on], 1)])
                ax.set_xticklabels(['', round(tpoints[stim_on_frame+nframes_on], 1)])
                ax.set_ylim([ymin, ymax])
                if icfg > 0:
                    ax.set_yticks([])
                    ax.set_yticklabels([])
                    ax.set_xticks([])
                    ax.set_xticklabels([])
                    sns.despine(ax=ax, offset=4, trim=True, left=True, bottom=True)
                else:
                    ax.set_ylabel('intensity')
                    ax.set_xlabel('time (s)')
                    sns.despine(ax=ax, offset=4, trim=True)
                sem_plus = np.array(mean_traces[icfg,:]) + np.array(std_traces[icfg,:])
                sem_minus = np.array(mean_traces[icfg,:]) - np.array(std_traces[icfg,:])
                ax.fill_between(tpoints, sem_plus, y2=sem_minus, alpha=0.5, color='k')
            # ---------------------------------------------------------------------
                
            
            theta_pref = stimdf['ori'][best_cfg]
            theta_null = (stimdf['ori'][best_cfg] + 180) % 360    
            null_cfg = [c for c in curr_cfgs if stimdf['ori'][c]==theta_null]
            
            curr_resps = np.array([mean_responses[s] for s in curr_cfgs])
            curr_sems = np.array([sem_responses[s] for s in curr_cfgs])
            if curr_resps.min() < 0:
                offset = curr_resps.min()
                curr_resps = curr_resps - offset
                curr_sems = curr_sems - offset
            else:
                offset = 0 
                
            curr_oris = np.array([stimdf['ori'][c] for c in curr_cfgs])        
    
            # Least squares fit. Starting values found by inspection.
            r_max = mean_responses[curr_cfgs].max()
            r_pref = mean_responses.loc[best_cfg] + offset
            r_null = float(mean_responses.loc[null_cfg]) + offset
            sigma = np.mean(np.diff(curr_oris)) #/ 2.
            C_offset = np.mean([mean_responses.loc[c] for c in curr_cfgs if c not in [best_cfg, null_cfg]]) + offset #r_pref*-1 #0 #mean_responses.min()
            #init_params = [r_pref, r_null, theta_pref, sigma, C_offset]
                
            x = curr_oris.copy()        
            y = curr_resps.copy()
            try:
                
                # Fit for direction:
                init_params_dsi = [r_pref, r_null, theta_pref, sigma, C_offset]
                init_bounds = ([0, 0, -np.inf, sigma/2., -r_max], [3*r_max, 3*r_max, np.inf, np.inf, r_max])
                roi_fit_dir = fit_direction_selectivity(x, y, init_params_dsi, bounds=init_bounds)
                assert roi_fit_dir is not None
                roi_fit_dir['configs'] = curr_cfgs
                roi_fit_dir['mean_responses'] = curr_resps
                roi_fit_dir['offset'] = offset
                roi_fit_dir['oris'] = curr_oris
                
            except Exception as e:
                print("-- roi %i: no fit." % roi)
                roi_fit_dir = None
                
            if make_plots:
                # Plot data:
                #fig = pl.figure(figsize=(9,4)) 
                #ax = pl.subplot2grid((1,3), (0, 0), colspan=2) #pl.subplots(1, 3) #pl.figure()
                ax = pl.subplot2grid((2, 8), (1, 0), colspan=5)
                ax.plot(curr_oris, curr_resps, 'ko', markersize=5, lw=0)
                ax.errorbar(curr_oris, curr_resps, yerr=curr_sems, fmt='none', ecolor='k')
                ax.set_xticks(curr_oris)
                ax.set_xticklabels(curr_oris)
                ax.set_ylabel(response_type)
                ax.set_title('(sz %i, sf %.2f)' % (best_cfg_params['size'], best_cfg_params['sf']), fontsize=8)
                sns.despine(trim=True, offset=4, ax=ax)
                
                # Plot fit:
                if roi_fit_dir is not None and roi_fit_dir['success']:
                    if plot_interpolate:
                        # Interpolate the data using a cubic spline to "new_length" samples      
                        x_plot = np.array(oris_interp).copy()   
                        y_plot = double_gaussian( x_plot, *roi_fit_dir['popt'])
                    else:
                        x_plot = x.copy()
                        y_plot = roi_fit_dir['fit_y'].copy()
                    ax.plot(x_plot, y_plot, c=dir_fit_color, label='dir (r2=%.2f)' % roi_fit_dir['r2'])
                    ax.text(0, ax.get_ylim()[-1]*0.75, 'r2=%.2f' % roi_fit_dir['r2'], fontsize=6)
                else:            
                    ax.text(0, ax.get_ylim()[-1]*0.75, 'no fit', fontsize=6)
                
            
                # Plot polar graph:
                #ax = pl.subplot2grid((1,3), (0,2), colspan=1, polar=True)
                ax = pl.subplot2grid((2,8), (1,6), colspan=2, polar=True)
                thetas = np.array([np.deg2rad(c) for c in curr_oris])
                radii = curr_resps.copy()
                thetas = np.append(thetas, np.deg2rad(curr_oris[0]))  # append first value so plot line connects back to start
                radii = np.append(radii, curr_resps[0]) # append first value so plot line connects back to start
                ax.plot(thetas, radii, 'k-')
                ax.set_theta_zero_location("N")
                ax.set_yticks([curr_resps.min(), curr_resps.max()])
                ax.set_yticklabels(['', round(curr_resps.max(), 1)])
                
                if roi_fit_dir is not None and roi_fit_dir['success']:
                    if plot_interpolate:
                        x_plot = np.array(oris_interp).copy()   
                    else:
                        x_plot = x.copy()
                    x_plot_polar = np.append(x_plot, x_plot[0])
                    radii = double_gaussian( x_plot_polar, *roi_fit_dir['popt'])
                    thetas = np.array([np.deg2rad(c) for c in x_plot_polar])
                    ax.plot(thetas, radii, color=dir_fit_color)
    
                    #%
                    
                pl.subplots_adjust(top=0.8, hspace=0.5)
                fig.suptitle('roi %i' % int(roi+1))
                label_figure(fig, data_identifier)
                figname = 'roi%05d_fits' % int(roi+1)
                pl.savefig(os.path.join(roi_figdir, '%s.png' % figname))
                pl.close()
        
            if roi_fit_dir is not None:
                fit_results[roi] = roi_fit_dir
        
            roi_all_metrics = g.groupby(['config']).mean().loc[curr_cfgs]
            roi_all_metrics['roi'] = [roi for _ in range(len(curr_oris))]
            
            all_metrics_list.append(roi_all_metrics)
        
        print("--- FITS complete ---")
        #print("%i out of %i responsive cells (%s, thr: %.2f) appear orientation selective" % (len(fit_results), len(roi_list), metric_type, response_thr))
            
        with open(osi_results_fpath, 'wb') as f:
            pkl.dump(fit_results, f, protocol=pkl.HIGHEST_PROTOCOL)
            
                
        #%%
        
        all_metrics = pd.concat(all_metrics_list, axis=0)
        
        g = sns.pairplot(all_metrics[[s for s in all_metrics.columns if s != 'roi']])
        
        #sns.distplot([g.groupby(['config']).mean()[metric_type].max() for k, g in gdf], ax=ax, kde=False, norm_hist=True)
        #ax.set_title('%s (thr: %.3f)' % (metric_type, response_thr))
        #rstr = "%i out of %i cells meet min %s req. of %.2f" % (len(roi_list), nrois_total, response_thr)
        pl.subplots_adjust(top=0.9)
        g.fig.suptitle(fit_str, fontsize=8)
        
        label_figure(g.fig, data_identifier)
        
        
        pl.savefig(os.path.join(roi_fitdir, 'compare_all_metrics.png'))
        pl.close()
    
        #%%
        
        # Get gratings descriptives:
        
        #    OSI:  
        #        OSI = (umax - uorth) / (umax + uorth), 
        #            umax = mean response to preferred orientation
        #            uorth = mean response to orthogonal orientation (average of both directions)
        #            
        #    cvOSI:
        #        1 - circular variance = sum( R(theta) * exp(2 * i * theta) ) / sum( R(theta) ),
        #            where R(theta) is avearged repsonse to gratings with orientation theta
        #        
        #        Note: This takes into account both tuning width and depth of modulation.
        #        
        #    DSI:  
        #        DSI = (umax - uopp) / (umax + uopp), 
        #            umax = mean response to preferred direction
        #            uopp = mean response to opposite direction
        #              
        
        
        fit_thr = 0.9
        fit_roi_list = [r for r, results in fit_results.items() if results['r2'] >= fit_thr]
        print("%i out of %i responsive cells fit for osi/dsi." % (len(fit_roi_list), len(roi_list)))
        
        DSIs=[]; OSIs=[]; OSI_cvs=[]; DSI_cvs=[]; pref_oris=[]; curr_roi_list=[]; r2_values=[]
        for roi, fresults in fit_results.items():
            [r_pref_fit, r_null_fit, theta_pref, sigma, C_offset] = fresults['popt']
            
            mean_response = fresults['fit_y'] #fit_results[roi]['mean_responses']
            
            theta_prefE = int(curr_oris[np.where(np.abs(curr_oris - theta_pref) == np.min(np.abs(curr_oris - theta_pref)))])
            if r_pref_fit < r_null_fit:
                theta_prefE = (theta_prefE - 180) % 360. #list(curr_oris).index((theta_prefE - 180)%360.)
            theta_null = (theta_prefE - 180) % 360.
            theta_orthog = (theta_prefE - 90) % 360.
            theta_orthog_opp = (theta_orthog - 180) % 360.
            
            r_pref = max([mean_response[oris.index(theta_prefE)], mean_response[oris.index(theta_null)]])
            r_null = min([mean_response[oris.index(theta_prefE)], mean_response[oris.index(theta_null)]])
            
            #r_null = mean_response[oris.index(theta_null)]
            r_orthog = np.abs(mean_response[oris.index(theta_orthog)])
            r_orthog_opp = np.abs( mean_response[oris.index(theta_orthog_opp)] )
            r_orth = np.mean([r_orthog, r_orthog_opp])
            
            rDSI = (r_pref - r_null) / (r_pref + r_null)
            rOSI = (r_pref - r_orth) / (r_pref + r_orth)
            
            curr_resps =  mean_response #fit_results[roi]['mean_responses']
            curr_oris =  fresults['oris']
            rOSI_cv = 1 - np.abs( np.sum( [r_theta * np.exp(2j*theta) for r_theta, theta in zip(curr_resps, curr_oris)] ) / np.sum(curr_resps) )
            rDSI_cv = 1 - np.abs( np.sum( [r_theta * np.exp(1j*theta) for r_theta, theta in zip(curr_resps, curr_oris)] ) / np.sum(curr_resps) )
            
            DSIs.append(rDSI)
            OSIs.append(rOSI)
            OSI_cvs.append(rOSI_cv)
            DSI_cvs.append(rDSI_cv)
            pref_oris.append(theta_prefE)
            curr_roi_list.append(roi)
            r2_values.append(fresults['r2'])
            
            
        statsdf = pd.DataFrame({'OSI': OSIs,
                                'DSI': DSIs,
                                'OSI_cv': OSI_cvs,
                                'DSI_cv': DSI_cvs,
                                'pref_ori': pref_oris,
                                'roi': curr_roi_list,
                                'r2': r2_values})
                
        
        
        #%%
    #    
    #    #metric_osi = 'dff'
    #    
    #    statsdf = get_gratings_stats(gdf, sdf, roi_list=roi_list, metric=metric_osi)
    #    
    #    bad_cells = []
    #    bad_cells = statsdf[( (statsdf['OSI'] > 1) | (statsdf['OSI'] < 0)\
    #                         | (statsdf['DSI'] > 1) | (statsdf['DSI'] < 0)\
    #                         | (statsdf['OSI_cv'] > 1) | (statsdf['OSI_cv'] < 0)\
    #                         | (statsdf['DSI_cv'] > 1) | (statsdf['DSI_cv'] < 0) )]['roi'].values
    #    print("--- removing %i cells with bad OSI/DSI" % len(bad_cells))
    #    roi_list = [r for r in roi_list if r not in bad_cells]
    #    statsdf = get_gratings_stats(gdf, sdf, roi_list=roi_list, metric=metric_osi)
    #    print("%i out of %i cells meet min req. of %.2f" % (len(roi_list), nrois_total, response_thr))
    #
    #    
    #    #%% Set output dir
    #    
    #    osi_dsi_str = 'osi_dsi_responsemin_%s%.2f_osimetric_%s' % (metric_type, response_thr, metric_osi)    
    #    output_dir = os.path.join(traceid_dir, 'figures', 'population', osi_dsi_str)
    #    if not os.path.exists(output_dir):
    #        os.makedirs(output_dir)
    #    print("Saving figures to: %s" % output_dir)
    #    
    #    #%%
    #    fig, ax = pl.subplots() #pl.figure()
    #    sns.distplot([g.groupby(['config']).mean()[metric_type].max() for k, g in gdf], ax=ax, kde=False, norm_hist=True)
    #    ax.set_title('%s (thr: %.3f)' % (metric_type, response_thr))
    #    rstr = "%i out of %i cells meet min req. of %.2f" % (len(roi_list), nrois_total, response_thr)
    #    ax.text(0, 0, rstr)
    #    label_figure(fig, data_identifier)
    #    pl.savefig(os.path.join(output_dir, 'hist_values_%s_%.3f.png' % (metric_type, response_thr)))
    #    pl.close()
        
        #%% Plot some figures
        
        osi_results_fpath = os.path.join(roi_fitdir, 'roistats.pkl') #% osi_dsi_str
        
        with open(osi_results_fpath, 'wb') as f:
            pkl.dump(statsdf, f, protocol=pkl.HIGHEST_PROTOCOL)
        print("Saved roi gratings stats.")
        
        
        #%%
        
        curr_color = 'cornflowerblue'
        non_quant = ['roi', 'pref_ori', 'r2']
        quant_stats = [r for r in statsdf.columns if r not in non_quant]
        
        df = statsdf[quant_stats]
        sns.set(style='ticks')
        
        # Plot all:
        g = sns.PairGrid(df, aspect=1)
        g = g.map_diag(sns.distplot, kde=False, hist=True, rug=True,\
                       hist_kws={"histtype": "step", "linewidth": 2, "color": curr_color, "alpha": 1.0})
        g = g.map_offdiag(pl.scatter, marker='+')
        
        g.set(xlim=(0,1), ylim=(0,1))
        g.set(xticks=[0, 1])
        g.set(yticks=[0, 1])
        sns.despine(trim=True)
        
        cleanup_axes(g.axes[:, 1:].flat, which_axis='y')
        cleanup_axes( g.axes[:-1, :].flat, which_axis='x')
            
        pl.subplots_adjust(top=0.9) #)
        g.fig.suptitle('%s (all)' % (response_type))
    
        label_figure(g.fig, data_identifier)
        
        figname = 'distN_osi_dsi_circvar_all'
        pl.savefig(os.path.join(roi_fitdir, '%s.png' % figname))
        pl.close()
        
        #%% Plot thresholded:
        fit_thr = 0.9
        
        cmap = 'hls'
        noris = len(curr_oris)
        
        
        
        fig, ax = pl.subplots()
        colorvals = sns.color_palette(cmap, noris) # len(gratings_sconfigs))
        if statsdf.shape[0] > 0:
            hist_gratings_stats(statsdf, colorvals, ax=ax) 
        ax.set_title('preferred orientation', fontsize=18)
        ax.set_ylabel('counts')
        
        label_figure(fig, data_identifier)
        figname = 'hist_dir_selectivity_all' 
        pl.savefig(os.path.join(roi_fitdir, '%s.png' % figname))
        pl.close()
        
        
        fig, ax = pl.subplots()
        colorvals = sns.color_palette(cmap, noris) # len(gratings_sconfigs))
        if statsdf.shape[0] > 0:
            hist_gratings_stats(statsdf[statsdf['r2']>=fit_thr], colorvals, ax=ax) 
        ax.set_title('preferred orientation', fontsize=18)
        ax.set_ylabel('counts')
        
        label_figure(fig, data_identifier)
        figname = 'hist_dir_selectivity_fit_thr_%.2f' % fit_thr
        pl.savefig(os.path.join(roi_fitdir, '%s.png' % figname))
        pl.close()
        
        #%
        ddf = statsdf[statsdf['r2'] >= fit_thr][quant_stats]
        g = sns.PairGrid(ddf, aspect=1)
        g = g.map_diag(sns.distplot, kde=False, hist=True, rug=True,\
                       hist_kws={"histtype": "step", "linewidth": 2, "color": curr_color, "alpha": 1.0})
        g = g.map_offdiag(pl.scatter, marker='+')
        
        g.set(xlim=(0,1), ylim=(0,1))
        g.set(xticks=[0, 1])
        g.set(yticks=[0, 1])
        sns.despine(trim=True)
        
        cleanup_axes(g.axes[:, 1:].flat, which_axis='y')
        cleanup_axes( g.axes[:-1, :].flat, which_axis='x')
            
        pl.subplots_adjust(top=0.9) #)
        g.fig.suptitle('%s (fit thr: %.2f)' % (response_type, fit_thr))
    
        label_figure(g.fig, data_identifier)
        
        figname = 'distN_osi_dsi_circvar_fit_thr_%.2f' % fit_thr
        pl.savefig(os.path.join(roi_fitdir, '%s.png' % figname))
        pl.close()
        
        #%%
        plot_overlay = False
        if plot_overlay:
            roi_dir = '/n/coxfs01/2p-data/JC084/20190522/ROIs/rois001_efa39d'
            overlay_osi_rois(statsdf, roi_dir, oris, cmap=cmap, output_dir=roi_fitdir)
            
            
            #%%
    return statsdf

#%%
def extract_options(options):
    
    parser = optparse.OptionParser()

    parser.add_option('-D', '--root', action='store', dest='rootdir', default='/n/coxfs01/2p-data',\
                      help='data root dir (root project dir containing all animalids) [default: /n/coxfs01/2pdata]')
    parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')

    # Set specific session/run for current animal:
    parser.add_option('-S', '--session', action='store', dest='session', default='', \
                      help='session dir (format: YYYMMDD_ANIMALID')
    parser.add_option('-A', '--fov', action='store', dest='fov', default='FOV1_zoom2p0x', \
                      help="acquisition folder (ex: 'FOV1_zoom3x') [default: FOV1]")
    parser.add_option('-R', '--run', action='store', dest='run', default='retino_run1', \
                      help="name of run dir containing tiffs to be processed (ex: gratings_phasemod_run1)")
    parser.add_option('-t', '--traceid', action='store', dest='traceid', default='traces001', \
                      help="name of traces ID [default: traces001]")
    
    parser.add_option('-d', '--data-type', action='store', dest='trace_type', default='corrected', \
                      help="Trace type to use for analysis [default: corrected]")
    
    parser.add_option('--new', action='store_true', dest='create_new', default=False, \
                      help="Set flag to calculate roi stats anew.")
    parser.add_option('-V', '--area', action='store', dest='visual_area', default='', \
                      help="Name of visual area, if --segment is flagged")
    
    parser.add_option('-T', '--thr', action='store', dest='response_thr', default=1.5, \
                      help="Min snr or zscore value for cells to fit (default: 1.5)")
    parser.add_option('-M', '--metric', action='store', dest='metric_type', default='snr', \
                      help="Metric to use for creating RF maps (default: snr)")

    parser.add_option('-o', '--osi-metric', action='store', dest='metric_osi', default='dff', \
                      help="Metric to use for caluclating OSI/DSI stats (default: dff)")
    
    parser.add_option('-n', '--nproc', action='store', dest='n_processes', default=1, \
                      help="N processes to use (default: 1)")
    
    (options, args) = parser.parse_args(options)

    return options

#%%
rootdir = '/n/coxfs01/2p-data'
animalid = 'JC080' 
session = '20190603' #'20190319'
fov = 'FOV1_zoom2p0x' 
run = 'combined_gratings_static'
traceid = 'traces001' #'traces002'

data_identifier = '|'.join([animalid, session, fov, run, traceid])

create_new=True
n_processes=1


#%%
def main(options):
    optsE = extract_options(options)
    rootdir = optsE.rootdir
    animalid = optsE.animalid
    session = optsE.session
    fov = optsE.fov
    run = optsE.run
    traceid = optsE.traceid
    metric_type = optsE.metric_type
    metric_osi = optsE.metric_osi
    response_thr = optsE.response_thr
    
    create_new = optsE.create_new
    n_processes = int(optsE.n_processes)
    
    statsdf = calculate_gratings_stats(animalid, session, fov, run, traceid, 
                                 metric_type=metric_type, response_thr=response_thr,
                                 metric_osi=metric_osi,
                                 rootdir=rootdir, create_new=create_new, 
                                 n_processes=n_processes)
    

    print "((( RFs done! )))))"
        
        
if __name__ == '__main__':
    main(sys.argv[1:])

