#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 11:49:39 2018

@author: juliana
"""
import numpy as np
import pandas as pd
import datetime
import os
import pylab as pl
from collections import Counter
import seaborn as sns

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

from matplotlib import patches
    
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