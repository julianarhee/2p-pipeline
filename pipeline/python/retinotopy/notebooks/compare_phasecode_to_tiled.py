#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 14:36:54 2018

@author: juliana
"""
import glob
import os
import json
import pandas as pd
import numpy as np
import pylab as pl
import seaborn as sns
import cPickle as pkl

from pipeline.python.utils import label_figure

def intersect(a, b):
    """ return the intersection of two lists """
    return list(set(a) & set(b))

def union(a, b):
    """ return the union of two lists """
    return list(set(a) | set(b))

def rescale_values(values, newmin=0, newmax=200):
    old_range = max(values) - min(values)  
    new_range = newmax - newmin
    new_values = [( ((val - min(values)) * new_range) / old_range) + newmin for val in values]
    return new_values


#%%
# Combine data from multiple acquisitions for a given visual area:

rootdir = '/n/coxfs01/2p-data'

# Combine different conditions of the SAME acquisition:
animalid = 'JC022'
session = '20181005'
acquisition = 'FOV2_zoom2p7x'

# Load RF estimate results for TILING data:
# -----------------------------------------------------------------------------

LM_1 = [animalid, session, acquisition, 'combined_gratings_static', 'traces003']
LM_2 = [animalid, session, acquisition, 'combined_objects_static', 'traces003']

fov_lists = [LM_1, LM_2]
fov_traceid_dirs = []
for fov_info in fov_lists:
    traceid_dirs = glob.glob(os.path.join(rootdir, '/'.join(fov_info[0:-1]), 'traces', '%s*' % fov_info[-1]))
    if len(traceid_dirs)==0:
        print "Specified src not found:", str(fov_info)
    elif len(traceid_dirs) > 1:
        print "Multiple possible sources:"
        for di, df in enumerate(traceid_dirs):
            print di, df
            select = input('Select IDX of traceid dir to use: ')
            traceid_dir = traceid_dirs[di]
    else:
        traceid_dir = traceid_dirs[0]
    
    fov_traceid_dirs.append(traceid_dir)
    
RF = {}
for traceid_dir in fov_traceid_dirs:
    data_fpath = glob.glob(os.path.join(traceid_dir, 'rf_estimates', 'rf_results*.json'))[0]
    curr_run = os.path.split(traceid_dir.split('/traces')[0])[-1]
    with open(data_fpath, 'r') as f:
        RF[curr_run] = json.load(f)
    print "Loaded RF results for run: %s" % curr_run

# Let's just look at each result first:
fit_thr = 0.5
good_rois_1 = [roi for roi, results in RF[RF.keys()[0]].items() 
                if results['results']['r2'] > fit_thr 
                and 0 < results['results']['width_x'] < 150 
                and 0 < results['results']['width_y'] < 150 
                and -50 < results['results']['peak_x'] < 50 
                and -40 < results['results']['peak_y'] < 40 
                and results['results']['amplitude'] > 0]
 
good_rois_2 = [roi for roi, results in RF[RF.keys()[1]].items() 
                if results['results']['r2'] > fit_thr 
                and 0 < results['results']['width_x'] < 150 
                and 0 < results['results']['width_y'] < 150 
                and -50 < results['results']['peak_x'] < 50 
                and -40 < results['results']['peak_y'] < 40 
                and results['results']['amplitude'] > 0]

widths_1 = [(RF[RF.keys()[0]][roi]['results']['width_x'], RF[RF.keys()[0]][roi]['results']['width_y']) for roi in good_rois_1]
widths_2 = [(RF[RF.keys()[1]][roi]['results']['width_x'], RF[RF.keys()[1]][roi]['results']['width_y']) for roi in good_rois_2]


# Look at retino info
# -----------------------------------------------------------------------------
run1_id = '%s_%s' % (LM_1[-2], LM_1[-1])
run2_id = '%s_%s' % (LM_2[-2], LM_2[-1])

ss_path = glob.glob(os.path.join(rootdir, animalid, session, acquisition, 'session_summary_*%s_retino*_%s*_%s*.pkl' % (acquisition, run1_id, run2_id)))[0]
with open(ss_path, 'rb') as f:
    S = pkl.load(f)


nrois_total = len(S.retinotopy['data'])
all_visual_rois = [int(i) for i in union(RF[RF.keys()[0]].keys(), RF[RF.keys()[1]].keys())]
print "---- %i out of total %i rois were responsive to objects or gratings." % (len(all_visual_rois), nrois_total)

retino_good_rois = [roi for roi in all_visual_rois if all([S.retinotopy['data'][roi].conditions[ci].fit_results['r2'] >= fit_thr for ci in range(2)])]
print "---- %i out of %i visual rois had gaussian fit for az/el (thr=%0.2f)" % (len(retino_good_rois), len(all_visual_rois), fit_thr)

retino_az_cond = [c for c in range(2) if S.retinotopy['data'][roi].conditions[c].name == 'right'][0]
retino_el_cond = [c for c in range(2) if S.retinotopy['data'][roi].conditions[c].name == 'top'][0]

retino_widths = [(S.retinotopy['data'][roi].conditions[retino_az_cond].fit_results['sigma'], S.retinotopy['data'][roi].conditions[retino_el_cond].fit_results['sigma']) 
                        for roi in retino_good_rois]


# Plot distributions of averaged RF estimates for each condition:
# -----------------------------------------------------------------------------

colorgroup1 = 'g'
colorgroup2= 'magenta'
colorgroup3 = 'blue'

fig, axes = pl.subplots(2,2, figsize=(12,10))
sns.distplot([np.mean(widths) for widths in widths_1], label=RF.keys()[0], ax=axes.flat[0],
              rug=True, rug_kws={"color": colorgroup1},
              kde_kws={"color": colorgroup1, "lw": 3, "label": None, "alpha": 0.5},
              hist_kws={"histtype": "step", "linewidth": 1, "alpha": 0.2, "color": "g"})
sns.distplot([np.mean(widths) for widths in widths_2], label=RF.keys()[1], ax=axes.flat[0],
              rug=True, rug_kws={"color": colorgroup2},
              kde_kws={"color": colorgroup2, "lw": 3, "label": None, "alpha": 0.5},
              hist_kws={"histtype": "step", "linewidth": 1, "alpha": 0.2, "color": colorgroup2})
sns.distplot([np.mean(widths) for widths in retino_widths], label='retino', ax=axes.flat[0],
              rug=True, rug_kws={"color": colorgroup3},
              kde_kws={"color": colorgroup3, "lw": 3, "label": "KDE", "alpha": 0.5},
              hist_kws={"histtype": "step", "linewidth": 1, "alpha": 0.2, "color": colorgroup3})
axes.flat[0].set_xlabel('mean RF widths')
axes.flat[0].text(-20, 0.9*axes.flat[0].get_ylim()[-1], 'n fit: %i' % len(widths_1), fontsize=14, color=colorgroup1)
axes.flat[0].text(-20, 0.85*axes.flat[0].get_ylim()[-1], 'n fit: %i' % len(widths_2), fontsize=14, color=colorgroup2)
axes.flat[0].text(-20, 0.80*axes.flat[0].get_ylim()[-1], 'n fit: %i' % len(retino_widths), fontsize=14, color=colorgroup3)
sns.despine(offset=4, trim=True, ax=axes.flat[0])


# Create dataframe for TILED data and RETINO data:
common_rois = intersect(good_rois_1, good_rois_2)
all_rois = union(good_rois_1, good_rois_2)

rdf1 = pd.concat([pd.DataFrame(data=RF[RF.keys()[0]][roi]['results'].values(), columns=[roi], index=RF[RF.keys()[0]][roi]['results'].keys()) \
                      for roi in good_rois_1], axis=1)
rdf2 = pd.concat([pd.DataFrame(data=RF[RF.keys()[1]][roi]['results'].values(), columns=[roi], index=RF[RF.keys()[1]][roi]['results'].keys()) \
                      for roi in good_rois_2], axis=1)

width_df = pd.concat([rdf1, rdf2], axis=1)
width_df = width_df.groupby(width_df.columns, axis=1).mean().T  # Average together common ROIs

retino_rdict = dict((roi, {'width_x': S.retinotopy['data'][roi].conditions[retino_az_cond].fit_results['sigma'], 
                           'width_y': S.retinotopy['data'][roi].conditions[retino_el_cond].fit_results['sigma'], 
                           'r2_x': S.retinotopy['data'][roi].conditions[retino_az_cond].fit_results['r2'], 
                           'r2_y': S.retinotopy['data'][roi].conditions[retino_el_cond].fit_results['r2']}) for roi in retino_good_rois)
retino_df = pd.concat([pd.DataFrame(data=rdict.values(), columns=[roi], index=rdict.keys()) for roi, rdict in retino_rdict.items()], axis=1)

    
axes.flat[1].clear()
#roi_colors = sns.color_palette("hls", len(all_rois))
#roi_cdict = dict((roi, roi_colors[ri]) for ri, roi in enumerate(all_rois))
#im = axes.flat[1].scatter(rdf1.T['width_x'], rdf1.T['width_y'], s=rescale_values(rdf1.T['r2']), 
#                  alpha=0.7, c=[roi_cdict[r] for r in rdf1.columns.tolist()], marker='o', label=RF.keys()[0])
#im = axes.flat[1].scatter(rdf2.T['width_x'], rdf2.T['width_y'], s=rescale_values(rdf2.T['r2']), 
#                  alpha=0.7, c=[roi_cdict[r] for r in rdf2.columns.tolist()], marker='x', label=RF.keys()[1])

im = axes.flat[1].scatter(rdf1.T['width_x'], rdf1.T['width_y'], s=rescale_values(rdf1.T['r2']), 
                  alpha=0.5, c=colorgroup1, marker='o', label=RF.keys()[0])
im = axes.flat[1].scatter(rdf2.T['width_x'], rdf2.T['width_y'], s=rescale_values(rdf2.T['r2']), 
                  alpha=0.5, c=colorgroup2, marker='o', label=RF.keys()[1])
im = axes.flat[1].scatter(retino_df.T['width_x'], retino_df.T['width_y'], s=rescale_values(retino_df.T[['r2_x', 'r2_y']].mean(axis=1)), 
                  alpha=0.5, c=colorgroup3, marker='o', label='retino')

axes.flat[1].set_xlabel('width_x')
axes.flat[1].set_ylabel('width_y')
axes.flat[1].set_xlim([0, 120])
axes.flat[1].set_ylim([0, 120])
sns.despine(top=True, right=True, trim=True, ax=axes.flat[1])
axes.flat[1].legend()
axes.flat[1].text(2, axes.flat[1].get_ylim()[-1]-20, 'max r2: %.3f\nmin r2: %.3f' % (max(max(rdf1.T['r2']), max(rdf1.T['r2'])), min(min(rdf1.T['r2']), min(rdf1.T['r2']))))
    
pl.suptitle('%s - %s - %s' % (animalid, session, acquisition))

#%%
#
#universal_rois = intersect([int(i) for i in all_rois], retino_good_rois)
#
#tiled_univ = width_df[width_df.index.isin([str(i) for i in universal_rois])]
#tiled_univ_widths = tiled_univ[['width_x', 'width_y']].mean(axis=1)
#tiled_univ_r2s = tiled_univ['r2']
#
#retino_univ = retino_df.T[retino_df.T.index.isin(universal_rois)]
#retino_univ_widths = retino_univ[['width_x', 'width_y']].mean(axis=1)
#retino_univ_r2s = retino_univ[['r2_x', 'r2_y']].mean(axis=1)

roi_list = union([int(i) for i in all_rois], retino_good_rois)

tiled_univ_widths = [np.mean([width_df.T[str(roi)]['width_x'], width_df.T[str(roi)]['width_y']]) if str(roi) in all_rois else 0 for roi in roi_list]
tiled_univ_r2s = [width_df.T[str(roi)]['r2'] if str(roi) in all_rois else 0 for roi in roi_list]

retino_univ_widths = [np.mean([retino_df[roi]['width_x'], retino_df[roi]['width_y']]) if roi in retino_good_rois else 0  for roi in roi_list]
retino_univ_r2s = [np.mean([retino_df[roi]['r2_x'], retino_df[roi]['r2_y']]) if roi in retino_good_rois else 0  for roi in roi_list]

axes.flat[2].clear()
im = axes.flat[2].scatter(retino_univ_widths, tiled_univ_widths, s=80, marker='o', alpha=0.7)
axes.flat[2].set_xlabel('retino - mean widths')
axes.flat[2].set_ylabel('tiled - mean widths')
axes.flat[2].set_ylim([0, 80])
axes.flat[2].set_xlim([0, 80])
sns.despine(top=True, right=True, trim=True, ax=axes.flat[2])


axes.flat[3].clear()
sns.distplot(tiled_univ_r2s, label='tiled',
              rug=True, rug_kws={"color": 'k'}, kde=False,
              kde_kws={"color": 'k', "lw": 3, "label": "KDE", "alpha": 0.5},
              hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1.0, "color": 'k'}, bins=20)
sns.distplot(retino_univ_r2s, label='retino', kde=False,
              rug=True, rug_kws={"color": 'r'},
              kde_kws={"color": 'r', "lw": 3, "label": "KDE", "alpha": 0.5},
              hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": 'r'}, bins=20)
axes.flat[3].set_xlim([0.4, 1.0])
axes.flat[3].set_xlabel('r2')
axes.flat[3].set_ylabel('counts')
sns.despine(top=True, right=True, trim=True, ax=axes.flat[2])
axes.flat[3].legend()


retino_id = 'retino_%s' % (ss_path.split('_retino')[1].split('_')[2])

data_identifier = '_'.join([animalid, session, acquisition, run1_id, run2_id, retino_id])
label_figure(fig, data_identifier)

pl.savefig(os.path.join(rootdir, animalid, session, acquisition, 'RF_comparisons.png'))


#sns.despine(top=True, right=True, trim=True, ax=axes.flat[3])
#divider = make_axes_locatable(axes.flat[3])
#cax = divider.append_axes("right", size="5%", pad=0.05)
#pl.colorbar(im, cax=cax, label='r2')

#%%

