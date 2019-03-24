#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 17:28:03 2019

@author: julianarhee
"""

import os
import glob
import json
import copy
import pylab as pl

import numpy as np
import scipy as sp
import pandas as pd
from mpl_toolkits.axes_grid1 import AxesGrid
from pipeline.python.utils import natural_keys, label_figure

#%%
#rootdir = '/n/coxfs01/2p-data'
#animalid = 'JC067' #'JC059'
#session = '20190319' #'20190227'
#fov = 'FOV1_zoom2p0x' #'FOV4_zoom4p0x'
#run = 'combined_blobs_static'
#traceid = 'traces002' #'traces001'

rootdir = '/n/coxfs01/2p-data'
animalid = 'JC059' #'JC059'
session = '20190228' #'20190227'
fov = 'FOV1_zoom4p0x' #'FOV4_zoom4p0x'
run = 'combined_blobs_static'
traceid = 'traces001' #'traces001'


fov_dir = os.path.join(rootdir, animalid, session, fov)
traceid_dir = glob.glob(os.path.join(fov_dir, run, 'traces', '%s*' % traceid))[0]
data_fpath = glob.glob(os.path.join(traceid_dir, 'data_arrays', '*.npz'))[0]
dset = np.load(data_fpath)
dset.keys()

data_identifier = '|'.join([animalid, session, fov, run, traceid])
print data_identifier


#%%

# Set dirs:
sorted_dir = sorted(glob.glob(os.path.join(traceid_dir, 'response_stats*')))[-1]
print "Selected stats results: %s" % os.path.split(sorted_dir)[-1]

# Set output dir:
output_dir = os.path.join(sorted_dir, 'visualization')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    

#%%

# Load parsed data:
trace_type = 'corrected'
traces = dset[trace_type]

# Format condition info:
sdf = pd.DataFrame(dset['sconfigs'][()]).T
labels = pd.DataFrame(data=dset['labels_data'], columns=dset['labels_columns'])

# Load roi stats:    
stats_fpath = glob.glob(os.path.join(sorted_dir, 'roistats_results.npz'))[0]
rstats = np.load(stats_fpath)
rstats.keys()

visual_rois = rstats['sorted_visual']
selective_rois = rstats['sorted_selective']

print "Found %i cells that pass responsivity test (%s, p<%.2f)." % (len(visual_rois), rstats['responsivity_test'], rstats['visual_pval'])
print "Found %i cells that pass responsivity test (%s, p<%.2f)." % (len(selective_rois), rstats['selectivity_test'], rstats['selective_pval'])

#%%

# zscore the traces:
zscored_traces_list = []
for trial, tmat in labels.groupby(['trial']):
    #print trial    
    stim_on_frame = tmat['stim_on_frame'].unique()[0]
    nframes_on = tmat['nframes_on'].unique()[0]
    curr_traces = traces[tmat.index, :]
    bas_std = curr_traces[0:stim_on_frame, :].std(axis=0)
    curr_zscored_traces = pd.DataFrame(curr_traces, index=tmat.index).divide(bas_std, axis='columns')
    
    zscored_traces_list.append(curr_zscored_traces)

ztraces = pd.concat(zscored_traces_list, axis=0)
ztraces.head()
#
    
#%%

# Get single value for each trial and sort by config:
trials_by_cond = dict()
for k, g in labels.groupby(['config']):
    trials_by_cond[k] = sorted([int(tr[5:])-1 for tr in g['trial'].unique()])

zscores = dset['zscore']
zscores_by_cond = dict()
for cfg, trial_ixs in trials_by_cond.items():
    zscores_by_cond[cfg] = zscores[trial_ixs, :]
    
avg_zscores_by_cond = pd.DataFrame([zscores_by_cond[cfg].mean(axis=0) for cfg in sorted(zscores_by_cond.keys(), key=natural_keys)])


# Sort mean (or max) zscore across trials for each config, and find "best config"
visual_max_avg_zscore = np.array([avg_zscores_by_cond[rid].max() for rid in visual_rois])
visual_sort_by_max_zscore = np.argsort(visual_max_avg_zscore)[::-1]
sorted_visual = visual_rois[visual_sort_by_max_zscore]

selective_max_avg_zscore = np.array([avg_zscores_by_cond[rid].max() for rid in selective_rois])
selective_sort_by_max_zscore = np.argsort(selective_max_avg_zscore)[::-1]
sorted_selective = selective_rois[selective_sort_by_max_zscore]

print [r for r in sorted_selective if r not in sorted_visual]

print sorted_selective[0:10]

#%%
rows = 'size'
cols = 'morphlevel'

sizes = sorted(sdf['size'].unique())
morphlevels = sorted(sdf['morphlevel'].unique())

config_trial_ixs = dict()
cix = 0
for si, size in enumerate(sorted(sizes)):
    for mi, morph in enumerate(morphlevels):
        config_trial_ixs[cix] = {}
        cfg = sdf[(sdf['size']==size) & (sdf['morphlevel']==morph)].index.tolist()[0]
        trial_ixs = sorted( list(set( [int(tr[5:])-1 for tr in labels[labels['config']==cfg]['trial']] )) )
        config_trial_ixs[cix]['config'] = cfg
        config_trial_ixs[cix]['trial_ixs'] = trial_ixs
        cix += 1


if not os.path.exists(os.path.join(output_dir, 'roi_trials_by_cond')):
    os.makedirs(os.path.join(output_dir, 'roi_trials_by_cond'))
print "Saving figures to:", os.path.join(output_dir, 'roi_trials_by_cond')

#rid = 137 #14

for rid in sorted_selective:
    print rid
    roi_zscores = zscores[:, rid]
    roi_trace = ztraces[rid] #traces[:, rid]
    traces_by_config = dict((config, []) for config in labels['config'].unique())
    for k, g in labels.groupby(['config', 'trial']):
        traces_by_config[k[0]].append(roi_trace[g.index])
    for config in traces_by_config.keys():
        traces_by_config[config] = np.vstack(traces_by_config[config])
    
    
    fig = pl.figure(figsize=(18,4))
    grid = AxesGrid(fig, 111,
                    nrows_ncols=(len(sizes), len(morphlevels)),
                    axes_pad=0.2,
                    cbar_mode='single',
                    cbar_location='right',
                    cbar_pad=0.1)
    
    #aix = 0
    #zscores_by_cond_list = []
    for aix in sorted(config_trial_ixs.keys()): # Ordered by stim conditions
        #print aix
        ax = grid.axes_all[aix]
        cfg = config_trial_ixs[aix]['config']
        trial_ixs = config_trial_ixs[aix]['trial_ixs']
        
        curr_cond_traces = traces_by_config[cfg]
        im = ax.imshow(curr_cond_traces, cmap='inferno')
        ax.set_aspect(2.5)
        ax.set_axis_off()
        
        # get zscore values:
        curr_cond_zscores = roi_zscores[trial_ixs]
        ax.set_title('%.2f (std %.2f)' % (curr_cond_zscores.mean(), sp.stats.sem(curr_cond_zscores)), fontsize=6)
        #aix += 1
        #zscores_by_cond_list.append(pd.Series(curr_cond_zscores, name=cfg))
        
        ax.axvline(x=stim_on_frame, color='w', lw=0.5, alpha=0.5)
        ax.axvline(x=stim_on_frame+nframes_on, color='w', lw=0.5, alpha=0.5)
            
    fig.suptitle('roi %i' % int(rid+1))
    
    cbar = ax.cax.colorbar(im)
    cbar = grid.cbar_axes[0].colorbar(im)
    
    figname = 'zscored_trials_roi%05d' % int(rid+1)
    label_figure(fig, data_identifier)
    pl.savefig(os.path.join(output_dir, 'roi_trials_by_cond', '%s.png' % figname))
    
    #roi_zscores_by_cond = pd.DataFrame(zscores_by_cond_list).T
    #print roi_zscores_by_cond.head()

    pl.close()


#%%
#
## Do any stats correlate nicely with mag of response?
responsive_test = 'RManova1'
vstats_fpath = glob.glob(os.path.join(sorted_dir,  'responsivity_%s*' % responsive_test, '*%s_results.json' % responsive_test ))[0]
with open(vstats_fpath, 'r') as f:
    vstats = json.load(f)
#
#stat_types = ['F', 'mse', 'eta', 'eta2_p']
#
#fig, axes = pl.subplots(1, len(stat_types))
#for aix, stat in enumerate(stat_types):
#    ax = axes[aix]
#    max_avg_zscore = np.array([avg_zscores_by_cond[rid].max() for rid in selective_rois])
#    vstat_val = [vstats[str(rid)][stat] for rid in selective_rois]
#    ax.scatter(max_avg_zscore, vstat_val)
#    
#
##%%
#rid = 14
#
#eta2s = []
#for rid in selective_rois:
#    aov_results_fpath = os.path.join(sorted_dir, 'responsivity', 'SPanova2_results', 'visual_anova_results_%s.txt' % rid)
#    with open(aov_results_fpath, 'rb') as f:
#        aov = f.read().strip()
#    
#    strt = aov.find('\nepoch * config') + 1
#    ed = aov[strt:].find('\n')
#    epoch_config_eta2 = float([s for s in aov[strt:strt+ed].split(' ') if s !=''][11])
#    eta2s.append(epoch_config_eta2)
#    
#pl.figure()
#pl.scatter(max_avg_zscore, eta2s)
#
#sort_by_eta2 = np.argsort(eta2s)[::-1]


#%%

use_selective = False
plot_topN = False


if use_selective:
    roi_list = copy.copy(sorted_selective)
    sorter = 'selective'
else:
    roi_list = copy.copy(sorted_visual)
    sorter = 'visual'
    
if plot_topN:
    sort_order = 'top'
else:
    sort_order = 'bottom'
    
    


nr = 10
nc = 6

#roi_list = [r for r in sorted_selective if r not in sorted_visual]
#sorter = 'selXvis'

fig = pl.figure(figsize=(10,8))
grid = AxesGrid(fig, 111,
                nrows_ncols=(nr, nc),
                axes_pad=0.2,
                cbar_mode='single',
                cbar_location='right',
                cbar_pad=0.1)

aix = 0
nplots = min([nr*nc, len(roi_list)])

if not plot_topN:
    plot_order = roi_list[-nplots:]
else:
    plot_order = roi_list[0:nplots]
    
    
for rid in plot_order: #0:nplots]: #[0:nr*nc]:
    
    roi_zscores = zscores[:, rid]
    roi_trace = ztraces[rid] #[:, rid]

    traces = dict((config, []) for config in labels['config'].unique())
    for k, g in labels.groupby(['config', 'trial']):
        traces[k[0]].append(roi_trace[g.index])
    for config in traces.keys():
        traces[config] = np.vstack(traces[config])

    best_cfg = 'config%03d' % int(avg_zscores_by_cond[rid].argmax()+1)
    curr_cond_traces = traces[best_cfg]
    
    ax = grid.axes_all[aix]
    im = ax.imshow(curr_cond_traces, cmap='inferno')
    ax.set_aspect(2.5)
    ax.set_axis_off()
        
    ax.axvline(x=stim_on_frame, color='w', lw=0.5, alpha=0.5)
    ax.axvline(x=stim_on_frame+nframes_on, color='w', lw=0.5, alpha=0.5)
        
        
    ax.set_title('%i (%.2f)' % (rid, avg_zscores_by_cond[rid].max()), fontsize=6)
    aix += 1
    
cbar = ax.cax.colorbar(im)
cbar = grid.cbar_axes[0].colorbar(im)

for a in np.arange(aix, nr*nc):
    grid.axes_all[a].set_axis_off()
    

label_figure(fig, data_identifier)
figname = 'sorted_%s_best_cfg_zscored_trials_%s%i' % (sorter, sort_order, nplots)
pl.savefig(os.path.join(output_dir, '%s.png' % figname))
print figname

#    
#for rid in sorted_selective:
#    print vstats[str(rid)]['p']
#
#pl.figure();
#pl.plot([vstats[str(rid)]['p'] for rid in sorted_selective])


#%%
fig, ax = pl.subplots(figsize=(10,5)) #pl.figure(figsize=(10,5))

ax.plot(sorted_visual, np.array([avg_zscores_by_cond[rid].max() for rid in sorted_visual]), 'k.', markersize=10, label='visual', alpha=0.5) #, markersize=20, alpha=20)
ax.plot(sorted_selective, np.array([avg_zscores_by_cond[rid].max() for rid in sorted_selective]), 'r.', markersize=10, label='selective', alpha=0.5) #, markersize=20, alpha=20)
ax.set_ylabel('zscore')
ax.set_xlabel('roi')

ax.axhline(y=2, linestyle='--', color='k')

for rid in sorted_selective[0:10]:
    print rid
    ax.annotate('%i' % rid, (rid, float(avg_zscores_by_cond[rid].max())))

    
pl.legend()
label_figure(fig, data_identifier)

pl.savefig(os.path.join(output_dir, 'visual_selective_rois.png'))



