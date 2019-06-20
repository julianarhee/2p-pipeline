#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 17:11:03 2019

@author: julianarhee
"""

import optparse
import glob
import os
import copy

import numpy as np
import pandas as pd
import cPickle as pkl
import pylab as pl
import seaborn as sns

from pipeline.python.utils import label_figure, natural_keys

def extract_options(options):
    
    parser = optparse.OptionParser()

    parser.add_option('-D', '--root', action='store', dest='rootdir', default='/n/coxfs01/2p-data',\
                      help='data root dir (root project dir containing all animalids) [default: /n/coxfs01/2pdata]')
    parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')

    # Set specific session/run for current animal:
    parser.add_option('-S', '--session', action='store', dest='session', default='', \
                      help='session dir (format: YYYMMDD_ANIMALID')
    parser.add_option('-A', '--acq', action='store', dest='acquisition', default='FOV1', \
                      help="acquisition folder (ex: 'FOV1_zoom3x') [default: FOV1]")
    parser.add_option('-R', '--run', action='store', dest='run', default='retino_run1', \
                      help="name of run dir containing tiffs to be processed (ex: gratings_phasemod_run1)")
    parser.add_option('-t', '--traceid', action='store', dest='traceid', default='traces001', \
                      help="name of traces ID [default: traces001]")
    parser.add_option('-d', '--data-type', action='store', dest='trace_type', default='corrected', \
                      help="Trace type to use for analysis [default: corrected]")
    parser.add_option('--segment', action='store_true', dest='segment', default=False, \
                      help="Set flag to use segmentation of FOV for select visual area")
    parser.add_option('-V', '--area', action='store', dest='visual_area', default='', \
                      help="Name of visual area, if --segment is flagged")
    (options, args) = parser.parse_args(options)

    return options


#%%

rootdir = '/n/coxfs01/2p-data'
animalid = 'JC099' #'JC059'
session = '20190609' #'20190227'
fov = 'FOV1_zoom2p0x' #'FOV4_zoom4p0x'
run = 'combined_blobs_static'
traceid = 'traces001' #'traces001'
segment = False
visual_area = ''
trace_type = 'corrected'
select_rois = False


options = ['-i', animalid, '-S', session, '-A', fov, '-R', run, '-t', traceid]
if segment:
    options.extend(['--segment', '-V', visual_area])
    

fov_dir = os.path.join(rootdir, animalid, session, fov)
traceid_dirs = glob.glob(os.path.join(fov_dir, run, 'traces', '%s*' % traceid))
if len(traceid_dirs) > 1:
    print "More than 1 trace ID found:"
    for ti, traceid_dir in enumerate(traceid_dirs):
        print ti, traceid_dir
    sel = input("Select IDX of traceid to use: ")
    traceid_dir = traceid_dirs[int(sel)]
else:
    traceid_dir = traceid_dirs[0]
#traceid = os.path.split(traceid_dir)[-1]
    
    
data_fpath = glob.glob(os.path.join(traceid_dir, 'data_arrays', '*.npz'))[0]
dset = np.load(data_fpath)
dset.keys()

data_identifier = '|'.join([animalid, session, fov, run, traceid, visual_area])
print data_identifier


#%%
segment=False
included_rois = []
if segment:
    segmentations = glob.glob(os.path.join(fov_dir, 'visual_areas', '*.pkl'))
    assert len(segmentations) > 0, "Specified to segment, but no segmentation file found in acq. dir!"
    if len(segmentations) == 1:
        segmentation_fpath = segmentations[0]
    else:
        for si, spath in enumerate(sorted(segmentations, key=natural_keys)):
            print si, spath
        sel = input("Select IDX of seg file to use: ")
        segmentation_fpath = sorted(segmentations, key=natural_keys)[sel]
    with open(segmentation_fpath, 'rb') as f:
        seg = pkl.load(f)
            
    included_rois = seg.regions[visual_area]['included_rois']
    print "Found %i rois in visual area %s" % (len(included_rois), visual_area)

#%%

# Set dirs:
sorted_dir = sorted(glob.glob(os.path.join(traceid_dir, 'response_stats*')))[-1]
print "Selected stats results: %s" % os.path.split(sorted_dir)[-1]


#%%

# Load parsed data:
traces = dset[trace_type] #, index=zscored_traces.index)

select_rois = True


# Format condition info:
aspect_ratio = 1 #1.747
sdf = pd.DataFrame(dset['sconfigs'][()]).T

if 'color' in sdf.columns:
    sdf = sdf[sdf['color']=='']
sdf['size'] = [round(sz/aspect_ratio, 1) for sz in sdf['size']]
sdf.head()
c_offset = int(sdf.index.tolist()[0][6:]) 
    

labels = pd.DataFrame(data=dset['labels_data'], columns=dset['labels_columns'])

# Load roi stats:    
stats_fpath = glob.glob(os.path.join(sorted_dir, 'roistats_results.npz'))[0]
rstats = np.load(stats_fpath)
rstats.keys()

#%%
# zscore the traces:
# -----------------------------------------------------------------------------
baselines_list = []
zscores_list = []
snr_list = []
for trial, tmat in labels.groupby(['trial']):
    #print trial    
    stim_on_frame = tmat['stim_on_frame'].unique()[0]
    nframes_on = tmat['nframes_on'].unique()[0]
    curr_traces = traces[tmat.index, :]
    bas_std = curr_traces[0:stim_on_frame, :].std(axis=0)
    bas_mean = curr_traces[0:stim_on_frame, :].mean(axis=0)
    stim_mean = curr_traces[stim_on_frame:stim_on_frame+nframes_on, :].mean(axis=0)
    curr_zs = (stim_mean - bas_mean) / bas_std
    curr_snr = stim_mean / bas_mean
    zscores_list.append(curr_zs)
    baselines_list.append(bas_mean)
    snr_list.append(curr_snr)


zscores = np.array(zscores_list)
baselines = np.array(baselines_list)
snrs = np.array(zscores_list)

#%%
use_zscore = False

# Only take subset of trials where image shown (not controls):
labels = labels[labels['config'].isin(sdf.index.tolist())] # Ignore "control" trials for now
#traces = traces[labels.index.tolist(), :]
trial_ixs = np.array([int(t[5:])-1 for t in sorted(labels['trial'].unique(), key=natural_keys)])

#%%
# Sort ROIs by zscore by cond
# -----------------------------------------------------------------------------
snrs = np.array(snr_list)
zscores = np.array(zscores_list)

# Get single value for each trial and sort by config:
trials_by_cond = dict()
for k, g in labels.groupby(['config']):
    trials_by_cond[k] = sorted([int(tr[5:])-1 for tr in g['trial'].unique()])

resp_by_cond = dict()
for cfg, t_ixs in trials_by_cond.items():
    if use_zscore:
        resp_by_cond[cfg] = zscores[t_ixs, :]  # For each config, array of size ntrials x nrois
    else:
        resp_by_cond[cfg] = snrs[t_ixs, :]
    
avg_resp_by_cond = pd.DataFrame([resp_by_cond[cfg].mean(axis=0) \
                                    for cfg in sorted(resp_by_cond.keys(), key=natural_keys)],\
                                    index=[int(cf[6:])-1 for cf in sorted(resp_by_cond.keys(), key=natural_keys)]) # nconfigs x nrois
    
# Sort mean (or max) zscore across trials for each config, and find "best config"
visual_max_avg_zscore = np.array([avg_resp_by_cond[rid].max() for rid in visual_rois])
visual_sort_by_max_zscore = np.argsort(visual_max_avg_zscore)[::-1]
sorted_visual = visual_rois[visual_sort_by_max_zscore]

selective_max_avg_zscore = np.array([avg_resp_by_cond[rid].max() for rid in selective_rois])
selective_sort_by_max_zscore = np.argsort(selective_max_avg_zscore)[::-1]
sorted_selective = selective_rois[selective_sort_by_max_zscore]

print [r for r in sorted_selective if r not in sorted_visual]
print sorted_selective[0:10]

roi_selector = 'all'

if roi_selector == 'visual':
    roi_list = np.array(sorted(copy.copy(sorted_visual)))
elif roi_selector == 'selective':
    roi_list = np.array(sorted(copy.copy(sorted_selective)))
elif roi_selector == 'all':
    roi_list = np.arange(0, avg_resp_by_cond.shape[-1])
    
    
responsive_cells = np.array([i for i, iv in enumerate(roi_list) if avg_resp_by_cond[iv].max() > 1.5])
responses = avg_resp_by_cond[roi_list[responsive_cells]]

#%%

snrs = snrs[:, responsive_cells]
snrs = snrs[trial_ixs, :]
print(snrs.shape)


zscores = zscores[:, responsive_cells]
zscores = zscores[trial_ixs, :]
print(zscores.shape)
zscores = pd.DataFrame(zscores)

#%%

fig_subdir = 'corrs_%s' % roi_selector 

if segment:
    curr_figdir = os.path.join(traceid_dir, 'figures', 'population', fig_subdir, visual_area)
else:
    curr_figdir = os.path.join(traceid_dir, 'figures', 'population', fig_subdir)
if not os.path.exists(curr_figdir):
    os.makedirs(curr_figdir)
print "Saving plots to: %s" % curr_figdir



#%%
cluster_corrs = True

#bdf = baselines.T
bdf = responses.T
print(bdf.shape)

#bdf.columns = ['|'.join([str(sdf['morphlevel']['config%03d' % int(c+1)]), str(int(sdf['size']['config%03d' % int(c+1)]))]) for c in bdf.columns.tolist()]
bdf.columns = ['|'.join([str(sdf['morphlevel'][cfg]), str(int(sdf['size'][cfg]))]) for cfg in bdf.columns.tolist()]



bcorrs = bdf.corr()

cmap = sns.diverging_palette(220, 20, sep=20, as_cmap=True)

fig, ax = pl.subplots(figsize=(8,8))
sns.heatmap(bcorrs, cmap=cmap)
pl.xticks(range(len(bcorrs.columns)), bcorrs.columns, rotation=45, fontsize=6);
pl.yticks(range(len(bcorrs.columns)), bcorrs.columns, rotation=45, fontsize=6);
ax.set_aspect('equal')

label_figure(fig, data_identifier)

if cluster_corrs:
    pl.savefig(os.path.join(curr_figdir, 'corr_ordered_configs.png'))
else:
    pl.savefig(os.path.join(curr_figdir, 'baseline_trial2trial_ordered_trials.png'))


#%%
dist_metric = 'euclidean' #'mahalanobis'

#fig, ax = pl.subplots()
cg = sns.clustermap(bcorrs, metric=dist_metric) # metric='mahalanobis')
pl.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
pl.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), rotation=45)

# For the hierarchical clustering distance values, you can access the linkage matrics for rows or columns with:
    
cg.dendrogram_col.linkage # linkage matrix for columns
cg.dendrogram_row.linkage # linkage matrix for rows

label_figure(cg.fig, data_identifier)
pl.savefig(os.path.join(curr_figdir, 'clustermap_corrs_%s.png' % dist_metric))



#%%
import scipy.cluster.hierarchy as sch



def plot_corr(df,size=10, ix_labels=None, cmap='RdYlGn'):
    '''Plot a graphical correlation matrix for a dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot'''
    
    # Compute the correlation matrix for the received dataframe
    corr = df.corr()
    
    # Plot the correlation matrix
    fig, ax = pl.subplots(figsize=(size, size))
    cax = ax.matshow(corr, cmap=cmap) #, vmin=-1, vmax=1)
    pl.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=4);
    pl.yticks(range(len(corr.columns)), corr.columns, fontsize=4);
    
    if ix_labels is not None:
        ax.set_xticklabels(ix_labels)
        ax.set_yticklabels(ix_labels)
    # Add the colorbar legend
    cbar = fig.colorbar(cax, ticks=[-1, 0, 1], aspect=40, shrink=.8)

    return fig

#%%
X = bcorrs.values

d = sch.distance.pdist(X, metric='euclidean')   # vector of ('55' choose 2) pairwise distances

L = sch.linkage(d, method='complete')
ind = sch.fcluster(L, 0.5*d.max(), 'distance')
columns = [bdf.columns.tolist()[i] for i in list((np.argsort(ind)))]
df = bdf.reindex_axis(columns, axis=1)


#ix_labels = ['|'.join([str(sdf['morphlevel']['config%03d' % int(c+1)]), str(int(sdf['size']['config%03d' % int(c+1)]))]) for c in df.columns.tolist()]

fig = plot_corr(df, size=10, cmap=cmap)



label_figure(fig, data_identifier)
if cluster_corrs:
    pl.savefig(os.path.join(curr_figdir, 'corr_clustered_configs.png'))
else:
    pl.savefig(os.path.join(curr_figdir, 'baseline_trial2trial_clustered_trials.png'))




