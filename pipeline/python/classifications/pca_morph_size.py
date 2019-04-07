#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 15:59:12 2019

@author: julianarhee
"""



import os
import glob
import json
import copy
import pylab as pl
import seaborn as sns
import cPickle as pkl
import numpy as np
import scipy as sp
from scipy import stats
import pandas as pd
import matplotlib.cm as cm
import sklearn as sk
from sklearn import decomposition

from pipeline.python.utils import natural_keys, label_figure


from mpl_toolkits.axes_grid1 import AxesGrid
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm
from mpl_toolkits.mplot3d import Axes3D

from mpl_toolkits.mplot3d import Axes3D
import sklearn as sk

#%%


def plot_pca_label_points(X_r, y, label_names=[], label_colors=[], ax=None):
    
    if ax is None:
        print "Creating new axis"
        fig, ax = pl.subplots()

    # Get target names
    target_names = sorted(np.unique(y))
    
    if label_names is None:
        label_names = copy.copy(target_names)
        
    # Create default cmap if none provided:
    if label_colors is None:
        label_colors = sns.cubehelix_palette(len(label_names))
    
    # Plot:
    for color, i, target_name in zip(label_colors, label_names, target_names):
        ax.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, label=target_name,
                   alpha=.8, lw=lw, edgecolor='k') #abel=target_name)        
    
    # Clean up axes:
    xlim = max([abs(xl) for xl in ax.get_xlim()])
    ylim = max([abs(yl) for yl in ax.get_ylim()])
    ax.set_xlim([-xlim, xlim])
    ax.set_ylim([-ylim, ylim])
    ax.xaxis.set_major_locator(MaxNLocator(3, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(3, integer=True))
    
    return ax

#%%
def plot_pca_label_points_3D(X_r, y, label_names=[], cmap=None, \
                             ax=None, annotate=False, markersize=100):
    
    if ax is None:
        print "Creating new axis"
        fig = pl.figure(1, figsize=(8, 6))
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=50, azim=50)

    # Get target names
    target_names = sorted(np.unique(y))
    
    if label_names is None:
        label_names = copy.copy(target_names)
    
    # Create default cmap if none provided:
    if cmap is None:
        cmap = sns.cubehelix_palette(as_cmap=True)
    
    if annotate:
        for name, label in zip(label_names, target_names):
            ax.text3D(X_r[y == label, 0].mean(),
                      X_r[y == label, 1].mean() + 1.5,
                      X_r[y == label, 2].mean(), name,
                      horizontalalignment='center',
                      bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
        
    # Reorder the labels to have colors matching the cluster results
    ax.scatter(X_r[:, 0], X_r[:, 1], X_r[:, 2], c=y, cmap=cmap, s=np.ones(y.shape)*markersize,
               edgecolor='w')
    
    #ax.w_xaxis.set_ticklabels([])
    #ax.w_yaxis.set_ticklabels([])
    #ax.w_zaxis.set_ticklabels([])

    return ax



#%%
visual_area = ''
segment = False


#rootdir = '/n/coxfs01/2p-data'
#animalid = 'JC070' #'JC059'
#session = '20190316' #'20190227'
#fov = 'FOV1_zoom2p0x' #'FOV4_zoom4p0x'
#run = 'combined_blobs_static'
#traceid = 'traces001' #'traces001'

#rootdir = '/n/coxfs01/2p-data'
#animalid = 'JC067' 
#session = '20190319' #'20190319'
#fov = 'FOV1_zoom2p0x' 
#run = 'combined_blobs_static'
#traceid = 'traces001' #'traces002'

#rootdir = '/n/coxfs01/2p-data'
#animalid = 'JC073' #'JC059'
#session = '20190322' #'20190227'
#fov = 'FOV1_zoom2p0x' #'FOV4_zoom4p0x'
#run = 'combined_blobs_static'
#traceid = 'traces001' #'traces001'
#segment = True
#visual_area = 'LI'

#rootdir = '/n/coxfs01/2p-data'
#animalid = 'JC059' #'JC059'
#session = '20190228' #'20190227'
#fov = 'FOV1_zoom4p0x' #'FOV4_zoom4p0x'
#run = 'combined_blobs_static'
#traceid = 'traces001' #'traces001'


fov_dir = os.path.join(rootdir, animalid, session, fov)
traceid_dir = glob.glob(os.path.join(fov_dir, run, 'traces', '%s*' % traceid))[0]
data_fpath = glob.glob(os.path.join(traceid_dir, 'data_arrays', '*.npz'))[0]
dset = np.load(data_fpath)
dset.keys()

data_identifier = '|'.join([animalid, session, fov, run, traceid, visual_area])
print data_identifier


#%%

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

# Set output dir:
#output_dir = os.path.join(sorted_dir, 'visualization')
#if segment:
#    output_dir = os.path.join(sorted_dir, 'visualization', visual_area)
#if not os.path.exists(output_dir):
#    os.makedirs(output_dir)
#print "Saving output to:", output_dir

#%%

# Load parsed data:
trace_type = 'corrected'
traces = dset[trace_type]
zscores = dset['zscore']

# Format condition info:
aspect_ratio = 1.747
sdf = pd.DataFrame(dset['sconfigs'][()]).T
sdf['size'] = [round(sz/aspect_ratio, 1) for sz in sdf['size']]
labels = pd.DataFrame(data=dset['labels_data'], columns=dset['labels_columns'])

# Load roi stats:    
stats_fpath = glob.glob(os.path.join(sorted_dir, 'roistats_results.npz'))[0]
rstats = np.load(stats_fpath)
rstats.keys()

#%%
if segment and len(included_rois) > 0:
    all_rois = np.array(copy.copy(included_rois))
else:
    all_rois = np.arange(0, rstats['nrois_total'])

visual_rois = np.array([r for r in rstats['sorted_visual'] if r in all_rois])
selective_rois = np.array([r for r in rstats['sorted_selective'] if r in all_rois])

print "Found %i cells that pass responsivity test (%s, p<%.2f)." % (len(visual_rois), rstats['responsivity_test'], rstats['visual_pval'])
print "Found %i cells that pass responsivity test (%s, p<%.2f)." % (len(selective_rois), rstats['selectivity_test'], rstats['selective_pval'])

#%%

# zscore the traces:
# -----------------------------------------------------------------------------
zscored_traces_list = []
for trial, tmat in labels.groupby(['trial']):
    #print trial    
    stim_on_frame = tmat['stim_on_frame'].unique()[0]
    nframes_on = tmat['nframes_on'].unique()[0]
    curr_traces = traces[tmat.index, :]
    bas_std = curr_traces[0:stim_on_frame, :].std(axis=0)
    curr_zscored_traces = pd.DataFrame(curr_traces, index=tmat.index).divide(bas_std, axis='columns')
    
    zscored_traces_list.append(curr_zscored_traces)

zscored_traces = pd.concat(zscored_traces_list, axis=0)
zscored_traces.head()

raw_traces = pd.DataFrame(traces, index=zscored_traces.index)
raw_traces.head()
    
#%

# Sort ROIs by zscore by cond
# -----------------------------------------------------------------------------

# Get single value for each trial and sort by config:
trials_by_cond = dict()
for k, g in labels.groupby(['config']):
    trials_by_cond[k] = sorted([int(tr[5:])-1 for tr in g['trial'].unique()])

GM = zscores.mean()
zscores_gm = zscores - GM
zscores_by_cond = dict()
zscores_by_cond_GM = dict()
for cfg, trial_ixs in trials_by_cond.items():
    zscores_by_cond[cfg] = zscores[trial_ixs, :]  # For each config, array of size ntrials x nrois
    zscores_by_cond_GM[cfg] = zscores_gm[trial_ixs, :]
    
avg_zscores_by_cond = pd.DataFrame([zscores_by_cond[cfg].mean(axis=0) \
                                    for cfg in sorted(zscores_by_cond.keys(), key=natural_keys)]) # nconfigs x nrois
    
avg_zscores_by_cond_GM = pd.DataFrame([zscores_by_cond_GM[cfg].mean(axis=0) \
                                    for cfg in sorted(zscores_by_cond_GM.keys(), key=natural_keys)]) # nconfigs x nrois

    
# Sort mean (or max) zscore across trials for each config, and find "best config"
visual_max_avg_zscore = np.array([avg_zscores_by_cond[rid].max() for rid in visual_rois])
visual_sort_by_max_zscore = np.argsort(visual_max_avg_zscore)[::-1]
sorted_visual = visual_rois[visual_sort_by_max_zscore]

selective_max_avg_zscore = np.array([avg_zscores_by_cond[rid].max() for rid in selective_rois])
selective_sort_by_max_zscore = np.argsort(selective_max_avg_zscore)[::-1]
sorted_selective = selective_rois[selective_sort_by_max_zscore]

print [r for r in sorted_selective if r not in sorted_visual]

print sorted_selective[0:10]

#%

## Get SNR
## -----------------------------------------------------------------------------
#bas_means = np.vstack([raw_traces.iloc[trial_indices.index][0:stim_on_frame].mean(axis=0) \
#                       for trial, trial_indices in labels.groupby(['trial'])])
#stim_means = np.vstack([raw_traces.iloc[trial_indices.index][stim_on_frame:(stim_on_frame+nframes_on)].mean(axis=0) \
#                       for trial, trial_indices in labels.groupby(['trial'])])
#snrs = stim_means/bas_means
  

sizes = sorted(sdf['size'].unique())
morphlevels = sorted(sdf['morphlevel'].unique())



#%%


fig_subdir = 'pca' 

if segment:
    curr_figdir = os.path.join(traceid_dir, 'figures', 'population', fig_subdir, visual_area)
else:
    curr_figdir = os.path.join(traceid_dir, 'figures', 'population', fig_subdir)
if not os.path.exists(curr_figdir):
    os.makedirs(curr_figdir)
print "Saving plots to: %s" % curr_figdir



#%%

subtract_GM = True

lw = 0.5
n_components=2
size_colors = sns.cubehelix_palette(len(sizes))
size_cmap = sns.cubehelix_palette(as_cmap=True, rot=0.4, hue=1)
morph_colors = sns.diverging_palette(220, 20, n=len(morphlevels))
morph_cmap = sns.diverging_palette(220, 20, as_cmap=True)

# PCA data:
if subtract_GM:
    X = avg_zscores_by_cond_GM[selective_rois]
else:
    X = avg_zscores_by_cond[selective_rois]
pca = sk.decomposition.PCA(n_components=n_components)
X_r = pca.fit(X).transform(X)

    
fig, axes = pl.subplots(1,2, figsize=(8,5))
fig.subplots_adjust(top=0.8, bottom=0.3, wspace=0.2, hspace=0.2, left=0.1)

y_size = np.array([sdf['size'][cfg] for cfg in sdf.index.tolist()])
ax = axes[0]
ax = plot_pca_label_points(X_r, y_size, label_names=sizes, label_colors=size_colors, ax=ax)

ax.text(ax.get_xlim()[0], ax.get_ylim()[-1]*1.02, \
        'expl. var. %.2f' % np.sum(pca.explained_variance_ratio_), fontsize=6)
# Add legend:
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2),
          ncol=len(sizes), fancybox=False, shadow=False, fontsize=6)
fig.text(0.25, 0.15 , 'size')



# Label MORPH LEVEL:
y_morph = np.array([sdf['morphlevel'][cfg] for cfg in sdf.index.tolist()])

# Plot
ax = axes[1]
ax = plot_pca_label_points(X_r, y_morph, label_names=morphlevels, label_colors=morph_colors, ax=ax)
#ax.text(ax.get_xlim()[0], ax.get_ylim()[-1]*1.02, \
#        'expl. var. %.2f' % np.sum(pca.explained_variance_ratio_), fontsize=6)

# Create colorbar for morphlevels:
morph_cmap = sns.diverging_palette(220, 20, as_cmap=True)
bounds = np.arange(0, len(morphlevels))
norm = BoundaryNorm(bounds, morph_cmap.N)
mappable = cm.ScalarMappable(cmap=morph_cmap)
mappable.set_array(bounds)

cbar_ax = fig.add_axes([0.58, 0.22, 0.3, 0.02])
cbar = fig.colorbar(mappable, cax=cbar_ax, boundaries=np.arange(-0.5,len(morphlevels),1), \
                    ticks=bounds, norm=norm, orientation='horizontal')

cbar.ax.tick_params(axis='both', which='both',length=0)
cbar.ax.set_xticklabels(morphlevels, fontsize=6) #(['%i' % i for i in morphlevels])  # horizontal colorbar
cbar.ax.set_xlabel('morph', fontsize=10)

pl.suptitle('PCA (n=%i)' % n_components, y=0.9)

label_figure(fig, data_identifier)

if subtract_GM:
    figname = 'pca_%icomps_averaged_condns_GM_morph_size' % n_components
else:
    figname = 'pca_%icomps_averaged_condns_morph_size' % n_components
pl.savefig(os.path.join(curr_figdir, '%s.png' % figname))

print figname

#%%

# Label SIZE:
fig = pl.figure(1, figsize=(8, 6))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=30, azim=100)
#ax = Axes3D(fig, rect=[0, .02, .9, 0.9], elev=50, azim=50)

#X = avg_zscores_by_cond_GM[selective_rois]
#y = np.array([int(sdf['size'][cfg]) for cfg in sdf.index.tolist()])

n_components=3
pca = sk.decomposition.PCA(n_components=n_components)
pca.fit(X)
X_r = pca.transform(X)
ax = plot_pca_label_points_3D(X_r, y_size, label_names=sizes, cmap=size_cmap, \
                             ax=ax, annotate=True, markersize=100)

fig.text(0.05, 0.1, 'expl. var %.2f' % np.sum(pca.explained_variance_ratio_))
fig.suptitle('pca size (avg per cond)', y=0.95)

label_figure(fig, data_identifier)

#%
figname = 'pca_3d_size_averaged_condns_ncomps%i_view.png' % (n_components) 
pl.savefig(os.path.join(curr_figdir, '%s.png' % figname))

print figname
#pl.close()

#%%
ax.azim = 130
ax.elev = 50
figname = 'pca_3d_size_averaged_condns_ncomps%i_view2.png' % (n_components) 
pl.savefig(os.path.join(curr_figdir, '%s.png' % figname))
print figname

pl.close()

#%%

# Label MORPH:

fig = pl.figure(1, figsize=(8, 6))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=50, azim=50)

ax = plot_pca_label_points_3D(X_r, y_morph, label_names=morphlevels, cmap=morph_cmap, \
                             ax=ax, annotate=True, markersize=100)

fig.text(0.05, 0.1, 'expl. var %.2f' % np.sum(pca.explained_variance_ratio_))
fig.suptitle('pca morph (avg per cond)', y=0.95)

label_figure(fig, data_identifier)

figname = 'pca_3d_morph_averaged_condns_ncomps%i_view.png' % (n_components) 
pl.savefig(os.path.join(curr_figdir, '%s.png' % figname))
print figname

#%%

figname = 'pca_3d_morph_averaged_condns_ncomps%i_view2.png' % (n_components) 
pl.savefig(os.path.join(curr_figdir, '%s.png' % figname))
print figname

#%%



# PCA subset of data:

#X = avg_zscores_by_cond[selective_rois] - GM
#y = np.array([int(sdf['morphlevel'][cfg]) for cfg in sdf.index.tolist()])


for curr_size in sizes:
    
    fig = pl.figure(1, figsize=(8, 6))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=50, azim=50)
    ax.clear()
    
    curr_cfgs = sdf[sdf['size']==curr_size]['morphlevel'].index.tolist()
    
    curr_trial_ixs = np.array(sorted([int(trial[5:])-1 for trial in labels[labels['config'].isin(curr_cfgs)]['trial'].unique()]))
    curr_cfg_ixs = np.array([int(cfg[6:])-1 for cfg in curr_cfgs])
    
    curr_zs = zscores_gm[:, selective_rois]
    X = curr_zs[curr_trial_ixs, :]
    
    curr_trials = ['trial%05d' % int(tix+1) for tix in curr_trial_ixs]
    curr_y = np.array([sdf['morphlevel'][labels[labels['trial']==trial]['config'].unique()[0]] for trial in curr_trials ])
    
    n_components=3
    pca = sk.decomposition.PCA(n_components=n_components)
    pca.fit(X)
    X_r = pca.transform(X)
    ax = plot_pca_label_points_3D(X_r, curr_y, label_names=morphlevels, cmap=morph_cmap, \
                                 ax=ax, annotate=True, markersize=100)
    
    
    fig.text(0.05, 0.1, 'expl. var %.2f' % np.sum(pca.explained_variance_ratio_))
    
    fig.suptitle('pca morph (sz=%i)' % curr_size)
    
    label_figure(fig, data_identifier)
    
    figname = 'pca_3d_morph_at_sz%i_ncomps%i' % (curr_size, n_components) 
    pl.savefig(os.path.join(curr_figdir, '%s.png' % figname))
    print figname
    
    pl.close()

#%%
    figname = 'pca_3d_morph_at_sz%i_ncomps%i_view2' % (curr_size, n_components) 
    pl.savefig(os.path.join(curr_figdir, '%s.png' % figname))
    print figname
    
    