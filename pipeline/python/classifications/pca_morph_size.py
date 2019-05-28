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


import matplotlib.collections as mcoll
import matplotlib.path as mpath
from mpl_toolkits.mplot3d.art3d import Line3DCollection

def colorline(x, y, z=None, spacing=None, cmap=pl.get_cmap('rainbow'), 
              #norm=pl.Normalize(0.0, 1.0),
              linewidth=3, alpha=1.0, ax = None):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if spacing is None:
        spacing = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(spacing, "__iter__"):  # to check for numerical input -- this is a hack
        spacing = np.array([spacing])

    spacing = np.asarray(spacing)

    
    segments = make_segments(x, y, z=z)
    if z is not None:
        lc = Line3DCollection(segments, cmap=cmap, #norm=norm, 
                              linewidth=linewidth, alpha=alpha)
    else:
        lc = mcoll.LineCollection(segments, array=spacing, cmap=cmap, #norm=norm,
                              linewidth=linewidth, alpha=alpha)

    lc.set_array(spacing)
    
    if ax is None:
        ax = pl.gca()
        
    if z is not None:
        ax.add_collection3d(lc)
    else:
        ax.add_collection(lc)

    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    if z is not None:
        ax.set_zlim(z.min(), z.max())
    
    return lc


def make_segments(x, y, z=None):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """
    if z is not None:
        points = np.array([x, y, z]).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
    else:
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


#%%

def plot_pca_label_points(X_r, y_labels, label_names=[], label_colors=[], ax=None,\
                          cmap = 'jet', plot_markers=True, color_connections=False, 
                          connect_all=False, connect_within=False, markersize=100, lw=1):
    
    if ax is None:
        print "Creating new axis"
        fig, ax = pl.subplots()

    # Get target names
    target_names = sorted(np.unique(y_labels))
    
    if label_names is None:
        label_names = copy.copy(target_names)
        
    # Create default cmap if none provided:
    if label_colors is None:
        label_colors = ['k' for _ in range(len(target_names))]
    
    # Plot:
    if connect_all:
        if color_connections:
            x = X_r[:, 0] #np.random.rand(N)
            y = X_r[:, 1] #np.random.rand(N)    
            path = mpath.Path(np.column_stack([x, y]))
            verts = path.interpolated(steps=3).vertices
            x, y = verts[:, 0], verts[:, 1]
            colorline(x, y, spacing=y_labels, cmap=cmap, linewidth=lw, ax=ax)
        else:
            pc1s = []; pc2s = [];
            for color, i, targname in zip(label_colors, label_names, target_names):
                pc1s.extend(X_r[y_labels == i, 0])
                pc2s.extend(X_r[y_labels == i, 1])
            ax.plot(pc1s, pc2s, zorder=1, color='k', lw=lw) 
    

    for color, i, target_name in zip(label_colors, label_names, target_names):
        if connect_within:
            ax.plot(X_r[y_labels == i, 0], X_r[y_labels == i, 1], zorder=2, 
                        color=color, label=target_name,
                        alpha=.8, lw=lw, marker='o', markersize=markersize) #abel=target_name)        
        if plot_markers:
            ax.scatter(X_r[y_labels == i, 0], X_r[y_labels == i, 1], zorder=2,
                       s=markersize,
                       color=color, label=target_name,
                       alpha=.8, lw=lw, edgecolor='k') #abel=target_name)        
    
    
#    # Plot:
#    for color, i, target_name in zip(label_colors, label_names, target_names):
#        ax.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, label=target_name,
#                   alpha=.8, lw=lw, edgecolor='k') #abel=target_name)        
    
    # Clean up axes:
    xlim = max([abs(xl) for xl in ax.get_xlim()])
    ylim = max([abs(yl) for yl in ax.get_ylim()])
    ax.set_xlim([-xlim, xlim])
    ax.set_ylim([-ylim, ylim])
    ax.xaxis.set_major_locator(MaxNLocator(3, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(3, integer=True))
    
    return ax

#%%
def plot_pca_label_points_3D(X_r, y_labels, label_names=[], label_colors=[], 
                             cmap='jet',
                             plot_markers=True, color_connections=False,
                             connect_all=False, connect_within=False,
                             ax=None, annotate=False, markersize=100, lw=1, alpha=1):
    
    if ax is None:
        print "Creating new axis"
        fig = pl.figure(1, figsize=(8, 6))
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=50, azim=50)

    # Get target names
    target_names = sorted(np.unique(y_labels))
    
    if label_names is None:
        label_names = copy.copy(target_names)
    
    # Create default cmap if none provided:
    if len(label_colors) == 0:
        label_colors = ['k' for _ in range(len(target_names))]
        
    if annotate:
        for name, label in zip(label_names, target_names):
            ax.text3D(X_r[y_labels == label, 0].mean(),
                      X_r[y_labels == label, 1].mean(),
                      X_r[y_labels == label, 2].mean(),
                      str(name),
                      horizontalalignment='center',
                      bbox=dict(alpha=.7, edgecolor='w', facecolor='w'))

    print  "Labeling..."
    print label_names
    # Plot:
    if connect_all:
        if color_connections:
            #x = X_r[:, 0] #np.random.rand(N)
            #y = X_r[:, 1] #np.random.rand(N)
            #z = X_r[:, 2] #np.random.rand(N)
            
            x = []; y = []; z = [];
            for i in label_names:
                x.extend(X_r[y_labels == i, 0])
                y.extend(X_r[y_labels == i, 1])
                z.extend(X_r[y_labels == i, 2])
            
            verts = np.array([list(i) for i in zip(x,y,z)])
            x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
            colorline(x, y, z, spacing=y_labels, cmap=cmap, linewidth=lw, ax=ax)
        else:
            pc1s = []; pc2s = []; pc3s = [];
            for i in label_names:
                pc1s.extend(X_r[y_labels == i, 0])
                pc2s.extend(X_r[y_labels == i, 1])
                pc3s.extend(X_r[y_labels == i, 2])
            ax.plot(pc1s, pc2s, pc3s, zorder=1, color='k', lw=lw) 

    for color, i, target_name in zip(label_colors, label_names, target_names):
        if connect_within:
            ax.plot(X_r[y_labels == i, 0], X_r[y_labels == i, 1], X_r[y_labels == i, 2],
                        zorder=2, 
                        color=color, label=target_name,
                        alpha=.8, lw=lw, marker='o', markersize=markersize)       
        if plot_markers:
            ax.scatter(X_r[y_labels == i, 0], X_r[y_labels == i, 1], X_r[y_labels == i, 2], 
                       zorder=2,
                       s=markersize,
                       color=color, label=target_name,
                       alpha=alpha, lw=lw, edgecolor=color) #abel=target_name)        


    # Reorder the labels to have colors matching the cluster results
#    ax.scatter(X_r[:, 0], X_r[:, 1], X_r[:, 2], c=y, cmap=cmap, s=np.ones(y.shape)*markersize,
#               edgecolor='w')
    
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
#
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


rootdir = '/n/coxfs01/2p-data'
animalid = 'JC076' 
session = '20190502' #'20190319'
fov = 'FOV1_zoom2p0x' 
run = 'combined_blobs_static'
traceid = 'traces001' #'traces002'

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
aspect_ratio = 1 #1.747
sdf = pd.DataFrame(dset['sconfigs'][()]).T

if 'color' in sdf.columns:
    sdf = sdf[sdf['color']=='']
sdf['size'] = [round(sz/aspect_ratio, 1) for sz in sdf['size']]
sdf.head()


# Only take subset of trials where image shown (not controls):
labels = pd.DataFrame(data=dset['labels_data'], columns=dset['labels_columns'])
labels = labels[labels['config'].isin(sdf.index.tolist())]
#traces = traces[labels.index.tolist(), :]
trial_ixs = np.array([int(t[5:])-1 for t in sorted(labels['trial'].unique(), key=natural_keys)])
#zscores = zscores[trial_ixs, :]

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
    curr_zscored_traces = pd.DataFrame(curr_traces).divide(bas_std, axis='columns')# pd.DataFrame(curr_traces, index=tmat.index).divide(bas_std, axis='columns')
    zscored_traces_list.append(curr_zscored_traces)

zscored_traces = pd.concat(zscored_traces_list, axis=0).reset_index()
zscored_traces.head()

traces = traces[labels.index.tolist(), :]
raw_traces = pd.DataFrame(traces, index=zscored_traces.index)
raw_traces.head()
    
#%%

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
                                    for cfg in sorted(zscores_by_cond.keys(), key=natural_keys)],\
                                    index=[int(cf[6:])-1 for cf in sorted(zscores_by_cond.keys(), key=natural_keys)]) # nconfigs x nrois
    
avg_zscores_by_cond_GM = pd.DataFrame([zscores_by_cond_GM[cfg].mean(axis=0) \
                                    for cfg in sorted(zscores_by_cond_GM.keys(), key=natural_keys)],\
                                    index=[int(cf[6:])-1 for cf in sorted(zscores_by_cond.keys(), key=natural_keys)]) # nconfigs x nrois

# nconfigs x nrois

    
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
  

sizes = sorted([int(round(s)) for s in sdf['size'].unique()])
morphlevels = sorted(sdf['morphlevel'].unique())


size_colors = sns.cubehelix_palette(len(sizes))
size_cmap = sns.cubehelix_palette(as_cmap=True, rot=0.4, hue=1)
morph_colors = sns.diverging_palette(220, 20, n=len(morphlevels))
morph_cmap = sns.diverging_palette(220, 20, as_cmap=True)

    
    

#%%

roi_selector = 'visual'

if roi_selector == 'visual':
    roi_list = copy.copy(sorted_visual)
elif roi_selector == 'selective':
    roi_list = copy.copy(sorted_selective)  
    
    
fig_subdir = 'pca_%s' % roi_selector 

if segment:
    curr_figdir = os.path.join(traceid_dir, 'figures', 'population', fig_subdir, visual_area)
else:
    curr_figdir = os.path.join(traceid_dir, 'figures', 'population', fig_subdir)
if not os.path.exists(curr_figdir):
    os.makedirs(curr_figdir)
print "Saving plots to: %s" % curr_figdir



#%%

# -----------------------------------------------------------------------------
# 2D
# -----------------------------------------------------------------------------


#subtract_GM = True
#n_components=2

# PCA data:
#if subtract_GM:
#    X = avg_zscores_by_cond_GM[selective_rois]
#else:
#    X = avg_zscores_by_cond[selective_rois]

X = avg_zscores_by_cond[roi_list[0:30]]
print X.shape

X_std = X - X.mean()


# 
num_obs, num_vars = X_std.shape
print "N obs: %i, N vars: %i" % (num_obs, num_vars)

U, S, V = np.linalg.svd(X_std) 
eigvals = S**2 / np.sum(S**2)  # NOTE (@amoeba): These are not PCA eigenvalues. 
                               # This question is about SVD.

fig = pl.figure(figsize=(8,5))
sing_vals = np.arange(num_vars) + 1
pl.plot(sing_vals, eigvals, 'bo-', linewidth=1)
pl.title('Scree Plot')
pl.xlabel('Principal Component')
pl.ylabel('Eigenvalue')
pl.legend(['Eigenvalues from SVD'], loc='best', borderpad=0.3, markerscale=0.4)

pl.savefig(os.path.join(curr_figdir, 'skree_nobs_%i_nvars_%i.png' % (num_obs, num_vars)))

                
#%%


n_components= 2

connect_all = True
color_connections = False
plot_markers = True
connect_within=False
markersize = 50
lw = 0.5
annotate = True

pca = sk.decomposition.PCA(n_components=n_components)
X_r = pca.fit(X_std).transform(X_std)

#y = np.array([sdf['size']['config%03d' % int(cix+1)] for cix in avg_zscores_by_cond_GM.index.tolist()])
label_names =  copy.copy(sizes)
label_colors = copy.copy(size_colors)

fig, axes = pl.subplots(1,2, figsize=(8,5))
fig.subplots_adjust(top=0.8, bottom=0.3, wspace=0.2, hspace=0.2, left=0.1)

y_size = np.array([sdf['size']['config%03d' % int(cix+1)] for cix in avg_zscores_by_cond_GM.index.tolist()]) #np.array([sdf['size'][cfg] for cfg in sdf.index.tolist()])
ax = axes[0]
ax = plot_pca_label_points(X_r, y_size, label_names=sizes, label_colors=size_colors, ax=ax,
                           connect_all=connect_all, connect_within=connect_within, color_connections=color_connections,
                           markersize=markersize, lw=lw, cmap=size_cmap)

ax.text(ax.get_xlim()[0], ax.get_ylim()[-1]*1.02, \
        'expl. var. %.2f' % np.sum(pca.explained_variance_ratio_), fontsize=6)
# Add legend:
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2),
          ncol=len(sizes), fancybox=False, shadow=False, fontsize=6)
fig.text(0.25, 0.15 , 'size')



# Label MORPH LEVEL:
y_morph = np.array([sdf['morphlevel']['config%03d' % int(cix+1)] for cix in avg_zscores_by_cond_GM.index.tolist()]) #np.array([sdf['morphlevel'][cfg] for cfg in sdf.index.tolist()])

# Plot
ax = axes[1]
ax = plot_pca_label_points(X_r, y_morph, label_names=morphlevels, label_colors=morph_colors, ax=ax, 
                           connect_all=connect_all, connect_within=connect_within, color_connections=color_connections,
                           markersize=markersize, lw=lw, cmap=morph_cmap)


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

#%

figname = 'pca_%icomps_averaged_condns_morph_size_%iobs_%ivars' % (n_components, num_obs, num_vars)
pl.savefig(os.path.join(curr_figdir, '%s.png' % figname))

print figname



#%%
n_components=3
pca = sk.decomposition.PCA(n_components=n_components)
pca.fit(X_std)
X_r = pca.transform(X_std)

splitter = 'size'

if splitter == 'size':
    elev = -166 #62 #25 #13 #10 # 40 #2# 83.4
    azim = -65 #-126 #35 #-60 #-63 #-130 #135
    split_values = copy.copy(sizes)
    labeler = 'morphlevel'
    curr_cmap = morph_cmap
    curr_colors = morph_colors
    figheight = 4
    
elif splitter == 'morphlevel':
    elev = 63 #62 #71 #43 #35 # 77 #35
    azim = 22 #-126 #46 #73 #35 #121
    split_values = copy.copy(morphlevels)
    labeler = 'size'
    curr_cmap = size_cmap
    curr_colors = size_colors
    figheight = 2
    
fig = pl.figure(figsize=(25, figheight))

cfg_start_ix = int(sdf.index.tolist()[0][6:]) - 1
for size_ix, curr_val in enumerate(sorted(split_values)):
    
    curr_cfg_ixs = np.array([int(cf[6:])-1 for cf in sdf[sdf[splitter]==curr_val].index.tolist()])
    #curr_zscores =  avg_zscores_by_cond_GM[avg_zscores_by_cond_GM.index.isin(curr_cfg_ixs)][selective_rois]
    #print curr_zscores.shape
    curr_cfg_ixs_adj = [ci - cfg_start_ix for ci in curr_cfg_ixs]
    curr_xs = X_r[curr_cfg_ixs_adj, :]
    curr_ys = np.array([sdf[labeler]['config%03d' % int(cix+1)] for cix in curr_cfg_ixs]) 

    #ax = Axes3D(fig, elev=elev_view3, azim=azim_view3)
    ax = fig.add_subplot(1, len(split_values), size_ix+1, projection='3d', azim=azim, elev=elev)
    
    ax = plot_pca_label_points_3D(curr_xs, curr_ys, ax=ax,
                                  label_names=curr_ys, label_colors=curr_colors, cmap=curr_cmap,
                                  connect_all=connect_all, color_connections=color_connections,
                                  plot_markers=plot_markers, annotate=False, markersize=markersize, lw=lw)
    ax.set_title(curr_val)
    
fig.text(0.2, 0.1, 'expl. var %.2f' % np.sum(pca.explained_variance_ratio_))
fig.suptitle(splitter)

pl.subplots_adjust(top=0.8)
label_figure(fig, data_identifier)             
figname = 'pca_%icomps_split_by_%s_label_%s_%iobs_%ivars_view2' % (n_components, splitter, labeler, num_obs, num_vars)
print figname
pl.savefig(os.path.join(curr_figdir, '%s.png' % figname))



#%%

# -----------------------------------------------------------------------------
# 3D
# -----------------------------------------------------------------------------

connect_all = True
color_connections = False
plot_markers = True
connect_within=False
markersize = 30
lw = 1
annotate = True


azim_view1 = -90 #30
elev_view1 = 30

    
azim_view1 = -113 #30
elev_view1 = 34


# Label SIZE:

#X = avg_zscores_by_cond_GM[selective_rois]
#y = np.array([int(sdf['size'][cfg]) for cfg in sdf.index.tolist()])

n_components=3
pca = sk.decomposition.PCA(n_components=n_components)
pca.fit(X)
X_r = pca.transform(X)

fig = pl.figure(figsize=(12,8))
#ax = Axes3D(fig, rect=[0, 0, .6, 1], elev=elev_view3, azim=azim_view3)
ax1 = fig.add_subplot(1, 2, 1, projection='3d', azim=azim_view1, elev=elev_view1)

ax1 = plot_pca_label_points_3D(X_r, y_size, ax=ax1,
                               label_names=sizes, label_colors=size_colors, cmap=size_cmap, 
                              connect_all=connect_all, color_connections=False,
                              plot_markers=plot_markers, annotate=annotate, markersize=markersize, lw=1)
fig.text(0.2, 0.1, 'expl. var %.2f' % np.sum(pca.explained_variance_ratio_))
ax1.set_title('size labels')               


ax2 = fig.add_subplot(1, 2, 2, projection='3d', azim=azim_view1, elev=elev_view1)
ax2 = plot_pca_label_points_3D(X_r, y_morph, ax=ax2,
                              label_names=morphlevels, label_colors=morph_colors, cmap=morph_cmap,
                              connect_all=connect_all, color_connections=False,
                              plot_markers=plot_markers, annotate=annotate, markersize=markersize, lw=1)
ax2.set_title('morph labels')               
fig.text(0.7, 0.1, 'expl. var %.2f' % np.sum(pca.explained_variance_ratio_))
pl.subplots_adjust(wspace=0.00, left=0., top=0.8, right=0.95)


fig.suptitle('pca size (avg per cond)', y=0.95)
label_figure(fig, data_identifier)

#%
figname = 'pca_%icomps_averaged_condns_morph_size_%iobs_%ivars' % (n_components, num_obs, num_vars)

pl.savefig(os.path.join(curr_figdir, '%s.png' % figname))

print figname
#pl.close()

#%%
azim_view2 = 130
elev_view2 = 50

ax1.azim = azim_view2
ax1.elev = elev_view2

ax2.azim = azim_view2
ax2.elev = elev_view2

figname = 'pca_%icomps_averaged_condns_morph_size_%iobs_%ivars_view2' % (n_components, num_obs, num_vars)
pl.savefig(os.path.join(curr_figdir, '%s.png' % figname))


#%%

azim_view3 = -100
elev_view3 = 35

ax1.azim = azim_view3
ax1.elev = elev_view3

ax2.azim = azim_view3
ax2.elev = elev_view3


figname = 'pca_%icomps_averaged_condns_morph_size_%iobs_%ivars_view3' % (n_components, num_obs, num_vars)
pl.savefig(os.path.join(curr_figdir, '%s.png' % figname))
print figname


#%%
azim_view4 = -100
elev_view4 = 5

azim_view4 = -112 #-20
elev_view4 = 64 #5

ax1.azim = azim_view4
ax1.elev = elev_view4

ax2.azim = azim_view4
ax2.elev = elev_view4


figname = 'pca_%icomps_averaged_condns_morph_size_%iobs_%ivars_view5' % (n_components, num_obs, num_vars)
pl.savefig(os.path.join(curr_figdir, '%s.png' % figname))
print figname




#%%

azim_view4 = -108 #150
elev_view4 = 7

fig = pl.figure(figsize=(12, 6))
ax1 = fig.add_subplot(1, 2, 1, projection='3d', azim=azim_view4, elev=elev_view4)
ax1 = plot_pca_3d_primary_secondary(X_r, y_size, y_morph, ax=ax1, label_colors=size_colors, cmap=size_cmap,
                                   annotate=False, markersize=markersize, lw=.5, alpha=0.8)

ax2 = fig.add_subplot(1, 2, 2, projection='3d', azim=azim_view4, elev=elev_view4)
ax2 = plot_pca_3d_primary_secondary(X_r, y_morph, y_size, ax=ax2, label_colors=morph_colors, cmap=morph_cmap,
                                   annotate=False, markersize=markersize, lw=.5, alpha=0.8)


# Create colorbar for morphlevels:
cbar_axes = [0.45, 0.25, 0.01, 0.2]
cbar = create_mappable_cbar(size_cmap, sizes, cbar_title='size', orientation='vertical', cbar_axes=cbar_axes)

# Create colorbar for morphlevels:
cbar_axes = [0.9, 0.25, 0.01, 0.2]
cbar = create_mappable_cbar(morph_cmap, morphlevels, cbar_title='morph', orientation='vertical', cbar_axes=cbar_axes)

pl.subplots_adjust(wspace=0.1, left=0.05, top=0.8)
fig.suptitle('grouped labels (nc=%i, exp.var=%.2f)' % (n_components, np.sum(pca.explained_variance_ratio_)))
label_figure(fig, data_identifier)

figname = 'grouped_pca_%icomps_averaged_condns_%iobs_%ivars_view2' % (n_components, num_obs, num_vars)
pl.savefig(os.path.join(curr_figdir, '%s.png' % figname))
print figname



def create_mappable_cbar(colormap, colorlabels, cbar_title='', orientation='horizontal',
                         cbar_axes = [0.58, 0.22, 0.3, 0.02]):
    ncolors = len(colorlabels)
    bounds = np.arange(0, ncolors)
    norm = BoundaryNorm(bounds, colormap.N)
    mappable = cm.ScalarMappable(cmap=colormap)
    mappable.set_array(bounds)
    
    cbar_ax = fig.add_axes(cbar_axes)
    cbar = fig.colorbar(mappable, cax=cbar_ax, boundaries=np.arange(-0.5, ncolors, 1), \
                        ticks=bounds, norm=norm, orientation=orientation)
    
    cbar.ax.tick_params(axis='both', which='both',length=0)
    if orientation == 'horizontal':
        cbar.ax.set_xticklabels(colorlabels, fontsize=6) #(['%i' % i for i in morphlevels])  # horizontal colorbar
    else:
        cbar.ax.set_yticklabels(colorlabels, fontsize=6) #(['%i' % i for i in morphlevels])  # horizontal colorbar
    cbar.ax.set_xlabel('%s' % cbar_title, fontsize=10)
    
    return cbar



def plot_pca_3d_primary_secondary(X_r, y_labels, y_labels2, label_colors=[], cmap='jet',
                                  ax=None, annotate=False, markersize=100, lw=1, alpha=1, edgecolor='k'):
    
    if ax is None:
        print "Creating new axis"
        fig = pl.figure(1, figsize=(8, 6))
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=50, azim=50)
    
    label_names = sorted(np.unique(y_labels))
    target_names = copy.copy(label_names)
        
    if annotate:
        for name, label in zip(label_names, target_names):
            ax.text3D(X_r[y_labels == label, 0].mean(),
                      X_r[y_labels == label, 1].mean(),
                      X_r[y_labels == label, 2].mean(),
                      str(name),
                      horizontalalignment='center',
                      bbox=dict(alpha=.7, edgecolor='w', facecolor='w'))

    
    # Connect points based on secondary label:
    label_names2 = sorted(np.unique(y_labels2))
    pc1s = []; pc2s = []; pc3s = [];
    for i in label_names2:
        pc1s = X_r[y_labels2 == i, 0]
        pc2s = X_r[y_labels2 == i, 1]
        pc3s = X_r[y_labels2 == i, 2]
        ax.plot(pc1s, pc2s, pc3s, zorder=1, color='k', lw=lw) 
                
    # PLot points and color by primary label:
    for color, i, target_name in zip(label_colors, label_names, target_names):
        ax.scatter(X_r[y_labels == i, 0], X_r[y_labels == i, 1], X_r[y_labels == i, 2], 
                   zorder=2,
                   s=markersize,
                   color=color, label=target_name,
                   alpha=alpha, lw=lw, edgecolor=edgecolor) #abel=target_name)        

    return ax





#%%

# PCA data:
X = pd.DataFrame(zscores[:, roi_list[0:30]])
print X.shape

X_std = X - X.mean()


# 
num_obs, num_vars = X_std.shape
print "N obs: %i, N vars: %i" % (num_obs, num_vars)

U, S, V = np.linalg.svd(X_std) 
eigvals = S**2 / np.sum(S**2)  # NOTE (@amoeba): These are not PCA eigenvalues. 
                               # This question is about SVD.

fig = pl.figure(figsize=(8,5))
sing_vals = np.arange(num_vars) + 1
pl.plot(sing_vals, eigvals, 'bo-', linewidth=1)
pl.title('Scree Plot')
pl.xlabel('Principal Component')
pl.ylabel('Eigenvalue')
pl.legend(['Eigenvalues from SVD'], loc='best', borderpad=0.3, markerscale=0.4)
                
pl.savefig(os.path.join(curr_figdir, 'skree_nobs_%i_nvars_%i_alltrials.png' % (num_obs, num_vars)))
     


#%%
#sns.distplot(X_std.values.ravel())

n_components=5
pca = sk.decomposition.PCA(n_components=n_components)
pca.fit(X_std)
X_r = pca.transform(X_std)

markersize=10
    
splitter = 'morphlevel'

if splitter == 'size':
    elev = 77 #83.4
    azim = 43 #135
    split_values = copy.copy(sizes)
    labeler = 'morphlevel'
    curr_cmap = morph_cmap
    curr_colors = morph_colors
    figheight = 4
    
elif splitter == 'morphlevel':
    elev = 77 #35
    azim = 121
    split_values = copy.copy(morphlevels)
    labeler = 'size'
    curr_cmap = size_cmap
    curr_colors = size_colors
    figheight = 3

    
fig = pl.figure(figsize=(25, figheight))
for val_ix, curr_val in enumerate(sorted(split_values)):

    ax = fig.add_subplot(1, len(split_values), val_ix+1, projection='3d', azim=azim, elev=elev)

    curr_cfgs = sdf[sdf[splitter]==curr_val][labeler].index.tolist()
    
    curr_trial_ixs = np.array(sorted([int(trial[5:])-1 for trial in labels[labels['config'].isin(curr_cfgs)]['trial'].unique()]))
    curr_cfg_ixs = np.array([int(cfg[6:])-1 for cfg in curr_cfgs])
    
    #curr_zs = #zscores_gm[:, selective_rois]
    curr_x = X_r[curr_trial_ixs, :]
    
    curr_trials = ['trial%05d' % int(tix+1) for tix in curr_trial_ixs]
    curr_ys = np.array([sdf[labeler][labels[labels['trial']==trial]['config'].unique()[0]] for trial in curr_trials ])
    
    ax = plot_pca_label_points_3D(curr_x, curr_ys, ax=ax,
                              label_names=sdf[labeler].unique(), label_colors=curr_colors, cmap=curr_cmap,
                              connect_all=False, color_connections=False,
                              plot_markers=plot_markers, annotate=False, markersize=markersize, alpha=0.8)

    ax.set_title("%i" % (curr_val), fontsize=12)

label_figure(fig, data_identifier)

pl.subplots_adjust(top=0.8)
pl.suptitle("%s (exp. var. %.2f)" % (splitter, np.sum(pca.explained_variance_ratio_)))

figname = 'pca_%icomps_split_by_%s_label_%s_%iobs_%ivars_view1' % (n_components, splitter, labeler, num_obs, num_vars)
pl.savefig(os.path.join(curr_figdir, '%s.png' % figname))

print figname

#pl.close()



    
