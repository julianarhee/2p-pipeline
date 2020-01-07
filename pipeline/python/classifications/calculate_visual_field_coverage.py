#!/usr/bin/env python2
# coding: utf-8

# In[1]:


import matplotlib
matplotlib.use('agg')
import os
import sys
import glob
import json
import h5py
import copy
import cv2
import imutils
import itertools
import time
import optparse
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

from matplotlib.patches import Ellipse, Rectangle, Polygon
from shapely.geometry.point import Point
from shapely.geometry import box
from shapely import affinity

from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import MaxNLocator
from matplotlib.path import Path
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable


import matplotlib_venn as mpvenn

import multiprocessing as mp

#%%

# ############################################
# Functions for processing visual field coverage
# ############################################

from matplotlib.patches import Ellipse, Rectangle, Polygon
from shapely.geometry.point import Point
from shapely.geometry import box
from shapely import affinity
from shapely.ops import cascaded_union

def create_ellipse(center, lengths, angle=0):
    """
    create a shapely ellipse. adapted from
    https://gis.stackexchange.com/a/243462
    """
    circ = Point(center).buffer(1)
    ell = affinity.scale(circ, int(lengths[0]), int(lengths[1]))
    ellr = affinity.rotate(ell, angle)
    return ellr

def rfs_to_polys(rffits, sigma_scale=2.35):
    '''
    rffits (pd dataframe)
        index : roi indices (same as gdf.rois)
        columns : r2, sigma_x, sigma_y, theta, x0, y0 (already converted) 
        
    '''
    rf_polys=[]
    for roi in rffits.index.tolist():
        _, sx, sy, th, x0, y0 = rffits.loc[roi]
        s_ell = create_ellipse((x0, y0), (abs(sx)*sigma_scale, abs(sy)*sigma_scale), np.rad2deg(th))
        rf_polys.append(s_ell)
    return rf_polys

def stimsize_poly(sz, xpos=0, ypos=0):
    
    ry_min = ypos - sz/2.
    rx_min = xpos - sz/2.
    ry_max = ypos + sz/2.
    rx_max = xpos + sz/2.
    s_blobs = box(rx_min, ry_min, rx_max, ry_max)
    
    return s_blobs

def get_overlap_stats(roi_list, rf_polys, rf_dist_from_center, stiminfo):
    xpos = stiminfo['stimulus_xpos']
    ypos = stiminfo['stimulus_ypos']

    # Get size range of stimuli shown
    if 'gratings' in stiminfo['stimulus_sizes'].keys():
        gratings_sz = min(stiminfo['stimulus_sizes']['gratings'])
        print("Gratings - apertured size: %i" % gratings_sz)
    else:
        gratings_sz = None
        print("No localized gratings")
    blobs_sz_min = min(stiminfo['stimulus_sizes']['blobs'])
    blobs_sz_max = max(stiminfo['stimulus_sizes']['blobs'])
    print("Blobs - min/max size: (%i, %i)" % (blobs_sz_min, blobs_sz_max))

    # Create shapes for each stim size (bounding box)
    print("stimuli presented @:", xpos, ypos)
    stim_polys = [stimsize_poly(blob_sz, xpos=xpos, ypos=ypos)\
                  for blob_sz in stiminfo['stimulus_sizes']['blobs']]
    stim_labels = ['%i-deg' % blob_sz for blob_sz in stiminfo['stimulus_sizes']['blobs']]

    ## Caculate overlaps and put into dataframe
    overlaps=[]
    for s_label, s_poly in zip(stim_labels, stim_polys):
        tdf = pd.DataFrame({'overlap': [(s_ell.intersection(s_poly)).area / s_ell.area \
                                   for s_ell in rf_polys],
                            'distance': rf_dist_from_center,
                            'stimulus': [s_label for _ in range(len(rf_polys))],
                            'roi': roi_list})
        overlaps.append(tdf)

    if gratings_sz is not None:
        s_gratings = create_ellipse((xpos, ypos), (gratings_sz/2., gratings_sz/2.), 0)
        tdf = pd.DataFrame({'overlap': [(s_ell.intersection(s_gratings)).area / s_ell.area \
                                       for s_ell in rf_polys],
                            'distance': rf_dist_from_center,
                            'stimulus': ['gratings' for _ in range(len(rf_polys))],
                            'roi': roi_list})
        overlaps.append(tdf)

    overlap_df = pd.concat(overlaps)
    #overlap_df['color'] = [colordict[e] if e=='gratings' else 'cornflowerblue' for e in overlap_df['stimulus']]
    
    return overlap_df

# Session summary plots

def draw_rf_polys(rf_polys, ax=None):
    if ax is None:
        fig, ax = pl.subplots()
    rf_patches = [Polygon(np.array(rf_shape.exterior.coords.xy).T, 
                          edgecolor='k', alpha=0.5, facecolor='none', lw=0.2)\
                         for rf_shape in rf_polys]
    for rp in rf_patches:
        ax.add_patch(rp)
    return ax


def plot_overlap_distributions(overlap_df, ax=None):
    if ax is None:
        fig, ax = pl.subplots()

    ax = sns.boxplot(x="stimulus", y="overlap", data=overlap_df, ax=ax, color='k',
                     saturation=1.0, notch=True, boxprops=dict(alpha=.5))
    ax.set_xlabel('')
    ax.set_ylabel('Overlap area\n(% of RF)', fontsize=8)
    ax.tick_params(axis='both', which='both', length=0, labelsize=8)
    #ax.tick_params(axis='y', which='both', length=0, labelsize=8)
    sns.despine(trim=True, offset=4, ax=ax, bottom=True)
    return ax

def hist_rf_size(rf_avg_size, ax=None):
    if ax is None:
        fig, ax = pl.subplots()
    sns.distplot(rf_avg_size, ax=ax, color='k')
    ax.set_xlim([0, max(rf_avg_size)+10])
    ax.set_xlabel('Average RF size\n(deg)', fontsize=8)
    ax.set_ylabel('kde', fontsize=8)
    ax.tick_params(axis='both', which='both', length=3, labelsize=8)
    #ax2a.yaxis.set_major_locator(MaxNLocator(2))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    sns.despine(trim=True, offset=4, ax=ax)
    return ax

def hist_rf_dist(rf_dist_from_center, ax=None):
    if ax is None:
        fig, ax = pl.subplots()
    sns.distplot(rf_dist_from_center, ax=ax, color='k')
    ax.set_xlabel('RF distance from\nstimulus center', fontsize=8)
    ax.set_ylabel('kde', fontsize=8)
    ax.tick_params(axis='both', which='both', length=3, labelsize=8)
    #ymax = max([ax.get_ylim()[-1], ax.get_ylim()[-1]])
    #ax2b.set_ylim([0, ymax])
    #ax2b.yaxis.set_major_locator(MaxNLocator(2))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    sns.despine(trim=True, offset=4, ax=ax)
    return ax


def scatter_fovpos_rfpos(fovinfo, rffits, ax=None, axis='azimuth', 
                         xlim=58.78, ylim=33.66, units='um'):
    if ax is None:
        fig, ax = pl.subplots()
        
    if axis=='azimuth':
        ctx_pos = fovinfo['positions']['ml_pos']
        rf_pos = rffits['x0']
        axisname = 'Azimuth'
        miny = xlim*-1
        maxy = xlim
    else:
        ctx_pos = fovinfo['positions']['ap_pos']
        rf_pos = rffits['y0']
        axisname = 'Elevation'
        miny = ylim*-1
        maxy = ylim
        
    colors = ['k' for _ in range(len(ctx_pos))]
    ax.scatter(ctx_pos, rf_pos, c=colors, alpha=0.3) # FOV y-axis is left-right on brain
    ax.set_ylabel('%s\n(rel. deg.)' % axisname, fontsize=8)
    ax.set_xlabel('FOV position\n(%s)' % units, fontsize=8)
    ax.set_xticks(np.linspace(0, 1200, 5))
    ax.set_yticks(np.linspace(np.floor(miny), np.ceil(maxy), 5)) #[x0, x1])
    ax.yaxis.set_major_locator(MaxNLocator(5))

    return ax

def summarize_visual_field_coverage(gdfs, fovinfo, overlap_df, stiminfo,
                                    rf_polys, rf_avg_size, rf_dist_from_center,
                                    rf_exp_name='rfs'):

    # PLOT
    fig = pl.figure(figsize=(8,6))
    fig.patch.set_alpha(1)

    # Screen visualization ----------------------------------------------------
    ax0 = pl.subplot2grid((3, 4), (0, 0), colspan=2, rowspan=1)
    screen_bounds = stiminfo['screen_bounds']
    ax0.set_xlim([screen_bounds[1], screen_bounds[3]])
    ax0.set_ylim([screen_bounds[0], screen_bounds[2]])
    ax0.set_aspect(stiminfo['screen_aspect'])
    ax0.tick_params(axis='both', which='both', length=0, labelsize=6)

    # Draw receptive fields, calculate overlap(s):
    ax0 = draw_rf_polys(rf_polys, ax=ax0)

    xpos = stiminfo['stimulus_xpos']
    ypos = stiminfo['stimulus_ypos']

    # Draw stimulus size patches:
    stim_polys = [stimsize_poly(blob_sz, xpos=xpos, ypos=ypos)\
                  for blob_sz in stiminfo['stimulus_sizes']['blobs']]
    stim_labels = ['%i-deg' % sz for sz in stiminfo['stimulus_sizes']['blobs']]
    stim_patches = [Polygon(np.array(stim_shape.exterior.coords.xy).T, 
                            edgecolor='orange', alpha=0.5, 
                            lw=2, facecolor='none', label=stim_label)\
                            for stim_label, stim_shape in zip(stim_labels, stim_polys)]
    ax0.add_patch(stim_patches[0])
    ax0.add_patch(stim_patches[-1])

    # ---- Proportion of RF overlapping with stimulus bounds ----
    ax = pl.subplot2grid((3, 4), (0, 2), colspan=2, rowspan=1)
    ax = plot_overlap_distributions(overlap_df, ax=ax)

    # ---- Average RF size -----------------------
    ax2a = pl.subplot2grid((3, 4), (1, 2), colspan=1, rowspan=1)
    ax2a = hist_rf_size(rf_avg_size, ax=ax2a)

    # ---- Distance from stimulus center -----------------------
    ax2b = pl.subplot2grid((3, 4), (1, 3), colspan=1, rowspan=1)
    ax2b = hist_rf_dist(rf_dist_from_center, ax=ax2b)

    # ---- Spatially sorted ROIs vs. RF position -----------------------
    ax3a = pl.subplot2grid((3, 4), (1, 0), colspan=1, rowspan=1)
    ax3a = scatter_fovpos_rfpos(fovinfo, gdfs[rf_exp_name].fits, ax=ax3a, axis='azimuth')
    ax3b = pl.subplot2grid((3, 4), (1, 1), colspan=1, rowspan=1)
    ax3b = scatter_fovpos_rfpos(fovinfo, gdfs[rf_exp_name].fits, ax=ax3b, axis='elevation')

    # Adjust subplots
    pl.subplots_adjust(left=0.1, top=0.9, right=0.99, wspace=0.6, hspace=0.5)
    bbox_s = ax2b.get_position()
    bbox_s2 = [bbox_s.x0 - 0.01, bbox_s.y0,  bbox_s.width, bbox_s.height] 
    ax2b.set_position(bbox_s2) # set a new position

    # Move upper-left plot over to reduce white space
    bbox = ax0.get_position()
    bbox2 = [bbox.x0 - 0.04, bbox.y0+0.0,  bbox.width-0.04, bbox.height+0.05] 
    ax0.set_position(bbox2) # set a new position

    return fig




def get_session_object(animalid, session, fov, traceid='traces001', trace_type='corrected',
                       create_new=True, rootdir='/n/coxfs01/2p-data'):
    
    # Create output dir for session summary 
    summarydir = os.path.join(rootdir, animalid, session, fov, 'summaries')
    session_outfile = os.path.join(summarydir, 'sessiondata.pkl')
    
    # Load existing sessiondata file
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
    
    try:
        print("Found %i experiments in current session:" % len(S.experiment_list), S.experiment_list)
        assert 'rfs' in S.experiment_list or 'rfs10' in S.experiment_list, "ERROR:  No receptive field mapping found for current dataset: [%s|%s|%s]" % (S.animalid, S.session, S.fov)
    except Exception as e:
        print e
        return None
    return S


animalid = 'JC084' #JC076'
#session = '20190522' #'20190501'
#fov = 'FOV1_zoom2p0x'

def extract_options(options):
    
    parser = optparse.OptionParser()

    parser.add_option('-D', '--root', action='store', dest='rootdir', default='/n/coxfs01/2p-data',\
                      help='data root dir (root project dir containing all animalids) [default: /n/coxfs01/2p-data]')
    parser.add_option('-a', '--aggr', action='store', dest='aggregate_dir', default='/n/coxfs01/julianarhee/aggregate-visual-areas',\
                      help='data root dir (root project dir containing all animalids) [default: /n/coxfs01/julianarhee/aggregate-visual-areas')

    parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')

    # Set specific session/run for current animal:
    parser.add_option('-S', '--session', action='store', dest='session', default='', \
                      help='session dir (format: YYYMMDD_ANIMALID')
    parser.add_option('-A', '--fov', action='store', dest='fov', default='FOV1_zoom2p0x', \
                      help="acquisition folder (ex: 'FOV1_zoom3x') [default: FOV1]")
    parser.add_option('-t', '--traceid', action='store', dest='traceid', default='traces001', \
                      help="name of traces ID [default: traces001]")

    parser.add_option('-M', '--resp', action='store', dest='response_type', default='dff', \
                      help="Response metric to use for creating RF maps (default: dff)")
    (options, args) = parser.parse_args(options)

    return options


#%%

def main(options):
    opts = extract_options(options)


    # Set output dir:
    #aggregate_dir = '/n/coxfs01/julianarhee/aggregate-visual-areas'
    #outdir = os.path.join(aggregate_dir, 'receptive-fields', 'visual-field-coverage')
    outdir = opts.aggregate_dir
    print outdir
    if not os.path.exists(os.path.join(outdir, 'heatmaps')):
        os.makedirs(os.path.join(outdir, 'heatmaps'))
    if not os.path.exists(os.path.join(outdir, 'summaries')):
        os.makedirs(os.path.join(outdir, 'summaries'))


    # Response params
    traceid = opts.traceid #'traces001'
    response_type = opts.response_type #'dff'
    create_new = False
    trace_type = 'corrected'
    responsive_test = 'nstds'
    responsive_thr = 10.
    convert_um = True


    # RF conversion info
    sigma_scale = 2.35
    min_sigma=5
    max_sigma=50
    dx = dy = 1.0  # grid resolution; this can be adjusted


    # Get aggregate data:
    from pipeline.python.classifications import get_dataset_stats as gd
    import cPickle as pkl

    optsE = gd.extract_options(['-t', traceid])
    rootdir = optsE.rootdir
    aggregate_dir = optsE.aggregate_dir
    fov_type = optsE.fov_type
    traceid = optsE.traceid
    print aggregate_dir
    
    sdata_fpath = os.path.join(aggregate_dir, 'dataset_info.pkl')
    if os.path.exists(sdata_fpath):
        with open(sdata_fpath, 'rb') as f:
            sdata = pkl.load(f)
    else:
        sdata = gd.aggregate_session_info(traceid=optsE.traceid, trace_type=optsE.trace_type, 
                                           state=optsE.state, fov_type=optsE.fov_type, 
                                           visual_areas=optsE.visual_areas,
                                           blacklist=optsE.blacklist, 
                                           rootdir=optsE.rootdir)
        with open(sdata_fpath, 'wb') as f:
            pkl.dump(sdata, f, protocol=pkl.HIGHEST_PROTOCOL)


#

    animalid = opts.animalid
    fov = opts.fov
    session = opts.session
    visual_area = sdata[((sdata['animalid']==animalid) 
                        & (sdata['session']==session)
                        & (sdata['fov']==fov))]['visual_area'].unique()[0]
    print("VISUAL AREA: %s" % visual_area)
 
    S = get_session_object(animalid, session, fov, traceid=traceid, trace_type=trace_type,
                       create_new=True, rootdir=rootdir)
    # Get Receptive Field measures:
    rf_exp_name = 'rfs10' if 'rfs10' in S.experiment_list else 'rfs'
    print(rf_exp_name)

    # Get grouped roi stat metrics:
    gdfs, statsdir, stats_desc, nostats = rstats.get_session_stats(S, response_type=response_type, 
                                                      experiment_list=S.experiment_list,
                                                      responsive_test=responsive_test, 
                                                      responsive_thr=responsive_thr,
                                                      traceid=traceid, trace_type=trace_type,
                                                      create_new=True, rootdir=rootdir,
                                                      pretty_plots=False, update_self=True)
    roi_list = gdfs[rf_exp_name].rois
    data_identifier = '|'.join([S.animalid, S.session, S.fov, S.traceid, S.rois])
    data_identifier

    # Get stimulus info for RFs
    row_vals = gdfs[rf_exp_name].fitinfo['row_vals']
    col_vals = gdfs[rf_exp_name].fitinfo['col_vals']
    xres = np.unique(np.diff(row_vals))[0]
    yres = np.unique(np.diff(col_vals))[0]
    print("x-/y-res: %i, %i" % (xres, yres))

    # Identify stimulus location for current session
    xpos, ypos = S.get_stimulus_coordinates()

    # Get screen bounds [bottom left upper right]
    screen_bounds = [S.screen['linminH'], S.screen['linminW'], S.screen['linmaxH'], S.screen['linmaxW']]
    screen_aspect = S.screen['resolution'][0] / S.screen['resolution'][1]

    stiminfo = {'stimulus_sizes': S.get_stimulus_sizes(),
                'screen_bounds': screen_bounds,
                'screen_aspect': screen_aspect,
                'stimulus_xpos': xpos,
                'stimulus_ypos': ypos}

    # Get FOV info
    masks, zimg = S.load_masks(rois='rois001')
    rf_rois = gdfs[rf_exp_name].fits.index.tolist() 
    fovinfo = coor.get_roi_fov_info(masks, zimg, roi_list=rf_rois)
    print("... got FOV info.")

    # Convert RF params into shape
    rf_polys = rfs_to_polys(gdfs[rf_exp_name].fits)

    # Vectorize shapes
    miny, minx, maxy, maxx = screen_bounds #[-Y, -X, +Y, +X]

    # Create vertex coordinates for each grid cell...
    xx = np.arange(minx, maxx, dx)
    yy = np.arange(miny, maxy, dy)
    nx = len(xx)
    ny = len(yy)
    x, y = np.meshgrid(xx, yy)
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x,y)).T

    path = Path(np.array(rf_polys[0].exterior.coords.xy).T)
    grid = path.contains_points(points).astype(int)

    for rp in rf_polys[1:]:
        poly_verts = np.array(rp.exterior.coords.xy).T
        path = Path(poly_verts)
        grid += path.contains_points(points).astype(int)
    grid = grid.reshape((ny,nx))

    # Normalize
    #grid = grid / float(len(rf_polys))

    # Plot
    fig, ax = pl.subplots()
    im = ax.imshow(grid, extent=(minx, maxx, miny, maxy), origin='lower', cmap='bone')
    ax_divider = make_axes_locatable(ax)
    # add an axes to the right of the main axes.
    cax = ax_divider.append_axes("right", size="5%", pad="2%")
    cb1 = pl.colorbar(im, cax=cax)
    ax.plot(xpos, ypos, 'r*')
    label_figure(fig, data_identifier)
    pl.savefig(os.path.join(outdir, 'heatmaps', '%s_%s-%s-%s_RF-heatmap.png' % (visual_area, animalid, session, fov)))
    pl.close()

    # Calculate and visualize overlap stats
    rf_avg_size = np.mean([abs(gdfs[rf_exp_name].fits['sigma_x'])*sigma_scale, \
                         abs(gdfs[rf_exp_name].fits['sigma_y'])*sigma_scale], axis=0)
    rf_dist_from_center = np.sqrt((gdfs[rf_exp_name].fits['x0'] - xpos)**2 \
                                + (gdfs[rf_exp_name].fits['y0'] - ypos)**2)

    overlap_df = get_overlap_stats(roi_list, rf_polys, rf_dist_from_center, stiminfo)
    fig = summarize_visual_field_coverage(gdfs, fovinfo, overlap_df, stiminfo,
                                    rf_polys, rf_avg_size, rf_dist_from_center, rf_exp_name=rf_exp_name)
    label_figure(fig, data_identifier)
    pl.savefig(os.path.join(outdir, 'summaries','%s_%s-%s-%s_VF-coverage-stats.png' % (visual_area, animalid, session, fov)))
    pl.close()



    print("*** done! ***")

        
if __name__ == '__main__':
    main(sys.argv[1:])


