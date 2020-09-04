#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 12:06:11 2019

@author: julianarhee
"""

#%%
import sys
import os
import glob
import json
import cv2
import optparse

import cPickle as pkl
import numpy as np
import pylab as pl
import tifffile as tf

from scipy import stats
from skimage.measure import block_reduce


from mpl_toolkits.axes_grid1 import make_axes_locatable

from pipeline.python.retinotopy import utils as rutils
from pipeline.python.utils import natural_keys, label_figure, get_screen_dims

from pipeline.python.coregistration import align_fov as coreg

import matplotlib as mpl
import matplotlib.cm as cmx


#%%


def filter_map_by_magratio(magratio, phase, trials_by_cond, use_cont=False,
                            dims=(512, 512), ds_factor=2, cond='right', 
                            mag_thr=None, mag_perc=0.05):
    if mag_thr is None:
        mag_thr = magratio.max().max()*mag_perc
        
    currmags = magratio[trials_by_cond[cond]]
    currmags[currmags<mag_thr] = np.nan
    currmags_mean = np.nanmean(currmags, axis=1)
    #d1 = int(np.sqrt(currmags_mean.shape[0]))
    d1 = dims[0] / ds_factor
    d2 = dims[1] / ds_factor
    currmags_map = np.reshape(currmags_mean, (d1, d2))
    

    currphase = phase[trials_by_cond[cond]]
    currphase_mean = stats.circmean(currphase, low=-np.pi, high=np.pi, axis=1)
    currphase_mean_c = rutils.correct_phase_wrap(currphase_mean)

    currphase_mean_c[np.isnan(currmags_mean)] = np.nan
    currphase_map_c = np.reshape(currphase_mean_c, (d1, d2))
    
    return currmags_map, currphase_map_c, mag_thr


def plot_filtered_maps(cond, currmags_map, currphase_map_c, mag_thr):
    fig, axes = pl.subplots(1, 2) #pl.figure()
    im = axes[0].imshow(currmags_map)
    divider = make_axes_locatable(axes[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    
    im2 = axes[1].imshow(currphase_map_c, cmap='nipy_spectral', vmin=0, vmax=2*np.pi)
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im2, cax=cax, orientation='vertical')
    
    pl.subplots_adjust(wspace=0.5)
    fig.suptitle('%s (mag_thr: %.4f)' % (cond, mag_thr))

    return fig


def make_continuous(mapvals):
    map_c = mapvals.copy()
    map_c = -1*map_c
    map_c = map_c % (2*np.pi)
    return map_c


#%%
def plot_maps(absolute_az, absolute_el, surface_img=None, elev_cutoff=None,
        cmap='nipy_spectral', vmin=0, vmax=2*np.pi):

    fig, axes = pl.subplots(1,2)
    ax = axes[0]
    if surface_img is not None:
        ax.imshow(surface_img, cmap='gray', origin='upper')
    im1 = ax.imshow(absolute_az, cmap='nipy_spectral', vmin=vmin, vmax=vmax, 
                    alpha=0.7, origin='upper')

    ax = axes[1]
    if surface_img is not None:
        ax.imshow(surface_img, cmap='gray')
    im2 = ax.imshow(absolute_el, cmap='nipy_spectral', vmin=vmin, vmax=vmax, 
                    alpha=0.7, origin='upper')

    cbar1_orientation='horizontal'
    cbar1_axes = [0.37, 0.85, 0.1, 0.1]
    cbar2_orientation='vertical'
    cbar2_axes = [0.8, 0.85, 0.1, 0.1]

    cnorm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    scalarmap = cmx.ScalarMappable(norm=cnorm, cmap='nipy_spectral')
    print("scaled cmap lim:", scalarmap.get_clim())
    bounds = np.linspace(vmin, vmax)
    scalarmap.set_array(bounds)

    cbar2_ax = fig.add_axes(cbar2_axes)
    cbar2 = fig.colorbar(im2, cax=cbar2_ax, orientation=cbar2_orientation)
    cbar2.ax.axhline(y=cbar2.norm(vmin*elev_cutoff), color='w', lw=1)
    cbar2.ax.axhline(y=cbar2.norm(vmax*elev_cutoff), color='w', lw=1)
    cbar2.ax.axis('off')


    cbar1_ax = fig.add_axes(cbar1_axes)
    cbar1 = fig.colorbar(im1, cax=cbar1_ax, orientation=cbar1_orientation)
    cbar1.ax.axhline(y=cbar2.norm(vmin*elev_cutoff), color='w', lw=1)
    cbar1.ax.axhline(y=cbar2.norm(vmax*elev_cutoff), color='w', lw=1)
    cbar1.ax.axis('off')

    cbar1.outline.set_visible(False)
    cbar2.outline.set_visible(False)

    #pl.subplots_adjust(top=0.8)

    for ax in axes.flat:
        ax.axis('off')

    return fig


#%%

rootdir = '/n/coxfs01/2p-data'
animalid = 'JC091' # 'JC076' #'JC091' #'JC059'
session = '20190607' #'20190420' #20190623' #'20190227'
fov = 'FOV1_zoom2p0x' #'FOV1_zoom2p0x' #'FOV4_zoom4p0x'
run = 'retino_run1'
traceid = 'analysis001' #'traces001'
#visual_area = ''

def extract_options(options):
    
    parser = optparse.OptionParser()

    parser.add_option('-D', '--root', action='store', dest='rootdir', 
                      default='/n/coxfs01/2p-data',\
                      help='data root dir [default: /n/coxfs01/2pdata]')
    parser.add_option('-i', '--animalid', action='store', dest='animalid', 
                        default='', help='Animal ID')

    # Set specific session/run for current animal:
    parser.add_option('-S', '--session', action='store', dest='session', default='', \
                      help='session dir (format: YYYMMDD_ANIMALID')
    parser.add_option('-A', '--fov', action='store', dest='fov', default='FOV1_zoom2p0x', \
                      help="acquisition folder (ex: 'FOV1_zoom3x') [default: FOV1_zoom2p0x]")
    parser.add_option('-R', '--run', action='store', dest='run', default='retino_run1', \
                      help="name of run (default: retino_run1")
    parser.add_option('-t', '--traceid', action='store', dest='traceid', 
                        default='analysis001', \
                        help="name of traces ID [default: analysis001]")
    parser.add_option('--cont', action='store_true', dest='use_cont', default=False, \
                        help="flag to use cont (match aggreg_gradient())")
       

    parser.add_option('--thr', action='store', dest='mag_thr', default=0.0025,
                    help='pixel threshold for mag-ratio (default: 0.003)')

    (options, args) = parser.parse_args(options)

    return options


def main(options):

    opts = extract_options(options)
    rootdir = opts.rootdir
    animalid = opts.animalid
    session = opts.session
    fov = opts.fov
    run = opts.run
    traceid = opts.traceid
    use_cont = opts.use_cont
    mag_thr = float(opts.mag_thr)

    run_dir = os.path.join(rootdir, animalid, session, fov, run)
    retinoid, RID = rutils.load_retino_analysis_info(animalid, session, fov, 
                                run, traceid, use_pixels=True, rootdir=rootdir)

    data_identifier = '|'.join([animalid, session, fov, run, retinoid])
    print("*** Dataset: %s ***" % data_identifier)

    #%% # Get processed retino data:
    processed_fpaths = glob.glob(os.path.join(RID['DST'], 'files', '*.h5'))
    print("Found %i processed retino runs." % len(processed_fpaths))

    # Make output dir
    absolute_maps_dir = os.path.join(run_dir, 'retino_analysis', 'absolute_maps')
    if not os.path.exists(absolute_maps_dir):
        os.makedirs(absolute_maps_dir)
            
    #%% Get condition info for trials:
    mwinfo = rutils.load_mw_info(animalid, session, fov, run)
    # Get run info:
    scaninfo = rutils.get_protocol_info(animalid, session, fov, run=run)

    print "---------------------------------"
    # Get stimulus info:
    stiminfo, trials_by_cond = rutils.get_retino_stimulus_info(mwinfo, scaninfo)
    print "Trials by condN:", trials_by_cond

    #%% Get screen info
    screen = get_screen_dims() 
    screen_left = -1*screen['azimuth_deg']/2.
    screen_right = screen['azimuth_deg']/2.
    screen_top = screen['altitude_deg']/2.
    screen_bottom = -1*screen['altitude_deg']/2.
    # adjust elevation limit to show only monitor extent    
    elev_cutoff = screen_top / screen_right
    print("<<<cutoff: %.2f>>>" % elev_cutoff)

    #%%
    # tiff_fpaths = glob.glob(os.path.join(RID['PARAMS']['tiff_source'], '*.tif'))
    conditions_fpath = glob.glob(os.path.join(run_dir, 'paradigm', 'files', 
                                                        'parsed_trials*.json'))[0]

    fit, magratio, phase, trials_by_cond = rutils.trials_to_dataframes(processed_fpaths, 
                                                                        conditions_fpath)
    #%%

    #mag_thr = 0.0025 # 0.0025
    d2 = scaninfo['pixels_per_line']
    d1 = scaninfo['lines_per_frame']

    magmaps = {}
    phasemaps = {}
    magthrs = {}
    for cond in trials_by_cond.keys():    
        magmaps[cond], phasemaps[cond], magthrs[cond] = filter_map_by_magratio(
                                                            magratio, phase, trials_by_cond,
                                                            cond=cond, use_cont=use_cont,
                                                            mag_thr=mag_thr, dims=(d1, d2))
        fig = plot_filtered_maps(cond, magmaps[cond], phasemaps[cond], magthrs[cond])

    ph_left = phasemaps['left'].copy()
    ph_right = phasemaps['right'].copy()
    ph_top = phasemaps['top'].copy()
    ph_bottom = phasemaps['bottom'].copy()
    print("got phase:", np.nanmin(ph_left), np.nanmax(ph_left)) # (0, 2*np.pi)

    if use_cont:
#        ph_left = make_continuous(ph_left)     
#        ph_right = make_continuous(ph_right) 
#        ph_top = make_continuous(ph_top)
#        ph_bottom = make_continuous(ph_bottom) 
        cont_str='continuous-first'
    else:
        cont_str=''
        
    absolute_az = (ph_left - ph_right) / 2.
    delay_az = (ph_left + ph_right) / 2.

    absolute_el = (ph_bottom - ph_top) / 2.
    delay_el = (ph_bottom + ph_top) / 2.

    vmin, vmax = (-np.pi, np.pi) # Now in range (-np.pi, np.pi)
    print("got absolute:", np.nanmin(absolute_az.min), np.nanmax(absolute_az.max))
    print("Delay:", np.nanmin(delay_az.min), np.nanmax(delay_az.max))

    if use_cont:
#        absolute_el = make_continuous(absolute_el) #-1*absolute_el
#        absolute_az = make_continuous(absolute_az)
#        delay_el = make_continuous(delay_el) #-1*absolute_el
#        delay_az = make_continuous(delay_az)
#        vmin, vmax = (0, 2*np.pi)
        print("[cont] got absolute:", np.nanmin(absolute_az), np.nanmax(absolute_az))
        print("[cont] Delay:", np.nanmin(delay_az.min), np.nanmax(delay_az))


    #%% #Plot absolute maps + delay maps
    fig, axes = pl.subplots(2,2)
    im1 = axes[0,0].imshow(absolute_az, cmap='nipy_spectral_r', vmin=vmin, vmax=vmax)
    im2 = axes[0,1].imshow(absolute_el, cmap='nipy_spectral_r', vmin=vmin, vmax=vmax)
    axes[1,0].imshow(delay_az, cmap='nipy_spectral', vmin=vmin, vmax=vmax)
    axes[1,1].imshow(delay_el, cmap='nipy_spectral', vmin=vmin, vmax=vmax)

    cbar1_orientation='horizontal'
    cbar1_axes = [0.35, 0.85, 0.1, 0.1]
    cbar2_orientation='vertical'
    cbar2_axes = [0.75, 0.85, 0.1, 0.1]

    cbaxes = fig.add_axes(cbar1_axes) 
    cb = pl.colorbar(im1, cax = cbaxes, orientation=cbar1_orientation)  
    cb.ax.axis('off')
    cb.outline.set_visible(False)

    cbaxes = fig.add_axes(cbar2_axes) 
    cb = pl.colorbar(im2, cax = cbaxes, orientation=cbar2_orientation)
    #cb.ax.set_ylim([cb.norm(-np.pi*top_cutoff), cb.norm(np.pi*top_cutoff)])
    cb.ax.axhline(y=cb.norm(vmin*elev_cutoff), color='w', lw=1)
    cb.ax.axhline(y=cb.norm(vmax*elev_cutoff), color='w', lw=1)
    cb.ax.axis('off')
    cb.outline.set_visible(False)

    pl.subplots_adjust(top=0.8)

    for ax in axes.flat:
        ax.axis('off')
        
    label_figure(fig, data_identifier)
    figname = 'acquisview_absolute_and_delay_maps_magthr_%.3f_%s' % (mag_thr, cont_str)
    pl.savefig(os.path.join(absolute_maps_dir, '%s.png' % figname))

    #%% Load surface image to plot overlay:
    #surface_fpath = glob.glob(os.path.join(rootdir, animalid, 'macro_maps', '*', '*urf*'))[0]
    #surface_img = cv2.imread(surface_fpath, -1)
    #print(surface_img.shape)
    overlay_surface = False
    if overlay_surface:
        ch_num = 2
        fov_imgs = glob.glob(os.path.join(rootdir, animalid, session, fov, 
                                        'anatomical', 'processed',\
                                        'processed*', 'mcorrected_*mean_deinterleaved', 
                                        'Channel%02d' % ch_num, 'File*', '*.tif'))
    else:
        ch_num = 1
        fov_imgs = glob.glob(os.path.join(run_dir, 'processed', 'processed*', 
                                            'mcorrected_*mean_deinterleaved',\
                                            'Channel%02d' % ch_num, 'File*', '*.tif')) 
    imlist = []
    for anat in fov_imgs:
        im = tf.imread(anat)
        imlist.append(im)
    surface_img = np.array(imlist).mean(axis=0)

    #pl.figure()
    #pl.imshow(surface_img, cmap='gray')
    if surface_img.shape[0] != absolute_az.shape[0]:
        reduce_factor = surface_img.shape[0] / absolute_az.shape[0]
        surface_img = block_reduce(surface_img, (2,2), func=np.mean)
        
    #%%
    #vmin = -np.pi
    #vmax = np.pi
    fig = plot_maps(absolute_az, absolute_el, surface_img=surface_img, 
                        vmin=vmin, vmax=vmax, elev_cutoff=elev_cutoff)
    label_figure(fig, data_identifier)
    figname = 'acquisview_absolute_maps_magthr_%.3f_%s' % (mag_thr, cont_str)
    pl.savefig(os.path.join(run_dir, 'retino_analysis', 'absolute_maps', '%s.png' % figname))



    #transf_ = coreg.orient_2p_to_macro(img, zoom_factor=zoom_factor, 
    #                                    save=False, normalize=True)
    #scaled_ = coreg.scale_2p_fov(transf_, pixel_size=pixel_size)

    az_transf = coreg.orient_2p_to_macro(absolute_az, zoom_factor=1., 
                                        save=False, normalize=True)
    el_transf = coreg.orient_2p_to_macro(absolute_el, zoom_factor=1., 
                                        save=False, normalize=True)
    fig = plot_maps(az_transf, el_transf, surface_img=surface_img, 
                    vmin=vmin, vmax=vmax, elev_cutoff=elev_cutoff)
    label_figure(fig, data_identifier)
    figname = 'naturalview_absolute_maps_magthr_%.3f_%s' % (mag_thr, cont_str)
    pl.savefig(os.path.join(run_dir, 'retino_analysis', 'absolute_maps', '%s.png' % figname))




if __name__=='__main__':
    main(sys.argv[1:])

# %%
