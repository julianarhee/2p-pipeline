#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 11:00:17 2020

@author: julianarhee
"""
#%%
import matplotlib as mpl
mpl.use('agg')
import os
import glob
import json
#import h5py
import copy
#import cv2
#import imutils
import sys
import optparse
import shutil
import traceback
import time



def spatially_sort_compare_position(fovinfo, transform=True):
    from pipeline.python.classifications import experiment_classes as util

    zimg = fovinfo['zimg']
    roi_contours = fovinfo['roi_contours']
    posdf = fovinfo['positions']
    npix_y, npix_x = zimg.shape
    
    # # Sort ROIs by x,y position:
    sorted_roi_indices_xaxis, sorted_roi_contours_xaxis = coor.spatially_sort_contours(roi_contours, sort_by='x', dims=(npix_x, npix_y))
    sorted_roi_indices_yaxis, sorted_roi_contours_yaxis = coor.spatially_sort_contours(roi_contours, sort_by='y', dims=(npix_x, npix_y))
    
    _, sorted_roi_centroids_xaxis = coor.spatially_sort_contours(roi_contours, sort_by='x', dims=(npix_x, npix_y), get_centroids=True)
    _, sorted_roi_centroids_yaxis = coor.spatially_sort_contours(roi_contours, sort_by='y', dims=(npix_x, npix_y), get_centroids=True)

    # x-axis in FOV = posterior to anterior, from left to right (0-->512)
    # y-axis in FOV = lateral to medial, from top to bottom (0-->512)
    
    # Color by RF position:
    rf_rois = posdf.index.tolist()
    #% #### Plot

    convert_um = True
    fig, axes = pl.subplots(2,2)
    fig.patch.set_alpha(1)
    ### Plot ALL sorted rois:
    ax = axes[0,1] 
    util.plot_roi_contours(zimg, sorted_roi_indices_xaxis, sorted_roi_contours_xaxis, 
                           label_rois=rf_rois, label=False, single_color=False, overlay=True,
                           clip_limit=0.02, draw_box=False, thickness=2, 
                           ax=ax, transform=transform)
    ax.axis('off')
                        
    ax = axes[0,0]
    util.plot_roi_contours(zimg, sorted_roi_indices_yaxis, sorted_roi_contours_yaxis,
                           label_rois=rf_rois, label=False, single_color=False, overlay=True,
                           clip_limit=0.02, draw_box=False, thickness=2, 
                           ax=ax, transform=transform)
    ax.axis('off')
                    
    ### Plot corresponding RF centroids:
    colors = ['k' for roi in rf_rois]
    units = 'um' if convert_um else 'pixels'
    # Get values for azimuth:    
    ax = axes[1,0]
    ax.scatter(posdf['xpos_fov'], posdf['xpos_rf'], c=colors, alpha=0.5)
    ax.set_title('Azimuth')
    ax.set_ylabel('RF position (rel. deg.)')
    ax.set_xlabel('FOV position (%s)' % units)
    #ax.set_xlim([0, ylim])
    sns.despine(offset=4, trim=True, ax=ax)
    # Get values for elevation:
    ax = axes[1,1]
    ax.scatter(posdf['ypos_fov'], posdf['ypos_rf'], c=colors, alpha=0.5)
    ax.set_title('Elevation')
    ax.set_ylabel('RF position (rel. deg.)')
    ax.set_xlabel('FOV position (%s)' % units)
    #ax.set_xlim([0, xlim])
    ax.axis('on')
    sns.despine(offset=4, trim=True, ax=ax)
    
    pl.subplots_adjust(wspace=0.5)
    
    return fig
    


    exp = util.ReceptiveFields(rfname, animalid, session, fov, 
                               traceid=traceid, trace_type='corrected')

#    # Create output dir in "summmaries" folder
#    statsdir, stats_desc = util.create_stats_dir(animalid, session, fov,
#                                                  traceid=traceid, trace_type=trace_type,
#                                                  response_type=response_type, 
#                                                  responsive_test=None, responsive_thr=0)
#    if not os.path.exists(os.path.join(statsdir, 'receptive_fields')):
#        os.makedirs(os.path.join(statsdir, 'receptive_fields'))
#    print("Saving stats output to: %s" % statsdir)    

    exp = util.ReceptiveFields(rfname, animalid, session, fov, 
                               traceid=traceid, trace_type='corrected')

#    # Create output dir in "summmaries" folder
#    statsdir, stats_desc = util.create_stats_dir(animalid, session, fov,
#                                                  traceid=traceid, trace_type=trace_type,
#                                                  response_type=response_type, 
#                                                  responsive_test=None, responsive_thr=0)
#    if not os.path.exists(os.path.join(statsdir, 'receptive_fields')):
#        os.makedirs(os.path.join(statsdir, 'receptive_fields'))
#    print("Saving stats output to: %s" % statsdir)    

    #% Plot spatially ordered rois
    if len(glob.glob(os.path.join(statsdir, 'receptive_fields', 'spatially_sorted*.svg'))) == 0: 
        print("Getting FOV info for rois.")
        fig = spatially_sort_compare_position(estats.fovinfo, transform=transform_fov)
        label_figure(fig, data_identifier)
        pl.savefig(os.path.join(statsdir, 'receptive_fields', \
                'spatially_sorted_rois_%s%s.svg' % (rfname, view_str)))
        pl.close()

