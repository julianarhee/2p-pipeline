#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 16:40:28 2021

@author: julianarhee
"""

#%%
import os
import glob
import json
import copy
import optparse
import sys
import traceback
import matplotlib as mpl
mpl.use('agg')
import rf_utils as rfutils
import py3utils as p3
import plotting as pplot


def extract_options(options):
    
    parser = optparse.OptionParser()
    parser.add_option('-D', '--root', action='store', dest='rootdir', default='/n/coxfs01/2p-data',\
                      help='data root dir (root project dir containing all animalids) [default: /n/coxfs01/2pdata]')
    parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')

    # Set specific session/run for current animal:
    parser.add_option('-S', '--session', action='store', dest='session', default='', \
                      help='session dir (format: YYYMMDD_ANIMALID')
    parser.add_option('-A', '--fov', action='store', dest='fov', default='FOV1_zoom2p0x', \
                      help="acquisition folder (ex: 'FOV1_zoom3x') [default: FOV1_zoom2p0x]")
    parser.add_option('-R', '--run', action='store', dest='run', default='rfs', \
                      help="name of run dir containing tiffs to be processed (ex: rfs)")
    parser.add_option('-t', '--traceid', action='store', dest='traceid', default='traces001', \
                      help="name of traces ID [default: traces001]")
    parser.add_option('-d', '--data-type', action='store', dest='trace_type', default='corrected', \
                      help="Trace type to use for analysis [default: corrected]")
       
    parser.add_option('-M', '--resp', action='store', dest='response_type', default='dff', \
                      help="Response metric to use for creating RF maps (default: dff)")
    
    parser.add_option('--new', action='store_true', dest='create_new', default=False, \
                      help="Flag to refit all rois")

    # pretty plotting options
    parser.add_option('--pretty', action='store_true', dest='make_pretty_plots', default=False, \
                      help="Flag to make pretty plots for roi fits")
    parser.add_option('-f', '--ellipse-fc', action='store', dest='ellipse_fc', 
                      default='none', help="[prettyplots] Ellipse face color (default:none)")
    parser.add_option('-e', '--ellipse-ec', action='store', dest='ellipse_ec', 
                      default='w', help="[prettyplots] Ellipse edge color (default:w)")
    parser.add_option('-l', '--ellipse-lw', action='store', dest='ellipse_lw', 
                      default=2, help="[prettyplots] Ellipse linewidth (default:2)")
    parser.add_option('--no-ellipse', action='store_false', dest='plot_ellipse', 
                      default=True, help="[prettyplots] Flag to NOT plot fit RF as ellipse")
    parser.add_option('-L', '--linecolor', action='store', dest='linecolor', 
                      default='darkslateblue', help="[prettyplots] Color for traces (default:darkslateblue)")
    parser.add_option('-c', '--cmap', action='store', dest='cmap', 
                      default='bone', help="[prettyplots] Cmap for RF maps (default:bone)")
    parser.add_option('-W', '--legend-lw', action='store', dest='legend_lw', 
                      default=2.0, help="[prettyplots] Lw for df/f legend (default:2)")
    parser.add_option('--fmt', action='store', dest='plot_format', 
                      default='svg', help="[prettyplots] Plot format (default:svg)")
    parser.add_option('-y', '--scaley', action='store', dest='scaley', default=None, 
                        help="[prettyplots] Set to float to set scale y across all plots (default: max of current trace)")
    parser.add_option('--nrois',  action='store', dest='nrois_plot', default=10, 
                      help="[prettyplots] N rois plot")

    # RF fitting options
    parser.add_option('--no-scale', action='store_false', dest='scale_sigma', 
                      default=False, help="flag to NOT scale sigma (use true sigma)")
    parser.add_option('--sigma', action='store', dest='sigma_scale', 
                      default=2.35, help="Sigma size to scale (FWHM, 2.35)")
    parser.add_option('-F', '--fit-thr', action='store', dest='fit_thr', default=0.5, 
                      help="Fit threshold (default:0.5)")

    parser.add_option('-p', '--post', action='store', dest='post_stimulus_sec', default=0.5, 
                      help="N sec to include in stimulus-response calculation for maps (default:0.5)")
    parser.add_option('--load', action='store_true', dest='reload_data', default=False, 
                      help="flag to reload/reprocess data arrays")
    parser.add_option('-n', '--nproc', action='store', dest='n_processes', default=1, 
                      help="N processes")

    parser.add_option('--sphere', action='store_true', 
                        dest='do_spherical_correction', default=False, help="N processes")
    (options, args) = parser.parse_args(options)

    return options



#%%%
def main(options):

    optsE = extract_options(options)
    
    rootdir = optsE.rootdir
    animalid = optsE.animalid
    session = optsE.session
    fov = optsE.fov
    run = optsE.run
    traceid = optsE.traceid
    trace_type = optsE.trace_type
    
    #segment = optsE.segment
    #visual_area = optsE.visual_area
    #select_rois = optsE.select_rois
    
    response_type = optsE.response_type
    do_spherical_correction = optsE.do_spherical_correction
    
    #response_thr = optsE.response_thr
    create_new= optsE.create_new

    fit_thr = float(optsE.fit_thr) 
    post_stimulus_sec = float(optsE.post_stimulus_sec)
    reload_data = optsE.reload_data

    scaley = float(optsE.scaley) if optsE.scaley is not None else optsE.scaley
    
    make_pretty_plots = optsE.make_pretty_plots
    plot_format = optsE.plot_format

    n_processes = int(optsE.n_processes)
    test_subset=False

    fit_results, fit_params = rfutils.fit_2d_receptive_fields(animalid, session, fov, 
                                run, traceid, trace_type=trace_type, 
                                post_stimulus_sec=post_stimulus_sec,
                                scaley=scaley,
                                fit_thr=fit_thr,
                                reload_data=reload_data,
                                #visual_area=visual_area, select_rois=select_rois,
                                response_type=response_type, #response_thr=response_thr, 
                                do_spherical_correction=do_spherical_correction,
                                create_new=create_new,
                                make_pretty_plots=make_pretty_plots, 
                                nrois_plot=int(optsE.nrois_plot),
                                ellipse_ec=optsE.ellipse_ec, 
                                ellipse_fc=optsE.ellipse_fc, 
                                ellipse_lw=optsE.ellipse_lw, 
                                plot_ellipse=optsE.plot_ellipse,
                                linecolor=optsE.linecolor, cmap=optsE.cmap, 
                                legend_lw=optsE.legend_lw, 
                                plot_format=plot_format, n_processes=n_processes, 
                                test_subset=test_subset)
    
    print("--- fit %i rois total ---" % (len(fit_results.keys())))

    #%

    print("((( RFs done! )))))")
       
    return
 
if __name__ == '__main__':
    main(sys.argv[1:])



