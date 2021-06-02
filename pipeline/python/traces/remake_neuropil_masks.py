#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 08 15:51:55 2020

@author: julianarhee
"""

import os
import glob
import json
import h5py
import cv2
import traceback
import h5py
import sys
import optparse
import matplotlib as mpl
mpl.use('agg')

import numpy as np
import pylab as pl

from pipeline.python.traces import get_traces as gtraces
from pipeline.python.utils import natural_keys, label_figure

def create_neuropil_masks(masks, niterations=20, gap_iterations=4, verbose=False):

    if verbose:
        um_per_pix = (2.312 + 1.904)/2.
        inner_um = um_per_pix*gap_niterations
        outer_um = um_per_pix*np_niterations
        print("Annulus: %.2f-%.2f um" % (inner_um, outer_um))

    # Create kernel for dilating ROIs:
    kernel = np.ones((3,3),masks.dtype)

    nrois = masks.shape[-1]
    np_masks = np.empty(masks.shape, dtype=masks.dtype)
    print "*** creating NP masks of shape: %s" % str(np_masks.shape)

    for ridx in range(nrois):
        rmask = masks[:,:,ridx]
#         if niterations==20:
#             gap_iterations = 8
#         else:
#             gap_iterations = 4
        gap = cv2.dilate(rmask, kernel, iterations=gap_iterations)
        dilated = cv2.dilate(rmask, kernel, iterations=niterations)

        # Subtract to get annulus region:
        annulus = (dilated - gap)

        # Get full mask image to subtract overlaps:
        allmasks = np.sum(masks, axis=-1)
        summed = annulus + allmasks
        summed[summed>1] = 0  # Find where this neuropil overlaps any other cells
        summed[allmasks > 0] = 0
        neuropil = summed.copy()

        # Add annulus back in to make neuropil area = 2, everythign else = 1:
        #summed += annulus
        #neuropil = summed - allmasks
        np_masks[:,:,ridx] = neuropil.astype('bool')

    return np_masks


def remake_neuropil_annulus(maskdict_path, np_niterations=24, gap_niterations=4, 
                            curr_slice='Slice01', plot_masks=False):
    #curr_slice = 'Slice01'

    # Set output dir
    traceid_dir = os.path.split(maskdict_path)[0]
    mask_figdir = os.path.join(traceid_dir, 'figures', 'masks', 'neuropil_annulus')
    if not os.path.exists(mask_figdir):
        os.makedirs(mask_figdir)
    print(mask_figdir)

    MASKS = h5py.File(maskdict_path, 'a')
    filenames = sorted(MASKS.keys(), key=natural_keys)
    try:
        for curr_file in filenames:
            print("... making masks %s" % curr_file)
            # Get masks and reshape
            filegrp = MASKS[curr_file]
            msks = filegrp[curr_slice]['maskarray'][:]
            d, nr = msks.shape
            d1, d2 = filegrp[curr_slice]['zproj'].shape # should double check this in case not equal
            msks_r = np.reshape(msks, (d1, d2, nr), order='C')

            # Make NP mask
            np_masks = create_neuropil_masks(msks_r, niterations=np_niterations, gap_iterations=gap_niterations)
            npil_arr = gtraces.masks_to_normed_array(np_masks)

            # Save to file
            npil = filegrp['%s/np_maskarray' % curr_slice]
            npil[...] = npil_arr
            npil.attrs['np_niterations'] = np_niterations
            npil.attrs['gap_niterations'] = gap_niterations

            if plot_masks:
                figid = maskdict_path
                m1 = msks_r.sum(axis=-1)
                soma_masks = np.ma.masked_where(m1==0, m1)
                m2 = np_masks.sum(axis=-1)
                neuropil_masks = np.ma.masked_where(m2==0, m2)
                fig, ax = pl.subplots()
                ax.imshow(soma_masks, cmap='Blues', alpha=1)
                ax.imshow(neuropil_masks, cmap='Reds', alpha=0.7)
                label_figure(fig, figid)
                figname = '%s_masks_outer%i_inner%i' % (curr_file, np_niterations, gap_niterations)
                pl.savefig(os.path.join(mask_figdir, '%s.svg' % figname))
                pl.close()

    except Exception as e:
        print "------------------------------------------"
        print "*** ERROR creating masks: %s, %s ***" % (curr_file, curr_slice)
        traceback.print_exc()
        print "------------------------------------------"

    finally:
        MASKS.close()
        
    return maskdict_path


def create_masks_for_all_runs(animalid, session, fov, experiment=None, traceid='traces001', 
                            np_niterations=24, gap_niterations=4, rootdir='/n/coxfs01/2p-data', plot_masks=True):

    print("Creating masks for all runs.")
    # Get runs to extract
    session_dir = os.path.join(rootdir, animalid, session)
    if experiment is not None:
        all_rundirs = [r for r in sorted(glob.glob(os.path.join(session_dir, fov, '%s_run*' % experiment)), key=natural_keys)\
                   if 'retino' not in r and 'compare' not in r] 


    else:
        all_rundirs = [r for r in sorted(glob.glob(os.path.join(session_dir, fov, '*_run*')), key=natural_keys)\
                       if 'retino' not in r and 'compare' not in r] 

    #run_dir = all_rundirs[0]
    for ri, run_dir in enumerate(all_rundirs): 
        try:
            maskdict_path = create_masks_for_run(run_dir, traceid=traceid, np_niterations=np_niterations,
                         gap_niterations=gap_niterations, rootdir=rootdir, plot_masks=plot_masks)

        #filetraces_dir = apply_masks_for_run(run_dir, maskdict_path, traceid=traceid, np_correction_factor=np_correction_factor)
        except Exception as e:
            print("***ERROR creating masks: %s" % run_dir)
            continue

        print("... finished %i of %i" % (int(ri+1), len(all_rundirs)))

    print("~~~~~~~ FINISHED CREATING MASKS ~~~~~~~~~~~~.")

    return 


def apply_masks_for_all_runs(animalid, session, fov, experiment=None, 
                            traceid='traces001', np_correction_factor=0.7, rootdir='/n/coxfs01/2p-data'):

    print("Applying masks to tifs for all runs.")
    # Get runs to extract
    session_dir = os.path.join(rootdir, animalid, session)
    if experiment is not None:
        all_rundirs =  [r for r in sorted(glob.glob(os.path.join(session_dir, fov, '%s_run*' % experiment)), key=natural_keys)\
                   if 'retino' not in r and 'compare' not in r] 
    else:
        all_rundirs = [r for r in sorted(glob.glob(os.path.join(session_dir, fov, '*_run*')), key=natural_keys)\
                   if 'retino' not in r and 'compare' not in r] 

    for ri, run_dir in enumerate(all_rundirs): 
        try:
            # Get trace extraction info
            if 'retino' in run_dir:
                TID = gtraces.load_AID(run_dir, traceid)
            else:
                TID = gtraces.load_TID(run_dir, traceid, auto=True)
            
            # Set mask path
            maskdict_path = os.path.join(TID['DST'], 'MASKS.hdf5')
            filetraces_dir = os.path.join(TID['DST'], 'files')

            filetraces_dir = apply_masks_for_run(filetraces_dir, maskdict_path, np_correction_factor=np_correction_factor, rootdir=rootdir)
        except Exception as e:
            print("***ERROR creaitng masks: %s" % run_dir)
            continue
 
        print("... finished %i of %i" % (int(ri+1), len(all_rundirs)))

    print("~~~~~~~~~~~~~~ FINISHED APPLYING MASKS ~~~~~~~~~~~~~~~.")
    
    return filetraces_dir


    
def create_masks_for_run(run_dir, traceid='traces001', np_niterations=24, gap_niterations=4, rootdir='/n/coxfs01/2p-data', plot_masks=True):

    # Get trace extraction info
    if 'retino' in run_dir:
        print("...getting RETINO ID info...")
        TID = gtraces.load_AID(run_dir, traceid)
    else:
        TID = gtraces.load_TID(run_dir, traceid, auto=True)
    session_dir = run_dir.split('/FOV')[0]
    print(session_dir)
    RID = gtraces.load_RID(session_dir, TID['PARAMS']['roi_id'])

    # Set output dir
    mask_figdir = os.path.join(TID['DST'], 'figures', 'masks', 'neuropil_annulus')
    if not os.path.exists(mask_figdir):
        os.makedirs(mask_figdir)

    # Load existing masks
    maskinfo = gtraces.get_mask_info(TID, RID, nslices=1, rootdir=rootdir)
    maskdict_path = os.path.join(TID['DST'], 'MASKS.hdf5')

    #### Make new NP mask
    maskdict_path = remake_neuropil_annulus(maskdict_path, np_niterations=np_niterations, 
                                        gap_niterations=gap_niterations, plot_masks=plot_masks)

   
    return maskdict_path

def apply_masks_for_run(filetraces_dir, maskdict_path, np_correction_factor=0.7, rootdir='/n/coxfs01/2p-data'):

    #### Apply new NP mask to extract traces
    print "--- Using SUBTRACTION method, (global) correction-factor: ", np_correction_factor
    filetraces_dir = gtraces.append_neuropil_subtraction(maskdict_path,
                                                 np_correction_factor,
                                                 filetraces_dir,
                                                 create_new=True,
                                                 rootdir=rootdir)

    return filetraces_dir


def extract_options(options):
    parser = optparse.OptionParser()

    # PATH opts:
    parser.add_option('-D', '--root', action='store', dest='rootdir', default='/n/coxfs01/2p-data', 
                      help='root project dir containing all animalids [default: /n/coxfs01/2pdata]')
    parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', 
                      help='Animal ID')
    parser.add_option('-S', '--session', action='store', dest='session', default='', 
                      help='Session (format: YYYYMMDD)')
    
    # Set specific session/run for current animal:
    parser.add_option('-A', '--fov', action='store', dest='fov', default='FOV1_zoom2p0x', 
                      help="fov name (default: FOV1_zoom2p0x)")
    parser.add_option('-E', '--experiment', action='store', dest='experiment', default=None, 
                      help="experiment name (e.g,. gratings, rfs, rfs10, or blobs)") #: FOV1_zoom2p0x)")
    
    parser.add_option('-t', '--traceid', action='store', dest='traceid', default='traces001', 
                      help="traceid (default: traces001)")

    # Neuropil mask params
    parser.add_option('-N', '--np-outer', action='store', dest='np_niterations', default=24, 
                      help="Num cv dilate iterations for outer annulus (default: 24, ~50um for zoom2p0x)")
    parser.add_option('-g', '--np-inner', action='store', dest='gap_niterations', default=4, 
                      help="Num cv dilate iterations for inner annulus (default: 4, gap ~8um for zoom2p0x)")
    parser.add_option('-c', '--factor', action='store', dest='np_correction_factor', default=0.7, 
                      help="Neuropil correction factor (default: 0.7)")

    # Alignment params
    parser.add_option('-p', '--iti-pre', action='store', dest='iti_pre', default=1.0, 
                      help="pre-stim amount in sec (default: 1.0)")
    parser.add_option('-P', '--iti-post', action='store', dest='iti_post', default=1.0, 
                      help="post-stim amount in sec (default: 1.0)")

    parser.add_option('--plot', action='store_true', dest='plot_masks', default=False, 
                      help="set flat to plot soma and NP masks")
    parser.add_option('--apply-only', action='store_true', dest='apply_masks_only', default=False, 
                      help="set flag to just APPLY soma and NP masks")


#    parser.add_option('-r', '--rows', action='store', dest='rows',
#                          default=None, help='Transform to plot along ROWS (only relevant if >2 trans_types)')
#    parser.add_option('-c', '--columns', action='store', dest='columns',
#                          default=None, help='Transform to plot along COLUMNS')
#    parser.add_option('-H', '--hue', action='store', dest='subplot_hue',
#                          default=None, help='Transform to plot by HUE within each subplot')
#    parser.add_option('-d', '--response', action='store', dest='response_type',
#                          default='dff', help='Traces to plot (default: dff)')
#
#    parser.add_option('-f', '--filetype', action='store', dest='filetype',
#                          default='svg', help='File type for images [default: svg]')
#

    (options, args) = parser.parse_args(options)

    return options


def make_masks(animalid, session, fov, experiment=None, traceid='traces001', np_niterations=24, gap_niterations=4,
                np_correction_factor=0.7, rootdir='/n/coxfs01/2p-data', plot_masks=True, apply_masks_only=False):

    if not apply_masks_only:
        print("1. Creating neuropil masks")
        create_masks_for_all_runs(animalid, session, fov, experiment=experiment, traceid=traceid, 
                               np_niterations=np_niterations, gap_niterations=gap_niterations, 
                                rootdir=rootdir, plot_masks=plot_masks)
        print("---- completed NP mask extraction ----")
    else:
        print("---- skipping NP mask extraction ----")

    print("2. Applying neuropil masks")
    filetraces_dir = apply_masks_for_all_runs(animalid, session, fov, traceid=traceid, experiment=experiment, 
                            rootdir=rootdir, np_correction_factor=np_correction_factor)
    print("---- applied NP masks to tifs ----")

    traceid_dir = os.path.split(filetraces_dir)[0]
    with open(os.path.join(traceid_dir, 'extraction_params.json'), 'r') as f:
    #with open(os.path.join(traceid_dir, 'event_alignment.json'), 'r') as f:
        eparams = json.load(f)
    eparams.update({'np_niterations': np_niterations,
                    'gap_niterations': gap_niterations,
                    'np_correction_factor': np_correction_factor})
    with open(os.path.join(traceid_dir, 'extraction_params.json'), 'w') as f: 
        json.dump(eparams, f, indent=4, sort_keys=True)

    print("--- updated extraction info ---")


    print("FINISHED.")
    print("---------------------------------------------")
    print("Here's a summary")
    print("---------------------------------------------")
    print("Neuropil annulus: %i gap and %i outer iterations" % (gap_niterations, np_niterations))

    um_per_pix = (2.312 + 1.904)/2.
    inner_um = um_per_pix*gap_niterations
    outer_um = um_per_pix*np_niterations 
    print("i.e., %.2f-%.2f micron annulus" % (inner_um, outer_um))
    print("Neuropil correction factor was %.2f" % np_correction_factor)
    print("---------------------------------------------")
    
    return None

def main(options):
   
    opts = extract_options(options)
    animalid = opts.animalid
    session = opts.session
    fov = opts.fov
    traceid = opts.traceid
    np_niterations = int(opts.np_niterations)
    gap_niterations = int(opts.gap_niterations)
    np_correction_factor = float(opts.np_correction_factor)
    plot_masks = opts.plot_masks
    rootdir = opts.rootdir
    apply_masks_only = opts.apply_masks_only
    experiment=opts.experiment

    make_masks(animalid, session, fov, experiment=experiment, traceid=traceid, 
                    np_niterations=np_niterations, gap_niterations=gap_niterations,
                np_correction_factor=np_correction_factor, rootdir=rootdir, plot_masks=plot_masks,
                apply_masks_only=apply_masks_only)
    print("done!")

if __name__ == '__main__':
    main(sys.argv[1:])
    


    
    



