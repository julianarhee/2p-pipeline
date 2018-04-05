#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 2018

@author: cesarechavarria
"""


import os
import json
import pprint
import re
import pkg_resources
import optparse
import sys
import hashlib
from pipeline.python.utils import write_dict_to_json, get_tiff_paths
import numpy as np
from checksumdir import dirhash

pp = pprint.PrettyPrinter(indent=4)

#-----------------------------------------------------
#           MISC FXNS
#-----------------------------------------------------
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]


#-----------------------------------------------------
#           RELEVANT FXNS
#-----------------------------------------------------

def create_retinoid(options):

    options = extract_options(options)

    rootdir = options.rootdir
    homedir = options.homedir
    notnative = options.notnative
    print "ROOT:", rootdir
    print "HOME:", homedir

    animalid = options.animalid
    session = options.session
    acquisition = options.acquisition
    run = options.run

    tiffsource = options.tiffsource
    sourcetype = options.sourcetype

    pixelflag = options.pixelflag
    roi_id = options.roi_id
    if roi_id is not None:
        pixelflag = False


    downsample = int(options.downsample)
    smooth_fwhm = options.smooth_fwhm

    rolling_mean = options.rolling_mean
    average_frames = options.time_average

    signal_channel = options.signal_channel

    auto = options.default

    #define session directory
    session_dir = os.path.join(homedir, animalid, session)

    # Set retino analysis dir:
    run_dir = os.path.join(session_dir, acquisition, run)
    analysis_dir = os.path.join(run_dir, 'retino_analysis')
    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)

    # Get paths to tiffs for analysis:
    tiffpaths = get_tiff_paths(rootdir=homedir, animalid=animalid, session=session,
                               acquisition=acquisition, run=run,
                               tiffsource=tiffsource, sourcetype=sourcetype, auto=auto)

    # Create analysis params 
    tiff_sourcedir = os.path.split(tiffpaths[0])[0]

    if pixelflag:
        PARAMS = get_params_dict_pixels(tiff_sourcedir,downsample,smooth_fwhm,rolling_mean,average_frames,
                                        notnative,rootdir,homedir)
    else:
        #get RID
        with open(os.path.join(session_dir, 'ROIs', 'rids_%s.json' % session), 'r') as f:
            roidict = json.load(f)
        RID = roidict[roi_id]

            # Check if there are any TIFFs to exclude:
        orig_roi_dst = RID['DST']
        if homedir not in orig_roi_dst:
            orig_root = orig_roi_dst.split('/%s/%s' % (animalid, session))[0]
            print "ORIG root:", orig_root
            rparams_dir = orig_roi_dst.replace(orig_root, homedir)
        else:
            rparams_dir = orig_roi_dst
        print "Loading PARAM info... Looking in ROI dst dir: %s" % rparams_dir
        with open(os.path.join(rparams_dir, 'roiparams.json'), 'r') as f:
            roiparams = json.load(f)
        excluded_tiffs = roiparams['excluded_tiffs']

        PARAMS = get_params_dict_rois(signal_channel, tiff_sourcedir, RID, 
                                downsample,smooth_fwhm,rolling_mean,average_frames, excluded_tiffs,
                                notnative, rootdir, homedir, auto)


    RETINOID = initialize_retinoid(PARAMS, run_dir, auto=auto)

    # Create ANALYSIS output directory:
    analysis_name = '_'.join((RETINOID['analysis_id'], RETINOID['analysis_hash']))
    curr_analysis_dir = os.path.join(analysis_dir, analysis_name)
    if not os.path.exists(curr_analysis_dir):
        os.makedirs(curr_analysis_dir)

    # # Check RETINOID fields to include RETINOID hash, and save updated to analysis DICT:
    if RETINOID['analysis_hash'] not in RETINOID['DST']:
        RETINOID['DST'] = RETINOID['DST'] + '_' + RETINOID['analysis_hash']
    update_retinoid_records(RETINOID, run_dir)

    # Write to tmp_retinoid folder in current run source:
    tmp_retinoid_dir = os.path.join(analysis_dir, 'tmp_retinoids')
    if not os.path.exists(tmp_retinoid_dir):
        os.makedirs(tmp_retinoid_dir)

    tmp_retinoid_path = os.path.join(tmp_retinoid_dir, 'tmp_retinoid_%s.json' % RETINOID['analysis_hash'])
    write_dict_to_json(RETINOID, tmp_retinoid_path)

    print "********************************************"
    print "Created params for ANALYSIS SET, hash: %s" % RETINOID['analysis_hash']
    print "********************************************"

    return RETINOID



#%%
def extract_options(options):
    choices_sourcetype = ('raw', 'mcorrected', 'bidi')
    default_sourcetype = 'mcorrected'

    parser = optparse.OptionParser()

    # PATH opts:
    parser.add_option('-D', '--root', action='store', dest='rootdir', default='/nas/volume1/2photon/data', help='data root dir (root project dir containing all animalids) [default: /nas/volume1/2photon/data, /n/coxfs01/2pdata if --slurm]')
    parser.add_option('-H', '--home', action='store', dest='homedir', default='/nas/volume1/2photon/data', help='current data root dir (if creating params with path-root different than what will be used for actually doing the processing.')
    parser.add_option('--notnative', action='store_true', dest='notnative', default=False, help="Set flag if not setting params on same system as processing. MUST rsync data sources.")

    parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')
    parser.add_option('-S', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID')
    parser.add_option('-A', '--acq', action='store', dest='acquisition', default='FOV1', help="acquisition folder (ex: 'FOV1_zoom3x') [default: FOV1]")
    parser.add_option('-R', '--run', action='store', dest='run', default='', help="name of run dir containing tiffs to be processed (ex: gratings_phasemod_run1)")
    parser.add_option('--slurm', action='store_true', dest='slurm', default=False, help="Set if running as SLURM job on Odyssey")

    parser.add_option('-s', '--tiffsource', action='store', dest='tiffsource', default=None, help="Name of folder containing tiffs to be processed (ex: processed001). Should be child of <run>/processed/")
    parser.add_option('-t', '--source-type', type='choice', choices=choices_sourcetype, action='store', dest='sourcetype', default=default_sourcetype, help="Type of tiff source. Valid choices: %s [default: %s]" % (choices_sourcetype, default_sourcetype))

    parser.add_option('--pixels', action='store_true', dest='pixelflag', default=True, help="Analyze images as pixel arrays (instead of ROIs)")

    parser.add_option('-r', '--roi-id', action='store', dest='roi_id', default=None, help="ROI ID for rid param set to use (created with set_roi_params.py, e.g., rois001, rois005, etc.)")
    parser.add_option('-c', '--channel', action='store', dest='signal_channel', default=1, help="Signal channel [default: 1]")

    #spatial pre-processing options, applied on images
    parser.add_option('-d', '--downsample', action='store', dest='downsample', default=2, help='Factor by which to downsample images (integer)')    
    parser.add_option('-f', '--fwhm', action='store', dest='smooth_fwhm', default=None, help='full-width at half-max size of guassian kernel for smoothing images(odd integer)')

    #temporal pre-processing options, applied on timecourses
    parser.add_option('-m', '--rollingmean', action='store_true', dest='rolling_mean', default=False, help='Boolean to indicate whether to subtract rolling mean from signal')
    parser.add_option('-w', '--timeaverage', action='store', dest='time_average', default=5, help='Size of time window with which to average frames (integer)')

    parser.add_option('--default', action='store_true', dest='default', default='store_false', help="Use all DEFAULT params, for params not specified by user (prevent interactive)")

    (options, args) = parser.parse_args(options)

    if options.slurm is True:
        if 'coxfs01' not in options.rootdir:
            options.rootdir = '/n/coxfs01/2p-data'

    if options.notnative is False:
        print "NATIVE~~"
        options.homedir = options.rootdir

    return options

#%%
def get_params_dict_pixels(tiff_sourcedir,downsample,smooth_fwhm,rolling_mean,average_frames,
                            notnative=False, rootdir='', homedir=''):
    PARAMS = dict()
    PARAMS['tiff_source'] = tiff_sourcedir
    PARAMS['roi_type'] = 'pixels'
    PARAMS['downsample_factor'] = downsample
    PARAMS['smooth_fwhm'] = smooth_fwhm
    PARAMS['minus_rolling_mean'] = rolling_mean
    PARAMS['average_frames'] = average_frames

    # Replace PARAM paths with processing-machine paths:
    if notnative is True:
        PARAMS['tiff_source'] = PARAMS['tiff_source'].replace(homedir, rootdir)

    PARAMS['hashid'] = hashlib.sha1(json.dumps(PARAMS, sort_keys=True)).hexdigest()[0:6]
    return PARAMS

def get_params_dict_rois(signal_channel, tiff_sourcedir, RID, 
                    downsample,smooth_fwhm,rolling_mean,average_frames,excluded_tiffs=[],
                    notnative=False, rootdir='', homedir='', auto=False):
    PARAMS = dict()
    PARAMS['downsample_factor'] = downsample #None
    PARAMS['smooth_fwhm'] = smooth_fwhm
    PARAMS['minus_rolling_mean'] = rolling_mean
    PARAMS['average_frames'] = average_frames
    PARAMS['tiff_source'] = tiff_sourcedir
    PARAMS['excluded_tiffs'] = excluded_tiffs
    PARAMS['signal_channel'] = signal_channel
    PARAMS['roi_id'] = RID['roi_id']
    PARAMS['rid_hash'] = RID['rid_hash']
    PARAMS['roi_type'] = RID['roi_type']

    # Replace PARAM paths with processing-machine paths:
    if notnative is True:
        PARAMS['tiff_source'] = PARAMS['tiff_source'].replace(homedir, rootdir)

    PARAMS['hashid'] = hashlib.sha1(json.dumps(PARAMS, sort_keys=True)).hexdigest()[0:6]

    return PARAMS


#%%
def load_analysisdict(run_dir):

    analysisdict = None

    run = os.path.split(run_dir)[1]
    analysisdict_filepath = os.path.join(run_dir, 'retino_analysis', 'analysisids_%s.json' % run)

    # Load analysis "processdict" file:
    if os.path.exists(analysisdict_filepath):
        print "exists!"
        with open(analysisdict_filepath, 'r') as f:
            analysisdict = json.load(f)

    return analysisdict

def initialize_retinoid(PARAMS, run_dir, auto=False):
    # Create ANALYSIS ID (RETINOID):
    print "************************"
    print "Initializing analysis ID."
    print "************************"
    analysis_id = get_analysis_id(PARAMS, run_dir, auto=auto)

    retinoid = dict()
    #version = pkg_resources.get_distribution('pipeline').version

    #tid['version'] = version
    retinoid['analysis_id'] = analysis_id
    retinoid['PARAMS'] = PARAMS
    retinoid['SRC'] = PARAMS['tiff_source']
    retinoid['DST'] = os.path.join(run_dir, 'retino_analysis', analysis_id)

    retinoid['analysis_hash'] = hashlib.sha1(json.dumps(retinoid, sort_keys=True)).hexdigest()[0:6]

    return retinoid

def update_retinoid_records(RETINOID, run_dir):

    print "************************"
    print "Updating JSONS..."
    print "************************"
    run = os.path.split(run_dir)[1]
    analysisdict_filepath = os.path.join(run_dir, 'retino_analysis', 'analysisids_%s.json' % run)

    if os.path.exists(analysisdict_filepath):
        with open(analysisdict_filepath, 'r') as f:
            analysisdict = json.load(f)
    else:
        analysisdict = dict()

    analysis_id = RETINOID['analysis_id']
    analysisdict[analysis_id] = RETINOID

    #% Update Process Info DICT:
    write_dict_to_json(analysisdict, analysisdict_filepath)

    print "ANALYSIS Set Info UPDATED."


def get_analysis_id(PARAMS, run_dir, auto=False):

    analysis_id = None

    print "********************************"
    print "Checking previous analysis IDs..."
    print "********************************"

    # Load previously created PIDs:
    analysisdict = load_analysisdict(run_dir)

    #Check current PARAMS against existing PId params by hashid:
    if analysisdict is None or len(analysisdict.keys()) == 0:
        analysisdict = dict()
        existing_tids = []
        is_new_tid = True
        print "No existing analysis IDs found."
    else:
        existing_tids = sorted([str(k) for k in analysisdict.keys()], key=natural_keys)
        print existing_tids
        matching_tids = sorted([tid for tid in existing_tids if analysisdict[tid]['PARAMS']['hashid'] == PARAMS['hashid']], key=natural_keys)
        is_new_tid = False
        if len(matching_tids) > 0:
            while True:
                print "WARNING **************************************"
                print "Current param config matches existing analysis ID:"
                for tidx, tid in enumerate(matching_tids):
                    print "******************************************"
                    print tidx, tid
                    pp.pprint(analysisdict[tid])
                if auto is True:
                    check_tidx = ''
                else:
                    check_tidx = raw_input('Enter IDX of analysis id to re-use, or hit <ENTER> to create new: ')
                if len(check_tidx) == 0:
                    is_new_tid = True
                    break
                else:
                    confirm_reuse = raw_input('Re-use analysis ID %s? Press <Y> to confirm, any key to try again:' % existing_tids[int(check_tidx)])
                    if confirm_reuse == 'Y':
                        is_new_tid = False
                        break
        else:
            is_new_tid = True

    if is_new_tid is True:
        # Create new PID by incrementing num of process dirs found:
        analysis_id = 'analysis%03d' % int(len(existing_tids)+1)
        print "Creating NEW analysis ID: %s" % analysis_id
    else:
        # Re-using an existing PID:
        analysis_id = existing_tids[int(check_tidx)]
        print "Reusing existing analysis id: %s" % analysis_id

    return analysis_id

def get_params_dict(tiff_sourcedir,pixelflag, RID=None):
    PARAMS = dict()
    PARAMS['tiff_source'] = tiff_sourcedir
    PARAMS['hashid'] = hashlib.sha1(json.dumps(PARAMS, sort_keys=True)).hexdigest()[0:6]
    if pixelflag:
        PARAMS['roi_type'] = 'pixels'
    else:
        PARAMS['roi_type'] = RID['roi_type']
        PARAMS['roi_id'] = RID['roi_id']
        PARAMS['rid_hash'] = RID['rid_hash']
    
    return PARAMS

#-----------------------------------------------------
#           MAIN SET OF ACTIONS
#-----------------------------------------------------

def main(options):

    retinoid = create_retinoid(options)

    print "****************************************************************"
    print "Created ANALYSIS ID."
    print "----------------------------------------------------------------"
    pp.pprint(retinoid)
    print "****************************************************************"


if __name__ == '__main__':
    main(sys.argv[1:])






