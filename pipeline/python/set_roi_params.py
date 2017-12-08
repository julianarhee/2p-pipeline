#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 12:42:10 2017

@author: julianarhee
"""

import os
import json
import pprint
import re
import pkg_resources
import pandas as pd
import optparse
import sys
import hashlib
import copy
from caiman.source_extraction.cnmf import utilities as cmu
import caiman as cm

from checksumdir import dirhash
from stat import S_IREAD, S_IRGRP, S_IROTH, S_IWRITE, S_IWGRP, S_IWOTH
import shutil

pp = pprint.PrettyPrinter(indent=4)

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]

def create_rid(options):

    parser = optparse.OptionParser()

    # PATH opts:
    parser.add_option('-R', '--root', action='store', dest='rootdir', default='/nas/volume1/2photon/data', help='data root dir (root project dir containing all animalids) [default: /nas/volume1/2photon/data, /n/coxfs01/2pdata if --slurm]')
    parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')

    # Set specific session/run for current animal:
    parser.add_option('-S', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID')
    parser.add_option('-A', '--acq', action='store', dest='acquisition', default='FOV1', help="acquisition folder (ex: 'FOV1_zoom3x') [default: FOV1]")
    parser.add_option('-r', '--run', action='store', dest='run', default='', help="name of run dir containing tiffs to be processed (ex: gratings_phasemod_run1)")
#    parser.add_option('--default', action='store_true', dest='auto', default='store_false', help="Use all DEFAULT params, for params not specified by user (no interactive)")

    parser.add_option('-s', '--tiffsource', action='store', dest='tiffsource', default=None, help="name of folder containing tiffs to be processed (ex: processed001). should be child of <run>/processed/")
    parser.add_option('-t', '--sourcetype', action='store', dest='sourcetype', default='raw', help="type of source tiffs (e.g., bidi, raw, mcorrected) [default: 'raw']")
    parser.add_option('-o', '--roi-type', action='store', dest='roi_type', default='', help="roi type ['caiman2D', 'manual2D_circle', etc.]")

    (options, args) = parser.parse_args(options) 

    
    rootdir = options.rootdir
    animalid = options.animalid
    session = options.session
    acquisition = options.acquisition
    run = options.run
    tiffsource = options.tiffsource
    sourcetype = options.sourcetype
    roi_type = options.roi_type

    session_dir = os.path.join(rootdir, animalid, session)
    roi_dir = os.path.join(session_dir, 'ROIs')
    if not os.path.exists(roi_dir):
        os.makedirs(roi_dir)

    rundir = os.path.join(rootdir, animalid, session, acquisition, run)
    tiffpaths = get_tiff_paths(rootdir=rootdir, animalid=animalid, session=session, acquisition=acquisition, run=run,
                           tiffsource=tiffsource, sourcetype=sourcetype)

    tiff_dir = os.path.split(tiffpaths[0])[0]
    print "SRC: %s, found %i tiffs." % (tiff_dir, len(tiffpaths))
	 
    roi_options = None

    PARAMS = set_params(tiff_dir, roi_options, roi_type=roi_type)
    rid = initialize_rid(PARAMS, session_dir)

    roi_name = '_'.join((rid['roi_id'], rid['params_hashid']))
    curr_roi_dir = os.path.join(roi_dir, roi_name)
    if not os.path.exists(curr_roi_dir):
	os.makedirs(curr_roi_dir)

    # Rename RID fields to include roiparams hash:
    if rid['params_hashid'] not in rid['DST']:
	rid['DST'] = rid['DST'] + '_' + rid['params_hashid']
    update_roi_records(rid, session_dir)

    print "Created new ROI ID, hash:", rid['params_hashid']

    return rid

def get_tiff_paths(rootdir='', animalid='', session='', acquisition='', run='', tiffsource=None, sourcetype=None, auto=False):
#    rootdir = '/nas/volume1/2photon/data'
#    animalid = 'JR063'
#    session = '20171128_JR063_testbig'
#    acquisition = 'FOV1_zoom1x'
#    run = 'gratings_static_run1'
    #tiffsource = 'process002'
    #sourcetype = 'mcorrected'
    
    rundir = os.path.join(rootdir, animalid, session, acquisition, run)
    processed_dir = os.path.join(rundir, 'processed')
    
    if tiffsource is None:
        while True:
            if auto is True:
                tiffsource = 'raw'
                break
            tiffsource_idx = raw_input('No tiffsource specified. Enter <R> for raw, or <P> for processed: ')
            processed_dirlist = sorted([p for p in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, p))], key=natural_keys)
            if len(processed_dirlist) == 0 or tiffsource_idx == 'R':
                tiffsource = 'raw'
                if len(processed_dirlist) == 0:
                    print "No processed dirs... Using raw."
                confirm_tiffsource = raw_input('Press <Y> to use raw.')
                if confirm_tiffsource == 'Y':
                    break  
            elif len(processed_dirlist) > 0:
                for pidx, pfolder in enumerate(sorted(processed_dirlist, key=natural_keys)):
                    print pidx, pfolder
                tiffsource_idx = int(input("Enter IDX of processed source to use: "))
                tiffsource = processed_dirlist[tiffsource_idx]
                confirm_tiffsource = raw_input('Tiffs are %s? Press <Y> to confirm. ' % tiffsource)
                if confirm_tiffsource == 'Y':
                    break
                
    if sourcetype is None:
        while True:
            if auto is True or tiffsource == 'raw':
                sourcetype = 'raw'
                break
            print "Specified PROCESSED tiff source, but not process type."
            curr_processed_dir = os.path.join(rundir, 'processed', tiffsource)
            processed_typlist = sorted([t for t in os.listdir(curr_processed_dir) if os.path.isdir(os.path.join(curr_processed_dir, t))], key=natural_keys)
            for tidx, tname in enumerate(processed_typlist):
                print tidx, tname
            sourcetype_idx = int(input('Enter IDX of processed dir to use: '))
            sourcetype = processed_typlist[sourcetype_idx]
            confirm_sourcetype = raw_input('Tiffs are from %s? Press <Y> to confirm. ' % sourcetype)
            if confirm_sourcetype == 'Y':
                break
            
    try:
        if 'processed' in tiffsource:
            tiffsrc_parent = [p for p in os.listdir(os.path.join(rundir, 'processed')) if tiffsource in p][0]
            tiffsrc_folder = [p for p in os.listdir(os.path.join(rundir, 'processed', tiffsrc_parent)) if sourcetype in p][0]
            tiff_path = os.path.join(rundir, 'processed', tiffsrc_parent, tiffsrc_folder)
        else:
            tiffsrc_parent = [p for p in os.listdir(rundir) if 'raw' in p][0]
            tiff_path = os.path.join(rundir, tiffsrc_parent)
            
        tiff_fns = [t for t in os.listdir(tiff_path) if t.endswith('tif')]
        tiffpaths = sorted([os.path.join(tiff_path, fn) for fn in tiff_fns], key=natural_keys)
        print "Found %i TIFFs for cNMF ROI extraction." % len(tiff_fns)
    except:
        print "ERROR: tiffsource %s of type %s not found in run dir %s" % (tiffsource, sourcetype, rundir)

    return tiffpaths

def load_roidict(session_dir):
    
    roidict = None
    
    session = os.path.split(session_dir)[1]
    roidict_filepath = os.path.join(session_dir, 'ROIs', 'rids_%s.json' % session)
    
    # Load analysis "processdict" file:
    if os.path.exists(roidict_filepath):
        print "exists!"
        with open(roidict_filepath, 'r') as f:
            roidict = json.load(f)
            
    return roidict


def set_params(tiffsource, roi_options, roi_type=''):
    
    PARAMS = dict()
    
    PARAMS['tiff_source'] = tiffsource
    
    tiffparent = os.path.split(tiffsource)[0]
    
    # Check mmap hash:
    if roi_type == 'caiman2D':
        if roi_options is None:
	    tiffs = [t for t in os.listdir(tiffsource) if t.endswith('tif')]
	    m_orig = cm.load_movie_chain([os.path.join(tiffsource, tiffs[0])])
            roi_options = cmu.CNMFSetParams(m_orig)
	else:
	    mmap_dirs = [m for m in os.listdir(tiffparent) if 'mmap' in m]
	    if len(mmap_dirs) == 1:
		mmap_dir =  os.path.join(tiffparent, mmap_dirs[0])
	    else:
		mmap_dir = tiffsource + '_mmap'
	    
	    excluded_files = [f for f in os.listdir(mmap_dir) if not f.endswith('mmap')]
	    mmap_hash = dirhash(mmap_dir, 'sha1', excluded_files=excluded_files)[0:6]
	    if mmap_hash not in mmap_dir:
		PARAMS['mmap_source'] = mmap_dir + '_' + mmap_hash
		os.rename(mmap_dir, PARAMS['mmap_source'])
		print "Renamed mmap with hash:", PARAMS['mmap_source']
	    else:
		PARAMS['mmap_source'] = mmap_dir
	roi_options = to_json(roi_options)

    elif 'manual' in roi_type:
        roi_options = dict()
        roi_options['roi_type'] = roi_type
        	
    PARAMS['options'] = roi_options
    PARAMS['roi_type'] = roi_type
   
    PARAMS['hashid'] = hashlib.sha1(json.dumps(PARAMS, sort_keys=True)).hexdigest()[0:6]
    print "PARAMS hashid is:", PARAMS['hashid']

    return PARAMS


def get_roi_id(PARAMS, session_dir, auto=False):
    
    roi_id = None
    
    print "********************************"
    print "Checking previous ROI IDs..."
    print "********************************"
    
    # Load previously created PIDs:
    roidict = load_roidict(session_dir)
    
    #Check current PARAMS against existing PId params by hashid:
    if roidict is None or len(roidict.keys()) == 0:
        roidict = dict()
        existing_rids = []
        is_new_rid = True
        print "No existing RIDs found."
    else:
        existing_rids = sorted([str(k) for k in roidict.keys()], key=natural_keys)
        matching_rids = sorted([rid for rid in existing_rids if roidict[rid]['hashid'] == PARAMS['hashid']], key=natural_keys)
        is_new_rid = False
        if len(matching_rids) > 0:            
            while True:
                print "WARNING **************************************"
                print "Current param config matches existing RID:"
                for ridx, rid in enumerate(matching_rids):
                    print "******************************************"
                    print ridx, rid
                    pp.pprint(roidict[rid])
                if auto is True:
                    check_ridx = ''
                else:
                    check_ridx = raw_input('Enter IDX of rid to re-use, or hit <ENTER> to create new: ')
                if len(check_ridx) == 0:
                    is_new_rid = True
                    break
                else:
                    confirm_reuse = raw_input('Re-use RID %s? Press <Y> to confirm, any key to try again:' % existing_rids[int(check_ridx)])
                    if confirm_reuse == 'Y':
                        is_new_rid = False
                        break
        else:
            is_new_rid = True

    if is_new_rid is True:
        # Create new PID by incrementing num of process dirs found:
        roi_id = 'rois%03d' % int(len(existing_rids)+1)
        print "Creating NEW roi ID: %s" % roi_id
    else:
        # Re-using an existing PID:
        roi_id = existing_rids[int(check_ridx)]
        print "Reusing existing rid: %s" % roi_id
 
    return roi_id

def to_json(curropts):  
    jsontypes = (list, tuple, str, int, float)
    for pkey in curropts.keys():
        if isinstance(curropts[pkey], dict):
            for subkey in curropts[pkey].keys():
                if curropts[pkey][subkey] is not None and not isinstance(curropts[pkey][subkey], jsontypes) and len(curropts[pkey][subkey].shape) > 1:
                    curropts[pkey][subkey] = curropts[pkey][subkey].tolist()
    return curropts


def initialize_rid(PARAMS, session_dir, auto=False):
    
    print "************************"
    print "Initialize ROI ID."
    print "************************"
    roi_id = get_roi_id(PARAMS, session_dir, auto=auto)

    rid = dict()
    version = pkg_resources.get_distribution('pipeline').version
    
    rid['version'] = version 
    rid['roi_id'] = roi_id
    rid['SRC'] = PARAMS['tiff_source'] #source_dir
    rid['DST'] = os.path.join(session_dir, 'ROIs', roi_id)
    rid['roi_type'] = PARAMS['roi_type']

    rid['PARAMS'] = PARAMS
    
    # deal with jsonify:
    curropts = to_json(PARAMS['options'])
    rid['PARAMS']['options'] = curropts

    # TODO:  Generate hash for full PID dict
    rid['params_hashid'] = hashlib.sha1(json.dumps(rid, sort_keys=True)).hexdigest()[0:6]
        
    return rid


def update_roi_records(rid, session_dir):

    print "************************"
    print "Updating JSONS..."
    print "************************"
    session = os.path.split(session_dir)[1]
    roidict_filepath = os.path.join(session_dir, 'ROIs', 'rids_%s.json' % session)
    
    if os.path.exists(roidict_filepath):
        with open(roidict_filepath, 'r') as f:
            roidict = json.load(f)
    else:
        roidict = dict()

    roi_id = rid['roi_id'] 
    roidict[roi_id] = rid

    #% Update Process Info DICT:
    with open(roidict_filepath, 'w') as f:
        json.dump(roidict, f, sort_keys=True, indent=4)

    print "ROI Info UPDATED."
    

def main(options):
    
    rid = create_rid(options)
    
    print "****************************************************************"
    print "Created RID."
    print "----------------------------------------------------------------"
    pp.pprint(rid)
    print "****************************************************************"
    
if __name__ == '__main__':
    main(sys.argv[1:]) 

#%%
#tiffpaths = get_tiff_paths(rootdir=rootdir, animalid=animalid, session=session, acquisition=acquisition, run=run)

#PARAMS = set_roi_params(tiff_dir, cnm.options)
#rid = initialize_rid(PARAMS, session_dir)
#update_roi_records(rid, session_dir)
