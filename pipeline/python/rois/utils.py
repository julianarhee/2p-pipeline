#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 17:56:01 2018

@author: julianarhee
"""
import matplotlib
matplotlib.use('Agg')
import os
import h5py
import json
import datetime
import optparse
import pprint
import time
import traceback
import re
import pylab as pl
import numpy as np
from pipeline.python.utils import natural_keys, get_source_info, replace_root
import pprint
pp = pprint.PrettyPrinter(indent=4)

#%%


def load_RID(session_dir, roi_id, auto=False):

    roi_base_dir = os.path.join(session_dir, 'ROIs') #acquisition, run)
    session = os.path.split(session_dir)[1]
    roidict_path = os.path.join(session_dir, 'ROIs', 'rids_%s.json' % session)
    tmp_rid_dir = os.path.join(roi_base_dir, 'tmp_rids')

    try:
        print "Loading params for ROI SET, id %s" % roi_id
        roidict_path = os.path.join(roi_base_dir, 'rids_%s.json' % session)
        with open(roidict_path, 'r') as f:
            roidict = json.load(f)
        RID = roidict[roi_id]
        #pp.pprint(RID)
    except Exception as e:
        print "No ROI SET entry exists for specified id: %s" % roi_id
        print e
        try:
            print "Checking tmp roi-id dir..."
            if auto is False:
                while True:
                    tmpfns = [t for t in os.listdir(tmp_rid_dir) if t.endswith('json')]
                    for ridx, ridfn in enumerate(tmpfns):
                        print ridx, ridfn
                    userchoice = raw_input("Select IDX of found tmp roi-id to view: ")
                    with open(os.path.join(tmp_rid_dir, tmpfns[int(userchoice)]), 'r') as f:
                        tmpRID = json.load(f)
                    print "Showing tid: %s, %s" % (tmpRID['roi_id'], tmpRID['rid_hash'])
                    pp.pprint(tmpRID)
                    userconfirm = raw_input('Press <Y> to use this roi ID, or <q> to abort: ')
                    if userconfirm == 'Y':
                        RID = tmpRID
                        break
                    elif userconfirm == 'q':
                        break
        except Exception as E:
            print "---------------------------------------------------------------"
            print "No tmp roi-ids found either... ABORTING with error:"
            print E
            print "---------------------------------------------------------------"

    return RID

#%%
def get_source_paths(session_dir, RID, check_motion=True, subset=False, mcmetric='zproj_corrcoefs', rootdir=''): #, acquisition='', run='', process_id=''):
    '''
    Get fullpaths to ROI source files, original tiff/mmap files, filter by MC-evaluation for excluded tiffs.
    Provide acquisition, run, process_id if source is NOT motion-corrected (default reference file and channel = 1).
    '''
    #if acquisition=='' or run=='' or process_id=='':
    tiff_sourcedir = RID['SRC']
    path_parts = tiff_sourcedir.split(session_dir)[-1].split('/')
    acquisition = path_parts[1]
    run = path_parts[2]
    process_dirname = path_parts[4]
    process_id = process_dirname.split('_')[0]
    print "Getting source paths: %s, %s, %s..." % (acquisition, run, process_id)

    session = os.path.split(session_dir)[-1]
    animalid = os.path.split(os.path.split(session_dir)[0])[-1]
    print "SESSION:", session
    print "ANIMALID:", animalid
    rootdir = rootdir

    roi_source_dir = RID['DST']
    if rootdir not in roi_source_dir:
        print "ORIG:", roi_source_dir
        roi_source_dir = replace_root(roi_source_dir, rootdir, animalid, session)
        print "NEW:", roi_source_dir
    roi_type = RID['roi_type']
    #roi_id = RID['roi_id']
    excluded_tiffs = []
    mcmetrics_filepath = None

    if roi_type == 'caiman2D':
        # Get ROI source paths:
        src_nmf_dir = os.path.join(roi_source_dir, 'nmfoutput')
        roi_source_paths = sorted([os.path.join(src_nmf_dir, n) for n in os.listdir(src_nmf_dir) if n.endswith('npz')], key=natural_keys) # Load nmf files
        # Get TIFF/mmap source from which ROIs were extracted:
        src_mmap_dir = RID['PARAMS']['mmap_source']
        if rootdir not in src_mmap_dir:
            src_mmap_dir = replace_root(src_mmap_dir, rootdir, animalid, session)
            print "***SRC MMAP:", src_mmap_dir
        tiff_source_paths = sorted([os.path.join(src_mmap_dir, f) for f in os.listdir(src_mmap_dir) if f.endswith('mmap')], key=natural_keys)

    elif roi_type == 'coregister':
        src_rid_dir = RID['PARAMS']['options']['source']['roi_dir']
        if rootdir not in src_rid_dir:
            src_rid_dir = replace_root(src_rid_dir, rootdir, animalid, session)
            print "***SRC RID:", src_rid_dir
        src_roi_id = RID['PARAMS']['options']['source']['roi_id']
        src_roi_type = RID['PARAMS']['options']['source']['roi_type']

        src_session_dir = os.path.split(os.path.split(src_rid_dir)[0])[0]
        src_session = os.path.split(src_session_dir)[1]
        src_roidict_filepath = os.path.join(session_dir, 'ROIs', 'rids_%s.json' % src_session)
        with open(src_roidict_filepath, 'r') as f:
            src_roidict = json.load(f)
        if src_roi_type == 'caiman2D':
            # Get ROI source paths:
            src_nmf_dir = os.path.join(src_rid_dir, 'nmfoutput')
            roi_source_paths = sorted([os.path.join(src_nmf_dir, n) for n in os.listdir(src_nmf_dir) if n.endswith('npz')], key=natural_keys)
            # Get TIFF/mmap source from which ROIs were extracted:
            src_mmap_dir = src_roidict[src_roi_id]['PARAMS']['mmap_source']
            if rootdir not in src_mmap_dir:
                src_mmap_dir = replace_root(src_mmap_dir, rootdir, animalid, session)
                print "***SRC MMAP:", src_mmap_dir
            tiff_source_paths = sorted([os.path.join(src_mmap_dir, f) for f in os.listdir(src_mmap_dir) if f.endswith('mmap')], key=natural_keys)

    # Get filenames for matches between roi source and tiff source:
#    if subset is False:
#        assert len(roi_source_paths) == len(tiff_source_paths), "Mismatch in N tiffs (%i) and N roi sources (%i)." % (len(roi_source_paths), len(tiff_source_paths))
    filenames = []
    for roi_src in roi_source_paths:
        # Get filename base
        # filenames = sorted([str(re.search('File(\d{3})', nmffile).group(0)) for nmffile in roi_source_paths], key=natural_keys)
        filenames.append(str(re.search('File(\d{3})', roi_src).group(0)))
    filenames = sorted(filenames, key=natural_keys)
    print "Found %i ROI SOURCE files." % len(roi_source_paths)

    if check_motion is True:
        print "Checking MC eval, metric: %s" % mcmetric
        filenames, excluded_tiffs, mcmetrics_filepath = check_mc_evaluation(RID, filenames, mcmetric_type=mcmetric,
                                                       acquisition=acquisition, run=run, process_id=process_id,
                                                       rootdir=rootdir, animalid=animalid, session=session)
        if len(excluded_tiffs) > 0:
            bad_roi_fns = []
            bad_tiff_fns = []
            for badfn in excluded_tiffs:
                if len([r for r in roi_source_paths if badfn in r]) > 0:
                    bad_roi_fns.append([r for r in roi_source_paths if badfn in r][0])
                if len([r for r in tiff_source_paths if badfn in r]) > 0:
                    bad_tiff_fns.append([r for r in tiff_source_paths if badfn in r][0])
            roi_source_paths = sorted([r for r in roi_source_paths if r not in bad_roi_fns], key=natural_keys)
            tiff_source_paths = sorted([r for r in tiff_source_paths if r not in bad_tiff_fns], key=natural_keys)
        print "%i MC-fail tiffs found. Returning %i roi source paths, with %i corresponding tiff sources." % (len(excluded_tiffs), len(roi_source_paths), len(tiff_source_paths))
    else:
        print "You told me not to check motion-correction evaluation. Only returning found roi and tiff source paths..."

    return roi_source_paths, tiff_source_paths, filenames, excluded_tiffs, mcmetrics_filepath

#%% If motion-corrected (standard), check evaluation:
def check_mc_evaluation(RID, filenames, mcmetric_type='zproj_corrcoefs',
                            acquisition='', run='', process_id='', rootdir='', animalid='', session=''):

    # TODO:  Make this include other eval types?
    mcmetric_options = ['zproj_corrcoefs', 'within_file']

    if mcmetric_type not in mcmetric_options:
        print "Unknown MC METRIC type specified: %s. Using default..." % mcmetric_type
        mcmetric_type='zproj_corrcoefs'

    roi_src_dir = RID['SRC']
    if rootdir not in roi_src_dir:
        roi_src_dir = replace_root(roi_src_dir, rootdir, animalid, session)

    #print "Loading Motion-Correction Info...======================================="
    mcmetrics_filepath = None
    excluded_tiffs = []
    mc_evaluated = False
    if 'mcorrected' in RID['SRC']:
        try:
            mceval_dir = '%s_evaluation' % roi_src_dir
            assert 'mc_metrics.hdf5' in os.listdir(mceval_dir), "MC output file not found!"
            mcmetrics_filepath = os.path.join(mceval_dir, 'mc_metrics.hdf5')
            mcmetrics = h5py.File(mcmetrics_filepath, 'r')
            print "Loaded MC eval file. Found metric types:"
            for ki, k in enumerate(mcmetrics):
                print ki, k
            mc_evaluated = True
        except Exception as e:
            print e
            print "Unable to load motion-correction evaluation info."

    if mc_evaluated is True:
        # Use zprojection corrs to find bad files:
        bad_files = mcmetrics['zproj_corrcoefs'].attrs['bad_files']
        if len(bad_files) > 0:
            print "Found %i files that fail MC metric %s:" % (len(bad_files), mcmetric_type)
            for b in bad_files:
                print b
            fidxs_to_exclude = [int(f[4:]) for f in bad_files]
            if len(fidxs_to_exclude) > 1:
                exclude_str = ','.join([i for i in fidxs_to_exclude])
            else:
                exclude_str = str(fidxs_to_exclude[0])
        else:
            exclude_str = ''

        # Get process info from attrs stored in metrics file:
        mc_ref_channel = mcmetrics.attrs['ref_channel']
        mc_ref_file = mcmetrics.attrs['ref_file']
    else:
        session_dir = RID['DST'].split('/ROIs')[0]
        acquisition_dir = os.path.join(session_dir, acquisition)
        info = get_source_info(acquisition_dir, run, process_id)
        pp.pprint(info)
        mc_ref_channel = info['ref_channel']
        mc_ref_file = info['ref_filename']
        exclude_str = ''
        del info

    if len(exclude_str) > 0:
        filenames =[f for f in filenames if int(f[4:]) not in [int(x) for x in exclude_str.split(',')]]
        excluded_tiffs = ['File%03d' % int(fidx) for fidx in exclude_str.split(',')]

    print "-------------------------------------------------------------------"
    print "Motion-correction info :"
    if mc_evaluated is False:
        print "No MC evaluation found. Using MC INFO for reference."
    print "MC reference is %s, %s." % (mc_ref_file, mc_ref_channel)
    print "Found %i tiff files to exclude based on MC EVAL: %s." % (len(excluded_tiffs), mcmetric_type)
    print "-------------------------------------------------------------------"

    #print "======================================================================="

    return sorted(filenames, key=natural_keys), excluded_tiffs, mcmetrics_filepath
