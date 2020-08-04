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
import glob
from skimage import exposure

import cv2
import imutils

import pylab as pl
import numpy as np
from pipeline.python.utils import natural_keys, get_source_info, replace_root, write_dict_to_json
import pprint
pp = pprint.PrettyPrinter(indent=4)
import cPickle as pkl

#%%


def load_roi_coords(animalid, session, fov, roiid=None, 
                    convert_um=True, traceid='traces001', 
                    create_new=False,rootdir='/n/coxfs01/2p-data'):

    from pipeline.python.retinotopy import convert_coords as cc
    fovinfo = None
    roiid = get_roiid_from_traceid(animalid, session, fov, traceid=traceid)
    
    # create outpath
    roidir = glob.glob(os.path.join(rootdir, animalid, session, 
                        'ROIs', '%s*' % roiid))[0]
    fovinfo_fpath = os.path.join(roidir, 'fov_info.pkl')

    if not create_new:
        try:
            with open(fovinfo_fpath, 'rb') as f:
                fovinfo = pkl.load(f)
            assert 'fov_xpos' in fovinfo.keys() and 'ml_pos' in fovinfo.keys(), "Bad fovinfo file, redoing"
        except AssertionError:
            create_new = True

    if create_new:
        print("... calculating roi-2-fov info")
        masks, zimg = load_roi_masks(animalid, session, fov, rois=roiid)
        fovinfo = cc.calculate_roi_coords(masks, zimg, convert_um=convert_um)
        with open(fovinfo_fpath, 'wb') as f:
            pkl.dump(fovinfo, f, protocol=pkl.HIGHEST_PROTOCOL)

    return fovinfo

    
def get_roiid_from_traceid(animalid, session, fov, run_type=None, 
                            traceid='traces001', rootdir='/n/coxfs01/2p-data'):
    
    if run_type is not None:
        if int(session) < 20190511 and 'rfs' in run_type:
            run_name = 'gratings'

        a_traceid_dict = glob.glob(os.path.join(rootdir, animalid, session, 
                                    fov, '*%s*' % run_type, 'traces', 
                                    'traceids*.json'))[0]
    else:
        a_traceid_dict = glob.glob(os.path.join(rootdir, animalid, session, 
                                    fov, '*run*', 'traces', 'traceids*.json'))[0]
    with open(a_traceid_dict, 'r') as f:
        tracedict = json.load(f)
    
    tid = tracedict[traceid]
    roiid = tid['PARAMS']['roi_id']
    
    return roiid

def load_roi_masks(animalid, session, fov, rois=None, rootdir='/n/coxfs01/2p-data'):
    mask_fpath = glob.glob(os.path.join(rootdir, animalid, session, 
                                'ROIs', '%s*' % rois, 'masks.hdf5'))[0]
    mfile = h5py.File(mask_fpath, 'r')

    # Load and reshape masks
    masks = mfile[mfile.keys()[0]]['masks']['Slice01'][:].T
    #print(masks.shape)
    mfile[mfile.keys()[0]].keys()

    zimg = mfile[mfile.keys()[0]]['zproj_img']['Slice01'][:].T
    
    return masks, zimg




#%% PLOTTING..................................................................


def get_roi_contours(roi_masks, roi_axis=0):
    
    cnts = []
    nrois = roi_masks.shape[roi_axis]
    for ridx in range(nrois):
        if roi_axis == 0:
            im = np.copy(roi_masks[ridx, :, :])
        else:
            im = np.copy(roi_masks[:, :, ridx])
        im[im>0] = 1
        im[im==0] = np.nan #1
        im = im.astype('uint8')
        edged = cv2.Canny(im, 0, 0.9)
        tmp_cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        tmp_cnts = tmp_cnts[0] if imutils.is_cv2() else tmp_cnts[1]
        cnts.append((ridx, tmp_cnts[0]))
    print "Created %i contours for rois." % len(cnts)

    return cnts

def uint16_to_RGB(img):
    im = img.astype(np.float64)/img.max()
    im = 255 * im
    im = im.astype(np.uint8)
    rgb = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    return rgb

def plot_roi_contours(zproj, cnts, clip_limit=0.01, ax=None, 
                          roi_highlight = [], cmap=None,
                          label_all=False, roi_color_default=(127, 127, 127),
                          label_highlight=False, roi_color_highlight=(0, 255, 0),
                          thickness=1, fontsize=12):


    if ax is None:
        fig, ax = pl.subplots(1, figsize=(10,10))


    #clip_limit=0.02
    # Create ZPROJ img to draw on:
    refRGB = uint16_to_RGB(zproj)        
#    p2, p98 = np.percentile(refRGB, (1, 99))
#    img_rescale = exposure.rescale_intensity(refRGB, in_range=(p2, p98))
    im_adapthist = exposure.equalize_adapthist(refRGB, clip_limit=clip_limit)
    im_adapthist *= 256
    im_adapthist= im_adapthist.astype('uint8')
    ax.imshow(im_adapthist) #pl.figure(); pl.imshow(refRGB) # cmap='gray')

    orig = im_adapthist.copy()

    # loop over the contours individually
    for rid, cnt in enumerate(cnts):
        contour = np.squeeze(cnt[-1])
        

        # draw the contours on the image
        orig = refRGB.copy()

        if len(contour.shape) == 1:
            xpos = contour[0]
            ypos = contour[1]
            single = True
        else:
            xpos = contour[-1, 0]
            ypos = contour[-1, 1]
            single = False
            
        if len(roi_highlight) > 0 and rid in roi_highlight:
            col255 = tuple([cval/255. for cval in roi_color_highlight])
            if label_highlight:
                ax.text(xpos, ypos, str(rid+1), color='gray', fontsize=fontsize)
        else:
            if cmap is None:
                col255 = tuple([cval/255. for cval in roi_color_default])
            else:
                curr_color = cmap[rid]
                if len(curr_color) == 4:
                    curr_color = curr_color[0:-1]
                if any(curr_color) > 1:
                    col255 = tuple([cval/255. for cval in curr_color])
                else:
                    col255 = curr_color
            if label_all:
                ax.text(xpos, ypos, str(rid+1), color='gray', fontsize=fontsize)

        #cv2.drawContours(orig, cnt, -1, col255, thickness)
        if single:
            ax.plot(contour[0], contour[1], color=col255)
        else:
            ax.plot(contour[:, 0], contour[:, 1], color=col255)
        ax.imshow(orig)
        
#%%
def save_roi_params(RID, evalparams=None, keep_good_rois=True, excluded_tiffs=[], rootdir=''):
    roiparams = dict()
    rid_dir = RID['DST']
    if rootdir not in rid_dir:
        session_dir = rid_dir.split('/ROIs/')[0]
        session = os.path.split(session_dir)[-1]
        animalid = os.path.split(os.path.split(session_dir)[0])[-1]
        rid_dir = replace_root(rid_dir, rootdir, animalid, session)

    roiparams['eval'] = evalparams
    roiparams['keep_good_rois'] = keep_good_rois
    roiparams['excluded_tiffs'] = excluded_tiffs
    roiparams['roi_type'] = RID['roi_type']
    roiparams['roi_id'] = RID['roi_id']
    roiparams['rid_hash'] = RID['rid_hash']

    roiparams_filepath = os.path.join(rid_dir, 'roiparams.json') # % (str(roi_id), str(rid_hash)))
    write_dict_to_json(roiparams, roiparams_filepath)

    print "Saved ROI params to: %s" % roiparams_filepath

    return roiparams


def get_roi_eval_path(src_roi_dir, eval_key, auto=False):
    src_eval_filepath = None
    src_eval = None
    try:
        print "-----------------------------------------------------------"
        print "Loading evaluation results for src roi set"
        # Load eval info:
        src_eval_filepath = os.path.join(src_roi_dir, 'evaluation', 'evaluation_%s' % eval_key, 'evaluation_results_%s.hdf5' % eval_key)
        assert os.path.exists(src_eval_filepath), "Specfied EVAL src file does not exist!\n%s" % src_eval_filepath
        src_eval = h5py.File(src_eval_filepath, 'r')
    except Exception as e:
        print "Error loading specified eval file:\n%s" % src_eval_filepath
        traceback.print_exc()
        print "-----------------------------------------------------------"
        try:
            evaldict_filepath = os.path.join(src_roi_dir, 'evaluation', 'evaluation_info.json')
            with open(evaldict_filepath, 'r') as f:
                evaldict = json.load(f)
            eval_list = sorted(evaldict.keys(), key=natural_keys)
            print "Found evaluation keys:"
            if auto is False:
                while True:
                    if len(eval_list) > 1:
                        for eidx, ekey in enumerate(eval_list):
                            print eidx, ekey
                            eval_select_idx = input('Select IDX of evaluation key to view: ')
                    else:
                        eval_select_idx = 0
                        print "Only 1 evaluation set found: %s" % eval_list[eval_select_idx]
                    pp.pprint(evaldict[eval_list[eval_select_idx]])
                    confirm_eval = raw_input('Enter <Y> to use this eval set, or <n> to return: ')
                    if confirm_eval == 'Y':
                        eval_key = eval_list[eval_select_idx].split('evaluation_')[-1]
                        print "Using key: %s" % eval_key
                        break
            else:
                print "Auto is ON, using most recent evaluation set: %s" % eval_key
                eval_key = eval_list[-1].split('evaluation_')[-1]
                pp.pprint(evaldict[eval_list[-1]])

            src_eval_filepath = os.path.join(src_roi_dir, 'evaluation', 'evaluation_%s' % eval_key, 'evaluation_results_%s.hdf5' % eval_key)
            src_eval = h5py.File(src_eval_filepath, 'r')
        except Exception as e:
            print "ERROR: Can't load source evaluation file - %s" % eval_key
            traceback.print_exc()
            print "Aborting..."
            print "-----------------------------------------------------------"

    return src_eval_filepath

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
def get_info_from_tiff_dir(tiff_sourcedir, session_dir):
    info = dict()
    #path_parts = tiff_sourcedir.split(session_dir)[-1].split('/')
    session = os.path.split(session_dir)[-1]
    animalid = os.path.split(os.path.split(session_dir)[0])[-1]#path_parts[1]
    if 'processed' in tiff_sourcedir:
        path_parts = tiff_sourcedir.split('/processed/')
        process_dirname = os.path.split(path_parts[1])[0]
        process_id = process_dirname.split('_')[0]
    else: #raw:
        path_parts = tiff_sourcedir.split('/raw')[0]
        suffix = tiff_sourcedir.split(path_parts)[-1]
        process_dirname = suffix.split('/')[1]
        process_id = 'raw'
    run = os.path.split(path_parts[0])[-1] #path_parts[2]
    acquisition = os.path.split(os.path.split(path_parts[0])[0])[-1]

    info['acquisition'] = acquisition
    info['run'] = run
    info['session'] = session
    info['process_id'] = process_id
    info['process_dirname'] = process_dirname
    info['animalid'] = animalid

    return info

def get_source_paths(session_dir, RID, check_motion=True, subset=False, mcmetric='zproj_corrcoefs', rootdir=''): #, acquisition='', run='', process_id=''):
    '''
    Get fullpaths to ROI source files, original tiff/mmap files, filter by MC-evaluation for excluded tiffs.
    Provide acquisition, run, process_id if source is NOT motion-corrected (default reference file and channel = 1).
    '''
    #if acquisition=='' or run=='' or process_id=='':
    tiff_sourcedir = RID['SRC']
    info = get_info_from_tiff_dir(tiff_sourcedir, session_dir)
    acquisition = info['acquisition']
    run = info['run']
    process_id = info['process_id']

    print "Getting source paths:"
    print "ACQUISITION: %s | RUN: %s | PROCESS-ID: %s..." % (acquisition, run, process_id)

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
        src_roidict_filepath = os.path.join(src_session_dir, 'ROIs', 'rids_%s.json' % src_session)
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

    elif 'manual2D' in roi_type:
        rid_src_dir = RID['SRC']
        print "SRC: %s, ROOT: %s" % (rid_src_dir, rootdir)
        if rootdir not in rid_src_dir:
            rid_src_dir = replace_root(rid_src_dir, rootdir, animalid, session)
        roi_source_paths = sorted([os.path.join(rid_src_dir, t) for t in os.listdir(rid_src_dir) if t.endswith('tif')], key=natural_keys)
        tiff_source_paths = roi_source_paths

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
                exclude_str = ','.join([str(i) for i in fidxs_to_exclude])
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
