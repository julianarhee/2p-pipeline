#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 20:31:55 2018

@author: juliana
"""

import os
import sys
import logging
import json
import traceback
from pipeline.python.rois.coregister_rois import collate_slurm_output
from pipeline.python.rois.get_rois import standardize_rois
from pipeline.python.set_roi_params import post_rid_cleanup

def collate_results(rid_path):
    coreg_results_path = None 
    roi_hash = os.path.splitext(os.path.split(rid_path)[-1])[0].split('_')[-1]
    logdir = os.path.join(os.path.split(rid_path)[0], "logging_%s" % roi_hash)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    print "Logging to dir: %s" % logdir

    logging.basicConfig(level=logging.DEBUG, filename="%s/logfile_%s_collate_coreg" % (logdir, roi_hash), filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")

    # Collate coregistered ROIs:
    logging.info("RID %s -- starting collating COREG output across files" % roi_hash)
    logging.info(rid_path)
    try:
        coreg_results_path = collate_slurm_output(rid_path, rootdir='/n/coxfs01/2p-data')
        logging.info("RID %s -- collating complete!" % roi_hash)
    except Exception as e:
        print "***Error collating results..."
        traceback.print_exc()

    return coreg_results_path


def format_rois(rid_path, coreg_results_path):
    mask_filepath = None
    roi_hash = os.path.splitext(os.path.split(rid_path)[-1])[0].split('_')[-1]
    try:
        with open(rid_path, 'r') as f:
            RID = json.load(f)
        session_dir = RID['DST'].split('/ROIs/')[0]
        roi_id = RID['roi_id']
        zproj_type = 'mean'
        auto = True
        check_motion = RID['PARAMS']['eval']['check_motion']
        mcmetric = RID['PARAMS']['eval']['mcmetric']
        keep_good_rois = RID['PARAMS']['options']['keep_good_rois']
         
        # FORMAT coregistered ROIs to standard:
        logging.info('RID %s -- standarizing ROI format.' % roi_hash)
        mask_filepath = standardize_rois(session_dir, roi_id, auto=auto, zproj_type=zproj_type, check_motion=check_motion, mcmetric=mcmetric, coreg_results_path=coreg_results_path, keep_good_rois=keep_good_rois)
    except Exception as e:
        print "***Error formatting ROIs."
        traceback.print_exc()
 
    return mask_filepath
 

def main():

    rid_path = sys.argv[1]

    coreg_results_path = collate_results(rid_path)
    logging.info("Coreg results path: %s" % coreg_results_path)

    mask_filepath = format_rois(rid_path, coreg_results_path)
    logging.info('COMPLETE! Mask file saved to: %s' % mask_filepath)
    
    if coreg_results_path is not None and mask_filepath is not None: 
        logging.info('Cleaning up tmp RID file...')
        roi_hash = os.path.splitext(os.path.split(rid_path)[-1])[0].split('_')[-1]
        session_dir = rid_path.split('/ROIs/')[0]
        post_rid_cleanup(session_dir, roi_hash)
        logging.info('****DONE!****')
    
if __name__=="__main__":
    main()
