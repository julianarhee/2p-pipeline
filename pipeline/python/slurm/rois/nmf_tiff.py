#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 16:49:35 2018

@author: julianarhee
"""

import os
import sys
import logging
import json
from pipeline.python.rois.caiman2D import extract_nmf_from_rid, par_mmap_tiffs

def main():

    rid_path = sys.argv[1]
    file_num = int(sys.argv[2])
    nproc = sys.argv[3]
    if len(nproc) == 0:
        nproc = 12
    else:
        nproc = int(nproc)


    roi_hash = os.path.splitext(os.path.split(rid_path)[-1])[0].split('_')[-1]
    logdir = os.path.join(os.path.split(rid_path)[0], "logging_%s" % roi_hash)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    print "Logging to dir: %s" % logdir

    logging.basicConfig(level=logging.DEBUG, filename="%s/logfile_%s_nmf_%i" % (logdir, roi_hash, file_num), filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")

    # First, check that mmapped files exist:
    with open(rid_path, 'r') as f:
        tRID = json.load(f)
    tmp_mmap_dir = tRID['PARAMS']['mmap_source']
    if not os.path.isdir(tmp_mmap_dir):
        mmap = True
    elif len([t for t in os.listdir(tmp_mmap_dir) if t.endswith('mmap')]) < file_num:
        mmap = True
    else:
        mmap = False

    if mmap is True:
        print "MEMMAP!"
        logging.info('RID %s -- Need to memmapping tifs. Starting...' % roi_hash)
        mmap_paths = par_mmap_tiffs(rid_path)
        logging.info('RID %s -- Done memmapping!' % roi_hash)


    logging.info("RID %s -- starting ROI extraction for File %i" % (roi_hash, file_num))
    logging.info(rid_path)

    nmfopts_hash, ngood_rois = extract_nmf_from_rid(rid_path, file_num, nproc=nproc, rootdir='/n/coxfs01/2p-data')

    logging.info("FINISHED cNMF roi extraction for File %i:\n%s" % (file_num, roi_hash))
    logging.info("Found %i ROIs that pass initial evaluation." % ngood_rois)

    logging.info("RID %s -- extraction done!" % roi_hash)

if __name__=="__main__":
    main()

