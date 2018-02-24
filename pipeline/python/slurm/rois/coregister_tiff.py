#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 20:53:37 2018

@author: juliana
"""
import os
import sys
import logging
from pipeline.python.rois.coregister_rois import coregister_file_by_rid

def main():

    rid_path = sys.argv[1]
    file_num = int(sys.argv[2])

    roi_hash = os.path.splitext(os.path.split(rid_path)[-1])[0].split('_')[-1]
    logdir = os.path.join(os.path.split(rid_path)[0], "logging_%s" % roi_hash)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    print "Logging to dir: %s" % logdir

    logging.basicConfig(level=logging.DEBUG, filename="%s/logfile_%s_coreg_%i" % (logdir, roi_hash, file_num), filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")


    logging.info("RID %s -- starting ROI coregistration for File %i" % (roi_hash, file_num))
    logging.info(rid_path)

    tmp_fpath = coregister_file_by_rid(rid_path, filenum=file_num, rootdir='/n/coxfs01/2p-data')

    logging.info("FINISHED COREGISRATION for File %i:" % (file_num))
    logging.info("Tmp coreg results for file saved to: %s" % tmp_fpath)

    logging.info("RID %s -- coregistration done!" % roi_hash)

if __name__=="__main__":
    main()

