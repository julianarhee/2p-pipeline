#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 20:31:55 2018

@author: juliana
"""

import os
import sys
import logging
from pipeline.python.rois.coregister_rois import collate_slurm_output

def main():

    rid_path = sys.argv[1]

    roi_hash = os.path.splitext(os.path.split(rid_path)[-1])[0].split('_')[-1]
    logdir = os.path.join(os.path.split(rid_path)[0], "logging_%s" % roi_hash)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    print "Logging to dir: %s" % logdir

    logging.basicConfig(level=logging.DEBUG, filename="%s/logfile_%s_collate_coreg" % (logdir, roi_hash), filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")


    logging.info("RID %s -- starting collating COREG output across files" % (roi_hash)
    logging.info(rid_path)

    coreg_results_path = collate_slurm_output(rid_path, rootdir='/n/coxfs01/2p-data')

    logging.info("RID %s -- collating complete!" % roi_hash)
    logging.info("Coreg results path: %s" % coreg_results_path)

if __name__=="__main__":
    main()
