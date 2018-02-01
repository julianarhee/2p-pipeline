#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 16:49:35 2018

@author: julianarhee
"""

import os
import sys
import logging
from pipeline.python.rois.caiman2D import extract_nmf_from_rid

def main():
    
    rid_path = sys.argv[1]
    file_num = int(sys.argv[2])

    roi_hash = os.path.splitext(os.path.split(rid_path)[-1])[0].split('_')[-1]
    
    logging.basicConfig(level=logging.DEBUG, filename="logfile_%s_nmf_%i" % (roi_hash, file_num), filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    
    logging.info("RID %s -- starting ROI extraction for File %i" % (roi_hash, file_num))
    logging.info(rid_path)
    
    nmfopts_hash, ngood_rois = extract_nmf_from_rid(rid_path, file_num)
    
    logging.info("FINISHED cNMF roi extraction for File %i:\n%s" % (roi_hash, file_num))
    logging.info("Found %i ROIs that pass initial evaluation." % ngood_rois)
    
    logging.info("RID %s -- extraction done!" % roi_hash)

if __name__=="__main__":
    main()
    
