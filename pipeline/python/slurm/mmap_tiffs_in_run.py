#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 16:49:35 2018

@author: julianarhee
"""

import os
import sys
import logging
from pipeline.python.rois.caiman2D import par_mmap_tiffs

def main():
    
    pid_path = sys.argv[1]
    roi_hash = os.path.splitext(os.path.split(pid_path)[-1])[0].split('_')[-1]
    logdir = os.path.join(os.path.split(pid_path)[0], "logging_%s" % roi_hash)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    print "Logging to: %s" % logdir

    logging.basicConfig(level=logging.DEBUG, filename="%s/logfile_%s_memmap" % (logdir, roi_hash), filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    
    logging.info("RID %s -- starting memmapping ..." % roi_hash) 
    logging.info(pid_path)
    
    mmap_paths = par_mmap_tiffs(pid_path)
    
    logging.info("FINISHED memmapping tiffs from RID:\n%s" % pid_path)
    logging.info("Created %i .mmap files." % len(mmap_paths))
    logging.info("RID %s -- memmapping done!" % roi_hash)

if __name__=="__main__":
     
    main()
    
    
