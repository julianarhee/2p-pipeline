#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 17:19:12 2017

@author: julianarhee
"""
import os
import json
import pprint
import re
import optparse
import sys
from pipeline.python.preprocessing.process_raw import process_pid
#from multiprocessing import Process, Pool
import multiprocessing as mp
import psutil

pp = pprint.PrettyPrinter(indent=4)

# GENERAL METHODS:
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]


def get_curr_pid():
    p = psutil.Process(os.getpid())
    print "OS PID: ", p.pid
    print "RSS:", p.memory_int().rss/1024
    print "VMS:", p.memory_int().vms/1024
    
def memory_usage():
    """Memory usage of the current process in kilobytes."""
    status = None
    result = {'peak': 0, 'rss': 0}
    try:
        p = psutil.Process(os.getpid())
        print "OS PID: ", p.pid

        # This will only work on systems with a /proc file system
        # (like Linux).
        status = open('/proc/self/status')
        for line in status:
            parts = line.split()
            key = parts[0][2:-1].lower()
            if key in result:
                result[key] = int(parts[1])
    finally:
        if status is not None:
            status.close()

#    testdir = '/home/julianarhee/Repositories/2p-pipeline/pipeline/python/tests/proc_info'
#    pinfo_fn = 'proc_%s.json' % p.pid
#    with open(os.path.join(testdir, pinfo_fn), 'w') as f:
#        json.dump(result, f, indent=4)
#
    return result


if __name__ == "__main__":

    # PATH opts:
    # -------------------------------------------------------------
    parser = optparse.OptionParser()

    parser.add_option('-R', '--root', action='store', dest='rootdir', default='/nas/volume1/2photon/data', help='source dir (root project dir containing all expts) [default: /nas/volume1/2photon/data]')
    parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')
    parser.add_option('-S', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID') 
    parser.add_option('-n', '--nproc', action='store', dest='nprocesses', default=4, help='num processes to use [default: 4]') 

    parser.add_option('--slurm', action='store_true', dest='slurm', default=False, help="set if running as SLURM job on Odyssey")

    (options, args) = parser.parse_args() 

    # -------------------------------------------------------------
    # INPUT PARAMS:
    # -------------------------------------------------------------
    rootdir = options.rootdir #'/nas/volume1/2photon/projects'
    animalid = options.animalid
    session = options.session #'20171003_JW016' #'20170927_CE059'
    nprocesses = int(options.nprocesses)
    slurm = options.slurm
    if slurm is True:
        print "SLURM"
	    rootdir = '/n/coxfs01/julianarhee/testdata'
	    path_to_si_base = '/n/coxfs01/2p-pipeline/pkgs/ScanImageTiffReader-1.1-Linux'
	    path_to_si_reader = os.path.join(path_to_si_base, 'share/python')
	    sys.path.append(path_to_si_reader)

    session_dir = os.path.join(rootdir, animalid, session)
    pid_dir = os.path.join(session_dir, 'tmp_spids')
    print pid_dir
    if len(os.listdir(pid_dir)) == 0:
        print "No PIDs to process."
        process_batch = False
    else:
        process_batch = True
        pid_paths = [os.path.join(pid_dir, p) for p in os.listdir(pid_dir) if p.endswith('json') and 'pid_' in p]

    opts_list = []
    for pid_path in pid_paths:
        with open(pid_path, 'r') as f:
            pinfo = json.load(f)
        opts = ['-R', pinfo['rootdir'], '-i', pinfo['animalid'], '-S', pinfo['session'], '-A', pinfo['acquisition'], '-r', pinfo['run'], '-p', pinfo['pid']]
	if slurm is True:
	    opts.extend(['--slurm'])
        opts_list.append(opts)
	print "PID:", pid_path, opts
        #Process(target=process_pid, args=(opts,)).start()

    pool = mp.Pool(processes=nprocesses) #, memory_usage)
    results = [pool.apply_async(process_pid, args=(opts,)) for opts in opts_list]
    output = [p.get() for p in results]
    print "COMPLETED:", output
    pool.close()
    pool.join()

    finished_dir = os.path.join(pid_dir, 'completed')
    if not os.path.exists(finished_dir):
        os.makedirs(finished_dir)
    for pid_path in pid_paths:
        pid_fn = os.path.split(pid_path)[1]
        shutil.move(pid_path, os.path.join(finished_dir, pid_fn))
