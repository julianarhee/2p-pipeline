#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 17:19:12 2017

@author: julianarhee
"""
import os
import json
import copy
import pprint
import re
import pkg_resources
import pandas as pd
import optparse
import sys
from pipeline.python.set_pid_params import create_pid
from pipeline.python.preprocessing.process_raw import process_pid
from multiprocessing import Process

pp = pprint.PrettyPrinter(indent=4)

# GENERAL METHODS:
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]

#%%
# PATH opts:
# -------------------------------------------------------------
parser = optparse.OptionParser()

parser.add_option('-D', '--root', action='store', dest='rootdir', default='/nas/volume1/2photon/data', help='source dir (root project dir containing all expts) [default: /nas/volume1/2photon/data]')
parser.add_option('-H', '--home', action='store', dest='homedir', default='/nas/volume1/2photon/data', help='current data root dir (if creating params with path-root different than what will be used for actually doing the processing.')
parser.add_option('--notnative', action='store_true', dest='notnative', default=False, help="Set flag if not setting params on same system as processing.")

parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')
parser.add_option('-S', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID') 
parser.add_option('--slurm', action='store_true', dest='slurm', default=False, help="set if running as SLURM job on Odyssey")
#parser.add_option('--flyback', action='store_true', dest='correct_flyback', default=False, help="do flyback correction on acquisitions with _volume_ in name")
#parser.add_option('-F', '--nflyback', action='store', dest='nflyback_frames', default=0, help="Number of flyback frames to remove from top of volumes")
#parser.add_option('--motion', action='store_true', dest='correct_motion', default=False, help="do motion correction")
#parser.add_option('-M', '--mcmethod', action='store', dest='mcmethod', default='Acquisition2P', help="Method of motion-correction to use [default: Acquisition2P]")
#parser.add_option('-a', '--mcalgorithm', action='store', dest='mcalgorithm', default='@withinFile_withinFrame_lucasKanade', help="Algorithm to use for motion-correction [default: @withinFile_withinFrame_lucasKanade]")
#parser.add_option('--bidi', action='store_true', dest='correct_bidir', default=False, help="do bidirectional scan correction")
parser.add_option('--indie', action='store_true', dest='individual', default=False, help="Set flag if want to use tmp_pid created individually for each run")

(options, args) = parser.parse_args() 

# -------------------------------------------------------------
# INPUT PARAMS:
# -------------------------------------------------------------
rootdir = options.rootdir #'/nas/volume1/2photon/projects'
homedir = options.homedir
notnative = options.notnative
    
animalid = options.animalid
session = options.session #'20171003_JW016' #'20170927_CE059'
slurm = options.slurm
if slurm is True:
    rootdir = '/n/coxfs01/2p-data'
tiffsource = 'raw'
individual = options.individual


#correct_flyback = options.correct_flyback
#nflyback_frames = options.nflyback_frames
#correct_bidir = options.correct_bidir
#correct_motion = options.correct_motion
#mc_method = options.mcmethod
#mc_algorithm = options.mcalgorithm


# -------------------------------------------------------------
# Set paths:
# -------------------------------------------------------------
if notnative is True:
    session_dir = os.path.join(homedir, animalid, session)
else:
    session_dir = os.path.join(rootdir, animalid, session)
    
acquisitions = [a for a in os.listdir(session_dir) if os.path.isdir(os.path.join(session_dir, a))]
session_dict = dict((acq, []) for acq in acquisitions)
for acq in session_dict.keys():
    session_dict[acq] = dict((r, []) for r in os.listdir(os.path.join(session_dir, acq))\
                                    if os.path.isdir(os.path.join(session_dir, acq, r))\
                                    and 'anat' not in r)
pp.pprint(session_dict)

#base_opts = ['-R', rootdir, '-i', animalid, '-S', session, '-t', tiffsource, '--default']
#if correct_bidir is True:
#    base_opts.extend(['--bidi'])
#if correct_motion is True:
#    base_opts.extend(['--motion', '-M', mc_method, '-a', mc_algorithm])

pid_savedir = os.path.join(session_dir, 'tmp_spids')
if not os.path.exists(pid_savedir):
    os.makedirs(pid_savedir)
print "Saving PIDs to process in dir: ", pid_savedir

#print base_opts
for curr_acq in session_dict.keys():
    print curr_acq
    for curr_run in session_dict[curr_acq].keys():
        pinfo = dict()
        create_new = False
        if individual is True:
            # Look in current run tmp_pid dir for incomplete PID set(s):
            curr_tmp_pid_dir = os.path.join(session_dir, curr_acq, curr_run, 'processed', 'tmp_pids')
            
            # If there isn't actually and PID set in the tmp_pid dir, don't include this run in the batch:
            if not os.path.exists(curr_tmp_pid_dir):
                session_dict[curr_acq].pop(curr_run, session_dict[curr_acq][curr_run])
                continue
            
            # Otherwise, add found tmp_pid if there is only one, or 
            found_tmp_pid_fns = [f for f in os.listdir(curr_tmp_pid_dir) if f.endswith('json')]
            pid_list = []
            if len(found_tmp_pid_fns) == 1:
                with open(os.path.join(curr_tmp_pid_dir, found_tmp_pid_fns[0]), 'r') as fp:
                    pid = json.load(fp)
                pid_list.append(pid)
            elif len(found_tmp_pid_fns) > 1:
                print "Found more than 1 PID file in acq %s, run %s." % (curr_acq, curr_run)
                while True:
                    for pidx, pidfile in enumerate(found_tmp_pid_fns):
                        print pidx, pidfile
                    view_pidx = raw_input("Select IDX of PID file to view:")
                    if view_pidx == 'Q':
                        break
                    with open(os.path.join(curr_tmp_pid_dir, found_tmp_pid_fns[int(view_pidx)]), 'r') as ft:
                        pid = json.load(ft)
                    pp.pprint(pid)
                    
                    confirm_pidx = raw_input("Enter <P> to use this PID, or <ENTER> to view another, or <Q> to quit:")
                    if confirm_pidx == 'P':
                        if pid not in pid_list:
                            pid_list.append(pid)
                    elif confirm_pidx == 'Q' and len(pid_list) > 0:
                        break
#            else:
#                print "No tmp PID files found in acq %s, run %s. Skipping..."
#                create_new = True
#        
#        if create_new is True:
#            curr_opts = copy.copy(base_opts)
#            curr_opts.extend(['-A', curr_acq, '-r', curr_run])
#            if 'volume' in curr_acq and correct_flyback is True:
#                curr_opts.extend(['--flyback', '-F', nflyback_frames])
#            print curr_opts
#            pid = create_pid(curr_opts)
        
#        pp.pprint(pid)
#            
#        pinfo['rootdir'] = rootdir
#        pinfo['animalid'] = animalid
#        pinfo['session'] = session
#        pinfo['acquisition'] = curr_acq
#        pinfo['run'] = curr_run
#        pinfo['pid'] = pid['pid_hash']

#        pid_fn = 'pid_%s.json' % pinfo['pid']        
#        with open(os.path.join(pid_savedir, pid_fn), 'w') as f:
#            json.dump(pinfo, f, indent=4)
            
        for pid in pid_list:
            pinfo = dict()
            pinfo['rootdir'] = rootdir
            pinfo['animalid'] = animalid
            pinfo['session'] = session
            pinfo['acquisition'] = curr_acq
            pinfo['run'] = curr_run
            pinfo['pid'] = pid['pid_hash']
            
            # Save PID info to dict:
            pid_fn = 'pid_%s.json' % pinfo['pid']        
            with open(os.path.join(pid_savedir, pid_fn), 'w') as f:
                json.dump(pinfo, f, indent=4)
        
pid_files = [p for p in os.listdir(pid_savedir) if p.endswith('json') and 'pid' in p]
print "Created PIDs for session %s | acquisitions: runs --" % session
pp.pprint(pid_files)
print "Requesting %i PIDs to process." % len(pid_files)

#for curr_acq in session_dict.keys():
#    for curr_run in session_dict[curr_run].keys():
#        pinfo['rootdir'] = rootdir
#        pinfo['animalid'] = animalid
#        pinfo['session'] = session
#        pinfo['acquisition'] = curr_acq
#        pinfo['run'] = curr_run
#        pinfo['pid'] = session_dict[curr_acq][curr_run]
#
#session_pids_fn = 'tmp_session_pids.json'
#with open(os.path.join(session_dir, session_pids_fn), 'w') as f:
#    json.dump(session_dict, f, indent=4)
#
#print "Saved tmp session PIDS to:"
#print os.path.join(session_dir, session_pids_fn)
#
#for curr_acq in session_dict.keys():
#    for curr_run in session_dict[curr_acq].keys():
#        pid = session_dict[curr_acq][curr_run]
#        opts = ['-R', rootdir, '-i', animalid, '-S', session, '-A', curr_acq, '-r', curr_run, '-p', pid]
#        Process(target=process_pid, args=(opts,)).start()
#
