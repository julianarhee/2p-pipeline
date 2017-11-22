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
import pkg_resources
import pandas as pd
import optparse

#import scipy.io
#import pprint

pp = pprint.PrettyPrinter(indent=4)

def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]


def initialize_pid(source, process_dict, acquisition_dir, run):

    # Load existing PIDs if they exist:
    process_id, is_new_pid = get_process_id(process_dict)

    pid = dict()
    version = pkg_resources.get_distribution('pipeline').version

    pid['version'] = version 
    pid['process_id'] = process_id
    pid['SRC'] = os.path.join(acquisition_dir, run, source) #source_dir
    pid['DST'] = os.path.join(acquisition_dir, run, process_id)
    pid['mcparams'] = None
 
    return pid
     

def load_processdict(acquisition_dir, run):
    processdict = None
    processdict_table = None 
    processdict_filepath = os.path.join(acquisition_dir, 'pid_info_%s.json' % run)
    processdict_tablepath = os.path.join(acquisition_dir, 'pid_info_%s.txt' % run)
    
    # Load analysis "processdict" file:
    if os.path.exists(processdict_filepath):
        with open(processdict_filepath, 'r') as f:
            processdict = json.load(f)
 
    # Also load TABLE version: 
    if os.path.exists(processdict_tablepath):        
        processdict_table = pd.read_csv(processdict_tablepath, sep='\s+', header=0, index_col=0)
    
    return processdict, processdict_table
              

def get_process_id(processdict):

    process_id = None

    # First check current params against existing analyses:
    if processdict is None:
        processdict = dict()
        existing_pids = []
        is_new_pid = True
        #process_id = "process%03d" % int(len(existing_pids)+1)
        print "No existing PIDs found."
    else:
        existing_pids = sorted([str(k) for k in processingdict.keys()], key=natural_keys)
        print "Found existing PIDs:"
        for pidx, pid in enumerate(existing_pids):
            print pidx, pid

        # Show existing PIDs:
        while True:
            check_pidx = raw_input('Enter IDX of pid to view or hit <ENTER> to create new: ')
            if len(check_pidx) == 0:
                is_new_pid = True
                break
            else:
                print "Viewing PID: %s" existing_pids[int(check_pidx)]
                pp.pprint(processdict[existing_pids[int(check_pidx)]])
                reuse_idx = raw_input('Enter <R> to re-use current pid.')
                if reuse_idx == 'R':
                    is_new_pid = False
                    break
 
        if is_new_pid is True:
            # Create new PID by incrementing num of process dirs found:
            process_id = 'process%03d' % int(len(existing_pids)+1)
            print "Creating NEW pid: %s" % process_id
        else:
            # Re-using an existing PID:
            process_id = existing_pids[int(check_pidx)]
            print "Reusing existing pid: %s" % process_id
 
    return process_id, is_new_pid #, processdict


def set_mcparams(acquisition_dir, run, process_id,
                    motion_corrected=True,
                    bidi_corrected=True,
                    flyback_corrected=False,
                    ref_channel=1,
                    ref_file=1,
                    method=None,
                    algorithm=None,
                    source=None,
                    destination=None
                    ):

    mcparams = dict()
    mcparams['corrected'] = motion_corrected
    mcparams['bidi_corrected'] = bidi_corrected
    mcparams['flyback_corrected'] = flyback_corrected 
    mcparams['method'] = method
    mcparams['algorithm'] = algorithm
    mcparams['ref_channel'] = ref_channel
    mcparams['ref_file'] = ref_file
    mcparams['source_dir'] = source
    mcparams['dest_dir'] = destination

    if source is None:
        mcparams['source_dir'] = os.path.join(acquisition_dir, run, 'raw')
    if destination is None:
        mcparams['dest_dir'] = os.path.join(acquisition_dir, run, 'processed', process_id)

    return mcparams

 
def check_process_id(tmp_pid, processdict):

    existing_pids = processdict.keys()
    
    # Check if all PID fields match existing ones...
    keys_to_check = 

    matching_analysis = sorted([epid for epid in existing_pids if len([i for i in tmp_pid.keys() if tmp_pid[i] == processdict[epid][i]) == len(tmp_pid.keys())], key=natural_keys)
    if len(matching_analysis) > 0:    
        for m, mi in enumerate(matching_analysis):
            print m, mi
            
        while True:
            user_choice = raw_input("Found matching analysis ID. Press N to create new, or IDX of analysis_id to reuse: ")
            
            if user_choice == 'N':
                is_new_pid = True
                process_id = "analysis%02d" % int(len(existing_pids)+1)
            elif user_choice.isdigit():
                is_new_pid = False
                process_id = matching_analysis[int(user_choice)]

            print "Selected processing params:"
            pp.pprint(processdict[process_id]) 

            confirm = raw_input('Using analysis ID: %s. Press Y/n to confirm: ' % process_id)
            if confirm == 'Y':
                break
            
    else:
        print "New process params specified. Creating new pid."
        is_new_pid = True
                
    if is_new_pid is True:
        # Create new PID by incrementing num of process dirs found:
        process_id = 'process%03d' % int(len(existing_pids)+1)
        print "Creating NEW pid: %s" % process_id
       
    else:
        # Check re-used analysis fields and make sure they all match:
        for field in processdict[process_id].keys():
            if field == 'process_id':
                continue
            if not processdict[process_id][field] == I[field]:
                # Stay in loop until user supplies valid entries:
                while True:
                    print "Mismatch found in field: %s" % field
                    print "[0] OLD: %s || [1] NEW: %s" % (processdict[process_id][field], I[field])
                    overwrite_choice = input('Press 0 to keep old, or 1 to overwrite with new value: ')
                    if int(overwrite_choice) == 0:
                        print "Selected to keep OLD key-value pair:", field, processdict[process_id][field]
                        confirm_choice = input('Press Y to confirm: ')
                    elif int(overwrite_choice) == 1:
                        print "Selected to overwrite with NEW key-value pair:", field, I[field]
                        confirm_choice = input('Press Y to confirm: ')
                    
                    if confirm_choice == 'Y' and int(overwrite_choice) == 0: 
                        I[field] = processdict[process_id][field]
                        break
                    elif confirm_choice == 'Y' and int(overwrite_choice) == 1: 
                        processdict[process_id][field] = I[field]
                        break
                     
       
                
#%% Check which ACQMETA fields are anlaysis-ID-specific:

def update_records(pid, processdict, processdict_table, acquisition_dir, run):

    processdict_filepath = os.path.join(acquisition_dir, 'pid_info_%s.json' % run)
    processdict_tablepath = os.path.join(acquisition_dir, 'pid_info_%s.txt' % run)
    
    acquisition_meta_fn = os.path.join(acquisition_dir, 'reference_%s.json' % run)
    #acquisition_meta_mat = os.path.join(acquisition_dir, 'reference_%s.mat' % run)

    with open(acquisition_meta_fn, 'r') as f:
        acqmeta = json.load(f)
     
    with open(acqmeta['mcparams_path'], 'r') as f:
        mcparams = json.load(f)

    mc_id = pid['mc_id']
    mcparams['mc_id'] = pid['mcparams']
    with open(acqmeta['mcparams_path'], 'w') as f:
        json.dump(f)
 
    #mcparams = scipy.io.loadmat(acqmeta['mcparams_path'])
#    mcparams = mcparams[I['mc_id']] 
#    I['mcparams'] = mcparams
# 
    process_id = pid['process_id'] 
    processdict[process_id] = pid
        
    new_table_entry = pd.DataFrame.from_dict({process_id: pid}, orient='index')
    if processdict_table is None:
        processdict_table = new_table_entry
    else:
        processdict_table.append(new_table_entry)
    
    #% Update Process Info DICT:
    with open(processdict_filepath, 'w') as f:
        json.dump(processdict, f, sort_keys=True, indent=4)
       
    #% Update Process Info TABLE: 
    processdict_table.to_csv(processdict_tablepath, sep='\t', header=True, index=True, index_label='Row') #header=False)

#    with open(acquisition_meta_fn, 'w') as f:
#        json.dump(acqmeta, f, sort_keys=True, indent=4)
#    
    #scipy.io.savemat(acquisition_meta_mat, mdict=acqmeta)
    
    print "Process Info UPDATED."



parser = optparse.OptionParser()

# PATH opts:
parser.add_option('-P', '--sipath', action='store', dest='path_to_si_reader', default='~/Downloads/ScanImageTiffReader-1.1-Linux/share/python', help='path to dir containing ScanImageTiffReader.py')
parser.add_option('-R', '--root', action='store', dest='root', default='/nas/volume1/2photon/data', help='data root dir (root project dir containing all animalids) [default: /nas/volume1/2photon/data, /n/coxfs01/2pdata if --slurm]')
parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')

# Set specific session/run for current animal:
parser.add_option('-S', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID')
parser.add_option('-A', '--acq', action='store', dest='acquisition', default='FOV1', help="acquisition folder (ex: 'FOV1_zoom3x') [default: FOV1]")
parser.add_option('-r', '--run', action='store', dest='run', default='', help="name of run dir containing tiffs to be processed (ex: gratings_phasemod_run1)")

# SET source/dest dirs for current processing run:
parser.add_option('-s', '--source', action='store', dest='source_dir', default='', help="name of folder containing tiffs to be processed (ex: processed001). should be child of <run>/processed/")

# MC params:
parser.add_option('--motion', action='store_true', dest='mc', default=False, help='Set flag if should run motion-correction.')
parser.add_option('--bidi', action='store_true', dest='bidi', default=False, help='Set flag if correct bidirectional scanning phase offset.')
parser.add_option('--flyback', action='store_true', dest='flyback', default=False, help='Set flag if extra flyback frames were corrected in process_raw.py')
parser.add_option('-c', '--channel', action='store', dest='ref_channel', default=1, help='Index of CHANNEL to use for reference if doing motion correction [default: 1]')
parser.add_option('-f', '--file', action='store', dest='ref_file', default=1, help='Index of FILE to use for reference if doing motion correction [default: 1]')
parser.add_option('-M', '--method', action='store', dest='mc_method', default=None, help='Method for motion-correction. OPTS: Acquisition2P, NoRMCorre [default: Acquisition2P]')
parser.add_option('-a', '--algo', action='store', dest='algorithm', default=None, help='Algorithm to use for motion-correction, e.g., @withinFile_withinFrame_lucasKanade if method=Acquisition2P, or nonrigid if method=NoRMCorre')

(options, args) = parser.parse_args() 

mc_methods = ['Acquisition2P', 'NoRMCorre']
mc_algos = dict((mc, []) for mc in mc_methods)
mc_algos = {'Acquisition2P': [@withinFile_withinFrame_lucasKanade, @lucasKanade_plus_nonrigid],
            'NoRMCorre': ['rigid', 'nonrigid']}

root = options.root
animalid = options.animalid
session = options.session
acquisition = options.acquisition
run = options.run

source = options.source

correct_motion = options.mc
ref_file = options.ref_file
ref_channel = options.ref_channel

# Check user-provided MC params:
if correct_motion is True:
    mc_method = options.mc_method
    if mc_method is None:
        while True:
            print "No MC method specified. Use default [Acquisition2P]?"
            mc_choice = raw_input('Enter <Y> to use default, or <o> to see options:')
            if mc_choice == 'Y':
                print "Using default."
                mc_method = 'Acquisition2P'
                break
            elif mc_choice == 'o':
                for mcid, mcname in enumerate(mc_methods):
                    print mcid, mcname
                mc_select = input('Enter IDX of motion-correction method to use:')
                mc_method = mc_methods[mc_select]
                break
    mc_algorithm = options.algorithm
    if mc_algorithm is None or (mc_algorithm not in mc_algos[mc_method]):
        print "No MC algorithm specified... Here are the options:"
        for algonum, algoname in enumerate(mc_algos[mc_method]):
            print algonum, algoname
        algo_choice = input('Enter IDX of mc algorithm to use:')
        mc_algorithm = mc_algos[mc_method][algo_choice] 
else:
    ref_channel = None
    ref_file = None
    mc_method = None
    mc_algorithm = None

print "Using MC method: %s, algorithm: %s." % (str(mc_method), str(mc_algorithm))
if correct_motion is True:
    print "Ref file: %i, Ref channel: %i" % (ref_channel, ref_file)

acquisition_dir = os.path.join(source, animalid, session, acquisition)

# Load existing PID info:
processdict, processdict_table = load_processdict(acquisistion_dir, run)

# Create NEW pid, if relevant:
tmp_pid = initialize_pid(source, processdict, acquisition_dir, run)

# Set MCPARAMS:
mcparams = set_mcparams(acquisition_dir, run, process_id,
                        motion_corrected=correct_motion,
                        bidi_corrected=options.bidi,
                        flyback_corrected=options.flyback,
                        ref_channel=ref_channel,
                        ref_file=ref_file,
                        method=mc_method,
                        algorithm=mc_algorithm,
                        source=source,
                        destination=tmp_pid['dest'])

#TODO:  Create check_mcparams s.t. (a) re-use MCPARAMS ID if exact same params exist, 
# (b) generate or return mc_id to store in PID dict after check:
mc_id, mcparams = check_mcparams(mcparams, acquisition_dir, run)

tmp_pid['mcparams'] = mcparams

#TODO:  Checking PID might be unncess. if always generating NEW PID, and already check MCPARAMS ID in prev step:
pid, processdict, processdict_table = check_process_id(tmp_pid, processdict)

update_records(pid, processdict, processdict_table, acquisition_dir, run)



#%%
#if __name__ == '__main__':
#    I = main(**infodict)

