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

#import scipy.io
#import pprint

pp = pprint.PrettyPrinter(indent=4)

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]

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
              

def check_init(processdict, I):

    version = pkg_resources.get_distribution('pipeline').version
    process_id = None

    # First check current params against existing analyses:
    if processdict is None:
        processdict = dict()
        existing_pids = []
        is_new_pid = True
        process_id = "process%03d" % int(len(existing_pids)+1)
        print "No existing PIDs found."
    else:
        existing_pids = sorted([str(k) for k in processingdict.keys()], key=natural_keys)
        print "Found existing PIDs:"
        for pidx, pid in enumerate(existing_pids):
            print pidx, pid

        matching_keys = [i for i in I.keys() if I[i] == processdict[existing_pid][i]]
        matching_analysis = sorted([epid for epid in existing_pids if len(matching_keys) == len(I.keys())], key=natural_keys)
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
        processdict[process_id] = I

        pp.pprint(processdict)
        
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
                     
    
    I['process_id'] = process_id
    I['version'] = version
    
    return I, is_new_pid, processdict
                
#%% Check which ACQMETA fields are anlaysis-ID-specific:

def update_records(I, processdict, processdict_table, acquisition_dir, run):

    processdict_filepath = os.path.join(acquisition_dir, 'pid_info_%s.json' % run)
    #processdict_tablepath = os.path.join(acquisition_dir, 'pid_info_%s.txt' % run)
    
    acquisition_meta_fn = os.path.join(acquisition_dir, 'reference_%s.json' % run)
    #acquisition_meta_mat = os.path.join(acquisition_dir, 'reference_%s.mat' % run)

    with open(acquisition_meta_fn, 'r') as f:
        acqmeta = json.load(f)
     
    with open(acqmeta['mcparams_path'], 'r') as f:
        mcparams = json.load(f)
 
    #mcparams = scipy.io.loadmat(acqmeta['mcparams_path'])
    mcparams = mcparams[I['mc_id']] 
    I['mcparams'] = mcparams
 
#    pid_fields = []
#    for field in acqmeta.keys():
#        if isinstance(acqmeta[field], dict):
#            pid_fields.append(str(field))
#    print(pid_fields)
#    
#
#    #%
#    if is_new_pid is True:
#        
#        process_id = I['process_id']
#        
#        for field in pid_fields:
#            if field in I.keys():
#                acqmeta[field][process_id] = I[field]
#            elif field=='bidi':
#                acqmeta[field][process_id] = int(mcparams['bidi_corrected'])
#            elif field=='trace_id':
#                acqmeta[field][process_id] = os.path.join(acqmeta['trace_dir'], I['roi_id'], I['mc_id'])
#            elif field=='data_dir':
#                acqmeta[field][process_id] = os.path.join(acquisition_dir, run, 'processed', 'DATA')
#            elif field=='simeta_path':
#                acqmeta[field][process_id] = os.path.join(acquisition_dir, run, 'processed', 'DATA', 'simeta.mat')
#            else:
#                print(field)
#       
    process_id = I['process_id'] 
    processdict[process_id] = I
        
    new_table_entry = pd.DataFrame.from_dict({process_id: I}, orient='index')
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


def main(I, acquisition_dir, run, infodict=infodict):


    processdict, processdict_table = load_processdict(acquisition_dir, run)
    
    I, is_new_pid, processdict = check_init(processdict, I)
    
    update_records(I, processdict, processdict_table, acquisition_dir, run)
    
    return I

#%%
if __name__ == '__main__':

    I = main(**infodict)

