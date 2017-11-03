#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 17:19:12 2017

@author: julianarhee
"""

import os
import json
import pandas as pd
import scipy.io

def load_rolodex(acquisition_dir):
    
    import os
    
    rolodex = None
    rolodex_table = None
    
    rolodex_filepath = os.path.join(acquisition_dir, 'analysis_record.json')
    rolodex_tablepath = os.path.join(acquisition_dir, 'analysis_record.txt')
    
    # Load analysis "rolodex" file:
    if os.path.exists(rolodex_filepath):
        with open(rolodex_filepath, 'r') as f:
            rolodex = json.load(f)
    
        # Also load TABLE version:
        rolodex_table  = pd.read_csv(rolodex_tablepath, sep="\s+", header=0, index_col=0)
    
    return rolodex, rolodex_table
              

def check_init(rolodex, I):
    # First check current params against existing analyses:
    existing_analysis_ids = [str(k) for k in rolodex.keys()]
    print(existing_analysis_ids)
    

    matching_analysis = [existing_id for existing_id in existing_analysis_ids if len([i for i in I.keys() if I[i]==rolodex[existing_id][i]])==len(I.keys())]
    if len(matching_analysis)>0:    
        for m,mi in enumerate(matching_analysis):
            print(m, mi)
            
        while True:
            user_choice = raw_input("Found matching analysis ID. Press N to create new, or IDX of analysis_id to reuse: ")
            
            if user_choice=='N':
                new_analysis_id = True
                analysis_id = "analysis%02d" % int(len(existing_analysis_ids)+1)
            elif user_choice.isdigit():
                new_analysis_id = False
                analysis_id = matching_analysis[int(user_choice)]
            
            confirm = raw_input('Using analysis ID: %s. Press Y/n to confirm: ' % analysis_id)
            if confirm=='Y':
                break
                
#        else:
#            print("RE-USING old analysis ID match: %s", matching_analysis[0])
#            new_analysis_id = False
#            analysis_id = matching_analysis[0]

    if new_analysis_id is True:
        print("Creating NEW analysis ID...")
        print("Existing analysis IDs:")
        for idx,existing_id in enumerate(existing_analysis_ids):
            print(idx, existing_id)
        
        
        print("Creating NEW analysis ID: %s" % analysis_id)
        confirm_id = raw_input("Press Y/n to confirm ID: ")
        
        if not confirm_id=='Y':
            while True:
                analysis_id = raw_input("Enter new analysis_id (ex: analysis01): ")
                if not analysis_id in existing_analysis_ids:
                    print("NEW ID: %s" % analysis_id)
                    break
                else:
                    print("ID EXISTS! Try again...")
    
    I['analysis_id'] = analysis_id
    
    return I, new_analysis_id
                
#%% Check which ACQMETA fields are anlaysis-ID-specific:

def update_records(I, rolodex, rolodex_table, new_analysis_id, acquisition_dir, functional='functional'):

    rolodex_filepath = os.path.join(acquisition_dir, 'analysis_record.json')
    rolodex_tablepath = os.path.join(acquisition_dir, 'analysis_record.txt')
    
    acquisition_meta_fn = os.path.join(acquisition_dir, 'reference_%s.json' % functional)
    acquisition_meta_mat = os.path.join(acquisition_dir, 'reference_%s.mat' % functional)

    with open(acquisition_meta_fn, 'r') as f:
        acqmeta = json.load(f)
    
    
    mcparams = scipy.io.loadmat(acqmeta['mcparams_path'])
    mcparams = mcparams[I['mc_id']] 
    
    analysis_id_fields = []
    for field in acqmeta.keys():
        if isinstance(acqmeta[field], dict):
            analysis_id_fields.append(str(field))
    print(analysis_id_fields)
    

    #%
    if new_analysis_id is True:
        
        analysis_id = I['analysis_id']
        
        for field in analysis_id_fields:
            if field in I.keys():
                acqmeta[field][analysis_id] = I[field]
            elif field=='bidi':
                acqmeta[field][analysis_id] = int(mcparams['bidi_corrected'])
            elif field=='trace_id':
                acqmeta[field][analysis_id] = os.path.join(acqmeta['trace_dir'], I['roi_id'], I['mc_id'])
            elif field=='data_dir':
                acqmeta[field][analysis_id] = os.path.join(acquisition_dir, functional, 'DATA')
            elif field=='simeta_path':
                acqmeta[field][analysis_id] = os.path.join(acquisition_dir, functional, 'DATA', 'simeta.mat')
            else:
                print(field)
        
        rolodex[analysis_id] = I
        
        new_table_entry = pd.DataFrame.from_dict({analysis_id: I}, orient='index')
        rolodex_table.append(new_table_entry)
        
    #% Update ACQMETA, ROLODEX (.json, .txt):
    with open(rolodex_filepath, 'w') as f:
        json.dump(rolodex, f, sort_keys=True, indent=4)
        
    rolodex_table.to_csv(rolodex_tablepath, sep='\t', header=True, index=True, index_label='Row') #header=False)

    with open(acquisition_meta_fn, 'w') as f:
        json.dump(acqmeta, f, sort_keys=True, indent=4)
    
    #scipy.io.savemat(acquisition_meta_mat, mdict=acqmeta)
    
    print("ALL RECORDS UPDATED.")


def main(I, acquisition_dir, functional='functional'):


    rolodex, rolodex_table = load_rolodex(acquisition_dir)
    
    I, new_analysis_id = check_init(rolodex, I)
    
    update_records(I, rolodex, rolodex_table, new_analysis_id, acquisition_dir, functional=functional)
    

#%%
if __name__=='__main__':
    #import sys
    import os
    import json
    import pandas as pd
    import scipy.io

    main(**infodict)

