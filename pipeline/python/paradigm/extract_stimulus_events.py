#!/usr/bin/env python2
'''
This script combines behavior/stimulation info with acquisition/frame info.

Input:
    - Raw behavior files saved in <run_dir>/raw/paradigm_files/ -- both serial data (*.txt) and protocol info (*.mwk)
    - Assumes .mwk files for behavior events and csv-saved .txt file that samples the acquisition rig at 1kHz

Steps:

1.  Creates a SINGLE parsed .json for EACH behavior file (.mwk) that contains relevant info for each trial in that file.
    - MW stimulus-presentation info is extracted with process_mw_files.py

    Output:

    a.  parsed_trials_<BEHAVIOR_FILE_NAME>.json files (1 for multiple .tifs, or 1 for each .tif if one_to_one = True)
        -- these output files are saved to: <RUN_DIR>/paradigm/files/
        -- (OPTIONAL: also can create a stimorder.txt file for EACH .tif to be aligned (option if not using .mwk), for align_acquisition_events.py)

2.  Aligns image-acquisition events (serial data stored in .txt files) with behavior events using the parsed behavior info from Step 1.
    All trials are combined across all .tif files and behavior (i.e., 'aux') files and creates dictionary for each trial in the whole run (collapses across blocks).

    Output:

    a.  trials_<TRILAINFO_HASH>.json (SINGLE file)
        -- this file is saved to:  <RUN_DIR>/paradigm/
        -- each dict in this file is of format:

            'trial00001': {
                    'trial_hash'         :  hash created for entire trial dictionary in input parsed-trials file (from Step 1)
                    'block_idx'          :  tif file index in run (i..e, block number, 0-indexed)
                    'ntiffs_per_auxfile' :  total number of tiffs associated with this AUX file
                    'behavior_data_path' :  path to input files containing pasred trial info for each behavior file (from Step 1)
                    'serial_data_path'   :  path to input files containing serial data for each frame acquired frame
                    'start_time_ms'      :  trial start time (in msec) relative to start of run (i.e., when SI frame-trigger received)
                    'end_time_ms'        :  trial end time (ms) relative to start of run
                    'stim_dur_ms'        :  duration of stim on
                    'iti_dur_ms'         :  duration of ITI period after stim offset
                    'stimuli'            :  stimulus-info dict from MW, of format:
                            {
                            'filepath'   :  path to stimulus on stimulation computer
                            'filehash'   :  file hash of shown stimulus
                            'position'   :  x,y position (tuple of floats),
                            'rotation'   :  rotation specified by protocol (float),
                            'scale'      :  x,y size of stimulus (tuple of floats),
                            'stimulus'   :  name or index of shown stimulus,
                            'type'       :  type of stimulus (e.g., image, drifting_grating)
                            }
                    'trial_in_run'       :  index (1-indexed) of current trial across whole run (across all behavior files, if multiple exist),
                    'frame_stim_on'      :  index (0-indexed) of the closest-matching frame to stimulus onset (index in .tif)
                    'frame_stim_off'     :  index (0-indexed) of frame at which stimulus goes off
                    }

Notes:

This output is used by align_acquisition_events.py to use the frame_stim_on and frame_stim_off for each trial across all files,
combined with a user-specified baseline period (default, 1 sec) to get trial-epoch aligned frame indices.
'''

import os
import sys
import json
import re
import hashlib
import optparse
import shutil
import copy
import traceback
import glob
import numpy as np
import pandas as pd
import cPickle as pkl
from collections import Counter
from pipeline.python.paradigm import process_mw_files as mw
from pipeline.python.paradigm import utils as util

from pipeline.python.utils import hash_file

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def dump_last_state(mwtrial_path, trialevents, trial, starting_frame_set, first_found_frame,
                        bitcodes, modes_by_frame, frame_bitcodes, serialfn_path):
    curr_state = {'last_trial': trial,
                  'starting_frame_set': starting_frame_set,
                  'first_found_frame': first_found_frame,
                  'bitcodes': bitcodes,
                  'modes_by_frame': modes_by_frame,
                  'frame_bitcodes': frame_bitcodes,
                  'mwtrial_path': mwtrial_path,
                  'serialfn_path': serialfn_path}
    
    with open(os.path.join(os.path.split(mwtrial_path)[0], 'tmp_trial_events.pkl'), 'w') as f:
        pkl.dump(trialevents, f, protocol=pkl.HIGHEST_PROTOCOL)
    with open(os.path.join(os.path.split(mwtrial_path)[0], 'tmp_curr_frames.pkl'), 'wb') as f:
        pkl.dump(curr_state, f, protocol=pkl.HIGHEST_PROTOCOL)   
        
#%%



def extract_frames_to_trials(serialfn_path, mwtrial_path, runinfo, blank_start=True, verbose=False, remove_skipped_trials=False):
    
#%%
    framerate = runinfo['frame_rate']
    trialevents = None

    ### LOAD MW DATA.
    with open(mwtrial_path, 'r') as f:
        mwtrials = json.load(f)
    print "N parsed trials (MW):", len(mwtrials)

    ### LOAD SERIAL DATA.
    serialdata = pd.read_csv(serialfn_path, sep='\t')

    #%
    framerate = runinfo['frame_rate']
    trialevents = None

    if verbose is True:
        print serialdata.columns

    ser = pd.DataFrame(serialdata)
    
    # Get "visible" codes (when backlight on):
    if 'backlight_sensor' in ser.columns:
        sdata = ser[ser['backlight_sensor']==1]

    frame_ons = [i+1 for i in np.where(np.diff(ser['frame_trigger'])==1)[0]] # Get all indices where frame trigger when from 0 to 1
    sdata_frame_ixs = [0]
    sdata_frame_ixs.extend([ser.iloc[fi].name for fi in frame_ons]) # Get corrresponding idx in serial data file -- each index is a 'frame' (should be ntifs * nvolumes or less)
    sdata_frame_ixs = np.array(sorted(sdata_frame_ixs))
    
    # Check expected n frames vs. found:
    nexpected_frames = runinfo['nvolumes'] * runinfo['ntiffs']
    print "Expected %s frames, found %i (missed %i frames)." % (nexpected_frames, len(sdata_frame_ixs), nexpected_frames - len(sdata_frame_ixs))
    missing_frame_triggers = nexpected_frames != len(sdata_frame_ixs)
    
    
    # Check that no frame triggers were skipped/missed:
    diffs = np.diff(frame_ons)
    nreads_per_frame = max(set(diffs), key=list(diffs).count)
    print "Nreads per frame:", nreads_per_frame
    long_breaks = np.where(diffs>nreads_per_frame*2)[0]
    for lix, lval in enumerate(long_breaks):
        subintervals = list(set(frame_ons[lval+1:long_breaks[lix]]))
        if any(subintervals > nreads_per_frame+1) or any(subintervals < nreads_per_frame-1):
            print "WARNING -- extra missed frame-triggers in tif %i." % lix+1

    tif_start_frames = [int(b+1) for b in long_breaks]
    tif_start_frames.append(0)
    tif_start_frames = sorted(tif_start_frames)
    
    tif_ixs = sorted(np.unique([tdict['block_idx'] for tr,tdict in mwtrials.items()]))
    

    #%%
    durs = []
    adjusted_frame_count = []
    
    trialevents = dict()
    
    #frames_and_codes = [(ix, iv) for ix, iv, in zip(ser['pixel_clock'].index.tolist(), ser['pixel_clock'].values)]
    new_start_ix = 0
    prev_trial = 'trial00001'
    for trial in sorted(mwtrials.keys(), key=natural_keys):

        curr_tif_ix = mwtrials[trial]['block_idx']
        
        # Get frame triggers belonging to current tif:
        if curr_tif_ix == tif_ixs[-1]:
            curr_fr_triggers = sdata_frame_ixs[tif_start_frames[curr_tif_ix]:]
        else:
            curr_fr_triggers = sdata_frame_ixs[tif_start_frames[curr_tif_ix]:tif_start_frames[curr_tif_ix+1]]
        
        curr_reads = sdata.loc[(sdata.index >= curr_fr_triggers[0]) & (sdata.index <= curr_fr_triggers[-1])]
        curr_frames_and_codes = [(ix, iv) for ix, iv in zip(curr_reads['pixel_clock'].index.tolist(), curr_reads['pixel_clock'].values)]
        
        
        print "Parsing %s" % trial
        #if trial == 'trial00165':
        #    break
    
        # Create hash of current MWTRIAL dict:
        mwtrial_hash = hashlib.sha1(json.dumps(mwtrials[trial], sort_keys=True)).hexdigest()
    
        #print trial
        trialevents[mwtrial_hash] = dict()
        trialevents[mwtrial_hash]['trial'] = trial
        trialevents[mwtrial_hash]['stim_dur_ms'] = mwtrials[trial]['stim_off_times'] - mwtrials[trial]['stim_on_times']
    
        bitcodes = mwtrials[trial]['all_bitcodes']
        
        if mwtrials[trial]['missing_si_trigger']:
            missed_last_trigger = True
        else:
            missed_last_trigger = False
            
        #361154
        prev_val = -1
        min_nreps = 5
        found_bitcodes = []
        
        # Reset start index if starting new tif for thsi trial:
        if mwtrials[prev_trial]['block_idx'] != curr_tif_ix:
            new_start_ix = 0
        curr_codes = iter(curr_frames_and_codes[new_start_ix:]) # iter(frames_and_codes[new_start_ix:])
        
        cval = next(curr_codes)  
        
        for bi, bitcode in enumerate(bitcodes):
            nskips = 0; finding_frames = True;
            #print bi, bitcode
#            if bi == 31:
#                break
            # If first bitcode of trial N matches last bitcode of trial N-1, cycle through until the next one is found:
            if bi==0 and bitcode == mwtrials[prev_trial]['all_bitcodes'][-1]:
                while cval[1] == mwtrials[prev_trial]['all_bitcodes'][-1]: 
                    cval = next(curr_codes)
                    
            # Only accept bitcode assignment to SI frame if we have multiple repeats (even w dynamic stim, should have at least 4 reps)
            nreps_curr = 0; no_first_match = None;
            while ((no_first_match is None) or (nreps_curr < min_nreps)) and finding_frames:
                #print cval, nskips
                cval = next(curr_codes)  
                nskips += 1
                
                if no_first_match is None and cval[1] == bitcode:
                    no_first_match = True
                    #print "... found a first match"
                
                if cval[1] == bitcode and prev_val == bitcode: # First time, already have had 1 occurrence
                    nreps_curr += 1
                prev_val = cval[1]

                if (len(bitcodes) > 2) and bi > 0 and (nreps_curr < min_nreps):
                    if (nskips > nreads_per_frame):
                        finding_frames = False
                else:
                    finding_frames = (nreps_curr < min_nreps)
                   
            #print cval, nskips
            
            if (len(bitcodes) > 2) and bi > 0 and nskips > nreads_per_frame:
                #curr_valid_bitframes = sorted([f for f in found_bitcodes if f != 'missed'], key=lambda x: x[0])
                # Skip through curr bitcodes until next new value found:
                stuck_val = cval
                while cval[1] == stuck_val[1]:
                    cval = next(curr_codes)
                found_bitcodes.append('missed')
            else:
                # Subtract 2 indices from nreps_curr since 1st rep is counted after 1st occurrence, and last rep is counted after last found:
                last_found_idx = curr_frames_and_codes.index(cval)
                if bi == len(bitcodes)-1:
                    last_found_match = curr_frames_and_codes[last_found_idx]
                    found_bitcodes.append(last_found_match)
                else:
                    assert curr_frames_and_codes[last_found_idx - min_nreps + 2][1] == bitcode
                    first_found_match = curr_frames_and_codes[last_found_idx - min_nreps + 2]
                    found_bitcodes.append(first_found_match)
        
        # Update starting point of next trial's search iteration:
        new_start_ix = curr_frames_and_codes.index(cval)
        
        # Identify frame number of first and last found stimulus updates in current trial: 
        triggered_frame_on = np.argmin(np.abs(found_bitcodes[0][0] - sdata_frame_ixs))
        triggered_frame_off = np.argmin(np.abs(found_bitcodes[-1][0] - sdata_frame_ixs))
        
        # Calculate stimulus duration and check that it matches what's expected:
        stim_dur = (triggered_frame_off - triggered_frame_on) / framerate
        print "%s: %.1f" % (trial, round(stim_dur, 2))
        
        # If there are missed frame triggers, calculate expected frame stim-off:
        if missing_frame_triggers and np.abs(round(stim_dur, 1) - (mwtrials[trial]['stim_duration']/1E3)) > 0.1:
            triggered_frame_off = triggered_frame_on + int(round(((mwtrials[trial]['stim_duration']/1E3) * framerate)))
            stim_dur = (triggered_frame_off - triggered_frame_on) / framerate
            adjusted_frame_count.append(trial)
            
        
        if round(stim_dur) != round(mwtrials[trial]['stim_duration']/1E3):
            break
        prev_trial = trial
    
        trialevents[mwtrial_hash]['stim_on_idx'] = triggered_frame_on  
        trialevents[mwtrial_hash]['stim_off_idx'] = triggered_frame_off 
        trialevents[mwtrial_hash]['mw_trial'] = mwtrials[trial]
        trialevents[mwtrial_hash]['block_frame_offset'] = 0
        
        durs.append(stim_dur)

#%%
    #%
    # Do a final check of all stim durs:
    rounded_durs = list(set([round(d, 1) for d in durs]))
    unique_mw_stim_durs = list(set([round(mwtrials[t]['stim_duration']/1E3, 1) for t in mwtrials.keys()]))
    if len(rounded_durs) != len(unique_mw_stim_durs):
        print " *** WARNING -- funky stim durs found:", rounded_durs
        print " --- found MW stim durs:", unique_mw_stim_durs

    print " --- Adjusted %i trials after start bit-code found." % len(adjusted_frame_count)
        
#    if skipped != {}:
#        # Rename original parsed:
#        mw_fname = os.path.splitext(os.path.split(mwtrial_path)[-1])[0]
#        orig_fpath = os.path.join(os.path.split(mwtrial_path)[0], '_orig_%s.json' % mw_fname)
#        with open(orig_fpath, 'w') as f:
#            json.dump(mwtrials, f, indent=True, sort_keys=True)
#    
#    # Remove skipped, then resave:
#    for hkey,nkey in skipped.items():
#        print "Removing skipped trial: %s (%s)" % (nkey, hkey)
#        if hkey in trialevents.keys():
#            del trialevents[hkey]
#        if nkey in mwtrials.keys():
#            del mwtrials[nkey]

    with open(mwtrial_path, 'w') as f:
        json.dump(mwtrials, f, indent=True, sort_keys=True)
    print "Updated MWtrials and trial events."
    print "TOTAL N trials:", len(mwtrials.keys())
#%
    return trialevents


    
#%%

#
#def extract_frames_to_trials(serialfn_path, mwtrial_path, runinfo, blank_start=True, verbose=False, remove_skipped_trials=False):
#    '''
#    For every bitcode of every trial, find corresponding SI frame.
#    
#    TODO:  this is slow, make it faster...
#    
#    '''
#    #%%
#    framerate = runinfo['frame_rate']
#    trialevents = None
#
#    ### LOAD MW DATA.
#    with open(mwtrial_path, 'r') as f:
#        mwtrials = json.load(f)
#    print "N parsed trials (MW):", len(mwtrials)
#
#    ### LOAD SERIAL DATA.
#    serialdata = pd.read_csv(serialfn_path, sep='\t')
#    if verbose is True:
#        print serialdata.columns
#    
#    # ---------------------------------------------------------------------------------------
#    #abstime = serialdata[' abosolute_arduino_time']
#    ### Extract events from serialdata:
#    frame_triggers = serialdata['frame_trigger']
#    all_bitcodes = serialdata['pixel_clock']
#    
#    ### Find frame ON triggers (from NIDAQ-SI):
#    frame_on_idxs = [idx+1 for idx,diff in enumerate(np.diff(frame_triggers)) if diff==1]
#    frame_on_idxs.append(0)
#    frame_on_idxs = sorted(frame_on_idxs)
#    print "Found %i serial triggers" % len(frame_on_idxs)
#    
#    # Check that no frame triggers were skipped/missed:
#    diffs = np.diff(frame_on_idxs)
#    nreads_per_frame = max(set(diffs), key=list(diffs).count)
#    print "Nreads per frame:", nreads_per_frame
#    long_breaks = np.where(diffs>nreads_per_frame*2)[0]
#    for lix, lval in enumerate(long_breaks):
#        subintervals = list(set(frame_on_idxs[lval+1:long_breaks[lix]]))
#        if any(subintervals > nreads_per_frame+1) or any(subintervals < nreads_per_frame-1):
#            print "WARNING -- extra missed frame-triggers in tif %i." % lix+1
#
#
##    # Re-sort frame_idxs:
#    frame_on_idxs = sorted(list(set(frame_on_idxs)))
#    
#    nexpected_frames = runinfo['nvolumes'] * runinfo['ntiffs']
#    print "NVOLUMES: %i" % runinfo['nvolumes']
#    print "N TIFFS: %i" % runinfo['ntiffs']
#    nfound_frames = len(frame_on_idxs)
#    frame_offsets = {}
#    if nexpected_frames != nfound_frames:
#        # First check if there are discard frames (volumetric):
#        raw_dir = serialfn_path.split('/paradigm_files/')[0]
#        raw_si_path = glob.glob(os.path.join(raw_dir, '*SI*.json'))[0]
#        with open(raw_si_path, 'r') as f: rawsi = json.load(f)
#        nframes_per_volume = rawsi['File001']['SI']['hFastZ']['numFramesPerVolume'] #['numDiscardFlybackFrames'] 
#        nexpected_frames = nexpected_frames * nframes_per_volume 
#       
#    if nexpected_frames != nfound_frames: 
#        nframes_off = nexpected_frames - nfound_frames
#        print "*** Warning:  N expected (%i) does not match N found (%i).\n Missing %i frames." % (nexpected_frames, nfound_frames, nframes_off)
#        if nframes_off < 0:
#            # More frames found than expected (we're skipping a chunk of the frames):
#            frames_per_chunk = np.diff(long_breaks)
#            funky_stretches = [frix for frix,chunklength in enumerate(frames_per_chunk) if abs(chunklength - runinfo['nvolumes']) > 2] 
#            tif_files_with_delay = [tix+1 for tix in funky_stretches] # add 1 cuz diff adds 1 and break counts begin after File001
#            if len(tif_files_with_delay) > 0:
#                first_file_offset = min(tif_files_with_delay)
#                # All subsequent tifs should have appropriate offset:
#                last_tif = first_file_offset
#                nframes_to_add = {}
#                for tix in np.arange(first_file_offset, runinfo['ntiffs']):
#                    if tix in tif_files_with_delay:
#                        tmp_last_tif = [k for k in tif_files_with_delay if k > last_tif]
#                        if len(tmp_last_tif) > 0: # We hit the last offset
#                            last_tif = tmp_last_tif[0]
#                        
#                    nframes_to_add.update({tix: frames_per_chunk[last_tif-1]})
#            else:
#                # There are extra frames because we're ignoring chunks at the end (liekly incompletel chunks):
#                first_file_offset = runinfo['ntiffs']
#                
#            for tix in range(runinfo['ntiffs']):
#                if tix >= first_file_offset:
#                    print "---> Found a funky delay (extra frames not beloning to a tif) prior to File%03d" % (tix+1)
#                    frame_offsets.update({tix: nframes_to_add[tix]})
#                else:
#                    frame_offsets.update({tix: 0})
#        else:
#            frame_offsets = dict((tix, 0) for tix in range(runinfo['ntiffs']))
# 
#    else:
#        frame_offsets = dict((tix, 0) for tix in range(runinfo['ntiffs']))
#                
#                
#    use_loop = False # True 
#
#    ### Get arduino-processed bitcodes for each frame: frame_on_idxs[8845]
#    frame_bitcodes = dict(); #ix = 0; #codes = [];
#    missed_triggers = 0
#    for idx,frameidx in enumerate(frame_on_idxs):
#        #framenum = 'frame'+str(idx)
#        if idx==len(frame_on_idxs)-1:
#            bcodes = all_bitcodes[frameidx:]
#        else:
#            bcodes = all_bitcodes[frameidx:frame_on_idxs[idx+1]]
#
#    #### Split bitcodes per frame in half for higher "resolution" of bcodes per frame
#        halfmark = int(np.floor(len(bcodes)/2))
#
#        tmp_codes_0 = bcodes[0:halfmark]
#        tmp_codes_1 = bcodes[halfmark:]
#        
#        if (nreads_per_frame/2)+1 < len(tmp_codes_0) < 100:
#            missed_triggers += 1
#            # This is a missed trigger, and not a inter-block mark
#            frame_bitcodes['%i_p0a' % idx] = bcodes[0:halfmark/2]
#            frame_bitcodes['%i_p0b' % idx] = bcodes[halfmark/2:halfmark]
#            frame_bitcodes['%i_p1a' % idx] = bcodes[halfmark:halfmark+halfmark/2]
#            frame_bitcodes['%i_p1b' % idx] = bcodes[halfmark+halfmark/2:]
#        else:
#            frame_bitcodes['%i_p0' % idx] = tmp_codes_0
#            frame_bitcodes['%i_p1' % idx] = tmp_codes_1
#
#        # ix += 1
#       
#    print "Found %i missed triggers! - index: %i, trigger ix: %i" % (missed_triggers, idx, frameidx)
#    print sorted(frame_bitcodes.keys()[0:5], key=natural_keys)
#
#    ### Find first frame of MW experiment start:
#    modes_by_frame = dict((fr, int(frame_bitcodes[fr].mode()[0])) \
#                              for fr in sorted(frame_bitcodes.keys(), key=natural_keys)) # range(len(frame_on_idxs)))
##%
#    # Take the 2nd frame that has the first-stim value (in case bitcode of Image on Trial1 is 0):
#    trialnames = sorted(mwtrials.keys(), key=natural_keys)
#    if 'grating' in mwtrials[trialnames[0]]['stimuli']['type']:
#        first_stim_frame = [k for k in sorted(modes_by_frame.keys(), key=natural_keys) if modes_by_frame[k]>0][0]
#    else:
#        first_stim_frame = [k for k in sorted(modes_by_frame.keys(), key=natural_keys) if modes_by_frame[k]>0][1] #[0]
#
#
#    ### Get all bitcodes and corresonding frame-numbers for each trial:
#    trialevents = dict()
#    allframes = sorted(frame_bitcodes.keys(), key=natural_keys) #, key=natural_keys
#
#    curr_frames = sorted(allframes[int(first_stim_frame.split('_')[0])+1:], key=natural_keys)
#    print "First stim frame:", curr_frames[0]
#    curr_frame_vals = list((k, modes_by_frame[k]) for k in curr_frames)
#
#    first_frame = first_stim_frame
#
###first_frame = 2289 
##F = copy.copy(curr_frames)
##
##F2 = copy.copy(curr_frames)
##
#######
##curr_frames = copy.copy(F)
##first_frame = F[0]
##prev_trial = 'trial00096'
##tidx = 96
##trial = 'trial00097'
## all_bitcodes[58390:58420]
#    #%%
##    trial = tmpframes['last_trial'] #'trial00287'
##    tidx = int(trial[5:]) - 1
##    curr_frames = tmpframes['starting_frame_set']
##    prev_trial = 'trial%05d' % (int(trial[5:])-1) #'trial00286' #'trial00001'
##    frame_bitcodes = tmpframes['frame_bitcodes']
##    modes_by_frame = tmpframes['modes_by_frame']
##    first_frame = '%i_p0' % int(curr_frames[0].split('_')[0])
###    
#    min_iti = min([mwtrials[t]['iti_duration']/1E3 for t in mwtrials.keys()])
#    prev_trial = 'trial00001'
#    skip_trial = False
#    #trial_frames = []
#    ntrials = len(mwtrials.keys())
#    durs = []
#    skipped = {}
#    
#    for tidx, trial in enumerate(sorted(mwtrials.keys(), key=natural_keys)): #[0:254]: #[0:46]):
#    #for tidx, trial in zip(np.arange(tidx, len(mwtrials.keys())), sorted(mwtrials.keys(), key=natural_keys)[tidx:]):
#        most_recent_skipped_list = [v for k, v in skipped.items()]
#        if trial in most_recent_skipped_list:
#            continue
#  
#        print "Parsing %s" % trial
#        #if trial == 'trial00379':
#        #    break
#    
#        # Create hash of current MWTRIAL dict:
#        mwtrial_hash = hashlib.sha1(json.dumps(mwtrials[trial], sort_keys=True)).hexdigest()
#        starting_frame_set = np.copy(curr_frames)
#
#        #print trial
#        trialevents[mwtrial_hash] = dict()
#        trialevents[mwtrial_hash]['trial'] = trial
#        trialevents[mwtrial_hash]['stim_dur_ms'] = mwtrials[trial]['stim_off_times'] - mwtrials[trial]['stim_on_times']
#
#        bitcodes = mwtrials[trial]['all_bitcodes']
#        
#        if mwtrials[trial]['missing_si_trigger']:
#            missed_last_trigger = True
#        else:
#            missed_last_trigger = False
#       
#        # With pre-ITI, first stimulus bitcode of first trial can be 0, 
#	# so make sure we skip all the actual blanks until we hit the first preITI.
#        if bitcodes[0] == 0 and trial=='trial00001':
#            iter_curr_frame_vals = iter(curr_frame_vals) 
#            #-----
#            first_frame = curr_frame_vals[0]; pframe = curr_frame_vals[0]; currframes_counter=0;
#            if verbose:
#                print "... first", first_frame
#                print "...", curr_frame_vals[0:5]
#            while first_frame[1] == 0:
#                pframe = np.copy(first_frame)
#                first_frame = next(iter_curr_frame_vals)
#                currframes_counter += 1
#            curr_frames = curr_frames[currframes_counter:]
#            curr_frame_vals = curr_frame_vals[currframes_counter:]
#            first_frame = curr_frame_vals[0][0]
#
#        # First, check if first "stimulus" bit-code is actually an image (in case frame-trigger  update missed at start)
#        minframes = 5 #4 #5 #4
#        
##        if trial == 'trial00131':
##            break
#        
#        if skip_trial: # Skip the PREVIOUS trials, so jump full trial's worth of frame indices:
#            print "... Skipping previous trials' frames (%s)..." % prev_trial
#            nframes_to_skip = 0
##            iti_frames = round(round(mwtrials[prev_trial]['iti_duration']/1E3, 1) * framerate, 1) 
##            stim_frames = round(round(mwtrials[prev_trial]['stim_duration']/1E3, 1) * framerate, 1) 
##            # NOTE:  Trials skipped if massive frame-shift error in bitcodes
##            # This is likely an indication of funky behavior during acquisition - we should skip these trials
##            # Based on 20180814 - JC008 gratings data, we don't want to skip full trial's worth of frames.
##            nframes_to_skip = int((stim_frames + iti_frames)) / 2
##            # Add nframes to actual frame index:
##            new_start_ix = int(curr_frames[0].split('_')[0]) + nframes_to_skip
##            # Get the subdivided frame key that this new frame corresponds to:
##            relative_frame_ix = [int(fr.split('_')[0]) for fr in curr_frames].index(new_start_ix)
##            curr_frames = sorted(curr_frames[relative_frame_ix:], key=natural_keys)
##
#       
#        elif (blank_start is True):
#            if tidx == 0:
#                if (np.median(frame_bitcodes[first_frame]) == mwtrials[trial]['stim_bitcode']):
#                    # If this is the first trial, don't skip anything
#                    print "FIRST TRIAL"
#                    nframes_to_skip= 0
#                else:
#                    nframes_to_skip = 0 #jj int(round(min_iti * framerate)) #3)
#
#            elif tidx > 0:
#                if bitcodes[0] == mwtrials[prev_trial]['all_bitcodes'][-1]:
#                    # Check that LAST trial's ITI is not the same as CURRENT trial's STIM
#                    # --> This can happen in between .tif files, since there is an additional PRE-ITI at start of each block.
#                    print "... Found a repeat! skipping..."
#
#                    if mwtrials[prev_trial]['block_idx'] != mwtrials[trial]['block_idx']:
#                        print "... and skipping extra ITI for block start"
#                        # Skip TWO ITI durs (start and end) plus a little extra
#                        #nframes_to_skip = int(round(np.floor((mwtrials[prev_trial]['iti_duration'])/1E3)*2.5 * framerate))
#                        if int(runinfo['session']) < 20180525:
#                            nframes_to_skip = int(round((min_iti * 1 + 1) * framerate))
#                        elif mwtrials[prev_trial]['missing_si_trigger']:
#                            nframes_to_skip = 1
#                        else:
#                            #nframes_to_skip = int(round((min_iti * 2 + 1) * framerate))
#                            # jj nframes_to_skip = int(round(((mwtrials[prev_trial]['iti_duration']/1E3) + min_iti*1 + .5) * framerate))
#                            nframes_to_skip = int(round(min_iti * framerate))
#
#                    else:
#                        #nframes_to_skip = int(np.floor(mwtrials[prev_trial]['iti_duration']/1E3) * framerate) #3)
#                        nframes_to_skip = 0 #jj int(round(min_iti * framerate)) #3)
#
#                elif mwtrials[prev_trial]['block_idx'] != mwtrials[trial]['block_idx']:
#                    print "... BLOCK start found, skipping extra..."
#                    if (mwtrials[prev_trial]['missing_si_trigger'] is False):
#                        # Skip 2 (start and end block) ITIs -- extra frames if start of new block (back to back ITIs in serial data):
#                        #nframes_to_skip = int((np.floor(mwtrials[prev_trial]['iti_duration']/1E3) * 1.5) * framerate) 
#                        nframes_to_skip = int(round(min_iti * 1.0) * framerate) 
#                    else:
#                        nframes_to_skip = 1
#                else:
#                    # Only skip 1 ITI's worth of frames:
#                    nframes_to_skip = 0 # jj int(round(min_iti * framerate)) #3)
#
#            print "... Skipping %.2f sec worth of frames." % (nframes_to_skip/framerate)
#                   
#            # Add nframes to actual frame index:
#            # jj new_start_ix = int(curr_frames[0].split('_')[0]) + nframes_to_skip
#            # Get the subdivided frame key that this new frame corresponds to:
#            # jj relative_frame_ix = [int(fr.split('_')[0]) for fr in curr_frames].index(new_start_ix)
#            # jj curr_frames = sorted(curr_frames[relative_frame_ix:], key=natural_keys)
#        curr_frames = sorted(curr_frames[nframes_to_skip:], key=natural_keys)          
#        curr_frame_vals = list((k, modes_by_frame[k]) for k in curr_frames)
#        if verbose:
#            print '... START:', curr_frames[0]
#
##%
#        #### For each bitcode of current trial, get the correpsonding frame(s):
#        first_found_frame = [] #8542 [(14, 8547), (6, 8592)]
#   
#        #print "N bitcodes:", len(bitcodes)
#        prevframe = 0; currframe = 0;
#        skip_trial = False
#        try:
#            for bi, bitcode in enumerate(bitcodes):
#                iter_curr_frame_vals = iter(curr_frame_vals) 
#                #-----
#                first_frame = curr_frame_vals[0]; pframe = curr_frame_vals[0]; currframes_counter=0;
#                if verbose:
#                    print "... first", first_frame
#                    print "...", curr_frame_vals[0:5]
#                while first_frame[1] != bitcode:
#                    pframe = np.copy(first_frame)
#                    #print pframe, bitcode
#                    first_frame = next(iter_curr_frame_vals)
#                    currframes_counter += 1
#                #-----
#                # jj first_frame = [(fi, fr) for fi, fr in enumerate(curr_frame_vals) if fr[1]==bitcode][0]
#                if bi > 0 and len(bitcodes) > 2:
#                    # ** Check for skipped frames -- should be relatively consecutive, otherwise we're frame-shifting.
#                    currframe = int(tuple(pframe)[0].split('_')[0])
#                    prevframe = int(first_found_frame[bi-1][0].split('_')[0])
#                    # Make sure that the time difference (s) between consecutive frames is below some min.
#                    frames_bw_bitcodes = (currframe - prevframe)
#                    if missed_last_trigger and frames_bw_bitcodes >= 20:
#                        break
#                    else:
#                        assert (currframe - prevframe)/framerate <= (20./framerate), "Break found in %s (missed %.2f). Skipping!" % (trial, (currframe - prevframe)/framerate)
#                first_found_frame.append(first_frame) #.append(pframe)
#                curr_frames = curr_frames[currframes_counter:]
#                curr_frame_vals = curr_frame_vals[currframes_counter:]
#
#            # 1) First skip:  Find the end of the current bitcode. 
#            #    Last bitcode in trial is ITI start. This is just the stim dequeue.
#            #    The next different bitcode should correspond to the long ITI after stim-dequeue of current trial,
#            #    i.e., the num skipped indices should be >= ITI duration.
#            # 2) Second skip:  Find start of the next trial's first stimulus bitcode.
#            #    Revert the frame counter 1 so that the next loop starts on the first bitcode of the next trial.
#            #    Unless the next trial occurs after a new block, in which case we skip the first ITI (blank start) of the block.
#            if trial != sorted(mwtrials.keys(), key=natural_keys)[-1]: # < len(mwtrials.keys()) - 1:
#                #curr_frame_vals = list((k, modes_by_frame[k]) for k in curr_frames)
#                last_frame = curr_frame_vals[0]; currframes_counter=0; nskips=0;
#                iter_curr_frame_vals = iter(curr_frame_vals)
#                if verbose:
#                    print '... last:', last_frame, bitcode
#                last_bitcode = np.copy(bitcode)
#                while (last_frame[1] == bitcode and nskips==0) or nskips < 2:
#                    last_frame = next(iter_curr_frame_vals)
#                    
#                    if last_bitcode != last_frame[1]:
#                        if verbose:
#                            print '...', last_frame, last_bitcode, '(', bitcode, ')' 
#                        nskips += 1
#                        last_bitcode = last_frame[1]
#                    currframes_counter += 1
#
#                #last_frame = [(fi, fr) for fi, fr in enumerate(curr_frame_vals) if fr[1]!=bitcode]
#                #print "...", last_frame[0], last_frame[1], bitcode
#                if verbose:
#                    print '... last bitcode: %i, first bitcode of next trial %i' % (bitcode, last_bitcode) 
#                    print "... Skipping %i indices." % currframes_counter
#                curr_frames = curr_frames[currframes_counter-2:]
#                curr_frame_vals = curr_frame_vals[currframes_counter-2:] 
#            
#        except Exception as e:
#            ok_to_pass = False
#            if isinstance(e, StopIteration) and missed_last_trigger and trial == sorted(mwtrials.keys(), key=natural_keys)[-1]:
#                #and (trial == sorted(mwtrials.keys(), key=natural_keys)[-1] or mwtrials[prev_trial]['block_idx'] != mwtrials[trial]['block_idx']):
#                # Last trial (or last trial in block), and missed SI triggers at end of tif blocks.
#                ok_to_pass = True
#                
#            elif mwtrials[prev_trial]['block_idx'] != mwtrials[trial]['block_idx'] and mwtrials[prev_trial]['missing_si_trigger'] and bitcodes[0] == mwtrials[prev_trial]['all_bitcodes'][-1]:
#                # Reset iterative loop:
#                nframes_to_skip = int(round(min_iti * framerate))*2
#                curr_frames = starting_frame_set.copy()
#                curr_frames = curr_frames[nframes_to_skip:]
#                curr_frame_vals = list((k, modes_by_frame[k]) for k in curr_frames)
#
#                #### For each bitcode of current trial, get the correpsonding frame(s):
#                first_found_frame = [] #8542 [(14, 8547), (6, 8592)]
#           
#                #print "N bitcodes:", len(bitcodes)
#                prevframe = 0; currframe = 0;
#                skip_trial = False
#                try:
#                    for bi, bitcode in enumerate(bitcodes):
#                        iter_curr_frame_vals = iter(curr_frame_vals) 
#                        #-----
#                        first_frame = curr_frame_vals[0]; pframe = curr_frame_vals[0]; currframes_counter=0;
#                        if verbose:
#                            print "... first", first_frame
#                            print "...", curr_frame_vals[0:5]
#                        while first_frame[1] != bitcode:
#                            pframe = np.copy(first_frame)
#                            #print pframe, bitcode
#                            first_frame = next(iter_curr_frame_vals)
#                            currframes_counter += 1
#                        #-----
#                        # jj first_frame = [(fi, fr) for fi, fr in enumerate(curr_frame_vals) if fr[1]==bitcode][0]
#                        if bi > 0 and len(bitcodes) > 2:
#                            # ** Check for skipped frames -- should be relatively consecutive, otherwise we're frame-shifting.
#                            currframe = int(tuple(pframe)[0].split('_')[0])
#                            prevframe = int(first_found_frame[bi-1][0].split('_')[0])
#                            # Make sure that the time difference (s) between consecutive frames is below some min.
#                            frames_bw_bitcodes = (currframe - prevframe)
#                            if missed_last_trigger and frames_bw_bitcodes >= 20:
#                                break
#                            else:
#                                assert (currframe - prevframe)/framerate <= (20./framerate), "Break found in %s (missed %.2f). Skipping!" % (trial, (currframe - prevframe)/framerate)
#                        first_found_frame.append(first_frame) #.append(pframe)
#                        curr_frames = curr_frames[currframes_counter:]
#                        curr_frame_vals = curr_frame_vals[currframes_counter:]
#        
#                    # 1) First skip:  Find the end of the current bitcode. 
#                    #    Last bitcode in trial is ITI start. This is just the stim dequeue.
#                    #    The next different bitcode should correspond to the long ITI after stim-dequeue of current trial,
#                    #    i.e., the num skipped indices should be >= ITI duration.
#                    # 2) Second skip:  Find start of the next trial's first stimulus bitcode.
#                    #    Revert the frame counter 1 so that the next loop starts on the first bitcode of the next trial.
#                    #    Unless the next trial occurs after a new block, in which case we skip the first ITI (blank start) of the block.
#                    if trial != sorted(mwtrials.keys(), key=natural_keys)[-1]: # < len(mwtrials.keys()) - 1:
#                        #curr_frame_vals = list((k, modes_by_frame[k]) for k in curr_frames)
#                        last_frame = curr_frame_vals[0]; currframes_counter=0; nskips=0;
#                        iter_curr_frame_vals = iter(curr_frame_vals)
#                        if verbose:
#                            print '... last:', last_frame, bitcode
#                        last_bitcode = np.copy(bitcode)
#                        while (last_frame[1] == bitcode and nskips==0) or nskips < 2:
#                            last_frame = next(iter_curr_frame_vals)
#                            
#                            if last_bitcode != last_frame[1]:
#                                if verbose:
#                                    print '...', last_frame, last_bitcode, '(', bitcode, ')' 
#                                nskips += 1
#                                last_bitcode = last_frame[1]
#                            currframes_counter += 1
#        
#                        #last_frame = [(fi, fr) for fi, fr in enumerate(curr_frame_vals) if fr[1]!=bitcode]
#                        #print "...", last_frame[0], last_frame[1], bitcode
#                        if verbose:
#                            print '... last bitcode: %i, first bitcode of next trial %i' % (bitcode, last_bitcode) 
#                            print "... Skipping %i indices." % currframes_counter
#                        curr_frames = curr_frames[currframes_counter-2:]
#                        curr_frame_vals = curr_frame_vals[currframes_counter-2:] 
#                    ok_to_pass = True
#                    
#                except Exception as e:
#                    print e
#                
#                
#            if not ok_to_pass:
#                print e
#                traceback.print_exc() 
#                print "\n%i | %i --sorted(mwtrials.keys(), key=natural_keys)[-1] Prev: %i, Curr: %i" % (bi, bitcode, prevframe, currframe)
#                print "bitcodes:", bitcodes
#                # Flag skip_trial TRUE so that on the next trial's parsing, we know to skip a portion of the frames...
#                skip_trial = True 
#    
#                # Reset iterative loop effects:
#                curr_frames = starting_frame_set.copy() # Revert incrementally-shortened frame-bank
#                curr_frame_vals = list((k, modes_by_frame[k]) for k in curr_frames)
#    
#                dump_last_state(mwtrial_path, trialevents, trial, starting_frame_set, first_found_frame,
#                                    bitcodes, modes_by_frame, frame_bitcodes, serialfn_path)        
#                # Identify which trials skipped:
#                starting_frames = sorted(starting_frame_set, key=natural_keys)
#                starting_bitcodes = [modes_by_frame[init_fr_ix] for init_fr_ix in starting_frames[0:10]]
#                # Get rid of repeats across frames
#                repeat_ixs = [rm_fr_ix+1 for rm_fr_ix, fr_val in enumerate(np.diff(starting_bitcodes)) if fr_val == 0]
#                next_found_trial_bitcodes = [next_bval for next_bix, next_bval in enumerate(starting_bitcodes) if next_bix not in repeat_ixs]
#                matching_trials = [mwt for mwt in mwtrials.keys() if ', '.join([str(bitcode_int) for bitcode_int in next_found_trial_bitcodes]) in str(mwtrials[mwt]['all_bitcodes'])]
#                next_found_trial = sorted([found_trial for found_trial in matching_trials if int(found_trial[5:]) > int(trial[5:])], key=natural_keys)[0]
#                
#                curr_skipped_trials = ['trial%05d' % int(skip_ix) for skip_ix in np.arange(int(trial[5:]), int(next_found_trial[5:]))]
#                #break
#        prev_trial = trial
#        
#        if skip_trial:
#            # Store skip info to remove from MW dicts and trialevents dict.
#            # Continue to skip bitcode/trial-duration checks for skipped trial.
#            for skipped_trial in curr_skipped_trials:
#                curr_mwtrial_hash = hashlib.sha1(json.dumps(mwtrials[skipped_trial], sort_keys=True)).hexdigest()
#
#                skipped.update({curr_mwtrial_hash: skipped_trial})
#                print "** SKIPPED: %s" % skipped_trial            
#            #continue
#
#        stim_on_frame = int(first_found_frame[0][0].split('_')[0])
#        last_found_frame_tmp = int(first_found_frame[-1][0].split('_')[0])
#        stim_dur_curr = (last_found_frame_tmp - stim_on_frame) / framerate
#        if missed_last_trigger and round(stim_dur_curr, 1) < round((mwtrials[trial]['stim_duration']/1E3), 1):
#            # Artifically add "last frame" index:
#            print "Fixing stim dur for missed trigger..."
#            stim_off_frame = stim_on_frame + int(round((mwtrials[trial]['stim_duration']/1E3)*framerate))
#        else:
#            try:
#                assert round(stim_dur_curr, 1) == round((mwtrials[trial]['stim_duration']/1E3), 1), "Bad stim duration..! %s:" % trial 
#                stim_off_frame = int(first_found_frame[-1][0].split('_')[0])
#            except Exception as e:
#                dump_last_state(mwtrial_path, trialevents, trial, starting_frame_set, first_found_frame,
#                                    bitcodes, modes_by_frame, frame_bitcodes, serialfn_path)        
#    
#                assert round(stim_dur_curr) == round(mwtrials[trial]['stim_duration']/1E3), "Bad stim duration..! %s:" % trial 
#            
#        # Do checks for stimulus duration:        
#        stim_dur_curr = round( (stim_off_frame - stim_on_frame)/framerate, 1) #round((float(first_found_frame[-1][0].split('_')[0]) - float(first_found_frame[0][0].split('_')[0]))/framerate, 1)
#        print stim_dur_curr, 'sec [%s]' % trial
##%%
#        # Append current trial's info to main trialevents dict:
#        durs.append(stim_dur_curr)
#        #trial_frames.append(first_found_frame)
#        
#        trialevents[mwtrial_hash]['stim_on_idx'] = stim_on_frame #int(first_found_frame[0][0].split('_')[0])
#        trialevents[mwtrial_hash]['stim_off_idx'] = stim_off_frame #int(first_found_frame[-1][0].split('_')[0])
#        trialevents[mwtrial_hash]['mw_trial'] = mwtrials[trial]
#        trialevents[mwtrial_hash]['block_frame_offset'] = frame_offsets[mwtrials[trial]['block_idx']]
#        
#        #prev_trial = trial
#    #%%
#    # Do a final check of all stim durs:
#    rounded_durs = list(set([round(d, 1) for d in durs]))
#    unique_mw_stim_durs = list(set([round(mwtrials[t]['stim_duration']/1E3, 1) for t in mwtrials.keys()]))
#    if len(rounded_durs) != len(unique_mw_stim_durs):
#        print " *** WARNING -- funky stim durs found:", rounded_durs
#        print " --- found MW stim durs:", unique_mw_stim_durs
#    
#    if skipped != {}:
#        # Rename original parsed:
#        mw_fname = os.path.splitext(os.path.split(mwtrial_path)[-1])[0]
#        orig_fpath = os.path.join(os.path.split(mwtrial_path)[0], '_orig_%s.json' % mw_fname)
#        with open(orig_fpath, 'w') as f:
#            json.dump(mwtrials, f, indent=True, sort_keys=True)
#    
#    # Remove skipped, then resave:
#    for hkey,nkey in skipped.items():
#        print "Removing skipped trial: %s (%s)" % (nkey, hkey)
#        if hkey in trialevents.keys():
#            del trialevents[hkey]
#        if nkey in mwtrials.keys():
#            del mwtrials[nkey]
#
#    with open(mwtrial_path, 'w') as f:
#        json.dump(mwtrials, f, indent=True, sort_keys=True)
#    print "Updated MWtrials and trial events."
#    print "TOTAL N trials:", len(mwtrials.keys())
##%%
#    return trialevents
#

#%%
#gaps = []
#for fi in np.arange(0, len(first_found_frame)-1):
#    nsec_diff = (float(first_found_frame[fi+1][0].split('_')[0]) - float(first_found_frame[fi][0].split('_')[0])) / framerate
#    nframes_diff = int(first_found_frame[fi+1][0].split('_')[0]) - int(first_found_frame[fi][0].split('_')[0])
#    if nframes_diff > 1:
#        print "%i: %i, %i" % (fi, nsec_diff, nframes_diff)
#        gaps.append(first_found_frame[fi][0])
    
#%%

def extract_options(options):
    parser = optparse.OptionParser()

    parser.add_option('-D', '--root', action='store', dest='rootdir', default='/nas/volume1/2photon/data', help='data root dir (root project dir containing all animalids) [default: /nas/volume1/2photon/data, /n/coxfs01/2pdata if --slurm]')
    parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')

    # Set specific session/run for current animal:
    parser.add_option('-S', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID')
    parser.add_option('-A', '--acq', action='store', dest='acquisition', default='FOV1', help="acquisition folder (ex: 'FOV1_zoom3x') [default: FOV1]")
    parser.add_option('-R', '--run', action='store', dest='run', default='', help="name of run dir containing tiffs to be processed (ex: gratings_phasemod_run1)")
    parser.add_option('--slurm', action='store_true', dest='slurm', default=False, help="set if running as SLURM job on Odyssey")
    parser.add_option('--verbose', action='store_true', dest='verbose', default=False, help="set if want to print extra info for parsing")


    parser.add_option('--dynamic', action="store_true",
                      dest="dynamic", default=False, help="Set flag if using image stimuli that are moving (*NOT* movies).")

    parser.add_option('--retinobar', action="store_true",
                      dest="retinobar", default=False, help="Set flag if stimulus is moving-bar for retinotopy.")
    parser.add_option('--phasemod', action="store_true",
                      dest="phasemod", default=False, help="Set flag if using dynamic, phase-modulated gratings.")
    parser.add_option('-t', '--triggervar', action="store",
                      dest="frametrigger_varname", default='frame_trigger', help="Temp way of dealing with multiple trigger variable names [default: frame_trigger]")
    parser.add_option('--multi', action="store_false",
                      dest="single_run", default=True, help="Set flag if multiple start/stops in run.")
    parser.add_option('--no-blank', action="store_false",
                      dest="blank_start", default=True, help="Set flag if no ITI blank period before first trial.")
    parser.add_option('-b', '--boundidx', action="store",
                      dest="boundidx", default=0, help="Bound idx if single_run is True [default: 0]")

    parser.add_option('--auto', action="store_true",
                      dest="auto", default=False, help="Set flag if NOT interactive.")

    parser.add_option('--backlight', action="store_true",
                      dest="backlight_sensor", default=False, help="Set flag if using backlight sensor.")
    parser.add_option('-E', action="store",
                      dest="experiment_type", default=None, help="Set to tiled_retinotopy if doing tiled gratings")



    (options, args) = parser.parse_args(options)

    return options

#%%
#rootdir = '/mnt/odyssey'
#animalid = 'CE077'
#session = '20180516'
#acquisition = 'FOV1_zoom1x'
#run = 'blobs_movies_run2'
#slurm = False
#retinobar = False
#phasemod = False
#trigger_varname = 'frame_trigger'
#stimorder_files = False


def parse_acquisition_events(run_dir, blank_start=True, verbose=False):

    run = os.path.split(run_dir)[-1]
    runinfo_path = os.path.join(run_dir, '%s.json' % run)

    with open(runinfo_path, 'r') as fr:
        runinfo = json.load(fr)
    nfiles = runinfo['ntiffs']
    file_names = sorted(['File%03d' % int(f+1) for f in range(nfiles)], key=natural_keys)

    #%

    # Set outpath to save trial info file for whole run:
    outdir = os.path.join(run_dir, 'paradigm')

    #%
    # =============================================================================
    # Get SERIAL data:
    # =============================================================================
    paradigm_rawdir = os.path.join(run_dir, runinfo['rawtiff_dir'], 'paradigm_files')
    serialdata_fns = sorted([s for s in os.listdir(paradigm_rawdir) if s.endswith('txt') and 'serial' in s and not s.startswith('._')], key=natural_keys)
    print "Found %02d serial-data files, and %i TIFFs." % (len(serialdata_fns), nfiles)

    if len(serialdata_fns) < nfiles:
        one_to_one = False
    else:
        one_to_one = True

    # Load MW info:
    paradigm_outdir = os.path.join(run_dir, 'paradigm', 'files')
    mwtrial_fns = sorted(glob.glob(os.path.join(paradigm_outdir, 'parsed_trials*.json')), key=natural_keys)
    print "Found %02d MW files, and %02d ARD files." % (len(mwtrial_fns), len(serialdata_fns))


    #%
    # =============================================================================
    # Create <RUN_DIR>/paradigm/trials_<TRIALINFO_HASH>.json file
    # =============================================================================
    RUN = dict()
    trialnum = 0
    for fid,serialfn in enumerate(sorted(serialdata_fns, key=natural_keys)):

        framerate = runinfo['frame_rate'] # 44.68 #float(runinfo['frame_rate'])

        currfile = "File%03d" % int(fid+1)

        print "================================="
        print "Processing files:"
        print "MW: ", mwtrial_fns[fid]
        print "ARD: ", serialdata_fns[fid]
        print "---------------------------------"

        # Load MW parsed trials:
        mwtrial_path = os.path.join(paradigm_outdir, mwtrial_fns[fid])

        # Load Acquisition serialdata info:
        serialfn_path = os.path.join(paradigm_rawdir, serialfn)

        # Align MW events to frame-events from serialdata:
        trialevents = extract_frames_to_trials(serialfn_path, mwtrial_path, runinfo, blank_start=blank_start, verbose=verbose)

        # Sort trials in run by time:
        sorted_trials_in_run = sorted(trialevents.keys(), key=lambda x: trialevents[x]['stim_on_idx'])
        sorted_stim_frames = [(trialevents[t]['stim_on_idx'], trialevents[t]['stim_off_idx']) for t in sorted_trials_in_run]

        # Create a dictionary for each trial in the run that specifies ALL info:
        # SI info:
        #     - frame indices for sitm ON/OFF
        #     - meta info (block number in run, ntiffs per behavior file, etc.)
        # AUX info:
        #     - stimulus info (from MW)
        #     - stimulus presentation info
        # META info:
        #     - paths to MW and serial data info that are the source of this dict's contents
        trialnum = 0
        for trialhash in sorted_trials_in_run:
            #trialnum += 1
            trialname = trialevents[trialhash]['trial'] #'trial%05d' % int(trialnum)
            trialnum = int(trialname[5:])

            RUN[trialname] = dict()
            RUN[trialname]['trial_hash'] = trialhash
            RUN[trialname]['block_idx'] = trialevents[trialhash]['mw_trial']['block_idx']
            if one_to_one is True:
                RUN[trialname]['ntiffs_per_auxfile'] = 1
            else:
                RUN[trialname]['ntiffs_per_auxfile'] = nfiles
            RUN[trialname]['behavior_data_path'] = mwtrial_path
            RUN[trialname]['serial_data_path'] = serialfn_path

            RUN[trialname]['start_time_ms'] = trialevents[trialhash]['mw_trial']['start_time_ms']
            RUN[trialname]['end_time_ms'] = trialevents[trialhash]['mw_trial']['end_time_ms']
            RUN[trialname]['stim_dur_ms'] = trialevents[trialhash]['mw_trial']['stim_off_times']\
                                                    - trialevents[trialhash]['mw_trial']['stim_on_times']
            RUN[trialname]['iti_dur_ms'] = trialevents[trialhash]['mw_trial']['iti_duration']
            RUN[trialname]['stimuli'] = trialevents[trialhash]['mw_trial']['stimuli']

            RUN[trialname]['frame_stim_on'] = trialevents[trialhash]['stim_on_idx']
            RUN[trialname]['frame_stim_off'] = trialevents[trialhash]['stim_off_idx']
            RUN[trialname]['trial_in_run'] = trialnum
            RUN[trialname]['block_frame_offset'] = trialevents[trialhash]['block_frame_offset']


    # Get unique hash for current RUN dict:
    run_trial_hash = hashlib.sha1(json.dumps(RUN, indent=4, sort_keys=True)).hexdigest()[0:6]

    # Move old files to subdir 'old' so that there is no confusion with hashed files:
    existing_files = [f for f in os.listdir(outdir) if 'trials_' in f and f.endswith('json') and run_trial_hash not in f]
    if len(existing_files) > 0:
        old = os.path.join(os.path.split(outdir)[0], 'paradigm', 'old')
        if not os.path.exists(old):
            os.makedirs(old)
        for f in existing_files:
            shutil.move(os.path.join(outdir, f), os.path.join(old, f))

    parsed_run_outfile = os.path.join(outdir, 'trials_%s.json' % run_trial_hash)
    with open(parsed_run_outfile, 'w') as f:
        json.dump(RUN, f, sort_keys=True, indent=4)
        
        
    configs, stimtype = util.save_stimulus_configs(RUN, outdir)


    return parsed_run_outfile

#%%

#options = ['-D', '/mnt/odyssey', '-i', 'CE077', '-S', '20180425', '-A', 'FOV1_zoom1x','-R', 'blobs_run1']
#options = ['-D', '/mnt/odyssey', '-i', 'CE077', '-S', '20180609', '-A', 'FOV1_zoom1x','-R', 'blobs_run1']
#options = ['-D', '/mnt/odyssey', '-i', 'CE077', '-S', '20180627', '-A', 'FOV1_zoom1x','-R', 'gratings_rotating_static']
#options = ['-D', '/Volumes/coxfs01/2p-data', '-i', 'JC008', '-S', '20180814', '-A', 'FOV1_zoom1x',
#           '-R', 'gratings_run1']
options = ['--slurm', '-i', 'JC076', '-S', '20190405', '-A', 'FOV2_zoom2p0x',
           '-R', 'gratings_run1']


           #%%
def main(options):
    # ================================================================================
    # MW trial extraction:
    # ================================================================================
    options = extract_options(options)

    # Set USER INPUT options:
    rootdir = options.rootdir
    animalid = options.animalid
    session = options.session
    acquisition = options.acquisition
    run = options.run
    verbose = options.verbose 
    slurm = options.slurm

    if slurm is True and 'coxfs01' not in rootdir:
        rootdir = '/n/coxfs01/2p-data'

    # MW specific options:
    retinobar = options.retinobar
    phasemod = options.phasemod
    dynamic = options.dynamic
    auto = options.auto
    
    trigger_varname = options.frametrigger_varname
    single_run = options.single_run
    blank_start = options.blank_start
    boundidx = int(options.boundidx)

    stimorder_files = False #True
    backlight_sensor = options.backlight_sensor
    experiment_type = options.experiment_type

    mwopts = ['-D', rootdir, '-i', animalid, '-S', session, '-A', acquisition, '-R', run, '-t', trigger_varname, '-b', boundidx, '-E', experiment_type]
    if slurm is True:
        mwopts.extend(['--slurm'])
    if dynamic is True:
        mwopts.extend(['--dynamic'])
    if retinobar is True:
        mwopts.extend(['--retinobar'])
    if phasemod is True:
        mwopts.extend(['--phasemod'])
    if single_run is False:
        mwopts.extend(['--multi'])
    if verbose is True:
        mwopts.extend(['--verbose'])
    if auto is True:
        mwopts.extend(['--auto'])
    if backlight_sensor:
        mwopts.extend(['--backlight'])
    
    #%
    paradigm_outdir = mw.parse_mw_trials(mwopts)
    print "----------------------------------------"
    print "Extracted MW events!"
    print "Outfile saved to:\n%s" % paradigm_outdir
    print "----------------------------------------"

    #%
    if stimorder_files is True:
        mw.create_stimorder_files(paradigm_outdir)

    # Set reference path and get SERIALDATA info:
    # ================================================================================
    if retinobar is False:
        run_dir = os.path.join(rootdir, animalid, session, acquisition, run)
        parsed_run_outfile = parse_acquisition_events(run_dir, blank_start=blank_start, verbose=verbose)
    print "----------------------------------------"
    print "ACQUISITION INFO saved to dir: %s" % paradigm_outdir
    print "----------------------------------------"


if __name__ == '__main__':
    main(sys.argv[1:])

