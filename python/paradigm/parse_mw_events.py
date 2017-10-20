#!/usr/bin/env python2

import numpy as np
import os
import cPickle as pkl
import scipy.signal
import numpy.fft as fft
import sys
import optparse
from PIL import Image
import re
import itertools
from scipy import ndimage

import time
import datetime

import pandas as pd

from bokeh.io import gridplot, output_file, show
from bokeh.plotting import figure
import csv

import pymworks 
import pandas as pd
import operator
import codecs

import scipy.io
import copy


# Abstract struct class       
class Struct:
    def __init__ (self, *argv, **argd):
        if len(argd):
            # Update by dictionary
            self.__dict__.update (argd)
        else:
            # Update by position
            attrs = filter (lambda x: x[0:2] != "__", dir(self))
            for n in range(len(argv)):
                setattr(self, attrs[n], argv[n])


class cycstruct(Struct):
    times = []
    idxs = 0
    vals = 0
    ordernum = 0
    triggers = 0


def get_timekey(item):
    return item.time


# prepend = '/Users/julianarhee'
# source_dir = os.path.join(prepend,'nas/volume1/2photon/RESDATA/20170724_CE051/retinotopy1') #options.source_dir #'/nas/volume1/2photon/RESDATA/TEFO/20160118_AG33/fov1_gratings1'
# stimtype = 'bar' #options.stimtype #'grating'
# mask = False #options.mask # False
# long_trials = False #options.long_trials #True
# no_ard = False #options.no_ard

# # Look in child dir (of source_dir) to find mw_data paths:
# mw_files = os.listdir(os.path.join(source_dir, 'mw_data'))
# mw_files = [m for m in mw_files if m.endswith('.mwk')]


# # In[6]:


# mwfile = mw_files[0]

# fn_base = mwfile[:-4]
# mw_data_dir = os.path.join(source_dir, 'mw_data')
# mw_fn = fn_base+'.mwk'
# dfn = os.path.join(mw_data_dir, mw_fn)
# dfns = [dfn]

# print "MW file: ", dfns


def get_session_bounds(dfn):

    df = None
    df = pymworks.open(dfn)                                                          # Open the datafile

    # First, find experiment start, stop, or pause events:
    modes = df.get_events('#state_system_mode')                                      # Find timestamps for run-time start and end (2=run)
    start_ev = [i for i in modes if i['value']==2][0]                                # 2=running, 0 or 1 = stopped

    run_idxs = [i for i,e in enumerate(modes) if e['time']>start_ev['time']]         # Get all "run states" if more than 1 found

    end_ev = next(i for i in modes[run_idxs[0]:] if i['value']==0 or i['value']==1)  # Find the first "stop" event after the first "run" event

    # Create a list of runs using start/stop-event times (so long as "Stop" button was not pressed during acquisition, only 1 chunk of time)
    bounds = []
    bounds.append([start_ev.time, end_ev.time])
    for r in run_idxs[1:]: 
        if modes[r].time < bounds[-1][1]: #end_ev.time:  # Ignore any extra "run" events if there was no actual "stop" event
            print "skipping extra START ev..."
            continue
        else:                            # Otherwise, find the next "stop" event if any additional/new "run" events found.
            try:
                stop_ev = next(i for i in modes[r:] if i['value']==0 or i['value']==1)
            except StopIteration:
                end_event_name = 'trial_end'
                print "NO STOP DETECTED IN STATE MODES. Using alternative timestamp: %s." % end_event_name
                stop_ev = df.get_events(end_event_name)[-1]
                print stop_ev
            bounds.append([modes[r]['time'], stop_ev['time']])

    bounds[:] = [x for x in bounds if ((x[1]-x[0])/1E6)>1]
    # print "................................................................"
    print "****************************************************************"
    print "Parsing file\n%s... " % dfn
    print "Found %i start events in session." % len(bounds)
    print "Bounds: ", bounds
    for bidx, bound in enumerate(bounds):
        print "bound ID:", bidx, (bound[1]-bound[0])/1E6, "sec"
    print "****************************************************************"

    return df, bounds


def get_trigger_times(df, boundary, triggername=''):
    # deal with inconsistent trigger-naming:
    codec_list = df.get_codec()
    if len(triggername)==0:
	trigger_names = [i for i in codec_list.values() if ('trigger' in i or 'Trigger' in i) and 'flag' not in i]
        if len(trigger_names) > 1:
	    print "Found > 1 name for frame-trigger:"
	    print "Choose: ", trigger_names
            print "Hint: RSVP could be FrameTrigger, otherwise frame_trigger."
	    trigg_var_name = raw_input("Type var name to use: ")
	    trigg_evs = df.get_events(trigg_var_name)
	else:
	    trigg_evs = df.get_events(trigger_names[0])
    else:
	trigg_evs = df.get_events(triggername)

    # Only include SI trigger events if they were acquired while MW was actually "running" (i.e., start/stop time boundaries):
    trigg_evs = [t for t in trigg_evs if t.time >= boundary[0] and t.time <= boundary[1]]
    #print trigg_evs

    getout=0
    while getout==0:
        # Find all trigger LOW events after the first onset trigger (frame_trigger=0 when SI frame-trigger is high, =1 otherwise)
        tmp_first_trigger_idx = [i for i,e in enumerate(trigg_evs) if e.value==0][0]        # Find 1st SI frame trigger received by MW (should be "0")
        first_off_ev = next(i for i in trigg_evs[tmp_first_trigger_idx:] if i['value']==1)  # Find the next "frame-off" event from SI (i.e., when MW is waiting for the next DI trigger)
        first_off_idx = [i.time for i in trigg_evs].index(first_off_ev.time)                # Get corresponding timestamp for first SI frame-off event

        # NOTE:  In previous versions of MW protocols, frame-trigger incorrectly reassigned on/off values...
        # Make sure only 1 OFF event for each ON event, and vice versa.
        # Should abort to examine trigger values and tstamps, but likely, will want to take the "frame ON" event immediately before the found "frame OFF" event.
        # (This is because previously, we didn't realize MW's receipt of DI from SI was actually "0" (and frame_trigger was  being used as a flag to turn HIGH, i.e., 1, if trigger received from SI))
        if not trigg_evs[first_off_idx-1].time==trigg_evs[tmp_first_trigger_idx].time:
            print "Incorrect sequence of frame-triggers detected in MW trigger events received from SI:"
            trigg_evs

            # Let USER decide what to do next:
            print "Press <q> to quit and examine. Press <ENTER> to just use frame-ON idx immediately before found frame-OFF idx: "
            user_choice = raw_input()
            valid_response = 0
            while not valid_response:
                if user_choice=='':
                    print "Moving on..."
                    do_quickndirty = True
                    valid_response = 1
                elif user_choice=='q':
                    print "quitting..."
                    do_quickndirty = False
                    valid_response = 1
                else:
                    "Invalid entry provided. Try again."
                    user_choice = raw_input()
            if do_quickndirty:
                first_on_idx = first_off_idx - 1
                first_on_ev = trigg_evs[first_on_idx]
            else:
                print "ABORTING!"
                getout=1
        else:
            first_on_idx = first_off_idx -1
            first_on_ev = trigg_evs[first_on_idx]
            getout=1

    #print "first_on_idx: ", first_on_idx
    print "first SI-trigger ON event received: ", first_on_ev
    #print "first_off_idx: ", first_off_idx
    print "first SI-trigger OFF event received: ", first_off_ev
    print "Duration of first run: {0:.4f} sec.".format((first_off_ev.time - first_on_ev.time)/1E6)

    
    # Now, get all "trigger" boundaries that demarcate each "run" after first run:
    print "Incrementally searching to find each pair of ON/OFF trigger events..."
    found_trigger_evs = [[first_on_ev, first_off_ev]] # placeholder for off ev
    start_idx = copy.copy(first_off_idx)
    #print trigg_evs
    while start_idx < len(trigg_evs)-1: 
        #print start_idx
        try:
            found_new_start = False
            early_abort = False
            curr_chunk = trigg_evs[start_idx+1:] # Look for next OFF event after last off event 

            # try:
            curr_off_ev = next(i for i in curr_chunk if i['value']==1)
            curr_off_idx = [i.time for i in trigg_evs].index(curr_off_ev.time)
            curr_start_idx = curr_off_idx - 1  # next "frame-start" should be immediately before next found "frame-off" event
            curr_start_ev = trigg_evs[curr_start_idx]
            # if trigg_evs[curr_start_idx]['value']!=0:
            # # i.e., if prev-found ev with value=1 is not a true frame-on trigger (just a repeated event with same value), just ignore it.
            #     continue
            # else:
            found_new_start = True
            # except IndexError:
            #     break

            last_found_idx = [i.time for i in trigg_evs].index(curr_off_ev.time)
            found_trigger_evs.append([curr_start_ev, curr_off_ev])
            start_idx = last_found_idx #start_idx + found_idx
            #print start_idx
        except StopIteration:
            print "Got to STOP."
            if found_new_start is True:
                early_abort = True
            break

    # If no proper off-event found for a given start event (remember, we always look for the next OFF event), just use the end of the session as t-end.
    # Since we look for the next OFF event (rather than the next start), if we break out of the loop, we haven't cycled through all the trigg_evs.
    # This likely means that there is another frame-ON event, but not corresponding OFF event.
    if early_abort is True: 
        if found_new_start is True:
            found_trigger_evs.append([curr_chunk[curr_idx], end_ev])
        else:
            found_trigger_evs[-1][1] = end_ev


    trigger_evs = [t for t in found_trigger_evs if (t[1].time - t[0].time) > 1]
    trigger_times = [[t[0].time, t[1].time] for t in trigger_evs]
    # Remove trigger periods < 1sec (shorter than a trial): 
    trigger_times = [t for t in trigger_times if (t[1]-t[0])/1E6>1.0]
    print "TTT:", len(trigger_times)
    print "........................................................................................"
    print "Found %i chunks from frame-on/-off triggers:" % len(trigger_times)
    print "........................................................................................"
    for tidx,trigger in enumerate(trigger_times):
        print tidx, ": ", (trigger[1]-trigger[0])/1E6
    print "........................................................................................"
    if len(trigger_times)==1:
        user_run_selection = [0] #trigger_times[0]
    else: 
        runs_selected = 0
        while not runs_selected:
            print "Choose runs. Formatting hints:"
            print "To choose RANGE:  <0:20> to include 0th through 20th runs."
            print "To select specific runs:  <0,1,2,5> to only include runs 0,1,2, and 5."
            tmp_user_run_selection = raw_input("Select indices of runs to include, or press <enter> to accept all:\n")
            # user_run_selection = [int(i) for i in user_run_selection]
            # if any([i>= len(trigger_times) for i in user_run_selection]):
            if len(tmp_user_run_selection)==1 or ',' in tmp_user_run_selection:
                user_run_selection = [int(i) for i in tmp_user_run_selection.split(',')]
                if any([i>= len(trigger_times) for i in user_run_selection]):
                    print len(user_run_selection)
                    print "Bad index selected, try again."
                    continue
                else:
                    for i in user_run_selection:
                        print "Run:", i
                    confirm_selection = raw_input("Press <enter> to accept. Press 'r' to re-try.")
                    if confirm_selection=='':
                        runs_selected = 1
                    else:
                        continue
            elif len(tmp_user_run_selection)==0:
                user_run_selection = np.arange(0, len(trigger_times))
                print "Selected ALL runs.\n"
                runs_selected = 1
            elif ':' in tmp_user_run_selection:
                firstrun, lastrun = tmp_user_run_selection.split(':')
                user_run_selection = [i for i in np.arange(int(firstrun), int(lastrun)+1)]
                for i in user_run_selection:
                    print "Run:", i
                confirm_selection = raw_input("Press <enter> to accept. Press 'r' to re-try.")
                if confirm_selection=='':
                    runs_selected = 1
                else:
                    continue
            # else:
            #     for i in user_run_selection:
            #         print "Run:", i
            #     confirm_selection = raw_input("Press <enter> to accept. Press 'r' to re-try.")
            #     if confirm_selection=='':
            #         runs_selected = 1
            #     else:
            #         continue
            
    print "Selected %i runs." % len(user_run_selection)
    #if len(user_run_selection)>1:
    trigger_times = [trigger_times[i] for i in user_run_selection]

    return trigger_times, user_run_selection


def get_pixelclock_events(df, boundary, trigger_times=[]):
    # Get pixel-clock events:
    tmp_display_evs = df.get_events('#stimDisplayUpdate')                                                  # Get all stimulus-display-update events
    display_evs = [e for e in tmp_display_evs if e.value and not e.value[0]==None]                         # Filter out empty display-update events
    display_evs = [d for d in display_evs if d.time <= boundary[1] and d.time >= boundary[0]]              # Only include display-update events within time boundary of the session

    tmp_pixelclock_evs = [i for i in display_evs for v in i.value if 'bit_code' in v.keys()]                      # Filter out any display-update events without a pixel-clock event
    print "N pix-evs found in boundary: %i" % len(tmp_pixelclock_evs)
    
    if len(trigger_times)==0:
        pixelclock_evs = tmp_pixelclock_evs
    else:
        pixelclock_evs = [p for p in tmp_pixelclock_evs if p.time <= trigger_times[-1][1] and p.time >= trigger_times[0][0]] # Make sure pixel events are within trigger times...
    print "Got %i pix code events within SI frame-trigger bounds." % len(pixelclock_evs)
    #pixelevents.append(pixelclock_evs)
    
    return pixelclock_evs



def get_bar_events(dfn, stimtype='bar', triggername='', remove_orphans=True):
    """
    Open MW file and get time-stamped boundaries for acquisition.

    dfns : list of strings
        contains paths to each .mwk to be parsed

    remove_orphans : boolean
        for each response event, find best matching display update event
        if set to 'True' remove display events with unknown outcome events

    returns:
        pixelevents   : list of MW events for each pixel-clock update within bounds of acquisition trigger times
        stimevents    : dict with keys=runs, values=cycstruct class containing time, position, order in sesssion, and associated trigger times
        trigger_times : list of tuples containg frame-on and frame-off trigger time-stamps (in us)
    """
    #for dfn in dfns:

    df, bounds = get_session_bounds(dfn)

    # Use chunks of MW "run"-states to get all associate events:

    pixelevents = []
    stimulusevents = [] #dict()
    #trialevents = []
    triggertimes = []
    info = []
    for bidx,boundary in enumerate(bounds):
        #bidx = 0
        #boundary = bounds[0]
        if (boundary[1] - boundary[0]) < 1000000:
            print "Not a real boundary, only %i seconds found. Skipping." % int(boundary[1] - boundary[0])
            #continue

        print "................................................................"
        print "SECTION %i" % bidx
        print "................................................................"

        trigg_times, user_run_selection = get_trigger_times(df, boundary, triggername=triggername)
        print "selected runs:", user_run_selection
        pixelclock_evs = get_pixelclock_events(df, boundary, trigger_times=trigg_times)

        pixelevents.append(pixelclock_evs)

        # Get Image events:
        bar_update_evs = [i for i in pixelclock_evs for v in i.value if '_bar' in v['name']]

        # Get condition/run info:
        condition_evs = df.get_events('condition')
        print len(condition_evs)
        condition_names = ['left', 'right', 'bottom', 'top']  # 0=left start, 1=right start, 2=bottom start, 3=top start
        run_start_idxs = [i+1 for i,v in enumerate(condition_evs[0:len(condition_evs)-1]) if v.value==-1 and condition_evs[i+1].value>=0]  # non-run values for "condition" is -1
        run_start_idxs = [run_start_idxs[selected_run] for selected_run in user_run_selection]
        for run_idx,run_start_idx in enumerate(run_start_idxs):
            print "Run", run_idx, ": ", condition_names[condition_evs[run_start_idx].value]

        nruns = len(run_start_idxs)

        # Get all cycle info for each run (should be ncycles per run):
        ncycles = df.get_events('ncycles')[-1].value          # Use last value, since default value may be different
        target_freq = df.get_events('cyc_per_sec')[-1].value
        print "Target frequency: {0:.2f} Hz, {ncycles} cycles.".format(target_freq, ncycles=ncycles)

        # Use frame trigger times for each run to get bar-update events for each run:
        bar_evs_by_run = []
        for run_idx in range(nruns):         
            bar_evs_by_run.append([b for b in bar_update_evs if b.time <= trigg_times[run_idx][-1] and b.time >= trigg_times[run_idx][0]])

        print "Expected run duration: ~{0:.2f} seconds.".format((1/target_freq)*ncycles)
        print "Found %i runs." % nruns
        for runidx,bar_evs in enumerate(bar_evs_by_run):
            print "Run {runidx}: {0:.2f} s.".format((bar_evs[-1].time - bar_evs[0].time)/1E6, runidx=runidx)


        # For each run, parse bar-update events into the stuff we care about:
        # Each run has many "bar states", stored as list: [[t1, (xpos1, ypos1)], [t2, (xpos2, ypos2)], ..., [tN, (xposN, yposN)]]
        bar_states = []
        for curr_run_bar_evs in bar_evs_by_run:
            time_xy = [[update.time, (update.value[1]['pos_x'], update.value[1]['pos_y'])] for update in curr_run_bar_evs]
            bar_states.append(time_xy)

        # Sort bar events into a dict that contains all the session's runs:
        order_in_session = 0
        stimevents = dict()
        for ridx,run in enumerate(bar_states):
            if np.sum(np.diff([r[1][1] for r in run]))==0:                    # VERTICAL bar, since ypos does not change.
                positions = [i[1][0] for i in run]                            # Only "xpos" is changing value.
                if positions[0] < 0:                                          # LEFT of center is negative, so bar starts at left.
                    restarts = list(np.where(np.diff(positions) < 0)[0] + 1)  # Cycle starts occur when pos. goes from POS-->NEG.
                    curr_run = 'left'
                else:                                                         # RIGHT of center is positive, bar starts from right.
                    restarts = list(np.where(np.diff(positions) > 0)[0] + 1)  # Cycle starts occur when goes from NEG-->POS.
                    curr_run = 'right'
            else:                                                             # HORIZONTAL bar, xpos doesn't change.
                positions = [i[1][1] for i in run] 
                if positions[0] < 0:                                          # BELOW center is negative, bar starts at bottom.
                    restarts = list(np.where(np.diff(positions) < 0)[0] + 1)
                    curr_run = 'bottom'
                else:
                    restarts = list(np.where(np.diff(positions) > 0)[0] + 1)  # ABOVE center is positive, bar starts at top.
                    curr_run = 'top'

            restarts.append(0)                                                # Add 0 so first start is included in all starting-position indices.
            if curr_run in stimevents.keys():                                        # Add repetition number if this condition is a repeat
                ncond_rep = len([i for i in stimevents.keys() if i==curr_run]) 
                curr_run = curr_run + '_' + str(ncond_rep+1)

            stimevents[curr_run] = cycstruct()
            stimevents[curr_run].states = run
            stimevents[curr_run].idxs = sorted(restarts)
            stimevents[curr_run].vals = positions
            stimevents[curr_run].ordernum = order_in_session
            stimevents[curr_run].triggers = trigg_times[ridx] 
            order_in_session += 1
            
        stimulusevents.append(stimevents)
        triggertimes.append(trigg_times)
    
        session_info = get_session_info(df, stimtype='bar')
        session_info['tboundary'] = boundary
        
        info.append(session_info)

    # pdev_info = [(v['bit_code'], p.time) for p in pdevs for v in p.value if 'bit_code' in v.keys()]
    #return pixelevents, stimevents, triggtimes, session_info
    return pixelevents, stimulusevents, triggertimes, info


def get_session_info(df, stimtype='grating'):
    info = dict()
    if stimtype=='bar':
        ncycles = df.get_events('ncycles')[-1].value
        info['ncycles'] = ncycles
        info['target_freq'] = df.get_events('cyc_per_sec')[-1].value
        info['barwidth'] = df.get_events('bar_size_deg')[-1].value
    else:
        stimdurs = df.get_events('distractor_presentation_time')
        info['stimduration'] = stimdurs[-1].value
        itis = df.get_events('ITI_time')
        info['ITI'] = itis[-1].value
        sizes = df.get_events('stim_size')
        info['stimsize'] = sizes[-1].value
        info['ITI'] = itis[-1].value
        # stimulus types?
        # ntrials?
    return info

     
def get_stimulus_events(dfn, stimtype='grating', phasemod=True, triggername='frame_trigger', pixelclock=True):
    df, bounds = get_session_bounds(dfn)
    print bounds

    codec = df.get_codec()

    # Use chunks of MW "run"-states to get all associate events:
    pixelevents = []
    stimulusevents = [] #dict()
    trialevents = []
    triggertimes = []
    info = []
    for bidx,boundary in enumerate(bounds):
        #bidx = 0
        #boundary = bounds[0]
        if (boundary[1] - boundary[0]) < 3000000:
            print "Not a real boundary, only %i seconds found. Skipping." % int(boundary[1] - boundary[0])
            continue

        print "................................................................"
        print "SECTION %i" % bidx
        print "................................................................"

        trigg_times, user_run_selection = get_trigger_times(df, boundary, triggername=triggername)
      
        print "selected runs:", user_run_selection
        if pixelclock:
            num_non_stimuli = 3 # N stimuli on screen: pixel clock, background, image
            # Don't use trigger-times, since unclear how high/low values assigned from SI-DAQ...
            pixelclock_evs = get_pixelclock_events(df, boundary) #, trigger_times=trigg_times)
        else:
            num_non_stimuli = 2 # background + image
        pixelevents.append(pixelclock_evs)

        # Get Image events:
        if stimtype=='image':
            # do stuff
            pass
        elif stimtype=='grating':
            #tmp_image_evs = [d for d in display_evs for i in d.value if i['name']=='gabor']
            tmp_image_evs = [d for d in pixelclock_evs for i in d.value if 'type'in i.keys() and i['type']=='drifting_grating']

            start_times = [i.value[1]['start_time'] for i in tmp_image_evs] # Use start_time to ignore dynamic pixel-code of drifting grating since stim as actually static
            find_static = np.where(np.diff(start_times) > 0)[0] + 1
            find_static = np.append(find_static, 0)
            find_static = sorted(find_static)
            if phasemod:
                find_static = find_static[::2]

            image_evs = [tmp_image_evs[i] for i in find_static]
            print "Found %i total image onset events." % len(image_evs)

            first_stim_index = pixelclock_evs.index(image_evs[0])
            if first_stim_index>0:
                pre_iti_ev = pixelclock_evs[first_stim_index-1]
                pre_blank = True
            else:
                pre_blank = False

            # Get blank ITIs:
            print "Getting subsequent ITI for each image event..."
            im_idx = [[t.time for t in pixelclock_evs].index(i.time) for i in image_evs]
            iti_evs = []
            for im in im_idx:
                try:
                    next_iti = next(i for i in pixelclock_evs[im:] if len(i.value)==(num_non_stimuli-1))
                    iti_evs.append(next_iti)
                except StopIteration:
                    print "No ITI found after last image onset event.\n"
                #print display_evs[im]
            print "Found %i iti events after a stimulus onset." % len(iti_evs)

            
            # Double-check that stim onsets are happening BEFORE iti onsets (should be very close to stim ON duration):
            stim_durs = []
            off_sync = []
            for idx,(stim,iti) in enumerate(zip(image_evs, iti_evs)):
                stim_durs.append(iti.time - stim.time)
                if (iti.time - stim.time) < 0:
                    off_sync.append(idx)
            if len(off_sync)>0:
                print "WARNING: found %i off-sync events." % len(off_sync)
            print "Confirming {nonsets} stimulus onsets, with min-max durations of {0:.6f}-{0:.6f} sec.".format(min(stim_durs)/1E6, max(stim_durs)/1E6, nonsets=len(stim_durs))

            # Remove extra image-event if it does not have offset/ITI:
            if image_evs[-1].time > iti_evs[-1].time: # early-abort
                print "Removing extra image event that has no offset."
                image_evs.pop(-1) 

            if pre_blank:
                tmp_trial_evs = image_evs + iti_evs + [pre_iti_ev]
            else:
                tmp_trial_evs = image_evs + iti_evs

            trial_evs = sorted(tmp_trial_evs, key=get_timekey)

            #trialevents.append(tmp_trialevents)
            print "Length of trial epochs: ", len(trial_evs)
            print "Number of trials found: ", len(image_evs)
            
        stimulusevents.append(image_evs)
        trialevents.append(trial_evs)
        triggertimes.append(trigg_times)
        
        session_info = get_session_info(df, stimtype=stimtype)
        session_info['tboundary'] = boundary
        info.append(session_info)

    return pixelevents, stimulusevents, trialevents, triggertimes, info
    
    


def get_image_events(df, boundary, pixelclock_evs=[], stimtype='grating', mask=False):
    
    # Get all stimulus-udpate events within bounds:
    tmp_display_evs = df.get_events('#stimDisplayUpdate')                                                  # Get all stimulus-display-update events
    display_evs = [e for e in tmp_display_evs if e.value and not e.value[0]==None]                         # Filter out empty display-update events
    display_evs = [d for d in display_evs if d.time <= boundary[1] and d.time >= boundary[0]]              # Only include display-update events within time boundary of the session

    if len(pixelclock_evs)>0:
        pixelclock_evs = [i for i in display_evs for v in i.value if 'bit_code' in v.keys()]
        num_non_stimuli = 3 # N stimuli on screen: pixel clock, background, image
    if len(pixelclock_evs)==0:
        print "No pixel clock."
        pixelclock_evs = display_evs
        num_non_stimuli = 2 # N stimuli on screen: background, image
       
    # Get stimulus-onset info parsed into trials:
    if stimtype=='image':
        tmp_image_evs = [d for d in pixelclock_evs for i in d.value if 'filename' in i.keys() and '.png' in i['filename']]
        #stimevents.append(imdevs)

        # Find blank ITIs:
        if mask is True:
            iti_evs = [i for i in pixelclock_evs for v in i.value if v['name']=='blue_mask' and i not in tmp_image_evs]
        else:
            iti_evs = [i for i in pixelclock_evs if i.time>image_evs[0].time and i not in tmp_image_evs]

        tmp_trial_evs = tmp_image_evs + iti_evs
        trial_evs = sorted(tmp_trial_evs, key=get_timekey)
        
        image_evs = tmp_image_evs

    elif stimtype=='grating':
        #tmp_image_evs = [d for d in display_evs for i in d.value if i['name']=='gabor']
        tmp_image_evs = [d for d in display_evs for i in d.value if i['type']=='drifting_grating']
        
        start_times = [i.value[1]['start_time'] for i in tmp_image_evs] # Use start_time to ignore dynamic pixel-code of drifting grating since stim as actually static
        find_static = np.where(np.diff(start_times) > 0)[0]
        find_static = np.append(find_static, 0)
        find_static = sorted(find_static)
        image_evs = [tmp_image_evs[i+1] for i in find_static]
        print "got image events"
        #stimevents.append(imtrials)
        
        # Make sure only the 1st stimulus after a new-stim flag is counted (for guarantee-reward) experiments:
        newstim_evs = df.get_events('new_stimulus')
        new_stim_evs = [i for i in newstim_evs if i.value==1]
        print "N new-stimulus events:", len(new_stim_evs)
        first_image_idxs = []
        for idx,newev in enumerate(new_stim_evs[0:-1]):
            possible_image_evs = [i for i,ev in enumerate(image_evs) if ev.time>newev.time and ev.time<new_stim_evs[idx+1].time]
            first_image_idxs.append(possible_image_evs)
        first_image_idxs = [i[0] for i in first_image_idxs if len(i)>0]
        image_evs = [image_evs[i] for i,ev in enumerate(image_evs) if i in first_image_idxs]
        print "N image evs after taking only first image: ", len(image_evs)
        
        # Filter out image-events that were aborted:
        aborted_evs = df.get_events('trial_aborted')
        aborted_idxs = []
        aborted_evs = []
        for idx,imev in enumerate(image_evs[0:-1]):
            check_abort = [i.value for i in aborted_evs if i.time>imev.time and i.time<image_evs[idx+1].time]
            if sum(check_abort)>0:
                aborted_idxs.append(idx)
                aborted_evs.append(imev)
        print "N aborted images: ", len(aborted_idxs)
        image_evs = [image_evs[i] for i,ev in enumerate(image_evs) if i not in aborted_idxs]
        print "N image evs after removing aborted: ", len(image_evs)
        
        # Find blank ITIs:
        if mask is True:
            iti_evs = [i for i in pixelclock_evs if i.time>tmp_image_evs[0].time and i not in tmp_image_evs]
        else:
#             prevdev = [[i for i,d in enumerate(display_evs) if d.time < t.time][-1] for t in image_evs[1:]]
#             lastdev = [i for i,d in enumerate(display_evs) if d.time > image_evs[-1].time and len(d.value)<num_non_stimuli] # ignore the last "extra" ev (has diff time-stamp) - just wnt 1st non-grating blank
#             iti_evs = [display_evs[i] for i in prevdev]
#             if len(lastdev)>0:
#                 iti_evs.append(display_evs[lastdev[0]])
            #nonstim_evs = [i for i in display_evs if i not in tmp_image_evs]
            
            im_idx = [[t.time for t in display_evs].index(i.time) for i in image_evs]
            iti_evs = []
            for im in im_idx:
                try:
                    next_iti = next(i for i in display_evs[im:] if len(i.value)==(num_non_stimuli-1))
                    iti_evs.append(next_iti)
                except StopIteration:
                    print "No ITI found after this (should be last image_ev).\n"
                    #print display_evs[im]
            print "got iti events"
        
        # Check that we got all the blanks:
#         blanks = [i for i,p in enumerate(pixelclock_evs) if len(p.value)==(num_non_stimuli-1)]
#         mismatches = [i for i,(p,t) in enumerate(zip([pixelclock_evs[x] for x in blanks], iti_evs)) if not p==t]   
#         if len(mismatches)>0:
#             print "Mismatches found in parsing trials...."
#             print mismatches

        # Append a "trial off" at end, if needed:
        if image_evs[-1].time > iti_evs[-1].time: # early-abort
            print "Removing extra image event that has no offset."
            image_evs.pop(-1)                
        tmp_trial_evs = image_evs + iti_evs
        trial_evs = sorted(tmp_trial_evs, key=get_timekey)

    #trialevents.append(tmp_trialevents)
    print "Length of trial epochs: ", len(trial_evs)
    print "Number of trials found: ", len(image_evs)
    
    return image_evs, trial_evs, aborted_evs
