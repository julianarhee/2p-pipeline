#!/usr/bin/env python2

import numpy as np
import os
import optparse
import shutil
# from bokeh.io import gridplot, output_notebook, output_file, show
# from bokeh.plotting import figure
# output_notebook()

import cPickle as pkl
import pandas as pd
import re

from json_tricks.np import dump, dumps, load, loads
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
import json

import scipy.io
import copy

#%%
def serialize_json(instance=None, path=None):
    dt = {}
    dt.update(vars(instance))
    return dt


def get_timekey(item):
    return item.time


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]

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

#%% PARSE_MW_EVENTS methods:

#%%
def get_session_bounds(dfn, verbose=False):

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
            if verbose:
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
    print "****************************************************************"
    if verbose:
        print "Bounds: ", bounds
        for bidx, bound in enumerate(bounds):
            print "bound ID:", bidx, (bound[1]-bound[0])/1E6, "sec"

    return df, bounds


def get_trigger_times(df, boundary, triggername='', arduino_sync=True, verbose=False):
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

    getout=0
    # Find all trigger LOW events after the first onset trigger (frame_trigger=0 when SI frame-trigger is high, =1 otherwise)
    while getout==0:
        # Find 1st SI frame trigger received by MW (should be "0")
        tmp_first_trigger_idx = [i for i,e in enumerate(trigg_evs) if e.value==0][0]
        try:
            # Find the next "frame-off" event from SI (i.e., when MW is waiting for the next DI trigger)
            first_off_ev = next(i for i in trigg_evs[tmp_first_trigger_idx:] if i['value']==1)

            # Get corresponding timestamp for first SI frame-off event
            first_off_idx = [i.time for i in trigg_evs].index(first_off_ev.time)

        except StopIteration:
            print "No trigger OFF event found."
            early_abort = True
            for i,t in enumerate(trigg_evs):
                print i, t
                first_off_idx = len(trigg_evs)-1
                first_off_ev = trigg_evs[-1]
            #print "DUR:", (first_off_ev.time - trigg_evs[tmp_first_trigger_idx].time) / 1E6

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

#    if verbose is True:
#        #print "first_on_idx: ", first_on_idx
#        print "first SI-trigger ON event received: ", first_on_ev
#        #print "first_off_idx: ", first_off_idx
#        print "first SI-trigger OFF event received: ", first_off_ev

    # Now, get all "trigger" boundaries that demarcate each "run" after first run:
    print "Incrementally searching to find each pair of ON/OFF trigger events..."

    found_trigger_evs = [[first_on_ev, first_off_ev]] # placeholder for off ev
    chunkidx = 0
    print "Chunk %i: dur (s): %.2f" % (chunkidx, (first_off_ev.time-first_on_ev.time)/1E6)
    start_idx = copy.copy(first_off_idx)
    #print trigg_evs
    if start_idx<len(trigg_evs)-1:
        while start_idx < len(trigg_evs)-1:
            #print start_idx
            try:
                chunkidx += 1
                found_new_start = False
                early_abort = False
                curr_chunk = trigg_evs[start_idx+1:] # Look for next OFF event after last off event

                # try:
                curr_off_ev = next(i for i in curr_chunk if i['value']==1)
                curr_off_idx = [i.time for i in trigg_evs].index(curr_off_ev.time)
                curr_start_idx = curr_off_idx - 1  # next "frame-start" should be immediately before next found "frame-off" event
                curr_start_ev = trigg_evs[curr_start_idx]
                if curr_start_ev.value==0:
                    found_new_start = True
                    found_trigger_evs.append([curr_start_ev, curr_off_ev])
                    print "Chunk %i: dur (s): %.2f" % (chunkidx, (curr_off_ev.time - curr_start_ev.time)/1E6)
                else:
                    found_new_start = False

                last_found_idx = [i.time for i in trigg_evs].index(curr_off_ev.time)
                start_idx = last_found_idx #start_idx + found_idx
                #print start_idx

            except StopIteration:
                check_new_starts = [i for i in curr_chunk if i['value']==0]
                if len(check_new_starts) > 0:
                    print "Found new trigger-start, but no OFF event."
                    found_new_start = True
                else:
                    found_new_start = False
                if verbose is True:
                    print "Got to STOP."

                if found_new_start is True:
                    early_abort = True
                    break

        # If no proper off-event found for a given start event (remember, we always look for the next OFF event), just use the end of the session as t-end.
        # Since we look for the next OFF event (rather than the next start), if we break out of the loop, we haven't cycled through all the trigg_evs.
        # This likely means that there is another frame-ON event, but not corresponding OFF event.
        if early_abort is True:
            if found_new_start is True:
                print "Missing final frame-off event, just appending last frame-trigg event to go with found START ev."
                last_on_ev = curr_chunk[0]
                #print last_on_ev
                last_ev = trigg_evs[-1]
                found_trigger_evs.append([last_on_ev, last_ev])
            else:
                found_trigger_evs[-1][1] = trigg_evs[-1]


    trigger_evs = [t for t in found_trigger_evs if (t[1].time - t[0].time) > 1]
    trigger_times = [[t[0].time, t[1].time] for t in trigger_evs]

    # Remove trigger periods < 1sec (shorter than a trial):
    trigger_times = [t for t in trigger_times if (t[1]-t[0])/1E6>1.0]
    if verbose is True and len(trigger_times) > 1:
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


    print "Selected %i runs, if dur: %s sec." % (len(user_run_selection), (trigger_times[0][1] - trigger_times[0][0])/1E6)
    #if len(user_run_selection)>1:
    trigger_times = [trigger_times[i] for i in user_run_selection]

    return trigger_times, user_run_selection

#%%
def get_pixelclock_events(df, boundary, trigger_times=[], verbose=False):

    # Get all stimulus-display-update events:
    tmp_display_evs = df.get_events('#stimDisplayUpdate')
    # Filter out empty display-update events:
    display_evs = [e for e in tmp_display_evs if e.value and not e.value[0]==None]
    # Only include display-update events within time boundary of the session:
    display_evs = [d for d in display_evs if d.time <= boundary[1] and d.time >= boundary[0]]
    # Filter out any display-update events without a pixel-clock event:
    tmp_pixelclock_evs = [i for i in display_evs if 'bit_code' in i.value[-1].keys()]


    if verbose is True:
        print [p for p in tmp_pixelclock_evs if not 'bit_code' in p.value[-1].keys()]
        print "N pix-evs found in boundary: %i" % len(tmp_pixelclock_evs)

    if len(trigger_times)==0:
        pixelclock_evs = tmp_pixelclock_evs
    else:
        # Make sure pixel events are within trigger times...
        pixelclock_evs = [p for p in tmp_pixelclock_evs if p.time <= trigger_times[-1][1] and p.time >= trigger_times[0][0]]

    print "Got %i pix code events within SI frame-trigger bounds." % len(pixelclock_evs)

    return pixelclock_evs

#%%
def get_session_info(df, stimulus_type=None):
    info = dict()
    if stimulus_type=='retinobar':
        ncycles = df.get_events('ncycles')[-1].value
        info['ncycles'] = ncycles
        info['target_freq'] = df.get_events('cyc_per_sec')[-1].value
        info['barwidth'] = df.get_events('bar_size_deg')[-1].value
        info['stimulus'] = stimulus_type
    else:
        stimdurs = df.get_events('distractor_presentation_time')
        info['stimduration'] = stimdurs[-1].value
        itis = df.get_events('ITI_time')
        info['ITI'] = itis[-1].value
        sizes = df.get_events('stim_size')
        info['stimsize'] = sizes[-1].value
        info['ITI'] = itis[-1].value
        info['stimulus'] = stimulus_type

        # stimulus types?
        # ntrials?
    return info

#%%
def get_stimulus_events(dfn, phasemod=False, triggername='frame_trigger', pixelclock=True, verbose=False):

    df, bounds = get_session_bounds(dfn)
    #print bounds
    codec = df.get_codec()

    # Use chunks of MW "run"-states to get all associated events:
    pixelevents = []
    stimulusevents = [] #dict()
    trialevents = []
    triggertimes = []
    info = []
    for bidx,boundary in enumerate(bounds):
        if (boundary[1] - boundary[0]) < 3000000:
            print "Not a real boundary, only %i seconds found. Skipping." % int(boundary[1] - boundary[0])
            continue

        print "................................................................"
        print "SECTION %i" % bidx
        print "................................................................"

        trigg_times, user_run_selection = get_trigger_times(df, boundary, triggername=triggername)

        ### Get all pixel-clock events in current run:
       # print "selected runs:", user_run_selection
        if pixelclock:
            num_non_stimuli = 3 # N stimuli on screen: pixel clock, background, image
            # Don't use trigger-times, since unclear how high/low values assigned from SI-DAQ:
            pixelclock_evs = get_pixelclock_events(df, boundary) # trigger_times=trigg_times) #, trigger_times=trigg_times)
        else:
            num_non_stimuli = 2 # background + image

        pixelevents.append(pixelclock_evs)

        # Get stimulus type:
        stimtype = [d for d in pixelclock_evs if len(d.value) > 1 and 'type' in d.value[1].keys()][0].value[1]['type']

        ### Get Image events:
        if stimtype=='image':
            image_evs = [d for d in pixelclock_evs for i in d.value if 'type'in i.keys() and i['type']=='image']
        elif 'grating' in stimtype:
            tmp_image_evs = [d for d in pixelclock_evs for i in d.value if 'type'in i.keys() and i['type']=='drifting_grating']

	    # Use start_time to ignore dynamic pixel-code of drifting grating since stim as actually static
	    start_times = [i.value[1]['start_time'] for i in tmp_image_evs]
	    find_static = np.where(np.diff(start_times) > 0)[0] + 1
	    find_static = np.append(find_static, 0)
	    find_static = sorted(find_static)
	    if phasemod:
		# Just grab every other (for phaseMod, 'start_time' for phase1 and phase2 are different (just take 1st):
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
        if verbose is True:
            print "Getting subsequent ITI for each image event..."

        im_idx = [[t.time for t in pixelclock_evs].index(i.time) for i in image_evs]
        iti_evs = []
        for im in im_idx:
            try:
                next_iti = next(i for i in pixelclock_evs[im:] if len(i.value)==(num_non_stimuli-1))
                iti_evs.append(next_iti)
            except StopIteration:
                print "No ITI found after last image onset event.\n"

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
        if verbose is True:
            print "Length of trial epochs: ", len(trial_evs)
            print "Number of trials found: ", len(image_evs)

        stimulusevents.append(image_evs)
        trialevents.append(trial_evs)
        triggertimes.append(trigg_times)

        session_info = get_session_info(df, stimulus_type=stimtype)
        session_info['tboundary'] = boundary
        info.append(session_info)

    return pixelevents, stimulusevents, trialevents, triggertimes, info

#%%

def get_bar_events(dfn, triggername='', remove_orphans=True):
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

        session_info = get_session_info(df, boundary, stimulus_type='retinobar')
        session_info['tboundary'] = boundary

        info.append(session_info)

    # pdev_info = [(v['bit_code'], p.time) for p in pdevs for v in p.value if 'bit_code' in v.keys()]
    #return pixelevents, stimevents, triggtimes, session_info
    return pixelevents, stimulusevents, triggertimes, info


#%%
def check_nested(evs):
    if len(evs) > 1 and type(evs[0]) == list:
        evs = [item for sublist in evs for item in sublist]
        evs = list(set(evs))
        evs.sort(key=operator.itemgetter(1))
    else:
        evs = evs[0]

    return evs

#%%
def extract_trials(curr_dfn, retinobar=False, phasemod=False, trigger_varname='frame_trigger', verbose=False):

    print "Current file: ", curr_dfn
    if retinobar is True:
        pixelevents, stimevents, trigger_times, session_info = get_bar_events(curr_dfn, triggername=trigger_varname)
    else:
        pixelevents, stimevents, trialevents, trigger_times, session_info = get_stimulus_events(curr_dfn, phasemod=phasemod, triggername=trigger_varname, verbose=verbose)

    # -------------------------------------------------------------------------
    # For EACH boundary found for a given datafile (dfn), make sure all the events are concatenated together:
    # -------------------------------------------------------------------------
    pixelevents = check_nested(pixelevents)
    # Check that all possible pixel vals are used (otherwise, pix-clock may be missing input):
    # print [p for p in pixelevents if 'bit_code' not in p.value[-1].keys()]
    n_codes = set([i.value[-1]['bit_code'] for i in pixelevents])
    if len(n_codes)<16:
        print "Check pixel clock -- missing bit values..."
    stimevents = check_nested(stimevents)
    if retinobar is False:
        trialevents = check_nested(trialevents)
    trigger_times = check_nested(trigger_times)
    session_info = check_nested(session_info)
    print session_info

    if verbose is True:
        print "================================================================"
        print "MW parsing summary:"
        print "================================================================"

        print "Found %i pixel clock events." % len(pixelevents)
        print "Found %i stimulus on events." % len(stimevents)
        if retinobar is False:
            	print "Found %i trial epoch (stim ON + ITI) events." % len(trialevents)
        #print "Found %i runs (i.e., trigger boundary events)." % len(trigger_times)

    # on FLASH protocols, first real iamge event is 41
    print "Found %i trials, corresponding to %i TIFFs." % (len(stimevents), len(trigger_times))
    refresh_rate = 60


    # -------------------------------------------------------------------------
    # Creat trial-dicts for each trial in run:
    # -------------------------------------------------------------------------
    if retinobar is True:

        nexpected_pixelevents = int(round((1/session_info['target_freq']) * session_info['ncycles'] * refresh_rate)) # * len(trigger_times)))
        print "Expected %i pixel events, missing %i pevs." % (nexpected_pixelevents, nexpected_pixelevents-len(pixelevents))

        stimnames = ['left', 'right', 'top', 'bottom']

        # GET TRIAL INFO FOR DB:
        trial_list = [(stimevents[k].ordernum, k) for k in stimevents.keys()]
        trial_list.sort(key=lambda x: x[0])
        trial = dict((i+1, dict()) for i in range(len(stimevents)))

        for trialidx,mvtrial in enumerate(trial_list):
            mvname = mvtrial[1]
            trialnum = trialidx + 1
            trial[trialnum]['start_time_ms'] = round(stimevents[mvname].states[0][0]/1E3)
            trial[trialnum]['end_time_ms'] = round(stimevents[mvname].states[-1][0]/1E3)
            stimname = [i for i in stimnames if i in mvname][0]
            stimsize = session_info['target_freq']
            trial[trialnum]['stimuli'] = {'stimulus': stimname, 'position': stimevents[mvname].states[0][1], 'scale': stimsize}
            trial[trialnum]['stim_on_times'] = round(stimevents[mvname].states[0][0]/1E3)
            trial[trialnum]['stim_off_times'] = round(stimevents[mvname].states[-1][0]/1E3)
    else:

        ntrials = len(stimevents)
        post_itis = sorted(trialevents[2::2], key=get_timekey) # 0=pre-blank period, 1=first-static-stim-ON, 2=first-post-stim-ITI

        # Get dynamic-grating bicode events:
        dynamic_stim_bitcodes = []
        bitcodes_by_trial = dict((i+1, dict()) for i in range(len(stimevents)))
        #for stim,iti in zip(sorted(stimevents, key=get_timekey), sorted(itis, key=get_timekey)):
        for trialidx,(stim,iti) in enumerate(zip(sorted(stimevents, key=get_timekey), sorted(post_itis, key=get_timekey))):
            trialnum = trialidx + 1
            # For each trial, store all associated stimulus-bitcode events (including the 1st stim-onset) as a list of
            # display-update events related to that trial:
            current_bitcode_evs = [p for p in sorted(pixelevents, key=get_timekey) if p.time>=stim.time and p.time<=iti.time] # p.time<=iti.time to get bit-code for post-stimulus ITI
            current_bitcode_values = [p.value[-1]['bit_code'] for p in sorted(current_bitcode_evs, key=get_timekey)]
            dynamic_stim_bitcodes.append(current_bitcode_evs)
            bitcodes_by_trial[trialnum] = current_bitcode_values #current_bitcode_evs

        # Roughly calculate how many pixel-clock events there should be. For static images, there should be 1 bitcode-event per trial.
        # For drifting gratings, on a 60Hz monitor, there should be 60-61 bitcode-events per trial.
        if 'grating' in session_info['stimulus']:
            nexpected_pixelevents = (ntrials * (session_info['stimduration']/1E3) * refresh_rate) + ntrials + 1
        else:
            nexpected_pixelevents = (ntrials * (session_info['stimduration']/1E3)) + ntrials + 1
        nbitcode_events = sum([len(tr) for tr in dynamic_stim_bitcodes]) + 1 #len(itis) + 1 # Add an extra ITI for blank before first stimulus

        if not nexpected_pixelevents == nbitcode_events:
            print "Expected %i pixel events, missing %i pevs." % (nexpected_pixelevents, nexpected_pixelevents-nbitcode_events)

        # Create trial struct:
        trial = dict() # dict((i+1, dict()) for i in range(len(stimevents)))
        stimevents = sorted(stimevents, key=get_timekey)
        trialevents = sorted(trialevents, key=get_timekey)
        run_start_time = trialevents[0].time
        for trialidx,(stim,iti) in enumerate(zip(sorted(stimevents, key=get_timekey), sorted(post_itis, key=get_timekey))):
            trialnum = trialidx + 1
            trialname = 'trial%05d' % int(trialnum)

            # blankidx = trialidx*2 + 1
            trial[trialname] = dict()
            trial[trialname]['start_time_ms'] = round(stim.time/1E3)
            trial[trialname]['end_time_ms'] = round((iti.time/1E3 + session_info['ITI']))
            stimtype = stim.value[1]['type']
            stimname = stim.value[1]['name']
            stimrotation = stim.value[1]['rotation']
            if 'grating' in stimtype:
                #ori = stim.value[1]['rotation']
                #sf = round(stim.value[1]['frequency'], 2)
                #stimname = 'grating-ori-%i-sf-%f' % (ori, sf)
                stimpos = [stim.value[1]['xoffset'], stim.value[1]['yoffset']]
                stimsize = (stim.value[1]['width'], stim.value[1]['height'])
                phase = stim.value[1]['current_phase']
                freq = stim.value[1]['frequency']
                speed = stim.value[1]['speed']
                direction = stim.value[1]['direction']
#                stimfile = 'NA'
#                stimhash = 'NA'

                trial[trialname]['stimuli'] = {'stimulus': stimname,
                                              'position': stimpos,
                                              'scale': stimsize,
                                              'type': stimtype,
                                              'rotation': stimrotation,
                                              'phase': phase,
                                              'frequency': freq,
                                              'speed': speed,
                                              'direction': direction}

            else:
                # TODO:  fill this out with the appropriate variable tags for RSVP images
                #stimname = stim.value[1]['name'] #''
                stimpos = (stim.value[1]['pos_x'], stim.value[1]['pos_y']) #''
                stimsize = (stim.value[1]['size_x'], stim.value[1]['size_y'])
#                phase = 0
#                freq = 0
#                speed = 0
#                direction = 0
                stimfile = stim.value[1]['filename']
                stimhash = stim.value[1]['file_hash']

                trial[trialname]['stimuli'] = {'stimulus': stimname,
                                              'position': stimpos,
                                              'scale': stimsize,
                                              'type': stimtype,
                                              'filepath': stimfile,
                                              'filehash': stimhash,
                                              'rotation': stimrotation
                                              }

            #stimtype = stim.value[1]['type']
#            if stimtype == 'image':
#                stimfile = stim.value[1]['filename']
#                stimhash = stim.value[1]['file_hash']
#                phase = 0
#                freq = 0
#                speed = 0
#                direction = 0
#            else:
#                stimfile = 'NA'
#                stimhash = 'NA'
#
#            trial[trialname]['stimuli'] = {'stimulus': stimname,
#                                          'position': stimpos,
#                                          'scale': stimsize,
#                                          'type': stimtype,
#                                          'filepath': stimfile,
#                                          'filehash': stimhash,
#                                          'rotation': stimrotation,
#                                          'phase': phase,
#                                          'frequency': freq,
#                                          'speed': speed,
#                                          'direction': direction}


            trial[trialname]['stim_on_times'] = round((stim.time - run_start_time)/1E3)
            trial[trialname]['stim_off_times'] = round((iti.time - run_start_time)/1E3)
            trial[trialname]['all_bitcodes'] = bitcodes_by_trial[trialnum]
            #if stim.value[-1]['name']=='pixel clock':
            trial[trialname]['stim_bitcode'] = stim.value[-1]['bit_code']
            trial[trialname]['iti_bitcode'] = iti.value[-1]['bit_code']
            trial[trialname]['iti_duration'] = session_info['ITI']
            trial[trialname]['block_idx'] = [tidx for tidx, tval in enumerate(trigger_times) if stim.time > tval[0] and stim.time <= tval[1]][0]

    # -------------------------------------------------------------------------
    # Do some checks:
    # -------------------------------------------------------------------------

    print "Each TIFF duration is about (sec): "
    for idx,t in enumerate(trigger_times):
        print idx, (t[1] - t[0])/1E6

	# Check stimulus durations:
	#print len(stimevents)
	iti_events = trialevents[2::2]
	#print len(iti_events)

	stim_durs = []
	off_syncs = []
	for idx,(stim,iti) in enumerate(zip(stimevents, iti_events)):
	    stim_durs.append(iti.time - stim.time)
	    if (iti.time - stim.time)<0:
	        off_syncs.append(idx)
	print "%i bad sync-ing between stim-onsets and ITIs." % len(off_syncs)

	# PLOT stim durations:
	print "N stim ONs:", len(stim_durs)
	print "min stim dur (s):", min(stim_durs)/1E6
	print "max stim dur(s):", max(stim_durs)/1E6



    # TODO:  RETINOBAR-specific extraction....
    # -------------------------------------------------------------------------
    # Create "pydict" to store all MW stimulus/trial info in matlab-accessible format for GUI:

    if retinobar is True:
        pydict = dict()
        print "Offset between first MW stimulus-display-update event and first SI frame-trigger:"
        for ridx,run in enumerate(stimevents.keys()):
            pydict[run] ={'time': [i[0] for i in stimevents[run].states],
                          'pos': stimevents[run].vals,
                          'idxs': stimevents[run].idxs,
                          'ordernum': stimevents[run].ordernum,
                          'MWdur': (stimevents[run].states[-1][0] - stimevents[run].states[0][0]) / 1E6,
                          'offset': stimevents[run].states[0][0] - stimevents[run].triggers[0],
                          'MWtriggertimes': stimevents[run].triggers}
            print "run %i: %s ms" % (ridx+1, str(pydict[run]['offset']/1E3))


    return trial

#%%

def save_trials(trial, paradigm_outdir, curr_dfn_base):

    #% save trial info as pkl for easyloading:
#    trialinfo_fn = 'trial_info_%s.pkl' % curr_dfn_base
#    #with open(os.path.join(data_dir, 'mw_data', trialinfo_fn), 'wb') as f:
#    with open(os.path.join(paradigm_outdir, trialinfo_fn), 'wb') as f:
#        pkl.dump(trial, f, protocol=pkl.HIGHEST_PROTOCOL)
#        f.close()

    # also save as json for easy reading:
    trialinfo_json = 'parsed_trials_%s.json' % curr_dfn_base
    with open(os.path.join(paradigm_outdir, trialinfo_json), 'w') as f:
        json.dump(trial, f, sort_keys=True, indent=4)


def create_stimorder_files(sourcepath):
    '''
    Loads trial dicts from .json files in sourcepath, saves a text file with stimulus order to same dir.
    '''

    trialfn_paths = sorted([os.path.join(sourcepath, j) for j in os.listdir(sourcepath) if j.endswith('json')], key=natural_keys)

    for ti, tfn in enumerate(sorted(trialfn_paths, key=natural_keys)):
        currfile = 'File%03d' % int(ti+1)

        with open(tfn, 'r') as f:
            trials = json.load(f)

        stimorder = [trials[t]['stimuli']['stimulus'] for t in sorted(trials.keys(), key=natural_keys)]

        with open(os.path.join(sourcepath, 'stimorder_%s.txt' % currfile),'w') as f:
            f.write('\n'.join([str(n) for n in stimorder])+'\n')

# In[5]:

def parse_mw_trials(options):

    parser = optparse.OptionParser()

    # PATH opts:
    parser.add_option('-D', '--root', action='store', dest='rootdir', default='/nas/volume1/2photon/data', help='data root dir (root project dir containing all animalids) [default: /nas/volume1/2photon/data, /n/coxfs01/2pdata if --slurm]')
    parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')

    # Set specific session/run for current animal:
    parser.add_option('-S', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID')
    parser.add_option('-A', '--acq', action='store', dest='acquisition', default='FOV1', help="acquisition folder (ex: 'FOV1_zoom3x') [default: FOV1]")
    parser.add_option('-R', '--run', action='store', dest='run', default='', help="name of run dir containing tiffs to be processed (ex: gratings_phasemod_run1)")

    parser.add_option('--slurm', action='store_true', dest='slurm', default=False, help="set if running as SLURM job on Odyssey")

    parser.add_option('--retinobar', action="store_true",
                      dest="retinobar", default=False, help="Set flag if using moving-bar stimulus.")
    parser.add_option('--phasemod', action="store_true",
                      dest="phasemod", default=False, help="Set flag if using dynamic, phase-modulated grating stimulus.")
    parser.add_option('--verbose', action="store_true",
                      dest="verbose", default=False, help="Set flag if want to print all output (for debugging).")

    parser.add_option('-t', '--triggervar', action="store",
                      dest="frametrigger_varname", default='frame_trigger', help="Temp way of dealing with multiple trigger variable names [default: frame_trigger]")


    (options, args) = parser.parse_args(options)

    trigger_varname = options.frametrigger_varname
    rootdir = options.rootdir
    animalid = options.animalid
    session = options.session
    acquisition = options.acquisition
    run = options.run
    retinobar = options.retinobar #'grating'
    phasemod = options.phasemod
    verbose = options.verbose

    slurm = options.slurm
    if slurm is True and 'coxfs01' not in rootdir:
        rootdir = '/n/coxfs01/2p-data'

    # Set paths:
    run_dir = os.path.join(rootdir, animalid, session, acquisition, run)
    raw_foldername = [r for r in os.listdir(run_dir) if 'raw' in r and os.path.isdir(os.path.join(run_dir, r))][0]
    paradigm_indir = os.path.join(run_dir, raw_foldername, 'paradigm_files')

    paradigm_outdir = os.path.join(run_dir, 'paradigm', 'files')
    if not os.path.exists(paradigm_outdir):
        os.makedirs(paradigm_outdir)


    # Get raw MW files:
    raw_files = [f for f in os.listdir(paradigm_indir) if 'mwk' in f or 'serial' in f]
    mw_dfns = sorted([os.path.join(paradigm_indir, m) for m in raw_files if m.endswith('.mwk')], key=natural_keys)

    # TODO:  adjust path-setting to allow for multiple reps of the same experiment
    if verbose is True:
        print "MW files: ", mw_dfns

    # Get MW events
    for didx in range(len(mw_dfns)):

        curr_dfn = mw_dfns[didx]
        curr_dfn_base = os.path.split(curr_dfn)[1][:-4]
        print "Current file: ", curr_dfn

        trials = extract_trials(curr_dfn, retinobar=retinobar, phasemod=phasemod, trigger_varname=trigger_varname, verbose=verbose)

        save_trials(trials, paradigm_outdir, curr_dfn_base)


    print "Finished creating parsed trials. JSONS saved to:", paradigm_outdir

    return paradigm_outdir


def main(options):
    parse_mw_trials(options)


if __name__ == '__main__':
    main(sys.argv[1:])


	# In[16]:


#    if parse_trials is True:
#        mw_codes_by_file = []
#        mw_times_by_file = []
#        mw_trials_by_file = []
#        offsets_by_file = []
#        runs = dict()
#        for idx,triggers in enumerate(trigger_times):
#            curr_trialevents = [i for i in trialevents if i.time<=triggers[1] and i.time>=triggers[0]]
#
#            # Get TIME for each stimulus trial start:
#            mw_times = np.array([i.time for i in curr_trialevents])
#            print "first 10 mw t-intervals:", (mw_times[0:11]-mw_times[0])/1E6
#            # Get ID of each stimulus:
#            mw_codes = []
#            for i in curr_trialevents:
#                if len(i.value)>2: # contains image stim
#                    if stimtype=='grating':
#                        stim_config = (i.value[1]['rotation'], round(i.value[1]['frequency'],1))
#                        stim_idx = [gidx for gidx,grating in enumerate(sorted(image_ids)) if grating==stim_config][0]+1
#                    else: # static image
#                        #TODO:
#                        # get appropriate stim_idx that is the equivalent of unique stim identity for images
#                        stim_idx = [idx for idx,im in enumerate(sorted(image_ids)) if im==i.value[1]['name']][0]+1
#                        #pass
#                else:
#                    stim_idx = 0
#                mw_codes.append(stim_idx)
#            mw_codes = np.array(mw_codes)
#            print "First 10 mw_codes:", mw_codes[0:11]
#
#            # Append list of stimulus times and IDs for each SI file:
#            mw_times_by_file.append(mw_times)
#            mw_codes_by_file.append(mw_codes)
#            mw_trials_by_file.append(curr_trialevents)
#
#            # Calculate offset between first stimulus-update event and first SI-frame trigger:
#            curr_offset = mw_times[0] - triggers[0] # time diff b/w detected frame trigger and stim display
#            offsets_by_file.append(curr_offset)
#
#            # Make silly dict() to keep organization consistent between event-based experiments,
#            # where each SI file contains multple discrete trials, versus movie-stimuli experiments,
#            # where each SI file contains one "trial" for that movie condition:
#            if stimtype != 'bar':
#                rkey = 'run'+str(idx)
#                runs[rkey] = idx #dict() #i
#
#        print "Average offset between stim update event and frame triggger is: ~%0.2f ms" % float(np.mean(offsets_by_file)/1000.)
#
#    mw_file_durs = [i[1]-i[0] for i in trigger_times]
#
#
#    # In[17]:
#
#
#    # Rearrange dicts to match retino structures:
#    pydict = dict()
#
#    # Create "pydict" to store all MW stimulus/trial info in matlab-accessible format for GUI:
#    if stimtype=='bar':
#        print "Offset between first MW stimulus-display-update event and first SI frame-trigger:"
#        for ridx,run in enumerate(stimevents.keys()):
#            pydict[run] ={'time': [i[0] for i in stimevents[run].states],                    'pos': stimevents[run].vals,                    'idxs': stimevents[run].idxs,                    'ordernum': stimevents[run].ordernum,                    'MWdur': (stimevents[run].states[-1][0] - stimevents[run].states[0][0]) / 1E6,                    'offset': stimevents[run].states[0][0] - stimevents[run].triggers[0],                    'MWtriggertimes': stimevents[run].triggers}
#            print "run %i: %s ms" % (ridx+1, str(pydict[run]['offset']/1E3))
#
#    else:
#        for ridx,run in enumerate(runs.keys()):
#            pydict[run] = {'time': mw_times_by_file[ridx],                        'ordernum': runs[run],
#                            'offset': offsets_by_file[ridx],
#                            'stimIDs': mw_codes_by_file[ridx],
#                            'MWdur': mw_file_durs[ridx],\
#                            'MWtriggertimes': trigger_times[ridx]}
#
#            if stimtype=='grating':
#                pydict[run]['idxs'] = [i for i,tmptev in enumerate(mw_trials_by_file[ridx]) if any(['gabor' in v['name'] for v in tmptev.value])],
#            else:
#                pydict[run]['idxs'] = [i for i,tmptev in enumerate(mw_trials_by_file[ridx]) if any([v['name'] in image_ids for v in tmptev.value])],
#
#
#    # In[18]:
#
#    # pydict.keys()
#
#
#    # In[19]:
#
#
#    # Ignoring ARDUINO stuff for now:
#    pydict['stimtype'] = stimtype
#
#    #pydict['mw_times_by_file'] = mw_times_by_file
#    #pydict['mw_file_durs'] = mw_file_durs
#    #pydict['mw_frame_trigger_times'] = frame_trigger_times
#    #pydict['offsets_by_file'] = offsets_by_file
#    #pydict['mw_codes_by_file'] = mw_codes_by_file
#    pydict['mw_dfn'] = mw_dfn
#    pydict['source_dir'] = paradigm_dir #source_dir
#    pydict['fn_base'] = curr_dfn_base #fn_base
#    pydict['stimtype'] = stimtype
#    if stimtype=='bar':
#        pydict['condtypes'] = ['left', 'right', 'top', 'bottom']
#        pydict['runs'] = stimevents.keys()
#        pydict['info'] = session_info
#    elif stimtype=='image':
#        pydict['condtypes'] = sorted(image_ids)
#        pydict['runs'] = runs.keys()
#    elif stimtype=='grating':
#        pydict['condtypes'] = sorted(image_ids)
#        pydict['runs'] = runs.keys()
#
#    tif_fn = curr_dfn_base+'.mat' #fn_base+'.mat'
#    print tif_fn
#    # scipy.io.savemat(os.path.join(source_dir, condition, tif_fn), mdict=pydict)
#    #scipy.io.savemat(os.path.join(data_dir, 'mw_data', tif_fn), mdict=pydict)
#    #print os.path.join(data_dir, 'mw_data', tif_fn)
#    scipy.io.savemat(os.path.join(paradigm_dir, tif_fn), mdict=pydict)
#    print os.path.join(paradigm_dir, tif_fn)
#
#    # Save json:
#    pydict_json = curr_dfn_base+'.json'
#    with open(os.path.join(paradigm_dir, pydict_json), 'w') as f:
#        dump(pydict, f, indent=4)
