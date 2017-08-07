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


prepend = '/Users/julianarhee'
source_dir = os.path.join(prepend,'nas/volume1/2photon/RESDATA/20170724_CE051/retinotopy1') #options.source_dir #'/nas/volume1/2photon/RESDATA/TEFO/20160118_AG33/fov1_gratings1'
stimtype = 'bar' #options.stimtype #'grating'
mask = False #options.mask # False
long_trials = False #options.long_trials #True
no_ard = False #options.no_ard

# Look in child dir (of source_dir) to find mw_data paths:
mw_files = os.listdir(os.path.join(source_dir, 'mw_data'))
mw_files = [m for m in mw_files if m.endswith('.mwk')]


# In[6]:


mwfile = mw_files[0]

fn_base = mwfile[:-4]
mw_data_dir = os.path.join(source_dir, 'mw_data')
mw_fn = fn_base+'.mwk'
dfn = os.path.join(mw_data_dir, mw_fn)
dfns = [dfn]

print "MW file: ", dfns


# In[7]:


# Parse pixel, stimulus, trial, and (if relevant) MW trigger events:

# if stimtype=='bar':
#     pevs, runs, trigg_times, info = get_bar_events(dfns, stimtype=stimtype)
# else:
#     pevs, ievs, tevs, trigg_times, info = get_stimulus_events(dfns, stimtype=stimtype)
    


# In[8]:

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
        if modes[r].time < end_ev.time:  # Ignore any extra "run" events if there was no actual "stop" event
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


def get_bar_events(dfns, remove_orphans=True):
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
    for dfn in dfns:

        df, bounds = get_session_bounds(dfn):

        # df = None
        # df = pymworks.open(dfn)                                                          # Open the datafile

        # # First, find experiment start, stop, or pause events:
        # modes = df.get_events('#state_system_mode')                                      # Find timestamps for run-time start and end (2=run)
        # start_ev = [i for i in modes if i['value']==2][0]                                # 2=running, 0 or 1 = stopped

        # run_idxs = [i for i,e in enumerate(modes) if e['time']>start_ev['time']]         # Get all "run states" if more than 1 found

        # end_ev = next(i for i in modes[run_idxs[0]:] if i['value']==0 or i['value']==1)  # Find the first "stop" event after the first "run" event

        # # Create a list of runs using start/stop-event times (so long as "Stop" button was not pressed during acquisition, only 1 chunk of time)
        # bounds = []
        # bounds.append([start_ev.time, end_ev.time])
        # for r in run_idxs[1:]: 
        #     if modes[r].time < end_ev.time:  # Ignore any extra "run" events if there was no actual "stop" event
        #         continue
        #     else:                            # Otherwise, find the next "stop" event if any additional/new "run" events found.
        #         try:
        #             stop_ev = next(i for i in modes[r:] if i['value']==0 or i['value']==1)
        #         except StopIteration:
        #             end_event_name = 'trial_end'
        #             print "NO STOP DETECTED IN STATE MODES. Using alternative timestamp: %s." % end_event_name
        #             stop_ev = df.get_events(end_event_name)[-1]
        #             print stop_ev
        #         bounds.append([modes[r]['time'], stop_ev['time']])

        # bounds[:] = [x for x in bounds if ((x[1]-x[0])/1E6)>1]
        # # print "................................................................"
        # print "****************************************************************"
        # print "Parsing file\n%s... " % dfn
        # print "Found %i start events in session." % len(bounds)
        # print "Bounds: ", bounds
        # for bidx, bound in enumerate(bounds):
        #     print "bound ID:", bidx, (bound[1]-bound[0])/1E6, "sec"
        # print "****************************************************************"

        # Use chunks of MW "run"-states to get all associate events:

        pixelevents = []
        stimevents = dict()
        trialevents = []
        session_info = dict()
        for bidx,boundary in enumerate(bounds):
            bidx = 0
            boundary = bounds[0]
            if (boundary[1] - boundary[0]) < 1000000:
                print "Not a real boundary, only %i seconds found. Skipping." % int(boundary[1] - boundary[0])
                #continue

            print "................................................................"
            print "SECTION %i" % bidx
            print "................................................................"

            trigger_times = get_trigger_times(df, boundary)
            pixelclock_evs = get_pixelclock_events(df, boundary, trigger_times=trigger_times)
        #     # deal with inconsistent trigger-naming:
        #     codec_list = df.get_codec()
        #     trigger_names = [i for i in codec_list.values() if ('trigger' in i or 'Trigger' in i) and 'flag' not in i]
        #     if len(trigger_names) > 1:
        #         print "Found > 1 name for frame-trigger:"
        #         print "Choose: ", trigger_names
        #         print "Hint: RSVP could be FrameTrigger, otherwise frame_trigger."
        #         trigg_var_name = raw_input("Type var name to use: ")
        #         trigg_evs = df.get_events(trigg_var_name)
        #     else:
        #         trigg_evs = df.get_events(trigger_names[0])

        #     # Only include SI trigger events if they were acquired while MW was actually "running" (i.e., start/stop time boundaries):
        #     trigg_evs = [t for t in trigg_evs if t.time >= boundary[0] and t.time <= boundary[1]]
        #     #print trigg_evs

        #     getout=0
        #     while getout==0:
        #         # Find all trigger LOW events after the first onset trigger (frame_trigger=0 when SI frame-trigger is high, =1 otherwise)
        #         tmp_first_trigger_idx = [i for i,e in enumerate(trigg_evs) if e.value==0][0]        # Find 1st SI frame trigger received by MW (should be "0")
        #         first_off_ev = next(i for i in trigg_evs[tmp_first_trigger_idx:] if i['value']==1)  # Find the next "frame-off" event from SI (i.e., when MW is waiting for the next DI trigger)
        #         first_off_idx = [i.time for i in trigg_evs].index(first_off_ev.time)                # Get corresponding timestamp for first SI frame-off event

        #         # NOTE:  In previous versions of MW protocols, frame-trigger incorrectly reassigned on/off values...
        #         # Make sure only 1 OFF event for each ON event, and vice versa.
        #         # Should abort to examine trigger values and tstamps, but likely, will want to take the "frame ON" event immediately before the found "frame OFF" event.
        #         # (This is because previously, we didn't realize MW's receipt of DI from SI was actually "0" (and frame_trigger was  being used as a flag to turn HIGH, i.e., 1, if trigger received from SI))
        #         if not trigg_evs[first_off_idx-1].time==trigg_evs[tmp_first_trigger_idx].time:
        #             print "Incorrect sequence of frame-triggers detected in MW trigger events received from SI:"
        #             trigg_evs

        #             # Let USER decide what to do next:
        #             print "Press <q> to quit and examine. Press <ENTER> to just use frame-ON idx immediately before found frame-OFF idx: "
        #             user_choice = raw_input()
        #             valid_response = 0
        #             while not valid_response:
        #                 if user_choice=='':
        #                     print "Moving on..."
        #                     do_quickndirty = True
        #                     valid_response = 1
        #                 elif user_choice=='q':
        #                     print "quitting..."
        #                     do_quickndirty = False
        #                     valid_response = 1
        #                 else:
        #                     "Invalid entry provided. Try again."
        #                     user_choice = raw_input()
        #             if do_quickndirty:
        #                 first_on_idx = first_off_idx - 1
        #                 first_on_ev = trigg_evs[first_on_idx]
        #             else:
        #                 print "ABORTING!"
        #                 getout=1
        #         else:
        #             first_on_idx = first_off_idx -1
        #             first_on_ev = trigg_evs[first_on_idx]
        #             getout=1

        #     #print "first_on_idx: ", first_on_idx
        #     print "first on event: ", first_on_ev
        #     #print "first_off_idx: ", first_off_idx
        #     print "first off event: ", first_off_ev
        #     print "Duration of first run: {0:.4f} sec.".format((first_off_ev.time - first_on_ev.time)/1E6)

        #     # Now, get all "trigger" boundaries that demarcate each "run" after first run:

        #     found_trigger_evs = [[first_on_ev, first_off_ev]] # placeholder for off ev
        #     start_idx = copy.copy(first_off_idx)
        #     print trigg_evs
        #     while start_idx < len(trigg_evs)-1: 
        #         print start_idx
        #         try:
        #             found_new_start = False
        #             early_abort = False
        #             curr_chunk = trigg_evs[start_idx+1:] # Look for next OFF event after last off event 

        #             # try:
        #             curr_off_ev = next(i for i in curr_chunk if i['value']==1)
        #             curr_off_idx = [i.time for i in trigg_evs].index(curr_off_ev.time)
        #             curr_start_idx = curr_off_idx - 1  # next "frame-start" should be immediately before next found "frame-off" event
        #             curr_start_ev = trigg_evs[curr_start_idx]
        #             # if trigg_evs[curr_start_idx]['value']!=0:
        #             # # i.e., if prev-found ev with value=1 is not a true frame-on trigger (just a repeated event with same value), just ignore it.
        #             #     continue
        #             # else:
        #             found_new_start = True
        #             # except IndexError:
        #             #     break

        #             last_found_idx = [i.time for i in trigg_evs].index(curr_off_ev.time)
        #             found_trigger_evs.append([curr_start_ev, curr_off_ev])
        #             start_idx = last_found_idx #start_idx + found_idx
        #             print start_idx
        #         except StopIteration:
        #             print "Got to STOP."
        #             if found_new_start is True:
        #                 early_abort = True
        #             break
            
        #     # If no proper off-event found for a given start event (remember, we always look for the next OFF event), just use the end of the session as t-end.
        #     # Since we look for the next OFF event (rather than the next start), if we break out of the loop, we haven't cycled through all the trigg_evs.
        #     # This likely means that there is another frame-ON event, but not corresponding OFF event.
        #     if early_abort is True: 
        #         if found_new_start is True:
        #             found_trigger_evs.append([curr_chunk[curr_idx], end_ev])
        #         else:
        #             found_trigger_evs[-1][1] = end_ev


        #     trigger_evs = [t for t in found_trigger_evs if (t[1].time - t[0].time) > 1]
        #     trigger_times = [[t[0].time, t[1].time] for t in trigger_evs]
        #     print "........................................................................................"
        #     print "Found %i chunks from frame-on/-off triggers:" % len(trigger_times)
        #     print "........................................................................................"
        #     for tidx,trigger in enumerate(trigger_times):
        #         print tidx, ": ", (trigger[1]-trigger[0])/1E6
        #     print "........................................................................................"
        #     runs_selected = 0
        #     while not runs_selected:
        #         user_run_selection = input("Select indices [EX: 0,1,2,4] of runs to include, or press <enter> to accept all:\n")
        #         print "Selected %i runs." % len(user_run_selection)
        #         if any([i>= len(trigger_times) for i in user_run_selection]):
        #             print "Bad index selected, try again."
        #             continue
        #         else:
        #             confirm_selection = raw_input("Press <enter> to accept. Press 'r' to re-try.")
        #             if confirm_selection=='':
        #                 runs_selected = 1
        #             else:
        #                 continue
                
        #     trigger_times = [trigger_times[i] for i in user_run_selection]

            # # Get pixel-clock events:
            # tmp_display_evs = df.get_events('#stimDisplayUpdate')                                                  # Get all stimulus-display-update events
            # display_evs = [e for e in tmp_display_evs if e.value and not e.value[0]==None]                         # Filter out empty display-update events
            # display_evs = [d for d in display_evs if d.time <= boundary[1] and d.time >= boundary[0]]              # Only include display-update events within time boundary of the session

            # tmp_pixelclock_evs = [i for i in display_evs for v in i.value if 'bit_code' in v.keys()]                      # Filter out any display-update events without a pixel-clock event
            # print "N pix-evs found in boundary: %i" % len(tmp_pixelclock_evs)

            # pixelclock_evs = [p for p in tmp_pixelclock_evs if p.time <= trigger_times[-1][1] and p.time >= trigger_times[0][0]] # Make sure pixel events are within trigger times...
            # print "Got %i pix code events for current session." % len(pixelclock_evs)
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
                bar_evs_by_run.append([b for b in bar_update_evs if b.time <= trigger_times[run_idx][-1] and b.time >= trigger_times[run_idx][0]])
                
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
                stimevents[curr_run].triggers = trigger_times[ridx] 
                order_in_session += 1


        session_info['ncycles'] = ncycles
        session_info['target_freq'] = df.get_events('cyc_per_sec')[-1].value
        session_info['barwidth'] = df.get_events('bar_size_deg')[-1].value

        # pdev_info = [(v['bit_code'], p.time) for p in pdevs for v in p.value if 'bit_code' in v.keys()]
        return pixelevents, stimevents, trigger_times, session_info

