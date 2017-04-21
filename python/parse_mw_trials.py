#!/usr/bin/env python2

import numpy as np
import os
from skimage.measure import block_reduce
from scipy.misc import imread
import cPickle as pkl
import scipy.signal
import numpy.fft as fft
import sys
import optparse
from libtiff import TIFF
from PIL import Image
import re
import itertools
from scipy import ndimage

import time
import datetime

import tifffile as tiff
import pandas as pd
import numpy.fft as fft

from bokeh.io import gridplot, output_file, show
from bokeh.plotting import figure
import csv

import pymworks 
import pandas as pd
import operator
import codecs

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import scipy.io
import copy

# User conda env:  retinodev

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

# class positions(Struct):
#     time = 0
#     xpos = 0
#     ypos = 0
#     bitcode = 0

class cycstruct(Struct):
    times = []
    idxs = 0
    vals = 0
    ordernum = 0
    triggers = 0



def get_bar_events(dfns, remove_orphans=True, stimtype='image'):
    """
    Parse session .mwk files.
    Key is session name values are lists of dicts for each trial in session.
    Looks for all response and display events that occur within session.

    dfns : list of strings
        contains paths to each .mwk file to be parsed
    
    remove_orphans : boolean
        for each response event, best matching display update event
        set this to 'True' to remove display events with unknown outcome events
    """

    #trialdata = {}                                                              # initiate output dict
    
    for dfn in dfns:
        df = None
        df = pymworks.open(dfn)                                                 # open the datafile

        #sname = os.path.split(dfn)[1]
        #trialdata[sname] = []

        modes = df.get_events('#state_system_mode')                             # find timestamps for run-time start and end (2=run)
        # run_idxs = np.where(np.diff([i['time'] for i in modes])<20)             # 20 is kind of arbitray, but mode is updated twice for "run"
        start_ev = [i for i in modes if i['value']==2][0]
        # last_ev = [i for i in modes if i['time'] > start_ev['time'] and i['value']==1][0]

        # stop_ev_ev = [i for i in modes if i['time']>start_ev['time'] and (i['value']==0 or i['value']==1)]
        run_idxs = [i for i,e in enumerate(modes) if e['time']>start_ev['time']]

        # Find first stop event after first run event:
        end_ev = next(i for i in modes[run_idxs[0]:] if i['value']==0 or i['value']==1)
        bounds = []
        bounds.append([start_ev.time, end_ev.time])

        # Check for any other start-stop events in session:
        for r in run_idxs[1:]: #[0]:
            if modes[r].time < end_ev.time:
                continue
            else:
                try:
                    # stop_ev = next(i for i in modes[r:] if i['value']==0 or i['value']==1)
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
        print "****************************************************************"

        P = []
        I = dict()
        POS = dict()
        for bidx,boundary in enumerate(bounds):
            if (boundary[1] - boundary[0]) < 1000000:
                print "Not a real boundary, only %i seconds found. Skipping." % int(boundary[1] - boundary[0])
                continue

            print "................................................................"
            print "SECTION %i" % bidx
            print "................................................................"

            # deal with inconsistent trigger-naming:
            codec_list = df.get_codec()
            trigger_names = [i for i in codec_list.values() if ('trigger' in i or 'Trigger' in i) and 'flag' not in i]
                # trigg_evs = df.get_events('frame_triggered')
                # trigg_evs = df.get_events('FrameTrigger')
                # trigg_evs = df.get_events('frame_trigger')
            if len(trigger_names) > 1:
                print "Found > 1 name for frame-trigger:"
                print "Choose: ", trigger_names
                print "Hint: RSVP could be FrameTrigger, otherwise frame_trigger."
                trigg_var_name = raw_input("Type var name to use: ")
                trigg_evs = df.get_events(trigg_var_name)
            else:
                trigg_evs = df.get_events(trigger_names[0])

            trigg_evs = [t for t in trigg_evs if t.time >= boundary[0] and t.time <= boundary[1]]
            # trigg_indices = np.where(np.diff([t.value for t in trigg_evs]) == 1)[0] # when trigger goes from 0 --> 1, start --> end
            # trigg_times = [[trigg_evs[i], trigg_evs[i+1]] for i in trigg_indices]

            # Find all trigger LOW events after the first onset trigger:
            tmp_first_trigger = [i for i,e in enumerate(trigg_evs) if e.value==0][0] # First find 1st set assignment to DI 
            first_off_ev = next(i for i in trigg_evs[tmp_first_trigger:] if i['value']==1) # Since 1=frame OFF, first find this one
            first_off_idx = np.where([i.time==first_off_ev.time for i in trigg_evs])[0]
            first_trigger_idx = first_off_idx - 1 # since prev versions of protocol incorrectly re-assign trigger variable, sometimes there are repeats of "0" (which means frame-start, and shoudl not have been re-assigned in protocol -- this is fixed in later versions)
            first_trigger_ev = trigg_evs[first_trigger_idx]

            curr_idx = copy.copy(first_off_idx)

            trigg_times = [[first_trigger_ev, first_off_ev]] # placeholder for off ev
            start_idx = copy.copy(curr_idx)
            while start_idx < len(trigg_evs)-1: 
                try:
                    found_new_start = False
                    early_abort = False
                    curr_chunk = trigg_evs[start_idx+1:] # Look for next OFF event after last off event 
      
                    try:
                        
                        #curr_idx = [i for i,e in enumerate(curr_chunk) if e.value==1][0] # Find first DI high.
                        curr_off_ev = next(i for i in curr_chunk if i['value']==1)
                        curr_off_idx = [i.time for i in trigg_evs].index(curr_off_ev.time)
                        curr_start_idx = curr_off_idx - 1
                        curr_start_ev = trigg_evs[curr_start_idx]
                        if trigg_evs[curr_start_idx]['value']!=0:
                        # i.e., if next-found ev with value=1 is not a true frame-off trigger (just a repeated event with same value)
                            #early_abort = True
                            break

                        found_new_start = True
                    except IndexError:
                        break

                    #off_ev = next(i for i in curr_chunk[curr_idx:] if i['value']==1) # Find next DI low.
                    #found_idx = [i.time for i in trigg_evs].index(off_ev.time) # Get index of DI low.
                    last_found_idx = [i.time for i in trigg_evs].index(curr_off_ev.time)
                    #trigg_times[-1][1] = trigg_evs[prev_off_idx]
                    #trigg_times.append([curr_chunk[curr_idx], 0])
                    trigg_times.append([curr_start_ev, curr_off_ev])
                    start_idx = last_found_idx #start_idx + found_idx
                    print start_idx
                except StopIteration:
                    print "Got to STOP."
                    if found_new_start is True:
                        early_abort = True
                        #trigg_times.append([curr_chunk[curr_idx], end_ev])
                    break
            if early_abort is True:
                if found_new_start is True:
                    trigg_times.append([curr_chunk[curr_idx], end_ev])
                else:
                    trigg_times[-1][1] = end_ev


            trigg_times = [t for t in trigg_times if (t[1].time - t[0].time) > 1]


 
#            first_trigger = [i for i,e in enumerate(trigg_evs) if e.value==0][0]
#            curr_idx = copy.copy(first_trigger)
#
#            trigg_times = []
#            start_idx = curr_idx
#            while start_idx < len(trigg_evs)-1: 
#
#                try:
#                    
#                    curr_chunk = trigg_evs[start_idx:]
#                    try:
#                        curr_idx = [i for i,e in enumerate(curr_chunk) if e.value==0][0] # Find first DI high.
#                    except IndexError:
#                        break
#
#                    stop_ev = next(i for i in curr_chunk[curr_idx:] if i['value']==1) # Find next DI low.
#                    found_idx = [i.time for i in trigg_evs].index(stop_ev.time) # Get index of DI low.
#                    
#                    trigg_times.append([curr_chunk[curr_idx], stop_ev])
#                    start_idx = found_idx #start_idx + found_idx
#                    print start_idx
#
#
#                except StopIteration:
#                    print "Got to STOP."
#                    break
#
#            trigg_times = [t for t in trigg_times if t[1].time - t[0].time > 1]
#
            # Check stimDisplayUpdate events vs announceStimulus:
            stim_evs = df.get_events('#stimDisplayUpdate')
            devs = [e for e in stim_evs if e.value and not e.value[0]==None]
            devs = [d for d in devs if d.time <= boundary[1] and d.time >= boundary[0]]

            tmp_pdevs = [i for i in devs for v in i.value if 'bit_code' in v.keys()]

            # Get rid of "repeat" events from state updates.
            pdevs = [i for i in tmp_pdevs if i.time<= boundary[1] and i.time>=boundary[0]]
            print "N pix-evs found in boundary: %i" % len(pdevs)
            nons = np.where(np.diff([i.value[-1]['bit_code'] for i in pdevs])==0)[0] # pix stim event is always last
            pdevs = [p for i,p in enumerate(pdevs) if i not in nons]

            pdevs = [p for p in pdevs if p.time <= trigg_times[-1][1].time and p['time'] >= trigg_times[0][0].time] # Make sure pixel events are within trigger times...

            pdev_info = [(v['bit_code'], p.time) for p in pdevs for v in p.value if 'bit_code' in v.keys()]

            print "Got %i pix code events." % len(pdev_info)
            P.append(pdev_info)


            idevs = [i for i in pdevs for v in i.value if 'bit_code' in v.keys()]

            if stimtype=='bar':
                # do stuff

                bardevs = [i for i in idevs for v in i.value if '_bar' in v['name']]

                conds = df.get_events('condition')
                conds = [cond for cond in conds if cond.value>=0]
                ncycles = df.get_events('ncycles')[-1].value
                cycnums = df.get_events('cycnum')
                cycends = [i for (i,c) in enumerate(cycnums) if c.value==ncycles+1]
                cycles = []
                sidx = 0
                for cyc in cycends:
                    cyc_chunk = cycnums[sidx:cyc+1]
                    cyc_start = len(cyc_chunk) - [i.value for i in cyc_chunk][::-1].index(1) - 1 # find last occurrence of value1
                    cycles.append(cyc_chunk[cyc_start:cyc+1])
                    sidx = cyc + 1

                bartimes = []
                triggtimes = []
                for cidx,cycle in enumerate(cycles):            
                    bartimes.append([b for b in bardevs if b.time < cycle[-1].time and b.time > trigg_times[cidx][0].time])
                    triggtimes.append([trigg_times[cidx][0].time, trigg_times[cidx][1].time])

                tpositions = []
                for update in bartimes:
                    tpos = [[i.time, (i.value[1]['pos_x'], i.value[1]['pos_y'])] for i in update]
                    tpositions.append(tpos)

                # POS = dict(
                onum = 0
                for ridx,run in enumerate(tpositions):
                    if run[0][1][1]==0: # vertical cond, ypos=0
                        posvec = [i[1][0] for i in run]
                        if posvec[0] < 0: # bar starting on LEFT
                            restarts = list(np.where(np.diff(posvec) < 0)[0] + 1)
                            curr_run = 'left'
                        else:  
                            restarts = list(np.where(np.diff(posvec) > 0)[0] + 1)
                            curr_run = 'right'
                    else: # horizontal cond, xpos = 0
                        posvec = [i[1][1] for i in run] 
                        if posvec[0] < 0: # bar is starting at BOTTOM
                            restarts = list(np.where(np.diff(posvec) < 0)[0] + 1)
                            curr_run = 'bottom'
                        else:
                            restarts = list(np.where(np.diff(posvec) > 0)[0] + 1)
                            curr_run = 'top'

                    if curr_run in POS.keys():
                        ncond_rep = len([i for i in POS.keys() if i==curr_run])
                        curr_run = curr_run + '_' + str(ncond_rep+1)

                    POS[curr_run] = cycstruct()
                    POS[curr_run].times = run
                    restarts.append(0)
                    POS[curr_run].idxs = sorted(restarts)
                    POS[curr_run].vals = posvec
                    POS[curr_run].ordernum = onum
                    POS[curr_run].triggers = triggtimes[ridx] 
                    onum += 1


                I['ncycles'] = ncycles
                I['target_freq'] = df.get_events('cyc_per_sec')[-1].value
                I['barwidth'] = df.get_events('bar_size_deg')[-1].value
                #pix_evs = df.get_events('#pixelClockCode')
        return P, POS, trigg_times, I 



def get_stimulus_events(dfns, remove_orphans=True, stimtype='image'):
    """
    Parse session .mwk files.
    Key is session name values are lists of dicts for each trial in session.
    Looks for all response and display events that occur within session.

    dfns : list of strings
        contains paths to each .mwk file to be parsed
    
    remove_orphans : boolean
        for each response event, best matching display update event
        set this to 'True' to remove display events with unknown outcome events
    """
    
    for dfn in dfns:
        df = None
        df = pymworks.open(dfn)  # Open datafile.

        # Get state system events, and identify start/stop points:
        # 0 or 1 = not running
        # 2 = running
        modes = df.get_events('#state_system_mode') 
        start_ev = [i for i in modes if i['value']==2][0]
        run_idxs = [i for i,e in enumerate(modes) if e['time']>start_ev['time']]

        # Find first stop event after first run event:
        end_ev = next(i for i in modes[run_idxs[0]:] if i['value']==0 or i['value']==1)
        bounds = []
        bounds.append([start_ev.time, end_ev.time])

        # Check for any other start-stop events in session:
        for r in run_idxs[1:]: #[0]:
            if modes[r].time < end_ev.time:
                continue
            else:
                try:
                    # stop_ev = next(i for i in modes[r:] if i['value']==0 or i['value']==1)
                    stop_ev = next(i for i in modes[r:] if i['value']==0 or i['value']==1)
                except StopIteration:
                    end_event_name = 'trial_end'
                    print "NO STOP DETECTED IN STATE MODES. Using alternative timestamp: %s." % end_event_name
                    stop_ev = df.get_events(end_event_name)[-1]
                    print stop_ev
                bounds.append([modes[r]['time'], end_ev['time']])

        bounds[:] = [x for x in bounds if ((x[1]-x[0])/1E6)>1]

        print "****************************************************************"
        print "Parsing file\n%s... " % dfn
        print "Found %i start events in session." % len(bounds)
        print "Bounds: ", bounds
        print "****************************************************************"

        pixelevents = []
        stimevents = []
        trialevents = []
        info = dict()
        for bidx,boundary in enumerate(bounds):
            if (boundary[1] - boundary[0]) < 1000000:
                print "Not a real boundary, only %i seconds found. Skipping." % int(boundary[1] - boundary[0])
                continue

            print "................................................................"
            print "SECTION %i" % bidx
            print "................................................................"

            # Deal with inconsistent trigger-naming:
            codec_list = df.get_codec()
            trigger_names = [i for i in codec_list.values() if ('trigger' in i or 'Trigger' in i) and 'flag' not in i]
                # trigg_evs = df.get_events('frame_triggered')
                # trigg_evs = df.get_events('FrameTrigger')
                # trigg_evs = df.get_events('frame_trigger')
            if len(trigger_names) > 1:
                print "Found > 1 name for frame-trigger:"
                print "Choose: ", trigger_names
                print "Hint: RSVP could be FrameTrigger, otherwise frame_trigger."
                trigg_var_name = raw_input("Type var name to use: ")
                trigg_evs = df.get_events(trigg_var_name)
            else:
                trigg_evs = df.get_events(trigger_names[0])
            trigg_evs = [t for t in trigg_evs if t.time >= boundary[0] and t.time <= boundary[1]]

            # Find all trigger LOW events after the first onset trigger:
            first_trigger = [i for i,e in enumerate(trigg_evs) if e.value==0][0]
            curr_idx = copy.copy(first_trigger)

            tmp_trigg_times = []
            start_idx = curr_idx
            while start_idx < len(trigg_evs)-1: 
                try:
                    found_new_start = False
                    early_abort = False
                    curr_chunk = trigg_evs[start_idx:]
                    try:
                        curr_idx = [i for i,e in enumerate(curr_chunk) if e.value==0][0] # Find first DI high.
                        found_new_start = True
                    except IndexError:
                        break

                    off_ev = next(i for i in curr_chunk[curr_idx:] if i['value']==1) # Find next DI low.
                    found_idx = [i.time for i in trigg_evs].index(off_ev.time) # Get index of DI low.
                    tmp_trigg_times.append([curr_chunk[curr_idx], off_ev])
                    start_idx = found_idx #start_idx + found_idx
                    print start_idx
                except StopIteration:
                    print "Got to STOP."
                    if found_new_start is True:
                        early_abort = True
                        tmp_trigg_times.append([curr_chunk[curr_idx], end_ev])
                    break
            trigg_times = [t for t in tmp_trigg_times if t[1].time - t[0].time > 1]

            # Check stimDisplayUpdate events vs pixel-clock events:
            stim_evs = df.get_events('#stimDisplayUpdate')
            devs = [e for e in stim_evs if e.value and not e.value[0]==None]
            devs = [d for d in devs if d.time <= boundary[1] and d.time >= boundary[0]]

            tmp_pdevs = [i for i in devs for v in i.value if 'bit_code' in v.keys()]

            # Get rid of "repeat" events from state updates.
            pdevs = [i for i in tmp_pdevs if i.time<= boundary[1] and i.time>=boundary[0]]
            print "N pix-evs found in boundary: %i" % len(pdevs)
            i#bitcode_idxs = [i for p in pdevs for i in range(len(p.value) if 'bit_code' in p.value[i].keys()]
            bitcode_idxs = []
            for p in pdevs:
                bitcode_idxs.append([i for i,v in enumerate(p.value) if p.value[i].keys()[0]=='bit_code'])
            #bitcode_idxs = list(itertools.chain.from_iterable(bitcode_idxs))
            bitcode_idxs = [b[0] for b in bitcode_idxs]
            nons = np.where(np.diff([i.value[kn]['bit_code'] for i,kn in zip(pdevs, bitcode_idxs)])==0)[0]
            #nons = np.where(np.diff([i.value[kn]['bit_code'] for kn in range(len(i.value)) if 'bit_code' in i.value[kn].keys() for i in pdevs])==0)[0] # pix stim event is always last
            nons += 1 # to remove actual bad/repeated event
            
            pdevs = [p for i,p in enumerate(pdevs) if i not in nons]
            pdevs = [p for p in pdevs if p.time <= trigg_times[-1][1].time and p['time'] >= trigg_times[0][0].time] # Make sure pixel events are within trigger times...
            pdev_info = [(v['bit_code'], p.time) for p in pdevs for v in p.value if 'bit_code' in v.keys()]
            print "Got %i pix code events." % len(pdev_info)
            pixelevents.append(pdev_info)

            # Get TRIAL-SPECIFIC events:
            idevs = [i for i in pdevs for v in i.value if 'bit_code' in v.keys()]

            if stimtype=='image':
                imdevs = [d for d in devs for i in d.value if 'filename' in i.keys() and '.png' in i['filename']]
                stimevents.append(imdevs)
                
                # Find blank ITIs:
                if mask is True:
                    tdevs = [i for i in idevs for v in i.value if v['name']=='blue_mask' and i not in imdevs]
                else:
                    tdevs = [i for i in idevs if i.time>imdevs[0].time and i not in imdevs]
                
                tmp_trialevents = imdevs + tdevs
                tmp_trialevents = sorted(tmp_trialevents, key=get_timekey)


            if stimtype=='grating':
                imdevs = [d for d in devs for i in d.value if i['name']=='gabor']

                start_times = [i.value[1]['start_time'] for i in imdevs] # Use start_time to ignore dynamic pixel-code of drifting grating since stim as actually static
                find_static = np.where(np.diff(start_times) > 0)[0]
                find_static = np.append(find_static, 0)
                find_static = sorted(find_static)
                imtrials = [imdevs[i+1] for i in find_static]
                stimevents.append(imtrials)

                # Find blank ITIs:
                if mask is True:
                    tdevs = [i for i in idevs if i.time>imdevs[0].time and i not in imdevs]
                else:
                    prevdev = [[i for i,d in enumerate(devs) if d.time < t.time][-1] for t in imtrials[1:]]
                    lastdev = [i for i,d in enumerate(devs) if d.time > imtrials[-1].time and len(d.value)<3][0] # ignore the last "extra" ev (has diff time-stamp) - just wnt 1st non-grating blank
                    tdevs = [devs[i] for i in prevdev]
                    tdevs.append(devs[lastdev])
                
                tmp_trialevents = imtrials + tdevs
                tmp_trialevents = sorted(tmp_trialevents, key=get_timekey)

            # trial_ends = [i for i in df.get_events('Announce_TrialEnd') if i.value==1]
            
            # E.append(trial_ends)
            trialevents.append(tmp_trialevents)


            #pix_evs = df.get_events('#pixelClockCode')
        return pixelevents, stimevents, trialevents, trigg_times


def get_timekey(item):
    return item.time


import optparse

parser = optparse.OptionParser()
# parser.add_option('--fn', action="store", dest="fn_base",
#                   default="", help="shared filename for ard and mw files")
parser.add_option('--source', action="store", dest="source_dir",
                  default="", help="source (parent) dir containing mw_data and ard_data dirs")
parser.add_option('--stim', action="store",
                  dest="stimtype", default="grating", help="stimulus type (gratings or rsvp)?")
parser.add_option('--mask', action="store_true",
                  dest="mask", default=False, help="blue mask or no mask [default: False]?")
parser.add_option('--long', action="store_true",
                  dest="long_trials", default=False, help="long (10sec) or short (3sec) trials [default: False]?")
parser.add_option('--noard', action="store_true",
                  dest="no_ard", default=False, help="No arduino triggers saved? [default: False]")


(options, args) = parser.parse_args()

# fn_base = options.fn_base #'20160118_AG33_gratings_fov1_run1'
source_dir = options.source_dir #'/nas/volume1/2photon/RESDATA/TEFO/20160118_AG33/fov1_gratings1'
stimtype = options.stimtype #'grating'
mask = options.mask # False
long_trials = options.long_trials #True

no_ard = options.no_ard

mw_files = os.listdir(os.path.join(source_dir, 'mw_data'))
mw_files = [m for m in mw_files if m.endswith('.mwk')]

for mwfile in mw_files:
    fn_base = mwfile[:-4]

    # --------------------------------------------------------------------
    # MW codes:
    # --------------------------------------------------------------------

    mw_data_dir = os.path.join(source_dir, 'mw_data')
    mw_fn = fn_base+'.mwk'
    dfn = os.path.join(mw_data_dir, mw_fn)
    dfns = [dfn]


    if stimtype=='bar':
        pevs, runs, trigg_times, info = get_bar_events(dfns, stimtype=stimtype)
    else:
        pevs, ievs, tevs, trigg_times = get_stimulus_events(dfns, stimtype=stimtype)

    if len(pevs) > 1:
        pevs = [item for sublist in pevs for item in sublist]
        pevs = list(set(pevs))
        pevs.sort(key=operator.itemgetter(1))
    else:
        pevs = pevs[0]
    print "Found %i pixel clock events." % len(pevs)

    # on FLASAH protocols, first real iamge event is 41
    if stimtype!='bar':
	if len(tevs) > 1:
	    tevs = [item for sublist in tevs for item in sublist]
	    tevs = list(set(tevs))
	    tevs.sort(key=operator.itemgetter(1))
	else:
	    tevs = tevs[0]
        print "Found %i trial events." % len(tevs)



    if stimtype=='bar':
        nexpected_pevs = int(round((1/info['target_freq']) * info['ncycles'] * 60 * len(trigg_times)))
        print "Expected %i pixel events, missing %i pevs." % (nexpected_pevs, nexpected_pevs-len(pevs))
        # on FLASH protocols, first real iamge event is 41
        print "Found %i conditions, corresponding to %i TIFFs." % (len(runs), len(trigg_times))

    else:
        ievs = ievs[0]
        print "Found %i image trials." % len(ievs)
        # if len(ievs) > 1:
        #   ievs = [item for sublist in ievs for item in sublist]
        #   ievs = list(set(ievs))
        #   ievs.sort(key=operator.itemgetter(1))

        print "Found %i stimulus update events across trials." % len(tevs)

        print "Expecting %i TIFFs." % len(trigg_times)

    # May need to fix this:
    print "Each chunk duration is: ", [(t[1].time - t[0].time)/1E6 for t in trigg_times]


    n_codes = set([i[0] for i in pevs])
    if len(n_codes)<16:
        print "Check pixel clock -- missing bit values..."

    if stimtype=='bar':
        pydict = dict()
        for ridx,run in enumerate(runs.keys()):
            #frame_trigger_times = [[i[0].time, i[1].time] for i in trigg_times] # Use frame-trigger times to find dur of each tif file in MW time
            #mw_file_durs = [i[1]-i[0] for i in frame_trigger_times]

            pydict[run] ={'time': [i[0] for i in runs[run].times],\
                        'pos': runs[run].vals,\
                        'idxs': runs[run].idxs,\
                        'ordernum': runs[run].ordernum,\
                        'MWdur': (runs[run].times[-1][0] - runs[run].times[0][0]) / 1E6,\
                        'offset': runs[run].times[0][0] - runs[run].triggers[0],\
                        'MWtriggertimes': runs[run].triggers}
            #'MWdur': float((runs[run].triggers[1]-runs[run].triggers[0])/1E6),\
            print "Offset %i: %s" % (ridx+1, str(pydict[run]['offset']/1E6))
            #pydict['triggers'] = frame_trigger_times
            #pydict['runs'] = POS.keys()
            #pydict['stimtype'] = stimtype

            # Add some other info:
            #pydict['info'] = info

            # tif_fn = fn_base+'.mat'
            # # scipy.io.savemat(os.path.join(source_dir, condition, tif_fn), mdict=pydict)
            # scipy.io.savemat(os.path.join(source_dir, 'mw_data', tif_fn), mdict=pydict)
            # print os.path.join(source_dir, 'mw_data', tif_fn)

    elif stimtype == 'image':
        all_ims = [i.value[1]['name'] for i in ievs]
        image_ids = sorted(list(set(all_ims)))

        mw_times = np.array([i.time for i in tevs]) #np.array([i[0] for i in pevs])
        mw_codes = []
        if mask is True:
            nexpected_imvalues = 4
        else:
            nexpected_imvalues = 3

        for i in tevs:
            if len(i.value)==nexpected_imvalues:
                stim_name = i.value[1]['name']
                stim_idx = [iidx for iidx,image in enumerate(image_ids) if image==stim_name][0]+1
            else:
                stim_idx = 0
            mw_codes.append(stim_idx)
        mw_codes = np.array(mw_codes)

    elif stimtype == 'grating':
        # Need to find all unique grating types:
        all_combos = [(i.value[1]['rotation'], round(i.value[1]['frequency'],1)) for i in ievs]
        image_ids = sorted(list(set(all_combos))) # should be 35 for gratings -- sorted by orientation (7) x SF (5)

        mw_times = np.array([i.time for i in tevs])
        mw_codes = []
        for i in tevs:
            if len(i.value)>2: # contains image stim
                stim_config = (i.value[1]['rotation'], round(i.value[1]['frequency'],1))
                stim_idx = [gidx for gidx,grating in enumerate(sorted(image_ids)) if grating==stim_config][0]+1
            else:
                stim_idx = 0
            mw_codes.append(stim_idx)
        mw_codes = np.array(mw_codes)


    parse_files = False
    if stimtype != 'bar':
        t_mw_intervals = np.diff(mw_times)
        parse_files = True


    if parse_files is True:
        # find_matching_fidxs = np.where(t_mw_intervals > 3500000)[0]
        if long_trials is True:
            #nsecs = 10500000 # 3500000
            nsecs = 10900000 # 3500000
        else:
            nsecs = 3050000 #2100000
        find_matching_fidxs = np.where(t_mw_intervals > nsecs)[0]
        mw_file_idxs = [i+1 for i in find_matching_fidxs]
        mw_file_idxs.append(0)
        mw_file_idxs = np.array(sorted(mw_file_idxs))
        print "Found %i MW file chunks." % len(mw_file_idxs)
        if len(trigg_times) != len(mw_file_idxs):
            badparsing = True
        else:
            badparsing = False

        while badparsing is True:
            if len(trigg_times) < len(mw_file_idxs):
                print "Current interval is: %i" % nsecs
                nsecs = float(raw_input('Enter larger value: '))
            else:
                print "Current interval is: %i" % nsecs
                nsecs = float(raw_input('Enter smaller value: '))
            find_matching_fidxs = np.where(t_mw_intervals > nsecs)[0]
            mw_file_idxs = [i+1 for i in find_matching_fidxs]
            mw_file_idxs.append(0)
            mw_file_idxs = np.array(sorted(mw_file_idxs))
            print "Found %i MW file chunks." % len(mw_file_idxs)
            if len(trigg_times) == len(mw_file_idxs):
                print "GOT IT!"
                badparsing = False

        rel_mw_times = mw_times - mw_times[0] # Divide by 1E6 to get in SEC


        frame_trigger_times = [[i[0].time, i[1].time] for i in trigg_times] # Use frame-trigger times to find dur of each tif file in MW time
        mw_file_durs = [i[1]-i[0] for i in frame_trigger_times]


        mw_codes_by_file = []
        mw_times_by_file = []
        mw_trials_by_file = []
        offsets_by_file = []
        runs = dict()
        for i in range(len(mw_file_idxs)):
            if i==range(len(mw_file_idxs))[-1]:
                curr_mw_times = mw_times[mw_file_idxs[i]:]
                curr_mw_codes = mw_codes[mw_file_idxs[i]:]
                curr_mw_trials = tevs[mw_file_idxs[i]:]
            else:
                curr_mw_times = mw_times[mw_file_idxs[i]:mw_file_idxs[i+1]]
                curr_mw_codes = mw_codes[mw_file_idxs[i]:mw_file_idxs[i+1]]
                curr_mw_trials = tevs[mw_file_idxs[i]:mw_file_idxs[i+1]]

            mw_times_by_file.append(curr_mw_times)
            mw_codes_by_file.append(curr_mw_codes)
            mw_trials_by_file.append(curr_mw_trials)

            curr_offset = curr_mw_times[0] - frame_trigger_times[i][0] # time diff b/w detected frame trigger and stim display
            offsets_by_file.append(curr_offset)
            if stimtype != 'bar':
                rkey = 'run'+str(i)
                runs[rkey] = i #dict() #i



        print "Average offset between stim update event and frame triggger is: ~%0.2f ms" % float(np.mean(offsets_by_file)/1000.)
    # file_no = 2
    # rel_times = list(mw_times_by_file[file_no-1] - mw_times_by_file[file_no-1][0])
    # mw_rel_times = rel_times + offsets_by_file[file_no-1]
    # mw_codes = mw_codes_by_file[file_no-1]


    if no_ard is False:
    # --------------------------------------------------------------------
    # ARDUINO codes:
    # --------------------------------------------------------------------

        ard_fn = fn_base+'.txt'
        ard_data_dir = os.path.join(source_dir, 'ard_data')
        ard_dfn = os.path.join(ard_data_dir, ard_fn)


        f = codecs.open(ard_dfn, 'r')
        data = f.read()
        bad_end = False
        if not data[-1]=='_':
           bad_end = True
        packets = data.split('__')
        nsplits = packets[0].count('*')
        ard_evs = []
        for pidx,packet in enumerate(packets):
            #print pidx
            if pidx==0:
                packet = packet[1:]
            elif pidx==len(packets)-1:
                if bad_end is True:
                  continue
                else:
                  packet = packet[:-1]
            try:
                if nsplits == 2:
                    pin = packet.split('*', 2)[0]
                    tstring = packet.split('*', 2)[2]
                else:
                    pin = packet.split('*', 1)[0]
                    tstring = packet.split('*', 1)[1]
                try:
                    pin_id = int(pin) #get_int(pin) #pin #int(pin)
                    #pin_id = sum(ord(c) << (i * 8) for i, c in enumerate(pin[::-1]))
                    tstamp = int(tstring) #sum(ord(c) << (i * 8) for i, c in enumerate(tstring[::-1]))
                    ard_evs.append([pin_id, tstamp])
                except NameError as e:
                    print pidx
                    print pin_id
                    print tstring

            except IndexError as e:
                print pidx
                print packet

        ard_times = np.array([i[1] for i in ard_evs])
        ard_codes = np.array([i[0] for i in ard_evs])



        # Check for arduino roll-overs:
        tdiffs = np.array(np.diff(ard_times))
        rollover_idxs = np.where(tdiffs<0)
        if any(rollover_idxs):
            print "Found rollover points!  Fix this..."

        onsets = np.where(np.diff([e[0] for e in ard_evs]) > 0)[0]   # where trigger val changes from 0-->1
        onsets = [t+1 for t in onsets]
        offsets = np.where(np.diff([e[0] for e in ard_evs]) < 0)[0]  # where trigger val changes from 1-->0
        offsets = [t+1 for t in offsets]


        if len(onsets) != len(offsets):
            print "N offsets %i does not match N onsets %i." % (len(offsets), len(onsets))
            offsets.append(len(ard_evs)-1)
        else:
            print "N offsets and onsets match! Found %i frame events." % len(onsets)

        no_off_trigger = False
        if ard_evs[-1][0]==1:
            no_off_trigger = True


        tr_start_idx = onsets[0]
        too_short = 700 #500 # if diff b/w offsets is < 100, interval is only 100*1500us = 150ms... 
        on_intervals = np.where(np.diff(onsets) > too_short)[0] #[t for t in np.diff(onsets) if t>too_short]
        #off_intervals = np.where(np.diff(offsets) > too_short)[0]

        new_file_idxs = [onsets[i+1] for i in on_intervals]
        new_file_idxs.append(tr_start_idx)
        new_file_idxs = sorted(new_file_idxs)
        new_file_markers = [ard_evs[i] for i in sorted(new_file_idxs)]

        end_file_idxs = []
        for i,nidx in enumerate(new_file_idxs[0:-1]):
            sublist = ard_evs[nidx:new_file_idxs[i+1]]
            last_off = np.where(np.diff([e[0] for e in sublist]) < 0)[0][-1]
            off_idx = nidx + last_off + 1
            end_file_idxs.append(off_idx)
        if no_off_trigger:
            end_file_idxs.append(len(ard_evs)-1) # just add the last trigger event for end-of-file
        else:
            sublist = ard_evs[onsets[-1]:] # Find all evs from last onset onward.
            final_trigger = np.where(np.diff([e[0] for e in sublist]) < 0)[0][0] # Find first off-trigger
            final_trigger_idx = onsets[-1] + final_trigger + 1 # add 2 to get actual off-trigger event after the 1st <0 difference found
            end_file_idxs.append(final_trigger_idx)

        end_file_markers = [ard_evs[i] for i in sorted(end_file_idxs)]
        print "Found %i TIF file chunks." % len(new_file_idxs)
        if len(new_file_markers)==1:
            print "DUR of session is: %f sec." % ((end_file_markers[0][1] - new_file_markers[0][1])/1E6)
        else:
            for midx in range(len(new_file_markers)):
                print "DUR of chunk is: %0.2f sec." % ((end_file_markers[midx][1] - new_file_markers[midx][1])/1E6)



        # # Plot to check file chunks
        # plt.plot([e[0] for e in evs], 'k*')
        # plt.plot(new_file_idxs, np.ones((1,len(new_file_idxs)))[0], 'g*', markersize=10)
        # plt.plot(end_file_idxs, np.ones((1,len(end_file_idxs)))[0], 'r*', markersize=10)
        # plt.title('Number of TIF files should be: %i' % len(new_file_markers))


        onset_evs = [ard_evs[on] for on in onsets]
        offset_evs = [ard_evs[off] for off in offsets]
        ard_acquisition_evs = []
        ard_frame_onset_times = []
        ard_frame_offset_times = []
        ard_onset_idxs = []
        ard_offset_idxs = []

        ard_file_durs = []
        ard_file_trigger_times = []
        ard_file_trigger_vals = []
        for sidx,eidx in zip(new_file_idxs, end_file_idxs):

            curr_acquisition_evs = ard_evs[sidx:eidx+1]
            ftimes = [e[1] for e in curr_acquisition_evs]

            curr_onset_evs = [ev for ev in onset_evs if ev[1]>=curr_acquisition_evs[0][1] and ev[1]<=curr_acquisition_evs[-1][1]]
            curr_offset_evs = [ev for ev in offset_evs if ev[1]>=curr_acquisition_evs[0][1] and ev[1]<=curr_acquisition_evs[-1][1]]

            curr_onset_times = [e[1] for e in curr_onset_evs]
            curr_offset_times = [e[1] for e in curr_offset_evs]

            curr_onset_idxs = [i for (i,ev) in enumerate(onset_evs) if ev[1]>=curr_acquisition_evs[0][1] and ev[1]<=curr_acquisition_evs[-1][1]]
            curr_offset_idxs = [i for (i,ev) in enumerate(offset_evs) if ev[1]>=curr_acquisition_evs[0][1] and ev[1]<=curr_acquisition_evs[-1][1]]


            frame_trigger_trange = (max(np.diff(curr_onset_times)) - min(np.diff(curr_onset_times))) / 1E3
            print "%i: Range of frame trigger time stamps: %0.2f - %0.2f ms." % (sidx, min(np.diff(curr_onset_times))/1E3, max(np.diff(curr_onset_times))/1E3)

            frame_durs = np.array(curr_offset_times) - np.array(curr_onset_times)

            ard_acquisition_evs.append(curr_acquisition_evs)
            ard_frame_onset_times.append(curr_onset_times)
            ard_frame_offset_times.append(curr_offset_times)
            ard_onset_idxs.append(curr_onset_idxs)
            ard_offset_idxs.append(curr_offset_idxs)


            ard_file_dur = (ard_times[eidx] - ard_times[sidx])/1E6
            ard_file_durs.append(ard_file_dur)

            ard_frame_tval = ard_codes[sidx:eidx+1]
            ard_frame_ttime = ard_times[sidx:eidx+1]
            ard_file_trigger_vals.append(ard_frame_tval)
            ard_file_trigger_times.append(ard_frame_ttime)



    # Rearrange dicts to match retino structures:
    if stimtype == 'image':
        pydict = dict()
        for ridx,run in enumerate(runs.keys()):
            pydict[run] = {'time': mw_times_by_file[ridx],\
                            'ordernum': runs[run], 
                            'idxs': [i for i,tmptev in enumerate(mw_trials_by_file[ridx]) if any([v['name'] in image_ids for v in tmptev.value])],
                            'offset': offsets_by_file[ridx],
                            'stimIDs': mw_codes_by_file[ridx],
                            'MWdur': mw_file_durs[ridx],\
                            'MWtriggertimes': frame_trigger_times[ridx]}
    if stimtype == 'grating':
        pydict = dict()
        for ridx,run in enumerate(runs.keys()):
            pydict[run] = {'time': mw_times_by_file[ridx],\
                            'ordernum': runs[run], 
                            'idxs': [i for i,tmptev in enumerate(mw_trials_by_file[ridx]) if any(['gabor' in v['name'] for v in tmptev.value])],
                            'offset': offsets_by_file[ridx],
                            'stimIDs': mw_codes_by_file[ridx],
                            'MWdur': mw_file_durs[ridx],\
                            'MWtriggertimes': frame_trigger_times[ridx]}
    if no_ard is False:
        for ridx,run in enumerate(runs.keys()):
            pydict[run]['tframe'] = ard_frame_onset_times[ridx]
            pydict[run]['SIdur'] = ard_file_durs[ridx]

            pydict[run]['SItriggertimes'] = ard_file_trigger_times[ridx]
            pydict[run]['SItriggervals'] = ard_file_trigger_vals[ridx]

            pydict['ard_dfn'] = ard_dfn

    pydict['stimtype'] = stimtype

    #pydict['mw_times_by_file'] = mw_times_by_file
    #pydict['mw_file_durs'] = mw_file_durs
    #pydict['mw_frame_trigger_times'] = frame_trigger_times
    #pydict['offsets_by_file'] = offsets_by_file
    #pydict['mw_codes_by_file'] = mw_codes_by_file
    pydict['mw_dfn'] = dfn
    pydict['source_dir'] = source_dir
    pydict['fn_base'] = fn_base
    pydict['stimtype'] = stimtype
    if stimtype=='bar':
        pydict['condtypes'] = ['left', 'right', 'top', 'bottom']
        pydict['runs'] = runs.keys()
        pydict['info'] = info
    elif stimtype=='image':
        pydict['condtypes'] = sorted(image_ids)
        pydict['runs'] = runs.keys()
    elif stimtype=='grating':
        pydict['condtypes'] = sorted(image_ids)
        pydict['runs'] = runs.keys()

    #if no_ard is False:

        #pydict = dict()
        #pydict['acquisition_evs'] = ard_acquisition_evs
        #pydict['frame_onset_times'] = ard_frame_onset_times
        #pydict['frame_offset_times'] = ard_frame_offset_times
        #pydict['onset_idxs'] = ard_onset_idxs
        #pydict['offset_idxs'] = ard_offset_idxs
        #pydict['stop_ev_time'] = trialends[-1]['time'] #stop_ev['time']

        #pydict['ard_file_durs'] = ard_file_durs
        # pydict['ard_file_trigger_times'] = ard_file_trigger_times
        # pydict['ard_file_trigger_vals'] = ard_file_trigger_vals
    
        # pydict['ard_dfn'] = ard_dfn


    tif_fn = fn_base+'.mat'
    # scipy.io.savemat(os.path.join(source_dir, condition, tif_fn), mdict=pydict)
    scipy.io.savemat(os.path.join(source_dir, 'mw_data', tif_fn), mdict=pydict)
    print os.path.join(source_dir, 'mw_data', tif_fn)



# tif_fn = fn_base+'.mat'
# scipy.io.savemat(os.path.join(source_dir, 'mw_data', tif_fn), mdict=pydict)






# # Look at traces:

# si_experiment = 'fov6_rsvp_nomask_test_10trials_00002'
# curr_roi = 'roi1.xls'

# # check_channels = ['ch1', 'ch2', 'ch3']
# # roi_files = ['ch1_rois/'+curr__roi, 'ch2_rois/'+curr__roi, 'ch3_rois/'+curr__roi]
# # check_channels = ['ch1', 'ch2']
# # roi_files = [ch1_roi, ch2_roi]

# check_channels = ['ch1']
# roi_path = os.path.join(source_dir, si_experiment)
# roi_files = [curr_roi]

# nreps = 1490 # n volumes
# acquisition_rate = 4.11

# nframes_per_cyc = 3.*acquisition_rate #(1/target_freq) * acquisition_rate
# moving_win_sz = nframes_per_cyc * 2

# check_rois = dict()
# for ch_idx,curr_ch in enumerate(check_channels):

#   cr = csv.reader(open(os.path.join(roi_path, curr_ch+'_rois', roi_files[ch_idx]),"rb"), delimiter='\t')
#   arr = range(nreps) #adjust to needed
#   x = 0
#   for ridx,row in enumerate(cr):
#     if ridx==0:    
#       continue
#     else:
#       arr[x] = row
#       x += 1

#   y = [float(a[1]) for a in np.array(arr)]

#   if rolling is True:
#       # pix_padded = [np.ones(moving_win_sz)*y[y.keys()[0]], y, np.ones(moving_win_sz)*y[y.keys()[-1]]]
#       pix_padded = [np.ones(moving_win_sz)*y[0], y, np.ones(moving_win_sz)*y[-1]]
#       tmp_pix = list(itertools.chain(*pix_padded))
#       tmp_pix_rolling = np.convolve(tmp_pix, np.ones(moving_win_sz)/moving_win_sz, 'same')
#       remove_pad = (len(tmp_pix_rolling) - len(y) ) / 2
#       rpix = np.array(tmp_pix_rolling[remove_pad:-1*remove_pad])
#       y -= rpix




# # plt.figure()
# # y_sec = np.array(range(len(y))) / acquisition_rate
# # plt.plot(y_sec, y, 'k')

# # n=0
# # for tidx in range(len(rel_mw_times_sec)/2):
# #   #plt.hlines(ypos_markers, rel_mw_times_sec[n], rel_mw_times_sec[n+1], colors='b', linestyles='solid')
# #   x1 = rel_mw_times_sec[n]
# #   x2 = rel_mw_times_sec[n+1]
# #   plt.plot((x1, x2), (1, 1), 'b', linewidth=10)
# #   print n
# #   n+=2


# def smooth(x,window_len=11,window='hanning'):
#         if x.ndim != 1:
#                 raise ValueError, "smooth only accepts 1 dimension arrays."
#         if x.size < window_len:
#                 raise ValueError, "Input vector needs to be bigger than window size."
#         if window_len<3:
#                 return x
#         if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
#                 raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
#         s=np.r_[2*x[0]-x[window_len-1::-1],x,2*x[-1]-x[-1:-window_len:-1]]
#         if window == 'flat': #moving average
#                 w=np.ones(window_len,'d')
#         else:  
#                 w=eval('np.'+window+'(window_len)')
#         y=np.convolve(w/w.sum(),s,mode='same')
#         return y[window_len:-window_len+1]




# # Look at whole trace:

# # smoothed = smooth(y, window_len=10)

# mw_delay_sec = 80. / 1E3
# offset= mw_delay_sec * acquisition_rate

# plt.figure()
# plt.plot(y_sec+offset, y, 'k', alpha=0.5)
# plt.plot(y_sec+offset, smoothed, 'g', linewidth=2)
# n=0
# for tidx in range(len(rel_mw_times_sec)/2):
#     #plt.hlines(ypos_markers, rel_mw_times_sec[n], rel_mw_times_sec[n+1], colors='b', linestyles='solid')
#     x1 = rel_mw_times_sec[n]
#     x2 = rel_mw_times_sec[n+1]
#     plt.plot((x1, x2), (1, 1), 'b', linewidth=2, alpha=0.2)
#     print n
#     n+=2


# # Parse TIFF frames off of rel. MW times:
# smoothed = smooth(y, window_len=8)

# mw_trial_starts = rel_mw_times_sec[0::2]

# onsets = []
# traces = []
# for t in range(len(mw_trial_starts)-1):
#     if t is 0 or t is 1:
#         continue
#     else:
#         tON = mw_trial_starts[t]
#         pre = tON - 2.0
#         post = tON + 3.0
#         si_on_idx = np.argmax(y_sec > tON)
#         si_pre_idx = min(range(len(y_sec)), key=lambda i: abs(y_sec[i]-pre))
#         si_post_idx = min(range(len(y_sec)), key=lambda i: abs(y_sec[i]-post))

#         si_frames = smoothed[si_pre_idx:si_post_idx+1]
#         #si_frames = y[si_pre_idx:si_post_idx+1]

#         t_mw.append(np.array([mw_trial_starts[t-1] tON mw_trial_starts[t+1] mw_trial_starts[t+2]]))
#         t_si.append([y_])
#         traces.append(si_frames)


# trials_by_stim = [i.value[1]['filename'].split('/')[-1] for i in ievs]
# stim_list = list(set(trials_by_stim))
# stims = dict()
# for stim in stim_list:
#     stims[stim] = [idx for idx,im in enumerate(trials_by_stim) if im==stim]


# si_by_trials = dict()
# for sidx,sname in enumerate(stims.keys()):
#     trial_idxs = stims[sname]
#     trial_idxs = [i for i in trial_idxs if i!=0 and i<118]
#     stim_traces = [traces[tidx] for tidx in trial_idxs]
#     si_by_trials[sname] = stim_traces


# plt.figure()
# for s in range(len(stim_list)):
#     curr_stim = stim_list[s]
#     curr_traces = si_by_trials[curr_stim]
#     min_tpoints = min([i.shape for i in curr_traces])

#     for tidx,t in enumerate(curr_traces):
#         print tidx
#         if len(t) > min_tpoints[0]:
#             curr_traces[tidx] = t[0:-1]

#     mean_trace = np.mean(curr_traces,0)

#     plt.subplot(2,6,s)
#     plt.plot(mean_trace)
#     plt.plot((8, 10), (1, 1), 'k', linewidth=2, alpha=0.2)

