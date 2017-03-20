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

        bounds = []
        for r in run_idxs: #[0]:
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
        I = []
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
                trigg_var_name = raw_input("Type var name to use: ")
                trigg_evs = df.get_events(trigg_var_name)
            else:
                trigg_evs = df.get_events(trigger_names[0])

            trigg_evs = [t for t in trigg_evs if t.time > boundary[0] and t.time < boundary[1]]
            trigg_indices = np.where(np.diff([t.value for t in trigg_evs]) == 1)[0] # when trigger goes from 0 --> 1, start --> end
            trigg_times = [[trigg_evs[i], trigg_evs[i+1]] for i in trigg_indices]

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
            pdev_info = [(v['bit_code'], p.time) for p in pdevs for v in p.value if 'bit_code' in v.keys()]

            print "Got %i pix code events." % len(pdev_info)
            P.append(pdevs)


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
                for cidx,cycle in enumerate(cycles):            
                    bartimes.append([b for b in bardevs if b.time < cycle[-1].time and b.time > trigg_starts[cidx][0].time])

                tpositions = []
                for update in bartimes:
                    tpos = [[i.time, (i.value[1]['pos_x'], i.value[1]['pos_y'])] for i in update]
                    tpositions.append(tpos)

                POS = dict()
                for cond in tpositions:
                    if cond[0][1][1]==0: # vertical cond, ypos=0
                        posvec = [i[1][0] for i in cond]
                        if posvec[0] < 0: # bar starting on LEFT
                            restarts = list(np.where(np.diff(posvec) < 0)[0] + 1)
                            curr_cond_type = 'left'
                        else:  
                            restarts = list(np.where(np.diff(posvec) > 0)[0] + 1)
                            curr_cond_type = 'right'
                    else: # horizontal cond, xpos = 0
                        posvec = [i[1][1] for i in cond] 
                        if posvec[0] < 0: # bar is starting at BOTTOM
                            restarts = list(np.where(np.diff(posvec) < 0)[0] + 1)
                            curr_cond_type = 'bottom'
                        else:
                            restarts = list(np.where(np.diff(posvec) > 0)[0] + 1)
                            curr_cond_type = 'top'

                    POS[curr_cond_type] = cycstruct()
                    POS[curr_cond_type].times = cond
                    restarts.append(0)
                    POS[curr_cond_type].idxs = sorted(restarts)
                    POS[curr_cond_type].vals = posvec

            I['ncycles'] = ncycles
            I['target_freq'] = df.get_events('cyc_per_sec')[-1].value
            T['barwidth'] = df.get_events('bar_size_deg')[-1].value
            #pix_evs = df.get_events('#pixelClockCode')
        return P, POS, trigg_times, I 






def get_pixel_clock_events(dfns, remove_orphans=True, stimtype='image'):
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
        s_ev = next(i for i in modes[run_idxs[0]:] if i['value']==0 or i['value']==1)
        bounds = []
        bounds.append([start_ev.time, stop_ev.time])

        # Check for any other start-stop events in session:
        for r in run_idxs[1:]: #[0]:
            if modes[r].time < stop_ev.time:
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
        I = []
        E = []
        for bidx,boundary in enumerate(bounds):
            if (boundary[1] - boundary[0]) < 1000000:
                print "Not a real boundary, only %i seconds found. Skipping." % int(boundary[1] - boundary[0])
                continue

            print "................................................................"
            print "SECTION %i" % bidx
            print "................................................................"
            #M1:#tmp_devs = df.get_events('#stimDisplayUpdate')                      # get *all* display update events
            # tmp_devs = [i for i in tmp_devs if i['time']<= boundary[1] and\
            #             i['time']>=boundary[0]]                                 # only grab events within run-time bounds (see above)

            #M1:#devs = [e for e in tmp_devs if not e.value[0]==[None]]

            # deal with inconsistent trigger-naming:
            codec_list = df.get_codec()
            trigger_names = [i for i in codec_list.values() if ('trigger' in i or 'Trigger' in i) and 'flag' not in i]
                # trigg_evs = df.get_events('frame_triggered')
                # trigg_evs = df.get_events('FrameTrigger')
                # trigg_evs = df.get_events('frame_trigger')
            if len(trigger_names) > 1:
                print "Found > 1 name for frame-trigger:"
                print "Choose: ", trigger_names
                trigg_var_name = raw_input("Type var name to use: ")
                trigg_evs = df.get_events(trigg_var_name)
            else:
                trigg_evs = df.get_events(trigger_names[0])

            trigg_evs = [t for t in trigg_evs if t.time >= boundary[0] and t.time <= boundary[1]]
            # trigg_indices = np.where(np.diff([t.value for t in trigg_evs]) == 1)[0] # when trigger goes from 0 --> 1, start --> end
            # trigg_times = [[trigg_evs[i], trigg_evs[i+1]] for i in trigg_indices]
            first_trigger = [i for i,e in enumerate(trigg_evs) if e.value==0][0]
            curr_idx = copy.copy(first_trigger)

            trigg_times = []
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
                    
                    trigg_times.append([curr_chunk[curr_idx], off_ev])
                    start_idx = found_idx #start_idx + found_idx
                    print start_idx


                except StopIteration:
                    print "Got to STOP."
                    if found_new_start is True:
                        early_abort = True
                        trigg_times.append([curr_chunk[curr_idx], ])
                    break

            if early_abort is True:
                trigg_times[-1] = trigg_times[-1][0], bo
            trigg_times = [t for t in trigg_times if t[1].time - t[0].time > 1]

            # Check stimDisplayUpdate events vs announceStimulus:
            stim_evs = df.get_events('#stimDisplayUpdate')
            devs = [e for e in stim_evs if e.value and not e.value[0]==None]
            devs = [d for d in devs if d.time <= boundary[1] and d.time >= boundary[0]]

            tmp_pdevs = [i for i in devs for v in i.value if 'bit_code' in v.keys()]
            # tmp_pdev_info = [(v['bit_code'], i.time) for i in devs for v in i.value if 'bit_code' in v.keys()]

            # Get rid of "repeat" events from state updates.
            #tmp_pdevs = [p for i,p in enumerate(pdevs) if i not in nons]
            # pdevs = [i for i in tmp_pdevs if i[1]<= boundary[1] and i[1]>=boundary[0]]
            pdevs = [i for i in tmp_pdevs if i.time<= boundary[1] and i.time>=boundary[0]]
            print "N pix-evs found in boundary: %i" % len(pdevs)
            # nons = np.where(np.diff([i[0] for i in pdevs])==0)[0]
            nons = np.where(np.diff([i.value[-1]['bit_code'] for i in pdevs])==0)[0] # pix stim event is always last
            pdevs = [p for i,p in enumerate(pdevs) if i not in nons]
            pdev_info = [(v['bit_code'], p.time) for p in pdevs for v in p.value if 'bit_code' in v.keys()]

            print "Got %i pix code events." % len(pdev_info)

            P.append(pdevs)

            idevs = [i for i in pdevs for v in i.value if 'bit_code' in v.keys()]

            #pix_evs = df.get_events('#pixelClockCode')
            if stimtype=='image':
                imdevs = [d for d in devs for i in d.value if 'filename' in i.keys() and '.png' in i['filename']]
                I.append(imdevs)
                
                # Find blank ITIs:
                #prevdev = [[i for i,d in enumerate(devs) if d.time < t.time][-1] for t in imdevs[1:]]
                #lastdev = [i for i,d in enumerate(devs) if d.time > imdevs[-1].time and len(d.value)<3][0] # ignore the last "extra" ev (has diff time-stamp) - just wnt 1st non-grating blank
                if mask is True:
                    tdevs = [i for i in idevs for v in i.value if v['name']=='blue_mask' and i not in imdevs]
                else:
                    tdevs = [i for i in idevs if i.time>imdevs[0].time and i not in imdevs]

                #tdevs.append(devs[lastdev])
                
                T = imdevs + tdevs
                T = sorted(T, key=get_timekey)


            if stimtype=='grating':
                imdevs = [d for d in devs for i in d.value if i['name']=='gabor']

                start_times = [i.value[1]['start_time'] for i in imdevs] # Use start_time to ignore dynamic pixel-code of drifting grating since stim as actually static
                find_static = np.where(np.diff(start_times) > 0)[0]
                find_static = np.append(find_static, 0)
                find_static = sorted(find_static)
                imtrials = [imdevs[i+1] for i in find_static]
                I.append(imtrials)

                # Find blank ITIs:
                if mask is True:
                    tdevs = [i for i in idevs if i.time>imdevs[0].time and i not in imdevs]
                else:
                    prevdev = [[i for i,d in enumerate(devs) if d.time < t.time][-1] for t in imtrials[1:]]
                    lastdev = [i for i,d in enumerate(devs) if d.time > imtrials[-1].time and len(d.value)<3][0] # ignore the last "extra" ev (has diff time-stamp) - just wnt 1st non-grating blank
                    tdevs = [devs[i] for i in prevdev]
                    tdevs.append(devs[lastdev])
                
                T = imtrials + tdevs
                T = sorted(T, key=get_timekey)

            trial_ends = [i for i in df.get_events('Announce_TrialEnd') if i.value==1]
            
            E.append(trial_ends)

            #pix_evs = df.get_events('#pixelClockCode')
        return P, I, E, T, trigg_evs


def get_timekey(item):
    return item.time

# source_dir = '/media/juliana/Seagate Backup Plus Drive/RESDATA'
# # condition_dir = '20161222_JR030W_retinotopy2'
# # run_dir = 'fov1_bar037Hz_retinotopy_run2_00007'
# # slice_dir = 'ch1_slices'
# date = '20161222'
# animal = 'JR030W'
# experiment = 'rsvp_25reps'
# run = 'run1'


# source_dir = '/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W'
# source_dir = '/nas/volume1/2photon/RESDATA/20161222_JR030W/20161222_JR030W_gratings1'

# source_dir = '/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/NMF'
# fn_base = '20161219_JR030W_grating2'

# source_dir = '/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/rsvp/fov6_rsvp_bluemask_test_10trials/fov6_rsvp_bluemask_test_10trials_00001'
# fn_base = '20161219_JR030W_rsvp_bluemask_test_10trials'

# fn_base = '20161221_JR030W_rsvp_25reps'
# source_dir = '/nas/volume1/2photon/RESDATA/20161221_JR030W/rsvp'
# stimtype = 'image'

###################################
# fn_base = '20161219_JR030W_gratings_bluemask_5trials_2'
# source_dir = '/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/gratings/fov6_gratings_bluemask_5trials_00003/'
# stimtype = 'grating'
# mask = True

# fn_base = '20161222_JR030W_gratings_10reps_run1'
# source_dir = '/nas/volume1/2photon/RESDATA/20161222_JR030W/20161222_JR030W_gratings1'
# stimtype = 'grating'
# mask = False

import optparse

parser = optparse.OptionParser()
parser.add_option('--fn', action="store", dest="fn_base",
                  default="", help="shared filename for ard and mw files")
parser.add_option('--source', action="store", dest="source_dir",
                  default="", help="source (parent) dir containing mw_data and ard_data dirs")
parser.add_option('--stim', action="store",
                  dest="stimtype", default="grating", help="stimulus type (gratings or rsvp)?")
parser.add_option('--mask', action="store_true",
                  dest="mask", default=False, help="blue mask or no mask?")
parser.add_option('--long', action="store_true",
                  dest="long_trials", default=False, help="long (10sec) or short (3sec) trials?")
parser.add_option('--noard', action="store_true",
                  dest="no_ard", default=False, help="No arduino triggers saved?")

# fn_base = '20160118_AG33_gratings_fov1_run1'
# source_dir = '/nas/volume1/2photon/RESDATA/TEFO/20160118_AG33/fov1_gratings1'
# stimtype = 'grating'
# mask = False
# long_trials = True
(options, args) = parser.parse_args()

fn_base = options.fn_base #'20160118_AG33_gratings_fov1_run1'
source_dir = options.source_dir #'/nas/volume1/2photon/RESDATA/TEFO/20160118_AG33/fov1_gratings1'
stimtype = options.stimtype #'grating'
mask = options.mask # False
long_trials = options.long_trials #True

no_ard = options.no_ard

# fn_base = '20160115_AG33_gratings_fov3_run3'
# source_dir = '/nas/volume1/2photon/RESDATA/TEFO/20160115_AG33/'
# condition = 'fov3_gratings1'
# stimtype = 'grating'
# mask = False

# fn_base = '20160118_AG33_gratings_fov1_run6'
# source_dir = '/nas/volume1/2photon/RESDATA/TEFO/20160118_AG33/fov1_gratings1'
# condition = 'fov1_gratings1'
# stimtype = 'grating'
# mask = False

###################################
# date = '20161222' #'20161219'
# animal = 'JR030W'
# experiment = 'gratings_10reps_run1' #'rsvp_nomask_test_10trials'
# # run = '1'

# # fn_base = '_'.join([date, animal, experiment, run])
# fn_base = '_'.join([date, animal, experiment])


# stimtype = 'grating' # 'image'

# --------------------------------------------------------------------
# MW codes:
# --------------------------------------------------------------------

# mw_data_dir = os.path.join(source_dir, 'mw_data')
mw_data_dir = os.path.join(source_dir, 'mw_data')
mw_fn = fn_base+'.mwk'
dfn = os.path.join(mw_data_dir, mw_fn)
dfns = [dfn]
# df = pymworks.open(dfn)
# pix = df.get_events('#pixelClockCode')

if stimtype=='bar':
    pevs, POS, trigg_times = get_bar_events(dfns, stimtype=stimtype)
else:
    pevs, ievs, trialends, tevs, trigg_evs = get_pixel_clock_events(dfns, stimtype=stimtype)

if len(pevs) > 1:
    pevs = [item for sublist in pevs for item in sublist]
    pevs = list(set(pevs))
    pevs.sort(key=operator.itemgetter(1))
else:
    pevs = pevs[0]
print "Found %i pixel clock events." % len(pevs)

# on FLASAH protocols, first real iamge event is 41

ievs = ievs[0]
print "Found %i image trials." % len(ievs)
# if len(ievs) > 1:
#   ievs = [item for sublist in ievs for item in sublist]
#   ievs = list(set(ievs))
#   ievs.sort(key=operator.itemgetter(1))

print "Found %i stimulus update events across trials." % len(tevs)

# May need to fix this:
trialends = trialends[0]
print "MW session duration is: ", (trialends[-1]['time'] - tevs[0].time )/1E6
# for i in range(len(trialends)-1):
#     print "Starting at 0: %f" % ((trialends[i]['time'] - ievs[i]['time'])/1E6)
# for i in range(len(trialends)-1):
#     print "Starting at 1: %f" % ((trialends[i+1]['time'] - ievs[i]['time'])/1E6)

# (trialends[0]['time'] - ievs[0]['time']) / 1E6.


n_codes = set([i[0] for i in pevs])
if len(n_codes)<16:
    print "Check pixel clock -- missing bit values..."


if stimtype == 'image':
    all_ims = [i.value[1]['name'] for i in ievs]
    image_ids = sorted(list(set(all_ims)))

    mw_times = np.array([i.time for i in tevs]) #np.array([i[0] for i in pevs])
    mw_codes = []
    for i in tevs:
        if len(i.value)==4:
            stim_name = i.value[1]['name']
            stim_idx = [iidx for iidx,image in enumerate(image_ids) if image==stim_name][0]+1
        else:
            stim_idx = 0
        mw_codes.append(stim_idx)
    mw_codes = np.array(mw_codes)

elif stimtype == 'grating':
    # Need to find all unique grating types:
    all_combos = [(i.value[1]['rotation'], round(i.value[1]['frequency'],1)) for i in ievs]
    gratings = sorted(list(set(all_combos))) # should be 35 for gratings -- sorted by orientation (7) x SF (5)

    mw_times = np.array([i.time for i in tevs])
    mw_codes = []
    for i in tevs:
        if len(i.value)>2: # contains image stim
            stim_config = (i.value[1]['rotation'], round(i.value[1]['frequency'],1))
            stim_idx = [gidx for gidx,grating in enumerate(sorted(gratings)) if grating==stim_config][0]+1
        else:
            stim_idx = 0
        mw_codes.append(stim_idx)
    mw_codes = np.array(mw_codes)


t_mw_intervals = np.diff(mw_times)

parse_files = True

if parse_files is True:
    # find_matching_fidxs = np.where(t_mw_intervals > 3500000)[0]
    if long_trials is True:
        #nsecs = 10500000 # 3500000
        nsecs = 10900000 # 3500000
    else:
        nsecs = 3500000
    find_matching_fidxs = np.where(t_mw_intervals > nsecs)[0]
    mw_file_idxs = [i+1 for i in find_matching_fidxs]
    mw_file_idxs.append(0)
    mw_file_idxs = np.array(sorted(mw_file_idxs))
    print "Found %i MW file chunks." % len(mw_file_idxs)
    rel_mw_times = mw_times - mw_times[0] # Divide by 1E6 to get in SEC

else:

    strt = np.where(np.diff(mw_times) > 17000)
    trial_start_idx = strt[0][0]+1

    ntrials = len(mw_times[trial_start_idx:]) / 2.

    plt.plot(mw_times[trial_start_idx:], np.zeros((ntrials*2,1)), 'b*')

    rel_mw_times = mw_times[trial_start_idx:-1] - mw_times[trial_start_idx]
    rel_mw_times_sec = rel_mw_times / 1E6
    plt.plot(rel_mw_times_sec, np.zeros((len(rel_mw_times),1)), 'r*')


# import matplotlib.patches as patches
# ypos_markers = np.ones((len(rel_mw_times_sec),1))
# n=0
# for tidx in range(len(rel_mw_times_sec)):
#   plt.hlines(ypos_markers, rel_mw_times_sec[n], rel_mw_times_sec[n+1], colors='k', linestyles='solid')
#   n+=2


# mw_file_durs = []
# for i,midx in enumerate(mw_file_idxs[0:-1]):
#     mw_end = np.mean(t_mw_intervals[midx+1:mw_file_idxs[i+1]-1:2]) # off for 2...
#     mw_file_dur = (sum(t_mw_intervals[midx:mw_file_idxs[i+1]-1])+mw_end)/1E6
#     mw_file_durs.append(mw_file_dur)

stupid_off = True # mistake in mwk protocol that sets DI to value...

trigger_times = [i.time for i in trigg_evs] # Use frame-trigger times to find dur of each tif file in MW time
mw_file_durs = []
frame_trigger_times = []
tidx = 0
if stupid_off is True:
    find_stops = [i for i,v in enumerate(trigg_evs) if v.value==1] # 1 is OFF
    if trigg_evs[-1].value != 1: # This means MW stopped after TIF file finished (rather than during or before)
        find_stops.append(len(trigg_evs)-1)
    # if not find_stops:
    #     mw_file_dur = trigg_evs[-1].time - trigg_evs[0].time
    #     frame_trigger_times.append([trigg_evs[0].time, trigg_evs[-1].time])
    # else:
    for next_idx in find_stops:
        mw_file_dur = trigg_evs[next_idx].time - trigg_evs[tidx].time
        if mw_file_dur==0:
            continue
        else:
            mw_file_durs.append(mw_file_dur)
            frame_trigger_times.append([trigg_evs[tidx].time, trigg_evs[next_idx].time])
            tidx = next_idx + 1

else:
    for idx in range(len(mw_file_idxs)):

        mw_file_dur = trigger_times[tidx+1] - trigger_times[tidx]
        mw_file_durs.append(mw_file_dur)
        frame_trigger_times.append([trigger_times[tidx], trigger_times[tidx+1]])
        tidx += 2


mw_codes_by_file = []
mw_times_by_file = []
mw_trials_by_file = []
offsets_by_file = []
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


# file_no = 2
# rel_times = list(mw_times_by_file[file_no-1] - mw_times_by_file[file_no-1][0])
# mw_rel_times = rel_times + offsets_by_file[file_no-1]
# mw_codes = mw_codes_by_file[file_no-1]





# --------------------------------------------------------------------
# ARDUINO codes:
# --------------------------------------------------------------------

# ard_path = '/nas/volume1/2photon/RESDATA/ard_data'
# ard_data_dir = '/Users/julianarhee/Documents/MWorks/PyData'
# ard_fn = 'raw00.txt'
# fn_base = '20161222_JR030W_gratings_10reps_run1'
# ard_fn = fn_base+'.txt'

ard_fn = fn_base+'.txt'
ard_data_dir = os.path.join(source_dir, 'ard_data')
ard_dfn = os.path.join(ard_data_dir, ard_fn)


# evs = get_arduino_events(ard_dfn)
# ard_times = np.array([i[1] for i in evs])
# ard_codes = np.array([i[0] for i in evs])


# def get_int(bitcode):

#     a = int(bitcode[3])*(2**0)
#     b = int(bitcode[2])*(2**1)
#     c = int(bitcode[1])*(2**2)
#     d = int(bitcode[0])*(2**3)

#     return a+b+c+d

nsplits = 2

f = codecs.open(ard_dfn, 'r')
data = f.read()
bad_end = False
if not data[-1]=='_':
    bad_end = True
packets = data.split('__')
evs = []
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
            evs.append([pin_id, tstamp])
        except NameError as e:
            print pidx
            print pin_id
            print tstring

    except IndexError as e:
        print pidx
        print packet

ard_times = np.array([i[1] for i in evs])
ard_codes = np.array([i[0] for i in evs])



# Check for arduino roll-overs:
tdiffs = np.array(np.diff(ard_times))
rollover_idxs = np.where(tdiffs<0)
if any(rollover_idxs):
    print "Found rollover points!  Fix this..."


# Check arduino sampling rate:
# max_allowed = 500 # make sure not skipping > 500us between samplings... (should be every 1ms)
# sampling_range = [min(np.diff([e[1] for e in evs])), max(np.diff([e[1] for e in evs]))]
# if sampling_range[1] - sampling_range[0] > max_allowed:
#     print "Missing large sampling chunk: %i" % (sampling_range[1] - sampling_range[0])
# else:
#     print "Sampling rate passes: min - max intervals are %i - %i us." % (int(sampling_range[0]), int(sampling_range[1]))


# Get blocks of tif files:
onsets = np.where(np.diff([e[0] for e in evs]) > 0)[0]   # where trigger val changes from 0-->1
onsets = [t+1 for t in onsets]
offsets = np.where(np.diff([e[0] for e in evs]) < 0)[0]  # where trigger val changes from 1-->0
offsets = [t+1 for t in offsets]


if len(onsets) != len(offsets):
    print "N offsets %i does not match N onsets %i." % (len(offsets), len(onsets))
else:
    print "N offsets and onsets match! Found %i frame events." % len(onsets)

off_trigger = False
if evs[-1][0]==1:
    off_trigger = True


tr_start_idx = onsets[0]
too_short = 700 #500 # if diff b/w offsets is < 100, interval is only 100*1500us = 150ms... 
on_intervals = np.where(np.diff(onsets) > too_short)[0] #[t for t in np.diff(onsets) if t>too_short]
#off_intervals = np.where(np.diff(offsets) > too_short)[0]

new_file_idxs = [onsets[i+1] for i in on_intervals]
new_file_idxs.append(tr_start_idx)
new_file_idxs = sorted(new_file_idxs)
new_file_markers = [evs[i] for i in sorted(new_file_idxs)]

end_file_idxs = []
if len(new_file_idxs)==1 and not off_trigger:
    sublist = evs[new_file_idxs[0]:]
    last_off = np.where(np.diff([e[0] for e in sublist]) < 0)[0][-1]
    end_file_idxs.append(last_off+new_file_idxs[0]+1)
    #end_file_idxs.append(len(evs)-2) # Take 2nd to last trigger ev, i.e., the last tstamp for finished frame (val=1)
else:
    for i,nidx in enumerate(new_file_idxs[0:-1]):
        sublist = evs[nidx:new_file_idxs[i+1]]
        last_off = np.where(np.diff([e[0] for e in sublist]) < 0)[0][-1]
        off_idx = nidx + last_off + 1
        end_file_idxs.append(off_idx)
if off_trigger:
    end_file_idxs.append(len(evs)-1) # just add the last trigger event for end-of-file

end_file_markers = [evs[i] for i in sorted(end_file_idxs)]
print "Found %i TIF file chunks." % len(new_file_markers)
if len(new_file_markers)==1:
    print "DUR of session is: %f sec." % ((end_file_markers[0][1] - new_file_markers[0][1])/1E6)

# Plot to check file chunks
plt.plot([e[0] for e in evs], 'k*')
plt.plot(new_file_idxs, np.ones((1,len(new_file_idxs)))[0], 'g*', markersize=10)
plt.plot(end_file_idxs, np.ones((1,len(end_file_idxs)))[0], 'r*', markersize=10)
plt.title('Number of TIF files should be: %i' % len(new_file_markers))


acquisition_evs = evs[new_file_idxs[0]:end_file_idxs[0]+1]
ftimes = [e[1] for e in acquisition_evs]

onset_evs = [evs[on] for on in onsets]
offset_evs = [evs[off] for off in offsets]

onset_times = [e[1] for e in onset_evs]
offset_times = [e[1] for e in offset_evs]

frame_trigger_trange = (max(np.diff(onset_times)) - min(np.diff(onset_times))) / 1E3
print "Range of frame trigger time stamps: %0.2f - %0.2f ms." % (max(np.diff(onset_times))/1E3, min(np.diff(onset_times))/1E3)

frame_durs = np.array(offset_times) - np.array(onset_times)


ard_file_durs = []
ard_file_trigger_times = []
ard_file_trigger_vals = []
for sidx,eidx in zip(new_file_idxs, end_file_idxs):
    ard_file_dur = (ard_times[eidx] - ard_times[sidx])/1E6
    ard_file_durs.append(ard_file_dur)

    ard_frame_tval = ard_codes[sidx:eidx+1]
    ard_frame_ttime = ard_times[sidx:eidx+1]
    
    ard_file_trigger_vals.append(ard_frame_tval)
    ard_file_trigger_times.append(ard_frame_ttime)



# Better aligning?

# curr_mw_times = list(curr_mw_times)
# curr_mw_times.append(stop_ev['time'])
# mw_secs = ((curr_mw_times - curr_mw_times[0]) + curr_offset) / 1E6
# ard_frame_secs = (ard_frame_ttime - ard_frame_ttime[0]) / 1E6
# ard_converted = [(e[0], e[1]-evs[tr_start_idx][1]) for e in evs[sidx:eidx]]

# nslices = 30
# nvolumes = 1850
# ntotal_frames = nslices * nvolumes
# frame_secs = np.interp(range(ntotal_frames), range(len(ard_frame_secs)), ard_frame_secs)

# [ard_converted[e] for (e,s) in enumerate(frame_secs) if abs(ard_converted[e][1]/1E6 - s) == min([abs(a[1]/1E6 - s) for a in ard_converted])]

# matched_evs = []
# for s in frame_secs:
#     matching_ev = [np.where(abs([e[1]/1E6 for e in ard_converted] - s) == min(abs([e[1]/1E6 for e in ard_converted] - s)))[0] for s in frame_secs]
#     matched_evs.append(ard_converted[matching_ev])


# for i in curr_mw_times:
#     np.where(abs(ard_frame_secs - i) == min(abs(ard_frame_secs - i)))[0]

# no_ard = True
if no_ard is True:
    pydict = dict()
    pydict['mw_times_by_file'] = mw_times_by_file
    pydict['mw_file_durs'] = mw_file_durs
    pydict['mw_frame_trigger_times'] = frame_trigger_times
    pydict['offsets_by_file'] = offsets_by_file
    pydict['mw_codes_by_file'] = mw_codes_by_file
    pydict['mw_dfn'] = dfn
    pydict['source_dir'] = source_dir
    pydict['fn_base'] = fn_base
    pydict['stimtype'] = stimtype
    if stimtype=='image':
        pydict['stim_idxs'] = sorted(image_ids)
    elif stimtype=='grating':
        pydict['gratings'] = sorted(gratings)


else:

    pydict = dict()
    pydict['acquisition_evs'] = acquisition_evs
    pydict['frame_onset_times'] = onset_times
    pydict['frame_offset_times'] = offset_times
    pydict['onset_idxs'] = onsets
    pydict['offset_idxs'] = offsets
    pydict['stop_ev_time'] = trialends[-1]['time'] #stop_ev['time']

    pydict['ard_file_durs'] = ard_file_durs
    pydict['ard_file_trigger_times'] = ard_file_trigger_times
    pydict['ard_file_trigger_vals'] = ard_file_trigger_vals
    
    pydict['mw_times_by_file'] = mw_times_by_file
    pydict['mw_file_durs'] = mw_file_durs
    pydict['mw_frame_trigger_times'] = frame_trigger_times
    pydict['offsets_by_file'] = offsets_by_file
    pydict['mw_codes_by_file'] = mw_codes_by_file
    pydict['ard_fn'] = ard_dfn
    pydict['mw_dfn'] = dfn
    pydict['source_dir'] = source_dir
    pydict['fn_base'] = fn_base
    pydict['stimtype'] = stimtype
    if stimtype=='image':
        pydict['stim_idxs'] = sorted(image_ids)
    elif stimtype=='grating':
        pydict['gratings'] = sorted(gratings)

# tif_fn = 'fov1_gratings_10reps_run1.mat'

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

