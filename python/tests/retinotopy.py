#!/usr/bin/env python2

import os
import numpy as np
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
        stop_ev = next(i for i in modes[run_idxs[0]:] if i['value']==0 or i['value']==1)
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
                    
                    curr_chunk = trigg_evs[start_idx:]
                    try:
                        curr_idx = [i for i,e in enumerate(curr_chunk) if e.value==0][0] # Find first DI high.
                    except IndexError:
                        break

                    stop_ev = next(i for i in curr_chunk[curr_idx:] if i['value']==1) # Find next DI low.
                    found_idx = [i.time for i in trigg_evs].index(stop_ev.time) # Get index of DI low.
                    
                    trigg_times.append([curr_chunk[curr_idx], stop_ev])
                    start_idx = found_idx #start_idx + found_idx
                    print start_idx


                except StopIteration:
                    print "Got to STOP."
                    break

            trigg_times = [t for t in trigg_times if t[1].time - t[0].time > 1]

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
                for cidx,cycle in enumerate(cycles):            
                    bartimes.append([b for b in bardevs if b.time < cycle[-1].time and b.time > trigg_times[cidx][0].time])

                tpositions = []
                for update in bartimes:
                    tpos = [[i.time, (i.value[1]['pos_x'], i.value[1]['pos_y'])] for i in update]
                    tpositions.append(tpos)

                # POS = dict(
                onum = 0
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

                    if curr_cond_type in POS.keys():
                        ncond_rep = len([i for i in POS.keys() if i==curr_cond_type])
                        curr_cond_type = curr_cond_type + '_' + str(ncond_rep+1)

                    POS[curr_cond_type] = cycstruct()
                    POS[curr_cond_type].times = cond
                    restarts.append(0)
                    POS[curr_cond_type].idxs = sorted(restarts)
                    POS[curr_cond_type].vals = posvec
                    POS[curr_cond_type].ordernum = onum
                    onum += 1


                I['ncycles'] = ncycles
                I['target_freq'] = df.get_events('cyc_per_sec')[-1].value
                I['barwidth'] = df.get_events('bar_size_deg')[-1].value
                #pix_evs = df.get_events('#pixelClockCode')
        return P, POS, trigg_times, I 


def get_timekey(item):
    return item.time

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
parser.add_option('-f', '--fn', action="store", dest="fn_base",
                  default="", help="shared filename for ard and mw files")
parser.add_option('-s', '--source', action="store", dest="source_dir",
                  default="", help="source (parent) dir containing mw_data and ard_data dirs")
parser.add_option('-S', '--stim', action="store",
                  dest="stimtype", default="grating", help="stimulus type (gratings or rsvp)?")
parser.add_option('-m', '--mask', action="store_true",
                  dest="mask", default=False, help="blue mask or no mask?")
parser.add_option('-L', '--long', action="store_true",
                  dest="long_trials", default=False, help="long (10sec) or short (3sec) trials?")
parser.add_option('-a', '--noard', action="store_true",
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

mw_files = os.listdir(os.path.join(source_dir, 'mw_data'))
mw_files = [m for m in mw_files if m.endswith('.mwk')]

for mw_file in mw_files:
    fn_base = mw_file[:-4]
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

    pevs, POS, trigg_times, info = get_bar_events(dfns, stimtype=stimtype)

    if len(pevs) > 1:
        pevs = [item for sublist in pevs for item in sublist]
        pevs = list(set(pevs))
        pevs.sort(key=operator.itemgetter(1))
    else:
        pevs = pevs[0]
    print "Found %i pixel clock events." % len(pevs)

    nexpected_pevs = int(round((1/info['target_freq']) * info['ncycles'] * 60 * len(trigg_times)))
    print "Expected %i pixel events, missing %i pevs." % (nexpected_pevs, nexpected_pevs-len(pevs))
    # on FLASH protocols, first real iamge event is 41

    print "Found %i conditions, corresponding to %i TIFFs." % (len(POS), len(trigg_times))

    # May need to fix this:
    print "Each cond duration is: ", [(t[1].time - t[0].time)/1E6 for t in trigg_times]


    n_codes = set([i[0] for i in pevs])
    if len(n_codes)<16:
        print "Check pixel clock -- missing bit values..."



    pydict = dict()
    for ridx,run in enumerate(POS.keys()):
        pydict[run] ={'time': [i[0] for i in POS[run].times], 'pos': POS[run].vals, 'idxs': POS[run].idxs, 'ordernum': POS[run].ordernum}
        pydict['triggers'] = [(t[0].time, t[1].time) for t in trigg_times]
        pydict['runs'] = POS.keys()
        pydict['stimtype'] = stimtype

        # Add some other info:
        pydict['info'] = info

        tif_fn = fn_base+'.mat'
        # scipy.io.savemat(os.path.join(source_dir, condition, tif_fn), mdict=pydict)
        scipy.io.savemat(os.path.join(source_dir, 'mw_data', tif_fn), mdict=pydict)
        print os.path.join(source_dir, 'mw_data', tif_fn)










    # t_mw_intervals = np.diff(mw_times)

    # parse_files = True

    # if parse_files is True:
    #     # find_matching_fidxs = np.where(t_mw_intervals > 3500000)[0]
    #     if long_trials is True:
    #         #nsecs = 10500000 # 3500000
    #         nsecs = 10900000 # 3500000
    #     else:
    #         nsecs = 3500000
    #     find_matching_fidxs = np.where(t_mw_intervals > nsecs)[0]
    #     mw_file_idxs = [i+1 for i in find_matching_fidxs]
    #     mw_file_idxs.append(0)
    #     mw_file_idxs = np.array(sorted(mw_file_idxs))
    #     print "Found %i MW file chunks." % len(mw_file_idxs)
    #     rel_mw_times = mw_times - mw_times[0] # Divide by 1E6 to get in SEC

    # else:

    #     strt = np.where(np.diff(mw_times) > 17000)
    #     trial_start_idx = strt[0][0]+1

    #     ntrials = len(mw_times[trial_start_idx:]) / 2.

    #     plt.plot(mw_times[trial_start_idx:], np.zeros((ntrials*2,1)), 'b*')

    #     rel_mw_times = mw_times[trial_start_idx:-1] - mw_times[trial_start_idx]
    #     rel_mw_times_sec = rel_mw_times / 1E6
    #     plt.plot(rel_mw_times_sec, np.zeros((len(rel_mw_times),1)), 'r*')


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

    # stupid_off = True # mistake in mwk protocol that sets DI to value...

    # # trigger_times = [i.time for i in trigg_evs] # Use frame-trigger times to find dur of each tif file in MW time
    # mw_file_durs = []
    # frame_trigger_times = []
    # tidx = 0
    # if stupid_off is True:
    #     find_stops = [i for i,v in enumerate(trigg_evs) if v.value==1] # 1 is OFF
    #     if trigg_evs[-1].value != 1: # This means MW stopped after TIF file finished (rather than during or before)
    #         find_stops.append(len(trigg_evs)-1)
    #     # if not find_stops:
    #     #     mw_file_dur = trigg_evs[-1].time - trigg_evs[0].time
    #     #     frame_trigger_times.append([trigg_evs[0].time, trigg_evs[-1].time])
    #     # else:
    #     for next_idx in find_stops:
    #         mw_file_dur = trigg_evs[next_idx].time - trigg_evs[tidx].time
    #         if mw_file_dur==0:
    #             continue
    #         else:
    #             mw_file_durs.append(mw_file_dur)
    #             frame_trigger_times.append([trigg_evs[tidx].time, trigg_evs[next_idx].time])
    #             tidx = next_idx + 1

    # else:
    #     for idx in range(len(mw_file_idxs)):

    #         mw_file_dur = trigger_times[tidx+1] - trigger_times[tidx]
    #         mw_file_durs.append(mw_file_dur)
    #         frame_trigger_times.append([trigger_times[tidx], trigger_times[tidx+1]])
    #         tidx += 2


    # mw_codes_by_file = []
    # mw_times_by_file = []
    # mw_trials_by_file = []
    # offsets_by_file = []
    # for i in range(len(mw_file_idxs)):
    #     if i==range(len(mw_file_idxs))[-1]:
    #         curr_mw_times = mw_times[mw_file_idxs[i]:]
    #         curr_mw_codes = mw_codes[mw_file_idxs[i]:]
    #         curr_mw_trials = tevs[mw_file_idxs[i]:]
    #     else:
    #         curr_mw_times = mw_times[mw_file_idxs[i]:mw_file_idxs[i+1]]
    #         curr_mw_codes = mw_codes[mw_file_idxs[i]:mw_file_idxs[i+1]]
    #         curr_mw_trials = tevs[mw_file_idxs[i]:mw_file_idxs[i+1]]

    #     mw_times_by_file.append(curr_mw_times)
    #     mw_codes_by_file.append(curr_mw_codes)
    #     mw_trials_by_file.append(curr_mw_trials)

    #     curr_offset = curr_mw_times[0] - frame_trigger_times[i][0] # time diff b/w detected frame trigger and stim display
    #     offsets_by_file.append(curr_offset)


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


    # Check arduino sampling rate:
    # max_allowed = 500 # make sure not skipping > 500us between samplings... (should be every 1ms)
    # sampling_range = [min(np.diff([e[1] for e in evs])), max(np.diff([e[1] for e in evs]))]
    # if sampling_range[1] - sampling_range[0] > max_allowed:
    #     print "Missing large sampling chunk: %i" % (sampling_range[1] - sampling_range[0])
    # else:
    #     print "Sampling rate passes: min - max intervals are %i - %i us." % (int(sampling_range[0]), int(sampling_range[1]))


    # Get blocks of tif files:
    onsets = np.where(np.diff([e[0] for e in ard_evs]) > 0)[0]   # where trigger val changes from 0-->1
    onsets = [t+1 for t in onsets]
    offsets = np.where(np.diff([e[0] for e in ard_evs]) < 0)[0]  # where trigger val changes from 1-->0
    offsets = [t+1 for t in offsets]


    if len(onsets) != len(offsets):
        print "N offsets %i does not match N onsets %i." % (len(offsets), len(onsets))
    else:
        print "N offsets and onsets match! Found %i frame events." % len(onsets)

    no_off_trigger = False
    if ard_evs[-1][0]==1:
        no_off_trigger = True


    tr_start_idx = onsets[0]
    too_short = 100 #500 # if diff b/w offsets is < 100, interval is only 100*1500us = 150ms... 
    on_intervals = np.where(np.diff(onsets) > too_short)[0] #[t for t in np.diff(onsets) if t>too_short]
    #off_intervals = np.where(np.diff(offsets) > too_short)[0]

    new_file_idxs = [onsets[i+1] for i in on_intervals]
    new_file_idxs.append(tr_start_idx)
    new_file_idxs = sorted(new_file_idxs)
    new_file_markers = [ard_evs[i] for i in sorted(new_file_idxs)]

    #if len(onsets) != len(offsets):
    end_file_idxs = []
    #    if len(new_file_idxs)==1 and not off_trigger:
    #   sublist = ard_evs[new_file_idxs[0]:]
    #   last_off = np.where(np.diff([e[0] for e in sublist]) < 0)[0][-1]
    #   end_file_idxs.append(last_off+new_file_idxs[0]+1)
    #   #end_file_idxs.append(len(evs)-2) # Take 2nd to last trigger ev, i.e., the last tstamp for finished frame (val=1)
    #    else:
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

    # Plot to check file chunks
    plt.plot([e[0] for e in ard_evs], 'k*')
    plt.plot(new_file_idxs, np.ones((1,len(new_file_idxs)))[0], 'g*', markersize=10)
    plt.plot(end_file_idxs, np.ones((1,len(end_file_idxs)))[0], 'r*', markersize=10)
    plt.title('Number of TIF files should be: %i' % len(new_file_markers))


    # acquisition_evs = evs[new_file_idxs[0]:end_file_idxs[0]+1]
    # ftimes = [e[1] for e in acquisition_evs]

    # onset_evs = [evs[on] for on in onsets]
    # offset_evs = [evs[off] for off in offsets]

    # onset_times = [e[1] for e in onset_evs]
    # offset_times = [e[1] for e in offset_evs]

    # frame_trigger_trange = (max(np.diff(onset_times)) - min(np.diff(onset_times))) / 1E3
    # print "Range of frame trigger time stamps: %0.2f - %0.2f ms." % (max(np.diff(onset_times))/1E3, min(np.diff(onset_times))/1E3)

    # frame_durs = np.array(offset_times) - np.array(onset_times)

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




    # no_ard = True
    #pydict = dict()
    #if no_ard is True:
    
    pydict['mw_times_by_file'] = mw_times_by_file
    pydict['mw_file_durs'] = mw_file_durs
    pydict['mw_frame_trigger_times'] = frame_trigger_times
    pydict['offsets_by_file'] = offsets_by_file
    pydict['mw_codes_by_file'] = mw_codes_by_file
    pydict['mw_dfn'] = dfn
    pydict['source_dir'] = source_dir
    pydict['fn_base'] = fn_base

    if no_ard is False:

        #pydict = dict()
        pydict['acquisition_evs'] = ard_acquisition_evs
        pydict['frame_onset_times'] = ard_frame_onset_times
        pydict['frame_offset_times'] = ard_frame_offset_times
        pydict['onset_idxs'] = ard_onset_idxs
        pydict['offset_idxs'] = ard_offset_idxs
        #pydict['stop_ev_time'] = trialends[-1]['time'] #stop_ev['time']

        pydict['ard_file_durs'] = ard_file_durs
        pydict['ard_file_trigger_times'] = ard_file_trigger_times
        pydict['ard_file_trigger_vals'] = ard_file_trigger_vals
    
    # pydict['mw_times_by_file'] = mw_times_by_file
    # pydict['mw_file_durs'] = mw_file_durs
    # pydict['mw_frame_trigger_times'] = frame_trigger_times
    # pydict['offsets_by_file'] = offsets_by_file
    # pydict['mw_codes_by_file'] = mw_codes_by_file

    pydict['ard_fn'] = ard_dfn
    pydict['mw_dfn'] = dfn
    pydict['source_dir'] = source_dir
    pydict['fn_base'] = fn_base
    pydict['stimtype'] = stimtype
    if stimtype=='image':
        pydict['condtypes'] = sorted(image_ids)
    elif stimtype=='grating':
        pydict['condtypes'] = sorted(gratings)
    elif stimtype=='bar':
        pydict['condtypes'] = ['left', 'right', 'top', 'bottom']
    
        pydict['info'] = info

    # tif_fn = 'fov1_gratings_10reps_run1.mat'

    tif_fn = fn_base+'.mat'
    # scipy.io.savemat(os.path.join(source_dir, condition, tif_fn), mdict=pydict)
    scipy.io.savemat(os.path.join(source_dir, 'mw_data', tif_fn), mdict=pydict)
    print os.path.join(source_dir, 'mw_data', tif_fn)

