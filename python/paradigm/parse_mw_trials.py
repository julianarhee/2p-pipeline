#!/usr/bin/env python2

import numpy as np
import os

# from bokeh.io import gridplot, output_notebook, output_file, show
# from bokeh.plotting import figure
# output_notebook()

import cPickle as pkl
import shutil

import time
import datetime

import pandas as pd
import scipy.io
import copy
import re

from json_tricks.np import dump, dumps, load, loads

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

# In[3]:

# Import functions for getting event info needed to parse MW trials:
from parse_mw_events import get_bar_events, get_stimulus_events


# In[4]:

# Set datafile paths in opts:
# prepend = '/' #'/Users/julianarhee'
# source_dir = os.path.join(prepend,'nas/volume1/2photon/RESDATA')  # options.source_dir

# session = '20161222_JR030W'

# experiment = 'gratings2'
# #parser.add_option("--experiment", action="append", dest="experiment_list")

# stimtype = 'grating' #options.stimtype #'grating'
# no_ard = True #options.no_ard

import optparse

parser = optparse.OptionParser()
parser.add_option('-S', '--source', action='store', dest='source', default='/nas/volume1/2photon/projects', help='source dir (root project dir containing all expts) [default: /nas/volume1/2photon/projects]')
parser.add_option('-E', '--experiment', action='store', dest='experiment', default='', help='experiment type (parent of session dir)') 
parser.add_option('-s', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID') 
parser.add_option('-A', '--acq', action='store', dest='acquisition', default='', help="acquisition folder (ex: 'FOV1_zoom3x')")
parser.add_option('-f', '--functional', action='store', dest='functional_dir', default='functional', help="folder containing functional TIFFs. [default: 'functional']")

# # parser.add_option('--fn', action="store", dest="fn_base",
# #                   default="", help="shared filename for ard and mw files")
# parser.add_option('-S', '--source', action="store", dest="source_dir",
#                   default="", help="source (parent) dir containing mw_data and ard_data dirs")
# parser.add_option('-s', '--session', action="store", dest="session",
#                   default="", help="session [ex., 'YYYMMDD_animalname']")
# parser.add_option('-r', '--run', action="append", dest="experiment_list",
#                   default=[], help="(list of) runs or experiment folder [ex., 'retinotopy5']")

parser.add_option('--stim', action="store",
                  dest="stimtype", default="grating", help="stimulus type [options: grating, image, bar].")
parser.add_option('--phasemod', action="store_true",
                  dest="phasemod", default=False, help="include if stimulus mod (phase-modulation).")
parser.add_option('--ard', action="store_false",
                  dest="no_ard", default=True, help="Flag to parse arduino serialdata")
parser.add_option('-t', '--triggervar', action="store",
                  dest="frametrigger_varname", default='frame_trigger', help="Temp way of dealing with multiple trigger variable names [default: frame_trigger]")



(options, args) = parser.parse_args()
trigger_varname = options.frametrigger_varname

(options, args) = parser.parse_args() 

source = options.source #'/nas/volume1/2photon/projects'
experiment = options.experiment #'scenes' #'gratings_phaseMod' #'retino_bar' #'gratings_phaseMod'
session = options.session #'20171003_JW016' #'20170927_CE059' #'20170902_CE054' #'20170825_CE055'
acquisition = options.acquisition #'FOV1' #'FOV1_zoom3x' #'FOV1_zoom3x_run2' #'FOV1_planar'
functional_dir = options.functional_dir #'functional' #'functional_subset'

acquisition_dir = os.path.join(source, experiment, session, acquisition)


# fn_base = options.fn_base #'20160118_AG33_gratings_fov1_run1'
# source_dir = options.source_dir #'/nas/volume1/2photon/RESDATA/TEFO/20160118_AG33/fov1_gratings1'
# session = options.session
# experiment_list = options.experiment_list
# experiment = experiment_list[0]
# if len(experiment_list)==1:
#     data_dir = os.path.join(source_dir, session, experiment)
#     
stimtype = options.stimtype #'grating'
no_ard = options.no_ard
phasemod = options.phasemod

# In[5]:

# Look in child dir (of source_dir) to find mw_data paths:
# data_dir = os.path.join(source_dir, session, experiment)
#mw_dir = os.path.join(data_dir, 'mw_data')
#mwfiles = os.listdir(mw_dir)
paradigm_dir = os.path.join(acquisition_dir, functional_dir, 'paradigm_files')
raw_paradigm_dir = os.path.join(paradigm_dir, 'raw')
if not os.path.exists(raw_paradigm_dir):
    os.makedirs(raw_paradigm_dir)
print "Moving files to RAW dir:", raw_paradigm_dir

raw_files = [f for f in os.listdir(paradigm_dir) if 'mwk' in f or 'serial' in f]
for fi in raw_files:
    shutil.move(os.path.join(paradigm_dir, fi), raw_paradigm_dir)

mwfiles = os.listdir(raw_paradigm_dir)
mwfiles = [m for m in mwfiles if m.endswith('.mwk')]

# TODO:  adjust path-setting to allow for multiple reps of the same experiment

# In[6]:

# Cycle through MW files (1 file per experiment rep):
all_dfns = []
for mwfile in mwfiles:
    fn_base = mwfile[:-4]
    mw_fn = fn_base+'.mwk'
    mw_dfn = os.path.join(raw_paradigm_dir, mw_fn)
    all_dfns.append(mw_dfn)
    if not no_ard:
        ard_fn = fn_base+'.txt'
        ard_dfn = os.path.join(mw_dir, ard_fn)
        all_dfns.append(ard_dfn)
        
mw_dfns = [f for f in all_dfns if f.endswith('.mwk')]
ar_dfns = [f for f in all_dfns if f.endswith('.txt')]

print "MW files: ", mw_dfns


# In[7]:

# Get MW events
# didx = 0
mw_dfns = sorted(mw_dfns, key=natural_keys)
ar_dfns = sorted(ar_dfns, key=natural_keys)

for didx in range(len(mw_dfns)):
    curr_dfn = mw_dfns[didx]
    curr_dfn_base = os.path.split(curr_dfn)[1][:-4]
    print "Current file: ", curr_dfn
    if stimtype=='bar':
        pixelevents, stimevents, trigger_times, session_info = get_bar_events(curr_dfn, triggername=trigger_varname)
    else:
        pixelevents, stimevents, trialevents, trigger_times, session_info = get_stimulus_events(curr_dfn, stimtype=stimtype, phasemod=phasemod, triggername=trigger_varname)

    # In[8]:

    # For EACH boundary found for a given datafile (dfn), make sure all the events are concatenated together:
    print "================================================================"
    print "MW parsing summary:"
    print "================================================================"
    if len(pixelevents) > 1 and type(pixelevents[0])==list:
        pixelevents = [item for sublist in pixelevents for item in sublist]
        pixelevents = list(set(pixelevents))
        pixelevents.sort(key=operator.itemgetter(1))
    else:
        pixelevents = pixelevents[0]
    print "Found %i pixel clock events." % len(pixelevents)

    # Check that all possible pixel vals are used (otherwise, pix-clock may be missing input):
    print [p for p in pixelevents if 'bit_code' not in p.value[-1].keys()]
    n_codes = set([i.value[-1]['bit_code'] for i in pixelevents])
    if len(n_codes)<16:
        print "Check pixel clock -- missing bit values..."


    if len(stimevents) > 1 and type(stimevents[0])==list:
        stimevents = [item for sublist in stimevents for item in sublist]
        stimevents = list(set(stimevents))
        stimevents.sort(key=operator.itemgetter(1))
    else:
        stimevents = stimevents[0]
    print "Found %i stimulus on events." % len(stimevents)
    
    if not stimtype=='bar':
	if len(trialevents) > 1 and type(trialevents[0])==list:
	    trialevents = [item for sublist in trialevents for item in sublist]
	    trialevents = list(set(trialevents))
	    trialevents.sort(key=operator.itemgetter(1))
	else:
	    trialevents = trialevents[0]
	print "Found %i trial epoch (stim ON + ITI) events." % len(trialevents)


    if len(trigger_times) > 1 and type(trigger_times[0])==list:
        trigger_times = [item for sublist in trigger_times for item in sublist]
        trigger_times = list(set(trigger_times))
        trigger_times.sort(key=operator.itemgetter(1))
    else:
        trigger_times = trigger_times[0]
    print "Found %i runs (i.e., trigger boundary events)." % len(trigger_times)

    if len(session_info) > 1 and type(session_info[0])==list:
        session_info = [item for sublist in session_info for item in sublist]
        session_info = list(set(trigger_times))
        session_info.sort(key=operator.itemgetter(1))
    else:
        session_info = session_info[0]
    #print "N sessions in file: .", len(session_info)
    print session_info


    # In[10]:

    # on FLASH protocols, first real iamge event is 41
    print "Found %i trials, corresponding to %i TIFFs." % (len(stimevents), len(trigger_times))
    refresh_rate = 60

    # Creat trial-dicts for each trial in run, in format for NDA database:

    if stimtype=='bar':

        nexpected_pixelevents = int(round((1/session_info['target_freq']) * session_info['ncycles'] * refresh_rate * len(trigger_times)))
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
        nexpected_pixelevents = (ntrials * (session_info['stimduration']/1E3) * refresh_rate) + ntrials + 1
        nbitcode_events = sum([len(tr) for tr in dynamic_stim_bitcodes]) + 1 #len(itis) + 1 # Add an extra ITI for blank before first stimulus

        print "Expected %i pixel events, missing %i pevs." % (nexpected_pixelevents, nexpected_pixelevents-nbitcode_events)

        # GET TRIAL INFO FOR DB:
        trial = dict((i+1, dict()) for i in range(len(stimevents)))
        stimevents = sorted(stimevents, key=get_timekey)
        trialevents = sorted(trialevents, key=get_timekey)
        run_start_time = trialevents[0].time
        for trialidx,(stim,iti) in enumerate(zip(sorted(stimevents, key=get_timekey), sorted(post_itis, key=get_timekey))):
            trialnum = trialidx + 1
            # blankidx = trialidx*2 + 1
            trial[trialnum]['start_time_ms'] = round(stim.time/1E3)
            trial[trialnum]['end_time_ms'] = round((iti.time/1E3 + session_info['ITI']))
            if stimtype=='grating':
                ori = stim.value[1]['rotation']
                sf = round(stim.value[1]['frequency'], 2)
                stimname = 'grating-ori-%i-sf-%f' % (ori, sf)
                stimpos = [stim.value[1]['xoffset'], stim.value[1]['yoffset']]
            else:
                # TODO:  fill this out with the appropriate variable tags for RSVP images
                stimname = ''
                stimpos = ''
            stimsize = stim.value[1]['height']
            trial[trialnum]['stimuli'] = {'stimulus': stimname, 'position': stimpos, 'scale': stimsize}
            trial[trialnum]['stim_on_times'] = round((stim.time - run_start_time)/1E3)
            trial[trialnum]['stim_off_times'] = round((iti.time - run_start_time)/1E3)
            trial[trialnum]['all_bitcodes'] = bitcodes_by_trial[trialnum]
            #if stim.value[-1]['name']=='pixel clock':
            trial[trialnum]['stim_bitcode'] = stim.value[-1]['bit_code']
            trial[trialnum]['iti_bitcode'] = iti.value[-1]['bit_code']
	    trial[trialnum]['iti_duration'] = session_info['ITI']


    # save trial info as pkl for easyloading: 
    trialinfo_fn = 'trial_info_%s.pkl' % curr_dfn_base 
    #with open(os.path.join(data_dir, 'mw_data', trialinfo_fn), 'wb') as f:
    with open(os.path.join(paradigm_dir, trialinfo_fn), 'wb') as f:
        pkl.dump(trial, f, protocol=pkl.HIGHEST_PROTOCOL)
        f.close()

    # also save as json for easy reading:
    trialinfo_json = 'trial_info_%s.json' % curr_dfn_base
    with open(os.path.join(paradigm_dir, trialinfo_json), 'w') as f:
        dump(trial, f, indent=4)


    # In[12]:

    print "Each TIFF duration is about (sec): "
    for idx,t in enumerate(trigger_times):
        print idx, (t[1] - t[0])/1E6


    # In[13]:


    # Create "pydict" to store all MW stimulus/trial info in matlab-accessible format for GUI:
    if stimtype=='bar':
        pydict = dict()
        print "Offset between first MW stimulus-display-update event and first SI frame-trigger:"
        for ridx,run in enumerate(stimevents.keys()):
            pydict[run] ={'time': [i[0] for i in stimevents[run].states],                    'pos': stimevents[run].vals,                    'idxs': stimevents[run].idxs,                    'ordernum': stimevents[run].ordernum,                    'MWdur': (stimevents[run].states[-1][0] - stimevents[run].states[0][0]) / 1E6,                    'offset': stimevents[run].states[0][0] - stimevents[run].triggers[0],                    'MWtriggertimes': stimevents[run].triggers}
            print "run %i: %s ms" % (ridx+1, str(pydict[run]['offset']/1E3))

    elif stimtype == 'image':
        # TODO:  need to test this, only debugged with GRATINGS.
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

        stimevents = sorted(stimevents, key=lambda e: e.time)
        trialevents = sorted(trialevents, key=lambda e: e.time)

        all_combos = [(i.value[1]['rotation'], round(i.value[1]['frequency'],1)) for i in stimevents]
        image_ids = sorted(list(set(all_combos))) # should be 35 for gratings -- sorted by orientation (7) x SF (5)


    # In[14]:

    parse_trials = False
    if stimtype != 'bar':
        #t_mw_intervals = np.diff(mw_times)
        parse_trials = True


	# In[15]:

	# Check stimulus durations:
	print len(stimevents)
	iti_events = trialevents[1::2]
	print len(iti_events)

	stim_durs = []
	off_syncs = []
	for idx,(stim,iti) in enumerate(zip(stimevents, iti_events)):
	    stim_durs.append(iti.time - stim.time)
	    if (iti.time - stim.time)<0:
		off_syncs.append(idx)
	print "%i bad sync-ing between stim-onsets and ITIs." % len(off_syncs)

	# PLOT stim durations:
	print "N stim ONs:", len(stim_durs)
	print "min:", min(stim_durs)
	print "max:", max(stim_durs)
	# print len([i for i,v in enumerate(stim_durs) if v>1100000])
	# p1 = figure(title="Stim ON durations (ms)",tools="save, zoom_in, zoom_out, pan",
	#             background_fill_color="white")
	# hist, edges = np.histogram(np.array(stim_durs)/1E6, density=True, bins=100)
	# p1.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
	#         fill_color="#036564", line_color="#033649")
	# show(p1)


	# In[16]:


    if parse_trials is True:
        mw_codes_by_file = []
        mw_times_by_file = []
        mw_trials_by_file = []
        offsets_by_file = []
        runs = dict()
        for idx,triggers in enumerate(trigger_times):
            curr_trialevents = [i for i in trialevents if i.time<=triggers[1] and i.time>=triggers[0]]

            # Get TIME for each stimulus trial start:
            mw_times = np.array([i.time for i in curr_trialevents])

            # Get ID of each stimulus:
            mw_codes = []
            for i in curr_trialevents:
                if len(i.value)>2: # contains image stim
                    if stimtype=='grating':
                        stim_config = (i.value[1]['rotation'], round(i.value[1]['frequency'],1))
                        stim_idx = [gidx for gidx,grating in enumerate(sorted(image_ids)) if grating==stim_config][0]+1
                    else: # static image
                        #TODO:
                        # get appropriate stim_idx that is the equivalent of unique stim identity for images
                        pass
                else:
                    stim_idx = 0
                mw_codes.append(stim_idx)
            mw_codes = np.array(mw_codes)

            # Append list of stimulus times and IDs for each SI file:
            mw_times_by_file.append(mw_times)
            mw_codes_by_file.append(mw_codes)
            mw_trials_by_file.append(curr_trialevents)

            # Calculate offset between first stimulus-update event and first SI-frame trigger:
            curr_offset = mw_times[0] - triggers[0] # time diff b/w detected frame trigger and stim display
            offsets_by_file.append(curr_offset)

            # Make silly dict() to keep organization consistent between event-based experiments, 
            # where each SI file contains multple discrete trials, versus movie-stimuli experiments, 
            # where each SI file contains one "trial" for that movie condition:
            if stimtype != 'bar':
                rkey = 'run'+str(idx)
                runs[rkey] = idx #dict() #i

        print "Average offset between stim update event and frame triggger is: ~%0.2f ms" % float(np.mean(offsets_by_file)/1000.)

    mw_file_durs = [i[1]-i[0] for i in trigger_times]


    # In[17]:


    # Rearrange dicts to match retino structures:
    pydict = dict()

    # Create "pydict" to store all MW stimulus/trial info in matlab-accessible format for GUI:
    if stimtype=='bar':
        print "Offset between first MW stimulus-display-update event and first SI frame-trigger:"
        for ridx,run in enumerate(stimevents.keys()):
            pydict[run] ={'time': [i[0] for i in stimevents[run].states],                    'pos': stimevents[run].vals,                    'idxs': stimevents[run].idxs,                    'ordernum': stimevents[run].ordernum,                    'MWdur': (stimevents[run].states[-1][0] - stimevents[run].states[0][0]) / 1E6,                    'offset': stimevents[run].states[0][0] - stimevents[run].triggers[0],                    'MWtriggertimes': stimevents[run].triggers}
            print "run %i: %s ms" % (ridx+1, str(pydict[run]['offset']/1E3))

    else:
        for ridx,run in enumerate(runs.keys()):
            pydict[run] = {'time': mw_times_by_file[ridx],                        'ordernum': runs[run], 
                            'offset': offsets_by_file[ridx],
                            'stimIDs': mw_codes_by_file[ridx],
                            'MWdur': mw_file_durs[ridx],\
                            'MWtriggertimes': trigger_times[ridx]}

            if stimtype=='grating':
                pydict[run]['idxs'] = [i for i,tmptev in enumerate(mw_trials_by_file[ridx]) if any(['gabor' in v['name'] for v in tmptev.value])],
            else:
                pydict[run]['idxs'] = [i for i,tmptev in enumerate(mw_trials_by_file[ridx]) if any([v['name'] in image_ids for v in tmptev.value])],


    # In[18]:

    # pydict.keys()


    # In[19]:


    # Ignoring ARDUINO stuff for now:
    pydict['stimtype'] = stimtype

    #pydict['mw_times_by_file'] = mw_times_by_file
    #pydict['mw_file_durs'] = mw_file_durs
    #pydict['mw_frame_trigger_times'] = frame_trigger_times
    #pydict['offsets_by_file'] = offsets_by_file
    #pydict['mw_codes_by_file'] = mw_codes_by_file
    pydict['mw_dfn'] = mw_dfn
    pydict['source_dir'] = paradigm_dir #source_dir
    pydict['fn_base'] = curr_dfn_base #fn_base
    pydict['stimtype'] = stimtype
    if stimtype=='bar':
        pydict['condtypes'] = ['left', 'right', 'top', 'bottom']
        pydict['runs'] = stimevents.keys()
        pydict['info'] = session_info
    elif stimtype=='image':
        pydict['condtypes'] = sorted(image_ids)
        pydict['runs'] = runs.keys()
    elif stimtype=='grating':
        pydict['condtypes'] = sorted(image_ids)
        pydict['runs'] = runs.keys()

    tif_fn = curr_dfn_base+'.mat' #fn_base+'.mat'
    print tif_fn
    # scipy.io.savemat(os.path.join(source_dir, condition, tif_fn), mdict=pydict)
    #scipy.io.savemat(os.path.join(data_dir, 'mw_data', tif_fn), mdict=pydict)
    #print os.path.join(data_dir, 'mw_data', tif_fn)
    scipy.io.savemat(os.path.join(paradigm_dir, tif_fn), mdict=pydict)
    print os.path.join(paradigm_dir, tif_fn)

    # Save json:
    pydict_json = curr_dfn_base+'.json'
    with open(os.path.join(paradigm_dir, pydict_json), 'w') as f:
        dump(pydict, f, indent=4)
