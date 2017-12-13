###!/usr/bin/env python2
import os
import json
import re
import scipy.io as spio
import numpy as np
from json_tricks.np import dump, dumps, load, loads
from mat2py import loadmat
import cPickle as pkl
import operator

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]

class StimInfo:
    def __init__(self):
        self.stimid = []
        self.trials = []
        self.frames = []
        self.frames_sec = []
        self.stim_on_idx = []
	self.stim_dur = None #[]
	self.iti_dur = None #[]
	self.volumerate = None

def serialize_json(instance=None, path=None):
    dt = {}
    dt.update(vars(instance))
    return dt


# source = '/nas/volume1/2photon/projects'
# experiment = 'scenes'
# session = '20171003_JW016'
# acquisition = 'FOV1'
# functional_dir = 'functional'
# mw = False

# stim_on_sec = float(options.stim_on_sec) #2. # 0.5
## iti_pre = float(options.iti_pre)
# same_order = False #True

import optparse

parser = optparse.OptionParser()

parser.add_option('-R', '--root', action='store', dest='rootdir', default='/nas/volume1/2photon/data', help='data root dir (root project dir containing all animalids) [default: /nas/volume1/2photon/data, /n/coxfs01/2pdata if --slurm]')
parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')

# Set specific session/run for current animal:
parser.add_option('-S', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID')
parser.add_option('-A', '--acq', action='store', dest='acquisition', default='FOV1', help="acquisition folder (ex: 'FOV1_zoom3x') [default: FOV1]")
parser.add_option('-r', '--run', action='store', dest='run', default='', help="name of run dir containing tiffs to be processed (ex: gratings_phasemod_run1)")

parser.add_option('--slurm', action='store_true', dest='slurm', default=False, help="set if running as SLURM job on Odyssey")


parser.add_option('--custom', action="store_true",
                  dest="custom_mw", default=False, help="Not using MW (custom params must be specified)")
parser.add_option('--order', action="store_true",
                  dest="same_order", default=False, help="Set if same stimulus order across all files (1 stimorder.txt)")

parser.add_option('-p', '--pre', action="store",
                  dest="iti_pre", default=1.0, help="Time (s) pre-stimulus to use for baseline. [default: 1.0]")

# parser.add_option('-i', '--iti', action="store",
#                   dest="iti_full", default='', help="Time (s) between stimuli (inter-trial interval).")


# Only need to set these if using custom-paradigm file:
parser.add_option('-O', '--stimon', action="store",
                  dest="stim_on_sec", default=0, help="Time (s) stimulus ON.")

parser.add_option('-t', '--vol', action="store",
                  dest="vols_per_trial", default=0, help="Num volumes per trial. Specifiy if custom_mw=True")
parser.add_option('-v', '--first', action="store",
                  dest="first_stim_volume_num", default=0, help="First volume stimulus occurs (py-indexed). Specifiy if custom_mw=True")


parser.add_option('--flyback', action="store_true",
                  dest="flyback_corrected", default=False, help="Set if corrected extra flyback frames (in process_raw.py->correct_flyback.py")


(options, args) = parser.parse_args()

flyback_corrected = options.flyback_corrected

(options, args) = parser.parse_args()


# Set USER INPUT options:
rootdir = options.rootdir
animalid = options.animalid
session = options.session
acquisition = options.acquisition
run = options.run

custom_mw = options.custom_mw
same_order = options.same_order #False #True
vols_per_trial = int(options.vols_per_trial)
first_stim_volume_num = int(options.first_stim_volume_num)

abort = False
if custom_mw is True and (vols_per_trial==0 or first_stim_volume_num==0):
    print "Must set N vols-per-trial and IDX of first volume stimulus begins."
    abort = True

if abort is False:

    run_dir = os.path.join(rootdir, animalid, session, acquisition, run)

    # Load reference info:
    runinfo_path = os.path.join(run_dir, '%s.json' % run)
    with open(runinfo_path, 'r') as fr:
        runinfo = json.load(fr)
    nfiles = runinfo['ntiffs']
    file_names = sorted(['File%03d' % int(f+1) for f in range(nfiles)], key=natural_keys)

    frame_idxs = runinfo['frame_idxs']
    if len(frame_idxs) > 0:
        print "Found %i frames from flyback correction." % len(frame_idxs)
    else:
        frame_idxs = np.arange(0, runinfo['nvolumes'])

#    if flyback_corrected is True:
#        frame_idxs = runinfo['frame_idxs']
#        print "Found %i frames from flyback correction." % len(frame_idxs)

    ntiffs = runinfo['ntiffs']
    file_names = sorted(['File%03d' % int(f+1) for f in range(ntiffs)], key=natural_keys)
    volumerate = runinfo['volume_rate']

    #vols_per_trial = float(options.vols_per_trial)
    #first_stim_volume_num = int(options.first_stim_volume_num)

    iti_pre = float(options.iti_pre)

    if custom_mw is True:
#        with open(runinfo['raw_simeta_path'], 'r') as f:
#            simeta = json.load(f)
#        volumerate = float(simeta['File001']['SI']['hRoiManager']['scanVolumeRate'])

        stim_on_sec = float(options.stim_on_sec) #2. # 0.5

        first_stimulus_volume_num = int(options.first_stim_volume_num) #50
        vols_per_trial = float(options.vols_per_trial) # 15
        #iti_full = (vols_per_trial - (stim_on_sec * volumerate)) / volumerate
        iti_full = (vols_per_trial/volumerate) - stim_on_sec
        iti_post = iti_full - iti_pre
        print "First stim on:", first_stimulus_volume_num
        print "Volumes per trial:", vols_per_trial
        print "ITI POST (s):", iti_post
        print "ITT full (s):", iti_full
        print "TRIAL dur (s):", stim_on_sec + iti_full
        print "Vols per trial (calc):", (stim_on_sec + iti_pre + iti_post) * volumerate
#     else:
#         stim_on_sec = float(options.stim_on_sec) #2. # 0.5
#         iti_full = float(options.iti_full)# 4.
#         iti_post = iti_full - iti_pre
#         print "ITI POST:", iti_post

    # =================================================================================


    ### Get PARADIGM INFO:
    paradigm_outdir = os.path.join(run_dir, 'paradigm')
    if custom_mw is False:
        with open(os.path.join(paradigm_outdir, 'parsed_trials.pkl'), 'rb') as f:
            trialdict = pkl.load(f)
        print "Trial Info dicts found for %i files:" % len(trialdict.keys())

    ### Get stim-order files:
    stimorder_fns = sorted([f for f in os.listdir(paradigm_outdir) if 'stimorder' in f or 'stim_order' in f])
    print "Found %i stim-order files, and %i TIFFs." % (len(stimorder_fns), nfiles)
    if len(stimorder_fns) < nfiles:
        if same_order:
            # Same stimulus order for each file (complete set)
            stimorder_fns = np.tile(stimorder_fns, [nfiles,])


    # =================================================================================
    # Create stimulusdict:
    # =================================================================================

    # stimdict[stim][currfile].trials
    # stimdict[stim][currfile].frames
    # stimdict[stim][currfile].frames_sec
    # stimdict[stim][currfile].stim_on_idx

    stimdict = dict()
    for fi in range(nfiles):
        currfile= "File%03d" % int(fi+1)

#        nvolumes = int(simeta[currfile]['SI']['hFastZ']['numVolumes'])
#        #nslices = len(ref['slices']) #int(simeta[currfile]['SI']['hFastZ']['numVolumes'])
#        nslices = int(simeta[currfile]['SI']['hFastZ']['numFramesPerVolume'])
#        framerate = float(simeta[currfile]['SI']['hRoiManager']['scanFrameRate'])
#        volumerate = float(simeta[currfile]['SI']['hRoiManager']['scanVolumeRate'])
#        	#print "framerate:", framerate
#        	#print "volumerate:", volumerate

        nvolumes = runinfo['nvolumes']
        nslices = int(len(runinfo['slices']))
        volumerate = runinfo['volume_rate']
        frames_tsecs = np.arange(0, nvolumes)*(1/volumerate)

        # Load stim-order:
        stim_fn = stimorder_fns[fi] #'stim_order.txt'
        with open(os.path.join(paradigm_outdir, stim_fn)) as f:
            stimorder = f.readlines()
        curr_stimorder = [l.strip() for l in stimorder]
        unique_stims = sorted(set(curr_stimorder), key=natural_keys)

        for trialnum,stim in enumerate(curr_stimorder):
            currtrial = str(trialnum+1)

            if custom_mw is False:
                stim_on_sec = round(trialdict[currfile][currtrial]['stim_dur_ms']/1E3)
                iti_full = round(trialdict[currfile][currtrial]['iti_dur_ms']/1E3)
                iti_post = iti_full - iti_pre

            nframes_on = stim_on_sec * volumerate #int(round(stim_on_sec * volumerate))
            nframes_iti_pre = iti_pre * volumerate
            nframes_iti_post = iti_post*volumerate # int(round(iti_post * volumerate))
            nframes_iti_full = iti_full * volumerate #int(round(iti_full * volumerate))
            nframes_post_onset = (stim_on_sec + iti_post) * volumerate

            if not stim in stimdict.keys():
                stimdict[stim] = dict()
            if not currfile in stimdict[stim].keys():
                stimdict[stim][currfile] = StimInfo()

            if custom_mw is True:
                if trialnum==0:
                    first_frame_on = first_stimulus_volume_num
                else:
                    first_frame_on += vols_per_trial
            else:
                #first_frame_on = int(round(trialdict[currfile][currtrial]['stim_on_idx']/nslices))
                first_frame_on = int(trialdict[currfile][currtrial]['stim_on_idx'])

                if flyback_corrected is True:
                    if first_frame_on in frame_idxs:
                        first_frame_on = frame_idxs.index(first_frame_on)
                    else:
                        if first_frame_on+1 in frame_idxs:
                            first_frame_on = frame_idxs.index(first_frame_on+1)
                        elif first_frame_on-1 in frame_idxs:
                            first_frame_on = frame_idxs.index(first_frame_on-1)
                        else:
                            print "NO match found for FIRST frame ON:", first_frame_on

            preframes = list(np.arange(int(first_frame_on - nframes_iti_pre), first_frame_on, 1))
            postframes = list(np.arange(int(first_frame_on + 1), int(round(first_frame_on + nframes_post_onset))))

            framenums = [preframes, [first_frame_on], postframes]
            framenums = reduce(operator.add, framenums)
            #print "POST FRAMES:", len(framenums)
            diffs = np.diff(framenums)
            consec = [i for i in np.diff(diffs) if not i==0]

            if len(consec)>0:
                print "BAD FRAMES:", trialnum, stim, framenums

            if custom_mw is True:
                stimname = 'stimulus%02d' % int(stim)
            else:
                stimname = trialdict[currfile][currtrial]['name']


# 	    if flyback_corrected is True:
#                 for f in framenums:
#                     if f in frame_idxs:
#                         match = frame_idxs.index(f)
#                     else:
#                         if f-1 in frame_idxs:
#                             match = frame_idxs.index(f-1)
#                         elif f+1 in frame_idxs:
#                             match = frame_idxs.index(f+1)
#                         else:
#                             print "NO MATCH FOUND for frame:", f
#                             #framenums = [frame_idxs.index(f) for f in framenums]
#
#                     if first_frame_on in frame_idxs:
#                         first_frame_on = frame_idxs.index(first_frame_on)
#                     else:
#                         if first_frame_on-1 in frame_idxs:
#                             first_frame_on = frame_idxs.index(first_frame_on-1)
#                         elif first_frame_on+1 in frame_idxs:
#                                     first_frame_on = frame_idxs.index(first_frame_on+1)
#                         else:
#                             print "NO match found for FIRST frame ON:", first_frame_on
#
            #print "sec to plot:", len(framenums)/volumerate

            #frametimes = [frames_tsecs[f] for f in framenums]

            stimdict[stim][currfile].stimid.append(stimname) #trialdict[currfile][currtrial]['name'])
            stimdict[stim][currfile].trials.append(trialnum)
            stimdict[stim][currfile].frames.append(framenums)
            #stimdict[stim][currfile].frames_sec.append(frametimes)
            stimdict[stim][currfile].stim_on_idx.append(first_frame_on)
            stimdict[stim][currfile].stim_dur = round(stim_on_sec) #.append(stim_on_sec)
            stimdict[stim][currfile].iti_dur = round(iti_full) #.append(iti_full)
            stimdict[stim][currfile].volumerate = volumerate

        #print [len(stimdict[stim][currfile].frames[i]) for i in range(len(stimdict[stim][currfile].frames))]

    # Save to PKL:
    curr_stimdict_pkl = 'stimdict.pkl' #% currfile # % currslice
    print curr_stimdict_pkl
    with open(os.path.join(paradigm_outdir, curr_stimdict_pkl), 'wb') as f:
        pkl.dump(stimdict, f, protocol=pkl.HIGHEST_PROTOCOL) #, f, indent=4)

    # Save to JSON:
    for fi in range(nfiles):
        currfile = "File%03d" % int(fi+1)
        for stim in stimdict.keys():
            stimdict[stim][currfile] = serialize_json(stimdict[stim][currfile])

    curr_stimdict_json = 'stimdict.json' #% currfile # % currslice
    print curr_stimdict_json
    with open(os.path.join(paradigm_outdir, curr_stimdict_json), 'w') as f:
        dump(stimdict, f, indent=4)


