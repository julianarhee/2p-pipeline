#!/usr/bin/env python2
'''
python files_to_trials.py -h for opts.

Requires stimdict containing relevant frame indices for each trial:
    - output of create_stimdict.py
    - <path_to_paradigm_files>/stimdict.pkl (.json)

Requires stimorder file (1 for all files, or 1 for each file):
    - order of stimuli (by IDX) in a given TIFF file
    - output of extract_acquisition_events.py
    - <path_to_paradigm_files>/stimorder_FileXXX.txt (or: stim_order.txt)


Requires tracestruct containing traces (frames_in_files x nrois matrix) for each file:
    - output of get_rois_and_traces step of run_pipeline.m (MATLAB)
    - <path_to_analysis_specific_trace_structs>/traces_SliceXX_ChannelXX.mat

OUTPUTS:
    - stimtraces dicts :  traces-by-roi for each trial (list of arrays) for each stimulus
    - <path_to_analysis_specific_trace_structs>/Parsed/stimtraces_ChannelXX_SliceXX.mat (.json, .pkl)

'''

import os
import json
import re
import datetime
import optparse
import h5py
import pprint
import numpy as np
import cPickle as pkl
import scipy.io
pp = pprint.PrettyPrinter(indent=4)

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]

class StimInfo:
    def _init_(self):
        self.stimid = ''
        self.trials = []
        self.frames = []
        self.frames_sec = []
        self.stim_on_idx = []

def serialize_json(instance=None, path=None):
    dt = {}
    dt.update(vars(instance))

#%%
parser = optparse.OptionParser()

parser.add_option('-R', '--root', action='store', dest='rootdir', default='/nas/volume1/2photon/data', help='data root dir (root project dir containing all animalids) [default: /nas/volume1/2photon/data, /n/coxfs01/2pdata if --slurm]')
parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')

# Set specific session/run for current animal:
parser.add_option('-S', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID')
parser.add_option('-A', '--acq', action='store', dest='acquisition', default='FOV1', help="acquisition folder (ex: 'FOV1_zoom3x') [default: FOV1]")
parser.add_option('-r', '--run', action='store', dest='run', default='', help="name of run dir containing tiffs to be processed (ex: gratings_phasemod_run1)")

parser.add_option('--slurm', action='store_true', dest='slurm', default=False, help="set if running as SLURM job on Odyssey")

parser.add_option('-t', '--trace-id', action="store",
                  dest="trace_id", default='', help="Name of trace extraction set to use.")

(options, args) = parser.parse_args()


# Set USER INPUT options:
rootdir = options.rootdir
animalid = options.animalid
session = options.session
acquisition = options.acquisition
run = options.run
slurm = options.slurm
trace_id = options.trace_id


#%% Load TRACE ID:
try:
    tracedict_path = os.path.join(trace_basedir, 'traceids_%s.json' % run)
    with open(tracedict_path, 'r') as tr:
        tracedict = json.load(tr)
    TID = tracedict[trace_id]
    print "USING TRACE ID: %s" % TID['trace_id']
    pp.pprint(TID)
except Exception as e:
    print "Unable to load TRACE params info: %s:" % trace_id
    print "Aborting with error:"
    print e


#%%

# Load reference info:
run_dir = os.path.join(rootdir, animalid, session, acquisition, run)
runinfo_path = os.path.join(run_dir, '%s.json' % run)

with open(runinfo_path, 'r') as fr:
    runinfo = json.load(fr)
ntiffs = runinfo['ntiffs']
file_names = sorted(['File%03d' % int(f+1) for f in range(nfiles)], key=natural_keys)
nvolumes = runinfo['nvolumes']
frame_idxs = runinfo['frame_idxs']
if len(frame_idxs) == 0:
    # No flyback
    frame_idxs = np.arange(0, nvolumes)


# Get PARADIGM INFO:
# -----------------------------------------------------------------------------
paradigm_outdir = os.path.join(run_dir, 'paradigm')

# Load stimulus dict:
print "Loading parsed trials-to-frames file for run..."
try:
    parsed_fn = [f for f in os.listdir(paradigm_outdir) if 'frames_' in f and f.endswith('hdf5')]
    assert len(parsed_fn)==1, "More than 1 frame-trial file found!"
    paradigm_filepath = os.path.join(paradigm_outdir, parsed_fn[0])
    framestruct = h5py.File(paradigm_filepath, 'r')
    trial_list = sorted(framestruct.keys(), key=natural_keys)
    file_list = sorted(list(set(['File%03d' % int(framestruct[t].attrs['aux_file_idx']+1) for t in trial_list])), key=natural_keys)
    print "Found %i behavior files, with %i trials each. Total %i in run %s." % (len(file_list), len(trial_list)/len(file_list), len(trial_list), run)
except Exception as e:
    print "-------------------------------------------------------------------"
    print "No frame-trial hdf5 file found. Did you run align_acquisition_events.py?"
    print "Aborting with error:"
    print e
    print "-------------------------------------------------------------------"


#%% Load raw traces:
trace_basedir = os.path.join(rootdir, animalid, session, acquisition, run, 'traces')
try:
    trace_name = [t for t in os.listdir(trace_basedir) if trace_id in t and os.path.isdir(os.path.join(trace_basedir, t))][0]
    tracestruct_fns = [f for f in os.listdir(os.path.join(trace_basedir, trace_name, 'extracted')) if f.endswith('hdf5') and 'rawtraces' in f]
    print "Found tracestructs for %i tifs in dir: %s" % (len(tracestruct_fns), trace_name)
    traceid_dir = os.path.join(trace_basedir, trace_name)
    trace_source_dir = os.path.join(traceid_dir, 'extracted')
except Exception as e:
    print "Unable to find extracted tracestructs from trace set: %s" % trace_id
    print "Aborting with error:"
    print e

#%% Get VOLUME indices to align to frame indices:
nslices = len(runinfo['slices'])
nslices_full = int(round(runinfo['frame_rate']/runinfo['volume_rate']))
print "Creating volume index list for %i total slices. %i frames were discarded for flyback." % (nslices_full, nslices_full - nslices)

vol_idxs = np.empty((nvolumes*nslices_full,))
vcounter = 0
for v in range(nvolumes):
    vol_idxs[vcounter:vcounter+nslices_full] = np.ones((nslices_full, )) * v
    vcounter += nslices_full
vol_idxs = [int(v) for v in vol_idxs]

#%%


#%%
# -----------------------------------------------------------------------------
# SPLIT TRACES BY STIMULUS:
# -----------------------------------------------------------------------------
# To look at all traces for ROI 3 for stimulus 1:
# traces_by_stim['1']['Slice01'][:,roi,:]

for stim in sorted(stimdict.keys(), key=natural_keys):
    stimname = 'stim%03d' % int(stim)

    stimtrial = 0
    num_trials_counted = 0
    for fi,currfile in enumerate(sorted(file_names, key=natural_keys)):
        curr_trace_fn = [t for t in trace_fns if currfile in t][0]
        print "Loading traces:", curr_trace_fn
        tracestruct = h5py.File(os.path.join(trace_dir, curr_trace_fn), 'r')
        print "Done."

        # increment stimtrial
        stimtrial += num_trials_counted

        for curr_slice in sorted(tracestruct.keys(), key=natural_keys):
            print curr_slice
            # Hierarchy: /Slice/stimulus/
            if curr_slice not in file_grp.keys():
                slice_grp = file_grp.create_group(curr_slice)
            else:
                slice_grp = file_grp[curr_slice]

            curr_ntrials = len(stimdict[stim][currfile].frames)
            num_trials_counted = 0
            for currtrial_idx in range(curr_ntrials):
                num_trials_counted += 1
                stimtrialname = 'trial%05d' % int(stimtrial + num_trials_counted)
                print stimtrialname

                #volumerate = stimdict[stim][currfile].volumerate
                stim_on_sec = stimdict[stim][currfile].stim_dur #[currtrial_idx]
                nframes_on = stim_on_sec * runinfo['frame_rate'] #volumerate #nt(round(stim_on_sec * volumerate))
                iti_sec = stimdict[stim][currfile].iti_dur
                baseline_sec =  stimdict[stim][currfile].baseline_dur

                # 1. Get absolute frame idxs (frame idx in entire file) for current trial:
                currtrial_frames = stimdict[stim][currfile].frames[currtrial_idx]
                currtrial_frames = [int(i) for i in currtrial_frames]

                # 2. Get frame idx for stim ON:
                nvolumes_on = stim_on_sec * runinfo['volume_rate']
                curr_frame_onset = stimdict[stim][currfile].stim_on_idx[currtrial_idx]
                stim_on_frames = [curr_frame_onset, curr_frame_onset + nframes_on]

                # 3. Get corresponding frame idx matches by slice:
                currtrial_volumes = sorted(list(set([vol_idxs[f] for f in currtrial_frames])))
                curr_volume_onset = vol_idxs[curr_frame_onset]
                stim_on_volumes = [curr_volume_onset, curr_volume_onset + nvolumes_on]

                # 4. Use volume indices to extract traces by slice:
                rawtracemat = tracestruct[curr_slice]['rawtraces'][currtrial_volumes, :]

                if stimname not in slice_grp.keys():
                    stim_grp = slice_grp.create_group(stimname)
                else:
                    stim_grp = file_grp[curr_slice][stimname]

                if stimtrialname not in stim_grp.keys():
                    tset = stim_grp.create_dataset(stimtrialname, rawtracemat.shape, dtype=rawtracemat.dtype)

                tset[...] = rawtracemat

                for infokey in stimdict[stim][currfile].stiminfo.keys():
                    tset.attrs[str(infokey)] = stimdict[stim][currfile].stiminfo[infokey]

                tset.attrs['sourcefile'] = tracestruct.attrs['source_file']
                tset.attrs['trial_in_file'] = stimdict[stim][currfile].trials[currtrial_idx]
                tset.attrs['stim_on_volumes'] = stim_on_volumes
                tset.attrs['trial_volumes'] = currtrial_volumes
                tset.attrs['stim_on_frames'] = stim_on_frames
                tset.attrs['trial_frames'] = currtrial_frames

                tset.attrs['stim_dur'] = stim_on_sec
                tset.attrs['iti_dur'] = iti_sec
                tset.attrs['baseline_dur'] = baseline_sec
                tset.attrs['frame_rate'] = runinfo['frame_rate']
                tset.attrs['volume_rate'] = runinfo['volume_rate']

#%%

#            for curr_slice in sorted(tracestruct.keys(), key=natural_keys):
#                print curr_slice
#                # Hierarchy: /Slice/stimulus/
#                if curr_slice not in file_grp.keys():
#                    slice_grp = file_grp.create_group(curr_slice)
#                if stimname not in slice_grp.keys():
#                    stim_grp = slice_grp.create_group(stimname)
#
#                rawtracemat = tracestruct[curr_slice]['rawtraces'][currtrial_volumes, :]
#
#                if stimtrialname not in stim_grp.keys():
#                    tset = stim_grp.create_dataset(stimtrialname, rawtracemat.shape, dtype=rawtracemat.dtype)
#
#                tset[...] = rawtracemat
#
#                for infokey in stimdict[stim][currfile].stiminfo.keys():
#                    tset.attrs[str(infokey)] = stimdict[stim][currfile].stiminfo[infokey]
#
#                tset.attrs['sourcefile'] = tracestruct.attrs['source_file']
#                tset.attrs['trial_in_file'] = stimdict[stim][currfile].trials[currtrial_idx]
#                tset.attrs['stim_on_volumes'] = stim_on_volumes
#                tset.attrs['trial_volumes'] = currtrial_volumes
#                tset.attrs['stim_on_frames'] = stim_on_frames
#                tset.attrs['trial_frames'] = currtrial_frames
#
#                tset.attrs['stim_dur'] = stim_on_sec
#                tset.attrs['iti_dur'] = iti_sec
#                tset.attrs['baseline_dur'] = baseline_sec
#                tset.attrs['frame_rate'] = runinfo['frame_rate']
#                tset.attrs['volume_rate'] = runinfo['volume_rate']


#        check_stimname = list(set(stimdict[stim][currfile].stiminfo['stimulus']))
#        if len(check_stimname)>1:
#            print "******************************"
#            print "Bad Stim to Trial parsing!."
#            print "------------------------------"
#            print check_stimname
#            print "STIM:", stim, "File:", currfile
#            print "------------------------------"
#            print "Check extract_acquisition_events.py and create_stimdict.py"
#            print "******************************"
#        else:
#            #print check_stimname
#            stimname = check_stimname[0]

#    if stimname not in file_grp.keys():
#        stim_grp = file_grp.create_group(stimname)
#    stim_grp.attrs['ntrials'] = stim_ntrials[stim]
#    stim_grp.attrs['stim_dur'] = stim_on_sec
#    stim_grp.attrs['iti_dur'] = iti_sec
#    stim_grp.attrs['baseline_dur'] = baseline_sec
#    stim_grp.attrs['frame_rate'] = runinfo['frame_rate']
#    stim_grp.attrs['volume_rate'] = runinfo['volume_rate']

#
#    stimtraces[stim]['name'] = stimname
#    stimtraces[stim]['traces'] = np.asarray(curr_traces_allrois)
#    stimtraces[stim]['raw_traces'] = np.asarray(raw_traces_allrois)
#    stimtraces[stim]['df'] = np.asarray(df_allrois)
#    stimtraces[stim]['frames_stim_on'] = stim_on_frames
#    stimtraces[stim]['filesource'] = filesource
#
#   # print stimtraces[stim]['frames_stim_on']
#    stimtraces[stim]['frames'] = np.asarray(curr_frames_allrois)
#    stimtraces[stim]['ntrials'] = stim_ntrials[stim]
#    stimtraces[stim]['nrois'] = nrois
#    stimtraces[stim]['volumerate'] = volumerate
#    stimtraces[stim]['stim_dur'] = stim_on_sec
#    stimtraces[stim]['iti_dur'] = iti_sec


#
#    curr_stimtraces_basename = '%s_stimtraces_%s_%s' % (analysis_id, currslice, currchannel)
#    with open(os.path.join(parsed_traces_dir, '%s.json' % curr_stimtraces_basename), 'w') as f:
#        dump(stimtraces, f, indent=4)
#
#    with open(os.path.join(parsed_traces_dir, '%s.pkl' % curr_stimtraces_basename), 'wb') as f:
#        pkl.dump(stimtraces, f, protocol=pkl.HIGHEST_PROTOCOL)



#
## Split all traces by stimulus-ID:
## ----------------------------------------------------------------------------
#
##stimtraces_all_slices = dict()
#
#for slice_idx,trace_fn in enumerate(sorted(trace_fns_by_slice, key=natural_keys)):
#
#    print "EXTRACING FROM:", trace_fn
#    currslice = "Slice%02d" % int(slices[slice_idx]) # int(slice_idx+1)
#    stimtraces = dict((stim, dict()) for stim in stimdict.keys())
#
#    if pickled_traces is False:
#        tracestruct = loadmat(os.path.join(trace_dir, trace_fn))
#    else:
#        with open(os.path.join(trace_dir, trace_fn), 'rb') as f:
#            tracestruct = pkl.load(f)
#
#    file_names = [tracestruct['file'][fi]['filename'] for fi in range(len(tracestruct['file']))]
#    print "Files included:", file_names
#
#    # To look at all traces for ROI 3 for stimulus 1:
#    # traces_by_stim['1']['Slice01'][:,roi,:]
#    for stim in sorted(stimdict.keys(), key=natural_keys):
#        df_allrois = []
#        raw_traces_allrois = []
#        curr_traces_allrois = []
#        curr_frames_allrois = []
#        stim_on_frames = []
#        filesource = []
#        for fi,currfile in enumerate(sorted(file_names, key=natural_keys)):
##            nframes = int(simeta[currfile]['SI']['hFastZ']['numVolumes'])
##            framerate = float(simeta[currfile]['SI']['hRoiManager']['scanFrameRate'])
##            volumerate = float(simeta[currfile]['SI']['hRoiManager']['scanVolumeRate'])
##            frames_tsecs = np.arange(0, nframes)*(1/volumerate)
#
#            curr_ntrials = len(stimdict[stim][currfile].frames)
#
#            if deconvolved is True and 'deconvolved' in tracestruct['file'][fi].keys():
#                deconvtraces = tracestruct['file'][fi]['deconvolved']
#            if df is True:
#                if 'df' in tracestruct['file'][fi].keys():
#                    dftraces = tracestruct['file'][fi]['df']
#                    #print df.shape
#                else:
#                    dftraces = None
#            #else:
#            #    print "Specified deconv traces, but none found."
#
#            if isinstance(tracestruct['file'][fi], dict):
#                rawtraces = tracestruct['file'][fi]['rawtracemat']
#                currtraces = tracestruct['file'][fi]['tracematDC']
#            else:
#                rawtraces = tracestruct['file'][fi].rawtracemat
#                currtraces = tracestruct['file'][fi].tracematDC
#
#            if df is True and dftraces is not None:
#                if not dftraces.shape[0]==ref['nvolumes']:
#                    dftraces = dftraces.T
#                #print dftraces.shape
#            if not rawtraces.shape[0]==ref['nvolumes']:
#                rawtraces = rawtraces.T
#            if not currtraces.shape[0]==ref['nvolumes']:
#                currtraces = currtraces.T
#            NR = currtraces.shape[1]
##            if not NR==52:
##                continue
#
#            #print currfile, rawtraces.shape, currtraces.shape
#            for currtrial_idx in range(curr_ntrials):
#                volumerate = stimdict[stim][currfile].volumerate
#                stim_on_sec = stimdict[stim][currfile].stim_dur #[currtrial_idx]
#                nframes_on = stim_on_sec * volumerate #nt(round(stim_on_sec * volumerate))
#                iti_sec = stimdict[stim][currfile].iti_dur
#
#                #print stimdict[stim][currfile].frames[currtrial_idx]
#                currtrial_frames = stimdict[stim][currfile].frames[currtrial_idx]
#                currtrial_frames = [int(i) for i in currtrial_frames]
#                #print "CURRTRIAL FRAMES:", currtrial_frames
#                #print len(currtrial_frames)
#
#                # .T to make rows = rois, cols = frames
#                nframes = currtraces.shape[0]
#                nrois = currtraces.shape[1]
#                #print nframes, nrois, currtrial_frames.shape
#                #print currtraces.shape
#                raw_traces_allrois.append(rawtraces[currtrial_frames, :])
#                curr_traces_allrois.append(currtraces[currtrial_frames, :])
#                curr_frames_allrois.append(currtrial_frames)
#
#                if df is True and dftraces is not None:
#                    df_allrois.append(dftraces[currtrial_frames,:])
#
#                 #print stimdict[stim][currfile].stim_on_idx
#                curr_frame_onset = stimdict[stim][currfile].stim_on_idx[currtrial_idx]
#                stim_on_frames.append([curr_frame_onset, curr_frame_onset + nframes_on])
#
#                filesource.append(currfile)
#
#            check_stimname = list(set(stimdict[stim][currfile].stimid))
#            if len(check_stimname)>1:
#                print "******************************"
#                print "Bad Stim to Trial parsing!."
#                print "------------------------------"
#                print check_stimname
#                print "STIM:", stim, "File:", currfile
#                print "------------------------------"
#                print "Check extract_acquisition_events.py and create_stimdict.py"
#                print "******************************"
#            else:
#                #print check_stimname
#                stimname = check_stimname[0]
#
#
#
#        stimtraces[stim]['name'] = stimname
#        stimtraces[stim]['traces'] = np.asarray(curr_traces_allrois)
#        stimtraces[stim]['raw_traces'] = np.asarray(raw_traces_allrois)
#        stimtraces[stim]['df'] = np.asarray(df_allrois)
#        stimtraces[stim]['frames_stim_on'] = stim_on_frames
#        stimtraces[stim]['filesource'] = filesource
#
#       # print stimtraces[stim]['frames_stim_on']
#        stimtraces[stim]['frames'] = np.asarray(curr_frames_allrois)
#        stimtraces[stim]['ntrials'] = stim_ntrials[stim]
#        stimtraces[stim]['nrois'] = nrois
#        stimtraces[stim]['volumerate'] = volumerate
#        stimtraces[stim]['stim_dur'] = stim_on_sec
#        stimtraces[stim]['iti_dur'] = iti_sec
#
#
#
#    curr_stimtraces_basename = '%s_stimtraces_%s_%s' % (analysis_id, currslice, currchannel)
#    with open(os.path.join(parsed_traces_dir, '%s.json' % curr_stimtraces_basename), 'w') as f:
#        dump(stimtraces, f, indent=4)
#
#    with open(os.path.join(parsed_traces_dir, '%s.pkl' % curr_stimtraces_basename), 'wb') as f:
#        pkl.dump(stimtraces, f, protocol=pkl.HIGHEST_PROTOCOL)
#
#
#    # save as .mat:
#    stimtraces_mat = dict()
#    for stim in sorted(stimdict.keys(), key=natural_keys):
#        currstim = "stim%02d" % int(stim)
#        #print currstim
#        stimtraces_mat[currstim] = stimtraces[stim]
#
#    scipy.io.savemat(os.path.join(parsed_traces_dir, '%s.mat' % curr_stimtraces_basename), mdict=stimtraces_mat)

