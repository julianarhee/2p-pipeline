#!/usr/bin/env python2

import scipy.io
import os
import uuid
import numpy as np

# optparse for user-input:
source = '/nas/volume1/2photon/RESDATA/TEFO'
session = '20161219_JR030W'
#experiment = 'gratingsFinalMask2'
datastruct_idx = 1
animal = 'R2B1'
receipt_date = '2016-12-30'

# todo:  parse everything by session, instead of in bulk (i.e., all animals)...
# animals = ['R2B1', 'R2B2']
# receipt_dates = ['2016-12-20', '2016-12-30']

# Need better loadmat for dealing with .mat files in a non-annoying way:

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    
    return _check_keys(data)


def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    
    return dict        


def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        elif isinstance(elem,np.ndarray):
            dict[strg] = _tolist(elem)
        else:
            dict[strg] = elem

    return dict


def _tolist(ndarray):
    '''
    A recursive function which constructs lists from cellarrays 
    (which are loaded as numpy ndarrays), recursing into the elements
    if they contain matobjects.
    '''
    elem_list = []            
    for sub_elem in ndarray:
        if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
            elem_list.append(_todict(sub_elem))
        elif isinstance(sub_elem,np.ndarray):
            elem_list.append(_tolist(sub_elem))
        else:
            elem_list.append(sub_elem)

    return elem_list


# ----------------------------------------------------------------------------
# Get DICTS that parse existing (kinda) data analysis info from .pkl or.mats
# ----------------------------------------------------------------------------

def get_animal_info(animal, receipt_date='0000-00-00', sex='female'):
    animal_info = dict()
    a_uuid = str(uuid.uuid4())
    animal_info[a_uuid] = {'id': a_uuid,\
			   'name': animal,\
			   'receipt_date': receipt_date,\
			   'sex': sex\
			 }

    return animal_info


def get_session_info(a_uuid, date='0000-00-00', start_time='00:00:00', run_list=[], microscope={'scope': 'tefo', 'rev': 'rev_uuid'}):
    session_info = dict()
    s_uuid = str(uuid.uuid4())
    session_info = {'id': s_uuid,\
		    'animal':  a_uuid,\
		    'date': date,\
		    'start_time': start_time,\
		    'microscope': microscope['scope'],\
		    'microscope_rev': microscope['rev'],\
		    'runs': [str(r) for r in run_list]\
		   }

    return session_info


def get_run_info(s_uuid, run_name, runmeta, run_num=0):

    r_uuid = str(uuid.uuid4())
    # a given run may have 1 or more tiff files associated with it.
    # for continuous stimuli (retinotopic mapping), 1 tiff = 1 trial.
    # for event-related (static stimulus shown for some x seconds), 1 tiff contains multiple trials. 
 
    # tiffs = [(tidx, str(i.mw.runname)) for tidx,i in enumerate(runmeta['file'])]
    if type(runmeta['file'])==dict:
        # then, only 1 file, and immediate keys are 'mw' and 'si' (i.e., don't need to index into files)
        tiffs = [(str(runmeta['file']['mw']['runName']), 0)]
    else:
        tiffs = [(str(runmeta['file'][tiffidx].mw.runName), runmeta['file'][tiffidx].mw.orderNum) for tiffidx in range(len(runmeta['file']))] 
   
    run_info = {'id': r_uuid,\
                'session': s_uuid,\
                'run_name': run_name,\
                'tiffs': tiffs,\
                'run_number': str(run_num),\
                'imaging_fov_um': str(runmeta['volumesize']),\
                'stages_um': str(runmeta['stages_um']),\
                'vol_rate': str(runmeta['volumerate'])\
              }

    return run_info


def get_trials(r_uuid, run_name, tiffs, runmeta, outputpath):

    trials = dict()

   # get all trial info:
    if runmeta['stimType']=='bar':
        # just need to count each file in current run.
        for tiff in tiffs:
            # trials will be a dict():  
            # {'left': 'trial1_uuid', 'right': 'trial2_uuid', 'top1': 'trial3_uuid', 'top2': 'trial4_uuid'...}
            trials[tiff[0]] = str(uuid.uuid4()); 
    else:
        # need to cycle into nested trialstuct and get all trials:    
        trialstruct = loadmat(os.path.join(outputpath, 'stimreps.mat')) # mat structs saved as stuct or '-v7.3' need to be read with h5py (just resaved without '-v7.3' handles in psth.m to avoid this)
        stims = dir(trialstruct['slice'][10].info) # slice index does not matter here, since will be the same for all slices.
        stimnames = [n for n in stims if 'stim' in n]
        for stim in stimnames:
            # to get ids for each trial:
            # trialstruct['slice'][10].info.stim5[0].trialidxinrun, for trials 0:ntrials of current stim
            trials[stim] = dict() 
            trials[stim]['trialidx_in_run'] = [eval("trialstruct['slice'][10].info.%s[%i].trialIdxInRun" % (stim, trial)) for trial in range(len(eval("trialstruct['slice'][10].info.%s" % stim)))]
            trials[stim]['trial_uuid'] = str(uuid.uuid4())

    return trials


def get_cell_info(run_info, dstruct):
    nrois = dstruct['nrois'] # todo:  this assumes that ALL runs have the same ROIs (which we want eventually, but need alignment for)
    maskmat = loadmat(dstruct['masks3D']) # todo: save path to 3D-coord masks (saved as cell array in matlab?)
    masks = dict((k, v) for k,v in maskmat['maskcell3d'].iteritems())  # todo:  haven't created this, so not sure if this is correct call

    # todo:  combine ALL slice rois and convert to master list of 3D coords: (0, (x0, y0, z0)), (1, (x1,y1,z1), ... (N, (xN, yN, zN))
    cell_info = dict.fromkeys([i[0] for i in dstruct['centroids']], dict()) 
    tmp = dict((k, {'id': str(uuid.uuid4())}) for k,v in cell_info.iteritems())
    cell_info.update(tmp)
    for k in cells_info.keys():
        cell_info[k].update(run=run_info.keys())
        cell_info[k].update(mask=masks[k])

    return cell_info

# ----------------------------------------------------------------------------
# TEST OUTPUT: 
# ----------------------------------------------------------------------------



# Animal Info:
# ----------------------------------------------------------------------------
# Animal info (and subsequent sections) will be defined by whether this
# script should be run on a per-animal or per-animal-per-session basis.

master = get_animal_info(animal, receipt_date)


# Session Info: 
# ----------------------------------------------------------------------------
# There can (but need not be) multiple sessions per animal. Each session can 
# be identified by some combination of date and microscope type.

sessionpath = os.path.join(source, session)

# Get all the meta info:
sessionmeta_fn =  os.listdir(sessionpath)
sessionmeta_fn = [f for f in sessionmeta_fn if 'sessionmeta' in f and f.endswith('.mat')][0]

sessionmeta = loadmat(os.path.join(sessionpath, sessionmeta_fn))

date = sessionmeta['date']
start_time = sessionmeta['time']
microscope = sessionmeta['scope']
run_list = sessionmeta['runs']

a_uuid = master.keys()[-1]
session_info =  get_session_info(master[a_uuid], date, start_time, run_list, microscope) 


# Run Info:
# ----------------------------------------------------------------------------
# There can (but need not be) multiple runs per session. A given run may have
# a single-trial or multi-trial format.

run_info = dict()
for runidx,run in enumerate(session_info['runs']):
    runpath = os.path.join(source, session, run)
    runmeta_fn = os.listdir(os.path.join(runpath, 'analysis', 'meta'))
    runmeta_fn = [f for f in runmeta_fn if 'meta' in f and f.endswith('.mat')][0]
    runmeta = loadmat(os.path.join(runpath, 'analysis', 'meta', runmeta_fn))
    
    # todo:  FIX createMetaStruct.m s.t. new fields are saved:
    runmeta['volumesize'] = [500, 500, 210]
    runmeta['volumerate'] = 4.11
    runmeta['stages_um'] = [12568, 52037, 59050] # THIS IS MANUAL ENTRY (not incorp. in SI & pipeline yet)
    
    run_info[run] = get_run_info(session_info['id'], run, runmeta, runidx)
 

# Cell / Trial Info:
# ----------------------------------------------------------------------------
# All runs should have the same ROIs, so that comparing cell activity across
# different runs is possible.  However, for now, am assuming that we're only
# grabbing cell info (time courses, trials, etc.) within a given run, so we
# grab all that info by providing some run.

curr_run = run_info.keys()[0] # Just choose this one for now, but should be in argsin.

r_uuid = run_info[curr_run]['id']
run_name = run_info[curr_run]['run_name']
tiff_list = run_info[curr_run]['tiffs']


# -- Load a specific analysis info/structs (group of mat files indexed by
# datastruct_idx for specific run:

# todo:  We don't know which analysis-method / roi-segmentation we are using
# yet, so for now, just hard-code didxs that are reasonable to use for testing
# a set of runs:
didxs = dict()
didxs['retinotopyFinal'] = 1
didxs['retinotopyFinalMask'] = 2
didxs['gratingsFinalMask2'] = 1

runpath = os.path.join(source, session, curr_run)
runmeta_fn = os.listdir(os.path.join(runpath, 'analysis', 'meta'))
runmeta_fn = [f for f in runmeta_fn if 'meta' in f and f.endswith('.mat')][0]
runmeta = loadmat(os.path.join(runpath, 'analysis', 'meta', runmeta_fn))

datastruct = 'datastruct_%03d' % datastruct_idx
dstruct_fn = os.listdir(os.path.join(runpath,'analysis', datastruct))
dstruct_fn = [f for f in dstruct_fn if f.endswith('.mat')][0]

dstruct = loadmat(os.path.join(runpath, 'analysis', datastruct, dstruct_fn))

outputpath = dstruct['outputDir']

trials = get_trials(r_uuid, run_name, tiff_list, runmeta, outputpath)


# cell_info = get_cell_info(run_info, dstruct)




