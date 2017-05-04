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
import scipy.io as spio

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
		    'runs': run_list\
		   }

    return session_info


def get_run_info(s_uuid, run_name, runmeta, run_num=0):
    r_uuid = str(uuid.uuid4())
    # There can be "sub-runs" within a run (for ex., left_run1 and left_run2 could be two runs within retinotopy037Hz run
    # todo:  make sure all sessionmeta run names match the assigned "run names" within pymw structs.
#    if len(runmeta['file']) > 1:
#        tiffs = [(tidx, str(i.mw.runName)) for tidx,i in enumerate(runmeta['file'])]
#        # May not need this, is confusing -- within a "run" there may be multiple files, which are also called "runs" upstream.
#        # For event-related runs, there may be multiple TIFFS, each of which contain multiple trials
#        # For continuous runs, there is one tiff per trial.
#        tiffidx_in_run = [i for i,tiff in enumerate(tiffs) if tiff[0]==run_name][0]
#    else:
#        tiffidx_in_run = 0
#
    tiffs = [(tidx, str(i.mw.runName)) for tidx,i in enumerate(runmeta['file'])]
    
    run_info = {'id': r_uuid,\
                'session': s_uuid,\
                'run_name': run_name,\
                'tiffs': [(i[1], runmeta['file'][i[0]].mw.orderNum) for i in tiffs],\
                'run_number': str(run_num),\
                'imaging_fov_um':  str(runmeta.volumesize),\
                'vol_rate': str(runmeta.volumerate),\
                #'trials': trial_uuids\
              }

    return run_info


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

    run_info[run] = get_run_info(session_info['id'], run, runmeta, runidx)
 

# Cell Info:
datastruct = 'datastruct_%03d' % datastruct_idx
dstruct_fn = os.listdir(os.path.join(runpath,'analysis', datastruct))
dstruct_fn = [f for f in dstruct_fn if f.endswith('.mat')][0]

dstruct = loadmat(os.path.join(runpath, 'analysis', datastruct, dstruct_fn))


outputpath = dstruct['outputPath']
def get_trials(r_uuid, run_name, runmeta, outputpath):

    if len(runmeta['file']) > 1:
        tiffs = [(tidx, str(i.mw.runName)) for tidx,i in enumerate(runmeta['file'])]
        # May not need this, is confusing -- within a "run" there may be multiple files, which are also called "runs" upstream.
        # For event-related runs, there may be multiple TIFFS, each of which contain multiple trials
        # For continuous runs, there is one tiff per trial.
        tiffidx_in_run = [i for i,tiff in enumerate(tiffs) if tiff[0]==run_name][0]
    else:
        tiffidx_in_run = 0

   
    # Get all trial info, if nec:
    if runmeta['stimType']=='bar':
        # just need to count each file in current run.
        for tiff in tiffs:
            trials[tiff[1]] = uuid.uuid4(); # trials will be a dict():  {'left': 'trial1_uuid', 'right': 'trial2_uuid', 'top1': 'trial3_uuid', 'top2': 'trial4_uuid'...}

    else:
        # Need to cycle into nested trialstuct and get all trials:    
        trialstruct = loadmat(os.path.join(outputpath, 'stimReps.mat')) # mat structs saved as stuct or '-v7.3' need to be read with h5py
        stims = dir(trialstruct['slice'][10].info)
        stimnames = [n for n in stims if 'stim' in n]
        for stim in stimnames:
            # TO GET IDS for each trial:
            # trialstruct['slice'][10].info.stim5[0].trialIdxInRun, for trials 0:nTrials of current stim

            trials[stim] = trialstruct['slice'][10].info.stim[0].trialIdxInRun







cell_info = get_cell_info(run_info, dstruct)


def get_cell_info(run_info, dstruct):
    nrois = dstruct['nrois'] # todo:  this assumes that ALL runs have the same ROIs (which we want eventually, but need alignment for)
    maskmat = scipy.io.loadmat(dstruct['masks3D']) # todo: save path to 3D-coord masks (saved as cell array in matlab?)
    masks = dict((k, v) for k,v in maskmat['maskcell3d'].iteritems())  # todo:  haven't created this, so not sure if this is correct call

    # todo:  combine ALL slice rois and convert to master list of 3D coords: (0, (x0, y0, z0)), (1, (x1,y1,z1), ... (N, (xN, yN, zN))
    cell_info = dict.fromkeys([i[0] for i in dstruct['centroids']], dict()) 
    tmp = dict((k, {'id': str(uuid.uuid4())}) for k,v in cell_info.iteritems())
    cell_info.update(tmp)
    for k in cells_info.keys():
        cell_info[k].update(run=run_info.keys())
        cell_info[k].update(mask=masks[k])

    return cell_info


def get_trial_info(run, trial):
    


# sessions_by_animal = {
#    'R2B1': {'2016-12-19': {'time': '14:00',\
#                            'scope': 'tefo',\
#                            'runs': ['fov2_bar5', 'fov2_bar6']\
#                          }
#           },\
#    'R2B2': {'2016-12-19': {'time': '12:00',\
#                            'scope': 'tefo',\
#                            'runs': ['retinotopyFinal', 'retinotopyFinalMask', 'gratingsFinalMask2', 'rsvpFinal2']\
#                          },\
#             '2016-12-21': {'time': '20:00',\
#                            'scope': 'res',\
#                            'runs': ['retinotopy037Hz', 'rsvp1', 'rsvp2', 'rsvp3']\
#                          }
#           }\
#    }


