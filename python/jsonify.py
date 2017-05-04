#!/usr/bin/env python2

import scipy.io
import os
import uuid

source = '/nas/volume1/2photon/RESDATA/TEFO'
session = '20161219_JR030W'
experiment = 'gratingsFinalMask2'
datastruct_idx = 1

datastruct = 'datastruct_%03d' % datastruct_idx

structpath = os.path.join(source, session, experiment, 'analysis')

dstruct_fn = os.listdir(os.path.join(structpath, datastruct))
dstruct_fn = [f for f in dstruct_fn if f.endswith('.mat')][0]

dstruct = scipy.io.loadmat(os.path.join(structpath, datastruct, dstruct_fn))


# Get all the meta info:
metastruct_fn = os.listdir(os.path.join(structpath, 'meta'))
metastruct_fn = [f for f in metastruct_fn if 'meta' in f and f.endswith('.mat')][0]

metastruct = scipy.io.loadmat(os.path.join(structpath, 'meta', metastruct_fn))

animals = ['R2B1'] #, 'R2B2']
animals_uuid = [str(uuid.uuid4()) for animalidx in range(len(animals))]

receipt_dates = ['2016-12-30']

sessions_by_animal = {
    'R2B1': {'2016-12-19': {'time': '14:00', 'scope': 'tefo'}}, \
    'R2B2': {'2016-12-19': {'time': '12:00', 'scope': 'tefo'}, \
             '2016-12-21': {'time': '20:00', 'scope': 'res'}} \
    }


master = dict()

for aidx,animal in enumerate(animals):
    a_uuid = animals_uuid[aidx]
    
    master[a_uuid] = dict()
 
    master[a_uuid]['id'] = a_uuid
    master[a_uuid]['name'] = animal
    master[a_uuid]['receipt_date'] = receipt_dates[aidx]
    master[a_uuid]['sex'] = 'female'


    sessions = sessions_by_animal[animal]
    sessions_uuid = [str(uuid.uuid4()) for seshidx in range(len(sessions))]
    master[a_uuid]['sessions'] = sessions_uuid
    master[a_uuid]['session_info'] = dict()
 
    for sidx,session_date in enumerate(sessions):
        s_uuid = sessions_uuid[sidx] 
        master[a_uuid]['session_info'][s_uuid] = {\
					    "animal": a_uuid,\
					    "date": session_date,\
					    "start_time": sessions[session_date]['time'],\
					    "microscope": sessions[session_date]['scope']\
					     }
        master[a_uuid]['session_runs'][s_uuid] = sessions[session_date]['runs'] 

