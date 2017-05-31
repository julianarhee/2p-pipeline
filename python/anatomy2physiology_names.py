#!/usr/bin/env python2

import os
import cPickle as pkl
import json


dictpath = '/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/dicts/final'

target_uuid = 'fcdee2c3-99e9-45dd-8728-fe31730211c5'


neuronmap_fn = 'neuronID_map_%s.pkl' % target_uuid 
savepath = os.path.join(dictpath, neuronmap_fn)


# Mapping betweeen reconstuction IDs and cellIDs from EM-TEFO alignment:
namedict_path = '/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/em7_centroids/neuron-locations.json'

with open(namedict_path) as f:
    names = json.load(f)

matches = [(c['cell_id'], c['neuron_id']) for c in names]


# cell_info dict (.pkl) that keys everything with c_uuids, and has 'cell_idx' as a key containing the EM-TEFO alignment name:
cellinfo_fn = 'cell_info_%s.pkl' % target_uuid
ndacelldict_path = os.path.join(dictpath, cellinfo_fn)

with open(ndacelldict_path, 'rb') as f:    
    cellinfo = pkl.load(f)


neuronID_map = dict((cuuid, {'cell_name': cellinfo[cuuid]['cell_idx'], 'neuron_id': matches['cell%04d' % cellinfo[cuuid]['cell_idx']]}) for cuuid in cellinfo.keys() if 'cell%04d' % cellinfo[cuuid]['cell_idx'] in cellids)


with open(savepath, 'wb') as f:
    pkl.dump(neuronID_map, f, protocol=pkl.HIGHEST_PROTOCOL)

