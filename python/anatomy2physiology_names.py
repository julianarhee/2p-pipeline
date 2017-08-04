!/usr/bin/env python2

import os
import cPickle as pkl
import json


dictpath = '/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/dicts/final'

target_uuid = 'fcdee2c3-99e9-45dd-8728-fe31730211c5'


neuronmap_fn = 'expressing_neuronID_map_%s.pkl' % target_uuid 
savepath = os.path.join(dictpath, neuronmap_fn)


# Mapping betweeen reconstuction IDs and cellIDs from EM-TEFO alignment:
namedict_path = '/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/em7_centroids/neuron-locations.json'

with open(namedict_path) as f:
    names = json.load(f)

#matches = [(c['cell_id'], c['neuron_id']) for c in names]
matches = dict((c['cell_id'], c['neuron_id']) for c in names)

# use bootstrapping to find "true" neurons (matlab, GUI branch):
path_to_saved_cell_ids = '/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/em7_centroids/expressingCells.mat'
found_cells = scipy.io.loadmat(path_to_saved_cell_ids)
found_cell_ids = found_cells['cellIDs']
tmp_cellids = ['cell%04d' % int(i) for i in found_cell_ids]
cellids = [i for i in tmp_cellids if i in matches.keys()]

# cell_info dict (.pkl) that keys everything with c_uuids, and has 'cell_idx' as a key containing the EM-TEFO alignment name:
cellinfo_fn = 'cell_info_%s.pkl' % target_uuid
ndacelldict_path = os.path.join(dictpath, cellinfo_fn)

with open(ndacelldict_path, 'rb') as f:    
    cellinfo = pkl.load(f)


neuronID_map = dict((cuuid, {'cell_name': cellinfo[cuuid]['cell_idx'], 'neuron_id': matches['cell%04d' % cellinfo[cuuid]['cell_idx']]}) for cuuid in cellinfo.keys() if 'cell%04d' % cellinfo[cuuid]['cell_idx'] in cellids)


with open(savepath, 'wb') as f:
    pkl.dump(neuronID_map, f, protocol=pkl.HIGHEST_PROTOCOL)

# dump into pkl
idkeyfn = '/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/em7_centroids/expressing_cell2neuron_key.pkl'
expressing_cell2neuron_dict = dict((c, matches[c]) for c in cellids)
with open(idkeyfn, 'wb') as fw:
    pkl.dump(expressing_cell2neuron_dict, fw, protocol=pkl.HIGHEST_PROTOCOL)

# dump in json to match neuron-locations.json:
import json
neurondict = '/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/em7_centroids/neuron-locations.json'
with open(neurondict, 'r') as f:
    data = json.load(f)

foundcells = '/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/em7_centroids/expressing_cell2neuron_key.pkl'
with open(foundcells, 'rb') as ff:
    cells = pkl.load(ff)

jneurons = [i for i in data if i['cell_id'] in cells.keys()]

jfn = '/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/em7_centroids/expressing_cell2neuron_key.json'
with open(jfn, 'w') as wf:
    json.dump(jneurons, wf)


#####################################################################################
# Combine all info from "neuron-locations.json" and "expressing cells" and transforms
# from centroid-EM stuff (i.e., "directEM7" manual alignments) into a single file:
#####################################################################################
# Combine anatomy and physio:
centroidsEM_fn = '/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/em7_centroids/centroids_EM.pkl'
with open(centroidsEM_fn, 'rb') as f:
    centroidsEM = pkl.load(f)

# RENAME THE KEYS:
for j in jneurons:
    #j['func_id'] = j.pop('cell_id')
    j['anat_id'] = j.pop('neuron_id')
    j['anat_x'] = j.pop('x')
    j['anat_y'] = j.pop('y')
    j['anat_z'] = j.pop('z')

for j in jneurons:
    curr_func_id = j['func_id']
    j['func_x'] = centroidsEM[curr_func_id]['TEFO'][0]
    j['func_y'] = centroidsEM[curr_func_id]['TEFO'][1]
    j['func_z'] = centroidsEM[curr_func_id]['TEFO'][2]

    j['em_x'] = centroidsEM[curr_func_id]['EM'][0]
    j['em_y'] = centroidsEM[curr_func_id]['EM'][1]
    j['em_z'] = centroidsEM[curr_func_id]['EM'][2]


jfn_all = '/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/em7_centroids/expressing_neurons_all_coords.json'
with open(jfn_all, 'w') as f:
    json.dump(jneurons, f)



with open(jfn_all, 'r') as f:
    JN = json.load(f)

import copy
# OH wait, now cind the ACTUAL TeFo centroids (not seeds):
import scipy.io
tefo_centroids_fn = '/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/em7_centroids/expressing_cells_name_idx.mat'
tefo_centroids = scipy.io.loadmat(tefo_centroids_fn)
cellidxs = tefo_centroids['cell_index']
cellids = tefo_centroids['cellIDs']
found_centers = tefo_centroids['found_centroids']
for cidx,cellid in enumerate(cellids):
    curr_cell_name = "cell%04d" % cellid
    for j in JN:
	print "found match: %i" % cidx
        if j['func_id']==curr_cell_name:
            j['func_x'] = found_centers[cidx][0]
            j['func_y'] = found_centers[cidx][1]
            j['func_z'] = found_centers[cidx][2]

jfn_found_centroids = '/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/em7_centroids/expressing_neurons_all_centroids.json'
with open(jfn_found_centroids, 'w') as f:
    json.dump(JN, f)

