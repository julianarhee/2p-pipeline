#!/usr/bin/env python2

import os
import numpy as np
import tifffile as tf
from skimage import img_as_uint

source = '/nas/volume1/2photon/RESDATA/TEFO'
session = '20161219_JR030W'
maskdir = 'em7_centroids' # SAVE DIR for output mat with centroids.

centroidpath = os.path.join(source, session, maskdir)

if not os.path.exists(centroidpath):
    os.mkdir(centroidpath)


if csv:
    csvpath = os.path.join(centroidpath, [i for i in os.listdir(centroidpath) if i.endswith('.csv')][0])

    emdict = dict()
    with open(csvpath, mode='r') as infile:
	reader = csv.reader(infile)
	for ridx,row in enumerate(reader):
	    if 'Cell' in row[0]:
		continue
	    else:
		idxstring = '%04d' % int(row[0])
		emdict['cell'+idxstring] = {'idx': int(row[0]), 'EM': np.array([float(i) for i in re.findall(r"[-+]?\d*\.\d+|\d+", row[1])]), 'TEFO': np.array([float(i) for i in re.findall(r"[-+]?\d*\.\d+|\d+", row[2])])}

    print "N cells: ", len(emdict.keys())
    badparse = [i for i in emdict.keys() if len(emdict[i]['TEFO']) > 3]

    for fixkey in badparse:
	nvals = len(emdict[fixkey]['TEFO'])
	trueidx = 0
	for coord in range(0, nvals, 2):
	    baseval = emdict[fixkey]['TEFO'][coord]
	    ndigs = emdict[fixkey]['TEFO'][coord+1]
	    if trueidx==2:
		emdict[fixkey]['TEFO'][trueidx] = baseval/(10**ndigs)
	    else:
		emdict[fixkey]['TEFO'][trueidx] = baseval*(10**ndigs)
	    trueidx += 1
	emdict[fixkey]['TEFO'] = emdict[fixkey]['TEFO'][0:3]


else:
    # NEED TO BE IN env:  py3
    emdict = dict()
    sourcepath = '/nas/volume1/2photon/RESDATA/phase1_block2/alignment2/em_to_tefo/final'
    cell_fn = 'cells_mapper_2_20170519.pkl'

    import pickle as pkl
    with open(os.path.join(sourcepath, cell_fn), 'rb') as f:
	cells = pkl.load(f)

    emdict_ids = ['cell'+'%04d' % int(ckey) for ckey in cells.keys()]
    for cellid in emdict_ids:
	if 'centroid_directEM7' in cells[int(cellid[4:])].keys():
	    emdict[cellid] = dict()
	    emdict[cellid] = {'idx': int(cellid[4:]), 'EM': cells[int(cellid[4:])]['centroid_EM'], 'TEFO': cells[int(cellid[4:])]['centroid_directEM7'], 'rgb': cells[int(cellid[4:])]['new_rgb']}


centroidmat_fn = 'centroids_EM.mat'
scipy.io.savemat(os.path.join(centroidpath, centroidmat_fn), mdict=emdict)

centroidpkl_fn = 'centroids_EM.pkl'
with open(os.path.join(centroidpath, centroidpkl_fn), 'wb') as fo:
    pkl.dump(emdict, fo, 2)


def sub2ind(array_shape, rows, cols):
    ind = rows*array_shape[1] + cols
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    return ind

def ind2sub(array_shape, ind):
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    rows = (ind.astype('int') / array_shape[1])
    cols = ind % array_shape[1]
    return (rows, cols)

# GET MASKS:

tiffmask_fn = 'test_May18_direct_finalEM7_cells_to_tefo_color_single.tif'

stack = tf.imread(os.path.join(sourcepath, tiffmask_fn))
d3, d2, d1, c  = stack.shape


ids = [(cellid, [np.where(np.all(stack==emdict[cellid]['rgb'], axis=3))]) for cellid in emdict.keys()]


# Make sure colors are unique:
colors = np.array([emdict[cellid]['rgb'] for cellid in emdict.keys()])
b = np.ascontiguousarray(colors).view(np.dtype((np.void, colors.dtype.itemsize * colors.shape[1])))
_, idx = np.unique(b, return_index=True)

unique_colors = colors[idx]

if not len(emdict.keys())==len(unique_colors):
    print("Cell IDs do not have unique color assignments!\nFound %i ROIs, %i colors" % (len(emdict.keys()), len(unique_colors)))



# GET 3D MASKS:
maskdict = dict()
for cellid in emdict.keys():
    locs = np.where(np.all(stack==emdict[cellid]['rgb'], axis=3))
    zcoords = locs[0]
    ycoords = locs[2]
    xcoords = locs[1]
    npixels = len(xcoords)
    #coords_one = [(x+1, y+1, z+1) for (x,y,z) in zip(xcoords, ycoords, zcoords)] 
    coords_zero = [(x, y, z) for (x,y,z) in zip(xcoords, ycoords, zcoords)] 
    masks = np.zeros((d1,d2,d3))
    for coord in coords_zero:
        masks[coord] = 1

    maskdict[cellid] = masks

maskmat_fn = 'masks_EM.mat'
scipy.io.savemat(os.path.join(centroidpath, maskmat_fn), mdict=maskdict)


# Sanity check:
# for a given slice in MATLAB, should list all ROIs containing that slice:
[mkey for mkey in maskdict.keys() if np.any(maskdict[mkey][:,:,14])]

