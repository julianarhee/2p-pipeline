#!/usr/bin/env python2

import os
import numpy as np
import tifffile as tf
from skimage import img_as_uint

source = '/nas/volume1/2photon/RESDATA/TEFO'
session = '20161219_JR030W'
maskdir = 'em_centroids'

centroidpath = os.path.join(source, session, maskdir)

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


centroidmat_fn = 'centroids_EM.mat'
scipy.io.savemat(os.path.join(centroidpath, centroidmat_fn), mdict=emdict)







tiffpath = os.path.join(source, session, run)
tiffs = os.listdir(tiffpath)
tiffs = [t for t in tiffs if t.endswith('.tif')]

savepath = os.path.join(tiffpath, 'DATA')
if not os.path.exists(savepath):
    os.mkdir(savepath)


nflyback = 8
ndiscard = 8

nslices_full = 38

nchannels = 2                         # n channels in vol (ch1, ch2, ch1, ch2, ...)
nvolumes = 340


for tiffidx in range(len(tiffs)):
    stack = tf.imread(os.path.join(tiffpath, tiffs[tiffidx]))
 
