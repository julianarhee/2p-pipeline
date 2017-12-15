#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 11:57:24 2017

@author: julianarhee
"""

import os
import h5py
import json
import re
import hashlib
import tifffile as tf
import pylab as pl
import numpy as np
import pprint
pp = pprint.PrettyPrinter(indent=4)



from pipeline.python.utils import natural_keys


def hash_file(fpath, hashtype='sha1'):

    BLOCKSIZE = 65536
    if hashtype=='md5':
        hasher = hashlib.md5()
    else:
        hasher = hashlib.sha1()

    with open(fpath, 'rb') as afile:
        buf = afile.read(BLOCKSIZE)
        while len(buf) > 0:
            hasher.update(buf)
            buf = afile.read(BLOCKSIZE)

    #print(hasher.hexdigest())

    return hasher.hexdigest()[0:6]


#%%
rootdir = '/nas/volume1/2photon/data'
animalid = 'JR063'
session = '20171202_JR063'

session_dir = os.path.join(rootdir, animalid, session)
roi_dir = os.path.join(session_dir, 'ROIs')

roi_name = 'rois002'

tiff_dir = ''

auto = False

#%%
with open(os.path.join(roi_dir, 'rids_%s.json' % session), 'r') as f:
    roidict = json.load(f)
RID = roidict[roi_name]

# Get TIFF source:
if tiff_dir is None or len(tiff_dir) == 0:
    tiff_dir = RID['SRC']
tiff_files = sorted([t for t in os.listdir(tiff_dir) if t.endswith('tif')], key=natural_keys)
print "Found %i tiffs in dir %s.\nExtracting traces with ROI set %s." % (len(tiff_files), tiff_dir, roi_name)

# Get associated RUN info:
pathparts = tiff_dir.split(session_dir)[1].split('/')
acquisition = pathparts[1]
run = pathparts[2]
runmeta_path = os.path.join(session_dir, acquisition, run, '%s.json' % run)
with open(runmeta_path, 'r') as r:
    runinfo = json.load(r)

nslices = len(runinfo['slices'])
nchannels = runinfo['nchannels']
nvolumes = runinfo['nvolumes']

#%%
trace_dir = os.path.join(session_dir, 'Traces')

# Create HASH from tiff source and ROI params:
TID = dict()
TID['tiff_source'] = tiff_dir
TID['roi_id'] = RID['roi_id']
TID['rid_hash'] = RID['rid_hash']
TID['roi_type'] = RID['roi_type']
trace_hash = hashlib.sha1(json.dumps(TID, sort_keys=True)).hexdigest()[0:6]
TID['trace_hash'] = trace_hash

tracedict_path = os.path.join(trace_dir, 'tid_%s.json' % session)
if os.path.exists(tracedict_path):
    with open(tracedict_path, 'r') as tr:
        tracedict = json.load(tr)
else:
    tracedict = dict()

# Check that this is a unique trace set:
match_tids = [t for t in tracedict.keys() if tracedict[t]['trace_hash'] == TID['trace_hash']]
new_trace_set = True

if auto is False:
    # Allow user-interactive mode to reload existing TID:
    if len(match_tids) > 0:
        while True:
            print "Found matching trace-set config:"
            for mix, mid in enumerate(match_tids):
		print mix, mid
                pp.pprint(tracedict[mid])
            uchoice = raw_input('Press IDX of trace set to load, or press <N> to create new trace set: ')
            if uchoice == 'N':
                new_trace_set = True
            else:
                confirm_uchoice = raw_input('Load existing trace set %s? Press <Y> to confirm.' % tracedict[match_tids[int(uchoice)]])
                if confirm_uchoice == 'Y':
                    new_trace_set = False
                    TID = tracedict[match_tids[int(uchoice)]]
                    break
    else:
        new_trace_set = True

if new_trace_set is True:
    trace_id = 'traces%03d' % int(len(tracedict.keys()) + 1)
    TID['trace_id'] = trace_id


tracedict['trace_id'] = TID
with open(tracedict_path, 'w') as tw:
    json.dump(tracedict, tw, indent=4, sort_keys=True)

trace_basedir = os.path.join(trace_dir, '%s_%s' % (TID['trace_id'], TID['trace_hash']))
trace_outdir = os.path.join(trace_basedir, 'extracted')
if not os.path.exists(trace_outdir):
    os.makedirs(trace_outdir)

trace_figdir = os.path.join(trace_basedir, 'figures')
if not os.path.exists(trace_figdir):
    os.makedirs(trace_figdir)

trace_fn_base = 'rawtraces_%s' % TID['trace_hash']


#%%

# Load mask file:
mask_path = os.path.join(RID['DST'], 'masks', 'masks.h5')
f = h5py.File(mask_path, "r")

roi_slices = sorted([str(s) for s in f['/masks'].keys()], key=natural_keys)

# For each specified SLICE in this ROI set, extract traces:
MASKS = dict()
for sidx in range(len(roi_slices)):

    curr_slice = roi_slices[sidx]
    MASKS[curr_slice] = dict()

    # Get average image:
    avg_source = os.path.split(f['/masks/%s' % curr_slice].attrs['source'])[0]
    avg_slice_fn = os.path.split(f['/masks/%s' % curr_slice].attrs['source'])[1]
    file_name = re.search(r"File\d{3}", avg_slice_fn).group()
    channel_name = re.search(r"Channel\d{2}", avg_slice_fn).group()
    sigchannel_idx = int(channel_name[7:]) - 1
    avg_slice_path = os.path.join(avg_source, channel_name, file_name, avg_slice_fn)
    avg = tf.imread(avg_slice_path)
    MASKS[curr_slice]['img_path'] = avg_slice_path

    d1,d2 = avg.shape
    d = d1*d2

    # Plot labeled ROIs on avg img:
    pl.figure()
    pl.imshow(avg)
    curr_rois = [str(r) for r in f['/masks/%s' % curr_slice].keys()]
    for roi in curr_rois:
        masktmp = f['/masks/%s/%s' % (curr_slice, roi)].value.T
        msk = masktmp.copy()
        msk[msk==0] = np.nan
        pl.imshow(msk, interpolation='None', alpha=0.3, cmap=pl.cm.hot)
        [ys, xs] = np.where(masktmp>0)
        pl.text(xs[int(round(len(xs)/4))], ys[int(round(len(ys)/4))], roi, weight='bold')
        pl.axis('off')

    pl.savefig(os.path.join(trace_figdir, '%s_%s_%s' % (curr_slice, RID['roi_id'], RID['rid_hash'])))
    pl.close()

    # Create maskarray:
    nrois = len(curr_rois)
    maskarray = np.empty((d, nrois))
    for roi in curr_rois:
        ridx = int(roi[3:])-1
        masktmp = f['/masks/%s/%s' % (curr_slice, roi)].value.T
        masktmp = np.reshape(masktmp, (d,), order='C')
        npixels = len(np.nonzero(masktmp)[0])
        maskarray[:,ridx] = masktmp/npixels

    MASKS[curr_slice]['maskarray'] = maskarray


# Extract ROIs for each specified slice for each file:
file_hashdict = dict()
for tfn in tiff_files:

    curr_file = str(re.search(r"File\d{3}", tfn).group())
    print "Extracting traces: %s" % curr_file

    # Create outfile:
    trace_fn = '%s_%s.hdf5' % (curr_file, trace_fn_base)
    trace_outfile_path = os.path.join(trace_outdir, trace_fn)

    # Load input tiff file:
    currtiff_path = os.path.join(tiff_dir, tfn)

    print "Reading tiff..."
    tiff = tf.imread(currtiff_path)
    T, d1, d2 = tiff.shape
    d = d1*d2

    tiffR = np.reshape(tiff, (T, d), order='C')

    # First get signal channel only:
    tiffR = tiffR[sigchannel_idx::nchannels,:]

    # Apply masks to each slice:
    file_grp = h5py.File(trace_outfile_path, 'w')
    file_grp.attrs['source_file'] = currtiff_path
    file_grp.attrs['file_id'] = curr_file
    file_grp.attrs['dims'] = (d1, d2, nslices, T/nslices)

    mask_slice_names = sorted([str(k) for k in MASKS.keys()], key=natural_keys)
    for sl in range(len(mask_slice_names)):

        curr_slice = 'Slice%02d' % int(mask_slice_names[sl][5:])
        print curr_slice
        maskdict = MASKS[mask_slice_names[sl]]


        tiffslice = tiffR[sl::nslices, :]
        tracemat = tiffslice.dot(maskdict['maskarray'])
        dims = (d1, d2, T/nvolumes)

        if curr_slice not in file_grp.keys():
            slice_grp = file_grp.create_group(curr_slice)

        if 'masks' not in slice_grp.keys():
            mset = slice_grp.create_dataset('masks', maskdict['maskarray'].shape, maskdict['maskarray'].dtype)
        mset.attrs['roi_id'] = str(RID['roi_id'])
        mset.attrs['rid_hash'] = str(RID['rid_hash'])
        mset.attrs['roi_type'] = str(RID['roi_type'])
        mset.attrs['nrois'] = maskdict['maskarray'].shape[1]
        mset.attrs['img_path'] = maskdict['img_path']

        mset[...] = maskdict['maskarray']

        if 'rawtraces' not in slice_grp.keys():
            tset = slice_grp.create_dataset('rawtraces', tracemat.shape, dtype=tracemat.dtype)
        tset[...] = tracemat
        tset.attrs['nframes'] = tracemat.shape[0]
        tset.attrs['dims'] = dims

    #trace = [np.mean(tiff[t,currmask>0]) for t in range(tiff.shape[0])]

    # Create hash of current raw tracemat:
    rawfile_hash = hash_file(trace_outfile_path)
    file_hashdict[os.path.splitext(trace_fn)[0]] = rawfile_hash


with open(os.path.join(trace_outdir, 'fileinfo_%s.json' % TID['trace_hash']), 'w') as f:
    json.dump(file_hashdict, f, indent=4, sort_keys=True)







