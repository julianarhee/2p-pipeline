#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 11:57:24 2017

@author: julianarhee
"""
import matplotlib
matplotlib.use('Agg')
import os
import h5py
import json
import re
import datetime
import optparse
import pprint
import tifffile as tf
import pylab as pl
import numpy as np
from pipeline.python.utils import natural_keys, hash_file

pp = pprint.PrettyPrinter(indent=4)

#%%

#rootdir = '/nas/volume1/2photon/data'
#animalid = 'CE059' #'JR063'
#session = '20171009_CE059' #'20171202_JR063'
#acquisition = 'FOV1_zoom3x' #'FOV1_zoom1x_volume'
#run = 'gratings_phasemod' #'scenes'
#slurm = False
#
#trace_id = 'traces001'
#auto = False
#
#if slurm is True:
#    if 'coxfs01' not in rootdir:
#        rootdir = '/n/coxfs01/2p-data'

#%%

parser = optparse.OptionParser()

# PATH opts:
parser.add_option('-R', '--root', action='store', dest='rootdir', default='/nas/volume1/2photon/data', help='data root dir (root project dir containing all animalids) [default: /nas/volume1/2photon/data, /n/coxfs01/2pdata if --slurm]')
parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')
parser.add_option('-S', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID')
parser.add_option('-A', '--acq', action='store', dest='acquisition', default='FOV1', help="acquisition folder (ex: 'FOV1_zoom3x') [default: FOV1]")
parser.add_option('-r', '--run', action='store', dest='run', default='', help="name of run dir containing tiffs to be processed (ex: gratings_phasemod_run1)")
parser.add_option('--default', action='store_true', dest='default', default='store_false', help="Use all DEFAULT params, for params not specified by user (no interactive)")
parser.add_option('--slurm', action='store_true', dest='slurm', default=False, help="set if running as SLURM job on Odyssey")
parser.add_option('-t', '--trace-id', action='store', dest='trace_id', default='', help="Trace ID for current trace set (created with set_trace_params.py, e.g., traces001, traces020, etc.)")

(options, args) = parser.parse_args()

# Set USER INPUT options:
rootdir = options.rootdir
animalid = options.animalid
session = options.session
acquisition = options.acquisition
run = options.run

trace_id = options.trace_id
slurm = options.slurm

auto = options.default

if slurm is True:
    if 'coxfs01' not in rootdir:
        rootdir = '/n/coxfs01/2p-data'

#%%
# =============================================================================
# Load specified trace-ID parameter set:
# =============================================================================
run_dir = os.path.join(rootdir, animalid, session, acquisition, run)
trace_dir = os.path.join(run_dir, 'traces')
tmp_tid_dir = os.path.join(trace_dir, 'tmp_tids')

if not os.path.exists(trace_dir):
    os.makedirs(trace_dir)
try:
    print "Loading params for TRACE SET, id %s" % trace_id
    tracedict_path = os.path.join(trace_dir, 'traceids_%s.json' % run)
    with open(tracedict_path, 'r') as f:
        tracedict = json.load(f)
    TID = tracedict[trace_id]
    pp.pprint(TID)
except Exception as e:
    print "No TRACE SET entry exists for specified id: %s" % trace_id
    print e
    try:
        print "Checking tmp trace-id dir..."
        if auto is False:
            while True:
                tmpfns = [t for t in os.listdir(tmp_tid_dir) if t.endswith('json')]
                for tidx, tidfn in enumerate(tmpfns):
                    print tidx, tidfn
                userchoice = raw_input("Select IDX of found tmp trace-id to view: ")
                with open(os.path.join(tmp_tid_dir, tmpfns[int(userchoice)]), 'r') as f:
                    tmpTID = json.load(f)
                print "Showing tid: %s, %s" % (tmpTID['trace_id'], tmpTID['trace_hash'])
                pp.pprint(tmpTID)
                userconfirm = raw_input('Press <Y> to use this trace ID, or <q> to abort: ')
                if userconfirm == 'Y':
                    TID = tmpTID
                    break
                elif userconfirm == 'q':
                    break
    except Exception as E:
        print "---------------------------------------------------------------"
        print "No tmp trace-ids found either... ABORTING with error:"
        print e
        print "---------------------------------------------------------------"


#%%
# =============================================================================
# Get meta info for current run and source tiffs using trace-ID params:
# =============================================================================
trace_hash = TID['trace_hash']

tiff_dir = TID['SRC']
roi_name = TID['PARAMS']['roi_id']
tiff_files = sorted([t for t in os.listdir(tiff_dir) if t.endswith('tif')], key=natural_keys)
print "Found %i tiffs in dir %s.\nExtracting traces with ROI set %s." % (len(tiff_files), tiff_dir, roi_name)

# Get associated RUN info:
runmeta_path = os.path.join(run_dir, '%s.json' % run)
with open(runmeta_path, 'r') as r:
    runinfo = json.load(r)

nslices = len(runinfo['slices'])
nchannels = runinfo['nchannels']
nvolumes = runinfo['nvolumes']
ntiffs = runinfo['ntiffs']

#%%
# =============================================================================
# Create mask array and save mask images for quick reference:
# =============================================================================

traceid_dir = os.path.join(trace_dir, '%s_%s' % (TID['trace_id'], TID['trace_hash']))
trace_outdir = os.path.join(traceid_dir, 'files')
if not os.path.exists(trace_outdir):
    os.makedirs(trace_outdir)

trace_figdir = os.path.join(traceid_dir, 'figures')
if not os.path.exists(trace_figdir):
    os.makedirs(trace_figdir)

trace_fn_base = 'rawtraces_%s' % TID['trace_hash']


#%%
# =============================================================================
# Create mask array and save mask images for each slice specified in ROI set:
# =============================================================================

# Load ROI set specified in Traces param set:
roi_dir = os.path.join(rootdir, animalid, session, 'ROIs')
roidict_path = os.path.join(roi_dir, 'rids_%s.json' % session)
with open(roidict_path, 'r') as f:
    roidict = json.load(f)
RID = roidict[TID['PARAMS']['roi_id']]
rid_hash = RID['rid_hash']

# Load mask file:
mask_path = os.path.join(RID['DST'], 'masks', 'masks_%s.h5' % rid_hash)
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


#%%
# =============================================================================
# Extract ROIs for each specified slice for each file:
# =============================================================================

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


#%%
# =============================================================================
# Create time courses for ROIs:
# =============================================================================


#% Load raw traces:
try:
    tracestruct_fns = [f for f in os.listdir(trace_outdir) if f.endswith('hdf5') and 'rawtraces' in f]
    print "Found tracestructs for %i tifs in dir: %s" % (len(tracestruct_fns), trace_outdir)
    trace_source_dir = trace_outdir
except Exception as e:
    print "Unable to find extracted tracestructs from trace set: %s" % trace_id
    print "Aborting with error:"
    print e


# Get VOLUME indices to align to frame indices:
# -----------------------------------------------------------------------------
nslices = len(runinfo['slices'])
nslices_full = int(round(runinfo['frame_rate']/runinfo['volume_rate']))
print "Creating volume index list for %i total slices. %i frames were discarded for flyback." % (nslices_full, nslices_full - nslices)

vol_idxs = np.empty((nvolumes*nslices_full,))
vcounter = 0
for v in range(nvolumes):
    vol_idxs[vcounter:vcounter+nslices_full] = np.ones((nslices_full, )) * v
    vcounter += nslices_full
vol_idxs = [int(v) for v in vol_idxs]

# Create output file for parsed traces:
# -----------------------------------------------------------------------------
trace_fn = 'roi_trials_.hdf5'
trace_outfile_path = os.path.join(traceid_dir, trace_fn)

roi_outfile = h5py.File(trace_outfile_path, 'w')
roi_outfile.attrs['tiff_source'] = TID['SRC']
roi_outfile.attrs['trace_id'] = TID['trace_id']
roi_outfile.attrs['trace_hash'] = TID['trace_hash']
roi_outfile.attrs['trace_source'] = traceid_dir
roi_outfile.attrs['roiset_id'] = RID['roi_id']
roi_outfile.attrs['roiset_hash'] = RID['rid_hash']
roi_outfile.attrs['run'] = run
roi_outfile.attrs['session'] = session
roi_outfile.attrs['acquisition'] = acquisition
roi_outfile.attrs['animalid'] = animalid
roi_outfile.attrs['creation_time'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# Concatenate time courses across all files to create single time-source for RUN:
# -----------------------------------------------------------------------------
tmp_tracestruct = h5py.File(os.path.join(trace_source_dir, tracestruct_fns[0]), 'r')
total_nframes_in_run = tmp_tracestruct['Slice01']['rawtraces'].shape[0] * ntiffs
tmp_tracestruct.close(); del tmp_tracestruct

nframes_in_file = runinfo['nvolumes']
curr_frame_idx = 0
for tracefile in sorted(tracestruct_fns, key=natural_keys):
    print "Loading file:", tracefile
    tracestruct = h5py.File(os.path.join(trace_source_dir, tracefile), 'r')         # keys are SLICES (Slice01, Slice02, ...)

    roi_counter = 0
    for currslice in sorted(tracestruct.keys(), key=natural_keys):
        print "Loading slice:", currslice
        maskarray = tracestruct[currslice]['masks']
        d1, d2 = tracestruct[currslice]['rawtraces'].attrs['dims'][0:-1]
        T = tracestruct[currslice]['rawtraces'].attrs['nframes']
        nrois = maskarray.shape[1]
        masks = np.reshape(maskarray, (d1, d2, nrois), order='C')

        for roi in range(nrois):
            roi_counter += 1
            roiname = 'roi%05d' % int(roi_counter)

            if roiname not in roi_outfile.keys():
                roi_grp = roi_outfile.create_group(roiname)
                roi_grp.attrs['roi_idx'] = roi
                roi_grp.attrs['slice'] = currslice
                roi_grp.attrs['roi_img_path'] = tracestruct[currslice]['masks'].attrs['img_path']
            else:
                roi_grp = roi_outfile[roiname]

            if 'mask' not in roi_grp.keys():
                roi_mask = roi_grp.create_dataset('mask', masks[:,:,roi].shape, masks[:,:,roi].dtype)
                roi_mask[...] = masks[:,:,roi]
                roi_mask.attrs['roi_idx'] = roi
                roi_mask.attrs['slice'] = currslice

            if 'timecourse' not in roi_grp.keys():
                roi_tcourse = roi_grp.create_dataset('timecourse', (total_nframes_in_run,), tracestruct[currslice]['rawtraces'][:, roi].dtype)
            else:
                roi_tcourse = roi_outfile[roiname]['timecourse']

            roi_tcourse[curr_frame_idx:curr_frame_idx+nframes_in_file] = tracestruct[currslice]['rawtraces'][:, roi]
            roi_tcourse.attrs['trace_source'] = os.path.join(trace_source_dir, tracefile)

    curr_frame_idx += nframes_in_file


# Rename FRAME file with hash:
roi_outfile.close()

roi_tcourse_filehash = hash_file(trace_outfile_path)
new_filename = os.path.splitext(trace_outfile_path)[0] + roi_tcourse_filehash + os.path.splitext(trace_outfile_path)[1]
os.rename(trace_outfile_path, new_filename)

print "Finished extracting time course for run %s by roi." % run
print "Saved ROI TIME COURSE file to:", new_filename

#%% move tmp file:
tmp_tid_fn = 'tmp_tid_%s.json' % trace_hash
completed_tid_dir = os.path.join(tmp_tid_dir, 'completed')
if not os.path.exists(completed_tid_dir):
    os.makedirs(completed_tid_dir)
if os.path.exists(os.path.join(tmp_tid_dir, tmp_tid_fn)):
    os.rename(os.path.join(tmp_tid_dir, tmp_tid_fn), os.path.join(completed_tid_dir, tmp_tid_fn))

