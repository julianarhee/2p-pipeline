#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Extracted time courses using a set of trace params (TID) defined with set_trace_params.py

Outputs to <traceid_dir>:
    a) ./figures/SliceXX_roisXX_<rid_hash>.png 
        Masks overlaid on zprojected reference slice img with labeled rois.
        
    b) ./files/FileXXX_rawtraces_<traceid_hash>.hdf5
        Extracted raw traces using specified trace params (and specified ROI set)
        /FileXXX - group
            /SliceXX - group
                /masks - datatset
                    [d1xd2xnrois] array
                    - attrs: 
                        roi_id: RID name, 
                        rid_hash: RID hash, 
                        roi_type: roi type, 
                        nrois: num rois in set,
                        roi_idxs:  original indices of rois in set (may be different than count if src is different)
                /zproj - dataset
                    [d1xd2] zprojected image
                    - attrs:
                        img_source: file path to zprojected reference image
                /rawtraces - dataset
                    [Txnrois] array
                    - atttrs:
                        nframes:  T (num of frames in file)
                        dims: original dimensions of movie for reshaping (d1xd2)
                /frames_tsec - dataset
                    corresponding time in secs for each point in timecourse
                
    c) ./roi_timecourses.hdf5 
        Trace arrays split up by ROI.
        /roi00001 - Group
            (length-5 name of ROI in set, 1-indexed)
            - attrs:
                slice:  slice on which current roi (or com) is
                roi_img_path: source of zprojected slice image
            /mask - dataset
                [d1xd2] array
                - attrs:
                    id_in_set:  same as group name (int)
                    id_in_slice:  id in current slice (if multi-slice, will be different than group name)
                    idx_in_slice:  index in slice (if id in slice different than count of rois in slice)
                    slice:  'SliceXX'
            /timecourse = dataset
                [T,] timecourse of roi
                - attrs:
                    source_file:  path to file from which trace is extracted

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
import traceback
import tifffile as tf
import pylab as pl
import numpy as np
from pipeline.python.utils import natural_keys, hash_file
from pipeline.python.set_trace_params import post_tid_cleanup

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
#mask_path = os.path.join(RID['DST'], 'masks', 'masks_%s.h5' % rid_hash)
mask_path = os.path.join(RID['DST'], 'masks.hdf5')
maskfile = h5py.File(mask_path, "r")
is_3D = bool(maskfile.attrs['is_3D'])

filenames = maskfile.keys()

# Check if masks are split up by slices:
if type(maskfile[filenames[0]]['masks']) == h5py.Dataset:
    slice_masks = False
else:
    slice_keys = [s for s in maskfile[filenames[0]]['masks'].keys() if 'Slice' in s]
    if len(slice_keys) > 0:
        slice_masks = True
    else:
        slice_masks = False

#%
if slice_masks:
    roi_slices = sorted([str(s) for s in maskfile[filenames[0]]['masks'].keys()], key=natural_keys)
else:
    roi_slices = sorted(["Slice%02d" % int(s+1) for s in range(nslices)], key=natural_keys)

#%% For each specified SLICE in this ROI set, create 2D mask array:
print "======================================================================="

print "TID %s -- Getting mask info..." % trace_hash

MASKS = dict()
for fidx, curr_file in enumerate(filenames):
    MASKS[curr_file] = dict()
    
    for sidx, curr_slice in enumerate(roi_slices):

        MASKS[curr_file][curr_slice] = dict()
    
        # Get average image:
        if slice_masks:
            avg =np.array( maskfile[curr_file]['zproj_img'][curr_slice]).T # T is needed for MATLAB masks... (TODO: check roi_blobs)
            MASKS[curr_file][curr_slice]['zproj_img'] =  avg #maskfile[curr_file]['zproj_img'][curr_slice].attrs['source_file']
            MASKS[curr_file][curr_slice]['zproj_source'] = maskfile[curr_file]['zproj_img'][curr_slice].attrs['source_file']
            MASKS[curr_file][curr_slice]['src_roi_idxs'] = maskfile[curr_file]['masks'][curr_slice].attrs['src_roi_idxs']
        else:
            avg = maskfile[curr_file]['zproj_img']
            MASKS[curr_file][curr_slice]['zproj_img'] = avg #maskfile[curr_file].attrs['source_file']
            #roinames = maskfile[curr_file]['coords'].keys()
            MASKS[curr_file][curr_slice]['zproj_source'] = maskfile[curr_file].attrs['source_file'] #maskfile[curr_file]['coords'][roinames[0]].attrs['roi_source'] #maskfile[curr_file].attrs['source_file']
            MASKS[curr_file][curr_slice]['src_roi_idxs'] = maskfile[curr_file]['masks'].attrs['src_roi_idxs']

        d1,d2 = avg.shape
        d = d1*d2

        # Plot labeled ROIs on avg img:
        pl.figure()
        pl.imshow(avg)
        if slice_masks:
            curr_rois = ["roi%05" % int(ridx+1) for ridx,roi in enumerate(maskfile[curr_file]['masks'][curr_slice].attrs['src_roi_idxs'])]
            print curr_rois
        else:
            curr_rois = ["roi%05d" % int(ridx+1) for ridx,roi in enumerate(maskfile[curr_file]['masks'].attrs['src_roi_idxs'])]
            print curr_rois
            
        # Create maskarray:
        print "Plotting ROIs:", curr_rois
        nrois = len(curr_rois)
        maskarray = np.empty((d, nrois))
        maskarray_roi_ids = np.empty((nrois,)) 
        for ridx, roi in enumerate(curr_rois):
            roi_id = int(roi[3:])-1
            if slice_masks: 
                masktmp = np.array(maskfile[curr_file]['masks'][curr_slice]).T[:,:,ridx] # T is needed for MATLAB masks... (TODO: check roi_blobs)
            else:
                if len(roi_slices) > 1:
                    masktmp = maskfile[curr_file]['masks'][:,:,sidx,ridx]
                else:
                    masktmp = maskfile[curr_file]['masks'][:,:,ridx]
 
            msk = masktmp.copy()
            msk[msk==0] = np.nan
            pl.imshow(msk, interpolation='None', alpha=0.3, cmap=pl.cm.hot)
            [ys, xs] = np.where(masktmp>0)
            pl.text(xs[int(round(len(xs)/4))], ys[int(round(len(ys)/4))], str(ridx+1), weight='bold')
            pl.axis('off')
            
            # Normalize by size:
            masktmp = np.reshape(masktmp, (d,), order='C')
            npixels = len(np.nonzero(masktmp)[0])
            maskarray[:,ridx] = masktmp/npixels
            maskarray_roi_ids[ridx] = roi_id
        
        MASKS[curr_file][curr_slice]['mask_array'] = maskarray
            
        pl.savefig(os.path.join(trace_figdir, '%s_%s_%s.png' % (curr_slice, RID['roi_id'], RID['rid_hash'])))
        pl.close()


#%%
# ================================================================================
# Load reference info:
# ================================================================================
run_dir = os.path.join(rootdir, animalid, session, acquisition, run)
runinfo_path = os.path.join(run_dir, '%s.json' % run)

with open(runinfo_path, 'r') as fr:
    runinfo = json.load(fr)
nfiles = runinfo['ntiffs']
file_names = sorted(['File%03d' % int(f+1) for f in range(nfiles)], key=natural_keys)
frames_tsec = runinfo['frame_tstamps_sec']

#%%
# =============================================================================
# Extract ROIs for each specified slice for each file:
# =============================================================================

print "TID %s -- Applying masks to traces..." % trace_hash

signal_channel_idx = int(TID['PARAMS']['signal_channel']) - 1 # 0-indexing into tiffs

file_hashdict = dict()
for tfn in tiff_files:

    curr_file = str(re.search(r"File\d{3}", tfn).group())
    print "Extracting traces: %s" % curr_file

    try:
        # Create outfile:
        trace_fn = '%s_%s.hdf5' % (curr_file, trace_fn_base)
        trace_outfile_path = os.path.join(trace_outdir, trace_fn)

        # Load input tiff file:
        print "Reading tiff..."
        currtiff_path = os.path.join(tiff_dir, tfn)
        tiff = tf.imread(currtiff_path)
        T, d1, d2 = tiff.shape
        d = d1*d2
        tiffR = np.reshape(tiff, (T, d), order='C')
    
        # First get signal channel only:
        tiffR = tiffR[signal_channel_idx::nchannels,:]

        # Apply masks to each slice:
        file_grp = h5py.File(trace_outfile_path, 'w')
        file_grp.attrs['source_file'] = currtiff_path
        file_grp.attrs['file_id'] = curr_file
        file_grp.attrs['dims'] = (d1, d2, nslices, T/nslices)
        file_grp.attrs['mask_sourcefile'] = mask_path

        # Get curr file's masks, or use reference:
        if curr_file in MASKS.keys():
            single_ref = False
            mask_key = curr_file
        else:
            single_ref = True
            mask_key = MASKS.keys()[0]

        for sl in range(len(roi_slices)):
    
            curr_slice = 'Slice%02d' % int(roi_slices[sl][5:])
            print curr_slice
            maskarray = MASKS[mask_key][roi_slices[sl]]['mask_array']

            # Get frame tstamps:
            curr_tstamps = np.array(frames_tsec[sl::nslices])
            
            # Get current frames:
            tiffslice = tiffR[sl::nslices, :]
            tracemat = tiffslice.dot(maskarray)
            dims = (d1, d2, T/nvolumes)
    
            if curr_slice not in file_grp.keys():
                slice_grp = file_grp.create_group(curr_slice)
            else:
                slice_grp = file_grp[curr_slice]
    
            # Save masks:
            if 'masks' not in slice_grp.keys():
                mset = slice_grp.create_dataset('masks', maskarray.shape, maskarray.dtype)
            mset[...] = maskarray
            mset.attrs['roi_id'] = str(RID['roi_id'])
            mset.attrs['rid_hash'] = str(RID['rid_hash'])
            mset.attrs['roi_type'] = str(RID['roi_type'])
            mset.attrs['nrois'] = maskarray.shape[1]
            mset.attrs['src_roi_idxs'] = MASKS[mask_key][curr_slice]['src_roi_idxs']
                        
            # Save zproj img:
            zproj = MASKS[mask_key][curr_slice]['zproj_img']
            if 'zproj' not in slice_grp.keys():
                zset = slice_grp.create_dataset('zproj', zproj.shape, zproj.dtype)
            zset[...] = zproj
            zset.attrs['img_source'] = MASKS[mask_key][curr_slice]['zproj_source']

            # Save fluor trace:
            if 'rawtraces' not in slice_grp.keys():
                tset = slice_grp.create_dataset('rawtraces', tracemat.shape, tracemat.dtype)
            tset[...] = tracemat
            tset.attrs['nframes'] = tracemat.shape[0]
            tset.attrs['dims'] = dims
            
            # Save tstamps:
            if 'frames_tsec' not in slice_grp.keys():
                fset = slice_grp.create_dataset('frames_tsec', curr_tstamps.shape, curr_tstamps.dtype)
            fset[...] = curr_tstamps
    
        # Create hash of current raw tracemat:
        rawfile_hash = hash_file(trace_outfile_path)
        file_hashdict[os.path.splitext(trace_fn)[0]] = rawfile_hash

    except Exception as e:
        print "--- Error extracting traces from file %s ---" % curr_file
        traceback.print_exc()
        print "---------------------------------------------------------------"
    
    finally:
        file_grp.close()
    
with open(os.path.join(trace_outdir, 'fileinfo_%s.json' % TID['trace_hash']), 'w') as f:
    json.dump(file_hashdict, f, indent=4, sort_keys=True)


#%%
# =============================================================================
# Create time courses for ROIs:
# =============================================================================

print "TID %s -- sorting traces by ROI..." % trace_hash

#% Load raw traces:
try:
    tracestruct_fns = [f for f in os.listdir(trace_outdir) if f.endswith('hdf5') and 'rawtraces' in f]
    print "Found tracestructs for %i tifs in dir: %s" % (len(tracestruct_fns), trace_outdir)
    trace_source_dir = trace_outdir
except Exception as e:
    print "Unable to find extracted tracestructs from trace set: %s" % trace_id
    print "Aborting with error:"
    traceback.print_exc()

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
trace_fn = 'roi_timecourses.hdf5'
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
try:
    for trace_fn in sorted(tracestruct_fns, key=natural_keys):
        try:
            print "Loading file:", trace_fn
            tracefile = h5py.File(os.path.join(trace_source_dir, trace_fn), 'r')         # keys are SLICES (Slice01, Slice02, ...)
        
            roi_counter = 0
            for currslice in sorted(tracefile.keys(), key=natural_keys):
                print "Loading slice:", currslice
                maskarray = tracefile[currslice]['masks']
                d1, d2 = tracefile[currslice]['rawtraces'].attrs['dims'][0:-1]
                T = tracefile[currslice]['rawtraces'].attrs['nframes']
                src_roi_idxs = tracefile[currslice]['masks'].attrs['src_roi_idxs']
                nrois = maskarray.shape[1]
                masks = np.reshape(maskarray, (d1, d2, nrois), order='C')
        
                for ridx, roi in enumerate(src_roi_idxs):
                    roi_counter += 1
                    roiname = 'roi%05d' % int(roi_counter)
        
                    if roiname not in roi_outfile.keys():
                        roi_grp = roi_outfile.create_group(roiname)
                        roi_grp.attrs['slice'] = currslice
                        roi_grp.attrs['roi_img_path'] = tracefile[currslice]['zproj'].attrs['img_source']
                    else:
                        roi_grp = roi_outfile[roiname]
        
                    if 'mask' not in roi_grp.keys():
                        roi_mask = roi_grp.create_dataset('mask', masks[:,:,ridx].shape, masks[:,:,ridx].dtype)
                        roi_mask[...] = masks[:,:,ridx]
                        roi_grp.attrs['id_in_set'] = roi_counter #roi
                        roi_grp.attrs['id_in_src'] = roi #ridx
                        roi_grp.attrs['idx_in_slice'] = ridx
                        roi_mask.attrs['slice'] = currslice
        
                    if 'timecourse' not in roi_grp.keys():
                        roi_tcourse = roi_grp.create_dataset('timecourse', (total_nframes_in_run,), tracefile[currslice]['rawtraces'][:, ridx].dtype)
                    else:
                        roi_tcourse = roi_outfile[roiname]['timecourse']
        
                    roi_tcourse[curr_frame_idx:curr_frame_idx+nframes_in_file] = tracefile[currslice]['rawtraces'][:, ridx]
                    roi_tcourse.attrs['source_file'] = os.path.join(trace_source_dir, trace_fn)
        
            curr_frame_idx += nframes_in_file
        except Exception as e:
            print "--- ERROR splitting tracefile for ROIs. --- %s" % tracefile
            traceback.print_exc()
        finally:
            tracefile.close()
except Exception as e:
    print "--- ERROR processing tracefile %s ---" % tracefile
    traceback.print_exc()
finally:
    roi_outfile.close()
    
## Rename FRAME file with hash:
#roi_outfile.close()

#roi_tcourse_filehash = hash_file(trace_outfile_path)
#new_filename = "%s_%s.%s" % (os.path.splitext(trace_outfile_path)[0], roi_tcourse_filehash, os.path.splitext(trace_outfile_path)[1])
#os.rename(trace_outfile_path, new_filename)

print "======================================================================="
print "TID %s -- Finished extracting time course for run %s by roi." % (trace_hash, run)
print "Saved ROI TIME COURSE file to:", trace_outfile_path

#%% move tmp file:
tmp_tid_fn = 'tmp_tid_%s.json' % trace_hash
completed_tid_dir = os.path.join(tmp_tid_dir, 'completed')
if not os.path.exists(completed_tid_dir):
    os.makedirs(completed_tid_dir)
if os.path.exists(os.path.join(tmp_tid_dir, tmp_tid_fn)):
    os.rename(os.path.join(tmp_tid_dir, tmp_tid_fn), os.path.join(completed_tid_dir, tmp_tid_fn))

print "Cleaned up tmp tid files."
print "======================================================================="
