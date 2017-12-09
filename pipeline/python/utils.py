#!/usr/bin/env python2 
import os
import numpy as np
import tifffile as tf
import json
import re
import shutil

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def jsonify_array(curropts):  
    jsontypes = (list, tuple, str, int, float)
    for pkey in curropts.keys():
        if isinstance(curropts[pkey], dict):
            for subkey in curropts[pkey].keys():
                if curropts[pkey][subkey] is not None and not isinstance(curropts[pkey][subkey], jsontypes) and len(curropts[pkey][subkey].shape) > 1:
                    curropts[pkey][subkey] = curropts[pkey][subkey].tolist()
    return curropts

def write_dict_to_json(pydict, writepath):
    jstring = json.dumps(pydict, indent=4, allow_nan=True, sort_keys=True)
    f = open(writepath, 'w')
    print >> f, jstring
    f.close()
                
def interleave_tiffs(source_dir, write_dir, runinfo_path):
    '''
    source_dir (str) : path to folder containing tiffs to interleave
    runinfo_path (str) : path to .json contaning run meta info
    write_dir (str) : path to save interleaved tiffs to
    '''
    with open(runinfo_path, 'r') as f:
        runinfo = json.load(f)
    nfiles = runinfo['ntiffs']
    nchannels = runinfo['nchannels']
    nslices = len(runinfo['slices'])
    nvolumes = runinfo['nvolumes']
    ntotalframes = nslices * nvolumes * nchannels
    basename = runinfo['base_filename']

    if not os.path.exists(write_dir):
        os.makedirs(write_dir)
    print "Writing INTERLEAVED tiffs to:", write_dir

    tiffs = sorted([t for t in os.listdir(source_dir) if t.endswith('tif')], key=natural_keys)
    for fidx in range(nfiles):
        print "Interleaving file %i of %i." % (int(fidx+1), nfiles)
        curr_file = 'File%03d' % int(fidx+1)
        interleaved_fn = "{basename}_{currfile}.tif".format(basename=basename, currfile=curr_file)
        print "New tiff name:", interleaved_fn
        curr_file_fns = [t for t in tiffs if curr_file in t]
        sample = tf.imread(os.path.join(source_dir, curr_file_fns[0])) 
        print "Found %i tiffs for current file." % len(curr_file_fns)
        stack = np.empty((ntotalframes, sample.shape[1], sample.shape[2]), dtype=sample.dtype)
        for fn in curr_file_fns:
            curr_tiff = tf.imread(os.path.join(source_dir, fn))
            sl_idx = int(fn.split('Slice')[1][0:2]) - 1
            ch_idx = int(fn.split('Channel')[1][0:2]) - 1
            slice_indices = np.arange((sl_idx*nchannels)+ch_idx, ntotalframes)
            idxs = slice_indices[::(nslices*nchannels)]
            stack[idxs,:,:] = curr_tiff

        tf.imsave(os.path.join(write_dir, interleaved_fn), stack)

def deinterleave_tiffs(source_dir, write_dir, runinfo_path):
    '''
    source_dir (str) : path to folder containing interleaved tiffs 
    write_dir (str): path to save deinterleaved tiffs to (sorted by Channel, File)
    runinfo_path (str) : path to .json containing meta info about run
    '''
    with open(runinfo_path, 'r') as f:
        runinfo = json.load(f)
    nfiles = runinfo['ntiffs']
    nchannels = runinfo['nchannels']
    nslices = len(runinfo['slices'])
    nvolumes = runinfo['nvolumes']
    ntotalframes = nslices * nvolumes * nchannels
    basename = runinfo['base_filename']

    if not os.path.exists(write_dir):
        os.makedirs(write_dir)

    print "Writing DEINTERLEAVED tiffs to:", write_dir

    tiffs = sorted([t for t in os.listdir(source_dir) if t.endswith('tif')], key=natural_keys)
    good_to_go = True
    if not len(tiffs) == nfiles:
        print "Mismatch in num tiffs. Expected %i files, found %i tiffs in dir:\n%s" % (ntiffs, len(tiffs), source_dir)
        good_to_go = False
    if good_to_go: 
        # Load in each TIFF and deinterleave:
        for fidx,filename in enumerate(sorted(tiffs, key=natural_keys)):
            print "Deinterleaving File %i of %i [%s]" % (int(fidx+1), nfiles, filename)
            stack = tf.imread(os.path.join(source_dir, filename))
            print "Size:", stack.shape
            curr_file = "File%03d" % int(fidx+1)
            for ch_idx in range(nchannels):
                curr_channel = "Channel%02d" % int(ch_idx+1)
                for sl_idx in range(nslices):
                    curr_slice = "Slice%02d" % int(sl_idx+1)
                    frame_idx = ch_idx + sl_idx*nchannels
                    slice_indices = np.arange(frame_idx, ntotalframes, (nslices*nchannels))
                    print "nslices:", len(slice_indices)
                    curr_slice_fn = "{basename}_{currslice}_{currchannel}_{currfile}.tif".format(basename=basename, currslice=curr_slice, currchannel=curr_channel, currfile=curr_file)
                    tf.imsave(os.path.join(write_dir, curr_slice_fn), stack[slice_indices, :, :])


def sort_deinterleaved_tiffs(source_dir, runinfo_path):
    '''
    source_dir (str) : path to folder containing deinterleaved tiffs 
    runinfo_path (str) : path to .json containing meta info about run
    '''
    with open(runinfo_path, 'r') as f:
        runinfo = json.load(f)
    nfiles = runinfo['ntiffs']
    nchannels = runinfo['nchannels']
    nslices = len(runinfo['slices'])
    nvolumes = runinfo['nvolumes']
    ntotalframes = nslices * nvolumes * nchannels
    basename = runinfo['base_filename']

    channel_names = ['Channel%02d' % int(ci + 1) for ci in range(nchannels)]
    file_names = ['File%03d' % int(fi + 1) for fi in range(nfiles)]
    print "Expected channels:", channel_names
    print "Expected file:", file_names

    all_tiffs = sorted([t for t in os.listdir(source_dir) if t.endswith('tif')], key=natural_keys)
    expected_ntiffs = nfiles * nchannels * nslices
    good_to_go = True
    if not len(all_tiffs) == expected_ntiffs:
        print "**WARNING*********************"
        print "Mismatch in tiffs found (%i) and expected n tiffs (%i)." % (len(all_tiffs), expected_ntiffs)
        good_to_go = False
    else:
        print "Found %i TIFFs in source:", source_dir
        print "Expected n tiffs:", expected_ntiffs

    if good_to_go is True:
        for channel_name in channel_names:
            print "Sorting %s" % channel_name
            tiffs_by_channel = [t for t in all_tiffs if channel_name in t]
            channel_dir = os.path.join(source_dir, channel_name)
            if not os.path.exists(channel_dir):
                os.makedirs(channel_dir)
            for ch_tiff in tiffs_by_channel:
                shutil.move(os.path.join(source_dir, ch_tiff), os.path.join(channel_dir, ch_tiff))
            for file_name in file_names:
                print "Sorting %s" % file_name
                tiffs_by_file = [t for t in tiffs_by_channel if file_name in t]
                print "Curr file tiffs:", tiffs_by_file
                file_dir = os.path.join(channel_dir, file_name)
                print "File dir:", file_dir
                if not os.path.exists(file_dir):
                    os.makedirs(file_dir)
                for fi_tiff in tiffs_by_file:
                    shutil.move(os.path.join(channel_dir, fi_tiff), os.path.join(file_dir, fi_tiff))
    print "Done organizing tiffs."

