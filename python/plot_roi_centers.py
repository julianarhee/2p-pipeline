#!/usr/bin/env python2

import os
import numpy as np
import tifffile as tf

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


source = '/nas/volume1/2photon/RESDATA/TEFO'
session = '20161219_JR030W'
run = 'retinotopyFinal'
didx = 14 #5
datastruct = 'datastruct_%03d' % didx


# Specifiy TIFF paths (if not raw):
runpath = os.path.join(source, session, run)
#tiffdir = 'DATA/int16'
#tiffpath = os.path.join(runpath, tiffdir)
tiffpath = '/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/average_volumes/avg_frames_conditions_channel01'

savepath = os.path.join(runpath, 'analysis', datastruct)

tiffs = os.listdir(tiffpath)
tiffs = [t for t in tiffs if t.endswith('.tif')]

savepath = os.path.join(tiffpath, 'DATA')
if not os.path.exists(savepath):
    os.mkdir(savepath)


#tiffidx = 3 
#stack = tf.imread(os.path.join(tiffpath, tiffs[tiffidx]))
#
#print "TIFF: %s" % tiffs[tiffidx]
#print "size: ", stack.shape
#print "dtype: ", stack.dtype
#
## Just get green channel:
#nvolumes = 340
#nslices = 22
#nframes_single_channel = nvolumes * nslices
#channel = 1
#if stack.shape[0] > nframes_single_channel:
#    print "Getting single channel: Channel %02d" % channel
#    volume = stack[::2, :, :]
#else:
#    volume = copy.deepcopy(stack)
#del stack
#
## Get averaged volume:
#if volume.shape[0] > nslices:
#    print "Averaging across time to produce average volume."
#    avg = np.empty((nslices, volume.shape[1], volume.shape[2]), dtype=volume.dtype)
#    for zslice in range(avg.shape[0]):
#        avg[zslice,:,:] = np.mean(volume[zslice::nslices,:,:], axis=0)
#
#    # Save AVG tif:
#    tf.imsave(os.path.join(savepath, 'avg.tif'), avg)
#

# Load centroids:
datastruct_fn = 'datastruct_%03d.mat' % didx
datastruct_path = os.path.join(runpath, 'analysis', datastruct, datastruct_fn)

dstruct = loadmat(datastruct_path)

centers = dstruct['maskInfo']['seeds']



