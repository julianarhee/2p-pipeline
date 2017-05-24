#!/usr/bin/env python2

import scipy.io as spio
import os
import uuid
import numpy as np
import cPickle as pkl
import h5py
import pandas as pd
import optparse
from skimage import img_as_uint

## optparse for user-input:
#source = '/nas/volume1/2photon/RESDATA/TEFO'
#session = '20161219_JR030W'
##experiment = 'gratingsFinalMask2'
## datastruct_idx = 1
#animal = 'R2B1'
#receipt_date = '2016-12-30'
#create_new = True 
#
# todo:  parse everything by session, instead of in bulk (i.e., all animals)...
# animals = ['R2B1', 'R2B2']
# receipt_dates = ['2016-12-20', '2016-12-30']

# Need better loadmat for dealing with .mat files in a non-annoying way:

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


dstructpath = '/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/retinotopyFinal/analysis/datastruct_014/datastruct_014.mat'
dstruct = loadmat(dstructpath)
meta = loadmat(dstruct['metaPath'])

tracestructnames = dstruct['traceNames3D']

tiffidx = 3

tracestruct = loadmat(os.path.join(dstruct['tracesPath'], tracestructnames[tiffidx]))
#rawtraces = tracestruct['rawTracesNMF']
#rawtraces = tracestruct['rawTraceMatDCNMF']
rawtraces = tracestruct['rawTraceMatNMF']
#rawtracestmp = tracestruct['rawTraceMatDCNMF']
#rawtraces = exposure.rescale_intensity(rawtracestmp, out_range=(0, 2**16-1))

#old_min = np.nanmin(rawtraces)
#old_max = np.nanmax(rawtraces)
#old_range = (old_max - old_min)

#new_min = 0
#new_max = 2**16
#new_range = (new_max - new_min)

old_max = rawtraces.max()
old_min = rawtraces.min()
new_max = 255.
new_min = 0.
old_range = float(old_max) - float(old_min)
new_range = 255.


maskstruct = loadmat(dstruct['maskarraymatPath'])
masks = maskstruct['maskmat']
nframes = rawtraces.shape[0]
nrois = rawtraces.shape[1]

volumesize = meta['volumeSizePixels']
szX = volumesize[0]
szY = volumesize[1]
szZ = volumesize[2]


nmfvolume = np.zeros((nframes, szZ, szY, szX), dtype='uint8')
for roi in range(nrois):
    masks[:, roi][np.isnan(masks[:, roi])] = 0. # make sure no NaNs.
    roimasktmp = np.reshape(masks[:,roi], volumesize, order='F') # Read in mask as in matlab
    roimask = np.swapaxes(roimasktmp, 0, 1) # Swap x,y to go from row,col idxs to x,y-image idxs
    roimask = np.swapaxes(roimask, 0, 2)    # Swap t to get t,y,x, order for img-space
    currtrace = np.zeros((nframes, roimask.shape[0], roimask.shape[1], roimask.shape[2]), dtype='uint8')
    
    roitrace = (((rawtraces[:,roi] - old_min) * float(new_range)) / old_range) + new_min
    #currcellname = 'cell'+'%04d' % maskstruct['maskids'][roi]
    for t in range(rawtraces[:,roi].shape[0]):
        currtrace[t, np.logical_or(currtrace[t,:,:,:], roimask)] = roitrace[t] #rawtraces[t, roi]
    nmfvolume = np.add(nmfvolume, currtrace) 
 

#nmfvolume = np.empty((szX, szY, szZ, nframes), dtype='uint16')
#for roi in range(nrois):
#    roimask = np.reshape(masks[:,roi], volumesize, order='F')
#    currtrace = np.empty((roimask.shape[0], roimask.shape[1], roimask.shape[2], nframes), dtype='uint16')
#    currtrace[np.logical_or(currtrace[:,:,:,0], roimask)] = rawtraces[:, roi]
#    nmfvolume = nmfvolume + currtrace #currtrace #rawtraces[:,roi]
#
#nmfvolume = np.swapaxes(nmfvolume, 0, 3)
#nmfvolume = np.swapaxes(nmfvolume, 1, 2)
#

outpath = '/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/retinotopyFinal/memfiles/processed_tiffs'
#outfile_fn = 'nmf_File%03d_rawtracematdcnmf.tif' % int(tiffidx+1)
outfile_fn = 'nmf_File%03d_processedNMF.tif' % int(tiffidx+1)
tf.imsave(os.path.join(outpath, outfile_fn), nmfvolume)


check_mask = False

# Check mask:
if check_mask:
    sums = np.nansum(masks, axis=0)
    print sums.shape
    print max(sums)
    print [i for i,s in enumerate(sums) if s>5]

    roimask = np.reshape(masks[:,roi], volumesize, order='F')

    roimask2 = scipy.ndimage.filters.gaussian_filter(np.reshape(masks[:,roi], volumesize, order='F'), sigma=(.5,.5,.5))

    src = mlab.pipeline.scalar_field(roimask)
    mlab.pipeline.iso_surface(src, contours=[roimask2.min()+0.1*roimask2.ptp(), ], opacity=0.3)
    mlab.pipeline.iso_surface(src, contours=[roimask2.max()-0.1*roimask2.ptp(), ],)

    mlab.show()

check_average = False

# Try averaging cycles:
currtrace = rawtraces[:, 45]
stepsize = len(currtrace)/ncycles
cycidxs = np.arange(0, len(currtrace), stepsize)

if check_average:
    currtrace = rawtraces[:, 45]
    avgtracemat = []
    for cyc in cycidxs:
	if cyc+stepsize > len(currtrace):
	    continue
	else:
	    avgtracemat.append(currtrace[cyc:cyc+stepsize])
    avgtracemat = np.array(avgtracemat)
    avgtrace = np.mean(avgtracemat, axis=0)
    plt.plot(avgtrace)



#    old_max = stack.max()
#    old_min = stack.min()
#    new_max = 255.
#    new_min = 0.
#    old_range = float(old_max) - float(old_min)
#    new_range = 255.
#    for i in range(stack.shape[2]):
#
#    #    old_max = max(stack[:,:,i].ravel())
#    #    old_min = min(stack[:,:,i].ravel())
#    #    old_range = (old_max - old_min)  
#    #    new_min = 0.
#    #    new_range = 255. #(0 - 255)  
#        stack[:,:,i] = (((stack[:,:,i] - old_min) * float(new_range)) / old_range) + new_min
# 

alltraces = []
nframes_avg = stepsize

avgvolume = np.zeros((nframes_avg, szZ, szY, szX), dtype='uint16')
for roi in range(nrois):
    masks[:, roi][np.isnan(masks[:, roi])] = 0. # make sure no NaNs.
    roimasktmp = np.reshape(masks[:,roi], volumesize, order='F') # Read in mask as in matlab
    roimask = np.swapaxes(roimasktmp, 0, 1) # Swap x,y to go from row,col idxs to x,y-image idxs
    roimask = np.swapaxes(roimask, 0, 2)    # Swap t to get t,y,x, order for img-space
    currtrace = np.zeros((nframes_avg, roimask.shape[0], roimask.shape[1], roimask.shape[2]), dtype='uint16')
    roitrace = (((rawtraces[:,roi] - old_min) * float(new_range)) / old_range) + new_min

    tmptrace = []
    for cyc in cycidxs:
    	if cyc+stepsize > len(roitrace): #len(rawtraces[:, roi]):
	    continue
        else:
	    tmptrace.append(roitrace[cyc:cyc+stepsize])
            #tmptrace.append(rawtraces[cyc:cyc+stepsize, roi])
    avgtrace = np.mean(np.array(tmptrace), axis=0)

    for t in range(len(avgtrace)):
        currtrace[t, np.logical_or(currtrace[t,:,:,:], roimask)] = avgtrace[t]
    avgvolume = np.add(avgvolume, currtrace) 

#avgvolume = np.empty((szX, szY, szZ, nframes_avg), dtype='uint16')
#for roi in range(nrois):
#    roimask = np.reshape(masks[:,roi], volumesize, order='F')
#    currtrace = np.empty((szX, szY, szZ, nframes_avg), dtype='uint16')
#    tmptrace = []
#    for cyc in cycidxs:
#    	if cyc+stepsize > len(rawtraces[:, roi]):
#	    continue
#        else:
#	    tmptrace.append(rawtraces[cyc:cyc+stepsize, roi])
#    tmptrace = np.mean(np.array(tmptrace), axis=0)
#    currtrace[np.logical_or(currtrace[:,:,:,0], roimask)] = tmptrace #rawtraces[:, roi]
#    avgvolume = avgvolume + currtrace #currtrace #rawtraces[:,roi]
#    alltraces.append(tmptrace)
#
#avgvolume = np.swapaxes(avgvolume, 0, 3)
#avgvolume = np.swapaxes(avgvolume, 1, 2)
#

outpath = '/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/retinotopyFinal/memfiles/processed_tiffs'
#outfile_fn = 'nmf_File%03d_avgcycle.tif' % int(tiffidx+1)
outfile_fn = 'nmf_File%03d_avgcycle_processedNMF.tif' % int(tiffidx+1)
tf.imsave(os.path.join(outpath, outfile_fn), avgvolume)


