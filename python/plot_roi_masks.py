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
import tifffile as tf


# TODO:  fix so that all datapath naming is consistent, and don't have to hard-code.
dstructpath = '/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/retinotopyFinal/analysis/datastruct_014/datastruct_014.mat'
outpath = '/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/retinotopyFinal/memfiles/processed_tiffs'

tiffidx = 3
outfile_fn = 'nmf_File%03d_masks_color_overlay.tif' % int(tiffidx+1)

tiffpath = '/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/retinotopyFinal/memfiles/averaged/finalvolume/R2B1_tefo_avg_channel01_RGB.tif'


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


# TODO:  hard-coded for EM masks that ship w/ colors, fix to generate new maps
cellmap_source = '/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/em7_centroids' 
cellmap_path = os.path.join(cellmap_source, 'cells_mapper_2_20170519_py2.pkl')
cellmap = pkl.load(open(cellmap_path, 'rb'))

get_colormap = [i for i in os.listdir(cellmap_source) if 'colormap' in i]
if len(get_colormap)==0:
    colormap_fn = 'em7_centroids_colormap.pkl'
    colormap = dict(('cell'+'%04d' % int(ckey), cellmap[ckey]['new_rgb']) for ckey in cellmap.keys())
    with open(os.path.join(cellmap_source, colormap_fn), 'wb') as f:
        pkl.dump(colormap, f, protocol=pkl.HIGHEST_PROTOCOL)

else:
    colormap_fn = get_colormap[0]
    colormap = pkl.load(open(os.path.join(cellmap_source, colormap_fn), 'rb'))


# Load averaged TIFF stack:
tiffstack = tf.imread(tiffpath)
print "Avgerage volume shape: ", tiffstack.shape
if not tiffstack.dtype=='uint8':
    tiffstack = img_as_ubyte(tiffstack)
print "Dtype: ", tiffstack.dtype



# Load datastruct and meta info:
dstruct = loadmat(dstructpath)
meta = loadmat(dstruct['metaPath'])

maskstruct = loadmat(dstruct['maskarraymatPath'])
masks = maskstruct['maskmat']
nrois = masks.shape[1]

volumesize = meta['volumeSizePixels']
szX = volumesize[0]
szY = volumesize[1]
szZ = volumesize[2]

check_masks = False

if check_masks:
    roi = 0
    roimask = np.reshape(masks[:, roi], volumesize, order='F')
    roimask = np.swapaxes(roimask, 0, 1)

    roimask = np.reshape(masks[:, roi], [szZ, szY, szX], order='C')
    roimask = np.swapaxes(roimask, 1, 2)

    from mayavi import mlab
    src = mlab.pipeline.scalar_field(roimask)
    mlab.pipeline.iso_surface(src, contours=[roimask.min()+0.1*roimask.ptp(), ], opacity=0.3)
    mlab.pipeline.iso_surface(src, contours=[roimask.max()-0.1*roimask.ptp(), ],)

    mlab.show()



#nmfvolume = np.empty((szX, szY, szZ, 3), dtype='uint8')
nmfvolume = np.zeros((szZ, szY, szX, 3), dtype='uint8')
for roi in range(nrois):
    masks[:, roi][np.isnan(masks[:, roi])] = 0. # make sure no NaNs.
    roimasktmp = np.reshape(masks[:,roi], volumesize, order='F') # Read in mask as in matlab
    #roimasktmp = np.reshape(masks[:,roi], [22, 120, 120], order='C')
    roimask = np.swapaxes(roimasktmp, 0, 1) # Swap x,y to go from row,col idxs to x,y-image idxs
    roimask = np.swapaxes(roimask, 0, 2)    # Swap t to get t,y,x, order for img-space
    currtrace = np.zeros((roimask.shape[0], roimask.shape[1], roimask.shape[2], 3), dtype='uint8')
    currcellname = 'cell'+'%04d' % maskstruct['maskids'][roi]
    currtrace[np.logical_or(currtrace[:,:,:,0], roimask)] = colormap[currcellname]  
    nmfvolume = np.add(nmfvolume, currtrace) 
    if nmfvolume[12, 0, 0, 0]>0:
        print roi, currcellname

#nmfvolume = np.swapaxes(nmfvolume, 0, 2)
#nmfvolume = np.swapaxes(nmfvolume, 1, 2) # swap x,y again to draw in image-coords

tf.imsave(os.path.join(outpath, outfile_fn), nmfvolume)

#     volume[coord[0], coord[1], coord[2],:] = colormap[currcellname]
#     tiffstack[coord[0], coord[1], coord[2], :] = np.zeros(colormap[currcellname].shape)
# 
# outpath = '/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/retinotopyFinal/memfiles/processed_tiffs'
# outfile_fn = 'nmf_File%03d_centroid_color.tif' % int(tiffidx+1)
# tf.imsave(os.path.join(outpath, outfile_fn), nmfvolume)
# 
tiffstack[nmfvolume!=0] = 0
overlayvolume = np.add(tiffstack, nmfvolume)
outfile_fn = 'nmf_File%03d_masks_color_overlay.tif' % int(tiffidx+1)
tf.imsave(os.path.join(outpath, outfile_fn), overlayvolume)



