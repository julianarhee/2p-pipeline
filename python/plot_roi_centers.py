#!/usr/bin/env python2

import os
import numpy as np
import tifffile as tf
import scipy.io as spio
from skimage import img_as_uint, img_as_ubyte


source = '/nas/volume1/2photon/RESDATA/TEFO'
session = '20161219_JR030W'
run = 'retinotopyFinal'
didx = 14 #5
datastruct = 'datastruct_%03d' % didx


# Specifiy TIFF paths (if not raw):
runpath = os.path.join(source, session, run)

tiffpath = '/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/retinotopyFinal/memfiles/averaged/finalvolume/R2B1_tefo_avg_channel01_RGB.tif'


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


# Load averaged TIFF stack:
tiffstack = tf.imread(tiffpath)
print "Avgerage volume shape: ", tiffstack.shape
if not tiffstack.dtype=='uint8':
    tiffstack = img_as_ubyte(tiffstack)
print "Dtype: ", tiffstack.dtype


savepath = os.path.join(runpath, 'analysis', datastruct, 'figures')
if not os.path.exists(savepath):
    os.mkdir(savepath)


# TODO:  as in plot_roi_masks.py, need to fix so that can generate new colormap given 
# some set of centroids/masks at the outset.

cellmap_source = '/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/em7_centroids' 
cellmap_fn = 'cells_mapper_2_20170519_py2.pkl'
cellmap_path = os.path.join(cellmap_source, cellmap_fn) 
cellmap = pkl.load(open(cellmap_path, 'rb'))

get_colormap = [i for i in os.listdir(cellmap_source) if 'colormap' in i]
if len(get_colormap)==0:
    print "Creating colormap from source: ", cellmap_source
    print "Source file: ", cellmap_fn
    colormap_fn = 'em7_centroids_colormap.pkl'
    colormap = dict(('cell'+'%04d' % int(ckey), cellmap[ckey]['new_rgb']) for ckey in cellmap.keys())
    with open(os.path.join(cellmap_source, colormap_fn), 'wb') as f:
        pkl.dump(colormap, f, protocol=pkl.HIGHEST_PROTOCOL)
else:
    colormap_fn = get_colormap[0]
    print "Loading colormap from source: ", cellmap_source
    print "Source file: ", colormap_fn
    colormap = pkl.load(open(os.path.join(cellmap_source, colormap_fn), 'rb'))




# Load centroids:
datastruct_fn = 'datastruct_%03d.mat' % didx
datastruct_path = os.path.join(runpath, 'analysis', datastruct, datastruct_fn)

dstruct = loadmat(datastruct_path)

centers = dstruct['maskInfo']['seeds']
nrois = len(centers)

d3, d2, d1, ch = tiffstack.shape
volume = np.zeros((d3, d2, d1, 3), dtype='uint8')
for roi in range(nrois):
    coordtmp = np.array([int(c-1) for c in centers[roi]])
    #if np.any(np.mod(coord),2):
    #    coords.append = [
    coord = np.array([coordtmp[2], coordtmp[0], coordtmp[1]])
    #coord = np.swapaxes(coord, 0, 1)
    #coord = np.swapaxes(coord, 0, 2)
    currcellname = 'cell'+'%04d' % maskstruct['maskids'][roi]

    volume[coord[0], coord[1], coord[2],:] = colormap[currcellname]
    tiffstack[coord[0], coord[1], coord[2], :] = np.zeros(colormap[currcellname].shape)

outpath = '/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/retinotopyFinal/memfiles/processed_tiffs'
outfile_fn = 'nmf_File%03d_centroid_color.tif' % int(tiffidx+1)
tf.imsave(os.path.join(outpath, outfile_fn), nmfvolume)


nmfvolume = np.add(tiffstack, volume)
outfile_fn = 'nmf_File%03d_centroid_color_overlay.tif' % int(tiffidx+1)
tf.imsave(os.path.join(outpath, outfile_fn), nmfvolume)


#
#
#    masks[:, roi][np.isnan(masks[:, roi])] = 0. # make sure no NaNs.
#    roimasktmp = np.reshape(masks[:,roi], volumesize, order='F') # Read in mask as in matlab
#    #roimasktmp = np.reshape(masks[:,roi], [22, 120, 120], order='C')
#    roimask = np.swapaxes(roimasktmp, 0, 1) # Swap x,y to go from row,col idxs to x,y-image idxs
#    roimask = np.swapaxes(roimask, 0, 2)    # Swap t to get t,y,x, order for img-space
#    currtrace = np.zeros((roimask.shape[0], roimask.shape[1], roimask.shape[2], 3), dtype='uint8')
#    currcellname = 'cell'+'%04d' % maskstruct['maskids'][roi]
#    currtrace[np.logical_or(currtrace[:,:,:,0], roimask)] = colormap[currcellname]  
#    nmfvolume = np.add(nmfvolume, currtrace) 
#    if nmfvolume[12, 0, 0, 0]>0:
#        print roi, currcellname
#
##nmfvolume = np.swapaxes(nmfvolume, 0, 2)
##nmfvolume = np.swapaxes(nmfvolume, 1, 2) # swap x,y again to draw in image-coords
#
#outpath = '/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/retinotopyFinal/memfiles/processed_tiffs'
#outfile_fn = 'nmf_File%03d_centroid_color.tif' % int(tiffidx+1)
#tf.imsave(os.path.join(outpath, outfile_fn), nmfvolume)
#
#
