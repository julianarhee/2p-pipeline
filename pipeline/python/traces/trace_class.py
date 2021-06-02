#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 12:51:58 2018

@author: juliana
"""

import optparse
import os
import glob
import json
import h5py

import numpy as np
import tifffile as tf
import pylab as pl
import cPickle as pkl


from caiman.utils.visualization import get_contours
from pipeline.python.utils import natural_keys, label_figure
from pipeline.python.rois import utils as rutil # import get_roi_contours, uint16_to_RGB, plot_roi_contours


options = ['-D', '/n/coxfs01/2p-data','-i', 'CE077', '-S', '20180521', '-A', 'FOV2_zoom1x',
           '-R', 'gratings_run1', '-t', 'traces001']

def extract_options(options):

    parser = optparse.OptionParser()

    parser.add_option('-D', '--root', action='store', dest='rootdir',
                          default='/nas/volume1/2photon/data',
                          help='data root dir (dir w/ all animalids) [default: /nas/volume1/2photon/data, /n/coxfs01/2pdata if --slurm]')
    parser.add_option('--slurm', action='store_true', dest='slurm', default=False, help="set if running as SLURM job on Odyssey")

    parser.add_option('-i', '--animalid', action='store', dest='animalid',
                          default='', help='Animal ID')

    # Set specific session/run for current animal:
    parser.add_option('-S', '--session', action='store', dest='session',
                          default='', help='session dir (format: YYYMMDD_ANIMALID')
    parser.add_option('-A', '--acq', action='store', dest='acquisition',
                          default='FOV1', help="acquisition folder (ex: 'FOV1_zoom3x') [default: FOV1]")
    parser.add_option('-R', '--run', dest='run', default='', action='store', help="run name")
    parser.add_option('-t', '--traceid', dest='traceid', default=None, action='store', help="traceid (e.g., traces001)")
    
    (options, args) = parser.parse_args(options)
    if options.slurm:
        options.rootdir = '/n/coxfs01/2p-data'
    
    return options

#%%
    
class struct():
    pass 


class TraceSet():
    def __init__(self, animalid, session, acquisition, run, traceid, 
                 processid='processed001', rootdir='/n/coxfs01/2p-data'):
        self.source =  struct()
        self.source.rootdir = rootdir
        self.source.animalid = animalid
        self.source.session = session
        self.source.acquisition = acquisition
        self.source.run = run
        self.source.processid = processid
        self.source.traceid = traceid
        
        self.TID = self.get_traceid_info()
        
        
    def get_traceid_info(self):
        run_dir = os.path.join(self.source.rootdir, self.source.animalid, self.source.session, self.source.acquisition, self.source.run)
        print run_dir
        
        traceids_fpath = glob.glob(os.path.join(run_dir, 'traces', 'traceids_*.json'))[0]
        with open(traceids_fpath, 'r') as f: tids = json.load(f)
        TID = tids[self.source.traceid]
        
        return TID
    
            
    def get_fov_image(self, process_type='mcorrected', signal_channel=1):
    
        # Get Mean image path to visualize:
        # -----------------------------------------------------------------------------
        mean_img_paths = glob.glob(os.path.join(self.source.rootdir, self.source.animalid,
                                                self.source.session, self.source.acquisition, 
                                                self.source.run, 'processed', 
                                                '%s*' % self.source.processid, 
                                                '%s*' % process_type, 
                                                'Channel%02d' % signal_channel, 
                                                'File*', '*.tif')) 
        fovs = []
        for fpath in mean_img_paths:
            img = tf.imread(fpath)
            fovs.append(img)
        fov_image = np.mean(np.dstack(fovs), axis=-1)
    
        return fov_image

    
    def get_masks(self):
                       
        run_dir = os.path.join(self.source.rootdir, self.source.animalid, self.source.session, self.source.acquisition, self.source.run)
            
        transformed_mask_fpath = glob.glob(os.path.join(run_dir, 'traces', '%s*' % self.source.traceid, 'MASKS.hdf5'))[0]
        
        # Get MC reference file:
        processids_fpath = glob.glob(os.path.join(run_dir, 'processed', 'pids_*.json'))[0]
        with open(processids_fpath, 'r') as f: pids = json.load(f)
        PID = pids[self.source.processid]
        reference_file = 'File%03d' % PID['PARAMS']['motion']['ref_file']
        
    
        # Get masks from reference file:
        # -----------------------------------------------------------------------------
        mfile = h5py.File(transformed_mask_fpath, 'r')
        masks = mfile[reference_file]['Slice01']['maskarray'][:]
        
        # If manual2D_circle, convert to 1s:
        if self.TID['PARAMS']['roi_type'] == 'manual2D_circle':
            masks[masks > 0] = 1
            
        dims = mfile[reference_file]['Slice01']['zproj'].shape
        
        return masks, dims
    
    def get_coordinates(self, masks, dims):
        
        coords = get_contours(masks, dims, thr=0.8)

        return coords

#%%

import matplotlib as mpl

optsE = extract_options(options)    

T = TraceSet(optsE.animalid, optsE.session, optsE.acquisition, optsE.run, optsE.traceid)


# Set output dirs:
run_dir = os.path.join(T.source.rootdir, T.source.animalid, T.source.session,
                       T.source.acquisition, T.source.run)
output_dir = glob.glob(os.path.join(run_dir, 'traces', '%s*' % T.source.traceid, 'figures'))[0]
if not os.path.exists(output_dir): os.makedirs(output_dir)

data_identifier = '_'.join([T.source.animalid, T.source.session, T.source.acquisition, T.source.run, T.source.traceid])

fov_image = T.get_fov_image()
masks, dims = T.get_masks()
mask_r = np.reshape(masks, (dims[0], dims[1], masks.shape[-1]))
roi_contours = rutil.get_roi_contours(mask_r, roi_axis=-1)
nrois = masks.shape[-1]

# PLOT:
fig, ax = pl.subplots(1,1, figsize=(10,10))
cmap = pl.cm.jet # define colormap
cmap_list = [cmap(i) for i in range(cmap.N)] # extract all colors
cmap = cmap.from_list('roi_cmap', cmap_list, cmap.N)
bounds = np.linspace(0, nrois, nrois+1)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

roi_cmap = [cmap_list[norm(i)] for i in range(nrois)]

rutil.plot_roi_contours(fov_image, roi_contours, ax=ax, thickness=0.1, label_all=False, clip_limit=0.02,
                        cmap=roi_cmap)
pl.axis('off')

ax2 = fig.add_axes([0.9, 0.1, 0.03, 0.8])
cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm, spacing='proportional', ticks=bounds, boundaries=bounds, format='%1i')
cb.set_ticks([0, nrois])

label_figure(fig, data_identifier)

pl.savefig(os.path.join(output_dir, 'roi_contours.png'))


coord_info = T.get_coordinates(masks, dims)

fig, ax = pl.subplots(1, figsize=(10, 10))
ax.imshow(fov_image, cmap='gray')
for ri in range(len(coord_info)):
    xp = coord_info[ri]['CoM'][0]
    yp = coord_info[ri]['CoM'][1]
    ax.plot(xp, yp, 'r.')
    ax.text(xp+1, yp+1, str(ri+1), color='blue', fontsize=8)
pl.axis('off')
label_figure(fig, data_identifier)

pl.savefig(os.path.join(output_dir, 'roi_coms.png'))

coord_outpath = os.path.join(output_dir, 'coordinates.pkl')
with open(coord_outpath, 'wb') as f:
    pkl.dump(coord_info, f, protocol=pkl.HIGHEST_PROTOCOL)
    