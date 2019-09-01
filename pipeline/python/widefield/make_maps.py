#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 15:54:26 2019

@author: julianarhee
"""

import glob
import os
import shutil

import numpy as np
import pylab as pl
from mpl_toolkits.axes_grid1 import make_axes_locatable



animalid = 'JC076'
session = '20190404'

outdir = os.path.join('/n/coxfs01/julianarhee/aggregate-visual-areas/widefield-maps', 'figures')
if not os.path.exists(outdir):
    os.makedirs(outdir)

rootdir = '/n/coxfs01/widefield-data'
subdirs = 'analyzed_data/Retinotopy/phase_encoding/Images_Cartesian_Constant'
analysis_subdirs = 'Analyses/timecourse/not_motion_corrected/excludeEdges_averageFrames_11_minusRollingMean'

datapaths = glob.glob(os.path.join(rootdir, subdirs, animalid, '%s*' % session, analysis_subdirs, 'Files', '*.npz'))


data = np.load(datapaths[0])
data.keys()

d1, d2 = data['szX'], data['szY']

_, nframes = data['frameArrayGroupedAvg'].shape

farray = np.reshape(data['frameArrayGroupedAvg'], (d2, d1, nframes))

fig, ax = pl.subplots()
im = ax.imshow(farray.mean(axis=-1), cmap='gray')
#divider = make_axes_locatable(ax)
#cax = divider.append_axes("right", size='2%', pad=0.1, shrink=.5)
fig.colorbar(im, label='dF/F', shrink=0.5, pad=0.05)
ax.axis('off')

pl.savefig(os.path.join(outdir, '%s_%s_dFF_azimuth.png' % (animalid, session)))

root2p = '/n/coxfs01/2p-data'
macro_2psurf = glob.glob(os.path.join(root2p, animalid, 'macro_maps', session, '*Surf*.png'))[0]
fname_new = '%s_%s_surface.png' % (animalid, session)
shutil.copy(macro_2psurf, os.path.join(outdir, fname_new))


#%%


midp = int(farray.shape[-1])/2
tchunks = np.arange(0, midp, 10)


s_ix1 = 60
e_ix1 = 80

#s = np.where(tchunks==s_ix1)[0]
#e= np.where(tchunks==e_ix1)[0]

s_ix2 = 120 #int(tchunks[len(tchunks)-e+5])
e_ix2 = 140 #int(tchunks[len(tchunks)-s+5])


nperiods = data['groupPeriods']
degpos = np.linspace(0, 120, farray.shape[-1]/nperiods)
pos1_deg = degpos[s_ix1:e_ix1]
pos2_deg = degpos[s_ix2:e_ix2]


ixs1 = np.hstack([np.arange(s_ix1, e_ix1), np.arange(midp+s_ix1, midp+e_ix1)])
ixs2 = np.hstack([np.arange(s_ix2, e_ix2), np.arange(midp+s_ix2, midp+e_ix2)])

fig, axes = pl.subplots(1, 2)
axes[0].imshow(farray[:, :, ixs1].mean(axis=-1), cmap='gray')
axes[1].imshow(farray[:, :, ixs2].mean(axis=-1), cmap='gray')
for ax in axes:
    ax.axis('off')
axes[0].set_title("Pos1: %.2f, %.2f" % (pos1_deg[0], pos1_deg[-1]))
axes[1].set_title("Pos2: %.2f, %.2f" % (pos2_deg[0], pos2_deg[-1]))

pl.savefig(os.path.join(outdir, '%s_%s_example_positions_azimuth.png' % (animalid, session)))



#%%

quantilemin = 10.
F0 = np.percentile(np.reshape(farray, (d2*d1, nframes)), quantilemin, axis=-1)


darray = (farray - F0.mean() ) / F0.mean()
pl.figure()
pl.imshow(darray.mean(axis=-1), cmap='gray')
pl.colorbar()













tf.imsave(farray, )