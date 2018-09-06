#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 14:54:08 2018

@author: julianarhee
"""


import matplotlib as mpl
mpl.use('agg')
import os
import sys
import optparse
import seaborn as sns
import numpy as np
import pandas as pd
import pylab as pl
from scipy import stats

from pipeline.python.paradigm import utils as util


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
                          default='', help='session dir (format: YYYYMMDD')
    parser.add_option('-A', '--acq', action='store', dest='acquisition',
                          default='FOV1_zoom1x', help="acquisition folder [default: FOV1_zoom1x]")
    parser.add_option('-T', '--trace-type', action='store', dest='trace_type',
                          default='raw', help="trace type [default: 'raw']")
    parser.add_option('-R', '--run', dest='run', default='', action='store', help="run name")
    parser.add_option('-t', '--traceid', dest='traceid', default='traces001', action='store', help="trace ID (default: traces001)")
    
    # Set specific session/run for current animal:
#    parser.add_option('-d', '--datatype', action='store', dest='datatype',
#                          default='corrected', help='Traces to plot (must be in dataset.npz [default: corrected]')

    
    (options, args) = parser.parse_args(options)
    if options.slurm:
        options.rootdir = '/n/coxfs01/2p-data'
    
    return options


#%%


options = ['-D', '/Volumes/coxfs01/2p-data', '-i', 'CE077', 
           '-S', '20180724', '-A', 'FOV1_zoom1x',
           '-R', 'gratings_drifting_static', '-t', 'cnmf_20180802_12_20_12']




#%%

optsE = extract_options(options)

#traceid_dir = util.get_traceid_dir(options)
run = optsE.run
traceid = optsE.traceid
acquisition_dir = os.path.join(optsE.rootdir, optsE.animalid, optsE.session, optsE.acquisition)

traceid_dir = util.get_traceid_from_acquisition(acquisition_dir, run, traceid)
data_fpath = os.path.join(traceid_dir, 'data_arrays', 'datasets.npz')
dataset = np.load(data_fpath)
print "Loaded data from: %s" % traceid_dir

#%%


S = dataset['spikes']
S[S<=0.0002] = 0.



F = dataset['dff']


framerate = 44.69


ridx = 1

# Plot DFF and spikes on same plot:
fig, ax1 = pl.subplots()
t = np.arange(0, 3000)

m = F[t, ridx].max()
cm = S[t, ridx].max()


 scaling_factor = m / cm
spikes_scaled = S[t, ridx] * scaling_factor

ax1.plot(t, F[t, ridx], 'b-')
ax1.set_xlabel('frames (%.2f Hz' % framerate)
# Make the y-axis label, ticks and tick labels match the line color.
ax1.set_ylabel('dff', color='b')
ax1.tick_params('y', colors='b')

ax2 = ax1.twinx()
#ax2.plot(t, spikes_scaled, 'r-')
ax2.plot(t, S[t, ridx], 'r-')

ax2.set_ylabel('inferred spikes', color='r')
ax2.tick_params('y', colors='r')

fig.tight_layout()
pl.show()



#% Plot firing rate:
tbin = 200 # ms
nframes_bin = int(round(tbin / ((1./framerate) * 1E3)))

run_info = dataset['run_info'][()]
nframes_per_trial = run_info['nframes_per_trial']
ntrials_per_block = 10
nframes = sum(nframes_per_trial * ntrials_per_block)

print "Binning %i frames for a total of %.3f bins per block." % (nframes_bin, nframes/nframes_bin)

chunks = np.arange(0, nframes, step=nframes_bin)

spike_train = S[0:nframes, ridx]


spike_rate = [sum(spike_train[chunk_start:chunk_start+nframes_bin])/1E3 for chunk_start in chunks]

tpoints = np.arange(0, nframes) / framerate
cpoints = chunks / framerate

fig, axes = pl.subplots(3, 1)
axes[0].plot(tpoints, F[0:nframes, ridx], 'b'); axes[0].set_ylabel('df/f')
axes[1].plot(tpoints, spike_train, 'k'); axes[1].set_ylabel('inferrred spikes')
axes[2].plot(cpoints, spike_rate, 'r'); axes[2].set_ylabel('rate (Hz)')



#%%



