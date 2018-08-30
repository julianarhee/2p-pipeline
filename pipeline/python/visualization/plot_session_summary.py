#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 16:41:33 2018

@author: juliana
"""

import os
import glob
import optparse
import json
import h5py
import datetime
import matplotlib
matplotlib.use('agg')

import pandas as pd
import pylab as pl
import numpy as np
import pandas as pd
import seaborn as sns
import tifffile as tf
from PIL import Image, ImageChops
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from pipeline.python.utils import natural_keys, replace_root, label_figure
from pipeline.python.paradigm import utils as util
from pipeline.python.classifications.test_responsivity import calculate_roi_responsivity, group_roidata_stimresponse, find_barval_index
from pipeline.python.classifications import osi_dsi as osi
from pipeline.python.classifications import linear_svc as lsvc
from pipeline.python.traces.utils import load_TID

from collections import Counter


def load_traceid_zproj(traceid_dir, rootdir=''):
    run_dir = traceid_dir.split('/traces')[0]
    traceid = os.path.split(traceid_dir)[-1].split('_')[0]
    TID = load_TID(run_dir, traceid)
    
    session_dir = os.path.split(os.path.split(run_dir)[0])[0]
    session = os.path.split(session_dir)[-1]
    with open(os.path.join(session_dir, 'ROIs', 'rids_%s.json' % session), 'r') as f:
        rdict = json.load(f)
    
    RID = rdict[TID['PARAMS']['roi_id']]
    ref_file = 'File%03d' % int(RID['PARAMS']['options']['ref_file'])
    
    maskpath = os.path.join(traceid_dir, 'MASKS.hdf5')
    masks = h5py.File(maskpath, 'r')
    
    # Plot on MEAN img:
    if ref_file not in masks.keys():
        ref_file = masks.keys()[0]
    img_src = masks[ref_file]['Slice01']['zproj'].attrs['source']
    if 'std' in img_src:
        img_src = img_src.replace('std', 'mean')
    if rootdir not in img_src:
        animalid = traceid_dir.split(optsE.rootdir)[-1].split('/')[1]
        session = traceid_dir.split(optsE.rootdir)[-1].split('/')[2]
        img_src = replace_root(img_src, rootdir, animalid, session)
    
    img = tf.imread(img_src)

    return img    


def trim_whitespace(im, scale=2):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, scale, -1)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)
    
    
def colorcode_histogram(bins, ppatches, color='m'):

    indices_to_label = [find_barval_index(v, ppatches) for v in bins]
    for ind in indices_to_label:
        ppatches.patches[ind].set_color(color)
        ppatches.patches[ind].set_alpha(0.5)
    
    
    
options = ['-D', '/mnt/odyssey', '-i', 'CE077', '-S', '20180817', '-A', 'FOV2_zoom1x',
           '-d', 'corrected',
           '-R', 'gratings_drifting', '-t', 'traces001',
           ]


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
    parser.add_option('-d', '--trace-type', action='store', dest='trace_type',
                          default='corrected', help="trace type [default: 'corrected']")

    parser.add_option('-R', '--run', dest='run', default='', action='store', help="run name")
    parser.add_option('-t', '--traceid', dest='traceid', default=None, action='store', help="datestr YYYYMMDD_HH_mm_SS")

    parser.add_option('--par', action='store_true', dest='multiproc', default=False, help="set if want to run MP on roi stats, when possible")
    parser.add_option('--nproc', action='store', dest='nprocesses', default=4, help="N processes if running in par (default=4)")
    parser.add_option('--new', action='store_true', dest='create_new', default=False, help="set to run anew")

    (options, args) = parser.parse_args(options)

    return options

#%%

options = ['-D', '/mnt/odyssey', '-i', 'CE077', '-S', '20180817', '-A', 'FOV2_zoom1x',
           '-d', 'corrected',
           '-R', 'gratings_drifting', '-t', 'traces001',
           ]

optsE = extract_options(options)

acquisition_dir = os.path.join(optsE.rootdir, optsE.animalid, optsE.session, optsE.acquisition) 
traceid_dir = util.get_traceid_from_acquisition(acquisition_dir, optsE.run, optsE.traceid)
traceid = os.path.split(traceid_dir)[-1]
sort_dir = os.path.join(traceid_dir, 'sorted_rois')
data_identifier ='_'.join([optsE.rootdir, optsE.animalid, optsE.session])

# Load data array:
data_fpath = os.path.join(traceid_dir, 'data_arrays', 'datasets.npz')
dataset = np.load(data_fpath)

# Load ROI stats:
try:
    roistats_fpath = sorted(glob.glob(os.path.join(acquisition_dir, optsE.run, 'traces', traceid, 'sorted_rois', 'roistats_results*.npz')))[-1]
    roistats = np.load(roistats_fpath)
    rois_visual = roistats['sorted_visual']
    rois_selective = roistats['sorted_selective']
    
except Exception as e:
    print "** No roi stats found... Testing responsivity now."
    roistats_fpath = calculate_roi_responsivity(options)

roidata = dataset['corrected']
labels_df = pd.DataFrame(data=dataset['labels_data'], columns=dataset['labels_columns'])
df_by_rois = group_roidata_stimresponse(roidata, labels_df)
nrois_total = roidata.shape[-1]
sconfigs = dataset['sconfigs'][()]



# Create histogram of all ROIs, use z-score value:
max_zscores_all = pd.Series([roidf.groupby('config')['zscore'].mean().max() for roi, roidf in df_by_rois])
max_zscores_visual = [df_by_rois.get_group(roi).groupby('config')['zscore'].mean().max() for roi in rois_visual]
max_zscores_selective = [df_by_rois.get_group(roi).groupby('config')['zscore'].mean().max() for roi in rois_selective]

#%%
fig, axes = pl.subplots(2,3, figsize=(20,15)) #pl.figure()
axes_flat = axes.flat
hist_ix = 1
zproj_ix = 0
retino_ix = 2
osi_ix = 3
gratings_ix = 4

# SUBPLOT 1:  Histogram of visual vs. selective ROIs:
# -----------------------------------------------------------------------------
count, division = np.histogram(max_zscores_all, bins=100)
ppatches = max_zscores_all.hist(bins=division, color='gray', ax=axes_flat[hist_ix], grid=False)

# Highlight p-eta2 vals for significant neurons:
visual_bins = list(set([binval for ix,binval in enumerate(division) for zscore in max_zscores_visual if division[ix] < zscore <= division[ix+1]]))
selective_bins = list(set([binval for ix,binval in enumerate(division) for zscore in max_zscores_selective if division[ix] < zscore <= division[ix+1]]))

colorcode_histogram(visual_bins, ppatches, color='magenta')
colorcode_histogram(selective_bins, ppatches, color='cornflowerblue')
#axes_flat[hist_ix].set_xlabel('zscore')
axes_flat[hist_ix].set_ylabel('counts')
axes_flat[hist_ix].set_title('distN of zscores (all rois)', fontsize=10)

# Label histogram colors:
visual_pval= 0.05; visual_test = str(roistats['responsivity_test']).split('_')[-1];
visual_str = 'visual: %i/%i (p<%.2f, %s)' % (len(rois_visual), nrois_total, visual_pval, visual_test)
texth = axes_flat[hist_ix].get_ylim()[-1] + 2
textw = axes_flat[hist_ix].get_xlim()[0] + .1
axes_flat[hist_ix].text(textw, texth, visual_str, fontdict=dict(color='magenta', size=10))

selective_pval = 0.05; selective_test = str(roistats['selectivity_test']);
selective_str = 'sel: %i/%i (p<%.2f, %s)' % (len(rois_selective), len(rois_visual), selective_pval, selective_test)
axes_flat[hist_ix].text(textw, texth-1, selective_str, fontdict=dict(color='cornflowerblue', size=10))
axes_flat[hist_ix].set_ylim([0, texth + 2])
axes_flat[hist_ix].set_xlim([0, axes_flat[hist_ix].get_xlim()[-1]])

sns.despine(trim=True, offset=4, ax=axes_flat[hist_ix])


# SUBPLOT 2:  Mean / zproj image
# -----------------------------------------------------------------------------
zproj = load_traceid_zproj(traceid_dir, rootdir=optsE.rootdir)
axes_flat[zproj_ix].imshow(zproj, cmap='gray')
axes_flat[zproj_ix].axis('off')


# SUBPLOT 3:  Retinotopy:
# -----------------------------------------------------------------------------
retinovis_fpath = glob.glob(os.path.join(optsE.rootdir, optsE.animalid, optsE.session, optsE.acquisition, 
                                         'retino_*', 'retino_analysis', 'analysis*', 'visualization', '*.png'))[0]
retino = Image.open(retinovis_fpath)
retino = trim_whitespace(retino, scale=8)
axes_flat[retino_ix].imshow(retino)
axes_flat[retino_ix].axis('off')

#%

# SUBPLOT 4:  DistN of preferred orientations:
# -----------------------------------------------------------------------------
metric = 'meanstim'
selectivity = osi.get_OSI_DSI(df_by_rois, sconfigs, roi_list=rois_visual, metric=metric)
cmap = 'hls'
colorvals = sns.color_palette(cmap, len(sconfigs))
osi.hist_preferred_oris(selectivity, colorvals, metric=metric, sort_dir=sort_dir, save_and_close=False, ax=axes_flat[osi_ix])
sns.despine(trim=True, offset=4, ax=axes_flat[osi_ix])



# SUBPLOT 5:  Decoding performance for linear classifier on orientations:
# -----------------------------------------------------------------------------
clfparams = lsvc.get_default_gratings_params()
cX, cy, inputdata, is_cnmf = lsvc.get_formatted_traindata(clfparams, dataset, traceid)
cX_std = StandardScaler().fit_transform(cX)
if cX_std.shape[0] > cX_std.shape[1]: # nsamples > nfeatures
    clfparams['dual'] = False
else:
    clfparams['dual'] = True
cX, cy, clfparams['class_labels'] = lsvc.format_stat_dataset(clfparams, cX_std, cy, sconfigs, relabel=False)
    
if clfparams['classifier'] == 'LinearSVC':
    svc = LinearSVC(random_state=0, dual=clfparams['dual'], multi_class='ovr', C=clfparams['C'])

predicted, true = lsvc.do_cross_validation(svc, clfparams, cX_std, cy, data_identifier=data_identifier)
lsvc.plot_normed_confusion_matrix(predicted, true, clfparams, ax=axes_flat[gratings_ix])
        
        

# SUBPLOT 6:  Complex stimuli...
# -----------------------------------------------------------------------------




