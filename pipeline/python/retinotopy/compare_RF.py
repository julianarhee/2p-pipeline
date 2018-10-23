#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 19:49:15 2018

@author: juliana
"""
import os
import json
import glob

import cPickle as pkl
import pylab as pl
import seaborn as sns
import numpy as np
import pandas as pd

from pipeline.python.visualization.plot_session_summary import SessionSummary
from pipeline.python.retinotopy.split_fov import load_retino_id
from pipeline.python.utils import natural_keys, label_figure


#%%
def get_retinotopy_sconfigs(acquisition_dir, retino_run='retino_'):
    print 'Getting paradigm file info'
    paradigm_fpath = glob.glob(os.path.join(acquisition_dir, '%s*' % retino_run, 'paradigm', 'files', '*.json'))[0]
    with open(paradigm_fpath, 'r') as r: mwinfo = json.load(r)
    # pp.pprint(mwinfo)
    
    rep_list = [(k, v['stimuli']['stimulus']) for k,v in mwinfo.items()]
    unique_conditions = np.unique([rep[1] for rep in rep_list])
    conditions = dict((cond, [int(run) for run,config in rep_list if config==cond]) for cond in unique_conditions)
    print conditions
    sconfigs = dict((cond, {'files': files}) for cond, files in conditions.items() )
    
    start_pos_az = list(set([mwinfo[str(t)]['stimuli']['position'][0] for t in conditions['right']]))
    start_pos_el = list(set([mwinfo[str(t)]['stimuli']['position'][1] for t in conditions['top']]))
    stim_freq = list(set([mw['stimuli']['scale'] for c,mw in mwinfo.items()]))
    if len(start_pos_az) == 1:
        sconfigs['right']['position'] = (start_pos_az[0], 0.)
    else:
        sconfigs['right']['position'] = [(posx, 0.) for posx in start_pos_az]
    if len(start_pos_el) == 1:
        sconfigs['top']['position'] = (0., start_pos_el[0])
    else:
        sconfigs['top']['position'] = [(0., posy) for posy in start_pos_el]
    assert len(stim_freq) == 1, "More than 1 stim frequency found: %s" % str(stim_freq)
    sconfigs['right']['frequency'] = stim_freq[0]
    sconfigs['top']['frequency'] = stim_freq[0]
    
    return sconfigs


#%%
class struct():
    pass 

rootdir = '/n/coxfs01/2p-data'

#%%

animalid = 'JC022'
session = '20181005'
acquisition = 'FOV3_zoom2p7x'

acquisition_dir = os.path.join(rootdir, animalid, session, acquisition)

# Create class for VISUAL AREA
class FOV():
    def __init__(self, animalid, session, acquisition, rootdir='/n/coxfs01/2p-data'):
        self.rootdir = rootdir
        self.animalid = animalid
        self.session = session
        self.acquisition = acquisition
        
        self.run_list = self.get_run_list()
        
        self.runs = struct()
        
    def get_run_list(self):
        acquisition_dir = os.path.join(self.rootdir, self.animalid, self.session, self.acquisition)
        
        all_runs = [run for run in os.listdir(acquisition_dir) if os.path.isdir(os.path.join(acquisition_dir, run)) and run!='anatomical']
        run_list = []
        if any(['combined' in run for run in all_runs]):
            combo_runs = [run for run in all_runs if 'combined' in run]
            for combo_run in combo_runs:
                stim_type = combo_run.split('_')[1]
                single_runs_replace = [run for run in all_runs if 'combined' not in run and stim_type in run]
                run_list.extend([run for run in all_runs if run not in single_runs_replace])
        else:
            run_list.extend(all_runs)
        run_list = list(set(run_list))
        
        return run_list
    
    def get_run_data_path(self, run):
        # We only care about traces that have been fully extracted, so look for data arrays:
        darray_paths = sorted(glob.glob(os.path.join(self.rootdir, self.animalid, self.session, self.acquisition,
                                              run, 'traces', 'traces*', 'data_arrays', '*.npz')), key=natural_keys)
        if len(darray_paths) > 1:
            print "Found %i extracted traces sets:" % len(darray_paths)
            for di, dpath in enumerate(darray_paths):
                print di, dpath
            selected_di = input("Select IDX of traces ID to use: ")
            data_fpath = darray_paths[selected_di]
        else:
            data_fpath = darray_paths[0]
                
        return data_fpath

    def load_run_data(self):
        
        for run in self.run_list:            
            if 'retino' not in run:
                data_fpath = self.get_run_data_path(run)
                traceid = data_fpath.split('/traces/')[-1].split('/')[0]
                dset = np.load(data_fpath)
                
                labels_df = pd.DataFrame(data=dset['labels_data'], columns=dset['labels_columns'])
                sconfigs_df = pd.DataFrame(dset['sconfigs'][()]).T
                data_array = dset['corrected']
                
            else:
                # Load retino shit:
                roi_analyses = sorted(glob.glob(os.path.join(rootdir, animalid, session, acquisition, 
                                                                 'retino*', 'retino_analysis', 'analysis*', 'visualization', '*.json')), key=natural_keys)
                if len(roi_analyses) > 1:
                    print "More than 1 roi-based analysis found:"
                    for ri, ranalysis in enumerate(roi_analyses):
                        print ri, ranalysis
                    selected_ri = input('Select IDX of roi analysis to use:')
                    retino_traceid_dir = roi_analyses[selected_ri].split('/visualization')[0]
                else:
                    assert len(roi_analyses) == 1, "No ROI based analyses found..."
                    retino_traceid_dir = roi_analyses[0].split('/visualization')[0]
                
                traceid = os.path.split(retino_traceid_dir)[-1].split('_')[0]
                retino_run = os.path.split(retino_traceid_dir.split('/retino_analysis/')[0])[-1]
                data_fpath = glob.glob(os.path.join(retino_traceid_dir, 'traces', '*.pkl'))
                if len(data_fpath) == 1:
                    with open(data_fpath[0], 'rb') as f:
                        dset = pkl.load(f)
                    data_array = dset['traces']['right']['traces']
                else:
                    data_array = None
                    
                sconfigs = get_retinotopy_sconfigs(acquisition_dir, retino_run=retino_run)
                sconfigs_df = pd.DataFrame(sconfigs).T
                
            rundata = {'data_fpath': data_fpath,
                       'traceid': traceid,
                       'data': data_array,
                       'labels': labels_df,
                       'sconfigs': sconfigs_df}

            if 'retino' in run:
                self.runs.retinotopy = rundata
            elif 'gratings' in run:
                self.runs.gratings = rundata
            elif 'blobs' in run:
                self.runs.blobs = rundata
            elif 'objects' in run:
                self.runs.objects = rundata
        
        
#%%
# Combine different conditions of the SAME acquisition:
animalid = 'JC015'
session = '20180919'
acquisition = 'FOV1_zoom2p0x'
retino_run = 'retino_run1'

#use_azimuth = True
#use_single_ref = True
#retino_file_ix = 0

cmap = cm.Spectral_r

#%%  if split fov:

visualareas_fpath = os.path.join(rootdir, animalid, session, acquisition, 'visual_areas', 'visual_areas.pkl')

if os.path.exists(visualareas_fpath):
    with open(visualareas_fpath, 'rb') as f:
        fov = pkl.load(f)


#%%
ss_path = glob.glob(os.path.join(rootdir, animalid, session, acquisition, 'session_summary_*%s_retino*.pkl' % acquisition))[0]
with open(ss_path, 'rb') as f:
    S = pkl.load(f)

#TODO:   Create RF object (since SessionSummary() might not exist, or is not finished running...)
 
fit_thr = 0.5
nrois_total = len(S.retinotopy['data'])
retino_good_rois = [roi for roi in range(nrois_total) if all([S.retinotopy['data'][roi].conditions[ci].fit_results['r2'] >= fit_thr for ci in range(2)])]
print "--- %i out of %i ROIs fit for retinotopy RF estimates." % (len(retino_good_rois), nrois_total)


retino_az_cond = [c for c in range(2) if S.retinotopy['data'][roi].conditions[c].name == 'right'][0]
retino_el_cond = [c for c in range(2) if S.retinotopy['data'][roi].conditions[c].name == 'top'][0]

retino_widths = [(S.retinotopy['data'][roi].conditions[retino_az_cond].fit_results['sigma'], S.retinotopy['data'][roi].conditions[retino_el_cond].fit_results['sigma']) 
                        for roi in retino_good_rois]


retino_rdict = dict((roi, {'width_x': S.retinotopy['data'][roi].conditions[retino_az_cond].fit_results['sigma'], 
                           'width_y': S.retinotopy['data'][roi].conditions[retino_el_cond].fit_results['sigma'], 
                           'r2_x': S.retinotopy['data'][roi].conditions[retino_az_cond].fit_results['r2'], 
                           'r2_y': S.retinotopy['data'][roi].conditions[retino_el_cond].fit_results['r2']}) for roi in retino_good_rois)
retino_df = pd.concat([pd.DataFrame(data=rdict.values(), columns=[roi], index=rdict.keys()) for roi, rdict in retino_rdict.items()], axis=1)


#%%

retinoID = load_retino_id(os.path.join(fov.source.rootdir, fov.source.animalid, fov.source.session, fov.source.acquisition, fov.source.run), fov.source.retinoID_rois)
roi_masks = fov.get_roi_masks(retinoID)
nrois = roi_masks.shape[-1]
roi_contours = get_roi_contours(roi_masks, roi_axis=-1)


#%%

LI_mask2 = fov.regions['LI']['region_mask']

LI_mask_copy2 = np.copy(LI_mask2)
LI_mask_copy2[LI_mask2==0] = np.nan

LI_rois2 = [ri for ri in range(nrois) if ((roi_masks[:, :, ri] + LI_mask_copy2) > 1).any()]

LI_widths2 = [np.mean( [res['width_x'], res['width_y']] ) for roi, res in retino_rdict.items() if roi in LI_rois2 ]



#%%

LM_rois2 = retino_good_rois #[ri for ri in range(nrois) if ((roi_masks[:, :, ri] + LI_mask_copy2) > 1).any()]
LM_widths2 = [np.mean( [res['width_x'], res['width_y']] ) for roi, res in retino_rdict.items() if roi in LM_rois2 ]

#%%
# Mask ROIs with area mask:
LI_mask = fov.regions['LI']['region_mask']

LI_mask_copy = np.copy(LI_mask)
LI_mask_copy[LI_mask==0] = np.nan

LI_rois1 = [ri for ri in range(nrois) if ((roi_masks[:, :, ri] + LI_mask_copy) > 1).any()]


# Mask ROIs with area mask:
LM_mask = fov.regions['LM']['region_mask']

LM_mask_copy = np.copy(LM_mask)
LM_mask_copy[LM_mask==0] = np.nan

LM_rois1 = [ri for ri in range(nrois) if ((roi_masks[:, :, ri] + LM_mask_copy) > 1).any()]

#%
#% 
LM_widths1 = [np.mean( [res['width_x'], res['width_y']] ) for roi, res in retino_rdict.items() if roi in LM_rois1 ]
LI_widths1 = [np.mean( [res['width_x'], res['width_y']] ) for roi, res in retino_rdict.items() if roi in LI_rois1 ]


#%%
LI = list(np.copy(LI_widths1))
LI.extend(LI_widths2)
print len(LI)

#%%

LM = list(np.copy(LM_widths1))
LM.extend(LM_widths2)
print len(LM)

#%%
color1 = 'g'
color2 = 'magenta'

fig, ax = pl.subplots(1, figsize=(8,5))

sns.distplot(LM, label='LM', ax=ax,
              rug=True, rug_kws={"color": color1},
              kde_kws={"color": color1, "lw": 3, "label": "KDE", "alpha": 0.5},
              hist_kws={"histtype": "step", "linewidth": 1, "alpha": 0.2, "color": color1})

sns.distplot(LI, label='LI', ax=ax,
              rug=True, rug_kws={"color": color2},
              kde_kws={"color": color2, "lw": 3, "label": "KDE", "alpha": 0.5},
              hist_kws={"histtype": "step", "linewidth": 1, "alpha": 0.2, "color": color2})
ax.set_xlabel('mean RF widths')

