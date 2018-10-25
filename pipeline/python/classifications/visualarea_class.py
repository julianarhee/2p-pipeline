#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 14:18:57 2018

@author: juliana
"""

import os
import glob
import json

from pipeline.python.utils import natural_keys, label_figure
from pipeline.python.retinotopy import estimate_RF_size as rf

import cPickle as pkl
import seaborn as sns
import pylab as pl
import numpy as np

#rootdir = '/n/coxfs01/2p-data'
#animalid = 'CE077'
#excluded_sessions = ['20180525','20180321', '20180331', '20180423']

#%%
#session_list = sorted([os.path.split(session_dir)[-1] for session_dir in glob.glob(os.path.join(rootdir, animalid, '2018*')) \
#                    if os.path.split(session_dir)[-1] not in excluded_sessions \
#                    and len(os.path.split(session_dir)[-1].split('_'))==1], key=natural_keys)
#
#session = session_list[0]
#
#fov_list = sorted([os.path.split(fov_dir)[-1] for fov_dir in glob.glob(os.path.join(rootdir, animalid, session, 'FOV*')) \
#                        if os.path.isdir(fov_dir)], key=natural_keys)
#                            
#fov = fov_list[0]

#%%


#
#class SESSION():
#    def __init__(self, animalid, session, rootdir='/n/coxfs01/2p-data'):
#        self.rootdir  = rootdir
#        self.animalid = animalid
#        self.session = session
#        
#        self.FOVs = []
#        self.fov_list = self.get_fovs()
#        
#    def get_fovs(self):
#        fov_list = sorted([os.path.split(fov_dir)[-1] for fov_dir in glob.glob(os.path.join(self.rootdir, self.animalid, self.session, 'FOV*')) \
#                        if os.path.isdir(fov_dir)], key=natural_keys)
#    
#        for fov in fov_list:
#            self.FOVs.append(FOV(animalid, session, fov, rootdir=self.rootdir))
#            
#        return fov_list
#    
#    
    
class ANIMAL():
    def __init__(self, animalid, excluded_sessions=[], rootdir='/n/coxfs01/2p-data'):
        self.rootdir  = rootdir
        self.animalid = animalid
        self.sessions = {} 
        self.excluded_sessions = excluded_sessions
        self.session_list = self.get_sessions()

#        self.sessions = {} #dict(('%s_%s' % (sesh, fov), []) for sesh, fov in self.session_list.items() )
    
    def get_sessions(self):
        session_list = {}
        session_dirs = sorted([os.path.split(session_dir)[-1] for session_dir in glob.glob(os.path.join(self.rootdir, self.animalid, '2018*')) \
                            if os.path.split(session_dir)[-1] not in self.excluded_sessions \
                            and len(os.path.split(session_dir)[-1].split('_'))==1], key=natural_keys)
    
        for session in session_dirs:
            session_list[session] = self.get_fovs_in_session(session)
            
        return session_list
    
    def get_fovs_in_session(self, session):
        fov_list = sorted([os.path.split(fov_dir)[-1] for fov_dir in glob.glob(os.path.join(self.rootdir, self.animalid, session, 'FOV*')) \
                        if os.path.isdir(fov_dir)], key=natural_keys)
        excluded_fovs = []
        for fov in fov_list:
            fovclass = FOV(self.animalid, session, fov, rootdir=self.rootdir)
            if fovclass.retino_id is None:
                excluded_fovs.append(fovclass.acquisition)
                continue
            
            fovkey = '%s_%s' % (session, fov)
#            if fovkey not in self.sessions.keys():
#                self.sessions[fovkey] = []
            
            self.sessions[fovkey] = fovclass
            
        final_fovs = [f for f in fov_list if f not in excluded_fovs]
        
        return final_fovs
    
class FOV():
    def __init__(self, animalid, session, acquisition, rootdir='/n/coxfs01/2p-data'):
        self.rootdir  = rootdir
        self.animalid = animalid
        self.session = session
        self.acquisition = acquisition
        self.RFs = [] 
        self.retino_id = self.get_roi_retinotopy()
         
        if self.retino_id is not None:
            self.roi_id = self.retino_id['PARAMS']['roi_id']
            self.run_list = self.get_run_list()
        
        #self.RFs = []
        
    def estimate_RFs_by_roi(self, fitness_thr=0.4, size_thr=0.1):
        acquisition_dir = os.path.join(self.rootdir, self.animalid, self.session, self.acquisition)
        print "retino id:", self.retino_id['analysis_id']
        RF_estimates, _ = rf.get_RF_size_estimates(acquisition_dir, fitness_thr=fitness_thr, size_thr=fitness_thr, analysis_id=self.retino_id['analysis_id'])
        
        return RF_estimates
        
    def get_roi_retinotopy(self):
        rid_fpaths = sorted(glob.glob(os.path.join(self.rootdir, self.animalid, self.session, self.acquisition, 'retino*', 'retino_analysis', 'analysisids*.json')), key=natural_keys)
        if len(rid_fpaths)==0:
            print "%s: No retino IDs found!" % self.session
            return None
        
        if len(rid_fpaths) > 1:
            print "Found multiple ROI retino analyses:"
            for ri, rpath in enumerate(rid_fpaths):
                print ri, rpath
            rid_dict_fpath = rid_fpaths[int(input('Select IDX of roi ID to use: '))]
        else:
            rid_dict_fpath = rid_fpaths[0]
        with open(rid_dict_fpath, 'r') as f: rids = json.load(f)
        
        roi_ids = sorted([k for k,v in rids.items() if v['PARAMS']['roi_type'] != 'pixels'], key=natural_keys)
        assert len(roi_ids) > 0, "[RETINO]: No ROI based analyses found!"
        if len(roi_ids) > 1:
            print "... Multiple ROI analyses found:"
            for ri, rid in enumerate(roi_ids):
                print '...', ri, rid
            retino_rid = roi_ids[input('... Select IDX of roi ID to use: ')]
        else:
            retino_rid = roi_ids[0]
        
        retino_id = rids[retino_rid]
        
        return retino_id
    
    def get_run_list(self):
        run_dirs = sorted([rundir for rundir in glob.glob(os.path.join(self.rootdir, self.animalid, self.session, self.acquisition, '*run*')) \
                           if 'retino' not in rundir and os.path.isdir(rundir)], key=natural_keys)
    
        run_names = [os.path.split(rundir)[-1] for rundir in run_dirs]
        
        return run_names
    
#%%
#    

def get_animal(animalid, excluded_sessions=[], rootdir='/n/coxfs01/2p-data'):
    A = ANIMAL(animalid, excluded_sessions=excluded_sessions, rootdir=rootdir)
    for session, fov in A.sessions.items():
        print session, len(fov.run_list)
        subdir = os.path.join(A.rootdir, A.animalid, 'fovs')
        if not os.path.exists(subdir): os.makedirs(subdir)
        fov_fpath = os.path.join(subdir, '%s.pkl' % session)
        if os.path.exists(fov_fpath):
            with open(fov_fpath, 'rb') as f:
                fov = pkl.load(f)
        if 'RFs' not in dir(fov):
            fov.RFs = []
        if len(fov.RFs) == 0:
            print "empty FOVs"
            RFs = fov.estimate_RFs_by_roi()
            fov.RFs.append(RFs)
        if 'conditions' not in dir(fov.RFs[0]):
            if 'conditions' in dir(fov.RFs[0][0]):
                rlist = fov.RFs[0]
                fov.RFs = rlist
        with open(os.path.join(subdir, '%s.pkl' % session), 'wb') as f:
            pkl.dump(fov, f, protocol=pkl.HIGHEST_PROTOCOL)
        print "Finished: %s" % session, "N rois", len(fov.RFs)
        A.sessions[session] = fov

    ## SAVE:
    #animal_fpath = os.path.join(rootdir, animalid, 'FOVS.pkl')
    #with open(animal_fpath, 'wb') as f:
    #    pkl.dump(A, f, protocol=pkl.HIGHEST_PROTOCOL)

    return A 
      
def get_good_RFs(A, fit_thr=0.5, rootdir='/n/coxfs01/2p-data'):
    rf_fits = {}
    #animal_fpath = os.path.join(rootdir, animalid, 'FOVS.pkl')
    #print "Loading FOV info for animal: %s" % animal_fpath
    #with open(animal_fpath, 'rb') as f:
    #    A = pkl.load(f)
   
    for session, fov in A.sessions.items():
        nrois_total = len(fov.RFs)  
        if nrois_total == 0:
            print "No rois!"
            rf_fits[session] = 0
            continue
       
        retino_good_rois = [roi for roi in range(nrois_total) if all([fov.RFs[roi].conditions[ci].fit_results['r2'] >= fit_thr for ci in range(2)])]
        print "%s: %i out of %i good RF fits." % (session, len(retino_good_rois), nrois_total)
        #rf_fits[session] = good
    
        retino_az_cond = [c for c in range(2) if fov.RFs[roi].conditions[c].name == 'right'][0]
        retino_el_cond = [c for c in range(2) if fov.RFs[roi].conditions[c].name == 'top'][0]
        
        retino_widths = [(fov.RFs[roi].conditions[retino_az_cond].fit_results['sigma'], fov.RFs[roi].conditions[retino_el_cond].fit_results['sigma']) for roi in retino_good_rois]
        retino_rdict = dict((roi, {'width_x': fov.RFs[roi].conditions[retino_az_cond].fit_results['sigma'], 
                           'width_y': fov.RFs[roi].conditions[retino_el_cond].fit_results['sigma'], 
                           'r2_x': fov.RFs[roi].conditions[retino_az_cond].fit_results['r2'], 
                           'r2_y': fov.RFs[roi].conditions[retino_el_cond].fit_results['r2']}) for roi in retino_good_rois)
        rf_fits[session] = retino_rdict

    # Save to animal dir:
    with open(os.path.join(rootdir, A.animalid, 'rf_fits.pkl'), 'wb') as f:
        pkl.dump(rf_fits, f, protocol=pkl.HIGHEST_PROTOCOL)
        

    return rf_fits 
  
def hist_rf_widths(rf_fits, visual_area='visual area', ax=None, color='g', kde=True):
    rf_widths = [np.mean( [res['width_x'], res['width_y']] ) for session, rfdicts in rf_fits.items() for roi, res in rfdicts.items() ] 
    if ax is None:
        fig, ax = pl.subplots(1)

    if kde:
        kde_kws = {"color": color, "lw": 3, "label": "KDE", "alpha": 0.5}
        hist_kws = {"histtype": "step", "linewidth": 1, "alpha": 0.2, "color": color}
    else:
        kde_kws = {}
        hist_kws = {"histtype": "step", "linewidth": 2, "alpha": 0.5, "color": color}
    
    sns.distplot(rf_widths, label=visual_area, ax=ax, kde=kde,
              rug=True, rug_kws={"color": color},
              kde_kws=kde_kws,
              hist_kws=hist_kws)
 
    #return fig
#%%
    
#
#fit_thr = 0.5
#nrois_total = len(S.retinotopy['data'])
#retino_good_rois = [roi for roi in range(nrois_total) if all([S.retinotopy['data'][roi].conditions[ci].fit_results['r2'] >= fit_thr for ci in range(2)])]
#print "--- %i out of %i ROIs fit for retinotopy RF estimates." % (len(retino_good_rois), nrois_total)
#
#
#retino_az_cond = [c for c in range(2) if S.retinotopy['data'][roi].conditions[c].name == 'right'][0]
#retino_el_cond = [c for c in range(2) if S.retinotopy['data'][roi].conditions[c].name == 'top'][0]
#
#retino_widths = [(S.retinotopy['data'][roi].conditions[retino_az_cond].fit_results['sigma'], S.retinotopy['data'][roi].conditions[retino_el_cond].fit_results['sigma']) 
#                        for roi in retino_good_rois]
#
#
#retino_rdict = dict((roi, {'width_x': S.retinotopy['data'][roi].conditions[retino_az_cond].fit_results['sigma'], 
#                           'width_y': S.retinotopy['data'][roi].conditions[retino_el_cond].fit_results['sigma'], 
#                           'r2_x': S.retinotopy['data'][roi].conditions[retino_az_cond].fit_results['r2'], 
#                           'r2_y': S.retinotopy['data'][roi].conditions[retino_el_cond].fit_results['r2']}) for roi in retino_good_rois)
#retino_df = pd.concat([pd.DataFrame(data=rdict.values(), columns=[roi], index=rdict.keys()) for roi, rdict in retino_rdict.items()], axis=1)
#
#
#





    
