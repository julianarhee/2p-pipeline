#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 13:54:28 2019

@author: julianarhee
"""

import os
import glob
import json 

import numpy as np
import cPickle as pkl

import seaborn as sns
import pylab as pl
import pandas as pd

import itertools

from pipeline.python.utils import natural_keys
from pipeline.python.classifications import utils as util

rootdir = '/n/coxfs01/2p-data'

animalids = ['JC076', 'JC078', 'JC080', 'JC083', 'JC084', 'JC085', 'JC090', 'JC091', 'JC097', 'JC099']

fov_type = 'zoom2p0x'

#%%
class MetaData():
    def __init__(self, animalid, rootdir='/n/coxfs01/2p-data'):
        self.animalid = animalid
        self.anesthetized_session_list = []
        self.sessions = {}
    
    def get_sessions(self, fov_type='zoom2p0x', session_list = [],
                     create_new=False, rootdir='/n/coxfs01/2p-data'):

        # Check if anesthetized info / visual area info stored in metafile:
        create_meta = False
        meta_info_file = os.path.join(rootdir, animalid, 'sessionmeta.json')
        if os.path.exists(meta_info_file):
            try:
                with open(meta_info_file, 'r') as f:
                    meta_info = json.load(f)
            except Exception as e:
                print("...creating new meta file")
                create_meta = True
        else:
            create_new = True
            create_meta = True
            
        if create_meta:
            meta_info = {}
        
        # Get all session for current animal:
        if len(session_list) == 0:
            session_paths =  sorted(glob.glob(os.path.join(rootdir, animalid,  '*', 'FOV*_%s' % fov_type)), key=natural_keys)
        else:
            session_paths = sorted([glob.glob(os.path.join(rootdir, animalid,  '%s' % s.split('_')[0], 
                                                           '%s_%s' % (s.split('_')[1], fov_type)))[0]\
                                   for s in session_list], key=natural_keys)
        print("Found %i acquisitions." % len(session_paths))
        for si, session_path in enumerate(session_paths):
            session_name = os.path.split(session_path.split('/FOV')[0])[-1]
            fov_name = os.path.split(session_path)[-1]
            print("[%s]: %s - %s" % (animalid, session_name, fov_name))
            skey = '%s_%s' % (session_name, fov_name.split('_')[0])

            # Load session data, if exists:
            output_dir = os.path.join(rootdir, animalid, session_name, fov_name, 'summaries')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            session_outfile = os.path.join(output_dir, 'sessiondata.pkl')                
            
            if not create_new:
                try:
                    assert os.path.exists(session_outfile), "... session object does not exist, creating new."
                    print("... loading session object...") #% (animalid, session_name))
                    with open(session_outfile, 'rb') as f:
                        S = pkl.load(f)
                        assert 'visual_area' in dir(S), "... No visual area found, creating new."
                except Exception as e:
                    print e
                    create_new = True
            
            # Update meta info if this is a new session:
            if skey not in meta_info.keys():
                user_input = raw_input('Was this session anesthetized? [Y/n]')
                if user_input == 'Y':
                    #self.anesthetized_session_list.append(session_name)
                    state = 'anesthetized'
                else:
                    state = 'awake'
                visual_area = raw_input('Enter visual area recorded: ')
                meta_info.update({skey: {'state': state,
                                        'visual_area': visual_area}})
            else:
                state = meta_info[skey]['state']
                visual_area = meta_info[skey]['visual_area']
                

            if create_new:
                print("Creating new session object...") #% (animalid, session_name))
                S = util.Session(animalid, session_name, fov_name, 
                                 visual_area=visual_area, state=state,
                                 rootdir=rootdir)
                #S.load_data(traceid=traceid, trace_type='corrected')
                # Save session data object
                with open(session_outfile, 'wb') as f:
                    pkl.dump(S, f, protocol=pkl.HIGHEST_PROTOCOL)
                
            self.sessions[skey] = S
            
            if state == 'anesthetized' and skey not in self.anesthetized_session_list:
                self.anesthetized_session_list.append(skey)
                
            with open(meta_info_file, 'w') as f:
                json.dump(meta_info, f, sort_keys=True, indent=4)
        return self.sessions
    
            
    def load_experiments(self, experiment=None, select_subset=False,
                         state=None, visual_area=None,
                         traceid='traces001', trace_type='corrected', load_raw=False,
                         responsive_thr=0.01, responsive_test='ROC',
                         receptive_field_fit='zscore0.00_no_trim',
                         update=True, get_grouped=True,
                         rootdir='/n/coxfs01/2p-data'):
        
        # Make sure output dir exists:
        if not os.path.exists(os.path.join(rootdir, 'summary_stats', 'animals')):
            os.makedirs(os.path.join(rootdir, 'summary_stats', 'animals'))
            
        #session_stats = {}
        assert len(self.sessions.keys()) > 0, "** no sessions found! **"
        for skey, sobj in self.sessions.items():
            
            if select_subset:
                # Select subset of sessions, based on visual area or state:
                if state == 'awake' and skey in self.anesthetized_session_list:
                    continue
                if state == 'anesthetized' and skey not in self.anesthetized_session_list:
                    continue
                if visual_area is not None and sobj.visual_area != visual_area:
                    continue
            
            # Do correction on experiment names for sessions before 20190511
            if experiment == 'rfs' and int(sobj.session) < 20190511:
                experiment_name = 'gratings'
            elif experiment == 'gratings' and int(sobj.session) < 20190511: 
                continue
            elif experiment == 'blobs' and sobj.animalid == 'JC078' and sobj.session == '20190426':
                continue
            else:
                experiment_name = experiment
            
            # Either load traces/labels or summary stats:
            if load_raw:
                if sobj.data.traces is None:
                    expdict = sobj.load_data(experiment=experiment_name, traceid=traceid, trace_type=trace_type)
                else:
                    expdict = sobj.experiments[experiment_name]
            else:
                expdict = sobj.get_grouped_stats(experiment_type=experiment_name,
                                  traceid=traceid, trace_type=trace_type, 
                                  responsive_test=responsive_test, responsive_thr=responsive_thr,
                                  receptive_field_fit=receptive_field_fit,
                                  update=update, get_grouped=get_grouped,
                                  rootdir=rootdir)
        
            #if expdict is not None:
            #    session_stats[skey] = expdict
            if expdict is not None:
                outfile = os.path.join(rootdir, 'summary_stats', 'animals', '%s_%s.pkl' % (self.animalid, skey))
                with open(outfile, 'wb') as f:
                    pkl.dump(expdict, f, protocol=pkl.HIGHEST_PROTOCOL)
                print("... saved session states: %s" % outfile)
            if update:
                # Load session data, if exists:
                sobj.save_session(rootdir=rootdir)
                print("... updated session object: %s" % skey)
                    
        return #session_stats
            
    
    def save_session_stats(self, outfile=None, rootdir='/n/coxfs01/2p-data'):
        #if outfile is None:
        #    outfile = os.path.join(rootdir, '%s.pkl' % self.animalid)
        
        #with open(outfile, 'wb') as f:
        #    pkl.dump(self, f, protocol=pkl.HIGHEST_PROTOCOL)
            
        for skey, sobj in self.sessions.items():
            curr_outfile = os.path.join(rootdir, '%s_%s.pkl' % (self.animalid, skey))
            with open(curr_outfile, 'wb') as f:
                pkl.dump(sobj, f, protocol=pkl.HIGHEST_PROTOCOL)
            
        print("--- saved animal data to:\n%s" % outfile)
                     
#%%

import h5py
from datetime import datetime

def df_to_sarray(df):
    """
    Convert a pandas DataFrame object to a numpy structured array.
    Also, for every column of a str type, convert it into 
    a 'bytes' str literal of length = max(len(col)).

    :param df: the data frame to convert
    :return: a numpy structured array representation of df
    """

    def make_col_type(col_type, col):
        try:
            if 'numpy.object_' in str(col_type.type):
                maxlens = col.dropna().str.len()
                if maxlens.any():
                    maxlen = maxlens.max().astype(int) 
                    col_type = ('S%s' % maxlen, 1)
                else:
                    col_type = 'f2'
            return col.name, col_type
        except:
            print(col.name, col_type, col_type.type, type(col))
            raise

    v = df.values            
    types = df.dtypes
    numpy_struct_types = [make_col_type(types[col], df.loc[:, col]) for col in df.columns]
    dtype = np.dtype(numpy_struct_types)
    z = np.zeros(v.shape[0], dtype)
    for (i, k) in enumerate(z.dtype.names):
        #print k
        # This is in case you have problems with the encoding, remove the if branch if not
        try:
            if dtype[i].str.startswith('|S'):
                z[k] = df[k].str.encode('latin').astype('S')
            else:
                z[k] = v[:, i]
        except:
            print(k, v[:, i])
            raise

    return z, dtype



def save_df_to_h5(df, dataframe_outfile):    
    sa, saType = df_to_sarray(df)
    f = h5py.File(dataframe_outfile, 'a')
    if len(f.keys()) > 0:
        for k in f.keys(): del f[k];
    f.create_dataset('df', data=sa, dtype=saType)
    f.close()
    
    
def load_exp_from_h5(curr_outfile):
    edicts = {}
    f = h5py.File(curr_outfile, 'r')    
    try:
        for skey, edict in f.items():
            edicts[skey] = {}
            for stimulus, sarray in edict.items():
                edata = util.Struct()
                if not isinstance(f[skey][stimulus]['responses_df'], h5py.Dataset):
                    edata.gdf = {}
                    for k in f[skey][stimulus]['responses_df'].keys():
                        if not isinstance(f[skey][stimulus]['responses_df'][k], h5py.Dataset):
                            edata.gdf[k] = {}
                            for kk in f[skey][stimulus]['responses_df'][k].keys():
                                sa = f[skey][stimulus]['responses_df'][k][kk][:]
                                edata.gdf[k][kk] = pd.DataFrame(sa)
                        else:
                            sa = f[skey][stimulus]['responses_df'][k][:]
                            edata.gdf[k] = pd.DataFrame(sa)
                else:
                    sa1 = f[skey][stimulus]['responses_df'][:]
                    edata.gdf = pd.DataFrame(sa1)
                sa2 = f[skey][stimulus]['stimuli_df'][:]
                edata.sdf = pd.DataFrame(sa2)
                edata.rois = f[skey][stimulus]['rois'][:]
                edata.nrois = f[skey][stimulus].attrs['nrois']
                edata.experiment_id = f[skey][stimulus].attrs['experiment_id']
                edicts[skey][stimulus] = edata
    except Exception as e:
        print e
    finally:
        f.close()

    return edicts

#%%
def create_animal_objects(animalid, trace_type='corrected', traceid='traces001', state='awake',
                          basedir='/n/coxfs01/2p-data/summary_stats'):
    
    
    A = MetaData(animalid)
    A.get_sessions(fov_type=fov_type, create_new=True)
    
    
    all_sessions = A.load_experiments(experiment=None, select_subset=False,                                      
                                       load_raw=False, update=True, get_grouped=False,
                                       trace_type=trace_type, traceid=traceid)

    animal_outfile = os.path.join(basedir, 'animals', '%s_%s_%s.pkl' % (animalid, traceid, trace_type))
    A.save_animal(outfile=animal_outfile)
    
    return

def get_animal_list(visual_area='AREA', state='STATE', experiment=None,
                    traceid='traces001', trace_type='dff',
                    basedir='/n/coxfs01/2p-data/summary_stats'):
    
    animal_dict = {}
    animal_meta = glob.glob(os.path.join(rootdir, 'JC*', 'sessionmeta.json'))
    for ameta in animal_meta:
        animalid = os.path.split(os.path.split(ameta)[0])[-1]
        with open(ameta, 'r') as f:
            meta = json.load(f)
        included = [k for k, v in meta.items() if v['state']==state and v['visual_area'] == visual_area]
        if len(included) > 0:
            animal_dict[animalid] = included
    
    return animal_dict
    

def create_visual_area_datasets(visual_area='AREA', state='STATE', experiment=None,
                    traceid='traces001', trace_type='dff', fov_type='zoom2p0x',
                    receptive_field_fit='zscore0.00_no_trim', 
                    responsive_test='ROC', responsive_thr=0.01,
                    basedir='/n/coxfs01/2p-data/summary_stats'):
    

    animal_dict = get_animal_list(visual_area=visual_area, state=state,
                                  traceid=traceid, trace_type=trace_type,
                                  basedir=basedir)

    for animalid, session_list in animal_dict.items():
        print("[%s]: Getting data: %s" % (visual_area, animalid))
        
        A = MetaData(animalid)
        A.get_sessions(fov_type=fov_type, create_new=False, session_list=session_list)
        
        area_outdir = os.path.join(basedir, visual_area)
        if not os.path.exists(area_outdir): 
            os.makedirs(area_outdir)
        
        sstats = {}
        for skey, sobj in A.sessions.items():
            sstats[skey] = sobj.get_grouped_stats(experiment_type=None,
                                                      traceid=traceid, trace_type=trace_type, 
                                                      responsive_test=responsive_test, responsive_thr=responsive_thr,
                                                      receptive_field_fit=receptive_field_fit,
                                                      update=False, get_grouped=get_grouped,
                                                      rootdir=rootdir)

        curr_outfile = os.path.join(area_outdir, '%s_%s_%s_stats.pkl' % (animalid, visual_area, state))   
        with open(curr_outfile, 'wb') as f:
            pkl.dump(sstats, f, protocol=pkl.HHIGHEST_PROTOCOL)

        stats_info = {'receptive_field_fit': receptive_field_fit,
                      'responsive_thr': responsive_thr,
                      'responsive_test': responsive_test,
                      'traceid': traceid,
                      'trace_type': trace_type,
                      'fov_type': fov_type}
        
        curr_statsfile = '%s.json' % os.path.splitext(curr_outfile)[0]
        with open(curr_statsfile, 'w') as f:
            json.dump(stats_info, f, sort_keys=True, indent=4)
                     
    
    return allstats


    #%%
def save_stats_all_sessions(curr_sessions, outfile='/tmp/animal.h5'):
    # Open hdf5 file: 
    f = h5py.File(outfile, 'a')
    try:
        for si, (skey, expdict) in enumerate(curr_sessions.items()):
            #print expdict
            sgrp = f.create_group(skey) if skey not in f.keys() else f[skey]
            if len(sgrp.keys()) > 0:
                for k in sgrp.keys(): del sgrp[k];
                
            for stimulus, edata in expdict.items():
                #print stimulus
                try:
                    egrp = sgrp.create_group(stimulus)
                    # Save the structured array
                    if isinstance(edata.gdf, dict):
                        for k, v in edata.gdf.items():
                            if isinstance(v, dict):
                                for kk, vv in v.items():
                                    #print kk
                                    v[str(kk)] = v.pop(kk)
                                    v[str(kk)].columns = [str(i) for i in vv.columns.tolist()]
                                    sa, saType = df_to_sarray(v[str(kk)])
                                    egrp.create_dataset('responses_df/%s/%s' % (k, kk), data=sa, dtype=saType)
                            else:
                                for col in v.columns.tolist():
                                    v[str(col)] = v.pop(col)
                                sa, saType = df_to_sarray(v)
                                egrp.create_dataset('responses_df/%s' % k, data=sa, dtype=saType)
                    else:
                        sa, saType = df_to_sarray(edata.gdf)
                        egrp.create_dataset('responses_df', data=sa, dtype=saType)

                    if not(isinstance(edata.sdf, pd.DataFrame)):
                        edata.sdf = pd.DataFrame(edata.sdf)
                    for col in edata.sdf.columns.tolist():
                        edata.sdf[str(col)] = edata.sdf.pop(col)
                        if col == 'color' and any(edata.sdf[col] == ''):
                            edata.sdf.loc[edata.sdf[col]=='', 'color'] = None
                    sa, saType = df_to_sarray(edata.sdf)
                    egrp.create_dataset('stimuli_df', data=sa, dtype=saType)
                    egrp.create_dataset('rois', data=np.array(edata.rois), dtype=np.array(edata.rois).dtype)
                    egrp.attrs['experiment_id'] = edata.experiment_id
                    egrp.attrs['nrois'] = edata.nrois
                    egrp.attrs['date_created'] = datetime.now().strftime("%Y%d%m %H:%M:%S")
#                    
#                    if 'fits' in dir(edata):
#                        edata.fits.columns = [str(c) for cin edata.fits.columns]
#                        sa, saType = df_to_sarray(edata.fits)
#                        egrp.create_dataset('fits', data=sa, dtypes=saType)
#                    
                    # Retrieve it and check it is ok when you transform it into a pandas DataFrame
                except Exception as e:
                    print e
                    print stimulus
    except Exception as e:
        print e
    finally:
        f.close()
        
        
        

#%%

# Create output dir:
summary_outdir = os.path.join(rootdir, 'summary_stats')
if not os.path.exists(summary_outdir):
    os.makedirs(summary_outdir)
    
if not os.path.exists(os.path.join(summary_outdir, 'animals')):
    os.makedirs(os.path.join(summary_outdir, 'animals'))
    

#%%
palette = itertools.cycle(sns.color_palette())
traceid = 'traces001'
trace_type = 'dff' 

get_grouped = False   # Set to False to save dataframe as hdf5
create_new = True
#
#animalid = 'JC084'
#A = create_animal_objects(animalid, trace_type=trace_type, traceid=traceid,
#                          basedir=summary_outdir) #='/n/coxfs01/2p-data')
#                    
#%%
for animalid in animalids:
    
    create_animal_objects(animalid,
                          trace_type=trace_type, traceid=traceid,
                          basedir=summary_outdir) #='/n/coxfs01/2p-data')
                

#%%

stat_name = 'peakdff' 




    
#%%

stat_name = 'rfs'
dfs = []
for visual_area in ['V1', 'Lm', 'Li']:
    datafiles = glob.glob(os.path.join(summary_outdir, visual_area, '*.h5'))
    print("[%s] Found %i datafiles" % (visual_area, len(datafiles)))
    
    for dfile in datafiles:
        print dfile
        
        curr_sessions = load_exp_from_h5(dfile)
        
        for skey, expdict in curr_sessions.items():
            print skey, expdict.keys()
            if stat_name == 'rfs':
                ekeys = [k for k, v in expdict.items() if 'rfs' in k]
                if len(ekeys) == 0:
                    continue
                elif len(ekeys) == 1:
                    edata = expdict[ekeys[0]]
                else:
                    for ei, ekey in enumerate(ekeys):
                        print ei, ekey
                    sel = input("select IDX of rfs to use: ")
                    edata = expdict[ekeys[int(sel)]]
            
        if get_grouped is False:
            groupdf = edata.gdf.groupby(edata.gdf.index)
        else:
            groupdf = edata.gdf
            
        if stat_name == 'rfs':
            fitdf = edata.fits
            stat_values  = edata.fits[['sigma_x', 'sigma_y']].mean(axis=1)
        elif stat_name == 'peakdff' and trace_type == 'dff':
            stat_values = groupdf.max()['meanstim'].values
        elif stat_name == 'zscore' and trace_type == 'corrected':
            stat_values = groupdf.max()['zscore'].values
            
        dfs.append(pd.DataFrame({'%s' % stat_name: stat_values,
                                  'session': [skey for _ in range(len(stat_values))],
                                  'animalid': [animalid for _ in range(len(stat_values))],
                                  'visual_area': [visual_area for _ in range(len(stat_values))],
                                  'stimulus': [stimulus for _ in range(len(stat_values))],
                                  'visual_cells': [len(stat_values) for _ in range(len(stat_values))],
                                  'total_cells': [edata.nrois for _ in range(len(stat_values))]}) )


            
#%%
df = pd.concat(dfs, axis=0)

dataframe_fname = 'df_%s_%s_%s' % (traceid, trace_type, stat_name)

dataframe_outfile = os.path.join(summary_outdir, '%s.pkl' % dataframe_fname)
with open(dataframe_outfile, 'wb') as f:
    pkl.dump(df, f, protocol=pkl.HHIGHEST_PROTOCOL)

dataframe_outfile = os.path.join(summary_outdir, '%s.h5' % dataframe_fname)
save_df_to_h5(df, dataframe_outfile)




            #%%
            
        #%%
    
df = pd.concat(dfs, axis=0)

fig, ax = pl.subplots()

#%%


currdf = df[df['stimulus']=='blobs']

area_colors = sns.color_palette('Set1', len(df['visual_area'].unique()))
area_colordict = dict((s, color) for s, color in zip(df['visual_area'].unique(), area_colors))

animal_colors = sns.color_palette('cubehelix', len(animalids))
animal_colordict = dict((animalid, color) for animalid, color in zip(animalids, animal_colors))

session_colors = sns.color_palette('hls', len(df['session'].unique()))
session_colordict = dict((s, color) for s, color in zip(df['session'].unique(), session_colors))

    
#%%
from matplotlib.lines import Line2D

area_markerdict = {'V1': 'o', 
                   'Lm': '^',
                   'Li': 's'}

animal_colors = sns.color_palette('colorblind', len(animalids))
animal_colordict = dict((animalid, color) for animalid, color in zip(animalids, animal_colors))

session_list = sorted(df['session'].unique(), key=natural_keys)

fig, ax = pl.subplots(figsize=(10,8))

for stim in sorted( df['stimulus'].unique() ):
    currdf = df[df['stimulus']==stim]
    
    curr_sessions = sorted(currdf['session'].unique(), key=natural_keys)
    session_ixs = [session_list.index(s) for s in curr_sessions]
    cell_counts = [currdf[currdf['session']==s]['cells'].unique()[0] for s in curr_sessions]
    
    color_by_animal = [animal_colordict[currdf[currdf['session']==s]['animalid'].unique()[0]] for s in curr_sessions]
    color_by_area = [area_colordict[currdf[currdf['session']==s]['visual_area'].unique()[0]] for s in curr_sessions]
    marker_by_area = [area_markerdict[currdf[currdf['session']==s]['visual_area'].unique()[0]] for s in curr_sessions]
    
        
    for xp, yp, c, m in zip( session_ixs, cell_counts, color_by_animal, marker_by_area ):
        if stim == 'gratings':
            ax.scatter([xp], [yp], edgecolor=c, facecolor='none', marker=m, s=100, lw=2)
        elif stim == 'blobs':
            ax.scatter([xp], [yp], edgecolor=c, facecolor=c, marker=m, s=100)

pl.subplots_adjust(left=0.15, right=0.8)

ax.set_xlabel('FOV')
ax.set_ylabel('# cells')
ax.set_xticks(np.arange(0, len(df['session'].unique())))
ax.set_xticklabels('')

#ax[1].scatter(range(len(session_list)), cell_counts, c=color_by_area)
color_lines = [Line2D([0], [0], color=c, lw=4) for k, c in sorted(animal_colordict.items(), key=lambda x: x[0])]
color_labels = [k for k, v in sorted(animal_colordict.items(), key=lambda x: x[0])]
color_legend = pl.legend(color_lines, color_labels, loc='lower left',  fontsize=8,
                         bbox_to_anchor= (1., 0.4), ncol=1, frameon=False)

shape_lines = [Line2D([0], [0], marker=area_markerdict[visual_area], color='w', label=visual_area,
                          markerfacecolor='k', markersize=10) for visual_area in sorted(area_colordict.keys())]
shape_labels = sorted(area_colordict.keys())
shape_legend = pl.legend(shape_lines, shape_labels, loc='lower-left', fontsize=8,
                         bbox_to_anchor= (1.1, 0.2), frameon=False)


fill_lines = [Line2D([0], [0], marker='o', lw=0, color='k', label='blobs',
                          markerfacecolor='k', markersize=10),
              Line2D([0], [0], marker='o', lw=0, color='k', label='gratings',
                          markerfacecolor='none', markersize=10)]
    
fill_labels = ['blobs', 'gratings']
fill_legend = pl.legend(fill_lines, fill_labels, loc='lower-left', fontsize=8,
                         bbox_to_anchor= (1.15, 0.35), frameon=False)


pl.gca().add_artist(color_legend)
pl.gca().add_artist(shape_legend)




#%%
sns.violinplot(x='session', y='%s' % stat_name, ax=ax, data=df)

    
g = sns.catplot(x="stimulus", y="%s" % stat_name, hue="animalid", col="visual_area",
                data=df, kind="box",
                height=4, aspect=.7)
                

g = sns.boxplot(x="animalid", y="%s" % stat_name, hue="visual_area",
                data=df)
                
    
g = sns.catplot(x="visual_area", y="%s" % stat_name, hue="stimulus",# col="visual_area",
                data=df, kind="box")
                
#%%

fig, ax = pl.subplots()
g = sns.catplot(x="peakdff", y='visual_area', hue="stimulus", #row="visual_area",
                data=df, kind="box")

#%%

sns.set()
g = sns.PairGrid(df,hue='visual_area')
g.map_diag(pl.hist)
g.map_offdiag(pl.scatter)

for col in ['visual_area', 'stimulus', 'session', 'animalid']:
    df[col] = df[col].astype('category')

categorical = df.dtypes[df.dtypes != 'int64'][df.dtypes != 'float64'].index.tolist()

numerical =  df.dtypes[((df.dtypes == 'int64') | (df.dtypes == 'float64'))].index.tolist()


fig, ax = pl.subplots(2, 4, figsize=(20, 10))
for variable, subplot in zip(categorical, ax.flatten()):
    sns.countplot(df[variable], ax=subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(90)
        
        
        
fig, ax = pl.subplots(3, 3, figsize=(15, 10))
for var, subplot in zip(categorical, ax.flatten()):
    sns.boxplot(x=var, y='cells', data=df, ax=subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(90)
        
        
sorted_nb = df.groupby(['session'])['cells'].median().sort_values()
sns.boxplot(x=df['session'], y=df['cells'], order=list(sorted_nb.index))



cond_plot = sns.FacetGrid(data=df, col='session', hue='stimulus', col_wrap=4)
cond_plot.map(pl.hist, 'peakdff', histtype='step') #, 'peakdff');
pl.legend()

g = sns.FacetGrid(data=df, col='visual_area',  col_wrap=4)
g = g.map(sns.stripplot, 'session', 'peakdff', 'stimulus', edgecolor="w", 
          linewidth=0.1, dodge=True, jitter=True, size=2)

pl.legend()


g = sns.FacetGrid(data=df, col='visual_area', hue='animalid', col_wrap=4)
g = g.map(sns.stripplot, 'stimulus', 'cells', edgecolor="w", 
          linewidth=0.1, dodge=False, jitter=True, size=5)
g.add_legend()