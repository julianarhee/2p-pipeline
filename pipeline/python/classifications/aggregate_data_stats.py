import os
import glob
import shutil
import json
import re
import sys
import optparse
import itertools
import copy
import traceback

import statsmodels as sm
import scipy.stats as spstats
import pandas as pd
import numpy as np
import pylab as pl
import seaborn as sns
import cPickle as pkl

from pipeline.python.classifications import experiment_classes as util
from pipeline.python.utils import label_figure, natural_keys, reformat_morph_values, add_meta_to_df, isnumber, split_datakey, split_datakey_str

# ===============================================================
# Dataset selection
# ===============================================================
def get_sorted_fovs(filter_by='drop_repeats', excluded_sessions=[]):
    '''
    For each animal, dict of visual areas and list of tuples (each tuple is roughly similar fov)
    Use this to filter out repeated FOVs.
    '''
    fov_keys = {'JC076': {'V1': [('20190420', '20190501')],
                          'Lm': [('20190423_fov1')],
                          'Li': [('20190422', '20190502')]},

                'JC078': {'Lm': [('20190426', '20190504', '20190509'),
                                 ('20190430', '20190513')]},

                'JC080': {'Lm': [('20190506', '20190603'), 
                                 ('20190602_fov2')],
                          'Li': [('20190602_fov1')]},

                'JC083': {'V1': [('20190507', '20190510', '20190511')],
                          'Lm': [('20190508', '20190512', '20190517')]},

                'JC084': {'V1': [('20190522')],
                          'Lm': [('20190525')]},

                'JC085': {'V1': [('20190622')]},

                'JC089': {'Li': [('20190522')]},

                'JC090': {'Li': [('20190605')]},

                'JC091': {'Lm': [('20190627')],
                          'Li': [('20190602', '20190607'),
                                 ('20190606', '20190614'),
                                 ('20191007', '20191008')]},

                'JC092': {'Li': [('20190527_fov2'),
                                 ('20190527_fov3'),
                                 ('20190528')]},

                'JC097': {'V1': [('20190613'),
                                 ('20190615_fov1', '20190617'),
                                 ('20190615_fov2', '20190616')],
                          'Lm': [('20190615_fov3'),
                                 ('20190618')]},

                'JC099': {'Li': [('20190609', '20190612'),
                                 ('20190617')]},

                'JC110': {'V1': [('20191004_fov2', '20191006')],
                          'Lm': [('20191004_fov3'), ('20191004_fov4')]},

                'JC111': {'Li': [('20191003')]},

                'JC113': {'Lm': [('20191012_fov3')],
                          'Li': [('20191012_fov1'), ('20191012_fov2', '20191017', '20191018')]},

                'JC117': {'V1': [('20191111_fov1')],
                          'Lm': [('20191104_fov2'), ('20191111_fov2')],
                          'Li': [('20191104_fov1', '20191105')]},

                'JC120': {'V1': [('20191106_fov3')],
                          'Lm': [('20191106_fov4')],
                          'Li': [('20191106_fov1', '20191111')]},
                #}

                'JC061': {'Lm': [('20190306_fov2'), ('20190306_fov3')]},

                'JC067': {'Li': [('20190319'), ('20190320')]},

                'JC070': {'Li': [('20190314_fov1', '20190315'), # 20190315 better, more reps
                                  ('20190315_fov2'),
                                  ('20190316_fov1'), 
                                  ('20190321_fov1', '20190321_fov2')],
                          'Lm': [('20190314_fov2', '20190315_fov3')]},
                'JC073': {'Lm': [('20190322', '20190327')],
                          'Li': [('20190322', '20190327')]} #920190322 better
                }
    

    return fov_keys

def all_datasets_by_area(visual_areas=[]):
    if len(visual_areas)==0:
        visual_areas = ['V1', 'Lm', 'Li']
        
    fkeys = get_sorted_fovs()
    
    ddict = dict((v, []) for v in visual_areas)
    for animalid, sinfo in fkeys.items():
        for visual_area, slist in sinfo.items():
            if visual_area not in ddict.keys():
                continue
            for sublist in slist:
                if isinstance(sublist, tuple):
                    sessions_ = ['%s_%s_%s' % (s.split('_')[0], animalid, s.split('_')[-1]) \
                                 if len(s.split('_'))>1 else '%s_%s_fov1' % (s, animalid) for s in sublist]
                else:
                    sessions_ = ['%s_%s_%s' % (sublist.split('_')[0], animalid, sublist.split('_')[-1]) \
                             if len(sublist.split('_'))>1 else '%s_%s_fov1' % (sublist, animalid)]

                ddict[visual_area].extend(sessions_)

    return ddict


def get_metadata(traceid='traces001', filter_by='most_cells', stimulus=None, stimulus_only=False,
                 fov_type='zoom2p0x', state='awake', excluded_sessions=[],
                 aggregate_dir='/n/coxfs01/julianarhee/aggregate-visual-areas'):
       
    # Get all datasets
    sdata = get_aggregate_info(traceid=traceid, fov_type=fov_type, state=state)
    if stimulus == 'gratings':
        included_sessions = get_gratings_datasets(filter_by=filter_by, 
                                                 excluded_sessions=excluded_sessions,
                                                 as_dict=False)
        
    elif stimulus == 'rfs':
        included_sessions = get_rf_datasets(filter_by=filter_by, 
                                            excluded_sessions=excluded_sessions,
                                            as_dict=False)
    elif stimulus == 'blobs':
        included_sessions = get_blob_datasets(filter_by=filter_by, 
                                            excluded_sessions=excluded_sessions,
                                            as_dict=False)
 
    else:
        print("Unknow <%s>. Select from: gratings, rfs, blobs, or all" % str(stimulus))
        return None
    
    print("Selecting %i dsets" % len(included_sessions))
    dsets = pd.concat([g for v, g in sdata.groupby(['visual_area', 'animalid', 'session', 'fovnum']) \
                       if '%s_%s' % (v[2], v[1]) in included_sessions \
                       or '%s_%s_fov%i' % (v[2], v[1], v[3]) in included_sessions])
        
    if stimulus_only:
        if stimulus == 'rfs':
            return dsets[dsets['experiment'].isin(['rfs', 'rfs10'])]
        else:
            return dsets[dsets['experiment'].isin([stimulus])]
    else:        
        return dsets


def include_dsets_with(edata, experiment='blobs', also_include='rfs'):
    if also_include=='rfs':
        dsets = pd.concat([g for k, g in edata.groupby(['animalid', 'session', 'fov']) if 
                    (experiment in g['experiment'].values 
                    and ('rfs' in g['experiment'].values or 'rfs10' in g['experiment'].values)) ])
    else:
        dsets = pd.concat([g for k, g in edata.groupby(['animalid', 'session', 'fov']) if 
                    (experiment in g['experiment'].values) and (also_include in g['experiment'].values)]) 

    return dsets
 
def get_blob_datasets(filter_by='most_fits', excluded_sessions=[], as_dict=True, 
                        has_gratings=False, has_rfs=False, response_type='dff',
                    responsive_test='nstds', responsive_thr=10.0):
    '''
    filter_by:  None to return all, most_fits to retun ones with most # responsive.
    Others are untested
    '''
    from pipeline.python.retinotopy import segment_retinotopy as seg

    included_sessions=[]
    #excluded_sessions.extend(always_exclude)
    
    # Get meta info 
    sdata = get_aggregate_info() 
    if has_gratings:
        sdata = include_dsets_with(sdata, experiment='blobs', also_include='gratings')

    # Get blob metadata only - and only if have RFs
    if has_rfs:
        edata = include_dsets_with(sdata, experiment='blobs', also_include='rfs')
    else:
        edata = sdata[sdata['experiment']=='blobs']

    if filter_by == 'most_fits':
        dsets = edata[edata['experiment']=='blobs'].copy()
        assigned_cells = seg.get_cells_by_area(dsets)
        best_dfs = get_dsets_with_most_blobs(dsets, assigned_cells, response_type=response_type,
                        responsive_test=responsive_test, responsive_thr=responsive_thr)
        if as_dict:
            included_sessions = make_metadict_from_df(best_dfs)
        else:
            included_sessions = best_dfs

    elif filter_by is None:
        if as_dict:
            included_sessions = make_metadict_from_df(dsets)
        else:
            included_sessions = edata
    else:
        included_sessions = get_blob_datasets_drop(filter_by=filter_by, has_gratings=has_gratings,
                                    excluded_sessions=excluded_sessions, as_dict=as_dict)

    return included_sessions 


def get_blob_datasets_drop(filter_by='first', has_gratings=False,
                        excluded_sessions=[], as_dict=True):

    included_sessions = []

    # Blobs runs w/ incorrect stuff
    always_exclude = ['20190426_JC078', # backlight test, but good for A/B
                      '20191008_JC091'] # rf test
    excluded_sessions.extend(always_exclude)

    if filter_by is None:
        #ddict = all_datasets_by_area()
        sdata = get_aggregate_info()
        bd = sdata[sdata['experiment']=='blobs'].copy()
        v1_include = bd[bd['visual_area']=='V1']['datakey'].unique()
        lm_include = bd[bd['visual_area']=='Lm']['datakey'].unique()
        li_include = bd[bd['visual_area']=='Li']['datakey'].unique()
    else:
        # Only sessions > 20190511 should have regular gratings
        v1_include = [#'20190511_JC083',  only if has_gratings=True
                      '20190522_JC084',
                      '20190622_JC085',
                      '20190613_JC097', 
                      '20190616_JC097', 
                      '20190617_JC097',
                      '20191006_JC110']
 
        lm_include = [#'20190513_JC078', # only include if has_gratings=True
                      #'20190603_JC080', # only include if has_gratings=True
                      #'20190512_JC083', # OR, 20190517_JC083 slightly worse?
                      '20190525_JC084',
                      '20190627_JC091',
                      '20190618_JC097']
       
        li_include = ['20190605_JC090',
                      #'20190602_JC091', # 20190607_JC091 also good
                      #'20190614_JC091', # 20190606_JC091 also good 
                      #'20191008_JC091',
                      #'20190612_JC099', # 20190609_JC099 also good
                      '20190617_JC099',
                      '20191018_JC113',
                      '20191105_JC117',
                      '20191111_JC120']
        
        if filter_by=='last': #is False:
            lm_include.extend(['20190517_JC083']) # 20190512_JC083
            li_include.extend(['20190607_JC091',  # 20190602_JC091
                               '20190614_JC091',  # 20190606_JC091 
                               '20190612_JC099']) # 20190609_JC099

        elif filter_by=='first': #default_take_first:
            lm_include.extend(['20190512_JC083'])
            li_include.extend(['20190602_JC091', 
                               '20190606_JC091', 
                               '20190609_JC099'])

#        elif filter_by == 'unique_a':
#            lm_include.extend(['20190512_JC083'])
#            li_include.extend(['20190602_JC091', 
#                               '20190614_JC091', 
#                               '20190612_JC099'])

        else:
            print("Filter <%s> UNKNOWN." % str(filter_by))
            return None
              
        if has_gratings:
            v1_include.extend(['20190511_JC083'])

            lm_include.extend(['20190513_JC078',
                               '20190603_JC080',
                               '20190512_JC083']) # Also, 20190517_JC083
        else:
            # Sometimes, same FOV scanned twice, pick earlier/later
            if filter_by=='first': #default_take_first:
                v1_include.extend(['20190420_JC076',  # not 20190501_JC076
                                   '20190507_JC083']) # not 20190510_JC083
                lm_include.extend(['20190504_JC078']) # not 20190509_JC078
            else:
                # pick the later one
                v1_include.extend(['20190501_JC076',
                                   '20190510_JC083'])

                lm_include.extend(['20190509_JC078'])
            
            # and add good dsets without gratings
            lm_include.extend(['20190430_JC078',
                               '20190506_JC080',
                               '20190508_JC083'])
            
            li_include.extend(['20190502_JC076'])


    included_ = [v1_include, lm_include, li_include]
    for incl in included_:
        included_sessions.extend(incl)
    included_sessions = [i for i in list(set(included_sessions)) \
                                    if i not in excluded_sessions]

    if as_dict:
        return {'V1': v1_include, 'Lm': lm_include, 'Li': li_include}
    
    return included_sessions


def make_metadict_from_df(df):
    ddict=None
    ddict = dict((k, list(v['datakey'].unique())) for k, v in df[['visual_area', 'datakey']]\
                    .drop_duplicates().groupby(['visual_area']))

    return ddict

def get_gratings_datasets(filter_by='most_fits', excluded_sessions=[], as_dict=True,
                        response_type='dff', responsive_test='nstds', responsive_thr=10.0):
    '''
    filter_by:  None to return all, most_fits to retun ones with most # responsive.
    Others are untested
    ''' 
    from pipeline.python.retinotopy import segment_retinotopy as seg

    included_sessions=[]

    #excluded_sessions.extend(always_exclude)
    # Get blob metadata only - and only if have RFs
    sdata = get_aggregate_info() 
    if has_rfs:
        edata = include_dsets_with(sdata, experiment='gratings', also_include='rfs')
    else:
        edata = sdata[sdata['experiment']=='gratings']

    if filter_by is 'most_fits':
        # Get meta info 
        dsets = edata[edata['experiment']=='gratings'].copy()
        assigned_cells = seg.get_cells_by_area(dsets)
        best_dfs = get_dsets_with_most_gratings(dsets, assigned_cells, response_type=response_type,
                                responsive_test=responsive_test, responsive_thr=responsive_thr)
        if as_dict:
            included_sessions = make_metadict_from_df(best_dfs)
        else:
            included_sessions = best_dfs
    else:
        included_sessions = get_gratings_datasets_drop(filter_by=filter_by, 
                                    excluded_sessions=excluded_sessions, as_dict=as_dict)

    return included_sessions 

def get_gratings_datasets_drop(filter_by='first', excluded_sessions=[], as_dict=True):

    included_sessions = []
    
    # Blobs runs w/ incorrect stuff
    always_exclude = ['20190426_JC078']
    excluded_sessions.extend(always_exclude)
    
    if filter_by is None:
        sdata = get_aggregate_info()
        bd = sdata[sdata['experiment']=='gratings'].copy()
        v1_include = bd[bd['visual_area']=='V1']['datakey'].unique()
        lm_include = bd[bd['visual_area']=='Lm']['datakey'].unique()
        li_include = bd[bd['visual_area']=='Li']['datakey'].unique()
   
    else:
        # Only sessions > 20190511 should have regular gratings
        v1_include = ['20190511_JC083', 
                      '20190522_JC084',
                      '20190622_JC085',
                      '20190613_JC097', '20190616_JC097', '20190617_JC097',
                      '20191006_JC110']
 
        lm_include = ['20190513_JC078', 
                      '20190603_JC080', 
                      #'20190512_JC083', # 20190517_JC083 slightly worse?
                      '20190525_JC084',
                      '20190627_JC091',
                      '20190618_JC097']
       
        li_include = ['20190605_JC090',
                      #'20190602_JC091', # 20190607_JC091 also good
                      #'20190614_JC091', # 20190606_JC091 also good 
                      '20191008_JC091',
                      #'20190612_JC099', # 20190609_JC099 also good
                      '20190617_JC099',
                      '20191018_JC113',
                      '20191105_JC117',
                      '20191111_JC120']

        if filter_by=='first':
            lm_include.extend(['20190512_JC083'])
            li_include.extend(['20190602_JC091',
                               '20190606_JC091',
                               '20190609_JC099']) 
        else:
            lm_include.extend(['20190517_JC083'])
            li_include.extend(['20190607_JC091',
                               '20190614_JC091',
                               '20190612_JC099'])
       
    #else:
    #    print("Filter <%s> UNKNOWN." % str(filter_by))
    #    return None
       
    included_ = [v1_include, lm_include, li_include]

    for incl in included_:
        included_sessions.extend(incl)
    included_sessions = [i for i in list(set(included_sessions)) if i not in excluded_sessions]

    if as_dict:
        return {'V1': v1_include, 'Lm': lm_include, 'Li': li_include}
    
    return included_sessions

def get_rf_datasets(filter_by='max_responsive', as_dict=True, excluded_sessions=[]): 
    from pipeline.python.retinotopy import segment_retinotopy as seg

    if filter_by in ['max_responsive', 'most_cells']: 
        sdata = get_aggregate_info(traceid=traceid, fov_type=fov_type, state=state)
        rf_dsets = sdata[(sdata['experiment'].isin(['rfs', 'rfs10'])) & ~(sdata['datakey'].isin(excluded_sessions))]
        assigned_cells = seg.get_cells_by_area(rf_dsets)
        rf_dsets_max = get_dsets_with_max_rfs(rf_dsets, assigned_cells)
        
        if as_dict:
            rf_dict = dict((v, g['datakey'].unique()) for v, g in rf_dsets_max.groupby(['visual_area']))
            return rf_dict
        else:
            return rf_dsets_max['datakey'].unique()
    else:
        rf_dsets = get_rf_datasets_drop(filter_by=filter_by, excluded_sessions=excluded_sessions, as_dict=as_dict)
        return rf_dsets


def get_rf_datasets_drop(filter_by='drop_repeats', excluded_sessions=[], as_dict=True, return_excluded=False):
    #TODO:  fix this to return INCLUDED dsets -- this is for RFs
    '''From classifications/retino_structure.py --  
    '''
    
    ddict = all_datasets_by_area()
    
    # Blobs runs w/ incorrect stuff
    always_exclude = ['20190426_JC078']
    excluded_sessions.extend(always_exclude)
    
    if filter_by is None:
        v1_repeats = []
        lm_repeats = []
        li_repeats = []
    
    elif filter_by=='drop_repeats':
        # Sessions with repeat FOVs
        print("droppin repeats")
        v1_repeats = ['20190501_JC076', 
                      '20190507_JC083', '20190510_JC083', #'20190511_JC083']
                      '20190615_JC097_fov1', '20190615_JC097_fov2', '20190615_JC097_fov3',
                      '20191004_JC110_fov2']
 
        lm_repeats = ['20190426_JC078', '20190504_JC078', '20190430_JC078', 
                      '20190506_JC080', 
                      '20190508_JC083', '20190512_JC083', #'20190517_JC083']
                      '20190615_JC097',
                      '20191108_JC113']
       
        li_repeats = ['20190422_JC076',
                      '20190602_JC091',  
                      '20190606_JC091', 
                      #'20190527_JC092',
                      '20190609_JC099',
                      '20191012_JC113_fov1',
                      '20191012_JC113_fov2', '20191017_JC113',
                      '20191104_JC117_fov1',
                      '20191106_JC120_fov1']
    else:
        print("Filter <%s> UNKNOWN." % str(filter_by))
        return None
       
    also_exclude = [v1_repeats, lm_repeats, li_repeats]

    for excl in also_exclude:
        excluded_sessions.extend(excl)
    excluded_sessions = list(set(excluded_sessions))

    print("[filter_by=%s] Excluding %i total repeats" % (str(filter_by), len(excluded_sessions)))

    if return_excluded:
        return excluded_sessions
    
    session_dict = {}
    included_sessions = []
    for k, v in ddict.items():
        included = [vv for vv in v if vv not in excluded_sessions]
        session_dict[k] = included
        included_sessions.extend(included)
        
    if as_dict:
        return session_dict
    else: 
        return included_sessions


#def get_assigned_cells_with_rfs(rf_dsets):
def select_best_fovs(counts_by_fov, criterion='max', colname='cell'):
    # Cycle thru all dsets and drop repeats
    fovkeys = get_sorted_fovs()
    incl_dsets=[]
    for (visual_area, animalid), g in counts_by_fov.groupby(['visual_area', 'animalid']):
        curr_dsets=[]
        try:
            # Check for FOVs that had wrongly assigned visual areas compared to assigned
            if visual_area not in fovkeys[animalid].keys():
                v_area=[]
                for v, vdict in fovkeys[animalid].items():
                    for dk in g['datakey'].unique():
                        a_match = [k for k in vdict for df in g['datakey'].unique() if \
                                    '%s_%s' % (dk.split('_')[0], dk.split('_')[[2]]) in k \
                                     or dk.split('_')[0] in k]
                        if len(a_match)>0:
                            v_area.append(v)
                if len(v_area)>0:
                    curr_dsets = fovkeys[animalid][v_area[0]]
            else:
                curr_dsets = fovkeys[animalid][visual_area]
        except Exception as e:
            print("[%s] Animalid does not exist: %s " % (visual_area, animalid))
            continue

        # Check for sessions/dsets NOT included in current visual area dict
        # This is correctional: if a given FOV is NOT in fovkeys dict, it was a non-repeat FOV
        # for that visual area.
        dkeys_flat = list(itertools.chain(*curr_dsets))
        # These are datakeys assigned to current visual area:
        reformat_dkeys_check = ['%s_%s' % (s.split('_')[0], s.split('_')[2]) \
                                    for s in g['datakey'].unique()] 
        # Assigned dkeys not in original source dict (which was made manually)
        missing_segmented_fovs = [s for s in reformat_dkeys_check \
                                if (s not in dkeys_flat) and (s.split('_')[0] not in dkeys_flat) ] 

        #for s in missing_segmented_fovs:
        #    curr_dsets.append(s)

        missing_dsets=[]
        for fkey in missing_segmented_fovs:
            found_areas = [k for k, v in fovkeys[animalid].items() \
                             if any([fkey in vv for vv in v]) or any([fkey.split('_')[0] in vv for vv in v])]    
            for va in found_areas:
                if fovkeys[animalid][va] not in missing_dsets:
                    missing_dsets.append(fovkeys[animalid][va])
        curr_dsets.extend(list(itertools.chain(*missing_dsets)))

        # Select "best" dset if there is a repeat
        if g.shape[0]>1:
            for dkeys in curr_dsets:
                if isinstance(dkeys, tuple):
                    # Reformat listed session strings in fovkeys dict.
                    curr_datakeys = ['_'.join([dk.split('_')[0], animalid, dk.split('_')[-1]])
                            if len(dk.split('_'))>1 \
                            else '_'.join([dk.split('_')[0], animalid, 'fov1']) for dk in dkeys]
                    # Get df data for current "repeat" FOVs
                    which_fovs = g[g['datakey'].isin(curr_datakeys)]
                    # Find which has most cells
                    max_loc = choose_best_fov(which_fovs, criterion=criterion, colname=colname)
                    #max_loc = np.where(which_fovs['cell']==which_fovs['cell'].max())[0]
                    incl_dsets.append(which_fovs.iloc[max_loc])
                else:
                    # THere are no repeats, so just format, then append df data
                    curr_datakey = '_'.join([dkeys.split('_')[0], animalid, dkeys.split('_')[-1]]) \
                                    if len(dkeys.split('_'))>1 \
                                    else '_'.join([dkeys.split('_')[0], animalid, 'fov1'])
                    incl_dsets.append(g[g['datakey']==curr_datakey])
        else:
            #if curr_dsets=='%s_%s' % (session, fov) or curr_dsets==session:
            incl_dsets.append(g)
    incl = pd.concat(incl_dsets, axis=0).reset_index(drop=True)
    
    return incl


def choose_best_fov(which_fovs, criterion='max', colname='cell'):
    if criterion=='max':
        max_loc = np.where(which_fovs[colname]==which_fovs[colname].max())[0]
    else: 
        max_loc = np.where(which_fovs[colname]==which_fovs[colname].min())[0]

    return max_loc

def get_dsets_with_most_cells(allthedata): #assigned_cells):
    # Count how many cells fit total per site
    countby = ['visual_area', 'datakey', 'cell']
    counts_by_fov = allthedata[countby].drop_duplicates()\
                        .groupby(['visual_area', 'datakey']).count().reset_index()
    counts_by_fov = split_datakey(counts_by_fov)
    
    best_dfs = select_best_fovs(counts_by_fov, criterion='max', colname='cell')
   
    return best_dfs 
    
def get_dsets_with_max_rfs(rf_dsets, assigned_cells):

    # Load all RF data
    from pipeline.python.classifications import rf_utils as rfutils
    all_rfdfs = rfutils.load_aggregate_rfs(rf_dsets)
    all_rfs = get_rfdata(assigned_cells, all_rfdfs, verbose=False, average_repeats=True)
   
    best_dfs = get_dsets_with_most_cells(all_rfs) #, assigned_cells)
    
    return best_dfs


def get_retino_metadata(experiment='retino', animalids=None,
                        roi_type='manual2D_circle', traceid=None,
                        rootdir='/n/coxfs01/2p-data', 
                        aggregate_dir='/n/coxfs01/julianarhee/aggregate-visual-areas'):

    sdata_fpath = os.path.join(aggregate_dir, 'dataset_info.pkl')
    with open(sdata_fpath, 'rb') as f:
        sdata = pkl.load(f)
   
    meta_list=[]
    for (animalid, session, fov), g in sdata.groupby(['animalid', 'session', 'fov']):
        if animalids is not None:
            if animalid not in animalids:
                continue
        exp_list = [e for e in g['experiment'].values if experiment in e] 
        if len(exp_list)==0:
            print('skipping, no retino (%s, %s, %s)' % (animalid, session, fov)) 
        retino_dirs = glob.glob(os.path.join(rootdir, animalid, session, fov, '%s*' % experiment,
                                'retino_analysis'))
        # get analysis ids for non-pixel
        for retino_dir in retino_dirs:
            retino_run = os.path.split(os.path.split(retino_dir)[0])[-1]
            if traceid is None:
                rid_fpath = glob.glob(os.path.join(retino_dir, 'analysisids_*.json'))[0]
                with open(rid_fpath, 'r') as f:
                    retids = json.load(f)
                traceids = [r for r, res in retids.items() if res['PARAMS']['roi_type']==roi_type] 
                for traceid in traceids: 
                    meta_list.append(tuple([animalid, session, fov, retino_run, traceid]))
            else:
                meta_list.append(tuple([animalid, session, fov, retino_run, traceid]))

    return meta_list


def aggregate_responsive_retino(assigned_rois, traceid='traes001', mag_thr=0.01, 
                                pass_criterion='max', verbose=False, create_new=False,
                                dst_dir='/n/coxfs01/julianarhee/aggregate-visual-areas/data-stats'):

    from pipeline.python.retinotopy import utils as ret_utils
    
    aggr_fpath = os.path.join(dst_dir, 'aggr_retino_magratio_%s-thr-%.2f.pkl' % (pass_criterion, mag_thr))

    if not create_new:
        try:
            with open(aggr_fpath, 'rb') as f:
                retino_cells = pkl.load(f)

        except Exception as e:
            print("---> Error loading aggr retino cells. Re-running.")
            create_new=True

    if create_new:
        tmp_=[]
        for (visual_area, datakey), g in assigned_rois.groupby(['visual_area', 'datakey']):

            # Load retino fft results
            session, animalid, fovn = datakey.split('_')
            fov = 'FOV%i_zoom2p0x' % int(fovn[3:])

            responsive_cells, _ = ret_utils.get_responsive_cells(animalid, session, fov, traceid=traceid, retinorun=None, 
                                     pass_criterion=pass_criterion, mag_thr=mag_thr, create_new=False)

            keep_cells = np.intersect1d(g['cell'].values, responsive_cells)
            nrois_t = len(g['cell'].unique())

            rois_ = g[g['cell'].isin(keep_cells)]
            tmp_.append(rois_)
            if verbose:
                print("[%s,%s] %i of %i cells repsonsive" % (visual_area, datakey, len(keep_cells), nrois_t))

        retino_cells = pd.concat(tmp_, axis=0).reset_index(drop=True)
        
        # Save
        with open(aggr_fpath, 'wb') as f:
            pkl.dump(retino_cells, f, protocol=pkl.HIGHEST_PROTOCOL)
        print("---> Saved: %s" % aggr_fpath)
 
    return retino_cells


#
def get_responsive_all_experiments(sdata, response_type='dff', traceid='traces001', 
                                  responsive_test='nstds', responsive_thr=10.0, 
                                  trial_epoch='stimulus', verbose=False, visual_areas=None,
                                  retino_mag_thr=0.01, retino_pass_criterion='max', return_missing=True):
    '''
    For all segmented visual areass and cells, return the ones that are "responsive" for each experiment type.
    '''
    from pipeline.python.retinotopy import segment_retinotopy as seg
    c_=[]
    missing_seg=[]
    for experiment in ['gratings', 'blobs', 'rfs', 'retino']:
        if experiment == 'rfs':
            edata = sdata[sdata['experiment'].isin(['rfs', 'rfs10'])]
            # rfdf = aggr.load_rfdf_and_pos(edata, response_type=response_type, 
            #                               rf_filter_by=None, reliable_only=True)
            assigned_rois, missing_ = seg.get_cells_by_area(edata, return_missing=True)
            tmpcells = get_dsets_with_max_rfs(edata, assigned_rois)
        elif experiment=='retino':
            edata = sdata[sdata['experiment'].isin(['retino'])]
            assigned_rois, missing_ = seg.get_cells_by_area(edata, return_missing=True)
            tmpcells = aggregate_responsive_retino(assigned_rois, traceid=traceid,
                                                   mag_thr=retino_mag_thr, 
                                                   pass_criterion=retino_pass_criterion, verbose=verbose,
                                                    create_new=False)
        else:
            exp_meta, tmpcells, EXP, missing_ = get_source_data(experiment, 
                                                response_type=response_type,
                                                responsive_test=responsive_test, 
                                                responsive_thr=responsive_thr, 
                                                trial_epoch=trial_epoch,
                                                return_missing=True, check_configs=True)

        if visual_areas is None:
            visual_areas = tmpcells['visual_area'].unique()

        cells_ = tmpcells[tmpcells['visual_area'].isin(visual_areas)]
        #cells.groupby(['visual_area']).count()
        cells_['experiment'] = experiment
        print(cells_.shape)
        c_.append(cells_)
        missing_seg.extend(missing_)

    aggr_cells = pd.concat(c_, axis=0).reset_index(drop=True)

    missing_seg = list(set(missing_seg))
    assert len([k for k in aggr_cells['datakey'].unique() if k in missing_seg])==0, \
    "There are included dsets w/ missing seg. Fix this."
    
    if return_missing:
        return aggr_cells, missing_seg
    else:
        return aggr_cels

def get_ncells_by_experiment(aggr_cells, total_nrois, experiment=None):
    
    if experiment is None:
        # Get all cells
        visual_cells = aggr_cells[['visual_area', 'datakey', 'cell']].drop_duplicates()
    else:
        if isinstance(experiment, str):
            curr_cells = aggr_cells[aggr_cells['experiment']==experiment].copy()
        else:
            curr_cells = aggr_cells[aggr_cells['experiment'].isin(experiment)].copy()
        visual_cells = curr_cells[['visual_area', 'datakey', 'cell']].drop_duplicates()
    # get counts
    total_visual = visual_cells.groupby(['visual_area', 'datakey']).count().reset_index()\
                    .rename(columns={'cell': 'visual'})
    counts = total_visual.merge(total_nrois)
    counts['fraction'] = counts['visual']/counts['total']
    
    #total_visual.groupby(['visual_area']).sum()
    return counts


# --------------------------------------------------------------------------
# Data shaping/formating
# --------------------------------------------------------------------------

# Overlaps, cell assignments, etc.
def get_neuraldf_for_cells_in_area(cells, MEANS, datakey=None, visual_area=None):
    '''
    For a given dataframe (index=trials, columns=cells), only return cells
    in specified visual area
    '''
    neuraldf=None
    try:
        if isinstance(MEANS, dict):
            if 'V1' in MEANS.keys(): # dict of dict
                neuraldf_dict = MEANS[visual_area]
            else:
                neuraldf_dict = MEANS
        elif isinstance(MEANS, pd.DataFrame):
            MEANS = neuraldf_dataframe_to_dict(MEANS)
            neuraldf_dict = MEANS[visual_area] 

        assert datakey in neuraldf_dict.keys(), "%s--not found in RESPONSES" % datakey
        assert datakey in cells['datakey'].values, "%s--not found in SEGMENTED" % datakey

        curr_rois = cells[(cells['datakey']==datakey) 
                        & (cells['visual_area']==visual_area)]['cell'].astype(int).values
        curr_cols = [i for i in np.array(curr_rois.copy()) if i in neuraldf_dict[datakey].columns.tolist()]
        #curr_cols = list(curr_rois.copy())
        neuraldf = neuraldf_dict[datakey][curr_cols].copy()
        neuraldf['config'] = neuraldf_dict[datakey]['config'].copy()
    except Exception as e:
        return neuraldf
 
    return neuraldf


def get_active_cells_in_current_datasets(rois, MEANS, verbose=False):
    '''
    For segmented cells, return those cells that are responsive (MEANS)
    '''
    d_=[]
    for (visual_area, datakey), g in rois.groupby(['visual_area', 'datakey']):
        if datakey not in MEANS.keys():
            #print("missing: %s" % datakey)
            continue
        included_cells = [i for i in MEANS[datakey].columns if i in g['cell'].values]
        tmpd = g[g['cell'].isin(included_cells)].copy()
        if verbose:
            print('[%s] %s: %i of %i responsive' % (visual_area, datakey, len(included_cells), len(g)))
        d_.append(tmpd)

    cells = pd.concat(d_, axis=0).reset_index(drop=True)
    
    return cells

def load_aggregate_data(experiment, traceid='traces001', response_type='dff', epoch='stimulus', 
                       responsive_test='ROC', responsive_thr=0.05, n_stds=0.0,
                        check_configs=True, equalize_now=False, zscore_now=False,
                        return_configs=False, images_only=False, diff_configs=['20190314_JC070_fov1'],
                        aggregate_dir='/n/coxfs01/julianarhee/aggregate-visual-areas'):
    '''
    Return dict of neural dataframes (keys are datakeys).

    check_configs (bool) : Get each dataset's stim configs (sdf), and rename matching configs to match master.
    Note, check_configs *only* tested with experiment=blobs.

    equalize_now (bool) : Random sample trials per config so that same # trials/config.
    zscore_now (bool) : Zscore neurons' responses.
    ''' 
    data_outfile = get_aggregate_data_filepath(experiment, traceid=traceid, 
                        response_type=response_type, epoch=epoch,
                        responsive_test=responsive_test, 
                        responsive_thr=responsive_thr, n_stds=n_stds,
                        aggregate_dir=aggregate_dir)
    # print("...loading: %s" % data_outfile)

    with open(data_outfile, 'rb') as f:
        MEANS = pkl.load(f)
    print("...loading: %s" % data_outfile)

    #### Fix config labels 
    if check_configs or return_configs:
        SDF, renamed_configs = check_sdfs(MEANS.keys(), traceid=traceid, 
                                          images_only=images_only, return_incorrect=True)
        if check_configs:
            sdf_master = get_master_sdf(images_only=images_only)
            for k, renamed_c in renamed_configs.items():
                if k in diff_configs:
                    print("(skipping %s)" % k)
                    continue
                #print("... updating %s" % k)
                updated_cfgs = [renamed_c[cfg] for cfg in MEANS[k]['config']]
                MEANS[k]['config'] = updated_cfgs

    if images_only: #Update MEANS dict
        for k, md in MEANS.items():
            incl_configs = SDF[k].index.tolist()
            MEANS[k] = md[md['config'].isin(incl_configs)]

    if equalize_now:
        # Get equal counts
        print("---equalizing now---")
        MEANS = equal_counts_per_condition(MEANS)

    if zscore_now:
        MEANS = zscore_data(MEANS)

    if return_configs: 
        return MEANS, SDF
    else:
        return MEANS

def equal_counts_per_condition(MEANS):
    '''
    MEANS: dict
        keys = datakeys
        values = neural dataframes (columns=rois, index=trial numbers)
    
    Resample so that N trials per condition is the same as min N.
        '''

    for k, v in MEANS.items():
        v_df = equal_counts_df(v)
        
        MEANS[k] = v_df

    return MEANS


def get_source_data(experiment, traceid='traces001',
                    responsive_test='nstds', responsive_thr=10., response_type='dff',
                    trial_epoch='stimulus', fov_type='zoom2p0x', state='awake', 
                    verbose=False, visual_area=None, datakey=None, return_configs=False,
                    images_only=False,
                    return_missing=False, check_configs=True, equalize_now=False,zscore_now=False): 
    '''
    Returns metainfo, cell dataframe, and dict of neuraldfs for all 
    responsive cells in assigned visual areas.
    
    Loads dict of neuraldfs.
    Gets all assigned cells for each datakey.
    Returns all responsive cells assigned to visual area.
    '''
    from pipeline.python.retinotopy import segment_retinotopy as seg

    #### Get neural responses 
    means0 = load_aggregate_data(experiment, 
                responsive_test=responsive_test, responsive_thr=responsive_thr, 
                response_type=response_type, epoch=trial_epoch,
                check_configs=check_configs, equalize_now=equalize_now, zscore_now=zscore_now,
                return_configs=return_configs, images_only=images_only)
    if return_configs:
        MEANS, SDF = means0
    else:
        MEANS = means0
 
   # Get dataset metainfo
    sdata, rois = get_aggregate_info(traceid=traceid, fov_type=fov_type, state=state, return_cells=True)
    excluded_datakeys = ['20190327_JC073_fov1', '20190314_JC070_fov1'] if images_only else []
    print("SDF, images_only=%s (excluding dsetes: %s)" % (str(images_only), str(excluded_datakeys)))
    sdata = sdata[~sdata.isin(excluded_datakeys)]

    edata = sdata[sdata['experiment']==experiment].copy()
     
    # Get cell assignemnts (based on retinotopy/segment_retinotopy.py)
    #rois, missing_seg = seg.get_cells_by_area(edata, return_missing=True) do this already in get_aggregate_info()
    cells = get_active_cells_in_current_datasets(rois, MEANS, verbose=False)

    # Assign global index
    cells = assign_global_cell_id(cells)

    if (visual_area is not None) or (datakey is not None):
        means0 = MEANS.copy()
        meta0 = edata.copy()
        cells = select_cells(cells, visual_area=visual_area, datakey=datakey)
        dkeys = cells['datakey'].unique()
        vareas = cells['visual_area'].unique()
        edata = meta0[(meta0['datakey']==datakey) & (meta0['visual_area']==visual_area)] 
        MEANS = dict((k, means0[k]) for k in dkeys) #MEANS[datakey]
       
    if return_missing:
        if return_configs:
            return sdata, cells, MEANS, SDF, missing_seg
        else:
            return sdata, cells, MEANS, missing_seg
    else:
        if return_configs:
            return sdata, cells, MEANS, SDF
        else:
            return sdata, cells, MEANS

def zscore_neuraldf(neuraldf):
    cols_drop = ['config', 'trial'] if 'trial' in neuraldf.columns else ['config']
    data = neuraldf.drop(cols_drop, 1) #sample_data[curr_roi_list].copy()
    zdata = (data - np.nanmean(data)) / np.nanstd(data)
    zdf = pd.DataFrame(zdata, index=neuraldf.index, columns=data.columns)
    zdf['config'] = neuraldf['config']
    if 'trial' in neuraldf.columns:
        zdf['trial'] = neuraldf['trial']

    return zdf


def zscore_data(MEANS):
    for k, v in MEANS.items():
        zdf = zscore_neuraldf(v)
        MEANS[k] = zdf

    return MEANS

def equal_counts_df(neuraldf, equalize_by='config'):
    curr_counts = neuraldf[equalize_by].value_counts()
    if len(curr_counts.unique())==1:
        return neuraldf #continue
        
    min_ntrials = curr_counts.min()
    all_cfgs = neuraldf[equalize_by].unique()

    kept_trials=[]
    for cfg in all_cfgs:
        curr_trials = neuraldf[neuraldf[equalize_by]==cfg].index.tolist()
        np.random.shuffle(curr_trials)
        kept_trials.extend(curr_trials[0:min_ntrials])
    kept_trials=np.array(kept_trials)

    assert len(neuraldf.loc[kept_trials][equalize_by].value_counts().unique())==1, \
            "Bad resampling... Still >1 n_trials"

    return neuraldf.loc[kept_trials]




def get_master_sdf(images_only=False):

    obj = util.Objects('JC084', '20190522', 'FOV1_zoom2p0x', traceid='traces001')
    sdf_master = obj.get_stimuli()
    if images_only:
        sdf_master=sdf_master[sdf_master['morphlevel']!=-1].copy()

    return sdf_master


def check_sdfs(stim_datakeys, experiment='blobs', traceid='traces001', images_only=False, 
                rename=True, return_incorrect=False, diff_configs=['20190314_JC070_fov1'] ):

    '''
    Checks config names and reutrn master dict of all stimconfig dataframes
    Notes: only tested with blobs, and renaming only works with blobs.
    '''

    sdf_master = get_master_sdf(images_only=False)
    n_configs = sdf_master.shape[0]
    
    #### Check that all datasets have same stim configs
    SDF={}
    renamed_configs={}
    for datakey in stim_datakeys:
        session, animalid, fov_ = datakey.split('_')
        fovnum = int(fov_[3:])
        if experiment=='blobs':
            obj = util.Objects(animalid, session, 'FOV%i_zoom2p0x' %  fovnum, traceid=traceid)
        elif experiment=='gratings':
            obj = util.Gratings(animalid, session, 'FOV%i_zoom2p0x '% fovnum, traceid=traceid)
        else:
            print("Unvailable experiemnt type for master SDFs: %s" % experiment)
            return None

        sdf = obj.get_stimuli()
        #if images_only:
        #    sdf = sdf[sdf['morphlevel']!=-1]

        if len(sdf['xpos'].unique())>1 or len(sdf['ypos'].unique())>1:
            print("*Warning* <%s> More than 1 pos? x: %s, y: %s" \
                    % (datakey, str(sdf['xpos'].unique()), str(sdf['ypos'].unique())))
 
        if experiment=='blobs' and (sdf.shape[0]!=sdf_master.shape[0]):
            #print("%s: diff keys" % datakey)
#            if sdf.shape[0]==45:
#                # missing morphlevels, likely lum controls
#                c_list = sdf.index.tolist()
#                sdf.index = ['config%03d' % (int(ci[6:])+5) for ci in c_list]
#                updated_keys = dict((k, v) for k, v in zip(c_list, sdf.index.tolist()))
            #else:
            # Compare key config values and find "matches"
            key_names = ['morphlevel', 'size']
            updated_keys={}
            for old_ix in sdf.index:
                #try:
                new_ix = sdf_master[(sdf_master[key_names] == sdf.loc[old_ix,  key_names]).all(1)].index[0]
                #except Exception as e:
                    #print("Not found: %s" % str(sdf.loc[old_ix]))
                #    continue
                updated_keys.update({old_ix: new_ix})
       
            if rename and (datakey not in diff_configs): 
                sdf = sdf.rename(index=updated_keys)

            # Save renamed key 
            renamed_configs[datakey] = updated_keys
 
        if experiment=='blobs' and images_only:
            SDF[datakey] = sdf[sdf['morphlevel']!=-1].copy()
        else:
            SDF[datakey] = sdf
    
    ignore_params = ['xpos', 'ypos', 'position', 'color']
    if experiment != 'blobs':
        ignore_params.extend(['size'])
    
    compare_params = [p for p in sdf_master.columns if p not in ignore_params] 
    different_configs = renamed_configs.keys()

    sdf_master = get_master_sdf(images_only=images_only)
    assert all([all(sdf_master[compare_params]==d[compare_params]) for k, d in SDF.items() \
              if k not in different_configs]), "Incorrect stimuli..."

    if return_incorrect:
        return SDF, renamed_configs
    else:
        return SDF


def experiment_datakeys(experiment='blobs', has_gratings=False, has_rfs=False, stim_filterby='most_fits',
                        experiment_only=True, traceid='traces001'):

    # Drop duplicates and whatnot fovs
    if experiment=='blobs':
        g_str = 'hasgratings' if has_gratings else 'blobsonly'
        edata = get_blob_datasets(filter_by=stim_filterby, has_gratings=has_gratings, has_rfs=has_rfs,
                                     as_dict=False)
    else:
        g_str = 'gratingsonly'
        edata = get_gratings_datasets(filter_by=stim_filterby, has_rfs=has_rfs, as_dict=False)
    
    exp_dict = make_metadict_from_df(edata)


    dictkeys = [d for d in list(itertools.chain(*exp_dict.values()))]
    stim_datakeys = ['%s_%s_fov%i' % (s.split('_')[0], s.split('_')[1], 
                       edata[(edata['animalid']==s.split('_')[1]) 
                        & (edata['session']==s.split('_')[0])]['fovnum'].unique()[0]) for s in dictkeys]

    if experiment_only:                 
        emeta = edata[edata['datakey'].isin(stim_datakeys)]
    else:
        sdata = get_aggregate_info(traceid=traceid)
        emeta = pd.concat([sdata[(sdata['visual_area']==v) & (sdata['datakey'].isin(dkeys))] \
                            for v, dkeys in exp_dict.items()]) 

    return emeta, exp_dict #expmeta

def neuraldf_dict_to_dataframe(NEURALDATA, response_type='response'):
    ndfs = []

    
    if isinstance(NEURALDATA[NEURALDATA.keys()[0]], dict):
        for visual_area, vdict in NEURALDATA.items():
            for datakey, neuraldf in vdict.items():
                metainfo = {'visual_area': visual_area, 'datakey': datakey}
                ndf = add_meta_to_df(neuraldf.copy(), metainfo)
                ndf['trial'] = ndf.index.tolist()
                id_vars = ['visual_area', 'datakey', 'config', 'trial']
                if 'arousal' in neuraldf.columns:
                    id_vars.append('arousal')
                melted = pd.melt(ndf, id_vars=id_vars, 
                                 var_name='cell', value_name=response_type)
                ndfs.append(melted)
    else:
        for datakey, neuraldf in NEURALDATA.items():
            metainfo = {'datakey': datakey}
            ndf = add_meta_to_df(neuraldf.copy(), metainfo)
            ndf['trial'] = ndf.index.tolist()
            id_vars = ['datakey', 'config', 'trial']
            if 'arousal' in neuraldf.columns:
                id_vars.append('arousal')

            melted = pd.melt(ndf, id_vars=id_vars, 
                             var_name='cell', value_name=response_type)
            ndfs.append(melted)

    NDATA = pd.concat(ndfs, axis=0)
   
    return NDATA

def unstacked_neuraldf_to_stacked(ndf, response_type='response', id_vars = ['config', 'trial']):
    ndf['trial'] = ndf.index.tolist()
    melted = pd.melt(ndf, id_vars=id_vars, 
                     var_name='cell', value_name=response_type)

    return melted


def stacked_neuraldf_to_unstacked(ndf): #neuraldf):
#    l_ = [g[['response']].T.rename(columns=g['cell']) for (cg, trial), g in neuraldf.groupby(['config', 'trial'])]
#    trialnums = [trial for (cg, trial), g in neuraldf.groupby(['config', 'trial'])]
#    configvals = [cg for (cg, trial), g in neuraldf.groupby(['config', 'trial'])]
#
#    rdf = pd.concat(l_, axis=0)
#    rdf.index=trialnums
#    rdf['config'] = configvals
    other_cols = [k for k in ndf.columns if k not in ['cell', 'response']]
    n2 = ndf.pivot_table(columns=['cell'], index=other_cols)

    rdf = pd.DataFrame(data=n2.values, columns=n2.columns.get_level_values('cell'), 
                 index=n2.index.get_level_values('trial'))
    rdf['config'] = n2.index.get_level_values('config')
    
    return rdf


def neuraldf_dataframe_to_dict(NDATA):
    '''
    Takes full, stacked dataframe, converts to dict of dicts
    '''
    visual_areas = NDATA['visual_area'].unique()
    NEURALDATA = dict((v, dict()) for v in visual_areas)

    for (visual_area, datakey), neuraldf in NDATA.groupby(['visual_area', 'datakey']):

        rdf = stacked_neuraldf_to_unstacked(neuraldf)

        NEURALDATA[visual_area][datakey] = rdf.sort_index()

    return NEURALDATA


def get_neuraldata(assigned_cells, MEANS, stack=False, verbose=False):
    '''
    cells (dataframe)
        Cells assigned to each datakey/visual area 
        From: get_source_data()
    MEANS (dict)
        Aggregated neuraldfs (dict, keys=dkey, values=dfs).
        From: get_source_data()

    Returns:
    
    NEURALDATA (dict)
        keys=visual areas
        values = MEANS (i.e., dict of dfs) for each visual area
        Only inclues cells that are assigned to the specified area.
    '''
    visual_areas = assigned_cells['visual_area'].unique()

    NEURALDATA = dict((visual_area, {}) for visual_area in visual_areas)
    rf_=[]
    for (visual_area, datakey), curr_c in assigned_cells.groupby(['visual_area', 'datakey']):
        if visual_area not in NEURALDATA.keys():
            print("... skipping: %s" % visual_area)
            continue

       # Get neuradf for these cells only
        neuraldf = get_neuraldf_for_cells_in_area(curr_c, MEANS, 
                                                  datakey=datakey, visual_area=visual_area)
        if verbose:
            # Which cells are in assigned area
            n_resp = int(MEANS[datakey].shape[1]-1)
            curr_assigned = curr_c['cell'].unique() 
            print("[%s] %s: %i cells responsive (%i in fov)" % (visual_area, datakey, len(curr_assigned), n_resp))
            if neuraldf is not None:
                print("Neuraldf: %s" % str(neuraldf.shape)) 
            else:
                print("No keys: %s|%s" % (visual_area, datakey))

        if neuraldf is not None:
            NEURALDATA[visual_area].update({datakey: neuraldf})

    if stack:
        NEURALDATA = neuraldf_dict_to_dataframe(NEURALDATA)

    return NEURALDATA


# -------- RFs -----------------------------------------------------------

def get_rfdata(assigned_cells, rfdf, verbose=False, visual_area=None, datakey=None, average_repeats=True):
    '''
    cells (dataframe)
        Cells assigned to each datakey/visual area 
        From: get_source_data()
    rfdf (dataframe)
        Loaded RF params and positions (from rf_utils.get_rf_positions())

    Returns:
    
    RFDATA (dataframe)
        Corresponding rfdf info for NEURALDATA cells.
    '''
    from pipeline.python.classifications import rf_utils as rfutils

    rf_=[]
    for (visual_area, datakey), curr_c in assigned_cells.groupby(['visual_area', 'datakey']):
        # Which cells have receptive fields
        cells_with_rfs = rfdf[rfdf['datakey']==datakey]['cell'].unique()

        # Which cells with RFs are in assigned area
        curr_assigned = curr_c[curr_c['cell'].isin(cells_with_rfs)]
        assigned_with_rfs = curr_assigned['cell'].unique()
        if verbose:
            print("[%s] %s: %i/%i cells with RFs" % (visual_area, datakey, len(cells_with_rfs), len(assigned_with_rfs)))

        if len(assigned_with_rfs) > 0:
            # Update RF dataframe
            if average_repeats:
                curr_rfdf = rfdf[(rfdf['datakey']==datakey) & (rfdf['cell'].isin(assigned_with_rfs))].copy()
                # Means by cell id (some dsets have rf-5 and rf10 measurements, average these)
                meanrf = curr_rfdf.groupby(['cell']).mean().reset_index()
                mean_thetas = curr_rfdf.groupby(['cell'])['theta'].apply(spstats.circmean, low=0, high=2*np.pi).values
                meanrf['theta'] = mean_thetas
                meanrf['visual_area'] = [visual_area for _ in  np.arange(0, len(assigned_with_rfs))] # reassign area
                meanrf['experiment'] = ['average_rfs' for _ in np.arange(0, len(assigned_with_rfs))]
                # Add the meta/non-numeric info
                non_num = [c for c in curr_rfdf.columns if c not in meanrf.columns and c!='experiment']
                metainfo = pd.concat([g[non_num].iloc[0] for c, g in curr_rfdf.groupby(['cell'])], axis=1).T.reset_index(drop=True)
                final_rf = pd.concat([metainfo, meanrf], axis=1)
                final_rf = rfutils.update_rf_metrics(final_rf, scale_sigma=True)
                rf_.append(final_rf)

            else: 
                for rname, rdf_ in rfdf.groupby(['experiment']):
                    curr_rfdf = rdf_[(rdf_['datakey']==datakey) & (rdf_['cell'].isin(assigned_with_rfs))]
                    if len(curr_rfdf)==0:
                        continue
                    curr_ncells = curr_rfdf['cell'].unique()

                    # Means by cell id (some dsets have rf-5 and rf10 measurements, average these)
                    meanrf = curr_rfdf.groupby(['cell']).mean().reset_index()
                    mean_thetas = curr_rfdf.groupby(['cell'])['theta'].apply(spstats.circmean, low=0, high=2*np.pi).values
                    meanrf['theta'] = mean_thetas
                    meanrf['visual_area'] = [visual_area for _ in  np.arange(0, len(curr_ncells))] # reassign area
                    meanrf['experiment'] = [rname for _ in np.arange(0, len(curr_ncells))]
                    # Add the meta/non-numeric info
                    non_num = [c for c in curr_rfdf.columns if c not in meanrf.columns and c!='experiment']
                    metainfo = pd.concat([g[non_num].iloc[0] for c, g in curr_rfdf.groupby(['cell'])], axis=1).T.reset_index(drop=True)
                    final_rf = pd.concat([metainfo, meanrf], axis=1)
                    final_rf = rfutils.update_rf_metrics(final_rf, scale_sigma=True)
                    rf_.append(final_rf)

    RFDATA = pd.concat(rf_, axis=0).reset_index(drop=True)

    return RFDATA

def get_neuraldata_and_rfdata_2(assigned_cells, rfdf, MEANS, verbose=False, stack=False):
    NEURALDATA = get_neuraldata(assigned_cells, MEANS, stack=stack, verbose=verbose)
    RFDATA = get_rfdata(assigned_cells, rfdf, verbose=verbose, visual_area=visual_area, datakey=datakey)
    updated_cells = cells_in_experiment_df(assigned_cells, RFDATA)

    return NEURALDATA, RFDATA, updated_cells

def get_common_cells_from_dataframes(NEURALDATA, RFDATA):
    ndf_list=[]
    rdf_list=[]
    for (visual_area, datakey), rfdf in RFDATA.groupby(['visual_area', 'datakey']):
        rf_rois = rfdf['cell'].unique()
        if isinstance(NEURALDATA, pd.DataFrame):
            neuraldf = NEURALDATA[(NEURALDATA['visual_area']==visual_area)
                                & (NEURALDATA['datakey']==datakey)]
            blob_rois = neuraldf['cell'].unique()
            common_rois = np.intersect1d(blob_rois, rf_rois)
            new_neuraldf = neuraldf[neuraldf['cell'].isin(common_rois)]
        else:
            if 'V1' in NEURALDATA.keys():
                neuraldf = NEURALDATA[visual_area][datakey]
            else:
                neuraldf = NEURALDATA[datakey] 
            blob_rois = neuraldf['cell'].unique()
            common_rois = np.intersect1d(blob_rois, rf_rois)
            new_neuraldf = neuraldf[common_rois]
            new_neuraldf['config'] = neuraldf['config']
            
        ndf_list.append(new_neuraldf)
        new_rfdf = rfdf[rfdf['cell'].isin(common_rois)]
        rdf_list.append(new_rfdf)
    N = pd.concat(ndf_list, axis=0)
    R = pd.concat(rdf_list, axis=0)

    return N, R

def cells_in_experiment_df(assigned_cells, rfdf):
    if isinstance(rfdf, dict):
        rfdf = neuraldf_dict_to_dataframe(rfdf) #, response_type='response'):

    updated_cells = pd.concat([assigned_cells[(assigned_cells['visual_area']==v) 
                              & (assigned_cells['datakey']==dk) 
                              & (assigned_cells['cell'].isin(g['cell'].unique()))] \
                        for (v, dk), g in rfdf.groupby(['visual_area', 'datakey'])])
    return updated_cells

def get_neuraldata_and_rfdata(assigned_cells, rfdf, MEANS,
                            visual_areas=['V1','Lm','Li'], verbose=False, stack=False):
    '''
    cells (dataframe)
        Cells assigned to each datakey/visual area 
        From: get_source_data()
    MEANS (dict)
        Aggregated neuraldfs (dict, keys=dkey, values=dfs).
        From: get_source_data()
    rfdf (dataframe)
        Loaded RF params and positions (from rf_utils.get_rf_positions())

    Returns:
    
    NEURALDATA (dict)
        keys=visual areas
        values = MEANS (i.e., dict of dfs) for each visual area
        Only inclues cells that are assigned to the specified area.

    RFDATA (dataframe)
        Corresponding rfdf info for NEURALDATA cells.
    '''
    NEURALDATA = dict((visual_area, {}) for visual_area in visual_areas)
    rf_=[]
    for (visual_area, datakey), curr_c in assigned_cells.groupby(['visual_area', 'datakey']):
        if visual_area not in NEURALDATA.keys():
            continue

        # Which cells have receptive fields
        cells_with_rfs = rfdf[rfdf['datakey']==datakey]['cell'].unique()

        # Which cells with RFs are in assigned area
        curr_assigned = curr_c[curr_c['cell'].isin(cells_with_rfs)].copy()
        assigned_with_rfs = curr_assigned['cell'].unique()
        if verbose:
            print("[%s] %s: %i cells with RFs (%i responsive)" \
                % (visual_area, datakey, len(cells_with_rfs), len(assigned_with_rfs)))

        if len(assigned_with_rfs) > 0:
            # Get neuradf for these cells only
            neuraldf = get_neuraldf_for_cells_in_area(curr_assigned, MEANS, 
                                                    datakey=datakey, visual_area=visual_area)
            NEURALDATA[visual_area].update({datakey: neuraldf})

            # Update RF dataframe
            curr_rfdf = rfdf[(rfdf['datakey']==datakey) & (rfdf['cell'].isin(assigned_with_rfs))]

            # Means by cell id (some dsets have rf-5 and rf10 measurements, average these)
            meanrf = curr_rfdf.groupby(['cell']).mean().reset_index()
            mean_thetas = curr_rfdf.groupby(['cell'])['theta'].apply(spstats.circmean, low=0, high=2*np.pi).values
            meanrf['theta'] = mean_thetas
            meanrf['visual_area'] = visual_area 
            #[visual_area for _ in  np.arange(0, len(assigned_with_rfs))] # reassign area
            meanrf['experiment'] = 'average_rfs' #['average_rfs' for _ in np.arange(0, len(assigned_with_rfs))]

            # Add the meta/non-numeric info
            non_num = [c for c in curr_rfdf.columns if c not in meanrf.columns and c!='experiment']
            metainfo = pd.concat([g[non_num].iloc[0] for c, g in curr_rfdf.groupby(['cell'])], axis=1).T.reset_index(drop=True)
            final_rf = pd.concat([metainfo, meanrf], axis=1)
            rf_.append(final_rf)

    if len(rf_)==0:
        return None, None
 
    RFDATA = pd.concat(rf_, axis=0)

    if stack:
        NEURALDATA = neuraldf_dict_to_dataframe(NEURALDATA)

    # Update cells
    updated_cells = cells_in_experiment_df(assigned_cells, RFDATA) 

    return NEURALDATA, RFDATA, updated_cells


def load_rfdf_and_pos(sdata, assigned_cells=None, response_type='dff', rf_filter_by=None, reliable_only=True,
                        rf_fit_thr=0.05, traceid='traces001', assign_cells=True,
                        aggregate_dir='/n/coxfs01/2p-data'):
    '''
    Does the same thing as rfutils.load_rfdf_with_positions(assign_cells=True)
    '''

    from pipeline.python.retinotopy import fit_2d_rfs as fitrf
    from pipeline.python.retinotopy import segment_retinotopy as seg
    from pipeline.python.classifications import rf_utils as rfutils

    rf_fit_desc = fitrf.get_fit_desc(response_type=response_type)
    aggr_rf_dir = os.path.join(aggregate_dir, 
                        'receptive-fields', '%s__%s' % (traceid, rf_fit_desc))
    # load presaved data
    reliable_str = 'reliable' if reliable_only else ''
    df_fpath =  os.path.join(aggr_rf_dir, 
                        'fits_and_coords_%s_%s.pkl' % (rf_filter_by, reliable_str))
    rf_dsets = sdata[sdata['experiment'].isin(['rfs', 'rfs10'])].copy()
    rfdf = rfutils.get_rf_positions(rf_dsets, df_fpath)

    # if assign
    if assign_cells:
        if (assigned_cells is None):
            assigned_cells, _ = seg.get_cells_by_area(rf_dsets, return_missing=True)
        rfdf = get_rfdata(assigned_cells, rfdf, average_repeats=False)


    return rfdf

def get_counts_by_datakey(stim_overlaps):
    c_list=[]
    i=0
    for (visual_area, datakey), g in stim_overlaps.groupby(['visual_area', 'datakey']):
        rf_rids = sorted(g['cell'].unique())
        c_list.append(pd.DataFrame({'visual_area': visual_area, 'datakey': datakey, 
                                     'n_cells': len(rf_rids)}, index=[i]))
        i+=1
    counts_by_dset = pd.concat(c_list, axis=0)
    counts_by_dset['session'] = [d.split('_')[0] for d in counts_by_dset['datakey']]
    counts_by_dset['animalid'] = [d.split('_')[1] for d in counts_by_dset['datakey']]
    counts_by_dset['fovnum'] = [int(d.split('_')[2][3:]) for d in counts_by_dset['datakey']]

    return counts_by_dset



#
# -------- Blobs -----------------------------------------------------------
def aggr_cells_blobs(assigned_cells, traceid='traces001', response_type='dff',
                   responsive_test='nstds', responsive_thr=10.0):
    d_list=[]
    for (visual_area, animalid, session, fovnum, fov, datakey), g in \
        assigned_cells.groupby(['visual_area', 'animalid', 'session', 'fovnum', 'fov', 'datakey']):

        roi_list, nrois_total = util.get_responsive_cells(animalid, session, fov, traceid=traceid, 
                                                         run='blobs',
                                                         response_type=response_type,
                                                         responsive_test=responsive_test, 
                                                         responsive_thr=responsive_thr)
        curr_rois = [r for r in roi_list if r in g['cell'].unique()]
        df_ = pd.DataFrame({'cell': curr_rois})
        metadict = {'visual_area': visual_area, 'datakey': datakey}
        df_ = add_meta_to_df(df_, metadict)
        d_list.append(df_)

    allthedata = pd.concat(d_list, axis=0).reset_index(drop=True)
    
    return allthedata

def get_dsets_with_most_blobs(edata, assigned_cells, traceid='traces001', response_type='dff',
                                responsive_test='nstds', responsive_thr=10.0):
    
    #edata = sdata[sdata['experiment']=='gratings'].copy()
    exp_cells = pd.concat([g for (visual_area, datakey), g \
                                in assigned_cells.groupby(['visual_area', 'datakey']) \
                                if datakey in edata['datakey'].values])

    # Load all gratings data
    allthedata = aggr_cells_blobs(exp_cells, traceid=traceid, response_type=response_type, 
                       responsive_test=responsive_test, responsive_thr=responsive_thr)

    best_dfs = get_dsets_with_most_cells(allthedata) #, exp_cells) 
    
    return best_dfs




# -------- gratings -----------------------------------------------------------
def aggr_gratings_fits(assigned_cells, traceid='traces001', response_type='dff', 
                       responsive_test='nstds', responsive_thr=10.,
                       n_bootstrap_iters=1000, n_resamples=20, 
                       return_missing=True, rootdir='/n/coxfs01/2p-data'):
    '''
    assigned_cells:  dataframe w/ assigned cells of dsets that have gratings
    '''
    from pipeline.python.classifications import bootstrap_osi as osi

    no_fits = []
    g_ = []
    i = 0
    for (visual_area, datakey, animalid, session, fov), g in assigned_cells.groupby(['visual_area', 'datakey', 'animalid', 'session', 'fov']):
        try:
            # Get all osi results for current dataset
            exp = util.Gratings(animalid, session, fov, traceid=traceid, rootdir=rootdir)
            bootresults_tmp, fitparams = exp.get_tuning(response_type=response_type,
                                                   responsive_test=responsive_test,
                                                   responsive_thr=responsive_thr,
                                                   n_bootstrap_iters=n_bootstrap_iters,
                                                   n_resamples=n_resamples, verbose=False)
            # Get OSI results
            currcells = g['cell'].unique()
            bootresults = dict((k, v) for k, v in bootresults_tmp.items() if k in currcells)
        except Exception as e:
            print(e)
            print('ERROR: %s' % datakey)
            continue
            
        # Get fits
        rmetrics, rmetrics_by_cfg = osi.get_good_fits(bootresults, fitparams, gof_thr=None, verbose=False)
        if rmetrics is None:
            no_fits.append('%s_%s' % (visual_area, datakey))
            continue

        meandf = rmetrics.copy()
        metainfo = {'visual_area': visual_area, 'animalid': animalid, 
                    'session': session, 'fov': fov, 'datakey': datakey}
        meandf = putils.add_meta_to_df(meandf, metainfo)
        g_.append(meandf)
        i += 1
    gdata = pd.concat(g_, axis=0)

    if verbose:
        print("Datasets with NO fits found:")
        for s in no_fits:
            print(s)

    if return_missing:
        return gdata, no_fits

def get_dsets_with_most_gratings(edata, assigned_cells, traceid='traces001', response_type='dff',
                                responsive_test='nstds', responsive_thr=10.0, 
                                n_bootstrap_iters=1000, n_resamples=20):
    
    #edata = sdata[sdata['experiment']=='gratings'].copy()
    exp_cells = pd.concat([g for (visual_area, datakey), g \
                                in assigned_cells.groupby(['visual_area', 'datakey']) \
                                if datakey in edata['datakey'].values])

    # Load all gratings data
    gdata, missing_ = aggr_gratings_fits(exp_cells, traceid=traceid, response_type=response_type, 
                       responsive_test=responsive_test, responsive_thr=responsive_thr,
                       n_bootstrap_iters=n_bootstrap_iters, n_resamples=n_resamples)

    best_dfs = get_dsets_with_most_cells(gdata) #, exp_cells) 
    
    return best_dfs


# =============================================================
# Resample data, specify distribution
# =============================================================

def match_neuraldata_distn(NEURALDATA, src='Li'):
    '''
    Resample data to match distribution of max response values of a given visual area
    This should only be used for POOLED analyses (by_ncells, for ex.)
    '''
    # Get parameters for src distn to mimic
    min_ncells = int(NEURALDATA[['visual_area', 'datakey', 'cell']]
                     .drop_duplicates().groupby(['visual_area']).count().min()[0])
    max_ndata, dist_mean, dist_sigma = get_params_for_source_distn(NEURALDATA, src=src)

    # Select new cells that match these params
    selected_cells = match_source_distn(NEURALDATA, src=src, n_samples=min_ncells)
    print(selected_cells[['visual_area', 'datakey', 'cell']]
            .drop_duplicates().groupby(['visual_area']).count())

    N2 = pd.concat([NEURALDATA[(NEURALDATA['visual_area']==visual_area) 
                      & (NEURALDATA['datakey']==datakey) 
                      & (NEURALDATA['cell'].isin(g['cell'].unique()))].copy() 
                    for (visual_area, datakey), g \
                    in selected_cells.groupby(['visual_area', 'datakey'])], axis=0)

    return N2, selected_cells


def select_dataframe_subset(selected_cells, RFDATA):
    R2 = pd.concat([RFDATA[(RFDATA['visual_area']==visual_area) 
                      & (RFDATA['datakey']==datakey) 
                      & (RFDATA['cell'].isin(g['cell'].unique()))].copy() 
                    for (visual_area, datakey), g \
                    in selected_cells.groupby(['visual_area', 'datakey'])], axis=0)
    return R2

def match_distns_neuraldata_and_rfdata(NEURALDATA, RFDATA, src='Li'):
    N2, selected_cells = match_neuraldata_distn(NEURALDATA, src=src)
    R2 = select_dataframe_subset(selected_cells, RFDATA)
    return N2, R2


def match_source_distn(NDATA, src='Li', n_samples=100):
    '''
    NDATA (dataframe)
        Stacked data for trial metrics (neuraldfs)
    src (str)
        Source distribution to model.
    curr_ncells (int)
        Sample size to return
    '''
    # Get response to best config (max repsonse)
    max_ndata, dist_mean, dist_sigma = get_params_for_source_distn(NDATA, src=src)
    # Ignore the src area (i.e., don't resample)
    areas_to_resample = [v for v in max_ndata['visual_area'].unique() if v!=src]
    # Select cells that match src distn
    selected_cells = generate_matched_distn(max_ndata, visual_areas=areas_to_resample, 
                                            mean=dist_mean, sigma=dist_sigma, n_samples=n_samples)

    return selected_cells

def get_params_for_source_distn(NDATA, src='Li'):
    '''
    Get each cell's max response, return mean and sigma for creating distn.
    '''
    max_ndata = group_cells_by_max_response(NDATA)

    # Get mean and std of distn
    dist_mean = max_ndata.groupby(['visual_area']).mean().loc[src]['response']
    dist_sigma = max_ndata.groupby(['visual_area']).std().loc[src]['response']

    return max_ndata, dist_mean, dist_sigma

def group_cells_by_max_response(NDATA):
    '''
    NDATA (dataframe)
        Stacked trial responses 
    '''
    # For each cell, get activity profile (averaged across trial reps)
    mean_ndata = NDATA.groupby(['visual_area', 'datakey', 'cell', 'config']).mean().reset_index()

    # For each cell, get MAX across configs
    max_ndata = mean_ndata.groupby(['visual_area', 'datakey', 'cell']).max().reset_index()

    return max_ndata

def get_cell_dataframe(NDATA):
    '''
    NDATA (dataframe)
        Stacked trial responses 
    '''
    # For each cell, get activity profile (averaged across trial reps)
    mean_ndata = NDATA.groupby(['visual_area', 'datakey', 'cell', 'config']).mean().reset_index()

    # For each cell, get MAX across configs
    max_ndata = mean_ndata.groupby(['visual_area', 'datakey', 'cell']).max().reset_index()

    return max_ndata


def generate_matched_distn(max_ndata, visual_areas=None, mean=None, sigma=None, n_samples=None):
    '''
    Return cell list (as dataframe) for best matched responses.
    '''
    # General new neuraldf 
    src_dist = np.random.normal(loc=mean, scale=sigma, size=n_samples)
    selected_cells = []
    if visual_areas is None:
        visual_areas = max_ndata['visual_area'].unique()
    for visual_area, vdf in max_ndata.groupby(['visual_area']):
        xd = vdf.sort_values(by='response').copy()
        if visual_area not in visual_areas:
            selected_cells.append(xd)
            continue
        ix=[]
        match_ixs = take_closest_index_no_repeats(src_dist, xd['response'].values)
        #xd.iloc[match_ixs]
        selected_cells.append(xd.iloc[match_ixs])
    selected_df = pd.concat(selected_cells, axis=0)
 
    return selected_df
    
def take_closest_index_no_repeats(list1, list2):
    match_ixs=[]
    orig_list2 = copy.copy(list2)
    for i in range(len(list1)):
        if (len(list2)) >= 1: #When there are elements in list2

            temp_result = abs(list1[i] - list2) #Matrix subtraction

            min_val = np.amin(temp_result) #Getting the minimum value to get closest element
            min_val_index = np.where(temp_result == min_val) #To find index of minimum value
            closest_element = list2[min_val_index] #Actual value of closest element in list2
            list2 = list2[list2 != closest_element] #Remove closest element after found

            #print(i, list1[i], min_val_index[0][0], closest_element[0]) 
            #List1 Index, Element to find, List2 Index, Closest Element
            match_ixs.append(np.where(orig_list2==closest_element[0])[0][0]) #min_val_index[0][0])

        else: #All elements are already found

            print(i, list1[i], 'No further closest unique closest elements found in list2')

    return match_ixs

# =============================================================
# Select and subsample cells
# =============================================================
def select_cells(cells, visual_area=None, datakey=None):
    if visual_area is not None:
        if datakey is not None:
            currcells = cells[(cells['visual_area']==visual_area)
                            & (cells['datakey']==datakey)].copy()
        else:
            currcells = cells[(cells['visual_area']==visual_area)].copy()
    else:
        currcells = cells.copy()

    return currcells

def assign_global_cell_id(cells):
    cells['global_ix'] = 0
    for v, g in cells.groupby(['visual_area']):
        cells['global_ix'].loc[g.index] = np.arange(0, g.shape[0])
    return cells.reset_index(drop=True)

def global_cells(cells, remove_too_few=True, min_ncells=5,  return_counts=False):
    '''
    cells - dataframe, each row is a cell, has datakey/visual_area fields

    Returns:
    
    roidf (dataframe)
        Globally-indexed rois ('dset_roi' = roi ID in dataset, 'roi': global index)
    
    roi_counters (dict)
        Counts of cells by area (optional)

    '''
    visual_areas=cells['visual_area'].unique() #['V1', 'Lm', 'Li']
    print("Assigned visual areas: %s" % str(visual_areas))
 
    incl_keys = []
    if remove_too_few:
        for (v, k), g in cells.groupby(['visual_area', 'datakey']):
            if len(g['cell'].unique()) < min_ncells:
                continue
            incl_keys.append(k) 
    else:
        incl_keys = cells['datakey'].unique()
 
    nocells=[]; notrials=[];
    roi_counters = dict((v, 0) for v in visual_areas)
    #count_of_sel = cells[['visual_area', 'datakey', 'cell']].drop_duplicates().groupby(['visual_area', 'datakey']).count().reset_index()
    #print(count_of_sel.groupby(['visual_area']).sum())

    roidf = []
    for (visual_area, datakey), g in cells[cells['datakey'].isin(incl_keys)].groupby(['visual_area', 'datakey']):

        roi_counter = roi_counters[visual_area]

        # Reindex roi ids for global
        roi_list = sorted(g['cell'].unique()) #[int(r) for r in ddf.columns if r != 'config']
        nrs = len(roi_list)
        roi_ids = [i+roi_counter for i, r in enumerate(roi_list)]
      
        # Append to full df
        roi_dict = {'roi': roi_ids,
                   'dset_roi': roi_list,
                   'visual_area': [visual_area for _ in np.arange(0, nrs)],
                   'datakey': [datakey for _ in np.arange(0, nrs)]}
        if 'global_ix' in g.columns:
            roi_dict.update({'global_ix': g['global_ix'].values})

        roidf.append(pd.DataFrame(roi_dict))
      
        # Update global roi id counter
        roi_counters[visual_area] += len(roi_ids)

    if len(roidf)==0:
        if return_counts:
            return None, None
        else:
            return None

    roidf = pd.concat(roidf, axis=0).reset_index(drop=True)  #.groupby(['visual_area']).count()
    #for k, v in global_rois.items():
    #    print(k, len(v))
      
    roidf['animalid'] = [d.split('_')[1] for d in roidf['datakey']]
    roidf['session'] = [d.split('_')[0] for d in roidf['datakey']]
    roidf['fovnum'] = [int(d.split('_')[2][3:]) for d in roidf['datakey']]
   
    if return_counts:
        return roidf, roi_counters
    else:
        return roidf

def get_pooled_cells(stim_overlaps, assigned_cells=None, remove_too_few=False, 
                      overlap_thr=0.8, min_ncells=20, visual_areas=None, return_counts=True):
    '''
    stim_overlaps (dataframe)
        Dataframe of all cell IDs and overlap values for all dkeys and visual areas.
    cells - dataframe, each row is a cell, has datakey/visual_area fields

    Returns:
    
    roidf (dataframe)
        Globally-indexed rois ('dset_roi' = roi ID in dataset, 'roi': global index)
        All included cells (responsive, pass overlap_thr, exclude dsets w/ too_few cells)
   
    roi_counters (dict)
        Counts of cells by area (optional)
    '''
    incl_keys = []
    if remove_too_few:
        for (v, k), g in stim_overlaps.groupby(['visual_area', 'datakey']):
            if len(g['cell'].unique()) < min_ncells:
                continue
            incl_keys.append(k) 
    else:
        incl_keys = stim_overlaps['datakey'].unique()

    if visual_areas is None:
        visual_areas = stim_overlaps['visual_area'].unique()

    # Filter out cells that dont pass overlap threshold
    filtered_ = filter_cells_by_overlap(stim_overlaps[stim_overlaps['datakey'].isin(incl_keys)],
                                overlap_thr=overlap_thr,  visual_areas=visual_areas)
    if assigned_cells is not None:
        updated_ = cells_in_experiment_df(assigned_cells, filtered_)
    else:
        updated_  = filtered_.copy()

    globalcells, cellcounts = global_cells(updated_, remove_too_few=remove_too_few, 
                                min_ncells=min_ncells, return_counts=True)

    #globalcells, cellcounts = filter_rois(stim_overlaps[stim_overlaps['datakey'].isin(incl_keys)], 
    #                                    overlap_thr=overlap_thr, return_counts=True, visual_areas=visual_areas)

    if return_counts:
        return globalcells, cellcounts
    else:
        return globalcells

def filter_cells_by_overlap(stim_overlaps,overlap_thr=0.5, visual_areas=None):
    '''
    Only get cells that pass overlap_thr of some value.
    '''
    # visual_areas=['V1', 'Lm', 'Li']
    if visual_areas is None:
        visual_areas = stim_overlaps['visual_area'].unique()
 
    nocells=[]; notrials=[];
    roi_counters = dict((v, 0) for v in visual_areas)
    
    pass_overlaps = stim_overlaps[stim_overlaps['perc_overlap']>=overlap_thr].copy()
    n_orig = len(stim_overlaps['cell'].unique())
    n_pass = len(pass_overlaps['cell'].unique())
    print("%i of %i cells pass overlap (thr=%.2f)" % (n_pass, n_orig, overlap_thr))

    cols = ['visual_area', 'datakey', 'cell']
    if 'global_ix' in pass_overlaps.columns:
        cols.append(['global_ix'])

    filtered_cells = pass_overlaps[cols].drop_duplicates().reset_index(drop=True)

    return filtered_cells

def filter_rois(stim_overlaps, overlap_thr=0.50, return_counts=False, visual_areas=None):
    '''
    Only get cells that pass overlap_thr of some value.
    '''
    # visual_areas=['V1', 'Lm', 'Li']
    if visual_areas is None:
        visual_areas = stim_overlaps['visual_area'].unique()
 
    nocells=[]; notrials=[];
    roi_counters = dict((v, 0) for v in visual_areas)
    
    pass_overlaps = stim_overlaps[stim_overlaps['perc_overlap']>=overlap_thr]
    n_orig = len(stim_overlaps['cell'].unique())
    n_pass = len(pass_overlaps['cell'].unique())
    print("%i of %i cells pass overlap (thr=%.2f)" % (n_pass, n_orig, overlap_thr))

    #count_of_sel = pass_overlaps[['visual_area', 'datakey', 'cell']].drop_duplicates().groupby(['visual_area', 'datakey']).count().reset_index()
    #print(count_of_sel.groupby(['visual_area']).sum())

    roidf = []
    for (visual_area, datakey), g in pass_overlaps.groupby(['visual_area', 'datakey']):

        roi_counter = roi_counters[visual_area]
        # Reindex roi ids for global
        roi_list = sorted(g['cell'].unique()) #[int(r) for r in ddf.columns if r != 'config']
        nrs = len(roi_list)
        roi_ids = [i+roi_counter for i, r in enumerate(roi_list)]
      
        # Append to full df
        roi_dict = {'roi': roi_ids,
                   'dset_roi': roi_list,
                   'visual_area': [visual_area for _ in np.arange(0, nrs)],
                   'datakey': [datakey for _ in np.arange(0, nrs)]}
        if 'global_ix' in g.columns:
            roi_dict.update({'global_ix': g['global_ix'].values})

        roidf.append(pd.DataFrame(roi_dict))
        # Update global roi id counter
        roi_counters[visual_area] += len(roi_ids)

    if len(roidf)==0:
        if return_counts:
            return None, None
        else:
            return None

    roidf = pd.concat(roidf, axis=0) #.groupby(['visual_area']).count()
   
    roidf['animalid'] = [d.split('_')[1] for d in roidf['datakey']]
    roidf['session'] = [d.split('_')[0] for d in roidf['datakey']]
    roidf['fovnum'] = [int(d.split('_')[2][3:]) for d in roidf['datakey']]
   
    if return_counts:
        return roidf, roi_counters
    else:
        return roidf


def get_blobs_and_rf_meta(experiment='blobs', has_gratings=False, stim_filterby=None,
                            traceid='traces001', fov_type='zoom2p0x', state='awake'):
    #### Get metadata for experiment type
    sdata = get_aggregate_info(traceid=traceid, fov_type=fov_type, state=state)
    edata, expmeta = experiment_datakeys(experiment=experiment,
                            has_gratings=has_gratings, stim_filterby=stim_filterby, 
                            has_rfs=True, experiment_only=False)
   # experiment_datakeys(experiment='blobs', has_gratings=False, has_rfs=False, stim_filterby='most_fits',
   #                     experiment_only=True):

 
    return edata 
#


# ===============================================================
# SNR 
# ===============================================================

def load_snr_data(experiment='blobs', traceid='traces001', responsive_test='nstds', 
                 responsive_thr=10.0, trial_epoch='stimulus', 
                 aggregate_dir='/n/coxfs01/julianarhee/aggregate-visual-areas'):
    SNR=None
    fname = 'aggr_%s_trialmeans_%s_%s-thr-%.2f_snr_%s' \
                % (experiment, traceid, responsive_test, responsive_thr, trial_epoch)
    #print(fname)
    create_new_df=False
    outfile = os.path.join(aggregate_dir, 'data-stats', '%s.pkl' % fname)
    try:
        print("... loading SNR (%s)" % fname)
        with open(outfile, 'rb') as f:
            SNR = pkl.load(f)
        print(SNR.shape)
    except Exception as e:
        traceback.print_exc() #print(e)
        #return None
    
    return SNR, outfile


def create_snr_df(RCELLS, experiment='blobs', trial_epoch='stimulus', traceid='traces001', 
                    outfile='/tmp/results.pkl'):
    '''
    RCELLS (df): all assigned and responsive cells for each dset/visual area
    Labels are loaded from experiment_classes.py()
    '''
    from pipeline.python.classifications import test_responsivity as resp
    print("... creating SNR df")
    print("... dst: %s" % outfile)

    d_=[]
    for (visual_area, datakey), g in RCELLS.groupby(['visual_area', 'datakey']):

        # Load experiment data
        session, animalid, fovnum = split_datakey_str(datakey)
        E = util.Objects(animalid, session, 'FOV%i_zoom2p0x' % fovnum, traceid=traceid)
        E.load(trace_type='corrected')

        # Get trial metrics for each roi
        curr_rois = g['cell'].unique().astype(int)
        nframes_on = float(E.data.labels['nframes_on'].unique())
        nframes_post = (nframes_on*0.5) if trial_epoch=='plushalf' else 0.
        
        gdf = resp.group_roidata_stimresponse(E.data.traces, E.data.labels, roi_list=curr_rois, 
                                          nframes_post=nframes_post, return_grouped=False)
        
        gdf['snr'] = gdf['stim_mean'] / gdf['base_std']
        
        df_ = gdf[['snr', 'config', 'cell']]
        df_['visual_area'] = visual_area
        df_['datakey'] = datakey
        d_.append(df_)

    SNR = pd.concat(d_, axis=0).reset_index(drop=True)
    #print(SNR.shape)
    SNR = split_datakey(SNR)

    print("... saving")
    with open(outfile, 'wb') as f:
        pkl.dump(SNR, f, protocol=pkl.HIGHEST_PROTOCOL)
    print("Saved! %s" % outfile)
    
    return SNR
    
    
def get_snr_data(RCELLS, experiment='blobs', traceid='traces001', responsive_test='nstds', 
                 responsive_thr=10.0, trial_epoch='stimulus', create_new=False, rename_configs=True,
                aggregate_dir='/n/coxfs01/julianarhee/aggregate-visual-areas'):
    
    if not create_new:
        SNR, outfile = load_snr_data(experiment=experiment, traceid=traceid, responsive_test=responsive_test,
                                        responsive_thr=responsive_thr, trial_epoch=trial_epoch)
        create_new = SNR is None
        
    if create_new:

        fname = 'aggr_%s_trialmeans_%s_%s-thr-%.2f_snr_%s' \
                    % (experiment, traceid, responsive_test, responsive_thr, trial_epoch)
        outfile = os.path.join(aggregate_dir, 'data-stats', '%s.pkl' % fname)
        SNR = create_snr_df(RCELLS, experiment=experiment, trial_epoch=trial_epoch, traceid=traceid,
                            outfile=outfile)

    if rename_configs:
        SNR, _ = rename_neuraldf_configs(SNR, experiment=experiment, traceid=traceid)
#         stim_datakeys=NEURALDATA['datakey'].unique()
#         SDF, renamed_configs = aggr.check_sdfs(stim_datakeys, experiment='blobs', 
#                                 traceid=traceid, images_only=True, rename=True, return_incorrect=True)
#         for k, curr_lut in renamed_configs.items():
#             new_cfgs = [curr_lut[c] for c in SNR[SNR['datakey']==k]['config']]
#             SNR.loc[SNR['datakey']==k, 'config'] = new_cfgs
    
    # Save mean 
    outdir, outfname = os.path.split(outfile)
    f_id = outfname.split('_trialmeans_')[-1]
    mean_fname = 'snr_by_cell_%s' % f_id
    outf = os.path.join(aggregate_dir, 'data-stats', fname) 
    print(mean_fname)
    mean_snr = SNR.groupby(['visual_area', 'datakey', 'cell', 'config']).mean().reset_index()
    with open(outf, 'wb') as f:
        pkl.dump(mean_snr, f, protocol=pkl.HIGHEST_PROTOCOL)
        
    return SNR

def rename_neuraldf_configs(NEURALDATA, experiment='blobs', traceid='traces001'):

    stim_datakeys = NEURALDATA['datakey'].unique()

    SDF, renamed_configs = check_sdfs(stim_datakeys, experiment=experiment, 
                            traceid=traceid, images_only=True, rename=True, return_incorrect=True)
    
    for k, curr_lut in renamed_configs.items():
        new_cfgs = [curr_lut[c] for c in NEURALDATA[NEURALDATA['datakey']==k]['config']]
        NEURALDATA.loc[NEURALDATA['datakey']==k, 'config'] = new_cfgs

    return NEURALDATA, SDF

def get_mean_snr(experiment='blobs', traceid='traces001', responsive_test='nstds', 
                 responsive_thr=10.0, trial_epoch='stimulus',
                aggregate_dir='/n/coxfs01/julianarhee/aggregate-visual-areas'):
    mean_snr=None
    fname = 'aggr_by_cell_%s_%s_%s-thr-%.2f_snr_%s' \
                % (experiment, traceid, responsive_test, responsive_thr, trial_epoch)
    outf = os.path.join(aggregate_dir, 'data-stats', '%s.pkl' % fname)
    try:
        print("... loading SNR (%s)" % fname)
        with open(outf, 'rb') as f:
            mean_snr = pkl.load(f)
        print(mean_snr.shape)
    except Exception as e:
        print(e)
        return None
    
    return mean_snr

def threshold_cells_by_snr(mean_snr, globalcells, snr_thr=10.0, max_snr_thr=None):
    
    # mean_snr = SNR.groupby(['visual_area', 'datakey', 'cell', 'config']).mean().reset_index()
    if max_snr_thr is not None:
        thresh_snr = mean_snr[(mean_snr['snr']>=snr_thr) & (mean_snr['snr']<=max_snr_thr)]\
                        .groupby(['visual_area', 'datakey', 'cell'])\
                        .mean().reset_index()
    else:
        thresh_snr = mean_snr[mean_snr['snr']>=snr_thr].groupby(['visual_area', 'datakey', 'cell'])\
                    .mean().reset_index()

    # Get global cells that pass threshold
    CELLS = pd.concat([globalcells[(globalcells['visual_area']==visual_area)
                                 & (globalcells['datakey']==datakey)
                                 & (globalcells['dset_roi'].isin(g['cell'].unique()))] \
                       for (visual_area, datakey), g in thresh_snr.groupby(['visual_area', 'datakey'])])
    CELLS['cell'] = CELLS['dset_roi'] # Add 'cell' column to use as 'assigned_cells' df
    
    return CELLS


# ===============================================================
# Plotting
# ===============================================================
from matplotlib.lines import Line2D

def crop_legend_labels(ax, n_hues, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12,title='', n_cols=1):
    # Get the handles and labels.
    leg_handles, leg_labels = ax.get_legend_handles_labels()
    # When creating the legend, only use the first two elements
    leg = ax.legend(leg_handles[0:n_hues], leg_labels[0:n_hues], title=title,
            bbox_to_anchor=bbox_to_anchor, fontsize=fontsize, loc=loc, ncol=n_cols)
    return leg


def plot_pairwise_by_axis(plotdf, curr_metric='abs_coef', c1='az', c2='el', 
                          c1_label=None, c2_label=None,
                          compare_var='cond', fontsize=10, fontcolor='k', fmt='%.2f', xytext=(0, 10),
                          area_colors=None, legend=True, legend_fontsize=8, bbox_to_anchor=(1.5,1.1), ax=None):
    if ax is None:
        fig, ax = pl.subplots(figsize=(5,4), dpi=150)
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
    ax = pairwise_compare_single_metric(plotdf, curr_metric=curr_metric, ax=ax,
                                        c1=c1, c2=c2, compare_var=compare_var, area_colors=area_colors)
    plotdf.apply(annotateBars, ax=ax, axis=1, fontsize=fontsize, 
                    fontcolor=fontcolor, fmt=fmt, xytext=xytext) 

    # Set x labels
    if c1_label is None:
        c1_label=c1
    if c2_label is None:
        c2_label=c2
    set_split_xlabels(ax, a_label=c1_label, b_label=c2_label)

    if legend:
        # Get counts of samples for legend
        legend_elements = get_counts_for_legend(plotdf, area_colors=area_colors, 
                                                markersize=10, marker='_')
        ax.legend(handles=legend_elements, bbox_to_anchor=bbox_to_anchor, fontsize=legend_fontsize)

    return ax #fig

def paired_ttests(comdf, curr_metric='avg_size', 
                c1='rfs', c2='rfs10', compare_var='experiment',
                visual_areas=['V1', 'Lm', 'Li']):
    r_=[]
    for ai, visual_area in enumerate(visual_areas):

        plotdf = comdf[comdf['visual_area']==visual_area].copy()
        a_vals = plotdf[plotdf[compare_var]==c1].sort_values(by='datakey')[curr_metric].values
        b_vals = plotdf[plotdf[compare_var]==c2].sort_values(by='datakey')[curr_metric].values
        #print(a_vals, b_vals)

        tstat, pval = spstats.ttest_rel(np.array(a_vals), np.array(b_vals))
     
        print('%s: %.2f (p=%.2f)' % (visual_area, tstat, pval))
        
        res = pd.DataFrame({'visual_area': visual_area, 't_stat': tstat, 'p_val': pval}, 
                        index=[ai])
        r_.append(res)
    
    statdf = pd.concat(r_, axis=0)

    return statdf


def plot_paired(plotdf, aix=0, curr_metric='avg_size', ax=None,
                c1='rfs', c2='rfs10', compare_var='experiment',
                marker='o', offset=0.25, color='k', label=None, lw=0.5, alpha=1, return_vals=False):

    if ax is None:
        fig, ax = pl.subplots()
        
    a_vals = plotdf[plotdf[compare_var]==c1].sort_values(by='datakey')[curr_metric].values
    b_vals = plotdf[plotdf[compare_var]==c2].sort_values(by='datakey')[curr_metric].values

    by_exp = [(a, e) for a, e in zip(a_vals, b_vals)]
    for pi, p in enumerate(by_exp):
        ax.plot([aix-offset, aix+offset], p, marker=marker, color=color,
                alpha=alpha, lw=lw,  zorder=0, markerfacecolor=None,
                markeredgecolor=color, label=label)
    if return_vals:
        return ax, a_vals, b_vals

    else:
        tstat, pval = spstats.ttest_rel(a_vals, b_vals)
        print("(t-stat:%.2f, p=%.2f)" % (tstat, pval))
        
        return ax


def pairwise_compare_single_metric(comdf, curr_metric='avg_size', 
                                    c1='rfs', c2='rfs10', compare_var='experiment',
                                    ax=None, marker='o', visual_areas=['V1', 'Lm', 'Li'],
                                    area_colors=None):
    assert 'datakey' in comdf.columns, "Need a sorter, 'datakey' not found."

    if area_colors is None:
        visual_areas = ['V1', 'Lm', 'Li']
        colors = ['magenta', 'orange', 'dodgerblue'] 
        #sns.color_palette(palette='colorblind') #, n_colors=3)
        area_colors = {'V1': colors[0], 'Lm': colors[1], 'Li': colors[2]}
    offset = 0.25 
    if ax is None:
        fig, ax = pl.subplots(figsize=(5,4), dpi=150)
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
    
    # Plot paired values
    aix=0
    for ai, visual_area in enumerate(visual_areas):

        plotdf = comdf[comdf['visual_area']==visual_area]
        ax = plot_paired(plotdf, aix=aix, curr_metric=curr_metric, ax=ax,
                        c1=c1, c2=c2, compare_var=compare_var, offset=offset,
                        marker=marker, color=area_colors[visual_area], lw=0.5)

#        a_vals = plotdf[plotdf[compare_var]==c1].sort_values(by='datakey')[curr_metric].values
#        b_vals = plotdf[plotdf[compare_var]==c2].sort_values(by='datakey')[curr_metric].values
#
#        by_exp = [(a, e) for a, e in zip(a_vals, b_vals)]
#        for pi, p in enumerate(by_exp):
#            ax.plot([aix-offset, aix+offset], p, marker=marker, color=area_colors[visual_area], 
#                    alpha=1, lw=0.5,  zorder=0, markerfacecolor=None, 
#                    markeredgecolor=area_colors[visual_area])
#        tstat, pval = spstats.ttest_rel(a_vals, b_vals)
#        print("%s: (t-stat:%.2f, p=%.2f)" % (visual_area, tstat, pval))
        aix = aix+1

    # Plot average
    sns.barplot("visual_area", curr_metric, data=comdf, 
                hue=compare_var, hue_order=[c1, c2], #zorder=0,
                ax=ax, order=visual_areas,
                errcolor="k", edgecolor=('k', 'k', 'k'), facecolor=(1,1,1,0), linewidth=2.5)
    ax.legend_.remove()

    set_split_xlabels(ax, a_label=c1, b_label=c2)
    
    return ax

def set_split_xlabels(ax, offset=0.25, a_label='rfs', b_label='rfs10', rotation=0, ha='center', ncols=3):
    locs = []
    labs = []
    for li in np.arange(0, ncols):
        locs.extend([li-offset, li+offset])
        labs.extend([a_label, b_label])
    ax.set_xticks(locs)
    ax.set_xticklabels(labs, rotation=rotation, ha=ha)

#    ax.set_xticks([0-offset, 0+offset, 1-offset, 1+offset, 2-offset, 2+offset])
#    ax.set_xticklabels([a_label, b_label, a_label, b_label, a_label, b_label], 
#                        rotation=rotation, ha=ha)
#    ax.set_xlabel('')
    ax.tick_params(axis='x', size=0)
    sns.despine(bottom=True, offset=4)
    return ax



def annotateBars(row, ax, fontsize=12, fmt='%.2f', fontcolor='k', xytext=(0, 10)): 
    for p in ax.patches:
        ax.annotate(fmt % p.get_height(), (p.get_x() + p.get_width() / 2., 0.), #p.get_height()),
                    ha='center', va='center', fontsize=fontsize, color=fontcolor, 
                    rotation=0, xytext=xytext, #(0, 10),
             textcoords='offset points')
        
    return None

def plot_mannwhitney(mdf, metric='I_rs', multi_comp_test='holm', 
                        ax=None, y_loc=None, offset=0.1):
    if ax is None:
        fig, ax = pl.subplots()

    print("********* [%s] Mann-Whitney U test(mc=%s) **********" % (metric, multi_comp_test))
    statresults = do_mannwhitney(mdf, metric=metric, multi_comp_test=multi_comp_test)
    #print(statresults)
    
    # stats significance
    ax = annotate_stats_areas(statresults, ax, y_loc=y_loc, offset=offset)
    print("****************************")
    
    return statresults, ax


def do_mannwhitney(mdf, metric='I_rs', multi_comp_test='holm'):
    '''
    bonferroni : one-step correction

    sidak : one-step correction

    holm-sidak : step down method using Sidak adjustments

    holm : step-down method using Bonferroni adjustments

    simes-hochberg : step-up method (independent)

    hommel : closed method based on Simes tests (non-negative)

    fdr_bh : Benjamini/Hochberg (non-negative)

    fdr_by : Benjamini/Yekutieli (negative)

    fdr_tsbh : two stage fdr correction (non-negative)

    fdr_tsbky : two stage fdr correction (non-negative)
    '''
    visual_areas = ['V1', 'Lm', 'Li']
    mpairs = list(itertools.combinations(visual_areas, 2))

    pvalues = []
    for mp in mpairs:
        d1 = mdf[mdf['visual_area']==mp[0]][metric]
        d2 = mdf[mdf['visual_area']==mp[1]][metric]

        # compare samples
        stat, p = spstats.mannwhitneyu(d1, d2)
        # interpret
        alpha = 0.05
        if p > alpha:
            interp_str = '... Same distribution (fail to reject H0)'
        else:
            interp_str = '... Different distribution (reject H0)'
        # print('[%s] Statistics=%.3f, p=%.3f, %s' % (str(mp), stat, p, interp_str))

        pvalues.append(p)

    reject, pvals_corrected, _, _ = sm.stats.multitest.multipletests(pvalues, 
                                                                     alpha=0.05, 
                                                                     method=multi_comp_test)
    results = []
    for mp, rej, pv in zip(mpairs, reject, pvals_corrected):
        results.append((mp, rej, pv))
        print('[%s] p=%.3f (%s), reject H0=%s' % (str(mp), pv, multi_comp_test, rej))

    return results


def annotate_stats_areas(statresults, ax, lw=1, color='k', 
                        y_loc=None, offset=0.1, 
                         visual_areas=['V1', 'Lm', 'Li']):
   
    if y_loc is None:
        y_loc = round(ax.get_ylim()[-1], 1)*1.2
        #print(y_ht)
        offset = y_loc*offset #0.1

    for ci, cpair in enumerate(statresults):
        if cpair[1]:
            v1, v2 = cpair[0]
            x1 = visual_areas.index(v1)
            x2 = visual_areas.index(v2)
            y1 = y_loc+(ci*offset)
            y2 = y1
            ax.plot([x1,x1, x2, x2], [y1, y2, y2, y1], linewidth=lw, color=color)
            ctrx = x1 + (x2-x1)/2.
            star_str = '**' if cpair[2]<0.01 else '*'
            ax.text(ctrx, y1+(offset/8.), star_str)

    return ax

def get_counts_for_legend(df, area_colors=None, markersize=10, marker='_', lw=1,
              visual_areas=['V1', 'Lm', 'Li']):
    from matplotlib.lines import Line2D

    if area_colors is None:
        colors = ['magenta', 'orange', 'dodgerblue'] #sns.color_palette(palette='colorblind') #, n_colors=3)
        area_colors = {'V1': colors[0], 'Lm': colors[1], 'Li': colors[2]}
    visual_areas = area_colors.keys()

    dkey_name = 'retinokey' if 'datakey' not in df.columns else 'datakey'

    if 'animalid' not in df.columns:
        df['animalid'] = [s.split('_')[1] for s in df[dkey_name]]
        df['session'] = [s.split('_')[0] for s in df[dkey_name]]

    # Get counts
    if 'cell' in df.columns or 'roi' in df.columns:
        roistr = 'cell' if 'cell' in df.columns else 'roi'
        counts = df.groupby(['visual_area', 'animalid', dkey_name])[roistr].count().reset_index()
        counts.rename(columns={roistr: 'n_cells'}, inplace=True)
    elif 'n_cells' in df.columns or 'ncells' in df.columns:
        roi_str = 'n_cells' if 'n_cells' in df.columns else 'ncells'
        counts = df.groupby(['visual_area']).mean().reset_index()[['visual_area', roi_str]]
    else:
        counts = df.groupby(['visual_area', 'animalid', dkey_name]).count().reset_index()

    # Get counts of samples for legend
    n_rats = dict((v, len(g['animalid'].unique())) \
                        for v, g in df.groupby(['visual_area']))
    n_fovs = dict((v, len(g[[dkey_name]].drop_duplicates())) \
                        for v, g in df.groupby(['visual_area']))
    for v in area_colors.keys():
        if v not in n_rats.keys():
            n_rats.update({v: 0})
        if v not in n_fovs.keys():
            n_fovs.update({v: 0})
    if 'n_cells' in counts.columns:
        n_cells = dict((v, g['n_cells'].sum()) \
                        for v, g in counts.groupby(['visual_area']))
        legend_elements = [Line2D([0], [0], marker=marker, markersize=markersize, \
                                  lw=lw, color=area_colors[v], 
                                  markerfacecolor=area_colors[v],
                                  label='%s (n=%i rats, %i fovs, avg %i cells)' % (v, n_rats[v], n_fovs[v], n_cells[v]))\
                           for v in visual_areas]
    else:
        legend_elements = [Line2D([0], [0], marker=marker, markersize=markersize, \
                                  lw=lw, color=area_colors[v], 
                                  markerfacecolor=area_colors[v],
                                  label='%s (n=%i rats, %i fovs)' % (v, n_rats[v], n_fovs[v]))\
                           for v in visual_areas]

        
    return legend_elements

# ===============================================================
# Screen info 
# ===============================================================
def get_stim_info(animalid, session, fov):
    S = util.Session(animalid, session, fov) #'FOV%i_zoom2p0x' % fovnum)
    xpos, ypos = S.get_stimulus_coordinates()

    screenleft, screenright = S.screen['linminW'], S.screen['linmaxW']
    screenbottom, screentop = S.screen['linminH'], S.screen['linmaxH']
    screenaspect = S.screen['resolution'][0] / S.screen['resolution'][1]
    
    screen_width_deg = S.screen['linmaxW']*2.
    screen_height_deg = S.screen['linmaxH']*2.

    pix_per_degW = S.screen['resolution'][0] / screen_width_deg
    pix_per_degH = S.screen['resolution'][1] / screen_height_deg 

    #print(pix_per_degW, pix_per_degH)
    pix_per_deg = np.mean([pix_per_degW, pix_per_degH])
    print("avg pix/deg: %.2f" % pix_per_deg)

    stiminfo = {#'screen_bounds': [screenbottom, screenleft, screentop, screenright],
                'screen_aspect': screenaspect,
                'pix_per_deg': pix_per_deg,
                'stimulus_xpos': xpos,
                'stimulus_ypos': ypos,
                'screen_left': -1*screen_width_deg, #screenleft,
                  'screen_right': screen_width_deg, #screenright,
                  'screen_top': screen_height_deg, #screentop,
                  'screen_bottom': -1*screen_height_deg, #screenbottom,
                  'screen_xres': S.screen['resolution'][0],
                  'screen_yres': S.screen['resolution'][1]}

    return stiminfo

def get_aggregate_stimulation_info(expdf):
    s_list = []
    i=0
    for (visual_area, animalid, session, fovnum), tmpd in expdf.groupby(['visual_area', 'animalid', 'session', 'fovnum']):
        datakey = '_'.join([session, animalid, 'fov%i' % fovnum])
        fov = 'FOV%i_zoom2p0x' % fovnum
        stiminfo = get_stim_info(animalid, session, fov) 
        s_ = pd.DataFrame(stiminfo, index=[i])
        metadict={'visual_area': visual_area, 'animalid': animalid, 
                  'session': session, 'fovnum': fovnum, 'datakey': datakey}
        s_ = add_meta_to_df(s_, metadict)
        s_list.append(s_)
        i+=1
    screeninfo = pd.concat(s_list, axis=0)
    return screeninfo


# ===============================================================
# Data loading
# ===============================================================

def get_trial_alignment(animalid, session, fovnum, curr_exp, traceid='traces001',
        rootdir='/n/coxfs01/2p-data'):
    try:
        extraction_files = sorted(glob.glob(os.path.join(rootdir, animalid, session, 'FOV%i*' % fovnum, 
                                '*%s*' % curr_exp, 'traces', 
                                '%s*' % traceid,'event_alignment.json')), key=natural_keys) 
            # 'extraction_params.json'))
        assert len(extraction_files) > 0, "(%s|%s|fov%i) No extraction info found..." % (animalid, session, fovnum)
    except AssertionError:
#        extraction_files = sorted(glob.glob(os.path.join(rootdir, animalid, session, 'FOV%i*' % fovnum, 
#                                '*%s*' % curr_exp, 'traces', 
#                                '%s*' % traceid,'extraction_params.json')), key=natural_keys) 
        return None
    
    for i, ifile in enumerate(extraction_files):
        with open(ifile, 'r') as f:
            info = json.load(f)
        if i==0:
            infodict = dict((k, [v]) for k, v in info.items() if isnumber(v)) 
        else:
            for k, v in info.items():
                if isnumber(v): 
                    infodict[k].append(v)
    try: 
        dkey = '%s_%s_fov%i' % (session, animalid, fovnum)
        for k, v in infodict.items():
            nvs = np.unique(v)
            assert len(nvs)==1, "%s: more than 1 value found: (%s, %s)" % (dkey, k, str(nvs))
            infodict[k] = np.unique(v)[0]
    except AssertionError:
        return -1

    return infodict

            
def aggregate_alignment_info(edata, experiment='blobs', traceid='traces001'):
    i=0
    d_=[]
    for (visual_area, animalid, session, fovnum, datakey), g in edata[edata['experiment']==experiment].groupby(['visual_area', 'animalid', 'session', 'fovnum', 'datakey']):

        # Alignment info
        alignment_info = get_trial_alignment(animalid, session, fovnum, experiment, traceid=traceid)
        if alignment_info==-1:
            print("Realign: %s" % datakey)
            continue
        iti_pre_ms = float(alignment_info['iti_pre'])*1000
        iti_post_ms = float(alignment_info['iti_post'])*1000
        #print("ITI pre/post: %.1f ms, %.1f ms" % (iti_pre_ms, iti_post_ms))
        d_.append(pd.DataFrame({'visual_area': visual_area, 
                                 'iti_pre': float(alignment_info['iti_pre']),
                                 'iti_post': float(alignment_info['iti_post']),
                                 'stim_dur': float(alignment_info['stim_on_sec']),
                                 'datakey': datakey, 
                                'animalid': animalid, 'session': session, 
                                'fovnum': fovnum}, index=[i]))
        i+=1
    A = pd.concat(d_, axis=0).reset_index(drop=True)  

    return A


def load_traces(animalid, session, fovnum, curr_exp, traceid='traces001',
                response_type='dff', 
                responsive_test='ROC', responsive_thr=0.05, n_stds=2.5,
                redo_stats=False, n_processes=1):
    '''
    redo_stats: use carefully, will re-run responsivity test if True
   
    To return ALL selected cells, set responsive_test to None
    '''

    # Load experiment neural data
    fov = 'FOV%i_zoom2p0x' % fovnum
    if curr_exp=='blobs':
        exp = util.Objects(animalid, session, fov, traceid=traceid)
    else:
        exp = util.Gratings(animalid, session, fov, traceid=traceid) 
    exp.load(trace_type=response_type, update_self=True, make_equal=False)    
    labels = exp.data.labels.copy()

    # Get stimulus config info
    sdf = exp.data.sdf
    if curr_exp == 'blobs':
        sdf = reformat_morph_values(sdf)

    # Get responsive cells
    if responsive_test is not None:
        responsive_cells, ncells_total = exp.get_responsive_cells(
                                                #response_type=response_type,\
                                                responsive_test=responsive_test,
                                                responsive_thr=responsive_thr,
                                                create_new=redo_stats, 
                                                n_processes=n_processes)
        #print("%i responsive" % len(responsive_cells))
        if responsive_cells is None:
            print("NO LOADING")
            return None, None, None
        traces = exp.data.traces[responsive_cells]
    else:
        traces = exp.data.traces

    return traces, labels, sdf


def traces_to_trials(traces, labels, epoch='stimulus', metric='mean', n_on=None):
    '''
    Returns dataframe w/ columns = roi ids, rows = mean response to stim ON per trial
    Last column is config on given trial.
    '''
    s_on = int(labels['stim_on_frame'].mean())
    if epoch=='stimulus':
        n_on = int(labels['nframes_on'].mean()) 
    elif epoch=='firsthalf':
        n_on = int(labels['nframes_on'].mean()/2.)
    elif epoch=='plushalf':
        half_dur = labels['nframes_on'].mean()/2.
        n_on = int(labels['nframes_on'].mean() + half_dur) 

    roi_list = traces.columns.tolist()
    trial_list = np.array([int(trial[5:]) for trial, g in labels.groupby(['trial'])])
    if epoch in ['stimulus', 'firsthalf', 'plushalf']:
        mean_responses = pd.DataFrame(np.vstack([np.nanmean(traces.iloc[g.index[s_on:s_on+n_on]], axis=0)\
                                            for trial, g in labels.groupby(['trial'])]),
                                             columns=roi_list, index=trial_list)
        if metric=='zscore':
            std_responses = pd.DataFrame(np.vstack([np.nanstd(traces.iloc[g.index[0:s_on]], axis=0)\
                                            for trial, g in labels.groupby(['trial'])]),
                                             columns=roi_list, index=trial_list)
            tmp = mean_responses.divide(std_baseline)
            mean_responses = tmp.copy()
    elif epoch == 'baseline':
        mean_responses = pd.DataFrame(np.vstack([np.nanmean(traces.iloc[g.index[0:s_on]], axis=0)\
                                            for trial, g in labels.groupby(['trial'])]),
                                             columns=roi_list, index=trial_list)
        if metric=='zscore':
            std_baseline = pd.DataFrame(np.vstack([np.nanstd(traces.iloc[g.index[0:s_on]], axis=0)\
                                            for trial, g in labels.groupby(['trial'])]),
                                             columns=roi_list, index=trial_list)
            tmp = mean_responses.divide(std_baseline)
            mean_responses = tmp.copy()
 
    condition_on_trial = np.array([g['config'].unique()[0] for trial, g in labels.groupby(['trial'])])
    mean_responses['config'] = condition_on_trial

    return mean_responses


def get_aggregate_info_unassigned(traceid='traces001', fov_type='zoom2p0x', state='awake', create_new=False,
                    visual_areas=['V1', 'Lm', 'Li'],
                    aggregate_dir='/n/coxfs01/julianarhee/aggregate-visual-areas',
                    rootdir='/n/coxfs01/2p-data', exclude=[]):
                      
    from pipeline.python.classifications import get_dataset_stats as gd

    sdata_fpath = os.path.join(aggregate_dir, 'dataset_info.pkl')
    if os.path.exists(sdata_fpath) and create_new is False:
        with open(sdata_fpath, 'rb') as f:
            sdata = pkl.load(f)
    else:
        tmpdata = gd.aggregate_session_info(traceid=traceid, 
                                           state=state, fov_type=fov_type, 
                                           visual_areas=visual_areas,
                                           rootdir=rootdir)
        tmpdata['fovnum'] = [int(re.findall(r'FOV(\d+)_', x)[0]) for x in tmpdata['fov']]
        sdata = tmpdata.copy().drop_duplicates().reset_index(drop=True)

        with open(sdata_fpath, 'wb') as f:
            pkl.dump(sdata, f, protocol=pkl.HIGHEST_PROTOCOL)
    
    if 'datakey' not in sdata.columns:        
        sdata['datakey'] = ['%s_%s_fov%i' % (session, animalid, fovnum) 
                              for session, animalid, fovnum in zip(sdata['session'].values, 
                                                                   sdata['animalid'].values,
                                                                   sdata['fovnum'].values)]

    sdata = sdata[~sdata['datakey'].isin(exclude)]
    return sdata


def get_aggregate_info(traceid='traces001', fov_type='zoom2p0x', state='awake',
                        visual_areas=['V1', 'Lm', 'Li', 'Ll'], return_cells=False):
    from pipeline.python.retinotopy import segment_retinotopy as seg
    sdata = get_aggregate_info_unassigned(traceid=traceid, fov_type=fov_type, state=state,
                    visual_areas=visual_areas)
    cells, missing_seg = seg.get_cells_by_area(sdata, return_missing=True)

    d_=[]
    all_dkeys = cells[['visual_area', 'datakey']].drop_duplicates().reset_index(drop=True)
    for (visual_area, datakey), g in all_dkeys.groupby(['visual_area', 'datakey']):
        if visual_area not in visual_areas:
            continue
        found_exps = sdata[(sdata['datakey']==datakey)]['experiment'].values
        tmpd = pd.DataFrame({'experiment': found_exps})
        tmpd['visual_area'] = visual_area
        tmpd['datakey'] = datakey
        d_.append(tmpd)
    all_sdata = pd.concat(d_, axis=0).reset_index(drop=True)
    all_sdata = split_datakey(all_sdata)
    all_sdata['fovnum'] = [int(f.split('_')[0][3:]) for f in all_sdata['fov']]

    if return_cells:
        return all_sdata, cells
    else:
        return all_sdata



def get_aggregate_data_filepath(experiment, traceid='traces001', response_type='dff', 
                        epoch='stimulus', 
                       responsive_test='ROC', responsive_thr=0.05, n_stds=0.0,
                       aggregate_dir='/n/coxfs01/julianarhee/aggregate-visual-areas'):
    
    data_dir = os.path.join(aggregate_dir, 'data-stats')
    sdata = get_aggregate_info(traceid=traceid)
    #### Get DATA
    #load_data = False
    data_desc_base = create_dataframe_name(traceid=traceid, 
                                            response_type=response_type, 
                                            responsive_test=responsive_test,
                                            responsive_thr=responsive_thr,
                                            epoch=epoch)
    
    #if responsive_test is None:
        #if use_all:
        #    data_desc = 'aggr_%s_trialmeans_%s_ALL_%s_%s' % (experiment, traceid, response_type, epoch)
        #else:
    #data_desc = 'aggr_%s_trialmeans_%s_None-thr-%.2f_%s_%s' % (experiment, traceid, responsive_thr, response_type, epoch)

    #else:
    data_desc = 'aggr_%s_%s' % (experiment, data_desc_base)
    data_outfile = os.path.join(data_dir, '%s.pkl' % data_desc)

    return data_outfile #print(data_desc)
    

def create_dataframe_name(traceid='traces001', response_type='dff', 
                             epoch='stimulus',
                             responsive_test='ROC', responsive_thr=0.05, n_stds=0.0): 

    data_desc = 'trialmeans_%s_%s-thr-%.2f_%s_%s' % (traceid, str(responsive_test), responsive_thr, response_type, epoch)
    return data_desc


def load_trial_metrics(animalid, session, fovnum, experiment, traceid='traces001', 
                   response_type='dff', epoch='stimulus',
                   responsive_test='ROC', responsive_thr=0.05, n_stds=2.5, 
                   create_new=False, redo_stats=False, n_processes=1,
                   rootdir='/n/coxfs01/2p-data'):
    tmetrics=None

    fns = glob.glob(os.path.join(rootdir, animalid, session, 'FOV%i_zoom2p0x' % fovnum, 
                        'combined_%s_static' % experiment, 'traces', '%s*' % traceid,
                    'summary_stats', responsive_test, 'METRICS_*%s.pkl' % epoch))
    assert len(fns)==1, "What to load? %s" % str(fns)

    with open(fns[0], 'rb') as f:
        tmetrics = pkl.load(f)

    return tmetrics

def save_trial_metrics(animalid, session, fovnum, experiment, traceid='traces001', 
                   response_type='dff', epoch='stimulus',
                   responsive_test='ROC', responsive_thr=0.05, n_stds=2.5, 
                   create_new=False, redo_stats=False, n_processes=1,
                   rootdir='/n/coxfs01/2p-data'):
    from pipeline.python.classifications import test_responsivity as resp
    '''
    epoch options
        stimulus: use full stimulus period
        baseline: average over baseline period
        firsthalf: use first HALF of stimulus period
        plushalf:  use stimulus period + extra half 
    '''
    # output
    traceid_dir = glob.glob(os.path.join(rootdir, animalid, session, 
                            'FOV%i_*' % fovnum, 'combined_%s_static' % experiment,
                            'traces', '%s*' % traceid))[0]
    if responsive_test is not None:
        statdir = os.path.join(traceid_dir, 'summary_stats', str(responsive_test))
    else:
        statdir = os.path.join(traceid_dir, 'summary_stats')
    data_desc_base = create_dataframe_name(traceid=traceid, 
                                            response_type=response_type, 
                                            responsive_test=responsive_test,
                                            responsive_thr=responsive_thr,
                                            epoch=epoch)
    ndf_fpath = os.path.join(statdir, 'METRICS_%s.pkl' % data_desc_base)
    print("... creating trial metrics (%s)" % data_desc_base) #ndf_fpath) 
   
    # Load traces
    traces, labels, sdf = load_traces(animalid, session, fovnum, 
                                      experiment, traceid=traceid, 
                                      response_type='corrected',
                                      responsive_test=responsive_test, 
                                      responsive_thr=responsive_thr, 
                                      n_stds=n_stds,
                                      redo_stats=redo_stats, 
                                      n_processes=n_processes)
    if traces is None:
        return None
    # Calculate mean trial metric
    nframes_on = float(labels['nframes_on'].unique())
    nframes_post = (nframes_on*0.5) if epoch=='plushalf' else 0.

    gdf = resp.group_roidata_stimresponse(traces, labels, 
                                      nframes_post=nframes_post, return_grouped=False)        
    gdf['snr'] = gdf['stim_mean'] / gdf['base_std']

    # save
    with open(ndf_fpath, 'wb') as f:
        pkl.dump(gdf, f, protocol=pkl.HIGHEST_PROTOCOL)

    return gdf 


def get_neuraldf(animalid, session, fovnum, experiment, traceid='traces001', 
                   response_type='dff', epoch='stimulus',
                   responsive_test='ROC', responsive_thr=0.05, n_stds=2.5, 
                   create_new=False, redo_stats=False, n_processes=1,
                   rootdir='/n/coxfs01/2p-data'):
    '''
    epoch options
        stimulus: use full stimulus period
        baseline: average over baseline period
        firsthalf: use first HALF of stimulus period
        plushalf:  use stimulus period + extra half 
    '''
    # output
    traceid_dir = glob.glob(os.path.join(rootdir, animalid, session, 
                            'FOV%i_*' % fovnum, 'combined_%s_static' % experiment,
                            'traces', '%s*' % traceid))[0]
    if responsive_test is not None:
        statdir = os.path.join(traceid_dir, 'summary_stats', str(responsive_test))
    else:
        statdir = os.path.join(traceid_dir, 'summary_stats')
    data_desc_base = create_dataframe_name(traceid=traceid, 
                                            response_type=response_type, 
                                            responsive_test=responsive_test,
                                            responsive_thr=responsive_thr,
                                            epoch=epoch)
    ndf_fpath = os.path.join(statdir, '%s.pkl' % data_desc_base)
    
    create_new = redo_stats is True
    if not create_new:
        try:
            with open(ndf_fpath, 'rb') as f:
                mean_responses = pkl.load(f)
        except Exception as e:
            print("Unable to get neuraldf. Creating now.")
            create_new=True
    
    if create_new:
        # Load traces
        trace_type = 'df' if response_type=='zscore' else response_type
        traces, labels, sdf = load_traces(animalid, session, fovnum, 
                                          experiment, traceid=traceid, 
                                          response_type=trace_type,
                                          responsive_test=responsive_test, 
                                          responsive_thr=responsive_thr, 
                                          n_stds=n_stds,
                                          redo_stats=redo_stats, 
                                          n_processes=n_processes)
        if traces is None:
            return None
        # Calculate mean trial metric
        metric = 'zscore' if response_type=='zscore' else 'mean'
        mean_responses = traces_to_trials(traces, labels, epoch=epoch, metric=response_type)

        # save
        with open(ndf_fpath, 'wb') as f:
            pkl.dump(mean_responses, f, protocol=pkl.HIGHEST_PROTOCOL)

    return mean_responses

def save_trial_metrics_cycle(experiment, traceid='traces001', 
                       response_type='dff', epoch='stimulus',
                       responsive_test='ROC', responsive_thr=0.05, n_stds=2.5, 
                       create_new=False, redo_stats=False, redo_fov=False,
                       always_exclude=['20190426_JC078'], n_processes=1,
                       aggregate_dir='/n/coxfs01/julianarhee/aggregate-visual-areas'):
    '''
    create_new: remake aggregate file
    redo_stats: for each loaded FOV, re-calculate stats 
    redo_fov: create new neuraldf (otherwise just loads existing)
    '''
    #if experiment=='gratings':
    #    always_exclude.append('20190517_JC083')

    #### Load mean trial info for responsive cells
    data_dir = os.path.join(aggregate_dir, 'data-stats')
    sdata = get_aggregate_info(traceid=traceid)
    
    #### Get DATA   

    print("RUNNING METRICS: %s" % experiment)
    dsets = sdata[sdata['experiment']==experiment].copy()
    no_stats = []
    #DATA = {}
    for (animalid, session, fovnum), g in dsets.groupby(['animalid', 'session', 'fovnum']):
        datakey = '%s_%s_fov%i' % (session, animalid, fovnum)
        if '%s_%s' % (session, animalid) in always_exclude:
            continue 
        else:
            print(datakey)
        mean_responses = save_trial_metrics(animalid, session, fovnum, experiment, 
                            traceid=traceid, 
                            response_type=response_type, epoch=epoch,
                            responsive_test=responsive_test, 
                            responsive_thr=responsive_thr, n_stds=n_stds,
                            redo_stats=any([redo_fov, redo_stats]))          
        if mean_responses is None:
            print("NO stats, rerun: %s" % datakey)
            no_stats.append(datakey)
            continue

    print("There were %i datasets without stats:" % len(no_stats))
    for d in no_stats:
        print(d)
    
    return #data_outfile



def aggregate_and_save(experiment, traceid='traces001', 
                       response_type='dff', epoch='stimulus',
                       responsive_test='ROC', responsive_thr=0.05, n_stds=2.5, 
                       create_new=False, redo_stats=False, redo_fov=False,
                       always_exclude=['20190426_JC078'], n_processes=1,
                       aggregate_dir='/n/coxfs01/julianarhee/aggregate-visual-areas'):
    '''
    create_new: remake aggregate file
    redo_stats: for each loaded FOV, re-calculate stats 
    redo_fov: create new neuraldf (otherwise just loads existing)
    '''
    #if experiment=='gratings':
    #    always_exclude.append('20190517_JC083')

    #### Load mean trial info for responsive cells
    data_dir = os.path.join(aggregate_dir, 'data-stats')
    sdata = get_aggregate_info(traceid=traceid)
    
    #### Get DATA   
    data_outfile = get_aggregate_data_filepath(experiment, traceid=traceid, 
                        response_type=response_type, epoch=epoch,
                        responsive_test=responsive_test, 
                        responsive_thr=responsive_thr, n_stds=n_stds,
                        aggregate_dir=aggregate_dir)
    data_desc = os.path.splitext(os.path.split(data_outfile)[-1])[0]
    print(data_desc)

    if create_new:
        print("Getting data: %s" % experiment)
        print("Saving data to %s" % data_outfile)
        dsets = sdata[sdata['experiment']==experiment].copy()
        no_stats = []
        DATA = {}
        for (animalid, session, fovnum), g in dsets.groupby(['animalid', 'session', 'fovnum']):
            datakey = '%s_%s_fov%i' % (session, animalid, fovnum)
            if '%s_%s' % (session, animalid) in always_exclude:
                continue 
            else:
                print(datakey)
            mean_responses = get_neuraldf(animalid, session, fovnum, experiment, 
                                traceid=traceid, 
                                response_type=response_type, epoch=epoch,
                                responsive_test=responsive_test, 
                                responsive_thr=responsive_thr, n_stds=n_stds,
                                create_new=redo_fov, redo_stats=any([redo_fov, redo_stats]))          
            if mean_responses is None:
                print("NO stats, rerun: %s" % datakey)
                no_stats.append(datakey)
                continue
            DATA[datakey] = mean_responses

        # Save
        with open(data_outfile, 'wb') as f:
            pkl.dump(DATA, f, protocol=pkl.HIGHEST_PROTOCOL)
        print("Done!")

    print("There were %i datasets without stats:" % len(no_stats))
    for d in no_stats:
        print(d)
    
    print("Saved aggr to: %s" % data_outfile)

    return data_outfile

def extract_options(options):
    parser = optparse.OptionParser()

    # PATH opts:
    parser.add_option('-D', '--root', action='store', dest='rootdir', default='/n/coxfs01/2p-data', 
                      help='root project dir containing all animalids [default: /n/coxfs01/2pdata]')
    parser.add_option('-G', '--aggr', action='store', dest='aggregate_dir', default='/n/coxfs01/julianarhee/aggregate-visual-areas', 
                      help='aggregate analysis dir [default: aggregate-visual-areas]')
    parser.add_option('--zoom', action='store', dest='fov_type', default='zoom2p0x', 
                      help="fov type (zoom2p0x)") 
    parser.add_option('--state', action='store', dest='state', default='awake', 
                      help="animal state (awake)") 


  
    # Set specific session/run for current animal:
    parser.add_option('-E', '--experiment', action='store', dest='experiment', default='', 
                      help="experiment name (e.g,. gratings, rfs, rfs10, or blobs)") 

    choices_e = ('stimulus', 'firsthalf', 'plushalf', 'baseline')
    default_e = 'stimulus'
    parser.add_option('-e', '--epoch', action='store', dest='epoch', 
            default=default_e, type='choice', choices=choices_e,
            help="Trial epoch to average, choices: %s. (default: %s" % (choices_e, default_e))


    parser.add_option('-t', '--traceid', action='store', dest='traceid', default='traces001', 
                      help="traceid (default: traces001)")

    choices_c = ('all', 'ROC', 'nstds', None, 'None')
    default_c = 'nstds'
    parser.add_option('-R', '--responsive_test', action='store', dest='responsive_test', 
            default=default_c, type='choice', choices=choices_c,
            help="Responsive test, choices: %s. (default: %s" % (choices_c, default_c))

    parser.add_option('--thr', action='store', dest='responsive_thr', default=0.05, 
                      help="responsive test thr (default: 0.05 for ROC)")
    parser.add_option('-d', '--response', action='store', dest='response_type', default='dff', 
                      help="response type (default: dff)")
    parser.add_option('--nstds', action='store', dest='nstds_above', default=2.5, 
                      help="only for test=nstds, N stds above (default: 2.5)")
    parser.add_option('--new', action='store_true', dest='create_new', default=False, 
                      help="flag to create new")
    
    parser.add_option('-X', '--exclude', action='store', dest='always_exclude', 
                      default=['20190426_JC078'],
                      help="Datasets to exclude bec incorrect or overlap")

    parser.add_option('-n', '--nproc', action='store', dest='n_processes', 
                      default=1,
                      help="N processes (default=1)")
    parser.add_option('--redo-stats', action='store_true', dest='redo_stats', 
                      default=False,
                      help="Flag to redo tests for responsivity")
    parser.add_option('--redo-fov', action='store_true', dest='redo_fov', 
                      default=False,
                      help="Flag to recalculate neuraldf from traces")

    parser.add_option('-i', '--animalid', action='store', dest='animalid', default=None,
                      help="animalid (e.g., JC110)")
    parser.add_option('-S', '--session', action='store', dest='session', default=None,
                      help="session (format: YYYYMMDD)")
    parser.add_option('-A', '--fovnum', action='store', dest='fovnum', default=None,
                      help="fovnum (default: all fovs)")

    parser.add_option('--all',  action='store_true', dest='aggregate', default=False,
                      help="Set flag to cycle thru ALL dsets")

    parser.add_option('--metrics',  action='store_true', dest='do_metrics', default=False,
                      help="Set flag to cycle thru and save all metrics for each dset")






    (options, args) = parser.parse_args(options)

    return options


# Select response filters
# responsive_test='ROC'
# responsive_thr = 0.05
# response_type = 'df'
# experiment = 'blobs'
#always_exclude = ['20190426_JC078']



def main(options):
    opts = extract_options(options)
    experiment = opts.experiment
    traceid = opts.traceid
    response_type = opts.response_type
    responsive_test = None if opts.responsive_test in ['None', 'none', None] else opts.responsive_test
    responsive_thr = 0 if responsive_test is None else float(opts.responsive_thr) 
    n_stds = float(opts.nstds_above) if responsive_test=='nstds' else 0.
    create_new = opts.create_new
    epoch = opts.epoch
    n_processes = int(opts.n_processes)
    redo_stats = opts.redo_stats 
    redo_fov = opts.redo_fov

    run_aggregate = opts.aggregate
    aggregate_dir = opts.aggregate_dir
    fov_type=opts.fov_type
    state=opts.state

    do_metrics = opts.do_metrics
    if run_aggregate: 
        data_outfile = aggregate_and_save(experiment, traceid=traceid, 
                                       response_type=response_type, epoch=epoch,
                                       responsive_test=responsive_test, 
                                       n_stds=n_stds,
                                       responsive_thr=responsive_thr, 
                                       create_new=any([create_new, redo_stats, redo_fov]),
                                       n_processes=n_processes,
                                       redo_stats=redo_stats, redo_fov=redo_fov)
    elif do_metrics:
         save_trial_metrics_cycle(experiment, traceid=traceid, 
                                       response_type=response_type, epoch=epoch,
                                       responsive_test=responsive_test, 
                                       n_stds=n_stds,
                                       responsive_thr=responsive_thr, 
                                       create_new=any([create_new, redo_stats, redo_fov]),
                                       n_processes=n_processes,
                                       redo_stats=redo_stats, redo_fov=redo_fov)
   
    else:
        animalid = opts.animalid
        session = opts.session
        fov = opts.fovnum
        if fov is not None:
            fovnum = int(fov) if isnumber(fov) else int(fov.split('_')[0][3:]) 

        assert animalid is not None, "NO animalid specified, aborting"

        if session is None or fov is None:
            sdata = get_aggregate_info(traceid=traceid, fov_type=fov_type, state=state)
            #dsets = sdata[(sdata['animalid']==animalid) & (sdata['experiment']==experiment)].copy()
            if session is not None:
                dsets = sdata[(sdata['animalid']==animalid) & (sdata['experiment']==experiment) 
                            & (sdata['session']==session)].copy()
            else:
                dsets = sdata[(sdata['animalid']==animalid) & (sdata['experiment']==experiment)].copy()
 
            for (animalid, session, fovnum, datakey), g in dsets.groupby(['animalid', 'session', 'fovnum', 'datakey']):
                print("------------------------------------------------------------")
                print("getting stats: %s" % datakey)
                print("------------------------------------------------------------")
                try:
                    neuraldf = get_neuraldf(animalid, session, fovnum, experiment, traceid=traceid, 
                                               response_type=response_type, epoch=epoch,
                                               responsive_test=responsive_test, 
                                               n_stds=n_stds,
                                               responsive_thr=responsive_thr, 
                                               create_new=redo_fov,
                                               n_processes=n_processes,
                                               redo_stats=redo_stats)
                except Exception as e:
                    print("Error getting data: %s. Skipping." % datakey)
                    continue
        else:
            neuraldf = get_neuraldf(animalid, session, fovnum, experiment, traceid=traceid, 
                                               response_type=response_type, epoch=epoch,
                                               responsive_test=responsive_test, 
                                               n_stds=n_stds,
                                               responsive_thr=responsive_thr, 
                                               create_new=redo_fov,
                                               n_processes=n_processes,
                                                redo_stats=redo_stats)
                
         
    print("saved data.")
   

if __name__ == '__main__':
    main(sys.argv[1:])
