import os
import glob
import shutil
import json
import re
import sys
import optparse
import itertools

import statsmodels as sm
import scipy.stats as spstats
import pandas as pd
import numpy as np
import pylab as pl
import seaborn as sns
import cPickle as pkl

from pipeline.python.classifications import experiment_classes as util
from pipeline.python.utils import label_figure, natural_keys, reformat_morph_values

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
                          'Li': [('20191106_fov1', '20191111')]}
                }

    return fov_keys

def all_datasets_by_area(visual_areas=[]):
    if len(visual_areas)==0:
        visual_areas = ['V1', 'Lm', 'Li']
        
    fkeys = get_sorted_fovs()
    
    ddict = dict((v, []) for v in visual_areas)
    for animalid, sinfo in fkeys.items():
        for visual_area, slist in sinfo.items():
            for sublist in slist:
                if isinstance(sublist, tuple):
                    sessions_ = ['%s_%s_%s' % (s.split('_')[0], animalid, s.split('_')[-1]) \
                                 if len(s.split('_'))>1 else '%s_%s' % (s, animalid) for s in sublist]
                else:
                    sessions_ = ['%s_%s_%s' % (sublist.split('_')[0], animalid, sublist.split('_')[-1]) \
                             if len(sublist.split('_'))>1 else '%s_%s' % (sublist, animalid)]

                ddict[visual_area].extend(sessions_)

    return ddict


def get_metadata(traceid='traces001', filter_by='most_cells', stimulus=None, stimulus_only=False,
                 fov_type='zoom2p0x', state='awake', excluded_sessions=[],
                 aggregate_dir='/n/coxfs01/julianarhee/aggregate-visual-areas'):
       
    # Get all datasets
    sdata = get_aggregate_info(traceid=traceid, fov_type=fov_type, state=state,
                             aggregate_dir=aggregate_dir)

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

def get_blob_datasets(filter_by='first', has_gratings=False,
                        excluded_sessions=[], as_dict=True):

    included_sessions = []
    
    # Blobs runs w/ incorrect stuff
    always_exclude = ['20190426_JC078', # backlight test, but good for A/B
                      '20191008_JC091'] # rf test
    excluded_sessions.extend(always_exclude)

    if filter_by is None:
        v1_repeats = []
        lm_repeats = []
        li_repeats = []
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


def get_gratings_datasets(filter_by='first', excluded_sessions=[], as_dict=True):

    included_sessions = []
    
    # Blobs runs w/ incorrect stuff
    always_exclude = ['20190426_JC078']
    excluded_sessions.extend(always_exclude)
    

    if filter_by is None:
        v1_repeats = []
        lm_repeats = []
        li_repeats = []
    
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


def get_rf_datasets(filter_by='drop_repeats', excluded_sessions=[], as_dict=True, return_excluded=False):
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
    
# ===============================================================
# Plotting
# ===============================================================
from matplotlib.lines import Line2D

def annotateBars(row, ax, fontsize=12, fmt='%.2f'): 
    for p in ax.patches:
        ax.annotate(fmt % p.get_height(), (p.get_x() + p.get_width() / 2., 0.), #p.get_height()),
                    ha='center', va='center', fontsize=fontsize, color='k', 
                    rotation=0, xytext=(0, 20),
             textcoords='offset points')
        

def plot_mannwhitney(mdf, metric='I_rs', multi_comp_test='holm', ax=None):
    if ax is None:
        fig, ax = pl.subplots()

    print("********* [%s] Mann-Whitney U test(mc=%s) **********" % (metric, multi_comp_test))
    statresults = do_mannwhitney(mdf, metric=metric, multi_comp_test=multi_comp_test)
    #print(statresults)
    
    # stats significance
    ax = annotate_stats_areas(statresults, ax)
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

    return ax

def get_counts_for_legend(df, area_colors=None, markersize=10, marker='_',
              visual_areas=['V1', 'Lm', 'Li']):
    from matplotlib.lines import Line2D

    if area_colors is None:
        colors = ['magenta', 'orange', 'dodgerblue'] #sns.color_palette(palette='colorblind') #, n_colors=3)
        area_colors = {'V1': colors[0], 'Lm': colors[1], 'Li': colors[2]}


    # Get counts
    if 'cell' in df.columns:
        counts = df.groupby(['visual_area', 'animalid', 'datakey'])['cell'].count().reset_index()
        counts.rename(columns={'cell': 'n_cells'}, inplace=True)
    else:
        counts = df.groupby(['visual_area', 'animalid', 'datakey']).count().reset_index()

    # Get counts of samples for legend
    n_rats = dict((v, len(g['animalid'].unique())) for v, g in counts.groupby(['visual_area']))
    n_fovs = dict((v, len(g[['datakey']].drop_duplicates())) for v, g in counts.groupby(['visual_area']))
    if 'cell' in df.columns.tolist():
        n_cells = dict((v, g['n_cells'].sum()) for v, g in counts.groupby(['visual_area']))
        legend_elements = [Line2D([0], [0], marker='_', markersize=10, \
                                  lw=1, color=area_colors[v], markerfacecolor=area_colors[v],
                                  label='%s (n=%i rats, %i fovs, %i cells)' % (v, n_rats[v], n_fovs[v], n_cells[v]))\
                           for v in visual_areas]
    else:
        legend_elements = [Line2D([0], [0], marker='_', markersize=10, \
                                  lw=1, color=area_colors[v], markerfacecolor=area_colors[v],
                                  label='%s (n=%i rats, %i fovs)' % (v, n_rats[v], n_fovs[v]))\
                           for v in visual_areas]

        
    return legend_elements

# ===============================================================
# Data loading
# ===============================================================

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
            return None, None, None
        traces = exp.data.traces[responsive_cells]
    else:
        traces = exp.data.traces

    return traces, labels, sdf


def traces_to_trials(traces, labels, epoch='stimulus', metric='mean'):
    '''
    Returns dataframe w/ columns = roi ids, rows = mean response to stim ON per trial
    Last column is config on given trial.
    '''
    s_on = int(labels['stim_on_frame'].mean())
    n_on = int(labels['nframes_on'].mean())

    roi_list = traces.columns.tolist()
    trial_list = np.array([int(trial[5:]) for trial, g in labels.groupby(['trial'])])
    if epoch=='stimulus':
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


def get_aggregate_info(traceid='traces001', fov_type='zoom2p0x', state='awake', create_new=False,
                       visual_areas=['V1', 'Lm', 'Li'],
                         aggregate_dir='/n/coxfs01/julianarhee/aggregate-visual-areas',
                         rootdir='/n/coxfs01/2p-data'):
                       
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
            
    return sdata

def get_aggregate_data_filepath(experiment, traceid='traces001', response_type='dff', epoch='stimulus',
                       responsive_test='ROC', responsive_thr=0.05, n_stds=0.0,
                       aggregate_dir='/n/coxfs01/julianarhee/aggregate-visual-areas'):
    
    data_dir = os.path.join(aggregate_dir, 'data-stats')
    sdata = get_aggregate_info(traceid=traceid)
    #### Get DATA
    #load_data = False
    data_desc = 'aggr_%s_trialmeans_%s_%s-thr-%.2f_%s_%s' % (experiment, traceid, responsive_test, responsive_thr, response_type, epoch)
    data_outfile = os.path.join(data_dir, '%s.pkl' % data_desc)

    return data_outfile #print(data_desc)
    

def load_aggregate_data(experiment, traceid='traces001', response_type='dff', epoch='stimulus',
                       responsive_test='ROC', responsive_thr=0.05, n_stds=0.0,
                       aggregate_dir='/n/coxfs01/julianarhee/aggregate-visual-areas'):
    
    data_outfile = get_aggregate_data_filepath(experiment, traceid=traceid, 
                        response_type=response_type, epoch=epoch,
                        responsive_test=responsive_test, 
                        responsive_thr=responsive_thr, n_stds=n_stds,
                       aggregate_dir=aggregate_dir)
    # print("...loading: %s" % data_outfile)

    with open(data_outfile, 'rb') as f:
        DATA = pkl.load(f)
    print("...loading: %s" % data_outfile)

    return DATA


def aggregate_and_save(experiment, traceid='traces001', 
                       response_type='dff', epoch='stimulus',
                       responsive_test='ROC', responsive_thr=0.05, n_stds=2.5, 
                       create_new=False, redo_stats=False,
                       always_exclude=['20190426_JC078'], n_processes=1,
                       aggregate_dir='/n/coxfs01/julianarhee/aggregate-visual-areas'):

    #### Load mean trial info for responsive cells
    data_dir = os.path.join(aggregate_dir, 'data-stats')
    sdata = get_aggregate_info(traceid=traceid)
    
    #### Get DATA   
    #data_desc = '%s_%s-%s_%s-thr-%.2f_%s' % (experiment, traceid, response_type, responsive_test, responsive_thr, epoch)
    data_outfile = get_aggregate_data_filepath(experiment, traceid=traceid, 
                        response_type=response_type, epoch=epoch,
                        responsive_test=responsive_test, 
                        responsive_thr=responsive_thr, n_stds=n_stds,
                        aggregate_dir=aggregate_dir)
    data_desc = os.path.splitext(os.path.split(data_outfile)[-1])[0]

    #if not os.path.exists(data_outfile):
    #    load_data = True
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
                print("NO stats, rerun: %s" % datakey)
                no_stats.append(datakey)
                continue
            # Calculate mean trial metric
            metric = 'zscore' if response_type=='zscore' else 'mean'
            mean_responses = traces_to_trials(traces, labels, epoch=epoch, 
                                                metric=response_type)
            DATA[datakey] = mean_responses #{'data': mean_responses,
                                    #'sdf': sdf}

        # Save
        with open(data_outfile, 'wb') as f:
            pkl.dump(DATA, f, protocol=pkl.HIGHEST_PROTOCOL)
        print("Done!")

    print("There were %i datasets without stats:" % len(no_stats))
    for d in no_stats:
        print(d)

    return data_outfile

def extract_options(options):
    parser = optparse.OptionParser()

    # PATH opts:
    parser.add_option('-D', '--root', action='store', dest='rootdir', default='/n/coxfs01/2p-data', 
                      help='root project dir containing all animalids [default: /n/coxfs01/2pdata]')
   
    # Set specific session/run for current animal:
    parser.add_option('-E', '--experiment', action='store', dest='experiment', default='', 
                      help="experiment name (e.g,. gratings, rfs, rfs10, or blobs)") #: FOV1_zoom2p0x)")

    parser.add_option('-e', '--epoch', action='store', dest='epoch', default='stimulus', 
                      help="trial epoch (default: stimulus)")
 
    parser.add_option('-t', '--traceid', action='store', dest='traceid', default='traces001', 
                      help="traceid (default: traces001)")
    parser.add_option('--test', action='store', dest='responsive_test', default='ROC', 
                      help="responsive test (default: ROC, set to None if want all cells returned)")
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
    parser.add_option('--do-stats', action='store_true', dest='redo_stats', 
                      default=False,
                      help="Flag to redo tests for responsivity")



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
    responsive_test = opts.responsive_test
    if responsive_test in ['None', 'none']:
        responsive_test = None
    responsive_thr = float(opts.responsive_thr)
    n_stds = float(opts.nstds_above)
    create_new = opts.create_new
    epoch = opts.epoch
    n_processes = int(opts.n_processes)
    redo_stats = opts.redo_stats 
    data_outfile = aggregate_and_save(experiment, traceid=traceid, 
                                       response_type=response_type, epoch=epoch,
                                       responsive_test=responsive_test, 
                                       n_stds=n_stds,
                                       responsive_thr=responsive_thr, 
                                       create_new=create_new,
                                        n_processes=n_processes,
                                        redo_stats=redo_stats)
    
    print("saved data.")
   

if __name__ == '__main__':
    main(sys.argv[1:])
