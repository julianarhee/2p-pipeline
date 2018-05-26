#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 12:40:21 2018

@author: juliana
"""


#%%
#
#def pyvt_raw_epochXoriXsf(rdata, trans_types, save_fig=False, output_dir='/tmp', fname='boxplot(intensity~epoch_X_config).png'):
#
#    # Make sure trans_types are sorted:
#    trans_types = sorted(trans_types, key=natural_keys)
#
#    assert len(list(set(rdata['nframes_on'])))==1, "More than 1 idx found for nframes on... %s" % str(list(set(rdata['nframes_on'])))
#    assert len(list(set(rdata['first_on'])))==1, "More than 1 idx found for first frame on... %s" % str(list(set(rdata['first_on'])))
#
#    nframes_on = int(round(list(set(rdata['nframes_on']))[0]))
#    first_on =  int(round(list(set(rdata['first_on']))[0]))
#
#    df_groups = np.copy(trans_types).tolist()
#    df_groups.extend(['trial', 'raw'])
#    groupby_list = np.copy(trans_types).tolist()
#    groupby_list.extend(['trial'])
#
#    currdf = rdata[df_groups] #.sort_values(trans_types)
#    grp = currdf.groupby(groupby_list)
#    config_trials = {} # dict((config, []) for config in list(set(currdf['config'])))
#    for k,g in grp: #config_trials.keys():
#        if k[0] not in config_trials.keys():
#            config_trials[k[0]] = {}
#        if k[1] not in config_trials[k[0]].keys():
#            config_trials[k[0]][k[1]] = {}
#
#        config_trials[k[0]][k[1]] = sorted(list(set(currdf.loc[(currdf['ori']==k[0])
#                                                        & (currdf['sf']==k[1])]['trial'])), key=natural_keys)
#
#    idx = 0
#    df_list = []
#    for k,g in grp:
#        #print k
#        base_mean= g['raw'][0:first_on].mean()
#        base_std = g['raw'][0:first_on].std()
#        stim_mean = g['raw'][first_on:first_on+nframes_on].mean()
#
#        df_list.append(pd.DataFrame({'ori': k[0],
#                                     'sf': k[1],
#                                     'trial': 'trial%05d' % int(config_trials[k[0]][k[1]].index(k[2]) + 1),
#                                     'epoch': 'baseline',
#                                     'intensity': base_mean}, index=[idx]))
#        df_list.append(pd.DataFrame({'ori': k[0],
#                                     'sf': k[1],
#                                     'trial': 'trial%05d' % int(config_trials[k[0]][k[1]].index(k[2]) + 1),
#                                     'epoch': 'stimulus',
#                                     'intensity': stim_mean}, index=[idx+1]))
#        idx += 2
#    df = pd.concat(df_list, axis=0)
#    df = df.sort_values(['epoch', 'ori', 'sf'])
#    df = df.reset_index(drop=True)
#    #pdf.pivot_table(index=['trial'], columns=['config', 'epoch'], values='intensity')
#
#    # Format pandas df into pyvttbl dataframe:
#    df_factors = np.copy(trans_types).tolist()
#    df_factors.extend(['trial', 'epoch', 'intensity'])
#
#    Trial = namedtuple('Trial', df_factors)
#    pdf = pt.DataFrame()
#    for idx in xrange(df.shape[0]):
#        pdf.insert(Trial(df.loc[idx, 'ori'],
#                         df.loc[idx, 'sf'],
#                         df.loc[idx, 'trial'],
#                         df.loc[idx, 'epoch'],
#                         df.loc[idx, 'intensity'])._asdict())
#
#    if save_fig:
#        factor_list = np.copy(trans_types).tolist()
#        factor_list.extend(['epoch'])
#        pdf.box_plot('intensity', factors=factor_list, fname=fname, output_dir=output_dir)
#
#    return pdf
#%%
#def pyvt_stimdf_oriXsf(rdata, trans_types, save_fig=False, output_dir='/tmp', fname='dff_boxplot(intensity~epoch_X_config).png'):
#
#    # Make sure trans_types are sorted:
#    trans_types = sorted(trans_types, key=natural_keys)
#
#    assert len(list(set(rdata['nframes_on'])))==1, "More than 1 idx found for nframes on... %s" % str(list(set(rdata['nframes_on'])))
#    assert len(list(set(rdata['first_on'])))==1, "More than 1 idx found for first frame on... %s" % str(list(set(rdata['first_on'])))
#
#    nframes_on = int(round(list(set(rdata['nframes_on']))[0]))
#    first_on =  int(round(list(set(rdata['first_on']))[0]))
#
#    df_groups = np.copy(trans_types).tolist()
#    df_groups.extend(['trial', 'df'])
#    groupby_list = np.copy(trans_types).tolist()
#    groupby_list.extend(['trial'])
#
#    currdf = rdata[df_groups] #.sort_values(trans_types)
#    grp = currdf.groupby(groupby_list)
#    config_trials = {} # dict((config, []) for config in list(set(currdf['config'])))
#    for k,g in grp: #config_trials.keys():
#        if k[0] not in config_trials.keys():
#            config_trials[k[0]] = {}
#        if k[1] not in config_trials[k[0]].keys():
#            config_trials[k[0]][k[1]] = {}
#
#        config_trials[k[0]][k[1]] = sorted(list(set(currdf.loc[(currdf['ori']==k[0])
#                                                        & (currdf['sf']==k[1])]['trial'])), key=natural_keys)
#
#    idx = 0
#    df_list = []
#    for k,g in grp:
#        #print k
#        base_mean= g['df'][0:first_on].mean()
#        base_std = g['df'][0:first_on].std()
#        stim_mean = g['df'][first_on:first_on+nframes_on].mean()
#        zscore_val = stim_mean / base_std
#
#        df_list.append(pd.DataFrame({'ori': k[0],
#                                     'sf': k[1],
#                                     'trial': k[2], #'trial%05d' % int(config_trials[k[0]][k[1]].index(k[2]) + 1),
#                                     'dff': stim_mean,
#                                     'zscore': zscore_val}, index=[idx]))
#
#        idx += 1
#    df = pd.concat(df_list, axis=0)
#    df = df.sort_values(trans_types)
#    df = df.reset_index(drop=True)
#    #pdf.pivot_table(index=['trial'], columns=['config', 'epoch'], values='intensity')
#
#    # Format pandas df into pyvttbl dataframe:
#    df_factors = np.copy(trans_types).tolist()
#    df_factors.extend(['trial', 'dff'])
#
#    Trial = namedtuple('Trial', df_factors)
#    pdf = pt.DataFrame()
#    for idx in xrange(df.shape[0]):
#        pdf.insert(Trial(df.loc[idx, 'ori'],
#                         df.loc[idx, 'sf'],
#                         df.loc[idx, 'trial'],
#                         df.loc[idx, 'dff'])._asdict())
#
#    if save_fig:
#        factor_list = np.copy(trans_types).tolist()
#        pdf.box_plot('dff', factors=factor_list, fname=fname, output_dir=output_dir)
#
#    return pdf

#%%
#def pyvt_stimdf_configs(rdata, save_fig=False, output_dir='/tmp', fname='boxplot(intensity~epoch_X_config).png'):
#
#    '''
#    '''
#
#    assert len(list(set(rdata['nframes_on'])))==1, "More than 1 idx found for nframes on... %s" % str(list(set(rdata['nframes_on'])))
#    assert len(list(set(rdata['first_on'])))==1, "More than 1 idx found for first frame on... %s" % str(list(set(rdata['first_on'])))
#
#    nframes_on = int(round(list(set(rdata['nframes_on']))[0]))
#    first_on =  int(round(list(set(rdata['first_on']))[0]))
#
#    df_groups = ['config', 'trial', 'df']
#    groupby_list = ['config', 'trial']
#
#    currdf = rdata[df_groups] #.sort_values(trans_types)
#    grp = currdf.groupby(groupby_list)
#    config_trials = {} # dict((config, []) for config in list(set(currdf['config'])))
#    for k,g in grp: #config_trials.keys():
#        if k[0] not in config_trials.keys():
#            config_trials[k[0]] = {}
#
#        config_trials[k[0]] = sorted(list(set(currdf.loc[currdf['config']==k[0]]['trial'])), key=natural_keys)
#
#    idx = 0
#    df_list = []
#    for k,g in grp:
#        #print k
#        base_mean= g['df'][0:first_on].mean()
#        base_std = g['df'][0:first_on].std()
#        stim_mean = g['df'][first_on:first_on+nframes_on].mean()
#
#        df_list.append(pd.DataFrame({'config': k[0],
#                                     'trial': k[1], #'trial%05d' % int(config_trials[k[0]].index(k[1]) + 1),
#                                     'dff': stim_mean}, index=[idx]))
#        idx += 1
#    df = pd.concat(df_list, axis=0)
#    df = df.sort_values(['config'])
#    df = df.reset_index(drop=True)
#
#    #pdf.pivot_table(index=['trial'], columns=['config', 'epoch'], values='intensity')
#
#    # Format pandas df into pyvttbl dataframe:
#    df_factors = ['config', 'trial', 'dff']
#
#    Trial = namedtuple('Trial', df_factors)
#    pdf = pt.DataFrame()
#    for idx in xrange(df.shape[0]):
#        pdf.insert(Trial(df.loc[idx, 'config'],
#                         df.loc[idx, 'trial'],
#                         df.loc[idx, 'dff'])._asdict())
#
#    if save_fig:
#        factor_list = ['config']
#        pdf.box_plot('dff', factors=factor_list, fname=fname, output_dir=output_dir)
#
#    return pdf



#%%
#def pyvt_raw_epochXsinglecond(rdata, curr_config='config001'):
#
#    '''
#    Treat single condition for single ROI as dataset, and do ANOVA with 'epoch' as factor.
#    Test for main-effect of trial epoch -- but this is just 1-way...?
#    '''
#
#    assert len(list(set(rdata['nframes_on'])))==1, "More than 1 idx found for nframes on... %s" % str(list(set(rdata['nframes_on'])))
#    assert len(list(set(rdata['first_on'])))==1, "More than 1 idx found for first frame on... %s" % str(list(set(rdata['first_on'])))
#
#    nframes_on = int(round(list(set(rdata['nframes_on']))[0]))
#    first_on =  int(round(list(set(rdata['first_on']))[0]))
#
#    df_groups = ['config', 'trial', 'raw']
#    groupby_list = ['config', 'trial']
#
#    currdf = rdata[df_groups]
#
#    # Split DF by current stimulus config:
#    currdf = currdf[currdf['config']==curr_config]
#
#    grp = currdf.groupby(groupby_list)
#    trial_list = sorted(list(set(currdf['trial'])), key=natural_keys)
#
#    idx = 0
#    df_list = []
#    for k,g in grp:
#        print k
#        base_mean= g['raw'][0:first_on].mean()
#        base_std = g['raw'][0:first_on].std()
#        stim_mean = g['raw'][first_on:first_on+nframes_on].mean()
#
#        df_list.append(pd.DataFrame({'config': k[0],
#                                     'trial': 'trial%05d' % int(trial_list.index(k[1]) + 1),
#                                     'epoch': 'baseline',
#                                     'intensity': base_mean}, index=[idx]))
#        df_list.append(pd.DataFrame({'config': k[0],
#                                     'trial': 'trial%05d' % int(trial_list.index(k[1]) + 1),
#                                     'epoch': 'stimulus',
#                                     'intensity': stim_mean}, index=[idx+1]))
#        idx += 2
#    df = pd.concat(df_list, axis=0)
#    idxs = pd.Index(xrange(0, len(df)))
#    df = df.sort_values(['epoch', 'config'])
#    df = df.reset_index(drop=True)
#
#    #pdf.pivot_table(index=['trial'], columns=['config', 'epoch'], values='intensity')
#
#    # Format pandas df into pyvttbl dataframe:
#    df_factors = ['trial', 'epoch', 'intensity']
#
#    Trial = namedtuple('Trial', df_factors)
#    pdf = pt.DataFrame()
#    for idx in xrange(df.shape[0]):
#        pdf.insert(Trial(df.loc[idx, 'trial'],
#                         df.loc[idx, 'epoch'],
#                         df.loc[idx, 'intensity'])._asdict())
#
#    return pdf
#
