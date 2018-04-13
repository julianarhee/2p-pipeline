#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 14:02:29 2018

@author: juliana
"""


# SI = 0 (no selectivity) --> 1 (high selectivity)
# 1. Split trials in half, assign shapes to which neuron shows highest/lowest activity
# 2. Use other half to measure activity to assigned shapes and calculate SI:
#    SI = (Rmost - Rleast) / (Rmost + Rleast)

SI = (Rmost - Rleast) / (Rmost + Rleast)


#%%

import h5py
import os
import json

import pandas as pd
import numpy as np
import pylab as pl
import seaborn as sns

from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm


from pipeline.python.utils import natural_keys
import pipeline.python.traces.combine_runs as cb
import pipeline.python.paradigm.align_acquisition_events as acq

#%%

rootdir = '/mnt/odyssey'
animalid = 'CE077'
session = '20180321'
acquisition = 'FOV1_zoom1x'
rundir = 'blobs_run3_blobs_run4'
tracedir = 'traces002_traces002'

trace_type = 'raw'
filter_pupil = True
pupil_radius_min = 20
pupil_radius_max = 60
pupil_dist_thr = 3.0

traceid_dir = os.path.join(rootdir, animalid, session, acquisition, rundir, tracedir)

#%% # Load ROIDATA file:
roidata_filepath = os.path.join(traceid_dir, 'ROIDATA_098054_626d01_raw.hdf5')
DATA = pd.HDFStore(roidata_filepath, 'r')
datakeys = DATA.keys()
print "Found %i data keys" % len(datakeys)
if len(datakeys) == 1:
    datakey = datakeys[0]
print "Loading dataframe..."
DATA = DATA[datakey]

#%% Set filter params:

if filter_pupil is True:
    pupil_params = acq.set_pupil_params(radius_min=pupil_radius_min,
                                        radius_max=pupil_radius_max,
                                        dist_thr=pupil_dist_thr,
                                        create_empty=False)
elif filter_pupil is False:
    pupil_params = acq.set_pupil_params(create_empty=True)

#%% Calculate metrics & get stats ---------------------------------------------

STATS, stats_filepath = cb.get_combined_stats(DATA, datakey, traceid_dir, trace_type=trace_type, filter_pupil=filter_pupil, pupil_params=pupil_params)

#%%

trial_list = sorted(list(set(STATS['trial'])), key=natural_keys)
roi_list = sorted(list(set(STATS['roi'])), key=natural_keys)

roi = roi_list[0]
roiSTATS = STATS[STATS['roi']==roi]
grouped = roiSTATS.groupby(['config', 'trial']).agg({'stim_df': 'mean',
                                                         'baseline_df': 'mean'
                                                        }).dropna()

df = roiSTATS[['config', 'trial', 'baseline_df', 'stim_df', 'xpos', 'ypos', 'morphlevel', 'yrot', 'size']]
df = roiSTATS[['config', 'trial', 'baseline_df', 'stim_df']].dropna()

tmpd = df.pivot_table(['stim_df', 'baseline_df'], ['config', 'trial']).T

data = []
data.append(pd.DataFrame({'epoch': np.tile('baseline', (len(tmpd.loc['baseline_df'].values),)),
              'df': tmpd.loc['baseline_df'].values,
              'config': [cfg[0] for cfg in tmpd.loc['baseline_df'].index.tolist()],
              'trial': [cfg[1] for cfg in tmpd.loc['baseline_df'].index.tolist()]
              }))
data.append(pd.DataFrame({'epoch': np.tile('stimulus', (len(tmpd.loc['stim_df'].values),)),
              'df': tmpd.loc['stim_df'].values,
              'config': [cfg[0] for cfg in tmpd.loc['baseline_df'].index.tolist()],
              'trial': [cfg[1] for cfg in tmpd.loc['baseline_df'].index.tolist()]
              }))
data = pd.concat(data)

#%%
#subd = data[['config', 'df', 'epoch']]

interaction_plot(data.epoch, data.config, data.df,
             colors=[np.random.rand(3,) for c in range(9)],
             markers=['.','D','_','|','*','v','^','<','>'], ms=10)

config_list = sorted(list(set(data.config)), key=natural_keys)

#%%
# Calculate degrees of freedom and sample size(N):

N = len(data.df)
df_a = len(data.epoch.unique()) - 1
df_b = len(data.config.unique()) - 1
df_axb = df_a*df_b
df_w = N - (len(data.epoch.unique())*len(data.config.unique()))

#%%
# Calculate sum of squares:

grand_mean = data['df'].mean()  # Grand mean
ssq_a = sum([(data[data.epoch ==l].df.mean()-grand_mean)**2 for l in data.epoch])    # Sum of squares for factor A (trial epoch)
ssq_b = sum([(data[data.config ==l].df.mean()-grand_mean)**2 for l in data.config])  # Sum of squares for factor B (stim config)
ssq_t = sum((data.df - grand_mean)**2)                                               # Sum of squares total

#%%
# Calculate Sum of Squares Within (error/residual):

within_sums = []
for stimconfig in config_list:
    currconfig = data[data.config == stimconfig]
    currconfig_epoch_means = [currconfig[currconfig.epoch == e].df.mean() for e in currconfig.epoch]
    curr_ss = sum((currconfig.df - currconfig_epoch_means)**2)
    within_sums.append(curr_ss)

ssq_w = sum(within_sums)

#%%
# Calculate Sum of Squares Interaction (ss for interaction b/w A and B):
ssq_axb = ssq_t - ssq_a - ssq_b - ssq_w

#%%
# Calculate mean squares:

ms_a = ssq_a / df_a        # Mean square A
ms_b = ssq_b / df_b        # Mean square B
ms_axb = ssq_axb / df_axb  # Mean suare interaction
ms_w = ssq_w / df_w        # Mean square residual

#%%
# Calculate F-statistic:
f_a = ms_a / ms_w
f_b = ms_b / ms_w
f_axb = ms_axb / ms_w

#%%
# Get p-values to see if F-ratios above critical valueeee:

p_a = stats.f.sf(f_a, df_a, df_w)
p_b = stats.f.sf(f_b, df_b, df_w)
p_axb = stats.f.sf(f_axb, df_axb, df_w)



#%%
# Store info in dataframe:
results = {'sum_sq':[ssq_a, ssq_b, ssq_axb, ssq_w],
           'dof':[df_a, df_b, df_axb, df_w],
           'F':[f_a, f_b, f_axb, 'NaN'],
            'PR(>F)':[p_a, p_b, p_axb, 'NaN']}

columns=['sum_sq', 'dof', 'F', 'PR(>F)']

aov_table1 = pd.DataFrame(results, columns=columns,
                          index=['epoch', 'config',
                          'epoch:config', 'Residual'])


#%%
# Get effect sizes:

def eta_squared(aov):
    aov['eta_sq'] = 'NaN'
    aov['eta_sq'] = aov[:-1]['sum_sq']/sum(aov['sum_sq'])
    return aov

def omega_squared(aov):
    if 'dof' in aov.keys():
        dfkey = 'dof'
    else:
        dfkey = 'df'
    mse = aov['sum_sq'][-1]/aov[dfkey][-1]
    aov['omega_sq'] = 'NaN'
    aov['omega_sq'] = (aov[:-1]['sum_sq']-(aov[:-1][dfkey]*mse))/(sum(aov['sum_sq'])+mse)
    return aov


eta_squared(aov_table1)
omega_squared(aov_table1)
print(aov_table1)


#%%  Use STATSMODELS:
formula = 'df ~ C(epoch) + C(config) + C(epoch):C(config)'
model = ols(formula, data).fit()
aov_table = anova_lm(model, typ=2)


eta_squared(aov_table)
print(aov_table)

#%%

import statsmodels.api as sm

res = model.resid

pl.figure()
pl.subplot(1,2,1)
sm.graphics.qqplot(res, line='s')

pl.subplot(1,2,2)
sm.graphics.qqplot(res, line='45', fit=True)
pl.show()


#%%


