#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 17:25:15 2018

@author: juliana
"""

import matplotlib
matplotlib.use('agg')
import glob
import os
import numpy as np
import pandas as pd
import pylab as pl
import cPickle as pkl

import statsmodels
import statsmodels.api as sm
from statsmodels.formula.api import ols
import scipy.stats as stats

from pipeline.python.visualization.plot_session_summary import SessionSummary


def load_session_summary(acquisition_dir):
    #%
    # First check if saved session summary info exists:
    ss_fpaths = glob.glob(os.path.join(acquisition_dir, 'session_summary*.pkl'))
    assert len(ss_fpaths) > 0, "No summaries found! Did you run plot_session_summary() for acq:\n--> %s ??" % acquisition_dir

    if len(ss_fpaths) > 1:
        print "More than 1 SS fpath found:"
        for si, spath in enumerate(ss_fpaths):
            print si, spath
        selected = raw_input("Select IDX to plot: ")    
        session_summary_fpath = ss_fpaths[int(selected)]
    else:
        session_summary_fpath = ss_fpaths[0]

    with open(session_summary_fpath, 'rb') as f:
        S = pkl.load(f)
                
    return S

rootdir = '/mnt/odyssey'
#rootdir = '/n/coxfs01/2p-data'
animalid = 'JC015'
session = '20180915'
acquisition = 'FOV1_zoom2p7x'
stimulus = 'blobs'

# Load SessionSummary object:
acquisition_dir = os.path.join(rootdir, animalid, session, acquisition)
S = load_session_summary(acquisition_dir)

# Select user-specified stimulus group:
assert stimulus in dir(S), "Specified stimtype -- %s -- not found! Found:\n" % (stimulus, '\n'.join([attr for attr in dir(S) if '__' not in attr]))
if stimulus == 'gratings':
    data = S.gratings
elif stimulus == 'blobs':
    data = S.blobs
elif stimulus == 'objects':
    data = S.objects

roidata = data['roidata']
sdf = pd.DataFrame(data['sconfigs'])
rois = data['roistats']['rois_visual']



rdata = roidata.get_group(rois[0])
grp = rdata.groupby('config')

transforms = []
for curr_group in grp.groups.keys():
    curr_transforms = pd.DataFrame(data=np.tile(sdf[curr_group], (grp.get_group(curr_group).shape[0],1)), columns=sdf[curr_group].index)
    config_df = pd.concat([grp.get_group(curr_group).reset_index(drop=True), curr_transforms], axis=1)
    transforms.append(config_df)
transforms = pd.concat(transforms).reset_index(drop=True)

if stimulus == 'blobs' or stimulus == 'objects':
    objects = transforms.groupby('object')
else:
    objects = transforms.groupby('ori')
    
transforms_tested = data['transforms_tested']

pos = pd.DataFrame(data=['%.1f_%.1f' % (x, y) for x, y, in zip(transforms['xpos'], transforms['ypos'])], columns=['pos'], index=transforms.index)
transforms = pd.concat([transforms, pos], axis=1)


df = transforms[['object', 'zscore', 'pos']]


def eta_squared(aov):
    aov['eta_sq'] = 'NaN'
    aov['eta_sq'] = aov[:-1]['sum_sq']/sum(aov['sum_sq'])
    return aov
 
def omega_squared(aov):
    mse = aov['sum_sq'][-1]/aov['df'][-1]
    aov['omega_sq'] = 'NaN'
    aov['omega_sq'] = (aov[:-1]['sum_sq']-(aov[:-1]['df']*mse))/(sum(aov['sum_sq'])+mse)
    return aov

metric = 'zscore'
object_list = df['object'].unique()
for curr_object in object_list:
    #F, p = stats.shapiro(df[metric][df['object'] == curr_object])
    F, p = stats.kstest(df[metric][df['object'] == curr_object], 'norm')
    print "%s: (TS: %.2f, p=%.2f) " % (curr_object, F, p)

stats.levene(df[metric][df['object'] == object_list[0]], #, 
             df[metric][df['object'] == object_list[1]],
             df[metric][df['object'] == object_list[2]])


formula = 'zscore ~ C(object) + C(pos) + C(object):C(pos)'
model = ols(formula, df).fit()
model.summary()

aov_table = statsmodels.stats.anova.anova_lm(model, typ=2)
eta_squared(aov_table)
omega_squared(aov_table)
print aov_table

res = model.resid 
fig = sm.qqplot(res, line='s')
pl.show()


#%%

        for trans_ix, transform in enumerate(self.blobs['transforms_tested']):
            tix = aix + trans_ix
            plot_list = []
            for roi, df in rois:
                if roi not in rois_to_plot:
                    continue
                
                df2 = df.pivot_table(index='object', columns=transform, values=self.blobs['metric'])
                #new_df = pd.concat([df2, pd.Series(data=[roi for _ in range(df2.shape[0])], index=df2.index, name='roi')], axis=1)
                plot_list.append(df2)
                
            data = pd.concat(plot_list, axis=0)
            
