#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 14:47:49 2019

@author: julianarhee
"""

import glob
import os
import copy
import json

import numpy as np
import  pylab as pl
import pandas as pd
import cPickle as pkl

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from pipeline.python.utils import label_figure, natural_keys

# V1 -------------------------------------------------------- # JC076: 20190420, *20190501 

# JC083: 20190507, 20190510, 20190511
# JC084: 20190522
# JC085:  20190622
# JC097: 20190613, 20190616, 20190617

# LM --------------------------------------------------------
# JC078:  20190504, 20190509
# JC080:  20190506, 20190603
# JC083:  20190508, 20190512, 20190517
# JC084:  20190525
# JC091:  20190627
# JC097:  20190618

# Li --------------------------------------------------------
# JC076: 20190502
# JC090: 20190605
# JC091:  '20190602', '20190606', '20190607', '20190614'
# JC099: 20190609, 20190612, 20190617
# ------------------------------------------------------------



correlation_dict = dict()
correlation_dict = {'size': {},
                    'morph': {},
                    'features': {}}


#%%


rootdir = '/n/coxfs01/2p-data'
#session = '20190626' #'20190319'
#fov = 'FOV1_zoom2p0x' 
run = 'combined_blobs_static'
traceid = 'traces001' #'traces002'
visual_area = 'V1'
segment = False

animalid = 'JC083' 

with open(os.path.join(rootdir, animalid, 'sessionmeta.json'), 'r') as f:
    sessionmeta = json.load(f)

session_keys = [k for k, v in sessionmeta.items() if v['state']==state and v['visual_area']==visual_area]

session_list = ['20190618'] #, '20190512', '20190517']


for session in session_list:
    
    fov_dir = os.path.join(rootdir, animalid, session, fov)
    traceid_dir = glob.glob(os.path.join(fov_dir, run, 'traces', '%s*' % traceid))[0]
    data_fpath = glob.glob(os.path.join(traceid_dir, 'data_arrays', '*.npz'))[0]
    dset = np.load(data_fpath)
    dset.keys()
    
    data_identifier = '|'.join([animalid, session, fov, run, traceid, visual_area])
    print data_identifier
    
    
    #%
    
    included_rois = []
    if segment:
        segmentations = glob.glob(os.path.join(fov_dir, 'visual_areas', '*.pkl'))
        assert len(segmentations) > 0, "Specified to segment, but no segmentation file found in acq. dir!"
        if len(segmentations) == 1:
            segmentation_fpath = segmentations[0]
        else:
            for si, spath in enumerate(sorted(segmentations, key=natural_keys)):
                print si, spath
            sel = input("Select IDX of seg file to use: ")
            segmentation_fpath = sorted(segmentations, key=natural_keys)[sel]
        with open(segmentation_fpath, 'rb') as f:
            seg = pkl.load(f)
                
        included_rois = seg.regions[visual_area]['included_rois']
        print "Found %i rois in visual area %s" % (len(included_rois), visual_area)
    
    #%
    # Load roi stats:  
    sorted_dir = sorted(glob.glob(os.path.join(traceid_dir, 'response_stats*')))[-1]
    print "Selected stats results: %s" % os.path.split(sorted_dir)[-1]
    
    stats_fpath = glob.glob(os.path.join(sorted_dir, 'roistats_results.npz'))[0]
    rstats = np.load(stats_fpath)
    #rstats.keys()
    
    #%
    if segment and len(included_rois) > 0:
        all_rois = np.array(copy.copy(included_rois))
    else:
        all_rois = np.arange(0, rstats['nrois_total'])
    
    visual_rois = np.array([r for r in rstats['sorted_visual'] if r in all_rois])
    selective_rois = np.array([r for r in rstats['sorted_selective'] if r in all_rois])
    
    print "Found %i cells that pass responsivity test (%s, p<%.2f)." % (len(visual_rois), rstats['responsivity_test'], rstats['visual_pval'])
    print "Found %i cells that pass responsivity test (%s, p<%.2f)." % (len(selective_rois), rstats['selectivity_test'], rstats['selective_pval'])
    
    #%
    
    # Load parsed data:
    trace_type = 'dff'
    traces = dset[trace_type]
    #zscores = dset['zscore']
    
    # Format condition info:
    aspect_ratio = 1 #1.747
    sdf = pd.DataFrame(dset['sconfigs'][()]).T
    
    if 'color' in sdf.columns:
        sdf = sdf[sdf['color']=='']
    sdf['size'] = [round(sz/aspect_ratio, 1) for sz in sdf['size']]
    sdf.head()
    
    
    # Only take subset of trials where image shown (not controls):
    labels = pd.DataFrame(data=dset['labels_data'], columns=dset['labels_columns'])
    
    
    #%
    
    # zscore the traces:
    # -----------------------------------------------------------------------------
    baselines = []
    zscores_list = []
    snr_list = []
    for trial, tmat in labels.groupby(['trial']):
        #print trial    
        stim_on_frame = tmat['stim_on_frame'].unique()[0]
        nframes_on = tmat['nframes_on'].unique()[0]
        curr_traces = traces[tmat.index, :]
        bas_std = curr_traces[0:stim_on_frame, :].std(axis=0)
        bas_mean = curr_traces[0:stim_on_frame, :].mean(axis=0)
        stim_mean = curr_traces[stim_on_frame:stim_on_frame+nframes_on, :].mean(axis=0)
        if trace_type == 'dff':
            curr_zs = (stim_mean) / bas_std
        else:
            curr_zs = (stim_mean - bas_mean) / bas_std
        curr_snr = stim_mean / bas_mean
        zscores_list.append(curr_zs)
        baselines.append(bas_mean)
        snr_list.append(curr_snr)
    
        
    #%
    use_zscore = True
    
    labels = labels[labels['config'].isin(sdf.index.tolist())] # Ignore "control" trials for now
    #traces = traces[labels.index.tolist(), :]
    trial_ixs = np.array([int(t[5:])-1 for t in sorted(labels['trial'].unique(), key=natural_keys)])
    #zscores = zscores[trial_ixs, :]
    
    
    
    #%
    # Sort ROIs by zscore by cond
    # -----------------------------------------------------------------------------
    snrs = np.array(snr_list)
    zscores = np.array(zscores_list)
    
    # Get single value for each trial and sort by config:
    trials_by_cond = dict()
    for k, g in labels.groupby(['config']):
        trials_by_cond[k] = sorted([int(tr[5:])-1 for tr in g['trial'].unique()])
    
    resp_by_cond = dict()
    for cfg, t_ixs in trials_by_cond.items():
        if use_zscore:
            resp_by_cond[cfg] = zscores[t_ixs, :]  # For each config, array of size ntrials x nrois
            response_type = 'zscore'
        else:
            resp_by_cond[cfg] = snrs[t_ixs, :]
            response_type = 'snr'
    response_thr = 2.0
        
    avg_resp_by_cond = pd.DataFrame([resp_by_cond[cfg].mean(axis=0) \
                                        for cfg in sorted(resp_by_cond.keys(), key=natural_keys)],\
                                        index=[int(cf[6:])-1 for cf in sorted(resp_by_cond.keys(), key=natural_keys)]) # nconfigs x nrois
    
                                     #%
                                
    # Sort mean (or max) zscore across trials for each config, and find "best config"
    visual_max_avg_zscore = np.array([avg_resp_by_cond[rid].max() for rid in visual_rois])
    visual_sort_by_max_zscore = np.argsort(visual_max_avg_zscore)[::-1]
    sorted_visual = visual_rois[visual_sort_by_max_zscore]
    
    selective_max_avg_zscore = np.array([avg_resp_by_cond[rid].max() for rid in selective_rois])
    selective_sort_by_max_zscore = np.argsort(selective_max_avg_zscore)[::-1]
    sorted_selective = selective_rois[selective_sort_by_max_zscore]
    
    print [r for r in sorted_selective if r not in sorted_visual]
    print sorted_selective[0:10]
    
    roi_selector = 'all'
    
    if roi_selector == 'visual':
        roi_list = np.array(sorted(copy.copy(sorted_visual)))
    elif roi_selector == 'selective':
        roi_list = np.array(sorted(copy.copy(sorted_selective)))
    else:
        roi_list = np.arange(0, avg_resp_by_cond.shape[-1])
        roi_selector = '%s_%.2f' % (response_type, response_thr)
        
    responsive_cells = np.array([i for i, iv in enumerate(roi_list) if avg_resp_by_cond[iv].max() > response_thr])
    responses = avg_resp_by_cond[roi_list[responsive_cells]]
    
    #%
    
    snrs = snrs[:, responsive_cells]
    snrs = snrs[trial_ixs, :]
    print(snrs.shape)
    
    
    zscores = zscores[:, responsive_cells]
    zscores = zscores[trial_ixs, :]
    print(zscores.shape)
    zscores = pd.DataFrame(zscores)
    
    
    
    #%
    fig_subdir = 'regr_%s_%s' % (trace_type, roi_selector )
    
    if segment:
        curr_figdir = os.path.join(traceid_dir, 'figures', 'population', fig_subdir, visual_area)
    else:
        curr_figdir = os.path.join(traceid_dir, 'figures', 'population', fig_subdir)
    if not os.path.exists(curr_figdir):
        os.makedirs(curr_figdir)
    print "Saving plots to: %s" % curr_figdir
    
    #%
    
    trial_configs = [g['config'].unique()[0] for k, g in labels.groupby(['trial', 'config'])]
    
    
    from sklearn.model_selection import train_test_split
    import scipy.stats as sp
    import seaborn as sns
    from sklearn import linear_model, preprocessing
    
    X = pd.DataFrame(preprocessing.StandardScaler().fit_transform(zscores))
    
    hue_var1 = 'morphlevel'
    regr_var1 = 'size'
    targets1 = [sdf[regr_var1][cfg] for cfg in trial_configs]
    
    xTrain1, xTest1, yTrain1, yTest1 = train_test_split(X, targets1, test_size = 0.2, random_state = 0)
    regr1 = linear_model.LinearRegression()
    regr1.fit(xTrain1, yTrain1)
    yPrediction1 = regr1.predict(xTest1)
    
    
    hue_var2 = 'size'
    regr_var2 = 'morphlevel'
    targets2 = [sdf[regr_var2][cfg] for cfg in trial_configs]
    
    xTrain2, xTest2, yTrain2, yTest2 = train_test_split(X, targets2, test_size = 0.2, random_state = 0)
    regr2 = linear_model.LinearRegression()
    regr2.fit(xTrain2, yTrain2)
    yPrediction2 = regr2.predict(xTest2)
    
    
    fig, axes = pl.subplots(1, 3, figsize=(20,6))
    ax = axes[0]
    hue_labels = [sdf[hue_var1][trial_configs[ti]] for ti in xTest1.index.tolist()]
    hue_vals = sorted(np.unique(hue_labels))
    g = sns.swarmplot(yTest1, yPrediction1, hue=hue_labels, ax=ax, palette=sns.color_palette("Blues", len(hue_vals)))
    g.legend(loc='center left', bbox_to_anchor=(0., -0.15), ncol=len(hue_vals), fontsize=6)
    ax.set_xlabel('actual')
    ax.set_ylabel('predicted')
    r, p = sp.pearsonr(yTest1, yPrediction1)
    ax.set_title('%s: R %.2f (p=%.2f)' % (regr_var1, r, p))
    correlation_dict['size']['%s_%s' % (animalid, session)] = r
    
    ax = axes[1]
    hue_labels = [sdf[hue_var2][trial_configs[ti]] for ti in xTest2.index.tolist()]
    hue_vals = sorted(np.unique(hue_labels))
    g = sns.swarmplot(yTest2, yPrediction2, hue=hue_labels, ax=ax, palette=sns.color_palette("Blues", len(hue_vals)))
    g.legend(loc='center left', bbox_to_anchor=(0.2, -0.15), ncol=len(hue_vals), fontsize=6)
    ax.set_xlabel('actual')
    ax.set_ylabel('predicted')
    r, p = sp.pearsonr(yTest2, yPrediction2)
    ax.set_title('%s: R %.2f (p=%.2f)' % (regr_var2, r, p))
    correlation_dict['morph']['%s_%s' % (animalid, session)] = r
    
    ax = axes[2]
    ax.scatter(regr1.coef_, regr2.coef_)
    r, p = sp.pearsonr(regr1.coef_, regr2.coef_)
    ax.set_title('coefs: R %.2f (p=%.2f)' % (r, p))
    correlation_dict['features']['%s_%s' % (animalid, session)] = r

    pl.subplots_adjust(bottom=0.2)
    
    label_figure(fig, data_identifier)
    
    pl.savefig(os.path.join(curr_figdir, 'corr_actual_v_predicted_train1feature.png'))



#%%


from sklearn import metrics
print("MSE: %.2f | Var. score: %.2f" % (metrics.mean_squared_error(yTest1, yPrediction1), metrics.r2_score(yTest1, yPrediction1)) )
print("MSE: %.2f | Var. score: %.2f" % (metrics.mean_squared_error(yTest2, yPrediction2), metrics.r2_score(yTest2, yPrediction2)) )
print("RMSE 1: %.2f | RMSE 2: %.2f" % (np.sqrt(metrics.mean_squared_error(yTest1, yPrediction1)),np.sqrt(metrics.mean_squared_error(yTest2, yPrediction2)) ) )


#%%


#### BREAK ####
V1_size_corrs = [0.84, 0.71, 0.76, 0.79, 0.78, 0.81, 0.81, 0.81]
V1_morph_corrs = [0.66, 0.56, 0.51, 0.46, 0.27, 0.27, 0.37, 0.37]
LI_size_corrs = [0.76, 0.65, 0.74, 0.6, 0.71, 0.74, 0.59, 0.65, 0.44]
LI_morph_corrs = [0.2, 0.18, 0.35, 0.14, 0.15, 0.08, 0.22, 0.07, 0.14]

LM_morph_corrs = [v for k, v in correlation_dict['morph'].items()]
LM_size_corrs = [v for k, v in correlation_dict['size'].items()]



size_corrs = copy.copy(V1_size_corrs) #[0.84, 0.71, 0.76, 0.79, 0.78, 0.81, 0.81, 0.81]
size_corrs.extend(LM_size_corrs)
size_corrs.extend(LI_size_corrs)

area_names = ['V1' for _ in range(len(V1_size_corrs))]
area_names.extend(['LM' for _ in range(len(LM_size_corrs))])
area_names.extend(['LI' for _ in range(len(LI_size_corrs))])


morph_corrs = copy.copy(V1_morph_corrs)
morph_corrs.extend(LM_morph_corrs)
morph_corrs.extend(LI_morph_corrs)

corrs = copy.copy(size_corrs)
corrs.extend(morph_corrs)

visual_area = copy.copy(area_names)
visual_area.extend(area_names)

transform = ['size' for _ in range(len(size_corrs))]
transform.extend(['morph' for _ in range(len(morph_corrs))])

df = pd.DataFrame({'area': visual_area,
                   'corrs': corrs,
                   'transform': transform})
df['area'] = df['area'].astype('category')
df['transform'] = df['transform'].astype('category')

sns.catplot(x="transform", y="corrs", hue="area", kind="swarm", data=df)
sns.catplot(x="area", y="corrs", hue="area", col="transform", kind="swarm", data=df)


                
#%%
param = 'size'
#ntrials, nrois = zscores.shape
#zscores_r = np.ravel(zscores)
#nlabels_r = np.ravel(np.array([np.tile(sdf[param][trial_configs[ti]], nrois) for ti in zscores.index.tolist()]))
ntrials, nrois = responses.shape
zscores_r = np.ravel(responses)
nlabels_r = np.ravel(np.array([np.tile(sdf[param]['config%03d' % int(ti+1)], nrois) for ti in responses.index.tolist()]))


fig, ax = pl.subplots()
ax.scatter(nlabels_r, zscores_r)
r, p = sp.pearsonr(zscores_r, nlabels_r)
ax.set_title('%s: R %.2f (p=%.2f)' % (param, r, p))




#%%

  

sizes = sorted([int(round(s)) for s in sdf['size'].unique()])
morphlevels = sorted(sdf['morphlevel'].unique())


#%%

indep_var = 'size'
dep_var = 'morphlevel'

#%%

rdf =  responses.T

yvals_list = []
indep_vals = sdf[indep_var].unique()
dep_vals = sdf[dep_var].unique()
for ival in dep_vals:
    curr_cfg_ixs = [int(c[6:])-1 for c in sdf.index.tolist() if sdf[dep_var][c]==ival]
    curr_df = rdf[curr_cfg_ixs]
    yvals = pd.DataFrame(data=np.ravel(curr_df), columns=[ival])
    target = np.tile([sdf[indep_var]['config%03d' % int(ci+1)] for ci in curr_cfg_ixs], rdf.shape[0])
    #labels = np.tile([sdf[dep_var]['config%03d' % int(ci+1)] for ci in curr_cfg_ixs], rdf.shape[0])
    yvals_list.append(yvals)
    
df = pd.concat(yvals_list, axis=1)
df['target'] = target


df.plot(x='target', y=10.0, style='o')


X_train = df[dep_vals]#.values.reshape(-1,2)
y_train = df['target']


X_train = np.random.rand(responses.shape[0], responses.shape[1]) #np.ones_like(responses.values)

#%%

X_train = responses.copy()
y_train1 =[sdf['morphlevel']['config%03d' % int(ci+1)] for ci in responses.index.tolist()]
y_train2 =[sdf['size']['config%03d' % int(ci+1)] for ci in responses.index.tolist()]

regr1 = LinearRegression()
regr1.fit(X_train, y_train1)

regr2 = LinearRegression()
regr2.fit(X_train, y_train2)

pl.scatter(regr1.coef_, regr2.coef_)

fig, ax = pl.subplots()
for vi in np.unique(y_train2):
    cfg_ixs = [int(ci[6:])-1 for ci in sdf.index.tolist() if sdf['size'][ci]==vi]
    curr_vals1 = X_train.loc[cfg_ixs].dot(regr1.coef_) #np.ravel(X_train.loc[cfg_ixs])
    ax.scatter(curr_vals1, curr_vals2)
    

w = responses.dot(regr.coef_) + regr.intercept_

pl.figure()
pl.scatter(y_labels, w)

for idx, col_name in enumerate(X_train.columns):
    print("The coefficient for {} is {}".format(col_name, regr.coef_[idx]))

#%%


X_train = rdf['snr'].values.reshape(1, -1)
y_train = rdf['size'].values.reshape(1, -1)

rdf = pd.DataFrame({'snr': np.ravel(responses, order='F'),
                    'rid': np.ravel([np.tile(c, 45) for c in responses.columns.tolist()]),
                    'morph': np.ravel([[sdf['morphlevel']['config%03d' % int(ci+1)] for ci in responses.index.tolist()] for _ in range(nrois)]),
                    'size': np.ravel([[sdf['size']['config%03d' % int(ci+1)] for ci in responses.index.tolist()] for _ in range(nrois)]) })

sns.swarmplot(x='morph', y='size', hue='snr', data=rdf, dodge=True) #, legend=False)
    
regr = LinearRegression()
regr.fit(X_train, y_train)


pl.figure()
pl.scatter(y_train, X_train)
pl.plot(y_train, regr.predict(X_train))

#%%

X = responses.copy()
target = np.array([sdf['size']['config%03d' % int(ci+1)] for ci in responses.index.tolist()])
label = np.array([sdf['morphlevel']['config%03d' % int(ci+1)] for ci in responses.index.tolist()])
    
regr = LinearRegression()
regr.fit(X, target)

pl.figure()
for ri in responses.columns.tolist():
    #vmat = np.reshape(responses[ri], (9, 5))
    pl.scatter(target, responses[ri].values, c=label, cmap='jet')


pl.plot(target, regr.predict(X))



#%%
yvals = np.ravel(responses)
xvals = np.ravel(pd.DataFrame([np.tile(sdf[indep_var]['config%03d' % int(ci+1)], responses.shape[-1])\
                               for ci in responses.index.tolist()]))

#%%
resp_type = 'snr'

df  = pd.DataFrame({'%s' % resp_type: yvals,
                   '%s' % indep_var: xvals})


X = df[resp_type].values.reshape(-1, 1)
Y = df[indep_var].values
X_std = preprocessing.StandardScaler().fit_transform(X)


#%%
from sklearn import linear_model, preprocessing

regr = linear_model.LinearRegression()
regr.fit(X, Y)


print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

pl.figure()
pl.scatter(Y, X)
pl.plot(Y, regr.predict(X))

#%%


lrg = LinearRegression()


y_size = np.array([sdf['size']['config%03d' % int(cix+1)] for cix in avg_zscores_by_cond.index.tolist()]) #np.array([sdf['size'][cfg] for cfg in sdf.index.tolist()])
y = np.array([y_size for _ in range(X_std.shape[-1])]).shape

lrg.fit(X, Y)

pl.scatter(X_std, y_size, color = 'red')
pl.plot(X_std, lrg.predict(X_std), color = 'blue')