#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 19:52:41 2018

@author: juliana
"""

#% HSITOGRAM:   Get max zscore across all configs for each ROI:
# -------------------------------------------------------------------------

roistats_filepath = '/mnt/odyssey/CE074/20180215/FOV1_zoom1x_V1/gratings_phasemod/traces/traces004_c04dde/metrics/pupil_size30-dist8-blinks1_12575665366856094372/roi_stats_12575665366856094372_20180309184424.hdf5'
roidata_filepath  = '/mnt/odyssey/CE074/20180215/FOV1_zoom1x_V1/gratings_phasemod/traces/traces004_c04dde/ROIDATA_784044.hdf5'

# Reformat DATA stuct of old data:
DATA = pd.HDFStore(roidata_filepath, 'r')
df_list = []
for roi in DATA.keys():
    if '/' in roi:
        roiname = roi[1:]
    else:
        roiname = roi
    dfr = DATA[roi]
    dfr['roi'] = pd.Series(np.tile(roiname, (len(dfr .index),)), index=dfr.index)
    df_list.append(dfr)
DATA = pd.concat(df_list, axis=0, ignore_index=True)
transform_dict, object_transformations = vis.get_object_transforms(DATA)
trans_types = object_transformations.keys()

# Load STATS:
STATS = pd.HDFStore(roistats_filepath, 'r')['/df']

roi_list = sorted(list(set(STATS['roi'])), key=natural_keys)
transform_dict, object_transformations = vis.get_object_transforms(DATA)
trans_types = object_transformations.keys()

if 'mean_%s' % metric_type not in STATS.keys():
    # Get stats on ROIs:
    group_vars = ['roi']
    trans_types = object_transformations.keys()
    group_vars.extend([t for t in trans_types])
    grouped = STATS.groupby(group_vars, as_index=False)         # Group dataframe by variables-of-interest

    # metric summaries to add:
    metrics = ['zscore', 'stim_df']
    for metric_type in metrics:
        zscores = grouped[metric_type].mean()                                                # Get mean of 'metric_type' for each combination of transforms
        zscores['sem_%s' % metric_type] = grouped[metric_type].aggregate(stats.sem)[metric_type]             # Get SEM
        zscores = zscores.rename(columns={metric_type: 'mean_%s' % metric_type})                # Rename 'zscore' column to 'mean_zscore' so we can merge
        STATS = STATS.merge(zscores)#.sort_values([xval_trans])                         # Merge summary stats to each corresponding row (indexed by columns values in that row)

    # Update STATS dataframe on disk:
    STATS.to_hdf(stats_filepath, datakey,  mode='r+')





roi_list = sorted(list(set(STATS['roi'])), key=natural_keys)

metric_type = 'zscore'
max_config_zscores = [max(list(set(STATS[STATS['roi']==roi]['mean_%s' % metric_type]))) for roi in roi_list]
pl.figure()
sns.distplot(max_config_zscores, kde=False) #, hist=False, kde=False, norm_hist=True) #, kde=False, norm_hist=True, bins=20, fit=norm)
#np.hist(max_config_zscores, bins=50, normed=True)
pl.xlabel('max zscore')
pl.title("%s %s %s %s" % (animalid, session, fov, stimulus))

curr_tuning_dir = os.path.join(combined_tracedir, 'figures', 'tuning')
figname = "hist_rois_max_%s_%s_%s.png" % (trace_type, metric_type, selected_metric)
figpath = os.path.join(curr_tuning_dir, figname)
pl.savefig(figpath)
pl.close()
