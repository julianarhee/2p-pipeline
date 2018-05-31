#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 20:36:25 2018

@author: juliana
"""

from pipeline.python.paradigm import align_acquisition_events as acq

    # Get paradigm/AUX info:
    # =============================================================================
    paradigm_dir = os.path.join(run_dir, 'paradigm')
    trial_info = acq.get_alignment_specs(paradigm_dir, si_info, iti_pre)

    print "-------------------------------------------------------------------"
    print "Getting frame indices for trial epochs..."
    parsed_frames_filepath = acq.assign_frames_to_trials(si_info, trial_info, paradigm_dir, create_new=create_new)

    # Get all unique stimulus configurations:
    # =========================================================================
    print "-----------------------------------------------------------------------"
    print "Getting stimulus configs..."
    configs, stimtype = acq.get_stimulus_configs(trial_info)
    
    run_info, stimconfigs, labels_df, raw_df = util.get_run_details(options)

    # Set up output dir:
    output_basedir = os.path.join(run_info['traceid_dir'],  'classifiers')
    if not os.path.exists(output_basedir):
        os.makedirs(output_basedir)
    if not os.path.exists(os.path.join(output_basedir, 'figures')):
        os.makedirs(os.path.join(output_basedir, 'figures'))
        
    # Also create output dir for population-level figures:
    population_figdir = os.path.join(run_info['traceid_dir'],  'figures', 'population')
    if not os.path.exists(population_figdir):
        os.makedirs(population_figdir)

    # Get processed traces:
    _, traces_df, F0_df = util.load_roiXtrials_df(traceid_dir, trace_type='processed', dff=False, smoothed=False)
    _, dff_df, _ = util.load_roiXtrials_df(traceid_dir, trace_type='processed', dff=True, smoothed=False)

    # Check data:
    fig = pl.figure(figsize=(80, 20))
    roi = 'roi00001'
    pl.plot(raw_df[roi], label='raw')
    pl.plot(F0_df[roi], label='drift')
    pl.plot(traces_df[roi], label='corrected')
    pl.legend()
    pl.savefig(os.path.join(traceid_dir, '%s_drift_correction.png' % roi))
    
            
    # Smooth traces: ------------------------------------------------------
    util.test_file_smooth(traceid_dir, use_raw=False, ridx=0, fmin=0.001, fmax=0.02, save_and_close=False, output_dir=output_basedir)
    frac = 0.02
    # ---------------------------------------------------------------------
    
    _, smoothed_df, _ = util.load_roiXtrials_df(traceid_dir, trace_type='processed', dff=True, smoothed=True, frac=frac)
    _, smoothed_X, _ = util.load_roiXtrials_df(traceid_dir, trace_type='processed', dff=False, smoothed=True, frac=frac)


    # Check smoothing
    fig = pl.figure(figsize=(80, 20))
    roi = 'roi00001'
    pl.plot(dff_df[roi], label='df/f')
    pl.plot(smoothed_df[roi], label='smoothed')
    pl.legend()
    pl.savefig(os.path.join(traceid_dir, '%s_dff_smoothed.png' % roi))
   
    fig = pl.figure(figsize=(80, 20))
    roi = 'roi00001'
    pl.plot(traces_df[roi], label='raw (F0)')
    pl.plot(smoothed_X[roi], label='smoothed')
    pl.legend()
    pl.savefig(os.path.join(traceid_dir, '%s_dff_smoothed.png' % roi))
   

    # Get label info:
    sconfigs = util.format_stimconfigs(stimconfigs)

    ylabels = labels_df['config'].values
    groups = labels_df['trial'].values
    tsecs = labels_df['tsec']


#%%
    ridx = 0
    roi_id = 'roi%05d' % ridx
    trace_type = 'smoothedDF' #corrected'
    assert trace_type in dataset.keys()
    xdata = dataset[trace_type]
    ydata = dataset['ylabels']
    tsecs = dataset['tsecs']

    run_info = dataset['run_info'][()]
    nframes_per_trial = run_info['nframes_per_trial']
    ntrials_by_cond = run_info['ntrials_by_cond']
    ntrials_total = sum([val for k,val in ntrials_by_cond.iteritems()])
    #trial_labels = np.reshape(ydata, (ntrials_total, nframes_per_trial))[:,0]

    # Get stimulus info:
    sconfigs = dataset['sconfigs'][()]
    transform_dict, object_transformations = util.get_transforms(sconfigs)
    trans_types = [trans for trans in transform_dict.keys() if len(transform_dict[trans]) > 1]

    # Get trial and timing info:
    trials = np.hstack([np.tile(i, (nframes_per_trial, )) for i in range(ntrials_total)])
    tsecs = np.reshape(tsecs, (nframes_per_trial*ntrials_total,))
    print tsecs.shape
    
    # Create raw dataframe:
    new_columns=[]
    for trans in trans_types:
        trans_vals = [sconfigs[c][trans] for c in ydata]
        new_columns.append(pd.DataFrame(data=trans_vals, columns=[trans], index=xrange(len(ydata))))
    new_columns.append(pd.DataFrame(data=trials, columns=['trial'], index=xrange(len(ydata))))
    new_columns.append(pd.DataFrame(data=tsecs, columns=['tsec'], index=xrange(len(ydata))))
    
    config_df = pd.concat(new_columns, axis=1)
    roi_df = pd.DataFrame(data=xdata[:,ridx], columns=[trace_type], index=xrange(len(ydata)))
    df = pd.concat([roi_df, config_df], axis=1).reset_index(drop=True)

    if len(trans_type) == 1:
        stim_grid = (transform_dict[trans_types[0]],)
    elif len(trans_type) == 2:
        stim_grid = (trans_type[0], trans_type[1])

    ncols = len(stim_grid[0])
    columns = trans_types[0]
    col_order = sorted(stim_grid[0])
    nrows = 1; rows = None; row_order=None
    if len(stim_grid) == 2:
        nrows = len(stim_grid[1])
        rows = stim_grid[1]
        row_order = sorted(stim_grid[1])
        

    g1 = sns.FacetGrid(df, row=rows, col=columns, sharex=True, sharey=True, hue='trial', row_order=row_order, col_order=col_order)
    g1.map(pl.plot, "tsec", trace_type, linewidth=0.2, color='k', alpha=0.5)
    
    
    #%%
    
    tracemat = np.reshape(xdata[:, ridx], (ntrials_total, nframes_per_trial))
    print tracemat.shape
    
    tpoints = np.reshape(tsecs, (ntrials_total, nframes_per_trial))[0,:]
    labeled_trials = np.reshape(ydata, (ntrials_total, nframes_per_trial))[:,0]
    sconfigs_df = pd.DataFrame(sconfigs).T
    sgroups = sconfigs_df.groupby(trans_types)

    sns.set_style('white')
    fig, axes = pl.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(15,3))
    axes = axes.flat    
    traces_list = []
    pi = 0
    for k,g in sgroups:
        curr_configs = g.index.tolist()
        config_ixs = [ci for ci,cv in enumerate(labeled_trials) if cv in curr_configs]
        subdata = tracemat[config_ixs, :]
        trace_mean = np.mean(subdata, axis=0)
        trace_sem = stats.sem(subdata, axis=0)
        
        axes[pi].plot(tpoints, trace_mean, color='k', linewidth=2)
        axes[pi].fill_between(tpoints, trace_mean-trace_sem, trace_mean+trace_sem, color='k', alpha=0.2)
        
        start_val = tsecs[run_info['stim_on_frame']]
        end_val = tsecs[run_info['stim_on_frame'] + int(round(run_info['nframes_on']))]
        pi += 1

    sns.despine(offset=4, trim=True)
    for ax in axes[2:]:
        ax.yaxis.offsetText.set_visible(False)
        
        ax.set_yticks(())
        ax.set_yticklabels(())
        ax.set_ylabel('')

        
#        traces_list.append(pd.DataFrame({'mean': trace_mean,
#                                         'sem': trace_sem,
#                                         'config': k}))
#    tracedf = pd.concat(traces_list, axis=0).reset_index(drop=True)
    
    for ax in axes.flat:
        ax.plot()
    
    axes[pi].plot(tpoints, preds_mean, color=colorvals[ci], label=svc.classes_[ci], linewidth=2)
    axes[pi].fill_between(tpoints, preds_mean-preds_sem, preds_mean+preds_sem, color=colorvals[ci], alpha=0.2)
    
        
    
    #plotstats = get_facet_stats(config_list, g1, value='df')
    pl.subplots_adjust(top=0.78)

    nrows = len(g1.row_names)
    ncols = len(g1.col_names)
    for ri, rowval in enumerate(g1.row_names):
        for ci, colval in enumerate(g1.col_names):
            currax = g1.facet_axis(ri, ci)
            configDF = subDF[((subDF[rows]==rowval) & (subDF[columns]==colval))]
            if len(configDF)==0:
                continue
            dfmat = []
            tmat = []
            for trial in list(set(configDF['trial'])):
#                if trial in exclude_trials:
#                    continue
                dftrace = np.array(configDF[configDF['trial']==trial]['df'])
                dfmat.append(np.array(configDF[configDF['trial']==trial]['df']))
                tmat.append(np.array(configDF[configDF['trial']==trial]['tsec']))
            dfmat = np.array(dfmat); tmat = np.array(tmat);
            curr_ntrials = dfmat.shape[0]
            # Plot MEAN DF trace:
            mean_df = np.mean(dfmat, axis=0)
            mean_tsec = np.mean(tmat, axis=0)
            # Set y-value for stimulus-bar position:
            if ri == 0 and ci == 0:
                stimbar_pos = np.nanmin(dfmat) - np.nanmin(dfmat) *-.25
            # Plot:
            currax.plot(mean_tsec, mean_df, trace_color, linewidth=1, alpha=1)
            currax.plot([mean_tsec[int(first_on)], mean_tsec[int(first_on)]+nsecs_on], [stimbar_pos, stimbar_pos], stimbar_color, linewidth=2, alpha=1)
            currax.annotate("n = %i" % curr_ntrials, xy=get_axis_limits(currax, xscale=0.2, yscale=0.8))

    pl.subplots_adjust(top=0.78)
    g1.fig.suptitle("%s - stim %s" % (roi, objectid))
    
    
    
    
    
\
        
    #tracevec = xdata[:, ridx]
    
    
    psth_from_full_trace(roi, tracevec, mean_tsecs, ntrials_total, nframes_per_trial,
                              color_codes=color_codes, conditions=(row_configs, column_configs),
                              stim_on_frame=stim_on_frame, nframes_on=nframes_on,
                              plot_legend=True, as_percent=True,
                              save_and_close=True, roi_psth_dir=roi_psth_dir)


def psth_from_full_trace(roi, tracevec, mean_tsecs, ntrials_total, nframes_per_trial,
                                  color_codes=None, conditions=None,
                                  stim_on_frame=None, nframes_on=None,
                                  plot_legend=True, plot_average=True, as_percent=False,
                                  roi_psth_dir='/tmp', save_and_close=True):

    '''Pasre a full time-series (of a given run) and plot as stimulus-aligned
    PSTH for a given ROI.
    '''

    pl.figure()
    traces = np.reshape(tracevec, (ntrials_total, nframes_per_trial))
    
    ncols = len(stim_grid[0])
    nrows = 1
    if len(stim_grid) == 2:
        nrows = len(stim_grid[1])
        
    
    fig, axes = pl.subplots(nrows, ncols, figsize=(10,6))
    
    

    if as_percent:
        multiplier = 100
        units_str = ' (%)'
    else:
        multiplier = 1
        units_str = ''

    if color_codes is None:
        color_codes = sns.color_palette("Greys_r", nr*2)
        color_codes = color_codes[0::2]
    if orientations is None:
        orientations = np.arange(0, nr)

    for c in range(traces.shape[0]):
        pl.plot(mean_tsecs, traces[c,:] * multiplier, c=color_codes[c], linewidth=2, label=orientations[c])

    if plot_average:
        pl.plot(mean_tsecs, np.mean(traces, axis=0)*multiplier, c='r', linewidth=2.0)
    sns.despine(offset=4, trim=True)

    if stim_on_frame is not None and nframes_on is not None:
        stimbar_loc = traces.min() - (0.1*traces.min()) #8.0

        stimon_frames = mean_tsecs[stim_on_frame:stim_on_frame + nframes_on]
        pl.plot(stimon_frames, stimbar_loc*np.ones(stimon_frames.shape), 'g')

    pl.xlabel('tsec')
    pl.ylabel('mean df/f%s' % units_str)
    pl.title(roi)

    if plot_legend:
        pl.legend(orientations)

    if save_and_close:
        pl.savefig(os.path.join(roi_psth_dir, '%s_psth_mean.png' % roi))
        pl.close()





