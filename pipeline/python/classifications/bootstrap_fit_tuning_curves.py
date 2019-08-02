
# coding: utf-8

# In[2]:



import datetime
import os
import cv2
import glob
import h5py
import sys
import optparse
import copy

import pylab as pl
from collections import Counter
import seaborn as sns
import cPickle as pkl
import numpy as np
import pylab as pl
import pandas as pd
import seaborn as sns
import tifffile as tf

from pipeline.python.classifications import osi_dsi as osi
from pipeline.python.classifications import test_responsivity as resp #import calculate_roi_responsivity, group_roidata_stimresponse, find_barval_index
#from pipeline.python.classifications import osi_dsi as osi
#from pipeline.python.visualization import get_session_summary as ss
from pipeline.python.utils import natural_keys, label_figure

from pipeline.python.retinotopy import fit_2d_rfs as rf

from pipeline.python.utils import uint16_to_RGB
from skimage import exposure
from matplotlib import patches

from scipy import stats
from scipy.interpolate import interp1d
import scipy.optimize as spopt

#%%

# #############################################################################
# Fitting functions:
# #############################################################################

def get_init_params(response_vector):
    theta_pref = response_vector.idxmax()
    theta_null = (theta_pref + 180) % 360.
    r_pref = response_vector.loc[theta_pref]
    r_null = response_vector.loc[theta_null]
    sigma = np.mean(np.diff([response_vector.index.tolist()]))
    non_prefs = [t for t in response_vector.index.tolist() if t not in [theta_pref, theta_null]]
    r_offset = np.mean([response_vector.loc[t] for t in non_prefs])
    return r_pref, r_null, theta_pref, sigma, r_offset


def angdir180(x):
    '''wraps anguar diff values to interval 0, 180'''
    return min(np.abs([x, x-360, x+360]))

def double_gaussian( x, c1, c2, mu, sigma, C ):
    #(c1, c2, mu, sigma) = params
    x1vals = np.array([angdir180(xi - mu) for xi in x])
    x2vals = np.array([angdir180(xi - mu - 180 ) for xi in x])
    res =   C + c1 * np.exp( - x1vals**2.0 / (2.0 * sigma**2.0) )             + c2 * np.exp( - x2vals**2.0 / (2.0 * sigma**2.0) )

#    res =   C + c1 * np.exp( - ((x - mu) % 360.)**2.0 / (2.0 * sigma**2.0) ) \
#            + c2 * np.exp( - ((x + 180 - mu) % 360.)**2.0 / (2.0 * sigma**2.0) )

#        res =   c1 * np.exp( - (x - mu)**2.0 / (2.0 * sigma**2.0) ) \
#                #+ c2 * np.exp( - (x - mu2)**2.0 / (2.0 * sigma2**2.0) )
    return res


def fit_direction_selectivity(x, y, init_params=[0, 0, 0, 0, 0], bounds=[np.inf, np.inf, np.inf, np.inf, np.inf]):
    roi_fit = None
    
    popt, pcov = spopt.curve_fit(double_gaussian, x, y, p0=init_params, maxfev=1000, bounds=bounds)
    fitr = double_gaussian( x, *popt)
        
    # Get residual sum of squares 
    residuals = y - fitr
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - (ss_res / ss_tot)
    
    if pcov.max() == np.inf: # or r2 == 1: #round(r2, 3) < 0.15 or 
        success = False
    else:
        success = True
        
    if success:
        roi_fit = {'pcov': pcov,
                     'popt': popt,
                     'fit_y': fitr,
                     'r2': r2,
                     #'x': x,
                     #'y': y,
                     'init': init_params,
                     'success': success}
    return roi_fit


def interp_values(response_vector, n_intervals=3, wrap_value=None):
    resps_interp = []
    rvectors = copy.copy(response_vector)
    if wrap_value is not None:
        rvectors = np.append(response_vector, wrap_value)
    for orix, rvec in enumerate(rvectors[0:-1]):
        if rvec == rvectors[-2]:
            resps_interp.extend(np.linspace(rvec, rvectors[orix+1], endpoint=True, num=n_intervals+1))
        else:
            resps_interp.extend(np.linspace(rvec, rvectors[orix+1], endpoint=False, num=n_intervals))          
    return resps_interp

#%%
def bootstrap_tuning_curves(gdf, roi_list=[]):

    roi_fits = []
    for ri, roi in enumerate(roi_list): #[30, 91, 93, 151]:
        fit_results = []

        # -----------------
        if ri % 20 == 0:
            print("...fitting %i of %i rois" % (ri, len(roi_list)))

        roi_df = gdf.get_group(roi)
        #metric_abs = np.abs(roi_df[response_type])
        #roi_df[metric_type] = metric_abs

        # Find best config:
        best_cfg = roi_df.groupby(['config']).mean()[metric_type].idxmax()
        best_cfg_params = stimdf.loc[best_cfg][[p for p in params if p!='ori']]
        curr_cfgs = sorted([c for c in stimdf.index.tolist() \
                            if all(stimdf.loc[c][[p for p in params if p!='ori']] == best_cfg_params)],\
                            key = lambda x: stimdf['ori'][x])


        # Get all trials of current set of cfgs:
        trialdf = roi_df[roi_df['config'].isin(curr_cfgs)]
        rdf = trialdf[['config', 'trial', response_type]]
        grouplist = [group_configs(group, response_type) for config, group in rdf.groupby(['config'])]
        responses_df = pd.concat(grouplist, axis=1)

        # Bootstrap distN of responses (rand w replacement):
        bootdf = [responses_df.sample(60, replace=True).mean(axis=0) for ni in range(n_bootstrap_iters)]
        bootstrapped_df = pd.concat(bootdf, axis=1)
        bootstrapped_df.index = [sdf['ori'][c] for c in bootstrapped_df.index]

        # Find init params for tuning fits and set fit constraints:
        init_params = get_init_params(bootstrapped_df[0])
        r_pref, r_null, theta_pref, sigma, r_offset = init_params
        init_bounds = ([0, 0, -np.inf, sigma/2., -r_pref], [3*r_pref, 3*r_pref, np.inf, np.inf, r_pref])

        # Interpolate values for finer steps:
        asi=[];dsi=[];r2=[];
        for niter in bootstrapped_df.columns:
            oris_interp = interp_values(tested_oris, n_intervals=n_intervals_interp, wrap_value=360)
            resps_interp = interp_values(bootstrapped_df[niter], n_intervals=n_intervals_interp, wrap_value=bootstrapped_df[niter][0])


            init_params = get_init_params(bootstrapped_df[niter])
            r_pref, r_null, theta_pref, sigma, r_offset = init_params
            init_bounds = ([0, 0, -np.inf, sigma/2., -r_pref], [3*r_pref, 3*r_pref, np.inf, np.inf, r_pref])

            fail = False
            try:
                rfit = fit_direction_selectivity(oris_interp, resps_interp, init_params, bounds=init_bounds)
                asi_t = get_ASI(rfit['fit_y'][0:], oris_interp[0:])
                dsi_t = get_DSI(rfit['fit_y'][0:], oris_interp[0:])

                asi.append(asi_t)
                dsi.append(dsi_t)
                r2.append(rfit['r2'])

                rfit['x'] = oris_interp
                rfit['y'] = resps_interp
                rfit['n_intervals_interp'] = n_intervals_interp

                fit_results.append(rfit)

            except Exception as e:
                print(e)
                fail = True
                print("... skipping %i" % roi)
                break

        if len(fit_results) > 0 and make_plots:
            print("... plotting")
            fig, ax, ax1, ax2 = plot_roi_tuning_raw_and_fit(roi, responses_df, curr_cfgs,
                                                            df_traces, labels, stimdf, fit_results, trace_type='dff')
            curr_oris = sorted(sdf['ori'].unique())
            ax1.set_xticks(curr_oris)
            ax1.set_xticklabels(curr_oris)
            ax1.set_title('(sz %i, sf %.2f)' % (best_cfg_params['size'], best_cfg_params['sf']), fontsize=8)

            fig.suptitle('roi %i' % int(roi+1))
            label_figure(fig, data_identifier)

            #-------------------------

            pl.savefig(os.path.join(roi_fitdir_figures, 'roi%05d.png' % int(roi+1)))
            pl.close()

        if not fail:
            roi_fits.append(pd.DataFrame({'ASI': asi,
                                          'DSI': dsi,
                                          'r2': r2,
                                          'cell': [roi for _ in np.arange(0, len(asi))]}))

        #print("[%i] ASI: %.3f (+/- %.3f), DSI: %.3f (+/- %.3f) (bootstrap %i iter)" % (roi, np.mean(asi), np.std(asi), np.mean(dsi), np.std(dsi), n_bootstrap_iters))


    ### Save bootstrap fit results 

    fitdf = pd.concat(roi_fits, axis=0)
    fitdf.head()
    fitdf['cell'] = [str(i) for i in fitdf['cell'].values] #fitdf['cell'].astype('category')


    with open(tuning_fit_results_path, 'wb') as f:
        pkl.dump(fitdf, f, protocol=pkl.HIGHEST_PROTOCOL)

    return fitdf

#%%
# #############################################################################
# Metric calculations:
# #############################################################################

def group_configs(group, response_type):
    config = group['config'].unique()[0]
    group.index = np.arange(0, group.shape[0])

    return pd.DataFrame(data={'%s' % config: group[response_type]})

def get_ASI(response_vector, thetas):
    if np.max(thetas) > np.pi:
        thetas = [np.deg2rad(th) for th in thetas]
    asi = np.abs( np.sum( [theta_resp * np.exp((2j*2*np.pi*theta_val) / (2*np.pi))\
                           for theta_resp, theta_val in zip(response_vector, thetas)] ) ) / np.sum(np.abs(response_vector))
    return asi

def get_DSI(response_vector, thetas):
    if np.max(thetas) > np.pi:
        thetas = [np.deg2rad(th) for th in thetas]
    dsi = np.abs( np.sum( [theta_resp * np.exp((1j*2*np.pi*theta_val) / (2*np.pi))\
                           for theta_resp, theta_val in zip(response_vector, thetas)] ) ) / np.sum(np.abs(response_vector))
    return dsi


# In[21]:

# #############################################################################
# Plotting functions:
# #############################################################################

def cleanup_axes(axes_list, which_axis='y'):    
    for ax in axes_list: 
        if which_axis=='y':
            # get the yticklabels from the axis and set visibility to False
            for label in ax.get_yticklabels():
                label.set_visible(False)
            ax.yaxis.offsetText.set_visible(False)
        elif which_axis=='x':
            # get the xticklabels from the axis and set visibility to False
            for label in ax.get_xticklabels():
                label.set_visible(False)
            ax.xaxis.offsetText.set_visible(False)


def plot_psth_roi(roi, raw_traces, labels, curr_cfgs, stimdf,  trace_type='dff', fig=None, nr=1, nc=1, s_row=0, colspan=1):
    if fig is None:
        fig = pl.figure()

    pl.figure(fig.number)
        
    # ---------------------------------------------------------------------
    #% plot raw traces:
    mean_traces, std_traces, tpoints = osi.get_mean_and_std_traces(roi, raw_traces, labels, curr_cfgs, stimdf)

    ymin = (mean_traces - std_traces ).min()
    ymax = (mean_traces + std_traces ).max()
    for icfg in range(len(curr_cfgs)):
        ax = pl.subplot2grid((nr, nc), (s_row, icfg), colspan=colspan)
        ax.plot(tpoints, mean_traces[icfg, :], color='k')
        ax.set_xticks([tpoints[stim_on_frame], round(tpoints[stim_on_frame+nframes_on], 1)])
        ax.set_xticklabels(['', round(tpoints[stim_on_frame+nframes_on], 1)])
        ax.set_ylim([ymin, ymax])
        if icfg > 0:
            ax.set_yticks([]); ax.set_yticklabels([]);
            ax.set_xticks([]); ax.set_xticklabels([]);
            sns.despine(ax=ax, offset=4, trim=True, left=True, bottom=True)
        else:
            ax.set_ylabel(trace_type); ax.set_xlabel('time (s)');
            sns.despine(ax=ax, offset=4, trim=True)
        sem_plus = np.array(mean_traces[icfg,:]) + np.array(std_traces[icfg,:])
        sem_minus = np.array(mean_traces[icfg,:]) - np.array(std_traces[icfg,:])
        ax.fill_between(tpoints, sem_plus, y2=sem_minus, alpha=0.5, color='k')

    return fig, ax


def plot_tuning_curve_roi(curr_oris, curr_resps, curr_sems=None, response_type='dff',
                          fig=None, ax=None, nr=1, nc=1, colspan=1, s_row=0, s_col=0, color='k',
                         marker='o', lw=1, markersize=5):
    if fig is None:
        fig = pl.figure()
    
    pl.figure(fig.number)
        
    # Plot tuning curve:
    if ax is None:
        ax = pl.subplot2grid((nr, nc), (s_row, s_col), colspan=colspan)
    ax.plot(curr_oris, curr_resps, color=color, marker=marker, markersize=markersize, lw=lw)
    if curr_sems is not None:
        ax.errorbar(curr_oris, curr_resps, yerr=curr_sems, fmt='none', ecolor=color)
    ax.set_xticks(curr_oris)
    ax.set_xticklabels(curr_oris)
    ax.set_ylabel(response_type)
    #ax.set_title('(sz %i, sf %.2f)' % (best_cfg_params['size'], best_cfg_params['sf']), fontsize=8)
    #sns.despine(trim=True, offset=4, ax=ax)
    
    return fig, ax

def plot_tuning_polar_roi(curr_oris, curr_resps, curr_sems=None, response_type='dff',
                          fig=None, ax=None, nr=1, nc=1, colspan=1, s_row=0, s_col=0, color='k'):
    if fig is None:
        fig = pl.figure()
    
    pl.figure(fig.number)
    
    # Plot polar graph:
    if ax is None:
        ax = pl.subplot2grid((nr,nc), (s_row, s_col), colspan=colspan, polar=True)
    thetas = np.array([np.deg2rad(c) for c in curr_oris])
    radii = curr_resps.copy()
    thetas = np.append(thetas, np.deg2rad(curr_oris[0]))  # append first value so plot line connects back to start
    radii = np.append(radii, curr_resps[0]) # append first value so plot line connects back to start
    ax.plot(thetas, radii, '-', color=color)
    ax.set_theta_zero_location("N")
    ax.set_yticks([curr_resps.min(), curr_resps.max()])
    ax.set_yticklabels(['', round(curr_resps.max(), 1)])

    
    return fig, ax


# In[28]:


def plot_roi_tuning_raw_and_fit(roi, responses_df, curr_cfgs,
                                raw_traces, labels,stimdf, fit_results,
                               trace_type='dff'):

    fig = pl.figure(figsize=(12,8))
    fig.patch.set_alpha(1)
    nr=2; nc=8;
    s_row=0
    fig, ax = plot_psth_roi(roi, raw_traces, labels, curr_cfgs, stimdf, 
                            trace_type=trace_type,
                            fig=fig, nr=nr, nc=nc, s_row=0)
    ymin = np.min([0, ax.get_ylim()[0]])
    ax.set_ylim([ymin, ax.get_ylim()[1]])


    curr_oris = np.array([stimdf['ori'][c] for c in curr_cfgs])  
    curr_resps = responses_df.mean()
    curr_sems = responses_df.sem()
    fig, ax1 = plot_tuning_curve_roi(curr_oris, curr_resps, curr_sems=curr_sems, 
                                     response_type=trace_type,
                                     fig=fig, nr=nr, nc=nc, s_row=1, colspan=5,
                                     marker='o', markersize=5, lw=0)


    fig, ax2 = plot_tuning_polar_roi(curr_oris, curr_resps, curr_sems=curr_sems, 
                                     response_type=trace_type,
                                     fig=fig, nr=nr, nc=nc, s_row=1, s_col=6, colspan=2)


    if fit_results is not None:
        oris_interp = np.array([rfit['x'] for rfit in fit_results]).mean(axis=0)
        resps_interp = np.array([rfit['y'] for rfit in fit_results]).mean(axis=0)
        resps_interp_sem = stats.sem(np.array([rfit['y'] for rfit in fit_results]), axis=0)
        resps_fit = np.array([rfit['fit_y'] for rfit in fit_results]).mean(axis=0)
        n_intervals_interp = rfit['n_intervals_interp']

        fig, ax1 = plot_tuning_curve_roi(oris_interp[0:-n_intervals_interp], 
                                         resps_fit[0:-n_intervals_interp], 
                                         curr_sems=resps_interp_sem[0:-n_intervals_interp], 
                                         response_type=trace_type,color='cornflowerblue',
                                         markersize=0, lw=1, marker=None,
                                         fig=fig, ax=ax1, nr=nr, nc=nc, s_row=1, colspan=5)

        fig, ax2 = plot_tuning_polar_roi(oris_interp, 
                                         resps_fit, 
                                         curr_sems=resps_interp_sem, 
                                         response_type=trace_type, color='cornflowerblue',
                                         fig=fig, ax=ax2, nr=nr, nc=nc, s_row=1, s_col=6, colspan=2)
        
    ymin = np.min([0, ax1.get_ylim()[0]])
    ax1.set_ylim([ymin,  ax1.get_ylim()[1]])
    
    ax1.set_yticks([ymin, ax1.get_ylim()[1]])
    ax1.set_yticklabels([round(ymin, 2), round( ax1.get_ylim()[1], 2)])
    sns.despine(trim=True, offset=4, ax=ax1)

    
    if any([rfit['success'] for rfit in fit_results]):
        r2_avg = np.mean([rfit['r2'] for rfit in fit_results])
        ax1.text(0, ax1.get_ylim()[-1]*0.75, 'r2=%.2f' % r2_avg, fontsize=6)
    else:
        ax1.text(0, ax.get_ylim()[-1]*0.75, 'no fit', fontsize=6)
    
    return fig, ax, ax1, ax2



def compare_selectivity_all_fits(fitdf, fit_thr=0.9):
    
    strong_fits = [r for r, v in fitdf.groupby(['cell']) if v.mean()['r2'] >= fit_thr]
    print("%i out of %i cells with strong fits (%.2f)" % (len(strong_fits), len(fitdf['cell'].unique()), fit_thr))
    
    df = fitdf[fitdf['cell'].isin(strong_fits)]

    g = sns.PairGrid(df, hue='cell', vars=['ASI', 'DSI', 'r2'])
    g.fig.patch.set_alpha(1)
    
    
    g = g.map_offdiag(pl.scatter, marker='o',  alpha=0.5, s=1)
    
    
    g = g.map_diag(pl.hist, normed=True) #histtype="step",  
    
    g.set(ylim=(0, 1))
    g.set(xlim=(0, 1))
    
    #sns.distplot, kde=False, hist=True, rug=True,\
                   #hist_kws={"histtype": "step", "linewidth": 2, "alpha": 1.0})
    
    if df.shape[0] < 10:
        g = g.add_legend(bbox_to_anchor=(1.01,.5))
    
    pl.subplots_adjust(left=0.1, right=0.85)

    cleanup_axes(g.axes[:, 1:].flat, which_axis='y')
    cleanup_axes( g.axes[:-1, :].flat, which_axis='x')
    
    
    return g.fig, strong_fits


def sort_by_selectivity(fit_thr=0.9, topn=10):
    strong_fits = [r for r, v in fitdf.groupby(['cell']) if v.mean()['r2'] >= fit_thr]
    print("%i out of %i cells with strong fits (%.2f)" % (len(strong_fits), len(fitdf['cell'].unique()), fit_thr))
    
    df = fitdf[fitdf['cell'].isin(strong_fits)]
        
    df.loc[:, 'cell'] = np.array([int(c) for c in df['cell'].values])
    
    top_asi = df.groupby(['cell']).mean().sort_values(['ASI'], ascending=False)
    top_dsi = df.groupby(['cell']).mean().sort_values(['DSI'], ascending=False)
    top_r2 = df.groupby(['cell']).mean().sort_values(['r2'], ascending=False)
    
    top_asi_cells = top_asi.index.tolist()[0:topn]
    top_dsi_cells = top_dsi.index.tolist()[0:topn]
    top_r2_cells = top_r2.index.tolist()[0:topn]

    top10_asi = [roi for rank, roi in enumerate(top_asi.index.tolist()) if rank < topn]
    top10_dsi = [roi for rank, roi in enumerate(top_dsi.index.tolist()) if rank < topn]
    
    df.loc[:, 'top_asi'] = np.array([ roi if roi in top10_asi else -10 for roi in df['cell']])
    df.loc[:, 'top_dsi'] = np.array([ roi if roi in top10_dsi else -10 for roi in df['cell']])

    
    #% # Convert to str for plotting:
        
    df.loc[:, 'top_asi'] = [str(s) for s in df['top_asi'].values]
    df.loc[:, 'top_dsi'] = [str(s) for s in df['top_dsi'].values]

    
    return df, top_asi_cells, top_dsi_cells


def compare_topn_selective(df, color_by='ASI', palette='cubehelix'):
    
    hue = 'top_asi' if color_by=='ASI' else 'top_dsi'
#    if color_by == 'ASI':
#        hue = 'top_asi'
#        palette = asi_colordict
#    elif color_by == 'DSI':
#        hue = 'top_dsi'
#        palette = dsi_colordict
#    
    g = sns.PairGrid(df, hue=hue, vars=['ASI', 'DSI'], palette=palette, size=5)#,
                    #hue_kws={"alpha": alphadict.values()}) # 'cubehelix_r') #'cubehelix') #'')
    
    g.fig.patch.set_alpha(1)
    g = g.map_offdiag(pl.scatter, marker='o', s=5, alpha=0.7) #, color=[asi_colordict[r] for r in ddf[hue]]) #=alphadict.values()[::-1]) #,  alpha=0.5, s=5, )
    
    
    g = g.map_diag(pl.hist, normed=True, alpha=0.5) #histtype="step",  
    
    
    g.set(ylim=(0, 1))
    g.set(xlim=(0, 1))
    
    g = g.add_legend(bbox_to_anchor=(1.0,0.2))
    for li, lh in enumerate(g._legend.legendHandles): 
        if not all([round(l, 1)==0.5 for l in lh.get_facecolor()[0][0:3]]):
            
            lh.set_alpha(1)
            lh._sizes = [20] 
        
    pl.subplots_adjust(left=0.15, right=0.8, top=0.9, bottom=0.1, wspace=0.2, hspace=0.2)
    
    #g.set(xlim=(0,1), ylim=(0,1))
    #g.set(xticks=[0, 1])
    #g.set(yticks=[0, 1])
    #sns.despine(trim=True)
            
    cleanup_axes(g.axes[:, 1:].flat, which_axis='y')
    cleanup_axes( g.axes[:-1, :].flat, which_axis='x')
    
    
    return g.fig

#%%

rootdir = '/n/coxfs01/2p-data'
animalid = 'JC084' 
session = '20190522' #'20190319'
fov = 'FOV1_zoom2p0x' 
run = 'combined_gratings_static'
traceid = 'traces001' #'traces002'
trace_type = 'corrected'

data_identifier = '|'.join([animalid, session, fov, run, traceid])

create_new=True
n_processes=1


# In[5]:


traceid_dir =  glob.glob(os.path.join(rootdir, animalid, session, fov, run, 'traces', '%s*' % traceid))[0]
data_fpath = glob.glob(os.path.join(traceid_dir, 'data_arrays', 'datasets.npz'))[0]
dset = np.load(data_fpath)

data_identifier = '|'.join([animalid, session, fov, run, traceid])


#%% Add baseline offset back into raw traces:
    
F0 = np.nanmean(dset['corrected'][:] / dset['dff'][:] )
print("offset: %.2f" % F0)
raw_traces = pd.DataFrame(dset['corrected']) + F0


labels = pd.DataFrame(data=dset['labels_data'], columns=dset['labels_columns'])
sdf = pd.DataFrame(dset['sconfigs'][()]).T

fr = 44.65 #dset['run_info'][()]['framerate']
nframes_on = labels['nframes_on'].unique()[0]
stim_on_frame = labels['stim_on_frame'].unique()[0]
nframes_post_onset = nframes_on + int(round(1.*fr))

#%%

gdf = resp.group_roidata_stimresponse(raw_traces.values, labels, return_grouped=True) # Each group is roi's trials x metrics
nrois_total = len(gdf.groups)

# In[12]:

responsive_test = 'ROC'

if responsive_test == 'ROC':

    stats_dir = os.path.join(traceid_dir, 'summary_stats', responsive_test)
    stats_fpath = glob.glob(os.path.join(stats_dir, '*results*.pkl'))
    assert len(stats_fpath) > 0, "No stats results found for: %s" % stats_dir
    with open(stats_fpath[0], 'rb') as f:
        rstats = pkl.load(f)

    responsive_thr = 0.05
    roi_list = [r for r, res in rstats.items() if res['pval'] < responsive_thr]
    fit_str = 'remove-bas_pass-ROC-thr%.2f' % responsive_thr

print fit_str

#%%
# # Create output dir

#roi_fitdir = os.path.join(traceid_dir, 'figures', 'fits', 'tuning_by_roi_%s' % response_type)
desc_str = 'Liang2018_Andermann'
roi_fitdir = os.path.join(traceid_dir,'tuning', desc_str, fit_str) #, desc_str)
#roi_fitdir = os.path.join(traceid_dir, 'figures', 'tuning', desc_str)


if not os.path.exists(roi_fitdir):
    os.makedirs(roi_fitdir)
print("Saving roi fits to: %s" % roi_fitdir)


#%%  # Bootstrap responses
tested_oris = sdf['ori'].unique()
oris_interp = interp_values(tested_oris, n_intervals=3, wrap_value=360)

#response_type = 'zscore'
n_bootstrap_iters = 100
n_intervals_interp = 3


oris_interp = interp_values(tested_oris, n_intervals=n_intervals_interp, wrap_value=360)


#%% # Convert raw + offset traces to df/F traces


tmp_df = []
for k, g in labels.groupby(['trial']):
    tmat = raw_traces.loc[g.index]
    bas_mean = np.nanmean(tmat[0:stim_on_frame], axis=0)
    tmat_df = (tmat - bas_mean) / bas_mean
    tmp_df.append(tmat_df)
    
    

df_traces = pd.concat(tmp_df, axis=0)
print df_traces.shape


#%% # Fit all rois in list
print "Fitting %i rois:" % len(roi_list), roi_list

response_type = 'dff'
metric_type = 'dff'
make_plots = True


constant_params = ['aspect', 'luminance', 'position', 'stimtype']
params = [c for c in sdf.columns if c not in constant_params]
stimdf = sdf[params]



#%% # Plot and fit ROIs

roi_fitdir_figures = os.path.join(roi_fitdir, 'roi_fits')
if not os.path.exists(roi_fitdir_figures):
    os.makedirs(roi_fitdir_figures)
if make_plots:
    print("Saving ROI fit plots to:\n%s" % roi_fitdir_figures)


# In[35]:


tuning_fit_results_path = os.path.join(roi_fitdir, 'tuning_bootstrap_results.pkl')
if os.path.exists(tuning_fit_results_path):
    with open(tuning_fit_results_path, 'rb') as f:
        fitdf = pkl.load(f)
    do_fits = False
else:
    do_fits = True



# In[36]:


if do_fits:
    fitdf = bootstrap_tuning_curves(gdf, roi_list=roi_list)
    
    
#%%

fit_thr = 0.9
fig, strong_fits = compare_selectivity_all_fits(fitdf, fit_thr=fit_thr)
label_figure(fig, data_identifier)

nrois_fit = len(fitdf['cell'].unique())
nrois_thr = len(strong_fits)

figname = 'bootstrap%i_tuning_fit_thr_%.2f_%iof%i' % (n_bootstrap_iters, fit_thr,
                                                      nrois_thr, nrois_fit)

pl.savefig(os.path.join(roi_fitdir, '%s.png' % figname))
#pl.close()
print("Saved:\n%s" % os.path.join(roi_fitdir, '%s.png' % figname))


#%%
# ##### Compare metrics

# Sort cells by ASi and DSi    
topn=10
df, top_asi_cells, top_dsi_cells = sort_by_selectivity(fit_thr=fit_thr, topn=topn)

#% Set color palettes:
palette = sns.color_palette('cubehelix', len(top_asi_cells))
main_alpha = 0.8
sub_alpha = 0.01
asi_colordict = dict(( str(roi), palette[i]) for i, roi in enumerate(top_asi_cells))
for k, v in asi_colordict.items():
    asi_colordict[k] = (v[0], v[1], v[2], main_alpha)
    
dsi_colordict = dict(( str(roi), palette[i]) for i, roi in enumerate(top_dsi_cells))
for k, v in dsi_colordict.items():
    dsi_colordict[k] = (v[0], v[1], v[2], main_alpha)
      
asi_colordict.update({ str(-10): (0.8, 0.8, 0.8, sub_alpha)})
dsi_colordict.update({ str(-10): (0.8, 0.8, 0.8, sub_alpha)})


#%% PLOT by ASI:

nrois_fit = len(fitdf['cell'].unique())
nrois_pass = len(df['cell'].unique())

color_by = 'ASI'

if color_by == 'ASI':
    palette = asi_colordict
elif color_by == 'DSI':
    palette = dsi_colordict
        
fig = compare_topn_selective(df, color_by=color_by, palette=palette)
label_figure(fig, data_identifier)

figname = 'bootstrap%i_tuning_fit_thr_%.2f_%iof%i_colortop10_%s' % (n_bootstrap_iters, fit_thr,
                                                      nrois_pass, nrois_fit, color_by)

pl.savefig(os.path.join(roi_fitdir, '%s.png' % figname))

#%% Color by DSI:
color_by = 'DSI'

if color_by == 'ASI':
    palette = asi_colordict
elif color_by == 'DSI':
    palette = dsi_colordict
        
fig = compare_topn_selective(df, color_by=color_by, palette=palette)
label_figure(fig, data_identifier)

figname = 'bootstrap%i_tuning_fit_thr_%.2f_%iof%i_colortop10_%s' % (n_bootstrap_iters, fit_thr,
                                                      nrois_pass, nrois_fit, color_by)

pl.savefig(os.path.join(roi_fitdir, '%s.png' % figname))


# In[429]: ALL CELLS:



#g = sns.PairGrid(fitdf, hue='cell', vars=['ASI', 'DSI', 'r2'])
#
#g = g.map_diag(sns.distplot, kde=False, hist=True, rug=True,
#               hist_kws={"histtype": "step", "linewidth": 2, "alpha": 1.0})
#g = g.map_offdiag(pl.scatter, marker='+')
#
#g.set(xlim=(0,1), ylim=(0,1))
#g.set(xticks=[0, 1])
#g.set(yticks=[0, 1])
#sns.despine(trim=True)
#
#cleanup_axes(g.axes[:, 1:].flat, which_axis='y')
#cleanup_axes( g.axes[:-1, :].flat, which_axis='x')
#

