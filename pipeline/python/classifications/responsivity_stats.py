
# coding: utf-8

# In[1]:


import os
import glob
import json
import h5py
import copy
import optparse
import sys

import numpy as np
import pandas as pd
import pylab as pl
import seaborn as sns
import cPickle as pkl
import matplotlib.gridspec as gridspec

from pipeline.python.classifications import experiment_classes as util
from pipeline.python.classifications import test_responsivity as resp
from pipeline.python.utils import label_figure, natural_keys, convert_range

from pipeline.python.retinotopy import fit_2d_rfs as fitrf
from matplotlib.patches import Ellipse, Rectangle

from shapely.geometry.point import Point
from shapely import affinity
from matplotlib.patches import Polygon

from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import MaxNLocator

import matplotlib_venn as mpvenn
import itertools
#get_ipython().magic(u'matplotlib notebook')


# In[2]:


def intersection(lst1, lst2): 
    return list(set(lst1) & set(lst2)) 


def create_ellipse(center, lengths, angle=0):
    """
    create a shapely ellipse. adapted from
    https://gis.stackexchange.com/a/243462
    """
    circ = Point(center).buffer(1)
    ell = affinity.scale(circ, int(lengths[0]), int(lengths[1]))
    ellr = affinity.rotate(ell, angle)
    return ellr

from shapely.geometry import box

#
#def check_path_intersection(ref_patch, sample_patch, all_contained=False):
#    ref_path = ref_patch.get_patch_transform().transform_path(ref_patch.get_path())
#
#    verts = sample_patch.get_path().vertices
#    trans = sample_patch.get_patch_transform()
#    pts = trans.transform(verts)
#    
#    if all_contained:
#        return all(g_path.contains_points(pts))
#    else:
#        return any(g_path.contains_points(pts))
#

# In[40]:



def compare_rf_resolution(gdfs, animalid, session, fov, traceid='traces001', response_type='dff'):
    '''
    Assumes trace_type = 'dff' and calculates using 'stim_mean' stat metric.
    '''
    
    roi_lists = dict((exp, d.rois) for exp, d in gdfs.items())
    
    print("Found expmts for both RFs and RFs10")
    rf_set_labels = ['rfs', 'rfs10']
    rf_sets = [set(gdfs[k].rois) for k in rf_set_labels]
    
    ## Get distribution of RF sizes
    rfits = util.get_receptive_field_fits(animalid, session, fov, traceid=traceid,
                                          response_type=response_type, run='combined_rfs_static')
    rfits10 = util.get_receptive_field_fits(animalid, session, fov, traceid=traceid,
                                            response_type=response_type, run='combined_rfs10_static')

    rfits_df = gdfs['rfs'].fits
    xres = list(set(np.diff(rfits['col_vals'])))[0]
    yres = list(set(np.diff(rfits['row_vals'])))[0]
    print("rfs: X- and Y-res: (%i, %i)" % (xres, yres))

    rfits_df10 =  gdfs['rfs10'].fits
    xres10 = list(set(np.diff(rfits10['col_vals'])))[0]
    yres10 = list(set(np.diff(rfits10['row_vals'])))[0]
    print("rfs10: X- and Y-res: (%i, %i)" % (xres10, yres10))

    rf_colors = {'rfs': 'red', 'rfs10': 'cornflowerblue'}
    sigma_scale = 2.36

    #fig, axes = pl.subplots(2,5, figsize=(10,4))
    fig = pl.figure(figsize=(10, 6))
    fig.patch.set_alpha(1)

    ## Venn diagram of overlap ROIs:
    ax = pl.subplot2grid((2, 5), (0, 0), colspan=1, rowspan=1)
    v = mpvenn.venn2(rf_sets, set_labels=rf_set_labels, ax=ax)
    for p in v.patches:
        p.set_alpha(0)
    c=mpvenn.venn2_circles(rf_sets, ax=ax) #set_labels=roi_set_labels, ax=ax)
    for ci in range(len(c)):
        c[ci].set_edgecolor(rf_colors[rf_set_labels[ci]])
        c[ci].set_alpha(0.5)

    ## Distribution of peak dF/Fs:
    ax = pl.subplot2grid((2, 5), (0, 1), colspan=2, rowspan=1)
    peak_dfs = [gdfs['rfs'].gdf.get_group(roi).groupby(['config']).mean()[response_type].max() for roi in roi_lists['rfs']]
    peak_dfs10 = [gdfs['rfs10'].gdf.get_group(roi).groupby(['config']).mean()[response_type].max() for roi in roi_lists['rfs10']]
    weights_rfs = np.ones_like(peak_dfs) / float(len(peak_dfs))
    sns.distplot(peak_dfs, ax=ax, color=rf_colors['rfs'], label='rfs (n=%i)' % len(peak_dfs),
                 kde=False, hist=True,
                 hist_kws={'histtype': 'step', 'alpha': 0.5, 'weights': weights_rfs, 'normed':0, 'lw': 2})
    weights_rfs10 = np.ones_like(peak_dfs10) / float(len(peak_dfs10))
    sns.distplot(peak_dfs10, ax=ax, color=rf_colors['rfs10'], label='rfs10 (n=%i)' % len(peak_dfs10),
                 kde=False, hist=True,
                 hist_kws={'histtype': 'step', 'alpha': 0.5, 'weights': weights_rfs10, 'normed':0, 'lw': 2})
    ax.legend(loc='upper right', fontsize=6)
    ax.set_xlim([0, ax.get_xlim()[-1]])
    if response_type == 'stim_mean':
        response_unit = 'intensity'
    elif response_type == 'zscore':
        response_unit = 'std'
    else:
        response_unit = response_type
        
    ax.set_xlabel('peak %s' % response_unit)
    ax.set_ylabel('fraction')
    #ax.set_ylim([0, max([max(peak_dfs), max(peak_dfs10)])])
    sns.despine(trim=True, offset=2)

    ## Distribution of avg RF size:
    ax = pl.subplot2grid((2, 5), (0, 3), colspan=2, rowspan=1)
    size_rfs = rfits_df[['sigma_x', 'sigma_y']].mean(axis=1)
    size_rfs10 = rfits_df10[['sigma_x', 'sigma_y']].mean(axis=1)
    weights_size_rfs = np.ones_like(size_rfs) / float(len(size_rfs))
    weights_size_rfs10 = np.ones_like(size_rfs10) / float(len(size_rfs10))
    sns.distplot(size_rfs, ax=ax, color=rf_colors['rfs'], label='rfs (avg. %.2f)' % np.mean(size_rfs),
                 kde=False, hist=True,
                 hist_kws={'histtype': 'step', 'alpha': 0.5, 'weights': weights_size_rfs, 'normed':0, 'lw': 2})
    sns.distplot(size_rfs10, ax=ax, color=rf_colors['rfs10'], label='rfs10 (avg. %.2f)' % np.mean(size_rfs10),
                 kde=False, hist=True,
                 hist_kws={'histtype': 'step', 'alpha': 0.5, 'weights': weights_size_rfs10, 'normed':0, 'lw': 2})
    ax.legend(loc='upper right', fontsize=6)
    ax.set_xlim([0, ax.get_xlim()[-1]+5])
    ax.set_xlabel('average RF size')
    ax.set_ylabel('fraction')
    sns.despine(trim=True, offset=2)



    ## Distribution of avg RF size:
    rois_in_both_rfs = intersection(rfits_df.index.tolist(), rfits_df10.index.tolist())
    rf_params=['sigma_x', 'sigma_y', 'x0', 'y0', 'theta']
    for ai in range(len(rf_params)):
        ax = pl.subplot2grid((2, 5), (1, ai), colspan=1, rowspan=1)
        ax.set_title(rf_params[ai])

        if rf_params[ai] == 'theta':
            rf_vals = [np.rad2deg(th) % 360. for th in rfits_df[rf_params[ai]][rois_in_both_rfs]]
            rf_vals10 = [np.rad2deg(th) % 360. for th in rfits_df10[rf_params[ai]][rois_in_both_rfs]]
        else:
            rf_vals = rfits_df[rf_params[ai]][rois_in_both_rfs]
            rf_vals10 = rfits_df10[rf_params[ai]][rois_in_both_rfs]
        ax.scatter(rf_vals, rf_vals10,
                   s=5, marker='o', c='k', alpha=0.5)
        ax.set_xlabel('rfs'); ax.set_ylabel('rfs10');
        minv = min([min(rf_vals), min(rf_vals10)])
        maxv = max([max(rf_vals), max(rf_vals10)])
        ax.set_xlim([minv, maxv])
        ax.set_ylim([minv, maxv])
        #sns.despine(trim=True, ax=ax)
        ax.set_aspect('equal')
    pl.subplots_adjust(wspace=1, hspace=.1, left=0.1)
    
    return fig



def compare_experiments_responsivity(gdfs, response_type='dff', exp_names=[], exp_colors={}):

    print("Comparing experiments:", exp_names)
    tmp_roi_list = [gdfs[k].rois for k in exp_names]
    event_rois = list(set(itertools.chain(*tmp_roi_list)))

    roi_sets = [set(gdfs[k].rois) for k in exp_names]
    roi_set_labels = copy.copy(exp_names)
    #print(roi_set_labels)

    fig, axes = pl.subplots(1,3, figsize=(10,4))
    fig.patch.set_alpha(1)

    ax = axes[0]
    
    if len(exp_names) > 2:
        v = mpvenn.venn3(roi_sets, set_labels=roi_set_labels, ax=ax)
        for pid in v.id2idx.keys():
            v.get_patch_by_id(pid).set_alpha(0)
    
        c=mpvenn.venn3_circles(roi_sets, ax=ax) #set_labels=roi_set_labels, ax=ax)
        for ci in range(len(c)):
            c[ci].set_edgecolor(exp_colors[roi_set_labels[ci]])
            c[ci].set_alpha(0.5)
    elif len(exp_names)==1:
        pass
    else:
        v = mpvenn.venn2(roi_sets, set_labels=roi_set_labels, ax=ax)
#        for pid in v.id2idx.keys():
#            v.get_patch_by_id(pid).set_alpha(0)
    
        for s in range(len(v.patches)):
            v.patches[s].set_alpha(0) #(s).set_alpha(0)
            
        c=mpvenn.venn2_circles(subsets=roi_sets, lw=2, ax=ax) #set_labels=roi_set_labels, ax=ax)
        for ci in range(len(c)):
            c[ci].set_edgecolor(exp_colors[roi_set_labels[ci]])
            c[ci].set_alpha(0.5)

    # Fraction of cells:
    ax = axes[1]
    for exp_name in exp_names:
        peak_values = gdfs[exp_name].gdf.max()[response_type].values
        weights = np.ones_like(peak_values) / float(len(event_rois))
        exp_str = '%s (%i)' % (exp_name, len(peak_values))
        sns.distplot(peak_values, label=exp_str, ax=ax, norm_hist=0, kde=False,
                     rug=False, rug_kws={"alpha": 0.5},
                     hist_kws={"histtype": "step", "linewidth": 2, "alpha": 0.5,
                             'weights': weights, 'normed': 0, "color": exp_colors[exp_name]})

    ax.set_xlabel('peak response\n(dF/F)', fontsize=8)
    ax.set_ylabel('fraction of\nresponsive cells', fontsize=8)
    ax.legend()
    ax.set_xlim([min([0, ax.get_xlim()[0]]), max([3, ax.get_xlim()[1]+2])])
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    sns.despine(trim=True, offset=4, ax=ax)
    ax.set_title('Total: %i unique cells' % len(event_rois), fontsize=8)

    # ALL responses -- counts of cells
    ax = axes[2]
    all_values = []
    for exp_name in exp_names:
        all_values.extend(gdfs[exp_name].gdf.max()[response_type].values)
    #weights = np.ones_like(all_values) / float(len(all_values))
    sns.distplot(all_values, ax=ax, norm_hist=0, kde=False,
                rug=False,
                hist_kws={"histtype": "step", "linewidth": 2, "alpha": 0.9, # 'weights': weights, 
                            'normed': 0, "color": 'k'}) #[ename]})
    ax.set_xlim([min([0, ax.get_xlim()[0]]), max([3, ax.get_xlim()[1]+2])])
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(3))
    sns.despine(trim=True, offset=4, ax=ax)
    ax.set_xlabel('peak response\n(dF/F)', fontsize=8)
    ax.set_ylabel('counts')

    pl.subplots_adjust(wspace=0.5, top=0.8, left=0.01, bottom=0.2)

    return fig


# In[14]:


#rootdir = '/n/coxfs01/2p-data'
#
#animalid = 'JC085'
#session = '20190622'
#fov = 'FOV1_zoom2p0x'
#traceid = 'traces001'

# In[5]:
    
def extract_options(options):
    parser = optparse.OptionParser()

    # PATH opts:
    parser.add_option('-D', '--root', action='store', dest='rootdir', default='/n/coxfs01/2p-data', help='data root dir (root project dir containing all animalids) [default: /nas/volume1/2photon/data, /n/coxfs01/2pdata if --slurm]')
    parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')

    # Set specific session/run for current animal:
    parser.add_option('-S', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID')
    parser.add_option('-A', '--acq', action='store', dest='fov', default='FOV1_zoom2p0x', help="acquisition folder (default: FOV1)")
    parser.add_option('-t', '--traceid', action='store', dest='traceid', default='traces001', help="traceid (default: traces001)")
    parser.add_option('-T', '--trace_type', action='store', dest='trace_type', default='dff', help="trace type (default: dff, for calculating mean stats)")

    parser.add_option('--new', action='store_true', dest='create_new', default=False, help="Create all session objects from scratch")

    (options, args) = parser.parse_args(options)

    return options

#%%
def get_session_stats(S, response_type='dff', responsive_test='ROC', trace_type='corrected',
                      experiment_list=None, traceid='traces001', pretty_plots=False,
                      rootdir='/n/coxfs01/2p-data', create_new=True):

    # Create output dirs:    
    output_dir = os.path.join(rootdir, S.animalid, S.session, S.fov, 'summaries')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(output_dir)

    stats_desc = '-'.join([traceid, trace_type, response_type, responsive_test])
    
    statsdir = os.path.join(output_dir, 'stats')
    if not os.path.exists(statsdir):
        os.makedirs(statsdir)

    statsfigdir = os.path.join(statsdir, 'figures')
    if not os.path.exists(statsfigdir):
        os.makedirs(statsfigdir)
        
    # Create or load stats:
    stats_fpath = os.path.join(statsdir, 'sessionstats_%s.pkl' % stats_desc)
    if os.path.exists(stats_fpath) and create_new is False:
        print("found stats")
        try:
            print("loading existing stats")
            with open(stats_fpath, 'rb') as f:
                gdfs = pkl.load(f)
            assert len(gdfs.keys()) > 0, "No experiment stats found! creating new..."
        except Exception as e:
            print e
            create_new = True
    else:
        create_new = True


    if experiment_list is None:
        experiment_list = S.get_experiment_list(traceid=traceid, trace_type=trace_type)
        print("found %i experiments" % len(experiment_list))

    datasets_nostats=[]
    if os.path.exists(stats_fpath) and create_new is False:
        with open(stats_fpath, 'rb') as f:
            gdfs = pkl.load(f)

    if create_new:
        print("Calculating stats")
        # # Calculate stats using dff
        mag_ratio_thr = 0.01
        gdfs = {}
        for exp_name in experiment_list:
            if 'dyn' in exp_name:
                continue
            
            rename=False; new_name=None;
            if int(S.session) < 2019511 and exp_name == 'rfs':
                # is actually called 'gratings'
                exp_name = 'gratings'
                rename = True
                new_name = 'rfs'
                
            print("[%s] Loading roi lists..." % exp_name)
            estats, nostats = S.get_grouped_stats(exp_name, response_type=response_type,
                                         responsive_test=responsive_test, 
                                         pretty_plots=pretty_plots,
                                         traceid=traceid, trace_type=trace_type,
                                         responsive_thr=mag_ratio_thr, update=True)
            if estats is not None:
                if rename:
                    print("[%s] - renaming gratings back to rfs")
                    estats[new_name] = estats.pop(exp_name)
                gdfs.update(estats)
                
            datasets_nostats.extend(nostats)
            
#            print([S.animalid, S.session, S.fov, S.traceid, S.rois])
#            data_identifier = '|'.join([S.animalid, S.session, S.fov, S.traceid, S.rois])
#            data_identifier
#            print(data_identifier)
        
        with open(stats_fpath, 'wb') as f:
            pkl.dump(gdfs, f, protocol=pkl.HIGHEST_PROTOCOL)
        print("Saved stats to file: %s" % stats_fpath)
    
    return gdfs, statsdir, stats_desc, datasets_nostats
        

# In[15]:

    
def visualize_session_stats(animalid, session, fov, response_type='dff', responsive_test='ROC',
                            traceid='traces001', trace_type='corrected', experiment_list=None,
                            rootdir='/n/coxfs01/2p-data', create_new=False,
                            altdir=None):
    
    
    # Create output_dir
    
    output_dir = os.path.join(rootdir, animalid, session, fov, 'summaries')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(output_dir)

    
    # Get meta info:
    meta_fpath = os.path.join(rootdir, animalid, 'sessionmeta.json')
    with open(meta_fpath, 'r') as f:
        meta = json.load(f)
        
    skey = [k for k, v in meta.items() if session in k and k.split('_')[-1] in fov][0]
    visual_area = meta[skey]['visual_area']
    state = meta[skey]['state']
    
    if altdir is not None:
        alternate_savedir = os.path.join(altdir, '%s' % visual_area)
        if not os.path.exists(alternate_savedir):
            os.makedirs(alternate_savedir)
    else:
        alternate_savedir = None

    print("creating new session object...")
    S = util.Session(animalid, session, fov, rootdir=rootdir, visual_area=visual_area, state=state)

    gdfs, stats_dir, stats_desc, nostats = get_session_stats(S, traceid=traceid,
                                                    response_type=response_type, trace_type=trace_type,
                                                    responsive_test=responsive_test, 
                                                    rootdir=rootdir, create_new=create_new,
                                                    experiment_list=experiment_list)
    
    statsfigdir = os.path.join(stats_dir, 'figures')
    if not os.path.exists(statsfigdir):
        os.makedirs(statsfigdir)
    
    none2compare = False
    print("=============ROI SUMMARY=============")
    if len(gdfs.keys()) == 0:
        print('NO STATS FOUND FOR:')
        for ni, nostats in enumerate(nostats):
            print ni, nostats
        none2compare = True
    else:
        for exp_name, exp_gdf in gdfs.items():
            print('%s: %i rois' % (exp_name, len(exp_gdf.rois)))
    print("=====================================")

    if none2compare:
        return nostats
    
    
    data_identifier = '|'.join([S.animalid, S.session, S.fov, traceid]) # S.rois])
    
    compare_rf_exps = False
    if 'rfs' in gdfs.keys() and 'rfs10' in gdfs.keys():
        fig = compare_rf_resolution(gdfs, animalid, session, fov, traceid=traceid)
        
        label_figure(fig, '%s_%s' % (data_identifier, stats_desc))
        pl.savefig(os.path.join(statsfigdir, 'compare_rfs_vs_rfs10_%s.png' % stats_desc))
        if alternate_savedir is not None:
            pl.savefig(os.path.join(alternate_savedir, "%s_%s_%s_compareRFs_%s.png" % (state, visual_area, data_identifier.replace('|', '-'), stats_desc)))
    
        # # Visualize responses to event-based experiments:

    rf_exp_name = 'rfs10' if 'rfs10' in gdfs.keys() else 'rfs'
    exp_names = [r for r in gdfs.keys() if 'rfs' not in r and 'retino' not in r]
    if any(['rfs' in r for r in gdfs.keys()]):
        exp_names.append(rf_exp_name)
    exp_names = sorted(exp_names)
    print("All experiments:", gdfs.keys())
    
    tmp_roi_list = [v.rois for k, v in gdfs.items()]
    all_rois = list(set(itertools.chain(*tmp_roi_list)))
    

    exp_colors= {rf_exp_name: 'black',
                'gratings': 'orange',
                'blobs': 'blue'}
    #              'retino': 'gray'}
    

    fig = compare_experiments_responsivity(gdfs, exp_names=exp_names, exp_colors=exp_colors)
    
    label_figure(fig, data_identifier)
    pl.savefig(os.path.join(statsfigdir, "cell_counts_peak_w%s_%s.png" % (rf_exp_name, stats_desc)))

    if alternate_savedir is not None:
        pl.savefig(os.path.join(alternate_savedir, "%s_%s_%s_roistats_%s.png" % (state, visual_area, data_identifier.replace('|', '-'), stats_desc)))


    print("--- done! ---")
    
    return nostats

#%%


def main(options):
    opts = extract_options(options)
    visualize_session_stats(opts.animalid, opts.session, opts.fov,
                            traceid=opts.traceid, trace_type=opts.trace_type,
                            rootdir=opts.rootdir, create_new=opts.create_new)
    
# In[51]:

if __name__ == '__main__':
    main(sys.argv[1:])
    

# In[ ]:




