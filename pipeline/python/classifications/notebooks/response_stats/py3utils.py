import os

import glob
import json
import traceback

# import pickle as pkl
import statsmodels as sm
import dill as pkl
import numpy as np
import pylab as pl
import seaborn as sns
import scipy.stats as spstats
import pandas as pd
import importlib

import scipy as sp
import itertools
import matplotlib as mpl
from matplotlib.lines import Line2D
import statsmodels as sm
#import statsmodels.api as sm

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)       

import re

# ###############################################################
# Analysis specific
# ###############################################################
def decode_analysis_id(visual_area=None, prefix='split_pupil', response_type='dff', responsive_test='ROC',
                       overlap_thr=None, trial_epoch='plushalf', C_str='tuneC'):
    overlap_str = 'noRF' if overlap_thr in [None, 'None'] else 'overlap%.2f' % overlap_thr
    results_id = '%s_%s__%s-%s_%s__%s__%s' \
                    % (prefix, visual_area, response_type, responsive_test, overlap_str, trial_epoch, C_str)
    return results_id

def load_split_pupil_input(animalid, session, fovnum, curr_id='results_id', traceid='traces001', 
                          rootdir='/n/coxfs01/2p-data'):

    curr_inputfiles = glob.glob(os.path.join(rootdir, animalid, session, 'FOV%i_*' % fovnum,
                        'combined_blobs_static', 'traces',
                       '%s*' % traceid, 'decoding', 'inputdata_%s.pkl' % curr_id))
    try:
        assert len(curr_inputfiles)==1, "More than 1 input file: %s" % str(curr_inputfiles)
        with open(curr_inputfiles[0], 'rb') as f:
            res = pkl.load(f)  
    except UnicodeDecodeError:
        with open(curr_inputfiles[0], 'rb') as f:
            res = pkl.load(f, encoding='latin1')  
    except Exception as e:
        traceback.print_exc()
        return None
    
    return res

# ###############################################################
# Data formatting
# ###############################################################

def unstacked_neuraldf_to_stacked(ndf, response_type='response', id_vars = ['config', 'trial']):
    ndf['trial'] = ndf.index.tolist()
    melted = pd.melt(ndf, id_vars=id_vars,
                     var_name='cell', value_name=response_type)

    return melted

def stacked_neuraldf_to_unstacked(ndf): #neuraldf):

    other_cols = [k for k in ndf.columns if k not in ['cell', 'response']]
    n2 = ndf.pivot_table(columns=['cell'], index=other_cols)

    rdf = pd.DataFrame(data=n2.values, columns=n2.columns.get_level_values('cell'),
                 index=n2.index.get_level_values('trial'))
    rdf['config'] = n2.index.get_level_values('config')

    return rdf


# ###############################################################
# General
# ###############################################################
def isnumber(n):
    try:
        float(n)   # Type-casting the string to `float`.
                   # If string is not a valid `float`,
                   # it'll raise `ValueError` exception
    except ValueError:
        return False
    except TypeError:
        return False

    return True

def split_datakey(df):
    df['animalid'] = [s.split('_')[1] for s in df['datakey']]
    df['fov'] = ['FOV%i_zoom2p0x' % int(s.split('_')[2][3:]) for s in df['datakey']]
    df['session'] = [s.split('_')[0] for s in df['datakey']]
    return df

def split_datakey_str(s):
    session, animalid, fovn = s.split('_')
    fovnum = int(fovn[3:])
    return session, animalid, fovnum


def choose_best_fov(which_fovs, criterion='max', colname='cell'):
    if criterion=='max':
        max_loc = np.where(which_fovs[colname]==which_fovs[colname].max())[0]
    else:
        max_loc = np.where(which_fovs[colname]==which_fovs[colname].min())[0]

    return max_loc

# def load_sorted_fovs(aggregate_dir='/n/coxfs01/julianarhee/aggregate-visual-areas'):
#     dsets_file = os.path.join(aggregate_dir, 'data-stats', 'sorted_datasets.json')
#     with open(dsets_file, 'r') as f:
#         fov_keys = json.load(f)
    
#     return fov_keys

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

                
    
# ###############################################################
# Plotting:
# ###############################################################
def print_means(plotdf, groupby=['visual_area', 'arousal'], params=None):
    if params is None:
        params = [k for k in plotdf.columns if k not in groupby]
        
    m_ = plotdf.groupby(groupby)[params].mean().reset_index()
    s_ = plotdf.groupby(groupby)[params].std().reset_index()
    for p in params:
        m_['%s_std' % p] = s_[p].values
    print("MEANS:")
    print(m_)

def set_threecolor_palette(c1='magenta', c2='orange', c3='dodgerblue', cmap=None, soft=False,
                            visual_areas = ['V1', 'Lm', 'Li']):
    if soft:
        c1='turquoise';c2='cornflowerblue';c3='orchid';

    # colors = ['k', 'royalblue', 'darkorange'] #sns.color_palette(palette='colorblind') #, n_colors=3)
    # area_colors = {'V1': colors[0], 'Lm': colors[1], 'Li': colors[2]}
    if cmap is not None:
        c1, c2, c3 = sns.color_palette(palette=cmap, n_colors=len(visual_areas))#'colorblind') #, n_colors=3) 
    area_colors = dict((k, v) for k, v in zip(visual_areas, [c1, c2, c3]))

    return visual_areas, area_colors

def set_plot_params(lw_axes=0.25, labelsize=6, color='k', dpi=100):
    import pylab as pl
    #### Plot params
    pl.rcParams['font.size'] = labelsize
    #pl.rcParams['text.usetex'] = True
    
    pl.rcParams["axes.labelsize"] = labelsize
    pl.rcParams["axes.linewidth"] = lw_axes
    pl.rcParams["xtick.labelsize"] = labelsize
    pl.rcParams["ytick.labelsize"] = labelsize
    pl.rcParams['xtick.major.width'] = lw_axes
    pl.rcParams['xtick.minor.width'] = lw_axes
    pl.rcParams['ytick.major.width'] = lw_axes
    pl.rcParams['ytick.minor.width'] = lw_axes
    pl.rcParams['legend.fontsize'] = labelsize
    
    pl.rcParams['figure.figsize'] = (5, 4)
    pl.rcParams['figure.dpi'] = dpi
    pl.rcParams['savefig.dpi'] = dpi
    pl.rcParams['svg.fonttype'] = 'none' #: path
        
    
    for param in ['xtick.color', 'ytick.color', 'axes.labelcolor', 'axes.edgecolor']:
        pl.rcParams[param] = color

    return 
    
# Plotting
def label_figure(fig, data_identifier):
    fig.text(0, 1,data_identifier, ha='left', va='top', fontsize=8)

    
def crop_legend_labels(ax, n_hues, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12,
                        title='', ncol=1, markerscale=1):
    # Get the handles and labels.
    leg_handles, leg_labels = ax.get_legend_handles_labels()
    # When creating the legend, only use the first two elements
    leg = ax.legend(leg_handles[0:n_hues], leg_labels[0:n_hues], title=title,
            bbox_to_anchor=bbox_to_anchor, fontsize=fontsize, loc=loc, ncol=ncol, 
            markerscale=markerscale)
    return leg

def get_empirical_ci(stat, ci=0.95):
    p = ((1.0-ci)/2.0) * 100
    lower = np.percentile(stat, p) #max(0.0, np.percentile(stat, p))
    p = (ci+((1.0-ci)/2.0)) * 100
    upper = np.percentile(stat, p) # min(1.0, np.percentile(x0, p))
    #print('%.1f confidence interval %.2f and %.2f' % (alpha*100, lower, upper))
    return lower, upper

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


def plot_mannwhitney(mdf, metric='I_rs', multi_comp_test='holm',
                        ax=None, y_loc=None, offset=0.1, lw=0.25, fontsize=6):
    if ax is None:
        fig, ax = pl.subplots()

    print("********* [%s] Mann-Whitney U test(mc=%s) **********" % (metric, multi_comp_test))
    statresults = do_mannwhitney(mdf, metric=metric, multi_comp_test=multi_comp_test)
    #print(statresults)

    # stats significance
    ax = annotate_stats_areas(statresults, ax, y_loc=y_loc, offset=offset, 
                                lw=lw, fontsize=fontsize)
    print("****************************")

    return statresults, ax


# Stats
def annotate_stats_areas(statresults, ax, lw=1, color='k',
                        y_loc=None, offset=0.1, fontsize=6,
                         visual_areas=['V1', 'Lm', 'Li']):

    if y_loc is None:
        y_loc = round(ax.get_ylim()[-1], 1)*1.2
        offset = y_loc*offset #0.1

    for ci in statresults[statresults['reject']].index.tolist():
    #np.arange(0, statresults[statresults['reject']].shape[0]):
        v1, v2, pv, uv = statresults.iloc[ci][['d1', 'd2', 'p_val', 'U_val']].values
        x1 = visual_areas.index(v1)
        x2 = visual_areas.index(v2)
        y1 = y_loc+(ci*offset)
        y2 = y1
        ax.plot([x1,x1, x2, x2], [y1, y2, y2, y1], linewidth=lw, color=color)
        ctrx = x1 + (x2-x1)/2.
        star_str = '**' if pv<0.01 else '*'
        ax.text(ctrx, y1+(offset/8.), star_str, fontsize=fontsize)

    return ax


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
    
    #import statsmodels.api as sm

    
    visual_areas = ['V1', 'Lm', 'Li']
    mpairs = list(itertools.combinations(visual_areas, 2))

    pvalues = []
    stats = []
    nsamples = []
    for mp in mpairs:
        d1 = mdf[mdf['visual_area']==mp[0]][metric]
        d2 = mdf[mdf['visual_area']==mp[1]][metric]

        # compare samples
        stat, p = spstats.mannwhitneyu(d1, d2)
        n1=len(d1)
        n2=len(d2)

        # interpret
        alpha = 0.05
        if p > alpha:
            interp_str = '... Same distribution (fail to reject H0)'
        else:
            interp_str = '... Different distribution (reject H0)'
        # print('[%s] Statistics=%.3f, p=%.3f, %s' % (str(mp), stat, p, interp_str))

        pvalues.append(p)
        stats.append(stat)
        nsamples.append((n1, n2))

    reject, pvals_corrected, _, _ = sm.stats.multitest.multipletests(pvalues,
                                                                     alpha=0.05,
                                                                     method=multi_comp_test)
#    r_=[]
#    for mp, rej, pv, st, ns in zip(mpairs, reject, pvals_corrected, stats, nsamples):
#        print('[%s] p=%.3f (%s), reject H0=%s' % (str(mp), pv, multi_comp_test, rej))
#        r_.append(pd.Series({'d1': mp[0], 'd2': mp[1], 'n1': ns[0], 'n2': ns[1],
#                             'reject': rej, 'p_val': pv, 'U_val': st}))
#    results = pd.concat(r_, axis=1).T.reset_index(drop=True)
    results = pd.DataFrame({'d1': [mp[0] for mp in mpairs],
                            'd2': [mp[1] for mp in mpairs],
                            'reject': reject,
                            'p_val': pvals_corrected,
                            'U_val': stats,
                            'n1': [ns[0] for ns in nsamples],
                            'n2': [ns[1] for ns in nsamples]})
    print(results)

    return results



def paired_ttest_from_df(plotdf, metric='avg_size', c1='rfs', c2='rfs10', compare_var='experiment',
                            round_to=None, return_vals=False, ttest=True):
    
    a_vals = plotdf[plotdf[compare_var]==c1].sort_values(by='datakey')[metric].values
    b_vals = plotdf[plotdf[compare_var]==c2].sort_values(by='datakey')[metric].values
    #print(a_vals, b_vals)
    if ttest:
        tstat, pval = spstats.ttest_rel(np.array(a_vals), np.array(b_vals))
    else:
        tstat, pval = spstats.wilcoxon(np.array(a_vals), np.array(b_vals))

    #print('%s: %.2f (p=%.2f)' % (visual_area, tstat, pval))
    if round_to is not None:
        tstat = round(tstat, round_to)
        pval = round(pval, round_to)

    pdict = {'t_stat': tstat, 'p_val': pval}

    if return_vals:
        return pdict, a_vals, b_vals
    else:
        return pdict


def paired_ttests(comdf, metric='avg_size',  round_to=None,
                c1='rfs', c2='rfs10', compare_var='experiment', ttest=True,
                visual_areas=['V1', 'Lm', 'Li']):
    r_=[]
    for ai, visual_area in enumerate(visual_areas):

        plotdf = comdf[comdf['visual_area']==visual_area].copy()

        pdict = paired_ttest_from_df(plotdf, c1=c1, c2=c2, metric=metric,
                        compare_var=compare_var, round_to=round_to, return_vals=False, ttest=ttest)
        pdict.update({'visual_area': visual_area})
        res = pd.DataFrame(pdict, index=[ai])
        r_.append(res)

    statdf = pd.concat(r_, axis=0)

    return statdf




### Data loading
def convert_columns_byte_to_str(df):
#     str_df = df.select_dtypes([np.object])
#     str_df = str_df.stack().str.decode('utf-8').unstack()
#     for col in str_df:
#         df[col] = str_df[col]
    new_columns = dict((c, c.decode("utf-8") ) for c in df.columns.tolist())
    df = df.rename(columns=new_columns)
    return df

def load_corrected_dff_traces(animalid, session, fov, experiment='blobs', traceid='traces001',
                              return_traces=True, epoch='stimulus', metric='mean', return_labels=False,
                              rootdir='/n/coxfs01/2p-data'):
    print('... calculating F0 for df/f')
    # Load corrected
    soma_fpath = glob.glob(os.path.join(rootdir, animalid, session, fov,
                                    '*%s_static' % (experiment), 'traces', '%s*' % traceid,
                                    'data_arrays', 'np_subtracted.npz'))[0]
    dset = np.load(soma_fpath, allow_pickle=True)
    Fc = pd.DataFrame(dset['data']) # np_subtracted:  Np-corrected trace, with baseline subtracted

    # Load raw (pre-neuropil subtraction)
    raw = np.load(soma_fpath.replace('np_subtracted', 'raw'))
    F0_raw = pd.DataFrame(raw['f0'])

    # Calculate df/f
    dff = Fc.divide(F0_raw) # dff 

    if return_traces:
        if return_labels:
            labels = pd.DataFrame(data=dset['labels_data'],columns=dset['labels_columns'])
            labels = convert_columns_byte_to_str(labels)

            return dff, labels
        else:
            return dff
    else:
        labels = pd.DataFrame(data=dset['labels_data'],columns=dset['labels_columns'])
        labels = convert_columns_byte_to_str(labels)
        dfmat = traces_to_trials(dff, labels, epoch=epoch, metric=metric)
        return dfmat



    
