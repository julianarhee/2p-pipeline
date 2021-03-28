#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 19:56:49 2021

@author: julianarhee
"""
import sys
import os
import optparse
import psignifit as ps
import glob
# import pickle as pkl
import dill as pkl
import numpy as np
import pylab as pl
import seaborn as sns
import scipy.stats as spstats
import pandas as pd
import importlib
import scipy as sp
import itertools


def split_datakey_str(s):
    session, animalid, fovn = s.split('_')
    fovnum = int(fovn[3:])
    return session, animalid, fovnum



def get_tracedir_from_datakey(datakey, experiment='blobs', rootdir='/n/coxfs01/2p-data', traceid='traces001'):
    session, animalid, fovn = split_datakey_str(datakey)
    
    darray_dirs = glob.glob(os.path.join(rootdir, animalid, session, 'FOV%i_zoom2p0x' % fovn,
                            'combined_%s_static' % experiment, 'traces', 
                             '%s*' % traceid, 'data_arrays', 'np_subtracted.npz'))
    
    assert len(darray_dirs)==1, "More than 1 data array dir found: %s" % str(darray_dirs)
    
    traceid_dir = darray_dirs[0].split('/data_arrays')[0]
    return traceid_dir

                
def load_source_dataframes(traceid='traces001', 
                responsive_test='ROC', responsive_thr=0.05, 
                aggregate_dir = '/n/coxfs01/julianarhee/aggregate-visual-areas'):
    D = None
    stats_dir = os.path.join(aggregate_dir, 'data-stats')

    src_data_dir = os.path.join(aggregate_dir, 'data_stats', 'tmp_data')

    fbasename = 'neuraldata_%s_corrected_%s-thr-%.2f_*.pkl' \
                % (traceid, responsive_test, responsive_thr)
    dfns = glob.glob(os.path.join(src_data_dir, '%s' % fbasename))
    print(dfns)

    src_datafile = os.path.join(src_data_dir, dfns[0])
    with open(src_datafile, 'rb') as f:
        D = pkl.load(f, encoding='latin1')

    return D

def get_data(traceid='traces001', responsive_test='ROC', responsive_thr=0.05):
    D = load_source_dataframes(traceid=traceid, 
            responsive_test=responsive_test, responsive_thr=responsive_thr)

    DATA = D['DATA']
    sdata = D['sdata']
    SDF = D['SDF']
    selective_df = None

    return DATA, SDF, selective_df



def get_morph_levels(midp=53, levels=[-1, 0, 14, 27, 40, 53, 66, 79, 92, 106]):

    a_morphs = np.array(sorted([m for m in levels if m<midp and m!=-1]))[::-1]
    b_morphs = np.array(sorted([m for m in levels if m>midp and m!=-1]))

    d1 = dict((k, list(a_morphs).index(k)+1) for k in a_morphs)
    d2 = dict((k, list(b_morphs).index(k)+1) for k in b_morphs)
    morph_lut = d1.copy()
    morph_lut.update(d2)
    morph_lut.update({midp: 0, -1: -1})

    return morph_lut, a_morphs, b_morphs

def split_sample_half(g):
    nt = int(np.floor(len(g['trial'].unique())/2.))
    d1 = g[['trial', 'response']].sample(n=nt, replace=False)
    d2 = g[~g['trial'].isin(d1['trial'])].sample(n=nt)[['trial', 'response']]

    #pB = len(np.where(d1['response'].values>d2['response'].values)[0])/float(len(d1))
    return d1, d2



def get_hits_and_fas(resp_stim, resp_bas, n_crit=50):
    
    # Get N conditions
    n_conditions = len(resp_stim) #curr_cfg_ixs)
 
    # Set criterion (range between min/max response)
    min_val = min(list(itertools.chain(*resp_stim))) #resp_stim.min()
    max_val = max(list(itertools.chain(*resp_stim))) #[r.max() for r in resp_stim]) #resp_stim.max() 
    crit_vals = np.linspace(min_val, max_val, n_crit)
   
    # For each crit level, calculate p > crit (out of N trials)   
    p_hits = np.array([[sum(rs > crit)/float(len(rs)) for crit in crit_vals] for rs in resp_stim])
    p_fas = np.array([[sum(rs > crit)/float(len(rs)) for crit in crit_vals] for rs in resp_bas])

    return p_hits, p_fas, crit_vals


def split_AB_morphstep(rdf, param='morphstep', Eff=None, include_ref=True, class_a=0, class_b=106):
    '''
    rdf_sz = responses at each morph level (including diff sizes)
    Eff = effective class overall (0 or 106)
    resp_B, corresonds to response distn of Effective stimulus
    resp_A, corresponds to response distn if Ineffective stimulus
    '''
    # Split responses into A and B distns at each morph step
    Eff_obj = 'A' if Eff==class_a else 'B'
    Ineff_obj = 'B' if Eff_obj=='A' else 'B'
    resp_A=[]; resp_B=[]; resp_cfgs=[]; resp_counts=[];
    
    if include_ref:
       # Split responded to morphstep=0 in half:
        split_halves = [split_sample_half(g) for c, g in rdf[rdf.object=='M'].groupby(['size', param])]
        resp_A_REF = [t[0]['response'].values for t in split_halves]
        resp_B_REF = [t[1]['response'].values for t in split_halves]
        resp_cfgs_REF = [c for c, g in rdf[rdf.object=='M'].groupby(['size', param])]
        resp_counts_REF = [g.shape[0] for c, g in rdf[rdf.object=='M'].groupby(['size', param])]
        
        # Add to resp
        resp_A.extend(resp_A_REF)
        resp_B.extend(resp_B_REF)
        resp_counts.extend(resp_counts_REF)
        resp_cfgs.extend(resp_cfgs_REF)

    # Get all the other responses
    resp_A_ = [g['response'].values for c, g in rdf[rdf.object==Ineff_obj].groupby(['size', param])]
    resp_B_ = [g['response'].values for c, g in rdf[rdf.object==Eff_obj].groupby(['size', param])]
    
    # Corresponding configs
    resp_cfgs1_ = [c for c, g in rdf[rdf.object==Ineff_obj].groupby(['size', param])]
    resp_cfgs2_ = [c for c, g in rdf[rdf.object==Eff_obj].groupby(['size', param])]
    assert resp_cfgs1_==resp_cfgs2_, \
        "ERR: morph levels and sizes don't match for object A and B"
    # Corresponding counts
    resp_counts1_ = [g.shape[0] for c, g in rdf[rdf.object==Ineff_obj].groupby(['size', param])]
    resp_counts2_ = [g.shape[0] for c, g in rdf[rdf.object==Eff_obj].groupby(['size', param])]
    assert resp_counts1_==resp_counts2_, \
        "ERR: Trial counts don't match for object A and B"
    
    resp_cfgs.extend(resp_cfgs1_)
    resp_counts.extend(resp_counts1_)
    
    return resp_A, resp_B, resp_cfgs, resp_counts


def split_AB_morphlevel(rdf, param='morphlevel', Eff=None, include_ref=True, class_a=0, class_b=106):
    '''
    rdf_sz = responses at each morph level (~ including diff sizes)
    Eff = effective class overall (0 or 106)
    resp_B, corresonds to response distn of Effective stimulus
    resp_A, corresponds to response distn if Ineffective stimulus
    '''
    Ineff = class_b if Eff==class_a else class_a
    resp_A=[]; resp_B=[]; resp_cfgs=[]; resp_counts=[];

    # Responses to Everythign that's not "Ineffective" stimuli
    resp_B = [g['response'].values for c, g in rdf[rdf['morphlevel']!=Ineff].groupby(['size', param])]
    resp_cfgs = [c for c, g in rdf[rdf['morphlevel']!=Ineff].groupby(['size', param])]
    resp_counts = [g.shape[0] for c, g in rdf[rdf['morphlevel']!=Ineff].groupby(['size', param])]

    # Responses to "Ineffective" (baseline distN)
    if include_ref:
        # Split responses to Eff in half
        split_halves = [split_sample_half(g) for c, g in rdf[rdf['morphlevel']==Ineff].groupby(['size', param])]
        resp_A = [t[0]['response'].values for t in split_halves]
        resp_A_REF = [t[1]['response'].values for t in split_halves]
        resp_cfgs_REF = [c for c, g in rdf[rdf['morphlevel']==Ineff].groupby(['size', param])]
        resp_counts_REF = [g.shape[0] for c, g in rdf[rdf['morphlevel']==Ineff].groupby(['size', param])]

        # Add to list of comparison DistNs
        resp_B.extend(resp_A_REF)
        resp_cfgs.extend(resp_cfgs_REF)
        resp_counts.extend(resp_counts_REF)
    else:
        resp_A = [g['response'].values for c, g in rdf[rdf['morphlevel']==Ineff].groupby(['size', param])]
        

    return resp_A, resp_B, resp_cfgs, resp_counts

            
def split_signal_distns(rdf, param='morphlevel', n_crit=50, include_ref=True, Eff=None,
                       class_a=0, class_b=106):
    '''
    param=='morphstep':
        Compare objectA vs B distributions at each morph step 
        Levels: 0=53/53, 1=40/66, 2=27/79, 3=4=14/92, 4=0/106
        Split trials into halves to calculate "chance" distN (evens/odds)
    param=='morph_ix' or 'morphlevel'
        Compare Ineffective distN against Effective distNs.
        Eff = prefered A or B, Ineff, the other object.
        Split trials into halfs for Ineff distN (evens/odds)
        
    '''
    Ineff=class_b if Eff==class_a else class_a
    if param=='morphstep':
        resp_A, resp_B, resp_cfgs, resp_counts = split_AB_morphstep(rdf, param=param, Eff=Eff, include_ref=include_ref)
        
    else:
        resp_A, resp_B, resp_cfgs, resp_counts = split_AB_morphlevel(rdf, param=param, 
                                                                    Eff=Eff, include_ref=include_ref)

    p_hits, p_fas, crit_vals = get_hits_and_fas(resp_B, resp_A, n_crit=n_crit)
    
    return p_hits, p_fas, resp_cfgs, resp_counts
    
    
def calculate_auc(p_hits, p_fas, resp_cfgs1, reverse_eff=False, Eff=None):
    if p_fas.shape[0] < p_hits.shape[0]:
        altconds = list(np.unique([c[0] for c in resp_cfgs1]))
        true_auc = [-np.trapz(p_hits[ci, :], x=p_fas[altconds.index(sz), :]) \
                            for ci, (sz, mp) in enumerate(resp_cfgs1)]
    else:
        true_auc = [-np.trapz(p_hits[ci, :], x=p_fas[ci, :]) for ci in np.arange(0, len(resp_cfgs1))]
        
    aucs = pd.DataFrame({'AUC': true_auc, 
                          param: [r[1] for r in resp_cfgs1], 
                         'size': [r[0] for r in resp_cfgs1]}) 
    if reverse_eff and Eff==0:
        # flip
        for sz, a in aucs.groupby(['size']):
            aucs.loc[a.index, 'AUC'] = a['AUC'].values[::-1]
        
    return aucs


def get_auc_AB(rdf, param='morphlevel', n_crit=50, include_ref=True, reverse_eff=False,
                  class_a=0, class_b=106, return_probs=False):
    '''
    Calculate AUCs for A vs. B at each size. 
    Note:  rdf must contain columns 'morphstep' and 'size' (morphstep LUT from: get_morph_levels())

    include_ref: include morphlevel=0 (or morph_ixx) by splitting into half.
    Compare p_hit (morph=0) to p_fa (morph=106), calculate AUC.
    '''
    # Get Eff/Ineff
    means = rdf[rdf.morphlevel.isin([class_a, class_b])].groupby(['object']).mean()
    Eff = class_a if means['response']['A'] > means['response']['B'] else class_b

    p_hits, p_fas, resp_cfgs, counts = split_signal_distns(rdf, param=param, n_crit=n_crit, 
                                                        include_ref=include_ref, Eff=Eff)
        
    aucs =  calculate_auc(p_hits, p_fas, resp_cfgs, reverse_eff=reverse_eff, Eff=Eff)
    aucs['n_trials'] = counts
    aucs['Eff'] = Eff
    
    if return_probs:
        return aucs, p_hits, p_fas, resp_cfgs
    else:
        return aucs.sort_values(by=['size', 'morphlevel']).reset_index()

def plot_auc_for_cell(rdf, param='morphlevel', class_a=0, class_b=106, n_crit=50, include_ref=True, cmap='RdBu'):
    
    means = rdf[rdf.morphlevel.isin([class_a, class_b])].groupby(['object']).mean()
    print(means)
    # Get Eff/Ineff
    Eff = class_a if means['response']['A'] > means['response']['B'] else class_b

    # Calculate p_hits/p_fas for plot
    p_hits, p_fas, resp_cfgs1 = split_signal_distns(rdf, param=param, n_crit=n_crit, 
                                                    include_ref=include_ref, Eff=Eff)
    print(p_hits.shape, p_fas.shape, len(resp_cfgs1))
    aucs = calculate_auc(p_hits, p_fas, resp_cfgs1, reverse_eff=False, Eff=Eff)

    # Plot----
    mdiffs = sorted(aucs[param].unique())
    mdiff_colors = sns.color_palette(cmap, n_colors=len(mdiffs))
    colors = dict((k, v) for k, v in zip(mdiffs, mdiff_colors))

    fig, axn = pl.subplots(1, len(sizes), figsize=(8,3))    
    for ci, (sz, mp) in enumerate(resp_cfgs1):
        si = sizes.index(sz)
        ax=axn[si]
        if param=='morphstep':
            ax.plot(p_fas[ci, :], p_hits[ci, :], color=colors[mp], label=mp)
        else:
            ax.plot(p_fas[si, :], p_hits[ci, :], color=colors[mp], label=mp)
        ax.set_title(sz)
        ax.set_aspect('equal')
        ax.plot([0, 1], [0, 1], linestyle=':', color='k', lw=1)
    ax.legend(bbox_to_anchor=(1, 1.2), loc='lower right', title=param, ncol=5)
    pl.subplots_adjust(left=0.1, right=0.9, bottom=0.2, hspace=0.5, wspace=0.5, top=0.7)
    return fig

def data_matrix_from_auc(auc_, param='morphlevel', normalize=False):
    auc_['n_chooseB'] = auc_['AUC'] * auc_['n_trials']
    auc_['n_chooseB'] = np.ceil(auc_['n_chooseB']) #.astype(int)
    
    if normalize:
        maxv = float(auc_[param].max())
        auc_[param] = auc_[param]/maxv
    
    sort_cols = [param]
    if 'size' in auc_.columns:
        sort_cols.append('size')
        
    data = auc_.sort_values(by=sort_cols)[[param, 'n_chooseB', 'n_trials']].values

    return data

def aggregate_AUC(DATA, SDF, param='morphlevel', midp=53, reverse_eff=False,
                  selective_only=False, selective_df=None, create_new=False):
    tmp_res = '/n/coxfs01/julianarhee/aggregate-visual-areas/data_stats/tmp_data/AUC.pkl'
    if not create_new:
        try:
            with open(tmp_res, 'rb') as f:
                AUC = pkl.load(f, encoding='latin1')
                print(AUC.head())
        except Exception as e:
            create_new = True

    if create_new:
        print("... creating new AUC dfs")
        AUC = create_auc_dataframe(DATA, SDF, param=param, midp=midp)
        with open(tmp_res, 'wb') as f:
            pkl.dump(AUC, f, protocol=2)

    if selective_only:
        print("... getting selective only")
        assert selective_df is not None, \
            "[ERR]. Requested SELECTIVE ONLY. Must provide selective_df. Aborting."
        
        mAUC = AUC.copy()
        AUC = pd.concat([mAUC[(mAUC.visual_area==va) & (mAUC.datakey==dk) & (mAUC['cell'].isin(sg['cell'].unique()))] \
                for (va, dk), sg in selective_df.groupby(['visual_area', 'datakey'])])
        
    if reverse_eff:
        print("... reversing")
        mAUC = AUC.copy()
        to_reverse = mAUC[mAUC['Eff']==0].copy()
        for (va, dk, c, sz), auc_ in to_reverse.groupby(['visual_area', 'datakey', 'cell', 'size']):
            # flip
            AUC.loc[auc_.index, 'AUC'] = auc_['AUC'].values[::-1]

    return AUC


def create_auc_dataframe(DATA, SDF, param='morphlevel', midp=53):
    '''
    DATA: neuraldata dataframe (all data)
    SDF: dict of stimconfig dfs
    reverse_eff:  set False to allow negative sigmoid
    selective_only:  Must provide dataframe of selective cells if True
    '''
    a_=[]
    DATA['cell'] = DATA['cell'].astype(int)
    for (va, dk), nd in DATA.groupby(['visual_area', 'datakey']):

        # get selective cells
#         if selective_only:
#             seldf = selective_df[(selective_df.visual_area==va) & (selective_df.datakey==dk)].copy()
#             sel_cells = seldf['cell'].unique().astype(int)
#         else:
        sel_cells = nd['cell'].unique().astype(int)

        # add stimulus info
        sdf = SDF[dk].copy()
        morphlevels = sdf['morphlevel'].unique()
        max_morph = max(morphlevels)

        sizes = list(sdf['size'].unique())
        if midp not in morphlevels:
            print("... (%s, %s) Unknown midpoint in morphs: %s" % (dk, va, str(morphlevels)))
            continue
        nd['size'] = [sdf['size'][c] for c in nd['config']]
        nd['morphlevel'] = [sdf['morphlevel'][c] for c in nd['config']]
        ndf = nd[(nd['cell'].isin(sel_cells)) & (nd['morphlevel']!=-1)].copy()

        morph_lut, a_morphs, b_morphs = get_morph_levels(levels=morphlevels, midp=midp)
        # update neuraldata
        ndf['morphstep'] = [morph_lut[m] for m in ndf['morphlevel']]
        ndf['morph_ix'] = [m/float(max_morph) for m in ndf['morphlevel']]

        ndf['object'] = None
        ndf.loc[ndf.morphlevel.isin(a_morphs), 'object'] = 'A'
        ndf.loc[ndf.morphlevel.isin(b_morphs), 'object'] = 'B'
        ndf.loc[ndf.morphlevel==midp, 'object'] = 'M'

        # calculate AUCs
        # reverse_eff = param!='morphstep'
        AUC0 = ndf.groupby('cell').apply(get_auc_AB, param=param, reverse_eff=False)
        AUC0['visual_area'] = va
        AUC0['datakey'] = dk
        a_.append(AUC0)

    mAUC = pd.concat(a_).reset_index() #.drop('level_1', axis=1)
    
    return mAUC
        

def do_neurometric(curr_visual_area, curr_datakey, param='morphlevel', 
                    sigmoid='gauss', fit_experiment='2AFC', allow_negative=True,
                    max_auc=0.70, fit_new=False, create_auc=False, 
                    traceid='traces001', responsive_test='ROC', responsive_thr=0.05,
                    experiment='blobs'):
    # Load source data
    DATA, SDF, selective_df = get_data(traceid=traceid, 
                    responsive_test=responsive_test, responsive_thr=responsive_thr)
    assert curr_visual_area in DATA['visual_area'].unique(), "Visual area <%s> not in DATA." % curr_visual_area
    assert curr_datakey in DATA['datakey'].unique(), "Datakey <%s> not in DATA" % curr_datakey

    fit_neurometric_curves(curr_visual_area, curr_datakey, DATA, SDF, 
                            param=param, sigmoid=sigmoid, max_auc=max_auc, 
                            fit_experiment=fit_experiment, allow_negative=allow_negative,
                            fit_new=fit_new, create_auc=create_auc,
                            traceid=traceid, experiment=experiment)
    return


def fit_neurometric_curves(curr_visual_area, curr_datakey, DATA, SDF, 
                            param='morphlevel', sigmoid='gauss', fit_experiment='2AFC',
                            allow_negative=True,
                            max_auc=0.7, fit_new=False, create_auc=False,
                            traceid='traces001', experiment='blobs'):
    fitopts = dict()
    fitopts['expType'] = fit_experiment
    fitopts['threshPC'] = 0.5 

    # Load AUC
    AUC = aggregate_AUC(DATA,SDF, param=param, midp=53,reverse_eff=False,
                        selective_only=False, selective_df=None, create_new=create_auc)

    currAUC = AUC[(AUC.visual_area==curr_visual_area) & (AUC.datakey==curr_datakey)].copy()

    # Set output dir
    traceid_dir = get_tracedir_from_datakey(curr_datakey, traceid=traceid, experiment=experiment)
    curr_dst_dir = os.path.join(traceid_dir, 'neurometric', 'fits')
    if not os.path.exists(curr_dst_dir):
        os.makedirs(curr_dst_dir)
    print("Saving ROI results to:\n   %s" % curr_dst_dir)

    # Cells that pass performance criterion
    pass_cells = currAUC[currAUC['AUC']>=max_auc]['cell'].unique()

    print("%i of %i cells pass crit (%.2f)" % (len(pass_cells), len(currAUC['cell'].unique()), max_auc))
    pass_auc = currAUC[currAUC['cell'].isin(pass_cells)].copy()
    if len(pass_cells)==0:
        print("****[%s, %s] no cells." % (va, dk))
        return 

    if create_auc or fit_new:
        old_fns = glob.glob(os.path.join(curr_dst_dir, '*_rid*.pkl'))
        print("~~~~ deleting %i old files ~~~~~" % len(old_fns))
        for o_ in old_fns:
            os.remove(o_)

    for ri, (rid, auc_r) in enumerate(pass_auc.groupby(['cell'])):
        if ri%10==0:
            print("... fitting %i of %i cells (%s)" % (int(ri+1), len(pass_cells), sigmoid))

        Eff = int(auc_r['Eff'].unique())
        sigmoid_ = 'neg_%s' % sigmoid if (Eff==0 and allow_negative) else sigmoid
        fitopts['sigmoidName'] = sigmoid_

        fn = '%s_rid%03d.pkl' % (sigmoid_, rid)
        outfile = os.path.join(curr_dst_dir, fn)
        if os.path.exists(outfile):
            continue

        results={}
        for sz, auc_sz in auc_r.groupby(['size']):
            # format data
            data = data_matrix_from_auc(auc_sz, param=param)

            # fit
            res = ps.psignifit(data, fitopts)
            results[sz] = res

        fn = '%s_rid%03d.pkl' % (sigmoid_, rid)
        outfile = os.path.join(curr_dst_dir, fn)

        with open(outfile, 'wb') as f:
            pkl.dump(results, f, protocol=2)
        print("... saved: %s" % fn)

    print("DONE!")
    return None

def extract_options(options):
    parser = optparse.OptionParser()

    # PATH opts:
    parser.add_option('-D', '--root', action='store', dest='rootdir', default='/n/coxfs01/2p-data', 
                      help='root project dir containing all animalids [default: /n/coxfs01/2pdata]')
  
    # Set specific session/run for current animal:
    parser.add_option('-E', '--experiment', action='store', dest='experiment', default='blobs', 
                      help="experiment name (e.g,. gratings, rfs, rfs10, or blobs)") 

    choices_e = ('stimulus', 'firsthalf', 'plushalf', 'baseline')
    default_e = 'stimulus'
    parser.add_option('-e', '--epoch', action='store', dest='epoch', 
            default=default_e, type='choice', choices=choices_e,
            help="Trial epoch to average, choices: %s. (default: %s" % (choices_e, default_e))

    parser.add_option('-t', '--traceid', action='store', dest='traceid', default='traces001', 
                      help="traceid (default: traces001)")

    choices_c = ('all', 'ROC', 'nstds', None, 'None')
    default_c = 'ROC'
    parser.add_option('-R', '--responsive_test', action='store', dest='responsive_test', 
            default=default_c, type='choice', choices=choices_c,
            help="Responsive test, choices: %s. (default: %s" % (choices_c, default_c))

    parser.add_option('--thr', action='store', dest='responsive_thr', default=0.05, 
                      help="responsive test thr (default: 0.05 for ROC)")

    parser.add_option('-d', '--response', action='store', dest='response_type', default='corrected', 
                      help="response type (default: corrected, for AUC)")
    parser.add_option('--nstds', action='store', dest='nstds_above', default=2.5, 
                      help="only for test=nstds, N stds above (default: 2.5)")
    parser.add_option('--auc', action='store_true', dest='create_auc', default=False, 
                      help="flag to create AUC dataframes anew")
    parser.add_option('--new', action='store_true', dest='fit_new', default=False, 
                      help="flag to fit each cell anew")
    
    parser.add_option('-n', '--nproc', action='store', dest='n_processes', 
                      default=1)


    parser.add_option('-p', '--param', action='store', dest='param',
                      default='morphlevel', help='Physical param for neurometric curve fit (default: morphlevel)')
    parser.add_option('-s', '--sigmoid', action='store', dest='sigmoid',
                      default='gauss', help='Sigmoid to fit (default: gauss)')
    parser.add_option('-c', '--max-auc', action='store', dest='max_auc',
                      default=0.7, help='Criterion value, max AUC (default: 0.7)')
    parser.add_option('-f', '--fit-experiment', action='store', dest='fit_experiment',
                      default='2AFC', help='Experiment type to fit (default: 2AFC)')
    parser.add_option('--reverse', action='store_false', dest='allow_negative',
                      default=True, help='Set flag to fit all to same func (default: allows neg sigmoid fits)')




    parser.add_option('-V', '--visual-area', action='store', dest='curr_visual_area',
                      default=None, help='Visual area to fit (default: None)')
    parser.add_option('-k', '--datakey', action='store', dest='curr_datakey',
                      default=None, help='Datakey to fit (default: None)')




    (options, args) = parser.parse_args(options)

    return options


responsive_test='ROC'
responsive_thr=10. if responsive_test=='nstds' else 0.05
experiment='blobs'

traceid = 'traces001'
rootdir = '/n/coxfs01/2p-data'

fit_new=False
create_auc=False
param = 'morphlevel'
max_auc = 0.70

curr_visual_area = 'V1'
curr_datakey = '20190616_JC097_fov2'

def main(options):
    opts = extract_options(options)
    rootdir = opts.rootdir
    experiment = opts.experiment
    responsive_test=opts.responsive_test
    responsive_thr = float(opts.responsive_thr)
    traceid = opts.traceid
    
    max_auc = float(opts.max_auc)
    curr_visual_area = opts.curr_visual_area
    curr_datakey = opts.curr_datakey
    create_auc = opts.create_auc
    fit_new = opts.fit_new
    
    param = opts.param
    sigmoid = opts.sigmoid
    fit_experiment = opts.fit_experiment
    allow_negative = opts.allow_negative
 
    do_neurometric(curr_visual_area, curr_datakey, param=param, sigmoid=sigmoid, 
                    fit_experiment=fit_experiment, allow_negative=allow_negative,
                    max_auc=max_auc, fit_new=fit_new, create_auc=create_auc,
                    traceid=traceid, responsive_test=responsive_test, responsive_thr=responsive_thr,
                    experiment=experiment)

    print('.done.')


if __name__ == '__main__':
    main(sys.argv[1:])
