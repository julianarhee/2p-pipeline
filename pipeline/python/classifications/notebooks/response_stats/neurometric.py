import os
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
import matplotlib as mpl
from matplotlib.lines import Line2D
import py3utils as p3

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)       

import re

def get_rid_from_str(s, ndec=3):
    #print(re.findall(r"rid\d{%s}" % ndec, s)[0][3:])
    return int(re.findall(r"rid\d{%s}" % ndec, s)[0][3:])

def load_fitparams(dk, roi_list=None, allow_negative=True, param='morphlevel', 
                  split_pupil=False, sigmoid='gauss', return_dicts=False, return_missing=False):
    results={}
    roifits=None
    
    sigmoid_dir = sigmoid if allow_negative else '%s_reverse' % sigmoid
    prefix = '%s_meanAUC_' % param if split_pupil else '%s_' % param
    fit_subdir = 'split_pupil/fits/%s' % sigmoid_dir if split_pupil else 'fits/%s' % sigmoid_dir

    if split_pupil:
        return_dicts=False # no dict saving here 
    
    if param=='morphstep':
        single_eff=False
        allow_negative=False
    
    traceid_dir = get_tracedir_from_datakey(dk)
    if roi_list is None:
        roi_fit_fns = glob.glob(os.path.join(traceid_dir, 'neurometric', fit_subdir, '%srid*.pkl' % prefix))
    else:
        roi_fit_fns = [os.path.join(traceid_dir, 'neurometric', fit_subdir, '%srid%03d.pkl' % (prefix, rid)) \
                      for rid in roi_list]
        
    r_=[]; missing=[];
    for rfn in roi_fit_fns:
        fn = os.path.split(rfn)[-1]
        rid = get_rid_from_str(fn)
        try:
            with open(rfn, 'rb') as f:
                rd = pkl.load(f)
        except Exception as e:
            #print(e)
            missing.append(rid)
            continue
        if return_dicts:
            assert 'results' in rd.keys(), "No results dict found, skipping [%s]" % fn
            results[rid] = rd['results']    
        r_.append(rd if split_pupil else rd['pars'])
    if len(r_)>0:
        roifits = pd.concat(r_).reset_index(drop=True)
   
    #print("Missing %i cells" % len(missing)) 
    if return_dicts:
        if return_missing:
            return roifits, results, missing
        else:
            return roifits, results
    else:
        if return_missing:
            return roifits, missing
        else:
            return roifits
    
    
    
# #####################################################################
# Data sourcing
# #####################################################################
def get_tracedir_from_datakey(datakey, experiment='blobs', 
                    rootdir='/n/coxfs01/2p-data', traceid='traces001'):
    session, animalid, fovn = p3.split_datakey_str(datakey)
    darray_dirs = glob.glob(os.path.join(rootdir, animalid, session, 'FOV%i_zoom2p0x' % fovn,
                            'combined_%s_static' % experiment, 'traces', 
                             '%s*' % traceid, 'data_arrays', 'np_subtracted.npz'))
    assert len(darray_dirs)==1, "More than 1 data array dir found: %s" % str(darray_dirs)
    traceid_dir = darray_dirs[0].split('/data_arrays')[0]
    
    return traceid_dir
     
def load_aggregate_AUC( param='morphlevel', midp=53, allow_negative=True,
                  selective_only=False, selective_df=None, create_new=False,
                  single_eff=True, exclude=['20190314_JC070_fov1'],
                  anchors_=[0, 14, 92, 106]):
 
    '''
    Load previously calculate AUC curves. 
    single_eff, only 1 EFF object allowed (no switching).
    '''
    fname = 'AUC_single' if single_eff else 'AUC'
    tmp_res = '/n/coxfs01/julianarhee/aggregate-visual-areas/data-stats/tmp_data/%s.pkl' % fname
    make_single=False
    if single_eff:
        if not os.path.exists(tmp_res):
            print("No aingle Eff file exists for AUC regular. Creating now.")
            make_single=True
            tmp_res = '/n/coxfs01/julianarhee/aggregate-visual-areas/data-stats/tmp_data/AUC.pkl'
        
    try:
        with open(tmp_res, 'rb') as f:
            AUC = pkl.load(f, encoding='latin1')
            print(AUC.head())
    except Exception as e:
        traceback.print_exc()
        return None

    if 'object' not in AUC.columns:
        AUC['object'] = None
        AUC.loc[AUC['morphlevel']==53, 'object'] = 'M'
        AUC.loc[AUC['morphlevel']<53, 'object'] = 'A'
        AUC.loc[AUC['morphlevel']>53, 'object'] = 'B'

    if single_eff and make_single:
        for (va, dk, c), g in AUC.groupby(['visual_area', 'datakey', 'cell']):
            if 'arousal' in g.columns:
                auc_r = g[g[true_labels]].copy()
            else:
                aucr_ = g.copy()
            objects = [i for i, ar in auc_r.groupby(['object'])]
            pref_object = objects[auc_r[auc_r['morphlevel'].isin(anchors_)]
                                .groupby(['object'])['AUC'].mean().argmax()]
            sEff = 0 if pref_object=='A' else 106
            AUC.loc[g.index, 'Eff'] = sEff 
        # save
        tmp_res = '/n/coxfs01/julianarhee/aggregate-visual-areas/data-stats/tmp_data/AUC_single.pkl'
        with open(tmp_res, 'wb') as f:
            pkl.dump(AUC, f, protocol=2)

    if selective_only:
        print("... getting selective only")
        assert selective_df is not None, \
            "[ERR]. Requested SELECTIVE ONLY. Need selective_df. Aborting."
        
        mAUC = AUC.copy()
        AUC = pd.concat([mAUC[(mAUC.visual_area==va) & (mAUC.datakey==dk) 
                            & (mAUC['cell'].isin(sg['cell'].unique()))] \
                for (va, dk), sg in selective_df.groupby(['visual_area', 'datakey'])])
        
    if allow_negative is False:
        print("... reversing")
        mAUC = AUC.copy()
        to_reverse = mAUC[mAUC['Eff']==0].copy()
        for (va, dk, c, sz), auc_ in to_reverse.groupby(['visual_area', 'datakey', 'cell', 'size']):
            # flip
            AUC.loc[auc_.index, 'AUC'] = auc_['AUC'].values[::-1]

    return AUC



# #####################################################################
# Fitting, plotting
# #####################################################################

from psignifit import getSigmoidHandle as getSig

def default_options():

    options = {'expType': '2AFC',
               'sigmoidName': 'gauss',
               'threshPC': 0.5}
    
    return options

def plot_sigmoid_from_params(fit, options, xmin=0, xmax=106, npoints=50, lc='k',
                            lw=1, label=None, ax=None):
    '''
    Plots 1 curve. Expected to be used w. output of ps.psignifit().
    fit = np.array, [threshold, width, lambda, gamma, eta] (i.e,. res['Fit'] from psignifit)
    options = dict(), **note: options['sigmoidHandle'] should be a function handle
    '''
    xv, fv = fit_sigmoid(fit, options['sigmoidHandle'], xmin=xmin, xmax=xmax, npoints=npoints)
    
    if ax is None:
        fig, ax = pl.subplots()
  
    ax.plot(xv, fv, c=lc, lw=lw, clip_on=False, label=label)
    

def fit_sigmoid(fit, fh, xmin=0, xmax=106, npoints=50):
    '''
    fh:  function handle (call ps.getSigmoidHandle.getSigmoidHandle())
    '''
    xv       = np.linspace(xmin, xmax, num=npoints)
    fv   = (1 - fit[2] - fit[3]) * fh(xv,     fit[0], fit[1]) + fit[3]
    
    return xv, fv

def get_fit_values(fparams, fh, xmin=0, xmax=106, npoints=50):
    '''
    Create dict of fit values (interpolated) for easier plotting by conditions
    '''
    if isinstance(fparams, pd.DataFrame):
        parnames = ['threshold', 'width', 'lambda', 'gamma','eta']
        fit, = fparams[parnames].values
    else:
        fit = fparams
    x, v = fit_sigmoid(fit, fh, xmin=xmin, xmax=xmax, npoints=npoints)
    add_cols= ['visual_area', 'datakey', 'cell', 'size', 'Eff']
    if 'arousal' in fparams.columns:
        add_cols.extend(['arousal', 'true_labels'])
        
    finfo = fparams[add_cols].drop_duplicates()
    
    df_ = pd.DataFrame({'x': x, 'fitv': v})
    for a in add_cols:
        df_[a] = finfo[a].values[0]
        
    return df_




# #####################################################################
# from classifications/neurometric_fits.py
# #####################################################################

def get_morph_levels(midp=53, levels=[-1, 0, 14, 27, 40, 53, 66, 79, 92, 106]):

    a_morphs = np.array(sorted([m for m in levels if m<midp and m!=-1]))[::-1]
    b_morphs = np.array(sorted([m for m in levels if m>midp and m!=-1]))

    d1 = dict((k, list(a_morphs).index(k)+1) for k in a_morphs)
    d2 = dict((k, list(b_morphs).index(k)+1) for k in b_morphs)
    morph_lut = d1.copy()
    morph_lut.update(d2)
    morph_lut.update({midp: 0, -1: -1})

    return morph_lut, a_morphs, b_morphs



def add_morph_info(ndf, sdf, morph_lut, a_morphs, b_morphs, midp=53):
    # add stimulus info
    morphlevels = sdf['morphlevel'].unique()
    max_morph = max(morphlevels)

    assert midp in  morphlevels, "... Unknown midpoint in morphs: %s" % str(morphlevels)
        
    ndf['size'] = [sdf['size'][c] for c in ndf['config']]
    ndf['morphlevel'] = [sdf['morphlevel'][c] for c in ndf['config']]
    ndf0=ndf.copy()
    ndf = ndf0[ndf0['morphlevel']!=-1].copy()

    morph_lut, a_morphs, b_morphs = get_morph_levels(levels=morphlevels, midp=midp)
    # update neuraldata
    ndf['morphstep'] = [morph_lut[m] for m in ndf['morphlevel']]
    ndf['morph_ix'] = [m/float(max_morph) for m in ndf['morphlevel']]

    ndf['object'] = None
    ndf.loc[ndf.morphlevel.isin(a_morphs), 'object'] = 'A'
    ndf.loc[ndf.morphlevel.isin(b_morphs), 'object'] = 'B'
    ndf.loc[ndf.morphlevel==midp, 'object'] = 'M'

    return ndf



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
    Ineff_obj = 'B' if Eff_obj=='A' else 'A'
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
    resp_A.extend(resp_A_)
    resp_B.extend(resp_B_) 
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
    
    
def calculate_auc(p_hits, p_fas, resp_cfgs1, param='morphlevel'): #, reverse_eff=False, Eff=None):
    if p_fas.shape[0] < p_hits.shape[0]:
        altconds = list(np.unique([c[0] for c in resp_cfgs1]))
        true_auc = [-np.trapz(p_hits[ci, :], x=p_fas[altconds.index(sz), :]) \
                            for ci, (sz, mp) in enumerate(resp_cfgs1)]
    else:
        true_auc = [-np.trapz(p_hits[ci, :], x=p_fas[ci, :]) for ci in np.arange(0, len(resp_cfgs1))]
        
    aucs = pd.DataFrame({'AUC': true_auc, 
                          param: [r[1] for r in resp_cfgs1], 
                         'size': [r[0] for r in resp_cfgs1]}) 
#    if reverse_eff and Eff==0:
#        # flip
#        for sz, a in aucs.groupby(['size']):
#            aucs.loc[a.index, 'AUC'] = a['AUC'].values[::-1]
#        
    return aucs


def get_auc_AB(rdf, param='morphlevel', n_crit=50, 
                include_ref=True, allow_negative=True,
                class_a=0, class_b=106, return_probs=False,
                anchors_=[0, 14, 92, 106], single_eff=False):
    '''
    Calculate AUCs for A vs. B at each size. 
    Note:  rdf must contain columns 'morphstep' and 'size' (morphstep LUT from: get_morph_levels())

    include_ref: include morphlevel=0 (or morph_ixx) by splitting into half.
    Compare p_hit (morph=0) to p_fa (morph=106), calculate AUC.
    '''
    # Get Eff/Ineff
    objects = [i for i, ar in rdf.groupby(['object'])]
    max_ix = rdf[rdf.morphlevel.isin(anchors_)]\
                .groupby(['object'])['response'].mean().argmax()
    pref_obj = objects[max_ix]
    Eff = class_a if pref_obj=='A' else  class_b
    rdf = p3.equal_counts_df(rdf, equalize_by='config')

    p_hits, p_fas, resp_cfgs, counts = split_signal_distns(rdf, param=param, n_crit=n_crit, 
                                                        include_ref=include_ref, Eff=Eff)
        
    aucs =  calculate_auc(p_hits, p_fas, resp_cfgs, param=param)#, 
#                             reverse_eff=not(allow_negative)) #, Eff=Eff)
    if single_eff:
        aucs['Eff'] = Eff
    else:
        aucs['Eff'] = None
        for sz, ac in aucs.groupby(['size']):
            max_ix = rdf[(rdf['size']==sz) & (rdf.morphlevel.isin(anchors_))]\
                        .groupby(['object'])['response'].mean().argmax()
            Eff = class_a if objects[max_ix]=='A' else class_b
            aucs.loc[ac.index, 'Eff'] = Eff
            if Eff==0 and allow_negative is False:
                # flip
                aucs.loc[ac.index, 'AUC'] = ac['AUC'].values[::-1]
               
    aucs['n_trials'] = counts
    # aucs['Eff'] = Eff
    
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
    auc_['n_chooseB'] = [round(i) for i in auc_['n_chooseB'].values] #.astype(int)
    
    if normalize:
        maxv = float(auc_[param].max())
        auc_[param] = auc_[param].values/maxv
    
    sort_cols = [param]
    if 'size' in auc_.columns:
        sort_cols.append('size')
        
    data = auc_.sort_values(by=sort_cols)[[param, 'n_chooseB', 'n_trials']].values

    return data
