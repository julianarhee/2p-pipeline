import os
import glob
import cv2

import numpy as np
import pandas as pd
import pylab as pl
import cPickle as pkl
import scipy.stats as spstats
import seaborn as sns
import traceback

import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib import colors as mcolors
from pipeline.python.utils import convert_range, get_screen_dims, isnumber
# ------------------------------------------------------------------------------------
# General stats
# ------------------------------------------------------------------------------------
def get_aniso_index(r_df):
    '''
    Measure of horizontally vs. vertically anisotropic (-1=horizontal, 1=vertical).
    Note:  0 can be either isotropic or oblique.
    
    theta_Mm_c : converted theta, range -90 to 90 [see aggregate_rf_data()]
    anisotropy : range [0, 1], where 0=isotropic and 1=anisotropic
    
    Convert abs(sin(theta)) range from [0, 1] to [-1, 1], then multiply by anisotropy index
    '''
    sins = abs(np.sin(r_df['theta_Mm_c']))
    sins_c = convert_range(sins, oldmin=0, oldmax=1, newmin=-1, newmax=1)
    r_df['aniso_index'] = sins_c * r_df['anisotropy']
    # print(r_df['aniso_index'].min(), r_df['aniso_index'].max())
   
    return r_df

def assign_saturation(hue_param, saturation_param, cmap='hsv', min_v=0, max_v=1):
    norm = mpl.colors.Normalize(vmin=min_v, vmax=max_v)
    scalar_cmap = cm.ScalarMappable(norm=norm, cmap=cmap)

    norm_ai = (saturation_param - min_v) / (max_v - min_v)
    theta_rgb = scalar_cmap.to_rgba(abs(np.sin(hue_param)))
    theta_hsv = mcolors.rgb_to_hsv(theta_rgb[0:3])
    theta_hsv[1] = norm_ai
    theta_col = mcolors.hsv_to_rgb(theta_hsv)     
    return theta_col

def compare_rf_size(df, metric='avg_size', cdf=False, ax=None, alpha=1, lw=2,
                   area_colors=None, visual_areas=['V1', 'Lm', 'Li']):
    if area_colors is None:
        visual_areas = ['V1', 'Lm', 'Li']
        colors = ['magenta', 'orange', 'dodgerblue'] #sns.color_palette(palette='colorblind') #, n_colors=3)
        area_colors = {'V1': colors[0], 'Lm': colors[1], 'Li': colors[2]}


    if ax is None:
        fig, ax = pl.subplots(figsize=(6,4))
        fig.patch.set_alpha(1)

    for visual_area in visual_areas:
        nrats = len(df[df['visual_area']==visual_area]['animalid'].unique())
        ncells_total = df[df['visual_area']==visual_area].shape[0]
        values = df[df['visual_area']==visual_area]['%s' % metric].values
        weights = np.ones_like(values)/float(len(values))
        ax.hist(values, 
                cumulative=cdf,
                label='%s (n=%i rats, %i cells)' % (visual_area, nrats, ncells_total),
                color=area_colors[visual_area],
                histtype='step', alpha=alpha, lw=lw,
                normed=0, weights=weights)
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=8)
    #sns.despine(ax=ax, trim=True, offset=2)
    ax.set_xlabel(metric) #'average size (deg)')
    if cdf:
        ax.set_ylabel('CDF')
    else:
        ax.set_ylabel('fraction')
        
    return ax


def plot_all_rfs(RFs, MEANS, screeninfo, cmap='cubehelix', dpi=150):
    '''
    Plot ALL receptive field pos, mark CoM by FOV. Colormap = datakey.
    One subplot per visual area.
    '''
    screenright = float(screeninfo['azimuth_deg']/2)
    screenleft = -1*screenright #float(screeninfo['screen_right'].unique())
    screentop = float(screeninfo['altitude_deg']/2)
    screenbottom = -1*screentop
    screenaspect = float(screeninfo['resolution'][0]) / float(screeninfo['resolution'][1])


    visual_areas = ['V1', 'Lm', 'Li']
    is_split_by_area = 'V1' in MEANS.keys()

    fig, axn = pl.subplots(1,3, figsize=(10,8), dpi=dpi)
    for visual_area, v_df in RFs.groupby(['visual_area']):
        ai = visual_areas.index(visual_area)
        ax = axn[ai]
        dcolors = sns.color_palette(cmap, n_colors=len(v_df['datakey'].unique()))
        for di, (datakey, d_df) in enumerate(v_df.groupby(['datakey'])):
           
            if is_split_by_area:
                exp_rids = [r for r in MEANS[visual_area][datakey] if isnumber(r)]
            else: 
                exp_rids = [r for r in MEANS[datakey] if isnumber(r)]     
            rf_rids = d_df['cell'].unique()
            common_to_rfs_and_blobs = np.intersect1d(rf_rids, exp_rids)
            curr_df = d_df[d_df['cell'].isin(common_to_rfs_and_blobs)].copy()
            
            sns.scatterplot('x0', 'y0', data=curr_df, ax=ax, color=dcolors[di],
                           s=10, marker='o', alpha=0.5) 

            x = curr_df['x0'].values
            y=curr_df['y0'].values
            
            ncells_rfs = len(rf_rids)
            ncells_common = len(common_to_rfs_and_blobs) #curr_df.shape[0]
            m=np.ones(curr_df['x0'].shape)
            cgx = np.sum(x*m)/np.sum(m)
            cgy = np.sum(y*m)/np.sum(m)
            #print('The center of mass: (%.2f, %.2f)' % (cgx, cgy))
            ax.plot(cgx, cgy, marker='+', markersize=20, color=dcolors[di], 
                    label='%s (%s, %i/%i)' 
                            % (visual_area, datakey, ncells_common, ncells_rfs), lw=3) 
        ax.set_title(visual_area)
        ax.legend(bbox_to_anchor=(0.95, -0.4), fontsize=8) #1))

    for ax in axn:
        ax.set_xlim([screenleft, screenright])
        ax.set_ylim([screenbottom, screentop])
        ax.set_aspect('equal')
        ax.set_ylabel('')
        ax.set_xlabel('')
        
    pl.subplots_adjust(top=0.9, bottom=0.4)

    return fig




# ------------------------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------------------------
def load_aggregate_rfs(rf_dsets, traceid='traces001', 
                        fit_desc='fit-2dgaus_dff-no-cutoff', 
                        reliable_only=True, verbose=False):
    rf_dpaths, no_fits = get_fit_dpaths(rf_dsets, traceid=traceid, fit_desc=fit_desc)
    rfdf = aggregate_rf_data(rf_dpaths, reliable_only=reliable_only, 
                                        fit_desc=fit_desc, traceid=traceid, verbose=verbose)
    rfdf = rfdf.reset_index(drop=True)
    return rfdf


def get_rf_positions(rf_dsets, df_fpath, traceid='traces001', 
                        fit_desc='fit-2dgaus_dff-no-cutoff', reliable_only=True, verbose=False):
    from pipeline.python.rois.utils import load_roi_coords

    rfdf = load_aggregate_rfs(rf_dsets, traceid=traceid, fit_desc=fit_desc, 
                                reliable_only=reliable_only, verbose=verbose)
    get_positions = False
    if os.path.exists(df_fpath) and get_positions is False:
        print("Loading existing RF coord conversions...")
        try:
            with open(df_fpath, 'rb') as f:
                df= pkl.load(f)
            rfdf = df['df']
        except Exception as e:
            get_positions = True

    if get_positions:
        print("Calculating RF coord conversions...")
        pos_params = ['fov_xpos', 'fov_xpos_pix', 'fov_ypos', 'fov_ypos_pix', 'ml_pos','ap_pos']
        for p in pos_params:
            rfdf[p] = ''
        p_list=[]
        for (animalid, session, fovnum), g in rfdf.groupby(['animalid', 'session', 'fovnum']):
            fcoords = load_roi_coords(animalid, session, 'FOV%i_zoom2p0x' % fovnum, 
                                      traceid=traceid, create_new=False)

            for ei, e_df in g.groupby(['experiment']):
                cell_ids = e_df['cell'].unique()
                p_ = fcoords['roi_positions'].loc[cell_ids]
                for p in pos_params:
                    rfdf[p][e_df.index] = p_[p].values
        with open(df_fpath, 'wb') as f:
            pkl.dump(rfdf, f, protocol=pkl.HIGHEST_PROTOCOL)
    return rfdf


# OVERLAPS
def pick_rfs_with_most_overlap(rfdf, MEANS):
    r_list=[]
    for datakey, expdf in MEANS.items(): #corrs.groupby(['datakey']):
        # Get active blob cells
        exp_rids = [r for r in expdf.columns if isnumber(r)]     
        # Get current fov's RFs
        rdf = rfdf[rfdf['datakey']==datakey].copy()
        
        # If have both rfs/rfs10, pick the best one
        if len(rdf['experiment'].unique())>1:
            rf_rids = rdf[rdf['experiment']=='rfs']['cell'].unique()
            rf10_rids = rdf[rdf['experiment']=='rfs10']['cell'].unique()
            same_as_rfs = np.intersect1d(rf_rids, exp_rids)
            same_as_rfs10 = np.intersect1d(rf10_rids, exp_rids)
            rfname = 'rfs' if len(same_as_rfs) > len(same_as_rfs10) else 'rfs10'
            print("%s: Selecting %s, overlappig rfs, %i | rfs10, %i (of %i cells)" 
                  % (datakey, rfname, len(same_as_rfs), len(same_as_rfs10), len(exp_rids)))
            r_list.append(rdf[rdf['experiment']==rfname])
        else:
            r_list.append(rdf)
    RFs = pd.concat(r_list, axis=0)

    return RFs

def calculate_overlaps(RFs, datakeys=None, experiment='blobs'):
    from pipeline.python.classifications import experiment_classes as util
    from pipeline.python.classifications.aggregate_data_stats import add_meta_to_df
    
    rf_fit_params = ['cell', 'std_x', 'std_y', 'theta', 'x0', 'y0']
    if datakeys is None:
        datakeys=RFs['datakey'].unique()

    o_list=[]
    for (visual_area, animalid, session, fovnum, datakey), g in RFs.groupby(['visual_area', 'animalid', 'session', 'fovnum', 'datakey']):  
        if datakey not in datakeys: #MEANS.keys():
            continue
        
        # Convert RF fit params to polygon
        rfname = g['experiment'].unique()[0]
        #print(rfname) 
        g.index = g['cell'].values
        rf_polys = rfs_to_polys(g[rf_fit_params])

        S = util.Session(animalid, session, 'FOV%i_zoom2p0x' % fovnum, get_anatomical=False)
        stim_xpos, stim_ypos = S.get_stimulus_coordinates(experiments=[experiment])
        stim_sizes = S.get_stimulus_sizes(size_tested=[experiment])

        # Convert stimuli to polyon bounding boxes
        stim_polys = [(blob_sz, stimsize_poly(blob_sz, xpos=stim_xpos, ypos=stim_ypos))                   for blob_sz in stim_sizes[experiment]]
        
        # Get all pairwise overlaps (% of smaller ellipse that overlaps larger ellipse)
        overlaps = pd.concat([get_proportion_overlap(rf_poly, stim_poly) \
                    for stim_poly in stim_polys \
                    for rf_poly in rf_polys]).rename(columns={'row': 'cell', 'col': 'stim_size'})
        metadict={'visual_area': visual_area, 'animalid': animalid, 'rfname': rfname,
                  'session': session, 'fovnum': fovnum, 'datakey': datakey}
        o_ = add_meta_to_df(overlaps, metadict)
        o_list.append(o_)

    stim_overlaps = pd.concat(o_list, axis=0).reset_index(drop=True)
    return stim_overlaps





def aggregate_rf_dataframes(filter_by, fit_desc=None, scale_sigma=True, fit_thr=0.5, 
                            traceid='traces001',
                            reliable_only=True, verbose=False,
                            fov_type='zoom2p0x', state='awake', stimulus='rfs', 
                            excluded_sessions = ['JC110_20191004_FOV1_zoom2p0x',
                                                 'JC080_20190602_FOV1_zoom2p0x',
                                                 'JC113_20191108_FOV1_zoom2p0x', 
                                                 'JC113_20191108_FOV2_zoom2p0x']):
    
    from pipeline.python.classifications import aggregate_data_stats as aggr
                            
    assert fit_desc is not None, "No fit_desc provided!"
    #### Get metadata
    dsets = aggr.get_metadata(traceid=traceid, fov_type=fov_type, state=state, 
                                  filter_by=filter_by, stimulus='rfs')
    
    rf_dsets = dsets[dsets['experiment'].isin(['rfs', 'rfs10'])].copy()

    #### Check for any datasets that need RF fits
    rf_dpaths, _ = get_fit_dpaths(rf_dsets, traceid=traceid, fit_desc=fit_desc, 
                                              excluded_sessions=excluded_sessions)

    #### Get RF dataframe for all datasets (filter to include only good fits)
    all_df = aggregate_rf_data(rf_dpaths, scale_sigma=scale_sigma, verbose=verbose,
                                reliable_only=reliable_only,
                                fit_desc=fit_desc, traceid=traceid)
    all_df.groupby(['visual_area', 'experiment'])['datakey'].count()

    #### Filter for good fits only
    r_df = all_df[all_df['r2'] > fit_thr].copy().reset_index(drop=True)
    dkey_dict = dict((v, dict((dk, di) for di, dk in enumerate(vdf['datakey'].unique()))) \
                     for v, vdf in r_df.groupby(['visual_area'])) 
    r_df['datakey_ix'] = [dkey_dict[r_df['visual_area'][i]][r_df['datakey'][i]] \
                          for i in r_df.index.tolist()]    
    
    return r_df, dkey_dict


def get_fit_dpaths(dsets, traceid='traces001', fit_desc=None,
                    excluded_sessions = ['JC110_20191004_FOV1_zoom2p0x',
                                         'JC080_20190602_FOV1_zoom2p0x',
                                         'JC113_20191108_FOV1_zoom2p0x', 
                                         'JC113_20191108_FOV2_zoom2p0x'],
                    rootdir='/n/coxfs01/2p-data'):
    '''
    rfdata: (dataframe)
        Metadata (subset of 'sdata') of all datasets to include in current analysis
        
    Gets paths to fit_results.pkl, which contains all (fit-able) results for each cell.
    Adds new column of paths to rfdata.
    '''
    assert fit_desc is not None, "No fit-desc specified!"
    
    rfdata = dsets.copy()
    fit_these = []
    dpaths = {}
    unknown = []
    for (visual_area, animalid, session, fov), g in dsets.groupby(['visual_area', 'animalid', 'session', 'fov']): #animalid in rfdata['animalid'].unique():
        skey = '_'.join([animalid, session, fov])
        if skey in excluded_sessions:
            rfdata = rfdata.drop(g.index)
            continue

        rfruns = g['experiment'].unique()
        for rfname in rfruns:
            curr_rfname = 'gratings' if int(session) < 20190511 else rfname
            fpath = glob.glob(os.path.join(rootdir, animalid, session, '*%s' % fov, 
                                        'combined_%s_*' % curr_rfname, 
                                        'traces', '%s*' % traceid, 
                                        'receptive_fields', fit_desc, 
                                        'fit_results.pkl'))
            if len(fpath) > 0:
                assert len(fpath)==1, "Too many paths: %s" % str(fpath)
                dpaths['-'.join([animalid, session, fov, rfname])] = fpath[0]
            elif len(fpath) == 0:
                fit_these.append((animalid, session, fov, rfname))
            else:
                print("[%s] %s - warning: unknown file paths" % (skey, rfname))
    print("N dpaths: %i, N unfit: %i" % (len(dpaths), len(fit_these)))
    print("N datasets included: %i, N sessions excluded: %i" % (rfdata.shape[0], len(excluded_sessions)))
    
    rdata = rfdata.reset_index()
    fillpaths = ['' for _ in range(rfdata.shape[0])]
    for skey, fpath in dpaths.items():
        animalid, session, fov, rfname = skey.split('-')
        df_ix = rdata[ (rdata['animalid']==animalid) \
                           & (rdata['session']==session) \
                           & (rdata['fov']==fov) \
                           & (rdata['experiment']==rfname)].index.tolist()[0]
        fillpaths[df_ix] = fpath
        
    rdata['path'] = fillpaths
    rdata = rdata.drop_duplicates().reset_index(drop=True)
    
    return rdata, fit_these


def aggregate_rf_data(rf_dpaths, fit_desc=None, traceid='traces001', fit_thr=0.5, 
                      scale_sigma=True, sigma_scale=2.35, verbose=False, 
                      response_type='None', reliable_only=True,
                      rootdir='/n/coxfs01/2p-data'):
    '''
    Combines fit_results.pkl(fit from data) and evaluation_results.pkl (evaluated fits via bootstrap)
    and gets fit results only for those cells that are good/robust fits based on bootstrap analysis.
    '''
    from pipeline.python.retinotopy import fit_2d_rfs as fitrf
    from pipeline.python.classifications import evaluate_receptivefield_fits as evalrf


    df_list = []
    for (visual_area, animalid, session, fovnum, experiment), g in rf_dpaths.groupby(['visual_area', 'animalid', 'session', 'fovnum', 'experiment']):
        datakey = '%s_%s_fov%i' % (session, animalid, fovnum) #'-'.join([animalid, session, fovnum])
        #print(datakey)
        fov = 'FOV%i_zoom2p0x' % fovnum
        try:
            #### Load evaluation results (bootstrap analysis of each fit paramater)
            curr_rfname = experiment if int(session)>=20190511 else 'gratings'

            #### Load eval results 
            eval_results, eval_params = evalrf.load_eval_results(
                                                animalid, session, fov,
                                                experiment=curr_rfname,
                                                traceid=traceid, 
                                                fit_desc=fit_desc)   
            if eval_results is None:
                print('-- no good (%s), skipping' % datakey)
                continue
            
            #### Load fit results from measured
            fit_results, fit_params = fitrf.load_fit_results(
                                                animalid, session, fov,
                                                experiment=curr_rfname,
                                                traceid=traceid, 
                                                fit_desc=fit_desc)

            #fit_rois = sorted(fit_results['fit_results'].keys())
            fit_rois = sorted(eval_results['data']['cell'].unique())

            scale_sigma = fit_params['scale_sigma']
            sigma_scale = fit_params['sigma_scale']
            rfit_df = fitrf.rfits_to_df(fit_results, 
                            scale_sigma=scale_sigma, 
                            sigma_scale=sigma_scale,
                            row_vals=fit_params['row_vals'], 
                            col_vals=fit_params['col_vals'], 
                            roi_list=fit_rois)

            #### Identify cells with measured params within 95% CI of bootstrap distN
            pass_rois = rfit_df[rfit_df['r2']>fit_thr].index.tolist()
            param_list = [param for param in rfit_df.columns if param != 'r2']
            # Note: get_good_fits(rfit_df, eval_results, param_list=param_list) returns same
            # as reliable_rois, since checks if all params within 95% CI
            reliable_rois = evalrf.get_reliable_fits(eval_results['pass_cis'],
                                                     pass_criterion='all')
            if verbose:
                print("[%s] %s: %i of %i fit rois pass for all params" % (visual_area, datakey, len(pass_rois), len(fit_rois)))
                print("...... : %i of %i fit rois passed as reliiable" % (len(reliable_rois), len(pass_rois)))

            #### Create dataframe with params only for good fit cells
            keep_rois = reliable_rois if reliable_only else pass_rois
            passdf = rfit_df.loc[keep_rois].copy()
            # "un-scale" size, if flagged
            if not scale_sigma:
                sigma_x = passdf['sigma_x']/sigma_scale
                sigma_y = passdf['sigma_y'] / sigma_scale
                passdf['sigma_x'] = sigma_x
                passdf['sigma_y'] = sigma_y

            tmpmeta = pd.DataFrame({'cell': keep_rois,
                                    'datakey': [datakey for _ in np.arange(0, len(keep_rois))],
                                    'animalid': [animalid for _ in np.arange(0, len(keep_rois))],
                                    'session': [session for _ in np.arange(0, len(keep_rois))],
                                    'fovnum': [fovnum for _ in np.arange(0, len(keep_rois))],
                                    'visual_area': [visual_area for _ in np.arange(0, len(keep_rois))],
                                    'experiment': [experiment for _ in np.arange(0, len(keep_rois))]}, index=passdf.index)

            fitdf = pd.concat([passdf, tmpmeta], axis=1).reset_index(drop=True)
            df_list.append(fitdf)

        except Exception as e:
            print("***ERROR: %s" % datakey)
            traceback.print_exc()
            continue
            
    rfdf = pd.concat(df_list, axis=0).reset_index(drop=True)
    
    rfdf = update_rf_metrics(rfdf, scale_sigma=scale_sigma)

    return rfdf

def update_rf_metrics(rfdf, scale_sigma=True):
    # Include average RF size (average of minor/major axes of fit ellipse)
    if scale_sigma:
        rfdf = rfdf.rename(columns={'sigma_x': 'fwhm_x', 'sigma_y': 'fwhm_y'})
        rfdf['std_x'] = rfdf['fwhm_x']/2.35
        rfdf['std_y'] = rfdf['fwhm_y']/2.35
    else:
        rfdf = rfdf.rename(columns={'sigma_x': 'std_x', 'sigma_y': 'std_y'})
        rfdf['fwhm_x'] = rfdf['std_x']*2.35
        rfdf['fwhm_y'] = rfdf['std_y']*2.35

    rfdf['fwhm_avg'] = rfdf[['fwhm_x', 'fwhm_y']].mean(axis=1)
    rfdf['std_avg'] = rfdf[['std_x', 'std_y']].mean(axis=1)
    rfdf['area'] = np.pi * (rfdf['std_x'] * rfdf['std_y'])

    # Add some additional common info
    #### Split fx, fy for theta comp
    rfdf['fx'] = abs(rfdf[['std_x', 'std_y']].max(axis=1) * np.cos(rfdf['theta']))
    rfdf['fy'] = abs(rfdf[['std_x', 'std_y']].max(axis=1) * np.sin(rfdf['theta']))
    rfdf['ratio_xy'] = rfdf['std_x']/rfdf['std_y']

    # Convert thetas to [-90, 90]
    thetas = [(t % np.pi) - np.pi if 
                    ((np.pi/2.)<t<(np.pi) or (((3./2)*np.pi)<t<2*np.pi)) \
                    else (t % np.pi) for t in rfdf['theta'].values]
    rfdf['theta_c'] = thetas

    # Anisotropy metrics
    #rfdf['anisotropy'] = [(sx-sy)/(sx+sy) for (sx, sy) in rfdf[['std_x', 'std_y']].values]
    # Find indices where std_x < std_y
    swap_ixs = rfdf[rfdf['std_x'] < rfdf['std_y']].index.tolist()

    # Get thetas in deg for plotting (using Ellipse() patch function)
    # Note: regardless of whether std_x or _y bigger, when plotting w/ width=Major, height=minor
    #       or width=std_x, height=std_y, should have correct theta orientation 
    # theta_Mm_deg = Major, minor as width/height, corresponding theta for Ellipse(), in deg.
    rfdf['theta_Mm_deg'] = np.rad2deg(rfdf['theta'].copy())
    rfdf['theta_Mm_deg'][swap_ixs] = [ (theta + 90) % 360 if (90 <= theta < 360) \
                                          else (((theta) % 90) + 90) % 360
                                    for theta in np.rad2deg(rfdf['theta'][swap_ixs].values) ]        

    # Get true major and minor axes 
    rfdf['major_axis'] = [max([sx, sy]) for sx, sy in rfdf[['std_x', 'std_y']].values]
    rfdf['minor_axis'] = [min([sx, sy]) for sx, sy in rfdf[['std_x', 'std_y']].values]

    # Get anisotropy index from these (0=isotropic, >0=anisotropic)
    rfdf['anisotropy'] = [(sx-sy)/(sx+sy) for (sx, sy) in rfdf[['major_axis', 'minor_axis']].values]

    # Calculate true theta that shows orientation of RF relative to major/minor axes
    nu_thetas = [(t % np.pi) - np.pi if ((np.pi/2.)<t<(np.pi) or (((3./2)*np.pi)<t<2*np.pi)) \
                 else (t % np.pi) for t in np.deg2rad(rfdf['theta_Mm_deg'].values) ]
    rfdf['theta_Mm_c'] = nu_thetas


    # Get anisotropy index
    sins = abs(np.sin(rfdf['theta_Mm_c']))
    sins_c = convert_range(sins, oldmin=0, oldmax=1, newmin=-1, newmax=1)
    rfdf['aniso_index'] = sins_c * rfdf['anisotropy']
 
    return rfdf


def get_good_fits(rfit_df, eval_results, param_list=[]):
    '''
    Returns only those ROIs that pass bootstrap test (measured value within 95% of CI).
    Params tested given by parma_list, otherwise, all params.
    
    rfit_df: (dataframe)
        This is fit_results['fit_results'] dict combined into dataframe (sigma should be scaled).
    
    eval_results: (dict)
        These are the CIs, data, boot params for doing RF fit evaluation.
        
    param_list: (list)
        Subset of fitted params to check that they are within the 95% CI of bootstrapped distn. 
        (default is all params)
    '''
    if len(param_list)==0:
        param_list = [param for param in rfit_df.columns if param != 'r2']

    fit_rois = sorted(rfit_df.index.tolist())
    pass_by_param = [[rid for rid in fit_rois if \
                     eval_results['cis']['%s_lower' % fparam][rid] <= rfit_df[fparam][rid] <= eval_results['cis']['%s_upper' % fparam][rid]] \
                    for fparam in param_list]

    pass_rois = list(set.intersection(*map(set, pass_by_param)))
    
    return pass_rois


# ------------------------------------------------------------------------------------
# RF luminance calculations
# ------------------------------------------------------------------------------------

def get_rf_luminances(animalid, session, fovnum, curr_exp, traceid='traces001', response_type='dff',
                            stimulus_dir='/home/julianarhee/Repositories/protocols/physiology/stimuli/images'):
    '''
    Load object stimuli, load RF fits.  Calculate luminance as dot product of image array and RF map array.
    
    Returns:
        rflum_df:  pd.DataFrame() w/ RF luminances for each stimulus condition and ROI's receptive field.
    '''
    from pipeline.python.classifications import experiment_classes as util
    fov = 'FOV%i_zoom2p0x' % fovnum
    
    # Load images
    images = load_image_stimuli(stimulus_dir=stimulus_dir)

    # Get RF/screen param info
    screen_info = get_screen_info(animalid, session, fov)

    # RF fit data
    exp = util.ReceptiveFields(curr_exp, animalid, session, fov, traceid=traceid)
    rfstats, rois_rfs, nrois_total = exp.get_rf_fits(response_type=response_type, fit_thr=0.05)
    rfparams = get_rfparams(screen_info, rfstats)

    # Get stimulus info for objects
    exp = util.Objects(animalid, session, fov, traceid=traceid)
    sdf = exp.get_stimuli()
    sdf = reformat_morph_values(sdf)

    # # Get ROIs for both
    # print("Blobs: %i, RFs: %i" % (len(rois_objects), len(rois_rfs)))
    # roi_list = np.intersect1d(rois_objects, rois_rfs)
    print("%i rois." % len(rois_rfs))

    rflum_df = calculate_rf_luminances(images, rfstats, rfparams, sdf, roi_list=rois_rfs)

    return rflum_df

def calculate_rf_luminances(images, rfstats, rfparams, sdf, roi_list=None):
    from pipeline.python.utils import natural_keys

    if roi_list is None:
        roi_list = rfstats['fit_results'].keys()
    image_list = sorted(images.keys(), key=natural_keys)
    
    # Check for non-image conditions, i.e., full-screen controls
    all_conditions = sorted(sdf['morphlevel'].unique())
    nonimage_conditions = [i for i in all_conditions if 'M%i' % i not in image_list]
    print("Non image conditions:", nonimage_conditions)
    image_list.extend(nonimage_conditions)
    
    pix_per_deg = rfparams['pix_per_deg']
    sizes = sorted(sdf['size'].unique())
    
    rfdf = []
    for curr_object in image_list:
        for size_deg in sizes:
            
            # Transform stimulus image
            lum_level = None
            if curr_object == -1:
                lum_level = float(sdf[(sdf['morphlevel']==-1) & (sdf['size']==size_deg)]['color'])
                imarray = np.ones((rfparams['screen_resolution'][1], rfparams['screen_resolution'][0]))*lum_level*255.
                print(imarray.max().max())
                curr_object_name = 'fullscreen'
            else:
                curr_img = images[curr_object]
                imscreen = transform_stim_image(curr_img, rfparams, size_deg=size_deg)
                curr_object_name = curr_object
                
                # Get arrays in correct orientation for multiplying
                imarray = np.flipud(imscreen).copy()
    
            for rid in roi_list:
                # Transform rfmap to screen
                rfmap = rfstats['fit_results'][rid]['data']
                if rfmap.min().min() < 0:
                    print("NONNEG")
                    rfmap = rfmap - rfmap.min().min()
                    
                rfscreen = transform_rfmap(rfmap, rfparams)
                rfarray = rfscreen.copy()
                lumarray = imarray * rfarray

                # Calculate max possible luminance
                max_brightness = np.ones(imarray.shape)*255.
                max_lum = max_brightness.ravel().dot(rfarray.ravel())

                # Express RF luminance as fraction of RF max brightness
                fraction_lum = lumarray.sum() / max_lum

                rdf = pd.DataFrame({'object': curr_object_name,
                                    'size': size_deg,
                                    'RF_luminance': fraction_lum,
                                    'rid': rid}, index=[rid])
                rfdf.append(rdf)

    rfdf = pd.concat(rfdf, axis=0)
    return rfdf


def plot_luminance_calculation(imarray, rfarray, lumarray, rfparams,
                               rf_cmap='hot', lum_cmap='jet'):
    rf_cmap = 'hot'
    lum_cmap = 'jet'
    #plot_roi = False
    fig, axes = pl.subplots(1, 3, figsize=(15,3))

    axes[0].imshow(imarray, origin='bottom', alpha=1, cmap='gray')
    axes[1].imshow(rfarray, origin='bottom', alpha=1, cmap=rf_cmap)
    axes[2].imshow(lumarray, origin='bottom', alpha=1, cmap=lum_cmap)

    for ax in axes:

        # Draw cells for RF tiling boundaries
        for rv in rfparams['col_vals_pix']:
            ax.axvline(rv - pix_per_deg*rfparams['spacing']/2., color='w', lw=0.5)
        ax.axvline(rv + pix_per_deg*rfparams['spacing']/2., color='w', lw=0.5)
        for rv in rfparams['row_vals_pix']:
            ax.axhline(rv - pix_per_deg*rfparams['spacing']/2., color='w', lw=0.5)
        ax.axhline(rv + pix_per_deg*rfparams['spacing']/2., color='w', lw=0.5)

        # Label coordinates
        ax.set_xticks(rfparams['col_vals_pix'])
        ax.set_xticklabels([int(i) for i in rfparams['col_vals']], fontsize=6)

        ax.set_yticks(rfparams['row_vals_pix'])
        ax.set_yticklabels([int(i) for i in rfparams['row_vals']], fontsize=6)
        
    return fig

from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm
import matplotlib.cm as cm

def create_color_bar(fig, hue_colors, hue_values, hue_param='label', #cmap='cube_helix', 
                     orientation='horizontal', cbar_axes=[0.58, 0.17, 0.3, 0.02]):

    cmap = ListedColormap(hue_colors)
    bounds = np.arange(0, len(hue_values))
    norm = BoundaryNorm(bounds, cmap.N)
    mappable = cm.ScalarMappable(cmap=cmap)
    mappable.set_array(bounds)

    cbar_ax = fig.add_axes(cbar_axes)
    cbar = fig.colorbar(mappable, cax=cbar_ax, boundaries=np.arange(-0.5,len(hue_values),1), \
                        ticks=bounds, norm=norm, orientation='horizontal')
    cbar.ax.tick_params(axis='both', which='both',length=0)
    cbar.ax.set_xticklabels(hue_values, fontsize=6) #(['%i' % i for i in morphlevels])  # horizontal colorbar
    cbar.ax.set_xlabel(hue_param, fontsize=12)

    return cbar

# ------------------------------------------------------------------------------------
# Generic screen/stimuli functions
# ------------------------------------------------------------------------------------
def reformat_morph_values(sdf):
    control_ixs = sdf[sdf['morphlevel']==-1].index.tolist()
    sizevals = np.array([round(s, 1) for s in sdf['size'].unique() if s not in ['None', None] and not np.isnan(s)] )
    sdf.loc[sdf.morphlevel==-1, 'size'] = pd.Series(sizevals, index=control_ixs)
    sdf['size'] = [round(s, 1) for s in sdf['size'].values]

    return sdf

def load_image_stimuli(stimulus_dir='/home/julianarhee/Repositories/protocols/physiology/stimuli/images'):
    object_list = ['D1', 'M14', 'M27', 'M40', 'M53', 'M66', 'M79', 'M92', 'D2']

    image_paths = []
    for obj in object_list:
        stimulus_type = 'Blob_%s_Rot_y_fine' % obj
        image_paths.extend(glob.glob(os.path.join(stimulus_dir, stimulus_type, '*_y0.png')))
    print("%i images found for %i objects" % (len(image_paths), len(object_list)))

    images = {}
    for object_name, impath in zip(object_list, image_paths):
        im = cv2.imread(impath)
        if object_name == 'D1':
            object_name = 'M0'
        if object_name == 'D2':
            object_name = 'M106'
        images[object_name] = im[:, :, 0]
    return images


def get_screen_info(animalid, session, fov, rootdir='/n/coxfs01/2p-data'):
    '''
    Load stimulus info for a given session, including coordinates of stimuli.
    '''
    
    from pipeline.python.classifications import experiment_classes as util

    S = util.Session(animalid, session, fov, rootdir=rootdir)

    # Get screen bounds: [bottom left upper right]
    screen_bounds = [S.screen['linminH'], S.screen['linminW'], S.screen['linmaxH'], S.screen['linmaxW']]
    screen_aspect = S.screen['resolution'][0] / S.screen['resolution'][1]

    screen_width_deg = S.screen['linmaxW']*2
    screen_height_deg = S.screen['linmaxH']*2

    pix_per_degW = S.screen['resolution'][0] / screen_width_deg
    pix_per_degH = S.screen['resolution'][1] / screen_height_deg 

    #print(pix_per_degW, pix_per_degH)
    pix_per_deg = np.mean([pix_per_degW, pix_per_degH])
    print("avg pix/deg: %.2f" % pix_per_deg)

    stim_xpos, stim_ypos = S.get_stimulus_coordinates()

    screen_info = S.screen.copy()
    screen_info['stim_pos'] = (stim_xpos, stim_ypos)
    screen_info['pix_per_deg'] = pix_per_deg
    
    return screen_info


# ------------------------------------------------------------------------------------
# RF calculations
# ------------------------------------------------------------------------------------

def get_rfparams(screen_info, rfstats):
    '''
    Get all params necessary for converting screen coordinates and stimulus coordinates.
    Includes info specific to RF mapping experiment.
    '''
    
    rfparams = {'screen_xlim_deg': (screen_info['linminW'], screen_info['linmaxW']),
            'screen_ylim_deg': (screen_info['linminH'], screen_info['linmaxH']),
            'screen_resolution': tuple(screen_info['resolution']),
            'col_vals': rfstats['col_vals'],
            'row_vals': rfstats['row_vals'],
            'spacing': np.diff(rfstats['row_vals']).mean(),\
            'stim_pos': tuple(screen_info['stim_pos']),
               'pix_per_deg': screen_info['pix_per_deg']}
    rfparams = update_rfparams(rfparams)
    
    return rfparams


def update_rfparams(rfparams):
    '''
    Updates all sizing params for screen, visual-degree space, etc. after transforming to 
    coordinate space of MW screen.
    '''

    screen_pix_x, screen_pix_y = rfparams['screen_resolution']
    screen_xmin_deg, screen_xmax_deg = rfparams['screen_xlim_deg']
    screen_ymin_deg, screen_ymax_deg = rfparams['screen_ylim_deg']
    stim_xpos, stim_ypos = rfparams['stim_pos']
    
    # Convert specified stim position to pixel space
    stim_xpos_pix = convert_range(stim_xpos, newmin=0, newmax=screen_pix_x, 
                                  oldmin=screen_xmin_deg, oldmax=screen_xmax_deg)
    stim_ypos_pix = convert_range(stim_ypos, newmin=0, newmax=screen_pix_y, 
                                  oldmin=screen_ymin_deg, oldmax=screen_ymax_deg)

    # Create "screen" array to project image onto
    stim_xpos_pix = int(round(stim_xpos_pix))
    stim_ypos_pix = int(round(stim_ypos_pix))
    #print(stim_xpos_pix, stim_ypos_pix)

    row_vals_pix = [convert_range(rv, newmin=0, newmax=screen_pix_y, 
                    oldmin=screen_ymin_deg, oldmax=screen_ymax_deg) for rv in rfparams['row_vals']]

    col_vals_pix = [convert_range(cv, newmin=0, newmax=screen_pix_x, 
                    oldmin=screen_xmin_deg, oldmax=screen_xmax_deg) for cv in rfparams['col_vals']]

    converted = {'stim_pos_pix': (stim_xpos_pix, stim_ypos_pix),
                 'row_vals_pix': row_vals_pix, 
                 'col_vals_pix': col_vals_pix}
    
    rfparams.update(converted)
    
    return rfparams

def pairwise_compare_single_metric(comdf, curr_metric='avg_size', 
                                    c1='rfs', c2='rfs10', compare_var='experiment',
                                    ax=None, marker='o', visual_areas=['V1', 'Lm', 'Li'],
                                    area_colors=None):
    assert 'datakey' in comdf.columns, "Need a sorter, 'datakey' not found."

    if area_colors is None:
        visual_areas = ['V1', 'Lm', 'Li']
        colors = ['magenta', 'orange', 'dodgerblue'] #sns.color_palette(palette='colorblind') #, n_colors=3)
        area_colors = {'V1': colors[0], 'Lm': colors[1], 'Li': colors[2]}


    offset = 0.25
    
    if ax is None:
        fig, ax = pl.subplots(figsize=(5,4), dpi=dpi)
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
    
    # Plot paired values
    aix=0
    for ai, visual_area in enumerate(visual_areas):

        plotdf = comdf[comdf['visual_area']==visual_area]
        a_vals = plotdf[plotdf[compare_var]==c1].sort_values(by='datakey')[curr_metric].values
        b_vals = plotdf[plotdf[compare_var]==c2].sort_values(by='datakey')[curr_metric].values

        by_exp = [(a, e) for a, e in zip(a_vals, b_vals)]
        for pi, p in enumerate(by_exp):
            ax.plot([aix-offset, aix+offset], p, marker=marker, color=area_colors[visual_area], 
                    alpha=1, lw=0.5,  zorder=0, markerfacecolor=None, 
                    markeredgecolor=area_colors[visual_area])
        tstat, pval = spstats.ttest_rel(a_vals, b_vals)
        print("%s: (t-stat:%.2f, p=%.2f)" % (visual_area, tstat, pval))
        aix = aix+1

    # Plot average
    sns.barplot("visual_area", curr_metric, data=comdf, 
                hue=compare_var, hue_order=[c1, c2], #zorder=0,
                ax=ax, order=visual_areas,
                errcolor="k", edgecolor=('k', 'k', 'k'), facecolor=(1,1,1,0), linewidth=2.5)
    ax.legend_.remove()

    set_split_xlabels(ax, a_label=c1, b_label=c2)
    
    return ax

def set_split_xlabels(ax, offset=0.25, a_label='rfs', b_label='rfs10', rotation=0, ha='center'):
    ax.set_xticks([0-offset, 0+offset, 1-offset, 1+offset, 2-offset, 2+offset])
    ax.set_xticklabels([a_label, b_label, a_label, b_label, a_label, b_label], rotation=rotation, ha=ha)
    ax.set_xlabel('')
    ax.tick_params(axis='x', size=0)
    sns.despine(bottom=True, offset=4)
    return ax

# ------------------------------------------------------------------------------------
# Coordinate remapping
# ------------------------------------------------------------------------------------

def convert_range(oldval, newmin=None, newmax=None, oldmin=None, oldmax=None):
    oldrange = (oldmax - oldmin)
    newrange = (newmax - newmin)
    newval = (((oldval - oldmin) * newrange) / oldrange) + newmin
    return newval

def transform_rfmap(rfmap, rfparams):

    # Normalize rf map to range bw (0, 1)
    normed_rfmap = rfmap/rfmap.max()

    # Resize RF map to match image array
    rfsize = int(np.ceil(rfparams['pix_per_deg'] * rfparams['spacing']))
    #print("rf tile size:", rfsize)

    # Create RF map array
    screen_x, screen_y = rfparams['screen_resolution']
    rfscreen = np.ones((screen_y, screen_x))
    for rii, ri in enumerate(rfparams['row_vals_pix']):
        for cii, ci in enumerate(rfparams['col_vals_pix']):
            r_ix = int(round(ri-(rfsize/2.)))
            c_ix = int(round(ci-(rfsize/2.)))
            #print(r_ix, c_ix)
            rfscreen[r_ix:r_ix+rfsize, c_ix:c_ix+rfsize] = normed_rfmap[rii, cii]

    return rfscreen


def transform_stim_image(curr_img, rfparams, size_deg=30., verbose=False):
    
    screen_pix_x, screen_pix_y = rfparams['screen_resolution']
    stim_xpos_pix, stim_ypos_pix = rfparams['stim_pos_pix']
    pix_per_deg = rfparams['pix_per_deg']
    
    # Resize image (specify pixels based on selected size in degrees)
    imr_pix = resize_image_to_screen(curr_img, size_deg=size_deg, pix_per_deg=pix_per_deg) #, aspect_scale=1.747)

    # Pad resized image to match rf screen
    x_pad2 = round(screen_pix_x - (stim_xpos_pix + imr_pix.shape[1]/2.)) # Get far right edge
    x_pad1 = round(stim_xpos_pix - (imr_pix.shape[1]/2.)) # Get left edge
    y_pad1 = round(screen_pix_y - (stim_ypos_pix + imr_pix.shape[0]/2.)) # Get top edge
    y_pad2 = round(stim_ypos_pix - (imr_pix.shape[0]/2.)) # Get bottom edge

    imscreen = np.pad(imr_pix, (( int(abs(y_pad1)), int(abs(y_pad2)) ), \
                                ( int(abs(x_pad1)), int(abs(x_pad2)) )), mode='constant', constant_values=0)
    #print(size_deg, imscreen.shape)

    # Check if image is blown up beyond array size
    if x_pad2 < 0:     # need to trim right edge:
        imscreen = imscreen[:, 0:screen_pix_x]
        if verbose:
            print("...overblown on right edge", imscreen.shape)
    elif x_pad1 < 0:   # need to trim left edge
        trim_left = screen_pix_x - imscreen.shape[1]
        imscreen = imscreen[:, trim_left:]
        print("...overblown on left edge", imscreen.shape)

    if y_pad2 < 0:     # need to trim bottom edge:
        imscreen = imscreen[0:screen_pix_y, :]
        if verbose:
            print("...overblown on bottom edge", imscreen.shape)
    elif y_pad1 < 0:   # need to trim top edge
        trim_top = screen_pix_y - imscreen.shape[0]
        imscreen = imscreen[trim_top:, :]
        if verbose:
            print("...overblown on top edge", imscreen.shape)

    # Check if need extra padding:
    if imscreen.shape[0] < screen_pix_y:
        n_pad_extra = screen_pix_y - imscreen.shape[0]
        imscreen = np.pad(im_screen, ((0, n_pad_extra), (0, 0)), mode='constant', constant_value=0)
        if verbose:
            print("...padding %i to bottom" % n_pad_extra, imscreen.shape)
    elif imscreen.shape[0] > screen_pix_y:
        imscreen = imscreen[0:screen_pix_y, :]
        if verbose:
            print("...trimming %i off bottom" % (imscreen.shape[0]-screen_pix_y), imscreen.shape)

    if imscreen.shape[1] < screen_pix_x:
        n_pad_extra = screen_pix_x - imscreen.shape[1]
        imscreen = np.pad(im_screen, ((0, 0), (0, n_pad_extra)), mode='constant', constant_value=0)
        if verbose:
            print("...padding %i to right" % n_pad_extra, imscreen.shape)
    elif imscreen.shape[1] > screen_pix_x:
        imscreen = imscreen[:, 0:screen_pix_x]
        if verbose:
            print("...trimming %i off right" % (imscreen.shape[1]-screen_pix_x), imscreen.shape)

    return imscreen

def resize_image_to_screen(im, size_deg=30, pix_per_deg=16.06, aspect_scale=1.747):
    ref_dim = max(im.shape)
    #resize_factor = ((size_deg*pix_per_deg) / ref_dim ) / pix_per_deg
    #print(resize_factor)
    #scale_factor = resize_factor * aspect_scale
    scale_factor = (size_deg*aspect_scale)/(1./pix_per_deg) / ref_dim
    imr = cv2.resize(im, None, fx=scale_factor, fy=scale_factor)

    return imr


def resize_image_to_coords(im, size_deg=30, pix_per_deg=16.05, aspect_scale=1.747):
    print(pix_per_deg)
    ref_dim = max(im.shape)
    resize_factor = ((size_deg*pix_per_deg) / ref_dim ) / pix_per_deg
    scale_factor = resize_factor * aspect_scale
    
    imr = cv2.resize(im, None, fx=scale_factor, fy=scale_factor)
    
    return imr


# --------------------------------------------------------
# RF geometry functions
# --------------------------------------------------------
from matplotlib.patches import Ellipse, Rectangle, Polygon
from shapely.geometry.point import Point
from shapely.geometry import box
from shapely import affinity
from shapely.ops import cascaded_union

def create_ellipse(center, lengths, angle=0):
    """
    create a shapely ellipse. adapted from
    https://gis.stackexchange.com/a/243462
    """
    circ = Point(center).buffer(1)
    ell = affinity.scale(circ, int(lengths[0]), int(lengths[1]))
    ellr = affinity.rotate(ell, angle)
    return ellr

def rfs_to_polys(rffits, sigma_scale=2.35):
    '''
    rffits (pd dataframe)
        index : roi indices (same as gdf.rois)
        columns : r2, sigma_x, sigma_y, theta, x0, y0 (already converted) 
    
    returns list of polygons to do calculations with
    '''
    sz_param_names = [f for f in rffits.columns if '_' in f]
    sz_metrics = np.unique([f.split('_')[0] for f in sz_param_names])
    sz_metric = sz_metrics[0]
    assert sz_metric in ['fwhm', 'std'], "Unknown size metric: %s" % str(sz_metrics)

    sigma_scale = 1.0 if sz_metric=='fwhm' else sigma_scale
    roi_param = 'cell' if 'cell' in rffits.columns else 'rid'

    rf_columns=[roi_param, '%s_x' % sz_metric, '%s_y' % sz_metric, 'theta', 'x0', 'y0']
    #print(rf_columns, '[%s] Scale sigma: %.2f' % (sz_metric, sigma_scale))
    rffits = rffits[rf_columns]
    rf_polys=[(rid, 
        create_ellipse((x0, y0), (abs(sx)*sigma_scale, abs(sy)*sigma_scale), np.rad2deg(th))) \
        for rid, sx, sy, th, x0, y0 in rffits.values]
#    rf_polys = []
#    for roi in rffits['cell']: #.index.tolist():
#        _, sx, sy, th, x0, y0 = rffits.loc[roi]
#        s_ell = create_ellipse((x0, y0), (abs(sx)*sigma_scale, abs(sy)*sigma_scale), np.rad2deg(th))
#        rf_polys.append((roi, s_ell))
    return rf_polys

def stimsize_poly(sz, xpos=0, ypos=0):
    
    ry_min = ypos - sz/2.
    rx_min = xpos - sz/2.
    ry_max = ypos + sz/2.
    rx_max = xpos + sz/2.
    s_blobs = box(rx_min, ry_min, rx_max, ry_max)
    
    return s_blobs

def get_proportion_overlap(poly_tuple1, poly_tuple2):
    r1, poly1 = poly_tuple1
    r2, poly2 = poly_tuple2

    area_of_smaller = min([poly1.area, poly2.area])
    overlap_area = poly1.intersection(poly2).area
    perc_overlap = overlap_area/area_of_smaller
    #print(perc_overlap, overlap_area, area_of_smaller)
    odf = pd.DataFrame({'row':r1,
                        'col': r2,
                        'area_overlap': overlap_area,
                        'perc_overlap': perc_overlap}, index=[0])
    
    return odf


# ===================================================
# plotting
# ===================================================
def anisotropy_polarplot(rdf, metric='anisotropy', cmap='spring_r', alpha=0.5, marker='o', ax=None, dpi=150):

    vmin=0; vmax=1;
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    iso_cmap = cm.ScalarMappable(norm=norm, cmap=cmap)
    
    if ax is None:
        fig, ax = pl.subplots(1, subplot_kw=dict(projection='polar'), figsize=(4,3), dpi=dpi)

    thetas = rdf['theta_Mm_c'].values #% np.pi
    ratios = rdf[metric].values
    ax.scatter(thetas, ratios, s=30, c=ratios, cmap=cmap, alpha=alpha) # c=thetas, cmap='hsv', alpha=0.7)

    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    # ax.set_theta_direction(1)
    ax.set_xticklabels(['0$^\circ$', '', '90$^\circ$', '', '', '', '-90$^\circ$', ''])
    ax.set_rlabel_position(135) #315)
    ax.set_xlabel('')
    ax.set_yticklabels(['', 0.4, '', 0.8])
    ax.set_ylabel(metric, fontsize=12)

    # Grid lines and such
    ax.spines['polar'].set_visible(False)
    pl.subplots_adjust(left=0.1, right=0.9, wspace=0.2, bottom=0.3, top=0.8, hspace=0.5)

    # Colorbar
    iso_cmap._A = []
    cbar_ax = ax.figure.add_axes([0.4, 0.15, 0.2, 0.03])
    cbar = ax.figure.colorbar(iso_cmap, cax=cbar_ax, orientation='horizontal', ticks=[0, 1])
    if metric == 'anisotropy':
        xlabel_min = 'Iso\n(%.1f)' % (vmin) 
        xlabel_max= 'Aniso\n(%.1f)' % (vmax) 
    else:             
        xlabel_min = 'H\n(%.1f)' % (vmin) if hue_param in ['angle', 'aniso_index'] else '%.2f' % vmin
        xlabel_max= 'V\n(%.1f)' % (vmax) if hue_param in ['angle', 'aniso_index'] else '%.2f' % vmax
    cbar.ax.set_xticklabels([xlabel_min, xlabel_max])  # horizontal colorbar
    cbar.ax.tick_params(which='both', size=0)

    return ax

def draw_rf_on_screen(rdf, hue_param='aniso_index', shape_str='ellipse', ax=None, dpi=150,
                      ellipse_scale=0.3, ellipse_alpha=0.2, ellipse_lw=1, ellipse_facecolor='none', 
                      axis_lw=2, axis_alpha=0.9, n_plot_rfs=-1, n_plot_skip=1, 
                      centroid_size=5, centroid_alpha=0.3, vmin=-1, vmax=1):

    # Get screen info
    screen = get_screen_dims()
    screenleft, screenright = [-screen['azimuth_deg']*0.5, screen['azimuth_deg']*0.5]
    screenbottom, screentop = [-screen['altitude_deg']*0.5, screen['altitude_deg']*0.5]

    metric = 'anisotropy' if hue_param=='angle' else hue_param
    sat_param = 'aniso' if hue_param=='angle' else 'none'
    cmap = cm.cool if hue_param in ['angle', 'aniso_index'] else cm.spring_r
    #n_plot_rfs = -1
    #n_plot_skip = 1
    #axis_lw=2
    #axis_alpha=0.9
    #ellipse_lw=1
    #ellipse_alpha=0.2
    #ellipse_facecolor = 'none'
    borderpad=0

    centroid_size = centroid_size if shape_str=='centroid' else 2
    centroid_alpha = centroid_alpha if shape_str=='centroid' else 1.0
    ellipse_scale = ellipse_scale if shape_str=='ellipse' else 1.0

    vmin = vmin if metric=='aniso_index' else 0 
    vmax = vmax if metric=='aniso_index' else 1

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    scalar_cmap = cm.ScalarMappable(norm=norm, cmap=cmap)

    if ax is None:
        fig, ax = pl.subplots(1, figsize=(5,4), dpi=dpi)
        
    ax.set_xlim([screenleft-borderpad, screenright+borderpad])
    ax.set_ylim([screenbottom-borderpad, screentop+borderpad])
    
    rf_indices = rdf.index.tolist()[0::n_plot_skip] #n_plot_rfs]
    for i in rf_indices:
                      
        # Current ROI's RF fit params
        x0, y0, std_x, std_y, theta, theta_c, aniso, aniso_v = rdf[[
            'x0', 'y0', 'std_x', 'std_y', 'theta', 'theta_Mm_c', metric, 'anisotropy']].loc[i]
        
        # Set color based on hue_param and cmap
        if hue_param == 'angle':
            theta_col = rfutils.assign_saturation(theta_c, aniso, cmap=cmap, min_v=vmin, max_v=vmax)
        elif hue_param == 'aniso_index':
            theta_col = scalar_cmap.to_rgba(aniso)
        elif hue_param == 'aniso':
            theta_col = scalar_cmap.to_rgba(aniso)
                      
        # PLOT
        if 'centroid' in shape_str:
            ax.plot(x0, y0, marker='o', color=theta_col, alpha=centroid_alpha, markersize=centroid_size)

        if 'ellipse' in shape_str:
            el = Ellipse((x0, y0), width=std_x*ellipse_scale, height=std_y*ellipse_scale, 
                         angle=theta, edgecolor=theta_col, facecolor=ellipse_facecolor, 
                         alpha=ellipse_alpha, lw=ellipse_lw)
            ax.add_artist(el)

        if 'major' in shape_str:
            M = rdf[['std_x', 'std_y']].loc[i].max()  
            m = rdf[['std_x', 'std_y']].loc[i].min()  
            xe = (M/2.) * np.cos(np.deg2rad(theta)) if std_x>std_y else -(M/2.) * np.sin(np.deg2rad(theta))
            ye = (M/2.) * np.sin(np.deg2rad(theta)) if std_x>std_y else (M/2.) * np.cos(np.deg2rad(theta))
            ax.plot([x0, x0+xe], [y0, y0+ye], color=theta_col, alpha=axis_alpha, lw=axis_lw)

        if 'minor' in shape_str:
            xe2 = (m/2.) * np.sin(np.deg2rad(theta)) if std_x>std_y else -(m/2.) * np.cos(np.deg2rad(180-theta))
            ye2 = (m/2.) * np.cos(np.deg2rad(theta)) if std_x>std_y else (m/2.) * np.sin(np.deg2rad(180-theta))
            ax.plot([x0, x0+xe2], [y0, y0+ye2], color=theta_col, alpha=axis_alpha, lw=axis_lw)
    ax.set_aspect('equal')

    # COLOR BAR
    scalar_cmap._A = []
    cbar_ax = ax.figure.add_axes([0.43, 0.1, 0.15, 0.05])
    cbar = ax.figure.colorbar(scalar_cmap, cax=cbar_ax, orientation='horizontal', ticks=[vmin, vmax])
    if hue_param == 'anisotropy':
        xlabel_min = 'Iso\n(%.1f)' % (vmin) 
        xlabel_max= 'Aniso\n(%.1f)' % (vmax) 
    else:             
        xlabel_min = 'H\n(%.1f)' % (vmin) if hue_param in ['angle', 'aniso_index'] else '%.2f' % vmin
        xlabel_max= 'V\n(%.1f)' % (vmax) if hue_param in ['angle', 'aniso_index'] else '%.2f' % vmax
                 
    cbar.ax.set_xticklabels([xlabel_min, xlabel_max])  # horizontal colorbar

    return ax
