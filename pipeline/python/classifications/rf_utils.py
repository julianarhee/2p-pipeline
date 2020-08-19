import os
import glob
import cv2

import numpy as np
import pandas as pd
import pylab as pl
import cPickle as pkl
import traceback

# ------------------------------------------------------------------------------------
# General stats
# ------------------------------------------------------------------------------------

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


# ------------------------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------------------------
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

def get_fit_dpaths(dsets, traceid='traces001', fit_desc=None, excluded_sessions=[],
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
                                        'combined_%s_*' % curr_rfname, 'traces', '%s*' % traceid, 
                                        'receptive_fields', fit_desc, 'fit_results.pkl'))
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
            
    rfdf = pd.concat(df_list, axis=0) #.reset_index(drop=True)

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
