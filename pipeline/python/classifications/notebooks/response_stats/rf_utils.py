import os
import itertools
import glob
import traceback
import json

import pylab as pl
import matplotlib as mpl
import dill as pkl
import scipy.stats as spstats
import numpy as np
import pandas as pd
import py3utils as p3
from py3utils import convert_range

from matplotlib.patches import Ellipse, Rectangle, Polygon
from shapely.geometry.point import Point
from shapely.geometry import box
from shapely import affinity


# plotting
def anisotropy_polarplot(rdf, metric='anisotropy', cmap='spring_r', alpha=0.5, 
                            marker='o', markersize=30, ax=None, 
                            hue_param='aniso_index', cbar_bbox=[0.4, 0.15, 0.2, 0.03]):

    vmin=0; vmax=1;
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    iso_cmap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    
    if ax is None:
        fig, ax = pl.subplots(1, subplot_kw=dict(projection='polar'), figsize=(4,3))

    thetas = rdf['theta_Mm_c'].values #% np.pi # all thetas should point the same way
    ratios = rdf[metric].values
    ax.scatter(thetas, ratios, s=markersize, c=ratios, cmap=cmap, alpha=alpha)

    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
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
    cbar_ax = ax.figure.add_axes(cbar_bbox)
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



# Ellipse fitting and formatting
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
    rffits = rffits[rf_columns]
    rf_polys=dict((rid, 
        create_ellipse((x0, y0), (abs(sx)*sigma_scale, abs(sy)*sigma_scale), np.rad2deg(th))) \
        for rid, sx, sy, th, x0, y0 in rffits.values)

    return rf_polys

def stimsize_poly(sz, xpos=0, ypos=0):
    from shapely.geometry import box
 
    ry_min = ypos - sz/2.
    rx_min = xpos - sz/2.
    ry_max = ypos + sz/2.
    rx_max = xpos + sz/2.
    s_blobs = box(rx_min, ry_min, rx_max, ry_max)
    
    return s_blobs

def calculate_overlap(poly1, poly2, r1='poly1', r2='poly2'):
    #r1, poly1 = poly_tuple1
    #r2, poly2 = poly_tuple2

    #area_of_smaller = min([poly1.area, poly2.area])
    intersection_area = poly1.intersection(poly2).area
    union_area = poly1.union(poly2).area
    area_overlap = intersection_area/union_area 

    area_of_smaller = min([poly1.area, poly2.area])
    overlap_area = poly1.intersection(poly2).area
    perc_overlap = overlap_area/area_of_smaller


    odf = pd.DataFrame({'poly1':r1,
                        'poly2': r2,
                        'area_overlap': area_overlap, #overlap_area,
                        'perc_overlap': perc_overlap}, index=[0])
    
    return odf


def get_proportion_overlap(poly_tuple1, poly_tuple2):
    r1, poly1 = poly_tuple1
    r2, poly2 = poly_tuple2

    area_of_smaller = min([poly1.area, poly2.area])
    overlap_area = poly1.intersection(poly2).area
    perc_overlap = overlap_area/area_of_smaller

    odf = pd.DataFrame({'row':r1,
                        'col': r2,
                        'area_overlap': overlap_area,
                        'perc_overlap': perc_overlap}, index=[0])
    
    return odf


def get_rf_overlaps(rf_polys):
    '''
    tuning_ (pd.DataFrame): nconds x nrois.
    Each entry is the mean response (across trials) for a given stim condition.
    '''
    # Calculate signal corrs
    o_=[]
    rois_ = sorted(rf_polys.keys())
    # Get unique pairs, then iterate thru and calculate pearson's CC
    for col_a, col_b in itertools.combinations(rois_, 2):
        df_ = calculate_overlap(rf_polys[col_a], rf_polys[col_b], \
                                  r1=col_a, r2=col_b)
        o_.append(df_)
    overlapdf = pd.concat(o_)
                   
    return overlapdf


# Data processing
def rfits_to_df(fitr, row_vals=[], col_vals=[], roi_list=None, fit_params={},
                scale_sigma=True, sigma_scale=2.35, convert_coords=True, spherical=False):
    '''
    Takes each roi's RF fit results, converts to screen units, and return as dataframe.
    Scale to make size FWFM if scale_sigma is True.
    '''
    if roi_list is None:
        roi_list = sorted(fitr.keys())
       
    sigma_scale = sigma_scale if scale_sigma else 1.0

    fitdf = pd.DataFrame({'x0': [fitr[r]['x0'] for r in roi_list],
                          'y0': [fitr[r]['y0'] for r in roi_list],
                          'sigma_x': [fitr[r]['sigma_x'] for r in roi_list],
                          'sigma_y': [fitr[r]['sigma_y'] for r in roi_list],
                          'theta': [fitr[r]['theta'] % (2*np.pi) for r in roi_list],
                          'r2': [fitr[r]['r2'] for r in roi_list]},
                              index=roi_list)

    if convert_coords:
        if spherical:
            fitdf = convert_fit_to_coords_spherical(fitdf, fit_params, spherical=spherical, scale_sigma=False)
            fitdf['sigma_x'] = fitdf['sigma_x']*sigma_scale
            fitdf['sigma_y'] = fitdf['sigma_y']*sigma_scale
        else:
            x0, y0, sigma_x, sigma_y = convert_fit_to_coords(fitdf, row_vals, col_vals)
            fitdf['x0'] = x0
            fitdf['y0'] = y0
            fitdf['sigma_x'] = sigma_x * sigma_scale
            fitdf['sigma_y'] = sigma_y * sigma_scale

    return fitdf

def apply_scaling_to_df(row, grid_points=None, new_values=None):
    #r2 = row['r2']
    #theta = row['theta']
    #offset = row['offset']
    x0, y0, sx, sy = get_scaled_sigmas(grid_points, new_values,
                                             row['x0'], row['y0'], 
                                             row['sigma_x'], row['sigma_y'], row['theta'],
                                             convert=True)
    return x0, y0, sx, sy #sx, sy, x0, y0


def get_spherical_coords(cart_pointsX=None, cart_pointsY=None, cm_to_degrees=True,
                    resolution=(1080, 1920),
                   xlim_degrees=(-59.7, 59.7), ylim_degrees=(-33.6, 33.6)):

    # Monitor size and position variables
    width_cm = 103; #%56.69;  % 103 width of screen, in cm
    height_cm = 58; #%34.29;  % 58 height of screen, in cm
    pxXmax = resolution[1] #1920; #%200; % number of pixels in an image that fills the whole screen, x
    pxYmax = resolution[0] #1080; #%150; % number of pixels in an image that fills the whole screen, y

    # Eye info
    cx = width_cm/2. # % eye x location, in cm
    cy = height_cm/2. # %11.42; % eye y location, in cm
    eye_dist = 30.; #% in cm

    # Distance to bottom of screen, along the horizontal eye line
    zdistBottom = np.sqrt((cy**2) + (eye_dist**2)) #; %24.49;     % in cm
    zdistTop    = np.sqrt((cy**2) + (eye_dist**2)) #; %14.18;     % in cm

    # Internal conversions
    top = height_cm-cy;
    bottom = -cy;
    right = cx;
    left = cx - width_cm;

    if cart_pointsX is None or cart_pointsY is None:
        [xi, yi] = np.meshgrid(np.arange(0, pxXmax), np.arange(0, pxYmax))
        print(xi.shape, yi.shape)

        cart_pointsX = left + (float(width_cm)/pxXmax)*xi;
        cart_pointsY = top - (float(height_cm)/pxYmax)*yi;
        cart_pointsZ = zdistTop + ((zdistBottom-zdistTop)/float(pxYmax))*yi
    else:
        cart_pointsZ = zdistTop + ((zdistBottom-zdistTop)/float(pxYmax))*cart_pointsY

    if cm_to_degrees:
        xmin_cm=cart_pointsX.min(); xmax_cm=cart_pointsX.max();
        ymin_cm=cart_pointsY.min(); ymax_cm=cart_pointsY.max();
        xmin_deg, xmax_deg = xlim_degrees
        ymin_deg, ymax_deg = ylim_degrees
        cart_pointsX = convert_range(cart_pointsX, oldmin=xmin_cm, oldmax=xmax_cm, 
                                       newmin=xmin_deg, newmax=xmax_deg)
        cart_pointsY = convert_range(cart_pointsY, oldmin=ymin_cm, oldmax=ymax_cm, 
                                       newmin=ymin_deg, newmax=ymax_deg)
        cart_pointsZ = convert_range(cart_pointsZ, oldmin=ymin_cm, oldmax=ymax_cm, 
                                       newmin=ymin_deg, newmax=ymax_deg)

    sphr_pointsTh, sphr_pointsPh, sphr_pointsR = cart2sph(cart_pointsZ, cart_pointsX, cart_pointsY)

    return cart_pointsX, cart_pointsY, sphr_pointsTh, sphr_pointsPh


def warp_spherical(image_values, cart_pointsX, cart_pointsY, sphr_pointsTh, sphr_pointsPh, 
                    normalize_range=True, in_radians=True, method='linear'):
    from scipy.interpolate import griddata

    xmaxRad = sphr_pointsTh.max()
    ymaxRad = sphr_pointsPh.max()

    # normalize max of Cartesian to max of Spherical
    fx = xmaxRad/cart_pointsX.max() if normalize_range else 1.
    fy = ymaxRad/cart_pointsY.max() if normalize_range else 1.
    x0 = cart_pointsX.copy()*fx
    y0 = cart_pointsY.copy()*fy
   
    if in_radians and not normalize_range:
        points = np.array( (np.rad2deg(sphr_pointsTh).flatten(), np.rad2deg(sphr_pointsPh).flatten()) ).T
    else:
        points = np.array( (sphr_pointsTh.flatten(), sphr_pointsPh.flatten()) ).T

    values_ = image_values.flatten()
    #values_y = cart_pointsY.flatten()

    warped_values = griddata( points, values_, (x0,y0) , method=method)
    
    return warped_values



def get_screen_lim_pixels(lin_coord_x, lin_coord_y, row_vals=None, col_vals=None):
    
    #pix_per_deg=16.050716 pix_per_deg = screeninfo['pix_per_deg']
    stim_size = float(np.unique(np.diff(row_vals)))

    right_lim = max(col_vals) + (stim_size/2.)
    left_lim = min(col_vals) - (stim_size/2.)
    top_lim = max(row_vals) + (stim_size/2.)
    bottom_lim = min(row_vals) - (stim_size/2.)

    # Get actual stimulated locs in pixels
    i_x, i_y = np.where( np.abs(lin_coord_x-right_lim) == np.abs(lin_coord_x-right_lim).min() )
    pix_right_edge = int(np.unique(i_y))

    i_x, i_y = np.where( np.abs(lin_coord_x-left_lim) == np.abs(lin_coord_x-left_lim).min() )
    pix_left_edge = int(np.unique(i_y))
    #print("AZ bounds (pixels): ", pix_right_edge, pix_left_edge)

    i_x, i_y = np.where( np.abs(lin_coord_y-top_lim) == np.abs(lin_coord_y-top_lim).min() )
    pix_top_edge = int(np.unique(i_x))

    i_x, i_y = np.where( np.abs(lin_coord_y-bottom_lim) == np.abs(lin_coord_y-bottom_lim).min() )
    pix_bottom_edge = int(np.unique(i_x))
    #print("EL bounds (pixels): ", pix_top_edge, pix_bottom_edge)

    # Check expected tile size
    #ncols = len(col_vals); nrows = len(row_vals);
    #expected_sz_x = (pix_right_edge-pix_left_edge+1) * (1./pix_per_deg) / ncols
    #expected_sz_y = (pix_bottom_edge-pix_top_edge+1) * (1./pix_per_deg) / nrows
    #print("tile sz-x, -y should be ~(%.2f, %.2f) deg" % (expected_sz_x, expected_sz_y))
    
    return (pix_bottom_edge, pix_left_edge, pix_top_edge,  pix_right_edge)


def coordinates_for_transformation(fit_params):
    ds_factor = fit_params['downsample_factor']
    col_vals = fit_params['col_vals']
    row_vals = fit_params['row_vals']
    nx = len(col_vals)
    ny = len(row_vals)

    # Downsample screen resolution
    resolution_ds = [int(i/ds_factor) for i in fit_params['screen']['resolution'][::-1]]

    # Get linear coordinates in degrees (downsampled)
    lin_x, lin_y = get_lin_coords(resolution=resolution_ds, cm_to_deg=True) 
    print("Screen res (ds=%ix): [%i, %i]" % (ds_factor, resolution_ds[0], resolution_ds[1]))

    # Get Spherical coordinate mapping
    cart_x, cart_y, sphr_x, sphr_y = get_spherical_coords(cart_pointsX=lin_x, 
                                                            cart_pointsY=lin_y,
                                                            cm_to_degrees=False) # already in deg

    screen_bounds_pix = get_screen_lim_pixels(lin_x, lin_y, 
                                            row_vals=row_vals, col_vals=col_vals)
    (pix_bottom_edge, pix_left_edge, pix_top_edge, pix_right_edge) = screen_bounds_pix
 
    # Trim and downsample coordinate space to match corrected map
    cart_x_ds  = cv2.resize(cart_x[pix_top_edge:pix_bottom_edge, pix_left_edge:pix_right_edge], 
                            (nx, ny))
    cart_y_ds  = cv2.resize(cart_y[pix_top_edge:pix_bottom_edge, pix_left_edge:pix_right_edge], 
                            (nx, ny))

    sphr_x_ds  = cv2.resize(sphr_x[pix_top_edge:pix_bottom_edge, pix_left_edge:pix_right_edge], 
                            (nx,ny))
    sphr_y_ds  = cv2.resize(sphr_y[pix_top_edge:pix_bottom_edge, pix_left_edge:pix_right_edge], 
                            (nx, ny))

    grid_x, grid_y = np.meshgrid(range(nx),range(ny)[::-1])
    grid_points = np.array( (grid_x.flatten(), grid_y.flatten()) ).T
    cart_values = np.array( (cart_x_ds.flatten(), cart_y_ds.flatten()) ).T
    sphr_values = np.array( (np.rad2deg(sphr_x_ds).flatten(), np.rad2deg(sphr_y_ds).flatten()) ).T

    return grid_points, cart_values, sphr_values


def convert_fit_to_coords_spherical(fitdf, fit_params, scale_sigma=True, sigma_scale=2.35, spherical=True):
    sigma_scale = sigma_scale if scale_sigma else 1.0

    grid_points, cart_values, sphr_values = coordinates_for_transformation(fit_params)
    
    if spherical:
        converted = fitdf.apply(apply_scaling_to_df, args=(grid_points, sphr_values), axis=1)
    else:
        converted = fitdf.apply(apply_scaling_to_df, args=(grid_points, cart_values), axis=1)
    newdf = pd.DataFrame([[x0, y0, sx*sigma_scale, sy*sigma_scale] \
                        for x0, y0, sx, sy in converted.values], index=converted.index, 
                        columns=['x0', 'y0', 'sigma_x', 'sigma_y'])
    fitdf[['sigma_x', 'sigma_y', 'x0', 'y0']] = newdf[['sigma_x', 'sigma_y', 'x0', 'y0']]

    return fitdf


def convert_fit_to_coords(fitdf, row_vals, col_vals, rid=None):
    
    if rid is not None:
        xx = convert_range(fitdf['x0'][rid], 
                            newmin=min(col_vals), newmax=max(col_vals), 
                            oldmax=len(col_vals)-1, oldmin=0) 
        sigma_x = convert_range(abs(fitdf['sigma_x'][rid]), 
                            newmin=0, newmax=max(col_vals)-min(col_vals), 
                            oldmax=len(col_vals)-1, oldmin=0) 
        yy = convert_range(fitdf['y0'][rid], 
                            newmin=min(row_vals), newmax=max(row_vals), 
                            oldmax=len(row_vals)-1, oldmin=0) 
        sigma_y = convert_range(abs(fitdf['sigma_y'][rid]), 
                            newmin=0, newmax=max(row_vals)-min(row_vals), 
                            oldmax=len(row_vals)-1, oldmin=0)
    else:
        xx = convert_range(fitdf['x0'], 
                            newmin=min(col_vals), newmax=max(col_vals), 
                            oldmax=len(col_vals)-1, oldmin=0) 
        sigma_x = convert_range(abs(fitdf['sigma_x']), 
                            newmin=0, newmax=max(col_vals)-min(col_vals), 
                            oldmax=len(col_vals)-1, oldmin=0) 
        yy = convert_range(fitdf['y0'], 
                            newmin=min(row_vals), newmax=max(row_vals), 
                            oldmax=len(row_vals)-1, oldmin=0) 
        sigma_y = convert_range(abs(fitdf['sigma_y']), 
                            newmin=0, newmax=max(row_vals)-min(row_vals), 
                            oldmax=len(row_vals)-1, oldmin=0)
    
    return xx, yy, sigma_x, sigma_y



# ###############################################3
# Data loading
# ###############################################3
def get_fit_desc(response_type='dff', do_spherical_correction=False):
    if do_spherical_correction:
        fit_desc = 'fit-2dgaus_%s_sphr' % response_type
    else:
        fit_desc = 'fit-2dgaus_%s-no-cutoff' % response_type        

    return fit_desc

def create_rf_dir(animalid, session, fov, run_name, traceid='traces001', response_type='dff', 
                do_spherical_correction=False, fit_thr=0.5, rootdir='/n/coxfs01/2p-data'):
    # Get RF dir for current fit type
    fit_desc = get_fit_desc(response_type=response_type, do_spherical_correction=do_spherical_correction)
    fov_dir = os.path.join(rootdir, animalid, session, fov)

    if 'combined' in run_name:
        traceid_dirs = [t for t in glob.glob(os.path.join(fov_dir, run_name, 'traces', '%s*' % traceid))]
    else: 
        traceid_dirs = [t for t in glob.glob(os.path.join(fov_dir, 'combined_%s_*' % run_name, 'traces', '%s*' % traceid))]
    if len(traceid_dirs) > 1:
        print("[creating RF dir, %s] More than 1 trace ID found:" % run_name)
        for ti, traceid_dir in enumerate(traceid_dirs):
            print(ti, traceid_dir)
        sel = input("Select IDX of traceid to use: ")
        traceid_dir = traceid_dirs[int(sel)]
    else:
        traceid_dir = traceid_dirs[0]
    #traceid = os.path.split(traceid_dir)[-1]
         
    rfdir = os.path.join(traceid_dir, 'receptive_fields', fit_desc)
    if not os.path.exists(rfdir):
        os.makedirs(rfdir)

    return rfdir, fit_desc

def load_fit_results(animalid, session, fov, experiment='rfs', 
                traceid='traces001', response_type='dff', 
                fit_desc=None, do_spherical_correction=False, 
                rootdir='/n/coxfs01/2p-data'): 
    fit_results = None
    fit_params = None
    try: 
        if fit_desc is None:
            assert response_type is not None, "No response_type or fit_desc provided"
            fit_desc = get_fit_desc(response_type=response_type, 
                            do_spherical_correction=do_spherical_correction)  
        rfname = 'gratings' if int(session) < 20190511 else experiment
        rfname = rfname.split('_')[1] if 'combined' in rfname else rfname
        rfdir = glob.glob(os.path.join(rootdir, animalid, session, fov, 
                        '*%s_*' % rfname, #experiment
                        'traces', '%s*' % traceid, 'receptive_fields', 
                        '%s' % fit_desc))[0]
    except AssertionError as e:
        traceback.print_exc()
       
    # Load results
    rf_results_fpath = os.path.join(rfdir, 'fit_results.pkl')
    with open(rf_results_fpath, 'rb') as f:
        fit_results = pkl.load(f, encoding='latin1')
   
    # Load params 
    rf_params_fpath = os.path.join(rfdir, 'fit_params.json')
    with open(rf_params_fpath, 'r') as f:
        fit_params = json.load(f)
        
    return fit_results, fit_params
 

def load_eval_results(animalid, session, fov, experiment='rfs',
                        traceid='traces001', response_type='dff', 
                        fit_desc=None, do_spherical_correction=False,
                        rootdir='/n/coxfs01/2p-data'):

    eval_results=None; eval_params=None;            
    if fit_desc is None:
        fit_desc = get_fit_desc(response_type=response_type,
                                        do_spherical_correction=do_spherical_correction)
    rfname = experiment.split('_')[1] if 'combined' in experiment else experiment
    try: 
        #print("Checking to load: %s" % fit_desc)
        rfdir = glob.glob(os.path.join(rootdir, animalid, session, fov, '*%s_*' % rfname, 
                        'traces', '%s*' % traceid, 'receptive_fields', '%s*' % fit_desc))[0]
        evaldir = os.path.join(rfdir, 'evaluation')
        assert os.path.exists(evaldir), "No evaluation exists\n(%s)\n. Aborting" % evaldir
    except IndexError as e:
        traceback.print_exc()
        return None, None
    except AssertionError as e:
        traceback.print_exc()
        return None, None

    # Load results
    rf_eval_fpath = os.path.join(evaldir, 'evaluation_results.pkl')
    assert os.path.exists(rf_eval_fpath), "No eval result: %s" % rf_eval_fpath
    with open(rf_eval_fpath, 'rb') as f:
        eval_results = pkl.load(f, encoding='latin1')
   
    #  Load params 
    eval_params_fpath = os.path.join(evaldir, 'evaluation_params.json')
    with open(eval_params_fpath, 'r') as f:
        eval_params = json.load(f)
        
    return eval_results, eval_params

def get_reliable_fits(pass_cis, pass_criterion='all', single=False):
    if single is True:
        keep_rids = [i for i in pass_cis.index.tolist() if pass_cis[pass_criterion][i]==True]
    else:       
        param_cols = [p for p in pass_cis.columns if p!='cell']
        if pass_criterion=='all':
            tmp_ci = pass_cis[param_cols].copy()
            keep_rids = [i for i in pass_cis.index.tolist() if all(tmp_ci.loc[i])]
        elif pass_criterion=='any':
            tmp_ci = pass_cis[param_cols].copy()
            keep_rids = [i for i in pass_cis.index.tolist() if any(tmp_ci.loc[i])]
        elif pass_criterion=='size':
            keep_rids = [i for i in pass_cis.index.tolist() 
                            if (pass_cis['sigma_x'][i]==True and pass_cis['sigma_y'][i]==True)]
        elif pass_criterion=='position':
            keep_rids = [i for i in pass_cis.index.tolist() 
                            if (pass_cis['x0'][i]==True and pass_cis['y0'][i]==True)]
        elif pass_criterion=='most':
            tmp_ci = pass_cis[param_cols].copy()
            keep_rids = [i for i in pass_cis.index.tolist() 
                            if sum([pv==True 
                            for pv in tmp_ci.loc[rid]])/float(len(param_cols))>0.5]
        else:   
            keep_rids = [i for i in pass_cis.index.tolist() if any(pass_cis.loc[i])]
       
    pass_df = pass_cis.loc[keep_rids]
 
    reliable_rois = sorted(pass_df.index.tolist())

    return reliable_rois

 
def cycle_and_load(rfmeta, assigned_cells, fit_desc=None, traceid='traces001', fit_thr=0.5, 
                      scale_sigma=True, sigma_scale=2.35, verbose=False, 
                      response_type='None', reliable_only=True,
                      rootdir='/n/coxfs01/2p-data'):
    '''
    Combines fit_results.pkl(fit from data) and evaluation_results.pkl (evaluated fits via bootstrap)
    and gets fit results only for those cells that are good/robust fits based on bootstrap analysis.
    '''

    df_list = []
    for (visual_area, datakey, experiment), g in \
                            rfmeta.groupby(['visual_area', 'datakey', 'experiment']):
        if experiment not in ['rfs', 'rfs10']:
            continue
        session, animalid, fovnum = p3.split_datakey_str(datakey)
        fov = 'FOV%i_zoom2p0x' % fovnum
        curr_cells = assigned_cells[(assigned_cells.visual_area==visual_area)
                                   & (assigned_cells.datakey==datakey)]['cell'].unique() #g['cell'].unique() 
        try:
            curr_rfname = experiment if int(session)>=20190511 else 'gratings'
            #### Load eval results 
            eval_results, eval_params = load_eval_results(animalid, session, fov,
                                                experiment=curr_rfname, traceid=traceid, 
                                                fit_desc=fit_desc)   
            if eval_results is None:
                print('-- no good (%s (%s, %s)), skipping' % (datakey, visual_area, experiment))
                continue
            
            #### Load fit results from measured
            fit_results, fit_params = load_fit_results(
                                                animalid, session, fov,
                                                experiment=curr_rfname,
                                                traceid=traceid, 
                                                fit_desc=fit_desc)
            #fit_rois = sorted(fit_results['fit_results'].keys())
            fit_rois = sorted(eval_results['data']['cell'].unique())

            scale_sigma = fit_params['scale_sigma']
            sigma_scale = fit_params['sigma_scale']
            rfit_df = rfits_to_df(fit_results, scale_sigma=scale_sigma, sigma_scale=sigma_scale,
                            row_vals=fit_params['row_vals'], col_vals=fit_params['col_vals'], 
                            roi_list=fit_rois)

            #### Identify cells with measured params within 95% CI of bootstrap distN
            pass_rois = rfit_df[rfit_df['r2']>fit_thr].index.tolist()
            param_list = [param for param in rfit_df.columns if param != 'r2']
            # Note: get_good_fits(rfit_df, eval_results, param_list=param_list) returns same
            # as reliable_rois, since checks if all params within 95% CI
            reliable_rois = get_reliable_fits(eval_results['pass_cis'],
                                                     pass_criterion='all')
            if verbose:
                print("[%s] %s: %i of %i fit rois pass for all params" \
                            % (visual_area, datakey, len(pass_rois), len(fit_rois)))
                print("...... : %i of %i fit rois passed as reliiable" \
                            % (len(reliable_rois), len(pass_rois)))

            #### Create dataframe with params only for good fit cells
            keep_rois = reliable_rois if reliable_only else pass_rois
            if curr_cells is not None:
                keep_rois = [r for r in keep_rois if r in curr_cells]

            passdf = rfit_df.loc[keep_rois].copy()
            # "un-scale" size, if flagged
            if not scale_sigma:
                sigma_x = passdf['sigma_x']/sigma_scale
                sigma_y = passdf['sigma_y'] / sigma_scale
                passdf['sigma_x'] = sigma_x
                passdf['sigma_y'] = sigma_y

            passdf['cell'] = keep_rois
            passdf['datakey'] = datakey
            passdf['animalid'] = animalid
            passdf['session'] = session
            passdf['fovnum'] = fovnum
            passdf['visual_area'] = visual_area
            passdf['experiment'] = experiment
            df_list.append(passdf)

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
    rfdf.loc[swap_ixs, 'theta_Mm_deg'] = [ (theta + 90) % 360 if (90 <= theta < 360) \
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
    sins_c = p3.convert_range(sins, oldmin=0, oldmax=1, newmin=-1, newmax=1)
    rfdf['aniso_index'] = sins_c * rfdf['anisotropy']
 
    return rfdf


def get_fit_dpaths(dsets, traceid='traces001', fit_desc=None,
                    excluded_sessions = ['JC110_20191004_fov1',
                                         'JC080_20190602_fov1',
                                         'JC113_20191108_fov1', 
                                         'JC113_20191108_fov2'],
                    rootdir='/n/coxfs01/2p-data'):
    '''
    rfdata: (dataframe)
        Metadata (subset of 'sdata') of all datasets to include in current analysis
        
    Gets paths to fit_results.pkl, which contains all (fit-able) results for each cell.
    Adds new column of paths to rfdata.
    '''
    assert fit_desc is not None, "No fit-desc specified!"
    
    rfmeta = dsets.copy()
    fit_these = []
    dpaths = {}
    unknown = []
    for (va, datakey), g in dsets.groupby(['visual_area','datakey']):
        session, animalid, fovnum = p3.split_datakey_str(datakey)
        fov='FOV%i_zoom2p0x' % fovnum
        if datakey in excluded_sessions:
            rfmeta = rfmeta.drop(g.index)
            continue
        rfruns = g['experiment'].unique()
        for rfname in rfruns:
            curr_rfname = 'gratings' if int(session) < 20190511 else rfname
            fpath = glob.glob(os.path.join(rootdir, animalid, session, '*%s' % fov, 
                                        'combined_%s_*' % curr_rfname, 'traces', '%s*' % traceid, 
                                        'receptive_fields', fit_desc, 'fit_results.pkl'))
            if len(fpath) > 0:
                assert len(fpath)==1, "Too many paths: %s" % str(fpath)
                dpaths[(va, datakey, rfname)] = fpath[0] #['-'.join([animalid, session, fov, rfname])] = fpath[0]
            elif len(fpath) == 0:
                fit_these.append((animalid, session, fov, rfname))
            else:
                print("[%s] %s - warning: unknown file paths" % (datakey, rfname))
    print("N dpaths: %i, N unfit: %i" % (len(dpaths), len(fit_these)))
    print("N datasets included: %i, N sessions excluded: %i" % (rfmeta.shape[0], len(excluded_sessions)))
   
    rmeta= pd.concat([g for (va, dk, rfname), g in rfmeta.groupby(['visual_area', 'datakey', 'experiment'])\
                if (va, dk, rfname) in dpaths.keys()])
    rmeta['path'] = None
    for (va, dk, rfname), g in rmeta.groupby(['visual_area', 'datakey', 'experiment']):
        curr_fpath = dpaths[(va, dk, rfname)]
        rmeta.loc[g.index, 'path'] = curr_fpath
        
    rmeta = rmeta.drop_duplicates().reset_index(drop=True)
    
    return rmeta, fit_these


def aggregate_rfdata(rf_dsets, assigned_cells, traceid='traces001', 
                        fit_desc='fit-2dgaus_dff-no-cutoff', reliable_only=True, verbose=False):
    # Gets all results for provided datakeys (sdata, for rfs/rfs10)
    # Aggregates results for the datakeys
    # assigned_cells:  cells assigned by visual area

    # Only try to load rfdata if we can find fit + evaluation results
    rfmeta, no_fits = get_fit_dpaths(rf_dsets, traceid=traceid, fit_desc=fit_desc)
    rfdf = cycle_and_load(rfmeta, assigned_cells, reliable_only=reliable_only,
                                        fit_desc=fit_desc, traceid=traceid, verbose=verbose)
    rfdf = rfdf.reset_index(drop=True)

    return rfdf

def add_rf_positions(rfdf, calculate_position=False, traceid='traces001'):
    '''
    Add ROI position info to RF dataframe (converted and pixel-based).
    Set calculate_position=True, to re-calculate.
    '''
    import roi_utils as rutils
    print("Adding RF position info...")
    pos_params = ['fov_xpos', 'fov_xpos_pix', 'fov_ypos', 'fov_ypos_pix', 'ml_pos','ap_pos']
    for p in pos_params:
        rfdf[p] = ''
    p_list=[]
    #for (animalid, session, fovnum, exp), g in rfdf.groupby(['animalid', 'session', 'fovnum', 'experiment']):
    for (va, dk, exp), g in rfdf.groupby(['visual_area', 'datakey', 'experiment']):
        session, animalid, fovnum = split_datakey_str(dk)

        fcoords = rutils.load_roi_coords(animalid, session, 'FOV%i_zoom2p0x' % fovnum,
                                  traceid=traceid, create_new=False)

        #for ei, e_df in g.groupby(['experiment']):
        cell_ids = g['cell'].unique()
        p_ = fcoords['roi_positions'].loc[cell_ids]
        for p in pos_params:
            rfdf.loc[g.index, p] = p_[p].values

    return rfdf


def average_rfs_select(rfdf):
    final_rfdf=None
    rf_=[]
    for (visual_area, datakey), curr_rfdf in rfdf.groupby(['visual_area', 'datakey']):
        final_rf=None
        if visual_area=='V1' and 'rfs' in curr_rfdf['experiment'].values:
            final_rf = curr_rfdf[curr_rfdf.experiment=='rfs'].copy()
        elif visual_area in ['Lm', 'Li']:
            # Which cells have receptive fields
            rois_ = curr_rfdf['cell'].unique()

            # Means by cell id (some dsets have rf-5 and rf10 measurements, average these)
            meanrf = curr_rfdf.groupby(['cell']).mean().reset_index()
            mean_thetas = curr_rfdf.groupby(['cell'])['theta'].apply(spstats.circmean, low=0, high=2*np.pi).values
            meanrf['theta'] = mean_thetas
            meanrf['visual_area'] = visual_area
            meanrf['experiment'] = ['average_rfs' if len(g['experiment'].values)>1 \
                                    else str(g['experiment'].unique()[0]) for c, g in curr_rfdf.groupby(['cell'])]
            #meanrf['experiment'] = ['average_rfs' for _ in np.arange(0, len(assigned_with_rfs))]

            # Add the meta/non-numeric info
            non_num = [c for c in curr_rfdf.columns if c not in meanrf.columns and c!='experiment']
            metainfo = pd.concat([g[non_num].iloc[0] for c, g in \
                                curr_rfdf.groupby(['cell'])], axis=1).T.reset_index(drop=True)
            final_rf = pd.concat([metainfo, meanrf], axis=1)            
            final_rf = update_rf_metrics(final_rf, scale_sigma=True)
        rf_.append(final_rf)

    final_rfdf = pd.concat(rf_).reset_index(drop=True)

    return final_rfdf


def average_rfs(rfdf):
    final_rfdf=None
    rf_=[]
    for (visual_area, datakey), curr_rfdf in rfdf.groupby(['visual_area', 'datakey']):
        # Means by cell id (some dsets have rf-5 and rf10 measurements, average these)
        meanrf = curr_rfdf.groupby(['cell']).mean().reset_index()
        mean_thetas = curr_rfdf.groupby(['cell'])['theta'].apply(spstats.circmean, low=0, high=2*np.pi).values
        meanrf['theta'] = mean_thetas
        meanrf['visual_area'] = [visual_area for _ in  np.arange(0, len(assigned_with_rfs))] # reassign area
        meanrf['experient'] = ['average_rfs' if len(g['experiment'].values)>1 \
                                else str(g['experiment'].unique()) for c, g in curr_rfdf.groupby(['cell'])]
        #meanrf['experiment'] = ['average_rfs' for _ in np.arange(0, len(assigned_with_rfs))]

        # Add the meta/non-numeric info
        non_num = [c for c in curr_rfdf.columns if c not in meanrf.columns and c!='experiment']
        metainfo = pd.concat([g[non_num].iloc[0] for c, g in \
                            curr_rfdf.groupby(['cell'])], axis=1).T.reset_index(drop=True)
        final_rf = pd.concat([metainfo, meanrf], axis=1)
        final_rf = update_rf_metrics(final_rf, scale_sigma=True)
        rf_.append(final_rf)

    final_rfdf = pd.concat(r_).reset_index(drop=True)

    return final_rfdf


