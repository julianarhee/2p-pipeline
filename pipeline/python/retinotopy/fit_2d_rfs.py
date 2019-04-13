#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 15:25:28 2019

@author: julianarhee
"""


import os
import glob
import json
import copy
import pylab as pl
import seaborn as sns
import cPickle as pkl
import numpy as np
import scipy as sp
from scipy import stats
import pandas as pd
from mpl_toolkits.axes_grid1 import AxesGrid
from pipeline.python.utils import natural_keys, label_figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
#%%


def get_rois_by_visual_area(fov_dir):
    
    included_rois = []
    segmentations = glob.glob(os.path.join(fov_dir, 'visual_areas', '*.pkl'))
    assert len(segmentations) > 0, "Specified to segment, but no segmentation file found in acq. dir!"
    if len(segmentations) == 1:
        segmentation_fpath = segmentations[0]
    else:
        for si, spath in enumerate(sorted(segmentations, key=natural_keys)):
            print si, spath
        sel = input("Select IDX of seg file to use: ")
        segmentation_fpath = sorted(segmentations, key=natural_keys)[sel]
    with open(segmentation_fpath, 'rb') as f:
        seg = pkl.load(f)
            
    included_rois = seg.regions[visual_area]['included_rois']
    print "Found %i rois in visual area %s" % (len(included_rois), visual_area)

    return included_rois



def get_responsive_rois(traceid_dir, included_rois=[]):
    # Set dirs:
    try:
        sorting_subdir = 'response_stats'
        sorted_dir = sorted(glob.glob(os.path.join(traceid_dir, '%s*' % sorting_subdir)))[-1]
    except Exception as e:
        sorting_subdir = 'sorted_rois'
        sorted_dir = sorted(glob.glob(os.path.join(traceid_dir, '%s*' % sorting_subdir)))[-1]
    sort_name = os.path.split(sorted_dir)[-1]
    print "Selected stats results: %s" % sort_name
    
    
    # Load roi stats:    
    rstats_fpath = glob.glob(os.path.join(sorted_dir, 'roistats_results.npz'))[0]
    rstats = np.load(rstats_fpath)
    
    #%
    if len(included_rois) > 0:
        all_rois = np.array(copy.copy(included_rois))
    else:
        all_rois = np.arange(0, rstats['nrois_total'])
    
    visual_rois = np.array([r for r in rstats['sorted_visual'] if r in all_rois])
    selective_rois = np.array([r for r in rstats['sorted_selective'] if r in all_rois])
    
    print "Found %i cells that pass responsivity test (%s, p<%.2f)." % (len(visual_rois), rstats['responsivity_test'], rstats['visual_pval'])
    print "Found %i cells that pass responsivity test (%s, p<%.2f)." % (len(selective_rois), rstats['selectivity_test'], rstats['selective_pval'])
    
    #del rstats
    
    return visual_rois, selective_rois, rstats_fpath



#%%


rootdir = '/n/coxfs01/2p-data'
animalid = 'JC076' #'JC059'
session = '20190410' #'20190227'
fov = 'FOV1_zoom2p0x' #'FOV4_zoom4p0x'
run = 'combined_gratings_static'
traceid = 'traces001' #'traces001'
segment = False
visual_area = ''


fov_dir = os.path.join(rootdir, animalid, session, fov)
traceid_dir = glob.glob(os.path.join(fov_dir, run, 'traces', '%s*' % traceid))[0]
data_fpath = glob.glob(os.path.join(traceid_dir, 'data_arrays', '*.npz'))[0]
dset = np.load(data_fpath)
dset.keys()

data_identifier = '|'.join([animalid, session, fov, run, traceid, visual_area])
print data_identifier


#%%
if segment:
    included_rois = get_rois_by_visual_area(fov_dir, segment=segment)
else:
    included_rois = []
    
visual_rois, selective_rois, rstats_fpath = get_responsive_rois(traceid_dir, included_rois=included_rois)

# Set output dir:
output_dir = os.path.join(traceid_dir, 'figures', 'receptive_fields')
if segment:
    output_dir = os.path.join(output_dir, visual_area)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
print "Saving output to:", output_dir



#%%#%%


#%

# Load parsed data:
trace_type = 'corrected'
traces = dset[trace_type]

raw_traces = pd.DataFrame(traces) #, index=zscored_traces.index)


# Format condition info:
aspect_ratio = 1.747
sdf = pd.DataFrame(dset['sconfigs'][()]).T
sdf['size'] = [round(sz/aspect_ratio, 1) for sz in sdf['size']]
labels = pd.DataFrame(data=dset['labels_data'], columns=dset['labels_columns'])

#%


# Get single value for each trial and sort by config:
trials_by_cond = dict()
for k, g in labels.groupby(['config']):
    trials_by_cond[k] = sorted([int(tr[5:])-1 for tr in g['trial'].unique()])
del traces

# zscore the traces:
# -----------------------------------------------------------------------------
#zscores = dset['zscore']

zscored_traces_list = []
zscores_list = []
for trial, tmat in labels.groupby(['trial']):
    #print trial    
    stim_on_frame = tmat['stim_on_frame'].unique()[0]
    nframes_on = tmat['nframes_on'].unique()[0]
    curr_traces = raw_traces.iloc[tmat.index] ##traces[tmat.index, :]
    bas_std = curr_traces.iloc[0:stim_on_frame].std(axis=0)
    #curr_zscored_traces = pd.DataFrame(curr_traces, index=tmat.index).divide(bas_std, axis='columns')
    curr_zscored_traces = pd.DataFrame(curr_traces).divide(bas_std, axis='columns')
    zscored_traces_list.append(curr_zscored_traces)
    zscores_list.append(curr_zscored_traces.iloc[stim_on_frame:stim_on_frame+(nframes_on*2)].mean(axis=0)) # Get average zscore value for current trial
    
zscored_traces = pd.concat(zscored_traces_list, axis=0)
zscores =  pd.concat(zscores_list, axis=1).T # cols=rois, rows = trials


zscores_by_cond = dict()
for cfg, trial_ixs in trials_by_cond.items():
    zscores_by_cond[cfg] = zscores.iloc[trial_ixs]  # For each config, array of size ntrials x nrois


#%

# Sort ROIs by zscore by cond
# -----------------------------------------------------------------------------
avg_zscores_by_cond = pd.DataFrame([zscores_by_cond[cfg].mean(axis=0) \
                                    for cfg in sorted(zscores_by_cond.keys(), key=natural_keys)]) # nconfigs x nrois

    
# Sort mean (or max) zscore across trials for each config, and find "best config"
visual_max_avg_zscore = np.array([avg_zscores_by_cond[rid].max() for rid in visual_rois])
visual_sort_by_max_zscore = np.argsort(visual_max_avg_zscore)[::-1]
sorted_visual = visual_rois[visual_sort_by_max_zscore]

selective_max_avg_zscore = np.array([avg_zscores_by_cond[rid].max() for rid in selective_rois])
selective_sort_by_max_zscore = np.argsort(selective_max_avg_zscore)[::-1]
sorted_selective = selective_rois[selective_sort_by_max_zscore]

print [r for r in sorted_selective if r not in sorted_visual]

print sorted_selective[0:10]



#%%%
rows = 'ypos'
cols = 'xpos'

row_vals = sorted(sdf[rows].unique())
col_vals = sorted(sdf[cols].unique())


#%%
config_trial_ixs = dict()
cix = 0
for si, row_val in enumerate(sorted(row_vals)):
    for mi, col_val in enumerate(col_vals):
        config_trial_ixs[cix] = {}
        cfg = sdf[(sdf[rows]==row_val) & (sdf[cols]==col_val)].index.tolist()[0]
        trial_ixs = sorted( list(set( [int(tr[5:])-1 for tr in labels[labels['config']==cfg]['trial']] )) )
        config_trial_ixs[cix]['config'] = cfg
        config_trial_ixs[cix]['trial_ixs'] = trial_ixs
        cix += 1


#%%


import scipy.optimize as opt
from matplotlib.patches import Ellipse


def twoD_Gaussian((x, y), amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    # b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    b = (np.sin(2*theta))/(2*sigma_x**2) - (np.sin(2*theta))/(2*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
#    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
#                            + c*((y-yo)**2)))
    g = offset + amplitude*np.exp( -a*((x-xo)**2) - b*(x-xo)*(y-yo) - c*((y-y0)**2) )
    return g.ravel()

#p1, success = opt.leastsq(twoD_gauss, coordmap_r, args=(xx, yy, initial_guess[:]))  #[0]

def twoD_gauss((x, y), b, x0, y0, sigma_x, sigma_y, theta, a):

    RF = a + b * np.exp( -( ((x-x0)*np.cos(theta) + (y-y0)*np.sin(theta)) / (np.sqrt(2)*sigma_x) )**2 - ( ( -(x-x0)*np.sin(theta) + (y-y0)*np.cos(theta) ) / (np.sqrt(2)*sigma_y) )**2 )
    
    return RF.ravel()



#%%

import scipy as sp

def gaussian(height, center_x, center_y, width_x, width_y, rotation):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)

    rotation = np.deg2rad(rotation)
    center_x = center_x * np.cos(rotation) - center_y * np.sin(rotation)
    center_y = center_x * np.sin(rotation) + center_y * np.cos(rotation)

    def rotgauss(x,y):
        xp = x * np.cos(rotation) - y * np.sin(rotation)
        yp = x * np.sin(rotation) + y * np.cos(rotation)
        g = height*np.exp(
            -(((center_x-xp)/width_x)**2+
              ((center_y-yp)/width_y)**2)/2.)
        return g
    return rotgauss

def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    X, Y = np.indices(data.shape)
    
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(y)]
    width_x = np.sqrt(abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
        
        
#    y = (X*data).sum()/total
#    x = (Y*data).sum()/total
#    
#    row = data[int(y), :] # Get all 'x' values where y==center  #col = data[:, int(y)]
#    width_x = np.sqrt(abs((np.arange(row.size)-y)**2*row).sum()/row.sum())
#    
#    col = data[:, int(x)]
#    width_y = np.sqrt(abs((np.arange(col.size)-x)**2*col).sum()/col.sum())
#    
    
    
    height = data.max()
    #return height, x, y, width_x, width_y, 0.0
    return height, y, x, width_y, width_x, 0.0


def fitgaussian(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data)
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) - data)
    p, pcov, infod, msg, success = sp.optimize.leastsq(errorfunction, params, full_output=True)
    return p, pcov
        


    
#
#%%
from scipy.signal import argrelextrema
from scipy.interpolate import splrep, sproot, splev, interp1d

class MultiplePeaks(Exception): pass
class NoPeaksFound(Exception): pass

def fwhm(x, y, k=3):
    """
    Determine full-with-half-maximum of a peaked set of points, x and y.

    Assumes that there is only one peak present in the datasset.  The function
    uses a spline interpolation of order k.
    """

    half_max = max(y)/2.0
    s = splrep(x, y - half_max, k=k)
    roots = sproot(s)

    if len(roots) > 2:
        raise MultiplePeaks("The dataset appears to have multiple peaks, and "
                "thus the FWHM can't be determined.")
    elif len(roots) < 2:
        raise NoPeaksFound("No proper peaks were found in the data set; likely "
                "the dataset is flat (e.g. all zeros).")
    else:
        return abs(roots[1] - roots[0])


def raw_fwhm(arr):
    
    interpf = interp1d(np.linspace(0, len(arr)-1, num=len(arr)), arr, kind='linear')
    xnew = np.linspace(0, len(arr)-1, num=len(arr)*3)
    ynew = interpf(xnew)
    
    
    hm = ((ynew.max() - ynew.min()) / 2 ) + ynew.min()
    pk = ynew.argmax()
    if pk == 0:
        r2 = pk + np.abs(ynew[pk:] - hm).argmin()
        return abs(xnew[r2]*2)
    else:
        r1 = np.abs(ynew[0:pk] - hm).argmin()
        r2 = pk + np.abs(ynew[pk:] - hm).argmin()
        
        return abs(xnew[r2]-xnew[r1])

#%%

#map_thr = 0.6
hard_cutoff = True
plot_zscored = True
set_to_min = True

use_moments = False

sig_scale = 2.35


# Create subdir for saving figs/results based on fit params:
# -----------------------------------------------------------------------------
fit_method = 'moments' if use_moments else '2dgaus'
minval = 'min' if set_to_min else 'zeros'

if hard_cutoff:
    map_thr = 1.5
    cutoff_type = 'hard_thr'
else:
    map_thr = 0.6
    cutoff_type = 'perc_max'
    
trace_type = 'zscored_RF' if plot_zscored else 'raw_RF'

# Create output dir to save ROI plots in:
roi_rf_dir = os.path.join(output_dir, 'rfs_by_roi_%s_%s_%.2f_set_%s' % (fit_method, cutoff_type, map_thr, minval))
if not os.path.exists(roi_rf_dir):
    os.makedirs(roi_rf_dir)
print "Saving figures to:", roi_rf_dir


#%%


results_outfile = 'roi_fit_results_%s_%s_%.2f_set_%s.pkl' % (fit_method, cutoff_type, map_thr, minval)
print results_outfile

do_fits = False
if os.path.exists(os.path.join(output_dir, results_outfile)):
    print "Loading existing results..."
    with open(os.path.join(output_dir, results_outfile), 'rb') as f:
        results = pkl.load(f)
        
else:
    do_fits = True
    
#%%

#rid = 106 ##89 #106 #36 # 89
if do_fits:
    results ={}
    RF = {}
    for rid in selective_rois:
        #%
        print rid
        metric_name = 'zscore' if plot_zscored else 'snr'
        value_name = 'zscore' if plot_zscored else 'intensity'
    
        # coordmap_r = np.reshape(avg_zscores_by_cond[rid], (len(col_vals), len(row_vals))).T
        coordmap_r = np.reshape(avg_zscores_by_cond[rid], (len(col_vals), len(row_vals))).T
         
        rfmap = coordmap_r.copy()
        if hard_cutoff:
            rfmap[coordmap_r < map_thr] = coordmap_r.min()*0.3 if set_to_min else 0
        else:
            rfmap[coordmap_r < coordmap_r.max()*map_thr] =  coordmap_r.min()*0.3 if set_to_min else 0

        # Do fit 
        # ---------------------------------------------------------------------
        # Set params for fit:
        xi = np.arange(0, len(col_vals))
        yi = np.arange(0, len(row_vals))


        try:
#            if use_moments:
#                xx, yy = np.meshgrid(xi, yi)      
#                popt, pcov = fitgaussian(rfmap)
#                gfunc = gaussian(*popt)
#                fitr = gfunc(yy, xx).T.ravel()   #.ravel() # x and y are flipped because treataed as array, not image
#                plot_xx = xx #yy.T
#                plot_yy = yy #xx.T
#                
#            else:
            xx, yy = np.meshgrid(xi, yi)      
            plot_xx = xx.copy()
            plot_yy = yy.copy()
            amplitude = rfmap.max()
            y0, x0 = np.where(rfmap == rfmap.max())
            print "x0, y0: (%i, %i)" % (int(x0), int(y0))
        
            try:
                sigma_x = fwhm(xi, rfmap.sum(axis=0))
            except Exception as e:
                print e
                sigma_x = raw_fwhm(rfmap.sum(axis=0)) 
            try:
                sigma_y = fwhm(yi, rfmap.sum(axis=1))
            except Exception as e:
                print e
                sigma_y = raw_fwhm(rfmap.sum(axis=1))
            print "sig-X, sig-Y:", sigma_x, sigma_y
            theta = 0
            offset=0
            initial_guess = (amplitude, int(x0), int(y0), sigma_x, sigma_y, theta, offset)
            popt, pcov = opt.curve_fit(twoD_gauss, (xx, yy), rfmap.ravel(), p0=initial_guess, maxfev=2000)
            fitr = twoD_gauss((xx, yy), *popt)
                
            # Get residual sum of squares 
            residuals = rfmap.ravel() - fitr
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((rfmap.ravel() - np.mean(rfmap.ravel()))**2)
            r2 = 1 - (ss_res / ss_tot)
            if round(r2, 3) < 0.15 or len(np.where(fitr > fitr.min())[0]) < 4:
                cando_fit = False
            else:
                cando_fit = True
        except Exception as e:
            print e
            r2 = None
            cando_fit = False
        
        
        fig, axes = pl.subplots(1,2, figsize=(8, 4)) # pl.figure()
        ax = axes[0]
        im = ax.imshow(rfmap, cmap='inferno')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')
        
        if cando_fit:    
            # Draw ellipse contour:
            ax.contour(plot_xx, plot_yy, fitr.reshape(rfmap.shape), 3, colors='b')
            
            # Draw ellipse:
#            if use_moments:
#                amp_f, x0_f, y0_f, sigx_f, sigy_f, theta_f = popt # x and y are flipped because array, not image
#                offset_f = None
#            else:
            amp_f, x0_f, y0_f, sigx_f, sigy_f, theta_f, offset_f = popt
            ell = Ellipse((x0_f, y0_f), abs(sigx_f)*sig_scale, abs(sigy_f)*sig_scale, angle=np.rad2deg(theta_f)) #theta_f)
            ell.set_alpha(.5)
            ell.set_edgecolor('w')
            ax.add_patch(ell)
            ax.text(0, -1, 'r2=%.2f, theta=%.2f' % (r2, theta_f), color='k')
            
            ax2 = axes[1]
            #asp = np.abs( np.diff(ax.get_ylim())[0] / np.diff(ax.get_xlim())[0] )
            #ax2.set_aspect(asp)
            im2 = ax2.imshow(pcov)
            ax2.set_yticks(np.arange(0, 7))
            ax2.set_yticklabels(['amp', 'x0', 'y0', 'sigx', 'sigy', 'theta', 'offset'], rotation=0)
            
            
            divider = make_axes_locatable(ax2)
            cax2 = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im2, cax=cax2, orientation='vertical')
            

        else:
            ax.text(0, -1, 'no fit')

            
        fig.suptitle('roi %i' % int(rid+1))

        bbox1 = ax.get_position()
        subplot_ht = bbox1.height
        bbox2 = ax2.get_position()
        ax2.set_position([bbox2.x0, bbox1.y0, subplot_ht, subplot_ht])
        
        pl.subplots_adjust(wspace=0.3, left=0.1, right=0.9)
        
        label_figure(fig, data_identifier)
        figname = '%s_roi%05d' % (trace_type, int(rid+1))
        pl.savefig(os.path.join(roi_rf_dir, '%s.png' % figname))
        
        pl.close()
        
        #%
        if cando_fit:
            RF[rid] = {'amplitude': amp_f,
                       'x0': x0_f,
                       'y0': y0_f,
                       'sigma_x': sigx_f,
                       'sigma_y': sigy_f,
                       'theta': theta_f,
                       'offset': offset_f,
                       'r2': r2,
                       'fit': fitr,
                       'data': rfmap}
        
    results['fits'] = RF
    results['map_thr'] = map_thr
    results['cut_off'] = cutoff_type
    results['set_min_val'] = minval
    results['zcored'] = plot_zscored
    results['row_vals'] = row_vals
    results['col_vals'] = col_vals
    results['use_moments'] = use_moments
    results['xx'] = plot_xx
    results['yy'] = plot_yy
    
    with open(os.path.join(output_dir, results_outfile), 'wb') as f:
        pkl.dump(results, f, protocol=pkl.HIGHEST_PROTOCOL)
        



    
#%%

fit_thr = 0.51
fitdf = pd.DataFrame(results['fits']).T
fitted_rois = fitdf[fitdf['r2'] > fit_thr].sort_values('r2', axis=0, ascending=False).index.tolist()

print "%i out of %i fit rois with r2 > %.2f" % (len(fitted_rois), fitdf.shape[0], fit_thr)

#%%
plot_ellipse = True
single_colorbar = False

cmap = 'magma' if plot_ellipse else 'inferno' # inferno
cbar_mode = 'single' if single_colorbar else  'each'

vmin = max([avg_zscores_by_cond.min().min(), 0])
vmax = min([5, avg_zscores_by_cond.max().max()])


nr=6# 6 #6
nc=10 #10 #10
fig = pl.figure(figsize=(nc*2,nr+2))
grid = AxesGrid(fig, 111,
            nrows_ncols=(nr, nc),
            axes_pad=0.5,
            cbar_mode=cbar_mode,
            cbar_location='right',
            cbar_pad=0.05, cbar_size="3%")

    
for aix, rid in enumerate(fitted_rois[0:nr*nc]):
    ax = grid.axes_all[aix]
    ax.clear()
    coordmap = np.reshape(avg_zscores_by_cond[rid], (len(col_vals), len(row_vals))).T
    
    #rfmap = fitdf['data'][rid]
    im = ax.imshow(coordmap, cmap=cmap) #, vmin=vmin, vmax=vmax)
    ax.contour(results['xx'], results['yy'], fitdf['fit'][rid].reshape(coordmap.shape), 1, colors='w')
    ax.set_title('roi %i (r2=%.2f)' % (int(rid+1), fitdf['r2'][rid]), fontsize=8)
    
    if plot_ellipse:
        # = Ellipse((x0_f, y0_f), abs(sigx_f)*sig_scale, abs(sigy_f)*sig_scale, angle=np.rad2deg(theta_f)) #theta_f)

        ell = Ellipse((fitdf['x0'][rid], fitdf['y0'][rid]), abs(fitdf['sigma_x'][rid])*sig_scale, abs(fitdf['sigma_y'][rid])*sig_scale, angle=np.rad2deg(fitdf['theta'][rid]))
        ell.set_alpha(0.5)
        ell.set_edgecolor('w')
        ell.set_facecolor('cornflowerblue')
        ax.add_patch(ell)
        
    if not single_colorbar:
        cbar = ax.cax.colorbar(im)
        cbar = grid.cbar_axes[aix].colorbar(im)
        cbar_yticks = [coordmap.min(), coordmap.max()]
        cbar.cbar_axis.axes.set_yticks(cbar_yticks)
        cbar.cbar_axis.axes.set_yticklabels([int(round(cy)) for cy in cbar_yticks], fontsize=8)
    
    ax.set_ylim([0, len(row_vals)])
    ax.set_xlim([0, len(col_vals)])
    
    ax.invert_yaxis()

if single_colorbar:
    cbar = ax.cax.colorbar(im)
    cbar = grid.cbar_axes[0].colorbar(im)

#%
for a in np.arange(0, nr*nc):
    grid.axes_all[a].set_axis_off() 

if not single_colorbar and len(fitted_rois) < (nr*nc):
    for nix in np.arange(len(fitted_rois), nr*nc):
        grid.cbar_axes[nix].remove()

pl.subplots_adjust(left=0.05, right=0.95, wspace=0.3, hspace=0.3)

label_figure(fig, data_identifier)
figname = 'RF_fits_%s_FIT%s_%s_%.2f_set_%s_top%i_fit_thr_%.2f' % (trace_type, fit_method,  cutoff_type, map_thr, minval, len(fitted_rois), fit_thr)
if plot_ellipse:
    figname = '%s_ellipse' % figname
pl.savefig(os.path.join(output_dir, '%s.png' % figname))
print figname



#%%

from matplotlib.pyplot import cm




fig, ax = pl.subplots(figsize=(12, 6))
rcolors=iter(cm.rainbow(np.linspace(0,1,len(fitted_rois))))

ax.set_yticks(np.arange(0, len(row_vals)))
ax.set_yticklabels([int(r) for r in row_vals])

ax.set_xticks(np.arange(0, len(col_vals)))
ax.set_xticklabels([int(r) for r in col_vals])

for rid in fitted_rois:
    rcolor = next(rcolors)
    #ax.plot(fitdf['x0'][rid], fitdf['y0'][rid], marker='*', color=rcolor)
    
    ell = Ellipse((fitdf['x0'][rid], fitdf['y0'][rid]), abs(fitdf['sigma_x'][rid])*sig_scale, abs(fitdf['sigma_y'][rid])*sig_scale, angle=np.rad2deg(fitdf['theta'][rid]))
    ell.set_alpha(0.5)
    ell.set_edgecolor(rcolor)
    ell.set_facecolor('none')
    ax.add_patch(ell)
#ax.invert_yaxis()

#ax.set_ylim([0, len(row_vals)])
#ax.set_xlim([0, len(col_vals)])

figname = 'overlaid_RFs_%s_FIT%s_%s_%.2f_set_%s_top%i_fit_thr_%.2f' % (trace_type, fit_method,  cutoff_type, map_thr, minval, len(fitted_rois), fit_thr)
pl.savefig(os.path.join(output_dir, '%s.png' % figname))
print figname


#%%

fig, ax = pl.subplots()    
coordmap = np.reshape(avg_zscores_by_cond[rid], (len(col_vals), len(row_vals))).T
im = ax.imshow(coordmap, cmap=cmap) #, vmin=vmin, vmax=vmax)
ax.contour(xx, yy, fitdf['fit'][rid].reshape(rfmap.shape), 2, colors='w')

ell = Ellipse((fitdf['x0'][rid], fitdf['y0'][rid]), abs(fitdf['sigma_x'][rid])*3, abs(fitdf['sigma_y'][rid])*3, angle=fitdf['theta'][rid])
ell.set_alpha(0.5)
ell.set_edgecolor('w')
ell.set_facecolor('cornflowerblue')
ax.add_patch(ell)

    
    