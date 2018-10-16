#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 16:05:22 2018

@author: juliana
"""

import glob
import os
import traceback
import json

import numpy as np
import cPickle as pkl
import pandas as pd
import pylab as pl
import seaborn as sns
import scipy.optimize as opt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from pipeline.python.visualization.plot_session_summary import SessionSummary


#%%

def gaussian2D((x, y), amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2)))
    return g.ravel()


def gaussian(height, center_x, center_y, width_x, width_y):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x,y: height*np.exp(
                -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)

def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    X, Y = np.indices(data.shape) # X is row indices (i.e., YPOS), Y is col indicies (i.e., XPOS)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(y)]
    width_x = np.sqrt(np.abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(np.abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
    height = data.max()
    return height, x, y, width_x, width_y

def fitgaussian(data, verbose=False):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data)
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) -
                                 data)
    
    if verbose:
        params, pcov, info, msg, success = opt.leastsq(errorfunction, params, full_output=1)
        return params, pcov
    else:
        p, success = opt.leastsq(errorfunction, params, full_output=0)
        return p, success

#%%
        

def r2_from_pcov(ydata, yfit):
    residuals = ydata - yfit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((ydata - np.mean(ydata))**2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared
        
def plot_RF_fit(xvals, yvals, zvals, figpath=None,  method=1, cmap=cm.hot):
    roi_name = os.path.splitext(os.path.split(figpath)[-1])[0]
    
    ## DataFrame from 2D-arrays
    xvals = np.array(xvals)
    yvals = np.array(yvals)
    zvals = np.array(zvals)
    df3D = pd.DataFrame({'x': xvals, 'y': yvals, 'z': zvals}, index=range(len(xvals)))

    fig = pl.figure(figsize=(10,4))
    ax1 = fig.add_subplot(121)
    
    xpositions = sorted(np.unique(xvals))
    ypositions = sorted(np.unique(yvals))
    nx = len(xpositions); ny = len(ypositions);
    x_interval = np.mean(np.diff(xpositions))
    y_interval = np.mean(np.diff(ypositions))
    
    gy, gx = np.meshgrid(xpositions, ypositions)
    maxix = df3D[df3D['z']==df3D['z'].max()].index.tolist()[0]
    
    xo = df3D.iloc[maxix]['x']
    yo = df3D.iloc[maxix]['y']
    amp = df3D.iloc[maxix]['z']
    
    # Plot actual data:
    data_grid = df3D['z'].values.reshape((ny, nx), order='F')
    im = ax1.imshow(data_grid, cmap=cmap)
    pl.xticks(np.arange(len(xpositions)), xpositions, rotation=45)
    pl.yticks(np.arange(len(ypositions)), ypositions, rotation=45)
    
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    pl.colorbar(im, cax=cax)

    try:
        no_fit = False
        if method==1:
            # Method 1: Fit using scipy.opt
            #initial_guess = (amp, xo, yo, x_interval, y_interval, 0, 0)
            theta_fits = []
            for theta in np.arange(0, np.pi, np.pi/10.):
                try:
                    initial_guess = (amp, xo, yo, x_interval, y_interval, theta, 0)
                    popt, pcov = opt.curve_fit(gaussian2D, (xvals, yvals), df3D['z'].values, p0=initial_guess)
                    data_fitted = gaussian2D((xvals, yvals), *popt)                # Fit data
                    r_squared = r2_from_pcov(df3D['z'].values, data_fitted)        # Get r2
                    theta_fits.append((theta, r_squared))
                    
                except RuntimeError:
                    # Try various rotations:
                    theta_fits.append((theta, 0))
            
            best_theta = np.array([t[1] for t in theta_fits]).argmax()
            theta = theta_fits[best_theta][0]
            
            initial_guess = (amp, xo, yo, x_interval, y_interval, theta, 0)
            popt, pcov = opt.curve_fit(gaussian2D, (xvals, yvals), df3D['z'].values, p0=initial_guess)
            data_fitted = gaussian2D((xvals, yvals), *popt)                # Fit data
            r_squared = r2_from_pcov(df3D['z'].values, data_fitted)        # Get r2
            
            (height, fx, fy, width_x, width_y, theta, offset) = popt
            data_fitted = data_fitted.reshape(data_grid.shape, order='F')  # Reshape for plotting

            estim_x = width_x * 2.
            estim_y = width_y * 2.
            peak_x = fx
            peak_y = fy
            
        else:
            # Method 2:  Fit with leastsq
            popt, pcov, msg, success = fitgaussian(data_grid, verbose=True)        # Fit a gaussian to the actual data
            fit = gaussian(*popt)                                    # Use fitted params to create a new gauss2D function
            data_fitted = fit(*np.indices(data_grid.shape))          # Create a fitted 2D gaus to plot (NOTE: Y indices are actualy "x" and X indices are actually "y")
            (height, fy, fx, width_y, width_x, rot) = popt
            r_squared = r2_from_pcov(data_grid, data_fitted)
            
            estim_x = width_x #* x_interval
            estim_y = width_y #* y_interval
            peak_x = fx #xpositions[int(round(fx))]
            peak_y = fy #ypositions[int(round(fy))]


        ax1.contour(data_fitted, cmap=cm.Greys_r) #cmap=pl.cm.hot)


    except RuntimeError:
        print "*** Bad fit: %s ***" % roi_name
        #traceback.print_exc()
        r_squared = None
        data_fitted = None
        estim_x = None; estim_y = None; peak_x = None; peak_y = None; height=None;
        no_fit = True
    

    ax2 = fig.add_subplot(122, projection='3d', azim=-124, elev=35)
    ax2.plot_trisurf(df3D.x, df3D.y, df3D.z, cmap=cmap, linewidth=0.2)

    if no_fit:
        ax2.text(43,4,ax2.get_zlim()[1], 'NO FIT',
                fontsize=16, horizontalalignment='right', color='black',
                verticalalignment='bottom') #, transform=ax1.transAxes)
    else:
        ax2.text(ax2.get_xlim()[-1], 4, ax2.get_zlim()[1], 'x : %.1f\ny : %.1f\nwidth_x : %.1f\nwidth_y : %.1f' % (peak_x, peak_y, estim_x, estim_y),
                fontsize=16, horizontalalignment='left', color='black',
                verticalalignment='top') #, transform=ax2.transAxes)
        
        
    
    if figpath is not None:
        roi_name = os.path.splitext(os.path.split(figpath)[-1])[0]
        pl.suptitle(roi_name)
        pl.savefig(figpath)
        
    pl.close()

    return estim_x, estim_y, peak_x, peak_y, height, r_squared


#%%
rootdir = '/n/coxfs01/2p-data'

#
#animalid = 'JC015'
#session = '20180919'
#acquisition = 'FOV1_zoom2p0x'
#retino_run = 'retino_run1'
#retino_id = 'analysis004'
#gratings_run = 'combined_gratings_static'
#gratings_id = 'traces003'
#
#animalid = 'JC015'
#session = '20180917'
#acquisition = 'FOV1_zoom2p0x'
#retino_run = 'retino_run1'
#retino_id = 'analysis003'
#gratings_run = 'combined_gratings_static'
#gratings_id = 'traces002'

# * this session was loaded from tmp_analysis004.pkl file...
#animalid = 'JC015'
#session = '20180925'
#acquisition = 'FOV1_zoom2p0x'
#retino_run = 'retino_run1'
#retino_id = 'analysis004'
#gratings_run = 'combined_gratings_static'
#gratings_id = 'traces003'

animalid = 'JC015'
session = '20180925'
acquisition = 'FOV1_zoom2p0x'
retino_run = 'retino_run1'
retino_id = 'analysis004'
tiled_run = 'combined_objects_static'
tiled_traceid = 'traces003'

#animalid = 'JC022'
#session = '20181005'
#acquisition = 'FOV2_zoom2p7x'
#retino_run = 'retino_run1'
#retino_id = 'analysis002'
#gratings_run = 'combined_gratings_static'
#gratings_id = 'traces001'

acquisition_dir = os.path.join(rootdir, animalid, session, acquisition)




# Load SessionSummary data (combo data paths w. stats):
# -----------------------------------------------------------------------------
ss_paths = glob.glob(os.path.join(acquisition_dir, 'session_summary_*%s_%s*_%s_%s*.pkl' % (retino_run, retino_id, tiled_run, tiled_traceid)))

assert len(ss_paths) == 1, "More than 1 specified combo found:\n ... retino -- %s, %s\n ... tiling data -- %s, %s." % (retino_run, retino_id, tiled_run, gratings_id)
ss_fpath = ss_paths[0]

with open(ss_fpath, 'rb') as f:
    S = pkl.load(f)

# Set up output dir for RF estimate results:
# ------------------------------------------
tile_dir = glob.glob(os.path.join(acquisition_dir, tiled_run, 'traces', '%s*' % tiled_traceid))[0]
tile_results_dir = os.path.join(tile_dir, 'rf_estimates')
if not os.path.exists(os.path.join(tile_results_dir, 'figures')):
    os.makedirs(os.path.join(tile_results_dir, 'figures'))
print "Saving RF estimate results to:", tile_results_dir


#%%
# Tiled data:
# ------------------
metric = 'zscore'
# TODO:  Re-extract data stats so we can also consider stim-OFF periods

if 'gratings' in tiled_run:
    dset = S.gratings
elif 'objects' in tiled_run:
    dset = S.objects
elif 'blobs' in tiled_run:
    dset = S.blobs

nrois_total = len(dset['roidata'].groups.keys())

visual_rois = dset['roistats']['rois_visual']
print "TILED: Found %i of %i visual rois." % (len(visual_rois), nrois_total)

#roi = visual_rois[3]
sconfigs = pd.DataFrame(dset['sconfigs']).T
config_grps = sconfigs.groupby(['xpos', 'ypos'])
xpositions = sconfigs['xpos'].unique().astype('float')
ypositions = sconfigs['ypos'].unique().astype('float')
size = sconfigs['size'].unique().astype('float')
if len(size) == 1: stimsize = size[0]
x_interval = np.mean(np.diff(xpositions))
y_interval = np.mean(np.diff(ypositions))

print "Tiling %i xpos, %i ypos (size: %i)" % (len(xpositions), len(ypositions), stimsize)
sorted(config_grps.groups.keys(), key=lambda x: (x[0], x[1]))


rf_results_fpath = os.path.join(tile_results_dir, 'rf_results.json')

if os.path.exists(rf_results_fpath):
    print "Loading existing results."
    with open(rf_results_fpath, 'r') as f:
        RF_data = json.load(f)

else:
    
    RF_data = {}
    for roi in visual_rois:
        roidata = dset['roidata'].get_group(roi)
    
        xvals=[]; yvals=[]; zvals=[];
        for k,g in config_grps:
            xvals.append(k[0])
            yvals.append(k[1])
            zvals.append(np.mean(roidata[roidata['config'].isin(g.index.tolist())].groupby('config')[metric].mean().values))
        
        figpath = os.path.join(tile_results_dir, 'figures', 'roi%05d.png' % int(roi+1))
        
        estim_x, estim_y, peak_x, peak_y, amp, r_squared = plot_RF_fit(xvals, yvals, zvals, method=1, figpath=figpath)
        
        RF_data[roi] = {'x': xvals, 
                        'y': yvals,
                        'z': zvals}
        
        RF_data[roi]['results'] = {'width_x': estim_x,
                                   'width_y': estim_y,
                                   'peak_x': peak_x,
                                   'peak_y': peak_y,
                                   'amplitude': amp,
                                   'r2': r_squared}
    
    # Save all results:
    with open(rf_results_fpath, 'w') as f:
        json.dump(RF_data, f, indent=4, sort_keys=True)

    
#%%
# Look at distN of RF sizes:
    
fit_thr = 0.5
good_rois = [roi for roi,res in RF_data.items() if res['results']['r2'] >= fit_thr \
                 and 0 < res['results']['width_x'] < 150 \
                 and 0 < res['results']['width_y'] < 150 \
                 and xpositions.min()-x_interval*2 <= res['results']['peak_x'] <= xpositions.max()+x_interval*2 \
                 and ypositions.min()-y_interval*2 <= res['results']['peak_y'] <= ypositions.max()+y_interval*2 
                 ]

results = {}
results['width_x'] = [RF_data[r]['results']['width_x'] for r in good_rois]
results['width_y'] = [RF_data[r]['results']['width_y'] for r in good_rois]
results['r2'] = [RF_data[r]['results']['r2'] for r in good_rois]

print "-----------------------------------------------------------------------"
print "%i out of %i visual rois fit with 2D gaussian (R2 >= %.2f)" % (len(good_rois), len(visual_rois), fit_thr)
print "-----------------------------------------------------------------------"

pl.figure()
df = pd.DataFrame(results, index=good_rois)
sns.jointplot('width_x', 'width_y', data=df, kind="hex", xlim=(0, 150), ylim=(0, 150))
pl.savefig(os.path.join(tile_results_dir, 'joint_x_y.png'))

pl.figure()
sns.distplot(df['width_x'], label='x')
sns.distplot(df['width_y'], label='y')
pl.xlim([-20, 150])
pl.xlabel('widths (deg)')
pl.legend()
pl.savefig(os.path.join(tile_results_dir, 'distplot_x_y.png'))


pl.figure()
all_widths = pd.concat([df['width_x'], df['width_y']], axis=0).reset_index(drop=True)
sns.distplot(all_widths)
pl.xlim([-20, 150])
pl.savefig(os.path.join(tile_results_dir, 'all_widths.png'))

#%%
# Plot using `.trisurf()`:

#def gauss2d(xy, amp, x0, y0, a, b, c):
#    x, y = xy
#    inner = a * (x - x0)**2
#    inner += 2 * b * (x - x0)**2 * (y - y0)**2
#    #inner += 2 * b * (x - x0) * (y - y0)
#    inner += c * (y - y0)**2
#    return amp * np.exp(-inner)


#%%


#gx, gy = np.meshgrid(range(len(xpositions)), range(len(ypositions)))

#rf_grid = np.array(z).reshape(gx.shape, order='F')
#pl.pcolor(rf_grid, cmap=cm.coolwarm)
#pl.xticks(np.arange(len(xpositions))+0.5, xpositions, rotation=45,)
#pl.yticks(np.arange(len(ypositions))+0.5, ypositions, rotation=45)

#rf_grid = np.array(z).reshape(gx.shape, order='F')
#ax1.imshow(np.flipud(rf_grid))
#pl.xticks(np.arange(len(xpositions)), xpositions, rotation=45)
#pl.yticks(np.arange(len(ypositions)), ypositions[::-1], rotation=45)


#import scipy.optimize as opt
#
#def twoD_Gaussian((x, y), amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
#    xo = float(xo)
#    yo = float(yo)    
#    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
#    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
#    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
#    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
#                            + c*((y-yo)**2)))
#    return g.ravel()
#
#initial_guess = (df3D.max()['z'], df3D.max()['x'], df3D.max()['y'], stimsize, stimsize, 0, 0)
#popt, pcov = opt.curve_fit(twoD_Gaussian, (x, y), rf_grid.ravel(), p0=initial_guess)
#
## Plot fit:
#rf_fitted = twoD_Gaussian((x, y), *popt)
#ax1.contour(range(len(xpositions)), range(len(ypositions))[::-1], rf_fitted.reshape(rf_grid.shape), 8, colors='k')
#
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

#def moments(data):
#    """Returns (height, x, y, width_x, width_y)
#    the gaussian parameters of a 2D distribution by calculating its
#    moments """
#    total = data.sum()
#    X, Y = np.indices(data.shape)
#    x = (X*data).sum()/total
#    y = (Y*data).sum()/total
#    col = data[:, int(y)]
#    width_x = np.sqrt(abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
#    row = data[int(x), :]
#    width_y = np.sqrt(abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
#    height = data.max()
#    return height, x, y, width_x, width_y, 0.0
def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    Y, X = np.indices(data.shape)
    y = np.argmax((X*np.abs(data)).sum(axis=1)/total)
    x = np.argmax((Y*np.abs(data)).sum(axis=0)/total)
    col = data[:, int(y)]
    width_x = np.sqrt(np.abs((np.arange(col.size)-y)*col).sum()/np.abs(col).sum())
    row = data[:, int(x)]
    width_y = np.sqrt(np.abs((np.arange(row.size)-x)*row).sum()/np.abs(row).sum())
    #width = ( width_x + width_y ) / 2.
    #height = np.median(data.ravel())
    height = data.max()
    return height, x, y, width_x, width_y, 0.0


def fitgaussian(data, verbose=False):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data)
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) - data)
#    p, success = scipy.optimize.leastsq(errorfunction, params)
#    return p
#
    if verbose:
        params, pcov, info, msg, success = opt.leastsq(errorfunction, params, full_output=1)
        return params, pcov, msg, success
    else:
        p, success = opt.leastsq(errorfunction, params, full_output=0)
        return p, success


