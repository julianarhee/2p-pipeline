#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 14:22:56 2018

@author: juliana
"""


import os
import optparse
import sys
import json
import re
import datetime
import matplotlib as mpl
mpl.use('agg')
import numpy as np
import pylab as pl
import seaborn as sns
import pandas as pd
import h5py
import traceback
import pprint
pp = pprint.PrettyPrinter(indent=4)

import glob
from pipeline.python.utils import natural_keys


#%%
def convert_values(oldval, newmin, newmax, oldmax=None, oldmin=None):
    oldrange = (oldmax - oldmin)
    newrange = (newmax - newmin)
    newval = (((oldval - oldmin) * newrange) / oldrange) + newmin
    return newval

# Convert degs to centimeters:
def get_linear_coords(width, height, resolution, leftedge=None, rightedge=None, bottomedge=None, topedge=None):
    #width = 103 # in cm
    #height = 58 # in cm
    #resolution = [1920, 1080]

    if leftedge is None:
        leftedge = -1*width/2.
    if rightedge is None:
        rightedge = width/2.
    if bottomedge is None:
        bottomedge = -1*height/2.
    if topedge is None:
        topedge = height/2.

    print "center 2 Top/Anterior:", topedge, rightedge


    mapx = np.linspace(leftedge, rightedge, resolution[0] * ((rightedge-leftedge)/float(width)))
    mapy = np.linspace(bottomedge, topedge, resolution[1] * ((topedge-bottomedge)/float(height)))

    lin_coord_x, lin_coord_y = np.meshgrid(mapx, mapy, sparse=False)

    return lin_coord_x, lin_coord_y

def get_retino_info(width=81.28, height=45.77, resolution=[1600, 900],
                    azimuth='right', elevation='top',
                    leftedge=None, rightedge=None, bottomedge=None, topedge=None):

    lin_coord_x, lin_coord_y = get_linear_coords(width, height, resolution, leftedge=leftedge, rightedge=rightedge, bottomedge=bottomedge, topedge=topedge)
    linminW = lin_coord_x.min(); linmaxW = lin_coord_x.max()
    linminH = lin_coord_y.min(); linmaxH = lin_coord_y.max()

    retino_info = {}
    retino_info['width'] = width
    retino_info['height'] = height
    retino_info['resolution'] = resolution
    aspect_ratio = float(height)/float(width)
    retino_info['aspect'] = aspect_ratio
    retino_info['azimuth'] = azimuth
    retino_info['elevation'] = elevation
    retino_info['linminW'] = linminW
    retino_info['linmaxW'] = linmaxW
    retino_info['linminH'] = linminH
    retino_info['linmaxH'] = linmaxH
    retino_info['bounding_box'] = [leftedge, bottomedge, rightedge, topedge]

    return retino_info



def convert_lincoords_lincolors(rundf, rinfo, stat_type='mean'):

    #angX = rundf.loc[slice(rinfo['azimuth']), 'phase_%s' % stat_type].values
    angX = rundf.xs(rinfo['azimuth'], axis=0)['phase_%s' % stat_type].values 
    angY = rundf.xs(rinfo['elevation'], axis=0)['phase_%s' % stat_type].values
    #print rundf.xs(rinfo['azimuth'], axis=0).head()
    #print rundf.head() #.loc[slice(rinfo['azimuth']), 'phase_%s' % stat_type].head()

    # Convert phase range to linear-coord range:
    linX = convert_values(angX, rinfo['linminW'], rinfo['linmaxW'], oldmax=2*np.pi, oldmin=0) #0, oldmin=2*np.pi)  # If cond is 'right':  positive values = 0, negative values = 2pi
    linY = convert_values(angY, rinfo['linminH'], rinfo['linmaxH'], oldmax=2*np.pi, oldmin=0)  # If cond is 'top':  positive values = 0, negative values = 2pi
    linC = np.arctan2(linY,linX)

    return linX, linY, linC


def plot_roi_retinotopy(linX, linY, rgbas, retino_info, curr_metric='magratio_mean',
                        alpha_min=0, alpha_max=1, color_position=False,
                        output_dir='', figname='roi_retinotopy.png', save_and_close=True):
    sns.set()
    fig = pl.figure(figsize=(10,8))
    ax = fig.add_subplot(111) #, aspect=retino_info['aspect'])
    if color_position is True:
        pl.scatter(linX, linY, s=150, c=rgbas, cmap='hsv', vmin=-np.pi, vmax=np.pi) #, vmin=0, vmax=2*np.pi)
        magcmap = mpl.cm.Greys
    else:
        pl.scatter(linX, linY, s=150, c=rgbas, cmap='inferno', alpha=0.75, edgecolors='w') #, vmin=0, vmax=2*np.pi)
        magcmap=mpl.cm.inferno

    #pl.gca().invert_xaxis()  # Invert x-axis so that negative values are on left side
#    pl.xlim([retino_info['linminW'], retino_info['linmaxW']])
#    pl.ylim([retino_info['linminH'], retino_info['linmaxH']])
    pl.xlim([-1*retino_info['width']/2., retino_info['width']/2.])
    pl.ylim([-1*retino_info['height']/2., retino_info['height']/2.])
    print("Screen limits: %s" % str([-1*retino_info['width']/2., retino_info['width']/2.]))

    pl.xlabel('x position')
    pl.ylabel('y position')
    pl.title('ROI position selectivity (%s)' % curr_metric)
    pos = ax.get_position()
    ax2 = fig.add_axes([pos.x0+.8, pos.y0, 0.01, pos.height])
    #magcmap = mpl.cm.Greys
#    if alpha_max < 0.05:
#        alpha_max = 0.05
    magnorm = mpl.colors.Normalize(vmin=alpha_min, vmax=alpha_max)
    cb = mpl.colorbar.ColorbarBase(ax2, cmap=magcmap, norm=magnorm, orientation='vertical')

    if save_and_close is True:
        datestr = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        figname = '%s_%s.png' % (os.path.splitext(figname)[0], datestr)
        pl.savefig(os.path.join(output_dir, figname))
        pl.close()


#%%

def extract_options(options):

    parser = optparse.OptionParser()

    parser.add_option('-D', '--root', action='store', dest='rootdir',
                          default='/n/coxfs01/2p-data',
                          help='data root dir (dir containing all animalids) [default: /nas/volume1/2photon/data, /n/coxfs01/2pdata if --slurm]')
    parser.add_option('-i', '--animalid', action='store', dest='animalid',
                          default='', help='Animal ID')

    # Set specific session/run for current animal:
    parser.add_option('-S', '--session', action='store', dest='session',
                          default='', help='session dir (format: YYYMMDD_ANIMALID')
    parser.add_option('-A', '--acq', action='store', dest='acquisition',
                          default='FOV1', help="acquisition folder (ex: 'FOV1_zoom3x') [default: FOV1]")
    parser.add_option('-R', '--run', action='store', dest='run', default='', help="name of run dir containing tiffs to be processed (ex: gratings_phasemod_run1)")
    parser.add_option('--slurm', action='store_true', dest='slurm', default=False, help="set if running as SLURM job on Odyssey")

    parser.add_option('-t', '--trace-id', action='store', dest='trace_id', default='', help="Trace ID for current trace set (created with set_trace_params.py, e.g., traces001, traces020, etc.)")

    parser.add_option('--default', action='store_true', dest='auto', default=False, help="set if want to use all defaults")

    parser.add_option('--positions', action='store_true', dest='color_position', default=False, help="set if want to view position responses as color map (retinotopy)")

    parser.add_option('-B', '--bbox', dest='boundingbox_runs', default=[], nargs=1, action='append', help="RUN that is a bounding box run (only for retino)")
    parser.add_option('-l', '--left', dest='leftedge', default=None, action='store', help="left edge of bounding box")
    parser.add_option('-r', '--right', dest='rightedge', default=None, action='store', help="right edge of bounding box")
    parser.add_option('-u', '--upper', dest='topedge', default=None, action='store', help="upper edge of bounding box")
    parser.add_option('-b', '--lower', dest='bottomedge', default=None, action='store', help="bottom edge of bounding box")

    (options, args) = parser.parse_args(options)

    return options


#%%

def get_retino_datafile_paths(acquisition_dir, run, traceid):
    dfpaths = {}

    # This is likely a retino run
    trace_basename = 'retino_analysis'
    tdict_path = os.path.join(acquisition_dir, run, trace_basename, 'analysisids_%s.json' % run)
    hash_type = 'analysis_hash'

    with open(tdict_path, 'r') as f:
        tdict = json.load(f)
    trace_idname = '%s_%s' % (traceid, tdict[traceid][hash_type])
    traceid_dir = os.path.join(acquisition_dir, run, trace_basename, trace_idname)

    dfilepath = [os.path.join(traceid_dir, 'files', f) for f in os.listdir(os.path.join(traceid_dir, 'files')) if 'retino_data' in f]

    dfpaths[run] = dfilepath

    return dfpaths

def assign_mag_ratios(dataframes, run, stat_type='mean', metric_type='magratio'):

    # Check if we want ZSCORE or MAG-RATIO:
#    if dataframes[run]['is_phase']:
#        metric_type = 'magratio'

    xrun = dataframes[run]['df']
    curr_metric = '%s_%s' % (metric_type, stat_type)
    best_mag_ratios = xrun.groupby(['roi']).agg({curr_metric: {'%s1' % curr_metric: 'max'}})
    best_mag_ratios.columns = best_mag_ratios.columns.get_level_values(1)
    zdf = pd.concat([best_mag_ratios], axis=1)

    return zdf


def visualize_position_data(dataframes, zdf, retino_info,
                            set_response_alpha=True, stat_type='mean', color_position=False,
                            acquisition_str='', output_dir='/tmp', save_and_close=True):

    # Get RGBA mapping normalized to mag-ratio values:
    norm = mpl.colors.Normalize(vmin=-np.pi, vmax=np.pi)
    cmap = mpl.cm.get_cmap('hsv')
    mapper = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    retino_runs = [k for k in retino_info.keys() if 'retino' in k]
    if color_position is False and len(retino_runs)>1:
        same_alpha=True
        alpha_min = min([zdf[k].min() for k in zdf.keys()])
        alpha_max = min([zdf[k].max() for k in zdf.keys()])
    else:
        same_alpha=False
        alpha_min=None; alpha_max=None

    visinfo = dict((run, dict()) for run in dataframes.keys())
    for runnum, run in enumerate(dataframes.keys()):
        rundf = dataframes[run]['df']
        if dataframes[run]['is_phase']:
            curr_metric = 'magratio_%s%i' % (stat_type, int(runnum+1))
            linX, linY, linC = convert_lincoords_lincolors(rundf, retino_info[run], stat_type=stat_type)
#        else:
#            curr_metric = 'zscore_%s%i' % (stat_type, int(runnum+1))
#            linX, linY, linC = assign_lincoords_lincolors(rundf, zdf[curr_metric].values, stat_type=stat_type)

        visinfo[run]['linX'] = linX
        visinfo[run]['linY'] = linY
        visinfo[run]['linC'] = linC
        visinfo[run]['metric'] = curr_metric

        rgbas = np.array([mapper.to_rgba(v) for v in linC])
        magratios = zdf[curr_metric]
        if alpha_min is None:
            alpha_min = magratios.min()
        if alpha_max is None:
            alpha_max = magratios.max()

        if set_response_alpha is True:
            alphas = np.array(magratios / magratios.max())
            rgbas[:, 3] = alphas
        visinfo[run]['rgbas'] = rgbas

        if dataframes[run]['is_phase']:
            if set_response_alpha is False:
                alphas = np.array(zdf[curr_metric] / zdf[curr_metric].max())
                rgbas[:, 3] = alphas

            # Plot each ROI's "position" color-coded with angle map:
            if color_position is True:
                figname = '%s_R%i-%s_position_selectivity_%s_Cpos.png' % (acquisition_str, int(runnum+1), run, curr_metric)
                plot_roi_retinotopy(linX, linY, rgbas, retino_info[run], curr_metric=curr_metric,
                                    alpha_min=zdf[curr_metric].min(), alpha_max=zdf[curr_metric].max(),
                                    output_dir=output_dir, figname=figname, save_and_close=save_and_close)
            else:
                figname = '%s_R%i-%s_position_selectivity_%s_Cmagr.png' % (acquisition_str, int(runnum+1), run, curr_metric)
                plot_roi_retinotopy(linX, linY, magratios, retino_info[run], curr_metric=curr_metric,
                                    alpha_min=alpha_min, alpha_max=alpha_max, color_position=color_position,
                                    output_dir=output_dir, figname=figname, save_and_close=save_and_close)

    return visinfo


#%%

def get_metricdf(dfpaths):
    #all_dfs = []
    #retino_runs = [f for f in dfpaths.keys() if 'retino_analysis' in dfpaths[f][0]]
    #event_runs = [f for f in dfpaths.keys() if 'traces' in dfpaths[f]]
    run_list = dfpaths.keys()
    dataframes = {}

    currdf= []
    for run in run_list:
        if 'retino_analysis' in dfpaths[run][0]:
            is_retino = True
            rundir = (dfpaths[run][0]).split('/retino_analysis')[0]
            paradigmdir = os.path.join(rundir, 'paradigm', 'files')
            paradigm_info_fn = [f for f in os.listdir(paradigmdir) if 'parsed_' in f][0]
            with open(os.path.join(paradigmdir, paradigm_info_fn), 'r') as f:
                paradigm_info = json.load(f)
            #stimconfigs = sorted(list(set([paradigm_info[f]['stimuli']['stimulus'] for f in paradigm_info.keys()])), key=natural_keys)

            for di,df in enumerate(dfpaths[run]):
                if os.stat(df).st_size == 0:
                    continue
                rundf = h5py.File(df, 'r')
                if len(rundf) == 0:
                    continue
                curr_file = str(re.search('File(\d{3})', df).group(0))
                file_index_str = str(int(curr_file[4:]))

                currconfig = paradigm_info[file_index_str]['stimuli']['stimulus']
                currtrial = 'trial%05d' % int(file_index_str)
                run_name = os.path.split(df.split('/retino_analysis')[0])[-1]
                if len(run_name.split('_')) > 3:
                    run_name = run_name.split('_')[0]

                print "Getting mag-ratios for each ROI in run: %s, trial %s" % (run_name, file_index_str) # di+1)
                roi_list = ['roi%05d' % int(r+1) for r in range(rundf['mag_ratio_array'].shape[0])]
                phase_convert = -1 * rundf['phase_array'][:]
                phase_convert = phase_convert % (2*np.pi)
                currdf.append(pd.DataFrame({'roi': roi_list,
                                       'config': np.tile(currconfig, (len(roi_list),)),
                                       'trial': np.tile(currtrial, (len(roi_list),)),
                                       'magratio': rundf['mag_ratio_array'][:],
                                       'phase': phase_convert,
                                       'run': run_name
                                       }))
            data = pd.concat(currdf, axis=0)
            metricdf = data.groupby(['config', 'roi']).agg({'magratio': {'magratio_mean': 'mean', 'magratio_max': 'max'},
                                                            'phase': {'phase_mean': 'mean'}
                                                            })
            # Get phases at max mag-ratio:
            phases_at_max = [data[(data['roi']==ind[1]) & (data['config']==ind[0]) & (data['magratio']==val)]['phase'].values[0]
                                for ind, val in zip(metricdf['magratio']['magratio_max'].index.tolist(), metricdf['magratio']['magratio_max'].values)]
            metricdf.columns = metricdf.columns.get_level_values(1)
            metricdf['phase_atmax'] = phases_at_max

        elif 'traces' in dfpaths[run]:
            is_retino = False
            df = dfpaths[run]
            traceid = os.path.split(df.split('/metrics')[0])[-1]
            if len(traceid.split('_')) > 1:
                is_combo = True
            else:
                is_combo = False
            if is_combo:
                rundf = pd.HDFStore(df, 'r')
                rundf = rundf[rundf.keys()[0]]
            else:
                rundf = pd.HDFStore(df, 'r')['/df']

            run_name = os.path.split(df.split('/traces')[0])[-1]
            if len(run_name.split('_')) > 3:
                run_name = run_name.split('_')[0]

            print "Compiling zscores for each ROI in run: %s" % run_name
            roi_list = sorted(list(set(rundf['roi'])), key=natural_keys)

            subdf = rundf[['roi', 'config', 'trial', 'stim_df', 'zscore', 'xpos', 'ypos']]
            metricdf = subdf.groupby(['config', 'xpos', 'ypos', 'roi']).agg({'zscore': {'zscore_mean': 'mean', 'zscore_max': 'max'},
                                                             'stim_df': {'stimdf_mean': 'mean', 'stimdf_max': 'max'}
                                                             })
            # Concatenate all info for this current trial:
            metricdf.columns = metricdf.columns.get_level_values(1)

        dataframes[run_name] = {}
        dataframes[run_name]['df'] = metricdf
        dataframes[run_name]['is_phase'] = is_retino


    return dataframes #, gridinfo

#%%
def get_screen_info(animalid, session, fov=None, interactive=True, rootdir='/n/coxfs01/2p-data'):
        
    screen = {}
    
    try:
        # Get bounding box values from epi:
        epi_session_paths = sorted(glob.glob(os.path.join(rootdir, animalid, 'epi_maps', '20*')), key=natural_keys)
        epi_sessions = sorted([os.path.split(s)[-1].split('_')[0] for s in epi_session_paths], key=natural_keys)
        print("Found epi sessions: %s" % str(epi_sessions))
        if len(epi_sessions) > 0:
            epi_sesh = [datestr for datestr in sorted(epi_sessions, key=natural_keys) if int(datestr) <= int(session)][-1] # Use most recent session
            print("Most recent: %s" % str(epi_sesh))

            epi_fpaths = glob.glob(os.path.join(rootdir, animalid, 'epi_maps', '*%s*' % epi_sesh, 'screen_boundaries*.json'))
            if len(epi_fpaths) == 0:
                epi_fpaths = glob.glob(os.path.join(rootdir, animalid, 'epi_maps', '*%s*' % epi_sesh, '*', 'screen_boundaries*.json'))

        else:
            print("No EPI maps found for session: %s * (trying to use tmp session boundaries file)")
            epi_fpaths = glob.glob(os.path.join(rootdir, animalid, 'epi_maps', 'screen_boundaries*.json'))
        
        assert len(epi_fpaths) > 0, "No epi screen info found!"
        
        # Each epi run should have only 2 .json files (1 for each condition):
        if len(epi_fpaths) > 2:
            print("-- found %i screen boundaries files: --" % len(epi_fpaths))
            repeat_epi_sessions = sorted(list(set( [os.path.split(s)[0] for s in epi_fpaths] )), key=natural_keys)
            for ei, repeat_epi in enumerate(sorted(repeat_epi_sessions, key=natural_keys)):
                print(ei, repeat_epi)
            if interactive:
                selected_epi = input("Select IDX of epi run to use: ")
            else:
                assert fov is not None, "ERROR: not interactive, but no FOV specified and multiple epis for session %s" % session
                
                selected_fovs = [fi for fi, epi_session_name in enumerate(repeat_epi_sessions) if fov in epi_session_name]
                print("Found FOVs: %s" % str(selected_fovs))

                if len(selected_fovs) == 0:
                    selected_epi = sorted(selected_fovs, key=natural_keys)[-1]
                else:
                    selected_epi = selected_fovs[0]
                
            epi_fpaths = [s for s in epi_fpaths if repeat_epi_sessions[selected_epi] in s]
        
        print("-- getting screen info from:", epi_fpaths)
        
        for epath in epi_fpaths:
            with open(epath, 'r') as f:
                epi = json.load(f)
            
            screen['azimuth'] = epi['screen_params']['screen_size_x_degrees']
            screen['elevation'] = epi['screen_params']['screen_size_t_degrees']
            screen['resolution'] = [epi['screen_params']['screen_size_x_pixels'], epi['screen_params']['screen_size_y_pixels']]

            if 'screen_boundaries' in epi.keys():
                if 'boundary_left_degrees' in epi['screen_boundaries'].keys():
                    screen['bb_left'] = epi['screen_boundaries']['boundary_left_degrees']
                    screen['bb_right'] = epi['screen_boundaries']['boundary_right_degrees']
                elif 'boundary_down_degrees' in epi['screen_boundaries'].keys():
                    screen['bb_lower'] = epi['screen_boundaries']['boundary_down_degrees']
                    screen['bb_upper'] = epi['screen_boundaries']['boundary_up_degrees']
            
            else:
                screen['bb_lower'] = -1*screen['elevation']/2.0
                screen['bb_upper'] = screen['elevation']/2.0
                screen['bb_left']  = -1*screen['azimuth']/2.0
                screen['bb_right'] = screen['azimuth']/2.0
    
    except Exception as e:
        traceback.print_exc()
        
    return screen

#%%
#    options = ['-D', '/mnt/odyssey', '-i', 'CE077', '-S', '20180425', '-A', 'FOV1_zoom1x',
#               '-R', 'retino_run1', '-t', 'analysis001']


#%%

def roi_retinotopy(options):

    options = extract_options(options)

    rootdir = options.rootdir
    animalid = options.animalid
    session = options.session
    acquisition = options.acquisition
    run = options.run
    traceid = options.trace_id
    slurm = options.slurm
    if slurm is True and 'coxfs01' not in rootdir:
        rootdir = '/n/coxfs01/2p-data'

    boundingbox_runs = options.boundingbox_runs
    if options.leftedge is not None:
        leftedge = float(options.leftedge)
        rightedge = float(options.rightedge)
    if options.topedge is not None:
        topedge = float(options.topedge)
        bottomedge = float(options.bottomedge)
    color_position = options.color_position

    acquisition_dir = os.path.join(rootdir, animalid, session, acquisition)

    # Get dataframe paths for runs to be compared:
    dfpaths = get_retino_datafile_paths(acquisition_dir, run, traceid)
    print dfpaths

    # Get base dir for retinotopy output:
    analysis_fpaths = dfpaths[run]
    base_dir = analysis_fpaths[0].split('/files')[0]
    output_dir = os.path.join(base_dir, 'visualization')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print "Saving output figures to:", output_dir

    # load screen coord info:
#    try:
#        screen_info_fpaths = glob.glob(os.path.join(rootdir, animalid, 'epi_maps', '*', 'screen_*.json'))
#        if len(screen_info_fpaths) == 0:
#            screen_info_fpaths = glob.glob(os.path.join(rootdir, animalid, 'epi_maps', 'screen_*.json'))
#        assert len(screen_info_fpaths) > 0, "*** No SCREEN INFO file found in epi_maps."
#        screen_info_fpath = screen_info_fpaths[0]
#        with open(screen_info_fpath, 'r') as f:
#            screen = json.load(f)
#        screen_width = screen['screen_params']['screen_size_x_degrees']
#        screen_height = screen['screen_params']['screen_size_t_degrees']
#        screen_resolution = [screen['screen_params']['screen_size_x_pixels'], screen['screen_params']['screen_size_y_pixels']]
#    except Exception as e:
#        print("--- using default screen params ---")
#        screen_width = 81.28
#        screen_height = 45.77
#        screen_resolution = [1920, 1024]
    if slurm:
        interactive = False
    else:
        interactive = True
    print("INTERACTIVE:", interactive)
    screen_info = get_screen_info(animalid, session, fov=acquisition.split('_')[0], interactive=interactive, rootdir=rootdir)
    if len(screen_info.keys()) == 0:
        screen_width = 81.28
        screen_height = 45.77
        screen_resolution = [1920, 1024]
    else:
         screen_width = screen_info['azimuth']
         screen_height = screen_info['elevation']
         screen_resolution = screen_info['resolution']

    print("*********************************")
    pp.pprint(screen_info)
    print("*********************************")


    # Create DF for easy plotting:
    print "Getting DF..."

    stat_type = 'mean'
    set_response_alpha = False


    dataframes = get_metricdf(dfpaths)
    zdf = assign_mag_ratios(dataframes, run, stat_type=stat_type, metric_type='magratio') #=stat_type, run2_stat_type=stat_type)

    run_list = dataframes.keys()
    retino_info = {}
    for run in run_list:
        if run in boundingbox_runs:
            retino_info[run] = get_retino_info(width=screen_width, height=screen_height, resolution=screen_resolution, azimuth='right', elevation='top',
                                          leftedge=leftedge, rightedge=rightedge,
                                          bottomedge=bottomedge, topedge=topedge)
        else:
            retino_info[run] = get_retino_info(width=screen_width, height=screen_height, resolution=screen_resolution, azimuth='right', elevation='top')
    pp.pprint(retino_info)


    # Get conversions for retinotopy & grid protocols:
    acquisition_str = '%s_%s_%s' % (animalid, session, acquisition)
    visinfo = visualize_position_data(dataframes, zdf, retino_info,
                                      set_response_alpha=set_response_alpha,
                                      stat_type=stat_type,
                                      color_position=color_position,
                                      save_and_close=False)

    if color_position:
        cmap_str = 'Cpos' #magr'
    else:
        cmap_str = 'Cmagr' #pos'
       
    datestr = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    figname = '%s_%s_magratio_mean_%s_%s.png' % (acquisition_str, run, cmap_str, datestr)
    pl.savefig(os.path.join(output_dir, figname))


    # Save monitor/retino info:
    with open(os.path.join(output_dir, 'retinotopy.json'), 'w') as f:
        json.dump(retino_info, f, indent=4, sort_keys=True)

#%%

def main(options):

    roi_retinotopy(options)

#%%

if __name__ == '__main__':
    main(sys.argv[1:])
