
import os
import sys
import numpy as np
import cPickle as pkl
import json
import optparse
import seaborn as sns
import pylab as pl
import itertools
import holoviews as hv

import pandas as pd
from optparse import OptionParser

from pipeline.python.utils import natural_keys
from pipeline.python.traces.utils import get_metric_set

from bokeh import events

from bokeh.layouts import row, column, widgetbox
from bokeh.models import BoxSelectTool, LassoSelectTool, Spacer, ColumnDataSource, DataRange1d, Select, CustomJS, Div, Plot, Range1d
from bokeh.charts import TimeSeries
from bokeh.models.glyphs import ImageURL
from bokeh.plotting import figure, curdoc, show, output_notebook
from bokeh.palettes import Spectral11
from bokeh.core.properties import value

from bokeh.models import HoverTool

def nix(val, lst):
    return [x for x in lst if x != val]


def scatter_with_hover(source, x, y,
                       fig=None, cols=None, name=None, marker='x',
                       fig_width=500, fig_height=500, tools=['tap', 'box_zoom', 'reset'], **kwargs):
    """
    Plots an interactive scatter plot of `x` vs `y` using bokeh, with automatic
    tooltips showing columns from `df`.
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data to be plotted
    x : str
        Name of the column to use for the x-axis values
    y : str
        Name of the column to use for the y-axis values
    fig : bokeh.plotting.Figure, optional
        Figure on which to plot (if not given then a new figure will be created)
    cols : list of str
        Columns to show in the hover tooltip (default is to show all)
    name : str
        Bokeh series name to give to the scattered data
    marker : str
        Name of marker to use for scatter plot
    **kwargs
        Any further arguments to be passed to fig.scatter
    Returns
    -------
    bokeh.plotting.Figure
        Figure (the same as given, or the newly created figure)
    Example
    -------
    fig = scatter_with_hover(df, 'A', 'B')
    show(fig)
    fig = scatter_with_hover(df, 'A', 'B', cols=['C', 'D', 'E'], marker='x', color='red')
    show(fig)
    Author
    ------
    Robin Wilson <robin@rtwilson.com>
    with thanks to Max Albert for original code example
    """

    # If we haven't been given a Figure obj then create it with default
    # size etc.
    if fig is None:
        fig = figure(width=fig_width, height=fig_height, tools=tools)

    # We're getting data from the given dataframe
    #source = ColumnDataSource(data=df)

    # We need a name so that we can restrict hover tools to just this
    # particular 'series' on the plot. You can specify it (in case it
    # needs to be something specific for other reasons), otherwise
    # we just use 'main'
    if name is None:
        name = 'main'

    # Actually do the scatter plot - the easy bit
    # (other keyword arguments will be passed to this function)
    scatter = fig.scatter(x, y, source=source, name=name, marker=marker, **kwargs)

    # Now we create the hover tool, and make sure it is only active with
    # the series we plotted in the previous line
    hover = HoverTool(names=[name])

    if cols is None:
        # Display *all* columns in the tooltips
        #hover.tooltips = [(c, '@' + c) for c in df.columns]
        hover.tooltips = [(c, '@' + c) for c in source.data.keys()]
    else:
        # Display just the given columns in the tooltips
        hover.tooltips = [(c, '@' + c) for c in cols]

    hover.tooltips.append(('index', '$index'))

    # Finally add/enable the tool
    fig.add_tools(hover)

    return fig, scatter

class RunBase(object):
    def __init__(self, run):
        print run
        self.run = run
        self.traceid = None
        self.pupil_size_thr = None
        self.pupil_dist_thr = None
        self.pupil_max_nblinks = 1

    def set_params(self, paramslist):
        #params = getattr(parservalues, 'trace_info')
        self.traceid = paramslist[0]
        self.pupil_size_thr = paramslist[1]
        self.pupil_dist_thr = paramslist[2]


class FileOptionParser(object):
    def __init__(self):
        self.last_run = None
        self.run_list = []

    def set_info(self, option, opt, value, parser):
        if option.dest=="run":
            print "Creating"
            cls = RunBase
        else:
            assert False

        print value
        self.last_run = cls(value)
        self.run_list.append(self.last_run)
        setattr(parser.values, option.dest, self.last_run)


def extract_options(options):
    fop = FileOptionParser()

    parser = optparse.OptionParser()

    parser.add_option('-D', '--root', action='store', dest='rootdir',
                          default='/nas/volume1/2photon/data',
                          help='data root dir (dir containing all animalids) [default: /nas/volume1/2photon/data, /n/coxfs01/2pdata if --slurm]')
    parser.add_option('-i', '--animalid', action='store', dest='animalid',
                          default='', help='Animal ID')

    # Set specific session/run for current animal:
    parser.add_option('-S', '--session', action='store', dest='session',
                          default='', help='session dir (format: YYYMMDD_ANIMALID')
    parser.add_option('-A', '--acq', action='store', dest='acquisition',
                          default='FOV1', help="acquisition folder (ex: 'FOV1_zoom3x') [default: FOV1]")

    parser.add_option('-R', '--run', dest='run', type='string',
                          action='callback', callback=fop.set_info, help="Supply multiple runs for comparison, all runs used otherwise")

    parser.add_option('-t', '--traces', dest='trace_info', default=[], nargs=1,
                          action='append', help="Corresponding trace ID to specified runs.")


    parser.add_option('--slurm', action='store_true', dest='slurm', default=False, help="set if running as SLURM job on Odyssey")
    parser.add_option('--default', action='store_true', dest='auto', default=False, help="set if want to use all defaults")

    #    parser.add_option('-t', '--trace-id', action='store', dest='trace_id', default='', help="Trace ID for current trace set (created with set_trace_params.py, e.g., traces001, traces020, etc.)")

    # Pupil filtering info:
    parser.add_option('--no-pupil', action="store_false",
                      dest="filter_pupil", default=True, help="Set flag NOT to filter PSTH traces by pupil threshold params")
    parser.add_option('-r', '--rad', action="store",
                      dest="pupil_size_thr", default=25, help="Cut-off for pupil radius, if --pupil set [default: 30]")
    parser.add_option('-d', '--dist', action="store",
                      dest="pupil_dist_thr", default=15, help="Cut-off for pupil distance from start, if --pupil set [default: 5]")
    parser.add_option('-b', '--blinks', action="store",
                      dest="pupil_max_nblinks", default=1, help="Cut-off for N blinks allowed in trial, if --pupil set [default: 1 (i.e., 0 blinks allowed)]")

    (options, args) = parser.parse_args(options)

    for f in fop.run_list:
        run_params = [t for t in options.trace_info if f.run in t][0]
        print run_params
        params_list = [p for p in run_params.split(',') if not p==f.run]
        f.set_params(params_list)
    #print [(f.run, f.traceid, f.pupil_max_nblinks) for f in fop.run_list]

    return options, fop.run_list

#%%
def get_object_transforms(DF):
    '''
    Returns 2 dicts:
        transform_dict = lists all transforms tested in dataset for each transform type
        object_transformations = lists all objects tested on each transform type
    '''

    if 'ori' in DF.keys():
        stimtype = 'grating'
    else:
        stimtype = 'image'

    transform_dict = {'xpos': list(set(DF['xpos'])),
                       'ypos': list(set(DF['ypos'])),
                       'size': list(set((DF['size'])))
                       }
    if stimtype == 'image':
        transform_dict['yrot'] = list(set(DF['yrot']))
        transform_dict['morphlevel'] = list(set(DF['morphlevel']))
    else:
        transform_dict['ori'] = sorted(list(set(DF['ori'])))
        transform_dict['sf'] = sorted(list(set(DF['sf'])))
    trans_types = [t for t in transform_dict.keys() if len(transform_dict[t]) > 1]

    object_transformations = {}
    for trans in trans_types:
        if stimtype == 'image':
            curr_objects = [list(set(DF[DF[trans] == t]['object'])) for t in transform_dict[trans]]
            if len(list(itertools.chain(*curr_objects))) == len(transform_dict[trans]):
                # There should be a one-to-one correspondence between object id and the transformation (i.e., morphs)
                included_objects = list(itertools.chain(*curr_objects))
            else:
                included_objects = list(set(curr_objects[0]).intersection(*curr_objects[1:]))
        else:
            included_objects = transform_dict[trans]
            print included_objects
        object_transformations[trans] = included_objects

    return transform_dict, object_transformations


def get_dataframe_paths(acquisition_dir, trace_info):
    dfpaths = dict()
    for idx, info in enumerate(trace_info):
        dfilepath = None
        rkey = 'run%i' % int(idx+1)

        #runs[rkey]['run'] = info.run
        tdict_path = os.path.join(acquisition_dir, info.run, 'traces', 'traceids_%s.json' % info.run)
        with open(tdict_path, 'r') as f:
            tdict = json.load(f)
        tracename = '%s_%s' % (info.traceid, tdict[info.traceid]['trace_hash'])
        traceid_dir = os.path.join(acquisition_dir, info.run, 'traces', tracename)

        pupil_str = 'pupil_size%i-dist%i-blinks%i' % (float(info.pupil_size_thr), float(info.pupil_dist_thr), int(info.pupil_max_nblinks))
        pupil_dir = [os.path.join(traceid_dir, 'metrics', p) for p in os.listdir(os.path.join(traceid_dir, 'metrics')) if pupil_str in p][0]

        dfilepath = [os.path.join(pupil_dir, f) for f in os.listdir(pupil_dir) if 'roi_stats_' in f][0]
        dfpaths[rkey] = dfilepath

    return dfpaths

#%%
def create_zscore_df(dfpaths):
    all_dfs = []
    for df in dfpaths.values():
        rundf = pd.HDFStore(df, 'r')['/df']
        run_name = os.path.split(df.split('/traces/')[0])[-1]
        print "Compiling zscores for each ROI in run: %s" % run_name
        roi_list = sorted(list(set(rundf['roi'])), key=natural_keys)
        nrois = len(roi_list)
        trial_list = sorted(list(set(rundf['trial'])), key=natural_keys)
        confg_list = sorted(list(set(rundf['config'])), key=natural_keys)

        max_zscores_by_trial = [max([np.float(rundf[((rundf['roi']==roi) & (rundf['trial']==trial))]['zscore'])
                                    for trial in trial_list]) for roi in sorted(roi_list, key=natural_keys)]
        max_zscores_by_stim = [max([np.nanmean(rundf[((rundf['roi']==roi) & (rundf['config']==config))]['zscore'])
                                    for config in confg_list]) for roi in sorted(roi_list, key=natural_keys)]

        curr_df = pd.DataFrame({'roi': roi_list,
                                'max_zscore_trial': np.array(max_zscores_by_trial),
                                'max_zscore_stim': np.array(max_zscores_by_stim),
                                'run': np.tile(run_name, (nrois,))
                                })

        # Concatenate all info for this current trial:
        all_dfs.append(curr_df)

    # Finally, concatenate all trials across all configs for current ROI dataframe:
    DF = pd.concat(all_dfs, axis=0)

    return DF
    

def get_tuning_paths(dfpaths, roi_list):
    
    tuning_paths = dict((roi, []) for roi in roi_list)
    if '/' in roi_list[0]:
        slash = True
    else:
        slash = False
    for roi in roi_list:
        for run in sorted(dfpaths.keys(), key=natural_keys):
            metric_desc = os.path.split(os.path.split(dfpaths[run])[0])[-1] # pupil filter folder name
            traceid_dir = dfpaths[run].split('/metrics')[0]
            figdir = os.path.join(traceid_dir, 'figures', 'tuning', 'raw', 'zscore', metric_desc)
            if slash is True:
                roi_figpath = [os.path.join(figdir, f) for f in os.listdir(figdir) if roi[1:] in f and f.endswith('png')][0]
            else:
                roi_figpath = [os.path.join(figdir, f) for f in os.listdir(figdir) if roi in f and f.endswith('png')][0]
            tuning_paths[roi].append(r'file://'+str(roi_figpath))
            #tuning_paths[roi].append(r'http://localhost:5006/test_scatter/file://' + str(roi_figpath))
    dframe = pd.DataFrame(tuning_paths)
    
    return pd.DataFrame(tuning_paths)




opts = ['-D', '/mnt/odyssey', '-i', 'CE074', '-S', '20180215', '-A', 'FOV1_zoom1x_V1', '-R', 'gratings_phasemod', '-t', 'gratings_phasemod,traces004,30,8', '-R', 'blobs', '-t', 'blobs,traces003,25,15']

options, trace_info = extract_options(opts)
trace_info = list(trace_info)

rootdir = options.rootdir
animalid = options.animalid
session = options.session
acquisition = options.acquisition

acquisition_dir = os.path.join(rootdir, animalid, session, acquisition)

# Get dataframe paths for runs to be compared:
dfpaths = get_dataframe_paths(acquisition_dir, trace_info)

STATS1 = pd.HDFStore(dfpaths['run1'], 'r')['/df']
STATS2 = pd.HDFStore(dfpaths['run2'], 'r')['/df']


sfs = sorted(list(set(STATS1['sf'])))
oris = sorted(list(set(STATS1['ori'])))

nlines = len(sfs)
cpalette=Spectral11[0:nlines]

        
print "Getting DF..."
# Create DF for easy plotting:
animal_dir = os.path.join(rootdir, animalid)
zdf_path = os.path.join(animal_dir, 'stats.pkl')
if not os.path.exists(zdf_path):
    zdf = create_zscore_df(dfpaths)
    with open(zdf_path, 'wb') as f:
        pkl.dump(zdf, f, protocol=pkl.HIGHEST_PROTOCOL)
else:
    with open(zdf_path, 'rb') as f:
        zdf = pkl.load(f)
        
roi_list = sorted(list(set(zdf['roi'])), key=natural_keys)

imgpaths = get_tuning_paths(dfpaths, roi_list)




                # ("data (x,y)", "($x, $y)"),
                # ])


print "Plotting..."

TOOLS="pan,wheel_zoom,box_zoom,tap,lasso_select,reset"

run_list = list(set(zdf['run']))

if  'ori' in STATS1.keys():
    stimtype1 = 'grating'
else:
    stimtype1 = 'image'
if 'ori' in STATS2.keys():
    stimtype2 = 'grating'
else:
    stimtype2 = 'image'

run_list = list(set(zdf['run']))
stat_list = [col for col in zdf if col != 'roi' and col != 'run']

run1 = run_list[0]
run2 = run_list[1] 
stat1 = stat_list[0]
stat2 = stat_list[0]

roi_list = sorted(list(set(zdf['roi'])), key=natural_keys)
nrois = len(roi_list)
roi_name = roi_list[0]

DSET1 = {}; DSET2 = {}
# Get DATASET1 info:
transform_dict1, object_transformations1 = get_object_transforms(STATS1)
DSET1['name'] = run_list[0]
if stimtype1 == 'image':
    DSET1['transforms'] = [t for t in transform_dict1.keys() if len(transform_dict1[t]) > 1]
    DSET1['objects'] = {}
    for trans in DSET1['transforms']:
        if trans=='morphlevel':
            DSET1['objects'][trans] = sorted(transform_dict1[trans])
        else:
            DSET1['objects'][trans] = object_transformations1[trans]
    DSET1['splitter'] = 'object'

else:
    DSET1['objects'] = {}
    DSET1['transforms'] = ['ori']
    for trans in DSET1['transforms']:
        DSET1['objects'][trans] = sorted(list(set(STATS1['sf'])))
    DSET1['splitter'] = 'sf'

# Get DATASET2 info:
transform_dict2, object_transformations2 = get_object_transforms(STATS2)
DSET2['name'] = run_list[1]
if stimtype2 == 'image':
    DSET2['transforms'] = [t for t in transform_dict2.keys() if len(transform_dict2[t]) > 1]
    DSET2['objects'] = {}
    for trans in DSET2['transforms']:
        if trans=='morphlevel':
            DSET2['objects'][trans] = sorted(transform_dict2[trans])
        else:
            DSET2['objects'][trans] = object_transformations2[trans]
    DSET2['splitter'] = 'object'
else:
    DSET2['objects'] = {}
    DSET2['transforms'] = ['ori']
    for trans in DSET2['transforms']:
        DSET2['objects'][trans] = sorted(list(set(STATS2['sf'])))
    DSET2['splitter'] = 'sf'

    
transmenu1 = DSET1['transforms'][0]
transmenu2 = DSET2['transforms'][0]



def load_dset(run1):
    if run1 == DSET1['name']:
        # run1 is actual run1:
        stim_info = DSET1.copy()
        cstats1 = STATS1.copy()
    elif run1 == DSET2['name']:
        stim_info = DSET2.copy()
        cstats1 = STATS2.copy()
    return cstats1, stim_info


def get_joint_data(run1, run2, stat1, stat2):
    zdfsub = pd.DataFrame({'x': np.array(zdf[zdf['run']==run1][stat1]),
                       'y': np.array(zdf[zdf['run']==run2][stat2]),
                       'roi': zdf[zdf['run']==run1]['roi']})
    #source1 = ColumnDataSource(data=zdfsub)
    # zdfsub = {'x': np.array(zdf[zdf['run']==run1][stat1]),
    #         'y': np.array(zdf[zdf['run']==run2][stat2]),
    #         'roi': zdf[zdf['run']==run1]['roi']}
    return zdfsub

def get_all_tuning_curves(roi_name, run1, run2):
    cstats1, stim_info1 = load_dset(run1)
    cstats2, stim_info2 = load_dset(run2)
    source2 = dict((run, {}) for run in [run1, run2])
    
    for run in [run1, run2]:
        if run==run1:
            cstats = cstats1.copy()
            stim_info = stim_info1.copy()
        else:
            cstats = cstats2.copy()
            stim_info = stim_info2.copy()

        # Get tuning curves for AXIS 1:----------------------------
        splitter=stim_info['splitter']
        for trans in stim_info['transforms']:
            resps = {}
            transform_values = sorted(list(set(cstats[trans])))
            print trans, transform_values
            objects = stim_info['objects'][trans]
            if trans == 'morphlevel':
                obj_str = 'moprh_yrot0'
                grouped = cstats[((cstats['roi']==roi_name))].groupby('morphlevel')['zscore']
                resps.update({'%s_%s' % (run, obj_str): np.array(grouped.mean())})
                # Get sem bars:
                yerr = np.array(grouped.aggregate(lambda x: np.std(x, ddof=1)/np.sqrt(x.count())))
                y_err_x = []
                y_err_y = []
                for px, py, err in zip(sorted(transform_values), np.array(grouped.mean()), yerr):
                    y_err_x.append((px, px))
                    y_err_y.append((py - err, py + err))
                # Get sem:
                resps.update( {'%s_%s_yerr_x' % (run, obj_str): y_err_x})    
                resps.update( {'%s_%s_yerr_y' % (run, obj_str): y_err_y})   
            else:
                for obj in objects:
                    # Get mean:
                    grouped = cstats[((cstats['roi']==roi_name) & (cstats[splitter]==obj))].groupby(trans)['zscore']
                    resps.update({ '%s_%s' % (run, str(obj)): np.array(grouped.mean())})
                    # Get sem:
                    yerr = np.array(grouped.aggregate(lambda x: np.std(x, ddof=1)/np.sqrt(x.count())))
                    y_err_x = []
                    y_err_y = []
                    for px, py, err in zip(sorted(transform_values), np.array(grouped.mean()), yerr):
                        y_err_x.append((px, px))
                        y_err_y.append((py - err, py + err))

                    resps.update( {'%s_%s_yerr_x' % (run, str(obj)): y_err_x})    
                    resps.update( {'%s_%s_yerr_y' % (run, str(obj)): y_err_y})    
            resps.update({'%s_x' % run: sorted(transform_values)})   
            
            cds = ColumnDataSource(data=resps)
            source2[run][trans] = cds
                
    print "Got updated tuning curve data for Ax1, Ax2"
    
    return source2

    
    
def get_tuning_data(roi_name, run1, run2, transmenu1, transmenu2):
    cstats1, stim_info1 = load_dset(run1)
    cstats2, stim_info2 = load_dset(run2)
    
    # Get tuning curve for AXIS 1:----------------------------
    resps = dict()
    #print "TRANS1:", transmenu1
    splitter=stim_info1['splitter']
    transforms = sorted(list(set(cstats1[transmenu1])))
    ids1 = stim_info1['objects'][transmenu1]
    for obj in ids1:
        print obj
        # Get mean:
        grouped = cstats1[((cstats1['roi']==roi_name) & (cstats1[splitter]==obj))].groupby(transmenu1)['zscore']
        resps.update({ str(obj): np.array(grouped.mean())})
        
        # Get sem:
        yerr = np.array(grouped.aggregate(lambda x: np.std(x, ddof=1)/np.sqrt(x.count())))
        y_err_x = []
        y_err_y = []
        for px, py, err in zip(sorted(transforms), np.array(grouped.mean()), yerr):
            y_err_x.append((px, px))
            y_err_y.append((py - err, py + err))

        resps.update( {'%s_yerr_x' % str(obj): y_err_x})    
        resps.update( {'%s_yerr_y' % str(obj): y_err_y})    
    
    resps.update({'x': sorted(transforms)})   
    source2a = ColumnDataSource(data=resps)

    # Get tuning curve for AXIS 2:----------------------------
    splitter=stim_info2['splitter']
    resps = dict()
    transforms = sorted(list(set(cstats2[transmenu2])))
    ids2 = stim_info2['objects'][transmenu2]
    #print splitter, ids2
    if transmenu2 == 'morphlevel':
        obj_str = 'moprh_yrot0'
        grouped = cstats2[((cstats2['roi']==roi_name))].groupby('morphlevel')['zscore']
        resps.update({obj_str: np.array(grouped.mean())})
        # Get sem bars:
        yerr = np.array(grouped.aggregate(lambda x: np.std(x, ddof=1)/np.sqrt(x.count())))
        y_err_x = []
        y_err_y = []
        for px, py, err in zip(sorted(transforms), np.array(grouped.mean()), yerr):
            y_err_x.append((px, px))
            y_err_y.append((py - err, py + err))

        # Get sem:
        resps.update( {'%s_yerr_x' % obj_str: y_err_x})    
        resps.update( {'%s_yerr_y' % obj_str: y_err_y})   
            
    else:                 
        for obj in ids2:
            grouped = cstats2[((cstats2['roi']==roi_name) & (cstats2[splitter]==obj))].groupby(transmenu2)['zscore']
            # Get means:
            resps.update({ str(obj): np.array(grouped.mean())})
            
            # Get sem bars:
            yerr = np.array(grouped.aggregate(lambda x: np.std(x, ddof=1)/np.sqrt(x.count())))
            y_err_x = []
            y_err_y = []
            for px, py, err in zip(sorted(transforms), np.array(grouped.mean()), yerr):
                y_err_x.append((px, px))
                y_err_y.append((py - err, py + err))
                
            # Get sem:
            resps.update( {'%s_yerr_x' % str(obj): y_err_x})    
            resps.update( {'%s_yerr_y' % str(obj): y_err_y})    
            
    resps.update({'x': sorted(transforms)})  
    source2b = ColumnDataSource(data=resps)
    
    print "Got updated tuning curve data for Ax1, Ax2"

    return source2a, ids1, source2b, ids2


def errorbar(fig, x=[], y=[], name=None, source=None, yerr_x=None, yerr_y=None, xerr_x=None, xerr_y=None,
             color='red', line_width=3, point_kwargs={}, error_kwargs={}):
    
    if source is None:
        #h1 = fig.circle(x, y, color=color, **point_kwargs)
        h1 = fig.line(x, y, legend=value(name), line_color=color, line_width=line_width )

        if xerr_x:
            x_err_x = []
            x_err_y = []
            for px, py, err in zip(x, y, xerr):
                x_err_x.append((px - err, px + err))
                x_err_y.append((py, py))
            h2 = fig.multi_line(x_err_x, x_err_y, color=color, line_width=line_width, **error_kwargs)

        if yerr_x:
            y_err_x = []
            y_err_y = []
            for px, py, err in zip(x, y, yerr):
                y_err_x.append((px, px))
                y_err_y.append((py - err, py + err))
            h2 = fig.multi_line(y_err_x, y_err_y, color=color, **error_kwargs)
    else:
        
        #h1 = fig.circle(x=x, y=y, source=source, color=color, **point_kwargs)
        h1 = fig.line(x=x, y=y, legend=value(name), source=source, line_color=color, line_width=line_width)
        
        if xerr_x:
            h2 = fig.multi_line(xs=x_err_x, ys=x_err_y, source=source, line_color=color, line_width=line_width, **error_kwargs)

        if yerr_x:
            h2 = fig.multi_line(xs=yerr_x, ys=yerr_y, source=source, line_color=color, line_width=line_width, **error_kwargs)

            
    return h1, h2



#output_notebook()


#def modify_doc(doc):

# set up widgets----------------------------------------------------
transmenu1_select = Select(value=DSET1['transforms'][0], title='transforms1', options=DSET1['transforms'])
transmenu2_select = Select(value=DSET2['transforms'][0], title='transforms2', options=DSET2['transforms'])

run1_select = Select(value=run1, title='X-dim', options=run_list)
run2_select = Select(value=run2, title='Y-dim', options=run_list)

stat1_select = Select(value=stat1, title='X-Metric', options=stat_list)
stat2_select = Select(value=stat2, title='X-Metric', options=stat_list)

# Get data sources:
source1 = ColumnDataSource(data=dict(index=[], roi=[], x=[], y=[]))
data = get_joint_data(run1, run2, stat1, stat2)
source1.data = source1.from_df(data[['roi', 'x', 'y']])

# Get Tuning curve data-------------------------------------------------------------
#source2a, ids1, source2b, ids2 = get_tuning_data(roi_name, run1, run2, transmenu1, transmenu2)
source2 = get_all_tuning_curves(roi_name, run1, run2)


#def update():
# plotting --------------------------------------------------------
# create scatter plot with hover:-------------------------------------------------------------
p, scatter = scatter_with_hover(source1, x='x', y='y', size=10, color="#3A5785", alpha=0.6, tools=TOOLS,
                      fig_width=700, fig_height=700, marker='o')
ds0 = scatter.data_source


# plot tuning curve 1:
transform_dict = dict((run, []) for run in run_list)
tplot1 = figure(title='tuning 1', plot_width=400, plot_height=400, tools=TOOLS)
hover = HoverTool(
            tooltips=[
                ("index", "$index"),
                ("transform_value", "$x"),
                ("zscore", "$y")
            ])

# Finally add/enable the tool
tplot1.add_tools(hover)
    
lines1 = dict((run, dict()) for run in [run1, run2])
for run in source2.keys():
    for trans in source2[run].keys():
        lines1[run][trans] = []
        line_names = [k.split('%s_' % run)[-1] for k in source2[run][trans].data.keys() if 'yerr' not in k]
        line_names = sorted([n for n in line_names if not n=='x'], key=natural_keys)
        for i, line in enumerate(line_names):
            curr_cds = source2[run][trans]
            h1, h2 = errorbar(tplot1, x='%s_x' % run, y='%s_%s' % (run, line), name=line, source=curr_cds, yerr_x='%s_%s_yerr_x' % (run, line), yerr_y='%s_%s_yerr_y' % (run, line), color=cpalette[i], line_width=2)
            lines1[run][trans].append((h1, h2))
            
            lines1[run][trans][i][0].visible = (run==run1) & (trans==transmenu1)
            lines1[run][trans][i][1].visible = (run==run1) & (trans==transmenu1)
            
            transform_dict[run].append(trans)
    transform_dict[run] = list(set(transform_dict[run]))


# plot tuning curve 2:          
tplot2 = figure(title='tuning 2', plot_width=400, plot_height=400, tools=TOOLS)
lines2 = dict((run, dict()) for run in [run1, run2])
for run in source2.keys():
    for trans in source2[run].keys():
        lines2[run][trans] = []
        line_names = [k.split('%s_' % run)[-1] for k in source2[run][trans].data.keys() if 'yerr' not in k]
        line_names = sorted([n for n in line_names if not n=='x'], key=natural_keys)
        for i, line in enumerate(line_names):
            curr_cds = source2[run][trans]
            h1, h2 = errorbar(tplot2, x='%s_x' % run, y='%s_%s' % (run, line), name=line, source=curr_cds, yerr_x='%s_%s_yerr_x' % (run, line), yerr_y='%s_%s_yerr_y' % (run, line), color=cpalette[i], line_width=2)
            lines2[run][trans].append((h1, h2))
            
            lines2[run][trans][i][0].visible = (run==run2) & (trans==transmenu2)
            lines2[run][trans][i][1].visible = (run==run2) & (trans==transmenu2)
hover = HoverTool(
            tooltips=[
                ("index", "$index"),
                ("transform_value", "$x"),
                ("zscore", "$y")
            ])
# Finally add/enable the tool
tplot2.add_tools(hover)
    
    
# plot histograms:-------------------------------------------------------------
d1 = np.array(zdf[zdf['run']==run1_select.value][stat1_select.value]) #source.data['d1']
d2 = np.array(zdf[zdf['run']==run2_select.value][stat2_select.value]) #source.data['d2']

# create the horizontal histogram
hhist, hedges = np.histogram(d1, bins=20)
hzeros = np.zeros(len(hedges)-1)
hmax = max(hhist)*1.1
LINE_ARGS = dict(color="#3A5785", line_color=None)

ph = figure(toolbar_location=None, plot_width=p.plot_width, plot_height=100, x_range=p.x_range,
            y_range=(-hmax, hmax), min_border=10, min_border_left=50, y_axis_location="right")
ph.xgrid.grid_line_color = None
ph.yaxis.major_label_orientation = np.pi/4
ph.background_fill_color = "#fafafa"

ph.quad(bottom=0, left=hedges[:-1], right=hedges[1:], top=hhist, color="white", line_color="#3A5785")
hh1 = ph.quad(bottom=0, left=hedges[:-1], right=hedges[1:], top=hzeros, alpha=0.5, **LINE_ARGS)
hh2 = ph.quad(bottom=0, left=hedges[:-1], right=hedges[1:], top=hzeros, alpha=0.1, **LINE_ARGS)

# create the vertical histogram
vhist, vedges = np.histogram(d2, bins=20)
vzeros = np.zeros(len(vedges)-1)
vmax = max(vhist)*1.1

pv = figure(toolbar_location=None, plot_width=100, plot_height=p.plot_height, x_range=(-vmax, vmax),
            y_range=p.y_range, min_border=10, y_axis_location="right")
pv.ygrid.grid_line_color = None
pv.xaxis.major_label_orientation = np.pi/4
pv.background_fill_color = "#fafafa"

pv.quad(left=0, bottom=vedges[:-1], top=vedges[1:], right=vhist, color="white", line_color="#3A5785")
vh1 = pv.quad(left=0, bottom=vedges[:-1], top=vedges[1:], right=vzeros, alpha=0.5, **LINE_ARGS)
vh2 = pv.quad(left=0, bottom=vedges[:-1], top=vedges[1:], right=vzeros, alpha=0.1, **LINE_ARGS)




def select_roi(attr, old, new):
    global roi_name
    roi_name = source1.data['roi'][new['1d']['indices'][0]]
    print("TapTool callback executed on Roi {}".format(roi_name))
    update_menu(attr, old, new)

    
# set up callbacks----------------------------------------------------
def update_menu(attr, old, new):
    global roi_name
    
    run1, run2 = run1_select.value, run2_select.value
    #print "UPDATE:", run1, run2
    stat1, stat2 = stat1_select.value, stat2_select.value
    #print "UPDATE:", stat1, stat2

    transmenu1, transmenu2 = transmenu1_select.value, transmenu2_select.value
    #print "UPDATE:", transmenu1, transmenu2   

    # Get joint ROI data for selcted runs:
    src1 = get_joint_data(run1, run2, stat1, stat2)
    #print "SOURCE1:", run1, run2
    ds0.data = ColumnDataSource(src1).data

    # Check whether an ROI has been selected:
    if isinstance(old, dict):
        roi_name = source1.data['roi'][new['1d']['indices'][0]]
        print("TapTool callback executed on Roi {}".format(roi_name))
        
    # Update tuning curve data:
    src2 = get_all_tuning_curves(roi_name, run1, run2)
        

    # Update tuning curve menus, respectively:
    if run1 == DSET1['name']:
        transmenu1_select.options = DSET1['transforms']
    elif run1 == DSET2['name']:
        transmenu1_select.options = DSET2['transforms']

    if str(run2) == str(DSET1['name']):
        transmenu2_select.options = DSET1['transforms']
    elif str(run2) == str(DSET2['name']):
        transmenu2_select.options = DSET2['transforms']

    if transmenu1_select.value not in transmenu1_select.options:
        print "SWITCHING 1:"
        transmenu1_select.value = transmenu1_select.options[0]
    if transmenu2_select.value not in transmenu2_select.options:
        print "SWITCHING 2:"
        transmenu2_select.value = transmenu2_select.options[0]

    transmenu1, transmenu2 = transmenu1_select.value, transmenu2_select.value

    # print "Run1: %s, %s" % (run1, transmenu1)
    # print "Run2: %s, %s" % (run2, transmenu2)
    for run in run_list:
        for trans in transform_dict[run]:
            for i in range(len(lines1[run][trans])):
                lines1[run][trans][i][0].data_source.data = src2[run][trans].data
                lines1[run][trans][i][1].data_source.data = src2[run][trans].data
                
                lines1[run][trans][i][0].visible = (run==run1) & (trans==transmenu1)
                lines1[run][trans][i][1].visible = (run==run1) & (trans==transmenu1)
                
            for i in range(len(lines2[run][trans])):
                lines2[run][trans][i][0].data_source.data = src2[run][trans].data
                lines2[run][trans][i][1].data_source.data = src2[run][trans].data
                
                lines2[run][trans][i][0].visible = (run==run2) & (trans==transmenu2)
                lines2[run][trans][i][1].visible = (run==run2) & (trans==transmenu2)
                
        
scatter.data_source.on_change('selected',select_roi)        
for r in [run1_select, run2_select]:
    r.on_change('value', update_menu)
for st in [stat1_select, stat2_select]:
    st.on_change('value', update_menu)
for tr in [transmenu1_select, transmenu2_select]:
    tr.on_change('value', update_menu)


# set up layout

main_controls =  widgetbox([run1_select, stat1_select, run2_select, stat2_select], width=200) #column(ticker1, ticker2, stats)

layout1 = row(column(row(p, pv), row(ph, Spacer(width=200, height=200))))

layout = column(main_controls, row(layout1, column(tplot1, transmenu1_select), column(tplot2, transmenu2_select)))

curdoc().add_root(layout) #add_root(layout)

curdoc().title = "ROI selectivity"
