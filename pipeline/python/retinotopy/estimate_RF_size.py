
# coding: utf-8

# In[1]:


import matplotlib as mpl
mpl.use('Agg')
import os
import h5py
import json
import re
import sys
import datetime
import optparse
import pprint
import cPickle as pkl
import tifffile as tf
import pylab as pl
import numpy as np
from scipy import ndimage
import cv2
import glob
from scipy.optimize import curve_fit
import seaborn as sns
from pipeline.python.retinotopy import visualize_rois as vis
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1 import make_axes_locatable


from pipeline.python.utils import natural_keys, label_figure, replace_root
from pipeline.python.retinotopy import utils as util
from pipeline.python.retinotopy import do_retinotopy_analysis as ra

pp = pprint.PrettyPrinter(indent=4)

#%%


#-----------------------------------------------------
#           FUNCTIONS FOR ROI trace extraction:
#-----------------------------------------------------


#def get_retino_traces(RID, retinoid_dir, mwinfo, runinfo, tiff_fpaths, create_new=False):
#
#    # Set output dir:
#    output_dir = os.path.join(retinoid_dir,'traces')
#    if not os.path.exists(output_dir):
#            os.makedirs(output_dir)
#    print output_dir
#
#    avg_trace_fpath = os.path.join(output_dir, 'averaged_roi_traces.pkl')
#    if os.path.exists(avg_trace_fpath) and create_new is False:
#        with open(avg_trace_fpath, 'rb') as f:
#            traces = pkl.load(f)
#    else:
#        acquisition_dir = os.path.split(retinoid_dir.split('/retino_analysis')[0])[0]
#        session_dir = os.path.split(acquisition_dir)[0]
#        masks = load_roi_masks(session_dir, RID)
#
#        # Block reduce masks to match downsampled tiffs:
#        dsample = RID['PARAMS']['downsample_factor']
#        if dsample is None:
#            dsample = 1
#        else:
#            dsample = int(dsample)
#        masks = ra.block_mean_stack(masks, dsample, along_axis=0)
#        print masks.shape
#        
#        # Combine reps of the same condition.
#        # Reshape masks and averaged tiff stack, extract ROI traces
#        traces = average_retino_traces(RID, mwinfo, runinfo, tiff_fpaths, masks, output_dir=output_dir)
#        
#    return traces
#
#
#def average_retino_traces(RID, mwinfo, runinfo, tiff_fpaths, masks, output_dir='/tmp'):
#    
#    rep_list = [(k, v['stimuli']['stimulus']) for k,v in mwinfo.items()]
#    unique_conditions = np.unique([rep[1] for rep in rep_list])
#    conditions = dict((cond, [int(run) for run,config in rep_list if config==cond]) for cond in unique_conditions)
#    print("CONDITIONS:", conditions)
#    
#    rtraces = {}
#
#    # First check if extracted_traces file exists:
#    traces_fpath = glob.glob(os.path.join(output_dir, 'extracted_traces*.h5'))
#    extract_from_stack = False
#    try:
#        assert len(traces_fpath) == 1, "*** unable to find unique extracted_traces.h5 in dir:\n%s" % output_dir
#        traces_fpath = traces_fpath[0]
#        print("... Loading extracted traces: %s" % traces_fpath)
#        extracted = h5py.File(traces_fpath, 'r')
#        print("Found %i files of extracted traces." % len(extracted.keys()))
#        for curr_cond in conditions.keys():
#            curr_tstack = np.array([extracted['File%03d' % int(rep)]['corrected'][:] for rep in conditions[curr_cond]])
#            print("... cond: %s (stack size: %s)" % (curr_cond, str(curr_tstack.shape)))
#            rtraces[curr_cond] = np.mean(curr_tstack, axis=0)
#            print rtraces[curr_cond].shape
#
#    except Exception as e:
#        print e
#        print("Extracting ROI traces from tiff stacks...")
#        extract_from_stack = True
#    finally:
#        extracted.close()
#
#    if extract_from_stack: 
#        cstack = get_averaged_condition_stack(conditions, tiff_fpaths, RID)
#
#        for curr_cond in cstack.keys():
#            roi_traces = apply_masks_to_tifs(masks, cstack[curr_cond])
#            #print roi_traces.shape
#            rtraces[curr_cond] = roi_traces
#      
# 
#    # Smooth roi traces:
#    traceinfo = dict((cond, dict()) for cond in rtraces.keys())
#    for curr_cond in rtraces.keys():
#        # get some info from paradigm and run file
#        stack_info = dict()
#        stack_info['stimulus'] = curr_cond
#        stack_info['stimfreq'] = np.unique([v['stimuli']['scale'] for k,v in mwinfo.items() if v['stimuli']['stimulus']==curr_cond])[0]
#        stack_info['frame_rate'] = runinfo['frame_rate']
#        stack_info['n_reps'] = len(conditions[curr_cond])
#        pp.pprint(stack_info)
#
#        traces = ra.process_array(rtraces[curr_cond], RID, stack_info)
#
#        traceinfo[curr_cond]['traces'] = traces
#        traceinfo[curr_cond]['info'] = stack_info
#
#    traces = {'mwinfo': mwinfo,
#             'conditions': conditions,
#             'source_tifs': tiff_fpaths,
#             'RETINOID': RID,
#              'masks': masks,
#              'traces': traceinfo
#             }
#    avg_trace_fpath = os.path.join(output_dir, 'averaged_roi_traces.pkl')
#    with open(avg_trace_fpath, 'wb') as f: pkl.dump(traces, f, protocol=pkl.HIGHEST_PROTOCOL)
#    print "Saved processed ROI traces to:\n%s\n" % avg_trace_fpath
#    return traces
#


#
#def load_roi_masks(session_dir, RID):
#    assert RID['PARAMS']['roi_type'] != 'pixels', "ROI type for analysis should not be pixels. This is: %s" % RID['PARAMS']['roi_type']
#    print 'Getting masks'
#    # Load ROI set specified in analysis param set:
#    roidict_fpath = glob.glob(os.path.join(session_dir, 'ROIs', 'rids_*.json'))[0]
#    with open(roidict_fpath, 'r') as f: roidict = json.load(f)
#
#    roi_dir = roidict[RID['PARAMS']['roi_id']]['DST']
#    session = os.path.split(session_dir)[-1]
#    animalid = os.path.split(os.path.split(session_dir)[0])[-1]
#    rootdir = os.path.split(os.path.split(session_dir)[0])[0]
#    
#    if rootdir not in roi_dir:
#        roi_dir = replace_root(roi_dir, rootdir, animalid, session)
#    mask_fpath = os.path.join(roi_dir, 'masks.hdf5')
#    maskfile = h5py.File(mask_fpath,  'r')#read
#    masks = maskfile[maskfile.keys()[0]]['masks']['Slice01']
#    print masks.shape
#    
#    return masks
#
#
#def get_averaged_condition_stack(conditions, tiff_fpaths, RID):
#    cstack = {}
#    condition_list = conditions.keys()
#    curr_cond = 'right'
#    for curr_cond in condition_list:
#        curr_tiff_fpaths = [tiff_fpaths[int(i)-1] for i in conditions[curr_cond]]
#        for tidx, tiff_fpath in enumerate(curr_tiff_fpaths):    
#            print "Loading: ", tiff_fpath
#            tiff_stack = ra.get_processed_stack(tiff_fpath, RID)
#            szx, szy, nframes = tiff_stack.shape
#            #print szx, szy, nframes
#            if tidx == 0:
#                # initiate stack
#                stack = np.empty(tiff_stack.shape, dtype=tiff_stack.dtype)
#            stack =  stack + tiff_stack
#
#        # Get stack average:
#        cstack[curr_cond] = stack / len(curr_tiff_fpaths)
#
#    return cstack
#
#
#
#def apply_masks_to_tifs(masks, stack):
#    szx, szy, nframes = stack.shape
#    nrois = masks.shape[0]
#    maskr = np.reshape(masks, (nrois, szx*szy))
#    stackr = np.reshape(stack, (szx*szy, nframes))
#    #print "masks:", maskr.shape
#    #print "stack:", stackr.shape
#    roi_traces = np.dot(maskr, stackr)
#    return roi_traces
#
#
#%%


# ## Load retinotopy run source

# In[5]:

def gaus(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))


class ActivityInfo:
    def __init__(self, condition, roi_trace):
        self.name = condition
        self.activity_trace = roi_trace
        
    def parse_cycles(self, framestart, stimframes_incl):
        ncycles = len(framestart)
        parsed_traces = np.zeros((ncycles,stimframes_incl))
        #print parsed_traces.shape
        #print self.activity_trace.shape
        for cycle in range(0,ncycles):
            #print framestart[cycle]
            ixs_to_fill = framestart[cycle] + stimframes_incl
            if ixs_to_fill > len(self.activity_trace):
                self.activity_trace = np.pad(self.activity_trace, ((0, ixs_to_fill - len(self.activity_trace))), mode='constant', constant_values=0)
            parsed_traces[cycle,:]=self.activity_trace[framestart[cycle]:framestart[cycle]+stimframes_incl]
        self.parsed_traces = parsed_traces
        

    def fit_gaussian_to_trace(self):
        x0 = np.arange(0,self.parsed_traces.shape[-1])
        y = np.mean(self.parsed_traces, axis=0) #-y0.min()
        try:
            center_start = np.argmax(y)-(np.argmax(y)/1)
            center_end = np.argmax(y)+(np.argmax(y)/1)*2
            if center_start < 0:
                center_start = 0
            if center_end > len(y):
                center_end = len(y)
            centered = y[np.arange(center_start, center_end)]
            popt, pcov = curve_fit(gaus, x0, y, p0=(y[np.argmax(y)], np.argmax(y), 1))

            y_fit = gaus(x0,*popt)
            #print(popt)

            ss_res = np.sum((y - y_fit)**2)
            ss_tot = np.sum((y-np.mean(y))**2)
            r_squared = 1 - (ss_res / ss_tot)
            #assert r_squared >= 0, "R2 is negative..."
            #print("R2:", r_squared)
            A, x0, sigma = popt
            
        except RuntimeError:
            print("Error - curve_fit failed")
            A = None; x0 = None; sigma = None;
            r_squared = 0
            y_fit = None
        self.average_y = y
        self.fit_results = {'r2': r_squared,
                           'A': A,
                           'x0': x0,
                           'sigma': sigma,
                           'y_fit': y_fit}
        
    def estimate_RF_size(self, frames_per_degree, fitness_thr=0.5, size_thr=0.1):

        if self.fit_results['r2'] > fitness_thr:
            
            y = self.average_y
            y_fit = self.fit_results['y_fit']
            
            fit_norm = y_fit/np.max(y_fit)

            border_start = np.where(fit_norm>=size_thr)[0]
            if len(border_start)==0:
                border_start = 0
            else:
                border_start = border_start[0]
            border_end = np.where(fit_norm[border_start:]<=size_thr)[0]
            if len(border_end) == 0:
                border_end = len(fit_norm)-1
            else:
                border_end = border_end[0]
            #print(border_start, border_end) #rf_size_frames)

            # extrapolate around peak, in case edge:
            peak = np.argmax(y)
            tail2 = border_end - peak
            tail1 = peak - border_start
            if tail1 < tail2:
                border_edge1 = peak - tail2
                border_edge2 = peak + tail2
            elif tail2 < tail1:
                border_edge2 = peak + tail1
                border_edge1 = peak - tail1
            else:
                border_edge1 = peak - tail1
                border_edge2 = peak + tail2

            rf_size_frames = border_edge2 - border_edge1
            #print rf_size_frames
        else:
            rf_size_frames = 0
            peak = 0
            border_start = 0; border_end = 0
            
        self.peak_ix = peak
        self.borders = (border_start, border_end)
        self.RF_frames = rf_size_frames
        self.RF_degrees = rf_size_frames / frames_per_degree

        
class RetinoROI:
    def __init__(self, roi_ix):
        self.name = 'roi%06d' % (roi_ix+1)
        self.idx = roi_ix
        self.conditions = []
    
    def parse(self, condition, roi_trace, framestart, stimframes_incl):
        cond_ix = [ci for ci, cond in enumerate(self.conditions) if cond.name==condition]
        if len(cond_ix) == 0:
            self.conditions.append(ActivityInfo(condition, roi_trace))
            cond_ix = -1
        else:
            cond_ix = cond_ix[0]
            
        self.conditions[cond_ix].parse_cycles(framestart, stimframes_incl)
    
    def fit(self, condition, frames_per_degree, fitness_thr=0.5, size_thr=0.1):
        cond_ix = [ci for ci, cond in enumerate(self.conditions) if cond.name==condition]

        cond_ix = cond_ix[0]
        self.conditions[cond_ix].fit_gaussian_to_trace()
        self.conditions[cond_ix].estimate_RF_size(frames_per_degree, 
                                   fitness_thr=fitness_thr, 
                                   size_thr=size_thr)
    
    def print_info(self):
        for cond in self.conditions:
            print "Name: %s | RF size (deg): %i" % (self.name, cond.RF_degrees)



def get_RF_size_estimates(acquisition_dir, fitness_thr=0.4, size_thr=0.1, analysis_id=None, retino_run='retino*', slurm=False):
    print "*** GETTING ESTIMATES ***"
    run_dir = glob.glob(os.path.join(acquisition_dir, '%s' % retino_run))[0]
    run = os.path.split(run_dir)[1]
    
    session_dir = os.path.split(acquisition_dir)[0]
    acquisition = os.path.split(acquisition_dir)[1]
    session = os.path.split(session_dir)[1]
    animalid = os.path.split(os.path.split(session_dir)[0])[1]
    rootdir = os.path.split(os.path.split(session_dir)[0])[0]
    print "SESSION:", session, "ACQ:", acquisition, analysis_id
    print "RUN: %s, ANALYSIS ID: %s" % (run, analysis_id)
    if analysis_id is None:
        analysis_id = 'analysis'
        retino_roi_analysis = glob.glob(os.path.join(rootdir, animalid, session, acquisition, 'retino*', 'retino_analysis', '%s*' % analysis_id, 'visualization'))[0]
        retinoid_dir = os.path.split(retino_roi_analysis)[0]

    else:
       retinoid_dir = glob.glob(os.path.join(rootdir, animalid, session, acquisition, '%s' % run, 'retino_analysis', '%s*' % analysis_id))[0]
    
    
    retinoids_fpath = glob.glob(os.path.join(acquisition_dir, '%s' % run, 'retino_analysis', 'analysisids_*.json'))[0]
    retinoid = os.path.split(retinoid_dir)[1]
    
    with open(retinoids_fpath, 'r') as f: rids = json.load(f)
    RID = rids[retinoid.split('_')[0]]
    pp.pprint(RID)
    
    # =============================================================================
    # Get meta info for current run and source tiffs using analysis-ID params:
    # =============================================================================
    #analysis_hash = RID['analysis_hash']
    
    tiff_dir = RID['SRC']
    if rootdir not in tiff_dir:
        tiff_dir = replace_root(tiff_dir, rootdir, animalid, session)
        
    tiff_fpaths = sorted([os.path.join(tiff_dir, t) for t in os.listdir(tiff_dir) if t.endswith('tif')], key=natural_keys)
    #print "Found %i tiffs in dir %s.\nExtracting analysis with ROI set %s." % (len(tiff_files), tiff_dir, roi_name)
    
    # Get associated RUN info:
    print('... loading scan info')
    runmeta_path = os.path.join(run_dir, '%s.json' % run)
    with open(runmeta_path, 'r') as r:
        runinfo = json.load(r)
 
    
    # =============================================================================
    # Load paradigm info
    # =============================================================================
    print('... loading paradigm info')
    print os.listdir(os.path.join(run_dir, 'paradigm', 'files'))
    paradigm_fpath = glob.glob(os.path.join(run_dir, 'paradigm', 'files', 'parsed_trials*.json'))[0]
    print paradigm_fpath
    with open(paradigm_fpath, 'r') as r: mwinfo = json.load(r)
    # pp.pprint(mwinfo)
    
    # =============================================================================
    # Process/Load ROI traces for averaged retino conditions:
    # =============================================================================
    
    create_new = False
    print('... processing ROI traces')
    print "RETINOID dir:", retinoid_dir
    traces = util.get_retino_traces(RID, retinoid_dir, mwinfo, runinfo, tiff_fpaths, create_new=create_new)
    
    #print "Conditions:",  traces['traces'].keys()
    traceinfo =  traces['traces']
    nrois = traces['masks'].shape[0]
    
    
    # =============================================================================
    # Get screen info from epi runs
    # =============================================================================
    if slurm: 
        interactive=False
    else:
        interactive=True
    screen_info = util.get_screen_info(animalid, session, fov=acquisition.split('_')[0], interactive=True, rootdir=rootdir)
    el_degrees_per_cycle = screen_info['elevation']
    az_degrees_per_cycle = screen_info['azimuth']

    
    frate = runinfo['frame_rate']
    nframes = runinfo['nvolumes']
    
    
    # #### Average all cycles to fit curve
    # ### Fit gaussian to average response to cycle:
    
    
    ROIs = []
    for curr_roi in range(nrois):
        roi = RetinoROI(curr_roi)
    
        for curr_cond in traceinfo.keys():
            #print curr_cond
            stack_info = traceinfo[curr_cond]['info']
            roi_traces = traceinfo[curr_cond]['traces']
    
            # Get frame info for current cond
            stimfreq = stack_info['stimfreq']
            stimperiod = 1.0/stimfreq
    
            ncycles = int(round((nframes/frate) / stimperiod))
    
            stimframes_start_int= stimperiod*frate
            stimframes_incl = int(np.floor(stimperiod*frate))
            framestart = np.round(np.arange(0,stimframes_start_int*ncycles,stimframes_start_int)).astype('int')
    
            # Get screen info for current cond:
            nframes_per_cycle = stimframes_incl
            #cycles_per_sec = stack_info['stimfreq']
    
            cycles_per_degree = az_degrees_per_cycle
#            if curr_cond == 'right' or curr_cond == 'left':
#                cycles_per_degree = az_degrees_per_cycle
#            else:
#                cycles_per_degree = el_degrees_per_cycle
    
            frames_per_degree = nframes_per_cycle / cycles_per_degree
            #print frames_per_degree
    
            roi.parse(curr_cond, roi_traces[curr_roi, :], framestart, stimframes_incl)
            roi.fit(curr_cond, frames_per_degree, fitness_thr=fitness_thr, size_thr=size_thr)
            
        ROIs.append(roi)
        
        
    return ROIs, retinoid, screen_info



#%% PLOTTING:
def plot_ROI_positions(acquisition_dir, run, retinoid, screen_info=None, roi_size=100, ax=None, interactive=True):
    
    dfpaths = vis.get_retino_datafile_paths(acquisition_dir, run, retinoid.split('_')[0])
    
    # Convert data to dataframe:
    dataframes = vis.get_metricdf(dfpaths)
    zdf = vis.assign_mag_ratios(dataframes, run, stat_type='mean', metric_type='magratio') #=stat_type, run2_stat_type=stat_type)
    print zdf.head()

    # Get path info:
    acquisition = os.path.split(acquisition_dir)[-1]
    session_dir = os.path.split(acquisition_dir)[0]
    session = os.path.split(session_dir)[-1]
    animalid = os.path.split( os.path.split(session_dir)[0])[-1]
    rootdir = session_dir.split('/%s' % animalid)[0]

    # Get screen info:
    if screen_info is None: 
        screen_info = vis.get_screen_info(animalid, session, fov=acquisition.split('_')[0], interactive=True, rootdir=rootdir)

    if len(screen_info.keys()) == 0:
        screen_width = 117.56 #81.28
        screen_height = 67.32 #45.77
        screen_resolution = [1024, 768] #[1920, 1024]
    else:
        screen_width = screen_info['azimuth']
        screen_height = screen_info['elevation']
        screen_resolution = screen_info['resolution']

    #retino_info = vis.get_retino_info(width=screen_width, height=screen_height, resolution=screen_resolution, azimuth='right', elevation='top')
    retino_info = util.get_retino_info(animalid, session, fov=acquisition, interactive=interactive, rootdir=rootdir, \
                        azimuth='right', elevation='top') #,
#                        leftedge=None, rightedge=None, 
#                        bottomedge=None, topedge=None)
#

    #retino_info = vis.get_retino_info(azimuth='right', elevation='top')
    print("*********SCREEN************")
    pp.pprint(retino_info)

    rundf = dataframes[run]['df']

    # Get phase info to screen coords (in degrees):
    linX, linY, linC = vis.convert_lincoords_lincolors(rundf, retino_info)

    
    # Get RGBA mapping normalized to mag-ratio values:
    curr_metric = 'magratio_mean1'
    # norm = mpl.colors.Normalize(vmin=-np.pi, vmax=np.pi)
    # cmap = mpl.cm.get_cmap('hsv')
    # mapper = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    # rgbas = np.array([mapper.to_rgba(v) for v in linC])
    # alphas = np.array(zdf[curr_metric] / zdf[curr_metric].max())
    magratios = zdf[curr_metric]

    
    if ax is None:
        fig, ax = pl.subplots()
    im = ax.scatter(linX, linY, s=roi_size, c=magratios, cmap='inferno', alpha=0.75, edgecolors='w') #, vmin=0, vmax=2*np.pi)
    magcmap=mpl.cm.inferno
    
    #ax.invert_xaxis()  # Invert x-axis so that negative values are on left side
    #    pl.xlim([retino_info['linminW'], retino_info['linmaxW']])
    #    pl.ylim([retino_info['linminH'], retino_info['linmaxH']])
    ax.set_xlim([-1*retino_info['width']/2., retino_info['width']/2.])
    ax.set_ylim([-1*retino_info['height']/2., retino_info['height']/2.])


#    pos = ax.get_position()
#    ax2 = ax.figure.add_axes([pos.x0+(1-pos.x0)*0.8, pos.y0*0.5, 0.02, 0.5])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    
    alpha_min = magratios.min()
    alpha_max = magratios.max()
    
    magnorm = mpl.colors.Normalize(vmin=alpha_min, vmax=alpha_max)
    pl.colorbar(im, cax=cax, cmap=magcmap, norm=magnorm)
    cax.yaxis.set_ticks_position('right')

    #cb = mpl.colorbar.ColorbarBase(ax2, cmap=magcmap, norm=magnorm, orientation='vertical')
    
    return linX, linY
    
#%


def plot_RF_position_and_size(ROIs, acquisition_dir, run, retinoid, screen_info=None, ax=None, interactive=True):


    assert len(np.unique([roi.conditions[0].name for roi in ROIs])) == 1
    assert len(np.unique([roi.conditions[1].name for roi in ROIs])) == 1

    az_rfs = [roi.conditions[0].RF_degrees for roi in ROIs]
    el_rfs = [roi.conditions[1].RF_degrees for roi in ROIs]
    nrois = len(ROIs)
    
    if ax is None:
        fig, ax = pl.subplots(figsize=(20,10))
     
    linX, linY = plot_ROI_positions(acquisition_dir, run, retinoid, screen_info=screen_info, ax=ax, interactive=interactive)
        
    ells = [Ellipse(xy=[linX[ri], linY[ri]], width=az_rfs[ri], height=el_rfs[ri]) for ri in range(nrois)
                if az_rfs[ri] > 0 and el_rfs[ri] > 0]
    
    r2_values = [np.mean([roi.conditions[0].fit_results['r2'], roi.conditions[1].fit_results['r2']]) for roi in ROIs]
    
    for ei,e in enumerate(ells): #[0:20]):
        ax.add_artist(e)
        #e.set_clip_box(ax.bbox)
#        if r2_values[ei] < 0:
#            continue
#        print r2_values[ei]
        #e.set_alpha(r2_values[ei])
        e.set_alpha(0.2)
        e.set_facecolor('none')
        e.set_linestyle('-')
        e.set_edgecolor('k')
    return ax



#%%

#rootdir = '/n/coxfs01/2p-data' #
#rootdir = '/mnt/odyssey'
#animalid = 'CE077'
#session = '20180523'
#acquisition = 'FOV1_zoom1x'
#run = 'retino'
#analysis_id = 'analysis001'
#default = False
#slurm = False
#
#fitness_threshold = 0.4
#size_threshold = 0.1

def extract_options(options):

    parser = optparse.OptionParser()

    parser.add_option('-D', '--root', action='store', dest='rootdir',
                          default='/n/coxfs01/2p-data',
                          help='data root dir (dir w/ all animalids) [default: /nas/volume1/2photon/data, /n/coxfs01/2pdata if --slurm]')
    parser.add_option('--slurm', action='store_true', dest='slurm', default=False, help="set if running as SLURM job on Odyssey")

    parser.add_option('-i', '--animalid', action='store', dest='animalid',
                          default='', help='Animal ID')

    # Set specific session/run for current animal:
    parser.add_option('-S', '--session', action='store', dest='session',
                          default='', help='session dir (format: YYYMMDD_ANIMALID')
    parser.add_option('-A', '--acq', action='store', dest='acquisition',
                          default='FOV1', help="acquisition folder (ex: 'FOV1_zoom3x') [default: FOV1]")
    parser.add_option('-R', '--run', action='store', dest='run',
                          default='retino_run1', help="run folder (ex: 'retino_run1') [default: retino_run1]")


    # Run specific info:
    parser.add_option('-r', '--retino', dest='retino_traceid', default=None, action='store', 
                      help='analysisid for RETINO [default assumes only 1 roi-based analysis]')
    parser.add_option('--fitness', dest='fitness_thr', default=0.5, action='store',
                      help='Threshold for R2 values for fitting response trace [default: 0.5]')
    parser.add_option('--size', dest='size_thr', default=0.1, action='store', 
                      help='Percent of max of gaussian fit to use to estimate RF size [default: 0.1]')

    #parser.add_option('-t', '--traceid', dest='traceid', default=None, action='store', help="datestr YYYYMMDD_HH_mm_SS")
     
    (options, args) = parser.parse_args(options)
    if options.slurm is True and '/n/coxfs01' not in options.rootdir:
        options.rootdir = '/n/coxfs01/2p-data'
    return options


# In[6]:



def estimate_RFs_and_plot(options):
    
    optsE = extract_options(options)
    rootdir = optsE.rootdir
    animalid = optsE.animalid
    session = optsE.session
    acquisition = optsE.acquisition
    fitness_thr = optsE.fitness_thr
    size_thr = optsE.size_thr
    analysis_id = optsE.retino_traceid
    run = optsE.run
    slurm = optsE.slurm
    
    # =============================================================================
    # Get run and analysis info using analysis-ID params:
    # =============================================================================
    acquisition_dir = os.path.join(rootdir, animalid, session, acquisition)
    print "RUN:", run
 
    ROIs, retinoid, screen_info = get_RF_size_estimates(acquisition_dir, retino_run=run,
                                                         fitness_thr=fitness_thr, 
                                                         size_thr=size_thr, 
                                                         analysis_id=analysis_id, slurm=slurm)
                            
    roi_outdir = os.path.join(acquisition_dir, run, 'retino_analysis', retinoid, 'visualization')
    if not os.path.exists(roi_outdir): os.makedirs(roi_outdir)
    roi_outfile = os.path.join(roi_outdir, '%s_RF_estimates.pkl' % retinoid)
    with open(roi_outfile, 'wb') as f:
        pkl.dump(ROIs, f, protocol=pkl.HIGHEST_PROTOCOL)
        
    #az_rfs = [roi.conditions[0].RF_degrees for roi in ROIs]
    #sns.distplot(az_rfs, kde=False, bins=50)
    if slurm: 
        interactive = False
    else:
        interactive = True 
    data_identifier = '|'.join([animalid, session, acquisition, run, retinoid])
    ax = plot_RF_position_and_size(ROIs, acquisition_dir, run, retinoid, screen_info=screen_info, ax=None, interactive=interactive)
    label_figure(ax.figure, data_identifier)
    pl.savefig(os.path.join(roi_outdir, 'RF_estimates_and_centroids_%s.png' % data_identifier.replace('|', '_')))

#%%
def main(options):
    estimate_RFs_and_plot(options)



if __name__ == '__main__':
    main(sys.argv[1:])
    
    

#%%

# TODO: Check that found peak matches PHASE based on fft...


#%%

#
#ncyles = len(framestart)
#parsed_traces = np.zeros((ncycles,stimframes_incl))
#for cycle in range(0,ncycles):
#    parsed_traces[cycle,:]=roi_traces[curr_roi,framestart[cycle]:framestart[cycle]+stimframes_incl]
#
#x0 = np.arange(0,parsed_traces.shape[-1])
#y = np.mean(parsed_traces, axis=0)
##y2 = y0 - y0.min()
#
#try:
#    center_start = np.argmax(y)-(np.argmax(y)/1)
#    center_end = np.argmax(y)+(np.argmax(y)/1)*2
#    if center_start < 0:
#        center_start = 0
#    if center_end > len(y):
#        center_end = len(y)
#    centered = y[np.arange(center_start, center_end)]
#    popt, pcov = curve_fit(gaus, x0, y, p0=(y[np.argmax(y)], np.argmax(y), 1))
#                          
#    y_fit = gaus(x0,*popt)
#    print(popt)
#
#    ss_res = np.sum((y - y_fit)**2)
#    ss_tot = np.sum((y-np.mean(y))**2)
#    r_squared = 1 - (ss_res / ss_tot)
#    print("R2:", r_squared)
#    
#except RuntimeError:
#    print("Error - curve_fit failed")
#
#if r_squared > fitness_threshold:
#    
#    fit_norm = y_fit/np.max(y_fit)
#
#    border_start = np.where(fit_norm>=size_threshold)[0]
#    if len(border_start)==0:
#        border_start = 0
#    else:
#        border_start = border_start[0]
#    border_end = np.where(fit_norm[border_start:]<=size_threshold)[0]
#    if len(border_end) == 0:
#        border_end = len(fit_norm)-1
#    else:
#        border_end = border_end[0]
#    print(border_start, border_end) #rf_size_frames)
#
#    # extrapolate around peak, in case edge:
#    peak = np.argmax(y)
#    tail2 = border_end - peak
#    tail1 = peak - border_start
#    if tail1 < tail2:
#        border_edge1 = peak - tail2
#        border_edge2 = peak + tail2
#    elif tail2 < tail1:
#        border_edge2 = peak + tail1
#        border_edge1 = peak - tail1
#    else:
#        border_edge1 = peak - tail1
#        border_edge2 = peak + tail2
#
#    rf_size_frames = border_edge2 - border_edge1
#    print rf_size_frames
#else:
#    rf_size_frames = 0
#
#print rf_size_frames
#
#
#plt.plot(y[0:], 'b+', label='data')
#plt.plot(y_fit[0:], 'r', label='fit')
#pl.plot(np.arange(border_start, border_end), y_fit[np.arange(border_start, border_end)], 
#                'k--', linewidth=2, label='FW')
#pl.legend()
#pl.title('roi%06d' % (curr_roi+1))

