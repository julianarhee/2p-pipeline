
import os
import json
import re
import scipy.io as spio
import numpy as np
from bokeh.plotting import figure
import tifffile as tf
import seaborn as sns
# %matplotlib notebook
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import skimage.color

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)

    return _check_keys(data)


def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])

    return dict


def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        elif isinstance(elem,np.ndarray):
            dict[strg] = _tolist(elem)
        else:
            dict[strg] = elem

    return dict


def _tolist(ndarray):
    '''
    A recursive function which constructs lists from cellarrays 
    (which are loaded as numpy ndarrays), recursing into the elements
    if they contain matobjects.
    '''
    elem_list = []
    for sub_elem in ndarray:
        if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
            elem_list.append(_todict(sub_elem))
        elif isinstance(sub_elem,np.ndarray):
            elem_list.append(_tolist(sub_elem))
        else:
            elem_list.append(sub_elem)

    return elem_list


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]


class StimInfo:
    def _init_(self):
        self.stimid = ''
        self.trials = []
        self.frames = []
        self.frames_sec = []
        self.stim_on_idx = []


source = '/nas/volume1/2photon/projects'
experiment = 'scenes'
session = '20171003_JW016'
acquisition = 'FOV1'
functional_dir = 'functional'

curr_file_idx = 2
curr_slice_idx = 20
curr_roi_method = 'blobs_DoG'

acquisition_dir = os.path.join(source, experiment, session, acquisition)
figdir = os.path.join(acquisition_dir, 'example_figures')

# Load reference info:
ref_json = 'reference_%s.json' % functional_dir 
with open(os.path.join(acquisition_dir, ref_json), 'r') as fr:
    ref = json.load(fr)

# Get ROI methods:
roi_methods_dir = os.path.join(acquisition_dir, 'ROIs')
roi_methods = os.listdir(roi_methods_dir)
roi_methods = [str(r) for r in roi_methods]
roi_methods_dict = dict()
print "Loading..."
for r in roi_methods:
    roiparams = loadmat(os.path.join(roi_methods_dir, r, 'roiparams.mat'))
    roiparams = roiparams['roiparams']
    roi_methods_dict[r] = dict()
    #roi_methods_dict[r]['maskpaths'] = roiparams['maskpaths']
    maskpaths = roiparams['maskpaths']
    if isinstance(maskpaths, unicode):
        roi_methods_dict[r]['Slice01'] = dict()
        masks = loadmat(maskpaths); masks = masks['masks']
        roi_methods_dict[r]['Slice01']['nrois'] = masks.shape[2]
        roi_methods_dict[r]['Slice01']['masks'] = masks       
    else:
        for si,sl in enumerate(maskpaths):
            masks = loadmat(sl); masks = masks['masks']
            roi_methods_dict[r]['Slice{:02d}'.format(si+1)] = dict()
            roi_methods_dict[r]['Slice{:02d}'.format(si+1)]['nrois'] = masks.shape[2]
            roi_methods_dict[r]['Slice{:02d}'.format(si+1)]['masks'] = masks

# Get TRACE methods:
trace_methods_dir = os.path.join(acquisition_dir, 'Traces')
trace_methods = os.listdir(trace_methods_dir)
trace_methods = [str(r) for r in trace_methods]

print "Trace methods:", trace_methods

# Get SLICE list:
if isinstance(ref['slices'], int):
    slice_names = ['Slice01']
else:
    slice_names = ["Slice{:02d}".format(i+1) for i in range(len(ref['slices']))]
slice_names = sorted(slice_names, key=natural_keys)
curr_slice_name = slice_names[curr_slice_idx]

# Get FILE list:
average_source = 'Averaged_Slices_Corrected'
signal_channel = 1
average_slice_dir = os.path.join(acquisition_dir, functional_dir, 'DATA', average_source, "Channel{:02d}".format(signal_channel))
file_names = [f for f in os.listdir(average_slice_dir) if '_vis' not in f]
print "File names:", file_names
nfiles = len(file_names)

# Get AVERAGE slices for current file:
curr_file_name = file_names[curr_file_idx]
curr_slice_dir = os.path.join(average_slice_dir, curr_file_name)
slice_fns = sorted([f for f in os.listdir(curr_slice_dir) if f.endswith('.tif')], key=natural_keys)

# Get TRACE structs for current-file, current-slice:
trace_types = ['raw', 'meansub', 'df/f']
tracestruct = loadmat(os.path.join(ref['trace_dir'], ref['trace_structs'][curr_slice_idx]))
traces = tracestruct['file'][0]
print traces.df_f.T.shape

# Print summary info:
nfiles = len(tracestruct['file'])
nframes = traces.df_f.T.shape[1]
nrois = traces.df_f.T.shape[0]
print "N files:", nfiles
print "N frames:", nframes
print "N rois:", nrois

# Get average slice image for current-file, current-slice:
curr_slice_fn = slice_fns[curr_slice_idx]
avg_tiff_path = os.path.join(curr_slice_dir, curr_slice_fn)
with tf.TiffFile(avg_tiff_path) as tif:
    avgimg = tif.asarray()


# Get PARADIGM INFO:
path_to_functional = os.path.join(acquisition_dir, functional_dir)
paradigm_dir = 'paradigm_files'
path_to_paradigm_files = os.path.join(path_to_functional, paradigm_dir)

stiminfo_basename = 'stiminfo'

# Load reference info:
ref_json = 'reference_%s.json' % functional_dir
with open(os.path.join(acquisition_dir, ref_json), 'r') as fr:
    ref = json.load(fr)

# Load SI meta data:
si_basepath = ref['raw_simeta_path'][0:-4]
simeta_json_path = '%s.json' % si_basepath
with open(simeta_json_path, 'r') as fs:
    simeta = json.load(fs)

nframes = int(simeta[curr_file_name]['SI']['hFastZ']['numVolumes'])
framerate = float(simeta[curr_file_name]['SI']['hRoiManager']['scanFrameRate'])
volumerate = float(simeta[curr_file_name]['SI']['hRoiManager']['scanVolumeRate'])
frames_tsecs = np.arange(0, nframes)*(1/volumerate)


# frame info:
first_frame_on = 50
stim_on_sec = 0.5
iti = 1.
vols_per_trial = 15

nframes_on = stim_on_sec * volumerate
nframes_off = vols_per_trial - nframes_on
frames_iti = round(iti * volumerate)

# Create stimulus-dict:
stimdict = dict()
for fi in range(nfiles):
    currfile = "File%03d" % int(fi+1)
    stim_fn = 'stim_order.txt'

    # Load stim-order:
    with open(os.path.join(path_to_paradigm_files, stim_fn)) as f:
        stimorder = f.readlines()
    curr_stimorder = [l.strip() for l in stimorder]
    unique_stims = sorted(set(curr_stimorder), key=natural_keys)
    first_frame_on = 50
    for trialnum,stim in enumerate(curr_stimorder):
        #print "Stim on frame:", first_frame_on
        if not stim in stimdict.keys():
            stimdict[stim] = dict()
        if not currfile in stimdict[stim].keys():
            stimdict[stim][currfile] = StimInfo()
            stimdict[stim][currfile].trials = []
            stimdict[stim][currfile].frames = []
            stimdict[stim][currfile].frames_sec = []
            stimdict[stim][currfile].stim_on_idx = []

        framenums = list(np.arange(int(first_frame_on-frames_iti), int(first_frame_on+(vols_per_trial))))
        frametimes = [frames_tsecs[f] for f in framenums]
        stimdict[stim][currfile].trials.append(trialnum)      
        stimdict[stim][currfile].frames.append(framenums)
        stimdict[stim][currfile].frames_sec.append(frametimes)
        stimdict[stim][currfile].stim_on_idx.append(framenums.index(first_frame_on))
        first_frame_on = first_frame_on + vols_per_trial


# 
# stiminfo_json = '%s.json' % stiminfo_basename
# stiminfo_mat = '%s.mat' % stiminfo_basename
# 
# with open(os.path.join(path_to_paradigm_files, stiminfo_json), 'w') as fw:
#     json.dumps(jsonify(stimdict), fw, sort_keys=True, indent=4)
# scipy.io.savemat(os.path.join(path_to_paradigm_files, stiminfo_mat), mdict=stimdict)


# Split all traces by stimulus-ID:
# ----------------------------------------------------------------------------
stim_ntrials = dict()
for stim in stimdict.keys():
    stim_ntrials[stim] = 0
    for fi in stimdict[stim].keys():
        stim_ntrials[stim] += len(stimdict[stim][fi].trials)

# To look at all traces for ROI 3 for stimulus 1:
# traces_by_stim['1'][:,roi,:]

traces_by_stim = dict()
frames_stim_on = dict()
for stim in stimdict.keys():
    repidx = 0
    curr_traces_allrois = []
    stim_on_frames = []
    for fi,currfile in enumerate(sorted(file_names, key=natural_keys)):
        frames_by_trial = stimdict[stim][currfile].frames
        for currtrial in range(len(frames_by_trial)):
            currframes = stimdict[stim][currfile].frames[currtrial]

            curr_traces_allrois.append(tracestruct['file'][fi].tracematDC.T[:, currframes])
            repidx += 1
            
            curr_frame_onset = stimdict[stim][currfile].stim_on_idx[currtrial]
            
            stim_on_frames.append([curr_frame_onset, curr_frame_onset + stim_on_sec*volumerate])

    traces_by_stim[stim] = np.asarray(curr_traces_allrois)
    frames_stim_on[stim] = stim_on_frames



# ---------------------------------------------------------------------------
# PLOTTING:
# ----------------------------------------------------------------------------

def get_spaced_colors(n):
    max_value = 16581375 #255**3
    interval = int(max_value / n)
    colors = [hex(I)[2:].zfill(6) for I in range(1, max_value, interval)]

    return [(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors]

color_by_roi = True

spacing = 25
nstimuli = len(stimdict.keys())

if color_by_roi:
    colorvals255 = get_spaced_colors(nrois)
else:
    coorvals255 = get_spaced_colors(nstimuli)
#colorvals255 = colorvals255[1:] 
colorvals = np.true_divide(colorvals255, 255)
print len(colorvals255)

plot_rois = np.arange(0, nrois, 2) #int(nrois/2)
fig = plt.figure(figsize=(nstimuli,int(len(plot_rois)/2)))
gs = gridspec.GridSpec(len(plot_rois), 1) #, height_ratios=[1,1,1,1]) 
gs.update(wspace=0.01, hspace=0.01)
for ridx,roi in enumerate(plot_rois): #np.arange(0, nrois, 2): # range(plot_rois): # nrois
    #rowindex = roi + roi*nstimuli
    print "ROI:", roi
    plt.subplot(gs[ridx])
    plt.axis('off')
    #ax = plt.gca()
        
    for stimnum,stim in enumerate(traces_by_stim.keys()):
        #plt.subplot(gs[roi, stimnum])
        #print stim
        raw = traces_by_stim[stim][:, roi, :]
        #avg = np.mean(raw, axis=0)
        xvals = np.arange(0, raw.shape[1]) + stimnum*spacing
        #xvals = np.tile(np.arange(0, raw.shape[1]), (raw.shape[0], 1))
        curr_dfs = np.empty((raw.shape[0], raw.shape[1])) 
        for trial in range(raw.shape[0]):
            frame_on = frames_stim_on[stim][trial][0]
            baseline = np.mean(raw[trial,0:frame_on])
            df = (raw[trial,:] - baseline / baseline)
            curr_dfs[trial,:] = df
            if color_by_roi:
                plt.plot(xvals, df, color=colorvals[roi], alpha=1, linewidth=0.2)
            else:
                plt.plot(xvals, df, color=colorvals[stimnum], alpha=1, linewidth=0.2)
            
            stim_frames = xvals[0]+frames_stim_on[stim][trial]
            
            plt.plot(stim_frames, np.ones((2,))*-20, color='k')
        
        # Plot average:
        avg = np.mean(curr_dfs, axis=0) 
        if color_by_roi:
            plt.plot(xvals, avg, color=colorvals[roi], alpha=1, linewidth=2)
        else:
            plt.plot(xvals, avg, color=colorvals[stimnum], alpha=1, linewidth=2)
    
    if roi<len(plot_rois)-1:
        #sns.despine(bottom=True)
        plt.axis('off')
        plt.ylabel(str(roi))
    else:
        #ax.axes.get_xaxis().set_visible(False)
        #ax.axes.get_xaxis().set_ticks([])
        plt.yticks([0, 100])

#fig.tight_layout()
sns.despine(offset=1, trim=True)

figname = 'all_rois_traces_by_stim.png'
plt.savefig(os.path.join(figdir, figname), bbox_inches='tight')
#plt.show()

# PLOT ROIs:
img = np.copy(avgimg)

curr_masks = roi_methods_dict[curr_roi_method][curr_slice_name]['masks']
label_masks = np.zeros((curr_masks.shape[0], curr_masks.shape[1]))
print label_masks.shape
roi_idx = 1
for roi in plot_rois:
    label_masks[curr_masks[:,:,roi]==1] = int(roi_idx)
    roi_idx += 1

plt.figure()
# plt.imshow(img)
imgnorm = np.true_divide((img - img.min()), (img.max()-img.min()))
#plt.imshow(imgnorm, cmap='gray'); plt.colorbar()
plt.imshow(skimage.color.label2rgb(label_masks, image=imgnorm, alpha=0.1, colors=colorvals255, bg_label=0))
plt.axis('off')

figname = 'all_rois_average_slice.png'
plt.savefig(os.path.join(figdir, figname), bbox_inches='tight')
#plt.show()


