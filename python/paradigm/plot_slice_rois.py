

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
from json_tricks.np import dump, dumps, load, loads
import mat2py
#from mat2py import loadmat
from skimage import color
import cPickle as pkl
from skimage import exposure
import scipy.io

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]

# def get_spaced_colors(n):
#     max_value = 16581375 #255**3
#     interval = int(max_value / n)
#     colors = [hex(I)[2:].zfill(6) for I in range(1, max_value, interval)]

#     return [(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors]


import optparse

parser = optparse.OptionParser()
parser.add_option('-S', '--source', action='store', dest='source', default='/nas/volume1/2photon/projects', help='source dir (root project dir containing all expts) [default: /nas/volume1/2photon/projects]')
parser.add_option('-E', '--experiment', action='store', dest='experiment', default='', help='experiment type (parent of session dir)') 
parser.add_option('-s', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID') 
parser.add_option('-A', '--acq', action='store', dest='acquisition', default='', help="acquisition folder (ex: 'FOV1_zoom3x')")
parser.add_option('-f', '--functional', action='store', dest='functional_dir', default='functional', help="folder containing functional TIFFs. [default: 'functional']")

parser.add_option('-z', '--slice', action="store",
                  dest="sliceidx", default=0, help="Slice index to look at (0-index) [default: 0]")
# parser.add_option('-r', '--method', action="store",
#                   dest="roi_method", default='blobs_DoG', help="roi method [default: 'blobsDoG]")
# 
parser.add_option('-R', '--rois', action="store",
                  dest="rois_to_plot", default='', help="index of ROIs to plot")

parser.add_option('-I', '--id', action="store",
                  dest="analysis_id", default='', help="analysis_id (includes mcparams, roiparams, and ch). see <acquisition_dir>/analysis_record.json for help.")


parser.add_option('-b', '--background', action="store",
                  dest="background", default=0.3, help="threshold below which to raise intensity [default: 0.3]")
parser.add_option('--no-color', action="store_true",
                  dest="no_color", default=False, help="Don't plot by color (just grayscale)")

parser.add_option('--stim-color', action="store_false",
                  dest="color_by_roi", default=True, help="Color by STIM instead of ROI (default: color by ROI id).")



(options, args) = parser.parse_args() 

analysis_id = options.analysis_id

source = options.source #'/nas/volume1/2photon/projects'
experiment = options.experiment #'scenes' #'gratings_phaseMod' #'retino_bar' #'gratings_phaseMod'
session = options.session #'20171003_JW016' #'20170927_CE059' #'20170902_CE054' #'20170825_CE055'
acquisition = options.acquisition #'FOV1' #'FOV1_zoom3x' #'FOV1_zoom3x_run2' #'FOV1_planar'
functional_dir = options.functional_dir #'functional' #'functional_subset'

curr_slice_idx = int(options.sliceidx)

#roi_method = options.roi_method
no_color = options.no_color
color_by_roi = options.color_by_roi

#color_by_roi = True
cmaptype = 'rainbow'
background_offset = float(options.background) #0.3 #0.8


tmprois = options.rois_to_plot

# ---------------------------------------------------------------------------------


acquisition_dir = os.path.join(source, experiment, session, acquisition)

# Create tmp fig dir
figbase = os.path.join(acquisition_dir, 'figures', analysis_id) #'example_figures'
if not os.path.exists(figbase):
    os.makedirs(figbase)

if len(tmprois)==0:
    figdir = figbase
else:
    figdir = os.path.join(figbase, 'roi_subsets')
if not os.path.exists(figdir):
    os.mkdir(figdir)
print "Saving ROI slice images to dir:", figdir
 
# Load reference info:
ref_json = 'reference_%s.json' % functional_dir 
with open(os.path.join(acquisition_dir, ref_json), 'r') as fr:
    ref = json.load(fr)

# Load SI meta data:
si_basepath = ref['raw_simeta_path'][0:-4]
simeta_json_path = '%s.json' % si_basepath
with open(simeta_json_path, 'r') as fs:
    simeta = json.load(fs)


# Get ROIPARAMS:
roi_dir = os.path.join(ref['roi_dir'], ref['roi_id'][analysis_id]) #, 'ROIs')
roiparams = mat2py.loadmat(os.path.join(roi_dir, 'roiparams.mat'))
if 'roiparams' in roiparams.keys():
    roiparams = roiparams['roiparams']
maskpaths = roiparams['maskpaths']
print maskpaths
if not isinstance(maskpaths, list) and (len(maskpaths)==1 or isinstance(maskpaths, unicode)):
    maskpaths = [maskpaths] #[str(i) for i in maskpaths]

# Check slices to see if maskpaths exist for all slices, or just a subset:
if 'sourceslices' in roiparams.keys():
    slices = roiparams['sourceslices']
else:
    slices = np.arange(1, len(maskpaths)+1) #range(len(maskpaths))
print "Found masks for slices:", slices
if isinstance(slices, int):
    slices = [slices]
if not isinstance(slices, list): # i.e., only 1 slice
    slices = [int(i) for i in slices]
if not isinstance(slices[0], int):
    slices = [s for s in sublist for sublist in slices]
print "Found masks for slices:", slices


# Load masks:
masks = dict(("Slice%02d" % int(slice_idx), dict()) for slice_idx in slices)
for sidx,maskpath in zip(sorted(slices), sorted(maskpaths, key=natural_keys)):
    slice_name = "Slice%02d" % int(sidx) #+1)
    print "Loading masks: %s..." % slice_name 
    tmpmasks = mat2py.loadmat(maskpath); tmpmasks = tmpmasks['masks']
    masks[slice_name]['nrois'] =  tmpmasks.shape[2]
    masks[slice_name]['masks'] = tmpmasks


slice_names = sorted(masks.keys(), key=natural_keys)
print "SLICE NAMES:", slice_names
curr_slice_name = slice_names[curr_slice_idx]
nrois = masks[curr_slice_name]['nrois']
print "NROIS:", nrois
currmasks = masks[curr_slice_name]['masks']

# roi_string = options.rois_to_plot
# if len(roi_string)==0:
#     rois_to_plot = np.arange(0, nrois)
# else:
#     rois_to_plot = options.rois_to_plot.split(',')
#     rois_to_plot = [int(r) for r in rois_to_plot]
# print "ROIS TO PLOT:", rois_to_plot
# 

# Select ROIs:
#rois_to_plot = options.rois_to_plot.split(',')
#rois_to_plot = [int(r) for r in rois_to_plot]


roi_interval = 1
if len(tmprois)==0:
    rois_to_plot = np.arange(0, nrois, roi_interval) #int(nrois/2)
    sort_name = '_all' #% roi_interval
else:
    rois_to_plot = options.rois_to_plot.split(',')
    rois_to_plot = [int(r) for r in rois_to_plot]
    roi_string = "".join(['r%i' % int(r) for r in rois_to_plot])
    print roi_string
    sort_name = '_selected_%s' % roi_string

print "ROIS TO PLOT:", rois_to_plot

# Create ROI dir in figures: 
# if len(tmprois)==0:
#     figdir = os.path.join(figbase, 'rois') #'rois')
# else:
#     figdir = os.path.join(figbase, 'rois')
figdir = os.path.join(figbase, 'rois')
if not os.path.exists(figdir):
    os.makedirs(figdir)
print "Saving ROI slice images to dir:", figdir
 

   
if no_color is True:
    colorvals = np.zeros((nrois, 3));
    colorvals[:,1] = 1
else:
    cmaptype = 'rainbow'
    colormap = plt.get_cmap(cmaptype)
    colorvals = colormap(np.linspace(0, 1, nrois)) #get_spaced_colors(nrois)
    colorvals255 = [c[0:-1]*255 for c in colorvals]


# Get FILE ("tiff") list
#avg_dir = options.avg_dir
avg_dir = ref['average_source'][analysis_id]
average_source = 'Averaged_Slices_%s' % avg_dir
signal_channel = ref['signal_channel'][analysis_id] #int(options.selected_channel)
average_slice_dir = os.path.join(acquisition_dir, functional_dir, 'DATA', average_source, "Channel{:02d}".format(signal_channel))
file_names = [f for f in os.listdir(average_slice_dir) if '_vis' not in f]
print "File names:", file_names
nfiles = len(file_names)

# Get AVERAGE slices (for current file):
mcparams = scipy.io.loadmat(ref['mcparams_path'])
if 'mcparams' in mcparams.keys():
    mcparams = mcparams['mcparams']
ref_file_idx = mcparams[ref['mc_id'][analysis_id]]['ref_file'][0][0][0]
if not isinstance(ref_file_idx, int):
    ref_file_idx = ref_file_idx[0]
   
print "REF FILE:", ref_file_idx
avg_slice_fn = "File%03d_visible" % int(ref_file_idx)
ref_slice_path = os.path.join(average_slice_dir, avg_slice_fn)
slice_fns = sorted([f for f in os.listdir(ref_slice_path) if f.endswith('.tif')], key=natural_keys)
curr_slice_fn = slice_fns[curr_slice_idx]

with tf.TiffFile(os.path.join(ref_slice_path, curr_slice_fn)) as tif:
    avgimg = tif.asarray()


   

# PLOT ROIs:
img = np.copy(avgimg)

plt.figure()
plt.imshow(img)
plt.subplot(1,2,1)
imgscale = exposure.rescale_intensity(img, in_range=(avgimg.min()+(avgimg.max()*0.1), avgimg.max()-(avgimg.max()*0.1)))
print imgscale.min(), imgscale.max()
plt.imshow(imgscale, cmap='gray')
plt.subplot(1,2,2)
plt.imshow(img, cmap='gray')
plt.show()


#img = np.copy(avgimg)
factor = 1
imgnorm = np.true_divide((img - img.min()), factor*(img.max()-img.min()))
# imgnorm = np.true_divide((imgscale - imgscale.min()), factor*(imgscale.max()-imgscale.min()))

#imgnorm[imgnorm<background_offset] += 0.15

alpha = 0.9 #0.8 #1 #0.8 #0.5 #0.99 #0.8
nr,nc = imgnorm.shape
color_mask = np.zeros((nr, nc, 3))
#color_mask = np.dstack((imgnorm, imgnorm, imgnorm)) 
for roi in rois_to_plot:
    color_mask[currmasks[:,:,roi]==1] = colorvals[roi][0:3]

# Construct RGB version of grey-level image
img_color = np.dstack((imgnorm, imgnorm, imgnorm))

# Convert the input image and color mask to Hue Saturation Value (HSV)
# colorspace
img_hsv = color.rgb2hsv(img_color)
color_mask_hsv = color.rgb2hsv(color_mask)

# Replace the hue and saturation of the original image
# with that of the color mask
img_hsv[..., 0] = color_mask_hsv[..., 0]
img_hsv[..., 1] = color_mask_hsv[..., 1] #* alpha
#img_hsv[..., 2] = 0.5


img_masked = color.hsv2rgb(img_hsv)

plt.figure()
plt.imshow(img_masked, cmap=cmaptype)

for roi in rois_to_plot:
    [ys, xs] = np.where(currmasks[:,:,roi]==1)
    #plt.text(xs[0], ys[0], str(roi))
    plt.text(xs[int(round(len(xs)/4))], ys[int(round(len(ys)/4))], str(roi), weight='bold') 

plt.axis('off')
plt.tight_layout()

figname = 'rois_average_slice%i%s.png' % (curr_slice_idx, sort_name)
plt.savefig(os.path.join(figdir, figname), bbox_inches='tight')

