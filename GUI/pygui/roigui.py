import pandas as pd
from bokeh.layouts import row, widgetbox, column, layout
from bokeh.models import Select, Slider, ColumnDataSource
from bokeh.charts import Histogram
from bokeh.io import show
from bokeh.application.handlers import FunctionHandler
from bokeh.application import Application
from bokeh.models.widgets import RadioGroup
from bokeh.plotting import curdoc, figure
from bokeh.models.glyphs import Patches

# Load reference dict:
import json
import os
import tifffile as tf
import scipy.io as spio
import numpy as np
from skimage import color, img_as_float

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


source = '/nas/volume1/2photon/projects'
experiment = 'gratings_phaseMod'
session = '20170825_CE055'
acquisition = 'FOV1_planar'
functional_dir = 'functional_test'

acquisition_dir = os.path.join(source, experiment, session, acquisition)


# Load REFERENCE struct:
# ----------------------------------------------------------------------------
refdict_fn = 'reference_%s2.json' % functional_dir
with open(os.path.join(acquisition_dir, refdict_fn), 'r') as fr:
    ref = json.load(fr)

# Get ROI methods:
# ----------------------------------------------------------------------------
roi_methods_dir = os.path.join(ref['acquisition_base_dir'], 'ROIs')
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
# ----------------------------------------------------------------------------
trace_methods_dir = os.path.join(ref['acquisition_base_dir'], 'Traces')
trace_methods = os.listdir(trace_methods_dir)
trace_methods = [str(r) for r in trace_methods]

# Get SLICE list:
if isinstance(ref['slices'], int):
    slice_names = ['Slice01']
else:
    slice_names = ["Slice{:02d}".format(i+1) for i in range(len(ref['slices']))]

# Get FILE list:
# ----------------------------------------------------------------------------
average_slice_dir = os.path.join(ref['data_dir'], 'Averaged_Slices', "Channel{:02d}".format(ref['signal_channel']))
file_fns = [f for f in os.listdir(average_slice_dir) if '_visible' not in f]
#print file_fns
nfiles = len(file_fns)

# Get AVERAGE slices:
slice_fns = [f for f in os.listdir(os.path.join(average_slice_dir, file_fns[0])) if f.endswith('.tif')]
#print slice_fns

trace_types = ['raw', 'meansub', 'df/f']
nrois = 10
# Create the Document Application
#def modify_doc(doc):
    
# Create the main plot
# ----------------------------------------------------------------------------

def create_figure():
    
    curr_roi = roi_slider.value - 1
    curr_file = file_name.value
    curr_file_idx = file_fns.index(curr_file)
    
    current_slice_dir = os.path.join(average_slice_dir, curr_file)
    slice_fns = os.listdir(current_slice_dir)
    slice_fns = [s for s in slice_fns if s.endswith('.tif')]
    
    current_roi_method = roi_method.value
    current_slice_name = slice_name.value
    slice_idx = slice_names.index(current_slice_name)
 
    # GET average slice image:
    tiff_path = os.path.join(current_slice_dir, slice_fns[slice_idx])
    with tf.TiffFile(tiff_path) as tif:
	img = tif.asarray()
    # reverse y axis manualy
    img = img[::-1]


    curr_mask = roi_methods_dict[current_roi_method][current_slice_name]['masks'][:,:,curr_roi]
    #curr_mask[curr_mask==0] = None #None
    #curr_mask = np.ma.array(curr_mask)
    all_masks = np.sum(roi_methods_dict[current_roi_method][current_slice_name]['masks'], axis=2)
    #all_masks[all_masks==0] = None
    #all_masks = np.ma.array(all_masks)

   
    img = img_as_float(img)
   
    # Construc colored masks to superimpose: 
    rows,cols = img.shape
    cmask = np.zeros((rows,cols,3))
    #cmask[all_masks==1] = [0, 1, 0]
    cmask[curr_mask==1] = [1, 0, 0]  # Red blob
     
    # Construct RGB version of gray-scale TIFF img:
    img_color = np.dstack((img, img, img))
 
    # Convert input image and roi-cmask into HSV colorspace:
    img_hsv = color.rgb2hsv(img_color)
    cmask_hsv = color.rgb2hsv(cmask)

    # Replace H and S of original image with the colormask:
    img_hsv[..., 0] = cmask_hsv[..., 0]
    img_hsv[..., 1] = cmask_hsv[..., 1] * 0.5
    img_masked = color.hsv2rgb(img_hsv)

    # img_masked = img_masked.view(dtype=np.uint32) 
    
    # LOAD masks of selected ROI-type (and, if relevant, selected Slice):
    #roiparams_path = os.path.join(ref['acquisition_base_dir'], 'ROIs', current_roi_method, 'roiparams.mat')
    #roiparams = loadmat(roiparams_path); roiparams = roiparams['roiparams']
    #nrois = roiparams['nrois']

#     maskpaths = roiparams['maskpaths']
#     if isinstance(maskpaths, unicode):
# 	masks = loadmat(maskpaths); masks = masks['masks']
#     else:
# 	masks = loadmat(maskpaths[slice_idx]); masks = masks['masks']
# 
#    curr_mask = masks[:,:,curr_roi]
#     roi_list = []
#     for roi in range(roi_methods_dict[current_roi_method][current_slice_name]['nrois']):
#    curr_mask = roi_methods_dict[current_roi_method][current_slice_name]['masks'][:,:,curr_roi]
# 	curr_mask = curr_mask[::-1]
# 	xs=[]; ys=[]
# 	for x in range(curr_mask.shape[1]):
# 	    for y in range(curr_mask.shape[0]):
# 		if curr_mask[x,y]==1:
# 		    xs.append(x)
# 		    ys.append(y)
# 	xs = [float(x)/curr_mask.shape[0] for x in xs]
# 	ys = [float(y)/curr_mask.shape[1] for y in ys]
#         roi_list.append([xs, ys])

    #roisource = ColumnDataSource(dict(xs=[roi[0] for roi in roi_list], ys=[roi[1] for roi in roi_list]))
    #glyph = Patches(xs="xs", ys="ys", fill_alpha=0., line_width=2)

    #mask_img = np.ma.masked_array(curr_mask==0, curr_mask)

    # LOAD TRACES of selected ROI-type, Slice, and File:
    tracestruct = loadmat(os.path.join(ref['trace_dir'], ref['trace_structs'][slice_idx]))
    if nfiles==1:
	traces = tracestruct['file']
    else:
	traces = tracestruct['file'][curr_file_idx]
    

    # Plot selected traces:
    curr_trace_type = trace_types[trace_type_group.active]
    if curr_trace_type=='raw':
	#curr_trace = traces['rawtracemat'][:,curr_roi]
	selected_mat = 'rawtracemat'
	curr_ylabel = 'intensity'
	to_perc = 1.
    elif curr_trace_type=='meansub':
	#curr_trace = traces['tracematDC'][:,curr_roi]
	selected_mat = 'tracematDC'
	curr_ylabel = 'intensity'
	to_perc = 1.
    elif curr_trace_type=='df/f':
	#curr_trace = traces['df_f'][:,curr_roi]*100.
	selected_mat = 'df_f'
	curr_ylabel = 'df/f %'
	to_perc = 100.
    
    curr_trace = [traces[selected_mat][f][curr_roi]*to_perc for f in range(len(traces[selected_mat]))]
	

    # Create figure 1:
    s1 = figure(plot_width=600, plot_height=400, x_range=(0,1), y_range=(0,1))
    
#    s1.image([img], 0, 0, 1, 1, dilate=True) #], x=[0], y=[0], dw=None, dh=None, dilate=True)
#    s1.patches(xs=[roi[0] for roi in roi_list], ys=[roi[1] for roi in roi_list], alpha=0.5)    
#     for roi in roi_list: 
#         s1.patch(roi[0], roi[1], line_width=2, alpha=0.5, fill_alpha=0.0)
        #s1.patch(xs, ys, line_width=2, alpha=0.5, fill_alpha=0.0)
#    s1.image_rgba([curr_mask], 0, 0, 1, 1, alpha=0.5)

    s1.image([img], 0, 0, 1, 1) #, 0, 0, 1, 1) #, alpha=0.3)
    print "PLOTTED" 
    print img_masked.shape 
    print img_masked.dtype
	
    # Set the x axis label
    s1.xaxis.axis_label = current_slice_name

    # Set the y axis label
    s1.yaxis.axis_label = ''
    s1.xaxis.major_tick_line_color = None  # turn off x-axis major ticks
    s1.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
    s1.yaxis.major_tick_line_color = None  # turn off y-axis major ticks
    s1.yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks
    s1.xaxis.major_label_text_font_size = '0pt'  # preferred method for removing tick labels
    s1.yaxis.major_label_text_font_size = '0pt'  # preferred method for removing tick labels

    
    # Create figure 2:
    s2 = figure(plot_width=1200, plot_height=400) #, x_range=(0,1), y_range=(0,1))
    s2.line(range(1, len(curr_trace)), curr_trace)
    s2.xaxis.axis_label = 'frame'
    s2.yaxis.axis_label = curr_ylabel
    
    #p = column(s1, s2)
    return s1, s2


# Update the plot
# ----------------------------------------------------------------------------
def update(attr, old, new):
    layout.children[1].children[1], layout.children[2] = create_figure()
    # print layout.children[1].children
    #push_notebook()
    roi_slider.end = roi_methods_dict[roi_method.value][slice_name.value]['nrois']    



# ----------------------------------------------------------------------------------
# Controls
# ----------------------------------------------------------------------------------
file_name = Select(title='File:', options=file_fns, value=file_fns[0])
file_name.on_change('value', update)
file_menu = widgetbox([file_name], width=200)


# ROI-METHODS:
roi_method = Select(title="ROI method:", options=roi_methods, value=roi_methods[0])
roi_method.on_change('value', update)
roi_menu = widgetbox([roi_method], width=200)

slice_name = Select(title="Slice:", options=slice_names, value=slice_names[0])
slice_name.on_change('value', update)
slice_menu = widgetbox([slice_name], width=200)

roi_slider = Slider(start=1, end=5, value=1, step=1, title="ROI num")
roi_slider.on_change('value', update) #, callback=update)


# TRACE-METHODS:
trace_method = Select(title="Trace method:", options=trace_methods, value=trace_methods[0])
trace_method.on_change('value', update)
trace_menu = widgetbox([trace_method], width=200)

trace_type_group = RadioGroup(labels=trace_types, active=0)
trace_type_group.on_change('active', update) #lambda attr, old, new: update())
# ----------------------------------------------------------------------------------



# make it
# ----------------------------------------------------------------------------
s1, s2 = create_figure()
layout = column(roi_slider, row(column(slice_menu, roi_menu, trace_method, trace_type_group), s1), s2)
curdoc().add_root(layout)
curdoc().title = 'ROI GUI'

    #layout2 = row(column(trace_method, trace_type_group), s2)
    #doc.add_root(layout)
    #l = layout([layout1], [layout2])
    #l = layout([layout1]) #, sizing_mode='stretch_both')
#    doc.add_root(layout)

# Set up the Application 
#handler = FunctionHandler(modify_doc)
#app = Application(handler)
