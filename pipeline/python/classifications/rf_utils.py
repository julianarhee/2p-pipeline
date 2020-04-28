import os
import glob
import cv2

import numpy as np
import pandas as pd

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