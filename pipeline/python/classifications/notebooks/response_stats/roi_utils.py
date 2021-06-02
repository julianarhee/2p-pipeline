import os
import cv2
import traceback 
import h5py
import glob
import json
import imutils
import h5py
import tifffile as tf
import dill as pkl
import numpy as np
import pandas as pd
import py3utils as p3



def get_masks_and_centroids(dk, traceid='traces001',
                        rootdir='/n/coxfs01/2p-data'):
    '''
    Load zprojected image, masks (nrois, d2, d2), and centroids for dataset.
    '''
    session, animalid, fovnum = p3.split_datakey_str(dk)
    fov = 'FOV%i_zoom2p0x' % fovnum

    # Load zimg
    roiid = get_roiid_from_traceid(animalid, session, 'FOV%i_*' % fovnum, 
                                          'gratings', traceid=traceid)
    zimg_path = glob.glob(os.path.join(rootdir, animalid, session, \
                                       'ROIs', '%s*' % roiid, 'figures', '*.tif'))[0]
    zimg = tf.imread(zimg_path)
    zimg = zimg[:, :, 1]
    # Load masks for centroids
    masks, _ = load_roi_masks(animalid, session, fov, rois=roiid, 
                                       rois_first=True)
    # Get centroids, better for plotting
    centroids =  get_roi_centroids(masks)

    return zimg, masks, centroids

def get_roi_centroids(masks):
    '''Calculate center of soma, then return centroid coords.
    '''
    centroids=[]
    for roi in range(masks.shape[0]):
        img = masks[roi, :, :].copy()
        x, y = np.where(img>0)
        centroid = ( round(sum(x) / len(x)), round(sum(y) / len(x)) )
        centroids.append(centroid)
    
    nrois_total = masks.shape[0]
    ctr_df = pd.DataFrame(centroids, columns=['x', 'y'], index=range(nrois_total))

    return ctr_df


def load_roi_masks(animalid, session, fov, rois=None, 
                rois_first=False, rootdir='/n/coxfs01/2p-data'):
    '''
    Loads ROI masks (orig) hdf5 file.
    Returns masks, zimg
    '''
    masks=None; zimg=None;
    mask_fpath = glob.glob(os.path.join(rootdir, animalid, session, 
                                'ROIs', '%s*' % rois, 'masks.hdf5'))[0]
    try:
        mfile = h5py.File(mask_fpath, 'r')

        # Load and reshape masks
        reffile = list(mfile.keys())[0]
        masks = mfile[reffile]['masks']['Slice01'][:].T
        #print(masks.shape)

        zimg = mfile[reffile]['zproj_img']['Slice01'][:].T
       
        if rois_first:
            masks_r0 = np.swapaxes(masks, 0, 2)
            masks = np.swapaxes(masks_r0, 1, 2)
    except Exception as e:
        traceback.print_exc()
    finally:
        mfile.close()
 
    return masks, zimg


def load_roi_coords(animalid, session, fov, roiid=None,
                    convert_um=True, traceid='traces001',
                    create_new=False,rootdir='/n/coxfs01/2p-data'):
    fovinfo = None
    roiid = get_roiid_from_traceid(animalid, session, fov, traceid=traceid)
    # create outpath
    roidir = glob.glob(os.path.join(rootdir, animalid, session,
                        'ROIs', '%s*' % roiid))[0]
    fovinfo_fpath = os.path.join(roidir, 'fov_info.pkl')
    if not create_new:
        try:
            # print("... loading roi coords")
            with open(fovinfo_fpath, 'rb') as f:
                fovinfo = pkl.load(f, encoding='latin1')
            assert 'roi_positions' in fovinfo.keys(), "Bad fovinfo file, redoing"
        except Exception as e: #AssertionError:
            traceback.print_exc()
            create_new = True

    if create_new:
        print("... calculating roi-2-fov info")
        masks, zimg = load_roi_masks(animalid, session, fov, rois=roiid)
        fovinfo = calculate_roi_coords(masks, zimg, convert_um=convert_um)
        with open(fovinfo_fpath, 'wb') as f:
            pkl.dump(fovinfo, f, protocol=pkl.HIGHEST_PROTOCOL)

    return fovinfo

def get_roiid_from_traceid(animalid, session, fov, run_type=None,
                            traceid='traces001', rootdir='/n/coxfs01/2p-data'):

    if run_type is not None:
        if int(session) < 20190511 and 'rfs' in run_type:
            run_name = 'gratings'

        a_traceid_dict = glob.glob(os.path.join(rootdir, animalid, session,
                                    fov, '*%s*' % run_type, 'traces',
                                    'traceids*.json'))[0]
    else:
        a_traceid_dict = glob.glob(os.path.join(rootdir, animalid, session,
                                    fov, '*run*', 'traces', 'traceids*.json'))[0]
    with open(a_traceid_dict, 'r') as f:
        tracedict = json.load(f)

    tid = tracedict[traceid]
    roiid = tid['PARAMS']['roi_id']

    return roiid


def calculate_roi_coords(masks, zimg, roi_list=None, convert_um=True):
    '''
    Get FOV info relating cortical position to RF position of all cells.
    Info should be saved in: rfdir/fov_info.pkl
    
    Returns:
        fovinfo (dict)
            'roi_positions': dataframe
                fov_xpos: micron-converted fov position (IRL is AP-axis)
                fov_ypos: " " (IRL is ML-axis)
                fov_xpos_pix: coords in pixel space
                fov_ypos_pix': " "
                ml_pos: transformed, rotated coords
                ap_pos: transformed, rotated coors (natural view)
            'zimg': 
                (array) z-projected image 
            'roi_contours': 
                (list) roi contours, classifications.convert_coords.contours_from_masks()
            'xlim' and 'ylim': 
                (float) FOV limits (in pixels or um) for (natural) azimuth and elevation axes
    '''


    print("... getting fov info")
    # Get masks
    npix_y, npix_x, nrois_total = masks.shape

    if roi_list is None:
        roi_list = range(nrois_total)

    # Create contours from maskL
    roi_contours = contours_from_masks(masks)

    # Convert to brain coords (scale to microns)
    fov_pos_x, fov_pos_y, xlim, ylim, centroids = get_roi_position_in_fov(roi_contours,
                                                               roi_list=roi_list,
                                                                 convert_um=convert_um,
                                                                 npix_y=npix_y,
                                                                 npix_x=npix_x)

    #posdf = pd.DataFrame({'ml_pos': fov_pos_y, 'ap_pos': fov_pos_x, #fov_x,
    posdf = pd.DataFrame({'fov_xpos': fov_pos_x, # corresponds to AP axis ('ap_pos')
                          'fov_ypos': fov_pos_y, # corresponds to ML axis ('ml_pos')
                          'fov_xpos_pix': [c[0] for c in centroids],
                          'fov_ypos_pix': [c[1] for c in centroids]
                          }, index=roi_list)

    posdf = transform_fov_posdf(posdf, ml_lim=ylim, ap_lim=xlim)
    # Save fov info
    pixel_size = p3.get_pixel_size()
    fovinfo = {'zimg': zimg,
                'convert_um': convert_um,
                'pixel_size': pixel_size,
                'roi_contours': roi_contours,
                'roi_positions': posdf,
                'ap_lim': xlim, # natural orientation AP (since 2p fov is rotated 90d)
                'ml_lim': ylim} # natural orientation ML

    return fovinfo


def contours_from_masks(masks):
    # Cycle through all ROIs and get their edges
    # (note:  tried doing this on sum of all ROIs, but fails if ROIs are overlapping at all)
    tmp_roi_contours = []
    for ridx in range(masks.shape[-1]):
        im = masks[:,:,ridx]
        im[im>0] = 1
        im[im==0] = np.nan #1
        im = im.astype('uint8')
        edged = cv2.Canny(im, 0, 0.9)
        tmp_cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        tmp_cnts = tmp_cnts[0] if imutils.is_cv2() else tmp_cnts[1]
        tmp_roi_contours.append((ridx, tmp_cnts[0]))
    print("Created %i contours for rois." % len(tmp_roi_contours))

    return tmp_roi_contours


def get_roi_position_in_fov(tmp_roi_contours, roi_list=None,
                            convert_um=True, npix_y=512, npix_x=512):
                            #xaxis_conversion=2.3, yaxis_conversion=1.9):

    '''
    From 20190605 PSF measurement:
        xaxis_conversion = 2.312
        yaxis_conversion = 1.904
    '''
    print("... (not sorting)")

    if not convert_um:
        xaxis_conversion = 1.
        yaxis_conversion = 1.
    else:
        (xaxis_conversion, yaxis_conversion) = p3.get_pixel_size()

    # Get ROI centroids:
    #print(tmp_roi_contours[0])
    centroids = [get_contour_center(cnt[1]) for cnt in tmp_roi_contours]

    # Convert pixels to um:
    xlinspace = np.linspace(0, npix_x*xaxis_conversion, num=npix_x)
    ylinspace = np.linspace(0, npix_y*yaxis_conversion, num=npix_y)

    xlim=xlinspace.max() if convert_um else npix_x
    ylim = ylinspace.max() if convert_um else npix_y

    if roi_list is None:
        roi_list = [cnt[1] for cnt in tmp_roi_contours] #range(len(tmp_roi_contours)) #sorted_roi_indices_xaxis))
        #print(roi_list[0:10])

    fov_pos_x = [xlinspace[pos[0]] for pos in centroids]
    fov_pos_y = [ylinspace[pos[1]] for pos in centroids]
    
    return fov_pos_x, fov_pos_y, xlim, ylim, centroids

def get_contour_center(cnt):
    cnt =(cnt).astype(np.float32)
    M = cv2.moments(cnt)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return (cX, cY)

# ############################################
# Functions for processing ROIs (masks)
# ############################################
def transform_rotate_coordinates(positions, ap_lim=1177., ml_lim=972.):
    # Rotate 90 degrees (ie., rotate counter clockwise about origin: (-y, x))
    # For image, where 0 is at top (y-axis points downward), 
    # then rotate CLOCKWISE, i.e., (y, -x)
    # Flip L/R, too.
    # (y, -x):  (pos[1], 512-pos[0]) --> 512 to make it non-neg, and align to image
    # flip l/r: (512-pos[1], ...) --> flips x-axis l/r 
    positions_t = [(ml_lim-pos[1], ap_lim-pos[0]) for pos in positions]

    return positions_t

def transform_fov_posdf(posdf, fov_keys=('fov_xpos', 'fov_ypos'),
                         ml_lim=972, ap_lim=1177.):
    posdf_transf = posdf.copy()

    fov_xkey, fov_ykey = fov_keys
    fov_xpos = posdf_transf[fov_xkey].values
    fov_ypos = posdf_transf[fov_ykey].values

    o_coords = [(xv, yv) for xv, yv in zip(fov_xpos, fov_ypos)]
    t_coords = transform_rotate_coordinates(o_coords, ap_lim=ap_lim, ml_lim=ml_lim)
    posdf['ml_pos'] = [t[0] for t in t_coords]
    posdf['ap_pos'] = [t[1] for t in t_coords]

    return posdf

