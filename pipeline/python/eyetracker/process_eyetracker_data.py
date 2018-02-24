#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 2018

@author: cesarechavarria
"""

import cv2
import os
import sys
import optparse
import time
import shutil
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc,interpolate,stats,signal, spatial, ndimage
import json
import re
import pylab as pl
import seaborn as sns
import pandas as pd
import h5py

#-----------------------------------------------------
#          SOME FUNCTIONS FOR VARIOUS PARTS
#-----------------------------------------------------

#miscellaneous functions
def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)
    
def load_obj(name):
    with open(name, 'r') as r:
        fileinfo = json.load(r)
    return fileinfo
    
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def block_mean(im0, fact):
    im1 = cv2.boxFilter(im0,0, (fact, fact), normalize = 1)
    im2 = cv2.resize(im1,None,fx=1.0/fact, fy=1.0/fact, interpolation = cv2.INTER_CUBIC)
    return im2
    
    assert isinstance(fact, int), type(fact)
    sx, sy = ar.shape
    X, Y = np.ogrid[0:sx, 0:sy]
    regions = sy/fact * (X/fact) + Y/fact
    res = ndimage.mean(ar, labels=regions, index=np.arange(regions.max() + 1))
    res.shape = (sx/fact, sy/fact)
    return res



#load frame time info
def get_frame_rate(relevant_dir):
    # READ IN FRAME TIMES FILE
    pfile=open(os.path.join(relevant_dir,'performance.txt'))

    #READ HEADERS AND FIND RELEVANT COLUMNS
    headers=pfile.readline()
    headers=headers.split()

    count = 0
    while count < len(headers):
        if headers[count]=='frame_rate':
            rate_idx=count
            break
        count = count + 1

    #read just first line
    for line in pfile:
        x=line.split()
        frame_rate = x[rate_idx]
        break
    pfile.close()
    return float(frame_rate)

def get_frame_attribute(relevant_dir,attr_string):
    #get frame-by-frame details
    # READ IN FRAME TIMES FILE
    pfile=open(os.path.join(relevant_dir,'frame_times.txt'))

    #READ HEADERS AND FIND RELEVANT COLUMNS
    headers=pfile.readline()
    headers=headers.split()

    count = 0
    while count < len(headers):
        if headers[count]== attr_string:
            sync_idx=count
            break
        count = count + 1

    frame_attr=[]
    # GET DESIRED DATA
    for line in pfile:
        x=line.split()
        frame_attr.append(x[sync_idx])

    frame_attr=np.array(map(float,frame_attr))
    pfile.close()

    return frame_attr

#image processing functions
def get_feature_info(process_img, box_pt1, box_pt2, feature_thresh, target_feature='pupil',criterion='area'):
    #get grayscale
    if process_img.ndim>2:
        process_img=np.mean(process_img,2)
    #apply restriction box 
    process_img = process_img[box_pt1[1]:box_pt2[1],box_pt1[0]:box_pt2[0]]

    #threshold
    img_roi = np.zeros(process_img.shape)
    if target_feature == 'pupil':
        thresh_array =process_img<feature_thresh
    else:
        thresh_array =process_img>feature_thresh
    if np.sum(thresh_array)>0:#continue if some points have passed threshold
        if criterion == 'area':
            #look for largest area
            labeled, nr_objects = ndimage.label(thresh_array) 
            pix_area = np.zeros((nr_objects,))
            for i in range(nr_objects):
                pix_area[i] = len(np.where(labeled==i+1)[0])
            img_roi[labeled == (np.argmax(pix_area)+1)]=255
            img_roi = img_roi.astype('uint8')
        else:
            #look for region closes to center of box
            x = np.linspace(-1, 1, process_img.shape[1])
            y = np.linspace(-1, 1, process_img.shape[0])
            xv, yv = np.meshgrid(x, y)

            [radius,theta]=cart2pol(xv,yv)

            img_roi = np.zeros(process_img.shape)
            labeled, nr_objects = ndimage.label(thresh_array) 
            pix_distance = np.zeros((nr_objects,))
            for i in range(nr_objects):
                pix_distance[i] = np.min((labeled==i+1)*radius)
            img_roi[labeled == (np.argmin(pix_distance)+1)]=255
            img_roi = img_roi.astype('uint8')

        #get contours
        tmp, contours, hierarchy = cv2.findContours(img_roi,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        #find contour with most points
        pt_array =np.zeros((len(contours),))
        for count,cnt in enumerate(contours):
            pt_array[count] = len(cnt)
        if len(contours)>0 and np.max(pt_array)>=5:#otherwise ellipse fit will fail
            elp_idx = np.argmax(pt_array)

            #fit ellipse
            elp = cv2.fitEllipse(contours[elp_idx])

            #unpack values
            elp_center = tuple((elp[0][0]+box_pt1[0], elp[0][1]+box_pt1[1]))
            elp_axes = elp[1]
            elp_orientation = elp[2]
            return elp_center, elp_axes, elp_orientation
        else:
            return tuple((0,0)), tuple((0,0)), 0
    else:
        return tuple((0,0)), tuple((0,0)), 0
 

def get_interp_ind(idx, interp_sites, step):
    redo = 0
    interp_idx = idx + step
    if interp_idx in interp_sites:
        redo = 1
    while redo:
        redo = 0
        interp_idx = interp_idx + step
        if interp_idx in interp_sites:
            redo = 1
    return interp_idx
        
    
def interpolate_sites(var_array, interp_sites):
    array_interp = np.copy(var_array)
    for target_idx in interp_sites:
        ind_pre = get_interp_ind(target_idx, interp_sites, -1)
        ind_post = get_interp_ind(target_idx, interp_sites, 1)
        x_good = np.array([ind_pre,ind_post])
        y_good = np.hstack((var_array[ind_pre],var_array[ind_post]))

        interpF = interpolate.interp1d(x_good, y_good,1)
        new_value=interpF(target_idx)

        array_interp[target_idx]=new_value
    return array_interp

def get_nice_signal(var_array, interp_flag, filt_kernel=11):
    #get sites to interpolate
    interp_list1 = np.where(var_array == 0)[0]
    interp_list2 = np.where(interp_flag== 1)[0]
    interp_sites = np.concatenate((interp_list1,interp_list2))
    interp_sites = np.unique(interp_sites)
    
    #get interpolated list
    var_interp = interpolate_sites(var_array, interp_sites)
    
    #do some medial filtering
    nice_array = signal.medfilt(var_interp, filt_kernel)
    return nice_array

def make_variable_plot(df, time, feature, stim_on_times, stim_off_times, blink_times, figure_file, line_width = 0.5):
    stim_bar_loc = df[feature].min() - 1
    star_loc = df[feature].max() + 1

    grid = sns.FacetGrid(df, aspect=15)
    grid.map(plt.plot, time, feature, linewidth=line_width)

    for ax in grid.axes.flat:
        for trial in range(len(stim_on_times)):
            ax.plot([stim_on_times[trial],stim_off_times[trial]],np.array([1,1])*stim_bar_loc, 'k', label=None)

        if len(blink_times)>0:
            ax.plot(blink_times, np.ones((len(blink_times),))*star_loc,'r*')#start blink events

    sns.despine(trim=True, offset=1)
    pl.savefig(figure_file, bbox_inches='tight')
    pl.close()

def extract_options(options):
    choices_sourcetype = ('raw', 'mcorrected', 'bidi')
    default_sourcetype = 'mcorrected'

    parser = optparse.OptionParser()

    # PATH opts:
    parser.add_option('-R', '--root', action='store', dest='rootdir', default='/nas/volume1/2photon/data', help='data root dir (root project dir containing all animalids) [default: /nas/volume1/2photon/data, /n/coxfs01/2pdata if --slurm]')
    parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')
    parser.add_option('-S', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID')
    parser.add_option('-A', '--acq', action='store', dest='acquisition', default='FOV1', help="acquisition folder (ex: 'FOV1_zoom3x') [default: FOV1]")
    parser.add_option('-r', '--run', action='store', dest='run', default='', help="name of run dir containing tiffs to be processed (ex: gratings_phasemod_run1)")

    parser.add_option('-m', '--movie', action='store_true', dest='make_movie', default='store_true', help='Boolean to indicate whether to make anotated movie of frames')

    parser.add_option('-d', '--downsample', action='store', dest='downsample', default=None, help='Factor by which to downsample images (integer)---not implemented yet')
    parser.add_option('-f', '--smooth', action='store', dest='space_filt_size', default=None, help='size of box filter for smoothing images(integer)')

    parser.add_option('-t', '--timefilt', action='store', dest='time_filt_size', default=5, help='Size of median filter to smooth signals over time(integer)')


    parser.add_option('--default', action='store_true', dest='default', default='store_false', help="Use all DEFAULT params, for params not specified by user (prevent interactive)")

    (options, args) = parser.parse_args(options)


    return options

def process_data(options):
    options = extract_options(options)

    # # Set USER INPUT options:
    rootdir = options.rootdir
    animalid = options.animalid
    session = options.session
    acquisition = options.acquisition
    run = options.run

    make_movie = options.make_movie

    downsample = options.downsample
    space_filt_size = options.space_filt_size

    time_filt_size = options.time_filt_size

    #***unpack some options***
    downsample_factor = None#hard-code,for now since it seems to alter ability to track upupil
    if downsample_factor is None:
        scale_factor = 1
    else:
        scale_factor = float(1.0/downsample_factor)



    #****define input directories***
    run_dir = os.path.join(rootdir, animalid, session, acquisition, run)

    eye_root_dir = os.path.join(run_dir,'raw','eyetracker_files')
    file_folder = os.listdir(eye_root_dir)[0]

    im_dir = os.path.join(eye_root_dir,file_folder,'frames')
    times_dir = os.path.join(eye_root_dir,file_folder,'times')

    #paradigm details
    para_file_dir = os.path.join(run_dir,'paradigm','files')
    para_file =  [f for f in os.listdir(para_file_dir) if f.endswith('.json')][0]#assuming a single file for all tiffs in run

    #make output directories
    output_root_dir = os.path.join(run_dir,'eyetracker')
    if not os.path.exists(output_root_dir):
        os.makedirs(output_root_dir)
    print 'Output Directory: %s'%(output_root_dir)

    output_file_dir = os.path.join(output_root_dir,'files')

    output_fig_dir = os.path.join(output_root_dir,'figures')
    if not os.path.exists(output_fig_dir):
        os.makedirs(output_fig_dir)

    output_mov_dir = os.path.join(output_root_dir,'movies')
    if not os.path.exists(output_mov_dir):
        os.makedirs(output_mov_dir)

    #****check if we have already processed, and save the data***
    output_fn = os.path.join(output_file_dir,'full_session_eyetracker_data_%s_%s_%s.h5'%(session,animalid,run))
    if os.path.isfile(output_fn):
        #load and continue onto trial parsing
        print 'already processed'
    else:
        if make_movie:
            tmp_dir = os.path.join(output_mov_dir, 'tmp')
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)
            os.makedirs(tmp_dir)


        #get relevant image list
        im_list = [name for name in os.listdir(im_dir) if os.path.isfile(os.path.join(im_dir, name))]
        sort_nicely(im_list)
        im0 = cv2.imread(os.path.join(im_dir,im_list[0]))

        #load user-specificed restriciton box
        user_rect = load_obj(os.path.join(output_file_dir,'user_restriction_box.json'))

        if 'pupil' in user_rect:
            pupil_x1_orig = int(min([user_rect['pupil'][0][0],user_rect['pupil'][1][0]])*scale_factor)
            pupil_x2_orig = int(max([user_rect['pupil'][0][0],user_rect['pupil'][1][0]])*scale_factor)
            pupil_y1_orig = int(min([user_rect['pupil'][0][1],user_rect['pupil'][1][1]])*scale_factor)
            pupil_y2_orig = int(max([user_rect['pupil'][0][1],user_rect['pupil'][1][1]])*scale_factor) 
            
            pupil_x1 = pupil_x1_orig
            pupil_y1 = pupil_y1_orig
            pupil_x2 = pupil_x2_orig
            pupil_y2 = pupil_y2_orig

            pupil_thresh = np.mean(im0[pupil_y1:pupil_y2,pupil_x1:pupil_x2])
            print 'threshold value for pupil: %10.4f'%(pupil_thresh)
            
        if 'cr' in user_rect:
            cr_x1_orig = int(min([user_rect['cr'][0][0],user_rect['cr'][1][0]])*scale_factor)
            cr_x2_orig = int(max([user_rect['cr'][0][0],user_rect['cr'][1][0]])*scale_factor)
            cr_y1_orig = int(min([user_rect['cr'][0][1],user_rect['cr'][1][1]])*scale_factor)
            cr_y2_orig = int(max([user_rect['cr'][0][1],user_rect['cr'][1][1]])*scale_factor) 
            
            cr_x1 = cr_x1_orig
            cr_y1 = cr_y1_orig
            cr_x2 = cr_x2_orig
            cr_y2 = cr_y2_orig

            cr_thresh = np.mean(im0[cr_y1:cr_y2,cr_x1:cr_x2])
            print 'threshold value for corneal reflection: %10.4f'%(cr_thresh)

        #make empty arrays
        if 'pupil' in user_rect:
            pupil_center_list = np.zeros((len(im_list),2))
            pupil_axes_list = np.zeros((len(im_list),2))
            pupil_orientation_list = np.zeros((len(im_list),))
        if 'cr' in user_rect:
            cr_center_list = np.zeros((len(im_list),2))
            cr_axes_list = np.zeros((len(im_list),2))
            cr_orientation_list = np.zeros((len(im_list),))
        flag_event = np.zeros((len(im_list),))

        for im_count in range(len(im_list)):
            #display count
            if im_count%1000==0:
                print 'Processing Image %d of %d....' %(im_count,len(im_list))
            
            #load image
            im0 = cv2.imread(os.path.join(im_dir,im_list[im_count]))
            im_disp = np.copy(im0)#save for drawing on, later
            if downsample_factor is not None:
                im0 = block_mean(im0, downsample_factor)
            if space_filt_size is not None:
                im0= cv2.boxFilter(im0,0, (space_filt_size, space_filt_size), normalize = 1)
            
            if 'pupil' in user_rect:
                #get features
                pupil_center, pupil_axes, pupil_orientation = get_feature_info(im0, (pupil_x1,pupil_y1), (pupil_x2,pupil_y2),\
                                                                               pupil_thresh, 'pupil')
                #save to array
                pupil_center_list[im_count,:] = pupil_center
                pupil_axes_list[im_count,:] = pupil_axes
                pupil_orientation_list[im_count] = pupil_orientation
                
                #draw and save to file
                ellipse_params = tuple((pupil_center,pupil_axes,pupil_orientation))
                if make_movie:
                    if downsample_factor is not None:
                        pupil_x1_disp = int(pupil_x1*downsample_factor)
                        pupil_x2_disp = int(pupil_x2*downsample_factor)
                        pupil_y1_disp = int(pupil_y1*downsample_factor)
                        pupil_y2_disp = int(pupil_y2*downsample_factor) 

                        ellipse_params_disp = tuple((tuple([downsample_factor*x for x in ellipse_params[0]]),tuple([downsample_factor*x for x in ellipse_params[1]]),downsample_factor*ellipse_params[2]))
                    else:
                        pupil_x1_disp = pupil_x1
                        pupil_x2_disp = pupil_x2
                        pupil_y1_disp = pupil_y1
                        pupil_y2_disp = pupil_y2
                        ellipse_params_disp = ellipse_params
                        
                    cv2.rectangle(im_disp,(pupil_x1_disp,pupil_y1_disp),(pupil_x2_disp,pupil_y2_disp),(0,255,255),1)
                    cv2.ellipse(im_disp, ellipse_params_disp,(0,0,255),1)
                
                if pupil_center[0]==0:#flag this frame, probably blinking
                    flag_event[im_count] = 1
                    #give yourself room for error after event
                    if flag_event[im_count-1]<1:
                        pupil_x1 = int(pupil_x1-(5*scale_factor))
                        pupil_y1 = int(pupil_y1-(5*scale_factor))
                        pupil_x2 = int(pupil_x2+(5*scale_factor))
                        pupil_y2 = int(pupil_y2+(5*scale_factor))
                else:
                     #adaptive part
                    dummy_img = np.zeros(im0.shape)
                    cv2.ellipse(dummy_img, ellipse_params,255,-1)
                    dummy_img = np.mean(dummy_img,2)
                    tmp, contours, hierarchy = cv2.findContours(dummy_img.astype('uint8'),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                    x,y,w,h = cv2.boundingRect(contours[0])
                    pupil_x1 = int(x-(3*scale_factor))
                    pupil_y1 = int(y-(3*scale_factor))
                    pupil_x2 = int(x+w+(3*scale_factor))
                    pupil_y2 = int(y+h+(3*scale_factor))

            if 'cr' in user_rect:
                #get features
                cr_center, cr_axes, cr_orientation = get_feature_info(im0, (cr_x1,cr_y1), (cr_x2,cr_y2),\
                                                                               cr_thresh, 'cr')
                #save to array
                cr_center_list[im_count,:] = cr_center
                cr_axes_list[im_count,:] = cr_axes
                cr_orientation_list[im_count] = cr_orientation
                
                #draw and save to file
                ellipse_params = tuple((cr_center,cr_axes,cr_orientation))
                if make_movie:
                    if downsample_factor is not None:
                        cr_x1_disp = int(cr_x1*downsample_factor)
                        cr_x2_disp = int(cr_x2*downsample_factor)
                        cr_y1_disp = int(cr_y1*downsample_factor)
                        cr_y2_disp = int(cr_y2*downsample_factor) 

                        ellipse_params_disp = tuple((tuple([downsample_factor*x for x in ellipse_params[0]]),tuple([downsample_factor*x for x in ellipse_params[1]]),downsample_factor*ellipse_params[2]))
                    else:
                        cr_x1_disp = cr_x1
                        cr_x2_disp = cr_x2
                        cr_y1_disp = cr_y1
                        cr_y2_disp = cr_y2
                        ellipse_params_disp = ellipse_params
                        
                    cv2.rectangle(im_disp,(cr_x1_disp,cr_y1_disp),(cr_x2_disp,cr_y2_disp),(255,255,0),1)
                    cv2.ellipse(im_disp, ellipse_params_disp,(255,0,0),1)
                
                if cr_center[0]==0:#flag this frame, probably blinking
                    flag_event[im_count] = 1
                    #give yourself room for error after event
                    if flag_event[im_count-1]<1:
                        cr_x1 = int(cr_x1-(5*scale_factor))
                        cr_y1 = int(cr_y1-(5*scale_factor))
                        cr_x2 = int(cr_x2+(5*scale_factor))
                        cr_y2 = int(cr_y2+(5*scale_factor))
                else:
                     #adaptive part
                    dummy_img = np.zeros(im0.shape)
                    cv2.ellipse(dummy_img, ellipse_params,255,-1)
                    dummy_img = np.mean(dummy_img,2)
                    tmp, contours, hierarchy = cv2.findContours(dummy_img.astype('uint8'),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                    x,y,w,h = cv2.boundingRect(contours[0])
                    cr_x1 = int(x-(3*scale_factor))
                    cr_y1 = int(y-(3*scale_factor))
                    cr_x2 = int(x+w+(3*scale_factor))
                    cr_y2 = int(y+h+(3*scale_factor))

            if make_movie: 
                cv2.imwrite((os.path.join(tmp_dir,im_list[im_count])), im_disp)

        #***get camera timestamps****
        frame_rate = get_frame_rate(times_dir)
        frame_period = 1.0/frame_rate
        frame_idx = get_frame_attribute(times_dir,'frame_number')
        #assume a steady frame rate
        camera_time = np.arange(0,frame_period*len(frame_idx),frame_period)
        #correct for unmatched vector lengths
        if camera_time.shape[0]>frame_idx.shape[0]:
            camera_time = np.delete(camera_time,-1)

        if make_movie:#
            video_file = os.path.join(output_mov_dir,'%s_%s_anotated_movie'%(session,animalid))
            cmd = 'ffmpeg -y -r %10.4f -i %s/%s.png -vcodec libx264 -f mp4 -pix_fmt yuv420p %s.mp4'%(frame_rate,tmp_dir,'%d',video_file)
            os.system(cmd)

            shutil.rmtree(tmp_dir)


        #****get trial info times***
        print 'Getting paradigm info from: %s'%(os.path.join(para_file_dir, para_file))
        with open(os.path.join(para_file_dir, para_file), 'r') as f:
            trial_info = json.load(f)

        stim_on_times = np.zeros((len(trial_info)))
        stim_off_times = np.zeros((len(trial_info)))
        for ntrial in range(len((trial_info))):
            trial_string = 'trial%05d'%(ntrial+1)
            stim_on_times[ntrial]=trial_info[trial_string]['stim_on_times']/1E3#convert to ms
            stim_off_times[ntrial]=trial_info[trial_string]['stim_off_times']/1E3#convert to ms

        #***get traces for relevant features**
        print 'Processing traces for extracted features'
        #blink times
        blink_events = np.where(flag_event==1)[0]#get indices of probable blink events
        if len(blink_events>0):
            blink_times = camera_time[blink_events]
        else:
            blink_times = []

        if 'pupil' in user_rect:
            #pixel radius
            tmp = np.mean(pupil_axes_list,1)#collapse across ellipse axes
            pupil_radius = get_nice_signal(tmp, flag_event, time_filt_size)#clean up data a bit

            # pupil position X 
            tmp = pupil_center_list[:,1].copy()
            pupil_pos_x = np.squeeze(get_nice_signal(tmp, flag_event, time_filt_size))#clean up data a bit

            # pupil position Y
            tmp = pupil_center_list[:,0].copy()
            pupil_pos_y = np.squeeze(get_nice_signal(tmp, flag_event, time_filt_size))#clean up data a bit

            #distance
            tmp_pos = np.transpose(np.vstack((pupil_pos_x,pupil_pos_y)))
            pupil_dist = np.squeeze(spatial.distance.cdist(tmp_pos,tmp_pos[0:1,:]))
        if 'cr' in user_rect:
            #pixel radius
            tmp = np.mean(cr_axes_list,1)#collapse across ellipse axes
            cr_radius = get_nice_signal(tmp, flag_event, time_filt_size)#clean up data a bit

            # cr position X 
            tmp = cr_center_list[:,1].copy()
            cr_pos_x = np.squeeze(get_nice_signal(tmp, flag_event, time_filt_size))#clean up data a bit

            # cr position Y
            tmp = cr_center_list[:,0].copy()
            cr_pos_y = np.squeeze(get_nice_signal(tmp, flag_event, time_filt_size))#clean up data a bit

            #cr distance
            tmp_pos = np.transpose(np.vstack((cr_pos_x,cr_pos_y)))
            cr_dist = np.squeeze((spatial.distance.cdist(tmp_pos,tmp_pos[0:1,:])))

        #**save complete traces**
        output_fn = os.path.join(output_file_dir,'full_session_eyetracker_data_%s_%s_%s.h5'%(session,animalid,run))

        print 'Saving feature info for the whole session in :%s'%(output_fn)

        file_grp = h5py.File(output_fn, 'w')#open file
        #save some general attributes
        file_grp.attrs['source_dir'] = im_dir
        file_grp.attrs['nframes'] = len(im_list)
        file_grp.attrs['frame_rate'] = frame_rate
        file_grp.attrs['time_filter_size'] = time_filt_size

        #define and store relevant features
        if 'camera_time' not in file_grp.keys():
            tset = file_grp.create_dataset('camera_time',camera_time.shape, camera_time.dtype)
        tset[...] = camera_time

        if 'blink_events' not in file_grp.keys():
            blink_ev_set = file_grp.create_dataset('blink_events',flag_event.shape, flag_event.dtype)
        blink_ev_set[...] = flag_event

        if 'blink_times' not in file_grp.keys():
            blink_set = file_grp.create_dataset('blink_times',blink_times.shape, blink_times.dtype)
        blink_set[...] = blink_times


        if 'pupil' in user_rect:
            if 'pupil_radius' not in file_grp.keys():
                pup_rad_set = file_grp.create_dataset('pupil_radius',pupil_radius.shape, pupil_radius.dtype)
            pup_rad_set[...] = pupil_radius

            if 'pupil_distance' not in file_grp.keys():
                pup_dist_set = file_grp.create_dataset('pupil_distance',pupil_dist.shape, pupil_dist.dtype)
            pup_dist_set[...] = pupil_dist

            if 'pupil_x' not in file_grp.keys():
                pup_x_set = file_grp.create_dataset('pupil_x',pupil_pos_x.shape, pupil_pos_x.dtype)
            pup_x_set[...] = pupil_pos_x

            if 'pupil_y' not in file_grp.keys():
                pup_y_set = file_grp.create_dataset('pupil_y',pupil_pos_y.shape, pupil_pos_y.dtype)
            pup_y_set[...] = pupil_pos_y

        if 'cr' in user_rect:
            if 'cr_radius' not in file_grp.keys():
                cr_rad_set = file_grp.create_dataset('cr_radius',cr_radius.shape, cr_radius.dtype)
            cr_rad_set[...] = cr_radius

            if 'cr_distance' not in file_grp.keys():
                cr_dist_set = file_grp.create_dataset('cr_distance',cr_dist.shape, cr_dist.dtype)
            cr_dist_set[...] = cr_dist

            if 'cr_x' not in file_grp.keys():
                cr_x_set = file_grp.create_dataset('cr_x',cr_pos_x.shape, cr_pos_x.dtype)
            cr_x_set[...] = cr_pos_x

            if 'cr_y' not in file_grp.keys():
                cr_y_set = file_grp.create_dataset('cr_y',cr_pos_y.shape, cr_pos_y.dtype)
            cr_y_set[...] = cr_pos_y
            
        file_grp.close()

        #***place feature info into dataframe and plot ***
        print 'Plotting extracted features for the full session, output folder: %s'%(output_fig_dir)
        if 'pupil' in user_rect:
            pupil_df = pd.DataFrame({'camera time': camera_time,
                                   'pupil radius': pupil_radius,
                                   'pupil distance': pupil_dist,
                                   'pupil position x': pupil_pos_x,
                                   'pupil position y': pupil_pos_y,
                                   })
            
            fig_file = os.path.join(output_fig_dir,'pupil_radius_%s_%s_%s.png'%(session,animalid,run))
            make_variable_plot(pupil_df, 'camera time', 'pupil radius',stim_on_times,stim_off_times, blink_times, fig_file)

            fig_file = os.path.join(output_fig_dir,'pupil_distance_%s_%s_%s.png'%(session,animalid,run))
            make_variable_plot(pupil_df, 'camera time', 'pupil distance',stim_on_times,stim_off_times, blink_times, fig_file)

            fig_file = os.path.join(output_fig_dir,'pupil_x_position_%s_%s_%s.png'%(session,animalid,run))
            make_variable_plot(pupil_df, 'camera time', 'pupil position x',stim_on_times,stim_off_times, blink_times, fig_file)

            fig_file = os.path.join(output_fig_dir,'pupil_y_position_%s_%s_%s.png'%(session,animalid,run))
            make_variable_plot(pupil_df, 'camera time', 'pupil position y',stim_on_times,stim_off_times, blink_times, fig_file)
        if 'cr' in user_rect:
            cr_df = pd.DataFrame({'camera time': camera_time,
                                   'cr radius': cr_radius,
                                   'cr distance': cr_dist,
                                   'cr position x': cr_pos_x,
                                   'cr position y': cr_pos_y,
                                   })
            
            fig_file = os.path.join(output_fig_dir,'cr_radius_%s_%s_%s.png'%(session,animalid,run))
            make_variable_plot(cr_df, 'camera time', 'cr radius',stim_on_times,stim_off_times, blink_times, fig_file)

            fig_file = os.path.join(output_fig_dir,'cr_distance_%s_%s_%s.png'%(session,animalid,run))
            make_variable_plot(cr_df, 'camera time', 'cr distance',stim_on_times,stim_off_times, blink_times, fig_file)

            fig_file = os.path.join(output_fig_dir,'cr_x_position_%s_%s_%s.png'%(session,animalid,run))
            make_variable_plot(cr_df, 'camera time', 'cr position x',stim_on_times,stim_off_times, blink_times, fig_file)

            fig_file = os.path.join(output_fig_dir,'cr_y_position_%s_%s_%s.png'%(session,animalid,run))
            make_variable_plot(cr_df, 'camera time', 'cr position y',stim_on_times,stim_off_times, blink_times, fig_file)


#-----------------------------------------------------
#           MAIN SET OF ACTIONS
#-----------------------------------------------------

def main(options):

    process_data(options)


if __name__ == '__main__':
    main(sys.argv[1:])


