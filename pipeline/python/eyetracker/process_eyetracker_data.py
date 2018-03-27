#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 2018

@author: cesarechavarria
"""
import matplotlib
matplotlib.use('Agg')
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
sns.set_style("darkgrid", {"axes.facecolor": ".9"})
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
        if target_idx < var_array.size:
            ind_pre = get_interp_ind(target_idx, interp_sites, -1)
            ind_post = get_interp_ind(target_idx, interp_sites, 1)
        
            x_good = np.array([ind_pre,ind_post])
            y_good = np.hstack((var_array[ind_pre],var_array[ind_post]))
            interpF = interpolate.interp1d(x_good, y_good,1)
            new_value=interpF(target_idx)
        else:
            new_value = var_array[target_idx-1]
       

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


def make_excluded_variable_plot(df, time, feature, info, blink_times, figure_file, line_width = 0.5):
    stim_bar_loc = df[feature].min() - 1
    star_loc = df[feature].max() + 1

    grid = sns.FacetGrid(df, aspect=15)
    grid.map(plt.plot, time, feature, linewidth=line_width)

    for ax in grid.axes.flat:
        for ntrial in range(len(info)):
            trial_string = 'trial%05d'%(ntrial+1)
            if info[trial_string]['include_trial'] :
                ax.plot([info[trial_string]['stim_on_times']/1E3,info[trial_string]['stim_off_times']/1E3],np.array([1,1])*stim_bar_loc, 'k', label=None)
            else:
                ax.plot([info[trial_string]['stim_on_times']/1E3,info[trial_string]['stim_off_times']/1E3],np.array([1,1])*stim_bar_loc, 'r', label=None)
        if len(blink_times)>0:
            ax.plot(blink_times, np.ones((len(blink_times),))*star_loc,'r*')#start blink events

    sns.despine(trim=True, offset=1)
    pl.savefig(figure_file, bbox_inches='tight')
    pl.close()
    
def make_variable_histogram(df,feature, figure_file):
    sns.distplot(df[feature]);
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
    parser.add_option('--retinobar', action='store_true', dest='retinobar', default=False, help="Boolean flag to indicate this is a retionotopy-style run")

    parser.add_option('-p', '--pupil', action='store', dest='pupil_thresh', default=None, help='manual pupil threshold')
    parser.add_option('-c', '--cornea', action='store', dest='cr_thresh', default=None, help='manual corneal reflection threshold')


    parser.add_option('-m', '--movie', action='store_true', dest='make_movie', default=True, help='Boolean to indicate whether to make anotated movie of frames')

    parser.add_option('-d', '--downsample', action='store', dest='downsample', default=None, help='Factor by which to downsample images (integer)---not implemented yet')
    parser.add_option('-f', '--smooth', action='store', dest='space_filt_size', default=5, help='size of box filter for smoothing images [default: 5 (int)]')

    parser.add_option('-t', '--timefilt', action='store', dest='time_filt_size', default=5, help='Size of median filter to smooth signals over time [default: 5 (int)]')

    parser.add_option('-b', '--baseline', action='store', dest='baseline', default=1, help='Length of baseline period (secs) for trial parsing')

    parser.add_option('--default', action='store_true', dest='default', default='store_false', help="Use all DEFAULT params, for params not specified by user (prevent interactive)")

    (options, args) = parser.parse_args(options)


    return options

def process_data(options):

    # # Set USER INPUT options:
    rootdir = options.rootdir
    animalid = options.animalid
    session = options.session
    acquisition = options.acquisition
    run = options.run
    retinobar = options.retinobar

    pupil_thresh = options.pupil_thresh
    cr_thresh = options.cr_thresh

    make_movie = options.make_movie

    downsample = options.downsample
    space_filt_size = options.space_filt_size
    if space_filt_size is not None:
        space_filt_size = int(space_filt_size)

    time_filt_size =  options.time_filt_size

    #***unpack some options***
    downsample_factor = None#hard-code,for now since it seems to alter ability to track upupil
    if downsample_factor is None:
        scale_factor = 1
    else:
        scale_factor = float(1.0/downsample_factor)



    #****define input directories***
    run_dir = os.path.join(rootdir, animalid, session, acquisition, run)
    raw_folder = [r for r in os.listdir(run_dir) if 'raw' in r and os.path.isdir(os.path.join(run_dir, r))][0]
    print 'Raw folder: %s'%(raw_folder)

    eye_root_dir = os.path.join(run_dir,raw_folder,'eyetracker_files')

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
        print 'Data already processed....'
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
            if pupil_thresh is None:
                pupil_thresh = np.mean(im0[pupil_y1:pupil_y2,pupil_x1:pupil_x2])
            else:
                pupil_thresh = int(pupil_thresh)
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
            if cr_thresh is None:
                cr_thresh = np.mean(im0[cr_y1:cr_y2,cr_x1:cr_x2])
            else:
                cr_thresh = int(cr_thresh)
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
                
                pupil_ratio = np.true_divide(pupil_axes[0],pupil_axes[1])
                if pupil_center[0]==0 or pupil_ratio <=.6 or pupil_ratio>(1.0/.6):#flag this frame, probably blinking
                    flag_event[im_count] = 1
                
                
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
                    if flag_event[im_count] == 0:
                        cv2.ellipse(im_disp, ellipse_params_disp,(0,0,255),1)
                    else:
                        cv2.ellipse(im_disp, ellipse_params_disp,(0,255,0),1)
                
                if flag_event[im_count] == 0:
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
                    
                    #save to array
                    pupil_center_list[im_count,:] = pupil_center
                    pupil_axes_list[im_count,:] = pupil_axes
                    pupil_orientation_list[im_count] = pupil_orientation
                    
                if im_count >10:    
                    if sum(flag_event[im_count-10:im_count])>= 5:
                        #back to beginning with latest size
                        pupil_x1 = int(pupil_x1_orig-(10*scale_factor))
                        pupil_y1 = int(pupil_y1_orig-(10*scale_factor))
                        pupil_x2 = int(pupil_x2_orig+(10*scale_factor))
                        pupil_y2 = int(pupil_y2_orig+(10*scale_factor))

                        
                    else:
                        #give yourself room for error after event
                        if flag_event[im_count-1]<1:
                            pupil_x1 = int(pupil_x1-(5*scale_factor))
                            pupil_y1 = int(pupil_y1-(5*scale_factor))
                            pupil_x2 = int(pupil_x2+(5*scale_factor))
                            pupil_y2 = int(pupil_y2+(5*scale_factor))


            if 'cr' in user_rect:
                #get features
                cr_center, cr_axes, cr_orientation = get_feature_info(im0, (cr_x1,cr_y1), (cr_x2,cr_y2),\
                                                                               cr_thresh, 'cr')
                
                cr_ratio = np.true_divide(cr_axes[0],cr_axes[1])
                if cr_center[0]==0 or cr_ratio <=.6:#flag this frame, probably blinking
                    flag_event[im_count] = 1
                
                
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
                    if flag_event[im_count] == 0:
                        cv2.ellipse(im_disp, ellipse_params_disp,(255,0,0),1)
                    else:
                        cv2.ellipse(im_disp, ellipse_params_disp,(0,255,0),1)
                
                if flag_event[im_count] == 0:
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
                    
                    #save to array
                    cr_center_list[im_count,:] = cr_center
                    cr_axes_list[im_count,:] = cr_axes
                    cr_orientation_list[im_count] = cr_orientation
                    
                if im_count >10:    
                    if sum(flag_event[im_count-10:im_count])>= 5:
                        #back to beginning with latest size
                        cr_x1 = int(cr_x1_orig-(10*scale_factor))
                        cr_y1 = int(cr_y1_orig-(10*scale_factor))
                        cr_x2 = int(cr_x2_orig+(10*scale_factor))
                        cr_y2 = int(cr_y2_orig+(10*scale_factor))

                        
                    else:
                        #give yourself room for error after event
                        if flag_event[im_count-1]<1:
                            cr_x1 = int(cr_x1-(5*scale_factor))
                            cr_y1 = int(cr_y1-(5*scale_factor))
                            cr_x2 = int(cr_x2+(5*scale_factor))
                            cr_y2 = int(cr_y2+(5*scale_factor))

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
        if retinobar:
          start_time_abs = trial_info['1']['start_time_ms']
          for ntrial in range(len((trial_info))):
                trial_string = '%d'%(ntrial+1)
                stim_on_times[ntrial]=(trial_info[trial_string]['start_time_ms']-start_time_abs)/1E3#convert to ms
                stim_off_times[ntrial]=(trial_info[trial_string]['end_time_ms']-start_time_abs)/1E3#convert to ms
        else:
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

            #radius - 1st deriv
            tmp = np.diff(pupil_radius)
            pupil_radius_diff1 = np.hstack((0,tmp))

            #radius - 2nd deriv
            tmp = np.diff(pupil_radius_diff1)
            pupil_radius_diff2 = np.hstack((0,tmp))
            
            #pupil aspect ratio
            tmp = np.true_divide(pupil_axes_list[:,0],pupil_axes_list[:,1])#collapse across ellipse axes
            pupil_aspect = get_nice_signal(tmp, flag_event, time_filt_size)#clean up data a bit

            #pupil orientation
            tmp = pupil_orientation_list[:]
            pupil_orientation = get_nice_signal(tmp, flag_event, time_filt_size)#clean up data a bit

            # pupil position X 
            tmp = pupil_center_list[:,1].copy()
            pupil_pos_x = np.squeeze(get_nice_signal(tmp, flag_event, time_filt_size))#clean up data a bit

            # pupil position Y
            tmp = pupil_center_list[:,0].copy()
            pupil_pos_y = np.squeeze(get_nice_signal(tmp, flag_event, time_filt_size))#clean up data a bit

            #distance
            tmp_pos = np.transpose(np.vstack((pupil_pos_x,pupil_pos_y)))
            pupil_dist = np.squeeze(spatial.distance.cdist(tmp_pos,tmp_pos[0:1,:]))

            #distance - 1st deriv
            tmp = np.diff(pupil_dist)
            pupil_dist_diff1 = np.hstack((0,tmp))
            
            #disance
        if 'cr' in user_rect:
            #pixel radius
            tmp = np.mean(cr_axes_list,1)#collapse across ellipse axes
            cr_radius = get_nice_signal(tmp, flag_event, time_filt_size)#clean up data a bit

            #radius - 1st deriv
            tmp = np.diff(cr_radius)
            cr_radius_diff1 = np.hstack((0,tmp))

            #radius - 2nd deriv
            tmp = np.diff(cr_radius_diff1)
            cr_radius_diff2 = np.hstack((0,tmp))
            
            #cr aspect ratio
            tmp = np.true_divide(cr_axes_list[:,0],cr_axes_list[:,1])#collapse across ellipse axes
            cr_aspect = get_nice_signal(tmp, flag_event, time_filt_size)#clean up data a bit

            #cr orientation
            tmp = cr_orientation_list[:]
            cr_orientation = get_nice_signal(tmp, flag_event, time_filt_size)#clean up data a bit

            # cr position X 
            tmp = cr_center_list[:,1].copy()
            cr_pos_x = np.squeeze(get_nice_signal(tmp, flag_event, time_filt_size))#clean up data a bit

            # cr position Y
            tmp = cr_center_list[:,0].copy()
            cr_pos_y = np.squeeze(get_nice_signal(tmp, flag_event, time_filt_size))#clean up data a bit

            #distance
            tmp_pos = np.transpose(np.vstack((cr_pos_x,cr_pos_y)))
            cr_dist = np.squeeze(spatial.distance.cdist(tmp_pos,tmp_pos[0:1,:]))

            #distance - 1st deriv
            tmp = np.diff(cr_dist)
            cr_dist_diff1 = np.hstack((0,tmp))
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

        if len(blink_times)>0:
            if 'blink_times' not in file_grp.keys():
                blink_set = file_grp.create_dataset('blink_times',blink_times.shape, blink_times.dtype)
            blink_set[...] = blink_times


        if 'pupil' in user_rect:
            if 'pupil_radius' not in file_grp.keys():
                pup_rad_set = file_grp.create_dataset('pupil_radius',pupil_radius.shape, pupil_radius.dtype)
            pup_rad_set[...] = pupil_radius

            if 'pupil_radius_diff1' not in file_grp.keys():
                pup_rad1_set = file_grp.create_dataset('pupil_radius_diff1',pupil_radius_diff1.shape, pupil_radius_diff1.dtype)
            pup_rad1_set[...] = pupil_radius_diff1
            
            if 'pupil_radius_diff2' not in file_grp.keys():
                pup_rad2_set = file_grp.create_dataset('pupil_radius_diff2',pupil_radius_diff2.shape, pupil_radius_diff2.dtype)
            pup_rad2_set[...] = pupil_radius_diff2
            

            if 'pupil_orientation' not in file_grp.keys():
                pup_ori_set = file_grp.create_dataset('pupil_orientation',pupil_orientation.shape, pupil_orientation.dtype)
            pup_ori_set[...] = pupil_orientation

            if 'pupil_aspect' not in file_grp.keys():
                pup_asp_set = file_grp.create_dataset('pupil_aspect',pupil_aspect.shape, pupil_aspect.dtype)
            pup_asp_set[...] = pupil_aspect

            if 'pupil_distance' not in file_grp.keys():
                pup_dist_set = file_grp.create_dataset('pupil_distance',pupil_dist.shape, pupil_dist.dtype)
            pup_dist_set[...] = pupil_dist

            if 'pupil_dist_diff1' not in file_grp.keys():
                pup_dist1_set = file_grp.create_dataset('pupil_dist_diff1',pupil_dist_diff1.shape, pupil_dist_diff1.dtype)
            pup_dist1_set[...] = pupil_dist_diff1
            
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

            if 'cr_radius_diff1' not in file_grp.keys():
                cr_rad1_set = file_grp.create_dataset('cr_radius_diff1',cr_radius_diff1.shape, cr_radius_diff1.dtype)
            cr_rad1_set[...] = cr_radius_diff1
            
            if 'cr_radius_diff2' not in file_grp.keys():
                cr_rad2_set = file_grp.create_dataset('cr_radius_diff2',cr_radius_diff2.shape, cr_radius_diff2.dtype)
            cr_rad2_set[...] = cr_radius_diff2
            

            if 'cr_orientation' not in file_grp.keys():
                cr_ori_set = file_grp.create_dataset('cr_orientation',cr_orientation.shape, cr_orientation.dtype)
            cr_ori_set[...] = cr_orientation

            if 'cr_aspect' not in file_grp.keys():
                cr_asp_set = file_grp.create_dataset('cr_aspect',cr_aspect.shape, cr_aspect.dtype)
            cr_asp_set[...] = cr_aspect

            if 'cr_distance' not in file_grp.keys():
                cr_dist_set = file_grp.create_dataset('cr_distance',cr_dist.shape, cr_dist.dtype)
            cr_dist_set[...] = cr_dist

            if 'cr_dist_diff1' not in file_grp.keys():
                cr_dist1_set = file_grp.create_dataset('cr_dist_diff1',cr_dist_diff1.shape, cr_dist_diff1.dtype)
            cr_dist1_set[...] = cr_dist_diff1
            
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
            pupil_fig_dir = os.path.join(output_fig_dir,'pupil')
            if not os.path.exists(pupil_fig_dir):
                os.makedirs(pupil_fig_dir)

            pupil_df = pd.DataFrame({'camera time': camera_time,
                                     'pupil radius': pupil_radius,
                                     'pupil radius diff1': pupil_radius_diff1,
                                     'pupil radius diff2': pupil_radius_diff2,
                                     'pupil aspect ratio': pupil_aspect,
                                     'pupil orientation': pupil_orientation,             
                                     'pupil distance': pupil_dist,
                                     'pupil distance diff1': pupil_dist_diff1,
                                     'pupil position x': pupil_pos_x,
                                     'pupil position y': pupil_pos_y,
                                    })
            
            fig_file = os.path.join(pupil_fig_dir,'pupil_radius_%s_%s_%s.png'%(session,animalid,run))
            make_variable_plot(pupil_df, 'camera time', 'pupil radius',stim_on_times,stim_off_times, blink_times, fig_file)
            
            fig_file = os.path.join(pupil_fig_dir,'pupil_radius_hist_%s_%s_%s.png'%(session,animalid,run))
            make_variable_histogram(pupil_df, 'pupil radius', fig_file)

            fig_file = os.path.join(pupil_fig_dir,'pupil_radius_diff1_%s_%s_%s.png'%(session,animalid,run))
            make_variable_plot(pupil_df, 'camera time', 'pupil radius diff1',stim_on_times,stim_off_times, blink_times, fig_file)
            
            fig_file = os.path.join(pupil_fig_dir,'pupil_radius_diff1_hist_%s_%s_%s.png'%(session,animalid,run))
            make_variable_histogram(pupil_df, 'pupil radius diff1', fig_file)

            fig_file = os.path.join(pupil_fig_dir,'pupil_radius_diff2_%s_%s_%s.png'%(session,animalid,run))
            make_variable_plot(pupil_df, 'camera time', 'pupil radius diff2',stim_on_times,stim_off_times, blink_times, fig_file)
            
            fig_file = os.path.join(pupil_fig_dir,'pupil_radius_diff2_hist_%s_%s_%s.png'%(session,animalid,run))
            make_variable_histogram(pupil_df, 'pupil radius diff2', fig_file)

            fig_file = os.path.join(pupil_fig_dir,'pupil_orientation_%s_%s_%s.png'%(session,animalid,run))
            make_variable_plot(pupil_df, 'camera time', 'pupil orientation',stim_on_times,stim_off_times, blink_times, fig_file)
            
            fig_file = os.path.join(pupil_fig_dir,'pupil_orientation_hist_%s_%s_%s.png'%(session,animalid,run))
            make_variable_histogram(pupil_df, 'pupil orientation', fig_file)
            
            fig_file = os.path.join(pupil_fig_dir,'pupil_aspect_ratio_%s_%s_%s.png'%(session,animalid,run))
            make_variable_plot(pupil_df, 'camera time', 'pupil aspect ratio',stim_on_times,stim_off_times, blink_times, fig_file)
            
            fig_file = os.path.join(pupil_fig_dir,'pupil_aspect_ratio_hist_%s_%s_%s.png'%(session,animalid,run))
            make_variable_histogram(pupil_df, 'pupil aspect ratio', fig_file)

            fig_file = os.path.join(pupil_fig_dir,'pupil_distance_%s_%s_%s.png'%(session,animalid,run))
            make_variable_plot(pupil_df, 'camera time', 'pupil distance',stim_on_times,stim_off_times, blink_times, fig_file)
            
            fig_file = os.path.join(pupil_fig_dir,'pupil_distance_hist_%s_%s_%s.png'%(session,animalid,run))
            make_variable_histogram(pupil_df, 'pupil distance', fig_file)

            fig_file = os.path.join(pupil_fig_dir,'pupil_distance_diff1_%s_%s_%s.png'%(session,animalid,run))
            make_variable_plot(pupil_df, 'camera time', 'pupil distance diff1',stim_on_times,stim_off_times, blink_times, fig_file)
            
            fig_file = os.path.join(pupil_fig_dir,'pupil_distance_diff1_hist_%s_%s_%s.png'%(session,animalid,run))
            make_variable_histogram(pupil_df, 'pupil distance diff1', fig_file)
            
            fig_file = os.path.join(pupil_fig_dir,'pupil_x_position_%s_%s_%s.png'%(session,animalid,run))
            make_variable_plot(pupil_df, 'camera time', 'pupil position x',stim_on_times,stim_off_times, blink_times, fig_file)
            
            fig_file = os.path.join(pupil_fig_dir,'pupil_x_position_hist_%s_%s_%s.png'%(session,animalid,run))
            make_variable_histogram(pupil_df, 'pupil position x', fig_file)
            
            fig_file = os.path.join(pupil_fig_dir,'pupil_y_position_%s_%s_%s.png'%(session,animalid,run))
            make_variable_plot(pupil_df, 'camera time', 'pupil position y',stim_on_times,stim_off_times, blink_times, fig_file)
             
            fig_file = os.path.join(pupil_fig_dir,'pupil_y_position_hist_%s_%s_%s.png'%(session,animalid,run))
            make_variable_histogram(pupil_df, 'pupil position y', fig_file)
        if 'cr' in user_rect:

            cr_fig_dir = os.path.join(output_fig_dir,'cr')
            if not os.path.exists(cr_fig_dir):
                os.makedirs(cr_fig_dir)

            cr_df = pd.DataFrame({'camera time': camera_time,
                                     'cr radius': cr_radius,
                                     'cr radius diff1': cr_radius_diff1,
                                     'cr radius diff2': cr_radius_diff2,
                                     'cr aspect ratio': cr_aspect,
                                     'cr orientation': cr_orientation,             
                                     'cr distance': cr_dist,
                                     'cr distance diff1': cr_dist_diff1,
                                     'cr position x': cr_pos_x,
                                     'cr position y': cr_pos_y,
                                    })
            
            fig_file = os.path.join(cr_fig_dir,'cr_radius_%s_%s_%s.png'%(session,animalid,run))
            make_variable_plot(cr_df, 'camera time', 'cr radius',stim_on_times,stim_off_times, blink_times, fig_file)
            
            fig_file = os.path.join(cr_fig_dir,'cr_radius_hist_%s_%s_%s.png'%(session,animalid,run))
            make_variable_histogram(cr_df, 'cr radius', fig_file)

            fig_file = os.path.join(cr_fig_dir,'cr_radius_diff1_%s_%s_%s.png'%(session,animalid,run))
            make_variable_plot(cr_df, 'camera time', 'cr radius diff1',stim_on_times,stim_off_times, blink_times, fig_file)
            
            fig_file = os.path.join(cr_fig_dir,'cr_radius_diff1_hist_%s_%s_%s.png'%(session,animalid,run))
            make_variable_histogram(cr_df, 'cr radius diff1', fig_file)

            fig_file = os.path.join(cr_fig_dir,'cr_radius_diff2_%s_%s_%s.png'%(session,animalid,run))
            make_variable_plot(cr_df, 'camera time', 'cr radius diff2',stim_on_times,stim_off_times, blink_times, fig_file)
            
            fig_file = os.path.join(cr_fig_dir,'cr_radius_diff2_hist_%s_%s_%s.png'%(session,animalid,run))
            make_variable_histogram(cr_df, 'cr radius diff2', fig_file)

            fig_file = os.path.join(cr_fig_dir,'cr_orientation_%s_%s_%s.png'%(session,animalid,run))
            make_variable_plot(cr_df, 'camera time', 'cr orientation',stim_on_times,stim_off_times, blink_times, fig_file)
            
            fig_file = os.path.join(cr_fig_dir,'cr_orientation_hist_%s_%s_%s.png'%(session,animalid,run))
            make_variable_histogram(cr_df, 'cr orientation', fig_file)
            
            fig_file = os.path.join(cr_fig_dir,'cr_aspect_ratio_%s_%s_%s.png'%(session,animalid,run))
            make_variable_plot(cr_df, 'camera time', 'cr aspect ratio',stim_on_times,stim_off_times, blink_times, fig_file)
            
            fig_file = os.path.join(cr_fig_dir,'cr_aspect_ratio_hist_%s_%s_%s.png'%(session,animalid,run))
            make_variable_histogram(cr_df, 'cr aspect ratio', fig_file)

            fig_file = os.path.join(cr_fig_dir,'cr_distance_%s_%s_%s.png'%(session,animalid,run))
            make_variable_plot(cr_df, 'camera time', 'cr distance',stim_on_times,stim_off_times, blink_times, fig_file)
            
            fig_file = os.path.join(cr_fig_dir,'cr_distance_hist_%s_%s_%s.png'%(session,animalid,run))
            make_variable_histogram(cr_df, 'cr distance', fig_file)

            fig_file = os.path.join(cr_fig_dir,'cr_distance_diff1_%s_%s_%s.png'%(session,animalid,run))
            make_variable_plot(cr_df, 'camera time', 'cr distance diff1',stim_on_times,stim_off_times, blink_times, fig_file)
            
            fig_file = os.path.join(cr_fig_dir,'cr_distance_diff1_hist_%s_%s_%s.png'%(session,animalid,run))
            make_variable_histogram(cr_df, 'cr distance diff1', fig_file)
            
            fig_file = os.path.join(cr_fig_dir,'cr_x_position_%s_%s_%s.png'%(session,animalid,run))
            make_variable_plot(cr_df, 'camera time', 'cr position x',stim_on_times,stim_off_times, blink_times, fig_file)
            
            fig_file = os.path.join(cr_fig_dir,'cr_x_position_hist_%s_%s_%s.png'%(session,animalid,run))
            make_variable_histogram(cr_df, 'cr position x', fig_file)
            
            fig_file = os.path.join(cr_fig_dir,'cr_y_position_%s_%s_%s.png'%(session,animalid,run))
            make_variable_plot(cr_df, 'camera time', 'cr position y',stim_on_times,stim_off_times, blink_times, fig_file)
             
            fig_file = os.path.join(cr_fig_dir,'cr_y_position_hist_%s_%s_%s.png'%(session,animalid,run))
            make_variable_histogram(cr_df, 'cr position y', fig_file)

def parse_data(options):

    # # Set USER INPUT options:
    rootdir = options.rootdir
    animalid = options.animalid
    session = options.session
    acquisition = options.acquisition
    run = options.run

    baseline_time = options.baseline

    #define input directories
    run_dir = os.path.join(rootdir, animalid, session, acquisition, run)

    input_root_dir = os.path.join(run_dir,'eyetracker')
    input_file_dir = os.path.join(input_root_dir,'files')


    #paradigm details
    para_file_dir = os.path.join(run_dir,'paradigm','files')
    para_file =  [f for f in os.listdir(para_file_dir) if f.endswith('.json')][0]#assuming a single file for all tiffs in run


    #make output directories
    output_root_dir = input_root_dir
    print 'Output Directory: %s'%(output_root_dir)

    output_file_dir = input_file_dir

    output_fig_dir = os.path.join(output_root_dir,'figures','parsed')
    if not os.path.exists(output_fig_dir):
        os.makedirs(output_fig_dir)

    #***load and unpack features***
    input_fn = os.path.join(input_file_dir,'full_session_eyetracker_data_%s_%s_%s.h5'%(session,animalid,run))

    print 'Loading eyetracker feature info from :%s'%(input_fn)

    file_grp = h5py.File(input_fn, 'r')#open file

    frame_rate = float(file_grp.attrs['frame_rate'])

    camera_time = file_grp['camera_time'][:]
    blink_events = file_grp['blink_events'][:]
    if 'blink_Times' not in file_grp.keys():
        blink_times = file_grp['blink_times'][:]
    pupil_radius = file_grp['pupil_radius'][:]
    pupil_radius_diff1 = file_grp['pupil_radius_diff1'][:]
    pupil_radius_diff2 = file_grp['pupil_radius_diff2'][:]
    pupil_x = file_grp['pupil_x'][:]
    pupil_y = file_grp['pupil_y'][:]
    pupil_dist = file_grp['pupil_distance'][:]
    pupil_dist_diff1 = file_grp['pupil_dist_diff1'][:]
    pupil_aspect = file_grp['pupil_aspect'][:]
    pupil_orientation = file_grp['pupil_orientation'][:]

    file_grp.close()


    #****get trial info times***
    print 'Getting paradigm info from: %s'%(os.path.join(para_file_dir, para_file))
    with open(os.path.join(para_file_dir, para_file), 'r') as f:
        trial_info = json.load(f)


    baseline_frames = int(baseline_time*frame_rate)

    iti_full_time = trial_info['trial00001']['iti_duration']/1E3#for pasing traces  
    iti_post_time = iti_full_time - baseline_time
    iti_post_frames = int(iti_post_time*frame_rate)

    stim_on_time = trial_info['trial00001']['stim_on_times']/1E3#convert to secs
    stim_off_time = trial_info['trial00001']['stim_off_times']/1E3#convert to sec
    stim_dur_time = stim_off_time - stim_on_time
    stim_dur_frames = int(stim_dur_time*frame_rate)

    post_onset_frames = stim_dur_frames+iti_post_frames

    trial_time = np.arange(0,(1.0/frame_rate)*(baseline_frames+stim_dur_frames+iti_post_frames),1/frame_rate) - baseline_time

    #***Parse trace by trial ***
    eye_info = dict()

    pup_rad_mat = []
    pup_dist_mat = []
    fig, ax = pl.subplots()
    fig2, ax2 = pl.subplots()


    for ntrial in range(0, len((trial_info))):
        if ntrial%100 == 0:
            print 'Parsing trial %d of %d'%(ntrial,len(trial_info))
        trial_string = 'trial%05d'%(ntrial+1)
        
        #copy some details from paradigm file
        eye_info[trial_string] = dict()
        eye_info[trial_string]['stimuli'] = trial_info[trial_string]['stimuli']
        eye_info[trial_string]['stim_on_times'] = trial_info[trial_string]['stim_on_times']
        eye_info[trial_string]['stim_off_times'] = trial_info[trial_string]['stim_off_times']
        eye_info[trial_string]['iti_duration'] = trial_info[trial_string]['iti_duration']
        
        #get times and indices of relevent events
        stim_on_time = trial_info[trial_string]['stim_on_times']/1E3#convert to ms
        on_idx = np.where(camera_time>=stim_on_time)[0][0]
        start_idx = on_idx - baseline_frames
        end_idx = on_idx + post_onset_frames
        off_idx = on_idx + stim_dur_frames

        eye_info[trial_string]['start_idx'] = start_idx
        eye_info[trial_string]['on_idx'] = on_idx
        eye_info[trial_string]['end_idx'] = end_idx
        eye_info[trial_string]['off_idx'] = off_idx


        #get some feature values for stimulation and baseline periods
        pupil_sz_baseline = np.mean(pupil_dist[start_idx:on_idx])
        eye_info[trial_string]['pupil_size_stim'] = np.mean(pupil_radius[on_idx:off_idx])
        eye_info[trial_string]['pupil_size_stim_min'] = np.min(pupil_radius[on_idx:off_idx])
        eye_info[trial_string]['pupil_size_stim_max'] = np.max(pupil_radius[on_idx:off_idx])
        eye_info[trial_string]['pupil_size_baseline'] = np.mean(pupil_radius[start_idx:on_idx])
        eye_info[trial_string]['pupil_size_baseline_min'] = np.min(pupil_radius[start_idx:on_idx])
        eye_info[trial_string]['pupil_size_baseline_max'] = np.max(pupil_radius[start_idx:on_idx])

        eye_info[trial_string]['pupil_size_diff1_stim'] = np.mean(pupil_radius_diff1[on_idx:off_idx])
        eye_info[trial_string]['pupil_size_diff1_stim_min'] = np.min(pupil_radius_diff1[on_idx:off_idx])
        eye_info[trial_string]['pupil_size_diff1_stim_max'] = np.max(pupil_radius_diff1[on_idx:off_idx])
        eye_info[trial_string]['pupil_size_diff1_baseline'] = np.mean(pupil_radius_diff1[start_idx:on_idx])
        eye_info[trial_string]['pupil_size_diff1_baseline_min'] = np.min(pupil_radius_diff1[start_idx:on_idx])
        eye_info[trial_string]['pupil_size_diff1_baseline_max'] = np.max(pupil_radius_diff1[start_idx:on_idx])
        
        pupil_dist_baseline = np.mean(pupil_dist[start_idx:on_idx])
        eye_info[trial_string]['pupil_dist_stim'] = np.mean(pupil_dist[on_idx:off_idx])
        eye_info[trial_string]['pupil_dist_stim_min'] = np.min(pupil_dist[on_idx:off_idx])
        eye_info[trial_string]['pupil_dist_stim_max'] = np.max(pupil_dist[on_idx:off_idx])
        eye_info[trial_string]['pupil_dist_baseline'] = np.mean(pupil_dist[start_idx:on_idx])
        eye_info[trial_string]['pupil_dist_baseline_min'] = np.min(pupil_dist[start_idx:on_idx])
        eye_info[trial_string]['pupil_dist_baseline_max'] = np.max(pupil_dist[start_idx:on_idx])

        eye_info[trial_string]['pupil_dist_diff1_stim'] = np.mean(pupil_dist_diff1[on_idx:off_idx])
        eye_info[trial_string]['pupil_dist_diff1_stim_min'] = np.min(pupil_dist_diff1[on_idx:off_idx])
        eye_info[trial_string]['pupil_dist_diff1_stim_max'] = np.max(pupil_dist_diff1[on_idx:off_idx])
        eye_info[trial_string]['pupil_dist_diff1_baseline'] = np.mean(pupil_dist_diff1[start_idx:on_idx])
        eye_info[trial_string]['pupil_dist_diff1_baseline_min'] = np.min(pupil_dist_diff1[start_idx:on_idx])
        eye_info[trial_string]['pupil_dist_diff1_baseline_max'] = np.max(pupil_dist_diff1[start_idx:on_idx])

        eye_info[trial_string]['pupil_x_stim'] = np.mean(pupil_x[on_idx:off_idx])
        eye_info[trial_string]['pupil_x_stim_min'] = np.min(pupil_x[on_idx:off_idx])
        eye_info[trial_string]['pupil_x_stim_max'] = np.max(pupil_x[on_idx:off_idx])
        eye_info[trial_string]['pupil_x_baseline'] = np.mean(pupil_x[start_idx:on_idx])
        eye_info[trial_string]['pupil_x_baseline_min'] = np.min(pupil_x[start_idx:on_idx])
        eye_info[trial_string]['pupil_x_baseline_max'] = np.max(pupil_x[start_idx:on_idx])

        eye_info[trial_string]['pupil_y_stim'] = np.mean(pupil_y[on_idx:off_idx])
        eye_info[trial_string]['pupil_y_stim_min'] = np.min(pupil_y[on_idx:off_idx])
        eye_info[trial_string]['pupil_y_stim_max'] = np.max(pupil_y[on_idx:off_idx])
        eye_info[trial_string]['pupil_y_baseline'] = np.mean(pupil_y[start_idx:on_idx])
        eye_info[trial_string]['pupil_y_baseline_min'] = np.min(pupil_y[start_idx:on_idx])
        eye_info[trial_string]['pupil_y_baseline_max'] = np.max(pupil_y[start_idx:on_idx])

        eye_info[trial_string]['blink_event_count_stim'] = np.sum(blink_events[on_idx:off_idx])
        eye_info[trial_string]['blink_event_count_baseline'] = np.sum(blink_events[start_idx:on_idx])

        eye_info[trial_string]['pupil_ratio_stim'] = np.mean(pupil_aspect[on_idx:off_idx])
        eye_info[trial_string]['pupil_ratio_stim_min'] = np.min(pupil_aspect[on_idx:off_idx])
        eye_info[trial_string]['pupil_ratio_stim_max'] = np.max(pupil_aspect[on_idx:off_idx])
        eye_info[trial_string]['pupil_ratio_baseline'] = np.mean(pupil_aspect[start_idx:on_idx])
        eye_info[trial_string]['pupil_ratio_baseline_min'] = np.min(pupil_aspect[start_idx:on_idx])
        eye_info[trial_string]['pupil_ratio_baseline_max'] = np.max(pupil_aspect[start_idx:on_idx])

        ax.plot(trial_time, pupil_radius[start_idx:end_idx]-pupil_sz_baseline,'k',alpha =0.1,linewidth = 0.5)
        pup_rad_mat.append(pupil_radius[start_idx:end_idx]-pupil_sz_baseline)
        
        ax2.plot(trial_time, pupil_dist[start_idx:end_idx]-pupil_dist_baseline,'k',alpha =0.1,linewidth = 0.5)
        pup_dist_mat.append(pupil_dist[start_idx:end_idx]-pupil_dist_baseline) 

   

    print 'Saving figures to: %s' % (output_fig_dir)
    ax.plot(trial_time, np.nanmean(pup_rad_mat,0),'k',alpha=1)
    ymin, ymax = ax.get_ylim()
    ax.axvline(x=0, ymin=ymin, ymax = ymax, linewidth=1, color='k',linestyle='--')
    ax.set_xlabel('Time ASO',fontsize=16)
    ax.set_ylabel('Pupil Radius',fontsize=16)
    sns.despine(offset=2, trim=True)

    fig_file = os.path.join(output_fig_dir,'parsed_pupil_size_%s_%s_%s.png'%(session,animalid,run))
    fig.savefig(fig_file, bbox_inches='tight')
    plt.close()

    ax2.plot(trial_time, np.nanmean(pup_dist_mat,0),'k',alpha=1)
    ymin, ymax = ax2.get_ylim()
    ax2.axvline(x=0, ymin=ymin, ymax = ymax, linewidth=1, color='k',linestyle='--')
    ax2.set_xlabel('Time ASO',fontsize=16)
    ax2.set_ylabel('Pupil Distance',fontsize=16)
    sns.despine(offset=2, trim=True)

    fig_file = os.path.join(output_fig_dir,'parsed_pupil_distance_%s_%s_%s.png'%(session,animalid,run))
    fig2.savefig(fig_file, bbox_inches='tight')
    pl.close()

    #save info to file
    output_fn = 'parsed_eye_%s_%s_%s.json'%(session,animalid,run)

    print 'Saving parsed eye info to: %s' % (os.path.join(output_file_dir,output_fn))

    with open(os.path.join(output_file_dir, output_fn), 'w') as f:
        trial_info = json.dump(eye_info,f)

#-----------------------------------------------------
#           MAIN SET OF ACTIONS
#-----------------------------------------------------

def main(options):

    options = extract_options(options)
    print 'Processing raw frames'
    process_data(options)

    if not options.retinobar:
        print 'Parsing eye features by trial'
        parse_data(options)


if __name__ == '__main__':
    main(sys.argv[1:])



