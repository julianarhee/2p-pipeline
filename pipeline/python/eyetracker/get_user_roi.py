# import the necessary packages
import cv2
import optparse
import json
import os
import numpy as np
from scipy import misc,interpolate,stats,signal, spatial, ndimage
import glob
from matplotlib import pyplot as plt
import time
import pickle

import re

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

# construct the argument parser and parse the arguments
parser = optparse.OptionParser()

# PATH opts:
parser.add_option('-R', '--root', action='store', dest='rootdir', default='/nas/volume1/2photon/data', help='data root dir (root project dir containing all animalids) [default: /nas/volume1/2photon/data, /n/coxfs01/2pdata if --slurm]')
parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')
parser.add_option('-S', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID')
parser.add_option('-A', '--acq', action='store', dest='acquisition', default='FOV1', help="acquisition folder (ex: 'FOV1_zoom3x') [default: FOV1]")
parser.add_option('-r', '--run', action='store', dest='run', default='', help="name of run dir containing tiffs to be processed (ex: gratings_phasemod_run1)")

(options, args) = parser.parse_args()

rootdir = options.rootdir
animalid = options.animalid
session = options.session
acquisition = options.acquisition
run = options.run

def save_obj(obj, name ):
    with open(name, 'w') as outfile:
        json.dump(obj, outfile)

#FUNCTIONS
def read_image(im_file,flip_flag=0,cv_flag=0):
    if cv_flag:
        im0 = cv2.imread(im_file)
    else:
        im0 = misc.imread(im_file)
    if flip_flag:
        im0 = cv2.flip(im0,1)

    if np.size(np.shape(im0)) > 2:
        im0=np.uint8(np.true_divide(im0,np.max(im0))*255)
    else:
        szY,szX=np.shape(im0)
        im0=np.uint8(np.true_divide(im0,np.max(im0))*255)
        im0=np.dstack((im0,im0,im0))

    return im0
 
# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
boxes = []

 
def click_and_draw(event, x, y, flags, param):
    # grab references to the global variables
    global boxes
    global window
    global display_image
 
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        #clear previous box
        display_image = clone.copy()
        boxes=[]
        #start new box
        print 'Start Mouse Position: '+str(x)+', '+str(y)
        sbox = [y, x]
        boxes.append(sbox)
        # print count
        # print sbox
    elif event == cv2.EVENT_LBUTTONUP:
          print 'End Mouse Position: '+str(x)+', '+str(y)
          ebox = [y, x]
          boxes.append(ebox)
          rect_pt1 = tuple((boxes[-2][1],boxes[-2][0]))
          rect_pt2 = tuple((boxes[-1][1],boxes[-1][0]))
          cv2.rectangle(display_image, rect_pt1, rect_pt2, (0, 255, 0), 2)
      #    k =  cv2.waitKey(0)
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

def mark_pupil(im0,user_rect,pupil_thresh = None):
    im_disp = np.copy(im0)
    im0= cv2.boxFilter(im0,0, (5, 5), normalize = 1)

    pupil_x1 = int(min([user_rect['pupil'][0][0],user_rect['pupil'][1][0]]))
    pupil_x2 = int(max([user_rect['pupil'][0][0],user_rect['pupil'][1][0]]))
    pupil_y1 = int(min([user_rect['pupil'][0][1],user_rect['pupil'][1][1]]))
    pupil_y2 = int(max([user_rect['pupil'][0][1],user_rect['pupil'][1][1]])) 

    if pupil_thresh is None:
        pupil_thresh = int(np.mean(im0[pupil_y1:pupil_y2,pupil_x1:pupil_x2]))
    else:
        pupil_thresh = int(pupil_thresh)
    print 'threshold value for pupil: %10.4f'%(pupil_thresh)

    pupil_center, pupil_axes, pupil_orientation = get_feature_info(im0, (pupil_x1,pupil_y1), (pupil_x2,pupil_y2),\
                                                                   pupil_thresh, 'pupil')



    #draw and save to file
    ellipse_params = tuple((pupil_center,pupil_axes,pupil_orientation))
    cv2.rectangle(im_disp,(pupil_x1, pupil_y1),(pupil_x2, pupil_y2),(0,255,255),1)
    cv2.ellipse(im_disp, ellipse_params,(0,0,255),1)

    return im_disp, pupil_thresh

def mark_cr(im0,user_rect,cr_thresh = None):
    im_disp = np.copy(im0)
    im0= cv2.boxFilter(im0,0, (5, 5), normalize = 1)

    cr_x1 = int(min([user_rect['cr'][0][0],user_rect['cr'][1][0]]))
    cr_x2 = int(max([user_rect['cr'][0][0],user_rect['cr'][1][0]]))
    cr_y1 = int(min([user_rect['cr'][0][1],user_rect['cr'][1][1]]))
    cr_y2 = int(max([user_rect['cr'][0][1],user_rect['cr'][1][1]])) 

    if cr_thresh is None:
        cr_thresh = int(np.mean(im0[cr_y1:cr_y2,cr_x1:cr_x2]))
    else:
        cr_thresh = int(cr_thresh)
    print 'threshold value for cr: %10.4f'%(cr_thresh)

    cr_center, cr_axes, cr_orientation = get_feature_info(im0, (cr_x1,cr_y1), (cr_x2,cr_y2),\
                                                                   cr_thresh, 'cr')



    #draw and save to file
    ellipse_params = tuple((cr_center,cr_axes,cr_orientation))
    cv2.rectangle(im_disp,(cr_x1, cr_y1),(cr_x2, cr_y2),(255,255,0),1)
    cv2.ellipse(im_disp, ellipse_params,(255,0,0),1)

    return im_disp, cr_thresh

        


#figure out input directories
run_dir = os.path.join(rootdir, animalid, session, acquisition, run)
raw_folder = [r for r in os.listdir(run_dir) if 'raw' in r and os.path.isdir(os.path.join(run_dir, r))][0]
print 'Raw folder: %s'%(raw_folder)

eye_root_dir = os.path.join(run_dir,raw_folder,'eyetracker_files')
file_folder = os.listdir(eye_root_dir)[0]
img_folder = os.path.join(eye_root_dir,file_folder,'frames')

#get list of images in folder
img_list = [name for name in os.listdir(img_folder) if os.path.isfile(os.path.join(img_folder, name))]
sort_nicely(img_list)

#pick first image as reference image
img_file = os.path.join(img_folder,img_list[0])

# load the image, clone it, and setup the mouse callback function
img = read_image(img_file,0)


clone = img.copy()
display_image = clone.copy()
roi_rect = dict()
window = 1
cv2.namedWindow("source image")
cv2.setMouseCallback("source image", click_and_draw)

print '******* PUPIL ROI ********'
print 'Use mouse to draw box around pupil'
print 'Press R to reset image/discard box'
print 'Press P to record pupil ROI'
print 'Press Q to exit'
while True:
    cv2.imshow("source image", display_image)
    key = cv2.waitKey(1) & 0xFF
 
    # if the 'r' key is pressed, reset image
    if key == ord("r"):
        display_image = clone.copy()
        boxes=[]
 
#   if the 'p' key is pressed, save point as pupil seed
    elif key == ord("p"):
        rect_pt1 = tuple((boxes[-2][1],boxes[-2][0]))
        rect_pt2 = tuple((boxes[-1][1],boxes[-1][0]))
        roi_rect['pupil'] = [rect_pt1, rect_pt2]
        print('Pupil ROI Recorded')
        boxes = []
        display_image = clone.copy()
        break
    # if the 'q' key is pressed, exit loop and close windows
    elif key == ord("q"):
        break
 
if 'pupil' in roi_rect.keys():
    print '******* PUPIL THRESHOLD ********'
    print 'Press W to increase pupil threshold'
    print 'Press S to decrease pupil threshold'
    print 'Press P to accept and record pupil threshold'


    marked_image, thresh = mark_pupil(display_image,roi_rect)

    while True:
        cv2.imshow("source image", marked_image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("w"):#increase threshold
            thresh = thresh+1
            if thresh >100:
                thresh = 100
            marked_image, thresh = mark_pupil(display_image,roi_rect,thresh)
        elif key == ord("s"):#decrease threshol
            thresh = thresh-1
            if thresh<0:
                thresh = 0
            marked_image, thresh = mark_pupil(display_image,roi_rect,thresh)
        elif key == ord("p"):
            roi_rect['pupil_thresh'] = thresh
            print 'Pupil Treshold Recorded!'
            break
        elif key == ord("q"):
            break
    
print '******* CORNEAL REFLECTION ROI ********'
print 'Use mouse to draw box around pupil'
print 'Press R to reset image/discard box'
print 'Press C to record corneal reflection ROI'
print 'Press Q to exit'
while True:
    cv2.imshow("source image", display_image)
    key = cv2.waitKey(1) & 0xFF
 
    # if the 'r' key is pressed, reset image
    if key == ord("r"):
        display_image = clone.copy()
        boxes=[]
 
#   if the 'p' key is pressed, save point as pupil seed
    elif key == ord("c"):
        rect_pt1 = tuple((boxes[-2][1],boxes[-2][0]))
        rect_pt2 = tuple((boxes[-1][1],boxes[-1][0]))
        roi_rect['cr'] = [rect_pt1, rect_pt2]
        print('Corneal Reflection ROI Recorded!')
        boxes = []
        display_image = clone.copy()
        break
    # if the 'q' key is pressed, exit loop and close windows
    elif key == ord("q"):
        break

if 'cr' in roi_rect.keys():
    print '******* PUPIL THRESHOLD ********'
    print 'Press W to increase cr threshold'
    print 'Press S to decrease pupil threshold'
    print 'Press C to accept and record pupil threshold'


    marked_image, thresh = mark_cr(display_image,roi_rect)

    while True:
        cv2.imshow("source image", marked_image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("w"):#increase threshold
            thresh = thresh+1
            if thresh >255:
                thresh = 255
            marked_image, thresh = mark_cr(display_image,roi_rect,thresh)
        elif key == ord("s"):#decrease threshol
            thresh = thresh-1
            if thresh<0:
                thresh = 0
            marked_image, thresh = mark_cr(display_image,roi_rect,thresh)
        elif key == ord("c"):
            roi_rect['cr_thresh'] = thresh
            print 'Corneal Reflection Treshold Recorded!'
            break
        elif key == ord("q"):
            break
    
#make output directories
output_root_dir = os.path.join(run_dir,'eyetracker')
if not os.path.exists(output_root_dir):
    os.makedirs(output_root_dir)

output_file_dir = os.path.join(output_root_dir,'files')
if not os.path.exists(output_file_dir):
    os.makedirs(output_file_dir)

output_file = os.path.join(output_file_dir,'user_restriction_box.json')
save_obj(roi_rect, output_file)
print('ROI info Saved!')

print('Closing Windows!')
cv2.destroyAllWindows()

