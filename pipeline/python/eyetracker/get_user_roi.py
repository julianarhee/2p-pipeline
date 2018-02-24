# import the necessary packages
import cv2
import optparse
import json
import os
import numpy as np
from scipy import misc
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


#figure out input directories
run_dir = os.path.join(rootdir, animalid, session, acquisition, run)

eye_root_dir = os.path.join(run_dir,'raw','eyetracker_files')
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
 
# keep looping until the 'q' key is pressed
while True:
	# display the image and wait for a keypress
	cv2.imshow("source image", display_image)
	key = cv2.waitKey(1) & 0xFF
 
	# if the 'r' key is pressed, reset image
	if key == ord("r"):
		display_image = clone.copy()
		boxes=[]
 
#	if the 'p' key is pressed, save point as pupil seed
	elif key == ord("p"):
		rect_pt1 = tuple((boxes[-2][1],boxes[-2][0]))
		rect_pt2 = tuple((boxes[-1][1],boxes[-1][0]))
		roi_rect['pupil'] = [rect_pt1, rect_pt2]
		print('Pupil Restriction Box Recorded')
		boxes = []
		display_image = clone.copy()

	# # if the 'c' key is pressed, save point as corneal reflection seed
	elif key == ord("c"):
		rect_pt1 = tuple((boxes[-2][1],boxes[-2][0]))
		rect_pt2 = tuple((boxes[-1][1],boxes[-1][0]))
		roi_rect['cr'] = [rect_pt1, rect_pt2]
		print('Corneal Reflection Restriction Box Recorded')
		boxes = []
		display_image = clone.copy()
	# if the 's' key is pressed, save seed points to file
	elif key == ord("s"):
		#make output directories
		output_root_dir = os.path.join(run_dir,'eyetracker')
		if not os.path.exists(output_root_dir):
		    os.makedirs(output_root_dir)

		output_file_dir = os.path.join(output_root_dir,'files')
		if not os.path.exists(output_file_dir):
		    os.makedirs(output_file_dir)

		output_file = os.path.join(output_file_dir,'user_restriction_box.json')
		save_obj(roi_rect, output_file)
		print('Restriction Boxes Saved!')
		break


	# if the 'q' key is pressed, exit loop and close windows
	elif key == ord("q"):
		break

print('Closing Windows!')
cv2.destroyAllWindows()

