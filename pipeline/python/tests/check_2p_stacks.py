#!/usr/bin/env python2

import numpy as np
import os
from skimage.measure import block_reduce
from scipy.misc import imread
import cPickle as pkl
import scipy.signal
import numpy.fft as fft
import sys
import optparse
from libtiff import TIFF
from PIL import Image
import re
import itertools
from scipy import ndimage

import time
import datetime

import tifffile as tiff
import pandas as pd
import numpy.fft as fft

from bokeh.io import gridplot, output_file, show
from bokeh.plotting import figure
import csv

# User conda env:  retinodev


source_dir = '/media/juliana/Seagate Backup Plus Drive/RESDATA'
condition_dir = '20161222_JR030W_retinotopy2'
run_dir = 'fov1_bar037Hz_retinotopy_run2_00007'
slice_dir = 'ch1_slices'


# channel_names = ['green', 'red']
# channels = [fn for fn in os.listdir(source) for channel in channel_names if channel in fn and 'scaled' in fn]

# stack = dict()
# for cidx,channel in enumerate(channels):
#   channel_fn = os.path.join(source, channel)
#   stack[channel_names[cidx]] = tiff.imread(channel_fn)

# green = stack['green'] - stack['red'] # Get rid of red/autofluorescent junk

# new_tif_name = channels[0] + '_' + channels[1]
# tiff.imsave(os.path.join(source, new_tif_name), green)


# curr_roi_fn = os.path.join(source, 'Values.xls')
# curr_roi = pd.read_clipboard()
# y = curr_roi[curr_roi.keys()[1]]



# path = '/media/juliana/IMDATA/TEFO/20161218_CE025/processed/fov2_bar6_00002_CH1'
# path = '/media/juliana/IMDATA/TEFO/20161218_CE025/processed'

# fn = 'fov2_bar6_00002_values.csv'
# fn = 'fov2_bar6_00002_slice10_values.xls'
# fn = 'fov2_bar4_00003_CH1_slice15_ROI_smaller.xls'
# fn = 'fov2_bar4_00003_CH1_slice15_ROI_example.xls'



# roi_path = '/media/juliana/IMDATA/TEFO/20161218_CE025/fov2_bar5_00001_ch1/rois'
# roi_path = '/media/juliana/IMDATA/TEFO/20161218_CE025/fov2_bar5_00001_ch2/rois'
# roi_path = '/media/juliana/IMDATA/TEFO/20161218_CE025/fov2_bar6_00002_ch1/rois'

# roi_file = 'roi1.xls'


# CHANNEL 1 ----------------------------------------------------------
# ch1_roi_path = '/media/juliana/IMDATA/TEFO/20161218_CE025/fov2_bar5_00001_ch1/s15_subset_rois'
# slice_number = os.path.split(ch1_roi_path)[1]
# condition = os.path.split(os.path.split(ch1_roi_path)[0])[1]

# # ch1_roi_files = os.listdir(ch1_roi_path)
# # ch1_roi_files = [i for i in ch1_roi_files if '.xls' in i]
# # print "CH 1 - Checking %i ROIs in dir: roi_path" % len(ch1_roi_files)


# CHANNEL 2 ----------------------------------------------------------
# ch2_roi_path = '/media/juliana/IMDATA/TEFO/20161218_CE025/fov2_bar5_00001_ch2/s15_subset_rois'
# ch2_roi_files = os.listdir(ch2_roi_path)
# ch2_roi_files = [i for i in ch2_roi_files if '.xls' in i]
# print "CH2 - Checking %i ROIs in dir: roi_path" % len(ch2_roi_files)


# PATH TO SLICE ROIs: 
# slice_path = '/media/juliana/IMDATA/TEFO/20161218_CE025/fov2_bar6_00002'


# slice_path = '/media/juliana/IMDATA/TEFO/20161218_CE025/fov2_bar2_00002'
slice_path = '/media/juliana/IMDATA/TEFO/20161219_JR030W/fov6_037Hz_nomask_bar1_00004'

ch1_roi_path = os.path.join(slice_path, 'ch1_rois')
ch2_roi_path = os.path.join(slice_path, 'ch2_rois')


ch1_roi_files = os.listdir(ch1_roi_path)
ch1_roi_files = [i for i in ch1_roi_files if '.xls' in i]
print "CH 1 - Checking %i ROIs in dir: roi_path" % len(ch1_roi_files)

ch2_roi_files = os.listdir(ch2_roi_path)
ch2_roi_files = [i for i in ch2_roi_files if '.xls' in i]
print "CH2 - Checking %i ROIs in dir: roi_path" % len(ch2_roi_files)

slice_number = os.path.split(ch1_roi_path)[1]
condition = os.path.split(os.path.split(ch1_roi_path)[0])[1]


if not len(ch2_roi_files) == len(ch1_roi_files):
  print "Found different # so ROIs..."
else:
  nrois = len(ch1_roi_files)

# Stimulation Params -------------------------------------------------
rolling = True
nreps = 1290 #455 #1290 #455
target_freq = 0.13 # 0.37 #0.37 #0.13      # stim frequency in Hz
acquisition_rate = 5.58 # vol rate in Hz


nframes_per_cyc = (1/target_freq) * acquisition_rate
moving_win_sz = nframes_per_cyc * 2

channels = ['ch1', 'ch2']
paths = [ch1_roi_path, ch2_roi_path]

# for roi in range(len(ch1_roi_files)):

ROI = dict()
for channel in channels:
  curr_roi_path = [p for p in paths if channel in p][0]
  curr_roi_files = os.listdir(curr_roi_path)
  curr_roi_files = [f for f in curr_roi_files if '.xls' in f]

  ROI[channel] = dict()
  for roi in curr_roi_files:
    roi_name = roi.split('.')[0]
    cr = csv.reader(open(os.path.join(curr_roi_path, roi),"rb"), delimiter='\t')

    arr = range(nreps) #adjust to needed
    x = 0
    for ridx,row in enumerate(cr):
      if ridx==0:    
        continue
      else:
        arr[x] = row
        x += 1

    y = [float(a[1]) for a in np.array(arr)]

    if rolling is True:
        # pix_padded = [np.ones(moving_win_sz)*y[y.keys()[0]], y, np.ones(moving_win_sz)*y[y.keys()[-1]]]
        pix_padded = [np.ones(moving_win_sz)*y[0], y, np.ones(moving_win_sz)*y[-1]]
        tmp_pix = list(itertools.chain(*pix_padded))
        tmp_pix_rolling = np.convolve(tmp_pix, np.ones(moving_win_sz)/moving_win_sz, 'same')
        remove_pad = (len(tmp_pix_rolling) - len(y) ) / 2
        rpix = np.array(tmp_pix_rolling[remove_pad:-1*remove_pad])
        y -= rpix

    ft = fft.fft(y)

    ROI[channel][roi_name] = ft


# PLOTTING ------------------------------------------------------------------- 

ch1_plot_ys = []
ch2_plot_ys = []
roi_names = []
ridx = 0
for r_ch1,r_ch2 in zip(ROI['ch1'].keys(), ROI['ch2'].keys()):
  green = ROI['ch1'][r_ch1]
  red = ROI['ch2'][r_ch2]

  N = len(green)
  #sampling_rate = 5.58 #60. #4.11 #2.14
  ts = 1 / acquisition_rate
  freqs = fft.fftfreq(N, ts)
  ch1_power = np.abs(green)**2
  ch2_power = np.abs(red)**2

  # freq_idx = np.where(abs(freqs-target_freq)==min(abs(freqs-target_freq)))[0][0]

  idx = np.argsort(freqs)
  # plt.figure()
  # plt.plot(freqs[idx], ch1_power[idx], 'g')
  # plt.plot(freqs[idx], ch2_power[idx], 'r')
  # plt.plot(freqs[freq_idx], ch1_power[freq_idx], 'g*')
  # plt.plot(freqs[freq_idx], ch2_power[freq_idx], 'r*')
  # plt.title(r_ch1)


  x = freqs[idx]
  freq_idx = np.where(abs(x-target_freq)==min(abs(x-target_freq)))[0][0]


  ch1_plot_ys.append(ch1_power[idx])
  ch2_plot_ys.append(ch2_power[idx])
  roi_names.append(r_ch1)

  ridx += 1


outfile_name = "powerspec_COND_%s_SLICE_%s.html" % (condition, slice_number) 
output_file(outfile_name)
s1 = figure(width=500, plot_height=500, title=roi_names[0])
s2 = figure(width=500, plot_height=500, title=roi_names[1])
s3 = figure(width=500, plot_height=500, title=roi_names[2])
s4 = figure(width=500, plot_height=500, title=roi_names[3])
s5 = figure(width=500, plot_height=500, title=roi_names[4])
s6 = figure(width=500, plot_height=500, title=roi_names[5])
s7 = figure(width=500, plot_height=500, title=roi_names[6])
s8 = figure(width=500, plot_height=500, title=roi_names[7])
s9 = figure(width=500, plot_height=500, title=roi_names[8])
s10 = figure(width=500, plot_height=500, title=roi_names[9])
s11 = figure(width=500, plot_height=500, title=roi_names[10])
s12 = figure(width=500, plot_height=500, title=roi_names[11])
s13 = figure(width=500, plot_height=500, title=roi_names[12])
s14 = figure(width=500, plot_height=500, title=roi_names[13])

subplots = [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14]
for sidx,s in enumerate(subplots):
  s.line(x, ch1_plot_ys[sidx], line_width=2, color="green", alpha=0.5)
  s.line(x, ch2_plot_ys[sidx], line_width=2, color="red", alpha=0.5)
  s.circle(x[freq_idx], ch1_plot_ys[sidx][freq_idx], size=5, color='green', alpha=1.0)
  s.circle(x[freq_idx], ch2_plot_ys[sidx][freq_idx], size=5, color='red', alpha=1.0)

# p = gridplot([[s1, s2], [s3, s4], [s5, s6]])
p = gridplot([[s1, s2], [s3, s4], [s5, s6], [s7, s8], [s9, s10], [s11, s12], [s13, s14]])


show(p)


# from bokeh.embed import components




# from bokeh.resources import CDN
# from bokeh.embed import file_html
# file_html(p, CDN, outfile_name)
show(p)









# 0.13 Hz frequency ---------------------------------------------------

nreps = 1290
target_freq = 0.13
acqusition_rate = 4.11

nframes_per_cyc = (1/target_freq) * acquisition_rate
moving_win_sz = nframes_per_cyc * 2


# roi_path = '/media/juliana/IMDATA/TEFO/20161218_CE025/fov2_bar6_00002'
# ch1_roi = 'ch1_s12_large_square_rois.xls'
# ch2_roi = 'ch2_s12_large_square_rois.xls'

# roi_path = '/media/juliana/IMDATA/TEFO/20161218_CE025/fov2_bar2_00002'
# ch1_roi = 'ch1_sl12_largesquare_roi.xls'
# ch2_roi = 'ch2_sl12_largesquare_roi.xls'


roi_path = '/media/juliana/IMDATA/TEFO/20161219_JR030W/fov6_013Hz_nomask_bar1_00004'
ch1_roi = 'ch1_sl18_largesquare_roi.xls'


# 0.37 Hz frequency ---------------------------------------------------

nreps = 333
target_freq = 0.37
acquisition_rate = 4.11

nframes_per_cyc = (1/target_freq) * acquisition_rate
moving_win_sz = nframes_per_cyc * 2

# nreps = 455
# target_freq = 0.37
# acquisition_rate = 4.11

# roi_path = '/media/juliana/IMDATA/TEFO/20161218_CE025/fov2_bar5_00001'
# ch1_roi = 'ch1_s15_large_square_roi.xls'
# ch2_roi = 'ch2_s15_large_square_roi.xls'

# roi_path = '/media/juliana/IMDATA/TEFO/20161218_CE025/fov2_bar5_00002'
# ch1_roi = 'ch1_sl10_largesquare_roi.xls'
# ch2_roi = 'ch2_sl10_largesquare_roi.xls'


roi_path = '/media/juliana/IMDATA/TEFO/20161219_JR030W/fov5_037Hz_nomask_bar1_00001'

ch1_roi = 'ch1_sl20_largesquare_roi.xls'

ch2_roi = 'ch2_sl12_largesquare_roi.xls'


# Extrac ROI values -----------------------------------------------------

check_channels = ['ch1', 'ch2']
roi_files = [ch1_roi, ch2_roi]

check_rois = dict()
for ch_idx,curr_ch in enumerate(check_channels):

  cr = csv.reader(open(os.path.join(roi_path, roi_files[ch_idx]),"rb"), delimiter='\t')
  arr = range(nreps) #adjust to needed
  x = 0
  for ridx,row in enumerate(cr):
    if ridx==0:    
      continue
    else:
      arr[x] = row
      x += 1

  y = [float(a[1]) for a in np.array(arr)]

  if rolling is True:
      # pix_padded = [np.ones(moving_win_sz)*y[y.keys()[0]], y, np.ones(moving_win_sz)*y[y.keys()[-1]]]
      pix_padded = [np.ones(moving_win_sz)*y[0], y, np.ones(moving_win_sz)*y[-1]]
      tmp_pix = list(itertools.chain(*pix_padded))
      tmp_pix_rolling = np.convolve(tmp_pix, np.ones(moving_win_sz)/moving_win_sz, 'same')
      remove_pad = (len(tmp_pix_rolling) - len(y) ) / 2
      rpix = np.array(tmp_pix_rolling[remove_pad:-1*remove_pad])
      y -= rpix
  # else:
  #    pix = scipy.signal.detrend(y, type='constant') # HP filter - over time...
  #    y =  pix

  ft = fft.fft(y)
  check_rois[curr_ch] = ft



# PLOTTING: -------------------------------------------------------------

N = len(y)
#sampling_rate = 5.58 #60. #4.11 #2.14
ts = 1 / acquisition_rate
freqs = fft.fftfreq(N, ts)
ch1_power = np.abs(check_rois['ch1'])**2
ch2_power = np.abs(check_rois['ch2'])**2

  # freq_idx = np.where(abs(freqs-target_freq)==min(abs(freqs-target_freq)))[0][0]

idx = np.argsort(freqs)
# plt.figure()
# plt.plot(freqs[idx], ch1_power[idx], 'g')
# plt.plot(freqs[idx], ch2_power[idx], 'r')
# plt.plot(freqs[freq_idx], ch1_power[freq_idx], 'g*')
# plt.plot(freqs[freq_idx], ch2_power[freq_idx], 'r*')
# plt.title(r_ch1)


x = freqs[idx]
freq_idx = np.where(abs(x-target_freq)==min(abs(x-target_freq)))[0][0]

N = len(y)
#sampling_rate = 5.58 #60. #4.11 #2.14
ts = 1 / acquisition_rate
freqs = fft.fftfreq(N, ts)
power = np.abs(ft)**2

idx = np.argsort(freqs)
plt.figure()
plt.plot(x, ch1_power[idx], 'g')
plt.plot(x, ch2_power[idx], 'r')
plt.plot(x[freq_idx], ch1_power[idx][freq_idx], 'g*')
plt.plot(x[freq_idx], ch2_power[idx][freq_idx], 'r*')
plt.title(ch1_roi.split('.')[0])












# import csv
# import requests

# r = requests.get('http://vote.wa.gov/results/current/export/MediaResults.txt') 
# data = r.text
# reader = csv.reader(data.splitlines(), delimiter='\t')
# for row in reader:
#     print row