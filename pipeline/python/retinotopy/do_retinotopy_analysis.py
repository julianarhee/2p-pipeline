
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import h5py
import json
import re
import optparse
import pprint
import tifffile as tf
import pylab as pl
import numpy as np
from scipy import ndimage
import cv2

from pipeline.python.utils import natural_keys, replace_root

pp = pprint.PrettyPrinter(indent=4)

def extract_options(options):

	parser = optparse.OptionParser()

	# PATH opts:
	parser.add_option('-D', '--root', action='store', dest='rootdir', default='/nas/volume1/2photon/data', help='data root dir (root project dir containing all animalids) [default: /nas/volume1/2photon/data, /n/coxfs01/2pdata if --slurm]')
	parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')
	parser.add_option('-S', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID')
	parser.add_option('-A', '--acq', action='store', dest='acquisition', default='FOV1', help="acquisition folder (ex: 'FOV1_zoom3x') [default: FOV1]")
	parser.add_option('-R', '--run', action='store', dest='run', default='', help="name of run dir containing tiffs to be processed (ex: gratings_phasemod_run1)")
	parser.add_option('-d', '--analysis-id', action='store', dest='analysis_id', default='', help="ANALYSIS ID for retinoid param set to use (created with set_analysis_parameters.py, e.g., analysis001,  etc.)")
	parser.add_option('--slurm', action='store_true', dest='slurm', default=False, help="Set if running as SLURM job on Odyssey")
	(options, args) = parser.parse_args(options)

	if options.slurm is True:
		if 'coxfs01' not in options.rootdir:
			options.rootdir = '/n/coxfs01/2p-data'

	return options

#-----------------------------------------------------
#           FUNCTIONS FOR DATA PROCESSING
#-----------------------------------------------------

def block_mean(ar, fact):
	assert isinstance(fact, int), type(fact)
	sx, sy = ar.shape
	X, Y = np.ogrid[0:sx, 0:sy]
	regions = sy/fact * (X/fact) + Y/fact
	res = ndimage.mean(ar, labels=regions, index=np.arange(regions.max() + 1))
	res.shape = (sx/fact, sy/fact)
	return res

def block_mean_stack(stack0, ds_factor):
	im0 = block_mean(stack0[:,:,0],ds_factor) 
	stack1 = np.zeros((im0.shape[0],im0.shape[1],stack0.shape[2]))
	for i in range(0,stack0.shape[2]):
		stack1[:,:,i] = block_mean(stack0[:,:,i],ds_factor) 
	return stack1

def smooth_array(inputArray,fwhm):
	szList=np.array([None,None,None,11,None,21,None,27,None,31,None,37,None,43,None,49,None,53,None,59,None,55,None,69,None,79,None,89,None,99])
	sigmaList=np.array([None,None,None,.9,None,1.7,None,2.6,None,3.4,None,4.3,None,5.1,None,6.4,None,6.8,None,7.6,None,8.5,None,9.4,None,10.3,None,11.2,None,12])
	sigma=sigmaList[fwhm]
	sz=szList[fwhm]

	outputArray=cv2.GaussianBlur(inputArray, (sz,sz), sigma, sigma)
	return outputArray

def smooth_stack(stack0, fwhm):
	stack1 = np.zeros(stack0.shape)
	for i in range(0,stack0.shape[2]):
		stack1[:,:,i] = smooth_array(stack0[:,:,i], fwhm) 
	return stack1

def get_processed_stack(tiff_path_full,RETINOID):
	# Read in RAW tiff: 
	print('Loading file : %s'%(tiff_path_full))
	stack0 = tf.imread(tiff_path_full)
	#swap axes for familiarity
	stack1 = np.swapaxes(stack0,0,2)
	stack1 = np.swapaxes(stack1,1,0)
	del stack0 # to save space

	#block-reduce, if indicated
	if RETINOID['PARAMS']['downsample_factor'] is not None:
		print('Performing block-reduction on stack....')
		stack1 = block_mean_stack(stack1, int(RETINOID['PARAMS']['downsample_factor']))

	#spatial smoothing, if indicated
	if RETINOID['PARAMS']['smooth_fwhm'] is not None:
		print('Performing spatial smoothing on stack....')
		stack1 = smooth_stack(stack1, int(RETINOID['PARAMS']['smooth_fwhm']))
	return stack1
		
def process_array(roi_trace, RETINOID, stack_info):
	
	frame_rate = stack_info['frame_rate']
	stimfreq = stack_info['stimfreq']
	#hard-code for now, in the future, get from mworks file
	if RETINOID['PARAMS']['minus_rolling_mean']:
		print('Removing rolling mean from traces...')
		detrend_roi_trace = np.zeros(roi_trace.shape)

		windowsz = int(np.ceil((np.true_divide(1,stimfreq)*3)*frame_rate))

		for roi in range(roi_trace.shape[0]):
			tmp0=roi_trace[roi,:];
			tmp1=np.concatenate((np.ones(windowsz)*tmp0[0], tmp0, np.ones(windowsz)*tmp0[-1]),0)

			rolling_mean=np.convolve(tmp1, np.ones(windowsz)/windowsz, 'same')
			rolling_mean=rolling_mean[windowsz:-windowsz]

			detrend_roi_trace[roi,:]=np.subtract(tmp0,rolling_mean)
		roi_trace = detrend_roi_trace
		del detrend_roi_trace
	if RETINOID['PARAMS']['average_frames'] is not None:
		print('Performing temporal smoothing on traces...')
		smooth_roi_trace = np.zeros(roi_trace.shape)

		windowsz = int(RETINOID['PARAMS']['average_frames'])

		for roi in range(roi_trace.shape[0]):
			tmp0=roi_trace[roi,:];
			tmp1=np.concatenate((np.ones(windowsz)*tmp0[0], tmp0, np.ones(windowsz)*tmp0[-1]),0)

			tmp2=np.convolve(tmp1, np.ones(windowsz)/windowsz, 'same')
			tmp2=tmp2[windowsz:-windowsz]

			smooth_roi_trace[roi,:]=tmp2
		roi_trace = smooth_roi_trace
		del smooth_roi_trace
	return roi_trace

def do_regression(t,phi,roi_trace,npixels,tpoints,roi_type,signal_fit_idx):
	print('Doing regression')
	#doing regression to get amplitude and variance expained
	t=np.transpose(np.expand_dims(t,1))
	tmatrix=np.tile(t,(npixels,1))

	phimatrix=np.tile(phi,(1,tpoints))
	Xmatrix=np.cos(tmatrix+phimatrix)

	beta_array=np.zeros((npixels))
	varexp_array=np.zeros((npixels))
	if roi_type != 'pixels':
		signal_fit = np.zeros((npixels,tpoints))

	for midx in range(npixels):
		x=np.expand_dims(Xmatrix[midx,:],1)
		y=roi_trace[midx,:]
		beta=np.matmul(np.linalg.pinv(x),y)
		beta_array[midx]=beta
		yHat=x*beta
		if roi_type == 'pixels':
			if midx == signal_fit_idx:
				signal_fit=np.squeeze(yHat)
		else:
			signal_fit[midx,:]=np.squeeze(yHat)
		SSreg=np.sum((yHat-np.mean(y,0))**2)
		SStotal=np.sum((y-np.mean(y,0))**2)
		varexp_array[midx]=SSreg/SStotal

	return varexp_array, beta_array, signal_fit

def get_mask_traces(tiff_stack,masks):
	szx, szy, nframes = tiff_stack.shape
	nmasks, szx, szy = masks.shape

	roi_trace = np.zeros((nmasks,nframes))

	for frame in range(nframes):
		for midx in range(nmasks):
			#get frame and mask
			im0 = np.squeeze(tiff_stack[:,:,frame])
			single_mask = np.squeeze(masks[midx,:,:])

			#set zero values in mask as nans in image
			im0 = im0.astype('float32')
			im0[single_mask==0] = np.nan

			#take the average of non-zero mask values
			roi_trace[midx,frame] = np.nanmean(im0)
	return roi_trace



	

def analyze_tiff(tiff_path_full,tiff_fn,stack_info, RETINOID,file_dir,tiff_fig_dir,masks_file):


	#intialize file to save data
	data_fn = 'retino_data_%s.h5' %(tiff_fn[:-4])
	file_grp = h5py.File(os.path.join(file_dir,data_fn),  'w')

	 #get tiff stack
	tiff_stack = get_processed_stack(tiff_path_full,RETINOID)
	szx, szy, nframes = tiff_stack.shape

	#save some details
	file_grp.attrs['SRC'] = tiff_path_full
	file_grp.attrs['sz_info'] =  (szx, szy, nframes)
	file_grp.attrs['frame_rate'] = stack_info['frame_rate']
	file_grp.attrs['stimfreq'] = stack_info['stimfreq']



	if RETINOID['PARAMS']['roi_type'] == 'pixels':

		#reshape stack
		roi_trace = np.reshape(tiff_stack,(szx*szy, nframes))

	elif RETINOID['PARAMS']['roi_type'] == 'retino':
		#get saved masks for this tiff stack
		s0 = tiff_fn[:-4]
		file_str = s0[s0.find('File'):s0.find('File')+7]
		slice_str = s0[s0.find('Slice'):s0.find('Slice')+7]
		masks = masks_file[file_str]['masks'][slice_str][:]

		nmasks,szx, szy= masks.shape

		#apply masks to stack
		roi_trace = get_mask_traces(tiff_stack,masks)
		
	else:
		#get saved masks for this tiff stack
		s0 = tiff_fn[:-4]
		file_str = masks_file.keys()[0]
		print(file_str)
		slice_str = s0[s0.find('Slice'):s0.find('Slice')+7]
		masks = masks_file[file_str]['masks'][slice_str][:]
		#swap axes for familiarity
		masks = np.swapaxes(masks,1,2)

		nmasks, szx, szy= masks.shape

		#apply masks to stack
		roi_trace = get_mask_traces(tiff_stack,masks)

	roi_trace = process_array(roi_trace, RETINOID, stack_info)

	frame_rate = stack_info['frame_rate']
	stimfreq = stack_info['stimfreq']

	#Get fft  
	print('Getting fft....')
	fourier_data = np.fft.fft(roi_trace)


	#Get magnitude and phase data
	print('Analyzing phase and magnitude....')
	mag_data=abs(fourier_data)
	phase_data=np.angle(fourier_data)

	#label frequency bins
	freqs = np.fft.fftfreq(nframes, float(1/frame_rate))
	idx = np.argsort(freqs)
	freqs=freqs[idx]

	#sort magnitude and phase data
	mag_data=mag_data[:,idx]
	phase_data=phase_data[:,idx]

	#excluding DC offset from data
	freqs=freqs[np.round(nframes/2)+1:]
	mag_data=mag_data[:,np.round(nframes/2)+1:]
	phase_data=phase_data[:,np.round(nframes/2)+1:]

	freq_idx=np.argmin(np.absolute(freqs-stimfreq))#find out index of stimulation freq
	top_freq_idx=np.where(freqs>1)[0][0]#find out index of 1Hz, to cut-off zoomed out plot
	max_mod_idx=np.argmax(mag_data[:,freq_idx],0)#best pixel index

	#unpack values from frequency analysis
	mag_array = mag_data[:,freq_idx]                    
	phase_array = phase_data[:,freq_idx]      

	#get magnitude ratio
	tmp=np.copy(mag_data)
	np.delete(tmp,freq_idx,1)
	nontarget_mag_array=np.sum(tmp,1)
	mag_ratio_array=mag_array/nontarget_mag_array


	#figure out timing of points (assuming constant rate)
	frame_period = float(1/frame_rate)
	frametimes = np.arange(frame_period,frame_period*(nframes+1),frame_period)
	if len(frametimes)>nframes:
		to_del = len(frametimes)-signal_length
		frametimes = frametimes[:-to_del]

	#do regression, get some info from fit
	t=frametimes*(2*np.pi)*stimfreq
	phi=np.expand_dims(phase_array,1)
	varexp_array, beta_array, signal_fit = do_regression(t,phi,roi_trace,roi_trace.shape[0],roi_trace.shape[1],\
														 RETINOID['PARAMS']['roi_type'],max_mod_idx)
	print('Saving data to file')

	#save data values to structure
	if 'mag_array' not in file_grp.keys():
		magset = file_grp.create_dataset('mag_array',mag_array.shape, mag_array.dtype)
		magset[...] = mag_array
	if 'phase_array' not in file_grp.keys():
		phaseset = file_grp.create_dataset('phase_array',phase_array.shape, phase_array.dtype)
		phaseset[...] = phase_array
	if 'mag_ratio_array' not in file_grp.keys():
		ratioset = file_grp.create_dataset('mag_ratio_array',mag_ratio_array.shape, mag_ratio_array.dtype)
		ratioset[...] = mag_ratio_array
	if 'beta_array' not in file_grp.keys():
		betaset = file_grp.create_dataset('beta_array',beta_array.shape, beta_array.dtype)
		betaset[...] = beta_array
	if 'var_exp_array' not in file_grp.keys():
		varset = file_grp.create_dataset('var_exp_array',varexp_array.shape, varexp_array.dtype)
		varset[...] = varexp_array
	if RETINOID['PARAMS']['roi_type'] != 'pixels':
		if 'masks' not in file_grp.keys():
			mset = file_grp.create_dataset('masks',masks.shape, masks.dtype)
			mset[...] = masks
	file_grp.close()


	#VISUALIZE!!!
	print('Visualizing results')
	print('Output folder: %s'%(tiff_fig_dir))
	#visualize pixel-based results
	if RETINOID['PARAMS']['roi_type'] == 'pixels':

		fig_name = 'best_pixel_power_%s.png' %(tiff_fn[:-4])
		fig=plt.figure()
		plt.plot(freqs,mag_data[max_mod_idx,:])
		plt.xlabel('Frequency (Hz)',fontsize=16)
		plt.ylabel('Magnitude',fontsize=16)
		axes = plt.gca()
		ymin, ymax = axes.get_ylim()
		plt.axvline(x=freqs[freq_idx], ymin=ymin, ymax = ymax, linewidth=1, color='r')
		plt.savefig(os.path.join(tiff_fig_dir,fig_name))
		plt.close()

		fig_name = 'best_pixel_power_zoom_%s.png' %(tiff_fn[:-4])
		fig=plt.figure()
		plt.plot(freqs[0:top_freq_idx],mag_data[max_mod_idx,0:top_freq_idx])
		plt.xlabel('Frequency (Hz)',fontsize=16)
		plt.ylabel('Magnitude',fontsize=16)
		axes = plt.gca()
		ymin, ymax = axes.get_ylim()
		plt.axvline(x=freqs[freq_idx], ymin=ymin, ymax = ymax, linewidth=1, color='r')
		plt.savefig(os.path.join(tiff_fig_dir,fig_name))
		plt.close()

		stimperiod_t=np.true_divide(1,stimfreq)
		stimperiod_frames=stimperiod_t*frame_rate
		periodstartframes=np.round(np.arange(0,len(frametimes),stimperiod_frames))[:-1]
		periodstartframes = periodstartframes.astype('int')

		fig_name = 'best_pixel_frequency_fit_%s.png' %(tiff_fn[:-4])
		fig=plt.figure()
		plt.plot(frametimes,roi_trace[max_mod_idx,:],'b')
		plt.plot(frametimes,signal_fit,'r')
		plt.xlabel('Time (s)',fontsize=16)
		plt.ylabel('Pixel Value',fontsize=16)
		axes = plt.gca()
		ymin, ymax = axes.get_ylim()
		for f in periodstartframes:
			plt.axvline(x=frametimes[f], ymin=ymin, ymax = ymax, linewidth=1, color='k')
		axes.set_xlim([frametimes[0],frametimes[-1]])
		plt.savefig(os.path.join(tiff_fig_dir,fig_name))
		plt.close()


		mag_map = np.reshape(mag_array,(szy,szx))
		mag_ratio_map = np.reshape(mag_ratio_array,(szy,szx))
		phase_map = np.reshape(phase_array,(szy,szx))
		beta_map = np.reshape(beta_array,(szy,szx))
		varexp_map = np.reshape(varexp_array,(szy,szx))


		fig_name = 'phase_map_%s.png' %(tiff_fn[:-4])
		#set phase map range for visualization
		phase_map_disp=np.copy(phase_map)
		phase_map_disp[phase_map<0]=-phase_map[phase_map<0]
		phase_map_disp[phase_map>0]=(2*np.pi)-phase_map[phase_map>0]

		fig=plt.figure()
		plt.imshow(phase_map_disp,'nipy_spectral',vmin=0,vmax=2*np.pi)
		plt.colorbar()
		plt.savefig(os.path.join(tiff_fig_dir,fig_name))
		plt.close()

		fig_name = 'mag_map_%s.png' %(tiff_fn[:-4])
		fig=plt.figure()
		plt.imshow(mag_map)
		plt.colorbar()
		plt.savefig(os.path.join(tiff_fig_dir,fig_name))
		plt.close()

		fig_name = 'mag_ratio_map_%s.png' %(tiff_fn[:-4])
		fig=plt.figure()
		plt.imshow(mag_ratio_map)
		plt.colorbar()
		plt.savefig(os.path.join(tiff_fig_dir,fig_name))
		plt.close()

		fig_name = 'beta_map_%s.png' %(tiff_fn[:-4])
		fig=plt.figure()
		plt.imshow(beta_map)
		plt.colorbar()
		plt.savefig(os.path.join(tiff_fig_dir,fig_name))
		plt.close()

		fig_name = 'var_exp_map_%s.png' %(tiff_fn[:-4])
		fig=plt.figure()
		plt.imshow(varexp_map)
		plt.colorbar()
		plt.savefig(os.path.join(tiff_fig_dir,fig_name))
		plt.close()

		# #Read in average image (for viuslization)
		avg_dir = os.path.join('%s_mean_deinterleaved'%(str(RETINOID['SRC'])),'visible')
		s0 = tiff_fn[:-4]
		s1 = s0[s0.find('Slice'):]

		avg_fn = 'vis_mean_%s.tif'%(s1)

		im0 = tf.imread(os.path.join(avg_dir, avg_fn))
		if RETINOID['PARAMS']['downsample_factor'] is not None:
			ds = int(RETINOID['PARAMS']['downsample_factor'])
			im0 = block_mean(im0,ds)
		im1 = np.uint8(np.true_divide(im0,np.max(im0))*255)
		im2 = np.dstack((im1,im1,im1))

		fig_name = 'phase_map_overlay_%s.png' %(tiff_fn[:-4])


		fig=plt.figure()
		plt.imshow(im2,'gray')
		plt.imshow(phase_map_disp,'nipy_spectral',alpha=0.25,vmin=0,vmax=2*np.pi)
		plt.colorbar()
		plt.savefig(os.path.join(tiff_fig_dir,fig_name))
		plt.close()
	else:

		#make figure directory for stimulus type
		fig_dir = os.path.join(tiff_fig_dir, file_str,'spectrum')
		if not os.path.exists(fig_dir):
				os.makedirs(fig_dir)

		for midx in range(nmasks):
			fig_name = 'full_spectrum_mask%04d_%s.png' %(midx+1,RETINOID['PARAMS']['roi_type'])
			fig=plt.figure()
			plt.plot(freqs,mag_data[midx,:])
			plt.xlabel('Frequency (Hz)',fontsize=16)
			plt.ylabel('Magnitude',fontsize=16)
			axes = plt.gca()
			ymin, ymax = axes.get_ylim()
			plt.axvline(x=freqs[freq_idx], ymin=ymin, ymax = ymax, linewidth=1, color='r')
			plt.savefig(os.path.join(fig_dir,fig_name))
			plt.close()

		for midx in range(nmasks):
			fig_name = 'zoom_spectrum_mask%04d_%s.png' %(midx+1,RETINOID['PARAMS']['roi_type'])
			fig=plt.figure()
			plt.plot(freqs[0:top_freq_idx],mag_data[midx,0:top_freq_idx])
			plt.xlabel('Frequency (Hz)',fontsize=16)
			plt.ylabel('Magnitude',fontsize=16)
			axes = plt.gca()
			ymin, ymax = axes.get_ylim()
			plt.axvline(x=freqs[freq_idx], ymin=ymin, ymax = ymax, linewidth=1, color='r')
			plt.savefig(os.path.join(fig_dir,fig_name))
			plt.close()


		fig_dir = os.path.join(tiff_fig_dir, file_str,'timecourse')
		if not os.path.exists(fig_dir):
				os.makedirs(fig_dir)

		stimperiod_t=np.true_divide(1,stimfreq)
		stimperiod_frames=stimperiod_t*frame_rate
		periodstartframes=np.round(np.arange(0,len(frametimes),stimperiod_frames))[:-1]
		periodstartframes = periodstartframes.astype('int')

		for midx in range(nmasks):
			fig_name = 'timecourse_fit_mask%04d_%s.png' %(midx+1,RETINOID['PARAMS']['roi_type'])
			fig=plt.figure()
			plt.plot(frametimes,roi_trace[midx,:],'b')
			plt.plot(frametimes,signal_fit[midx,:],'r')
			plt.xlabel('Time (s)',fontsize=16)
			plt.ylabel('Pixel Value',fontsize=16)
			axes = plt.gca()
			ymin, ymax = axes.get_ylim()
			for f in periodstartframes:
				plt.axvline(x=frametimes[f], ymin=ymin, ymax = ymax, linewidth=1, color='k')
			axes.set_xlim([frametimes[0],frametimes[-1]])
			plt.savefig(os.path.join(fig_dir,fig_name))
			plt.close()

		#set phase map range for visualization
		phase_array_disp=np.copy(phase_array)
		phase_array_disp[phase_array<0]=-phase_array[phase_array<0]
		phase_array_disp[phase_array>0]=(2*np.pi)-phase_array[phase_array>0]

		#mark rois
		magratio_roi = np.empty((szy,szx))
		magratio_roi[:] = np.NAN

		mag_roi = np.copy(magratio_roi)
		varexp_roi = np.copy(magratio_roi)
		phase_roi = np.copy(magratio_roi)

		for midx in range(nmasks):
			maskpix = np.where(np.squeeze(masks[midx,:,:]))
			print(len(maskpix))
			magratio_roi[maskpix]=mag_ratio_array[midx]
			mag_roi[maskpix]=mag_array[midx]
			varexp_roi[maskpix]=varexp_array[midx]
			phase_roi[maskpix]=phase_array_disp[midx]

		fig_dir = tiff_fig_dir

		# #Read in average image (for viuslization)
		avg_dir = os.path.join('%s_mean_deinterleaved'%(str(RETINOID['SRC'])),'visible')
		s0 = tiff_fn[:-4]
		s1 = s0[s0.find('Slice'):]

		avg_fn = 'vis_mean_%s.tif'%(s1)

		im0 = tf.imread(os.path.join(avg_dir, avg_fn))
		if RETINOID['PARAMS']['downsample_factor'] is not None:
			ds = int(RETINOID['PARAMS']['downsample_factor'])
			im0 = block_mean(im0,ds)
		im1 = np.uint8(np.true_divide(im0,np.max(im0))*255)
		im2 = np.dstack((im1,im1,im1))

		fig_name = 'phase_info_%s.png' %(tiff_fn[:-4])
		fig=plt.figure()
		plt.imshow(im2,'gray')
		plt.imshow(phase_roi,'nipy_spectral',alpha = 0.5,vmin=0,vmax=2*np.pi)
		plt.colorbar()
		plt.savefig(os.path.join(fig_dir,fig_name))
		plt.close()

		fig_name = 'mag_info_%s.png' %(tiff_fn[:-4])
		fig=plt.figure()
		plt.imshow(im2,'gray')
		plt.imshow(mag_roi, alpha = 0.5)
		plt.colorbar()
		plt.savefig(os.path.join(fig_dir,fig_name))
		plt.close()

		fig_name = 'mag_ratio_info_%s.png' %(tiff_fn[:-4])
		fig=plt.figure()
		plt.imshow(im2,'gray')
		plt.imshow(magratio_roi, alpha = 0.5)
		plt.colorbar()
		plt.savefig(os.path.join(fig_dir,fig_name))
		plt.close()

		fig_name = 'varexp_info_%s.png' %(tiff_fn[:-4])
		fig=plt.figure()
		plt.imshow(im2,'gray')
		plt.imshow(varexp_roi, alpha = 0.5)
		plt.colorbar()
		plt.savefig(os.path.join(fig_dir,fig_name))
		plt.close()

		fig_name = 'phase_nice_%s.png' %(tiff_fn[:-4])
		dpi = 80
		szY,szX = im1.shape
		# What size does the figure need to be in inches to fit the image?
		figsize = szX / float(dpi), szY / float(dpi)
		# Create a figure of the right size with one axes that takes up the full figure
		fig = plt.figure(figsize=figsize)
		ax = fig.add_axes([0, 0, 1, 1])
		# Hide spines, ticks, etc.
		ax.axis('off')
		ax.imshow(im2,'gray')
		ax.imshow(phase_roi,'nipy_spectral',alpha = 0.5,vmin=0,vmax=2*np.pi)
		fig.savefig(os.path.join(fig_dir,fig_name), dpi=dpi, transparent=True)
		plt.close()

		fig_dir = os.path.join(tiff_fig_dir,'histos')
		if not os.path.exists(fig_dir):
				os.makedirs(fig_dir)

		if np.max(mag_ratio_array)>np.min(mag_ratio_array):
			fig_fn = 'roi_mag_ratio%s.png'%(tiff_fn[:-4])
			bin_loc = np.arange(0,np.max(mag_ratio_array)+.002,.002)
			plt.hist(mag_ratio_array,bin_loc)
			plt.xlabel('Magnitude Ratio')
			plt.ylabel('ROI Count')
			plt.savefig(os.path.join(fig_dir,fig_fn))
			plt.close()

		if np.max(phase_array)>np.min(phase_array):
			fig_fn = 'roi_phase%s.png'%(tiff_fn[:-4])
			bin_loc = np.arange(0,(2*np.pi)+.2,.2)
			plt.hist(phase_array,bin_loc)
			plt.xlabel('Phase')
			plt.ylabel('ROI Count')
			plt.savefig(os.path.join(fig_dir,fig_fn))
			plt.close()

		if np.max(varexp_array)>np.min(varexp_array):
			fig_fn = 'roi_varexp%s.png'%(tiff_fn[:-4])
			bin_loc = np.arange(0,np.max(varexp_array)+.01,.01)
			plt.hist(varexp_array,bin_loc)
			plt.xlabel('Variance Explained')
			plt.ylabel('ROI')
			plt.savefig(os.path.join(fig_dir,fig_fn))
			plt.close()


def do_analysis(options):
	 #Get options
	options = extract_options(options)

	# # Set USER INPUT options:
	rootdir = options.rootdir
	animalid = options.animalid
	session = options.session
	acquisition = options.acquisition
	analysis_id = options.analysis_id
	run = options.run

	#%%
	# =============================================================================
	# Load specified analysis-ID parameter set:
	# =============================================================================
	run_dir = os.path.join(rootdir, animalid, session, acquisition, run)
	analysis_dir = os.path.join(run_dir, 'retino_analysis')
	tmp_retinoid_dir = os.path.join(analysis_dir, 'tmp_retinoids')


	if not os.path.exists(analysis_dir):
		os.makedirs(analysis_dir)
	try:
		print "Loading params for ANALYSIS SET, id %s" % analysis_id
		analysisdict_path = os.path.join(analysis_dir, 'analysisids_%s.json' % run)
		with open(analysisdict_path, 'r') as f:
			analysisdict = json.load(f)
		RETINOID = analysisdict[analysis_id]
		pp.pprint(RETINOID)
	except Exception as e:
		print "No analysis SET entry exists for specified id: %s" % analysis_id
		print e
		try:
			print "Checking tmp analysis-id dir..."
			if auto is False:
				while True:
					tmpfns = [t for t in os.listdir(tmp_retinoid_dir) if t.endswith('json')]
					for tidx, tidfn in enumerate(tmpfns):
						print tidx, tidfn
					userchoice = raw_input("Select IDX of found tmp analysis-id to view: ")
					with open(os.path.join(tmp_retinoid_dir, tmpfns[int(userchoice)]), 'r') as f:
						tmpRETINOID = json.load(f)
					print "Showing tid: %s, %s" % (tmpRETINOID['analysis_id'], tmpRETINOID['analysis_hash'])
					pp.pprint(tmpRETINOID)
					userconfirm = raw_input('Press <Y> to use this analysis ID, or <q> to abort: ')
					if userconfirm == 'Y':
						RETINOID = tmpRETINOID
						break
					elif userconfirm == 'q':
						break
		except Exception as E:
			print "---------------------------------------------------------------"
			print "No tmp analysis-ids found either... ABORTING with error:"
			print e
			print "---------------------------------------------------------------"

	#%%
	# =============================================================================
	# Get meta info for current run and source tiffs using analysis-ID params:
	# =============================================================================
	analysis_hash = RETINOID['analysis_hash']

	tiff_dir = RETINOID['SRC']
	#roi_name = RETINOID['PARAMS']['roi_id']
	tiff_files = sorted([t for t in os.listdir(tiff_dir) if t.endswith('tif')], key=natural_keys)
	#print "Found %i tiffs in dir %s.\nExtracting analysis with ROI set %s." % (len(tiff_files), tiff_dir, roi_name)

	# Get associated RUN info:
	runmeta_path = os.path.join(run_dir, '%s.json' % run)
	with open(runmeta_path, 'r') as r:
		runinfo = json.load(r)

	nslices = len(runinfo['slices'])
	nchannels = runinfo['nchannels']
	nvolumes = runinfo['nvolumes']
	ntiffs = runinfo['ntiffs']

	#Set file and figure directory
	file_dir = os.path.join(RETINOID['DST'],'files')
	if not os.path.exists(file_dir):
			os.makedirs(file_dir)
			
	fig_base_dir = os.path.join(RETINOID['DST'],'figures')
	if not os.path.exists(fig_base_dir):
			os.makedirs(fig_base_dir)


	#-----Get info from paradigm file
	para_file_dir = os.path.join(run_dir,'paradigm','files')
	para_file =  [f for f in os.listdir(para_file_dir) if f.endswith('.json')][0]#assuming a single file for all tiffs in run
	print 'Getting paradigm file info from %s'%(os.path.join(para_file_dir, para_file))

	with open(os.path.join(para_file_dir, para_file), 'r') as r:
		parainfo = json.load(r)


	#OPEN MASK FILE
	if RETINOID['PARAMS']['roi_type'] != 'pixels':
		print 'Getting masks'
		# Load ROI set specified in analysis param set:
		roi_dir = os.path.join(rootdir, animalid, session, 'ROIs')
		roidict_path = os.path.join(roi_dir, 'rids_%s.json' % session)
		with open(roidict_path, 'r') as f:
			roidict = json.load(f)
		RID = roidict[RETINOID['PARAMS']['roi_id']]
		rid_hash = RID['rid_hash']

		# Load mask file:
		if rootdir not in RID['DST']:
			rid_dst = replace_root(RID['DST'], rootdir, animalid, session)
			notnative = True
		else:
			rid_dst = RID['DST']
			notnative = False

		mask_path = os.path.join(rid_dst, 'masks.hdf5')
		masks_file = h5py.File(mask_path,  'r')#read
	else:
		masks_file = None


	#%%
	# =============================================================================
	# Cyce through TIFF stacks and analayze
	# =============================================================================
	#TODO: skip tiff if already processed
	for tiff_count, tiff_fn in enumerate(tiff_files):    	    

		tiff_path_full = os.path.join(RETINOID['PARAMS']['tiff_source'],tiff_fn)

		#get some info from paradigm and put into a dict to pass to some fxns later
		stack_info = dict()
		stack_info['stimulus'] = parainfo[str(tiff_count+1)]['stimuli']['stimulus']
		stack_info['stimfreq'] = parainfo[str(tiff_count+1)]['stimuli']['scale']
		stack_info['frame_rate'] = runinfo['frame_rate']

		#make figure directory for stimulus type
		tiff_fig_dir = os.path.join(fig_base_dir, stack_info['stimulus'])
		if not os.path.exists(tiff_fig_dir):
				os.makedirs(tiff_fig_dir)

		data_fn = 'retino_data_%s.h5' %(tiff_fn[:-4])
		if os.path.isfile(os.path.join(file_dir,data_fn)): #check if we already analyzed this tiff
			print('TIFF already analyzed!')
		else:
			analyze_tiff(tiff_path_full,tiff_fn,stack_info, RETINOID,file_dir,tiff_fig_dir, masks_file)
	if RETINOID['PARAMS']['roi_type'] != 'pixels':
		masks_file.close()

#-----------------------------------------------------
#           MAIN SET OF ACTIONS
#-----------------------------------------------------

def main(options):

	do_analysis(options)


if __name__ == '__main__':
	main(sys.argv[1:])

