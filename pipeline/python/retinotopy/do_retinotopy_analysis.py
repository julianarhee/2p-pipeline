
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import h5py
import json
import copy
import re
import optparse
import pprint
import tifffile as tf
import pylab as pl
import numpy as np
from scipy import ndimage
import cv2
import glob
from pipeline.python.paradigm import process_mw_files as mw

from pipeline.python.utils import natural_keys, replace_root
from pipeline.python.retinotopy.visualize_rois import roi_retinotopy
from pipeline.python.traces import get_traces as traces

pp = pprint.PrettyPrinter(indent=4)

def extract_options(options):

    parser = optparse.OptionParser()

    # PATH opts:
    parser.add_option('-D', '--root', action='store', dest='rootdir', default='/n/coxfs01/2p-data', 
                        help='data root dir (root project dir containing all animalids) [default: /n/coxfs01/2pdata if --slurm]')
    parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')
    parser.add_option('-S', '--session', action='store', dest='session', default='', help='session (format: YYYMMDD_ANIMALID')
    parser.add_option('-A', '--acq', action='store', dest='acquisition', default='FOV1', help="acquisition (ex: 'FOV1_zoom3x') [default: FOV1]")
    parser.add_option('-R', '--run', action='store', dest='run', default='', help="run containing tiffs to be processed (ex: gratings_phasemod_run1)")
    parser.add_option('-d', '--analysis-id', action='store', dest='analysis_id', default='', 
                        help="ANALYSIS ID for retinoid param set to use (created with set_analysis_parameters.py, e.g., analysis001,  etc.)")
    parser.add_option('--slurm', action='store_true', dest='slurm', default=False, help="Set if running as SLURM job on Odyssey")

    parser.add_option('-a', '--np-niter', action='store', dest='np_niter', default=20, help="n iterations for creating neuropil annulus (default: 20. standard is 20 for zoom2p0x, 10 for zoom4p0x")


    (options, args) = parser.parse_args(options)

    if options.slurm is True and 'coxfs01' not in options.rootdir:
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

def block_mean_stack(stack0, ds_factor, along_axis=2):
    if along_axis==2:
        im0 = block_mean(stack0[:,:,0],ds_factor) 
        print im0.shape
        stack1 = np.zeros((im0.shape[0],im0.shape[1],stack0.shape[2]))
        for i in range(0,stack0.shape[2]):
            stack1[:,:,i] = block_mean(stack0[:,:,i],ds_factor) 
    else:
        # This is for downsampling masks:
        im0 = block_mean(stack0[0,:,:],ds_factor) 
        print "... block mean on MASKS by %i (target: %s)" % (ds_factor, str(im0.shape))

        stack1 = np.zeros((stack0.shape[0], im0.shape[0], im0.shape[1]))
        for i in range(stack0.shape[0]):
            stack1[i,:,:] = block_mean(stack0[i,:,:],ds_factor) 

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

def get_processed_stack(tiff_path_full,RETINOID, slicenum=0):
    # Read in RAW tiff: 
    print('Loading file : %s'%(tiff_path_full))
    stack0 = tf.imread(tiff_path_full)
    print stack0.shape
    # CHeck for 2 channels!
    rundir = tiff_path_full.split('/processed/')[0]
    runinfo_fpath = glob.glob(os.path.join(rundir, '*.json'))[0]
    with open(runinfo_fpath, 'r') as f: runinfo = json.load(f)
    nvolumes = runinfo['nvolumes']
    nslices = len(runinfo['slices'])
    if stack0.shape[0] != nvolumes:
        if runinfo['nchannels'] == 2:
            tmpstack = copy.copy(stack0)
            stack0 = tmpstack[0::2, :, :]
            print "Got SIGNAL channel 01 only."
        elif nslices > 1:
            tmpstack = copy.copy(stack0)
            stack0 = tmpstack[slicenum::nslices, :, :]
            print "Got SLICE %i only." % int(slicenum+1)


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



	

def analyze_tiff(tiff_path_full,tiff_fn,stack_info, RETINOID,file_dir,tiff_fig_dir,masks_file,slicenum=0, np_cfactor=0.7):


    #intialize file to save data
        
    data_str = tiff_fn[:-4]
    s0 = data_str 
    file_str = str(re.search('File(\d{3})', tiff_fn).group(0))
    slice_str = str(re.search('Slice(\d{2})', tiff_fn).group(0))
    print "***** Analysing: %s, %s *****" % (file_str, slice_str)

    #file_str = s0[s0.find('File'):s0.find('File')+7]
    #slice_str = s0[s0.find('Slice'):s0.find('Slice')+7]

    if 'Slice' not in data_str:
        data_str = '%s_Slice%02d' % (data_str, int(slicenum+1))
    print data_str

    data_fn = 'retino_data_%s.h5' % data_str
    file_grp = h5py.File(os.path.join(file_dir,data_fn),  'w')
    
    # Initialize file to to save extracted ROI traces:
    traces_dir = os.path.join(file_dir.split('/files')[0], 'traces')
    if not os.path.exists(traces_dir): 
        os.makedirs(traces_dir)
    traces_fn = os.path.join(traces_dir, 'extracted_traces.h5')
    traces_outfile = h5py.File(traces_fn, 'a')

    #get tiff stack
    tiff_stack = get_processed_stack(tiff_path_full,RETINOID,slicenum=slicenum)
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
        s0 = data_str #tiff_fn[:-4]
        file_str = s0[s0.find('File'):s0.find('File')+7]
        slice_str = s0[s0.find('Slice'):s0.find('Slice')+7]
        masks = masks_file[file_str]['masks'][slice_str][:]
        if RETINOID['PARAMS']['downsample_factor'] is not None:
            masks = block_mean_stack(masks, int(RETINOID['PARAMS']['downsample_factor']), along_axis=0)

        nmasks, szy, szx= masks.shape

        #apply masks to stack
        roi_trace = get_mask_traces(tiff_stack,masks)
    
    else:
        #get saved masks for this tiff stack
        assert file_str in masks_file.keys(), "... warped mask for file %s not found." % file_str
        mask_file_str = file_str         

        slice_str = s0[s0.find('Slice'):s0.find('Slice')+7]
        #assert slice_str in masks_file[mask_file_str].keys(), "... warped mask for file %s -- slice %s --  not found." % (file_str, slice_str)
        if slice_str not in masks_file[mask_file_str].keys():
            print "...warped mask for file %s -- slice %s --  not found." % (file_str, slice_str) 
            return

        mask_slice_str = slice_str
        print('...processing file %s | slice %s' % (file_str, slice_str))
                

        #print masks_file[masks_file.keys()[0]]['masks'].keys()
        #slice_str = masks_file[file_str]['masks'].keys()[0]
		#masks = masks_file[file_str]['masks'][slice_str][:]
        masks = masks_file[mask_file_str][mask_slice_str]['maskarray'][:]
        # reshape mask array
        d1, d2 = masks_file[mask_file_str][mask_slice_str]['zproj'].shape
        nrois = masks.shape[-1]
        masks = np.reshape(masks, (d1, d2, nrois))
        
        # swap axes to make it nrois, d2, d1
        masks = masks.T #np.swapaxes(0, 2)
        print "...Loaded processed masks: %s" % str(masks.shape)

        # swap axes for familiarity
    	    masks = np.swapaxes(masks,1,2) # visualization  

        if RETINOID['PARAMS']['downsample_factor'] is not None:
            masks = block_mean_stack(masks, int(RETINOID['PARAMS']['downsample_factor']), along_axis=0)
            
        print "...MASKS:", masks.shape
        print "...TIFFS:", tiff_stack.shape
        
        nrois, mask_d1, mask_d2 = masks.shape
        tiff_d1, tiff_d2, nframes = tiff_stack.shape 
        if mask_d1 > tiff_d1:
            ds_1 = mask_d1 / tiff_d1
            ds_2 = mask_d2/ tiff_d2
            print "...Resizing masks from %s by (%i, %i)." % (str(masks.shape), ds_1, ds_2)
            masks_ds = np.empty((nrois, tiff_d1, tiff_d2), dtype=masks.dtype)
            for ri in range(nrois):
                mask_ds = cv2.resize(masks[ri, :, :], (mask_d1//ds_1, mask_d2//ds_2), interpolation=cv2.INTER_NEAREST)
                masks_ds[ri, :, :] = mask_ds
            masks = copy.copy(masks_ds)
            print "...Resized masks: %s" % str(masks.shape)
            
        nmasks, szy, szx= masks.shape

        #apply masks to stack
        #roi_trace = get_mask_traces(tiff_stack,masks)

        # apply masks to stack and do neuropil correction:
        np_maskarray = masks_file[mask_file_str][mask_slice_str]['np_maskarray'][:]
        print("...NP masks: %s" % str(np_maskarray.shape))

        if RETINOID['PARAMS']['downsample_factor'] is not None:
            np_masks = np.reshape(np_maskarray, (d1, d2, nrois))
            # swap axes to make it nrois, d2, d1
            np_masks = np_masks.T #np.swapaxes(0, 2)
            
            #swap axes for familiarity
            np_masks = np.swapaxes(np_masks,1,2) # visualization  
            np_masks = block_mean_stack(np_masks, int(RETINOID['PARAMS']['downsample_factor']), along_axis=0)
            print "...Reshaping np masks: %s" % str(np_masks.shape)
            np_maskarray = np.reshape(np_masks, (nmasks, szx*szy)).T

        masks_r = np.reshape(masks, (nmasks, szx*szy))
        tiffs_r = np.reshape(tiff_stack, (tiff_d1*tiff_d2, nframes))
 
        #  TRACES outfile:  Save processed roi trace
        mset = traces_outfile.create_dataset('/'.join([file_str, 'masks']), masks_r.shape, masks_r.dtype)
        mset[...] = masks_r     
        mset.attrs['nmasks'] = nmasks
        mset.attrs['dims'] = (szx, szy)
        npset = traces_outfile.create_dataset('/'.join([file_str, 'np_masks']), np_maskarray.shape, np_maskarray.dtype)
        npset[...] = np_maskarray     
        npset.attrs['nmasks'] = nmasks
        npset.attrs['dims'] = (szx, szy)
        
        # APply masks to tiff:
        soma_trace = masks_r.dot(tiffs_r)
        np_trace = np_maskarray.T.dot(tiffs_r)
        roi_trace = soma_trace - (np_cfactor * np_trace)
 
        print "... applied masks. roi_trace shape: %s" % str(roi_trace.shape)

        #  TRACES outfile:  Save extracted traces
        rset = traces_outfile.create_dataset('/'.join([file_str, 'raw']), soma_trace.shape, soma_trace.dtype)
        rset[...] = soma_trace 
        nset = traces_outfile.create_dataset('/'.join([file_str, 'neuropil']), np_trace.shape, np_trace.dtype)
        nset[...] = np_trace 
        #nset.attrs['correction_factor'] = np_cfactor
        tset = traces_outfile.create_dataset('/'.join([file_str, 'corrected']), roi_trace.shape, roi_trace.dtype)
        tset[...] = roi_trace 
        tset.attrs['correction_factor'] = np_cfactor


        	roi_trace = process_array(roi_trace, RETINOID, stack_info)
    
    #  TRACES outfile:  Save processed roi trace
    pset = traces_outfile.create_dataset('/'.join([file_str, 'processed']), roi_trace.shape, roi_trace.dtype)
    pset[...] = roi_trace 
    pset.attrs['source'] = tiff_path_full
    pset.attrs['dims'] = szx, szy, nframes
    traces_outfile.close()
    print("... Extracted traces!")


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
   
    # Add fit signal to retino output:
    if 'signal_fit' not in file_grp.keys():
        fitset = file_grp.create_dataset('signal_fit', signal_fit.shape, signal_fit.dtype)
        fitset[...] = signal_fit


        
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

        fig_name = 'best_pixel_power_%s.png' % data_str #(tiff_fn[:-4])
        fig=plt.figure()
        plt.plot(freqs,mag_data[max_mod_idx,:])
        plt.xlabel('Frequency (Hz)',fontsize=16)
        plt.ylabel('Magnitude',fontsize=16)
        axes = plt.gca()
        ymin, ymax = axes.get_ylim()
        plt.axvline(x=freqs[freq_idx], ymin=ymin, ymax = ymax, linewidth=1, color='r')
        plt.savefig(os.path.join(tiff_fig_dir,fig_name))
        plt.close()

        fig_name = 'best_pixel_power_zoom_%s.png' % data_str #(tiff_fn[:-4])
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

        fig_name = 'best_pixel_frequency_fit_%s.png' % data_str #(tiff_fn[:-4])
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


        fig_name = 'phase_map_%s.png' % data_str #(tiff_fn[:-4])
        #set phase map range for visualization
        phase_map_disp=np.copy(phase_map)
        phase_map_disp[phase_map<0]=-phase_map[phase_map<0]
        phase_map_disp[phase_map>0]=(2*np.pi)-phase_map[phase_map>0]

        fig=plt.figure()
        plt.imshow(phase_map_disp,'nipy_spectral',vmin=0,vmax=2*np.pi)
        plt.colorbar()
        plt.savefig(os.path.join(tiff_fig_dir,fig_name))
        plt.close()

        fig_name = 'mag_map_%s.png' % data_str #(tiff_fn[:-4])
        fig=plt.figure()
        plt.imshow(mag_map)
        plt.colorbar()
        plt.savefig(os.path.join(tiff_fig_dir,fig_name))
        plt.close()

        fig_name = 'mag_ratio_map_%s.png' % data_str #(tiff_fn[:-4])
        fig=plt.figure()
        plt.imshow(mag_ratio_map)
        plt.colorbar()
        plt.savefig(os.path.join(tiff_fig_dir,fig_name))
        plt.close()

        fig_name = 'beta_map_%s.png' % data_str #(tiff_fn[:-4])
        fig=plt.figure()
        plt.imshow(beta_map)
        plt.colorbar()
        plt.savefig(os.path.join(tiff_fig_dir,fig_name))
        plt.close()

        fig_name = 'var_exp_map_%s.png' % data_str #(tiff_fn[:-4])
        fig=plt.figure()
        plt.imshow(varexp_map)
        plt.colorbar()
        plt.savefig(os.path.join(tiff_fig_dir,fig_name))
        plt.close()

        # #Read in average image (for viuslization)
        avg_dir = os.path.join('%s_mean_deinterleaved'%(str(RETINOID['SRC'])),'visible')
        
        # Find corresponding avg img/file path by current file name:
        curr_file = str(re.search('File(\d{3})', tiff_fn).group(0))
        curr_slice = 'Slice%02d' % int(slicenum+1)
        #curr_slice = str(re.search('Slice(\d{2})', tiff_fn).group(0))


        avg_img_path = glob.glob(os.path.join(avg_dir, '*%s_*%s.tif' % (curr_slice, curr_file)))[0] 
        print "Loaded avg img: %s" % avg_img_path

        im0 = tf.imread(avg_img_path)

#		s0 = tiff_fn[:-4]
#		s1 = s0[s0.find('Slice'):]
#		avg_fn = 'vis_mean_%s.tif'%(s1)
#		im0 = tf.imread(os.path.join(avg_dir, avg_fn))
        if RETINOID['PARAMS']['downsample_factor'] is not None:
            ds = int(RETINOID['PARAMS']['downsample_factor'])
            im0 = block_mean(im0,ds)


        # Resize images to make square:
        im_d1, im_d2 = im0.shape
        if im_d1 != im_d2:
            dim_r = max([im_d1, im_d2])
            im0 = cv2.resize(im0, (dim_r, dim_r))
#            magratio_roi = cv2.resize(magratio_roi, (dim_r, dim_r))
#            mag_roi = cv2.resize(mag_roi, (dim_r, dim_r))
#            varexp_roi = cv2.resize(varexp_roi, (dim_r, dim_r))
            phase_map_disp = cv2.resize(phase_map_disp, (dim_r, dim_r))
            
            

        im1 = np.uint8(np.true_divide(im0,np.max(im0))*255)
        im2 = np.dstack((im1,im1,im1))

        fig_name = 'phase_map_overlay_%s.png' % data_str #curr_file #(tiff_fn[:-4])


        fig=plt.figure()
        plt.imshow(im2,'gray')
        plt.imshow(phase_map_disp,'nipy_spectral',alpha=0.25,vmin=0,vmax=2*np.pi)
        plt.colorbar()
        plt.savefig(os.path.join(tiff_fig_dir,fig_name))
        plt.close()
    else:

        #make figure directory for stimulus type
        fig_dir = os.path.join(tiff_fig_dir, '%s_%s' % (file_str, slice_str),'spectrum')
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


        fig_dir = os.path.join(tiff_fig_dir, '%s_%s' % (file_str, slice_str),'timecourse')
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
            #print(len(maskpix))
            magratio_roi[maskpix]=mag_ratio_array[midx]
            mag_roi[maskpix]=mag_array[midx]
            varexp_roi[maskpix]=varexp_array[midx]
            phase_roi[maskpix]=phase_array_disp[midx]

        fig_dir = tiff_fig_dir

        # #Read in average image (for viuslization)
        avg_dir = os.path.join('%s_mean_deinterleaved'%(str(RETINOID['SRC'])),'visible')
        # Find corresponding avg img/file path by current file name:
        curr_file = str(re.search('File(\d{3})', tiff_fn).group(0))
        curr_slice = 'Slice%02d' % int(slicenum+1)
        avg_img_path = glob.glob(os.path.join(avg_dir, '*%s_*%s.tif' % (curr_slice, curr_file)))[0] 
        print "Loaded avg img: %s" % avg_img_path

        im0 = tf.imread(avg_img_path)

#		s0 = tiff_fn[:-4]
#		s1 = s0[s0.find('Slice'):]
#		avg_fn = 'vis_mean_%s.tif'%(s1)
#		im0 = tf.imread(os.path.join(avg_dir, avg_fn))
        if RETINOID['PARAMS']['downsample_factor'] is not None:
            ds = int(RETINOID['PARAMS']['downsample_factor'])
            im0 = block_mean(im0,ds)
            
        # Resize images to make square:
        im_d1, im_d2 = im0.shape
        if im_d1 != im_d2:
            dim_r = max([im_d1, im_d2])
            im0 = cv2.resize(im0, (dim_r, dim_r))
            magratio_roi = cv2.resize(magratio_roi, (dim_r, dim_r))
            mag_roi = cv2.resize(mag_roi, (dim_r, dim_r))
            varexp_roi = cv2.resize(varexp_roi, (dim_r, dim_r))
            phase_roi = cv2.resize(phase_roi, (dim_r, dim_r))
            
            
        im1 = np.uint8(np.true_divide(im0,np.max(im0))*255)
        im2 = np.dstack((im1,im1,im1))

        fig_name = 'phase_info_%s.png' % data_str #curr_file #(tiff_fn[:-4])
        fig=plt.figure()
        plt.imshow(im2,'gray')
        plt.imshow(phase_roi,'nipy_spectral',alpha = 0.5,vmin=0,vmax=2*np.pi)
        plt.axis('off')
        plt.colorbar()
        plt.savefig(os.path.join(fig_dir,fig_name))
        plt.close()

        fig_name = 'mag_info_%s.png' % data_str #curr_file #(tiff_fn[:-4])
        fig=plt.figure()
        plt.imshow(im2,'gray')
        plt.imshow(mag_roi, alpha = 0.5)
        plt.axis('off')
        plt.colorbar()
        plt.savefig(os.path.join(fig_dir,fig_name))
        plt.close()

        fig_name = 'mag_ratio_info_%s.png' % data_str #curr_file #(tiff_fn[:-4])
        fig=plt.figure()
        plt.imshow(im2,'gray')
        plt.imshow(magratio_roi, alpha = 0.5)
        plt.axis('off')
        plt.colorbar()
        plt.savefig(os.path.join(fig_dir,fig_name))
        plt.close()

        fig_name = 'varexp_info_%s.png' % data_str #curr_file #(tiff_fn[:-4])
        fig=plt.figure()
        plt.imshow(im2,'gray')
        plt.imshow(varexp_roi, alpha = 0.5)
        plt.axis('off')
        plt.colorbar()
        plt.savefig(os.path.join(fig_dir,fig_name))
        plt.close()

        fig_name = 'phase_nice_%s.png' % data_str #curr_file #(tiff_fn[:-4])
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
            fig_fn = 'roi_mag_ratio_%s.png'% data_str #(tiff_fn[:-4])
            bin_loc = np.arange(0,np.max(mag_ratio_array)+.002,.002)
            plt.hist(mag_ratio_array,bin_loc)
            plt.xlabel('Magnitude Ratio')
            plt.ylabel('ROI Count')
            plt.savefig(os.path.join(fig_dir,fig_fn))
            plt.close()

        if np.max(phase_array)>np.min(phase_array):
            fig_fn = 'roi_phase_%s.png'% data_str #(tiff_fn[:-4])
            bin_loc = np.arange(0,(2*np.pi)+.2,.2)
            plt.hist(phase_array,bin_loc)
            plt.xlabel('Phase')
            plt.ylabel('ROI Count')
            plt.savefig(os.path.join(fig_dir,fig_fn))
            plt.close()

        if np.max(varexp_array)>np.min(varexp_array):
            fig_fn = 'roi_varexp_%s.png' % data_str #(tiff_fn[:-4])
            bin_loc = np.arange(0,np.max(varexp_array)+.01,.01)
            plt.hist(varexp_array,bin_loc)
            plt.xlabel('Variance Explained')
            plt.ylabel('ROI')
            plt.savefig(os.path.join(fig_dir,fig_fn))
            plt.close()



#%%


options = ['--slurm', '-i', 'JC097', '-S', '20190615', '-A', 'FOV4_zoom1p0x', '-R', 'retino_run1', 
           '-d', 'analysis003', '-a', 20]


#%%

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

    np_niter = int(options.np_niter)

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
    if rootdir not in tiff_dir:
        tiff_dir = replace_root(tiff_dir, rootdir, animalid, session)

    #roi_name = RETINOID['PARAMS']['roi_id']
    tiff_files = sorted([t for t in os.listdir(tiff_dir) if t.endswith('tif')], key=natural_keys)



    # Get associated RUN info:
    runmeta_path = os.path.join(run_dir, '%s.json' % run)
    with open(runmeta_path, 'r') as r:
        runinfo = json.load(r)

    nslices = len(runinfo['slices'])
    nchannels = runinfo['nchannels']
    nvolumes = runinfo['nvolumes']
    ntiffs = runinfo['ntiffs']

    #Set file and figure directory
    retino_dest_dir = RETINOID['DST']
    if rootdir not in retino_dest_dir:
        retino_dest_dir = replace_root(retino_dest_dir, rootdir, animalid, session)

    file_dir = os.path.join(retino_dest_dir, 'files')
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
                    
    fig_base_dir = os.path.join(retino_dest_dir, 'figures')
    if not os.path.exists(fig_base_dir):
        os.makedirs(fig_base_dir)


    #-----Get info from paradigm file
    para_file_dir = os.path.join(run_dir,'paradigm','files')
    if not os.path.exists(para_file_dir): os.makedirs(para_file_dir)
    para_files =  [f for f in os.listdir(para_file_dir) if f.endswith('.json')]#assuming a single file for all tiffs in run
    if len(para_files) == 0:
        # Paradigm info not extracted yet:
        raw_para_files = [f for f in glob.glob(os.path.join(run_dir, 'raw*', 'paradigm_files', '*.mwk')) if not f.startswith('.')]
        print run_dir
        assert len(raw_para_files) == 1, "No raw .mwk file found, and no processed .mwk file found. Aborting!"
        raw_para_file = raw_para_files[0]           
        print "Extracting .mwk trials: %s" % raw_para_file 
        fn_base = os.path.split(raw_para_file)[1][:-4]
        trials = mw.extract_trials(raw_para_file, retinobar=True, trigger_varname='frame_trigger', verbose=True)
        para_fpath = mw.save_trials(trials, para_file_dir, fn_base)
        para_file = os.path.split(para_fpath)[-1]
    else:
        assert len(para_files) == 1, "Unable to find unique .mwk file..."
        para_file = para_files[0]

    print 'Getting paradigm file info from %s'%(os.path.join(para_file_dir, para_file))

    with open(os.path.join(para_file_dir, para_file), 'r') as r:
        parainfo = json.load(r)


    #%%OPEN MASK FILE
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

        #mask_path = os.path.join(rid_dst, 'masks.hdf5')
        #masks_file = h5py.File(mask_path,  'r')#read
        mask_path = os.path.join(retino_dest_dir, 'MASKS.hdf5')
        if not os.path.exists(mask_path):
            maskinfo = traces.get_mask_info(RETINOID, RID, nslices=nslices, rootdir=rootdir)
            print "************ masks info **************"
            print maskinfo

            mask_path = traces.get_masks(mask_path, maskinfo, RID, save_warp_images=True, do_neuropil_correction=True, niter=np_niter, rootdir=rootdir)
        masks_file = h5py.File(mask_path, 'r')

    else:
        masks_file = None


    # check for tiffs to exclude:
    excluded_tiffs = RETINOID['PARAMS']['excluded_tiffs']
    ex_tifs = [t for t in tiff_files if str(re.search('File(\\d{3})', t).group(0)) in excluded_tiffs]
    print "EXCLUDING:", ex_tifs
    tiff_files = [t for t in tiff_files if t not in ex_tifs]


    #%%
    # =============================================================================
    # Cyce through TIFF stacks and analayze
    # =============================================================================
    #TODO: skip tiff if already processed
    for tiff_count, tiff_fn in enumerate(tiff_files):    	    

        tiff_path_full = os.path.join(RETINOID['PARAMS']['tiff_source'],tiff_fn)

        #get some info from paradigm and put into a dict to pass to some fxns later
        stack_info = dict()
        tiffnum = int(str(re.search('File(\\d{3})', tiff_fn).group(0))[4:])

        stack_info['stimulus'] = parainfo[str(tiffnum)]['stimuli']['stimulus']
        stack_info['stimfreq'] = parainfo[str(tiffnum)]['stimuli']['scale']
        stack_info['frame_rate'] = runinfo['frame_rate']
        stack_info['nslices'] = len(runinfo['slices'])

        #make figure directory for stimulus type
        tiff_fig_dir = os.path.join(fig_base_dir, stack_info['stimulus'])
        if not os.path.exists(tiff_fig_dir): os.makedirs(tiff_fig_dir)
       
        for slicenum in range(nslices):
 
            # Check if analyzed file exists for current tif, else analyze:
            #curr_file = str(re.search('File(\d{3})', tiff_fn).group(0))
            fid_str = os.path.splitext(tiff_fn)[0]
            data_str = 'retino_data_%s' % fid_str #(tiff_fn[:-4])
            if 'Slice' not in data_str:
                data_str = '%s_Slice%02d' % (data_str, int(slicenum+1))
            data_fn = '%s.h5' % data_str
            analyzed_fpath = os.path.join(file_dir, data_fn)
            if os.path.isfile(analyzed_fpath) and os.stat(analyzed_fpath).st_size > 0: #check if we already analyzed this tiff
                print('TIFF already analyzed!')
            else:
                analyze_tiff(tiff_path_full,tiff_fn,stack_info, RETINOID,file_dir,tiff_fig_dir, masks_file, slicenum=slicenum)

    if RETINOID['PARAMS']['roi_type'] != 'pixels':
        masks_file.close()
        
        visualize_opts = ['-D', rootdir, '-i', animalid, '-S', session, '-A', acquisition,
                            '-R', run, '-t', analysis_id]

        roi_retinotopy(visualize_opts)
        print "Plotted ROIs to screen coordinates."

#-----------------------------------------------------
#           MAIN SET OF ACTIONS
#-----------------------------------------------------

def main(options):

    do_analysis(options)


if __name__ == '__main__':
    main(sys.argv[1:])

