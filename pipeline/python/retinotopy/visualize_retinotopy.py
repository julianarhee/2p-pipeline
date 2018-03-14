
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
from scipy import ndimage, misc,interpolate,stats,signal
import seaborn as sns
import cv2
from pipeline.python.utils import atoi , natural_keys

pp = pprint.PrettyPrinter(indent=4)

#-----------------------------------------------------
#           FUNCTIONS FOR DATA PROCESSING
#-----------------------------------------------------

def smooth_phase_array(theta,sigma,sz):
    #build 2D Gaussian Kernel
    kernelX = cv2.getGaussianKernel(sz, sigma); 
    kernelY = cv2.getGaussianKernel(sz, sigma); 
    kernelXY = kernelX * kernelY.transpose(); 
    kernelXY_norm=np.true_divide(kernelXY,np.max(kernelXY.flatten()))
    
    #get x and y components of unit-length vector
    componentX=np.cos(theta)
    componentY=np.sin(theta)
    
    #convolce
    componentX_smooth=signal.convolve2d(componentX,kernelXY_norm,mode='same',boundary='symm')
    componentY_smooth=signal.convolve2d(componentY,kernelXY_norm,mode='same',boundary='symm')

    theta_smooth=np.arctan2(componentY_smooth,componentX_smooth)
    return theta_smooth

def smooth_array(inputArray,fwhm,phaseArray=False):
    szList=np.array([None,5,None,11,None,21,None,27,None,31,None,37,None,43,None,49,None,53,None,59,None,55,None,69,None,79,None,89,None,99])
    sigmaList=np.array([None,.5,None,.9,None,1.7,None,2.6,None,3.4,None,4.3,None,5.1,None,6.4,None,6.8,None,7.6,None,8.5,None,9.4,None,10.3,None,11.2,None,12])
    sigma=sigmaList[fwhm]
    sz=szList[fwhm]
    if phaseArray:
        outputArray=smooth_phase_array(inputArray,sigma,sz)
    else:
        outputArray=cv2.GaussianBlur(inputArray, (sz,sz), sigma, sigma)
    return outputArray

def extract_options(options):

	parser = optparse.OptionParser()

	# PATH opts:
	parser.add_option('-R', '--root', action='store', dest='rootdir', default='/nas/volume1/2photon/data', help='data root dir (root project dir containing all animalids) [default: /nas/volume1/2photon/data, /n/coxfs01/2pdata if --slurm]')
	parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')
	parser.add_option('-S', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID')
	parser.add_option('-A', '--acq', action='store', dest='acquisition', default='FOV1', help="acquisition folder (ex: 'FOV1_zoom3x') [default: FOV1]")
	parser.add_option('-r', '--run', action='store', dest='run', default='', help="name of run dir containing tiffs to be processed (ex: gratings_phasemod_run1)")
	parser.add_option('-a', '--analysis-id', action='store', dest='analysis_id', default='', help="ANALYSIS ID for retinoid param set to use (created with set_analysis_parameters.py, e.g., analysis001,  etc.)")
	parser.add_option('--thresh', action='store', dest='ratio_thresh', default=None, help="threshold to use for magnitude ratio")
	parser.add_option('--fwhm', action='store', dest='smooth_fwhm', default=None, help="full-width at half-max of smoothing kernel(odd integer)")
	parser.add_option('--slurm', action='store_true', dest='slurm', default=False, help="Set if running as SLURM job on Odyssey")
	(options, args) = parser.parse_args(options)

	if options.slurm is True:
		if 'coxfs01' not in options.rootdir:
			options.rootdir = '/n/coxfs01/2p-data'

	return options


def visualize_retino(options):

	#Get options
	options = extract_options(options)

	# # Set USER INPUT options:
	rootdir = options.rootdir
	animalid = options.animalid
	session = options.session
	acquisition = options.acquisition
	analysis_id = options.analysis_id
	run = options.run
	ratio_thresh = options.ratio_thresh
	if ratio_thresh is not None:
		ratio_thresh = float(ratio_thresh)
	smooth_fwhm = options.smooth_fwhm
	if smooth_fwhm is not None:
		smooth_fwhm = int(smooth_fwhm)


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

	#set file and figure directory
	file_dir = os.path.join(RETINOID['DST'],'files')
	if not os.path.exists(file_dir):
			os.makedirs(file_dir)

	fig_base_dir = os.path.join(RETINOID['DST'],'figures')
	if not os.path.exists(fig_base_dir):
			os.makedirs(fig_base_dir)

	print 'Getting paradigm file info'
	para_file_dir = os.path.join(run_dir,'paradigm','files')
	para_file =  [f for f in os.listdir(para_file_dir) if f.endswith('.json')][0]#assuming a single file for all tiffs in run

	with open(os.path.join(para_file_dir, para_file), 'r') as r:
		parainfo = json.load(r)

	#cycle through tiff stax
	for tiff_count, tiff_fn in enumerate(tiff_files):
		print '**** Visualizing %s *****'%(tiff_fn)

		tiff_path_full = os.path.join(RETINOID['PARAMS']['tiff_source'],tiff_fn)

		s_string=''
		t_string=''
		folder_name = ''

		#read in data
		data_fn = 'retino_data_%s.h5' %(tiff_fn[:-4])
		file_grp = h5py.File(os.path.join(file_dir,data_fn),  'r')

		#unpack some info
		szx,szy,nframes = file_grp.attrs['sz_info']

		#arrays into maps
		mag_ratio_map = np.reshape(file_grp['mag_ratio_array'][:],(szy,szx))
		phase_map = np.reshape(file_grp['phase_array'][:],(szy,szx))

		if RETINOID['PARAMS']['downsample_factor'] is not None:
			f = float(RETINOID['PARAMS']['downsample_factor'])
			mag_ratio_map = cv2.resize(mag_ratio_map,(0,0),fx = f, fy = f)
			phase_map = cv2.resize(phase_map,(0,0),fx = f, fy = f)

		#smooth image, if indicated
		if smooth_fwhm is not None:
			phase_map=smooth_array(phase_map,smooth_fwhm,phaseArray=True)
			mag_ratio_map=smooth_array(mag_ratio_map,smooth_fwhm)
			s_string='_fwhm_'+str(smooth_fwhm)
			folder_name = 'fwhm_'+str(smooth_fwhm)

		#set phase map range for visualization
		phase_map_disp=np.copy(phase_map)
		phase_map_disp[phase_map<0]=-phase_map[phase_map<0]
		phase_map_disp[phase_map>0]=(2*np.pi)-phase_map[phase_map>0]


		#apply threshhold
		if ratio_thresh is not None:
			phase_map_disp[mag_ratio_map<ratio_thresh]=np.nan
			t_string='_thresh_'+str(ratio_thresh)+'_'
			if folder_name == '':
				folder_name = 'thresh_'+str(ratio_thresh)
			else:
				folder_name = folder_name + '_thresh_'+str(ratio_thresh)
			
		#make figure directory for stimulus type
		fig_dir = os.path.join(fig_base_dir, parainfo[str(tiff_count+1)]['stimuli']['stimulus'],folder_name)
		if not os.path.exists(fig_dir):
				os.makedirs(fig_dir)

		print 'Outputting figures to: %s'%(fig_dir)


		# #Read in average image (for viuslization)
		avg_dir = os.path.join('%s_mean_slices'%(str(RETINOID['SRC'])),'visible')
		s0 = tiff_fn[:-4]
		s1 = s0[s0.find('Slice'):]

		avg_fn = 'vis_mean_%s.tif'%(s1)

		im0 = tf.imread(os.path.join(avg_dir, avg_fn))
		#if RETINOID['PARAMS']['downsample_factor'] is not None:
		 #   im0 = block_mean(im0,int(RETINOID['PARAMS']['downsample_factor']))
		im1 = np.uint8(np.true_divide(im0,np.max(im0))*255)
		im_back = np.dstack((im1,im1,im1))

		fig_fn = 'overlay_image%s%s%s.png'%(s_string, t_string,tiff_fn[:-4])

		dpi = 80
		szY,szX,szZ = im_back.shape
		# What size does the figure need to be in inches to fit the image?
		figsize = szX / float(dpi), szY / float(dpi)

		# Create a figure of the right size with one axes that takes up the full figure
		fig = plt.figure(figsize=figsize)
		ax = fig.add_axes([0, 0, 1, 1])

		# Hide spines, ticks, etc.
		ax.axis('off')

		ax.imshow(im_back, 'gray')
		ax.imshow(phase_map_disp,'nipy_spectral',alpha=.5,vmin=0,vmax=2*np.pi)
		fig.savefig(os.path.join(fig_dir, fig_fn), dpi=dpi, transparent=True)
		plt.close()


		#save some histograms
		sns.set_style("darkgrid", {"axes.facecolor": ".9"})

		fig_fn = 'phase_hist%s%s%s.png'%(s_string, t_string,tiff_fn[:-4])

		phase_info = np.copy(phase_map_disp)
		phase_info = np.true_divide(phase_info, 2*np.pi)
		bin_loc = np.arange(0,1.05,.05)
		plt.hist(np.ndarray.flatten(phase_info[~np.isnan(phase_info)]),bin_loc)
		plt.xlabel('Phase')
		plt.ylabel('Pixel Count')
		plt.savefig(os.path.join(fig_dir,fig_fn))
		plt.close()



		#get area of each blob
		labeled, nr_objects = ndimage.label(~np.isnan(phase_map_disp) )
		pix_area = np.zeros((nr_objects,))
		phase_mean = np.zeros((nr_objects,))
		phase_sd = np.zeros((nr_objects,))
		ratio_mean = np.zeros((nr_objects,))
		for i in range(nr_objects):
			pix_area[i] = len(np.where(labeled==i+1)[0])
			phase_mean[i]= np.nanmean(phase_map_disp[np.where(labeled==i+1)[0],np.where(labeled==i+1)[1]])
			phase_sd[i] = np.nanstd(phase_map_disp[np.where(labeled==i+1)[0],np.where(labeled==i+1)[1]])
			ratio_mean[i] = np.nanmean(mag_ratio_map[np.where(labeled==i+1)[0],np.where(labeled==i+1)[1]])
			
		if np.max(pix_area)>np.min(pix_area):
			fig_fn = 'blob_area_hist%s%s%s.png'%(s_string, t_string,tiff_fn[:-4])
			bin_loc = np.arange(0,np.max(pix_area)+10,10)
			plt.hist(pix_area,bin_loc)
			plt.xlabel('Pixel Area')
			plt.ylabel('Region Count')
			plt.savefig(os.path.join(fig_dir,fig_fn))
			plt.close()

		if np.max(phase_mean)>np.min(phase_mean):
			fig_fn = 'blob_phase_mean%s%s%s.png'%(s_string, t_string,tiff_fn[:-4])
			bin_loc = np.arange(0,np.max(phase_mean)+.2,.2)
			plt.hist(phase_mean,bin_loc)
			plt.xlabel('Mean Phase')
			plt.ylabel('Region Count')
			plt.savefig(os.path.join(fig_dir,fig_fn))
			plt.close()

		if np.max(phase_sd)>np.min(phase_sd):
			fig_fn = 'blob_phase_sd%s%s%s.png'%(s_string, t_string,tiff_fn[:-4])
			bin_loc = np.arange(0,np.max(phase_sd)+.05,.05)
			plt.hist(phase_sd, bin_loc)
			plt.xlabel('SD Phase')
			plt.ylabel('Region Count')
			plt.savefig(os.path.join(fig_dir,fig_fn))
			plt.close()

		if np.max(ratio_mean)>np.min(ratio_mean):
			fig_fn = 'blob_ratio_mean%s%s%s.png'%(s_string, t_string,tiff_fn[:-4])
			#bin_loc = np.arange(0,np.max(phase_mean)+.002,.002)
			plt.hist(ratio_mean)
			plt.xlabel('Mean Mag Ratio')
			plt.ylabel('Region Count')
			plt.savefig(os.path.join(fig_dir,fig_fn))
			plt.close()





#-----------------------------------------------------
#           MAIN SET OF ACTIONS
#-----------------------------------------------------

def main(options):

	visualize_retino(options)


if __name__ == '__main__':
	main(sys.argv[1:])
