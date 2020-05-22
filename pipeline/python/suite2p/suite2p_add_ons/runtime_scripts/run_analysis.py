import numpy as np
import sys
import os
import glob
# option to import from github folder
#sys.path.insert(0, '/n/coxfs01/cechavarria/repos/suite2p')
import suite2p
from suite2p.run_s2p import run_s2p
import shutil
import optparse
from custom_scripts import unpack_ops

from custom_scripts import custom_figure_functions



 
def extract_options(options):
	parser = optparse.OptionParser()

	# PATH opts:
	parser.add_option('-D', '--root', action='store', dest='rootdir', default='/n/coxfs01/2p-data', help='source dir (root project dir containing all expts) [default: /n/coxfs01/2p-data]')
	parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')
	parser.add_option('-S', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID') 
	parser.add_option('-A', '--acq', action='store', dest='acquisition', default='', help="acquisition folder (ex: 'FOV1_zoom3x')")
	parser.add_option('-R', '--run', action='store', dest='run', default='', help='name of run to process') 
	parser.add_option('-Y', '--analysis', action='store', dest='analysis', default='', help='Analysis to process. [ex: suite2p_analysis001]')


	(options, args) = parser.parse_args() 

	return options

def run_analysis(options):

	#unpack options
	rootdir = options.rootdir
	animalid = options.animalid
	session = options.session
	acquisition = options.acquisition
	run = options.run

	analysis_header = options.analysis



	#figure out directories to search
	data_dir = os.path.join(rootdir,animalid,session,acquisition,run)
	analysis_dir = os.path.join(data_dir,'processed',analysis_header)


	#loading set parameters
	data = np.load(os.path.join(analysis_dir,'suite2p','ops0.npz'), allow_pickle=True)

	ops = data['ops'].item()
	db = data['db'].item()

	#make scratch folder for bianries
	if not os.path.isdir(ops['fast_disk']):
		os.makedirs(ops['fast_disk']) 
		

	# #create folders and copy files, if we want to use a previously-generated registration
	if 'source_reg' in ops.keys():
		print('Copying registration info from: %s'%(ops['source_reg']))
		reg_fn = os.path.join(ops['source_reg'],'suite2p','ops1.npy')

		dest_dir = os.path.join(analysis_dir,'suite2p')
		if not os.path.isdir(dest_dir):
			os.makedirs(dest_dir)
		print('Copying registration info to: %s'%(dest_dir))

		shutil.copy(reg_fn,dest_dir)

		#update relevant fields in ops file [assuming planar data only..]
		ops_prev = np.load(os.path.join(dest_dir,'ops1.npy')).item()

		ops_prev['save_path'] = os.path.join(ops['save_path0'], 'suite2p', 'plane%d'%0)
		ops_prev['ops_path'] = os.path.join(ops_prev['save_path'],'ops.npy')
		ops_prev['reg_file'] = os.path.join(ops['fast_disk'], 'suite2p', 'plane%d'%0, 'data.bin')

		ops1 = np.array([ops_prev])
		np.save(os.path.join(dest_dir,'ops1.npy'),ops1)
		
		#make output directory
		if not os.path.isdir(ops_prev['save_path']):
			os.makedirs(ops_prev['save_path'])
		
	binary_dir = os.path.join(analysis_dir,'binaries')
	if 'source_reg' in ops.keys():#copy binaries
			print('Copying binaries from:%s'%(os.path.join(ops['source_reg'],'binaries')))
			shutil.copytree(os.path.join(ops['source_reg'],'binaries'),binary_dir)

	# # #make folder for binaries
	# binary_dir = os.path.join(analysis_dir,'binaries')
	# if not os.path.isdir(binary_dir):
	# 	if 'source_reg' in ops.keys():#copy binaries
	# 		print('Copying binaries from:%s'%(os.path.join(ops['source_reg'],'binaries')))
	# 		shutil.copytree(os.path.join(ops['source_reg'],'binaries'),binary_dir)
	# 		#move them to scratch folder
	# 		shutil.move(os.path.join(binary_dir,'suite2p'),ops['fast_disk'])
	# 	else:
	# 		os.makedirs(binary_dir) 
	# else:
	# 	# #binaries exist, move them to scratch folder
	# 	shutil.move(os.path.join(binary_dir,'suite2p'),ops['fast_disk'])


	print('-----------Running Suite2p------------')
	# run analysis
	opsEnd=run_s2p(ops=ops,db=db)
	print('---------Suite2p Done---------')


	# #move binaries to analysis directory, we will move them back later
	# # #binaries exist, move them to scratch folder
	# print('Moving binaries to %s:'%(binary_dir))
	# shutil.move(os.path.join(ops['fast_disk'],'suite2p'),binary_dir)

	#unpack ops
	unpack_ops.unpack_file(options)

	#make figures
	custom_figure_functions.make_roi_figures(options)


#-----------------------------------------------------
#           MAIN SET OF ACTIONS
#-----------------------------------------------------

def main(options):
	
	options = extract_options(options)
	run_analysis(options)
	

	
#%%

if __name__ == '__main__':
	main(sys.argv[1:])

