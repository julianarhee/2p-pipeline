
import matplotlib

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import optparse
matplotlib.use('Agg')
import numpy as np
import sys
import os
import json_tricks



def unpack_file(options):
	#get options
	rootdir = options.rootdir
	animalid = options.animalid
	session = options.session
	acquisition = options.acquisition
	run = options.run
	analysis_header = options.analysis

	 
	#figure out directories to search
	data_dir = os.path.join(rootdir,animalid,session,acquisition,run)
	analysis_dir = os.path.join(data_dir,'processed',analysis_header)
	ops_dir = os.path.join(analysis_dir,'suite2p')
	output_dir = os.path.join(analysis_dir,'suite2p','plane0')

	#make figure directory
	fig_dir = ops_dir
	#get file
	ops_file = os.path.join(ops_dir,'ops1.npy')
	ops_data = np.load(ops_file)


	#get rid of images in dictionary
	tmp = ops_data[0].copy()
	for x in ops_data[0]:
	    if isinstance(tmp[x],np.ndarray):
	        if tmp[x].size > 100:#get rid of large arrays
	            print(x)
	            del tmp[x]
	    elif isinstance(tmp[x],np.int32):#convert remaining arrays to lists
	        tmp[x] = int(tmp[x])

	tmp['outer_neuropil_radius'] = 'Inf'#change to string
	#dump to json
	with open(os.path.join(ops_dir,'ops1.json'), 'w') as fp:
		json_tricks.dump(tmp, fp, indent=4)


	#saving some images to dictionary

	#MEAN IMAGE
	M = ops_data[0]['meanImg']

	dpi = 80
	szY,szX = M.shape
	# What size does the figure need to be in inches to fit the image?
	figsize = szX / float(dpi), szY / float(dpi)

	# Create a figure of the right size with one axes that takes up the full figure
	f = plt.figure(figsize=figsize)
	ax = f.add_axes([0, 0, 1, 1])

	# Hide spines, ticks, etc.
	ax.axis('off')

	ax.imshow(M,'gray')
	f.savefig(os.path.join(fig_dir,'meanImg.png'), dpi=dpi, transparent=True)
	plt.close()

	#MEAN IMAGE ENHANCED
	M = ops_data[0]['meanImgE']

	dpi = 80
	szY,szX = M.shape
	# What size does the figure need to be in inches to fit the image?
	figsize = szX / float(dpi), szY / float(dpi)

	# Create a figure of the right size with one axes that takes up the full figure
	f = plt.figure(figsize=figsize)
	ax = f.add_axes([0, 0, 1, 1])

	# Hide spines, ticks, etc.
	ax.axis('off')

	ax.imshow(M,'gray')
	f.savefig(os.path.join(fig_dir,'meanImgE.png'), dpi=dpi, transparent=True)
	plt.close()

	#CORRELATION IMAGE
	M = ops_data[0]['Vcorr']

	dpi = 80
	szY,szX = M.shape
	# What size does the figure need to be in inches to fit the image?
	figsize = szX / float(dpi), szY / float(dpi)

	# Create a figure of the right size with one axes that takes up the full figure
	f = plt.figure(figsize=figsize)
	ax = f.add_axes([0, 0, 1, 1])

	# Hide spines, ticks, etc.
	ax.axis('off')

	ax.imshow(M,'gray')
	f.savefig(os.path.join(fig_dir,'Vcorr.png'), dpi=dpi, transparent=True)
	plt.close()



class Struct():
    pass

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




#-----------------------------------------------------
#           MAIN SET OF ACTIONS
#-----------------------------------------------------

def main(options):
    
    print('Wirting ops file to Json')
    options = extract_options(options)
    unpack_file(options)
    

    
#%%

if __name__ == '__main__':
    main(sys.argv[1:])
