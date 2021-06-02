import numpy as np
import sys
import os
import glob
import optparse

# option to import from github folder
sys.path.insert(0, '/n/coxfs01/2p-pipeline/repos/suite2p')
import suite2p
from suite2p.run_s2p import run_s2p
import shutil

#%%

def get_ops(save_path='', scratch_folder='/scratch/tmp'):
    #using parameters optimized for 2x arm
    ops = {
            'fast_disk': scratch_folder, # used to store temporary binary file, defaults to save_path0 (set as a string NOT a list)
            'save_path0': save_path, # analysis_dir, # stores results, defaults to first item in data_path
            'delete_bin': False, # whether to delete binary file after processing
            # main settings
            'nplanes' : 1, # each tiff has these many planes in sequence
            'nchannels' : 1, # each tiff has these many channels per plane
            'functional_chan' : 1, # this channel is used to extract functional ROIs (1-based)
            'diameter':10, # this is the main parameter for cell detection, 2-dimensional if Y and X are different (e.g. [6 12])
            'tau':  0.7, # this is the main parameter for deconvolution
            'fs': 44.65,  # sampling rate (total across planes)
            # output settings
            'save_mat': False, # whether to save output as matlab files
            'combined': True, # combine multiple planes into a single result /single canvas for GUI
            # parallel settings
            'num_workers': 0, # 0 to select num_cores, -1 to disable parallelism, N to enforce value
            'num_workers_roi': -1, # 0 to select number of planes, -1 to disable parallelism, N to enforce value
            # registration settings
            'do_registration': True, # whether to register data
            'nimg_init': 200, # subsampled frames for finding reference image
            'batch_size': 1000, # number of frames per batch
            'maxregshift': 0.1, # max allowed registration shift, as a fraction of frame max(width and height)
            'align_by_chan' : 1, # when multi-channel, you can align by non-functional channel (1-based)
            'reg_tif': False, # whether to save registered tiffs
            'subpixel' : 10, # precision of subpixel registration (1/subpixel steps)
            #if using previously generate registration
       #     'source_reg' : registered_dir,
            # non rigid registration settings
            'nonrigid': True, # whether to use nonrigid registration
            'block_size': [128, 128], # block size to register
            #'keep_movie_raw': True,
            #'two_step_registration': True,
            'snr_thresh': 1.2, # if any nonrigid block is below this threshold, it gets smoothed until above this threshold. 1.0 results in no smoothing
            'maxregshiftNR': 5, # maximum pixel shift allowed for nonrigid, relative to rigid
            # cell detection settings
            'connected': True, # whether or not to keep ROIs fully connected (set to 0 for dendrites)
            'navg_frames_svd': 5000, # max number of binned frames for the SVD
            'nsvd_for_roi': 1000, # max number of SVD components to keep for ROI detection
            'max_iterations': 30, # maximum number of iterations to do cell detection
            'ratio_neuropil': 5.0, # ratio between neuropil basis size and cell radius
            'ratio_neuropil_to_cell': 3, # minimum ratio between neuropil radius and cell radius
            'tile_factor': 3.0, # use finer (>1) or coarser (<1) tiles for neuropil estimation during cell detection
            'threshold_scaling': 1., # adjust the automatically determined threshold by this scalar multiplier
            'max_overlap': 0.75, # cells with more overlap than this get removed during triage, before refinement
            'inner_neuropil_radius': 2, # number of pixels to keep between ROI and neuropil donut
            'outer_neuropil_radius': np.inf, # maximum neuropil radius
            'min_neuropil_pixels': 100, # minimum number of pixels in the neuropil
            'high_pass': 100, # running mean subtraction with window of size 'high_pass' (use low values for 1P)
            # deconvolution settings
            'baseline': 'maximin', # baselining mode
            'win_baseline': 60., # window for maximin
            'sig_baseline': 10., # smoothing constant for gaussian filter
            'prctile_baseline': 8.,# optional (whether to use a percentile baseline)
            'neucoeff': 1.0,  # neuropil coefficient
            #custom arguments
            'add_global_baseline': True,
            'add_neuropil_baseline': True,
          }
    return ops


def extract_options(options):
    choices_sourcetype = ('raw', 'mcorrected', 'bidi')
    default_sourcetype = 'mcorrected'

    parser = optparse.OptionParser()

    # PATH opts:
    parser.add_option('-D', '--root', action='store', dest='rootdir', default='/n/coxfs01/2p-data', help='data root dir (root project dir containing all animalids) [default: /n/coxfs01/2pdata]')
    parser.add_option('-T', '--scratch', action='store', dest='scratchdir', default='/scratch/tmp', help='scratch dir for binaries [default: /scratch/tmp]')


    parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')
    parser.add_option('-S', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD')
    parser.add_option('-A', '--acq', action='store', dest='fov', default='FOV1_zoom2p0x', help="acquisition folder (ex: 'FOV1_zoom2p0x') [default: FOV1_zoom2p0x]")

    #parser.add_option('-R', '--run', action='store', dest='run', default='', help="name of run dir containing tiffs to be processed (ex: gratings_phasemod_run1)")


    #parser.add_option('-s', '--tiffsource', action='store', dest='tiffsource', default=None, help="Name of folder containing tiffs to be processed (ex: processed001). Should be child of <run>/processed/")

    (options, args) = parser.parse_args(options)

    return options


def main(options):
    #provide some info
    optsE = extract_options(options)
    rootdir = optsE.rootdir #'/n/coxfs01/2p-data'

    animalid = optsE.animalid #'JC085'
    session = optsE.session #'20190624'
    fov = optsE.fov #'FOV1_zoom4p0x'
    #run = 'all_combined'
    #scratch_root = optsE.scratchdir #'/scratch/tmp'

    #analysis_header = 'suite2p_analysis001'
    #analysis_header = 'suite2p'

    #figure out directories to search
    data_dir = os.path.join(rootdir, animalid, session, fov, 'all_combined') #run)
    print(data_dir)
    
    # check if analyses exist
    existing_dirs = glob.glob(os.path.join(data_dir, 'processed', 'suite2p*'))
    dir_index = len(existing_dirs)+1
    analysis_header = 'suite2p%03d' % dir_index
    analysis_dir = os.path.join(data_dir, 'processed', analysis_header)
    print("Found %i existing dirs. Current header: %s" % (len(existing_dirs), analysis_header))

    #raw_dir = glob.glob(os.path.join(data_dir,'raw'))[0]
    #raw_dir = glob.glob(os.path.join(data_dir,'raw*'))[0]
    raw_dir = glob.glob(os.path.join(data_dir, 'block_reduced'))[0]
    print(raw_dir)
    # #scratch folder for binaries
    scratch_folder = os.path.join(analysis_dir,'binaries')
    #scratch_folder = os.path.join(scratch_root,animalid,session,acquisition,run,analysis_header)

    # registered_analysis = 'suite2p_analysis001'
    # #folder with previous registration, if using it
    # registered_dir = os.path.join(data_dir,'processed',registered_analysis)

    #set parameters and save bit
    ops_dir = os.path.join(analysis_dir,'suite2p')
    if not os.path.isdir(ops_dir):
        os.makedirs(ops_dir)

    print(analysis_dir)
    # set your options for running
    # overwrites the run_s2p.default_ops
    ops = get_ops(save_path=analysis_dir, scratch_folder=scratch_folder)

     # provide an h5 path in 'h5py' or a tiff path in 'data_path'
    # db overwrites any ops (allows for experiment specific settings)
    db = {
          'h5py': [], # a single h5 file path
          'h5py_key': 'data',
          'look_one_level_down': False, # whether to look in ALL subfolders when searching for tiffs
          'data_path': [raw_dir], # a list of folders with tiffs 
                                                 # (or folder of folders with tiffs if look_one_level_down is True, or subfolders is not empty)
                                                
          'subfolders': [], # choose subfolders of 'data_path' to look in (optional)
          'fast_disk': scratch_folder, # string which specifies where the binary file will be stored (should be an SSD)
      #    'tiff_list': ['FOV1_retino_00001.tif'] # list of tiffs in folder * data_path *!
        }


    print('Saving to: %s'%(ops_dir))
    np.savez(os.path.join(ops_dir,'ops0.npz'),ops=ops,db=db)


if __name__=='__main__':
    main(sys.argv[1:])
