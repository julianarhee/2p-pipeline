import numpy as np
import sys
import os
import glob
# option to import from github folder
sys.path.insert(0, '/n/coxfs01/cechavarria/repos/suite2p')
import suite2p
from suite2p.run_s2p import run_s2p

# #%%
# def extract_options(options):
#     choices_sourcetype = ('raw', 'mcorrected', 'bidi')
#     default_sourcetype = 'mcorrected'

#     parser = optparse.OptionParser()

#     # PATH opts:
#     parser.add_option('-D', '--root', action='store', dest='rootdir', default='/n/coxfs01/2p-data', help='data root dir (root project dir containing all animalids) [default: /nas/volume1/2photon/data, /n/coxfs01/2pdata if --slurm]')

#     parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')
#     parser.add_option('-S', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID')
#     parser.add_option('-A', '--acq', action='store', dest='acquisition', default='FOV1', help="acquisition folder (ex: 'FOV1_zoom3x') [default: FOV1]")
#     parser.add_option('-R', '--run', action='store', dest='run', default='', help="name of run dir containing tiffs to be processed (ex: gratings_phasemod_run1)")

#     (options, args) = parser.parse_args(options)


# def main(options):

#     tid = create_tid(options)

#     print "****************************************************************"
#     print "Created TRACE ID."
#     print "----------------------------------------------------------------"
#     pp.pprint(tid)
#     print "****************************************************************"


# if __name__ == '__main__':
#     main(sys.argv[1:])

#provide some info
rootdir = '/n/coxfs01/2p-data'
animalid = 'CE077'
session = '20180612'
acquisition = 'FOV1_zoom1x'
run = 'blobs_run1'
scratch_folder = '/scratch/tmp'

if not os.path.isdir(scratch_folder):
    os.mkdir(scratch_folder)  

#figure out directories to search
data_dir = os.path.join(rootdir,animalid,session,acquisition,run)
raw_dir = glob.glob(os.path.join(data_dir,'raw*'))[0]
print(raw_dir)

analysis_header = 'suite2p_analysis001'
analysis_dir = os.path.join(data_dir,analysis_header)

# set your options for running
# overwrites the run_s2p.default_ops
ops = {
        'fast_disk': scratch_folder, # used to store temporary binary file, defaults to save_path0 (set as a string NOT a list)
        'save_path0': analysis_dir, # stores results, defaults to first item in data_path
        'delete_bin': True, # whether to delete binary file after processing
        # main settings
        'nplanes' : 1, # each tiff has these many planes in sequence
        'nchannels' : 1, # each tiff has these many channels per plane
        'functional_chan' : 1, # this channel is used to extract functional ROIs (1-based)
        'diameter':10, # this is the main parameter for cell detection, 2-dimensional if Y and X are different (e.g. [6 12])
        'tau':  1., # this is the main parameter for deconvolution
        'fs': 44.67,  # sampling rate (total across planes)
        # output settings
        'save_mat': False, # whether to save output as matlab files
        'combined': True, # combine multiple planes into a single result /single canvas for GUI
        # parallel settings
        'num_workers': 0, # 0 to select num_cores, -1 to disable parallelism, N to enforce value
        'num_workers_roi': -1, # 0 to select number of planes, -1 to disable parallelism, N to enforce value
        # registration settings
        'do_registration': True, # whether to register data
        'nimg_init': 200, # subsampled frames for finding reference image
        'batch_size': 200, # number of frames per batch
        'maxregshift': 0.1, # max allowed registration shift, as a fraction of frame max(width and height)
        'align_by_chan' : 1, # when multi-channel, you can align by non-functional channel (1-based)
        'reg_tif': True, # whether to save registered tiffs
        'subpixel' : 10, # precision of subpixel registration (1/subpixel steps)
        # non rigid registration settings
        'nonrigid': True, # whether to use nonrigid registration
        'block_size': [128, 128], # block size to register
        'snr_thresh': 1.2, # if any nonrigid block is below this threshold, it gets smoothed until above this threshold. 1.0 results in no smoothing
        'maxregshiftNR': 5, # maximum pixel shift allowed for nonrigid, relative to rigid
        # cell detection settings
        'connected': True, # whether or not to keep ROIs fully connected (set to 0 for dendrites)
        'navg_frames_svd': 5000, # max number of binned frames for the SVD
        'nsvd_for_roi': 1000, # max number of SVD components to keep for ROI detection
        'max_iterations': 20, # maximum number of iterations to do cell detection
        'ratio_neuropil': 6., # ratio between neuropil basis size and cell radius
        'ratio_neuropil_to_cell': 3, # minimum ratio between neuropil radius and cell radius
        'tile_factor': 1., # use finer (>1) or coarser (<1) tiles for neuropil estimation during cell detection
        'threshold_scaling': 1., # adjust the automatically determined threshold by this scalar multiplier
        'max_overlap': 0.75, # cells with more overlap than this get removed during triage, before refinement
        'inner_neuropil_radius': 2, # number of pixels to keep between ROI and neuropil donut
        'outer_neuropil_radius': np.inf, # maximum neuropil radius
        'min_neuropil_pixels': 350, # minimum number of pixels in the neuropil
        'high_pass': 100, # running mean subtraction with window of size 'high_pass' (use low values for 1P)
        # deconvolution settings
        'baseline': 'maximin', # baselining mode
        'win_baseline': 60., # window for maximin
        'sig_baseline': 10., # smoothing constant for gaussian filter
        'prctile_baseline': 8.,# optional (whether to use a percentile baseline)
        'neucoeff': .7,  # neuropil coefficient
      }

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
    }

# run one experiment
opsEnd=run_s2p(ops=ops,db=db)