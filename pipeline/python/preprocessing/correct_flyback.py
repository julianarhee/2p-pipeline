#!/usr/bin/env python2
'''
Run this script to parse RAW acquisition files (.tif) from SI.
Native format is int16. This script will remove all artifact and discard frames, and save to new .tif (int16, uint16).

Append _visible.tif to crop display range for viewing on BOSS.

Run python correct_flyback.py -h for all input options.
'''
import sys
import os
import numpy as np
import tifffile as tf
from skimage import img_as_uint
from skimage import exposure
import optparse
import json
import scipy.io
import shutil
from json_tricks.np import dump, dumps, load, loads
import re
from stat import S_IREAD, S_IRGRP, S_IROTH, S_IWRITE, S_IWGRP, S_IWOTH
from pipeline.python.set_pid_params import get_default_pid
from checksumdir import dirhash
import pprint
pp = pprint.PrettyPrinter(indent=4)

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def do_flyback_correction(options):

    parser = optparse.OptionParser()

    # PATH opts:
    # -----------------------------------------------------------
    parser.add_option('-R', '--root', action='store', dest='rootdir', default='/nas/volume1/2photon/data', help='source dir (root project dir containing all expts) [default: /nas/volume1/2photon/data]')
    parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID') 

    parser.add_option('-S', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID') 
    parser.add_option('-A', '--acq', action='store', dest='acquisition', default='', help="acquisition folder (ex: 'FOV1_zoom3x')")
    parser.add_option('-r', '--run', action='store', dest='run', default='', help="name of run dir containing tiffs to be processed (ex: gratings_phasemod_run1)")
    parser.add_option('-p', '--pid', action='store', dest='pid_hash', default='', help="PID hash of current processing run (6 char), default will create new if set_pid_params.py not run")
    parser.add_option('-H', '--raw', action='store', dest='source_hash', default='', help="PID hash of raw source dir (6 char), default will use rawtiff_dir hash stored in runinfo metadata (from get_scanimage_data.py)")

    # TIFF saving opts:
    # parser.add_option('--native', action='store_false', dest='uint16', default=True, help='Keep int16 tiffs as native [default: convert to uint16]')
    parser.add_option('--visible', action='store_true', dest='visible', default=False, help='Also create tiffs with adjusted display-range to make visible.')
    parser.add_option('-m', '--min', action='store', dest='displaymin', default=10, help='min display range value [default: 10]')
    parser.add_option('-M', '--max', action='store', dest='displaymax', default=2000, help='max display range value [default: 2000]')

    # SUBSTACK opts (for correcting flyback):
    # -----------------------------------------------------------
    parser.add_option('--correct-flyback', action='store_true', dest='correct_flyback', default=False, help='Create substacks of data-only by removing all artifact/discard frames [default: False]')

    parser.add_option('--flyback', action='store', dest='nflyback', default=0, help='Num flyback frames at top of stack [default: 0]')
    parser.add_option('--discard', action='store', dest='ndiscard', default=0, help='Num discard frames at end of stack [default: 0]')
    parser.add_option('--notiffs', action='store_false', dest='save_tiffs', default=True, help='Flag to run without saving TIFFs')
 
    # TIFF acquisition info:
    # -----------------------------------------------------------
    parser.add_option('-c', '--nchannels', action='store', dest='nchannels', default=1, help='Num interleaved channels in raw tiffs to be processed [default: 1]')
    parser.add_option('-v', '--nvolumes', action='store', dest='nvolumes', default=1, help='Num volumes acquired [default: 1]')
    parser.add_option('-z', '--nslices', action='store', dest='nslices', default=1, help='Num slices specified, no discard [default: 1]')

    # FOV-cropping opts (prob shouldn't go here):
    # -----------------------------------------------------------
    # parser.add_option('--crop', action='store_true', dest='crop_fov', default=False, help='Crop FOV in x,y for smaller data size [default: False]')
    # parser.add_option('-X', '--xstart', action='store', dest='x_startidx', default=0, help='Starting idx for x-dimension, i.e., columns [default: 0]')
    # parser.add_option('-Y', '--ystart', action='store', dest='y_startidx', default=0, help='Starting idx for y-dimension, i.e., rows [default: 0]')
    # parser.add_option('-W', '--width', action='store', dest='width', default='', help='Width of FOV, i.e., cols [default: '']')
    # parser.add_option('-H', '--height', action='store', dest='height', default='', help='Height of FOV, i.e., rows [default: '']')


    (options, args) = parser.parse_args(options) 
    save_tiffs = options.save_tiffs
    if save_tiffs is True:
        print "Correcting flyback and saving TIFFs."
    else:
        print "Not saving TIFFs, just getting meta-data."

    # crop_fov = options.crop_fov
    # x_startidx = int(options.x_startidx)
    # y_startidx = int(options.y_startidx)

    correct_flyback = options.correct_flyback
    #uint16 = options.uint16
    nflyback = int(options.nflyback)
    ndiscard = int(options.ndiscard)
    nchannels = int(options.nchannels)
    nvolumes = int(options.nvolumes)
    nslices = int(options.nslices)
    nslices_full = nslices + ndiscard
    print "nvolumes:", nvolumes
    print "nslices specified:", nslices
    print "n expected after substack:", nslices - nflyback

    rootdir = options.rootdir 
    animalid = options.animalid
    session = options.session 
    acquisition = options.acquisition
    run = options.run
    pid_hash = options.pid_hash
    source_hash = options.source_hash
    
    # -------------------------------------------------------------
    # Set basename for files created containing meta/reference info:
    # -------------------------------------------------------------
    raw_simeta_basename = 'SI_%s' % run #functional_dir
    run_info_basename = '%s' % run #functional_dir
    pid_info_basename = 'pids_%s' % run
    # -------------------------------------------------------------

    visible = options.visible
    if visible:
	    displaymin = options.displaymin
	    displaymax = options.displaymax

    # Identify RAW tiff dir from acquisition:
    # -----------------------------------------------------------------------------------
    acquisition_dir = os.path.join(rootdir, animalid, session, acquisition)
    raw_tiff_dir = [os.path.join(acquisition_dir, run, rd) for rd in os.listdir(os.path.join(acquisition_dir, run)) if 'raw' in rd and os.path.isdir(os.path.join(acquisition_dir, run, rd))][0] # SHOULD ONLY BE ONE SINGLE RAW TIFF FOLDER
    
    if source_hash not in raw_tiff_dir or len(source_hash) == 0:
        print "WARNING*********************************"
        print "Source hash does not match RAW tiff dir:\n%s" % raw_tiff_dir
        print "Using default stored rawtiff_dir from run info struct: %s" % run_info_basename
        print "****************************************"
        with open(os.path.join(acquisition_dir, run, '%s.json' % run_info_basename), 'r') as f:
            runmeta = json.load(f)
        raw_tiff_dir = os.path.join(acquisition_dir, run, runmeta['rawtiff_dir'])
        print raw_tiff_dir
    else:
        print "Checking for flyback correction for tiffs from source:\n%s" % raw_tiff_dir
        
    # Set and create PROCESSED write dir:
    # -----------------------------------------------------------------------------------
    processed_dir = os.path.join(acquisition_dir, run, 'processed') #, 'DATA', 'Raw')    
    # if not os.path.exists(processed_dir):
    #     os.makedirs(processed_dir)

    # Load PID dict to access pids and hashes:
    # -----------------------------------------------------------------------------------
    if os.path.exists(os.path.join(processed_dir, '%s.json' % pid_info_basename)):
        with open(os.path.join(processed_dir, '%s.json' % pid_info_basename), 'r') as f:
            processdict = json.load(f)
    else:
        processdict = dict()
    
    
    # Load user-created PID tmp file, or create default if none exists (no bidir- or motion-correction):
    # -----------------------------------------------------------------------------------
    tmp_pid_dir = os.path.join(processed_dir, 'tmp_pids')
    if not os.path.exists(tmp_pid_dir):
        os.makedirs(tmp_pid_dir)
        
    tmp_pids = [p for p in os.listdir(tmp_pid_dir) if p.endswith('json')]
    if len(pid_hash) == 0:
        print "No PID hash specified..."
        if len(tmp_pids) == 1:
            pid_hash = tmp_pids[0]
            print "But only 1 PID process exists anyway:", tmp_pids[0]
            
    print "Running PID %s" % pid_hash
    tmp_pid_fn = [h for h in tmp_pids if pid_hash in h]
    if len(tmp_pid_fn) > 0:
        # PID Exists
        with open(os.path.join(tmp_pid_dir, tmp_pid_fn[0]), 'r') as f:
            PID = json.load(f)
    else:
        # ONLY do flyback correction, create PID with default params:
        PID = get_default_pid(rootdir=rootdir, animalid=animalid, session=session, acquisition=acquisition,
                              run=run, correct_flyback=correct_flyback, nflyback_frames=nflyback)
    
    
    pid_hash = PID['tmp_hashid']    
    tmp_pid_fn = 'tmp_pid_%s.json' % pid_hash
    print "PID %s: FLYBACK CORRECTION step for PID:" % pid_hash
    pp.pprint(PID)            
    
    # Set specific PID dir for current processing run:
    
    # Increment process_id so can't overwrite, even though source is always "raw"
    # Write flyback-corrected TIFFs to 'raw' subdir (may want to play with n flyback corrected)
    # processed_dirs = [p for p in os.listdir(save_basepath) if os.path.isdir(os.path.join(save_basepath, p))] 
    # increment_processed = int(len(processed_dirs) + 1)
    # process_id = 'processed%03d' % increment_processed
    # savepath = os.path.join(save_basepath, process_id, 'raw')
    # if not os.path.exists(savepath):
    #     os.makedirs(savepath)

    # -----------------------------------------------------------------------------
    # Update SOURCE/DEST paths for current PID, if needed:
    # -----------------------------------------------------------------------------
    paramspath = os.path.join(tmp_pid_dir, tmp_pid_fn)
    
    processed_folder = '%s_%s' % (PID['process_id'], pid_hash)
    processed_dest_path = os.path.join(processed_dir, processed_folder)
    if not os.path.exists(processed_dest_path):
        os.makedirs(processed_dest_path)
        
    # Update processing SRC/sourcedir, if default 'raw' changed to raw_<hashid>:
    if source_hash not in PID['SRC']:
        PID['SRC'] = raw_tiff_dir
        PID['PARAMS']['preprocessing']['sourcedir'] = raw_tiff_dir
    
    # Update processing DST/(destdir) with pid_hash:
    if pid_hash not in PID['DST']:
        PID['DST'] = processed_dest_path
        
    write_dir = os.path.join(PID['DST'], 'raw')
    if correct_flyback:
        if not os.path.exists(write_dir):
            os.makedirs(write_dir)
        PID['PARAMS']['preprocessing']['destdir'] = write_dir
    
    with open(paramspath, 'w') as f:
        json.dump(PID, f, indent=4, sort_keys=True)
        
    source_dir = PID['PARAMS']['preprocessing']['sourcedir']
    # -----------------------------------------------------------------------------

    
    # GET TIFFS:
    tiffs = os.listdir(source_dir)
    tiffs = sorted([t for t in tiffs if t.endswith('.tif')], key=natural_keys)
    print "Found %i TIFFs." % len(tiffs)
    
    for tiffidx,tiffname in enumerate(tiffs):
        # Adjust TIFF file name so that all included files increment correctly,
        # and naming format matches standard:	
        origname = tiffname.split('.')[0]
    	prefix = '_'.join(origname.split('_')[0:-1])
        prefix = prefix.replace('-', '_')
        newtiff_fn = '%s_File%03d.tif' % (prefix, int(tiffidx+1)) 
       
        if correct_flyback:
            print "Creating file in DATA dir:", newtiff_fn
            
            # Read in RAW tiff: 
            stack = tf.imread(os.path.join(source_dir, tiffs[tiffidx]))
    	    # if uint16:
    	    # stack = img_as_uint(stack) 
    	    print "TIFF: %s" % tiffs[tiffidx]
    	    print "size: ", stack.shape
    	    print "dtype: ", stack.dtype
    	   
    	    # First, remove DISCARD frames:
    	    nslices_orig = nslices_full - ndiscard # N specified slices during acquis.     
            if save_tiffs is True:
                start_idxs = np.arange(0, stack.shape[0], nslices_full*nchannels)
                substack = np.empty((nslices_orig*nchannels*nvolumes, stack.shape[1], stack.shape[2]), dtype=stack.dtype)
                print "Removing SI discard frames. Tmp stack shape:", substack.shape 
 
                newstart = 0
                for x in range(len(start_idxs)):    
                    substack[newstart:newstart+(nslices_orig*nchannels),:,:] = stack[start_idxs[x]:start_idxs[x]+(nslices_orig*nchannels), :, :]
                    newstart = newstart + (nslices_orig*nchannels)
             
                print "Removed discard frames. New substack shape is: ", substack.shape    
 
            # Also get frame indices of kept-frames to exclude discard: 
            start_idxs_single = np.arange(0, stack.shape[0]/nchannels, nslices_full)
            allframe_idxs = np.arange(0, stack.shape[0]/nchannels)
            frame_idxs = np.empty((nslices_orig*nvolumes,))
            print "Getting frame idxs. Initial n frames:", len(frame_idxs)
            newstart = 0 
            for x in range(len(start_idxs_single)):
                frame_idxs[newstart:newstart+(nslices_orig)] = allframe_idxs[start_idxs_single[x]:start_idxs_single[x] + (nslices_orig)]
                newstart = newstart + (nslices_orig) 
    	     
    	    # Next, crop off FLYBACK frames: 
            nslices_crop = nslices_orig - nflyback 
            if save_tiffs is True:    
                start_idxs = np.arange(nflyback*nchannels, substack.shape[0], nslices_orig*nchannels)
                final = np.empty((nslices_crop*nchannels*nvolumes, substack.shape[1], substack.shape[2]), dtype=stack.dtype)
            
                newstart = 0
                for x in range(len(start_idxs)):
                    final[newstart:newstart+(nslices_crop*nchannels),:,:] = substack[start_idxs[x]:start_idxs[x]+(nslices_crop*nchannels), :, :]
                    newstart = newstart + (nslices_crop*nchannels)
                
                print "Removed flyback frames. Final shape is: ", final.shape

                # Write substack to DATA dir: 
                print "Saving..."
                tf.imsave(os.path.join(write_dir, newtiff_fn), final)
 
            # Again, get frame-idxs of kept-frames: 
            start_idxs_single = np.arange(nflyback, nslices_orig*nvolumes, nslices_orig)
            frame_idxs_final = np.empty((nslices_crop*nvolumes,))
            newstart = 0
            for x in range(len(start_idxs_single)):
                frame_idxs_final[newstart:newstart+(nslices_crop)] = frame_idxs[start_idxs_single[x]:start_idxs_single[x]+(nslices_crop)]
                newstart = newstart + (nslices_crop)
     
            print "Created frame-idx array. Final shape: ", frame_idxs_final.shape
                   		    
     	else:
#     	    print "Not creating substacks from input tiffs."
#     	    if uint16:
#                 print "Converting raw tiff to uint16."
#                 stack = tf.imread(os.path.join(raw_tiff_dir, tiffs[tiffidx]))
#     	        final = img_as_uint(stack)
    
#                 dtype_fn = '%s_uint16.tif' % newtiff_fn.split('.')[0] 
#     	        if save_tiffs is True: 
#                     tf.imsave(os.path.join(write_dir, dtype_fn), final)
     	    
            frame_idxs_final = [] # if not index correction needed, just leave blank 
             
#     	if crop_fov:
#             if not correct_flyback:  # stack not yet read in:
#                 final = tf.imread(os.path.join(raw_tiff_dir, tiffs[tiffidx]))
    
#     	    if len(options.width)==0:
#     		    width = int(input('No width specified. Starting idx is: %i.\nEnter image width: ' % x_startidx))
#     	    else:
#     		    width = int(options.width)
#     	    if len(options.height)==0:
#     		    height = int(input('No height specified. Starting idx is: %i.\nEnter image height: ' % y_startidx))
#     	    else:
#     		    height = int(options.height)
    
#     	    x_endidx = x_startidx + width
#     	    y_endidx = y_startidx + height
    	    
#     	    final = final[:, y_startidx:y_endidx, x_startidx:x_endidx]
#     	    print "Cropped FOV. New size: ", final.shape
                
#             # TODO: add extra info to SI-meta or reference-struct, if want to keep this option...
#             cropped_fn = '%s_cropped.tif' % newtiff_fn.split('.')[0] #'File%03d_visible.tif' % int(tiffidx+1)
#     	    if save_tiffs is True:
#                 tf.imsave(os.path.join(write_dir, cropped_fn), final)
    	    
    	if visible: 
    	    ranged = exposure.rescale_intensity(final, in_range=(displaymin, displaymax))
    	    rangetiff_fn = '%s_visible.tif' % newtiff_fn.split('.')[0] #'File%03d_visible.tif' % int(tiffidx+1)
    	    if save_tiffs is True:
                tf.imsave(os.path.join(write_dir, rangetiff_fn), ranged)

    
        # ----------------------------------------------------------------------   
        # ADJUST METADATA, if needed:  
        # ----------------------------------------------------------------------   
        frame_idxs = [int(f) for f in frame_idxs]
        
        # 1.  Update REFMETA struct:
        with open(os.path.join(acquisition_dir, run, '%s.json' % run_info_basename), 'r') as fr:
    	    runmeta = json.load(fr)
        runmeta['base_filename'] = prefix
        runmeta['frame_idxs'] = frame_idxs_final
    
        if correct_flyback:    
            print "Changing REF info:" 
            print "Orig N slices:", nslices_orig
            print "New N slices with correction:", nslices_crop #len(range(1, nslices_crop+1))  
            runmeta['slices'] = range(1, nslices_crop+1)
            runmeta['ntiffs'] = len(tiffs) 
        # else:
        #     if save_tiffs is True:
        #         if not visible and not crop_fov and not uint16:
        #             print "Copying RAW tiff to DATA dir. No changes."
        #             shutil.copy(os.path.join(source_dir, tiffs[tiffidx]), os.path.join(write_dir, newtiff_fn))
        
        # Save updated JSON:
        with open(os.path.join(acquisition_dir, run, '%s.json' % run_info_basename), 'w') as fw:
    	    #json.dump(refinfo, fw)
            dump(runmeta, fw, indent=4)
    
        # Also save updated MAT:
        #refinfo_mat = "%s.mat" % refinfo_basename
        #scipy.io.savemat(os.path.join(acquisition_dir, refinfo_mat), mdict=refinfo)
    

        
    # 2.  Update SIMETA info:
    if correct_flyback:
        raw_simeta_fn = [j for j in os.listdir(raw_tiff_dir) if j.endswith('json')][0]
        with open(os.path.join(raw_tiff_dir, raw_simeta_fn), 'r') as fj:
            raw_simeta = json.load(fj)

        adj_simeta = dict()
        print "Adjusting SIMETA data for downstream correction processes..."
        filenames = sorted([k for k in raw_simeta.keys() if 'File' in k], key=natural_keys)

        for fi in filenames:
            adj_simeta[fi] = dict()
            frame_idxs = runmeta['frame_idxs']
            nslices_orig = raw_simeta[fi]['SI']['hStackManager']['numSlices']
            ndiscard_orig = raw_simeta[fi]['SI']['hFastZ']['numDiscardFlybackFrames']

            nslices_selected = len(runmeta['slices'])
            print "Final N slices selected:", nslices_selected
            ndiscarded_extra = nslices_orig - nslices_selected

            # Rewrite relevant SI fields:
            raw_simeta[fi]['SI']['hStackManager']['nSlices'] = nslices_selected
            if raw_simeta[fi]['SI']['hStackManager']['zs'] is not None:
                raw_simeta[fi]['SI']['hStackManager']['zs'] = raw_simeta[fi]['SI']['hStackManager']['zs'][ndiscarded_extra:]
            raw_simeta[fi]['SI']['hFastZ']['numDiscardFlybackFrames'] = 0
            raw_simeta[fi]['SI']['hFastZ']['numFramesPerVolume'] = nslices_selected
            raw_simeta[fi]['SI']['hFastZ']['discardFlybackFames'] = 0 # flag this so Acquisition2P's parseScanImageTiff tkaes correct n slices

            if len(frame_idxs) > 0:
                raw_simeta[fi]['imgdescr'] = [raw_simeta[fi]['imgdescr'][int(i)] for i in frame_idxs]

            adj_simeta[fi]['SI'] = raw_simeta[fi]['SI']
            adj_simeta[fi]['frameNumbers'] = [f['frameNumbers'] for f in raw_simeta[fi]['imgdescr']]
            adj_simeta[fi]['frameTimestamps_sec'] = [f['frameTimestamps_sec'] for f in raw_simeta[fi]['imgdescr']]
            adj_simeta[fi]['frameNumberAcquisition'] = [f['frameNumberAcquisition'] for f in raw_simeta[fi]['imgdescr']]
            adj_simeta[fi]['epoch'] = raw_simeta[fi]['imgdescr'][-1]['epoch']

    # else:
    #     print "Copying SI meta data over..."
    #     adj_simeta = raw_simeta

        with open(os.path.join(write_dir, raw_simeta_fn), 'w') as fw:
            dump(adj_simeta, fw, indent=4) #, sort_keys=True)

    # ========================================================================================
    # UPDATE PREPROCESSING SOURCE/DEST DIRS, if needed:
    # ========================================================================================
    write_hash = None
    if correct_flyback:
        # 3.  Update write-dir hash ids:
        excluded_files = [str(f) for f in os.listdir(write_dir) if not f.endswith('tif')]
        print "Checking flyback hash, excluded_files:", excluded_files
        write_hash = dirhash(write_dir, 'sha1', excluded_files=excluded_files)[0:6]
        newwrite_dir = write_dir + '_%s' % write_hash
        os.rename(write_dir, newwrite_dir)
        
        # Make sure newly created TIFFs are READ-ONLY:
        if correct_flyback:
            for f in os.listdir(newwrite_dir):
                os.chmod(os.path.join(newwrite_dir, f), S_IREAD|S_IRGRP|S_IROTH)  
            
        PID['PARAMS']['preprocessing']['destdir'] = newwrite_dir
        print "OUTPUT DIR: %s" % PID['PARAMS']['preprocessing']['destdir']

    with open(os.path.join(tmp_pid_dir, tmp_pid_fn), 'w') as f:
        print tmp_pid_fn
        json.dump(PID, f, indent=4, sort_keys=True)            
    # ========================================================================================
    
    
    # if write_hash is None:
    #     write_hash = source_hash
        
    return write_hash, pid_hash
    
def main(options):
    
    flyback_hash, pid_hash = do_flyback_correction(options)
    
    print "PID %s: Finished flyback-correction step: output dir hash %s" % (pid_hash, flyback_hash)
    
if __name__ == '__main__':
    main(sys.argv[1:]) 
