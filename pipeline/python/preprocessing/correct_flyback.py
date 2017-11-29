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
from stat import S_IREAD, S_IRGRP, S_IROTH
from pipeline.python.set_pid_params import get_basic_pid

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def do_flyback_correction(options):

    parser = optparse.OptionParser()

    # PATH opts:
    parser.add_option('-R', '--root', action='store', dest='rootdir', default='/nas/volume1/2photon/data', help='source dir (root project dir containing all expts) [default: /nas/volume1/2photon/data]')
    parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID') 

    parser.add_option('-S', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID') 
    parser.add_option('-A', '--acq', action='store', dest='acquisition', default='', help="acquisition folder (ex: 'FOV1_zoom3x')")
    parser.add_option('-r', '--run', action='store', dest='run', default='', help="name of run dir containing tiffs to be processed (ex: gratings_phasemod_run1)")
    parser.add_option('-h', '--hash', action='store', dest='source_hash', default='', help="hash of source dir (8 char)")

    # TIFF saving opts:
    parser.add_option('--native', action='store_false', dest='uint16', default=True, help='Keep int16 tiffs as native [default: convert to uint16]')
    parser.add_option('--visible', action='store_true', dest='visible', default=False, help='Also create tiffs with adjusted display-range to make visible.')
    parser.add_option('-m', '--min', action='store', dest='displaymin', default=10, help='min display range value [default: 10]')
    parser.add_option('-M', '--max', action='store', dest='displaymax', default=2000, help='max display range value [default: 2000]')

    # SUBSTACK opts (for correcting flyback):
    parser.add_option('--correct-flyback', action='store_true', dest='correct_flyback', default=False, help='Create substacks of data-only by removing all artifact/discard frames [default: False]')

    parser.add_option('--flyback', action='store', dest='nflyback', default=8, help='Num flyback frames at top of stack [default: 8]')
    parser.add_option('--discard', action='store', dest='ndiscard', default=8, help='Num discard frames at end of stack [default: 8]')
    parser.add_option('--notiffs', action='store_false', dest='save_tiffs', default=True, help='Flag to run without saving TIFFs')
 
    # TIFF acquisition info:
    parser.add_option('-c', '--nchannels', action='store', dest='nchannels', default=2, help='Num interleaved channels in raw tiffs to be processed [default: 2]')
    parser.add_option('-v', '--nvolumes', action='store', dest='nvolumes', default=340, help='Num volumes acquired [default: 340]')
    parser.add_option('-z', '--nslices', action='store', dest='nslices', default=30, help='Num slices specified, no discard [default: 30]')

    # FOV-cropping opts (prob shouldn't go here):
    parser.add_option('--crop', action='store_true', dest='crop_fov', default=False, help='Crop FOV in x,y for smaller data size [default: False]')
    parser.add_option('-X', '--xstart', action='store', dest='x_startidx', default=0, help='Starting idx for x-dimension, i.e., columns [default: 0]')
    parser.add_option('-Y', '--ystart', action='store', dest='y_startidx', default=0, help='Starting idx for y-dimension, i.e., rows [default: 0]')
    parser.add_option('-W', '--width', action='store', dest='width', default='', help='Width of FOV, i.e., cols [default: '']')
    parser.add_option('-H', '--height', action='store', dest='height', default='', help='Height of FOV, i.e., rows [default: '']')


    (options, args) = parser.parse_args(options) 
    save_tiffs = options.save_tiffs
    if save_tiffs is True:
        print "Correcting flyback and saving TIFFs."
    else:
        print "Not saving TIFFs, just getting meta-data."

    crop_fov = options.crop_fov
    x_startidx = int(options.x_startidx)
    y_startidx = int(options.y_startidx)

    correct_flyback = options.correct_flyback
    uint16 = options.uint16
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
    source_hash = options.source_hash
    
    # ----------------------------------------------------------------------------------   
    # Set reference-struct basename. Should match that created in get_scanimage_data.py:
    # ---------------------------------------------------------------------------------- 
    refinfo_basename = 'reference_%s' % run #functional_dir
    pidinfo_basename = 'pid_info_%s' % run
    # ----------------------------------------------------------------------------------

    visible = options.visible
    if visible:
	    displaymin = options.displaymin
	    displaymax = options.displaymax

    # Identify RAW tiff dir from acquisition:
    acquisition_dir = os.path.join(rootdir, animalid, session, acquisition)
    raw_tiff_dir = [os.path.join(acquisition_dir, run, rd) for rd in os.listdir(os.path.join(acquisition_dir, run)) if 'raw' in rd and os.path.isdir(os.path.join(acquisition_dir, run, rd))][0] # SHOULD ONLY BE ONE SINGLE RAW TIFF FOLDER
    if source_hash not in raw_tiff_dir:
        print "WARNING*********************************"
        print "Source hash does not match RAW tiff dir:\n%s" % raw_tiff_dir
        print "****************************************"
    else:
        print "Checking for flyback correction for tiffs from source:\n%s" % raw_tiff_dir
        
    # Set and create PROCESSED write dir:
    save_basepath = os.path.join(acquisition_dir, run, 'processed') #, 'DATA', 'Raw')    
    if not os.path.exists(save_basepath):
        os.makedirs(save_basepath)

    # Load PID dict to access pids and hashes:
    if os.path.exists(os.path.join(save_basepath, '%s.json' % pidinfo_basename)):
        with open(os.path.join(save_basepath, '%s.json' % pidinfo_basename), 'r') as f:
            processdict = json.load(f)
    
    
    
    else:
        PID = get_default_pid(rootdir=rootdir, animalid=animalid, session=session, acquisition=acquisition,run=run, correct_flyback=correct_flyback, nflyback_frames=nflyback)
        
    # Load user-created PID tmp file, or create default if no tmp file exists (no bidir- or motion-correction):
    if os.path.exists(os.path.join(save_basepath, 'tmp_processparams.json')):
        with open(os.path.join(save_basepath, 'tmp_processparams.json'), 'r') as f:
            PID = json.load(f)
    else:
        PID = get_default_pid(rootdir=rootdir, animalid=animalid, session=session, acquisition=acquisition,run=run, correct_flyback=correct_flyback, nflyback_frames=nflyback)
    

    processdict[process_id] = PID
    with open(os.path.join(save_basepath, '%s.json' % pidinfo_basename), 'w') as f:
        json.dump(processdict, f, indent=4, sort_keys=True)
        
        
    # Set specific PID dir for current processing run:
    
    # Increment process_id so can't overwrite, even though source is always "raw"
    # Write flyback-corrected TIFFs to 'raw' subdir (may want to play with n flyback corrected)
    processed_dirs = [p for p in os.listdir(save_basepath) if os.path.isdir(os.path.join(save_basepath, p))] 
    increment_processed = int(len(processed_dirs) + 1)
    process_id = 'processed%03d' % increment_processed
    savepath = os.path.join(save_basepath, process_id, 'raw')
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    

    
    # Get TIFFs to process:
    tiffs = os.listdir(raw_tiff_dir)
    tiffs = sorted([t for t in tiffs if t.endswith('.tif')], key=natural_keys)
    print "Found %i TIFFs." % len(tiffs)
    
    for tiffidx,tiffname in enumerate(tiffs):
        # Adjust TIFF file name so that all included files increment correctly,
        # and naming format matches standard:	
        origname = tiffname.split('.')[0]
    	prefix = '_'.join(origname.split('_')[0:-1])
        prefix = prefix.replace('-', '_')
        newtiff_fn = '%s_File%03d.tif' % (prefix, int(tiffidx+1)) 
        print "Creating file in DATA dir:", newtiff_fn
       
        if correct_flyback:
            # Read in RAW tiff: 
            stack = tf.imread(os.path.join(raw_tiff_dir, tiffs[tiffidx]))
    	    if uint16:
    		    stack = img_as_uint(stack) 
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
                tf.imsave(os.path.join(savepath, newtiff_fn), final)
 
            # Again, get frame-idxs of kept-frames: 
            start_idxs_single = np.arange(nflyback, nslices_orig*nvolumes, nslices_orig)
            frame_idxs_final = np.empty((nslices_crop*nvolumes,))
            newstart = 0
            for x in range(len(start_idxs_single)):
                frame_idxs_final[newstart:newstart+(nslices_crop)] = frame_idxs[start_idxs_single[x]:start_idxs_single[x]+(nslices_crop)]
                newstart = newstart + (nslices_crop)
     
            print "Created frame-idx array. Final shape: ", frame_idxs_final.shape
                   		    
    	else:
    	    print "Not creating substacks from input tiffs."
    	    if uint16:
                print "Converting raw tiff to uint16."
                stack = tf.imread(os.path.join(raw_tiff_dir, tiffs[tiffidx]))
    	        final = img_as_uint(stack)
    
                dtype_fn = '%s_uint16.tif' % newtiff_fn.split('.')[0] 
    	        if save_tiffs is True: 
                    tf.imsave(os.path.join(savepath, dtype_fn), final)
     	    
            frame_idxs_final = [] # if not index correction needed, just leave blank 
    
           
    	if crop_fov:
            if not correct_flyback:  # stack not yet read in:
                final = tf.imread(os.path.join(raw_tiff_dir, tiffs[tiffidx]))
    
    	    if len(options.width)==0:
    		    width = int(input('No width specified. Starting idx is: %i.\nEnter image width: ' % x_startidx))
    	    else:
    		    width = int(options.width)
    	    if len(options.height)==0:
    		    height = int(input('No height specified. Starting idx is: %i.\nEnter image height: ' % y_startidx))
    	    else:
    		    height = int(options.height)
    
    	    x_endidx = x_startidx + width
    	    y_endidx = y_startidx + height
    	    
    	    final = final[:, y_startidx:y_endidx, x_startidx:x_endidx]
    	    print "Cropped FOV. New size: ", final.shape
                
            # TODO: add extra info to SI-meta or reference-struct, if want to keep this option...
            cropped_fn = '%s_cropped.tif' % newtiff_fn.split('.')[0] #'File%03d_visible.tif' % int(tiffidx+1)
    	    if save_tiffs is True:
                tf.imsave(os.path.join(savepath, cropped_fn), final)
    	    
    	if visible: 
    	    ranged = exposure.rescale_intensity(final, in_range=(displaymin, displaymax))
    	    rangetiff_fn = '%s_visible.tif' % newtiff_fn.split('.')[0] #'File%03d_visible.tif' % int(tiffidx+1)
    	    if save_tiffs is True:
                tf.imsave(os.path.join(savepath, rangetiff_fn), ranged)

    
        # ----------------------------------------------------------------------   
        # ADJUST METADATA, if needed:  
        # ----------------------------------------------------------------------   

        # Update REFMETA struct:
        refinfo_json = "%s.json" % refinfo_basename
        with open(os.path.join(acquisition_dir, run, refinfo_json), 'r') as fr:
    	    refinfo = json.load(fr)
        refinfo['base_filename'] = prefix
        refinfo['frame_idxs'] = frame_idxs_final
    
        if correct_flyback:    
            print "Changing REF info:" 
            print "Orig N slices:", nslices_orig
            print "New N slices with correction:", nslices_crop #len(range(1, nslices_crop+1))  
            refinfo['slices'] = range(1, nslices_crop+1)
            refinfo['ntiffs'] = len(tiffs) 
        else:
            if save_tiffs is True:
                if not visible and not crop_fov and not uint16:
                    print "Copying RAW tiff to DATA dir. No changes."
                    shutil.copy(os.path.join(raw_tiff_dir, tiffs[tiffidx]), os.path.join(savepath, newtiff_fn))
        
        # Save updated JSON:
        refinfo_json = "%s.json" % refinfo_basename
        with open(os.path.join(acquisition_dir, run, refinfo_json), 'w') as fw:
    	    #json.dump(refinfo, fw)
            dump(refinfo, fw, indent=4)
    
        # Also save updated MAT:
        #refinfo_mat = "%s.mat" % refinfo_basename
        #scipy.io.savemat(os.path.join(acquisition_dir, refinfo_mat), mdict=refinfo)
    
        # Make sure newly created TIFFs are READ-ONLY:
        os.chmod(os.path.join(savepath, newtiff_fn), S_IREAD|S_IRGRP|S_IROTH)  

        # Update SIMETA info:
        raw_simeta_fn = [j for j in os.listdir(raw_tiff_dir) if j.endswith('json')][0]
        with open(os.path.join(raw_tiff_dir, raw_simeta_fn), 'r') as fj:
            raw_simeta = json.load(fj)

        if correct_flyback:
            adj_simeta = dict()
            print "Adjusting SIMETA data for downstream correction processes..."
            filenames = sorted([k for k in raw_simeta.keys() if 'File' in k], key=natural_keys)

            for fi in filenames:
                adj_simeta[fi] = dict()
                frame_idxs = refinfo['frame_idxs']
                nslices_orig = raw_simeta[fi]['SI']['hStackManager']['numSlices']
                ndiscard_orig = raw_simeta[fi]['SI']['hFastZ']['numDiscardFlybackFrames']

                nslices_selected = refinfo['nslices']
                ndiscarded_extra = nslices_orig - nslices_selected

                # Rewrite relevant SI fields:
                raw_simeta[fi]['SI']['hStackManager']['nSlices'] = nslices_selected
                raw_simeta[fi]['SI']['hStackManager']['zs'] = raw_simeta['SI']['hStackManager']['zs'][ndiscarded_extra:]
                raw_simeta[fi]['SI']['hFastZ']['numDiscardFlybackFrames'] = 0
                raw_simeta[fi]['SI']['hFastZ']['numFramesPerVolume'] = nslices_selected
                raw_simeta[fi]['SI']['hFastZ']['discardFlybackFames'] = 0 # flag this so Acquisition2P's parseScanImageTiff tkaes correct n slices

                if len(frame_idxs) > 0:
                    raw_simeta[fi]['imgdescr'] = [raw_simeta[fi]['imgdescr'][i] for i in frame_idxs]

                adj_simeta[fi]['SI'] = raw_simeta[fi]['SI']
                adj_simeta[fi]['frameNumbers'] = [f['frameNumbers'] for f in raw_simeta[fi]['imgdescr']]
                adj_simeta[fi]['frameTimestamps_sec'] = [f['frameTimestamps_sec'] for f in raw_simeta[fi]['imgdescr']]
                adj_simeta[fi]['frameNumberAcquisition'] = [f['frameNumberAcquisition'] for f in raw_simeta[fi]['imgdescr']]
                adj_simeat[fi]['epoch'] = raw_simeta[fi]['imgdesc'][-1]['epoch']

        else:
            print "Copying SI meta data over..."
            adj_simeta = raw_simeta

        with open(os.path.join(save_basepath, process_id, raw_simeta_fn), 'w') as fw:
            dump(adj_simeta, fw, indent=4) #, sort_keys=True)
 

if __name__ == '__main__':
    main(sys.argv[1:]) 
