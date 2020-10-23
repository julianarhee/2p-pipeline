import matplotlib
matplotlib.use('Agg')
import os
import glob
import copy
import sys
import h5py
import json
import re
import datetime
import optparse
import pprint
import traceback
import time
import skimage
import shutil
from scipy.ndimage import filters
import matplotlib.pyplot as plt
import matplotlib
import scipy.signal
import pylab as pl
import numpy as np

def preprocess_trace(F,ops):
    #lifted from suite2p dcnv module
    sig = ops['sig_baseline']
    win = int(ops['win_baseline']*ops['fs'])    
    if ops['baseline']=='maximin':
        Flow = filters.gaussian_filter(F,    [0., sig])
        Flow = filters.minimum_filter1d(Flow,    win)
        Flow = filters.maximum_filter1d(Flow,    win)
    elif ops['baseline']=='constant':
        Flow = filters.gaussian_filter(F,    [0., sig])
        Flow = np.amin(Flow)
    elif ops['baseline']=='constant_prctile':
        Flow = np.percentile(F, ops['prctile_baseline'], axis=1)
        Flow = np.expand_dims(Flow, axis = 1)
    else:
        Flow = 0.

    F = F - Flow

    return F

def get_df_F(F,ops):
    #lifted from suite2p dcnv module
    sig = ops['sig_baseline']
    win = int(ops['win_baseline']*ops['fs'])    
    if ops['baseline']=='maximin':
        Flow = filters.gaussian_filter(F,    [0., sig])
        Flow = filters.minimum_filter1d(Flow,    win)
        Flow = filters.maximum_filter1d(Flow,    win)
    elif ops['baseline']=='constant':
        Flow = filters.gaussian_filter(F,    [0., sig])
        Flow = np.amin(Flow)
    elif ops['baseline']=='constant_prctile':
        Flow = np.percentile(F, ops['prctile_baseline'], axis=1)
        Flow = np.expand_dims(Flow, axis = 1)
    else:
        Flow = 0.

    F = (F - Flow)/Flow

    return F

def get_comma_separated_args(option, opt, value, parser):
  setattr(parser.values, option.dest, value.split(','))


class struct: pass


def parse_trace(opts):
    traceid = '%s_s2p'%(opts.traceid)
    run_list = opts.run_list
    nruns = len(run_list)

    #% Set up paths:    
    acquisition_dir = os.path.join(opts.rootdir, opts.animalid, opts.session, opts.acquisition)


    s2p_source_dir = os.path.join(acquisition_dir, opts.run,'processed', opts.analysis, 'suite2p','plane0')


    #s2p files
    s2p_raw_trace_fn = os.path.join(s2p_source_dir,'F.npy')
    s2p_np_trace_fn = os.path.join(s2p_source_dir,'Fneu.npy')
    s2p_stat_fn = os.path.join(s2p_source_dir,'stat.npy')
    s2p_ops_fn = os.path.join(s2p_source_dir,'ops.npy')
    s2p_cell_fn = os.path.join(s2p_source_dir,'iscell.npy')
    s2p_spks_fn = os.path.join(s2p_source_dir,'spks.npy')

    print(s2p_ops_fn)

    #load them in
    s2p_stat = np.load(s2p_stat_fn)
    s2p_ops = np.load(s2p_ops_fn).item()
    s2p_raw_trace_data = np.load(s2p_raw_trace_fn)
    s2p_iscell = np.load(s2p_cell_fn)[:,0]
    s2p_np_trace_data = np.load(s2p_np_trace_fn)
    s2p_spks_data = np.load(s2p_spks_fn)

    #get offset
    reg_offset = np.sqrt(np.power(s2p_ops['xoff'],2)+np.power(s2p_ops['yoff'],2))
    reg_offset = scipy.signal.medfilt(reg_offset,11)

    #get motion
    kernel = np.array([1,1,1,1,1,-1,-1,-1,-1,-1])/float(5)
    offset_pad = np.pad(reg_offset, ((0),(kernel.size-1)), 'edge')
    reg_motion = np.abs(np.convolve(offset_pad,kernel,'valid'))

    #correct for neuropil
    corrected_traces = s2p_raw_trace_data - (s2p_ops['neucoeff']*s2p_np_trace_data)
    print('neucoeff: ',s2p_ops['neucoeff'])


    #add in neuropil baseline
    if 'add_neuropil_baseline' in s2p_ops:
        print(s2p_ops['add_neuropil_baseline'])
        if s2p_ops['add_neuropil_baseline']:
            np_offset = np.mean(s2p_np_trace_data,1)
            np_offset = np.expand_dims(np_offset,1)
            np_offset = np.tile(np_offset,(1,s2p_np_trace_data.shape[1]))
            corrected_traces = corrected_traces + np_offset


    # baseline operation
    s2p_cell_trace_data  = preprocess_trace(corrected_traces, s2p_ops)

    #get global baseline and add it back in
    offset = np.mean(corrected_traces,1)
    print(s2p_ops['add_global_baseline'])
    if s2p_ops['add_global_baseline']:
        offset = np.expand_dims(offset,1)
        offset = np.tile(offset,(1,corrected_traces.shape[1]))
        s2p_cell_trace_data  = s2p_cell_trace_data +offset

    #correcting for negative and very small values
    if np.min(s2p_cell_trace_data)<10:
        s2p_cell_trace_data = s2p_cell_trace_data + (abs(np.min(s2p_cell_trace_data))+10)

        
    #get fractional fluoresence using baseline methods in ops structure
    s2p_cell_df_f_data = get_df_F(s2p_cell_trace_data, s2p_ops)

    #bin spikes
    conv_win_size = int(round(s2p_ops['fs']/4))#500 ms binning
    print((1/conv_win_size))
    spks_pad = np.pad(s2p_spks_data, ((0,0),(conv_win_size-1, 0)), 'edge')


    binned_spks_data = np.array([np.convolve(spks_pad[i,:],np.ones((conv_win_size,))*(1/float(conv_win_size)),'valid') for i in range(spks_pad.shape[0])])

    del s2p_spks_data,spks_pad


     
        
    cell_idxs = np.where(s2p_iscell==1)[0]

    #get roi info
    roi_center_x = np.zeros((len(s2p_stat),))
    roi_center_y = np.zeros((len(s2p_stat),))
    roi_compact = np.zeros((len(s2p_stat),))
    roi_skew = np.zeros((len(s2p_stat),))
    roi_aspect_ratio = np.zeros((len(s2p_stat),))
    roi_footprint = np.zeros((len(s2p_stat),))
    roi_radius = np.zeros((len(s2p_stat),))
    roi_npix = np.zeros((len(s2p_stat),))

    for ridx in range(len(s2p_stat)):
        roi_center_x[ridx] = np.mean(s2p_stat[ridx]['xpix'])
        roi_center_y[ridx] = np.mean(s2p_stat[ridx]['ypix'])
        roi_compact[ridx] = s2p_stat[ridx]['compact']
        roi_skew[ridx] = s2p_stat[ridx]['skew']
        roi_aspect_ratio[ridx] = s2p_stat[ridx]['aspect_ratio']
        roi_radius = s2p_stat[ridx]['radius']
        roi_footprint[ridx] = s2p_stat[ridx]['footprint']
        roi_npix[ridx] = s2p_stat[ridx]['npix']


    last_idx = 0#start off at the beginning

    #go through runs
    for indie_run in run_list:
    #indie_run = run_list[0]
        print(indie_run)


        #get run info
        run_dir = os.path.join(acquisition_dir, indie_run)
        traceid_dir = os.path.join(run_dir,'traces',traceid)
        if not os.path.isdir(traceid_dir):
            os.makedirs(traceid_dir)


        with open(os.path.join(run_dir, '%s.json' % indie_run), 'r') as fr:
            scan_info = json.load(fr)
        all_frames_tsecs = np.array(scan_info['frame_tstamps_sec'])
        nslices_full = len(all_frames_tsecs) / scan_info['nvolumes']
        nslices = len(scan_info['slices'])
        if scan_info['nchannels']==2:
            all_frames_tsecs = np.array(all_frames_tsecs[0::2])

        #    if nslices_full > nslices:
        #        # There are discard frames per volume to discount
        #        subset_frame_tsecs = []
        #        for slicenum in range(nslices):
        #            subset_frame_tsecs.extend(frame_tsecs[slicenum::nslices_full])
        #        frame_tsecs = np.array(sorted(subset_frame_tsecs))
        print("N tsecs:", len(all_frames_tsecs))
        framerate = scan_info['frame_rate']
        volumerate = scan_info['volume_rate']
        nvolumes = scan_info['nvolumes']
        nfiles = scan_info['ntiffs']
        frames_tsec = scan_info['frame_tstamps_sec']
        nslices = int(len(scan_info['slices']))
        nslices_full = int(round(scan_info['frame_rate']/scan_info['volume_rate']))

        #get some dimension info
        d1 = s2p_ops['Ly']
        d2 = s2p_ops['Lx']
        T = len(frames_tsec)#assume constant tiff size,for now
        d = d1*d2
        dims = (d1, d2, T/nslices)


        curr_slice = 'Slice01'#hard-code for now
        sl = 0

        for tiff_count in range(0,nfiles):
        #    tiff_count = 0
            curr_file = 'File%03d'%(tiff_count+1)
            print(curr_file)


            #get s2p traces, for ROIs classified as cells ONLY
            idx0 = last_idx
            idx1 = idx0+T
            print(idx0,idx1)
            last_idx = idx1

            s2p_cell_traces = np.transpose(s2p_cell_trace_data[:,idx0:idx1])
            s2p_np_traces = np.transpose(s2p_np_trace_data[:,idx0:idx1])
            s2p_raw = np.transpose(s2p_raw_trace_data[:,idx0:idx1])
            
            s2p_cell_df_f_trace = np.transpose(s2p_cell_df_f_data[:,idx0:idx1])
            binned_spks_trace =  np.transpose(binned_spks_data[:,idx0:idx1])

            motion_trace = reg_motion[idx0:idx1]
            offset_trace = reg_offset[idx0:idx1]



            # Create outfile:
            if not os.path.isdir(os.path.join(traceid_dir, 'files')):
                os.makedirs(os.path.join(traceid_dir, 'files'))

            filetraces_fn = '%s_rawtraces_s2p.hdf5' % (curr_file)
            filetraces_filepath = os.path.join(traceid_dir, 'files', filetraces_fn)
            file_grp = h5py.File(filetraces_filepath, 'w')

            file_grp.attrs['source_file'] = os.path.join(s2p_source_dir,'F.npy')
            file_grp.attrs['file_id'] = tiff_count+1
            file_grp.attrs['signal_channel'] = s2p_ops['functional_chan']
            file_grp.attrs['dims'] = (d1, d2, nslices, T/nslices)
            file_grp.attrs['mask_sourcefile'] = os.path.join(s2p_source_dir,'stat.npy')
            file_grp.attrs['s2p_cell_rois'] = cell_idxs
            file_grp.attrs['roi_center_y'] = roi_center_y
            file_grp.attrs['roi_center_x'] = roi_center_x
            file_grp.attrs['roi_compact'] = roi_compact
            file_grp.attrs['roi_skew'] = roi_skew
            file_grp.attrs['roi_aspect_ratio'] = roi_aspect_ratio
            file_grp.attrs['roi_radius'] = roi_radius
            file_grp.attrs['roi_footprint'] = roi_footprint
            file_grp.attrs['roi_npix'] = roi_npix

            file_grp.attrs['offset'] = offset_trace
            file_grp.attrs['max_offset'] = np.max(reg_offset)
            file_grp.attrs['min_offset'] = np.min(reg_offset)

            file_grp.attrs['motion'] = motion_trace
            file_grp.attrs['max_motion'] = np.max(reg_motion)
            file_grp.attrs['min_motion'] = np.min(reg_motion)




            # Get frame tstamps:
            curr_tstamps = np.array(frames_tsec[sl::nslices_full])
            print("*TSTAMPS: %s" % str(curr_tstamps.shape))
            tstamps_indices = np.array([frames_tsec.index(ts) for ts in curr_tstamps])

            fset = file_grp.create_dataset('/'.join([curr_slice, 'frames_tsec']), curr_tstamps.shape, curr_tstamps.dtype)
            fset[...] = curr_tstamps

            tset = file_grp.create_dataset('/'.join([curr_slice, 'frames_indices']), tstamps_indices.shape, tstamps_indices.dtype)
            tset[...] = tstamps_indices 

            # Save RAW trace:
            tset = file_grp.create_dataset('/'.join([curr_slice, 'traces', 'pixel_value', 'raw']), s2p_raw.shape, s2p_raw.dtype)
            tset[...] = s2p_raw
            curr_nframes, curr_nrois = s2p_raw.shape
            print("... saved tracemat: %s" % str(s2p_raw.shape))
            tset.attrs['nframes'] = curr_nframes #tracemat.shape[0]
            tset.attrs['nrois'] = curr_nrois #tracemat.shape[1]
            tset.attrs['dims'] = dims

            np_traces = file_grp.create_dataset('/'.join([curr_slice, 'traces', 'pixel_value', 'neuropil']), s2p_np_traces.shape, s2p_np_traces.dtype)
            np_traces[...] = s2p_np_traces
            print("... saved np tracemat: %s" % str(s2p_np_traces.shape))

            np_corrected = file_grp.create_dataset('/'.join([curr_slice, 'traces', 'pixel_value', 'cell']), s2p_cell_traces.shape, s2p_cell_traces.dtype)
            np_corrected.attrs['correction_factor'] = s2p_ops['neucoeff']
            np_corrected[...] = s2p_cell_traces
            
            dffset = file_grp.create_dataset('/'.join([curr_slice, 'traces', 'global_df_f' ,'cell']), s2p_cell_df_f_trace.shape, s2p_cell_df_f_trace.dtype)
            dffset[...] = s2p_cell_df_f_trace
            
            spks_set = file_grp.create_dataset('/'.join([curr_slice, 'traces', 'spks', 'cell']), binned_spks_trace.shape, binned_spks_trace.dtype)
            spks_set[...] = binned_spks_trace

            offset_data = file_grp.create_dataset('/'.join([curr_slice, 'roi_global_baseline']), offset.shape, offset.dtype)
            offset_data[...] = offset



            if file_grp is not None:
                file_grp.close()


    print(last_idx)
    print(s2p_cell_trace_data.shape)

    if not(last_idx == s2p_cell_trace_data.shape[1]):
        raise ValueError

        

def extract_options(options):
    parser = optparse.OptionParser()


    # PATH opts:
    parser.add_option('-D', '--root', action='store', dest='rootdir', default='/n/coxfs01/2p-data', help='source dir (root project dir containing all expts) [default: /n/coxfs01/2p-data]')
    parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')
    parser.add_option('-S', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID') 
    parser.add_option('-A', '--acq', action='store', dest='acquisition', default='', help="acquisition folder (ex: 'FOV1_zoom3x')")
    parser.add_option('-R', '--run', action='store', dest='run', default='', help='name of s2p run to process') 
    parser.add_option('-Y', '--analysis', action='store', dest='analysis', default='', help='Analysis to process. [ex: suite2p_analysis001]')
    parser.add_option('-T', '--traceid', action='store', dest='traceid', default='', help="(ex: traces001_s2p)")
    parser.add_option('-r', '--run_list', action='callback', dest='run_list', default='',type='string',callback=get_comma_separated_args, help='comma-separated names of run dirs containing tiffs to be processed (ex: run1, run2, run3)')




    (options, args) = parser.parse_args() 

    return options




#-----------------------------------------------------
#           MAIN SET OF ACTIONS
#-----------------------------------------------------

def main(options): 
    
    options = extract_options(options)

    print('----- Parsing Suite2p Trace-----')
    parse_trace(options)


    
#%%

if __name__ == '__main__':
    main(sys.argv[1:])
