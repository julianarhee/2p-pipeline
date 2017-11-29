#!/usr/bin/env python2
'''
Run this script to get ScanImage metadata from RAW acquisition files (.tif) from SI.
Requires ScanImageTiffReader (download: http://scanimage.vidriotechnologies.com/display/SIH/ScanImage+Home)
Assumes Linux, unless otherwise specified (see: options.path_to_si_reader)

Run python get_scanimage_data.py -h for all input options.
'''

import os
import sys
import optparse
import json
import re
import scipy.io
import numpy as np
from checksumdir import dirhash
from stat import S_IREAD, S_IRGRP, S_IROTH
from caiman.utils import utils
from os.path import expanduser
home = expanduser("~")

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError


def format_si_value(value):
    num_format = re.compile(r'\-?[0-9]+\.?[0-9]*|\.?[0-9]')
    sci_format = re.compile('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?')
   
    try:
        return eval(value)
    except:
        if value == 'true':
            return True
        elif value == 'false':
            return False
#        elif value == 'NaN':
#            return np.nan
#        elif value == 'inf' or value == 'Inf':
#            return np.inf
        elif len(re.findall(num_format, value))>0:  # has numbers
            if '[' in value:
                ends = [value.index('[')+1,  value.index(']')]
                tmpvalue = value[ends[0]:ends[1]]
                if ';' in value:
                    rows = tmpvalue.split(';'); 
                    value = [[float(i) for i in re.findall(num_format, row)] for row in rows]
                else:
                    value = [float(i) for i in re.findall(num_format, tmpvalue)]
        else:
            return value
 
#    if "'" in item:
#        value = str(value) 
#    elif len(re.findall(num_format, value))>0:  # has numbers 
#        if value.isdigit():
#            value = int(value)
#        elif '[' in value:
#            ends = [value.index('[')+1,  value.index(']')]
#            tmpvalue = value[ends[0]:ends[1]]
#            if ';' in value:
#                rows = tmpvalue.split(';'); 
#                value = [[float(i) for i in re.findall(num_format, row)] for row in rows]
#            else:
#                value = [float(i) for i in re.findall(num_format, tmpvalue)]                    
#


def get_meta(options):
 
    parser = optparse.OptionParser()

    # PATH opts:
    parser.add_option('-P', '--sipath', action='store', dest='path_to_si_base', default='~/Downloads/ScanImageTiffReader-1.1-Linux', help='path to dir containing ScanImageTiffReader.py')
    parser.add_option('-R', '--root', action='store', dest='rootdir', default='/nas/volume1/2photon/data', help='source dir (root project dir containing all expts) [default: /nas/volume1/2photon/data]')
    parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')
    parser.add_option('-S', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID')
    parser.add_option('-A', '--acq', action='store', dest='acquisition', default='FOV1', help="acquisition folder (ex: 'FOV1_zoom3x') [default: FOV1]")
    parser.add_option('-r', '--run', action='store', dest='run', default='', help="name of run dir containing tiffs to be processed (ex: gratings_phasemod_run1)")

    parser.add_option('--rerun', action='store_false', dest='new_acquisition', default=True, help="set if re-running to get metadata for previously-processed acquisition")


    (options, args) = parser.parse_args(options) 

    new_acquisition = options.new_acquisition
    if new_acquisition is False:
        print "This is a RE-RUN."

    path_to_si_base = options.path_to_si_base

    rootdir = options.rootdir
    animalid = options.animalid
    session = options.session
    acquisition = options.acquisition
    run = options.run

    # -------------------------------------------------------------
    # Set basename for files created containing meta/reference info:
    # -------------------------------------------------------------
    raw_simeta_basename = 'SI_%s' % run #functional_dir
    reference_info_basename = 'reference_%s' % run #functional_dir
    # -------------------------------------------------------------
    # -------------------------------------------------------------

    if '~' in path_to_si_base:
        path_to_si_base = path_to_si_base.replace('~', home)
    path_to_si_reader = os.path.join(path_to_si_base, 'share/python')
    print path_to_si_reader
    sys.path.append(path_to_si_reader)
    from ScanImageTiffReader import ScanImageTiffReader

    acquisition_dir = os.path.join(rootdir, animalid, session, acquisition)

    # Get RAW tiffs from acquisition:
    rawtiff_dir = os.path.join(acquisition_dir, run, 'raw')
    rawtiffs = sorted([t for t in os.listdir(rawtiff_dir) if t.endswith('.tif')], key=natural_keys)
    nontiffs = sorted([t for t in os.listdir(rawtiff_dir) if t not in rawtiffs], key=natural_keys)
    print rawtiffs
    
    # Generate SHA1-hash for tiffs in 'raw' dir:
    rawtiff_hash = dirhash(rawtiff_dir, 'sha1', excluded_files=nontiffs)
    hashid = rawtiff_hash[0:6]
    rawdir = 'raw_%s' % hashid
    
    # Rename RAW dir to include len 8 hash:
    os.rename(rawtiff_dir, os.path.join(acquisition_dir, run, rawdir))

    # Extract and parse SI metadata:
    scanimage_metadata = dict()
    scanimage_metadata['filenames'] = []
    scanimage_metadata['session'] = session
    scanimage_metadata['acquisition'] = acquisition
    scanimage_metadata['run'] = run

    for fidx,rawtiff in enumerate(sorted(rawtiffs, key=natural_keys)):
	
        curr_file = 'File{:03d}'.format(fidx+1)
        print "Processing:", curr_file
        
        currtiffpath = os.path.join(acquisition_dir, run, rawdir, rawtiff)
            
        # Make sure TIFF is READ ONLY:
        os.chmod(currtiffpath, S_IREAD|S_IRGRP|S_IROTH)  
        
        scanimage_metadata[curr_file] = {'SI': None}

        metadata = ScanImageTiffReader(currtiffpath).metadata()
        meta = metadata.splitlines()
        del metadata

        # descs = ScanImageTiffReader(os.path.join(acquisition_dir, rawtiff)).descriptions()
        # vol=ScanImageTiffReader("my.tif").data();

        # Get ScanImage metadata:
        SI = [l for l in meta if 'SI.' in l]
        del meta

        # Iterate through list of SI. strings and turn into dict:
        SI_struct = {}
        for item in SI:
            t = SI_struct
            fieldname = item.split(' = ')[0] #print fieldname
            fvalue = item.split(' = ')[1]
            value = format_si_value(fvalue)

            for ix,part in enumerate(fieldname.split('.')):
                nsubfields = len(fieldname.split('.'))
                if ix==nsubfields-1:
                    t.setdefault(part, value)
                else:
                    t = t.setdefault(part, {})

        # Get img descriptions for each frame:
        imgdescr = utils.get_image_description_SI(currtiffpath)
 
        
        # print SI_struct.keys()
        scanimage_metadata['filenames'].append(rawtiff)
        scanimage_metadata[curr_file]['SI'] = SI_struct['SI']
        scanimage_metadata[curr_file]['imgdescr'] = imgdescr

        # Save dict:
        raw_simeta_json = '%s.json' % raw_simeta_basename
        with open(os.path.join(acquisition_dir, run, 'raw', raw_simeta_json), 'w') as fp:
            json.dump(scanimage_metadata, fp, sort_keys=True, indent=4, default=set_default)
            #json.dumps(scanimage_metadata, fp, default=set_default, sort_keys=True, indent=4)



        # Also save as .mat for now:
        #raw_simeta_mat = '%s.mat' % raw_simeta_basename
        #scipy.io.savemat(os.path.join(acquisition_dir, raw_simeta_mat), mdict=scanimage_metadata, long_field_names=True)
        #print "Saved .MAT to: ", os.path.join(acquisition_dir, raw_simeta_mat)


        # Create REFERENCE info file or overwrite relevant fields, if exists: 
        refinfo_json = '%s.json' % reference_info_basename
        if new_acquisition is True:
            refinfo = dict()
        elif os.path.exists(os.path.join(acquisition_dir, run, refinfo_json)):
            with open(os.path.join(acquisition_dir, run, refinfo_json), 'r') as fp:
		        refinfo = json.load(fp)
        else:
            refinfo = dict() 

        refinfo['rootdir'] = rootdir 
        refinfo['animal_id'] = animalid #experiment
        refinfo['session'] = session
        refinfo['acquisition'] = acquisition
        refinfo['run'] = run #functional_dir
        refinfo['rawtiff_dir'] = rawdir
        specified_nslices =  int(scanimage_metadata['File001']['SI']['hStackManager']['numSlices'])
        refinfo['slices'] = range(1, specified_nslices+1) 
        refinfo['ntiffs'] = len(rawtiffs)
        if isinstance(scanimage_metadata['File001']['SI']['hChannels']['channelSave'], int):
            refinfo['nchannels'] =  scanimage_metadata['File001']['SI']['hChannels']['channelSave']
        else:
            refinfo['nchannels'] = len(scanimage_metadata['File001']['SI']['hChannels']['channelSave']) # if i.isdigit()])
        refinfo['nvolumes'] = int(scanimage_metadata['File001']['SI']['hFastZ']['numVolumes'])
        refinfo['lines_per_frame'] = int(scanimage_metadata['File001']['SI']['hRoiManager']['linesPerFrame'])
        refinfo['pixels_per_line'] = int(scanimage_metadata['File001']['SI']['hRoiManager']['pixelsPerLine'])
        refinfo['raw_simeta_path'] = os.path.join(acquisition_dir, run, rawdir, raw_simeta_json) #raw_simeta_mat)

        if 'acquisition_base_dir' not in refinfo.keys():
            refinfo['acquisition_base_dir'] = acquisition_dir
        if 'mcparams_path' not in refinfo.keys():
            # TODO:  Don't make this specifically tied to MAT
            refinfo['mcparams_path'] = os.path.join(acquisition_dir, run, 'processed', 'mcparams.mat')
        if 'roi_dir' not in refinfo.keys():
            refinfo['roi_dir'] = os.path.join(acquisition_dir, 'ROIs')
        if 'trace_dir' not in refinfo.keys():
            refinfo['trace_dir'] = os.path.join(acquisition_dir, 'Traces')

        with open(os.path.join(acquisition_dir, run, refinfo_json), 'w') as fp:
            json.dump(refinfo, fp, indent=4)
        
        #refinfo_mat = '%s.mat' % reference_info_basename
        #scipy.io.savemat(os.path.join(acquisition_dir, refinfo_mat), mdict=refinfo)
    
    # Make sure SI META data is now read-only:
    if new_acquisition is True:
        os.chmod(os.path.join(acquisition_dir, run, 'raw_%s' % hashid, raw_simeta_json), S_IREAD|S_IRGRP|S_IROTH)

    return hashid

def main(options):
    
    hashid = get_meta(options)
    
    print "Extracted meta data. Raw tiff hash: raw_%s" % hashid
    

if __name__ == '__main__':
    main(sys.argv[1:]) 
