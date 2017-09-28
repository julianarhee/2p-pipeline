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
from os.path import expanduser
home = expanduser("~")

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]


def main(options):
 
    parser = optparse.OptionParser()

    # PATH opts:
    parser.add_option('-P', '--sipath', action='store', dest='path_to_si_reader', default='~/Downloads/ScanImageTiffReader-1.1-Linux/share/python', help='path to dir containing ScanImageTiffReader.py')

    parser.add_option('-S', '--source', action='store', dest='source', default='/nas/volume1/2photon/projects', help='source dir (root project dir containing all expts) [default: /nas/volume1/2photon/projects]')
    parser.add_option('-E', '--experiment', action='store', dest='experiment', default='', help='experiment type (parent of session dir)')
 
    parser.add_option('-s', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID')
    parser.add_option('-A', '--acq', action='store', dest='acquisition', default='', help="acquisition folder (ex: 'FOV1_zoom3x')")
    parser.add_option('-f', '--functional', action='store', dest='functional_dir', default='functional', help="folder containing functional TIFFs. [default: 'functional']")

    (options, args) = parser.parse_args(options) 

    path_to_si_reader = options.path_to_si_reader

    source = options.source
    experiment = options.experiment
    session = options.session
    acquisition = options.acquisition
    functional_dir = options.functional_dir

    # -------------------------------------------------------------
    # Set basename for files created containing meta/reference info:
    # -------------------------------------------------------------
    raw_simeta_basename = 'SI_raw_%s' % functional_dir
    reference_info_basename = 'reference_%s' % functional_dir
    # -------------------------------------------------------------
    # -------------------------------------------------------------

    if '~' in path_to_si_reader:
	path_to_si_reader = path_to_si_reader.replace('~', home)
    print path_to_si_reader
    sys.path.append(path_to_si_reader)
    from ScanImageTiffReader import ScanImageTiffReader

    acquisition_dir = os.path.join(source, experiment, session, acquisition)

    rawtiffs = os.listdir(acquisition_dir)
    rawtiffs = [t for t in rawtiffs if t.endswith('.tif')]
    print rawtiffs

    scanimage_metadata = dict()
    scanimage_metadata['filenames'] = []
    scanimage_metadata['session'] = session
    scanimage_metadata['acquisition'] = acquisition
    scanimage_metadata['experiment'] = experiment 


    for fidx,rawtiff in enumerate(sorted(rawtiffs, key=natural_keys)):
	curr_file = 'File{:03d}'.format(fidx+1)
        print "Processing:", curr_file
	scanimage_metadata[curr_file] = {'SI': None}

	metadata = ScanImageTiffReader(os.path.join(acquisition_dir, rawtiff)).metadata()
	meta = metadata.splitlines()

	# descs = ScanImageTiffReader(os.path.join(acquisition_dir, rawtiff)).descriptions()
	# vol=ScanImageTiffReader("my.tif").data();


	# Get ScanImage metadata:
	SI = [l for l in meta if 'SI.' in l]

	# Iterate through list of SI. strings and turn into dict:
	SI_struct = {}
	for item in SI:
	    t = SI_struct
	    fieldname = item.split(' = ')[0]
	    value = item.split(' = ')[1]
	    for ix,part in enumerate(fieldname.split('.')):
		nsubfields = len(fieldname.split('.'))
		if ix==nsubfields-1:
		    t.setdefault(part, value)
		else:
		    t = t.setdefault(part, {})
	# print SI_struct.keys()
	scanimage_metadata['filenames'].append(rawtiff)
	scanimage_metadata[curr_file]['SI'] = SI_struct['SI']

    # Save dict:
    raw_simeta_json = '%s.json' % raw_simeta_basename
    with open(os.path.join(acquisition_dir, raw_simeta_json), 'w') as fp:
	json.dump(scanimage_metadata, fp, sort_keys=True, indent=4)


    # Also save as .mat for now:
    raw_simeta_mat = '%s.mat' % raw_simeta_basename
    scipy.io.savemat(os.path.join(acquisition_dir, raw_simeta_mat), mdict=scanimage_metadata, long_field_names=True)
    print "Saved .MAT to: ", os.path.join(acquisition_dir, raw_simeta_mat)


    # Create REFERENCE info file:
    refinfo = dict()
    refinfo['source'] = source
    refinfo['experiment'] = experiment
    refinfo['session'] = session
    refinfo['acquisition'] = acquisition
    refinfo['functional'] = functional_dir
    specified_nslices =  int(scanimage_metadata['File001']['SI']['hStackManager']['numSlices'])
    refinfo['slices'] = range(1, specified_nslices+1) 
    refinfo['ntiffs'] = len(rawtiffs)
    refinfo['nchannels'] = len([i for i in scanimage_metadata[curr_file]['SI']['hChannels']['channelSave'] if i.isdigit()])

    refinfo_json = '%s.json' % reference_info_basename
    with open(os.path.join(acquisition_dir, refinfo_json), 'w') as fp:
        json.dump(refinfo, fp, indent=4)
    
    refinfo_mat = '%s.mat' % reference_info_basename
    scipy.io.savemat(os.path.join(acquisition_dir, refinfo_mat), mdict=refinfo)


if __name__ == '__main__':
    main(sys.argv[1:]) 
