
import matplotlib

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import optparse
import numpy as np
import time
import sys
import os
import glob
# option to import from github folder
sys.path.insert(0, '/n/coxfs01/cechavarria/repos/suite2p')
from suite2p import dcnv


def deconvolve_traces(opts):


    #% Set up paths:    
    acquisition_dir = os.path.join(opts.rootdir, opts.animalid, opts.session, opts.acquisition)

    s2p_source_dir = os.path.join(acquisition_dir, opts.run,'processed', opts.analysis, 'suite2p','plane0')

    #s2p files
    s2p_raw_trace_fn = os.path.join(s2p_source_dir,'F.npy')
    s2p_np_trace_fn = os.path.join(s2p_source_dir,'Fneu.npy')

    s2p_stat_fn = os.path.join(s2p_source_dir,'stat.npy')
    s2p_ops_fn = os.path.join(s2p_source_dir,'ops.npy')


    #load them in
    s2p_stat = np.load(s2p_stat_fn)
    s2p_ops = np.load(s2p_ops_fn).item()
    s2p_raw_trace_data = np.load(s2p_raw_trace_fn)
    s2p_np_trace_data = np.load(s2p_np_trace_fn)

    #doing spike deconv with correct tau param
    #from Dana et al (2018 paper):
    #In  agreement  with  the  cultured neuron results, jGCaMP7f has faster kinetics 
    #than the other jGCaMP7 sensors and is comparable to GCaMP6f. 
    #jGCaMP7s has slower decay time than all the other sensors (Fig. 5b)

    #therefore using same value for tau (decay constant) as suggested by Suite2p
    #for GCaMP6f data: 0.7
    s2p_ops['tau'] = 0.7

    #remove neuropil
    Fc0 = s2p_raw_trace_data - s2p_ops['neucoeff'] * s2p_np_trace_data
    #baseline operation
    Fc1 = dcnv.preprocess(Fc0, s2p_ops)

    print('Deconvoling trace')
    # get spikes
    spks = dcnv.oasis(Fc1, s2p_ops)

    print(spks.shape)

    s2p_spks_fn = os.path.join(s2p_source_dir,'spks.npy')
    print('Saving to: %s'%(s2p_spks_fn))
    np.save(s2p_spks_fn,spks)



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
    
    options = extract_options(options)
    deconvolve_traces(options)
    

    
#%%

if __name__ == '__main__':
    main(sys.argv[1:])
