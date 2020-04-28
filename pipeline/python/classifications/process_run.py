#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 12:53:03 2018

@author: juliana
"""
import optparse
import os
import glob
import shutil
import copy
import sys
import pprint
pp = pprint.PrettyPrinter(indent=4)
import cPickle as pkl
import numpy as np
import pandas as pd

from pipeline.python.visualization import get_session_summary as ss
from pipeline.python.paradigm import utils as util
from pipeline.python.utils import natural_keys, label_figure
from pipeline.python.classifications import test_responsivity as resp #import calculate_roi_responsivity, group_roidata_stimresponse, find_barval_index
from pipeline.python.paradigm import plot_responses as psth

#animalid = optsE.animalid
#session = optsE.session
#acquisition = optsE.acquisition
#run_list = optsE.run_list
#traceid_list = optsE.traceid_list
#
def process_run_data(animalid, session, acquisition, run_list, traceid_list, 
                         rootdir='/n/coxfs01/2p-data',
                         stimtype='', data_type='corrected', 
                         response_metric='zscore', metric='zscore',
                         make_equal=True, create_new=False, nproc=2,
                         pval_selective=0.05, pval_visual=0.05,
                         visual_test_type='RManova1'):

    acquisition_dir = os.path.join(rootdir, animalid, session, acquisition)
    if stimtype == '':
        stimtype = run_list[0].split('_')[0:-1]
      
    print "STIM: %s" % stimtype
    is_gratings = any(r in stimtype for r in ['grating', 'rfs'])
 
    # Combine runs: 
    traceid_dir = ss.get_traceid_dir_from_lists(acquisition_dir, run_list, traceid_list, stimtype=stimtype, make_equal=make_equal, create_new=create_new)

    # Load combined dataset, and make trial nums equal for stats, etc.: 
    print "---> Loading dataset: %s" % traceid_dir

    data_fpath = glob.glob(os.path.join(traceid_dir, 'data_arrays', '*.npz'))[0]
    dataset = np.load(data_fpath)
    
    rinfo = dataset['run_info'] if isinstance(dataset['run_info'], dict) else dataset['run_info'][()]
    pp.pprint(rinfo['ntrials_by_cond'])
    if len(list(set([v for k,v in rinfo['ntrials_by_cond'].items()]))) > 1:
        dataset, data_fpath = util.get_equal_reps(dataset, data_fpath)


    traceid = os.path.split(traceid_dir)[-1]
    run = os.path.split(traceid_dir.split('/traces/')[0])[-1]
    print "Run: %s | traceid: %s" % (run, traceid)
    
    # Get stats for responsivity and selectivity:
    roistats = ss.get_roi_stats(rootdir, animalid, session, acquisition, 
                                   run, traceid, create_new=create_new, 
                                   nproc=nproc, metric=response_metric,
                                   pval_selective=pval_selective, 
                                   pval_visual=pval_visual,
                                   visual_test_type=visual_test_type)
    # Group data by ROIs:
    roidata, labels_df, sconfigs = ss.get_data_and_labels(dataset, data_type=data_type)
    df_by_rois = resp.group_roidata_stimresponse(roidata, labels_df)
    data, transforms_tested = ss.get_object_transforms(df_by_rois, roistats, sconfigs, metric=metric, is_gratings=is_gratings)
     
    object_dict = {'source': run,
                   'traceid': traceid,
                   'data_fpath': data_fpath,
                   'roistats': roistats,
                   'roidata': df_by_rois,
                   'sconfigs': sconfigs,
                   'transforms': data,
                   'transforms_tested': transforms_tested,
                   'metric': metric}
    
    processed_run_fpath = os.path.join(traceid_dir, 'processed_run.pkl')
    with open(processed_run_fpath, 'wb') as f:
        pkl.dump(object_dict, f, protocol=pkl.HIGHEST_PROTOCOL)
   
    return processed_run_fpath


def extract_options(options):

    def comma_sep_list(option, opt, value, parser):
        setattr(parser.values, option.dest, value.split(','))

    parser = optparse.OptionParser()

    parser.add_option('-D', '--root', action='store', dest='rootdir',
                          default='/nas/volume1/2photon/data',
                          help='data root dir (dir w/ all animalids) [default: /nas/volume1/2photon/data, /n/coxfs01/2pdata if --slurm]')
    parser.add_option('--slurm', action='store_true', dest='slurm', default=False, help="set if running as SLURM job on Odyssey")

    parser.add_option('-i', '--animalid', action='store', dest='animalid',
                          default='', help='Animal ID')

    # Set specific session/run for current animal:
    parser.add_option('-S', '--session', action='store', dest='session',
                          default='', help='session dir (format: YYYMMDD_ANIMALID')
    parser.add_option('-A', '--acq', action='store', dest='acquisition',
                          default='FOV1', help="acquisition folder (ex: 'FOV1_zoom3x') [default: FOV1]")

    # Run specific info:
    #parser.add_option('-g', '--gratings-traceid', dest='gratings_traceid_list', default=[], action='append', nargs=1, help="traceid for GRATINGS [default: []]")
    parser.add_option('--all', dest='combine_all', default=False, action='store_true', help='set flag to combine all runs (with prefix specified by stimtype -s) with matching traceids (specified by -t)')
    parser.add_option('-t', '--traceids', dest='traceid', default=None, action='store', help='common traceid (e.g., traces001) for combing all runs of a given stimtype (set --all)')
   
    parser.add_option('-T', '--traceid-list', dest='traceid_list', default=[], type='string', action='callback', callback=comma_sep_list, help="traceids for corresponding runs [default: []]")
    parser.add_option('-R', '--run-list', dest='run_list', default=[], type='string', action='callback', callback=comma_sep_list, help='list of run IDs [default: []')
    parser.add_option('-s', '--stim', dest='stimtype', default='', action='store', help='stimulus type (must be: gratings, blobs, or objects in run name)')

    parser.add_option('-d', '--dtype', dest='data_type', default='corrected', action='store', help='data_type (corrected, dff, etc.) for raw data (default: corrected)')
    parser.add_option('-m', '--metric', dest='metric', default='zscore', action='store', help='metric (e.g., zscore, meastim) to use for first-pass visualiztion of stim responses')
    
    parser.add_option('--equal', action='store_true', dest='make_equal', default=False, help="set if make ntrials per cond equal (need for roistats)")
    parser.add_option('--new', action='store_true', dest='create_new', default=False, help="set to re-extract data summaries")
    parser.add_option('-n', '--nproc', action='store', dest='nprocesses', default=4, help="N processes if running in par (default=4)")

    # PLOTTING PSTH opts:
    parser.add_option('--psth', action='store_true', dest='plot_psth', default=False, help='Set flag to plot PSTHs for all ROIs. Set plotting grid opts')
    parser.add_option('-f', action='store', dest='psth_dtype', default='corrected', help='Data type to plot for PSTHs.')

    parser.add_option('-r', '--rows', action='store', dest='psth_rows', default=None, help='PSTH: transform to plot on ROWS of grid')
    parser.add_option('-c', '--cols', action='store', dest='psth_cols', default=None, help='PSTH: transform to plot on COLS of grid')
    parser.add_option('-H', '--hues', action='store', dest='psth_hues', default=None, help='PSTH: transform to plot for HUES of each subplot')

    parser.add_option('--pvis', action='store', dest='pval_visual', default=0.5, help='P-value for visual responsivity test (SP anova/ RM anova) [default=0.05]')
    parser.add_option('--psel', action='store', dest='pval_selective', default=0.5, help='P-value for selectivity test (KW) [default=0.05]')
    parser.add_option('-M','--resp-metric', action='store', dest='response_metric', default='zscore', help='Response value to use for calculating stats [default: zscore (alt: meanstim)]')
    parser.add_option('-v','--visual-test', action='store', dest='visual_test_type', default='RManova1', help='Test to use for visual responsivity [default: RManova1 (alt: SPanova2)]')



    (options, args) = parser.parse_args(options)
    if options.slurm is True and '/n/coxfs01' not in options.rootdir:
        options.rootdir = '/n/coxfs01/2p-data'
    return options


#%%
    
options = ['-D', '/n/coxfs01/2p-data', '-i', 'CE077', '-S', '20180521', '-A', 'FOV2_zoom1x',
           '-R', 'blobs_run1,blobs_run2',
           '-t', 'traces002,traces002',
           '-s', 'blobs',
           '--new',
           '-n', 2
           ]

#metric = 'zscore'
#data_type = 'corrected'

def main(options):
    optsE = extract_options(options)

    if optsE.combine_all:
        # Get all runs of specified stimtype:
        acquisition_dir = os.path.join(optsE.rootdir, optsE.animalid, optsE.session, optsE.acquisition)
        found_runs = sorted(glob.glob(os.path.join(acquisition_dir, '%s_run*' % optsE.stimtype)), key=natural_keys)
        print "Found %i runs of stimtype: %s" % (len(found_runs), optsE.stimtype)
        for fi, frun in enumerate(found_runs):
            print fi, frun
            
        optsE.run_list = [os.path.split(found_run)[-1] for found_run in found_runs] 
        optsE.traceid_list = [optsE.traceid for _ in range(len(optsE.run_list))]
        # Make sure traceids exist:
        run_list = copy.copy(optsE.run_list)
        traceid_list = copy.copy(optsE.traceid_list)
        exclude_runs = []
        for ri, (r, t) in enumerate(zip(sorted(optsE.run_list, key=natural_keys), optsE.traceid_list)):
            found_dfile = glob.glob(os.path.join(acquisition_dir, r, 'traces', '%s*' % t, 'data_arrays', '*.npz'))
            if len(found_dfile) == 0:
                exclude_runs.append(ri)
        print("*** Excluding: %s" % str(exclude_runs))

        optsE.run_list = [r for ri, r in enumerate(sorted(run_list, key=natural_keys)) if ri not in exclude_runs]
        optsE.traceid_list = [t for ti, t in enumerate(traceid_list) if ti not in exclude_runs]
    else:

        if (len(optsE.run_list) > len(optsE.traceid_list)):
            if len(optsE.traceid_list)==1:
                common_traceid = optsE.traceid_list[0]
            elif optsE.traceid is not None:
                common_traceid = optsE.traceid
        optsE.traceid_list = [common_traceid for _ in range(len(optsE.run_list))]

    print "******************************************"
    print "Processing runs [traceids]:"
    print optsE.run_list 
    print optsE.traceid_list

    for run, traceid in zip(optsE.run_list, optsE.traceid_list):
        print run, traceid
    print "******************************************"
   

    processed_run_fpath = process_run_data(optsE.animalid, optsE.session, optsE.acquisition, optsE.run_list, optsE.traceid_list, rootdir=optsE.rootdir,
                                           stimtype=optsE.stimtype, data_type=optsE.data_type, metric=optsE.metric,
                                           make_equal=optsE.make_equal, create_new=optsE.create_new, nproc=int(optsE.nprocesses), pval_selective=optsE.pval_selective, pval_visual=optsE.pval_visual, visual_test_type=optsE.visual_test_type, response_metric=optsE.response_metric)

    print "Finished processing data!"
    print "Saved processed run to:\n--> %s" % processed_run_fpath
   
    # PLOT:
    if optsE.plot_psth is True and optsE.stimtype is not None:
        psth_opts = ['-D', optsE.rootdir, '-i', optsE.animalid, '-S', optsE.session, '-A', optsE.acquisition, '-R', 'combined_%s_static' % optsE.stimtype, '-t', optsE.traceid]
        psth_opts.extend(['-d', optsE.psth_dtype])
        if optsE.psth_rows is not None and optsE.psth_rows != 'None':
            psth_opts.extend(['-r', optsE.psth_rows])
        if optsE.psth_cols is not None and optsE.psth_cols != 'None':
            psth_opts.extend(['-c', optsE.psth_cols])
        if optsE.psth_hues is not None and optsE.psth_hues!='None':
            print "Specified HUE:", optsE.psth_hues
            psth_opts.extend(['-H', optsE.psth_hues])
        psth_opts.extend(['--shade'])

        psth_dir = psth.make_clean_psths(psth_opts)

        print "*******************************************************************"
        print "DONE!"
        print "All output saved to: %s" % psth_dir
        print "*******************************************************************"
 
      
if __name__ == '__main__':
    main(sys.argv[1:]) 
