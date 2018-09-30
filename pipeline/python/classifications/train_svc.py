#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 16:32:57 2018

@author: juliana
"""


import sys
import optparse
import math
import os
import time
import multiprocessing as mp
import cPickle as pkl
from pipeline.python.classifications import linearSVC_class as svc
from pipeline.python.utils import print_elapsed_time

#%%

def extract_options(options):

    def comma_sep_list(option, opt, value, parser):
        setattr(parser.values, option.dest, value.split(','))


    parser = optparse.OptionParser()

    parser.add_option('-D', '--root', action='store', dest='rootdir',
                          default='/nas/volume1/2photon/data',
                          help='data root dir (dir w/ all animalids) [default: /nas/volume1/2photon/data, /n/coxfs01/2pdata if --slurm]')
    parser.add_option('-i', '--animalid', action='store', dest='animalid',
                          default='', help='Animal ID')

    # Set specific session/run for current animal:
    parser.add_option('-S', '--session', action='store', dest='session',
                          default='', help='session dir (format: YYYMMDD_ANIMALID')
    parser.add_option('-A', '--acq', action='store', dest='acquisition',
                          default='FOV1', help="acquisition folder (ex: 'FOV1_zoom3x') [default: FOV1]")
    parser.add_option('-R', '--run', action='store', dest='run',
                          default='', help="RUN name (e.g., gratings_run1)")
    parser.add_option('-t', '--traceid', action='store', dest='traceid',
                          default='', help="traceid name (e.g., traces001)")
    
    parser.add_option('--slurm', action='store_true', dest='slurm', default=False, help="set if running as SLURM job on Odyssey")
    parser.add_option('--par', action='store_true', dest='multiproc', default=False, help="set if want to run MP on roi stats, when possible")
    parser.add_option('--nproc', action='store', dest='nprocesses', default=4, help="N processes if running in par (default=4)")

    # Classifier info:
    parser.add_option('-r', '--rois', action='store', dest='roi_selector', default='all', help="(options: all, visual)")
    parser.add_option('-d', '--dtype', action='store', dest='data_type', default='stat', help="(options: frames, stat)")
    stat_choices = {'stat': ['meanstim', 'meanstimdff', 'zscore'],
                    'frames': ['trial', 'stimulus', 'post']}
    parser.add_option('-s', '--stype', action='store', dest='stat_type', default='meanstim', 
                      help="If dtype is STAT, options: %s. If dtype is FRAMES, options: %s" % (str(stat_choices['stat']), str(stat_choices['frames'])))

    parser.add_option('-p', '--indata_type', action='store', dest='inputdata_type', default='corrected', help="data processing type (dff, corrected, raw, etc.)")
    parser.add_option('--null', action='store_true', dest='get_null', default=False, help='Include a null class in addition to stim conditions')
    parser.add_option('-N', '--name', action='store', dest='class_name', default='', help='Name of transform to classify (e.g., ori, xpos, morphlevel, etc.)')
    
    parser.add_option('--subset', action='store', dest='class_subset', default='', help='Subset of class_name types to learn')
    parser.add_option('-c', '--const', dest='const_trans', default='', type='string', action='callback', 
                          callback=comma_sep_list, help="Transform name to hold constant if classifying a different transform")
    parser.add_option('-v', '--tval', dest='trans_value', default='', type='string', action='callback', 
                          callback=comma_sep_list, help="Value to set const_trans to")

    parser.add_option('-L', '--clf', action='store', dest='classifier', default='LinearSVC', help='Classifier type (default: LinearSVC)')
    parser.add_option('-k', '--cv', action='store', dest='cv_method', default='kfold', help='Method of cross-validation (default: kfold)')
    parser.add_option('-f', '--folds', action='store', dest='cv_nfolds', default=5, help='N folds for CV (default: 5)')
    parser.add_option('-C', '--cval', action='store', dest='C_val', default=1e9, help='Value for C param if using SVC (default: 1e9)')
    parser.add_option('-g', '--groups', action='store', dest='cv_ngroups', default=1, help='N groups for CV, relevant only for data_type=frames (default: 1)')
    parser.add_option('-b', '--bin', action='store', dest='binsize', default=10, help='Bin size, relevant only for data_type=frames (default: 10)')

    (options, args) = parser.parse_args(options)
    
    assert options.stat_type in stat_choices[options.data_type], "Invalid STAT selected for data_type %s. Run -h for options." % options.data_type

    return options

#%%

options = ['-D', '/mnt/odyssey', '-i', 'JC015', '-S', '20180915', '-A', 'FOV1_zoom2p7x',
           '-R', 'combined_gratings_static', '-t', 'traces001',
           '-r', 'visual', '-d', 'stat', '-s', 'meanstim',
           '-p', 'corrected', '-N', 'ori',
           '-c', 'xpos',
           '--nproc=4'
           ]


def train_test_linearSVC(options):

    optsE = extract_options(options)
    nprocs = int(optsE.nprocesses)
    
    C = svc.TransformClassifier(optsE)
    C.load_dataset()
    C.create_classifier_dirs()
    C.initialize_classifiers()
    
    C.label_classifier_data()
    
    t_train_mp = time.time()
    def trainer(C, clf_list, out_q):
        
        outdict = {}
        for clf in clf_list:
            C.train_classifier(clf)
            outdict[os.path.split(clf.classifier_dir)[-1]] = 'done!'
        out_q.put(outdict)
        
    # Each process gets "chunksize' filenames and a queue to put his out-dict into:
    clf_list = C.classifiers
    out_q = mp.Queue()
    chunksize = int(math.ceil(len(C.classifiers) / float(nprocs)))
    procs = []
    for i in range(nprocs):
        p = mp.Process(target=trainer, 
                       args=(C, 
                             clf_list[chunksize * i:chunksize * (i + 1)],
                             out_q))
        procs.append(p)
        p.start()

    # Collect all results into single results dict. We should know how many dicts to expect:
    resultdict = {}
    for i in range(nprocs):
        resultdict.update(out_q.get())

    # Wait for all worker processes to finish
    for p in procs:
        print "Finished:", p
        p.join()
        
    print_elapsed_time(t_train_mp)
    print resultdict

    with open(os.path.join(C.classifier_dir, 'TransformClassifier.pkl'), 'wb') as f:
        pkl.dump(C, f, protocol=pkl.HIGHEST_PROTOCOL)
    print "Saved object to:", os.path.join(C.classifier_dir, 'TransformClassifier.pkl')

    
#%%

def main(options):
    train_test_linearSVC(options)



if __name__ == '__main__':
    main(sys.argv[1:])
