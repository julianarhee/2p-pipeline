#!/usr/local/bin/python

import argparse
import uuid
import sys
import os
import commands
import json
import glob
import pandas as pd
import cPickle as pkl
import numpy as np

parser = argparse.ArgumentParser(
    description = '''Look for XID files in session directory.\nFor PID files, run tiff-processing and evaluate.\nFor RID files, wait for PIDs to finish (if applicable) and then extract ROIs and evaluate.\n''',
    epilog = '''AUTHOR:\n\tJuliana Rhee''')

parser.add_argument('-E', '--exp', dest='experiment_type', action='store', default='rfs', help='Experiment type (e.g., rfs')

parser.add_argument('-e', '--email', dest='email', action='store', default='rhee@g.harvard.edu', help='Email to send log files')
parser.add_argument('-t', '--traceid', dest='traceid', action='store', default=None, help='Traceid to use as reference for selecting retino analysis')

parser.add_argument('-R', '--resp-test', dest='responsive_test', action='store', default='nstds', help='Responsive test (default=nstds, options: ROC, nstds, None)')
parser.add_argument('-o', '--overlap', dest='overlap_thr', action='store', default=0.5, help='Overlap threshold (default=0.5)')

parser.add_argument('-X', '--analysis', dest='analysis_type', action='store', default='by_ncells', help='Analysis type (options: by_ncells, single_cells. default=by_ncells')


parser.add_argument('-C', '--cvalue', dest='c_value', action='store', default=None, help='C value (default=None, tune C)')


parser.add_argument('-v', '--area', dest='visual_area', action='store', default=None, help='Visual area to process (default, all)')
parser.add_argument('-k', '--datakeys', nargs='*', dest='included_datakeys', action='append', help='Use like: -k DKEY DKEY DKEY')
parser.add_argument('--match', dest='match_distns', action='store_true', default=False, help='Set if match distns (only if --analysis=by_ncells)')

parser.add_argument('--epoch', dest='trial_epoch', action='store', default='stimulus', help='Trial epoch for data input (options: stimulus, firsthalf, plushalf, baseline. default=stimulus')

parser.add_argument('--snr', dest='threshold_snr', action='store_true', default=False, help='Set to threshold SNR')
parser.add_argument('--retino', dest='has_retino', action='store_true', default=False, help='Set to filter rois by retino')
parser.add_argument('--dff', dest='threshold_dff', action='store_true', default=False, help='Set to filter rois by dff')


parser.add_argument('-S', '--sample-sizes', nargs='+', dest='sample_sizes', default=[], help='Use like: -S 1 2 4')

parser.add_argument('-T', '--test', action='store', dest='test_type', default=None, help='Test type, if generalization (options: size_single, size_subset, morph)')

parser.add_argument('--shuffle-thr', action='store', dest='shuffle_thr', default=0.05, help='Thr for shuffle test (default=0.05)')
parser.add_argument('--shuffle-drop', action='store_true', dest='shuffle_drop', default=False, help='Set to drop repeats for aggregated dsets)')




args = parser.parse_args()


def info(info_str):
    sys.stdout.write("INFO:  %s \n" % info_str)
    sys.stdout.flush() 

def error(error_str):
    sys.stdout.write("ERR:  %s \n" % error_str)
    sys.stdout.flush() 

def fatal(error_str):
    sys.stdout.write("ERR: %s \n" % error_str)
    sys.stdout.flush()
    sys.exit(1)

def load_metadata(experiment, responsive_test='nstds', responsive_thr=10.,
                  rootdir='/n/coxfs01/2p-data', visual_area=None,
                  aggregate_dir='/n/coxfs01/julianarhee/aggregate-visual-areas',
                  traceid='traces001'):
    from pipeline.python.classifications import aggregate_data_stats as aggr
    sdata = aggr.get_aggregate_info(traceid=traceid) #, fov_type=fov_type, state=state)
    #print('%s:' % experiment, sdata.head())
    sdata_exp = sdata[sdata['experiment']==experiment] 
  
    if visual_area is not None:
        sdata_exp = sdata_exp[sdata_exp['visual_area']==visual_area]
 
    return sdata_exp


# -----------------------------------------------------------------
# ARGS
# -----------------------------------------------------------------
ROOTDIR = '/n/coxfs01/2p-data'
experiment = args.experiment_type
email = args.email

visual_area = None if args.visual_area in ['None', None] else args.visual_area
traceid = args.traceid
responsive_test = args.responsive_test
overlap_thr = None if args.overlap_thr in ['None', None] else float(args.overlap_thr)

analysis_type = args.analysis_type
c_value = None if args.c_value in ['None', None] else float(args.c_value)
c_str = 'tune-C' if c_value is None else 'C-%.2f' % c_value
trial_epoch = args.trial_epoch
shuffle_thr = args.shuffle_thr
shuffle_drop = args.shuffle_drop

match_distns = args.match_distns
threshold_snr = args.threshold_snr
has_retino = args.has_retino
threshold_dff = args.threshold_dff

test_type = args.test_type
sample_sizes = [int(i) for i in args.sample_sizes]

# Create a (hopefully) unique prefix for the names of all jobs in this 
# particular run of the pipeline. This makes sure that runs can be
# identified unambiguously
piper = uuid.uuid4()
piper = str(piper)[0:4]
if threshold_snr:
    match_str='snr_'
else:
    match_str = 'matchdistns_' if match_distns else ''

if threshold_dff:
    overlap_str = 'threshdff'
else:
    if has_retino:
        overlap_str = 'retino'
    else:
        overlap_str = 'noRF' if overlap_thr is None else 'overlap%i' % int(overlap_thr*10)

test_str = 'TEST_%s' % test_type if test_type is not None else ''

# Set LOGS
logdir = 'LOG_%s__%s%s_%s_%s__%s_%s' % (analysis_type, match_str, str(visual_area), experiment,  trial_epoch, overlap_str, test_str) 
if not os.path.exists(logdir):
    os.mkdir(logdir)

# Remove old logs
old_logs = glob.glob(os.path.join(logdir, '*.err'))
old_logs.extend(glob.glob(os.path.join(logdir, '*.out')))
old_logs.extend(glob.glob(os.path.join(logdir, '*.txt')))

for r in old_logs:
    os.remove(r)

#####################################################################
#                          find XID files                           #
#####################################################################
# Note: the syntax a+=(b) adds b to the array a
# Open log lfile
sys.stdout = open('%s/INFO_%s_%s_%s.txt' % (logdir, analysis_type, piper, experiment), 'w')

def load_metadata(experiment, responsive_test='nstds', responsive_thr=10.,
                  rootdir='/n/coxfs01/2p-data', visual_area=None,
                  aggregate_dir='/n/coxfs01/julianarhee/aggregate-visual-areas',
                  traceid='traces001'):
    from pipeline.python.classifications import aggregate_data_stats as aggr
    sdata = aggr.get_aggregate_info(traceid=traceid) #, fov_type=fov_type, state=state)
    print("%s: %s" % (experiment, sdata.head()))
    sdata_exp = sdata[sdata['experiment']==experiment] 
      
    if visual_area is not None:
        sdata_exp = sdata_exp[sdata_exp['visual_area']==visual_area]
 
    return sdata_exp

################################################################################
#                               run the pipeline                               #
################################################################################

C_str = 'tuneC' if c_value is None else 'C-%.2f' % c_value
jobids = [] # {}
if analysis_type=='by_ncells':
    # -----------------------------------------------------------------
    # BY NCELLS
    # -----------------------------------------------------------------
    datakey=None
    if len(sample_sizes)==0:
        sample_sizes = [1, 4, 16, 32, 64, 128, 256] #[2**i for i in np.arange(0, 9)]  
    visual_areas = ['V1', 'Lm', 'Li'] if visual_area is None else [visual_area]
    info("Testing %i areas: %s" % (len(visual_areas), str(visual_areas)))
    info("Testing %i sample size: %s" % (len(sample_sizes), str(sample_sizes)))

    for visual_area in visual_areas:
        for ncells in sample_sizes: #ncells_test:
            mtag = '%s_%s_%i' % (visual_area, C_str, ncells) 
            #
            if match_distns:
                cmd = "sbatch --job-name={PROCID}.{ANALYSIS}.{MTAG} \
                -o '{LOGDIR}/{PROCID}.{ANALYSIS}.{MTAG}.out' \
                -e '{LOGDIR}/{PROCID}.{ANALYSIS}.{MTAG}.err' \
    /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/slurm/decode_by_ncells_match.sbatch \
        {EXP} {TRACEID} {RTEST} {OVERLAP} {ANALYSIS} {CVAL} {VAREA} {NCELLS} {DKEY} {EPOCH}".format(
                    PROCID=piper, MTAG=mtag, LOGDIR=logdir,
                    EXP=experiment, TRACEID=traceid, ANALYSIS=analysis_type,
                    RTEST=responsive_test, OVERLAP=overlap_thr, 
                    CVAL=c_value, VAREA=visual_area, NCELLS=ncells, DKEY=datakey, EPOCH=trial_epoch) 
 
            else:
                if threshold_snr:
                    cmd = "sbatch --job-name={PROCID}.{ANALYSIS}.{MTAG} \
                    -o '{LOGDIR}/{PROCID}.{ANALYSIS}.{MTAG}.out' \
                    -e '{LOGDIR}/{PROCID}.{ANALYSIS}.{MTAG}.err' \
            /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/slurm/decode_by_ncells_snr.sbatch \
            {EXP} {TRACEID} {RTEST} {OVERLAP} {ANALYSIS} {CVAL} {VAREA} {NCELLS} {DKEY} {EPOCH}".format(
                        PROCID=piper, MTAG=mtag, LOGDIR=logdir,
                        EXP=experiment, TRACEID=traceid, ANALYSIS=analysis_type,
                        RTEST=responsive_test, OVERLAP=overlap_thr, 
                        CVAL=c_value, VAREA=visual_area, NCELLS=ncells, DKEY=datakey, EPOCH=trial_epoch) 
                elif has_retino:
                    cmd = "sbatch --job-name={PROCID}.{ANALYSIS}.{MTAG} \
                    -o '{LOGDIR}/{PROCID}.{ANALYSIS}.{MTAG}.out' \
                    -e '{LOGDIR}/{PROCID}.{ANALYSIS}.{MTAG}.err' \
            /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/slurm/decode_by_ncells_retino.sbatch \
            {EXP} {TRACEID} {RTEST} {OVERLAP} {ANALYSIS} {CVAL} {VAREA} {NCELLS} {DKEY} {EPOCH}".format(
                        PROCID=piper, MTAG=mtag, LOGDIR=logdir,
                        EXP=experiment, TRACEID=traceid, ANALYSIS=analysis_type,
                        RTEST=responsive_test, OVERLAP=overlap_thr, 
                        CVAL=c_value, VAREA=visual_area, NCELLS=ncells, DKEY=datakey, EPOCH=trial_epoch) 
                elif threshold_dff:
                    cmd = "sbatch --job-name={PROCID}.{ANALYSIS}.{MTAG} \
                    -o '{LOGDIR}/{PROCID}.{ANALYSIS}.{MTAG}.out' \
                    -e '{LOGDIR}/{PROCID}.{ANALYSIS}.{MTAG}.err' \
            /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/slurm/decode_by_ncells_dff.sbatch \
            {EXP} {TRACEID} {RTEST} {OVERLAP} {ANALYSIS} {CVAL} {VAREA} {NCELLS} {DKEY} {EPOCH}".format(
                        PROCID=piper, MTAG=mtag, LOGDIR=logdir,
                        EXP=experiment, TRACEID=traceid, ANALYSIS=analysis_type,
                        RTEST=responsive_test, OVERLAP=overlap_thr, 
                        CVAL=c_value, VAREA=visual_area, NCELLS=ncells, DKEY=datakey, EPOCH=trial_epoch) 


                else: 
                    if shuffle_drop:
                        cmd = "sbatch --job-name={PROCID}.{ANALYSIS}.{MTAG} \
                        -o '{LOGDIR}/{PROCID}.{ANALYSIS}.{MTAG}.out' \
                        -e '{LOGDIR}/{PROCID}.{ANALYSIS}.{MTAG}.err' \
                /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/slurm/decode_by_ncells_thr_drop.sbatch \
                {EXP} {TRACEID} {RTEST} {OVERLAP} {ANALYSIS} {CVAL} {VAREA} {NCELLS} {DKEY} {EPOCH} {TEST} {PASS}".format(
                            PROCID=piper, MTAG=mtag, LOGDIR=logdir,
                            EXP=experiment, TRACEID=traceid, ANALYSIS=analysis_type,
                            RTEST=responsive_test, OVERLAP=overlap_thr, 
                            CVAL=c_value, VAREA=visual_area, NCELLS=ncells, DKEY=datakey, 
                            EPOCH=trial_epoch, TEST=test_type, PASS=shuffle_thr) 
                    else: 
                        cmd = "sbatch --job-name={PROCID}.{ANALYSIS}.{MTAG} \
                        -o '{LOGDIR}/{PROCID}.{ANALYSIS}.{MTAG}.out' \
                        -e '{LOGDIR}/{PROCID}.{ANALYSIS}.{MTAG}.err' \
                /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/slurm/decode_by_ncells_thr.sbatch \
                {EXP} {TRACEID} {RTEST} {OVERLAP} {ANALYSIS} {CVAL} {VAREA} {NCELLS} {DKEY} {EPOCH} {TEST} {PASS}".format(
                            PROCID=piper, MTAG=mtag, LOGDIR=logdir,
                            EXP=experiment, TRACEID=traceid, ANALYSIS=analysis_type,
                            RTEST=responsive_test, OVERLAP=overlap_thr, 
                            CVAL=c_value, VAREA=visual_area, NCELLS=ncells, DKEY=datakey, 
                            EPOCH=trial_epoch, TEST=test_type, PASS=shuffle_thr) 
                    #
            status, joboutput = commands.getstatusoutput(cmd)
            jobnum = joboutput.split(' ')[-1]
            jobids.append(jobnum)
            info("[%s]: %s" % (jobnum, mtag))

elif analysis_type in ['by_fov', 'split_pupil']:
    # -----------------------------------------------------------------
    # BY FOV 
    # -----------------------------------------------------------------
    ncells=None
    dsets = load_metadata(experiment, visual_area=visual_area)
    included_datakeys = args.included_datakeys
    print("dkeys:", included_datakeys)
    #['20190614_jc091_fov1', '20190602_jc091_fov1', '20190609_jc099_fov1']
    if included_datakeys is not None: #len(included_datakeys) > 0:
        dsets = dsets[dsets['datakey'].isin(included_datakeys[0])]
        
    if len(dsets)==0:
        fatal("no fovs found.")
    info("found %i [%s] datasets to process." % (len(dsets), experiment))

    old_rats = ['JC061', 'JC067', 'JC070', 'JC073']
    for (visual_area, datakey), g in dsets.groupby(['visual_area', 'datakey']):
        #if analysis_type=='split_pupil' and datakey.split('_')[1] not in old_rats:
        #    info("--- skipping %s" % datakey)
        #    continue

        mtag = '%s_%s_%s' % (visual_area, datakey,C_str) 
        if test_type is not None:
            mtag = 'GEN%s-%s-%s' % (test_type, responsive_test, mtag)

        if has_retino:
            cmd = "sbatch --job-name={analysis}.{procid}.{mtag} \
                    -o '{logdir}/{procid}.{mtag}.{analysis}.out' \
                    -e '{logdir}/{procid}.{mtag}.{analysis}.err' \
            /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/slurm/decode_by_ncells_retino.sbatch \
            {exp} {traceid} {rtest} {overlap} {analysis} {cval} {varea} {ncells} {dkey} {epoch}".format(
                procid=piper, mtag=mtag, logdir=logdir,
                exp=experiment, traceid=traceid, analysis=analysis_type,
                rtest=responsive_test, overlap=overlap_thr, 
                cval=c_value, varea=visual_area, ncells=ncells, dkey=datakey, epoch=trial_epoch) 
        elif threshold_dff:
            cmd = "sbatch --job-name={procid}.{mtag}.{analysis} \
                    -o '{logdir}/{procid}.{mtag}.{analysis}.out' \
                    -e '{logdir}/{procid}.{mtag}.{analysis}.err' \
            /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/slurm/decode_by_ncells_dff.sbatch \
            {exp} {traceid} {rtest} {overlap} {analysis} {cval} {varea} {ncells} {dkey} {epoch}".format(
                procid=piper, mtag=mtag, logdir=logdir,
                exp=experiment, traceid=traceid, analysis=analysis_type,
                rtest=responsive_test, overlap=overlap_thr, 
                cval=c_value, varea=visual_area, ncells=ncells, dkey=datakey, epoch=trial_epoch) 
        #
        else:
            cmd = "sbatch --job-name={procid}.{mtag}.{analysis} \
                    -o '{logdir}/{procid}.{mtag}.{analysis}.out' \
                    -e '{logdir}/{procid}.{mtag}.{analysis}.err' \
            /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/slurm/decode_by_ncells.sbatch \
            {exp} {traceid} {rtest} {overlap} {analysis} {cval} {varea} {ncells} {dkey} {epoch} {test}".format(
                procid=piper, mtag=mtag, logdir=logdir,
                exp=experiment, traceid=traceid, analysis=analysis_type,
                rtest=responsive_test, overlap=overlap_thr, 
                cval=c_value, varea=visual_area, ncells=ncells, dkey=datakey, epoch=trial_epoch, test=test_type) 
        #
        status, joboutput = commands.getstatusoutput(cmd)
        jobnum = joboutput.split(' ')[-1]
        jobids.append(jobnum)
        info("[%s]: %s" % (jobnum, mtag))


elif analysis_type=='single_cells':
    # -----------------------------------------------------------------
    # SINGLE CELLS 
    # -----------------------------------------------------------------
    ncells=None
    dsets = load_metadata(experiment, visual_area=visual_area)
    included_datakeys=[]
    if args.included_datakeys is not None:
        included_datakeys = args.included_datakeys[0]
    print("dkeys:", included_datakeys)

 
    #['20190614_jc091_fov1', '20190602_jc091_fov1', '20190609_jc099_fov1']
    if len(included_datakeys) > 0:
        dsets = dsets[dsets['datakey'].isin(included_datakeys)]

    if len(dsets)==0:
        fatal("no fovs found.")
    info("found %i [%s] datasets to process." % (len(dsets), experiment))
    for (visual_area, datakey), g in dsets.groupby(['visual_area', 'datakey']):
        #mtag = '%s_%s_%s' % (datakey, visual_area, c_str) 
        mtag = '%s_%s_%s' % (visual_area, datakey,C_str) 
        if test_type is not None:
            mtag = 'GEN%s-%s-%s' % (test_type, responsive_test, mtag)
 
        #
        cmd = "sbatch --job-name={procid}.{analysis}.{mtag} \
                -o '{logdir}/{procid}.{mtag}.{analysis}.out' \
                -e '{logdir}/{procid}.{mtag}.{analysis}.err' \
        /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/slurm/decode_single_cells.sbatch \
        {exp} {traceid} {rtest} {overlap} {analysis} {cval} {varea} {ncells} {dkey} {epoch} {test}".format(
            procid=piper, mtag=mtag, logdir=logdir,
            exp=experiment, traceid=traceid, analysis=analysis_type,
            rtest=responsive_test, overlap=overlap_thr, 
            cval=c_value, varea=visual_area, ncells=ncells, dkey=datakey, epoch=trial_epoch, test=test_type) 
        #
        status, joboutput = commands.getstatusoutput(cmd)
        jobnum = joboutput.split(' ')[-1]
        jobids.append(jobnum)
        info("[%s]: %s" % (jobnum, mtag))

else:
    info("WARNING: UNKNOWN ANALYSIS_TYPE <%s> %" % str(analysis_type))

info("****done!****")

for jobdep in jobids:
    print(jobdep)
    cmd = "sbatch --job-name={JOBDEP}.checkstatus \
		-o 'log/checkstatus.{EXP}.{JOBDEP}.out' \
		-e 'log/checkstatus.{EXP}.{JOBDEP}.err' \
                  --depend=afternotok:{JOBDEP} \
                  /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/slurm/checkstatus.sbatch \
                  {JOBDEP} {EMAIL}".format(JOBDEP=jobdep, EMAIL=email, EXP=experiment)
    #info("Submitting MCEVAL job with CMD:\n%s" % cmd)
    status, joboutput = commands.getstatusoutput(cmd)
    jobnum = joboutput.split(' ')[-1]
    #eval_jobids[phash] = jobnum
    #info("MCEVAL calling jobids [%s]: %s" % (phash, jobnum))
    print("... checking: %s (%s)" % (jobdep, jobnum))




