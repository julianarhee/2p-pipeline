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
match_distns = args.match_distns

trial_epoch = args.trial_epoch

# Create a (hopefully) unique prefix for the names of all jobs in this 
# particular run of the pipeline. This makes sure that runs can be
# identified unambiguously
piper = uuid.uuid4()
piper = str(piper)[0:4]
match_str = 'matchdistns_' if match_distns else ''
if overlap_thr is None:
    logdir = 'LOG__%s%s_%s_%s__%s_no-rfs' % (match_str, analysis_type, str(visual_area), experiment, trial_epoch) 
else:
    logdir = 'LOG__%s%s_%s_%s__%s_overlap-%i' % (match_str, analysis_type, str(visual_area), experiment,  trial_epoch, int(overlap_thr*10)) 
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
    ncells_test = [2**i for i in np.arange(0, 9)]  
    if visual_area is not None:
        visual_areas = [visual_area]    
    else:
        visual_areas = ['V1', 'Lm', 'Li']
    info("Testing %i areas: %s" % (len(visual_areas), str(visual_areas)))
    info("Testing %i sample size: %s" % (len(ncells_test), str(ncells_test)))

    for visual_area in visual_areas:
        for ncells in ncells_test:
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
                cmd = "sbatch --job-name={PROCID}.{ANALYSIS}.{MTAG} \
                -o '{LOGDIR}/{PROCID}.{ANALYSIS}.{MTAG}.out' \
                -e '{LOGDIR}/{PROCID}.{ANALYSIS}.{MTAG}.err' \
        /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/slurm/decode_by_ncells.sbatch \
        {EXP} {TRACEID} {RTEST} {OVERLAP} {ANALYSIS} {CVAL} {VAREA} {NCELLS} {DKEY} {EPOCH}".format(
                    PROCID=piper, MTAG=mtag, LOGDIR=logdir,
                    EXP=experiment, TRACEID=traceid, ANALYSIS=analysis_type,
                    RTEST=responsive_test, OVERLAP=overlap_thr, 
                    CVAL=c_value, VAREA=visual_area, NCELLS=ncells, DKEY=datakey, EPOCH=trial_epoch) 
                #
            status, joboutput = commands.getstatusoutput(cmd)
            jobnum = joboutput.split(' ')[-1]
            jobids.append(jobnum)
            info("[%s]: %s" % (jobnum, mtag))
elif analysis_type in ['by_fov', 'split_pupil']:
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

    for (visual_area, datakey), g in dsets.groupby(['visual_area', 'datakey']):
        mtag = '%s_%s_%s' % (datakey, visual_area, C_str) 
        #
        cmd = "sbatch --job-name={procid}.{mtag}.{analysis} \
                -o '{logdir}/{procid}.{mtag}.{analysis}.out' \
                -e '{logdir}/{procid}.{mtag}.{analysis}.err' \
        /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/slurm/decode_by_ncells.sbatch \
        {exp} {traceid} {rtest} {overlap} {analysis} {cval} {varea} {ncells} {dkey}".format(
            procid=piper, mtag=mtag, logdir=logdir,
            exp=experiment, traceid=traceid, analysis=analysis_type,
            rtest=responsive_test, overlap=overlap_thr, 
            cval=c_value, varea=visual_area, ncells=ncells, dkey=datakey) 
        #
        status, joboutput = commands.getstatusoutput(cmd)
        jobnum = joboutput.split(' ')[-1]
        jobids.append(jobnum)
        info("[%s]: %s" % (jobnum, mtag))


else:
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
        mtag = '%s_%s_%s' % (datakey, visual_area, c_str) 
        #
        cmd = "sbatch --job-name={procid}.{mtag}.{analysis} \
                -o '{logdir}/{procid}.{mtag}.{analysis}.out' \
                -e '{logdir}/{procid}.{mtag}.{analysis}.err' \
        /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/slurm/decode_single_cells.sbatch \
        {exp} {traceid} {rtest} {overlap} {analysis} {cval} {varea} {ncells} {dkey}".format(
            procid=piper, mtag=mtag, logdir=logdir,
            exp=experiment, traceid=traceid, analysis=analysis_type,
            rtest=responsive_test, overlap=overlap_thr, 
            cval=c_value, varea=visual_area, ncells=ncells, dkey=datakey) 
        #
        status, joboutput = commands.getstatusoutput(cmd)
        jobnum = joboutput.split(' ')[-1]
        jobids.append(jobnum)
        info("[%s]: %s" % (jobnum, mtag))

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




