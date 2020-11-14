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

parser.add_argument('-x', '--analysis', dest='analysis_type', action='store', default='by_ncells', help='Analysis type (options: by_ncells, single_cells. default=by_ncells')


parser.add_argument('-C', '--cvalue', dest='c_value', action='store', default=1.0, help='C value (default=1, set None to tune)')


parser.add_argument('-v', '--area', dest='visual_area', action='store', default=None, help='Visual area to process (default, all)')





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


ROOTDIR = '/n/coxfs01/2p-data'
EXPERIMENT = args.experiment_type
email = args.email

visual_area = None if args.visual_area in ['None', None] else args.visual_area

traceid = args.traceid
responsive_test = args.responsive_test
overlap_thr = float(args.overlap_thr)

analysis_type = args.analysis_type
c_value = None if args.c_value in ['None', None] else float(args.c_value)


# Create a (hopefully) unique prefix for the names of all jobs in this 
# particular run of the pipeline. This makes sure that runs can be
# identified unambiguously
piper = uuid.uuid4()
piper = str(piper)[0:4]
logdir = 'LOG__%s_%s_%s_overlap%i' % (analysis_type, str(visual_area), EXPERIMENT, int(overlap_thr*10)) 
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
sys.stdout = open('%s/INFO_%s_%s_%s.txt' % (logdir, analysis_type, piper, EXPERIMENT), 'w')

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
# STEP1: TIFF PROCESSING. All PIDs will be processed in parallel since they don't
#        have any dependencies

C_str = 'tuneC' if c_value is None else 'C-%.2f' % c_value

jobids = [] # {}
if analysis_type=='by_ncells':
    # -----------------------------------------------------------------
    # BY NCELLS
    # -----------------------------------------------------------------
    datakey=None
    ncells_test = [2**i for i in np.arange(0, 9)]  
    visual_areas = ['V1', 'Lm', 'Li']
    info("Testing %i areas: %s" % (len(visual_areas), str(visual_areas)))
    info("Testing %i sample size: %s" % (len(ncells_test), str(ncells_test)))

    for visual_area in visual_areas: #['V1', 'Lm', 'Li']:
        for ncells in ncells_test:
            #print("[%s]: %i (%s)" % (visual_area, ncells, C_str))
            mtag = '%s_%s_%i' % (visual_area, C_str, ncells) 
            #
            cmd = "sbatch --job-name={PROCID}.{ANALYSIS}.{MTAG} \
                -o '{LOGDIR}/{PROCID}.{ANALYSIS}.{MTAG}.out' \
                -e '{LOGDIR}/{PROCID}.{ANALYSIS}.{MTAG}.err' \
                /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/slurm/decode_by_ncells.sbatch \
        {EXP} {TRACEID} {RTEST} {OVERLAP} {ANALYSIS} {CVAL} {VAREA} {NCELLS} {DKEY}".format(
                PROCID=piper, MTAG=mtag, LOGDIR=logdir,
                EXP=EXPERIMENT, TRACEID=traceid, ANALYSIS=analysis_type,
                RTEST=responsive_test, OVERLAP=overlap_thr, 
                CVAL=c_value, VAREA=visual_area, NCELLS=ncells, DKEY=datakey) 
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
    dsets = load_metadata(EXPERIMENT, visual_area=visual_area)
    incl = ['20190614_JC091_fov1']
    #dsets = dsets[dsets['datakey'].isin(incl)]
    if len(dsets)==0:
        fatal("NO FOVs found.")
    info("Found %i [%s] datasets to process." % (len(dsets), EXPERIMENT))
    for (visual_area, datakey), g in dsets.groupby(['visual_area', 'datakey']):
        #print("[%s]: %s (%s)" % (visual_area, datakey, C_str))
        mtag = '%s_%s_%s' % (datakey, visual_area, C_str) 
        #
        cmd = "sbatch --job-name={PROCID}.{MTAG}.{ANALYSIS} \
                -o '{LOGDIR}/{PROCID}.{MTAG}.{ANALYSIS}.out' \
                -e '{LOGDIR}/{PROCID}.{MTAG}.{ANALYSIS}.err' \
                /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/slurm/decode_single_cells.sbatch \
        {EXP} {TRACEID} {RTEST} {OVERLAP} {ANALYSIS} {CVAL} {VAREA} {NCELLS} {DKEY}".format(
            PROCID=piper, MTAG=mtag, LOGDIR=logdir,
            EXP=EXPERIMENT, TRACEID=traceid, ANALYSIS=analysis_type,
            RTEST=responsive_test, OVERLAP=overlap_thr, 
            CVAL=c_value, VAREA=visual_area, NCELLS=ncells, DKEY=datakey) 
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
                  {JOBDEP} {EMAIL}".format(JOBDEP=jobdep, EMAIL=email, EXP=EXPERIMENT)
    #info("Submitting MCEVAL job with CMD:\n%s" % cmd)
    status, joboutput = commands.getstatusoutput(cmd)
    jobnum = joboutput.split(' ')[-1]
    #eval_jobids[phash] = jobnum
    #info("MCEVAL calling jobids [%s]: %s" % (phash, jobnum))
    print("... checking: %s (%s)" % (jobdep, jobnum))




