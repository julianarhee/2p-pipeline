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

parser = argparse.ArgumentParser(
    description = '''Look for XID files in session directory.\nFor PID files, run tiff-processing and evaluate.\nFor RID files, wait for PIDs to finish (if applicable) and then extract ROIs and evaluate.\n''',
    epilog = '''AUTHOR:\n\tJuliana Rhee''')
parser.add_argument('-A', '--fov', dest='fov_type', action='store', default='zoom2p0x', help='FOV type (e.g., zoom2p0x)')
parser.add_argument('-E', '--exp', dest='experiment_type', action='store', default='blobs', help='Experiment type (e.g., blobs)')
parser.add_argument('-e', '--email', dest='email', action='store', default='rhee@g.harvard.edu', help='Email to send log files')
parser.add_argument('-t', '--traceid', dest='traceid', action='store', default='traces001', help='Traceid to use as reference for selecting retino analysis')
parser.add_argument('-v', '--area', dest='visual_area', action='store', default=None, help='visual area (optional, otherwise, does all)')
parser.add_argument('-a', '--align', dest='alignment_type', action='store', default='stimulus', help='(Type of alignment, options: stimulus, trial [default: stimulus])')



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



ROOTDIR = '/n/coxfs01/2p-data'
FOV = args.fov_type
EXPERIMENT = args.experiment_type
email = args.email
traceid = args.traceid
visual_area = args.visual_area
alignment_type = args.alignment_type

# Create a (hopefully) unique prefix for the names of all jobs in this 
# particular run of the pipeline. This makes sure that runs can be
# identified unambiguously
piper = uuid.uuid4()
piper = str(piper)[0:4]
logdir = '%s_log' % visual_area
if not os.path.exists(logdir):
    os.mkdir(logdir)

# Remove old logs
old_logs = glob.glob(os.path.join(logdir, '*.err'))
old_logs.extend(glob.glob(os.path.join(logdir, '*.out')))
for r in old_logs:
    os.remove(r)

#####################################################################
#                          find XID files                           #
#####################################################################

# Note: the syntax a+=(b) adds b to the array a
# Open log lfile
sys.stdout = open('%s_info_%s.txt' % (visual_area, EXPERIMENT), 'w')


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


#meta_list = [('JC120', '20191111', 'FOV1_zoom2p0x', 'rfs10', 'traces001')] #,
#             ('JC083', '20190510', 'FOV1_zoom2p0x', 'rfs', 'traces001'),
#             ('JC083', '20190508', 'FOV1_zoom2p0x', 'rfs', 'traces001'),
#             ('JC084', '20190525', 'FOV1_zoom2p0x', 'rfs', 'traces001')]
#

dsets = load_metadata(EXPERIMENT, visual_area=visual_area)
#incl = ['20190522_JC084_fov1', '20190525_JC084_fov']
#dsets = dsets[dsets['datakey'].isin(incl)]

if len(dsets)==0:
    fatal("NO FOVs found.")

info("Found %i [%s] datasets to process." % (len(dsets), EXPERIMENT))

#meta_list=meta_list[0:3]

################################################################################
#                               run the pipeline                               #
################################################################################
 

# STEP1: TIFF PROCESSING. All PIDs will be processed in parallel since they don't
#        have any dependencies

jobids = [] # {}

for (visual_area, datakey), g in dsets.groupby(['visual_area', 'datakey']): 
    print("[%s]: %s" % (visual_area, datakey))
    session, animalid, fovn = datakey.split('_')
    fov = 'FOV%i_zoom2p0x' % int(fovn[3:])
    mtag = '-'.join([datakey, visual_area])
    cmd = "sbatch --job-name={PROCID}.{MTAG} \
            -o '{LOGDIR}/{PROCID}.{MTAG}.out' \
            -e '{LOGDIR}/{PROCID}.{MTAG}.err' \
            /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/slurm/parse_face_data.sbatch \
            {ANIMALID} {SESSION} {FOV} {EXP} {TRACEID} {ALIGN}".format(
            PROCID=piper, MTAG=mtag, LOGDIR=logdir,
            ANIMALID=animalid,
            SESSION=session, FOV=fov, EXP=EXPERIMENT, TRACEID=traceid, ALIGN=alignment_type) 
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




