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
parser.add_argument('-E', '--exp', dest='experiment_type', action='store', default='rfs', help='Experiment type (e.g., rfs')
parser.add_argument('-e', '--email', dest='email', action='store', default='rhee@g.harvard.edu', help='Email to send log files')

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


# Create a (hopefully) unique prefix for the names of all jobs in this 
# particular run of the pipeline. This makes sure that runs can be
# identified unambiguously
piper = uuid.uuid4()
piper = str(piper)[0:8]
if not os.path.exists('log'):
    os.mkdir('log')

old_logs = glob.glob(os.path.join('log', '*.err'))
old_logs.extend(glob.glob(os.path.join('log', '*.out')))
for r in old_logs:
    os.remove(r)

#####################################################################
#                          find XID files                           #
#####################################################################

# Get PID(s) based on name
# Note: the syntax a+=(b) adds b to the array a
ROOTDIR = '/n/coxfs01/2p-data'
FOV = args.fov_type
EXP = args.experiment_type
email = args.email

# Open log lfile
sys.stdout = open('log/loginfo_%s.txt' % EXP, 'w')

def load_metadata(rootdir='/n/coxfs01/2p-data', aggregate_dir='/n/coxfs01/julianarhee/aggregate-visual-areas',
                    experiment='', traceid='traces001'):

    sdata_fpath = os.path.join(aggregate_dir, 'dataset_info.pkl')
    with open(sdata_fpath, 'rb') as f:
        sdata = pkl.load(f)
   
    meta_list=[]
    for (animalid, session, fov), g in sdata.groupby(['animalid', 'session', 'fov']):
        exp_list = [e for e in g['experiment'].values if experiment in e]
        for e in exp_list:
            if experiment in e:
                #rfname = 'gratings' if int(session)<20190511 else e
                meta_list.append(tuple([animalid, session, fov, e, traceid]))    
                existing_dirs = glob.glob(os.path.join(rootdir, animalid, session, fov, '*%s*' % e,
                                                'traces', '%s*' % traceid, 'receptive_fields', 
                                                'fit-2dgaus_dff*'))
                for edir in existing_dirs:
                    tmp_od = os.path.split(edir)[0]
                    tmp_rt = os.path.split(edir)[-1]
                    old_dir = os.path.join(tmp_od, '_%s' % tmp_rt)
                    if os.path.exists(old_dir):
                        continue
                    else:
                        os.rename(edir, old_dir)    
                    info('renamed: %s' % old_dir)
    return meta_list


#meta_list = [('JC120', '20191111', 'FOV1_zoom2p0x', 'rfs10', 'traces001')] #,
#             ('JC083', '20190510', 'FOV1_zoom2p0x', 'rfs', 'traces001'),
#             ('JC083', '20190508', 'FOV1_zoom2p0x', 'rfs', 'traces001'),
#             ('JC084', '20190525', 'FOV1_zoom2p0x', 'rfs', 'traces001')]
#


meta_list = load_metadata(experiment=EXP)

if len(meta_list)==0:
    fatal("NO FOVs found.")


info("Found %i [%s] datasets to process." % (len(meta_list), EXP))
#for mi, meta in enumerate(meta_list):
#    info("... %s" % '|'.join(meta))

################################################################################
#                               run the pipeline                               #
################################################################################
 

# STEP1: TIFF PROCESSING. All PIDs will be processed in parallel since they don't
#        have any dependencies

jobids = [] # {}
for (animalid, session, fov, experiment, traceid) in meta_list:
    mtag = '-'.join([session, animalid, fov, experiment])
    cmd = "sbatch --job-name={PROCID}.rfs.{MTAG} \
		-o 'log/{PROCID}.rfs.{MTAG}.out' \
		-e 'log/{PROCID}.rfs.{MTAG}.err' \
		/n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/slurm/process_rfs.sbatch \
		{ANIMALID} {SESSION} {FOV} {EXP} {TRACEID}".format(
                        PROCID=piper, MTAG=mtag, ANIMALID=animalid,
                        SESSION=session, FOV=fov, EXP=experiment, TRACEID=traceid) #pid_file)
    #info("Submitting PROCESSPID job with CMD:\n%s" % cmd)
    status, joboutput = commands.getstatusoutput(cmd)
    jobnum = joboutput.split(' ')[-1]
    jobids.append(jobnum)
    info("[%s]: %s" % (jobnum, mtag))


#info("****done!****")

for jobdep in jobids:
    print(jobdep)
    cmd = "sbatch --job-name={JOBDEP}.checkstatus \
		-o 'log/checkstatus.rfs.{JOBDEP}.out' \
		-e 'log/checkstatus.rfs.{JOBDEP}.err' \
                  --depend=afternotok:{JOBDEP} \
                  /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/slurm/checkstatus.sbatch \
                  {JOBDEP} {EMAIL}".format(JOBDEP=jobdep, EMAIL=email)
    #info("Submitting MCEVAL job with CMD:\n%s" % cmd)
    status, joboutput = commands.getstatusoutput(cmd)
    jobnum = joboutput.split(' ')[-1]
    #eval_jobids[phash] = jobnum
    #info("MCEVAL calling jobids [%s]: %s" % (phash, jobnum))
    print("... checking: %s (%s)" % (jobdep, jobnum))




