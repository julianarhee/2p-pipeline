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
parser.add_argument('-E', '--exp', dest='experiment_type', action='store', default='retino', help='Experiment type (e.g., rfs')
parser.add_argument('-e', '--email', dest='email', action='store', default='rhee@g.harvard.edu', help='Email to send log files')
parser.add_argument('-t', '--traceid', dest='traceid', action='store', default='traces001', help='Traceid to use as reference for selecting retino analysis')

parser.add_argument('-p', '--pass-crit', dest='pass_criterion', action='store', default='pixels', help='Criterion to use for selecting ROIs for gradient calculation (default: npmean, use only if -E retino. Choices: all, either, any, npmean, pixels)')

parser.add_argument('-V', '--area', dest='visual_area', action='store', default=None, help='Set to run analysis on 1 visual area only')



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
EXP = args.experiment_type
email = args.email
traceid = args.traceid
pass_criterion = args.pass_criterion
visual_area = args.visual_area

# Create a (hopefully) unique prefix for the names of all jobs in this 
# particular run of the pipeline. This makes sure that runs can be
# identified unambiguously
piper = uuid.uuid4()
piper = str(piper)[0:4]

logdir = 'LOG__%s_%s' % (EXP, visual_area)
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
sys.stdout = open('%s/INFO_%s_%s.txt' % (logdir, piper, EXP), 'w')

def get_retino_metadata(experiment='retino', animalids=None,
                        roi_type='manual2D_circle', traceid=None,
                        rootdir='/n/coxfs01/2p-data', visual_area=None,
                        aggregate_dir='/n/coxfs01/julianarhee/aggregate-visual-areas'):

    sdata_fpath = os.path.join(aggregate_dir, 'dataset_info.pkl')
    with open(sdata_fpath, 'rb') as f:
        sdata = pkl.load(f)
  
    if visual_area is not None:
        edata = sdata[sdata['visual_area']==visual_area].copy()
    else:
        edata = sdata.copy()
 
    meta_list=[]
    for (animalid, session, fov), g in edata.groupby(['animalid', 'session', 'fov']):
        if animalids is not None:
            if animalid not in animalids:
                continue
        exp_list = [e for e in g['experiment'].values if experiment in e] 
        if len(exp_list)==0:
            info('skipping, no retino (%s, %s, %s)' % (animalid, session, fov)) 
        retino_dirs = glob.glob(os.path.join(rootdir, animalid, session, fov, '%s*' % experiment,
                                'retino_analysis'))
        # get analysis ids for non-pixel
        for retino_dir in retino_dirs:
            retino_run = os.path.split(os.path.split(retino_dir)[0])[-1]
            if traceid is None:
                rid_fpath = glob.glob(os.path.join(retino_dir, 'analysisids_*.json'))[0]
                with open(rid_fpath, 'r') as f:
                    retids = json.load(f)
                traceids = [r for r, res in retids.items() if res['PARAMS']['roi_type']==roi_type] 
                for traceid in traceids: 
                    meta_list.append(tuple([animalid, session, fov, retino_run, traceid]))
            else:
                meta_list.append(tuple([animalid, session, fov, retino_run, traceid]))

    return meta_list


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



#if EXP=='retino':
meta_list = get_retino_metadata(traceid=traceid, visual_area=visual_area)
#meta_list = [k for k in meta_list if k[0]=='JC084']

#else:
#    meta_list = load_metadata(experiment=EXP)

if len(meta_list)==0:
    fatal("NO FOVs found.")

info("Found %i [%s] datasets to process (FILTER: %s)." % (len(meta_list), EXP, pass_criterion))
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
    cmd = "sbatch --job-name={PROCID}.{EXP}.{MTAG} \
		-o '{LOG}/{PROCID}.{EXP}.{MTAG}.out' \
		-e '{LOG}/{PROCID}.{EXP}.{MTAG}.err' \
		/n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/slurm/retino_gradients.sbatch \
		{ANIMALID} {SESSION} {FOV} {EXP} {TRACEID} {FILTER}".format(
                        LOG=logdir, PROCID=piper, MTAG=mtag, ANIMALID=animalid,
                        SESSION=session, FOV=fov, EXP=experiment, TRACEID=traceid, FILTER=pass_criterion) #pid_file)
    #info("Submitting PROCESSPID job with CMD:\n%s" % cmd)
    status, joboutput = commands.getstatusoutput(cmd)
    jobnum = joboutput.split(' ')[-1]
    jobids.append(jobnum)
    info("[%s]: %s" % (jobnum, mtag))


#info("****done!****")

for jobdep in jobids:
    print(jobdep)
    cmd = "sbatch --job-name={JOBDEP}.checkstatus \
		-o '{LOG}/checkstatus.{EXP}.{JOBDEP}.out' \
		-e '{LOG}/checkstatus.{EXP}.{JOBDEP}.err' \
                  --depend=afternotok:{JOBDEP} \
                  /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/slurm/checkstatus.sbatch \
                  {JOBDEP} {EMAIL}".format(LOG=logdir, JOBDEP=jobdep, EMAIL=email, EXP=EXP)
    #info("Submitting MCEVAL job with CMD:\n%s" % cmd)
    status, joboutput = commands.getstatusoutput(cmd)
    jobnum = joboutput.split(' ')[-1]
    #eval_jobids[phash] = jobnum
    #info("MCEVAL calling jobids [%s]: %s" % (phash, jobnum))
    print("... checking: %s (%s)" % (jobdep, jobnum))




