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
parser.add_argument('-t', '--traceid', dest='traceid', action='store', default=None, help='Traceid to use as reference for selecting retino analysis')
parser.add_argument('-v', '--visual_area', dest='visual_area', action='store', default=None, help='Add to only process datasets for a given visual area')


parser.add_argument('-p', '--pre', dest='iti_pre', action='store', default=1.0, help='PRE stimulus dur (sec, default=1)')
parser.add_argument('-P', '--post', dest='iti_post', action='store', default=1.0, help='POST stimulus dur (sec, default=1)')

parser.add_argument( '--all', dest='run_all', action='store_true', default=False, help='Run on all specified datasets, ignoring extraction_info')



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
iti_pre = float(args.iti_pre)
iti_post = float(args.iti_post)
visual_area = args.visual_area
run_all = args.run_all

# Create a (hopefully) unique prefix for the names of all jobs in this 
# particular run of the pipeline. This makes sure that runs can be
# identified unambiguously
piper = uuid.uuid4()
piper = str(piper)[0:8]
logdir='LOG__%s_%s' % (EXP, visual_area)
if not os.path.exists(logdir):
    os.mkdir(logdir)

old_logs = glob.glob(os.path.join(logdir, '*.err'))
old_logs.extend(glob.glob(os.path.join(logdir, '*.out')))
old_logs.extend(glob.glob(os.path.join(logdir, '*.txt')))

for r in old_logs:
    os.remove(r)

#####################################################################
#                          find XID files                           #
#####################################################################
# Get PID(s) based on name
# Note: the syntax a+=(b) adds b to the array a

# Open log lfile
sys.stdout = open('%s/INFO_%s.txt' % (logdir, EXP), 'w')


def get_trial_alignment(animalid, session, fovnum, curr_exp, traceid='traces001',
        rootdir='/n/coxfs01/2p-data'):
    extraction_files = glob.glob(os.path.join(rootdir, animalid, session, 
                            'FOV%i*' % fovnum, '*%s*' % curr_exp, 
                            'traces', '%s*' % traceid, 'event_alignment.json'))
    assert len(extraction_files) > 0, "No extraction info found..."

    for i, ifile in enumerate(extraction_files):
        with open(ifile, 'r') as f:
            info = json.load(f)
        if i==0:
            infodict = dict((k, [v]) for k, v in info.items() if isnumber(v)) 
        else:
            for k, v in info.items():
                if isnumber(v):
                    infodict[k].append(v) 
    for k, v in infodict.items():
        nvs = np.unique(v)
        assert len(nvs)==1, "more than 1 value found: (%s, %s)" % (k, str(nvs))
        infodict[k] = np.unique(v)[0]
    return infodict


def load_metadata(experiment, iti_pre=1.0, iti_post=1., 
                  run_datakeys=[],
                  visual_area=None,run_all=False,
                  rootdir='/n/coxfs01/2p-data', 
                  aggregate_dir='/n/coxfs01/julianarhee/aggregate-visual-areas',
                  traceid='traces001'):
    from pipeline.python.classifications import aggregate_data_stats as aggr
    sdata = aggr.get_aggregate_info(traceid=traceid) #, fov_type=fov_type, state=state)

    dsets = sdata[sdata['experiment']==experiment] 
    if visual_area is not None:
        dsets = dsets[dsets['visual_area']==visual_area]

    meta_list=[]
    for (animalid, session, fovnum, datakey), g in dsets.groupby(['animalid', 'session', 'fovnum', 'datakey']):
        fov = g['fov'].values[0]
        
        if run_all:
            meta_list.append(tuple([animalid, session, fov, experiment, traceid]))
            continue

        # Get alignment info
        alignment_info = aggr.get_trial_alignment(animalid, session, 
                                                fovnum, experiment, traceid=traceid)
        if alignment_info is None:
            print(session, animalid, fovnum)
            meta_list.append(tuple([animalid, session, fov, experiment, traceid]))    
        elif alignment_info==-1:
            print("REALIGN: %s" % datakey)
        elif datakey in run_datakeys:
            meta_list.append(tuple([animalid, session, fov, experiment, traceid]))    
           
        else: 
            curr_iti_pre = float(alignment_info['iti_pre'])
            curr_iti_post = float(alignment_info['iti_post'])    
            if (curr_iti_pre != iti_pre) or (curr_iti_post != iti_post):
                print(session, animalid, fovnum)
                meta_list.append(tuple([animalid, session, fov, experiment, traceid]))    

    return meta_list


#meta_list = [('JC084', '20190522', 'FOV1_zoom2p0x', 'blobs', 'traces001')] #,
#             ('JC083', '20190510', 'FOV1_zoom2p0x', 'rfs', 'traces001'),
#             ('JC083', '20190508', 'FOV1_zoom2p0x', 'rfs', 'traces001'),
#             ('JC084', '20190525', 'FOV1_zoom2p0x', 'rfs', 'traces001')]
#

meta_list = load_metadata(EXP, visual_area=visual_area, run_all=run_all)

if len(meta_list)==0:
    fatal("NO FOVs found.")

info("Found %i [%s] datasets to process (pre/post=%i/%i)." % (len(meta_list), EXP, iti_pre, iti_post))

#meta_list=meta_list[0:3]

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
            /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/slurm/align_trials.sbatch \
            {ANIMALID} {SESSION} {FOV} {EXP} {TRACEID} {PRE} {POST}".format(
            LOG=logdir,
            PROCID=piper, MTAG=mtag, ANIMALID=animalid,
            SESSION=session, FOV=fov, EXP=experiment, TRACEID=traceid, PRE=iti_pre, POST=iti_post) 
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
                  {JOBDEP} {EMAIL}".format(JOBDEP=jobdep, EMAIL=email, EXP=EXP)
    #info("Submitting MCEVAL job with CMD:\n%s" % cmd)
    status, joboutput = commands.getstatusoutput(cmd)
    jobnum = joboutput.split(' ')[-1]
    #eval_jobids[phash] = jobnum
    #info("MCEVAL calling jobids [%s]: %s" % (phash, jobnum))
    print("... checking: %s (%s)" % (jobdep, jobnum))




