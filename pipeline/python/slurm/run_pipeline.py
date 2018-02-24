#!/usr/local/bin/python

import argparse
import uuid
import sys
import os
import commands

parser = argparse.ArgumentParser(
    description = '''Look for XID files in session directory.\nFor PID files, run tiff-processing and evaluate.\nFor RID files, wait for PIDs to finish (if applicable) and then extract ROIs and evaluate.\n''',
    epilog = '''AUTHOR:\n\tJuliana Rhee''')
parser.add_argument('-i', '--animalid', dest='animalid', action='store', default='', help='Animal ID')
parser.add_argument('-S', '--session', dest='session', action='store',  default='', help='session (fmt: YYYYMMDD)')
parser.add_argument('-A', '--acquisition', dest='acquisition', action='store',  default='', help='acquisition folder')
parser.add_argument('-R', '--run', dest='run', action='store',  default='', help='run folder')
parser.add_argument('-p', '--pid', dest='pidhash', action='store',  default='', help='6-char PID hash')

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

sys.stdout = open('log/pidinfo.txt', 'w')

#####################################################################
#                          find XID files                           #
#####################################################################

# Get PID(s) based on name
# Note: the syntax a+=(b) adds b to the array a
ROOTDIR = '/n/coxfs01/2p-data'
ANIMALID = args.animalid
SESSION = args.session
PIDHASH = args.pidhash
ACQUISITION = args.acquisition
RUN = args.run
if len(PIDHASH) == 0 and len(ACQUISITION) == 0 and len(RUN) == 0:
    PIDDIR = os.path.join(ROOTDIR, ANIMALID, SESSION, 'tmp_spids')
    if not os.path.exists(PIDDIR):
        fatal("Unknown ANIMALID [%s] or SESSION [%s]... exiting!" % (ANIMALID, SESSION))
else:
    if (len(ACQUISITION) and len(RUN)) > 0:
        PIDDIR = os.path.join(ROOTDIR, ANIMALID, SESSION, ACQUISITION, RUN, 'processed', 'tmp_pids')
        if not os.path.exists(PIDDIR):
            fatal("Specified PID path does not exist!\nACQ: %s | RUN: %s" % (ACQUISITION, SESSION))
        else:
            if len(PIDHASH) > 0:
                PIDPATH = os.path.join(PIDDIR, 'tmp_pid_%s.json' % PIDHASH)
                if not os.path.exists(PIDPATH):
                    fatal("Specified PID PATH does not exist!\nUnknown dest: %s" % PIDPATH)
                else:
                    info("Specified PID to process: %s" % PIDHASH)
            else:
                info("Checking PID dir: %s" % PIDDIR)   
    elif len(PIDHASH) > 0:
       fatal("ACQUISITION [%s] or RUN [%s] not specified... exiting!" % (ACQUISITION, RUN))

pid_files = [os.path.join(PIDDIR, p) for p in os.listdir(PIDDIR) if p.endswith('json')]
eval_files = [os.path.join(os.path.split(p)[0], 'completed', os.path.split(p)[-1]) for p in pid_files]
# make sure there are pids
if len(pid_files) == 0:
     fatal("no PIDs found in dir %s" % PIDDIR)

info("Found %i PID files to process." % len(pid_files))
for pi,pid_path in enumerate(pid_files):
    info("    %i, %s" % (pi, pid_path))

################################################################################
#                               run the pipeline                               #
################################################################################
 

# STEP1: TIFF PROCESSING. All PIDs will be processed in parallel since they don't
#        have any dependencies

pid_jobids = {}
for pid_file in pid_files:
    phash = os.path.splitext(os.path.split(pid_file)[-1])[0].split('_')[-1]
    cmd = "sbatch --job-name={PROCID}.processpid.{PHASH} \
		-o 'log/{PROCID}.processpid.{PHASH}.out' \
		-e 'log/{PROCID}.processpid.{PHASH}.err' \
		/n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/slurm/preprocessing/process_pid_file.sbatch \
		{FILEPATH}".format(PROCID=piper, PHASH=phash, FILEPATH=pid_file)
    #info("Submitting PROCESSPID job with CMD:\n%s" % cmd)
    status, joboutput = commands.getstatusoutput(cmd)
    jobnum = joboutput.split(' ')[-1]
    pid_jobids[phash] = jobnum
    info("PROCESSING jobids [%s]: %s" % (phash, jobnum))


# STEP2: MC EVALUATION. Each mceval call will start when the corresponding processing 
#        call finishes sucessfully. Note, if STEP1 fails, all jobs depending on
#        it will remain in the queue and need to be canceled explicitly.
#        An alternative would be to use 'afterany' and make each job check for
#        the successful execution of the prerequisites.

info("Found %i PID files to evaluate." % len(eval_files))
for pi,eval_path in enumerate(eval_files):
    info("    %i, %s" % (pi, eval_path))


eval_jobids = {}
for eval_file in eval_files:
    phash = os.path.splitext(os.path.split(eval_file)[-1])[0].split('_')[-1]
    jobdep = pid_jobids[phash]
    cmd = "sbatch --job-name={PROCID}.mceval.{PHASH} \
                  -o 'log/{PROCID}.mceval.{PHASH}.out' \
                  -e 'log/{PROCID}.mceval.{PHASH}.err' \
                  --depend=afterok:{JOBDEP} \
                  /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/slurm/evaluation/evaluate_pid_file.sbatch \
                  {FILEPATH}".format(PROCID=piper, PHASH=phash, FILEPATH=eval_file, JOBDEP=jobdep)
    #info("Submitting MCEVAL job with CMD:\n%s" % cmd)
    status, joboutput = commands.getstatusoutput(cmd)
    jobnum = joboutput.split(' ')[-1]
    eval_jobids[phash] = jobnum
    info("MCEVAL calling jobids [%s]: %s" % (phash, jobnum))


# STEP3: ROI EXTRACTION: Each nmf call will start when the corresponding alignments
#        finish sucessfully. Note, if STEP1 fails, all jobs depending on
#        it will remain in the queue and need to be canceled explicitly.
#        An alternative would be to use 'afterany' and make each job check for
#        the successful execution of the prerequisites.


