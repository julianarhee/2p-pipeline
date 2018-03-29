#!/usr/local/bin/python

import argparse
import uuid
import sys
import os
import commands
import traceback
import json

parser = argparse.ArgumentParser(
    description = '''Look for XID files in session directory.\nFor PID files, run tiff-processing and evaluate.\nFor RID files, wait for PIDs to finish (if applicable) and then extract ROIs and evaluate.\n''',
    epilog = '''AUTHOR:\n\tJuliana Rhee''')
parser.add_argument('-i', '--animalid', dest='animalid', action='store', default='', help='Animal ID')
parser.add_argument('-S', '--session', dest='session', action='store',  default='', help='session (fmt: YYYYMMDD)')
#parser.add_argument('-A', '--acquisition', dest='acquisition', action='store',  default='', help='acquisition folder')
#parser.add_argument('-R', '--run', dest='run', action='store',  default='', help='run folder')
parser.add_argument('-r', '--rid', dest='ridhash', action='store',  default='', help='6-char RID hash')
parser.add_argument('-b', '--start-file', dest='first_tiff', action='store',  default=1, help='File num of first tiff to process [default: 1]')
parser.add_argument('-e', '--end-file', dest='last_tiff', action='store',  default=None, help='File num of last tiff to process [default: all tiffs found in RID SRC dir]')

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

sys.stdout = open('log/coreg_jobsummary.txt', 'w')

#####################################################################
#                          find XID files                           #
#####################################################################

# Get PID(s) based on name
# Note: the syntax a+=(b) adds b to the array a
ROOTDIR = '/n/coxfs01/2p-data'
ANIMALID = args.animalid
SESSION = args.session
RIDHASH = args.ridhash
#ACQUISITION = args.acquisition
#RUN = args.run
RIDDIR = os.path.join(ROOTDIR, ANIMALID, SESSION, 'ROIs', 'tmp_rids')

first_tiff = int(args.first_tiff)
last_tiff = args.last_tiff

if not os.path.exists(RIDDIR):
    fatal("Unknown ANIMALID [%s] or SESSION [%s]... exiting!" % (ANIMALID, SESSION))
RIDPATH = os.path.join(RIDDIR, 'tmp_rid_%s.json' % RIDHASH)
if not os.path.exists(RIDPATH):
    fatal("Specified RID PATH does not exist!\nUnknown dest: %s" % RIDPATH)

info("Processing RID file: %s" % RIDPATH)

###############################################################################
#                               run the pipeline                               #
################################################################################


# STEP3: ROI EXTRACTION: Each nmf call will start when the corresponding alignments
#        finish sucessfully. Note, if STEP1 fails, all jobs depending on
#        it will remain in the queue and need to be canceled explicitly.
#        An alternative would be to use 'afterany' and make each job check for
#        the successful execution of the prerequisites.
coregister = True
if coregister is True:
    try:
        with open(RIDPATH, 'r') as f:
            RID = json.load(f)
        ntiffs = len([t for t in os.listdir(RID['SRC']) if t.endswith('tif')])
    except Exception as e:
        error(traceback.print_exc())
        fatal("Unable to load RID.")
    print "NTIFFS: %i" % ntiffs

    if last_tiff is None:
        last_tiff = ntiffs
    else:
        last_tiff = int(last_tiff)

    rid_jobids = {}
#    for rid_filenum in range(ntiffs):
    rhash = os.path.splitext(os.path.split(RIDPATH)[-1])[0].split('_')[-1]
    cmd = "sbatch --array={FIRST}-{LAST} \
                --job-name={PROCID}.coreg.{RHASH} \
    		/n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/slurm/rois/coregister_tiff.sbatch \
    		{FILEPATH}".format(PROCID=piper, RHASH=rhash, FILEPATH=RIDPATH, FIRST=first_tiff, LAST=last_tiff)
    #info("Submitting COREGISTRATION job with CMD:\n%s" % cmd)
    status, joboutput = commands.getstatusoutput(cmd)
    #print "JOB INFO:"
    #print joboutput
    jobnum = joboutput.split(' ')[-1]
    info("COREG jobids: %s" % jobnum)

    cmd = "sbatch --job-name={PROCID}.collate.{RHASH} \
                  --depend=afterok:{JOBDEP} \
                  /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/slurm/rois/collate_coreg_results.sbatch {FILEPATH}".format(PROCID=piper, RHASH=rhash, FILEPATH=RIDPATH, JOBDEP=jobnum)
    #info("Submitted COREG-COLLATE job with CMD:\n%s" % cmd)
    status, joboutput = commands.getstatusoutput(cmd)
    jobnum = joboutput.split(' ')[-1]
    info("COREG-COLLATE calling jobids: %s" % jobnum)


#    eval_jobids = {}
#    for pid_file in pid_files:
#        phash = os.path.splitext(os.path.split(pid_file)[-1])[0].split('_')[-1]
#        jobdep = pid_jobids[pid_file]
#        cmd = "sbatch --job-name={PROCID}.mceval.{PHASH} \
#                      -o 'log/{PROCID}.mceval.{PHASH}.out' \
#                      -e 'log/{PROCID}.mceval.{PHASH}.err' \
#                      --depend=afterok:{JOBDEP} \
#                      /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/slurm/evaluate_pid_file.sbatch \
#                      {FILEPATH}".format(PROCID=piper, PHASH=phash, FILEPATH=pid_file, JOBDEP=jobdep)
#        info("Submitting MCEVAL job with CMD:\n%s" % cmd)
#        status, joboutput = commands.getstatusoutput(cmd)
#        jobnum = joboutput.split(' ')[-1]
#        eval_jobids[pid_file] = jobnum
#        info("MCEVAL calling jobids: %s" % jobnum)
#
