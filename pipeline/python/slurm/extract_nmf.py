#!/usr/local/bin/python

import argparse
import uuid
import sys
import os
import commands
import traceback
import json
import shutil

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
parser.add_argument('--snr', dest='min_snr', action='store',  default=1.5, help='Min SNR value for ROI evaluation [default: 1.5 (nmf extr. uses 2.0 default)]')
parser.add_argument('--rcorr', dest='rval_thr', action='store',  default=0.8, help='Min spatial corr threshold for ROI evaluation [default: 0.6 (nmf extr. uses 0.8 default)]')
parser.add_argument('--mmap', dest='mmap_tiffs', action='store_true', default=False, help='Set flag to memmap tifs')
parser.add_argument('--evaluate', dest='evaluate_rois', action='store_true', default=False, help='Set flag to do low-threshold evaluation for NMF rois')


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

sys.stdout = open('log/nmf_jobsummary.txt', 'w')

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

min_snr = float(args.min_snr)
rval_thr = float(args.rval_thr)
mmap_tiffs = args.mmap_tiffs
evaluate_rois = args.evaluate_rois

if not os.path.exists(RIDDIR):
    fatal("Unknown ANIMALID [%s] or SESSION [%s]... exiting!" % (ANIMALID, SESSION))

rid_files = [os.path.join(RIDDIR, r) for r in os.listdir(RIDDIR) if r.endswith('json')]
if len(RIDHASH) > 0:
    rid_files  = [r for r in rid_files if RIDHASH in r]

#eval_files = [os.path.join(os.path.split(r)[0], 'completed', os.path.split(r)[-1]) for r in rid_files]
#info("Processing %i RID files in dir: %s" % (RIDDIR, len(rid_files))

# make sure there are rids
if len(rid_files) == 0:
     fatal("no RIDs found in dir %s" % RIDDIR)

info("Found %i RID files to process." % len(rid_files))
for ri,rid_path in enumerate(rid_files):
    info("-----%i, %s" % (ri, rid_path))


###############################################################################
#                               run the pipeline                               #
################################################################################
completed_dir = os.path.join(ROOTDIR, ANIMALID, SESSION, 'ROIs', 'tmp_rids', 'completed')
error_dir = os.path.join(ROOTDIR, ANIMALID, SESSION, 'ROIs', 'tmp_rids', 'error')
if not os.path.exists(completed_dir):
    os.makedirs(completed_dir)
if not os.path.exists(error_dir):
    os.makedirs(error_dir)


# STEP1: MEMMAP TIFFS. All RIDs will be processed in parallel since they don't
#        have any dependencies

if mmap_tiffs is True:
    info("*******************MEMMAPPING********************")
    mm_jobids = {}
    for rid_file in rid_files:
        rhash = os.path.splitext(os.path.split(rid_file)[-1])[0].split('_')[-1]
        #info("----Creating job for RID:  %s." % rhash)
        cmd = "sbatch --job-name={PROCID}.mmap.{RHASH} \
    		-o 'log/{PROCID}.mmap.{RHASH}.out' \
    		-e 'log/{PROCID}.mmap.{RHASH}.err' \
    		/n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/slurm/rois/mmap_tiffs_rid_file.sbatch \
    		{FILEPATH}".format(PROCID=piper, RHASH=rhash, FILEPATH=rid_file)
        #info("Submitting MEMMAP job with CMD:\n%s" % cmd)
        status, joboutput = commands.getstatusoutput(cmd)
        jobnum = joboutput.split(' ')[-1]
        mm_jobids[rhash] = jobnum
        info("MEMMAP jobids [%s]: %s" % (rhash, jobnum))
    
    
# STEP2: NMF EXTRACTION. Each nmf call will start when the corresponding memmapping 
#        call finishes sucessfully. Note, if STEP1 fails, all jobs depending on
#        it will remain in the queue and need to be canceled explicitly.
#        An alternative would be to use 'afterany' and make each job check for
#        the successful execution of the prerequisites.
do_nmf = True 
if do_nmf is True:
    info("*********************NMF*************************") 
    rid_jobids = {}
    for rid_file in rid_files:
        rhash = os.path.splitext(os.path.split(rid_file)[-1])[0].split('_')[-1]
        #info("*****Creating NMF extraction job for RID %s." % rhash)
        try:
            with open(rid_file, 'r') as f:
                RID = json.load(f)
            ntiffs = len([t for t in os.listdir(RID['SRC']) if t.endswith('tif')])
        except Exception as e:
            error(traceback.print_exc())
            fatal("Unable to load RID: %s." % rid_file)
        print "NTIFFS [%s]: %i" % (rhash, ntiffs)
        if last_tiff is None:
            last_tiff = ntiffs
        else:
            last_tiff = int(last_tiff)
        jobdep = mm_jobids[rhash]
        cmd = "sbatch --array={FIRST}-{LAST} \
                       --job-name={PROCID}.nmf.{RHASH} \
                        --depend=afterok:{JOBDEP} \
                        /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/slurm/rois/nmf_tiff.sbatch \
                          {FILEPATH}".format(PROCID=piper, RHASH=rhash, FILEPATH=rid_file, FIRST=first_tiff, LAST=last_tiff, JOBDEP=jobdep)
        #info("Submitting NMF job with CMD:\n%s" % cmd)
        status, joboutput = commands.getstatusoutput(cmd)
        jobnum = joboutput.split(' ')[-1]
        rid_jobids[rhash] = jobnum
        info("NMF calling jobids [%s]: %s" % (rhash, jobnum))
            

# STEP3: ROI EVALUATION: Each evaluation call will start when the corresponding NMF step 
#        finishes sucessfully. Note, if STEP2 fails, all jobs depending on
#        it will remain in the queue and need to be canceled explicitly.
#        An alternative would be to use 'afterany' and make each job check for
#        the successful execution of the prerequisites.

if evaluate_rois is True:
    info("******************EVALUATION*********************") 
    info('PARAMS:  min snr %.2f, rval thr %.2f' % (min_snr, rval_thr))
    eval_jobids = {}
    for rid_file in rid_files:
        rhash = os.path.splitext(os.path.split(rid_file)[-1])[0].split('_')[-1]
        jobdep = rid_jobids[rhash]
        cmd = "sbatch --job-name={PROCID}.coreg.{RHASH} \
                --depend=afterok:{JOBDEP} \
    		/n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/slurm/evaluation/eval_rois.sbatch \
    		{FILEPATH} {SNR} {RCORR}".format(PROCID=piper, RHASH=rhash, FILEPATH=rid_file, FIRST=first_tiff, LAST=last_tiff, JOBDEP=jobdep, SNR=min_snr, RCORR=rval_thr)
        #info("Submitting ROI EVAL job with CMD:\n%s" % cmd)
        status, joboutput = commands.getstatusoutput(cmd)
        #print "JOB INFO:"
        #print joboutput
        jobnum = joboutput.split(' ')[-1]
        eval_jobids[rhash] = jobnum
        info("ROI EVAL jobids [%s]: %s" % (rhash, jobnum))

 
