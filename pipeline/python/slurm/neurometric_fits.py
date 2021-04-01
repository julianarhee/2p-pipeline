#!/usr/local/bin/python

import argparse
import uuid
import sys
import os
#import commands
import subprocess
import json
import glob
import pandas as pd
#import cPickle as pkl
import dill as pkl
import numpy as np

parser = argparse.ArgumentParser(
    description = '''Look for XID files in session directory.\nFor PID files, run tiff-processing and evaluate.\nFor RID files, wait for PIDs to finish (if applicable) and then extract ROIs and evaluate.\n''',
    epilog = '''AUTHOR:\n\tJuliana Rhee''')

parser.add_argument('-E', '--exp', dest='experiment_type', action='store', default='blobs', help='Experiment type (default: blobs)')
parser.add_argument('-t', '--traceid', dest='traceid', action='store', default='traces001', help='Traceid to use as reference for selecting retino analysis (default: traces001)')


parser.add_argument('-e', '--email', dest='email', action='store', default='rhee@g.harvard.edu', help='Email to send log files')

parser.add_argument('-R', '--resp-test', dest='responsive_test', action='store', default='ROC', help='Responsive test (default=ROC, options: ROC, nstds, None)')

parser.add_argument('-p', '--param', dest='param', action='store', default='morphlevel', help='X param to fit (default=morphlevel, options: morphlevel, morphstep)')

parser.add_argument('-f', '--func', dest='sigmoid', action='store', default='gauss', help='Sigmoid to fit (default=gauss, options: gumbel)')

parser.add_argument('--reverse', dest='allow_negative', action='store_false', default=True, help='Set flag to flip AUC, no negative sigmoids (default=fits neg_gauss if Eff==A)')

parser.add_argument('--pupil-auc', dest='do_pupilauc', action='store_true', default=False, help='Set flag to run AUC calculation across all pupil iterations)')

parser.add_argument('--pupil', dest='split_pupil', action='store_true', default=False, help='Set flag to fit curves from averaged AUC from split_pupil iters')

parser.add_argument('--iter', dest='by_iter', action='store_true', default=False, help='Set flag to fit curves to each ITER, from split_pupil iters')


parser.add_argument('-v', '--area', dest='visual_area', action='store', default=None, help='Visual area to process (default, all)')
parser.add_argument('-k', '--datakeys', nargs='*', dest='included_datakeys', action='append', help='Use like: -k DKEY DKEY DKEY')

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


# -----------------------------------------------------------------
# ARGS
# -----------------------------------------------------------------
ROOTDIR = '/n/coxfs01/2p-data'
experiment = args.experiment_type
email = args.email

visual_area = None if args.visual_area in ['None', None] else args.visual_area
traceid = args.traceid
responsive_test = args.responsive_test
responsive_thr = 10. if responsive_test=='nstds' else 0.05

sigmoid = args.sigmoid
param = args.param
allow_negative = args.allow_negative
do_pupilauc = args.do_pupilauc
split_pupil = args.split_pupil
by_iter = args.by_iter


# Create a (hopefully) unique prefix for the names of all jobs in this 
# particular run of the pipeline. This makes sure that runs can be
# identified unambiguously
piper = uuid.uuid4()
piper = str(piper)[0:4]

# Set LOGS
a_type = 'AUC' if do_pupilauc else 'FIT'

logdir = 'LOG_%s_%s_%s' % (a_type, str(visual_area), responsive_test) 
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
sys.stdout = open('%s/INFO_%s.txt' % (logdir, piper), 'w')

def load_metadata(experiment, visual_area=None, visual_areas=['V1', 'Lm', 'Li'],
                aggregate_dir='/n/coxfs01/julianarhee/aggregate-visual-areas'):

    sdata_fpath = os.path.join(aggregate_dir, 'dataset_info.pkl')
    with open(sdata_fpath, 'rb') as f:
        sdata = pkl.load(f, encoding='latin1')

    sdata_exp = sdata[(sdata['experiment']==experiment) 
                    & (sdata.visual_area.isin(visual_areas))] 
  
    if visual_area is not None:
        sdata_exp = sdata_exp[sdata_exp['visual_area']==visual_area]

    return sdata_exp


################################################################################
#                               run the pipeline                               #
################################################################################

jobids = [] # {}
# -----------------------------------------------------------------
# BY FOV 
# -----------------------------------------------------------------
dsets = load_metadata(experiment=experiment, visual_area=visual_area)
included_datakeys = args.included_datakeys
print("dkeys:", included_datakeys)

if included_datakeys is not None: #len(included_datakeys) > 0:
    dsets = dsets[dsets['datakey'].isin(included_datakeys[0])]
    
if len(dsets)==0:
    fatal("no fovs found.")
info("found %i [%s] datasets to process." % (len(dsets), experiment))

old_rats = ['JC061', 'JC067', 'JC070', 'JC073']
for (visual_area, datakey), g in dsets.groupby(['visual_area', 'datakey']):
    if do_pupilauc:
        mtag = 'psigni-pupilAUC-%s-%s_%s_%s' \
                    % (sigmoid, param, visual_area, datakey) 
        cmd = "sbatch --job-name={procid}.{mtag} \
                -o '{logdir}/{procid}.{mtag}.{rtest}.out' \
                -e '{logdir}/{procid}.{mtag}.{rtest}.err' \
        /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/slurm/neurometric_pupilauc.sbatch \
        {exp} {traceid} {rtest} {varea} {dkey} {param} {sigmoid} {allow_neg}"\
            .format(procid=piper, mtag=mtag, logdir=logdir,
            exp=experiment, traceid=traceid, 
            rtest=responsive_test, varea=visual_area, dkey=datakey, 
            param=param, sigmoid=sigmoid, allow_neg=allow_negative) 
    #
    elif split_pupil:
        if by_iter:
            mtag = 'psigni-iterpupil-%s-%s_%s_%s' \
                        % (sigmoid, param, visual_area, datakey) 
            cmd = "sbatch --job-name={procid}.{mtag} \
                    -o '{logdir}/{procid}.{mtag}.{rtest}.out' \
                    -e '{logdir}/{procid}.{mtag}.{rtest}.err' \
            /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/slurm/neurometric_iterpupil.sbatch \
            {exp} {traceid} {rtest} {varea} {dkey} {param} {sigmoid} {allow_neg}"\
                .format(procid=piper, mtag=mtag, logdir=logdir,
                exp=experiment, traceid=traceid, 
                rtest=responsive_test, varea=visual_area, dkey=datakey, 
                param=param, sigmoid=sigmoid, allow_neg=allow_negative) 
        else: 
            mtag = 'psigni-splitpupil-%s-%s_%s_%s' \
                        % (sigmoid, param, visual_area, datakey) 
            cmd = "sbatch --job-name={procid}.{mtag} \
                    -o '{logdir}/{procid}.{mtag}.{rtest}.out' \
                    -e '{logdir}/{procid}.{mtag}.{rtest}.err' \
            /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/slurm/neurometric_splitpupil.sbatch \
            {exp} {traceid} {rtest} {varea} {dkey} {param} {sigmoid} {allow_neg}"\
                .format(procid=piper, mtag=mtag, logdir=logdir,
                exp=experiment, traceid=traceid, 
                rtest=responsive_test, varea=visual_area, dkey=datakey, 
                param=param, sigmoid=sigmoid, allow_neg=allow_negative) 
     
    else:

        mtag = 'psigni-%s-%s_%s_%s' \
                    % (sigmoid, param, visual_area, datakey) 
        cmd = "sbatch --job-name={procid}.{mtag} \
                -o '{logdir}/{procid}.{mtag}.{rtest}.out' \
                -e '{logdir}/{procid}.{mtag}.{rtest}.err' \
        /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/slurm/neurometric_fits.sbatch \
        {exp} {traceid} {rtest} {varea} {dkey} {param} {sigmoid} {allow_neg}"\
            .format(procid=piper, mtag=mtag, logdir=logdir,
            exp=experiment, traceid=traceid, 
            rtest=responsive_test, varea=visual_area, dkey=datakey, 
            param=param, sigmoid=sigmoid, allow_neg=allow_negative) 
    #
    status, joboutput = subprocess.getstatusoutput(cmd)
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
    status, joboutput = subprocess.getstatusoutput(cmd)
    jobnum = joboutput.split(' ')[-1]
    #eval_jobids[phash] = jobnum
    #info("MCEVAL calling jobids [%s]: %s" % (phash, jobnum))
    print("... checking: %s (%s)" % (jobdep, jobnum))




