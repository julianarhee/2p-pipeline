#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument('-cmd', type=str)
parser.add_argument('-days', type=int, default=0)
parser.add_argument('-hours', type=int, default=2)
parser.add_argument('-mem', type=int, default=8192)
parser.add_argument('-cores', type=int, default=1)
parser.add_argument('-partition', type=str, default='cox', choices=['cox', 'general','bigmem','serial_requeue','gpu','unrestricted'])
parser.add_argument('-mail', action='store_true')
parser.add_argument('-expt', type=int)
parser.add_argument('-mock', action='store_true')

parser.add_argument('-paramfile', type=argparse.FileType('r'))

settings = parser.parse_args(); 

N = sum(1 for line in settings.paramfile)

fo = open("array_job.sbatch", "w")

jobarrayname = "d%d_%d" % (settings.expt, N)
jobname = "d%d_%%a_%d" % (settings.expt, N)
fo.write("#!/bin/bash\n\n")
fo.write("#SBATCH --job-name=%s\n" % jobarrayname)
fo.write("#SBATCH --output=/n/home00/juliana.rhee/slurmbatch/results/expt%d/logs/%s.out\n" % (settings.expt, jobname))
fo.write("#SBATCH --error=/n/home00/juliana.rhee/slurmbatch/results/expt%d/logs/%s.err\n" % (settings.expt,jobname))
fo.write("#SBATCH -t %d-%d:00\n"% (settings.days,settings.hours))
fo.write("#SBATCH -p %s\n" % settings.partition)
fo.write("#SBATCH -n %d\n" % settings.cores)
fo.write("#SBATCH -N 1\n")
fo.write("#SBATCH --mem=%d\n" % settings.mem)
if settings.mail:
    fo.write('#SBATCH --mail-type=FAIL\n')
    fo.write('#SBATCH --mail-user=rhee@g.harvard.edu\n')
fo.write("\ncd %s\n" %  os.getcwd())
#fo.write("module load gcc/4.9.3-fasrc01 tensorflow/0.12.0-fasrc02\n")
fo.write("module load Anaconda/5.0.1-fasrc01\n")
fo.write("source activate /n/coxfs01/2p-pipeline/envs/suite2p\n")

fo.write("python %s -paramfile %s -line ${SLURM_ARRAY_TASK_ID}\n" % (settings.cmd,settings.paramfile.name))

fo.close()

if not settings.mock:
    os.system("sbatch --array=0-%d %s"% (N-1,fo.name))
