#!/bin/bash#
# batch_process_pids.sbatch
#
#SBATCH -J processpid # A single job name for the array
#SBATCH -p cox # best partition for single core small jobs
#SBATCH -n 1 # one core
#SBATCH -N 1 # on one node
#SBATCH -t 0-0:10 # Running time of 2 hours
#SBATCH --mem 400 # Memory request of 4 GB (400MB)
#SBATCH -o processpid_%A_%a.out # Standard output
#SBATCH -e processpid_%A_%a.err # Standard error

# grab filename from array exported from 'parent' shell:
FILENAME=${FILES[$SLURM_ARRAY_TASK_ID]}

# make and move into new directory, and run:
mkdir ${FILENAME}_out
cd ${FILENAME}_out

python /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/slurm/process_run.py ${SPATH}/${FILENAME}

