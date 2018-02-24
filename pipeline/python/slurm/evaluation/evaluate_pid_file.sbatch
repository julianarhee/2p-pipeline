#!/bin/bash
# process_run.sbatch
#
#SBATCH -p cox # run on cox gpu to use correct env 
#SBATCH -n 12 # one core
#SBATCH -N 1 # on one node
#SBATCH -t 0-0:05 # Running time of 5 min
#SBATCH --mem 32768 # Memory request of 12 GB 

# load modules
module load matlab/R2015b-fasrc01
module load Anaconda/5.0.1-fasrc01

# activate 2p-pipeline environment:
source activate /n/coxfs01/2p-pipeline/envs/pipeline

FILENAME="$1"
ZPROJ="mean"
echo "File: ${FILENAME}"

# run processing on raw data
python /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/slurm/evaluate_pid.py $FILENAME $ZPROJ 12

