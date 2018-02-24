#!/bin/bash
# process_pid_file.sbatch
#
#SBATCH -p cox # run on cox gpu to use correct env 
#SBATCH -n 12 # one core
#SBATCH -N 1 # on one node
#SBATCH -t 0-3:00 # Running time of 3 hours
#SBATCH --mem 70656 # Memory request of 70 GB (inc to 98304)

# load modules
module load matlab/R2015b-fasrc01
module load Anaconda/5.0.1-fasrc01

# activate 2p-pipeline environment:
source activate /n/coxfs01/2p-pipeline/envs/pipeline

# grab filename from array exported from 'parent' shell:
FILENAME="$1"
echo "File: ${FILENAME}"

# run processing on raw data
python /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/slurm/process_pid.py $FILENAME

