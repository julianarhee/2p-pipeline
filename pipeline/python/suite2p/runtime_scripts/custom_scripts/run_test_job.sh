#!/bin/bash
# process_run.sbatch
#
#SBATCH -J test # A single job name for the array
#SBATCH -p cox # run on cox gpu to use correct env 
#SBATCH -n 24
#SBATCH -N 1 # on one node
#SBATCH -t 0-6:00 # Running time of 3 hours
#SBATCH --mem 800 #70656 #
#SBATCH -o slurm.%N.%j.out # Standard output
#SBATCH -e slurm.%N.%j.err # Standard error

# load modules

module load Anaconda/5.0.1-fasrc01

# activate suite2p environment:
source activate /n/coxfs01/cechavarria/envs/suite2p


# run processing on raw data
python /n/coxfs01/cechavarria/repos/suite2p/python/print_and_wait.py