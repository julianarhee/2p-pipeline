#!/bin/bash
# bootstrap_roc.sbatch
#
#SBATCH -J roc # A single job name for the array
#SBATCH -p cox # run on cox gpu to use correct env 
#SBATCH -n 8 # one core
#SBATCH -N 1 # on one node
#SBATCH -t 0-04:00 # Running time of 3 hours
#SBATCH --mem 32764 #70656 # Memory request of 70 GB (set to 98304 if exceed lim)
#SBATCH -o roc_%A_%a.out # Standard output
#SBATCH -e roc_%A_%a.err # Standard error
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rhee@g.harvard.edu


# load modules
module load centos6/0.0.1-fasrc01
module load matlab/R2015b-fasrc01
module load Anaconda/5.0.1-fasrc01

# activate 2p-pipeline environment:
source activate /n/coxfs01/2p-pipeline/envs/pipeline

# grab filename from array exported from 'parent' shell:
#FILENAME="$1"
#echo "File: ${FILENAME}"

echo ${1}
echo ${2}
echo ${3}
echo ${4}

# run processing on raw data
python /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/classifications/bootstrap_roc.py --slurm -i ${1} -S ${2} -A ${3} -E ${4} -t ${5} -n 8 --plot 
