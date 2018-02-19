#!/bin/bash
# process_run.sbatch
#
#SBATCH -J evalroi # A single job name for the array
#SBATCH -p cox # run on cox gpu to use correct env 
#SBATCH -n 12 # one core
#SBATCH -N 1 # on one node
#SBATCH -t 0-1:00 # Running time of 3 hours
#SBATCH --mem-per-cpu 18432 #20480 #18432  # Memory request of 4 GB 
#SBATCH -o evalroi_%A_f%a.out # Standard output
#SBATCH -e evalroi_%A_f%a.err # Standard error

# load modules
module load matlab/R2015b-fasrc01
module load Anaconda/5.0.1-fasrc01

# activate 2p-pipeline environment:
source activate /n/coxfs01/2p-pipeline/envs/pipeline

# grab filename from array exported from 'parent' shell:
echo "roi info: ${RIDPATH}"

# run processing on raw data
python /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/slurm/eval_rois.py $RIDPATH 12

