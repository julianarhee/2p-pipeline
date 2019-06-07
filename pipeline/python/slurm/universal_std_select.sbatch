#!/bin/bash
# universal_std.sbatch
#
#SBATCH -p cox # run on cox gpu to use correct env 
#SBATCH -n 8 # one core
#SBATCH -N 1 # on one node
#SBATCH -t 0-3:00 # Running time of 3 hours
#SBATCH --mem-per-cpu=8192 #70656 # Memory request of 70 GB (inc to 98304)

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
echo ${5}

# run processing on raw data
python /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/rois/get_warped_ref.py --slurm -i ${1} -S ${2} -A ${3} -r ${4} -n 8 -e 1.0 --new -n 8 -z max -s ${5}

