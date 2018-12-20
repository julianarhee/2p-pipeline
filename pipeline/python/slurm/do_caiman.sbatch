#!/bin/bash
# do_caiman.sbatch
#
#SBATCH -p cox # run on cox gpu to use correct env 
#SBATCH -n 16 # one core
#SBATCH -N 1 # on one node
#SBATCH -t 0-6:00 # Running time of 3 hours
#SBATCH --mem 196608 #18432 #70656 # Memory request of 70 GB (inc to 98304)

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
echo ${6}
echo ${7}


# run processing on raw data
python /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/rois/caiman_steps.py --slurm -i ${1} -S ${2} -A ${3} -R ${4} -t ${5} -d ${6} --gSig=${7} --psth -p corrected -r ${8} -C ${9} -H ${10}

