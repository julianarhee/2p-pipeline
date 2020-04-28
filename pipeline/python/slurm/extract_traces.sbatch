#!/bin/bash
# extract_traces.sbatch
#
#SBATCH -p cox # run on cox gpu to use correct env 
#SBATCH -n 1 # one core
#SBATCH -N 1 # on one node
#SBATCH -t 0-3:00 # Running time of 3 hours
#SBATCH --mem 32768 #18432 #70656 # Memory request of 70 GB (inc to 98304)

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
python /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/traces/get_traces.py --slurm -i ${1} -S ${2} -A ${3} -R ${4} -t ${5} -c 0.7 -a ${6} --neuropil --warp --align --raw --nonnegative --post=${8} --pre=${7}
#--psth -d dff -r ${8} -C ${9} -H ${10}

