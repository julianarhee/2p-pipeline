#!/bin/bash
# patch_extraction.sbatch
#
#SBATCH -p cox # run on cox gpu to use correct env 
#SBATCH -n 16 # one core
#SBATCH -N 1 # on one node
#SBATCH -t 0-24:00 # Running time of 3 hours
#SBATCH --mem=196608 #32768 #122880 #18432 #70656 # Memory request of 70 GB (inc to 98304)
#SBATCH -o patch_%A_%a.out # Standard output
#SBATCH -e patch_%A_%a.err # Standard error
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rhee@g.harvard.edu


# load modules
module load centos6/0.0.1-fasrc01
module load matlab/R2015b-fasrc01
module load Anaconda/5.0.1-fasrc01
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# activate 2p-pipeline environment:
source activate /n/coxfs01/2p-pipeline/envs/caiman

# grab filename from array exported from 'parent' shell:
#FILENAME="$1"
#echo "File: ${FILENAME}"

# run processing on raw data
python /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/traces/extract_caiman.py -i ${1} -S ${2} -A ${3} -E ${4} -n 16


#if [ -z "$7" ]; then   
#    python /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/traces/extract_caiman.py -i ${1} -S ${2} -A ${3} -E ${4} -n 8 --prefix=${5} --mmap=${6} 
#
#else
#    python /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/traces/extract_caiman.py -i ${1} -S ${2} -A ${3} -E ${4} -n 8 --prefix=${5} --mmap=${6} ${7}
#
#fi


