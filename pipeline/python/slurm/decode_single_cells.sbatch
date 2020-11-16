#!/bin/bash
# decode_single_cells.sbatch
#
#SBATCH -J decode # A single job name for the array
#SBATCH -p shared # run on cox gpu to use correct env 
#SBATCH -n 8 # one core
#SBATCH -N 1 # on one node
#SBATCH -t 0-05:00 # Running time of 3 hours
#SBATCH --mem 16384 #70656 # Memory request of 70 GB (set to 98304 if exceed lim)
#SBATCH -o decode_%A_%a.out # Standard output
#SBATCH -e decode_%A_%a.err # Standard error
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rhee@g.harvard.edu


# load modules
module load centos6/0.0.1-fasrc01
module load matlab/R2015b-fasrc01
module load Anaconda/5.0.1-fasrc01

# activate 2p-pipeline environment:
source activate /n/coxfs01/2p-pipeline/envs/pipeline

EXP="$1"
TRACEID="$2"
RTEST="$3"
OVERLAP="$4"
ANALYSIS="$5"
CVAL="$6"
VAREA="$7"
SAMPLESIZE="$8"
DKEY="$9"

echo "ANALYSIS: ${ANALYSIS}"
echo "overlap: ${OVERLAP}"
echo "varea: ${VAREA}"
echo "ncells: ${SAMPLE_SIZE}"
echo "dkey: ${DKEY}"

#
python /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/classifications/decode_by_ncells.py -E $EXP -t $TRACEID -R $RTEST -n 8 -N 100 --new -o $OVERLAP -C ${CVAL} -V $VAREA -S $SAMPLESIZE -X $ANALYSIS -k $DKEY

#{EXP} {TRACEID} {RTEST} {OVERLAP} {ANALYSIS} {CVAL} {VAREA} {NCELLS} {DKEY}".format(

