#!/bin/bash
# mmap_tiffs_rid.sbatch
#
#SBATCH -J mmap # A single job name for the array
#SBATCH -p cox # run on cox gpu to use correct env 
#SBATCH -n 12 # one core
#SBATCH -N 1 # on one node
#SBATCH -t 0-0:30 # Running time of 30 min
#SBATCH --mem 76800 #71680 #65536 # Memory request of 12 GB 
#SBATCH -o mmap_%A_%a.out # Standard output
#SBATCH -e mmap_%A_%a.err # Standard error

# load modules
module load centos6/0.0.1-fasrc01
module load matlab/R2015b-fasrc01
module load Anaconda/5.0.1-fasrc01

# activate 2p-pipeline environment:
source activate /n/coxfs01/2p-pipeline/envs/pipeline

# grab filename from array exported from 'parent' shell:
for v in ${!FILES_*};do FILES[${v#FILES_}]="${!v}"; done

FILENAME=${FILES[$SLURM_ARRAY_TASK_ID]}
echo "File: ${FILENAME}"

## create logging dir
#BASEDIR="$(dirname "$FILENAME")"
#OUTDIR="${BASEDIR}/logging_$RIDHASH"
#echo ${OUTDIR}
#
## make and move into new directory, and run:
#if [ ! -d "$OUTDIR" ]; then
#    mkdir $OUTDIR
#fi
#cd ${OUTDIR}
#
# run processing on raw data
python /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/slurm/rois/mmap_tiffs_rid.py $FILENAME

