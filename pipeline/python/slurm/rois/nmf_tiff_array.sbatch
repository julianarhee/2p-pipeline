#!/bin/bash
# nmf_tiff_array.sbatch
#
#SBATCH -J nmf # A single job name for the array
#SBATCH -p cox # run on cox gpu to use correct env 
#SBATCH -n 12 # one core
#SBATCH -N 1 # on one node
#SBATCH -t 0-2:00 # Running time of 3 hours
#SBATCH --mem 98304 #6350 #32768 #65536 # Memory request of 12 GB 
#SBATCH -o nmf_%A_f%a.out # Standard output
#SBATCH -e nmf_%A_f%a.err # Standard error

# load modules
module load matlab/R2015b-fasrc01
module load Anaconda/5.0.1-fasrc01

# activate 2p-pipeline environment:
source activate /n/coxfs01/2p-pipeline/envs/pipeline

# grab filename from array exported from 'parent' shell:
echo "p info: ${PARAMSPATH}"
echo "SLURM ID: ${SLURM_ARRAY_TASK_ID}"
#
## create logging dir
#BASEDIR="$(dirname "$PARAMSPATH")"
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
python /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/slurm/rois/nmf_tiff.py $PARAMSPATH $SLURM_ARRAY_TASK_ID 12
#python /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/slurm/nmf_tiff.py $PARAMSPATH $TIFFNUM 12

