#!/bin/bash
# retino_gradients.sbatch
#
#SBATCH -J gradients # A single job name for the array
#SBATCH -p cox # run on cox gpu to use correct env 
#SBATCH -n 1 # one core
#SBATCH -N 1 # on one node
#SBATCH -t 0-0:20 # Running time of 3 hours
#SBATCH --mem 8192 #70656 # Memory request of 70 GB (set to 98304 if exceed lim)
#SBATCH -o retinograd_%A_%a.out # Standard output
#SBATCH -e retinograd_%A_%a.err # Standard error
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rhee@g.harvard.edu


# load modules
module load centos6/0.0.1-fasrc01
module load matlab/R2015b-fasrc01
module load Anaconda/5.0.1-fasrc01

# activate 2p-pipeline environment:
source activate /n/coxfs01/2p-pipeline/envs/pipeline

# grab filename from array exported from 'parent' shell:
#for v in ${!ANIMALID_*};do ANIMALID[${v#ANIMALID_}]="${!v}"; done

#FILENAME=${FILES[$SLURM_ARRAY_TASK_ID]}
#echo "File: ${FILENAME}"

ANIMALID="$1"
SESSION="$2"
FOV="$3"
EXP="$4"
TRACEID="$5"
FILTER="$6"

echo "ANIMALID: ${ANIMALID}"
echo "SESSION: ${SESSION}"

#
## create logging dir
#BASEDIR="$(dirname "$FILENAME")"
#OUTDIR="${BASEDIR}/logging_$PIDHASH"
#echo ${OUTDIR}
#
## make and move into new directory, and run:
#if [ ! -d "$OUTDIR" ]; then
#    mkdir $OUTDIR
#fi
#cd ${OUTDIR}

# run processing on raw data
#python /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/classifications/gradient_estimation.py -i $ANIMALID -S $SESSION -A $FOV -R $EXP -t $TRACEID -c magenta --plot-examples -p $FILTER --thr 0.01 -M ridge 

python /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/classifications/gradient_estimation.py -i $ANIMALID -S $SESSION -A $FOV -R $EXP -t $TRACEID -c magenta --plot-examples -p $FILTER --thr 0.002

#python /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/classifications/evaluate_receptivefield_fits.py -i $ANIMALID -S $SESSION -A $FOV -R $EXP -t $TRACEID -M dff -p 0.5 -c magenta -n 1 --test

