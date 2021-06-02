#!/bin/bash
# responsive_test.sbatch
#
#SBATCH -J rtest # A single job name for the array
#SBATCH -p cox # run on cox gpu to use correct env 
#SBATCH -n 1 # one core
#SBATCH -N 1 # on one node
#SBATCH -t 0-00:10 # Running time of 3 hours
#SBATCH --mem 4000 #70656 # Memory request of 70 GB (set to 98304 if exceed lim)
#SBATCH -o rtest_%A_%a.out # Standard output
#SBATCH -e rtest_%A_%a.err # Standard error
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
RTYPE="$6"
RTEST="$7"
RTHR="$8"
EPOCH="$9"

echo "ANIMALID: ${ANIMALID}"
echo "SESSION: ${SESSION}"

# {ANIMALID} {SESSION} {FOV} {EXP} {TRACEID} {RTYPE} {RTEST} {RTHR}".format(

# run processing on raw data
python /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/classifications/aggregate_data_stats.py -i $ANIMALID -S $SESSION -A $FOV -E $EXP -t $TRACEID --new -n 1 --redo-fov --redo-stats -d $RTYPE -R $RTEST --thr=$RTHR -e $EPOCH


