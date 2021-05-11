#!/bin/bash
# process_rfs_spherical.sbatch
#
#SBATCH -J procrf # A single job name for the array
#SBATCH -p cox # run on cox gpu to use correct env 
#SBATCH -n 1 # one core
#SBATCH -N 1 # on one node
#SBATCH -t 0-01:00 # Running time of 3 hours
#SBATCH --mem 16384 #70656 # Memory request of 70 GB (set to 98304 if exceed lim)
#SBATCH -o rfs_%A_%a.out # Standard output
#SBATCH -e rfs_%A_%a.err # Standard error
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rhee@g.harvard.edu


# load modules
module load centos6/0.0.1-fasrc01
#module load matlab/R2015b-fasrc01
module load Anaconda/5.0.1-fasrc01

# activate 2p-pipeline environment:
source activate /n/coxfs01/2p-pipeline/envs/behavior3

ANIMALID="$1"
SESSION="$2"
FOV="$3"
EXP="$4"
TRACEID="$5"

echo "ANIMALID: ${ANIMALID}"
echo "SESSION: ${SESSION}"

# run processing on raw data
python /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/classifications/notebooks/response_stats/fit_rfs.py -i $ANIMALID -S $SESSION -A $FOV -R $EXP -t $TRACEID -M dff -p 0.5 -n 1 --sphere --new


