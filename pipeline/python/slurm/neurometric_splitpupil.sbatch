#!/bin/bash
# fit_neurometric_splitpupil.sbatch
#
#SBATCH -J neurometric # A single job name for the array
#SBATCH -p shared # run on cox gpu to use correct env 
#SBATCH -n 8 # one core
#SBATCH -N 1 # on one node
#SBATCH -t 0-02:00 # Running time of 3 hours
#SBATCH --mem 16384 #70656 # Memory request of 70 GB (set to 98304 if exceed lim)
#SBATCH -o neurometric_%A_%a.out # Standard output
#SBATCH -e neurometric_%A_%a.err # Standard error
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rhee@g.harvard.edu


# load modules
module load centos6/0.0.1-fasrc01
module load matlab/R2015b-fasrc01
module load Anaconda/5.0.1-fasrc01

# activate 2p-pipeline environment:
source activate /n/coxfs01/2p-pipeline/envs/behavior3

EXP="$1"
TRACEID="$2"
RTEST="$3"
VAREA="$4"
DKEY="$5"
PARAM="$6"
SIGMOID="$7"
NEGSIG="$8"


echo "varea: ${VAREA}"
echo "dkey: ${DKEY}"

#
if [ $NEGSIG == "False" ]; then

    python /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/classifications/neurometric_fits.py -E $EXP -t $TRACEID -R $RTEST -V $VAREA -k $DKEY -p $PARAM -s $SIGMOID --reverse --pupil -n 8 --new

else
    python /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/classifications/neurometric_fits.py -E $EXP -t $TRACEID -R $RTEST -V $VAREA -k $DKEY -p $PARAM -s $SIGMOID --pupil -n 8 --new

fi
