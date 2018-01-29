#!/bin/bash

ANIMALID="$1"
SESSION="$2"
SPATH="/n/coxfs01/2p-data/${ANIMALID}/${SESSION}/tmp_spids" 
#SPATH="/nas/volume1/2photon/data/${ANIMALID}/${SESSION}/tmp_spids"
echo $SPATH

FILES=($SPATH/*.json)

# get size of array
NUMFILES=${#FILES[@]}

# subtract 1 for 0-indexing
ZBNUMFILES=$(($NUMFILES - 1))

#if [ $ZBNUMFILES -ge 0 ]; then
#    for i in "${FILES[@]}"
#    do
#        echo $i
#    done
#fi

# load modules
module load matlab/R2015b-fasrc01
module load Anaconda/5.0.01-fasrc01

# activate 2p-pipeline environment:
source activate /n/coxfs01/2p-pipeline/envs/pipeline

# grab the files, and export it so the 'child' sbatch jobs can access it:
export FILES

# submit to SLURM
if [ $ZBNUMFILES -ge 0 ]; then
    sbatch --array=0-$ZBNUMFILES process_run.sbatch
fi

source deactivate
