#!/bin/bash

ANIMALID="$1"
SESSION="$2"
SPATH="/n/coxfs01/2p-data/${ANIMALID}/${SESSION}/tmp_spids" 
echo $SPATH

export FILES=($SPATH/*.json)

# get size of array
NUMFILES=${#FILES[@]}

# subtract 1 for 0-indexing
ZBNUMFILES=$(($NUMFILES - 1))
echo "N files: $ZBNUMFILES"
if [ $ZBNUMFILES -ge 0 ]; then
    for i in "${FILES[@]}"
    do
        echo $i
    done
    # submit to slurm
    sbatch --array=0-$ZBNUMFILES /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/slurm/process_run.sbatch
fi

