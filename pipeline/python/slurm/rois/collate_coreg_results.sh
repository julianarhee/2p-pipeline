#!/bin/bash
ANIMALID="$1"
SESSION="$2"
RIDHASH="$3"
RIDPATH="/n/coxfs01/2p-data/${ANIMALID}/${SESSION}/ROIs/tmp_rids"
#echo $RIDPATH
echo "Requested single RID for collating coregistration results - ${RIDHASH}."

FILES=($RIDPATH/*$RIDHASH.json)

# get size of array
NUMFILES=${#FILES[@]}

# subtract 1 for 0-indexing
ZBNUMFILES=$(($NUMFILES - 1))

if [ $ZBNUMFILES == 0 ]; then
    PARAMSPATH=${FILES[0]}
    echo "Params path: $PARAMSPATH"

    export PARAMSPATH RIDHASH

    # submit to slurm
    sbatch /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/slurm/collate_coreg_results.sbatch $PARAMSPATH

fi

