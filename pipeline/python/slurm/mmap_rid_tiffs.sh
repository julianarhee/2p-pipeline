#!/bin/bash

ANIMALID="$1"
SESSION="$2"
RIDPATH="/n/coxfs01/2p-data/${ANIMALID}/${SESSION}/ROIs/tmp_rids"
echo $RIDPATH

if [ "$#" -gt 2 ]; then
    RIDHASH="$3"
    echo "Requested single ROI ID to memmap."
else
    RIDHASH=""
fi

FILES=($RIDPATH/*$RIDHASH.json)
echo "RID LIST: $FILES"

# get size of array
NUMFILES=${#FILES[@]}

# subtract 1 for 0-indexing
ZBNUMFILES=$(($NUMFILES - 1))
echo "N files: $ZBNUMFILES"

if [ $ZBNUMFILES -ge 0 ]; then
    # export filenames individually
    for i in ${!FILES[*]};do export FILES_$i="${FILES[$i]}";done	
     
    # submit to slurm
    sbatch --array=0-$ZBNUMFILES /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/slurm/mmap_tiffs.sbatch    
fi

