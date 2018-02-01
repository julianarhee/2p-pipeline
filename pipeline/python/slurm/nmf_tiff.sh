#!/bin/bash

ANIMALID="$1"
SESSION="$2"
RIDHASH="$3"
RIDPATH="/n/coxfs01/2p-data/${ANIMALID}/${SESSION}/ROIs/tmp_rids"
#echo $RIDPATH
echo "Requested single ROI ID to memmap - ${RIDHASH}."

if [ "$#" -gt 3 ]; then
    NTIFFS="$4"
    echo "Requesting NMF extraction on ${NTIFFS} tiff files."
else
    NTIFFS=1
fi
echo "N tiffs: ${NTIFFS}"

FILES=($RIDPATH/*$RIDHASH.json)

# get size of array
NUMFILES=${#FILES[@]}

# subtract 1 for 0-indexing
ZBNUMFILES=$(($NUMFILES - 1))

if [ $ZBNUMFILES == 0 ]; then
    PARAMSPATH=${FILES[0]}
    echo "Params path: ${PARAMSPATH}"
    
    for TIFFNUM in $(seq 1 $NUMFILES); do
    echo "RID ${RIDHASH}, File ${TIFFNUM}"    
    export PARAMSPATH TIFFNUM
    
    # submit to slurm
    sbatch -o out_r${RIDHASH}_f${TIFFNUM}.out -e out_r${RIDHASH}_f${TIFFNUM}.err --job-name=nmf_${RIDHASH}_f${TIFFNUM} /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/slurm/nmf_tiff.sbatch 
    sleep 1
    done
fi

