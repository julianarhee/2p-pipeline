#!/bin/bash

ANIMALID="$1"
SESSION="$2"
if [ "$#" -gt 2 ]; then
    ACQUISITION="$3"
    RUN="$4"
    echo "Processing SINGLE run..."
    if [ "$#" -gt 4 ]; then
        PIDHASH="$5"
        echo "Requested single PID to process."
    else
        PIDHASH=""
    fi
    PIDPATH="/n/coxfs01/2p-data/${ANIMALID}/${SESSION}/${ACQUISITION}/${RUN}/processed/tmp_pids"
else
    echo "Processing specified runs in session..."
    PIDPATH="/n/coxfs01/2p-data/${ANIMALID}/${SESSION}/tmp_spids"
    PIDHASH=""
fi
#SPATH="/n/coxfs01/2p-data/${ANIMALID}/${SESSION}/tmp_spids" 
#SPATH="/nas/volume1/2photon/data/${ANIMALID}/${SESSION}/tmp_spids"
echo $PIDPATH

#export PYTHONBUFFERED=1
FILES=($PIDPATH/*$PIDHASH.json)
echo "PID LIST: $FILES"

# get size of array
NUMFILES=${#FILES[@]}

# subtract 1 for 0-indexing
ZBNUMFILES=$(($NUMFILES - 1))
echo "N files: $ZBNUMFILES"

if [ $ZBNUMFILES -ge 0 ]; then
    # export filenames individually
    for i in ${!FILES[*]};do export FILES_$i="${FILES[$i]}";done	
     
    # submit to slurm
    sbatch --array=0-$ZBNUMFILES /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/slurm/process_runs.sbatch    
fi

