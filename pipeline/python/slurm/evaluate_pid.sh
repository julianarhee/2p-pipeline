#!/bin/bash

ANIMALID="$1"
SESSION="$2"
ZPROJ="mean"
if [ "$#" -gt 2 ]; then
    ACQUISITION="$3"
    RUN="$4"
    echo "Processing SINGLE run..."
    if [ "$#" -gt 4 ]; then
        PIDHASH="$5"
        echo "Requested single PID to evaluate."
    else
        PIDHASH=""
    fi
    PIDPATH="/n/coxfs01/2p-data/${ANIMALID}/${SESSION}/${ACQUISITION}/${RUN}/processed/tmp_pids/completed"
else
    echo "Evaluating specified runs in session..."
    PIDPATH="/n/coxfs01/2p-data/${ANIMALID}/${SESSION}/tmp_spids/completed"
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
    echo "$ZPROJ"
    export PIDHASH ZPROJ
 
    # submit to slurm
    sbatch --array=0-$ZBNUMFILES /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/slurm/evaluate_pid.sbatch    
fi
