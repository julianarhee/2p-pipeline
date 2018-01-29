#!/bin/bash

ANIMALID="$1"
SESSION="$2"
#SPATH="/n/coxfs01/2p-data/${ANIMALID}/${SESSION}/tmp_spids" 
SPATH="/nas/volume1/2photon/data/${ANIMALID}/${SESSION}/tmp_spids"
echo $SPATH

export PYTHONBUFFERED=1

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
    FILENAME=$i 
    python /home/julianarhee/Repositories/2p-pipeline/pipeline/python/slurm/process_run.py ${FILENAME}
    done    
    # submit to slurm
    #sbatch --array=0-$ZBNUMFILES /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/slurm/process_run.sbatch
    

    #OUTDIR="${FILENAME}_out"
    #echo $OUTDIR

    # make and move into new directory, and run:
    #mkdir ${OUTDIR} #${FILENAME}_out
    #cd ${OUTDIR} #${FILENAME}_out

fi

