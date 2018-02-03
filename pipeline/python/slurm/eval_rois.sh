#!/bin/bash

ANIMALID="$1"
SESSION="$2"
RIDHASH="$3"
echo "Requested single ROI ID to memmap - $RIDHASH."

if [ "$#" -gt 3 ]; then
    NTIFFS="$4"
    echo "Requesting NMF extraction on ${NTIFFS} tiff files."
else
    NTIFFS=1
fi
echo "N tiffs: ${NTIFFS}"


RIDPATH="/n/coxfs01/2p-data/${ANIMALID}/${SESSION}/ROIs/tmp_rids/tmp_rid_${RIDHASH}.json"

echo "Params path: $RIDPATH"
export RIDPATH RIDHASH

# submit to slurm
sbatch /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/slurm/eval_rois.sbatch

