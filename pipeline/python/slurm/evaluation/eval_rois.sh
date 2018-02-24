#!/bin/bash

ANIMALID="$1"
SESSION="$2"
RIDHASH="$3"
echo "Requested single ROI ID to evaluate - $RIDHASH."

if [ "$#" == 5 ]; then
    SNR="$4"
    RCORR="$5"
elif [ "$#" == 4]; then
    SNR="$4"
    RCORR="0.8"
else
    SNR="2.0"
    RCORR="0.8"
fi
echo "Setting min SNR value to: ${SNR}."
 echo "Setting min spatial corr thr to: ${RCORR}."
  
#else
#    NTIFFS=1
#fi
#echo "N tiffs: ${NTIFFS}"
#

RIDPATH="/n/coxfs01/2p-data/${ANIMALID}/${SESSION}/ROIs/tmp_rids/tmp_rid_${RIDHASH}.json"

echo "Params path: $RIDPATH"
export RIDPATH RIDHASH SNR RCORR

# submit to slurm
sbatch /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/slurm/eval_rois.sbatch

