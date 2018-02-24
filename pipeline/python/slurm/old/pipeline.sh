#! /bin/bash

function usage() {
    cat <<EOF
SYNOPSIS
  pipeline.sh      - run 2p-extraction pipeline
  pipeline.sh help - display this help message

DESCRIPTION
  Look for XID files in session directory.
  For PID files, run tiff-processing, evaluate.
  For RID files, wait for PIDs to finish, then extract rois, evaluate.  

AUTHOR
  Juliana Rhee
EOF
}

function info() {
    echo "INFO: $@" >&2
}
function error() {
    echo "ERR:  $@" >&2
}
function fatal() {
    echo "ERR:  $@" >&2
    exit 1
}


# Create a (hopefully) unique prefix for the names of all jobs in this 
# particular run of the pipeline. This makes sure that runs can be
# identified unambiguously
run=$(uuidgen | tr '-' ' ' | awk '{print $1}')

# show help message if there were any arguments
if [[ $1 == "help" ]]; then usage; exit; fi

#####################################################################
#                          find XID files                           #
#####################################################################

# Get PID(s) based on name
# Note: the syntax a+=(b) adds b to the array a
ANIMALID="$1"
ESSION="$2"
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
    PIDPATH="/n/coxfs01/2p-data/${ANIMALID}/${SESSION}/${ACQUISITION
}/${RUN}/processed/tmp_pids"
else
    echo "Processing specified runs in session..."
    PIDPATH="/n/coxfs01/2p-data/${ANIMALID}/${SESSION}/tmp_spids"
    PIDHASH=""
fi
pid_files=($PIDPATH/*$PIDHASH.json)
echo "PID LIST: $pid_files"

# make sure there are pids
if [[ ${#pid_files[@]} -eq 0 ]]; then
    fatal "no PIDs found in dir ${PIDPATH}"
fi

info "PIDs: n = ${#pid_files[@]}" 
for pid in ${pid_files[@]}; do
    info "    $pid"
done


################################################################################
#                               run the pipeline                               #
################################################################################
mkdir -p log 

# STEP1: TIFF PROCESSING. All PIDs will be processed in parallel since they don't
#        have any dependencies

pid_jobids=()
for f in ${pid_files[@]}; do
    n=$(basename ${f%%.*})
    pid_jobids+=($(sbatch --job-name=$run.processpid.$n \
        -o "log/$run.processpid.$n.out" \
        -e "log/$run.procsspid.$n.err" \
        process_pid_file.sbatch $f))
done

info "PROCESSING jobids: ${pid_jobid}"

# STEP2: MC EVALUATION. Each mceval call will start when the corresponding processing 
#        call finishes sucessfully. Note, if STEP1 fails, all jobs depending on
#        it will remain in the queue and need to be canceled explicitly.
#        An alternative would be to use 'afterany' and make each job check for
#        the successful execution of the prerequisites.
eval_jobids=()
for i in $(seq 1 ${#pid_files[@]}); do
    idx=$((i - 1))
    n=$(basename ${pid_files[$idx]%%.*})
    FN="${pid_files[$idx]}"
    echo "$FN"
    mceval_jobids+=($(sbatch --job-name=$run.mceval.$n \
        -o "log/$run.mceval.$n.out" \
        -e "log/$run.mceval.$n.err" \
        --dependency=afterok:${pid_jobids[$idx]} \ 
        /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/slurm/evaluate_pid_file.sbatch $FN))
done
info "MCEVAL calling jobids: ${peak_jobids[@]}"

# STEP3: ROI EXTRACTION: Each nmf call will start when the corresponding alignments
#        finish sucessfully. Note, if STEP1 fails, all jobs depending on
#        it will remain in the queue and need to be canceled explicitly.
#        An alternative would be to use 'afterany' and make each job check for
#        the successful execution of the prerequisites.


