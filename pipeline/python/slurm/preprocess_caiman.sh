#


if [ -z "$6" ]; then

    sbatch -o preprocess-caiman_${1}-${2}-${3}-${4}_ds-${5}.out -e preprocess-caiman_${1}-${2}-${3}-${4}_ds-${5}.err /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/slurm/preprocess_caiman_ds.sbatch ${1} ${2} ${3} ${4} ${5}

else

    sbatch -o preprocess-caiman_${1}-${2}-${3}-${4}-${5}_caiman-${6}.out -e preprocess-caiman_${1}-${2}-${3}-${4}-${5}_caiman-${6}.err /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/slurm/preprocess_caiman_opts.sbatch ${1} ${2} ${3} ${4} ${5} ${6} ${7}


fi
