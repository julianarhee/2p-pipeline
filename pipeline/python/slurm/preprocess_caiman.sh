#
sbatch -o preprocess-caiman_${1}-${2}-${3}-${4}_ds-${5}.out -e preprocess-caiman_${1}-${2}-${3}-${4}_ds-${5}.err /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/slurm/preprocess_caiman.sbatch ${1} ${2} ${3} ${4} ${5}
