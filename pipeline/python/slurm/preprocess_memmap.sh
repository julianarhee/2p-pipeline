#


sbatch -o mmap-caiman_${1}-${2}-${3}-${4}_${5}.out -e mmap-caiman_${1}-${2}-${3}-${4}_${5}.err /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/slurm/preprocess_caiman_memmap.sbatch ${1} ${2} ${3} ${4} ${5}

