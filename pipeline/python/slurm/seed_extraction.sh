#
sbatch -o seed_caiman_${1}_${2}_${3}_${4}_${5}_${6}.out -e seed_caiman_${1}_${2}_${3}_${4}_${5}_${6}.err /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/slurm/seed_extraction.sbatch ${1} ${2} ${3} ${4} ${5} ${6}
