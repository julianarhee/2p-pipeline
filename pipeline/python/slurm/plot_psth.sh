# create logging dir + run command


sbatch -o plotpsth_${1}_${2}_${3}_${4}_${5}.out -e plotpsth_${1}_${2}_${3}_${4}_${5}.err /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/slurm/plot_psth.sbatch ${1} ${2} ${3} ${4} ${5} ${6} ${7} ${8} ${9}
