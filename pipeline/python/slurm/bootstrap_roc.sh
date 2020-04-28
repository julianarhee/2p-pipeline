
sbatch -o boot-ROC_${1}-${2}-${3}-${4}-${5}.out -e boot-ROC_${1}-${2}-${3}-${4}-${5}.err /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/slurm/bootstrap_roc.sbatch ${1} ${2} ${3} ${4} ${5}
