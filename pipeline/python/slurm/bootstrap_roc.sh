# create logging dir + run command

#mkdir ${1}_${2}_${3}_${4}_STD_out
#cd ${1}_${2}_${3}_${4}_STD_out

sbatch -o bootstrap_roc_${1}_${2}_${3}_${4}_${5}.out -e bootstrap_roc_${1}_${2}_${3}_${4}_${5}.err /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/slurm/bootstrap_roc.sbatch ${1} ${2} ${3} ${4} ${5}
