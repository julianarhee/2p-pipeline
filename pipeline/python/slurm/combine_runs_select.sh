#
#mkdir ${1}_${2}_${3}_${4}_${5}_out
#cd ${1}_${2}_${3}_${4}_${5}_out

sbatch -o combine_select_${1}_${2}_${3}_${4}_${5}.out -e combine_select_${1}_${2}_${3}_${4}_${5}.err /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/slurm/combine_runs_select.sbatch ${1} ${2} ${3} ${4} ${5} ${6}
