# create logging dir + run command

#mkdir ${1}_${2}_${3}_${4}_STD_out
#cd ${1}_${2}_${3}_${4}_STD_out
 
sbatch -o GRAT-batch_${1}-${2}-resp-${3}-thr-${4}-nstds-${5}_nboot${6}-nsamples${7}.out -e GRAT-batch_${1}-${2}-resp-${3}-thr-${4}-nstds-${5}_nboot${6}-nsamples${7}.err /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/slurm/batch_stats_gratings.sbatch ${1} ${2} ${3} ${4} ${5} ${6} ${7}
