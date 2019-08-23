# create logging dir + run command

#mkdir ${1}_${2}_${3}_${4}_STD_out
#cd ${1}_${2}_${3}_${4}_STD_out
 
sbatch -o RESP_batch_${1}_${2}_resp-${3}_thr-${4}_stds-${5}.out -e RESP_batch_${1}_${2}_resp-${3}_thr-${4}_stds-${5}.err /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/slurm/batch_stats_resp.sbatch ${1} ${2} ${3} ${4} ${5}
