# create logging dir + run command

#mkdir ${1}_${2}_${3}_${4}_STD_out
#cd ${1}_${2}_${3}_${4}_STD_out

sbatch -o activitymap_${1}_${2}_${3}_${4}.out -e activitymap_${1}_{2}_${3}_${4}.err /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/slurm/universal_std.sbatch ${1} ${2} ${3} ${4}
