#
mkdir ${1}_${2}_${3}_${4}_STD_out
cd ${1}_${2}_${3}_${4}_STD_out

sbatch -o ${2}_${3}_${4}.out -e ${2}_${3}_${4}.err /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/slurm/std_activity_map.sbatch ${1} ${2} ${3} ${4}
