#
mkdir ${1}_${2}_${3}_${4}_${5}_STD_out
cd ${1}_${2}_${3}_${4}_${5}_STD_out

sbatch -o ${2}_${3}_${4}_${5}.out -e ${2}_${3}_${4}_${5}.err /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/slurm/retino_analysis.sbatch ${1} ${2} ${3} ${4} ${5}
