#


if [ -z "$7" ]; then   

    sbatch -o patch_caiman_${1}_${2}_${3}_${5}.out -e patch_caiman_${1}_${2}_${3}_${5}.err /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/slurm/patch_extraction.sbatch ${1} ${2} ${3} ${4} ${5} ${6} 

else
    sbatch -o patch_caiman_${1}_${2}_${3}_${5}.out -e patch_caiman_${1}_${2}_${3}_${5}.err /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/slurm/patch_extraction.sbatch ${1} ${2} ${3} ${4} ${5} ${6} ${7}

fi
