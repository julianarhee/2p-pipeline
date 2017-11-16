#!/bin/bash

INPUT=""
for var in "$@"
do
    INPUT=$INPUT"'"$var"',"
done
INPUT=${INPUT%?}
echo $INPUT

matlab -nodisplay -nodesktop -nosplash -r "add_repo_paths; bash_process_tiffs($INPUT)"
