#!/bin/bash

INLOG=$1
OUTLOG=$2
EMAIL=$3

#mail ${EMAIL} < ${OUTLOG}

badcode="1:0"

# Create array of job ids
i=0
while IFS= read -r line; 
do JID=$(echo $line | grep -Po "(?<=\[)[^\]]*(?=\])")
if [ -n "${JID}" ]; then
    
    arr[$i]="$JID"
    i=$((i+1))
    
fi
done < ${INLOG}


# Query for exit status, save to file if bad exit
# ----------------------------------------------------
# JOBID|JOBNAME|PARTITION|GROUP|NCORES|STATUS|EXITCODE 
# ----------------------------------------------------

for x in "${arr[@]}"; do
    JID=$x
    OUT=$(sacct -n -P -j "${JID}" |grep -v "^[0-9]*\.");
    #echo "${OUT}"

    IFS=\| read -a fields <<<"$OUT"
    EXCODE="${fields[6]}" #$(echo "${fields[6]}");
    if [ "${EXCODE}" == "${badcode}" ]; then
        echo "${fields[1]}" 
    fi
done >${OUTLOG}

mail ${EMAIL} < ${OUTLOG}

echo "DONE."
