#!/usr/bin/env bash

# delete previous output from slurm
rm -rf *.out

# submit the job to the queue
sbatch submission.slurm > submission.txt

if [[ ! `cat submission.txt` =~ "Submitted" ]]; then
   echo "Issue submitting..."
   cat submission.txt
   rm -f submission.txt
   exit 1
fi

JOBNUM=`cat submission.txt | awk '{print $4}'`

rm -f submission.txt

# wait for the job to get picked up and start producing output
until [ -f slurm-$JOBNUM.out ]
do
  sleep 1
done

# open the output file and follow the file as new output is added
less +F *.out
