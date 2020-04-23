#!/usr/bin/env bash

# set Julia package directory
jlproject=$HOME/ProximalDistanceAlgorithms

# extract name of experiment from job name
experiment=$(cut -d'_' -f1 <<< $jobname)

# error = Merged with joblog
#$ -o $logdir/$jobname.$JOB_ID
#$ -j y
# request multiple cores
#$ -pe shared $cores
# request runtime and memory PER CORE
#$ -l arch=$arch,h_rt=$h_rt,h_data=$h_data
# Email address to notify
#$ -M $USER@mail
# Notify when job ends or aborts
#$ -m ea

# get host name
host=$(hostname)

# initialize log with information about job, host, and hardware
echo `date` ": Job $JOB_ID running on host $host..."
echo
lscpu
echo

# load modules needed for the job:
echo "loading modules..."
. /u/local/Modules/default/init/modules.sh
module load julia/1.2.0
which julia

# move to folder with Julia scripts
cd $jlproject/experiments

# read parameters for tasks
while read jlinput
    do
    julia --project=$jlproject -e "$jlinput"
done < $jlproject/experiments/$experiment/jobs/$jobname.in

echo `date` ": Job $JOB_ID complete."
