#!/usr/bin/env bash

# input: jobname
jobname=$1
# set working directory to this script's location
cd $(dirname $0)

# retrieve parameters from master list and pass to script
cat joblist | grep $jobname | xargs ./submit.sh
