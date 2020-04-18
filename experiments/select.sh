#!/usr/bin/env bash

# input: jobname

# set working directory to this script's location
cd $(dirname $0)

# retrieve parameters from master list and pass to script
cat joblist | grep $1 | xargs ./submit.sh
