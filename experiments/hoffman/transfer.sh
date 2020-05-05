#!/usr/bin/env bash

dest="alandero@hoffman2.idre.ucla.edu"
localdir="~/Projects/ProximalDistanceAlgorithms/experiments"
remotedir="~/ProximalDistanceAlgorithms/experiments"
dir="$1"

# set working directory to this script's location
cd $(dirname $0)

scp -r "./$dir" "$dest:$remotedir"
