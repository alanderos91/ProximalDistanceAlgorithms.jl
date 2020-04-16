#!/usr/local/bin/bash

dest="alandero@hoffman2.idre.ucla.edu"
localdir="~/Projects/ProximalDistanceAlgorithms/experiments"
remotedir="~/ProximalDistanceAlgorithms/experiments"
dir="$1"

scp -r "./$dir" "$dest:$remotedir"
