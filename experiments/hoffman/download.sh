#!/usr/bin/env bash

src="alandero@hoffman2.idre.ucla.edu"
maindir="~/ProximalDistanceAlgorithms/experiments"
dir="$1"

# set working directory to this script's location
cd $(dirname $0)

# download the mirrored directory
scp -r "$src:$maindir/$dir" ./
