#!/usr/bin/env bash

# set working directory to this script's location
cd $(dirname $0)

read -p "Enter job name (<experiment>_<desc>): " jobname
experiment=$(cut -d'_' -f1 <<< $jobname)

if [[ "$experiment" != "metric" && "$experiment" != "cvxreg" && "$experiment" != "cvxcluster" && "$experiment" != "denoise" ]]; then
    echo "Unrecognized experiment '$experiment'"
    exit 1
fi

read -p "Enter architecture (intel-E5-*): " arch

if [[ -z $arch ]]; then
    echo "  Setting to default intel-E5-*"
    arch="intel-E5-*"
fi

# need add escaped double-quotes
arch="\\\"$arch\\\""

read -p "Enter number of cores (4): " cores

if [[ -z $cores ]]; then
    echo "  Setting to default 4"
    cores="4"
fi

read -p "Enter hard limit for requested time (1:00:00): " h_rt

if [[ -z $h_rt ]]; then
    echo "  Setting to default 1:00:00"
    h_rt="1:00:00"
fi

read -p "Enter hard limit for requested memory (8G): " h_data

if [[ -z $h_data ]]; then
    echo "  Setting to default 8G"
    h_data="8G"
fi

echo "Creating job:"
echo "  $jobname $arch $cores $h_rt $h_data"
read -r -p "Is this correct? [y/Y] " confirm
confirm=${confirm^^}

if [[ "$confirm" == "Y" ]]; then
    echo "$jobname $arch $cores $h_rt $h_data" >> joblist
fi
