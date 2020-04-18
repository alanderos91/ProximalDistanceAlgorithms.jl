#!/usr/bin/env bash

# set working directory to this script's location
cd $(dirname $0)

jobname=$1
experiment=$(cut -d'_' -f1 <<< $jobname)

if [[ "$experiment=metric" ]]; then
    jlinclude="include(\\\"metric_projection.jl\\\")"
elif [[ "$experiment=cvxreg" ]]; then
    jlinclude="include(\\\"convex_regression.jl\\\")"
elif [[ "$experiment=cvxcluster" ]]; then
    jlinclude="include(\\\"convex_clustering.jl\\\")"
elif [[ "$experiment=denoise" ]]; then
    jlinclude="include(\\\"image_denoising.jl\\\")"
else
    echo "Unrecognized experiment '$experiment'"
    exit 1
fi

# pull params from command line
params=()
params[0]="key=$2;"
params[1]="strategy=$3;"
params[2]="seed=$4;"
params[3]="maxiters=$5"
params[4]="sample_rate=$6"
params[5]="ntrials=$7"

# kinda unsafe...?
while IFS= read -r -p "Additional arguments (end with empty line): " keyval; do
    [[ $keyval ]] || break
    echo "${params[@]} $keyval $jlinclude" >> $experiment/jobs/$jobname.in
done
