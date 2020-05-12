#!/usr/bin/env zsh

jobname=${1}

# set Julia package directory
pkg=$HOME/ProximalDistanceAlgorithms

# directory with scripts
dir=$pkg/experiments/aw-area51

# shorthand for running benchmark
alias jlbenchmark=$(julia --project=$pkg ${dir}/metric_benchmark.jl)

while read n
    do
    # no acceleration
    fname=SD_${n}_none
    logfile=${dir}/metric/logs/${fname}

    jlbenchmark --nodes $n --algorithm SD --filename ${fname}.dat > ${logfile}.out 2&>1

    # Nesterov acceleration
    fname=SD_${n}_nesterov
    logfile=${dir}/metric/logs/${fname}

    jlbenchmark --nodes $n --algorithm SD --accel --filename ${fname}.dat > ${logfile}.out 2&>1

done < ${dir}/metric/jobs/${jobname}.in
