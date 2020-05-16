#!/usr/bin/env zsh

# extract jobname which contains information for problem sizes
JOBNAME=${1}

# set Julia package directory
PKG=${HOME}/ProximalDistanceAlgorithms

# directory with scripts
DIR=${PKG}/experiments/aw-area51

# function for running benchmark
jlbenchmark () {
    julia --project=${PKG} ${DIR}/benchmark_metric.jl "$@";
}

# redirect all output to a randomly generated log file
PREFIX=$(date +"%Y-%m-%d")
LOG_FILE=$(mktemp ${DIR}/metric/logs/${PREFIX}-XXXXXX)
exec 1>${LOG_FILE} # redirect STDOUT
exec 2>&1          # redirect STDERR to STDOUT

# add header
echo $(date)
echo "Metric Projection Benchmarks"
echo
echo "benchmark:        ${JOBNAME}"
echo "Julia project:    ${PKG}"
echo "scripts:          ${DIR}"
echo

# set maximum number of iterations
MAXITERS=5000

while read n
    do
    # no acceleration
    FNAME=SD_${n}_none
    jlbenchmark --nodes ${n} --algorithm SD --maxiters ${MAXITERS} --filename ${FNAME}.dat

    # Nesterov acceleration
    FNAME=SD_${n}_nesterov
    jlbenchmark --nodes ${n} --algorithm SD --maxiters ${MAXITERS} --accel --filename ${FNAME}.dat
done < ${DIR}/metric/jobs/${JOBNAME}.in
