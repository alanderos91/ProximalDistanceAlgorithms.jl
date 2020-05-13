#!/usr/bin/env zsh

# extract jobname which should be the name of a file in data/ directory
JOBNAME=${1}

# set Julia package directory
PKG=${HOME}/ProximalDistanceAlgorithms

# directory with scripts
DIR=${PKG}/experiments/aw-area51

# function for running benchmark
jlbenchmark () {
    julia --project=${PKG} ${DIR}/benchmark_cvxcluster.jl "$@";
}

# redirect all output to a randomly generated log file
PREFIX=$(date +"%Y-%m-%d")
LOG_FILE=$(mktemp ${DIR}/cvxcluster/logs/${PREFIX}-XXXXXX)
exec 1>${LOG_FILE} # redirect STDOUT
exec 2>&1          # redirect STDERR to STDOUT

# add header
echo $(date)
echo "Convex Clustering Benchmarks"
echo
echo "benchmark:        ${JOBNAME}"
echo "Julia project:    ${PKG}"
echo "scripts:          ${DIR}"
echo

# set maximum number of iterations
MAXITERS=2000

# no acceleration
FNAME=SD_${JOBNAME}_none
jlbenchmark --data ${JOBNAME}.dat --algorithm SD --maxiters ${MAXITERS} --filename ${FNAME}.dat

# Nesterov acceleration
FNAME=SD_${JOBNAME}_nesterov
jlbenchmark --data ${JOBNAME}.dat --algorithm SD --maxiters ${MAXITERS} --accel --filename ${FNAME}.dat
