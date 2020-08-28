#!/usr/bin/env zsh

# extract jobname which should be the name of a file in data/ directory
JOBNAME=${1}

# set Julia package directory
PKG=${HOME}/Projects/ProximalDistanceAlgorithms

# directory with scripts
DIR=${PKG}/experiments/aw-area51

# function for running benchmark
jlbenchmark () {
    julia --project=${PKG} ${DIR}/benchmark_cvxcluster.jl "$@";
}

# redirect all output to a randomly generated log file
PREFIX=$(date +"%Y-%m-%d")
LOG_FILE=$(mktemp ${DIR}/cvxcluster/logs/${PREFIX}-${JOBNAME}-XXXXXX)
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
MAXITERS=3000

# each algorithm, except ADMM, should be run with Nesterov acceleration

while read dataset
    do
    # MM
    FNAME=MM_CG_${dataset}
    jlbenchmark --data ${dataset}.dat --algorithm MM --ls CG --maxiters ${MAXITERS} --accel \
        --filename ${FNAME}.dat \
        --step 5e-3 --start 0.1 \
        --atol 1e-8

    # Steepest Descent
    FNAME=SD_${dataset}
    jlbenchmark --data ${dataset}.dat --algorithm SD --maxiters ${MAXITERS} --accel \
        --filename ${FNAME}.dat \
        --step 5e-3 --start 0.1 \
        --atol 1e-8
done < ${DIR}/cvxcluster/jobs/${JOBNAME}.in
