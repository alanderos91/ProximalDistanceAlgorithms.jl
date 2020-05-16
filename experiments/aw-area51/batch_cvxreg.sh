#!/usr/bin/env zsh

# extract jobname which contains information for problem sizes
JOBNAME=${1}

# set Julia package directory
PKG=${HOME}/ProximalDistanceAlgorithms

# directory with scripts
DIR=${PKG}/experiments/aw-area51

# function for running benchmark
jlbenchmark () {
    julia --project=${PKG} ${DIR}/benchmark_cvxreg.jl "$@";
}

# redirect all output to a randomly generated log file
PREFIX=$(date +"%Y-%m-%d")
LOG_FILE=$(mktemp ${DIR}/cvxreg/logs/${PREFIX}-${JOBNAME}-XXXXXX)
exec 1>${LOG_FILE} # redirect STDOUT
exec 2>&1          # redirect STDERR to STDOUT

# add header
echo $(date)
echo "Convex Regression Benchmarks"
echo
echo "benchmark:        ${JOBNAME}"
echo "Julia project:    ${PKG}"
echo "scripts:          ${DIR}"
echo

# set maximum number of iterations
MAXITERS=5000

while read probsize
    do
    d=$(cut -d',' -f1 <<< ${probsize})
    n=$(cut -d',' -f2 <<< ${probsize})

    # no acceleration
    FNAME=SD_${d}_${n}_none
    jlbenchmark --features ${d} --samples ${n} --algorithm SD --maxiters ${MAXITERS} --filename ${FNAME}.dat

    # Nesterov acceleration
    FNAME=SD_${d}_${n}_nesterov
    jlbenchmark --features ${d} --samples ${n} --algorithm SD --maxiters ${MAXITERS} --accel --filename ${FNAME}.dat
done < ${DIR}/cvxreg/jobs/${JOBNAME}.in
