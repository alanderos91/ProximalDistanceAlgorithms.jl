#!/usr/bin/env zsh

# extract jobname which contains information for problem sizes
JOBNAME=${1}

# set Julia package directory
PKG=${HOME}/Projects/ProximalDistanceAlgorithms

# directory with scripts
DIR=${PKG}/experiments/aw-area51

# directory to Julia
JLDIR=${HOME}/julia-1.5

# function for running benchmark
jlbenchmark () {
    ${JLDIR}/bin/julia --project=${PKG} ${DIR}/benchmark_imgtvd.jl "$@";
}

# redirect all output to a randomly generated log file
PREFIX=$(date +"%Y-%m-%d")
LOG_FILE=$(mktemp ${DIR}/denoise/logs/${PREFIX}-${JOBNAME}-XXXXXX)
exec 1>${LOG_FILE} # redirect STDOUT
exec 2>&1          # redirect STDERR to STDOUT

# add header
echo $(date)
echo "Image Denoising Benchmarks"
echo
echo "benchmark:        ${JOBNAME}"
echo "Julia project:    ${PKG}"
echo "scripts:          ${DIR}"
echo

# set maximum number of iterations
MAXITERS=5000

# each algorithm, except ADMM, should be run with Nesterov acceleration
while read image
    do

    # Steepest Descent
    FNAME=SD_${image}_l0
    jlbenchmark --image ${image} --algorithm SD --proj l0 --maxiters ${MAXITERS} --accel\
        --filename ${FNAME}.dat \
        --start 0.5 --step 2e-2 \
        --atol 1e-4 --rtol 1e-4

    FNAME=SD_${image}_l1
    jlbenchmark --image ${image} --algorithm SD --proj l1 --maxiters ${MAXITERS} --accel \
        --filename ${FNAME}.dat \
        --start 0.5 --step 2e-2 \
        --atol 1e-4 --rtol 1e-4
done < ${DIR}/denoise/jobs/${JOBNAME}.in
