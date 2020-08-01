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
MAXITERS=2000

# each algorithm, except ADMM, should be run with Nesterov acceleration
while read image
    do

    # # MM
    # FNAME=MM_LSQR_${image}
    # jlbenchmark --image ${image} --algorithm MM --ls LSQR --maxiters ${MAXITERS} --accel --filename ${FNAME}.dat

    FNAME=MM_CG_${image}
    jlbenchmark --image ${image} --algorithm MM --ls CG --maxiters ${MAXITERS} --accel --filename ${FNAME}.dat --step 5e-2

    # Steepest Descent
    FNAME=SD_${image}
    jlbenchmark --image ${image} --algorithm SD --maxiters ${MAXITERS} --accel --filename ${FNAME}.dat --step 5e-2

    # # ADMM
    # FNAME=ADMM_LSQR_${image}
    # jlbenchmark --image ${image} --algorithm ADMM --ls LSQR --maxiters ${MAXITERS} --filename ${FNAME}.dat

    FNAME=ADMM_CG_${image}
    jlbenchmark --image ${image} --algorithm ADMM --ls CG --maxiters ${MAXITERS} --filename ${FNAME}.dat --step 5e-2

    # # MM Subspace{5}
    # FNAME=MMS5_LSQR_${image}
    # jlbenchmark --image ${image} --algorithm MMS --subspace 5 --ls LSQR --maxiters ${MAXITERS} --accel --filename ${FNAME}.dat
    #
    # FNAME=MMS5_CG_${image}
    # jlbenchmark --image ${image} --algorithm MMS --subspace 5 --ls CG --maxiters ${MAXITERS} --accel --filename ${FNAME}.dat
    #
    # # MM Subspace{10}
    # FNAME=MMS10_LSQR_${image}
    # jlbenchmark --image ${image} --algorithm MMS --subspace 10 --ls LSQR --maxiters ${MAXITERS} --accel --filename ${FNAME}.dat
    #
    # FNAME=MMS10_CG_${image}
    # jlbenchmark --image ${image} --algorithm MMS --subspace 10 --ls CG --maxiters ${MAXITERS} --accel --filename ${FNAME}.dat
done < ${DIR}/denoise/jobs/${JOBNAME}.in
