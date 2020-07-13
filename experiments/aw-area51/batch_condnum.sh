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
    ${JLDIR}/bin/julia --project=${PKG} ${DIR}/benchmark_condnum.jl "$@";
}

# redirect all output to a randomly generated log file
PREFIX=$(date +"%Y-%m-%d")
LOG_FILE=$(mktemp ${DIR}/condnum/logs/${PREFIX}-${JOBNAME}-XXXXXX)
exec 1>${LOG_FILE} # redirect STDOUT
exec 2>&1          # redirect STDERR to STDOUT

# add header
echo $(date)
echo "Condition Number Benchmarks"
echo
echo "benchmark:        ${JOBNAME}"
echo "Julia project:    ${PKG}"
echo "scripts:          ${DIR}"
echo

# set maximum number of iterations
MAXITERS=5000

# each algorithm, except ADMM, should be run with Nesterov acceleration
while read probsize
    do
    p=$(cut -d',' -f1 <<< ${probsize})
    percent=$(cut -d',' -f2 <<< ${probsize})

    # MM
    FNAME=MM_${p}_${percent}
    jlbenchmark --p ${p} --percent ${percent} --algorithm MM --ls NA --maxiters ${MAXITERS} --accel --filename ${FNAME}.dat

    # Steepest Descent
    FNAME=SD_${p}_${percent}
    jlbenchmark --p ${p} --percent ${percent} --algorithm SD --maxiters ${MAXITERS} --accel --filename ${FNAME}.dat

    # ADMM
    FNAME=ADMM_${p}_${percent}
    jlbenchmark --p ${p} --percent ${percent} --algorithm ADMM --ls NA --maxiters ${MAXITERS} --filename ${FNAME}.dat

    # # MM Subspace{5}
    # FNAME=MMS5_LSQR_${p}_${percent}
    # jlbenchmark --p ${p} --percent ${percent} --algorithm MMS --subspace 5 --ls LSQR --maxiters ${MAXITERS} --accel --filename ${FNAME}.dat

    FNAME=MMS5_CG_${p}_${percent}
    jlbenchmark --p ${p} --percent ${percent} --algorithm MMS --subspace 5 --ls CG --maxiters ${MAXITERS} --accel --filename ${FNAME}.dat

    # # MM Subspace{10}
    # FNAME=MMS10_LSQR_${p}_${percent}
    # jlbenchmark --p ${p} --percent ${percent} --algorithm MMS --subspace 10 --ls LSQR --maxiters ${MAXITERS} --accel --filename ${FNAME}.dat

    # FNAME=MMS10_CG_${p}_${percent}
    # jlbenchmark --p ${p} --percent ${percent} --algorithm MMS --subspace 10 --ls CG --maxiters ${MAXITERS} --accel --filename ${FNAME}.dat
done < ${DIR}/condnum/jobs/${JOBNAME}.in
