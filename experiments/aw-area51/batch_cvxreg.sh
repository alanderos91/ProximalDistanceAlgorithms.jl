#!/usr/bin/env zsh

# extract jobname which contains information for problem sizes
JOBNAME=${1}

# set Julia package directory
PKG=${HOME}/Projects/ProximalDistanceAlgorithms

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
MAXITERS=3000

# each algorithm, except ADMM, should be run with Nesterov acceleration
while read probsize
    do
    d=$(cut -d',' -f1 <<< ${probsize})
    n=$(cut -d',' -f2 <<< ${probsize})

    # MM
    FNAME=MM_LSQR_${d}_${n}
    jlbenchmark --features ${d} --samples ${n} --algorithm MM --ls LSQR --maxiters ${MAXITERS} --accel --filename ${FNAME}.dat --atol 1e-6

    FNAME=MM_CG_${d}_${n}
    jlbenchmark --features ${d} --samples ${n} --algorithm MM --ls CG --maxiters ${MAXITERS} --accel --filename ${FNAME}.dat --atol 1e-6

    # Steepest Descent
    FNAME=SD_${d}_${n}
    jlbenchmark --features ${d} --samples ${n} --algorithm SD --maxiters ${MAXITERS} --accel --filename ${FNAME}.dat --atol 1e-6

    # ADMM
    FNAME=ADMM_LSQR_${d}_${n}
    jlbenchmark --features ${d} --samples ${n} --algorithm ADMM --ls LSQR --maxiters ${MAXITERS} --filename ${FNAME}.dat --atol 1e-6

    FNAME=ADMM_CG_${d}_${n}
    jlbenchmark --features ${d} --samples ${n} --algorithm ADMM --ls CG --maxiters ${MAXITERS} --filename ${FNAME}.dat --atol 1e-6

    # MM Subspace{5}
    FNAME=MMS5_LSQR_${d}_${n}
    jlbenchmark --features ${d} --samples ${n} --algorithm MMS --subspace 5 --ls LSQR --maxiters ${MAXITERS} --accel --filename ${FNAME}.dat --atol 1e-6

    FNAME=MMS5_CG_${d}_${n}
    jlbenchmark --features ${d} --samples ${n} --algorithm MMS --subspace 5 --ls CG --maxiters ${MAXITERS} --accel --filename ${FNAME}.dat --atol 1e-6

    # MM Subspace{10}
    FNAME=MMS10_LSQR_${d}_${n}
    jlbenchmark --features ${d} --samples ${n} --algorithm MMS --subspace 10 --ls LSQR --maxiters ${MAXITERS} --accel --filename ${FNAME}.dat --atol 1e-6
    
    FNAME=MMS10_CG_${d}_${n}
    jlbenchmark --features ${d} --samples ${n} --algorithm MMS --subspace 10 --ls CG --maxiters ${MAXITERS} --accel --filename ${FNAME}.dat --atol 1e-6

    # SD + ADMM hybrid
    FNAME=SDADMM_CG_${d}_${n}
    jlbenchmark --features ${d} --samples ${n} --algorithm SDADMM --accel --ls CG --maxiters ${MAXITERS} \
        --filename ${FNAME}.dat \
        --atol 0.0
done < ${DIR}/cvxreg/jobs/${JOBNAME}.in
