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
    ${JLDIR}/bin/julia --project=${PKG} ${DIR}/benchmark_metric.jl "$@";
}

# redirect all output to a randomly generated log file
PREFIX=$(date +"%Y-%m-%d")
LOG_FILE=$(mktemp ${DIR}/metric/logs/${PREFIX}-${JOBNAME}-XXXXXX)
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
MAXITERS=3000

# each algorithm, except ADMM, should be run with Nesterov acceleration
while read n
    do
    # MM
    FNAME=MM_${n}
    jlbenchmark --nodes ${n} --algorithm MM --maxiters ${MAXITERS} --accel --filename ${FNAME}.dat

    # Steepest Descent
    FNAME=SD_${n}
    jlbenchmark --nodes ${n} --algorithm SD --maxiters ${MAXITERS} --accel --filename ${FNAME}.dat

    # ADMM
    FNAME=ADMM_${n}
    jlbenchmark --nodes ${n} --algorithm ADMM --maxiters ${MAXITERS} --filename ${FNAME}.dat

    # MM Subspace{5}
    FNAME=MMS5_LSQR_${n}
    jlbenchmark --nodes ${n} --algorithm MMS --subspace 5 --ls LSQR --maxiters ${MAXITERS} --accel --filename ${FNAME}.dat

    FNAME=MMS5_CG_${n}
    jlbenchmark --nodes ${n} --algorithm MMS --subspace 5 --ls CG --maxiters ${MAXITERS} --accel --filename ${FNAME}.dat

    # MM Subspace{10}
    FNAME=MMS10_LSQR_${n}
    jlbenchmark --nodes ${n} --algorithm MMS --subspace 10 --ls LSQR --maxiters ${MAXITERS} --accel --filename ${FNAME}.dat

    FNAME=MMS10_CG_${n}
    jlbenchmark --nodes ${n} --algorithm MMS --subspace 10 --ls CG --maxiters ${MAXITERS} --accel --filename ${FNAME}.dat

    # SD + ADMM hybrid
    FNAME=SDADMM_${n}
    jlbenchmark --nodes ${n} --algorithm SDADMM --accel --maxiters ${MAXITERS} \
        --filename ${FNAME}.dat \
        --atol 1e-12
        
done < ${DIR}/metric/jobs/${JOBNAME}.in
