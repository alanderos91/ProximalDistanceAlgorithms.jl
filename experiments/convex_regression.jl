# ----- script arguments -----
#
# key: one of :MM or :SD
# d: number of covariates
# n: number of samples
# maxiters: maximum number of algorithm iterations
# sample_rate: number of entries in convergence log
# ntrials: number of times to run benchmark
# sigma: standard deviation of perturbations
#

using ProximalDistanceAlgorithms, LinearAlgebra
using Random, DataFrames, CSV

# utilities
function initialize_log(::MM, maxiters, sample_rate)
    return MMLogger(maxiters ÷ sample_rate, sample_rate)
end

function initialize_log(::SteepestDescent, maxiters, sample_rate)
    return SDLogger(maxiters ÷ sample_rate, sample_rate)
end

# Example 1: squared 2-norm
φ(x) = dot(x, x)

function run_benchmark(φ, algorithm, d, n, maxiters, sample_rate, ntrials, σ, accel)
    # create a sample convergence history
    y, y_truth, X = cvxreg_example(φ, d, n, σ)
    y_scaled, X_scaled = mazumder_standardization(y, X)
    sample_log = initialize_log(algorithm, maxiters, sample_rate)
    @timed cvxreg_fit(algorithm, y_scaled, X_scaled,
        maxiters = maxiters,
        ρ_init   = 1.0,
        penalty  = fast_schedule,
        history  = sample_log,
        accel    = accel)

    # benchmark data
    loss      = Vector{Float64}(undef, ntrials)
    objective = Vector{Float64}(undef, ntrials)
    penalty   = Vector{Float64}(undef, ntrials)
    gradient  = Vector{Float64}(undef, ntrials)
    cpu_time  = Vector{Float64}(undef, ntrials)
    memory    = Vector{Float64}(undef, ntrials)

    # create and benchmark multiple problem instances in the same class
    for k = 1:ntrials
        # simulate data
        y, y_truth, X = cvxreg_example(φ, d, n, σ)

        # standardize according to mazumder
        y_scaled, X_scaled = mazumder_standardization(y, X)

        # create convergence log
        history = initialize_log(algorithm, maxiters, sample_rate)

        # run algorithm
        result = @timed cvxreg_fit(algorithm, y_scaled, X_scaled,
            maxiters = maxiters,
            ρ_init   = 1.0,
            penalty  = fast_schedule,
            history  = history,
            accel    = accel)

        # record benchmark data
        loss[k]      = history.loss[end]
        objective[k] = history.objective[end]
        penalty[k]   = history.penalty[end]
        gradient[k]  = history.penalty[end]
        cpu_time[k]  = result[2]         # seconds
        memory[k]    = result[3] / 1e6   # MB
    end

    # save results in DataFrame
    df = DataFrame(
            covariates = d,
            samples    = n,
            loss       = loss,
            objective  = objective,
            penalty    = penalty,
            gradient   = gradient,
            cpu_time   = cpu_time,
            memory     = memory)

    hf = DataFrame(
            iteration = sample_log.iteration,
            loss      = sample_log.loss,
            objective = sample_log.objective,
            penalty   = sample_log.penalty,
            gradient  = sample_log.g)

    return df, hf
end

# finish processing script arguments
σ = sigma

if key == :MM
    algorithm = MM()
elseif key == :SD
    algorithm = SteepestDescent()
else
    error("Unrecognized algorithm option $(key)")
end

prefix = String(key)

# set unique name for problem instance
problem = "d_$(d)_n_$(n)_sigma_$(σ)"

# output files
fname = "$(prefix)_$(problem)_$(strategy).dat"
benchmark_file = joinpath("cvxreg", "benchmarks", fname)
figure_file = joinpath("cvxreg", "figures", fname)

# print benchmark parameters
println("""
[Problem Parameters]
    covariates  = $(d)
    samples     = $(n)
    std. dev.   = $(σ)

[Benchmark Settings]
    algorithm   = $(algorithm)
    maxiters    = $(maxiters)
    sample_rate = $(sample_rate)
    ntrials     = $(ntrials)

[Output]
    benchmark   = $(benchmark_file)
    figure file = $(figure_file)
""")

# run the benchmark
Random.seed!(seed)
df, hf = run_benchmark(φ, algorithm, d, n, maxiters, sample_rate, ntrials, σ)

# save benchmark data
CSV.write(benchmark_file, df)

# save convergence history
CSV.write(figure_file, hf)
