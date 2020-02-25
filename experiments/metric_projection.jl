# ----- script arguments -----
#
# key: one of :MM or :SD
# n: number of nodes
# maxiters: maximum number of algorithm iterations
# sample_rate: number of entries in convergence log
# ntrials: number of times to run benchmark
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

function run_benchmark(algorithm, n, maxiters, sample_rate, ntrials)
    # create a sample convergence history
    W, D = metric_example(n)
    sample_log = initialize_log(algorithm, maxiters, sample_rate)
    @timed metric_projection(algorithm, W, D,
        maxiters = maxiters,
        ρ_init   = 1.0,
        penalty  = fast_schedule,
        history  = sample_log)

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
        W, D = metric_example(n)

        # create convergence log
        history = initialize_log(algorithm, maxiters, sample_rate)

        # run algorithm
        result = @timed metric_projection(algorithm, W, D,
            maxiters = maxiters,
            ρ_init   = 1.0,
            penalty  = fast_schedule,
            history  = history)

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
            nodes      = n,
            loss       = loss,
            objective  = objective,
            penalty    = penalty,
            gradient   = gradient,
            cpu_time   = cpu_time,
            memory     = memory)

    hf = DataFrame(
            loss      = sample_log.loss,
            objective = sample_log.objective,
            penalty   = sample_log.penalty,
            gradient  = sample_log.g)

    return df, hf
end

# finish processing script arguments
if key == :MM
    algorithm = MM()
elseif key == :SD
    algorithm = SteepestDescent()
else
    error("Unrecognized algorithm option $(key)")
end

prefix = String(key)

# set unique name for problem instance
problem = "nodes_$(n)"

# output files
benchmark_file = joinpath("metric", "benchmarks",
    prefix * "_" * problem * ".dat")

figure_file = joinpath("metric", "figures",
    prefix * "_" * problem * ".dat")

# print benchmark parameters
println("""
[Problem Parameters]
    nodes = $(n)

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
df, hf = run_benchmark(algorithm, n, maxiters, sample_rate, ntrials)

# save benchmark data
CSV.write(benchmark_file, df)

# save convergence history
CSV.write(figure_file, hf)
