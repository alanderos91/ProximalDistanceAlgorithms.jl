# ----- script arguments -----
#
# key: one of :MM or :SD
# strategy: one of :none or :nesterov
# n: number of nodes
# maxiters: maximum number of algorithm iterations
# sample_rate: number of entries in convergence log
# ntrials: number of times to run benchmark
#

using Dates
using ProximalDistanceAlgorithms, LinearAlgebra
using Random, DataFrames, CSV

# utilities
function initialize_log(::MM, maxiters, sample_rate)
    return MMLogger(maxiters ÷ sample_rate, sample_rate)
end

function initialize_log(::SteepestDescent, maxiters, sample_rate)
    return SDLogger(maxiters ÷ sample_rate, sample_rate)
end

# penalty function
rho_schedule(ρ, iteration) = iteration % 50 == 0 ? ρ*2.0 : ρ

function rho_schedule(T, W, n, ρ, iteration)
    if iteration % 50 == 0
        ρ_new = 2.0 * ρ
        for i in 1:n
            diff = W[i,i] / ρ_new - W[i,i] / ρ
            T[i,i] = T[i,i] + diff
        end
        ρ = ρ_new
    end

    return ρ
end

function run_benchmark(algorithm, n, maxiters, sample_rate, ntrials, accel)
    # create a sample convergence history
    W, D = metric_example(n, weighted = true)
    sample_log = initialize_log(algorithm, maxiters, sample_rate)
    @timed metric_projection(algorithm, W, D,
        maxiters = maxiters,
        ρ_init   = 1.0,
        penalty  = rho_schedule,
        history  = sample_log,
        accel    = accel)

    # benchmark data
    loss      = Vector{Float64}(undef, ntrials)
    objective = Vector{Float64}(undef, ntrials)
    penalty   = Vector{Float64}(undef, ntrials)
    gradient  = Vector{Float64}(undef, ntrials)
    cpu_time  = Vector{Float64}(undef, ntrials)
    memory    = Vector{Float64}(undef, ntrials)
    stepsize  = Vector{Float64}(undef, ntrials)

    # create and benchmark multiple problem instances in the same class
    for k = 1:ntrials
        # simulate data
        W, D = metric_example(n, weighted = true)

        # create convergence log
        history = initialize_log(algorithm, maxiters, sample_rate)

        # run algorithm
        result = @timed metric_projection(algorithm, W, D,
            maxiters = maxiters,
            ρ_init   = 1.0,
            penalty  = rho_schedule,
            history  = history,
            accel    = accel)

        # record benchmark data
        loss[k]      = history.loss[end]
        objective[k] = history.objective[end]
        penalty[k]   = history.penalty[end]
        gradient[k]  = history.g[end]
        stepsize[k]  = history.γ[end]
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
            stepsize   = stepsize,
            cpu_time   = cpu_time,
            memory     = memory)

    hf = DataFrame(
            iteration = sample_log.iteration,
            loss      = sample_log.loss,
            objective = sample_log.objective,
            penalty   = sample_log.penalty,
            gradient  = sample_log.g,
            stepsize  = sample_log.γ)

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
fname = "$(prefix)_$(problem)_$(strategy).dat"
benchmark_file = joinpath("metric", "benchmarks", fname)
figure_file = joinpath("metric", "figures", fname)

# print benchmark parameters
println("""
########## nodes = $(n) ##########

[Benchmark Settings]
    algorithm   = $(algorithm)
    accel.      = $(strategy)
    maxiters    = $(maxiters)
    sample_rate = $(sample_rate)
    ntrials     = $(ntrials)
    seed        = $(seed)

[Output]
    benchmark   = $(benchmark_file)
    figure file = $(figure_file)""")

# run the benchmark
Random.seed!(seed)
benchmark_time = @elapsed begin
    df, hf = run_benchmark(algorithm, n, maxiters, sample_rate, ntrials, Val(strategy))
end

# save benchmark data
CSV.write(benchmark_file, df)

# save convergence history
CSV.write(figure_file, hf)

println("""
    elapsed     = $(round(benchmark_time, sigdigits=6)) s
""")
