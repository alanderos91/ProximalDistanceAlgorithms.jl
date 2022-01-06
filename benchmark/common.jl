using ProximalDistanceAlgorithms
using Random, StableRNGs
using CSV, DataFrames, DelimitedFiles

# make sure we use only use 8 threads for OpenBLAS/MKL
using LinearAlgebra#, MKL
BLAS.set_num_threads(8)

if !isdir("results") mkdir("results") end

function get_outer_view(trace)
    n = length(trace.iteration)
    start_indices = findall(isequal(0), trace.iteration)
    end_indices = Int[1]
    group = Int[]
    
    for (i, index) in enumerate(start_indices)
        if i == length(start_indices)
            foreach(ii -> push!(group, i), index:n)
            push!(end_indices, n)       # final outer iteration
        else
            foreach(ii -> push!(group, i), index:start_indices[i+1]-1)
            i > 1 && push!(end_indices, index-1)
        end
    end
    @views outer_trace = (
        iteration=collect(1:length(end_indices)),
        inner_iterations=trace.iteration[end_indices],
        loss=trace.loss[end_indices],
        distance=trace.distance[end_indices],
        objective=trace.objective[end_indices],
        gradient=trace.gradient[end_indices],
        rho=trace.rho,
    )

    return outer_trace, group
end

do_nothing(df::DataFrame) = nothing
do_nothing(r::Int, sol) = nothing
const DEFAULT = do_nothing
const SAVEDIR = "results"

function parse_options()
    algorithms = []
    for arg in ARGS
        if arg == "MM"
            push!(algorithms, MM())
        elseif arg == "SD"
            push!(algorithms, SteepestDescent())
        elseif arg == "ADMM"
            push!(algorithms, ADMM())
        elseif arg == "Hybrid"
            push!(algorithms, SDADMM())
        elseif arg == "MMS3"
            push!(algorithms, MMSubSpace(3))
        elseif arg == "MMS5"
            push!(algorithms, MMSubSpace(5))
        elseif arg == "MMS10"
            push!(algorithms, MMSubSpace(10))
        end
    end
    return algorithms
end

function summarize_options(filename, algorithm, options)
    open(filename, "w+") do io
        write(io, "algorithm = $(typeof(algorithm))\n")
        for (key, value) in pairs(options)
            if key == :penalty
                rho = value(1.0, 1)
                write(io, "rho = $(rho)\n")
            else
                write(io, "$(key) = $(value)\n")
            end
        end
    end;
end

function benchmark(basename, algorithm, f::F, nreplicates, options,
    df_callback::CB1=DEFAULT,
    sol_callback::CB2=DEFAULT,) where{F,CB1,CB2}
    # get filenames for every output
    summary_filename = joinpath(SAVEDIR, basename*"-summary.txt")
    history_filename = joinpath(SAVEDIR, basename*"-convergence.dat")
    benchmark_filename = joinpath(SAVEDIR, basename*"-benchmark.dat")

    # generate a summary so we don't forget settings
    summarize_options(summary_filename, algorithm, options)

    # generate convergence history
    print("Extracting convergence history... ")
    history, logger = initialize_history(1000, 1)
    options_with_logger = (;options..., callback=logger)
    @time f(algorithm, options_with_logger)
    println()

    # save convergence history to file
    println("Saving convergence history to:\n  $(history_filename)")
    outer_trace, outer_indicator = get_outer_view(history)
    if isempty(history.rho)
        history_df = DataFrame(
            algorithm = string(typeof(algorithm)),
            outer     = outer_indicator,
            inner     = history.iteration,
            loss      = history.loss,
            distance  = history.distance,
            objective = history.objective,
            gradient  = history.gradient,
        )
    else
        history_df = DataFrame(
            algorithm = string(typeof(algorithm)),
            outer     = outer_indicator,
            inner     = history.iteration,
            loss      = history.loss,
            distance  = history.distance,
            objective = history.objective,
            gradient  = history.gradient,
            rho       = [history.rho[i] for i in outer_indicator],
        )
    end
    df_callback(history_df)
    CSV.write(history_filename, history_df)
    println()

    # run the benchmark
    println("Starting benchmark...")
    cpu_time  = Vector{Float64}(undef, nreplicates)
    memory    = Vector{Float64}(undef, nreplicates)
    for r = 1:nreplicates
        print("    collecting sample ($(r)/$(nreplicates))... ")
        @time begin
            # run algorithm
            result = @timed f(algorithm, options)
            solution = result.value

            # save results
            cpu_time[r] = result.time         # seconds
            memory[r]   = result.bytes / 1e6  # MB

            # handle additional details in solutions
            sol_callback(r, solution)
        end
    end
    println()

    # save benchmark results to file
    println("Saving results to:\n  $(benchmark_filename)")
    total_outer = maximum(outer_trace.iteration)
    total_inner = sum(outer_trace.inner_iterations)
    benchmark_df = DataFrame(
        algorithm = string(typeof(algorithm)),
        time      = cpu_time,
        memory    = memory,
        outer     = total_outer,
        inner     = total_inner,
    )
    df_callback(benchmark_df)
    CSV.write(benchmark_filename, benchmark_df)
    println()

    return nothing
end
