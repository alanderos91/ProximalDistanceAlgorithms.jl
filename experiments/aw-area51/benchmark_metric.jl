using ArgParse
using ProximalDistanceAlgorithms

global const DIR = joinpath(pwd(), "experiments", "aw-area51", "metric")

# loads common interface + packages
include("common.jl")

# command line interface
function metric_projection_interface(args)
    options = ArgParseSettings(
        prog = "Metric Benchmark",
        description = "Benchmarks proximal distance algorithm on metric projection problem"
    )

    @add_arg_table! options begin
        "--nodes"
            help     = "nodes in dissimilarity matrix"
            arg_type = Int
            required = true
        "--algorithm"
            help     = "choice of algorithm"
            arg_type = Symbol
            required = true
        "--subspace"
            help     = "subspace size for MMS methods"
            arg_type = Int
            default  = 3
        "--ls"
            help     = "choice of linear solver"
            arg_type = Symbol
            default  = :LSQR
        "--maxiters"
            help     = "maximum iterations"
            arg_type = Int
            default  = 1000
        "--nsamples"
            help     = "samples from @timed."
            arg_type = Int
            default  = 10
        "--accel"
            help     = "toggles Nesterov acceleration"
            action   = :store_true
        "--rtol"
            help     = "relative tolerance on loss"
            arg_type = Float64
            default  = 1e-6
        "--atol"
            help     = "absolute tolerance on distance"
            arg_type = Float64
            default  = 1e-6
        "--seed"
            help     = "problem randomization seed"
            arg_type = Int64
            default  = 5357
        "--filename"
            help     = "base file name"
            arg_type = String
            default  = ""
    end

    return parse_args(options)
end

# create a problem instance from command-line arguments
function metric_projection_instance(options)
    n = options["nodes"]

    W, Y = metric_example(n, weighted = false)
    problem = (W = W, Y = Y)
    problem_size = (n = n,)

    println("    Metric Projection; $(n) nodes\n")

    return problem, problem_size
end

# inlined wrapper
@inline function run_metric_projection(algorithm, problem; kwargs...)
    penalty(ρ, n) = min(1e6, 1.09 ^ floor(n/20))

    X = metric_projection(algorithm, problem.Y; penalty = penalty, kwargs...)

    return (X = X,)
end

function metric_save_results(file, problem, problem_size, solution, cpu_time, memory)
    # save benchmark results
    df = DataFrame(
            nodes    = problem_size.n,
            cpu_time = cpu_time,
            memory   = memory
        )
    CSV.write(file, df)

    # get filename without extension
    basefile = splitext(file)[1]

    # save input
    save_array(basefile * ".in", problem.Y)

    # save solution
    save_array(basefile * ".out", solution.X)

    return nothing
end

# run the benchmark
interface     = metric_projection_interface
run_solver    = run_metric_projection
make_instance = metric_projection_instance
save_results  = metric_save_results

run_benchmark(interface, run_solver, make_instance, save_results, ARGS)
