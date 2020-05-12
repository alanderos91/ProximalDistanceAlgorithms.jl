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
        "--ftol"
            help     = "tolerance for loss"
            arg_type = Float64
            default  = 1e-6
        "--dtol"
            help     = "tolerance for distance"
            arg_type = Float64
            default  = 1e-5
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

    W, D = metric_example(n, weighted = false)
    problem = (W = W, D = D)
    problem_size = (n,)

    println("    Metric Projection; $(n) nodes\n")

    return problem, problem_size
end

# inlined wrapper
@inline function run_metric_projection(algorithm, problem; kwargs...)
    metric_projection(algorithm, problem.W, problem.D; kwargs...)
end

# run the benchmark
interface     = metric_projection_interface
run_solver    = run_metric_projection
make_instance = metric_projection_instance

run_benchmark(interface, run_solver, make_instance, ARGS)