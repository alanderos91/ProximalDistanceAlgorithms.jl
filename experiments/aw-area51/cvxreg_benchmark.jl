using ArgParse
using ProximalDistanceAlgorithms
using LinearAlgebra

global const DIR = joinpath(pwd(), "experiments", "aw-area51", "cvxreg")

# loads common interface + packages
include("common.jl")

# command line interface
function cvxreg_interface(args)
    options = ArgParseSettings(
        prog = "Convex Regression Benchmark",
        description = "Benchmarks proximal distance algorithm on convex regression problem"
    )

    @add_arg_table! options begin
        "--features"
            help     = "number of features in data"
            arg_type = Int
            required = true
        "--samples"
            help     = "number of samples in data"
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
function cvxreg_instance(options)
    d = options["features"]
    n = options["samples"]

    y, y_truth, X = cvxreg_example(x -> dot(x,x), d, n, 0.1)
    y, X = mazumder_standardization(y, X)
    problem = (y = y, X = X)
    problem_size = (d, n,)

    println("    Convex Regression; $(d) features, $(n) samples\n")

    return problem, problem_size
end

# inlined wrapper
@inline function run_cvxreg(algorithm, problem; kwargs...)
    cvxreg_fit(algorithm, problem.y, problem.X; kwargs...)
end

# run the benchmark
interface     = cvxreg_interface
run_solver    = run_cvxreg
make_instance = cvxreg_instance

run_benchmark(interface, run_solver, make_instance, ARGS)