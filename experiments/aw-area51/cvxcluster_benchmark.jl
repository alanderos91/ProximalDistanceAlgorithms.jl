using ArgParse
using ProximalDistanceAlgorithms
using LinearAlgebra
using CSV, DataFrames

global const DIR = joinpath(pwd(), "experiments", "aw-area51", "cvxcluster")

# loads common interface + packages
include("common.jl")

# command line interface
function cvxcluster_interface(args)
    options = ArgParseSettings(
        prog = "Convex Clustering Benchmark",
        description = "Benchmarks proximal distance algorithm on convex clustering problem"
    )

    @add_arg_table! options begin
        "--data"
            help     = "name of file in data/ directory"
            arg_type = String
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
function cvxcluster_instance(options)
    # read in data as a matrix
    file = options["data"]
    X = cvxcluster_load_data(file)

    # problem dimensions
    d, n = size(X)

    # create weights
    W = ones(n, n)

    problem = (W = W, X = X)
    problem_size = (d, n,)

    println("    Convex Clustering; $(d) features, $(n) samples\n")

    return problem, problem_size
end

function cvxcluster_load_data(file)
    df = CSV.read(joinpath("data", file), copycols = true)
    
end

# inlined wrapper
@inline function run_cvxcluster(algorithm, problem; kwargs...)
    convex_clustering_path(algorithm, problem.W, problem.X; kwargs...)
end

# run the benchmark
interface     = cvxcluster_interface
run_solver    = run_cvxcluster
make_instance = cvxcluster_instance

run_benchmark(interface, run_solver, make_instance, ARGS)
