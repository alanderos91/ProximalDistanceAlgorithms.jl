using ArgParse
using ProximalDistanceAlgorithms
using LinearAlgebra, Statistics

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
        "--subspace"
            help     = "subspaze size for MMS methods"
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
            default  = 1e-4
        "--rho"
            help     = "initial value for penalty coefficient"
            arg_type = Float64
            default  = 1.0
        "--mu"
            help     = "initial value for step size in ADMM"
            arg_type = Float64
            default  = 1.0
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
    problem = (y = y, X = X, y_truth = y_truth)
    problem_size = (d = d, n = n,)

    println("    Convex Regression; $(d) features, $(n) samples\n")

    return problem, problem_size
end

# inlined wrapper
@inline function run_cvxreg(algorithm, problem, options; kwargs...)
    kw = Dict(kwargs)
    ρ0 = kw[:rho]
    penalty(ρ, n) = min(1e6, ρ0 * 1.75 ^ floor(n/100))

    # extra processing step for hybrid algorithm
    if algorithm isa SDADMM
        maxiters = kw[:maxiters]
        phase1 = round(Int, 2/3 * maxiters) # 2/3 iterations allocated for SD
        phase2 = round(Int, 1/3 * maxiters) # 1/3 iterations allocated for ADMM
        
        θ, ξ = cvxreg_fit(algorithm, problem.y, problem.X;
            phase1=phase1,
            phase2=phase2,
            penalty=penalty,
            kwargs...)
    else
        θ, ξ = cvxreg_fit(algorithm, problem.y, problem.X;
            penalty=penalty,
            kwargs...)
    end

    return (θ = θ, ξ = ξ)
end

function cvxreg_save_results(file, problem, problem_size, solution, cpu_time, memory)
    # compute mean squared error with respect to ground truth
    MSE = mean((solution.θ .- problem.y_truth) .^ 2)

    # save benchmark results
    df = DataFrame(
            features = problem_size.d,
            samples  = problem_size.n,
            cpu_time = cpu_time,
            memory   = memory,
            MSE      = MSE
        )
    CSV.write(file, df)

    # get filename without extension
    basefile = splitext(file)[1]

    # save input
    save_array(basefile * "_y.in", problem.y)
    save_array(basefile * "_truth.in", problem.y_truth)
    save_array(basefile * "_X.in", problem.X)

    # save solution
    save_array(basefile * "_theta.out", solution.θ)
    save_array(basefile * "_xi.out", solution.ξ)

    return nothing
end

# run the benchmark
interface     = cvxreg_interface
run_solver    = run_cvxreg
make_instance = cvxreg_instance
save_results  = cvxreg_save_results

run_benchmark(interface, run_solver, make_instance, save_results, ARGS)
