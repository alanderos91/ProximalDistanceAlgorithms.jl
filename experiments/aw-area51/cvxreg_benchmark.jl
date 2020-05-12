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
            default  = 1e-4
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
    problem_size = (d = d, n = n,)

    println("    Convex Regression; $(d) features, $(n) samples\n")

    return problem, problem_size
end

# inlined wrapper
@inline function run_cvxreg(algorithm, problem; kwargs...)
    # min(1e4, ρ_init * (1.5)^(floor(50 / n)))
    rho_schedule(ρ, iteration) = min(1e4, iteration % 50 == 0 ? 1.5*ρ : ρ)

    cvxreg_fit(algorithm, problem.y, problem.X; penalty = rho_schedule, kwargs...)
end

function cvxreg_save_results(file, problem, problem_size, solution, cpu_time, memory)
    # save benchmark results
    df = DataFrame(
            features = problem_size.d,
            samples  = problem_size.n,
            cpu_time = cpu_time,
            memory   = memory
        )
    CSV.write(file, df)

    # get filename without extension
    basefile = splitext(file)[1]

    # save input
    CSV.write(basefile * "_y.in", Tables.table(problem.y))
    CSV.write(basefile * "_X.in", Tables.table(problem.y))
    
    # save solution
    CSV.write(basefile * "_theta.out", Tables.table(solution[1]))
    CSV.write(basefile * "_xi.out", Tables.table(solution[2]))

    return nothing
end

# run the benchmark
interface     = cvxreg_interface
run_solver    = run_cvxreg
make_instance = cvxreg_instance
save_results  = cvxreg_save_results

run_benchmark(interface, run_solver, make_instance, save_results, ARGS)
