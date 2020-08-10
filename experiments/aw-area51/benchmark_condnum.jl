using ArgParse
using ProximalDistanceAlgorithms
using LinearAlgebra, MatrixDepot

global const DIR = joinpath(pwd(), "experiments", "aw-area51", "condnum")

# loads common interface + packages
include("common.jl")

fidelity(A, B) = 100 * sum(1 .- abs.(sign.(A) .- sign.(B))) / length(B)

function condnum_interface(args)
    options = ArgParseSettings(
        prog = "Condition Number Benchmark",
        description = "Benchmarks proximal distance algorithm for projection that reduces the condition number of a matrix"
    )

    @add_arg_table! options begin
        "--p"
            help     = "defines size of smaller dimension, i.e. number of singular values"
            arg_type = Int
            required = true
        "--percent"
            help     = "percent reduction in condition number"
            arg_type = Float64
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

function condnum_instance(options)
    p = options["p"]
    α = options["percent"]

    M = matrixdepot("randcorr", p)
    F = svd(M)
    c = (1-α) * cond(M)

    problem = (M = M, F = F, α = α, c = c)
    problem_size = (p = p,)

    println("    Reduce Condition Number; $(p) singular values\n")

    return problem, problem_size
end

@inline function run_condnum(algorithm, problem, options; kwargs...)
    kw = Dict(kwargs)
    ρ0 = kw[:rho]

    penalty(ρ, n) = min(1e6, ρ0 * 1.1 ^ floor(n/20))

    output = reduce_cond(algorithm, problem.c, problem.F; penalty = penalty, kwargs...)

    return (X = output,)
end

function condnum_save_results(file, problem, problem_size, solution, cpu_time, memory)
    # save benchmark results
    df = DataFrame(
            p = problem_size.p,
            α = problem.α,
            cpu_time = cpu_time,
            memory   = memory,
            c = problem.c,
            condM = cond(problem.M),
            condX = cond(solution.X),
            fidelity = fidelity(solution.X, problem.M)
        )
    CSV.write(file, df)

    # get filename without extension
    basefile = splitext(file)[1]

    # save input
    save_array(basefile * ".in", problem.M)

    # save solution
    save_array(basefile * ".out", solution.X)

    return nothing
end

# run the benchmark
interface     = condnum_interface
run_solver    = run_condnum
make_instance = condnum_instance
save_results  = condnum_save_results

run_benchmark(interface, run_solver, make_instance, save_results, ARGS)
