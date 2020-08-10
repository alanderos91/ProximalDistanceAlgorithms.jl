using ArgParse
using ProximalDistanceAlgorithms
using Clustering
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
        "--rho"
            help     = "initial value for penalty coefficient"
            arg_type = Float64
            default  = 1.0
        "--mu"
            help     = "initial value for step size in ADMM"
            arg_type = Float64
            default  = 1.0
        "--step"
            help     = "step size for path heuristic"
            arg_type = Float64
            default  = 0.05
        "--start"
            help     = "initial sparsity level"
            arg_type = Float64
            default  = 0.5
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
    X, classes, nclasses = convex_clustering_data(file)

    # problem dimensions
    d, n = size(X)

    # create weights
    W = gaussian_weights(X, phi = 0.1)

    problem = (W = W, X = X, classes = classes)
    problem_size = (d = d, n = n, nclasses = nclasses)

    println("    Convex Clustering; $(d) features, $(n) samples, $(nclasses) classes\n")

    return problem, problem_size
end

function cvxcluster_save_results(file, problem, problem_size, solution, cpu_time, memory)
    # save benchmark results
    df = DataFrame(
            features = problem_size.d,
            samples  = problem_size.n,
            classes  = problem_size.nclasses,
            cpu_time = cpu_time,
            memory   = memory
        )
    CSV.write(file, df)

    # get filename without extension
    basefile = splitext(file)[1]
    basefile = splitext(file)[1]

    # save assignments
    open(basefile * "_assignment.out", "w") do io
        writedlm(io, ["nu" "classes" "assignment"])
        for (assignment, s) in zip(solution.assignment, solution.sparsity)
            nclasses = length(unique(assignment))
            writedlm(io, [s nclasses assignment...])
        end
    end

    # save validation metrics
    open(basefile * "_validation.out", "w") do io
        writedlm(io, ["sparsity" "classes" "VI" "ARI" "NMI"])
        for (assignment, s) in zip(solution.assignment, solution.sparsity)

            # compare assignments against truth
            VI  = Clustering.varinfo(problem.classes, assignment)
            ARI = Clustering.randindex(problem.classes, assignment)[1]
            NMI = Clustering.mutualinfo(problem.classes, assignment, normed = true)
            nclasses = length(unique(assignment))
            writedlm(io, [sparsity nclasses VI ARI NMI])
        end
    end

    return nothing
end

# inlined wrapper
@inline function run_cvxcluster(algorithm, problem, options; kwargs...)
    kw = Dict(kwargs)
    ρ0 = kw[:rho]
    st = options["start"]
    sz = options["step"]
    penalty(ρ, n) = min(1e6, ρ0 * 1.2 ^ floor(n/20))

    convex_clustering_path(algorithm, problem.W, problem.X;
        penalty=penalty,
        start=st,
        stepsize=sz, kwargs...)
end

# run the benchmark
interface     = cvxcluster_interface
run_solver    = run_cvxcluster
make_instance = cvxcluster_instance
save_results  = cvxcluster_save_results

run_benchmark(interface, run_solver, make_instance, save_results, ARGS)
