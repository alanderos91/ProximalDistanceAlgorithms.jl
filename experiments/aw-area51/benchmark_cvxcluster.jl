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
function cvxcluster_instance(options)
    # read in data as a matrix
    file = options["data"]
    X, classes, k = convex_clustering_data(file)

    # problem dimensions
    d, n = size(X)

    # create weights
    W = ones(n, n)

    problem = (W = W, X = X, classes = classes)
    problem_size = (d = d, n = n, k = k)

    println("    Convex Clustering; $(d) features, $(n) samples\n")

    return problem, problem_size
end

function cvxcluster_save_results(file, problem, problem_size, solution, cpu_time, memory)
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

    # find k-means solution
    kmeans_clustering = kmeans(problem.X, problem_size.k, maxiter = 2000)

    # save cluster assignments + validation metrics
    open(basefile * ".out", "w") do io
        for (U, ν) in zip(solution.U, solution.ν)
            # get cluster assignments
            _, assignment, k = assign_classes(U)

            # compare assignments against truth
            vi1  = Clustering.varinfo(problem.classes, assignment)
            ari1 = Clustering.randindex(problem.classes, assignment)[1]

            # compare assignments against k-means
            vi2  = Clustering.varinfo(kmeans_clustering, assignment)
            ari2 = Clustering.randindex(kmeans_clustering, assignment)[1]

            writedlm(io, [ν k vi1 ari1 vi2 ari2 assignment...])
        end
    end

    return nothing
end

# inlined wrapper
@inline function run_cvxcluster(algorithm, problem; kwargs...)
    # min(1e4, ρ_init * (1.25)^(floor(50 / n)))
    rho_schedule(ρ, iteration) = min(1e4, iteration % 50 == 0 ? 1.5*ρ : ρ)

    convex_clustering_path(algorithm, problem.W, problem.X; penalty = rho_schedule, kwargs...)
end

# run the benchmark
interface     = cvxcluster_interface
run_solver    = run_cvxcluster
make_instance = cvxcluster_instance
save_results  = cvxcluster_save_results

run_benchmark(interface, run_solver, make_instance, save_results, ARGS)
