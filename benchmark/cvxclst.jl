
using Statistics, Clustering
include("common.jl")

const examples = ("iris", "zoo", "gaussian300", "spiral500")
const fileprefix = "cvxclst"
const nreplicates = 3
const common_options = (;
    init_sparsity = 0.0,
    stepsize = 1e-2,
    magnitude= -4,
    nouter   = 100,
    ninner   = 10^4,
    penalty  = (ρ, n) -> 1.2*ρ,
    accel    = Val(:nesterov),
    delay    = 0,
    gtol     = 1e-2,
    dtol     = 1e-4,
    rtol     = 1e-6,
)

algorithms = parse_options()

for example in examples
    # Load dataset
    X, true_classes, number_classes = convex_clustering_data(example*".dat")
    μ, σ = mean(X, dims=2), std(X, dims=2)
    X .= (X .- μ) ./ σ
    W = gaussian_weights(X, phi=0.0)
    d, n = size(X)

    # callback to calling solver
    f = function(algorithm, options)
        convex_clustering_path(algorithm, W, X; options...)
    end
    
    # create array to store MSE in each replicate
    VI = Vector{Float64}(undef, nreplicates)
    ARI = Vector{Float64}(undef, nreplicates)
    NMI = Vector{Float64}(undef, nreplicates)

    # callback to append problem size to DataFrames
    cb1 = function(df)
        df[!, :dataset] .= example
        df[!, :samples] .= n
        df[!, :features] .= d
        df[!, :classes] .= number_classes
        if :time in propertynames(df)
            # benchmark results
            df[!, :VI] = VI
            df[!, :ARI] = ARI
            df[!, :NMI] = NMI
            select!(df, [:dataset, :features, :samples, :classes, :algorithm, :time, :memory, :outer, :inner, :VI, :ARI, :NMI])
        else
            # convergence history
            select!(df, [:dataset, :features, :samples, :classes, :algorithm, :outer, :inner, :loss, :distance, :objective, :gradient])
        end
        return df
    end

    # callback to record MSE inside replicates loop
    cb2 = function(basename, r, solution)
        # Record results in DataFrame
        df = DataFrame(
            dataset=example,
            samples=n,
            features=d,
            classes=number_classes,
            replicate=r,
            sparsity=solution.sparsity,
            clusters=solution.number_classes,
            VI=map(x -> Clustering.varinfo(x, true_classes), solution.assignment),
            ARI=map(x -> Clustering.randindex(x, true_classes)[1], solution.assignment),
            NMI=map(x -> Clustering.mutualinfo(x, true_classes, normed=true), solution.assignment),
            assignment=solution.assignment,
        )
        filename = joinpath(SAVEDIR, basename*"-rep=$(r).dat")
        CSV.write(filename, df)

        # Record performance metrics
        VI[r] = minimum(df.VI)      # VI = 0 is perfect; [0, log(n)]
        ARI[r] = maximum(df.ARI)    # ARI = 1 is perfect; [-1, 1]
        NMI[r] = maximum(df.NMI)    # NMI = 1 is perfect; [0, 1]

        return nothing
    end

    for algorithm in algorithms
        if algorithm isa SteepestDescent
            # Other methods
            basename = fileprefix * "-$(example)" * "-$(typeof(algorithm))"
            options = (; common_options...)
            benchmark(basename, algorithm, f, nreplicates, options, cb1, (r,sol)->cb2(basename,r,sol))
        else
            # method w/ LSMR; fusion matrix is extremely ill-conditioned
            basename = fileprefix * "-$(example)" * "-$(typeof(algorithm))" * "-LSMR"
            options_with_LSMR = (; common_options..., ls=Val(:LSMR))
            benchmark(basename, algorithm, f, nreplicates, options_with_LSMR, cb1, (r,sol)->cb2(basename,r,sol))            
        end
    end
end
