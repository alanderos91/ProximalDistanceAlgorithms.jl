
include("common.jl")

const problem_sizes = (16, 32, 64, 128, 256)
const fileprefix = "metric"
const nreplicates = 3
const common_options = (;
    nouter   = 200,
    ninner   = 10^5,
    penalty  = (ρ, n) -> 1.2*ρ,
    accel    = Val(:nesterov),
    delay    = 0,
    gtol     = 1e-3,
    dtol     = 1e-2,
    rtol     = 1e-6,
)

algorithms = parse_options()

for N in problem_sizes
    # Simulate problem instance
    rng = StableRNG(1903)
    _, X = metric_example(N, rng=rng)
    
    # callback to calling solver
    f = function(algorithm, options)
        metric_projection(algorithm, X; options...)
    end
    
    # callback to append problem size to DataFrame
    cb = function(df)
        df[!,:nodes] .= N
        if :time in propertynames(df)
            select!(df, [:nodes, :algorithm, :time, :memory, :outer, :inner])
        else
            select!(df, [:nodes, :algorithm, :outer, :inner, :loss, :distance, :objective, :gradient, :rho])
        end
        return df
    end

    for algorithm in algorithms
        if algorithm isa MMSubSpace
            # MMSubSpace w/ LSMR
            basename = fileprefix * "-$(N)" * "-$(typeof(algorithm))" * "-LSMR"
            options_with_LSMR = (; common_options..., ls=Val(:LSMR))
            benchmark(basename, algorithm, f, nreplicates, options_with_LSMR, cb)

            # MMSubSpace w/ LSQR
            basename = fileprefix * "-$(N)" * "-$(typeof(algorithm))" * "-LSQR"
            options_with_LSQR = (; common_options..., ls=Val(:LSQR))
            benchmark(basename, algorithm, f, nreplicates, options_with_LSQR, cb)

            # MMSubSpace w/ CG
            basename = fileprefix * "-$(N)" * "-$(typeof(algorithm))" * "-CG"
            options_with_CG = (; common_options..., ls=Val(:CG))
            benchmark(basename, algorithm, f, nreplicates, options_with_CG, cb)
        else
            # Other methods
            basename = fileprefix * "-$(N)" * "-$(typeof(algorithm))"
            options = (; common_options...)
            benchmark(basename, algorithm, f, nreplicates, options, cb)
        end
    end
end
