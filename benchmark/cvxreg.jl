
using Statistics
include("common.jl")

const number_samples = (50, 100, 200, 400)
const number_features = (1, 2, 10, 20)
const fileprefix = "cvxreg"
const nreplicates = 3
const common_options = (;
    nouter   = 200,
    ninner   = 10^4,
    penalty  = (ρ, n) -> 1.2*ρ,
    accel    = Val(:nesterov),
    delay    = 0,
    gtol     = 1e-3,
    dtol     = 1e-2,
    rtol     = 1e-6,
)

algorithms = parse_options()

for n in number_samples, d in number_features
    # Simulate problem instance
    rng = StableRNG(1903)
    y, y_truth, X = cvxreg_example(x -> dot(x, x), d, n, 0.1)
    
    # normalization
    X_scaling = Diagonal([1 / norm(col) for col in eachcol(X)])
    y_scaling = 1 / norm(y)

    X_scaled = X * X_scaling
    y_scaled = y * y_scaling

    # callback to calling solver
    f = function(algorithm, options)
        cvxreg_fit(algorithm, y_scaled, X_scaled; options...)
    end
    
    # create array to store MSE in each replicate
    mse = Vector{Float64}(undef, nreplicates)

    # callback to append problem size to DataFrames
    cb1 = function(df)
        df[!, :samples] .= n
        df[!, :features] .= d
        if :time in propertynames(df)
            # benchmark results
            df[!, :mse] .= mse
            select!(df, [:features, :samples, :algorithm, :time, :memory, :outer, :inner, :mse])
        else
            # convergence history
            select!(df, [:features, :samples, :algorithm, :outer, :inner, :loss, :distance, :objective, :gradient, :rho])
        end
        return df
    end

    # callback to record MSE inside replicates loop
    cb2 = function(r, solution)
        θ, _ = solution
        θ_rescaled = θ ./ y_scaling
        mse[r] = mean((θ_rescaled .- y_truth) .^ 2)
        return nothing
    end

    for algorithm in algorithms
        if algorithm isa SteepestDescent
            # Other methods
            basename = fileprefix * "-d=$(d)-n=$(n)" * "-$(typeof(algorithm))"
            options = (; common_options...)
            benchmark(basename, algorithm, f, nreplicates, options, cb1, cb2)
        else
            # method w/ LSQR
            basename = fileprefix * "-d=$(d)-n=$(n)" * "-$(typeof(algorithm))" * "-LSQR"
            options_with_LSQR = (; common_options..., ls=Val(:LSQR))
            benchmark(basename, algorithm, f, nreplicates, options_with_LSQR, cb1, cb2)

            # method w/ CG
            basename = fileprefix * "-d=$(d)-n=$(n)" * "-$(typeof(algorithm))" * "-CG"
            options_with_CG = (; common_options..., ls=Val(:CG))
            benchmark(basename, algorithm, f, nreplicates, options_with_CG, cb1, cb2)            
        end
    end
end
