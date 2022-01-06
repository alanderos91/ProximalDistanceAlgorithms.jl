
using MatrixDepot
include("common.jl")

const matrix_sizes = (10, 100, 1000)
const factors = (2, 4, 16, 32)
const fileprefix = "condnum"
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

fidelity(A, B) = 100 * sum(1 .- abs.(sign.(A) .- sign.(B))) / length(B)

algorithms = parse_options()

for p in matrix_sizes, α in factors
    # Simulate problem instance. Want condition number to be an order of magnitude greater than dimension.
    Random.seed!(5357)
    M = matrixdepot("randcorr", p)
    condM = cond(M)
    while log10(condM) < log10(p) + 1
        M = matrixdepot("randcorr", p)
        condM = cond(M)
    end

    F = svd(M)
    c = condM / α

    # callback to calling solver
    f = function(algorithm, options)
        reduce_cond(algorithm, c, F; options...)
    end
    
    # allocate worker array for assessing solution quality
    condX = Vector{Float64}(undef, nreplicates)
    fi = Vector{Float64}(undef, nreplicates)

    # callback to append problem size to DataFrame
    cb1 = function(df)
        df[!,:p] .= p
        df[!,:a] .= α
        if :time in propertynames(df)
            df[!,:condM] .= condM
            df[!,:condX] .= condX
            df[!,:fidelity] .= fi
            select!(df, [:p, :a, :condM, :condX, :fidelity, :algorithm, :time, :memory, :outer, :inner])
        else
            select!(df, [:p, :a, :algorithm, :outer, :inner, :loss, :distance, :objective, :gradient, :rho])
        end
        return df
    end

    # callback to record MSE inside replicates loop
    cb2 = function(r, solution)
        X = solution
        condX[r] = cond(X)
        fi[r] = fidelity(M, X)
        return nothing
    end

    for algorithm in algorithms
        if algorithm isa MMSubSpace
            # MMSubSpace w/ LSQR
            basename = fileprefix * "-p=$(p)-a=$(α)" * "-$(typeof(algorithm))" * "-LSQR"
            options_with_LSQR = (; common_options..., ls=Val(:LSQR))
            benchmark(basename, algorithm, f, nreplicates, options_with_LSQR, cb1, cb2)

            # MMSubSpace w/ CG
            basename = fileprefix * "-p=$(p)-a=$(α)" * "-$(typeof(algorithm))" * "-CG"
            options_with_CG = (; common_options..., ls=Val(:CG))
            benchmark(basename, algorithm, f, nreplicates, options_with_CG, cb1, cb2)
        else
            # Other methods
            basename = fileprefix * "-p=$(p)-a=$(α)" * "-$(typeof(algorithm))"
            options = (; common_options...)
            benchmark(basename, algorithm, f, nreplicates, options, cb1, cb2)
        end
    end
end
