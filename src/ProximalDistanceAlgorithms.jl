module ProximalDistanceAlgorithms

using LinearAlgebra

include(joinpath("convex_regression", "linear_operators.jl"))
include(joinpath("convex_regression", "proxdist.jl"))
export cvxreg_fit

end # module
