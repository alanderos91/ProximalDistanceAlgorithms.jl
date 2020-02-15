module ProximalDistanceAlgorithms

using LinearAlgebra
using Convex

include(joinpath("convex_regression", "linear_operators.jl"))
include(joinpath("convex_regression", "proxdist.jl"))
include(joinpath("convex_regression", "convex_wrapper.jl"))

export cvxreg_fit

end # module
