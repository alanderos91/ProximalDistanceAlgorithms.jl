module ProximalDistanceAlgorithms

using LinearAlgebra, Printf
using Convex

# default penalty schedule
__default_schedule(ρ, iteration) = ρ

# default logging behavior
__default_logger(data, iteration) = nothing

# algorithm types
abstract type AlgorithmOption end

"""Use proximal point iteration."""
struct ProximalPoint <: AlgorithmOption end

"""Use steepest descent iteration."""
struct SteepestDescent <: AlgorithmOption end

"""
Build a portable representation of a problem using Convex.jl.
The problem can be passed to a supported black-box solver.
See the Convex.jl documentation for more details.
"""
struct BlackBox <: AlgoritimOption end

include(joinpath("convex_regression", "linear_operators.jl"))
include(joinpath("convex_regression", "steepest_descent.jl"))
include(joinpath("convex_regression", "proximal_point.jl"))
include(joinpath("convex_regression", "utilities.jl"))

export cvxreg_fit, cvxreg_example, mazumder_standardization

end # module
