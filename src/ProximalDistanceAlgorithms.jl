module ProximalDistanceAlgorithms

using LinearAlgebra, Statistics, SparseArrays, IterativeSolvers
using Convex
using RecipesBase

import IterativeSolvers: CGStateVariables

# default penalty schedule
__default_schedule(ρ::Real, iteration::Integer) = ρ
__default_schedule(T, W, n::Integer, ρ::Real, iteration::Integer) = ρ

# default logging behavior
__default_logger(data, iteration) = nothing

# algorithm types
abstract type AlgorithmOption end

"""Use proximal point iteration."""
struct ProximalPoint <: AlgorithmOption end

"""Use steepest descent iteration."""
struct SteepestDescent <: AlgorithmOption end

"""Use algorithm map implied by distance majorization"""
struct MM <: AlgorithmOption end

"""
Build a portable representation of a problem using Convex.jl.
The problem can be passed to a supported black-box solver.
See the Convex.jl documentation for more details.
"""
struct BlackBox <: AlgorithmOption end

export ProximalPoint, SteepestDescent, MM, BlackBox

# example: convex regression
include(joinpath("convex_regression", "linear_operators.jl"))
include(joinpath("convex_regression", "steepest_descent.jl"))
include(joinpath("convex_regression", "mm.jl"))
include(joinpath("convex_regression", "proximal_point.jl"))
include(joinpath("convex_regression", "black_box.jl"))
include(joinpath("convex_regression", "utilities.jl"))

export cvxreg_fit, cvxreg_example, mazumder_standardization

# example: metric nearness problem
include(joinpath("metric_nearness", "linear_operators.jl"))
include(joinpath("metric_nearness", "steepest_descent.jl"))
include(joinpath("metric_nearness", "mm.jl"))
include(joinpath("metric_nearness", "utilities.jl"))

export metric_projection, metric_example

# example: convex clustering
include(joinpath("convex_clustering", "linear_operators.jl"))
include(joinpath("convex_clustering", "steepest_descent.jl"))
include(joinpath("convex_clustering", "utilities.jl"))

export convex_clustering, convex_clustering_path,
    gaussian_weights, knn_weights, gaussian_clusters, get_clustering, count_clusters

# example: total variation image denoising
include(joinpath("image_denoising", "steepest_descent.jl"))

export image_denoise, prox_l1_ball!, prox_l2_ball!

# convergence metrics
include("logging.jl")

export MMLogger, SDLogger

# suggested penalty schedules
include("penalty.jl")

# general utilities
include("utilities.jl")
include("acceleration.jl")

export slow_schedule, fast_schedule, get_acceleration_strategy

end # module
