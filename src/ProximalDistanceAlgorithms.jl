module ProximalDistanceAlgorithms

using Distances, Clustering
using LinearAlgebra, Statistics, SparseArrays, IterativeSolvers
using Convex
using RecipesBase
using CSV, DataFrames

import IterativeSolvers: CGStateVariables
import LinearMaps
import LinearMaps: LinearMap, AdjointMap, TransposeMap

# algorithm types
abstract type AlgorithmOption end

"""Use steepest descent iteration."""
struct SteepestDescent <: AlgorithmOption end

"""Use algorithm map implied by distance majorization"""
struct MM <: AlgorithmOption end

"""Use ADMM"""
struct ADMM <: AlgorithmOption end

"""
Build a portable representation of a problem using Convex.jl.
The problem can be passed to a supported black-box solver.
See the Convex.jl documentation for more details.
"""
struct BlackBox <: AlgorithmOption end

export SteepestDescent, MM, BlackBox, ADMM

# convergence metrics + common operations
include("common.jl")
include("acceleration.jl")

export initialize_history, instantiate_fusion_matrix
export slow_schedule, fast_schedule

# example: convex regression
include(joinpath("convex_regression", "linear_operators.jl"))
include(joinpath("convex_regression", "steepest_descent.jl"))
include(joinpath("convex_regression", "mm.jl"))
include(joinpath("convex_regression", "black_box.jl"))
include(joinpath("convex_regression", "utilities.jl"))

export cvxreg_fit, cvxreg_example, mazumder_standardization

# example: metric nearness problem
include(joinpath("metric_nearness", "linear_operators.jl"))
include(joinpath("metric_nearness", "ADMM.jl"))
include(joinpath("metric_nearness", "steepest_descent.jl"))
include(joinpath("metric_nearness", "mm.jl"))
include(joinpath("metric_nearness", "utilities.jl"))

export metric_projection, metric_example, MetricFM

# example: convex clustering
include(joinpath("convex_clustering", "linear_operators.jl"))
include(joinpath("convex_clustering", "steepest_descent.jl"))
include(joinpath("convex_clustering", "utilities.jl"))

export convex_clustering, convex_clustering_path,
    gaussian_weights, knn_weights, gaussian_cluster, assign_classes,
    convex_clustering_data

# example: total variation image denoising
include(joinpath("image_denoising", "steepest_descent.jl"))
include(joinpath("image_denoising", "linear_operators.jl"))

export image_denoise, prox_l1_ball!, prox_l2_ball!

end # module
