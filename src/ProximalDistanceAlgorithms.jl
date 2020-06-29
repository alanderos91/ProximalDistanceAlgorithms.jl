module ProximalDistanceAlgorithms

using Distances, Clustering
using Parameters
using LinearAlgebra, Statistics, SparseArrays, IterativeSolvers
using Convex
using RecipesBase
using CSV, DataFrames

using DataStructures: heapify!, percolate_down!

import IterativeSolvers: CGStateVariables
import LinearMaps
import LinearMaps: LinearMap, AdjointMap, TransposeMap

# algorithm types
abstract type AlgorithmOption end

# traits
needs_gradient(::AlgorithmOption) = false
needs_hessian(::AlgorithmOption) = false
needs_linsolver(::AlgorithmOption) = false

"""Use steepest descent iteration."""
struct SteepestDescent <: AlgorithmOption end

# traits
needs_gradient(::SteepestDescent) = true
needs_hessian(::SteepestDescent) = false
needs_linsolver(::SteepestDescent) = false

"""Use algorithm map implied by distance majorization"""
struct MM <: AlgorithmOption end

# traits
needs_gradient(::MM) = true
needs_hessian(::MM) = true
needs_linsolver(::MM) = true

"""Use ADMM"""
struct ADMM <: AlgorithmOption end

# traits
needs_gradient(::ADMM) = true
needs_hessian(::ADMM) = true
needs_linsolver(::ADMM) = true

"""
Build a portable representation of a problem using Convex.jl.
The problem can be passed to a supported black-box solver.
See the Convex.jl documentation for more details.
"""
struct BlackBox <: AlgorithmOption end

export SteepestDescent, MM, BlackBox, ADMM

# convergence metrics + common operations
include("common.jl")
include("optimize.jl")
include("acceleration.jl")

export initialize_history, instantiate_fusion_matrix
export slow_schedule, fast_schedule

# example: convex regression
include(joinpath("convex_regression", "operators.jl"))
include(joinpath("convex_regression", "implementation.jl"))

export cvxreg_fit, cvxreg_example, mazumder_standardization
export CvxRegBlockA, CvxRegBlockB, CvxRegFM

# example: metric nearness problem
include(joinpath("metric_nearness", "operators.jl"))
include(joinpath("metric_nearness", "implementation.jl"))

export metric_projection, metric_example
export MetricFM, MetricFGM

# example: convex clustering
include(joinpath("convex_clustering", "operators.jl"))
include(joinpath("convex_clustering", "implementation.jl"))

export convex_clustering, convex_clustering_path, convex_clustering_data
export gaussian_weights, knn_weights, gaussian_cluster, assign_classes
export CvxClusterFM

# example: total variation image denoising
include(joinpath("image_denoising", "operators.jl"))
include(joinpath("image_denoising", "implementation.jl"))

export denoise_image
export ImgTvdFM

# example: improving condition number
include(joinpath("condition_number", "operators.jl"))
include(joinpath("condition_number", "implementation.jl"))

export reduce_cond
export CondNumFM

end # module
