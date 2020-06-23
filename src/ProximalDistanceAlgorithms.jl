module ProximalDistanceAlgorithms

using Distances, Clustering
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
include(joinpath("convex_regression", "operators.jl"))
include(joinpath("convex_regression", "steepest_descent.jl"))
include(joinpath("convex_regression", "MM.jl"))
include(joinpath("convex_regression", "black_box.jl"))
include(joinpath("convex_regression", "utilities.jl"))

export cvxreg_fit, cvxreg_example, mazumder_standardization
export CvxRegBlockA, CvxRegBlockB, CvxRegFM

# example: metric nearness problem
include(joinpath("metric_nearness", "operators.jl"))
include(joinpath("metric_nearness", "ADMM.jl"))
include(joinpath("metric_nearness", "steepest_descent.jl"))
include(joinpath("metric_nearness", "MM.jl"))
include(joinpath("metric_nearness", "utilities.jl"))

export metric_projection, metric_example
export MetricFM, MetricFGM

# example: convex clustering
include(joinpath("convex_clustering", "operators.jl"))
include(joinpath("convex_clustering", "steepest_descent.jl"))
include(joinpath("convex_clustering", "MM.jl"))
include(joinpath("convex_clustering", "utilities.jl"))

export convex_clustering, convex_clustering_path, convex_clustering_data
export gaussian_weights, knn_weights, gaussian_cluster, assign_classes
export CvxClusterFM

# example: total variation image denoising
include(joinpath("image_denoising", "operators.jl"))
include(joinpath("image_denoising", "steepest_descent.jl"))
include(joinpath("image_denoising", "MM.jl"))
include(joinpath("image_denoising", "utilities.jl"))

export image_denoise
export ImgTvdFM

end # module
