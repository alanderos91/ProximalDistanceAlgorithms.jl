using Test, SparseArrays, LinearAlgebra, Random
using ProximalDistanceAlgorithms
using MKL

ProxDist = ProximalDistanceAlgorithms

# helper functions for testing fusion matrices

function D_mul_x(D, x, y, z)
    @time mul!(z, D, x)
end

function Dt_mul_z(D, x, y, z)
    @time mul!(y, D', z)
end

function DtD_mul_x(D, x, y, z)
    DtD = D'D
    @time mul!(y, DtD, x)
end

function get_test_string(f)
    if occursin("DtD_mul", string(f))
        str = "y = DtD*x"
    elseif occursin("D_mul", string(f))
        str = "z = D*x  "
    elseif occursin("Dt_mul", string(f))
        str = "x = Dt*z "
    end
    return str
end

function get_op_string(A)
    if A isa ProxDist.FusionMatrix
        str = "LinearMap   "
    elseif A isa DenseMatrix
        str = "DenseMatrix "
    else
        str = "SparseMatrix"
    end

    return str
end

function print_info(f, A)
    str1 = get_test_string(f)
    str2 = get_op_string(A)
    print("$(str1), $(str2)  ")
end

# fusion matrices
include("convex_regression.jl")
include("convex_clustering.jl")
include("metric_nearness.jl")
include("image_denoising.jl")
include("condition_number.jl")

# projections
include("projections.jl")
