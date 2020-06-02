using Test, SparseArrays, LinearAlgebra
using ProximalDistanceAlgorithms

ProxDist = ProximalDistanceAlgorithms

# helper functions for testing fusion matrices

function D_mul_x(D, x, y, z)
    mul!(z, D, x)
end

function Dt_mul_z(D, x, y, z)
    mul!(y, D', z)
end

function DtD_mul_x(D, x, y, z)
    mul!(y, D'D, x)
end

function get_test_string(f)
    if f === D_mul_x
        str = "z = D*x  "
    elseif f === Dt_mul_z
        str = "x = Dt*z "
    elseif f === DtD_mul_x
        str = "y = DtD*x"
    end
    return str
end

function get_op_string(A)
    if A isa ProxDist.FusionMatrix
        str = "LinearMap   "
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

# include("convex_regression.jl")
# include("convex_clustering.jl")
include("metric_nearness.jl")
# include("image_denoising.jl")
