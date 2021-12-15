struct CvxClusterFM{T} <: FusionMatrix{T}
    d::Int
    n::Int
    M::Int
    N::Int
end

# constructors

function CvxClusterFM{T}(d::Integer, n::Integer) where {T<:Number}
    M = d*binomial(n, 2)
    N = d*n

    return CvxClusterFM{T}(d, n, M, N)
end

# default eltype to Int64
CvxClusterFM(d::T, n::T) where T<:Integer = CvxClusterFM{Int}(d, n)
CvxClusterFM(d::T, n::T, M::T, N::T) where T<:Integer = CvxClusterFM{Int}(d, n, M, N)

# implementation
Base.size(D::CvxClusterFM) = (D.M, D.N)

"""
```
cvxclst_apply_fusion_matrix!(Y, U)
```
"""
function LinearAlgebra.mul!(z::AbstractVecOrMat, D::CvxClusterFM, x::AbstractVector)
    d, n, = D.d, D.n

    # copy terms to be subtracted
    offset = 0
    for i in 1:n-1
        for j in 1:n-i, k in 1:d
            @inbounds z[offset+d*(j-1)+k] = -x[d*(i-1)+k]
        end
        offset += d*(n-i)
    end

    # walk along off-diagonal
    offset = 0
    for i in 1:n-1
        for j in 1:n-i, k in 1:d
            @inbounds z[offset+d*(j-1)+k] += x[d*(i-1)+d*j+k]
        end
        offset += d*(n-i)
    end

    return z
end

function LinearAlgebra.mul!(x::AbstractVecOrMat, Dt::TransposeMap{<:Any,<:CvxClusterFM}, z::AbstractVector)
    D = Dt.lmap
    d, n, = D.d, D.n

    # initialize for accumulation
    fill!(x, 0)

    offset = 0
    for i in 1:n-1
        # apply with block with subtractions
        for j in 1:n-i, k in 1:d
            @inbounds x[d*(i-1)+k] -= z[offset+d*(j-1)+k]
        end

        # apply block with additions
        for j in 1:n-i, k in 1:d
            @inbounds x[d*(i-1)+d*j+k] += z[offset+d*(j-1)+k]
        end

        offset += d*(n-i)
    end

    return x
end

# overrides for _unsafe_mul!; see https://github.com/Jutho/LinearMaps.jl/issues/157
LinearMaps._unsafe_mul!(y::AbstractVecOrMat, Dt::TransposeMap{<:Any,<:CvxClusterFM}, x::AbstractVector) = mul!(y, Dt, x)

struct CvxClusterFGM{T} <: FusionGramMatrix{T}
    d::Int  # number of features
    n::Int  # number of samples
    N::Int  # total number of variables
end

# constructors
function CvxClusterFGM{T}(d::Integer, n::Integer) where T<:Number
    return CvxClusterFGM{T}(d, n, d*n)
end

# default to Int64
CvxClusterFGM(d, n) = CvxClusterFGM{Int}(d, n)

# implementation
Base.size(DtD::CvxClusterFGM) = (DtD.N, DtD.N)

function LinearAlgebra.mul!(y::AbstractVecOrMat, DtD::CvxClusterFGM, x::AbstractVector)
    d, n = DtD.d, DtD.n

    for j in 1:n, k in 1:d
        s = zero(eltype(y))

        # apply terms below diagonal
        for i in 1:j-1
            @inbounds s -= x[d*(i-1)+k]
        end

        @inbounds s += (n-1)*x[d*(j-1)+k]

        # apply terms above diagonal
        for i in j+1:n
            @inbounds s -= x[d*(i-1)+k]
        end

        @inbounds y[d*(j-1)+k] = s
    end

    return y
end

Base.:(*)(Dt::TransposeMap{T,CvxClusterFM{T}}, D::CvxClusterFM{T}) where T = CvxClusterFGM{T}(D.d, D.n, D.N)
