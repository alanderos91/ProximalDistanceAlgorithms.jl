struct MetricFM{T} <: FusionMatrix{T}
    n::Int # nodes in dissimilarity matrix
    M::Int # rows implied by n
    N::Int # columns implied by n
    I::TriVecIndices
    tmpv::Vector{Float64}
end

# constructors

function MetricFM{T}(n::Integer) where {T<:Number}
    N = binomial(n, 2)
    M = N * (n-1)
    I = TriVecIndices(n)
    tmpv = zeros(n)
    return MetricFM{T}(n, M, N, I, tmpv)
end

# default eltype to Int64
MetricFM(n::Integer) = MetricFM{Int}(n)

# implementation
Base.size(D::MetricFM) = (D.M, D.N)

function LinearAlgebra.mul!(z::AbstractVecOrMat, D::MetricFM, x::AbstractVector)
    edge = 0 # edge counter
    n = D.n
    T = D.I

    # A*x: 0 ≤ x_ik + x_kj - x_ij
    @inbounds for j in 1:n-2, i in j+1:n-1
        a = x[T[i,j]] # fix one edge

        @inbounds for k in i+1:n
            b = x[T[k,i]]
            c = x[T[k,j]]

            # check edges of one triangle
            z[edge += 1] = -a + b + c
            z[edge += 1] = -b + a + c
            z[edge += 1] = -c + a + b
        end
    end

    # I*x = x
    copyto!(z, edge+1, x, 1, length(x))

    return z
end

function LinearAlgebra.mul!(x::AbstractVecOrMat, Dt::TransposeMap{<:Any,<:MetricFM}, z::AbstractVector)
    edge = 0
    D = Dt.lmap
    N = size(D, 2)
    n = D.n
    I = D.I

    # I*z[block2]
    copyto!(x, 1, z, N*(n-2)+1, N)

    # T'*z[block1]
    @inbounds for j in 1:n-2, i in j+1:n-1, k in i+1:n
        abc = z[edge += 1]
        bac = z[edge += 1]
        cab = z[edge += 1]

        x[I[i,j]] += -abc + bac + cab
        x[I[k,j]] += -cab + abc + bac
        x[I[k,i]] += -bac + abc + cab
    end

    return x
end

struct MetricFGM{T} <: FusionGramMatrix{T}
    n::Int      # number of nodes
    N::Int      # size of matrix
    tmpv::Vector{Float64}
end

# constructors
function MetricFGM{T}(n::Integer) where {T<:Number}
    N = binomial(n, 2)
    tmpv = zeros(n)
    return MetricFGM{T}(n, N, tmpv)
end

# default to Float64
MetricFGM(n::Integer) = MetricFGM{Float64}(n)

# implementation
Base.size(DtD::MetricFGM) = (DtD.N, DtD.N)

function LinearAlgebra.mul!(y::AbstractVecOrMat, DtD::MetricFGM, x::AbstractVector)
    @unpack n, tmpv = DtD

    # clear sums and update cache, v = M'*x
    __update_DtD_cache__!(tmpv, x, n)

    # complete mat-vec operation: (T'T + I)*x = 3(n-1) x - M*M'*x
    idx = 1
    for j in 1:n
        @inbounds tmpvj = tmpv[j]
        @simd for i in j+1:n
            @inbounds y[idx] = 3*(n-1) * x[idx] - (tmpv[i] + tmpvj)
            idx += 1
        end
    end

    return y
end

function __update_DtD_cache__!(tmpv, x, n)
    idx = 1
    fill!(tmpv, 0)
    for j in 1:n, i in j+1:n
        @inbounds tmpv[j] += x[idx]
        @inbounds tmpv[i] += x[idx]
        idx += 1
    end
    return tmpv
end

Base.:(*)(Dt::TransposeMap{T,MetricFM{T}}, D::MetricFM{T}) where T = MetricFGM{T}(D.n, D.N, D.tmpv)

#
# (I + ρ D'D)⁻¹ = a * I + a*b*ρ * M*M' + 4*a*b*c*ρ² * ones(m,m)
#
# M*M' = (3*n-4) I - T'T = 3*(n-1) I - D'D
# a = inv(3*(n-1)*ρ+1)
# b = inv((2*n-1)*ρ+1)
# c = inv((n-1)*ρ+1)
#
function LinearAlgebra.ldiv!(y, H::ProxDistHessian{T,matT1,matT2}, x) where {T,matT1<:UniformScaling,matT2<:MetricFGM}
    # pull fusion gram matrix and other parameters
    @unpack DtD, ρ = H
    @unpack n, tmpv = DtD

    # define constants
    a = inv(3*(n-1)*ρ+1)
    b = inv((2*n-1)*ρ+1)
    c = inv((n-1)*ρ+1)

    c1 = a
    c2 = a * b * ρ
    c3 = 4 * sum(x) * a * b * c * abs2(ρ)

    # clear sums and update cache, v = M'*x
    __update_DtD_cache__!(tmpv, x, n)

    # complete mat-vec operation: (T'T + I)*x = 3(n-1) x - M*M'*x
    idx = 1
    for j in 1:n
        tmpvj = tmpv[j]
        @simd for i in j+1:n
            @inbounds y[idx] = c1 * x[idx] + c2 * (tmpv[i] + tmpvj) + c3
            idx += 1
        end
    end

    return y
end
