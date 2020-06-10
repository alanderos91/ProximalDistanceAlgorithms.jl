struct MetricFM{T} <: FusionMatrix{T}
    n::Int # nodes in dissimilarity matrix
    M::Int # rows implied by n
    N::Int # columns implied by n
    I::TriVecIndices # map Cartesian coordinates to trivec index
end

# constructors

function MetricFM{T}(n::Integer) where {T<:Number}
    N = binomial(n, 2)
    M = N * (n-1)

    return MetricFM{T}(n, M, N)
end

function MetricFM{T}(n::Integer, M::Integer, N::Integer) where {T<:Number}
    I = TriVecIndices(n)

    return MetricFM{T}(n, M, N, I)
end

# default eltype to Int64
MetricFM(n::Integer) = MetricFM{Int}(n)
MetricFM(n::Integer, M::Integer, N::Integer) = MetricFM{Int}(n, M, N)

# implementation
Base.size(D::MetricFM) = (D.M, D.N)

"""
```
apply_fusion_matrix!(z, D::MetricFM, x)
```

Multiply `x` by `D = [A, I]`, where `A*x ≥ 0` encodes triangle inequality
constraints and `x ≥ 0` encodes non-negativity.
"""
function apply_fusion_matrix!(z, D::MetricFM, x::SubArray)
    edge = 0 # edge counter
    n = D.n
    # X = trivec_parent(x, n)
    X = parent(x)
    L = LinearIndices((1:n, 1:n))

    # A*x: 0 ≤ x_ik + x_kj - x_ij
    @inbounds for j in 1:n-2, i in j+1:n-1
        a = X[L[i,j]] # fix one edge

        @inbounds for k in i+1:n
            b = X[L[k,i]]
            c = X[L[k,j]]

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

function apply_fusion_matrix!(z, D::MetricFM, x::AbstractVector)
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

function apply_fusion_matrix_transpose!(x::SubArray, D::MetricFM, z)
    edge = 0
    N = size(D, 2)
    n = D.n
    # X = trivec_parent(x, n)
    X = parent(x)
    L = LinearIndices((1:n, 1:n))

    # I*z[block2]
    copyto!(x, 1, z, N*(n-2)+1, N)

    # T'*z[block1]
    @inbounds for j in 1:n-2, i in j+1:n-1, k in i+1:n
        abc = z[edge += 1]
        bac = z[edge += 1]
        cab = z[edge += 1]

        X[L[i,j]] += -abc + bac + cab
        X[L[k,j]] += -cab + abc + bac
        X[L[k,i]] += -bac + abc + cab
    end

    return z
end

function apply_fusion_matrix_transpose!(x::AbstractVector, D::MetricFM, z)
    edge = 0
    N = size(D, 2)
    n = D.n
    T = D.I

    # I*z[block2]
    copyto!(x, 1, z, N*(n-2)+1, N)

    # T'*z[block1]
    @inbounds for j in 1:n-2, i in j+1:n-1, k in i+1:n
        abc = z[edge += 1]
        bac = z[edge += 1]
        cab = z[edge += 1]

        x[T[i,j]] += -abc + bac + cab
        x[T[k,j]] += -cab + abc + bac
        x[T[k,i]] += -bac + abc + cab
    end

    return z
end

function instantiate_fusion_matrix(D::MetricFM{T}) where {T<:Number}
    n = D.n
    nrows = 3*3*binomial(n,3)

    I = zeros(Int, nrows)
    J = zeros(Int, nrows)
    V = zeros(T, nrows)

    edge = 0
    for j in 1:n-2, i in j+1:n-1, k in i+1:n
        # map to linear indices
        s1 = trivec_index(n, i, j)
        s2 = trivec_index(n, k, i)
        s3 = trivec_index(n, k, j)

        edge += 1; __set_row!(I, J, V, s1, s2, s3, edge) # T_ijk
        edge += 1; __set_row!(I, J, V, s2, s1, s3, edge) # T_jik
        edge += 1; __set_row!(I, J, V, s3, s1, s2, edge) # T_kij
    end

    return [sparse(I, J, V); LinearAlgebra.I]
end

function __set_row!(I, J, V, s1, s2, s3, t)
    z = 3*(t-1)+1

    I[z] = t
    J[z] = s1
    V[z] = -1

    I[z+1] = t
    J[z+1] = s2
    V[z+1] = 1

    I[z+2] = t
    J[z+2] = s3
    V[z+2] = 1

    return nothing
end

struct MetricFGM{T} <: FusionGramMatrix{T}
    n::Int      # number of nodes
    N::Int      # size of matrix
    indices::TriVecIndices
end

# constructors
function MetricFGM{T}(n::Integer) where {T<:Number}
    N = binomial(n, 2)
    indices = TriVecIndices(n)
    return MetricFGM{T}(n, N, indices)
end

# default to Float64
MetricFGM(n::Integer) = MetricFGM{Float64}(n)

# implementation
Base.size(DtD::MetricFGM) = (DtD.N, DtD.N)

# function apply_fusion_gram_matrix!(y::AbstractVector, DtD::MetricFGM, x::AbstractVector)
#     n = DtD.n
#     indices = DtD.indices
#
#     # apply I block of D'D
#     y .= x
#
#     # apply T'T block of D'D
#     @inbounds for j in 1:n-2, i in j+1:n-1
#         i1 = indices[i,j]; a = x[i1]
#
#         @inbounds for k in i+1:n
#             i2 = indices[k,i]; b = x[i2]
#             i3 = indices[k,j]; c = x[i3]
#
#             y[i1] += 3*a - b - c
#             y[i2] += 3*b - a - c
#             y[i3] += 3*c - a - b
#         end
#     end
#
#     return y
# end

function apply_fusion_gram_matrix!(y::SubArray, DtD::MetricFGM, x::SubArray)
    n = DtD.n
    # X = trivec_parent(x, n)
    # Y = trivec_parent(y, n)
    X = parent(x)
    Y = parent(y)
    L = LinearIndices((1:n, 1:n))

    # apply I block of D'D
    copyto!(y, 1, x, 1, length(x))

    # apply T'T block of D'D
    for j in 1:n-2, i in j+1:n-1
        ij = L[i,j]; a = X[ij]
        for k in i+1:n
            ki = L[k,i]; b = X[ki]
            kj = L[k,j]; c = X[kj]

            abc = 3*a - b - c
            bac = 3*b - a - c
            cab = 3*c - a - b

            Y[ij] += abc
            Y[ki] += bac
            Y[kj] += cab
        end
    end

    return y
end

Base.:(*)(Dt::TransposeMap{T,MetricFM{T}}, D::MetricFM{T}) where T = MetricFGM{T}(D.n, D.N, TriVecIndices(D.n))
