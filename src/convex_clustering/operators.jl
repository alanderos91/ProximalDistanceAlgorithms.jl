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
function apply_fusion_matrix!(z, D::CvxClusterFM, x)
    d, n, M = D.d, D.n, D.M
    lidx1 = LinearIndices((1:d,1:n)) # for x
    lidx2 = LinearIndices((1:d,1:M)) # for z
    tidx = TriVecIndices(n)

    @inbounds for j in 1:n, i in j+1:n
        l = tidx[i,j]
        @inbounds for k in 1:d
            ki = lidx1[k,i]
            kj = lidx1[k,j]
            kl = lidx2[k,l]
            z[kl] = x[ki] - x[kj]
        end
    end

    return z
end

function apply_fusion_matrix_transpose!(x, D::CvxClusterFM, z)
    d, n, M = D.d, D.n, D.M
    lidx1 = LinearIndices((1:d,1:n)) # for x
    lidx2 = LinearIndices((1:d,1:M)) # for z
    tidx = TriVecIndices(n)

    for j in 1:n, i in j+1:n
        l = tidx[i,j]
        for k in 1:d
            ki = lidx1[k,i]
            kl = lidx2[k,l]
            @inbouds x[ki] += z[kl]
        end
        for k in 1:d
            kj = lidx1[k,j]
            kl = lidx2[k,l]
            @inbouds x[kj] -= z[kl]
        end
    end
end

"""
```
instantiate_fusion_matrix(D::CvxClusterFM)
```

Construct a block matrix `D` such that `Dblock * vec(A) == A[:,i] - A[:,j]`.
The result `D` is compatible with `vec(A)`, where `A` is `d` by `n`.

Blocks are stacked in dictionary order.
For example, if `n = 3` then the blocks are ordered `(2,1), (3,1), (3,2)`.
"""
function instantiate_fusion_matrix(D::CvxClusterFM{T}) where {T<:Number}
    Idn = I(n)  # n by n identity matrix
    Idd = I(d)  # d by d identity matrix

    # standard basis vectors in R^n
    e = [Idn[:,i] for i in 1:n]

    # Construct blocks for fusion matrix.
    # The blocks (i,j) are arranged in dictionary order.
    # Loop order is crucial to keeping blocks in dictionary order.
    Dblock = [kron((e[i] - e[j])', Idd) for j in 1:n for i in j+1:n]
    S = sparse(vcat(Dblock...))

    return S
end

# """
# ```
# __evaluate_weighted_gradient_norm(Q)
# ```
# """
# function __evaluate_weighted_gradient_norm(Q)
#     d, n = size(Q)
#     val = zero(eltype(Q))
#
#     for j in 1:n, i in j+1:n
#         # @views δ_ij = SqEuclidean()(Q[:,i], Q[:,j])
#         for k in 1:d
#             δ_ijk = Q[k,i] - Q[k,j]
#             val = val + δ_ijk^2
#         end
#     end
#
#     return val
# end
