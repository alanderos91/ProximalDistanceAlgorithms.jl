"""
```
cvxclst_fusion_matrix(d, n)
```

Construct a block matrix `D` such that `Dblock * vec(A) == A[:,i] - A[:,j]`.
The result `D` is compatible with `vec(A)`, where `A` is `d` by `n`.

Blocks are stacked in dictionary order.
For example, if `n = 3` then the blocks are ordered `(2,1), (3,1), (3,2)`.
"""
function cvxclst_fusion_matrix(d, n)
    Idn = I(n)  # n by n identity matrix
    Idd = I(d)  # d by d identity matrix

    # standard basis vectors in R^n
    e = [Idn[:,i] for i in 1:n]

    # Construct blocks for fusion matrix.
    # The blocks (i,j) are arranged in dictionary order.
    # Loop order is crucial to keeping blocks in dictionary order.
    Dblock = [kron((e[i] - e[j])', Idd) for j in 1:n for i in j+1:n]
    D = sparse(vcat(Dblock...))

    return D
end

"""
```
cvxclst_apply_fusion_matrix!(Y, U)
```
"""
function cvxclst_apply_fusion_matrix!(Y, U)
    d, n = size(U)
    for j in 1:n, i in j+1:n
        l = tri2vec(i, j, n)
        for k in 1:d
            Y[k,l] = U[k,i] - U[k,j]
        end
    end

    return Y
end

"""
```
cvxclst_apply_fusion_matrix(U)
```
"""
function cvxclst_apply_fusion_matrix(U)
    d, n = size(U)
    Y = zeros(d, binomial(n,2))

    cvxclst_apply_fusion_matrix!(Y, U)

    return y
end

"""
```
cvxclst_apply_fusion_matrix!(Q, Y)
```
"""
function cvxclst_apply_fusion_matrix_transpose!(Q, Y)
    d, n = size(Q)

    for j in 1:n, i in j+1:n
        l = tri2vec(i, j, n)
        @simd for k in 1:d
            @inbounds Q[k,i] += Y[k,l]
        end
        @simd for k in 1:d
            @inbounds Q[k,j] -= Y[k,l]
        end
    end

    return Q
end

"""
```
cvxclst_apply_fusion_matrix(Y)
```
"""
function cvxclst_apply_fusion_matrix_transpose(Y)
    n = size(W, 1)
    m = length(y)
    d = m ÷ binomial(n, 2)

    Q = zeros(d, n)

    cvxclst_apply_fusion_matrix_transpose!(Q, Y)

    return Q
end

"""
```
__evaluate_weighted_gradient_norm(Q)
```
"""
function __evaluate_weighted_gradient_norm(Q)
    d, n = size(Q)
    val = zero(eltype(Q))

    for j in 1:n, i in j+1:n
        # @views δ_ij = SqEuclidean()(Q[:,i], Q[:,j])
        for k in 1:d
            δ_ijk = Q[k,i] - Q[k,j]
            val = val + δ_ijk^2
        end
    end

    return val
end
