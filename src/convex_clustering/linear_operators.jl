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
cvxclst_apply_fusion_matrix!(y, W, U)
```

Multiply `vec(U)` by `W*D` and store the result in `y`.
"""
function cvxclst_apply_fusion_matrix!(y, W, U)
    d, n = size(U)

    idx = 1
    for j in 1:n, i in j+1:n, k in 1:d
        y[idx] = W[i,j] * (U[k,i] - U[k,j])
        idx += 1
    end

    return y
end

"""
```
cvxclst_apply_fusion_matrix(W, U)

Multiply `vec(U)` by `W*D`.
```
"""
function cvxclst_apply_fusion_matrix(W, U)
    d, n = size(U)
    y = zeros(eltype(U), d*binomial(n,2))

    cvxclst_apply_fusion_matrix!(y, W, U)

    return y
end

"""
```
cvxclst_apply_fusion_matrix!(Q, W, y)
```

Multiply `y` by `D'*W` and store the result in `Q`.
"""
function cvxclst_apply_fusion_matrix_transpose!(Q, W, y)
    d, n = size(Q)

    for j in 1:n, i in j+1:n
        l = tri2vec(i, j, n)
        start = d*(l-1)+1
        stop  = d*l
        block = start:stop

        for (k, idx) in enumerate(block)
            Q[k,i] += W[i,j] * y[idx]
            Q[k,j] -= W[i,j] * y[idx]
        end
    end

    return y
end

"""
```
cvxclst_apply_fusion_matrix!(W, y)
```

Multiply `y` by `D'*W` and return the result in matrix form.
"""
function cvxclst_apply_fusion_matrix_transpose(W, y)
    n = size(W, 1)
    m = length(y)
    d = m ÷ binomial(n, 2)

    Q = zeros(d, n)

    cvxclst_apply_fusion_matrix_transpose!(Q, W, y)

    return Q
end

"""
```
__evaluate_weighted_gradient_norm(W, Q)
```

Evaluate the norm squared of `W*D*vec(Q)`.
"""
function __evaluate_weighted_gradient_norm(W, Q)
    d, n = size(Q)
    val = zero(eltype(Q))

    for j in 1:n, i in j+1:n
        δ_ij = distance(W, Q, i, j)
        val = val + δ_ij^2
    end

    return val
end
