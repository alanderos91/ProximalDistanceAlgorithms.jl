"""
```
construct_fustion_matrix(d, n)
```

Construct a block matrix `D` such that `Dblock * vec(A) == A[:,i] - A[:,j]`.
The result `D` is compatible with `vec(A)`, where `A` is `d` by `n`.

Blocks are stacked in dictionary order.
For example, if `n = 3` then the blocks are ordered `(2,1), (3,1), (3,2)`.
"""
function cvxcluster_fusion_matrix(d, n)
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

function cvxclst_apply_fusion_matrix!(y, W, U)
    d, n = size(U)

    idx = 1
    for j in 1:n, i in j+1:n, k in 1:d
        y[idx] = W[i,j] * (U[k,i] - U[k,j])
        idx += 1
    end

    return y
end

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

function __accumulate_averaging_step!(Q, W, U)
    d, n = size(U)  # extract dimensions

    for j in 1:n, i in j+1:n, k in 1:d
        δ_ijk = W[i,j]^2*(U[k,i] - U[k,j])
        Q[k,i] = Q[k,i] + δ_ijk
        Q[k,j] = Q[k,j] - δ_ijk
    end

    return nothing
end

function __accumulate_all_operations!(Q, W, U, Iv, Jv)
    d, n = size(U)  # extract dimensions
    blocks = zip(Iv, Jv)

    for j in 1:n, i in j+1:n, k in 1:d
        if (i,j) in blocks
            δ_ijk = zero(eltype(W))
        else
            δ_ijk = (W[i,j]^2)*(U[k,i] - U[k,j])
        end

        Q[k,i] = Q[k,i] + δ_ijk
        Q[k,j] = Q[k,j] - δ_ijk
    end

    return nothing
end

function __apply_sparsity_correction!(Q, W, U, Iv, Jv)
    d, n = size(U)  # extract dimensions

    for (i, j) in zip(Iv, Jv), k in 1:d
        δ_ijk = (W[i,j]^2)*(U[k,i] - U[k,j])

        Q[k,i] = Q[k,i] - δ_ijk
        Q[k,j] = Q[k,j] + δ_ijk
    end

    return nothing
end

#
# I: set of i indices
# J: set of j indices
# v: list of norm differences norm(u_i - u_j)
# W: weights
# U: current cluster assignments
# k: number of blocks
function __find_large_blocks!(I, J, v, W, U)
    n = size(U, 2)

    for j in 1:n, i in j+1:n
        v_ij = distance(W, U, i, j)
        l = searchsortedlast(v, v_ij)

        if l > 0
            popfirst!(v)
            insert!(v, l, v_ij)

            popfirst!(I)
            insert!(I, l, i)

            popfirst!(J)
            insert!(J, l, j)
        end
    end

    return nothing
end

function __evaluate_weighted_gradient_norm(W, Q)
    d, n = size(Q)
    val = zero(eltype(Q))

    for j in 1:n, i in j+1:n
        δ_ij = distance(W, Q, i, j)
        val = val + δ_ij^2
    end

    return val
end

# function test(d, n, k)
#     # simulate data
#     U = rand(d, n)
#
#     # initialize data structures
#     Iv = zeros(Int, k)
#     Jv = zeros(Int, k)
#     v = zeros(k)
#
#     # check
#     @time find_large_blocks!(Iv, Jv, v, U, k)
#
#     return Iv, Jv, v
# end
#
# function forloopcheck(d, n)
#     x = rand(d)
#
#     @time for j in 1:n, i in 1:j-1
#         norm(x)
#     end
# end

# Implementation notes:
#
# 1. Want to compute ||u_i - u_j|| efficiently
#   a. Store U as d by n matrix --> use my code
#   b. Store U as list of column vectors --> use `norm`
#
# 2. Want to find k-largest blocks according to ||u_i - u_j||.
#   a. The number of comparisons is n choose 2, so would like to
#    avoid storing differences explicitly. Lazy approach?
#   b. Finding the k-largest blocks should have an O(n log k) algorithm (heaps).
