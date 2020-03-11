function __accumulate_averaging_step!(Q, W, wbar, U)
    d, n = size(U)  # extract dimensions
    T = eltype(U)   # extract field type

    for i in 1:n, k in 1:d
        # compute the leave-one-out arithmetic means utilde[i], online
        utilde_i = zero(T)

        # indices j < i, standard recurrence relation
        for j in 1:i-1
            utilde_i = utilde_i + (W[i,j]^2 * U[k,j] - utilde_i) / j
        end

        # indices j > i, needs correction in denominator
        for j in i+1:n
            utilde_i = utilde_i + (W[i,j]^2 * U[k,j] - utilde_i) / (j-1)
        end

        # accumulate the centering operation
        Q[k,i] = Q[k,i] + 2*(n-1) * (wbar[i]*U[k,i] - utilde_i)
    end

    return nothing
end

function __accumulate_sparsity_correction!(Q, W, U Iv, Jv)
    d, n = size(U)  # extract dimensions

    # iterate over k-largest cluster distances
    for l in eachindex(Iv)
        i, j = Iv[l], Jv[l]

        # apply correction to each block
        for k in 1:d
            δ_ij = W[i,j]*(U[k,i] - U[k,j])

            Q[k,i] = Q[k,i] - δ_ij
            Q[k,j] = Q[k,j] + δ_ij
        end
    end

    return nothing
end

# compute norm(u_i - u_j), norm of vector from point i to point j
function __compute_difference_norm(W, U, i, j)
    d = size(U, 1)
    dist = zero(eltype(U)) # careful with Ints

    for k in 1:d
        δ_ij = W[i,j]*(U[k,i] - U[k,j])
        dist = dist + δ_ij^2
    end

    return dist
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

    for j in 1:n, i in 1:j-1
        v_ij = __compute_difference_norm(W, U, i, j)
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

    for j in 1:n, i in 1:j-1
        block_val = __compute_difference_norm(W, Q, i, j)
        val = val + block_val
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
