function __accumulate_averaging_step!(Q, W, U)
    d, n = size(U)  # extract dimensions

    for j in 1:n, i in 1:j-1, k in 1:d
        δ_ijk = W[i,j]^2*(U[k,i] - U[k,j])
        Q[k,i] = Q[k,i] + δ_ijk
        Q[k,j] = Q[k,j] - δ_ijk
    end

    return nothing
end

function __accumulate_sparsity_correction!(Q, W, U, Iv, Jv)
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
        v_ij = __distance(W, U, i, j)
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
        block_val = __distance(W, Q, i, j)
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
