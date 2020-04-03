##### penalized optimization
"""
```
cvxclst_evaluate_objective(U, X, y, ρ)
```

Evaluate the penalized objective with coefficient `ρ` based on centroid matrix `U` and data `X`.
The variable `y` represents the vector pointing towards the projection of `W*D*vec(U)`.
"""
function cvxclst_evaluate_objective(U, X, y, ρ)
    loss = dot(U,U) - 2*dot(U,X) + dot(X,X)
    penalty = dot(y, y)
    objective = 0.5 * (loss + ρ*penalty)

    return loss, penalty, objective
end

##### sparse fused block projection #####

"""
```
distance(A, i, j)
```

Compute the Euclidean distance between `A[:,i]` and `A[:,j]` in-place.
"""
function distance(A, i, j)
    T = eltype(A)
    d_ij = zero(sqrt(one(T)))

    for k in 1:size(A, 1)
        d_ij += (A[k,i] - A[k,j])^2
    end

    return sqrt(d_ij)
end

"""
```
distance(W, A, i, j)
```

Compute the weighted Euclidean distance between `A[:,i]` and `A[:,j]` in-place.
"""
function distance(W, A, i, j)
    return W[i,j] * distance(A, i, j)
end

"""
```
sparse_fused_block_projection(W, A, [k = 1])
```

Compute a `k`-sparse fused block projection of `A[:,i] - A[:,j]`.
The matrix `W` applies a weight to each Euclidean distance between centroids.
Only the lower triangular region is used; that is, `W[i,j]` assumes `i > j`.

For the sake of computational performance, we assume `A` is `d` by `n`, where `d` is the number of features and `n` is the number of samples.

The optional argument `0 <= k <= binomial(n, 2)` enforces the number of non-zero
blocks within the projection.
"""
# function sparse_fused_block_projection(W, A, k = 1)
#     d, n = size(A)          # d features, n samples
#     k_max = binomial(n, 2)  # total number of unique comparisons
#
#     T = eltype(A)                   # element type of A
#     distT = typeof(sqrt(one(T)))    # type for distances
#
#     # enforce 0 <= k <= k_max
#     k = min(k, k_max)
#     k = max(k, 0)
#
#     # data structures for finding k-largest distances
#     # e.g. Iv[1], Jv[1] => (i, j)
#     # such that distance v[1] is the smallest in the list v
#     Iv = zeros(Int, k)              # index i
#     Jv = zeros(Int, k)              # index j
#     v  = zeros(distT, k)            # list of distances, in ascending order
#     y  = zeros(distT, d * k_max)    # allocate memory for projection
#
#     # zero-overhead implementation
#     k > 0 && sparse_fused_block_projection!(y, Iv, Jv, v, W, A)
#
#     return y, Iv, Jv, v
# end
function sparse_fused_block_projection(W, A, k = 1)
    d, n = size(A)          # d features, n samples
    k_max = binomial(n, 2)  # total number of unique comparisons

    T = eltype(A)                   # element type of A
    distT = typeof(sqrt(one(T)))    # type for distances

    # enforce 0 <= k <= k_max
    k = min(k, k_max)
    k = max(k, 0)

    # allocate memory for W*D*vec(A) and its projection
    y = zeros(distT, d * k_max)
    index = collect(1:length(y))
    buffer = zero(y)

    # compute y = W*D*vec(A) in-place
    cvxclst_apply_fusion_matrix!(y, W, A)

    # compute projection in-place
    k > 0 && sparse_fused_block_projection!(buffer, y, index, k)

    return y, buffer, index
end

"""
```
sparse_fused_block_projection!(y, Iv, Jv, v, W, A)
```

In-place version of `sparse_fused_block_projection`.

- The vector `y` stores the result of the projection.
- The inputs `Iv`, `Jv` store the pairs `(i,j)` and `v` stores the weighted distances.
"""
function sparse_fused_block_projection!(y, Iv, Jv, v, W, A)
    # identify the pairs (i,j) that we should keep
    find_large_blocks!(Iv, Jv, v, W, A)

    # ... then construct projection of D*vec(A)
    apply_projection!(y, W, A, Iv, Jv)

    return y, Iv, Jv, v
end

function sparse_fused_block_projection!(buffer, y, index, K)
    find_large_blocks!(index, y)
    apply_projection!(buffer, y, index, K)
end

"""
```
find_large_blocks!(Iv, Jv, v, W, A)
```

Compute the k-largest weighted centroid distances `W[i,j] * norm(A[:,i] - A[:,j])` in-place with a single sweep.
The inputs `Iv`, `Jv` store the pairs `(i,j)` and `v` stores the weighted distances.
We assume `length(Iv) == length(Jv) == length(v) == k`.
"""
function find_large_blocks!(Iv, Jv, v, W, A)
    n = size(A, 2)

    for j in 1:n, i in j+1:n
        # compute weighted Euclidean distance between centroids for i and j
        v_ij = distance(W, A, i, j)

        # binary search on v to find rank of v_ij relative to the current list
        l = searchsortedlast(v, v_ij)

        # l = 0 if v_ij < every value in v
        if l > 0
            # Update parallel arrays Iv, Jv, v:
            #   - First delete the smallest entry in v.
            #   - Iinsert the new element at position l.
            popfirst!(v)
            insert!(v, l, v_ij)

            popfirst!(Iv)
            insert!(Iv, l, i)

            popfirst!(Jv)
            insert!(Jv, l, j)
        end
    end

    return Iv, Jv, v
end

function find_large_blocks!(indices, y)
    sortperm!(indices, y, rev = true, initialized = true)
end

"""
```
tri2vec(i, j, n)
```

Map a Cartesian coordinate `(i,j)` with `i > j` to its index in dictionary
order, assuming `i` ranges from `1` to `n`.
"""
function tri2vec(i, j, n)
    return (i-j) + n*(j-1) - (j*(j-1))>>1
end

"""
```
apply_projection!(y, A, Iv, Jv)
```

Apply a sparse fused block projection of `A[:,i] - A[:,j]` in-place by storing
the result in `y`. The inputs `Iv` and `Jv` encode the k-largest centroid
differences, where k is the length of both `Iv` and of `Jv`. We assume the pairs
`(i,j)` obey the rule `i > j`.
"""
function apply_projection!(y, W, A, Iv, Jv)
    d, n = size(A)

    # for each pair (i,j) with i > j:
    for (i,j) in zip(Iv, Jv)
        # find the block corresponding to (i,j) within the vector y
        l = tri2vec(i, j, n)

        start = d*(l-1) + 1
        stop  = d*l
        block = start:stop

        # copy A[:,i] - A[:,j] in-place to y
        for (k, idx) in enumerate(block)
            y[idx] = W[i,j] * (A[k,i] - A[k,j])
        end
    end

    return y
end

function apply_projection!(dest, src, index, K)
    for k in 1:K
        dest[index[k]] = src[index[k]]
    end
end

##### cluster assignment #####
"""
Finds the connected components of a graph.
Nodes should be numbered 1,2,...
"""
function connect!(component, A)
#
    nodes = size(A, 1)
    fill!(component, 0)
    components = 0
    for j = 1:nodes
        if component[j] > 0 continue end
        components = components + 1
        component[j] = components
        visit!(component, A, j)
    end
    return (component, components)
end

"""
Recursively assigns components by depth first search.
"""
function visit!(component, A, j::Int)
#
    nodes = size(A, 1)
    for i in 1:nodes
        if A[i,j] == 1 # check that i is a neighbor of j
            if component[i] > 0 continue end
            component[i] = component[j]
            visit!(component, A, i)
        end
    end
end

function assign_classes!(class, A, U)
    d, n = size(U)

    # update adjacency matrix
    for j in 1:n, i in j+1:n
        if distance(U, i, j) < 5e-4
            A[i,j] = 1
            A[j,i] = 1
        else
            A[i,j] = 0
            A[j,i] = 0
        end
    end

    # assign classes based on connected components
    class, nclasses = connect!(class, A)

    return (A, class, nclasses)
end

##### weights #####

"""
```
gaussian_weights(X; [phi = 1.0])
```

Assign weights to each pair of samples `(i,j)` based on a Gaussian kernel.
The parameter `phi` scales the influence of the distance `norm(X[:,i] - X[:,j])^2`.

**Note**: Samples are assumed to be given in columns.
"""
function gaussian_weights(X; phi = 1.0)
    d, n = size(X)

    T = eltype(X)
    W = zeros(n, n)

    for j in 1:n, i in j+1:n
        δ_ij = distance(X, i, j)
        w_ij = exp(-phi*δ_ij^2)

        W[i,j] = w_ij
        W[j,i] = w_ij
    end

    return W
end

"""
```
knn_weights(W, k)
```

Threshold weights `W` based on `k` nearest neighbors.
"""
function knn_weights(W, k)
    n = size(W, 1)
    w = [W[i,j] for j in 1:n for i in j+1:n] |> vec
    i = 1
    neighbors = tri2vec.((i+1):n, i, n)
    keep = neighbors[sortperm(w[neighbors], rev = true)[1:k]]

    for i in 2:(n-1)
        group_A = tri2vec.((i+1):n, i, n)
        group_B = tri2vec.(i, 1:(i-1), n)
        neighbors = [group_A; group_B]
        knn = neighbors[sortperm(w[neighbors], rev = true)[1:k]]
        keep = union(knn, keep)
    end

    i = n
    neighbors = tri2vec.(i, 1:(i-1), n)
    knn = neighbors[sortperm(w[neighbors], rev = true)[1:k]]
    keep = union(knn, keep)

    W_knn = zero(W)

    for j in 1:n, i in j+1:n
        l = tri2vec(i, j, n)
        if l in keep
            W_knn[i,j] = W[i,j]
            W_knn[j,i] = W[i,j]
        end
    end

    return W_knn
end

##### simulation #####

"""
```
gaussian_clusters(centers, n)
```

Simulate Gaussian centroids in 2D with the given `centers`.
Each cluster is generated with `n` points.
"""
function gaussian_clusters(centers, n)
    cluster = Matrix{Float64}[]

    for center in centers
        d = length(center)
        data = center .+ 0.1*randn(d, n)
        push!(cluster, data)
    end

    return hcat(cluster...)
end

function cvxclstr_search(X, maxiters, K)
    d, n = size(X)
    α = [1 / norm(X[:, i]) for i in 1:n]
    Y = X * Diagonal(α)
    W = gaussian_weights(Y)

    history = SDLogger(length(K), maxiters)

    result = Vector{Int}[]
    for k in K
        _, class, _ = @time convex_clustering(SteepestDescent(), W, Y, maxiters = maxiters, penalty = fast_schedule, K = k, history = history)
        push!(result, class)
    end

    return result, history
end
