##### penalized optimization
"""
```
cvxclst_evaluate_objective(U, X, y, ρ)
```

Evaluate the penalized objective with coefficient `ρ` based on centroid matrix `U` and data `X`.
The variable `y` represents the vector pointing towards the projection of `W*D*vec(U)`.
"""
function cvxclst_evaluate_objective(U, X, Y, ρ)
    loss = dot(U, U) - 2*dot(U, X) + dot(X, X)
    loss = abs(loss)
    # loss = SqEuclidean(1e-12)(U, X)
    penalty = dot(Y, Y)
    objective = 0.5 * (loss + ρ*penalty)

    return loss, penalty, objective
end

##### sparse fused block projection #####

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
function sparse_block_projection(W, U, K = 1)
    d, n = size(U)
    K_max = binomial(n, 2)  # number of unique comparisons
    Y = zeros(d, m)         # encodes differences between columns of U
    Δ = zeros(n, n)         # encodes pairwise distances
    index = collect(1:n*n)  # index vector

    # enforce 0 <= K <= K_max
    K = min(K, K_max)
    K = max(K, 0)

    return sparse_block_projection!(Y, Δ, index, W, U, K)
end

"""
```
sparse_fused_block_projection!(buffer, y, index, K)
```

In-place version of `sparse_fused_block_projection`.
"""
function sparse_block_projection!(Y, Δ, index, W, U, K)
    d, n = size(U)

    if K > 0
        # compute pairwise distances
        pairwise!(Δ, Euclidean(1e-12), U, dims = 2)
        @. Δ = W * Δ

        # mask upper triangular part to extract unique comparisons
        for j in 1:n, i in 1:j-1
            @inbounds Δ[i,j] = 0
        end
        δ = vec(Δ)

        # find the K largest distances
        J = partialsortperm!(index, δ, 1:K, rev = true, initialized = true)

        # # compute Y - P(Y)
        # for j in 1:n, i in j+1:n # dictionary order, (i,j) with i > j
        #     l = tri2vec(i, j, n)
        #     ix = (j-1)*n + i
        #
        #     if ix in J # block is preserved in projection
        #         for k in 1:d
        #             Y[k,l] = 0
        #         end
        #     else
        #         for k in 1:d
        #             Y[k,l] = W[i,j] * (U[k,i] - U[k,j])
        #         end
        #     end
        # end
        ix2coord = CartesianIndices(Δ)
        for ix in J
            i = ix2coord[ix][1]
            j = ix2coord[ix][2]

            if i > j
                l = tri2vec(i,j,n)
                for k in 1:d
                    Y[k,l] = W[i,j] * (U[k,i] - U[k,j])
                end
            end
        end
    else
        fill!(Y, 0)
    end

    return Y
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

function assign_classes!(class, A, U, tol)
    d, n = size(U)

    # update adjacency matrix
    for j in 1:n, i in j+1:n
        if distance(U, i, j) < tol
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

function assign_classes(U, tol = 1e-2)
    n = size(U, 2)
    A = zeros(Bool, n, n)
    class = zeros(Int, n)
    return assign_classes!(class, A, U, tol)
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
        @views δ_ij = Euclidean(1e-12)(X[:,i], X[:,j])
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
