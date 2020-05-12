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
    Y = zeros(d, K_max)     # encodes differences between columns of U
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
        pairwise!(Δ, Euclidean(), U, dims = 2)
        Δ .= Δ .* W

        # mask upper triangular part to extract unique comparisons
        for j in 1:n, i in 1:j-1
            @inbounds Δ[i,j] = 0
        end
        δ = vec(Δ)

        # find the K largest distances
        J = partialsortperm!(index, δ, 1:K, rev = true, initialized = true)

        ix2coord = CartesianIndices(Δ)
        for ix in J
            i = ix2coord[ix][1]
            j = ix2coord[ix][2]

            if i > j
                l = tri2vec(i,j,n)
                for k in 1:d
                    Y[k,l] = U[k,i] - U[k,j]
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

function assign_classes!(class, A, Δ, U, tol)
    n = size(Δ, 1)

    Δ = pairwise(Euclidean(), U, dims = 2)

    # update adjacency matrix
    for j in 1:n, i in j+1:n
        abs_dist = log(10, Δ[i,j])

        if (abs_dist < -tol)
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

function assign_classes(U, tol = 3.0)
    n = size(U, 2)
    A = zeros(Bool, n, n)
    Δ = zeros(n, n)
    class = zeros(Int, n)

    return assign_classes!(class, A, Δ, U, tol)
end

function var_information(a, b)
    C = counts(a, b)
    isempty(C) && return 0.0
    countsA = a isa ClusteringResult ? counts(a) : sum(C, dims=2)
    countsB = b isa ClusteringResult ? counts(b) : sum(C, dims=1)
    VI = 0.0
    @inbounds for (i, ci) in enumerate(countsA), (j, cj) in enumerate(countsB)
        cij = C[i,j]
        if cij > 0.0
            VI += cij * log(cij*cij / (ci*cj))
        end
    end
    return -VI/sum(countsA)
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
function gaussian_weights(X; phi = 0.5)
    d, n = size(X)

    T = eltype(X)
    W = zeros(n, n)

    for j in 1:n, i in j+1:n
        @views δ_ij = SqEuclidean()(X[:,i], X[:,j])
        w_ij = exp(-phi*δ_ij)

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

Simulate a cluster with `n` points centered at the given `centroid`.
"""
function gaussian_cluster(centroid, n)
    d = length(centroid)
    cluster = centroid .+ 0.1 * randn(d, n)

    return cluster
end

##### example data sets #####
function convex_clustering_data(file)
    dir = dirname(@__DIR__) # should point to src
    dir = dirname(dir)      # should point to top-level directory

    df = CSV.read(joinpath(dir, "data", file), copycols = true)

    if basename(file) == "mammals.dat" # classes in column 2
        # extract data as features × columns
        data = convert(Matrix{Float64}, df[:, 3:end-1])
        X = copy(transpose(data))

        # retrieve class assignments
        class_name = unique(df[:,2])
        classes = convert(Vector{Int}, indexin(df[:,2], class_name))
    else # classes in last column
        # extract data as features × columns
        data = convert(Matrix{Float64}, df[:,1:end-1])
        X = copy(transpose(data))

        # retrieve class assignments
        class_name = unique(df[:,end])
        classes = convert(Vector{Int}, indexin(df[:,end], class_name))
    end

    return X, classes, length(class_name)
end
