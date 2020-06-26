function cvxclst_eval(::AlgorithmOption, optvars, derivs, operators, buffers, ρ)
    u = optvars.u

    ∇f = derivs.∇f
    ∇d = derivs.∇d
    ∇h = derivs.∇h

    D = operators.D
    x = operators.x
    o = operators.o
    K = operators.K
    compute_projection = operators.compute_projection

    z = buffers.z
    U = buffers.U
    ds = buffers.ds
    ss = buffers.ss
    Pz = buffers.Pz

    mul!(z, D, u)

    # compute projection
    evaluate_distances!(ds, U)
    copyto!(ss, ds)
    P = compute_projection(ss, o, K)

    # evaluate projection
    d, n = D.d, D.n
    offset = 0

    @inbounds for block in eachindex(ds)
        Δ = ds[block]
        r = (P(Δ) == Δ)
        @inbounds for k in 1:d
            Pz[k+offset] = r*z[k+offset]
            z[k+offset] = (1-r)*z[k+offset]
        end
        offset += d
    end

    @. ∇f = u - x
    mul!(∇d, D', z)
    @. ∇h = ∇f + ρ * ∇d

    loss = SqEuclidean()(u, x) / 2
    penalty = dot(z, z)
    normgrad = dot(∇h, ∇h)

    return loss, penalty, normgrad
end

##### distances #####
function evaluate_distances!(xs, U)
    d, n = size(U)

    k = 0
    for j in 1:n, i in j+1:n
        ui = view(U, 1:d, i)
        uj = view(U, 1:d, j)
        xs[k+=1] = SqEuclidean()(ui, uj)
    end
    return xs
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
