"""
```
convex_clustering(algorithm::AlgorithmOption, weights, data;)
```
"""
function convex_clustering(algorithm::AlgorithmOption, weights, data;
    K::Integer=0,
    o::Base.Ordering=Base.Order.Forward,
    rho::Real=1.0,
    mu::Real=1.0, kwargs...)
    #
    # extract problem information
    d, n = size(data)
    m = binomial(n, 2)
    M = d*m
    N = d*n

    # allocate optimization variable
    X = copy(data)
    x = vec(X)
    if algorithm isa ADMM
        y = zeros(M)
        λ = zeros(M)
        variables = (x = x, y = y, λ = λ)
    else
        variables = (x = x,)
    end

    # allocate derivatives
    ∇f = needs_gradient(algorithm) ? similar(x) : nothing
    ∇q = needs_gradient(algorithm) ? similar(x) : nothing
    ∇h = needs_gradient(algorithm) ? similar(x) : nothing
    ∇²f = needs_hessian(algorithm) ? I : nothing
    derivatives = (∇f = ∇f, ∇²f = ∇²f, ∇q = ∇q, ∇h = ∇h)

    # generate operators
    T1 = typeof(o)
    T2 = eltype(data)
    block_norm = zeros(m)
    cache = zeros(m)
    P = BlockSparseProjection{T1,T2}(d, block_norm, cache, K)
    a = copy(x)
    D = CvxClusterFM(d, n)
    if algorithm isa SteepestDescent
        H = nothing
    elseif algorithm isa MM
        H = ProxDistHessian(N, rho, ∇²f, D'D)
    else
        H = ProxDistHessian(N, mu, ∇²f, D'D)
    end
    operators = (D = D, P = P, H = H, a = a)

    # allocate buffers for mat-vec multiplication, projections, and so on
    z = similar(Vector{eltype(x)}, M)
    Pz = similar(z)
    v = similar(z)
    b = needs_linsolver(algorithm) ? similar(x) : nothing
    buffers = (z = z, Pz = Pz, v = v, b = b)

    # select linear solver, if needed
    if needs_linsolver(algorithm)
        b1 = similar(x)
        b2 = similar(x)
        b3 = similar(x)
        linsolver = CGIterable(H, x, b1, b2, b3, 1e-8, 0.0, 1.0, N, 0)
    else
        linsolver = nothing
    end

    # create views, if needed
    views = nothing

    # pack everything into ProxDistProblem container
    objective = cvxclst_objective
    algmap = cvxclst_iter
    prob = ProxDistProblem(variables, derivatives, operators, buffers, views, linsolver)

    # solve the optimization problem
    optimize!(algorithm, objective, algmap, prob, rho, mu; kwargs...)

    return X
end

#########################
#       objective       #
#########################

function cvxclst_objective(::AlgorithmOption, prob, ρ)
    @unpack x = prob.variables
    @unpack ∇f, ∇q, ∇h = prob.derivatives
    @unpack D, P, a = prob.operators
    @unpack z, Pz, v = prob.buffers

    # evaulate gradient of loss
    @. ∇f = x - a

    # evaluate gradient of penalty
    mul!(z, D, x)
    P(Pz, z)        # TODO: figure out how to fix this ugly notation
    @. v = z - Pz
    mul!(∇q, D', v)
    @. ∇h = ∇f + ρ * ∇q

    loss = SqEuclidean()(x, a) / 2
    penalty = dot(v, v)
    normgrad = dot(∇h, ∇h)

    return loss, penalty, normgrad
end

############################
#      algorithm maps      #
############################

function cvxclst_iter(::SteepestDescent, prob, ρ, μ)
    @unpack x = prob.variables
    @unpack ∇h = prob.derivatives
    @unpack D = prob.operators
    @unpack z = prob.buffers

    # evaluate step size, γ
    mul!(z, D, ∇h)
    a = dot(∇h, ∇h)     # ||∇h(x)||^2
    b = dot(z, z)       # ||D*∇h(x)||^2
    γ = a / (a + ρ*b + eps())

    # steepest descent, x_new = x_old - γ*∇h(x_old)
    axpy!(-γ, ∇h, x)

    return γ
end

function cvxclst_iter(::MM, prob, ρ, μ)
    @unpack D, a = prob.operators
    @unpack b, Pz = prob.buffers
    linsolver = prob.linsolver

    # build RHS of Ax = b
    mul!(b, D', Pz)
    axpby!(1, a, ρ, b)

    # solve the linear system; assuming x bound to linsolver
    __do_linear_solve!(linsolver, b)

    return 1.0
end

function cvxclst_iter(::ADMM, prob, ρ, μ)
    @unpack x, y, λ = prob.variables
    @unpack D, P, a = prob.operators
    @unpack z, Pz, v, b = prob.buffers
    linsolver = prob.linsolver

    # y block update
    α = (ρ / μ)
    @. v = z + λ; Pv = Pz; P(Pv, v)
    @inbounds @simd for j in eachindex(y)
        y[j] = α/(1+α) * Pv[j] + 1/(1+α) * v[j]
    end

    # x block update
    @. v = y - λ
    mul!(b, D', v)
    axpby!(1, a, μ, b)
    __do_linear_solve!(linsolver, b)

    # λ block update
    mul!(z, D, x)
    @inbounds @simd for j in eachindex(λ)
        λ[j] = λ[j] / μ + z[j] - y[j]
    end

    return μ
end

#################
#   utilities   #
#################

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

#################
#   examples    #
#################

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
