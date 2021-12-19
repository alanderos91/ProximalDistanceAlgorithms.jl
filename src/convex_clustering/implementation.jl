@doc raw"""
    convex_clustering(algorithm::AlgorithmOption, weights, data; kwargs...)

Cluster `data` using an approximate convexification of ``k``-means.

**Note**: This function should only be used for exploratory analyses.
See [`convex_clustering_path`](@ref) for an algorithm that returns a list of candidate clusterings.

The `data` are assumed to be arranged as a `features` by `samples` matrix.

See also: [`MM`](@ref), [`StepestDescent`](@ref), [`ADMM`](@ref), [`MMSubSpace`](@ref), [`initialize_history`](@ref), [`optimize!`](@ref), [`anneal!`](@ref)

# Keyword Arguments

- `sparsity::Real=0.5`: A parameter controlling the number of clusters with `s=0` assigning each sample its own cluster and `s=1` coalescing to a single cluster.
- `ls=Val(:LSQR)`: Choice of linear solver for `MM`, `ADMM`, and `MMSubSpace` methods. Choose one of `Val(:LSQR)` or `Val(:CG)` for LSQR or conjugate gradients, respectively.
"""
function convex_clustering(algorithm::AlgorithmOption, weights, data; sparsity::Real=0.5, ls::LS=Val(:LSQR), kwargs...) where LS
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

    if algorithm isa MMSubSpace
        K = subspace_size(algorithm)
        G = zeros(N, K)
        derivatives = (∇f = ∇f, ∇²f = ∇²f, ∇q = ∇q, ∇h = ∇h, G = G)
    else
        derivatives = (∇f = ∇f, ∇²f = ∇²f, ∇q = ∇q, ∇h = ∇h)
    end

    # generate operators
    k = round(Int, (1-sparsity)*m)
    P = ColumnL0Projection(k, collect(1:m), zeros(Int, m), zeros(m))
    a = copy(x)
    w = Float64[]
    for j in 1:n, i in j+1:n push!(w, weights[i,j]) end
    D = CvxClusterFM(d, n)
    W = Diagonal(w)
    operators = (D = D, W = W, P = P, a = a)

    # allocate buffers for mat-vec multiplication, projections, and so on
    z = similar(Vector{eltype(x)}, M)
    Pz = similar(z)
    v = similar(z)

    # select linear solver, if needed
    if needs_linsolver(algorithm)
        if algorithm isa MMSubSpace
            K = subspace_size(algorithm)
            β = zeros(K)

            if ls isa Val{:LSQR}
                A₁ = LinearMap(I, N)
                A₂ = D
                A = MMSOp1(A₁, A₂, G, x, x, 1.0)
                b = similar(typeof(x), size(A, 1))
                linsolver = LSQRWrapper(A, β, b)
            else
                b = similar(typeof(x), K)
                linsolver = CGWrapper(G, β, b)
            end
        else
            if ls isa Val{:LSQR}
                A₁ = LinearMap(I, N)
                A₂ = D
                A = QuadLHS(A₁, A₂, x, 1.0)
                b = similar(typeof(x), size(A, 1))
                linsolver = LSQRWrapper(A, x, b)
            else
                b = similar(x)
                linsolver = CGWrapper(D, x, b)
            end
        end
    else
        b = nothing
        linsolver = nothing
    end

    if algorithm isa ADMM
        mul!(y, D, x)
        y_prev = similar(y)
        r = similar(y)
        s = similar(x)
        tmpx = similar(x)
        buffers = (z = z, Pz = Pz, v = v, b = b, y_prev = y_prev, s = s, r = r, tmpx = tmpx)
    elseif algorithm isa MMSubSpace
        tmpGx1 = zeros(N)
        tmpGx2 = zeros(N)
        tmpx = similar(x)
        buffers = (z = z, Pz = Pz, v = v, b = b, β = β, tmpx = tmpx, tmpGx1 = tmpGx1, tmpGx2 = tmpGx2)
    else
        tmpx = similar(x)
        buffers = (z = z, Pz = Pz, v = v, b = b, tmpx = tmpx)
    end

    # create views, if needed
    views = nothing

    # pack everything into ProxDistProblem container
    objective = cvxclst_objective
    algmap = cvxclst_iter
    old_variables = deepcopy(variables)
    prob = ProxDistProblem(variables, old_variables, derivatives, operators, buffers, views, linsolver)
    prob_tuple = (objective, algmap, prob)

    # solve the optimization problem
    optimize!(algorithm, prob_tuple; kwargs...)

    return X
end

@doc raw"""
convex_clustering_path(algorithm::AlgorithmOption, weights, data; kwargs...)

Cluster `data` using an approximate convexification of ``k``-means.

The `data` are assumed to be arranged as a `features` by `samples` matrix. This
algorithm performs clustering by varying a sparsity parameter `nu` in a
monotonic fashion and minimizing a penalized objective. At the end of each
minimization step, we update `nu` by counting the number of pairs `data[:,i] ≈ data[:,j]`.
This heuristic defines a solution path that explores the large space of
possible clusterings in a frugal manner. Sparse weights can greatly accelerate
the algorithm but may miss clusterings.
Returns a `NamedTuple` with fields `assignment` and `ν_path`.

Setting `atol` to smaller values will generally improve the quality of clusterings and reduce the number of minimization steps. However, individual
minimization steps become more expensive as a result.

See also: [`MM`](@ref), [`StepestDescent`](@ref), [`ADMM`](@ref), [`MMSubSpace`](@ref), [`initialize_history`](@ref), [`optimize!`](@ref), [`anneal!`](@ref)

# Keyword Arguments

- `start::Real=0.5`: Starting sparsity value where `1-start` is the proportion of significant differences.
- `stepsize::Real=0.05`: A value between `0` and `1` that discretizes the search space. If no additional points are found to coalesce, then decrease `nu` by `stepsize*nu_max`.
- `radius::Real=2`: A value determining how close centroid assignments must in order to be considered clustered.
- `ls=Val(:LSQR)`: Choice of linear solver for `MM`, `ADMM`, and `MMSubSpace` methods. Choose one of `Val(:LSQR)` or `Val(:CG)` for LSQR or conjugate gradients, respectively.
"""
function convex_clustering_path(algorithm::AlgorithmOption, weights, data;
    init_sparsity::Real=0.5,
    max_sparsity::Real=1.0,
    stepsize::Real=0.05,
    magnitude::Real=-2,
    callback::cbT=DEFAULT_CALLBACK,
    ls::LS=Val(:LSQR), kwargs...) where {cbT, LS}
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
    old_variables = deepcopy(variables)

    # allocate derivatives
    ∇f = needs_gradient(algorithm) ? similar(x) : nothing
    ∇q = needs_gradient(algorithm) ? similar(x) : nothing
    ∇h = needs_gradient(algorithm) ? similar(x) : nothing
    ∇²f = needs_hessian(algorithm) ? I : nothing

    if algorithm isa MMSubSpace
        K = subspace_size(algorithm)
        G = zeros(N, K)
        derivatives = (∇f = ∇f, ∇²f = ∇²f, ∇q = ∇q, ∇h = ∇h, G = G)
    else
        derivatives = (∇f = ∇f, ∇²f = ∇²f, ∇q = ∇q, ∇h = ∇h)
    end

    # generate operators
    a = copy(x)
    w = Float64[]
    for j in 1:n, i in j+1:n push!(w, weights[i,j]) end
    D = CvxClusterFM(d, n)
    W = Diagonal(w)

    # allocate buffers for mat-vec multiplication, projections, and so on
    z = similar(Vector{eltype(x)}, M)
    Pz = similar(z)
    v = similar(z)

    # select linear solver, if needed
    if needs_linsolver(algorithm)
        if algorithm isa MMSubSpace
            K = subspace_size(algorithm)
            β = zeros(K)

            if ls isa Val{:LSQR}
                A₁ = LinearMap(I, N)
                A₂ = D
                A = MMSOp1(A₁, A₂, G, x, x, 1.0)
                b = similar(typeof(x), size(A, 1))
                linsolver = LSQRWrapper(A, β, b)
            else
                b = similar(typeof(x), K)
                linsolver = CGWrapper(G, β, b)
            end
        else
            if ls isa Val{:LSQR}
                A₁ = LinearMap(I, N)
                A₂ = D
                A = QuadLHS(A₁, A₂, x, 1.0)
                b = similar(typeof(x), size(A, 1))
                linsolver = LSQRWrapper(A, x, b)
            else
                b = similar(x)
                linsolver = CGWrapper(D, x, b)
            end
        end
    else
        b = nothing
        linsolver = nothing
    end

    if algorithm isa ADMM
        mul!(y, D, x)
        buffers = (z=z, Pz=Pz, v=v, b=b, y_prev=similar(y), s=similar(x), r=similar(y), tmpx=similar(x))
    elseif algorithm isa MMSubSpace
        buffers = (z=z, Pz=Pz, v=v, b=b, β=β, tmpx=similar(x), tmpGx1=zeros(N), tmpGx2=zeros(N))
    else
        tmpx = similar(x)
        buffers = (z = z, Pz = Pz, v = v, b = b, tmpx = tmpx)
    end

    # create views, if needed
    views = nothing

    # pack everything into ProxDistProblem container
    objective = cvxclst_objective
    algmap = cvxclst_iter

    # arrays needed for projection
    projection_idx, projection_buffer, projection_scores = collect(1:m), zeros(Int, m), zeros(m)

    # allocate outputs
    assignment = Vector{Int}[]
    number_classes = Int[]
    sparsity = Float64[]

    # create a closure to count constraints
    Z = reshape(z, d, m) # use matrix view of Z
    count_constraints = function()
        mul!(z, D, x) # compute differences
        number_coalesced = 0
        for zᵢ in eachcol(Z) # count differences withhin the specificed order of magnitude
            number_coalesced += log10(norm(zᵢ)) ≤ magnitude
        end
        number_coalesced
    end

    # initialize solution path heuristic
    nconstraint = count_constraints()
    if 0 ≤ init_sparsity ≤ max_sparsity
        s = init_sparsity
    else
        s = nconstraint / m
    end

    prog = ProgressUnknown("Searching clustering path...")
    ProgressMeter.next!(prog)
    while init_sparsity ≤ s ≤ max_sparsity
        k = round(Int, (1-s)*m)
        P = ColumnL0Projection(k, projection_idx, projection_buffer, projection_scores)
        operators = (D = D, W = W, P = P, a = a)
        prob = ProxDistProblem(variables, old_variables, derivatives, operators, buffers, views, linsolver)
        prob_tuple = (objective, algmap, prob)

        result = optimize!(algorithm, prob_tuple; kwargs...)

        # update history
        callback(Val(:inner), algorithm, result.iters, result, prob, 0.0, 0.0)

        # record current solution
        _, assigned_classes, classes = assign_classes(X, magnitude)
        push!(assignment, assigned_classes)
        push!(number_classes, classes)
        push!(sparsity, 100*s)

        showvalues = [
            (:sparsity, 100*s),
            (:classes, classes),
            (:distance, sqrt(result.distance)),
            (:gradient, sqrt(result.gradient)),
        ]

        # count satisfied constraints
        nconstraint = count_constraints()

        # increase s using the observed sparsity level or a fixed increase
        s_proposal = nconstraint / m
        if s_proposal > s + stepsize
            s = s_proposal
        else
            s = s + stepsize
        end
        s = s + stepsize
        
        ProgressMeter.next!(prog, showvalues=showvalues)
    end
    ProgressMeter.finish!(prog)

    solution_path = (assignment=assignment, number_classes=number_classes, sparsity=sparsity)

    return solution_path
end

#########################
#       objective       #
#########################

function cvxclst_objective(::AlgorithmOption, prob, ρ)
    @unpack x = prob.variables
    @unpack ∇f, ∇q, ∇h = prob.derivatives
    @unpack D, W, P, a = prob.operators
    @unpack z, Pz, v = prob.buffers

    # evaulate gradient of loss
    @. ∇f = x - a

    # evaluate gradient of penalty
    nrows, ncols = D.d, round(Int, D.M / D.d) # TODO: cache m = M / d that was used to make D
    mul!(z, D, x)  # compute difference
    Pz_mat = reshape(Pz, nrows, ncols)
    copyto!(Pz, z)
    rmul!(Pz_mat, W) # update to weighted differences
    P(Pz_mat)
    rdiv!(Pz_mat, W) # map back to original scale
    @. v = z - Pz
    mul!(∇q, D', v)
    @. ∇h = ∇f + ρ * ∇q

    loss = SqEuclidean()(x, a)
    distance = dot(v, v)
    objective = 1//2 * loss + ρ/2 * distance
    normgrad = dot(∇h, ∇h)

    return IterationResult(loss, objective, distance, normgrad)
end

############################
#      algorithm maps      #
############################

function cvxclst_iter(::SteepestDescent, prob, ρ, μ)
    @unpack x = prob.variables
    @unpack ∇f, ∇q, ∇h = prob.derivatives
    @unpack D, W, P, a = prob.operators
    @unpack z, Pz, v = prob.buffers

    # projection + gradient
    nrows, ncols = D.d, round(Int, D.M / D.d) # TODO: cache m = M / d that was used to make D
    @. ∇f = x - a
    mul!(z, D, x)  # compute difference
    Pz_mat = reshape(Pz, nrows, ncols)
    copyto!(Pz, z)
    rmul!(Pz_mat, W) # update to weighted differences
    P(Pz_mat)
    rdiv!(Pz_mat, W) # map back to original scale
    @. v = z - Pz
    mul!(∇q, D', v)
    @. ∇h = ∇f + ρ * ∇q

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
    @unpack x = prob.variables
    @unpack ∇²f = prob.derivatives
    @unpack D, W, P, a = prob.operators
    @unpack b, z, Pz, tmpx = prob.buffers
    linsolver = prob.linsolver

    # projection
    nrows, ncols = D.d, round(Int, D.M / D.d) # TODO: cache m = M / d that was used to make D
    mul!(z, D, x)  # compute difference
    Pz_mat = reshape(Pz, nrows, ncols)
    copyto!(Pz, z)
    rmul!(Pz_mat, W) # update to weighted differences
    P(Pz_mat)
    rdiv!(Pz_mat, W) # map back to original scale

    if linsolver isa LSQRWrapper
        # build LHS of A*x = b
        # forms a BlockMap so non-allocating
        # however, A*x and A'b have small allocations due to views?
        A₁ = LinearMap(I, size(D, 2))
        A₂ = D
        A = QuadLHS(A₁, A₂, tmpx, √ρ)

        # build RHS of A*x = b; b = [a; √ρ * P(D*x)]
        n = length(a)
        copyto!(b, 1, a, 1, n)
        for k in eachindex(Pz)
            b[n+k] = √ρ * Pz[k]
        end
    else
        # assemble LHS
        A = ProxDistHessian(∇²f, D'D, tmpx, ρ)

        # build RHS of A*x = b; b = a + ρ*D'P(D*x)
        mul!(b, D', Pz)
        axpby!(1, a, ρ, b)
    end

    # solve the linear system
    linsolve!(linsolver, x, A, b)

    return 1.0
end

function cvxclst_iter(::ADMM, prob, ρ, μ)
    @unpack x, y, λ = prob.variables
    @unpack ∇²f = prob.derivatives
    @unpack D, W, P, a = prob.operators
    @unpack z, Pz, v, b, tmpx = prob.buffers
    linsolver = prob.linsolver

    # x block update
    @. v = y - λ
    if linsolver isa LSQRWrapper
        # build LHS of A*x = b
        # forms a BlockMap so non-allocating
        # however, A*x and A'b have small allocations due to views?
        A₁ = LinearMap(I, size(D, 2))
        A₂ = D
        A = QuadLHS(A₁, A₂, tmpx, √μ) # A = [I; √μ*D]

        # build RHS of A*x = b; b = [a; √μ * (y-λ)]
        n = length(a)
        copyto!(b, 1, a, 1, length(a))
        for k in eachindex(v)
            b[n+k] = √μ * v[k]
        end
    else
        # assemble LHS
        A = ProxDistHessian(∇²f, D'D, tmpx, μ)

        # build RHS of A*x = b; b = a + μ*D'(y-λ)
        mul!(b, D', v)
        axpby!(1, a, μ, b)
    end

    # solve the linear system
    linsolve!(linsolver, x, A, b)

    # y block update
    nrows, ncols = D.d, round(Int, D.M / D.d) # TODO: cache m = M / d that was used to make D
    mul!(z, D, x)  # compute difference
    α = (ρ / μ)
    @. v = z + λ
    Pv = Pz
    Pv_mat = reshape(Pv, nrows, ncols)
    copyto!(Pv, v)
    rmul!(Pv_mat, W) # update to weighted differences
    P(Pv_mat)        # project to sparsity
    rdiv!(Pv_mat, W) # map back to original scale
    @inbounds @simd for j in eachindex(y)
        y[j] = α/(1+α) * Pv[j] + 1/(1+α) * v[j]
    end

    # λ block update
    @inbounds for j in eachindex(λ)
        λ[j] = λ[j] + (z[j] - y[j])
    end

    return μ
end

function cvxclst_iter(::MMSubSpace, prob, ρ, μ)
    @unpack x = prob.variables
    @unpack ∇²f, ∇h, ∇f, ∇q, G = prob.derivatives
    @unpack D, W, P, a = prob.operators
    @unpack β, b, z, v, Pz, tmpx, tmpGx1, tmpGx2 = prob.buffers
    linsolver = prob.linsolver

    # projection + gradient
    nrows, ncols = D.d, round(Int, D.M / D.d) # TODO: cache m = M / d that was used to make D
    @. ∇f = x - a
    mul!(z, D, x)  # compute difference
    Pz_mat = reshape(Pz, nrows, ncols)
    copyto!(Pz, z)
    rmul!(Pz_mat, W) # update to weighted differences
    P(Pz_mat)
    rdiv!(Pz_mat, W) # map back to original scale
    @. v = z - Pz
    mul!(∇q, D', v)
    @. ∇h = ∇f + ρ * ∇q

    if linsolver isa LSQRWrapper
        # build LHS of A*x = b
        # forms a BlockMap so non-allocating
        # however, A*x and A'b have small allocations due to views?
        A₁ = LinearMap(I, size(D, 2))
        A₂ = D
        A = MMSOp1(A₁, A₂, G, tmpGx1, tmpGx2, √ρ)

        # build RHS of A*x = b; b = [a; √ρ * P(D*x)]
        n = length(a)
        for j in 1:n
            b[j] = -∇f[j]   # A₁*x - a
        end
        for k in eachindex(Pz)
            b[n+k] = √ρ * Pz[k]
        end
    else
        # build LHS, A = G'*H*G = G'*(∇²f + ρ*D'D)*G
        H = ProxDistHessian(∇²f, D'D, tmpx, ρ)
        A = MMSOp2(H, G, tmpGx1, tmpGx2)

        # build RHS, b = -G'*∇h
        mul!(b, G', ∇h)
        @. b = -b
    end

    # solve the linear system
    linsolve!(linsolver, x, A, b)

    # apply the update, x = x + G*β
    mul!(x, G, β, true, true)

    return norm(β)
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

function assign_classes!(class, A, Δ, U, magnitude)
    n = size(Δ, 1)

    Δ = pairwise(Euclidean(), U, dims = 2)

    # update adjacency matrix
    for j in 1:n, i in j+1:n
        if log10(Δ[i,j]) ≤ magnitude
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

function assign_classes(U, magnitude::Real=-2)
    n = size(U, 2)
    A = zeros(Bool, n, n)
    Δ = zeros(n, n)
    class = zeros(Int, n)

    return assign_classes!(class, A, Δ, U, magnitude)
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
    neighbors = trivec_index.(n, (i+1):n, i)
    keep = neighbors[sortperm(w[neighbors], rev = true)[1:k]]

    for i in 2:(n-1)
        group_A = trivec_index.(n, (i+1):n, i)
        group_B = trivec_index.(n, i, 1:(i-1))
        neighbors = [group_A; group_B]
        knn = neighbors[sortperm(w[neighbors], rev = true)[1:k]]
        keep = union(knn, keep)
    end

    i = n
    neighbors = trivec_index.(n, i, 1:(i-1))
    knn = neighbors[sortperm(w[neighbors], rev = true)[1:k]]
    keep = union(knn, keep)

    W_knn = zero(W)

    for j in 1:n, i in j+1:n
        l = trivec_index(n, i, j)
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

Optionally, pass a `rng` to make results reproducible or specify the standard deviation `sigma`.
"""
function gaussian_cluster(centroid, n; rng::AbstractRNG=StableRNG(1234), sigma::Real=0.1)
    d = length(centroid)
    cluster = centroid .+ sigma * randn(rng, d, n)

    return cluster
end

#################
#   examples    #
#################

function convex_clustering_data(file)
    dir = dirname(@__DIR__) # should point to src
    dir = dirname(dir)      # should point to top-level directory

    df = CSV.read(joinpath(dir, "data", file), DataFrame, copycols = true)

    if basename(file) == "mammals.dat" # classes in column 2
        # extract data as features × columns
        data = Matrix{Float64}(df[:, 3:end-1])
        X = copy(transpose(data))

        # retrieve class assignments
        class_name = unique(df[:,2])
        classes = convert(Vector{Int}, indexin(df[:,2], class_name))
    else # classes in last column
        # extract data as features × columns
        data = Matrix{Float64}(df[:,1:end-1])
        X = copy(transpose(data))

        # retrieve class assignments
        class_name = unique(df[:,end])
        classes = convert(Vector{Int}, indexin(df[:,end], class_name))
    end

    return X, classes, length(class_name)
end
