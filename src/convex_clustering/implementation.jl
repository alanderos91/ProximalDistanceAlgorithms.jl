@doc raw"""
    convex_clustering(algorithm::AlgorithmOption, weights, data; kwargs...)

Cluster `data` using an approximate convexification of ``k``-means.

**Note**: This function should only be used for exploratory analyses.
See [`convex_clustering_path`](@ref) for an algorithm that returns a list of candidate clusterings.

The `data` are assumed to be arranged as a `features` by `samples` matrix.
A sparsity parameter `nu` quantifies the number of constraints `data[:,i] ≈ data[:,j]` that are allowed to be violated. The choice `nu = 0` assigns
each sample to the same cluster whereas `nu = binomial(samples, 2)` forces
samples into their own clusters. Setting `rev=false` reverses this relationship.

See also: [`MM`](@ref), [`StepestDescent`](@ref), [`ADMM`](@ref), [`MMSubSpace`](@ref), [`initialize_history`](@ref), [`optimize!`](@ref), [`anneal!`](@ref)

# Keyword Arguments

- `nu::Integer=0`: A sparsity parameter that controls clusterings.
- `rev::Bool=true`: A flag that changes the interpretation of `nu` from constraint violations (`rev=true`) to constraints satisfied (`rev=false`).
This indirectly affects the performance of the algorithm and should only be used when crossing the threshold `nu = binomial(samples, 2) ÷ 2`.
- `ls=Val(:LSQR)`: Choice of linear solver for `MM`, `ADMM`, and `MMSubSpace` methods. Choose one of `Val(:LSQR)` or `Val(:CG)` for LSQR or conjugate gradients, respectively.
"""
function convex_clustering(algorithm::AlgorithmOption, weights, data; nu::Integer=0, ls::LS=Val(:LSQR), kwargs...) where LS
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
    P = ColumnL0Projection(nu, collect(1:m), zeros(Int, m), zeros(m))
    a = copy(x)
    w = Float64[]
    for j in 1:n, i in j+1:n push!(w, weights[i,j]) end
    D = CvxClusterFM(d, n)
    W = kron(Diagonal(w), LinearMap(I, d))
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
    start::Real=0.5,
    stepsize::Real=0.05,
    radius::Real=2,
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
    W = kron(Diagonal(w), LinearMap(I, d))

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

    # arrays needed for projection
    projection_idx, projection_buffer, projection_scores = collect(1:m), zeros(Int, m), zeros(m)

    # allocate outputs
    assignment = Vector{Int}[]
    number_classes = Int[]
    sparsity = Float64[]

    # initialize solution path heuristic
    mul!(z, D, x)
    nconstraint = 0
    for k in 1:m
        # extract the corresponding column
        startix = d * (k-1) + 1
        stopix  = startix + d - 1
        col     = @view z[startix:stopix]
        colnorm_k = norm(col)

        # derivatives within 10^-3 are set to 0
        nconstraint += (log10(colnorm_k) ≤ -3)
    end

    rmax = 1.0
    rstep = stepsize
    if 0 ≤ start ≤ rmax
        r = 1 - start
    else
        r = 1 - nconstraint / m
    end

    prog = ProgressThresh(zero(r), "Searching clustering path...")
    while r ≥ 0
        nu = round(Int, r * m)
        P = ColumnL0Projection(nu, projection_idx, projection_buffer, projection_scores)
        operators = (D = D, W = W, P = P, a = a)
        prob = ProxDistProblem(variables, old_variables, derivatives, operators, buffers, views, linsolver)
        prob_tuple = (objective, algmap, prob)

        result = optimize!(algorithm, prob_tuple; callback=callback, kwargs...)

        # update history
        callback(Val(:inner), algorithm, result.iters, result, prob, 0.0, 0.0)

        # record current solution
        _, assigned_classes, classes = assign_classes(X, radius)
        s = round(100 * (1-r), sigdigits=4)
        push!(assignment, assigned_classes)
        push!(number_classes, classes)
        push!(sparsity, s)

        showvalues = [
            (:sparsity, s),
            (:classes, classes),
            (:distance, sqrt(result.distance)),
            (:gradient, sqrt(result.gradient)),
        ]

        # count satisfied constraints
        nconstraint = 0
        for k in 1:m
            # extract the corresponding column
            startix = d * (k-1) + 1
            stopix  = startix + d - 1
            col     = @view z[startix:stopix]
            colnorm_k = norm(col)

            # derivatives within 10^-2 are set to 0
            nconstraint += (log10(colnorm_k) ≤ -radius)
        end

        # decrease r with a heuristic that guarantees a decrease
        rnew = 1 - nconstraint / m
        if rnew < r - rstep
            r = rnew
        else
            r = r - rstep
        end
        
        ProgressMeter.update!(prog, r, showvalues=showvalues)
    end

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
    mul!(Pz, W, z) # update to weighted differences
    Pz_mat = reshape(Pz, nrows, ncols)
    copyto!(Pz, z); P(Pz_mat)
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
    @unpack x = prob.variables
    @unpack ∇²f = prob.derivatives
    @unpack D, a = prob.operators
    @unpack b, Pz, tmpx = prob.buffers
    linsolver = prob.linsolver

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
    Pv = Pz; mul!(Pv, W, v) # update to weighted differences
    P(reshape(Pv, nrows, ncols)) # project to top k order statistics
    @inbounds @simd for j in eachindex(y)
        y[j] = α/(1+α) * Pv[j] + 1/(1+α) * v[j]
    end

    # λ block update
    @inbounds @simd for j in eachindex(λ)
        λ[j] = λ[j] + z[j] - y[j]
    end

    return μ
end

function cvxclst_iter(::MMSubSpace, prob, ρ, μ)
    @unpack x = prob.variables
    @unpack ∇²f, ∇h, ∇f, G = prob.derivatives
    @unpack D, a = prob.operators
    @unpack β, b, Pz, tmpx, tmpGx1, tmpGx2 = prob.buffers
    linsolver = prob.linsolver

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
