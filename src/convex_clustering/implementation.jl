@doc raw"""
    convex_clustering(algorithm::AlgorithmOption, weights, data; kwargs...)

Cluster `data` using an approximate convexification of ``k``-means.

**Note**: This function should only be used for exploratory analyses.
See [`convex_clustering_path`](@ref) for an algorithm that returns a list of candidate clusterings.

The `data` are assumed to be arranged as a `features` by `samples` matrix.
A sparsity parameter `nu` quantifies the number of constraints `data[:,i] ≈ data[:,j]` that are allowed to be violated. The choice `nu = 0` assigns
each sample to the same cluster whereas `nu = binomial(samples, 2)` forces
samples into their own clusters. Setting `rev=false` reverses this relationship.

See also: [`MM`](@ref), [`StepestDescent`](@ref), [`ADMM`](@ref), [`MMSubSpace`](@ref), [`initialize_history`](@ref)

# Keyword Arguments

- `nu::Integer=0`: A sparsity parameter that controls clusterings.
- `rev::Bool=true`: A flag that changes the interpretation of `nu` from constraint violations (`rev=true`) to constraints satisfied (`rev=false`).
This indirectly affects the performance of the algorithm and should only be used when crossing the threshold `nu = binomial(samples, 2) ÷ 2`.
- `rho::Real=1.0`: An initial value for the penalty coefficient. This should match with the choice of annealing schedule, `penalty`.
- `mu::Real=1.0`: An initial value for the step size in `ADMM()`.
- `ls=Val(:LSQR)`: Choice of linear solver for `MM`, `ADMM`, and `MMSubSpace` methods. Choose one of `Val(:LSQR)` or `Val(:CG)` for LSQR or conjugate gradients, respectively.
- `maxiters::Integer=100`: The maximum number of iterations.
- `penalty::Function=__default_schedule__`: A two-argument function `penalty(rho, iter)` that computes the penalty coefficient at iteration `iter+1`. The default setting does nothing.
- `history=nothing`: An object that logs convergence history.
- `rtol::Real=1e-6`: A convergence parameter measuring the relative change in the loss model, $\frac{1}{2} \|(x-y)\|^{2}$.
- `atol::Real=1e-4`: A convergence parameter measuring the magnitude of the squared distance penalty $\frac{\rho}{2} \mathrm{dist}(Dx,C)^{2}$.
- `accel=Val(:none)`: Choice of an acceleration algorithm. Options are `Val(:none)` and `Val(:nesterov)`.
"""
function convex_clustering(algorithm::AlgorithmOption, weights, data;
    nu::Integer=0,
    rev::Bool=true,
    rho::Real=1.0,
    mu::Real=1.0,
    ls::LS=Val(:LSQR), kwargs...) where LS
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
    block_norm = zeros(m)
    cache = zeros(m)
    compute_proj = SparseProjectionClosure(rev, nu)
    P = BlockSparseProjection(d, block_norm, cache, compute_proj)
    a = copy(x)
    D = CvxClusterFM(d, n)
    operators = (D = D, P = P, a = a)

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
                b = similar(typeof(x), size(A₂, 1))
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
    prob = ProxDistProblem(variables, derivatives, operators, buffers, views, linsolver)

    # solve the optimization problem
    optimize!(algorithm, objective, algmap, prob, rho, mu; kwargs...)

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

See also: [`MM`](@ref), [`StepestDescent`](@ref), [`ADMM`](@ref), [`MMSubSpace`](@ref), [`initialize_history`](@ref)

# Keyword Arguments

- `stepsize::Float64=0.05`: A value between `0` and `1` that discretizes the search space. If no additional points are found to coalesce, then decrease `nu` by `stepsize*nu_max`.
- `rho::Real=1.0`: An initial value for the penalty coefficient. This should match with the choice of annealing schedule, `penalty`.
- `mu::Real=1.0`: An initial value for the step size in `ADMM()`.
- `ls=Val(:LSQR)`: Choice of linear solver for `MM`, `ADMM`, and `MMSubSpace` methods. Choose one of `Val(:LSQR)` or `Val(:CG)` for LSQR or conjugate gradients, respectively.
- `maxiters::Integer=100`: The maximum number of iterations.
- `penalty::Function=__default_schedule__`: A two-argument function `penalty(rho, iter)` that computes the penalty coefficient at iteration `iter+1`. The default setting does nothing.
- `history=nothing`: An object that logs convergence history.
- `rtol::Real=1e-6`: A convergence parameter measuring the relative change in the loss model, $\frac{1}{2} \|(x-y)\|^{2}$.
- `atol::Real=1e-4`: A convergence parameter measuring the magnitude of the squared distance penalty $\frac{\rho}{2} \mathrm{dist}(Dx,C)^{2}$.
- `accel=Val(:none)`: Choice of an acceleration algorithm. Options are `Val(:none)` and `Val(:nesterov)`.
"""
function convex_clustering_path(algorithm::AlgorithmOption, weights, data;
    stepsize::Real=0.05,
    rho::Real=1.0,
    mu::Real=1.0,
    history::histT=nothing,
    ls::LS=Val(:LSQR), kwargs...) where {histT, LS}
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
    block_norm = zeros(m)   # stores column-wise distances
    cache = zeros(m)        # mirrors block_norm; cache for pivot search
    a = copy(x)
    D = CvxClusterFM(d, n)

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

    # allocate output
    assignment = Vector{Int}[]
    ν_path = Int[]

    # allocate storage for distance matrix
    distance = pairwise(SqEuclidean(), X, dims = 2)

    # initialize solution path heuristic
    νmax = binomial(n, 2)
    ν = νmax-1

    prog = ProgressThresh(0, "Searching clustering path...")
    while ν ≥ 0
        if ν > (νmax >> 1)
            # search by smallest blocks; i.e. partial sort in ascending order
            compute_proj = SparseProjectionClosure(false, νmax - ν)
        else
            # search by largest blocks; i.e. partial sort in descending order
            compute_proj = SparseProjectionClosure(true, ν)
        end

        P = BlockSparseProjection(d, block_norm, cache, compute_proj)
        operators = (D = D, P = P, a = a)
        prob = ProxDistProblem(variables, derivatives, operators, buffers, views, linsolver)

        _, iter, _ = optimize!(algorithm, objective, algmap, prob, rho, mu; kwargs...)

        # update history
        if !(history === nothing)
            f_loss, h_dist, h_ngrad = objective(algorithm, prob, 1.0)
            data = package_data(f_loss, h_dist, h_ngrad, stepsize, 1.0)
            update_history!(history, data, iter-1)
        end

        # record current solution
        push!(assignment, assign_classes(X)[2])
        push!(ν_path, ν)

        # count satisfied constraints
        nconstraint = 0
        pairwise!(distance, SqEuclidean(), X, dims = 2)
        for j in 1:n, i in j+1:n
            # distances within 10^-3 are set to 0
            nconstraint += (log(10, abs(weights[i,j] * distance[i,j])) ≤ -3)
        end

        # decrease ν with a heuristic that guarantees a decrease
        νstep = round(Int, stepsize*νmax)

        if νmax - nconstraint - 1 < ν - νstep
            ν = νmax - nconstraint - 1
        else
            ν = ν - νstep
        end
        ProgressMeter.update!(prog, ν)
    end

    solution_path = (assignment = assignment, nu = ν_path)

    return solution_path
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
    @unpack D, P, a = prob.operators
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
            @inbounds b[n+k] = √μ * v[k]
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
    mul!(z, D, x)
    α = (ρ / μ)
    @. v = z + λ; Pv = Pz; P(Pv, v)
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
        @inbounds for j in 1:n
            b[j] = -∇f[j]   # A₁*x - a
        end
        @inbounds for k in eachindex(Pz)
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
