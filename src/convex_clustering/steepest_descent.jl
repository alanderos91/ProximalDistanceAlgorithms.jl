function cvxclst_steepest_descent!(Q, U, y, p, X, W, ρ, Iv, Jv, v)
    # 1a. form the gradient:
    fill!(Iv, 0); fill!(Jv, 0); fill!(v, 0); fill!(Q, 0); fill!(p, 0)

    sparse_fused_block_projection!(p, Iv, Jv, v, W, U)
    cvxclst_apply_fusion_matrix!(y, W, U)
    # mul!(y, WD, vec(U))

    @. y = y - p
    cvxclst_apply_fusion_matrix_transpose!(Q, W, y)
    # mul!(vec(Q), transpose(WD), y)

    for K in eachindex(Q)
        Q[K] = (U[K] - X[K]) + ρ*Q[K]
    end

    # 1b. evaluate loss, penalty, and objective:
    loss = dot(U,U) - 2*dot(U,X) + dot(X,X)
    penalty = dot(y, y)
    objective = 0.5 * (loss + ρ*penalty)

    # 2. compute stepsize
    a = dot(Q, Q)                               # norm^2 of gradient
    b = __evaluate_weighted_gradient_norm(W, Q) # norm^2 of W*D*gradient
    γ = a / (a + ρ*b + eps())

    # 3. apply the update
    for K in eachindex(U)
        U[K] = U[K] - γ*Q[K]
    end

    return γ, sqrt(a), loss, objective, penalty
end

function convex_clustering(::SteepestDescent, W, X;
    ρ_init::Real      = 1.0,
    maxiters::Integer = 100,
    penalty::Function = __default_schedule,
    history::FuncLike = __default_logger,
    K::Integer        = 2) where FuncLike
    #
    # extract problem dimensions
    d, n = size(X)

    # allocate optimization variable
    U = copy(X)

    # allocate gradient and auxilliary variable
    Q = similar(X)
    y = zeros(eltype(X), d*binomial(n, 2))
    p = zero(y)

    # initialize data structures for sparsity projection
    Iv = zeros(Int, K)
    Jv = zeros(Int, K)
    v  = zeros(K)

    # initialize penalty coefficient
    ρ = ρ_init

    # D = cvxcluster_fusion_matrix(d, n)
    # w = [W[i,j] for j in 1:n for i in j+1:n]
    # WD = Diagonal(repeat(w, inner=d)) * D

    # call main subroutine
    convex_clustering!(Q, U, Iv, Jv, v, y, p, X, W, ρ, maxiters, penalty, history)

    return U
end

function convex_clustering!(Q, U, Iv, Jv, v, y, p, X, W, ρ, maxiters, penalty, history)
    for iteration in 1:maxiters
        # iterate the algorithm map
        data = cvxclst_steepest_descent!(Q, U, y, p, X, W, ρ, Iv, Jv, v)

        # check for updates to the penalty coefficient
        ρ = penalty(ρ, iteration)

        # check for updates to the convergence history
        history(data, iteration)
    end
end

function convex_clustering_path(::SteepestDescent, W, X;
    ρ_init::Real          = 1.0,
    maxiters::Integer     = 100,
    penalty::Function     = __default_schedule,
    history::FunctionLike = __default_logger,
    K_path::AbstractVector{Int}   = Int[]) where FunctionLike
    #
    # extract problem dimensions
    d, n = size(X)

    # allocate optimization variable
    U = zero(X)

    # allocate gradient and auxilliary variable
    Q = similar(X)
    y = zeros(eltype(X), d*binomial(n, 2))
    p = zero(y)

    # allocate solution path
    path = [similar(U) for _ in eachindex(K_path)]

    for (i, K) in enumerate(K_path)
        # initialize data structures for sparsity projection
        Iv = zeros(Int, K)
        Jv = zeros(Int, K)
        v  = zeros(K)

        # set the starting value for the penalty coefficient
        ρ = ρ_init

        # use the old U as a warm-start
        convex_clustering!(Q, U, Iv, Jv, v, y, p, X, W, ρ, maxiters, penalty, history)

        # save solution for the subproblem
        copyto!(path[i], U)
    end

    return path
end
