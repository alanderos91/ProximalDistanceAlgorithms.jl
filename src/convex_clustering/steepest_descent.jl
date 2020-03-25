function cvxclst_steepest_descent!(Q, U, X, W, ρ, Iv, Jv, v)
    # 1. form the gradient:

    # 1a. partial evaluation: Dt*Wt*[W*D*u - P(y)]
    fill!(Iv, 0); fill!(Jv, 0); fill!(v, 0); fill!(Q, 0)
    __find_large_blocks!(Iv, Jv, v, W, U)
    __accumulate_averaging_step!(Q, W, U)
    __apply_sparsity_correction!(Q, W, U, Iv, Jv)

    # 1b. evaluate loss, penalty, and objective:
    loss = 0.5 * (dot(U,U) - 2*dot(U,X) + dot(X,X))
    penalty = dot(Q, Q)
    objective = loss + 0.5*ρ*penalty

    # 1c. finish forming the gradient: (u-x) + ρ*Dt*Wt*[W*D*u - P(y)]
    for K in eachindex(Q)
        Q[K] = (U[K] - X[K]) + ρ*Q[K]
    end

    # 2. compute stepsize
    a = dot(Q, Q)                               # norm^2 of gradient
    b = __evaluate_weighted_gradient_norm(W, Q) # norm^2 of W*D*gradient
    γ = a / (a + ρ*b)

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
    U = zero(X)

    # allocate gradient and auxilliary variable
    Q = similar(X)

    # initialize data structures for sparsity projection
    Iv = zeros(Int, K)
    Jv = zeros(Int, K)
    v  = zeros(K)

    # initialize penalty coefficient
    ρ = ρ_init

    # call main subroutine
    convex_clustering!(Q, U, Iv, Jv, v, X, W, ρ, maxiters, penalty, history)

    return U
end

function convex_clustering!(Q, U, Iv, Jv, v, X, W, ρ, maxiters, penalty, history)
    for iteration in 1:maxiters
        # iterate the algorithm map
        data = cvxclst_steepest_descent!(Q, U, X, W, ρ, Iv, Jv, v)

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
        convex_clustering!(Q, U, Iv, Jv, v, X, W, ρ, maxiters, penalty, history)

        # save solution for the subproblem
        copyto!(path[i], U)
    end

    return path
end
