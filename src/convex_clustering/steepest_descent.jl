function cvxclst_steepest_descent!(Q, U, X, W, wbar, ρ, Iv, Jv, v)
    # 1. form the gradient:

    # 1a. partial evaluation: ρ*Dt*Wt*[W*D*u - P(y)]
    fill!(Iv, 0); fill!(Jv, 0); fill!(v, 0); fill!(Q, 0)
    __find_large_blocks!(Iv, Jv, v, W, U)
    __accumulate_averaging_step!(Q, W, wbar, U)
    __accumulate_sparsity_correction!(Q, W, U, Iv, Jv)

    # 1b. evaluate loss, penalty, and objective:
    loss = 0.5 * (dot(U,U) - 2*dot(U,X) + dot(X,X))
    penalty = dot(Q, Q)
    objective = loss + 0.5*ρ*penalty

    # 1c. finish forming the gradient with the final centering step
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
    history::FuncLike = __default_logger) where FuncLike
    #
    return nothing
end
