function metric_steepest_descent!(X, Q, W, D, ρ)
    n = size(X, 1)

    # 1. form the gradient:
    # Q = W ∘ (X - D) + ρ*(X - (X)_+ + TtT*vec(X) - Tt*(T*vec(X))_-)

    # 1a. (X - (X)_+ + TtT*vec(X) - Tt*(T*vec(X))_-)
    fill!(Q, 0)
    __apply_TtT_minus_npproj!(Q, X)
    __accumulate_I_minus_nnproj!(Q, X)

    # 1b. evaluate loss, penalty, and objective:
    loss = 0.5 * (dot(X,X) - 2*dot(D,X) + dot(D,D))
    penalty = dot(Q, Q)
    objective = loss + 0.5*ρ*penalty

    # 1c. finish forming the gradient
    # @. Q = W .* (X - D) + ρ*Q
    for j in 1:n, i in j+1:n
        Q[i,j] = W[i,j] * (X[i,j] - D[i,j]) + ρ*Q[i,j]
    end

    # 2. compute stepsize
    a = dot(Q, Q)                               # norm^2 of gradient
    b = __apply_T_evaluate_norm_squared(Q)      # norm^2 of T*gradient
    c = __evaulate_weighted_norm_squared(W, Q)  # norm^2 of W^1/2*gradient
    γ = a / (c + ρ*(a + b))

    # 3. apply the update
    # @. X = X - γ*Q
    for j in 1:n, i in j+1:n
        X[i,j] = X[i,j] - γ*Q[i,j]
    end

    return γ, sqrt(a), loss, objective, penalty
end

function metric_projection(::SteepestDescent, W, D;
    ρ_init::Real      = 1.0,
    maxiters::Integer = 100,
    penalty::Function = __default_schedule,
    history::FuncLike = __default_logger) where FuncLike
    #
    # extract problem dimensions
    n = size(D, 1)

    # assume symmetry in D and zeros in diagonal
    D_tri = LowerTriangular(D)
    W_tri = LowerTriangular(W)

    # allocate optimization variable
    X = copy(D_tri)

    # allocate gradient
    Q = similar(X)

    # initialize penalty coefficient
    ρ = ρ_init

    for iteration in 1:maxiters
        # iterate the algorithm map
        data = metric_steepest_descent!(X, Q, W_tri, D_tri, ρ)

        # check for updates to the penalty coefficient
        ρ = penalty(ρ, iteration)

        # check for updates to the convergence history
        history(data, iteration)
    end

    return X
end
