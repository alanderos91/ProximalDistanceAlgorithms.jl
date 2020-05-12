function metric_evaluate!(Q, W, D, X, ρ)
    n = size(X, 1)

    # evaluate loss, penalty, objective and gradient
    fill!(Q, 0)
    penalty1 = metric_apply_operator1!(Q, X)
    penalty2 = metric_accumulate_operator2!(Q, X)

    loss = zero(eltype(X))

    for j in 1:n, i in j+1:n
        Q[i,j] = W[i,j] * (X[i,j] - D[i,j]) + ρ*Q[i,j]
        loss = loss + W[i,j]*(X[i,j] - D[i,j])^2
    end

    penalty = penalty1 + penalty2
    gradient = dot(Q, Q)

    return loss, penalty, gradient
end

function metric_steepest_descent!(X, Q, W, D, ρ, gradient)
    # evaluate stepsize
    a = gradient                               # norm^2 of gradient
    b = __apply_T_evaluate_norm_squared(Q)     # norm^2 of T*gradient
    c = __evaulate_weighted_norm_squared(W, Q) # norm^2 of W^1/2*gradient
    γ = a / (c + ρ*(a + b))

    # move in the direction of steepest descent
    n = size(X, 1)
    for j in 1:n, i in j+1:n
        X[i,j] = X[i,j] - γ*Q[i,j]
    end

    return γ
end

function metric_projection(::SteepestDescent, W, D;
    ρ_init::Real      = 1.0,
    maxiters::Integer = 100,
    penalty::Function = __default_schedule,
    history::histT    = nothing,
    ftol::Real        = 1e-6,
    dtol::Real        = 1e-4,
    accel::accelT     = Val(:none)) where {histT, accelT}
    #
    # extract problem dimensions
    n = size(D, 1)

    # allocate optimization variable
    X = copy(D)

    # allocate gradient
    Q = similar(X)

    # construct acceleration strategy
    strategy = get_acceleration_strategy(accel, X)

    # initialize
    ρ = ρ_init

    loss, distance, gradient = metric_evaluate!(Q, W, D, X, ρ)
    data = package_data(loss, distance, ρ, gradient, zero(loss))
    update_history!(history, data, 0)

    loss_old = loss
    loss_new = Inf
    dist_old = distance
    dist_new = Inf
    iteration = 1

    while not_converged(loss_old, loss_new, dist_old, dist_new, ftol, dtol) && iteration ≤ maxiters
        # iterate the algorithm map
        stepsize = metric_steepest_descent!(X, Q, W, D, ρ, gradient)
        
        # penalty schedule + acceleration
        ρ_new = penalty(ρ, iteration)       # check for updates to the penalty coefficient
        ρ != ρ_new && restart!(strategy, X) # check for restart due to changing objective
        apply_momentum!(X, strategy)        # apply acceleration strategy
        ρ = ρ_new                           # update penalty

        # convergence history
        loss, distance, gradient = metric_evaluate!(Q, W, D, X, ρ)
        data = package_data(loss, distance, ρ, gradient, stepsize)
        update_history!(history, data, iteration)

        loss_old = loss_new
        loss_new = loss
        dist_old = dist_new
        dist_new = distance
        iteration += 1
    end

    # symmetrize
    for j in 1:n, i in j+1:n
        X[j,i] = X[i,j]
    end

    return X
end
