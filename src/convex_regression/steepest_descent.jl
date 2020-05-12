function cvxreg_evaluate!(∇θ, ∇ξ, U, V, y, X, optvars, ρ)
    θ = optvars.θ
    ξ = optvars.ξ

    # evaluate loss
    loss = dot(θ,θ) - 2*dot(y, θ) + dot(y,y)

    # compute B*z = D*θ + H*ξ
    apply_D_plus_H!(U, X, θ, ξ)

    # project onto non-positive orthant
    @. V = min(0, U)

    # evaluate B*z - proj(B*z) and check distance penalty
    @. U = U - V
    penalty = dot(U, U)

    # form the gradient
    apply_Dt!(∇θ, U)
    @. ∇θ = θ - y + ρ*∇θ    # θ block
    apply_Ht!(∇ξ, X, U)
    @. ∇ξ = ρ*∇ξ            # ξ blocks

    a = dot(∇θ, ∇θ)
    b = dot(∇ξ, ∇ξ)

    return loss, penalty, a, b
end

function cvxreg_steepest_descent!(optvars, ∇θ, ∇ξ, U, X, ρ, a, b)
    θ = optvars.θ
    ξ = optvars.ξ

    # compute the step size
    apply_D_plus_H!(U, X, ∇θ, ∇ξ)
    c = dot(U, U)

    γ = (a + b) / (a + ρ*c + eps())

    # apply the steepest descent update
    @. θ = θ - γ*∇θ
    @. ξ = ξ - γ*∇ξ

    return γ
end

function cvxreg_fit(::SteepestDescent, y, X;
    ρ_init::Real      = 1.0,
    maxiters::Integer = 100,
    penalty::Function = __default_schedule,
    history::histT    = nothing,
    ftol::Real        = 1e-6,
    dtol::Real        = 1e-4,
    accel::accelT     = Val(:none)) where {histT, accelT}
    #
    # extract problem information
    d, n = size(X)

    # allocate matrices for intermediates
    U = zeros(n, n)
    V = zeros(n, n)

    # allocate function estimates and subgradients
    θ = copy(y)
    ξ = zeros(d, n)

    # collect into named tuple
    optvars = (θ = θ, ξ = ξ)

    # allocate gradients
    ∇θ = zero(θ)
    ∇ξ = zero(ξ)

    # construct acceleration strategy
    strategy = get_acceleration_strategy(accel, optvars)

    # initialize
    ρ = ρ_init

    loss, distance, a, b = cvxreg_evaluate!(∇θ, ∇ξ, U, V, y, X, optvars, ρ)
    data = package_data(loss, distance, ρ, sqrt(a+b), zero(loss))
    update_history!(history, data, 0)

    loss_old = loss
    loss_new = Inf
    dist_old = distance
    dist_new = Inf
    iteration = 1

    while not_converged(loss_old, loss_new, dist_old, dist_new, ftol, dtol) && iteration ≤ maxiters
        # iterate the algorithm map
        stepsize = cvxreg_steepest_descent!(optvars, ∇θ, ∇ξ, U, X, ρ, a, b)

        # penalty schedule + acceleration
        ρ_new = penalty(ρ, iteration)             # check for updates to the penalty coefficient
        ρ != ρ_new && restart!(strategy, optvars) # check for restart due to changing objective
        apply_momentum!(optvars, strategy)        # apply acceleration strategy
        ρ = ρ_new                                 # update penalty

        # convergence history
        loss, distance, a, b = cvxreg_evaluate!(∇θ, ∇ξ, U, V, y, X, optvars, ρ)
        data = package_data(loss, distance, ρ, sqrt(a+b), stepsize)
        update_history!(history, data, iteration)

        loss_old = loss_new
        loss_new = loss
        dist_old = dist_new
        dist_new = distance
        iteration += 1
    end

    return θ, ξ
end
