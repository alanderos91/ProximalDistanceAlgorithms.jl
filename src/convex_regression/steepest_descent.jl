function cvxreg_steepest_descent!(θ, ∇θ, ξ, ∇ξ, U, V, y, X, ρ)
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

    # compute the step size
    a = dot(∇θ, ∇θ)
    b = dot(∇ξ, ∇ξ)

    apply_D_plus_H!(U, X, ∇θ, ∇ξ)
    c = dot(U, U)

    γ = (a + b) / (a + ρ*c)

    # apply the steepest descent update
    @. θ = θ - γ*∇θ
    @. ξ = ξ - γ*∇ξ

    # evaluate loss and objective
    loss = 0.5 * (dot(θ,θ) - 2*dot(y, θ) + dot(y,y))
    objective = loss + 0.5*ρ*penalty

    return γ, sqrt(a+b), loss, objective, penalty
end

function cvxreg_fit(::SteepestDescent, y, X;
    ρ_init::Real      = 1.0,
    maxiters::Integer = 100,
    penalty::Function = __default_schedule,
    history::FuncLike = __default_logger) where FuncLike
    # extract problem information
    d, n = size(X)

    # allocate matrices for intermediates
    U = zeros(n, n)
    V = zeros(n, n)

    # allocate function estimates and subgradients
    θ = copy(y)
    ξ = zeros(d, n)

    # allocate gradients
    ∇θ = zero(θ)
    ∇ξ = zero(ξ)

    # extras
    ρ = ρ_init

    for iteration in 1:maxiters
        data = cvxreg_steepest_descent!(θ, ∇θ, ξ, ∇ξ, U, V, y, X, ρ)

        ρ = penalty(ρ, iteration)

        history(data, iteration)
    end

    return θ, ξ
end
