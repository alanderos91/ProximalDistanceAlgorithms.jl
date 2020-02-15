function cvxreg_iteration!(θ, ∇θ, ξ, ∇ξ, U, V, y, X, ρ)
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

    return γ, (a+b), loss, objective, penalty
end

function cvxreg_fit(y, X; ρ_init = 1.0, maxiters = 100)
    # extract problem information
    d, n = size(X)

    # allocate matrices for intermediates
    U = zeros(n, n)
    V = zeros(n ,n)

    # allocate function estimates and subgradients
    θ = copy(y)
    ξ = zeros(d, n)

    # allocate gradients
    ∇θ = zero(θ)
    ∇ξ = zero(ξ)

    # extras
    ρ = ρ_init

    println("  iter  |    loss   | objective |  penalty  | step size |  gradient")
    for iteration in 1:maxiters
        γ, ∇, l, o, p = cvxreg_iteration!(θ, ∇θ, ξ, ∇ξ, U, V, y, X, ρ)

        if iteration % 10^3 == 0
            @printf("%5.1e | %+2.2e | %+2.2e | %+2.2e | %+2.2e | %+2.2e\n",
                iteration, l, o, p, γ, ∇)
        end
    end

    return θ, ξ, ∇θ, ∇ξ
end
