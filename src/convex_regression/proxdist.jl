function cvxreg_iteration!(θ, ∇θ, ξ, ∇ξ, U, V, y, X, ρ)
    # compute the intermediate U = B*z by blocks
    apply_D!(U, θ)        # vec(U) = D*θ
    apply_H!(V, X, ξ)     # vec(V) = H*ξ
    # BLAS.axpy!(1.0, V, U)
    @. U = U + V

    # compute projection onto non-positive orthant and store in V
    @. V = min(0, U)

    # finish evaluating B*z - proj(B*z) and store in U
    # BLAS.axpy!(-1.0, V, U)
    @. U = U - V

    # form the gradient
    apply_Dt!(∇θ, U)
    @. ∇θ = θ - y + ρ*∇θ    # θ block
    apply_Ht!(∇ξ, X, U)
    @. ∇ξ = ρ*∇ξ            # ξ blocks

    # compute the step size
    a = dot(∇θ, ∇θ) # norm^2 of ∇θ
    b = dot(∇ξ, ∇ξ) # norm^2 of ∇ξ
    
    apply_D!(U, ∇θ)
    apply_H!(V, X, ∇ξ)
    # BLAS.axpy!(1.0, V, U)
    @. U = U + V
    c = dot(U, U)   # norm^2 of B*∇(θ,ξ)
    
    γ = (a + b) / (a + ρ*c)

    # apply the steepest descent update
    @. θ = θ - γ*∇θ
    @. ξ = ξ - γ*∇ξ

    return γ, a + b
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

    for iteration in 1:maxiters
        γ, grad = cvxreg_iteration!(θ, ∇θ, ξ, ∇ξ, U, V, y, X, ρ)

        if iteration % 1000 == 0
            ρ *= 1 + 1e-2
        end
        # iteration % 500 == 0 && println("grad = $(grad), ρ = $(ρ)")
    end
    
    return θ, ξ, ∇θ, ∇ξ
end
