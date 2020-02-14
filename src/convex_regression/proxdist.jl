function cvxreg_iteration!()
    # compute the intermediate U = B*z by blocks
    apply_D!(U, θ)        # vec(U) = D*θ
    apply_H!(V, X, ξ)     # vec(V) = H*ξ
    BLAS.axpy!(1.0, V, U)

    # compute projection onto non-positive orthant and store in V
    @. V = min(0, U)

    # finish evaluating B*z - proj(B*z) and store in U
    BLAS.axpy!(-1.0, V, U)

    # form the gradient

    # θ block
    apply_Dt!(∇θ, U)
    @. ∇θ = θ - y + ρ*∇θ

    # ξ blocks
    apply_Ht!(∇ξ, X, U)

    # compute the step size
    a = zero(ρ) # norm of ∇θ
    b = zero(ρ) # norm of ∇ξ
    c = zero(ρ) # norm of B*∇(θ,ξ)

    # apply the steepest descent update
    @. θ = θ - γ*∇θ
    @. ξ = ξ - γ*ρ*∇ξ

    return nothing
end
