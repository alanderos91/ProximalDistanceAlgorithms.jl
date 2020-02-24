function cvxreg_mm!(θ, ξ, ∇θ, ∇ξ, U, V, b, b1, b2, w, w1, w2, y, X, T, cg_iterator, ρ)
# function cvxreg_mm!(θ, ξ, y, X, D, H, T, ρ)
    # compute B*z = D*θ + H*ξ
    apply_D_plus_H!(U, X, θ, ξ)

    # project onto non-positive orthant
    @. V = min(0, U)

    # compute blocks b1 = Dt*v and b2 = Ht*v
    apply_Dt!(b1, V)
    apply_Ht!(b2, X, V)

    # finish forming RHS of T*w = b
    @. b1 = b1 + y / ρ

    # evaluate B*z - proj(B*z) and check distance penalty
    @. U = U - V
    penalty = dot(U, U)

    # form the gradient
    apply_Dt!(∇θ, U)
    @. ∇θ = θ - y + ρ*∇θ    # θ block
    apply_Ht!(∇ξ, X, U)
    @. ∇ξ = ρ*∇ξ            # ξ blocks

    # evaluate norm of gradient
    g = sqrt(dot(∇θ, ∇θ) + dot(∇ξ, ∇ξ))

    # solve linear system
    __do_linear_solve!(cg_iterator, b)

    # apply update to θ block
    copyto!(θ, w1)

    # apply update to ξ blocks
    copyto!(ξ, w2)

    # evaluate loss and objective
    loss = 0.5 * (dot(θ,θ) - 2*dot(y, θ) + dot(y,y))
    objective = loss + 0.5*ρ*penalty

    return g, loss, objective, penalty
end

function cvxreg_fit(::MM, y, X;
    ρ_init::Real      = 1.0,
    maxiters::Integer = 100,
    penalty::Function = __default_schedule,
    history::FuncLike = __default_logger) where FuncLike
    # extract problem information
    d, n = size(X)

    # initialize penalty coefficient
    ρ = ρ_init

    # allocate function estimates and subgradients
    θ = copy(y)
    ξ = zeros(d, n)

    # intermediate for mat-vec mult + projection
    U = zeros(n, n)
    V = zeros(n, n)

    # allocate gradients
    ∇θ = zero(θ)
    ∇ξ = zero(ξ)

    # itermediates for linear solve T*w = b
    b = zeros(n+n*d)        # RHS
    b1 = @view b[1:n]       # θ block
    b2 = @view b[n+1:end]   # ξ block

    w = zeros(n+n*d)        # LHS
    w1 = @view w[1:n]       # θ block
    w2 = @view w[n+1:end]   # ξ block

    # sparse matrix T = (A'A/ρ + B'B)
    D = make_D(n)
    H = make_H(X)
    T = sparse([
            Matrix(I(n)/ρ + D'D)  Matrix(D'H)
            Matrix(H'D)           Matrix(H'H)
        ])::SparseMatrixCSC

    cg_iterator = CGIterable(
        T, w,
        similar(b), similar(b), similar(b),
        1e-8, 0.0, 1.0, size(T, 2), 0
    )

    for iteration in 1:maxiters
        data = cvxreg_mm!(θ, ξ, ∇θ, ∇ξ, U, V, b, b1, b2, w, w1, w2, y, X, T, cg_iterator, ρ)
        # data = cvxreg_mm!(θ, ξ, y, X, D, H, T, ρ)

        ρ = penalty(T, n, ρ, iteration)

        history(data, iteration)
    end

    return θ, ξ
end
