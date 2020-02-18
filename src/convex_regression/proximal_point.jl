function __prox_loss!(θ, y)
    @. θ = θ - y
    loss = norm(θ)
    if loss ≥ 0.5
        @. θ = y + (1 - 0.5/loss)*θ
    else
        @. θ = y
    end

    return θ
end

function __prox_penalty!(c, ρ)
    α = 1/(1+ρ)
    @. c = α*c + (1-α)*min(0, c)

    return c
end

function __proj_kernel!(θ, ξ, vec_ξ, c, b, X, Dt, Ht, T)
    # apply linear operators
    __apply_D_plus_H!(b, X, θ, ξ)
    @. c = c - b

    # linear solve
    mul!(b, T, c)

    # update variables
    mul!(θ, Dt, b, 1.0, 1.0)
    mul!(vec_ξ, Ht, b, 1.0, 1.0)
    @. c = c - b

    return θ, ξ, vec_ξ, c
end

function cvxreg_fit(::ProximalPoint, y, X;
    ρ_init::Real      = 1.0,
    maxiters::Integer = 100,
    penalty::Function = __default_schedule,
    history::FuncLike = __default_logger) where FuncLike
    # extract problem information
    d, n = size(X)

    # build constraint matrices
    Dt, Ht, T = __build_matrices(X)

    # allocate vectors for intermediates
    c = zeros(n*(n-1))
    b = zeros(n*(n-1))

    # allocate function estimates and subgradients
    θ = copy(y)
    ξ = zeros(d, n)
    vec_ξ = vec(ξ)

    # extras
    ρ = ρ_init

    # initialize c = B*z algorithm
    __apply_D_plus_H!(c, X, θ, ξ)

    for iteration in 1:maxiters
        __proj_kernel!(θ, ξ, vec_ξ, c, b, X, Dt, Ht, T)
        __prox_loss!(θ, y)
        __prox_penalty!(c, ρ)

        ρ = penalty(ρ, iteration)

        history(data, iteration)
    end

    return θ, ξ
end
