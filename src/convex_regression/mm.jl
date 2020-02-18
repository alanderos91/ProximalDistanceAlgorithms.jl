function cvxreg_mm!()

end

function cvxreg_fit(::MM, y, X;
    ρ_init::Real      = 1.0,
    maxiters::Integer = 100,
    penalty::Function = __default_schedule,
    history::FuncLike = __default_logger) where FuncLike
    # extract problem information
    d, n = size(X)

    # allocate function estimates and subgradients
    θ = copy(y)
    ξ = zeros(d, n)

    # extras
    ρ = ρ_init

    for iteration in 1:maxiters
        data = cvxreg_mm!()

        ρ = penalty(ρ, iteration)

        history(data, iteration)
    end

    return θ, ξ
end
