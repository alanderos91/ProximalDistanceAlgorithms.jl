function imgtvd_steepest_descent!(Q, dx, dy, U, W, ρ, epsilon)
    m, n = size(U) # m rows, n columns

    # compute derivatives on columns, dx = Dx * U
    for j in 1:n, i in 1:m-1
        dx[i,j] = U[i+1,j] - U[i,j]
    end

    # compute derivatives on rows, dy = Dy * u
    for j in 1:n-1, i in 1:m
        dy[i,j] = U[i,j+1] - U[i,j] # TODO: do this by fixing a column
    end

    # project onto Euclidean ball of radius epsilon
    norm_d = sqrt(dot(dx, dx) + dot(dy, dy))
    c = epsilon / norm_d
    c = norm_d > 1 ? c : one(c)
    @. dx = dx - c * dx
    @. dy = dy - c * dy

    # evaluate loss, penalty, and objective
    loss      = dot(U, U) - 2*dot(U, W) + dot(W, W)
    penalty   = dot(dx, dx) + dot(dy, dy)
    objective = 0.5 * (loss + ρ*penalty)

    # mul. by tranpose: Dx
    for j in 1:n
        Q[1,j] = -dx[1,j]
    end

    for j in 1:n, i in 2:m-1
        Q[i,j] = dx[i-1,j] - dx[i,j]
    end

    for j in 1:n
        Q[m,j] = dx[m-1,j]
    end

    # mul. by transpose: Dy
    for i in 1:m
        Q[i,1] += -dy[i,1]
    end

    for j in 2:n-1, i in 1:m
        Q[i,j] += dy[i,j-1] - dy[i,j]
    end

    for i in 1:m
        Q[i,n] += dy[i,n-1]
    end

    # complete the gradient
    for idx in eachindex(Q)
        Q[idx] = U[idx] - W[idx] + ρ*Q[idx]
    end

    # compute step size
    a = dot(Q, Q)
    b = zero(a)
    for j in 1:n, i in 1:m-1
        b = b + (Q[i+1,j] - Q[i,j])^2
    end
    for j in 1:n-1, i in 1:m
        b = b + (Q[i,j+1] - Q[i,j])^2
    end
    γ = a / (a + ρ*b + eps())
    normgrad = sqrt(a)

    # apply the update
    for idx in eachindex(U)
        U[idx] = U[idx] - γ*Q[idx]
    end

    return γ, normgrad, loss, objective, penalty
end

function image_denoise(::SteepestDescent, W;
    ρ_init::Real      = 1.0,
    maxiters::Integer = 100,
    penalty::Function = __default_schedule,
    history::FuncLike = __default_logger,
    epsilon::Real     = 1.0) where FuncLike
    #
    m, n = size(W)   # m pixels by n pixels
    T = eltype(W)    # element type

    U  = copy(W)            # solution
    Q  = zero(W)            # gradient
    dx = zeros(T, m-1, n)   # derivatives along rows
    dy = zeros(T, m, n-1)   # derivatives along cols

    ρ = ρ_init

    for iteration in 1:maxiters
        # iterate the algorithm map
        data = imgtvd_steepest_descent!(Q, dx, dy, U, W, ρ, epsilon)

        # check for updates to the penalty coefficient
        ρ = penalty(ρ, iteration)

        # check for updates to the convergence history
        history(data, iteration)
    end

    return U
end
