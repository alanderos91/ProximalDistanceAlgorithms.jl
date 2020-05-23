function imgtvd_evaluate!(Q, z, dx, dy, index, U, W, ν, ρ)
    m, n = size(U) # m rows, n columns

    # apply the fusion matrix
    imgtvd_apply_D!(z, dx, dy, U)

    # apply sparse projection
    L = length(z)
    if ν ≤ (L >> 1) # P(z) is highly sparse
        # find large entries for projection
        sortperm!(index, z,
            alg = PartialQuickSort(ν), rev = true, initialized = true)

        # compute z - P(z)
        for ix in 1:ν
            k = index[ix]
            z[k] = 0
        end
    else # z - P(z) is highly sparse
        # find small entries for projection
        sortperm!(index, z,
            alg = PartialQuickSort(L-ν), rev = false, initialized = true)

        # compute z - P(z)
        for ix in (L-ν+1):L
            k = index[ix]
            z[k] = 0
        end
    end

    unsafe_copyto!(dx, 1, z, 1, length(dx))
    unsafe_copyto!(dy, 1, z, length(dx) + 1, length(dy))

    fill!(Q, 0)
    # mul. by tranpose: Dx
    imgtvd_apply_Dx_transpose!(Q, dx)

    # mul. by transpose: Dy
    imgtvd_apply_Dy_transpose!(Q, dy)

    # add contribution from extra row
    Q[end] += z[end]

    # finish forming the gradient
    for idx in eachindex(Q)
        Q[idx] = U[idx] - W[idx] + ρ*Q[idx]
    end

    loss = SqEuclidean()(U, W)
    penalty = dot(z, z)
    gradient = dot(Q, Q)

    return loss, penalty, gradient
end

function prox_l2_ball!(dx, dy, z, epsilon)
    norm_d = sqrt(dot(dx, dx) + dot(dy, dy))
    c = epsilon / norm_d
    c = norm_d > 1 ? c : one(c)
    @. dx = dx - c * dx
    @. dy = dy - c * dy

    return nothing
end

function prox_l1_ball!(dx, dy, z, epsilon)
    norm_d = norm(dx, 1) + norm(dy, 1)
    if norm_d - epsilon > epsilon * 1e-8 # constraint is violated
        copyto!(z, dx)
        copyto!(z, length(dx) + 1, dy, 1, length(dy))
        @. z = abs(z)
        sort!(z, rev = true)

        j = 1
        λ = z[j] - epsilon

        while j < length(z) && (z[j] ≤ λ || z[j+1] > λ)
            λ = (j*λ + z[j+1]) / (j+1)
            j += 1
        end

        for k in 1:length(dx)
            dx[k] = dx[k] - sign(dx[k]) * max(0, abs(dx[k]) - λ)
        end
        for k in 1:length(dy)
            dy[k] = dy[k] - sign(dy[k]) * max(0, abs(dy[k]) - λ)
        end
    else # penalty term is zero
        @. dx = zero(eltype(dx))
        @. dy = zero(eltype(dy))
    end

    return nothing
end

function imgtvd_steepest_descent!(U, Q, ρ, gradient)
    m, n = size(U)
    # compute step size
    a = gradient
    b = zero(a)
    for j in 1:n, i in 1:m-1
        @inbounds b = b + (Q[i+1,j] - Q[i,j])^2
    end
    for j in 1:n-1, i in 1:m
        @inbounds b = b + (Q[i,j+1] - Q[i,j])^2
    end
    b += Q[end]^2
    γ = a / (a + ρ*b + eps())

    # move in the direction of steepest descent
    for idx in eachindex(U)
        @inbounds U[idx] = U[idx] - γ*Q[idx]
    end

    return γ
end

function image_denoise(::SteepestDescent, W;
    ρ_init::Real      = 1.0,
    maxiters::Integer = 100,
    penalty::Function = __default_schedule,
    history::histT    = nothing,
    ν::Integer        = 0,
    ftol::Real        = 1e-6,
    dtol::Real        = 1e-4,
    accel::accelT     = Val(:none)) where {histT, accelT}
    #
    # extract problem dimensions
    m, n = size(W)   # m pixels by n pixels

    # allocate optimization variable
    U  = copy(W)

    # allocate gradient and auxiliary variables
    Q  = zero(W)         # gradient
    dx = zeros(m-1, n)   # derivatives along rows
    dy = zeros(m, n-1)   # derivatives along cols
    z  = zeros(length(dx) + length(dy) + 1) # for projection
    index = collect(1:length(z))

    # construct type for acceleration strategy
    strategy = get_acceleration_strategy(accel, U)

    # packing
    solution   = (Q = Q, U = U)
    projection = (z = z, dx = dx, dy = dy, index = index)
    inputs     = (W = W, ρ_init = ρ_init, penalty = penalty)
    settings   = (maxiters = maxiters, history = history, ftol = ftol, dtol = dtol, strategy = strategy)

    image_denoise!(solution, projection, inputs, settings, ν, true)

    return U
end

function image_denoise!(solution, projection, inputs, settings, ν, trace)
    # image and gradient
    U = solution.U
    Q = solution.Q

    # data structures for projection
    z     = projection.z
    dx    = projection.dx
    dy    = projection.dy
    index = projection.index

    # input data
    W       = inputs.W
    ρ_init  = inputs.ρ_init
    penalty = inputs.penalty

    # settings
    maxiters = settings.maxiters
    history  = settings.history
    ftol     = settings.ftol
    dtol     = settings.dtol
    strategy = settings.strategy

    # initialize
    ρ = ρ_init

    loss, distance, gradient = imgtvd_evaluate!(Q, z, dx, dy, index, U, W, ν, ρ)
    data = package_data(loss, distance, ρ, gradient, zero(loss))

    if trace
        update_history!(history, data, 0)
    end

    loss_old = loss
    loss_new = Inf
    dist_old = distance
    dist_new = Inf
    iteration = 1

    while not_converged(loss_old, loss_new, dist_old, dist_new, ftol, dtol) && iteration ≤ maxiters
        # iterate the algorithm map
        stepsize = imgtvd_steepest_descent!(U, Q, ρ, gradient)

        # penalty schedule + acceleration
        ρ_new = penalty(ρ, iteration)       # check for updates to the penalty coefficient
        ρ != ρ_new && restart!(strategy, U) # check for restart due to changing objective
        apply_momentum!(U, strategy)        # apply acceleration strategy
        ρ = ρ_new                           # update penalty

        # convergence history
        loss, distance, gradient = imgtvd_evaluate!(Q, z, dx, dy, index, U, W, ν, ρ)
        data = package_data(loss, distance, ρ, gradient, stepsize)

        if trace # only record outside clustering path algorithm
            update_history!(history, data, iteration)
        end

        loss_old = loss_new
        loss_new = loss
        dist_old = dist_new
        dist_new = distance
        iteration += 1
    end

    if !trace
        update_history!(history, data, iteration-1)
    end

    return U
end
