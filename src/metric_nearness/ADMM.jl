"""
Update `X` by taking a step in Newton's method.
"""
function metric_admm_update_x!(optvars, linsys, ρ, μ)
    X = optvars.X
    x = optvars.x
    y = optvars.y
    λ = optvars.λ
    D = optvars.D # assuming weights all equal to 1
    T = optvars.T

    cg_iterator = linsys.cg_iterator
    b = linsys.b

    n = size(X, 1)

    # set up RHS of Ax = b
    b .= transpose(T) * (T*x - y + λ)
    t = 1
    for j in 1:n, i in j+1:n
        b[t] = b[t] + (X[i,j] - D[i,j]) / μ
        t += 1
    end

    # solve the linear system
    __trivec_copy!(x, X)
    __do_linear_solve!(cg_iterator, b)

    # apply the update:
    # x_new = x_old - Newton direction
    for j in 1:n, i in j+1:n
        k = trivec_index(n, i, j)
        X[i,j] = X[i,j] - x[k]
    end

    return nothing
end

function metric_admm_update_y!(optvars, linsys, ρ, μ)
    y = optvars.y
    α = (ρ / μ) / (1 + ρ / μ)

    for k in eachindex(y)
        y[k] = α * min(0, y[k]) + (1-α) * y[k]
    end

    return nothing
end

function metric_admm_update_λ!(optvars, linsys, ρ, μ)
    x = optvars.x
    y = optvars.y
    λ = optvars.λ
    T = optvars.T

    λ .= λ + μ*(T*x - y)

    return nothing
end

function metric_projection(::ADMM, W, D;
    ρ_init::Real      = 1.0,
    maxiters::Integer = 100,
    penalty::Function = __default_schedule,
    history::FuncLike = __default_logger,
    accel::accelT     = Val(:none)) where {FuncLike, accelT}
    #
    # extract problem dimensions
    n = size(D, 1)
    m = binomial(n, 2)

    # assume symmetry in D and zeros in diagonal
    D_tri = LowerTriangular(D)
    W_tri = LowerTriangular(W)

    # allocate optimization variable
    X = copy(D_tri)
    x = zeros(eltype(X), m)
    y = zeros(eltype(X), m*(n-2))
    λ = zeros(eltype(X), m*(n-2))

    # initialize penalty coefficient + tuning parameter
    ρ = ρ_init
    μ = 1.0

    # allocate matrix for conjugate gradient
    T = metric_fusion_matrix(n)
    A = T'T
    for i in 1:n
        A[i,i] = 1/μ + A[i,i]
    end
    __trivec_copy!(x, X)
    mul!(y, T, x)

    # intermediates for solving A*trivec(X) = trivec(Z)
    B = similar(X)
    b = zeros(eltype(B), m)

    # initialize conjugate gradient solver
    cg_iterator = CGIterable(
        A, x,
        similar(b), similar(b), similar(b),
        1e-8, 0.0, 1.0, size(A, 2), 0
    )

    # pack variables into named tuple
    optvars  = (D = D_tri, W = W_tri, X = X, x = x, y = y, λ = λ, T = T)
    linsys = (A = A, B = B, b = b, cg_iterator = cg_iterator)

    # construct acceleration strategy
    # strategy = get_acceleration_strategy(accel, X)

    for iteration in 1:maxiters
        # iterate the algorithm map
        metric_admm_update_x!(optvars, linsys, ρ, μ)
        metric_admm_update_y!(optvars, linsys, ρ, μ)
        metric_admm_update_λ!(optvars, linsys, ρ, μ)

        # check for updates to the penalty coefficient
        # ρ_new = penalty(ρ, iteration)
        μ_new = 1 / (iteration + 1)
        for i in 1:n
            diff = 1 / μ_new - 1 / μ
            A[i,i] = A[i,i] + diff
        end
        μ = μ_new
        # apply acceleration strategy
        # ρ != ρ_new && restart!(strategy, X)
        # apply_momentum!(X, strategy)

        # check for updates to the convergence history
        # history(data, iteration)

        # update penalty
        # ρ = ρ_new
    end

    return optvars
end
