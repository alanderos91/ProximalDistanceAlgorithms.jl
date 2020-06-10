function metric_admm_update_x!(optvars, derivs, operators, buffers, ρ)
    x = optvars.x
    y = optvars.y
    λ = optvars.λ
    ∇f = derivs.∇f
    D = operators.D
    μ = operators.H.ρ

    cg_iterator = buffers.cg_iterator
    b = buffers.b
    η = buffers.η
    z = buffers.z

    # set up RHS of Ax = b := ∇f + μ*D' * (D*x - y + λ)
    mul!(z, D, x)
    b .= ∇f
    mul!(b, D', z, μ, 1.0)
    mul!(b, D', y, -μ, 1.0)
    mul!(b, D', λ, μ, 1.0)

    # solve the linear system
    __do_linear_solve!(cg_iterator, b)

    # apply the update:
    # x_new = x_old - Newton direction
    axpy!(-1.0, η, x)

    return nothing
end

function metric_admm_update_y!(optvars, derivs, operators, buffers, ρ)
    x = optvars.x
    y = optvars.y
    λ = optvars.λ
    D = operators.D
    P = operators.P
    μ = operators.H.ρ
    z = buffers.z
    Pz = buffers.Pz

    # evaluate z = D*x + λ and P(z)
    mul!(z, D, x)
    axpy!(1, λ, z)
    @. Pz = P(z)

    # y = α*Pz + (1-α)*z
    α = (ρ / μ) / (1 + ρ / μ)
    y .= (1-α)*z
    axpy!(α, Pz, y)

    return nothing
end

function metric_admm_update_λ!(optvars, derivs, operators, buffers, ρ)
    x = optvars.x
    y = optvars.y
    λ = optvars.λ
    D = operators.D
    μ = operators.H.ρ

    # λ =  λ + μ*(D*x - y)
    mul!(λ, D, x, μ, 1.0) # λ = λ + μ*D*x
    axpy!(-μ, y, λ)    # λ = λ - μ*y

    return nothing
end

function metric_iter(::ADMM, optvars, derivs, operators, buffers, ρ)
    metric_admm_update_x!(optvars, derivs, operators, buffers, ρ)
    metric_admm_update_y!(optvars, derivs, operators, buffers, ρ)
    metric_admm_update_λ!(optvars, derivs, operators, buffers, ρ)

    return operators.H.ρ
end

function metric_projection(algorithm::ADMM, W, A; μ::Real = 1.0, kwargs...)
    #
    # extract problem dimensions
    n = size(A, 1)      # number of nodes
    m1 = binomial(n, 2) # number of unique non-negativity constraints
    m2 = m1*(n-2)       # number of unique triangle edges
    N = m1              # total number of optimization variables
    M = m1 + m2         # total number of constraints

    # allocate optimization variable
    X = copy(A)
    x = trivec_view(X)
    y = zeros(M)
    λ = zeros(M)
    optvars = (x = x, y = y, λ = λ)

    # allocate derivatives
    ∇f = trivec_view(zero(X))    # loss
    ∇d = trivec_view(zero(X))    # distance
    ∇h = trivec_view(zero(X))    # objective
    ∇²f = I                      # Hessian for loss
    derivs = (∇f = ∇f, ∇²f = ∇²f, ∇d = ∇d, ∇h = ∇h)

    # generate operators
    D = MetricFM(n, M, N)   # fusion matrix
    P(x) = max.(x, 0)       # projection onto non-negative orthant
    H = ProxDistHessian(N, μ, ∇²f, D'D) # this needs to be set to ρ_init
    a = trivec_view(A)
    operators = (D = D, P = P, H = H, a = a)

    # allocate any additional arrays for mat-vec multiplication
    z = zeros(M)
    Pz = similar(z)
    η = trivec_view(similar(X))
    b = trivec_view(similar(X))

    # initialize conjugate gradient solver
    b1 = trivec_view(similar(X))
    b2 = trivec_view(similar(X))
    b3 = trivec_view(similar(X))
    cg_iterator = CGIterable(H, η, b1, b2, b3, 1e-8, 0.0, 1.0, size(H, 2), 0)

    buffers = (z = z, Pz = Pz, b = b, η = η, cg_iterator = cg_iterator)

    optimize!(algorithm, metric_eval, metric_iter, optvars, derivs, operators, buffers; kwargs...)

    # symmetrize solution
    for j in 1:n, i in j+1:n
        X[j,i] = X[i,j]
    end

    return X
end
