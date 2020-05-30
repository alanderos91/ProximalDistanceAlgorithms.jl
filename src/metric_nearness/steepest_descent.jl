function metric_eval(::SteepestDescent, optvars, derivs, operators, buffers, ρ)
    x = optvars.x
    ∇f = derivs.∇f
    ∇d = derivs.∇d
    ∇h = derivs.∇h
    D = operators.D
    P = operators.P
    y = operators.y
    z = buffers.z

    mul!(z, D, x)
    @. z = z - P(z)
    @. ∇f = x - y
    mul!(∇d, D', z)
    @. ∇h = ∇f + ρ * ∇d

    loss = SqEuclidean()(x, y) / 2  # 1/2 * ||W^1/2 * (x-y)||^2
    penalty = dot(z, z)             # D*x - P(D*x)
    normgrad = dot(∇h, ∇h)          # ||∇h(x)||^2

    return loss, penalty, normgrad
end

function metric_iter(::SteepestDescent, optvars, derivs, operators, buffers, ρ)
    x = optvars.x
    ∇h = derivs.∇h
    D = operators.D
    z = buffers.z

    # evaluate stepsize
    mul!(z, D, ∇h)
    a = dot(∇h, ∇h)     # ||∇h(x)||^2
    b = dot(z, z)       # ||D*∇h(x)||^2
    c = a               # ||W^1/2 * ∇h(x)||^2
    γ = a / (c + ρ*(a + b) + eps())

    # move in the direction of steepest descent
    @. x = x - γ*∇h

    return γ
end

function metric_projection(algorithm::SteepestDescent, W, Y; kwargs...)
    #
    # extract problem dimensions
    n = size(Y, 1)      # number of nodes
    m1 = binomial(n, 2) # number of unique non-negativity constraints
    m2 = m1*(n-2)       # number of unique triangle edges
    N = m1              # total number of optimization variables
    M = m1 + m2         # total number of constraints

    # allocate optimization variable
    X = copy(Y)
    x = trivec_view(X)
    optvars = (x = x,)

    # allocate derivatives
    ∇f = trivec_view(zero(X))    # loss
    ∇d = trivec_view(zero(X))    # distance
    ∇h = trivec_view(zero(X))    # objective
    derivs = (∇f = ∇f, ∇d = ∇d, ∇h = ∇h)

    # generate operators
    D = MetricFM(n, M, N)   # fusion matrix
    P(x) = max.(x, 0)       # projection onto non-negative orthant
    y = trivec_view(Y)
    operators = (D = D, P = P, y = y)

    # allocate any additional arrays for mat-vec multiplication
    z = zeros(M)
    buffers = (z = z,)

    optimize!(algorithm, metric_eval, metric_iter, optvars, derivs, operators, buffers; kwargs...)

    # symmetrize solution
    for j in 1:n, i in j+1:n
        X[j,i] = X[i,j]
    end

    return X
end
