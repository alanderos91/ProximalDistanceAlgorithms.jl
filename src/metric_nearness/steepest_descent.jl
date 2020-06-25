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
    # @. x = x - γ*∇h
    axpy!(-γ, ∇h, x)

    return γ
end

function metric_projection(algorithm::SteepestDescent, W, A; kwargs...)
    #
    # extract problem dimensions
    n = size(A, 1)      # number of nodes
    m1 = binomial(n, 2) # number of unique non-negativity constraints
    m2 = m1*(n-2)       # number of unique triangle edges
    N = m1              # total number of optimization variables
    M = m1 + m2         # total number of constraints

    inds = sizehint!(Int[], binomial(n,2))
    mapping = LinearIndices((1:n, 1:n))
    for j in 1:n, i in j+1:n
        push!(inds, mapping[i,j])
    end

    # allocate optimization variable
    X = copy(A)
    x = trivec_view(X, inds)
    optvars = (x = x,)

    # allocate derivatives
    ∇f = trivec_view(zero(X), inds)    # loss
    ∇d = trivec_view(zero(X), inds)    # distance
    ∇h = trivec_view(zero(X), inds)    # objective
    derivs = (∇f = ∇f, ∇d = ∇d, ∇h = ∇h)

    # generate operators
    D = MetricFM(n, M, N)   # fusion matrix
    P(x) = max.(x, 0)       # projection onto non-negative orthant
    a = trivec_view(A, inds)
    operators = (D = D, P = P, a = a)

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
