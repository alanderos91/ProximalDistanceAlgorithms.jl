function metric_iter(::MM, optvars, derivs, operators, buffers, ρ)
    x = optvars.x
    D = operators.D
    P = operators.P
    W = operators.W
    a = operators.a

    cg_iterator = buffers.cg_iterator
    b = buffers.b
    z = buffers.z
    Pz = buffers.Pz

    # set up RHS of Ax = b := W*y + ρ*D'P(D*x)
    mul!(z, D, x)
    @. Pz = P(z)

    mul!(b, W, a)
    mul!(b, D', Pz, ρ, 1.0)

    # solve the linear system
    __do_linear_solve!(cg_iterator, b)

    return 1.0
end

function metric_projection(algorithm::MM, W, A; kwargs...)
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
    ∇²f = I                      # Hessian for loss
    derivs = (∇f = ∇f, ∇²f = ∇²f, ∇d = ∇d, ∇h = ∇h)

    # generate operators
    D = MetricFM(n, M, N)   # fusion matrix
    P(x) = max.(x, 0)       # projection onto non-negative orthant
    H = ProxDistHessian(N, 1.0, ∇²f, D'D) # this needs to be set to ρ_init
    a = trivec_view(A, inds)
    operators = (D = D, P = P, H = H, W = ∇²f, a = a)

    # allocate any additional arrays for mat-vec multiplication
    z = zeros(M)
    Pz = similar(z)
    b = trivec_view(similar(X), inds)

    # initialize conjugate gradient solver
    b1 = trivec_view(similar(X), inds)
    b2 = trivec_view(similar(X), inds)
    b3 = trivec_view(similar(X), inds)
    cg_iterator = CGIterable(H, x, b1, b2, b3, 1e-8, 0.0, 1.0, size(H, 2), 0)

    buffers = (z = z, Pz = Pz, b = b, cg_iterator = cg_iterator)

    optimize!(algorithm, metric_eval, metric_iter, optvars, derivs, operators, buffers; kwargs...)

    # symmetrize solution
    for j in 1:n, i in j+1:n
        X[j,i] = X[i,j]
    end

    return X
end
