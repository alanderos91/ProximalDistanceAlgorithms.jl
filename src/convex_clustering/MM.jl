function cvxclst_iter(::MM, optvars, derivs, operators, buffers, ρ)
    u = optvars.u
    D = operators.D
    x = operators.x

    cg_iterator = buffers.cg_iterator
    b = buffers.b
    z = buffers.z
    Pz = buffers.Pz

    # set up RHS of Ax = b := W*y + ρ*D'P(D*x)
    b .= x
    mul!(b, D', Pz, ρ, 1.0)

    # solve the linear system
    __do_linear_solve!(cg_iterator, b)

    return 1.0
end


function convex_clustering(algorithm::MM, W, X;
    K::Integer = 0,
    o::Base.Ordering = Base.Order.Forward,
    psort::Function = partial_quicksort, kwargs...)
    #
    # extract problem dimensions
    d, n = size(X)      # d features by n samples
    m = binomial(n, 2)  # number of constraints
    M = d*m             # number of rows
    N = d*n             # number of columns

    # allocate optimization variable
    U = zero(X)
    u = vec(U)
    optvars = (u = u,)

    # allocate derivatives
    ∇f = zeros(N)   # loss
    ∇d = zeros(N)   # distance
    ∇h = zeros(N)   # objective
    ∇²f = I         # Hessian for loss
    derivs = (∇f = ∇f, ∇²f = ∇²f, ∇d = ∇d, ∇h = ∇h)

    # generate operators
    D = CvxClusterFM(d, n)
    x = vec(X)
    H = ProxDistHessian(N, 1.0, ∇²f, D'D)
    operators = (D = D, x = x, o = o, K = K, H = H, compute_projection = psort)

    # allocate any additional arrays for mat-vec multiplication
    z = zeros(M)
    Pz = zeros(M)
    ds = zeros(m)
    b = similar(u)

    # initialize conjugate gradient solver
    b1 = similar(u)
    b2 = similar(u)
    b3 = similar(u)
    cg_iterator = CGIterable(H, u, b1, b2, b3, 1e-8, 0.0, 1.0, size(H, 2), 0)

    buffers = (z = z, U = U, Pz = Pz, ds = ds, b = b, cg_iterator = cg_iterator)

    optimize!(algorithm, cvxclst_eval, cvxclst_iter, optvars, derivs, operators, buffers; kwargs...)

    return U
end
