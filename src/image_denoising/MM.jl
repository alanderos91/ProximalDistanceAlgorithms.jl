function imgtvd_iter(::MM, optvars, derivs, operators, buffers, ρ)
    u = optvars.u
    D = operators.D
    w = operators.w

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


function image_denoise(algorithm::SteepestDescent, W;
    K::Integer = 0,
    o::Base.Ordering = Base.Order.Forward,
    psort::Function = partial_quicksort, kwargs...)
    #
    # extract problem dimensions
    n, p = size(W)          # n pixels by p pixels
    m1 = (n-1)*p            # number of column derivatives
    m2 = n*(p-1)            # number of row derivatives
    M = m1 + m2 + 1         # add extra row for PSD
    N = n*p                 # number of variables

    # allocate optimization variable
    U = zero(W)
    u = vec(U)
    optvars = (u = u,)

    # allocate derivatives
    ∇f = zeros(N)   # loss
    ∇d = zeros(N)   # distance
    ∇h = zeros(N)   # objective
    ∇²f = I         # Hessian for loss
    derivs = (∇f = ∇f, ∇²f = ∇²f, d = ∇d, ∇h = ∇h)

    # generate operators
    D = ImgTvdFM(n, p)
    w = vec(W)
    H = ProxDistHessian(N, 1.0, ∇²f, D'D)
    operators = (D = D, w = w, o = o, K = K, compute_projection = psort)

    # allocate any additional arrays for mat-vec multiplication
    z = zeros(M)
    Pz = zeros(M)
    ds = zeros(M)
    b = similar(u)

    # initialize conjugate gradient solver
    b1 = similar(u)
    b2 = similar(u)
    b3 = similar(u)
    cg_iterator = CGIterable(H, u, b1, b2, b3, 1e-8, 0.0, 1.0, size(H, 2), 0)
    buffers = (z = z, Pz = Pz, ds = ds, b = b, cg_iterator = cg_iterator)

    optimize!(algorithm, imgtvd_eval, imgtvd_iter, optvars, derivs, operators, buffers; kwargs...)

    return U
end
