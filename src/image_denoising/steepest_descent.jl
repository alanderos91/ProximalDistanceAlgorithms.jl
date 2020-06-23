function imgtvd_iter(::SteepestDescent, optvars, derivs, operators, buffers, ρ)
    u = optvars.u
    ∇h = derivs.∇h
    D = operators.D
    z = buffers.z

    # evaluate stepsize
    mul!(z, D, ∇h)
    a = dot(∇h, ∇h)
    b = dot(z, z)
    γ = a / (a + ρ*b + eps())

    # move in the direction of steepest descent
    @. u = u - γ*∇h

    return γ
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
    ∇f = zeros(N)
    ∇d = zeros(N)
    ∇h = zeros(N)
    derivs = (∇f = ∇f, ∇d = ∇d, ∇h = ∇h)

    # generate operators
    D = ImgTvdFM(n, p)
    w = vec(W)
    operators = (D = D, w = w, o = o, K = K, compute_projection = psort)

    # allocate any additional arrays for mat-vec multiplication
    z = zeros(M)
    Pz = zeros(M)
    ds = zeros(M)
    buffers = (z = z, Pz = Pz, ds = ds)

    optimize!(algorithm, imgtvd_eval, imgtvd_iter, optvars, derivs, operators, buffers; kwargs...)

    return U
end
