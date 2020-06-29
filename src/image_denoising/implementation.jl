"""
```
image_denoise(algorithm::AlgorithmOption, image;)
```
"""
function denoise_image(algorithm::AlgorithmOption, image;
    K::Integer=0,
    o::Base.Ordering=Base.Order.Forward,
    rho::Real=1.0,
    mu::Real=1.0, kwargs...)
    #
    # extract problem information
    n, p = size(image)      # n pixels × p pixels
    m1 = (n-1)*p            # number of column derivatives
    m2 = n*(p-1)            # number of row derivatives
    M = m1 + m2 + 1         # add extra row for PSD matrix
    N = n*p                 # number of variables

    # allocate optimization variable
    X = copy(image)
    x = vec(X)
    if algorithm isa ADMM
        y = zeros(M)
        λ = zeros(M)
        variables = (x = x, y = y, λ = λ)
    else
        variables = (x = x,)
    end

    # allocate derivatives
    ∇f = needs_gradient(algorithm) ? similar(x) : nothing
    ∇q = needs_gradient(algorithm) ? similar(x) : nothing
    ∇h = needs_gradient(algorithm) ? similar(x) : nothing
    ∇²f = needs_hessian(algorithm) ? I : nothing
    derivatives = (∇f = ∇f, ∇²f = ∇²f, ∇q = ∇q, ∇h = ∇h)

    # generate operators
    D = ImgTvdFM(n, p)
    a = copy(x)
    compute_proj = cache -> compute_sparse_projection(cache, o, K)
    if algorithm isa SteepestDescent
        H = nothing
    elseif algorithm isa MM
        H = ProxDistHessian(N, rho, ∇²f, D'D)
    else
        H = ProxDistHessian(N, mu, ∇²f, D'D)
    end
    operators = (D = D, compute_proj = compute_proj, H = H, a = a)

    # allocate buffers for mat-vec multiplication, projections, and so on
    z = similar(Vector{eltype(x)}, M)
    Pz = similar(z)
    v = similar(z)
    cache = zeros(M-1)
    b = needs_linsolver(algorithm) ? similar(x) : nothing
    buffers = (z = z, Pz = Pz, v = v, b = b, cache = cache)

    # select linear solver, if needed
    if needs_linsolver(algorithm)
        b1 = similar(x)
        b2 = similar(x)
        b3 = similar(x)
        linsolver = CGIterable(H, x, b1, b2, b3, 1e-8, 0.0, 1.0, N, 0)
    else
        linsolver = nothing
    end

    # create views, if needed
    views = nothing

    # pack everything into ProxDistProblem container
    objective = imgtvd_objective
    algmap = imgtvd_iter
    prob = ProxDistProblem(variables, derivatives, operators, buffers, views, linsolver)

    # solve the optimization problem
    optimize!(algorithm, objective, algmap, prob, rho, mu; kwargs...)

    return X
end

#########################
#       objective       #
#########################

function imgtvd_objective(::AlgorithmOption, prob, ρ)
    @unpack x = prob.variables
    @unpack ∇f, ∇q, ∇h = prob.derivatives
    @unpack D, compute_proj, a = prob.operators
    @unpack z, Pz, v, cache = prob.buffers

    # evaulate gradient of loss
    @. ∇f = x - a

    # evaluate gradient of penalty
    mul!(z, D, x)

    # compute pivot for projection operator
    copyto!(cache, 1, z, 1, length(cache))
    @. cache = abs.(cache)
    P = compute_proj(cache)

    # finish evaluating penalty gradient
    for k in eachindex(cache)
        indicator = P(abs(z[k])) == abs(z[k])
        Pz[k] = indicator*z[k]
        v[k] = z[k] - Pz[k]
    end
    Pz[end] = z[end]
    v[end] = 0
    mul!(∇q, D', v)
    @. ∇h = ∇f + ρ * ∇q

    loss = SqEuclidean()(x, a) / 2
    penalty = dot(v, v)
    normgrad = dot(∇h, ∇h)

    return loss, penalty, normgrad
end

############################
#      algorithm maps      #
############################

function imgtvd_iter(::SteepestDescent, prob, ρ, μ)
    @unpack x = prob.variables
    @unpack ∇h = prob.derivatives
    @unpack D = prob.operators
    @unpack z = prob.buffers

    # evaluate step size, γ
    mul!(z, D, ∇h)
    a = dot(∇h, ∇h)     # ||∇h(x)||^2
    b = dot(z, z)       # ||D*∇h(x)||^2
    γ = a / (a + ρ*b + eps())

    # steepest descent, x_new = x_old - γ*∇h(x_old)
    axpy!(-γ, ∇h, x)

    return γ
end

function imgtvd_iter(::MM, prob, ρ, μ)
    @unpack D, a = prob.operators
    @unpack b, Pz = prob.buffers
    linsolver = prob.linsolver

    # build RHS of Ax = b
    mul!(b, D', Pz)
    axpby!(1, a, ρ, b)

    # solve the linear system; assuming x bound to linsolver
    __do_linear_solve!(linsolver, b)

    return 1.0
end

function imgtvd_iter(::ADMM, prob, ρ, μ)
    @unpack x, y, λ = prob.variables
    @unpack D, compute_proj, a = prob.operators
    @unpack z, v, b, cache = prob.buffers
    linsolver = prob.linsolver

    # compute pivot for projection operator
    @. v = z + λ
    copyto!(cache, 1, v, 1, length(cache))
    @. cache = abs(cache)
    P = compute_proj(cache)

    # y block update
    α = (ρ / μ)
    @inbounds @simd for j in eachindex(cache)
        zpλj = z[j] + λ[j]
        indicator = P(abs(zpλj)) == abs(zpλj)
        y[j] = α/(1+α) * indicator*zpλj + 1/(1+α) * zpλj
    end
    y[end] = z[end] + λ[end]

    # x block update
    @. v = y - λ
    mul!(b, D', v)
    axpby!(1, a, μ, b)
    __do_linear_solve!(linsolver, b)

    # λ block update
    mul!(z, D, x)
    @inbounds @simd for j in eachindex(λ)
        λ[j] = λ[j] / μ + z[j] - y[j]
    end

    return μ
end
