function connum_iter(::MM, optvars, derivs, operators, buffers, ρ)
    x = optvars.x
    D = operators.D
    y = operators.y
    Pz = buffers.Pz
    p = length(y)
    c = D.c

    # compute x = (I + ρ*D'D)^{-1} * (I; √ρ D)' * (y; P(z))
    mul!(x, D', Pz)
    axpby!(1, y, ρ, x)

    a = -2*c*ρ
    b = (1 + ρ*p*(c^2+1)) / a
    u = 1 / (a*b)
    v = sum(x) / (b + p)

    @simd for k in eachindex(x)
        @inbounds x[k] = u*(x[k] - v)
    end

    return 1.0
end

function reduce_cond(algorithm::MM, c, M; kwargs...)
    #
    # extract problem information
    y, U, Vt = extract_svd(M)
    p = length(y)

    # allocate optimization variable
    x = copy(y)
    optvars = (x = x,)

    # allocate derivatives
    ∇f = similar(x)
    ∇d = similar(x)
    ∇h = similar(x)
    ∇²f = I
    derivs = (∇f = ∇f, ∇²f = ∇²f, ∇d = ∇d, ∇h = ∇h)

    # generate operators
    D = ConNumFM(c, p)
    M, N = size(D)
    P(x) = min.(x, 0)
    H = ProxDistHessian(N, 1.0, ∇²f, D'D)
    operators = (D = D, P = P, H = H, y = y)

    # allocate any additional arrays for mat-vec multiplication
    z = zeros(M)
    Pz = zeros(M)

    cg_iterator = CGIterable(H, Float64[], Float64[], Float64[], Float64[], 1e-8, 0.0, 1.0, size(H, 2), 0) # dirty hack

    buffers = (z = z, Pz = Pz, cg_iterator = cg_iterator)

    optimize!(algorithm, connum_eval, connum_iter, optvars, derivs, operators, buffers; kwargs...)

    return U*Diagonal(x)*Vt
end
