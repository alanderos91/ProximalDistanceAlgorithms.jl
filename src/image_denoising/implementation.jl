"""
```
image_denoise(algorithm::AlgorithmOption, image;)
```
"""
function denoise_image(algorithm::AlgorithmOption, image;
    nu::Integer=0,
    rev::Bool=true,
    rho::Real=1.0,
    mu::Real=1.0, ls=Val(:LSQR), kwargs...)
    #
    # extract problem information
    # n, p = size(image)      # n pixels × p pixels
    n = size(image, 1)
    p = size(image, 2)
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

    if algorithm isa MMSubSpace
        K = subspace_size(algorithm)
        G = zeros(N, K)
        derivatives = (∇f = ∇f, ∇²f = ∇²f, ∇q = ∇q, ∇h = ∇h, G = G)
    else
        derivatives = (∇f = ∇f, ∇²f = ∇²f, ∇q = ∇q, ∇h = ∇h)
    end

    # generate operators
    D = ImgTvdFM(n, p)
    a = copy(x)
    f = SparseProjectionClosure(rev, nu)

    operators = (D = D, compute_proj = f, a = a)

    # allocate buffers for mat-vec multiplication, projections, and so on
    z = similar(Vector{eltype(x)}, M)
    Pz = similar(z)
    v = similar(z)
    cache = zeros(M-1)

    # select linear solver, if needed
    if needs_linsolver(algorithm)
        if algorithm isa MMSubSpace
            K = subspace_size(algorithm)
            β = zeros(K)

            if ls isa Val{:LSQR}
                A₁ = LinearMap(I, N)
                A₂ = D
                A = MMSOp1(A₁, A₂, G, x, x, 1.0)
                b = similar(typeof(x), size(A, 1))
                linsolver = LSQRWrapper(A, β, b)
            else
                b = similar(typeof(x), K)
                linsolver = CGWrapper(G, β, b)
            end
        else
            if ls isa Val{:LSQR}
                A₁ = LinearMap(I, N)
                A₂ = D
                A = QuadLHS(A₁, A₂, x, 1.0)
                b = similar(typeof(x), size(A₂, 1))
                linsolver = LSQRWrapper(A, x, b)
            else
                b = similar(x)
                linsolver = CGWrapper(D, x, b)
            end
        end
    else
        b = nothing
        linsolver = nothing
    end

    if algorithm isa ADMM
        mul!(y, D, x)
        r = similar(y)
        s = similar(y)
        tmpx = similar(x)
        buffers = (z = z, Pz = Pz, v = v, b = b, cache = cache, r = r, s = s, tmpx = tmpx)
    elseif algorithm isa MMSubSpace
        tmpGx1 = zeros(N)
        tmpGx2 = zeros(N)
        tmpx = similar(x)
        buffers = (z = z, Pz = Pz, v = v, b = b, cache = cache, β = β, tmpx = tmpx, tmpGx1 = tmpGx1, tmpGx2 = tmpGx2)
    else
        tmpx = similar(x)
        buffers = (z = z, Pz = Pz, v = v, b = b, cache = cache, tmpx = tmpx)
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

"""
```
image_denoise_path(algorithm::AlgorithmOption, image; kwargs...)
```
"""
function denoise_image_path(algorithm::AlgorithmOption, image;
    rho::Real=1.0,
    mu::Real=1.0,
    ls::LS=Val(:LSQR), kwargs...) where LS
    #
    # extract problem information
    # n, p = size(image)      # n pixels × p pixels
    n = size(image, 1)
    p = size(image, 2)
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

    if algorithm isa MMSubSpace
        K = subspace_size(algorithm)
        G = zeros(N, K)
        derivatives = (∇f = ∇f, ∇²f = ∇²f, ∇q = ∇q, ∇h = ∇h, G = G)
    else
        derivatives = (∇f = ∇f, ∇²f = ∇²f, ∇q = ∇q, ∇h = ∇h)
    end

    # generate operators
    D = ImgTvdFM(n, p)
    a = copy(x)

    # allocate buffers for mat-vec multiplication, projections, and so on
    z = similar(Vector{eltype(x)}, M)
    Pz = similar(z)
    v = similar(z)
    cache = zeros(M-1)

    # select linear solver, if needed
    if needs_linsolver(algorithm)
        if algorithm isa MMSubSpace
            K = subspace_size(algorithm)
            β = zeros(K)

            if ls isa Val{:LSQR}
                A₁ = LinearMap(I, N)
                A₂ = D
                A = MMSOp1(A₁, A₂, G, x, x, 1.0)
                b = similar(typeof(x), size(A, 1))
                linsolver = LSQRWrapper(A, β, b)
            else
                b = similar(typeof(x), K)
                linsolver = CGWrapper(G, β, b)
            end
        else
            if ls isa Val{:LSQR}
                A₁ = LinearMap(I, N)
                A₂ = D
                A = QuadLHS(A₁, A₂, x, 1.0)
                b = similar(typeof(x), size(A₂, 1))
                linsolver = LSQRWrapper(A, x, b)
            else
                b = similar(x)
                linsolver = CGWrapper(D, x, b)
            end
        end
    else
        b = nothing
        linsolver = nothing
    end

    if algorithm isa ADMM
        mul!(y, D, x)
        r = similar(y)
        s = similar(y)
        tmpx = similar(x)
        buffers = (z = z, Pz = Pz, v = v, b = b, cache = cache, r = r, s = s, tmpx = tmpx)
    elseif algorithm isa MMSubSpace
        tmpGx1 = zeros(N)
        tmpGx2 = zeros(N)
        tmpx = similar(x)
        buffers = (z = z, Pz = Pz, v = v, b = b, cache = cache, β = β, tmpx = tmpx, tmpGx1 = tmpGx1, tmpGx2 = tmpGx2)
    else
        tmpx = similar(x)
        buffers = (z = z, Pz = Pz, v = v, b = b, cache = cache, tmpx = tmpx)
    end

    # create views, if needed
    views = nothing

    # pack everything into ProxDistProblem container
    objective = imgtvd_objective
    algmap = imgtvd_iter

    # allocate output
    X_path = typeof(X)[]
    ν_path = Int[]

    # initialize solution path heuristic
    νmax = M-1
    @showprogress 1 "Searching solution path..." for s in range(0.05, step=0.1, stop = 0.95)
        # this is an unavoidable branch made worse by parameterization of
        # projection operator
        if round(Int, s*νmax) > (νmax >> 1)
            ν = round(Int, (1-s)*νmax)
            f = SparseProjectionClosure(true, ν)
        else
            ν = round(Int, s*νmax)
            f = SparseProjectionClosure(false, ν)
        end

        operators = (D = D, compute_proj = f, a = a)
        prob = ProxDistProblem(variables, derivatives, operators, buffers, views, linsolver)

        optimize!(algorithm, objective, algmap, prob, rho, mu; kwargs...)

        # record current solution
        push!(X_path, copy(X))
        push!(ν_path, round(Int, s*νmax))
    end

     solution_path = (img = X_path, ν = ν_path)

    return solution_path
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
    @unpack x = prob.variables
    @unpack ∇²f = prob.derivatives
    @unpack D, a = prob.operators
    @unpack b, Pz, tmpx = prob.buffers
    linsolver = prob.linsolver

    if linsolver isa LSQRWrapper
        # build LHS of A*x = b
        # forms a BlockMap so non-allocating
        # however, A*x and A'b have small allocations due to views?
        A₁ = LinearMap(I, size(D, 2))
        A₂ = D
        A = QuadLHS(A₁, A₂, tmpx, √ρ)

        # build RHS of A*x = b; b = [a; √ρ * P(D*x)]
        n = length(a)
        copyto!(b, 1, a, 1, n)
        for k in eachindex(Pz)
            b[n+k] = √ρ * Pz[k]
        end
    else
        # assemble LHS
        A = ProxDistHessian(∇²f, D'D, tmpx, ρ)

        # build RHS of A*x = b; b = a + ρ*D'P(D*x)
        mul!(b, D', Pz)
        axpby!(1, a, ρ, b)
    end

    # solve the linear system
    linsolve!(linsolver, x, A, b)

    return 1.0
end

function imgtvd_iter(::ADMM, prob, ρ, μ)
    @unpack x, y, λ = prob.variables
    @unpack ∇²f = prob.derivatives
    @unpack D, compute_proj, a = prob.operators
    @unpack z, v, b, cache, tmpx = prob.buffers
    linsolver = prob.linsolver

    # x block update
    @. v = y - λ
    if linsolver isa LSQRWrapper
        # build LHS of A*x = b
        # forms a BlockMap so non-allocating
        # however, A*x and A'b have small allocations due to views?
        A₁ = LinearMap(I, size(D, 2))
        A₂ = D
        A = QuadLHS(A₁, A₂, tmpx, √μ) # A = [I; √μ*D]

        # build RHS of A*x = b; b = [a; √μ * (y-λ)]
        n = length(a)
        copyto!(b, 1, a, 1, length(a))
        @inbounds for k in eachindex(v)
            b[n+k] = √μ * v[k]
        end
    else
        # assemble LHS
        A = ProxDistHessian(∇²f, D'D, tmpx, μ)

        # build RHS of A*x = b; b = a + μ*D'(y-λ)
        mul!(b, D', v)
        axpby!(1, a, μ, b)
    end

    # solve the linear system
    linsolve!(linsolver, x, A, b)

    # compute pivot for projection operator
    mul!(z, D, x)
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
    y[end] = z[end] + λ[end] # corresponds to extra constraint

    # λ block update
    @inbounds @simd for j in eachindex(λ)
        λ[j] = λ[j] + μ * (z[j] - y[j])
    end

    return μ
end

function imgtvd_iter(::MMSubSpace, prob, ρ, μ)
    @unpack x = prob.variables
    @unpack ∇²f, ∇h, ∇f, G = prob.derivatives
    @unpack D = prob.operators
    @unpack β, b, v, tmpx, tmpGx1, tmpGx2 = prob.buffers
    linsolver = prob.linsolver

    # solve linear system Gt*At*A*G * β = Gt*At*b for stepsize
    if linsolver isa LSQRWrapper
        # build LHS, A = [A₁, A₂] * G
        A₁ = LinearMap(I, size(D, 2))
        A₂ = D
        A = MMSOp1(A₁, A₂, G, tmpGx1, tmpGx2, √ρ)

        # build RHS, b = -∇h
        n = size(A₁, 1)
        @inbounds for j in 1:size(A₁, 1)
            b[j] = -∇f[j]   # A₁*x - a
        end
        @inbounds for j in eachindex(v)
            b[n+j] = -√ρ*v[j]
        end
    else
        # build LHS, A = G'*H*G = G'*(∇²f + ρ*D'D)*G
        H = ProxDistHessian(∇²f, D'D, tmpx, ρ)
        A = MMSOp2(H, G, tmpGx1, tmpGx2)

        # build RHS, b = -G'*∇h
        mul!(b, G', ∇h)
        @. b = -b
    end

    # solve the linear system
    linsolve!(linsolver, β, A, b)

    # apply the update, x = x + G*β
    mul!(x, G, β, true, true)

    return norm(β)
end
