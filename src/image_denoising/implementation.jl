@doc raw"""
    denoise_image(algorithm::AlgorithmOption, image;)

Remove noise from the input `image` by minimizing its total variation.

A sparsity parameter `nu` enforces derivatives `image[i+1,j] - image[i,j]` and
`image[i,j+1] - image[i,j]` to be zero. The choice `nu = 0` coerces zero
variation whereas `nu = (n-1)*p + n*(p-1)` preserves the original image.
Setting `rev=false` reverses this relationship.

The function [`denoise_image_path`](@ref) provides a solution path.

See also: [`MM`](@ref), [`StepestDescent`](@ref), [`ADMM`](@ref), [`MMSubSpace`](@ref), [`initialize_history`](@ref)

# Keyword Arguments

- `nu::Integer=0`: A sparsity parameter that controls clusterings.
- `rev::Bool=true`: A flag that changes the interpretation of `nu` from constraint violations (`rev=true`) to constraints satisfied (`rev=false`).
This indirectly affects the performance of the algorithm and should only be used when crossing the threshold `nu = [(n-1)*p + n*(p-1)] ÷ 2`.
- `rho::Real=1.0`: An initial value for the penalty coefficient. This should match with the choice of annealing schedule, `penalty`.
- `mu::Real=1.0`: An initial value for the step size in `ADMM()`.
- `ls=Val(:LSQR)`: Choice of linear solver for `MM`, `ADMM`, and `MMSubSpace` methods. Choose one of `Val(:LSQR)` or `Val(:CG)` for LSQR or conjugate gradients, respectively.
- `maxiters::Integer=100`: The maximum number of iterations.
- `penalty::Function=__default_schedule__`: A two-argument function `penalty(rho, iter)` that computes the penalty coefficient at iteration `iter+1`. The default setting does nothing.
- `history=nothing`: An object that logs convergence history.
- `rtol::Real=1e-6`: A convergence parameter measuring the relative change in the loss model, $\frac{1}{2} \|(x-y)\|^{2}$.
- `atol::Real=1e-4`: A convergence parameter measuring the magnitude of the squared distance penalty $\frac{\rho}{2} \mathrm{dist}(Dx,C)^{2}$.
- `accel=Val(:none)`: Choice of an acceleration algorithm. Options are `Val(:none)` and `Val(:nesterov)`.
"""
function denoise_image(algorithm::AlgorithmOption, image;
    nu::Integer=0,
    tv::Real=1.0,
    rev::Bool=true,
    rho::Real=1.0,
    mu::Real=1.0, ls=Val(:LSQR), proj=Val(:l0), kwargs...)
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
    cache = zeros(M-1)

    if proj isa Val{:l0}
        P = L0Projection(nu, cache)
    elseif proj isa Val{:l1}
        P = L1Projection(tv, cache)
    else
        error("unsupported projection choice, $(proj)")
    end

    operators = (D = D, P = P, a = a)

    # allocate buffers for mat-vec multiplication, projections, and so on
    z = similar(Vector{eltype(x)}, M)
    Pz = similar(z)
    v = similar(z)

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
                b = similar(typeof(x), size(A, 1))
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
        y_prev = similar(y)
        r = similar(y)
        s = similar(x)
        tmpx = similar(x)
        buffers = (z = z, Pz = Pz, v = v, b = b, y_prev = y_prev, r = r, s = s, tmpx = tmpx)
    elseif algorithm isa MMSubSpace
        tmpGx1 = zeros(N)
        tmpGx2 = zeros(N)
        tmpx = similar(x)
        buffers = (z = z, Pz = Pz, v = v, b = b, β = β, tmpx = tmpx, tmpGx1 = tmpGx1, tmpGx2 = tmpGx2)
    else
        tmpx = similar(x)
        buffers = (z = z, Pz = Pz, v = v, b = b, tmpx = tmpx)
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

@doc raw"""
    image_denoise_path(algorithm::AlgorithmOption, image; kwargs...)

Remove noise from the input `image` by minimizing its total variation.

This function returns images obtained with 5% sparsity up to 95% sparsity, in
increments of 10%. Results are stored in a  `NamedTuple` with fields `img` and
`ν_path`.

See also: [`MM`](@ref), [`StepestDescent`](@ref), [`ADMM`](@ref), [`MMSubSpace`](@ref), [`initialize_history`](@ref)

# Keyword Arguments

- `rho::Real=1.0`: An initial value for the penalty coefficient. This should match with the choice of annealing schedule, `penalty`.
- `mu::Real=1.0`: An initial value for the step size in `ADMM()`.
- `ls=Val(:LSQR)`: Choice of linear solver for `MM`, `ADMM`, and `MMSubSpace` methods. Choose one of `Val(:LSQR)` or `Val(:CG)` for LSQR or conjugate gradients, respectively.
- `maxiters::Integer=100`: The maximum number of iterations.
- `penalty::Function=__default_schedule__`: A two-argument function `penalty(rho, iter)` that computes the penalty coefficient at iteration `iter+1`. The default setting does nothing.
- `history=nothing`: An object that logs convergence history.
- `rtol::Real=1e-6`: A convergence parameter measuring the relative change in the loss model, $\frac{1}{2} \|(x-y)\|^{2}$.
- `atol::Real=1e-4`: A convergence parameter measuring the magnitude of the squared distance penalty $\frac{\rho}{2} \mathrm{dist}(Dx,C)^{2}$.
- `accel=Val(:none)`: Choice of an acceleration algorithm. Options are `Val(:none)` and `Val(:nesterov)`.
"""
function denoise_image_path(algorithm::AlgorithmOption, image;
    stepsize::Real=0.1,
    start::Real=Inf,
    rho::Real=1.0,
    mu::Real=1.0,
    history::histT=nothing,
    ls::LS=Val(:LSQR), proj=Val(:l0), kwargs...) where {histT, LS}
    #
    # extract problem information
    # n, p = size(image)      # n pixels × p pixels
    n = size(image, 1)
    p = size(image, 2)
    m1 = (n-1)*p            # number of column derivatives
    m2 = n*(p-1)            # number of row derivatives
    M = m1 + m2 + 1         # add extra row for PSD matrix
    N = n*p                 # number of variables

    # check that stepsize and start are reasonable
    if !(0 < stepsize < 1)
        error("argument stepsize must be between 0 and 1! (stepsize = $(stepsize))")
    end

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
                b = similar(typeof(x), size(A, 1))
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
        y_prev = similar(y)
        r = similar(y)
        s = similar(x)
        tmpx = similar(x)
        buffers = (z = z, Pz = Pz, v = v, b = b, y_prev = y_prev, r = r, s = s, tmpx = tmpx)
    elseif algorithm isa MMSubSpace
        tmpGx1 = zeros(N)
        tmpGx2 = zeros(N)
        tmpx = similar(x)
        buffers = (z = z, Pz = Pz, v = v, b = b, β = β, tmpx = tmpx, tmpGx1 = tmpGx1, tmpGx2 = tmpGx2)
    else
        tmpx = similar(x)
        buffers = (z = z, Pz = Pz, v = v, b = b, tmpx = tmpx)
    end

    # create views, if needed
    views = nothing

    # pack everything into ProxDistProblem container
    objective = imgtvd_objective
    algmap = imgtvd_iter

    # allocate output
    X_path = typeof(X)[]
    s_path = Float64[]

    # initialize solution path heuristic
    mul!(z, D, x)
    nconstraint = 0
    for j in 1:M-1
        # derivatives within 10^-3 are set to 0
        nconstraint += (log10(abs(z[j])) ≤ -3)
    end

    rmax = 1.0
    rstep = stepsize
    if 0 < start ≤ rmax
        r = start
    else
        r = 1 - nconstraint / (M-1)
    end

    prog = ProgressThresh(zero(r), "Searching solution path...")
    while r ≥ 0
        if proj isa Val{:l0}
            nu = round(Int, r*(M-1))
            P = L0Projection(nu, cache)
        elseif proj isa Val{:l1}
            mul!(z, D, x)
            tv = r * norm(z, 1)
            P = L1Projection(tv, cache)
        else
            error("unsupported projection choice, $(proj)")
        end

        operators = (D = D, P = P, a = a)
        prob = ProxDistProblem(variables, derivatives, operators, buffers, views, linsolver)

        _, iter, _ = optimize!(algorithm, objective, algmap, prob, rho, mu; kwargs...)

        # update history
        if !(history === nothing)
            f_loss, h_dist, h_ngrad = objective(algorithm, prob, 1.0)
            data = package_data(f_loss, h_dist, h_ngrad, stepsize, 1.0)
            update_history!(history, data, iter-1)
        end

        # record current solution
        push!(X_path, copy(X))      # record solution
        push!(s_path, 100 * (1-r))  # record % sparsity

        # count satisfied constraints
        nconstraint = 0
        for j in 1:M-1
            # derivatives within 10^-3 are set to 0
            nconstraint += (log10(abs(z[j])) ≤ -3)
        end

        # decrease r with a heuristic that guarantees a decrease
        rnew = 1 - nconstraint / (M-1)
        if rnew < r - rstep
            r = rnew
        else
            r = r - rstep
        end
        ProgressMeter.update!(prog, r)
    end

     solution_path = (img = X_path, sparsity = s_path)

    return solution_path
end

#########################
#       objective       #
#########################

function imgtvd_objective(::AlgorithmOption, prob, ρ)
    @unpack x = prob.variables
    @unpack ∇f, ∇q, ∇h = prob.derivatives
    @unpack D, P, a = prob.operators
    @unpack z, Pz, v = prob.buffers

    # evaulate gradient of loss
    @. ∇f = x - a

    # evaluate gradient of penalty
    mul!(z, D, x)

    # finish evaluating penalty gradient
    M = length(z)
    zview = @view z[1:M-1]
    Pview = @view Pz[1:M-1]
    P(Pview, zview) # project z onto ball, storing it in Pz
    @. v = z - Pz   # compute vector pointing to constraint set

    # handle component coming from extra row in fusion matrix
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
    @unpack D, P, a = prob.operators
    @unpack z, Pz, v, b, tmpx = prob.buffers
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
        copyto!(b, 1, a, 1, n)
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

    # update z = D*x variable
    mul!(z, D, x)

    # project z onto ball, storing it in Pz
    @. v = z + λ
    M = length(v)
    vview = @view v[1:M-1]
    Pview = @view Pz[1:M-1]
    P(Pview, vview)

    # handle component coming from extra row in fusion matrix
    Pz[end] = v[end]

    # y block update
    α = (ρ / μ)
    @inbounds @simd for j in eachindex(v)
        # Pz = P(z+λ)
        # v = z + λ
        y[j] = α/(1+α) * Pz[j] + 1/(1+α) * v[j]
    end
    # y[end] = z[end] + λ[end] # corresponds to extra constraint

    # λ block update
    @inbounds @simd for j in eachindex(λ)
        λ[j] = λ[j] + z[j] - y[j]
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
        @inbounds for j in 1:n
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
