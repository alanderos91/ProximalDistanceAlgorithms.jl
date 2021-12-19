@doc raw"""
    cvxreg_fit(algorithm::SteepestDescent, response, covariates; kwargs...)

Fit a convex function to `response` data based on the given `covariates`.

The `response` should enter as a vector and samples in `covariates` should be arranged along columns. Returns two values, `(θ, ξ)`, representing function and subgradient estimates, respectively.

The penalized objective used is

```math
h_{\rho}(x) = \frac{1}{2} \|\theta-y\|^{2} + \frac{\rho}{2} \mathrm{dist}(D_{1}\theta + D_{2}\boldsymbol{\xi},C)^{2}
```

where ``Dx = [D_{1} D_{2}] [\theta; \xi]`` encodes subgradient constraints
``\theta_{i} - \langle{\boldsymbol{\xi}, \boldsymbol{x}_{i} -
\boldsymbol{x}_{j}}\rangle \le \theta_{j}``. The constraint set ``C``
represents a compatible non-positive orthant.

See also [`MM`](@ref), [`SteepestDescent`](@ref), [`ADMM`](@ref), [`MMSubSpace`](@ref), [`initialize_history`](@ref), [`optimize!`](@ref), [`anneal!`](@ref)

# Keyword Arguments

- `ls=Val(:LSQR)`: Choice of linear solver in `MM`, `ADMM`, and `MMSubSpace`. Choose one of `Val(:LSQR)` or `Val(:CG)` for LSQR or conjugate gradients, respectively.
"""
function cvxreg_fit(algorithm::AlgorithmOption, response, covariates; ls::LS=Val(:LSQR), kwargs...) where LS
    #
    # extract problem information
    d, n = size(covariates) # features × samples
    M = n*(n-1)             # number of subradient constraints
    N = n*(d+1)             # total number of optimization variables

    # allocate optimization variables
    x = zeros(N); copyto!(x, response)
    if algorithm isa ADMM
        y = zeros(M)
        λ = zeros(M)
        variables = (x = x, y = y, λ = λ)
    else
        variables = (x = x,)
    end

    # allocate derivatives
    ∇f = needs_gradient(algorithm) ? zero(x) : nothing
    ∇q = needs_gradient(algorithm) ? similar(x) : nothing
    ∇h = needs_gradient(algorithm) ? similar(x) : nothing

    if needs_hessian(algorithm)
        ∇²f = spzeros(N, N)
        for j in 1:n
            ∇²f[j,j] = 1
        end
    else
        ∇²f = nothing
    end

    if algorithm isa MMSubSpace
        K = subspace_size(algorithm)
        G = zeros(N, K)
        derivatives = (∇f = ∇f, ∇²f = ∇²f, ∇q = ∇q, ∇h = ∇h, G = G)
    else
        derivatives = (∇f = ∇f, ∇²f = ∇²f, ∇q = ∇q, ∇h = ∇h)
    end

    # generate operators
    D = CvxRegFM(covariates)
    DtD = D'D
    # D = CvxRegFM(covariates)
    a = response
    P(x) = min.(x, 0)
    # A₁ = [LinearMap(I, n) LinearMap(spzeros(n, n*d))]
    A₁ = spzeros(n, n*(d+1))
    for j in 1:n
        A₁[j,j] = 1
    end
    operators = (D = D, DtD = DtD, P = P, A₁ = A₁, a = a)

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
                A₂ = D
                A = QuadLHS(A₁, A₂, x, 1.0)
                b = similar(typeof(x), size(A₁,1)+M) # b has two blocks
                linsolver = LSQRWrapper(A, x, b)
            else
                b = similar(x)  # b has one block
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
    θ = view(x, 1:n)
    ξ = view(x, n+1:N)
    ∇h_θ = view(∇h, 1:n)
    ∇h_ξ = view(∇h, n+1:N)
    views = (θ = θ, ξ = ξ, ∇h_θ = ∇h_θ, ∇h_ξ = ∇h_ξ)

    # pack everything into ProxDistProblem container
    objective = cvxreg_objective
    algmap = cvxreg_iter
    old_variables = deepcopy(variables)
    prob = ProxDistProblem(variables, old_variables, derivatives, operators, buffers, views, linsolver)
    prob_tuple = (objective, algmap, prob)

    # solve the optimization problem
    optimize!(algorithm, prob_tuple; kwargs...)

    return copy(θ), reshape(ξ, d, n)
end

#########################
#       objective       #
#########################

function cvxreg_objective(::AlgorithmOption, prob, ρ)
    @unpack x = prob.variables
    @unpack ∇f, ∇q, ∇h = prob.derivatives
    @unpack D, P, a = prob.operators
    @unpack z, Pz, v = prob.buffers
    @unpack θ = prob.views

    # evaulate gradient of loss
    @inbounds for j in eachindex(θ)
        ∇f[j] = θ[j] - a[j]
    end

    # evaluate gradient of penalty
    mul!(z, D, x)
    @. Pz = P(z)
    @. v = z - Pz
    mul!(∇q, D', v)
    @. ∇h = ∇f + ρ * ∇q

    loss = SqEuclidean()(θ, a)
    distance = dot(v, v)
    objective = 1//2 * loss + ρ/2 * distance
    normgrad = dot(∇h, ∇h)

    return IterationResult(loss, objective, distance, normgrad)
end

############################
#      algorithm maps      #
############################

function cvxreg_iter(::SteepestDescent, prob, ρ, μ)
    @unpack x = prob.variables
    @unpack ∇f, ∇q, ∇h = prob.derivatives
    @unpack D, P, a = prob.operators
    @unpack z, Pz, v = prob.buffers
    @unpack θ, ∇h_θ, ∇h_ξ = prob.views

    # projection + gradient
    @inbounds for j in eachindex(θ)
        ∇f[j] = θ[j] - a[j]
    end
    mul!(z, D, x)
    @. Pz = P(z)
    @. v = z - Pz
    mul!(∇q, D', v)
    @. ∇h = ∇f + ρ * ∇q

    # evaluate step size, γ
    mul!(z, D, ∇h)
    a = dot(∇h_θ, ∇h_θ) # ||∇h_θ(x)||^2
    b = dot(∇h_ξ, ∇h_ξ) # ||∇h_ξ(x)||^2
    c = dot(z, z)       # ||D*∇h(x)||^2
    γ = (a + b) / (a + ρ*c + eps())

    # steepest descent, x_new = x_old - γ*∇h(x_old)
    axpy!(-γ, ∇h, x)

    return γ
end

function cvxreg_iter(::MM, prob, ρ, μ)
    @unpack x = prob.variables
    @unpack ∇²f = prob.derivatives
    @unpack D, P, DtD, A₁, a = prob.operators   # a is bound to response
    @unpack b, z, Pz, tmpx = prob.buffers
    linsolver = prob.linsolver

    # projection
    mul!(z, D, x)
    @. Pz = P(z)

    if linsolver isa LSQRWrapper
        # build LHS of A*x = b
        # forms a BlockMap so non-allocating
        # however, A*x and A'b have small allocations due to views?
        A₂ = D
        A = QuadLHS(A₁, A₂, tmpx, √ρ)

        # build RHS of A*x = b; b = [a; √ρ * P(D*x)]
        n = length(a)
        copyto!(b, 1, a, 1, n)
        for k in eachindex(Pz)
            b[n+k] = √ρ * Pz[k]
        end
    else
        # LHS of A*x = b is already stored
        A = ProxDistHessian(∇²f, DtD, tmpx, ρ)

        # build RHS of A*x = b; b = a + ρ*D'P(D*x)
        mul!(b, D', Pz)
        @. b = ρ*b
        @inbounds for j in eachindex(a)
            b[j] = a[j] + b[j]  # can't do axpby! due to shape
        end
    end

    # solve the linear system
    linsolve!(linsolver, x, A, b)

    return 1.0
end

function cvxreg_iter(::ADMM, prob, ρ, μ)
    @unpack x, y, λ = prob.variables
    @unpack ∇²f = prob.derivatives
    @unpack D, DtD, A₁, P, a = prob.operators
    @unpack z, Pz, v, b, tmpx = prob.buffers
    linsolver = prob.linsolver

    # x block update
    @. v = y - λ
    if linsolver isa LSQRWrapper
        # build LHS of A*x = b
        # forms a BlockMap so non-allocating
        # however, A*x and A'b have small allocations due to views?
        A = QuadLHS(A₁, D, tmpx, √μ)

        # build RHS of A*x = b; b = [a; √μ * (y-λ)]
        n = length(a)
        copyto!(b, 1, a, 1, length(a))
        @inbounds for k in eachindex(v)
            b[n+k] = √μ * v[k]
        end
    else
        # LHS of A*x = b is already stored
        A = ProxDistHessian(∇²f, DtD, tmpx, μ)

        # build RHS of A*x = b; b = a + μ*D'(y-λ)
        mul!(b, D', v)
        @. b = μ*b
        @inbounds for j in eachindex(a)
            b[j] = a[j] + b[j]  # can't do axpby! due to shape
        end
    end

    # solve the linear system
    linsolve!(linsolver, x, A, b)

    # y block update
    α = (ρ / μ)
    mul!(z, D, x)
    @inbounds @simd for j in eachindex(y)
        y[j] = α/(1+α) * P(z[j] + λ[j]) + 1/(1+α) * (z[j] + λ[j])
    end

    # λ block update
    @inbounds for j in eachindex(λ)
        λ[j] = λ[j] + (z[j] - y[j])
    end

    return μ
end

function cvxreg_iter(::MMSubSpace, prob, ρ, μ)
    @unpack θ, x = prob.variables
    @unpack ∇f, ∇q, ∇h, ∇²f, G = prob.derivatives
    @unpack D, DtD, A₁ = prob.operators
    @unpack β, b, v, tmpx, tmpGx1, tmpGx2 = prob.buffers
    linsolver = prob.linsolver

    # projection + gradient
    @inbounds for j in eachindex(θ)
        ∇f[j] = θ[j] - a[j]
    end
    mul!(z, D, x)
    @. Pz = P(z)
    @. v = z - Pz
    mul!(∇q, D', v)
    @. ∇h = ∇f + ρ * ∇q

    # solve linear system Gt*At*A*G * β = Gt*At*b for stepsize
    if linsolver isa LSQRWrapper
        # build LHS, A = [A₁, A₂] * G
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
        H = ProxDistHessian(∇²f, DtD, tmpx, ρ)
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

#################################
#   simulate problem instance   #
#################################

"""
Generate a `d` by `n` matrix `X` with `d` covariates and `n` samples.
Samples are uniform over the cube `[-1,1]^d`.

Output is `(X, xdata)`, where `xdata` stores each sample as a vector.
"""
function cvxreg_simulate_covariates(rng::AbstractRNG, d, n)
    xdata = sort!([2*rand(rng, d) .- 1 for _ in 1:n])
    X = hcat(xdata...)

    return X, xdata
end

"""
Evaluate `φ: R^d --> R` at the points in `xdata` and simulate samples
`y = φ(x) + ε` where `ε ~ N(0, σ²)`.

Output is returned as `(y, φ(x))`.
"""
function cvxreg_simulate_responses(rng::AbstractRNG, φ, xdata, σ)
    y_truth = φ.(xdata)
    noise = σ*randn(rng, length(xdata))
    y = y_truth + noise

    return y, y_truth
end

"""
Generate an instance of a convex regression problem based on a convex function `φ: R^d --> R` with `n` samples.

The `σ` parameter is the standard deviation of iid perturbations applied to the true values. Optionally, pass a 
`rng` to make simulation reproducible (default `StableRNG(1234)`).

Output is returned as `(y, φ(x), X)`.
"""
function cvxreg_example(φ, d, n, σ; rng::AbstractRNG=StableRNG(1234))
    X, xdata = cvxreg_simulate_covariates(rng, d, n)
    y, y_truth = cvxreg_simulate_responses(rng, φ, xdata, σ)

    return y, y_truth, X
end

"""
Standardize the responses and covariates as in Mazumder et al. 2018.
"""
function mazumder_standardization(y, X)
    X_scaled = copy(X)

    for i in 1:size(X, 1)
        X_scaled[i,:] = (X[i,:] .- mean(X[i,:])) / norm(X[i,:])
    end
    y_scaled = y ./ norm(y)

    return y_scaled, X_scaled
end

#################################
#   hybrid SD + ADMM prototype  #
#################################

function cvxreg_fit(algorithm::SDADMM, response, covariates;
    rho::Real=1.0, mu::Real=1.0, ls::LS=Val(:LSQR), phase1=10, phase2=10, kwargs...) where LS
    #
    # extract problem information
    d, n = size(covariates) # features × samples
    M = n*(n-1)             # number of subradient constraints
    N = n*(d+1)             # total number of optimization variables

    # allocate optimization variables
    x = zeros(N); copyto!(x, response)
    y = zeros(M)
    λ = zeros(M)
    SD_variables = (x = x,)
    ADMM_variables = (x = x, y = y, λ = λ)

    # allocate derivatives
    ∇f = zero(x)
    ∇q = similar(x)
    ∇h = similar(x)
    ∇²f = spzeros(N, N)
    for j in 1:n
        ∇²f[j,j] = 1
    end

    derivatives = (∇f = ∇f, ∇²f = ∇²f, ∇q = ∇q, ∇h = ∇h)

    # generate operators
    D = instantiate_fusion_matrix(CvxRegFM(covariates))
    DtD = D'D
    # D = CvxRegFM(covariates)
    a = response
    P(x) = min.(x, 0)
    # A₁ = [LinearMap(I, n) LinearMap(spzeros(n, n*d))]
    A₁ = spzeros(n, n*(d+1))
    for j in 1:n
        A₁[j,j] = 1
    end
    operators = (D = D, DtD = DtD, P = P, A₁ = A₁, a = a)

    # allocate buffers for mat-vec multiplication, projections, and so on
    z = similar(Vector{eltype(x)}, M)
    Pz = similar(z)
    v = similar(z)

    # choose algorithm for linear solve
    if ls isa Val{:LSQR}
        A₂ = D
        A = QuadLHS(A₁, A₂, x, 1.0)
        b = similar(typeof(x), size(A₁,1)+M) # b has two blocks
        linsolver = LSQRWrapper(A, x, b)
    else
        b = similar(x)  # b has one block
        linsolver = CGWrapper(D, x, b)
    end

    # finish initializing buffers
    y_prev = similar(y)
    r = similar(y)
    s = similar(x)
    tmpx = similar(x)
    buffers = (z = z, Pz = Pz, v = v, b = b, y_prev = y_prev, r = r, s = s, tmpx = tmpx)

    # create views, if needed
    θ = view(x, 1:n)
    ξ = view(x, n+1:N)
    ∇h_θ = view(∇h, 1:n)
    ∇h_ξ = view(∇h, n+1:N)
    views = (θ = θ, ξ = ξ, ∇h_θ = ∇h_θ, ∇h_ξ = ∇h_ξ)

    # make kwargs a NamedTuple
    kwt = values(kwargs)

    #
    # Phase 1: Steepest Descent
    #

    # update kwargs with interation limit
    SD_kwargs = (kwt..., maxiters=phase1,)

    # build problem and optimize
    objective = cvxreg_objective
    algmap = cvxreg_iter


    prob1 = ProxDistProblem(SD_variables, derivatives, operators, buffers, views, linsolver)
    _, iteration, _ = optimize!(SteepestDescent(), objective, algmap, prob1, rho, mu; SD_kwargs...)

    #
    # Phase 2: ADMM
    #

    # initialize ADMM variables
    mul!(y, D, x)

    # want the new solution to stay clsoe to current solution
    a = copy(x)

    # rebuild penalty funciton to account for iterations so far
    f = kwt.penalty
    f_penalty(rho, iter) = f(rho, iter + iteration - 1)
    rho = f(rho, iteration)

    # update kwargs with new penalty function and turn off acceleration
    ADMM_kwargs = (kwt..., penalty=f_penalty, accel=Val(:none), maxiters=phase2,)

    # build new problem and optimize
    objective = cvxreg_objective2
    derivatives = (∇f = ∇f, ∇²f = I, ∇q = ∇q, ∇h = ∇h)
    operators = (D = D, DtD = DtD, P = P, A₁ = I, a = a)

    prob2 = ProxDistProblem(ADMM_variables, derivatives, operators, buffers, views, linsolver)
    optimize!(ADMM(), objective, algmap, prob2, rho, mu; ADMM_kwargs...)

    return copy(θ), reshape(ξ, d, n)
end

function cvxreg_objective2(::AlgorithmOption, prob, ρ)
    @unpack x = prob.variables
    @unpack ∇f, ∇q, ∇h = prob.derivatives
    @unpack D, P, a = prob.operators
    @unpack z, Pz, v = prob.buffers
    @unpack θ = prob.views

    # evaulate gradient of loss
    @. ∇f = x - a

    # evaluate gradient of penalty
    mul!(z, D, x)
    @. Pz = P(z)
    @. v = z - Pz
    mul!(∇q, D', v)
    @. ∇h = ∇f + ρ * ∇q

    loss = SqEuclidean()(x, a) / 2
    penalty = dot(v, v)
    normgrad = dot(∇h, ∇h)

    return loss, penalty, normgrad
end
