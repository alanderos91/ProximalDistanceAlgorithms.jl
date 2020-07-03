"""
```
cvxreg_fit(algorithm::SteepestDescent, response, covariates; kwargs...)
"""
function cvxreg_fit(algorithm::AlgorithmOption, response, covariates;
    rho::Real=1.0, mu::Real=1.0, ls::LS=Val(:LSQR), kwargs...) where LS
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
    ∇f = needs_gradient(algorithm) ? similar(x) : nothing
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
    derivatives = (∇f = ∇f, ∇²f = ∇²f, ∇q = ∇q, ∇h = ∇h)

    # generate operators
    D = instantiate_fusion_matrix(CvxRegFM(covariates))
    # D = CvxRegFM(covariates)
    P(x) = min.(x, 0)
    A₁ = [LinearMap(I, n) LinearMap(0*I, n)] # needed to reduce to 1 vector
    a = response
    if needs_hessian(algorithm)
        if algorithm isa MM
            H = ProxDistHessian(N, rho, ∇²f, D'D)
        else
            H = ProxDistHessian(N, mu, ∇²f, D'D)
        end
    else
        H = nothing
    end
    operators = (D = D, P = P, H = H, A₁ = A₁, a = a)

    # allocate buffers for mat-vec multiplication, projections, and so on
    z = similar(Vector{eltype(x)}, M)
    Pz = similar(z)
    v = similar(z)

    # select linear solver, if needed
    if needs_linsolver(algorithm)
        if ls isa Val{:LSQR}
            b = similar(typeof(x), size(A₁,1)+M) # b has two blocks
            linsolver = LSQRWrapper([A₁;LinearMap(D)], x, b)
        else
            b = similar(x)  # b has one block
            linsolver = CGWrapper(D, x, b)
        end
    else
        b = nothing
        linsolver = nothing
    end

    if algorithm isa ADMM
        r = similar(y)
        s = similar(y)
        buffers = (z = z, Pz = Pz, v = v, b = b, r = r, s = s)
    else
        buffers = (z = z, Pz = Pz, v = v, b = b)
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
    prob = ProxDistProblem(variables, derivatives, operators, buffers, views, linsolver)

    # solve the optimization problem
    optimize!(algorithm, objective, algmap, prob, rho, mu; kwargs...)

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
    @inbounds @simd ivdep for j in eachindex(a)
        ∇f[j] = θ[j] - a[j]
    end

    # evaluate gradient of penalty
    mul!(z, D, x)
    @. Pz = P(z)
    @. v = z - Pz
    mul!(∇q, D', v)
    @. ∇h = ∇f + ρ * ∇q

    loss = SqEuclidean()(θ, a) / 2
    penalty = dot(v, v)
    normgrad = dot(∇h, ∇h)

    return loss, penalty, normgrad
end

############################
#      algorithm maps      #
############################

function cvxreg_iter(::SteepestDescent, prob, ρ, μ)
    @unpack x = prob.variables
    @unpack ∇h = prob.derivatives
    @unpack D = prob.operators
    @unpack z = prob.buffers
    @unpack ∇h_θ, ∇h_ξ = prob.views

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
    @unpack D, H, A₁, a = prob.operators   # a is bound to response
    @unpack b, Pz = prob.buffers
    linsolver = prob.linsolver

    if linsolver isa LSQRWrapper
        # build LHS of A*x = b
        # forms a BlockMap so non-allocating
        # however, A*x and A'b have small allocations due to views?
        A = [A₁; √ρ*LinearMap(D)]

        # build RHS of A*x = b; b = [a; √ρ * P(D*x)]
        n = length(a)
        copyto!(b, 1, a, 1, n)
        for k in eachindex(Pz)
            b[n+k] = √ρ * Pz[k]
        end
    else
        # LHS of A*x = b is already stored
        A = H

        # build RHS of A*x = b; b = a + ρ*D'P(D*x)
        mul!(b, D', Pz)
        @inbounds @simd ivdep for j in eachindex(a)
            b[j] = a[j] + ρ*b[j] # can't do axpby! due to shape
        end
    end

    # solve the linear system
    linsolve!(linsolver, x, A, b)

    return 1.0
end

function cvxreg_iter(::ADMM, prob, ρ, μ)
    @unpack x, y, λ = prob.variables
    @unpack A₁, D, H, P, a = prob.operators
    @unpack z, Pz, v, b = prob.buffers
    linsolver = prob.linsolver

    # x block update
    @. v = y - λ
    if linsolver isa LSQRWrapper
        # build LHS of A*x = b
        # forms a BlockMap so non-allocating
        # however, A*x and A'b have small allocations due to views?
        A = [A₁; √μ*LinearMap(D)]

        # build RHS of A*x = b; b = [a; √μ * (y-λ)]
        n = length(a)
        copyto!(b, 1, a, 1, length(a))
        for k in eachindex(v)
            @inbounds b[n+k] = √μ * v[k]
        end
    else
        # LHS of A*x = b is already stored
        A = H

        # build RHS of A*x = b; b = a + μ*D'(y-λ)
        mul!(b, D', v)
        @inbounds @simd ivdep for j in eachindex(a)
            b[j] = a[j] + μ*b[j] # can't do axpby! due to shape
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
        λ[j] = λ[j] + μ * (z[j] - y[j])
    end

    return μ
end

#################################
#   simulate problem instance   #
#################################

"""
Generate a `d` by `n` matrix `X` with `d` covariates and `n` samples.
Samples are uniform over the cube `[-1,1]^d`.

Output is `(X, xdata)`, where `xdata` stores each sample as a vector.
"""
function cvxreg_simulate_covariates(d, n)
    xdata = sort!([2*rand(d) .- 1 for _ in 1:n])
    X = hcat(xdata...)

    return X, xdata
end

"""
Evaluate `φ: R^d --> R` at the points in `xdata` and simulate samples
`y = φ(x) + ε` where `ε ~ N(0, σ²)`.

Output is returned as `(y, φ(x))`.
"""
function cvxreg_simulate_responses(φ, xdata, σ)
    y_truth = φ.(xdata)
    noise = σ*randn(length(xdata))
    y = y_truth + noise

    return y, y_truth
end

"""
Generate an instance of a convex regression problem based on a convex function `φ: R^d --> R` with `n` samples.
The `σ` parameter is the standard deviation of iid perturbations applied to the true values.

Output is returned as `(y, φ(x), X)`.
"""
function cvxreg_example(φ, d, n, σ)
    X, xdata = cvxreg_simulate_covariates(d, n)
    y, y_truth = cvxreg_simulate_responses(φ, xdata, σ)

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
