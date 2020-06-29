"""
```
cvxreg_fit(algorithm::SteepestDescent, response, covariates; kwargs...)
"""
function cvxreg_fit(algorithm::AlgorithmOption, response, covariates;
    rho::Real=1.0, mu::Real=1.0, kwargs...)
    #
    # extract problem information
    d, n = size(covariates) # features × samples
    M = n*n                 # number of subradient constraints
    N = n*(d+1)             # total number of optimization variables

    # allocate optimization variables
    x = zeros(N)
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
    D = CvxRegFM(covariates)
    P(x) = min.(x, 0)
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
    operators = (D = D, P = P, H = H, a = a)

    # allocate buffers for mat-vec multiplication, projections, and so on
    z = similar(Vector{eltype(x)}, M)
    Pz = similar(z)
    v = similar(z)
    b = needs_linsolver(algorithm) ? similar(x) : nothing
    buffers = (z = z, Pz = Pz, v = v, b = b)

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
    @unpack D, a = prob.operators
    @unpack b, Pz = prob.buffers
    linsolver = prob.linsolver

    # build RHS of Ax = b
    mul!(b, D', Pz)
    @inbounds @simd ivdep for j in eachindex(a)
        b[j] = a[j] + ρ*b[j]
    end

    # solve the linear system; assuming x bound to linsolver
    __do_linear_solve!(linsolver, b)

    return 1.0
end

function cvxreg_iter(::ADMM, prob, ρ, μ)
    @unpack x, y, λ = prob.variables
    @unpack D, P, a = prob.operators
    @unpack z, Pz, v, b = prob.buffers
    linsolver = prob.linsolver

    # y block update
    α = (ρ / μ)
    @inbounds @simd for j in eachindex(y)
        y[j] = α/(1+α) * P(z[j] + λ[j]) + 1/(1+α) * (z[j] + λ[j])
    end

    # x block update
    @. v = y - λ
    mul!(b, D', v)
    @simd ivdep for j in eachindex(a)
        @inbounds b[j] = a[j] + μ*b[j]
    end
    __do_linear_solve!(linsolver, b)

    # λ block update
    mul!(z, D, x)
    @inbounds @simd for j in eachindex(λ)
        λ[j] = λ[j] / μ + z[j] - y[j]
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
