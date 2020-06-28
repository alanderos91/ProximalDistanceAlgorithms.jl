"""
```
metric_projection(algorithm::AlgorithmOption, W, A; kwargs...)
```
"""
function metric_projection(algorithm::AlgorithmOption, W, A;
    rho::Real = 1.0, mu::Real = 1.0, kwargs...)
    #
    # extract problem dimensions
    n = size(A, 1)      # number of nodes
    m1 = binomial(n, 2) # number of unique non-negativity constraints
    m2 = m1*(n-2)       # number of unique triangle edges
    N = m1              # total number of optimization variables
    M = m1 + m2         # total number of constraints

    # allocate optimization variable
    X = copy(A)
    x = zeros(N)
    k = 0
    for j in 1:n, i in j+1:n
        @inbounds x[k+=1] = X[i,j]
    end
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
    D = MetricFM(n, M, N)   # fusion matrix
    P(x) = max.(x, 0)       # projection onto non-negative orthant
    a = similar(x)
    k = 0
    for j in 1:n, i in j+1:n
        a[k+=1] = A[i,j]
    end
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
    views = nothing

    # pack everything into ProxDistProblem container
    objective = metric_objective
    algmap = metric_iter
    prob = ProxDistProblem(variables, derivatives, operators, buffers, views, linsolver)

    # solve the optimization problem
    optimize!(algorithm, objective, algmap, prob, rho, mu; kwargs...)

    # symmetrize solution
    k = 0
    for j in 1:n, i in j+1:n
        @inbounds X[i,j] = x[k+=1]
        @inbounds X[j,i] = X[i,j]
    end

    return X
end

#########################
#       objective       #
#########################

function metric_objective(::AlgorithmOption, prob, ρ)
    @unpack x = prob.variables
    @unpack ∇f, ∇q, ∇h = prob.derivatives
    @unpack D, P, a = prob.operators
    @unpack z, Pz, v = prob.buffers

    mul!(z, D, x)
    @. Pz = P(z)
    @. v = z - Pz
    @. ∇f = x - a
    mul!(∇q, D', v)
    @. ∇h = ∇f + ρ * ∇q

    loss = SqEuclidean()(x, a) / 2  # 1/2 * ||W^1/2 * (x-y)||^2
    penalty = dot(v, v)             # D*x - P(D*x)
    normgrad = dot(∇h, ∇h)          # ||∇h(x)||^2

    return loss, penalty, normgrad
end

############################
#      algorithm maps      #
############################

function metric_iter(::SteepestDescent, prob, ρ, μ)
    @unpack x = prob.variables
    @unpack ∇h = prob.derivatives
    @unpack D = prob.operators
    @unpack z = prob.buffers

    # evaluate step size, γ
    mul!(z, D, ∇h)
    a = dot(∇h, ∇h)     # ||∇h(x)||^2
    b = dot(z, z)       # ||D*∇h(x)||^2
    c = a               # ||W^1/2 * ∇h(x)||^2
    γ = a / (c + ρ*b + eps())

    # steepest descent, x_new = x_old - γ*∇h(x_old)
    axpy!(-γ, ∇h, x)

    return γ
end

function metric_iter(::MM, prob, ρ, μ)
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

function metric_iter(::ADMM, prob, ρ, μ)
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
    axpby!(1, a, μ, b)
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

function metric_example(n; weighted = false)
    n < 3 && error("number of nodes must be ≥ 3")

    D = zeros(n, n)
    for j in 1:n, i in j+1:n
        u = 10*rand()
        D[i,j] = u
        D[j,i] = u
    end

    W = zeros(n, n)
    for j in 1:n, i in j+1:n
        u = weighted ? rand() : 1.0
        W[i,j] = u
        W[j,i] = u
    end

    return W, D
end
