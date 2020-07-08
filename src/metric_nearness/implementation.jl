@doc raw"""
    metric_projection(algorithm::AlgorithmOption, A, [W = I]; kwargs...)

Project a symmetric matrix `A` to its nearest semimetric in the sense of the Frobenius norm.

Diagonal entries are assumed to be zero and are ignored. Only the lower triangular part is used. The entries `W[i,j]` act as weights on `A[i,j]`, which are assumed to be non-negative.

The penalized objective used is

``
h_{\rho}(x) = \frac{1}{2} \|W^{1/2}(x-a)\|^{2} + \frac{\rho}{2} \mathrm{dist}(Dx,C)^{2}
``

where ``a = \mathrm{trivec}(A)`` is a vectorized version of `A` with non-redundant entries and ``C`` is the intersection of the metric cone on semimetrics with a compatible non-negative orthant.

See also: [`MM`](@ref), [`StepestDescent`](@ref), [`ADMM`](@ref)

# Keyword Arguments

- `rho::Real=1.0`: An initial value for the penalty coefficient. This should match with the choice of annealing schedule, `penalty`.
- `mu::Real=1.0`: An initial value for the step size in `ADMM()`.
- `ls=Val(:LSQR)`: Choice of linear solver in `MM()` and `ADMM()`. Choose one of `Val(:LSQR)` or `Val(:CG)` for LSQR and conjugate gradients, respectively.
- `maxiters::Integer=100`: The maximum number of iterations.
- `penalty::Function=__default_schedule__`: A two-argument function `penalty(rho, iter)` that computes the penalty coefficient at iteration `iter+1`. The default setting does nothing.
- `history=nothing`
- `rtol::Real=1e-6`: A convergence parameter measuring the relative change in the loss model, $\frac{1}{2} \|W^{1/2}(x-a)\|^{2}$.
- `atol::Real=1e-4`: A convergence parameter measuring the magnitude of the squared distance penalty $\frac{\rho}{2} \mathrm{dist}(D*x,C)^{2}$.
- `accel=Val(:none)`: Choice of an acceleration algorithm. Options are `Val(:none)` and `Val(:nesterov)`.

# Examples

**How to define an annealing schedule**:
```jldoctest
julia> f(rho, iter) = min(1e6, 1.1^floor(iter/20))
f (generic function with 1 method)

julia> f(1.0, 5)
1.0

julia> f(1.0, 20)
1.1

julia> g(rho, iter) = iter % 20 == 0 ? min(1e6, 1.1*rho) : rho
g (generic function with 1 method)

julia> f(1.0, 21) == g(1.0, 21)
true
```

**How to use `metric_projection`**:
```jldoctest
julia> A = [0.0 8.0 6.0; 8.0 0.0 1.0; 6.0 1.0 0.0]
3×3 Array{Float64,2}:
 0.0  8.0  6.0
 8.0  0.0  1.0
 6.0  1.0  0.0

julia> f(rho, iter) = min(1e6, 1.1^floor(iter/20))
f (generic function with 1 method)

julia> metric_projection(SteepestDescent(), A, penalty=f, maxiters=200)
3×3 Array{Float64,2}:
 0.0      7.70795  6.29205
 7.70795  0.0      1.29205
 6.29205  1.29205  0.0
```
"""
function metric_projection(algorithm::AlgorithmOption, A, W=I;
    rho::Real=1.0, mu::Real=1.0, ls=Val(:LSQR), kwargs...)
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
    operators = (D = D, P = P, a = a)

    # allocate buffers for mat-vec multiplication, projections, and so on
    z = similar(Vector{eltype(x)}, M)
    Pz = similar(z)
    v = similar(z)

    # select linear solver, if needed
    if needs_linsolver(algorithm)
        if ls isa Val{:LSQR}
            A = QuadLHS(LinearMap(I, N), D, 1.0)
            b = similar(typeof(x), N+M) # b has two block
            linsolver = LSQRWrapper(A, x, b)
        else
            b = similar(x) # b has one block
            linsolver = CGWrapper(D, x, b)
        end
    else
        b = nothing
        linsolver = nothing
    end

    if algorithm isa ADMM
        mul!(y, D, x)
        s = similar(y)
        r = similar(y)
        buffers = (z = z, Pz = Pz, v = v, b = b, s = s, r = r)
    else
        buffers = (z = z, Pz = Pz, v = v, b = b)
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
    a = dot(∇h, ∇h) # ||∇h(x)||^2
    b = dot(z, z)   # ||D*∇h(x)||^2
    c = a           # ||W^1/2 * ∇h(x)||^2
    γ = a / (c + ρ*b + eps())

    # steepest descent, x_new = x_old - γ*∇h(x_old)
    axpy!(-γ, ∇h, x)

    return γ
end

function metric_iter(::MM, prob, ρ, μ)
    @unpack x = prob.variables
    @unpack ∇²f = prob.derivatives
    @unpack D, a = prob.operators
    @unpack b, Pz = prob.buffers
    linsolver = prob.linsolver

    if linsolver isa LSQRWrapper
        # build LHS of A*x = b
        # forms a BlockMap so non-allocating
        # however, A*x and A'b have small allocations due to views?
        A = QuadLHS(LinearMap(I, size(D, 2)), D, √ρ)

        # build RHS of A*x = b; b = [a; √ρ * P(D*x)]
        n = length(a)
        copyto!(b, 1, a, 1, n)
        for k in eachindex(Pz)
            b[n+k] = √ρ * Pz[k]
        end
    else
        # LHS of A*x = b is already stored
        A = ProxDistHessian(size(D, 2), ρ, ∇²f, D'D)

        # build RHS of A*x = b; b = a + ρ*D'P(D*x)
        mul!(b, D', Pz)
        axpby!(1, a, ρ, b)
    end

    # solve the linear system
    linsolve!(linsolver, x, A, b)

    return 1.0
end

function metric_iter(::ADMM, prob, ρ, μ)
    @unpack x, y, λ = prob.variables
    @unpack ∇²f = prob.derivatives
    @unpack D, P, a = prob.operators
    @unpack z, Pz, v, b = prob.buffers
    linsolver = prob.linsolver

    # x block update
    @. v = y - λ
    if linsolver isa LSQRWrapper
        # build LHS of A*x = b
        # forms a BlockMap so non-allocating
        # however, A*x and A'b have small allocations due to views?
        A = QuadLHS(LinearMap(I, size(D, 2)), D, √μ)

        # build RHS of A*x = b; b = [a; √μ * (y-λ)]
        n = length(a)
        copyto!(b, 1, a, 1, length(a))
        for k in eachindex(v)
            @inbounds b[n+k] = √μ * v[k]
        end
    else
        # LHS of A*x = b is already stored
        A = ProxDistHessian(size(D, 2), μ, ∇²f, D'D)

        # build RHS of A*x = b; b = a + μ*D'(y-λ)
        mul!(b, D', v)
        axpby!(1, a, μ, b)
    end

    # solve the linear system
    linsolve!(linsolver, x, A, b)

    # y block update
    α = (ρ / μ)
    mul!(z, D, x)
    @inbounds for j in eachindex(y)
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

@doc """
    metric_example(n::Integer; weighted::Bool=false)

Simulate a dissimilarity matrix `D` on `n` nodes with weights `W`, if desired, and return the result as `(W, D)`.
"""
function metric_example(n::Integer; weighted::Bool=false)
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
