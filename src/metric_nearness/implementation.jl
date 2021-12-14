@doc raw"""
    metric_projection(algorithm::AlgorithmOption, A, [W = I]; kwargs...)

Project a symmetric matrix `A` to its nearest semimetric in the sense of the Frobenius norm.

Diagonal entries are assumed to be zero and are ignored. Only the lower triangular part is used. The entries `W[i,j]` act as weights on `A[i,j]`, which are assumed to be non-negative.

The penalized objective used is

```math
h_{\rho}(x) = \frac{1}{2} \|W^{1/2}(x-a)\|^{2} + \frac{\rho}{2} \mathrm{dist}(Dx,C)^{2}
```

where ``a = \mathrm{trivec}(A)`` is a vectorized version of `A` with non-redundant entries and ``C`` is the intersection of the metric cone on semimetrics with a compatible non-negative orthant.

See also: [`MM`](@ref), [`StepestDescent`](@ref), [`ADMM`](@ref), [`MMSubSpace`](@ref), [`initialize_history`](@ref)

# Keyword Arguments

- `rho::Real=1.0`: An initial value for the penalty coefficient. This should match with the choice of annealing schedule, `penalty`.
- `mu::Real=1.0`: An initial value for the step size in `ADMM`.
- `ls=Val(:LSQR)`: Choice of linear solver in `MM`, `ADMM`, and `MMSubSpace`. Choose one of `Val(:LSQR)` or `Val(:CG)` for LSQR or conjugate gradients, respectively.
- `maxiters::Integer=100`: The maximum number of iterations.
- `penalty::Function=__default_schedule__`: A two-argument function `penalty(rho, iter)` that computes the penalty coefficient at iteration `iter+1`. The default setting does nothing.
- `history=nothing`: An object that logs convergence history.
- `rtol::Real=1e-6`: A convergence parameter measuring the relative change in the loss model, $\frac{1}{2} \|W^{1/2}(x-a)\|^{2}$.
- `atol::Real=1e-4`: A convergence parameter measuring the magnitude of the squared distance penalty $\frac{\rho}{2} \mathrm{dist}(Dx,C)^{2}$.
- `accel=Val(:none)`: Choice of an acceleration algorithm. Options are `Val(:none)` and `Val(:nesterov)`.

### Examples

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
function metric_projection(algorithm::AlgorithmOption, A, W=I; ls=Val(:LSQR), kwargs...)
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

    if algorithm isa MMSubSpace
        K = subspace_size(algorithm)
        G = zeros(N, K)
        derivatives = (∇f = ∇f, ∇²f = ∇²f, ∇q = ∇q, ∇h = ∇h, G = G)
    else
        derivatives = (∇f = ∇f, ∇²f = ∇²f, ∇q = ∇q, ∇h = ∇h)
    end

    # generate operators
    D = MetricFM(n)     # fusion matrix
    DtD = D'D           # cache D'D, which creates a small array for reduction algorithms
    P(x) = max.(x, 0)   # projection onto non-negative orthant
    a = similar(x)
    k = 0
    for j in 1:n, i in j+1:n
        a[k+=1] = A[i,j]
    end
    operators = (D = D, DtD = DtD, P = P, a = a)

    # allocate buffers for mat-vec multiplication, projections, and so on
    z = similar(Vector{eltype(x)}, M)
    Pz = similar(z)
    v = similar(z)

    # linear solver only needed for MMSubSpace methods
    # TODO: check that this gets folded in correctly
    if algorithm isa MMSubSpace
        K = subspace_size(algorithm)
        β = zeros(K)

        if ls isa Val{:LSQR}
            A₁ = LinearMap(I, N)
            A₂ = D
            A = MMSOp1(A₁, A₂, G, x, x, 1.0)
            b = similar(typeof(x), N+M)
            linsolver = LSQRWrapper(A, β, b)
        else
            b = similar(typeof(x), K)
            linsolver = CGWrapper(G, β, b)
        end
    else
        b = similar(x)
        linsolver = nothing
    end

    if algorithm isa ADMM
        mul!(y, D, x)
        y_prev = similar(y)
        r = similar(y)
        s = similar(x)
        tmpx = similar(x)
        buffers = (z = z, Pz = Pz, v = v, b = b, y_prev = y_prev, s = s, r = r, tmpx = tmpx)
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
    objective = metric_objective
    algmap = metric_iter
    old_variables = deepcopy(variables)
    prob = ProxDistProblem(variables, old_variables, derivatives, operators, buffers, views, linsolver)
    prob_tuple = (objective, algmap, prob)

    # solve the optimization problem
    optimize!(algorithm, prob_tuple; kwargs...)

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

    loss = SqEuclidean()(x, a)  # ||W^1/2 * (x-y)||^2
    distance = dot(v, v)        # ||D*x - P(D*x)||^2
    objective = 1//2*loss + ρ/2 * distance
    normgrad = dot(∇h, ∇h)      # ||∇h(x)||^2

    return IterationResult(loss, objective, distance, normgrad)
end

############################
#      algorithm maps      #
############################

function metric_iter(::SteepestDescent, prob, ρ, μ)
    @unpack x = prob.variables
    @unpack ∇h = prob.derivatives
    @unpack D, DtD = prob.operators
    @unpack z, tmpx = prob.buffers

    # evaluate step size, γ
    mul!(tmpx, DtD, ∇h)
    a = dot(∇h, ∇h)     # ||∇h(x)||^2
    b = dot(∇h, tmpx)   # ||D*∇h(x)||^2
    c = a               # ||W^1/2 * ∇h(x)||^2
    γ = a / (c + ρ*b + eps())

    # steepest descent, x_new = x_old - γ*∇h(x_old)
    axpy!(-γ, ∇h, x)

    return γ
end

function metric_iter(::MM, prob, ρ, μ)
    @unpack x = prob.variables
    @unpack ∇²f = prob.derivatives
    @unpack D, DtD, a = prob.operators
    @unpack b, Pz, tmpx = prob.buffers

    # build RHS of A'A*x = A'b; A'b = a + ρ*D'P(D*x)
    mul!(b, D', Pz)
    axpby!(true, a, ρ, b)

    # build LHS of A'A*x = A'b; A'A = I + ρ*D'D
    A = ProxDistHessian(∇²f, DtD, tmpx, ρ)

    # solve the linear system directly
    ldiv!(x, A, b)

    return 1.0
end

function metric_iter(::ADMM, prob, ρ, μ)
    @unpack x, y, λ = prob.variables
    @unpack ∇²f = prob.derivatives
    @unpack D, DtD, P, a = prob.operators
    @unpack z, Pz, v, b, tmpx = prob.buffers

    # x block update
    @. v = y - λ
    
    # build RHS of A*x = b; b = a + μ*D'(y-λ)
    mul!(b, D', v)
    axpby!(1, a, μ, b)

    # build LHS of A'A*x = A'b; A'A = I + ρ*D'D
    A = ProxDistHessian(∇²f, DtD, tmpx, μ)

    # solve the linear system directly
    ldiv!(x, A, b)

    # y block update
    α = (ρ / μ)
    mul!(z, D, x)
    @inbounds for j in eachindex(y)
        y[j] = α/(1+α) * P(z[j] + λ[j]) + 1/(1+α) * (z[j] + λ[j])
    end

    # λ block update
    @inbounds for j in eachindex(λ)
        λ[j] = λ[j] + z[j] - y[j]
    end

    return μ
end

function metric_iter(::MMSubSpace, prob, ρ, μ)
    @unpack x = prob.variables
    @unpack ∇²f, ∇h, ∇f, G = prob.derivatives
    @unpack D, DtD = prob.operators
    @unpack β, b, v, tmpx, tmpGx1, tmpGx2 = prob.buffers
    linsolver = prob.linsolver

    # solve linear system Gt*At*A*G * β = Gt*At*b for stepsize
    if linsolver isa LSQRWrapper
        # build LHS, A = [A₁, A₂] * G
        # A₁ = W^1/2 in general
        # A₂ = √ρ*D
        A₁ = LinearMap(I, size(D, 2))
        A₂ = D
        A = MMSOp1(A₁, A₂, G, tmpGx1, tmpGx2, √ρ)

        # build RHS, b = -∇h
        n = length(∇f)
        for j in eachindex(∇f)
            b[j] = -∇f[j]
        end
        for j in eachindex(v)
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

#################################
#   hybrid SD + ADMM prototype  #
#################################

function metric_projection(algorithm::SDADMM, A, W=I;
    rho::Real=1.0, mu::Real=1.0, ls=Val(:LSQR), phase1=10, phase2=10, kwargs...)
    #
    # extract problem dimensions
    n = size(A, 1)      # number of nodes
    m1 = binomial(n, 2) # number of unique non-negativity constraints
    m2 = m1*(n-2)       # number of unique triangle edges
    N = m1              # total number of optimization variables
    M = m1 + m2         # total number of constraints

    # allocate optimization variables...
    X = copy(A)
    x = zeros(N)
    k = 0
    for j in 1:n, i in j+1:n
        @inbounds x[k+=1] = X[i,j]
    end

    # ... for ADMM
    y = zeros(M)
    λ = zeros(M)

    ADMM_variables = (x = x, y = y, λ = λ)

    # ... for SteepestDescent
    SD_variables = (x = x,)

    # allocate derivatives
    ∇f = similar(x)
    ∇q = similar(x)
    ∇h = similar(x)
    ∇²f = I

    derivatives = (∇f = ∇f, ∇²f = ∇²f, ∇q = ∇q, ∇h = ∇h)

    # generate operators
    D = MetricFM(n)     # fusion matrix
    DtD = D'D
    P(x) = max.(x, 0)   # projection onto non-negative orthant
    a = similar(x)
    k = 0
    for j in 1:n, i in j+1:n
        a[k+=1] = A[i,j]
    end
    operators = (D = D, DtD = DtD, P = P, a = a)

    # allocate buffers for mat-vec multiplication, projections, and so on
    z = similar(Vector{eltype(x)}, M)
    Pz = similar(z)
    v = similar(z)

    # linear solver not needed
    b = similar(x) # b has one block
    linsolver = nothing

    # finish initializing buffers
    y_prev = zero(y)
    r = similar(y)
    s = similar(x)
    tmpx = similar(x)
    buffers = (z = z, Pz = Pz, v = v, b = b, y_prev = y_prev, s = s, r = r, tmpx = tmpx)

    # create views, if needed
    views = nothing

    # make kwargs a NamedTuple
    kwt = values(kwargs)

    #
    # Phase 1: Steepest Descent
    #

    # update kwargs with iteration limit
    SD_kwargs = (kwt..., maxiters=phase1,)

    # build problem and optimize
    objective = metric_objective
    algmap = metric_iter
    
    prob1 = ProxDistProblem(SD_variables, derivatives, operators, buffers, views, linsolver)
    _, iteration, _ = optimize!(SteepestDescent(), objective, algmap, prob1, rho, mu; SD_kwargs...)

    #
    # Phase 2: ADMM
    #

    # initialize ADMM variables
    mul!(y, D, x)

    # want the new solution to stay close to current solution
    copyto!(a, x)

    # rebuild penalty function to account for iterations so far
    f = kwt.penalty
    f_penalty(rho, iter) = f(rho, iter + iteration - 1)
    rho = f(rho, iteration)

    # update kwargs with new penalty function and turn off acceleration
    ADMM_kwargs = (kwt..., penalty=f_penalty, accel=Val(:none), maxiters=phase2,)

    # build new problem and optimize

    prob2 = ProxDistProblem(ADMM_variables, derivatives, operators, buffers, views, linsolver)
    optimize!(ADMM(), objective, algmap, prob2, rho, mu; ADMM_kwargs...)

    # symmetrize solution
    k = 0
    for j in 1:n, i in j+1:n
        @inbounds X[i,j] = x[k+=1]
        @inbounds X[j,i] = X[i,j]
    end

    return X
end