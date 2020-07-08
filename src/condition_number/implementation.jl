@doc raw"""
    reduce_cond(algorithm::AlgorithmOption, c, A; kwargs...)

Project matrix `A` to its nearest matrix `B` (in the sense of the Frobenius norm) such that `cond(B) ≈ c`.

The penalized objective used is

```math
    h_{\rho}(x) = \frac{1}{2} \|x-y\|^{2} + \frac{\rho}{2} \mathrm{dist}{(Dx,C)}^{2}
```

where ``x`` and ``y`` are the singular values for `B` and `A`, respectively.
The object `A` can be a matrix, an `SVD` factorization produced by `svd(A)`, or a vector of singular values in decreasing order.

See also: [`MM`](@ref), [`StepestDescent`](@ref), [`ADMM`](@ref)

# Keyword Arguments

- `rho::Real=1.0`: An initial value for the penalty coefficient. This should match with the choice of annealing schedule, `penalty`.
- `mu::Real=1.0`: An initial value for the step size in `ADMM()`.
- `maxiters::Integer=100`: The maximum number of iterations.
- `penalty::Function=__default_schedule__`: A two-argument function `penalty(rho, iter)` that computes the penalty coefficient at iteration `iter+1`. The default setting does nothing.
- `history=nothing`
- `rtol::Real=1e-6`: A convergence parameter measuring the relative change in the loss model, $\frac{1}{2} \|(x-y)\|^{2}$.
- `atol::Real=1e-4`: A convergence parameter measuring the magnitude of the squared distance penalty $\frac{\rho}{2} \mathrm{dist}(Dx,C)^{2}$.
- `accel=Val(:none)`: Choice of an acceleration algorithm. Options are `Val(:none)` and `Val(:nesterov)`.
"""
function reduce_cond(algorithm::AlgorithmOption, c, A;
    rho::Real=1.0, mu::Real=1.0, ls=nothing, kwargs...)
    if !(ls === nothing)
        @warn "Iterative linear solver not required. Option $(ls) will be ignored."
    end
    #
    # extract problem dimensions
    σ, U, Vt = extract_svd(A)       # svs, left sv-vecs, right sv-vecs
    N = length(σ)                   # number of optimization variables
    M = N*N                         # number of constraints

    # allocate optiimzation variable
    x = copy(σ)
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
    D = CondNumFM(c, M, N)
    P(x) = min.(x, 0)
    operators = (D = D, P = P, σ = σ)

    # allocate buffers for mat-vec multiplication, projections, and so on
    z = similar(Vector{eltype(x)}, M)   # cache for D*x
    Pz = similar(z)                     # cache for P(D*x)
    v = similar(z)                      # cache for D*x - P(D*x)

    if algorithm isa ADMM
        s = similar(y)
        r = similar(y)
        mul!(y, D, x)
        buffers = (z = z, Pz = Pz, v = v, s = s, r = r)
    else
        buffers = (z = z, Pz = Pz, v = v)
    end

    # select linear solver, if needed
    # not used, H⁻¹ has an explicit inverse
    linsolver = nothing

    # create views, if needed
    views = nothing

    # pack everything into ProxDistProblem container
    objective = condnum_objective
    algmap = condnum_iter
    prob = ProxDistProblem(variables, derivatives, operators, buffers, views, linsolver)

    # solve the optimization problem
    optimize!(algorithm, objective, algmap, prob, rho, mu; kwargs...)

    return U*Diagonal(x)*Vt
end

#########################
#       objective       #
#########################

function condnum_objective(::AlgorithmOption, prob, ρ)
    @unpack x = prob.variables
    @unpack ∇f, ∇q, ∇h = prob.derivatives
    @unpack D, P, σ = prob.operators
    @unpack z, Pz, v = prob.buffers

    mul!(z, D, x)
    @. Pz = P(z)
    @. v = z - Pz
    @. ∇f = x - σ
    mul!(∇q, D', v)
    @. ∇h = ∇f + ρ*∇q

    loss = SqEuclidean()(x, σ)
    penalty = dot(v, v)
    normgrad = dot(∇h, ∇h)

    return loss, penalty, normgrad
end

############################
#      algorithm maps      #
############################


function condnum_iter(::SteepestDescent, prob, ρ, μ)
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

function condnum_iter(::MM, prob, ρ, μ)
    @unpack x = prob.variables
    @unpack D, σ = prob.operators
    @unpack Pz = prob.buffers

    # compute x = (I + ρ*D'D)^{-1} * (I; √ρ D)' * (σ; √ρ P(z))
    mul!(x, D', Pz)
    axpby!(1, σ, ρ, x)

    c = D.c
    p = D.N
    α = (1 + ρ*p*(c^2+1))
    β = 2*c*ρ
    u = 1/α
    w = sum(x) / (p - α/β)

    @simd ivdep for k in eachindex(x)
        @inbounds x[k] = u*(x[k] - w)
    end

    return 1.0
end

function condnum_iter(::ADMM, prob, ρ, μ)
    @unpack x, y, λ = prob.variables
    @unpack D, P, σ = prob.operators
    @unpack z, Pz, v = prob.buffers
    linsolver = prob.linsolver

    # x block update
    @. v = y - λ
    mul!(x, D', v)
    axpby!(1, σ, μ, x)

    c = D.c
    p = D.N
    α = 1 + μ*p*(c^2+1)
    β = 2*c*μ
    u = 1 / α
    w = sum(x) / (p-α/β)

    @simd for k in eachindex(x)
        @inbounds x[k] = u*(x[k] - w)
    end

    # y block update
    α = (ρ / μ)
    mul!(z, D, x)
    @simd for j in eachindex(y)
        @inbounds y[j] = α/(1+α) * P(z[j] + λ[j]) + 1/(1+α) * (z[j] + λ[j])
    end

    # λ block update
    @simd for j in eachindex(λ)
        @inbounds λ[j] = λ[j] + μ * (z[j] - y[j])
    end

    return μ
end

#################
#   utilities   #
#################

function extract_svd(M::Matrix)
    F = svd(M)
    return (F.S, F.U, F.Vt)
end
extract_svd(M::SVD) = (M.S, M.U, M.Vt)
extract_svd(M::Vector) = (M, LinearAlgebra.I, LinearAlgebra.I)
