"""
```
reduce_cond(algorithm::AlgorithmOption, c, M; kwargs...)
```
"""
function reduce_cond(algorithm::AlgorithmOption, c, M;
    rho::Real = 1.0, mu::Real = 1.0, kwargs...)
    #
    # extract problem dimensions
    σ, U, Vt = extract_svd(M)       # svs, left sv-vecs, right sv-vecs
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
    if algorithm isa SteepestDescent
        H = nothing
    elseif algorithm isa MM
        H = ProxDistHessian(N, rho, ∇²f, D'D)
    else
        H = ProxDistHessian(N, mu, ∇²f, D'D)
    end
    operators = (D = D, P = P, H = H, σ = σ)

    # allocate buffers for mat-vec multiplication, projections, and so on
    z = similar(Vector{eltype(x)}, M)   # cache for D*x
    Pz = similar(z)                     # cache for P(D*x)
    v = similar(z)                      # cache for D*x - P(D*x)
    buffers = (z = z, Pz = Pz, v = v)

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
    β = -2*c*ρ
    u = 1 / α
    v = sum(x) / (α/β + p)

    @simd ivdep for k in eachindex(x)
        @inbounds x[k] = u*(x[k] - v)
    end

    return 1.0
end

function condnum_iter(::ADMM, prob, ρ, μ)
    @unpack x, y, λ = prob.variables
    @unpack ∇f = prob.derivatives
    @unpack D, P, σ = prob.operators
    @unpack z, Pz, v = prob.buffers
    linsolver = prob.linsolver

    # y block update
    α = (ρ / μ)
    @inbounds @simd for j in eachindex(y)
        y[j] = α/(1+α) * P(z[j] + λ[j]) + 1/(1+α) * (z[j] + λ[j])
    end

    # x block update
    @. v = y - λ
    mul!(x, D', v)
    axpby!(1, σ, μ, x)

    c = D.c
    p = D.N
    α = (1 + μ*p*(c^2+1))
    β = -2*c*μ
    u = 1 / α
    v = sum(x) / (α/β + μ)

    @simd for k in eachindex(x)
        @inbounds x[k] = u*(x[k] - v)
    end

    # λ block update
    mul!(z, D, x)
    @inbounds @simd for j in eachindex(λ)
        λ[j] = λ[j] / μ + z[j] - y[j]
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
