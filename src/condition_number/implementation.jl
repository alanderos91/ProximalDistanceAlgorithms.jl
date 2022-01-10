@doc raw"""
    reduce_cond(algorithm::AlgorithmOption, c, A; kwargs...)

Project matrix `A` to its nearest matrix `B` (in the sense of the Frobenius
norm) such that `cond(B) ≈ c`.

The penalized objective used is

```math
h_{\rho}(x) = \frac{1}{2} \|x-y\|^{2} + \frac{\rho}{2} \mathrm{dist}{(Dx,C)}^{2}
```

where ``x`` and ``y`` are the singular values for `B` and `A`, respectively.
The object `A` can be a matrix, an `SVD` factorization produced by `svd(A)`, or a vector of singular values in decreasing order.

See also: [`MM`](@ref), [`StepestDescent`](@ref), [`ADMM`](@ref), [`MMSubSpace`](@ref), [`initialize_history`](@ref)

# Keyword Arguments

- `ls=Val(:LSMR)`: Choice of linear solver in `MM`, `ADMM`, and `MMSubSpace`. Choose one of `Val(:LSMR)`, `Val(:LSQR)`, `Val(:CG)` for LSMR, LSQR, or conjugate gradients, respectively.
"""
function reduce_cond(algorithm::AlgorithmOption, c, A; ls::LS=Val(:LSMR), kwargs...) where LS
    if !(algorithm isa MMSubSpace) && !(ls === nothing)
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

    if algorithm isa MMSubSpace
        K = subspace_size(algorithm)
        G = zeros(N, K)
        derivatives = (∇f = ∇f, ∇²f = ∇²f, ∇q = ∇q, ∇h = ∇h, G = G)
    else
        derivatives = (∇f = ∇f, ∇²f = ∇²f, ∇q = ∇q, ∇h = ∇h)
    end

    # generate operators
    D = CondNumFM(c, M, N)
    P(x) = min.(x, 0)
    operators = (D = D, P = P, σ = σ)

    # allocate buffers for mat-vec multiplication, projections, and so on
    z = similar(Vector{eltype(x)}, M)   # cache for D*x
    Pz = similar(z)                     # cache for P(D*x)
    v = similar(z)                      # cache for D*x - P(D*x)

    # select linear solver, if needed
    if algorithm isa MMSubSpace
        K = subspace_size(algorithm)
        β = zeros(K)

        if ls isa Val{:LSQR}
            A₁ = LinearMap(I, size(D, 2))
            A₂ = D
            A = MMSOp1(A₁, A₂, G, x, x, 1.0)
            b = similar(typeof(x), size(A, 1))
            linsolver = LSQRWrapper(A, β, b)
        elseif ls isa Val{:LSMR}
            A₁ = LinearMap(I, size(D, 2))
            A₂ = D
            A = MMSOp1(A₁, A₂, G, x, x, 1.0)
            b = similar(typeof(x), size(A, 1))
            linsolver = LSMRWrapper(A, β, b)
        elseif ls isa Val{:CG}
            b = similar(typeof(x), K)
            linsolver = CGWrapper(G, β, b)
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
        buffers = (z = z, Pz = Pz, v = v, b = b, y_prev = y_prev, s = s, r = r)
    elseif algorithm isa MMSubSpace
        tmpGx1 = zeros(N)
        tmpGx2 = zeros(N)
        tmpx = similar(x)
        buffers = (z = z, Pz = Pz, v = v, b = b, β = β, tmpx = tmpx, tmpGx1 = tmpGx1, tmpGx2 = tmpGx2)
    else
        buffers = (z = z, Pz = Pz, v = v, b = b)
    end

    # create views, if needed
    views = nothing

    # pack everything into ProxDistProblem container
    objective = condnum_objective
    algmap = condnum_iter
    old_variables = deepcopy(variables)
    prob = ProxDistProblem(variables, old_variables, derivatives, operators, buffers, views, linsolver)
    prob_tuple = (objective, algmap, prob)

    # solve the optimization problem
    optimize!(algorithm, prob_tuple; kwargs...)

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

    # projection + gradient
    mul!(z, D, x)
    @. Pz = P(z)
    @. v = z - Pz
    @. ∇f = x - σ
    mul!(∇q, D', v)
    @. ∇h = ∇f + ρ*∇q

    loss = SqEuclidean()(x, σ)
    distance = dot(v, v)
    objective = 1//2 * loss + ρ/2 * distance
    normgrad = dot(∇h, ∇h)

    return IterationResult(loss, objective, distance, normgrad)
end

############################
#      algorithm maps      #
############################


function condnum_iter(::SteepestDescent, prob, ρ, μ)
    @unpack x = prob.variables
    @unpack ∇f, ∇q, ∇h = prob.derivatives
    @unpack D, P, σ = prob.operators
    @unpack z, Pz, v = prob.buffers

    # projection + gradient
    mul!(z, D, x)
    @. Pz = P(z)
    @. v = z - Pz
    @. ∇f = x - σ
    mul!(∇q, D', v)
    @. ∇h = ∇f + ρ*∇q
    
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
    @unpack D, P, σ = prob.operators
    @unpack z, Pz = prob.buffers

    # projection
    mul!(z, D, x)
    @. Pz = P(z)

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
        @inbounds λ[j] = λ[j] + z[j] - y[j]
    end

    return μ
end

function condnum_iter(::MMSubSpace, prob, ρ, μ)
    @unpack x = prob.variables
    @unpack ∇²f, ∇h, ∇f, ∇q, G = prob.derivatives
    @unpack P, D, σ = prob.operators
    @unpack β, b, z, Pz, v, tmpx, tmpGx1, tmpGx2 = prob.buffers
    linsolver = prob.linsolver

    # projection + gradient
    mul!(z, D, x)
    @. Pz = P(z)
    @. v = z - Pz
    @. ∇f = x - σ
    mul!(∇q, D', v)
    @. ∇h = ∇f + ρ*∇q

    # solve linear system Gt*At*A*G * β = Gt*At*b for stepsize
    if linsolver isa LSQRWrapper || linsolver isa LSMRWrapper
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

#################
#   utilities   #
#################

function extract_svd(M::Matrix)
    F = svd(M)
    return (F.S, F.U, F.Vt)
end
extract_svd(M::SVD) = (M.S, M.U, M.Vt)
extract_svd(M::Vector) = (M, LinearAlgebra.I, LinearAlgebra.I)
