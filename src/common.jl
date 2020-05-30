##### annealing schedules #####

__default_schedule(ρ::Real, iteration::Integer) = ρ
slow_schedule(ρ::Real, iteration::Integer) = iteration % 250 == 0 ? 1.5*ρ : ρ
fast_schedule(ρ::Real, iteration::Integer) = iteration % 50 == 0 ? 1.1*ρ : ρ

##### convergence history #####

# notes:
# report 'distance' as dist(Dx,S)^2
# report 'objective' as 0.5 * (loss + rho * distance)
# report 'gradient' as norm(gradient)

function initialize_history(hint::Integer, sample_rate::Integer = 1)
    history = (
        sample_rate = sample_rate,
        loss      = sizehint!(Float64[], hint),
        distance  = sizehint!(Float64[], hint),
        objective = sizehint!(Float64[], hint),
        gradient  = sizehint!(Float64[], hint),
        stepsize  = sizehint!(Float64[], hint),
        rho       = sizehint!(Float64[], hint),
        iteration = sizehint!(Int[], hint),
    )

    return history
end

"""
Package arguments into a `NamedTuple` and standardize reporting:

- Input `distance` is assumed to be 'dist(Dx,C)^2' so we take its square root.
- Input `objective` is reported as '0.5 * (loss + rho * distance)'.
- Input `gradient` is assumed to be 'norm^2' so we take its square root.
"""
function package_data(loss, distance, gradient, stepsize, rho)
    data = (
        loss      = loss,
        distance  = sqrt(distance),
        objective = loss + rho * distance / 2,
        gradient  = sqrt(gradient),
        stepsize  = stepsize,
        rho       = rho,
    )

    return data
end

# default: do nothing
update_history!(::Nothing, data, iteration) = nothing

# implementation: object with named fields
function update_history!(history, data, iteration)
    if iteration % history.sample_rate == 0
        push!(history.loss, data.loss)
        push!(history.distance, data.distance)
        push!(history.objective, data.objective)
        push!(history.gradient, data.gradient)
        push!(history.stepsize, data.stepsize)
        push!(history.rho, data.rho)
        push!(history.iteration, iteration)
    end

    return history
end

##### checking convergence #####

"""
Evaluate convergence using the following three checks:

    1. relative change in `loss` is within `rtol`,
    2. relative change in `dist` is within `rtol`, and
    3. magnitude of `dist` is smaller than `atol`

Returns `true` if any of (1)-(3) are violated, `false` otherwise.
"""
function not_converged(loss, dist, rtol, atol)
    diff1 = abs(loss.new - loss.old)
    diff2 = abs(dist.new - dist.old)

    flag1 = diff1 > rtol * (loss.old + 1)
    flag2 = diff2 > rtol * (dist.old + 1)
    flag3 = dist.new > atol

    return flag1 || flag2 || flag3
end

##### wrapper around CGIterable #####

function __do_linear_solve!(cg_iterator, b)
    # unpack state variables
    x = cg_iterator.x
    u = cg_iterator.u
    r = cg_iterator.r
    c = cg_iterator.c

    # initialize variables according to cg_iterator! with initially_zero = true
    fill!(x, zero(eltype(x)))
    fill!(u, zero(eltype(u)))
    fill!(c, zero(eltype(c)))
    copyto!(r, b)

    tol = sqrt(eps(eltype(b)))
    cg_iterator.mv_products = 0
    cg_iterator.residual = norm(b)
    cg_iterator.prev_residual = one(cg_iterator.residual)
    cg_iterator.reltol = cg_iterator.residual * tol

    for _ in cg_iterator end

    return nothing
end

##### linear operators #####

abstract type FusionMatrix{T} <: LinearMap{T} end

# LinearAlgebra traits
LinearAlgebra.issymmetric(D::FusionMatrix) = false
LinearAlgebra.ishermitian(D::FusionMatrix) = false
LinearAlgebra.isposdef(D::FusionMatrix)    = false

# matrix-vector multiplication
function Base.:(*)(D::FusionMatrix, x::AbstractVector)
    M, N = size(D)
    length(x) == N || throw(DimensionMismatch())
    y = similar(x, promote_type(eltype(D), eltype(x)), M)
    apply_fusion_matrix!(y, D, x)

    return y
end

function Base.:(*)(D::AdjointMap{<:Any,<:FusionMatrix}, x::AbstractVector)
    throw(ArgumentError("Operation not implemented for FusionMatrix"))
end

function Base.:(*)(D::TransposeMap{<:Any,<:FusionMatrix}, x::AbstractVector)
    Dt = D.lmap
    M, N = size(Dt)
    length(x) == M || throw(DimensionMismatch())
    y = similar(x, promote_type(eltype(Dt), eltype(x)), N)
    apply_fusion_matrix_transpose!(y, Dt, x)

    return y
end

function LinearMaps.A_mul_B!(y::AbstractVector, D::FusionMatrix, x::AbstractVector)
    M, N = size(D)
    (length(x) == N && length(y) == M) || throw(DimensionMismatch("A_mul_B!"))
    apply_fusion_matrix!(y, D, x)

    return y
end

function LinearMaps.At_mul_B!(y::AbstractVector, D::FusionMatrix, x::AbstractVector)
    M, N = size(D)
    (length(x) == M && length(y) == N) || throw(DimensionMismatch("At_mul_B!"))
    apply_fusion_matrix_transpose!(y, D, x)

    return y
end

function LinearMaps.Ac_mul_B!(y::AbstractVector, D::FusionMatrix, x::AbstractVector)
    M, N = size(D)
    (length(x) == M && length(y) == N) || throw(DimensionMismatch("Ac_mul_B!"))
    apply_fusion_matrix_conjugate!(y, D, x)

    return y
end

# internal API

apply_fusion_matrix!(y, D::FusionMatrix, x) = error("not implemented for $(typeof(D))")

apply_fusion_matrix_transpose!(y, D::FusionMatrix, x) = error("not implemented for $(typeof(D))")

apply_fusion_matrix_conjugate!(y, D::FusionMatrix, x) = error("not implemented for $(typeof(D))")

apply_gram_matrix!(y, D::Fusionmatrix, x) = error("not implemented for $(typeof(D))")

instantiate_fusion_matrix(D::FusionMatrix) = error("not implemented for $(typeof(D))")

abstract type ProxDistHessian{T} <: LinearMap{T} end

# LinearAlgebra traits
LinearAlgebra.issymmetric(H::ProxDistHessian) = true
LinearAlgebra.ishermitian(H::ProxDistHessian) = false
LinearAlgebra.isposdef(H::ProxDistHessian)    = false

# matrix-vector multiplication
function Base.:(*)(H::ProxDistHessian, x::AbstractVector)
    N = size(H, 1)
    length(x) == N || throw(DimensionMismatch())
    y = similar(x, promote_type(eltype(H), eltype(x)), N)
    apply_hessian!(y, H, x)

    return y
end

function Base.:(*)(H::AdjointMap{<:Any,<:ProxDistHessian}, x::AbstractVector)
    throw(ArgumentError("Operation not implemented for ProxDistHessian"))
end

function Base.:(*)(H::TransposeMap{<:Any,<:ProxDistHessian}, x::AbstractVector)
    return
end

function LinearMaps.A_mul_B!(y::AbstractVector, H::ProxDistHessian, x::AbstractVector)
    N = size(H, 1)
    (length(x) == N && length(y) == N) || throw(DimensionMismatch("A_mul_B!"))
    apply_hessian!(y, H, x)

    return y
end

function LinearMaps.At_mul_B!(y::AbstractVector, H::ProxDistHessian, x::AbstractVector)
    A_mul_B!(y, H, x)
end

function LinearMaps.Ac_mul_B!(y::AbstractVector, H::ProxDistHessian, x::AbstractVector)
    A_mul_B!(y, H, x)
end

# internal API

apply_hessian!(y, H::ProxDistHessian, x) = error("not implemented for $(typeof(A))")

##### trivec

trivec_index(n, i, j) = (i-j) + n*(j-1) - (j*(j-1))>>1

function trivec_view(X)
    n = size(X, 1)
    inds = sizehint!(Int[], binomial(n,2))
    mapping = LinearIndices((1:n, 1:n))
    for j in 1:n, i in j+1:n
        push!(inds, mapping[i,j])
    end
    x = view(X, inds)
    return x
end

function trivec_parent(x::AbstractVector, n::Int)
    p = parent(x)
    x === p && throw(ArgumentError("input may not be a trivec view"))
    X = reshape(p, (n, n))
    return X
end

struct TriVecIndices <: AbstractArray{Int,2}
    n::Int
end

import Base: IndexStyle, axes, size, getindex, @_inline_meta

# AbstractArray implementation
Base.IndexStyle(::Type{<:TriVecIndices}) = IndexLinear()
Base.axes(iter::TriVecIndices) = (Base.OneTo(iter.n), Base.OneTo(iter.n))
Base.size(iter::TriVecIndices) = (iter.n, iter.n)
function Base.getindex(iter::TriVecIndices, i::Int, j::Int)
    @_inline_meta
    @boundscheck checkbounds(iter, i, j)
    j, i = extrema((i, j))
    l = (i == j) ? 0 : trivec_index(iter.n, i, j)
end

##### common solution interface #####

function optimize!(algorithm::AlgorithmOption, eval_h, M, optvars, gradients, operators, buffers;
    ρ_init::Real      = 1.0,
    maxiters::Integer = 100,
    penalty::Function = __default_schedule,
    history::histT    = nothing,
    rtol::Real        = 1e-6,
    atol::Real        = 1e-4,
    accel::accelT     = Val(:none)) where {histT, accelT}
    #
    # construct acceleration strategy
    strategy = get_acceleration_strategy(accel, optvars)

    # initialize
    ρ = ρ_init

    f_loss, h_dist, h_ngrad = eval_h(algorithm, optvars, gradients, operators, buffers, ρ)
    data = package_data(f_loss, h_dist, h_ngrad, one(f_loss), ρ)
    update_history!(history, data, 0)

    loss = (old = f_loss, new = Inf)
    dist = (old = h_dist, new = Inf)
    iteration = 1

    while not_converged(loss, dist, rtol, atol) && iteration ≤ maxiters
        # iterate the algorithm map
        stepsize = M(algorithm, optvars, gradients, operators, buffers, ρ)

        # penalty schedule + acceleration
        ρ_new = penalty(ρ, iteration)
        ρ != ρ_new && restart!(strategy, optvars)
        apply_momentum!(optvars, strategy)
        ρ = ρ_new

        # convergence history
        f_loss, h_dist, h_ngrad = eval_h(algorithm, optvars, gradients, operators, buffers, ρ)
        data = package_data(f_loss, h_dist, h_ngrad, stepsize, ρ)
        update_history!(history, data, iteration)

        loss = (old = loss.new, new = f_loss)
        dist = (old = dist.new, new = h_dist)
        iteration += 1
    end

    return optvars
end
