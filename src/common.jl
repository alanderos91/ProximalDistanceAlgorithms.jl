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
    # copyto!(r, b)
    for j in eachindex(r)
        @inbounds r[j] = b[j]
    end

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

apply_gram_matrix!(y, D::FusionMatrix, x) = error("not implemented for $(typeof(D))")

instantiate_fusion_matrix(D::FusionMatrix) = error("not implemented for $(typeof(D))")

abstract type FusionGramMatrix{T} <: LinearMap{T} end

# LinearAlgebra traits
LinearAlgebra.issymmetric(DtD::FusionGramMatrix) = true
LinearAlgebra.ishermitian(DtD::FusionGramMatrix) = false
LinearAlgebra.isposdef(DtD::FusionGramMatrix)    = false

# matrix-vector multiplication
function Base.:(*)(DtD::FusionGramMatrix, x::AbstractVector)
    N = size(DtD, 1)
    length(x) == N || throw(DimensionMismatch())
    y = similar(x, promote_type(eltype(DtD), eltype(x)), N)
    apply_fusion_gram_matrix!(y, DtD, x)

    return y
end

function Base.:(*)(DtD::AdjointMap{<:Any,<:FusionGramMatrix}, x::AbstractVector)
    throw(ArgumentError("Operation not implemented for $(typeof(DtD))"))
end

function Base.:(*)(DtD::TransposeMap{<:Any,<:FusionGramMatrix}, x::AbstractVector)
    N = size(DtD, 1)
    length(x) == N || throw(DimensionMismatch())
    y = similar(x, promote_type(eltype(DtD), eltype(x)), N)
    apply_fusion_gram_matrix!(y, DtD.lmap, x)

    return y
end

function LinearMaps.A_mul_B!(y::AbstractVector, DtD::FusionGramMatrix, x::AbstractVector)
    N = size(DtD, 1)
    (length(x) == N && length(y) == N) || throw(DimensionMismatch("A_mul_B!"))
    apply_fusion_gram_matrix!(y, DtD, x)

    return y
end

function LinearMaps.At_mul_B!(y::AbstractVector, DtD::FusionGramMatrix, x::AbstractVector)
    A_mul_B!(y, DtD, x)
end

function LinearMaps.Ac_mul_B!(y::AbstractVector, DtD::FusionGramMatrix, x::AbstractVector)
    A_mul_B!(y, DtD, x)
end

# internal API

apply_fusion_gram_matrix!(y, DtD::FusionGramMatrix, x) = throw(ArgumentError("Operation not implemented for $(typeof(DtD))"))

struct ProxDistHessian{T,matT1,matT2} <: LinearMap{T}
    N::Int
    ρ::T
    ∇²f::matT1
    DtD::matT2
end

Base.size(H::ProxDistHessian) = (H.N, H.N)

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
    N = size(H, 1)
    length(x) == N || throw(DimensionMismatch())
    y = similar(x, promote_type(eltype(H), eltype(x)), N)
    apply_hessian!(y, H.lmap, x)

    return y
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

function apply_hessian!(y, H::ProxDistHessian, x)
    mul!(y, H.DtD, x)
    mul!(y, H.∇²f, x, 1, H.ρ)
    return y
end

##### trivec

trivec_index(n, i, j) = (i-j) + n*(j-1) - (j*(j-1))>>1

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

#########################
#   sparse projections  #
#########################

const MaxParam = Base.Order.Reverse
const MinParam = Base.Order.Forward
const MaxParamT = typeof(MaxParam)
const MinParamT = typeof(MinParam)

struct SparseProjection{order,T} <: Function
    pivot::T
end

SparseProjection{order}(pivot::T) where {order,T} = SparseProjection{order,T}(pivot)
SparseProjection(order,pivot) = SparseProjection{typeof(order)}(pivot)

# parameterize by largest entries
(P::SparseProjection{MaxParamT})(x) = x ≥ P.pivot ? x : zero(x)

# parameterize by smallest entries
(P::SparseProjection{MinParamT})(x) = x > P.pivot ? x : zero(x)

function compute_sparse_projection(xs, ::MaxParamT, K)
    pivot = partialsort!(xs, K, rev = true)
    return SparseProjection{MaxParamT}(pivot)
end

function compute_sparse_projection(xs, ::MinParamT, K)
    pivot = partialsort!(xs, K, rev = false)
    return SparseProjection{MinParamT}(pivot)
end

function swap!(h, i, j)
    h[i], h[j] = h[j], h[i]
    return nothing
end

function partial_heapsort(xs, ord, K)
    n = length(xs)
    heapify!(xs, ord)
    j = 1
    while j < K && j < n
        swap!(xs, 1, n-j+1)
        percolate_down!(xs, 1, xs[1], ord, n-j)
        j += 1
    end
    J = max(1, n-K)
    swap!(xs, 1, J)
    lo, hi = extrema((xs[J], xs[n]))
    return SparseProjection(ord, lo, hi)
end
