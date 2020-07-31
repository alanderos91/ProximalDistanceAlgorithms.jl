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
function update_history!(history::NamedTuple, data, iteration)
    if iteration % history.sample_rate == 0
        push!(history.loss, data.loss)
        push!(history.distance, data.distance)
        push!(history.objective, data.objective)
        push!(history.gradient, data.gradient)
        push!(history.stepsize, data.stepsize)
        push!(history.rho, data.rho)
        push!(history.iteration, iteration)
    end

    return nothing
end

##### linear operators #####

abstract type FusionMatrix{T} <: LinearMap{T} end

# LinearAlgebra traits
LinearAlgebra.issymmetric(D::FusionMatrix) = false
LinearAlgebra.ishermitian(D::FusionMatrix) = false
LinearAlgebra.isposdef(D::FusionMatrix)    = false

# internal API

instantiate_fusion_matrix(D::FusionMatrix) = error("not implemented for $(typeof(D))")

abstract type FusionGramMatrix{T} <: LinearMap{T} end

# LinearAlgebra traits
LinearAlgebra.issymmetric(DtD::FusionGramMatrix) = true
LinearAlgebra.ishermitian(DtD::FusionGramMatrix) = true
LinearAlgebra.isposdef(DtD::FusionGramMatrix)    = false

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

struct SparseProjection{T}
    ismaxparam::Bool
    pivot::T
end

function (P::SparseProjection)(x)
    if P.ismaxparam
        x ≥ P.pivot ? x : zero(x)
    else
        x > P.pivot ? x : zero(x)
    end
end

struct SparseProjectionClosure
    ismaxparam::Bool
    ν::Int
end

function (C::SparseProjectionClosure)(xs)
    if C.ν > 0
        # pivot = C.ν < length(xs) ? partialsort!(xs, C.ν, rev = C.ismaxparam) : zero(eltype(xs))
        # quantile based algorithm
        # see: https://github.com/JuliaLang/Statistics.jl/blob/master/src/Statistics.j

        # sorting step: _quantilesort!
        n = length(xs)
        p = C.ismaxparam ? 1 - C.ν / n : C.ν / n
        lo = floor(Int, p*n)
        hi = ceil(Int, 1+p*n)
        sort!(xs, 1, n, Base.Sort.PartialQuickSort(lo:hi), Base.Sort.Forward)

        # quantile lookup: _quantile!
        alpha = 1.0
        beta = alpha

        m = alpha + p * (one(alpha) - alpha - beta)
        aleph = n*p + oftype(p, m)
        j = clamp(trunc(Int, aleph), 1, n-1)
        γ = clamp(aleph - j, 0, 1)
    
        if n == 1
            a = xs[1]
            b = xs[1]
        else
            a = xs[j]
            b = xs[j+1]
        end
        
        if isfinite(a) && isfinite(b)
            pivot = a + γ*(b-a)
        else
            pivot = (1-γ)*a + γ*b
        end
    else
        pivot = -Inf
    end
    return SparseProjection(C.ismaxparam, pivot)
end

struct BlockSparseProjection{T} <: Function
    block_size::Int
    block_norm::Vector{T}
    cache::Vector{T}
    compute_proj::SparseProjectionClosure
end

function (P::BlockSparseProjection)(y, x)
    block_size = P.block_size
    block_norm = P.block_norm
    cache = P.cache
    compute_proj = P.compute_proj

    # compute distances from x
    for j in eachindex(block_norm)
        offset = block_size*(j-1)
        s = zero(eltype(x))

        for k in 1:block_size
            s += x[offset+k]*x[offset+k]
        end
        block_norm[j] = s
        cache[j] = s
    end

    # compute projection operator
    proj = compute_proj(cache)

    # apply the projection
    for (j, s) in enumerate(block_norm)
        offset = block_size*(j-1)
        indicator = (proj(s) == s)

        for k in 1:block_size
            y[offset+k] = indicator * x[offset+k]
        end
    end

    return y
end

(P::BlockSparseProjection)(x) = P(similar(x), x)
