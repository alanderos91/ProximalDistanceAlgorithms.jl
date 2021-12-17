struct IterationResult
    loss::Float64
    objective::Float64
    distance::Float64
    gradient::Float64
end

# destructuring
Base.iterate(r::IterationResult) = (r.loss, Val(:objective))
Base.iterate(r::IterationResult, ::Val{:objective}) = (r.objective, Val(:distance))
Base.iterate(r::IterationResult, ::Val{:distance}) = (r.distance, Val(:gradient))
Base.iterate(r::IterationResult, ::Val{:gradient}) = (r.gradient, Val(:done))
Base.iterate(r::IterationResult, ::Val{:done}) = nothing

struct SubproblemResult
    iters::Int
    loss::Float64
    objective::Float64
    distance::Float64
    gradient::Float64
end

function SubproblemResult(iters, r::IterationResult)
    return SubproblemResult(iters, r.loss, r.objective, r.distance, r.gradient)
end

# destructuring
Base.iterate(r::SubproblemResult) = (r.iters, Val(:loss))
Base.iterate(r::SubproblemResult, ::Val{:loss}) = (r.loss, Val(:objective))
Base.iterate(r::SubproblemResult, ::Val{:objective}) = (r.objective, Val(:distance))
Base.iterate(r::SubproblemResult, ::Val{:distance}) = (r.distance, Val(:gradient))
Base.iterate(r::SubproblemResult, ::Val{:gradient}) = (r.gradient, Val(:done))
Base.iterate(r::SubproblemResult, ::Val{:done}) = nothing

##### annealing schedules #####

geometric_schedule(ρ::Real, n::Int, a::Real=1.2) = a*ρ
const DEFAULT_ANNEALING = geometric_schedule

##### convergence history #####

# notes:
# report 'distance' as dist(Dx,S)
# report 'objective' as 0.5 * (loss + rho * distance)
# report 'gradient' as norm(gradient)

function initialize_history(hint::Integer, sample_rate::Integer=1)
    history = (
        sample_rate = sample_rate,
        loss      = sizehint!(Float64[], hint),
        distance  = sizehint!(Float64[], hint),
        objective = sizehint!(Float64[], hint),
        gradient  = sizehint!(Float64[], hint),
        # stepsize  = sizehint!(Float64[], hint),
        rho       = sizehint!(Float64[], hint),
        iteration = sizehint!(Int[], hint),
    )

    # create a closure to pass into program
    logger = function(kind, algorithm, iter, result, problem, ρ, μ)
        update_history!(history, kind, algorithm, iter, result, problem, ρ, μ)
    end

    return history, logger
end

# implementation: object with named fields
function update_history!(history::NamedTuple, ::Val{:outer}, algorithm, iter, result, problem, ρ, μ)
    iter > 0 && push!(history.rho, ρ)
    return nothing
end

function update_history!(history::NamedTuple, ::Val{:inner}, algorithm, iter, result, problem, ρ, μ)
    if iter % history.sample_rate == 0
        push!(history.loss, result.loss)
        push!(history.distance, sqrt(result.distance))
        push!(history.objective, result.objective)
        push!(history.gradient, sqrt(result.gradient))
        # push!(history.stepsize, data.stepsize)
        push!(history.iteration, iter)
    end

    return nothing
end

##### printing convergence history #####

function initialize_printing_logger(or::Int, ir::Int)
    logger = function(kind, algorithm, iter, result, problem, ρ, μ)
        print_convergence_history(or, ir, kind, algorithm, iter, result, problem, ρ, μ)
    end

    return logger
end

function print_convergence_history(or::Int, ir::Int, ::Val{:outer}, algorithm, iter, result, problem, ρ, μ)
    if iter == 0
        println("\n\n     \tITER\tLOSS\t\tOBJECTIVE\tDISTANCE\tGRADIENT")
    end
    if iter % or == 0
        @printf "\n%s\t%4d\t%4.4e\t%4.4e\t%4.4e\t%4.4e" "OUTER" iter result.loss result.objective sqrt(result.distance) sqrt(result.gradient)
    end
    return nothing
end

function print_convergence_history(or::Int, ir::Int, ::Val{:inner}, algorithm, iter, result, problem, ρ, μ)
    if iter % ir == 0
        @printf "\n%s\t%4d\t%4.4e\t%4.4e\t%4.4e\t%4.4e" "INNER" iter result.loss result.objective sqrt(result.distance) sqrt(result.gradient)
    end
    return nothing
end

print_convergence_history(or::Int, ir::Int, ::Val{:path}, algorithm, iter, result, problem, ρ, μ) = nothing

##### default callbacks #####
__do_nothing_callback__(kind, algorithm, iter, result, problem, ρ, μ) = nothing

function print_convergence_history(kind, algorithm, iter, result, problem, ρ, μ)
    print_convergence_history(1, 100, kind, algorithm, iter, result, problem, ρ, μ)
end

const DEFAULT_CALLBACK = __do_nothing_callback__
const DEFAULT_LOGGING = print_convergence_history

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
