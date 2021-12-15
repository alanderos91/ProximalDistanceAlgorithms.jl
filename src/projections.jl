#################################
#   projection onto l0 ball     #
#################################

"""
Project `x` onto sparsity set with `k` non-zero elements.
Assumes `idx` enters as a vector of indices into `x`.
"""
function project_l0_ball!(x, idx, k, buffer)
    n = length(x)
    # do nothing if k > length(x)
    if k ≥ n return x end
    
    # fill with zeros if k ≤ 0
    if k ≤ 0 return fill!(x, 0) end
    
    # otherwise, find the spliting element
    search_by_top_k = k < n-k+1
    if search_by_top_k
        _k = k
        pivot = l0_search_partialsort!(idx, x, _k, true)
    else
        _k = n-k+1
        pivot = l0_search_partialsort!(idx, x, _k, false)
    end
    
    # preserve the top k elements
    p = abs(pivot)
    nonzero_count = 0
    for i in eachindex(x)
        if x[i] == 0 continue end
        if abs(x[i]) < p
            x[i] = 0
        else
            nonzero_count += 1
        end
    end

    # resolve ties
    if nonzero_count > k
        number_to_drop = nonzero_count - k
        _buffer_ = view(buffer, 1:number_to_drop)
        _indexes_ = findall(!iszero, x)
        sample!(_indexes_, _buffer_, replace=false)
        for i in _buffer_
            x[i] = 0
        end
    end

    return x
end

"""
Search `x` for the pivot that splits the vector into the `k`-largest elements in magnitude.

The search preserves signs and returns `x[k]` after partially sorting `x`.
"""
function l0_search_partialsort!(idx, x, k, rev::Bool)
    #
    # Based on https://github.com/JuliaLang/julia/blob/788b2c77c10c2160f4794a4d4b6b81a95a90940c/base/sort.jl#L863
    # This eliminates a mysterious allocation of ~48 bytes per call for
    #   sortperm!(idx, x, alg=algorithm, lt=isless, by=abs, rev=true, initialized=false)
    # where algorithm = PartialQuickSort(lo:hi)
    # Savings are small in terms of performance but add up for CV code.
    #
    lo = k
    hi = k+1

    # Order arguments
    lt  = isless
    by  = abs
    # rev = true
    o = Base.Order.Forward
    order = Base.Order.Perm(Base.Sort.ord(lt, by, rev, o), x)

    # sort!(idx, lo, hi, PartialQuickSort(k), order)
    Base.Sort.Float.fpsort!(idx, PartialQuickSort(lo:hi), order)

    return x[idx[k]]
end

# type to store extra arrays used in projection
struct L0Projection <: Function
    k::Int
    idx::Vector{Int}
    buffer::Vector{Int}
end

function (P::L0Projection)(x)
    project_l0_ball!(x, P.idx, P.k, P.buffer)
end

function project_l0_ball!(X::AbstractMatrix, idx, scores, k, buffer; by::Union{Val{:row}, Val{:col}}=Val(:row))
    # determine structure of sparsity
    if by isa Val{:row}
        n = size(X, 1)
        itr = axes(X, 1)
        itr2 = eachrow(X)
        f = i -> norm(view(X, i, :))
    elseif by isa Val{:col}
        n = size(X, 2)
        itr = axes(X, 2)
        itr2 = eachcol(X)
        f = i -> norm(view(X, :, i))
    else
        error("uncrecognized option `by=$(by)`.")
    end

    # do nothing if k > length(x)
    if k ≥ n return X end

    # fill with zeros if k ≤ 0
    if k ≤ 0 return fill!(X, 0) end

    # otherwise, map rows to a score used in ranking and find the spliting element
    map!(f, scores, itr)
    search_by_top_k = k < n-k+1
    if search_by_top_k
        _k = k
        pivot = l0_search_partialsort!(idx, scores, _k, true)
    else
        _k = n-k+1
        pivot = l0_search_partialsort!(idx, scores, _k, false)
    end

    # preserve the top k elements
    p = abs(pivot)
    nonzero_count = 0
    for (i, xᵢ) in enumerate(itr2)
        if scores[i] == 0 continue end

        # row is not in the top k
        if scores[i] < p
            fill!(xᵢ, 0)
            scores[i] = 0
        else # row is in the top k
            nonzero_count += 1
        end
    end
    
    # resolve ties
    if nonzero_count > k
        number_to_drop = nonzero_count - k
        _buffer_ = view(buffer, 1:number_to_drop)
        _indexes_ = findall(!iszero, scores)
        sample!(_indexes_, _buffer_, replace=false)
        for i in _buffer_
            if by isa Val{:row}
                fill!(view(X, i, :), 0)
            elseif by isa Val{:col}
                fill!(view(X, :, i), 0)
            end
        end
    end

    return X
end

struct StructuredL0Projection{KIND} <: Function
    k::Int
    idx::Vector{Int}
    buffer::Vector{Int}
    scores::Vector{Float64}
    kind::KIND
end

function (P::StructuredL0Projection)(X::AbstractMatrix)
    project_l0_ball!(X, P.idx, P.scores, P.k, P.buffer, by=P.kind)
end

ColumnL0Projection(k, idx, buffer, scores) = StructuredL0Projection(k, idx, buffer, scores, Val(:col))
RowL0Projection(k, idx, buffer, scores) = StructuredL0Projection(k, idx, buffer, scores, Val(:row))

#################################
#   projection onto l1 ball     #
#################################

function project_l1_ball!(y, ytmp, a, algorithm::Function=condat_algorithm2)
    #
    #   project y to the non-negative orthant
    #
    @. ytmp = abs(y)

    #
    #   compute the splitting element τ needed in KKT conditions
    #
    τ = algorithm(ytmp, a)

    #
    #   complete the projection:
    #   x = P_simplex(|y|)
    #   z = P_a(y) by sgn(y_i) * x_i
    #
    @inbounds @simd for k in eachindex(y)
        y[k] = sign(y[k]) * max(0, abs(y[k]) - τ)
    end

    return y
end

"Project `y` onto l1 ball using Algorithm 1 from Condat."
project_l1_ball1!(y, ytmp, a) = project_l1_ball!(y, ytmp, a, condat_algorithm1!)

"Project `y` onto l1 ball using Algorithm 2 from Condat."
project_l1_ball2!(y, ytmp, a) = project_l1_ball!(y, ytmp, a, condat_algorithm2!)

"""
Find the splitting element `τ` which determines the projection of `y-a` to the unit simplex.

This is version is based on quicksort, as described in

Laurent Condat.  Fast Projection onto the Simplex and the l1 Ball.
Mathematical Programming, Series A, Springer, 2016, 158 (1), pp.575-585.10.1007/s10107-015-0946-6. hal-01056171v2
"""
function condat_algorithm1!(y, a)
    # sort elements in descending order using quicksort
    sort!(y, rev=true)

    # initialization
    n = length(y)
    j = 1           # cardinality of the index set, I
    τ = y[j] - a    # initialize the splitting element, τ

    # add the largest elements until τ satisfies the splitting condition:
    #   y[i] - τ <= 0, for i not in the index set I, and
    #   y[i] - τ > 0,  for i in the index set I
    while j < n && (y[j] ≤ τ || y[j+1] > τ)
        # update τ and the cardinality of I
        τ = (j*τ + y[j+1]) / (j+1)
        j += 1
    end

    return τ
end

#
#   use functions from DataStructures.jl to implement array as binary maxheap
#
function maxheap!(xs)
    DataStructures.heapify!(xs, Base.Order.Reverse)
    return nothing
end

function push_down_max!(xs, len, root)
    DataStructures.percolate_down!(xs, root, Base.Order.Reverse, len)
    return nothing
end

function swap!(xs, i, j)
    xs[i], xs[j] = xs[j], xs[i]
    return nothing
end

"""
Find the splitting element `τ` which determines the projection of `y-a` to the unit simplex.

This version uses a heap, as described in

Laurent Condat.  Fast Projection onto the Simplex and the l1 Ball.
Mathematical Programming, Series A, Springer, 2016, 158 (1), pp.575-585.10.1007/s10107-015-0946-6. hal-01056171v2
"""
function condat_algorithm2!(y, a)
    # build a max-order heap
    maxheap!(y)

    # initialization
    n = length(y)
    j = 1           # cardinality of the index set, I
    τ = y[j] - a    # initialize the splitting element, τ

    swap!(y, 1, n)              # put the largest element at the end
    push_down_max!(y, n-j, 1)   # and use entries 1:n-j for the next heap
    k = n                       # index of the previous largest element

    # add the largest elements until τ satisfies the splitting condition:
    #   y[i] - τ <= 0, for i not in the index set I, and
    #   y[i] - τ > 0,  for i in the index set I
    # here, y[k] is the previous element and y[1] is the next largest element
    while j < n && (y[k] ≤ τ || y[1] > τ)
        # update τ and the cardinality of I
        τ = (j*τ + y[1]) / (j+1)
        j += 1
        k -= 1
        swap!(y, 1, k)
        push_down_max!(y, n-j, 1)
    end

    return τ
end

struct L1Projection{T} <: Function
    radius::T
    xtmp::Vector{T}
end

function (P::L1Projection)(x)
    project_l1_ball2!(x, P.xtmp, P.radius)
    return x
end
