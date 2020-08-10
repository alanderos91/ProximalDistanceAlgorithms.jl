#################################
#   projection onto l0 ball     #
#################################

"""
Project `x` onto the l0 ball with radius `k`.

The vector `xcopy` should enter as an identical copy of `x`.
"""
function project_l0_ball!(x, xcopy, k)
    #
    #   do nothing if k > length(x)
    #
    if k > length(x) return x end

    #
    #   fill with zeros if k ≤ 0
    #
    if k ≤ 0 return fill!(x, 0) end

    #
    # find the spliting element
    #
    pivot = l0_search_partialsort!(xcopy, k)

    #
    # apply the projection
    #
    @inbounds for i in 1:length(x)
        if abs(x[i]) < abs(pivot)
            x[i] = 0
        end
    end

    return x
end

"""
Search `x` for the pivot that splits the vector into the `k`-largest elements in magnitude.

The search preserves signs and returns `x[k]` after partially sorting `x`.
"""
function l0_search_partialsort!(x, k)
    lo = k
    hi = k+1
    algorithm = Base.Sort.PartialQuickSort(lo:hi)
    order = Base.Sort.ord(isless,abs,true)
    sort!(x, 1, length(x), algorithm, order)
    # partialsort!(x, k, by=abs, rev=true)

    return x[k]
end

struct L0Projection{T} <: Function
    nu::Int
    xtmp::Vector{T}
end

function (P::L0Projection)(dest, src)
    copyto!(dest, src)
    copyto!(P.xtmp, src)
    project_l0_ball!(dest, P.xtmp, P.nu)

    return dest
end

struct L0ColProjection{T} <: Function
    nu::Int
    ncols::Int
    colsz::Int
    colnorm::Vector{T}
    buffer::Vector{T}
end

function (P::L0ColProjection)(dest, src)
    @unpack nu, ncols, colsz, colnorm, buffer = P

    # compute column norms
    for k in 1:ncols
        # extract the corresponding column
        start = colsz * (k-1) + 1
        stop  = start + colsz - 1
        col   = @view src[start:stop]

        # compute norm for the column and save to vectors
        colnorm_k = norm(col)
        colnorm[k] = colnorm_k
        buffer[k]  = colnorm_k
    end

    # project column norms to l0 ball
    project_l0_ball!(colnorm, buffer, nu)

    # keep columns with nonzero norm
    for k in 1:ncols
        start = colsz * (k-1) + 1
        stop  = start + colsz - 1

        if colnorm[k] > 0
            for i in start:stop
                dest[i] = src[i]
            end
        else
            for i in start:stop
                dest[i] = 0
            end
        end
    end

    return dest
end

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
    radius::Float64
    xtmp::Vector{T}
end

function (P::L1Projection)(dest, src)
    copyto!(dest, src)
    copyto!(P.xtmp, src)
    project_l1_ball2!(dest, P.xtmp, P.radius)

    return dest
end
