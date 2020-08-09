#################################
#   projection onto l0 ball     #
#################################

"""
Project `x` onto the l0 ball with radius `k`.

The vector `xcopy` should enter as an identical copy of `x`.
"""
function project_l0_ball!(x, xcopy, k)
    #
    #   do nothing if k ≤ 0
    #
    if k ≤ 0 return x end

    # find the spliting element
    pivot = l0_search_partialsort!(xcopy, k)

    # apply the projection
    nnz = 0
    @inbounds for i in 1:length(x)
        if abs(x[i]) ≥ abs(pivot)
            nnz += 1
            if nnz ≥ k break end
        else
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
