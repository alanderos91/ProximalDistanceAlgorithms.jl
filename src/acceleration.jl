# no acceleration
get_accelerator(::Val{:none}, vars) = nothing
apply_momentum!(::Nothing, vars) = vars
restart!(::Nothing, vars) = nothing

# Nesterov acceleration
mutable struct Nesterov{T}
    oldvars::T
    n::Int

    # type can be a vector, matrix, or named tuple
    Nesterov(vars::T) where T = new{T}(deepcopy(vars), 1)
end

get_accelerator(::Val{:nesterov}, optvars) = Nesterov(optvars)

# dispatch for array types
function apply_momentum!(accelerator::Nesterov, optvars::AbstractArray)
    n = accelerator.n + 1
    accelerator.n = n
    __apply_momentum!(optvars, accelerator.oldvars, n)
    return optvars
end

# dispatch for named tuples
function apply_momentum!(accelerator::Nesterov, optvars::NamedTuple)
    n = accelerator.n + 1
    accelerator.n = n
    for (x, y) in zip(optvars, accelerator.oldvars)
        __apply_momentum!(x, y, n)
    end
    return optvars
end

# generic implementation for array types
function __apply_momentum!(x::AbstractArray, y::AbstractArray, n)
    @inbounds for i in eachindex(x)
        xi = x[i]
        yi = y[i]
        zi = xi + (n-1)/(n+2) * (xi-yi)
        y[i] = xi
        x[i] = zi
    end

    return nothing
end

# dispatch for arrays
function restart!(accelerator::Nesterov, optvars::AbstractArray)
    copyto!(accelerator.oldvars, optvars)
    # accelerator.n = 1
    return nothing
end

# dispatch for named tuples
function restart!(accelerator::Nesterov, optvars::NamedTuple)
    for (x, y) in zip(optvars, accelerator.oldvars)
        copyto!(y, x)
    end
    # accelerator.n = 1
    return nothing
end
