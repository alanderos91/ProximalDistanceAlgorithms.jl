# no acceleration
get_acceleration_strategy(::Val{:none}, X) = nothing
apply_momentum!(X, strategy::Nothing) = X
restart!(::Nothing, X) = nothing

# Nesterov acceleration
mutable struct Nesterov{T}
    Y::T
    n::Int

    # type can be a vector, matrix, or named tuple
    Nesterov(X::T) where T = new{T}(deepcopy(X), 1)
end

get_acceleration_strategy(::Val{:nesterov}, X) = Nesterov(X)

# dispatch for array types
function apply_momentum!(X::AbstractArray, strategy::Nesterov)
    n = strategy.n + 1
    strategy.n = n

    __apply_momentum!(X, strategy.Y, n)

    return X
end

# dispatch for named tuples
function apply_momentum!(optvars::NamedTuple, strategy::Nesterov)
    n = strategy.n + 1
    strategy.n = n

    for (X, Y) in zip(optvars, strategy.Y)
        __apply_momentum!(X, Y, n)
    end

    return optvars
end

# generic implementation for array types
function __apply_momentum!(X::AbstractArray, Y, n)
    for idx in eachindex(X)
        x = X[idx]
        y = Y[idx]

        z = x + (n-1)/(n+2) * (x-y)

        Y[idx] = x
        X[idx] = z
    end

    return nothing
end

# implementation for lower triangular matrix
function __apply_momentum!(X::LowerTriangular, Y, n)
    m1, m2 = size(X)

    for j in 1:m2, i in j+1:m1
        x = X[i,j]
        y = Y[i,j]

        z = x + (n-1)/(n+2) * (x-y)

        Y[i,j] = x
        X[i,j] = z
    end

    return nothing
end

# dispatch for arrays
function restart!(strategy::Nesterov, X::AbstractArray)
    copyto!(strategy.Y, X)
    strategy.n = 1

    return nothing
end

# dispatch for named tuples
function restart!(strategy::Nesterov, optvars::NamedTuple)
    for (X, Y) in zip(optvars, strategy.Y)
        copyto!(Y, X)
    end

    strategy.n = 1

    return nothing
end
