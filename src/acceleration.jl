# catches case without acceleration with dispatch to avoid extra branch
apply_momentum!(::Val{:none}, x, y, iter::Int, needs_rest::Bool) = 1

# checks if acceleration can be applied first
function apply_momentum!(accel, x, y, iter::Int, needs_reset::Bool)
    if needs_reset # Reset acceleration scheme
        __restart__(accel, x, y)
        iter = 1
    else # apply acceleration acceleration 
        __momentum__(accel, x, y, iter)
        iter += 1
    end
    return iter
end

############################
# dispatch for NamedTuples #
############################
__momentum__(accel, x::NamedTuple, y::NamedTuple, iter::Int) = foreach(z -> __momentum__(accel, z[1], z[2], iter), zip(x, y))
__restart__(accel, x::NamedTuple, y::NamedTuple) = foreach(z -> __restart__(accel, z[1], z[2]), zip(x, y))

###########################
# Nesterov implementation #
###########################

function __momentum__(::Val{:nesterov}, x::AbstractArray, y::AbstractArray, iter::Int, r::Int=3)
    γ = (iter - 1) / (iter + r - 1)
    @inbounds @simd for i in eachindex(x)
        xi, yi = x[i], y[i]
        zi = xi + γ * (xi - yi)
        x[i], y[i] = zi, xi
    end
end

__restart__(::Val{:nesterov}, x::AbstractArray, y::AbstractArray) = copyto!(y, x)
