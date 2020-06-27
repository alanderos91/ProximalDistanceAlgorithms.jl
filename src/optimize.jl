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
        if ρ != ρ_new
            restart!(strategy, optvars)
            operators, buffers = remake_operators(algorithm, operators, buffers, ρ_new)
        end
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
