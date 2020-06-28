##### container for problem variables and data #####

struct ProxDistProblem{VAR,DER,OPS,BUF,VWS,LS}
    variables::VAR
    derivatives::DER
    operators::OPS
    buffers::BUF
    views::VWS
    linsolver::LS
end

##### checking convergence #####

"""
Evaluate convergence using the following three checks:

    1. relative change in `loss` is within `rtol`,
    2. relative change in `dist` is within `rtol`, and
    3. magnitude of `dist` is smaller than `atol`

Returns `true` if any of (1)-(3) are violated, `false` otherwise.
"""
function check_convergence(loss, dist, rtol, atol)
    diff1 = abs(loss.new - loss.old)
    diff2 = abs(dist.new - dist.old)

    flag1 = diff1 > rtol * (loss.old + 1)
    flag2 = diff2 > rtol * (dist.old + 1)
    flag3 = dist.new > atol

    return flag1 || flag2 || flag3
end

##### remaking immutables #####

remake_hessian(H::ProxDistHessian, ρ) = ProxDistHessian(H.N, ρ, H.∇²f, H.DtD)

remake_operators(::AlgorithmOption, prob, ρ) = prob

function remake_operators(::MM, prob::ProxDistProblem{A,B,C,D,E,F}, ρ) where {A,B,C,D,E,F<:CGIterable}
    @unpack variables, derivatives, operators, buffers, views, linsolver = prob
    @unpack x, r, c, u, reltol, residual, prev_residual, maxiter, mv_products = linsolver

    H = remake_hessian(operators.H, ρ)
    operators = (operators..., H = H)
    linsolver = CGIterable(H, x, r, c, u, reltol, residual, prev_residual, maxiter, mv_products)

    return ProxDistProblem(variables, derivatives, operators, buffers, views, linsolver)
end

##### common solution interface #####

function optimize!(algorithm::AlgorithmOption, objective, algmap, prob, ρ, μ;
    maxiters::Integer = 100,
    penalty::Function = __default_schedule,
    history::histT    = nothing,
    rtol::Real        = 1e-6,
    atol::Real        = 1e-4,
    accel::accelT     = Val(:none)) where {histT, accelT}
    #
    # select acceleration algorithm
    accelerator = get_accelerator(accel, prob.variables)

    # initialize algorithm
    f_loss, h_dist, h_ngrad = objective(algorithm, prob, ρ)
    data = package_data(f_loss, h_dist, h_ngrad, one(f_loss), ρ)
    update_history!(history, data, 0)

    loss = (old = f_loss, new = Inf)
    dist = (old = h_dist, new = Inf)
    not_converged = check_convergence(loss, dist, rtol, atol)
    iteration = 1

    # main optimization loop
    while not_converged && iteration ≤ maxiters
        # apply algorithm map
        stepsize = algmap(algorithm, prob, ρ, μ)

        # update penalty and momentum
        ρ_new = penalty(ρ, iteration)
        if ρ != ρ_new
            restart!(accelerator, prob.variables)
            prob = remake_operators(algorithm, prob, ρ_new)
        end
        apply_momentum!(accelerator, prob.variables)
        ρ = ρ_new

        # convergence history
        f_loss, h_dist, h_ngrad = objective(algorithm, prob, ρ)
        data = package_data(f_loss, h_dist, h_ngrad, stepsize, ρ)
        update_history!(history, data, iteration)

        loss = (old = loss.new, new = f_loss)
        dist = (old = dist.new, new = h_dist)
        not_converged = check_convergence(loss, dist, rtol, atol)
        iteration += 1
    end

    converged = !not_converged

    return prob, iteration, converged
end
