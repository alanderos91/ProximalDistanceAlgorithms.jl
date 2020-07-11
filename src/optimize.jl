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

##### for computing adaptive stepsize in ADMM #####

function update_admm_residuals!(prob, μ)
    @unpack y = prob.variables
    @unpack D = prob.operators
    @unpack z, r, s, y_prev = prob.buffers

    @. r = z - y # assumes z = D*x
    @. y_prev = μ * (y_prev - y)
    mul!(s, D', y_prev)

    r_error = sqrt(BLAS.nrm2(length(r), r, 1))
    s_error = sqrt(BLAS.nrm2(length(s), s, 1))

    return r_error, s_error
end

##### for updating subspace in MMSubSpace #####

function update_subspace!(prob, iteration)
    @unpack ∇h, G = prob.derivatives

    # determine column to update
    N, K = size(G)
    modulus = iteration % K
    j = modulus > 0 ? modulus : K

    # update column j
    start = N*(j-1)+1
    copyto!(G, start, ∇h, 1, N)

    return nothing
end

##### before algorithm map #####

apply_before_algmap!(::AlgorithmOption, prob, iteration, ρ, μ) = nothing

function apply_before_algmap!(::ADMM, prob, iteration, ρ, μ)
    @unpack y = prob.variables
    @unpack y_prev = prob.buffers
    copyto!(y_prev, y)
    return nothing
end

function apply_before_algmap!(::MMSubSpace, prob, iteration, ρ, μ)
    update_subspace!(prob, iteration)
    return nothing
end

##### after algorithm map #####

apply_after_algmap!(::AlgorithmOption, prob, iteration, ρ, μ) = ρ, μ

function apply_after_algmap!(::ADMM, prob, iteration, ρ, μ)
    r_error, s_error = update_admm_residuals!(prob, μ)

    if (r_error / s_error) > 10   # emphasize dual feasibility
        μ = μ * 2
    elseif (s_error / r_error) > 10   # emphasize primal feasibility
        μ = μ / 2
    end

    return ρ, μ
end

##### common solution interface #####

@noinline function optimize!(algorithm::AlgorithmOption, objective, algmap, prob, ρ, μ;
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
        apply_before_algmap!(algorithm, prob, iteration, ρ, μ)

        # apply algorithm map
        stepsize = algmap(algorithm, prob, ρ, μ)

        # update penalty and momentum
        ρ_new = penalty(ρ, iteration)
        if ρ != ρ_new
            restart!(accelerator, prob.variables)
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

        ρ, μ = apply_after_algmap!(algorithm, prob, iteration, ρ, μ)
    end

    converged = !not_converged

    return prob, iteration, converged
end
