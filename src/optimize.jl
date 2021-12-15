##### container for problem variables and data #####

struct ProxDistProblem{VAR,DER,OPS,BUF,VWS,LS}
    variables::VAR
    old_variables::VAR
    derivatives::DER
    operators::OPS
    buffers::BUF
    views::VWS
    linsolver::LS
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

"""
TODO
"""
function optimize!(algorithm::AlgorithmOption, prob_tuple::probT;
    nouter::Int=100,
    dtol::Real=1e-6,
    rtol::Real=1e-6,
    rho_init::Real=1.0,
    rho_max::Real=1e8,
    penalty::Function=DEFAULT_ANNEALING,
    mu_init::Real=1.0,
    verbose::Bool=false,
    # cb::Function=DEFAULT_CALLBACK,
    kwargs...) where probT
    # get objective function, iteration map, and problem object
    __objective__, __iterate__, problem = prob_tuple

    # Initialize ρ and iteration count.
    ρ, iters = rho_init, 0

    # Check initial values for loss, objective, distance, and norm of gradient.
    init_result = __objective__(algorithm, problem, ρ)
    result = SubproblemResult(0, init_result)
    old = sqrt(result.distance)

    if old ≤ dtol
        SubproblemResult(0, result.loss, result.objective, result.distance, result.gradient)
    end

    for iter in 1:nouter
        # Solve minimization problem for fixed rho; always reset mu.
        result = anneal!(algorithm, prob_tuple, ρ, mu_init; verbose=verbose, kwargs...)
        verbose && @printf "\n%s\t%4d\t%4.4e\t%4.4e\t%4.4e\t%4.4e" "OUTER" iter result.loss result.objective sqrt(result.distance) sqrt(result.gradient)

        # Update total iteration count.
        iters += result.iters

        # Check for convergence to constrained solution.
        dist = sqrt(result.distance)
        if dist ≤ dtol || abs(dist - old) ≤ rtol * (1 + old)
            break
        elseif dist > old
            @warn "Failed to decrease distance penalty at iteration $(iter). Annealing schedule may be too aggressive."
            break
        else
          old = dist
        end
                
        # Update according to annealing schedule.
        ρ = ifelse(iter < nouter, min(rho_max, penalty(ρ, iter)), ρ)
    end
    println()

    return SubproblemResult(iters, result.loss, result.objective, result.distance, result.gradient)
end


"""
Solve minimization problem with fixed ρ.
"""
function anneal!(algorithm::AlgorithmOption, prob_tuple::probT, ρ, μ;
    ninner::Int=10^4,
    gtol::Real=1e-6,
    delay::Int=10,
    verbose::Bool=false,
    history::histT=nothing,
    accel::accelT=Val(:none),
    kwargs...
    ) where {probT, histT, accelT}
    # get objective function, iteration map, and problem object
    __objective__, __iterate__, problem = prob_tuple

    # Check initial values for loss, objective, distance, and norm of gradient.
    result = __objective__(algorithm, problem, ρ)
    old = result.objective

    if sqrt(result.gradient) < gtol
        return SubproblemResult(0, result)
    end

    # Initialize iteration counts.
    if problem.variables isa AbstractArray
        copyto!(problem.variables, problem.old_variables)
    else
        foreach(_varsubset_ -> copyto!(_varsubset_[1], _varsubset_[2]), zip(problem.variables, problem.old_variables))
    end
    iters = 0
    accel_iter = 1

    for iter in 1:ninner
        iters += 1

        apply_before_algmap!(algorithm, problem, iter, ρ, μ)

        # Apply the algorithm map to minimize the quadratic surrogate.
        __iterate__(algorithm, problem, ρ, μ)

        # Update loss, objective, distance, and gradient.
        result = __objective__(algorithm, problem, ρ)
        verbose && @printf "\n\t%s\t%4d\t%4.4e\t%4.4e\t%4.4e\t%4.4e" "INNER" iter result.loss result.objective sqrt(result.distance) sqrt(result.gradient)

        # data = package_data(f_loss, h_dist, h_ngrad, stepsize, ρ)
        # update_history!(history, data, iteration)

        # Assess convergence.
        obj = result.objective
        gradsq = sqrt(result.gradient)
        if gradsq ≤ gtol
            break
        elseif iter < ninner
            needs_reset = accel_iter < delay || obj > old
            accel_iter = apply_momentum!(accel, problem.variables, problem.old_variables, accel_iter, needs_reset)
            old = obj
            apply_after_algmap!(algorithm, problem, iter, ρ, μ)
        end
    end
    if problem.variables isa AbstractArray
        copyto!(problem.old_variables, problem.variables)
    else
        foreach(_varsubset_ -> copyto!(_varsubset_[1], _varsubset_[2]), zip(problem.old_variables, problem.variables))
    end

    return SubproblemResult(iters, result)
end
