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

@doc raw"""
Solve a distance penalized problem along an annealing path using a specific `algorithm`.

The `prob_tuple` should enter as `(objective, iteration_scheme, problem)`, where

- `objective` is a function to evaluate the $\rho$-penalized objective for the problem,
- `iteration_scheme` is a function to generate the next iterate for the given `algorithm`, and
- `problem` is a type storing data, parameter estimates, and other data structures.

**Note**: This function may exit early if minimizing a subproblem fails to decrease the distance penalty.

# Keyword Arguments

- `nouter`: The maximum number of subproblems to solve along the annealing path. (default=`100`).
- `dtol`: A control parameter applied $\mathrm{dist}(\mathbf{D} \mathbf{x}, S)$ that determines proximity of solutions to the constraint set. (default=`1e-1`)
- `rtol`: A control parameter for early exit when relative progress in minimizing the distance penalty slows to a crawl. Specifically, we check $|f(x_{n+1}) - f(x_{n})| \le \delta_{r} [1 + f(x_{n})]$ with $\delta_{r}$ = `rtol`.
- `rho_init`: Initial value for $\rho$. (default=`1e-6`)
- `rho_max`: Maximum value for $\rho$. (default=`1.0`)
- `penalty`: A function used to update $\rho_{t+1}$ from $\rho_{t}$ or $t$. (default=`geometric_progression`)
- `mu_init`: Initial value for $μ$ parameter used in ADMM. (default=`1.0`)
- `callback`: A function to invoke after minimizing a particular subproblem. See [`print_convergence_history`](@ref) for an example.

See also [`anneal!`](@ref) for additional arguments specific to solving a subproblem along the annealing path.
"""
function optimize!(algorithm::AlgorithmOption, prob_tuple::probT;
    nouter::Int=100,
    dtol::Real=1e-1,
    rtol::Real=1e-6,
    rho_init::Real=1.0,
    rho_max::Real=1e8,
    penalty::Function=DEFAULT_ANNEALING,
    mu_init::Real=1.0,
    callback::cbT=DEFAULT_CALLBACK,
    kwargs...) where {probT, cbT}
    # get objective function, iteration map, and problem object
    __objective__, __iterate__, problem = prob_tuple

    # Initialize ρ and iteration count.
    ρ, iters = rho_init, 0

    # Check initial values for loss, objective, distance, and norm of gradient.
    init_result = __objective__(algorithm, problem, ρ)
    result = SubproblemResult(0, init_result)
    old = sqrt(result.distance)
    callback(Val(:outer), algorithm, 0, result, problem, ρ, mu_init)

    if old ≤ dtol
        SubproblemResult(0, result.loss, result.objective, result.distance, result.gradient)
    end

    for iter in 1:nouter
        # Solve minimization problem for fixed rho; always reset mu.
        result = anneal!(algorithm, prob_tuple, ρ, mu_init; callback=callback, kwargs...)
        callback(Val(:outer), algorithm, iter, result, problem, ρ, mu_init)

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

    return SubproblemResult(iters, result.loss, result.objective, result.distance, result.gradient)
end


@doc raw"""
Solve a given minimization problem with fixed $\rho$ using a specific `algorithm`.

The `prob_tuple` should enter as `(objective, iteration_scheme, problem)`, where

- `objective` is a function to evaluate the $\rho$-penalized objective for the problem,
- `iteration_scheme` is a function to generate the next iterate for the given `algorithm`, and
- `problem` is a type storing data, parameter estimates, and other data structures.

# Keyword Arguments

- `ninner`: The maximum number of iterations to minimize the $\rho$-penalized objective. (default=`10^4`)
- `gtol`: A control parameter on the scale of the gradient used to assess convergence. Specifically, we run the given `iteration_scheme` until ``\|\nabla h_{\rho}(x)\| \le \delta_{g}`` with $\delta_{g}$ = `gtol`. (default=`1e-2`).
- `delay`: A fixed number of iterations to delay application of momentum to accelerate convergence (e.g. Nesterov acceleration). (default=`10`)
- `accel`: An option for choice of acceleration technique. Choices are `Val(:none)` for no acceleration and `Val(:nesterov)` for Nesterov acceleration. (default=`Val(:none)`)
- `callback`: A function to invoke after each update. See [`print_convergence_history`](@ref) for an example.
"""
function anneal!(algorithm::AlgorithmOption, prob_tuple::probT, ρ, μ;
    ninner::Int=10^4,
    gtol::Real=1e-2,
    delay::Int=10,
    callback::cbT=DEFAULT_CALLBACK,
    accel::accelT=Val(:none),) where {probT, cbT, accelT}
    # get objective function, iteration map, and problem object
    __objective__, __iterate__, problem = prob_tuple

    # Check initial values for loss, objective, distance, and norm of gradient.
    result = __objective__(algorithm, problem, ρ)
    callback(Val(:inner), algorithm, 0, result, problem, ρ, μ)
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
        callback(Val(:inner), algorithm, iter, result, problem, ρ, μ)

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
