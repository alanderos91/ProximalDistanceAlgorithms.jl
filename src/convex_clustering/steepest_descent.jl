function cvxclst_iter(::SteepestDescent, optvars, derivs, operators, buffers, ρ)
    u = optvars.u
    ∇h = derivs.∇h
    D = operators.D
    z = buffers.z

    # evaluate stepsize
    mul!(z, D, ∇h)
    a = dot(∇h, ∇h)     # ||∇h(x)||^2
    b = dot(z, z)       # ||D*∇h(x)||^2
    γ = a / (a + ρ*b + eps())

    # move in the direction of steepest descent
    @. u = u - γ*∇h

    return γ
end

function convex_clustering(algorithm::SteepestDescent, W, X;
    K::Integer = 0,
    o::Base.Ordering = Base.Order.Forward, kwargs...)
    #
    # extract problem dimensions
    d, n = size(X)      # d features by n samples
    m = binomial(n, 2)  # number of constraints
    M = d*m             # number of rows
    N = d*n             # number of columns

    # allocate optimization variable
    U = zero(X)
    u = vec(U)
    optvars = (u = u,)

    # allocate derivatives
    ∇f = zeros(N)
    ∇d = zeros(N)
    ∇h = zeros(N)
    derivs = (∇f = ∇f, ∇d = ∇d, ∇h = ∇h)

    # generate operators
    D = CvxClusterFM(d, n)
    x = vec(X)
    operators = (D = D, x = x, o = o, K = K, compute_projection = compute_sparse_projection)

    # allocate any additional arrays for mat-vec multiplication
    z = zeros(M)
    Pz = zeros(M)
    ds = zeros(m)
    ss = similar(ds)
    buffers = (z = z, U = U, Pz = Pz, ds = ds, ss = ss)

    optimize!(algorithm, cvxclst_eval, cvxclst_iter, optvars, derivs, operators, buffers; kwargs...)

    return U
end
#
# # for data structure re-use in convex_clustering_path
# function convex_clustering!(solution, projection, inputs, settings, K, trace)
#     # centroids and gradient
#     U = solution.U
#     Q = solution.Q
#
#     # data structures for projection
#     Y     = projection.Y
#     Δ     = projection.Δ
#     index = projection.index
#
#     # input data
#     W       = inputs.W
#     X       = inputs.X
#     ρ_init  = inputs.ρ_init
#     penalty = inputs.penalty
#
#     # settings
#     maxiters = settings.maxiters
#     history  = settings.history
#     ftol     = settings.ftol
#     dtol     = settings.dtol
#     strategy = settings.strategy
#
#     # initialize
#     ρ = ρ_init
#
#     loss, distance, gradient = cvxclst_evaluate!(Q, Y, Δ, index, W, U, X, K, ρ)
#     data = package_data(loss, distance, ρ, gradient, zero(loss))
#
#     if trace # only record outside clustering path algorithm
#         update_history!(history, data, 0)
#     end
#
#     loss_old = loss
#     loss_new = Inf
#     dist_old = distance
#     dist_new = Inf
#     iteration = 1
#
#     while not_converged(loss_old, loss_new, dist_old, dist_new, ftol, dtol) && iteration ≤ maxiters
#         # iterate the algorithm map
#         stepsize = cvxclst_steepest_descent!(U, Q, ρ, gradient)
#
#         # penalty schedule + acceleration
#         ρ_new = penalty(ρ, iteration)       # check for updates to the penalty coefficient
#         ρ != ρ_new && restart!(strategy, U) # check for restart due to changing objective
#         apply_momentum!(U, strategy)        # apply acceleration strategy
#         ρ = ρ_new                           # update penalty
#
#         # convergence history
#         loss, distance, gradient = cvxclst_evaluate!(Q, Y, Δ, index, W, U, X, K, ρ)
#         data = package_data(loss, distance, ρ, gradient, stepsize)
#
#         if trace # only record outside clustering path algorithm
#             update_history!(history, data, iteration)
#         end
#
#         loss_old = loss_new
#         loss_new = loss
#         dist_old = dist_new
#         dist_new = distance
#         iteration += 1
#     end
#
#     if !trace # record history only at the end for path algorithm
#         update_history!(history, data, iteration-1)
#     end
#
#     return U
# end

function convex_clustering_path(algorithm::SteepestDescent, W, X; history::histT = nothing, kwargs...) where histT
    #
    # extract problem dimensions
    d, n = size(X)      # d features by n samples
    m = binomial(n, 2)  # number of constraints
    M = d*m             # number of rows
    N = d*n             # number of columns

    # allocate optimization variable
    U = copy(X)
    u = vec(U)
    optvars = (u = u,)

    # allocate derivatives
    ∇f = zeros(N)
    ∇d = zeros(N)
    ∇h = zeros(N)
    derivs = (∇f = ∇f, ∇d = ∇d, ∇h = ∇h)

    # generate operators
    D = CvxClusterFM(d, n)
    x = vec(X)

    # allocate any additional arrays for mat-vec multiplication
    z = zeros(M)
    Pz = zeros(M)
    ds = zeros(m)
    ss = similar(ds)
    buffers = (z = z, U = U, Pz = Pz, ds = ds, ss = ss)

    # allocate outputs
    U_path = typeof(X)[]
    ν_path = Int[]

    # initialize
    ν_max = binomial(n, 2)
    ν = ν_max - 1

    # each instance uses the previous solution as the starting point
    while ν ≥ 0
        if ν ≤ ν_max >> 2
            operators = (D = D, x = x, o = MaxParam, K = ν, compute_projection = compute_sparse_projection)

            optimize!(algorithm, cvxclst_eval, cvxclst_iter, optvars, derivs, operators, buffers; kwargs...)
        else
            operators = (D = D, x = x, o = MinParam, K = ν_max - ν, compute_projection = compute_sparse_projection)

            optimize!(algorithm, cvxclst_eval, cvxclst_iter, optvars, derivs, operators, buffers; kwargs...)
        end

        # convergence history
        if !(history isa Nothing)
            f_loss, h_dist, h_ngrad = cvxclst_eval(algorithm, optvars, derivs, operators, buffers, 1.0)
            data = package_data(f_loss, h_dist, h_ngrad, 0.0, 0.0)
            update_history!(history, data, 0)
        end

        # add to solution path
        push!(U_path, copy(U))
        push!(ν_path, ν)

        # count satisfied constraints
        evaluate_distances!(ds, U)

        nconstraint = 0
        for Δ in ds
            nconstraint += (log(10, Δ) ≤ -3) # distances within 10^-3 are 0
        end

        # decrease ν with a heuristic that guarantees a decrease
        ν = min(ν - 1, ν_max - nconstraint - 1)
    end

    solution_path = (U = U_path, ν = ν_path)

    return solution_path
end
