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
    o::Base.Ordering = Base.Order.Forward,
    psort::Function = partial_quicksort, kwargs...)
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
    operators = (D = D, x = x, o = o, K = K, compute_projection = psort)

    # allocate any additional arrays for mat-vec multiplication
    z = zeros(M)
    Pz = zeros(M)
    ds = zeros(m)
    buffers = (z = z, U = U, Pz = Pz, ds = ds)

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
#
# function convex_clustering_path(::SteepestDescent, W, X;
#     ρ_init::Real      = 1.0,
#     maxiters::Integer = 100,
#     penalty::Function = __default_schedule,
#     history::histT    = nothing,
#     ftol::Real        = 1e-6,
#     dtol::Real        = 1e-4,
#     accel::accelT     = Val(:none)) where {histT, accelT}
#     #
#     # initialize
#     d, n = size(X)
#     ν_max = binomial(n, 2)
#     ν = ν_max
#
#     # solution path
#     U_path = typeof(X)[]
#     ν_path = Int[]
#
#     # allocate optimization variable
#     U = copy(X)
#
#     # allocate gradient and auxiliary variables
#     Q = similar(X)
#     Y = zeros(d, binomial(n, 2))
#     Δ = zeros(n, n)
#     index = collect(1:n*n)
#
#     # construct type for acceleration strategy
#     strategy = get_acceleration_strategy(accel, U)
#
#     # packing
#     solution   = (Q = Q, U = U)
#     projection = (Y = Y, Δ = Δ, index = index)
#     inputs     = (W = W, X = X, ρ_init = ρ_init, penalty = penalty)
#     settings   = (maxiters = maxiters, history = history, ftol = ftol, dtol = dtol, strategy = strategy)
#
#     # each instance uses the previous solution as the starting point
#     while ν ≥ 0
#         # solve problem with ν violated constraints
#         result = convex_clustering!(solution, projection, inputs, settings, ν, false)
#
#         # add to solution path
#         push!(U_path, copy(result))
#         push!(ν_path, ν)
#
#         # count satisfied constraints
#         Δ = pairwise!(Δ, Euclidean(), result, dims = 2)
#         @. Δ = log(10, Δ)
#
#         nconstraint = 0
#         for j in 1:n, i in j+1:n
#             nconstraint += (Δ[i,j] ≤ -3) # distances within 10^-3 are 0
#         end
#
#         # decrease ν with a heuristic that guarantees descent
#         ν = min(ν - 1, ν_max - nconstraint - 1)
#     end
#
#     solution_path = (U = U_path, ν = ν_path)
#
#     return solution_path
# end
