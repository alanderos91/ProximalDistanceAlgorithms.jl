function cvxclst_evaluate!(Q, Y, Δ, index, W, U, X, K, ρ)
    fill!(Q, 0)
    fill!(Y, 0)
    fill!(Δ, 0)

    d, n = size(U)
    sparse_block_projection!(Y, Δ, index, W, U, K)

    # compute U - P(U)
    for j in 1:n, i in j+1:n
        l = tri2vec(i, j, n)
        for k in 1:d
            Y[k,l] = (U[k,i] - U[k,j]) - Y[k,l]
        end
    end

    # finish forming the gradient
    cvxclst_apply_fusion_matrix_transpose!(Q, Y)
    for idx in eachindex(Q)
        Q[idx] = (U[idx] - X[idx]) + ρ*Q[idx]
    end

    loss = SqEuclidean()(U, X)
    penalty = dot(Y, Y)
    gradient = dot(Q, Q)

    return loss, penalty, gradient
end

function cvxclst_steepest_descent!(U, Q, ρ, gradient)
    # evaluate stepsize
    a = gradient                             # norm^2 of gradient
    b = __evaluate_weighted_gradient_norm(Q) # norm^2 of D*gradient
    γ = a / (a + ρ*b + eps())

    # move in the direction of steepest descent
    for idx in eachindex(U)
        U[idx] = U[idx] - γ*Q[idx]
    end

    return γ
end

function convex_clustering(::SteepestDescent, W, X;
    ρ_init::Real          = 1.0,
    maxiters::Integer     = 100,
    penalty::Function     = __default_schedule,
    history::FunctionLike = nothing,
    K::Integer            = 0,
    ftol::Real            = 1e-6,
    dtol::Real            = 1e-4,
    accel::accelT         = Val(:none)) where {FunctionLike,accelT}
    #
    # extract problem dimensions
    d, n = size(X)

    # allocate optimization variable
    U = zero(X)

    # allocate gradient and auxiliary variables
    Q = similar(X)
    Y = zeros(d, binomial(n, 2))
    Δ = zeros(n, n)
    index = collect(1:n*n)

    # construct type for acceleration strategy
    strategy = get_acceleration_strategy(accel, U)

    # packing
    solution   = (Q = Q, U = U)
    projection = (Y = Y, Δ = Δ, index = index, K = K)
    inputs     = (W = W, X = X, ρ_init = ρ_init, penalty = penalty)
    settings   = (maxiters = maxiters, history = history, ftol = ftol, dtol = dtol, strategy = strategy)

    convex_clustering!(solution, projection, inputs, settings, true)

    return U
end

# for data structure re-use in convex_clustering_path
function convex_clustering!(solution, projection, inputs, settings, trace)
    # centroids and gradient
    U = solution.U
    Q = solution.Q

    # data structures for projection
    Y     = projection.Y
    Δ     = projection.Δ
    index = projection.index
    K     = projection.K

    # input data
    W       = inputs.W
    X       = inputs.X
    ρ_init  = inputs.ρ_init
    penalty = inputs.penalty

    # settings
    maxiters = settings.maxiters
    history  = settings.history
    ftol     = settings.ftol
    dtol     = settings.dtol
    strategy = settings.strategy

    # initialize
    ρ = ρ_init

    loss, distance, gradient = cvxclst_evaluate!(Q, Y, Δ, index, W, U, X, K, ρ)
    data = package_data(loss, distance, ρ, gradient, zero(loss))

    if trace # only record outside clustering path algorithm
        update_history!(history, data, 0)
    end

    loss_old = loss
    loss_new = Inf
    dist_old = distance
    dist_new = Inf
    iteration = 1

    while not_converged(loss_old, loss_new, dist_old, dist_new, ftol, dtol) && iteration ≤ maxiters
        # iterate the algorithm map
        stepsize = cvxclst_steepest_descent!(U, Q, ρ, gradient)

        # penalty schedule + acceleration
        ρ_new = penalty(ρ, iteration)       # check for updates to the penalty coefficient
        ρ != ρ_new && restart!(strategy, U) # check for restart due to changing objective
        apply_momentum!(U, strategy)        # apply acceleration strategy
        ρ = ρ_new                           # update penalty

        # convergence history
        loss, distance, gradient = cvxclst_evaluate!(Q, Y, Δ, index, W, U, X, K, ρ)
        data = package_data(loss, distance, ρ, gradient, stepsize)

        if trace # only record outside clustering path algorithm
            update_history!(history, data, iteration)
        end

        loss_old = loss_new
        loss_new = loss
        dist_old = dist_new
        dist_new = distance
        iteration += 1
    end

    if !trace # record history only at the end for path algorithm
        update_history!(history, data, iteration-1)
    end

    return U
end

function convex_clustering_path(::SteepestDescent, W, X;
    ρ_init::Real      = 1.0,
    maxiters::Integer = 100,
    penalty::Function = __default_schedule,
    history::histT    = nothing,
    ftol::Real        = 1e-6,
    dtol::Real        = 1e-4,
    accel::accelT     = Val(:none)) where {histT, accelT}
    #
    # initialize
    d, n = size(X)
    ν_max = binomial(n, 2)
    ν = ν_max

    # solution path
    U_path = typeof(X)[]
    ν_path = Int[]

    # allocate optimization variable
    U = copy(X)

    # allocate gradient and auxiliary variables
    Q = similar(X)
    Y = zeros(d, binomial(n, 2))
    Δ = zeros(n, n)
    index = collect(1:n*n)

    # construct type for acceleration strategy
    strategy = get_acceleration_strategy(accel, U)

    # packing
    solution   = (Q = Q, U = U)
    inputs     = (W = W, X = X, ρ_init = ρ_init, penalty = penalty)
    settings   = (maxiters = maxiters, history = history, ftol = ftol, dtol = dtol, strategy = strategy)

    # each instance uses the previous solution as the starting point
    while ν ≥ 0
        # solve problem with ν violated constraints
        projection = (Y = Y, Δ = Δ, index = index, K = ν)
        result = convex_clustering!(solution, projection, inputs, settings, false)

        # add to solution path
        push!(U_path, copy(result))
        push!(ν_path, ν)

        # decrease ν with a heuristic that guarantees descent
        ν = min(ν - 1, ν_max - count_satisfied_constraints(result) - 1)
    end

    solution_path = (U = U_path, ν = ν_path)

    return solution_path
end

function count_satisfied_constraints(U, tol = 3.0)
    d, n = size(U)
    Δ = pairwise(Euclidean(), U, dims = 2)
    @. Δ = log(10, Δ)

    nconstraint = 0

    for j in 1:n, i in j+1:n
        nconstraint += (Δ[i,j] ≤ -tol)
    end

    return nconstraint
end
