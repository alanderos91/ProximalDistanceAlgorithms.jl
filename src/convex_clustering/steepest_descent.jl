function cvxclst_stepsize(Q, ρ)
    a = dot(Q, Q)                            # norm^2 of gradient
    b = __evaluate_weighted_gradient_norm(Q) # norm^2 of W*D*gradient
    γ = a / (a + ρ*b + eps())

    return γ, sqrt(a)
end

# used internally
function cvxclst_evaluate_gradient!(Q, Y, Δ, index, W, U, X, K, ρ)
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
    cvxclst_apply_fusion_matrix_transpose!(Q, Y)

    for idx in eachindex(Q)
        Q[idx] = (U[idx] - X[idx]) + ρ*Q[idx]
    end
end

function cvxclst_steepest_descent!(Q, Y, Δ, index, W, U, X, K, ρ)
    # 1a. form the gradient:
    cvxclst_evaluate_gradient!(Q, Y, Δ, index, W, U, X, K, ρ)

    # 1b. evaluate loss, penalty, and objective:
    loss, penalty, objective = cvxclst_evaluate_objective(U, X, Y, ρ)

    # 2. compute stepsize
    γ, normgrad = cvxclst_stepsize(Q, ρ)

    # 3. apply the update
    for idx in eachindex(U)
        U[idx] = U[idx] - γ*Q[idx]
    end

    return γ, normgrad, loss, objective, penalty
end

function cvxclst_subproblem!(Q, Y, Δ, index, W, U, X, K, ρ, maxiters, strategy, penalty)
    iteration = 1
    old_loss = 1.0
    rel = Inf
    dist = Inf

    data = (0.0, 0.0, 0.0, 0.0, 0.0)

    while iteration ≤ maxiters && (rel > 1e-6 || dist > 1e-4)
        # apply iteration map
        data = cvxclst_steepest_descent!(Q, Y, Δ, index, W, U, X, K, ρ)

        # check for updates to the penalty coefficient
        ρ_new = penalty(ρ, iteration)

        # apply acceleration strategy
        ρ != ρ_new && restart!(strategy, U)
        apply_momentum!(U, strategy)

        # update penalty
        ρ = ρ_new

        # convergence checks
        loss = sqrt(data[3])
        dist = sqrt(data[5])

        rel = abs(loss - old_loss) / (old_loss + 1)
        old_loss = loss
        iteration = iteration + 1
    end

    data = cvxclst_steepest_descent!(Q, Y, Δ, index, W, U, X, K, ρ)

    return data[1], data[2], data[3], data[4], data[5], iteration - 1
end

function convex_clustering(::SteepestDescent, W, X;
    ρ_init::Real          = 1.0,
    maxiters::Integer     = 100,
    penalty::Function     = __default_schedule,
    history::FunctionLike = __default_logger,
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

    # initialize penalty coefficient
    ρ = ρ_init

    data = cvxclst_steepest_descent!(Q, Y, Δ, index, W, U, X, K, ρ)

    # check for updates to the penalty coefficient
    ρ_new = penalty(ρ, 1)

    # apply acceleration strategy
    ρ != ρ_new && restart!(strategy, U)
    apply_momentum!(U, strategy)

    # update penalty
    ρ = ρ_new

    # check for updates to the convergence history
    history(data, 1)

    iteration = 2
    not_converged = true
    loss_old = data[3]
    loss_new = Inf
    dist_old = sqrt(data[5])
    dist_new = Inf

    while not_converged && iteration ≤ maxiters
        # iterate the algorithm map
        data = cvxclst_steepest_descent!(Q, Y, Δ, index, W, U, X, K, ρ)

        # check for updates to the penalty coefficient
        ρ_new = penalty(ρ, iteration)

        # apply acceleration strategy
        ρ != ρ_new && restart!(strategy, U)
        apply_momentum!(U, strategy)

        # update penalty
        ρ = ρ_new

        # check for updates to the convergence history
        history(data, iteration)

        # check for convergence
        loss_new = data[3]
        dist_new = sqrt(data[5])

        diff1 = abs(loss_new - loss_old)
        diff2 = abs(dist_new - dist_old)
        not_converged = diff1 > ftol * (loss_old + 1)
        not_converged = not_converged || (diff2 > ftol * (dist_old + 1))
        not_converged = not_converged || (dist_new > dtol)

        loss_old = loss_new
        dist_old = dist_new

        # increment iteration count
        iteration += 1
    end

    return U
end

function convex_clustering_path(::SteepestDescent, W, X;
    ρ_init::Real          = 1.0,
    maxiters::Integer     = 100,
    penalty::Function     = __default_schedule,
    ftol::Real            = 1e-6,
    dtol::Real            = 1e-4,
    accel::accelT         = Val(:none)) where {accelT}
    #
    # initialize
    n = size(X, 2)
    ν_max = binomial(n, 2)
    ν = ν_max

    # solution path
    U_path = typeof(X)[]
    ν_path = Int[]

    while ν ≥ 0
        # solve problem with ν violated constraints
        solution = convex_clustering(SteepestDescent(), W, X, K = ν,
            maxiters = maxiters, penalty = penalty, accel = accel,
            ftol = ftol, dtol = dtol)

        # add to solution path
        push!(U_path, solution)
        push!(ν_path, ν)

        # decrease ν with a heuristic that guarantees descent
        ν = min(ν - 1, ν_max - count_satisfied_constraints(solution) - 1)
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
