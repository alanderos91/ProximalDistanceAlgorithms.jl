function cvxclst_stepsize(W, Q, ρ)
    a = dot(Q, Q)                               # norm^2 of gradient
    b = __evaluate_weighted_gradient_norm(W, Q) # norm^2 of W*D*gradient
    γ = a / (a + ρ*b + eps())

    return γ, sqrt(a)
end

# used internally
function cvxclst_evaluate_gradient!(Q, y, p, W, U, X, Iv, Jv, v, ρ)
    fill!(Iv, 0); fill!(Jv, 0); fill!(v, 0); fill!(Q, 0); fill!(p, 0)

    sparse_fused_block_projection!(p, Iv, Jv, v, W, U)
    cvxclst_apply_fusion_matrix!(y, W, U)
    @. y = y - p
    cvxclst_apply_fusion_matrix_transpose!(Q, W, y)

    for idx in eachindex(Q)
        Q[idx] = (U[idx] - X[idx]) + ρ*Q[idx]
    end
end

function cvxclst_evaluate_gradient!(Q, y, p, index, W, U, X, k, ρ)
    fill!(Q, 0); fill!(p, 0)

    cvxclst_apply_fusion_matrix!(y, W, U)
    sparse_fused_block_projection!(p, y, index, k)
    @. y = y - p
    cvxclst_apply_fusion_matrix_transpose!(Q, W, y)

    for idx in eachindex(Q)
        Q[idx] = (U[idx] - X[idx]) + ρ*Q[idx]
    end
end

# for testing
function cvxclst_evaluate_gradient(W, U, X, ρ, k)
    p, Iv, Jv, v = sparse_fused_block_projection(W, U, k)
    y = cvxclst_apply_fusion_matrix(W, U)
    @. y = y - p
    Q = cvxclst_apply_fusion_matrix_transpose(W, y)
    for idx in eachindex(Q)
        Q[idx] = (U[idx] - X[idx]) + ρ*Q[idx]
    end

    return Q
end

# function cvxclst_steepest_descent!(Q, U, y, p, X, W, ρ, Iv, Jv, v)
#     # 1a. form the gradient:
#     cvxclst_evaluate_gradient!(Q, y, p, W, U, X, Iv, Jv, v, ρ)
#
#     # 1b. evaluate loss, penalty, and objective:
#     loss, penalty, objective = cvxclst_evaluate_objective(U, X, y, ρ)
#
#     # 2. compute stepsize
#     γ, normgrad = cvxclst_stepsize(W, Q, ρ)
#
#     # 3. apply the update
#     for idx in eachindex(U)
#         U[idx] = U[idx] - γ*Q[idx]
#     end
#
#     return γ, normgrad, loss, objective, penalty
# end

function cvxclst_steepest_descent!(Q, U, y, p, index, X, W, K, ρ)
    # 1a. form the gradient:
    cvxclst_evaluate_gradient!(Q, y, p, index, W, U, X, K, ρ)

    # 1b. evaluate loss, penalty, and objective:
    loss, penalty, objective = cvxclst_evaluate_objective(U, X, y, ρ)

    # 2. compute stepsize
    γ, normgrad = cvxclst_stepsize(W, Q, ρ)

    # 3. apply the update
    for idx in eachindex(U)
        U[idx] = U[idx] - γ*Q[idx]
    end

    return γ, normgrad, loss, objective, penalty
end

# function convex_clustering_path(::SteepestDescent, W, X;
#     ρ_init::Real          = 1.0,
#     maxiters::Integer     = 100,
#     penalty::Function     = __default_schedule,
#     history::FunctionLike = __default_logger,
#     K_path::AbstractVector{Int}   = Int[]) where FunctionLike
#     #
#     # extract problem dimensions
#     d, n = size(X)
#
#     # allocate optimization variable
#     U = zero(X)
#
#     # allocate gradient and auxilliary variable
#     Q = similar(X)
#     y = zeros(eltype(X), d*binomial(n, 2))
#     p = zero(y)
#
#     # allocate solution path
#     path = [similar(U) for _ in eachindex(K_path)]
#
#     for (i, K) in enumerate(K_path)
#         # initialize data structures for sparsity projection
#         Iv = zeros(Int, K)
#         Jv = zeros(Int, K)
#         v  = zeros(K)
#
#         # set the starting value for the penalty coefficient
#         ρ = ρ_init
#
#         # use the old U as a warm-start
#         convex_clustering!(Q, U, Iv, Jv, v, y, p, X, W, ρ, maxiters, penalty, history)
#
#         # save solution for the subproblem
#         copyto!(path[i], U)
#     end
#
#     return path
# end

function cvxclst_subproblem!(Q, U, y, p, index, X, W, K, ρ, maxiters, strategy, penalty)
    iteration = 1
    old_loss = 1.0
    rel = Inf
    dist = Inf

    data = (0.0, 0.0, 0.0, 0.0, 0.0)

    while iteration ≤ maxiters && (rel > 1e-6 || dist > 1e-4)
        # apply iteration map
        data = cvxclst_steepest_descent!(Q, U, y, p, index, X, W, K, ρ)

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

    data = cvxclst_steepest_descent!(Q, U, y, p, index, X, W, K, ρ)

    return data[1], data[2], data[3], data[4], data[5], iteration - 1
end

function convex_clustering(::SteepestDescent, W, X;
    ρ_init::Real          = 1.0,
    maxiters::Integer     = 100,
    penalty::Function     = __default_schedule,
    history::FunctionLike = __default_logger,
    accel::accelT         = Val(:none)) where {FunctionLike,accelT}
    #
    # extract problem dimensions
    d, n = size(X)

    # allocate optimization variable
    UL = copy(X)
    UR = copy(X)

    # allocate gradient and auxiliary variables
    Kmax = d * binomial(n, 2)
    Q = similar(X)
    y = zeros(eltype(X), Kmax)
    p = zero(y)

    # initialize data structures for sparsity projection
    index = collect(1:length(y))

    # construct acceleration strategy
    strategy = get_acceleration_strategy(accel, X)

    # allocate data for output
    Upath = typeof(X)[]
    Kpath = Int[]
    γpath = Float64[]
    gpath = Float64[]
    lpath = Float64[]
    opath = Float64[]
    dpath = Float64[]
    iters = Int[]

    # flags for checking convergence
    searching = true
    opt = Inf
    K = (Kmax + 1) >> 1
    KL = (K + 1) >> 1
    KR = (Kmax + K + 1) >> 1
    @show Kmax
    # initialize the search heuristic
    println("Searching with K = $(K)")
    γ, g, l, o, w, i = cvxclst_subproblem!(Q, UL, y, p, index, X, W, K, ρ_init, maxiters, strategy, penalty)
    println("   objective = $(o)")
    push!(Upath, copy(UL))
    push!(Kpath, K)
    push!(γpath, γ)
    push!(gpath, g)
    push!(lpath, l)
    push!(opath, o)
    push!(dpath, w)
    push!(iters, i)
    copyto!(UL, X)

    opt = o

    while searching
        # search left child
            println("Searching with K = $(KL)")
        γL, gL, lL, oL, dL, iL = cvxclst_subproblem!(Q, UL, y, p, index, X, W, KL, ρ_init, maxiters, strategy, penalty)
        println("   objective = $(oL)")

        # search right child
        println("Searching with K = $(KR)")
        γR, gR, lR, oR, dR, iR = cvxclst_subproblem!(Q, UR, y, p, index, X, W, KR, ρ_init, maxiters, strategy, penalty)
        println("   objective = $(oR)")

        if opt < oL && opt < oR
            # terminate the search
            searching = false
        elseif oL < opt < oR
            # search the left child
            K = KL
            opt = oL
            push!(Upath, copy(UL))
            push!(Kpath, KL)
            push!(γpath, γL)
            push!(gpath, gL)
            push!(lpath, lL)
            push!(opath, oL)
            push!(dpath, dL)
            push!(iters, iL)
        else oR < opt < oL
            # search the right child
            K = KR
            opt = oR
            push!(Upath, copy(UR))
            push!(Kpath, KR)
            push!(γpath, γR)
            push!(gpath, gR)
            push!(lpath, lR)
            push!(opath, oR)
            push!(dpath, dR)
            push!(iters, iR)
        end

        # update left and right children
        KL = (K + 1) >> 1
        KR = (Kmax + K + 1) >> 1
        copyto!(UL, X)
        copyto!(UR, X)
    end

    # package the output
    output = (U = Upath, K = Kpath,
        loss      = lpath,
        penalty   = dpath,
        objective = opath,
        gradient  = gpath,
        stepsize  = γpath,
        iterations = iters)

    return output
end
