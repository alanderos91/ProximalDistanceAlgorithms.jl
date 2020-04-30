function cvxclst_stepsize(W, Q, ρ)
    a = dot(Q, Q)                               # norm^2 of gradient
    b = __evaluate_weighted_gradient_norm(W, Q) # norm^2 of W*D*gradient
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
    for j in 1:n, i in j+1:n, k in 1:d
        l = tri2vec(i, j, n)
        Y[k,l] = W[i,j]*(U[k,i] - U[k,j]) - Y[k,l]
    end
    cvxclst_apply_fusion_matrix_transpose!(Q, W, Y)

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
    γ, normgrad = cvxclst_stepsize(W, Q, ρ)

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
    accel::accelT         = Val(:none)) where {FunctionLike,accelT}
    #
    # extract problem dimensions
    d, n = size(X)

    # allocate optimization variable
    UL = zero(X)
    UR = zero(X)

    # allocate gradient and auxiliary variables
    Kmax = binomial(n, 2)
    Q = similar(X)          # gradient
    Y = zeros(d, Kmax)      # encodes differences between columns of U
    Δ = zeros(n, n)         # encodes pairwise distances
    index = collect(1:n*n)  # index vector

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
    γ, g, l, o, w, i = cvxclst_subproblem!(Q, Y, Δ, index, W, UL, X, K, ρ_init, maxiters, strategy, penalty)
    println("   objective = $(o)")
    push!(Upath, copy(UL))
    push!(Kpath, K)
    push!(γpath, γ)
    push!(gpath, g)
    push!(lpath, l)
    push!(opath, o)
    push!(dpath, w)
    push!(iters, i)
    # copyto!(UL, X)
    fill!(UL, 0)

    opt = o

    while searching
        # search left child
        println("Searching with K = $(KL)")
        γL, gL, lL, oL, dL, iL = cvxclst_subproblem!(Q, Y, Δ, index, W, UL, X, K, ρ_init, maxiters, strategy, penalty)
        println("   objective = $(oL)")

        # search right child
        println("Searching with K = $(KR)")
        γR, gR, lR, oR, dR, iR = cvxclst_subproblem!(Q, Y, Δ, index, W, UR, X, K, ρ_init, maxiters, strategy, penalty)
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
        # copyto!(UL, X)
        # copyto!(UR, X)
        fill!(UL, 0)
        fill!(UR, 0)

        x1 = lpath[end]
        x2 = dpath[end]
        x3 = opath[end]

        if KL in Kpath || KR in Kpath
            searching = false
        end
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
