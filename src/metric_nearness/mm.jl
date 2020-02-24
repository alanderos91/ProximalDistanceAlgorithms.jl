function metric_mm!(X, x, ∇X, B, b, W, D, cg_iterator, ρ)
    n = size(X, 1)

    # form the gradient and RHS of A*x = b
    fill!(∇X, 0)
    fill!(B, 0)
    __apply_TtT_minus_npproj!(∇X, B, X)
    __accumulate_I_minus_nnproj!(∇X, B, X)

    # evaluate loss, penalty, and objective:
    loss = 0.5 * (dot(X, X) - 2*dot(D, X) + dot(D, D))
    penalty = dot(∇X, ∇X)
    objective = loss + 0.5*ρ*penalty

    # finish forming the gradient and b
    for j in 1:n, i in j+1:n
        ∇X[i,j] = W[i,j] * (X[i,j] - D[i,j]) + ρ*∇X[i,j]
        B[i,j] = B[i,j] + W[i,j] * D[i,j] / ρ
    end
    g = norm(∇X)

    # solve the linear system
    __trivec_copy!(b, B)
    __trivec_copy!(x, X)
    __do_linear_solve!(cg_iterator, b)

    # apply the update
    for j in 1:n, i in j+1:n
        k = trivec_index(n, i, j)
        X[i,j] = x[k]
    end

    return g, loss, objective, penalty
end

function metric_projection(::MM, W, D;
    ρ_init::Real      = 1.0,
    maxiters::Integer = 100,
    penalty::Function = __default_schedule,
    history::FuncLike = __default_logger) where FuncLike
    #
    # extract problem dimensions
    n = size(D, 1)
    m = binomial(n, 2)

    # assume symmetry in D and zeros in diagonal
    D_tri = LowerTriangular(D)
    W_tri = LowerTriangular(W)
    w = zeros(eltype(W), m)
    __trivec_copy!(w, W_tri)
    W_diag = Diagonal(w)

    # allocate optimization variable
    X = copy(D_tri)
    x = zeros(eltype(X), m)

    # allocate gradient
    ∇X = similar(X)

    # intermediates for solving A*trivec(X) = trivec(Z)
    B = similar(X)
    b = zeros(eltype(B), m)

    # initialize penalty coefficient
    ρ = ρ_init

    # sparse matrix A = (W/ρ + T'T)
    T = metric_matrix(n)
    A = T'T

    for k in 1:m
        A[k,k] += W_diag[k,k] / ρ + 1
    end

    # initialize conjugate gradient solver
    cg_iterator = CGIterable(
        A, x,
        similar(b), similar(b), similar(b),
        1e-8, 0.0, 1.0, size(A, 2), 0
    )

    for iteration in 1:maxiters
        # iterate the algorithm map
        data = metric_mm!(X, x, ∇X, B, b, W_tri, D_tri, cg_iterator, ρ)

        # check for updates to the penalty coefficient
        ρ = penalty(A, W_diag, m, ρ, iteration)

        # check for updates to the convergence history
        history(data, iteration)
    end

    return X
end
