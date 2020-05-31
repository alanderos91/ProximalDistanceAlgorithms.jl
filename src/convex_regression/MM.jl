function cvxreg_iter(::MM, optvars, derivs, operators, buffers, ρ)
    x = optvars.x
    D = operators.D
    P = operators.P
    y = operators.y

    cg_iterator = buffers.cg_iterator
    b = buffers.b
    z = buffers.z
    Pz = buffers.Pz

    # set up RHS of Ax = b := y + ρ*D'P(D*x)
    mul!(z, D, x)
    @. Pz = P(z)

    fill!(b, 0)
    copyto!(b, 1, y, 1, length(y))
    mul!(b, D', Pz, ρ, 1.0)

    # solve the linear system
    __do_linear_solve!(cg_iterator, b)

    return 1.0
end

function cvxreg_fit(algorithm::MM, y, X; kwargs...)
    #
    # extract problem information
    d, n = size(X)  # features by samples
    M = n*n         # number of subgradient constraints (includes redundancy)
    N = n*(d+1)     # total number of optimization variables

    # allocate optimization variable
    x = zeros(N)            # x = [θ; ξ]
    copyto!(x, 1, y, 1, n)  # θ = y
    optvars = (x = x,)

    # allocate derivatives
    ∇f = zero(x)                # loss
    ∇d = zero(x)                # distance
    ∇h = zero(x)                # objective
    ∇h_θ = view(∇h, 1:n)        # view of ∇h with block corres. to θ
    ∇h_ξ = view(∇h, n+1:N)      # view of ∇h with block corres. to ξ
    ∇²f = [I spzeros(n, n*d); spzeros(n*d, n) spzeros(n, n)]
    derivs = (∇f = ∇f, ∇²f = ∇²f, ∇d = ∇d, ∇h = ∇h, ∇h_θ = ∇h_θ, ∇h_ξ = ∇h_ξ)

    # generate operators
    A = CvxRegBlockA(n)     # A*θ = θ_j - θ_i
    B = CvxRegBlockB(X)     # B*ξ = <ξ_j, X_i - X_j>
    D = [A B]               # implemented as a BlockMap
    P(x) = min.(x, 0)       # projection onto non-positive orthant
    H = CvxRegHessian(n, 1.0, ∇²f, D'D)
    operators = (D = D, P = P, H = H, y = y)

    # allocate any additional arrays for mat-vec multiplication
    z = zeros(M)
    Pz = zeros(M)
    b = similar(x)
    θ = view(x, 1:n) # should be in optvars; needed to avoid issue with accel.
    ξ = view(x, n+1:N)

    # initialize conjugate gradient solver
    b1 = similar(x)
    b2 = similar(x)
    b3 = similar(x)
    cg_iterator = CGIterable(H, x, b1, b2, b3, 1e-8, 0.0, 1.0, size(H, 2), 0)

    buffers = (z = z, Pz = Pz, θ = θ, ξ = ξ, b = b, cg_iterator = cg_iterator)

    optimize!(algorithm, cvxreg_eval, cvxreg_iter, optvars, derivs, operators, buffers; kwargs...)

    return copy(θ), copy(reshape(ξ, d, n))
end
