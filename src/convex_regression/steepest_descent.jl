function cvxreg_iter(::SteepestDescent, optvars, derivs, operators, buffers, ρ)
    x = optvars.x       # [θ; ξ]
    ∇h = derivs.∇h      # full gradient
    ∇h_θ = derivs.∇h_θ  # view of ∇h with block corres. to θ
    ∇h_ξ = derivs.∇h_ξ  # view of ∇h with block corres. to ξ
    D = operators.D     # fusion matrix
    z = buffers.z       # z = D*x

    # evaluate step size
    mul!(z, D, ∇h)
    a = dot(∇h_θ, ∇h_θ) # ||∇h_θ(x)||^2
    b = dot(∇h_ξ, ∇h_ξ) # ||∇h_ξ(x)||^2
    c = dot(z, z)       # ||D*∇h(x)||^2
    γ = (a + b) / (a + ρ*c + eps())

    # apply the steepest descent update
    @. x = x - γ*∇h

    return γ
end

function cvxreg_fit(algorithm::SteepestDescent, y, X; kwargs...)
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
    derivs = (∇f = ∇f, ∇d = ∇d, ∇h = ∇h, ∇h_θ = ∇h_θ, ∇h_ξ = ∇h_ξ)

    # generate operators
    A = CvxRegBlockA(n)     # A*θ = θ_j - θ_i
    B = CvxRegBlockB(X)     # B*ξ = <ξ_j, X_i - X_j>
    D = [A B]               # implemented as a BlockMap
    P(x) = min.(x, 0)       # projection onto non-positive orthant
    operators = (D = D, P = P, y = y)

    # allocate any additional arrays for mat-vec multiplication
    z = zeros(M)
    Pz = zeros(M)
    θ = view(x, 1:n) # should be in optvars; needed to avoid issue with accel.
    ξ = view(x, n+1:N)
    buffers = (z = z, Pz = Pz, θ = θ, ξ = ξ)

    optimize!(algorithm, cvxreg_eval, cvxreg_iter, optvars, derivs, operators, buffers; kwargs...)

    return copy(θ), copy(reshape(ξ, d, n))
end
