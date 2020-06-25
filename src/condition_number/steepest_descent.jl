function connum_iter(::SteepestDescent, optvars, derivs, operators, buffers, ρ)
    x = optvars.x
    ∇h = derivs.∇h
    D = operators.D
    y = operators.y
    z = buffers.z

    # evaluate stepsize
    mul!(z, D, ∇h)
    a = dot(∇h, ∇h)
    b = dot(z, z)
    γ = a / (a + ρ*b + eps())

    # move in the direction of steepest descent
    axpy!(-γ, ∇h, x)

    return γ
end

# function should be able to handle three cases:
# if M is a Matrix we compute its SVD and return the new matrix
# if M is a SVD type just use that info and return the new matrix
# if M is a vector of singular values we can only return the new values
function reduce_cond(algorithm::SteepestDescent, c, M; kwargs...)
    #
    # extract problem information
    y, U, Vt = extract_svd(M)
    p = length(x)

    # allocate optimization variable
    x = copy(y)
    optvars = (x = x,)

    # allocate derivatives
    ∇f = similar(x)
    ∇d = similar(x)
    ∇h = similar(x)
    derivs = (∇f = ∇f, )

    # generate operators
    D = ConNumFM(c, p)
    P(x) = max.(x, 0)
    operators = (D = D, P = P, y = y)

    # allocate any additional arrays for mat-vec multiplication
    M, N = size(D)
    z = zeros(M)
    buffers = (z = z,)
    # r = zeros(N+M*N)

    optimize!(algorithm, connum_eval, connum_iter, optvars, derivs, operators, buffers; kwargs...)

    return U*Diagonal(x)*Vt
end
