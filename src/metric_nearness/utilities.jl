function metric_example(n; weighted = false)
    n < 3 && error("number of nodes must be ≥ 3")

    D = zeros(n, n)

    for j in 1:n, i in j+1:n
        u = 10*rand()

        D[i,j] = u
        D[j,i] = u
    end

    W = zeros(n, n)

    for j in 1:n, i in j+1:n
        u = weighted ? rand() : 1.0

        W[i,j] = u
        W[j,i] = u
    end

    return W, D
end

function metric_eval(::AlgorithmOption, optvars, derivs, operators, buffers, ρ)
    x = optvars.x
    ∇f = derivs.∇f
    ∇d = derivs.∇d
    ∇h = derivs.∇h
    D = operators.D
    P = operators.P
    a = operators.a
    z = buffers.z

    mul!(z, D, x)
    @. z = z - P(z)
    @. ∇f = x - a
    mul!(∇d, D', z)
    @. ∇h = ∇f + ρ * ∇d

    loss = SqEuclidean()(x, a) / 2  # 1/2 * ||W^1/2 * (x-y)||^2
    penalty = dot(z, z)             # D*x - P(D*x)
    normgrad = dot(∇h, ∇h)          # ||∇h(x)||^2

    return loss, penalty, normgrad
end
