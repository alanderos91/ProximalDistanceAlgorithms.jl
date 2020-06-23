function imgtvd_eval(::AlgorithmOption, optvars, derivs, operators, buffers, ρ)
    u = optvars.u
    ∇f = derivs.∇f
    ∇d = derivs.∇d
    ∇h = derivs.∇h

    D = operators.D
    w = operators.w
    o = operators.o
    K = operators.K
    compute_projection = operators.compute_projection

    z = buffers.z
    ds = buffers.ds
    Pz = buffers.Pz

    mul!(z, D, u)
    @. ds = abs(z)
    P = compute_projection(ds, o, K)
    @. Pz = P(z)
    @. z = z - Pz
    @. ∇f = u - w
    mul!(∇d, D', z)
    @. ∇h = ∇f + ρ * ∇d

    loss = SqEuclidean()(u, w) / 2
    penalty = dot(z, z)
    normgrad = dot(∇h, ∇h)

    return loss, penalty, normgrad
end
