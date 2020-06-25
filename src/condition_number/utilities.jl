function connum_eval(::AlgorithmOption, optvars, derivs, operators, buffers, ρ)
    x = optvars.x

    ∇f = derivs.∇f
    ∇d = derivs.∇d
    ∇h = derivs.∇h

    D = operators.D
    y = operators.y
    P = operators.P

    z = buffers.z
    Pz = buffers.Pz

    mul!(z, D, x)
    @. Pz = P(z)
    axpy!(-1, Pz, z)
    @. ∇f = x - y
    mul!(∇d, D', z)
    @. ∇h = ∇f + ρ*∇d
    # for k in eachindex(∇h)
    #     ∇h[k] = ∇f[k] + ρ*∇d[k]
    # end

    loss = SqEuclidean()(x, y) / 2
    penalty = dot(z, z)
    normgrad = dot(∇h, ∇h)

    return loss, penalty, normgrad
end

function extract_svd(M::Matrix)
    F = svd(M)
    return (F.S, F.U, F.Vt)
end
extract_svd(M::SVD) = (M.S, M.U, M.Vt)
extract_svd(M::Vector) = (M, LinearAlgebra.I, LinearAlgebra.I)
