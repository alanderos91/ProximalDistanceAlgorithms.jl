function connum_eval(::AlgorithmOption, optvars, derivs, operators, buffers, ρ)
    x = optvars.x

    ∇f = derivs.∇f
    ∇d = derivs.∇d
    ∇h = derivs.∇h

    D = operators.D
    y = operators.y
    P = operators.P

    z = buffers.z
    r = buffers.r

    mul!(z, D, u)
    @. z = z - P(z)
    @. ∇f = x - y
    mul!(∇d, D', z)
    @. ∇h = ∇f + ρ*∇d

    loss = SqEuclidean()(x, y) / 2
    penalty = dot(z, z)
    normgrad = dot(∇h, ∇h)

    return loss, penalty, normgrad
end

extract_svd(M::Matrix) = (x = svd(M); (x.S, x.U, x.Vt)
extract_svd(M::SVD) = (M.S, M.U, M.Vt)
extract_svd(M::Vector) = (M, LinearAlgebra.I, LinearAlgebra.I)
