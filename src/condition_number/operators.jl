struct ConNumFM{T} <: FusionMatrix{T}
    c::T
    M::Int
    N::Int
end

function ConNumFM(c::T, p::Integer) where {T<:Number}
    ConNumFM{T}(c, p*p, p)
end

# implementation
Base.size(D::ConNumFM) = (D.M, D.N)

function apply_fusion_matrix!(z, D::ConNumFM, x)
    p = size(D, 2)
    c = D.c
    for j in eachindex(x)
        @inbounds xj = x[j]
        offset = p*(j-1)
        @simd for i in eachindex(x)
            @inbounds z[offset+i] = xj - c*x[i]
        end
    end
    return z
end

function apply_fusion_matrix_transpose!(x, D::ConNumFM, z)
    p = size(D, 2)
    c = D.c

    fill!(x, 0)
    for j in eachindex(x)
        offset = p*(j-1)

        # apply sparse part of block
        @simd for i in eachindex(x)
            @inbounds x[i] = x[i] - c*z[offset+i]
        end

        # apply dense row in block
        xj = zero(eltype(x))
        @simd for i in eachindex(x)
            @inbounds xj = xj + z[offset+i]
        end
        x[j] += xj
    end
    return x
end

function instantiate_fusion_matrix(D::ConNumFM{T}) where T
    p², p = size(D)
    c = D.c

    C = kron(-c*ones(p), I(p))
    S = kron(I(p), ones(p))

    return sparse(C + S)
end

struct ConNumFGM{T} <: FusionGramMatrix{T}
    c::T
    N::Int
end

Base.size(DtD::ConNumFGM) = (DtD.N, DtD.N)

function apply_fusion_gram_matrix!(y, DtD::ConNumFGM, x)
    p = DtD.N
    c = DtD.c

    a = p*(c^2 + 1)
    b = -2*c*sum(x)
    for k in eachindex(x)
        y[k] = a*x[k] + b
    end
    return y
end

Base.:(*)(Dt::TransposeMap{T,ConNumFM{T}}, D::ConNumFM{T}) where T = ConNumFGM(D.c, D.N)
