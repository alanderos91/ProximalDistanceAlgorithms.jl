struct CondNumFM{T} <: FusionMatrix{T}
    c::T
    M::Int
    N::Int
end

function CondNumFM(c::T, p::Integer) where {T<:Number}
    CondNumFM{T}(c, p*p, p)
end

# implementation
Base.size(D::CondNumFM) = (D.M, D.N)

function apply_fusion_matrix!(z, D::CondNumFM, x)
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

function apply_fusion_matrix_transpose!(x, D::CondNumFM, z)
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

function instantiate_fusion_matrix(D::CondNumFM{T}) where T
    p², p = size(D)
    c = D.c

    C = kron(-c*ones(p), I(p))
    S = kron(I(p), ones(p))

    return sparse(C + S)
end

struct CondNumFGM{T} <: FusionGramMatrix{T}
    c::T
    N::Int
end

Base.size(DtD::CondNumFGM) = (DtD.N, DtD.N)

function apply_fusion_gram_matrix!(y, DtD::CondNumFGM, x)
    p = DtD.N
    c = DtD.c

    a = p*(c^2 + 1)
    b = -2*c*sum(x)
    for k in eachindex(x)
        y[k] = a*x[k] + b
    end
    return y
end

Base.:(*)(Dt::TransposeMap{T,CondNumFM{T}}, D::CondNumFM{T}) where T = CondNumFGM(D.c, D.N)
