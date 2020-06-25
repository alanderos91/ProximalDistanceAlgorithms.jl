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
    cval = -D.c * sum(x)
    for k in eachindex(x)
        xk = x[k]
        offset = p*(k-1)
        for j in eachindex(x)
            z[offset+j] = cval + xk
        end
    end
    return z
end

function apply_fusion_matrix_transpose!(x, D::ConNumFM, z)
    p = size(D, 2)
    cval = -D.c * sum(z)
    for k in eachindex(x)
        x[k] = cval
        offset = p*(k-1)
        for j in eachindex(x)
            x[k] += z[offset+j]
        end
    end
    return x
end

function instantiate_fusion_matrix(D::ConNumFM{T}) where T
    p², p = size(D)
    A = similar(Matrix{T}, p², p)
    for j in 1:p
        for i in 1:p*(j-1)
            A[i,j] = -D.c
        end
        for i in p*(j-1)+1:p*j
            A[i,j] = -D.c + 1
        end
        for i in p*j+1:p²
            A[i,j] = -D.c
        end
    end
    return A
end

struct ConNumFGM{T} <: FusionGramMatrix{T}
    c::T
    N::Int
end

Base.size(DtD::ConNumFGM) = (DtD.N, DtD.N)

function apply_fusion_gram_matrix!(y, DtD::ConNumFGM, x)
    p = DtD.N
    c = DtD.c
    xsum = sum(x)
    for k in eachindex(x)
        y[k] = ((c*p)^2 - 2*c*p) * xsum + p * x[k]
    end
    return y
end

Base.:(*)(Dt::TransposeMap{T,ConNumFM{T}}, D::ConNumFM{T}) where T = ConNumFGM(D.c, D.N)
