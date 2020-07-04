struct ImgTvdFM{T} <: FusionMatrix{T}
    m::Int  # width of image in pixels
    n::Int  # length of image in pixels
    M::Int  # total number of constraints
    N::Int  # total number of pixels

    function ImgTvdFM{T}(m::Integer, n::Integer) where T <: Number
        M = 2*m*n - m - n + 1
        N = m*n
        return new{T}(m,n,M,N)
    end
end

# default eltype to Int64
ImgTvdFM(m::Integer, n::Integer) = ImgTvdFM{Int}(m, n)

# implementation
Base.size(D::ImgTvdFM) = (D.M, D.N)

function LinearMaps.A_mul_B!(z::AbstractVector, D::ImgTvdFM, x::AbstractVector)
    m, n = D.m, D.n
    M, N = D.M, D.N
    xind = LinearIndices((1:m, 1:n))
    dxind = LinearIndices((1:(m-1), 1:n))
    dyind = LinearIndices((1:m, 1:(n-1)))
    offset = length(dxind)

    # compute derivatives along columns
    for j in 1:n
        for i in 1:m-1
            @inbounds z[dxind[i,j]] = x[xind[i+1,j]] - x[xind[i,j]]
        end
    end

    # compute derivatives along rows
    for j in 1:n-1
        for i in 1:m
            @inbounds z[dyind[i,j]+offset] = x[xind[i,j+1]] - x[xind[i,j]]
        end
    end

    # contribution from extra row to ensure non-singularity
    @inbounds z[end] = x[end]

    return z
end

function LinearMaps.At_mul_B!(x::AbstractVector, D::ImgTvdFM, z::AbstractVector)
    m, n = D.m, D.n
    M, N = D.M, D.N
    xind = LinearIndices((1:m, 1:n))
    dxind = LinearIndices((1:(m-1), 1:n))
    dyind = LinearIndices((1:m, 1:(n-1)))
    offset = length(dxind)
    fill!(x, 0)

    # apply Dx'
    for j in 1:n
        @inbounds x[xind[1,j]] += -z[dxind[1,j]]
    end

    for j in 1:n
        for i in 2:m-1
            @inbounds x[xind[i,j]] += z[dxind[i-1,j]] - z[dxind[i,j]]
        end
    end

    for j in 1:n
        @inbounds x[xind[m,j]] += z[dxind[m-1,j]]
    end

    # apply Dy'
    for i in 1:m
        @inbounds x[xind[i,1]] += -z[dyind[i,1]+offset]
    end

    for j in 2:n-1
        for i in 1:m
            @inbounds x[xind[i,j]] += z[dyind[i,j-1]+offset] - z[dyind[i,j]+offset]
        end
    end

    for i in 1:m
        @inbounds x[xind[i,n]] += z[dyind[i,n-1]+offset]
    end

    # apply extra row transpose
    @inbounds x[end] += z[end]

    return x
end

function instantiate_fusion_matrix(D::ImgTvdFM{T}) where T <: Number
    m, n = D.m, D.n

    # forward difference operator on columns
    Sx = spzeros(m-1, m)
    for i in 1:m-1
        Sx[i,i] = -1
        Sx[i,i+1] = 1
    end

    # forward difference operator on rows
    Sy = spzeros(n,n-1)
    for i in 1:n-1
        Sy[i,i] = -1
        Sy[i+1,i] = 1
    end

    # stack operators and add redundant row
    S = [kron(I(n), Sx); kron(Sy', I(m)); zeros(1, m*n)]
    S[end] = 1

    return S
end

struct ImgTvdFGM{T} <: FusionGramMatrix{T}
    m::Int
    n::Int
    N::Int

    function ImgTvdFGM{T}(m::Integer, n::Integer) where T <: Number
        new{T}(m,n,m*n)
    end
end

# default eltype to Int64
ImgTvdFGM(m::Integer, n::Integer) = ImgTvdFGM{Int}(m, n)

# implementation
Base.size(DtD::ImgTvdFGM) = (DtD.N, DtD.N)

function LinearMaps.A_mul_B!(y::AbstractVector, DtD::ImgTvdFGM, x::AbstractVector)
    m, n = DtD.m, DtD.n
    imind = LinearIndices((1:m, 1:n))

    # accumulate Dx'*dx
    @inbounds for j in 1:n
        # first row in block
        k = imind[1,j]
        y[k] = x[k] - x[k+1]

        # intermediate rows in block
        for i in 2:m-1
            k = imind[i,j]
            y[k] = -x[k-1] + 2*x[k] - x[k+1]
        end

        # last row in block
        k = imind[m,j]
        y[k] = -x[k-1] + x[k]
    end

    # accumulate Dy'dy

    # first block
    @inbounds for i in 1:m
        k = imind[i,1]
        y[k] += x[k] - x[k+m]
    end

    # intermediate blocks
    @inbounds for j in 2:n-1
        for i in 1:m
            k = imind[i,j]
            y[k] += -x[k-m] + 2*x[k] - x[k+m]
        end
    end

    # last block
    @inbounds for i in 1:m
        k = imind[i,n]
        y[k] += -x[k-m] + x[k]
    end

    # accumulate u*u'
    @inbounds y[end] += x[end]

    return y
end

Base.:(*)(Dt::TransposeMap{T,ImgTvdFM{T}}, D::ImgTvdFM{T}) where T = ImgTvdFGM{T}(D.m, D.n)
