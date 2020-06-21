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

function apply_fusion_matrix!(z, D::ImgTvdFM, x)
    m, n = D.m, D.n
    M, N = D.M, D.N
    xind = LinearIndices((1:m, 1:n))
    dxind = LinearIndices((1:(m-1), 1:n))
    dyind = LinearIndices((1:m, 1:(n-1)))
    offset = length(dxind)

    # compute derivatives along columns
    for j in 1:n
        @simd for i in 1:m-1
            @inbounds z[dxind[i,j]] = x[xind[i+1,j]] - x[xind[i,j]]
        end
    end

    # compute derivatives along rows
    for j in 1:n-1
        @simd for i in 1:m
            @inbounds z[dyind[i,j]+offset] = x[xind[i,j+1]] - x[xind[i,j]]
        end
    end

    # contribution from extra row to ensure non-singularity
    @inbounds z[end] = x[end]

    return z
end

function apply_fusion_matrix_transpose!(x, D::ImgTvdFM, z)
    m, n = D.m, D.n
    M, N = D.M, D.N
    xind = LinearIndices((1:m, 1:n))
    dxind = LinearIndices((1:(m-1), 1:n))
    dyind = LinearIndices((1:m, 1:(n-1)))
    offset = length(dxind)
    fill!(x, 0)

    # apply Dx'
    @simd for j in 1:n
        @inbounds x[xind[1,j]] += -z[dxind[1,j]]
    end

    for j in 1:n
        @simd for i in 2:m-1
            @inbounds x[xind[i,j]] += z[dxind[i-1,j]] - z[dxind[i,j]]
        end
    end

    @simd for j in 1:n
        @inbounds x[xind[m,j]] += z[dxind[m-1,j]]
    end

    # apply Dy'
    @simd for i in 1:m
        @inbounds x[xind[i,1]] += -z[dyind[i,1]+offset]
    end

    for j in 2:n-1
        @simd for i in 1:m
            @inbounds x[xind[i,j]] += z[dyind[i,j-1]+offset] - z[dyind[i,j]+offset]
        end
    end

    @simd for i in 1:m
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

#
# function imgtvd_apply_Dx!(dx, U)
#     m, n = size(U)
#     for j in 1:n
#         for i in 1:m-1
#             @inbounds dx[i,j] = U[i+1,j] - U[i,j]
#         end
#     end
#     return dx
# end
#
# function imgtvd_apply_Dx_transpose!(Q, dx)
#     m, n = size(Q)
#     for j in 1:n
#         @inbounds Q[1,j] += -dx[1,j]
#     end
#
#     for j in 1:n, i in 2:m-1
#         @inbounds Q[i,j] += dx[i-1,j] - dx[i,j]
#     end
#
#     for j in 1:n
#         @inbounds Q[m,j] += dx[m-1,j]
#     end
#     return Q
# end
#
# function imgtvd_apply_Dy!(dy, U)
#     m, n = size(U)
#     for j in 1:n-1
#         for i in 1:m
#             @inbounds dy[i,j] = U[i,j+1] - U[i,j]
#         end
#     end
#     return dy
# end
#
# function imgtvd_apply_Dy_transpose!(Q, dy)
#     m, n = size(Q)
#     for i in 1:m
#         @inbounds Q[i,1] += -dy[i,1]
#     end
#
#     for j in 2:n-1, i in 1:m
#         @inbounds Q[i,j] += dy[i,j-1] - dy[i,j]
#     end
#
#     for i in 1:m
#         @inbounds Q[i,n] += dy[i,n-1]
#     end
#     return Q
# end
#
# function imgtvd_apply_D!(z, dx, dy, U)
#     imgtvd_apply_Dx!(dx, U)
#     imgtvd_apply_Dy!(dy, U)
#     unsafe_copyto!(z, 1, dx, 1, length(dx))
#     unsafe_copyto!(z, length(dx)+1, dy, 1, length(dy))
#     @inbounds z[end] = U[end]
#     return z
# end
#
# function imgtvd_apply_D_transpose!(Q, dx, dy)
#     imgtvd_apply_Dx_transpose!(Q, dx)
#     imgtvd_apply_Dy_transpose!(Q, dy)
#     return Q
# end
