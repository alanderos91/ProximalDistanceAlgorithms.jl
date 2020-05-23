function imgtvd_Dx_matrix(m, n)
    D = zeros(m-1, m)
    for i in 1:m-1
        D[i,i] = -1
        D[i,i+1] = 1
    end
    return D
end

function imgtvd_Dy_matrix(m, n)
    D = zeros(n,n-1)
    for i in 1:n-1
        D[i,i] = -1
        D[i+1,i] = 1
    end
    return D
end

function imgtvd_fusion_matrix(m, n)
    Dx = imgtvd_Dx_matrix(m, n)
    Dy = imgtvd_Dy_matrix(m, n)
    D = [kron(I(n), Dx); kron(Dy', I(m)); zeros(1, m*n)]
    D[end] = 1
    return D
end

function imgtvd_apply_Dx!(dx, U)
    m, n = size(U)
    for j in 1:n
        for i in 1:m-1
            @inbounds dx[i,j] = U[i+1,j] - U[i,j]
        end
    end
    return dx
end

function imgtvd_apply_Dx_transpose!(Q, dx)
    m, n = size(Q)
    for j in 1:n
        @inbounds Q[1,j] += -dx[1,j]
    end

    for j in 1:n, i in 2:m-1
        @inbounds Q[i,j] += dx[i-1,j] - dx[i,j]
    end

    for j in 1:n
        @inbounds Q[m,j] += dx[m-1,j]
    end
    return Q
end

function imgtvd_apply_Dy!(dy, U)
    m, n = size(U)
    for j in 1:n-1
        for i in 1:m
            @inbounds dy[i,j] = U[i,j+1] - U[i,j]
        end
    end
    return dy
end

function imgtvd_apply_Dy_transpose!(Q, dy)
    m, n = size(Q)
    for i in 1:m
        @inbounds Q[i,1] += -dy[i,1]
    end

    for j in 2:n-1, i in 1:m
        @inbounds Q[i,j] += dy[i,j-1] - dy[i,j]
    end

    for i in 1:m
        @inbounds Q[i,n] += dy[i,n-1]
    end
    return Q
end

function imgtvd_apply_D!(z, dx, dy, U)
    imgtvd_apply_Dx!(dx, U)
    imgtvd_apply_Dy!(dy, U)
    unsafe_copyto!(z, 1, dx, 1, length(dx))
    unsafe_copyto!(z, length(dx)+1, dy, 1, length(dy))
    @inbounds z[end] = U[end]
    return z
end

function imgtvd_apply_D_transpose!(Q, dx, dy)
    imgtvd_apply_Dx_transpose!(Q, dx)
    imgtvd_apply_Dy_transpose!(Q, dy)
    return Q
end
