# matrix constructors

function metric_matrix(n::Integer)
    nrows = n*(n-1)*(n-2) ÷ 2

    I = zeros(3*nrows)
    J = zeros(3*nrows)
    V = zeros(3*nrows)

    I, J, V = metric_matrix!(I, J, V, n)

    return sparse(I, J, V)
end

function __set_row!(I, J, V, s1, s2, s3, t)
    z = 3*(t-1)+1

    I[z] = t
    J[z] = s1
    V[z] = 1

    I[z+1] = t
    J[z+1] = s2
    V[z+1] = -1

    I[z+2] = t
    J[z+2] = s3
    V[z+2] = -1

    return nothing
end

function metric_matrix!(I, J, V, n)
    t = 1
    for j in 1:n-2, i in j+1:n-1, k in i+1:n
        # map to linear indices
        s1 = trivec_index(n, i, j)
        s2 = trivec_index(n, k, i)
        s3 = trivec_index(n, k, j)

        __set_row!(I, J, V, s1, s2, s3, t); t += 1 # T_ijk
        __set_row!(I, J, V, s2, s1, s3, t); t += 1 # T_jik
        __set_row!(I, J, V, s3, s1, s2, t); t += 1 # T_kij
    end

    return I, J, V
end

# linear operators

function __apply_T!(y, X)
    Δ = 1 # edge counter
    n = size(X, 1)

    for j in 1:n-2, i in j+1:n-1
        a = X[i,j] # fix one edge

        for k in i+1:n
            b = X[k,i]
            c = X[k,j]

            # check edges of one triangle
            y[Δ] = a - b - c; Δ += 1
            y[Δ] = b - a - c; Δ += 1
            y[Δ] = c - a - b; Δ += 1
        end
    end

    return y
end

function __apply_TtT!(Q, X)
    n = size(X, 1)

    for j in 1:n-2, i in j+1:n-1
        a = X[i,j]  # fix one edge

        for k in i+1:n
            b = X[k,i]
            c = X[k,j]

            # check edges of one triangle
            abc = a - b - c
            bac = b - a - c
            cab = c - a - b

            # TtT*x - Tt*(T*x)_{-}
            Q_ij = 3*abc
            Q_ki = 3*bac
            Q_kj = 3*cab

            # accumulate
            Q[i,j] += t_ij
            Q[k,i] += t_ki
            Q[k,j] += t_kj
        end
    end

    return Q
end

function __apply_TtT_minus_npproj!(Q, X)
    n = size(X, 1)

    for j in 1:n-2, i in j+1:n-1
        a = X[i,j]

        for k in i+1:n
            b = X[k,i]
            c = X[k,j]

            abc = a - b - c; fabc = min(0, abc)
            bac = b - a - c; fbac = min(0, bac)
            cab = c - a - b; fcab = min(0, cab)

            # TtT*x - Tt*(T*x)_{-}
            Q_ij = 3*a - b - c - fabc + fbac + fcab
            Q_ki = 3*b - a - c - fbac + fabc + fcab
            Q_kj = 3*c - a - b - fcab + fabc + fbac

            # accumulate
            Q[i,j] += Q_ij
            Q[k,i] += Q_ki
            Q[k,j] += Q_kj
        end
    end

    return Q
end

# Q accumulates gradient
# Z accumulates W*y + ρ*(proj(...))
function __apply_TtT_minus_npproj!(Q, B, X)
    n = size(X, 1)

    for j in 1:n-2, i in j+1:n-1
        a = X[i,j]

        for k in i+1:n
            b = X[k,i]
            c = X[k,j]

            abc = a - b - c; fabc = min(0, abc)
            bac = b - a - c; fbac = min(0, bac)
            cab = c - a - b; fcab = min(0, cab)

            # TtT*x - Tt*(T*x)_{-}
            B_ij = fabc - fbac - fcab
            B_ki = fbac - fabc - fcab
            B_kj = fcab - fabc - fbac

            Q_ij = 3*a - b - c - B_ij
            Q_ki = 3*b - a - c - B_ki
            Q_kj = 3*c - a - b - B_kj

            # accumulate
            B[i,j] += B_ij
            B[k,i] += B_ki
            B[k,j] += B_kj

            Q[i,j] += Q_ij
            Q[k,i] += Q_ki
            Q[k,j] += Q_kj
        end
    end

    return Q, B
end

function __apply_I_minus_nnproj!(Q, X)
    n = size(X, 1)

    for j in 1:n, i in j+1:n
        Q[i,j] = X[i,j] - max(0, X[i,j])
    end

    return Q
end

function __accumulate_I_minus_nnproj!(Q, X)
    n = size(X, 1)

    for j in 1:n, i in j+1:n
        Q[i,j] += X[i,j] - max(0, X[i,j])
    end

    return Q
end

function __accumulate_I_minus_nnproj!(Q, B, X)
    n = size(X, 1)

    for j in 1:n, i in j+1:n
        B_ij = max(0, X[i,j])

        B[i,j] += B_ij
        Q[i,j] += X[i,j] - B_ij
    end

    return Q, B
end

function __apply_T_evaluate_norm_squared(Q)
    v = zero(eltype(Q))
    n = size(Q, 1)

    for j in 1:n-2, i in j+1:n
        a = Q[i,j]  # fix an edge

        for k in i+1:n
            b = Q[k,i]
            c = Q[k,j]

            # accumulate contributions from inequalities in one triangle
            v += (a-b-c)^2
            v += (b-a-c)^2
            v += (c-a-b)^2
        end
    end

    return v
end

function __evaulate_weighted_norm_squared(W, Q)
    v = zero(eltype(Q))
    n = size(Q, 1)

    for j in 1:n, i in j+1:n
        v += W[i,j] * Q[i,j]^2
    end

    return v
end

function __trivec_copy!(b, B)
    n = size(B, 1)

    for j in 1:n, i in j+1:n
        k = trivec_index(n, i, j)
        b[k] = B[i,j]
    end

    return b
end

trivec_index(n, i, j) = (i-j) + n*(j-1) - (j*(j-1))>>1
