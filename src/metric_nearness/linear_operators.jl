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
