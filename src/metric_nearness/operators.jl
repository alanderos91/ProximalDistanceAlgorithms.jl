function instantiate_fusion_matrix(D::MetricFM)
    n = D.n
    nrows = D.M
    ncols = D.N

    I = zeros(nrows)
    J = zeros(nrows)
    V = zeros(nrows)

    edge = 0
    for j in 1:n-2, i in j+1:n-1, k in i+1:n
        # map to linear indices
        s1 = trivec_index(n, i, j)
        s2 = trivec_index(n, k, i)
        s3 = trivec_index(n, k, j)

        __set_row!(I, J, V, s1, s2, s3, edge += 1) # T_ijk
        __set_row!(I, J, V, s2, s1, s3, edge += 1) # T_jik
        __set_row!(I, J, V, s3, s1, s2, edge += 1) # T_kij
    end

    return [sparse(I, J, V); I]
end

function __set_row!(I, J, V, s1, s2, s3, t)
    z = 3*(t-1)+1

    I[z] = t
    J[z] = s1
    V[z] = -1

    I[z+1] = t
    J[z+1] = s2
    V[z+1] = 1

    I[z+2] = t
    J[z+2] = s3
    V[z+2] = 1

    return nothing
end

struct MetricFM{T} <: FusionMatrix{T}
    n::Int # nodes in dissimilarity matrix
    M::Int # rows implied by n
    N::Int # columns implied by n
    I::TriVecIndices # map Cartesian coordinates to trivec index
end

# constructors

function MetricFM{T}(n::Integer) where {T<:Number}
    N = binomial(n, 2)
    M = N * (n-1)

    return MetricFM{T}(n, M, N)
end

function MetricFM{T}(n::Integer, M::Integer, N::Integer) where {T<:Number}
    I = TriVecIndices(n)

    return MetricFM{T}(n, M, N, I)
end

# default eltype to Int64
MetricFM(n::Integer) = MetricFM{Int}(n)
MetricFM(n::Integer, M::Integer, N::Integer) = MetricFM{Int}(n, M, N)

# implementation
Base.size(D::MetricFM) = (D.M, D.N)

"""
```
apply_fusion_matrix!(z, D::MetricFM, x)
```

Multiply `x` by `D = [A, I]`, where `A*x ≥ 0` encodes triangle inequality
constraints and `x ≥ 0` encodes non-negativity.
"""
function apply_fusion_matrix!(z, D::MetricFM, x::SubArray)
    edge = 0 # edge counter
    n = D.n
    X = trivec_parent(x, n)

    # A*x: 0 ≤ x_ik + x_kj - x_ij
    @inbounds for j in 1:n-2, i in j+1:n-1
        a = X[i,j] # fix one edge

        @inbounds for k in i+1:n
            b = X[k,i]
            c = X[k,j]

            # check edges of one triangle
            z[edge += 1] = -a + b + c
            z[edge += 1] = -b + a + c
            z[edge += 1] = -c + a + b
        end
    end

    # I*x = x
    copyto!(z, edge+1, x, 1, length(x))

    return z
end

function apply_fusion_matrix!(z, D::MetricFM, x::AbstractVector)
    edge = 0 # edge counter
    n = D.n
    T = D.I

    # A*x: 0 ≤ x_ik + x_kj - x_ij
    @inbounds for j in 1:n-2, i in j+1:n-1
        a = x[T[i,j]] # fix one edge

        @inbounds for k in i+1:n
            b = x[T[k,i]]
            c = x[T[k,j]]

            # check edges of one triangle
            z[edge += 1] = -a + b + c
            z[edge += 1] = -b + a + c
            z[edge += 1] = -c + a + b
        end
    end

    # I*x = x
    copyto!(z, edge+1, x, 1, length(x))

    return z
end

function apply_fusion_matrix_transpose!(x::SubArray, D::MetricFM, z)
    edge = 0
    N = size(D, 2)
    n = D.n
    X = trivec_parent(x, n)

    # I*z[block2]
    copyto!(x, 1, z, N*(n-2)+1, N)

    # T'*z[block1]
    @inbounds for j in 1:n-2, i in j+1:n-1, k in i+1:n
        abc = z[edge += 1]
        bac = z[edge += 1]
        cab = z[edge += 1]

        X[i,j] += -abc + bac + cab
        X[k,j] += -cab + abc + bac
        X[k,i] += -bac + abc + cab
    end

    return z
end

function apply_fusion_matrix_transpose!(x::AbstractVector, D::MetricFM, z)
    edge = 0
    N = size(D, 2)
    n = D.n
    T = D.I

    # I*z[block2]
    copyto!(x, 1, z, N*(n-2)+1, N)

    # T'*z[block1]
    @inbounds for j in 1:n-2, i in j+1:n-1, k in i+1:n
        abc = z[edge += 1]
        bac = z[edge += 1]
        cab = z[edge += 1]

        x[T[i,j]] += -abc + bac + cab
        x[T[k,j]] += -cab + abc + bac
        x[T[k,i]] += -bac + abc + cab
    end

    return z
end

# for testing: Tt*T * trivec(X)
@inbounds function metric_apply_gram_matrix!(Q, X)
    n = size(X, 1)
    unsafe_copyto!(Q, 1, X, 1, length(X))
    for j in 1:n-2, i in j+1:n-1
        a = X[i,j]  # fix one edge

        @simd for k in i+1:n
            b = X[k,i]
            c = X[k,j]

            # TtT*x
            Q_ij = 3*a - b - c
            Q_ki = 3*b - a - c
            Q_kj = 3*c - a - b

            # accumulate
            Q[i,j] += Q_ij
            Q[k,i] += Q_ki
            Q[k,j] += Q_kj
        end
    end

    return Q
end

# Operator 1: y = Tt*T * trivec(X); y - min(0, y)
@inbounds function metric_apply_operator1!(Q, X)
    n = size(X, 1)

    penalty = zero(eltype(Q))

    for j in 1:n-2, i in j+1:n-1
        a = X[i,j]

        @simd for k in i+1:n
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

            # accumulate penalty T*x - min(T*x, 0)
            penalty += (abc - fabc)^2
            penalty += (bac - fbac)^2
            penalty += (cab - fcab)^2
        end
    end

    return penalty
end
