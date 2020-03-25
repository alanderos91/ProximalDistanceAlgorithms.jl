function __distance(X, i, j)
    d = size(X, 1)
    δ_ij = zero(eltype(X))

    for k in 1:d
        δ_ij += (X[k,i] - X[k,j])^2
    end

    return sqrt(δ_ij)
end

function __distance(W, X, i, j)
    return W[i,j] * __distance(X, i, j)
end

"""
Finds the connected components of a graph.
Nodes should be numbered 1,2,...
"""
function connect(neighbor::Array{Array{Int, 1}, 1})
#
    nodes = length(neighbor)
    component = zeros(Int, nodes)
    components = 0
    for i = 1:nodes
        if component[i] > 0 continue end
        components = components + 1
        component[i] = components
        visit!(neighbor, component, i)
    end
    return (component, components)
end

"""
Recursively assigns components by depth first search.
"""
function visit!(neighbor::Array{Array{Int,1},1},
  component::Vector{Int}, i::Int)
#
    for j in neighbor[i]
        if component[j] > 0 continue end
        component[j] = component[i]
        visit!(neighbor, component, j)
    end
end

"""
Collects neighborhoods and weights from an adjacency matrix A.
"""
function adjacency_to_neighborhood(A::Matrix)
#
    (nodes, T) = (size(A, 1), eltype(A))
    neighbor = [Vector{Int}() for i = 1:nodes]
    weight = [Vector{T}() for i = 1:nodes]
    for i = 1:nodes
        for j = 1:nodes
            if A[i, j] != zero(T)
                push!(neighbor[i], j)
                push!(weight[i], A[i, j])
            end
        end
    end
    return (neighbor, weight)
end

function gaussian_weights(X; phi = 1.0)
    d, n = size(X)

    T = eltype(X)
    W = zeros(n, n)

    for j in 1:n, i in j+1:n
        δ_ij = __distance(X, i, j)
        w_ij = exp(-phi*δ_ij^2)

        W[i,j] = w_ij
        W[j,i] = w_ij
    end

    return W
end

function tri2vec(i, j, n)
  return (j-i) + n*(i-1) - (i*(i-1))>>1
end

function knn_weights(W, k)
    n = size(W, 1)
    w = [W[i,j] for j in 1:n for i in j+1:n] |> vec
    i = 1
    neighbors = tri2vec.(i, (i+1):n, n)
    keep = neighbors[sortperm(w[neighbors])[1:k]]

    for i in 2:(n-1)
        group_A = tri2vec.(i, (i+1):n, n)
        group_B = tri2vec.(1:(i-1), n, n)
        neighbors = [group_A; group_B]
        knn = neighbors[sortperm(w[neighbors])[1:k]]
        keep = union(knn, keep)
    end

    i = n
    neighbors = tri2vec.(1:(i-1), i, n)
    knn = neighbors[sortperm(w[neighbors])[1:k]]
    keep = union(knn, keep)

    W_knn = zero(W)

    for j in 1:n, i in j+1:n
        l = tri2vec(i, j, n)
        if l in keep
            W_knn[i,j] = W[i,j]
            W_knn[j,i] = W[i,j]
        end
    end

    return W_knn
end
