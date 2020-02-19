function metric_example(n; weighted = false)
    n < 3 && error("number of nodes must be ≥ 3")

    D = zeros(n, n)

    for j in 1:n, i in j+1:n
        u = 10*rand()

        D[i,j] = u
        D[j,i] = u
    end

    W = zeros(n, n)

    for j in 1:n, i in j+1:n
        u = weighted ? rand() : 1.0

        W[i,j] = u
        W[j,i] = u
    end

    return W, D
end
