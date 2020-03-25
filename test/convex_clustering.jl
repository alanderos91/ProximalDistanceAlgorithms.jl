import ProximalDistanceAlgorithms:
    __find_large_blocks!,
    __accumulate_averaging_step!,
    __apply_sparsity_correction!,
    __accumulate_all_operations!,
    tri2vec

function cvxcluster_initialize(d, n)
    # weights matrix
    # W = zeros(n, n)
    # for j in 1:n, i in 1:j-1
    #     w_ij = rand()
    #     W[i,j] = w_ij
    #     W[j,i] = w_ij
    # end
    W = ones(n, n)

    # synthetic data
    U = randn(d, n)

    # linear operators as matrices
    Imat = 1.0*I(n)
    e = [Imat[:,i] for i in 1:n]

    # fusion matrix
    D_ij = [kron((e[i] - e[j])', I(d)) for j in 1:n for i in j+1:n]
    D = vcat(D_ij...)

    # reshaped weight matrix
    w = zeros(d*binomial(n,2))
    ix = 1
    for j in 1:n, i in j+1:n
        for k in 1:d
            w[ix] = W[i,j]
            ix += 1
        end
    end
    Wdiag = Diagonal(w)

    # W * D and transpose(W * D)
    WD = Wdiag*D
    WDt = transpose(WD)

    return W, U, D, WD, WDt
end

function cvxcluster_auxilliary(U, WD)
    # allocate auxilliary variables
    Q = similar(U)          # output dimensions same as U
    x = similar(vec(U))     # output dimensions same as U, vectorized
    y = zeros(size(WD, 1))  # for (W*D)(U)
    P_y = zero(y)           # for projection

    return Q, x, y, P_y
end

@testset "linear operators" begin
    features = (2, 10)
    samples  = (50, 100)

    examples = [cvxcluster_initialize(d, n) for d in features, n in samples]

    # test: (W*D)'(W*D)*vec(U)
    for example in examples
        W, U, D, WD, WDt = example
        Q, x, y, P_y = cvxcluster_auxilliary(U, WD)

        mul!(y, WD, vec(U)) # expected
        mul!(x, WDt, y)
        @time begin
            fill!(Q, 0); __accumulate_averaging_step!(Q, W, U) # observed
        end
        @test x ≈ vec(Q)
    end

    # test: (W*D)'*[(W*D)*vec(U) - P(W*D*vec(U))]
    @testset "w/ explicit search" begin
        for example in examples
            W, U, D, WD, WDt = example
            Q, x, y, P_y = cvxcluster_auxilliary(U, WD)

            d, n = size(U)
            println("$(d) features, $(n) samples")

            K_max = binomial(n, 2)
            K_lo = round(Int, 0.3 * K_max)
            K_hi = round(Int, 0.7 * K_max)

            for K in (K_lo, K_hi, K_max)
                print("  sparsity = $(K)\t")
                # initialize data structures for sparsity projection
                Iv = zeros(Int, K)
                Jv = zeros(Int, K)
                v  = zeros(K)

                fill!(Iv, 0); fill!(Jv, 0); fill!(v, 0)
                __find_large_blocks!(Iv, Jv, v, W, U)

                # expected
                fill!(P_y, 0); mul!(y, WD, vec(U))
                for (i, j) in zip(Iv, Jv)
                    l = i < j ? tri2vec(i, j, n) : tri2vec(j, i, n)
                    block = (l-1)*d+1:l*d
                    for k in block
                        P_y[k] = y[k]
                    end
                end
                mul!(x, WDt, y - P_y)

                # observed
                @time begin
                    fill!(Q, 0); __accumulate_all_operations!(Q, W, U, Iv, Jv)
                end
                @test x ≈ vec(Q)
            end
        end
    end

    @testset "w/ sparsity correction" begin
        for example in examples
            W, U, D, WD, WDt = example
            Q, x, y, P_y = cvxcluster_auxilliary(U, WD)

            d, n = size(U)
            println("$(d) features, $(n) samples")

            K_max = binomial(n, 2)
            K_lo = round(Int, 0.3 * K_max)
            K_hi = round(Int, 0.7 * K_max)

            for K in (K_lo, K_hi, K_max)
                print("  sparsity = $(K)\t")
                # initialize data structures for sparsity projection
                Iv = zeros(Int, K)
                Jv = zeros(Int, K)
                v  = zeros(K)

                fill!(Iv, 0); fill!(Jv, 0); fill!(v, 0)
                __find_large_blocks!(Iv, Jv, v, W, U)

                # expected
                fill!(P_y, 0); mul!(y, WD, vec(U))
                for (i, j) in zip(Iv, Jv)
                    l = i < j ? tri2vec(i, j, n) : tri2vec(j, i, n)
                    block = (l-1)*d+1:l*d
                    for k in block
                        P_y[k] = y[k]
                    end
                end
                mul!(x, WDt, y - P_y)

                # observed
                @time begin
                    fill!(Q, 0)
                    __accumulate_averaging_step!(Q, W, U)
                    __apply_sparsity_correction!(Q, W, U, Iv, Jv)
                end
                @test x ≈ vec(Q)
            end
        end
    end
end
