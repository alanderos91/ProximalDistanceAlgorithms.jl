import ProximalDistanceAlgorithms:
    cvxclst_fusion_matrix,
    cvxclst_apply_fusion_matrix!,
    cvxclst_apply_fusion_matrix_transpose!,
    cvxclst_evaluate_gradient,
    cvxclst_stepsize,
    cvxclst_evaluate_objective,
    distance,
    sparse_fused_block_projection,
    tri2vec

function cvxcluster_initialize(d, n)
    # weights matrix
    W = zeros(n, n)
    for j in 1:n, i in 1:j-1
        w_ij = rand()
        W[i,j] = w_ij
        W[j,i] = w_ij
    end

    # synthetic data
    U = randn(d, n)

    # linear operators as matrices
    Imat = 1.0*I(n)
    e = [Imat[:,i] for i in 1:n]

    # fusion matrix
    D = cvxclst_fusion_matrix(d, n)

    # reshaped weight matrix
    w = zeros(d*binomial(n,2))
    ix = 1
    for j in 1:n, i in j+1:n
        for k in 1:d
            w[ix] = W[i,j]
            ix += 1
        end
    end

    # W*D and transpose(W*D)
    Wdiag = Diagonal(w)
    WD    = Wdiag*D
    WDt   = transpose(WD)

    return W, U, D, WD, WDt
end

@testset "Convex Clustering" begin
    # simulated examples for testing
    features = (2, 10)
    samples  = (50, 100)

    examples = [cvxcluster_initialize(d, n) for d in features, n in samples]

    @testset "linear operators" begin
        for example in examples
            W, U, D, WD, WDt = example

            d, n = size(U)
            println("$(d) features, $(n) samples")

            # auxiliary variables
            V  = zero(U)             # output dimensions same as U
            v  = similar(vec(U))     # vectorized U
            y1 = zeros(size(WD, 1))  # for (W*D) * u
            y2 = zero(y1)            # for (W*D) * u
            yproj = zero(y1)         # for projection of y

            println("  warm-up:")
            @time cvxclst_apply_fusion_matrix!(y1, W, U)
            @time cvxclst_apply_fusion_matrix_transpose!(V, W, y1)
            @time mul!(y2, WD, vec(U))
            @time mul!(v, WDt, y2)
            println()

            # reset
            fill!(y1, 0)
            fill!(V, 0)

            # test: (W*D)*u
            println("  (W*D) * u:")
            print("    operator: ")
            @time cvxclst_apply_fusion_matrix!(y1, W, U) # observed
            print("    mul!:     ")
            @time mul!(y2, WD, vec(U))                   # expected
            @test y1 ≈ y2
            println()

            # test: (WD)t(WD) * vec(U)
            println("  (W*D)t * y:")
            print("    operator: ")
            @time cvxclst_apply_fusion_matrix_transpose!(V, W, y1) # observed
            print("    mul!:     ")
            @time mul!(v, WDt, y2)
            @test vec(V) ≈ v
            println()
        end
    end

    @testset "steepest descent" begin
        for example in examples
            W, U, D, WD, WDt = example

            # parameters
            d, n = size(U)
            k = binomial(n, 2) ÷ 2
            ρ = 2.0

            X = U .- 4.0            # simulated data
            Δ = U-X                 # difference between centroids
            yproj, _, _, _ = sparse_fused_block_projection(W, U, k) # projection
            z = WD*vec(U) - yproj   # direction of projection

            println("$(d) features, $(n) samples")

            # test: gradient evaluation

            # observed
            Q = cvxclst_evaluate_gradient(W, U, X, ρ, k)

            # expected
            q = vec(Δ) + ρ*WDt*z

            @test vec(Q) ≈ q

            # test: step size calculation

            # observed
            γ1, normgrad1 = cvxclst_stepsize(W, Q, ρ)

            # expected
            a = dot(Q, Q)
            v = WD * vec(Q)
            b = dot(v, v)
            γ2 = a / (a + ρ*b + eps())
            normgrad2 = sqrt(a)

            @test γ1 ≈ γ2
            @test normgrad1 ≈ normgrad2

            # test: loss, penalty, objective

            # observed
            loss, penalty, objective = cvxclst_evaluate_objective(U, X, z, ρ)

            # expected
            @test loss ≈ dot(Δ, Δ)
            @test penalty ≈ dot(z, z)
            @test objective ≈ 0.5 * (loss + ρ*penalty)
        end
    end

    @testset "utilities" begin
        d, n = 100, 1000
        A = randn(d, n)
        W = Symmetric(rand(n, n))

        i = 348
        j = 32

        # test: norm(ui - uj)
        @test distance(A, i, j) ≈ norm(A[:,i] - A[:,j])

        # test: wij * norm(ui - uj)
        @test distance(W, A, i, j) ≈ W[i,j] * norm(A[:,i] - A[:,j])

        # test: tri2vec, dictionary ordering
        n = 4
        count = 1
        for j in 1:n, i in j+1:n
            @test tri2vec(i, j, n) == count
            count += 1
        end
    end
end
