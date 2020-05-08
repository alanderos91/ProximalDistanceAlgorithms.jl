import ProximalDistanceAlgorithms:
    cvxclst_fusion_matrix,
    cvxclst_apply_fusion_matrix!,
    cvxclst_apply_fusion_matrix_transpose!,
    cvxclst_stepsize,
    cvxclst_evaluate_objective,
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
    W = Diagonal(w)
    Dt = transpose(D)

    return W, U, D, Dt
end

@testset "Convex Clustering" begin
    # simulated examples for testing
    features = (2, 10)
    samples  = (50, 100, 512)

    examples = [cvxcluster_initialize(d, n) for d in features, n in samples]

    @testset "linear operators" begin
        for example in examples
            W, U, D, Dt = example

            d, n = size(U)
            m = binomial(n, 2)
            println("$(d) features, $(n) samples")

            # auxiliary variables
            V  = zero(U)             # output dimensions same as U
            v  = similar(vec(U))     # vectorized U
            Y1 = zeros(d, m)         # for (W*D) * u
            Y2 = zero(Y1)            # for (W*D) * u
            yproj = zero(Y1)         # for projection of y

            println("  warm-up:")
            @time cvxclst_apply_fusion_matrix!(Y1, U)
            @time cvxclst_apply_fusion_matrix_transpose!(V, Y1)
            @time mul!(vec(Y2), D, vec(U))
            @time mul!(v, Dt, vec(Y2))
            println()

            # reset
            fill!(Y1, 0)
            fill!(V, 0)

            # test: (W*D)*u
            println("  D*u:")
            print("    operator: ")
            @time cvxclst_apply_fusion_matrix!(Y1, U) # observed
            print("    mul!:     ")
            @time mul!(vec(Y2), D, vec(U))            # expected
            @test Y1 ≈ Y2
            println()

            # test: (WD)t(WD) * vec(U)
            println("  Dt*y:")
            print("    operator: ")
            @time cvxclst_apply_fusion_matrix_transpose!(V, Y1) # observed
            print("    mul!:     ")
            @time mul!(v, Dt, vec(Y2))
            @test vec(V) ≈ v
            println()
        end
    end

    # @testset "steepest descent" begin
    #     for example in examples
    #         W, U, D, WD, WDt = example
    #
    #         # parameters
    #         d, n = size(U)
    #         k = binomial(n, 2) ÷ 2
    #         ρ = 2.0
    #
    #         X = U .- 4.0            # simulated data
    #         Δ = U-X                 # difference between centroids
    #         yproj, _, _, _ = sparse_fused_block_projection(W, U, k) # projection
    #         z = WD*vec(U) - yproj   # direction of projection
    #
    #         println("$(d) features, $(n) samples")
    #
    #         # test: gradient evaluation
    #
    #         # observed
    #         Q = cvxclst_evaluate_gradient(W, U, X, ρ, k)
    #
    #         # expected
    #         q = vec(Δ) + ρ*WDt*z
    #
    #         @test vec(Q) ≈ q
    #
    #         # test: step size calculation
    #
    #         # observed
    #         γ1, normgrad1 = cvxclst_stepsize(W, Q, ρ)
    #
    #         # expected
    #         a = dot(Q, Q)
    #         v = WD * vec(Q)
    #         b = dot(v, v)
    #         γ2 = a / (a + ρ*b + eps())
    #         normgrad2 = sqrt(a)
    #
    #         @test γ1 ≈ γ2
    #         @test normgrad1 ≈ normgrad2
    #
    #         # test: loss, penalty, objective
    #
    #         # observed
    #         loss, penalty, objective = cvxclst_evaluate_objective(U, X, z, ρ)
    #
    #         # expected
    #         @test loss ≈ dot(Δ, Δ)
    #         @test penalty ≈ dot(z, z)
    #         @test objective ≈ 0.5 * (loss + ρ*penalty)
    #     end
    # end

    @testset "utilities" begin
        # test: tri2vec, dictionary ordering
        n = 4
        count = 1
        for j in 1:n, i in j+1:n
            @test tri2vec(i, j, n) == count
            count += 1
        end
    end
end
